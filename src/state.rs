//! Shared worker state, difficulty-weighted share rate and the JSONL/console log.
use crate::protocol::PoolJob;
use anyhow::{bail, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Generation {
    pub number: u64,
    pub job: Arc<PoolJob>,
}

/// Rolling (max 30 min) rate of transmitted shares: 2^256/(target+1) work per share.
pub struct ShareRate {
    start: Instant,
    entries: VecDeque<(Instant, f64)>,
}
const WINDOW: f64 = 1800.0;
impl ShareRate {
    pub fn new() -> Self {
        ShareRate { start: Instant::now(), entries: VecDeque::new() }
    }
    pub fn reset(&mut self, start: Instant) {
        self.start = start;
        self.entries.clear();
    }
    fn expire(&mut self, now: Instant) {
        while let Some((t, _)) = self.entries.front() {
            if now.duration_since(*t).as_secs_f64() >= WINDOW {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }
    pub fn record(&mut self, work: f64, now: Instant) {
        self.entries.push_back((now, work));
        self.expire(now);
    }
    pub fn snapshot(&mut self) -> Value {
        let now = Instant::now();
        self.expire(now);
        let elapsed = now.duration_since(self.start).as_secs_f64().min(WINDOW);
        let work: f64 = self.entries.iter().map(|(_, w)| w).sum();
        json!({"submitted_TH_s_30m": if elapsed > 0.0 { work / elapsed / 1e12 } else { 0.0 },
               "submitted_window_s": elapsed, "submitted_shares_30m": self.entries.len()})
    }
}

#[derive(Default, Clone, Copy)]
pub struct Counters {
    pub accepted: u64,
    pub rejected: u64,
    pub stale: u64,
}

pub struct State {
    latest: Mutex<Option<Generation>>,
    pub ready: AtomicBool,
    pub gpu_paused: AtomicBool,
    pub stop_mining: AtomicBool,
    pub abort: AtomicBool,
    error: Mutex<Option<String>>,
    pub counters: Mutex<Counters>,
    pub compression: Mutex<Option<String>>,
    pub share_rate: Mutex<ShareRate>,
    submit_previous: bool,
}

impl State {
    pub fn new() -> Arc<Self> {
        Arc::new(State {
            latest: Mutex::new(None),
            ready: AtomicBool::new(false),
            gpu_paused: AtomicBool::new(false),
            stop_mining: AtomicBool::new(false),
            abort: AtomicBool::new(false),
            error: Mutex::new(None),
            counters: Mutex::new(Counters::default()),
            compression: Mutex::new(None),
            share_rate: Mutex::new(ShareRate::new()),
            submit_previous: std::env::var("PEARL_SUBMIT_PREVIOUS_JOB").map(|v| v != "0").unwrap_or(true),
        })
    }
    pub fn update_job(&self, job: &PoolJob) -> Generation {
        let mut g = self.latest.lock().unwrap();
        match &*g {
            Some(cur) if *cur.job == *job => cur.clone(),
            cur => {
                let number = cur.as_ref().map(|c| c.number + 1).unwrap_or(0);
                let next = Generation { number, job: Arc::new(job.clone()) };
                *g = Some(next.clone());
                next
            }
        }
    }
    pub fn job(&self) -> Option<Generation> {
        self.latest.lock().unwrap().clone()
    }
    /// A superseded job stays submittable while the pool only refreshed the merkle part
    /// (pools may refresh the job at the same height and still accept such shares).
    pub fn submittable(&self, g: &Generation) -> bool {
        let cur = self.latest.lock().unwrap();
        match &*cur {
            Some(c) if c.number == g.number => true,
            Some(c) => self.submit_previous && g.job.same_work(&c.job),
            None => false,
        }
    }
    pub fn counters(&self) -> Counters {
        *self.counters.lock().unwrap()
    }
    pub fn add_stale(&self, n: u64) {
        self.counters.lock().unwrap().stale += n;
    }
    pub fn fail(&self, msg: String) {
        let mut e = self.error.lock().unwrap();
        if e.is_none() {
            *e = Some(msg);
        }
        self.abort.store(true, Ordering::SeqCst);
        self.stop_mining.store(true, Ordering::SeqCst);
        self.ready.store(true, Ordering::SeqCst);
    }
    pub fn check(&self) -> Result<()> {
        if let Some(e) = &*self.error.lock().unwrap() {
            bail!("worker failed: {e}");
        }
        if self.abort.load(Ordering::SeqCst) {
            bail!("workers aborted");
        }
        Ok(())
    }
}

pub fn now_unix() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

type LogItem = Option<(Value, Option<Value>)>;

/// JSONL file + console output on its own thread; accepted proofs saved as an artifact.
#[derive(Clone)]
pub struct Log {
    tx: Sender<LogItem>,
}

pub struct LogThread {
    pub log: Log,
    handle: Option<JoinHandle<()>>,
}

impl Log {
    pub fn start(path: PathBuf, json_output: bool, state: Arc<State>) -> Result<LogThread> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut file = std::fs::File::create(&path)?;
        let (tx, rx): (Sender<LogItem>, Receiver<LogItem>) = bounded(1024);
        let artifact_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let handle = std::thread::Builder::new().name("pearl-log".into()).spawn(move || {
            let mut run = || -> Result<()> {
                while let Ok(Some((row, artifact))) = rx.recv() {
                    writeln!(file, "{row}")?;
                    file.flush()?;
                    let line = if json_output { row.to_string() } else { crate::console::format_event(&row) };
                    println!("{line}");
                    if let Some(a) = artifact {
                        let tmp = artifact_dir.join("accepted-pool-proof.json.tmp");
                        std::fs::write(&tmp, serde_json::to_string_pretty(&a)? + "\n")?;
                        std::fs::rename(&tmp, artifact_dir.join("accepted-pool-proof.json"))?;
                    }
                }
                Ok(())
            };
            if let Err(e) = run() {
                state.fail(format!("log worker: {e}"));
            }
        })?;
        Ok(LogThread { log: Log { tx }, handle: Some(handle) })
    }
    pub fn event(&self, event: &str, fields: Value) {
        self.emit(event, fields, None)
    }
    pub fn emit(&self, event: &str, fields: Value, artifact: Option<Value>) {
        let mut row = Map::new();
        row.insert("time".into(), json!(now_unix()));
        row.insert("event".into(), json!(event));
        if let Value::Object(m) = fields {
            row.extend(m);
        }
        let _ = self.tx.send(Some((Value::Object(row), artifact)));
    }
}

impl LogThread {
    pub fn close(mut self) {
        let _ = self.log.tx.send(None);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Merge JSON objects (later keys win).
pub fn merge(parts: &[Value]) -> Value {
    let mut m = Map::new();
    for p in parts {
        if let Value::Object(o) = p {
            for (k, v) in o {
                m.insert(k.clone(), v.clone());
            }
        }
    }
    Value::Object(m)
}
