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

pub const DEFAULT_HISTORY_WINDOW_HOURS: u32 = 36;
pub const MAX_HISTORY_WINDOW_HOURS: u32 = 168;

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

#[derive(Default, Clone, Copy)]
pub struct GpuTelemetry {
    pub temperature_c: Option<u32>,
    pub power_limit_w: Option<f64>,
}

#[derive(Clone, Copy)]
pub struct TemperaturePoint {
    pub sequence: u64,
    pub timestamp_ms: u64,
    pub temperature_c: u32,
}

#[derive(Clone, Copy)]
pub struct HashratePoint {
    pub sequence: u64,
    pub timestamp_ms: u64,
    pub compute_tmac_s: f32,
    pub submitted_th_s: f32,
}

pub struct TrendSnapshot {
    pub reset: bool,
    pub cursor: u64,
    pub window_seconds: u64,
    pub temperature: Vec<TemperaturePoint>,
    pub hashrate: Vec<HashratePoint>,
}

struct TrendHistory {
    window_seconds: u64,
    next_sequence: u64,
    temperature: VecDeque<TemperaturePoint>,
    hashrate: VecDeque<HashratePoint>,
}

impl TrendHistory {
    fn new(window_seconds: u64) -> Self {
        Self { window_seconds: window_seconds.max(1), next_sequence: 0, temperature: VecDeque::new(), hashrate: VecDeque::new() }
    }

    fn sequence(&mut self) -> u64 {
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.next_sequence
    }

    fn record_temperature(&mut self, timestamp_ms: u64, temperature_c: u32) {
        let sequence = self.sequence();
        self.temperature.push_back(TemperaturePoint { sequence, timestamp_ms, temperature_c });
        self.prune(timestamp_ms);
    }

    fn record_hashrate(&mut self, timestamp_ms: u64, compute_tmac_s: f64, submitted_th_s: f64) {
        if !compute_tmac_s.is_finite() || !submitted_th_s.is_finite() || compute_tmac_s < 0.0 || submitted_th_s < 0.0 {
            return;
        }
        let sequence = self.sequence();
        self.hashrate.push_back(HashratePoint { sequence, timestamp_ms, compute_tmac_s: compute_tmac_s as f32, submitted_th_s: submitted_th_s as f32 });
        self.prune(timestamp_ms);
    }

    fn prune(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(self.window_seconds.saturating_mul(1000));
        while self.temperature.front().is_some_and(|p| p.timestamp_ms < cutoff) { self.temperature.pop_front(); }
        while self.hashrate.front().is_some_and(|p| p.timestamp_ms < cutoff) { self.hashrate.pop_front(); }
    }

    fn snapshot(&mut self, after: Option<u64>, now_ms: u64) -> TrendSnapshot {
        self.prune(now_ms);
        let oldest = self.temperature.front().map(|p| p.sequence).into_iter()
            .chain(self.hashrate.front().map(|p| p.sequence)).min();
        let reset = match after {
            None => true,
            Some(cursor) if cursor > self.next_sequence => true,
            Some(cursor) => oldest.map(|first| cursor.saturating_add(1) < first).unwrap_or(cursor < self.next_sequence),
        };
        let cursor = if reset { 0 } else { after.unwrap_or(0) };
        TrendSnapshot {
            reset,
            cursor: self.next_sequence,
            window_seconds: self.window_seconds,
            temperature: self.temperature.iter().copied().filter(|p| p.sequence > cursor).collect(),
            hashrate: self.hashrate.iter().copied().filter(|p| p.sequence > cursor).collect(),
        }
    }
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
    pub gpu_telemetry: Mutex<GpuTelemetry>,
    trend_history: Mutex<TrendHistory>,
    submit_previous: bool,
}

impl State {
    pub fn with_history_window_hours(hours: u32) -> Arc<Self> {
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
            gpu_telemetry: Mutex::new(GpuTelemetry::default()),
            trend_history: Mutex::new(TrendHistory::new(u64::from(hours.max(1)) * 3600)),
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
    pub fn update_gpu_telemetry(&self, temperature_c: u32, power_limit_w: f64) {
        *self.gpu_telemetry.lock().unwrap() = GpuTelemetry { temperature_c: Some(temperature_c), power_limit_w: Some(power_limit_w) };
        self.trend_history.lock().unwrap().record_temperature(now_unix_ms(), temperature_c);
    }
    pub fn record_hashrate(&self, compute_tmac_s: f64, submitted_th_s: f64) {
        self.trend_history.lock().unwrap().record_hashrate(now_unix_ms(), compute_tmac_s, submitted_th_s);
    }
    pub fn trend_snapshot(&self, after: Option<u64>) -> TrendSnapshot {
        self.trend_history.lock().unwrap().snapshot(after, now_unix_ms())
    }
    pub fn gpu_telemetry(&self) -> GpuTelemetry {
        *self.gpu_telemetry.lock().unwrap()
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

fn now_unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(u64::MAX as u128) as u64
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

#[cfg(test)]
mod trend_tests {
    use super::*;

    #[test]
    fn history_defaults_to_36_hours_and_streams_only_new_samples() {
        assert_eq!(State::with_history_window_hours(DEFAULT_HISTORY_WINDOW_HOURS).trend_snapshot(None).window_seconds, 36 * 3600);
        let mut history = TrendHistory::new(60);
        history.record_temperature(1_000_000, 82);
        history.record_hashrate(1_001_000, 300.0, 275.0);
        let initial = history.snapshot(None, 1_002_000);
        assert!(initial.reset);
        assert_eq!(initial.temperature.len(), 1);
        assert_eq!(initial.hashrate.len(), 1);
        let delta = history.snapshot(Some(initial.cursor), 1_003_000);
        assert!(!delta.reset);
        assert!(delta.temperature.is_empty());
        history.record_temperature(1_100_000, 84);
        let stale = history.snapshot(Some(0), 1_100_000);
        assert!(stale.reset);
        assert_eq!(stale.temperature.len(), 1);
        assert_eq!(stale.temperature[0].temperature_c, 84);
        assert!(stale.hashrate.is_empty());
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
