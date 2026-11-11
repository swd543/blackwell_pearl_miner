//! Continuous GPU producer (this thread), CPU proof worker, independent socket owner.
use crate::cuda::{memory_info, Device};
use crate::engine::{Engine, Snapshot};
use crate::nvml::{Guard, Nvml};
use crate::priority::{Options as PriorityOptions, PriorityWatch};
use crate::proof::ProofCache;
use crate::protocol::{hex, Kernel, MiningJob, Profile, M, N};
use crate::state::{merge, Generation, Log, State};
use crate::workers::{network_worker, proof_worker, put, NetworkConfig, ProofContext, ProofTask, VerifiedShare};
use anyhow::{Context, Result};
use crossbeam_channel::bounded;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct MineArgs {
    pub pool: String,
    pub wallet: String,
    pub worker: String,
    pub seconds: f64,
    pub profile: Profile,
    /// Default profile chosen by the miner: fall back to 448/K4096 without gzip proofs.
    pub default_profile: bool,
    pub hit_capacity: usize,
    pub stop_after: u64,
    pub json: bool,
    pub depth: usize,
    pub queue_size: usize,
    pub max_inflight: usize,
    pub priority: Vec<PathBuf>,
    pub background: bool,
    pub priority_resume_delay: f64,
    pub priority_util: u32,
    pub priority_slowdown: f64,
    pub power_limit: Option<f64>,
    pub max_temp: u32,
    pub device: i32,
    pub log_path: PathBuf,
    pub web_listen: Option<String>,
    pub web_allow_net: Vec<String>,
    pub restore_power_limit_script: Option<PathBuf>,
    pub history_window_hours: u32,
    pub nonce_start: Option<u64>,
    pub reconnect_seconds: u64,
}

static SIGNALS: AtomicU32 = AtomicU32::new(0);

extern "C" fn on_signal(_: libc::c_int) {
    // First signal: finish in-flight rounds and drain; second: exit at once.
    if SIGNALS.fetch_add(1, Ordering::SeqCst) >= 1 {
        unsafe { libc::_exit(130) };
    }
}

fn interrupted() -> bool {
    SIGNALS.load(Ordering::SeqCst) > 0
}

fn cpu_seconds() -> f64 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
    t(u.ru_utime) + t(u.ru_stime)
}

fn random_nonce() -> u64 {
    let mut b = [0u8; 8];
    getrandom::getrandom(&mut b).expect("entropy");
    // Leave headroom so the 64-bit counter never wraps within a job.
    u64::from_le_bytes(b) >> 1
}

struct Stats {
    start: Instant,
    cpu_start: f64,
    rounds: u64,
    macs: f64,
    paused: f64,
    pauses: u64,
    max_proof_queue: usize,
}

pub fn run(a: MineArgs) -> Result<i32> {
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
    let restore_power = crate::power::RestorePowerLimit::new(a.restore_power_limit_script.clone());
    let state = State::with_history_window_hours(a.history_window_hours);
    let started = Instant::now();
    let web = match &a.web_listen {
        Some(bind) => Some(crate::web::WebServer::start(bind, state.clone(), started, &a.web_allow_net, a.log_path.clone())?),
        None => None,
    };
    let log_thread = Log::start(a.log_path.clone(), a.json, state.clone())?;
    let log = log_thread.log.clone();
    if let Some(server) = &web {
        log.event("web_stats", json!({"bind":server.local_addr().to_string()}));
    }
    let result = mine(&a, &state, &log, &restore_power);
    if let Err(e) = &result {
        log.event("error", json!({"error": format!("{e:#}")}));
    }
    state.abort.store(true, Ordering::SeqCst);
    log_thread.close();
    drop(web);
    drop(restore_power);
    result
}

fn mine(a: &MineArgs, state: &Arc<State>, log: &Log, power_policy: &crate::power::RestorePowerLimit) -> Result<i32> {
    let mut profile = a.profile;
    log.event("startup", json!({"endpoint":a.pool,"wallet":a.wallet,"worker":a.worker,"variant":profile.variant(),
        "k":profile.k,"seconds":a.seconds,"log_path":a.log_path.display().to_string()}));
    let device = Device::open(a.device)?;
    let info = device.info()?;
    let yielded_power_limit = power_policy.manages_yield(a.power_limit).then_some(f64::from(crate::power::YIELDED_LIMIT_W));
    let guard = Guard::start(&info.pci_bus_id, a.power_limit, yielded_power_limit, a.max_temp, state.clone()).context("NVML telemetry")?;
    anyhow::ensure!(info.capability == (12, 0), "the kernels are built for compute capability 12.0 (found {} with {}.{})", info.name, info.capability.0, info.capability.1);

    let (proof_tx, proof_rx) = bounded::<Option<ProofTask>>(a.queue_size);
    let (submit_tx, submit_rx) = bounded::<Option<VerifiedShare>>(a.queue_size);
    let net = {
        let (state, log) = (state.clone(), log.clone());
        let cfg = NetworkConfig {
            endpoint: a.pool.clone(),
            wallet: a.wallet.clone(),
            worker: a.worker.clone(),
            stop_after: a.stop_after,
            max_inflight: a.max_inflight,
            ack_timeout: Duration::from_secs(15),
            retry: Duration::from_secs(a.reconnect_seconds),
        };
        std::thread::Builder::new().name("pearl-network".into()).spawn(move || network_worker(state, submit_rx, cfg, log))?
    };
    let prover = {
        let (state, log) = (state.clone(), log.clone());
        std::thread::Builder::new().name("pearl-proofs".into()).spawn(move || proof_worker(state, proof_rx, submit_tx, log))?
    };
    let priority = if a.priority.is_empty() && !a.background {
        None
    } else {
        let opts = PriorityOptions { resume_delay: a.priority_resume_delay, threshold: a.priority_util, background: a.background, ..Default::default() };
        Some(PriorityWatch::start(&a.priority, opts, Box::new(Nvml::open(&info.pci_bus_id)?))?)
    };
    let check = || -> Result<()> {
        state.check()?;
        guard.check()?;
        if let Some(p) = &priority {
            p.check()?;
        }
        Ok(())
    };

    while !state.ready.load(Ordering::SeqCst) {
        if interrupted() {
            break;
        }
        check()?;
        std::thread::sleep(Duration::from_millis(50));
    }
    let mut st = Stats { start: Instant::now(), cpu_start: cpu_seconds(), rounds: 0, macs: 0.0, paused: 0.0, pauses: 0, max_proof_queue: 0 };
    if !interrupted() {
        check()?;
        let compression = state.compression.lock().unwrap().clone().unwrap_or_default();
        if a.default_profile && compression != "gzip" {
            // Uncompressed long-K proofs exceed the 256 KiB frame: use the reference profile.
            profile = Profile::new(Kernel::K448, 4096)?;
            log.event("profile_fallback", json!({"reason":"pool did not negotiate gzip proofs","proof_compression":compression,"variant":448,"k":4096}));
        }
        let mut engine = Engine::new(profile, a.depth, a.hit_capacity)?;
        st.start = Instant::now();
        state.share_rate.lock().unwrap().reset(st.start);
        st.cpu_start = cpu_seconds();
        mining_loop(a, state, log, power_policy, &mut engine, priority.as_ref(), &check, &proof_tx, &mut st)?;
    }
    // Every completed current-job result is handed off before the sentinel.
    put(&proof_tx, None, state)?;
    while !prover.is_finished() || !net.is_finished() {
        check()?;
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = prover.join();
    let net_stats = net.join().map_err(|_| anyhow::anyhow!("network thread panicked"))?;
    check()?;
    let mut row = stats_row(state, &st, priority.is_some());
    row["interrupted"] = json!(interrupted());
    row["max_inflight_seen"] = json!(net_stats.max_pending_seen);
    log.event("complete", row);
    Ok(if state.counters().accepted > 0 || interrupted() { 0 } else { 2 })
}

fn stats_row(state: &State, st: &Stats, with_priority: bool) -> Value {
    let elapsed = st.start.elapsed().as_secs_f64().max(1e-9);
    let active = (elapsed - st.paused).max(1e-9);
    let c = state.counters();
    let rate = state.share_rate.lock().unwrap().snapshot();
    let mut row = merge(&[
        json!({"accepted":c.accepted,"rejected":c.rejected,"stale":c.stale}),
        rate,
        json!({"seconds":elapsed,"rounds":st.rounds,"TMAC_wall_s":st.macs/(active*1e12),"async_loop":true,
               "cpu_percent_one_core":(cpu_seconds()-st.cpu_start)/elapsed*100.0,"proof_queue_max":st.max_proof_queue}),
    ]);
    if with_priority {
        row["paused_seconds"] = json!(st.paused);
        row["pauses"] = json!(st.pauses);
    }
    row
}

/// Contention probe: interval between round completions vs the recent median.
struct RoundClock {
    last: Option<Instant>,
    skip: u32,
    reference: VecDeque<f64>,
    slowdown: f64,
}

impl RoundClock {
    fn reset(&mut self) {
        self.last = None;
        self.skip = 2;
        self.reference.clear();
    }
    fn done(&mut self, priority: Option<&PriorityWatch>) {
        let Some(p) = priority else { return };
        let now = Instant::now();
        let Some(last) = self.last.replace(now) else { return };
        if self.skip > 0 {
            self.skip -= 1;
            return;
        }
        let dt = now.duration_since(last).as_secs_f64();
        if self.reference.len() >= 5 {
            let mut v: Vec<f64> = self.reference.iter().copied().collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if v.len() % 2 == 1 { v[v.len() / 2] } else { (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0 };
            let ratio = dt / median;
            if ratio > self.slowdown {
                p.confirm(ratio);
                return;
            }
        }
        if self.reference.len() == 32 {
            self.reference.pop_front();
        }
        self.reference.push_back(dt);
    }
}

#[allow(clippy::too_many_arguments)]
fn mining_loop(
    a: &MineArgs,
    state: &Arc<State>,
    log: &Log,
    power_policy: &crate::power::RestorePowerLimit,
    engine: &mut Engine,
    priority: Option<&PriorityWatch>,
    check: &dyn Fn() -> Result<()>,
    proof_tx: &crossbeam_channel::Sender<Option<ProofTask>>,
    st: &mut Stats,
) -> Result<()> {
    let profile = engine.profile;
    let depth = engine.depth();
    let time_up = |st: &Stats| a.seconds > 0.0 && st.start.elapsed().as_secs_f64() >= a.seconds;
    let stop = || state.stop_mining.load(Ordering::SeqCst) || interrupted();
    let yielding = || priority.map(|p| p.is_busy()).unwrap_or(false);
    let manage_yield_power = power_policy.manages_yield(a.power_limit);
    let mut clock = RoundClock { last: None, skip: 2, reference: VecDeque::new(), slowdown: 1.0 + a.priority_slowdown / 100.0 };
    let mut last_stats = Instant::now();

    // Pause with no kernel in flight. Miner memory stays allocated for an instant resume;
    // proof work, pool connection and job updates keep running.
    let wait_for_gpu = |st: &mut Stats| -> Result<()> {
        let p = priority.unwrap();
        let began = Instant::now();
        st.pauses += 1;
        let holder = p.holder();
        let drain = holder["detected_at"].as_f64().map(|d| crate::state::now_unix() - d).unwrap_or(0.0);
        log.event("priority_pause", merge(&[holder, json!({"drain_s":drain})]));
        state.gpu_paused.store(true, Ordering::SeqCst);
        if manage_yield_power {
            if let Err(error) = power_policy.set_limit(crate::power::YIELDED_LIMIT_W) {
                log.event("error", json!({"error":format!("could not restore GPU power limit while yielding: {error:#}")}));
            }
        }
        while p.is_busy() && !stop() && !time_up(st) {
            check()?;
            std::thread::sleep(Duration::from_millis(100));
        }
        let paused = began.elapsed().as_secs_f64();
        st.paused += paused;
        if !stop() && !time_up(st) {
            if manage_yield_power {
                power_policy.set_limit(crate::power::MINING_LIMIT_W).context("could not reapply mining power limit before resuming")?;
            }
            state.gpu_paused.store(false, Ordering::SeqCst);
            log.event("priority_resume", json!({"paused_seconds":paused,"pauses":st.pauses}));
        }
        Ok(())
    };

    let deliver = |generation: &Generation, ctx: &Arc<ProofContext>, snap: Snapshot, st: &mut Stats| -> Result<()> {
        if snap.overflow > 0 {
            log.event("hit_overflow", json!({"dropped":snap.overflow,"capacity":a.hit_capacity}));
        }
        if !state.submittable(generation) {
            if !snap.groups.is_empty() {
                state.add_stale(snap.groups.len() as u64);
                log.event("round_discarded", json!({"reason":"job_changed","job_id":generation.job.job_id,"sequence":snap.sequence,"shares":snap.groups.len()}));
            }
            return Ok(());
        }
        for &group in &snap.groups {
            put(proof_tx, Some(ProofTask { generation: generation.clone(), ctx: ctx.clone(), group, signal: snap.signal, root_a: snap.root_a }), state)?;
        }
        st.max_proof_queue = st.max_proof_queue.max(proof_tx.len());
        Ok(())
    };

    while !time_up(st) && !stop() {
        check()?;
        if yielding() {
            wait_for_gpu(st)?;
            continue;
        }
        let generation = state.job().context("no pool job")?;
        let setup = Instant::now();
        let job = MiningJob::new((*generation.job).clone(), profile)?;
        let nonce = a.nonce_start.unwrap_or_else(random_nonce);
        let (heap, root_b) = engine.set_job(&job, nonce)?;
        let key = job.key;
        let ctx = Arc::new(ProofContext { cache: ProofCache::new(profile, heap, key, root_b)?, job });
        let (free, total) = memory_info()?;
        log.event("mining", json!({"job_id":generation.job.job_id,"m":M,"n":N,"k":profile.k,"rank":128,"variant":profile.variant(),
            "sampling":format!("{}x16",profile.sampling_rows()),"counted_macs_per_round":profile.macs_per_round(),
            "async_depth":depth,"job_setup_ms":setup.elapsed().as_secs_f64()*1000.0,"gpu_memory_mib":(total-free)>>20,
            "raw_target":format!("{:064x}",generation.job.raw_target),"verification_nbits":ctx.job.bits,"nonce_start":hex(&nonce.to_be_bytes())}));
        clock.reset();
        let current = |g: &Generation| state.job().map(|c| c.number == g.number).unwrap_or(false);
        let mut pending = VecDeque::new();
        for slot in 0..depth {
            if !current(&generation) || stop() || yielding() || time_up(st) {
                break;
            }
            engine.launch(slot)?;
            pending.push_back(slot);
        }
        while let Some(slot) = pending.pop_front() {
            check()?;
            let snap = engine.collect(slot)?;
            clock.done(priority);
            st.rounds += 1;
            st.macs += profile.macs_per_round();
            if current(&generation) && !stop() && !yielding() && !time_up(st) {
                engine.launch(slot)?;
                pending.push_back(slot);
            }
            // Feed the GPU before queuing CPU work.
            deliver(&generation, &ctx, snap, st)?;
            if pending.is_empty() && yielding() {
                wait_for_gpu(st)?;
                clock.reset();
                for slot in 0..depth {
                    if !current(&generation) || stop() || yielding() || time_up(st) {
                        break;
                    }
                    engine.launch(slot)?;
                    pending.push_back(slot);
                }
            }
            if last_stats.elapsed() > Duration::from_secs(30) {
                let row = stats_row(state, st, priority.is_some());
                state.record_hashrate(row["TMAC_wall_s"].as_f64().unwrap_or(0.0), row["submitted_TH_s_30m"].as_f64().unwrap_or(0.0));
                log.event("stats", row);
                last_stats = Instant::now();
            }
        }
    }
    Ok(())
}
