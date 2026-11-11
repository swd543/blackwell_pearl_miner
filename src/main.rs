//! PearlHash pool miner for NVIDIA Blackwell GPUs (compute capability 12.0).
mod console;
mod cuda;
mod engine;
mod kernels;
mod miner;
mod nvml;
mod pool;
mod power;
mod priority;
mod proof;
mod protocol;
mod state;
mod web;
mod workers;

use anyhow::{bail, ensure, Result};
use clap::Parser;
use protocol::{Kernel, MiningJob, PoolJob, Profile};
use std::path::PathBuf;
use zk_pow::api::proof::IncompleteBlockHeader;

#[derive(Parser)]
#[command(name = "bpm", version, about = "PearlHash pool miner for NVIDIA Blackwell GPUs (compute capability 12.0)")]
struct Cli {
    /// Pearl Stratum endpoint HOST:PORT
    #[arg(long)]
    pool: Option<String>,
    /// Payout address or pool account
    #[arg(long)]
    wallet: Option<String>,
    #[arg(long, default_value = "worker")]
    worker: String,
    /// Run time limit in seconds (0 = unlimited)
    #[arg(long, default_value_t = 0.0)]
    seconds: f64,
    /// Kernel: 780 (default) or 448
    #[arg(long)]
    variant: Option<u32>,
    /// Common dimension K (kernel 780: 8192/16384/32768/65536; kernel 448: 4096)
    #[arg(long)]
    k: Option<usize>,
    /// Maximum hits collected per round
    #[arg(long, default_value_t = 64)]
    hit_capacity: usize,
    /// Stop after this many accepted shares (0 = never)
    #[arg(long, default_value_t = 0)]
    stop_after_shares: u64,
    /// Print events as JSON lines instead of the human-readable display
    #[arg(long)]
    json: bool,
    /// Rounds in flight on the GPU
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(2..=4))]
    depth: u8,
    /// Capacity of the proof and submit queues
    #[arg(long, default_value_t = 64)]
    queue_size: usize,
    /// Maximum unacknowledged pool submissions
    #[arg(long, default_value_t = 4)]
    max_inflight: usize,
    /// Script or binary with GPU priority (e.g. an inference server that stays in VRAM): while a
    /// process running it, or started by it, computes on the GPU, mining pauses. Repeatable.
    #[arg(long, value_name = "PATH")]
    priority: Vec<PathBuf>,
    /// Yield the GPU to every other process that computes on it (no --priority needed)
    #[arg(long)]
    background: bool,
    /// Quiet time of the priority programs before mining resumes
    #[arg(long, default_value_t = 10.0, value_name = "SECONDS")]
    priority_resume_delay: f64,
    /// SM utilisation of a priority process that counts as GPU use
    #[arg(long, default_value_t = 5, value_name = "PERCENT")]
    priority_util: u32,
    /// Round-time increase that, together with CPU activity of a priority process, counts as GPU use
    #[arg(long, default_value_t = 20.0, value_name = "PERCENT")]
    priority_slowdown: f64,
    /// Stop mining unless NVML reports exactly this power limit (W); not checked by default
    #[arg(long, value_name = "WATTS")]
    power_limit: Option<f64>,
    /// Stop mining at this GPU temperature (°C)
    #[arg(long, default_value_t = 85)]
    max_temp: u32,
    /// CUDA device ordinal
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// JSONL event log (default: logs/bpm-<time>.jsonl in the working directory)
    #[arg(long)]
    log: Option<PathBuf>,
    /// Optional read-only HTTP stats bind (disabled by default; no authentication)
    #[arg(long, value_name = "IP:PORT")]
    web_listen: Option<String>,
    /// LAN CIDR allowed to access the stats page (repeatable; loopback is always allowed)
    #[arg(long, value_name = "CIDR", requires = "web_listen")]
    web_allow_net: Vec<String>,
    /// Trusted helper invoked with argument 300 on normal/error/first-signal shutdown
    #[arg(long, value_name = "SCRIPT")]
    restore_power_limit_script: Option<PathBuf>,
    /// Seconds between pool reconnect attempts
    #[arg(long, default_value_t = 5)]
    reconnect_seconds: u64,
    /// Offline self-test: mine a fixed job and verify proofs with the official verifier
    #[arg(long)]
    self_test: bool,
}

/// Self-test job: fixed header, block target (frequent hits at these dimensions).
fn test_job(profile: Profile) -> Result<MiningJob> {
    let nbits = 0x1c00ffff;
    let header = IncompleteBlockHeader { version: 0, prev_block: [0; 32], merkle_root: std::array::from_fn(|i| i as u8), timestamp: 1792738800, nbits };
    let raw = primitive_types::U256::from(nbits & 0x7fffff) << (8 * ((nbits >> 24) - 3));
    MiningJob::new(PoolJob { job_id: "self-test".into(), header: header.to_bytes().to_vec(), raw_target: raw, cert_version: 3 }, profile)
}

fn self_test(device: i32) -> Result<()> {
    let dev = cuda::Device::open(device)?;
    let info = dev.info()?;
    ensure!(info.capability == (12, 0), "the kernels are built for compute capability 12.0 (found {} with {}.{})", info.name, info.capability.0, info.capability.1);
    println!("{} ({} SMs, {} MiB)", info.name, info.sms, info.total >> 20);
    for p in [Profile::new(Kernel::K780, 65536)?, Profile::new(Kernel::K780, 16384)?, Profile::new(Kernel::K448, 4096)?] {
        let job = test_job(p)?;
        let mut e = engine::Engine::new(p, 2, 4096)?;
        let (heap, root_b) = e.set_job(&job, 0)?;
        let cache = proof::ProofCache::new(p, heap, job.key, root_b)?;
        let mut verified = 0;
        for _ in 0..2 {
            e.launch(0)?;
            let s = e.collect(0)?;
            for &g in s.groups.iter().take(3) {
                proof::verify(&job, &cache.proof(g, &s.signal, &s.root_a)?)?;
                verified += 1;
            }
        }
        ensure!(verified > 0, "kernel {} K={} produced no hits", p.variant(), p.k);
        println!("kernel {} K={}: {verified} proofs verified", p.variant(), p.k);
    }
    println!("self-test passed");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.self_test {
        return self_test(cli.device);
    }
    let kernel = match cli.variant {
        None | Some(780) => Kernel::K780,
        Some(448) => Kernel::K448,
        Some(v) => bail!("unsupported --variant {v} (780 or 448)"),
    };
    let default_profile = cli.variant.is_none() && cli.k.is_none();
    let k = cli.k.unwrap_or(match kernel {
        Kernel::K448 => 4096,
        // Shorter rounds hand the GPU over faster to priority programs.
        Kernel::K780 if !cli.priority.is_empty() || cli.background => 16384,
        Kernel::K780 => 65536,
    });
    let profile = Profile::new(kernel, k)?;
    let Some(pool) = cli.pool else { bail!("--pool HOST:PORT is required (or use --self-test)") };
    let Some(wallet) = cli.wallet else { bail!("--wallet is required") };
    ensure!((1.0..=1000.0).contains(&cli.priority_slowdown), "--priority-slowdown must be 1..1000 percent");
    ensure!(cli.hit_capacity >= 1 && (1..=1024).contains(&cli.queue_size) && (1..=16).contains(&cli.max_inflight), "invalid --hit-capacity, --queue-size or --max-inflight");
    let log_path = cli.log.unwrap_or_else(|| PathBuf::from(format!("logs/bpm-{}.jsonl", stamp())));
    let code = miner::run(miner::MineArgs {
        pool,
        wallet,
        worker: cli.worker,
        seconds: cli.seconds,
        profile,
        default_profile,
        hit_capacity: cli.hit_capacity,
        stop_after: cli.stop_after_shares,
        json: cli.json,
        depth: cli.depth as usize,
        queue_size: cli.queue_size,
        max_inflight: cli.max_inflight,
        web_listen: cli.web_listen,
        web_allow_net: cli.web_allow_net,
        restore_power_limit_script: cli.restore_power_limit_script,
        priority: cli.priority,
        background: cli.background,
        priority_resume_delay: cli.priority_resume_delay,
        priority_util: cli.priority_util,
        priority_slowdown: cli.priority_slowdown,
        power_limit: cli.power_limit,
        max_temp: cli.max_temp,
        device: cli.device,
        log_path,
        nonce_start: None,
        reconnect_seconds: cli.reconnect_seconds,
    })?;
    std::process::exit(code);
}

fn stamp() -> String {
    let t = unsafe { libc::time(std::ptr::null_mut()) };
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&t, &mut tm) };
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
}
