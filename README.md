# blackwell_pearl_miner (bpm)

This is the `swd543` fork of [pushsoftnow/blackwell_pearl_miner](https://github.com/pushsoftnow/blackwell_pearl_miner), retaining the upstream repository name and Rust/CUDA implementation.

PearlHash pool miner for NVIDIA Blackwell GPUs with compute capability 12.0 (sm_120).
The host program is written in Rust; the mining kernels are CUDA C++.

Every share is checked with the official Pearl verifier (`zk-pow`) before it is submitted.

## Quickstart (Kryptex pool)

```bash
git clone --recursive https://github.com/swd543/blackwell_pearl_miner.git
cd blackwell_pearl_miner
cargo build --release
./target/release/bpm --pool prl-eu.kryptex.network:7048 --wallet YOUR_KRYPTEX_ID --worker rig1
```

Replace `YOUR_KRYPTEX_ID` with your Kryptex account ID (it starts with `krx`) and `rig1` with
any worker name. Add `--background` if the GPU is shared with other programs (games, LLM
servers): mining then pauses whenever another process computes on the GPU.

No Kryptex account yet? Sign up here: **[kryptex.com](https://www.kryptex.com/?ref=973ba14d)**

> **Disclaimer:** This is a referral link. Using it is optional and has no impact on your
> mining: hashrate, shares and the miner itself are exactly the same with or without it, and
> your payouts are not reduced — Kryptex pays the referral bonus out of its own share.
> blackwell_pearl_miner is completely free and open source — there is **no dev fee**; 100 % of
> your hashrate goes to the pool and wallet you configure.

## Requirements

* Linux x86_64
* NVIDIA GPU with compute capability 12.0 and at least 2 GiB of free memory
* NVIDIA driver with CUDA 12.8 support or newer (provides `libcuda.so.1` and `libnvidia-ml.so.1`)
* CUDA Toolkit 12.8 or newer with `nvcc` (build only), and a host compiler supported by it
* Rust stable toolchain (`cargo`)

Tested on Ubuntu 24.04 with an RTX PRO 4500 Blackwell, driver 620.70, CUDA 13.4 and Rust 1.98.
Other compute capability 12.0 GPUs are expected to work but have not been tested.

## Build

```bash
git clone --recursive https://github.com/swd543/blackwell_pearl_miner.git
cd blackwell_pearl_miner
cargo build --release
```

If the repository was cloned without `--recursive`, run `git submodule update --init --recursive`.
The CUDA Toolkit is taken from `$CUDA_HOME` (or `$CUDA_PATH`, default `/usr/local/cuda`);
`$NVCC` overrides the compiler. The kernels are compiled into a library that is embedded in the
binary, so `target/release/bpm` is the only file needed at run time.

### CI release artifacts

Pushing a `v*` tag runs `.github/workflows/linux-release.yml` and attaches compiled Linux x86_64
tarballs for Fedora (rolling), Arch Linux (rolling), Debian 12, and Debian 13, plus SHA-256 files.
The archives contain `bpm` and the project notices; the NVIDIA driver is supplied by the host. The
workflow runs Rust tests and CLI smoke checks in each build environment. GPU self-tests require a
Blackwell GPU and are not run on GitHub-hosted runners. No Windows artifacts are produced.

## Usage

```bash
./target/release/bpm --self-test
./target/release/bpm --pool HOST:PORT --wallet ADDRESS_OR_ACCOUNT --worker NAME
```

`--self-test` mines a fixed job offline and verifies the resulting proofs for every kernel profile.

| Option | Meaning |
|---|---|
| `--pool HOST:PORT` | Pearl Stratum endpoint |
| `--wallet`, `--worker` | Credentials sent to the pool |
| `--seconds N` | Stop after N seconds (default: unlimited) |
| `--device N` | CUDA device ordinal (default 0) |
| `--variant 780\|448`, `--k K` | Kernel profile (default: kernel 780 with K = 65536) |
| `--power-limit W` | Require this GPU power limit while mining (not checked by default); with the helper, 250 W switches to 300 W while yielded |
| `--max-temp C` | Stop at this GPU temperature (default 85 °C) |
| `--log PATH` | JSONL event log (default `logs/bpm-<time>.jsonl`) |
| `--restore-power-limit-script PATH` | Run this trusted helper with `300` on exit and, for a 250 W mining limit, while yielded; reapply 250 W before resuming |
| `--web-listen IP:PORT` | Optional read-only HTTP dashboard and stats/history JSON endpoints; disabled by default |
| `--history-window-hours N` | In-memory GPU temperature/hashrate history, default 36 hours (1.5 days), range 1–168 |
| `--json` | Print events as JSON lines |
| `--priority PATH` | Yield the GPU to a program, see below |
| `--background` | Yield the GPU to every other program that computes on it |

`--help` lists all options. Stop the miner with Ctrl+C; it finishes the rounds in flight and
waits for outstanding pool acknowledgements. A second Ctrl+C exits immediately.

For power-limit restoration, pass only a trusted local policy helper, for example
`--restore-power-limit-script ./scripts/power-limit.sh`. That helper allows only 250 W and 300 W
and invokes `/usr/bin/nvidia-smi` through the existing non-interactive sudo rule; the miner itself
never invokes `nvidia-smi`. When paired with `--power-limit 250`, BPM calls the helper with `300`
while yielded and reapplies `250` before resuming. It also restores `300` on normal/error shutdown
and the first SIGINT/SIGTERM, but cannot run after SIGKILL, power loss, or a forced second signal.
Keep an external cleanup backstop for those cases.

The HTTP endpoint is deliberately opt-in and unauthenticated. Loopback is always allowed; other
clients are denied unless explicitly listed with repeatable `--web-allow-net` CIDRs. The LAN
page uses port 52693 to match the previous portal. Example for these host LANs:

```bash
--web-listen 0.0.0.0:52693 \
--web-allow-net 192.168.50.0/24 \
--web-allow-net 192.168.68.0/24
```

Only list trusted local subnets, and keep the host firewall enabled; do not expose the page to a
public network. The dashboard includes a bounded tail of formatted miner events and GPU
temperature/power-limit readings. Temperature reuses the miner's existing 5-second safety-guard
sample; no extra GPU polling is added. The power tile is the configured limit, not actual draw; no
additional NVML power-usage query is made. Temperature samples and existing 30-second hashrate stats
are retained in bounded RAM rings and streamed initially/incrementally to the chart. The time
axis expands to available history, then caps at the configured window instead of compressing early
samples against an empty 36-hour span. The default 36-hour window uses about 1 MiB of ring storage.
No history is written to disk, so it starts fresh when the miner restarts. Event tailing reads at
most 128 KiB from the existing JSONL log when a stats request arrives, omits the startup row, and
returns only a small allowlist of formatted event
types.

### Kernel profiles

* **Kernel 780** (default) works on packed rank-128 noise factors and needs about 0.5 GiB of GPU
  memory. It uses K = 65536, or K = 16384 with `--priority` or `--background` (shorter rounds). Long-K proofs are
  only accepted by pools that negotiate gzip-compressed proofs.
  Each round, the K columns inside every 128-column block are reordered to avoid shared-memory bank conflicts
  in the operand rebuild (bit-identical results, about +1 % at a fixed power limit).
* **Kernel 448** works on dense prepared operands with K = 4096 and needs about 0.8 GiB. If the
  pool does not negotiate gzip proofs, the miner switches to this profile automatically.

### Sharing the GPU (`--priority`)

`--priority PATH` (repeatable) names a script or binary that has precedence on the GPU, for
example an inference server that keeps its model in GPU memory. While a process running that
file, or started by it, computes on the GPU, the miner pauses; it resumes after
`--priority-resume-delay` seconds without GPU activity (default 10).

GPU activity is detected when the miner's own rounds slow down by more than
`--priority-slowdown` percent (default 20) while a priority process is using CPU time, or when
NVML reports at least `--priority-util` percent SM utilisation (default 5) for a priority
process. CPU load alone never pauses mining. No special privileges are required; processes of
other users are matched by their command line.

`--background` applies the same rules to every other process with a GPU context, without
naming it: the miner runs only while nobody else computes on the GPU. Processes that merely
hold GPU memory (an idle inference server, a desktop) do not pause mining. The three
`--priority-*` options tune this mode as well.

## Tests

```bash
cargo test --release
./target/release/bpm --self-test
```

## License

ISC, see [LICENSE](LICENSE). Third-party components and their licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This software is provided as is, without warranty. Mining puts sustained load on the GPU;
monitor temperature and power.
