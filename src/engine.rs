//! GPU engine: per-job preparation, one captured CUDA graph per ring slot, packed
//! result readback. One round:
//!   advance nonce -> Merkle root of A -> commitment -> noise(A) -> pack/prepare A
//!   -> zero hit count -> mining kernel -> copy nonce signal + root_A -> D2H.
use crate::cuda::{DevBuf, Event, Graph, HostBuf, Stream};
use crate::kernels::Kernels;
use crate::protocol::{Kernel, MiningJob, Profile, M, N, RANK};
use anyhow::{ensure, Result};
use std::ffi::c_void;

/// Kernel inputs. Kernel 780 reads only packed rank-128 factors: A is the constant 64
/// signal except its last row's 64-byte nonce and B is constant, so one K-byte row of each
/// suffices (the kernels address row m-1 through a shifted base pointer).
enum Operands {
    Lean { a_row: DevBuf, b_row: DevBuf, packed_a: DevBuf, packed_b: DevBuf, packed_ia: DevBuf, packed_ib: DevBuf },
    Dense { a: DevBuf, b: DevBuf, an: DevBuf, bn: DevBuf, pairs_a: DevBuf, pairs_b: DevBuf },
}

struct Slot {
    graph: Option<Graph>,
    result: DevBuf,
    host: HostBuf,
    event: Event,
    pending: bool,
    sequence: u64,
}

pub struct Snapshot {
    pub sequence: u64,
    pub groups: Vec<u32>,
    pub overflow: u32,
    pub signal: [u8; 64],
    pub root_a: [u8; 32],
}

pub struct Engine {
    pub profile: Profile,
    pub capacity: usize,
    support: &'static Kernels,
    stream: Stream,
    ops: Operands,
    key: DevBuf,
    target: DevBuf,
    root_a: DevBuf,
    root_b: DevBuf,
    key_a: DevBuf,
    key_b: DevBuf,
    nonce: DevBuf,
    nonce_siblings: DevBuf,
    eal: DevBuf,
    ear: DevBuf,
    ebl: DevBuf,
    ebr: DevBuf,
    slots: Vec<Slot>,
    sequence: u64,
    cert: i32,
}

fn rc(code: i32, what: &str) -> Result<()> {
    crate::cuda::check_kernel(code, what)
}

impl Engine {
    pub fn new(profile: Profile, depth: usize, capacity: usize) -> Result<Self> {
        ensure!((2..=4).contains(&depth) && capacity >= 1, "ring depth 2..4 and positive hit capacity required");
        let k = profile.k;
        let support = crate::kernels::get()?;
        let ops = match profile.kernel {
            Kernel::K780 => Operands::Lean {
                    a_row: DevBuf::filled(k, 64)?,
                    b_row: DevBuf::filled(k, 64)?,
                    packed_a: DevBuf::new(M * RANK)?,
                    packed_b: DevBuf::new(N * RANK)?,
                    packed_ia: DevBuf::new(k * 4)?,
                    packed_ib: DevBuf::new(k * 4)?,
                },
            Kernel::K448 => Operands::Dense {
                    a: DevBuf::filled(M * k, 64)?,
                    b: DevBuf::filled(N * k, 64)?,
                    an: DevBuf::new(M * k)?,
                    bn: DevBuf::new(N * k)?,
                    pairs_a: DevBuf::new(k * 2)?,
                    pairs_b: DevBuf::new(k * 2)?,
                },
        };
        let result_bytes = (capacity + 1) * 4 + 96;
        let slots = (0..depth)
            .map(|_| Ok(Slot { graph: None, result: DevBuf::new(result_bytes)?, host: HostBuf::new(result_bytes)?, event: Event::new()?, pending: false, sequence: 0 }))
            .collect::<Result<Vec<_>>>()?;
        Ok(Engine {
            profile,
            capacity,
            support,
            stream: Stream::new()?,
            ops,
            key: DevBuf::new(32)?,
            target: DevBuf::new(32)?,
            root_a: DevBuf::new(32)?,
            root_b: DevBuf::new(32)?,
            key_a: DevBuf::new(32)?,
            key_b: DevBuf::new(32)?,
            nonce: DevBuf::new(8)?,
            nonce_siblings: DevBuf::new(crate::proof::ProofCache::nonce_sibling_bytes(&profile))?,
            eal: DevBuf::new(M * RANK)?,
            ear: DevBuf::new(k * RANK)?,
            ebl: DevBuf::new(k * RANK)?,
            ebr: DevBuf::new(N * RANK)?,
            slots,
            sequence: 0,
            cert: 3,
        })
    }

    fn s(&self) -> *mut c_void {
        self.stream.raw()
    }

    /// Base address such that row m-1 of A lands on the stored row (lean) or the dense A.
    fn a_ptr(&self) -> *mut c_void {
        match &self.ops {
            Operands::Lean { a_row, .. } => (a_row.ptr - ((M - 1) * self.profile.k) as u64) as *mut c_void,
            Operands::Dense { a, .. } => a.p(),
        }
    }
    fn b_ptr(&self) -> *mut c_void {
        match &self.ops {
            Operands::Lean { b_row, .. } => (b_row.ptr - ((N - 1) * self.profile.k) as u64) as *mut c_void,
            Operands::Dense { b, .. } => b.p(),
        }
    }
    fn signal_ptr(&self) -> u64 {
        match &self.ops {
            Operands::Lean { a_row, .. } => a_row.ptr,
            Operands::Dense { a, .. } => a.ptr + ((M - 1) * self.profile.k) as u64,
        }
    }

    /// One round's A-side preparation (and B-side once per job when `with_b`).
    fn prepare(&self, with_b: bool) -> Result<()> {
        let (s, k, sp) = (self.s(), self.profile.k as i32, self.support);
        unsafe {
            rc((sp.advance_nonce)(self.a_ptr(), self.nonce.p(), k, s), "advance_nonce")?;
            rc((sp.merkle_root)(self.a_ptr(), self.key.p(), self.nonce_siblings.p(), self.root_a.p(), k, s), "merkle_root(A)")?;
            if with_b {
                // Constant B: its root is the cached constant tree with a constant nonce leaf.
                rc((sp.merkle_root)(self.b_ptr(), self.key.p(), self.nonce_siblings.p(), self.root_b.p(), k, s), "merkle_root(B)")?;
            }
            rc((sp.commit)(self.root_a.p(), self.root_b.p(), self.key.p(), self.key_a.p(), self.key_b.p(), self.cert, M as i32, N as i32, s), "commit")?;
            let which = if with_b { 3 } else { 1 };
            rc((sp.noise)(self.eal.p(), self.ear.p(), self.ebl.p(), self.ebr.p(), self.key_a.p(), self.key_b.p(), M as i32, N as i32, k, which, s), "noise")?;
            match &self.ops {
                Operands::Lean { packed_a, packed_b, packed_ia, packed_ib, .. } => {
                    rc((sp.build_indices)(self.ear.p(), packed_ia.p(), k, s), "build_indices(A)")?;
                    rc((sp.pack_basis)(self.eal.p(), packed_a.p(), s), "pack_basis(A)")?;
                    if with_b {
                        rc((sp.build_indices)(self.ebl.p(), packed_ib.p(), k, s), "build_indices(B)")?;
                        rc((sp.pack_basis)(self.ebr.p(), packed_b.p(), s), "pack_basis(B)")?;
                    }
                }
                Operands::Dense { a, b, an, bn, pairs_a, pairs_b } => {
                    rc((sp.prepare_packed)(a.p(), self.eal.p(), self.ear.p(), pairs_a.p(), an.p(), k, 0, s), "prepare(A)")?;
                    if with_b {
                        rc((sp.prepare_packed)(b.p(), self.ebr.p(), self.ebl.p(), pairs_b.p(), bn.p(), k, 1, s), "prepare(B)")?;
                    }
                }
            }
        }
        Ok(())
    }

    fn mine(&self, slot: &Slot) -> Result<()> {
        let (s, k) = (self.s(), self.profile.k as i32);
        let count = slot.result.p();
        let ids = (slot.result.ptr + 4) as *mut c_void;
        self.stream.memset(slot.result.ptr, 0, 4)?;
        let sp = self.support;
        unsafe {
            match &self.ops {
                Operands::Lean { packed_a, packed_b, packed_ia, packed_ib, .. } => rc(
                    (sp.k780)(packed_a.p(), packed_b.p(), packed_ia.p(), packed_ib.p(), M as i32, N as i32, k,
                        self.key_a.p(), self.target.p(), count, ids, self.capacity as i32, s),
                    "kernel 780",
                )?,
                Operands::Dense { an, bn, .. } => rc(
                    (sp.k448)(an.p(), bn.p(), self.key_a.p(), self.target.p(), count, ids, self.capacity as i32, s),
                    "kernel 448",
                )?,
            }
        }
        let meta = slot.result.ptr + (self.capacity as u64 + 1) * 4;
        self.stream.copy_d2d(meta, self.signal_ptr(), 64)?;
        self.stream.copy_d2d(meta + 64, self.root_a.ptr, 32)
    }

    /// Prepare a job: Merkle cache, B factors, one setup round (consumes nonce `start`),
    /// then capture one graph per slot. Returns the Merkle heap and root_B for proofs.
    pub fn set_job(&mut self, job: &MiningJob, start_nonce: u64) -> Result<(Vec<u8>, [u8; 32])> {
        ensure!(self.slots.iter().all(|s| !s.pending), "set_job with rounds in flight");
        ensure!(job.profile.kernel == self.profile.kernel && job.profile.k == self.profile.k, "job profile differs from engine");
        self.stream.sync()?;
        for s in &mut self.slots {
            s.graph = None;
        }
        self.cert = job.pool.cert_version as i32;
        self.key.upload(&job.key)?;
        self.target.upload(&job.bound)?;
        self.nonce.upload(&start_nonce.to_le_bytes())?;
        // Only setup needs the full tree. Download it for CPU proofs, retain the
        // nonce leaf's sibling chain on GPU, and free the tree before graph capture.
        // Captured graphs reference only the stable nonce_siblings allocation.
        let heap = {
            let tree = DevBuf::new(crate::proof::ProofCache::heap_bytes(&self.profile))?;
            unsafe { rc((self.support.merkle_init)(self.key.p(), tree.p(), self.profile.k as i32, self.s()), "merkle_init")? };
            self.stream.sync()?;
            let mut heap = vec![0u8; tree.bytes];
            tree.download(&mut heap)?;
            self.nonce_siblings.upload(&crate::proof::ProofCache::nonce_siblings(&self.profile, &heap)?)?;
            heap
        };
        self.prepare(true)?;
        self.stream.sync()?;
        for i in 0..self.slots.len() {
            self.stream.begin_capture()?;
            let captured = self.prepare(false).and_then(|_| self.mine(&self.slots[i]));
            let graph = self.stream.end_capture();
            captured?;
            self.slots[i].graph = Some(graph?);
        }
        let mut root_b = [0u8; 32];
        self.root_b.download(&mut root_b)?;
        Ok((heap, root_b))
    }

    pub fn depth(&self) -> usize {
        self.slots.len()
    }

    pub fn launch(&mut self, i: usize) -> Result<()> {
        let bytes = self.slots[i].result.bytes;
        let slot = &self.slots[i];
        ensure!(!slot.pending, "GPU slot {i} occupied");
        slot.graph.as_ref().expect("captured graph").launch(&self.stream)?;
        self.stream.copy_d2h(slot.host.ptr, slot.result.ptr, bytes)?;
        slot.event.record(&self.stream)?;
        let seq = self.sequence;
        self.sequence += 1;
        let slot = &mut self.slots[i];
        slot.pending = true;
        slot.sequence = seq;
        Ok(())
    }

    pub fn collect(&mut self, i: usize) -> Result<Snapshot> {
        let cap = self.capacity;
        let slot = &mut self.slots[i];
        ensure!(slot.pending, "GPU slot {i} empty");
        slot.event.sync()?;
        slot.pending = false;
        let raw = slot.host.slice();
        let count = u32::from_le_bytes(raw[..4].try_into().unwrap());
        let n = (count as usize).min(cap);
        let groups = (0..n).map(|j| u32::from_le_bytes(raw[4 + 4 * j..8 + 4 * j].try_into().unwrap())).collect();
        let meta = &raw[(cap + 1) * 4..];
        Ok(Snapshot {
            sequence: slot.sequence,
            groups,
            overflow: count.saturating_sub(cap as u32),
            signal: meta[..64].try_into().unwrap(),
            root_a: meta[64..96].try_into().unwrap(),
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.stream.sync();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::ProofCache;
    use crate::protocol::hex;
    use serde_json::json;

    /// Run explicitly on a trusted Blackwell GPU, never on hosted CI. Sorted hit IDs,
    /// roots and serialized proofs form an exact before/after regression fixture.
    #[test]
    #[ignore = "requires an idle Blackwell GPU and coordinated thermal/power controls"]
    fn gpu_round_regression() -> Result<()> {
        let device = crate::cuda::Device::open(0)?;
        let info = device.info()?;
        ensure!(info.capability == (12, 0), "Blackwell 12.0 required");
        let nvml = crate::nvml::Nvml::open(&info.pci_bus_id)?;
        let safe = || -> Result<()> {
            ensure!(nvml.temperature_c()? < 88, "GPU thermal cutoff reached");
            Ok(())
        };
        let mut fixtures = Vec::new();
        for (kernel, k) in [(Kernel::K780, 8192), (Kernel::K780, 16384),
                            (Kernel::K780, 32768), (Kernel::K780, 65536), (Kernel::K448, 4096)] {
            safe()?;
            let profile = Profile::new(kernel, k)?;
            let (free_before, _) = crate::cuda::memory_info()?;
            let mut engine = Engine::new(profile, 4, 4096)?;
            // A second job exercises recapture and sibling-path replacement on one engine.
            for (cert, nonce) in [(3, 0), (2, 0x123456789abcdef0u64)] {
                safe()?;
                let mut pool = crate::test_job(profile)?.pool;
                pool.cert_version = cert;
                pool.header[36] ^= cert as u8;
                let job = MiningJob::new(pool, profile)?;
                let (heap, root_b) = engine.set_job(&job, nonce)?;
                let heap_hash = blake3::hash(&heap[64..]).to_hex().to_string();
                let cache = ProofCache::new(profile, heap, job.key, root_b)?;
                let (free_after, _) = crate::cuda::memory_info()?;
                println!("GPU_MEMORY={}", json!({"variant":profile.variant(), "k":k,
                    "cert":cert, "used_bytes":free_before.saturating_sub(free_after)}));
                for slot in 0..engine.depth() { engine.launch(slot)?; }
                for round in 0..8 {
                    safe()?;
                    let slot = round % engine.depth();
                    let mut snap = engine.collect(slot)?;
                    ensure!(snap.overflow == 0, "regression fixture overflowed");
                    snap.groups.sort_unstable();
                    let mut proof_hashes = Vec::new();
                    for &group in snap.groups.iter().take(3) {
                        let bytes = crate::proof::verify(&job, &cache.proof(group, &snap.signal, &snap.root_a)?)?;
                        proof_hashes.push(blake3::hash(&bytes).to_hex().to_string());
                    }
                    // Check the nonce root even when a particular round has no hits.
                    cache.proof(0, &snap.signal, &snap.root_a)?;
                    fixtures.push(json!({"variant":profile.variant(), "k":k, "cert":cert,
                        "round":round, "sequence":snap.sequence, "signal":hex(&snap.signal),
                        "root_a":hex(&snap.root_a), "root_b":hex(&root_b), "heap":heap_hash,
                        "groups":snap.groups, "proofs":proof_hashes}));
                    if round < 4 { engine.launch(slot)?; }
                }
            }
            safe()?;
        }
        println!("GPU_FIXTURE={}", serde_json::to_string(&fixtures)?);
        Ok(())
    }
}
