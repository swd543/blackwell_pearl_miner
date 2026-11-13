//! Exact structured proofs from the immutable keyed-BLAKE3 Merkle heap and the private
//! nonce path. Hashing uses the official pearl-blake3 primitives; proofs are checked
//! with the official zk-pow verifier before submission.
use crate::protocol::{MiningJob, Profile, M, N};
use anyhow::{ensure, Context, Result};
use pearl_blake3::{Blake3Hasher, MerkleProof, MerkleTree};
use std::collections::{BTreeSet, HashMap};
use zk_pow::{
    api::{proof::SeedDerivation, verify::verify_plain_proof},
    ffi::plain_proof::{check_cert_version_eligible, MatrixMerkleProof, PlainProof},
};

pub const CHUNK: usize = 1024;

pub struct ProofCache {
    pub profile: Profile,
    nleaves: usize,
    capacity: usize,
    nonce_leaf: usize,
    /// Chaining values of the all-64 tree, heap order (node i at i*32), 2*capacity nodes.
    heap: Vec<u8>,
    hasher: Blake3Hasher,
    pub root_b: [u8; 32],
}

impl ProofCache {
    pub fn heap_bytes(profile: &Profile) -> usize {
        2 * (M * profile.k / CHUNK).next_power_of_two() * 32
    }

    /// One sibling CV per level, in the order consumed by the GPU root update.
    pub fn nonce_sibling_bytes(profile: &Profile) -> usize {
        let capacity = (M * profile.k / CHUNK).next_power_of_two();
        capacity.trailing_zeros() as usize * 32
    }

    pub fn nonce_siblings(profile: &Profile, heap: &[u8]) -> Result<Vec<u8>> {
        ensure!(heap.len() == Self::heap_bytes(profile), "wrong Merkle heap length");
        let capacity = (M * profile.k / CHUNK).next_power_of_two();
        let mut node = capacity + (M - 1) * profile.k / CHUNK;
        let mut siblings = Vec::with_capacity(Self::nonce_sibling_bytes(profile));
        while node > 1 {
            let start = (node ^ 1) * 32;
            siblings.extend_from_slice(&heap[start..start + 32]);
            node >>= 1;
        }
        Ok(siblings)
    }

    pub fn new(profile: Profile, heap: Vec<u8>, key: [u8; 32], gpu_root_b: [u8; 32]) -> Result<Self> {
        let nleaves = M * profile.k / CHUNK;
        let capacity = nleaves.next_power_of_two();
        ensure!(nleaves == capacity, "structured proofs need a power-of-two leaf count");
        ensure!(heap.len() == 2 * capacity * 32, "wrong Merkle heap length");
        let hasher = Blake3Hasher::with_key(key);
        let cache = ProofCache { profile, nleaves, capacity, nonce_leaf: (M - 1) * profile.k / CHUNK, heap, hasher, root_b: gpu_root_b };
        let root = cache.hasher.root_cv(&cache.node(2), &cache.node(3));
        ensure!(root == gpu_root_b, "cached CPU base root and GPU B root differ");
        Ok(cache)
    }

    fn node(&self, i: usize) -> [u8; 32] {
        self.heap[i * 32..(i + 1) * 32].try_into().unwrap()
    }

    fn nonce_chunk(signal: &[u8; 64]) -> [u8; CHUNK] {
        let mut chunk = [64u8; CHUNK];
        chunk[..64].copy_from_slice(signal);
        chunk
    }

    /// Nonce leaf CV and all its ancestors (root last), keyed by heap node index.
    fn nonce_path(&self, signal: &[u8; 64], root_a: &[u8; 32]) -> Result<HashMap<usize, [u8; 32]>> {
        ensure!(signal.iter().all(|b| *b == 63 || *b == 64), "invalid structured nonce signal snapshot");
        let mut path = HashMap::new();
        let mut node = self.capacity + self.nonce_leaf;
        let mut cv = self.hasher.chunk_cv(&Self::nonce_chunk(signal), self.nonce_leaf as u64);
        path.insert(node, cv);
        while node > 1 {
            let sibling = self.node(node ^ 1);
            let (l, r) = if node & 1 == 1 { (sibling, cv) } else { (cv, sibling) };
            cv = if node < 4 { self.hasher.root_cv(&l, &r) } else { self.hasher.parent_cv(&l, &r) };
            node >>= 1;
            path.insert(node, cv);
        }
        ensure!(&cv == root_a, "cached CPU nonce root and GPU A root differ");
        Ok(path)
    }

    fn matrix_proof(&self, rows: Vec<usize>, root: [u8; 32], path: &HashMap<usize, [u8; 32]>, signal: Option<&[u8; 64]>) -> Result<MatrixMerkleProof> {
        let leaves = MerkleTree::compute_leaf_indices_from_rows(&rows, (M, self.profile.k));
        ensure!(!leaves.is_empty() && *leaves.last().unwrap() < self.nleaves, "invalid proof rows");
        let leaf_data = leaves
            .iter()
            .map(|&i| match signal {
                Some(s) if i == self.nonce_leaf => Self::nonce_chunk(s),
                _ => [64u8; CHUNK],
            })
            .collect();
        let mut siblings = Vec::new();
        let mut current = leaves.clone();
        let (mut width, mut base) = (self.nleaves, self.capacity);
        while width > 1 {
            let selected: BTreeSet<usize> = current.iter().copied().collect();
            for &i in &current {
                let other = i ^ 1;
                if other < width && !selected.contains(&other) {
                    let node = base + other;
                    siblings.push(path.get(&node).copied().unwrap_or_else(|| self.node(node)));
                }
            }
            current = current.iter().map(|i| i / 2).collect::<BTreeSet<_>>().into_iter().collect();
            width = width.div_ceil(2);
            base >>= 1;
        }
        Ok(MatrixMerkleProof {
            row_indices: rows,
            proof: MerkleProof { leaf_data, leaf_indices: leaves, total_leaves: self.nleaves, root, siblings },
        })
    }

    pub fn proof(&self, group: u32, signal: &[u8; 64], root_a: &[u8; 32]) -> Result<PlainProof> {
        ensure!(group < self.profile.groups(), "GPU emitted invalid group index {group}");
        let path = self.nonce_path(signal, root_a)?;
        let (rows, cols) = self.profile.indices(group);
        Ok(PlainProof {
            m: M,
            n: N,
            k: self.profile.k,
            noise_rank: 128,
            a: self.matrix_proof(rows, *root_a, &path, Some(signal))?,
            bt: self.matrix_proof(cols, self.root_b, &HashMap::new(), None)?,
            moe: None,
        })
    }
}

/// Official verification at the share's compact target, then the bincode wire bytes.
pub fn verify(job: &MiningJob, proof: &PlainProof) -> Result<Vec<u8>> {
    check_cert_version_eligible(job.pool.cert_version, proof)?;
    verify_plain_proof(
        &job.header,
        proof,
        Some(job.bits),
        if job.pool.cert_version == 3 { SeedDerivation::Salted } else { SeedDerivation::Legacy },
    )
    .context("official verifier rejected GPU share")?;
    Ok(bincode::serialize(proof)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Kernel;

    #[test]
    fn compact_siblings_follow_the_nonce_path_for_every_profile() {
        for (kernel, k) in [(Kernel::K780, 8192), (Kernel::K780, 16384),
                            (Kernel::K780, 32768), (Kernel::K780, 65536), (Kernel::K448, 4096)] {
            let profile = Profile::new(kernel, k).unwrap();
            let capacity = M * k / CHUNK;
            let mut heap = vec![0u8; ProofCache::heap_bytes(&profile)];
            // Every sibling is distinct, including all eight words of each CV.
            let mut node = capacity + (M - 1) * k / CHUNK;
            let mut expected = Vec::new();
            while node > 1 {
                let sibling = node ^ 1;
                for word in 0..8 {
                    let value = ((sibling as u32) * 8 + word as u32).to_le_bytes();
                    heap[sibling * 32 + word * 4..sibling * 32 + word * 4 + 4].copy_from_slice(&value);
                    expected.extend_from_slice(&value);
                }
                node >>= 1;
            }
            let compact = ProofCache::nonce_siblings(&profile, &heap).unwrap();
            assert_eq!(compact, expected);
            assert_eq!(compact.len(), ProofCache::nonce_sibling_bytes(&profile));
            assert!(compact.len() <= 21 * 32);
            assert!(ProofCache::nonce_siblings(&profile, &heap[..heap.len() - 1]).is_err());
        }
    }
}
