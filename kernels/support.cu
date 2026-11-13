// SPDX-License-Identifier: ISC
// Per-round preparation kernels. Fixed square profile M = N = 32768, rank 128; the
// constant signal is 64 everywhere except the 64-byte nonce in row M-1 of A.
#include <cstdint>
#include <cuda_runtime.h>
#include "blake3/blake3.cuh"
#include "cutlass/device_kernel.h"
#include "gemm/noise_generation_kernel.h"
#include "tensor_hash/commitment_hash_from_merkle_roots_kernel.hpp"

namespace support {
constexpr int M = 32768;

// ---- Merkle cache: chaining values of the all-64 matrix tree (heap order) ----
__device__ auto key_cv(const uint32_t *key) {
  auto cv = cute::make_tensor<uint32_t>(cute::Shape<cute::_8>{});
#pragma unroll
  for (int i = 0; i < 8; ++i) cv(i) = key[i];
  return cv;
}
__device__ auto constant_chunk(const uint32_t *key, int leaf, const uint32_t *data) {
  auto cv = key_cv(key);
  auto msg = cute::make_tensor<uint32_t>(cute::Shape<cute::_16>{});
  for (int block = 0; block < 16; ++block) {
#pragma unroll
    for (int j = 0; j < 16; ++j) msg(j) = data ? data[block * 16 + j] : 0x40404040u;
    blake3::compress_msg_block_u32(msg, cv, {uint64_t(leaf), 64, uint32_t(blake3::KEYED_HASH | (block == 0 ? blake3::CHUNK_START : 0) | (block == 15 ? blake3::CHUNK_END : 0))});
  }
  return cv;
}
__global__ void leaves(const uint32_t *key, uint32_t *tree, int nleaves) {
  int leaf = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf >= nleaves) return;
  auto cv = constant_chunk(key, leaf, nullptr);
#pragma unroll
  for (int i = 0; i < 8; ++i) tree[(nleaves + leaf) * 8 + i] = cv(i);
}
__global__ void parents(const uint32_t *key, uint32_t *tree, int first, int count) {
  int node = first + blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= first + count) return;
  auto cv = key_cv(key);
  auto msg = cute::make_tensor<uint32_t>(cute::Shape<cute::_16>{});
#pragma unroll
  for (int i = 0; i < 16; ++i) msg(i) = tree[node * 16 + i];
  blake3::compress_msg_block_u32(msg, cv, blake3::COMPRESS_PARAMS_INNER_NODE);
#pragma unroll
  for (int i = 0; i < 8; ++i) tree[node * 8 + i] = cv(i);
}
// Root of the matrix whose only non-constant chunk is the nonce leaf, read from `a`.
// Sibling CVs are packed leaf-to-root; the full tree exists only during job setup.
__global__ void update(const uint32_t *a, const uint32_t *key, const uint32_t *siblings, uint32_t *root, int nleaves, int nonce_leaf) {
  if (threadIdx.x) return;
  auto cv = constant_chunk(key, nonce_leaf, a + size_t(nonce_leaf) * 256);
  auto msg = cute::make_tensor<uint32_t>(cute::Shape<cute::_16>{});
  for (int node = nleaves + nonce_leaf, level = 0; node > 1; node >>= 1, ++level) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      msg((node & 1) ? i + 8 : i) = cv(i);
      msg((node & 1) ? i : i + 8) = siblings[level * 8 + i];
    }
    cv = key_cv(key);
    blake3::compress_msg_block_u32(msg, cv, node == 2 || node == 3 ? blake3::COMPRESS_PARAMS_ROOT : blake3::COMPRESS_PARAMS_INNER_NODE);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) root[i] = cv(i);
}

// ---- Nonce and packed basis ----
__global__ void advance_nonce(int8_t *a, unsigned long long *counter, int k) {
  __shared__ unsigned long long nonce;
  if (threadIdx.x == 0) nonce = atomicAdd(counter, 1ull);
  __syncthreads();
  int lane = threadIdx.x;
  a[size_t(M - 1) * k + lane] = 63 + ((nonce >> lane) & 1ull);
}
// 16-row fragments of a 128-component basis, biased by +32.
__global__ void pack_basis_frag(const int8_t *input, uint8_t *output) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(M) * 128) {
    size_t row = i / 128, component = i % 128;
    output[(row / 16) * 2048 + component * 16 + row % 16] = uint8_t(int(input[i]) + 32);
  }
}
// Sparse +1/-1 rows -> component indices (plus, minus) and prescaled packed offsets.
__global__ void build_indices(const int8_t *permutation, uint32_t *packed, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= k) return;
  int plus = 0, minus = 0;
  for (int j = 0; j < 128; ++j) {
    int v = permutation[col * 128 + j];
    if (v == 1) plus = j;
    if (v == -1) minus = j;
  }
  packed[col] = uint32_t(plus * 16) | (uint32_t(minus * 16) << 16);
}

// ---- Dense prepared operands (signal + noise), MMA fragment layout ----
__global__ __launch_bounds__(128) void pack_pairs(const int8_t *permutation, uint16_t *pairs, int k) {
  int lane = threadIdx.x % 32, col = blockIdx.x * 4 + threadIdx.x / 32;
  if (col >= k) return;
  int plus = 0, minus = 0;
#pragma unroll
  for (int first = 0; first < 128; first += 32) {
    int v = permutation[col * 128 + first + lane];
    unsigned p = __ballot_sync(0xffffffffu, v == 1), m = __ballot_sync(0xffffffffu, v == -1);
    if (p) plus = first + __ffs(p) - 1;
    if (m) minus = first + __ffs(m) - 1;
  }
  if (lane == 0) pairs[col] = uint16_t(plus | (minus << 7));
}
__device__ unsigned shared_address(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
template <int K, bool B>
__global__ __launch_bounds__(256) void materialize_packed(const int8_t *signal, const int8_t *dense, const uint16_t *pairs, uint4 *output) {
  __shared__ __align__(16) uint8_t basis[128 * 16];
  int fragment = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int at = threadIdx.x; at < 128 * 16; at += 256) {
    int row = at / 128, component = at % 128;
    basis[component * 16 + row] = uint8_t(int(dense[(fragment * 16 + row) * 128 + component]) + 32);
  }
  __syncthreads();
  for (int tile = warp; tile < K / 32; tile += 8) {
    unsigned pair = pairs[tile * 32 + lane], p = (pair & 127) * 16, m = ((pair >> 7) & 127) * 16;
    uint32_t pv[4], mv[4], v[4];
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];" : "=r"(pv[0]), "=r"(pv[1]), "=r"(pv[2]), "=r"(pv[3]) : "r"(shared_address(basis + p)) : "memory");
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];" : "=r"(mv[0]), "=r"(mv[1]), "=r"(mv[2]), "=r"(mv[3]) : "r"(shared_address(basis + m)) : "memory");
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = pv[j] + 0x40404040u - mv[j];
    if (fragment == M / 16 - 1 && lane / 4 == 7 && tile < 2) {
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        lo |= uint32_t(64 - int(signal[size_t(M - 1) * K + tile * 32 + (lane % 4) * 4 + j])) << (8 * j);
        hi |= uint32_t(64 - int(signal[size_t(M - 1) * K + tile * 32 + 16 + (lane % 4) * 4 + j])) << (8 * j);
      }
      v[1] -= lo;
      v[3] -= hi;
    }
    output[(size_t(fragment) * (K / 32) + tile) * 32 + lane] = B ? make_uint4(v[0], v[2], v[1], v[3]) : make_uint4(v[0], v[1], v[2], v[3]);
  }
}
} // namespace support

using namespace support;

extern "C" int pm_merkle_init(const void *key, void *tree, int k, void *stream) {
  if (k < 2048 || k > 65536 || (k & (k - 1))) return cudaErrorInvalidValue;
  int nleaves = 32 * k;
  auto s = (cudaStream_t)stream;
  leaves<<<nleaves / 128, 128, 0, s>>>((const uint32_t *)key, (uint32_t *)tree, nleaves);
  for (int count = nleaves / 2; count >= 2; count >>= 1) parents<<<(count + 127) / 128, 128, 0, s>>>((const uint32_t *)key, (uint32_t *)tree, count, count);
  return cudaGetLastError();
}
extern "C" int pm_merkle_root(const void *a, const void *key, const void *siblings, void *root, int k, void *stream) {
  if (k < 2048 || k > 65536 || (k & (k - 1))) return cudaErrorInvalidValue;
  int nleaves = 32 * k, nonce_leaf = (M - 1) * k / 1024;
  update<<<1, 32, 0, (cudaStream_t)stream>>>((const uint32_t *)a, (const uint32_t *)key, (const uint32_t *)siblings, (uint32_t *)root, nleaves, nonce_leaf);
  return cudaGetLastError();
}
extern "C" int pm_advance_nonce(void *a, void *counter, int k, void *stream) {
  if (k < 2048 || k > 65536 || k % 128) return cudaErrorInvalidValue;
  advance_nonce<<<1, 64, 0, (cudaStream_t)stream>>>((int8_t *)a, (unsigned long long *)counter, k);
  return cudaGetLastError();
}
extern "C" int pm_build_indices(const void *in, void *packed, int k, void *stream) {
  if (k < 2048 || k > 65536 || k % 128) return cudaErrorInvalidValue;
  build_indices<<<(k + 127) / 128, 128, 0, (cudaStream_t)stream>>>((const int8_t *)in, (uint32_t *)packed, k);
  return cudaGetLastError();
}
extern "C" int pm_pack_basis(const void *in, void *out, void *stream) {
  pack_basis_frag<<<M * 128 / 256, 256, 0, (cudaStream_t)stream>>>((const int8_t *)in, (uint8_t *)out);
  return cudaGetLastError();
}
extern "C" int pm_prepare_packed(const void *signal, const void *dense, const void *permutation, void *pairs, void *output, int k, int is_b, void *stream) {
  if (k != 4096) return cudaErrorInvalidValue;
  auto s = (cudaStream_t)stream;
  pack_pairs<<<k / 4, 128, 0, s>>>((const int8_t *)permutation, (uint16_t *)pairs, k);
  auto sg = (const int8_t *)signal;
  auto dn = (const int8_t *)dense;
  auto pr = (const uint16_t *)pairs;
  auto out = (uint4 *)output;
  if (is_b) materialize_packed<4096, true><<<M / 16, 256, 0, s>>>(sg, dn, pr, out);
  else materialize_packed<4096, false><<<M / 16, 256, 0, s>>>(sg, dn, pr, out);
  return cudaGetLastError();
}
// Commitment hashes key_A/key_B from both Merkle roots; certificate 3 salts with (m, n).
extern "C" int pm_commit(const void *ra, const void *rb, const void *key, void *ka, void *kb, int cert, int m, int n, void *stream) {
  using Kernel = pearl::CommitmentHashFromMerkleRootsKernel;
  Kernel::Arguments args{(const uint8_t *)ra, (const uint8_t *)rb, (const uint8_t *)key, (uint8_t *)ka, (uint8_t *)kb, nullptr, nullptr, cert == 3, (uint32_t)m, (uint32_t)n};
  auto params = Kernel::to_underlying_arguments(args);
  cutlass::device_kernel<Kernel><<<1, 1, 0, (cudaStream_t)stream>>>(params);
  return cudaGetLastError();
}
// Noise factors, R = 128: which&1 -> EAL (m x R), EAR (k x R) from key_A; which&2 -> EBL (k x R), EBR (n x R) from key_B.
extern "C" int pm_noise(void *al, void *ar, void *bl, void *br, const void *ka, const void *kb, int m, int n, int k, int which, void *stream) {
  using Kernel = pearl::NoiseGenerationKernel<128, 64>;
  Kernel::Arguments args{(which & 1) ? (int8_t *)al : nullptr, nullptr, (which & 1) ? (int8_t *)ar : nullptr, nullptr, (which & 2) ? (int8_t *)bl : nullptr, nullptr,
                         (which & 2) ? (int8_t *)br : nullptr, nullptr, (which & 1) ? m : 0, (which & 1) ? k : 0, (which & 2) ? k : 0, (which & 2) ? n : 0,
                         (const uint8_t *)ka, (const uint8_t *)kb, nullptr, 0};
  auto params = Kernel::to_underlying_arguments(args);
  cutlass::device_kernel<Kernel><<<Kernel::get_grid_shape(params), Kernel::get_block_shape(), Kernel::SharedStorageSize, (cudaStream_t)stream>>>(params);
  return cudaGetLastError();
}
