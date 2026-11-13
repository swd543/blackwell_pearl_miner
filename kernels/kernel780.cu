// SPDX-License-Identifier: ISC
// Mining kernel on the packed rank-128 basis.
// CTA tile 160x256 (8 warps, warp tile 80x64), one barrier per K32 step. Operands are rebuilt
// from the basis in shared memory; each step stores the next stage while issuing MMAs.
#include <cuda_runtime.h>
#include <cstdint>
#include <utility>
#include "gemm/pow_utils.hpp"

namespace k780 {
constexpr int P0 = 5, P1 = 10, P2 = 12, P3 = 14, P4 = 25, P5 = 27, P6 = 29, P7 = 31, PL = 20;
constexpr int BM = 160, BN = 256, WM = 80, WN = 64, MI = WM / 16, NJ = WN / 8, NT = 256;
constexpr int FA = BM / 16, FB = BN / 16, FR = FA + FB, RQ = (FR + 7) / 8;
constexpr int BASA = FA * 2048, BASB = FB * 2048, STG = FR * 512, TRACE = NT * 16 * 4;
constexpr int OFF_BA = 0, OFF_BB = BASA, OFF_STG = BASA + BASB, OFF_TR = OFF_STG + 2 * STG, OFF_BAR = OFF_TR + TRACE;
constexpr int SHARED = OFF_BAR + 16;
static_assert(SHARED <= 101376 - 1024);
static_assert(P0 < P1 && P2 >= 2 * MI && P4 >= 4 * MI && RQ == 4);

// Pair offsets per K column: x = A (plus*16 | minus*16<<16), y = B.
using Pair = uint2;

struct Params {
  const uint8_t *basA, *basB;
  const Pair *pairs;
  const uint32_t *corr;
  int M, N, K;
  const uint32_t *key, *target;
  uint32_t *count, *ids;
  int cap;
};

__device__ Pair g_pairs[65536];
__device__ uint32_t g_zero[16];

__device__ __forceinline__ Pair ldpair(const Pair *p) {
  uint2 v;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned sa(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
#define LDSM(a, r) asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory")
#define LDS4(a, r0, r1, r2, r3) asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory")
#define PLDSM(pv, a, r) asm volatile("{.reg .pred pp; setp.ne.u32 pp, %5, 0; @pp ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];}" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]) : "r"(a), "r"(pv) : "memory")
#define PSTS4(pv, a, r0, r1, r2, r3) asm volatile("{.reg .pred pp; setp.ne.u32 pp, %5, 0; @pp st.shared.v4.u32 [%0], {%1,%2,%3,%4};}" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(pv) : "memory")
#define PLDS4(pv, a, r0, r1, r2, r3) asm volatile("{.reg .pred pp; setp.ne.u32 pp, %5, 0; @pp ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];}" : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3) : "r"(a), "r"(pv) : "memory")
#define STS4(a, r0, r1, r2, r3) asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory")
__device__ __forceinline__ void mma(int32_t *c, const uint32_t *a, const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The kernel body is a separate force-inlined function (this shapes the generated code).
// ptxas schedules this loop very sensitively: the tile-copy lambda, the pair-load lambda and the local copies of
// threadIdx.x and p.pairs below are part of the measured layout (-2.6 % cycles against 1b51df4), keep them.
__device__ __forceinline__ void body(const Params &p);
__global__ __launch_bounds__(NT, 1) void kernel(const __grid_constant__ Params p) { body(p); }
__device__ __forceinline__ void body(const Params &p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t tidv = threadIdx.x;
  const int lane = tidv & 31, warp = tidv >> 5, wr = warp / (BN / WN), wc = warp % (BN / WN);
  const Pair *pbase = p.pairs;
  const int tilesN = p.N / BN, tiles = (p.M / BM) * tilesN, S = p.K / 32;
  const unsigned bar = sa(sm + OFF_BAR), stg0 = sa(sm + OFF_STG);
  uint32_t *trace = (uint32_t *)(sm + OFF_TR);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int lastfrag = p.M / 16 - 1;
  const uint32_t cmask = (lane / 4 == 7) ? 0xffffffffu : 0u;
  // Rebuild roles: q0 own A fragment, q1/q2 own B fragment pairs, q3 extra A row (warps 0 and 5).
  const int fOwnA = wr * MI + wc, fExtra = wr * MI + 4;
  const unsigned xq = (warp == 0 || warp == 5) ? 1u : 0u, nxq = xq ^ 1u;
  unsigned rbase[RQ], roff[RQ];
#pragma unroll
  for (int q = 0; q < RQ; ++q) {
    const int f = q == 0 ? fOwnA : q == 1 ? FA + wc * (NJ / 2) + wr * 2 : q == 2 ? FA + wc * (NJ / 2) + wr * 2 + 1 : ((warp == 0 || warp == 5) ? fExtra : FR);
    rbase[q] = f < FA ? sa(sm + OFF_BA) + f * 2048 : sa(sm + OFF_BB) + (f < FR ? f - FA : 0) * 2048;
    roff[q] = f * 512 + lane * 16;
  }
  // Pairs of K step idx for this lane.
  auto ldp = [&](int idx) -> Pair {
    return ldpair(pbase + idx * 32 + lane);
  };
  unsigned aoff[MI], boff[NJ / 2];
#define RB(q) ((q) == 2 ? rbase[1] + 2048u : rbase[q])
#define RO(q) ((q) == 2 ? roff[1] + 512u : roff[q])
#define BO(jp) (((jp) & 1) ? boff[(jp) - 1] + 512u : boff[jp])
#pragma unroll
  for (int k = 0; k < MI; ++k) aoff[k] = (wr * MI + (k == 0 ? wc : ((k - 1) < wc ? k - 1 : k))) * 512 + lane * 16;
#pragma unroll
  for (int jp = 0; jp < NJ / 2; ++jp) boff[jp] = (FA + wc * (NJ / 2) + (jp < 2 ? wr * 2 + jp : (1 - wr) * 2 + jp - 2)) * 512 + lane * 16;
  unsigned phase = 0;
  auto issue = [&](int t) {
    const int row_ = (t / tilesN) * BM, col_ = (t % tilesN) * BN;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(BASA + BASB) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(sm + OFF_BA)), "l"(p.basA + size_t(row_ / 16) * 2048), "r"(BASA), "r"(bar) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(sm + OFF_BB)), "l"(p.basB + size_t(col_ / 16) * 2048), "r"(BASB), "r"(bar) : "memory");
  };
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int bm = tile / tilesN, bn = tile % tilesN, row = bm * BM;
    __syncthreads();
    if (threadIdx.x == 0) issue(tile);
    asm volatile("{ .reg .pred q; L: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra L; }" ::"r"(bar), "r"(phase) : "memory");
    phase ^= 1;
    const bool hasNonce = (row / 16 + FA - 1) >= lastfrag;
    int32_t acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][j][c] = 0;
    uint32_t rr[2][2][4];
    uint32_t ov[4];
    auto ldsm2 = [&](Pair pr, int q, int slot) {
      const unsigned pw = (q == 1 || q == 2) ? pr.y : pr.x;
      const unsigned po = __byte_perm(pw, 0, 0x4410), mo = pw >> 16;
      if (q == 3) {
        PLDSM(xq, RB(q) + po, rr[slot][0]);
        PLDSM(xq, RB(q) + mo, rr[slot][1]);
      } else {
        LDSM(RB(q) + po, rr[slot][0]);
        LDSM(RB(q) + mo, rr[slot][1]);
      }
    };
    auto sts2 = [&](unsigned stg, int q, int slot, int s, bool corr) {
      uint32_t a0 = rr[slot][0][0] + 0x40404040u - rr[slot][1][0], a1 = rr[slot][0][1] + 0x40404040u - rr[slot][1][1],
               a2 = rr[slot][0][2] + 0x40404040u - rr[slot][1][2], a3 = rr[slot][0][3] + 0x40404040u - rr[slot][1][3];
      const bool isA = (q == 0 || q == 3);
      const int fq = q == 0 ? fOwnA : fExtra;
      if (corr && hasNonce && isA && row / 16 + fq == lastfrag) {
        a1 -= cmask & p.corr[(s * 4 + lane % 4) * 2];
        a3 -= cmask & p.corr[(s * 4 + lane % 4) * 2 + 1];
      }
      ov[0] = a0; ov[1] = a1; ov[2] = a2; ov[3] = a3;
      if (q == 3) PSTS4(xq, stg + RO(q), a0, a1, a2, a3);
      else if (!isA) STS4(stg + RO(q), a0, a2, a1, a3);
      else STS4(stg + RO(q), a0, a1, a2, a3);
    };
    uint32_t af[2][MI][4], bf[NJ - 1][2], b7[2][2];
    // Prologue: stage 0 stored; q0, q1 of stage 1 loaded.
    {
      Pair pr = __ldg(pbase + lane);
#pragma unroll
      for (int q = 0; q < RQ; q += 2) {
        ldsm2(pr, q, 0);
        ldsm2(pr, q + 1, 1);
        sts2(stg0, q, 0, 0, true);
        sts2(stg0, q + 1, 1, 0, true);
      }
    }
    Pair prC = __ldg(pbase + 32 + lane);
    ldsm2(prC, 0, 0);
    ldsm2(prC, 1, 1);
    Pair prN = __ldg(pbase + 64 + lane), prNN = __ldg(pbase + ((S > 3 ? 96 : 0) + lane)), prL = prNN;
    asm volatile("bar.sync 0;" ::: "memory");
#pragma unroll
    for (int k = 0; k < MI; ++k) LDS4(stg0 + aoff[k], af[0][k][0], af[0][k][1], af[0][k][2], af[0][k][3]);
#pragma unroll
    for (int jp = 0; jp < NJ / 2 - 1; ++jp) LDS4(stg0 + BO(jp), bf[2 * jp][0], bf[2 * jp][1], bf[2 * jp + 1][0], bf[2 * jp + 1][1]);
    LDS4(stg0 + BO(NJ / 2 - 1), bf[NJ - 2][0], bf[NJ - 2][1], b7[0][0], b7[0][1]);
    uint32_t hacc = 0;
    // FOLD: last step of a 4-step group; FZ: streamed fold of the previous group.
    auto step = [&]<int PAR, int FOLD = 2, int FZ = 0>(const int s, const unsigned nxt, const bool doStore, const bool doIssue, const bool doLoad, const bool corr) {
      constexpr int Q = PAR, O = PAR ^ 1;
#pragma unroll
      for (int pp = 0; pp < (NJ - 1) * MI; ++pp) {
        const int j = pp / MI, i = pp % MI;
        if (FZ) hacc ^= uint32_t(acc[i][j][0]) ^ uint32_t(acc[i][j][1]) ^ uint32_t(acc[i][j][2]) ^ uint32_t(acc[i][j][3]);
        mma(acc[i][j], af[Q][i], bf[j]);
        if (doStore) {
          if (pp == P0) { sts2(nxt, 0, 0, s + 1, corr); af[O][0][0] = ov[0]; af[O][0][1] = ov[1]; af[O][0][2] = ov[2]; af[O][0][3] = ov[3]; }
          if (pp == P1) ldsm2(prC, 2, 0);
          if (pp == P2) { sts2(nxt, 1, 1, s + 1, corr); bf[0][0] = ov[0]; bf[0][1] = ov[2]; bf[1][0] = ov[1]; bf[1][1] = ov[3]; }
          if (pp == P3) ldsm2(prC, 3, 1);
          if (pp == P4) { sts2(nxt, 2, 0, s + 1, corr); bf[2][0] = ov[0]; bf[2][1] = ov[2]; bf[3][0] = ov[1]; bf[3][1] = ov[3]; }
          if (pp == P5) { sts2(nxt, 3, 1, s + 1, corr); af[O][MI - 1][0] = ov[0]; af[O][MI - 1][1] = ov[1]; af[O][MI - 1][2] = ov[2]; af[O][MI - 1][3] = ov[3]; }
        }
        if (pp == PL) prL = ldp((s + 4) & (S - 1));
        if (doIssue) {
          if (pp == P6) ldsm2(prN, 0, 0);
          if (pp == P7) ldsm2(prN, 1, 1);
        }
      }
      prC = prN;
      prN = prNN;
      prNN = prL;
      asm volatile("bar.sync 0;" ::: "memory");
      // Deferred column 7 with the current A set; loads of the next stage.
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        if (FZ) hacc ^= uint32_t(acc[i][NJ - 1][0]) ^ uint32_t(acc[i][NJ - 1][1]) ^ uint32_t(acc[i][NJ - 1][2]) ^ uint32_t(acc[i][NJ - 1][3]);
        mma(acc[i][NJ - 1], af[Q][i], b7[Q]);
        if (doLoad) {
          const int k = i + 1;
          if (k < MI) {
            if (k == MI - 1) PLDS4(nxq, nxt + aoff[k], af[O][k][0], af[O][k][1], af[O][k][2], af[O][k][3]);
            else LDS4(nxt + aoff[k], af[O][k][0], af[O][k][1], af[O][k][2], af[O][k][3]);
          } else {
            LDS4(nxt + BO(2), bf[4][0], bf[4][1], bf[5][0], bf[5][1]);
            LDS4(nxt + BO(3), bf[6][0], bf[6][1], b7[O][0], b7[O][1]);
          }
        }
      }
      if (FZ) {
        const int stp = (s >> 2) - 1, slot = stp & 15;
        trace[slot * NT + threadIdx.x] = stp < 16 ? hacc : pearl::rotl_xor<13>(trace[slot * NT + threadIdx.x], hacc);
        hacc = 0;
      }
      if (FOLD == 2 ? ((s & 3) == 3) : (FOLD == 1 && s == S - 1)) {
        uint32_t hx[NJ];
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          uint32_t v[MI * 4];
#pragma unroll
          for (int i = 0; i < MI; ++i) { v[i * 4] = acc[i][j][0]; v[i * 4 + 1] = acc[i][j][1]; v[i * 4 + 2] = acc[i][j][2]; v[i * 4 + 3] = acc[i][j][3]; }
          uint32_t t;
          asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(t) : "r"(v[0]), "r"(v[1]), "r"(v[2]));
#pragma unroll
          for (int k = 3; k + 1 < MI * 4; k += 2) asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(t) : "r"(t), "r"(v[k]), "r"(v[k + 1]));
          if ((MI * 4 - 3) % 2) asm volatile("xor.b32 %0, %1, %2;" : "=r"(t) : "r"(t), "r"(v[MI * 4 - 1]));
          hx[j] = t;
        }
        uint32_t h01, h23, h45, h67, h;
        asm volatile("xor.b32 %0, %1, %2;" : "=r"(h01) : "r"(hx[0]), "r"(hx[1]));
        asm volatile("xor.b32 %0, %1, %2;" : "=r"(h23) : "r"(hx[2]), "r"(hx[3]));
        asm volatile("xor.b32 %0, %1, %2;" : "=r"(h45) : "r"(hx[4]), "r"(hx[5]));
        asm volatile("xor.b32 %0, %1, %2;" : "=r"(h67) : "r"(hx[6]), "r"(hx[7]));
        asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(h) : "r"(h01), "r"(h23), "r"(h45));
        asm volatile("xor.b32 %0, %1, %2;" : "=r"(h) : "r"(h), "r"(h67));
        const int stp = s >> 2, slot = stp & 15;
        trace[slot * NT + threadIdx.x] = stp < 16 ? h : pearl::rotl_xor<13>(trace[slot * NT + threadIdx.x], h);
      }
    };
    const unsigned b0 = stg0, b1 = stg0 + STG;
    // Four steps per loop iteration. mode 0: first group, 1: steady state, 2: last group.
    // MID: steady-state group; its last step is never the last step of the tile, so it carries no final-fold code.
    auto grp = [&]<bool FIRST, bool MID, int... k>(std::integer_sequence<int, k...>, const int s0, const int mode) {
      (step.template operator()<(k & 1), ((k & 3) == 3 && !MID) ? 1 : 0, ((k & 3) == 0 && !(FIRST && k == 0)) ? 1 : 0>(
           s0 + k, (k & 1) ? b0 : b1, mode != 2 || s0 + k < S - 1, mode != 2 || s0 + k < S - 2, mode != 2 || s0 + k < S - 1, mode == 0 && k == 0),
       ...);
    };
    grp.template operator()<true, false>(std::make_integer_sequence<int, 4>{}, 0, 0);
#pragma unroll 1
    for (int s = 4; s < S - 4; s += 4) grp.template operator()<false, true>(std::make_integer_sequence<int, 4>{}, s, 1);
    grp.template operator()<false, false>(std::make_integer_sequence<int, 4>{}, S - 4, 2);
    {
      auto msg = cute::make_tensor<uint32_t>(cute::Int<16>{});
      auto digest = cute::make_tensor<uint32_t>(cute::Int<8>{});
#pragma unroll
      for (int j = 0; j < 16; ++j) msg(j) = trace[j * NT + threadIdx.x];
#pragma unroll
      for (int j = 0; j < 8; ++j) digest(j) = p.key[j];
      blake3::compress_msg_block_u32(msg, digest, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
      const uint32_t group = ((row / WM + wr) * (p.N / 64) + bn * (BN / 64) + wc) * 32 + lane;
      bool good = true;
#pragma unroll
      for (int j = 7; j >= 0; --j) {
        if (digest(j) > p.target[j]) { good = false; break; }
        if (digest(j) < p.target[j]) break;
      }
      if (good) {
        uint32_t at = atomicAdd(p.count, 1u);
        if (at < uint32_t(p.cap)) p.ids[at] = group;
      }
    }
  }
}

__global__ void build_pairs(const uint32_t *ia, const uint32_t *ib, Pair *out, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= k) return;
  out[i] = make_uint2(ia[i], ib[i]);
}

// Bank-conflict-aware column order. Within each 128-column K block the order of the columns is free
// (the hash folds prefix sums every 128 K), so the result is bit-identical. ldmatrix serves 8 lanes
// (= 8 K columns) per phase; each lane reads the 16-byte row of one basis component, whose bank
// group is (component mod 8). Four address streams share the column order: A plus, A minus,
// B plus, B minus, issued 10, 10, 16 and 16 times per CTA step. One warp per K block assigns the
// 128 columns greedily to 16 groups of 8 so that the weighted number of extra wavefronts
// (largest colour multiplicity - 1 per stream) grows least. Removes ~45 % of the bank conflicts.
__device__ __forceinline__ unsigned maxnib(unsigned h) {
  unsigned m = __vmaxu4(h & 0x0f0f0f0fu, (h >> 4) & 0x0f0f0f0fu);
  m = __vmaxu4(m, m >> 16);
  m = __vmaxu4(m, m >> 8);
  return m & 0xffu;
}
__global__ __launch_bounds__(128) void reorder_pairs(Pair *pairs, int k) {
  constexpr int W[4] = {10, 10, 16, 16};
  __shared__ Pair sp[4][128];
  __shared__ uint32_t scol[4][128];
  __shared__ uint8_t sgrp[4][128];
  __shared__ uint32_t sh[4][16][4];
  __shared__ int ssize[4][16];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, blk = blockIdx.x * 4 + w;
  if (blk * 128 >= k) return;
  Pair *g = pairs + size_t(blk) * 128, *pr = sp[w];
  uint32_t *col = scol[w], (*hist)[4] = sh[w];
  uint8_t *grp = sgrp[w];
  int *size = ssize[w];
  for (int c = lane; c < 128; c += 32) {
    const Pair v = g[c];
    pr[c] = v;
    col[c] = ((v.x >> 4) & 7) | (((v.x >> 20) & 7) << 8) | (((v.y >> 4) & 7) << 16) | (((v.y >> 20) & 7) << 24);
  }
  if (lane < 16) {
    size[lane] = 0;
    hist[lane][0] = hist[lane][1] = hist[lane][2] = hist[lane][3] = 0;
  }
  __syncwarp();
  for (int c = 0; c < 128; ++c) {
    const uint32_t cc = col[c];
    unsigned key = 0xffffffffu;
    if (lane < 16 && size[lane] < 8) {
      int delta = 0;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const unsigned h = hist[lane][s], cnt = (h >> (4 * ((cc >> (8 * s)) & 7))) & 15, mx = maxnib(h);
        if (cnt + 1 > (mx > 1 ? mx : 1)) delta += W[s];
      }
      key = (unsigned(delta) << 8) | (unsigned(size[lane]) << 4) | unsigned(lane);
    }
    key = __reduce_min_sync(0xffffffffu, key);
    if (lane == int(key & 15)) {
#pragma unroll
      for (int s = 0; s < 4; ++s) hist[lane][s] += 1u << (4 * ((cc >> (8 * s)) & 7));
      size[lane]++;
      grp[c] = uint8_t(lane);
    }
    __syncwarp();
  }
  // Group q takes positions 8q..8q+7 of the block (step q/4, lanes 8*(q%4)..+7).
  for (int c = lane; c < 128; c += 32) {
    const int q = grp[c];
    int pos = 0;
    for (int e = 0; e < c; ++e) pos += grp[e] == q;
    g[q * 8 + pos] = pr[c];
  }
}
} // namespace k780

// Packed basis (pack_basis_frag) for A and B, packed pair indices for A and B.
// Covers the first M/160*160 rows; the nonce row is never sampled.
extern "C" int pm_k780_launch(const void *basA, const void *basB, const void *ia, const void *ib, int m, int n, int k,
                              const void *key, const void *target, void *count, void *ids, int cap, void *stream) {
  using namespace k780;
  if (m % 16 || n % BN || k % 128 || k < 256 || k > 65536) return cudaErrorInvalidValue;
  Pair *pairs;
  uint32_t *zero;
  auto e = cudaGetSymbolAddress((void **)&pairs, g_pairs);
  if (e) return e;
  e = cudaGetSymbolAddress((void **)&zero, g_zero);
  if (e) return e;
  auto s = (cudaStream_t)stream;
  build_pairs<<<(k + 255) / 256, 256, 0, s>>>((const uint32_t *)ia, (const uint32_t *)ib, pairs, k);
  e = cudaGetLastError();
  if (e) return e;
  reorder_pairs<<<(k / 128 + 3) / 4, 128, 0, s>>>(pairs, k);
  e = cudaGetLastError();
  if (e) return e;
  const int mm = m / BM * BM;
  Params p{(const uint8_t *)basA, (const uint8_t *)basB, pairs, zero, mm, n, k,
           (const uint32_t *)key, (const uint32_t *)target, (uint32_t *)count, (uint32_t *)ids, cap};
  e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED);
  if (e) return e;
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (e) return e;
  kernel<<<sms, NT, SHARED, s>>>(p);
  return cudaGetLastError();
}
