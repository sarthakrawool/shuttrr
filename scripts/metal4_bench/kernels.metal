// Shutter metal4_bench kernels. Compiled at runtime by main.swift with makeLibrary(source:),
// three times, each with one of these macros prepended:
//   BENCH_SIMD      (a) plain simdgroup_matrix GEMM, MSL 3.1, no tensor API
//   BENCH_MPP       (b)+(c) Metal 4 TensorOps (mpp::tensor_ops::matmul2d), MSL 4.0, fp16 / bf16
//   BENCH_MPP_INT8  (c) TensorOps with int8 x int8 -> int32 (experimental)
// so that a failure in one section never hides the others.
//
// All kernels compute C[M,N] = A[M,K] @ B[K,N], all row-major, aligned shapes only
// (the host refuses shapes that are not multiples of the tile sizes). The grid is 1D;
// tile_coord() maps the threadgroup index to an output tile, in raster or Morton order.
//
// API sources (see README.md): MSL spec 4.1 ch. 2.22 + 7 (tensor, slice, matmul2d, execution
// scopes, supported dtypes), Apple's MPP Programming Guide (tile sizes, K blocking, Morton
// walk), and MLX 0.32.4's shipped NAX kernels (steel_gemm_thin_nax.h), which use exactly the
// tensor_inline / slice / cooperative-tensor calls used below and are compiled by Apple's
// toolchain in MLX CI. Nothing here has been compiled by the author (Linux, no Metal).

#include <metal_stdlib>
using namespace metal;

struct GemmParams {
  int M;        // rows of A / C handled by this dispatch
  int N;
  int K;
  int lda;      // row strides in elements
  int ldb;
  int ldc;
  int tiles_m;  // M / tileM
  int tiles_n;  // N / tileN
  int swizzle;  // 0 or 1 = raster order, S (power of two) = Morton order inside S x S blocks
  int pad;
};

// Keep the even bits of x and pack them (Morton deinterleave).
static inline uint compact1by1(uint x) {
  x &= 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

// 1D threadgroup index -> output tile. Returns false for the padding threadgroups that the
// Morton grid adds when tiles_m / tiles_n are not multiples of S (they exit immediately).
static inline bool tile_coord(uint lin, constant GemmParams& p, thread int& tm, thread int& tn) {
  if (p.swizzle <= 1) {
    tm = int(lin) / p.tiles_n;
    tn = int(lin) % p.tiles_n;
    return tm < p.tiles_m;
  }
  const uint S = uint(p.swizzle);
  const uint blocks_n = (uint(p.tiles_n) + S - 1) / S;
  const uint blk = lin / (S * S);
  const uint loc = lin % (S * S);
  tm = int((blk / blocks_n) * S + compact1by1(loc >> 1));
  tn = int((blk % blocks_n) * S + compact1by1(loc));
  return tm < p.tiles_m && tn < p.tiles_n;
}

// ===========================================================================================
// (a) Baseline: simdgroup_matrix GEMM in the style of MLX's pre-M5 "steel" kernels.
//     64x64x32 threadgroup tile staged in threadgroup memory, 2x2 simdgroups of 32x32,
//     8x8 fragments in float (inputs converted on load, as MLX steel does), fp32 accumulate.
// ===========================================================================================
#if defined(BENCH_SIMD)
#include <metal_simdgroup_matrix>

constant constexpr int SB_BM = 64;
constant constexpr int SB_BN = 64;
constant constexpr int SB_BK = 32;
constant constexpr int SB_LDA = SB_BK + 8;  // padded threadgroup row strides (elements)
constant constexpr int SB_LDB = SB_BN + 8;

// Element (row fm, cols fn, fn+1) of an 8x8 simdgroup_matrix held by this lane
// (same mapping as MLX steel/gemm/mma.h BaseMMAFrag::get_coord).
static inline short2 frag_coord(ushort lane) {
  const short qid = short(lane / 4);
  const short fm = (qid & 4) + short((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
  return short2(fn, fm);
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void gemm_simd(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup T As[SB_BM * SB_LDA];
  threadgroup T Bs[SB_BK * SB_LDB];
  int tm, tn;
  if (!tile_coord(tg, p, tm, tn)) {
    return;  // whole threadgroup exits together, before any barrier
  }

  const int row0 = tm * SB_BM;
  const int col0 = tn * SB_BN;
  const short sm = short(sg / 2);
  const short sn = short(sg % 2);
  const short2 fc = frag_coord(lane);
  const short fn = fc.x;
  const short fm = fc.y;

  simdgroup_matrix<float, 8, 8> acc[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  device const T* Ab = A + int64_t(row0) * p.lda;
  device const T* Bb = B + col0;

  for (int k0 = 0; k0 < p.K; k0 += SB_BK) {
    // A tile 64x32 and B tile 32x64, 4 consecutive elements per step, 128 threads.
    for (int i = tid; i < (SB_BM * SB_BK) / 4; i += 128) {
      const int r = i / (SB_BK / 4);
      const int c = (i % (SB_BK / 4)) * 4;
      device const T* src = Ab + int64_t(r) * p.lda + k0 + c;
      for (int e = 0; e < 4; e++) As[r * SB_LDA + c + e] = src[e];
    }
    for (int i = tid; i < (SB_BK * SB_BN) / 4; i += 128) {
      const int r = i / (SB_BN / 4);
      const int c = (i % (SB_BN / 4)) * 4;
      device const T* src = Bb + int64_t(k0 + r) * p.ldb + c;
      for (int e = 0; e < 4; e++) Bs[r * SB_LDB + c + e] = src[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk = 0; kk < SB_BK; kk += 8) {
      simdgroup_matrix<float, 8, 8> a[4];
      simdgroup_matrix<float, 8, 8> b[4];
      for (int i = 0; i < 4; i++) {
        const int r = sm * 32 + i * 8 + fm;
        a[i].thread_elements()[0] = float(As[r * SB_LDA + kk + fn]);
        a[i].thread_elements()[1] = float(As[r * SB_LDA + kk + fn + 1]);
      }
      for (int j = 0; j < 4; j++) {
        const int c = sn * 32 + j * 8 + fn;
        b[j].thread_elements()[0] = float(Bs[(kk + fm) * SB_LDB + c]);
        b[j].thread_elements()[1] = float(Bs[(kk + fm) * SB_LDB + c + 1]);
      }
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      const int r = row0 + sm * 32 + i * 8 + fm;
      const int c = col0 + sn * 32 + j * 8 + fn;
      C[int64_t(r) * p.ldc + c] = T(acc[i][j].thread_elements()[0]);
      C[int64_t(r) * p.ldc + c + 1] = T(acc[i][j].thread_elements()[1]);
    }
  }
}

template [[host_name("gemm_simd_half")]] [[kernel]] decltype(gemm_simd<half>) gemm_simd<half>;
template [[host_name("gemm_simd_bfloat")]] [[kernel]] decltype(gemm_simd<bfloat>) gemm_simd<bfloat>;
#endif  // BENCH_SIMD

// ===========================================================================================
// Metal 4 TensorOps (Metal Performance Primitives)
// ===========================================================================================
#if defined(BENCH_MPP) || defined(BENCH_MPP_INT8)
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using extents_t = metal::dextents<int32_t, 2>;
using strides_t = metal::array<int32_t, 2>;
// Note on index order (MSL spec / MPP guide): extents are listed innermost first, i.e.
// (columns, rows) for a row-major matrix, with strides {1, leading_dimension}.
#endif

#if defined(BENCH_MPP)
// -------------------------------------------------------------------------------------------
// (b) The basic form from the MSL 4.1 spec example (section 7.2.1) and Apple's tech talk:
//     one matmul2d per threadgroup, execution_simdgroups<4>, 64x64 output tile, K left
//     dynamic so the op loops over the whole K itself, raster threadgroup order, result
//     written straight to device memory (destination element type = T).
// -------------------------------------------------------------------------------------------
template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void mpp_basic(
    device T* A [[buffer(0)]],
    device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]]) {
  using tensor_t = metal::tensor<device T, extents_t, metal::tensor_inline>;
  int tm, tn;
  if (!tile_coord(tg, p, tm, tn)) {
    return;
  }
  tensor_t tA(A, extents_t(p.K, p.M), strides_t{1, p.lda});
  tensor_t tB(B, extents_t(p.N, p.K), strides_t{1, p.ldb});
  tensor_t tC(C, extents_t(p.N, p.M), strides_t{1, p.ldc});

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64);  // K = dynamic, mode = multiply
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

  auto mA = tA.slice(0, tm * 64);
  auto mB = tB.slice(tn * 64, 0);
  auto mC = tC.slice(tn * 64, tm * 64);
  op.run(mA, mB, mC);
}

template [[host_name("mpp_basic_half")]] [[kernel]] decltype(mpp_basic<half>) mpp_basic<half>;
template [[host_name("mpp_basic_bfloat")]] [[kernel]] decltype(mpp_basic<bfloat>) mpp_basic<bfloat>;

// -------------------------------------------------------------------------------------------
// (c) Apple's recommended structure (MPP Programming Guide sec. 2.3): one matmul2d per
//     simdgroup (execution_simdgroup), SM x SN simdgroup tile, WM x WN simdgroups per
//     threadgroup, explicit K blocking (BK) into an fp32 destination cooperative tensor
//     with a threadgroup barrier every BAR K-blocks, static slice extents (aligned shapes),
//     Morton threadgroup order chosen by the host via p.swizzle. Inputs read straight from
//     device memory (the guide: threadgroup staging is not needed). relaxed_precision = true
//     as in MLX (the spec says it only affects float operands).
// -------------------------------------------------------------------------------------------
template <typename T, int SM, int SN, int WM, int WN, int BK, int BAR>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void mpp_tuned(
    device T* A [[buffer(0)]],
    device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  using tensor_t = metal::tensor<device T, extents_t, metal::tensor_inline>;
  int tm, tn;
  if (!tile_coord(tg, p, tm, tn)) {
    return;
  }
  const int r0 = tm * (WM * SM) + int(sg / WN) * SM;
  const int c0 = tn * (WN * SN) + int(sg % WN) * SN;

  tensor_t tA(A, extents_t(p.K, p.M), strides_t{1, p.lda});
  tensor_t tB(B, extents_t(p.N, p.K), strides_t{1, p.ldb});

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      SM, SN, BK, false, false, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  using a_slice_t = decltype(tA.template slice<BK, SM>(0, 0));
  using b_slice_t = decltype(tB.template slice<SN, BK>(0, 0));
  auto acc = op.template get_destination_cooperative_tensor<a_slice_t, b_slice_t, float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++) {
    acc[i] = 0.0f;
  }

  const int nk = p.K / BK;
  for (int it = 0; it < nk; it++) {
    if constexpr (BAR > 0) {
      if ((it % BAR) == 0) {
        threadgroup_barrier(mem_flags::mem_none);
      }
    }
    const int k = it * BK;
    auto a = tA.template slice<BK, SM>(k, r0);
    auto b = tB.template slice<SN, BK>(c0, k);
    op.run(a, b, acc);
  }

  auto out = op.template get_destination_cooperative_tensor<a_slice_t, b_slice_t, T>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++) {
    out[i] = static_cast<T>(acc[i]);
  }
  tensor_t tC(C, extents_t(p.N, p.M), strides_t{1, p.ldc});
  auto d = tC.template slice<SN, SM>(c0, r0);
  out.store(d);
}

#define INST_TUNED(tname, T, SM, SN, WM, WN, BK, BAR)                                   \
  template [[host_name("mpp_tuned_" #tname "_sg" #SM "x" #SN "_w" #WM "x" #WN "_bk" #BK)]] \
  [[kernel]] decltype(mpp_tuned<T, SM, SN, WM, WN, BK, BAR>) mpp_tuned<T, SM, SN, WM, WN, BK, BAR>;

#define INST_TUNED_ALL(tname, T)                 \
  INST_TUNED(tname, T, 32, 32, 2, 2, 64, 2)      \
  INST_TUNED(tname, T, 32, 32, 2, 2, 128, 1)     \
  INST_TUNED(tname, T, 32, 32, 4, 4, 64, 2)      \
  INST_TUNED(tname, T, 32, 32, 4, 4, 128, 1)     \
  INST_TUNED(tname, T, 64, 64, 2, 2, 64, 2)      \
  INST_TUNED(tname, T, 64, 64, 2, 2, 128, 1)

INST_TUNED_ALL(half, half)
INST_TUNED_ALL(bfloat, bfloat)
#endif  // BENCH_MPP

#if defined(BENCH_MPP_INT8)
// -------------------------------------------------------------------------------------------
// (c) int8: char x char -> int (MSL spec 4.1 Table 7.3, "Metal 4"). Same structure as
//     mpp_tuned, int32 accumulator stored directly (C buffer is int32). EXPERIMENTAL.
// -------------------------------------------------------------------------------------------
template <int SM, int SN, int WM, int WN, int BK, int BAR>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void mpp_i8(
    device char* A [[buffer(0)]],
    device char* B [[buffer(1)]],
    device int* C [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  using tensor_i8 = metal::tensor<device char, extents_t, metal::tensor_inline>;
  using tensor_i32 = metal::tensor<device int, extents_t, metal::tensor_inline>;
  int tm, tn;
  if (!tile_coord(tg, p, tm, tn)) {
    return;
  }
  const int r0 = tm * (WM * SM) + int(sg / WN) * SM;
  const int c0 = tn * (WN * SN) + int(sg % WN) * SN;

  tensor_i8 tA(A, extents_t(p.K, p.M), strides_t{1, p.lda});
  tensor_i8 tB(B, extents_t(p.N, p.K), strides_t{1, p.ldb});

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      SM, SN, BK, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  using a_slice_t = decltype(tA.template slice<BK, SM>(0, 0));
  using b_slice_t = decltype(tB.template slice<SN, BK>(0, 0));
  auto acc = op.template get_destination_cooperative_tensor<a_slice_t, b_slice_t, int>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++) {
    acc[i] = 0;
  }
  const int nk = p.K / BK;
  for (int it = 0; it < nk; it++) {
    if constexpr (BAR > 0) {
      if ((it % BAR) == 0) {
        threadgroup_barrier(mem_flags::mem_none);
      }
    }
    const int k = it * BK;
    auto a = tA.template slice<BK, SM>(k, r0);
    auto b = tB.template slice<SN, BK>(c0, k);
    op.run(a, b, acc);
  }
  tensor_i32 tC(C, extents_t(p.N, p.M), strides_t{1, p.ldc});
  auto d = tC.template slice<SN, SM>(c0, r0);
  acc.store(d);
}

#define INST_I8(SM, SN, WM, WN, BK, BAR)                                          \
  template [[host_name("mpp_i8_sg" #SM "x" #SN "_w" #WM "x" #WN "_bk" #BK)]] [[kernel]] \
  decltype(mpp_i8<SM, SN, WM, WN, BK, BAR>) mpp_i8<SM, SN, WM, WN, BK, BAR>;

INST_I8(32, 32, 2, 2, 128, 1)
INST_I8(32, 32, 4, 4, 128, 1)
INST_I8(64, 64, 2, 2, 128, 1)
#endif  // BENCH_MPP_INT8
