#pragma once

#include "cutlass/arch/mma_sm90.h"
#include "cutlass/numeric_types.h"

#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"

namespace cudabox::gemm::sm90_pipelined_tma_mma {

template <class T> struct gemm_traits;

// ---------------------------------------------------------------------------
// fp16: F32 accumulator, F16 inputs, F16 output.
// ---------------------------------------------------------------------------
template <> struct gemm_traits<cute::half_t> {
  using Element = cute::half_t;
  using ElementAccumulator = float;

  // CTA tile: each CTA computes BlockM x BlockN, stepping by BlockK.
  using BlockM = cute::Int<256>;
  using BlockN = cute::Int<192>;
  using BlockK = cute::Int<128>;

  // Number of pipeline stages (smem A/B replicated this many times).
  using PipelineStages = cute::Int<2>;

  // Thread-block cluster shape (M, N, K). N>1 here multicasts A across CTAs.
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  // WGMMA atom: F32 += F16 * F16, K-major operands, SS = both A & B in SMEM.
  using MmaAtom =
      cute::SM90::GMMA::MMA_64x192x16_F32F16F16_SS<cute::SM90::GMMA::Major::K,
                                                   cute::SM90::GMMA::Major::K>;
  // Tile MmaAtom across the CTA: 2 atoms in M, 1 in N, 1 in K.
  using AtomLayoutMNK = cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>;

  // Swizzled SMEM layouts for A, B, and C tiles.
  using SmemAtomAB = decltype(cute::GMMA::Layout_K_SW128_Atom<Element>{});
  using SmemAtomC = decltype(cute::GMMA::Layout_K_SW128_Atom<Element>{});

  // Epilogue: RMEM (accumulator) -> SMEM (C tile). STSM is 16-bit-only.
  using SmemCopyAtomC = cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
};

// ---------------------------------------------------------------------------
// bf16: F32 accumulator, BF16 inputs, BF16 output.
// ---------------------------------------------------------------------------
template <> struct gemm_traits<cute::bfloat16_t> {
  using Element = cute::bfloat16_t;
  using ElementAccumulator = float;

  // CTA tile: each CTA computes BlockM x BlockN, stepping by BlockK.
  using BlockM = cute::Int<256>;
  using BlockN = cute::Int<192>;
  using BlockK = cute::Int<128>;

  // Number of pipeline stages (smem A/B replicated this many times).
  using PipelineStages = cute::Int<2>;

  // Thread-block cluster shape (M, N, K). N>1 here multicasts A across CTAs.
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  // WGMMA atom: F32 += F16 * F16, K-major operands, SS = both A & B in SMEM.
  using MmaAtom = cute::SM90::GMMA::MMA_64x192x16_F32BF16BF16_SS<
      cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>;
  // Tile MmaAtom across the CTA: 2 atoms in M, 1 in N, 1 in K.
  using AtomLayoutMNK = cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>;

  // Swizzled SMEM layouts for A, B, and C tiles.
  using SmemAtomAB = decltype(cute::GMMA::Layout_K_SW128_Atom<Element>{});
  using SmemAtomC = decltype(cute::GMMA::Layout_K_SW128_Atom<Element>{});

  // Epilogue: RMEM (accumulator) -> SMEM (C tile). STSM is 16-bit-only.
  using SmemCopyAtomC = cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
};

} // namespace cudabox::gemm::sm90_pipelined_tma_mma
