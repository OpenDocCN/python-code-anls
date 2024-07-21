# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\gemm\find_default_mma.h`

```
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*! \file
    \brief Cutlass provides helper template functions to figure out the right
   data structures to instantiate to run a GEMM with various parameters (see
   `cutlass/gemm/threadblock/default_mma.h`). However, due to template
   instantiation priority rules, it will only create an MmaMultiStage with
   kStages=3 (otherwise creates an MmePipelined - which is not compatible with
   FastF32). kStages=3 uses too much shared memory and we want to use kStages=2,
   so we just copy-pasted some code from `default_mma.h` and
   `default_mma_core.h` files and wrapped this template to allow our use case.

    This is really only for the FastF32 case - aka using TensorCores with fp32.
*/

#pragma once

#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

// Template struct to find the default MMA (Matrix Multiply-accumulate) configuration
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    typename Enable_ = void>
struct FindDefaultMma {
  // Indicates whether accumulators are in row-major order
  static constexpr bool AccumulatorsInRowMajor = false;
  // Option for clearing shared memory
  static constexpr SharedMemoryClearOption SharedMemoryClear =
      SharedMemoryClearOption::kNone;

  // Alias for the default MMA configuration based on template parameters
  using DefaultMma = cutlass::gemm::threadblock::DefaultMma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      Stages,
      Operator,
      AccumulatorsInRowMajor,
      SharedMemoryClear>;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
/// Specialization for sm80 / FastF32 / multistage with kStages=2
template <
    typename ElementA_,
    /// Type of elements in matrix A
    typename LayoutA_,
    /// Layout type for A matrix operand
    int kAlignmentA,
    /// Access granularity of A matrix in units of elements
    typename ElementB_,
    /// Type of elements in matrix B
    typename LayoutB_,
    /// Layout type for B matrix operand
    int kAlignmentB,
    /// Type for accumulating matrix multiplication results
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages in multistage matrix multiply
    int kStages,
    /// Functor defining the matrix multiply operation
    typename Operator>
struct FindDefaultMma<
    ElementA_,
    LayoutA_,
    kAlignmentA,
    ElementB_,
    LayoutB_,
    kAlignmentB,
    ElementAccumulator,
    /// Specifies row-major layout for matrix C
    layout::RowMajor,
    /// Specifies tensor core operation class
    arch::OpClassTensorOp,
    /// Specifies architecture tag for SM80
    arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    kStages,
    Operator,
    /// Enable specialization if alignment of A is greater than 1
    typename cutlass::platform::enable_if<(kAlignmentA > 1)>::type> {
  
  // Type alias for row-major layout for matrix C
  using LayoutC = layout::RowMajor;
  // Type alias for tensor operation class
  using OperatorClass = arch::OpClassTensorOp;
  // Type alias for architecture tag
  using ArchTag = arch::Sm80;

  // Type alias for default matrix multiply operation
  using DefaultMma_ = cutlass::gemm::threadblock::DefaultMma<
      ElementA_,
      LayoutA_,
      kAlignmentA,
      ElementB_,
      LayoutB_,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      3, // Hardcoded number of stages in default MMA
      Operator>;

  // Struct inheriting from DefaultMma_ to define the default MMA operation
  struct DefaultMma : DefaultMma_ {
    // Alias for the core of the matrix multiply operation
    using MmaCore_ = typename DefaultMma_::MmaCore;
    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
        typename MmaCore_::Shape,
        typename DefaultMma_::IteratorA,
        typename MmaCore_::SmemIteratorA,
        MmaCore_::kCacheOpA,
        typename DefaultMma_::IteratorB,
        typename MmaCore_::SmemIteratorB,
        MmaCore_::kCacheOpB,
        ElementAccumulator,
        LayoutC,
        typename MmaCore_::MmaPolicy,
        kStages>; // Number of stages passed to multistage matrix multiply
  };
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
```