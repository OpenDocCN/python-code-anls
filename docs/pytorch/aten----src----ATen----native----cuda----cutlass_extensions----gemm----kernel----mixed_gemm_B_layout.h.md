# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\kernel\mixed_gemm_B_layout.h`

```
/*
  This file exists so that we use the same weight layout for MoE grouped gemm and regular gemm when the weight is
  quantized. The preprocessing code reads this template to know how to organize the quantized weight matrices
  to be consumed by CUTLASS.

  Note that for int4, ThreadBlockK MUST be 64.
 */

#pragma once

#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/platform/platform.h>

#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>
#include <ATen/native/cuda/cutlass_extensions/tile_interleaved_layout.h>

namespace cutlass {
namespace gemm {
namespace kernel {

// Template struct defining layout details for matrix B depending on TypeB and Arch
template<typename TypeB, typename Arch, typename Enable = void>
struct LayoutDetailsB {
};

// Specialization for TypeB and architecture Sm70 (Volta)
template<typename TypeB>
struct LayoutDetailsB<TypeB, arch::Sm70> {
    static constexpr int ThreadblockK      = 64;  // Defines the size of ThreadBlockK for this specialization
    using Layout                           = layout::RowMajor;  // Specifies the matrix layout as RowMajor
    static constexpr int ElementsPerAccess = 8;   // Number of elements accessed per memory access
    using Operator                         = cutlass::arch::OpMultiplyAdd;  // Specifies the operator type
};

// Specializations for FP16 TypeB and architecture Arch with compute capability >= 75 (Turing+)
template<typename Arch>
struct LayoutDetailsB<half_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK      = 64;  // ThreadBlockK size for FP16
    using Layout                           = layout::RowMajor;  // Row-major layout for FP16
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<half_t>::value;  // Elements accessed per memory access
    using Operator                         = cutlass::arch::OpMultiplyAdd;  // Operator type
};

// Specializations for bfloat16 TypeB and architecture Arch with compute capability >= 75 (Turing+)
template<typename Arch>
struct LayoutDetailsB<bfloat16_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK      = 64;  // ThreadBlockK size for bfloat16
    using Layout                           = layout::RowMajor;  // Row-major layout for bfloat16
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<bfloat16_t>::value;  // Elements accessed per memory access
    using Operator                         = cutlass::arch::OpMultiplyAdd;  // Operator type
};

// Specializations for uint8_t TypeB and architecture Arch with compute capability >= 75 (Turing+)
template<typename Arch>
struct LayoutDetailsB<uint8_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK = 64;  // ThreadBlockK size for uint8_t

private:
    static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint8_t>::value;  // Elements per cache line
    static constexpr int ColumnsInterleaved   = ElementsPerCacheLine / ThreadblockK;  // Columns interleaved calculation

public:
    using Layout                           = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;  // Column-major interleaved layout
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint8_t>::value;  // Elements accessed per memory access
    using Operator                         = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;  // Operator for dequantizing after loading
};
    # 使用别名 Operator 表示 cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA
    using Operator = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
};

// 模板特化：当满足条件 Arch::kMinComputeCapability >= 75 时，使用该特化版本
template<typename Arch>
struct LayoutDetailsB<uint4b_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    // 定义 ThreadblockK 值为 64
    static constexpr int ThreadblockK = 64;

private:
    // 计算每个缓存行可以容纳的元素数
    static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint4b_t>::value;
    // 计算每个 Threadblock 中交错的列数
    static constexpr int ColumnsInterleaved   = ElementsPerCacheLine / ThreadblockK;

public:
    // 使用 ColumnMajorTileInterleave 布局，其中 ThreadblockK 和 ColumnsInterleaved 是参数
    using Layout                           = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
    // 计算每次访问可以处理的元素数
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint4b_t>::value;
    // 使用 OpMultiplyAddDequantizeInterleavedBToA 作为操作器
    using Operator                         = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
```