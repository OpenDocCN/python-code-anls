# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\threadblock\dq_mma_base.h`

```py
////////////////////////////////////////////////////////////////////////////////
// Copyright and license information for the CUDA GEMM template.
// SPDX-License-Identifier: BSD-3-Clause
//
// This section contains legal information regarding the use and distribution
// of the CUDA GEMM template code. It specifies conditions under which the code
// can be redistributed and used, both in source and binary forms.
////////////////////////////////////////////////////////////////////////////////

/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
    
    This file provides a template implementation for a double-buffered
    threadblock-scoped GEMM (General Matrix Multiply) kernel. It serves as a
    starting point or a template for implementing GEMM kernels on CUDA-enabled
    GPUs.
*/

#pragma once

#include <cutlass/aligned_buffer.h>   // Provides functionality for aligned memory buffers
#include <cutlass/arch/memory.h>      // Defines architecture-specific memory access methods
#include <cutlass/array.h>            // Includes functionality for array manipulation
#include <cutlass/cutlass.h>          // Main Cutlass library header
#include <cutlass/gemm/gemm.h>        // Definitions and utilities for GEMM operations
#include <cutlass/gemm/threadblock/mma_base.h>  // Defines the base class for MMA (Matrix Multiply Accumulate)
#include <cutlass/matrix_shape.h>     // Defines matrix shape and related utilities
#include <cutlass/numeric_types.h>    // Provides definitions for numeric types used in Cutlass

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////
// SFINAE trick so I can keep the same loop code for Volta and dispatch to the
// correct warp level mma. On volta, all data is stored to shared memory as FP16.
template<typename WarpMma, int kExpansionFactor = 1>


Explanation:
- The initial block of comments includes copyright and license information, detailing the terms under which the CUDA GEMM template code can be used and distributed.
- The `/*! \file ... */` comment block provides a brief overview of the file's purpose, describing it as a template for a double-buffered threadblock-scoped GEMM kernel implementation.
- `#pragma once`: Ensures that the header file is included only once during compilation to prevent multiple definitions.
- Includes several headers from the Cutlass library (`aligned_buffer.h`, `arch/memory.h`, `array.h`, `cutlass.h`, `gemm.h`, `mma_base.h`, `matrix_shape.h`, `numeric_types.h`) which provide various utilities and definitions necessary for implementing the GEMM kernel.
- Begins the namespace `cutlass::gemm::threadblock` to encapsulate the definitions related to GEMM operations within the Cutlass library.
- Defines a template `WarpMma` with an optional template parameter `kExpansionFactor`, facilitating polymorphic behavior in loop code across different GPU architectures.

This setup prepares the environment and includes necessary utilities for defining a CUDA GEMM kernel, offering both legal context and technical foundations for subsequent implementation.
// 在 CUDA 设备上执行 Warp 级别的 MMA 运算，将结果存储在 D 中
CUTLASS_DEVICE void run_warp_mma(WarpMma&                           warp_mma,
                                 typename WarpMma::FragmentC&       D,
                                 typename WarpMma::FragmentA const& A,
                                 typename WarpMma::FragmentB const& B,
                                 typename WarpMma::FragmentC const& C,
                                 const int                          warp_tileB_k_offset)
{
    // 调用 warp_mma 对象执行 MMA 运算，计算结果存储在 D 中
    warp_mma(D, A, B, C);
}

template<typename WarpMma, int kExpansionFactor = WarpMma::kExpansionFactor>
// 在 CUDA 设备上执行 Warp 级别的 MMA 运算，处理转换后的 A 和 B 片段，结果存储在 D 中
CUTLASS_DEVICE void run_warp_mma(WarpMma&                                      warp_mma,
                                 typename WarpMma::FragmentC&                  D,
                                 typename WarpMma::TransformedFragmentA const& A,
                                 typename WarpMma::TransformedFragmentB const& B,
                                 typename WarpMma::FragmentC const&            C,
                                 const int                                     warp_tileB_k_offset)
{
    // 调用 warp_mma 对象执行 MMA 运算，计算结果存储在 D 中，同时传递偏移量 warp_tileB_k_offset
    warp_mma(D, A, B, C, warp_tileB_k_offset);
}
////////////////////////////////////////////////////////////////////////////////

/// 结构体用于在 CUDA 核心和 SIMT 数学指令上计算矩阵乘积。
template<
    /// Gemm 问题的大小 - 概念：gemm::GemmShape<>
    typename Shape_,
    /// 描述调优细节的策略 (概念：MmaPolicy)
    typename Policy_,
    /// 规模类型
    typename ElementScale_,
    /// 阶段数
    int Stages,
    /// 用于部分特化
    typename Enable = bool>
class DqMmaBase {
public:
    ///< Gemm 问题的大小 - 概念：gemm::GemmShape<>
    using Shape = Shape_;

    ///< 描述调优细节的策略
    using Policy = Policy_;

    ///< 要加载的规模类型的比例
    using ElementScale = ElementScale_;

    //
    // 依赖类型
    //

    /// Warp 级别的 MMA 运算
    using Operator = typename Policy::Operator;

    /// 描述每个 Warp 从共享内存计算的整体 GEMM 的形状
    using WarpGemm = typename Policy::Operator::Shape;

    /// 描述填充 CTA 的 Warp 数量
    using WarpCount = GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN, Shape::kK / WarpGemm::kK>;

    /// 每个 Warp 级别 GEMM 操作的数量
    static int const kWarpGemmIterations = (WarpGemm::kK / Operator::Policy::MmaShape::kK);

    // 每次加载 Warp B 的 K 迭代数
    static constexpr int kNumKIterationsPerWarpBLoad =
        Operator::IteratorB::InstructionShape::kRow / Operator::InstructionShape::kK;

    // 确保 kWarpGemmIterations 能够整除 kNumKIterationsPerWarpBLoad
    static_assert(!(kWarpGemmIterations % kNumKIterationsPerWarpBLoad), "");

    // 计算每次加载 Warp B 的 Warp GEMM 迭代数
    static constexpr int kWarpGemmIterationsForB = kWarpGemmIterations / kNumKIterationsPerWarpBLoad;

    /// 阶段数
    static int const kStages = Stages;

    /// A 操作数的张量引用
    // 定义 TensorRefA 类型作为 Operator 模板中 ElementA 和 LayoutA 的 TensorRef
    using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

    /// 定义 TensorRefB 类型作为 Operator 模板中 ElementB 和 LayoutB 的 TensorRef
    using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

    //
    // 嵌套结构
    //

    /// 用于线程块范围的 GEMM 所需的共享存储对象
    class SharedStorage {
    public:
        //
        // 类型定义
        //

        /// A 矩阵操作数在共享内存中的形状
        using ShapeA =
            MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow, Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;

        /// B 矩阵操作数在共享内存中的形状
        using ShapeB =
            MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow, Shape::kN + Policy::SmemPaddingB::kColumn>;

    public:
        //
        // 数据成员
        //

        /// A 操作数的缓冲区
        AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

        /// B 操作数的缓冲区
        AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

        /// 线程块的缩放因子缓冲区
        AlignedBuffer<ElementScale, Shape::kN> operand_scale;

    public:
        //
        // 方法
        //

        /// 返回 A 矩阵的布局对象
        CUTLASS_DEVICE
        static typename Operator::LayoutA LayoutA()
        {
            return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
        }

        /// 返回 B 矩阵的布局对象
        CUTLASS_HOST_DEVICE
        static typename Operator::LayoutB LayoutB()
        {
            return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
        }

        /// 返回 A 操作数的 TensorRef
        CUTLASS_HOST_DEVICE
        TensorRefA operand_A_ref()
        {
            return TensorRefA{operand_A.data(), LayoutA()};
        }

        /// 返回 B 操作数的 TensorRef
        CUTLASS_HOST_DEVICE
        TensorRefB operand_B_ref()
        {
            return TensorRefB{operand_B.data(), LayoutB()};
        }
    };
protected:
    //
    // Data members
    //

    /// Iterator to load a warp-scoped tile of A operand from shared memory
    /// 用于从共享内存加载 A 操作数的 warp 范围内瓦片迭代器
    typename Operator::IteratorA warp_tile_iterator_A_;

    /// Iterator to load a warp-scoped tile of B operand from shared memory
    /// 用于从共享内存加载 B 操作数的 warp 范围内瓦片迭代器
    typename Operator::IteratorB warp_tile_iterator_B_;

public:
    /// Construct from tensor references
    /// 从张量引用构造对象

    CUTLASS_DEVICE
    DqMmaBase(
        ///< Shared storage needed for internal use by threadblock-scoped GEMM
        ///< 用于线程块范围 GEMM 内部使用的共享存储
        SharedStorage& shared_storage,
        ///< ID within the threadblock
        ///< 线程块内部的 ID
        int thread_idx,
        ///< ID of warp
        ///< warp 的 ID
        int warp_idx,
        ///< ID of each thread within a warp
        ///< warp 内每个线程的 ID
        int lane_idx):
        /// Initialize iterator for operand A using shared memory and lane index
        /// 使用共享内存和 lane 索引初始化 A 操作数的迭代器
        warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
        /// Initialize iterator for operand B using shared memory and lane index
        /// 使用共享内存和 lane 索引初始化 B 操作数的迭代器
        warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx)
    {
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
```