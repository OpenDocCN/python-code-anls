# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\threadblock\dq_mma_multistage.h`

```
/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include <cutlass/aligned_buffer.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>

#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/dq_mma_base.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h>
#include <ATen/native/cuda/cutlass_extensions/interleaved_numeric_conversion.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template<
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Data type of accumulator matrix C
    typename ElementOutput_,
    /// Policy describing tuning details (concept: gemm::GemmPolicy)
    typename GemmPolicy_,
    /// Used for partial specialization
    typename ArchTag_,
    /// Transformation applied to A operand before accumulation
    typename TransformationA_,
    /// Transformation applied to B operand before accumulation
    typename TransformationB_,
    /// Output tile size (concept: gemm::GemmShape)
    typename OutputTileSize_,
    /// Complex transformation applied to A operand after read
    typename OutputOp_,
    /// Complex transformation applied to B operand after read
    typename Accumulators_
>
struct DqMmaBase {
    /// Indicate the data type of the scalar accumulator
    using ElementAccumulator = typename GemmPolicy_::Operator::ElementAccumulator;

    /// The type of the cutlass-compatible aligned buffer
    using AlignedBuffer = cutlass::AlignedBuffer<ElementAccumulator, cutlass::kAlignment>;

    /// The architecture of the data
    using ArchTag = ArchTag_;

    /// The shape of the Gemm problem
    using Shape = Shape_;

    /// The output type
    using ElementOutput = ElementOutput_;

    /// Used for partial specialization
    using ArchTag = ArchTag_;

    /// Matrix transformation
    using TransformationA = TransformationA_;
    using TransformationB = TransformationB_;

    /// The strategy for computing output tiles
    using OutputTileSize = OutputTileSize_;

    /// Output tiles size is converted into specific tiling size
    using OutputOp = OutputOp_;

    /// The accoutant is applied to operand in
    cases another as which honour accumulation

 used general the theorem gives processors generated cache
    /// typename IteratorA_: 迭代器类型，用于在共享内存中迭代A操作数的瓦片
    /// (概念: WriteableTileIterator | RandomAccessTileIterator)

    /// typename SmemIteratorA_: 迭代器类型，用于在共享内存中迭代A操作数的瓦片
    /// (概念: WriteableTileIterator | RandomAccessTileIterator)

    /// cutlass::arch::CacheOperation::Kind CacheOpA: 操作A操作数的缓存操作类型

    /// typename IteratorB_: 迭代器类型，用于在全局内存中迭代B操作数的瓦片
    /// (概念: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)

    /// typename SmemIteratorB_: 迭代器类型，用于在共享内存中迭代B操作数的瓦片
    /// (概念: WriteableTileIterator | RandomAccessTileIterator)

    /// cutlass::arch::CacheOperation::Kind CacheOpB: 操作B操作数的缓存操作类型

    /// typename IteratorScale_: 规模数据的迭代器类型

    /// typename SmemIteratorScale_: 在共享内存中迭代规模数据的迭代器类型

    /// typename ElementC_: 累加器矩阵的数据类型

    /// typename LayoutC_: 累加器矩阵的布局类型

    /// typename Policy_: 描述调优细节的策略类型 (概念: MmaPolicy)

    /// int Stages: 阶段数

    /// typename TransformBAfterLDS_: 在LDS之后应用于B矩阵的转换器类型

    /// SharedMemoryClearOption SharedMemoryClear: 用于控制部分特化的共享内存清除选项，
    /// 可选值为SharedMemoryClearOption::kNone表示不清除

    /// typename Enable = bool: 用于部分特化的类型
// 定义了 DqMmaMultistage 类，继承自模板类 DqMmaBase，用于多阶段的量化混合矩阵乘法操作
class DqMmaMultistage: public DqMmaBase<Shape_, Policy_, typename IteratorScale_::Element, Stages> {
public:
    ///< Base class
    // 使用 Base 别名表示基类 DqMmaBase 的实例化，继承了形状 Shape_、策略 Policy_、IteratorScale_ 元素类型和 Stages 阶段数
    using Base = DqMmaBase<Shape_, Policy_, typename IteratorScale_::Element, Stages>;
    ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    // 定义 Shape 别名，表示 Gemm 问题的大小，通常用 gemm::GemmShape<> 概念表示
    using Shape = Shape_;
    ///< Iterates over tiles of A operand in global memory
    // 使用 IteratorA 别名迭代访问 A 操作数在全局内存中的瓦片
    using IteratorA = IteratorA_;
    ///< Iterates over tiles of B operand in global memory
    // 使用 IteratorB 别名迭代访问 B 操作数在全局内存中的瓦片
    using IteratorB = IteratorB_;
    ///< Data type of accumulator matrix
    // 累加器矩阵的数据类型
    using ElementC = ElementC_;
    ///< Layout of accumulator matrix
    // 累加器矩阵的布局类型
    using LayoutC = LayoutC_;
    ///< Policy describing tuning details
    // 描述调优细节的策略类型
    using Policy = Policy_;

    using IteratorScale = IteratorScale_;
    using ElementScale  = typename IteratorScale::Element;
    using LayoutScale   = typename IteratorScale::Layout;

    using SmemIteratorA     = SmemIteratorA_;
    using SmemIteratorB     = SmemIteratorB_;
    using SmemIteratorScale = SmemIteratorScale_;

    // 定义静态常量 kCacheOpA 和 kCacheOpB，表示操作数 A 和 B 的缓存操作类型
    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

    // 使用 TransformBAfterLDS 别名表示 TransformBAfterLDS_，用于描述 LDS 后的 B 操作数的变换
    using TransformBAfterLDS = TransformBAfterLDS_;

    //
    // Dependent types
    //

    /// Fragment of operand Scale loaded from global memory;
    // 从全局内存加载的 Scale 操作数的片段
    using FragmentScale = typename IteratorScale::Fragment;

    /// Fragment of accumulator tile
    // 累加器瓦片的片段类型
    using FragmentC = typename Policy::Operator::FragmentC;

    /// Warp-level Mma
    // Warp 级别的 Mma 运算
    using Operator = typename Policy::Operator;

    /// Minimum architecture is Sm80 to support cp.async
    // 最小架构为 Sm80，以支持 cp.async
    using ArchTag = arch::Sm80;

    // 使用 warp::MmaTensorOpDequantizer 类执行 MmaTensorOp 解量化操作，以支持 Operand::kB、ElementScale 类型和 LayoutScale 布局类型的 32 位操作
    using Dequantizer =
        warp::MmaTensorOpDequantizer<Operator, typename Base::WarpGemm, Operand::kB, ElementScale, LayoutScale, 32>;

    /// Complex transform on A operand
    // A 操作数的复杂变换类型，由 Operator::kTransformA 指定
    static ComplexTransform const kTransformA = Operator::kTransformA;

    /// Complex transform on B operand
    // B 操作数的复杂变换类型，由 Operator::kTransformB 指定
    static ComplexTransform const kTransformB = Operator::kTransformB;

    /// Internal structure exposed for introspection.
    // 用于内省的内部结构暴露
    // 确保基类的 kWarpGemmIterations 大于 1，以支持流水线结构
    static_assert(Base::kWarpGemmIterations > 1,
                  "The pipelined structure requires at least two warp-level "
                  "GEMM operations.");

    // 定义每个操作数 A 的一阶段所需的 cp.async 指令数量
    static int const AsyncCopyIterationsPerStageA = IteratorA::ThreadMap::Iterations::kCount;

    // 定义每个操作数 B 的一阶段所需的 cp.async 指令数量
    static int const AsyncCopyIterationsPerStageB = IteratorB::ThreadMap::Iterations::kCount;

    // 定义流水线结构中的阶段数目
    static int const kStages = Stages;

    // 计算每组操作数 A 所需的 cp.async 指令数量
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    // 计算每组操作数 B 所需的 cp.async 指令数量
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
/// 使用 WarpFragmentA 类型作为 Operator 模板参数 Operator::FragmentA 的别名
using WarpFragmentA = typename Operator::FragmentA;

/// 使用 WarpFragmentB 类型作为 Operator 模板参数 Operator::FragmentB 的别名
using WarpFragmentB = typename Operator::FragmentB;

/// 用于量化反操作的对象
Dequantizer warp_dequantizer_;

/// IteratorB 类型中的元素类型
using ElementB = typename IteratorB::Element;

/// IteratorB 类型中的布局详细信息
using LayoutDetailsForB = kernel::LayoutDetailsB<ElementB, ArchTag>;

/// 确定是否需要瓦片间插入的布尔常量表达式
static constexpr bool RequiresTileInterleave =
    layout::IsColumnMajorTileInterleave<typename LayoutDetailsForB::Layout>::value;

/// 断言，如果需要瓦片间插入，则布局 K 的尺寸必须与 ThreadblockK 相匹配
static_assert(!RequiresTileInterleave || (RequiresTileInterleave && (Shape::kK == LayoutDetailsForB::ThreadblockK)),
              "Layout K must match threadblockK");

//
// Data members
//

/// Iterator，用于将 A 操作数的线程块范围瓦片写入共享内存
SmemIteratorA smem_iterator_A_;

/// Iterator，用于将 B 操作数的线程块范围瓦片写入共享内存
SmemIteratorB smem_iterator_B_;

/// Iterator，用于将比例操作数的线程块范围瓦片写入共享内存
SmemIteratorScale smem_iterator_scale_;

/// 构造函数，从张量引用构造
CUTLASS_DEVICE
DqMmaMultistage(
    ///< 由线程块范围 GEMM 内部使用的共享存储
    typename Base::SharedStorage& shared_storage,
    ///< 线程块内的线程 ID
    int thread_idx,
    ///< Warp 的 ID
    int warp_idx,
    ///< Warp 内每个线程的 ID
    int lane_idx):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    /// Warp 的量化反操作器，使用共享存储中的操作数比例数据和形状 kN 的布局
    warp_dequantizer_({shared_storage.operand_scale.data(), LayoutScale(Shape::kN)},
                      (warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN)) / Base::WarpCount::kM,
                      lane_idx),
    /// IteratorA 初始化为共享存储中的 A 操作数引用，使用线程 ID
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
    /// IteratorB 初始化为共享存储中的 B 操作数引用，使用线程 ID
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
    /// IteratorScale 使用形状 kN 的布局、共享存储中的操作数比例数据和固定形状 {1, Shape::kN}，使用线程 ID
    smem_iterator_scale_(LayoutScale(Shape::kN), shared_storage.operand_scale.data(), {1, Shape::kN}, thread_idx)
{
    // 根据 warp_id 计算 warp 在线程块瓦片中的位置，映射到三个坐标：
    //   _m：warp 在 M 维度上的位置
    //   _n：warp 在 N 维度上的位置
    //   _k：warp 在 K 维度上的位置

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k  = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // 添加每个 warp 的偏移量，单位为 warp 级别的瓦片
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterationsForB * warp_idx_k, warp_idx_n});
}
    copy_tiles_and_advance(IteratorA& iterator_A, IteratorB& iterator_B, int group_start_A = 0, int group_start_B = 0)
    {
        // 设置迭代器A的起始索引，考虑到向量化，乘以每个向量的访问数
        iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
        // 设置共享内存迭代器A的起始索引
        this->smem_iterator_A_.set_iteration_index(group_start_A);

        // 异步拷贝操作，用于操作数A
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
            // 检查是否超出A的异步拷贝阶段迭代次数
            if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
                // 转换共享内存指针为目标类型的指针
                typename IteratorA::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorA::AccessType*>(this->smem_iterator_A_.get());

                // 计算源数据字节数
                int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value
                                      * IteratorA::ThreadMap::kElementsPerAccess / IteratorA::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_A.get();

                    // 根据共享内存清除选项执行异步拷贝（填充0或者正常拷贝）
                    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
                        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr, iterator_A.valid());
                    }
                    else {
                        cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr, iterator_A.valid());
                    }

                    ++iterator_A;
                }

                ++this->smem_iterator_A_;
            }
        }

        // 设置迭代器B的起始索引，考虑到向量化，乘以每个向量的访问数
        iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);
        // 设置共享内存迭代器B的起始索引
        this->smem_iterator_B_.set_iteration_index(group_start_B);

        // 异步拷贝操作，用于操作数B
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
            // 检查是否超出B的异步拷贝阶段迭代次数
            if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
                // 转换共享内存指针为目标类型的指针
                typename IteratorB::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorB::AccessType*>(this->smem_iterator_B_.get());

                // 计算源数据字节数
                int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value
                                      * IteratorB::ThreadMap::kElementsPerAccess / IteratorB::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_B.get();

                    // 根据共享内存清除选项执行异步拷贝（填充0或者正常拷贝）
                    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
                        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr, iterator_B.valid());
                    }
                    else {
                        cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr, iterator_B.valid());
                    }

                    ++iterator_B;
                }
                ++this->smem_iterator_B_;
            }
        }
    }

    /// Perform a threadblock-scoped matrix multiply-accumulate
    CUTLASS_DEVICE
    // 定义一个函数调用运算符重载，用于执行矩阵乘法运算
    void operator()(
        ///< GEMM 的问题大小
        int gemm_k_iterations,
        ///< 累加器的目的地瓦片
        FragmentC& accum,
        ///< 在全局内存中 A 操作数的迭代器
        IteratorA iterator_A,
        ///< 在全局内存中 B 操作数的迭代器
        IteratorB iterator_B,
        ///< 在全局内存中缩放因子的迭代器
        IteratorScale iterator_scale,
        ///< 累加器的初始值
        FragmentC const& src_accum)
    }


在这段代码中：

- `CUTLASS_DEVICE`：这是一个宏或者定义，通常用于指示编译器在设备上编译代码。
- `operator()`：这是一个函数调用运算符的重载，通常用于实现特定类型的对象可以像函数一样被调用的功能。
- 每个参数（如 `gemm_k_iterations`、`accum` 等）都有一个相应的注释，解释了其在函数中的作用和用途。

这些注释帮助阅读者理解每个参数的意义，以及这个函数的整体作用。
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


注释：


// 结束了当前的命名空间 threadblock
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// 结束了命名空间 gemm
}  // namespace gemm

// 结束了命名空间 cutlass
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


这段代码是一个 C++ 的命名空间的结尾部分。在 C++ 中，命名空间通过 `{}` 来定义作用域，这里的注释描述了每个 `}` 的作用，标识了命名空间的层级结构和结束位置。
```