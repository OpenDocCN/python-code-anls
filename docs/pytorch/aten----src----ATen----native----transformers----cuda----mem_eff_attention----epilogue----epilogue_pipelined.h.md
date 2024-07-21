# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\epilogue\epilogue_pipelined.h`

```
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  File copied from <cutlass/epilogue/threadblock/epilogue.h>
  then modified to:
  (1) load 2 source fragments at the same time (pipelining)
  (2) support reading from a different dtype
  (3) pass the row id to the OutputOp if it takes it
    (see MemoryEfficientAttentionNormalize)
  Note that in general the fragment passed to the OutputOp could
  span multiple rows but it does not happen with the configurations we have
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>  // 包含 CUDA RTC 模式下的断言库
#else
#include <cassert>           // 包含一般情况下的断言库
#endif

#include <cutlass/aligned_buffer.h>  // 引入对齐缓冲区
#include <cutlass/array.h>           // 引入数组支持
#include <cutlass/cutlass.h>         // 引入 Cutlass 库的主头文件
#include <cutlass/functional.h>      // 引入功能相关的头文件
#include <cutlass/layout/tensor.h>   // 引入张量布局定义
#include <cutlass/layout/vector.h>   // 引入向量布局定义
#include <cutlass/numeric_types.h>   // 引入数值类型定义
#include <cutlass/tensor_coord.h>    // 引入张量坐标支持

#include <cutlass/gemm/gemm.h>  // 引入 GEMM 计算相关的头文件

#include <cutlass/transform/pitch_linear_thread_map.h>  // 引入线性线程映射定义
#include <cutlass/transform/threadblock/regular_tile_iterator.h>  // 引入规则瓦片迭代器定义

#include <cutlass/epilogue/threadblock/epilogue_base.h>  // 引入线程块后处理基类
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>  // 引入谓词化瓦片迭代器定义
// 包含 Cutlass 库中定义的数值类型
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////

// Cutlass 命名空间开始
namespace cutlass {
namespace epilogue {
namespace threadblock {

// 应用 Epilogue 操作的结构体，根据不同的输入参数调用 Op 的不同成员函数
template <typename Op>
struct ApplyEpilogueOp {
  
  // 当源数据片段（source）需要被操作时，调用此静态函数
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      typename Op::FragmentAccumulator const& accum,
      typename Op::FragmentOutput const& source) {
    return output_op(accum, source); // 调用 Op 的操作符重载函数
  }
  
  // 当源数据片段（source）不需要被操作时，调用此静态函数
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      typename Op::FragmentAccumulator const& accum) {
    return output_op(accum); // 调用 Op 的操作符重载函数
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
    typename Shape_, ///< Threadblock 瓦片的形状（GemmShape 概念）
    typename WarpMmaOperator_, ///< Warp 级 MMA 运算符（gemm::warp::MmaTensorOp 概念）
    int PartitionsK, ///< K 维度的分区数
    typename OutputTileIterator_, ///< 用于写入输出张量的瓦片迭代器
    typename AccumulatorFragmentIterator_, ///< 选择累加器的片段迭代器
    typename WarpTileIterator_, ///< 用于写入 SMEM 累加器的 Warp 级瓦片迭代器
    typename SharedLoadIterator_, ///< 从 SMEM 加载的 Threadblock 级瓦片迭代器
    typename OutputOp_, ///< 输出操作符
    typename Padding_, ///< 添加到 SMEM 分配的填充以避免 bank 冲突（MatrixShape 概念）
    int FragmentsPerPartition = 1, ///< 用于粗化 Epilogue 的粒度
    int IterationsUnroll = ///< 当 Epilogue 操作很大时用于减少二进制大小
        (!IsEpilogueFunctorHeavy<OutputOp_>::value),
    typename OutputTileSourceIterator_ =
        OutputTileIterator_ ///< 读取张量的瓦片迭代器
    >
    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(output_op, destination_iterator, accumulators);
    } else {
      compute_source_needed_(
          output_op, destination_iterator, accumulators, source_iterator);
    }
  }
  CUTLASS_DEVICE
  void operator()(
      OutputOp const& output_op, ///< 输出操作符
      OutputTileIterator
          destination_iterator, ///< 目标的瓦片迭代器
      AccumulatorTile const&
          accumulators) { ///< 完整的 Warp 级累加器瓦片
    compute_source_not_needed_(output_op, destination_iterator, accumulators);
  }

 private:
  
  // 用于计算当源数据不需要时的 SMEM 累加器操作
  template <class Seq>
  struct acc2smem_source_not_needed;

  // 在给定序列 Seq 的情况下，计算 SMEM 累加器操作，当源数据不需要时使用
  template <size_t... Seq>
  struct acc2smem_source_not_needed<cutlass::index_sequence<Seq...>> {
    template <int Advance>


这段代码是一个 C++ 的模板实现，用于描述 Cutlass 库中的矩阵乘法（GEMM）中的线程块级别的后处理（epilogue）操作。
    /// 静态方法，用于推进累加器片段迭代器和瓦片迭代器
    CUTLASS_DEVICE static void helper(
        AccumulatorFragmentIterator accum_fragment_iterator,  ///< 累加器片段迭代器
        WarpTileIterator& warp_tile_iterator  ///< 瓦片迭代器的引用
    ) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;  // 推进累加器片段迭代器
      }

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {
        typename AccumulatorFragmentIterator::Fragment accum_fragment;

        accum_fragment_iterator.load(accum_fragment);  // 从累加器片段迭代器加载数据
        ++accum_fragment_iterator;  // 推进累加器片段迭代器

        warp_tile_iterator.store(accum_fragment);  // 将累加器片段数据存储到瓦片迭代器中
        if (p < Base::kFragmentsPerIteration - 1) {
          warp_tile_iterator.add_pointer_offset(kSmemPointerOffset);  // 如果不是最后一个片段，添加偏移量到瓦片迭代器
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        warp_tile_iterator.add_pointer_offset(
            kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));  // 调整瓦片迭代器的指针偏移量
      }
    }

    /// 静态方法，用于推送数据到瓦片迭代器
    CUTLASS_DEVICE
    static void push(
        size_t pos,  ///< 当前推送位置
        AccumulatorFragmentIterator const& iterator_begin,  ///< 累加器片段迭代器的起始位置
        WarpTileIterator& warp_tile_iterator  ///< 瓦片迭代器的引用
    ) {
      int dummy[] = {
          (pos == (Seq * Base::kFragmentsPerIteration)) &&  // 检查当前位置是否为预期推送位置
          (helper<Seq * Base::kFragmentsPerIteration>(
               iterator_begin, warp_tile_iterator),  // 调用 helper 方法推送数据到瓦片迭代器
           0)...};

      CUTLASS_UNUSED(dummy[0]);  // 确保 dummy 数组不被优化掉
    }
  };

  static_assert(
      kPartitionsK == 1 || Base::kFragmentsPerIteration == 1,
      "One of these must be exactly 1.");

  /// 将结果流到全局内存
  CUTLASS_DEVICE
  void compute_source_not_needed_(
      OutputOp const& output_op,  ///< 输出操作符
      OutputTileIterator
          destination_iterator,  ///< 目标瓦片迭代器
      AccumulatorTile const&
          accumulators  ///< 完整的 warp 级累加器瓦片
  ) {
    //
    // 迭代器遍历 warp 级别的累加器片段
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // 迭代累加器瓦片
    //
#pragma unroll(                                                          \
    IterationsUnroll                                                     \
        ? OutputTileIterator::kIterations / Base::kFragmentsPerIteration \
        : 1)
    // 使用宏指令 pragma unroll 对循环进行展开，展开次数由条件表达式决定
    for (int iter = 0; iter < OutputTileIterator::kIterations;
         iter += Base::kFragmentsPerIteration) {
      //
      // Convert and store fragment
      //

      __syncthreads();
      // 同步线程，等待所有线程达到同步点

      acc2smem_source_not_needed<cutlass::make_index_sequence<
          OutputTileIterator::kIterations / Base::kFragmentsPerIteration>>::
          push(iter, accum_fragment_iterator, this->warp_tile_iterator_);
      // 调用模板函数 acc2smem_source_not_needed 的静态成员函数 push，
      // 将数据推送到累加器片段迭代器和 warp_tile 迭代器

      __syncthreads();
      // 再次同步线程，确保数据加载完成

      //
      // Load fragments from shared memory
      //

      CUTLASS_PRAGMA_UNROLL
      // 使用宏指令 pragma unroll 对内部循环进行展开
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {
        typename SharedLoadIterator::Fragment
            aligned_accum_fragment[kPartitionsK];
        // 声明类型为 SharedLoadIterator::Fragment 的数组 aligned_accum_fragment，
        // 数组大小为 kPartitionsK

        shared_load_iterator_.load(aligned_accum_fragment[0]);
        // 调用 shared_load_iterator_ 的 load 函数，加载数据到 aligned_accum_fragment[0]

        if (p < Base::kFragmentsPerIteration - 1) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          // 如果 p 小于 Base::kFragmentsPerIteration - 1，则调整指针偏移量
        } else if (kPartitionsK > 1) {
          plus<typename SharedLoadIterator::Fragment> add_fragments;
          // 声明类型为 plus<typename SharedLoadIterator::Fragment> 的 add_fragments 变量

          CUTLASS_PRAGMA_UNROLL
          // 使用宏指令 pragma unroll 对内部循环进行展开
          for (int i = 1; i < kPartitionsK; ++i) {
            shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
            // 调整指针偏移量
            shared_load_iterator_.load(aligned_accum_fragment[i]);
            // 加载数据到 aligned_accum_fragment[i]
            aligned_accum_fragment[0] = add_fragments(
                aligned_accum_fragment[0], aligned_accum_fragment[i]);
            // 使用 add_fragments 对 aligned_accum_fragment[0] 和 aligned_accum_fragment[i] 进行加法操作
          }

          shared_load_iterator_.add_pointer_offset(
              (1 - kPartitionsK) * kSmemPointerOffset);
          // 根据条件调整指针偏移量
        }

        //
        // Compute the output result
        //

        typename OutputTileIterator::Fragment output_fragment;
        // 声明类型为 OutputTileIterator::Fragment 的 output_fragment 变量

        apply_output_operator_source_not_needed_(
            destination_iterator.thread_start_row(),
            output_fragment,
            output_op,
            aligned_accum_fragment[0]);
        // 调用 apply_output_operator_source_not_needed_ 函数，计算输出结果

        //
        // Store the final result
        //

        destination_iterator.store(output_fragment);
        // 调用 destination_iterator 的 store 函数，存储最终结果
        ++destination_iterator;
        // 迭代 destination_iterator
      }

      if (Base::kFragmentsPerIteration > 1) {
        shared_load_iterator_.add_pointer_offset(
            kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
        // 如果 Base::kFragmentsPerIteration 大于 1，则根据条件调整指针偏移量
      }
    }
  }

  template <class Seq>
  struct acc2smem_source_needed;

  template <size_t... Seq>
  struct acc2smem_source_needed<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE static void helper(
        AccumulatorFragmentIterator accum_fragment_iterator,
        WarpTileIterator& warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      // 使用宏指令 pragma unroll 对循环进行展开
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
        // 迭代 accum_fragment_iterator
      }

      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      // 声明类型为 AccumulatorFragmentIterator::Fragment 的 accum_fragment 变量
      accum_fragment_iterator.load(accum_fragment);
      // 调用 accum_fragment_iterator 的 load 函数，加载数据到 accum_fragment
      warp_tile_iterator.store(accum_fragment);
      // 调用 warp_tile_iterator 的 store 函数，存储数据到 warp_tile_iterator
    }

    CUTLASS_DEVICE
    // 声明 CUTLASS_DEVICE 修饰的成员函数
    /// 静态成员函数，将数据推送到指定位置
    static void push(
        size_t pos, /// 推送的位置
        AccumulatorFragmentIterator const& iterator_begin, /// 累加器片段迭代器的起始位置
        WarpTileIterator& warp_tile_iterator /// Warp 瓦片迭代器的引用
    ) {
      int dummy[] = {
          (pos == Seq) && /// 如果 pos 等于 Seq，则执行后面的逗号表达式
          (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...}; /// 调用辅助函数 helper，并将结果放入 dummy 数组
    }
  };

  /// 将结果流式写入全局内存
  CUTLASS_DEVICE
  void compute_source_needed_(
      OutputOp const& output_op, ///< 输出操作器
      OutputTileIterator
          destination_iterator, ///< 目的地瓦片迭代器
      AccumulatorTile const&
          accumulators, ///< 完整的 warp 级别累加器瓦片
      OutputTileSourceIterator
          source_iterator ///< 线程块瓦片在 GEMM 中的坐标（以线程块瓦片为单位）
  ) {
    typename OutputTileSourceIterator::Fragment source_fragment[2]; /// 输出瓦片源迭代器的片段数组，大小为2

    source_fragment[0].clear(); /// 清空第一个片段
    source_iterator.load(source_fragment[0]); /// 从源迭代器加载数据到第一个片段
    ++source_iterator; /// 迭代源迭代器到下一个位置
    source_fragment[1].clear(); /// 清空第二个片段

    //
    // 遍历 warp 级别的累加器片段
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators); /// 使用累加器瓦片构造累加器片段迭代器

    //
    // 遍历累加器瓦片
    //


这段代码主要是关于并行计算中的一些操作和迭代器的使用，其中包括了静态函数的调用和数据流式写入全局内存的过程。
#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    // 根据条件选择是否展开循环，条件是IterationsUnroll非零时展开OutputTileIterator::kIterations次，否则展开1次
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      if (iter > 0) {
        // 如果当前迭代不是第一次迭代，则进行线程同步
        __syncthreads();
      }
      //
      // 加载下一个迭代的源数据（流水线处理）
      //

      if (iter + 1 < OutputTileIterator::kIterations) {
        // 如果还有下一个迭代，加载下一个迭代的源数据
        source_iterator.load(source_fragment[(iter + 1) % 2]);
      }
      ++source_iterator;
      // 将所需的源数据推送到共享内存中
      acc2smem_source_needed<
          cutlass::make_index_sequence<OutputTileIterator::kIterations>>::
          push(iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      //
      // 从共享内存加载片段
      //

      typename SharedLoadIterator::Fragment
          aligned_accum_fragment[kPartitionsK];

      shared_load_iterator_.load(aligned_accum_fragment[0]);

      // 如果k-slices的数量大于1，则在k-slices之间进行归约操作
      if (kPartitionsK > 1) {
        plus<typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(
              aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }

        shared_load_iterator_.add_pointer_offset(
            (1 - kPartitionsK) * kSmemPointerOffset);
      }

      //
      // 计算输出结果
      //

      typename OutputTileIterator::Fragment output_fragment;

      apply_output_operator_(
          destination_iterator.thread_start_row(),
          output_fragment,
          output_op,
          aligned_accum_fragment[0],
          source_fragment[iter % 2]);

      //
      // 存储最终结果
      //

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }

  /// 辅助函数，用于对每个输出向量应用输出操作函数
  CUTLASS_DEVICE
  void apply_output_operator_(
      int begin_row,
      typename OutputTileIterator::Fragment& output_fragment,
      OutputOp const& output_op, ///< 输出操作函数
      typename SharedLoadIterator::Fragment const& aligned_accum_fragment,
      typename OutputTileSourceIterator::Fragment const& source_fragment) {
    OutputAccessType* output_frag_ptr =
        reinterpret_cast<OutputAccessType*>(&output_fragment);

    AccumulatorAccessType const* compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

    SourceAccessType const* source_frag_ptr =
        reinterpret_cast<SourceAccessType const*>(&source_fragment);

    int const kOutputOpIterations = OutputTileIterator::Fragment::kElements /
        OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      // 调用输出运算符
      output_frag_ptr[i] = ApplyEpilogueOp<OutputOp>::apply(
          output_op,
          begin_row + getRowOffset(i * OutputTileIterator::kElementsPerAccess),
          compute_frag_ptr[i],
          source_frag_ptr[i]);
    }
  }

  /// 调用输出函数对象处理每个输出向量的辅助函数
  CUTLASS_DEVICE
  void apply_output_operator_source_not_needed_(
      int begin_row,
      typename OutputTileIterator::Fragment& output_fragment,
      OutputOp const& output_op, ///< 输出运算符
      typename SharedLoadIterator::Fragment const& aligned_accum_fragment) {
    OutputAccessType* output_frag_ptr =
        reinterpret_cast<OutputAccessType*>(&output_fragment);

    AccumulatorAccessType const* compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

    int const kOutputOpIterations = OutputTileIterator::Fragment::kElements /
        OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      // 调用输出运算符
      output_frag_ptr[i] = ApplyEpilogueOp<OutputOp>::apply(
          output_op,
          begin_row + getRowOffset(i * OutputTileIterator::kElementsPerAccess),
          compute_frag_ptr[i]);
    }
  }

  // 这应该是 constexpr，但它仅在 c++14 中受支持
  constexpr int CUTLASS_HOST_DEVICE getRowOffset(int i) {
    using ThreadMap = typename OutputTileIterator::ThreadMap;

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int row_offset = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;
          int frag_row_idx =
              (row +
               ThreadMap::Iterations::kRow *
                   (group + ThreadMap::Iterations::kGroup * cluster));
          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int frag_idx = ThreadMap::kElementsPerAccess *
                (frag_row_idx * ThreadMap::Iterations::kColumn + column);
            if (i < frag_idx + ThreadMap::kElementsPerAccess) {
              return row_offset; // 返回行偏移量
            }
          }
        }
      }
    }
    return -1; // 默认返回值
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////


注释：


// 结束 threadblock 命名空间
};

////////////////////////////////////////////////////////////////////////////////

// 结束 epilogue 命名空间
} // namespace threadblock

// 结束 cutlass 命名空间
} // namespace epilogue
} // namespace cutlass

// 分隔线，用于标记代码段结束
////////////////////////////////////////////////////////////////////////////////


这段代码是 C++ 的命名空间结尾部分。在 C++ 中，命名空间用于避免名称冲突，通过将代码组织成逻辑上的模块，从而更好地组织和管理代码。上述代码中，分别结束了三个嵌套的命名空间：`threadblock`、`epilogue` 和 `cutlass`。每个 `}` 标志着命名空间的结束，而 `//` 开始的行注释用于说明当前位置是哪个命名空间的结尾。最后的 `////////////////////////////////////////////////////////////////////////////////` 是一条分隔线，用于清晰地分隔不同代码段或功能块。
```