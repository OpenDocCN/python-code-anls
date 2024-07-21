# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\gemm\custom_mma_multistage.h`

```py
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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include <cutlass/aligned_buffer.h>
#include <cutlass/arch/cache_operation.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_base.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    typename SmemIteratorA_,
    /// Iterates over tiles of B operand in global memory
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    typename SmemIteratorB_,
    /// Data type of accumulator matrix
    typename ElementOutput_,
    /// Number of stages used in the pipelined mainloop
    int Stages_,
    /// Operation perfomed by GEMM
    typename Operator_ = cutlass::arch::OpClassTensorOp,
    /// Implicit parameter to check if arch has accelerated
    typename ArchTag_ = arch::CacheOperation,
    /// Inverting and a singlular value is only possible with a particular
    /// Template parameter: Iterator type for operand A in shared memory
    typename SmemIteratorA_,
    /// Template parameter: Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Template parameter: Iterator type for operand B in global memory
    //  Iterator type for iterating over tiles of B operand in global memory
    //  (concepts: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Template parameter: Iterator type for operand B in shared memory
    /// Iterator type for iterating over tiles of B operand in shared memory
    /// (concepts: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Template parameter: Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Template parameter: Element type of accumulator matrix C
    typename ElementC_,
    /// Template parameter: Layout type of accumulator matrix C
    typename LayoutC_,
    /// Template parameter: Policy for tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Non-type parameter: Number of stages for the operation
    int Stages,
    /// Non-type parameter: Option for clearing shared memory (default: kNone)
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Non-type parameter: Upper bound on the K dimension
    int kMaxK = cutlass::platform::numeric_limits<int>::max(),
    /// Template parameter: Used for partial specialization (default: bool)
    typename Enable = bool>
// 定义一个继承自 CustomMmaBase 的模板类 CustomMmaMultistage，参数包括形状 Shape_、策略 Policy_、阶段数 Stages
class CustomMmaMultistage : public CustomMmaBase<Shape_, Policy_, Stages> {
 public:
  ///< Base class
  // 使用 Base 别名来引用 CustomMmaBase<Shape_, Policy_, Stages>
  using Base = CustomMmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  // Gemm 问题的大小，这里引用 gemm::GemmShape<> 概念
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  // 在全局内存中迭代 A 操作数的瓦片
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  // 在全局内存中迭代 B 操作数的瓦片
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  // 累加器矩阵的数据类型
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  // 累加器矩阵的布局
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  // 描述调优细节的策略
  using Policy = Policy_;

  // 使用 SmemIteratorA_ 和 SmemIteratorB_ 别名来定义 shared memory 中的迭代器
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  // 定义两个静态常量，分别表示对操作数 A 和 B 的缓存操作种类
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  // 累加器瓦片的片段类型，由 Policy::Operator::FragmentC 定义
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  // Warp 级别的 Mma 运算器类型，由 Policy::Operator 定义
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  // 最小架构要求为 Sm80，以支持 cp.async 操作
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  // A 操作数的复杂变换类型，由 Operator::kTransformA 定义
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  // B 操作数的复杂变换类型，由 Operator::kTransformB 定义
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  // 用于内省的内部结构体 Detail
  struct Detail {
    // 静态断言，确保管道化结构需要至少两个 Warp 级别的 GEMM 操作
    static_assert(
        Base::kWarpGemmIterations > 1,
        "The pipelined structure requires at least two warp-level "
        "GEMM operations.");

    /// Number of cp.async instructions to load one stage of operand A
    // 加载操作数 A 的一个阶段所需的 cp.async 指令数
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    // 加载操作数 B 的一个阶段所需的 cp.async 指令数
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    // 阶段数
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    // 加载操作数 A 的一个组所需的 cp.async 指令数
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) /
        Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    // 加载操作数 B 的一个组所需的 cp.async 指令数
  // 定义每个组的访问次数为每个阶段的异步复制迭代次数与WarpGemm迭代次数之和除以WarpGemm迭代次数向上取整
  static int const kAccessesPerGroupB =
      (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) /
      Base::kWarpGemmIterations;
};

static bool const kSmemContainsEntireMat = kMaxK <= Shape::kK * Stages;
// 如果Shared Memory能够包含整个矩阵，则并行加载阶段的数量为Stages；否则为Stages - 1
static constexpr int kNumStagesConcurrentLoad =
    kSmemContainsEntireMat ? Stages : Stages - 1;

private:
using WarpLoadedFragmentA = typename Operator::FragmentA;
using WarpLoadedFragmentB = typename Operator::FragmentB;
using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

private:
//
// Data members
//

/// 用于将A操作数的线程块范围瓦片写入共享内存的迭代器
SmemIteratorA smem_iterator_A_;

/// 用于将B操作数的线程块范围瓦片写入共享内存的迭代器
SmemIteratorB smem_iterator_B_;

bool prologue_done_;

// 设置为`True`以确保在GEMM足迹之外清零累加器
bool zero_outside_bounds_;

public:
/// 从张量引用构造
CUTLASS_DEVICE
CustomMmaMultistage(
    ///< 线程块范围GEMM内部使用所需的共享存储
    typename Base::SharedStorageA& shared_storageA,
    typename Base::SharedStorageB& shared_storageB,
    ///< 线程块内的ID
    int thread_idx,
    ///< warp的ID
    int warp_idx,
    ///< warp内每个线程的ID
    int lane_idx)
    : Base(shared_storageA, shared_storageB, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storageA.ref(), thread_idx),
      smem_iterator_B_(shared_storageB.ref(), thread_idx),
      prologue_done_(false),
      zero_outside_bounds_(false) {
  // 通过将warp_id映射到三个坐标来计算warp在线程块瓦片内的位置：
  //   _m: warp在M维度上的位置
  //   _n: warp在N维度上的位置
  //   _k: warp在K维度上的位置

  int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
  int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

  int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
  int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

  // 按瓦片级偏移添加每个warp的偏移量
  this->warp_tile_iterator_A_.add_tile_offset(
      {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    // 使用 warp_tile_iterator_B_ 添加瓦片偏移量，该瓦片偏移量由 kWarpGemmIterations 与 warp_idx_k 计算得出，并与 warp_idx_n 组合
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE
  CustomMmaMultistage(
      ///< 用于线程块范围 GEMM 内部使用的共享存储
      typename Base::SharedStorage& st,
      ///< 线程块内部的线程 ID
      int thread_idx,
      ///< warp 的 ID
      int warp_idx,
      ///< warp 内部每个线程的 ID
      int lane_idx)
      : CustomMmaMultistage(
            st.operand_A,
            st.operand_B,
            thread_idx,
            warp_idx,
            lane_idx) {}

  CUTLASS_DEVICE
  bool set_prologue_done(bool value) {
    // 设置 prologue_done_ 标志为给定的值
    prologue_done_ = value;
  }

  CUTLASS_DEVICE
  bool set_zero_outside_bounds(bool value) {
    // 设置 zero_outside_bounds_ 标志为给定的值
    zero_outside_bounds_ = value;
  }

  template <bool kLoadA = true, bool kLoadB = true>
  CUTLASS_DEVICE static void prologue(
      typename Base::SharedStorage& shared_storage,
      ///< A 操作数在全局内存中的迭代器
      IteratorA iterator_A,
      ///< B 操作数在全局内存中的迭代器
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_k) {
    // 调用基类的 prologue 函数，使用共享存储中的操作数 A 和 B，以及给定的迭代器和线程信息
    prologue<kLoadA, kLoadB>(
        shared_storage.operand_A,
        shared_storage.operand_B,
        iterator_A,
        iterator_B,
        thread_idx,
        problem_size_k);
  }

  template <bool kLoadA = true, bool kLoadB = true>
  CUTLASS_DEVICE static void prologue(
      typename Base::SharedStorageA& shared_storageA,
      typename Base::SharedStorageB& shared_storageB,
      ///< A 操作数在全局内存中的迭代器
      IteratorA iterator_A,
      ///< B 操作数在全局内存中的迭代器
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_k) {
    // 使用共享存储中的 A 和 B 操作数创建 smem 迭代器，然后调用 _prologue 函数进行前期处理
    SmemIteratorA smem_iterator_A(shared_storageA.ref(), thread_idx);
    SmemIteratorB smem_iterator_B(shared_storageB.ref(), thread_idx);
    int32_t iter = (problem_size_k + Base::Shape::kK - 1) / Base::Shape::kK;
    _prologue<kLoadA, kLoadB>(
        iterator_A, iterator_B, iter, smem_iterator_A, smem_iterator_B);
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(
      IteratorA& iterator_A,
      IteratorB& iterator_B,
      int group_start_A = 0,
      int group_start_B = 0) {
    // 设置 A 操作数迭代器的索引
    iterator_A.set_iteration_index(
        group_start_A * IteratorA::kAccessesPerVector);
    // 设置 smem 迭代器 A 的索引
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // 异步复制 A 操作数
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      // 遍历组A中的每个访问
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        // 检查是否未超过阶段A的异步复制迭代次数
        typename IteratorA::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType*>(
                this->smem_iterator_A_.get());
        // 获取共享内存迭代器A的指针，并转换为目标访问类型指针

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
            IteratorA::ThreadMap::kElementsPerAccess /
            IteratorA::kAccessesPerVector / 8;
        // 计算源数据的字节数，每个向量访问的位数乘以元素数量除以每个向量的访问次数，再除以8（字节转换）

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          // 对迭代器A中的每个向量访问进行循环
          auto gmem_ptr = iterator_A.get();
          // 获取全局内存迭代器A的指针

          if (zero_outside_bounds_ ||
              SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            // 如果需要在边界外清零或者清空选项为填充零
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
            // 调用异步填充函数，将全局内存数据异步复制到共享内存中，并进行填充零操作
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
            // 否则调用异步复制函数，将全局内存数据异步复制到共享内存中
          }

          ++iterator_A;
          // 迭代全局内存迭代器A
        }

        ++this->smem_iterator_A_;
        // 迭代共享内存迭代器A
      }
    }

    iterator_B.set_iteration_index(
        group_start_B * IteratorB::kAccessesPerVector);
    // 设置迭代器B的迭代索引

    this->smem_iterator_B_.set_iteration_index(group_start_B);
    // 设置共享内存迭代器B的迭代索引为组B的起始位置

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      // 遍历组B中的每个访问
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        // 检查是否未超过阶段B的异步复制迭代次数
        typename IteratorB::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType*>(
                this->smem_iterator_B_.get());
        // 获取共享内存迭代器B的指针，并转换为目标访问类型指针

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
            IteratorB::ThreadMap::kElementsPerAccess /
            IteratorB::kAccessesPerVector / 8;
        // 计算源数据的字节数，每个向量访问的位数乘以元素数量除以每个向量的访问次数，再除以8（字节转换）

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          // 对迭代器B中的每个向量访问进行循环
          auto gmem_ptr = iterator_B.get();
          // 获取全局内存迭代器B的指针

          if (zero_outside_bounds_ ||
              SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            // 如果需要在边界外清零或者清空选项为填充零
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
            // 调用异步填充函数，将全局内存数据异步复制到共享内存中，并进行填充零操作
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
            // 否则调用异步复制函数，将全局内存数据异步复制到共享内存中
          }

          ++iterator_B;
          // 迭代全局内存迭代器B
        }
        ++this->smem_iterator_B_;
        // 迭代共享内存迭代器B
      }
    }
  }
    for (int stage = 0; stage < kNumStagesConcurrentLoad;
         ++stage, --gemm_k_iterations) {
      // 清除迭代器A和B的掩码，如果gemm_k_iterations为0则不清除
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);

      // 设置迭代器A的迭代索引为0
      iterator_A.set_iteration_index(0);
      // 设置smem_iterator_A_的迭代索引为0
      smem_iterator_A_.set_iteration_index(0);

      // 异步复制操作，操作数A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        // 获取smem_iterator_A_的目标指针
        typename IteratorA::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType*>(
                smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          // 计算每个访问向量的源字节数
          int const kSrcBytes =
              sizeof_bits<typename IteratorA::Element>::value *
              IteratorA::ThreadMap::kElementsPerAccess /
              IteratorA::kAccessesPerVector / 8;

          // 根据iterator_A的有效性设置源字节数
          int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

          // 如果kLoadA为真，则进行异步填充操作
          if (kLoadA) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, iterator_A.get(), iterator_A.valid());
          }

          // 迭代器A自增
          ++iterator_A;
        }

        // smem_iterator_A_自增
        ++smem_iterator_A_;
      }

      // 设置迭代器B的迭代索引为0
      iterator_B.set_iteration_index(0);
      // 设置smem_iterator_B_的迭代索引为0
      smem_iterator_B_.set_iteration_index(0);

      // 异步复制操作，操作数B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        // 获取smem_iterator_B_的目标指针
        typename IteratorB::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType*>(
                smem_iterator_B_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          // 计算每个访问向量的源字节数
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;

          // 如果kLoadB为真，则进行异步填充操作
          if (kLoadB) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, iterator_B.get(), iterator_B.valid());
          }

          // 迭代器B自增
          ++iterator_B;
        }

        // smem_iterator_B_自增
        ++smem_iterator_B_;
      }

      // 移动到下一个阶段
      // 迭代器A和B增加瓦片偏移
      iterator_A.add_tile_offset({0, 1});
      iterator_B.add_tile_offset({1, 0});

      // smem_iterator_A_和smem_iterator_B_增加瓦片偏移
      smem_iterator_A_.add_tile_offset({0, 1});
      smem_iterator_B_.add_tile_offset({1, 0});

      // 定义cp.async阶段的边界
      cutlass::arch::cp_async_fence();
    }
  }

  /// 执行线程块范围的矩阵乘累加运算
  CUTLASS_DEVICE
  void operator()(
      ///< GEMM问题的迭代次数
      int gemm_k_iterations,
      ///< 累加器的目标瓦片
      FragmentC& accum,
      ///< 迭代器A，指向全局内存中的操作数A
      IteratorA iterator_A,
      ///< 迭代器B，指向全局内存中的操作数B
      IteratorB iterator_B,
      ///< 累加器的初始值
      FragmentC const& src_accum) {
    //
    // Prologue
    //
    // 如果尚未执行初始化操作
    if (!prologue_done_) {
      // 执行初始化操作，处理整个矩阵
      _prologue<true, true>(
          iterator_A,
          iterator_B,
          gemm_k_iterations,
          smem_iterator_A_,
          smem_iterator_B_);
    } else if (!kSmemContainsEntireMat) {
      // 如果共享内存不包含整个矩阵，则执行部分初始化操作
      _prologue<false, false>(
          iterator_A,
          iterator_B,
          gemm_k_iterations,
          smem_iterator_A_,
          smem_iterator_B_);
    } else {
      // 否则减去并行加载的阶段数
      gemm_k_iterations -= kNumStagesConcurrentLoad;
    }

    // 将累加器设置为源累积值
    accum = src_accum;

    //
    // 清除剩余的SMEM瓦片。对于某些内核，这是一个功能要求，以确保在GEMM足迹之外的所有累加器元素为零。
    //

    // 如果SharedMemoryClear选项为kClearLastStage
    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {
      /// 迭代器，用于将A操作数的线程块范围瓦片写入共享内存
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // A操作数的异步复制
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType*>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// 迭代器，用于将B操作数的线程块范围瓦片写入共享内存
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // B操作数的异步复制
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType*>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }

    // 等待直到kStages-2阶段已提交
    cutlass::arch::cp_async_wait<kNumStagesConcurrentLoad - 1>();
    __syncthreads();

    // 用于重叠共享内存加载和数学指令的片段对
    WarpLoadedFragmentA warp_loaded_frag_A[2];
    WarpLoadedFragmentB warp_loaded_frag_B[2];
    WarpTransformedFragmentA warp_transformed_frag_A[2];
    WarpTransformedFragmentB warp_transformed_frag_B[2];

    Operator warp_mma;

    // 设置kgroup_index为0
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    // 加载A操作数的瓦片
    this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
    // 加载B操作数的瓦片
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    // 如果gemm_k_iterations为0，则清除A和B迭代器的掩码
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);
    # 设置写入阶段的共享内存索引为 Base::kStages - 1
    int smem_write_stage_idx = Base::kStages - 1;
    # 设置读取阶段的共享内存索引为 0
    int smem_read_stage_idx = 0;

    # 使用 warp_mma 对象进行变换操作，处理 warp_transformed_frag_A 和 warp_transformed_frag_B
    # 从 warp_loaded_frag_A 和 warp_loaded_frag_B 中加载数据
    warp_mma.transform(
        warp_transformed_frag_A[0],
        warp_transformed_frag_B[0],
        warp_loaded_frag_A[0],
        warp_loaded_frag_B[0]);

    # 对于 tf32x3 内核，使用分段累加器进行累加。warp_mma 使用临时累加器，该临时累加器在每个主循环迭代中仅添加到最终累加器一次。
    # plus_accum 用于累加结果
    plus<FragmentC> plus_accum;

    # 定义临时累加器 tmp_accum
    FragmentC tmp_accum;

    # 如果使用的是 arch::OpMultiplyAddFastF32 或 arch::OpMultiplyAddComplexFastF32 中的数学运算操作符，
    # 则清空临时累加器 tmp_accum
    if (platform::is_same<
            typename Operator::MathOperator,
            arch::OpMultiplyAddFastF32>::value ||
        platform::is_same<
            typename Operator::MathOperator,
            arch::OpMultiplyAddComplexFastF32>::value) {
      tmp_accum.clear();
    }

    #
    # 主循环
    #

    CUTLASS_GEMM_LOOP
    }

    # 如果使用的是 arch::OpMultiplyAddFastF32 或 arch::OpMultiplyAddComplexFastF32 中的数学运算操作符，
    # 将临时累加器 tmp_accum 的结果添加到累加器 accum 中
    if (platform::is_same<
            typename Operator::MathOperator,
            arch::OpMultiplyAddFastF32>::value ||
        platform::is_same<
            typename Operator::MathOperator,
            arch::OpMultiplyAddComplexFastF32>::value) {
      accum = plus_accum(accum, tmp_accum);
    }

    # 如果 SharedMemoryClear 设置为 SharedMemoryClearOption::kZfill
    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
      # 提交和清除所有待处理和条件预测的 cp.async pnz 操作，来自 GEMM 主循环
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      # 同步线程
      __syncthreads();
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
```