# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\gemm\custom_mma_pipelined.h`

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
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>

#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>

#include <cutlass/gemm/gemm.h>

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
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename IteratorA_Smem_,
    /// Iterates over tiles of B operand in global memory
    /// (concept: ReadableTileIterator | ForwardTileIterator |
    /// MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename IteratorB_Smem_,
    /// Data type of accumulator matrix C
    typename ElementC_,
    /// Policy describing how to convert data from A to B
    typename LayoutA_,
    /// Policy describing how to convert data from B to A
    typename LayoutB_,
    /// Policy describing how to convert data from A and B to C
    typename LayoutC_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Operation performed by GEMM
    typename OperatorClass_,
    /// Number of threads in each threadblock (concept: gemm::GemmShape<>)
    typename ThreadblockShape_,
    /// Number of warp-level GEMM operations (concept: gemm::GemmShape<>)
    typename WarpShape_,
    /// Transformation applied to A operand
    typename TransformA_ = typename cutlass::gemm::threadblock::
            DefaultTransformA<typename GemmConfig::Shape,
                              typename GemmConfig::ThreadblockShape,
                              typename GemmConfig::OperatorClass>,
    /// Transformation applied to B operand
    typename TransformB_ = typename cutlass::gemm::threadblock::
            DefaultTransformB<typename GemmConfig::Shape,
                              typename GemmConfig::ThreadblockShape,
                              typename GemmConfig::OperatorClass>,
    /// Used for partial specialization
    typename GemmConfig_,
    /// Threadblock-scoped matrix multiply-add
    typename ThreadblockMma_,
    /// Implicit Gemm configuration parameter structure
    typename GemmKernelConfig_ = typename cutlass::gemm::threadblock::
            DefaultGemmKernelConfig<typename GemmConfig::Shape,
                                    typename GemmConfig::ThreadblockShape,
                                    typename GemmConfig::OperatorClass>,
    /// Tag indicating architecture target
    typename ArchTag_,
    /// Enable reduced precision math instructions
    bool EnableReducedPrecision_ = false,
    /// Internal tag indicating internal use only
    typename Internal_ = void>
struct Gemm;


注释：
    typename SmemIteratorA_,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Transformation applied to A operand
    typename TransformA_ = NumericArrayConverter<
        typename SmemIteratorA_::Element,
        typename IteratorA_::Element,
        IteratorA_::Fragment::kElements>,
    ///
    /// Transformation applied to B operand
    typename TransformB_ = NumericArrayConverter<
        typename SmemIteratorB_::Element,
        typename IteratorB_::Element,
        IteratorB_::Fragment::kElements>,
    /// Used for partial specialization
    typename Enable = bool>


注释：


// 定义模板参数 SmemIteratorA_，用于迭代 A 操作数的共享内存中的瓦片
/// 迭代 B 操作数在全局内存中的瓦片
//  （概念：ReadableTileIterator | ForwardTileIterator | MaskedTileIterator）
typename SmemIteratorA_,
// 迭代 B 操作数在共享内存中的瓦片
/// （概念：WriteableTileIterator | RandomAccessTileIterator）
typename IteratorB_,
// 数据累加器矩阵的数据类型
typename SmemIteratorB_,
// 数据累加器矩阵的布局类型
typename ElementC_,
// 描述调优细节的策略（概念：MmaPolicy）
typename LayoutC_,
// A 操作数的转换类型，将共享内存迭代器类型转换为片段类型
typename Policy_,
///
/// B 操作数的转换类型，将共享内存迭代器类型转换为片段类型
typename TransformA_ = NumericArrayConverter<
    typename SmemIteratorA_::Element,
    typename IteratorA_::Element,
    IteratorA_::Fragment::kElements>,
/// 用于部分特化的开关
typename TransformB_ = NumericArrayConverter<
    typename SmemIteratorB_::Element,
    typename IteratorB_::Element,
    IteratorB_::Fragment::kElements>,
/// 用于部分特化的开关，默认为布尔类型
typename Enable = bool>
  ///< Base class
  using Base = CustomMmaBase<Shape_, Policy_, 2>;

  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;

  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;

  ///< Data type of accumulator matrix
  using ElementC = ElementC_;

  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  ///< Iterator for shared memory loading of A operand tiles
  using SmemIteratorA = SmemIteratorA_;

  ///< Iterator for shared memory loading of B operand tiles
  using SmemIteratorB = SmemIteratorB_;

  ///< Transformation function for operand A
  using TransformA = TransformA_;

  ///< Transformation function for operand B
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma operator type
  using Operator = typename Policy::Operator;

  /// Architecture tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  /// Complex transform type for operand A
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform type for operand B
  static ComplexTransform const kTransformB = Operator::kTransformB;

  // Statically assert that MmaPipelined requires exactly two stages (Double-buffered pipeline)
  static_assert(
      (Base::kStages == 2),
      "MmaPipelined requires kStages set to value 2");

  /// Flag indicating whether shared memory contains the entire matrix
  static bool const kSmemContainsEntireMat = false;

 private:
  /// Warp-level fragment type for operand A
  using WarpFragmentA = typename Operator::FragmentA;

  /// Warp-level fragment type for operand B
  using WarpFragmentB = typename Operator::FragmentB;

 protected:
  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

 public:
  /// Constructor initializing the pipelined Mma operation
  CUTLASS_DEVICE
  CustomMmaPipelined(
      typename Base::SharedStorageA& shared_storageA,
      typename Base::SharedStorageB& shared_storageB,
      int thread_idx, ///< ID within the threadblock
      int warp_idx, ///< ID of warp
      int lane_idx ///< ID of thread within warp
      )
      : Base(shared_storageA, shared_storageB, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storageA.ref(), thread_idx),
        smem_iterator_B_(shared_storageB.ref(), thread_idx) {
    // Compute warp's position within the threadblock tile by mapping warp_id to
    // three coordinates:
    //   _m: warp's position along the M dimension
    //   _n: warp's position along the N dimension
    //   _k: warp's position along the K dimension
    // 计算 warp 线程索引在整体线程块中的索引，考虑到 warp 的二维分布
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    // 根据 warp 线程索引计算出在两个维度上的具体索引
    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // 在 warp 级别的迭代器中添加偏移量，单位为 warp 级别的瓦片
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }
  CUTLASS_DEVICE
  CustomMmaPipelined(
      ///< 线程块范围内 GEMM 所需的共享存储
      typename Base::SharedStorage& st,
      ///< 线程索引 ID
      int thread_idx,
      ///< warp 的 ID
      int warp_idx,
      ///< 每个线程在 warp 内的 ID
      int lane_idx)
      : CustomMmaPipelined(
            st.operand_A,
            st.operand_B,
            thread_idx,
            warp_idx,
            lane_idx) {}

  CUTLASS_DEVICE
  bool set_prologue_done(bool value) {
    // 管道化模式下不需要实现此方法
  }

  CUTLASS_DEVICE
  bool set_zero_outside_bounds(bool value) {
    // 管道化模式下不需要设置超出边界为零，共享内存始终被零填充
  }

  template <bool kLoadA = true, bool kLoadB = true>
  CUTLASS_DEVICE static void prologue(
      typename Base::SharedStorage& shared_storage,
      ///< A 操作数的全局内存迭代器
      IteratorA iterator_A,
      ///< B 操作数的全局内存迭代器
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_k) {
    // 此处的 prologue 方法在管道化模式下不实现具体功能
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
      ///< A 操作数的全局内存迭代器
      IteratorA iterator_A,
      ///< B 操作数的全局内存迭代器
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_k) {
    // 管道化模式下不需要实现此方法
  }

  /// 执行线程块范围内的矩阵乘积累加运算
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations, ///< 主循环的迭代次数
      FragmentC& accum, ///< 目标累加器瓦片
      IteratorA iterator_A, ///< A 操作数的全局内存迭代器
      IteratorB iterator_B, ///< B 操作数的全局内存迭代器
      FragmentC const& src_accum, ///< 源累加器瓦片
      TransformA transform_A =
          TransformA(), ///< 应用于 A 片段的变换
      TransformB transform_B =
          TransformB()) { ///< 应用于 B 片段的变换

    //
    // Prologue
    //

    // 在 'd' 输出操作数中执行累加操作
    // 将 src_accum 的值赋给 accum
    accum = src_accum;

    // 创建 FragmentA 和 FragmentB 的实例
    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    // 清空 FragmentA 和 FragmentB 的内容
    tb_frag_A.clear();
    tb_frag_B.clear();

    // 在 prolog 中加载最后一个 k 块
    iterator_A.load(tb_frag_A);
    iterator_B.load(tb_frag_B);

    // 迭代器递增
    ++iterator_A;
    ++iterator_B;

    // 将 transform_A 处理后的 tb_frag_A 存储到 smem_iterator_A_
    this->smem_iterator_A_.store(transform_A(tb_frag_A));
    // 将 transform_B 处理后的 tb_frag_B 存储到 smem_iterator_B_
    this->smem_iterator_B_.store(transform_B(tb_frag_B));

    // smem_iterator_A_ 和 smem_iterator_B_ 各自递增
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    // 同步线程块内所有线程的执行
    __syncthreads();

    // 创建两组用于重叠共享内存加载和数学指令的 WarpFragmentA 和 WarpFragmentB
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    // 设置 warp_tile_iterator_A_ 和 warp_tile_iterator_B_ 的 kgroup_index 为 0
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    // 加载 warp_tile_iterator_A_ 和 warp_tile_iterator_B_ 的数据到 warp_frag_A[0] 和 warp_frag_B[0]
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    // warp_tile_iterator_A_ 和 warp_tile_iterator_B_ 各自递增
    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    // 创建 Operator 实例 warp_mma
    Operator warp_mma;

    // 设置 smem_write_stage_idx 为 1，用于后续的共享内存写入阶段
    int smem_write_stage_idx = 1;

    // 当 gemm_k_iterations <= 1 时，清除 iterator_A 和 iterator_B 的掩码，以避免越界读取
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    // 在第一个 warp 级矩阵乘加中发出加载指令 *AFTER* 发出共享内存加载指令（这些具有最紧密的延迟要求）。

    //
    // Mainloop
    //

    // 注意：主循环不支持 Base::kWarpGemmIterations == 2。
    CUTLASS_GEMM_LOOP
    // 对 gemm_k_iterations 执行循环，gemm_k_iterations 表示 GEMM 操作中的 K 维度迭代次数
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // 遍历 GEMM 的 K 维度
      //

      CUTLASS_PRAGMA_UNROLL
      // 使用 CUTLASS_PRAGMA_UNROLL 对以下 for 循环进行展开优化
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        // 从共享内存加载 warp 级别的矩阵块，如果是最后一个组，则以 k 偏移为情况进行包装

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          // 将片段写入共享内存
          this->smem_iterator_A_.store(transform_A(tb_frag_A));

          this->smem_iterator_B_.store(transform_B(tb_frag_B));

          __syncthreads();

          ++this->smem_iterator_A_;
          ++this->smem_iterator_B_;

          // 如果 smem_write_stage_idx 等于 1，则为共享内存中的循环缓冲区返回迭代器添加负偏移
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          } else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0,
                 -Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {
          iterator_A.load(tb_frag_A);
          iterator_B.load(tb_frag_B);

          ++iterator_A;
          ++iterator_B;

          // 如果这是最后一次循环迭代，则避免读取超出边界
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }

        // 执行 warp_mma 操作
        warp_mma(
            accum,
            warp_frag_A[warp_mma_k % 2],
            warp_frag_B[warp_mma_k % 2],
            accum);
      }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
```