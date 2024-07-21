# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\gemm\mma_from_smem.h`

```py
/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cutlass/aligned_buffer.h> // 包含对齐缓冲区功能的头文件
#include <cutlass/arch/memory.h> // 包含架构内存配置的头文件
#include <cutlass/array.h> // 包含数组实用功能的头文件
#include <cutlass/cutlass.h> // 包含 CUTLASS 库的主头文件
#include <cutlass/epilogue/thread/linear_combination.h> // 包含线性组合线程块后处理的头文件
#include <cutlass/epilogue/threadblock/default_epilogue_simt.h> // 包含默认 SIMT 架构线程块后处理的头文件
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h> // 包含默认张量操作线程块后处理的头文件
#include <cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h> // 包含默认 Volta 张量操作线程块后处理的头文件
#include <cutlass/functional.h> // 包含功能性实用函数的头文件
#include <cutlass/gemm/gemm.h> // 包含 GEMM 核心功能的头文件
#include <cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h> // 包含 MMA 张量操作片段迭代器的头文件
#include <cutlass/matrix_shape.h> // 包含矩阵形状定义的头文件
#include <cutlass/numeric_conversion.h> // 包含数值类型转换的头文件
#include <cutlass/numeric_types.h> // 包含数值类型定义的头文件
#include <cutlass/platform/platform.h> // 包含平台相关功能的头文件
#include <cutlass/transform/threadblock/vector_iterator.h> // 包含向量迭代器功能的头文件

#include <cutlass/epilogue/threadblock/epilogue_smem_accumulator.h> // 包含 SMEM 累加器后处理的头文件
#include <cutlass/gemm/threadblock/mma_base.h> // 包含 MMA 基础功能的头文件
#include <cutlass/gemm/warp/mma_tensor_op_tile_access_iterator.h> // 包含 MMA 张量操作瓦片访问迭代器的头文件
#include <cutlass/gemm/threadblock/mma_pipelined.h> // 包含流水线 MMA 功能的头文件
#include <cutlass/gemm/threadblock/mma_multistage.h> // 包含多阶段 MMA 功能的头文件
#include <ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_thread_apply_logsumexp.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/make_residual_last.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_accum_lambda_iterator.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/default_warp_iterator_from_smem.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/make_residual_last.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/transpose_warp_iterator.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/warp_iterator_from_smem.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/// Shared storage object needed by accumulator
/// From 13_two_tensor_op_fusion/threadblock/b2b_mma_base_smem_accumulator.h
template <
    typename Shape_,                      // 模板参数：定义形状的类型
    typename Element_,                    // 模板参数：定义元素类型
    typename Layout_,                     // 模板参数：定义布局类型
    typename Padding_>                    // 模板参数：定义填充类型
class AccumulatorSharedStorage {
 public:
  //
  // Type definitions
  //

  using Shape = Shape_;                  // 使用模板参数定义的形状类型
  using Element = Element_;              // 使用模板参数定义的元素类型
  using Layout = Layout_;                // 使用模板参数定义的布局类型
  using Padding = Padding_;              // 使用模板参数定义的填充类型

  /// Tensor reference to the accumulator
  using TensorRefAccum = cutlass::TensorRef<Element, Layout>;

  /// Shape of the accumulator matrix in shared memory
  using ShapeAccum = cutlass::
      MatrixShape<Shape::kM + Padding::kRow, Shape::kN + Padding::kColumn>;

 public:
  //
  // Data members
  //

  /// Buffer for accumulator
  cutlass::AlignedBuffer<Element, ShapeAccum::kCount> accum;    // 累加器的缓冲区

 public:
  //
  // Methods
  //

  /// Returns a layout object for the Accum matrix
  CUTLASS_DEVICE
  static Layout LayoutAccum() {
    return Layout::packed({ShapeAccum::kRow, ShapeAccum::kColumn});   // 返回累加矩阵的布局对象
  }

  /// Returns a TensorRef to the Accumulator
  CUTLASS_HOST_DEVICE
  TensorRefAccum accum_ref() {
    return TensorRefAccum{accum.data(), LayoutAccum()};   // 返回累加器的TensorRef
  }
};

////////////////////////////////////////////////////////////////////////////////
// Taken from
// https://github.com/NVIDIA/cutlass/blob/master/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_base_smem_accumulator.h
////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,                      // 模板参数：定义GEMM问题的形状类型
    // Maximum K dimension - also the dimension of the shared-memory
    // holding `OperandA`
    int kMaxK_,                           // 最大的K维度，也是存储操作数A的共享内存的维度
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,                     // 模板参数：描述调优细节的策略类型
    /// Number of stages,
    int Stages,                           // 阶段的数量
    /// Layout in shared-memory of operand A
    typename SmemLayoutA,                 // 操作数A在共享内存中的布局类型
    /// Used for partial specialization
    typename Enable = bool>               // 用于部分特化
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  // 定义 Gemm 问题的大小，使用 gemm::GemmShape<> 概念
  static constexpr int kMaxK = kMaxK_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  //
  // Dependent types
  //

  /// Warp-level Mma
  using Operator = typename Policy::Operator;
  // 定义 warp 级别的 Mma 运算符

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy::Operator::Shape;
  // 定义描述从共享内存计算的整体 GEMM 的形状，每个 warp 计算一次

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<
      Shape::kM / WarpGemm::kM,
      Shape::kN / WarpGemm::kN,
      Shape::kK / WarpGemm::kK>;
  // 定义描述填充 CTA 的 warp 数量的形状

  using WarpCount1 = WarpCount;

  /// Number of warp-level GEMM operations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);
  // 定义 warp 级别的 GEMM 操作数量

  static int const kWarpGemmIterations1 = kWarpGemmIterations;

  /// Number of stages
  static int const kStages = Stages;
  // 定义阶段数目

  /// If this is true, we fill the entire shmem buffer at start
  /// and don't need to iterate through it in a circular fashion
  static bool const kSmemContainsEntireB = kMaxK <= Shape::kK * kStages;
  // 如果为真，表示在开始时填充整个共享内存缓冲区，无需循环迭代

  /// Tensor reference to the A operand
  using TensorRefA = TensorRef<typename Operator::ElementA, SmemLayoutA>;
  // 定义 A 操作数的张量引用类型

  /// Tensor reference to the B operand
  using TensorRefB =
      TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;
  // 定义 B 操作数的张量引用类型

  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the B matrix operand in shared memory
    using ShapeB = MatrixShape<
        Shape::kK * kStages + Policy::SmemPaddingB::kRow,
        Shape::kN + Policy::SmemPaddingB::kColumn>;
    // 定义共享内存中 B 矩阵操作数的形状

   public:
    //
    // Data members
    //

    /// Buffer for B operand
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;
    // B 操作数的缓冲区

   public:
    //
    // Methods
    //

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }
    // 返回 B 矩阵的布局对象

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }
    // 返回到 B 操作数的张量引用
  }
};

protected:
//
// Data members
//

// /// Iterator to load a warp-scoped tile of A operand from shared memory
// typename Operator::IteratorA warp_tile_iterator_A_;

/// Iterator to load a warp-scoped tile of B operand from shared memory
typename Operator::IteratorB warp_tile_iterator_B_;

public:
/// Construct from tensor references
CUTLASS_DEVICE
MmaBaseFromSharedMemory(
    ///< Shared storage needed for internal use by threadblock-scoped GEMM
    TensorRefB& b_tile,
    ///< ID within the threadblock
    int thread_idx,
    ///< ID of warp
    int warp_idx,
    ///< ID of each thread within a warp
    int lane_idx)
    : warp_tile_iterator_B_(b_tile, lane_idx) {}


注释：


// Closing the class definition with proper formatting.
}
};

// Declaring the following members as protected, indicating they are accessible to derived classes.
protected:
//
// Data members
//

// The next line is commented out, possibly to temporarily exclude it from compilation or indicate it's not currently in use.
// /// Iterator to load a warp-scoped tile of A operand from shared memory
// typename Operator::IteratorA warp_tile_iterator_A_;

// Iterator to load a warp-scoped tile of B operand from shared memory, used in the matrix multiplication operation.
typename Operator::IteratorB warp_tile_iterator_B_;

// The following section is for public interface.

// Constructor initializing MmaBaseFromSharedMemory object.
public:
/// Construct from tensor references
CUTLASS_DEVICE
MmaBaseFromSharedMemory(
    ///< Shared storage needed for internal use by threadblock-scoped GEMM
    TensorRefB& b_tile,
    ///< ID within the threadblock
    int thread_idx,
    ///< ID of warp
    int warp_idx,
    ///< ID of each thread within a warp
    int lane_idx)
    : warp_tile_iterator_B_(b_tile, lane_idx) {}


这段代码定义了一个类 `MmaBaseFromSharedMemory`，包括一些数据成员和一个构造函数，用于从共享内存中加载矩阵操作数的迭代器。
};

namespace {

// 如果继承了 WarpIteratorFromSmem 的必要特性但什么也不做，可以默认初始化，
// 并且使用几乎不占用空间的片段。这个 warp 迭代器在编译时被选择，
// 当操作数 A 的逐元素即时缩放被禁用时，相关于加载操作数 A 的缩放因子的操作会被编译器擦除。
template <typename TensorRef>
class NoOpWarpIteratorScale {
 public:
  // 在流水线+多阶段 MMA 实现中，我们保持一个片段数组。
  // 如果不使用缩放，我们不想浪费寄存器在缩放元素的片段上，因此理想情况下应该大小为 0。
  // 由于不允许零大小对象的数组，因此将大小设置为 1。
  // 编译器很可能会将其擦除。
  using Fragment = cutlass::Array<char, 1>;

  CUTLASS_HOST_DEVICE
  NoOpWarpIteratorScale() {}

  CUTLASS_HOST_DEVICE
  NoOpWarpIteratorScale(TensorRef const&, int) {}

  CUTLASS_HOST_DEVICE
  NoOpWarpIteratorScale& add_tile_offset(
      typename TensorRef::TensorCoord const&) {
    return *this;
  }

  CUTLASS_HOST_DEVICE
  NoOpWarpIteratorScale& operator++() {
    return *this;
  }

  CUTLASS_DEVICE
  void load(Fragment&) const {}
};

// 如果启用了缩放，执行片段与其缩放因子的逐元素乘法。
template <typename Fragment, typename FragmentScale, bool ScalingEnabled>
class FragmentElementwiseScaler;

// 启用缩放的特化版本。
template <typename Fragment, typename FragmentScale>
class FragmentElementwiseScaler<Fragment, FragmentScale, true> {
 public:
  // 将 scale_frag 转换为正确的类型，然后对片段应用逐元素乘法。
  CUTLASS_DEVICE
  static Fragment apply(Fragment frag, FragmentScale const& scale_frag) {
    Fragment converted_scale_frag = cutlass::NumericArrayConverter<
        typename Fragment::Element,
        typename FragmentScale::Element,
        FragmentScale::kElements>()(scale_frag);
    return cutlass::multiplies<Fragment>()(frag, converted_scale_frag);
  }
};

// 禁用缩放的特化版本。什么也不做，应该被编译器擦除。
template <typename Fragment, typename FragmentScale>
class FragmentElementwiseScaler<Fragment, FragmentScale, false> {
 public:
  CUTLASS_DEVICE
  static Fragment apply(Fragment frag, FragmentScale const&) {
    return frag;
  }
};
} // namespace

////////////////////////////////////////////////////////////////////////////////
// 从以下链接获取：https://github.com/NVIDIA/cutlass/blob/master/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_pipelined_smem_accumulator.h
////////////////////////////////////////////////////////////////////////////////

/// 用于计算针对 CUDA 核心和 SIMT 数学指令的矩阵乘积的结构体。
template <
    /// Gemm 问题的大小 - 概念：gemm::GemmShape<>
    typename Shape_,
    // BEGIN smem
    /// Iterates over the intermediate accumulator tile in shared memory
    typename WarpIteratorA_,
    /// whether or not to perform elementwise multiplication of A
    //  by another matrix (A_scale) that is also kept in shared memory prior
    //  to matmul A @ B
    bool ScaleOperandA_,
    /// Max GEMM problem size in K dimension
    int MaxK,
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
    /// Transformation applied to B operand
    typename TransformB_ = NumericArrayConverter<
        typename SmemIteratorB_::Element,
        typename IteratorB_::Element,
        IteratorB_::Fragment::kElements>,
    /// Used for partial specialization
    typename Enable = bool>
{
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_A_scale_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
}

/// Construct from tensor references
CUTLASS_DEVICE
MmaPipelinedFromSharedMemory(
    typename Base::TensorRefA a, ///< Operand A in shared memory
    typename Base::TensorRefB b_staging, ///< staging memory for loading B
    int thread_idx, ///< ID within the threadblock
    int warp_idx, ///< ID of warp
    int lane_idx) ///< ID of each thread within a warp
    : Base(b_staging, thread_idx, warp_idx, lane_idx),
      warp_tile_iterator_A_(a, lane_idx),
      smem_iterator_B_(b_staging, thread_idx)
{
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
}
    // 计算 warp 在 K 维度上在 threadblock 内的位置
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    // 根据 warp 在 threadblock 内的位置计算其在 M 和 N 维度上的索引
    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // 在 warp 级别上添加偏移，单位是 warp 级别的瓦片
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  // 为了与 MmaMultistageFromSharedMemory 的 API 兼容性，但不支持因为它会降低性能：
  // 旧版 sm80 以下的 GPU 不支持异步传输，必须浪费寄存器
  CUTLASS_DEVICE
  void set_prologue_done(bool value) {}
  CUTLASS_DEVICE
  static void prologue(
      typename Base::SharedStorage& shared_storage,
      IteratorB iterator_B1,
      int thread_idx,
      int problem_size_0_n) {}

  /// 执行一个 threadblock 范围的矩阵乘积累加运算
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations, ///< 主循环的迭代次数
      FragmentC& accum, ///< 目标累加器瓦片
      // IteratorA iterator_A,                             ///< A 矩阵的迭代器，全局内存中的操作数
      IteratorB iterator_B, ///< B 矩阵的迭代器，全局内存中的操作数
      FragmentC const& src_accum, ///< 源累加器瓦片
      // TransformA transform_A = TransformA(),            ///< 对 A 碎片应用的变换
      TransformB transform_B =
          TransformB()) { ///< 对 B 碎片应用的变换

    //
    // 前导部分
    //

    // 在 'd' 输出操作数中执行累加
    accum = src_accum;

    FragmentB tb_frag_B;

    tb_frag_B.clear();

    // 最后一个 k 块在前导部分中加载
    iterator_B.set_residual_tile(gemm_k_iterations == 1);
    iterator_B.load(tb_frag_B);

    ++iterator_B;

    this->smem_iterator_B_.store(transform_B(tb_frag_B));

    ++this->smem_iterator_B_;

    __syncthreads();

    // 注意，如果禁用了缩放，WarpFragmentAScale 和 WarpIteratorAScale 是空的/无操作。

    // 用于重叠共享内存加载和数学指令的片对
    WarpFragmentA warp_frag_A[2];
    WarpFragmentAScale warp_frag_A_scale[2];
    WarpFragmentB warp_frag_B[2];
    warp_frag_A[0].clear();
    warp_frag_A_scale[0].clear();
    warp_frag_B[0].clear();

    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_A_scale_.load(warp_frag_A_scale[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_A_scale_;
    ++this->warp_tile_iterator_B_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;
    // 避免读取超出边界
    iterator_B.set_residual_tile(gemm_k_iterations == 2);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    // 在发出共享内存加载后，发出第一个线程束级矩阵乘加操作（*AFTER*，即在最紧急延迟要求的情况下）。

    //
    // 主循环
    //

    // 注意：主循环不支持 Base::kWarpGemmIterations == 2。
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // 遍历 GEMM K 维度
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        // 从共享内存加载线程束级瓦片，如果是最后一组，则包装到 k 偏移。
        bool hasNext = true;

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          if (gemm_k_iterations > 1) {
            // 将片段写入共享内存
            this->smem_iterator_B_.store(transform_B(tb_frag_B));
          }

          __syncthreads();

          ++this->smem_iterator_B_;

          // 将负偏移添加到返回迭代器，以使其回到共享内存中循环缓冲区的“起点”：不重置迭代器 A，因为我们在这一点上继续迭代
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          } else {
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
          hasNext = gemm_k_iterations > 1;
        }

        // 只有在需要时才读取下一个
        if (hasNext) {
          this->warp_tile_iterator_B_.set_kgroup_index(
              (warp_mma_k + 1) % Base::kWarpGemmIterations);

          this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
          this->warp_tile_iterator_A_scale_.load(
              warp_frag_A_scale[(warp_mma_k + 1) % 2]);
          this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

          ++this->warp_tile_iterator_A_;
          ++this->warp_tile_iterator_A_scale_;
          ++this->warp_tile_iterator_B_;

          if (warp_mma_k == 0) {
            iterator_B.load(tb_frag_B);

            ++iterator_B;

            // 如果这是最后一次循环迭代，则避免读取超出边界
            iterator_B.set_residual_tile(gemm_k_iterations == 3);
            iterator_B.clear_mask(gemm_k_iterations <= 2);
          }
        }

        // 执行 warp_mma 操作
        warp_mma(
            accum,
            FragmentAScaler::apply(
                warp_frag_A[warp_mma_k % 2], warp_frag_A_scale[warp_mma_k % 2]),
            warp_frag_B[warp_mma_k % 2],
            accum);
      }
    }
  }
/// 结构体用于计算矩阵乘积，针对 CUDA 核心和 SIMT 数学指令进行优化。
template <
    /// Gemm 问题的大小 - 概念：gemm::GemmShape<>
    typename Shape1_,
    /// 在共享内存中迭代中间累加器瓦片
    typename WarpIteratorA1_,
    /// 是否对 A 执行元素级乘法，A 可能是另一个矩阵（A_scale）的缓存，这个矩阵也在矩阵乘积 A @ B 之前保存在共享内存中
    bool ScaleOperandA_,
    /// 在全局内存中迭代 B 操作数的瓦片（概念：ReadableTileIterator | ForwardTileIterator | MaskedTileIterator）
    typename IteratorB1_,
    /// 在共享内存中迭代 B 操作数的瓦片（概念：WriteableTileIterator | RandomAccessTileIterator）
    typename SmemIteratorB1_,
    /// 操作 B 操作数的缓存策略
    cutlass::arch::CacheOperation::Kind CacheOpB1,
    /// 累加器矩阵的数据类型
    typename ElementC_,
    /// 累加器矩阵的布局类型
    typename LayoutC_,
    /// 描述调优细节的策略（概念：MmaPolicy）
    typename Policy1_,
    /// 阶段的数量
    int Stages_,
    /// k 的最大值
    int kMaxK_,
    /// 用于部分特化
    typename Enable = bool>


这段代码定义了一个模板结构体，用于在 CUDA 核心上执行矩阵乘积操作，并针对不同的优化策略进行了参数化配置。
///< MmaMultistageFromSharedMemory 类继承自 MmaBaseFromSharedMemory，
///< 定义了多级 MMA 计算从共享内存中加载数据的行为。

class MmaMultistageFromSharedMemory : public MmaBaseFromSharedMemory<
                                          Shape1_,
                                          kMaxK_,
                                          Policy1_,
                                          Stages_,
                                          typename WarpIteratorA1_::Layout> {
 public:
  ///< 基类
  using Base = MmaBaseFromSharedMemory<
      Shape1_,
      kMaxK_,
      Policy1_,
      Stages_,
      typename WarpIteratorA1_::Layout>;

  ///< Gemm 问题的大小 - gemm::GemmShape<> 的概念
  using Shape1 = Shape1_;
  ///< 迭代器，用于在全局内存中迭代 B 操作数的瓦片
  using IteratorB1 = IteratorB1_;
  using IteratorB = IteratorB1;
  ///< 描述调优细节的策略
  using Policy1 = Policy1_;

  using SmemIteratorB1 = SmemIteratorB1_;
  ///< Warp 迭代器，用于 A1 矩阵在共享内存中的中间累加器瓦片迭代
  using WarpIteratorA1 = WarpIteratorA1_;
  ///< 如果禁用元素级 A 缩放，则这一切操作都不会执行。
  static constexpr bool ScaleOperandA = ScaleOperandA_;

  ///< Warp 迭代器，用于共享内存中保存的 A_scale 矩阵瓦片
  ///< 如果禁用元素级 A 缩放，则这一切操作都不会执行。
  using WarpIteratorAScale = typename cutlass::platform::conditional<
      ScaleOperandA,
      WarpIteratorA1,
      NoOpWarpIteratorScale<typename WarpIteratorA1::TensorRef>>::type;
  ///< 累加器矩阵的数据类型
  using ElementC = ElementC_;
  ///< 累加器矩阵的布局
  using LayoutC = LayoutC_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpB1 = CacheOpB1;
  ///< 共享内存是否包含整个 B 矩阵的标志
  static constexpr bool kSmemContainsEntireB = Base::kSmemContainsEntireB;

  //
  // Dependent types
  //

  /// 累加器瓦片的片段
  using FragmentC1 = typename Policy1::Operator::FragmentC;
  using FragmentC = FragmentC1;

  /// Warp 级 MMA 运算
  using Operator1 = typename Policy1::Operator;

  /// 最小的架构是 Sm80，支持 cp.async
  using ArchTag = arch::Sm80;

  /// B 操作数的复杂变换
  static ComplexTransform const kTransformB1 = Operator1::kTransformB;

  /// 内部结构，用于内省
  struct Detail {
    static_assert(
        Base::kWarpGemmIterations1 > 1,
        "The pipelined structure requires at least two warp-level "
        "GEMM operations.");

    /// 加载操作数 B 的一个阶段所需的 cp.async 指令数
    static int const TBLoadIterationsB1 =
        IteratorB1::ThreadMap::Iterations::kCount;

    /// 加载操作数 B 一组阶段所需的 cp.async 指令数
    static int const TBLoadIterationsBGroup1 =
        IteratorB1::ThreadMap::Iterations::kGroupCount;

    /// 线程映射类型
    using ThreadMap = typename IteratorB1::ThreadMap;
  };
};
    static int const kAccessesPerGroupB1 =
        (TBLoadIterationsB1 + Base::kWarpGemmIterations1 - 1) /
        Base::kWarpGemmIterations1;
  };

  static constexpr int kNumStagesConcurrentLoad =
      kSmemContainsEntireB ? Base::kStages : Base::kStages - 1;

 private:
  using WarpLoadedFragmentA1 = typename Operator1::FragmentA;
  /// fragment of OperandA scale matrix. if operand A scaling is disabled this
  /// is (almost) empty.
  using WarpLoadedFragmentA1Scale = typename WarpIteratorAScale::Fragment;
  using WarpLoadedFragmentB1 = typename Operator1::FragmentB;
  using WarpTransformedFragmentA1 = typename Operator1::TransformedFragmentA;
  using WarpTransformedFragmentB1 = typename Operator1::TransformedFragmentB;

  /// applies elementwise scaling to fragment of A. if operand A scaling is
  /// disabled this is a no-op.
  using FragmentAScaler = FragmentElementwiseScaler<
      WarpLoadedFragmentA1,
      WarpLoadedFragmentA1Scale,
      ScaleOperandA>;

 private:
  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A1 operand from intermediate
  /// accumulator tile
  WarpIteratorA1 warp_tile_iterator_A1_;

  /// Iterator to load a warp-scoped tile of A1_scale operand from shared memory
  /// if operand A scaling is disabled everything this does is a no-op.
  WarpIteratorAScale warp_tile_iterator_A1_scale_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB1 smem_iterator_B1_;

  bool prologue_done_;

 public:
  /// constructor for MMA with operand A scaling enabled.
  CUTLASS_DEVICE
  MmaMultistageFromSharedMemory(
      typename Base::TensorRefA a,
      typename Base::TensorRefA a_scale,
      typename Base::TensorRefB b_tile,
      int thread_idx,
      int warp_idx,
      int lane_idx)
      : Base(b_tile, thread_idx, warp_idx, lane_idx),
        warp_tile_iterator_A1_(a, lane_idx),
        warp_tile_iterator_A1_scale_(a_scale, lane_idx),
        smem_iterator_B1_(b_tile, thread_idx),
        prologue_done_(false) {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension
    int warp_idx_mn_1 =
        warp_idx % (Base::WarpCount1::kM * Base::WarpCount1::kN);
    int warp_idx_k_1 = warp_idx / (Base::WarpCount1::kM * Base::WarpCount1::kN);
    int warp_idx_m_1 = warp_idx_mn_1 % Base::WarpCount1::kM;
    int warp_idx_n_1 = warp_idx_mn_1 / Base::WarpCount1::kM;

    // Add per-warp offsets in units of warp-level tiles
    warp_tile_iterator_A1_.add_tile_offset(
        {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});
    warp_tile_iterator_A1_scale_.add_tile_offset(
        {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});


注释：
  /// 将瓦片偏移添加到瓦片迭代器 B
  this->warp_tile_iterator_B_.add_tile_offset(
      {Base::kWarpGemmIterations1 * warp_idx_k_1, warp_idx_n_1});
}

/// 从张量引用构造
CUTLASS_DEVICE
MmaMultistageFromSharedMemory(
    typename Base::TensorRefA a,
    typename Base::TensorRefB b_tile,
    ///< 线程块内部的线程 ID
    int thread_idx,
    ///< 瓦ARP 的ID
    int warp_idx,
    ///< 每个线程在瓦ARP内的ID
    int lane_idx)
    : Base(b_tile, thread_idx, warp_idx, lane_idx),
      warp_tile_iterator_A1_(a, lane_idx),
      smem_iterator_B1_(b_tile, thread_idx),
      prologue_done_(false) {
  // 根据 warp_id 计算瓦ARP在线程块瓦片中的位置，分配到三个坐标：
  //   _m: 瓦ARP在 M 维度上的位置
  //   _n: 瓦ARP在 N 维度上的位置
  //   _k: 瓦ARP在 K 维度上的位置

  int warp_idx_mn_1 =
      warp_idx % (Base::WarpCount1::kM * Base::WarpCount1::kN);
  int warp_idx_k_1 = warp_idx / (Base::WarpCount1::kM * Base::WarpCount1::kN);

  int warp_idx_m_1 = warp_idx_mn_1 % Base::WarpCount1::kM;
  int warp_idx_n_1 = warp_idx_mn_1 / Base::WarpCount1::kM;

  // 在 warp 级别瓦片单位中添加每个瓦ARP的偏移量
  warp_tile_iterator_A1_.add_tile_offset(
      {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});
  /// 将瓦片偏移添加到瓦片迭代器 B
  this->warp_tile_iterator_B_.add_tile_offset(
      {Base::kWarpGemmIterations1 * warp_idx_k_1, warp_idx_n_1});
}

CUTLASS_DEVICE
void set_prologue_done(bool value) {
  // 设置 prologue_done_ 标志位
  prologue_done_ = value;
}

CUTLASS_DEVICE
static void prologue(
    typename Base::SharedStorage& shared_storage,
    IteratorB iterator_B1,
    int thread_idx,
    int problem_size_0_n) {
  SmemIteratorB1 smem_iterator_B1(shared_storage.operand_B_ref(), thread_idx);
  _prologue(
      iterator_B1,
      (problem_size_0_n + Base::Shape::kK - 1) / Base::Shape::kK,
      smem_iterator_B1);
}

CUTLASS_DEVICE
void copy_tiles_and_advance_1(
    IteratorB1& iterator_B1,
    int group_start_B1 = 0) {
  iterator_B1.set_iteration_index(
      group_start_B1 * IteratorB1::kAccessesPerVector);
  this->smem_iterator_B1_.set_iteration_index(group_start_B1);

  // 加载操作数 B 的数据
  CUTLASS_PRAGMA_UNROLL


这些注释解释了代码中每个语句的作用和功能，确保了代码的可读性和理解性。
    for (int j = 0; j < Detail::kAccessesPerGroupB1; ++j) {
      // 迭代处理每个访存组的访问
      if (group_start_B1 + j < Detail::TBLoadIterationsB1) {
        // 检查是否未超出访问迭代次数的范围
        typename IteratorB1::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorB1::AccessType*>(
                this->smem_iterator_B1_.get());

        // 计算每个访问向量的源字节大小
        int const kSrcBytes = sizeof_bits<typename IteratorB1::Element>::value *
            IteratorB1::ThreadMap::kElementsPerAccess /
            IteratorB1::kAccessesPerVector / 8;

        // 对访问向量进行展开处理
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B1.get();

          // 执行异步零填充操作
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB1>(
              dst_ptr + v, gmem_ptr, iterator_B1.valid());

          ++iterator_B1;
        }
        // 更新线程块内共享内存迭代器
        ++this->smem_iterator_B1_;
      }
    }
  }

  CUTLASS_DEVICE
  static void _prologue(
      IteratorB& iterator_B1,
      int32_t gemm_k_iterations_1,
      SmemIteratorB1& smem_iterator_B1_) {
    // 执行多个完整的加载阶段
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < kNumStagesConcurrentLoad;
         ++stage, --gemm_k_iterations_1) {
      // 设置残余瓦片和清除掩码
      iterator_B1.set_residual_tile(gemm_k_iterations_1 == 1);
      iterator_B1.clear_mask(gemm_k_iterations_1 == 0);

      // 设置迭代索引为0
      iterator_B1.set_iteration_index(0);
      smem_iterator_B1_.set_iteration_index(0);

      // 加载操作数 B 的数据
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsB1; ++j) {
        typename IteratorB1::AccessType* dst_ptr =
            reinterpret_cast<typename IteratorB1::AccessType*>(
                smem_iterator_B1_.get());

        // 对访问向量进行展开处理
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
          // 计算每个访问向量的源字节大小
          int const kSrcBytes =
              sizeof_bits<typename IteratorB1::Element>::value *
              IteratorB1::ThreadMap::kElementsPerAccess /
              IteratorB1::kAccessesPerVector / 8;

          // 执行异步零填充操作
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB1>(
              dst_ptr + v, iterator_B1.get(), iterator_B1.valid());

          ++iterator_B1;
        }

        // 更新共享内存迭代器
        ++smem_iterator_B1_;
      }

      // 移动到下一个阶段
      iterator_B1.add_tile_offset({1, 0});

      smem_iterator_B1_.add_tile_offset({1, 0});

      // 定义 cp.async 阶段的边界
      cutlass::arch::cp_async_fence();
    }
    // 设置残余瓦片和清除掩码
    iterator_B1.set_residual_tile(gemm_k_iterations_1 == 1);
    iterator_B1.clear_mask(gemm_k_iterations_1 == 0);
  }

  /// 执行线程块范围内的矩阵乘累加运算
  CUTLASS_DEVICE
  void operator()(
      ///< GEMM 问题的迭代次数
      int gemm_k_iterations_1_,
      ///< 目的累加器瓦片
      FragmentC1& accum,
      ///< 全局内存中的 B1 操作数迭代器
      IteratorB1 iterator_B1,
      ///< 累加器的初始值
      FragmentC1 const& src_accum) {
    // 第二次 GEMM 计算

    //
    // 开场白
    //
    // 在输出操作数 'd' 中执行累加
    accum = src_accum;
    // 如果尚未执行 prologue，执行 prologue 操作
    if (!prologue_done_) {
      // 执行 prologue 操作，初始化迭代器和共享内存迭代器
      _prologue(iterator_B1, gemm_k_iterations_1_, smem_iterator_B1_);
    } else if (!kSmemContainsEntireB) {
      // 恢复迭代器的增量

      int gemm_k_iterations_1 = gemm_k_iterations_1_;
      // 执行多个完整的加载阶段
      CUTLASS_PRAGMA_UNROLL
      for (int stage = 0; stage < kNumStagesConcurrentLoad;
           ++stage, --gemm_k_iterations_1) {
        iterator_B1.set_iteration_index(0);
        this->smem_iterator_B1_.set_iteration_index(0);

        // 加载操作数 B
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::TBLoadIterationsB1; ++j) {
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
            ++iterator_B1;
          }
          ++this->smem_iterator_B1_;
        }
        iterator_B1.add_tile_offset({1, 0});
        this->smem_iterator_B1_.add_tile_offset({1, 0});
      }
      // 设置残余的瓦片和清除掩码
      iterator_B1.set_residual_tile(gemm_k_iterations_1 <= 1);
      iterator_B1.clear_mask(gemm_k_iterations_1 <= 0);
    }

    // DEPBAR+SYNC
    // 等待并同步各个加载阶段的完成
    cutlass::arch::cp_async_wait<kNumStagesConcurrentLoad - 1>();
    __syncthreads();

    // 如果禁用了缩放，则 WarpFragmentAScale 和 WarpIteratorAScale 是空操作
    // 用于重叠共享内存加载和数学指令的片对
    WarpLoadedFragmentA1 warp_loaded_frag_A1[2];
    WarpLoadedFragmentA1Scale warp_loaded_frag_A1_scale[2];
    WarpLoadedFragmentB1 warp_loaded_frag_B1[2];
    WarpTransformedFragmentA1 warp_transformed_frag_A1[2];
    WarpTransformedFragmentB1 warp_transformed_frag_B1[2];

    Operator1 warp_mma1;

    // 加载 Warp A 迭代器的片，并递增迭代器
    warp_tile_iterator_A1_.load(warp_loaded_frag_A1[0]);
    ++warp_tile_iterator_A1_;

    // 加载 Warp A 缩放迭代器的片，并递增迭代器
    warp_tile_iterator_A1_scale_.load(warp_loaded_frag_A1_scale[0]);
    ++warp_tile_iterator_A1_scale_;

    // 设置 Warp B 迭代器的 kgroup 索引为 0，并加载其片
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B1[0]);
    ++this->warp_tile_iterator_B_;

    // 设置共享内存写入阶段和读取阶段的索引
    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    // 使用 warp_mma1 进行变换和加载操作数 A 和 B 的片
    warp_mma1.transform(
        warp_transformed_frag_A1[0],
        warp_transformed_frag_B1[0],
        FragmentAScaler::apply(
            warp_loaded_frag_A1[0], warp_loaded_frag_A1_scale[0]),
        warp_loaded_frag_B1[0]);

    // tf32x3 核使用暂存累加器。warp_mma 使用临时累加器，并在每个主循环迭代中将其添加到最终累加器中。
    plus<FragmentC1> plus_accum;

    // 临时累加器
    FragmentC1 tmp_accum;

    // 如果是 OpMultiplyAddFastF32 或 OpMultiplyAddComplexFastF32，清空临时累加器
    if (platform::is_same<
            typename Operator1::MathOperator,
            arch::OpMultiplyAddFastF32>::value ||
        platform::is_same<
            typename Operator1::MathOperator,
            arch::OpMultiplyAddComplexFastF32>::value) {
      tmp_accum.clear();
    }

    //
    // 主循环
    //

    CUTLASS_PRAGMA_UNROLL
    }
    // 如果 Operator1 的 MathOperator 与 arch::OpMultiplyAddFastF32 相同
    // 或者与 arch::OpMultiplyAddComplexFastF32 相同，则执行以下代码块
    if (platform::is_same<
            typename Operator1::MathOperator,
            arch::OpMultiplyAddFastF32>::value ||
        platform::is_same<
            typename Operator1::MathOperator,
            arch::OpMultiplyAddComplexFastF32>::value) {
      // 使用 plus_accum 函数更新累加器 accum 的值，加上 tmp_accum 的值
      accum = plus_accum(accum, tmp_accum);
    }
  }
};

// 结构体模板：将普通的 Mma 转换为从共享内存中获取的对应版本
template <
    typename Mma_,                                     // 原始的 Mma 类型
    int kMaxK,                                          // 最大的 MMA 问题大小 K
    typename WarpIteratorA_,                            // Warp 迭代器 A
    bool kScaleOperandA,                                // 是否对操作数 A 进行共享内存中矩阵的逐元素乘法
    bool kTransposeA = false>                           // 是否对操作数 A 进行转置
struct DefaultMmaFromSharedMemory;

// 结构体模板的特化：用于处理流水线化的 Mma 操作
template <
    typename Shape_,                                    // Gemm 问题的大小和形状的概念：gemm::GemmShape<>
    typename IteratorA_,                                // 全局内存中 A 操作数的瓦片迭代器
    typename SmemIteratorA_,                            // 共享内存中 A 操作数的瓦片迭代器
    typename WarpIteratorA_,                            // Warp 内存中 A 操作数的迭代器
    typename IteratorB_,                                // 全局内存中 B 操作数的瓦片迭代器
    typename SmemIteratorB_,                            // 共享内存中 B 操作数的瓦片迭代器
    typename ElementC_,                                 // 累加器矩阵的数据类型
    typename LayoutC_,                                  // 累加器矩阵的布局
    typename Policy_,                                   // Mma 策略，描述调优细节的概念：MmaPolicy
    typename TransformA_,                               // 应用于 A 操作数的转换
    typename TransformB_,                               // 应用于 B 操作数的转换
    int kMaxK,                                          // 最大 MMA 问题大小 K
    bool kScaleOperandA,                                // 是否对操作数 A 进行共享内存中矩阵的逐元素乘法
    bool kTransposeA>                                   // 是否对操作数 A 进行转置
struct DefaultMmaFromSharedMemory<
    MmaPipelined<                                       // 流水线化的 Mma 类型
        Shape_,
        IteratorA_,
        SmemIteratorA_,
        IteratorB_,
        SmemIteratorB_,
        ElementC_,
        LayoutC_,
        Policy_,
        TransformA_,
        TransformB_>,
    kMaxK,
    WarpIteratorA_,
    kScaleOperandA,
    kTransposeA> {

  using RegularMma = MmaPipelined<                      // 原始的 Mma 类型
      Shape_,
      IteratorA_,
      SmemIteratorA_,
      IteratorB_,
      SmemIteratorB_,
      ElementC_,
      LayoutC_,
      Policy_,
      TransformA_,
      TransformB_>;

  using WarpShape = typename Policy_::Operator::Shape;  // Warp 的形状
  using InstructionShape = typename Policy_::Operator::InstructionShape;  // 指令的形状
  using ArchMmaOperator = typename Policy_::Operator;   // 架构的 Mma 操作符

  static constexpr bool kIsTransposedA = false;         // A 操作数是否被转置
  using WarpIteratorA = WarpIteratorA_;                 // Warp 迭代器 A

  using IteratorB =                                      // 重新计算迭代器 B 类型
      typename cutlass::transform::threadblock::MakeIteratorResidualLast<
          IteratorB_>::Iterator;

  using Mma = typename cutlass::gemm::threadblock::MmaPipelinedFromSharedMemory<
      Shape_,                                           // Gemm 问题的大小和形状
      WarpIteratorA,
      kScaleOperandA,
      kMaxK,
      IteratorB,
      SmemIteratorB_,
      ElementC_,
      LayoutC_,
      Policy_>;                                        // Mma 策略
};

template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    typename WarpIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    int kMaxK,
    /// whether or not to apply elementwise multiplication of operand A by
    /// another matrix in shared memory before usage in A @ B
    bool kScaleOperandA,
    bool kTransposeA>


注释：

/// Size of the Gemm problem defined by the shape template parameter
typename Shape_,
/// Iterator type for iterating over tiles of operand A in global memory,
//  supports concepts: ReadableTileIterator, ForwardTileIterator,
//  MaskedTileIterator
typename IteratorA_,
/// Iterator type for iterating over tiles of operand A in shared memory,
/// supports concepts: WriteableTileIterator, RandomAccessTileIterator
typename SmemIteratorA_,
typename WarpIteratorA_,
/// Cache operation type for operand A (e.g., Load, Store, None)
cutlass::arch::CacheOperation::Kind CacheOpA,
/// Iterator type for iterating over tiles of operand B in global memory,
//  supports concepts: ReadableTileIterator, ForwardTileIterator,
//  MaskedTileIterator
typename IteratorB_,
/// Iterator type for iterating over tiles of operand B in shared memory,
/// supports concepts: WriteableTileIterator, RandomAccessTileIterator
typename SmemIteratorB_,
/// Cache operation type for operand B (e.g., Load, Store, None)
cutlass::arch::CacheOperation::Kind CacheOpB,
/// Element data type of the accumulator matrix C
typename ElementC_,
/// Layout type of the accumulator matrix C (e.g., RowMajor, ColumnMajor)
typename LayoutC_,
/// Policy type describing tuning details for the operation, adheres to concept MmaPolicy
typename Policy_,
/// Number of stages involved in the operation
int Stages,
/// Option for clearing shared memory before use (e.g., ZFill, Predicate)
SharedMemoryClearOption SharedMemoryClear,
/// Maximum value of K dimension for the operation
int kMaxK,
/// Indicates whether to scale operand A by another matrix in shared memory before use in A @ B
bool kScaleOperandA,
/// Indicates whether operand A should be transposed before use in A @ B
bool kTransposeA>
// 定义一个模板结构体 DefaultMmaFromSharedMemory，用于生成默认的从共享内存中执行的多阶段矩阵乘法操作
template <
    // MmaMultistage 类型，描述了矩阵乘法操作的多阶段设置
    MmaMultistage<
        // 矩阵形状 Shape_
        Shape_,
        // 迭代器 IteratorA_
        IteratorA_,
        // 共享内存迭代器 SmemIteratorA_
        SmemIteratorA_,
        // A 矩阵的缓存操作 CacheOpA
        CacheOpA,
        // 迭代器 IteratorB_
        IteratorB_,
        // 共享内存迭代器 SmemIteratorB_
        SmemIteratorB_,
        // B 矩阵的缓存操作 CacheOpB
        CacheOpB,
        // 输出元素类型 ElementC_
        ElementC_,
        // 输出布局 LayoutC_
        LayoutC_,
        // 执行策略 Policy_
        Policy_,
        // 阶段数 Stages
        Stages,
        // 清空共享内存标记 SharedMemoryClear
        SharedMemoryClear>,
    // 最大 K 维度大小
    kMaxK,
    // Warp A 迭代器类型
    WarpIteratorA_,
    // A 矩阵是否进行转置的标志
    kScaleOperandA,
    // A 矩阵是否需要转置的标志
    kTransposeA>
struct DefaultMmaFromSharedMemory {
  // RegularMma 是 MmaMultistage 的一个别名，描述了常规的多阶段矩阵乘法操作
  using RegularMma = MmaMultistage<
      Shape_,
      IteratorA_,
      SmemIteratorA_,
      CacheOpA,
      IteratorB_,
      SmemIteratorB_,
      CacheOpB,
      ElementC_,
      LayoutC_,
      Policy_,
      Stages,
      SharedMemoryClear>;

  // 使用策略中的操作符定义 WarpShape 类型
  using WarpShape = typename Policy_::Operator::Shape;
  // 使用策略中的操作符定义指令形状 InstructionShape 类型
  using InstructionShape = typename Policy_::Operator::InstructionShape;
  // 定义 WarpIteratorA_ 的转置迭代器类型 WarpIteratorTranspose
  using WarpIteratorTranspose = TransposeWarpIterator<WarpIteratorA_>;
  // 是否对 Warp A 迭代器进行转置的标志
  static constexpr bool kIsTransposedA =
      WarpIteratorTranspose::kSupportsTranspose && kTransposeA;
  // 根据是否转置选择 WarpIteratorA_ 或 WarpIteratorTranspose::Iterator 作为 WarpIteratorA 的类型
  using WarpIteratorA = typename platform::conditional<
      kIsTransposedA,
      typename WarpIteratorTranspose::Iterator,
      WarpIteratorA_>::type;

  // 如果需要的话，减少阶段数目，确保不超过阈值 kStagesMax
  static int constexpr kStagesMax =
      (kMaxK + int(Shape_::kK) - 1) / int(Shape_::kK);
  static int constexpr kStages = cutlass::const_min(Stages, kStagesMax);

  // 定义 IteratorB 类型，将其转换为具有 ResidualLast 特性的迭代器
  using IteratorB =
      typename cutlass::transform::threadblock::MakeIteratorResidualLast<
          IteratorB_>::Iterator;
  
  // 定义 Mma 类型，描述从共享内存执行的多阶段矩阵乘法操作
  using Mma =
      typename cutlass::gemm::threadblock::MmaMultistageFromSharedMemory<
          Shape_,
          WarpIteratorA,
          kScaleOperandA,
          IteratorB,
          SmemIteratorB_,
          RegularMma::kCacheOpB,
          ElementC_,
          LayoutC_,
          Policy_,
          kStages,
          kMaxK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// 定义模板结构体 B2bGemm，用于执行 B2B 核心的矩阵乘法操作
template <
    // 迭代器类型 IteratorC，用于存储 C 矩阵的结果
    typename IteratorC,
    // 操作符类型 Operator，描述了乘法操作的细节
    typename Operator,
    // 标量类型 scalar_t，描述了操作数的数据类型
    typename scalar_t,
    // WarpShape_，描述了 Warp 的形状
    typename WarpShape_,
    // ThreadblockShape_，描述了 Threadblock 的形状
    typename ThreadblockShape_>
struct B2bGemm;

// 对于 Tensor Cores >= Sm75 的特化版本，用于 Ampere 架构等
template <
    // 矩阵加载的大小（MatrixShape 概念）
    typename Shape_,
    // 元素类型 Element_
    typename Element_,
    // 操作数的内存布局 Layout_
    typename Layout_,
    // 单个矩阵乘法操作的形状（MatrixShape 概念）
    typename InstructionShape_,
    // 相邻 MMA 指令之间的间隔（以 MMA 指令为单位，MatrixShape 概念）
    typename OpDelta_,
    // 操作符 Operator
    typename Operator,
    // 标量类型 scalar_t
    typename scalar_t,
    // WarpShape_
    typename WarpShape_,
    // ThreadblockShape_
    typename ThreadblockShape_>
struct B2bGemm<
    cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
        Shape_,
        Element_,
        Layout_,
        InstructionShape_,
        OpDelta_>,
    Operator,
    scalar_t,
    WarpShape_,
    ThreadblockShape_> {
  // 定义 SmemIteratorD0 类型的 smem_iterator_attn 对象，用于管理共享内存中的累加器
  SmemIteratorD0 smem_iterator_attn(shared_storage.accum_ref(), lane_id);
};
    // 将瓦片坐标乘以行列迭代次数，添加到共享内存迭代器的偏移量中
    smem_iterator_attn.add_tile_offset(
        tile_coords *
        cutlass::MatrixCoord{
            SmemIteratorD0::TileIterations::kRow,
            SmemIteratorD0::TileIterations::kColumn});

    // 创建一个名为 epilogue 的对象实例
    Epilogue epilogue;

    // 使用无操作的输出操作（NoOp），将共享内存迭代器和累加器传递给 epilogue 对象
    epilogue(OutputOpNoOp({}), smem_iterator_attn, accum);
  }

  // 静态函数：将累加器应用于共享内存
  static void CUTLASS_DEVICE accumApplyLSEToSmem(
      AccumulatorSharedStorage& shared_storage,
      FragmentC& accum,
      lse_scalar_t const* lse,
      int32_t lse_extents,
      int thread_id,
      int warp_id,
      int lane_id,
      cutlass::MatrixCoord const& tile_coords) {
    // 对齐要求为 32
    constexpr int32_t kAlignLSE = 32;

    // 创建迭代器对象 iterator_lse，用于处理 LSE 数据
    IteratorAccumulatorLSE iterator_lse(
        lse,
        {(int32_t)0, (int32_t)ceil_div(lse_extents, kAlignLSE) * kAlignLSE},
        thread_id,
        warp_id,
        cutlass::MatrixCoord{0, 0} // offset
    );

    // 创建共享内存迭代器对象 smem_iterator_attn，使用累加器的引用和 lane_id
    SmemIteratorD0 smem_iterator_attn(shared_storage.accum_ref(), lane_id);

    // 将瓦片坐标乘以行列迭代次数，添加到共享内存迭代器的偏移量中
    smem_iterator_attn.add_tile_offset(
        tile_coords *
        cutlass::MatrixCoord{
            SmemIteratorD0::TileIterations::kRow,
            SmemIteratorD0::TileIterations::kColumn});

    // 创建带 LSE 的 epilogue 对象实例
    EpilogueWithLSE epilogue;

    // 创建应用 LSE 的 EpilogueOpApplyLSE 对象实例 minus_lse_exp
    EpilogueOpApplyLSE minus_lse_exp({});

    // 将对象传递给 epilogue 函数，进行后处理
    epilogue(
        minus_lse_exp,
        smem_iterator_attn,
        accum,
        // scale - 未使用
        iterator_lse,
        // bias
        iterator_lse);
  }
};

// Volta Specialization
// only supported for f16
template <typename Operator, typename WarpShape_, typename ThreadblockShape_>
struct B2bGemm<
    cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
        cutlass::MatrixShape<32, 32>,
        float,
        cutlass::layout::RowMajor,
        cutlass::gemm::GemmShape<16, 16, 4>,
        cutlass::MatrixShape<1, 1>>,
    Operator,
    cutlass::half_t,
    WarpShape_,
    ThreadblockShape_> {
  using IteratorC =
      cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
          cutlass::MatrixShape<32, 32>,
          float,
          cutlass::layout::RowMajor,
          cutlass::gemm::GemmShape<16, 16, 4>,
          cutlass::MatrixShape<1, 1>>;
  using scalar_t = cutlass::half_t;
  using accum_t = IteratorC::Element;
  using WarpShape = WarpShape_;
  using ThreadblockShape = ThreadblockShape_;
  using FragmentC = IteratorC::Fragment;
  using lse_scalar_t = float;

  // Storage in shared-memory for Q.Kt
  // 定义共享内存中的存储布局，适用于 Volta 架构的张量操作乘法
  using SmemAccumulatorLayout =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16, 32>;
  // 线程块内部的累加器共享存储定义
  using AccumulatorSharedStorage =
      cutlass::gemm::threadblock::AccumulatorSharedStorage<
          ThreadblockShape,
          scalar_t,
          SmemAccumulatorLayout,
          cutlass::MatrixShape<0, 0> // Padding
          >;
  // 张量引用类型定义
  using TensorRef = cutlass::TensorRef<scalar_t, SmemAccumulatorLayout>;
  using Policy = typename IteratorC::Policy;
  using Element = accum_t;
  // Those are MmaVoltaTensorOpAccumulatorTileIterator private fields
  // Let's copy their values
  // 定义 MmaVoltaTensorOpAccumulatorTileIterator 的私有字段
  // 复制它们的值
  static int const kElementsPerPartial = 4;
  // 部分计算元素个数
  using EleShapePerPatial = typename cutlass::platform::conditional<
      cutlass::platform::is_same<Element, float>::value,
      cutlass::MatrixShape<2, 2>,
      cutlass::MatrixShape<1, 4>>::type;
  static int const kElementsPerMma = 8;
  // 每个 MMA 操作的元素个数
  static int const kAccumulatorPatials = 2;
  // 累加器部分个数
  using QuadShapePerPatialMma = cutlass::MatrixShape<4, 4>;

  static void CUTLASS_DEVICE accumToSmem(
      AccumulatorSharedStorage& shared_storage,
      FragmentC const& accum,
      int lane_id,
      cutlass::MatrixCoord const& tile_coords) {
    // ctor - from MmaVoltaTensorOpAccumulatorTileIterator
    // 构造函数 - 来自 MmaVoltaTensorOpAccumulatorTileIterator
    TensorRef ref_(shared_storage.accum_ref());
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
    int accum_m, accum_n;

    if (cutlass::platform::is_same<Element, float>::value) {
      // (quad[2],quad[0])+lane_in_quad[0]
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 + (lane_in_quad & 1);
      // (quad[1])+lane_in_quad[1]
      accum_n =
          ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials +
          (lane_in_quad & 2);
    } else {
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 +
          lane_in_quad; // (quad[2],quad[0])
      accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials;
    }
    cutlass::MatrixCoord lane_offset(accum_m, accum_n);

    // Tile offset
    // 添加参考坐标偏移量到 tile_coords，使用 cutlass 库中的 MatrixCoord 类
    ref_.add_coord_offset(
        tile_coords *
        cutlass::MatrixCoord(
            {IteratorC::Shape::kRow, IteratorC::Shape::kColumn}));

    // 定义存取类型为 cutlass 库中的 Array，使用 scalar_t 类型和 EleShapePerPatial::kColumn 大小
    using AccessType = cutlass::Array<scalar_t, EleShapePerPatial::kColumn>;

    // 存储循环 - 来自 MmaVoltaTensorOpAccumulatorTileIterator
    CUTLASS_PRAGMA_UNROLL
    for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn; ++tile_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int tile_m = 0; tile_m < Policy::TileIterations::kRow; ++tile_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
          CUTLASS_PRAGMA_UNROLL
          for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
            // 计算 Mma 累加器起始位置索引
            int mma_accum_start =
                (((tile_n * Policy::TileIterations::kRow + tile_m) *
                      Policy::MmaIterations::kColumn +
                  mma_n) *
                     Policy::MmaIterations::kRow +
                 mma_m) *
                kElementsPerMma;

            CUTLASS_PRAGMA_UNROLL
            for (int p = 0; p < kAccumulatorPatials; ++p) {
              CUTLASS_PRAGMA_UNROLL
              for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
                // 计算累加器索引 m 和 n
                int accum_m = tile_m * Policy::InterleavedTile::kRow +
                    mma_m * QuadShapePerPatialMma::kRow + m * 2;
                int accum_n = tile_n * Policy::InterleavedTile::kColumn +
                    mma_n * QuadShapePerPatialMma::kColumn +
                    p * Policy::InterleavedTile::kColumn / 2;
                int r = (accum_m + lane_offset.row());
                AccessType to_store;
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < EleShapePerPatial::kColumn; ++n) {
                  // 计算累加器中的索引并存储到 to_store 中
                  int idx = mma_accum_start + p * kElementsPerPartial +
                      m * EleShapePerPatial::kColumn + n;
                  int c = (accum_n + n + lane_offset.column());
                  to_store[n] = scalar_t(accum[idx]);
                }
                int c = (accum_n + lane_offset.column());
                // 断言 r 和 c 小于 32，确保不越界
                assert(r < 32);
                assert(c < 32);
                // 将 to_store 存储到 ref_ 中对应的位置
                *reinterpret_cast<AccessType*>(
                    ref_.data() + ref_.offset({r, c})) = to_store;
              }
            }
          }
        }
      }
    }
  }

  // 静态函数：将累加器应用 LSE 到共享内存
  static void CUTLASS_DEVICE accumApplyLSEToSmem(
      AccumulatorSharedStorage& shared_storage,
      typename IteratorC::Fragment& accum,
      lse_scalar_t const* lse,
      int lse_extent,
      int thread_id,
      int warp_id,
      int lane_id,
      cutlass::MatrixCoord const& tile_coords) {
    // 非优化的方式将 LSE 应用到寄存器
    // 注意：accum 是 attn.T
    // TODO: 针对每种架构进行优化
    static constexpr int WarpSize = 32;
    // 使用 cutlass 库中的 DefaultMmaAccumLambdaIterator 定义 AccumLambdaIterator
    using AccumLambdaIterator =
        typename DefaultMmaAccumLambdaIterator<IteratorC, accum_t, WarpSize>::
            Iterator;
    // 计算当前线程的起始偏移量
    auto lane_offset =
        AccumLambdaIterator::get_lane_offset(lane_id, warp_id, tile_coords);

    // 初始化并清空预取的 LSE（Load Store Engine）数据
    cutlass::Array<lse_scalar_t, IteratorC::Fragment::kElements> lse_prefetched;
    lse_prefetched.clear();

    // 初始化行索引和列索引
    int rowIdx = 0;
    int colIdx = 0;

    // 迭代每一行数据
    AccumLambdaIterator::iterateRows(
        lane_offset,
        [&](int accum_m) {
          // 对于每一行，递增行索引
          ++rowIdx;
          // 重置列索引为0
          colIdx = 0;
        },
        [&](int accum_m, int accum_n, int idx) {
          // 对于每个元素，如果是第一行，从 LSE 数组中获取数据或设置为无穷大
          if (rowIdx == 1) {
            lse_prefetched[colIdx] = accum_n < lse_extent
                ? lse[accum_n]
                : platform::numeric_limits<accum_t>::infinity();
          }
          // 对累加器数据应用指数函数，并减去预取的 LSE 数据
          accum[idx] = expf(accum[idx] - lse_prefetched[colIdx]);
          // 递增列索引
          ++colIdx;
        },
        [&](int accum_m) {});

    // 将累加器数据写回共享存储器中
    accumToSmem(shared_storage, accum, lane_id, tile_coords);
  }
};

// Simt Specialization
// for f32 on Sm70-Sm75 and f16/f32 below

template <
    typename Operator,  // 模板参数：运算符类型
    typename OperatorPolicy,  // 模板参数：运算符策略类型
    typename scalar_t,  // 模板参数：标量类型
    typename WarpShape_,  // 模板参数：Warp 形状类型
    typename ThreadblockShape_>  // 模板参数：Threadblock 形状类型
struct B2bGemm<
    cutlass::gemm::warp::MmaSimtTileIterator<  // 定义 B2bGemm 结构体，使用 MmaSimtTileIterator 作为迭代器
        cutlass::MatrixShape<32, 32>,  // 矩阵形状为 32x32
        cutlass::gemm::Operand::kC,  // 操作数类型为 kC
        float,  // 元素类型为 float
        cutlass::layout::RowMajor,  // 布局方式为行主序
        OperatorPolicy,  // 运算符策略类型
        1,  // 迭代次数参数 1
        1>,  // 迭代步长参数 1
    Operator,  // 操作符类型
    scalar_t,  // 标量类型
    WarpShape_,  // Warp 形状类型
    ThreadblockShape_> {  // Threadblock 形状类型
  using IteratorC = cutlass::gemm::warp::MmaSimtTileIterator<  // 定义别名 IteratorC 为 MmaSimtTileIterator 类型
      cutlass::MatrixShape<32, 32>,  // 矩阵形状为 32x32
      cutlass::gemm::Operand::kC,  // 操作数类型为 kC
      float,  // 元素类型为 float
      cutlass::layout::RowMajor,  // 布局方式为行主序
      OperatorPolicy,  // 运算符策略类型
      1,  // 迭代次数参数 1
      1>;  // 迭代步长参数 1
  using accum_t = typename IteratorC::Element;  // 定义别名 accum_t 为 IteratorC 的元素类型
  using WarpShape = WarpShape_;  // 定义别名 WarpShape 为 Warp 形状类型
  using ThreadblockShape = ThreadblockShape_;  // 定义别名 ThreadblockShape 为 Threadblock 形状类型
  using FragmentC = typename IteratorC::Fragment;  // 定义别名 FragmentC 为 IteratorC 的片段类型
  using lse_scalar_t = float;  // 定义别名 lse_scalar_t 为 float 类型

  // Storage in shared-memory for Q.Kt
  // 定义累加器在共享内存中的存储
  using AccumulatorSharedStorage =
      cutlass::gemm::threadblock::AccumulatorSharedStorage<
          ThreadblockShape,
          scalar_t,
          cutlass::layout::ColumnMajor,
          cutlass::MatrixShape<0, 0>  // Padding
          >;

  static void CUTLASS_DEVICE accumToSmem(
      AccumulatorSharedStorage& shared_storage,  // 累加器共享内存的引用
      FragmentC const& accum,  // 累加器片段的常量引用
      int lane_id,  // 线程 lane ID
      cutlass::MatrixCoord const& tile_coords) {  // 矩阵坐标的常量引用
    using Policy = typename IteratorC::Policy;  // 使用 IteratorC 的策略类型
    using Element = typename IteratorC::Element;  // 使用 IteratorC 的元素类型
    using Iterations = typename IteratorC::Iterations;  // 使用 IteratorC 的迭代次数类型
    using Delta = typename IteratorC::Delta;  // 使用 IteratorC 的增量类型

    auto ref_ = shared_storage.accum_ref();  // 获取共享内存中累加器的引用
    // ctor - MmaSimtTileIterator
    // 基于线程 ID 和 lane 布局计算偏移量
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
        MatrixCoord(Policy::LaneMmaShape::kM, Policy::LaneMmaShape::kN);

    ref_.add_coord_offset(lane_offset);  // 添加 lane 偏移量

    // Tile offset
    ref_.add_coord_offset(
        tile_coords *
        cutlass::MatrixCoord(
            {IteratorC::Shape::kRow, IteratorC::Shape::kColumn}));  // 添加矩阵坐标偏移量

    // store - MmaSimtTileIterator
    // 存储操作 - MmaSimtTileIterator
    CUTLASS_PRAGMA_UNROLL
    // 遍历 MMA 操作的多个迭代次数，用于计算矩阵乘法累积结果的索引
    for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      // 遍历 MMA 操作中的每个列的线程块
      for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
        CUTLASS_PRAGMA_UNROLL
        // 遍历 MMA 操作的多个迭代次数，用于计算矩阵乘法累积结果的索引
        for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
          CUTLASS_PRAGMA_UNROLL
          // 遍历 MMA 操作中的每个行的线程块
          for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
            // 计算累积结果在累积矩阵中的行索引
            int r =
                Policy::LaneMmaShape::kM * (mma_m * Policy::WarpShape::kRow) +
                m;
            // 计算累积结果在累积矩阵中的列索引
            int c = mma_n * Delta::kColumn + n;
            // 计算累积结果在累积矩阵中的全局索引
            int idx = n +
                Policy::LaneMmaShape::kN *
                    (mma_n +
                     Iterations::kColumn *
                         (m + mma_m * Policy::LaneMmaShape::kM));
            // 将累积结果写入到目标矩阵的指定位置
            ref_.at({r, c}) = scalar_t(accum[idx]);
          }
        }
      }
    }
  }

  // 将累积结果应用到共享内存
  static void CUTLASS_DEVICE accumApplyLSEToSmem(
      AccumulatorSharedStorage& shared_storage,
      typename IteratorC::Fragment& accum,
      lse_scalar_t const* lse,
      int lse_extent,
      int thread_id,
      int warp_id,
      int lane_id,
      cutlass::MatrixCoord const& tile_coords) {
    // 非优化的方式将 LSE 应用到寄存器中
    // 注意: accum 表示注意力分数的转置
    // TODO: 优化适配各种架构
    static constexpr int WarpSize = 32;
    // 定义累积 Lambda 迭代器类型
    using AccumLambdaIterator =
        typename DefaultMmaAccumLambdaIterator<IteratorC, accum_t, WarpSize>::
            Iterator;
    // 获取当前线程在瓦片内的偏移量
    auto lane_offset =
        AccumLambdaIterator::get_lane_offset(lane_id, warp_id, tile_coords);

    // 清空 LSE 预取数组
    cutlass::Array<lse_scalar_t, IteratorC::Fragment::kElements> lse_prefetched;
    lse_prefetched.clear();
    int rowIdx = 0;
    int colIdx = 0;
    // 迭代累积 Lambda 迭代器的行
    AccumLambdaIterator::iterateRows(
        lane_offset,
        [&](int accum_m) {
          ++rowIdx;
          colIdx = 0;
        },
        // 对累积 Lambda 迭代器的每个元素执行操作
        [&](int accum_m, int accum_n, int idx) {
          // 如果是第一行，从 LSE 中预取数据
          if (rowIdx == 1) {
            lse_prefetched[colIdx] = accum_n < lse_extent
                ? lse[accum_n]
                : platform::numeric_limits<accum_t>::infinity();
          }
          // 应用对数-softmax 操作到累积数据上
          accum[idx] = expf(accum[idx] - lse_prefetched[colIdx]);
          ++colIdx;
        },
        [&](int accum_m) {});
    // 将累积结果写入到共享内存中
    accumToSmem(shared_storage, accum, lane_id, tile_coords);
  }
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


注释：

// 结束 threadblock 命名空间
};

// 结束 gemm 命名空间
} // namespace gemm

// 结束 cutlass 命名空间
} // namespace cutlass

// 分隔线，表示代码的不同部分或模块之间的分界
/////////////////////////////////////////////////////////////////////////////////////////////////


这段代码是 C++ 中的命名空间结尾部分，用来结束嵌套的命名空间定义，并且标识代码中不同模块或部分之间的分界线。
```