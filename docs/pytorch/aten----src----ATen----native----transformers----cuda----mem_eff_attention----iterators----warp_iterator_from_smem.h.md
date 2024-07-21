# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\iterators\warp_iterator_from_smem.h`

```
/*
 * 版权声明：
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*! \file
    \brief 受 "cutlass/gemm/warp/mma_tensor_op_tile_access_iterator.h" 启发，
           从行主共享内存布局加载 GEMM 操作数的瓦片到寄存器中，以供 A100 TensorCores 使用。

    与 "mma_tensor_op_tile_access_iterator.h" 的区别在于：
    (1) 我们使用 "ldmatrix" 来加载瓦片，而不是手动加载（略微更快）
    (2) 我们支持对操作数进行转置（例如，当共享内存中包含 "A" 时读取 `A.transpose()`）

    这仅实现了特定的形状。
*/

#pragma once

#include <cutlass/gemm/gemm.h>

////////////////////////////////////////////////////////////////////////////////
namespace cutlass {
namespace gemm {
namespace warp {

template <
    /// 操作数标识
    Operand Operand_,
    /// A 元素的数据类型
    typename Element_,
    /// 指令形状
    typename InstructionShape_,
    /// 是否转置操作数
    bool kTranspose = false>
    // 另请参见：
    // https://docs.nvidia.com/cuda/archive/11.7.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
    // 16x8x8: kAccessesInner = 1（1 ldmatrix.x4）
    // 16x8x16: kAccessesInner = 2（2 ldmatrix.x4）
    int ldsm_vec_num = (lane_id >> 3);
    if (kOperand == Operand::kA) {
      // 如果操作数是 Operand::kA
      // 设置起始坐标为 (lane_id % 8, 0)
      origin_ = MatrixCoord(lane_id % 8, 0);
      // 静态断言，确保 InstructionCount::kRow * kTilesPerInstruction == 4，否则输出错误信息
      static_assert(
          InstructionCount::kRow * kTilesPerInstruction == 4,
          "can't use ldmatrix.x4");
      // 计算访问内存块的索引
      int access_m_idx = ldsm_vec_num % kTilesPerInstruction;
      // 计算内部索引
      int inner_idx = (ldsm_vec_num / kTilesPerInstruction) % kAccessesInner;
      // 计算指令内存块索引
      int inst_m_idx = ldsm_vec_num / (kTilesPerInstruction * kAccessesInner);
      // 计算偏移量
      MatrixCoord offset(
          access_m_idx * 8 + inst_m_idx * InstructionShape::kRow,
          inner_idx * 4 * kElementsPerAccess);
      // 如果需要转置，调整偏移量的行列顺序
      if (kTranspose) {
        offset = MatrixCoord(offset.column(), offset.row());
      }
      // 添加偏移量到起始坐标
      origin_ += offset;
    } else {
      // 如果操作数不是 Operand::kA
      // XXX: This is not tested or used，这段代码未经测试或使用
      // 设置起始坐标为 (0, lane_id % 8)
      origin_ = MatrixCoord(0, lane_id % 8);
      // 静态断言，确保 InstructionCount::kColumn * kAccessesInner == 4，无具体错误信息输出
      static_assert(InstructionCount::kColumn * kAccessesInner == 4, "");
      // 对每个指令和内部索引进行循环展开
      CUTLASS_PRAGMA_UNROLL
      for (int inst_n_idx = 0; inst_n_idx < InstructionCount::kColumn;
           ++inst_n_idx) {
        CUTLASS_PRAGMA_UNROLL
        for (int inner_idx = 0; inner_idx < kAccessesInner; ++inner_idx) {
          // 计算访问索引
          int access_idx = inner_idx + kAccessesInner * inst_n_idx;
          // 计算偏移量
          MatrixCoord offset(
              inner_idx * 4 * kElementsPerAccess, inst_n_idx * 8);
          // 如果访问索引与给定的 ldsm_vec_num 相等
          if (access_idx == ldsm_vec_num) {
            // 如果需要转置，调整偏移量的行列顺序
            if (kTranspose) {
              offset = MatrixCoord(offset.column(), offset.row());
            }
            // 添加偏移量到起始坐标
            origin_ += offset;
          }
        }
      }
    }

    // 将计算后的起始坐标添加到 ref_ 中
    ref_.add_coord_offset(origin_);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  WarpIteratorFromSmem& add_tile_offset(TensorCoord const& tile_offset) {
    // 计算坐标偏移量
    TensorCoord coord_offset(
        tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
    // 如果需要转置，调整偏移量的行列顺序
    if (kTranspose) {
      coord_offset = TensorCoord{coord_offset.column(), coord_offset.row()};
    }
    // 添加坐标偏移量到起始坐标
    origin_ += coord_offset;

    // 将坐标偏移量添加到 ref_ 中
    ref_.add_coord_offset(coord_offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  void advance() {
    // 如果操作数是 Operand::kA，调用 add_tile_offset 方法并传入 (0, 1) 偏移量
    if (kOperand == Operand::kA) {
      add_tile_offset({0, 1});
    } else {
      // 如果操作数不是 Operand::kA，调用 add_tile_offset 方法并传入 (1, 0) 偏移量
      add_tile_offset({1, 0});
    }

    // 重置迭代次数为 0
    iterations_ = 0;
  }

  /// increase iterations in a tile
  CUTLASS_HOST_DEVICE
  WarpIteratorFromSmem& operator++() {
    // 迭代次数加一
    iterations_++;

    // 如果迭代次数大于等于 kIterations，调用 advance 方法
    if (iterations_ >= kIterations)
      advance();

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    // 将 frag 解释为 AccessType 类型的指针
    AccessType* access_ptr = reinterpret_cast<AccessType*>(&frag);
    // 定义加载布局，根据 kTranspose 条件选择不同的布局类型
    using LoadLayout = typename platform::
        conditional<kTranspose, layout::ColumnMajor, layout::RowMajor>::type;

    CUTLASS_PRAGMA_UNROLL
    // 对于每个访存操作（每个access_m_idx），执行以下循环体
    for (int access_m_idx = 0; access_m_idx <
         (InstructionCount::kRow * kTilesPerInstruction * kAccessesInner) / 4;
         ++access_m_idx) {
      // 声明一个MatrixCoord对象offset，用于指定当前操作的偏移量
      MatrixCoord offset;
      // 如果操作数为Operand::kA，则计算A矩阵的偏移量
      if (kOperand == Operand::kA) {
        // 根据access_m_idx和iterations_计算A矩阵的行和列偏移量
        offset = MatrixCoord(
            access_m_idx * 16, iterations_ * InstructionShape::kColumn);
      } else {
        // 否则，计算B矩阵的偏移量
        // 根据iterations_计算B矩阵的行偏移量，列偏移量为0
        offset = MatrixCoord(iterations_ * InstructionShape::kRow, 0);
      }
      // 如果需要转置操作（kTranspose为真），交换offset的行列值
      if (kTranspose) {
        offset = MatrixCoord(offset.column(), offset.row());
      }
      // 调用cutlass库中的ldsm函数，从指定的内存位置(access_ptr[access_m_idx])加载数据，
      // 并将数据存储到ref_对象中的指定偏移位置(ref_.offset(offset))
      cutlass::arch::ldsm<LoadLayout, 4>(
          access_ptr[access_m_idx], ref_.data() + ref_.offset(offset));
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass
////////////////////////////////////////////////////////////////////////////////


注释：


// 结束当前的命名空间 'cutlass::gemm::warp'
};

////////////////////////////////////////////////////////////////////////////////

// 结束命名空间 'cutlass::gemm'
} // namespace warp

// 结束命名空间 'cutlass'
} // namespace gemm

// 结束命名空间 'cutlass'
} // namespace cutlass
////////////////////////////////////////////////////////////////////////////////


这段代码是C++中的命名空间闭合语句，用于结束命名空间的定义。每个 `}` 后的注释说明了它所闭合的命名空间或者作用域的范围。
```