# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\gemm\custom_mma_base.h`

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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include <cutlass/aligned_buffer.h>  // 包含对齐缓冲区
#include <cutlass/arch/memory.h>     // 包含架构内存
#include <cutlass/array.h>           // 包含数组
#include <cutlass/cutlass.h>         // 包含 Cutlass 库
#include <cutlass/gemm/gemm.h>       // 包含 GEMM 计算
#include <cutlass/gemm/threadblock/mma_base.h>  // 包含线程块内基础 MMA 计算
#include <cutlass/matrix_shape.h>    // 包含矩阵形状
#include <cutlass/numeric_types.h>   // 包含数值类型

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
    /// Buffer for B operand
    SharedStorageB operand_B;
  };

 public:
  /// Construct from tensor references
  CUTLASS_DEVICE
  CustomMmaBase(TensorRefA ref_A, TensorRefB ref_B) 
    : operand_A(ref_A), operand_B(ref_B) {}

  /// Accesses operand A
  CUTLASS_DEVICE
  TensorRefA A() const { return operand_A.ref(); }

  /// Accesses operand B
  CUTLASS_DEVICE
  TensorRefB B() const { return operand_B.ref(); }
};
    // 定义一个名为 operand_B 的 SharedStorageB 类型成员变量
    SharedStorageB operand_B;
  };

 protected:
  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  // 用于从共享内存加载 warp 级别的 A 操作数瓦片的迭代器
  typename Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  // 用于从共享内存加载 warp 级别的 B 操作数瓦片的迭代器
  typename Operator::IteratorB warp_tile_iterator_B_;

 public:
  /// Construct from tensor references
  // 从张量引用构造函数

  // 构造函数初始化列表：
  // - 初始化 warp_tile_iterator_A_，使用 shared_storageA 的引用和 lane_idx
  //   shared_storageA.ref() 返回共享存储中 A 操作数的引用
  //   lane_idx 是线程在 warp 内的索引
  // - 初始化 warp_tile_iterator_B_，使用 shared_storageB 的引用和 lane_idx
  //   shared_storageB.ref() 返回共享存储中 B 操作数的引用
  CUTLASS_DEVICE
  CustomMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorageA& shared_storageA,
      SharedStorageB& shared_storageB,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      : warp_tile_iterator_A_(shared_storageA.ref(), lane_idx),
        warp_tile_iterator_B_(shared_storageB.ref(), lane_idx) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
```