# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\iterators\default_warp_iterator_from_smem.h`

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
    \brief Instantiates the right WarpIterator to read from shared memory
    The class `DefaultWarpIteratorAFromSharedMemory` is useful when reading
        data dumped with `B2bGemm::accumToSmem`.
*/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_access_iterator.h>
#include <cutlass/platform/platform.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/warp_iterator_from_smem.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

// 定义一个模板结构体 DefaultWarpIteratorAFromSharedMemory，用于从共享内存中读取数据
template <
    typename WarpShape,                           // Warp 的形状
    typename InstructionShape,                     // 指令的形状
    typename RegularWarpIterator,                  // 正常的 Warp 迭代器
    typename Policy,                               // 策略
    typename Enable = void>
struct DefaultWarpIteratorAFromSharedMemory {};

// 对于 TensorOp - Ampere half 的情况，特化模板结构体 DefaultWarpIteratorAFromSharedMemory
template <typename RegularWarpIterator, typename Policy, int kInstrK>
struct DefaultWarpIteratorAFromSharedMemory<
    cutlass::gemm::GemmShape<32, 32, 32>,           // 使用 32x32x32 的 GemmShape
    cutlass::gemm::GemmShape<16, 8, kInstrK>,       // 使用 16x8xkInstrK 的 InstructionShape
    RegularWarpIterator,                            // 使用 RegularWarpIterator
    Policy,                                         // 使用 Policy
    typename std::enable_if<kInstrK == 2>::type>     // 当 kInstrK 等于 2 时启用此特化
{
    // 这里通常会定义适合当前条件的 Warp 迭代器
};
    # 如果满足以下条件之一：
    # - RegularWarpIterator 的元素大小为 16 位
    # - Policy 的操作符的 OpDelta 的 kRow 等于 1
    # 则使用 enable_if 启用这部分模板代码

  using OpDelta = typename Policy::Operator::Policy::OpDelta;
    # 定义 OpDelta 为 Policy 中操作符的 OpDelta 类型

  using WarpShape = cutlass::MatrixShape<32, 32>;
    # 定义 WarpShape 为 32x32 的矩阵形状

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, kInstrK>;
    # 定义 InstructionShape 为 gemm 操作的形状，包含 16, 8, kInstrK

  using WarpIterator = cutlass::gemm::warp::WarpIteratorFromSmem<
      cutlass::gemm::Operand::kA,
      typename RegularWarpIterator::Element,
      cutlass::MatrixShape<InstructionShape::kM, InstructionShape::kK>>;
    # 定义 WarpIterator 为从共享内存中创建的 Warp 迭代器，用于 gemm 操作的 kA 操作数，
    # 元素类型为 RegularWarpIterator::Element，
    # 矩阵形状为 InstructionShape 的 kM 和 kK 维度
};

// 结构体模板特化 - 处理A矩阵的默认Warp迭代器，适用于Ampere f32
template <typename WarpShape,                 // Warp形状
          typename RegularWarpIterator,       // 常规Warp迭代器
          typename Policy,                    // 策略
          typename = typename platform::enable_if<(
              sizeof_bits<typename RegularWarpIterator::Element>::value != 16 ||  // 如果元素大小不等于16位
              Policy::Operator::Policy::OpDelta::kRow != 1)>::type> {              // 或者策略中的行偏移不等于1
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;                      // 指令形状为16x8x8
  static constexpr auto kWarpSize = 32;                                             // Warp大小为32
  using OpDelta = typename Policy::Operator::Policy::OpDelta;                        // 操作的行偏移类型

  // 定义Warp迭代器类型为Ampere f32专用的Tensor操作乘法操作数瓦片访问迭代器
  using WarpIterator =
      cutlass::gemm::warp::MmaTensorOpMultiplicandTileAccessIterator<
          cutlass::MatrixShape<WarpShape::kM, WarpShape::kK>,                       // 矩阵形状
          cutlass::gemm::Operand::kA,                                               // 操作数A
          typename RegularWarpIterator::Element,                                    // 迭代器元素类型
          cutlass::layout::RowMajor,                                                // 行主布局
          cutlass::MatrixShape<InstructionShape::kM, InstructionShape::kK>,         // 指令形状
          OpDelta::kRow,                                                            // 行偏移
          kWarpSize>;                                                               // Warp大小
};

// 结构体模板特化 - 处理A矩阵的默认Warp迭代器，适用于Volta
template <typename WarpShape,                 // Warp形状
          typename RegularWarpIterator,       // 常规Warp迭代器
          typename Policy> {                  // 策略
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 4>;                     // 指令形状为16x16x4
  static constexpr auto kWarpSize = 32;                                             // Warp大小为32
  using OpDelta = typename Policy::Operator::Policy::OpDelta;                        // 操作的行偏移类型

  // 定义Warp迭代器类型为Volta专用的Tensor操作乘法操作数瓦片迭代器
  using WarpIterator =
      cutlass::gemm::warp::MmaVoltaTensorOpMultiplicandTileIterator<
          cutlass::MatrixShape<32, 32>,                                             // 矩阵形状
          cutlass::gemm::Operand::kA,                                               // 操作数A
          typename RegularWarpIterator::Element,                                    // 迭代器元素类型
          cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16, 32>,       // Volta专用的布局
          cutlass::MatrixShape<16, 4>,                                              // 指令形状
          OpDelta::kRow,                                                            // 行偏移
          kWarpSize>;                                                               // Warp大小
};

// 结构体模板特化 - 处理A矩阵的默认Warp迭代器，适用于Simt
template <typename WarpShape,                 // Warp形状
          typename RegularWarpIterator,       // 常规Warp迭代器
          typename Policy> {                  // 策略
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;                       // 指令形状为1x1x1
  static constexpr auto kWarpSize = 32;                                             // Warp大小为32

  // 我们使用相同的迭代器，因为我们复制了相同的共享内存结构。只需修改以处理非完整瓦片。
  using WarpIterator = RegularWarpIterator;                                          // 使用常规Warp迭代器
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
```