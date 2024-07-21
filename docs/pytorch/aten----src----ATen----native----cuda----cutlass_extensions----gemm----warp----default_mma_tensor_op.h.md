# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\warp\default_mma_tensor_op.h`

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
    \brief Default warp-level GEMM operators selected by data type, size, and layouts of operands.
*/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>

#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h>

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for m-by-n-by-kgroup
template<
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements,
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Data type of B elements
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Element type of C matrix
    typename ElementC,
    /// Computation type of D matrix
    typename ElementD = ElementC,
    /// Operator class
    typename OperatorClass = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag = arch::Sm80,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape = GemmShape_<>,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape = WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape = InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages = 2,
    /// Iterations used in the pipelined mainloop
    int Iterations = 1,
    /// Number of partitions of ThreadblockShape::k
    int PartitionsK = 1,
    /// Number of partitions of WarpShape::k
    int PartitionsKPerWarp = 1,
    /// Whether to enable back-to-back accumulation
    bool EnableBackToBack = false
>
struct DefaultMmaTensorOp;

} // namespace warp
} // namespace gemm
} // namespace cutlass
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<WarpShape_,
                          InstructionShape_,
                          ElementA,
                          LayoutA,
                          ElementB,
                          LayoutB,
                          ElementC,
                          LayoutC,
                          arch::OpMultiplyAddDequantizeInterleavedBToA,
                          PartitionsK,
                          AccumulatorsInRowMajor> {

private:
    // Shape for computing the FP16s
    using ComputeInstructionShape = InstructionShape_;

    // Chosen so we get K=16 for int8 and K=32 for int4.
    // 计算 Load 指令的 K 值，以确保对于 int8 使用 K=16，对于 int4 使用 K=32
    static constexpr int LoadInstructionK = 8 * sizeof_bits<ElementA>::value / sizeof_bits<ElementB>::value;

    // Shape for loading the narrow data type from shared memory
    // 从共享内存加载窄数据类型的形状
    using LoadInstructionShape = GemmShape<InstructionShape_::kM, InstructionShape_::kN, LoadInstructionK>;

public:
    // 定义了 Policy 类型，用于描述 warp 级别的张量操作
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<InstructionShape_,
                                                                             32,
                                                                             ElementA,
                                                                             cutlass::layout::RowMajor,
                                                                             ElementA,
                                                                             cutlass::layout::ColumnMajor,
                                                                             ElementC,
                                                                             cutlass::layout::RowMajor,
                                                                             arch::OpMultiplyAdd>,
                                                          cutlass::MatrixShape<1, 1>>;

    // 定义了 warp 级别的张量操作类型
    using Type = cutlass::gemm::warp::MmaTensorOpComputeBWithF16<WarpShape_,
                                                                 ElementA,
                                                                 LayoutA,
                                                                 ElementB,
                                                                 LayoutB,
                                                                 ElementC,
                                                                 LayoutC,
                                                                 Policy,
                                                                 LoadInstructionShape,
                                                                 PartitionsK,
                                                                 AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass
# 导入必要的模块
import os
import zipfile

# 定义函数，接收一个目录路径作为参数
def zipdir(dirname):
    # 初始化一个空的字节流对象
    bio = BytesIO()
    
    # 创建一个 ZIP 文件写入对象，将文件内容写入到 bio 中
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历指定目录下的所有文件和子目录
        for root, _, files in os.walk(dirname):
            for file in files:
                # 构造每个文件的完整路径
                filepath = os.path.join(root, file)
                # 将文件写入到 ZIP 文件中，使用相对路径作为 ZIP 文件中的路径
                zipf.write(filepath, os.path.relpath(filepath, dirname))
    
    # 将字节流指针移动到起始位置，以便后续读取
    bio.seek(0)
    
    # 返回构建好的 ZIP 文件的字节流
    return bio
```