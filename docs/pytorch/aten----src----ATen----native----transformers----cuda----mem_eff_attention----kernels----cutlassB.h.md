# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\kernels\cutlassB.h`

```py
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"

// 引入 CUDA 加速库的注意事项后向核心头文件
#pragma once
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>

// 使用 PyTorchMemEffAttention 命名空间
using namespace PyTorchMemEffAttention;

// ======== f16 / sm70 ========

// 定义 CUDA 全局函数，指定其启动配置和线程限制，处理半精度数据的反向注意力后向传播任务
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32, true>::Params p);

// 定义 CUDA 全局函数，指定其启动配置和线程限制，处理半精度数据的反向注意力后向传播任务
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32>::Params p);

// 定义 CUDA 全局函数，指定其启动配置和线程限制，处理半精度数据的反向注意力后向传播任务
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64, true>::Params p);

// 定义 CUDA 全局函数，指定其启动配置和线程限制，处理半精度数据的反向注意力后向传播任务
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64>::Params p);

// 定义 CUDA 全局函数，指定其启动配置和线程限制，处理半精度数据的反向注意力后向传播任务
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k128_seqaligned_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128, true>::Params p);

// 定义 CUDA 全局函数，指定其启动配置和线程限制，处理半精度数据的反向注意力后向传播任务
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>::kMinBlocksPerSm)
// 调用名为 `fmha_cutlassB_f16_aligned_128x64_k128_sm70` 的函数，并传入特定模板参数 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>::Params p`
fmha_cutlassB_f16_aligned_128x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>::Params p);

// 定义一个 CUDA 全局函数，使用 __launch_bounds__ 限定其启动参数
__global__ void __launch_bounds__(
    // 设置线程块的线程数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::kNumThreads`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    // 设置每个 SM 最小块数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)

// 调用名为 `fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm70` 的函数，并传入特定模板参数 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::Params p`
fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::Params p);

// 定义一个 CUDA 全局函数，使用 __launch_bounds__ 限定其启动参数
__global__ void __launch_bounds__(
    // 设置线程块的线程数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    // 设置每个 SM 最小块数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)

// 调用名为 `fmha_cutlassB_f16_aligned_64x64_k128_sm70` 的函数，并传入特定模板参数 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::Params p`
fmha_cutlassB_f16_aligned_64x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::Params p);

// 定义一个 CUDA 全局函数，使用 __launch_bounds__ 限定其启动参数
__global__ void __launch_bounds__(
    // 设置线程块的线程数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads,
    // 设置每个 SM 最小块数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)

// 调用名为 `fmha_cutlassB_f16_aligned_128x64_k65536_sm70` 的函数，并传入特定模板参数 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::Params p`
fmha_cutlassB_f16_aligned_128x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::Params p);

// 定义一个 CUDA 全局函数，使用 __launch_bounds__ 限定其启动参数
__global__ void __launch_bounds__(
    // 设置线程块的线程数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    // 设置每个 SM 最小块数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)

// 调用名为 `fmha_cutlassB_f16_aligned_64x64_k65536_sm70` 的函数，并传入特定模板参数 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p`
fmha_cutlassB_f16_aligned_64x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p);

// 定义一个 CUDA 全局函数，使用 __launch_bounds__ 限定其启动参数
__global__ void __launch_bounds__(
    // 设置线程块的线程数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads,
    // 设置每个 SM 最小块数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm)

// 调用名为 `fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm70` 的函数，并传入特定模板参数 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::Params p`
fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::Params p);

// 定义一个 CUDA 全局函数，使用 __launch_bounds__ 限定其启动参数
__global__ void __launch_bounds__(
    // 设置线程块的线程数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads`
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads,
    // 设置每个 SM 最小块数为 `AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64
    // 使用 Cutlass 库中的 AttentionBackwardKernel 类模板，针对具体的模板参数：
    // - 利用的架构是 Sm70（这通常指代 NVIDIA 的 GPU 架构，如Volta或Turing）
    // - 数据类型是 half_t，通常是半精度浮点数
    // - 模板参数为 <true, true, false, 128, 64, 128>，分别表示：
    //   - 是否支持 warp 级别的操作
    //   - 是否支持跨线程块的并发
    //   - 是否采用梯度检查（通常在开发和调试时使用）
    //   - 内存访问策略和块大小：128（warp_size）, 64（threadblock_shape_y）, 128（threadblock_shape_x）
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 128>::kMinBlocksPerSm
# 定义一个函数调用，调用名为 fmha_cutlassB_f16_aligned_128x64_k128_dropout_sm70 的函数，并传入特定的模板参数 AttentionBackwardKernel。
fmha_cutlassB_f16_aligned_128x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 128>::Params p);

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 进行限定：
#   - 设置线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads
#   - 设置每个 SM 最少需要的线程块数量为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
{
    # 定义一个函数调用，调用名为 fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm70 的函数，并传入特定的模板参数 AttentionBackwardKernel。
    fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);
}

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 进行限定：
#   - 设置线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads
#   - 设置每个 SM 最少需要的线程块数量为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
{
    # 定义一个函数调用，调用名为 fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm70 的函数，并传入特定的模板参数 AttentionBackwardKernel。
    fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p);
}

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 进行限定：
#   - 设置线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads
#   - 设置每个 SM 最少需要的线程块数量为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
{
    # 定义一个函数调用，调用名为 fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm70 的函数，并传入特定的模板参数 AttentionBackwardKernel。
    fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p);
}

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 进行限定：
#   - 设置线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads
#   - 设置每个 SM 最少需要的线程块数量为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
{
    # 定义一个函数调用，调用名为 fmha_cutlassB_f16_notaligned_64x64_k32_sm70 的函数，并传入特定的模板参数 AttentionBackwardKernel。
    fmha_cutlassB_f16_notaligned_64x64_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>::Params p);
}

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 进行限定：
#   - 设置线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads
#   - 设置每个 SM 最少需要的线程块数量为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
{
    # 定义一个函数调用，调用名为 fmha_cutlassB_f16_notaligned_64x64_k64_sm70 的函数，并传入特定的模板参数 AttentionBackwardKernel。
    fmha_cutlassB_f16_notaligned_64x64_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>::Params p);
}

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 进行限定：
#   - 设置线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>::kNumThreads
#   - 设置每个 SM 最少需要的线程块数量为 AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>::kMinBlocksPerSm
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>::kMinBlocksPerSm)
{
    # 定义一个函数调用，调用名为 fmha_cutlassB_f16_notaligned_128x64_k128_sm70 的函数，并传入特定的模板参数 AttentionBackwardKernel。
    fmha_cutlassB_f16_notaligned_128x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128,
    AttentionBackwardKernel<
        cutlass::arch::Sm70,  // 使用的 GPU 架构，这里是 Sm70
        cutlass::half_t,      // 使用的数据类型，这里是 half_t，即半精度浮点数
        false,                // 是否使用纹理内存，这里是 false
        false,                // 是否进行偏置加法，这里是 false
        false,                // 是否进行残差连接，这里是 false
        64,                   // 每个线程块的 X 维度大小，这里是 64
        64,                   // 每个线程块的 Y 维度大小，这里是 64
        128                   // 每个线程块的 Z 维度大小，这里是 128
    >::kMinBlocksPerSm       // 在当前 GPU 架构上每个 SM 的最小线程块数
# 调用名为 fmha_cutlassB_f16_notaligned_64x64_k128_sm70 的函数，并传入参数为 AttentionBackwardKernel 的模板参数对象 p
fmha_cutlassB_f16_notaligned_64x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 128>::Params p);

# 定义一个 CUDA 全局函数 __global__，并设置其启动限制为指定的线程数和每个 SM 最小块数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 65536>::kMinBlocksPerSm)
{
    # 调用名为 fmha_cutlassB_f16_notaligned_128x64_k65536_sm70 的函数，并传入参数为 AttentionBackwardKernel 的模板参数对象 p
    fmha_cutlassB_f16_notaligned_128x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 65536>::Params p);
}

# 调用名为 fmha_cutlassB_f16_notaligned_64x64_k65536_sm70 的函数，并传入参数为 AttentionBackwardKernel 的模板参数对象 p
fmha_cutlassB_f16_notaligned_64x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 65536>::Params p);

# 定义一个 CUDA 全局函数 __global__，并设置其启动限制为指定的线程数和每个 SM 最小块数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
{
    # 调用名为 fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm70 的函数，并传入参数为 AttentionBackwardKernel 的模板参数对象 p
    fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 32>::Params p);
}

# 定义一个 CUDA 全局函数 __global__，并设置其启动限制为指定的线程数和每个 SM 最小块数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
{
    # 调用名为 fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm70 的函数，并传入参数为 AttentionBackwardKernel 的模板参数对象 p
    fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 64>::Params p);
}

# 定义一个 CUDA 全局函数 __global__，并设置其启动限制为指定的线程数和每个 SM 最小块数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 128>::kMinBlocksPerSm)
{
    # 调用名为 fmha_cutlassB_f16_notaligned_128x64_k128_dropout_sm70 的函数，并传入参数为 AttentionBackwardKernel 的模板参数对象 p
    fmha_cutlassB_f16_notaligned_128x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 128>::Params p);
}

# 定义一个 CUDA 全局函数 __global__，并设置其启动限制为指定的线程数和每个 SM 最小块数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
{
    # 调用名为 fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm70 的函数，并传入参数为 AttentionBackwardKernel 的模板参数对象 p
    fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::Params p);
}

# 定义一个 CUDA 全局函数 __global__，并设置其启动限制为指定的线程数和每个 SM 最小块数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>::kMinBlocksPerSm)
{
    # 使用 Cutlass 库的 AttentionBackwardKernel 模板类，并指定以下参数：
    # - cutlass::arch::Sm70: 指定了使用的 GPU 架构为 Sm70（这里是 NVIDIA GPU 的架构代号）
    # - cutlass::half_t: 使用 half 精度数据类型
    # - false: 不进行 transpose 操作
    # - true: 计算过程中进行尺寸调整（padding）
    # - false: 不进行 split-k 操作
    # - 128: 矩阵乘法的 M 维度（高度维度）大小为 128
    # - 64: 矩阵乘法的 N 维度（宽度维度）大小为 64
    # - 65536: 每个 SM（流多处理器）所需的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>::kMinBlocksPerSm)
// 声明一个函数原型，函数名为 fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm70，参数类型为 AttentionBackwardKernel 的模板实例化对象的 Params 类型 p
fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>::Params p);

// 声明一个 CUDA 全局函数，使用 __launch_bounds__ 来设置其启动时的限制条件
__global__ void __launch_bounds__(
    // 设置每个线程块的最大线程数为 AttentionBackwardKernel 模板实例化对象的 kNumThreads
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 65536>::kNumThreads,
    // 设置每个 SM 最少需要的线程块数为 AttentionBackwardKernel 模板实例化对象的 kMinBlocksPerSm
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)

// 声明一个函数原型，函数名为 fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm70，参数类型为 AttentionBackwardKernel 的模板实例化对象的 Params 类型 p
fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 65536>::Params p);

// 声明一个模板函数 dispatch_cutlassB_f16_sm70，接受一个函数对象 cb 和一个整数 cc 作为参数
template <typename T> void dispatch_cutlassB_f16_sm70(T cb, int cc) {
    // 调用函数对象 cb，传入一个具体的 AttentionBackwardKernel 实例作为参数，并调用名为 fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm70 的函数
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32, true>(), fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm70);
    // 同上，调用不同的参数组合
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64, true>(), fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128, true>(), fmha_cutlassB_f16_aligned_128x64_k128_seqaligned_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>(), fmha_cutlassB_f16_aligned_128x64_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>(), fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 128>(), fmha_cutlassB_f16_aligned_128x64_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm70);
}
    // 调用 AttentionBackwardKernel 模板，生成并执行反向注意力机制的计算，使用的设备架构为 Sm70，数据类型为 half_t
    // 参数说明（顺序）：是否需要参考解，是否开启dropout，是否对齐数据，线程块大小为128x64，每个线程块处理的问题数量为65536
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm70);
    
    // 同上，线程块大小为64x64
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm70);
    
    // 使用 Sm70 设备架构，half_t 数据类型，不对齐数据，无参考解，无dropout，线程块大小为64x64，每个线程块处理的问题数量为32
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_sm70);
    
    // 同上，每个线程块处理的问题数量为64
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_sm70);
    
    // 同上，线程块大小为128x64，每个线程块处理的问题数量为128
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>(), fmha_cutlassB_f16_notaligned_128x64_k128_sm70);
    
    // 同上，线程块大小为64x64，每个线程块处理的问题数量为128
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_sm70);
    
    // 同上，线程块大小为128x64，每个线程块处理的问题数量为65536
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_notaligned_128x64_k65536_sm70);
    
    // 同上，线程块大小为64x64，每个线程块处理的问题数量为65536
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_sm70);
    
    // 同上，使用 Sm70 设备架构，half_t 数据类型，不对齐数据，需要参考解，开启dropout，线程块大小为64x64，每个线程块处理的问题数量为32
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm70);
    
    // 同上，每个线程块处理的问题数量为64
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm70);
    
    // 同上，线程块大小为128x64，每个线程块处理的问题数量为128
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 128>(), fmha_cutlassB_f16_notaligned_128x64_k128_dropout_sm70);
    
    // 同上，线程块大小为64x64，每个线程块处理的问题数量为128
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm70);
    
    // 同上，线程块大小为128x64，每个线程块处理的问题数量为65536
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm70);
    
    // 同上，线程块大小为64x64，每个线程块处理的问题数量为65536
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm70);
// ======== bf16 / sm80 ========

// 启动全局 CUDA 核函数，设置线程块的限制为指定数量的线程和最小块数
__global__ void __launch_bounds__(
    // 设置线程块的大小为特定配置下的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::kNumThreads,
    // 设置每个流多处理器（SM）上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::kMinBlocksPerSm)
// 定义一个 CUDA 核函数，用于 bfloat16 类型的注意力反向传播，采用特定的线程块大小和对齐方式，适用于 SM80 架构
fmha_cutlassB_bf16_aligned_64x64_k32_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::Params p);

// 启动全局 CUDA 核函数，设置线程块的限制为指定数量的线程
__global__ void __launch_bounds__(
    // 设置线程块的大小为特定配置下的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::kNumThreads,
    // 设置每个流多处理器（SM）上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::kMinBlocksPerSm)
// 定义一个 CUDA 核函数，用于 bfloat16 类型的注意力反向传播，采用特定的线程块大小，适用于 SM80 架构
fmha_cutlassB_bf16_aligned_64x64_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::Params p);

// 启动全局 CUDA 核函数，设置线程块的限制为指定数量的线程和最小块数
__global__ void __launch_bounds__(
    // 设置线程块的大小为特定配置下的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::kNumThreads,
    // 设置每个流多处理器（SM）上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::kMinBlocksPerSm)
// 定义一个 CUDA 核函数，用于 bfloat16 类型的注意力反向传播，采用特定的线程块大小和对齐方式，适用于 SM80 架构
fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::Params p);

// 启动全局 CUDA 核函数，设置线程块的限制为指定数量的线程
__global__ void __launch_bounds__(
    // 设置线程块的大小为特定配置下的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::kNumThreads,
    // 设置每个流多处理器（SM）上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::kMinBlocksPerSm)
// 定义一个 CUDA 核函数，用于 bfloat16 类型的注意力反向传播，采用特定的线程块大小，适用于 SM80 架构
fmha_cutlassB_bf16_aligned_64x64_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::Params p);

// 启动全局 CUDA 核函数，设置线程块的限制为指定数量的线程
__global__ void __launch_bounds__(
    // 设置线程块的大小为特定配置下的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kNumThreads,
    // 设置每个流多处理器（SM）上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
// 定义一个 CUDA 核函数，用于 bfloat16 类型的注意力反向传播，采用特定的线程块大小，适用于 SM80 架构
fmha_cutlassB_bf16_aligned_128x64_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::Params p);

// 启动全局 CUDA 核函数，设置线程块的限制为指定数量的线程和最小块数
__global__ void __launch_bounds__(
    // 设置线程块的大小为特定配置下的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::kNumThreads,
    // 设置每个流多处理器（SM）上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::kMinBlocksPerSm)
// 定义一个 CUDA 核函数，用于 bfloat16 类型的注意力反向传播，采用特定的线程块大小和对齐方式，适用于 SM80 架构
fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::Params p);

// 启动全局 CUDA 核函数，设置线程块的限制为指定数量的线程
__global__ void __launch_bounds__(
    // 设置线程块的大小为特定配置下的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::kNumThreads,
    // 设置每个流多处理器（SM）上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::kMinBlocksPerSm)
// 定义一个 CUDA 核函数，用于 bfloat16 类型的注意力反向传播，采用特定的线程块大小，适用于 SM80 架构
# 调用一个名为 fmha_cutlassB_bf16_aligned_128x128_k128_sm80 的函数，并传递指定的模板参数
fmha_cutlassB_bf16_aligned_128x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::Params p);

# 定义一个 CUDA 全局函数 __global__，指定其线程和块的启动限制
__global__ void __launch_bounds__(
    # 设定线程数量为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    # 设定每个 SM 上的最小块数为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)
# 调用一个名为 fmha_cutlassB_bf16_aligned_64x64_k128_seqaligned_sm80 的函数，并传递指定的模板参数
fmha_cutlassB_bf16_aligned_64x64_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::Params p);

# 定义一个 CUDA 全局函数 __global__，指定其线程和块的启动限制
__global__ void __launch_bounds__(
    # 设定线程数量为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::kNumThreads,
    # 设定每个 SM 上的最小块数为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
# 调用一个名为 fmha_cutlassB_bf16_aligned_64x64_k128_sm80 的函数，并传递指定的模板参数
fmha_cutlassB_bf16_aligned_64x64_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::Params p);

# 定义一个 CUDA 全局函数 __global__，指定其线程和块的启动限制
__global__ void __launch_bounds__(
    # 设定线程数量为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::kNumThreads,
    # 设定每个 SM 上的最小块数为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
# 调用一个名为 fmha_cutlassB_bf16_aligned_128x64_k65536_sm80 的函数，并传递指定的模板参数
fmha_cutlassB_bf16_aligned_128x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::Params p);

# 定义一个 CUDA 全局函数 __global__，指定其线程和块的启动限制
__global__ void __launch_bounds__(
    # 设定线程数量为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::kNumThreads,
    # 设定每个 SM 上的最小块数为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
# 调用一个名为 fmha_cutlassB_bf16_aligned_64x64_k65536_sm80 的函数，并传递指定的模板参数
fmha_cutlassB_bf16_aligned_64x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::Params p);

# 定义一个 CUDA 全局函数 __global__，指定其线程和块的启动限制
__global__ void __launch_bounds__(
    # 设定线程数量为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::kNumThreads,
    # 设定每个 SM 上的最小块数为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::kMinBlocksPerSm)
# 调用一个名为 fmha_cutlassB_bf16_aligned_64x64_k32_dropout_sm80 的函数，并传递指定的模板参数
fmha_cutlassB_bf16_aligned_64x64_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::Params p);

# 定义一个 CUDA 全局函数 __global__，指定其线程和块的启动限制
__global__ void __launch_bounds__(
    # 设定线程数量为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::kNumThreads,
    # 设定每个 SM 上的最小块数为 AttentionBackwardKernel 在 Sm80 架构、bfloat16 数据类型下的数值
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::kMinBlocksPerSm)
# 调用一个名为 fmha_cutlassB_bf16_aligned_64x64_k64_dropout_sm80 的函数，并传递指定的模板参数
fmha_cutlassB_bf16_aligned_64x64_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::Params p);
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm)



    // 调用名为 AttentionBackwardKernel 的模板类，使用以下参数：
    // - GPU 架构为 Sm80
    // - 数据类型为 bfloat16_t
    // - 启用模板参数为 true
    // - 模板中的三个 bool 参数都设置为 true
    // - 后续的三个参数分别为 128, 128, 128
    // 从模板类的常量成员 kMinBlocksPerSm 中获取最小块数值
fmha_cutlassB_bf16_aligned_128x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::Params p);

# 定义了四个全局函数声明和一个模板函数

template <typename T> void dispatch_cutlassB_bf16_sm80(T cb, int cc) {
    # 调用传入的回调函数cb，以不同的参数类型和模板参数调用具体的函数，这些函数的命名和参数描述表明它们与注意力机制的后向计算有关
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>(), fmha_cutlassB_bf16_aligned_64x64_k32_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32>(), fmha_cutlassB_bf16_aligned_64x64_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>(), fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64>(), fmha_cutlassB_bf16_aligned_64x64_k64_sm80);
    # 如果cc等于86或者89，调用函数，否则跳过
    if (cc == 86 || cc == 89) cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>(), fmha_cutlassB_bf16_aligned_128x64_k96_sm80);
    # 调用函数，参数包含不同的模板参数和函数名，用于后续的计算过程
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>(), fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>(), fmha_cutlassB_bf16_aligned_128x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>(), fmha_cutlassB_bf16_aligned_64x64_k128_seqaligned_sm80);
}
    # 调用 AttentionBackwardKernel 类，并传入指定的模板参数，生成实例并执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_bf16_aligned_64x64_k128_sm80);
    
    # 调用 AttentionBackwardKernel 类的另一个实例，并传入不同的模板参数，执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_bf16_aligned_128x64_k65536_sm80);
    
    # 继续调用 AttentionBackwardKernel 类的实例，以不同的模板参数执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_bf16_aligned_64x64_k65536_sm80);
    
    # 再次调用 AttentionBackwardKernel 类的实例，传入带有dropout的模板参数，执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 32>(), fmha_cutlassB_bf16_aligned_64x64_k32_dropout_sm80);
    
    # 继续调用 AttentionBackwardKernel 类的实例，使用不同的dropout参数执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 64>(), fmha_cutlassB_bf16_aligned_64x64_k64_dropout_sm80);
    
    # 再次调用 AttentionBackwardKernel 类的实例，传入更大的模板参数，并带有dropout设置，执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>(), fmha_cutlassB_bf16_aligned_128x128_k128_dropout_sm80);
    
    # 继续调用 AttentionBackwardKernel 类的实例，传入带有dropout设置的模板参数，执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_bf16_aligned_64x64_k128_dropout_sm80);
    
    # 再次调用 AttentionBackwardKernel 类的实例，使用不同的模板参数和dropout设置，执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_bf16_aligned_128x64_k65536_dropout_sm80);
    
    # 继续调用 AttentionBackwardKernel 类的实例，以另一组不同的模板参数和dropout设置执行回调函数 cb()
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_bf16_aligned_64x64_k65536_dropout_sm80);
// 结束前一段代码的大括号，开始定义新的代码段

// 使用特定的启动参数限制来调用 GPU 上的 Kernel 函数，这些参数指定了每个线程块的线程数量和每个 SM 上的最小块数。
// 以下是针对不同输入尺寸和对齐方式的不同 Kernel 函数的定义和调用。

// 在具有 SM80 架构和半精度数据类型的情况下，调用 AttentionBackwardKernel，使用特定的线程块大小和对齐方式
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32, true>::Params p);

// 在具有 SM80 架构和半精度数据类型的情况下，调用 AttentionBackwardKernel，使用特定的线程块大小但不指定对齐方式
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32>::Params p);

// 在具有 SM80 架构和半精度数据类型的情况下，调用 AttentionBackwardKernel，使用更大的线程块大小和对齐方式
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64, true>::Params p);

// 在具有 SM80 架构和半精度数据类型的情况下，调用 AttentionBackwardKernel，使用更大的线程块大小但不指定对齐方式
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64>::Params p);

// 在具有 SM80 架构和半精度数据类型的情况下，调用 AttentionBackwardKernel，使用更大的线程块大小和对齐方式，且使用不同的输入尺寸
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::Params p);

// 在具有 SM80 架构和半精度数据类型的情况下，调用 AttentionBackwardKernel，使用更大的线程块大小和对齐方式，且指定了对齐方式
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::Params p);

// 在具有 SM80 架构和半精度数据类型的情况下，调用 AttentionBackwardKernel，使用更大的线程块大小但不指定对齐方式
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::kMinBlocksPerSm)
// 声明一个函数，该函数调用了名为 "fmha_cutlassB_f16_aligned_128x128_k128_sm80" 的模板函数，其参数类型为 cutlass::arch::Sm80，数据类型为 cutlass::half_t，参数分别为 true, false, true, 128, 128, 128 的结构体 Params。
fmha_cutlassB_f16_aligned_128x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数为具体的线程块和最小 SM 单元数，该函数由名为 "AttentionBackwardKernel" 的模板生成，参数类型和值与上述类似，但不同之处在于数据大小为 64, 64, 128，且支持序列对齐。
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)
{
    // CUDA Kernel 内容被省略
}

// 声明一个函数，该函数调用了名为 "fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm80" 的模板函数，其参数类型为 cutlass::arch::Sm80，数据类型为 cutlass::half_t，参数分别为 true, false, false, 64, 64, 128 的结构体 Params。
fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128, true>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数为具体的线程块和最小 SM 单元数，该函数由名为 "AttentionBackwardKernel" 的模板生成，参数类型和值与上述类似，但不支持序列对齐。
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
{
    // CUDA Kernel 内容被省略
}

// 声明一个函数，该函数调用了名为 "fmha_cutlassB_f16_aligned_64x64_k128_sm80" 的模板函数，其参数类型为 cutlass::arch::Sm80，数据类型为 cutlass::half_t，参数分别为 true, false, false, 64, 64, 128 的结构体 Params。
fmha_cutlassB_f16_aligned_64x64_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数为具体的线程块和最小 SM 单元数，该函数由名为 "AttentionBackwardKernel" 的模板生成，参数类型和值与上述类似，但数据大小为 128, 64, 65536。
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
{
    // CUDA Kernel 内容被省略
}

// 声明一个函数，该函数调用了名为 "fmha_cutlassB_f16_aligned_64x64_k65536_sm80" 的模板函数，其参数类型为 cutlass::arch::Sm80，数据类型为 cutlass::half_t，参数分别为 true, false, false, 64, 64, 65536 的结构体 Params。
fmha_cutlassB_f16_aligned_64x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数为具体的线程块和最小 SM 单元数，该函数由名为 "AttentionBackwardKernel" 的模板生成，参数类型和值与上述类似，但数据大小为 64, 64, 65536。
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 32>::kMinBlocksPerSm)
{
    // CUDA Kernel 内容被省略
}

// 声明一个函数，该函数调用了名为 "fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm80" 的模板函数，其参数类型为 cutlass::arch::Sm80，数据类型为 cutlass::half_t，参数分别为 true, true, true, 64, 64, 32 的结构体 Params。
fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 32>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数为具体的线程块和最小 SM 单元数，该函数由名为 "AttentionBackwardKernel" 的模板生成，参数类型和值与上述类似，但数据大小为 64, 64, 64。
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 64>::kMinBlocksPerSm)
{
    // CUDA Kernel 内容被省略
}

// 继续声明下一个函数，由于空间限制，注释被截断
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm)
{
    // CUDA Kernel 内容被省略
}
    // 使用 AttentionBackwardKernel 模板，定义了一个具体的模板实例化：
    // - 模板参数：
    //   - cutlass::arch::Sm80: 表示目标计算架构是 NVIDIA 的 Ampere 架构的 SM80。
    //   - cutlass::half_t: 表示数据类型为半精度浮点数。
    //   - true, true, true: 表示模板中的布尔参数，具体含义需要查看模板定义来理解。
    //   - 128, 128, 128: 表示模板中的整数参数，具体含义需要查看模板定义来理解。
    // - 此处常量表达式计算了 AttentionBackwardKernel 在给定 SM80 架构下的最小块数。
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm
// 声明一个函数原型，参数为 AttentionBackwardKernel 类型的对象，返回类型为 void
fmha_cutlassB_f16_aligned_128x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::Params p);

// 定义一个 CUDA 核函数，设置其启动参数为指定数值
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)

// 声明一个函数原型，参数为 AttentionBackwardKernel 类型的对象，返回类型为 void
fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);

// 定义一个 CUDA 核函数，设置其启动参数为指定数值
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)

// 声明一个函数原型，参数为 AttentionBackwardKernel 类型的对象，返回类型为 void
fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p);

// 定义一个 CUDA 核函数，设置其启动参数为指定数值
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)

// 定义一个模板函数，接受类型 T 作为参数，返回类型为 void
template <typename T> void dispatch_cutlassB_f16_sm80(T cb, int cc) {
    // 调用回调函数 cb，传递一个具体的 AttentionBackwardKernel 实例和相应的函数指针 fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm80
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32, true>(), fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm80);
    // 同上，不过传递的是另一种实例和函数指针
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_sm80);
    // 继续传递其他实例和函数指针的调用
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64, true>(), fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_sm80);
    // 根据条件选择性地调用函数
    if (cc == 86 || cc == 89) cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>(), fmha_cutlassB_f16_aligned_128x64_k96_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>(), fmha_cutlassB_f16_aligned_128x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>(), fmha_cutlassB_f16_aligned_128x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128, true>(), fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_sm80);
}
    # 调用 AttentionBackwardKernel 类并传入参数，执行反向注意力机制的计算
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_sm80);
    
    # 调用 AttentionBackwardKernel 类并传入参数，执行反向注意力机制的计算
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_sm80);
    
    # 调用 AttentionBackwardKernel 类并传入参数，执行反向注意力机制的计算（包含 dropout 功能）
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm80);
    
    # 调用 AttentionBackwardKernel 类并传入参数，执行反向注意力机制的计算（包含 dropout 功能）
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm80);
    
    # 调用 AttentionBackwardKernel 类并传入参数，执行反向注意力机制的计算（包含 dropout 功能）
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>(), fmha_cutlassB_f16_aligned_128x128_k128_dropout_sm80);
    
    # 调用 AttentionBackwardKernel 类并传入参数，执行反向注意力机制的计算（包含 dropout 功能）
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm80);
    
    # 调用 AttentionBackwardKernel 类并传入参数，执行反向注意力机制的计算（包含 dropout 功能）
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm80);
    
    # 调用 AttentionBackwardKernel 类并传入参数，执行反向注意力机制的计算（包含 dropout 功能）
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm80);
// ======== f16 / sm50 ========

// 定义 CUDA 全局函数，使用 __launch_bounds__ 来设置每个线程块的线程数和每个 SM 的最小块数限制
__global__ void __launch_bounds__(
    // 使用 cutlass 库的 AttentionBackwardKernel 模板，选择架构为 Sm50，数据类型为 half_t
    // 参数为：是否启用向量化 true，是否进行 dropout true，是否支持重用 true，
    // 线程块大小 64x64，线程块内部的行数 32
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    // 同上，但指定每个 SM 的最小块数限制
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_aligned_64x64_k32_sm50，参数类型为 AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>::Params
fmha_cutlassB_f16_aligned_64x64_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>::Params p);

__global__ void __launch_bounds__(
    // 同上，但线程块内部的行数为 64
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_aligned_64x64_k64_sm50，参数类型为 AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>::Params
fmha_cutlassB_f16_aligned_64x64_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>::Params p);

__global__ void __launch_bounds__(
    // 同上，但线程块内部的行数为 128
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_aligned_64x64_k128_sm50，参数类型为 AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>::Params
fmha_cutlassB_f16_aligned_64x64_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>::Params p);

__global__ void __launch_bounds__(
    // 同上，但线程块内部的行数为 65536
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_aligned_64x64_k65536_sm50，参数类型为 AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>::Params
fmha_cutlassB_f16_aligned_64x64_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p);

__global__ void __launch_bounds__(
    // 同上，但启用 dropout
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm50，参数类型为 AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>::Params
fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>::Params p);

__global__ void __launch_bounds__(
    // 同上，但线程块内部的行数为 64
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm50，参数类型为 AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>::Params
fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>::Params p);

__global__ void __launch_bounds__(
    // 同上，但线程块内部的行数为 128
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm50，参数类型为 AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>::Params
fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);
// 声明一个函数，函数名为 fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm50，参数为一个特定模板实例的类型参数 p
fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);

// 定义一个 CUDA 核函数，使用 __launch_bounds__ 限定其线程块的大小
__global__ void __launch_bounds__(
    // 线程块的线程数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    // 每个 SM 最少需要的线程块数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
// 声明一个函数，函数名为 fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm50，参数为一个特定模板实例的类型参数 p
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p);

// 定义一个 CUDA 核函数，使用 __launch_bounds__ 限定其线程块的大小
__global__ void __launch_bounds__(
    // 线程块的线程数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads,
    // 每个 SM 最少需要的线程块数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
// 声明一个函数，函数名为 fmha_cutlassB_f16_notaligned_64x64_k32_sm50，参数为一个特定模板实例的类型参数 p
fmha_cutlassB_f16_notaligned_64x64_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 32>::Params p);

// 定义一个 CUDA 核函数，使用 __launch_bounds__ 限定其线程块的大小
__global__ void __launch_bounds__(
    // 线程块的线程数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads,
    // 每个 SM 最少需要的线程块数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
// 声明一个函数，函数名为 fmha_cutlassB_f16_notaligned_64x64_k64_sm50，参数为一个特定模板实例的类型参数 p
fmha_cutlassB_f16_notaligned_64x64_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 64>::Params p);

// 定义一个 CUDA 核函数，使用 __launch_bounds__ 限定其线程块的大小
__global__ void __launch_bounds__(
    // 线程块的线程数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 128>::kNumThreads,
    // 每个 SM 最少需要的线程块数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
// 声明一个函数，函数名为 fmha_cutlassB_f16_notaligned_64x64_k128_sm50，参数为一个特定模板实例的类型参数 p
fmha_cutlassB_f16_notaligned_64x64_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 128>::Params p);

// 定义一个 CUDA 核函数，使用 __launch_bounds__ 限定其线程块的大小
__global__ void __launch_bounds__(
    // 线程块的线程数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 65536>::kNumThreads,
    // 每个 SM 最少需要的线程块数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
// 声明一个函数，函数名为 fmha_cutlassB_f16_notaligned_64x64_k65536_sm50，参数为一个特定模板实例的类型参数 p
fmha_cutlassB_f16_notaligned_64x64_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 65536>::Params p);

// 定义一个 CUDA 核函数，使用 __launch_bounds__ 限定其线程块的大小
__global__ void __launch_bounds__(
    // 线程块的线程数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 32>::kNumThreads,
    // 每个 SM 最少需要的线程块数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
// 声明一个函数，函数名为 fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm50，参数为一个特定模板实例的类型参数 p
fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 32>::Params p);

// 继续定义一个 CUDA 核函数，使用 __launch_bounds__ 限定其线程块的大小
__global__ void __launch_bounds__(
    // 线程块的线程数从模板参数中获取
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
    # 使用 cutlass 库中的 AttentionBackwardKernel 模板类，具体模板参数如下：
    # - cutlass::arch::Sm50：指定 GPU 架构为 Sm50
    # - cutlass::half_t：指定数据类型为 half_t
    # - false, true, false：分别对应模板类的三个布尔类型参数
    # - 64, 64, 64：指定模板类的三个整数类型参数
    # 调用该模板类的静态成员常量 kMinBlocksPerSm，表示最小的每个 Streaming Multiprocessor (SM) 上的块数。
// 声明一个函数原型，函数名为 fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm50，
// 参数为一个模板化的类型 AttentionBackwardKernel，使用的是 cutlass 库的 Sm50 架构，
// 数据类型为 half_t（半精度浮点数），不进行对齐，使用 64x64 的矩阵尺寸，dropout 率为 50%
// 函数返回类型为 void，参数 p 为 AttentionBackwardKernel 的参数对象
fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 64>::Params p);

// 声明一个 CUDA 核函数 __global__，函数名为 __launch_bounds__，
// 使用指定的线程限制参数，以确保在执行时可以优化资源利用率
__global__ void __launch_bounds__(
    // 指定核函数的线程数为 AttentionBackwardKernel 类型实例化后的 kNumThreads 值
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 128>::kNumThreads,
    // 指定核函数在 SM 中的最小块数为 AttentionBackwardKernel 实例化后的 kMinBlocksPerSm 值
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm50，
// 参数同样为一个模板化的类型 AttentionBackwardKernel，使用的是 cutlass 库的 Sm50 架构，
// 数据类型为 half_t（半精度浮点数），不进行对齐，使用 64x64 的矩阵尺寸，dropout 率为 50%
// 函数返回类型为 void，参数 p 为 AttentionBackwardKernel 的参数对象
fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 128>::Params p);

// 声明一个 CUDA 核函数 __global__，函数名为 __launch_bounds__，
// 使用指定的线程限制参数，以确保在执行时可以优化资源利用率
__global__ void __launch_bounds__(
    // 指定核函数的线程数为 AttentionBackwardKernel 类型实例化后的 kNumThreads 值
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 65536>::kNumThreads,
    // 指定核函数在 SM 中的最小块数为 AttentionBackwardKernel 实例化后的 kMinBlocksPerSm 值
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)
// 函数名为 fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm50，
// 参数同样为一个模板化的类型 AttentionBackwardKernel，使用的是 cutlass 库的 Sm50 架构，
// 数据类型为 half_t（半精度浮点数），不进行对齐，使用 64x64 的矩阵尺寸，dropout 率为 50%
// 函数返回类型为 void，参数 p 为 AttentionBackwardKernel 的参数对象
fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 65536>::Params p);

// 定义一个模板函数 dispatch_cutlassB_f16_sm50，
// 接受一个函数对象 cb 和一个整数 cc 作为参数
template <typename T> void dispatch_cutlassB_f16_sm50(T cb, int cc) {
    // 调用 cb 函数，参数为指定的 AttentionBackwardKernel 类型实例和相应的函数名
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_sm50);
}
    # 调用 AttentionBackwardKernel 函数，使用指定的模板参数和参数列表
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm50);
    # 调用 AttentionBackwardKernel 函数，使用指定的模板参数和参数列表
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm50);
    # 调用 AttentionBackwardKernel 函数，使用指定的模板参数和参数列表
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm50);
    # 调用 AttentionBackwardKernel 函数，使用指定的模板参数和参数列表
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm50);
// 结束前一个代码块的大括号

// ======== f32 / sm50 ========

// 使用CUDA的全局函数声明，设置函数启动的线程块大小限制和最小块数限制
__global__ void __launch_bounds__(
    // 设置线程块的数量为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 32>::kNumThreads,
    // 设置每个SM最少块数为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
// 函数原型声明，函数名为fmha_cutlassB_f32_aligned_64x64_k32_sm50，参数为模板参数的结构体
fmha_cutlassB_f32_aligned_64x64_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 32>::Params p);

// 使用CUDA的全局函数声明，设置函数启动的线程块大小限制和最小块数限制
__global__ void __launch_bounds__(
    // 设置线程块的数量为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 64>::kNumThreads,
    // 设置每个SM最少块数为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
// 函数原型声明，函数名为fmha_cutlassB_f32_aligned_64x64_k64_sm50，参数为模板参数的结构体
fmha_cutlassB_f32_aligned_64x64_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 64>::Params p);

// 使用CUDA的全局函数声明，设置函数启动的线程块大小限制和最小块数限制
__global__ void __launch_bounds__(
    // 设置线程块的数量为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 128>::kNumThreads,
    // 设置每个SM最少块数为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
// 函数原型声明，函数名为fmha_cutlassB_f32_aligned_64x64_k128_sm50，参数为模板参数的结构体
fmha_cutlassB_f32_aligned_64x64_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 128>::Params p);

// 使用CUDA的全局函数声明，设置函数启动的线程块大小限制和最小块数限制
__global__ void __launch_bounds__(
    // 设置线程块的数量为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 65536>::kNumThreads,
    // 设置每个SM最少块数为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
// 函数原型声明，函数名为fmha_cutlassB_f32_aligned_64x64_k65536_sm50，参数为模板参数的结构体
fmha_cutlassB_f32_aligned_64x64_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 65536>::Params p);

// 检查CUDA版本和条件，如果满足条件则执行以下代码块
#if defined(CUDA_VERSION) && CUDA_VERSION == 12040 && !defined(USE_ROCM)
// 使用CUDA的全局函数声明，设置函数启动的线程块大小限制和最小块数限制
__global__ void __launch_bounds__(
    // 设置线程块的数量为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 32>::kNumThreads,
    // 设置每个SM最少块数为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 32>::kMinBlocksPerSm)
// 函数原型声明，函数名为fmha_cutlassB_f32_aligned_32x32_k32_dropout_sm50，参数为模板参数的结构体
fmha_cutlassB_f32_aligned_32x32_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 32>::Params p);

// 使用CUDA的全局函数声明，设置函数启动的线程块大小限制和最小块数限制
__global__ void __launch_bounds__(
    // 设置线程块的数量为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 64>::kNumThreads,
    // 设置每个SM最少块数为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 64>::kMinBlocksPerSm)
// 函数原型声明，函数名为fmha_cutlassB_f32_aligned_32x32_k64_dropout_sm50，参数为模板参数的结构体
fmha_cutlassB_f32_aligned_32x32_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 64>::Params p);
// 否则执行以下代码块
#else
// 使用CUDA的全局函数声明，设置函数启动的线程块大小限制和最小块数限制
__global__ void __launch_bounds__(
    // 设置线程块的数量为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 32>::kNumThreads,
    // 设置每个SM最少块数为特定模板参数值
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
// 函数原型声明，函数名为fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm50，参数为模板参数的结构体
fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 32>::Params p);
// 结束前一个代码块的大括号
#endif
    # 使用Cutlass库中的AttentionBackwardKernel模板，配置为运行在Sm50架构上，处理单精度浮点数数据，
    # 其他参数配置为：反向传播模式为真，使用张量核心，不支持alpha与beta混合精度计算，线程块尺寸为64x64，
    # warp大小为64。
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 64>::kNumThreads,
    
    # 同上，但是获取的是每个SM上最小的线程块数量。
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
// 声明一个函数 fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm50，接受一个 AttentionBackwardKernel 类型的参数 p
fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 64>::Params p);

// 定义一个 CUDA 全局函数 __global__，限制其启动条件为一组特定的线程数和最小块数，该函数名称为 fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm50
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 128>::Params p);

// 定义一个 CUDA 全局函数 __global__，限制其启动条件为一组特定的线程数和最小块数，该函数名称为 fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm50
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 65536>::Params p);

// 定义一个 CUDA 全局函数 __global__，限制其启动条件为一组特定的线程数和最小块数，该函数名称为 fmha_cutlassB_f32_notaligned_64x64_k32_sm50
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 32>::Params p);

// 定义一个 CUDA 全局函数 __global__，限制其启动条件为一组特定的线程数和最小块数，该函数名称为 fmha_cutlassB_f32_notaligned_64x64_k64_sm50
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 64>::Params p);

// 定义一个 CUDA 全局函数 __global__，限制其启动条件为一组特定的线程数和最小块数，该函数名称为 fmha_cutlassB_f32_notaligned_64x64_k128_sm50
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 128>::Params p);

// 定义一个 CUDA 全局函数 __global__，限制其启动条件为一组特定的线程数和最小块数，该函数名称为 fmha_cutlassB_f32_notaligned_64x64_k65536_sm50
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 65536>::Params p);

// 定义一个 CUDA 全局函数 __global__，限制其启动条件为一组特定的线程数和最小块数，该函数名称为 fmha_cutlassB_f32_notaligned_64x64_k32_dropout_sm50
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 32>::Params p);
    # 使用 Cutlass 库中的 AttentionBackwardKernel 类模板，指定模板参数为：
    # - cutlass::arch::Sm50：CUDA 架构类型为 Sm50
    # - float：数据类型为 float
    # - false, true, false：模板中的布尔类型参数
    # - 64, 64, 64：模板中的整数参数
    # 访问 AttentionBackwardKernel 类的静态成员常量 kNumThreads 和 kMinBlocksPerSm
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
# 定义函数原型，声明一个接受指定参数的函数，返回类型为void
fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 64>::Params p);

# 定义 CUDA 核函数，设置其启动参数为指定数量的线程和每个 SM 最小块数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 128>::Params p);

# 定义 CUDA 核函数，设置其启动参数为指定数量的线程和每个 SM 最小块数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_notaligned_64x64_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 65536>::Params p);

# 定义模板函数，接受一个类型为 T 的参数 cb 和一个整数 cc，返回类型为 void
template <typename T> void dispatch_cutlassB_f32_sm50(T cb, int cc) {
    # 调用 cb 函数，传入不同的 AttentionBackwardKernel 参数，以及对应的函数指针
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_sm50);
    # 如果 CUDA 版本为 12040，且未定义 USE_ROCM 宏
    # 调用对应的函数指针，传入特定参数的 AttentionBackwardKernel
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm50);
#else
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm50);
#endif
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 32>(), fmha_cutlassB_f32_notaligned_64x64_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 64>(), fmha_cutlassB_f32_notaligned_64x64_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 128>(), fmha_cutlassB_f32_notaligned_64x64_k128_sm50);
}
    // 调用 AttentionBackwardKernel 模板实例化的函数 cb，用于反向注意力计算
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, false, false, 64, 64, 65536>(), fmha_cutlassB_f32_notaligned_64x64_k65536);
    
    // 调用 AttentionBackwardKernel 模板实例化的函数 cb，用于反向注意力计算，带有非对齐特性和 32 个内核的 dropout
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 32>(), fmha_cutlassB_f32_notaligned_64x64_k32_dropout_sm50);
    
    // 调用 AttentionBackwardKernel 模板实例化的函数 cb，用于反向注意力计算，带有非对齐特性和 64 个内核的 dropout
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 64>(), fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm50);
    
    // 调用 AttentionBackwardKernel 模板实例化的函数 cb，用于反向注意力计算，带有非对齐特性和 128 个内核的 dropout
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 128>(), fmha_cutlassB_f32_notaligned_64x64_k128_dropout_sm50);
    
    // 调用 AttentionBackwardKernel 模板实例化的函数 cb，用于反向注意力计算，带有非对齐特性和 65536 个内核的 dropout
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, float, false, true, false, 64, 64, 65536>(), fmha_cutlassB_f32_notaligned_64x64_k65536_dropout_sm50);
// 结束上一个代码块的语法
}

// ======== f32 / sm70 ========

// 使用 CUDA __global__ 关键字定义一个 GPU 上的核函数，指定线程束的限制
__global__ void __launch_bounds__(
    // 设置线程束的上限为特定类型和参数的注意力机制后向传播核函数所需的线程数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 32>::kNumThreads,
    // 设置每个 SM 上执行该核函数的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
// 函数原型声明，调用由 cutlass 库定义的具体函数
fmha_cutlassB_f32_aligned_64x64_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 32>::Params p);

// 使用 CUDA __global__ 关键字定义一个 GPU 上的核函数，指定线程束的限制
__global__ void __launch_bounds__(
    // 设置线程束的上限为特定类型和参数的注意力机制后向传播核函数所需的线程数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 64>::kNumThreads,
    // 设置每个 SM 上执行该核函数的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
// 函数原型声明，调用由 cutlass 库定义的具体函数
fmha_cutlassB_f32_aligned_64x64_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 64>::Params p);

// 使用 CUDA __global__ 关键字定义一个 GPU 上的核函数，指定线程束的限制
__global__ void __launch_bounds__(
    // 设置线程束的上限为特定类型和参数的注意力机制后向传播核函数所需的线程数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 128>::kNumThreads,
    // 设置每个 SM 上执行该核函数的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
// 函数原型声明，调用由 cutlass 库定义的具体函数
fmha_cutlassB_f32_aligned_64x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 128>::Params p);

// 使用 CUDA __global__ 关键字定义一个 GPU 上的核函数，指定线程束的限制
__global__ void __launch_bounds__(
    // 设置线程束的上限为特定类型和参数的注意力机制后向传播核函数所需的线程数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 65536>::kNumThreads,
    // 设置每个 SM 上执行该核函数的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
// 函数原型声明，调用由 cutlass 库定义的具体函数
fmha_cutlassB_f32_aligned_64x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 65536>::Params p);

// 使用 CUDA __global__ 关键字定义一个 GPU 上的核函数，指定线程束的限制
__global__ void __launch_bounds__(
    // 设置线程束的上限为特定类型和参数的注意力机制后向传播核函数所需的线程数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 32>::kNumThreads,
    // 设置每个 SM 上执行该核函数的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
// 函数原型声明，调用由 cutlass 库定义的具体函数
fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 32>::Params p);

// 使用 CUDA __global__ 关键字定义一个 GPU 上的核函数，指定线程束的限制
__global__ void __launch_bounds__(
    // 设置线程束的上限为特定类型和参数的注意力机制后向传播核函数所需的线程数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 64>::kNumThreads,
    // 设置每个 SM 上执行该核函数的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
// 函数原型声明，调用由 cutlass 库定义的具体函数
fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 64>::Params p);

// 使用 CUDA __global__ 关键字定义一个 GPU 上的核函数，指定线程束的限制
__global__ void __launch_bounds__(
    // 设置线程束的上限为特定类型和参数的注意力机制后向传播核函数所需的线程数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 128>::kNumThreads,
    // 设置每个 SM 上执行该核函数的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
// 函数原型声明，调用由 cutlass 库定义的具体函数
fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 128>::Params p);
    // 实例化一个模板类 AttentionBackwardKernel，使用的模板参数包括:
    // - cutlass::arch::Sm70: 使用的架构为 SM70
    // - float: 使用的数据类型为 float
    // - true: 使用指定的布尔值
    // - true: 使用指定的布尔值
    // - false: 使用指定的布尔值
    // - 64: 使用的第一个整数模板参数
    // - 64: 使用的第二个整数模板参数
    // - 65536: 使用的第三个整数模板参数
    // kMinBlocksPerSm 是模板类 AttentionBackwardKernel 的静态成员变量
# 定义一个函数调用，调用名为 'fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm70' 的函数，其参数类型为 AttentionBackwardKernel 的模板参数
fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 65536>::Params p);

# 定义一个 CUDA 核函数，设定其启动配置
__global__ void __launch_bounds__(
    # 设置 CUDA 核函数的线程数量
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 32>::kNumThreads,
    # 设置 CUDA 核函数在每个流多处理器上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
# 定义一个函数调用，调用名为 'fmha_cutlassB_f32_notaligned_64x64_k32_sm70' 的函数，其参数类型为 AttentionBackwardKernel 的模板参数
fmha_cutlassB_f32_notaligned_64x64_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 32>::Params p);

# 定义一个 CUDA 核函数，设定其启动配置
__global__ void __launch_bounds__(
    # 设置 CUDA 核函数的线程数量
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 64>::kNumThreads,
    # 设置 CUDA 核函数在每个流多处理器上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
# 定义一个函数调用，调用名为 'fmha_cutlassB_f32_notaligned_64x64_k64_sm70' 的函数，其参数类型为 AttentionBackwardKernel 的模板参数
fmha_cutlassB_f32_notaligned_64x64_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 64>::Params p);

# 定义一个 CUDA 核函数，设定其启动配置
__global__ void __launch_bounds__(
    # 设置 CUDA 核函数的线程数量
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 128>::kNumThreads,
    # 设置 CUDA 核函数在每个流多处理器上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
# 定义一个函数调用，调用名为 'fmha_cutlassB_f32_notaligned_64x64_k128_sm70' 的函数，其参数类型为 AttentionBackwardKernel 的模板参数
fmha_cutlassB_f32_notaligned_64x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 128>::Params p);

# 定义一个 CUDA 核函数，设定其启动配置
__global__ void __launch_bounds__(
    # 设置 CUDA 核函数的线程数量
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 65536>::kNumThreads,
    # 设置 CUDA 核函数在每个流多处理器上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
# 定义一个函数调用，调用名为 'fmha_cutlassB_f32_notaligned_64x64_k65536_sm70' 的函数，其参数类型为 AttentionBackwardKernel 的模板参数
fmha_cutlassB_f32_notaligned_64x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 65536>::Params p);

# 定义一个 CUDA 核函数，设定其启动配置
__global__ void __launch_bounds__(
    # 设置 CUDA 核函数的线程数量
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 32>::kNumThreads,
    # 设置 CUDA 核函数在每个流多处理器上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
# 定义一个函数调用，调用名为 'fmha_cutlassB_f32_notaligned_64x64_k32_dropout_sm70' 的函数，其参数类型为 AttentionBackwardKernel 的模板参数
fmha_cutlassB_f32_notaligned_64x64_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 32>::Params p);

# 定义一个 CUDA 核函数，设定其启动配置
__global__ void __launch_bounds__(
    # 设置 CUDA 核函数的线程数量
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 64>::kNumThreads,
    # 设置 CUDA 核函数在每个流多处理器上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
# 定义一个函数调用，调用名为 'fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm70' 的函数，其参数类型为 AttentionBackwardKernel 的模板参数
fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 64>::Params p);

# 定义一个 CUDA 核函数，设定其启动配置
__global__ void __launch_bounds__(
    # 设置 CUDA 核函数的线程数量
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 128>::kNumThreads,
    # 设置 CUDA 核函数在每个流多处理器上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
# 定义一个函数调用，调用名为 'fmha_cutlassB_f32_notaligned_64x64_k128_dropout_sm70' 的函数，其参数类型为 AttentionBackwardKernel 的模板参数
fmha_cutlassB_f32_notaligned_64x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 128>::Params p);
# 最后一个 CUDA 核函数定义的前半部分
__global__ void __launch_bounds__(
    // 使用 Cutlass 库中的 AttentionBackwardKernel 模板类的静态成员变量
    // 该模板类的模板参数包括: 架构 Sm70, 数据类型 float, 计算精度为非混合精度, 启用分块矩阵乘法的转置操作, 不支持浮点数混合精度计算,
    // 块大小为 64x64, 并且每个 SM 上最大支持的线程数为 65536
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 65536>::kNumThreads,
    
    // 使用 Cutlass 库中的 AttentionBackwardKernel 模板类的静态成员变量
    // 获取该模板类的模板参数中每个 SM 上的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)
    
    
    这段代码是在使用 Cutlass 库中的模板类 `AttentionBackwardKernel` 的静态成员变量，获取特定模板参数下的两个值：`kNumThreads` 和 `kMinBlocksPerSm`。
// 定义函数模板 dispatch_cutlassB_f32_sm70，接受一个回调函数对象 cb 和一个整数 cc 作为参数
template <typename T> void dispatch_cutlassB_f32_sm70(T cb, int cc) {
    // 使用回调函数 cb 调用具有指定参数的 AttentionBackwardKernel 实例，这里是 Sm70 架构、float 类型、不对齐、无 dropout、64x64 矩阵、k=32 的实例
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_sm70);
    // 类似地，使用 cb 调用其他几个不同参数的 AttentionBackwardKernel 实例
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 32>(), fmha_cutlassB_f32_notaligned_64x64_k32_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 64>(), fmha_cutlassB_f32_notaligned_64x64_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 128>(), fmha_cutlassB_f32_notaligned_64x64_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 64, 64, 65536>(), fmha_cutlassB_f32_notaligned_64x64_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 32>(), fmha_cutlassB_f32_notaligned_64x64_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 64>(), fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 128>(), fmha_cutlassB_f32_notaligned_64x64_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 64, 64, 65536>(), fmha_cutlassB_f32_notaligned_64x64_k65536_dropout_sm70);
}

// 定义全局 CUDA 函数 __global__，用于启动具有指定线程块和线程数限制的 CUDA 核函数
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
// 声明 CUDA 核函数 fmha_cutlassB_f16_aligned_64x64_k32_sm75，接受 AttentionBackwardKernel 的参数并返回
fmha_cutlassB_f16_aligned_64x64_k32_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 32>::Params p);
# 使用 CUDA 的全局函数声明，指定每个线程块的线程数量和每个 SM 最少的线程块数量
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 64>::Params p);

# 使用 CUDA 的全局函数声明，指定每个线程块的线程数量和每个 SM 最少的线程块数量
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 128>::Params p);

# 使用 CUDA 的全局函数声明，指定每个线程块的线程数量和每个 SM 最少的线程块数量
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 128>::Params p);

# 使用 CUDA 的全局函数声明，指定每个线程块的线程数量和每个 SM 最少的线程块数量
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 65536>::Params p);

# 使用 CUDA 的全局函数声明，指定每个线程块的线程数量和每个 SM 最少的线程块数量
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p);

# 使用 CUDA 的全局函数声明，指定每个线程块的线程数量和每个 SM 最少的线程块数量，同时支持dropout
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 32>::Params p);

# 使用 CUDA 的全局函数声明，指定每个线程块的线程数量和每个 SM 最少的线程块数量，同时支持dropout
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 64>::Params p);
# 定义 CUDA 全局函数，使用特定的线程限制和块限制启动
__global__ void __launch_bounds__(
    # 使用注意力后向核函数的线程数作为线程限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 128>::kNumThreads,
    # 使用注意力后向核函数的最小块数作为块限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 128>::kMinBlocksPerSm)
# 调用具有特定参数的注意力后向核函数，数据类型为半精度浮点数，对齐方式为128x64，dropout设置为true，CUDA架构为Sm75
fmha_cutlassB_f16_aligned_128x64_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 128>::Params p);

__global__ void __launch_bounds__(
    # 使用注意力后向核函数的线程数作为线程限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    # 使用注意力后向核函数的最小块数作为块限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
# 调用具有特定参数的注意力后向核函数，数据类型为半精度浮点数，对齐方式为64x64，dropout设置为true，CUDA架构为Sm75
fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);

__global__ void __launch_bounds__(
    # 使用注意力后向核函数的线程数作为线程限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    # 使用注意力后向核函数的最小块数作为块限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
# 调用具有特定参数的注意力后向核函数，数据类型为半精度浮点数，对齐方式为128x64，dropout设置为true，CUDA架构为Sm75
fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p);

__global__ void __launch_bounds__(
    # 使用注意力后向核函数的线程数作为线程限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    # 使用注意力后向核函数的最小块数作为块限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
# 调用具有特定参数的注意力后向核函数，数据类型为半精度浮点数，对齐方式为64x64，dropout设置为true，CUDA架构为Sm75
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p);

__global__ void __launch_bounds__(
    # 使用注意力后向核函数的线程数作为线程限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads,
    # 使用注意力后向核函数的最小块数作为块限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
# 调用具有特定参数的注意力后向核函数，数据类型为半精度浮点数，对齐方式为64x64，dropout设置为false，CUDA架构为Sm75
fmha_cutlassB_f16_notaligned_64x64_k32_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 32>::Params p);

__global__ void __launch_bounds__(
    # 使用注意力后向核函数的线程数作为线程限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads,
    # 使用注意力后向核函数的最小块数作为块限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
# 调用具有特定参数的注意力后向核函数，数据类型为半精度浮点数，对齐方式为64x64，dropout设置为false，CUDA架构为Sm75
fmha_cutlassB_f16_notaligned_64x64_k64_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 64>::Params p);

__global__ void __launch_bounds__(
    # 使用注意力后向核函数的线程数作为线程限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 128>::kNumThreads,
    # 使用注意力后向核函数的最小块数作为块限制参数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 128>::kMinBlocksPerSm)
# 调用具有特定参数的注意力后向核函数，数据类型为半精度浮点数，对齐方式为128x64，dropout设置为false，CUDA架构为Sm75
# 调用名为 fmha_cutlassB_f16_notaligned_128x64_k128_sm75 的函数，传入适当的参数类型和配置
fmha_cutlassB_f16_notaligned_128x64_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 128>::Params p);

# 定义一个 CUDA 核函数，指定其启动参数，确保它能在 GPU 上正确执行
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 128>::kNumThreads,  # 指定核函数的线程数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 128>::kMinBlocksPerSm)  # 指定核函数的最小块数

# 调用名为 fmha_cutlassB_f16_notaligned_64x64_k128_sm75 的函数，传入适当的参数类型和配置
fmha_cutlassB_f16_notaligned_64x64_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 128>::Params p);

# 定义一个 CUDA 核函数，指定其启动参数，确保它能在 GPU 上正确执行
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 65536>::kNumThreads,  # 指定核函数的线程数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 65536>::kMinBlocksPerSm)  # 指定核函数的最小块数

# 调用名为 fmha_cutlassB_f16_notaligned_128x64_k65536_sm75 的函数，传入适当的参数类型和配置
fmha_cutlassB_f16_notaligned_128x64_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 65536>::Params p);

# 定义一个 CUDA 核函数，指定其启动参数，确保它能在 GPU 上正确执行
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 65536>::kNumThreads,  # 指定核函数的线程数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)  # 指定核函数的最小块数

# 调用名为 fmha_cutlassB_f16_notaligned_64x64_k65536_sm75 的函数，传入适当的参数类型和配置
fmha_cutlassB_f16_notaligned_64x64_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 65536>::Params p);

# 定义一个 CUDA 核函数，指定其启动参数，确保它能在 GPU 上正确执行
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 32>::kNumThreads,  # 指定核函数的线程数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 32>::kMinBlocksPerSm)  # 指定核函数的最小块数

# 调用名为 fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm75 的函数，传入适当的参数类型和配置
fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 32>::Params p);

# 定义一个 CUDA 核函数，指定其启动参数，确保它能在 GPU 上正确执行
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 64>::kNumThreads,  # 指定核函数的线程数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 64>::kMinBlocksPerSm)  # 指定核函数的最小块数

# 调用名为 fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm75 的函数，传入适当的参数类型和配置
fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 64>::Params p);

# 定义一个 CUDA 核函数，指定其启动参数，确保它能在 GPU 上正确执行
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 128>::kNumThreads,  # 指定核函数的线程数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 128>::kMinBlocksPerSm)  # 指定核函数的最小块数

# 调用名为 fmha_cutlassB_f16_notaligned_128x64_k128_dropout_sm75 的函数，传入适当的参数类型和配置
fmha_cutlassB_f16_notaligned_128x64_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 128>::Params p);
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)


注释：


// 使用 Cutlass 库中的 AttentionBackwardKernel 模板，配置如下参数：
// - GPU 架构为 Sm75
// - 数据类型为 half_t
// - 配置参数依次为：false, true, false, 64, 64, 128
// 访问该模板中的静态成员 kMinBlocksPerSm，表示每个 SM（Streaming Multiprocessor，流处理器）的最小块数。
// 声明函数 fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm75，参数为 AttentionBackwardKernel 类的实例
fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 128>::Params p);

// 定义 CUDA 全局函数 __global__，设置启动参数
__global__ void __launch_bounds__(
    // 设置线程块的大小为 AttentionBackwardKernel 类特定实例的线程数
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 65536>::kNumThreads,
    // 设置每个流多少个线程块的最小数量
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 65536>::kMinBlocksPerSm)
// 声明函数 fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm75，参数为 AttentionBackwardKernel 类的实例
fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 65536>::Params p);

// 声明函数 fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm75，参数为 AttentionBackwardKernel 类的实例
fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 65536>::Params p);

// 定义模板函数 dispatch_cutlassB_f16_sm75，接受一个函数对象 cb 和一个整数 cc
template <typename T> void dispatch_cutlassB_f16_sm75(T cb, int cc) {
    // 调用函数对象 cb，传递具体的 AttentionBackwardKernel 实例和函数名
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 128>(), fmha_cutlassB_f16_aligned_128x64_k128_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 128>(), fmha_cutlassB_f16_aligned_128x64_k128_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm75);
}
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 64、64、32，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 64、64、64，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 128、64、128，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 128>(), fmha_cutlassB_f16_notaligned_128x64_k128_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 64、64、128，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 128、64、65536，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_notaligned_128x64_k65536_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 64、64、65536，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，进行随机失活（dropout），无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 64、64、32，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，进行随机失活（dropout），无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 64、64、64，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，进行随机失活（dropout），无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 128、64、128，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 128>(), fmha_cutlassB_f16_notaligned_128x64_k128_dropout_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，进行随机失活（dropout），无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 64、64、128，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，进行随机失活（dropout），无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 128、64、65536，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm75);
    
    # 调用 AttentionBackwardKernel 函数，使用指定的参数和模板实例化对象，不使用对齐，进行随机失活（dropout），无输入输出复用，不进行类型转换，使用的线程块、线程和向量大小分别为 64、64、65536，目标架构为 Sm75
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm75);
// ======== f32 / sm75 ========

// 定义 CUDA 全局函数，设置其线程块的启动限制，确保适合于 Sm75 架构下的计算
__global__ void __launch_bounds__(
    // 设定线程数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 32>::kNumThreads,
    // 设定每个 SM 最小块数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
// 调用特定的函数模板化实例，传入参数 p
fmha_cutlassB_f32_aligned_64x64_k32_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 32>::Params p);

// 定义 CUDA 全局函数，设置其线程块的启动限制，确保适合于 Sm75 架构下的计算
__global__ void __launch_bounds__(
    // 设定线程数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 64>::kNumThreads,
    // 设定每个 SM 最小块数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
// 调用特定的函数模板化实例，传入参数 p
fmha_cutlassB_f32_aligned_64x64_k64_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 64>::Params p);

// 定义 CUDA 全局函数，设置其线程块的启动限制，确保适合于 Sm75 架构下的计算
__global__ void __launch_bounds__(
    // 设定线程数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 128>::kNumThreads,
    // 设定每个 SM 最小块数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
// 调用特定的函数模板化实例，传入参数 p
fmha_cutlassB_f32_aligned_64x64_k128_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 128>::Params p);

// 定义 CUDA 全局函数，设置其线程块的启动限制，确保适合于 Sm75 架构下的计算
__global__ void __launch_bounds__(
    // 设定线程数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 65536>::kNumThreads,
    // 设定每个 SM 最小块数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
// 调用特定的函数模板化实例，传入参数 p
fmha_cutlassB_f32_aligned_64x64_k65536_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 65536>::Params p);

// 定义 CUDA 全局函数，设置其线程块的启动限制，确保适合于 Sm75 架构下的计算，包含 Dropout 功能
__global__ void __launch_bounds__(
    // 设定线程数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 32>::kNumThreads,
    // 设定每个 SM 最小块数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
// 调用特定的函数模板化实例，传入参数 p
fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 32>::Params p);

// 定义 CUDA 全局函数，设置其线程块的启动限制，确保适合于 Sm75 架构下的计算，包含 Dropout 功能
__global__ void __launch_bounds__(
    // 设定线程数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 64>::kNumThreads,
    // 设定每个 SM 最小块数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
// 调用特定的函数模板化实例，传入参数 p
fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 64>::Params p);

// 定义 CUDA 全局函数，设置其线程块的启动限制，确保适合于 Sm75 架构下的计算，包含 Dropout 功能
__global__ void __launch_bounds__(
    // 设定线程数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 128>::kNumThreads,
    // 设定每个 SM 最小块数目为特定模板化参数的值
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
// 调用特定的函数模板化实例，传入参数 p
fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 128>::Params p);
    # 使用Cutlass库中的AttentionBackwardKernel模板类，针对SM75架构和float类型，配置为向后处理。
    # 参数true, true, false分别对应kernel是否需要加载，是否需要保存，以及是否需要执行block的逻辑
    # 64, 64, 65536分别对应threadblock的大小，行数和列数
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
# 定义 CUDA 函数以执行注意力机制的后向传播，使用 Cutlass 库实现
fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm75(
    # 使用 Sm75 架构，操作类型为 float，启用对齐，启用 dropout，矩阵大小为 64x64，K 矩阵大小为 65536
    typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 65536>::Params p
);

# 定义 CUDA 核函数，配置其启动参数以优化性能
__global__ void __launch_bounds__(
    # 设定每个线程块的线程数，使用 Sm75 架构，操作类型为 float，不对齐，不启用 dropout，矩阵大小为 64x64，K 矩阵大小为 32
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 32>::kNumThreads,
    # 设定每个 SM 上的最小线程块数
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm
)
fmha_cutlassB_f32_notaligned_64x64_k32_sm75(
    # 使用 Sm75 架构，操作类型为 float，不对齐，不启用 dropout，矩阵大小为 64x64，K 矩阵大小为 32
    typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 32>::Params p
);

# 定义 CUDA 核函数，配置其启动参数以优化性能
__global__ void __launch_bounds__(
    # 设定每个线程块的线程数，使用 Sm75 架构，操作类型为 float，不对齐，不启用 dropout，矩阵大小为 64x64，K 矩阵大小为 64
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 64>::kNumThreads,
    # 设定每个 SM 上的最小线程块数
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 64>::kMinBlocksPerSm
)
fmha_cutlassB_f32_notaligned_64x64_k64_sm75(
    # 使用 Sm75 架构，操作类型为 float，不对齐，不启用 dropout，矩阵大小为 64x64，K 矩阵大小为 64
    typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 64>::Params p
);

# 定义 CUDA 核函数，配置其启动参数以优化性能
__global__ void __launch_bounds__(
    # 设定每个线程块的线程数，使用 Sm75 架构，操作类型为 float，不对齐，不启用 dropout，矩阵大小为 64x64，K 矩阵大小为 128
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 128>::kNumThreads,
    # 设定每个 SM 上的最小线程块数
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 128>::kMinBlocksPerSm
)
fmha_cutlassB_f32_notaligned_64x64_k128_sm75(
    # 使用 Sm75 架构，操作类型为 float，不对齐，不启用 dropout，矩阵大小为 64x64，K 矩阵大小为 128
    typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 128>::Params p
);

# 定义 CUDA 核函数，配置其启动参数以优化性能
__global__ void __launch_bounds__(
    # 设定每个线程块的线程数，使用 Sm75 架构，操作类型为 float，不对齐，不启用 dropout，矩阵大小为 64x64，K 矩阵大小为 65536
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 65536>::kNumThreads,
    # 设定每个 SM 上的最小线程块数
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm
)
fmha_cutlassB_f32_notaligned_64x64_k65536_sm75(
    # 使用 Sm75 架构，操作类型为 float，不对齐，不启用 dropout，矩阵大小为 64x64，K 矩阵大小为 65536
    typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 65536>::Params p
);

# 定义 CUDA 核函数，配置其启动参数以优化性能
__global__ void __launch_bounds__(
    # 设定每个线程块的线程数，使用 Sm75 架构，操作类型为 float，不对齐，启用 dropout，矩阵大小为 64x64，K 矩阵大小为 32
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 32>::kNumThreads,
    # 设定每个 SM 上的最小线程块数
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 32>::kMinBlocksPerSm
)
fmha_cutlassB_f32_notaligned_64x64_k32_dropout_sm75(
    # 使用 Sm75 架构，操作类型为 float，不对齐，启用 dropout，矩阵大小为 64x64，K 矩阵大小为 32
    typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 32>::Params p
);

# 定义 CUDA 核函数，配置其启动参数以优化性能
__global__ void __launch_bounds__(
    # 设定每个线程块的线程数，使用 Sm75 架构，操作类型为 float，不对齐，启用 dropout，矩阵大小为 64x64，K 矩阵大小为 64
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 64>::kNumThreads,
    # 设定每个 SM 上的最小线程块数
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 64>::kMinBlocksPerSm
)
fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm75(
    # 使用 Sm75 架构，操作类型为 float，不对齐，启用 dropout，矩阵大小为 64x64，K 矩阵大小为 64
    typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 64>::Params p
);

# 定义 CUDA 核函数，配置其启动参数以优化性能
__global__ void __launch_bounds__(
    # 设定每个线程块的线程数，使用 Sm75 架构，操作类型为 float，不对齐，启用 dropout，矩阵大小为 64x64，K 矩阵大小为 128
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 128>::kNumThreads,
    # 设定每个 SM 上的最小线程块数
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 128>::kMinBlocksPerSm
)
fmha_cutlassB_f32_notaligned_64x64_k128_dropout_sm75(
    # 使用 Sm75 架构，操作类型为 float，不对齐
    # 使用 Cutlass 库中的 AttentionBackwardKernel 模板类，指定以下模板参数：
    # - cutlass::arch::Sm75: 目标 GPU 架构为 Sm75
    # - float: 数据类型为单精度浮点数
    # - false: 不进行转置操作
    # - true: 启用分块处理
    # - false: 不需要额外存储
    # - 64: 每个线程块的线程数为 64
    # - 64: 每个线程块的分块大小为 64x64
    # - 65536: 每个 SM 上的最小线程块数量为 65536
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 65536>::kNumThreads,
    # 使用相同的模板参数，获取每个 SM 上的最小线程块数量
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)
// 定义函数模板，用于调度不同参数的 AttentionBackwardKernel 实例
template <typename T> void dispatch_cutlassB_f32_sm75(T cb, int cc) {
    // 调用回调函数 cb，传入具体的参数和函数指针
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 32>(), fmha_cutlassB_f32_notaligned_64x64_k32_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 64>(), fmha_cutlassB_f32_notaligned_64x64_k64_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 128>(), fmha_cutlassB_f32_notaligned_64x64_k128_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 64, 64, 65536>(), fmha_cutlassB_f32_notaligned_64x64_k65536_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 32>(), fmha_cutlassB_f32_notaligned_64x64_k32_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 64>(), fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 128>(), fmha_cutlassB_f32_notaligned_64x64_k128_dropout_sm75);
    cb(AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 64, 64, 65536>(), fmha_cutlassB_f32_notaligned_64x64_k65536_dropout_sm75);
}

// 定义用于 f32 和 sm80 架构的全局 CUDA 核函数
__global__ void __launch_bounds__(
    // 设置 CUDA 核函数的线程数和每个 SM 最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
// 函数原型声明，用于定义特定参数的 AttentionBackwardKernel 实例
fmha_cutlassB_f32_aligned_64x64_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 32>::Params p);
# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 设置其执行参数
__global__ void __launch_bounds__(
    # 指定 CUDA 线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 64> 的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 64>::kNumThreads,
    # 指定 CUDA 线程块的最小数量为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 64> 的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
# 函数名称为 fmha_cutlassB_f32_aligned_64x64_k64_sm80，参数为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 64>::Params 类型 p
fmha_cutlassB_f32_aligned_64x64_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 64>::Params p);

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 设置其执行参数
__global__ void __launch_bounds__(
    # 指定 CUDA 线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 128> 的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 128>::kNumThreads,
    # 指定 CUDA 线程块的最小数量为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 128> 的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 128>::kMinBlocksPerSm)
# 函数名称为 fmha_cutlassB_f32_aligned_128x64_k128_sm80，参数为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 128>::Params 类型 p
fmha_cutlassB_f32_aligned_128x64_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 128>::Params p);

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 设置其执行参数
__global__ void __launch_bounds__(
    # 指定 CUDA 线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 128> 的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 128>::kNumThreads,
    # 指定 CUDA 线程块的最小数量为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 128> 的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
# 函数名称为 fmha_cutlassB_f32_aligned_64x64_k128_sm80，参数为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 128>::Params 类型 p
fmha_cutlassB_f32_aligned_64x64_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 128>::Params p);

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 设置其执行参数
__global__ void __launch_bounds__(
    # 指定 CUDA 线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 65536> 的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 65536>::kNumThreads,
    # 指定 CUDA 线程块的最小数量为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 65536> 的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
# 函数名称为 fmha_cutlassB_f32_aligned_128x64_k65536_sm80，参数为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 65536>::Params 类型 p
fmha_cutlassB_f32_aligned_128x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 65536>::Params p);

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 设置其执行参数
__global__ void __launch_bounds__(
    # 指定 CUDA 线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 65536> 的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 65536>::kNumThreads,
    # 指定 CUDA 线程块的最小数量为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 65536> 的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
# 函数名称为 fmha_cutlassB_f32_aligned_64x64_k65536_sm80，参数为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 65536>::Params 类型 p
fmha_cutlassB_f32_aligned_64x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 65536>::Params p);

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 设置其执行参数
__global__ void __launch_bounds__(
    # 指定 CUDA 线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 32> 的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 32>::kNumThreads,
    # 指定 CUDA 线程块的最小数量为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 32> 的最小块数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
# 函数名称为 fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm80，参数为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 32>::Params 类型 p
fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 32>::Params p);

# 定义一个 CUDA 全局函数，使用 __launch_bounds__ 设置其执行参数
__global__ void __launch_bounds__(
    # 指定 CUDA 线程块的大小为 AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 64> 的线程数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 64>::kNumThreads,
    # 指定 CUDA 线
    // 实例化 AttentionBackwardKernel 模板，使用的参数包括：
    // - cutlass::arch::Sm80: 目标架构为 NVIDIA 的 Ampere 架构的某个 GPU
    // - float: 计算所使用的数据类型为单精度浮点数
    // - true: 模板参数，表示需要执行反向传播
    // - true: 模板参数，表示使用权重转置（transpose）
    // - false: 模板参数，表示不进行下层格点（gating）操作
    // - 128, 64, 128: 模板参数，指定线程块的维度，分别为 (x, y, z)
    // - AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 128>::kMinBlocksPerSm: 类中的常量或静态成员，表示每个 SM 最小要求的线程块数
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 128>::kMinBlocksPerSm)
// 定义函数原型，声明一个名为 fmha_cutlassB_f32_aligned_128x64_k128_dropout_sm80 的函数，接受 AttentionBackwardKernel 类型的参数 p
fmha_cutlassB_f32_aligned_128x64_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 128>::Params p);

// 定义 GPU 的核函数，设置其线程块的启动限制
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
{
    // 空函数体，用于定义 kernel 的启动限制
}

// 同上，定义一个名为 fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm80 的函数，接受 AttentionBackwardKernel 类型的参数 p
fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 128>::Params p);

// 定义 GPU 的核函数，设置其线程块的启动限制
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
{
    // 空函数体，用于定义 kernel 的启动限制
}

// 同上，定义一个名为 fmha_cutlassB_f32_aligned_128x64_k65536_dropout_sm80 的函数，接受 AttentionBackwardKernel 类型的参数 p
fmha_cutlassB_f32_aligned_128x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 65536>::Params p);

// 定义 GPU 的核函数，设置其线程块的启动限制
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
{
    // 空函数体，用于定义 kernel 的启动限制
}

// 定义模板函数 dispatch_cutlassB_f32_sm80，接受一个类型为 T 的参数 cb 和一个整数参数 cc
template <typename T> void dispatch_cutlassB_f32_sm80(T cb, int cc) {
    // 调用参数 cb 函数，传入不同的 AttentionBackwardKernel 实例和相应的函数名作为参数
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 128>(), fmha_cutlassB_f32_aligned_128x64_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 65536>(), fmha_cutlassB_f32_aligned_128x64_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 128>(), fmha_cutlassB_f32_aligned_128x64_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_dropout_sm80);
}
    # 调用 AttentionBackwardKernel 类型的函数，传入指定的模板参数和函数参数
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 65536>(), fmha_cutlassB_f32_aligned_128x64_k65536_dropout_sm80);
    # 调用 AttentionBackwardKernel 类型的函数，传入指定的模板参数和函数参数
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_dropout_sm80);
// 模板函数，根据模板参数 DT 和 T 的类型进行分发处理
template <typename DT, typename T>
void dispatch_cutlassB(T cb, int cc = 0) {

    // 如果 DT 类型是 cutlass::half_t 并且 cc 在 [70, 75) 范围内
    if (std::is_same<DT, cutlass::half_t>::value && 70 <= cc && cc < 75) {
        // 调用适用于半精度和指定计算能力版本的函数
        dispatch_cutlassB_f16_sm70(cb, cc);
    }
    // 如果 DT 类型是 cutlass::bfloat16_t 并且 cc 在 [80, 100) 范围内
    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 100) {
        // 调用适用于 BF16 和指定计算能力版本的函数
        dispatch_cutlassB_bf16_sm80(cb, cc);
    }
    // 如果 DT 类型是 cutlass::half_t 并且 cc 在 [80, 100) 范围内
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 100) {
        // 调用适用于半精度和指定计算能力版本的函数
        dispatch_cutlassB_f16_sm80(cb, cc);
    }
    // 如果 DT 类型是 cutlass::half_t 并且 cc 在 [50, 70) 范围内
    if (std::is_same<DT, cutlass::half_t>::value && 50 <= cc && cc < 70) {
        // 调用适用于半精度和指定计算能力版本的函数
        dispatch_cutlassB_f16_sm50(cb, cc);
    }
    // 如果 DT 类型是 float 并且 cc 在 [50, 70) 范围内
    if (std::is_same<DT, float>::value && 50 <= cc && cc < 70) {
        // 调用适用于单精度浮点数和指定计算能力版本的函数
        dispatch_cutlassB_f32_sm50(cb, cc);
    }
    // 如果 DT 类型是 float 并且 cc 在 [70, 75) 范围内
    if (std::is_same<DT, float>::value && 70 <= cc && cc < 75) {
        // 调用适用于单精度浮点数和指定计算能力版本的函数
        dispatch_cutlassB_f32_sm70(cb, cc);
    }
    // 如果 DT 类型是 cutlass::half_t 并且 cc 在 [75, 80) 范围内
    if (std::is_same<DT, cutlass::half_t>::value && 75 <= cc && cc < 80) {
        // 调用适用于半精度和指定计算能力版本的函数
        dispatch_cutlassB_f16_sm75(cb, cc);
    }
    // 如果 DT 类型是 float 并且 cc 在 [75, 80) 范围内
    if (std::is_same<DT, float>::value && 75 <= cc && cc < 80) {
        // 调用适用于单精度浮点数和指定计算能力版本的函数
        dispatch_cutlassB_f32_sm75(cb, cc);
    }
    // 如果 DT 类型是 float 并且 cc 在 [80, 100) 范围内
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 100) {
        // 调用适用于单精度浮点数和指定计算能力版本的函数
        dispatch_cutlassB_f32_sm80(cb, cc);
    }
}
```