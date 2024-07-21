# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\kernels\cutlassF.h`

```
/*
cpp
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"

// 包含 CUDA 内存高效注意力机制的前向传递头文件
#pragma once
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

// 使用 PyTorchMemEffAttention 命名空间
using namespace PyTorchMemEffAttention;

// ======== bf16 / sm80 ========

// 定义 CUDA 核函数，使用 bfloat16 数据类型，针对 sm80 架构，数据对齐方式为 64x64
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);

// 定义 CUDA 核函数，使用 bfloat16 数据类型，针对 sm80 架构，数据对齐方式为 64x128
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x128_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);

// 定义 CUDA 核函数，使用 bfloat16 数据类型，针对 sm80 架构，数据对齐方式为 32x128，使用全局内存
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);

// 定义模板函数，用于分发不同类型的 bfloat16 核函数，针对 sm80 架构
template <typename T> void dispatch_cutlassF_bf16_sm80(T cb, int cc) {
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_bf16_aligned_64x64_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_bf16_aligned_64x128_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_bf16_aligned_32x128_gmem_sm80);
}

// ======== f16 / sm50 ========

// 定义 CUDA 核函数，使用 half 数据类型，针对 sm50 架构，数据对齐方式为 64x64
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::Params p);

// 定义 CUDA 核函数，使用 half 数据类型，针对 sm50 架构，数据对齐方式为 32x128
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::Params p);
// 定义 CUDA 全局函数，设置其线程块的大小和数量，适用于 SM50 架构
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kNumThreads,  // 每个线程块的线程数量
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)  // 每个 SM 最小线程块数量
fmha_cutlassF_f16_aligned_32x128_gmem_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::Params p);

// 定义 CUDA 全局函数，设置其线程块的大小和数量，适用于 SM50 架构
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kNumThreads,  // 每个线程块的线程数量
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kMinBlocksPerSm)  // 每个 SM 最小线程块数量
fmha_cutlassF_f16_notaligned_64x64_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::Params p);

// 定义 CUDA 全局函数，设置其线程块的大小和数量，适用于 SM50 架构
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kNumThreads,  // 每个线程块的线程数量
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kMinBlocksPerSm)  // 每个 SM 最小线程块数量
fmha_cutlassF_f16_notaligned_32x128_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::Params p);

// 定义 CUDA 全局函数，设置其线程块的大小和数量，适用于 SM50 架构
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kNumThreads,  // 每个线程块的线程数量
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)  // 每个 SM 最小线程块数量
fmha_cutlassF_f16_notaligned_32x128_gmem_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::Params p);

// 模板函数，调度不同的 CUDA 函数针对不同的输入参数 T 和 cc 进行调用
template <typename T> void dispatch_cutlassF_f16_sm50(T cb, int cc) {
    // 调用回调函数 cb，传入不同的 AttentionKernel 和函数指针，适用于 SM50 架构
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm50);
}

// ======== f16 / sm70 ========
// 定义 CUDA 全局函数，设置其线程块的大小和数量，适用于 SM70 架构
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kNumThreads,  // 每个线程块的线程数量
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kMinBlocksPerSm)  // 每个 SM 最小线程块数量
fmha_cutlassF_f16_aligned_64x64_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::Params p);

// 下面还有其他定义，但未在示例中展示，需要根据格式添加适当的注释
    # 使用 Cutlass 库中 AttentionKernel 模板，指定以下模板参数：
    # - cutlass::half_t：使用半精度浮点数作为数据类型
    # - cutlass::arch::Sm70：目标架构为 NVIDIA Volta 或 Turing 架构的 SM70
    # - true：使用 TensorCore 加速（是否使用 TensorCore 的布尔标志）
    # - 32, 128, 128：线程块的尺寸为 32x128x128
    # - true, true：确保内存分配合并和多流操作开启
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kNumThreads,
    
    # 使用 Cutlass 库中 AttentionKernel 模板，获取每个 SM70 架构的最小块数
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kMinBlocksPerSm)


这段代码是在使用 Cutlass 库中的 AttentionKernel 模板，根据不同的模板参数生成两个常量值。
// 定义函数签名，声明一个名为 fmha_cutlassF_f16_aligned_32x128_rf_sm70 的函数，使用了特定的模板参数
fmha_cutlassF_f16_aligned_32x128_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::Params p);

// 定义 GPU 核函数，设置其线程束和每个 SM 最小块数的限制
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)

// 定义函数签名，声明一个名为 fmha_cutlassF_f16_aligned_32x128_gmem_sm70 的函数，使用了特定的模板参数
fmha_cutlassF_f16_aligned_32x128_gmem_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::Params p);

// 定义 GPU 核函数，设置其线程束和每个 SM 最小块数的限制
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kMinBlocksPerSm)

// 定义函数签名，声明一个名为 fmha_cutlassF_f16_notaligned_64x64_rf_sm70 的函数，使用了特定的模板参数
fmha_cutlassF_f16_notaligned_64x64_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::Params p);

// 定义 GPU 核函数，设置其线程束和每个 SM 最小块数的限制
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kMinBlocksPerSm)

// 定义函数签名，声明一个名为 fmha_cutlassF_f16_notaligned_32x128_rf_sm70 的函数，使用了特定的模板参数
fmha_cutlassF_f16_notaligned_32x128_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::Params p);

// 定义 GPU 核函数，设置其线程束和每个 SM 最小块数的限制
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)

// 模板函数 dispatch_cutlassF_f16_sm70，根据参数 cb 和 cc 分发不同的模板实例
template <typename T> void dispatch_cutlassF_f16_sm70(T cb, int cc) {
    // 调用回调函数 cb，传入不同的 AttentionKernel 实例和相应的函数指针
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm70);
}

// 定义 GPU 核函数，设置其线程束和每个 SM 最小块数的限制，针对 Sm75 架构
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
// 定义一个函数原型，用于声明一个接受特定参数类型的函数，并命名为 fmha_cutlassF_f16_aligned_64x64_rf_sm75
fmha_cutlassF_f16_aligned_64x64_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数限制，以确保在 Sm75 架构上运行时，线程块的数量和每个 SM 最小块的数量符合特定的值
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::Params p);

// 定义一个函数原型，用于声明一个接受特定参数类型的函数，并命名为 fmha_cutlassF_f16_aligned_32x128_gmem_sm75
fmha_cutlassF_f16_aligned_32x128_gmem_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数限制，以确保在 Sm75 架构上运行时，线程块的数量和每个 SM 最小块的数量符合特定的值
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::Params p);

// 定义一个函数原型，用于声明一个接受特定参数类型的函数，并命名为 fmha_cutlassF_f16_notaligned_64x64_rf_sm75
fmha_cutlassF_f16_notaligned_64x64_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数限制，以确保在 Sm75 架构上运行时，线程块的数量和每个 SM 最小块的数量符合特定的值
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::Params p);

// 定义一个函数原型，用于声明一个接受特定参数类型的函数，并命名为 fmha_cutlassF_f16_notaligned_32x128_gmem_sm75
fmha_cutlassF_f16_notaligned_32x128_gmem_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::Params p);

// 定义一个 CUDA 全局函数，设置启动参数限制，以确保在 Sm75 架构上运行时，线程块的数量和每个 SM 最小块的数量符合特定的值
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)

// 模板函数 dispatch_cutlassF_f16_sm75，接受一个回调函数 cb 和一个整数参数 cc
template <typename T> void dispatch_cutlassF_f16_sm75(T cb, int cc) {
    // 使用给定的 AttentionKernel 参数调用回调函数 cb，传递具体的函数名 fmha_cutlassF_f16_aligned_64x64_rf_sm75
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm75);
    // 使用给定的 AttentionKernel 参数调用回调函数 cb，传递具体的函数名 fmha_cutlassF_f16_aligned_32x128_rf_sm75
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm75);
    // 使用给定的 AttentionKernel 参数调用回调函数 cb，传递具体的函数名 fmha_cutlassF_f16_aligned_32x128_gmem_sm75
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm75);
    // 使用给定的 AttentionKernel 参数调用回调函数 cb，传递具体的函数名 fmha_cutlassF_f16_notaligned_64x64_rf_sm75
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm75);
    // 使用给定的 AttentionKernel 参数调用回调函数 cb，传递具体的函数名 fmha_cutlassF_f16_notaligned_32x128_rf_sm75
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm75);
}
    // 调用函数 cb，传入一个 AttentionKernel 实例作为第一个参数，
    // 模板参数为 <cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>
    // 第二个参数为 fmha_cutlassF_f16_notaligned_32x128_gmem_sm75
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm75);
// ======== f16 / sm80 ========

// 定义 CUDA 全局函数，使用 __launch_bounds__ 限制线程块的大小和最小的每个 SM 的块数
__global__ void __launch_bounds__(
    // 使用注意力内核的模板参数和设备架构 Sm80，启用的数据类型为 half_t
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    // 最小的每个 SM 的块数
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
// 函数名称
fmha_cutlassF_f16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);

__global__ void __launch_bounds__(
    // 使用注意力内核的模板参数和设备架构 Sm80，启用的数据类型为 half_t
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    // 最小的每个 SM 的块数
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
// 函数名称
fmha_cutlassF_f16_aligned_64x128_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);

__global__ void __launch_bounds__(
    // 使用注意力内核的模板参数和设备架构 Sm80，启用的数据类型为 half_t
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    // 最小的每个 SM 的块数
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
// 函数名称
fmha_cutlassF_f16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);

// 定义一个模板函数，根据传入的函数对象 cb 和整数 cc 来调度不同的注意力内核任务
template <typename T> void dispatch_cutlassF_f16_sm80(T cb, int cc) {
    // 调用传入的函数对象 cb，执行注意力内核为 half_t 类型，设备架构为 Sm80，块大小为 64x64
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm80);
    // 调用传入的函数对象 cb，执行注意力内核为 half_t 类型，设备架构为 Sm80，块大小为 64x128
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_64x128_rf_sm80);
    // 调用传入的函数对象 cb，执行注意力内核为 half_t 类型，设备架构为 Sm80，块大小为 32x128
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm80);
}

// ======== f32 / sm50 ========

__global__ void __launch_bounds__(
    // 使用注意力内核的模板参数和设备架构 Sm50，启用的数据类型为 float
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kNumThreads,
    // 最小的每个 SM 的块数
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
// 函数名称
fmha_cutlassF_f32_aligned_64x64_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::Params p);

__global__ void __launch_bounds__(
    // 使用注意力内核的模板参数和设备架构 Sm50，启用的数据类型为 float
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kNumThreads,
    // 最小的每个 SM 的块数
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
// 函数名称
fmha_cutlassF_f32_aligned_32x128_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::Params p);

__global__ void __launch_bounds__(
    // 使用注意力内核的模板参数和设备架构 Sm50，启用的数据类型为 float
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kNumThreads,
    // 最小的每个 SM 的块数
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
// 函数名称
fmha_cutlassF_f32_aligned_32x128_gmem_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::Params p);

__global__ void __launch_bounds__(
    // 使用注意力内核的模板参数和设备架构 Sm50，数据类型为 float，不启用混合精度
    AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kNumThreads,
    // 最小的每个 SM 的块数
    // 使用 AttentionKernel 模板，其中的模板参数如下：
    //   - float: 表示数据类型为单精度浮点数
    //   - cutlass::arch::Sm50: 表示使用的 GPU 架构为 SM50（例如 NVIDIA 的某一代 GPU 架构）
    //   - false: 表示不启用某种特性（具体特性需要查看具体文档或定义）
    //   - 64, 64, 64: 表示模板中的一些维度参数，具体含义需要根据具体的模板定义来理解
    //   - true, true: 表示启用某些特性（具体特性需要查看具体文档或定义）
    // ::kMinBlocksPerSm: 访问模板中的某个常量或枚举值 kMinBlocksPerSm，该常量或枚举值表示每个 SM （Streaming Multiprocessor）最小的块数
    AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
// 定义一个函数原型，声明一个名为 fmha_cutlassF_f32_notaligned_64x64_rf_sm50 的函数，
// 该函数接受一个 AttentionKernel 类型的模板参数，参数类型为 float，模板参数包括硬件架构 Sm50，
// 并且使用非对齐内存访问模式，线程块大小为 64x64，寄存器文件使用模式为 rf。
// 函数声明没有具体实现，只是告知编译器该函数的存在和接口。

__global__ void __launch_bounds__(
    // 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
    // 这里设定了函数的线程块大小（由模板参数 kNumThreads 决定）和最小每个 SM 块数（由模板参数 kMinBlocksPerSm 决定）。
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
// 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
// 函数使用特定的模板参数创建名为 fmha_cutlassF_f32_notaligned_32x128_rf_sm50 的函数。
// 函数声明没有具体实现，只是告知编译器该函数的存在和接口。
fmha_cutlassF_f32_notaligned_32x128_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::Params p);

__global__ void __launch_bounds__(
    // 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
    // 这里设定了函数的线程块大小（由模板参数 kNumThreads 决定）和最小每个 SM 块数（由模板参数 kMinBlocksPerSm 决定）。
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
// 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
// 函数使用特定的模板参数创建名为 fmha_cutlassF_f32_notaligned_32x128_gmem_sm50 的函数。
// 函数声明没有具体实现，只是告知编译器该函数的存在和接口。
fmha_cutlassF_f32_notaligned_32x128_gmem_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::Params p);

// 定义一个模板函数，接受一个模板类型 T 和一个整数 cc 作为参数。
template <typename T> void dispatch_cutlassF_f32_sm50(T cb, int cc) {
    // 调用传入的回调函数 cb，传递一个特定的 AttentionKernel 实例和一个函数指针 fmha_cutlassF_f32_aligned_64x64_rf_sm50。
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm50);
    // 调用传入的回调函数 cb，传递一个特定的 AttentionKernel 实例和一个函数指针 fmha_cutlassF_f32_aligned_32x128_rf_sm50。
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm50);
    // 调用传入的回调函数 cb，传递一个特定的 AttentionKernel 实例和一个函数指针 fmha_cutlassF_f32_aligned_32x128_gmem_sm50。
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm50);
    // 调用传入的回调函数 cb，传递一个特定的 AttentionKernel 实例和一个函数指针 fmha_cutlassF_f32_notaligned_64x64_rf_sm50。
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm50);
    // 调用传入的回调函数 cb，传递一个特定的 AttentionKernel 实例和一个函数指针 fmha_cutlassF_f32_notaligned_32x128_rf_sm50。
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm50);
    // 调用传入的回调函数 cb，传递一个特定的 AttentionKernel 实例和一个函数指针 fmha_cutlassF_f32_notaligned_32x128_gmem_sm50。
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm50);
}

// ======== f32 / sm70 ========

__global__ void __launch_bounds__(
    // 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
    // 这里设定了函数的线程块大小（由模板参数 kNumThreads 决定）和最小每个 SM 块数（由模板参数 kMinBlocksPerSm 决定）。
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
// 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
// 函数使用特定的模板参数创建名为 fmha_cutlassF_f32_aligned_64x64_rf_sm70 的函数。
// 函数声明没有具体实现，只是告知编译器该函数的存在和接口。
fmha_cutlassF_f32_aligned_64x64_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::Params p);

__global__ void __launch_bounds__(
    // 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
    // 这里设定了函数的线程块大小（由模板参数 kNumThreads 决定）和最小每个 SM 块数（由模板参数 kMinBlocksPerSm 决定）。
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
// 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
// 函数使用特定的模板参数创建名为 fmha_cutlassF_f32_aligned_32x128_rf_sm70 的函数。
// 函数声明没有具体实现，只是告知编译器该函数的存在和接口。
fmha_cutlassF_f32_aligned_32x128_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::Params p);

__global__ void __launch_bounds__(
    // 定义一个全局函数，函数名和参数都由 CUDA 运行时使用的特殊语法控制。
    // 这里设定了函数的线程块大小（由模板参数 kNumThreads 决定）和最小每个 SM 块数（由模板参数 kMinBlocksPerSm
    # 使用模板参数为 float 的 AttentionKernel 类的静态成员变量 kNumThreads
    AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kNumThreads,
    # 使用模板参数为 float 的 AttentionKernel 类的静态成员变量 kMinBlocksPerSm
    AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
// 定义一个函数原型，参数为一个模板化的 AttentionKernel 实例化对象，使用 float 类型、cutlass::arch::Sm70 架构、非对齐内存访问，处理 64x64 的数据块，并指定寄存器文件 rf_sm70
fmha_cutlassF_f32_notaligned_64x64_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::Params p);

// 定义一个 CUDA 全局函数，设置其启动参数为针对 Sm70 架构的注意力机制内核，使用 32x128 的线程块配置和最小的每 SM 块数，处理 float 类型数据
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::Params p);

// 定义一个函数原型，参数为一个模板化的 AttentionKernel 实例化对象，使用 float 类型、cutlass::arch::Sm70 架构、非对齐内存访问，处理 32x128 的数据块，并指定寄存器文件 rf_sm70
fmha_cutlassF_f32_notaligned_32x128_gmem_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::Params p);

// 定义一个模板函数，接受一个函数对象 cb 和一个整数 cc 作为参数
template <typename T> void dispatch_cutlassF_f32_sm70(T cb, int cc) {
    // 调用 cb 函数对象，传入不同的 AttentionKernel 实例化对象和函数名，处理 Sm70 架构下的不同类型数据和配置
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm70);
}

// ======== f32 / sm75 ========

// 定义一个 CUDA 全局函数，设置其启动参数为针对 Sm75 架构的注意力机制内核，使用 64x64 的线程块配置和最小的每 SM 块数，处理 float 类型数据
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::Params p);

// 定义一个 CUDA 全局函数，设置其启动参数为针对 Sm75 架构的注意力机制内核，使用 32x128 的线程块配置和最小的每 SM 块数，处理 float 类型数据
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::Params p);

// 定义一个 CUDA 全局函数，设置其启动参数为针对 Sm75 架构的注意力机制内核，使用 32x128 的线程块配置和最小的每 SM 块数，处理 float 类型数据
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::Params p);
    # 使用模板参数为float的AttentionKernel类，具体配置如下：
    # - 数据类型为float
    # - 设备架构为SM75（NVIDIA的某种GPU架构）
    # - 是否进行混合精度计算为false
    # - 线程块大小为64x64x64
    # - 是否开启特定优化特性为true
    # - 是否开启特定优化特性为true
    # 访问该类的静态成员变量kNumThreads，返回线程数的数量。
    AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kNumThreads,
    
    # 使用模板参数为float的AttentionKernel类，具体配置如下：
    # - 数据类型为float
    # - 设备架构为SM75（NVIDIA的某种GPU架构）
    # - 是否进行混合精度计算为false
    # - 线程块大小为64x64x64
    # - 是否开启特定优化特性为true
    # - 是否开启特定优化特性为true
    # 访问该类的静态成员变量kMinBlocksPerSm，返回每个SM（流多处理器）的最小块数。
    AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
// 定义函数，根据给定的参数类型和配置生成一个函数调用，用于处理具体的注意力机制计算
fmha_cutlassF_f32_notaligned_64x64_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::Params p);

// 定义全局函数，设置启动参数，以确保使用正确数量的线程和块在设备上运行
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
// 定义函数，根据给定的参数类型和配置生成一个函数调用，用于处理具体的注意力机制计算
fmha_cutlassF_f32_notaligned_32x128_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::Params p);

// 定义全局函数，设置启动参数，以确保使用正确数量的线程和块在设备上运行
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
// 定义函数，根据给定的参数类型和配置生成一个函数调用，用于处理具体的注意力机制计算
fmha_cutlassF_f32_notaligned_32x128_gmem_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::Params p);

// 定义模板函数，根据不同的注意力核心配置，调用回调函数 cb 处理具体的计算任务
template <typename T> void dispatch_cutlassF_f32_sm75(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm75);
}

// ======== f32 / sm80 ========

// 定义全局函数，设置启动参数，以确保使用正确数量的线程和块在设备上运行
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
// 定义函数，根据给定的参数类型和配置生成一个函数调用，用于处理具体的注意力机制计算
fmha_cutlassF_f32_aligned_64x64_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);

// 定义全局函数，设置启动参数，以确保使用正确数量的线程和块在设备上运行
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
// 定义函数，根据给定的参数类型和配置生成一个函数调用，用于处理具体的注意力机制计算
fmha_cutlassF_f32_aligned_64x128_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);

// 定义全局函数，设置启动参数，以确保使用正确数量的线程和块在设备上运行
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
// 定义函数，根据给定的参数类型和配置生成一个函数调用，用于处理具体的注意力机制计算
fmha_cutlassF_f32_aligned_32x128_gmem_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);
    # 调用注意力机制内核函数，使用浮点数类型，基于Cutlass库的SM80架构，设置特定参数
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm80);
    
    # 调用注意力机制内核函数，使用浮点数类型，基于Cutlass库的SM80架构，设置特定参数
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_64x128_rf_sm80);
    
    # 调用注意力机制内核函数，使用浮点数类型，基于Cutlass库的SM80架构，设置特定参数
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm80);
// 模板函数，根据模板参数 DT 和整数参数 T，分派不同的函数回调 cb
template <typename DT, typename T>
void dispatch_cutlassF(T cb, int cc = 0) {

    // 如果 DT 是 bfloat16_t 类型，并且 cc 在 [80, 100) 范围内
    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 100) {
        // 调用特定于 bfloat16_t 和指定 cc 的函数
        dispatch_cutlassF_bf16_sm80(cb, cc);
    }
    // 如果 DT 是 half_t 类型，并且 cc 在 [50, 70) 范围内
    if (std::is_same<DT, cutlass::half_t>::value && 50 <= cc && cc < 70) {
        // 调用特定于 half_t 和指定 cc 的函数
        dispatch_cutlassF_f16_sm50(cb, cc);
    }
    // 如果 DT 是 half_t 类型，并且 cc 在 [70, 75) 范围内
    if (std::is_same<DT, cutlass::half_t>::value && 70 <= cc && cc < 75) {
        // 调用特定于 half_t 和指定 cc 的函数
        dispatch_cutlassF_f16_sm70(cb, cc);
    }
    // 如果 DT 是 half_t 类型，并且 cc 在 [75, 80) 范围内
    if (std::is_same<DT, cutlass::half_t>::value && 75 <= cc && cc < 80) {
        // 调用特定于 half_t 和指定 cc 的函数
        dispatch_cutlassF_f16_sm75(cb, cc);
    }
    // 如果 DT 是 half_t 类型，并且 cc 在 [80, 100) 范围内
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 100) {
        // 调用特定于 half_t 和指定 cc 的函数
        dispatch_cutlassF_f16_sm80(cb, cc);
    }
    // 如果 DT 是 float 类型，并且 cc 在 [50, 70) 范围内
    if (std::is_same<DT, float>::value && 50 <= cc && cc < 70) {
        // 调用特定于 float 和指定 cc 的函数
        dispatch_cutlassF_f32_sm50(cb, cc);
    }
    // 如果 DT 是 float 类型，并且 cc 在 [70, 75) 范围内
    if (std::is_same<DT, float>::value && 70 <= cc && cc < 75) {
        // 调用特定于 float 和指定 cc 的函数
        dispatch_cutlassF_f32_sm70(cb, cc);
    }
    // 如果 DT 是 float 类型，并且 cc 在 [75, 80) 范围内
    if (std::is_same<DT, float>::value && 75 <= cc && cc < 80) {
        // 调用特定于 float 和指定 cc 的函数
        dispatch_cutlassF_f32_sm75(cb, cc);
    }
    // 如果 DT 是 float 类型，并且 cc 在 [80, 100) 范围内
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 100) {
        // 调用特定于 float 和指定 cc 的函数
        dispatch_cutlassF_f32_sm80(cb, cc);
    }
}
```