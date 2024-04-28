# `.\transformers\kernels\deformable_detr\ms_deform_attn.h`

```py
/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once

// 包含 CPU 端的 deformable attention 头文件
#include "cpu/ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
// 如果编译支持 CUDA，则包含 CUDA 端的 deformable attention 头文件
#include "cuda/ms_deform_attn_cuda.h"
#endif

// 定义 ms_deform_attn_forward 函数，实现 deformable attention 的前向传播
at::Tensor
ms_deform_attn_forward(
    const at::Tensor &value,  // 输入特征 value
    const at::Tensor &spatial_shapes,  // 空间形状信息
    const at::Tensor &level_start_index,  // 每级开始索引
    const at::Tensor &sampling_loc,  // 采样位置
    const at::Tensor &attn_weight,  // 注意力权重
    const int im2col_step)  // im2col 步长
{
    // 如果输入特征在 CUDA 上，则执行 CUDA 端的前向传播
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
        // 如果编译不支持 CUDA，则报错
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    // 如果输入特征在 CPU 上，则报错
    AT_ERROR("Not implemented on the CPU");
}

// 定义 ms_deform_attn_backward 函数，实现 deformable attention 的反向传播
std::vector<at::Tensor>
ms_deform_attn_backward(
    const at::Tensor &value,  // 输入特征 value
    const at::Tensor &spatial_shapes,  // 空间形状信息
    const at::Tensor &level_start_index,  // 每级开始索引
    const at::Tensor &sampling_loc,  // 采样位置
    const at::Tensor &attn_weight,  // 注意力权重
    const at::Tensor &grad_output,  // 梯度输出
    const int im2col_step)  // im2col 步长
{
    // 如果输入特征在 CUDA 上，则执行 CUDA 端的反向传播
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
        // 如果编译不支持 CUDA，则报错
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    // 如果输入特征在 CPU 上，则报错
    AT_ERROR("Not implemented on the CPU");
}
```