# `.\kernels\deta\ms_deform_attn.h`

```
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

#include "cpu/ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

// 前向传播函数，用于实现可形变注意力机制的前向计算
at::Tensor
ms_deform_attn_forward(
    const at::Tensor &value,                  // 输入张量 value
    const at::Tensor &spatial_shapes,         // 空间形状信息张量
    const at::Tensor &level_start_index,      // 层级起始索引张量
    const at::Tensor &sampling_loc,           // 采样位置张量
    const at::Tensor &attn_weight,            // 注意力权重张量
    const int im2col_step)                    // im2col 步长参数
{
    // 如果输入张量在 CUDA 上，则调用 CUDA 实现的前向函数
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
        // 如果没有编译 GPU 支持，则抛出错误信息
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    // 如果在 CPU 上调用该函数，则抛出错误信息，表明未实现 CPU 版本
    AT_ERROR("Not implemented on the CPU");
}

// 反向传播函数，用于实现可形变注意力机制的反向计算
std::vector<at::Tensor>
ms_deform_attn_backward(
    const at::Tensor &value,                  // 输入张量 value
    const at::Tensor &spatial_shapes,         // 空间形状信息张量
    const at::Tensor &level_start_index,      // 层级起始索引张量
    const at::Tensor &sampling_loc,           // 采样位置张量
    const at::Tensor &attn_weight,            // 注意力权重张量
    const at::Tensor &grad_output,            // 梯度输出张量
    const int im2col_step)                    // im2col 步长参数
{
    // 如果输入张量在 CUDA 上，则调用 CUDA 实现的反向函数
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
        // 如果没有编译 GPU 支持，则抛出错误信息
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    // 如果在 CPU 上调用该函数，则抛出错误信息，表明未实现 CPU 版本
    AT_ERROR("Not implemented on the CPU");
}
```