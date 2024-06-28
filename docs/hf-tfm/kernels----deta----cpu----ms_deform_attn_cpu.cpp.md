# `.\kernels\deta\cpu\ms_deform_attn_cpu.cpp`

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

#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// 定义了一个函数，用于在 CPU 上执行 ms_deform_attn 的前向传播
at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value,                    // 输入张量 value
    const at::Tensor &spatial_shapes,           // 空间形状信息的张量
    const at::Tensor &level_start_index,        // 级别起始索引的张量
    const at::Tensor &sampling_loc,             // 采样位置的张量
    const at::Tensor &attn_weight,              // 注意力权重的张量
    const int im2col_step)                      // im2col 步长参数
{
    // 抛出错误，表示在 CPU 上尚未实现该函数
    AT_ERROR("Not implement on cpu");
}

// 定义了一个函数，用于在 CPU 上执行 ms_deform_attn 的反向传播
std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value,                    // 输入张量 value
    const at::Tensor &spatial_shapes,           // 空间形状信息的张量
    const at::Tensor &level_start_index,        // 级别起始索引的张量
    const at::Tensor &sampling_loc,             // 采样位置的张量
    const at::Tensor &attn_weight,              // 注意力权重的张量
    const at::Tensor &grad_output,              // 梯度输出的张量
    const int im2col_step)                      // im2col 步长参数
{
    // 抛出错误，表示在 CPU 上尚未实现该函数
    AT_ERROR("Not implement on cpu");
}
```