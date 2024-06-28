# `.\kernels\deformable_detr\cpu\ms_deform_attn_cpu.cpp`

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

// 包含标准库向量头文件
#include <vector>

// 包含 ATen 库的头文件，提供张量操作
#include <ATen/ATen.h>
// 包含 CUDA 上下文头文件，用于处理 CUDA 相关操作
#include <ATen/cuda/CUDAContext.h>

// 定义 CPU 下的前向传播函数，返回 ATen 张量
at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value,               // 输入张量 value
    const at::Tensor &spatial_shapes,      // 空间形状张量
    const at::Tensor &level_start_index,   // 层级起始索引张量
    const at::Tensor &sampling_loc,        // 采样位置张量
    const at::Tensor &attn_weight,         // 注意力权重张量
    const int im2col_step)                 // im2col 步长
{
    // 抛出错误，表明在 CPU 上未实现该函数
    AT_ERROR("Not implement on cpu");
}

// 定义 CPU 下的反向传播函数，返回 ATen 张量向量
std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value,               // 输入张量 value
    const at::Tensor &spatial_shapes,      // 空间形状张量
    const at::Tensor &level_start_index,   // 层级起始索引张量
    const at::Tensor &sampling_loc,        // 采样位置张量
    const at::Tensor &attn_weight,         // 注意力权重张量
    const at::Tensor &grad_output,         // 梯度输出张量
    const int im2col_step)                 // im2col 步长
{
    // 抛出错误，表明在 CPU 上未实现该函数
    AT_ERROR("Not implement on cpu");
}
```