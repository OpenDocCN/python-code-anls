# `.\kernels\deformable_detr\cpu\ms_deform_attn_cpu.h`

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

// 预处理指令，确保头文件只被包含一次
#pragma once

// 包含 PyTorch C++ 扩展库头文件
#include <torch/extension.h>

// 前向传播函数声明，计算注意力机制的前向传播
at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value,               // 输入的特征值张量
    const at::Tensor &spatial_shapes,      // 空间形状信息张量
    const at::Tensor &level_start_index,   // 层级起始索引张量
    const at::Tensor &sampling_loc,        // 采样位置张量
    const at::Tensor &attn_weight,         // 注意力权重张量
    const int im2col_step);                // im2col 步长

// 反向传播函数声明，计算注意力机制的反向传播
std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value,               // 输入的特征值张量
    const at::Tensor &spatial_shapes,      // 空间形状信息张量
    const at::Tensor &level_start_index,   // 层级起始索引张量
    const at::Tensor &sampling_loc,        // 采样位置张量
    const at::Tensor &attn_weight,         // 注意力权重张量
    const at::Tensor &grad_output,         // 梯度输出张量
    const int im2col_step);                // im2col 步长
```