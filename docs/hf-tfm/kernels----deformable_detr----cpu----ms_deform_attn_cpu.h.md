# `.\transformers\kernels\deformable_detr\cpu\ms_deform_attn_cpu.h`

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

// 预处理指令，防止头文件被重复包含
#pragma once
// 包含 PyTorch 扩展库头文件
#include <torch/extension.h>

// 前向传播函数声明
at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value,  // 输入特征图值
    const at::Tensor &spatial_shapes,  // 空间形状
    const at::Tensor &level_start_index,  // 级别开始索引
    const at::Tensor &sampling_loc,  // 采样位置
    const at::Tensor &attn_weight,  // 注意力权重
    const int im2col_step);  // im2col 步长

// 反向传播函数声明
std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value,  // 输入特征图值
    const at::Tensor &spatial_shapes,  // 空间形状
    const at::Tensor &level_start_index,  // 级别开始索引
    const at::Tensor &sampling_loc,  // 采样位置
    const at::Tensor &attn_weight,  // 注意力权重
    const at::Tensor &grad_output,  // 梯度输出
    const int im2col_step);  // im2col 步长
```