# `.\kernels\deta\cuda\ms_deform_attn_cuda.h`

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

// 包含 Torch C++ 扩展的头文件
#pragma once
#include <torch/extension.h>

// 声明 CUDA 前向传播函数，接受多个张量和整数参数
at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value,                    // 输入特征值张量
    const at::Tensor &spatial_shapes,           // 空间形状信息张量
    const at::Tensor &level_start_index,        // 层级起始索引张量
    const at::Tensor &sampling_loc,             // 采样位置张量
    const at::Tensor &attn_weight,              // 注意力权重张量
    const int im2col_step);                     // im2col 步长整数参数

// 声明 CUDA 反向传播函数，接受多个张量和整数参数，并返回张量向量
std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value,                    // 输入特征值张量
    const at::Tensor &spatial_shapes,           // 空间形状信息张量
    const at::Tensor &level_start_index,        // 层级起始索引张量
    const at::Tensor &sampling_loc,             // 采样位置张量
    const at::Tensor &attn_weight,              // 注意力权重张量
    const at::Tensor &grad_output,              // 梯度输出张量
    const int im2col_step);                     // im2col 步长整数参数
```