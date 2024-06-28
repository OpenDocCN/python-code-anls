# `.\kernels\deformable_detr\cuda\ms_deform_attn_cuda.h`

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

// 包含 Torch C++ 扩展库的头文件
#include <torch/extension.h>

// 声明 CUDA 前向函数，计算多尺度可变形注意力机制的前向传播
at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value,           // 输入张量：特征图
    const at::Tensor &spatial_shapes,  // 输入张量：空间形状
    const at::Tensor &level_start_index,  // 输入张量：每级起始索引
    const at::Tensor &sampling_loc,    // 输入张量：采样位置
    const at::Tensor &attn_weight,     // 输入张量：注意力权重
    const int im2col_step              // 输入整数：im2col 步骤
);

// 声明 CUDA BF16（BFloat16）前向函数，计算多尺度可变形注意力机制的前向传播
at::Tensor ms_deform_attn_cuda_forward_bf16(
    const at::Tensor &value,           // 输入张量：特征图
    const at::Tensor &spatial_shapes,  // 输入张量：空间形状
    const at::Tensor &level_start_index,  // 输入张量：每级起始索引
    const at::Tensor &sampling_loc,    // 输入张量：采样位置
    const at::Tensor &attn_weight,     // 输入张量：注意力权重
    const int im2col_step              // 输入整数：im2col 步骤
);

// 声明 CUDA 反向函数，计算多尺度可变形注意力机制的反向传播
std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value,           // 输入张量：特征图
    const at::Tensor &spatial_shapes,  // 输入张量：空间形状
    const at::Tensor &level_start_index,  // 输入张量：每级起始索引
    const at::Tensor &sampling_loc,    // 输入张量：采样位置
    const at::Tensor &attn_weight,     // 输入张量：注意力权重
    const at::Tensor &grad_output,     // 输入张量：梯度输出
    const int im2col_step              // 输入整数：im2col 步骤
);

// 声明 CUDA BF16（BFloat16）反向函数，计算多尺度可变形注意力机制的反向传播
std::vector<at::Tensor> ms_deform_attn_cuda_backward_bf16(
    const at::Tensor &value,           // 输入张量：特征图
    const at::Tensor &spatial_shapes,  // 输入张量：空间形状
    const at::Tensor &level_start_index,  // 输入张量：每级起始索引
    const at::Tensor &sampling_loc,    // 输入张量：采样位置
    const at::Tensor &attn_weight,     // 输入张量：注意力权重
    const at::Tensor &grad_output,     // 输入张量：梯度输出
    const int im2col_step              // 输入整数：im2col 步骤
);
```