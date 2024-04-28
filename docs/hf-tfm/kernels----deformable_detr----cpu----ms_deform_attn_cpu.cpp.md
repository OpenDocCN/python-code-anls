# `.\transformers\kernels\deformable_detr\cpu\ms_deform_attn_cpu.cpp`

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

// 包含必要的头文件
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// 前向传播函数的声明，CPU 版本
at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    // 抛出错误，表示在 CPU 上未实现
    AT_ERROR("Not implement on cpu");
}

// 反向传播函数的声明，CPU 版本
std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    // 抛出错误，表示在 CPU 上未实现
    AT_ERROR("Not implement on cpu");
}
```