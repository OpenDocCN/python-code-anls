# `.\transformers\kernels\deformable_detr\cuda\ms_deform_attn_cuda.h`

```py
// 包含torch扩展的头文件
#pragma once
#include <torch/extension.h>

// 前向传播函数声明，接受value、spatial_shapes、level_start_index、sampling_loc、attn_weight和im2col_step作为参数，返回一个Tensor
at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value,  // 输入特征张量
    const at::Tensor &spatial_shapes,  // 空间形状张量
    const at::Tensor &level_start_index,  // 级别开始索引张量
    const at::Tensor &sampling_loc,  // 采样位置张量
    const at::Tensor &attn_weight,  // 注意力权重张量
    const int im2col_step);  // im2col步长

// 反向传播函数声明，接受value、spatial_shapes、level_start_index、sampling_loc、attn_weight、grad_output和im2col_step作为参数，返回一个Tensor向量
std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value,  // 输入特征张量
    const at::Tensor &spatial_shapes,  // 空间形状张量
    const at::Tensor &level_start_index,  // 级别开始索引张量
    const at::Tensor &sampling_loc,  // 采样位置张量
    const at::Tensor &attn_weight,  // 注意力权重张量
    const at::Tensor &grad_output,  // 梯度输出张量
    const int im2col_step);  // im2col步长
```