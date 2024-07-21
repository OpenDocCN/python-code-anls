# `.\pytorch\torch\csrc\distributed\c10d\quantization\quantization_gpu.h`

```py
// 版权声明及许可信息，声明代码版权归 Meta Platforms, Inc. 及其关联公司所有
//
// 此源代码根据根目录中的 LICENSE 文件中的 BSD 风格许可证进行许可

// 预处理指令，确保头文件只被编译一次
#pragma once

// 包含 ATen 库的头文件
#include <ATen/ATen.h>

// 包含 vector 标准库的头文件
#include <vector>

// 定义命名空间 torch::distributed::c10d::quantization
namespace torch::distributed::c10d::quantization {

// 声明函数 _float_to_bfloat16_cuda，将输入张量从 float 转换为 bfloat16 类型，并在 CUDA 上执行
at::Tensor _float_to_bfloat16_cuda(const at::Tensor& input);

// 声明函数 _bfloat16_to_float_cuda，将输入张量从 bfloat16 转换为 float 类型，并在 CUDA 上执行
at::Tensor _bfloat16_to_float_cuda(const at::Tensor& input);

} // namespace torch::distributed::c10d::quantization
```