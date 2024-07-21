# `.\pytorch\torch\csrc\jit\tensorexpr\operators\matmul.h`

```
#pragma once
// 包含 Torch 的 Tensor Expression 库中的 Kernel 头文件

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// 定义 computeMatmul 函数，计算矩阵乘法
Tensor computeMatmul(
    const std::vector<ArgValue>& inputs,        // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量形状的表达式列表
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长的表达式列表
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device device);  // 指定计算所用的设备

// 定义 computeAddMM 函数，计算矩阵加法
Tensor computeAddMM(
    const std::vector<ArgValue>& inputs,        // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量形状的表达式列表
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长的表达式列表
    const std::optional<ScalarType>& outputType,  // 可选的输出数据类型
    at::Device device);  // 指定计算所用的设备

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```