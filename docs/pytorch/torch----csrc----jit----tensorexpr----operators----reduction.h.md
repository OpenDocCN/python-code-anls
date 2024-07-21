# `.\pytorch\torch\csrc\jit\tensorexpr\operators\reduction.h`

```py
#pragma once

# 预处理指令，确保此头文件只包含一次


#include <torch/csrc/jit/tensorexpr/kernel.h>

# 包含 Torch 库中的头文件 kernel.h


namespace torch {
namespace jit {
namespace tensorexpr {

# 命名空间开始：定义 torch、jit 和 tensorexpr 命名空间


TORCH_API Tensor computeSum(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

# 声明 computeSum 函数，计算输入数据的和，返回一个 Tensor 对象


TORCH_API Tensor computeMean(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

# 声明 computeMean 函数，计算输入数据的均值，返回一个 Tensor 对象


TORCH_API Tensor computeAdaptiveAvgPool2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

# 声明 computeAdaptiveAvgPool2d 函数，执行自适应平均池化操作，返回一个 Tensor 对象


Tensor computeMax(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

# 声明 computeMax 函数，计算输入数据的最大值，返回一个 Tensor 对象


} // namespace tensorexpr
} // namespace jit
} // namespace torch

# 命名空间结束：结束 torch、jit 和 tensorexpr 命名空间
```