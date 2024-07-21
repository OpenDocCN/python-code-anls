# `.\pytorch\torch\csrc\jit\tensorexpr\operators\conv2d.h`

```
#pragma once

#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// 定义了一个命名空间 tensorexpr，用于存放与张量表达式相关的函数和类

// 使用 BufHandle 类型的参数 input、weight、bias，以及 int 类型的参数 stride、pad 和 groups，
// 计算带偏置的二维深度可分离卷积，返回一个 Tensor 对象
TORCH_API Tensor conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    int stride,
    int pad,
    int groups);

// 使用 BufHandle 类型的参数 input、weight，以及 int 类型的参数 stride、pad 和 groups，
// 计算不带偏置的二维深度可分离卷积，返回一个 Tensor 对象
TORCH_API Tensor conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    int stride,
    int pad,
    int groups);

// 使用 BufHandle 类型的参数 input、weight、bias，以及 ExprHandle 类型的参数 N、C、H、W、K、CperG、R、S、stride、pad 和 groups，
// 计算带偏置的二维深度可分离卷积，返回一个 Tensor 对象
TORCH_API Tensor conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups);

// 使用 BufHandle 类型的参数 input、weight，以及 ExprHandle 类型的参数 N、C、H、W、K、CperG、R、S、stride、pad 和 groups，
// 计算不带偏置的二维深度可分离卷积，返回一个 Tensor 对象
TORCH_API Tensor conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups);

// 检查给定的卷积参数和张量信息是否支持计算
bool conv2dIsSupported(
    const TensorInfo& input,
    const TensorInfo& weight,
    const TensorInfo& bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& pad,
    const std::vector<int64_t>& dilation,
    int64_t groups);

// 检查使用 MKL-DNN 预打包卷积算法的支持情况
bool mkldnnPrepackedConvIsSupported(
    const TensorInfo& input,
    const TensorInfo& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& pad,
    const std::vector<int64_t>& dilation,
    int64_t groups);

// 根据输入、输出形状、步长和设备，计算二维卷积的结果 Tensor
Tensor computeConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 根据输入、输出形状、步长和设备，计算一维卷积的结果 Tensor
Tensor computeConv1d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 根据输入、输出形状、步长和设备，使用 Clamp 运行预打包的二维卷积计算结果 Tensor
Tensor computePrepackedConv2dClampRun(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 根据输入、输出形状、步长和设备，使用 Clamp 运行预打包的线性计算结果 Tensor
Tensor computePrepackedLinearClampRun(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 根据输入、输出形状、步长和设备，使用 MKL-DNN 预打包卷积运行结果 Tensor
Tensor computeMkldnnPrepackedConvRun(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```