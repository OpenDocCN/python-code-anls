# `.\pytorch\torch\csrc\jit\tensorexpr\operators\quantization.h`

```py
#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// 定义了一系列的 Torch API，用于量化计算

// 从参数中获取张量级别的量化参数表达式
TORCH_API ExprHandle quantizePerTensorQParamFromArg(ArgValue arg);

// 获取输入缓存的量化比例因子
TORCH_API double immQScale(const BufHandle& qx);

// 获取输入缓存的量化零点
TORCH_API int64_t immQZero(const BufHandle& qx);

// 获取输入缓存的量化数据类型
TORCH_API ScalarType immQDType(const BufHandle& qx);

// 检查输入缓存是否已经量化
TORCH_API bool isQuantized(const BufHandle& qx);

// 计算对张量进行量化的函数，返回量化后的张量
TORCH_API Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 外部调用的函数，用于计算对张量进行量化的操作
TORCH_API Tensor computeQuantizePerTensorExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算一维量化卷积的函数
TORCH_API Tensor computeQuantizedConv1d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算二维量化卷积的预打包函数
TORCH_API Tensor computeQuantizedConv2dPrepack(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算一维量化卷积的函数（重复定义）
TORCH_API Tensor computeQuantizedConv1d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算二维量化卷积的函数
TORCH_API Tensor computeQuantizedConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算带ReLU的二维量化卷积的函数
TORCH_API Tensor computeQuantizedConv2dRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算量化线性层的函数
TORCH_API Tensor computeQuantizedLinear(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算带ReLU的量化线性层的函数
TORCH_API Tensor computeQuantizedLinearRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算量化加法的函数
TORCH_API Tensor computeQuantizedAdd(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 外部调用的函数，用于计算量化加法的操作
Tensor computeQuantizedAddExternalCall(
    // 参数同上，但未完全定义
    // inputs 参数是一个包含 ArgValue 元素的常量引用向量，用于传递输入参数列表
    const std::vector<ArgValue>& inputs,
    // outputShape 参数是一个包含 ExprHandle 元素的常量引用向量，表示输出张量的形状
    const std::vector<ExprHandle>& outputShape,
    // outputStrides 参数是一个包含 ExprHandle 元素的常量引用向量，表示输出张量的步幅
    const std::vector<ExprHandle>& outputStrides,
    // outputType 是一个可选的标量类型，表示输出张量的数据类型
    const std::optional<ScalarType>& outputType,
    // device 参数表示计算设备，是一个 at::Device 类型的对象
    at::Device device);
// 计算量化乘法操作的函数签名和声明
TORCH_API Tensor computeQuantizedMul(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算量化乘法的标量版本的函数签名和声明
TORCH_API Tensor computeQuantizedMulScalar(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算量化拼接操作的函数签名和声明
TORCH_API Tensor computeQuantizedCat(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算量化ReLU操作的函数签名和声明
TORCH_API Tensor computeQuantizedRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算反量化操作的函数签名和声明
TORCH_API Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算外部调用的反量化操作的函数签名和声明
TORCH_API Tensor computeDequantizeExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算2D最近邻上采样操作的函数签名和声明
TORCH_API Tensor computeUpsampleNearest2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算外部调用的2D最近邻上采样操作的函数签名和声明
TORCH_API Tensor computeUpsampleNearest2dExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

// 计算外部调用的量化Sigmoid操作的函数签名和声明
TORCH_API Tensor computeQuantizedSigmoidExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device);
```