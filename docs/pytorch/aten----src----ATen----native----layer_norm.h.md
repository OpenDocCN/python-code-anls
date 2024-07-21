# `.\pytorch\aten\src\ATen\native\layer_norm.h`

```py
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/accumulate.h>

namespace at::native {

namespace {

// 内部函数，用于检查 Layer Normalization 的输入参数，并返回输入的尺寸 M 和 N
C10_ALWAYS_INLINE std::pair<int64_t, int64_t> _check_layer_norm_inputs(
    const Tensor& input,               // 输入张量
    IntArrayRef normalized_shape,      // 规范化形状的维度
    const Tensor& weight /* optional */, // 权重张量（可选）
    const Tensor& bias /* optional */)  // 偏置张量（可选）
{
  const int normalized_ndim = normalized_shape.size();  // 规范化形状的维度数量
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();  // 输入张量的形状
  const auto input_ndim = input.dim();     // 输入张量的维度数

  // 检查输入张量的形状是否符合规范化形状的要求
  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;  // 计算轴的位置
  const int64_t M =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);  // 计算 M 的大小
  const int64_t N =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());   // 计算 N 的大小

  return std::make_pair(M, N);  // 返回 M 和 N
}

} // namespace

// CPU 版本的 Layer Normalization 函数声明，将结果存储在 out 张量中
void layer_norm_cpu_out(
    at::Tensor& out,
    const at::Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N);

// RMS Norm 函数声明，计算 RMS Norm 并返回张量
Tensor rms_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    std::optional<double> eps);

// 前向传播函数指针类型定义，用于 Layer Normalization
using forward_fn = void (*)(
    const Tensor& /* X */,
    const Tensor& /* gamma */,
    const Tensor& /* beta */,
    int64_t /* M */,
    int64_t /* N */,
    double /* eps */,
    Tensor* /* Y */,
    Tensor* /* mean */,
    Tensor* /* rstd */);

// 反向传播函数指针类型定义，用于 Layer Normalization
using backward_fn = void (*)(
    const Tensor& /* dY */,
    const Tensor& /* X */,
    const Tensor& /* mean */,
    const Tensor& /* rstd */,
    const Tensor& /* gamma */,
    int64_t /* M */,
    int64_t /* N */,
    Tensor* /* dX */,
    Tensor* /* dgamma */,
    Tensor* /* dbeta */);

// 声明前向传播的分发函数，用于 Layer Normalization
DECLARE_DISPATCH(forward_fn, LayerNormKernel);

// 声明反向传播的分发函数，用于 Layer Normalization
DECLARE_DISPATCH(backward_fn, LayerNormBackwardKernel);

} // namespace at::native
```