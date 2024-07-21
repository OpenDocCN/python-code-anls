# `.\pytorch\aten\src\ATen\native\mkldnn\Relu.cpp`

```
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/relu_native.h>                // for mkldnn_relu, mkldnn_...
#include <ATen/ops/threshold_backward_native.h>  // for mkldnn_relu_backward
#endif



// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含原生函数头文件，否则包含具体操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/relu_native.h>                // 包含 mkldnn_relu, mkldnn_... 相关操作
#include <ATen/ops/threshold_backward_native.h>  // 包含 mkldnn_relu_backward 相关操作
#endif



#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_relu(const Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu: ATen not compiled with MKLDNN support");
}



// 如果未启用 MKLDNN 支持，则定义 mkldnn_relu 函数并抛出错误
#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

// 对于输入张量进行 relu 操作，如果未启用 MKLDNN，则抛出错误
Tensor mkldnn_relu(const Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu: ATen not compiled with MKLDNN support");
}



Tensor& mkldnn_relu_(Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu_: ATen not compiled with MKLDNN support");
}



// 对于输入张量进行原位操作的 relu，如果未启用 MKLDNN，则抛出错误
Tensor& mkldnn_relu_(Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu_: ATen not compiled with MKLDNN support");
}



Tensor mkldnn_relu_backward(const Tensor& grad_output, const Tensor& input, const Scalar& threshold) {
  TORCH_CHECK(false, "mkldnn_relu_backward: ATen not compiled with MKLDNN support");
}



// 对于输入张量进行 relu 反向传播操作，如果未启用 MKLDNN，则抛出错误
Tensor mkldnn_relu_backward(const Tensor& grad_output, const Tensor& input, const Scalar& threshold) {
  TORCH_CHECK(false, "mkldnn_relu_backward: ATen not compiled with MKLDNN support");
}



#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

Tensor mkldnn_relu(const Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }



// 如果启用了 MKLDNN 支持，则定义相关函数实现
#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

// 对于输入张量进行 relu 操作的具体实现
Tensor mkldnn_relu(const Tensor& input) {
  // 如果输入张量类型为 BFloat16，则检查设备是否支持相应的 CPU 指令集
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  // 将输入张量转换为 MKLDNN 格式的张量
  const ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor y;
  // 执行 MKLDNN 的 relu 操作
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  // 将 MKLDNN 格式的输出张量转换为新的 Torch 张量
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}



Tensor& mkldnn_relu_(Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu_: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  ideep::tensor& x = itensor_from_mkldnn(input);
  // 原位执行 MKLDNN 的 relu 操作
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return input;
}



Tensor mkldnn_relu_backward(const Tensor& grad_output, const Tensor& input, const Scalar& threshold) {
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor grady = itensor_from_mkldnn(grad_output);
  ideep::tensor gradx;
  // 执行 MKLDNN 的 relu 反向传播操作
  ideep::eltwise_backward::compute(x, grady, gradx,
      ideep::algorithm::eltwise_relu, /*alpha*/ 0.0);
  // 将 MKLDNN 格式的输出张量转换为新的 Torch 张量
  return new_with_itensor_mkldnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}



#endif // AT_MKLDNN_ENABLED



#endif // AT_PER_OPERATOR_HEADERS
```