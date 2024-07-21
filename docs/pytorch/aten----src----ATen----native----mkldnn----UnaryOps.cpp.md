# `.\pytorch\aten\src\ATen\native\mkldnn\UnaryOps.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sigmoid_native.h>          // for mkldnn_sigmoid, mkldnn_...
#include <ATen/ops/tanh_native.h>             // for mkldnn_tanh, mkldnn_tanh_
#endif

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

// 如果 ATen 没有编译 MKLDNN 支持，则以下函数将抛出错误信息
Tensor mkldnn_sigmoid(const Tensor& self) {
  TORCH_CHECK(false, "mkldnn_sigmoid: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  TORCH_CHECK(false, "mkldnn_sigmoid_: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_tanh(const Tensor& self) {
  TORCH_CHECK(false, "mkldnn_tanh: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_tanh_(Tensor& self) {
  TORCH_CHECK(false, "mkldnn_tanh_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

// 如果 ATen 编译了 MKLDNN 支持，则以下函数提供 MKLDNN 加速的 sigmoid 操作
Tensor mkldnn_sigmoid(const Tensor& self) {
  // 将 ATen Tensor 转换为 MKLDNN 的 ideep::tensor 对象
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  // 执行 MKLDNN 的 sigmoid 操作
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  // 将计算结果转换为 ATen Tensor 返回
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

// 如果 ATen 编译了 MKLDNN 支持，则以下函数提供原地 MKLDNN 加速的 sigmoid 操作
Tensor& mkldnn_sigmoid_(Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  // 执行原地 MKLDNN 的 sigmoid 操作
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  // 返回自身 Tensor
  return self;
}

// 如果 ATen 编译了 MKLDNN 支持，则以下函数提供 MKLDNN 加速的 tanh 操作
Tensor mkldnn_tanh(const Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  // 执行 MKLDNN 的 tanh 操作
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_tanh, ideep::prop_kind::forward);
  // 将计算结果转换为 ATen Tensor 返回
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

// 如果 ATen 编译了 MKLDNN 支持，则以下函数提供原地 MKLDNN 加速的 tanh 操作
Tensor& mkldnn_tanh_(Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  // 执行原地 MKLDNN 的 tanh 操作
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_tanh, ideep::prop_kind::forward);
  // 返回自身 Tensor
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED
```