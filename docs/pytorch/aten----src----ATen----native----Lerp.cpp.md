# `.\pytorch\aten\src\ATen\native\Lerp.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Lerp.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/lerp_native.h>
#endif

namespace at::meta {

// 定义了一个元函数 lerp_Tensor，用于计算两个张量之间的线性插值
TORCH_META_FUNC(lerp_Tensor)(
    const Tensor& self, const Tensor& end, const Tensor& weight) {
  // 检查输入张量的数据类型是否一致
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(),
              " for `end` but got dtype ", end.dtype());
  // 检查输入张量的数据类型是否与权重张量的数据类型一致
  TORCH_CHECK(self.dtype() == weight.dtype(), "expected dtype ", self.dtype(),
              " for `weight` but got dtype ", weight.dtype());
  // 构建张量迭代器配置，添加输出张量（如果可能获取），以及三个输入张量
  build(at::TensorIteratorConfig()
        .add_output(maybe_get_output())
        .add_const_input(self)
        .add_const_input(end)
        .add_const_input(weight));
}

// 定义了一个元函数 lerp_Scalar，用于计算张量与标量之间的线性插值
TORCH_META_FUNC(lerp_Scalar)(
    const Tensor& self, const Tensor& end, const Scalar& /*weight*/) {
  // 检查输入张量的数据类型是否一致
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(),
              " for `end` but got dtype ", end.dtype());
  // 构建二元操作，用于计算张量和结束张量之间的插值
  build_binary_op(maybe_get_output(), self, end);
}

}  // namespace at::meta

namespace at::native {

// 实现了 lerp_Tensor 的具体操作函数，用于在设备上执行张量和权重之间的线性插值
TORCH_IMPL_FUNC(lerp_Tensor)(
    const Tensor& /*self*/, const Tensor& /*end*/, const Tensor& weight, const Tensor& /*out*/) {
  // 调用特定设备上的线性插值核心函数 lerp_kernel_tensor_weight
  lerp_kernel_tensor_weight(device_type(), *this);
}

// 实现了 lerp_Scalar 的具体操作函数，用于在设备上执行张量和标量之间的线性插值
TORCH_IMPL_FUNC(lerp_Scalar)(
    const Tensor& /*self*/, const Tensor& /*end*/, const Scalar& weight, const Tensor& /*out*/) {
  // 调用特定设备上的线性插值核心函数 lerp_kernel_scalar_weight，并传入权重标量
  lerp_kernel_scalar_weight(device_type(), *this, weight);
}

// 定义了线性插值核心函数 lerp_kernel_scalar_weight 的调度分发
DEFINE_DISPATCH(lerp_kernel_scalar_weight);

// 定义了线性插值核心函数 lerp_kernel_tensor_weight 的调度分发
DEFINE_DISPATCH(lerp_kernel_tensor_weight);

} // namespace at::native
```