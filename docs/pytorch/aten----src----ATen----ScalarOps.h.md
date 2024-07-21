# `.\pytorch\aten\src\ATen\ScalarOps.h`

```
#pragma once
// 预处理指令：确保本头文件只被包含一次

#include <ATen/Tensor.h>
// 包含 ATen 库中的 Tensor 类头文件
#include <c10/core/Scalar.h>
// 包含 c10 库中的 Scalar 类头文件

#ifndef AT_PER_OPERATOR_HEADERS
// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 Functions.h
#include <ATen/Functions.h>
#else
// 否则，包含 scalar_tensor.h
#include <ATen/ops/scalar_tensor.h>
#endif

namespace at::detail {
// 进入 at::detail 命名空间

// 在向 1 元素 CPU 张量填充数值时，直接操作数据指针以避免额外开销。
// 理想情况下，这种快速通道应该在 TensorIterator 中实现，
// 但目前我们也想避免在 TensorIterator 中不可避免的 compute_types 开销。
Tensor& scalar_fill(Tensor& self, const Scalar& value);
// 声明 scalar_fill 函数，用于在张量中填充标量值

TORCH_API Tensor scalar_tensor_static(
    const Scalar& s,
    std::optional<ScalarType> dtype_opt,
    std::optional<Device> device_opt);
// 声明 scalar_tensor_static 函数，用于创建静态标量张量

} // namespace at::detail
// 离开 at::detail 命名空间

// 位于 c10 命名空间中，因为我们使用 ADL 在其中查找这些函数。
namespace c10 {

// FIXME: 应该是 Scalar::toTensor，但目前无法实现，因为需要通过派生类型（不属于核心部分）。
inline at::Tensor scalar_to_tensor(
    const Scalar& s,
    const Device device = at::kCPU) {
  // 如果设备是 CPU
  if (device == at::kCPU) {
    // 返回 CPU 上的静态标量张量
    return at::detail::scalar_tensor_static(s, s.type(), at::kCPU);
  }
  // 返回指定设备上的标量张量
  return at::scalar_tensor(s, at::device(device).dtype(s.type()));
}

} // namespace c10
// 离开 c10 命名空间

namespace at::native {

// 位于 at::native 命名空间中，用于创建包装的标量张量
inline Tensor wrapped_scalar_tensor(
    const Scalar& scalar,
    const Device device = at::kCPU) {
  // 调用 scalar_to_tensor 函数，获取标量对应的张量
  auto tensor = scalar_to_tensor(scalar, device);
  // 设置张量的 wrapped_number 标记为 true
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  // 返回创建的标量张量
  return tensor;
}

} // namespace at::native
// 离开 at::native 命名空间
```