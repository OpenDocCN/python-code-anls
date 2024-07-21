# `.\pytorch\aten\src\ATen\native\MetaTensor.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_strided_native.h>
#endif

// 定义命名空间 at::native
namespace at::native {

// 定义函数 empty_meta_symint，返回一个 Tensor
Tensor empty_meta_symint(
  SymIntArrayRef size, // 接受 SymIntArrayRef 类型的 size 参数
  std::optional<ScalarType> dtype_opt, // 可选的标量类型参数
  std::optional<Layout> layout_opt, // 可选的布局参数
  std::optional<Device> device_opt, // 可选的设备参数
  std::optional<bool> pin_memory_opt, // 可选的是否固定内存参数
  std::optional<c10::MemoryFormat> memory_format_opt // 可选的内存格式参数
) {
  // 将 SymIntArrayRef 类型的 size 转换为 opt_size，如果转换成功
  auto opt_size = asIntArrayRefSlowOpt(size);
  if (opt_size.has_value()) { // 如果成功获取到 opt_size
    // 调用 at::detail::empty_meta 函数创建一个空的 Tensor
    return at::detail::empty_meta(
        *opt_size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  }
  // 调用 at::detail::empty_symint_meta 函数创建一个空的 Tensor
  return at::detail::empty_symint_meta(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

// 定义函数 empty_strided_meta_symint，返回一个 Tensor
Tensor empty_strided_meta_symint(
  SymIntArrayRef size, // 接受 SymIntArrayRef 类型的 size 参数
  SymIntArrayRef stride, // 接受 SymIntArrayRef 类型的 stride 参数
  std::optional<ScalarType> dtype_opt, // 可选的标量类型参数
  std::optional<Layout> layout_opt, // 可选的布局参数
  std::optional<Device> device_opt, // 可选的设备参数
  std::optional<bool> pin_memory_opt // 可选的是否固定内存参数
) {
  // 调用 at::detail::empty_strided_symint_meta 函数创建一个空的、带步长的 Tensor
  return at::detail::empty_strided_symint_meta(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

} // namespace at::native
```