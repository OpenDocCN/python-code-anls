# `.\pytorch\aten\src\ATen\native\cpu\mixed_data_type.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/core/Tensor.h>
// 引入 ATen 库中的 Tensor 类定义

namespace at { namespace native {

inline ScalarType first_type() {
  return ScalarType::Undefined;
}
// 返回标量类型为 Undefined 的默认值

template <typename... Args>
inline ScalarType first_type(const Tensor& arg, const Args&... parameters) {
  return arg.defined() ? arg.scalar_type() : first_type(parameters...);
}
// 递归模板函数，返回参数列表中第一个有效 Tensor 的标量类型，或者 Undefined

template <typename... Args>
inline bool is_mixed_type(const Tensor& input, const Args&... parameters) {
  const auto parameter_type = first_type(parameters...);
  // 获取参数列表中第一个有效 Tensor 的标量类型
  return ((parameter_type != ScalarType::Undefined) &&
          (parameter_type != input.scalar_type()));
}
// 检查输入 Tensor 和参数列表中的 Tensor 是否有不同的标量类型

// currently on CPU, mixed data type is only supported
// when input is 'BFloat16' or 'Half' and parameters are 'Float'
inline void check_mixed_data_type(const Tensor& input) {
  TORCH_CHECK(at::isReducedFloatingType(input.scalar_type()),
      "mixed dtype (CPU): all inputs must share same datatype.");
}
// 检查 CPU 上输入 Tensor 的标量类型是否为减少的浮点类型

template <typename... Args>
inline void check_mixed_data_type(const Tensor& input, const Tensor& parameter, const Args&... parameters) {
  TORCH_CHECK(!parameter.defined() || parameter.scalar_type() == ScalarType::Float,
      "mixed dtype (CPU): expect parameter to have scalar type of Float");
  // 如果参数 Tensor 已定义，则要求其标量类型为 Float
  check_mixed_data_type(input, parameters...);
}
// 递归调用，检查 CPU 上输入 Tensor 和参数列表中的 Tensor 是否有不同的标量类型

inline ScalarType param_scalar_type(const Tensor& t, bool is_mixed_type) {
  return is_mixed_type ? ScalarType::Float : t.scalar_type();
}
// 根据混合类型标志返回参数 Tensor 的标量类型，如果混合类型为真则返回 Float

}}  // namespace at::native
```