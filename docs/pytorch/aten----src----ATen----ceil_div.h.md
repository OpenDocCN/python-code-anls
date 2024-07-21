# `.\pytorch\aten\src\ATen\ceil_div.h`

```py
#pragma once
#include <c10/macros/Macros.h>
#include <type_traits>

namespace at {

/**
   计算 ceil(a / b)，即 a 除以 b 向上取整的结果
   T 表示模板参数类型，必须是整数类型（std::is_integral_v<T> 条件）
*/
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
C10_ALWAYS_INLINE C10_HOST_DEVICE T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

/**
   计算 ceil(a / b) * b，即将 a 向上舍入到最接近 b 的整数倍的结果
   T 表示模板参数类型
*/
template <typename T>
C10_ALWAYS_INLINE C10_HOST_DEVICE T round_up(T a, T b) {
  return ceil_div(a, b) * b;
}

} // namespace at
```