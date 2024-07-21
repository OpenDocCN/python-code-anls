# `.\pytorch\torch\csrc\api\include\torch\detail\static.h`

```
#pragma once

#include <torch/csrc/utils/variadic.h>
#include <torch/types.h>

#include <cstdint>
#include <type_traits>

namespace torch {
namespace nn {
class Module;
} // namespace nn
} // namespace torch

namespace torch {
namespace detail {
/// Detects if a type T has a forward() method.
template <typename T>
struct has_forward {
  // Declare two types with differing size.
  using yes = int8_t;
  using no = int16_t;

  // Here we declare two functions. The first is only enabled if `&U::forward`
  // is well-formed and returns the `yes` type. In C++, the ellipsis parameter
  // type (`...`) always puts the function at the bottom of overload resolution.
  // This is specified in the standard as: 1) A standard conversion sequence is
  // always better than a user-defined conversion sequence or an ellipsis
  // conversion sequence. 2) A user-defined conversion sequence is always better
  // than an ellipsis conversion sequence This means that if the first overload
  // is viable, it will be preferred over the second as long as we pass any
  // convertible type. The type of `&U::forward` is a pointer type, so we can
  // pass e.g. 0.
  
  // 测试是否存在 forward() 方法
  template <typename U>
  static yes test(decltype(&U::forward));
  
  // 默认情况下的测试，使用省略号，优先级低于具体函数定义
  template <typename U>
  static no test(...);

  // 最终静态测试所选重载的返回类型大小是否与 yes 类型相同
  static constexpr bool value = (sizeof(test<T>(nullptr)) == sizeof(yes));
};

// 检查模板参数是否不是左值引用的类型
template <typename Head = void, typename... Tail>
constexpr bool check_not_lvalue_references() {
  return (!std::is_lvalue_reference<Head>::value ||
          std::is_const<typename std::remove_reference<Head>::type>::value) &&
      check_not_lvalue_references<Tail...>();
}

// 对于没有模板参数的特化版本，始终返回 true
template <>
inline constexpr bool check_not_lvalue_references<void>() {
  return true;
}

/// A type trait whose `value` member is true if `M` derives from `Module`.
// 如果 M 类型从 Module 派生，则该类型特性的 value 成员为 true
template <typename M>
using is_module =
    std::is_base_of<torch::nn::Module, typename std::decay<M>::type>;

// 如果 M 类型是 Module 的派生类，则启用 T 类型；否则不启用
template <typename M, typename T = void>
using enable_if_module_t =
    typename std::enable_if<is_module<M>::value, T>::type;
} // namespace detail
} // namespace torch
```