# `.\pytorch\c10\util\Optional.h`

```
#ifndef C10_UTIL_OPTIONAL_H_
#define C10_UTIL_OPTIONAL_H_

#include <optional>
#include <type_traits>

// Macros.h is not needed, but it does namespace shenanigans that lots
// of downstream code seems to rely on. Feel free to remove it and fix
// up builds.

// 命名空间 c10 开始

namespace c10 {
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::bad_optional_access;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::make_optional;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::nullopt;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::nullopt_t;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::optional;

// detail_ 命名空间开始
namespace detail_ {
// the call to convert<A>(b) has return type A and converts b to type A iff b
// decltype(b) is implicitly convertible to A
// 名为 convert 的模板函数定义，用于将参数转换为指定类型的值
template <class U>
constexpr U convert(U v) {
  return v;
}
} // namespace detail_

// value_or_else 模板函数定义开始
template <class T, class F>
// 使用 constexpr 修饰，接受 optional<T> 和 F 作为参数
constexpr T value_or_else(const optional<T>& v, F&& func) {
  // 静态断言，确保 F 的返回类型可以隐式转换为 T 类型
  static_assert(
      std::is_convertible_v<typename std::invoke_result_t<F>, T>,
      "func parameters must be a callable that returns a type convertible to the value stored in the optional");
  // 如果 optional 对象有值，则返回该值；否则调用 func 返回结果
  return v.has_value() ? *v : detail_::convert<T>(std::forward<F>(func)());
}

template <class T, class F>
// 使用 constexpr 修饰，接受 rvalue 引用的 optional<T> 和 F 作为参数
constexpr T value_or_else(optional<T>&& v, F&& func) {
  // 静态断言，确保 F 的返回类型可以隐式转换为 T 类型
  static_assert(
      std::is_convertible_v<typename std::invoke_result_t<F>, T>,
      "func parameters must be a callable that returns a type convertible to the value stored in the optional");
  // 如果 optional 对象有值，则返回该值；否则调用 func 返回结果
  return v.has_value() ? constexpr_move(std::move(v).contained_val())
                       : detail_::convert<T>(std::forward<F>(func)());
}
} // namespace c10

#endif // C10_UTIL_OPTIONAL_H_
```