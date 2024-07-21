# `.\pytorch\c10\util\TypeSafeSignMath.h`

```py
#pragma once

#include <c10/macros/Macros.h>  // 包含 c10 库中的宏定义
#include <limits>               // 包含数值极限定义
#include <type_traits>          // 包含类型特性的标准库

C10_CLANG_DIAGNOSTIC_PUSH()  // 使用 clang 编译时，推送诊断设置

#if C10_CLANG_HAS_WARNING("-Wstring-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wstring-conversion")  // 忽略字符串转换警告
#endif

#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")  // 忽略隐式整型到浮点型转换警告
#endif

namespace c10 {

/// 返回 false，因为对于无符号数 x < 0 不成立
template <typename T>
inline constexpr bool is_negative(
    const T& /*x*/,  // 参数 x，待检查的值
    std::true_type /*is_unsigned*/) {  // 标记为无符号类型
  return false;  // 返回 false
}

/// 返回 true，如果有符号变量 x < 0
template <typename T>
inline constexpr bool is_negative(const T& x, std::false_type /*is_unsigned*/) {
  return x < T(0);  // 检查 x 是否小于零，返回比较结果
}

/// 返回 true，如果 x < 0
/// 注意：对于无符号自定义类型可能会失败
///       大多数情况下，如果自定义类型有 constexpr 构造函数，可以修复此问题。
///       但是，c10::Half 类型不支持 :-(
template <typename T>
inline constexpr bool is_negative(const T& x) {
  return is_negative(x, std::is_unsigned<T>());  // 调用合适的 is_negative 版本进行检查
}

/// 返回无符号变量 x 的符号，作为 0 或 1
template <typename T>
inline constexpr int signum(const T& x, std::true_type /*is_unsigned*/) {
  return T(0) < x;  // 返回是否 x 大于零的比较结果
}

/// 返回有符号变量 x 的符号，作为 -1, 0, 1
template <typename T>
inline constexpr int signum(const T& x, std::false_type /*is_unsigned*/) {
  return (T(0) < x) - (x < T(0));  // 返回根据 x 的值判断的符号
}

/// 返回 x 的符号，作为 -1, 0, 1
/// 注意：对于无符号自定义类型可能会失败
///       大多数情况下，如果自定义类型有 constexpr 构造函数，可以修复此问题。
///       但是，c10::Half 类型不支持 :-(
template <typename T>
inline constexpr int signum(const T& x) {
  return signum(x, std::is_unsigned<T>());  // 调用合适的 signum 版本进行符号判断
}

/// 返回 true，如果 a 和 b 不都为负数
template <typename T, typename U>
inline constexpr bool signs_differ(const T& a, const U& b) {
  return is_negative(a) != is_negative(b);  // 判断 a 和 b 的符号是否相反
}

// 抑制在 GCC 编译时的符号比较警告
// 因为后续不考虑短路规则就会引发警告，参见 https://godbolt.org/z/Tr3Msnz99
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

/// 返回 true，如果 x 大于类型 Limit 的最大值
template <typename Limit, typename T>
inline constexpr bool greater_than_max(const T& x) {
  constexpr bool can_overflow =
      std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;  // 检查是否可能溢出
  return can_overflow && x > std::numeric_limits<Limit>::max();  // 返回是否 x 大于 Limit 类型的最大值
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/// 返回 true，如果 x < 类型 Limit 的最小值。标准比较
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(
    const T& x,
    std::false_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  return x < std::numeric_limits<Limit>::lowest();  // 返回是否 x 小于 Limit 类型的最小值
}
/// 返回 false，因为所有的限制都是有符号的，因此包括负值，但是 x 不能为负，因为它是无符号的
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(
    const T& /*x*/,  // 参数 x，被忽略，表示要比较的值
    std::false_type /*limit_is_unsigned*/,  // 标记：限制类型不是无符号的
    std::true_type /*x_is_unsigned*/) {  // 标记：x 是无符号的
  return false;  // 返回 false
}

/// 如果 x < 0，则返回 true，其中 0 是由 T 构造的
/// 限制类型不是有符号的，因此其最小值为零
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(
    const T& x,  // 参数 x，要比较的值
    std::true_type /*limit_is_unsigned*/,  // 标记：限制类型是无符号的
    std::false_type /*x_is_unsigned*/) {  // 标记：x 不是无符号的
  return x < T(0);  // 返回比较结果 x < 0
}

/// 返回 false，因为两个类型都是无符号的
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(
    const T& /*x*/,  // 参数 x，被忽略，表示要比较的值
    std::true_type /*limit_is_unsigned*/,  // 标记：限制类型是无符号的
    std::true_type /*x_is_unsigned*/) {  // 标记：x 是无符号的
  return false;  // 返回 false
}

/// 如果 x 小于类型 T 的最小值，则返回 true
/// 注意：对于无符号的自定义类型可能会失败
///       大多数情况下，如果自定义类型有 constexpr 构造函数，则可以修复这个问题。
///       然而，值得注意的是，c10::Half 并不具备这样的构造函数。
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(const T& x) {
  return less_than_lowest<Limit>(
      x, std::is_unsigned<Limit>(), std::is_unsigned<T>());  // 调用重载函数 less_than_lowest，根据类型判断比较逻辑
}
```