# `.\pytorch\c10\util\complex_utils.h`

```py
// 如果未定义宏 C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H，则产生编译错误
#if !defined(C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "c10/util/complex_utils.h is not meant to be individually included. Include c10/util/complex.h instead."
#endif

// 包含标准库中的 <limits> 头文件
#include <limits>

// 命名空间 c10
namespace c10 {

// 模板：用于判断类型 T 是否为复数类型，初始化为 false_type
template <typename T>
struct is_complex : public std::false_type {};

// 特化模板：当 T 为 std::complex<T> 类型时，设置为 true_type
template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

// 特化模板：当 T 为 c10::complex<T> 类型时，设置为 true_type
template <typename T>
struct is_complex<c10::complex<T>> : public std::true_type {};

// 模板：从 std::complex<double> 中提取 double 类型；对于其他类型保持不变
// TODO: 用更符合 C++17 语法的方式重写
template <typename T>
struct scalar_value_type {
  using type = T;
};

// 特化模板：从 std::complex<T> 中提取 T 类型
template <typename T>
struct scalar_value_type<std::complex<T>> {
  using type = T;
};

// 特化模板：从 c10::complex<T> 中提取 T 类型
template <typename T>
struct scalar_value_type<c10::complex<T>> {
  using type = T;
};

} // namespace c10

// 命名空间 std
namespace std {

// 特化模板：对于 c10::complex<T>，使用 T 的 numeric_limits
template <typename T>
class numeric_limits<c10::complex<T>> : public numeric_limits<T> {};

// 函数：检查 c10::complex<T> 是否具有 NaN 属性
template <typename T>
bool isnan(const c10::complex<T>& v) {
  // 如果实部或虚部有任一部分是 NaN，则返回 true
  return std::isnan(v.real()) || std::isnan(v.imag());
}

} // namespace std
```