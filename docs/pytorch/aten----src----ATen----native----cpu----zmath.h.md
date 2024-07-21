# `.\pytorch\aten\src\ATen\native\cpu\zmath.h`

```
#pragma once
// 用于复杂数的数学运算，对于其他数据类型则不执行任何操作。

#include <c10/util/complex.h>         // 包含复数操作相关的头文件
#include <c10/util/MathConstants.h>   // 包含数学常量的头文件
#include <ATen/NumericUtils.h>        // 包含数值计算工具的头文件

namespace at { namespace native {
inline namespace CPU_CAPABILITY {

template <typename SCALAR_TYPE, typename VALUE_TYPE=SCALAR_TYPE>
inline VALUE_TYPE zabs (SCALAR_TYPE z) {
  return z;  // 对于非复数类型，返回原始值
}

template<>
inline c10::complex<float> zabs <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(std::abs(z));  // 返回复数的绝对值
}

template<>
inline float zabs <c10::complex<float>, float> (c10::complex<float> z) {
  return std::abs(z);  // 返回复数的绝对值
}

template<>
inline c10::complex<double> zabs <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(std::abs(z));  // 返回复数的绝对值
}

template<>
inline double zabs <c10::complex<double>, double> (c10::complex<double> z) {
  return std::abs(z);  // 返回复数的绝对值
}

// 这个重载对应于非复数类型。
// 该函数在非复数类型上与其NumPy等价函数一致，
// 对于负实数返回`pi`，对于零或正实数返回`0`。
// 注意：会传播`nan`。
template <typename SCALAR_TYPE, typename VALUE_TYPE=SCALAR_TYPE>
inline VALUE_TYPE angle_impl (SCALAR_TYPE z) {
  if (at::_isnan(z)) {
    return z;  // 如果是NaN则直接返回
  }
  return z < 0 ? c10::pi<double> : 0;  // 如果小于0则返回π，否则返回0
}

template<>
inline c10::complex<float> angle_impl <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(std::arg(z), 0.0);  // 返回复数的辐角
}

template<>
inline float angle_impl <c10::complex<float>, float> (c10::complex<float> z) {
  return std::arg(z);  // 返回复数的辐角
}

template<>
inline c10::complex<double> angle_impl <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(std::arg(z), 0.0);  // 返回复数的辐角
}

template<>
inline double angle_impl <c10::complex<double>, double> (c10::complex<double> z) {
  return std::arg(z);  // 返回复数的辐角
}

template <typename SCALAR_TYPE, typename VALUE_TYPE=SCALAR_TYPE>
constexpr VALUE_TYPE real_impl (SCALAR_TYPE z) {
  return z;  // 对于非复数类型，返回原始值
}

template<>
constexpr c10::complex<float> real_impl <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(z.real(), 0.0);  // 返回复数的实部
}

template<>
constexpr float real_impl <c10::complex<float>, float> (c10::complex<float> z) {
  return z.real();  // 返回复数的实部
}

template<>
constexpr c10::complex<double> real_impl <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(z.real(), 0.0);  // 返回复数的实部
}

template<>
constexpr double real_impl <c10::complex<double>, double> (c10::complex<double> z) {
  return z.real();  // 返回复数的实部
}

template <typename SCALAR_TYPE, typename VALUE_TYPE=SCALAR_TYPE>
constexpr VALUE_TYPE imag_impl (SCALAR_TYPE /*z*/) {
  return 0;  // 对于非复数类型，返回0
}

template<>
constexpr c10::complex<float> imag_impl <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(z.imag(), 0.0);  // 返回复数的虚部
}

template<>
constexpr float imag_impl <c10::complex<float>, float> (c10::complex<float> z) {
  return z.imag();  // 返回复数的虚部
}

template<>
// 定义一个 constexpr 函数，返回复数 z 的虚部构成的复数，实部为 0.0
constexpr c10::complex<double> imag_impl <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(z.imag(), 0.0);
}

// 模板特化，返回复数 z 的虚部，实部为 0.0
template<>
constexpr double imag_impl <c10::complex<double>, double> (c10::complex<double> z) {
  return z.imag();
}

// 普通模板，返回 z 本身，即不进行任何操作
template <typename TYPE>
inline TYPE conj_impl (TYPE z) {
  return z; //No-Op
}

// 模板特化，返回复数 z 的共轭，实部不变，虚部取反
template<>
inline c10::complex<at::Half> conj_impl <c10::complex<at::Half>> (c10::complex<at::Half> z) {
  return c10::complex<at::Half>{z.real(), -z.imag()};
}

// 模板特化，返回复数 z 的共轭，实部不变，虚部取反
template<>
inline c10::complex<float> conj_impl <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(z.real(), -z.imag());
}

// 模板特化，返回复数 z 的共轭，实部不变，虚部取反
template<>
inline c10::complex<double> conj_impl <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(z.real(), -z.imag());
}

// 普通模板，返回 z 的向上取整结果
template <typename TYPE>
inline TYPE ceil_impl (TYPE z) {
  return std::ceil(z);
}

// 模板特化，返回复数 z 的向上取整结果，分别对实部和虚部进行操作
template <>
inline c10::complex<float> ceil_impl (c10::complex<float> z) {
  return c10::complex<float>(std::ceil(z.real()), std::ceil(z.imag()));
}

// 模板特化，返回复数 z 的向上取整结果，分别对实部和虚部进行操作
template <>
inline c10::complex<double> ceil_impl (c10::complex<double> z) {
  return c10::complex<double>(std::ceil(z.real()), std::ceil(z.imag()));
}

// 模板函数，返回复数 z 的符号函数结果
template<typename T>
inline c10::complex<T> sgn_impl (c10::complex<T> z) {
  if (z == c10::complex<T>(0, 0)) {
    return c10::complex<T>(0, 0);
  } else {
    return z / zabs(z);
  }
}

// 普通模板，返回 z 的向下取整结果
template <typename TYPE>
inline TYPE floor_impl (TYPE z) {
  return std::floor(z);
}

// 模板特化，返回复数 z 的向下取整结果，分别对实部和虚部进行操作
template <>
inline c10::complex<float> floor_impl (c10::complex<float> z) {
  return c10::complex<float>(std::floor(z.real()), std::floor(z.imag()));
}

// 模板特化，返回复数 z 的向下取整结果，分别对实部和虚部进行操作
template <>
inline c10::complex<double> floor_impl (c10::complex<double> z) {
  return c10::complex<double>(std::floor(z.real()), std::floor(z.imag()));
}

// 普通模板，返回 z 的四舍五入结果
template <typename TYPE>
inline TYPE round_impl (TYPE z) {
  return std::nearbyint(z);
}

// 模板特化，返回复数 z 的四舍五入结果，分别对实部和虚部进行操作
template <>
inline c10::complex<float> round_impl (c10::complex<float> z) {
  return c10::complex<float>(std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

// 模板特化，返回复数 z 的四舍五入结果，分别对实部和虚部进行操作
template <>
inline c10::complex<double> round_impl (c10::complex<double> z) {
  return c10::complex<double>(std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

// 普通模板，返回 z 的截断整数部分结果
template <typename TYPE>
inline TYPE trunc_impl (TYPE z) {
  return std::trunc(z);
}

// 模板特化，返回复数 z 的截断整数部分结果，分别对实部和虚部进行操作
template <>
inline c10::complex<float> trunc_impl (c10::complex<float> z) {
  return c10::complex<float>(std::trunc(z.real()), std::trunc(z.imag()));
}

// 模板特化，返回复数 z 的截断整数部分结果，分别对实部和虚部进行操作
template <>
inline c10::complex<double> trunc_impl (c10::complex<double> z) {
  return c10::complex<double>(std::trunc(z.real()), std::trunc(z.imag()));
}

// 普通模板，返回两个值中的最大值
template <typename TYPE, std::enable_if_t<!c10::is_complex<TYPE>::value, int> = 0>
inline TYPE max_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
    return std::numeric_limits<TYPE>::quiet_NaN();
  } else {
    return std::max(a, b);
  }
}

// 模板特化，返回两个复数中的实部和虚部分别的最大值
template <typename TYPE, std::enable_if_t<c10::is_complex<TYPE>::value, int> = 0>
inline TYPE max_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a)) {
    # 如果 a 是 NaN，则返回 a
    return a;
  } else if (_isnan<TYPE>(b)) {
    # 如果 b 是 NaN，则返回 b
    return b;
  } else {
    # 如果都不是 NaN，则返回绝对值较大的数，如果相等则返回 b
    return std::abs(a) > std::abs(b) ? a : b;
  }
// 结束 c10 命名空间
}

// 对于非复数类型，实现一个求最小值的函数模板
template <typename TYPE, std::enable_if_t<!c10::is_complex<TYPE>::value, int> = 0>
inline TYPE min_impl (TYPE a, TYPE b) {
  // 如果 a 或者 b 是 NaN，则返回 quiet NaN
  if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
    return std::numeric_limits<TYPE>::quiet_NaN();
  } else {
    // 否则返回 a 和 b 中的较小值
    return std::min(a, b);
  }
}

// 对于复数类型，实现一个求最小值的函数模板
template <typename TYPE, std::enable_if_t<c10::is_complex<TYPE>::value, int> = 0>
inline TYPE min_impl (TYPE a, TYPE b) {
  // 如果 a 是 NaN，则返回 a
  if (_isnan<TYPE>(a)) {
    return a;
  } else if (_isnan<TYPE>(b)) {
    // 如果 b 是 NaN，则返回 b
    return b;
  } else {
    // 否则比较 a 和 b 的绝对值，返回较小的那个
    return std::abs(a) < std::abs(b) ? a : b;
  }
}

// 结束 at::native 命名空间
}} //end at::native
```