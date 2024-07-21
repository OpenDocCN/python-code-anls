# `.\pytorch\c10\util\BFloat16-math.h`

```
#pragma once
// 使用预处理指令#pragma once确保头文件只被编译一次

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
// 引入C++头文件<c10/util/BFloat16.h>和<c10/util/Half.h>

C10_CLANG_DIAGNOSTIC_PUSH()
// 使用C10_CLANG_DIAGNOSTIC_PUSH()将Clang的诊断信息压栈，保留当前的诊断设置

#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
// 如果Clang支持-Wimplicit-float-conversion警告
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
// 则忽略-Wimplicit-float-conversion警告
#endif

namespace std {

template <typename T>
struct is_reduced_floating_point
    : std::integral_constant<
          bool,
          std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16>> {};
// 定义模板结构体is_reduced_floating_point，用于判断T是否为c10::Half或c10::BFloat16类型的缩减浮点数

template <typename T>
constexpr bool is_reduced_floating_point_v =
    is_reduced_floating_point<T>::value;
// 定义模板常量is_reduced_floating_point_v，用于获取is_reduced_floating_point<T>::value的值

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T acos(T a) {
  return std::acos(float(a));
}
// 如果T是缩减浮点数类型，则定义acos函数，将输入参数a转换为float类型后调用std::acos

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T asin(T a) {
  return std::asin(float(a));
}
// 如果T是缩减浮点数类型，则定义asin函数，将输入参数a转换为float类型后调用std::asin

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T atan(T a) {
  return std::atan(float(a));
}
// 如果T是缩减浮点数类型，则定义atan函数，将输入参数a转换为float类型后调用std::atan

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T atanh(T a) {
  return std::atanh(float(a));
}
// 如果T是缩减浮点数类型，则定义atanh函数，将输入参数a转换为float类型后调用std::atanh

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T erf(T a) {
  return std::erf(float(a));
}
// 如果T是缩减浮点数类型，则定义erf函数，将输入参数a转换为float类型后调用std::erf

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T erfc(T a) {
  return std::erfc(float(a));
}
// 如果T是缩减浮点数类型，则定义erfc函数，将输入参数a转换为float类型后调用std::erfc

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T exp(T a) {
  return std::exp(float(a));
}
// 如果T是缩减浮点数类型，则定义exp函数，将输入参数a转换为float类型后调用std::exp

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T expm1(T a) {
  return std::expm1(float(a));
}
// 如果T是缩减浮点数类型，则定义expm1函数，将输入参数a转换为float类型后调用std::expm1

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T log(T a) {
  return std::log(float(a));
}
// 如果T是缩减浮点数类型，则定义log函数，将输入参数a转换为float类型后调用std::log

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T log10(T a) {
  return std::log10(float(a));
}
// 如果T是缩减浮点数类型，则定义log10函数，将输入参数a转换为float类型后调用std::log10

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T log1p(T a) {
  return std::log1p(float(a));
}
// 如果T是缩减浮点数类型，则定义log1p函数，将输入参数a转换为float类型后调用std::log1p

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T log2(T a) {
  return std::log2(float(a));
}
// 如果T是缩减浮点数类型，则定义log2函数，将输入参数a转换为float类型后调用std::log2

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T ceil(T a) {
  return std::ceil(float(a));
}
// 如果T是缩减浮点数类型，则定义ceil函数，将输入参数a转换为float类型后调用std::ceil

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T cos(T a) {
  return std::cos(float(a));
}
// 如果T是缩减浮点数类型，则定义cos函数，将输入参数a转换为float类型后调用std::cos

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T floor(T a) {
  return std::floor(float(a));
}
// 如果T是缩减浮点数类型，则定义floor函数，将输入参数a转换为float类型后调用std::floor

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T nearbyint(T a) {
  return std::nearbyint(float(a));
}
// 如果T是缩减浮点数类型，则定义nearbyint函数，将输入参数a转换为float类型后调用std::nearbyint

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
// 如果T是缩减浮点数类型，则继续定义其它函数...
    typename T,
    // 定义模板参数 T，这里使用了 typename 语法，表示 T 是一个类型参数
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
    // 使用 std::enable_if_t 实现模板参数的条件约束：
    // 如果类型 T 满足 is_reduced_floating_point_v<T> 的条件，那么这个模板参数被设为 int 类型，默认值为 0；
    // 否则，这个模板参数将无效，不符合条件的 T 类型将导致编译器在编译时忽略这个模板。
// 返回浮点数类型变量 a 的正弦值，使用 std::sin 函数计算
inline T sin(T a) {
    return std::sin(float(a));
}

// 返回浮点数类型变量 a 的正切值，仅在 T 类型为缩减浮点数时启用该函数，使用 std::tan 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T tan(T a) {
    return std::tan(float(a));
}

// 返回浮点数类型变量 a 的双曲正弦值，仅在 T 类型为缩减浮点数时启用该函数，使用 std::sinh 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T sinh(T a) {
    return std::sinh(float(a));
}

// 返回浮点数类型变量 a 的双曲余弦值，仅在 T 类型为缩减浮点数时启用该函数，使用 std::cosh 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T cosh(T a) {
    return std::cosh(float(a));
}

// 返回浮点数类型变量 a 的双曲正切值，仅在 T 类型为缩减浮点数时启用该函数，使用 std::tanh 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T tanh(T a) {
    return std::tanh(float(a));
}

// 返回浮点数类型变量 a 的截断值，仅在 T 类型为缩减浮点数时启用该函数，使用 std::trunc 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T trunc(T a) {
    return std::trunc(float(a));
}

// 返回浮点数类型变量 a 的对数伽玛函数值，仅在 T 类型为缩减浮点数时启用该函数，使用 std::lgamma 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T lgamma(T a) {
    return std::lgamma(float(a));
}

// 返回浮点数类型变量 a 的平方根，仅在 T 类型为缩减浮点数时启用该函数，使用 std::sqrt 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T sqrt(T a) {
    return std::sqrt(float(a));
}

// 返回浮点数类型变量 a 的倒数平方根，仅在 T 类型为缩减浮点数时启用该函数，通过 1.0 / std::sqrt(float(a)) 计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T rsqrt(T a) {
    return 1.0 / std::sqrt(float(a));
}

// 返回浮点数类型变量 a 的绝对值，仅在 T 类型为缩减浮点数时启用该函数，使用 std::abs 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T abs(T a) {
    return std::abs(float(a));
}

// 返回浮点数类型变量 a 的 b 次方，仅在 T 类型为缩减浮点数时启用该函数，使用 std::pow 函数计算
#if defined(_MSC_VER) && defined(__CUDACC__)
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T pow(T a, double b) {
    return std::pow(float(a), float(b));
}
#else
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T pow(T a, double b) {
    return std::pow(float(a), b);
}
#endif

// 返回浮点数类型变量 a 的 b 次方，仅在 T 类型为缩减浮点数时启用该函数，使用 std::pow 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T pow(T a, T b) {
    return std::pow(float(a), float(b));
}

// 返回浮点数类型变量 a 除以浮点数类型变量 b 的余数，仅在 T 类型为缩减浮点数时启用该函数，使用 std::fmod 函数计算
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T fmod(T a, T b) {
    return std::fmod(float(a), float(b));
}
/*
  The following function is inspired from the implementation in `musl`
  Link to License: https://git.musl-libc.org/cgit/musl/tree/COPYRIGHT
  ----------------------------------------------------------------------
  Copyright © 2005-2020 Rich Felker, et al.

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  ----------------------------------------------------------------------
 */
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
C10_HOST_DEVICE inline T nextafter(T from, T to) {
  // Reference:
  // https://git.musl-libc.org/cgit/musl/tree/src/math/nextafter.c
  // Define an alias for the representation of T as an unsigned 16-bit integer
  using int_repr_t = uint16_t;
  // Alias T as float_t
  using float_t = T;
  // Constant for the number of bits in int_repr_t
  constexpr uint8_t bits = 16;
  // Define a union to allow type punning between T and int_repr_t
  union {
    float_t f;    // Access as float
    int_repr_t i; // Access as int_repr_t
  } ufrom = {from}, uto = {to};

  // Mask to isolate the sign bit (most significant bit) in int_repr_t
  int_repr_t sign_mask = int_repr_t{1} << (bits - 1);

  // Short-circuit: if either from or to is NaN, return NaN
  if (from != from || to != to) {
    return from + to;
  }

  // Short-circuit: if from and to are exactly the same, return from
  if (ufrom.i == uto.i) {
    return from;
  }

  // Mask out the sign bit to treat values as positive (equivalent to abs(x))
  int_repr_t abs_from = ufrom.i & ~sign_mask;
  int_repr_t abs_to = uto.i & ~sign_mask;

  // If from is effectively zero
  if (abs_from == 0) {
    // If both from and to are zero with different signs, preserve the sign of to
    if (abs_to == 0) {
      return to;
    }
    // Return the smallest positive subnormal number with the sign of to
    ufrom.i = (uto.i & sign_mask) | int_repr_t{1};
    return ufrom.f;
  }

  // If abs(from) > abs(to) or if from and to have different signs
  if (abs_from > abs_to || ((ufrom.i ^ uto.i) & sign_mask)) {
    ufrom.i--;  // Decrease the representation of from
  } else {
    ufrom.i++;  // Increase the representation of from
  }

  return ufrom.f;  // Return the resulting float value
}

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
```