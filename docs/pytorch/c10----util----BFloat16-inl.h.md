# `.\pytorch\c10\util\BFloat16-inl.h`

```
#pragma once

#include <c10/macros/Macros.h>  // 包含 c10 宏定义
#include <c10/util/bit_cast.h>  // 包含 bit_cast 工具函数

#include <limits>  // 包含标准库 limits

C10_CLANG_DIAGNOSTIC_PUSH()  // 开始忽略 Clang 编译器警告
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")  // 忽略隐式整数到浮点数转换的警告
#endif

#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
#if defined(CL_SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp>  // 包含 SYCL 1.2.1 版本的头文件
#else
#include <sycl/sycl.hpp>  // 包含 SYCL 2020 版本的头文件
#endif
#include <ext/oneapi/bfloat16.hpp>  // 包含 SYCL 的 bfloat16 类定义
#endif

namespace c10 {

/// Constructors
inline C10_HOST_DEVICE BFloat16::BFloat16(float value)
    :
#if defined(__CUDACC__) && !defined(USE_ROCM) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
      x(__bfloat16_as_ushort(__float2bfloat16(value)))  // CUDA 下使用特定函数进行 bfloat16 类型转换
#elif defined(__SYCL_DEVICE_ONLY__) && \
    defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
      x(c10::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(value)))  // SYCL 下使用 bit_cast 进行类型转换
#else
      // RNE by default
      x(detail::round_to_nearest_even(value))  // 默认使用最近偶数舍入模式进行转换
#endif
{
}

/// Implicit conversions
inline C10_HOST_DEVICE BFloat16::operator float() const {
#if defined(__CUDACC__) && !defined(USE_ROCM)
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));  // CUDA 下的 bfloat16 到 float 的转换
#elif defined(__SYCL_DEVICE_ONLY__) && \
    defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
  return float(*reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(&x));  // SYCL 下的 bfloat16 到 float 的转换
#else
  return detail::f32_from_bits(x);  // 使用位表示转换为 float
#endif
}

#if defined(__CUDACC__) && !defined(USE_ROCM)
inline C10_HOST_DEVICE BFloat16::BFloat16(const __nv_bfloat16& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);  // CUDA 下的 __nv_bfloat16 到 BFloat16 的转换
}
inline C10_HOST_DEVICE BFloat16::operator __nv_bfloat16() const {
  return *reinterpret_cast<const __nv_bfloat16*>(&x);  // BFloat16 到 CUDA 的 __nv_bfloat16 转换
}
#endif

#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
inline C10_HOST_DEVICE BFloat16::BFloat16(
    const sycl::ext::oneapi::bfloat16& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);  // SYCL 的 bfloat16 到 BFloat16 的转换
}
inline C10_HOST_DEVICE BFloat16::operator sycl::ext::oneapi::bfloat16() const {
  return *reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(&x);  // BFloat16 到 SYCL 的 bfloat16 转换
}
#endif

// CUDA intrinsics

#if defined(__CUDACC__) || defined(__HIPCC__)
inline C10_DEVICE BFloat16 __ldg(const BFloat16* ptr) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __ldg(reinterpret_cast<const __nv_bfloat16*>(ptr));  // CUDA 下的 __ldg 函数使用
#else
  return *ptr;  // 其他情况下的默认处理
#endif
}
#endif

/// Arithmetic

inline C10_HOST_DEVICE BFloat16
operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);  // BFloat16 的加法操作
}

inline C10_HOST_DEVICE BFloat16
operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);  // BFloat16 的减法操作
}

inline C10_HOST_DEVICE BFloat16
operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);  // BFloat16 的乘法操作
}

inline C10_HOST_DEVICE BFloat16 operator/(const BFloat16& a, const BFloat16& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);  // BFloat16 的除法操作，忽略除零错误
}
/// 一元减法操作符重载，返回给定BFloat16的负值
inline C10_HOST_DEVICE BFloat16 operator-(const BFloat16& a) {
  return -static_cast<float>(a);
}

/// 复合赋值加法操作符重载，将两个BFloat16相加，并将结果赋给左操作数
inline C10_HOST_DEVICE BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

/// 复合赋值减法操作符重载，将两个BFloat16相减，并将结果赋给左操作数
inline C10_HOST_DEVICE BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

/// 复合赋值乘法操作符重载，将两个BFloat16相乘，并将结果赋给左操作数
inline C10_HOST_DEVICE BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

/// 复合赋值除法操作符重载，将两个BFloat16相除，并将结果赋给左操作数
inline C10_HOST_DEVICE BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

/// 按位或操作符重载，将两个BFloat16按位或，并将结果赋给左操作数
inline C10_HOST_DEVICE BFloat16& operator|(BFloat16& a, const BFloat16& b) {
  a.x = a.x | b.x;
  return a;
}

/// 按位异或操作符重载，将两个BFloat16按位异或，并将结果赋给左操作数
inline C10_HOST_DEVICE BFloat16& operator^(BFloat16& a, const BFloat16& b) {
  a.x = a.x ^ b.x;
  return a;
}

/// 按位与操作符重载，将两个BFloat16按位与，并将结果赋给左操作数
inline C10_HOST_DEVICE BFloat16& operator&(BFloat16& a, const BFloat16& b) {
  a.x = a.x & b.x;
  return a;
}

/// BFloat16与float的加法操作符重载，将BFloat16转换为float，然后与给定的float相加
inline C10_HOST_DEVICE float operator+(BFloat16 a, float b) {
  return static_cast<float>(a) + b;
}

/// BFloat16与float的减法操作符重载，将BFloat16转换为float，然后与给定的float相减
inline C10_HOST_DEVICE float operator-(BFloat16 a, float b) {
  return static_cast<float>(a) - b;
}

/// BFloat16与float的乘法操作符重载，将BFloat16转换为float，然后与给定的float相乘
inline C10_HOST_DEVICE float operator*(BFloat16 a, float b) {
  return static_cast<float>(a) * b;
}

/// BFloat16与float的除法操作符重载，将BFloat16转换为float，然后与给定的float相除
inline C10_HOST_DEVICE float operator/(BFloat16 a, float b) {
  return static_cast<float>(a) / b;
}

/// float与BFloat16的加法操作符重载，将给定的float与BFloat16转换为float后相加
inline C10_HOST_DEVICE float operator+(float a, BFloat16 b) {
  return a + static_cast<float>(b);
}

/// float与BFloat16的减法操作符重载，将给定的float与BFloat16转换为float后相减
inline C10_HOST_DEVICE float operator-(float a, BFloat16 b) {
  return a - static_cast<float>(b);
}

/// float与BFloat16的乘法操作符重载，将给定的float与BFloat16转换为float后相乘
inline C10_HOST_DEVICE float operator*(float a, BFloat16 b) {
  return a * static_cast<float>(b);
}

/// float与BFloat16的除法操作符重载，将给定的float与BFloat16转换为float后相除
inline C10_HOST_DEVICE float operator/(float a, BFloat16 b) {
  return a / static_cast<float>(b);
}

/// float与BFloat16的复合赋值加法操作符重载，将给定的float与BFloat16转换为float后相加，并将结果赋给左操作数
inline C10_HOST_DEVICE float& operator+=(float& a, const BFloat16& b) {
  return a += static_cast<float>(b);
}

/// float与BFloat16的复合赋值减法操作符重载，将给定的float与BFloat16转换为float后相减，并将结果赋给左操作数
inline C10_HOST_DEVICE float& operator-=(float& a, const BFloat16& b) {
  return a -= static_cast<float>(b);
}

/// float与BFloat16的复合赋值乘法操作符重载，将给定的float与BFloat16转换为float后相乘，并将结果赋给左操作数
inline C10_HOST_DEVICE float& operator*=(float& a, const BFloat16& b) {
  return a *= static_cast<float>(b);
}

/// float与BFloat16的复合赋值除法操作符重载，将给定的float与BFloat16转换为float后相除，并将结果赋给左操作数
inline C10_HOST_DEVICE float& operator/=(float& a, const BFloat16& b) {
  return a /= static_cast<float>(b);
}

/// BFloat16与double的加法操作符重载，将BFloat16转换为double，然后与给定的double相加
inline C10_HOST_DEVICE double operator+(BFloat16 a, double b) {
  return static_cast<double>(a) + b;
}

/// BFloat16与double的减法操作符重载，将BFloat16转换为double，然后与给定的double相减
inline C10_HOST_DEVICE double operator-(BFloat16 a, double b) {
  return static_cast<double>(a) - b;
}

/// BFloat16与double的乘法操作符重载，将BFloat16转换为double，然后与给定的double相乘
inline C10_HOST_DEVICE double operator*(BFloat16 a, double b) {
  return static_cast<double>(a) * b;
}

/// BFloat16与double的除法操作符重载，将BFloat16转换为double，然后与给定的double相除
inline C10_HOST_DEVICE double operator/(BFloat16 a, double b) {
  return static_cast<double>(a) / b;
}

/// double与BFloat16的加法操作符重载，将给定的double与BFloat16转换为double后相加
inline C10_HOST_DEVICE double operator+(double a, BFloat16 b) {
  return a + static_cast<double>(b);
}

/// double与BFloat16的减法操作符重载，将给定的double与BFloat16转换为double后相减
inline C10_HOST_DEVICE double operator-(double a, BFloat16 b) {
  return a - static_cast<double>(b);
}

/// double与BFloat16的乘法操作符重载，将给定的double与BFloat16转换为double后相乘
inline C10_HOST_DEVICE double operator*(double a, BFloat16 b) {
  return a * static_cast<double>(b);
}

/// double与BFloat16的除法操作符重载，将给定的double与BFloat16转换为double后相除
inline C10_HOST_DEVICE double operator/(double a, BFloat16 b) {
  return a / static_cast<double>(b);
}
/// Arithmetic with ints

// 定义 BFloat16 类型与 int 类型的加法运算符重载
inline C10_HOST_DEVICE BFloat16 operator+(BFloat16 a, int b) {
  return a + static_cast<BFloat16>(b);  // 返回 BFloat16 类型对象与 int 类型相加的结果
}
// 定义 BFloat16 类型与 int 类型的减法运算符重载
inline C10_HOST_DEVICE BFloat16 operator-(BFloat16 a, int b) {
  return a - static_cast<BFloat16>(b);  // 返回 BFloat16 类型对象与 int 类型相减的结果
}
// 定义 BFloat16 类型与 int 类型的乘法运算符重载
inline C10_HOST_DEVICE BFloat16 operator*(BFloat16 a, int b) {
  return a * static_cast<BFloat16>(b);  // 返回 BFloat16 类型对象与 int 类型相乘的结果
}
// 定义 BFloat16 类型与 int 类型的除法运算符重载
inline C10_HOST_DEVICE BFloat16 operator/(BFloat16 a, int b) {
  return a / static_cast<BFloat16>(b);  // 返回 BFloat16 类型对象与 int 类型相除的结果
}

// 定义 int 类型与 BFloat16 类型的加法运算符重载
inline C10_HOST_DEVICE BFloat16 operator+(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) + b;  // 返回 int 类型与 BFloat16 类型对象相加的结果
}
// 定义 int 类型与 BFloat16 类型的减法运算符重载
inline C10_HOST_DEVICE BFloat16 operator-(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) - b;  // 返回 int 类型与 BFloat16 类型对象相减的结果
}
// 定义 int 类型与 BFloat16 类型的乘法运算符重载
inline C10_HOST_DEVICE BFloat16 operator*(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) * b;  // 返回 int 类型与 BFloat16 类型对象相乘的结果
}
// 定义 int 类型与 BFloat16 类型的除法运算符重载
inline C10_HOST_DEVICE BFloat16 operator/(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) / b;  // 返回 int 类型与 BFloat16 类型对象相除的结果
}

//// Arithmetic with int64_t

// 定义 BFloat16 类型与 int64_t 类型的加法运算符重载
inline C10_HOST_DEVICE BFloat16 operator+(BFloat16 a, int64_t b) {
  return a + static_cast<BFloat16>(b);  // 返回 BFloat16 类型对象与 int64_t 类型相加的结果
}
// 定义 BFloat16 类型与 int64_t 类型的减法运算符重载
inline C10_HOST_DEVICE BFloat16 operator-(BFloat16 a, int64_t b) {
  return a - static_cast<BFloat16>(b);  // 返回 BFloat16 类型对象与 int64_t 类型相减的结果
}
// 定义 BFloat16 类型与 int64_t 类型的乘法运算符重载
inline C10_HOST_DEVICE BFloat16 operator*(BFloat16 a, int64_t b) {
  return a * static_cast<BFloat16>(b);  // 返回 BFloat16 类型对象与 int64_t 类型相乘的结果
}
// 定义 BFloat16 类型与 int64_t 类型的除法运算符重载
inline C10_HOST_DEVICE BFloat16 operator/(BFloat16 a, int64_t b) {
  return a / static_cast<BFloat16>(b);  // 返回 BFloat16 类型对象与 int64_t 类型相除的结果
}

// 定义 int64_t 类型与 BFloat16 类型的加法运算符重载
inline C10_HOST_DEVICE BFloat16 operator+(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) + b;  // 返回 int64_t 类型与 BFloat16 类型对象相加的结果
}
// 定义 int64_t 类型与 BFloat16 类型的减法运算符重载
inline C10_HOST_DEVICE BFloat16 operator-(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) - b;  // 返回 int64_t 类型与 BFloat16 类型对象相减的结果
}
// 定义 int64_t 类型与 BFloat16 类型的乘法运算符重载
inline C10_HOST_DEVICE BFloat16 operator*(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) * b;  // 返回 int64_t 类型与 BFloat16 类型对象相乘的结果
}
// 定义 int64_t 类型与 BFloat16 类型的除法运算符重载
inline C10_HOST_DEVICE BFloat16 operator/(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) / b;  // 返回 int64_t 类型与 BFloat16 类型对象相除的结果
}

// Overloading < and > operators, because std::max and std::min use them.

// 定义 BFloat16 类型的大于（>)运算符重载
inline C10_HOST_DEVICE bool operator>(BFloat16& lhs, BFloat16& rhs) {
  return float(lhs) > float(rhs);  // 返回 BFloat16 类型对象之间的大小比较结果
}

// 定义 BFloat16 类型的小于（<）运算符重载
inline C10_HOST_DEVICE bool operator<(BFloat16& lhs, BFloat16& rhs) {
  return float(lhs) < float(rhs);  // 返回 BFloat16 类型对象之间的大小比较结果
}

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::BFloat16> {
 public:
  static constexpr bool is_signed = true;  // 是否为有符号类型
  static constexpr bool is_specialized = true;  // 是否特化
  static constexpr bool is_integer = false;  // 是否为整数类型
  static constexpr bool is_exact = false;  // 是否为精确类型
  static constexpr bool has_infinity = true;  // 是否支持无穷大
  static constexpr bool has_quiet_NaN = true;  // 是否支持静默 NaN
  static constexpr bool has_signaling_NaN = true;  // 是否支持信号 NaN
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;  // 是否支持非规格化数，参考 float 类型
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;  // 是否支持非规格化数损失，参考 float 类型
  static constexpr auto round_style = numeric_limits<float>::round_style;  // 取整方式，参考 float 类型
  static constexpr bool is_iec559 = false;  // 是否符合 IEC 559 标准
  static constexpr bool is_bounded = true;  // 是否有界
  static constexpr bool is_modulo = false;  // 是否为模数运算
  static constexpr int digits = 8;  // 数字的位数
  static constexpr int digits10 = 2;  // 十进制数字的位数
  static constexpr int max_digits10 = 4;  // 最大十进制数字的位数
  static constexpr int radix = 2;  // 基数
  static constexpr int min_exponent = -125;  // 最小指数
  static constexpr int min_exponent10 = -37;  // 最小十进制指数
  static constexpr int max_exponent = 128;  // 最大指数
  static constexpr int max_exponent10 = 38;  // 最大十进制指数
  static constexpr auto traps = numeric_limits<float>::traps;  // 是否捕捉异常，参考 float 类型
  static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;  // 在舍入之前是否处理微小值，参考 float 类型

  static constexpr c10::BFloat16 min() {  // 最小值
    return c10::BFloat16(0x0080, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 lowest() {  // 最低值
    return c10::BFloat16(0xFF7F, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 max() {  // 最大值
    return c10::BFloat16(0x7F7F, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 epsilon() {  // 机器精度
    return c10::BFloat16(0x3C00, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 round_error() {  // 舍入误差
    return c10::BFloat16(0x3F00, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 infinity() {  // 正无穷大
    return c10::BFloat16(0x7F80, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 quiet_NaN() {  // 静默 NaN
    return c10::BFloat16(0x7FC0, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 signaling_NaN() {  // 信号 NaN
    return c10::BFloat16(0x7F80, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 denorm_min() {  // 最小非规格化数
    return c10::BFloat16(0x0001, c10::BFloat16::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
```