# `.\pytorch\c10\util\Float8_e5m2-inl.h`

```
#pragma once

#include <c10/macros/Macros.h>  // 包含 c10 宏定义
#include <cstring>  // 包含字符串处理函数定义
#include <limits>  // 包含数值极限定义

C10_CLANG_DIAGNOSTIC_PUSH()  // 开启 Clang 编译器诊断信息保存
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")  // 忽略隐式整数转换为浮点数的警告
#endif

#define EXP_WIDTH_FP8 5  // 定义 FP8 格式的指数部分位宽为 5
#define MAN_WIDTH_FP8 2  // 定义 FP8 格式的尾数部分位宽为 2
#define EXP_BIAS_FP8 15  // 定义 FP8 格式的偏置值为 15

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE Float8_e5m2::Float8_e5m2(float value)
    : x(detail::fp8e5m2_from_fp32_value(value)) {}  // 使用给定的浮点数值初始化 FP8 格式对象

/// Implicit conversions

inline C10_HOST_DEVICE Float8_e5m2::operator float() const {
  return detail::fp8e5m2_to_fp32_value(x);  // 将 FP8 格式对象转换为浮点数值
}

/// Special values helpers

inline C10_HOST_DEVICE bool Float8_e5m2::isnan() const {
  return (x & 0b01111111) > 0b01111100;  // 判断 FP8 格式对象是否表示 NaN
}

inline C10_HOST_DEVICE bool Float8_e5m2::isinf() const {
  return (x & 0b01111111) == 0b01111100;  // 判断 FP8 格式对象是否表示无穷大
}

/// Arithmetic

inline C10_HOST_DEVICE Float8_e5m2
operator+(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) + static_cast<float>(b);  // FP8 格式对象的加法运算
}

inline C10_HOST_DEVICE Float8_e5m2
operator-(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) - static_cast<float>(b);  // FP8 格式对象的减法运算
}

inline C10_HOST_DEVICE Float8_e5m2
operator*(const Float8_e5m2& a, const Float8_e5m2& b) {
  return static_cast<float>(a) * static_cast<float>(b);  // FP8 格式对象的乘法运算
}

inline C10_HOST_DEVICE Float8_e5m2 operator/(
    const Float8_e5m2& a,
    const Float8_e5m2& b) __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);  // FP8 格式对象的除法运算，忽略除零错误
}

inline C10_HOST_DEVICE Float8_e5m2 operator-(const Float8_e5m2& a) {
  return -static_cast<float>(a);  // FP8 格式对象的取负运算
}

inline C10_HOST_DEVICE Float8_e5m2& operator+=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a + b;  // FP8 格式对象的自增运算
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2& operator-=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a - b;  // FP8 格式对象的自减运算
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2& operator*=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a * b;  // FP8 格式对象的自乘运算
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2& operator/=(
    Float8_e5m2& a,
    const Float8_e5m2& b) {
  a = a / b;  // FP8 格式对象的自除运算
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(Float8_e5m2 a, float b) {
  return static_cast<float>(a) + b;  // FP8 格式对象与浮点数的加法运算
}
inline C10_HOST_DEVICE float operator-(Float8_e5m2 a, float b) {
  return static_cast<float>(a) - b;  // FP8 格式对象与浮点数的减法运算
}
inline C10_HOST_DEVICE float operator*(Float8_e5m2 a, float b) {
  return static_cast<float>(a) * b;  // FP8 格式对象与浮点数的乘法运算
}
inline C10_HOST_DEVICE float operator/(Float8_e5m2 a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;  // FP8 格式对象与浮点数的除法运算，忽略除零错误
}

inline C10_HOST_DEVICE float operator+(float a, Float8_e5m2 b) {
  return a + static_cast<float>(b);  // 浮点数与 FP8 格式对象的加法运算
}
inline C10_HOST_DEVICE float operator-(float a, Float8_e5m2 b) {
  return a - static_cast<float>(b);  // 浮点数与 FP8 格式对象的减法运算
}
inline C10_HOST_DEVICE float operator*(float a, Float8_e5m2 b) {
  return a * static_cast<float>(b);  // 浮点数与 FP8 格式对象的乘法运算
}
inline C10_HOST_DEVICE float operator/(float a, Float8_e5m2 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);  // 浮点数与 FP8 格式对象的除法运算，忽略除零错误
}
    # 忽略 UBsan（Undefined Behavior Sanitizer）对浮点除零的检测
    __ubsan_ignore_float_divide_by_zero__ {
        # 执行浮点数除法操作，返回 a 除以 b 的结果（b 被转换为 float 类型）
        return a / static_cast<float>(b);
    }
/// 定义一系列操作符重载函数，用于 Float8_e5m2 类型和其他数据类型之间的算术运算

inline C10_HOST_DEVICE float& operator+=(float& a, const Float8_e5m2& b) {
  return a += static_cast<float>(b);
}
// 加法赋值运算符重载，将 float 类型和 Float8_e5m2 类型相加

inline C10_HOST_DEVICE float& operator-=(float& a, const Float8_e5m2& b) {
  return a -= static_cast<float>(b);
}
// 减法赋值运算符重载，将 float 类型和 Float8_e5m2 类型相减

inline C10_HOST_DEVICE float& operator*=(float& a, const Float8_e5m2& b) {
  return a *= static_cast<float>(b);
}
// 乘法赋值运算符重载，将 float 类型和 Float8_e5m2 类型相乘

inline C10_HOST_DEVICE float& operator/=(float& a, const Float8_e5m2& b) {
  return a /= static_cast<float>(b);
}
// 除法赋值运算符重载，将 float 类型和 Float8_e5m2 类型相除

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(Float8_e5m2 a, double b) {
  return static_cast<double>(a) + b;
}
// 浮点数加法运算符重载，将 Float8_e5m2 类型和 double 类型相加

inline C10_HOST_DEVICE double operator-(Float8_e5m2 a, double b) {
  return static_cast<double>(a) - b;
}
// 浮点数减法运算符重载，将 Float8_e5m2 类型和 double 类型相减

inline C10_HOST_DEVICE double operator*(Float8_e5m2 a, double b) {
  return static_cast<double>(a) * b;
}
// 浮点数乘法运算符重载，将 Float8_e5m2 类型和 double 类型相乘

inline C10_HOST_DEVICE double operator/(Float8_e5m2 a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}
// 浮点数除法运算符重载，将 Float8_e5m2 类型和 double 类型相除，忽略除零错误

inline C10_HOST_DEVICE double operator+(double a, Float8_e5m2 b) {
  return a + static_cast<double>(b);
}
// 浮点数加法运算符重载，将 double 类型和 Float8_e5m2 类型相加

inline C10_HOST_DEVICE double operator-(double a, Float8_e5m2 b) {
  return a - static_cast<double>(b);
}
// 浮点数减法运算符重载，将 double 类型和 Float8_e5m2 类型相减

inline C10_HOST_DEVICE double operator*(double a, Float8_e5m2 b) {
  return a * static_cast<double>(b);
}
// 浮点数乘法运算符重载，将 double 类型和 Float8_e5m2 类型相乘

inline C10_HOST_DEVICE double operator/(double a, Float8_e5m2 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}
// 浮点数除法运算符重载，将 double 类型和 Float8_e5m2 类型相除，忽略除零错误

/// Arithmetic with ints

inline C10_HOST_DEVICE Float8_e5m2 operator+(Float8_e5m2 a, int b) {
  return a + static_cast<Float8_e5m2>(b);
}
// 整数加法运算符重载，将 Float8_e5m2 类型和 int 类型相加

inline C10_HOST_DEVICE Float8_e5m2 operator-(Float8_e5m2 a, int b) {
  return a - static_cast<Float8_e5m2>(b);
}
// 整数减法运算符重载，将 Float8_e5m2 类型和 int 类型相减

inline C10_HOST_DEVICE Float8_e5m2 operator*(Float8_e5m2 a, int b) {
  return a * static_cast<Float8_e5m2>(b);
}
// 整数乘法运算符重载，将 Float8_e5m2 类型和 int 类型相乘

inline C10_HOST_DEVICE Float8_e5m2 operator/(Float8_e5m2 a, int b) {
  return a / static_cast<Float8_e5m2>(b);
}
// 整数除法运算符重载，将 Float8_e5m2 类型和 int 类型相除

inline C10_HOST_DEVICE Float8_e5m2 operator+(int a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) + b;
}
// 整数加法运算符重载，将 int 类型和 Float8_e5m2 类型相加

inline C10_HOST_DEVICE Float8_e5m2 operator-(int a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) - b;
}
// 整数减法运算符重载，将 int 类型和 Float8_e5m2 类型相减

inline C10_HOST_DEVICE Float8_e5m2 operator*(int a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) * b;
}
// 整数乘法运算符重载，将 int 类型和 Float8_e5m2 类型相乘

inline C10_HOST_DEVICE Float8_e5m2 operator/(int a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) / b;
}
// 整数除法运算符重载，将 int 类型和 Float8_e5m2 类型相除

//// Arithmetic with int64_t

inline C10_HOST_DEVICE Float8_e5m2 operator+(Float8_e5m2 a, int64_t b) {
  return a + static_cast<Float8_e5m2>(b);
}
// 64 位整数加法运算符重载，将 Float8_e5m2 类型和 int64_t 类型相加

inline C10_HOST_DEVICE Float8_e5m2 operator-(Float8_e5m2 a, int64_t b) {
  return a - static_cast<Float8_e5m2>(b);
}
// 64 位整数减法运算符重载，将 Float8_e5m2 类型和 int64_t 类型相减

inline C10_HOST_DEVICE Float8_e5m2 operator*(Float8_e5m2 a, int64_t b) {
  return a * static_cast<Float8_e5m2>(b);
}
// 64 位整数乘法运算符重载，将 Float8_e5m2 类型和 int64_t 类型相乘

inline C10_HOST_DEVICE Float8_e5m2 operator/(Float8_e5m2 a, int64_t b) {
  return a / static_cast<Float8_e5m2>(b);
}
// 64 位整数除法运算符重载，将 Float8_e5m2 类型和 int64_t 类型相除

inline C10_HOST_DEVICE Float8_e5m2 operator+(int64_t a, Float8_e5m2 b) {
  return static_cast<Float8_e5m2>(a) + b;
}
// 64 位整数加法运算符重载，将 int64_t 类型和 Float8_e5m2 类型相加
inline C10_HOST_DEVICE Float8_e5m2 operator-(int64_t a, Float8_e5m2 b) {
  // 定义整数与浮点数类 Float8_e5m2 的减法操作符重载
  return static_cast<Float8_e5m2>(a) - b;
}

inline C10_HOST_DEVICE Float8_e5m2 operator*(int64_t a, Float8_e5m2 b) {
  // 定义整数与浮点数类 Float8_e5m2 的乘法操作符重载
  return static_cast<Float8_e5m2>(a) * b;
}

inline C10_HOST_DEVICE Float8_e5m2 operator/(int64_t a, Float8_e5m2 b) {
  // 定义整数与浮点数类 Float8_e5m2 的除法操作符重载
  return static_cast<Float8_e5m2>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Float8_e5m2 to float.
/// 注意：我们不直接定义比较操作符，而是依赖于 c10::Float8_e5m2 到 float 的隐式转换。

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::Float8_e5m2> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = true;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 3;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 2;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr c10::Float8_e5m2 min() {
    // 返回最小的 Float8_e5m2 值，对应于 0x4 的位表示
    return c10::Float8_e5m2(0x4, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 max() {
    // 返回最大的 Float8_e5m2 值，对应于 0x7B 的位表示
    return c10::Float8_e5m2(0x7B, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 lowest() {
    // 返回最小的负 Float8_e5m2 值，对应于 0xFB 的位表示
    return c10::Float8_e5m2(0xFB, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 epsilon() {
    // 返回 Float8_e5m2 的机器精度值，对应于 0x34 的位表示
    return c10::Float8_e5m2(0x34, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 round_error() {
    // 返回 Float8_e5m2 的舍入误差，对应于 0x38 的位表示
    return c10::Float8_e5m2(0x38, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 infinity() {
    // 返回 Float8_e5m2 的正无穷大值，对应于 0x7C 的位表示
    return c10::Float8_e5m2(0x7C, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 quiet_NaN() {
    // 返回 Float8_e5m2 的静默 NaN 值，对应于 0x7F 的位表示
    return c10::Float8_e5m2(0x7F, c10::Float8_e5m2::from_bits());
  }
  static constexpr c10::Float8_e5m2 denorm_min() {
    // 返回 Float8_e5m2 的最小非规格化值，对应于 0x01 的位表示
    return c10::Float8_e5m2(0x01, c10::Float8_e5m2::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
```