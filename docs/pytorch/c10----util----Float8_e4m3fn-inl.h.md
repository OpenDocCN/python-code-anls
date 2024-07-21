# `.\pytorch\c10\util\Float8_e4m3fn-inl.h`

```py
#pragma once

#include <c10/macros/Macros.h> // 引入C10库中的宏定义

#include <cstdint> // 引入标准整数类型的头文件
#include <limits> // 引入数值极限的头文件

C10_CLANG_DIAGNOSTIC_PUSH() // 开始忽略特定编译器诊断信息
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion") // 忽略隐式整数到浮点数转换的警告
#endif

namespace c10 {

/// Constructors
// Float8_e4m3fn类的构造函数，将浮点数值转换为fp8e4m3fn格式
inline C10_HOST_DEVICE Float8_e4m3fn::Float8_e4m3fn(float value)
    : x(detail::fp8e4m3fn_from_fp32_value(value)) {}

/// Implicit conversions
// 将Float8_e4m3fn类型对象转换为float类型
inline C10_HOST_DEVICE Float8_e4m3fn::operator float() const {
  return detail::fp8e4m3fn_to_fp32_value(x); // 使用fp8e4m3fn_to_fp32_value函数进行转换
}

/// Special values helper
// 检查Float8_e4m3fn对象是否表示NaN（非数值）
inline C10_HOST_DEVICE bool Float8_e4m3fn::isnan() const {
  return (x & 0b01111111) == 0b01111111; // 检查fp8e4m3fn格式中的指数字段是否为0b01111111
}

/// Arithmetic
// Float8_e4m3fn类型对象的加法运算
inline C10_HOST_DEVICE Float8_e4m3fn
operator+(const Float8_e4m3fn& a, const Float8_e4m3fn& b) {
  return static_cast<float>(a) + static_cast<float>(b); // 执行浮点数加法运算
}

// Float8_e4m3fn类型对象的减法运算
inline C10_HOST_DEVICE Float8_e4m3fn
operator-(const Float8_e4m3fn& a, const Float8_e4m3fn& b) {
  return static_cast<float>(a) - static_cast<float>(b); // 执行浮点数减法运算
}

// Float8_e4m3fn类型对象的乘法运算
inline C10_HOST_DEVICE Float8_e4m3fn
operator*(const Float8_e4m3fn& a, const Float8_e4m3fn& b) {
  return static_cast<float>(a) * static_cast<float>(b); // 执行浮点数乘法运算
}

// Float8_e4m3fn类型对象的除法运算
inline C10_HOST_DEVICE Float8_e4m3fn operator/(
    const Float8_e4m3fn& a,
    const Float8_e4m3fn& b) __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b); // 执行浮点数除法运算，忽略除以零的情况
}

// Float8_e4m3fn类型对象的一元负运算
inline C10_HOST_DEVICE Float8_e4m3fn operator-(const Float8_e4m3fn& a) {
  return -static_cast<float>(a); // 执行浮点数的一元负运算
}

// Float8_e4m3fn类型对象的加法赋值运算
inline C10_HOST_DEVICE Float8_e4m3fn& operator+=(
    Float8_e4m3fn& a,
    const Float8_e4m3fn& b) {
  a = a + b; // 执行浮点数加法赋值运算
  return a;
}

// Float8_e4m3fn类型对象的减法赋值运算
inline C10_HOST_DEVICE Float8_e4m3fn& operator-=(
    Float8_e4m3fn& a,
    const Float8_e4m3fn& b) {
  a = a - b; // 执行浮点数减法赋值运算
  return a;
}

// Float8_e4m3fn类型对象的乘法赋值运算
inline C10_HOST_DEVICE Float8_e4m3fn& operator*=(
    Float8_e4m3fn& a,
    const Float8_e4m3fn& b) {
  a = a * b; // 执行浮点数乘法赋值运算
  return a;
}

// Float8_e4m3fn类型对象的除法赋值运算
inline C10_HOST_DEVICE Float8_e4m3fn& operator/=(
    Float8_e4m3fn& a,
    const Float8_e4m3fn& b) {
  a = a / b; // 执行浮点数除法赋值运算
  return a;
}

/// Arithmetic with floats

// Float8_e4m3fn对象与浮点数的加法运算
inline C10_HOST_DEVICE float operator+(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) + b; // 执行浮点数与浮点数的加法运算
}

// Float8_e4m3fn对象与浮点数的减法运算
inline C10_HOST_DEVICE float operator-(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) - b; // 执行浮点数与浮点数的减法运算
}

// Float8_e4m3fn对象与浮点数的乘法运算
inline C10_HOST_DEVICE float operator*(Float8_e4m3fn a, float b) {
  return static_cast<float>(a) * b; // 执行浮点数与浮点数的乘法运算
}

// Float8_e4m3fn对象与浮点数的除法运算
inline C10_HOST_DEVICE float operator/(Float8_e4m3fn a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b; // 执行浮点数与浮点数的除法运算，忽略除以零的情况
}

// 浮点数与Float8_e4m3fn对象的加法运算
inline C10_HOST_DEVICE float operator+(float a, Float8_e4m3fn b) {
  return a + static_cast<float>(b); // 执行浮点数与浮点数的加法运算
}

// 浮点数与Float8_e4m3fn对象的减法运算
inline C10_HOST_DEVICE float operator-(float a, Float8_e4m3fn b) {
  return a - static_cast<float>(b); // 执行浮点数与浮点数的减法运算
}

// 浮点数与Float8_e4m3fn对象的乘法运算
inline C10_HOST_DEVICE float operator*(float a, Float8_e4m3fn b) {
  return a * static_cast<float>(b); // 执行浮点数与浮点数的乘法运算
}

// 浮点数与Float8_e4m3fn对象的除法运算
inline C10_HOST_DEVICE float operator/(float a, Float8_e4m3fn b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b); // 执行浮点数与浮点数的除法运算，忽略除以零的情况
}

} // namespace c10
/// 定义了浮点数和 Float8_e4m3fn 类型之间的加法操作符重载函数
inline C10_HOST_DEVICE float& operator+=(float& a, const Float8_e4m3fn& b) {
  return a += static_cast<float>(b);
}

/// 定义了浮点数和 Float8_e4m3fn 类型之间的减法操作符重载函数
inline C10_HOST_DEVICE float& operator-=(float& a, const Float8_e4m3fn& b) {
  return a -= static_cast<float>(b);
}

/// 定义了浮点数和 Float8_e4m3fn 类型之间的乘法操作符重载函数
inline C10_HOST_DEVICE float& operator*=(float& a, const Float8_e4m3fn& b) {
  return a *= static_cast<float>(b);
}

/// 定义了浮点数和 Float8_e4m3fn 类型之间的除法操作符重载函数
inline C10_HOST_DEVICE float& operator/=(float& a, const Float8_e4m3fn& b) {
  return a /= static_cast<float>(b);
}

/// 定义了 Float8_e4m3fn 类型和 double 类型之间的加法操作符重载函数
inline C10_HOST_DEVICE double operator+(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) + b;
}

/// 定义了 Float8_e4m3fn 类型和 double 类型之间的减法操作符重载函数
inline C10_HOST_DEVICE double operator-(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) - b;
}

/// 定义了 Float8_e4m3fn 类型和 double 类型之间的乘法操作符重载函数
inline C10_HOST_DEVICE double operator*(Float8_e4m3fn a, double b) {
  return static_cast<double>(a) * b;
}

/// 定义了 Float8_e4m3fn 类型和 double 类型之间的除法操作符重载函数，忽略除以零的浮点异常
inline C10_HOST_DEVICE double operator/(Float8_e4m3fn a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

/// 定义了 double 类型和 Float8_e4m3fn 类型之间的加法操作符重载函数
inline C10_HOST_DEVICE double operator+(double a, Float8_e4m3fn b) {
  return a + static_cast<double>(b);
}

/// 定义了 double 类型和 Float8_e4m3fn 类型之间的减法操作符重载函数
inline C10_HOST_DEVICE double operator-(double a, Float8_e4m3fn b) {
  return a - static_cast<double>(b);
}

/// 定义了 double 类型和 Float8_e4m3fn 类型之间的乘法操作符重载函数
inline C10_HOST_DEVICE double operator*(double a, Float8_e4m3fn b) {
  return a * static_cast<double>(b);
}

/// 定义了 double 类型和 Float8_e4m3fn 类型之间的除法操作符重载函数，忽略除以零的浮点异常
inline C10_HOST_DEVICE double operator/(double a, Float8_e4m3fn b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// 定义了 Float8_e4m3fn 类型和 int 类型之间的加法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator+(Float8_e4m3fn a, int b) {
  return a + static_cast<Float8_e4m3fn>(b);
}

/// 定义了 Float8_e4m3fn 类型和 int 类型之间的减法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator-(Float8_e4m3fn a, int b) {
  return a - static_cast<Float8_e4m3fn>(b);
}

/// 定义了 Float8_e4m3fn 类型和 int 类型之间的乘法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator*(Float8_e4m3fn a, int b) {
  return a * static_cast<Float8_e4m3fn>(b);
}

/// 定义了 Float8_e4m3fn 类型和 int 类型之间的除法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator/(Float8_e4m3fn a, int b) {
  return a / static_cast<Float8_e4m3fn>(b);
}

/// 定义了 int 类型和 Float8_e4m3fn 类型之间的加法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator+(int a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) + b;
}

/// 定义了 int 类型和 Float8_e4m3fn 类型之间的减法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator-(int a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) - b;
}

/// 定义了 int 类型和 Float8_e4m3fn 类型之间的乘法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator*(int a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) * b;
}

/// 定义了 int 类型和 Float8_e4m3fn 类型之间的除法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator/(int a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) / b;
}

/// 定义了 Float8_e4m3fn 类型和 int64_t 类型之间的加法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator+(Float8_e4m3fn a, int64_t b) {
  return a + static_cast<Float8_e4m3fn>(b);
}

/// 定义了 Float8_e4m3fn 类型和 int64_t 类型之间的减法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator-(Float8_e4m3fn a, int64_t b) {
  return a - static_cast<Float8_e4m3fn>(b);
}

/// 定义了 Float8_e4m3fn 类型和 int64_t 类型之间的乘法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator*(Float8_e4m3fn a, int64_t b) {
  return a * static_cast<Float8_e4m3fn>(b);
}

/// 定义了 Float8_e4m3fn 类型和 int64_t 类型之间的除法操作符重载函数
inline C10_HOST_DEVICE Float8_e4m3fn operator/(Float8_e4m3fn a, int64_t b) {
  return a / static_cast<Float8_e4m3fn>(b);
}
// 定义运算符重载，允许整数与 c10::Float8_e4m3fn 类型的对象进行加法运算
inline C10_HOST_DEVICE Float8_e4m3fn operator+(int64_t a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) + b;
}

// 定义运算符重载，允许整数与 c10::Float8_e4m3fn 类型的对象进行减法运算
inline C10_HOST_DEVICE Float8_e4m3fn operator-(int64_t a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) - b;
}

// 定义运算符重载，允许整数与 c10::Float8_e4m3fn 类型的对象进行乘法运算
inline C10_HOST_DEVICE Float8_e4m3fn operator*(int64_t a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) * b;
}

// 定义运算符重载，允许整数与 c10::Float8_e4m3fn 类型的对象进行除法运算
inline C10_HOST_DEVICE Float8_e4m3fn operator/(int64_t a, Float8_e4m3fn b) {
  return static_cast<Float8_e4m3fn>(a) / b;
}

/// 注意：我们没有直接定义比较运算符，而是依赖于从 c10::Float8_e4m3fn 到 float 的隐式转换。

} // namespace c10

namespace std {

// 特化 numeric_limits 模板，为 c10::Float8_e4m3fn 类型定义数值特性
template <>
class numeric_limits<c10::Float8_e4m3fn> {
 public:
  static constexpr bool is_specialized = true;  // 表示特化了数值特性
  static constexpr bool is_signed = true;       // 表示有符号数
  static constexpr bool is_integer = false;      // 表示不是整数
  static constexpr bool is_exact = false;        // 表示不是精确数
  static constexpr bool has_infinity = false;    // 表示不包含无穷大
  static constexpr bool has_quiet_NaN = true;    // 表示包含静默 NaN
  static constexpr bool has_signaling_NaN = false;  // 表示不包含信号 NaN
  static constexpr auto has_denorm = true;       // 表示包含非规格化数
  static constexpr auto has_denorm_loss = true;  // 表示包含非规格化数损失
  static constexpr auto round_style = numeric_limits<float>::round_style;  // 采用与 float 相同的舍入方式
  static constexpr bool is_iec559 = false;       // 表示不符合 IEEE 754 标准
  static constexpr bool is_bounded = true;       // 表示有界
  static constexpr bool is_modulo = false;       // 表示不是模运算
  static constexpr int digits = 4;               // 数字位数
  static constexpr int digits10 = 0;             // 十进制数字位数
  static constexpr int max_digits10 = 3;         // 最大十进制数字位数
  static constexpr int radix = 2;                // 基数为 2
  static constexpr int min_exponent = -5;        // 最小指数为 -5
  static constexpr int min_exponent10 = -1;      // 最小十进制指数为 -1
  static constexpr int max_exponent = 8;         // 最大指数为 8
  static constexpr int max_exponent10 = 2;       // 最大十进制指数为 2
  static constexpr auto traps = numeric_limits<float>::traps;  // 与 float 相同的陷阱设置
  static constexpr auto tinyness_before = false;  // 在截尾之前不进行处理

  // 返回 c10::Float8_e4m3fn 类型的最小值
  static constexpr c10::Float8_e4m3fn min() {
    return c10::Float8_e4m3fn(0x08, c10::Float8_e4m3fn::from_bits());
  }

  // 返回 c10::Float8_e4m3fn 类型的最小有限值
  static constexpr c10::Float8_e4m3fn lowest() {
    return c10::Float8_e4m3fn(0xFE, c10::Float8_e4m3fn::from_bits());
  }

  // 返回 c10::Float8_e4m3fn 类型的最大有限值
  static constexpr c10::Float8_e4m3fn max() {
    return c10::Float8_e4m3fn(0x7E, c10::Float8_e4m3fn::from_bits());
  }

  // 返回 c10::Float8_e4m3fn 类型的 epsilon 值
  static constexpr c10::Float8_e4m3fn epsilon() {
    return c10::Float8_e4m3fn(0x20, c10::Float8_e4m3fn::from_bits());
  }

  // 返回 c10::Float8_e4m3fn 类型的舍入误差
  static constexpr c10::Float8_e4m3fn round_error() {
    return c10::Float8_e4m3fn(0x30, c10::Float8_e4m3fn::from_bits());
  }

  // 返回 c10::Float8_e4m3fn 类型的静默 NaN 值
  static constexpr c10::Float8_e4m3fn quiet_NaN() {
    return c10::Float8_e4m3fn(0x7F, c10::Float8_e4m3fn::from_bits());
  }

  // 返回 c10::Float8_e4m3fn 类型的最小非规格化数
  static constexpr c10::Float8_e4m3fn denorm_min() {
    return c10::Float8_e4m3fn(0x01, c10::Float8_e4m3fn::from_bits());
  }
};

} // namespace std

// 恢复之前的 Clang 诊断设置
C10_CLANG_DIAGNOSTIC_POP()
```