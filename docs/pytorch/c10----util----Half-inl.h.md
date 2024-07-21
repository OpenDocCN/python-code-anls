# `.\pytorch\c10\util\Half-inl.h`

```
#pragma once

#include <c10/macros/Macros.h> // 包含宏定义
#include <c10/util/bit_cast.h> // 包含位转换工具函数

#include <cstring> // 包含字符串操作函数
#include <limits> // 包含数值极限

#ifdef __CUDACC__
#include <cuda_fp16.h> // CUDA半精度浮点数头文件
#endif

#ifdef __HIPCC__
#include <hip/hip_fp16.h> // HIP半精度浮点数头文件
#endif

#if defined(CL_SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp> // SYCL 1.2.1头文件
#elif defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp> // SYCL 2020头文件
#endif

#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
#include <ATen/cpu/vec/vec_half.h> // AVX2或AVX512半精度向量头文件
#endif

C10_CLANG_DIAGNOSTIC_PUSH() // 开启Clang编译器警告
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion") // 忽略隐式整数到浮点数转换警告
#endif

namespace c10 {

#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
/// Constructors
inline Half::Half(float16_t value) : x(detail::fp16_to_bits(value)) {} // 构造函数，将float16_t类型值转换为Half类型
inline Half::operator float16_t() const {
  return detail::fp16_from_bits(x); // 将Half类型值转换为float16_t类型
}
#else

inline C10_HOST_DEVICE Half::Half(float value)
    :
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      x(__half_as_short(__float2half(value))) // CUDA或HIP环境下的构造函数
#elif defined(__SYCL_DEVICE_ONLY__)
      x(c10::bit_cast<uint16_t>(sycl::half(value))) // SYCL环境下的构造函数
#elif (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
      x(at::vec::float2half_scalar(value)) // AVX2或AVX512环境下的构造函数
#else
      x(detail::fp16_ieee_from_fp32_value(value)) // 其他环境下的构造函数
#endif
{
}

/// Implicit conversions

inline C10_HOST_DEVICE Half::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __half2float(*reinterpret_cast<const __half*>(&x)); // CUDA或HIP环境下的类型转换
#elif defined(__SYCL_DEVICE_ONLY__)
  return float(c10::bit_cast<sycl::half>(x)); // SYCL环境下的类型转换
#elif (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
  return at::vec::half2float_scalar(x); // AVX2或AVX512环境下的类型转换
#elif defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
  return detail::native_fp16_to_fp32_value(x); // aarch64环境下的类型转换
#else
  return detail::fp16_ieee_to_fp32_value(x); // 其他环境下的类型转换
#endif
}

#endif /* !defined(__aarch64__) || defined(C10_MOBILE) || defined(__CUDACC__) \
        */

#if defined(__CUDACC__) || defined(__HIPCC__)
inline C10_HOST_DEVICE Half::Half(const __half& value) {
  x = *reinterpret_cast<const unsigned short*>(&value); // CUDA或HIP环境下的构造函数
}
inline C10_HOST_DEVICE Half::operator __half() const {
  return *reinterpret_cast<const __half*>(&x); // CUDA或HIP环境下的类型转换
}
#endif

#ifdef SYCL_LANGUAGE_VERSION
inline C10_HOST_DEVICE Half::Half(const sycl::half& value) {
  x = *reinterpret_cast<const unsigned short*>(&value); // SYCL环境下的构造函数
}
inline C10_HOST_DEVICE Half::operator sycl::half() const {
  return *reinterpret_cast<const sycl::half*>(&x); // SYCL环境下的类型转换
}
#endif

// CUDA intrinsics

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)) || \
    (defined(__clang__) && defined(__CUDA__))
inline __device__ Half __ldg(const Half* ptr) {
  return __ldg(reinterpret_cast<const __half*>(ptr)); // CUDA环境下的CUDA内置函数
}
#endif

/// Arithmetic
inline C10_HOST_DEVICE Half operator+(const Half& a, const Half& b) {
  // 返回两个半精度浮点数的和
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE Half operator-(const Half& a, const Half& b) {
  // 返回两个半精度浮点数的差
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE Half operator*(const Half& a, const Half& b) {
  // 返回两个半精度浮点数的积
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE Half operator/(const Half& a, const Half& b)
    __ubsan_ignore_float_divide_by_zero__ {
  // 返回两个半精度浮点数的商，忽略除零错误检查
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE Half operator-(const Half& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || \
    defined(__HIP_DEVICE_COMPILE__)
  // 如果在支持的 GPU 架构下，返回负半精度浮点数
  return __hneg(a);
#elif defined(__SYCL_DEVICE_ONLY__)
  // 如果在 SYCL 设备上，返回负的 SYCL 半精度浮点数
  return -c10::bit_cast<sycl::half>(a);
#else
  // 否则，返回负的单精度浮点数
  return -static_cast<float>(a);
#endif
}

inline C10_HOST_DEVICE Half& operator+=(Half& a, const Half& b) {
  // 将两个半精度浮点数相加，并将结果赋值给第一个操作数
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE Half& operator-=(Half& a, const Half& b) {
  // 将两个半精度浮点数相减，并将结果赋值给第一个操作数
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE Half& operator*=(Half& a, const Half& b) {
  // 将两个半精度浮点数相乘，并将结果赋值给第一个操作数
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE Half& operator/=(Half& a, const Half& b) {
  // 将两个半精度浮点数相除，并将结果赋值给第一个操作数
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(Half a, float b) {
  // 返回半精度浮点数和单精度浮点数的和
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(Half a, float b) {
  // 返回半精度浮点数和单精度浮点数的差
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(Half a, float b) {
  // 返回半精度浮点数和单精度浮点数的积
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(Half a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  // 返回半精度浮点数和单精度浮点数的商，忽略除零错误检查
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, Half b) {
  // 返回单精度浮点数和半精度浮点数的和
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, Half b) {
  // 返回单精度浮点数和半精度浮点数的差
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, Half b) {
  // 返回单精度浮点数和半精度浮点数的积
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, Half b)
    __ubsan_ignore_float_divide_by_zero__ {
  // 返回单精度浮点数和半精度浮点数的商，忽略除零错误检查
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const Half& b) {
  // 将单精度浮点数和半精度浮点数相加，并将结果赋值给第一个操作数
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const Half& b) {
  // 将单精度浮点数和半精度浮点数相减，并将结果赋值给第一个操作数
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const Half& b) {
  // 将单精度浮点数和半精度浮点数相乘，并将结果赋值给第一个操作数
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const Half& b) {
  // 将单精度浮点数和半精度浮点数相除，并将结果赋值给第一个操作数
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(Half a, double b) {
  // 返回半精度浮点数和双精度浮点数的和
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(Half a, double b) {
  // 返回半精度浮点数和双精度浮点数的差
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(Half a, double b) {
  // 返回半精度浮点数和双精度浮点数的积
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(Half a, double b)
    // 返回半精度浮点数和双精度浮点数的商，未完待续...
    # 忽略 undefined behavior sanitizer (UBSan) 的浮点除零检查
    __ubsan_ignore_float_divide_by_zero__ {
        # 将整数变量 a 转换为 double 类型，然后执行除法操作，避免浮点除零异常
        return static_cast<double>(a) / b;
}

// 定义双精度浮点数和半精度浮点数的加法运算符重载
inline C10_HOST_DEVICE double operator+(double a, Half b) {
  return a + static_cast<double>(b);
}
// 定义双精度浮点数和半精度浮点数的减法运算符重载
inline C10_HOST_DEVICE double operator-(double a, Half b) {
  return a - static_cast<double>(b);
}
// 定义双精度浮点数和半精度浮点数的乘法运算符重载
inline C10_HOST_DEVICE double operator*(double a, Half b) {
  return a * static_cast<double>(b);
}
// 定义双精度浮点数和半精度浮点数的除法运算符重载，忽略浮点数除零错误
inline C10_HOST_DEVICE double operator/(double a, Half b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

// 定义半精度浮点数和整数的加法运算符重载
inline C10_HOST_DEVICE Half operator+(Half a, int b) {
  return a + static_cast<Half>(b);
}
// 定义半精度浮点数和整数的减法运算符重载
inline C10_HOST_DEVICE Half operator-(Half a, int b) {
  return a - static_cast<Half>(b);
}
// 定义半精度浮点数和整数的乘法运算符重载
inline C10_HOST_DEVICE Half operator*(Half a, int b) {
  return a * static_cast<Half>(b);
}
// 定义半精度浮点数和整数的除法运算符重载
inline C10_HOST_DEVICE Half operator/(Half a, int b) {
  return a / static_cast<Half>(b);
}

// 定义整数和半精度浮点数的加法运算符重载
inline C10_HOST_DEVICE Half operator+(int a, Half b) {
  return static_cast<Half>(a) + b;
}
// 定义整数和半精度浮点数的减法运算符重载
inline C10_HOST_DEVICE Half operator-(int a, Half b) {
  return static_cast<Half>(a) - b;
}
// 定义整数和半精度浮点数的乘法运算符重载
inline C10_HOST_DEVICE Half operator*(int a, Half b) {
  return static_cast<Half>(a) * b;
}
// 定义整数和半精度浮点数的除法运算符重载
inline C10_HOST_DEVICE Half operator/(int a, Half b) {
  return static_cast<Half>(a) / b;
}

//// Arithmetic with int64_t

// 定义半精度浮点数和64位整数的加法运算符重载
inline C10_HOST_DEVICE Half operator+(Half a, int64_t b) {
  return a + static_cast<Half>(b);
}
// 定义半精度浮点数和64位整数的减法运算符重载
inline C10_HOST_DEVICE Half operator-(Half a, int64_t b) {
  return a - static_cast<Half>(b);
}
// 定义半精度浮点数和64位整数的乘法运算符重载
inline C10_HOST_DEVICE Half operator*(Half a, int64_t b) {
  return a * static_cast<Half>(b);
}
// 定义半精度浮点数和64位整数的除法运算符重载
inline C10_HOST_DEVICE Half operator/(Half a, int64_t b) {
  return a / static_cast<Half>(b);
}

// 定义64位整数和半精度浮点数的加法运算符重载
inline C10_HOST_DEVICE Half operator+(int64_t a, Half b) {
  return static_cast<Half>(a) + b;
}
// 定义64位整数和半精度浮点数的减法运算符重载
inline C10_HOST_DEVICE Half operator-(int64_t a, Half b) {
  return static_cast<Half>(a) - b;
}
// 定义64位整数和半精度浮点数的乘法运算符重载
inline C10_HOST_DEVICE Half operator*(int64_t a, Half b) {
  return static_cast<Half>(a) * b;
}
// 定义64位整数和半精度浮点数的除法运算符重载
inline C10_HOST_DEVICE Half operator/(int64_t a, Half b) {
  return static_cast<Half>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Half to float.

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::Half> {
 public:
  // 特化的 numeric_limits 类模板，针对 c10::Half 类型
  static constexpr bool is_specialized = true;
  // c10::Half 类型为有符号类型
  static constexpr bool is_signed = true;
  // c10::Half 类型不是整数类型
  static constexpr bool is_integer = false;
  // c10::Half 类型不是精确类型
  static constexpr bool is_exact = false;
  // c10::Half 类型支持无穷大
  static constexpr bool has_infinity = true;
  // c10::Half 类型支持静默 NaN
  static constexpr bool has_quiet_NaN = true;
  // c10::Half 类型支持信号 NaN
  static constexpr bool has_signaling_NaN = true;
  // c10::Half 类型的 denorm 特性与 float 类型相同
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  // c10::Half 类型的 denorm_loss 特性与 float 类型相同
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  // c10::Half 类型的舍入方式与 float 类型相同
  static constexpr auto round_style = numeric_limits<float>::round_style;
  // c10::Half 类型符合 IEC 559 标准
  static constexpr bool is_iec559 = true;
  // c10::Half 类型有界
  static constexpr bool is_bounded = true;
  // c10::Half 类型不是模数类型
  static constexpr bool is_modulo = false;
  // c10::Half 类型的有效位数为 11
  static constexpr int digits = 11;
  // c10::Half 类型的十进制位数为 3
  static constexpr int digits10 = 3;
  // c10::Half 类型的最大十进制位数为 5
  static constexpr int max_digits10 = 5;
  // c10::Half 类型的基数为 2
  static constexpr int radix = 2;
  // c10::Half 类型的最小指数为 -13
  static constexpr int min_exponent = -13;
  // c10::Half 类型的最小十进制指数为 -4
  static constexpr int min_exponent10 = -4;
  // c10::Half 类型的最大指数为 16
  static constexpr int max_exponent = 16;
  // c10::Half 类型的最大十进制指数为 4
  static constexpr int max_exponent10 = 4;
  // c10::Half 类型的 traps 特性与 float 类型相同
  static constexpr auto traps = numeric_limits<float>::traps;
  // c10::Half 类型的 tinyness_before 特性与 float 类型相同
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  // 返回 c10::Half 类型的最小正数值
  static constexpr c10::Half min() {
    return c10::Half(0x0400, c10::Half::from_bits());
  }
  // 返回 c10::Half 类型的最负数值
  static constexpr c10::Half lowest() {
    return c10::Half(0xFBFF, c10::Half::from_bits());
  }
  // 返回 c10::Half 类型的最大有限值
  static constexpr c10::Half max() {
    return c10::Half(0x7BFF, c10::Half::from_bits());
  }
  // 返回 c10::Half 类型的机器精度
  static constexpr c10::Half epsilon() {
    return c10::Half(0x1400, c10::Half::from_bits());
  }
  // 返回 c10::Half 类型的舍入误差
  static constexpr c10::Half round_error() {
    return c10::Half(0x3800, c10::Half::from_bits());
  }
  // 返回 c10::Half 类型的无穷大值
  static constexpr c10::Half infinity() {
    return c10::Half(0x7C00, c10::Half::from_bits());
  }
  // 返回 c10::Half 类型的静默 NaN 值
  static constexpr c10::Half quiet_NaN() {
    return c10::Half(0x7E00, c10::Half::from_bits());
  }
  // 返回 c10::Half 类型的信号 NaN 值
  static constexpr c10::Half signaling_NaN() {
    return c10::Half(0x7D00, c10::Half::from_bits());
  }
  // 返回 c10::Half 类型的最小 denorm 值
  static constexpr c10::Half denorm_min() {
    return c10::Half(0x0001, c10::Half::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
```