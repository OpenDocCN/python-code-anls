# `.\pytorch\c10\util\Half.h`

```
#pragma once

/// Defines the Half type (half-precision floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32, instead of using CUDA half intrinsics.
/// Most uses of this type within ATen are memory bound, including the
/// element-wise kernels, and the half intrinsics aren't efficient on all GPUs.
/// If you are writing a compute bound kernel, you can use the CUDA half
/// intrinsics directly on the Half type from device code.

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/bit_cast.h>
#include <c10/util/complex.h>
#include <c10/util/floating_point_utils.h>
#include <type_traits>

#if defined(__cplusplus)
#include <cmath>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <ostream>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#endif

#if defined(CL_SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp> // for SYCL 1.2.1
#elif defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp> // for SYCL 2020
#endif

#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
#include <arm_neon.h>
#endif

namespace c10 {

namespace detail {

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */

/// This function converts a 16-bit half-precision floating-point number
/// represented as a 16-bit unsigned integer to a 32-bit single-precision
/// floating-point number represented as a 32-bit unsigned integer.
/// It performs the conversion based on the bit representation of the numbers.
/// This implementation avoids using traditional floating-point operations.
static inline uint32_t convert_half_to_float(uint16_t h) {
    // Bit-casting the half-precision value to its equivalent single-precision
    // value by reinterpreting the bit pattern.
    return detail::bit_cast<uint32_t, uint16_t>(h) << 16;
}

} // namespace detail

} // namespace c10
inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
  // 将半精度浮点数扩展到32位并将其移至32位字的高位：
  //      +---+-----+------------+-------------------+
  //      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
  //      +---+-----+------------+-------------------+
  // Bits  31  26-30    16-25            0-15
  const uint32_t w = (uint32_t)h << 16;

  // 提取输入数的符号到32位字的高位：
  //      +---+----------------------------------+
  //      | S |0000000 00000000 00000000 00000000|
  //      +---+----------------------------------+
  // Bits  31                 0-31
  const uint32_t sign = w & UINT32_C(0x80000000);

  // 提取输入数的尾数和偏置指数到32位字的0-30位：
  //      +---+-----+------------+-------------------+
  //      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
  //      +---+-----+------------+-------------------+
  // Bits  30  27-31     17-26            0-16
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);

  // 重新规范化移位是使半精度数规范化所需的位移数。如果初始数已规范化，则高6位（符号位为0和5位指数）为1。
  // 在这种情况下，renorm_shift == 0。如果数是非规范化的，则 renorm_shift > 0。
  // 注意，如果我们将非规范化的 nonsign 按 renorm_shift 左移，尾数的单位位将移动到指数中，将偏置指数变为1，
  // 并使尾数变得规范化（即没有前导1）。
#ifdef _MSC_VER
  unsigned long nonsign_bsr;
  _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
  uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
  uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
}
#endif
  // 如果 renorm_shift 大于 5，则将其减去 5；否则设为 0
  renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
  /*
   * 如果半精度数的指数为 15，则加法溢出至第 31 位，并且后续的右移将高 9 位变为 1。
   * 因此，如果半精度数的指数为 15（即为 NaN 或无穷大），inf_nan_mask == 0x7F800000；
   * 否则为 0x00000000。
   */
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
  /*
   * 如果 nonsign 为 0，则溢出为 0xFFFFFFFF，将第 31 位变为 1。否则，第 31 位保持为 0。
   * 通过右移 31 位有符号位扩展，将 bit 31 广播到 zero_mask 的所有位。
   * 因此，如果半精度数为零（+0.0h 或 -0.0h），zero_mask == 0xFFFFFFFF；否则为 0x00000000。
   */
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  /*
   * 1. 将 nonsign 左移 renorm_shift 位，以规范化它（如果输入为非规范数）。
   * 2. 将 nonsign 右移 3 位，使得指数（原始为 5 位）成为 8 位字段，而 10 位尾数移入 IEEE 单精度数的 23 位尾数的高 10 位。
   * 3. 将 0x70 加到指数（从第 23 位开始），以补偿指数偏差的差异（单精度数的 0x7F 减去半精度数的 0xF）。
   * 4. 从指数（从第 23 位开始）减去 renorm_shift，以考虑再规范化。由于 renorm_shift 小于 0x70，这可以与步骤 3 结合。
   * 5. 与 inf_nan_mask 二进制 OR，如果输入为 NaN 或无穷大，则将指数变为 0xFF。
   * 6. 与 ~zero_mask 的二进制 ANDNOT，如果输入为零，则将尾数和指数变为零。
   * 7. 与输入数的符号结合。
   */
  return sign |
      ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) |
        inf_nan_mask) &
       ~zero_mask);
}

/*
 * 将 IEEE 半精度格式的 16 位浮点数（位表示）转换为 IEEE 单精度格式的 32 位浮点数。
 *
 * @note 此实现依赖于 IEEE 类似（不假设舍入模式和不对非规范数执行操作）的浮点运算，
 *       并在整数和浮点变量之间进行位转换。
 */
}

/*
 * 将 IEEE 单精度格式的 32 位浮点数转换为 IEEE 半精度格式的 16 位浮点数（位表示）。
 *
 * @note 此实现依赖于 IEEE 类似（不假设舍入模式和不对非规范数执行操作）的浮点运算，
 *       并在整数和浮点变量之间进行位转换。
 */
// 将单精度浮点数转换为 IEEE 754 半精度浮点数表示
inline uint16_t fp16_ieee_from_fp32_value(float f) {
  // 定义用于转换的比例常量，将无穷大映射到 0x1.0p+112，将零映射到 0x1.0p-110
  constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
  constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
  // 初始化用于转换的比例值
  float scale_to_inf_val = 0, scale_to_zero_val = 0;
  // 将比例常量的位表示拷贝到对应的浮点数值
  std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
  std::memcpy(&scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
  // 定义实际使用的比例值
  const float scale_to_inf = scale_to_inf_val;
  const float scale_to_zero = scale_to_zero_val;

  // 根据不同编译器条件计算基础值 base
#if defined(_MSC_VER) && _MSC_VER == 1916
  // Microsoft Visual Studio 2017 版本 15.9 中特有的情况
  float base = ((signbit(f) != 0 ? -f : f) * scale_to_inf) * scale_to_zero;
#else
  // 默认情况下计算 base
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
#endif

  // 获取单精度浮点数 f 的位表示
  const uint32_t w = fp32_to_bits(f);
  // 左移一位的 w 值
  const uint32_t shl1_w = w + w;
  // 提取符号位
  const uint32_t sign = w & UINT32_C(0x80000000);
  // 计算偏置值 bias
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  // 根据偏置值调整 base
  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  // 提取 base 的位表示
  const uint32_t bits = fp32_to_bits(base);
  // 提取指数部分和尾数部分的位表示
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  // 计算非符号部分的位表示
  const uint32_t nonsign = exp_bits + mantissa_bits;
  // 组合成最终的 IEEE 754 半精度浮点数表示并返回
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
// 将 IEEE 754 半精度浮点数表示转换为具体的浮点数值
inline float16_t fp16_from_bits(uint16_t h) {
  return c10::bit_cast<float16_t>(h);
}

// 将具体的浮点数值转换为 IEEE 754 半精度浮点数表示
inline uint16_t fp16_to_bits(float16_t f) {
  return c10::bit_cast<uint16_t>(f);
}

// 对于 ARM 架构的情况，将半精度浮点数值直接转换为单精度浮点数值
inline float native_fp16_to_fp32_value(uint16_t h) {
  return static_cast<float>(fp16_from_bits(h));
}

// 对于 ARM 架构的情况，将单精度浮点数值转换为 IEEE 754 半精度浮点数表示
inline uint16_t native_fp16_from_fp32_value(float f) {
  return fp16_to_bits(static_cast<float16_t>(f));
}
#endif

} // namespace detail

// 半精度浮点数的结构体定义，使用 2 字节对齐
struct alignas(2) Half {
  unsigned short x; // 用于存储半精度浮点数的值

  struct from_bits_t {}; // 表示从位表示构造的辅助结构体
  // 返回用于从位表示构造半精度浮点数的类型
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  // 根据不同的编译器条件定义构造函数
#if defined(USE_ROCM)
  // ROCm 平台使用的构造函数声明
  C10_HOST_DEVICE Half() = default;
#else
  // 其他平台使用的默认构造函数声明
  Half() = default;
#endif

  // 根据位表示创建半精度浮点数的构造函数声明
  constexpr C10_HOST_DEVICE Half(unsigned short bits, from_bits_t) : x(bits) {}

  // 根据不同的编译器条件定义类型转换构造函数和操作符重载函数
#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
  // ARM 架构使用的构造函数和类型转换操作符声明
  inline Half(float16_t value);
  inline operator float16_t() const;
#else
  // 其他平台使用的构造函数和类型转换操作符声明
  inline C10_HOST_DEVICE Half(float value);
  inline C10_HOST_DEVICE operator float() const;
#endif

  // 根据不同的编译器条件定义与 CUDA 和 HIP 相关的构造函数和类型转换操作符
#if defined(__CUDACC__) || defined(__HIPCC__)
  // CUDA 和 HIP 使用的构造函数和类型转换操作符声明
  inline C10_HOST_DEVICE Half(const __half& value);
  inline C10_HOST_DEVICE operator __half() const;
#endif
#ifdef SYCL_LANGUAGE_VERSION
  // SYCL 使用的构造函数和类型转换操作符声明
  inline C10_HOST_DEVICE Half(const sycl::half& value);
  inline C10_HOST_DEVICE operator sycl::half() const;
#endif
};

// TODO : move to complex.h
// 模板特化：定义一个 complex 类模板，其元素类型为 Half，强制要求 4 字节对齐
template <>
struct alignas(4) complex<Half> {
  // 实部和虚部，类型为 Half
  Half real_;
  Half imag_;

  // 构造函数
  complex() = default;
  // Half 类型的构造函数不是 constexpr，因此这个构造函数也不能是 constexpr
  C10_HOST_DEVICE explicit inline complex(const Half& real, const Half& imag)
      : real_(real), imag_(imag) {}
  // 从 c10::complex<float> 类型进行转换构造
  C10_HOST_DEVICE inline complex(const c10::complex<float>& value)
      : real_(value.real()), imag_(value.imag()) {}

  // 转换运算符，将 complex<Half> 转换为 c10::complex<float>
  inline C10_HOST_DEVICE operator c10::complex<float>() const {
    return {real_, imag_};
  }

  // 返回实部和虚部
  constexpr C10_HOST_DEVICE Half real() const {
    return real_;
  }
  constexpr C10_HOST_DEVICE Half imag() const {
    return imag_;
  }

  // 复合赋值运算符重载：加法
  C10_HOST_DEVICE complex<Half>& operator+=(const complex<Half>& other) {
    real_ = static_cast<float>(real_) + static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) + static_cast<float>(other.imag_);
    return *this;
  }

  // 复合赋值运算符重载：减法
  C10_HOST_DEVICE complex<Half>& operator-=(const complex<Half>& other) {
    real_ = static_cast<float>(real_) - static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) - static_cast<float>(other.imag_);
    return *this;
  }

  // 复合赋值运算符重载：乘法
  C10_HOST_DEVICE complex<Half>& operator*=(const complex<Half>& other) {
    auto a = static_cast<float>(real_);
    auto b = static_cast<float>(imag_);
    auto c = static_cast<float>(other.real());
    auto d = static_cast<float>(other.imag());
    real_ = a * c - b * d;
    imag_ = a * d + b * c;
    return *this;
  }
};

// 在某些 MSVC 版本中，构建时可能会出现编译器错误。
// C4146: 对无符号类型应用一元负运算符，结果仍然是无符号的
// C4804: 在操作中对类型 'bool' 的不安全使用
// 可以通过禁用以下警告来解决。
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)
#pragma warning(disable : 4804)
#pragma warning(disable : 4018)
#endif

// 溢出检查可能涉及浮点到整数的转换，这可能触发精度丢失警告。
// 一旦代码被修复，可以重新启用警告。参见 T58053069。
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

// 对于布尔类型，可以将其转换为任何类型。
// 在未专门处理布尔类型的情况下，在 pytorch_linux_trusty_py2_7_9_build 中：
// `error: comparison of constant '255' with boolean expression is always false`
// 因此，下面的函数模板是为了避免这种情况而设计的。
template <typename To, typename From>
std::enable_if_t<std::is_same_v<From, bool>, bool> overflows(
    From /*f*/,
    bool strict_unsigned = false) {
  return false;
}

// 对于整数类型（但不是布尔类型），跳过 isnan 和 isinf 检查
template <typename To, typename From>
std::enable_if_t<std::is_integral_v<From> && !std::is_same_v<From, bool>, bool>
// 定义名为 overflows 的函数模板，处理从类型 From 到类型 To 的转换，检查是否溢出
overflows(From f, bool strict_unsigned = false) {
  // 使用 limit 类型别名指定 To 类型的数值极限
  using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
  // 如果 To 类型是无符号并且 From 类型是有符号，则允许负数通过二进制补码算术进行包装
  if constexpr (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    // 允许负数通过二进制补码算术进行包装。例如，对于 uint8，这允许 `a - b` 被视为 `a + 255 * b`
    if (!strict_unsigned) {
      return greater_than_max<To>(f) ||
          (c10::is_negative(f) &&
           -static_cast<uint64_t>(f) > static_cast<uint64_t>(limit::max()));
    }
  }
  // 检查是否小于 To 类型的最小值或者大于 To 类型的最大值
  return c10::less_than_lowest<To>(f) || greater_than_max<To>(f);
}

// 定义名为 overflows 的函数模板，处理从浮点类型 From 到类型 To 的转换，检查是否溢出
template <typename To, typename From>
std::enable_if_t<std::is_floating_point_v<From>, bool> overflows(
    From f,
    bool strict_unsigned = false) {
  // 使用 limit 类型别名指定 To 类型的数值极限
  using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
  // 如果 To 类型支持无限大，并且 f 是无穷大
  if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
    return false;
  }
  // 如果 To 类型不支持静默 NaN，并且 f 是 NaN
  if (!limit::has_quiet_NaN && (f != f)) {
    return true;
  }
  // 检查是否小于 To 类型的最小值或者大于 To 类型的最大值
  return f < limit::lowest() || f > limit::max();
}

// 定义名为 overflows 的函数模板，处理从复数类型 From 到类型 To 的转换，检查是否溢出
template <typename To, typename From>
std::enable_if_t<is_complex<From>::value, bool> overflows(
    From f,
    bool strict_unsigned = false) {
  // 如果 To 类型不是复数，而 From 类型的虚部不为零，则认为溢出
  if (!is_complex<To>::value && f.imag() != 0) {
    return true;
  }
  // 逐分量检查是否溢出
  // （技术上，当 !is_complex<To> 时，虚部溢出检查保证为假，但是任何优化器都应该能够理解这一点。）
  return overflows<
             typename scalar_value_type<To>::type,
             typename From::value_type>(f.real()) ||
      overflows<
             typename scalar_value_type<To>::type,
             typename From::value_type>(f.imag());
}

// 定义命名空间 c10 下的内联函数，重载输出流运算符，以便输出 Half 类型对象到输出流
C10_API inline std::ostream& operator<<(std::ostream& out, const Half& value) {
  // 输出 Half 类型对象的值转换为 float 类型后的结果到输出流
  out << (float)value;
  return out;
}

} // namespace c10

// 包含 Half 类型的实现文件，保持 IWYU（Include What You Use）的指令
#include <c10/util/Half-inl.h> // IWYU pragma: keep
```