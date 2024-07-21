# `.\pytorch\c10\util\Float8_e4m3fnuz.h`

```py
#pragma once

/// Defines the Float8_e4m3fnuz type (8-bit floating-point) including
/// conversions to standard C types and basic arithmetic operations. Note that
/// arithmetic operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration remains the same as Float8_e4m3fn:
/// s eeee mmm
/// 1 sign bit
/// 4 exponent bits
/// 3 mantissa bits
/// The key differences versus Float8_e4m3fn are:
/// bias = 8
/// no infinities or negative zero
/// NaN only when sign bit is 1, rest all 0s
///
/// Implementation based on the paper https://arxiv.org/pdf/2206.02915.pdf and
/// the existing Float8_e4m3fn implementation.

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/floating_point_utils.h>
#include <type_traits>

#if defined(__cplusplus)
#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#include <stdint.h>
#endif

#include <iosfwd>
#include <ostream>

namespace c10 {

namespace detail {

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FNUZ format, in bit representation.
 */
inline C10_HOST_DEVICE uint8_t fp8e4m3fnuz_from_fp32_value(float f) {
  /*
   * Binary representation of 256.0f, which is the first value not representable
   * (i.e. the first value which would overflow in to the sign bit, resulting in
   * a NaN) in fp8e4m3fnuz range:
   * 1 0000 000 - fp8e4m3fnuz
   * 0 10000111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fnuz_max = UINT32_C(0x87) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e4m3fnuz normal range
   * into denorm representation
   * magic number: ((127 - 8) + (23 - 3) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(0x8C) << 23;

  uint32_t f_bits = fp32_to_bits(f);

  uint32_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  f_bits ^= sign;

  if (f_bits >= fnuz_max) {
    // NaN -- sign bit set to 1, rest 0s.
    return 0x80;
  }

  if (f_bits < (UINT32_C(0x78) << 23) /* 2^-7 in float32 */) {
    // Input exponent is less than -7, the smallest e4m3fnuz exponent, so the
    // number will become subnormal.
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
    if (result == 0) {
      // fnuz types don't have negative zero.
      return 0;
    }
  } else {
    // resulting mantissa is odd
    uint8_t mant_odd = (f_bits >> 20) & 1;

    // update exponent, rounding bias part 1
    f_bits += ((uint32_t)(8 - 127) << 23) + 0x7FFFF;

    // rounding bias part 2
    f_bits += mant_odd;
    # 将 f_bits 和 mant_odd 相加，更新 f_bits 的值

    // take the bits!
    result = static_cast<uint8_t>(f_bits >> 20);
    # 取 f_bits 的高 20 位，然后转换为 uint8_t 类型，存入 result 变量中

  }

  result |= sign >> 24;
  # 将 sign 右移 24 位后的结果按位或运算到 result 上

  return result;
  # 返回计算得到的 result 变量作为函数的结果
} // 结束 detail 命名空间

} // 结束命名空间 detail

// 定义一个结构 Float8_e4m3fnuz，要求按照 1 字节对齐
struct alignas(1) Float8_e4m3fnuz {
  uint8_t x; // 一个无符号 8 位整数 x

  // 内部结构 from_bits_t
  struct from_bits_t {};

  // 静态方法，返回 from_bits_t 类型的对象
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  // 默认构造函数
  Float8_e4m3fnuz() = default;

  // 通过位表示构造的构造函数
  constexpr C10_HOST_DEVICE Float8_e4m3fnuz(uint8_t bits, from_bits_t)
      : x(bits) {}

  // 转换为 float 类型的运算符重载
  inline C10_HOST_DEVICE Float8_e4m3fnuz(float value);

  // 转换为 float 类型的运算符重载
  inline C10_HOST_DEVICE operator float() const;

  // 判断是否为 NaN 的方法
  inline C10_HOST_DEVICE bool isnan() const;
};

// 重载流输出操作符，输出 Float8_e4m3fnuz 对象的 float 值
C10_API inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e4m3fnuz& value) {
  out << (float)value; // 输出 value 的 float 值到流 out 中
  return out;
}

} // 结束命名空间 c10

// 包含 Float8_e4m3fnuz 的内联实现文件
#include <c10/util/Float8_e4m3fnuz-inl.h> // IWYU pragma: keep
```