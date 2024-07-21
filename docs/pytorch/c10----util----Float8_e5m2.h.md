# `.\pytorch\c10\util\Float8_e5m2.h`

```
#pragma once

/// Defines the Float8_e5m2 type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration:
/// s eeeee mm
/// 1 sign bit
/// 5 exponent bits
/// 2 mantissa bits
/// bias = 15
///
/// Implementation based on the paper https://arxiv.org/pdf/2209.05433.pdf
/// and inspired by Half implementation from pytorch/c10/util/Half.h

#include <c10/util/Half.h>

namespace c10 {

namespace detail {

/*
 * Convert a 8-bit floating-point number in fp8 E5M2 format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline C10_HOST_DEVICE float fp8e5m2_to_fp32_value(uint8_t input) {
  /*
   * Extend the fp8 E5M2 number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+----+---+-----------------------------+
   *      | S |EEEEE|MM|0000 0000 0000 0000 0000 0000|
   *      +---+----+---+-----------------------------+
   * Bits  31 26-30 24-25          0-23
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  uint16_t half_representation = input;
  half_representation <<= 8;
  return fp16_ieee_to_fp32_value(half_representation);
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
inline C10_HOST_DEVICE uint8_t fp8e5m2_from_fp32_value(float f) {
  /*
   * Binary representation of fp32 infinity
   * 0 11111111 00000000000000000000000
   */
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;

  /*
   * Binary representation of 65536.0f, which is the first value
   * not representable in fp8e5m2 range:
   * 0 11111 00 - fp8e5m2
   * 0 10001111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e5m2 normal range
   * into denorm representation
   * magic number: ((127 - 15) + (23 - 2) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint8_t result = 0u;

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

  if (f_bits >= fp8_max) {
    // NaN - all exponent and mantissa bits set to 1
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (f_bits < (UINT32_C(113) << 23)) {
      // 如果 f_bits 小于 2^(-14)，即最小的正规化 fp8e5m2 数字
      // 将 f_bits 增加到 denorm_mask 对应的 fp32 数值，并转换为 uint8_t 类型结果
      f_bits =
          fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      // 结果的尾数是奇数
      uint32_t mant_odd = (f_bits >> 21) & 1;

      // 更新指数，进行舍入偏置的第一部分
      f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;

      // 舍入偏置的第二部分
      f_bits += mant_odd;

      // 取出这些位！
      result = static_cast<uint8_t>(f_bits >> 21);
    }
  }

  // 将符号位合并到结果中
  result |= static_cast<uint8_t>(sign >> 24);
  // 返回结果
  return result;
}

} // namespace detail

// 定义一个名为 Float8_e5m2 的结构体，使用 alignas(1) 进行字节对齐
struct alignas(1) Float8_e5m2 {
  uint8_t x;  // 单个成员变量 x，用于存储 8 位的无符号整数

  // 内部嵌套结构 from_bits_t，用于构造特定类型的实例
  struct from_bits_t {};

  // 静态成员函数 from_bits，返回一个 from_bits_t 类型的实例
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  // 默认构造函数 Float8_e5m2()，使用默认方式初始化对象
  Float8_e5m2() = default;

  // 构造函数 Float8_e5m2，接受一个 bits 参数和 from_bits_t 类型的标记，用于初始化对象
  constexpr C10_HOST_DEVICE Float8_e5m2(uint8_t bits, from_bits_t) : x(bits) {}

  // 浮点数构造函数 Float8_e5m2，接受一个 float 类型的 value 参数，用于初始化对象
  inline C10_HOST_DEVICE Float8_e5m2(float value);

  // 类型转换运算符，将 Float8_e5m2 类型对象转换为 float 类型
  inline C10_HOST_DEVICE operator float() const;

  // 检查对象是否表示 NaN（非数值）
  inline C10_HOST_DEVICE bool isnan() const;

  // 检查对象是否表示无穷大
  inline C10_HOST_DEVICE bool isinf() const;
};

// 重载运算符 <<，用于将 Float8_e5m2 对象输出到 ostream 类型的输出流 out 中
C10_API inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e5m2& value) {
  out << (float)value;  // 将 Float8_e5m2 对象转换为 float 类型，并输出到输出流 out 中
  return out;
}

} // namespace c10

// 包含 Float8_e5m2 的内联实现文件
#include <c10/util/Float8_e5m2-inl.h> // IWYU pragma: keep
```