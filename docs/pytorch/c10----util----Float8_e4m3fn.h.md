# `.\pytorch\c10\util\Float8_e4m3fn.h`

```py
/*
 * Define the Float8_e4m3fn type (8-bit floating-point) including conversions
 * to standard C types and basic arithmetic operations. Note that arithmetic
 * operations are implemented by converting to floating point and
 * performing the operation in float32.
 * Binary configuration:
 * s eeee mmm
 * 1 sign bit
 * 4 exponent bits
 * 3 mantissa bits
 * bias = 7
 *
 * Implementation based on the paper https://arxiv.org/pdf/2209.05433.pdf
 * and inspired by Half implementation from pytorch/c10/util/Half.h
 */

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/floating_point_utils.h>

#if defined(__cplusplus)
#include <cmath>
#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#include <stdint.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <climits>
#include <iostream>

namespace c10 {

namespace detail {

/*
 * Convert a 8-bit floating-point number in fp8 E4M3FN format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline C10_HOST_DEVICE float fp8e4m3fn_to_fp32_value(uint8_t input) {
  /*
   * Extend the fp8 E4M3FN number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+----+---+-----------------------------+
   *      | S |EEEE|MMM|0000 0000 0000 0000 0000 0000|
   *      +---+----+---+-----------------------------+
   * Bits  31 27-30 24-26          0-23
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)input << 24;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the bits 0-30
   * of the 32-bit word:
   *
   *      +---+----+---+-----------------------------+
   *      | S |EEEE|MMM|0000 0000 0000 0000 0000 0000|
   *      +---+----+---+-----------------------------+
   * Bits  31  27-30 24-26      0-23
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * half-precision number normalized. If the initial number is normalized, some
   * of its high 5 bits (sign == 0 and 4-bit exponent) equals one. In this case
   * renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note
   * that if we shift denormalized nonsign by renorm_shift, the unit bit of
   * mantissa will shift into exponent, turning the biased exponent into 1, and
   * making mantissa normalized (i.e. without leading 1).
   */
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // 如果在 CUDA 或 HIP 设备编译环境下，则使用 CUDA 的 __clz 函数计算 nonsign 的前导零位数
  uint32_t renorm_shift = __clz(nonsign);
#elif defined(__SYCL_DEVICE_ONLY__)
  // 注意：__builtin_clz 不支持零作为输入
  // 在 SYCL 设备编译环境下，如果 nonsign 不为零，则使用 __builtin_clz 计算其前导零位数；否则设为 uint32_t 类型的最大值
  uint32_t renorm_shift =
      nonsign != 0 ? __builtin_clz(nonsign) : sizeof(uint32_t) * CHAR_BIT;
#elif defined(_MSC_VER)
  // 在 MSVC 编译环境下，使用 _BitScanReverse 获取 nonsign 的最高位设置的位数，计算后将结果与 31 异或得到 renorm_shift
  unsigned long nonsign_bsr;
  _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
  uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
  // 如果以上条件都不满足，则默认使用 __builtin_clz 计算 nonsign 的前导零位数（注意：零不支持输入）
  uint32_t renorm_shift =
      nonsign != 0 ? __builtin_clz(nonsign) : sizeof(uint32_t) * CHAR_BIT;
#endif
  // 如果 renorm_shift 大于 4，则将其减去 4，否则设为 0
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  /*
   * 当 fp8e4m3fn 数字的所有指数和尾数位均设为 1 时，加法会导致溢出到位 31，
   * 随后的位移操作将使高 9 位变为 1。因此，如果 fp8e4m3fn 数字为 NaN，则 inf_nan_mask == 0x7F800000；否则为 0x00000000。
   */
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x01000000) >> 8) & INT32_C(0x7F800000);
  /*
   * 如果 nonsign 为 0，则溢出为 0xFFFFFFFF，将位 31 设置为 1；否则位 31 保持为 0。
   * 通过有符号右移 31 位，将位 31 扩展到 zero_mask 的所有位。因此，如果半精度数字为零（+0.0h 或 -0.0h），zero_mask == 0xFFFFFFFF；否则为 0x00000000。
   */
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  /*
   * 1. 将 nonsign 左移 renorm_shift 位以规范化它（如果输入为非规范化数）
   * 2. 将 nonsign 右移 4 位，使指数（原始为 4 位）成为 8 位字段，3 位尾数移入 IEEE 单精度数的 23 位尾数的高 3 位
   * 3. 将指数（从位 23 开始）加上 0x78，以补偿指数偏差的不同（单精度数的 0x7F 减去 fp8e4m3fn 数的 0x07）
   * 4. 从指数（从位 23 开始）中减去 renorm_shift，以考虑重新规范化。由于 renorm_shift 小于 0x78，可以与步骤 3 结合
   * 5. 与 inf_nan_mask 进行按位或运算，如果输入为 NaN 或无穷大，则将指数变为 0xFF
   * 6. 与 zero_mask 进行按位非运算，如果输入为零，则将尾数和指数变为零
   * 7. 与输入数的符号进行组合
   */
  uint32_t result = sign |
      ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
        inf_nan_mask) &
       ~zero_mask);
  // 返回转换为单精度数的 result
  return fp32_from_bits(result);
}
inline C10_HOST_DEVICE uint8_t fp8e4m3fn_from_fp32_value(float f) {
    /*
     * 480.0f 的二进制表示，超出 fp8e4m3fn 能表示的范围：
     * 0 1111 111 - fp8e4m3fn
     * 0 10000111 11100000000000000000000 - fp32
     */
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

    /*
     * 小于 fp8e4m3fn 的正常范围的 fp32 数字转换成 denorm 表示的掩码
     * 魔数：((127 - 7) + (23 - 3) + 1)
     */
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

    // 将输入 f 转换为其位表示
    uint32_t f_bits = fp32_to_bits(f);

    // 初始化结果变量
    uint8_t result = 0u;

    /*
     * 将输入数字的符号提取到 32 位字中的高位：
     *
     *      +---+----------------------------------+
     *      | S |0000000 00000000 00000000 00000000|
     *      +---+----------------------------------+
     * 位    31                 0-31
     */
    const uint32_t sign = f_bits & UINT32_C(0x80000000);

    /*
     * 将符号位设置为 0
     */
    f_bits ^= sign;

    if (f_bits >= fp8_max) {
        // NaN - 所有指数和尾数位都设置为 1
        result = 0x7f;
    } else {
        if (f_bits < (UINT32_C(121) << 23)) {
            // 输入数字小于 2^(-6)，这是 fp8e4m3fn 的最小正常数
            f_bits =
                fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
            result = static_cast<uint8_t>(f_bits - denorm_mask);
        } else {
            // 结果尾数为奇数
            uint8_t mant_odd = (f_bits >> 20) & 1;

            // 更新指数，四舍五入偏差部分 1
            f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;

            // 四舍五入偏差部分 2
            f_bits += mant_odd;

            // 取得位！
            result = static_cast<uint8_t>(f_bits >> 20);
        }
    }

    // 将符号位加入结果
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}

} // namespace detail

struct alignas(1) Float8_e4m3fn {
    uint8_t x;

    // 从位表示构造 Float8_e4m3fn 的静态方法
    struct from_bits_t {};
    C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
        return from_bits_t();
    }

    // 默认构造函数
    Float8_e4m3fn() = default;

    // 从位表示构造 Float8_e4m3fn 的构造函数
    constexpr C10_HOST_DEVICE Float8_e4m3fn(uint8_t bits, from_bits_t)
        : x(bits) {}

    // 从浮点数值构造 Float8_e4m3fn 的构造函数
    inline C10_HOST_DEVICE Float8_e4m3fn(float value);

    // 转换为浮点数的操作符重载
    inline C10_HOST_DEVICE operator float() const;

    // 判断是否为 NaN 的方法
    inline C10_HOST_DEVICE bool isnan() const;
};

// Float8_e4m3fn 类的输出流操作符重载
C10_API inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e4m3fn& value) {
    out << (float)value;
    return out;
}

} // namespace c10

// 包含 Float8_e4m3fn 的内联实现文件
#include <c10/util/Float8_e4m3fn-inl.h> // IWYU pragma: keep
```