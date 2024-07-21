# `.\pytorch\c10\util\Float8_e5m2fnuz.h`

```
/*
 * 头文件防止重复包含，指示此处定义了 Float8_e5m2fnuz 类型，代表8位浮点数，
 * 包括转换为标准C类型和基本算术操作。注意算术操作是通过转换为float32后执行的。
 * 二进制配置保持与e5m2相同：
 * s eeeee mm
 * 1个符号位
 * 5个指数位
 * 2个尾数位
 * e5m2fnuz 的主要区别在于：
 * 偏置 = 16
 * 没有无穷大或负零
 * 仅当符号位为1，其余都为0时才有NaN
 *
 * 实现基于文档 https://arxiv.org/pdf/2206.02915.pdf 和现有的 Float8_e4m3fn 实现。
 */

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/floating_point_utils.h>

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
 * 将IEEE单精度格式的32位浮点数转换为fp8 E5M2格式的8位浮点数，以位表示形式。
 */
inline C10_HOST_DEVICE uint8_t fp8e5m2fnuz_from_fp32_value(float f) {
  /*
   * 65536.0f 的二进制表示，这是在fp8e4m3fnuz范围内不能表示的第一个值
   * （即第一个将溢出到符号位，导致NaN的值）：
   * 1 00000 00 - fp8e5m2fnuz
   * 0 10001111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fnuz_max = UINT32_C(0x8F) << 23;

  /*
   * 用于将低于fp8e5m2fnuz正常范围的fp32数转换为非规范化表示的掩码。
   * 魔术数：((127 - 16) + (23 - 2) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(0x85) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint32_t result = 0u;

  /*
   * 将输入数字的符号提取到32位字的高位：
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * 位    31                0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * 将符号位设置为0
   */
  f_bits ^= sign;

  if (f_bits >= fnuz_max) {
    // NaN -- 符号位设置为1，其余为0
    return 0x80;
  }

  if (f_bits < (UINT32_C(0x70) << 23) /* 2^-15 in float32 */) {
    // 输入指数小于-15，即e5m2fnuz最小指数，因此数字将变为次正规化。
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
    if (result == 0) {
      // fnuz类型没有负零。
      return 0;
    }
  } else {
    // 结果尾数为奇数
    uint8_t mant_odd = (f_bits >> 21) & 1;

    // 更新指数，四舍五入偏置部分1
    f_bits += ((uint32_t)(16 - 127) << 23) + 0xFFFFF;

    // 四舍五入偏置部分2
    f_bits += mant_odd;
    // 将 f_bits 的高 21 位向右移动，并将结果转换为 8 位无符号整数
    result = static_cast<uint8_t>(f_bits >> 21);
  }

  // 将符号位右移 24 位，并将结果按位或运算到 result 中
  result |= sign >> 24;
  // 返回最终的结果值
  return result;
} // namespace detail



} // namespace detail



struct alignas(1) Float8_e5m2fnuz {
  uint8_t x;

  // 内部类，用于标记从位表示创建对象
  struct from_bits_t {};
  
  // 静态函数，返回一个 from_bits_t 类型的对象
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  // 默认构造函数
  Float8_e5m2fnuz() = default;

  // 从位表示构造对象的构造函数
  constexpr C10_HOST_DEVICE Float8_e5m2fnuz(uint8_t bits, from_bits_t)
      : x(bits) {}
  
  // 从浮点数值构造对象的构造函数声明
  inline C10_HOST_DEVICE Float8_e5m2fnuz(float value);
  
  // 类型转换操作符，将对象转换为浮点数值
  inline C10_HOST_DEVICE operator float() const;
  
  // 判断对象是否表示 NaN 的成员函数声明
  inline C10_HOST_DEVICE bool isnan() const;
  
  // 判断对象是否表示无穷大的成员函数声明
  inline C10_HOST_DEVICE bool isinf() const;
};



C10_API inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e5m2fnuz& value) {
  // 重载输出流操作符，输出对象的浮点数值表示
  out << (float)value;
  return out;
}



} // namespace c10

// 包含 Float8_e5m2fnuz 类的内联实现文件
#include <c10/util/Float8_e5m2fnuz-inl.h> // IWYU pragma: keep



} // namespace c10
```