# `.\pytorch\test\cpp\tensorexpr\gtest_assert_float_eq.h`

```py
#pragma once

#include <cmath>
// 包含 cmath 头文件，提供数学函数和宏

using Bits = uint32_t;

// this avoids the "dereferencing type-punned pointer
// will break strict-aliasing rules" error
// 定义 Float 联合体，用于处理浮点数和位表示之间的转换
union Float {
  float float_;
  Bits bits_;
};

// # of bits in a number.
// 定义常量 kBitCount，表示一个数字的位数
static const size_t kBitCount = 8 * sizeof(Bits);
// The mask for the sign bit.
// 定义常量 kSignBitMask，表示符号位的掩码
static const Bits kSignBitMask = static_cast<Bits>(1) << (kBitCount - 1);

// GOOGLETEST_CM0001 DO NOT DELETE

// Converts an integer from the sign-and-magnitude representation to
// the biased representation.  More precisely, let N be 2 to the
// power of (kBitCount - 1), an integer x is represented by the
// unsigned number x + N.
//
// For instance,
//
//   -N + 1 (the most negative number representable using
//          sign-and-magnitude) is represented by 1;
//   0      is represented by N; and
//   N - 1  (the biggest number representable using
//          sign-and-magnitude) is represented by 2N - 1.
//
// Read http://en.wikipedia.org/wiki/Signed_number_representations
// for more details on signed number representations.
// 定义函数 SignAndMagnitudeToBiased，将符号-幅值表示转换为偏置表示
static Bits SignAndMagnitudeToBiased(const Bits& sam) {
  if (kSignBitMask & sam) {
    // sam represents a negative number.
    return ~sam + 1;
  } else {
    // sam represents a positive number.
    // 返回 sam 对应的偏置值
   `
    return kSignBitMask | sam;
  }


注释：


    # 将 kSignBitMask 和 sam 进行按位或运算，并返回结果
    # kSignBitMask 是一个表示符号位的
// 给定两个以符号加绝对值形式表示的数字，计算它们之间的距离作为无符号数返回。
static Bits DistanceBetweenSignAndMagnitudeNumbers(
    const Bits& sam1,     // 第一个符号加绝对值数字
    const Bits& sam2) {   // 第二个符号加绝对值数字
  // 将符号加绝对值转换为偏置表示
  const Bits biased1 = SignAndMagnitudeToBiased(sam1);
  const Bits biased2 = SignAndMagnitudeToBiased(sam2);
  // 返回两个偏置值之间的差距
  return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
}

// 比较两个浮点数时，允许的最大 ULP's (Units in the Last Place) 数量。
// 数值越大，允许的误差越大。值为 0 表示两个数必须完全相同才认为它们相等。
//
// 单个浮点操作的最大误差为 0.5 个最后一位单位。在 Intel CPU 上，所有浮点计算都使用 80 位精度，
// 而 double 类型有 64 位精度。因此，对于普通用途来说，4 应该足够了。
//
// 更多关于 ULP 的细节参见以下文章：
// http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
static const size_t kMaxUlps = 4;

// 如果 lhs 和 rhs 之间的 ULP 数量不超过 kMaxUlps，则返回 true。具体而言，此函数：
//
//   - 如果其中一个数是 NAN，则返回 false。
//   - 将非常大的数视为接近无穷大。
//   - 将 +0.0 和 -0.0 视为 0 DLP（double last place）的距离。
inline bool AlmostEquals(float lhs, float rhs) {
  // IEEE 标准规定，任何涉及 NAN 的比较操作必须返回 false。
  if (std::isnan(lhs) || std::isnan(rhs))
    return false;

  // 使用联合体 Float 来获取浮点数的二进制表示
  Float l = {lhs};
  Float r = {rhs};

  // 检查符号加绝对值表示的距离是否不超过 kMaxUlps
  return DistanceBetweenSignAndMagnitudeNumbers(l.bits_, r.bits_) <= kMaxUlps;
}
```