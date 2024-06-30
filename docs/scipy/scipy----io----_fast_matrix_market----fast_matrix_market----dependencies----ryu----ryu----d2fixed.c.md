# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\ryu\ryu\d2fixed.c`

```
/*
   Copyright 2018 Ulf Adams

   The contents of this file may be used under the terms of the Apache License,
   Version 2.0.

   (See accompanying file LICENSE-Apache or copy at
    http://www.apache.org/licenses/LICENSE-2.0)

   Alternatively, the contents of this file may be used under the terms of
   the Boost Software License, Version 1.0.

   (See accompanying file LICENSE-Boost or copy at
    https://www.boost.org/LICENSE_1_0.txt)

   Unless required by applicable law or agreed to in writing, this software
   is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
*/

// Runtime compiler options:
// -DRYU_DEBUG Generate verbose debugging output to stdout.
//
// -DRYU_ONLY_64_BIT_OPS Avoid using uint128_t or 64-bit intrinsics. Slower,
//     depending on your compiler.
//
// -DRYU_AVOID_UINT128 Avoid using uint128_t. Slower, depending on your compiler.

#include "ryu/ryu.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef RYU_DEBUG
#include <inttypes.h>
#include <stdio.h>
#endif

#include "ryu/common.h"
#include "ryu/digit_table.h"
#include "ryu/d2fixed_full_table.h"
#include "ryu/d2s_intrinsics.h"

// Definitions of constants related to double precision floating point numbers
#define DOUBLE_MANTISSA_BITS 52
#define DOUBLE_EXPONENT_BITS 11
#define DOUBLE_BIAS 1023

// Additional bits used for computing powers of 10
#define POW10_ADDITIONAL_BITS 120

#if defined(HAS_UINT128)
// Function to multiply a 128-bit integer by a 64-bit integer and obtain the low 128 bits
static inline uint128_t umul256(const uint128_t a, const uint64_t bHi, const uint64_t bLo, uint128_t* const productHi) {
  const uint64_t aLo = (uint64_t)a;
  const uint64_t aHi = (uint64_t)(a >> 64);

  const uint128_t b00 = (uint128_t)aLo * bLo;
  const uint128_t b01 = (uint128_t)aLo * bHi;
  const uint128_t b10 = (uint128_t)aHi * bLo;
  const uint128_t b11 = (uint128_t)aHi * bHi;

  const uint64_t b00Lo = (uint64_t)b00;
  const uint64_t b00Hi = (uint64_t)(b00 >> 64);

  const uint128_t mid1 = b10 + b00Hi;
  const uint64_t mid1Lo = (uint64_t)(mid1);
  const uint64_t mid1Hi = (uint64_t)(mid1 >> 64);

  const uint128_t mid2 = b01 + mid1Lo;
  const uint64_t mid2Lo = (uint64_t)(mid2);
  const uint64_t mid2Hi = (uint64_t)(mid2 >> 64);

  const uint128_t pHi = b11 + mid1Hi + mid2Hi;
  const uint128_t pLo = ((uint128_t)mid2Lo << 64) | b00Lo;

  *productHi = pHi;
  return pLo;
}

// Function to return the high 128 bits of the 256-bit product of a and b
static inline uint128_t umul256_hi(const uint128_t a, const uint64_t bHi, const uint64_t bLo) {
  // Reuse the umul256 implementation.
  // Optimizers will likely eliminate the instructions used to compute the
  // low part of the product.
  uint128_t hi;
  umul256(a, bHi, bLo, &hi);
  return hi;
}
#endif

// Unfortunately, gcc/clang do not automatically turn a 128-bit integer division
// into a multiplication, so we have to do it manually.


These annotations provide a detailed explanation of each significant part of the C header file, clarifying the purpose and context of various constants, macros, and functions defined within.
// 使用内联函数定义，计算给定的 128 位整数 v 对 10^9 取模后的结果
static inline uint32_t uint128_mod1e9(const uint128_t v) {
  // 将 128 位整数 v 与常数进行乘法运算，然后向右移动 29 位，并截断为 uint32_t 类型。
  // 这意味着我们只需要使用 61 位（29 + 32），因此在移位之前可以先截断为 uint64_t 类型。
  const uint64_t multiplied = (uint64_t) umul256_hi(v, 0x89705F4136B4A597u, 0x31680A88F8953031u);

  // 根据 d2s_intrinsics.h 中 mod1e9() 的注释，执行 uint32_t 类型的截断。
  const uint32_t shifted = (uint32_t) (multiplied >> 29);

  // 返回将原始的 uint32_t 类型的 v 减去 10^9 乘以 shifted 的结果。
  return ((uint32_t) v) - 1000000000 * shifted;
}

// 最佳情况：使用 128 位整数类型。
static inline uint32_t mulShift_mod1e9(const uint64_t m, const uint64_t* const mul, const int32_t j) {
  const uint128_t b0 = ((uint128_t) m) * mul[0]; // 0
  const uint128_t b1 = ((uint128_t) m) * mul[1]; // 64
  const uint128_t b2 = ((uint128_t) m) * mul[2]; // 128

#ifdef RYU_DEBUG
  // 调试模式下，检查 j 是否在指定的范围内。
  if (j < 128 || j > 180) {
    printf("%d\n", j);
  }
#endif

  // 断言 j 的取值范围在 [128, 180] 之间。
  assert(j >= 128);
  assert(j <= 180);

  // j 的取值范围：[128, 256)
  const uint128_t mid = b1 + (uint64_t) (b0 >> 64); // 64
  const uint128_t s1 = b2 + (uint64_t) (mid >> 64); // 128

  // 返回 s1 右移 (j - 128) 位后的结果，再对 10^9 取模。
  return uint128_mod1e9(s1 >> (j - 128));
}
static inline uint32_t mulShift_mod1e9(const uint64_t m, const uint64_t* const mul, const int32_t j) {
    uint64_t high0;                                   // 64位整数变量，用于存储umul128计算的高位结果
    const uint64_t low0 = umul128(m, mul[0], &high0); // 计算m和mul[0]的乘积，并得到低位结果和高位结果

    uint64_t high1;                                   // 64位整数变量，用于存储umul128计算的高位结果
    const uint64_t low1 = umul128(m, mul[1], &high1); // 计算m和mul[1]的乘积，并得到低位结果和高位结果

    uint64_t high2;                                   // 64位整数变量，用于存储umul128计算的高位结果
    const uint64_t low2 = umul128(m, mul[2], &high2); // 计算m和mul[2]的乘积，并得到低位结果和高位结果

    const uint64_t s0low = low0;                       // 将low0赋给s0low，用于后续计算
    (void) s0low; // unused                             // 占位符，未使用，用于消除编译器警告

    const uint64_t s0high = low1 + high0;               // 计算s0high为low1加上high0的结果
    const uint32_t c1 = s0high < low1;                  // 若s0high小于low1，设置c1为1，否则为0

    const uint64_t s1low = low2 + high1 + c1;           // 计算s1low为low2加上high1再加上c1的结果
    const uint32_t c2 = s1low < low2;                   // 若s1low小于low2，设置c2为1，否则为0

    const uint64_t s1high = high2 + c2;                 // 计算s1high为high2加上c2的结果

#ifdef RYU_DEBUG
    if (j < 128 || j > 180) {
        printf("%d\n", j);                             // 在调试模式下，如果j不在范围[128, 180]内，则打印j的值
    }
#endif

    assert(j >= 128);                                   // 断言，确保j大于等于128
    assert(j <= 180);                                   // 断言，确保j小于等于180

#if defined(HAS_64_BIT_INTRINSICS)
    const uint32_t dist = (uint32_t) (j - 128);         // 计算距离dist，即j减去128
    const uint64_t shiftedhigh = s1high >> dist;        // 将s1high向右移动dist位
    const uint64_t shiftedlow = shiftright128(s1low, s1high, dist); // 调用shiftright128函数，将s1low和s1high向右移动dist位
    return uint128_mod1e9(shiftedhigh, shiftedlow);     // 调用uint128_mod1e9函数，对shiftedhigh和shiftedlow进行模运算后返回结果
#else // HAS_64_BIT_INTRINSICS
    if (j < 160) {                                      // 若j小于160
        const uint64_t r0 = mod1e9(s1high);             // 对s1high进行模运算，并赋给r0
        const uint64_t r1 = mod1e9((r0 << 32) | (s1low >> 32)); // 对r0左移32位后与(s1low右移32位)按位或，并对结果进行模运算，并赋给r1
        const uint64_t r2 = ((r1 << 32) | (s1low & 0xffffffff)); // 对r1左移32位后与(s1low的低32位)按位或，并赋给r2
        return mod1e9(r2 >> (j - 128));                 // 对r2右移(j-128)位后进行模运算，并返回结果
    } else {                                             // 若j大于等于160
        const uint64_t r0 = mod1e9(s1high);             // 对s1high进行模运算，并赋给r0
        const uint64_t r1 = ((r0 << 32) | (s1low >> 32)); // 对r0左移32位后与(s1low右移32位)按位或，并赋给r1
        return mod1e9(r1 >> (j - 160));                 // 对r1右移(j-160)位后进行模运算，并返回结果
    }
#endif // HAS_64_BIT_INTRINSICS
}
#endif // HAS_UINT128

// Convert `digits` to a sequence of decimal digits. Append the digits to the result.
// The caller has to guarantee that:
//   10^(olength-1) <= digits < 10^olength
// e.g., by passing `olength` as `decimalLength9(digits)`.
static inline void append_n_digits(const uint32_t olength, uint32_t digits, char* const result) {
#ifdef RYU_DEBUG
    printf("DIGITS=%u\n", digits);                      // 在调试模式下，打印digits的值
#endif

    uint32_t i = 0;                                     // 初始化i为0
    while (digits >= 10000) {                           // 循环直到digits小于10000
#ifdef __clang__ // https://bugs.llvm.org/show_bug.cgi?id=38217
        const uint32_t c = digits - 10000 * (digits / 10000); // 计算c为digits减去10000乘以(digits除以10000)的结果
#else
        const uint32_t c = digits % 10000;              // 计算c为digits对10000取模的结果
#endif
        digits /= 10000;                                // 将digits除以10000，并更新digits的值
        const uint32_t c0 = (c % 100) << 1;             // 计算c0为(c对100取模)左移1位的结果
        const uint32_t c1 = (c / 100) << 1;             // 计算c1为(c除以100)左移1位的结果
        memcpy(result + olength - i - 2, DIGIT_TABLE + c0, 2); // 将DIGIT_TABLE中c0偏移的2个字节复制到result中
        memcpy(result + olength - i - 4, DIGIT_TABLE + c1, 2); // 将DIGIT_TABLE中c1偏移的2个字节复制到result中
        i += 4;                                         // 更新i的值加4
    }
    if (digits >= 100) {                                // 如果digits大于等于100
        const uint32_t c = (digits % 100) << 1;         // 计算c为(digits对100取模)左移1位的结果
        digits /= 100;                                  // 将digits除以100，并更新digits的值
        memcpy(result + olength - i - 2, DIGIT_TABLE + c, 2); // 将DIGIT_TABLE中c偏移的2个字节复制到result中
        i += 2;                                         // 更新i的值加2
    }
    if (digits >= 10) {                                 // 如果digits大于等于10
        const uint32_t c = digits << 1;                 // 计算c为digits左移1位的结果
        memcpy(result + olength - i - 2, DIGIT_TABLE + c, 2); // 将DIGIT_TABLE中c偏移的2个字节复制到result中
    } else {
        result[0] = (char) ('0' + digits);              // 将result的第一个字符设置为'0'加上digits的值对应的字符
    }
}
// Append decimal digits to the result string, with a specified length olength.
// The digits parameter represents the number whose decimal digits are to be appended.
static inline void append_d_digits(const uint32_t olength, uint32_t digits, char* const result) {
#ifdef RYU_DEBUG
  printf("DIGITS=%u\n", digits);
#endif

  uint32_t i = 0;
  // Loop through the digits, converting them into pairs of decimal digits and appending to the result.
  while (digits >= 10000) {
#ifdef __clang__ // Workaround for a specific bug in Clang compiler.
    const uint32_t c = digits - 10000 * (digits / 10000);
#else
    const uint32_t c = digits % 10000;
#endif
    digits /= 10000;
    const uint32_t c0 = (c % 100) << 1;  // Compute offset for first digit in DIGIT_TABLE
    const uint32_t c1 = (c / 100) << 1;  // Compute offset for second digit in DIGIT_TABLE
    // Copy the two digits from DIGIT_TABLE into the result buffer at appropriate positions.
    memcpy(result + olength + 1 - i - 2, DIGIT_TABLE + c0, 2);
    memcpy(result + olength + 1 - i - 4, DIGIT_TABLE + c1, 2);
    i += 4;  // Move the index by 4 positions in the result buffer.
  }
  // Handle remaining digits that are less than 10000.
  if (digits >= 100) {
    const uint32_t c = (digits % 100) << 1;
    digits /= 100;
    memcpy(result + olength + 1 - i - 2, DIGIT_TABLE + c, 2);
    i += 2;
  }
  // Handle the last digit if it's less than 10.
  if (digits >= 10) {
    const uint32_t c = digits << 1;
    result[2] = DIGIT_TABLE[c + 1];  // Store the second digit
    result[1] = '.';                // Place decimal point
    result[0] = DIGIT_TABLE[c];     // Store the first digit
  } else {
    result[1] = '.';                // Place decimal point
    result[0] = (char) ('0' + digits);  // Store the single digit
  }
}

// Append `count` decimal digits of `digits` to the result string.
// Any excess digits in `digits` are ignored.
static inline void append_c_digits(const uint32_t count, uint32_t digits, char* const result) {
#ifdef RYU_DEBUG
  printf("DIGITS=%u\n", digits);
#endif
  // Copy pairs of digits from DIGIT_TABLE until `count` digits are appended to `result`.
  uint32_t i = 0;
  for (; i < count - 1; i += 2) {
    const uint32_t c = (digits % 100) << 1;
    digits /= 100;
    memcpy(result + count - i - 2, DIGIT_TABLE + c, 2);
  }
  // If `count` is odd, append the last digit.
  if (i < count) {
    const char c = (char) ('0' + (digits % 10));
    result[count - i - 1] = c;
  }
}

// Append the last 9 decimal digits of `digits` to the result string.
// If `digits` has fewer than 9 digits, prepend with leading zeros.
static inline void append_nine_digits(uint32_t digits, char* const result) {
#ifdef RYU_DEBUG
  printf("DIGITS=%u\n", digits);
#endif
  // If digits is zero, directly append '000000000' to the result.
  if (digits == 0) {
    memset(result, '0', 9);
    return;
  }

  // Loop through `digits`, converting into pairs of decimal digits and appending to `result`.
  for (uint32_t i = 0; i < 5; i += 4) {
#ifdef __clang__ // Workaround for a specific bug in Clang compiler.
    const uint32_t c = digits - 10000 * (digits / 10000);
#else
    const uint32_t c = digits % 10000;
#endif
    digits /= 10000;
    const uint32_t c0 = (c % 100) << 1;  // Compute offset for first digit in DIGIT_TABLE
    const uint32_t c1 = (c / 100) << 1;  // Compute offset for second digit in DIGIT_TABLE
    // Copy the two digits from DIGIT_TABLE into the result buffer at appropriate positions.
    memcpy(result + 7 - i, DIGIT_TABLE + c0, 2);
    memcpy(result + 5 - i, DIGIT_TABLE + c1, 2);
  }
  result[0] = (char) ('0' + digits);  // Store the last digit
}

// Calculate the index for exponent `e` in the context of `RYU_DEBUG`.
static inline uint32_t indexForExponent(const uint32_t e) {
  return (e + 15) / 16;  // Calculate the index based on the given formula.
}

// Calculate the number of bits necessary for `idx` to represent the power of 10.
static inline uint32_t pow10BitsForIndex(const uint32_t idx) {
  return 16 * idx + POW10_ADDITIONAL_BITS;  // Compute the bits required for the given index.
}
// 计算给定索引下的字符串长度
static inline uint32_t lengthForIndex(const uint32_t idx) {
  // 对索引乘以16，并取对数的2为底的对数，加1为向上取整，再加上16为尾数长度，再加8为除以9时的向上取整
  return (log10Pow2(16 * (int32_t) idx) + 1 + 16 + 8) / 9;
}

// 将特殊的浮点数转换为字符串，通过printf函数输出
static inline int copy_special_str_printf(char* const result, const bool sign, const uint64_t mantissa) {
#if defined(_MSC_VER)
  // 在 Windows 平台上检查是否期望输出为 -nan
  if (sign) {
    result[0] = '-';
  }
  // 如果存在尾数
  if (mantissa) {
    // 如果尾数小于双精度浮点数尾数位的一半
    if (mantissa < (1ull << (DOUBLE_MANTISSA_BITS - 1))) {
      // 复制 "nan(snan)" 到结果字符串，并返回结果字符串长度
      memcpy(result + sign, "nan(snan)", 9);
      return sign + 9;
    }
    // 复制 "nan" 到结果字符串，并返回结果字符串长度
    memcpy(result + sign, "nan", 3);
    return sign + 3;
  }
#else
  // 如果存在尾数
  if (mantissa) {
    // 复制 "nan" 到结果字符串，并返回结果字符串长度
    memcpy(result, "nan", 3);
    return 3;
  }
  // 如果有符号位，将符号位复制到结果字符串
  if (sign) {
    result[0] = '-';
  }
#endif
  // 复制 "Infinity" 到结果字符串，并返回结果字符串长度
  memcpy(result + sign, "Infinity", 8);
  return sign + 8;
}

// 将双精度浮点数转换为固定格式的字符串，并输出到指定的结果缓冲区
int d2fixed_buffered_n(double d, uint32_t precision, char* result) {
  // 将双精度浮点数转换为位表示
  const uint64_t bits = double_to_bits(d);
#ifdef RYU_DEBUG
  // 在调试模式下输出双精度浮点数的位表示
  printf("IN=");
  for (int32_t bit = 63; bit >= 0; --bit) {
    printf("%d", (int) ((bits >> bit) & 1));
  }
  printf("\n");
#endif

  // 解析位表示，获取符号位、尾数和指数
  const bool ieeeSign = ((bits >> (DOUBLE_MANTISSA_BITS + DOUBLE_EXPONENT_BITS)) & 1) != 0;
  const uint64_t ieeeMantissa = bits & ((1ull << DOUBLE_MANTISSA_BITS) - 1);
  const uint32_t ieeeExponent = (uint32_t) ((bits >> DOUBLE_MANTISSA_BITS) & ((1u << DOUBLE_EXPONENT_BITS) - 1));

  // 对于特殊情况，直接返回相应的字符串表示
  if (ieeeExponent == ((1u << DOUBLE_EXPONENT_BITS) - 1u)) {
    return copy_special_str_printf(result, ieeeSign, ieeeMantissa);
  }
  if (ieeeExponent == 0 && ieeeMantissa == 0) {
    int index = 0;
    // 如果是零，根据符号位设置结果字符串的第一个字符
    if (ieeeSign) {
      result[index++] = '-';
    }
    // 添加 '0' 到结果字符串
    result[index++] = '0';
    // 如果有小数精度要求，添加小数点和相应位数的 '0'
    if (precision > 0) {
      result[index++] = '.';
      memset(result + index, '0', precision);
      index += precision;
    }
    return index;
  }

  int32_t e2;
  uint64_t m2;
  // 根据指数值分类处理
  if (ieeeExponent == 0) {
    e2 = 1 - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS;
    m2 = ieeeMantissa;
  } else {
    e2 = (int32_t) ieeeExponent - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS;
    m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
  }

#ifdef RYU_DEBUG
  // 在调试模式下输出调整后的尾数和指数
  printf("-> %" PRIu64 " * 2^%d\n", m2, e2);
#endif

  int index = 0;
  bool nonzero = false;
  // 如果有符号位，将符号位添加到结果字符串
  if (ieeeSign) {
    result[index++] = '-';
  }
  // 如果指数 e2 >= -52，根据指数获取相应索引和相关计算的位数
  if (e2 >= -52) {
    const uint32_t idx = e2 < 0 ? 0 : indexForExponent((uint32_t) e2);
    const uint32_t p10bits = pow10BitsForIndex(idx);
    const int32_t len = (int32_t) lengthForIndex(idx);
#ifdef RYU_DEBUG
    // 在调试模式下输出索引和长度信息
    printf("idx=%u\n", idx);
    printf("len=%d\n", len);
#endif
    // 从最后一个位置开始向前遍历数组，直到第一个位置为止
    for (int32_t i = len - 1; i >= 0; --i) {
      // 计算 j 值，j 是 p10bits 减去 e2 的结果
      const uint32_t j = p10bits - e2;
      // 使用 mulShift_mod1e9 函数计算 digits，这里 m2 左移 8 位后作为参数之一传入
      const uint32_t digits = mulShift_mod1e9(m2 << 8, POW10_SPLIT[POW10_OFFSET[idx] + i], (int32_t) (j + 8));
      // 如果 nonzero 为真，将 digits 追加到 result 的 index 处，并增加 index 9
      if (nonzero) {
        append_nine_digits(digits, result + index);
        index += 9;
      } else if (digits != 0) {
        // 如果 nonzero 为假且 digits 不为零，计算 digits 的十进制长度 olength
        const uint32_t olength = decimalLength9(digits);
        // 将 olength 长度的 digits 追加到 result 的 index 处，并增加 index 为 olength
        append_n_digits(olength, digits, result + index);
        index += olength;
        // 将 nonzero 设置为真，表示已经开始记录非零的数字
        nonzero = true;
      }
    }
  }
  // 如果所有数字均为零，则将 '0' 添加到 result 的 index 处，并增加 index 1
  if (!nonzero) {
    result[index++] = '0';
  }
  // 如果 precision 大于 0，则在 result 的 index 处添加小数点，并增加 index 1
  if (precision > 0) {
    result[index++] = '.';
  }
#ifdef RYU_DEBUG
  printf("e2=%d\n", e2);
#endif
// 如果指定的指数 e2 小于 0，则执行以下代码块
if (e2 < 0) {
    // 计算负指数 e2 的绝对值除以 16，得到索引 idx
    const int32_t idx = -e2 / 16;
#ifdef RYU_DEBUG
    printf("idx=%d\n", idx);
#endif
    // 计算要处理的数字的块数，每个块占9个十进制数字
    const uint32_t blocks = precision / 9 + 1;
    // roundUp 表示舍入方式：0 = 不舍入；1 = 无条件舍入；2 = 如果为奇数则舍入
    int roundUp = 0;
    uint32_t i = 0;
    // 如果块数小于等于预定义的 MIN_BLOCK_2[idx]，则执行以下代码块
    if (blocks <= MIN_BLOCK_2[idx]) {
        i = blocks;
        // 在结果字符串中填充 precision 个 '0' 字符
        memset(result + index, '0', precision);
        index += precision;
    } else if (i < MIN_BLOCK_2[idx]) {
        // 否则，如果 i 小于 MIN_BLOCK_2[idx]，则执行以下代码块
        i = MIN_BLOCK_2[idx];
        // 在结果字符串中填充 9 * i 个 '0' 字符
        memset(result + index, '0', 9 * i);
        index += 9 * i;
    }
    // 循环处理每个块
    for (; i < blocks; ++i) {
        // 计算要处理的位数 j，通过 ADDITIONAL_BITS_2 和 (-e2 - 16 * idx) 计算得到
        const int32_t j = ADDITIONAL_BITS_2 + (-e2 - 16 * idx);
        // 计算当前块的偏移量 p
        const uint32_t p = POW10_OFFSET_2[idx] + i - MIN_BLOCK_2[idx];
        // 如果偏移量 p 大于等于下一个块的偏移量 POW10_OFFSET_2[idx + 1]，则执行以下代码块
        if (p >= POW10_OFFSET_2[idx + 1]) {
            // 如果剩余的数字都是 0，则使用 memset 进行填充，无需舍入
            const uint32_t fill = precision - 9 * i;
            memset(result + index, '0', fill);
            index += fill;
            break;
        }
        // 计算乘法结果 digits，调用 mulShift_mod1e9 函数
        uint32_t digits = mulShift_mod1e9(m2 << 8, POW10_SPLIT_2[p], j + 8);
#ifdef RYU_DEBUG
        printf("digits=%u\n", digits);
#endif
        // 如果不是最后一个块，则追加九个数字到结果字符串
        if (i < blocks - 1) {
            append_nine_digits(digits, result + index);
            index += 9;
        } else {
            // 否则，计算最后一个块的最大位数
            const uint32_t maximum = precision - 9 * i;
            uint32_t lastDigit = 0;
            // 迭代处理最后一个块的位数，计算最后一个数字 lastDigit
            for (uint32_t k = 0; k < 9 - maximum; ++k) {
                lastDigit = digits % 10;
                digits /= 10;
            }
#ifdef RYU_DEBUG
            printf("lastDigit=%u\n", lastDigit);
#endif
            // 如果最后一个数字不是 5，则根据其大小决定是否需要舍入
            if (lastDigit != 5) {
                roundUp = lastDigit > 5;
            } else {
                // 否则，根据尾部零的情况决定是否需要舍入
                const int32_t requiredTwos = -e2 - (int32_t) precision - 1;
                const bool trailingZeros = requiredTwos <= 0
                    || (requiredTwos < 60 && multipleOfPowerOf2(m2, (uint32_t) requiredTwos));
                roundUp = trailingZeros ? 2 : 1;
#ifdef RYU_DEBUG
                printf("requiredTwos=%d\n", requiredTwos);
                printf("trailingZeros=%s\n", trailingZeros ? "true" : "false");
#endif
            }
            // 如果有剩余的最大位数，则追加到结果字符串中
            if (maximum > 0) {
                append_c_digits(maximum, digits, result + index);
                index += maximum;
            }
            break;
        }
    }
#ifdef RYU_DEBUG
    printf("roundUp=%d\n", roundUp);
#endif
    # 如果需要进行四舍五入操作
    if (roundUp != 0) {
      # 记录当前需要处理的索引位置
      int roundIndex = index;
      # 初始化小数点索引为0，因为小数点不能位于索引0处
      int dotIndex = 0;
      # 进入循环，处理四舍五入逻辑
      while (true) {
        # 将索引向前移动一位
        --roundIndex;
        char c;
        # 如果已经到达起始位置或者遇到负号，表明需要向前进位
        if (roundIndex == -1 || (c = result[roundIndex], c == '-')) {
          result[roundIndex + 1] = '1';
          # 如果存在小数点，将其置为0并在后面添加新的小数点
          if (dotIndex > 0) {
            result[dotIndex] = '0';
            result[dotIndex + 1] = '.';
          }
          # 在当前索引位置添加0
          result[index++] = '0';
          # 跳出循环
          break;
        }
        # 如果遇到小数点，更新小数点的索引位置
        if (c == '.') {
          dotIndex = roundIndex;
          continue;
        } else if (c == '9') {
          # 如果遇到数字9，将其置为0，并进位
          result[roundIndex] = '0';
          roundUp = 1;
          continue;
        } else {
          # 如果当前位数字小于9，直接加1
          if (roundUp == 2 && c % 2 == 0) {
            break;
          }
          result[roundIndex] = c + 1;
          break;
        }
      }
    }
  } else {
    # 如果不需要四舍五入，直接在结果中填充0
    memset(result + index, '0', precision);
    index += precision;
  }
  # 返回处理后的结果索引
  return index;
}

// 将双精度浮点数转换为固定精度字符串表示，存储在 result 中
void d2fixed_buffered(double d, uint32_t precision, char* result) {
  // 调用 d2fixed_buffered_n 函数计算字符串长度并将结果存储在 result 中
  const int len = d2fixed_buffered_n(d, precision, result);
  // 在字符串末尾添加 null 终止符
  result[len] = '\0';
}

// 将双精度浮点数转换为固定精度字符串表示，返回分配的内存地址指针
char* d2fixed(double d, uint32_t precision) {
  // 分配至少 2000 字节大小的缓冲区
  char* const buffer = (char*)malloc(2000);
  // 调用 d2fixed_buffered_n 函数计算字符串长度并将结果存储在 buffer 中
  const int index = d2fixed_buffered_n(d, precision, buffer);
  // 在字符串末尾添加 null 终止符
  buffer[index] = '\0';
  // 返回分配的缓冲区地址
  return buffer;
}

// 计算双精度浮点数的科学计数法表示，并存储在 result 中，返回结果长度
int d2exp_buffered_n(double d, uint32_t precision, char* result) {
  // 将双精度浮点数转换为整数表示
  const uint64_t bits = double_to_bits(d);
#ifdef RYU_DEBUG
  // 调试模式下输出转换前的二进制表示
  printf("IN=");
  for (int32_t bit = 63; bit >= 0; --bit) {
    printf("%d", (int) ((bits >> bit) & 1));
  }
  printf("\n");
#endif

  // 解码二进制表示为符号、尾数和指数
  const bool ieeeSign = ((bits >> (DOUBLE_MANTISSA_BITS + DOUBLE_EXPONENT_BITS)) & 1) != 0;
  const uint64_t ieeeMantissa = bits & ((1ull << DOUBLE_MANTISSA_BITS) - 1);
  const uint32_t ieeeExponent = (uint32_t) ((bits >> DOUBLE_MANTISSA_BITS) & ((1u << DOUBLE_EXPONENT_BITS) - 1));

  // 简单情况的退出条件
  if (ieeeExponent == ((1u << DOUBLE_EXPONENT_BITS) - 1u)) {
    // 处理特殊值情况（Infinity 和 NaN），并将结果存储在 result 中
    return copy_special_str_printf(result, ieeeSign, ieeeMantissa);
  }
  if (ieeeExponent == 0 && ieeeMantissa == 0) {
    // 处理零值情况，并将结果存储在 result 中
    int index = 0;
    if (ieeeSign) {
      result[index++] = '-';
    }
    result[index++] = '0';
    if (precision > 0) {
      result[index++] = '.';
      memset(result + index, '0', precision);
      index += precision;
    }
    memcpy(result + index, "e+00", 4);
    index += 4;
    return index;
  }

  int32_t e2;
  uint64_t m2;
  if (ieeeExponent == 0) {
    // 非规格化数的情况
    e2 = 1 - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS;
    m2 = ieeeMantissa;
  } else {
    // 规格化数的情况
    e2 = (int32_t) ieeeExponent - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS;
    m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
  }

#ifdef RYU_DEBUG
  // 调试模式下输出转换后的十进制表示
  printf("-> %" PRIu64 " * 2^%d\n", m2, e2);
#endif

  // 根据精度是否大于零，决定是否输出小数点
  const bool printDecimalPoint = precision > 0;
  ++precision;
  int index = 0;
  if (ieeeSign) {
    result[index++] = '-';
  }
  uint32_t digits = 0;
  uint32_t printedDigits = 0;
  uint32_t availableDigits = 0;
  int32_t exp = 0;
  if (e2 >= -52) {
    // 计算指数对应的索引和相关位数
    const uint32_t idx = e2 < 0 ? 0 : indexForExponent((uint32_t) e2);
    const uint32_t p10bits = pow10BitsForIndex(idx);
    const int32_t len = (int32_t) lengthForIndex(idx);
#ifdef RYU_DEBUG
    // 调试模式下输出索引和相关长度
    printf("idx=%u\n", idx);
    printf("len=%d\n", len);
#endif
    // 从后向前遍历数组，i 从 len-1 到 0 递减
    for (int32_t i = len - 1; i >= 0; --i) {
      // 计算 j，通常情况下约为 128，通过位移操作，将其推到 128 或以上，这在 mulShift_mod1e9 函数中会有略微更快的代码路径。
      // 实际上，可以通过增加乘数来代替位移操作。
      const uint32_t j = p10bits - e2;
      // 使用 mulShift_mod1e9 函数计算乘积 digits
      digits = mulShift_mod1e9(m2 << 8, POW10_SPLIT[POW10_OFFSET[idx] + i], (int32_t) (j + 8));
      // 如果已打印的数字不为零
      if (printedDigits != 0) {
        // 如果加上九位数字超过了精度要求，则设定可用数字为 9，并且跳出循环
        if (printedDigits + 9 > precision) {
          availableDigits = 9;
          break;
        }
        // 将九位数字追加到结果中，更新索引和已打印数字计数
        append_nine_digits(digits, result + index);
        index += 9;
        printedDigits += 9;
      } else if (digits != 0) {
        // 计算可用数字的位数
        availableDigits = decimalLength9(digits);
        // 计算指数 exp，更新已打印数字计数
        exp = i * 9 + (int32_t) availableDigits - 1;
        // 如果可用数字超过精度要求，则跳出循环
        if (availableDigits > precision) {
          break;
        }
        // 如果需要打印小数点，则将数字追加到结果中，并更新索引（+1 是为了小数点）
        if (printDecimalPoint) {
          append_d_digits(availableDigits, digits, result + index);
          index += availableDigits + 1; // +1 用于小数点
        } else {
          // 否则直接将数字字符加入结果中
          result[index++] = (char) ('0' + digits);
        }
        // 更新已打印数字计数和可用数字为零
        printedDigits = availableDigits;
        availableDigits = 0;
      }
    }
  }

  // 如果 e2 小于 0 并且可用数字为零
  if (e2 < 0 && availableDigits == 0) {
    // 计算索引 idx
    const int32_t idx = -e2 / 16;
#ifdef RYU_DEBUG
    // 如果定义了 RYU_DEBUG 宏，则输出调试信息，显示当前索引 idx、e2 的值以及 MIN_BLOCK_2[idx] 的最小块索引值
    printf("idx=%d, e2=%d, min=%d\n", idx, e2, MIN_BLOCK_2[idx]);
#endif
    // 对于每个 i 从 MIN_BLOCK_2[idx] 开始，直到 200 结束的循环
    for (int32_t i = MIN_BLOCK_2[idx]; i < 200; ++i) {
      // 计算 j 的值，通常约为 128，通过位移将其推到 128 或以上，这在 mulShift_mod1e9 中会有稍微更快的代码路径
      const int32_t j = ADDITIONAL_BITS_2 + (-e2 - 16 * idx);
      // 计算 p 的值，作为 POW10_OFFSET_2[idx] + i - MIN_BLOCK_2[idx] 的结果
      const uint32_t p = POW10_OFFSET_2[idx] + (uint32_t) i - MIN_BLOCK_2[idx];
      // 根据 p 的值选择是否执行 mulShift_mod1e9，并将结果存储在 digits 中
      // 如果 p 大于等于下一个索引处的 POW10_OFFSET_2[idx + 1]，则将 digits 设为 0
      digits = (p >= POW10_OFFSET_2[idx + 1]) ? 0 : mulShift_mod1e9(m2 << 8, POW10_SPLIT_2[p], j + 8);
#ifdef RYU_DEBUG
      // 如果定义了 RYU_DEBUG 宏，则输出调试信息，显示 m2 乘以 POW10_SPLIT_2[p] 的计算过程
      printf("exact=%" PRIu64 " * (%" PRIu64 " + %" PRIu64 " << 64) >> %d\n", m2, POW10_SPLIT_2[p][0], POW10_SPLIT_2[p][1], j);
      // 如果定义了 RYU_DEBUG 宏，则输出 digits 的值
      printf("digits=%u\n", digits);
#endif
      // 如果已经打印的数字不为 0
      if (printedDigits != 0) {
        // 如果已打印的数字加 9 大于 precision，则设定可用数字为 9 并跳出循环
        if (printedDigits + 9 > precision) {
          availableDigits = 9;
          break;
        }
        // 将 digits 中的九位数字追加到结果字符串中的索引位置，并更新索引和已打印的数字计数
        append_nine_digits(digits, result + index);
        index += 9;
        printedDigits += 9;
      } else if (digits != 0) {
        // 否则，如果 digits 不为 0
        // 计算可用的十进制位数，并计算指数 exp
        availableDigits = decimalLength9(digits);
        exp = -(i + 1) * 9 + (int32_t) availableDigits - 1;
        // 如果可用的位数大于 precision，则跳出循环
        if (availableDigits > precision) {
          break;
        }
        // 如果需要打印小数点，则将 digits 追加到结果字符串中的索引位置，并更新索引，包括额外的一个位置给小数点
        if (printDecimalPoint) {
          append_d_digits(availableDigits, digits, result + index);
          index += availableDigits + 1; // +1 是为了小数点
        } else {
          // 否则，直接将 digits 转换为字符追加到结果字符串中的索引位置
          result[index++] = (char) ('0' + digits);
        }
        // 更新已打印的数字和可用的数字
        printedDigits = availableDigits;
        availableDigits = 0;
      }
    }
  }

  // 计算最大可能的数字，即 precision 减去已打印的数字
  const uint32_t maximum = precision - printedDigits;
#ifdef RYU_DEBUG
  // 如果定义了 RYU_DEBUG 宏，则输出调试信息，显示可用的位数、digits 的值以及最大可能的数字
  printf("availableDigits=%u\n", availableDigits);
  printf("digits=%u\n", digits);
  printf("maximum=%u\n", maximum);
#endif
  // 如果可用的位数为 0，则将 digits 设为 0
  if (availableDigits == 0) {
    digits = 0;
  }
  uint32_t lastDigit = 0;
  // 如果可用的位数大于最大可能的数字
  if (availableDigits > maximum) {
    // 计算并获取 digits 的最后一位数字，并将 digits 除以 10
    for (uint32_t k = 0; k < availableDigits - maximum; ++k) {
      lastDigit = digits % 10;
      digits /= 10;
    }
  }
#ifdef RYU_DEBUG
  // 如果定义了 RYU_DEBUG 宏，则输出调试信息，显示最后一位数字 lastDigit 的值
  printf("lastDigit=%u\n", lastDigit);
#endif
  // roundUp 变量的含义：0 表示不进行四舍五入；1 表示无条件向上舍入；2 表示在奇数时进行向上舍入
  int roundUp = 0;
  // 如果最后一位数字不为 5
  if (lastDigit != 5) {
    // 如果最后一位数字大于 5，则设置 roundUp 为 1
    roundUp = lastDigit > 5;
  } else {
    // 否则，执行以下操作判断是否需要进行向上舍入
    // 计算 rexp，并获取 requiredTwos 的值
    const int32_t rexp = (int32_t) precision - exp;
    const int32_t requiredTwos = -e2 - rexp;
    // 判断是否有末尾的零
    bool trailingZeros = requiredTwos <= 0
      || (requiredTwos < 60 && multipleOfPowerOf2(m2, (uint32_t) requiredTwos));
    // 如果 rexp 小于 0，则计算 requiredFives，并继续判断是否有末尾的零
    if (rexp < 0) {
      const int32_t requiredFives = -rexp;
      trailingZeros = trailingZeros && multipleOfPowerOf5(m2, (uint32_t) requiredFives);
    }
    // 根据 trailingZeros 的情况，设置 roundUp 的值，如果有末尾的零则设为 2，否则为 1
    roundUp = trailingZeros ? 2 : 1;
#ifdef RYU_DEBUG
    // 如果定义了 RYU_DEBUG 宏，则输出调试信息，显示 requiredTwos 的值和 trailingZeros 是否为 true
    printf("requiredTwos=%d\n", requiredTwos);
    printf("trailingZeros=%s\n", trailingZeros ? "true" : "false");
#endif
  }
  // 如果已打印的数字不为 0，则继续执行以下操作
  if (printedDigits != 0) {
    // 如果 digits 等于 0，则在结果数组 result 的 index 位置开始，连续填充 maximum 个 '0'
    if (digits == 0) {
      memset(result + index, '0', maximum);
    } else {
      // 否则调用函数 append_c_digits，将 maximum 和 digits 作为参数传递，并在结果数组 result 的 index 位置开始追加字符
      append_c_digits(maximum, digits, result + index);
    }
    // 更新 index，使其指向结果数组中的下一个位置
    index += maximum;
  } else {
    // 如果不是 digits 为 0 的情况
    if (printDecimalPoint) {
      // 如果需要打印小数点，则调用函数 append_d_digits，将 maximum 和 digits 作为参数传递，并在结果数组 result 的 index 位置开始追加字符，同时增加 index 以留出位置给小数点
      append_d_digits(maximum, digits, result + index);
      index += maximum + 1; // +1 用于小数点
    } else {
      // 如果不需要打印小数点，则将字符 '0' 加入结果数组的 index 位置，并增加 index
      result[index++] = (char) ('0' + digits);
    }
  }
#ifdef RYU_DEBUG
  // 如果定义了 RYU_DEBUG 宏，则打印 roundUp 的值
  printf("roundUp=%d\n", roundUp);
#endif
  // 如果 roundUp 不为 0
  if (roundUp != 0) {
    // 保存当前 index 的值到 roundIndex
    int roundIndex = index;
    // 循环直到中断
    while (true) {
      // 将 roundIndex 减一
      --roundIndex;
      // 声明字符变量 c
      char c;
      // 如果 roundIndex 为 -1 或者 c 为 result[roundIndex] 并且 c 为 '-'
      if (roundIndex == -1 || (c = result[roundIndex], c == '-')) {
        // 将 result[roundIndex + 1] 设置为 '1'
        result[roundIndex + 1] = '1';
        // exp 加一
        ++exp;
        // 中断循环
        break;
      }
      // 如果 c 为 '.'
      if (c == '.') {
        // 继续下一次循环
        continue;
      } else if (c == '9') {
        // 如果 c 为 '9'，将 result[roundIndex] 设置为 '0'
        result[roundIndex] = '0';
        // roundUp 设置为 1
        roundUp = 1;
        // 继续下一次循环
        continue;
      } else {
        // 如果 roundUp 为 2 并且 c 为偶数
        if (roundUp == 2 && c % 2 == 0) {
          // 中断循环
          break;
        }
        // 将 result[roundIndex] 设置为 c + 1
        result[roundIndex] = c + 1;
        // 中断循环
        break;
      }
    }
  }
  // 将 'e' 添加到 result[index] 处
  result[index++] = 'e';
  // 如果 exp 小于 0
  if (exp < 0) {
    // 将 '-' 添加到 result[index] 处
    result[index++] = '-';
    // exp 取绝对值
    exp = -exp;
  } else {
    // 将 '+' 添加到 result[index] 处
    result[index++] = '+';
  }

  // 如果 exp 大于等于 100
  if (exp >= 100) {
    // 取 exp 的个位数
    const int32_t c = exp % 10;
    // 将 DIGIT_TABLE + 2 * (exp / 10) 的值复制到 result[index] 处，长度为 2
    memcpy(result + index, DIGIT_TABLE + 2 * (exp / 10), 2);
    // 将 result[index + 2] 设置为 '0' + c
    result[index + 2] = (char) ('0' + c);
    // index 加 3
    index += 3;
  } else {
    // 将 DIGIT_TABLE + 2 * exp 的值复制到 result[index] 处，长度为 2
    memcpy(result + index, DIGIT_TABLE + 2 * exp, 2);
    // index 加 2
    index += 2;
  }

  // 返回 index 的值
  return index;
}

// 将 double 类型的 d 转换为指定精度的科学计数法字符串，存放到 result 中
void d2exp_buffered(double d, uint32_t precision, char* result) {
  // 调用 d2exp_buffered_n 函数，将返回的长度保存到 len
  const int len = d2exp_buffered_n(d, precision, result);
  // 在 result[len] 处添加终止符 '\0'
  result[len] = '\0';
}

// 将 double 类型的 d 转换为指定精度的科学计数法字符串，返回动态分配的字符串指针
char* d2exp(double d, uint32_t precision) {
  // 分配 2000 字节的内存给 buffer
  char* const buffer = (char*)malloc(2000);
  // 调用 d2exp_buffered_n 函数，将返回的长度保存到 index
  const int index = d2exp_buffered_n(d, precision, buffer);
  // 在 buffer[index] 处添加终止符 '\0'
  buffer[index] = '\0';
  // 返回 buffer 指针
  return buffer;
}
```