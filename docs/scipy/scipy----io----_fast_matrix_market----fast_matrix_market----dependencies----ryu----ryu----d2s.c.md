# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\ryu\ryu\d2s.c`

```
/*
   版权所有 2018 年 Ulf Adams

   本文件内容可根据 Apache 许可证版本 2.0 使用。

   (参见随附的 LICENSE-Apache 文件或访问
    http://www.apache.org/licenses/LICENSE-2.0)

   或者，本文件内容可以根据 Boost 软件许可证版本 1.0 使用。

   (参见随附的 LICENSE-Boost 文件或访问
    https://www.boost.org/LICENSE_1_0.txt)

   除非适用法律要求或书面同意，本软件基于“原样”分发，无任何明示或
   含示的担保或条件。
*/

/*
   运行时编译器选项：
   -DRYU_DEBUG 生成详细的调试输出到标准输出。

   -DRYU_ONLY_64_BIT_OPS 避免使用 uint128_t 或 64 位内嵌函数。取决于编译器，可能较慢。

   -DRYU_OPTIMIZE_SIZE 使用更小的查找表。而不是存储每个所需的 5 的幂，只存储每 26 个条目，使用乘法计算中间值。
                        这样可以将查找表大小减小约 10 倍（只有一种情况，并且只有双倍），但会牺牲一些性能。
                        目前需要 MSVC 内嵌函数。
*/

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
#include "ryu/d2s_intrinsics.h"

// 根据模式包含小查找表或完整查找表。
#if defined(RYU_OPTIMIZE_SIZE)
#include "ryu/d2s_small_table.h"
#else
#include "ryu/d2s_full_table.h"
#endif

#define DOUBLE_MANTISSA_BITS 52
#define DOUBLE_EXPONENT_BITS 11
#define DOUBLE_BIAS 1023

// 内联函数，计算十进制长度为 17 的数值的函数。
static inline uint32_t decimalLength17(const uint64_t v) {
  // 这比循环略快。
  // 平均输出长度为 16.38 位数字，因此我们从高到低检查。
  // 函数前提条件：v 不是 18、19 或 20 位数字。
  // (17 位数字足以进行往返处理。)
  assert(v < 100000000000000000L);
  if (v >= 10000000000000000L) { return 17; }
  if (v >= 1000000000000000L) { return 16; }
  if (v >= 100000000000000L) { return 15; }
  if (v >= 10000000000000L) { return 14; }
  if (v >= 1000000000000L) { return 13; }
  if (v >= 100000000000L) { return 12; }
  if (v >= 10000000000L) { return 11; }
  if (v >= 1000000000L) { return 10; }
  if (v >= 100000000L) { return 9; }
  if (v >= 10000000L) { return 8; }
  if (v >= 1000000L) { return 7; }
  if (v >= 100000L) { return 6; }
  if (v >= 10000L) { return 5; }
  if (v >= 1000L) { return 4; }
  if (v >= 100L) { return 3; }
  if (v >= 10L) { return 2; }
  return 1;
}

// 表示 m * 10^e 的浮点十进制结构体。
typedef struct floating_decimal_64 {
  uint64_t mantissa;
  // 十进制指数范围是 -324 到 308，包含。
  // 如果需要，可以适合使用 short 类型。
  int32_t exponent;
} floating_decimal_64;
// 计算浮点数的二进制表示中的指数偏移量
static inline floating_decimal_64 d2d(const uint64_t ieeeMantissa, const uint32_t ieeeExponent) {
  int32_t e2;
  uint64_t m2;
  if (ieeeExponent == 0) {
    // 如果指数为零，计算偏移量和尾数
    // 通过减去 DOUBLE_BIAS 和 DOUBLE_MANTISSA_BITS 后再减 2，增加边界计算的精度
    e2 = 1 - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS - 2;
    m2 = ieeeMantissa;
  } else {
    // 否则，根据 IEEE 标准计算指数偏移量和尾数
    e2 = (int32_t) ieeeExponent - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS - 2;
    m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
  }
  // 判断尾数是否为偶数
  const bool even = (m2 & 1) == 0;
  // 决定是否接受边界值
  const bool acceptBounds = even;

#ifdef RYU_DEBUG
  // 调试信息：输出 m2 的值和调整后的指数
  printf("-> %" PRIu64 " * 2^%d\n", m2, e2 + 2);
#endif

  // 步骤 2：确定有效十进制表示的区间
  const uint64_t mv = 4 * m2;
  // 隐式的布尔值到整数转换。True 是 1，False 是 0。
  const uint32_t mmShift = ieeeMantissa != 0 || ieeeExponent <= 1;
  // 我们可以像这样计算 mp 和 mm：
  // uint64_t mp = 4 * m2 + 2;
  // uint64_t mm = mv - 1 - mmShift;

  // 步骤 3：使用 128 位算术将值转换为十进制幂次数基数
  uint64_t vr, vp, vm;
  int32_t e10;
  bool vmIsTrailingZeros = false;
  bool vrIsTrailingZeros = false;
  if (e2 >= 0) {
    // 对于正指数情况的处理
    // 计算对数的 10 的幂次数
    const uint32_t q = log10Pow2(e2) - (e2 > 3);
    e10 = (int32_t) q;
    // 计算一些常数
    const int32_t k = DOUBLE_POW5_INV_BITCOUNT + pow5bits((int32_t) q) - 1;
    const int32_t i = -e2 + (int32_t) q + k;
#if defined(RYU_OPTIMIZE_SIZE)
    // 计算 vr 使用的 128 位乘法与移位
    uint64_t pow5[2];
    double_computeInvPow5(q, pow5);
    vr = mulShiftAll64(m2, pow5, i, &vp, &vm, mmShift);
#else
    // 计算 vr 使用的 128 位乘法与移位
    vr = mulShiftAll64(m2, DOUBLE_POW5_INV_SPLIT[q], i, &vp, &vm, mmShift);
#endif
#ifdef RYU_DEBUG
    // 调试信息：输出计算过程中的一些中间结果
    printf("%" PRIu64 " * 2^%d / 10^%u\n", mv, e2, q);
    printf("V+=%" PRIu64 "\nV =%" PRIu64 "\nV-=%" PRIu64 "\n", vp, vr, vm);
#endif
    if (q <= 21) {
      // 如果幂次数小于等于 21，则计算尾数是否以 5 的倍数结尾
      const uint32_t mvMod5 = ((uint32_t) mv) - 5 * ((uint32_t) div5(mv));
      if (mvMod5 == 0) {
        vrIsTrailingZeros = multipleOfPowerOf5(mv, q);
      } else if (acceptBounds) {
        // 如果接受边界值，则判断尾数是否以 5 的倍数结尾
        vmIsTrailingZeros = multipleOfPowerOf5(mv - 1 - mmShift, q);
      } else {
        // 否则，减去 5 的倍数
        vp -= multipleOfPowerOf5(mv + 2, q);
      }
    }
  } else {
    // 对于负指数情况的处理
    const uint32_t q = log10Pow5(-e2) - (-e2 > 1);
    e10 = (int32_t) q + e2;
    const int32_t i = -e2 - (int32_t) q;
    const int32_t k = pow5bits(i) - DOUBLE_POW5_BITCOUNT;
    const int32_t j = (int32_t) q - k;
#if defined(RYU_OPTIMIZE_SIZE)
    // 计算 vr 使用的 128 位乘法与移位
    uint64_t pow5[2];
    double_computePow5(q, pow5);
    vr = mulShiftAll64(m2, pow5, i, &vp, &vm, mmShift);
#else
    // 计算 vr 使用的 128 位乘法与移位
    vr = mulShiftAll64(m2, DOUBLE_POW5_SPLIT[q], i, &vp, &vm, mmShift);
#endif
#ifdef RYU_DEBUG
    // 调试信息：输出计算过程中的一些中间结果
#endif
  }
}
    // 声明一个名为 pow5 的 uint64_t 类型数组，长度为 2
    uint64_t pow5[2];
    // 调用函数 double_computePow5，计算并存储 5 的幂次方在 pow5 数组中
    double_computePow5(i, pow5);
    // 调用 mulShiftAll64 函数，将 m2 与 pow5 数组中的值相乘并右移，结果存入 vr 中，
    // 同时计算并更新 vp、vm 的值，mmShift 是一个移位参数
    vr = mulShiftAll64(m2, pow5, j, &vp, &vm, mmShift);
#else
    // 在没有定义RYU_DEBUG时执行以下代码块
    vr = mulShiftAll64(m2, DOUBLE_POW5_SPLIT[i], j, &vp, &vm, mmShift);
#endif

#ifdef RYU_DEBUG
    // 如果定义了RYU_DEBUG，则打印以下调试信息：
    printf("%" PRIu64 " * 5^%d / 10^%u\n", mv, -e2, q);  // 打印mv乘以5的-e2次方再除以10的q次方
    printf("%u %d %d %d\n", q, i, k, j);  // 打印q、i、k、j的值
    printf("V+=%" PRIu64 "\nV =%" PRIu64 "\nV-=%" PRIu64 "\n", vp, vr, vm);  // 打印vp、vr、vm的值
#endif

if (q <= 1) {
    // 如果q小于等于1，则以下内容是关于vr、vp、vm是否是尾随零的判断：
    // 当mv至少有q个尾随0位时，{vr, vp, vm}也会有尾随零位。
    // mv = 4 * m2，所以它始终至少有两个尾随0位。
    vrIsTrailingZeros = true;
    if (acceptBounds) {
        // mm = mv - 1 - mmShift，所以当mmShift等于1时，它有1个尾随0位。
        vmIsTrailingZeros = mmShift == 1;
    } else {
        // mp = mv + 2，所以它始终至少有一个尾随0位。
        --vp;
    }
} else if (q < 63) { // TODO(ulfjack): 在这里使用更严格的界限。
    // 如果q小于63，则以下是判断vr是否有至少q个尾随零的逻辑：
    // 我们想知道完整的乘积是否至少有q个尾随零。
    // 我们需要计算min(p2(mv), p5(mv) - e2) >= q
    // <=> p2(mv) >= q && p5(mv) - e2 >= q
    // <=> p2(mv) >= q（因为-e2 >= q）
    vrIsTrailingZeros = multipleOfPowerOf2(mv, q);
#ifdef RYU_DEBUG
    printf("vr is trailing zeros=%s\n", vrIsTrailingZeros ? "true" : "false");
#endif
}

#ifdef RYU_DEBUG
// 打印额外的RYU_DEBUG信息：
printf("e10=%d\n", e10);  // 打印e10的值
printf("V+=%" PRIu64 "\nV =%" PRIu64 "\nV-=%" PRIu64 "\n", vp, vr, vm);  // 打印vp、vr、vm的值
printf("vm is trailing zeros=%s\n", vmIsTrailingZeros ? "true" : "false");  // 打印vm是否有尾随零位
printf("vr is trailing zeros=%s\n", vrIsTrailingZeros ? "true" : "false");  // 打印vr是否有尾随零位
#endif

// 第四步：在有效表示的区间内找到最短的十进制表示。
int32_t removed = 0;
uint8_t lastRemovedDigit = 0;
uint64_t output;

// 平均情况下，我们移除大约2个数字。
if (vmIsTrailingZeros || vrIsTrailingZeros) {
    // 一般情况，很少发生（约0.7%的情况）。
    for (;;) {
        const uint64_t vpDiv10 = div10(vp);
        const uint64_t vmDiv10 = div10(vm);
        if (vpDiv10 <= vmDiv10) {
            break;
        }
        const uint32_t vmMod10 = ((uint32_t) vm) - 10 * ((uint32_t) vmDiv10);
        const uint64_t vrDiv10 = div10(vr);
        const uint32_t vrMod10 = ((uint32_t) vr) - 10 * ((uint32_t) vrDiv10);
        vmIsTrailingZeros &= vmMod10 == 0;
        vrIsTrailingZeros &= lastRemovedDigit == 0;
        lastRemovedDigit = (uint8_t) vrMod10;
        vr = vrDiv10;
        vp = vpDiv10;
        vm = vmDiv10;
        ++removed;
    }
#ifdef RYU_DEBUG
    // 打印额外的RYU_DEBUG信息：
    printf("V+=%" PRIu64 "\nV =%" PRIu64 "\nV-=%" PRIu64 "\n", vp, vr, vm);  // 打印vp、vr、vm的值
    printf("d-10=%s\n", vmIsTrailingZeros ? "true" : "false");  // 打印vm是否有尾随零位
#endif
}
    // 如果 vm 是尾部的零
    if (vmIsTrailingZeros) {
      // 无限循环，直到条件不满足时退出
      for (;;) {
        // 将 vm 除以 10，获取商和余数
        const uint64_t vmDiv10 = div10(vm);
        const uint32_t vmMod10 = ((uint32_t) vm) - 10 * ((uint32_t) vmDiv10);
        // 如果 vm 的个位余数不为 0，则退出循环
        if (vmMod10 != 0) {
          break;
        }
        // 分别对 vp 和 vr 进行除以 10 操作
        const uint64_t vpDiv10 = div10(vp);
        const uint64_t vrDiv10 = div10(vr);
        // 计算 vr 的个位余数
        const uint32_t vrMod10 = ((uint32_t) vr) - 10 * ((uint32_t) vrDiv10);
        // 更新 vr 是否为尾部零的状态，并记录最后移除的数字
        vrIsTrailingZeros &= lastRemovedDigit == 0;
        lastRemovedDigit = (uint8_t) vrMod10;
        // 更新 vr、vp、vm 并增加移除计数
        vr = vrDiv10;
        vp = vpDiv10;
        vm = vmDiv10;
        ++removed;
      }
    }
#ifdef RYU_DEBUG
    // 如果定义了 RYU_DEBUG 宏，则输出 vr 和 lastRemovedDigit 的值
    printf("%" PRIu64 " %d\n", vr, lastRemovedDigit);
    // 输出 vrIsTrailingZeros 是否为 true
    printf("vr is trailing zeros=%s\n", vrIsTrailingZeros ? "true" : "false");
#endif

    // 如果 vrIsTrailingZeros 为 true，lastRemovedDigit 为 5，并且 vr 是偶数，则将 lastRemovedDigit 设为 4
    if (vrIsTrailingZeros && lastRemovedDigit == 5 && vr % 2 == 0) {
      // Round even if the exact number is .....50..0.
      lastRemovedDigit = 4;
    }

    // 如果 vr 处于边界之外或者需要进行四舍五入，则输出设为 vr + 1
    output = vr + ((vr == vm && (!acceptBounds || !vmIsTrailingZeros)) || lastRemovedDigit >= 5);
  } else {
    // 针对常见情况（约 99.3%），进行优化
    bool roundUp = false;
    // 将 vp 和 vm 分别除以 100
    const uint64_t vpDiv100 = div100(vp);
    const uint64_t vmDiv100 = div100(vm);

    // 如果 vpDiv100 大于 vmDiv100，则进行两位数字的优化移除（约 86.2%）
    if (vpDiv100 > vmDiv100) {
      const uint64_t vrDiv100 = div100(vr);
      const uint32_t vrMod100 = ((uint32_t) vr) - 100 * ((uint32_t) vrDiv100);
      roundUp = vrMod100 >= 50;
      vr = vrDiv100;
      vp = vpDiv100;
      vm = vmDiv100;
      removed += 2;
    }

    // 下面的循环迭代次数（大约），在无优化的情况下：
    // 0: 0.03%, 1: 13.8%, 2: 70.6%, 3: 14.0%, 4: 1.40%, 5: 0.14%, 6+: 0.02%
    // 在进行优化后：
    // 0: 70.6%, 1: 27.8%, 2: 1.40%, 3: 0.14%, 4+: 0.02%
    for (;;) {
      const uint64_t vpDiv10 = div10(vp);
      const uint64_t vmDiv10 = div10(vm);
      // 如果 vpDiv10 小于等于 vmDiv10，则跳出循环
      if (vpDiv10 <= vmDiv10) {
        break;
      }
      const uint64_t vrDiv10 = div10(vr);
      const uint32_t vrMod10 = ((uint32_t) vr) - 10 * ((uint32_t) vrDiv10);
      roundUp = vrMod10 >= 5;
      vr = vrDiv10;
      vp = vpDiv10;
      vm = vmDiv10;
      ++removed;
    }

#ifdef RYU_DEBUG
    // 如果定义了 RYU_DEBUG 宏，则输出 vr 和 roundUp 的值
    printf("%" PRIu64 " roundUp=%s\n", vr, roundUp ? "true" : "false");
    // 输出 vrIsTrailingZeros 是否为 true
    printf("vr is trailing zeros=%s\n", vrIsTrailingZeros ? "true" : "false");
#endif

    // 如果 vr 处于边界之外或者需要进行四舍五入，则输出设为 vr + 1
    output = vr + (vr == vm || roundUp);
  }

  // 计算 exp，即 e10 加上 removed
  const int32_t exp = e10 + removed;

#ifdef RYU_DEBUG
  // 如果定义了 RYU_DEBUG 宏，则输出 vp、vr、vm 的值以及 output 和 exp
  printf("V+=%" PRIu64 "\nV =%" PRIu64 "\nV-=%" PRIu64 "\n", vp, vr, vm);
  printf("O=%" PRIu64 "\n", output);
  printf("EXP=%d\n", exp);
#endif

  // 创建 floating_decimal_64 结构体并返回
  floating_decimal_64 fd;
  fd.exponent = exp;
  fd.mantissa = output;
  return fd;
}

// 将浮点数 v 转换为字符串，并存储在 result 中
static inline int to_chars(const floating_decimal_64 v, const bool sign, char* const result) {
  // Step 5: Print the decimal representation.
  int index = 0;
  // 如果有符号，将符号 '-' 存入 result
  if (sign) {
    result[index++] = '-';
  }

  // 将 v.mantissa 存入 output，并计算其长度 olength
  uint64_t output = v.mantissa;
  const uint32_t olength = decimalLength17(output);

#ifdef RYU_DEBUG
  // 如果定义了 RYU_DEBUG 宏，则输出 v.mantissa 和 olength 的值
  printf("DIGITS=%" PRIu64 "\n", v.mantissa);
  printf("OLEN=%u\n", olength);
  printf("EXP=%u\n", v.exponent + olength);
#endif
#endif

// 打印小数点后的数字。
// 下面的代码等价于：
// for (uint32_t i = 0; i < olength - 1; ++i) {
//   const uint32_t c = output % 10; output /= 10;
//   result[index + olength - i] = (char) ('0' + c);
// }
// result[index] = '0' + output % 10;
uint32_t i = 0;
// 我们倾向使用32位操作，即使在64位平台上。
// 我们最多有17位数字，而uint32_t可以存储9位数字。
// 如果output不适合uint32_t，我们会截断8位数字，
// 剩余的数字将适合于uint32_t。
if ((output >> 32) != 0) {
  // 昂贵的64位除法。
  const uint64_t q = div1e8(output);
  uint32_t output2 = ((uint32_t) output) - 100000000 * ((uint32_t) q);
  output = q;

  const uint32_t c = output2 % 10000;
  output2 /= 10000;
  const uint32_t d = output2 % 10000;
  const uint32_t c0 = (c % 100) << 1;
  const uint32_t c1 = (c / 100) << 1;
  const uint32_t d0 = (d % 100) << 1;
  const uint32_t d1 = (d / 100) << 1;
  // 将数字表中的字符复制到结果中的相应位置。
  memcpy(result + index + olength - i - 1, DIGIT_TABLE + c0, 2);
  memcpy(result + index + olength - i - 3, DIGIT_TABLE + c1, 2);
  memcpy(result + index + olength - i - 5, DIGIT_TABLE + d0, 2);
  memcpy(result + index + olength - i - 7, DIGIT_TABLE + d1, 2);
  i += 8;
}
uint32_t output2 = (uint32_t) output;
while (output2 >= 10000) {
#ifdef __clang__ // https://bugs.llvm.org/show_bug.cgi?id=38217
  const uint32_t c = output2 - 10000 * (output2 / 10000);
#else
  const uint32_t c = output2 % 10000;
#endif
  output2 /= 10000;
  const uint32_t c0 = (c % 100) << 1;
  const uint32_t c1 = (c / 100) << 1;
  // 将数字表中的字符复制到结果中的相应位置。
  memcpy(result + index + olength - i - 1, DIGIT_TABLE + c0, 2);
  memcpy(result + index + olength - i - 3, DIGIT_TABLE + c1, 2);
  i += 4;
}
if (output2 >= 100) {
  const uint32_t c = (output2 % 100) << 1;
  output2 /= 100;
  // 将数字表中的字符复制到结果中的相应位置。
  memcpy(result + index + olength - i - 1, DIGIT_TABLE + c, 2);
  i += 2;
}
if (output2 >= 10) {
  const uint32_t c = output2 << 1;
  // 这里不能使用memcpy：小数点位于这两个数字之间。
  result[index + olength - i] = DIGIT_TABLE[c + 1];
  result[index] = DIGIT_TABLE[c];
} else {
  result[index] = (char) ('0' + output2);
}

// 如果需要，打印小数点。
if (olength > 1) {
  result[index + 1] = '.';
  index += olength + 1;
} else {
  ++index;
}

// 打印指数部分。
result[index++] = 'E';
int32_t exp = v.exponent + (int32_t) olength - 1;
if (exp < 0) {
  result[index++] = '-';
  exp = -exp;
}

if (exp >= 100) {
  const int32_t c = exp % 10;
  // 将数字表中的字符复制到结果中的相应位置。
  memcpy(result + index, DIGIT_TABLE + 2 * (exp / 10), 2);
  result[index + 2] = (char) ('0' + c);
  index += 3;
} else if (exp >= 10) {
  // 将数字表中的字符复制到结果中的相应位置。
  memcpy(result + index, DIGIT_TABLE + 2 * exp, 2);
  index += 2;
} else {
  result[index++] = (char) ('0' + exp);
}

return index;
// 尝试将 IEEE 双精度浮点数转换为小整数形式，填充给定的 floating_decimal_64 结构体
static inline bool d2d_small_int(const uint64_t ieeeMantissa, const uint32_t ieeeExponent,
  floating_decimal_64* const v) {
  // 构造带隐含的整数部分的二进制表示 m2
  const uint64_t m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
  // 计算二进制指数的实际值 e2
  const int32_t e2 = (int32_t) ieeeExponent - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS;

  // 如果 e2 > 0，即 m2 * 2^e2 >= 2^53 是整数，这种情况先忽略
  if (e2 > 0) {
    // f = m2 * 2^e2 >= 2^53 是整数，暂不处理
    return false;
  }

  // 如果 e2 < -52，即 f < 1
  if (e2 < -52) {
    // f < 1，直接返回 false
    return false;
  }

  // 由于 2^52 <= m2 < 2^53，且 0 <= -e2 <= 52：
  // 这时 1 <= f = m2 / 2^-e2 < 2^53。
  // 测试尾部 -e2 位的小数部分是否为 0
  const uint64_t mask = (1ull << -e2) - 1;
  const uint64_t fraction = m2 & mask;
  if (fraction != 0) {
    // 如果小数部分不为 0，则返回 false
    return false;
  }

  // f 是范围 [1, 2^53) 内的整数
  // 注意：mantissa 可能包含尾部的十进制 0
  // 注意：由于 2^53 < 10^16，因此不需要调整 decimalLength17()。
  v->mantissa = m2 >> -e2; // 将整数部分赋给 mantissa
  v->exponent = 0; // 指数设为 0
  return true; // 返回 true，表示成功转换为小整数形式
}

// 将双精度浮点数 f 转换为字符串表示存入 result，返回结果字符串的长度
int d2s_buffered_n(double f, char* result) {
  // 步骤 1：解码浮点数，统一规范化和非规范化情况
  const uint64_t bits = double_to_bits(f);

#ifdef RYU_DEBUG
  printf("IN=");
  for (int32_t bit = 63; bit >= 0; --bit) {
    printf("%d", (int) ((bits >> bit) & 1));
  }
  printf("\n");
#endif

  // 将 bits 解码为符号、尾数和指数
  const bool ieeeSign = ((bits >> (DOUBLE_MANTISSA_BITS + DOUBLE_EXPONENT_BITS)) & 1) != 0;
  const uint64_t ieeeMantissa = bits & ((1ull << DOUBLE_MANTISSA_BITS) - 1);
  const uint32_t ieeeExponent = (uint32_t) ((bits >> DOUBLE_MANTISSA_BITS) & ((1u << DOUBLE_EXPONENT_BITS) - 1));

  // 情况区分；对简单情况进行提前退出处理
  if (ieeeExponent == ((1u << DOUBLE_EXPONENT_BITS) - 1u) || (ieeeExponent == 0 && ieeeMantissa == 0)) {
    // 处理特殊值，直接复制相应字符串到 result 中
    return copy_special_str(result, ieeeSign, ieeeExponent, ieeeMantissa);
  }

  floating_decimal_64 v;
  const bool isSmallInt = d2d_small_int(ieeeMantissa, ieeeExponent, &v);
  if (isSmallInt) {
    // 对于范围 [1, 2^53) 内的小整数，可能含有尾部十进制 0，
    // 对于科学计数法，需要将这些零移动到指数中。
    // （对于固定点表示法，如果需要的话，可能有益于在 to_chars 中仅修剪尾部零 - 一旦实现了固定点表示法输出。）
    for (;;) {
      const uint64_t q = div10(v.mantissa);
      const uint32_t r = ((uint32_t) v.mantissa) - 10 * ((uint32_t) q);
      if (r != 0) {
        break;
      }
      v.mantissa = q;
      ++v.exponent;
    }
  } else {
    // 对于大整数，调用 d2d 函数转换为十进制表示
    v = d2d(ieeeMantissa, ieeeExponent);
  }

  // 调用 to_chars 将 v 转换为字符串并存入 result，返回字符串长度
  return to_chars(v, ieeeSign, result);
}

// 将双精度浮点数 f 转换为字符串表示存入 result
void d2s_buffered(double f, char* result) {
  // 调用 d2s_buffered_n 完成转换，并将结果字符串终止符设为 '\0'
  const int index = d2s_buffered_n(f, result);
  result[index] = '\0';
}

// 分配空间并将双精度浮点数 f 转换为字符串返回
char* d2s(double f) {
  // 分配 25 字节空间作为结果字符串
  char* const result = (char*) malloc(25);
  // 调用 d2s_buffered 进行转换
  d2s_buffered(f, result);
  return result; // 返回结果字符串指针
}
```