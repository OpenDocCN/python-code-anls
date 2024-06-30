# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\ryu\ryu\f2s.c`

```
// 版权声明和许可条款声明
//
// 本文件的内容可以根据 Apache 许可证 2.0 版本使用。
// (请参阅附带的 LICENSE-Apache 文件或访问 http://www.apache.org/licenses/LICENSE-2.0)
//
// 或者，本文件的内容可以根据 Boost 软件许可证 1.0 版本使用。
// (请参阅附带的 LICENSE-Boost 文件或访问 https://www.boost.org/LICENSE_1_0.txt)
//
// 除非适用法律要求或书面同意，本软件基于 "如现有基础" 分发，不提供任何明示或隐含的担保或条件。
//
// 运行时编译选项：
// -DRYU_DEBUG 生成详细的调试输出到 stdout。

#include "ryu/ryu.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifdef RYU_DEBUG
#include <stdio.h>
#endif

#include "ryu/common.h"
#include "ryu/f2s_intrinsics.h"
#include "ryu/digit_table.h"

#define FLOAT_MANTISSA_BITS 23
#define FLOAT_EXPONENT_BITS 8
#define FLOAT_BIAS 127

// 表示 m * 10^e 的浮点十进制结构
typedef struct floating_decimal_32 {
  uint32_t mantissa; // 尾数部分
  int32_t exponent;  // 十进制指数，范围为 -45 到 38
} floating_decimal_32;

// 将 IEEE 754 单精度浮点数转换为十进制浮点表示
static inline floating_decimal_32 f2d(const uint32_t ieeeMantissa, const uint32_t ieeeExponent) {
  int32_t e2;
  uint32_t m2;
  
  if (ieeeExponent == 0) {
    // 对于规格化浮点数，重新计算指数和尾数
    e2 = 1 - FLOAT_BIAS - FLOAT_MANTISSA_BITS - 2;
    m2 = ieeeMantissa;
  } else {
    // 对于非规格化浮点数，重新计算指数和尾数
    e2 = (int32_t) ieeeExponent - FLOAT_BIAS - FLOAT_MANTISSA_BITS - 2;
    m2 = (1u << FLOAT_MANTISSA_BITS) | ieeeMantissa;
  }
  
  const bool even = (m2 & 1) == 0;
  const bool acceptBounds = even;

#ifdef RYU_DEBUG
  // 如果定义了 RYU_DEBUG 宏，则输出调试信息到 stdout
  printf("-> %u * 2^%d\n", m2, e2 + 2);
#endif

  // 步骤 2: 确定有效十进制表示的区间
  const uint32_t mv = 4 * m2;
  const uint32_t mp = 4 * m2 + 2;
  // 隐式的布尔值转换。真为 1，假为 0。
  const uint32_t mmShift = ieeeMantissa != 0 || ieeeExponent <= 1;
  const uint32_t mm = 4 * m2 - 1 - mmShift;

  // 步骤 3: 使用 64 位算术将结果转换为十进制的幂次表示
  uint32_t vr, vp, vm;
  int32_t e10;
  bool vmIsTrailingZeros = false;
  bool vrIsTrailingZeros = false;
  uint8_t lastRemovedDigit = 0;

  if (e2 >= 0) {
    // 计算 10 的幂次
    const uint32_t q = log10Pow2(e2);
    e10 = (int32_t) q;
    const int32_t k = FLOAT_POW5_INV_BITCOUNT + pow5bits((int32_t) q) - 1;
    const int32_t i = -e2 + (int32_t) q + k;
    vr = mulPow5InvDivPow2(mv, q, i);
    vp = mulPow5InvDivPow2(mp, q, i);
    vm = mulPow5InvDivPow2(mm, q, i);

#ifdef RYU_DEBUG
    // 如果定义了 RYU_DEBUG 宏，则输出调试信息到 stdout
    printf("%u * 2^%d / 10^%u\n", mv, e2, q);
    printf("V+=%u\nV =%u\nV-=%u\n", vp, vr, vm);
#endif
    // 如果 q 不等于 0 且 (vp - 1) / 10 小于等于 vm / 10，则执行以下操作
    if (q != 0 && (vp - 1) / 10 <= vm / 10) {
      // 即使我们不会在下面的循环中使用，我们仍然需要知道一个移除的数字。
      // 我们可以在上面使用 q = X - 1，但是那将需要 33 位来存储结果，
      // 我们发现即使在 64 位机器上，32 位算术也更快。
      const int32_t l = FLOAT_POW5_INV_BITCOUNT + pow5bits((int32_t) (q - 1)) - 1;
      // 记录最后移除的数字
      lastRemovedDigit = (uint8_t) (mulPow5InvDivPow2(mv, q - 1, -e2 + (int32_t) q - 1 + l) % 10);
    }
    // 如果 q 小于等于 9，则执行以下操作
    if (q <= 9) {
      // 在 24 位内能容纳的最大 5 的幂是 5^10，但 q <= 9 似乎也是安全的。
      // mp、mv、mm 中只有一个可能是 5 的倍数（multiple of 5）。
      if (mv % 5 == 0) {
        // 判断 mv 是否为 5 的倍数，如果是，则 vr 是尾随零
        vrIsTrailingZeros = multipleOfPowerOf5_32(mv, q);
      } else if (acceptBounds) {
        // 如果 acceptBounds 为真，则判断 mm 是否为 5 的倍数
        vmIsTrailingZeros = multipleOfPowerOf5_32(mm, q);
      } else {
        // 否则，减去 mp 中的 5 的倍数
        vp -= multipleOfPowerOf5_32(mp, q);
      }
    }
  } else {
    // 如果不满足上述条件，则执行以下操作
    const uint32_t q = log10Pow5(-e2);
    // 计算 e10
    e10 = (int32_t) q + e2;
    const int32_t i = -e2 - (int32_t) q;
    const int32_t k = pow5bits(i) - FLOAT_POW5_BITCOUNT;
    // 计算 vr、vp、vm
    int32_t j = (int32_t) q - k;
    vr = mulPow5divPow2(mv, (uint32_t) i, j);
    vp = mulPow5divPow2(mp, (uint32_t) i, j);
    vm = mulPow5divPow2(mm, (uint32_t) i, j);
#ifdef RYU_DEBUG
    // 如果定义了 RYU_DEBUG 宏，则输出调试信息：
    // 输出一个格式化的调试信息，包括 mv（整数）、-e2（整数）、q（整数）
    printf("%u * 5^%d / 10^%u\n", mv, -e2, q);
    // 输出四个整数：q、i、k、j
    printf("%u %d %d %d\n", q, i, k, j);
    // 输出两个整数（vp 和 vr）、一个整数（vm）
    printf("V+=%u\nV =%u\nV-=%u\n", vp, vr, vm);
#endif
    // 如果 q 不等于 0 并且 (vp - 1) / 10 小于等于 vm / 10
    if (q != 0 && (vp - 1) / 10 <= vm / 10) {
      // 计算 j 的值，该值用于指示舍入过程中的最后一个移除的数字
      j = (int32_t) q - 1 - (pow5bits(i + 1) - FLOAT_POW5_BITCOUNT);
      // 计算并记录最后一个移除的数字
      lastRemovedDigit = (uint8_t) (mulPow5divPow2(mv, (uint32_t) (i + 1), j) % 10);
    }
    // 如果 q 小于等于 1
    if (q <= 1) {
      // 如果 acceptBounds 为真，则判断 mmShift 是否为 1 来确定 vm 是否有尾随零位
      // 否则，减小 vp 的值
      vrIsTrailingZeros = true;
      if (acceptBounds) {
        vmIsTrailingZeros = mmShift == 1; // mmShift 为 1 时，vm 有一个尾随零位
      } else {
        --vp; // 减小 vp 的值，因为 mp = mv + 2，所以总有至少一个尾随零位
      }
    } else if (q < 31) { // 当 q 小于 31 时
      // 判断 vr 是否以 2^q-1 结尾的多个 0
      vrIsTrailingZeros = multipleOfPowerOf2_32(mv, q - 1);
#ifdef RYU_DEBUG
      // 如果定义了 RYU_DEBUG 宏，则输出 vr 是否以尾随零位结尾
      printf("vr is trailing zeros=%s\n", vrIsTrailingZeros ? "true" : "false");
#endif
    }
  }
#ifdef RYU_DEBUG
  // 如果定义了 RYU_DEBUG 宏，则输出 e10 的值
  printf("e10=%d\n", e10);
  // 输出两个整数（vp 和 vr）、一个整数（vm）
  printf("V+=%u\nV =%u\nV-=%u\n", vp, vr, vm);
  // 输出 vm 是否有尾随零位的信息
  printf("vm is trailing zeros=%s\n", vmIsTrailingZeros ? "true" : "false");
  // 输出 vr 是否有尾随零位的信息
  printf("vr is trailing zeros=%s\n", vrIsTrailingZeros ? "true" : "false");
#endif

  // 步骤 4：在有效表示的间隔中找到最短的十进制表示
  int32_t removed = 0;
  uint32_t output;
  // 如果 vm 或 vr 有尾随零位
  if (vmIsTrailingZeros || vrIsTrailingZeros) {
    // 通常情况，很少发生（约 4.0% 的概率）
    while (vp / 10 > vm / 10) {
#ifdef __clang__ // https://bugs.llvm.org/show_bug.cgi?id=23106
      // 编译器未意识到 vm % 10 可以由 vm / 10 计算得出
      // 因此，检查 vm 的个位数是否为 0
      vmIsTrailingZeros &= vm - (vm / 10) * 10 == 0;
#else
      // 检查 vm 的个位数是否为 0
      vmIsTrailingZeros &= vm % 10 == 0;
#endif
      // 检查 lastRemovedDigit 是否为 0
      vrIsTrailingZeros &= lastRemovedDigit == 0;
      // 记录当前 vr 的个位数
      lastRemovedDigit = (uint8_t) (vr % 10);
      // 缩小 vp、vr 和 vm 的值
      vr /= 10;
      vp /= 10;
      vm /= 10;
      ++removed; // 记录移除的位数
    }
#ifdef RYU_DEBUG
    // 如果定义了 RYU_DEBUG 宏，则输出缩小后的 vp、vr 和 vm 的值
    printf("V+=%u\nV =%u\nV-=%u\n", vp, vr, vm);
    // 输出 vm 是否有尾随零位的信息
    printf("d-10=%s\n", vmIsTrailingZeros ? "true" : "false");
#endif
    // 如果 vm 有尾随零位
    if (vmIsTrailingZeros) {
      // 进一步检查 vm 的个位数是否为 0
      while (vm % 10 == 0) {
        // 检查 lastRemovedDigit 是否为 0
        vrIsTrailingZeros &= lastRemovedDigit == 0;
        // 记录当前 vr 的个位数
        lastRemovedDigit = (uint8_t) (vr % 10);
        // 缩小 vp、vr 和 vm 的值
        vr /= 10;
        vp /= 10;
        vm /= 10;
        ++removed; // 记录移除的位数
      }
    }
#ifdef RYU_DEBUG
    // 如果定义了 RYU_DEBUG 宏，则输出 vr 的值和 lastRemovedDigit 的值
    printf("%u %d\n", vr, lastRemovedDigit);
    // 输出 vr 是否有尾随零位的信息
    printf("vr is trailing zeros=%s\n", vrIsTrailingZeros ? "true" : "false");
#endif
    // 如果 vr 有尾随零位，并且 lastRemovedDigit 为 5 且 vr 为偶数
    if (vrIsTrailingZeros && lastRemovedDigit == 5 && vr % 2 == 0) {
      // 如果 vr 需要向上舍入，则将 lastRemovedDigit 设为 4
      lastRemovedDigit = 4;
    }
    // 如果 vr 处于边界之外或需要向上舍入，则输出 vr + 1
    output = vr + ((vr == vm && (!acceptBounds || !vmIsTrailingZeros)) || lastRemovedDigit >= 5);
  } else {
    // 当vp和vm除以10后，只要vp仍然大于vm，就执行循环。这是针对通常情况（约96.0%）进行优化的特定处理。
    // 循环迭代次数的相对百分比如下：
    // 0次：13.6%，1次：70.7%，2次：14.1%，3次：1.39%，4次：0.14%，5次以上：0.01%
    while (vp / 10 > vm / 10) {
      // 记录被移除的最后一位数字
      lastRemovedDigit = (uint8_t) (vr % 10);
      // 除以10，更新vr、vp、vm的值
      vr /= 10;
      vp /= 10;
      vm /= 10;
      // 计数已经移除的位数
      ++removed;
    }
#ifdef RYU_DEBUG
    // 打印 vr 和 lastRemovedDigit 的值
    printf("%u %d\n", vr, lastRemovedDigit);
    // 打印 vr 是否为尾随零
    printf("vr is trailing zeros=%s\n", vrIsTrailingZeros ? "true" : "false");
#endif
    // 如果 vr 超出边界或需要向上舍入，则输出为 vr + 1
    output = vr + (vr == vm || lastRemovedDigit >= 5);
  }
  // 计算指数值 exp，包括 e10 和 removed 的和
  const int32_t exp = e10 + removed;

#ifdef RYU_DEBUG
  // 打印 vp, vr, vm 的值
  printf("V+=%u\nV =%u\nV-=%u\n", vp, vr, vm);
  // 打印输出值 output
  printf("O=%u\n", output);
  // 打印指数 exp 的值
  printf("EXP=%d\n", exp);
#endif

  // 创建浮点数结构体 fd，设置其指数为 exp，尾数为 output
  floating_decimal_32 fd;
  fd.exponent = exp;
  fd.mantissa = output;
  // 返回浮点数结构体 fd
  return fd;
}

static inline int to_chars(const floating_decimal_32 v, const bool sign, char* const result) {
  // Step 5: Print the decimal representation.
  int index = 0;
  // 如果有符号，添加负号到结果中
  if (sign) {
    result[index++] = '-';
  }

  // 将输出值设置为 v 的尾数
  uint32_t output = v.mantissa;
  // 计算输出值的十进制长度
  const uint32_t olength = decimalLength9(output);

#ifdef RYU_DEBUG
  // 打印输出值的数字
  printf("DIGITS=%u\n", v.mantissa);
  // 打印输出值的长度
  printf("OLEN=%u\n", olength);
  // 打印输出值的指数
  printf("EXP=%u\n", v.exponent + olength);
#endif

  // 打印十进制数字
  // 以下代码等同于：
  // for (uint32_t i = 0; i < olength - 1; ++i) {
  //   const uint32_t c = output % 10; output /= 10;
  //   result[index + olength - i] = (char) ('0' + c);
  // }
  // result[index] = '0' + output % 10;
  uint32_t i = 0;
  while (output >= 10000) {
#ifdef __clang__ // https://bugs.llvm.org/show_bug.cgi?id=38217
    // 使用 Clang 的特定处理方式来取得 output 的最后四位
    const uint32_t c = output - 10000 * (output / 10000);
#else
    // 取得 output 的最后四位
    const uint32_t c = output % 10000;
#endif
    output /= 10000;
    // 计算 c 的偏移量并复制到结果中
    const uint32_t c0 = (c % 100) << 1;
    const uint32_t c1 = (c / 100) << 1;
    memcpy(result + index + olength - i - 1, DIGIT_TABLE + c0, 2);
    memcpy(result + index + olength - i - 3, DIGIT_TABLE + c1, 2);
    i += 4;
  }
  // 如果 output 大于等于 100，则处理其余数字
  if (output >= 100) {
    const uint32_t c = (output % 100) << 1;
    output /= 100;
    memcpy(result + index + olength - i - 1, DIGIT_TABLE + c, 2);
    i += 2;
  }
  // 如果 output 大于等于 10，则处理余下的数字
  if (output >= 10) {
    const uint32_t c = output << 1;
    // 在这里不能使用 memcpy：十进制点放置在这两个数字之间
    result[index + olength - i] = DIGIT_TABLE[c + 1];
    result[index] = DIGIT_TABLE[c];
  } else {
    result[index] = (char) ('0' + output);
  }

  // 如果需要，打印十进制点
  if (olength > 1) {
    result[index + 1] = '.';
    index += olength + 1;
  } else {
    ++index;
  }

  // 打印指数
  result[index++] = 'E';
  int32_t exp = v.exponent + (int32_t) olength - 1;
  // 如果指数为负数，添加负号到结果中
  if (exp < 0) {
    result[index++] = '-';
    exp = -exp;
  }

  // 如果指数大于等于 10，则添加其余数字到结果中
  if (exp >= 10) {
    memcpy(result + index, DIGIT_TABLE + 2 * exp, 2);
    index += 2;
  } else {
    result[index++] = (char) ('0' + exp);
  }

  // 返回结果的长度
  return index;
}
#endif

  // 将浮点数的位表示解码为符号、尾数和指数。
  const bool ieeeSign = ((bits >> (FLOAT_MANTISSA_BITS + FLOAT_EXPONENT_BITS)) & 1) != 0;
  // 从位表示中提取尾数部分
  const uint32_t ieeeMantissa = bits & ((1u << FLOAT_MANTISSA_BITS) - 1);
  // 从位表示中提取指数部分
  const uint32_t ieeeExponent = (bits >> FLOAT_MANTISSA_BITS) & ((1u << FLOAT_EXPONENT_BITS) - 1);

  // 区分不同的情况，针对简单情况进行早期退出。
  if (ieeeExponent == ((1u << FLOAT_EXPONENT_BITS) - 1u) || (ieeeExponent == 0 && ieeeMantissa == 0)) {
    // 处理特殊情况，直接生成特殊浮点数字符串并返回
    return copy_special_str(result, ieeeSign, ieeeExponent, ieeeMantissa);
  }

  // 将解码后的浮点数转换为十进制表示
  const floating_decimal_32 v = f2d(ieeeMantissa, ieeeExponent);
  // 将十进制表示转换为字符数组，并返回结果
  return to_chars(v, ieeeSign, result);
}

// 将浮点数转换为字符串并存储在给定的字符数组中
void f2s_buffered(float f, char* result) {
  // 调用 f2s_buffered_n 函数获取转换后字符串的长度
  const int index = f2s_buffered_n(f, result);

  // 在字符数组末尾添加字符串结束符 '\0'
  result[index] = '\0';
}

// 将浮点数转换为字符串，并返回动态分配的结果字符串指针
char* f2s(float f) {
  // 分配一个包含 16 字节空间的字符数组作为结果存储
  char* const result = (char*) malloc(16);
  // 调用 f2s_buffered 函数将浮点数转换为字符串并存储在 result 中
  f2s_buffered(f, result);
  // 返回动态分配的结果字符串指针
  return result;
}
```