# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\ryu\ryu\d2s_intrinsics.h`

```
// 定义条件编译保护，防止重复包含头文件
#ifndef RYU_D2S_INTRINSICS_H
#define RYU_D2S_INTRINSICS_H

// 包含必要的标准库头文件
#include <assert.h>
#include <stdint.h>

// 根据具体平台定义 RYU_32_BIT_PLATFORM 宏
#include "ryu/common.h"

// 根据不同条件定义 128 位整数类型的别名 uint128_t
#if defined(__SIZEOF_INT128__) && !defined(_MSC_VER) && !defined(RYU_ONLY_64_BIT_OPS)
#define HAS_UINT128
#elif defined(_MSC_VER) && !defined(RYU_ONLY_64_BIT_OPS) && defined(_M_X64)
#define HAS_64_BIT_INTRINSICS
#endif

// 如果支持 128 位整数，则定义 uint128_t 类型别名
#if defined(HAS_UINT128)
typedef __uint128_t uint128_t;
#endif

// 如果支持 64 位平台特定的内联函数和宏定义
#if defined(HAS_64_BIT_INTRINSICS)

// 包含系统特定的头文件，如 <intrin.h>
#include <intrin.h>

// 定义计算两个 64 位整数乘积并获取高 64 位的函数 umul128
static inline uint64_t umul128(const uint64_t a, const uint64_t b, uint64_t* const productHi) {
  return _umul128(a, b, productHi);
}

// 定义右移 128 位整数的低 64 位的函数 shiftright128，以指定的位数 dist 进行右移
// 要求 0 < dist < 64
static inline uint64_t shiftright128(const uint64_t lo, const uint64_t hi, const uint32_t dist) {
  // 使用系统提供的 __shiftright128 内建函数进行右移操作，位移量 dist 必须在模 64 的范围内
  // 断言确保位移量 dist 小于 64
  assert(dist < 64);
  return __shiftright128(lo, hi, (unsigned char) dist);
}

#else // defined(HAS_64_BIT_INTRINSICS)
// 定义一个内联函数，计算两个64位无符号整数的乘积，并返回高64位，通过指针参数返回高位结果
static inline uint64_t umul128(const uint64_t a, const uint64_t b, uint64_t* const productHi) {
  // 将64位整数a拆分为低32位和高32位
  const uint32_t aLo = (uint32_t)a;
  const uint32_t aHi = (uint32_t)(a >> 32);
  // 将64位整数b拆分为低32位和高32位
  const uint32_t bLo = (uint32_t)b;
  const uint32_t bHi = (uint32_t)(b >> 32);

  // 计算乘积的各部分
  const uint64_t b00 = (uint64_t)aLo * bLo;
  const uint64_t b01 = (uint64_t)aLo * bHi;
  const uint64_t b10 = (uint64_t)aHi * bLo;
  const uint64_t b11 = (uint64_t)aHi * bHi;

  // 拆分乘积的低32位和高32位
  const uint32_t b00Lo = (uint32_t)b00;
  const uint32_t b00Hi = (uint32_t)(b00 >> 32);

  // 计算中间结果
  const uint64_t mid1 = b10 + b00Hi;
  const uint32_t mid1Lo = (uint32_t)(mid1);
  const uint32_t mid1Hi = (uint32_t)(mid1 >> 32);

  // 继续计算中间结果
  const uint64_t mid2 = b01 + mid1Lo;
  const uint32_t mid2Lo = (uint32_t)(mid2);
  const uint32_t mid2Hi = (uint32_t)(mid2 >> 32);

  // 计算最终的乘积的高64位和低64位
  const uint64_t pHi = b11 + mid1Hi + mid2Hi;
  const uint64_t pLo = ((uint64_t)mid2Lo << 32) | b00Lo;

  // 通过指针参数返回高64位
  *productHi = pHi;
  return pLo;
}

// 定义一个内联函数，将64位整数lo和hi向右移动dist位，并返回结果
static inline uint64_t shiftright128(const uint64_t lo, const uint64_t hi, const uint32_t dist) {
  // 断言确保移动位数dist小于64
  assert(dist < 64);
  assert(dist > 0);
  // 将hi向左移动(64 - dist)位，将lo向右移动dist位，并将结果合并返回
  return (hi << (64 - dist)) | (lo >> dist);
}

// 如果编译器支持64位整数的内联函数，则定义以下函数
#endif // defined(HAS_64_BIT_INTRINSICS)

#if defined(RYU_32_BIT_PLATFORM)

// 定义一个内联函数，返回两个64位无符号整数a和b的128位乘积的高64位
static inline uint64_t umulh(const uint64_t a, const uint64_t b) {
  // 调用umul128函数计算乘积，并返回高64位结果
  uint64_t hi;
  umul128(a, b, &hi);
  return hi;
}

// 在32位平台上，编译器通常会对64位除法生成调用库函数，即使除数是常量。
// 这里的函数使用与64位编译器类似的方式，通过乘法来实现除以常数。

// 返回x除以5的结果（通过乘法实现）
static inline uint64_t div5(const uint64_t x) {
  return umulh(x, 0xCCCCCCCCCCCCCCCDu) >> 2;
}

// 返回x除以10的结果（通过乘法实现）
static inline uint64_t div10(const uint64_t x) {
  return umulh(x, 0xCCCCCCCCCCCCCCCDu) >> 3;
}

// 返回x除以100的结果（通过乘法实现）
static inline uint64_t div100(const uint64_t x) {
  return umulh(x >> 2, 0x28F5C28F5C28F5C3u) >> 2;
}

// 返回x除以10^8的结果（通过乘法实现）
static inline uint64_t div1e8(const uint64_t x) {
  return umulh(x, 0xABCC77118461CEFDu) >> 26;
}

// 返回x除以10^9的结果（通过乘法实现）
static inline uint64_t div1e9(const uint64_t x) {
  return umulh(x >> 9, 0x44B82FA09B5A53u) >> 11;
}
static inline uint32_t mod1e9(const uint64_t x) {
  // 避免尽可能使用64位数学运算。
  // 返回 (uint32_t) (x - 1000000000 * div1e9(x)) 将执行32x64位乘法和64位减法。
  // x 和 1000000000 * div1e9(x) 的差值保证小于10^9，因此它们的最高32位必定相同，
  // 因此我们可以在执行减法之前将两边截断为 uint32_t 类型。
  // 我们还可以简化 (uint32_t) (1000000000 * div1e9(x))。
  // 在乘法之前截断而不是之后，因为乘以 div1e9(x) 的最高32位不会影响最低32位。
  return ((uint32_t) x) - 1000000000 * ((uint32_t) div1e9(x));
}

#else // defined(RYU_32_BIT_PLATFORM)

static inline uint64_t div5(const uint64_t x) {
  return x / 5;
}

static inline uint64_t div10(const uint64_t x) {
  return x / 10;
}

static inline uint64_t div100(const uint64_t x) {
  return x / 100;
}

static inline uint64_t div1e8(const uint64_t x) {
  return x / 100000000;
}

static inline uint64_t div1e9(const uint64_t x) {
  return x / 1000000000;
}

static inline uint32_t mod1e9(const uint64_t x) {
  return (uint32_t) (x - 1000000000 * div1e9(x));
}

#endif // defined(RYU_32_BIT_PLATFORM)

static inline uint32_t pow5Factor(uint64_t value) {
  const uint64_t m_inv_5 = 14757395258967641293u; // 5 * m_inv_5 = 1 (mod 2^64)
  const uint64_t n_div_5 = 3689348814741910323u;  // #{ n | n = 0 (mod 2^64) } = 2^64 / 5
  uint32_t count = 0;
  for (;;) {
    assert(value != 0);
    value *= m_inv_5;
    if (value > n_div_5)
      break;
    ++count;
  }
  return count;
}

// 返回 true 如果 value 是 5^p 的倍数。
static inline bool multipleOfPowerOf5(const uint64_t value, const uint32_t p) {
  // 我尝试了对 p 的不同情况进行分析，但在性能上没有差异。
  return pow5Factor(value) >= p;
}

// 返回 true 如果 value 是 2^p 的倍数。
static inline bool multipleOfPowerOf2(const uint64_t value, const uint32_t p) {
  assert(value != 0);
  assert(p < 64);
  // __builtin_ctzll 在这里似乎不会更快。
  return (value & ((1ull << p) - 1)) == 0;
}

// 我们需要进行64x128位乘法和随后的128位移位。
// 乘法：
//   第一个64位因子是变量，并且通过查找表传递了128位因子。我们知道64位因子只有55个有效位（即，最高的9位是零）。128位因子只有124个有效位（即，最高的4位是零）。
// 移位：
//   原则上，乘法结果需要179位来表示（55 + 124）。然后我们将这个值向右移动 j 位，其中 j 至少 j >= 115，所以结果保证适合于 179 - 115 = 64 位。
//   这意味着我们只需要64x128位乘法的最高64个有效位。
//
// 有几种方法可以实现这一点：
// 1. 最佳情况：编译器暴露了一个128位类型。
// 如果定义了 HAS_UINT128 宏，则使用 128 位整数类型执行乘法和移位操作
static inline uint64_t mulShift64(const uint64_t m, const uint64_t* const mul, const int32_t j) {
  // 将 m 转换为 128 位整数，并与 mul[0] 执行乘法，得到 b0
  const uint128_t b0 = ((uint128_t) m) * mul[0];
  // 将 m 转换为 128 位整数，并与 mul[1] 执行乘法，得到 b2
  const uint128_t b2 = ((uint128_t) m) * mul[1];
  // 取 b0 的高 64 位，将其加到 b2，并将结果右移 (j - 64) 位
  return (uint64_t) (((b0 >> 64) + b2) >> (j - 64));
}

// 如果定义了 HAS_UINT128 宏，则使用 128 位整数类型执行全部乘法、移位和返回操作
static inline uint64_t mulShiftAll64(const uint64_t m, const uint64_t* const mul, const int32_t j,
  uint64_t* const vp, uint64_t* const vm, const uint32_t mmShift) {
  // 计算 vp, vm 和返回值，通过调用 mulShift64 函数来执行具体的乘法和移位操作
  *vp = mulShift64(4 * m + 2, mul, j);
  *vm = mulShift64(4 * m - 1 - mmShift, mul, j);
  return mulShift64(4 * m, mul, j);
}

// 如果定义了 HAS_64_BIT_INTRINSICS 宏，则使用 64 位 CPU 指令集中的特定指令执行乘法和移位操作
static inline uint64_t mulShift64(const uint64_t m, const uint64_t* const mul, const int32_t j) {
  uint64_t high1;                                   // 存储高位结果
  const uint64_t low1 = umul128(m, mul[1], &high1); // 执行 m 与 mul[1] 的乘法，得到低位结果和高位结果
  uint64_t high0;                                   // 存储高位结果
  umul128(m, mul[0], &high0);                       // 执行 m 与 mul[0] 的乘法，得到高位结果
  const uint64_t sum = high0 + low1;                // 将高位结果与低位结果相加
  if (sum < high0) {
    ++high1; // 如果相加结果小于高位结果，则发生溢出，需要增加 high1
  }
  // 将 sum 和 high1 右移 (j - 64) 位，并返回结果
  return shiftright128(sum, high1, j - 64);
}
// 如果没有定义HAS_UINT128和HAS_64_BIT_INTRINSICS，使用此函数进行64位乘法并右移操作
static inline uint64_t mulShift64(const uint64_t m, const uint64_t* const mul, const int32_t j) {
  // m最多55位
  uint64_t high1;                                   // 128位结果的高64位
  const uint64_t low1 = umul128(m, mul[1], &high1); // 使用mul[1]对m进行乘法，得到低64位结果，并将高64位存储到high1
  uint64_t high0;                                   // 64位结果的高64位
  umul128(m, mul[0], &high0);                       // 使用mul[0]对m进行乘法，得到高64位结果，并忽略低64位
  const uint64_t sum = high0 + low1;                // 将高位结果和低位结果相加
  if (sum < high0) {
    ++high1; // 如果加法溢出，则增加高128位结果的高64位
  }
  return shiftright128(sum, high1, j - 64); // 对sum和高128位结果右移j-64位，并返回结果
}

// 如果没有定义HAS_UINT128和定义了HAS_64_BIT_INTRINSICS，使用此函数进行64位乘法并右移操作
static inline uint64_t mulShiftAll64(uint64_t m, const uint64_t* const mul, const int32_t j,
  uint64_t* const vp, uint64_t* const vm, const uint32_t mmShift) {
  m <<= 1; // 将m左移1位，相当于乘以2
  // m最多55位
  uint64_t tmp;
  const uint64_t lo = umul128(m, mul[0], &tmp); // 使用mul[0]对m进行乘法，得到低64位结果，并将高64位存储到tmp
  uint64_t hi;
  const uint64_t mid = tmp + umul128(m, mul[1], &hi); // 使用mul[1]对m进行乘法，得到中间64位结果，并将高64位存储到hi
  hi += mid < tmp; // 如果加法溢出，则增加高128位结果的高64位

  const uint64_t lo2 = lo + mul[0]; // 将lo与mul[0]相加，得到新的低64位结果
  const uint64_t mid2 = mid + mul[1] + (lo2 < lo); // 将mid与mul[1]相加，并加上可能的进位
  const uint64_t hi2 = hi + (mid2 < mid); // 将hi与mid2相加，并加上可能的进位
  *vp = shiftright128(mid2, hi2, (uint32_t) (j - 64 - 1)); // 对mid2和hi2右移j-64-1位，并将结果存储到vp

  if (mmShift == 1) {
    const uint64_t lo3 = lo - mul[0]; // 将lo与mul[0]相减，得到新的低64位结果
    const uint64_t mid3 = mid - mul[1] - (lo3 > lo); // 将mid与mul[1]相减，并减去可能的借位
    const uint64_t hi3 = hi - (mid3 > mid); // 将hi与mid3相减，并减去可能的借位
    *vm = shiftright128(mid3, hi3, (uint32_t) (j - 64 - 1)); // 对mid3和hi3右移j-64-1位，并将结果存储到vm
  } else {
    const uint64_t lo3 = lo + lo; // 将lo与自身相加，得到新的低64位结果
    const uint64_t mid3 = mid + mid + (lo3 < lo); // 将mid与自身相加，并加上可能的进位
    const uint64_t hi3 = hi + hi + (mid3 < mid); // 将hi与自身相加，并加上可能的进位
    const uint64_t lo4 = lo3 - mul[0]; // 将lo3与mul[0]相减，得到最终的低64位结果
    const uint64_t mid4 = mid3 - mul[1] - (lo4 > lo3); // 将mid3与mul[1]相减，并减去可能的借位
    const uint64_t hi4 = hi3 - (mid4 > mid3); // 将hi3与mid4相减，并减去可能的借位
    *vm = shiftright128(mid4, hi4, (uint32_t) (j - 64)); // 对mid4和hi4右移j-64位，并将结果存储到vm
  }

  return shiftright128(mid, hi, (uint32_t) (j - 64 - 1)); // 对mid和hi右移j-64-1位，并返回结果
}
```