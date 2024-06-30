# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\ryu\ryu\f2s_intrinsics.h`

```
// 定义头文件的宏保护，避免重复包含
#ifndef RYU_F2S_INTRINSICS_H
#define RYU_F2S_INTRINSICS_H

// 包含通用定义头文件，用于检测是否定义了 RYU_32_BIT_PLATFORM
#include "ryu/common.h"

// 根据宏定义选择使用完整的浮点数转换表格或优化大小的双精度转换表格
#if defined(RYU_FLOAT_FULL_TABLE)
// 使用完整的浮点数转换表格
#include "ryu/f2s_full_table.h"
#else
// 根据优化大小标志选择双精度转换表格
#if defined(RYU_OPTIMIZE_SIZE)
#include "ryu/d2s_small_table.h"
#else
#include "ryu/d2s_full_table.h"
#endif
// 定义浮点数 5 的负幂次幂和 5 的幂次幂的位数
#define FLOAT_POW5_INV_BITCOUNT (DOUBLE_POW5_INV_BITCOUNT - 64)
#define FLOAT_POW5_BITCOUNT (DOUBLE_POW5_BITCOUNT - 64)
#endif

// 静态内联函数，计算给定值的 5 的幂次因子数量
static inline uint32_t pow5factor_32(uint32_t value) {
  uint32_t count = 0;
  for (;;) {
    assert(value != 0);
    const uint32_t q = value / 5;
    const uint32_t r = value % 5;
    if (r != 0) {
      break;
    }
    value = q;
    ++count;
  }
  return count;
}

// 返回 true 如果 value 可以被 5^p 整除
static inline bool multipleOfPowerOf5_32(const uint32_t value, const uint32_t p) {
  return pow5factor_32(value) >= p;
}

// 返回 true 如果 value 可以被 2^p 整除
static inline bool multipleOfPowerOf2_32(const uint32_t value, const uint32_t p) {
  // __builtin_ctz 在此处似乎不会更快。
  return (value & ((1u << p) - 1)) == 0;
}

// 在这里避免使用 uint128_t 看起来稍微快一些，尽管对于 uint128_t 生成的代码看起来更好一些。
static inline uint32_t mulShift32(const uint32_t m, const uint64_t factor, const int32_t shift) {
  assert(shift > 32);

  // 这里的强制转换帮助 MSVC 避免调用 __allmul 库函数。
  const uint32_t factorLo = (uint32_t)(factor);
  const uint32_t factorHi = (uint32_t)(factor >> 32);
  const uint64_t bits0 = (uint64_t)m * factorLo;
  const uint64_t bits1 = (uint64_t)m * factorHi;

#if defined(RYU_32_BIT_PLATFORM)
  // 在 32 位平台上，我们可以避免 64 位右移，因为我们只需要结果的高 32 位，而移位值大于 32。
  const uint32_t bits0Hi = (uint32_t)(bits0 >> 32);
  uint32_t bits1Lo = (uint32_t)(bits1);
  uint32_t bits1Hi = (uint32_t)(bits1 >> 32);
  bits1Lo += bits0Hi;
  bits1Hi += (bits1Lo < bits0Hi);
  if (shift >= 64) {
    // s2f 可能会使用大于等于 64 的移位值调用此函数，我们需要处理这种情况。
    // 这可能会比 !defined(RYU_32_BIT_PLATFORM) 的情况慢。
    return (uint32_t)(bits1Hi >> (shift - 64));
  } else {
    const int32_t s = shift - 32;
    return (bits1Hi << (32 - s)) | (bits1Lo >> s);
  }
#endif
#else // RYU_32_BIT_PLATFORM
  // 计算两部分的和，其中 bits0 右移 32 位后加上 bits1
  const uint64_t sum = (bits0 >> 32) + bits1;
  // 将 sum 右移 (shift - 32) 位
  const uint64_t shiftedSum = sum >> (shift - 32);
  // 断言 shiftedSum 不大于 UINT32_MAX
  assert(shiftedSum <= UINT32_MAX);
  // 返回 shiftedSum 转换为 uint32_t 类型的值
  return (uint32_t) shiftedSum;
#endif // RYU_32_BIT_PLATFORM
}

static inline uint32_t mulPow5InvDivPow2(const uint32_t m, const uint32_t q, const int32_t j) {
#if defined(RYU_FLOAT_FULL_TABLE)
  // 使用 mulShift32 函数计算 m 乘以 FLOAT_POW5_INV_SPLIT[q]，右移 j 位的结果
  return mulShift32(m, FLOAT_POW5_INV_SPLIT[q], j);
#elif defined(RYU_OPTIMIZE_SIZE)
  // 计算 2^q / 5^q 的逆，将结果存储在 pow5 数组中
  uint64_t pow5[2];
  double_computeInvPow5(q, pow5);
  // 使用 mulShift32 函数计算 m 乘以 pow5[1] + 1，右移 j 位的结果
  return mulShift32(m, pow5[1] + 1, j);
#else
  // 计算 2^q / 5^q 的逆，从 DOUBLE_POW5_INV_SPLIT 表中获取第二部分，加上 1 后使用 mulShift32 函数计算结果
  return mulShift32(m, DOUBLE_POW5_INV_SPLIT[q][1] + 1, j);
#endif
}

static inline uint32_t mulPow5divPow2(const uint32_t m, const uint32_t i, const int32_t j) {
#if defined(RYU_FLOAT_FULL_TABLE)
  // 使用 mulShift32 函数计算 m 乘以 FLOAT_POW5_SPLIT[i]，右移 j 位的结果
  return mulShift32(m, FLOAT_POW5_SPLIT[i], j);
#elif defined(RYU_OPTIMIZE_SIZE)
  // 计算 2^i / 5^i，将结果存储在 pow5 数组中
  uint64_t pow5[2];
  double_computePow5(i, pow5);
  // 使用 mulShift32 函数计算 m 乘以 pow5[1]，右移 j 位的结果
  return mulShift32(m, pow5[1], j);
#else
  // 从 DOUBLE_POW5_SPLIT 表中获取第二部分，使用 mulShift32 函数计算 m 乘以这个值，右移 j 位的结果
  return mulShift32(m, DOUBLE_POW5_SPLIT[i][1], j);
#endif
}

#endif // RYU_F2S_INTRINSICS_H
```