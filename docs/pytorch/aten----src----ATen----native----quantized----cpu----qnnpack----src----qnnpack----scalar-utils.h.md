# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\scalar-utils.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#include <fp16/bitcasts.h>

#if defined(__clang__)
#if __clang_major__ == 3 && __clang_minor__ >= 7 || __clang_major__ > 3
// 定义一个宏用于告诉编译器忽略基础移位错误的检查
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB \
  __attribute__((__no_sanitize__("shift-base")))
#else
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
#endif
#elif defined(__GNUC__)
#if __GNUC__ >= 8
// 定义一个宏用于告诉编译器忽略基础移位错误的检查
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB \
  __attribute__((__no_sanitize__("shift-base")))
#elif __GNUC__ == 4 && __GNUC_MINOR__ >= 9 || __GNUC__ > 4
// GCC 版本在 4.9 到 8 之间支持 UBSan，但不支持 no_sanitize 属性
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
#ifndef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
// 如果未定义 PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND，则定义为 1
#define PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND 1
#endif
#else
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
#endif
#else
#define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
#endif

// 使用 PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB 宏，忽略基础移位错误的检查
PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
// 定义一个静态内联函数，对 int32_t 类型的数值进行算术右移
inline static int32_t asr_s32(int32_t x, uint32_t n) {
#ifdef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
#if defined(__x86_64__) || defined(__aarch64__)
  // 在支持的体系结构上执行算术右移操作
  return (int32_t)((uint64_t)(int64_t)x >> n);
#else
  // 对于不支持的体系结构，通过按位取反和逻辑右移实现算术右移
  return x >= 0 ? x >> n : ~(~x >> n);
#endif
#else
  // 在正常情况下直接执行算术右移操作
  return x >> n;
#endif
}

// 使用 PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB 宏，忽略基础移位错误的检查
PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
// 定义一个静态内联函数，对 int64_t 类型的数值进行算术右移
inline static int64_t asr_s64(int64_t x, uint32_t n) {
#ifdef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
  // 对于不支持的体系结构，通过按位取反和逻辑右移实现算术右移
  return x >= 0 ? x >> n : ~(~x >> n);
#else
  // 在正常情况下直接执行算术右移操作
  return x >> n;
#endif
}

// 定义一个静态内联函数，实现精确的 PyTorch 标量重新量化过程
inline static uint8_t pytorch_scalar_requantize_precise(
    int32_t value,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax) {
  // 断言确保缩放因子小于1.0
  assert(scale < 1.0f);
  // 断言确保缩放因子大于等于2的-32次方
  assert(scale >= 0x1.0p-32f);

  // 将浮点数缩放因子转换为32位无符号整数表示
  const uint32_t scale_bits = fp32_to_bits(scale);
  // 构建乘法器，保证其尾数部分的最高位为1
  const uint32_t multiplier =
      (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  // 计算右移位数，确保乘法结果在合适的范围内
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  // 断言确保右移位数在合理范围内
  assert(shift >= 24);
  assert(shift < 56);

  /*
   * 计算输入值的绝对值作为无符号32位整数。
   * 所有后续计算将使用无符号值，以避免在有符号操作中出现未定义行为。
   */
  const uint32_t abs_value = (value >= 0) ? (uint32_t)value : -(uint32_t)value;

  /* 计算32位因子的完整64位乘积 */
  const uint64_t product = (uint64_t)abs_value * (uint64_t)multiplier;

  /*
   * 将完整的64位乘积向右移动并进行四舍五入。
   * 四舍五入是向最接近的整数进行，中点情况会向上舍入（远离零）。
   */
  const uint64_t rounding = UINT64_C(1) << (shift - 1);
  const uint32_t abs_scaled_value = (uint32_t)((product + rounding) >> shift);

  /*
   * 将输入的符号复制到缩放后的绝对值输入。
   */
  const int32_t scaled_value =
      (int32_t)(value >= 0 ? abs_scaled_value : -abs_scaled_value);

  /* 将缩放值夹在介于smin和smax之间的零点之间 */
  int32_t clamped_value = scaled_value;
  const int32_t smin = (int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point;
  if (clamped_value < smin) {
    clamped_value = smin;
  }
  const int32_t smax = (int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point;
  if (clamped_value > smax) {
    clamped_value = smax;
  }

  /* 将零点添加到夹紧值 */
  const int32_t biased_value = clamped_value + (int32_t)(uint32_t)zero_point;

  return biased_value;
}
}



# 这行代码关闭了一个代码块。在很多编程语言中，使用大括号 `{}` 来标志代码块的开始和结束。
# 在这里，单独的一个右大括号 `}` 表示着一个代码块的结束。
```