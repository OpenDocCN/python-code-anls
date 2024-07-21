# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\gemmlowp-sse2.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <emmintrin.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

#include "gemmlowp-sse.h"

/* 使用 SSE2 指令集实现的 PyTorch QNNPACK 的量化模块 */

void pytorch_qnnp_requantize_gemmlowp__sse2(
    size_t n,                               // 待处理元素的数量，必须为 16 的倍数
    const int32_t* input,                   // 输入数组，包含要量化的元素
    float scale,                            // 量化比例因子
    uint8_t zero_point,                     // 量化零点
    uint8_t qmin,                           // 量化的最小值
    uint8_t qmax,                           // 量化的最大值
    uint8_t* output) {                      // 输出数组，存储量化后的结果

  assert(n % 16 == 0);                      // 确保输入元素数量是 16 的倍数
  assert(scale < 1.0f);                     // 确保量化比例因子小于 1.0
  assert(scale >= 0x1.0p-32f);              // 确保量化比例因子大于等于 2^-32

  const uint32_t scale_bits = fp32_to_bits(scale);  // 获取量化比例因子的位表示

  /* 计算量化参数 */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;  // 计算乘法器
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;    // 计算指数
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);                          // 计算右移量

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);  // 设置乘法器为 SSE 寄存器类型
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);  // 设置零点为 SSE 寄存器类型
  const __m128i vqmin = _mm_set1_epi8((char)qmin);        // 设置最小量化值为 SSE 寄存器类型
  const __m128i vqmax = _mm_set1_epi8((char)qmax);        // 设置最大量化值为 SSE 寄存器类型
  for (; n != 0; n -= 16) {                                // 迭代处理每组 16 个元素
    const __m128i x = _mm_loadu_si128((const __m128i*)input);   // 加载 4 个输入元素到 SSE 寄存器 x
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4)); // 加载 4 个输入元素到 SSE 寄存器 y
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8)); // 加载 4 个输入元素到 SSE 寄存器 z
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12)); // 加载 4 个输入元素到 SSE 寄存器 w
    input += 16;                                            // 更新输入指针

    const __m128i x_product = gemmlowp_sse_vqrdmulh_s32(x, vmultiplier); // 使用乘法器对 x 进行量化
    const __m128i y_product = gemmlowp_sse_vqrdmulh_s32(y, vmultiplier); // 使用乘法器对 y 进行量化
    const __m128i z_product = gemmlowp_sse_vqrdmulh_s32(z, vmultiplier); // 使用乘法器对 z 进行量化
    const __m128i w_product = gemmlowp_sse_vqrdmulh_s32(w, vmultiplier); // 使用乘法器对 w 进行量化

    const __m128i x_scaled = gemmlowp_sse_rdivbypo2_s32(x_product, shift); // 右移量化后的 x
    const __m128i y_scaled = gemmlowp_sse_rdivbypo2_s32(y_product, shift); // 右移量化后的 y
    const __m128i z_scaled = gemmlowp_sse_rdivbypo2_s32(z_product, shift); // 右移量化后的 z
    const __m128i w_scaled = gemmlowp_sse_rdivbypo2_s32(w_product, shift); // 右移量化后的 w

    const __m128i xy_packed = _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);  // 将 x_scaled 和 y_scaled 紧凑并加上零点
    const __m128i zw_packed = _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);  // 将 z_scaled 和 w_scaled 紧凑并加上零点
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);  // 将 xy_packed 和 zw_packed 紧凑成无符号 8 位整数
    const __m128i xyzw_clamped = _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);  // 将 xyzw_packed 进行上下限制

    _mm_storeu_si128((__m128i*)output, xyzw_clamped);    // 存储量化结果到输出数组
    output += 16;                                         // 更新输出指针
  }
}
```