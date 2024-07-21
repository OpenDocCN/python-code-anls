# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gavgpool\up8xm-sse2.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <qnnpack/q8gavgpool.h>

void pytorch_q8gavgpool_ukernel_up8xm__sse2(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  // 断言确保 m 至少为 1，n 小于 8
  assert(m >= 1);
  assert(n < 8);

  // 加载偏置值并初始化累加器向量
  const __m128i vbias =
      _mm_loadu_si128((const __m128i*)&quantization_params->sse2.bias);
  __m128i vacc_lo = vbias;
  __m128i vacc_hi = vbias;
  // 初始化零向量
  __m128i vzero = _mm_setzero_si128();
  // 处理 m 大于等于 8 的情况
  while (m >= 8) {
    // 加载输入向量并进行拆分和零扩展
    const __m128i vinput = _mm_loadl_epi64((const __m128i*)input);
    const __m128i vxinput = _mm_unpacklo_epi8(vinput, vzero);
    // 累加低位和高位部分
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi8(vxinput, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi8(vxinput, vzero));

    // 更新输入指针和剩余处理的行数
    input += input_stride;
    m--;
  }
  // 处理剩余的行数不足 8 的情况
  while (m-- != 0) {
    // 跳过输入的前 n 个元素
    input += n;
    __m128i vinput = _mm_setzero_si128();
    // 根据 n 的奇偶性加载输入向量的不同部分
    if (n & 1) {
      input -= 1;
      vinput = _mm_cvtsi32_si128((int)(uint32_t)*input);
    }
    if (n & 2) {
      vinput = _mm_slli_epi32(vinput, 16);
      input -= 2;
      vinput = _mm_insert_epi16(vinput, *((const uint16_t*)input), 0);
    }
    if (n & 4) {
      input -= 4;
      vinput = _mm_unpacklo_epi32(
          _mm_cvtsi32_si128((int)*((const uint32_t*)input)), vinput);
    }
    input += input_stride;

    // 对输入向量进行零扩展和拆分，然后累加到累加器向量中
    const __m128i vxinput = _mm_unpacklo_epi8(vinput, vzero);
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi8(vxinput, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi8(vxinput, vzero));
  }

  // 加载量化参数中的缩放因子
  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

  // 将累加器向量转换为浮点型并乘以缩放因子
  const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
  const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

  // 将浮点型累加结果转换回整型，并进行饱和转换和打包
  const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
  const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

  // 加载量化参数中的输出零点，对输出进行处理并进行饱和处理
  __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
  vout = _mm_adds_epi16(
      vout,
      _mm_load_si128(
          (const __m128i*)quantization_params->sse2.output_zero_point));
  vout = _mm_packus_epi16(vout, vout);
  vout = _mm_min_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
  vout = _mm_max_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

  // 根据 n 的奇偶性存储输出向量的不同部分
  if (n & 4) {
    *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
    output += 4;
    vout = _mm_srli_epi64(vout, 32);
  }
  if (n & 2) {
    *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
    output += 2;
    vout = _mm_srli_epi32(vout, 16);
  }
  if (n & 1) {
    *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
  }
}
```