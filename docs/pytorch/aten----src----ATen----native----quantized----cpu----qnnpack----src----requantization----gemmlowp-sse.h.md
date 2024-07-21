# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\gemmlowp-sse.h`

```py
/*
 * 版权声明：
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * 本源代码根据位于根目录下的 LICENSE 文件中的 BSD 风格许可证授权。
 */

#pragma once

#include <limits.h>

#include <immintrin.h>

/*
 * 以下代码源自 Google 的 gemmlowp 库。
 * 仅在 QNNPACK 单元测试和比较基准中使用，而非库本身。
 */

// 版权所有 2015 年 Google Inc. 保留所有权利。
//
// 根据 Apache 许可证版本 2.0 授权；
// 除非符合许可证的规定，否则不得使用此文件。
// 您可以获取许可证的副本：
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则软件按"原样"分发，
// 不附带任何明示或暗示的担保或条件。
// 请参阅许可证以获取特定语言的权限和限制。

// 定义静态内联函数 gemmlowp_sse_rdivbypo2_s32，用于 SSE 指令集下整数除法操作
static inline __m128i gemmlowp_sse_rdivbypo2_s32(__m128i x, int exponent) {
  const __m128i mask =
      _mm_set1_epi32((int32_t)((UINT64_C(1) << exponent) - UINT64_C(1)));  // 创建掩码，用于截断 x 的低位
  const __m128i remainder = _mm_and_si128(x, mask);  // 计算 x 除以 2^exponent 的余数
  const __m128i threshold = _mm_sub_epi32(
      _mm_srli_epi32(mask, 1), _mm_cmplt_epi32(x, _mm_setzero_si128()));  // 计算阈值，用于判断舍入方向
  return _mm_sub_epi32(
      _mm_sra_epi32(x, _mm_cvtsi32_si128(exponent)),  // 将 x 右移 exponent 位，相当于整数除法
      _mm_cmpgt_epi32(remainder, threshold));  // 根据余数与阈值的比较结果调整结果
}

// 定义静态内联函数 gemmlowp_sse_mul_s32，用于 SSE 指令集下整数乘法操作
static inline __m128i gemmlowp_sse_mul_s32(__m128i a, __m128i b) {
#ifdef __SSE4_1__
  return _mm_mul_epi32(a, b);  // 如果支持 SSE4.1 指令集，直接使用整数乘法指令
#else
  __m128i sign, zero, mul_us, a_neg, b_neg, mul_us_neg;
  sign = _mm_xor_si128(a, b);  // 计算 a 和 b 的异或，提取符号位
  sign = _mm_srai_epi32(sign, 31);  // 将符号位扩展到所有字段，如果为负则全为 1，否则全为 0
  sign = _mm_shuffle_epi32(
      sign,
      _MM_SHUFFLE(2, 2, 0, 0));  // 通过移位操作将符号位扩展到第 3 和第 1 个数据通道

  zero = _mm_setzero_si128();  // 初始化为全零向量
#ifdef __SSSE3__
  a_neg = _mm_abs_epi32(a);  // 使用 SSSE3 指令计算 a 和 b 的绝对值
  b_neg = _mm_abs_epi32(b);
#else /* pre-SSSE3 */
  const __m128i a_neg_mask = _mm_cmplt_epi32(a, zero);  // 计算 a 和 b 是否小于零的掩码
  a_neg = _mm_sub_epi32(_mm_xor_si128(a, a_neg_mask), a_neg_mask);  // 计算 a 的绝对值
  const __m128i b_neg_mask = _mm_cmplt_epi32(b, zero);
  b_neg = _mm_sub_epi32(_mm_xor_si128(b, b_neg_mask), b_neg_mask);  // 计算 b 的绝对值
#endif /* pre-SSSE3 */

  mul_us = _mm_mul_epu32(a_neg, b_neg);  // 使用无符号乘法计算乘积的低位部分
  mul_us_neg = _mm_sub_epi64(zero, mul_us);  // 计算乘积的高位部分
  mul_us_neg = _mm_and_si128(sign, mul_us_neg);  // 根据符号位确定最终乘积的正负
  mul_us = _mm_andnot_si128(sign, mul_us);
  return _mm_or_si128(mul_us, mul_us_neg);  // 合并正负部分得到最终结果
#endif
}
// 定义了一个静态内联函数，使用 SSE 指令集执行有符号整数乘法取高位运算
static inline __m128i gemmlowp_sse_vqrdmulh_s32(__m128i a, __m128i b) {
  // 如果 a == b == INT32_MIN，则会发生饱和
  const __m128i min = _mm_set1_epi32(INT32_MIN);
  // 判断是否需要饱和的掩码
  const __m128i saturation_mask =
      _mm_and_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, min));

  // 将 a 和 b 拆分成低位和高位两个 64 位整数
  const __m128i a0_a2 = a;
  const __m128i a1_a3 = _mm_srli_si128(a, 4);
  const __m128i b0_b2 = b;
  const __m128i b1_b3 = _mm_srli_si128(b, 4);

  // 计算 a0*b0 和 a2*b2 的乘积
  const __m128i a0b0_a2b2 = gemmlowp_sse_mul_s32(a0_a2, b0_b2);
  // 计算 a1*b1 和 a3*b3 的乘积
  const __m128i a1b1_a3b3 = gemmlowp_sse_mul_s32(a1_a3, b1_b3);

  // 执行舍入，并考虑到乘积将被加倍
  const __m128i nudge = _mm_set1_epi64x(1 << 30);
  const __m128i a0b0_a2b2_rounded = _mm_add_epi64(a0b0_a2b2, nudge);
  const __m128i a1b1_a3b3_rounded = _mm_add_epi64(a1b1_a3b3, nudge);

  // 对结果执行加倍操作
  const __m128i a0b0_a2b2_rounded_2x = _mm_slli_epi64(a0b0_a2b2_rounded, 1);
  const __m128i a1b1_a3b3_rounded_2x = _mm_slli_epi64(a1b1_a3b3_rounded, 1);

  // 获取乘积的高位部分
#ifdef __SSE4_1__
  // 如果支持 SSE4.1，则通过混合操作生成结果
  const __m128i result = _mm_blend_epi16(
      _mm_srli_epi64(a0b0_a2b2_rounded_2x, 32), a1b1_a3b3_rounded_2x, 0xCC);
#else
  // 如果不支持 SSE4.1，则通过转换为浮点数、Shuffle 操作生成结果
  const __m128i result0213 = _mm_castps_si128(_mm_shuffle_ps(
      _mm_castsi128_ps(a0b0_a2b2_rounded_2x),
      _mm_castsi128_ps(a1b1_a3b3_rounded_2x),
      _MM_SHUFFLE(3, 1, 3, 1)));
  const __m128i result = _mm_shuffle_epi32(result0213, _MM_SHUFFLE(3, 1, 2, 0));
#endif

  // 对溢出的结果进行饱和处理
#ifdef __SSE4_1__
  // 如果支持 SSE4.1，则通过混合选择操作饱和结果
  const __m128i saturated_result =
      _mm_blendv_epi8(result, min, saturation_mask);
#else
  // 如果不支持 SSE4.1，则通过与操作生成饱和结果
  const __m128i saturated_result = _mm_or_si128(
      _mm_and_si128(saturation_mask, min),
      _mm_andnot_si128(saturation_mask, result));
#endif

  // 返回饱和处理后的结果
  return saturated_result;
}
```