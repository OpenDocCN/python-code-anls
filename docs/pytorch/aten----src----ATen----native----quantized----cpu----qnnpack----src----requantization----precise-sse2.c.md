# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\precise-sse2.c`

```
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

void pytorch_qnnp_requantize_precise__sse2(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 16 == 0);  // 确保输入大小是16的倍数
  assert(scale < 1.0f);  // 确保缩放因子小于1.0
  assert(scale >= 0x1.0p-32f);  // 确保缩放因子大于等于2^-32

  const uint32_t scale_bits = fp32_to_bits(scale);  // 将浮点数缩放因子转换为整数位表示
  const uint32_t multiplier =
      (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);  // 提取尾数部分并设置指数位
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);  // 计算右移的位数
  assert(shift >= 24);  // 确保右移量大于等于24
  assert(shift < 56);   // 确保右移量小于56
  const uint64_t rounding = UINT64_C(1) << (shift - 1);  // 设置舍入常数

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);  // 设置用于乘法的 SSE2 向量
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);  // 设置零点偏移的 SSE2 向量
  const __m128i vqmin = _mm_set1_epi8((char)qmin);  // 设置量化最小值的 SSE2 向量
  const __m128i vqmax = _mm_set1_epi8((char)qmax);  // 设置量化最大值的 SSE2 向量
  const __m128i vshift = _mm_cvtsi32_si128((int)shift);  // 设置移位量的 SSE2 向量
  const __m128i vrounding = _mm_set1_epi64x(rounding);  // 设置舍入常数的 SSE2 向量
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);  // 加载输入数据向量 x
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));  // 加载输入数据向量 y
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));  // 加载输入数据向量 z
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));  // 加载输入数据向量 w
    input += 16;  // 移动输入指针到下一个块

    const __m128i x_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), x);  // 检查 x 向量中的负数
    const __m128i y_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), y);  // 检查 y 向量中的负数
    const __m128i z_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), z);  // 检查 z 向量中的负数
    const __m128i w_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), w);  // 检查 w 向量中的负数

    const __m128i x_abs0123 =
        _mm_sub_epi32(_mm_xor_si128(x, x_neg_mask), x_neg_mask);  // 计算 x 的绝对值
    const __m128i y_abs0123 =
        _mm_sub_epi32(_mm_xor_si128(y, y_neg_mask), y_neg_mask);  // 计算 y 的绝对值
    const __m128i z_abs0123 =
        _mm_sub_epi32(_mm_xor_si128(z, z_neg_mask), z_neg_mask);  // 计算 z 的绝对值
    const __m128i w_abs0123 =
        _mm_sub_epi32(_mm_xor_si128(w, w_neg_mask), w_neg_mask);  // 计算 w 的绝对值

    const __m128i x_abs1032 =
        _mm_shuffle_epi32(x_abs0123, _MM_SHUFFLE(2, 3, 0, 1));  // 重新排列 x 的绝对值
    const __m128i y_abs1032 =
        _mm_shuffle_epi32(y_abs0123, _MM_SHUFFLE(2, 3, 0, 1));  // 重新排列 y 的绝对值
    const __m128i z_abs1032 =
        _mm_shuffle_epi32(z_abs0123, _MM_SHUFFLE(2, 3, 0, 1));  // 重新排列 z 的绝对值
    const __m128i w_abs1032 =
        _mm_shuffle_epi32(w_abs0123, _MM_SHUFFLE(2, 3, 0, 1));  // 重新排列 w 的绝对值

    const __m128i x_absmul02 = _mm_mul_epu32(x_abs0123, vmultiplier);  // 计算 x 的绝对值乘以乘法因子的结果
    const __m128i y_absmul02 = _mm_mul_epu32(y_abs0123, vmultiplier);  // 计算 y 的绝对值乘以乘法因子的结果
    const __m128i z_absmul02 = _mm_mul_epu32(z_abs0123, vmultiplier);  // 计算 z 的绝对值乘以乘法因子的结果
    const __m128i w_absmul02 = _mm_mul_epu32(w_abs0123, vmultiplier);  // 计算 w 的绝对值乘以乘法因子的结果

    const __m128i x_absmul13 = _mm_mul_epu32(x_abs1032, vmultiplier);  // 计算 x 重新排列后的绝对值乘以乘法因子的结果
    // 使用 vmultiplier 对 y_abs1032 进行无符号整数乘法
    const __m128i y_absmul13 = _mm_mul_epu32(y_abs1032, vmultiplier);
    // 使用 vmultiplier 对 z_abs1032 进行无符号整数乘法
    const __m128i z_absmul13 = _mm_mul_epu32(z_abs1032, vmultiplier);
    // 使用 vmultiplier 对 w_abs1032 进行无符号整数乘法
    const __m128i w_absmul13 = _mm_mul_epu32(w_abs1032, vmultiplier);

    // 对 x_absmul02 进行饱和右移和舍入
    const __m128i x_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(x_absmul02, vrounding), vshift);
    // 对 x_absmul13 进行饱和右移和舍入
    const __m128i x_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(x_absmul13, vrounding), vshift);
    // 对 y_absmul02 进行饱和右移和舍入
    const __m128i y_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(y_absmul02, vrounding), vshift);
    // 对 y_absmul13 进行饱和右移和舍入
    const __m128i y_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(y_absmul13, vrounding), vshift);
    // 对 z_absmul02 进行饱和右移和舍入
    const __m128i z_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(z_absmul02, vrounding), vshift);
    // 对 z_absmul13 进行饱和右移和舍入
    const __m128i z_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(z_absmul13, vrounding), vshift);
    // 对 w_absmul02 进行饱和右移和舍入
    const __m128i w_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(w_absmul02, vrounding), vshift);
    // 对 w_absmul13 进行饱和右移和舍入
    const __m128i w_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(w_absmul13, vrounding), vshift);

    // 将 x_abs_scaled02 和 x_abs_scaled13 的结果按指定顺序打包
    const __m128i x_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x_abs_scaled02),
        _mm_castsi128_ps(x_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    // 将 y_abs_scaled02 和 y_abs_scaled13 的结果按指定顺序打包
    const __m128i y_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(y_abs_scaled02),
        _mm_castsi128_ps(y_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    // 将 z_abs_scaled02 和 z_abs_scaled13 的结果按指定顺序打包
    const __m128i z_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(z_abs_scaled02),
        _mm_castsi128_ps(z_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    // 将 w_abs_scaled02 和 w_abs_scaled13 的结果按指定顺序打包
    const __m128i w_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(w_abs_scaled02),
        _mm_castsi128_ps(w_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));

    // 将 x_abs_scaled0213 向量按指定顺序重排
    const __m128i x_abs_scaled =
        _mm_shuffle_epi32(x_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    // 将 y_abs_scaled0213 向量按指定顺序重排
    const __m128i y_abs_scaled =
        _mm_shuffle_epi32(y_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    // 将 z_abs_scaled0213 向量按指定顺序重排
    const __m128i z_abs_scaled =
        _mm_shuffle_epi32(z_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    // 将 w_abs_scaled0213 向量按指定顺序重排
    const __m128i w_abs_scaled =
        _mm_shuffle_epi32(w_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));

    // 对 x_abs_scaled 进行符号取反并减去 x_neg_mask
    const __m128i x_scaled =
        _mm_sub_epi32(_mm_xor_si128(x_abs_scaled, x_neg_mask), x_neg_mask);
    // 对 y_abs_scaled 进行符号取反并减去 y_neg_mask
    const __m128i y_scaled =
        _mm_sub_epi32(_mm_xor_si128(y_abs_scaled, y_neg_mask), y_neg_mask);
    // 对 z_abs_scaled 进行符号取反并减去 z_neg_mask
    const __m128i z_scaled =
        _mm_sub_epi32(_mm_xor_si128(z_abs_scaled, z_neg_mask), z_neg_mask);
    // 对 w_abs_scaled 进行符号取反并减去 w_neg_mask
    const __m128i w_scaled =
        _mm_sub_epi32(_mm_xor_si128(w_abs_scaled, w_neg_mask), w_neg_mask);

    // 将 x_scaled 和 y_scaled 向量进行饱和加法，并加上 vzero_point
    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    // 将 z_scaled 和 w_scaled 向量进行饱和加法，并加上 vzero_point
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    // 将 xy_packed 和 zw_packed 向量进行无符号饱和打包
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    # 使用 SSE 指令集中的 _mm_min_epu8 和 _mm_max_epu8 函数对 xyzw_packed 进行无符号整数字节级最小值和最大值的限制操作，并保存到 xyzw_clamped 中
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

    /*
     * 执行以下 SIMD 操作:
     * 4x PXOR (将寄存器置零)
     * 8x PSUBD (无符号整数32位减法)
     * 8x PXOR (按位异或)
     * 8x PSHUFD (按需求重新排列数据)
     * 8x PMULUDQ (无符号整数32位乘法)
     * 8x PSRLQ (逻辑右移)
     * 8x PADDQ (无符号整数64位加法)
     * 4x SHUFPS (按需求重新排列单精度浮点数)
     * 2x PACKSSDW (带符号32位整数转换)
     * 1x PACKUSWB (无符号字转换)
     * 2x PADDW (无符号16位整数加法)
     * 1x PMAXUB (无符号字节最大值)
     * 1x PMINUB (无符号字节最小值)
     * ---------------------
     * 总共63条指令
     */

    # 将 xyzw_clamped 中的数据存储到 output 所指向的内存地址，每次存储128位数据（16字节）
    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    # 更新 output 指针，使其指向下一个128位数据的存储地址
    output += 16;
}



# 这是一个单独的右花括号 '}'，用于结束某个代码块（如函数、循环、条件语句等）的定义或结构。
```