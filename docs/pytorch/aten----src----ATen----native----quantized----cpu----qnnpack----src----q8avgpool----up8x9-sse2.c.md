# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8avgpool\up8x9-sse2.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <qnnpack/q8avgpool.h>

void pytorch_q8avgpool_ukernel_up8x9__sse2(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(n != 0); // 确保处理的元素数量不为零
  assert(ks <= 9); // 确保卷积核尺寸不超过9
  assert(kc >= 8); // 确保通道数至少为8

  const __m128i vbias = // 加载偏置向量
      _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);
  const __m128i vzero = _mm_setzero_si128(); // 创建全零向量
  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale); // 加载缩放因子向量

  do {
    const uint8_t* i0 = input[0]; // 加载输入数据指针
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];
    input = (const uint8_t**)((uintptr_t)input + input_increment); // 更新输入数据指针

    if (ks < 2) { // 如果卷积核尺寸小于2，则使用零填充
      i1 = zero;
    }
    if (ks <= 2) { // 如果卷积核尺寸小于等于2，则使用零填充
      i2 = zero;
    }
    if (ks < 4) { // 如果卷积核尺寸小于4，则使用零填充
      i3 = zero;
    }
    if (ks <= 4) { // 如果卷积核尺寸小于等于4，则使用零填充
      i4 = zero;
    }
    if (ks < 6) { // 如果卷积核尺寸小于6，则使用零填充
      i5 = zero;
    }
    if (ks <= 6) { // 如果卷积核尺寸小于等于6，则使用零填充
      i6 = zero;
    }
    if (ks < 8) { // 如果卷积核尺寸小于8，则使用零填充
      i7 = zero;
    }
    if (ks <= 8) { // 如果卷积核尺寸小于等于8，则使用零填充
      i8 = zero;
    }

    size_t k = kc; // 初始化通道数计数器
    // 循环处理每个块，直到剩余块数 k 小于 8
    while (k >= 8) {
      // 加载并解压缩第 0 到第 8 个输入块的前 8 个字节数据
      const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);
      i0 += 8;
      const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);
      i1 += 8;
      const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);
      i2 += 8;
      const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);
      i3 += 8;
      const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);
      i4 += 8;
      const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);
      i5 += 8;
      const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);
      i6 += 8;
      const __m128i vi7 = _mm_loadl_epi64((const __m128i*)i7);
      i7 += 8;
      const __m128i vi8 = _mm_loadl_epi64((const __m128i*)i8);
      i8 += 8;

      // 解压缩每个输入块的字节为 16 位整数，并加上偏置 vzero
      const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
      const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
      const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
      const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
      const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
      const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
      const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
      const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);
      const __m128i vxi8 = _mm_unpacklo_epi8(vi8, vzero);

      // 将每个输入块的解压缩结果进行累加
      const __m128i vsum018 = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1), vxi8);
      const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
      const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
      const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

      // 组合部分累加结果
      const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);
      const __m128i vsum01678 = _mm_add_epi16(vsum018, vsum67);
      const __m128i vsum = _mm_add_epi16(vsum2345, vsum01678);

      // 加上偏置 vbias，并将结果展开为 32 位整数
      const __m128i vacc_lo =
          _mm_add_epi32(vbias, _mm_unpacklo_epi16(vsum, vzero));
      const __m128i vacc_hi =
          _mm_add_epi32(vbias, _mm_unpackhi_epi16(vsum, vzero));

      // 将累加结果转换为单精度浮点数，并乘以缩放因子 vscale
      const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
      const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

      // 将浮点数结果转换回整数类型
      const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
      const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

      // 将结果合并并加上输出的零点偏移量
      __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
      vout = _mm_adds_epi16(
          vout,
          _mm_load_si128(
              (const __m128i*)&quantization_params->sse2.output_zero_point));
      // 将结果截断至无符号 8 位整数范围内
      vout = _mm_packus_epi16(vout, vout);
      vout = _mm_min_epu8(
          vout,
          _mm_load_si128(
              (const __m128i*)&quantization_params->sse2.output_max));
      vout = _mm_max_epu8(
          vout,
          _mm_load_si128(
              (const __m128i*)&quantization_params->sse2.output_min));

      // 将结果存储到输出数组中
      _mm_storel_epi64((__m128i*)output, vout);
      output += 8;

      // 更新剩余块数
      k -= 8;
    }

    // 更新输出指针，以处理下一个输入块序列
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);
}


注释：


# 这行代码表示一个代码块的结束，即一个函数、循环、条件语句或其他代码结构的结束。
# 在此示例中，单独的一个右花括号可能是一个代码块的结尾，但缺少了对应的代码块的开头部分，无法确定其具体作用。
# 通常情况下，右花括号应该与其对应的左花括号一起使用，用于定义代码块的开始和结束。
# 在Python中，这种花括号的用法通常是在其他语言中使用的方式，而Python中的代码块通常是通过缩进来表示的。
```