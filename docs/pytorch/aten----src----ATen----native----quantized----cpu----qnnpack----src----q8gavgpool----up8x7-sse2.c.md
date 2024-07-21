# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gavgpool\up8x7-sse2.c`

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

void pytorch_q8gavgpool_ukernel_up8x7__sse2(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(m >= 1);
  assert(m <= 7);
  assert(n >= 8);

  const uint8_t* i0 = input;  // 初始化指向输入数据的指针 i0
  const uint8_t* i1 = i0 + input_stride;  // 指向第二行数据的指针 i1
  if (m < 2) {
    i1 = zero;  // 如果 m 小于 2，则 i1 指向 zero
  }
  const uint8_t* i2 = i1 + input_stride;  // 指向第三行数据的指针 i2
  if (m <= 2) {
    i2 = zero;  // 如果 m 小于等于 2，则 i2 指向 zero
  }
  const uint8_t* i3 = i2 + input_stride;  // 指向第四行数据的指针 i3
  if (m < 4) {
    i3 = zero;  // 如果 m 小于 4，则 i3 指向 zero
  }
  const uint8_t* i4 = i3 + input_stride;  // 指向第五行数据的指针 i4
  if (m <= 4) {
    i4 = zero;  // 如果 m 小于等于 4，则 i4 指向 zero
  }
  const uint8_t* i5 = i4 + input_stride;  // 指向第六行数据的指针 i5
  if (m < 6) {
    i5 = zero;  // 如果 m 小于 6，则 i5 指向 zero
  }
  const uint8_t* i6 = i5 + input_stride;  // 指向第七行数据的指针 i6
  if (m <= 6) {
    i6 = zero;  // 如果 m 小于等于 6，则 i6 指向 zero
  }
  const __m128i vbias =
      _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);  // 载入偏置向量
  const __m128i vzero = _mm_setzero_si128();  // 初始化零向量

  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);  // 载入缩放向量

  do {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);  // 载入第一行数据的前8个字节
    i0 += 8;  // 更新指针，指向下一个8字节数据
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);  // 载入第二行数据的前8个字节
    i1 += 8;  // 更新指针，指向下一个8字节数据
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);  // 载入第三行数据的前8个字节
    i2 += 8;  // 更新指针，指向下一个8字节数据
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);  // 载入第四行数据的前8个字节
    i3 += 8;  // 更新指针，指向下一个8字节数据
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);  // 载入第五行数据的前8个字节
    i4 += 8;  // 更新指针，指向下一个8字节数据
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);  // 载入第六行数据的前8个字节
    i5 += 8;  // 更新指针，指向下一个8字节数据
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);  // 载入第七行数据的前8个字节
    i6 += 8;  // 更新指针，指向下一个8字节数据

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);  // 将 vi0 每个字节拆分为低位和高位，与零向量合并
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);  // 将 vi1 每个字节拆分为低位和高位，与零向量合并
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);  // 将 vi2 每个字节拆分为低位和高位，与零向量合并
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);  // 将 vi3 每个字节拆分为低位和高位，与零向量合并
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);  // 将 vi4 每个字节拆分为低位和高位，与零向量合并
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);  // 将 vi5 每个字节拆分为低位和高位，与零向量合并
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);  // 将 vi6 每个字节拆分为低位和高位，与零向量合并

    // 计算累加和，每次处理两个元素
    __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vxi0, vzero));
    __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    // 将 vxi5 和 vzero 进行低位和高位拆分、拆包并加到 vacc_lo 和 vacc_hi 中
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    // 将 vxi6 和 vzero 进行低位和高位拆分、拆包并加到 vacc_lo 和 vacc_hi 中
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    // 将 vacc_lo 中的整数值转换为浮点数，然后乘以 vscale 得到 vacc_lo_f
    const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
    // 将 vacc_hi 中的整数值转换为浮点数，然后乘以 vscale 得到 vacc_hi_f
    const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

    // 将浮点数值 vacc_lo_f 转换为整数，得到 vscaled_lo
    const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
    // 将浮点数值 vacc_hi_f 转换为整数，得到 vscaled_hi
    const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

    // 将 vscaled_lo 和 vscaled_hi 合并为一个紧凑的结果向量 vout
    __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
    // 将 vout 向量中的每个元素加上输出的零点偏移量
    vout = _mm_adds_epi16(
        vout,
        _mm_load_si128(
            (const __m128i*)quantization_params->sse2.output_zero_point));
    // 将 vout 向量中的元素压缩到 8 位无符号整数
    vout = _mm_packus_epi16(vout, vout);
    // 将 vout 中的每个元素与输出的最大值进行比较，取较小值
    vout = _mm_min_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
    // 将 vout 中的每个元素与输出的最小值进行比较，取较大值
    vout = _mm_max_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

    // 将 vout 中的值存储到 output 指向的地址处，每次存储 8 个元素
    _mm_storel_epi64((__m128i*)output, vout);
    // 更新 output 指针，使其指向下一个位置
    output += 8;

    // 更新剩余处理元素的数量 n
    n -= 8;
  } while (n >= 8);

  // 如果剩余的元素数量 n 不为零，则处理剩余的不足 8 个元素
  if (n != 0) {
    // 计算地址偏移量，用于调整输入指针 i0 至 i6
    const size_t address_decrement = 8 - n;
    i0 = (const uint8_t*)((uintptr_t)i0 - address_decrement);
    i1 = (const uint8_t*)((uintptr_t)i1 - address_decrement);
    i2 = (const uint8_t*)((uintptr_t)i2 - address_decrement);
    i3 = (const uint8_t*)((uintptr_t)i3 - address_decrement);
    i4 = (const uint8_t*)((uintptr_t)i4 - address_decrement);
    i5 = (const uint8_t*)((uintptr_t)i5 - address_decrement);
    i6 = (const uint8_t*)((uintptr_t)i6 - address_decrement);
    // 构建位移量向量 vi_shift，用于右移 8 * address_decrement 位
    const __m128i vi_shift = _mm_cvtsi32_si128(8 * address_decrement);

    // 加载并右移输入指针 i0 至 i6 指向的数据，得到 vi0 至 vi6
    const __m128i vi0 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i0), vi_shift);
    const __m128i vi1 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i1), vi_shift);
    const __m128i vi2 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i2), vi_shift);
    const __m128i vi3 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i3), vi_shift);
    const __m128i vi4 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i4), vi_shift);
    const __m128i vi5 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i5), vi_shift);
    const __m128i vi6 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i6), vi_shift);

    // 将 vi0 至 vi6 中的每个值与 vzero 进行拆分和拆包，得到 vxi0 至 vxi6
    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    // 将 vbias 与 vxi0 的低位拆包结果相加，得到 vacc_lo
    __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vxi0, vzero));
    // 将 vbias 与 vxi0 的高位拆包结果相加，得到 vacc_hi
    __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vxi0, vzero));
    // 将 vacc_lo 与 vxi1 的低位拆包结果相加，得到更新后的 vacc_lo
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));


这段代码是使用 SSE2 指令集优化的向量化处理代码段，主要用于图像处理或者数字信号处理中的数据量化和压缩操作。
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_hi 中
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_lo 中
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_hi 中
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_lo 中
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_hi 中
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_lo 中
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_hi 中
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_lo 中
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_hi 中
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_lo 中
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    # 使用 SIMD 指令对两个 128 位整数进行加法运算，结果存储在 vacc_hi 中
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    # 将 vacc_lo 中的整数转换为单精度浮点数，再与 vscale 相乘
    const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
    # 将 vacc_hi 中的整数转换为单精度浮点数，再与 vscale 相乘
    const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

    # 将单精度浮点数转换为整数，结果存储在 vscaled_lo 中
    const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
    # 将单精度浮点数转换为整数，结果存储在 vscaled_hi 中
    const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

    # 将两个 128 位整数打包成 16 位整数，结果存储在 vout 中
    __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
    # 将 vout 中的整数值与输出的零点进行加法运算
    vout = _mm_adds_epi16(
        vout,
        _mm_load_si128(
            (const __m128i*)quantization_params->sse2.output_zero_point));
    # 将 vout 中的整数打包成 16 位无符号整数
    vout = _mm_packus_epi16(vout, vout);
    # 将 vout 中的值限制在输出上限内
    vout = _mm_min_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
    # 将 vout 中的值限制在输出下限内
    vout = _mm_max_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

    # 如果 n & 4 不为零，则将 vout 转换为 uint32_t 存储到 output 中，并更新 output 指针
    if (n & 4) {
      *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
      output += 4;
      vout = _mm_srli_epi64(vout, 32);
    }
    # 如果 n & 2 不为零，则将 vout 转换为 uint16_t 存储到 output 中，并更新 output 指针
    if (n & 2) {
      *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
      output += 2;
      vout = _mm_srli_epi32(vout, 16);
    }
    # 如果 n & 1 不为零，则将 vout 转换为 uint8_t 存储到 output 中
    if (n & 1) {
      *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
    }
}



# 这行代码关闭了一个代码块，与之前的 "{" 配对，表示一个代码块的结束。
```