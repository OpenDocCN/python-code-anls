# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gavgpool\mp8x7p7q-sse2.c`

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

// 定义了一个名为 pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2 的函数
void pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2(
    size_t m,  // 行数 m
    size_t n,  // 列数 n
    const uint8_t* input,  // 输入数据数组的指针
    size_t input_stride,  // 输入数据的行步长
    const uint8_t* zero,  // 用于填充的零值数组的指针
    int32_t* buffer,  // 缓冲区数组的指针，用于累加操作
    uint8_t* output,  // 输出数据数组的指针
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {  // 包含量化参数的结构体数组
  assert(m > 7);  // 断言 m 大于 7
  assert(n >= 8);  // 断言 n 大于等于 8

  const uint8_t* i0 = input;  // 初始化输入数据的第一行指针
  const uint8_t* i1 = i0 + input_stride;  // 初始化输入数据的第二行指针
  const uint8_t* i2 = i1 + input_stride;  // 初始化输入数据的第三行指针
  const uint8_t* i3 = i2 + input_stride;  // 初始化输入数据的第四行指针
  const uint8_t* i4 = i3 + input_stride;  // 初始化输入数据的第五行指针
  const uint8_t* i5 = i4 + input_stride;  // 初始化输入数据的第六行指针
  const uint8_t* i6 = i5 + input_stride;  // 初始化输入数据的第七行指针
  const size_t packed_n = (n + 7) & -8;  // 计算对齐到 8 的 n 的大小
  const size_t input_increment = 7 * input_stride - packed_n;  // 计算输入数据指针增量
  const __m128i vbias =
      _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);  // 加载偏置值
  const __m128i vzero = _mm_setzero_si128();  // 设置一个全零的 SSE 寄存器

  // 主循环，每次处理 8 列数据
  for (size_t k = 0; k < n; k += 8) {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);  // 加载第 0 行数据
    i0 += 8;  // 更新第 0 行指针
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);  // 加载第 1 行数据
    i1 += 8;  // 更新第 1 行指针
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);  // 加载第 2 行数据
    i2 += 8;  // 更新第 2 行指针
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);  // 加载第 3 行数据
    i3 += 8;  // 更新第 3 行指针
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);  // 加载第 4 行数据
    i4 += 8;  // 更新第 4 行指针
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);  // 加载第 5 行数据
    i5 += 8;  // 更新第 5 行指针
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);  // 加载第 6 行数据
    i6 += 8;  // 更新第 6 行指针

    // 解压缩数据到 16 位并累加到偏置值上
    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    // 累加到 32 位整数，并加上偏置值
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
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    // 将两个包含有16位整数的寄存器解压缩并将结果与高位累加到vacc_hi
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    // 将两个包含有16位整数的寄存器解压缩并将结果与低位累加到vacc_lo
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    // 将两个包含有16位整数的寄存器解压缩并将结果与高位累加到vacc_hi
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    // 将寄存器中的数据存储到内存中的acc数组中
    _mm_store_si128((__m128i*)acc, vacc_lo);
    // 将寄存器中的数据存储到内存中的acc数组的下一个位置
    _mm_store_si128((__m128i*)acc + 1, vacc_hi);
    // 将acc指针向后移动8个元素的位置
    acc += 8;
  }
  // 对于剩余的m，循环执行直到m小于等于7为止
  for (m -= 7; m > 7; m -= 7) {
    // 将各个输入指针向后移动input_increment个字节
    acc = buffer;
    i0 = (const uint8_t*)((uintptr_t)i0 + input_increment);
    i1 = (const uint8_t*)((uintptr_t)i1 + input_increment);
    i2 = (const uint8_t*)((uintptr_t)i2 + input_increment);
    i3 = (const uint8_t*)((uintptr_t)i3 + input_increment);
    i4 = (const uint8_t*)((uintptr_t)i4 + input_increment);
    i5 = (const uint8_t*)((uintptr_t)i5 + input_increment);
    i6 = (const uint8_t*)((uintptr_t)i6 + input_increment);

    /* 注意：超出7个元素的边界 */
    for (size_t k = 0; k < n; k += 8) {
      // 逐次加载8个字节长度的数据到__m128i类型变量vi0-vi6中，i0-i6指针递增
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
      
      // 分别加载两个128位寄存器中的累加器数据
      __m128i vacc_lo = _mm_load_si128((const __m128i*)acc);
      __m128i vacc_hi = _mm_load_si128((const __m128i*)acc + 1);

      // 解压缩vi0-vi6中的数据到32位数据中，进行累加到vacc_lo和vacc_hi中
      const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
      const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
      const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
      const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
      const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
      const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
      const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

      // 按照元素进行32位整数相加操作
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

      // 将累加结果存储回acc中
      _mm_store_si128((__m128i*)acc, vacc_lo);
      _mm_store_si128((__m128i*)acc + 1, vacc_hi);
      // acc指针递增8个字节
      acc += 8;
    }
  }

  // 加载量化参数的缩放因子
  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

  // 递增i0-i6指针，根据m的值选择性地将部分指针设为zero
  i0 = (const uint8_t*)((uintptr_t)i0 + input_increment);
  i1 = (const uint8_t*)((uintptr_t)i1 + input_increment);
  if (m < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*)((uintptr_t)i2 + input_increment);
  if (m <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*)((uintptr_t)i3 + input_increment);
  if (m < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*)((uintptr_t)i4 + input_increment);
  if (m <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*)((uintptr_t)i5 + input_increment);
  if (m < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*)((uintptr_t)i6 + input_increment);
  if (m <= 6) {
    i6 = zero;
  }

  // 重置acc指针为buffer起始位置
  acc = buffer;
  // 开始执行循环处理
  do {
    // 从地址 i0 加载一个 64 位的整数到寄存器 vi0 中
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);
    // 增加 i0 的地址偏移量，移动到下一个 64 位整数的位置
    i0 += 8;

    // 依此类推，加载 i1 到 vi1，增加 i1 的地址偏移量
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);
    i1 += 8;

    // 加载 i2 到 vi2，增加 i2 的地址偏移量
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);
    i2 += 8;

    // 加载 i3 到 vi3，增加 i3 的地址偏移量
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);
    i3 += 8;

    // 加载 i4 到 vi4，增加 i4 的地址偏移量
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);
    i4 += 8;

    // 加载 i5 到 vi5，增加 i5 的地址偏移量
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);
    i5 += 8;

    // 加载 i6 到 vi6，增加 i6 的地址偏移量
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);
    i6 += 8;

    // 加载累加器的低 128 位数据到 vacc_lo
    __m128i vacc_lo = _mm_load_si128((const __m128i*)acc);
    // 加载累加器的高 128 位数据到 vacc_hi
    __m128i vacc_hi = _mm_load_si128((const __m128i*)acc + 1);
    // 增加 acc 的地址偏移量，移动到累加器下一个位置
    acc += 8;

    // 将 vi0 到 vi6 中的每个值与 vzero 执行无符号字节展开
    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    // 将展开后的数据按照 16 位进行展开并与累加器的低/高位相加
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    // 将累加器中的整数转换为浮点数并乘以 vscale
    const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
    const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

    // 将浮点数转换回整数
    const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
    const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

    // 将 vscaled_lo 和 vscaled_hi 打包成 16 位整数
    __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);

    // 加载输出量化参数的零点，对 vout 进行调整和限制范围
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

    // 将 vout 存储到输出中
    _mm_storel_epi64((__m128i*)output, vout);
    output += 8;

    // 减少剩余的 n 值，检查是否还有剩余的元素需要处理
    n -= 8;
  } while (n >= 8);

  // 如果 n 不等于 0，计算地址的减量，这将处理剩余不足 8 的情况
  if (n != 0) {
    const size_t address_decrement = 8 - n;
    // 减去指定地址的偏移量，转换为指向常量 uint8_t 类型的指针
    i0 = (const uint8_t*)((uintptr_t)i0 - address_decrement);
    i1 = (const uint8_t*)((uintptr_t)i1 - address_decrement);
    i2 = (const uint8_t*)((uintptr_t)i2 - address_decrement);
    i3 = (const uint8_t*)((uintptr_t)i3 - address_decrement);
    i4 = (const uint8_t*)((uintptr_t)i4 - address_decrement);
    i5 = (const uint8_t*)((uintptr_t)i5 - address_decrement);
    i6 = (const uint8_t*)((uintptr_t)i6 - address_decrement);

    // 将 address_decrement 乘以 8 后，将结果转换为 __m128i 类型，用于后续的移位操作
    const __m128i vi_shift = _mm_cvtsi32_si128(8 * address_decrement);

    // 加载 i0-i6 指向的内存内容到 __m128i 类型的变量 vi0-vi6，并对其进行右移操作
    const __m128i vi0 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i0), vi_shift);
    const __m128i vi1 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i1), vi_shift);
    const __m128i vi2 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i2), vi_shift);
    const __m128i vi3 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i3), vi_shift);
    const __m128i vi4 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i4), vi_shift);
    const __m128i vi5 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i5), vi_shift);
    const __m128i vi6 = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i6), vi_shift);

    // 从存储器加载 __m128i 类型的变量 acc 的值到 vacc_lo 和 vacc_hi
    __m128i vacc_lo = _mm_load_si128((const __m128i*)acc);
    __m128i vacc_hi = _mm_load_si128((const __m128i*)acc + 1);

    // 将 vi0-vi6 进行字节解压，将其拆分为更细粒度的数据，与零填充（vzero）结合
    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    // 将 vxi0-vxi6 拆分为 16 位数据，与零填充结合，然后与 vacc_lo 和 vacc_hi 相加
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi0, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

    // 将 vacc_lo 和 vacc_hi 中的每个元素转换为单精度浮点数，然后乘以 vscale 得到浮点数结果
    const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
    const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

    // 将浮点数结果 vscaled_lo 和 vscaled_hi 转换为整数
    const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
    const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);
    # 将两组32位整数转换为16位整数，并将结果打包到一个128位寄存器中
    __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
    
    # 将vout中的每个16位整数加上对应的输出零点值，并存储结果到vout
    vout = _mm_adds_epi16(
        vout,
        _mm_load_si128(
            (const __m128i*)quantization_params->sse2.output_zero_point));
    
    # 将vout中的16位整数打包成8位整数，并存储结果到vout
    vout = _mm_packus_epi16(vout, vout);
    
    # 将vout中的每个8位整数与输出最大值向量逐元素比较，将每个元素取最小值
    vout = _mm_min_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
    
    # 将vout中的每个8位整数与输出最小值向量逐元素比较，将每个元素取最大值
    vout = _mm_max_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));
    
    # 根据n的低2位判断是否需要处理4字节数据
    if (n & 4) {
      # 将vout的高32位存储到output指向的内存中，并向后移动output指针4字节
      *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
      output += 4;
      # 将vout逻辑右移32位，处理下一组数据
      vout = _mm_srli_epi64(vout, 32);
    }
    
    # 根据n的低1位判断是否需要处理2字节数据
    if (n & 2) {
      # 将vout的第一个16位整数存储到output指向的内存中，并向后移动output指针2字节
      *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
      output += 2;
      # 将vout逻辑右移16位，处理下一组数据
      vout = _mm_srli_epi32(vout, 16);
    }
    
    # 根据n的最低位判断是否需要处理1字节数据
    if (n & 1) {
      # 将vout的低32位整数存储到output指向的内存中
      *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
    }
}



# 这是一个单独的代码行，只包含一个右花括号 '}'，用于结束某个代码块或语句的作用域。
```