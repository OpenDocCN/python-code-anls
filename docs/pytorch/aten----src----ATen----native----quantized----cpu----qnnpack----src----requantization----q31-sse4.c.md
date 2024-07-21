# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\q31-sse4.c`

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

#include <smmintrin.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_q31__sse4(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 16 == 0);  // 确保输入大小是16的倍数
  assert(scale < 1.0f);  // 确保缩放因子小于1
  assert(scale >= 0x1.0p-32f);  // 确保缩放因子大于等于2^-32

  /* Compute requantization parameters */
  const uint32_t scale_bits = fp32_to_bits(scale);  // 将浮点数缩放因子转换为32位整数表示

  /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
  const int32_t multiplier = (int32_t)(
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));  // 确保乘数在指定范围内
  assert(multiplier <= INT32_C(0x7FFFFF80));

  /* Shift is in [0, 31] range */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);  // 计算位移量
  assert(shift >= 0);  // 确保位移量非负
  assert(shift < 32);  // 确保位移量小于32

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);  // 创建128位整数向量，每个元素为乘数
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);  // 创建128位整数向量，每个元素为零点
  const __m128i vqmin = _mm_set1_epi8((char)qmin);  // 创建128位整数向量，每个元素为qmin
  const __m128i vqmax = _mm_set1_epi8((char)qmax);  // 创建128位整数向量，每个元素为qmax
  const __m128i vshift = _mm_cvtsi32_si128((int)shift);  // 将位移量转换为128位整数向量
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);  // 计算余数掩码
  const __m128i vremainder_mask = _mm_set1_epi32((int)remainder_mask);  // 创建128位整数向量，每个元素为余数掩码
  const __m128i vthreshold = _mm_set1_epi32((int)(remainder_mask >> 1));  // 创建128位整数向量，每个元素为阈值
  const __m128i vq31rounding = _mm_set1_epi64x(UINT64_C(0x40000000));  // 创建128位整数向量，每个元素为Q31舍入值
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);  // 加载128位整数向量，从输入中读取数据
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;  // 更新输入指针

    const __m128i x_rev = _mm_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1));  // 对x进行逆向字节顺序重排
    const __m128i y_rev = _mm_shuffle_epi32(y, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i z_rev = _mm_shuffle_epi32(z, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i w_rev = _mm_shuffle_epi32(w, _MM_SHUFFLE(2, 3, 0, 1));

    const __m128i x_product_even =
        _mm_add_epi64(_mm_mul_epi32(x, vmultiplier), vq31rounding);  // 计算偶数索引位置的乘积
    const __m128i y_product_even =
        _mm_add_epi64(_mm_mul_epi32(y, vmultiplier), vq31rounding);
    const __m128i z_product_even =
        _mm_add_epi64(_mm_mul_epi32(z, vmultiplier), vq31rounding);
    const __m128i w_product_even =
        _mm_add_epi64(_mm_mul_epi32(w, vmultiplier), vq31rounding);

    const __m128i x_product_odd =
        _mm_add_epi64(_mm_mul_epi32(x_rev, vmultiplier), vq31rounding);  // 计算奇数索引位置的乘积
    const __m128i y_product_odd =
        _mm_add_epi64(_mm_mul_epi32(y_rev, vmultiplier), vq31rounding);
    // 计算奇数位的乘积并加上舍入常量，得到 z_product_odd
    const __m128i z_product_odd =
        _mm_add_epi64(_mm_mul_epi32(z_rev, vmultiplier), vq31rounding);

    // 计算奇数位的乘积并加上舍入常量，得到 w_product_odd
    const __m128i w_product_odd =
        _mm_add_epi64(_mm_mul_epi32(w_rev, vmultiplier), vq31rounding);

    // 对偶数位的乘积进行 Q31 转换，即右移31位，得到 x_q31product_even
    const __m128i x_q31product_even = _mm_srli_epi64(x_product_even, 31);

    // 计算奇数位的乘积加上自身，得到 x_q31product_odd
    const __m128i x_q31product_odd =
        _mm_add_epi64(x_product_odd, x_product_odd);

    // 对偶数位的乘积进行 Q31 转换，得到 y_q31product_even
    const __m128i y_q31product_even = _mm_srli_epi64(y_product_even, 31);

    // 计算奇数位的乘积加上自身，得到 y_q31product_odd
    const __m128i y_q31product_odd =
        _mm_add_epi64(y_product_odd, y_product_odd);

    // 对偶数位的乘积进行 Q31 转换，得到 z_q31product_even
    const __m128i z_q31product_even = _mm_srli_epi64(z_product_even, 31);

    // 计算奇数位的乘积加上自身，得到 z_q31product_odd
    const __m128i z_q31product_odd =
        _mm_add_epi64(z_product_odd, z_product_odd);

    // 对偶数位的乘积进行 Q31 转换，得到 w_q31product_even
    const __m128i w_q31product_even = _mm_srli_epi64(w_product_even, 31);

    // 计算奇数位的乘积加上自身，得到 w_q31product_odd
    const __m128i w_q31product_odd =
        _mm_add_epi64(w_product_odd, w_product_odd);

    // 使用 blend 操作将偶数位和奇数位的结果合并，得到 x_q31product
    const __m128i x_q31product =
        _mm_blend_epi16(x_q31product_even, x_q31product_odd, 0xCC);

    // 使用 blend 操作将偶数位和奇数位的结果合并，得到 y_q31product
    const __m128i y_q31product =
        _mm_blend_epi16(y_q31product_even, y_q31product_odd, 0xCC);

    // 使用 blend 操作将偶数位和奇数位的结果合并，得到 z_q31product
    const __m128i z_q31product =
        _mm_blend_epi16(z_q31product_even, z_q31product_odd, 0xCC);

    // 使用 blend 操作将偶数位和奇数位的结果合并，得到 w_q31product
    const __m128i w_q31product =
        _mm_blend_epi16(w_q31product_even, w_q31product_odd, 0xCC);

    // 计算 x_q31product 与余数掩码的按位与，再与 x_q31product 的符号位进行比较，得到 x_remainder
    const __m128i x_remainder = _mm_add_epi32(
        _mm_and_si128(x_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), x_q31product));

    // 计算 y_q31product 与余数掩码的按位与，再与 y_q31product 的符号位进行比较，得到 y_remainder
    const __m128i y_remainder = _mm_add_epi32(
        _mm_and_si128(y_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), y_q31product));

    // 计算 z_q31product 与余数掩码的按位与，再与 z_q31product 的符号位进行比较，得到 z_remainder
    const __m128i z_remainder = _mm_add_epi32(
        _mm_and_si128(z_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), z_q31product));

    // 计算 w_q31product 与余数掩码的按位与，再与 w_q31product 的符号位进行比较，得到 w_remainder
    const __m128i w_remainder = _mm_add_epi32(
        _mm_and_si128(w_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), w_q31product));

    // 对 x_q31product 进行右移 vshift 位，并与阈值进行比较，得到 x_scaled
    const __m128i x_scaled = _mm_sub_epi32(
        _mm_sra_epi32(x_q31product, vshift),
        _mm_cmpgt_epi32(x_remainder, vthreshold));

    // 对 y_q31product 进行右移 vshift 位，并与阈值进行比较，得到 y_scaled
    const __m128i y_scaled = _mm_sub_epi32(
        _mm_sra_epi32(y_q31product, vshift),
        _mm_cmpgt_epi32(y_remainder, vthreshold));

    // 对 z_q31product 进行右移 vshift 位，并与阈值进行比较，得到 z_scaled
    const __m128i z_scaled = _mm_sub_epi32(
        _mm_sra_epi32(z_q31product, vshift),
        _mm_cmpgt_epi32(z_remainder, vthreshold));

    // 对 w_q31product 进行右移 vshift 位，并与阈值进行比较，得到 w_scaled
    const __m128i w_scaled = _mm_sub_epi32(
        _mm_sra_epi32(w_q31product, vshift),
        _mm_cmpgt_epi32(w_remainder, vthreshold));

    // 将 x_scaled 和 y_scaled 进行 pack，得到 xy_packed
    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);

    // 将 z_scaled 和 w_scaled 进行 pack，得到 zw_packed
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);

    // 将 xy_packed 和 zw_packed 进行更紧凑的 pack，得到 xyzw_packed
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);

    // 将 xyzw_packed 限制在 vqmin 和 vqmax 之间，得到 xyzw_clamped
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);
    /*
     * 执行 SIMD 指令序列，用于向内存中存储 128 位的整数数据
     * 存储位置由 output 指向的内存地址决定
     */
    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    // 将 output 指针向后移动 16 字节，以便指向下一个 128 位的存储位置
    output += 16;
    /*
     * 注：上述代码段使用了 SIMD 指令集来进行高效的数据处理和存储，
     *     _mm_storeu_si128 是 Intel Intrinsics 提供的函数，用于将 128 位数据存储到内存中。
     *     output 是一个指向内存位置的指针，用于确定数据存储的位置。
     */
}


注释：


# 这是一个单独的右括号 '}'，用于结束一个代码块、函数或条件语句的定义。
```