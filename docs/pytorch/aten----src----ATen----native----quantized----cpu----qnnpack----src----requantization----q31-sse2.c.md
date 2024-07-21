# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\q31-sse2.c`

```
/*
 * 该函数用于对一组Q31格式的整数进行重新量化为8位无符号整数，使用SSE2指令集加速计算。
 * 输入参数包括整数数组、缩放因子、零点、输出范围等。
 */
void pytorch_qnnp_requantize_q31__sse2(
    size_t n,                              // 待处理元素的数量，要求是16的倍数
    const int32_t* input,                  // 输入的Q31格式整数数组
    float scale,                           // 缩放因子
    uint8_t zero_point,                    // 量化零点
    uint8_t qmin,                          // 输出的最小量化值
    uint8_t qmax,                          // 输出的最大量化值
    uint8_t* output) {                     // 存放量化后结果的数组
  assert(n % 16 == 0);                     // 断言输入数量是16的倍数
  assert(scale < 1.0f);                    // 断言缩放因子小于1.0
  assert(scale >= 0x1.0p-32f);             // 断言缩放因子不小于2^-32

  /* 计算重新量化所需的参数 */

  // 将浮点数缩放因子转换为32位整数，并调整为乘法器所需的格式
  const uint32_t scale_bits = fp32_to_bits(scale);

  // 根据缩放因子的比特表示计算乘法器值，确保在特定范围内
  const int32_t multiplier = (int32_t)(
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // 计算右移位数，用于乘法结果的向右舍入
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  // 创建用于SSE指令集的常量数据
  const __m128i vmultiplier = _mm_set1_epi32(multiplier);  // 设置乘法器的128位向量
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);  // 设置零点的128位向量
  const __m128i vqmin = _mm_set1_epi8((char)qmin);  // 设置最小量化值的128位向量
  const __m128i vqmax = _mm_set1_epi8((char)qmax);  // 设置最大量化值的128位向量
  const __m128i vshift = _mm_cvtsi32_si128((int)shift);  // 设置右移位数的128位向量
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);  // 计算余数掩码
  const __m128i vremainder_mask = _mm_set1_epi32((int)remainder_mask);  // 设置余数掩码的128位向量
  const __m128i vthreshold = _mm_set1_epi32((int)(remainder_mask >> 1));  // 设置舍入阈值的128位向量
  const __m128i vq31rounding = _mm_set1_epi64x(UINT64_C(0x40000000));  // 设置Q31舍入常量的128位向量

  // 使用SSE2指令集对输入数据进行重新量化
  for (; n != 0; n -= 16) {
    // 加载16个Q31整数到四个128位寄存器中
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    // 计算每个整数的绝对值
    const __m128i x_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), x);
    const __m128i y_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), y);
    const __m128i z_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), z);
    const __m128i w_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), w);
    const __m128i x_abs =
        _mm_sub_epi32(_mm_xor_si128(x, x_neg_mask), x_neg_mask);
    const __m128i y_abs =
        _mm_sub_epi32(_mm_xor_si128(y, y_neg_mask), y_neg_mask);
    const __m128i z_abs =
        _mm_sub_epi32(_mm_xor_si128(z, z_neg_mask), z_neg_mask);
    const __m128i w_abs =
        _mm_sub_epi32(_mm_xor_si128(w, w_neg_mask), w_neg_mask);

    // 反转整数的字节顺序
    const __m128i x_abs_rev = _mm_shuffle_epi32(x_abs, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i y_abs_rev = _mm_shuffle_epi32(y_abs, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i z_abs_rev = _mm_shuffle_epi32(z_abs, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i w_abs_rev = _mm_shuffle_epi32(w_abs, _MM_SHUFFLE(2, 3, 0, 1));

    // 接下来的SSE2指令用于将每个输入Q31整数重新量化为8位无符号整数


This completes the annotation of the given C code snippet following the specified format and guidelines.
    // 计算 x_abs 和 vmultiplier 的无符号整数乘积
    const __m128i x_abs_product_even = _mm_mul_epu32(x_abs, vmultiplier);
    // 计算 y_abs 和 vmultiplier 的无符号整数乘积
    const __m128i y_abs_product_even = _mm_mul_epu32(y_abs, vmultiplier);
    // 计算 z_abs 和 vmultiplier 的无符号整数乘积
    const __m128i z_abs_product_even = _mm_mul_epu32(z_abs, vmultiplier);
    // 计算 w_abs 和 vmultiplier 的无符号整数乘积
    const __m128i w_abs_product_even = _mm_mul_epu32(w_abs, vmultiplier);

    // 通过按位混洗，生成 x_neg_mask 的偶数位掩码
    const __m128i x_neg_mask_even =
        _mm_shuffle_epi32(x_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    // 通过按位混洗，生成 y_neg_mask 的偶数位掩码
    const __m128i y_neg_mask_even =
        _mm_shuffle_epi32(y_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    // 通过按位混洗，生成 z_neg_mask 的偶数位掩码
    const __m128i z_neg_mask_even =
        _mm_shuffle_epi32(z_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    // 通过按位混洗，生成 w_neg_mask 的偶数位掩码
    const __m128i w_neg_mask_even =
        _mm_shuffle_epi32(w_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));

    // 计算 x_product_even，即通过按位异或和减法来生成 x_abs_product_even 和 x_neg_mask_even 的差值
    const __m128i x_product_even = _mm_sub_epi64(
        _mm_xor_si128(x_abs_product_even, x_neg_mask_even), x_neg_mask_even);
    // 计算 y_product_even，即通过按位异或和减法来生成 y_abs_product_even 和 y_neg_mask_even 的差值
    const __m128i y_product_even = _mm_sub_epi64(
        _mm_xor_si128(y_abs_product_even, y_neg_mask_even), y_neg_mask_even);
    // 计算 z_product_even，即通过按位异或和减法来生成 z_abs_product_even 和 z_neg_mask_even 的差值
    const __m128i z_product_even = _mm_sub_epi64(
        _mm_xor_si128(z_abs_product_even, z_neg_mask_even), z_neg_mask_even);
    // 计算 w_product_even，即通过按位异或和减法来生成 w_abs_product_even 和 w_neg_mask_even 的差值
    const __m128i w_product_even = _mm_sub_epi64(
        _mm_xor_si128(w_abs_product_even, w_neg_mask_even), w_neg_mask_even);

    // 将 x_product_even 结果加上 vq31rounding，实现向最近整数的舍入
    const __m128i x_rounded_product_even =
        _mm_add_epi64(x_product_even, vq31rounding);
    // 将 y_product_even 结果加上 vq31rounding，实现向最近整数的舍入
    const __m128i y_rounded_product_even =
        _mm_add_epi64(y_product_even, vq31rounding);
    // 将 z_product_even 结果加上 vq31rounding，实现向最近整数的舍入
    const __m128i z_rounded_product_even =
        _mm_add_epi64(z_product_even, vq31rounding);
    // 将 w_product_even 结果加上 vq31rounding，实现向最近整数的舍入
    const __m128i w_rounded_product_even =
        _mm_add_epi64(w_product_even, vq31rounding);

    // 计算 x_abs_rev 和 vmultiplier 的无符号整数乘积
    const __m128i x_abs_product_odd = _mm_mul_epu32(x_abs_rev, vmultiplier);
    // 计算 y_abs_rev 和 vmultiplier 的无符号整数乘积
    const __m128i y_abs_product_odd = _mm_mul_epu32(y_abs_rev, vmultiplier);
    // 计算 z_abs_rev 和 vmultiplier 的无符号整数乘积
    const __m128i z_abs_product_odd = _mm_mul_epu32(z_abs_rev, vmultiplier);
    // 计算 w_abs_rev 和 vmultiplier 的无符号整数乘积
    const __m128i w_abs_product_odd = _mm_mul_epu32(w_abs_rev, vmultiplier);

    // 通过按位混洗，生成 x_neg_mask 的奇数位掩码
    const __m128i x_neg_mask_odd =
        _mm_shuffle_epi32(x_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    // 通过按位混洗，生成 y_neg_mask 的奇数位掩码
    const __m128i y_neg_mask_odd =
        _mm_shuffle_epi32(y_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    // 通过按位混洗，生成 z_neg_mask 的奇数位掩码
    const __m128i z_neg_mask_odd =
        _mm_shuffle_epi32(z_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    // 通过按位混洗，生成 w_neg_mask 的奇数位掩码
    const __m128i w_neg_mask_odd =
        _mm_shuffle_epi32(w_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));

    // 计算 x_product_odd，即通过按位异或和减法来生成 x_abs_product_odd 和 x_neg_mask_odd 的差值
    const __m128i x_product_odd = _mm_sub_epi64(
        _mm_xor_si128(x_abs_product_odd, x_neg_mask_odd), x_neg_mask_odd);
    // 计算 y_product_odd，即通过按位异或和减法来生成 y_abs_product_odd 和 y_neg_mask_odd 的差值
    const __m128i y_product_odd = _mm_sub_epi64(
        _mm_xor_si128(y_abs_product_odd, y_neg_mask_odd), y_neg_mask_odd);
    // 计算 z_product_odd，即通过按位异或和减法来生成 z_abs_product_odd 和 z_neg_mask_odd 的差值
    const __m128i z_product_odd = _mm_sub_epi64(
        _mm_xor_si128(z_abs_product_odd, z_neg_mask_odd), z_neg_mask_odd);
    // 计算 w_product_odd，即通过按位异或和减法来生成 w_abs_product_odd 和 w_neg_mask_odd 的差值
    const __m128i w_product_odd = _mm_sub_epi64(
        _mm_xor_si128(w_abs_product_odd, w_neg_mask_odd), w_neg_mask_odd);

    // 将 x_product_odd 结果加上 vq31rounding，实现向最近整数的舍入
    const __m128i x_rounded_product_odd =
        _mm_add_epi64(x_product_odd, vq31rounding);
    // 将 y_product_odd 结果加上 vq31rounding，实现向最近整数的舍入
    const __m128i y_rounded_product_odd =
        _mm_add_epi64(y_product_odd, vq31rounding);
    // 计算奇数位产品的四舍五入结果，加上 Q31 的四舍五入值
    const __m128i z_rounded_product_odd =
        _mm_add_epi64(z_product_odd, vq31rounding);
    const __m128i w_rounded_product_odd =
        _mm_add_epi64(w_product_odd, vq31rounding);

    // 对偶数位产品进行 Q31 右移，得到 Q31 格式的结果
    const __m128i x_q31product_even =
        _mm_srli_epi64(x_rounded_product_even, 31);
    const __m128i x_q31product_odd = _mm_srli_epi64(x_rounded_product_odd, 31);
    const __m128i y_q31product_even =
        _mm_srli_epi64(y_rounded_product_even, 31);
    const __m128i y_q31product_odd = _mm_srli_epi64(y_rounded_product_odd, 31);
    const __m128i z_q31product_even =
        _mm_srli_epi64(z_rounded_product_even, 31);
    const __m128i z_q31product_odd = _mm_srli_epi64(z_rounded_product_odd, 31);
    const __m128i w_q31product_even =
        _mm_srli_epi64(w_rounded_product_even, 31);
    const __m128i w_q31product_odd = _mm_srli_epi64(w_rounded_product_odd, 31);

    // 对 Q31 格式的结果进行重新排序，得到按顺序 0, 2, 1, 3 的结果
    const __m128i x_q31product_0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x_q31product_even),
        _mm_castsi128_ps(x_q31product_odd),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i y_q31product_0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(y_q31product_even),
        _mm_castsi128_ps(y_q31product_odd),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i z_q31product_0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(z_q31product_even),
        _mm_castsi128_ps(z_q31product_odd),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i w_q31product_0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(w_q31product_even),
        _mm_castsi128_ps(w_q31product_odd),
        _MM_SHUFFLE(2, 0, 2, 0)));

    // 对重新排序的 Q31 结果进行进一步重排，得到按顺序 3, 1, 2, 0 的结果
    const __m128i x_q31product =
        _mm_shuffle_epi32(x_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i y_q31product =
        _mm_shuffle_epi32(y_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i z_q31product =
        _mm_shuffle_epi32(z_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i w_q31product =
        _mm_shuffle_epi32(w_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));

    // 计算 Q31 结果的余数部分，并根据阈值进行修正
    const __m128i x_remainder = _mm_add_epi32(
        _mm_and_si128(x_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), x_q31product));
    const __m128i y_remainder = _mm_add_epi32(
        _mm_and_si128(y_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), y_q31product));
    const __m128i z_remainder = _mm_add_epi32(
        _mm_and_si128(z_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), z_q31product));
    const __m128i w_remainder = _mm_add_epi32(
        _mm_and_si128(w_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), w_q31product));

    // 对修正后的 Q31 结果进行缩放，根据移位量进行右移操作
    const __m128i x_scaled = _mm_sub_epi32(
        _mm_sra_epi32(x_q31product, vshift),
        _mm_cmpgt_epi32(x_remainder, vthreshold));
    const __m128i y_scaled = _mm_sub_epi32(
        _mm_sra_epi32(y_q31product, vshift),
        _mm_cmpgt_epi32(y_remainder, vthreshold));
    const __m128i z_scaled = _mm_sub_epi32(
        _mm_sra_epi32(z_q31product, vshift),
        _mm_cmpgt_epi32(z_remainder, vthreshold));
    // 计算 z_scaled，将 z_q31product 右移 vshift 位，然后减去 z_remainder 大于 vthreshold 的掩码值

    const __m128i w_scaled = _mm_sub_epi32(
        _mm_sra_epi32(w_q31product, vshift),
        _mm_cmpgt_epi32(w_remainder, vthreshold));
    // 计算 w_scaled，将 w_q31product 右移 vshift 位，然后减去 w_remainder 大于 vthreshold 的掩码值

    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    // 将 x_scaled 和 y_scaled 向下转换为 16 位有符号整数，并加上 vzero_point

    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    // 将 z_scaled 和 w_scaled 向下转换为 16 位有符号整数，并加上 vzero_point

    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    // 将 xy_packed 和 zw_packed 合并为一个 64 位无符号整数，并压缩为 8 位无符号整数

    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);
    // 将 xyzw_packed 中的值限制在 vqmin 和 vqmax 之间，并取最大值

    /*
     * 16x PSHUFD
     * 4x SHUFPS
     * 8x PMULUDQ
     * 8x PXOR (setzero)
     * 12x PXOR
     * 4x PAND
     * 8x PADDQ
     * 4x PADDD
     * 2x PADDW
     * 8x PSUBQ
     * 8x PSUBD
     * 8x PSRLQ (immediate)
     * 4x PSRAD (register)
     * 12x PCMPGTD
     * 2x PACKSSDW
     * 1x PACKUSWB
     * 1x PMAXUB
     * 1x PMINUB
     * ---------------------
     * 111 instructions total
     */
    
    // 将 xyzw_clamped 中的值存储到 output 指向的内存地址，以 16 字节为单位存储
    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    // 更新 output 指针，指向下一个 16 字节位置
    output += 16;
}


注释：


# 这行代码关闭了一个代码块，通常用于结束一个函数或者控制流语句（如if、for等）
```