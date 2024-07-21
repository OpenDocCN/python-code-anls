# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\q31-ssse3.c`

```
  /*
   * 根据 Facebook, Inc. 及其附属公司的版权声明。
   * 保留所有权利。
   *
   * 此源代码使用 BSD 风格许可证，可以在源树的根目录下的 LICENSE 文件中找到。
   */

  #include <assert.h>
  #include <stdint.h>

  #include <tmmintrin.h>

  #include <fp16/bitcasts.h>
  #include <qnnpack/requantization-stubs.h>

  void pytorch_qnnp_requantize_q31__ssse3(
      size_t n,
      const int32_t* input,
      float scale,
      uint8_t zero_point,
      uint8_t qmin,
      uint8_t qmax,
      uint8_t* output) {
    assert(n % 16 == 0);  // 确保 n 是 16 的倍数
    assert(scale < 1.0f);  // 确保 scale 小于 1.0
    assert(scale >= 0x1.0p-32f);  // 确保 scale 大于等于 2^-32

    /* 计算重新量化的参数 */
    const uint32_t scale_bits = fp32_to_bits(scale);

    /* multiplier 在 [0x40000000, 0x7FFFFF80] 范围内 */
    const int32_t multiplier = (int32_t)(
        ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
    assert(multiplier >= INT32_C(0x40000000));
    assert(multiplier <= INT32_C(0x7FFFFF80));

    /* shift 在 [0, 31] 范围内 */
    const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
    assert(shift >= 0);
    assert(shift < 32);

    const __m128i vmultiplier = _mm_set1_epi32(multiplier);  // 将 multiplier 载入到 SSE 寄存器
    const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);  // 将 zero_point 载入到 SSE 寄存器
    const __m128i vqmin = _mm_set1_epi8((char)qmin);  // 将 qmin 载入到 SSE 寄存器
    const __m128i vqmax = _mm_set1_epi8((char)qmax);  // 将 qmax 载入到 SSE 寄存器
    const __m128i vshift = _mm_cvtsi32_si128((int)shift);  // 将 shift 载入到 SSE 寄存器
    const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
    const __m128i vremainder_mask = _mm_set1_epi32((int)remainder_mask);  // 将 remainder_mask 载入到 SSE 寄存器
    const __m128i vthreshold = _mm_set1_epi32((int)(remainder_mask >> 1));  // 将 threshold 载入到 SSE 寄存器
    const __m128i vq31rounding = _mm_set1_epi64x(UINT64_C(0x40000000));  // 将 Q31 舍入常数载入 SSE 寄存器

    for (; n != 0; n -= 16) {
      const __m128i x = _mm_loadu_si128((const __m128i*)input);  // 加载 input 到 SSE 寄存器 x
      const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));  // 加载 input + 4 到 SSE 寄存器 y
      const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));  // 加载 input + 8 到 SSE 寄存器 z
      const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));  // 加载 input + 12 到 SSE 寄存器 w
      input += 16;

      const __m128i x_abs = _mm_abs_epi32(x);  // 计算 x 的绝对值
      const __m128i y_abs = _mm_abs_epi32(y);  // 计算 y 的绝对值
      const __m128i z_abs = _mm_abs_epi32(z);  // 计算 z 的绝对值
      const __m128i w_abs = _mm_abs_epi32(w);  // 计算 w 的绝对值

      const __m128i x_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), x);  // 比较 x 是否小于零
      const __m128i y_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), y);  // 比较 y 是否小于零
      const __m128i z_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), z);  // 比较 z 是否小于零
      const __m128i w_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), w);  // 比较 w 是否小于零

      const __m128i x_abs_rev = _mm_shuffle_epi32(x_abs, _MM_SHUFFLE(2, 3, 0, 1));  // 将 x_abs 反转
      const __m128i y_abs_rev = _mm_shuffle_epi32(y_abs, _MM_SHUFFLE(2, 3, 0, 1));  // 将 y_abs 反转
      const __m128i z_abs_rev = _mm_shuffle_epi32(z_abs, _MM_SHUFFLE(2, 3, 0, 1));  // 将 z_abs 反转
      const __m128i w_abs_rev = _mm_shuffle_epi32(w_abs, _MM_SHUFFLE(2, 3, 0, 1));  // 将 w_abs 反转

      const __m128i x_abs_product_even = _mm_mul_epu32(x_abs, vmultiplier);  // 计算 x_abs 乘以 vmultiplier 的偶数部分
      const __m128i y_abs_product_even = _mm_mul_epu32(y_abs, vmultiplier);  // 计算 y_abs 乘以 vmultiplier 的偶数部分
    # 计算 z_abs 与 vmultiplier 的无符号整数乘法结果，存储在 z_abs_product_even 中
    const __m128i z_abs_product_even = _mm_mul_epu32(z_abs, vmultiplier);
    # 计算 w_abs 与 vmultiplier 的无符号整数乘法结果，存储在 w_abs_product_even 中
    const __m128i w_abs_product_even = _mm_mul_epu32(w_abs, vmultiplier);

    # 从 x_neg_mask 中提取出偶数位，存储在 x_neg_mask_even 中
    const __m128i x_neg_mask_even =
        _mm_shuffle_epi32(x_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    # 从 y_neg_mask 中提取出偶数位，存储在 y_neg_mask_even 中
    const __m128i y_neg_mask_even =
        _mm_shuffle_epi32(y_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    # 从 z_neg_mask 中提取出偶数位，存储在 z_neg_mask_even 中
    const __m128i z_neg_mask_even =
        _mm_shuffle_epi32(z_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    # 从 w_neg_mask 中提取出偶数位，存储在 w_neg_mask_even 中
    const __m128i w_neg_mask_even =
        _mm_shuffle_epi32(w_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));

    # 计算 x_abs_product_even 和 x_neg_mask_even 的结果，并应用补码操作，存储在 x_product_even 中
    const __m128i x_product_even = _mm_sub_epi64(
        _mm_xor_si128(x_abs_product_even, x_neg_mask_even), x_neg_mask_even);
    # 计算 y_abs_product_even 和 y_neg_mask_even 的结果，并应用补码操作，存储在 y_product_even 中
    const __m128i y_product_even = _mm_sub_epi64(
        _mm_xor_si128(y_abs_product_even, y_neg_mask_even), y_neg_mask_even);
    # 计算 z_abs_product_even 和 z_neg_mask_even 的结果，并应用补码操作，存储在 z_product_even 中
    const __m128i z_product_even = _mm_sub_epi64(
        _mm_xor_si128(z_abs_product_even, z_neg_mask_even), z_neg_mask_even);
    # 计算 w_abs_product_even 和 w_neg_mask_even 的结果，并应用补码操作，存储在 w_product_even 中
    const __m128i w_product_even = _mm_sub_epi64(
        _mm_xor_si128(w_abs_product_even, w_neg_mask_even), w_neg_mask_even);

    # 对偶数位的乘积结果进行四舍五入操作，存储在 x_rounded_product_even 中
    const __m128i x_rounded_product_even =
        _mm_add_epi64(x_product_even, vq31rounding);
    # 对偶数位的乘积结果进行四舍五入操作，存储在 y_rounded_product_even 中
    const __m128i y_rounded_product_even =
        _mm_add_epi64(y_product_even, vq31rounding);
    # 对偶数位的乘积结果进行四舍五入操作，存储在 z_rounded_product_even 中
    const __m128i z_rounded_product_even =
        _mm_add_epi64(z_product_even, vq31rounding);
    # 对偶数位的乘积结果进行四舍五入操作，存储在 w_rounded_product_even 中
    const __m128i w_rounded_product_even =
        _mm_add_epi64(w_product_even, vq31rounding);

    # 计算 x_abs_rev 与 vmultiplier 的无符号整数乘法结果，存储在 x_abs_product_odd 中
    const __m128i x_abs_product_odd = _mm_mul_epu32(x_abs_rev, vmultiplier);
    # 计算 y_abs_rev 与 vmultiplier 的无符号整数乘法结果，存储在 y_abs_product_odd 中
    const __m128i y_abs_product_odd = _mm_mul_epu32(y_abs_rev, vmultiplier);
    # 计算 z_abs_rev 与 vmultiplier 的无符号整数乘法结果，存储在 z_abs_product_odd 中
    const __m128i z_abs_product_odd = _mm_mul_epu32(z_abs_rev, vmultiplier);
    # 计算 w_abs_rev 与 vmultiplier 的无符号整数乘法结果，存储在 w_abs_product_odd 中
    const __m128i w_abs_product_odd = _mm_mul_epu32(w_abs_rev, vmultiplier);

    # 从 x_neg_mask 中提取出奇数位，存储在 x_neg_mask_odd 中
    const __m128i x_neg_mask_odd =
        _mm_shuffle_epi32(x_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    # 从 y_neg_mask 中提取出奇数位，存储在 y_neg_mask_odd 中
    const __m128i y_neg_mask_odd =
        _mm_shuffle_epi32(y_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    # 从 z_neg_mask 中提取出奇数位，存储在 z_neg_mask_odd 中
    const __m128i z_neg_mask_odd =
        _mm_shuffle_epi32(z_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    # 从 w_neg_mask 中提取出奇数位，存储在 w_neg_mask_odd 中
    const __m128i w_neg_mask_odd =
        _mm_shuffle_epi32(w_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));

    # 计算 x_abs_product_odd 和 x_neg_mask_odd 的结果，并应用补码操作，存储在 x_product_odd 中
    const __m128i x_product_odd = _mm_sub_epi64(
        _mm_xor_si128(x_abs_product_odd, x_neg_mask_odd), x_neg_mask_odd);
    # 计算 y_abs_product_odd 和 y_neg_mask_odd 的结果，并应用补码操作，存储在 y_product_odd 中
    const __m128i y_product_odd = _mm_sub_epi64(
        _mm_xor_si128(y_abs_product_odd, y_neg_mask_odd), y_neg_mask_odd);
    # 计算 z_abs_product_odd 和 z_neg_mask_odd 的结果，并应用补码操作，存储在 z_product_odd 中
    const __m128i z_product_odd = _mm_sub_epi64(
        _mm_xor_si128(z_abs_product_odd, z_neg_mask_odd), z_neg_mask_odd);
    # 计算 w_abs_product_odd 和 w_neg_mask_odd 的结果，并应用补码操作，存储在 w_product_odd 中
    const __m128i w_product_odd = _mm_sub_epi64(
        _mm_xor_si128(w_abs_product_odd, w_neg_mask_odd), w_neg_mask_odd);

    # 对奇数位的乘积结果进行四舍五入操作，存储在 x_rounded_product_odd 中
    const __m128i x_rounded_product_odd =
        _mm_add_epi64(x_product_odd, vq31rounding);
    # 对奇数位的乘积结果进行四舍五入操作，存储在 y_rounded_product_odd 中
    const __m128i y_rounded_product_odd =
        _mm_add_epi64(y_product_odd, vq31rounding);
    # 对奇数位的乘积结果进行四舍五入操作，存储在 z_rounded_product_odd 中
    const __m128i z_rounded_product_odd =
        _mm_add_epi64(z_product_odd, vq31rounding);
    # 计算奇数位置的加法结果，使用饱和运算的结果
    const __m128i w_rounded_product_odd =
        _mm_add_epi64(w_product_odd, vq31rounding);

    # 将偶数位置的乘积右移31位，得到Q31格式的结果
    const __m128i x_q31product_even =
        _mm_srli_epi64(x_rounded_product_even, 31);
    # 将奇数位置的乘积右移31位，得到Q31格式的结果
    const __m128i x_q31product_odd = _mm_srli_epi64(x_rounded_product_odd, 31);
    # 将偶数位置的乘积右移31位，得到Q31格式的结果
    const __m128i y_q31product_even =
        _mm_srli_epi64(y_rounded_product_even, 31);
    # 将奇数位置的乘积右移31位，得到Q31格式的结果
    const __m128i y_q31product_odd = _mm_srli_epi64(y_rounded_product_odd, 31);
    # 将偶数位置的乘积右移31位，得到Q31格式的结果
    const __m128i z_q31product_even =
        _mm_srli_epi64(z_rounded_product_even, 31);
    # 将奇数位置的乘积右移31位，得到Q31格式的结果
    const __m128i z_q31product_odd = _mm_srli_epi64(z_rounded_product_odd, 31);
    # 将偶数位置的乘积右移31位，得到Q31格式的结果
    const __m128i w_q31product_even =
        _mm_srli_epi64(w_rounded_product_even, 31);
    # 将奇数位置的乘积右移31位，得到Q31格式的结果
    const __m128i w_q31product_odd = _mm_srli_epi64(w_rounded_product_odd, 31);

    # 对Q31格式的偶数和奇数位置进行重新排序，形成0213顺序
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

    # 对0213顺序的Q31格式数据再次进行重新排序，形成3120顺序
    const __m128i x_q31product =
        _mm_shuffle_epi32(x_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i y_q31product =
        _mm_shuffle_epi32(y_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i z_q31product =
        _mm_shuffle_epi32(z_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i w_q31product =
        _mm_shuffle_epi32(w_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));

    # 计算Q31格式的乘积与余数掩码相与后加和，处理溢出情况
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

    # 计算Q31格式的乘积右移vshift位，并处理阈值vthreshold以上的情况
    const __m128i x_scaled = _mm_sub_epi32(
        _mm_sra_epi32(x_q31product, vshift),
        _mm_cmpgt_epi32(x_remainder, vthreshold));
    const __m128i y_scaled = _mm_sub_epi32(
        _mm_sra_epi32(y_q31product, vshift),
        _mm_cmpgt_epi32(y_remainder, vthreshold));
    // 计算 z_scaled，使用 SSE 指令对 z_q31product 进行右移操作，并减去大于阈值的 z_remainder
    const __m128i z_scaled = _mm_sub_epi32(
        _mm_sra_epi32(z_q31product, vshift),
        _mm_cmpgt_epi32(z_remainder, vthreshold));

    // 计算 w_scaled，使用 SSE 指令对 w_q31product 进行右移操作，并减去大于阈值的 w_remainder
    const __m128i w_scaled = _mm_sub_epi32(
        _mm_sra_epi32(w_q31product, vshift),
        _mm_cmpgt_epi32(w_remainder, vthreshold));

    // 将 x_scaled 和 y_scaled 合并为 xy_packed，并加上 vzero_point
    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);

    // 将 z_scaled 和 w_scaled 合并为 zw_packed，并加上 vzero_point
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);

    // 将 xy_packed 和 zw_packed 合并为 xyzw_packed，并转换为无符号 8 位整数
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);

    // 将 xyzw_packed 进行上下限幅处理，确保在 vqmin 和 vqmax 之间
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

    /*
     * 下面是一系列 SSE 指令的注释列表，总共 107 条指令，用于执行各种数据处理和操作
     * 指令涵盖了数据打包、加减法、逻辑运算、位移、比较等操作
     */

    // 将 xyzw_clamped 存储到 output 所指向的内存位置，使用非对齐存储
    _mm_storeu_si128((__m128i*)output, xyzw_clamped);

    // 更新 output 指针，指向下一个 128 位数据的位置
    output += 16;
}
}



# 这行代码表示一个代码块的结束，通常用于结束一个函数、循环、条件语句或其他代码块。
```