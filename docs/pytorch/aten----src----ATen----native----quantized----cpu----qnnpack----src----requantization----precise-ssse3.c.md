# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\precise-ssse3.c`

```py
/*
 * 以下是一个 C++ 函数，用于精确重新量化整数数组到无符号8位整数数组
 * 使用 SSSE3 SIMD 指令集进行加速计算
 */

void pytorch_qnnp_requantize_precise__ssse3(
    size_t n,                             // 输入数组的长度，必须是16的倍数
    const int32_t* input,                 // 输入整数数组的指针
    float scale,                          // 重新量化的比例因子
    uint8_t zero_point,                   // 重新量化的零点
    uint8_t qmin,                         // 输出的最小值限制
    uint8_t qmax,                         // 输出的最大值限制
    uint8_t* output) {                    // 输出无符号8位整数数组的指针

  assert(n % 16 == 0);                   // 断言：n必须是16的倍数，确保可以使用SSSE3向量化操作
  assert(scale < 1.0f);                  // 断言：scale必须小于1.0
  assert(scale >= 0x1.0p-32f);           // 断言：scale必须大于或等于2^-32

  // 将浮点数scale转换为32位整数，并提取有效部分作为乘法因子multiplier
  const uint32_t scale_bits = fp32_to_bits(scale);
  const uint32_t multiplier =
      (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  
  // 计算右移位数shift和舍入值rounding
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);                   // 断言：shift必须大于等于24
  assert(shift < 56);                    // 断言：shift必须小于56
  const uint64_t rounding = UINT64_C(1) << (shift - 1);

  // 使用SSSE3指令集创建常量向量
  const __m128i vmultiplier = _mm_set1_epi32(multiplier);    // 创建包含multiplier的128位整数向量
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);  // 创建包含zero_point的128位整数向量
  const __m128i vqmin = _mm_set1_epi8((char)qmin);            // 创建包含qmin的128位整数向量
  const __m128i vqmax = _mm_set1_epi8((char)qmax);            // 创建包含qmax的128位整数向量
  const __m128i vshift = _mm_cvtsi32_si128((int)shift);       // 创建包含shift的128位整数向量
  const __m128i vrounding = _mm_set1_epi64x(rounding);        // 创建包含rounding的128位整数向量

  // 遍历整数数组，每次处理16个元素，使用SSSE3指令集进行向量化计算
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);        // 加载16个输入整数到x向量
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));   // 加载接下来的16个输入整数到y向量
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));   // 加载接下来的16个输入整数到z向量
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));  // 加载接下来的16个输入整数到w向量
    input += 16;                                                      // 更新输入指针到下一组16个整数

    // 计算每个向量元素的绝对值
    const __m128i x_abs0123 = _mm_abs_epi32(x);
    const __m128i y_abs0123 = _mm_abs_epi32(y);
    const __m128i z_abs0123 = _mm_abs_epi32(z);
    const __m128i w_abs0123 = _mm_abs_epi32(w);

    // 对绝对值向量进行重新排序，以便进行乘法操作
    const __m128i x_abs1032 =
        _mm_shuffle_epi32(x_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i y_abs1032 =
        _mm_shuffle_epi32(y_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i z_abs1032 =
        _mm_shuffle_epi32(z_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i w_abs1032 =
        _mm_shuffle_epi32(w_abs0123, _MM_SHUFFLE(2, 3, 0, 1));

    // 使用乘法指令计算乘以乘法因子后的值
    const __m128i x_absmul02 = _mm_mul_epu32(x_abs0123, vmultiplier);
    const __m128i y_absmul02 = _mm_mul_epu32(y_abs0123, vmultiplier);
    const __m128i z_absmul02 = _mm_mul_epu32(z_abs0123, vmultiplier);
    const __m128i w_absmul02 = _mm_mul_epu32(w_abs0123, vmultiplier);

    const __m128i x_absmul13 = _mm_mul_epu32(x_abs1032, vmultiplier);
    const __m128i y_absmul13 = _mm_mul_epu32(y_abs1032, vmultiplier);
    const __m128i z_absmul13 = _mm_mul_epu32(z_abs1032, vmultiplier);
    const __m128i w_absmul13 = _mm_mul_epu32(w_abs1032, vmultiplier);

    // 对乘法结果进行移位和舍入操作，然后将结果重新量化到8位无符号整数范围内
    const __m128i x_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(x_absmul02, vrounding), vshift);
    const __m128i x_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(x_absmul13, vrounding), vshift);

    // 将结果截断并重新量化到输出的8位整数范围内
    // 然后将结果存储到输出数组中
    // 计算 y_absmul02 和 vrounding 的和后右移 vshift 位，结果存储在 y_abs_scaled02 中
    const __m128i y_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(y_absmul02, vrounding), vshift);
    
    // 计算 y_absmul13 和 vrounding 的和后右移 vshift 位，结果存储在 y_abs_scaled13 中
    const __m128i y_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(y_absmul13, vrounding), vshift);
    
    // 计算 z_absmul02 和 vrounding 的和后右移 vshift 位，结果存储在 z_abs_scaled02 中
    const __m128i z_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(z_absmul02, vrounding), vshift);
    
    // 计算 z_absmul13 和 vrounding 的和后右移 vshift 位，结果存储在 z_abs_scaled13 中
    const __m128i z_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(z_absmul13, vrounding), vshift);
    
    // 计算 w_absmul02 和 vrounding 的和后右移 vshift 位，结果存储在 w_abs_scaled02 中
    const __m128i w_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(w_absmul02, vrounding), vshift);
    
    // 计算 w_absmul13 和 vrounding 的和后右移 vshift 位，结果存储在 w_abs_scaled13 中
    const __m128i w_abs_scaled13 =
        _mm_srl_epi64(_mm_add_epi64(w_absmul13, vrounding), vshift);

    // 将 x_abs_scaled02 和 x_abs_scaled13 按顺序组合并转换为整型寄存器 x_abs_scaled0213
    const __m128i x_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x_abs_scaled02),
        _mm_castsi128_ps(x_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    
    // 将 y_abs_scaled02 和 y_abs_scaled13 按顺序组合并转换为整型寄存器 y_abs_scaled0213
    const __m128i y_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(y_abs_scaled02),
        _mm_castsi128_ps(y_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    
    // 将 z_abs_scaled02 和 z_abs_scaled13 按顺序组合并转换为整型寄存器 z_abs_scaled0213
    const __m128i z_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(z_abs_scaled02),
        _mm_castsi128_ps(z_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));
    
    // 将 w_abs_scaled02 和 w_abs_scaled13 按顺序组合并转换为整型寄存器 w_abs_scaled0213
    const __m128i w_abs_scaled0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(w_abs_scaled02),
        _mm_castsi128_ps(w_abs_scaled13),
        _MM_SHUFFLE(2, 0, 2, 0)));

    // 将 x_abs_scaled0213 寄存器中的元素按特定顺序重新排列，并存储在 x_abs_scaled 中
    const __m128i x_abs_scaled =
        _mm_shuffle_epi32(x_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    
    // 将 y_abs_scaled0213 寄存器中的元素按特定顺序重新排列，并存储在 y_abs_scaled 中
    const __m128i y_abs_scaled =
        _mm_shuffle_epi32(y_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    
    // 将 z_abs_scaled0213 寄存器中的元素按特定顺序重新排列，并存储在 z_abs_scaled 中
    const __m128i z_abs_scaled =
        _mm_shuffle_epi32(z_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));
    
    // 将 w_abs_scaled0213 寄存器中的元素按特定顺序重新排列，并存储在 w_abs_scaled 中
    const __m128i w_abs_scaled =
        _mm_shuffle_epi32(w_abs_scaled0213, _MM_SHUFFLE(3, 1, 2, 0));

    // 将 x_abs_scaled 中的元素按照 x 寄存器的符号进行反转，并存储在 x_scaled 中
    const __m128i x_scaled = _mm_sign_epi32(x_abs_scaled, x);
    
    // 将 y_abs_scaled 中的元素按照 y 寄存器的符号进行反转，并存储在 y_scaled 中
    const __m128i y_scaled = _mm_sign_epi32(y_abs_scaled, y);
    
    // 将 z_abs_scaled 中的元素按照 z 寄存器的符号进行反转，并存储在 z_scaled 中
    const __m128i z_scaled = _mm_sign_epi32(z_abs_scaled, z);
    
    // 将 w_abs_scaled 中的元素按照 w 寄存器的符号进行反转，并存储在 w_scaled 中
    const __m128i w_scaled = _mm_sign_epi32(w_abs_scaled, w);

    // 将 x_scaled 和 y_scaled 寄存器中的元素打包成 16 位有符号整数，加上 vzero_point，并存储在 xy_packed 中
    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    
    // 将 z_scaled 和 w_scaled 寄存器中的元素打包成 16 位有符号整数，加上 vzero_point，并存储在 zw_packed 中
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    
    // 将 xy_packed 和 zw_packed 寄存器中的元素打包成 8 位无符号整数，结果存储在 xyzw_packed 中
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    
    // 将 xyzw_packed 寄存器中的元素限制在 vqmin 和 vqmax 之间，并存储在 xyzw_clamped 中
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

    /*
     * 以下是指令计数和总结:
     * 4x PABSD
     * 8x PSHUFD
     * 8x PMULUDQ
     * 8x PSRLQ
     * 8x PADDQ
     * 4x SHUFPS
     * 4x PSIGND
     * 2x PACKSSDW
     * 1x PACKUSWB
     * 2x PADDW
     * 1x PMAXUB
     * 1x PMINUB
     * ---------------------
     * 51 条指令总数
     */

    // 将 xyzw_clamped 寄存器中的元素存储到 output 所指向的内存位置，并使 output 指向下一个 16 字节的位置
    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    output += 16;
}
}



# 这是一个代码块的结束，用于结束一个代码块或函数的定义。
```