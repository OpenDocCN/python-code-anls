# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\precise-sse4.c`

```py
/*
 * 精确重新量化函数，基于 SSE4 指令集
 * 
 * n: 输入数据元素的数量，要求是16的倍数
 * input: 输入的 int32_t 类型数组指针，长度为 n
 * scale: 缩放因子，类型为 float，要求小于1.0
 * zero_point: 零点偏移量，类型为 uint8_t
 * qmin: 输出量化的最小值，类型为 uint8_t
 * qmax: 输出量化的最大值，类型为 uint8_t
 * output: 输出的 uint8_t 类型数组指针，长度为 n
 */
void pytorch_qnnp_requantize_precise__sse4(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  // 确保 n 是 16 的倍数，以便进行 SSE4 128 位数据处理
  assert(n % 16 == 0);
  // 确保 scale 小于 1.0
  assert(scale < 1.0f);
  // 确保 scale 大于等于 2^-32
  assert(scale >= 0x1.0p-32f);

  // 将 scale 转换为整数表示，用于后续的乘法运算
  const uint32_t scale_bits = fp32_to_bits(scale);
  const uint32_t multiplier = (scale_bits << 8) | UINT32_C(0x80000000);
  // 计算右移的位数，确保输入的精度正确
  const uint32_t shift = 127 + 31 - (scale_bits >> 23);
  assert(shift >= 32);
  assert(shift < 64);
  // 计算舍入量，用于乘法后的舍入
  const uint64_t rounding = UINT64_C(1) << (shift - 1);

  // 将常数转换为 SSE 寄存器的格式，用于向量化运算
  const __m128i vmultiplier = _mm_set1_epi32(multiplier);
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);
  const __m128i vqmin = _mm_set1_epi8((char)qmin);
  const __m128i vqmax = _mm_set1_epi8((char)qmax);
  const __m128i vshiftlo = _mm_cvtsi32_si128((int)shift);
  const __m128i vshifthi = _mm_cvtsi32_si128((int)shift - 32);
  const __m128i vrounding = _mm_set1_epi64x(rounding);

  // 对输入数组进行向量化处理，每次处理 16 个元素
  for (; n != 0; n -= 16) {
    // 从内存中加载 4 个 int32_t 元素到 SSE 寄存器
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    // 计算绝对值并重新排列元素，以便进行乘法运算
    const __m128i x_abs0123 = _mm_abs_epi32(x);
    const __m128i y_abs0123 = _mm_abs_epi32(y);
    const __m128i z_abs0123 = _mm_abs_epi32(z);
    const __m128i w_abs0123 = _mm_abs_epi32(w);

    const __m128i x_abs1032 =
        _mm_shuffle_epi32(x_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i y_abs1032 =
        _mm_shuffle_epi32(y_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i z_abs1032 =
        _mm_shuffle_epi32(z_abs0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i w_abs1032 =
        _mm_shuffle_epi32(w_abs0123, _MM_SHUFFLE(2, 3, 0, 1));

    // 使用乘法器和位移器计算乘积并进行右移操作
    const __m128i x_absmul02 = _mm_mul_epu32(x_abs0123, vmultiplier);
    const __m128i y_absmul02 = _mm_mul_epu32(y_abs0123, vmultiplier);
    const __m128i z_absmul02 = _mm_mul_epu32(z_abs0123, vmultiplier);
    const __m128i w_absmul02 = _mm_mul_epu32(w_abs0123, vmultiplier);

    const __m128i x_absmul13 = _mm_mul_epu32(x_abs1032, vmultiplier);
    const __m128i y_absmul13 = _mm_mul_epu32(y_abs1032, vmultiplier);
    const __m128i z_absmul13 = _mm_mul_epu32(z_abs1032, vmultiplier);
    const __m128i w_absmul13 = _mm_mul_epu32(w_abs1032, vmultiplier);

    // 对乘积结果进行舍入和右移操作，得到最终量化结果
    const __m128i x_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(x_absmul02, vrounding), vshiftlo);
    const __m128i x_abs_scaled13 =
        _mm_srl_epi32(_mm_add_epi64(x_absmul13, vrounding), vshifthi);
    const __m128i y_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(y_absmul02, vrounding), vshiftlo);
    // 计算 y_absmul02 加上 vrounding 后的结果，然后右移 vshiftlo 指定的位数

    const __m128i y_abs_scaled13 =
        _mm_srl_epi32(_mm_add_epi64(y_absmul13, vrounding), vshifthi);
    // 计算 y_absmul13 加上 vrounding 后的结果，然后右移 vshifthi 指定的位数

    const __m128i z_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(z_absmul02, vrounding), vshiftlo);
    // 计算 z_absmul02 加上 vrounding 后的结果，然后右移 vshiftlo 指定的位数

    const __m128i z_abs_scaled13 =
        _mm_srl_epi32(_mm_add_epi64(z_absmul13, vrounding), vshifthi);
    // 计算 z_absmul13 加上 vrounding 后的结果，然后右移 vshifthi 指定的位数

    const __m128i w_abs_scaled02 =
        _mm_srl_epi64(_mm_add_epi64(w_absmul02, vrounding), vshiftlo);
    // 计算 w_absmul02 加上 vrounding 后的结果，然后右移 vshiftlo 指定的位数

    const __m128i w_abs_scaled13 =
        _mm_srl_epi32(_mm_add_epi64(w_absmul13, vrounding), vshifthi);
    // 计算 w_absmul13 加上 vrounding 后的结果，然后右移 vshifthi 指定的位数

    const __m128i x_abs_scaled =
        _mm_blend_epi16(x_abs_scaled02, x_abs_scaled13, 0xCC);
    // 使用掩码 0xCC 混合 x_abs_scaled02 和 x_abs_scaled13 的结果

    const __m128i y_abs_scaled =
        _mm_blend_epi16(y_abs_scaled02, y_abs_scaled13, 0xCC);
    // 使用掩码 0xCC 混合 y_abs_scaled02 和 y_abs_scaled13 的结果

    const __m128i z_abs_scaled =
        _mm_blend_epi16(z_abs_scaled02, z_abs_scaled13, 0xCC);
    // 使用掩码 0xCC 混合 z_abs_scaled02 和 z_abs_scaled13 的结果

    const __m128i w_abs_scaled =
        _mm_blend_epi16(w_abs_scaled02, w_abs_scaled13, 0xCC);
    // 使用掩码 0xCC 混合 w_abs_scaled02 和 w_abs_scaled13 的结果

    const __m128i x_scaled = _mm_sign_epi32(x_abs_scaled, x);
    // 使用 x 的符号位对 x_abs_scaled 进行符号扩展

    const __m128i y_scaled = _mm_sign_epi32(y_abs_scaled, y);
    // 使用 y 的符号位对 y_abs_scaled 进行符号扩展

    const __m128i z_scaled = _mm_sign_epi32(z_abs_scaled, z);
    // 使用 z 的符号位对 z_abs_scaled 进行符号扩展

    const __m128i w_scaled = _mm_sign_epi32(w_abs_scaled, w);
    // 使用 w 的符号位对 w_abs_scaled 进行符号扩展

    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    // 将 x_scaled 和 y_scaled 进行 32 位有符号整数到 16 位有符号整数的转换，然后加上 vzero_point

    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    // 将 z_scaled 和 w_scaled 进行 32 位有符号整数到 16 位有符号整数的转换，然后加上 vzero_point

    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    // 将 xy_packed 和 zw_packed 进行 16 位有符号整数到 8 位无符号整数的转换，得到合并的结果

    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);
    // 使用 vqmax 和 vqmin 对 xyzw_packed 进行上下限制，得到最终的结果

    /*
     * 4x PABSD
     * 4x PSHUFD
     * 8x PMULUDQ
     * 4x PSRLQ
     * 4x PSRLD
     * 8x PADDQ
     * 4x PBLENDW
     * 4x PSIGND
     * 2x PACKSSDW
     * 1x PACKUSWB
     * 2x PADDW
     * 1x PMAXUB
     * 1x PMINUB
     * ---------------------
     * 47 instructions total
     */

    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    // 将 xyzw_clamped 存储到 output 所指向的内存位置
    output += 16;
    // 将 output 指针向后移动 16 字节，准备存储下一个结果
}



# 这是一个单独的右花括号 '}'，用于结束一个代码块或数据结构的定义。
```