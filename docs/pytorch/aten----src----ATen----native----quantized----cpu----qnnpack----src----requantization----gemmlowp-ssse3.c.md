# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\gemmlowp-ssse3.c`

```py
/*
 * 该函数实现了基于 SSE 指令集的量化神经网络推断中的 Gemmlowp 矩阵乘法和量化过程
 * 以 SSSE3 指令进行优化
 */
void pytorch_qnnp_requantize_gemmlowp__ssse3(
    size_t n,                         // 输入向量的长度，要求为 16 的倍数
    const int32_t* input,             // 输入数据的指针，包含要进行量化的整数值
    float scale,                      // 量化的比例因子
    uint8_t zero_point,               // 零点值，用于量化的偏移量
    uint8_t qmin,                     // 输出量化的最小值
    uint8_t qmax,                     // 输出量化的最大值
    uint8_t* output) {                // 输出结果的指针，存储量化后的无符号整数值
  assert(n % 16 == 0);               // 断言：输入向量长度必须是 16 的倍数
  assert(scale < 1.0f);              // 断言：量化比例因子必须小于 1.0
  assert(scale >= 0x1.0p-32f);       // 断言：量化比例因子必须大于等于 2^-32

  const uint32_t scale_bits = fp32_to_bits(scale);  // 将浮点数比例因子转换为其二进制表示形式

  /* 计算量化参数 */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;  // 计算乘数
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;    // 计算指数
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);                                                       // 计算右移量

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);      // 将乘数转换为 SSE 寄存器类型
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);  // 将零点值转换为 SSE 寄存器类型
  const __m128i vqmin = _mm_set1_epi8((char)qmin);             // 将 qmin 转换为 SSE 寄存器类型
  const __m128i vqmax = _mm_set1_epi8((char)qmax);             // 将 qmax 转换为 SSE 寄存器类型

  /* 循环处理输入向量中的每 16 个元素 */
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);       // 加载输入数据向量的四个片段到 SSE 寄存器
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;                                                    // 更新输入数据指针位置

    /* 执行 Gemmlowp 算法中的量化乘法操作 */
    const __m128i x_product = gemmlowp_sse_vqrdmulh_s32(x, vmultiplier);
    const __m128i y_product = gemmlowp_sse_vqrdmulh_s32(y, vmultiplier);
    const __m128i z_product = gemmlowp_sse_vqrdmulh_s32(z, vmultiplier);
    const __m128i w_product = gemmlowp_sse_vqrdmulh_s32(w, vmultiplier);

    /* 执行右移操作 */
    const __m128i x_scaled = gemmlowp_sse_rdivbypo2_s32(x_product, shift);
    const __m128i y_scaled = gemmlowp_sse_rdivbypo2_s32(y_product, shift);
    const __m128i z_scaled = gemmlowp_sse_rdivbypo2_s32(z_product, shift);
    const __m128i w_scaled = gemmlowp_sse_rdivbypo2_s32(w_product, shift);

    /* 将量化结果打包成 8 位整数 */
    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);

    /* 将打包结果进一步压缩至 8 位无符号整数范围内 */
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);  // 对量化结果进行截断

    /* 将结果存储到输出数组中 */
    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    output += 16;  // 更新输出数据指针位置
  }
}
```