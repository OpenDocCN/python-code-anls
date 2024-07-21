# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\4x4c2-dq-sse2.c`

```
  /*
   * 设置四个128位整数寄存器初始为零
   */
  __m128i vacc0x0123 = _mm_setzero_si128();
  __m128i vacc1x0123 = _mm_setzero_si128();
  __m128i vacc2x0123 = _mm_setzero_si128();
  __m128i vacc3x0123 = _mm_setzero_si128();
  
  /*
   * 偏移指针 `w` 以16字节，通常用于对齐操作
   */
  w = (const void*)((uintptr_t)w + 16);

  /*
   * 设置指针 `a0`, `a1`, `a2`, `a3` 以读取矩阵A的数据，
   * 根据mr的值进行条件性地设置指针 `a1`, `a2`, `a3`
   */
  const uint8_t* a0 = a;
  const uint8_t* a1 = (const uint8_t*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const uint8_t* a2 = (const uint8_t*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const uint8_t* a3 = (const uint8_t*)((uintptr_t)a2 + a_stride);
  if (mr != 4) {
    a3 = a2;
  }

  /*
   * 设置128位整数寄存器 `va_zero_point` 为输入的零点偏置值
   */
  const __m128i va_zero_point = _mm_set1_epi16(quantization_params->input_zero_point);

  /*
   * 设置 `vb_zero_point` 为输出通道的四个卷积核零点偏置值
   */
  const int16_t vb_zero_point_0 =
    (int16_t)(uint16_t)quantization_params->kernel_zero_points[
    output_channel_index];
  const int16_t vb_zero_point_1 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 1];
  const int16_t vb_zero_point_2 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 2];
  const int16_t vb_zero_point_3 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 3];

  __m128i vb_zero_point = _mm_set_epi16(vb_zero_point_3,
                                        vb_zero_point_3,
                                        vb_zero_point_2,
                                        vb_zero_point_2,
                                        vb_zero_point_1,
                                        vb_zero_point_1,
                                        vb_zero_point_0,
                                        vb_zero_point_0
                                        );

  /*
   * 载入输出通道的倍乘因子 `vmultiplier`
   */
  const __m128 vmultiplier =
      _mm_loadu_ps(&quantization_params->multipliers[output_channel_index]);

  /*
   * 载入偏置值向量 `vbias`
   */
  const __m128 vbias = _mm_load_ps(b);

  /*
   * 设置128位整数寄存器 `vzero` 为零
   */
  const __m128i vzero = _mm_setzero_si128();

  /*
   * 主循环，每次处理8个元素
   */
  for (; k >= 8; k -= 8) {
    /*
     * 加载矩阵A的每一行的前8个元素，执行零点调整，存储到 `vxa0`, `vxa1`
     */
    const __m128i va0 = _mm_loadl_epi64((const __m128i*)a0);
    const __m128i vxa0 =
        sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
    a0 += 8;
    const __m128i va1 = _mm_loadl_epi64((const __m128i*)a1);
    const __m128i vxa1 =
        sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
    a1 += 8;
    const __m128i va2 = _mm_loadl_epi64((const __m128i*)a2);
    const __m128i vxa2 =
        sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
    a2 += 8;
    const __m128i va3 = _mm_loadl_epi64((const __m128i*)a3);
    const __m128i vxa3 =
        sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
    a3 += 8;

    /*
     * 更新累加寄存器 `vacc0x0123`, `vacc1x0123`, `vacc2x0123`, `vacc3x0123`，
     * 执行Q8 GEMM操作
     */
    // 加载va2指向的内存中的8个字节数据到__m128i类型的变量vxa2中，并进行零点调整
    const __m128i vxa2 =
        sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
    // a2指针移动到下一个8字节位置
    a2 += 8;

    // 加载a3指向的内存中的8个字节数据到__m128i类型的变量va3中
    const __m128i va3 = _mm_loadl_epi64((const __m128i*)a3);
    // 将va3进行零点调整后存入vxa3
    const __m128i vxa3 =
        sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
    // a3指针移动到下一个8字节位置
    a3 += 8;

    // 加载指针w指向的内存中的8字节数据到__m128i类型的变量vb0中
    const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
    // 将vb0进行零点调整后存入vxb0
    const __m128i vxb0 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

    // 使用vxa0和vxb0计算第一个累加器的结果并更新
    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    // 使用vxa1和vxb0计算第二个累加器的结果并更新
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    // 使用vxa2和vxb0计算第三个累加器的结果并更新
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    // 使用vxa3和vxb0计算第四个累加器的结果并更新
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));

    // 加载指针w+8字节指向的内存中的8字节数据到__m128i类型的变量vb1中
    const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
    // 将vb1进行零点调整后存入vxb1
    const __m128i vxb1 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

    // 使用vxa0和vxb1计算第一个累加器的结果并更新
    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 使用vxa1和vxb1计算第二个累加器的结果并更新
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 使用vxa2和vxb1计算第三个累加器的结果并更新
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 使用vxa3和vxb1计算第四个累加器的结果并更新
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

    // 加载指针w+16字节指向的内存中的8字节数据到__m128i类型的变量vb2中
    const __m128i vb2 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
    // 将vb2进行零点调整后存入vxb2
    const __m128i vxb2 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

    // 使用vxa0和vxb2计算第一个累加器的结果并更新
    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    // 使用vxa1和vxb2计算第二个累加器的结果并更新
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    // 使用vxa2和vxb2计算第三个累加器的结果并更新
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    // 使用vxa3和vxb2计算第四个累加器的结果并更新
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

    // 加载指针w+24字节指向的内存中的8字节数据到__m128i类型的变量vb3中
    const __m128i vb3 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
    // 将vb3进行零点调整后存入vxb3
    const __m128i vxb3 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);
    // w指针向后移动32字节
    w = (const void*)((uintptr_t)w + 32);

    // 使用vxa0和vxb3计算第一个累加器的结果并更新
    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    // 使用vxa1和vxb3计算第二个累加器的结果并更新
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    // 使用vxa2和vxb3计算第三个累加器的结果并更新
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    // 如果 k 不等于 0，则进行下列操作
    if (k != 0) {
        // 计算预减值，a_predecrement 表示 8 减去 k 的结果
        const size_t a_predecrement = 8 - k;
        // 创建一个 __m128i 类型的变量 va_shift，值为 8 * a_predecrement
        const __m128i va_shift = _mm_cvtsi32_si128(8 * a_predecrement);

        // 加载 a0 - a_predecrement 处的数据，右移 va_shift 位，得到 va0
        const __m128i va0 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a0 - a_predecrement)), va_shift);
        // 将 va0 的低位 8 个字节解包成 16 位整数，减去零点 va_zero_point，再处理为 8 位无符号整数，得到 vxa0
        const __m128i vxa0 = sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);

        // 加载 a1 - a_predecrement 处的数据，右移 va_shift 位，得到 va1
        const __m128i va1 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a1 - a_predecrement)), va_shift);
        // 将 va1 的低位 8 个字节解包成 16 位整数，减去零点 va_zero_point，再处理为 8 位无符号整数，得到 vxa1
        const __m128i vxa1 = sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);

        // 加载 a2 - a_predecrement 处的数据，右移 va_shift 位，得到 va2
        const __m128i va2 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a2 - a_predecrement)), va_shift);
        // 将 va2 的低位 8 个字节解包成 16 位整数，减去零点 va_zero_point，再处理为 8 位无符号整数，得到 vxa2
        const __m128i vxa2 = sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);

        // 加载 a3 - a_predecrement 处的数据，右移 va_shift 位，得到 va3
        const __m128i va3 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a3 - a_predecrement)), va_shift);
        // 将 va3 的低位 8 个字节解包成 16 位整数，减去零点 va_zero_point，再处理为 8 位无符号整数，得到 vxa3
        const __m128i vxa3 = sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);

        // 加载 w 处的数据，解包成 16 位整数，减去零点 vb_zero_point，得到 vxb0
        const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

        // 计算 vacc0x0123 = vacc0x0123 + vxa0 * vxb0 的结果
        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        // 计算 vacc1x0123 = vacc1x0123 + vxa1 * vxb0 的结果
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        // 计算 vacc2x0123 = vacc2x0123 + vxa2 * vxb0 的结果
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        // 计算 vacc3x0123 = vacc3x0123 + vxa3 * vxb0 的结果
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    }
    # 检查 k 是否大于 2
    if (k > 2) {
      # 加载指针 w 偏移 8 字节处的数据到 vb1
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
      # 计算 vxb1，将 vb1 转换成 8 位无符号整数并减去 vb_zero_point
      const __m128i vxb1 =
          _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

      # 计算 vacc0x0123，将 vxa0 的元素按顺序重排并与 vxb1 进行逐元素乘加
      vacc0x0123 = _mm_add_epi32(
          vacc0x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      # 类似地计算 vacc1x0123、vacc2x0123 和 vacc3x0123
      vacc1x0123 = _mm_add_epi32(
          vacc1x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      vacc2x0123 = _mm_add_epi32(
          vacc2x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      vacc3x0123 = _mm_add_epi32(
          vacc3x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

      # 检查 k 是否大于 4
      if (k > 4) {
        # 加载指针 w 偏移 16 字节处的数据到 vb2
        const __m128i vb2 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
        # 计算 vxb2，将 vb2 转换成 8 位无符号整数并减去 vb_zero_point
        const __m128i vxb2 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

        # 类似地计算 vacc0x0123、vacc1x0123、vacc2x0123 和 vacc3x0123，使用 vxa0 的第二个元素
        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

        # 检查 k 是否大于 6
        if (k > 6) {
          # 加载指针 w 偏移 24 字节处的数据到 vb3
          const __m128i vb3 =
              _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
          # 计算 vxb3，将 vb3 转换成 8 位无符号整数并减去 vb_zero_point
          const __m128i vxb3 =
              _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);

          # 类似地计算 vacc0x0123、vacc1x0123、vacc2x0123 和 vacc3x0123，使用 vxa0 的第三个元素
          vacc0x0123 = _mm_add_epi32(
              vacc0x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
          vacc1x0123 = _mm_add_epi32(
              vacc1x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
          vacc2x0123 = _mm_add_epi32(
              vacc2x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
          vacc3x0123 = _mm_add_epi32(
              vacc3x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
        }
      }
    }
  }

  __m128 vout0 = _mm_mul_ps(vmultiplier, _mm_cvtepi32_ps(vacc0x0123));
  // 使用乘法指令计算 vmultiplier 和 vacc0x0123 向量的乘积，并将结果存储在 vout0 中
  __m128 vout1 = _mm_mul_ps(vmultiplier, _mm_cvtepi32_ps(vacc1x0123));
  // 使用乘法指令计算 vmultiplier 和 vacc1x0123 向量的乘积，并将结果存储在 vout1 中
  __m128 vout2 = _mm_mul_ps(vmultiplier, _mm_cvtepi32_ps(vacc2x0123));
  // 使用乘法指令计算 vmultiplier 和 vacc2x0123 向量的乘积，并将结果存储在 vout2 中
  __m128 vout3 = _mm_mul_ps(vmultiplier, _mm_cvtepi32_ps(vacc3x0123));
  // 使用乘法指令计算 vmultiplier 和 vacc3x0123 向量的乘积，并将结果存储在 vout3 中

  vout0 = _mm_add_ps(vout0, vbias);
  // 将 vout0 和 vbias 向量相加，并将结果存储回 vout0
  vout1 = _mm_add_ps(vout1, vbias);
  // 将 vout1 和 vbias 向量相加，并将结果存储回 vout1
  vout2 = _mm_add_ps(vout2, vbias);
  // 将 vout2 和 vbias 向量相加，并将结果存储回 vout2
  vout3 = _mm_add_ps(vout3, vbias);
  // 将 vout3 和 vbias 向量相加，并将结果存储回 vout3

  float* c0 = c;
  // 初始化指针 c0 指向数组 c 的起始位置
  float* c1 = c0 + c_stride;
  // 初始化指针 c1 指向数组 c 中偏移 c_stride 后的位置
  if (mr < 2) {
    c1 = c0;
    // 如果 mr 小于 2，则将 c1 指向 c0，即将 c1 与 c0 指向同一位置
  }
  float* c2 = c1 + c_stride;
  // 初始化指针 c2 指向数组 c 中偏移 c1 + c_stride 后的位置
  if (mr <= 2) {
    c2 = c1;
    // 如果 mr 小于等于 2，则将 c2 指向 c1，即将 c2 与 c1 指向同一位置
  }
  float* c3 = c2 + c_stride;
  // 初始化指针 c3 指向数组 c 中偏移 c2 + c_stride 后的位置
  if (mr != 4) {
    c3 = c2;
    // 如果 mr 不等于 4，则将 c3 指向 c2，即将 c3 与 c2 指向同一位置
  }

  if (nr == 4) {
    _mm_storeu_ps(c0, vout0);
    // 将 vout0 中的四个单精度浮点数存储到 c0 指向的内存位置，不需要对齐
    _mm_storeu_ps(c1, vout1);
    // 将 vout1 中的四个单精度浮点数存储到 c1 指向的内存位置，不需要对齐
    _mm_storeu_ps(c2, vout2);
    // 将 vout2 中的四个单精度浮点数存储到 c2 指向的内存位置，不需要对齐
    _mm_storeu_ps(c3, vout3);
    // 将 vout3 中的四个单精度浮点数存储到 c3 指向的内存位置，不需要对齐
  } else {
    if (nr >= 2) {
      _mm_storel_pi((__m64*)c0, vout0);
      // 将 vout0 中的两个单精度浮点数的低位存储到 c0 指向的内存位置，需要 64 位对齐
      _mm_storel_pi((__m64*)c1, vout1);
      // 将 vout1 中的两个单精度浮点数的低位存储到 c1 指向的内存位置，需要 64 位对齐
      _mm_storel_pi((__m64*)c2, vout2);
      // 将 vout2 中的两个单精度浮点数的低位存储到 c2 指向的内存位置，需要 64 位对齐
      _mm_storel_pi((__m64*)c3, vout3);
      // 将 vout3 中的两个单精度浮点数的低位存储到 c3 指向的内存位置，需要 64 位对齐

      c0 += 2;
      // 将 c0 指针向后移动两个位置
      vout0 = _mm_shuffle_ps(vout0, vout0, _MM_SHUFFLE(2, 2, 2, 2));
      // 将 vout0 向量的第二个单精度浮点数复制到其余位置，实现数据交换
      c1 += 2;
      // 将 c1 指针向后移动两个位置
      vout1 = _mm_shuffle_ps(vout1, vout1, _MM_SHUFFLE(2, 2, 2, 2));
      // 将 vout1 向量的第二个单精度浮点数复制到其余位置，实现数据交换
      c2 += 2;
      // 将 c2 指针向后移动两个位置
      vout2 = _mm_shuffle_ps(vout2, vout2, _MM_SHUFFLE(2, 2, 2, 2));
      // 将 vout2 向量的第二个单精度浮点数复制到其余位置，实现数据交换
      c3 += 2;
      // 将 c3 指针向后移动两个位置
      vout3 = _mm_shuffle_ps(vout3, vout3, _MM_SHUFFLE(2, 2, 2, 2));
      // 将 vout3 向量的第二个单精度浮点数复制到其余位置，实现数据交换

      nr -= 2;
      // 减少剩余未处理的向量元素数量
    }
    if (nr != 0) {
      *c0 = _mm_cvtss_f32(vout0);
      // 将 vout0 向量的第一个单精度浮点数转换为标量并存储到 c0 指向的内存位置
      *c1 = _mm_cvtss_f32(vout1);
      // 将 vout1 向量的第一个单精度浮点数转换为标量并存储到 c1 指向的内存位置
      *c2 = _mm_cvtss_f32(vout2);
      // 将 vout2 向量的第一个单精度浮点数转换为标量并存储到 c2 指向的内存位置
      *c3 = _mm_cvtss_f32(vout3);
      // 将 vout3 向量的第一个单精度浮点数转换为标量并存储到 c3 指向的内存位置
    }
  }
}



# 这行代码关闭了一个代码块。在一些编程语言中，如Python和C，这对应于控制流或函数定义的结束。
```