# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\4x4c2-sse2.c`

```
  /*
   * 从权重矩阵中加载第一个 128 位的数据，初始化累加器
   */
  __m128i vacc0x0123 = _mm_loadu_si128((const __m128i*)w);

  /*
   * 复制第一个累加器到其余三个累加器
   */
  __m128i vacc1x0123 = vacc0x0123;
  __m128i vacc2x0123 = vacc0x0123;
  __m128i vacc3x0123 = vacc0x0123;

  /*
   * 将指针 w 向后移动 16 字节，以加载下一个 128 位数据
   */
  w = (const void*)((uintptr_t)w + 16);

  /*
   * 设置指向矩阵 A 中每一行的指针，根据矩阵行数 mr 来处理边界情况
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
   * 加载量化参数，包括输入的零点 va_zero_point 和输出的零点 vb_zero_point
   */
  const __m128i va_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.input_zero_point);
  const int16_t vb_zero_point_0 =
    (int16_t)(uint16_t)quantization_params->sse2.kernel_zero_points[
    output_channel_index];
  const int16_t vb_zero_point_1 =
      (int16_t)(uint16_t)quantization_params->sse2.kernel_zero_points[
        output_channel_index + 1];
  const int16_t vb_zero_point_2 =
      (int16_t)(uint16_t)quantization_params->sse2.kernel_zero_points[
        output_channel_index + 2];
  const int16_t vb_zero_point_3 =
      (int16_t)(uint16_t)quantization_params->sse2.kernel_zero_points[
        output_channel_index + 3];

  /*
   * 设置 vb_zero_point 向量，以便在计算中使用
   */
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
   * 初始化用于零点扩展的常量向量 vzero
   */
  const __m128i vzero = _mm_setzero_si128();

  /*
   * 主循环：处理 k 大于等于 8 的情况，每次处理 8 个元素
   */
  for (; k >= 8; k -= 8) {
    /*
     * 加载矩阵 A 的第一行数据，并进行零点扩展和零点减法操作
     */
    const __m128i va0 = _mm_loadl_epi64((const __m128i*)a0);
    const __m128i vxa0 =
        sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
    a0 += 8;

    /*
     * 加载矩阵 A 的第二行数据，并进行零点扩展和零点减法操作
     */
    const __m128i va1 = _mm_loadl_epi64((const __m128i*)a1);
    const __m128i vxa1 =
        sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
    a1 += 8;

    /*
     * 加载矩阵 A 的第三行数据，并进行零点扩展和零点减法操作
     */
    const __m128i va2 = _mm_loadl_epi64((const __m128i*)a2);
    const __m128i vxa2 =
        sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
    a2 += 8;

    /*
     * 加载矩阵 A 的第四行数据，并进行零点扩展和零点减法操作
     */
    const __m128i va3 = _mm_loadl_epi64((const __m128i*)a3);
    const __m128i vxa3 =
        sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
    a3 += 8;
    const __m128i vxa3 =
        sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
    // 使用 _mm_unpacklo_epi8 将 va3 和 vzero 解包成 16 位整数，然后调用 sub_zero_point 函数进行零点对齐操作，存入 vxa3
    a3 += 8;
    // 增加 a3 的值，移动到下一个数据块的位置

    const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
    // 从内存地址 w 处加载 64 位数据，存入 vb0
    const __m128i vxb0 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);
    // 使用 _mm_unpacklo_epi8 解包 vb0 和 vzero 成 16 位整数，然后与 vb_zero_point 执行减法，存入 vxb0

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    // 使用 _mm_shuffle_epi32 对 vxa0 进行按指定顺序重排，然后与 vxb0 执行乘加运算，结果累加到 vacc0x0123
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    // 同上，但使用 vxa1 和 vacc1x0123
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    // 同上，但使用 vxa2 和 vacc2x0123
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    // 同上，但使用 vxa3 和 vacc3x0123

    const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
    // 从内存地址 (uintptr_t)w + 8 处加载 64 位数据，存入 vb1
    const __m128i vxb1 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);
    // 使用 _mm_unpacklo_epi8 解包 vb1 和 vzero 成 16 位整数，然后与 vb_zero_point 执行减法，存入 vxb1

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 使用 _mm_shuffle_epi32 对 vxa0 进行按指定顺序重排，然后与 vxb1 执行乘加运算，结果累加到 vacc0x0123
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 同上，但使用 vxa1 和 vacc1x0123
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 同上，但使用 vxa2 和 vacc2x0123
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 同上，但使用 vxa3 和 vacc3x0123

    const __m128i vb2 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
    // 从内存地址 (uintptr_t)w + 16 处加载 64 位数据，存入 vb2
    const __m128i vxb2 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);
    // 使用 _mm_unpacklo_epi8 解包 vb2 和 vzero 成 16 位整数，然后与 vb_zero_point 执行减法，存入 vxb2

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    // 使用 _mm_shuffle_epi32 对 vxa0 进行按指定顺序重排，然后与 vxb2 执行乘加运算，结果累加到 vacc0x0123
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    // 同上，但使用 vxa1 和 vacc1x0123
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    // 同上，但使用 vxa2 和 vacc2x0123
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
    // 同上，但使用 vxa3 和 vacc3x0123

    const __m128i vb3 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
    // 从内存地址 (uintptr_t)w + 24 处加载 64 位数据，存入 vb3
    const __m128i vxb3 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);
    // 使用 _mm_unpacklo_epi8 解包 vb3 和 vzero 成 16 位整数，然后与 vb_zero_point 执行减法，存入 vxb3
    w = (const void*)((uintptr_t)w + 32);
    // 将 w 的地址向后移动 32 字节，指向下一个数据块的起始位置

    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    // 使用 _mm_shuffle_epi32 对 vxa0 进行按指定顺序重排，然后与 vxb3 执行乘加运算，结果累加到 vacc0x0123
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    // 同上，但使用 vxa1 和 vacc1x0123
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    // 同上，但使用 vxa2 和 vacc2x0123
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
    // 同上，但使用 vxa3 和 vacc3x0123
  }
  if (k != 0) {
    # 计算预先减量，以便得到正确的地址偏移量
    const size_t a_predecrement = 8 - k;
    
    # 创建一个包含位移值的 128 位整数，用于右移操作
    const __m128i va_shift = _mm_cvtsi32_si128(8 * a_predecrement);
    
    # 从地址 a0 - a_predecrement 处加载 64 位数据，然后使用 va_shift 进行右移操作
    const __m128i va0 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a0 - a_predecrement)), va_shift);
    # 解包 va0 中的字节并且执行零点减法，结果存储在 vxa0 中
    const __m128i vxa0 =
        sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
    
    # 类似地，处理 a1 的数据
    const __m128i va1 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a1 - a_predecrement)), va_shift);
    const __m128i vxa1 =
        sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
    
    # 类似地，处理 a2 的数据
    const __m128i va2 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a2 - a_predecrement)), va_shift);
    const __m128i vxa2 =
        sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
    
    # 类似地，处理 a3 的数据
    const __m128i va3 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a3 - a_predecrement)), va_shift);
    const __m128i vxa3 =
        sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
    
    # 加载地址 w 处的 64 位数据，并解包成 128 位数据，执行零点减法，存储在 vxb0 中
    const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
    const __m128i vxb0 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);
    
    # 将 vxa0 的第一个元素和 vxb0 执行点乘操作，并加到 vacc0x0123 中
    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    
    # 类似地，将 vxa1 的第一个元素和 vxb0 执行点乘操作，并加到 vacc1x0123 中
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    
    # 类似地，将 vxa2 的第一个元素和 vxb0 执行点乘操作，并加到 vacc2x0123 中
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    
    # 类似地，将 vxa3 的第一个元素和 vxb0 执行点乘操作，并加到 vacc3x0123 中
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
    // 如果 k 大于 2，加载偏移量为 8 的内存位置的 8 字节数据，并将其转换为 __m128i 类型
    const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
    // 将 vb1 的低八位展开成 16 位整数，并减去零点偏移 vb_zero_point
    const __m128i vxb1 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

    // 计算 vxa0 中按 1-1-1-1 顺序排列的元素与 vxb1 的点乘积并累加到 vacc0x0123
    vacc0x0123 = _mm_add_epi32(
        vacc0x0123,
        _mm_madd_epi16(
            _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 计算 vxa1 中按 1-1-1-1 顺序排列的元素与 vxb1 的点乘积并累加到 vacc1x0123
    vacc1x0123 = _mm_add_epi32(
        vacc1x0123,
        _mm_madd_epi16(
            _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 计算 vxa2 中按 1-1-1-1 顺序排列的元素与 vxb1 的点乘积并累加到 vacc2x0123
    vacc2x0123 = _mm_add_epi32(
        vacc2x0123,
        _mm_madd_epi16(
            _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
    // 计算 vxa3 中按 1-1-1-1 顺序排列的元素与 vxb1 的点乘积并累加到 vacc3x0123
    vacc3x0123 = _mm_add_epi32(
        vacc3x0123,
        _mm_madd_epi16(
            _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

    // 如果 k 大于 4
    if (k > 4) {
        // 加载偏移量为 16 的内存位置的 8 字节数据，并将其转换为 __m128i 类型
        const __m128i vb2 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
        // 将 vb2 的低八位展开成 16 位整数，并减去零点偏移 vb_zero_point
        const __m128i vxb2 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

        // 计算 vxa0 中按 2-2-2-2 顺序排列的元素与 vxb2 的点乘积并累加到 vacc0x0123
        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        // 计算 vxa1 中按 2-2-2-2 顺序排列的元素与 vxb2 的点乘积并累加到 vacc1x0123
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        // 计算 vxa2 中按 2-2-2-2 顺序排列的元素与 vxb2 的点乘积并累加到 vacc2x0123
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        // 计算 vxa3 中按 2-2-2-2 顺序排列的元素与 vxb2 的点乘积并累加到 vacc3x0123
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

        // 如果 k 大于 6
        if (k > 6) {
            // 加载偏移量为 24 的内存位置的 8 字节数据，并将其转换为 __m128i 类型
            const __m128i vb3 =
                _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
            // 将 vb3 的低八位展开成 16 位整数，并减去零点偏移 vb_zero_point
            const __m128i vxb3 =
                _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);

            // 计算 vxa0 中按 3-3-3-3 顺序排列的元素与 vxb3 的点乘积并累加到 vacc0x0123
            vacc0x0123 = _mm_add_epi32(
                vacc0x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
            // 计算 vxa1 中按 3-3-3-3 顺序排列的元素与 vxb3 的点乘积并累加到 vacc1x0123
            vacc1x0123 = _mm_add_epi32(
                vacc1x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
            // 计算 vxa2 中按 3-3-3-3 顺序排列的元素与 vxb3 的点乘积并累加到 vacc2x0123
            vacc2x0123 = _mm_add_epi32(
                vacc2x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
            // 计算 vxa3 中按 3-3-3-3 顺序排列的元素与 vxb3 的点乘积并累加到 vacc3x0123
            vacc3x0123 = _mm_add_epi32(
                vacc3x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
        }
    }
  }
}

const __m128 vmultiplier =
    _mm_loadu_ps(&quantization_params->sse2.requantization_scales[output_channel_index]);
// 加载输出通道索引处的重新量化比例到 __m128 类型的变量 vmultiplier

vacc0x0123 = _mm_cvtps_epi32(
              _mm_mul_ps(
                _mm_cvtepi32_ps(vacc0x0123),
                vmultiplier
                )
              );
// 将 vacc0x0123 向量转换为浮点型后乘以 vmultiplier 向量，再转换回整型

vacc1x0123 = _mm_cvtps_epi32(
              _mm_mul_ps(
                _mm_cvtepi32_ps(vacc1x0123),
                vmultiplier
                )
              );
// 同上，处理 vacc1x0123 向量

vacc2x0123 = _mm_cvtps_epi32(
              _mm_mul_ps(
                _mm_cvtepi32_ps(vacc2x0123),
                vmultiplier
                )
              );
// 同上，处理 vacc2x0123 向量

vacc3x0123 = _mm_cvtps_epi32(
              _mm_mul_ps(
                _mm_cvtepi32_ps(vacc3x0123),
                vmultiplier
                )
              );
// 同上，处理 vacc3x0123 向量

const __m128i voutput_zero_point = _mm_load_si128(
    (const __m128i*)quantization_params->sse2.output_zero_point);
// 加载输出零点到 __m128i 类型的变量 voutput_zero_point

const __m128i vacc01x0123 = _mm_adds_epi16(
    _mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
// 将 vacc0x0123 和 vacc1x0123 向量分别打包成 16 位整型，然后加上 voutput_zero_point

const __m128i vacc23x0123 = _mm_adds_epi16(
    _mm_packs_epi32(vacc2x0123, vacc3x0123), voutput_zero_point);
// 将 vacc2x0123 和 vacc3x0123 向量分别打包成 16 位整型，然后加上 voutput_zero_point

__m128i vout = _mm_packus_epi16(vacc01x0123, vacc23x0123);
// 将 vacc01x0123 和 vacc23x0123 向量打包成无符号 8 位整型

vout = _mm_min_epu8(
    vout,
    _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
// 将 vout 向量和 quantization_params->sse2.output_max 向量逐元素比较取小值

vout = _mm_max_epu8(
    vout,
    _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));
// 将 vout 向量和 quantization_params->sse2.output_min 向量逐元素比较取大值

uint8_t* c0 = c;
uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
if (mr < 2) {
  c1 = c0;
}
uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
if (mr <= 2) {
  c2 = c1;
}
uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
if (mr != 4) {
  c3 = c2;
}
// 根据 mr 和 nr 的值计算和设置 c0, c1, c2, c3 的地址偏移

if (nr == 4) {
  *((uint32_t*)c0) = (uint32_t)_mm_cvtsi128_si32(vout);
  *((uint32_t*)c1) = (uint32_t)_mm_cvtsi128_si32(_mm_srli_epi64(vout, 32));
  *((uint32_t*)c2) =
      (uint32_t)_mm_cvtsi128_si32(_mm_unpackhi_epi32(vout, vout));
  *((uint32_t*)c3) = (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(vout, 12));
} else {
  typedef PYTORCH_QNNP_UNALIGNED uint16_t unaligned_uint16_t;
  if (nr >= 2) {
    *((unaligned_uint16_t*)c0) = (uint16_t)_mm_extract_epi16(vout, 0);
    c0 += 2;
    *((unaligned_uint16_t*)c1) = (uint16_t)_mm_extract_epi16(vout, 2);
    c1 += 2;
    *((unaligned_uint16_t*)c2) = (uint16_t)_mm_extract_epi16(vout, 4);
    c2 += 2;
    *((unaligned_uint16_t*)c3) = (uint16_t)_mm_extract_epi16(vout, 6);
    c3 += 2;
    vout = _mm_srli_epi32(vout, 16);
    nr -= 2;
  }
  if (nr != 0) {
    *((uint8_t*)c0) = (uint8_t)_mm_cvtsi128_si32(vout);
    *((uint8_t*)c1) = (uint8_t)_mm_extract_epi16(vout, 2);
    *((uint8_t*)c2) = (uint8_t)_mm_extract_epi16(vout, 4);
    *((uint8_t*)c3) = (uint8_t)_mm_extract_epi16(vout, 6);
  }
}
}


注释：


# 这行代码结束了一个代码块，可能是一个函数、循环、条件语句或其他代码结构的结尾。
# 在大多数编程语言中，}符号用于表示代码块的结束。
# 在此示例中，}结束了一个函数或类的定义，具体上下文取决于前面的代码，这里应该是函数定义的结尾。
```