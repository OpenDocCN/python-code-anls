# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8clamp\sse2.c`

```py
/*
 * 实现了一个使用 SSE2 指令集的函数，用于对输入的 uint8_t 数组进行范围限制操作。
 * 如果输入的元素数量大于等于 8，将使用 SSE2 向量化指令进行处理，否则使用标量处理。
 */
void pytorch_u8clamp_ukernel__sse2(
    size_t n,  // 输入数组的元素数量
    const uint8_t* x,  // 输入的 uint8_t 数组指针
    uint8_t* y,  // 输出的 uint8_t 数组指针
    const union pytorch_qnnp_u8_clamping_params params[RESTRICT_STATIC 1]) {  // 参数结构体数组，包含了限制范围的参数
  assert(n != 0);  // 断言，确保输入的元素数量不为零

  if
    PYTORCH_QNNP_LIKELY(n >= 8) {  // 如果元素数量大于等于 8，则使用 SSE2 向量化指令处理
      const __m128i voutput_max =  // 加载输出的最大限制值到 SSE2 寄存器
          _mm_load_si128((const __m128i*)&params->sse2.output_max);
      const __m128i voutput_min =  // 加载输出的最小限制值到 SSE2 寄存器
          _mm_load_si128((const __m128i*)&params->sse2.output_min);
      for (; n >= 64; n -= 64) {  // 每次处理 64 个元素
        const __m128i vx0 = _mm_loadu_si128((const __m128i*)x);  // 加载 16 字节的输入数据
        const __m128i vx1 = _mm_loadu_si128((const __m128i*)x + 1);  // 加载接下来的 16 字节数据
        const __m128i vx2 = _mm_loadu_si128((const __m128i*)x + 2);  // 加载再接下来的 16 字节数据
        const __m128i vx3 = _mm_loadu_si128((const __m128i*)x + 3);  // 加载最后的 16 字节数据
        x += 64;  // 指向下一个处理的输入数据

        const __m128i vy0 =  // 对输入数据进行限制在最大和最小值之间的处理
            _mm_min_epu8(_mm_max_epu8(vx0, voutput_min), voutput_max);
        const __m128i vy1 =
            _mm_min_epu8(_mm_max_epu8(vx1, voutput_min), voutput_max);
        const __m128i vy2 =
            _mm_min_epu8(_mm_max_epu8(vx2, voutput_min), voutput_max);
        const __m128i vy3 =
            _mm_min_epu8(_mm_max_epu8(vx3, voutput_min), voutput_max);

        __builtin_prefetch(x + 640);  // 预取下一个处理的数据

        _mm_storeu_si128((__m128i*)y, vy0);  // 存储处理后的结果到输出数组
        _mm_storeu_si128((__m128i*)y + 1, vy1);
        _mm_storeu_si128((__m128i*)y + 2, vy2);
        _mm_storeu_si128((__m128i*)y + 3, vy3);
        y += 64;  // 指向下一个存储位置
      }
      for (; n >= 8; n -= 8) {  // 如果剩余元素数量大于等于 8，但不足 64 个，则按 8 个元素一组处理
        __m128i vout = _mm_loadl_epi64((const __m128i*)x);  // 加载 8 字节的输入数据
        x += 8;  // 指向下一个处理的输入数据
        vout = _mm_min_epu8(vout, voutput_max);  // 对输入数据进行最大最小值限制处理
        vout = _mm_max_epu8(vout, voutput_min);
        _mm_storel_epi64((__m128i*)y, vout);  // 存储处理后的结果到输出数组
        y += 8;  // 指向下一个存储位置
      }
      if (n != 0) {  // 处理剩余不足 8 个元素的情况
        const size_t n_increment = n - 8;  // 计算剩余元素数量与 8 的差值
        x = (const uint8_t*)((uintptr_t)x + n_increment);  // 调整输入指针位置
        y = (uint8_t*)((uintptr_t)y + n_increment);  // 调整输出指针位置

        __m128i vout = _mm_loadl_epi64((const __m128i*)x);  // 加载不足 8 字节的输入数据
        vout = _mm_min_epu8(vout, voutput_max);  // 对输入数据进行最大最小值限制处理
        vout = _mm_max_epu8(vout, voutput_min);
        _mm_storel_epi64((__m128i*)y, vout);  // 存储处理后的结果到输出数组
      }
    }
  else {  // 如果元素数量小于 8，则使用标量处理
    const uint32_t voutput_max = params->sse2.output_max[0];  // 加载输出的最大限制值
    const uint32_t voutput_min = params->sse2.output_min[0];  // 加载输出的最小限制值
    do {
      uint32_t vout = *x++;  // 加载单个输入数据
      vout = vout > voutput_max ? voutput_max : vout;  // 对输入数据进行最大最小值限制处理
      vout = vout < voutput_min ? voutput_min : vout;
      *y++ = (uint8_t)vout;  // 存储处理后的结果到输出数组
    } while (--n != 0);  // 循环直到处理完所有元素
  }
}
```