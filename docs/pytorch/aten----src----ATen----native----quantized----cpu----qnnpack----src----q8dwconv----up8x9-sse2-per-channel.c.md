# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\up8x9-sse2-per-channel.c`

```py
/*
 * 使用 AVX 指令集实现的深度可分离卷积的 SSE2 版本的微内核函数
 * 实现 8x9 的输入通道扩展，用于逐通道量化的卷积操作
 */
void pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2(
    size_t channels,  // 输入通道数
    size_t output_width,  // 输出宽度（每行像素数）
    const uint8_t** input,  // 输入数据指针的数组
    const void* weights,  // 卷积核权重
    uint8_t* output,  // 输出数据指针
    size_t input_stride,  // 输入行步长
    size_t output_increment,  // 输出行步长
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {  // 量化参数结构体数组
  const __m128i va_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.input_zero_point);  // 加载输入零点偏移量
  const __m128i vzero = _mm_setzero_si128();  // 设置零向量

  do {
    const uint8_t* i0 = input[0];  // 加载输入数据指针
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];

    input = (const uint8_t**)((uintptr_t)input + input_stride);  // 更新输入指针数组

    size_t c = channels;  // 初始化通道数
    const void* w = weights;  // 加载卷积核权重指针
    }
    }

    output = (uint8_t*)((uintptr_t)output + output_increment);  // 更新输出指针
  } while (--output_width != 0);  // 检查是否还有输出行待处理
}
```