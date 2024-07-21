# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\up8x9-sse2.c`

```
/*
 * 该函数是用于执行 8 通道输入和 9 通道内核的深度可分离卷积的特定微内核，基于 SSE2 指令集实现。
 */
void pytorch_q8dwconv_ukernel_up8x9__sse2(
    // 输入通道数
    size_t channels,
    // 输出宽度
    size_t output_width,
    // 输入数据的指针数组，包含 9 个输入通道的数据
    const uint8_t** input,
    // 卷积核的权重
    const void* weights,
    // 输出数据的指针
    uint8_t* output,
    // 输入数据的跨度
    size_t input_stride,
    // 输出数据的增量
    size_t output_increment,
    // 量化参数结构体数组，包含量化参数
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  
  // 加载输入的零点，作为 SSE2 寄存器 __m128i
  const __m128i va_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.input_zero_point);
  
  // 设置卷积核零点为 SSE2 寄存器 __m128i
  const __m128i vkernel_zero_point = _mm_set1_epi16(
      quantization_params->sse2.kernel_zero_points[0]);
  
  // 设置一个零值的 SSE2 寄存器 __m128i
  const __m128i vzero = _mm_setzero_si128();

  // 处理每个输出宽度
  do {
    // 按行加载 9 通道输入数据
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];

    // 更新输入指针数组到下一行数据
    input = (const uint8_t**)((uintptr_t)input + input_stride);

    // 循环处理每个通道
    size_t c = channels;
    const void* w = weights;

    // 更新输出指针到下一个输出元素
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);  // 继续直到处理完所有的输出宽度
}
```