# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\mp8x25-sse2-per-channel.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>  // 包含 AVX 和 SSE 指令集的头文件

#include <qnnpack/q8dwconv.h>  // 包含深度卷积的头文件

void pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2(
    size_t channels,  // 输入通道数
    size_t output_width,  // 输出宽度
    const uint8_t** input,  // 输入数据的指针数组
    const void* weights,  // 权重数据的指针
    int32_t* outacc32,  // 输出累加器的指针
    uint8_t* output,  // 输出数据的指针
    size_t input_stride,  // 输入数据的跨度
    size_t output_increment,  // 输出数据的增量
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {  // 卷积量化参数
  const __m128i vinput_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.input_zero_point);
  // 加载输入的零点偏移量，使用 SSE 指令集

  const __m128i vzero = _mm_setzero_si128();
  // 设置一个全为零的 SSE 寄存器

  do {
    int32_t* outacc = outacc32;
    const void* w = weights;
    // 初始化输出累加器和权重指针
    // 这里应该包含卷积操作的实现，但代码片段截断，无法提供详细操作的注释
    output = (uint8_t*)((uintptr_t)output + output_increment);
    // 更新输出数据指针，以便指向下一个输出位置
  } while (--output_width != 0);
  // 循环直到输出宽度减为零
}
```