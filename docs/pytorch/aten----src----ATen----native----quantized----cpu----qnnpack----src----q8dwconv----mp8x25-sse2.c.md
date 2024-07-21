# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\mp8x25-sse2.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/q8dwconv.h>

// 定义 SSE2 下的多通道 8 比特深度深度可分离卷积的内核函数
void pytorch_q8dwconv_ukernel_mp8x25__sse2(
    size_t channels,                                               // 输入通道数
    size_t output_width,                                           // 输出宽度
    const uint8_t** input,                                         // 输入数据指针的数组
    const void* weights,                                           // 权重数据指针
    int32_t* outacc32,                                             // 输出累加器数组
    uint8_t* output,                                               // 输出数据指针
    size_t input_stride,                                           // 输入数据跨度
    size_t output_increment,                                       // 输出增量
    const union pytorch_qnnp_conv_quantization_params              // 卷积量化参数
        quantization_params[RESTRICT_STATIC 1]) {
  const __m128i vinput_zero_point = _mm_load_si128(                 // 加载输入零点偏移量到 SSE2 寄存器
      (const __m128i*)quantization_params->sse2.input_zero_point);
  const __m128i vkernel_zero_point = _mm_set1_epi16(               // 设置权重零点偏移量到 SSE2 寄存器
      quantization_params->sse2.kernel_zero_points[0]);
  const __m128i vzero = _mm_setzero_si128();                       // 设置零向量到 SSE2 寄存器

  do {
    int32_t* outacc = outacc32;                                    // 输出累加器指针
    const void* w = weights;                                       // 当前处理的权重数据指针
    // 处理当前输出的多通道数据
    // （此处省略了具体的处理过程，需要根据具体的实现细节补充注释）
    output = (uint8_t*)((uintptr_t)output + output_increment);      // 更新输出指针位置
  } while (--output_width != 0);                                    // 循环直到处理完所有输出宽度
}
```