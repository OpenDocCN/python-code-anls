# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8maxpool\16x9p8q-sse2.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>  // 包含 SSE2 指令集的头文件

#include <qnnpack/u8maxpool.h>  // 包含 QNNPACK 的无符号 8 位最大池化函数声明

void pytorch_u8maxpool_ukernel_16x9p8q__sse2(
    size_t n,  // 输入通道数
    size_t ks,  // 池化窗口的尺寸
    size_t kc,  // 输入通道数，要求至少为 16
    const uint8_t** input,  // 输入数据的指针数组
    uint8_t* output,  // 输出数据的指针
    size_t input_increment,  // 输入数据增量
    size_t output_increment,  // 输出数据增量
    const union pytorch_qnnp_u8_clamping_params params[RESTRICT_STATIC 1]) {  // QNNPACK 参数结构体数组
  assert(n != 0);  // 断言输入通道数非零
  assert(ks != 0);  // 断言池化窗口尺寸非零
  assert(kc >= 16);  // 断言输入通道数至少为 16

  const __m128i voutput_max =  // 加载输出上限到 SSE2 寄存器
      _mm_load_si128((const __m128i*)params->sse2.output_max);
  const __m128i voutput_min =  // 加载输出下限到 SSE2 寄存器
      _mm_load_si128((const __m128i*)params->sse2.output_min);

  do {  // 循环处理每个输入通道
    uint8_t* o = output;  // 指向输出数据的指针

    // 实现池化操作，暂时省略具体的池化过程
    // 在此处应包含池化操作的代码

    input = (const uint8_t**)((uintptr_t)input + input_increment);  // 更新输入数据指针数组
    output = (uint8_t*)((uintptr_t)o + output_increment);  // 更新输出数据指针
  } while (--n != 0);  // 循环直到处理完所有输入通道
}
```