# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8zip\x3-sse2.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emmintrin.h>  // 包含 SSE2 指令集的头文件

#include <qnnpack/x8zip.h>  // 包含 x8zip 相关的头文件

void pytorch_qnnp_x8zip_x3__sse2(size_t n, const void* input, void* output) {
  const uint8_t* x = input;  // 声明指向输入数据的指针 x
  const uint8_t* y = x + n;  // 声明指向输入数据的指针 y，偏移 n 字节
  const uint8_t* z = y + n;  // 声明指向输入数据的指针 z，偏移另外 n 字节
  uint8_t* o = output;       // 声明指向输出数据的指针 o

  if (n >= 16) {  // 如果输入数据长度大于等于 16
    const __m128i vmask0x00FF00FF = _mm_set1_epi16(0x00FF);   // 声明并初始化 SSE 寄存器，用于处理 16 位整数数据
    const __m128i vmask0x0000FFFF = _mm_set1_epi32(0x0000FFFF);  // 声明并初始化 SSE 寄存器，用于处理 32 位整数数据
    } while (n >= 16);  // 进入循环处理 16 字节数据块
    }
  } else {  // 如果输入数据长度小于 16
    do {
      const uint8_t vx = *x++;  // 从指针 x 处读取一个字节，赋值给 vx，然后指针 x 向后移动一个字节
      const uint8_t vy = *y++;  // 从指针 y 处读取一个
```