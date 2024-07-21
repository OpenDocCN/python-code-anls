# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8zip\xm-neon.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>  // 包含 ARM NEON 指令集的头文件

#include <qnnpack/x8zip.h>  // 包含 x8zip 相关的头文件

void pytorch_qnnp_x8zip_xm__neon(
    size_t n,
    size_t m,
    const void* input,
    void* output) {
  const uint8_t* w = input;  // 指向输入数据的指针，类型为 uint8_t*
  const size_t input_increment = n * 3;  // 输入数据的增量
  const size_t output_increment = 4 - m * n;  // 输出数据的增量
  const uint8_t* last_input = w + n * (m - 1);  // 最后一个输入数据的指针
  void* last_output = (void*)((uintptr_t)output + (m - 4));  // 最后一个输出数据的指针

  if (n >= 8) {  // 如果 n 大于等于 8
    // 略
  } else {
    const uint8_t* i = input;  // 指向输入数据的指针，类型为 uint8_t*
    uint8_t* o = output;  // 指向输出数据的指针，类型为 uint8_t*
    size_t k = n;  // 循环变量 k 初始化为 n
    do {
      size_t l = m;  // 循环变量 l 初始化为 m
      const uint8_t* ii = i++;  // 指向当前输入数据的指针，类型为 uint8_t*
      do {
        *o++ = *ii;  // 将当前输入数据写入输出数据
        ii += n;  // 指向下一个输入数据
      } while (--l != 0);  // l 自减，直到为 0 结束内层循环
    } while (--k != 0);  // k 自减，直到为 0 结束外层循环
  }
}
```