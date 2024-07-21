# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8zip\xm-sse2.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emmintrin.h>

#include <qnnpack/x8zip.h>

void pytorch_qnnp_x8zip_xm__sse2(
    size_t n,
    size_t m,
    const void* input,
    void* output) {
  // 将输入指针转换为 uint8_t 类型
  const uint8_t* w = input;
  // 计算每次循环中输入指针的增量
  const size_t input_increment = n * 3;
  // 计算每次循环中输出指针的增量
  const size_t output_increment = 4 - m * n;
  // 计算最后一个输入元素的指针位置
  const uint8_t* last_input = w + n * (m - 1);
  // 计算最后一个输出元素的指针位置
  void* last_output = (void*)((uintptr_t)output + (m - 4));

  // 如果 n 大于等于 8，则执行下面的代码块
  if (n >= 8) {
    // 留空，暂无具体实现
  } else { // 如果 n 小于 8，则执行下面的代码块
    // 将输入指针转换为 uint8_t 类型
    const uint8_t* i = input;
    // 将输出指针转换为 uint8_t 类型
    uint8_t* o = output;
    // 初始化循环计数器 k 为 n
    size_t k = n;
    // 外层循环，每次循环迭代处理一个输入元素
    do {
      // 初始化内层循环计数器 l 为 m
      size_t l = m;
      // 内层循环，每次循环迭代处理一个输出元素
      const uint8_t* ii = i++;
      do {
        // 将当前输入指针 ii 指向的数据写入到输出指针 o 指向的位置
        *o++ = *ii;
        // 更新 ii 指针，移动到下一个输入元素
        ii += n;
      } while (--l != 0); // 内层循环结束条件，处理完所有输出元素
    } while (--k != 0); // 外层循环结束条件，处理完所有输入元素
  }
}
```