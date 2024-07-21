# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8lut\scalar.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <qnnpack/x8lut.h>

// 定义了一个函数 pytorch_x8lut_ukernel__scalar，用于对输入数组 x 中的每个元素进行查找表映射，并将结果存入数组 y 中
void pytorch_x8lut_ukernel__scalar(
    size_t n,                                      // 输入数组 x 和输出数组 y 的元素数量
    const uint8_t* x,                              // 输入数组 x，包含待映射的元素
    const uint8_t t[RESTRICT_STATIC 256],          // 查找表 t，用于将输入 x 的值映射到输出 y 的值
    uint8_t* y) {                                  // 输出数组 y，存储映射后的结果
  assert(n != 0);                                  // 断言：n 不为 0，确保输入数组的长度大于 0

  while (n >= 4) {                                 // 循环处理，每次处理四个元素
    const size_t vx0 = x[0];                       // 取输入数组 x 的第一个元素
    const size_t vx1 = x[1];                       // 取输入数组 x 的第二个元素
    const size_t vx2 = x[2];                       // 取输入数组 x 的第三个元素
    const size_t vx3 = x[3];                       // 取输入数组 x 的第四个元素
    x += 4;                                         // 将输入指针 x 向后移动四个位置

    const uint8_t vt0 = t[vx0];                    // 使用查找表 t 映射输入元素 vx0 的值
    const uint8_t vt1 = t[vx1];                    // 使用查找表 t 映射输入元素 vx1 的值
    const uint8_t vt2 = t[vx2];                    // 使用查找表 t 映射输入元素 vx2 的值
    const uint8_t vt3 = t[vx3];                    // 使用查找表 t 映射输入元素 vx3 的值

    y[0] = vt0;                                    // 将映射结果 vt0 存入输出数组 y 的第一个位置
    y[1] = vt1;                                    // 将映射结果 vt1 存入输出数组 y 的第二个位置
    y[2] = vt2;                                    // 将映射结果 vt2 存入输出数组 y 的第三个位置
    y[3] = vt3;                                    // 将映射结果 vt3 存入输出数组 y 的第四个位置
    y += 4;                                         // 将输出指针 y 向后移动四个位置

    n -= 4;                                         // 减少处理的元素数量
  }
  while (n != 0) {                                 // 处理剩余不足四个元素的情况
    const size_t vx = *x++;                        // 取输入数组 x 的当前元素并向后移动一个位置
    const uint8_t vt = t[vx];                      // 使用查找表 t 映射输入元素 vx 的值
    *y++ = vt;                                      // 将映射结果 vt 存入输出数组 y 的当前位置

    n--;                                            // 减少处理的元素数量
  };
}
```