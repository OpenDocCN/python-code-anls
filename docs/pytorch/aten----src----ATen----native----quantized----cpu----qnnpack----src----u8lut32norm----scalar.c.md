# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8lut32norm\scalar.c`

```
/*
c
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <fxdiv.h>

#include <qnnpack/u8lut32norm.h>

// 定义静态内联函数，计算输入数组 x 和 t 的加权和
static inline uint32_t compute_sum(
    size_t n,                 // 输入数组 x 的长度
    const uint8_t* x,         // 输入数组 x，存储 uint8_t 类型的数据
    const uint32_t* t) {      // 输入数组 t，存储 uint32_t 类型的数据
  assert(n != 0);             // 断言：输入长度 n 不为零

  uint32_t vsum = 0;          // 初始化加权和为 0
  do {
    const size_t vx = *x++;   // 获取数组 x 当前位置的值，并移动到下一个位置
    vsum += t[vx];            // 将 t 数组中对应 x[vx] 的值加到加权和中
  } while (--n != 0);         // 递减 n，并循环直到 n 为零
  return vsum;                // 返回计算得到的加权和
}

// 定义函数，实现基于标量的 u8lut32norm 内核
void pytorch_u8lut32norm_ukernel__scalar(
    size_t n,                 // 输入数组长度
    const uint8_t* x,         // 输入数组 x，存储 uint8_t 类型的数据
    const uint32_t* t,        // 输入数组 t，存储 uint32_t 类型的数据
    uint8_t* y) {             // 输出数组 y，存储 uint8_t 类型的数据
  assert(n != 0);             // 断言：输入长度 n 不为零

  // 计算输入数组 x 和 t 的加权和
  const uint32_t vsum = compute_sum(n, x, t);
  assert(vsum != 0);          // 断言：加权和 vsum 不为零

  // 使用加权和初始化 fxdiv_divisor_uint32_t 结构体
  struct fxdiv_divisor_uint32_t vsum_divisor = fxdiv_init_uint32_t(vsum);
  const uint32_t vrounding = (vsum >> 1);  // 计算舍入值

  do {
    const size_t vx = *x++;   // 获取数组 x 当前位置的值，并移动到下一个位置
    const uint32_t vt = t[vx];  // 获取数组 t 中对应 x[vx] 的值
    // 使用 fxdiv 库计算 vt * 256 + vrounding 除以 vsum_divisor 的商
    const uint32_t vq =
        fxdiv_quotient_uint32_t((vt << 8) + vrounding, vsum_divisor);
    // 将计算得到的 vq 转换为 uint8_t 类型，并限制在 0 到 255 之间
    const uint8_t vy = vq > 255 ? UINT8_C(255) : (uint8_t)vq;
    *y++ = vy;                // 将结果写入输出数组 y
  } while (--n != 0);         // 递减 n，并循环直到 n 为零
}
```