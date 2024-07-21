# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8clamp\neon.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <arm_neon.h>  // 引入 ARM NEON SIMD 指令集头文件

#include <qnnpack/u8clamp.h>  // 引入 QNNPACK 中的 uint8_t 类型的 clamp 函数头文件

void pytorch_u8clamp_ukernel__neon(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union pytorch_qnnp_u8_clamping_params params[restrict static 1]) {
  assert(n != 0);  // 断言输入尺寸 n 不为 0

  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);  // 加载输出最大值为 16 个字节
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);  // 加载输出最小值为 16 个字节

  if
    PYTORCH_QNNP_LIKELY(n >= 8) {  // 如果 n 大于等于 8，执行以下代码块
      for (; n >= 64; n -= 64) {  // 当 n 大于等于 64 时，每次处理 64 个元素
        const uint8x16_t vx0 = vld1q_u8(x);  // 加载输入向量 x 的第一个 16 字节
        x += 16;  // 指针移动到下一个 16 字节
        const uint8x16_t vx1 = vld1q_u8(x);  // 加载输入向量 x 的第二个 16 字节
        x += 16;  // 指针移动到下一个 16 字节
        const uint8x16_t vx2 = vld1q_u8(x);  // 加载输入向量 x 的第三个 16 字节
        x += 16;  // 指针移动到下一个 16 字节
        const uint8x16_t vx3 = vld1q_u8(x);  // 加载输入向量 x 的第四个 16 字节
        x += 16;  // 指针移动到下一个 16 字节

        const uint8x16_t vy0 =
            vminq_u8(vmaxq_u8(vx0, voutput_min), voutput_max);  // 对 vx0 进行上下限约束操作
        const uint8x16_t vy1 =
            vminq_u8(vmaxq_u8(vx1, voutput_min), voutput_max);  // 对 vx1 进行上下限约束操作
        const uint8x16_t vy2 =
            vminq_u8(vmaxq_u8(vx2, voutput_min), voutput_max);  // 对 vx2 进行上下限约束操作
        const uint8x16_t vy3 =
            vminq_u8(vmaxq_u8(vx3, voutput_min), voutput_max);  // 对 vx3 进行上下限约束操作

        __builtin_prefetch(x + 640);  // 预取下一个数据块，增加数据预取性能

        vst1q_u8(y, vy0);  // 存储 vy0 到输出向量 y
        y += 16;  // 指针移动到下一个 16 字节
        vst1q_u8(y, vy1);  // 存储 vy1 到输出向量 y
        y += 16;  // 指针移动到下一个 16 字节
        vst1q_u8(y, vy2);  // 存储 vy2 到输出向量 y
        y += 16;  // 指针移动到下一个 16 字节
        vst1q_u8(y, vy3);  // 存储 vy3 到输出向量 y
        y += 16;  // 指针移动到下一个 16 字节
      }
      for (; n >= 8; n -= 8) {  // 当 n 大于等于 8 时，每次处理 8 个元素
        uint8x8_t vout = vld1_u8(x);  // 加载输入向量 x 的 8 个字节
        x += 8;  // 指针移动到下一个 8 字节
        vout = vmin_u8(vout, vget_low_u8(voutput_max));  // 对 vout 进行上限约束操作
        vout = vmax_u8(vout, vget_low_u8(voutput_min));  // 对 vout 进行下限约束操作
        vst1_u8(y, vout);  // 存储 vout 到输出向量 y
        y += 8;  // 指针移动到下一个 8 字节
      }
      if (n != 0) {  // 如果剩余元素不为 0
        const size_t n_increment = n - 8;  // 计算需要增加的偏移量
        x = (const uint8_t*)((uintptr_t)x + n_increment);  // 更新输入指针位置
        y = (uint8_t*)((uintptr_t)y + n_increment);  // 更新输出指针位置

        uint8x8_t vout = vld1_u8(x);  // 加载剩余输入向量的 8 个字节
        vout = vmin_u8(vout, vget_low_u8(voutput_max));  // 对 vout 进行上限约束操作
        vout = vmax_u8(vout, vget_low_u8(voutput_min));  // 对 vout 进行下限约束操作
        vst1_u8(y, vout);  // 存储 vout 到输出向量 y
      }
    }
  else {  // 如果 n 小于 8，使用非向量化方式处理
    do {
      uint8x8_t vout = vld1_dup_u8(x);  // 加载输入向量 x 的 8 个字节，扩展为 16 个字节
      x += 1;  // 指针移动到下一个字节
      vout = vmin_u8(vout, vget_low_u8(voutput_max));  // 对 vout 进行上限约束操作
      vout = vmax_u8(vout, vget_low_u8(voutput_min));  // 对 vout 进行下限约束操作
      vst1_lane_u8(y, vout, 0);  // 存储 vout 到输出向量 y
      y += 1;  // 指针移动到下一个字节
    } while (--n != 0);  // 继续处理直到 n 减为 0
  }
}
```