# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8maxpool\sub16-neon.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <arm_neon.h>

#include <qnnpack/u8maxpool.h>

// 定义名为 pytorch_u8maxpool_ukernel_sub16__neon 的函数，使用 NEON 指令进行无符号 8 位最大池化
void pytorch_u8maxpool_ukernel_sub16__neon(
    size_t n,  // n 表示处理的输出张量的数量
    size_t ks, // ks 表示池化核大小
    size_t kc, // kc 表示通道数，且必须小于 16
    const uint8_t** input, // 指向输入张量的指针数组
    uint8_t* output,       // 指向输出张量的指针
    size_t input_increment,    // 输入张量中每行增量
    size_t output_increment,   // 输出张量中每行增量
    const union pytorch_qnnp_u8_clamping_params params[restrict static 1]) { // 用于限制输出范围的参数结构体数组
  assert(n != 0);   // 断言确保 n 不为零
  assert(ks != 0);  // 断言确保 ks 不为零
  assert(kc != 0);  // 断言确保 kc 不为零
  assert(kc < 16);  // 断言确保 kc 小于 16

  // 使用 NEON 指令加载输出最大值和最小值到 uint8x16_t 向量
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);

  // 外层循环，处理每个输出张量
  do {
    uint8x16_t vmax = vmovq_n_u8(0);  // 初始化 vmax 为全零向量

    size_t m = ks;   // 设定内层循环次数为 ks
    do {
      const uint8_t* i = *input++;  // 获取输入指针指向的数据，并移动到下一个输入张量
      i += kc;  // 跳过 kc 个字节，以处理每个输入张量的剩余数据
      uint8x16_t vi = vmax;  // vi 初始化为 vmax
      if (kc & 1) {
        i -= 1;
        vi = vld1q_lane_u8(i, vi, 0);  // 如果 kc 是奇数，加载一个字节到 vi 的最低位
      }
      if (kc & 2) {
        vi = vextq_u8(vi, vi, 14);  // 如果 kc 包含 2，使用 vextq_u8 扩展 vi
        i -= 2;
        vi = vreinterpretq_u8_u16(vld1q_lane_u16(
            __builtin_assume_aligned(i, 1), vreinterpretq_u16_u8(vi), 0));  // 加载一个短整数到 vi
      }
      if (kc & 4) {
        vi = vextq_u8(vi, vi, 12);  // 如果 kc 包含 4，使用 vextq_u8 扩展 vi
        i -= 4;
        vi = vreinterpretq_u8_u32(vld1q_lane_u32(
            __builtin_assume_aligned(i, 1), vreinterpretq_u32_u8(vi), 0));  // 加载一个整数到 vi
      }
      if (kc & 8) {
        i -= 8;
        vi = vcombine_u8(vld1_u8(i), vget_low_u8(vi));  // 如果 kc 包含 8，使用 vcombine_u8 组合两个 uint8x8_t 向量
      }
      vmax = vmaxq_u8(vmax, vi);  // 更新 vmax，保留每个位置的最大值
    } while (--m != 0);  // 内层循环，直到 m 减为零

    input = (const uint8_t**)((uintptr_t)input + input_increment);  // 更新输入指针数组的位置

    vmax = vminq_u8(vmax, voutput_max);  // 使用 vminq_u8 将 vmax 限制在 voutput_max 内
    vmax = vmaxq_u8(vmax, voutput_min);  // 使用 vmaxq_u8 将 vmax 限制在 voutput_min 内

    uint8x8_t vout = vget_low_u8(vmax);  // 获取 vmax 的低位作为输出向量的一部分
    if (kc & 8) {
      vst1_u8(output, vout);  // 如果 kc 包含 8，使用 vst1_u8 存储 vout 到输出指针位置
      output += 8;  // 更新输出指针位置
      vout = vget_high_u8(vmax);  // 获取 vmax 的高位作为输出向量的一部分
    }
    if (kc & 4) {
      vst1_lane_u32(
          __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);  // 如果 kc 包含 4，使用 vst1_lane_u32 存储 vout 到输出指针位置
      output += 4;  // 更新输出指针位置
      vout = vext_u8(vout, vout, 4);  // 使用 vext_u8 扩展 vout
    }
    if (kc & 2) {
      vst1_lane_u16(
          __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);  // 如果 kc 包含 2，使用 vst1_lane_u16 存储 vout 到输出指针位置
      output += 2;  // 更新输出指针位置
      vout = vext_u8(vout, vout, 2);  // 使用 vext_u8 扩展 vout
    }
    if (kc & 1) {
      vst1_lane_u8(output, vout, 0);  // 如果 kc 包含 1，使用 vst1_lane_u8 存储 vout 到输出指针位置
      output += 1;  // 更新输出指针位置
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);  // 更新输出指针位置

  } while (--n != 0);  // 外层循环，直到 n 减为零
}
```