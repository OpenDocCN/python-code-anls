# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\4x-sumrows-neon.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h> // 包含 ARM NEON 指令集的头文件

#include <qnnpack/q8gemm.h> // 包含 QNNPACK Q8 矩阵乘法相关的头文件

void pytorch_q8sumrows_ukernel_4x__neon(
    const uint8_t* restrict a,
    size_t m,
    size_t k,
    size_t stride,
    const int32_t multiplier,
    int32_t* restrict a_sum) {
  const uint8_t* a0 = a; // 初始化指向矩阵 a 的指针 a0
  const uint8_t* a1 = a0; // 初始化指向矩阵 a 的指针 a1
  if (m >= 2) {
    a1 += stride; // 如果 m 大于等于 2，将 a1 指针向后移动 stride 步长
  }
  const uint8_t* a2 = a1; // 初始化指向矩阵 a 的指针 a2
  if (m > 2) {
    a2 += stride; // 如果 m 大于 2，将 a2 指针向后移动 stride 步长
  }
  const uint8_t* a3 = a2; // 初始化指向矩阵 a 的指针 a3
  if (m == 4) {
    a3 += stride; // 如果 m 等于 4，将 a3 指针向后移动 stride 步长
  }

  uint32x4_t vacc0x0123 = vmovq_n_u32(0); // 初始化累加寄存器，用于存储 row 0 的累加结果
  uint32x4_t vacc1x0123 = vmovq_n_u32(0); // 初始化累加寄存器，用于存储 row 1 的累加结果
  uint32x4_t vacc2x0123 = vmovq_n_u32(0); // 初始化累加寄存器，用于存储 row 2 的累加结果
  uint32x4_t vacc3x0123 = vmovq_n_u32(0); // 初始化累加寄存器，用于存储 row 3 的累加结果
  for (; k >= 16; k -= 16) {
    // 逐行累加处理，每次处理 16 列
    // row 0
    const uint8x16_t va0x0_15 = vld1q_u8(a0); // 加载 a0 指针指向的 16 个 uint8_t 值到 va0x0_15 寄存器
    a0 += 16; // 将 a0 指针向后移动 16 个元素（16 字节）
    vacc0x0123 = vpadalq_u16(
        vacc0x0123, vaddl_u8(vget_low_u8(va0x0_15), vget_high_u8(va0x0_15))); // 将 va0x0_15 寄存器中的值累加到 vacc0x0123 寄存器中

    // row 1
    const uint8x16_t va1x0_15 = vld1q_u8(a1); // 加载 a1 指针指向的 16 个 uint8_t 值到 va1x0_15 寄存器
    a1 += 16; // 将 a1 指针向后移动 16 个元素（16 字节）
    vacc1x0123 = vpadalq_u16(
        vacc1x0123, vaddl_u8(vget_low_u8(va1x0_15), vget_high_u8(va1x0_15))); // 将 va1x0_15 寄存器中的值累加到 vacc1x0123 寄存器中

    // row 2
    const uint8x16_t va2x0_15 = vld1q_u8(a2); // 加载 a2 指针指向的 16 个 uint8_t 值到 va2x0_15 寄存器
    a2 += 16; // 将 a2 指针向后移动 16 个元素（16 字节）
    vacc2x0123 = vpadalq_u16(
        vacc2x0123, vaddl_u8(vget_low_u8(va2x0_15), vget_high_u8(va2x0_15))); // 将 va2x0_15 寄存器中的值累加到 vacc2x0123 寄存器中

    // row 3
    const uint8x16_t va3x0_15 = vld1q_u8(a3); // 加载 a3 指针指向的 16 个 uint8_t 值到 va3x0_15 寄存器
    a3 += 16; // 将 a3 指针向后移动 16 个元素（16 字节）
    vacc3x0123 = vpadalq_u16(
        vacc3x0123, vaddl_u8(vget_low_u8(va3x0_15), vget_high_u8(va3x0_15))); // 将 va3x0_15 寄存器中的值累加到 vacc3x0123 寄存器中
  }

  if (k >= 8) {
    // 处理剩余的列数大于等于 8 的情况
    vacc0x0123 = vaddw_u16(vacc0x0123, vpaddl_u8(vld1_u8(a0))); // 累加 a0 指针指向的 8 个 uint8_t 值到 vacc0x0123 寄存器中
    a0 += 8; // 将 a0 指针向后移动 8 个元素（8 字节）
    vacc1x0123 = vaddw_u16(vacc1x0123, vpaddl_u8(vld1_u8(a1))); // 累加 a1 指针指向的 8 个 uint8_t 值到 vacc1x0123 寄存器中
    a1 += 8; // 将 a1 指针向后移动 8 个元素（8 字节）
    vacc2x0123 = vaddw_u16(vacc2x0123, vpaddl_u8(vld1_u8(a2))); // 累加 a2 指针指向的 8 个 uint8_t 值到 vacc2x0123 寄存器中
    a2 += 8; // 将 a2 指针向后移动 8 个元素（8 字节）
    vacc3x0123 = vaddw_u16(vacc3x0123, vpaddl_u8(vld1_u8(a3))); // 累加 a3 指针指向的 8 个 uint8_t 值到 vacc3x0123 寄存器中
    a3 += 8; // 将 a3 指针向后移动 8 个元素（8 字节）
    k -= 8; // 更新剩余处理的列数
  }

  if (k >= 4) {
    // 处理剩余的列数大于等于 4 的情况
    vacc0x0123 = vaddw_u16(
        vacc0x0123,
        vget_low_u16(vmovl_u8(vreinterpret_u8_u32(
            vld1_dup_u32(__builtin_assume_aligned((const uint32_t*)a0, 1)))))); // 累加 a0 指针指向的 4 个 uint8_t 值到 vacc0x0123 寄存器中
    a0 += 4; // 将 a0 指针向后移动 4 个元素（4 字节）
    vacc1x0123 = vaddw_u16(
        vacc1x0123,
        vget_low_u16(vmovl_u8(vreinterpret_u8_u32(
            vld1_dup_u32(__builtin_assume_aligned((const uint32_t*)a1, 1)))))); // 累加 a1 指针指向的 4 个 uint8_t 值到 vacc1x0123
    // 减去4，用于调整计数器
    k -= 4;
  }

  // 计算四个向量的每对相邻元素之和，并合并成两个32位整数向量
  const uint32x2_t vsum0x01 =
      vpadd_u32(vget_low_u32(vacc0x0123), vget_high_u32(vacc0x0123));
  const uint32x2_t vsum1x01 =
      vpadd_u32(vget_low_u32(vacc1x0123), vget_high_u32(vacc1x0123));
  const uint32x2_t vsum2x01 =
      vpadd_u32(vget_low_u32(vacc2x0123), vget_high_u32(vacc2x0123));
  const uint32x2_t vsum3x01 =
      vpadd_u32(vget_low_u32(vacc3x0123), vget_high_u32(vacc3x0123));
  // 将四个32位整数向量的和合并为一个128位整数向量
  uint32x4_t vacc0123 = vcombine_u32(
      vpadd_u32(vsum0x01, vsum1x01), vpadd_u32(vsum2x01, vsum3x01));

  // 如果还剩余至少两个元素需要处理
  if (k >= 2) {
    // 从a0、a1、a2、a3加载16位数据并转换为8位数据，然后进行向量操作
    const uint8x8_t va0x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a0, 1)));
    a0 += 2;
    const uint8x8_t va1x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a1, 1)));
    a1 += 2;
    const uint8x8_t va2x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a2, 1)));
    a2 += 2;
    const uint8x8_t va3x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a3, 1)));
    a3 += 2;
    // 提取va0x01010101和va1x01010101的部分数据，并组合成新的8位数据向量
    const uint8x8_t va0x01_1x010101 = vext_u8(va0x01010101, va1x01010101, 2);
    // 提取va2x01010101和va3x01010101的部分数据，并组合成新的8位数据向量
    const uint8x8_t va2x01_3x010101 = vext_u8(va2x01010101, va3x01010101, 6);
    // 提取前面两个向量的部分数据并组合成新的8位数据向量
    const uint8x8_t va0123x01 = vext_u8(va0x01_1x010101, va2x01_3x010101, 4);
    // 将8位数据向量的每对相邻元素之和，加到128位整数向量上
    vacc0123 = vaddw_u16(vacc0123, vpaddl_u8(va0123x01));
    // 减去2，用于调整计数器
    k -= 2;
  }

  // 如果还有一个元素需要处理
  if (k > 0) {
    // 将a0、a1、a2、a3中的数据加载为8位数据向量，并添加到128位整数向量上
    uint8x8_t vax0x1x2x3 = vmov_n_u8(0);
    vax0x1x2x3 = vld1_lane_u8(a0, vax0x1x2x3, 0);
    vax0x1x2x3 = vld1_lane_u8(a1, vax0x1x2x3, 2);
    vax0x1x2x3 = vld1_lane_u8(a2, vax0x1x2x3, 4);
    vax0x1x2x3 = vld1_lane_u8(a3, vax0x1x2x3, 6);
    vacc0123 = vaddw_u16(vacc0123, vpaddl_u8(vax0x1x2x3));
  }

  // 将128位整数向量转换为32位有符号整数向量，并乘以倍数
  int32x4_t vsum0123 = vmulq_n_s32(vreinterpretq_s32_u32(vacc0123), multiplier);
  // 如果m为4，则将结果存储到a_sum中
  if (m == 4) {
    vst1q_s32(a_sum, vsum0123);
  } else {
    // 如果m至少为2，则存储前两个元素的结果到a_sum中，并调整指针和计数器
    if (m >= 2) {
      vst1_s32(a_sum, vget_low_s32(vsum0123));
      a_sum += 2;
      vsum0123 = vextq_s32(vsum0123, vsum0123, 2);
      m -= 2;
    }
    // 如果m不为0，则存储最后一个元素的结果到a_sum中
    if (m != 0) {
      vst1q_lane_s32(a_sum, vsum0123, 0);
    }
  }
}


注释：


# 这行代码表示一个代码块的结束，它与之前的代码结构有关联。
```