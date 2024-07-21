# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\8x8-neon.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8gemm.h>
#include <requantization/runtime-neon.h>

void pytorch_q8gemm_ukernel_8x8__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {
  // 加载权重矩阵的第一和第二列到寄存器中
  int32x4_t vacc0x0123 = vld1q_s32(w);
  w = (const void*)((uintptr_t)w + 16);
  int32x4_t vacc0x4567 = vld1q_s32(w);
  w = (const void*)((uintptr_t)w + 16);
  // 复制第一和第二列的值到其余寄存器中
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc1x4567 = vacc0x4567;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc2x4567 = vacc0x4567;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc3x4567 = vacc0x4567;
  int32x4_t vacc4x0123 = vacc0x0123;
  int32x4_t vacc4x4567 = vacc0x4567;
  int32x4_t vacc5x0123 = vacc0x0123;
  int32x4_t vacc5x4567 = vacc0x4567;
  int32x4_t vacc6x0123 = vacc0x0123;
  int32x4_t vacc6x4567 = vacc0x4567;
  int32x4_t vacc7x0123 = vacc0x0123;
  int32x4_t vacc7x4567 = vacc0x4567;

  // 初始化指针 a0 到 a7，根据 mr 的值选择适当的指针
  const uint8_t* a0 = a;
  const uint8_t* a1 = (const uint8_t*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const uint8_t* a2 = (const uint8_t*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const uint8_t* a3 = (const uint8_t*)((uintptr_t)a2 + a_stride);
  if (mr < 4) {
    a3 = a2;
  }
  const uint8_t* a4 = (const uint8_t*)((uintptr_t)a3 + a_stride);
  if (mr <= 4) {
    a4 = a3;
  }
  const uint8_t* a5 = (const uint8_t*)((uintptr_t)a4 + a_stride);
  if (mr < 6) {
    a5 = a4;
  }
  const uint8_t* a6 = (const uint8_t*)((uintptr_t)a5 + a_stride);
  if (mr <= 6) {
    a6 = a5;
  }
  const uint8_t* a7 = (const uint8_t*)((uintptr_t)a6 + a_stride);
  if (mr != 8) {
    a7 = a6;
  }

  // 加载输入和卷积核的零点到 NEON 寄存器中
  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);
  const uint8x8_t vb_zero_point =
      vld1_u8((const uint8_t*)&quantization_params->neon.kernel_zero_points
          [output_channel_index]);
  // 循环处理每组 8 个输入向量
  for (; k >= 8; k -= 8) {
    // 加载 8 个输入向量，并进行零点处理和类型转换
    const uint8x8_t va0 = vld1_u8(a0);
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
    a0 += 8;
    const uint8x8_t va1 = vld1_u8(a1);
    const int16x8_t vxa1 =
        vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
    a1 += 8;
    const uint8x8_t va2 = vld1_u8(a2);
    const int16x8_t vxa2 =
        vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
    a2 += 8;
    const uint8x8_t va3 = vld1_u8(a3);
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));
    a3 += 8;
    const uint8x8_t va4 = vld1_u8(a4);


这段代码使用了NEON指令集来进行低精度量化矩阵乘法的计算。
    const int16x8_t vxa4 =
        vreinterpretq_s16_u16(sub_zero_point(va4, va_zero_point));
    // 转换 va4 中的每个元素为 int16 类型，并减去零点偏移量，存入 vxa4

    a4 += 8;
    // 指针 a4 向后移动 8 个字节，指向下一个数据块

    const uint8x8_t va5 = vld1_u8(a5);
    // 从指针 a5 处加载 8 个 uint8 值到 va5 中

    const int16x8_t vxa5 =
        vreinterpretq_s16_u16(sub_zero_point(va5, va_zero_point));
    // 转换 va5 中的每个元素为 int16 类型，并减去零点偏移量，存入 vxa5

    a5 += 8;
    // 指针 a5 向后移动 8 个字节，指向下一个数据块

    const uint8x8_t va6 = vld1_u8(a6);
    // 从指针 a6 处加载 8 个 uint8 值到 va6 中

    const int16x8_t vxa6 =
        vreinterpretq_s16_u16(sub_zero_point(va6, va_zero_point));
    // 转换 va6 中的每个元素为 int16 类型，并减去零点偏移量，存入 vxa6

    a6 += 8;
    // 指针 a6 向后移动 8 个字节，指向下一个数据块

    const uint8x8_t va7 = vld1_u8(a7);
    // 从指针 a7 处加载 8 个 uint8 值到 va7 中

    const int16x8_t vxa7 =
        vreinterpretq_s16_u16(sub_zero_point(va7, va_zero_point));
    // 转换 va7 中的每个元素为 int16 类型，并减去零点偏移量，存入 vxa7

    a7 += 8;
    // 指针 a7 向后移动 8 个字节，指向下一个数据块

    const uint8x8_t vb01234567c0 = vld1_u8(w);
    // 从指针 w 处加载 8 个 uint8 值到 vb01234567c0 中

    w = (const void*)((uintptr_t)w + 8);
    // w 指针向后移动 8 个字节，指向下一个数据块

    const int16x8_t vxb01234567c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
    // 将 vb01234567c0 中的每个元素转换为 int16 类型，并减去零点偏移量，存入 vxb01234567c0

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    // 在 vacc0x0123 中的低 4 个 int32 数值中，用 vxb01234567c0 和 vxa0 的低 4 个 int16 数值进行乘加操作

    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    // 在 vacc0x4567 中的高 4 个 int32 数值中，用 vxb01234567c0 的高 4 个 int16 数值和 vxa0 的低 4 个 int16 数值进行乘加操作

    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    // 在 vacc1x0123 中的低 4 个 int32 数值中，用 vxb01234567c0 和 vxa1 的低 4 个 int16 数值进行乘加操作

    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    // 在 vacc1x4567 中的高 4 个 int32 数值中，用 vxb01234567c0 的高 4 个 int16 数值和 vxa1 的低 4 个 int16 数值进行乘加操作

    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    // 在 vacc2x0123 中的低 4 个 int32 数值中，用 vxb01234567c0 和 vxa2 的低 4 个 int16 数值进行乘加操作

    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    // 在 vacc2x4567 中的高 4 个 int32 数值中，用 vxb01234567c0 的高 4 个 int16 数值和 vxa2 的低 4 个 int16 数值进行乘加操作

    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    // 在 vacc3x0123 中的低 4 个 int32 数值中，用 vxb01234567c0 和 vxa3 的低 4 个 int16 数值进行乘加操作

    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    // 在 vacc3x4567 中的高 4 个 int32 数值中，用 vxb01234567c0 的高 4 个 int16 数值和 vxa3 的低 4 个 int16 数值进行乘加操作

    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
    // 在 vacc4x0123 中的低 4 个 int32 数值中，用 vxb01234567c0 和 vxa4 的低 4 个 int16 数值进行乘加操作

    vacc4x4567 = vmlal_lane_s16(
        vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
    // 在 vacc4x4567 中的高 4 个 int32 数值中，用 vxb01234567c0 的高 4 个 int16 数值和 vxa4 的低 4 个 int16 数值进行乘加操作

    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
    // 在 vacc5x0123 中的低 4 个 int32 数值中，用 vxb01234567c0 和 vxa5 的低 4 个 int16 数值进行乘加操作

    vacc5x4567 = vmlal_lane_s16(
        vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
    // 在 vacc5x4567 中的高 4 个 int32 数值中，用 vxb01234567c0 的高 4 个 int16 数值和 vxa5 的低 4 个 int16 数值进行乘加操作

    vacc6x0123 = vmlal_lane_s16(
        vacc6x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
    // 在 vacc6x0123 中的低 4 个 int32 数值中，用 vxb01234567c0 和 vxa6 的低 4 个 int16 数值进行乘加操作

    vacc6x4567 = vmlal_lane_s16
    # 使用 vmlal_lane_s16 函数将 vxa1 的低16位与 vxb01234567c1 的高16位相乘，并加到 vacc1x4567 上
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    # 使用 vmlal_lane_s16 函数将 vxa2 的低16位与 vxb01234567c1 的低16位相乘，并加到 vacc2x0123 上
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    # 使用 vmlal_lane_s16 函数将 vxa2 的低16位与 vxb01234567c1 的高16位相乘，并加到 vacc2x4567 上
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    # 使用 vmlal_lane_s16 函数将 vxa3 的低16位与 vxb01234567c1 的低16位相乘，并加到 vacc3x0123 上
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
    # 使用 vmlal_lane_s16 函数将 vxa3 的低16位与 vxb01234567c1 的高16位相乘，并加到 vacc3x4567 上
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
    # 使用 vmlal_lane_s16 函数将 vxa4 的低16位与 vxb01234567c1 的低16位相乘，并加到 vacc4x0123 上
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
    # 使用 vmlal_lane_s16 函数将 vxa4 的低16位与 vxb01234567c1 的高16位相乘，并加到 vacc4x4567 上
    vacc4x4567 = vmlal_lane_s16(
        vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
    # 使用 vmlal_lane_s16 函数将 vxa5 的低16位与 vxb01234567c1 的低16位相乘，并加到 vacc5x0123 上
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
    # 使用 vmlal_lane_s16 函数将 vxa5 的低16位与 vxb01234567c1 的高16位相乘，并加到 vacc5x4567 上
    vacc5x4567 = vmlal_lane_s16(
        vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
    # 使用 vmlal_lane_s16 函数将 vxa6 的低16位与 vxb01234567c1 的低16位相乘，并加到 vacc6x0123 上
    vacc6x0123 = vmlal_lane_s16(
        vacc6x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
    # 使用 vmlal_lane_s16 函数将 vxa6 的低16位与 vxb01234567c1 的高16位相乘，并加到 vacc6x4567 上
    vacc6x4567 = vmlal_lane_s16(
        vacc6x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
    # 使用 vmlal_lane_s16 函数将 vxa7 的低16位与 vxb01234567c1 的低16位相乘，并加到 vacc7x0123 上
    vacc7x0123 = vmlal_lane_s16(
        vacc7x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa7), 1);
    # 使用 vmlal_lane_s16 函数将 vxa7 的低16位与 vxb01234567c1 的高16位相乘，并加到 vacc7x4567 上
    vacc7x4567 = vmlal_lane_s16(
        vacc7x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa7), 1);

    # 使用 vld1_u8 函数加载指针 w 指向的内存中的8个字节到 vb01234567c2
    const uint8x8_t vb01234567c2 = vld1_u8(w);
    # 将指针 w 向后移动8个字节
    w = (const void*)((uintptr_t)w + 8);
    # 使用 vsubl_u8 函数将 vb01234567c2 与 vb_zero_point 逐元素相减，并转换成有符号16位整数类型
    const int16x8_t vxb01234567c2 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

    # 使用 vmlal_lane_s16 函数将 vxa0 的低16位与 vxb01234567c2 的低16位相乘，并加到 vacc0x0123 上
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    # 使用 vmlal_lane_s16 函数将 vxa0 的低16位与 vxb01234567c2 的高16位相乘，并加到 vacc0x4567 上
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    # 使用 vmlal_lane_s16 函数将 vxa1 的低16位与 vxb01234567c2 的低16位相乘，并加到 vacc1x0123 上
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    # 使用 vmlal_lane_s16 函数将 vxa1 的低16位与 vxb01234567c2 的高16位相乘，并加到 vacc1x4567 上
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    # 使用 vmlal_lane_s16 函数将 vxa2 的低16位与 vxb01234567c2 的低16位相乘，并加到 vacc2x0123 上
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    # 使用 vmlal_lane_s16 函数将 vxa2 的低16位与 vxb01234567c2 的高16位相乘，并加到 vacc2x4567 上
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    # 使用 vmlal_lane_s16 函数将 vxa3 的低16位与 vxb01234567c2 的低16位相乘，并加到 vacc3x0123 上
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
    # 使用 vmlal_lane_s16 函数将 vxa3 的低16位与 vxb01234567c2 的高16位相乘，并加到 vacc3x456
    // 使用 vmlal_lane_s16 函数将 vxb01234567c2 的高16位与低16位按照指定的 lane（2）对 vacc6x4567 进行 16 位乘加操作
    vacc6x4567 = vmlal_lane_s16(
        vacc6x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c2 的低16位与低16位按照指定的 lane（2）对 vacc7x0123 进行 16 位乘加操作
    vacc7x0123 = vmlal_lane_s16(
        vacc7x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa7), 2);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c2 的高16位与低16位按照指定的 lane（2）对 vacc7x4567 进行 16 位乘加操作
    vacc7x4567 = vmlal_lane_s16(
        vacc7x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa7), 2);

    // 加载并解析 w 指向的内存作为 uint8x8_t 类型的数据
    const uint8x8_t vb01234567c3 = vld1_u8(w);
    // 将 w 的指针位置后移 8 字节
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb01234567c3 向量减去 vb_zero_point 向量，并将结果转换为 int16x8_t 类型的向量
    const int16x8_t vxb01234567c3 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的低16位与低16位按照指定的 lane（3）对 vacc0x0123 进行 16 位乘加操作
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的高16位与低16位按照指定的 lane（3）对 vacc0x4567 进行 16 位乘加操作
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的低16位与低16位按照指定的 lane（3）对 vacc1x0123 进行 16 位乘加操作
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的高16位与低16位按照指定的 lane（3）对 vacc1x4567 进行 16 位乘加操作
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的低16位与低16位按照指定的 lane（3）对 vacc2x0123 进行 16 位乘加操作
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的高16位与低16位按照指定的 lane（3）对 vacc2x4567 进行 16 位乘加操作
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的低16位与低16位按照指定的 lane（3）对 vacc3x0123 进行 16 位乘加操作
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的高16位与低16位按照指定的 lane（3）对 vacc3x4567 进行 16 位乘加操作
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的低16位与低16位按照指定的 lane（3）对 vacc4x0123 进行 16 位乘加操作
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的高16位与低16位按照指定的 lane（3）对 vacc4x4567 进行 16 位乘加操作
    vacc4x4567 = vmlal_lane_s16(
        vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的低16位与低16位按照指定的 lane（3）对 vacc5x0123 进行 16 位乘加操作
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的高16位与低16位按照指定的 lane（3）对 vacc5x4567 进行 16 位乘加操作
    vacc5x4567 = vmlal_lane_s16(
        vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的低16位与低16位按照指定的 lane（3）对 vacc6x0123 进行 16 位乘加操作
    vacc6x0123 = vmlal_lane_s16(
        vacc6x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的高16位与低16位按照指定的 lane（3）对 vacc6x4567 进行 16 位乘加操作
    vacc6x4567 = vmlal_lane_s16(
        vacc6x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的低16位与低16位按照指定的 lane（3）对 vacc7x0123 进行 16 位乘加操作
    vacc7x0123 = vmlal_lane_s16(
        vacc7x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa7), 3);
    // 使用 vmlal_lane_s16 函数将 vxb01234567c3 的高16位与低16位按照指定的 lane（3）对 vacc
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c4) 与 vget_high_s16(vxa2) 的有符号 16 位整数乘法累加到 vacc2x4567 中的对应位置
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_low_s16(vxb01234567c4) 与 vget_high_s16(vxa3) 的有符号 16 位整数乘法累加到 vacc3x0123 中的对应位置
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c4) 与 vget_high_s16(vxa3) 的有符号 16 位整数乘法累加到 vacc3x4567 中的对应位置
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_low_s16(vxb01234567c4) 与 vget_high_s16(vxa4) 的有符号 16 位整数乘法累加到 vacc4x0123 中的对应位置
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c4) 与 vget_high_s16(vxa4) 的有符号 16 位整数乘法累加到 vacc4x4567 中的对应位置
    vacc4x4567 = vmlal_lane_s16(
        vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_low_s16(vxb01234567c4) 与 vget_high_s16(vxa5) 的有符号 16 位整数乘法累加到 vacc5x0123 中的对应位置
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c4) 与 vget_high_s16(vxa5) 的有符号 16 位整数乘法累加到 vacc5x4567 中的对应位置
    vacc5x4567 = vmlal_lane_s16(
        vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_low_s16(vxb01234567c4) 与 vget_high_s16(vxa6) 的有符号 16 位整数乘法累加到 vacc6x0123 中的对应位置
    vacc6x0123 = vmlal_lane_s16(
        vacc6x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c4) 与 vget_high_s16(vxa6) 的有符号 16 位整数乘法累加到 vacc6x4567 中的对应位置
    vacc6x4567 = vmlal_lane_s16(
        vacc6x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_low_s16(vxb01234567c4) 与 vget_high_s16(vxa7) 的有符号 16 位整数乘法累加到 vacc7x0123 中的对应位置
    vacc7x0123 = vmlal_lane_s16(
        vacc7x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa7), 0);
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c4) 与 vget_high_s16(vxa7) 的有符号 16 位整数乘法累加到 vacc7x4567 中的对应位置
    vacc7x4567 = vmlal_lane_s16(
        vacc7x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa7), 0);

    # 加载指针 w 所指向的 8 个 uint8_t 值到向量 vb01234567c5
    const uint8x8_t vb01234567c5 = vld1_u8(w);
    # 将 w 指针向后移动 8 字节
    w = (const void*)((uintptr_t)w + 8);
    # 将 vb01234567c5 向量中的 uint8_t 值转换为有符号 16 位整数，然后减去 vb_zero_point 向量，得到 vxb01234567c5 向量
    const int16x8_t vxb01234567c5 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

    # 使用 vmlal_lane_s16 函数，将 vget_low_s16(vxb01234567c5) 与 vget_high_s16(vxa0) 的有符号 16 位整数乘法累加到 vacc0x0123 中的对应位置
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c5) 与 vget_high_s16(vxa0) 的有符号 16 位整数乘法累加到 vacc0x4567 中的对应位置
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    # 使用 vmlal_lane_s16 函数，将 vget_low_s16(vxb01234567c5) 与 vget_high_s16(vxa1) 的有符号 16 位整数乘法累加到 vacc1x0123 中的对应位置
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c5) 与 vget_high_s16(vxa1) 的有符号 16 位整数乘法累加到 vacc1x4567 中的对应位置
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    # 使用 vmlal_lane_s16 函数，将 vget_low_s16(vxb01234567c5) 与 vget_high_s16(vxa2) 的有符号 16 位整数乘法累加到 vacc2x0123 中的对应位置
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
    # 使用 vmlal_lane_s16 函数，将 vget_high_s16(vxb01234567c5) 与 vget_high_s16(vxa2) 的有符号 16 位整数乘法累加到 vacc2x4567 中的对应位置
    vacc2x4567 = vmlal_lane_s16(
        vacc2
    // 使用 vget_high_s16(vxb01234567c5) 的高位元素和 vxa7 的高位元素进行 S16 向左乘积累加运算，结果累加到 vacc7x4567 中的对应位置
    vacc7x4567 = vmlal_lane_s16(
        vacc7x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa7), 1);

    // 从地址 w 处加载 8 个 uint8_t 类型数据到 vb01234567c6，然后将 w 向后移动 8 个字节
    const uint8x8_t vb01234567c6 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb01234567c6 中的每个元素减去 vb_zero_point，并将结果转换为 int16x8_t 类型存储到 vxb01234567c6
    const int16x8_t vxb01234567c6 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

    // 使用 vget_low_s16(vxb01234567c6) 和 vget_high_s16(vxa0) 的高位元素进行 S16 向左乘积累加运算，结果累加到 vacc0x0123 中的对应位置
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    // 使用 vget_high_s16(vxb01234567c6) 和 vget_high_s16(vxa0) 的高位元素进行 S16 向左乘积累加运算，结果累加到 vacc0x4567 中的对应位置
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    // 重复上述过程，针对 vacc1x0123、vacc1x4567、vacc2x0123、vacc2x4567、vacc3x0123、vacc3x4567、vacc4x0123、vacc4x4567、vacc5x0123、vacc5x4567、vacc6x0123、vacc6x4567、vacc7x0123、vacc7x4567
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
    vacc4x4567 = vmlal_lane_s16(
        vacc4x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
    vacc5x4567 = vmlal_lane_s16(
        vacc5x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
    vacc6x0123 = vmlal_lane_s16(
        vacc6x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
    vacc6x4567 = vmlal_lane_s16(
        vacc6x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
    vacc7x0123 = vmlal_lane_s16(
        vacc7x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa7), 2);
    vacc7x4567 = vmlal_lane_s16(
        vacc7x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa7), 2);

    // 重复上述过程，使用 vget_low_s16(vxb01234567c7) 和 vget_high_s16(vxa3) 的高位元素进行 S16 向左乘积累加运算，结果累加到相应的 vaccx0123 和 vaccx4567 中
    const uint8x8_t vb01234567c7 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c7 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
    // 使用 vmlal_lane_s16 函数，对 vacc3x4567 向量执行带有 vxb01234567c7 向量的高位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);

    // 使用 vmlal_lane_s16 函数，对 vacc4x0123 向量执行带有 vxb01234567c7 向量的低位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa4), 3);

    // 使用 vmlal_lane_s16 函数，对 vacc4x4567 向量执行带有 vxb01234567c7 向量的高位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc4x4567 = vmlal_lane_s16(
        vacc4x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa4), 3);

    // 使用 vmlal_lane_s16 函数，对 vacc5x0123 向量执行带有 vxb01234567c7 向量的低位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa5), 3);

    // 使用 vmlal_lane_s16 函数，对 vacc5x4567 向量执行带有 vxb01234567c7 向量的高位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc5x4567 = vmlal_lane_s16(
        vacc5x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa5), 3);

    // 使用 vmlal_lane_s16 函数，对 vacc6x0123 向量执行带有 vxb01234567c7 向量的低位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc6x0123 = vmlal_lane_s16(
        vacc6x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa6), 3);

    // 使用 vmlal_lane_s16 函数，对 vacc6x4567 向量执行带有 vxb01234567c7 向量的高位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc6x4567 = vmlal_lane_s16(
        vacc6x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa6), 3);

    // 使用 vmlal_lane_s16 函数，对 vacc7x0123 向量执行带有 vxb01234567c7 向量的低位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc7x0123 = vmlal_lane_s16(
        vacc7x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa7), 3);

    // 使用 vmlal_lane_s16 函数，对 vacc7x4567 向量执行带有 vxb01234567c7 向量的高位元素的 S16 类型的带符号乘法累加操作，使用第 3 个 lane。
    vacc7x4567 = vmlal_lane_s16(
        vacc7x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa7), 3);
  }
  if (k != 0) {
    // 计算 a_predecrement，表示未处理的元素数量。
    const size_t a_predecrement = 8 - k;
    // 创建 va_shift 向量，包含 -8 * a_predecrement 的值。
    const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
    // 加载并处理 a0 到 a7 的数据，转换为 vxa0 到 vxa7 向量。
    const uint8x8_t va0 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
    const uint8x8_t va1 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
    const int16x8_t vxa1 =
        vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
    const uint8x8_t va2 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
    const int16x8_t vxa2 =
        vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
    const uint8x8_t va3 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));
    const uint8x8_t va4 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a4 - a_predecrement)), va_shift));
    const int16x8_t vxa4 =
        vreinterpretq_s16_u16(sub_zero_point(va4, va_zero_point));
    const uint8x8_t va5 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a5 - a_predecrement)), va_shift));
    const int16x8_t vxa5 =
        vreinterpretq_s16_u16(sub_zero_point(va5, va_zero_point));
    const uint8x8_t va6 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a6 - a_predecrement)), va_shift));
    const int16x8_t vxa6 =
        vreinterpretq_s16_u16(sub_zero_point(va6, va_zero_point));
    const uint8x8_t va7 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a7 - a_predecrement)), va_shift));
    const int16x8_t vxa7 =
        vreinterpretq_s16_u16(sub_zero_point(va7, va_zero_point));

    // 加载 w 所指向的数据到 vb01234567c0 向量。
    const uint8x8_t vb01234567c0 = vld1_u8(w);
    // 更新 w 的指针，指向下一个位置。
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb01234567c0 中的每个元素减去 vb_zero_point，并转换为有符号16位整数类型
    const int16x8_t vxb01234567c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    // 使用 vxa0 的低位16位元素与 vxb01234567c0 的低位16位元素进行有符号16位整数乘累加运算，并存储到 vacc0x0123
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    // 使用 vxa0 的低位16位元素与 vxb01234567c0 的高位16位元素进行有符号16位整数乘累加运算，并存储到 vacc0x4567
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    // 使用 vxa1 的低位16位元素与 vxb01234567c0 的低位16位元素进行有符号16位整数乘累加运算，并存储到 vacc1x0123
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    // 使用 vxa1 的低位16位元素与 vxb01234567c0 的高位16位元素进行有符号16位整数乘累加运算，并存储到 vacc1x4567
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    // 使用 vxa2 的低位16位元素与 vxb01234567c0 的低位16位元素进行有符号16位整数乘累加运算，并存储到 vacc2x0123
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    // 使用 vxa2 的低位16位元素与 vxb01234567c0 的高位16位元素进行有符号16位整数乘累加运算，并存储到 vacc2x4567
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    // 使用 vxa3 的低位16位元素与 vxb01234567c0 的低位16位元素进行有符号16位整数乘累加运算，并存储到 vacc3x0123
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    // 使用 vxa3 的低位16位元素与 vxb01234567c0 的高位16位元素进行有符号16位整数乘累加运算，并存储到 vacc3x4567
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    // 使用 vxa4 的低位16位元素与 vxb01234567c0 的低位16位元素进行有符号16位整数乘累加运算，并存储到 vacc4x0123
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
    // 使用 vxa4 的低位16位元素与 vxb01234567c0 的高位16位元素进行有符号16位整数乘累加运算，并存储到 vacc4x4567
    vacc4x4567 = vmlal_lane_s16(
        vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
    // 使用 vxa5 的低位16位元素与 vxb01234567c0 的低位16位元素进行有符号16位整数乘累加运算，并存储到 vacc5x0123
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
    // 使用 vxa5 的低位16位元素与 vxb01234567c0 的高位16位元素进行有符号16位整数乘累加运算，并存储到 vacc5x4567
    vacc5x4567 = vmlal_lane_s16(
        vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
    // 使用 vxa6 的低位16位元素与 vxb01234567c0 的低位16位元素进行有符号16位整数乘累加运算，并存储到 vacc6x0123
    vacc6x0123 = vmlal_lane_s16(
        vacc6x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
    // 使用 vxa6 的低位16位元素与 vxb01234567c0 的高位16位元素进行有符号16位整数乘累加运算，并存储到 vacc6x4567
    vacc6x4567 = vmlal_lane_s16(
        vacc6x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
    // 使用 vxa7 的低位16位元素与 vxb01234567c0 的低位16位元素进行有符号16位整数乘累加运算，并存储到 vacc7x0123
    vacc7x0123 = vmlal_lane_s16(
        vacc7x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa7), 0);
    // 使用 vxa7 的低位16位元素与 vxb01234567c0 的高位16位元素进行有符号16位整数乘累加运算，并存储到 vacc7x4567
    vacc7x4567 = vmlal_lane_s16(
        vacc7x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa7), 0);

    }

  }

  // 加载输出通道索引处的四个浮点数到 requantization_scale_c0123
  const float32x4_t requantization_scale_c0123 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[output_channel_index]
          );
  // 加载输出通道索引加4处的四个浮点数到 requantization_scale_c4567
  const float32x4_t requantization_scale_c4567 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[
              output_channel_index + 4]);

  // 将 vacc0x0123 转换为浮点数并与 requantization_scale_c0123 执行乘法，存储到 vacc0x0123_f
  const float32x4_t vacc0x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x0123), requantization_scale_c0123);
  // 将 vacc1x0123 转换为浮点数并与 requantization_scale_c0123 执行乘法，存储到 vacc1x0123_f
  const float32x4_t vacc1x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x0123), requantization_scale_c0123);
  // 将 vacc2x0123 转换为浮点数并与 requantization_scale_c0123
    # 计算乘法：将向量acc3x4567中的整数转换为浮点数后，与requantization_scale_c4567向量逐元素相乘
    const float32x4_t vacc3x4567_f =
      vmulq_f32(vcvtq_f32_s32(vacc3x4567), requantization_scale_c4567);
    
    # 计算乘法：将向量acc4x0123中的整数转换为浮点数后，与requantization_scale_c0123向量逐元素相乘
    const float32x4_t vacc4x0123_f =
      vmulq_f32(vcvtq_f32_s32(vacc4x0123), requantization_scale_c0123);
    
    # 计算乘法：将向量acc5x0123中的整数转换为浮点数后，与requantization_scale_c0123向量逐元素相乘
    const float32x4_t vacc5x0123_f =
      vmulq_f32(vcvtq_f32_s32(vacc5x0123), requantization_scale_c0123);
    
    # 计算乘法：将向量acc6x0123中的整数转换为浮点数后，与requantization_scale_c0123向量逐元素相乘
    const float32x4_t vacc6x0123_f =
      vmulq_f32(vcvtq_f32_s32(vacc6x0123), requantization_scale_c0123);
    
    # 计算乘法：将向量acc7x0123中的整数转换为浮点数后，与requantization_scale_c0123向量逐元素相乘
    const float32x4_t vacc7x0123_f =
      vmulq_f32(vcvtq_f32_s32(vacc7x0123), requantization_scale_c0123);
    
    # 计算乘法：将向量acc4x4567中的整数转换为浮点数后，与requantization_scale_c4567向量逐元素相乘
    const float32x4_t vacc4x4567_f =
      vmulq_f32(vcvtq_f32_s32(vacc4x4567), requantization_scale_c4567);
    
    # 计算乘法：将向量acc5x4567中的整数转换为浮点数后，与requantization_scale_c4567向量逐元素相乘
    const float32x4_t vacc5x4567_f =
      vmulq_f32(vcvtq_f32_s32(vacc5x4567), requantization_scale_c4567);
    
    # 计算乘法：将向量acc6x4567中的整数转换为浮点数后，与requantization_scale_c4567向量逐元素相乘
    const float32x4_t vacc6x4567_f =
      vmulq_f32(vcvtq_f32_s32(vacc6x4567), requantization_scale_c4567);
    
    # 计算乘法：将向量acc7x4567中的整数转换为浮点数后，与requantization_scale_c4567向量逐元素相乘
    const float32x4_t vacc7x4567_f =
      vmulq_f32(vcvtq_f32_s32(vacc7x4567), requantization_scale_c4567);
#ifdef

  // 保存初始指针 c 的位置，用于后续计算
  uint8_t* c0 = c;
  // 计算下一个指针位置 c1
  uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
  // 如果 mr 小于 2，则 c1 与 c0 相同
  if (mr < 2) {
    c1 = c0;
  }
  // 计算下一个指针位置 c2
  uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
  // 如果 mr 小于等于 2，则 c2 与 c1 相同
  if (mr <= 2) {
    c2 = c1;
  }
  // 计算下一个指针位置 c3
  uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
  // 如果 mr 小于 4，则 c3 与 c2 相同
  if (mr < 4) {
    c3 = c2;
  }
  // 计算下一个指针位置 c4
  uint8_t* c4 = (uint8_t*)((uintptr_t)c3 + c_stride);
  // 如果 mr 小于等于 4，则 c4 与 c3 相同
  if (mr <= 4) {
    c4 = c3;
  }
  // 计算下一个指针位置 c5
  uint8_t* c5 = (uint8_t*)((uintptr_t)c4 + c_stride);
  // 如果 mr 小于 6，则 c5 与 c4 相同
  if (mr < 6) {
    c5 = c4;
  }
  // 计算下一个指针位置 c6
  uint8_t* c6 = (uint8_t*)((uintptr_t)c5 + c_stride);
  // 如果 mr 小于等于 6，则 c6 与 c5 相同
  if (mr <= 6) {
    c6 = c5;
  }
  // 计算下一个指针位置 c7
  uint8_t* c7 = (uint8_t*)((uintptr_t)c6 + c_stride);
  // 如果 mr 不等于 8，则 c7 与 c6 相同
  if (mr != 8) {
    c7 = c6;
  }
  // 如果 nr 等于 8，则按照向量顺序将数据存储到 c0 到 c7
  if (nr == 8) {
    vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
    vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
    vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
    vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
    vst1_u8(c4, vget_low_u8(vout4x01234567_5x01234567));
    vst1_u8(c5, vget_high_u8(vout4x01234567_5x01234567));
    vst1_u8(c6, vget_low_u8(vout6x01234567_7x01234567));
    vst1_u8(c7, vget_high_u8(vout6x01234567_7x01234567));
  } else {
    // 否则，如果 nr 大于等于 4，则按照向量顺序将数据存储到 c0 到 c7 中
    if (nr >= 4) {
      vst1q_lane_u32(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          0);
      c0 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          2);
      c1 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          0);
      c2 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          2);
      c3 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c4, 1),
          vreinterpretq_u32_u8(vout4x01234567_5x01234567),
          0);
      c4 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c5, 1),
          vreinterpretq_u32_u8(vout4x01234567_5x01234567),
          2);
      c5 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c6, 1),
          vreinterpretq_u32_u8(vout6x01234567_7x01234567),
          0);
      c6 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c7, 1),
          vreinterpretq_u32_u8(vout6x01234567_7x01234567),
          2);
      c7 += 4;
      // 调整向量顺序
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      vout4x01234567_5x01234567 =
          vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
      vout6x01234567_7x01234567 =
          vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 4);
      nr -= 4;
    }
    # 如果 nr 大于等于 2，则执行以下操作
    if (nr >= 2) {
      # 将 vout0x01234567_1x01234567 向量的低位字节转换为 uint16_t 类型后，存储到 c0 的内存地址中
      vst1q_lane_u16(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          0);
      # c0 指针向后移动两个 uint16_t 的位置
      c0 += 2;
      # 将 vout0x01234567_1x01234567 向量的高位字节转换为 uint16_t 类型后，存储到 c1 的内存地址中
      vst1q_lane_u16(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          4);
      # c1 指针向后移动两个 uint16_t 的位置
      c1 += 2;
      # 将 vout2x01234567_3x01234567 向量的低位字节转换为 uint16_t 类型后，存储到 c2 的内存地址中
      vst1q_lane_u16(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          0);
      # c2 指针向后移动两个 uint16_t 的位置
      c2 += 2;
      # 将 vout2x01234567_3x01234567 向量的高位字节转换为 uint16_t 类型后，存储到 c3 的内存地址中
      vst1q_lane_u16(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          4);
      # c3 指针向后移动两个 uint16_t 的位置
      c3 += 2;
      # 将 vout4x01234567_5x01234567 向量的低位字节转换为 uint16_t 类型后，存储到 c4 的内存地址中
      vst1q_lane_u16(
          __builtin_assume_aligned(c4, 1),
          vreinterpretq_u16_u8(vout4x01234567_5x01234567),
          0);
      # c4 指针向后移动两个 uint16_t 的位置
      c4 += 2;
      # 将 vout4x01234567_5x01234567 向量的高位字节转换为 uint16_t 类型后，存储到 c5 的内存地址中
      vst1q_lane_u16(
          __builtin_assume_aligned(c5, 1),
          vreinterpretq_u16_u8(vout4x01234567_5x01234567),
          4);
      # c5 指针向后移动两个 uint16_t 的位置
      c5 += 2;
      # 将 vout6x01234567_7x01234567 向量的低位字节转换为 uint16_t 类型后，存储到 c6 的内存地址中
      vst1q_lane_u16(
          __builtin_assume_aligned(c6, 1),
          vreinterpretq_u16_u8(vout6x01234567_7x01234567),
          0);
      # c6 指针向后移动两个 uint16_t 的位置
      c6 += 2;
      # 将 vout6x01234567_7x01234567 向量的高位字节转换为 uint16_t 类型后，存储到 c7 的内存地址中
      vst1q_lane_u16(
          __builtin_assume_aligned(c7, 1),
          vreinterpretq_u16_u8(vout6x01234567_7x01234567),
          4);
      # c7 指针向后移动两个 uint16_t 的位置
      c7 += 2;
      # 将 vout0x01234567_1x01234567 向量的内容循环左移两个字节
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      # 将 vout2x01234567_3x01234567 向量的内容循环左移两个字节
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      # 将 vout4x01234567_5x01234567 向量的内容循环左移两个字节
      vout4x01234567_5x01234567 =
          vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
      # 将 vout6x01234567_7x01234567 向量的内容循环左移两个字节
      vout6x01234567_7x01234567 =
          vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 2);
      # 减去处理的向量数
      nr -= 2;
    }
    # 如果 nr 不等于 0，则执行以下操作
    if (nr != 0) {
      # 将 vout0x01234567_1x01234567 向量的低位字节存储到 c0 的内存地址中
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      # 将 vout0x01234567_1x01234567 向量的高位字节存储到 c1 的内存地址中
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      # 将 vout2x01234567_3x01234567 向量的低位字节存储到 c2 的内存地址中
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      # 将 vout2x01234567_3x01234567 向量的高位字节存储到 c3 的内存地址中
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
      # 将 vout4x01234567_5x01234567 向量的低位字节存储到 c4 的内存地址中
      vst1q_lane_u8(c4, vout4x01234567_5x01234567, 0);
      # 将 vout4x01234567_5x01234567 向量的高位字节存储到 c5 的内存地址中
      vst1q_lane_u8(c5, vout4x01234567_5x01234567, 8);
      # 将 vout6x01234567_7x01234567 向量的低位字节存储到 c6 的内存地址中
      vst1q_lane_u8(c6, vout6x01234567_7x01234567, 0);
      # 将 vout6x01234567_7x01234567 向量的高位字节存储到 c7 的内存地址中
      vst1q_lane_u8(c7, vout6x01234567_7x01234567, 8);
    }
}



# 这行代码表示一个代码块的结束，匹配前面的一个开放的代码块或函数定义。
```