# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\4x8-dq-neon.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>  // 包含 ARM NEON 指令集的头文件

#include <qnnpack/q8gemm.h>  // 包含 QNNPACK Q8 GEMM 的头文件
#include <requantization/runtime-neon.h>  // 包含 NEON 运行时的重新量化头文件

void pytorch_q8gemm_dq_ukernel_4x8__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    const float* restrict b,
    float* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const struct pytorch_qnnp_conv_dynamic_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  int32x4_t vacc0x0123 = {};  // 初始化累加器向量，用于存储计算结果
  int32x4_t vacc0x4567 = {};  // 初始化累加器向量，用于存储计算结果
  int32x4_t vacc1x0123 = {};  // 初始化累加器向量，用于存储计算结果
  int32x4_t vacc1x4567 = {};  // 初始化累加器向量，用于存储计算结果
  int32x4_t vacc2x0123 = {};  // 初始化累加器向量，用于存储计算结果
  int32x4_t vacc2x4567 = {};  // 初始化累加器向量，用于存储计算结果
  int32x4_t vacc3x0123 = {};  // 初始化累加器向量，用于存储计算结果
  int32x4_t vacc3x4567 = {};  // 初始化累加器向量，用于存储计算结果
  w = (const void*)((uintptr_t)w + 32);  // 更新权重指针，假设每次偏移 32 字节

  const uint8_t* a0 = a;  // 初始化输入矩阵指针 a0
  const uint8_t* a1 = (const uint8_t*)((uintptr_t)a0 + a_stride);  // 初始化输入矩阵指针 a1
  if (mr < 2) {
    a1 = a0;  // 如果 mr 小于 2，a1 指向 a0，保证操作在最小 2 行范围内
  }
  const uint8_t* a2 = (const uint8_t*)((uintptr_t)a1 + a_stride);  // 初始化输入矩阵指针 a2
  if (mr <= 2) {
    a2 = a1;  // 如果 mr 小于等于 2，a2 指向 a1，保证操作在最小 3 行范围内
  }
  const uint8_t* a3 = (const uint8_t*)((uintptr_t)a2 + a_stride);  // 初始化输入矩阵指针 a3
  if (mr != 4) {
    a3 = a2;  // 如果 mr 不等于 4，a3 指向 a2，保证操作在最小 4 行范围内
  }

  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->input_zero_point);
  // 加载输入的零点偏移量到向量 va_zero_point

  // 假设 kernel_zero_points 是一个填充足够元素使其成为 8 的倍数的数组
  const uint8x8_t vb_zero_point =
      vld1_u8((const uint8_t*)&quantization_params->kernel_zero_points
          [output_channel_index]);
  // 加载输出通道对应的卷积核零点偏移量到向量 vb_zero_point

  const float32x4_t vmultiplier_c0123 =
      vld1q_f32(&quantization_params->multipliers[output_channel_index]);
  // 加载输出通道对应的乘法器到向量 vmultiplier_c0123
  const float32x4_t vmultiplier_c4567 =
      vld1q_f32(&quantization_params->multipliers[output_channel_index + 4]);
  // 加载输出通道对应的乘法器到向量 vmultiplier_c4567
  const float32x4_t vbias[] = {
    vld1q_f32(b),
    vld1q_f32(b + 4),
  };
  // 加载偏置向量 b 到 vbias 数组

  for (; k >= 8; k -= 8) {  // 迭代处理每 8 列的数据
    const uint8x8_t va0 = vld1_u8(a0);  // 加载输入矩阵第 0 行数据到向量 va0
    a0 += 8;  // 移动输入矩阵指针到下一列的起始位置
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
    // 计算输入矩阵第 0 行数据与 va_zero_point 的差值，存储到 vxa0 向量
    const uint8x8_t va1 = vld1_u8(a1);  // 加载输入矩阵第 1 行数据到向量 va1
    a1 += 8;  // 移动输入矩阵指针到下一列的起始位置
    const int16x8_t vxa1 =
        vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
    // 计算输入矩阵第 1 行数据与 va_zero_point 的差值，存储到 vxa1 向量
    const uint8x8_t va2 = vld1_u8(a2);  // 加载输入矩阵第 2 行数据到向量 va2
    a2 += 8;  // 移动输入矩阵指针到下一列的起始位置
    const int16x8_t vxa2 =
        vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
    // 计算输入矩阵第 2 行数据与 va_zero_point 的差值，存储到 vxa2 向量
    const uint8x8_t va3 = vld1_u8(a3);  // 加载输入矩阵第 3 行数据到向量 va3
    a3 += 8;  // 移动输入矩阵指针到下一列的起始位置
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));
    // 计算输入矩阵第 3 行数据与 va_zero_point 的差值，存储到 vxa3 向量

    const uint8x8_t vb01234567c0 = vld1_u8(w);  // 加载权重数据到向量 vb01234567c0
    w = (const void*)((uintptr_t)w + 8);  // 更新权重指针到下一列的起始位置
    const int16x8_t vxb01234567c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
    // 计算权重数据与 vb_zero_point 的差值，存储到 vxb01234567c0 向量

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    // 使用 vxb01234567c0 的低位和 vxa0 的低位元素进行乘法累加到 vacc0x0123
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    // 使用 vxb01234567c0 的高位和 vxa0 的低位元素进行乘法累加到 vacc0x4567
    # 使用 vmlal_lane_s16 函数将 vxb01234567c0 的低位元素与 vxa1 的低位元素进行带符号 16 位乘积累加到 vacc1x0123 中的第 0 位置
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c0 的高位元素与 vxa1 的低位元素进行带符号 16 位乘积累加到 vacc1x4567 中的第 0 位置
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c0 的低位元素与 vxa2 的低位元素进行带符号 16 位乘积累加到 vacc2x0123 中的第 0 位置
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c0 的高位元素与 vxa2 的低位元素进行带符号 16 位乘积累加到 vacc2x4567 中的第 0 位置
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c0 的低位元素与 vxa3 的低位元素进行带符号 16 位乘积累加到 vacc3x0123 中的第 0 位置
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c0 的高位元素与 vxa3 的低位元素进行带符号 16 位乘积累加到 vacc3x4567 中的第 0 位置
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);

    # 从地址 w 加载 8 个字节数据到 vb01234567c1
    const uint8x8_t vb01234567c1 = vld1_u8(w);
    # 将地址 w 增加 8 个字节
    w = (const void*)((uintptr_t)w + 8);
    # 将 vb01234567c1 中的元素转换为带符号 16 位整数并减去 vb_zero_point，得到 vxb01234567c1
    const int16x8_t vxb01234567c1 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

    # 使用 vmlal_lane_s16 函数将 vxb01234567c1 的低位元素与 vxa0 的低位元素进行带符号 16 位乘积累加到 vacc0x0123 中的第 1 位置
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c1 的高位元素与 vxa0 的低位元素进行带符号 16 位乘积累加到 vacc0x4567 中的第 1 位置
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c1 的低位元素与 vxa1 的低位元素进行带符号 16 位乘积累加到 vacc1x0123 中的第 1 位置
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c1 的高位元素与 vxa1 的低位元素进行带符号 16 位乘积累加到 vacc1x4567 中的第 1 位置
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c1 的低位元素与 vxa2 的低位元素进行带符号 16 位乘积累加到 vacc2x0123 中的第 1 位置
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c1 的高位元素与 vxa2 的低位元素进行带符号 16 位乘积累加到 vacc2x4567 中的第 1 位置
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c1 的低位元素与 vxa3 的低位元素进行带符号 16 位乘积累加到 vacc3x0123 中的第 1 位置
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c1 的高位元素与 vxa3 的低位元素进行带符号 16 位乘积累加到 vacc3x4567 中的第 1 位置
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);

    # 从地址 w 加载 8 个字节数据到 vb01234567c2
    const uint8x8_t vb01234567c2 = vld1_u8(w);
    # 将地址 w 增加 8 个字节
    w = (const void*)((uintptr_t)w + 8);
    # 将 vb01234567c2 中的元素转换为带符号 16 位整数并减去 vb_zero_point，得到 vxb01234567c2
    const int16x8_t vxb01234567c2 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

    # 使用 vmlal_lane_s16 函数将 vxb01234567c2 的低位元素与 vxa0 的低位元素进行带符号 16 位乘积累加到 vacc0x0123 中的第 2 位置
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    # 使用 vmlal_lane_s16 函数将 vxb01234567c2 的高位元素与 vxa0 的低位元素进行带符号 16 位乘积累加到 vacc0x4567 中的第 2 位置
    // 将 vxa0 的低 16 位数据提取出来，并将其乘以 vxb01234567c3 的第三个 lane 的 16 位数据，结果累加到 vacc0x0123 中
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    // 将 vxa0 的高 16 位数据提取出来，并将其乘以 vxb01234567c3 的第三个 lane 的 16 位数据，结果累加到 vacc0x4567 中
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    // 将 vxa1 的低 16 位数据提取出来，并将其乘以 vxb01234567c3 的第三个 lane 的 16 位数据，结果累加到 vacc1x0123 中
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    // 将 vxa1 的高 16 位数据提取出来，并将其乘以 vxb01234567c3 的第三个 lane 的 16 位数据，结果累加到 vacc1x4567 中
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    // 将 vxa2 的低 16 位数据提取出来，并将其乘以 vxb01234567c3 的第三个 lane 的 16 位数据，结果累加到 vacc2x0123 中
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    // 将 vxa2 的高 16 位数据提取出来，并将其乘以 vxb01234567c3 的第三个 lane 的 16 位数据，结果累加到 vacc2x4567 中
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    // 将 vxa3 的低 16 位数据提取出来，并将其乘以 vxb01234567c3 的第三个 lane 的 16 位数据，结果累加到 vacc3x0123 中
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
    // 将 vxa3 的高 16 位数据提取出来，并将其乘以 vxb01234567c3 的第三个 lane 的 16 位数据，结果累加到 vacc3x4567 中
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);

    // 加载指针 w 指向的 8 个字节数据到 vb01234567c4 中
    const uint8x8_t vb01234567c4 = vld1_u8(w);
    // 更新指针 w，使其指向下一个 8 个字节
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb01234567c4 与 vb_zero_point 每个字节做差，转换为有符号 16 位整数，存储到 vxb01234567c4 中
    const int16x8_t vxb01234567c4 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

    // 将 vxa0 的低 16 位数据提取出来，并将其乘以 vxb01234567c4 的第一个 lane 的 16 位数据，结果累加到 vacc0x0123 中
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    // 将 vxa0 的高 16 位数据提取出来，并将其乘以 vxb01234567c4 的第一个 lane 的 16 位数据，结果累加到 vacc0x4567 中
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    // 将 vxa1 的低 16 位数据提取出来，并将其乘以 vxb01234567c4 的第一个 lane 的 16 位数据，结果累加到 vacc1x0123 中
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    // 将 vxa1 的高 16 位数据提取出来，并将其乘以 vxb01234567c4 的第一个 lane 的 16 位数据，结果累加到 vacc1x4567 中
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    // 将 vxa2 的低 16 位数据提取出来，并将其乘以 vxb01234567c4 的第一个 lane 的 16 位数据，结果累加到 vacc2x0123 中
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    // 将 vxa2 的高 16 位数据提取出来，并将其乘以 vxb01234567c4 的第一个 lane 的 16 位数据，结果累加到 vacc2x4567 中
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    // 将 vxa3 的低 16 位数据提取出来，并将其乘以 vxb01234567c4 的第一个 lane 的 16 位数据，结果累加到 vacc3x0123 中
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
    // 将 vxa3 的高 16 位数据提取出来，并将其乘以 vxb01234567c4 的第一个 lane 的 16 位数据，结果累加到 vacc3x4567 中
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);

    // 加载指针 w 指向的 8 个字节数据到 vb01234567c5 中
    const uint8x8_t vb01234567c5 = vld1_u8(w);
    // 更新指针 w，使其指向下一个 8 个字节
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb01234567c5 与 vb_zero_point 每个字节做差，转换为有符号 16 位整数，存储到 vxb01234567c5 中
    const int16x8_t vxb01234567c5 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

    // 将 vxa0 的低 16 位数据提取出来，并将其乘以 vxb01234567c5 的第二个 lane 的 16 位数据，结果累加到 vacc0x0123 中
    vacc0x0123 = vmlal_lane_s16(
        vacc
    const uint8x8_t vb01234567c6 = vld1_u8(w);
    // 从地址 w 处加载一个 uint8x8_t 类型的数据到寄存器 vb01234567c6
    w = (const void*)((uintptr_t)w + 8);
    // 更新地址 w，使其指向下一个 8 字节的位置

    const int16x8_t vxb01234567c6 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
    // 将 vb01234567c6 和 vb_zero_point 进行无符号减法，并将结果转换为 int16x8_t 类型，存储在 vxb01234567c6 中

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    // 将 vxb01234567c6 的低位 8 个元素与 vxa0 的高位 8 个元素的第 2 个位置的元素进行饱和加法，并将结果累加到 vacc0x0123
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    // 将 vxb01234567c6 的高位 8 个元素与 vxa0 的高位 8 个元素的第 2 个位置的元素进行饱和加法，并将结果累加到 vacc0x4567
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    // 将 vxb01234567c6 的低位 8 个元素与 vxa1 的高位 8 个元素的第 2 个位置的元素进行饱和加法，并将结果累加到 vacc1x0123
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    // 将 vxb01234567c6 的高位 8 个元素与 vxa1 的高位 8 个元素的第 2 个位置的元素进行饱和加法，并将结果累加到 vacc1x4567
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    // 将 vxb01234567c6 的低位 8 个元素与 vxa2 的高位 8 个元素的第 2 个位置的元素进行饱和加法，并将结果累加到 vacc2x0123
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    // 将 vxb01234567c6 的高位 8 个元素与 vxa2 的高位 8 个元素的第 2 个位置的元素进行饱和加法，并将结果累加到 vacc2x4567
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    // 将 vxb01234567c6 的低位 8 个元素与 vxa3 的高位 8 个元素的第 2 个位置的元素进行饱和加法，并将结果累加到 vacc3x0123
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    // 将 vxb01234567c6 的高位 8 个元素与 vxa3 的高位 8 个元素的第 2 个位置的元素进行饱和加法，并将结果累加到 vacc3x4567

    const uint8x8_t vb01234567c7 = vld1_u8(w);
    // 从地址 w 处加载一个 uint8x8_t 类型的数据到寄存器 vb01234567c7
    w = (const void*)((uintptr_t)w + 8);
    // 更新地址 w，使其指向下一个 8 字节的位置

    const int16x8_t vxb01234567c7 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));
    // 将 vb01234567c7 和 vb_zero_point 进行无符号减法，并将结果转换为 int16x8_t 类型，存储在 vxb01234567c7 中

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    // 将 vxb01234567c7 的低位 8 个元素与 vxa0 的高位 8 个元素的第 3 个位置的元素进行饱和加法，并将结果累加到 vacc0x0123
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    // 将 vxb01234567c7 的高位 8 个元素与 vxa0 的高位 8 个元素的第 3 个位置的元素进行饱和加法，并将结果累加到 vacc0x4567
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    // 将 vxb01234567c7 的低位 8 个元素与 vxa1 的高位 8 个元素的第 3 个位置的元素进行饱和加法，并将结果累加到 vacc1x0123
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    // 将 vxb01234567c7 的高位 8 个元素与 vxa1 的高位 8 个元素的第 3 个位置的元素进行饱和加法，并将结果累加到 vacc1x4567
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    // 将 vxb01234567c7 的低位 8 个元素与 vxa2 的高位 8 个元素的第 3 个位置的元素进行饱和加法，并将结果累加到 vacc2x0123
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    // 将 vxb01234567c7 的高位 8 个元素与 vxa2 的高位 8 个元素的第 3 个位置的元素进行饱和加法，并将结果累加到 vacc2x4567
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
    // 将 vxb01234567c7 的低位 8 个元素与 vxa3 的高位 8 个元素的第 3 个位置的元素进行饱和加法，并将结果累加到 vacc3x0123
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
    // 将
    // 从数组 a3 的前一个元素中减去 a_predecrement 后加载 8 个字节的数据，转换成 uint8x8_t 类型
    const uint8x8_t va3 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));

    // 将加载的 uint8x8_t 类型数据转换成 int16x8_t 类型，同时进行零点调整
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));

    // 加载指针 w 所指向的 8 个字节数据到 uint8x8_t 类型
    const uint8x8_t vb01234567c0 = vld1_u8(w);
    // 将指针 w 增加 8 字节大小，以便下一次加载
    w = (const void*)((uintptr_t)w + 8);
    // 将加载的 uint8x8_t 类型数据转换成 int16x8_t 类型，同时进行零点调整
    const int16x8_t vxb01234567c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    // 对 vacc0x0123 到 vacc3x4567 进行乘法累加运算，vmlal_lane_s16 操作的详细说明需要在上下文中查找
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);

    }

  }

  // 使用乘法、加法和零点调整，计算并存储 vout0 到 vout3 的值
  float32x4_t vout0[] = {
    vaddq_f32(vmulq_f32(vmultiplier_c0123, vcvtq_f32_s32(vacc0x0123)), vbias[0]),
    vaddq_f32(vmulq_f32(vmultiplier_c4567, vcvtq_f32_s32(vacc0x4567)), vbias[1]),
  };
  float32x4_t vout1[] = {
    vaddq_f32(vmulq_f32(vmultiplier_c0123, vcvtq_f32_s32(vacc1x0123)), vbias[0]),
    vaddq_f32(vmulq_f32(vmultiplier_c4567, vcvtq_f32_s32(vacc1x4567)), vbias[1]),
  };
  float32x4_t vout2[] = {
    vaddq_f32(vmulq_f32(vmultiplier_c0123, vcvtq_f32_s32(vacc2x0123)), vbias[0]),
    vaddq_f32(vmulq_f32(vmultiplier_c4567, vcvtq_f32_s32(vacc2x4567)), vbias[1]),
  };
  float32x4_t vout3[] = {
    vaddq_f32(vmulq_f32(vmultiplier_c0123, vcvtq_f32_s32(vacc3x0123)), vbias[0]),
    vaddq_f32(vmulq_f32(vmultiplier_c4567, vcvtq_f32_s32(vacc3x4567)), vbias[1]),
  };

  // 将 vout0 到 vout3 的指针分别赋值给对应的指针变量
  float32x4_t * vout0_ptr = vout0;
  float32x4_t * vout1_ptr = vout1;
  float32x4_t * vout2_ptr = vout2;
  float32x4_t * vout3_ptr = vout3;

  // 计算输出 c0 到 c3 的地址，根据 mr 的值选择正确的输出地址
  float* c0 = c;
  float* c1 = c0 + c_stride;
  if (mr < 2) {
    c1 = c0;
  }
  float* c2 = c1 + c_stride;
  if (mr <= 2) {
    c2 = c1;
  }
  float* c3 = c2 + c_stride;
  if (mr != 4) {
    c3 = c2;
  }

  // 将 vout0_ptr 到 vout3_ptr 指向的数据存储到 c0 到 c3 所指向的地址，并更新地址指针
  for (; nr >= 4; nr -= 4) {
    vst1q_f32(c0, *vout0_ptr++);
    vst1q_f32(c1, *vout1_ptr++);
    vst1q_f32(c2, *vout2_ptr++);
    vst1q_f32(c3, *vout3_ptr++);

    c0 += 4;
    c1 += 4;
    c2 += 4;
    c3 += 4;
  }

  // 处理剩余不足 4 个元素的情况
  if (nr >= 2) {
    vst1_f32(c0, vget_low_f32(*vout0_ptr));
    vst1_f32(c1, vget_low_f32(*vout1_ptr));
    vst1_f32(c2, vget_low_f32(*vout2_ptr));
    vst1_f32(c3, vget_low_f32(*vout3_ptr));

    c0 += 2;
    (*vout0_ptr)[0] = (*vout0_ptr)[2];
    c1 += 2;
    (*vout1_ptr)[0] = (*vout1_ptr)[2];
    c2 += 2;
    (*vout2_ptr)[0] = (*vout2_ptr)[2];
    c3 += 2;
    # 将 vout3_ptr 指向的第一个元素的值赋给 vout3_ptr 指向的第三个元素
    (*vout3_ptr)[0] = (*vout3_ptr)[2];
    
    # 减去已处理的两个元素
    nr -= 2;
    
    # 如果还有剩余的元素没有处理
    if (nr != 0) {
        # 将 vout0_ptr 指向的第一个元素的值写入 c0 数组的第一个位置
        vst1q_lane_f32(c0, *vout0_ptr, 0);
        # 将 vout1_ptr 指向的第一个元素的值写入 c1 数组的第一个位置
        vst1q_lane_f32(c1, *vout1_ptr, 0);
        # 将 vout2_ptr 指向的第一个元素的值写入 c2 数组的第一个位置
        vst1q_lane_f32(c2, *vout2_ptr, 0);
        # 将 vout3_ptr 指向的第一个元素的值写入 c3 数组的第一个位置
        vst1q_lane_f32(c3, *vout3_ptr, 0);
    }
}



# 这行代码是一个单独的右大括号 '}'，用于结束一个代码块或函数定义。
# 在这个示例中，它可能用于结束一个类定义、函数定义或条件语句块等代码结构。
# 没有上下文的完整代码，无法确定其具体所属的代码块或作用。
```