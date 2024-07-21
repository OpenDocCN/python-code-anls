# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\6x4-neon.c`

```
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

void pytorch_q8gemm_ukernel_6x4__neon(
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
  // Load the first 4 elements of 'w' into a 128-bit vector 'vacc0x0123'
  int32x4_t vacc0x0123 = vld1q_s32(w);
  // Move 'w' pointer forward by 16 bytes to point to the next 4 elements
  w = (const void*)((uintptr_t)w + 16);
  // Duplicate 'vacc0x0123' into 'vacc1x0123', 'vacc2x0123', ..., 'vacc5x0123'
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc4x0123 = vacc0x0123;
  int32x4_t vacc5x0123 = vacc0x0123;

  // Set up pointers 'a0' to 'a5' to access the rows of matrix 'a'
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
  };
  const uint8_t* a5 = (const uint8_t*)((uintptr_t)a4 + a_stride);
  if (mr != 6) {
    a5 = a4;
  }

  // Load input zero point from 'quantization_params' into a neon vector
  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);
  // Load kernel zero point for the current output channel index into a neon vector
  uint8x8_t vb_zero_point =
      vld1_u8((const uint8_t*)&quantization_params->neon.kernel_zero_points
          [output_channel_index]);
  // Replicate the first 4 values of 'vb_zero_point' to fill all 8 lanes
  vb_zero_point = vset_lane_u8(vget_lane_u8(vb_zero_point, 0), vb_zero_point, 4);
  vb_zero_point = vset_lane_u8(vget_lane_u8(vb_zero_point, 1), vb_zero_point, 5);
  vb_zero_point = vset_lane_u8(vget_lane_u8(vb_zero_point, 2), vb_zero_point, 6);
  vb_zero_point = vset_lane_u8(vget_lane_u8(vb_zero_point, 3), vb_zero_point, 7);

  // Main loop iterating over 'k' elements, processing 8 elements per iteration
  for (; k >= 8; k -= 8) {
    // Load 8 elements from 'a0', 'a1', ..., 'a4' into neon vectors 'va0' to 'va4'
    const uint8x8_t va0 = vld1_u8(a0);
    a0 += 8;
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
    const uint8x8_t va1 = vld1_u8(a1);
    a1 += 8;
    const int16x8_t vxa1 =
        vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
    const uint8x8_t va2 = vld1_u8(a2);
    a2 += 8;
    const int16x8_t vxa2 =
        vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
    const uint8x8_t va3 = vld1_u8(a3);
    a3 += 8;
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));
    const uint8x8_t va4 = vld1_u8(a4);
    a4 += 8;
    const int16x8_t vxa4 =
        vreinterpretq_s16_u16(sub_zero_point(va4, va_zero_point));
    // ...
    // (remaining code is not shown due to space constraints)
  }
}
    const uint8x8_t va5 = vld1_u8(a5);
    // 从地址 a5 处加载 8 个 uint8_t 类型的数据到 va5 寄存器变量
    a5 += 8;
    // 将指针 a5 向后移动 8 个字节，指向下一个数据
    
    const int16x8_t vxa5 =
        vreinterpretq_s16_u16(sub_zero_point(va5, va_zero_point));
    // 将 va5 中的数据重新解释为 uint16_t 类型，并减去 va_zero_point，然后转换成 int16x8_t 类型保存到 vxa5 中
    
    const uint8x8_t vb0123c01 = vld1_u8(w);
    // 从地址 w 处加载 8 个 uint8_t 类型的数据到 vb0123c01 寄存器变量
    w = (const void*)((uintptr_t)w + 8);
    // 将指针 w 向后移动 8 个字节，指向下一个数据
    
    const int16x8_t vxb0123c01 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c01, vb_zero_point));
    // 将 vb0123c01 中的数据重新解释为 uint16_t 类型，减去 vb_zero_point，并转换成 int16x8_t 类型保存到 vxb0123c01 中
    
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c01), vget_low_s16(vxa0), 0);
    // 使用 vxb0123c01 的低位数据乘以 vxa0 的低位数据，结果加到 vacc0x0123 寄存器变量中的第 0 个位置
    
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c01), vget_low_s16(vxa1), 0);
    // 使用 vxb0123c01 的低位数据乘以 vxa1 的低位数据，结果加到 vacc1x0123 寄存器变量中的第 0 个位置
    
    // (重复上述操作，分别对 vacc2x0123, vacc3x0123, vacc4x0123, vacc5x0123 进行计算)
    
    const uint8x8_t vb0123c23 = vld1_u8(w);
    // 从地址 w 处加载 8 个 uint8_t 类型的数据到 vb0123c23 寄存器变量
    w = (const void*)((uintptr_t)w + 8);
    // 将指针 w 向后移动 8 个字节，指向下一个数据
    
    const int16x8_t vxb0123c23 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c23, vb_zero_point));
    // 将 vb0123c23 中的数据重新解释为 uint16_t 类型，减去 vb_zero_point，并转换成 int16x8_t 类型保存到 vxb0123c23 中
    
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c23), vget_low_s16(vxa0), 2);
    // 使用 vxb0123c23 的低位数据乘以 vxa0 的低位数据，结果加到 vacc0x0123 寄存器变量中的第 2 个位置
    
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c23), vget_low_s16(vxa1), 2);
    // 使用 vxb0123c23 的低位数据乘以 vxa1 的低位数据，结果加到 vacc1x0123 寄存器变量中的第 2 个位置
    
    // (重复上述操作，分别对 vacc2x0123, vacc3x0123, vacc4x0123, vacc5x0123 进行计算)
    
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa0), 3);
    // 使用 vxb0123c23 的高位数据乘以 vxa0 的低位数据，结果加到 vacc0x0123 寄存器变量中的第 3 个位置
    
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa1), 3);
    // 使用 vxb0123c23 的高位数据乘以 vxa1 的低位数据，结果加到 vacc1x0123 寄存器变量中的第 3 个位置
    
    // (重复上述操作，分别对 vacc2x0123, vacc3x0123, vacc4x0123 进行计算)
    // 使用 vmlal_lane_s16 函数将 vget_high_s16(vxb0123c23) 的第 3 个元素与 vacc5x0123 相加
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa5), 3);

    // 加载指针 w 指向的 8 个字节数据到 vb0123c45，并将 w 向后移动 8 字节
    const uint8x8_t vb0123c45 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb0123c45 减去 vb_zero_point，并转换为 int16x8_t 类型保存到 vxb0123c45
    const int16x8_t vxb0123c45 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c45, vb_zero_point));

    // 使用 vmlal_lane_s16 函数将 vget_low_s16(vxb0123c45) 的第 0 个元素与 vget_high_s16(vxa0) 的第 0 个元素与 vacc0x0123 相加
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa0), 0);
    // 类似地对 vacc1x0123 至 vacc5x0123 进行相同的操作，但是每次使用不同的寄存器和参数
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa3), 0);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa4), 0);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa5), 0);

    // 加载指针 w 指向的下一个 8 个字节数据到 vb0123c67，并将 w 向后移动 8 字节
    const uint8x8_t vb0123c67 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb0123c67 减去 vb_zero_point，并转换为 int16x8_t 类型保存到 vxb0123c67
    const int16x8_t vxb0123c67 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c67, vb_zero_point));

    // 继续使用 vmlal_lane_s16 函数将 vget_low_s16(vxb0123c67) 和 vget_high_s16(vxb0123c67) 的不同元素与 vget_high_s16(vxa0) 到 vget_high_s16(vxa5) 的不同元素与对应的 vacc0x0123 到 vacc5x0123 相加
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa3), 2);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa4), 2);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa5), 2);

    // 类似地，将 vget_high_s16(vxb0123c67) 的不同元素与 vget_high_s16(vxa0) 到 vget_high_s16(vxa4) 的不同元素与对应的 vacc0x0123 到 vacc4x0123 相加
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa3), 3);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa4), 3);
    // 累加矢量乘法操作，将 vget_high_s16(vxb0123c67) 的第 3 个元素与 vacc5x0123 累加
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa5), 3);
  }
  // 如果 k 不为 0，执行以下操作
  if (k != 0) {
    // 计算 a0 到 a5 中每个向量长度为 8-k 的数据量，并左移 8*k 位
    const size_t a_predecrement = 8 - k;
    // 创建一个移位量，用于左移 -8*k 位
    const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
    // 加载 a0 向量的前 8-k 个元素，左移 va_shift 指定的位数
    const uint8x8_t va0 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
    // 将 va0 向量转换为 int16x8_t 类型并进行零点减法操作，存入 vxa0
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
    // 类似处理 a1 到 a5 的操作，依次加载、移位、类型转换、零点减法
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

    // 加载并进行零点减法的 vb0123c0 向量，存入 vxb0123c0
    const uint8x8_t vb0123c0 = vreinterpret_u8_u32(vld1_dup_u32(w));
    // 更新指针 w，使其指向下一个元素
    w = (const void*)((uintptr_t)w + 4);
    // 将 vb0123c0 向量转换为 int16x8_t 类型，存入 vxb0123c0
    const int16x8_t vxb0123c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c0, vb_zero_point));

    // 分别对 vacc0x0123 到 vacc5x0123 进行累加操作，使用 vxb0123c0 和对应的 vxa0 到 vxa5 的低位元素
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa3), 0);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa4), 0);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa5), 0);

  }

  // 加载输出通道索引处的 requantization_scale_v
  const float32x4_t requantization_scale_v =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[
              output_channel_index]);

  // 将 vacc0x0123 到 vacc3x0123 分别转换为 float32x4_t 类型，并乘以 requantization_scale_v
  const float32x4_t vacc0x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x0123), requantization_scale_v);
  const float32x4_t vacc1x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x0123), requantization_scale_v);
  const float32x4_t vacc2x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x0123), requantization_scale_v);
  const float32x4_t vacc3x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x0123), requantization_scale_v);
  // 继续处理 vacc4x0123
  const float32x4_t vacc4x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc4x0123), requantization_scale_v);
    # 使用 NEON 指令，将整型向量 vacc4x0123 转换为单精度浮点向量，然后与 requantization_scale_v 向量逐元素相乘
    vmulq_f32(vcvtq_f32_s32(vacc4x0123), requantization_scale_v);
  # 声明并初始化一个单精度浮点向量 vacc5x0123_f，将整型向量 vacc5x0123 转换为单精度浮点向量，然后与 requantization_scale_v 向量逐元素相乘
  const float32x4_t vacc5x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc5x0123), requantization_scale_v);
#ifdef __aarch64__
  // 加载输出零点到一个 int16x8_t 向量
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  // 将浮点累加器向量转换为整数向量
  vacc0x0123 = vcvtnq_s32_f32(vacc0x0123_f);
  vacc1x0123 = vcvtnq_s32_f32(vacc1x0123_f);
  vacc2x0123 = vcvtnq_s32_f32(vacc2x0123_f);
  vacc3x0123 = vcvtnq_s32_f32(vacc3x0123_f);
  vacc4x0123 = vcvtnq_s32_f32(vacc4x0123_f);
  vacc5x0123 = vcvtnq_s32_f32(vacc5x0123_f);

  // 对累加器向量进行整数饱和加法，并加上输出零点
  const int16x8_t vacc01x0123 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc1x0123), voutput_zero_point);
  const int16x8_t vacc23x0123 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc3x0123), voutput_zero_point);
  const int16x8_t vacc45x0123 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc5x0123), voutput_zero_point);

  // 将整数向量转换为无符号整数向量，并合并为一个 uint8x16_t 向量
  uint8x16_t vout0123x0123 =
      vqmovun_high_s16(vqmovun_s16(vacc01x0123), vacc23x0123);
  // 将两个 uint8x8_t 向量合并为一个 uint8x16_t 向量
  uint8x8_t vout45x0123 = vqmovun_s16(vacc45x0123);

  // 加载输出最小值和最大值到 uint8x16_t 向量
  const uint8x16_t voutput_min =
      vld1q_dup_u8(&quantization_params->neon.output_min);
  const uint8x16_t voutput_max =
      vld1q_dup_u8(&quantization_params->neon.output_max);

  // 对输出向量进行饱和取值操作，确保在输出范围内
  vout0123x0123 = vmaxq_u8(vout0123x0123, voutput_min);
  vout45x0123 = vmax_u8(vout45x0123, vget_low_u8(voutput_min));
  vout0123x0123 = vminq_u8(vout0123x0123, voutput_max);
  vout45x0123 = vmin_u8(vout45x0123, vget_low_u8(voutput_max));
#endif
#else
  // 创建常量向量，用于进行数值限制和量化操作
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);  // 使用量化参数中的最小浮点数值创建常量向量
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);  // 使用量化参数中的最大浮点数值创建常量向量
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);  // 使用量化参数中的魔数浮点数值创建常量向量
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);  // 使用量化参数中的魔数整数值创建常量向量

  // 对输入向量进行数值限制，确保在[vfmin, vfmax]范围内
  const float32x4_t vacc0x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc0x0123_f, vfmin), vfmax);
  const float32x4_t vacc1x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc1x0123_f, vfmin), vfmax);
  const float32x4_t vacc2x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc2x0123_f, vfmin), vfmax);
  const float32x4_t vacc3x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc3x0123_f, vfmin), vfmax);
  const float32x4_t vacc4x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc4x0123_f, vfmin), vfmax);
  const float32x4_t vacc5x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc5x0123_f, vfmin), vfmax);

  // 将数值限制后的向量进行量化操作，将浮点数转换为整数
  vacc0x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc0x0123_f_clamped, vfmagic)), vimagic);
  vacc1x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc1x0123_f_clamped, vfmagic)), vimagic);
  vacc2x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc2x0123_f_clamped, vfmagic)), vimagic);
  vacc3x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc3x0123_f_clamped, vfmagic)), vimagic);
  vacc4x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc4x0123_f_clamped, vfmagic)), vimagic);
  vacc5x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc5x0123_f_clamped, vfmagic)), vimagic);

  // 将量化后的整数向量转换为16位整数，通过组合成更大的数据类型以提高效率
  const int16x8_t vacc01x0123 =
      vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc1x0123));
  const int16x8_t vacc23x0123 =
      vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc3x0123));
  const int16x8_t vacc45x0123 =
      vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc5x0123));

  // 将16位整数向量转换为无符号8位整数向量，准备输出到内存
  uint8x16_t vout0123x0123 =
      vcombine_u8(vqmovun_s16(vacc01x0123), vqmovun_s16(vacc23x0123));
  uint8x8_t vout45x0123 = vqmovun_s16(vacc45x0123);
#endif

  // 根据不同的行数和列数选择内存地址，用于写入量化后的结果
  uint8_t* c0 = c;  // 指向第0行的内存地址
  uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);  // 指向第1行的内存地址
  if (mr < 2) {
    c1 = c0;  // 如果行数小于2，则第1行与第0行共用内存地址
  }
  uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);  // 指向第2行的内存地址
  if (mr <= 2) {
    c2 = c1;  // 如果行数小于等于2，则第2行与第1行共用内存地址
  }
  uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);  // 指向第3行的内存地址
  if (mr < 4) {
    c3 = c2;  // 如果行数小于4，则第3行与第2行共用内存地址
  }
  uint8_t* c4 = (uint8_t*)((uintptr_t)c3 + c_stride);  // 指向第4行的内存地址
  if (mr <= 4) {
    c4 = c3;  // 如果行数小于等于4，则第4行与第3行共用内存地址
  }
  uint8_t* c5 = (uint8_t*)((uintptr_t)c4 + c_stride);  // 指向第5行的内存地址
  if (mr != 6) {
    c5 = c4;  // 如果行数不等于6，则第5行与第4行共用内存地址
  }

  // 根据列数的不同，将量化结果写入内存对应的地址
  if (nr == 4) {
    vst1q_lane_u32(
        __builtin_assume_aligned(c0, 1),
        vreinterpretq_u32_u8(vout0123x0123),
        0);  // 将第0列的数据写入内存
    vst1q_lane_u32(
        __builtin_assume_aligned(c1, 1),
        vreinterpretq_u32_u8(vout0123x0123),
        1);  // 将第1列的数据写入内存
    vst1q_lane_u32(
        __builtin_assume_aligned(c2, 1),
        vreinterpretq_u32_u8(vout0123x0123),
        2);  // 将第2列的数据写入内存
    vst1q_lane_u32(
        __builtin_assume_aligned(c3, 1),
        vreinterpretq_u32_u8(vout0123x0123),
        3);  // 将第3列的数据写入内存
    # 如果向量长度为1，执行以下操作
    vst1_lane_u32(
        __builtin_assume_aligned(c4, 1), vreinterpret_u32_u8(vout45x0123), 0);
    # 将向量 vout45x0123 中的第一个元素解释为无符号32位整数，存储到地址 c4 所指向的位置

    vst1_lane_u32(
        __builtin_assume_aligned(c5, 1), vreinterpret_u32_u8(vout45x0123), 1);
    # 将向量 vout45x0123 中的第二个元素解释为无符号32位整数，存储到地址 c5 所指向的位置
  } else {
    # 如果向量长度大于等于2，执行以下操作

    # 向 c0, c1, c2, c3 分别存储 vout0123x0123 中的第 0, 2, 4, 6 个元素作为无符号16位整数
    vst1q_lane_u16(
        __builtin_assume_aligned(c0, 1),
        vreinterpretq_u16_u8(vout0123x0123),
        0);
    c0 += 2;
    vst1q_lane_u16(
        __builtin_assume_aligned(c1, 1),
        vreinterpretq_u16_u8(vout0123x0123),
        2);
    c1 += 2;
    vst1q_lane_u16(
        __builtin_assume_aligned(c2, 1),
        vreinterpretq_u16_u8(vout0123x0123),
        4);
    c2 += 2;
    vst1q_lane_u16(
        __builtin_assume_aligned(c3, 1),
        vreinterpretq_u16_u8(vout0123x0123),
        6);
    c3 += 2;

    # 向 c4, c5 分别存储 vout45x0123 中的第 0, 2 个元素作为无符号16位整数
    vst1_lane_u16(
        __builtin_assume_aligned(c4, 1), vreinterpret_u16_u8(vout45x0123), 0);
    c4 += 2;
    vst1_lane_u16(
        __builtin_assume_aligned(c5, 1), vreinterpret_u16_u8(vout45x0123), 2);
    c5 += 2;

    # 将向量 vout0123x0123 向右移动两个位置，用于下一轮处理
    vout0123x0123 = vextq_u8(vout0123x0123, vout0123x0123, 2);
    # 将向量 vout45x0123 向右移动两个位置，用于下一轮处理
    vout45x0123 = vext_u8(vout45x0123, vout45x0123, 2);
    # 减少向量处理的长度计数器 nr
    nr -= 2;
  }

  # 如果向量长度不为0，执行以下操作
  if (nr != 0) {
    # 将 vout0123x0123 中的第 0, 4, 8, 12 个元素作为无符号8位整数存储到 c0, c1, c2, c3 所指向的位置
    vst1q_lane_u8(__builtin_assume_aligned(c0, 1), vout0123x0123, 0);
    vst1q_lane_u8(__builtin_assume_aligned(c1, 1), vout0123x0123, 4);
    vst1q_lane_u8(__builtin_assume_aligned(c2, 1), vout0123x0123, 8);
    vst1q_lane_u8(__builtin_assume_aligned(c3, 1), vout0123x0123, 12);

    # 将 vout45x0123 中的第 0, 4 个元素作为无符号8位整数存储到 c4, c5 所指向的位置
    vst1_lane_u8(__builtin_assume_aligned(c4, 1), vout45x0123, 0);
    vst1_lane_u8(__builtin_assume_aligned(c5, 1), vout45x0123, 4);
  }
}



# 这行代码关闭了一个代码块。在大多数编程语言中，闭合的大括号表示一个代码块的结束。
```