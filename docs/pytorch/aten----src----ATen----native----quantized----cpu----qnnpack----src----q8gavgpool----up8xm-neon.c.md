# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gavgpool\up8xm-neon.c`

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

#include <qnnpack/q8gavgpool.h>

void pytorch_q8gavgpool_ukernel_up8xm__neon(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[restrict static 1]) {
  assert(m >= 1);
  assert(n < 8);

  // 从量化参数结构中加载偏置值到 NEON 寄存器
  const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);
  // 初始化累加器向量，使用偏置值
  int32x4_t vacc_lo = vbias;
  int32x4_t vacc_hi = vbias;

  // 处理每个 m >= 8 的块
  while (m >= 8) {
    // 加载输入数据到 NEON 寄存器
    const uint8x8_t vinput = vld1_u8(input);
    // 更新输入指针以指向下一个数据块
    input += input_stride;
    // 将 uint8_t 类型转换为 int16x8_t 类型，以便进行加法操作
    const int16x8_t vxinput = vreinterpretq_s16_u16(vmovl_u8(vinput));
    // 分别更新低位和高位累加器
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vxinput));
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vxinput));

    // 减少处理的行数 m
    m -= 8;
  }

  // 处理剩余的行数 m < 8
  while (m-- != 0) {
    // 更新输入指针，跳过前 n 个字节
    input += n;
    // 初始化一个包含 0 的向量
    uint8x8_t vinput = vmov_n_u8(0);
    // 如果 n 的最低位为 1，则加载最后一个字节到向量
    if (n & 1) {
      input -= 1;
      vinput = vld1_lane_u8(input, vinput, 0);
    }
    // 如果 n 的次低位为 1，则加载最后两个字节到向量
    if (n & 2) {
      vinput = vext_u8(vinput, vinput, 6);
      input -= 2;
      vinput = vreinterpret_u8_u16(vld1_lane_u16(
          __builtin_assume_aligned(input, 1), vreinterpret_u16_u8(vinput), 0));
    }
    // 如果 n 的第三位为 1，则加载最后四个字节到向量
    if (n & 4) {
      vinput = vext_u8(vinput, vinput, 4);
      input -= 4;
      vinput = vreinterpret_u8_u32(vld1_lane_u32(
          __builtin_assume_aligned(input, 1), vreinterpret_u32_u8(vinput), 0));
    }
    // 更新输入指针以指向下一个数据块
    input += input_stride;

    // 将 uint8_t 类型转换为 int16x8_t 类型，以便进行加法操作
    const int16x8_t vxinput = vreinterpretq_s16_u16(vmovl_u8(vinput));
    // 分别更新低位和高位累加器
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vxinput));
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vxinput));
  }

  // 从量化参数结构中加载缩放因子到 NEON 寄存器
  const float32x4_t vscale =
      vdupq_n_f32(quantization_params->neon.scale);
  // 从量化参数结构中加载输出零点到 NEON 寄存器
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);

  // 将累加器中的整数值转换为浮点数，并乘以缩放因子
  float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);
  float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);
  vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
  vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);

  // 如果是 AArch64 架构
#if defined(__aarch64__)
  // 从量化参数结构中加载输出最小值到 NEON 寄存器
  const uint8x8_t voutput_min =
      vld1_dup_u8(&quantization_params->neon.output_min);
  // 从量化参数结构中加载输出最大值到 NEON 寄存器
  const uint8x8_t voutput_max =
      vld1_dup_u8(&quantization_params->neon.output_max);

  // 将浮点数累加器四舍五入为整数，加上输出零点，并限制在指定范围内
  vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
  vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
  const int16x8_t vacc = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
  // 将结果转换回 uint8_t 类型
  uint8x8_t vout = vqmovun_s16(vacc);
  // 对结果进行上下限制
  vout = vmax_u8(vout, voutput_min);
  vout = vmin_u8(vout, voutput_max);
#else
  // 复制四个浮点数到向量 vfmin 中
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  // 复制四个浮点数到向量 vfmax 中
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  // 复制四个浮点数到向量 vfmagic 中
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  // 复制四个整数到向量 vimagic 中
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);

  // 对低位四个浮点数向量进行范围限制，并将结果存入 vacc_lo_f
  vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
  // 对高位四个浮点数向量进行范围限制，并将结果存入 vacc_hi_f
  vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

  // 计算低位四个浮点数向量的四舍五入整数值，并应用魔数偏移，结果存入 vacc_lo
  vacc_lo = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
  // 计算高位四个浮点数向量的四舍五入整数值，并应用魔数偏移，结果存入 vacc_hi
  vacc_hi = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
  // 将两个四舍五入整数向量合并成一个十六位整数向量 vacc
  const int16x8_t vacc =
      vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
  // 将十六位整数向量 vacc 转换成八位无符号整数向量 vout
  uint8x8_t vout = vqmovun_s16(vacc);
#endif

// 如果 n 的最低两位是 1，则处理剩余的数据
if (n & 4) {
  // 将 vout 中的第一个字（32 位无符号整数）存储到 output 指向的地址处
  vst1_lane_u32(
      __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
  // 将 output 向前移动四个字节
  output += 4;
  // 将 vout 向右移动四个字节，将下一个字节组成新的 vout
  vout = vext_u8(vout, vout, 4);
}
// 如果 n 的次低两位是 1，则处理剩余的数据
if (n & 2) {
  // 将 vout 中的第一个字（16 位无符号整数）存储到 output 指向的地址处
  vst1_lane_u16(
      __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
  // 将 output 向前移动两个字节
  output += 2;
  // 将 vout 向右移动两个字节，将下一个字节组成新的 vout
  vout = vext_u8(vout, vout, 2);
}
// 如果 n 的最低位是 1，则处理剩余的数据
if (n & 1) {
  // 将 vout 中的第一个字节存储到 output 指向的地址处
  vst1_lane_u8(output, vout, 0);
}
```