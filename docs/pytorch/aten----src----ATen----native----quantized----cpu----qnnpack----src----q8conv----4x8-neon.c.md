# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8conv\4x8-neon.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8conv.h>
#include <requantization/runtime-neon.h>

void pytorch_q8conv_ukernel_4x8__neon(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {
  // Load input zero point for quantization from quantization_params
  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);

  // Load kernel zero point for the current output channel index
  // Assumes that kernel_zero_points is padded to be multiple of 8
  const uint8x8_t vb_zero_point =
      vld1_u8((const uint8_t*)&quantization_params->neon.kernel_zero_points
          [output_channel_index]);

  // Load 32-bit integer accumulator values from weight data
  int32x4_t vacc0x0123 = vld1q_s32(w);
  w = (void*)((uintptr_t)w + sizeof(int32x4_t));
  int32x4_t vacc0x4567 = vld1q_s32(w);
  w = (void*)((uintptr_t)w + sizeof(int32x4_t));
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc1x4567 = vacc0x4567;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc2x4567 = vacc0x4567;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc3x4567 = vacc0x4567;

  // Iterate over the kernel size ks, processing 4 rows of input
  do {
    const uint8_t* restrict a0 = *a++;
    const uint8_t* restrict a1 = *a++;
    const uint8_t* restrict a2 = *a++;
    const uint8_t* restrict a3 = *a++;

    size_t k = kc;
    // Remaining kernel processing is omitted here for brevity

  } while (--ks != 0);

  // Load requantization scales for the current output channel index
  const float32x4_t requantization_scale_c0123 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[output_channel_index]);
  const float32x4_t requantization_scale_c4567 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[output_channel_index + 4]);

  /*
   * Convert int32_t accumulator values to float32 and apply requantization scales.
   * This operation involves converting integers to floating-point values
   * and then multiplying them with FP32 scales, ensuring statistically unbiased
   * rounding.
   */
  const float32x4_t vacc0x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x0123), requantization_scale_c0123);
  const float32x4_t vacc1x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x0123), requantization_scale_c0123);
  const float32x4_t vacc2x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x0123), requantization_scale_c0123);
  const float32x4_t vacc3x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x0123), requantization_scale_c0123);
    # 将整型向量 vacc3x0123 转换为单精度浮点向量，然后与 requantization_scale_c0123 向量逐元素相乘
    vmulq_f32(vcvtq_f32_s32(vacc3x0123), requantization_scale_c0123);
  # 创建名为 vacc0x4567_f 的常量变量，其值为将整型向量 vacc0x4567 转换为单精度浮点向量后，与 requantization_scale_c4567 向量逐元素相乘的结果
  const float32x4_t vacc0x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x4567), requantization_scale_c4567);
  # 创建名为 vacc1x4567_f 的常量变量，其值为将整型向量 vacc1x4567 转换为单精度浮点向量后，与 requantization_scale_c4567 向量逐元素相乘的结果
  const float32x4_t vacc1x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x4567), requantization_scale_c4567);
  # 创建名为 vacc2x4567_f 的常量变量，其值为将整型向量 vacc2x4567 转换为单精度浮点向量后，与 requantization_scale_c4567 向量逐元素相乘的结果
  const float32x4_t vacc2x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x4567), requantization_scale_c4567);
  # 创建名为 vacc3x4567_f 的常量变量，其值为将整型向量 vacc3x4567 转换为单精度浮点向量后，与 requantization_scale_c4567 向量逐元素相乘的结果
  const float32x4_t vacc3x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x4567), requantization_scale_c4567);
#ifdef __aarch64__
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  /*
   * 使用ARMv8指令进行“浮点转整数四舍五入”操作，将浮点数向整数转换。
   * 这个操作在AArch64架构下始终可用，会在溢出时饱和结果。在最后阶段会进行最后的限制。
   */
  vacc0x0123 = vcvtnq_s32_f32(vacc0x0123_f);
  vacc1x0123 = vcvtnq_s32_f32(vacc1x0123_f);
  vacc2x0123 = vcvtnq_s32_f32(vacc2x0123_f);
  vacc3x0123 = vcvtnq_s32_f32(vacc3x0123_f);
  vacc0x4567 = vcvtnq_s32_f32(vacc0x4567_f);
  vacc1x4567 = vcvtnq_s32_f32(vacc1x4567_f);
  vacc2x4567 = vcvtnq_s32_f32(vacc2x4567_f);
  vacc3x4567 = vcvtnq_s32_f32(vacc3x4567_f);

  const int16x8_t vacc0x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

  uint8x16_t vout0x01234567_1x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
  uint8x16_t vout2x01234567_3x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);

  const uint8x16_t voutput_min =
      vld1q_dup_u8(&quantization_params->neon.output_min);
  const uint8x16_t voutput_max =
      vld1q_dup_u8(&quantization_params->neon.output_max);

  vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
  vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
  vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
  vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
#endif

  // 将指针c0指向的值赋给c0，并根据c_stride调整c1的地址
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
  // 如果mr小于2，则将c1指向c0的地址
  if (mr < 2) {
    c1 = c0;
  }
  // 根据c_stride调整c2的地址，并根据mr的值进行进一步的调整
  uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  // 根据c_stride调整c3的地址，并根据mr的值进行进一步的调整
  uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
  if (mr != 4) {
    c3 = c2;
  }
  // 如果nr为8，则将处理后的数据写入c0、c1、c2和c3中的相应位置
  if (nr == 8) {
    vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
    vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
    vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
    vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
  } else {
    if (nr >= 4) {
      # 将 vout0x01234567_1x01234567 的第一个 32 位数据存储到 c0 中
      vst1q_lane_u32(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          0);
      c0 += 4;
      # 将 vout0x01234567_1x01234567 的第三个 32 位数据存储到 c1 中
      vst1q_lane_u32(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          2);
      c1 += 4;
      # 将 vout2x01234567_3x01234567 的第一个 32 位数据存储到 c2 中
      vst1q_lane_u32(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          0);
      c2 += 4;
      # 将 vout2x01234567_3x01234567 的第三个 32 位数据存储到 c3 中
      vst1q_lane_u32(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          2);
      c3 += 4;
      # 将 vout0x01234567_1x01234567 向左移动 4 个字节，准备下一轮操作
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      # 将 vout2x01234567_3x01234567 向左移动 4 个字节，准备下一轮操作
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      # 减少剩余待处理数据量 nr
      nr -= 4;
    }
    if (nr >= 2) {
      # 将 vout0x01234567_1x01234567 的第一个 16 位数据存储到 c0 中
      vst1q_lane_u16(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          0);
      c0 += 2;
      # 将 vout0x01234567_1x01234567 的第五个 16 位数据存储到 c1 中
      vst1q_lane_u16(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          4);
      c1 += 2;
      # 将 vout2x01234567_3x01234567 的第一个 16 位数据存储到 c2 中
      vst1q_lane_u16(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          0);
      c2 += 2;
      # 将 vout2x01234567_3x01234567 的第五个 16 位数据存储到 c3 中
      vst1q_lane_u16(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          4);
      c3 += 2;
      # 将 vout0x01234567_1x01234567 向左移动 2 个字节，准备下一轮操作
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      # 将 vout2x01234567_3x01234567 向左移动 2 个字节，准备下一轮操作
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      # 减少剩余待处理数据量 nr
      nr -= 2;
    }
    if (nr != 0) {
      # 将 vout0x01234567_1x01234567 的第一个 8 位数据存储到 c0 中
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      # 将 vout0x01234567_1x01234567 的第九个 8 位数据存储到 c1 中
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      # 将 vout2x01234567_3x01234567 的第一个 8 位数据存储到 c2 中
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      # 将 vout2x01234567_3x01234567 的第九个 8 位数据存储到 c3 中
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
    }
}



# 这行代码表示一个代码块的结束，通常在条件语句、循环、函数定义或类定义的末尾出现。
```