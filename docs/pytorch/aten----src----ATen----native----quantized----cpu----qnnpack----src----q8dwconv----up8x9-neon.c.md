# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\up8x9-neon.c`

```
/*
 * 版权声明：
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * 本源代码使用BSD风格许可证授权，许可证详情请参阅源目录中的LICENSE文件。
 */

#include <arm_neon.h>

#include <qnnpack/q8dwconv.h>
#include <requantization/runtime-neon.h>

void pytorch_q8dwconv_ukernel_up8x9__neon(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {
  // 加载输入的零点偏移量为uint8x8_t类型
  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);
  // 复制卷积核的零点偏移量为uint8x8_t类型
  const uint8x8_t vkernel_zero_point =
      vdup_n_u8(quantization_params->neon.kernel_zero_points[0]);
  // 复制重新量化缩放因子为float32x4_t类型
  const float32x4_t requantization_scale_v =
      vdupq_n_f32(quantization_params->neon.requantization_scales[0]);
#ifdef __aarch64__
  // 加载输出的零点偏移量为int16x8_t类型
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  // 加载输出的最小值为uint8x8_t类型
  const uint8x8_t voutput_min =
      vld1_dup_u8(&quantization_params->neon.output_min);
  // 加载输出的最大值为uint8x8_t类型
  const uint8x8_t voutput_max =
      vld1_dup_u8(&quantization_params->neon.output_max);
#else
  // 加载浮点数类型的输出最小值为float32x4_t类型
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  // 加载浮点数类型的输出最大值为float32x4_t类型
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  // 加载浮点数类型的魔数为float32x4_t类型
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  // 加载整数类型的魔数为int32x4_t类型
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
#endif

#ifdef __aarch64__
  // 在AArch64架构上，由于寄存器数量较多，可以一次处理多个像素
  if (input_stride == 3 * sizeof(void*)) {
    // 如果输入跨度等于3倍指针大小，则执行以下操作
    }
    if (output_width == 0) {
      return;  // 如果输出宽度为0，则直接返回
    }
  }
#endif

  do {
    // 加载输入数据指针数组的各个元素
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];

    // 更新输入数据指针数组，使其指向下一个元素
    input = (const uint8_t**)((uintptr_t)input + input_stride);

    // 初始化通道数和权重
    size_t c = channels;
    const void* w = weights;
#ifdef __aarch64__
      // 将浮点数类型的累加结果向下转换为整数类型
      vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
      vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

      // 对累加结果加上输出的零点偏移量
      const int16x8_t vacc = vqaddq_s16(
          vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);

      // 将16位整数类型的累加结果转换为uint8x8_t类型
      uint8x8_t vout = vqmovun_s16(vacc);
      // 对结果进行最大值截断
      vout = vmax_u8(vout, voutput_min);
      // 对结果进行最小值截断
      vout = vmin_u8(vout, voutput_max);
#ifdef __aarch64__
      // 如果目标平台是 AArch64 架构，则执行以下代码段
      vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
      vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

      // 将浮点数转换为整数，并加上输出的零点偏移，得到16位整数向量 vacc
      const int16x8_t vacc = vqaddq_s16(
          vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);

      // 将16位整数向量 vacc 转换为无符号8位整数向量 vout
      uint8x8_t vout = vqmovun_s16(vacc);
      // 对 vout 中的每个元素进行饱和操作，确保不超出指定的最小和最大值范围
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);
#else
      // 如果不是 AArch64 架构，则执行以下代码段

      // 对浮点数向量进行截断和饱和操作，将每个元素限制在指定的最小和最大值范围内
      const float32x4_t vacc_lo_f_clamped =
          vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
      const float32x4_t vacc_hi_f_clamped =
          vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
      // 将浮点数向量转换为整数向量，然后加上魔数，最后得到16位整数向量 vacc
      vacc_lo = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)), vimagic);
      vacc_hi = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)), vimagic);
      const int16x8_t vacc =
          vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

      // 将16位整数向量 vacc 转换为无符号8位整数向量 vout
      uint8x8_t vout = vqmovun_s16(vacc);
#endif

      // 根据标志位 c 的值，分别处理不同位宽的输出数据
      if (c & 4) {
        // 将 vout 中的数据存储到 output 指向的地址，每次存储一个32位无符号整数
        vst1_lane_u32(
            __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
        output += 4;
        // 将 vout 向量的数据从索引为4的位置扩展到索引为0的位置
        vout = vext_u8(vout, vout, 4);
      }
      if (c & 2) {
        // 将 vout 中的数据存储到 output 指向的地址，每次存储一个16位无符号整数
        vst1_lane_u16(
            __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
        output += 2;
        // 将 vout 向量的数据从索引为2的位置扩展到索引为0的位置
        vout = vext_u8(vout, vout, 2);
      }
      if (c & 1) {
        // 将 vout 中的数据存储到 output 指向的地址，每次存储一个8位无符号整数
        vst1_lane_u8(__builtin_assume_aligned(output, 1), vout, 0);
        output++;
      }
    }

    // 更新 output 的地址，以便处理下一行数据
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);
}
```