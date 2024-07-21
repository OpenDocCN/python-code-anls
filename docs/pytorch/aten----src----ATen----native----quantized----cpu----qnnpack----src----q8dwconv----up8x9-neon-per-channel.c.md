# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\up8x9-neon-per-channel.c`

```
/*
 * 版权声明：
 * 版权所有（c）Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录下的LICENSE文件中的BSD风格许可证进行许可。
 */

#include <arm_neon.h>  // 包含 ARM NEON 指令集的头文件

#include <qnnpack/q8dwconv.h>  // 包含 QNNPACK 的深度可分离卷积头文件
#include <requantization/runtime-neon.h>  // 包含 NEON 运行时库的头文件

void pytorch_q8dwconv_ukernel_up8x9_per_channel__neon(
    size_t channels,  // 输入通道数
    size_t output_width,  // 输出宽度
    const uint8_t** input,  // 输入数据的指针数组
    const void* weights,  // 权重数据
    uint8_t* output,  // 输出数据
    size_t input_stride,  // 输入步幅
    size_t output_increment,  // 输出增量
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {  // 量化参数结构体数组
  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);  // 加载输入零点

#ifdef __aarch64__
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);  // 加载输出零点
  const uint8x8_t voutput_min =
      vld1_dup_u8(&quantization_params->neon.output_min);  // 加载输出最小值
  const uint8x8_t voutput_max =
      vld1_dup_u8(&quantization_params->neon.output_max);  // 加载输出最大值
#else
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);  // 加载量化参数的最小浮点值
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);  // 加载量化参数的最大浮点值
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);  // 加载量化参数的魔法浮点值
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);  // 加载量化参数的魔法整数值
#endif

#ifdef __aarch64__
  /* AArch64 上拥有更多寄存器，可以一次处理更多像素 */
  if (input_stride == 3 * sizeof(void*)) {  // 如果输入步幅等于 3 倍指针大小
    }
    if (output_width == 0) {  // 如果输出宽度为0，则直接返回
      return;
    }
  }
#endif

  do {
    const uint8_t* i0 = input[0];  // 加载输入数据指针
    const uint8_t* i1 = input[1];  // 加载输入数据指针
    const uint8_t* i2 = input[2];  // 加载输入数据指针
    const uint8_t* i3 = input[3];  // 加载输入数据指针
    const uint8_t* i4 = input[4];  // 加载输入数据指针
    const uint8_t* i5 = input[5];  // 加载输入数据指针
    const uint8_t* i6 = input[6];  // 加载输入数据指针
    const uint8_t* i7 = input[7];  // 加载输入数据指针
    const uint8_t* i8 = input[8];  // 加载输入数据指针

    input = (const uint8_t**)((uintptr_t)input + input_stride);  // 更新输入数据指针位置

    size_t c = channels;  // 初始化通道数
    const void* w = weights;  // 初始化权重数据指针
#ifdef __aarch64__
      vacc_lo = vcvtnq_s32_f32(vacc_lo_f);  // 将浮点累加结果转换为整数
      vacc_hi = vcvtnq_s32_f32(vacc_hi_f);  // 将浮点累加结果转换为整数

      const int16x8_t vacc = vqaddq_s16(
          vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);  // 对累加结果进行饱和加法

      uint8x8_t vout = vqmovun_s16(vacc);  // 将饱和加法结果转换为无符号字节
      vout = vmax_u8(vout, voutput_min);  // 与输出最小值比较取较大值
      vout = vmin_u8(vout, voutput_max);  // 与输出最大值比较取较小值
#else
      const float32x4_t vacc_lo_f_clamped =
          vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);  // 对浮点累加结果进行范围约束
      const float32x4_t vacc_hi_f_clamped =
          vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);  // 对浮点累加结果进行范围约束
      vacc_lo = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)), vimagic);  // 将浮点累加结果转换为整数并应用魔法值调整
      vacc_hi = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)), vimagic);  // 将浮点累加结果转换为整数并应用魔法值调整
      const int16x8_t vacc =
          vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));  // 将两个累加结果合并为一个 8 通道 16 位整数向量

      uint8x8_t vout = vqmovun_s16(vacc);  // 将累加结果转换为无符号 8 位整数
#endif

      // 存储8个无符号8位整数到output指向的内存位置
      vst1_u8(output, vout);
      // 更新output指针，使其指向下一个8字节的内存位置
      output += 8;
    }
#ifdef __aarch64__
      // 将vacc_lo_f和vacc_hi_f中的浮点数转换为整数向量
      vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
      vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

      // 创建包含vacc_lo和vacc_hi的有符号16位整数向量
      const int16x8_t vacc = vqaddq_s16(
          vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);

      // 将有符号16位整数向量vacc转换为无符号8位整数向量vout
      uint8x8_t vout = vqmovun_s16(vacc);
      // 将vout中的值限制在voutput_min和voutput_max之间
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);
#else
      // 将vacc_lo_f_clamped和vacc_hi_f_clamped中的浮点数向量转换为整数向量
      const float32x4_t vacc_lo_f_clamped =
          vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
      const float32x4_t vacc_hi_f_clamped =
          vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
      // 对vacc_lo和vacc_hi执行整数向量操作，并进行魔数调整
      vacc_lo = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)), vimagic);
      vacc_hi = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)), vimagic);
      // 创建包含vacc_lo和vacc_hi的有符号16位整数向量
      const int16x8_t vacc =
          vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

      // 将有符号16位整数向量vacc转换为无符号8位整数向量vout
      uint8x8_t vout = vqmovun_s16(vacc);
#endif

      // 如果c的最低2位为1，则执行以下操作
      if (c & 4) {
        // 将vout的第0个元素存储到output指向的内存位置，并更新output指针
        vst1_lane_u32(
            __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
        output += 4;
        // 将vout向左移动4个元素的位置，更新vout
        vout = vext_u8(vout, vout, 4);
      }
      // 如果c的最低1位为1，则执行以下操作
      if (c & 2) {
        // 将vout的前两个元素存储到output指向的内存位置，并更新output指针
        vst1_lane_u16(
            __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
        output += 2;
        // 将vout向左移动2个元素的位置，更新vout
        vout = vext_u8(vout, vout, 2);
      }
      // 如果c的最低1位为1，则执行以下操作
      if (c & 1) {
        // 将vout的第0个元素存储到output指向的内存位置，并更新output指针
        vst1_lane_u8(__builtin_assume_aligned(output, 1), vout, 0);
        output++;
      }
    }

    // 更新output指针，使其指向下一个输出行的起始位置
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);
}
```