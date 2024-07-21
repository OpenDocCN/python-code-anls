# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\mp8x25-neon-per-channel.c`

```
/*
 * 这段代码实现了基于 ARM NEON 指令集的量化卷积操作，用于深度卷积神经网络加速。
 */

#include <arm_neon.h>  // 包含 ARM NEON 指令集头文件

#include <qnnpack/q8dwconv.h>  // 包含 QNNPACK 的深度卷积函数头文件

void pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon(
    size_t channels,  // 输入通道数
    size_t output_width,  // 输出宽度
    const uint8_t** input,  // 输入数据的指针数组
    const void* weights,  // 权重数据
    int32_t* outacc32,  // 输出累加器
    uint8_t* output,  // 输出数据
    size_t input_stride,  // 输入步长
    size_t output_increment,  // 输出增量
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {  // 量化参数结构体

  const uint8x8_t vinput_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);
  // 装载输入零点值到 uint8x8_t 类型变量

#ifdef __aarch64__
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  // 装载输出零点值到 int16x8_t 类型变量
  const uint8x8_t voutput_min = vld1_dup_u8(&quantization_params->neon.output_min);
  // 装载输出最小值到 uint8x8_t 类型变量
  const uint8x8_t voutput_max = vld1_dup_u8(&quantization_params->neon.output_max);
  // 装载输出最大值到 uint8x8_t 类型变量
#else
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  // 使用量化参数装载 vfmin 到 float32x4_t 类型变量
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  // 使用量化参数装载 vfmax 到 float32x4_t 类型变量
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  // 使用量化参数装载 vfmagic 到 float32x4_t 类型变量
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
  // 使用量化参数装载 vimagic 到 int32x4_t 类型变量
#endif

  do {
    uint8_t* output_start = output;  // 记录输出起始位置的指针
    int32_t* outacc = outacc32;  // 记录累加器的指针
    const void* w = weights;  // 记录权重数据的指针
    }
    }

#ifdef __aarch64__
        vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
        vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
        // 对累加器结果进行量化和加零点值处理，得到 int16x8_t 类型的累加器

        uint8x8_t vout = vqmovun_s16(vacc);
        // 将 int16x8_t 类型的累加器结果转换为 uint8x8_t 类型的输出

        vout = vmax_u8(vout, voutput_min);
        // 使用 SIMD 指令执行 uint8x8_t 类型的向量最大值计算

        vout = vmin_u8(vout, voutput_max);
        // 使用 SIMD 指令执行 uint8x8_t 类型的向量最小值计算
#else
        const float32x4_t vacc_lo_f_clamped =
            vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
        // 对浮点型累加器结果进行范围限制处理

        const float32x4_t vacc_hi_f_clamped =
            vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
        // 对浮点型累加器结果进行范围限制处理

        vacc_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)), vimagic);
        // 对量化后的累加器结果执行后处理（减去魔数）

        vacc_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)), vimagic);
        // 对量化后的累加器结果执行后处理（减去魔数）

        const int16x8_t vacc =
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
        // 将两个 int32x4_t 类型的累加器结果组合成 int16x8_t 类型

        uint8x8_t vout = vqmovun_s16(vacc);
        // 将 int16x8_t 类型的累加器结果转换为 uint8x8_t 类型的输出
#endif
#else
        // 如果条件不满足，执行以下代码块

        // 将vacc_lo_f向量中的每个元素与vfmin向量中的对应元素比较，取较大值，并与vfmax向量中的对应元素比较，取较小值，得到vacc_lo_f_clamped向量
        const float32x4_t vacc_lo_f_clamped =
            vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);

        // 将vacc_hi_f向量中的每个元素与vfmin向量中的对应元素比较，取较大值，并与vfmax向量中的对应元素比较，取较小值，得到vacc_hi_f_clamped向量
        const float32x4_t vacc_hi_f_clamped =
            vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

        // 将vacc_lo_f_clamped向量中的元素与vfmagic向量中的对应元素相加，再将结果转换为整型向量，并与vimagic向量中的对应元素相减，得到vacc_lo向量
        vacc_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)), vimagic);

        // 将vacc_hi_f_clamped向量中的元素与vfmagic向量中的对应元素相加，再将结果转换为整型向量，并与vimagic向量中的对应元素相减，得到vacc_hi向量
        vacc_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)), vimagic);

        // 将vacc_lo和vacc_hi整型向量中的元素合并为一个int16x8_t类型的向量vacc
        const int16x8_t vacc =
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

        // 将vacc向量中的元素转换为无符号16位整数，并存储到vout向量中
        uint8x8_t vout = vqmovun_s16(vacc);
#endif

        // 如果c的最低两位为1，则执行以下代码块
        if (c & 4) {
          // 将vout向量的第0个元素转换为无符号32位整数，并存储到output指向的内存位置
          vst1_lane_u32(
              __builtin_assume_aligned(output, 1),
              vreinterpret_u32_u8(vout),
              0);
          // 将output指针移动4个字节
          output += 4;
          // 将vout向量进行位移，向左移动4个元素位置
          vout = vext_u8(vout, vout, 4);
        }
        // 如果c的第二位为1，则执行以下代码块
        if (c & 2) {
          // 将vout向量的第0个元素转换为无符号16位整数，并存储到output指向的内存位置
          vst1_lane_u16(
              __builtin_assume_aligned(output, 1),
              vreinterpret_u16_u8(vout),
              0);
          // 将output指针移动2个字节
          output += 2;
          // 将vout向量进行位移，向左移动2个元素位置
          vout = vext_u8(vout, vout, 2);
        }
        // 如果c的第一位为1，则执行以下代码块
        if (c & 1) {
          // 将vout向量的第0个元素转换为无符号8位整数，并存储到output指向的内存位置
          vst1_lane_u8(__builtin_assume_aligned(output, 1), vout, 0);
          // 将output指针移动1个字节
          output++;
        }
      }
    }

    // 将output指针转换为uintptr_t类型，加上output_increment，并再次转换为uint8_t指针，更新output指针位置
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);  // 当output_width不等于0时，重复执行循环
}
```