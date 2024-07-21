# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\mp8x25-neon.c`

```py
/*
 * 包含必要的头文件，其中包括 NEON SIMD 指令集的引入
 */
#include <arm_neon.h>

/*
 * QNNPACK 库中的深度可分离量化卷积函数的 NEON 实现
 * 使用 8 位输入和 25x8 过滤器的输入排列
 */
void pytorch_q8dwconv_ukernel_mp8x25__neon(
    size_t channels,                                     // 输入通道数
    size_t output_width,                                 // 输出宽度
    const uint8_t** input,                               // 输入数据的指针数组
    const void* weights,                                 // 卷积核权重
    int32_t* outacc32,                                   // 输出累加器数组
    uint8_t* output,                                     // 输出数据
    size_t input_stride,                                 // 输入数据跨度
    size_t output_increment,                             // 输出增量
    const union pytorch_qnnp_conv_quantization_params    // 量化参数结构体
        quantization_params[restrict static 1]) {

  // 加载输入量化参数：输入零点
  const uint8x8_t vinput_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);

  // 创建卷积核量化参数：卷积核零点
  const uint8x8_t vkernel_zero_point =
      vdup_n_u8(quantization_params->neon.kernel_zero_points[0]);

  // 加载重新量化缩放因子
  const float32x4_t requantization_scale_v =
      vdupq_n_f32(quantization_params->neon.requantization_scales[0]);

#ifdef __aarch64__
  // 加载输出量化参数：输出零点
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  // 加载输出量化参数：输出最小值
  const uint8x8_t voutput_min = vld1_dup_u8(&quantization_params->neon.output_min);
  // 加载输出量化参数：输出最大值
  const uint8x8_t voutput_max = vld1_dup_u8(&quantization_params->neon.output_max);
#else
  // 加载输出量化参数：vmin 的值
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  // 加载输出量化参数：vmax 的值
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  // 加载输出量化参数：vfmagic 的值
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  // 加载输出量化参数：vimagic 的值
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
#endif

  do {
    // 指向输出起始位置的指针
    uint8_t* output_start = output;
    // 输出累加器的指针
    int32_t* outacc = outacc32;
    // 权重指针
    const void* w = weights;

    // 循环体，处理每一个输出宽度
    // （具体操作暂未提供，应在此处继续填充代码和注释）
    }
    }

#ifdef __aarch64__
        // 转换累加器的低位和高位浮点值为整数值
        vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
        vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

        // 将累加器值与输出零点相加，并且饱和至 16 位有符号整数
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);

        // 将 16 位整数转换为 8 位无符号整数，并且进行上下限截断
        uint8x8_t vout = vqmovun_s16(vacc);
        vout = vmax_u8(vout, voutput_min);
        vout = vmin_u8(vout, voutput_max);
#else
        // 将累加器的低位和高位浮点值限制在 vfmin 和 vfmax 之间
        const float32x4_t vacc_lo_f_clamped =
            vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
        const float32x4_t vacc_hi_f_clamped =
            vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

        // 将浮点值重新量化为整数值，并且与 vimagic 相减
        vacc_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)), vimagic);
        vacc_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)), vimagic);

        // 合并低位和高位的 16 位整数值，并将其转换为 8 位无符号整数
        const int16x8_t vacc =
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

        // 将 16 位整数转换为 8 位无符号整数
        uint8x8_t vout = vqmovun_s16(vacc);
#endif

        // 对输出值进行饱和至 voutput_min 和 voutput_max 之间
        vout = vmax_u8(vout, voutput_min);
        vout = vmin_u8(vout, voutput_max);
#else
        // 计算并限制低位累加向量的浮点值
        const float32x4_t vacc_lo_f_clamped =
            vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
        // 计算并限制高位累加向量的浮点值
        const float32x4_t vacc_hi_f_clamped =
            vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
        // 将限制后的浮点累加值加上魔数并转换为整型，再减去魔数得到结果
        vacc_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)), vimagic);
        vacc_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)), vimagic);
        // 将低位和高位累加结果转换为短整型，组合成一个短整型向量
        const int16x8_t vacc =
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

        // 将短整型向量转换为无符号字节向量，用于存储输出
        uint8x8_t vout = vqmovun_s16(vacc);
#endif

        // 处理每次循环中向量长度为4的情况
        if (c & 4) {
          // 将第一个4字节存储到输出中
          vst1_lane_u32(
              __builtin_assume_aligned(output, 1),
              vreinterpret_u32_u8(vout),
              0);
          // 更新输出指针位置
          output += 4;
          // 将字节向量右移4位，准备处理下一个长度为4的情况
          vout = vext_u8(vout, vout, 4);
        }
        // 处理每次循环中向量长度为2的情况
        if (c & 2) {
          // 将前两个字节存储到输出中
          vst1_lane_u16(
              __builtin_assume_aligned(output, 1),
              vreinterpret_u16_u8(vout),
              0);
          // 更新输出指针位置
          output += 2;
          // 将字节向量右移2位，准备处理下一个长度为2的情况
          vout = vext_u8(vout, vout, 2);
        }
        // 处理每次循环中向量长度为1的情况
        if (c & 1) {
          // 将第一个字节存储到输出中
          vst1_lane_u8(__builtin_assume_aligned(output, 1), vout, 0);
          // 更新输出指针位置
          output++;
        }
      }
    }

    // 根据输出宽度增加输出指针，以便处理下一行数据
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);
}
```