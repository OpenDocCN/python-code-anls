# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\mp8x27-neon.c`

```
/**
 * 核心函数，用于执行带有27个权重的深度卷积（例如3x3x3大小的卷积核）。
 * 总体策略是：
 * 对于每一行的输出：
 *   对于该行中的每个输出像素：
 *     对于27个权重中的每一组9个权重（例如3x3x3卷积核的一个yz切片）：
 *       对于每组8个通道：
 *         从间接缓冲区和权重中加载输入像素值，
 *         进行乘法和加法运算，并在临时缓冲区中跟踪这些乘积和偏置的累计总和
 *       执行重新量化以获得最终输出
 *     将间接缓冲区移动到该行中的下一个像素
 *   将间接缓冲区移动到下一行
 */
void pytorch_q8dwconv_ukernel_mp8x27__neon(
    size_t channels,                                // 输入通道数
    size_t output_height,                           // 输出图像高度
    size_t output_width,                            // 输出图像宽度
    const uint8_t** input,                          // 输入数据的指针数组
    const void* weights,                            // 卷积核权重数据
    int32_t* outacc32,                              // 输出累加器（32位整数）指针
    uint8_t* output,                                // 输出数据指针
    size_t input_row_stride,                        // 间接缓冲区中行之间的跨度
    size_t input_col_stride,                        // 间接缓冲区中列之间的跨度
    size_t output_increment,                        // 输出缓冲区中像素之间的填充
    const union pytorch_qnnp_conv_quantization_params // 量化参数结构体
        quantization_params[restrict static 1]) {

  // 加载输入的零点值到向量寄存器
  const uint8x8_t vinput_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);

#ifdef __aarch64__
  // 加载输出的零点值、最小值和最大值到向量寄存器（ARM64架构）
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&quantization_params->neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&quantization_params->neon.output_max);
#else
  // 加载浮点数类型的最小值、最大值和量化相关的魔数到向量寄存器（ARM32架构）
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
#endif

  // 循环遍历每一行的输出
  for (size_t output_y = 0; output_y < output_height; output_y++) {
    const uint8_t** input_row_start = input;
#ifdef __aarch64__
          // 对累加器结果进行饱和运算，并添加输出的零点值（ARM64架构）
          vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
          vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

          const int16x8_t vacc = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi),
              voutput_zero_point);

          // 将饱和后的16位整数转换为无符号8位整数，并进行最大值最小值限制
          uint8x8_t vout = vqmovun_s16(vacc);
          vout = vmax_u8(vout, voutput_min);
          vout = vmin_u8(vout, voutput_max);
#else
          // 计算低部和高部的浮点值，并对其进行上下限约束
          const float32x4_t vacc_lo_f_clamped =
              vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
          const float32x4_t vacc_hi_f_clamped =
              vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
          // 将浮点数转换为整数，加上魔数，得到量化后的值
          vacc_lo = vsubq_s32(
              vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)),
              vimagic);
          vacc_hi = vsubq_s32(
              vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)),
              vimagic);
          // 合并成一个16位整数向量
          const int16x8_t vacc =
              vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

          // 将16位整数向量转换为8位无符号整数向量
          uint8x8_t vout = vqmovun_s16(vacc);
#ifdef __aarch64__
          // 在64位ARM架构上，重新量化低部和高部的浮点数向量
          vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
          vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

          // 将重新量化后的向量与输出零点相加，得到最终的量化结果
          const int16x8_t vacc = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi),
              voutput_zero_point);

          // 将16位整数向量转换为8位无符号整数向量，并进行上下限限制
          uint8x8_t vout = vqmovun_s16(vacc);
          vout = vmax_u8(vout, voutput_min);
          vout = vmin_u8(vout, voutput_max);
#else
          // 计算低部和高部的浮点值，并对其进行上下限约束
          const float32x4_t vacc_lo_f_clamped =
              vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
          const float32x4_t vacc_hi_f_clamped =
              vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
          // 将浮点数转换为整数，加上魔数，得到量化后的值
          vacc_lo = vsubq_s32(
              vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)),
              vimagic);
          vacc_hi = vsubq_s32(
              vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)),
              vimagic);
          // 合并成一个16位整数向量
          const int16x8_t vacc =
              vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

          // 将16位整数向量转换为8位无符号整数向量
          uint8x8_t vout = vqmovun_s16(vacc);
#endif

          // 根据标志位c，依次存储结果向量的值到输出中
          if (c & 4) {
            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u8(vout),
                0);
            output += 4;
            vout = vext_u8(vout, vout, 4);
          }
          if (c & 2) {
            vst1_lane_u16(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u16_u8(vout),
                0);
            output += 2;
            vout = vext_u8(vout, vout, 2);
          }
          if (c & 1) {
            vst1_lane_u8(__builtin_assume_aligned(output, 1), vout, 0);
            output++;
          }
        }
      }

      // 更新输出指针，跳到下一个输出行的起始位置
      output = (uint8_t*)((uintptr_t)output + output_increment);
    }
    // 更新输入指针，跳到下一个输入行的起始位置
    input = (const uint8_t**)((uintptr_t)input_row_start + input_row_stride);
  }
}
```