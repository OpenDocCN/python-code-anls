# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8avgpool\up8x9-neon.c`

```py
/*
 * 版权所有（C）Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码采用 BSD 风格许可证，可以在源代码根目录下的 LICENSE 文件中找到。
 */

#include <assert.h>

#include <arm_neon.h>

#include <qnnpack/q8avgpool.h>

// 定义一个 NEON 池化函数，用于处理 8x9 大小的池化操作
void pytorch_q8avgpool_ukernel_up8x9__neon(
    size_t n,                                            // n 是处理元素数量
    size_t ks,                                           // ks 是池化窗口的尺寸
    size_t kc,                                           // kc 是输入通道数
    const uint8_t** input,                               // 输入数据的指针数组
    const uint8_t* zero,                                 // 补齐用的零数据
    uint8_t* output,                                     // 输出数据指针
    size_t input_increment,                              // 输入数据增量
    size_t output_increment,                             // 输出数据增量
    const union pytorch_qnnp_avgpool_quantization_params // 量化参数结构体
        quantization_params[restrict static 1]) {

  // 断言，确保 n 不为 0
  assert(n != 0);
  // 断言，确保 ks 不超过 9
  assert(ks <= 9);
  // 断言，确保 kc 至少为 8
  assert(kc >= 8);

  // 加载偏置值到 NEON 寄存器
  const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);
  // 加载缩放因子到 NEON 寄存器
  const float32x4_t vscale = vdupq_n_f32(quantization_params->neon.scale);

  // 根据不同的架构加载输出零点、输出最小值和输出最大值到 NEON 寄存器
#if defined(__aarch64__)
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  const uint8x8_t voutput_min =
      vld1_dup_u8(&quantization_params->neon.output_min);
  const uint8x8_t voutput_max =
      vld1_dup_u8(&quantization_params->neon.output_max);
#else
  // 对于非 AArch64 架构，加载额外的量化参数到 NEON 寄存器
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
#endif

  // 开始处理池化操作，循环处理每个元素
  do {
    // 每次循环加载输入数据的指针
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];

    // 更新输入数据指针，以便下一次循环处理
    input = (const uint8_t**)((uintptr_t)input + input_increment);

    // 根据池化窗口的大小，对小于当前窗口大小的输入指针进行补齐
    if (ks < 2) {
      i1 = zero;
    }
    if (ks <= 2) {
      i2 = zero;
    }
    if (ks < 4) {
      i3 = zero;
    }
    if (ks <= 4) {
      i4 = zero;
    }
    if (ks < 6) {
      i5 = zero;
    }
    if (ks <= 6) {
      i6 = zero;
    }
    if (ks < 8) {
      i7 = zero;
    }
    if (ks <= 8) {
      i8 = zero;
    }

    // 开始处理每个通道的数据
    size_t k = kc;
    // 当 k 大于等于 8 时进入循环
    while (k >= 8) {
        // 加载连续的8个 uint8_t 类型的数据到寄存器 vi0-vi8，并更新指针 i0-i8
        const uint8x8_t vi0 = vld1_u8(i0);
        i0 += 8;
        const uint8x8_t vi1 = vld1_u8(i1);
        i1 += 8;
        const uint8x8_t vi2 = vld1_u8(i2);
        i2 += 8;
        const uint8x8_t vi3 = vld1_u8(i3);
        i3 += 8;
        const uint8x8_t vi4 = vld1_u8(i4);
        i4 += 8;
        const uint8x8_t vi5 = vld1_u8(i5);
        i5 += 8;
        const uint8x8_t vi6 = vld1_u8(i6);
        i6 += 8;
        const uint8x8_t vi7 = vld1_u8(i7);
        i7 += 8;
        const uint8x8_t vi8 = vld1_u8(i8);
        i8 += 8;

        // 计算各组元素的无符号 16 位整数累加和
        const uint16x8_t vsum018 = vaddw_u8(vaddl_u8(vi0, vi1), vi8);
        const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
        const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);
        const uint16x8_t vsum67 = vaddl_u8(vi6, vi7);

        // 合并累加和
        const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);
        const uint16x8_t vsum01678 = vaddq_u16(vsum018, vsum67);
        const uint16x8_t vsum = vaddq_u16(vsum2345, vsum01678);

        // 将累加和转换为有符号 32 位整数并加上偏置向量
        int32x4_t vacc_lo =
            vaddw_s16(vbias, vreinterpret_s16_u16(vget_low_u16(vsum)));
        int32x4_t vacc_hi =
            vaddw_s16(vbias, vreinterpret_s16_u16(vget_high_u16(vsum)));

        // 将有符号 32 位整数向量转换为单精度浮点数向量，并乘以比例因子
        float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);
        float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);

        vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
        vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);
#if defined(__aarch64__)
      // 将累加和转换为整数向量，截断为32位整数
      vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
      vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
      // 对累加和进行饱和加法操作，加上输出的零点偏移
      const int16x8_t vacc = vqaddq_s16(
          vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
      // 将16位整数向量转换为8位无符号整数向量
      uint8x8_t vout = vqmovun_s16(vacc);
      // 对输出向量进行上下限控制
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);
#else
      // 对累加和进行浮点数向量的上下限控制
      vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
      vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

      // 将浮点数累加和转换为整数向量，加上魔数后减去魔数偏移
      vacc_lo = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
      vacc_hi = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
      // 将两个32位整数向量合并为一个16位整数向量
      const int16x8_t vacc =
          vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
      // 将16位整数向量转换为8位无符号整数向量
      uint8x8_t vout = vqmovun_s16(vacc);
#endif

      // 将8位无符号整数向量写入输出数组
      vst1_u8(output, vout);
      // 更新输出数组指针，移动到下一个8字节位置
      output += 8;

      // 更新循环计数器，每次处理8个元素
      k -= 8;
    }
    // 如果 k 不等于 0，则执行以下操作
    if (k != 0) {
      // 计算地址增量，用于更新指针
      const size_t address_increment = k - 8;
      i0 = (const uint8_t*)((uintptr_t)i0 + address_increment);
      i1 = (const uint8_t*)((uintptr_t)i1 + address_increment);
      i2 = (const uint8_t*)((uintptr_t)i2 + address_increment);
      i3 = (const uint8_t*)((uintptr_t)i3 + address_increment);
      i4 = (const uint8_t*)((uintptr_t)i4 + address_increment);
      i5 = (const uint8_t*)((uintptr_t)i5 + address_increment);
      i6 = (const uint8_t*)((uintptr_t)i6 + address_increment);
      i7 = (const uint8_t*)((uintptr_t)i7 + address_increment);
      i8 = (const uint8_t*)((uintptr_t)i8 + address_increment);

      // 创建一个 64 位整数向量，用于左移地址增量倍数的位数
      const int64x1_t vshift = vmov_n_s64(8 * address_increment);

      // 加载并左移存储在指针 i0-i8 中的数据，并转换为 8 位无符号整数向量
      const uint8x8_t vi0 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i0)), vshift));
      const uint8x8_t vi1 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i1)), vshift));
      const uint8x8_t vi2 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i2)), vshift));
      const uint8x8_t vi3 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i3)), vshift));
      const uint8x8_t vi4 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i4)), vshift));
      const uint8x8_t vi5 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i5)), vshift));
      const uint8x8_t vi6 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i6)), vshift));
      const uint8x8_t vi7 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i7)), vshift));
      const uint8x8_t vi8 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i8)), vshift));

      // 对 vi0-vi8 中的数据进行加法运算，形成 16 位无符号整数向量
      const uint16x8_t vsum018 = vaddw_u8(vaddl_u8(vi0, vi1), vi8);
      const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
      const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);
      const uint16x8_t vsum67 = vaddl_u8(vi6, vi7);

      // 合并部分结果以形成更大的 16 位无符号整数向量
      const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);
      const uint16x8_t vsum01678 = vaddq_u16(vsum018, vsum67);
      const uint16x8_t vsum = vaddq_u16(vsum2345, vsum01678);

      // 对 vsum 中的数据进行加法运算，使用偏置向量 vbias
      int32x4_t vacc_lo =
          vaddw_s16(vbias, vreinterpret_s16_u16(vget_low_u16(vsum)));
      int32x4_t vacc_hi =
          vaddw_s16(vbias, vreinterpret_s16_u16(vget_high_u16(vsum)));

      // 将整数向量转换为单精度浮点数向量
      float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);
      float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);

      // 使用缩放因子对浮点数向量进行乘法运算
      vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
      vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);
#if defined(__aarch64__)
      // 如果目标平台为 ARM64 架构，则执行以下代码块
      vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
      vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
      // 将浮点数向下转换为整型，并进行饱和处理，加上输出零点
      const int16x8_t vacc = vqaddq_s16(
          vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
      // 将整型向无符号短整型转换，并进行饱和处理
      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);  // 对结果向量进行最大值限制
      vout = vmin_u8(vout, voutput_max);  // 对结果向量进行最小值限制
#else
      // 如果目标平台不是 ARM64 架构，则执行以下代码块
      vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
      vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
      
      // 对浮点数进行一些数学操作，加减和类型转换
      vacc_lo = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
      vacc_hi = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
      
      // 将两个 4x32 位整数向量转换为一个 8x16 位整数向量
      const int16x8_t vacc =
          vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
      // 将整型向无符号短整型转换
      uint8x8_t vout = vqmovun_s16(vacc);
#endif

      if (k & 4) {
        // 如果 k 的最后两位为 1，则执行以下代码块
        vst1_lane_u32(
            __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
        output += 4;  // 输出指针向后移动 4 字节
        vout = vext_u8(vout, vout, 4);  // 将向量中的元素进行循环移位
      }
      if (k & 2) {
        // 如果 k 的倒数第二位为 1，则执行以下代码块
        vst1_lane_u16(
            __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
        output += 2;  // 输出指针向后移动 2 字节
        vout = vext_u8(vout, vout, 2);  // 将向量中的元素进行循环移位
      }
      if (k & 1) {
        // 如果 k 的最后一位为 1，则执行以下代码块
        vst1_lane_u8(output, vout, 0);
        output += 1;  // 输出指针向后移动 1 字节
      }
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);  // 循环执行直到 n 不等于 0
}
```