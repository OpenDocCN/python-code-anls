# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gavgpool\up8x7-neon.c`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录下的LICENSE文件中使用BSD风格许可证。
 */

#include <assert.h>

#include <arm_neon.h>

#include <qnnpack/q8gavgpool.h>

void pytorch_q8gavgpool_ukernel_up8x7__neon(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[restrict static 1]) {
  assert(m >= 1);  // 确保m至少为1
  assert(m <= 7);  // 确保m最多为7
  assert(n >= 8);  // 确保n至少为8

  const uint8_t* i0 = input;            // 指向输入的首元素
  const uint8_t* i1 = i0 + input_stride;  // 指向第二行输入的首元素
  if (m < 2) {
    i1 = zero;  // 如果m小于2，则使用zero作为第二行输入
  }
  const uint8_t* i2 = i1 + input_stride;  // 指向第三行输入的首元素
  if (m <= 2) {
    i2 = zero;  // 如果m小于或等于2，则使用zero作为第三行输入
  }
  const uint8_t* i3 = i2 + input_stride;  // 指向第四行输入的首元素
  if (m < 4) {
    i3 = zero;  // 如果m小于4，则使用zero作为第四行输入
  }
  const uint8_t* i4 = i3 + input_stride;  // 指向第五行输入的首元素
  if (m <= 4) {
    i4 = zero;  // 如果m小于或等于4，则使用zero作为第五行输入
  }
  const uint8_t* i5 = i4 + input_stride;  // 指向第六行输入的首元素
  if (m < 6) {
    i5 = zero;  // 如果m小于6，则使用zero作为第六行输入
  }
  const uint8_t* i6 = i5 + input_stride;  // 指向第七行输入的首元素
  if (m <= 6) {
    i6 = zero;  // 如果m小于或等于6，则使用zero作为第七行输入
  }
  const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);  // 加载偏置向量
  const float32x4_t vscale = vdupq_n_f32(quantization_params->neon.scale);  // 加载缩放因子

#if defined(__aarch64__)
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);  // 加载输出零点向量
  const uint8x8_t voutput_min =
      vld1_dup_u8(&quantization_params->neon.output_min);  // 加载输出最小值向量
  const uint8x8_t voutput_max =
      vld1_dup_u8(&quantization_params->neon.output_max);  // 加载输出最大值向量
#else
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);  // 加载vfmin向量
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);  // 加载vfmax向量
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);  // 加载vfmagic向量
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);  // 加载vimagic向量
#endif

  do {
    const uint8x8_t vi0 = vld1_u8(i0);  // 加载第一行输入
    i0 += 8;
    const uint8x8_t vi1 = vld1_u8(i1);  // 加载第二行输入
    i1 += 8;
    const uint8x8_t vi2 = vld1_u8(i2);  // 加载第三行输入
    i2 += 8;
    const uint8x8_t vi3 = vld1_u8(i3);  // 加载第四行输入
    i3 += 8;
    const uint8x8_t vi4 = vld1_u8(i4);  // 加载第五行输入
    i4 += 8;
    const uint8x8_t vi5 = vld1_u8(i5);  // 加载第六行输入
    i5 += 8;
    const uint8x8_t vi6 = vld1_u8(i6);  // 加载第七行输入
    i6 += 8;

    const int16x8_t vsum016 =
        vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0, vi1), vi6));  // 计算第一、二、七行的累加和
    const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));  // 计算第三、四行的累加和
    const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));  // 计算第五、六行的累加和

    int32x4_t vacc_lo = vaddw_s16(vbias, vget_low_s16(vsum23));  // 计算累加和的低位
    int32x4_t vacc_hi = vaddw_s16(vbias, vget_high_s16(vsum23));  // 计算累加和的高位
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum45));  // 继续累加低位
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum45));  // 继续累加高位
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum016));  // 继续累加低位
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum016));  // 继续累加高位

    float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);  // 整数累加和转换为浮点数
    float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);  // 同上
    # 使用 NEON 指令对四个单精度浮点数向量 `vacc_lo_f` 中的每个元素乘以标量 `vscale`
    vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
    # 使用 NEON 指令对四个单精度浮点数向量 `vacc_hi_f` 中的每个元素乘以标量 `vscale`
    vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);
#if defined(__aarch64__)
    // 如果定义了 __aarch64__ 宏，则执行以下代码块
    vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
    // 将浮点数向下转换为整数，并存储在 vacc_lo 中
    vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
    // 将浮点数向下转换为整数，并存储在 vacc_hi 中
    const int16x8_t vacc = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
    // 将 vacc_lo 和 vacc_hi 转换为 16 位整数，然后进行饱和加法操作，并加上 voutput_zero_point
    uint8x8_t vout = vqmovun_s16(vacc);
    // 将 16 位整数转换为无符号 8 位整数
    vout = vmax_u8(vout, voutput_min);
    // 取 vout 和 voutput_min 之间的最大值
    vout = vmin_u8(vout, voutput_max);
    // 取 vout 和 voutput_max 之间的最小值
#else
    // 如果未定义 __aarch64__ 宏，则执行以下代码块
    vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
    // 将 vacc_lo_f 限制在 vfmin 和 vfmax 之间，并将结果存储在 vacc_lo_f 中
    vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
    // 将 vacc_hi_f 限制在 vfmin 和 vfmax 之间，并将结果存储在 vacc_hi_f 中

    vacc_lo = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
    // 将 vacc_lo_f 加上 vfmagic 后转换为整数，然后减去 vimagic，并将结果存储在 vacc_lo 中
    vacc_hi = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
    // 将 vacc_hi_f 加上 vfmagic 后转换为整数，然后减去 vimagic，并将结果存储在 vacc_hi 中

    const int16x8_t vacc =
        vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    // 将 vacc_lo 和 vacc_hi 转换为 16 位整数，并将结果合并为一个 128 位寄存器中的 16 位整数
    uint8x8_t vout = vqmovun_s16(vacc);
    // 将 16 位整数转换为无符号 8 位整数
#endif

    vst1_u8(output, vout);
    // 将 vout 存储到 output 指向的内存地址
    output += 8;
    // 将 output 指针向前移动 8 个字节

    n -= 8;
    // 将 n 减去 8
  } while (n >= 8);
  // 如果 n 大于等于 8，则继续循环

  if (n != 0) {
    // 如果 n 不等于 0，则执行以下代码块
    const size_t address_increment = n - 8;
    // 计算地址增量为 n 减去 8
    i0 = (const uint8_t*)((uintptr_t)i0 + address_increment);
    // 将 i0 指针向前移动 address_increment 个字节
    i1 = (const uint8_t*)((uintptr_t)i1 + address_increment);
    // 将 i1 指针向前移动 address_increment 个字节
    i2 = (const uint8_t*)((uintptr_t)i2 + address_increment);
    // 将 i2 指针向前移动 address_increment 个字节
    i3 = (const uint8_t*)((uintptr_t)i3 + address_increment);
    // 将 i3 指针向前移动 address_increment 个字节
    i4 = (const uint8_t*)((uintptr_t)i4 + address_increment);
    // 将 i4 指针向前移动 address_increment 个字节
    i5 = (const uint8_t*)((uintptr_t)i5 + address_increment);
    // 将 i5 指针向前移动 address_increment 个字节
    i6 = (const uint8_t*)((uintptr_t)i6 + address_increment);
    // 将 i6 指针向前移动 address_increment 个字节

    const int64x1_t vshift = vmov_n_s64(8 * address_increment);
    // 创建一个移位量为 8 * address_increment 的 64 位整数

    const uint8x8_t vi0 =
        vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i0)), vshift));
    // 将 i0 指针处的 8 个字节加载到 vi0，并左移 vshift 位
    const uint8x8_t vi1 =
        vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i1)), vshift));
    // 将 i1 指针处的 8 个字节加载到 vi1，并左移 vshift 位
    const uint8x8_t vi2 =
        vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i2)), vshift));
    // 将 i2 指针处的 8 个字节加载到 vi2，并左移 vshift 位
    const uint8x8_t vi3 =
        vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i3)), vshift));
    // 将 i3 指针处的 8 个字节加载到 vi3，并左移 vshift 位
    const uint8x8_t vi4 =
        vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i4)), vshift));
    // 将 i4 指针处的 8 个字节加载到 vi4，并左移 vshift 位
    const uint8x8_t vi5 =
        vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i5)), vshift));
    // 将 i5 指针处的 8 个字节加载到 vi5，并左移 vshift 位
    const uint8x8_t vi6 =
        vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(i6)), vshift));
    // 将 i6 指针处的 8 个字节加载到 vi6，并左移 vshift 位

    const int16x8_t vsum016 =
        vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0, vi1), vi6));
    // 将 vi0、vi1 和 vi6 中的无符号 8 位整数加到 16 位整数中
    const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
    // 将 vi2 和 vi3 中的无符号 8 位整数加到 16 位整数中
    const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));
    // 将 vi4 和 vi5 中的无符号 8 位整数加到 16 位整数中

    int32x4_t vacc_lo = vaddw_s16(vbias, vget_low_s16(vsum23));
    // 将 vbias 和 vsum23 的低位 4 个 16 位整数相加，并存储在 vacc_lo 中
    int32x4_t vacc_hi = vaddw_s16(vbias, vget_high_s16(vsum23));
    // 将 vbias 和 vsum23 的高位 4 个 16 位整数相加，并存储在 vacc_hi 中
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum45));
    // 将 vacc_lo 和 vsum45 的低位 4 个 16 位整数相加，并存储在 vacc_lo 中
    vacc_hi = vaddw_s16(vacc
#if defined(__aarch64__)
    // 如果目标平台是 __aarch64__

    // 将浮点累加结果转换为整型累加结果
    vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
    vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

    // 对整型累加结果进行饱和加法运算
    const int16x8_t vacc = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);

    // 将整型累加结果转换为无符号8位整数，并进行上下限制
    uint8x8_t vout = vqmovun_s16(vacc);
    vout = vmax_u8(vout, voutput_min);
    vout = vmin_u8(vout, voutput_max);
#else
    // 如果不是 __aarch64__ 平台

    // 将浮点累加结果限制在指定范围内
    vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
    vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

    // 将浮点累加结果转换为整型累加结果，并减去魔数
    vacc_lo = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
    vacc_hi = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);

    // 合并成一个整型累加结果向量
    const int16x8_t vacc =
        vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

    // 将整型累加结果转换为无符号8位整数
    uint8x8_t vout = vqmovun_s16(vacc);
#endif

// 根据 n 的值分别处理输出数据
if (n & 4) {
    // 如果 n 是 4 的倍数，将 vout 转换为无符号32位整数，并存储到 output
    vst1_lane_u32(
        __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
    output += 4;
    // 将 vout 向量右移4个元素
    vout = vext_u8(vout, vout, 4);
}
if (n & 2) {
    // 如果 n 是 2 的倍数，将 vout 转换为无符号16位整数，并存储到 output
    vst1_lane_u16(
        __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
    output += 2;
    // 将 vout 向量右移2个元素
    vout = vext_u8(vout, vout, 2);
}
if (n & 1) {
    // 如果 n 是奇数，将 vout 的第一个元素存储到 output
    vst1_lane_u8(output, vout, 0);
}
}
```