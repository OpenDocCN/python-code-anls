# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\fp32-neon.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <arm_neon.h>

#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_fp32__neon(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  // 确保 n 是 16 的倍数，以确保输入数据可以正确处理
  assert(n % 16 == 0);
  // 确保 scale 小于 1.0f，因为 scale 应该是小于输入范围的值
  assert(scale < 1.0f);
  // 确保 scale 大于等于 2^-32，这是 FP32 的最小正数值
  assert(scale >= 0x1.0p-32f);

  // 创建一个包含 scale 的 NEON 浮点数向量
  const float32x4_t vscale = vdupq_n_f32(scale);
#ifdef __aarch64__
  // 如果是 AArch64 架构，创建 NEON 向量以及标量
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  const uint8x16_t vqmin = vdupq_n_u8(qmin);
  const uint8x16_t vqmax = vdupq_n_u8(qmax);
#else
  // 如果是非 AArch64 架构，创建与 zero_point 相关的浮点 NEON 向量
  const float32x4_t vfmin = vdupq_n_f32(
      (float)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point));
  const float32x4_t vfmax = vdupq_n_f32(
      (float)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point));
  const float32x4_t vfmagic = vdupq_n_f32(12582912.0f);
  const int32x4_t vimagic =
      vdupq_n_s32(INT32_C(0x4B400000) - (int32_t)(uint32_t)zero_point);
#endif
  // 对每组 16 个输入数据进行处理
  for (; n != 0; n -= 16) {
    // 加载四组 int32 数据到 NEON 寄存器
    const int32x4_t x = vld1q_s32(input);
    const int32x4_t y = vld1q_s32(input + 4);
    const int32x4_t z = vld1q_s32(input + 8);
    const int32x4_t w = vld1q_s32(input + 12);
    input += 16;

    /*
     * 将 int32 输入转换为 FP32，并乘以 FP32 scale。
     * 这两个操作都涉及统计无偏舍入：
     * - 大的 int32 值无法精确表示为 FP32。ARM NEON 中的转换指令将其舍入为最接近的 FP32 值（向偶数舍入）。
     * - 两个 FP32 值的乘积通常不会精确地表示为一个 FP32 值，会被舍入为最接近的 FP32 值（向偶数舍入）。
     */
    const float32x4_t x_scaled = vmulq_f32(vcvtq_f32_s32(x), vscale);
    const float32x4_t y_scaled = vmulq_f32(vcvtq_f32_s32(y), vscale);
    const float32x4_t z_scaled = vmulq_f32(vcvtq_f32_s32(z), vscale);
    const float32x4_t w_scaled = vmulq_f32(vcvtq_f32_s32(w), vscale);

#ifdef __aarch64__
    /*
     * 利用 ARMv8 指令集中的“浮点转换为带舍入的有符号整数”指令。
     * 这是 ARMv8 中的一条指令（在 AArch64 架构中始终可用），在溢出时饱和结果。
     * 我们不需要专门考虑饱和结果，因为它们将在最后阶段被截断。
     */
    const int32x4_t x_rounded = vcvtnq_s32_f32(x_scaled);
    const int32x4_t y_rounded = vcvtnq_s32_f32(y_scaled);
    const int32x4_t z_rounded = vcvtnq_s32_f32(z_scaled);
    const int32x4_t w_rounded = vcvtnq_s32_f32(w_scaled);

    const int32x4_t x_rounded = vcvtnq_s32_f32(x_scaled);
    const int32x4_t y_rounded = vcvtnq_s32_f32(y_scaled);
    const int32x4_t z_rounded = vcvtnq_s32_f32(z_scaled);
    const int32x4_t w_rounded = vcvtnq_s32_f32(w_scaled);
#else
    /*
     * 在非 AArch64 架构下，使用魔数技巧来转换浮点数到整数。
     * 魔数和偏置是预先计算的常量，用于将浮点数结果转换为最接近的整数。
     * 在这里，使用的魔数是为了处理与 zero_point 相关的转换。
     */
    const float32x4_t x_rounded = vaddq_f32(
        vfmin, vmulq_f32(vfmagic, x_scaled));
    const float32x4_t y_rounded = vaddq_f32(
        vfmin, vmulq_f32(vfmagic, y_scaled));
    const float32x4_t z_rounded = vaddq_f32(
        vfmin, vmulq_f32(vfmagic, z_scaled));
    const float32x4_t w_rounded = vaddq_f32(
        vfmin, vmulq_f32(vfmagic, w_scaled));

    // 将浮点数舍入为整数，结果保存在 NEON 寄存器中
    const int32x4_t x_int = vcvtnq_s32_f32(x_rounded);
    const int32x4_t y_int = vcvtnq_s32_f32(y_rounded);
    const int32x4_t z_int = vcvtnq_s32_f32(z_rounded);
    const int32x4_t w_int = vcvtnq_s32_f32(w_rounded);

    // 调整整数值以匹配输出的量化范围（qmin 和 qmax）
    const int32x4_t x_biased = vaddq_s32(x_int, vimagic);
    const int32x4_t y_biased = vaddq_s32(y_int, vimagic);
    const int32x4_t z_biased = vaddq_s32(z_int, vimagic);
    const int32x4_t w_biased = vaddq_s32(w_int, vimagic);

    // 确保输出值在 qmin 和 qmax 之间，并转换为 uint8_t 类型
    const uint8x16_t x_clamped = vqmovun_s16(vqmovn_s32(x_biased));
    const uint8x16_t y_clamped = vqmovun_s16(vqmovn_s32(y_biased));
    const uint8x16_t z_clamped = vqmovun_s16(vqmovn_s32(z_biased));
    const uint8x16_t w_clamped = vqmovun_s16(vqmovn_s32(w_biased));

    // 将处理后的数据存储到输出数组中
    vst1q_u8(output, x_clamped);
    vst1q_u8(output + 16, y_clamped);
    vst1q_u8(output + 32, z_clamped);
    vst1q_u8(output + 48, w_clamped);
    output += 64;
#endif
  }
}


这样，每一行代码都被详细注释，解释了其在函数中的作用和实现细节。
    /*
     * 标准的 ARM NEON 最终序列：
     * - 将结果打包为 int16_t 并进行饱和处理
     * - 添加零点偏移
     * - 将结果打包为 uint8_t 并进行饱和处理
     * - 在 qmin 和 qmax 之间进行截断
     */
    const int16x8_t xy_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(x_rounded), y_rounded), vzero_point);
    // 将 x_rounded 和 y_rounded 各自转换为 int32_t，然后转换为 int16_t，并在高位拼接后与 vzero_point 相加，得到 xy_packed

    const int16x8_t zw_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(z_rounded), w_rounded), vzero_point);
    // 将 z_rounded 和 w_rounded 各自转换为 int32_t，然后转换为 int16_t，并在高位拼接后与 vzero_point 相加，得到 zw_packed

    const uint8x16_t xyzw_packed =
        vqmovun_high_s16(vqmovun_s16(xy_packed), zw_packed);
    // 将 xy_packed 和 zw_packed 分别转换为 uint16_t，然后在高位拼接后转换为 uint8_t，得到 xyzw_packed

    const uint8x16_t xyzw_clamped =
        vmaxq_u8(vminq_u8(xyzw_packed, vqmax), vqmin);
    // 将 xyzw_packed 与 vqmax 取最小值，再与 vqmin 取最大值，得到 xyzw_clamped

    vst1q_u8(output, xyzw_clamped);
    // 将 xyzw_clamped 存储到 output 指向的内存地址

    output += 16;
    // 更新 output 指针，移动到下一个 16 字节对齐的内存位置
/*
 * ARMv7 NEON offers only a floating-point to integer conversion instruction
 * with rounding towards zero. In lieu of conversion instruction with
 * rounding-to-nearest-even, we use a magic trick of adding a large number
 * (1.5 * 2**23) to scaled value to cause rounding to integer, and then
 * substracing this magic number as integer. This trick works only in a
 * limited range (absolute value of input must be less than 2**22), so
 * generally we have to clamp input to this range before using the magic.
 * However, clamping to any smaller range works just as well, and thus we
 * clamp to [qmin - zero point, qmax - zero point] range so that after we
 * add zero point to the result, it gets into target [qmin, qmax] range.
 */
const float32x4_t x_clamped = vminq_f32(vmaxq_f32(x_scaled, vfmin), vfmax);
const float32x4_t y_clamped = vminq_f32(vmaxq_f32(y_scaled, vfmin), vfmax);
const float32x4_t z_clamped = vminq_f32(vmaxq_f32(z_scaled, vfmin), vfmax);
const float32x4_t w_clamped = vminq_f32(vmaxq_f32(w_scaled, vfmin), vfmax);

/*
 * Conversion to integer using the "magic trick". Rounding is performed in
 * the output of addition operation, and result is rounded to nearest even
 * integer with ties to even.
 */
const int32x4_t x_biased = vsubq_s32(
    vreinterpretq_s32_f32(vaddq_f32(x_clamped, vfmagic)), vimagic);
const int32x4_t y_biased = vsubq_s32(
    vreinterpretq_s32_f32(vaddq_f32(y_clamped, vfmagic)), vimagic);
const int32x4_t z_biased = vsubq_s32(
    vreinterpretq_s32_f32(vaddq_f32(z_clamped, vfmagic)), vimagic);
const int32x4_t w_biased = vsubq_s32(
    vreinterpretq_s32_f32(vaddq_f32(w_clamped, vfmagic)), vimagic);

/*
 * Select low 8 bits of each 32-bit integer in the vectors for the output.
 * Since result is already clamped to [qmin, qmax] subrange of [0, 255],
 * saturation is not needed.
 */
const int16x8_t xy_packed =
    vcombine_s16(vmovn_s32(x_biased), vmovn_s32(y_biased));
const int16x8_t zw_packed =
    vcombine_s16(vmovn_s32(z_biased), vmovn_s32(w_biased));
const uint8x16_t xyzw_packed = vreinterpretq_u8_s8(
    vcombine_s8(vmovn_s16(xy_packed), vmovn_s16(zw_packed)));
    /*
     * AArch32 version:
     *   4x VCVT.F32.S32 Qd, Qm            // Convert 4 packed signed integers to 4 packed single-precision floats
     *   4x VMUL.F32 Qd, Qm, Qn            // Multiply 4 packed single-precision floats with another 4 packed single-precision floats
     *   4x VMIN.F32 Qd, Qm, Qn            // Compute minimum of 4 packed single-precision floats
     *   4x VMAX.F32 Qd, Qm, Qn            // Compute maximum of 4 packed single-precision floats
     *   4x VADD.F32 Qd, Qm, Qn            // Add 4 packed single-precision floats
     *   4x VSUB.S32 Qd, Qm, Qn            // Subtract 4 packed signed integers
     *   4x VMOVN.I32 Dd, Qm               // Narrow 4 packed signed integers to 2 packed signed integers (32-bit to 16-bit)
     *   2x VMOVN.I16 Dd, Qm               // Narrow 4 packed signed integers to 4 packed signed chars (16-bit to 8-bit)
     * ---------------------
     * 30 instructions total
     *
     * AArch64 version:
     *   4x SCVTF Vd.4S, Vn.4S            // Convert 4 packed single-precision floats to 4 packed integers
     *   4x FMUL Vd.4S, Vn.4S, Vm.4S      // Multiply 4 packed single-precision floats with another 4 packed single-precision floats
     *   4x FCVTNS Vd.4S, Vn.4S           // Convert 4 packed single-precision floats to 4 packed signed integers
     *   2x SQXTN Vd.4H, Vn.4S            // Narrow 4 packed single-precision floats to 4 packed signed shorts (16-bit)
     *   2x SQXTN2 Vd.8H, Vn.4S           // Narrow and shift 4 packed single-precision floats to 8 packed signed shorts (16-bit)
     *   2x ADD Vd.8H, Vn.8H, Vm.8H       // Add 8 packed signed shorts (16-bit)
     *   1x SQXTUN Vd.8B, Vn.8H           // Narrow and saturate 8 packed signed shorts to 8 packed unsigned chars (8-bit)
     *   1x SQXTUN2 Vd.16B, Vn.8H         // Narrow and shift 8 packed signed shorts to 16 packed unsigned chars (8-bit)
     *   1x UMIN Vd.16B, Vn.16B, Vm.16B   // Compute minimum of 16 packed unsigned chars
     *   1x UMAX Vd.16B, Vn.16B, Vm.16B   // Compute maximum of 16 packed unsigned chars
     * ---------------------
     * 22 instructions total
     */
    
    // Store 16 unsigned 8-bit values from xyzw_packed into memory at location 'output' and advance 'output' by 16 bytes
    vst1q_u8(output, xyzw_packed);
    output += 16;
#endif
  }
}


注释：


# 终止条件：此处是预处理器指令的结束标记，用于结束条件编译部分的代码块
#endif
  }
}
# 结束当前的函数定义块
```