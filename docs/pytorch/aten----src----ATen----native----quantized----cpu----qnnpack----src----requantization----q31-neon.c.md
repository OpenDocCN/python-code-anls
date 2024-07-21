# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\q31-neon.c`

```
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

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_q31__neon(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 16 == 0);  // 确保 n 是 16 的倍数，因为每次处理 16 个元素
  assert(scale < 1.0f);  // 确保 scale 小于 1.0
  assert(scale >= 0x1.0p-32f);  // 确保 scale 不小于 2^(-32)

  /* Compute requantization parameters */
  const uint32_t scale_bits = fp32_to_bits(scale);  // 将浮点数 scale 转换为整数表示
  /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
  const int32_t multiplier = (int32_t)(
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);  // 计算乘数 multiplier
  assert(multiplier >= INT32_C(0x40000000));  // 确保 multiplier 大于等于 2^30
  assert(multiplier <= INT32_C(0x7FFFFF80));  // 确保 multiplier 小于等于 2^31 - 128

  /* Shift is in [0, 31] range */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);  // 计算右移位数 shift
  assert(shift >= 0);  // 确保 shift 大于等于 0
  assert(shift < 32);  // 确保 shift 小于 32

  const int32x4_t vmultiplier = vdupq_n_s32(multiplier);  // 创建乘数的 NEON 向量
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);  // 创建零点的 NEON 向量
  const int32x4_t vshift = vdupq_n_s32(-shift);  // 创建右移量的 NEON 向量
  const int32x4_t vshift_eq_0_mask =
      vreinterpretq_s32_u32(vceqq_s32(vshift, vmovq_n_s32(0)));  // 创建用于掩码的 NEON 向量，用于处理右移量为 0 的情况
  const uint8x16_t vqmin = vdupq_n_u8(qmin);  // 创建 qmin 的 NEON 向量
  const uint8x16_t vqmax = vdupq_n_u8(qmax);  // 创建 qmax 的 NEON 向量
  for (; n != 0; n -= 16) {  // 循环处理每组 16 个元素
    const int32x4_t x = vld1q_s32(input);  // 加载第一组 4 个 32 位整数
    const int32x4_t y = vld1q_s32(input + 4);  // 加载第二组 4 个 32 位整数
    const int32x4_t z = vld1q_s32(input + 8);  // 加载第三组 4 个 32 位整数
    const int32x4_t w = vld1q_s32(input + 12);  // 加载第四组 4 个 32 位整数
    input += 16;  // 更新输入指针到下一组数据

    /*
     * Directly use VQRDMULH/SQRDMULH instruction for Q31 multiplication with
     * rounding. Although these instruction saturate out-of-range outputs, we
     * never hit this case in requantization.
     */
    const int32x4_t x_product = vqrdmulhq_s32(x, vmultiplier);  // 使用 NEON 指令进行带舍入的 Q31 乘法
    const int32x4_t y_product = vqrdmulhq_s32(y, vmultiplier);  // 对第二组数据进行相同操作
    const int32x4_t z_product = vqrdmulhq_s32(z, vmultiplier);  // 对第三组数据进行相同操作
    const int32x4_t w_product = vqrdmulhq_s32(w, vmultiplier);  // 对第四组数据进行相同操作

    /*
     * Shift the 32-bit product right with rounding.
     * Rounding is performed towards closest integer, with midpoints rounded up
     * (same as away from zero).
     *
     * We leverage the "right shift with rounding" instruction (VRSHL.S32 on ARM
     * NEON, SRSHL in ARM64 Advanced SIMD) to do the shift. However, as this
     * instruction rounds midpoints up, rather than away from zero, we adjust
     * the input by subtracting 1 from negative values, but only if shift is
     * non-zero.
     */
    const int32x4_t x_adjusted_product =
        vsraq_n_s32(x_product, vbicq_s32(x, vshift_eq_0_mask), 31);  // 右移并进行舍入，处理负值舍入问题
    const int32x4_t y_adjusted_product =
        vsraq_n_s32(y_product, vbicq_s32(y, vshift_eq_0_mask), 31);  // 对第二组数据进行相同操作
    # 对 z_product 应用 vsraq 指令，将结果右移 vbicq_s32(z, vshift_eq_0_mask) 的屏蔽位数，然后执行加法
    const int32x4_t z_adjusted_product =
        vsraq_n_s32(z_product, vbicq_s32(z, vshift_eq_0_mask), 31);
    
    # 对 w_product 应用 vsraq 指令，将结果右移 vbicq_s32(w, vshift_eq_0_mask) 的屏蔽位数，然后执行加法
    const int32x4_t w_adjusted_product =
        vsraq_n_s32(w_product, vbicq_s32(w, vshift_eq_0_mask), 31);
    
    # 使用 vshift 对 x_adjusted_product 进行左移位操作
    const int32x4_t x_scaled = vrshlq_s32(x_adjusted_product, vshift);
    
    # 使用 vshift 对 y_adjusted_product 进行左移位操作
    const int32x4_t y_scaled = vrshlq_s32(y_adjusted_product, vshift);
    
    # 使用 vshift 对 z_adjusted_product 进行左移位操作
    const int32x4_t z_scaled = vrshlq_s32(z_adjusted_product, vshift);
    
    # 使用 vshift 对 w_adjusted_product 进行左移位操作
    const int32x4_t w_scaled = vrshlq_s32(w_adjusted_product, vshift);
#ifdef __aarch64__
    // 在 ARM64 架构下，将 x_scaled 和 y_scaled 向下转换为 int16x8_t，然后与 vzero_point 相加，形成 xy_packed
    const int16x8_t xy_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(x_scaled), y_scaled), vzero_point);
    // 在 ARM64 架构下，将 z_scaled 和 w_scaled 向下转换为 int16x8_t，然后与 vzero_point 相加，形成 zw_packed
    const int16x8_t zw_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(z_scaled), w_scaled), vzero_point);
    // 在 ARM64 架构下，将 xy_packed 和 zw_packed 向下转换为 uint8x16_t，形成 xyzw_packed
    const uint8x16_t xyzw_packed =
        vqmovun_high_s16(vqmovun_s16(xy_packed), zw_packed);
#else
    // 在非 ARM64 架构下，将 x_scaled 和 y_scaled 向下转换为 int16x8_t，并组合成 xy_packed，然后与 vzero_point 相加
    const int16x8_t xy_packed = vqaddq_s16(
        vcombine_s16(vqmovn_s32(x_scaled), vqmovn_s32(y_scaled)), vzero_point);
    // 在非 ARM64 架构下，将 z_scaled 和 w_scaled 向下转换为 int16x8_t，并组合成 zw_packed，然后与 vzero_point 相加
    const int16x8_t zw_packed = vqaddq_s16(
        vcombine_s16(vqmovn_s32(z_scaled), vqmovn_s32(w_scaled)), vzero_point);
    // 在非 ARM64 架构下，将 xy_packed 和 zw_packed 向下转换为 uint8x16_t，形成 xyzw_packed
    const uint8x16_t xyzw_packed =
        vcombine_u8(vqmovun_s16(xy_packed), vqmovun_s16(zw_packed));
#endif

// 对 xyzw_packed 进行饱和限制，确保其在 vqmin 和 vqmax 之间
const uint8x16_t xyzw_clamped =
    vmaxq_u8(vminq_u8(xyzw_packed, vqmax), vqmin);

/*
 * AArch32 version:
 *   4x VQRDMULH.S32 Qd, Qm, Qn
 *   4x VAND Qd, Qm, Dn
 *   4x VSRA.S32 Qd, Qm, #31
 *   4x VRSHL.S32 Qd, Qm, Qn
 *   4x VQMOVN.S32 Dd, Qm
 *   2x VADD.S16 Qd, Qm, Qn
 *   2x VQMOVUN.S16 Dd, Qm
 *   1x VMAX.U8 Qd, Qm, Qn
 *   1x VMIN.U8 Qd, Qm, Qn
 * ---------------------
 * 26 instructions total
 *
 * AArch64 version:
 *   4x SQRDMULH Vd.4S, Vn.4S, Vm.4S
 *   4x AND Vd.16B, Vn.16B, Vm.16B
 *   4x SSRA Vd.4S, Vn.4S, #31
 *   4x SRSHL Vd.4S, Vn.4S, Vm.4S
 *   2x SQXTN Vd.4H, Vn.4S
 *   2x SQXTN2 Vd.8H, Vn.4S
 *   2x ADD Vd.8H, Vn.8H, Vm.8H
 *   1x SQXTUN Vd.8B, Vn.8H
 *   1x SQXTUN2 Vd.16B, Vn.8H
 *   1x UMIN Vd.16B, Vn.16B, Vm.16B
 *   1x UMAX Vd.16B, Vn.16B, Vm.16B
 * ---------------------
 * 26 instructions total
 */

// 将结果写入到 output 指向的内存中，并移动 output 指针到下一个 16 字节位置
vst1q_u8(output, xyzw_clamped);
output += 16;
```