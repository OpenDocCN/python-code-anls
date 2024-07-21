# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\fp32-psimd.c`

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

#include <psimd.h>

#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_fp32__psimd(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  // 断言输入大小是16的倍数，保证输入数据可以正确地处理
  assert(n % 16 == 0);
  // 断言缩放因子scale小于1.0，确保缩放因子在合理范围内
  assert(scale < 1.0f);
  // 断言缩放因子scale不小于2的-32次方，确保缩放因子在合理范围内
  assert(scale >= 0x1.0p-32f);

  // 使用psimd_splat_f32将scale转换成psimd_f32类型，便于SIMD操作
  const psimd_f32 vscale = psimd_splat_f32(scale);
  // 计算vfmin，将(qmin - zero_point)转换成psimd_f32类型
  const psimd_f32 vfmin = psimd_splat_f32(
      (float)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point));
  // 计算vfmax，将(qmax - zero_point)转换成psimd_f32类型
  const psimd_f32 vfmax = psimd_splat_f32(
      (float)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point));
  // 设置vfmagic，一个psimd_f32常数，用于后续计算
  const psimd_f32 vfmagic = psimd_splat_f32(12582912.0f);
  // 设置vimagic，一个psimd_s32常数，用于后续计算
  const psimd_s32 vimagic =
      psimd_splat_s32(INT32_C(0x4B400000) - (int32_t)(uint32_t)zero_point);
  // 循环处理每个输入的16个元素
  for (; n != 0; n -= 16) {
    // 加载16个int32_t的输入数据到psimd_s32类型的变量x, y, z, w中
    const psimd_s32 x = psimd_load_s32(input);
    const psimd_s32 y = psimd_load_s32(input + 4);
    const psimd_s32 z = psimd_load_s32(input + 8);
    const psimd_s32 w = psimd_load_s32(input + 12);
    input += 16;

    /*
     * 将int32_t类型的输入转换成FP32，并乘以FP32类型的缩放因子。
     * 这两个操作都涉及到舍入：
     * - 大的int32_t值不能精确地表示为FP32。我们期望转换指令会将其舍入为最近的FP32值，但Clang文档对__builtin_convertvector并不保证这一点。
     * - 两个FP32值的乘积通常不能精确地表示为FP32值，并将舍入为最近的FP32值，带有ties to even。
     */
    const psimd_f32 x_scaled = psimd_cvt_s32_f32(x) * vscale;
    const psimd_f32 y_scaled = psimd_cvt_s32_f32(y) * vscale;
    const psimd_f32 z_scaled = psimd_cvt_s32_f32(z) * vscale;
    const psimd_f32 w_scaled = psimd_cvt_s32_f32(w) * vscale;

    /*
     * Clang/gcc向量扩展不提供带有最近舍入到偶数的浮点到整数转换操作的内部函数。
     * 缺少这样的内部函数，我们使用一个魔术技巧，将一个大数（1.5 * 2 ** 23）加到缩放值上，导致舍入到整数，
     * 然后将这个魔术数作为整数减去。这个技巧仅在一个有限的范围内有效（输入的绝对值必须小于2 ** 22），
     * 因此通常我们需要在使用魔术之前将输入限制在这个范围内。但是，将输入限制在任何较小的范围内同样有效，
     * 因此我们将输入限制在[qmin - zero point, qmax - zero point]范围内，以便在将零点加到结果之后，
     * 它可以进入目标[qmin, qmax]范围内。
     */
    const psimd_f32 x_clamped =
        psimd_min_f32(psimd_max_f32(x_scaled, vfmin), vfmax);
    const psimd_f32 y_clamped =
        psimd_min_f32(psimd_max_f32(y_scaled, vfmin), vfmax);
    const psimd_f32 z_clamped =
        psimd_min_f32(psimd_max_f32(z_scaled, vfmin), vfmax);
    const psimd_f32 w_clamped =
        psimd_min_f32(psimd_max_f32(w_scaled, vfmin), vfmax);

    /*
     * 使用 psimd_max_f32 和 psimd_min_f32 函数对 y_scaled、z_scaled、w_scaled 进行范围限制，
     * 将它们约束在 vfmin 和 vfmax 之间，并分别存储在 y_clamped、z_clamped、w_clamped 中。
     */
    const psimd_s32 x_biased = (psimd_s32)(x_clamped + vfmagic) - vimagic;
    const psimd_s32 y_biased = (psimd_s32)(y_clamped + vfmagic) - vimagic;
    const psimd_s32 z_biased = (psimd_s32)(z_clamped + vfmagic) - vimagic;
    const psimd_s32 w_biased = (psimd_s32)(w_clamped + vfmagic) - vimagic;

    /*
     * 将 x_clamped、y_clamped、z_clamped、w_clamped 加上 vfmagic 后转换为整数，
     * 然后减去 vimagic，实现整数转换的"魔术技巧"。这里通过 psimd_s32 将结果存储在
     * x_biased、y_biased、z_biased、w_biased 中。
     */

    /*
     * 从每个 32 位整数向量中选择低 8 位用于输出。由于结果已经被限制在 [qmin, qmax] 的子范围内，
     * 不需要进行饱和处理。
     */
    const psimd_u16 xy_packed =
        psimd_concat_even_u16((psimd_u16)x_biased, (psimd_u16)y_biased);
    const psimd_u16 zw_packed =
        psimd_concat_even_u16((psimd_u16)z_biased, (psimd_u16)w_biased);

    const psimd_u8 xyzw_packed =
        psimd_concat_even_u8((psimd_u8)xy_packed, (psimd_u8)zw_packed);

    // 将 xyzw_packed 中的数据存储到内存地址 output 指向的位置
    psimd_store_u8(output, xyzw_packed);
    // 将 output 指针向前移动 16 个字节，准备存储下一个块的数据
    output += 16;
  }
}


注释：


# 这是一个单独的右花括号 '}'，用于结束一个代码块或语句。
```