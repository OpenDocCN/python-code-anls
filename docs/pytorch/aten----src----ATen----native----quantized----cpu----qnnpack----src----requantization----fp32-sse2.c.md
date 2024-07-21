# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\fp32-sse2.c`

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

#include <emmintrin.h>

#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_fp32__sse2(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 16 == 0);   // 确保输入长度是16的倍数，因为每次处理16个元素
  assert(scale < 1.0f);  // 确保缩放因子小于1.0
  assert(scale >= 0x1.0p-32f);  // 确保缩放因子大于等于2^-32

  // 将缩放因子转换成 SSE 寄存器格式
  const __m128 vscale = _mm_set1_ps(scale);
  // 将零点值转换成 SSE 寄存器格式
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);
  // 将最小量化值转换成 SSE 寄存器格式
  const __m128i vqmin = _mm_set1_epi8((char)qmin);
  // 将最大量化值转换成 SSE 寄存器格式
  const __m128i vqmax = _mm_set1_epi8((char)qmax);

  // 循环处理每个16个元素的数据块
  for (; n != 0; n -= 16) {
    // 依次加载四个128位整数型数据块
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    /*
     * 将 int32_t 输入转换为 FP32，并乘以 FP32 缩放因子。
     * 这两个操作都涉及到统计上无偏的舍入（使用默认的 MXCSR 舍入模式）：
     * - 大的 int32_t 值无法精确表示为 FP32。x86 上的 CVTDQ2PS 指令将其舍入为最接近的 FP32 值（默认情况下，按最接近的偶数舍入）。
     * - 两个 FP32 值的乘积通常不会精确地表示为一个 FP32 值，并且会按照最接近的偶数舍入为最接近的 FP32 值（默认情况下）。
     */
    const __m128 x_scaled = _mm_mul_ps(_mm_cvtepi32_ps(x), vscale);
    const __m128 y_scaled = _mm_mul_ps(_mm_cvtepi32_ps(y), vscale);
    const __m128 z_scaled = _mm_mul_ps(_mm_cvtepi32_ps(z), vscale);
    const __m128 w_scaled = _mm_mul_ps(_mm_cvtepi32_ps(w), vscale);
    /*
     * 使用 CVTPS2DQ 指令将经过缩放的 FP32 结果转换为 int32_t 类型。CVTPS2DQ 指令根据
     * 最接近的 FP32 值进行四舍五入（假设默认的 MXCSR 舍入模式）。然而，当转换溢出时，
     * 它会产生 INT32_MIN 作为结果。对于较大的正输入，转换结果可能会变为负数，这会
     * 影响最终的重新量化结果。例如，在 x86 SSE2 中，int32_t(float(INT32_MAX)) == INT32_MIN！
     * 这是因为 float(INT32_MAX) 四舍五入为 2**31，当它转换回整数时会导致 int32_t 溢出。
     *
     * 幸运的是，在这个重新量化方案中，我们可以证明溢出永远不会发生。最大的正输入是
     * INT32_MAX（2**31 - 1），当转换为 float 后变成 2**31。最大的缩放值是 0x1.FFFFFEp-1。
     * 当它们相乘时，结果是 2147483520（与 INT32_MAX = 2147483647 比较），在不溢出的情况下适合 int32_t。
     */
    const __m128i x_rounded = _mm_cvtps_epi32(x_scaled);
    const __m128i y_rounded = _mm_cvtps_epi32(y_scaled);
    const __m128i z_rounded = _mm_cvtps_epi32(z_scaled);
    const __m128i w_rounded = _mm_cvtps_epi32(w_scaled);
    
    /*
     * 在 x86 SSE2 上的标准最终序列：
     * - 将结果打包到 int16_t 并进行饱和
     * - 加上零点
     * - 将结果打包到 uint8_t 并进行饱和
     * - 将结果夹在 qmin 和 qmax 之间
     */
    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_rounded, y_rounded), vzero_point);
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_rounded, w_rounded), vzero_point);
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);
    
    /*
     * 总共 19 条指令：
     * 4x CVTDQ2PS
     * 4x MULPS
     * 4x CVTPS2DQ
     * 2x PACKSSDW
     * 1x PACKUSWB
     * 2x PADDW
     * 1x PMAXUB
     * 1x PMINUB
     */
    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    output += 16;
}
```