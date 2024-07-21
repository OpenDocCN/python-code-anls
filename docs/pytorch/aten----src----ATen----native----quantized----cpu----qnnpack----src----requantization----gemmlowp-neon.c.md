# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\gemmlowp-neon.c`

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

/*
 * The requantization implementation below is adapted from Google's gemmlowp
 * library. It is only used in QNNPACK unit tests and comparative benchmarks,
 * but not the library itself.
 */

// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// 定义了一个名为 pytorch_qnnp_requantize_gemmlowp__neon 的函数，用于 NEON 指令集下的量化函数实现
void pytorch_qnnp_requantize_gemmlowp__neon(
    size_t n,                             // 输入数据的长度，要求是 16 的倍数
    const int32_t* input,                 // 输入的整型数据数组指针
    float scale,                          // 缩放因子，用于量化过程中的缩放
    uint8_t zero_point,                   // 量化零点
    uint8_t qmin,                         // 量化的最小输出值
    uint8_t qmax,                         // 量化的最大输出值
    uint8_t* output) {                    // 输出的无符号整型数据数组指针，用于存放量化后的结果
  assert(n % 16 == 0);                    // 断言输入数据长度必须是 16 的倍数
  assert(scale < 1.0f);                   // 断言缩放因子小于 1.0
  assert(scale >= 0x1.0p-32f);            // 断言缩放因子不小于 2^-32

  const uint32_t scale_bits = fp32_to_bits(scale);  // 将浮点数缩放因子转换为整型表示

  /* Compute requantization parameters */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;  // 计算量化乘法因子
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;    // 计算指数部分
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);                          // 计算量化的移位量

  const int32x4_t vmultiplier = vdupq_n_s32(multiplier);  // 使用 NEON 指令创建乘法因子的向量
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);  // 创建零点的向量
  const int32x4_t vshift = vdupq_n_s32(-shift);  // 创建移位量的向量
  const uint8x16_t vqmin = vdupq_n_u8(qmin);      // 创建最小量化值的向量
  const uint8x16_t vqmax = vdupq_n_u8(qmax);      // 创建最大量化值的向量
  for (; n != 0; n -= 16) {                       // 迭代处理输入数据，每次处理 16 个元素
    const int32x4_t x = vld1q_s32(input);         // 加载输入数据到 NEON 寄存器中
    const int32x4_t y = vld1q_s32(input + 4);     // 加载下一个 4 个数据
    const int32x4_t z = vld1q_s32(input + 8);     // 加载下一个 4 个数据
    const int32x4_t w = vld1q_s32(input + 12);    // 加载下一个 4 个数据
    input += 16;                                  // 更新输入数据指针位置

    const int32x4_t x_product = vqrdmulhq_s32(x, vmultiplier);  // 使用 VQRDMUL 实现量化乘法
    const int32x4_t y_product = vqrdmulhq_s32(y, vmultiplier);  // 对每个向量元素进行乘法
    const int32x4_t z_product = vqrdmulhq_s32(z, vmultiplier);  // 并取高位结果
    const int32x4_t w_product = vqrdmulhq_s32(w, vmultiplier);  // 与真实乘法的效果相似

    const int32x4_t x_product_fixup = vshrq_n_s32(vandq_s32(x, vshift), 31);  // 计算修正项
    const int32x4_t y_product_fixup = vshrq_n_s32(vandq_s32(y, vshift), 31);  // 用于舍入
    const int32x4_t z_product_fixup = vshrq_n_s32(vandq_s32(z, vshift), 31);  // 防止溢出
    const int32x4_t w_product_fixup = vshrq_n_s32(vandq_s32(w, vshift), 31);  // 的修正值

    const int32x4_t x_adjusted_product = vqaddq_s32(x_product, x_product_fixup);  // 调整乘法结果


这段代码是一个基于NEON指令集的量化函数实现，用于将输入的整型数据根据给定的缩放因子和量化参数转换为无符号整型数据。
    # 将 y_product 和 y_product_fixup 的元素逐个相加，并进行饱和加法，结果存储在 y_adjusted_product 中
    const int32x4_t y_adjusted_product = vqaddq_s32(y_product, y_product_fixup);
    
    # 将 z_product 和 z_product_fixup 的元素逐个相加，并进行饱和加法，结果存储在 z_adjusted_product 中
    const int32x4_t z_adjusted_product = vqaddq_s32(z_product, z_product_fixup);
    
    # 将 w_product 和 w_product_fixup 的元素逐个相加，并进行饱和加法，结果存储在 w_adjusted_product 中
    const int32x4_t w_adjusted_product = vqaddq_s32(w_product, w_product_fixup);
    
    # 对 x_adjusted_product 中的每个元素进行按位右移，位移量由 vshift 决定，结果存储在 x_scaled 中
    const int32x4_t x_scaled = vrshlq_s32(x_adjusted_product, vshift);
    
    # 对 y_adjusted_product 中的每个元素进行按位右移，位移量由 vshift 决定，结果存储在 y_scaled 中
    const int32x4_t y_scaled = vrshlq_s32(y_adjusted_product, vshift);
    
    # 对 z_adjusted_product 中的每个元素进行按位右移，位移量由 vshift 决定，结果存储在 z_scaled 中
    const int32x4_t z_scaled = vrshlq_s32(z_adjusted_product, vshift);
    
    # 对 w_adjusted_product 中的每个元素进行按位右移，位移量由 vshift 决定，结果存储在 w_scaled 中
    const int32x4_t w_scaled = vrshlq_s32(w_adjusted_product, vshift);
#ifdef __aarch64__
    // 如果目标架构是 AArch64，则使用高级的 SIMD 操作方式
    const int16x8_t xy_packed = vqaddq_s16(
        // 将 x_scaled 和 y_scaled 各自转换为 32 位有符号整数后，取高位 4 个元素进行合并，再加上 vzero_point
        vqmovn_high_s32(vqmovn_s32(x_scaled), y_scaled), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(
        // 将 z_scaled 和 w_scaled 各自转换为 32 位有符号整数后，取高位 4 个元素进行合并，再加上 vzero_point
        vqmovn_high_s32(vqmovn_s32(z_scaled), w_scaled), vzero_point);
    const uint8x16_t xyzw_packed =
        // 将 xy_packed 和 zw_packed 合并成一个 16 位无符号整数向量
        vqmovun_high_s16(vqmovun_s16(xy_packed), zw_packed);
#else
    // 如果不是 AArch64 架构，则使用普通的 SIMD 操作方式
    const int16x8_t xy_packed = vqaddq_s16(
        // 将 x_scaled 和 y_scaled 各自转换为 32 位有符号整数后，合并成一个 8 个元素的向量，再加上 vzero_point
        vcombine_s16(vqmovn_s32(x_scaled), vqmovn_s32(y_scaled)), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(
        // 将 z_scaled 和 w_scaled 各自转换为 32 位有符号整数后，合并成一个 8 个元素的向量，再加上 vzero_point
        vcombine_s16(vqmovn_s32(z_scaled), vqmovn_s32(w_scaled)), vzero_point);
    const uint8x16_t xyzw_packed =
        // 将 xy_packed 和 zw_packed 合并成一个 16 位无符号整数向量
        vcombine_u8(vqmovun_s16(xy_packed), vqmovun_s16(zw_packed));
#endif

    const uint8x16_t xyzw_clamped =
        // 对 xyzw_packed 进行元素级别的上界和下界截断处理
        vmaxq_u8(vminq_u8(xyzw_packed, vqmax), vqmin);

    // 将处理后的 xyzw_clamped 向量存储到 output 指向的内存位置
    vst1q_u8(output, xyzw_clamped);
    // 更新 output 指针，使其指向下一个 16 个字节的位置
    output += 16;
}
```