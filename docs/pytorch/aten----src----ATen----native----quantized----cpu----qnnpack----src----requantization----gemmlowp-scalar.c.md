# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\gemmlowp-scalar.c`

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

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>
#include <qnnpack/scalar-utils.h>

#include "gemmlowp-scalar.h"

void pytorch_qnnp_requantize_gemmlowp__scalar(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 4 == 0);  // 确保输入数量是4的倍数，以便按4个元素一组处理
  assert(scale < 1.0f);  // 确保缩放因子小于1
  assert(scale >= 0x1.0p-32f);  // 确保缩放因子大于或等于2^-32

  const uint32_t scale_bits = fp32_to_bits(scale);

  /* 计算重新量化所需的参数 */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);

  const int32_t smin = (int32_t)(uint32_t)qmin;
  const int32_t smax = (int32_t)(uint32_t)qmax;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    const int32_t x_product = gemmlowp_scalar_vqrdmulh_s32(x, multiplier);  // 计算输入值乘以乘法器得到的乘积的高16位
    const int32_t y_product = gemmlowp_scalar_vqrdmulh_s32(y, multiplier);  // 同上，对y进行乘法运算
    const int32_t z_product = gemmlowp_scalar_vqrdmulh_s32(z, multiplier);  // 同上，对z进行乘法运算
    const int32_t w_product = gemmlowp_scalar_vqrdmulh_s32(w, multiplier);  // 同上，对w进行乘法运算

    const int32_t x_scaled = gemmlowp_scalar_rdivbypo2_s32(x_product, shift);  // 将x_product右移shift位
    const int32_t y_scaled = gemmlowp_scalar_rdivbypo2_s32(y_product, shift);  // 同上，对y_product进行右移
    const int32_t z_scaled = gemmlowp_scalar_rdivbypo2_s32(z_product, shift);  // 同上，对z_product进行右移
    const int32_t w_scaled = gemmlowp_scalar_rdivbypo2_s32(w_product, shift);  // 同上，对w_product进行右移

    /* 将零点添加到缩放后的值 */
    const int32_t x_biased = x_scaled + zero_point;  // 将x_scaled与零点相加
    const int32_t y_biased = y_scaled + zero_point;  // 同上，将y_scaled与零点相加
    const int32_t z_biased = z_scaled + zero_point;  // 同上，将z_scaled与零点相加
    const int32_t w_biased = w_scaled + zero_point;  // 同上，将w_scaled与零点相加

    /* 将缩放后的值限制在smin和smax之间 */
    const int32_t x_clamped =
        x_biased < smin ? smin : x_biased > smax ? smax : x_biased;  // 将x_biased限制在[smin, smax]范围内
    const int32_t y_clamped =
        y_biased < smin ? smin : y_biased > smax ? smax : y_biased;  // 同上，将y_biased限制在[smin, smax]范围内
    const int32_t z_clamped =
        z_biased < smin ? smin : z_biased > smax ? smax : z_biased;  // 同上，将z_biased限制在[smin, smax]范围内
    const int32_t w_clamped =
        w_biased < smin ? smin : w_biased > smax ? smax : w_biased;  // 同上，将w_biased限制在[smin, smax]范围内

    output[0] = (uint8_t)x_clamped;  // 将x_clamped强制转换为uint8_t类型并存储在输出数组中
    output[1] = (uint8_t)y_clamped;  // 同上，将y_clamped强制转换为uint8_t类型并存储在输出数组中
    output[2] = (uint8_t)z_clamped;  // 同上，将z_clamped强制转换为uint8_t类型并存储在输出数组中
    output[3] = (uint8_t)w_clamped;  // 同上，将w_clamped强制转换为uint8_t类型并存储在输出数组中
    output += 4;  // 将输出指针向后移动4个字节，以便下一组数据存储在正确的位置
  }
}
```