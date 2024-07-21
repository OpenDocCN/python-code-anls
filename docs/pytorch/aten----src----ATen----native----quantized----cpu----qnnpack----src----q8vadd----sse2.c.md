# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8vadd\sse2.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/common.h>
#include <qnnpack/q8vadd.h>
#include <qnnpack/scalar-utils.h>

// 定义一个用于 SSE2 的量化加法内核函数
void pytorch_q8vadd_ukernel__sse2(
    size_t n,  // 要处理的元素数量
    const uint8_t* a,  // 第一个输入数组指针
    const uint8_t* b,  // 第二个输入数组指针
    uint8_t* y,  // 输出数组指针
    const union pytorch_qnnp_add_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {  // 量化参数结构体数组
  if (n == 0) {  // 如果处理元素数量为零，直接返回
    return;
  } else {  // 否则进行加法运算
    // 从量化参数结构体中提取各种量化参数
    const int32_t vzero_point_product =
        quantization_params->sse2.zero_point_product[0];
    const uint32_t va_multiplier = quantization_params->sse2.a_multiplier;
    const uint32_t vb_multiplier = quantization_params->sse2.b_multiplier;
    const int32_t vremainder_mask = quantization_params->sse2.remainder_mask[0];
    const int32_t vremainder_threshold =
        quantization_params->sse2.remainder_threshold[0];
    const uint32_t vshift = quantization_params->sse2.shift;
    const int32_t vy_zero_point =
        (int32_t)quantization_params->sse2.y_zero_point[0];
    const int32_t vy_max =
        (int32_t)(uint32_t)quantization_params->sse2.y_max[0];
    const int32_t vy_min =
        (int32_t)(uint32_t)quantization_params->sse2.y_min[0];

    // 循环处理每个元素
    while (n-- != 0) {
      const uint32_t vxa = (uint32_t)*a++;  // 从第一个输入数组中读取数据
      const uint32_t vxb = (uint32_t)*b++;  // 从第二个输入数组中读取数据

      /* Multiply by factors and accumulate products */
      // 计算加权乘积和
      int32_t vacc = vzero_point_product + (int32_t)(vxa * va_multiplier) +
          (int32_t)(vxb * vb_multiplier);

      /* Shift right and round */
      // 右移并进行舍入
      const int32_t vrem = (vacc & vremainder_mask) - (int32_t)(vacc < 0);
      vacc = asr_s32(vacc, vshift) + (int32_t)(vrem > vremainder_threshold);

      /* Clamp and add output zero point */
      // 限制范围并加上输出零点
      int32_t vy = vacc + vy_zero_point;
      vy = vy >= vy_min ? vy : vy_min;  // 确保不低于最小值
      vy = vy <= vy_max ? vy : vy_max;  // 确保不超过最大值

      *y++ = (uint8_t)vy;  // 将处理后的结果写入输出数组
    }
  }
}
```