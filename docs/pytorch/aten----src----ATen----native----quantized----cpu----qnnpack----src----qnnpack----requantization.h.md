# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\requantization.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <fp16/bitcasts.h>

#include <qnnpack/params.h>
#include <qnnpack/scalar-utils.h>

static inline union pytorch_qnnp_q31_requantization_params
pytorch_qnnp_compute_scalar_requantization_params(
    float scale,
    uint8_t zero_point,
    uint8_t min,
    uint8_t max) {
  /* Compute requantization parameters */
  assert(scale < 1.0f);  // 确保缩放因子小于1.0
  assert(scale >= 0x1.0p-32f);  // 确保缩放因子大于或等于2^-32
  const uint32_t scale_bits = fp32_to_bits(scale);  // 将浮点数缩放因子转换为整数表示

  /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
  const int32_t multiplier = (int32_t)(
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);  // 计算乘数，确保在特定范围内
  assert(multiplier >= INT32_C(0x40000000));  // 确保乘数大于等于0x40000000
  assert(multiplier <= INT32_C(0x7FFFFF80));  // 确保乘数小于等于0x7FFFFF80

  /* Shift is in [0, 31] range */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);  // 计算移位量，确保在0到31之间
  assert(shift >= 0);  // 确保移位量大于等于0
  assert(shift < 32);  // 确保移位量小于32

  union pytorch_qnnp_q31_requantization_params params;  // 定义包含量化参数的联合体
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);  // 计算余数掩码
  const uint32_t remainder_threshold = remainder_mask >> 1;  // 计算余数阈值
  params.scalar.multiplier = multiplier;  // 存储乘数到参数结构体中
  params.scalar.remainder_mask = (int32_t)remainder_mask;  // 存储余数掩码到参数结构体中
  params.scalar.remainder_threshold = (int32_t)remainder_threshold;  // 存储余数阈值到参数结构体中
  params.scalar.shift = (uint32_t)shift;  // 存储移位量到参数结构体中
  params.scalar.min_less_zero_point =
      (int32_t)(uint32_t)min - (int32_t)(uint32_t)zero_point;  // 计算最小值减去零点值并存储到参数结构体中
  params.scalar.max_less_zero_point =
      (int32_t)(uint32_t)max - (int32_t)(uint32_t)zero_point;  // 计算最大值减去零点值并存储到参数结构体中
  params.scalar.zero_point = (int32_t)(uint32_t)zero_point;  // 存储零点值到参数结构体中
  return params;  // 返回计算得到的参数结构体
}

static inline union pytorch_qnnp_fp32_requantization_params
pytorch_qnnp_compute_scalar_fp32_requantization_params(
    float* scales,
    uint8_t zero_point,
    uint8_t min,
    uint8_t max) {

  union pytorch_qnnp_fp32_requantization_params params;  // 定义包含浮点量化参数的联合体
  params.scalar.scales = scales;  // 存储尺度参数到参数结构体中
  params.scalar.output_zero_point = zero_point;  // 存储输出零点值到参数结构体中
  params.scalar.output_max = max;  // 存储输出最大值到参数结构体中
  params.scalar.output_min = min;  // 存储输出最小值到参数结构体中
  params.scalar.min_less_zero_point = ((float)((int32_t)(uint32_t)min -
      (int32_t)(uint32_t)zero_point));  // 计算最小值减去零点值的浮点数并存储到参数结构体中
  params.scalar.max_less_zero_point = ((float)((int32_t)(uint32_t)max -
      (int32_t)(uint32_t)zero_point));  // 计算最大值减去零点值的浮点数并存储到参数结构体中
  params.scalar.magic = 12582912.0f;  // 存储魔法数到参数结构体中
  params.scalar.magic_less_zero_point = (INT32_C(0x4B400000) -
      (int32_t)(uint32_t)zero_point);  // 计算魔法数减去零点值并存储到参数结构体中
  return params;  // 返回构建好的参数结构体
}

static inline union pytorch_qnnp_q31_requantization_params
pytorch_qnnp_compute_requantization_params(
    float scale,
    uint8_t zero_point,
    uint8_t min,
    uint8_t max) {
    /* 计算重新量化参数 */

    /* 将浮点数 scale 转换为整数的二进制表示 */
    const uint32_t scale_bits = fp32_to_bits(scale);

    /* 计算乘数，范围在 [0x40000000, 0x7FFFFF80] 之间 */
    const int32_t multiplier = (int32_t)(
        ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
    assert(multiplier >= INT32_C(0x40000000));
    assert(multiplier <= INT32_C(0x7FFFFF80));

    /* 计算移位量，范围在 [0, 31] 之间 */
    const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
    assert(shift >= 0);
    assert(shift < 32);

    /* 声明 pytorch_qnnp_q31_requantization_params 结构的联合体变量 params */
    union pytorch_qnnp_q31_requantization_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  // 计算余数掩码，用于 SSE2 SIMD 算法
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  // 计算余数阈值，用于 SSE2 SIMD 算法
  const uint32_t remainder_threshold = remainder_mask >> 1;
  // 设置 SSE2 SIMD 参数：乘数设置为 multiplier
  params.sse2.multiplier[0] = multiplier;
  params.sse2.multiplier[1] = multiplier;
  params.sse2.multiplier[2] = multiplier;
  params.sse2.multiplier[3] = multiplier;
  // 设置 SSE2 SIMD 参数：舍入值设为 0x40000000
  params.sse2.rounding[0] = UINT64_C(0x40000000);
  params.sse2.rounding[1] = UINT64_C(0x40000000);
  // 设置 SSE2 SIMD 参数：余数掩码设为 remainder_mask
  params.sse2.remainder_mask[0] = (int32_t)remainder_mask;
  params.sse2.remainder_mask[1] = (int32_t)remainder_mask;
  params.sse2.remainder_mask[2] = (int32_t)remainder_mask;
  params.sse2.remainder_mask[3] = (int32_t)remainder_mask;
  // 设置 SSE2 SIMD 参数：余数阈值设为 remainder_threshold
  params.sse2.remainder_threshold[0] = (int32_t)remainder_threshold;
  params.sse2.remainder_threshold[1] = (int32_t)remainder_threshold;
  params.sse2.remainder_threshold[2] = (int32_t)remainder_threshold;
  params.sse2.remainder_threshold[3] = (int32_t)remainder_threshold;
  // 设置 SSE2 SIMD 参数：移位值设为 shift
  params.sse2.shift[0] = (uint64_t)(uint32_t)shift;
  params.sse2.shift[1] = (uint64_t)(uint32_t)shift;
  // 设置 SSE2 SIMD 参数：零点值设为 zero_point
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.zero_point[i] = (int16_t)(uint16_t)zero_point;
  }
  // 设置 SSE2 SIMD 参数：最大和最小值分别设为 max 和 min
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.max[i] = max;
    params.sse2.min[i] = min;
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  // 设置 NEON SIMD 参数：乘数设为 multiplier
  params.neon.multiplier = multiplier;
  // 设置 NEON SIMD 参数：右移位数设为 -shift
  params.neon.right_shift = -shift;
  // 设置 NEON SIMD 参数：零点值设为 zero_point
  params.neon.zero_point = (int16_t)(uint16_t)zero_point;
  // 设置 NEON SIMD 参数：最大和最小值分别设为 max 和 min
  params.neon.max = max;
  params.neon.min = min;
#else
  // 计算余数掩码，用于标量计算
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  // 计算余数阈值，用于标量计算
  const uint32_t remainder_threshold = remainder_mask >> 1;
  // 设置标量参数：乘数设为 multiplier
  params.scalar.multiplier = multiplier;
  // 设置标量参数：余数掩码设为 remainder_mask
  params.scalar.remainder_mask = (int32_t)remainder_mask;
  // 设置标量参数：余数阈值设为 remainder_threshold
  params.scalar.remainder_threshold = (int32_t)remainder_threshold;
  // 设置标量参数：移位值设为 shift
  params.scalar.shift = (uint32_t)shift;
  // 计算标量参数：min_less_zero_point 和 max_less_zero_point
  params.scalar.min_less_zero_point = (int32_t)(uint32_t)min - (int32_t)(uint32_t)zero_point;
  params.scalar.max_less_zero_point = (int32_t)(uint32_t)max - (int32_t)(uint32_t)zero_point;
  // 设置标量参数：零点值设为 zero_point
  params.scalar.zero_point = (int32_t)(uint32_t)zero_point;
#endif
  // 返回计算好的量化参数结构体
  return params;
}
    params.sse2.output_min[i] = output_min;



# 将变量 output_min 赋值给 params.sse2.output_min 列表的第 i 个元素
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  // 如果目标平台是 ARM 或 ARM64 架构，则设置 NEON 指令集的参数
  params.neon.input_zero_point = (int16_t)(uint16_t)input_zero_point;
  // 设置输入的零点偏移量为 16 位有符号整数的强制类型转换结果
  params.neon.kernel_zero_points = kernel_zero_points;
  // 设置内核的零点偏移数组
  params.neon.requantization_scales = requantization_scales;
  // 设置重新量化的比例因子数组
  params.neon.output_zero_point = (int16_t)(uint16_t)output_zero_point;
  // 设置输出的零点偏移量为 16 位有符号整数的强制类型转换结果
  params.neon.output_max = output_max;
  // 设置输出的最大值
  params.neon.output_min = output_min;
  // 设置输出的最小值
  params.neon.vfmin = ((float)((int32_t)(uint32_t)output_min -
      (int32_t)(uint32_t)output_zero_point));
  // 计算并设置 NEON 浮点向量的最小值
  params.neon.vfmax = ((float)((int32_t)(uint32_t)output_max -
      (int32_t)(uint32_t)output_zero_point));
  // 计算并设置 NEON 浮点向量的最大值
  params.neon.vfmagic = 12582912.0f;
  // 设置 NEON 浮点向量的魔数
  params.neon.vimagic = (INT32_C(0x4B400000) -
      (int32_t)(uint32_t)output_zero_point);
  // 计算并设置 NEON 整数向量的魔数
#else
  // 如果目标平台不是 ARM 或 ARM64 架构，则设置标量指令集的参数
  params.scalar.input_zero_point = (int32_t)(uint32_t)input_zero_point;
  // 设置输入的零点偏移量为 32 位有符号整数的强制类型转换结果
  params.scalar.kernel_zero_points = kernel_zero_points;
  // 设置内核的零点偏移数组
  params.scalar.requantization_scales = requantization_scales;
  // 设置重新量化的比例因子数组
  params.scalar.output_min_less_zero_point =
      (int32_t)(uint32_t)output_min - (int32_t)(uint32_t)output_zero_point;
  // 计算并设置输出的最小值减去零点偏移量
  params.scalar.output_max_less_zero_point =
      (int32_t)(uint32_t)output_max - (int32_t)(uint32_t)output_zero_point;
  // 计算并设置输出的最大值减去零点偏移量
  params.scalar.output_zero_point = (int32_t)(uint32_t)output_zero_point;
  // 设置输出的零点偏移量为 32 位有符号整数的强制类型转换结果
#endif
  // 返回计算好的参数结构体
  return params;
}

static inline union pytorch_qnnp_avgpool_quantization_params
pytorch_qnnp_compute_avgpool_quantization_params(
    int32_t bias,
    float scale,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max) {
  /* Compute requantization parameters */
  // 断言确保比例因子在合理范围内
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  // 定义用于存储平均池化量化参数的联合体
  union pytorch_qnnp_avgpool_quantization_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  // 如果目标平台是 x86 或 x86_64 架构，则使用 SSE2 指令集设置参数
  params.sse2.bias[0] = bias;
  params.sse2.bias[1] = bias;
  params.sse2.bias[2] = bias;
  params.sse2.bias[3] = bias;
  // 设置 SSE2 指令集的偏置值数组
  params.sse2.scale[0] = scale;
  params.sse2.scale[1] = scale;
  params.sse2.scale[2] = scale;
  params.sse2.scale[3] = scale;
  // 设置 SSE2 指令集的比例因子数组
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.output_zero_point[i] = (int16_t)(uint16_t)output_zero_point;
    // 设置 SSE2 指令集的输出零点偏移量数组
  }
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.output_max[i] = output_max;
    params.sse2.output_min[i] = output_min;
    // 设置 SSE2 指令集的输出最大值和最小值数组
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  // 如果目标平台是 ARM 或 ARM64 架构，则使用 NEON 指令集设置参数
  params.neon.bias = bias;
  // 设置 NEON 指令集的偏置值
  params.neon.scale = scale;
  // 设置 NEON 指令集的比例因子
  params.neon.output_zero_point = (int16_t)(uint16_t)output_zero_point;
  // 设置 NEON 指令集的输出零点偏移量
  params.neon.output_max = output_max;
  // 设置 NEON 指令集的输出最大值
  params.neon.output_min = output_min;
  // 设置 NEON 指令集的输出最小值
  params.neon.vfmin = ((float)((int32_t)(uint32_t)output_min -
      (int32_t)(uint32_t)output_zero_point));
  // 计算并设置 NEON 浮点向量的最小值
  params.neon.vfmax = ((float)((int32_t)(uint32_t)output_max -
      (int32_t)(uint32_t)output_zero_point));
  // 计算并设置 NEON 浮点向量的最大值
  params.neon.vfmagic = 12582912.0f;
  // 设置 NEON 浮点向量的魔数
  params.neon.vimagic = (INT32_C(0x4B400000) -
      (int32_t)(uint32_t)output_zero_point);
  // 计算并设置 NEON 整数向量的魔数
#endif
  // 返回计算好的参数结构体
  return params;
}
#else
  // 设置标量参数的偏置值为给定的偏置值
  params.scalar.bias = bias;
  // 设置标量参数的缩放因子为给定的缩放因子
  params.scalar.scale = scale;
  // 将输出的零点转换为32位有符号整数，然后存入标量参数
  params.scalar.output_zero_point = (int32_t)(uint32_t)output_zero_point;
  // 将输出的最大值转换为32位有符号整数，然后存入标量参数
  params.scalar.output_max = (int32_t)(uint32_t)output_max;
  // 将输出的最小值转换为32位有符号整数，然后存入标量参数
  params.scalar.output_min = (int32_t)(uint32_t)output_min;
#endif
  // 返回计算得到的参数结构体
  return params;
}

static inline union pytorch_qnnp_avgpool_quantization_params
pytorch_qnnp_compute_scalar_avgpool_quantization_params(
    int32_t bias,
    float scale,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max) {
  /* 计算重新量化参数 */
  // 断言确保缩放因子大于等于2^-32
  assert(scale >= 0x1.0p-32f);
  // 断言确保缩放因子小于256.0
  assert(scale < 256.0f);

  // 定义并初始化用于存储量化参数的联合体
  union pytorch_qnnp_avgpool_quantization_params params;
  // 设置标量参数的偏置值为给定的偏置值
  params.scalar.bias = bias;
  // 设置标量参数的缩放因子为给定的缩放因子
  params.scalar.scale = scale;
  // 将输出的零点转换为32位有符号整数，然后存入标量参数
  params.scalar.output_zero_point = (int32_t)(uint32_t)output_zero_point;
  // 将输出的最大值转换为32位有符号整数，然后存入标量参数
  params.scalar.output_max = (int32_t)(uint32_t)output_max;
  // 将输出的最小值转换为32位有符号整数，然后存入标量参数
  params.scalar.output_min = (int32_t)(uint32_t)output_min;
  // 返回计算得到的标量量化参数
  return params;
}

static inline union pytorch_qnnp_u8_clamping_params
pytorch_qnnp_compute_u8_clamping_params(
    uint8_t output_min,
    uint8_t output_max) {
  // 断言确保输出最小值不大于输出最大值
  assert(output_min <= output_max);

  // 定义并初始化用于存储量化参数的联合体
  union pytorch_qnnp_u8_clamping_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  // 对于x86平台，使用SSE2指令集，设置输出最大值和输出最小值
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.output_max[i] = output_max;
    params.sse2.output_min[i] = output_min;
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  // 对于ARM平台，使用NEON指令集，设置输出最大值和输出最小值
  params.neon.output_max = output_max;
  params.neon.output_min = output_min;
#else
  // 对于其他平台，设置标量参数的输出最小值和输出最大值
  params.scalar.output_min = (int32_t)(uint32_t)output_min;
  params.scalar.output_max = (int32_t)(uint32_t)output_max;
#endif
  // 返回计算得到的量化参数
  return params;
}

static inline union pytorch_qnnp_add_quantization_params
pytorch_qnnp_compute_add_quantization_params(
    uint8_t a_zero_point,
    uint8_t b_zero_point,
    uint8_t output_zero_point,
    float a_output_scale,
    float b_output_scale,
    uint8_t output_min,
    // 确保输出比例因子a_output_scale和b_output_scale大于等于2^-14
    // 这是由于assert函数的作用是确保其参数为真，如果不是，将会触发错误
    assert(a_output_scale >= 0x1.0p-14f);
    assert(b_output_scale >= 0x1.0p-14f);
    // 确保输出比例因子a_output_scale和b_output_scale小于2^8
    assert(a_output_scale < 0x1.0p+8f);
    assert(b_output_scale < 0x1.0p+8f);
    
    /* 计算重新量化参数 */
    // 计算最大输出比例因子max_output_scale，选择较大的a_output_scale和b_output_scale
    const float max_output_scale =
        a_output_scale > b_output_scale ? a_output_scale : b_output_scale;
    // 确保最大输出比例因子max_output_scale在2^-14到2^8的范围内
    assert(max_output_scale >= 0x1.0p-14f);
    assert(max_output_scale < 0x1.0p+8f);
    // 将最大输出比例因子max_output_scale转换为32位浮点数表示，计算其指数部分
    const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
    const int32_t max_scale_exponent = (int32_t)(max_scale_bits >> 23) - 127;
    /* 移位范围为[13, 31] */
    const uint32_t shift = (uint32_t)(21 - max_scale_exponent);
    // 确保移位量shift在0到31之间
    assert(shift < 32);
    assert(shift >= 13);
    
    const float scale_multiplier =
        fp32_from_bits((uint32_t)(21 - max_scale_exponent + 127) << 23);
    
    /* 乘数在[0, 2**22)范围内，最大乘数在[2**21, 2**22)范围内 */
    // 计算a_multiplier和b_multiplier，将a_output_scale和b_output_scale转换为整数乘以scale_multiplier
    const uint32_t a_multiplier =
        (uint32_t)(int32_t)lrintf(a_output_scale * scale_multiplier);
    const uint32_t b_multiplier =
        (uint32_t)(int32_t)lrintf(b_output_scale * scale_multiplier);
    // 确保较大的乘数（a_multiplier和b_multiplier中较大者）大于等于0x00200000
    assert(
        (a_multiplier > b_multiplier ? a_multiplier : b_multiplier) >=
        UINT32_C(0x00200000));
    // 确保a_multiplier和b_multiplier小于0x00400000
    assert(a_multiplier < UINT32_C(0x00400000));
    assert(b_multiplier < UINT32_C(0x00400000));
    
    union pytorch_qnnp_add_quantization_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  // 计算余数掩码，用于将数值限制在一个特定的范围内
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  // 计算余数阈值，用于判断是否超过阈值
  const uint32_t remainder_threshold = remainder_mask >> 1;
  // 计算零点乘积，用于后续量化操作
  const int32_t zero_point_product = (int32_t) -
      (a_multiplier * (uint32_t)a_zero_point +
       b_multiplier * (uint32_t)b_zero_point);
  // 将零点乘积值写入 SSE2 参数结构体的每个元素
  for (uint32_t i = 0; i < 4; i++) {
    params.sse2.zero_point_product[i] = zero_point_product;
  }
  // 将输出的零点值转换并写入 SSE2 参数结构体的每个元素
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.y_zero_point[i] = (int16_t)(uint16_t)output_zero_point;
  }
  // 将乘法因子的低位和高位分别写入 SSE2 参数结构体的对应元素
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.a_multiplier_lo[i] = (uint16_t)(uint32_t)a_multiplier;
    params.sse2.a_multiplier_hi[i] = (uint16_t)((uint32_t)a_multiplier >> 16);
    params.sse2.b_multiplier_lo[i] = (uint16_t)(uint32_t)b_multiplier;
    params.sse2.b_multiplier_hi[i] = (uint16_t)((uint32_t)b_multiplier >> 16);
  }
  // 将乘法因子写入 SSE2 参数结构体
  params.sse2.a_multiplier = a_multiplier;
  params.sse2.b_multiplier = b_multiplier;
  // 将余数掩码和余数阈值写入 SSE2 参数结构体的每个元素
  for (uint32_t i = 0; i < 4; i++) {
    params.sse2.remainder_mask[i] = remainder_mask;
    params.sse2.remainder_threshold[i] = remainder_threshold;
  }
  // 将移位值写入 SSE2 参数结构体
  params.sse2.shift = shift;
  // 将输出的最大值和最小值写入 SSE2 参数结构体的每个元素
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.y_max[i] = output_max;
    params.sse2.y_min[i] = output_min;
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  // 将输入的零点值写入 NEON 参数结构体
  params.neon.a_zero_point = a_zero_point;
  params.neon.b_zero_point = b_zero_point;
  // 将输出的零点值转换并写入 NEON 参数结构体
  params.neon.y_zero_point = (int16_t)(uint16_t)output_zero_point;
  // 将乘法因子转换为有符号整数并写入 NEON 参数结构体
  params.neon.a_multiplier = (int32_t)a_multiplier;
  params.neon.b_multiplier = (int32_t)b_multiplier;
  // 将移位值取负数后写入 NEON 参数结构体
  params.neon.right_shift = (int32_t)-shift;
  // 将输出的最大值和最小值写入 NEON 参数结构体
  params.neon.y_max = output_max;
  params.neon.y_min = output_min;
#else
  // 计算余数掩码，用于将数值限制在一个特定的范围内
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  // 计算余数阈值，用于判断是否超过阈值
  const uint32_t remainder_threshold = remainder_mask >> 1;
  // 计算零点乘积，用于后续量化操作
  params.scalar.zero_point_product = (int32_t) -
      (a_multiplier * (uint32_t)a_zero_point +
       b_multiplier * (uint32_t)b_zero_point);
  // 将乘法因子写入 Scalar 参数结构体
  params.scalar.a_multiplier = a_multiplier;
  params.scalar.b_multiplier = b_multiplier;
  // 将余数掩码和余数阈值写入 Scalar 参数结构体
  params.scalar.remainder_mask = (int32_t)remainder_mask;
  params.scalar.remainder_threshold = (int32_t)remainder_threshold;
  // 将移位值写入 Scalar 参数结构体
  params.scalar.shift = shift;
  // 将输出的零点值转换并写入 Scalar 参数结构体
  params.scalar.y_zero_point = (int32_t)(uint32_t)output_zero_point;
  // 将输出的最大值和最小值转换并写入 Scalar 参数结构体
  params.scalar.y_max = (int32_t)(uint32_t)output_max;
  params.scalar.y_min = (int32_t)(uint32_t)output_min;
#endif
  // 返回参数结构体
  return params;
}
    uint8_t output_max) {
  // 断言输出缩放因子大于等于 2^-10
  assert(a_output_scale >= 0x1.0p-10f);
  // 断言输出缩放因子大于等于 2^-10
  assert(b_output_scale >= 0x1.0p-10f);
  // 断言输出缩放因子小于 2^8
  assert(a_output_scale < 0x1.0p+8f);
  // 断言输出缩放因子小于 2^8
  assert(b_output_scale < 0x1.0p+8f);

  /* Compute requantization parameters */
  // 计算最大输出缩放因子
  const float max_output_scale =
      a_output_scale > b_output_scale ? a_output_scale : b_output_scale;
  // 断言最大输出缩放因子大于等于 2^-10
  assert(max_output_scale >= 0x1.0p-10f);
  // 断言最大输出缩放因子小于 2^8
  assert(max_output_scale < 0x1.0p+8f);
  // 将最大缩放因子转换为单精度浮点数位表示
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  // 计算最大缩放因子的指数部分
  const int32_t max_scale_exponent = (int32_t)(max_scale_bits >> 23) - 127;
  /* Shift is in [13, 31] range */
  // 计算位移，确保在 [13, 31] 范围内
  const uint32_t shift = (uint32_t)(21 - max_scale_exponent);
  // 断言位移小于 32
  assert(shift < 32);
  // 断言位移大于等于 13
  assert(shift >= 13);

  /* Multipliers are in [0, 2**22) range, largest multiplier is in [2**21,
   * 2**22) range */
  // 计算乘数，确保在 [0, 2^22) 范围内，最大的乘数在 [2^21, 2^22) 范围内
  const uint32_t a_multiplier = (uint32_t)(int32_t)lrintf(
      fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const uint32_t b_multiplier = (uint32_t)(int32_t)lrintf(
      fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  // 断言乘数大于等于 2^21
  assert(
      (a_multiplier > b_multiplier ? a_multiplier : b_multiplier) >=
      UINT32_C(0x00200000));
  // 断言乘数小于 2^22
  assert(a_multiplier < UINT32_C(0x00400000));
  assert(b_multiplier < UINT32_C(0x00400000));

  union pytorch_qnnp_add_quantization_params params;
  // 计算余数掩码
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  // 计算余数阈值
  const uint32_t remainder_threshold = remainder_mask >> 1;
  // 计算零点乘积并保存在参数结构中
  params.scalar.zero_point_product = (int32_t) -
      (a_multiplier * (uint32_t)a_zero_point +
       b_multiplier * (uint32_t)b_zero_point);
  // 保存乘数到参数结构中
  params.scalar.a_multiplier = a_multiplier;
  params.scalar.b_multiplier = b_multiplier;
  // 保存余数掩码到参数结构中
  params.scalar.remainder_mask = (int32_t)remainder_mask;
  // 保存余数阈值到参数结构中
  params.scalar.remainder_threshold = (int32_t)remainder_threshold;
  // 保存位移到参数结构中
  params.scalar.shift = shift;
  // 保存输出的零点到参数结构中
  params.scalar.y_zero_point = (int32_t)(uint32_t)output_zero_point;
  // 保存输出的最大值到参数结构中
  params.scalar.y_max = (int32_t)(uint32_t)output_max;
  // 保存输出的最小值到参数结构中
  params.scalar.y_min = (int32_t)(uint32_t)output_min;
  // 返回计算得到的参数结构
  return params;
// 对 Q31 数据进行重新量化，返回一个 8 位无符号整数
static inline uint8_t pytorch_qnnp_q31_requantize(
    int32_t n,  // 输入整数 n
    union pytorch_qnnp_q31_requantization_params params) {  // 量化参数结构体
  // 计算 n 与 multiplier 的乘积，并将结果向右移动 31 位，最后将其转换为 Q31 格式的整数
  const int64_t product = (int64_t)n * (int64_t)params.scalar.multiplier;
  const int32_t q31product =
      (int32_t)(uint32_t)((uint64_t)(product + INT64_C(0x40000000)) >> 31);
  // 计算余数，根据阈值调整结果 n
  const int32_t remainder =
      (q31product & params.scalar.remainder_mask) - (int32_t)(n < 0);
  n = asr_s32(q31product, params.scalar.shift) +
      (int32_t)(remainder > params.scalar.remainder_threshold);
  // 如果 n 小于最小的负零点，则将 n 设置为最小的负零点
  if (n < params.scalar.min_less_zero_point) {
    n = params.scalar.min_less_zero_point;
  }
  // 如果 n 大于最大的负零点，则将 n 设置为最大的负零点
  if (n > params.scalar.max_less_zero_point) {
    n = params.scalar.max_less_zero_point;
  }

  // 返回最终结果，加上零点，转换为 8 位无符号整数
  return (uint8_t)(n + params.scalar.zero_point);
}

// 对 FP32 数据进行重新量化，返回一个 8 位无符号整数
static inline uint8_t pytorch_qnnp_fp32_requantize(
    int32_t n,  // 输入整数 n
    union pytorch_qnnp_fp32_requantization_params params,
    int32_t output_channel_index) {  // 输出通道索引
  // 计算输出范围的最小值和最大值
  const long lmin =
      (long)((int32_t)(uint32_t)params.scalar.output_min -
          (int32_t)(uint32_t)params.scalar.output_zero_point);
  const long lmax =
      (long)((int32_t)(uint32_t)params.scalar.output_max -
          (int32_t)(uint32_t)params.scalar.output_zero_point);

  // 根据输出通道索引，缩放输入 n，并四舍五入到最接近的整数
  const float n_scaled = (float)n * params.scalar.scales[output_channel_index];
  const long n_rounded = lrintf(n_scaled);
  // 将结果限制在输出范围内
  const int32_t n_clamped = (int32_t)(
      n_rounded < lmin ? lmin : n_rounded > lmax ? lmax : n_rounded);
  // 加上零点，并转换为 8 位无符号整数
  const int32_t n_biased =
      n_clamped + (int32_t)(uint32_t)params.scalar.output_zero_point;

  return (uint8_t)n_biased;
}

// 对 FP32 数据进行特殊的重新量化，返回一个 8 位无符号整数
static inline uint8_t pytorch_qnnp_fp32_requantize_magic(
    int32_t n,  // 输入整数 n
    union pytorch_qnnp_fp32_requantization_params params,
    int32_t output_channel_index) {  // 输出通道索引
  // 获取量化参数中的特殊值
  const float fmin = params.scalar.min_less_zero_point;
  const float fmax = params.scalar.max_less_zero_point;
  const float fmagic = params.scalar.magic;
  const int32_t imagic = params.scalar.magic_less_zero_point;

  // 缩放输入 n，并将结果限制在给定的范围内
  const float n_scaled = (float)n * params.scalar.scales[output_channel_index];
  const float n_clamped =
      n_scaled < fmin ? fmin : n_scaled > fmax ? fmax : n_scaled;
  // 加上魔术数，并将结果转换为整数，再减去 imagic
  const int32_t n_biased = (int32_t)fp32_to_bits(n_clamped + fmagic) - imagic;

  // 返回最终结果，转换为 8 位无符号整数
  return (uint8_t)n_biased;
}

// 对平均池化操作的结果进行量化，返回一个 8 位无符号整数
static inline uint8_t pytorch_qnnp_avgpool_quantize(
    int32_t n,  // 输入整数 n
    union pytorch_qnnp_avgpool_quantization_params params) {  // 量化参数结构体
  // 缩放输入 n，并加上输出零点，四舍五入到最接近的整数
  const float scaled_n = ((float)n) * params.scalar.scale;
  int32_t n_rounded = (int32_t)lrintf(scaled_n) + params.scalar.output_zero_point;

  // 将结果限制在输出范围内
  const int32_t lmin =
      (int32_t)(uint32_t)params.scalar.output_min;
  const int32_t lmax =
      (int32_t)(uint32_t)params.scalar.output_max;
  n_rounded = (
      n_rounded < lmin ? lmin : n_rounded > lmax ? lmax : n_rounded);

  // 返回最终结果，转换为 8 位无符号整数
  return (uint8_t)n_rounded;
}

// 对两个输入值进行加法操作，并返回一个 8 位无符号整数
static inline uint8_t pytorch_qnnp_add_quantize(
    uint8_t a,  // 第一个输入值
    uint8_t b,  // 第二个输入值
    /* 计算加权乘积并累加 */
    int32_t acc = params.scalar.zero_point_product +
        (int32_t)((uint32_t)a * params.scalar.a_multiplier) +
        (int32_t)((uint32_t)b * params.scalar.b_multiplier);

    /* 右移并进行四舍五入 */
    const int32_t rem = (acc & params.scalar.remainder_mask) - (int32_t)(acc < 0);
    acc = asr_s32(acc, params.scalar.shift) +
        (int32_t)(rem > params.scalar.remainder_threshold);

    /* 截断并添加输出零点 */
    int32_t y = acc + params.scalar.y_zero_point;
    if (y >= params.scalar.y_max) {
        y = params.scalar.y_max;
    }
    if (y <= params.scalar.y_min) {
        y = params.scalar.y_min;
    }
    return (uint8_t)y;
}



# 这是一个单独的右花括号，用于闭合一个代码块。
```