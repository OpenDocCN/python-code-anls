# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\q8gemm.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

// 定义宏，声明一个特定的 Q8 矩阵乘法 micro-kernel 函数原型
#define DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      size_t k,                                  \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const void* w,                             \
      uint8_t* c,                                \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

// 声明一系列特定的 Q8 矩阵乘法 micro-kernel 函数
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_3x3c8__neon)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_2x4c8__neon)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x8__neon)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_6x4__neon)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_8x8__neon)

// 声明特定的 Q8 矩阵乘法 micro-kernel 函数，使用 AArch32 NEON 指令集
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x8__aarch32_neon)

// 声明特定的 Q8 矩阵乘法 micro-kernel 函数，使用 AArch64 NEON 指令集
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_8x8__aarch64_neon)

// 声明特定的 Q8 矩阵乘法 micro-kernel 函数，使用 SSE2 指令集
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_2x4c8__sse2)
DECLARE_PYTORCH_Q8GEMM_UKERNEL_FUNCTION(pytorch_q8gemm_ukernel_4x4c2__sse2)

// 定义宏，声明一个特定的动态量化 Q8 矩阵乘法 micro-kernel 函数原型
#define DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      size_t k,                                  \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const void* w,                             \
      const float* b,                            \
      float* c,                                  \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);

// 声明特定的动态量化 Q8 矩阵乘法 micro-kernel 函数
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(pytorch_q8gemm_dq_ukernel_4x8__neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_UKERNEL_FUNCTION(pytorch_q8gemm_dq_ukernel_4x4c2__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
// 定义宏 DECLARE_PYTORCH_Q8GEMM_XZP_UKERNEL_FUNCTION，用于声明 PyTorch 的 Q8GEMM XZP UKERNEL 函数
#define DECLARE_PYTORCH_Q8GEMM_XZP_UKERNEL_FUNCTION(fn_name)      \
  // 声明函数 fn_name，该函数实现了 Q8GEMM XZP UKERNEL 算法
  PYTORCH_QNNP_INTERNAL void fn_name(                     \
      // 输入参数 mr，表示行向量寄存器数目
      size_t mr,                                          \
      // 输入参数 nr，表示列向量寄存器数目
      size_t nr,                                          \
      // 输入参数 k，表示矩阵乘法的维度大小
      size_t k,                                           \
      // 输入参数 a，表示矩阵 A 的指针，类型为 uint8_t*
      const uint8_t* a,                                   \
      // 输入参数 a_stride，表示矩阵 A 的行步长
      size_t a_stride,                                    \
      // 输入参数 a_sum，表示矩阵 A 向量和的指针，类型为 int32_t*
      const int32_t* a_sum,                               \
      // 输入参数 w，表示权重矩阵 W 的指针，任意类型
      const void* w,                                      \
      // 输出参数 c，表示输出矩阵 C 的指针，类型为 uint8_t*
      uint8_t* c,                                         \
      // 输出参数 c_stride，表示矩阵 C 的行步长
      size_t c_stride,                                    \
      // 输入参数 requantization_params，表示重新量化参数的联合体指针
      const union pytorch_qnnp_q31_requantization_params* \
          requantization_params);

// 声明 PyTorch Q8GEMM XZP UKERNEL 函数 pytorch_q8gemm_xzp_ukernel_4x8c2__neon
DECLARE_PYTORCH_Q8GEMM_XZP_UKERNEL_FUNCTION(pytorch_q8gemm_xzp_ukernel_4x8c2__neon)
// 声明 PyTorch Q8GEMM XZP UKERNEL 函数 pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon
DECLARE_PYTORCH_Q8GEMM_XZP_UKERNEL_FUNCTION(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon)

// 声明 PyTorch Q8SUMROWS UKERNEL 函数 pytorch_q8sumrows_ukernel_4x__neon
PYTORCH_QNNP_INTERNAL void pytorch_q8sumrows_ukernel_4x__neon(
    // 输入参数 a，表示输入矩阵 A 的指针，类型为 uint8_t*
    const uint8_t* a,
    // 输入参数 m，表示矩阵 A 的行数
    size_t m,
    // 输入参数 k，表示矩阵 A 的列数和矩阵 B 的行数
    size_t k,
    // 输入参数 stride，表示矩阵 A 的行步长
    size_t stride,
    // 输入参数 multiplier，表示乘法运算的倍数，类型为 int32_t
    const int32_t multiplier,
    // 输出参数 row_sum，表示行和的指针，类型为 int32_t*
    int32_t* row_sum);

// 如果是 C++ 环境，则将声明放在 extern "C" 的外部
#ifdef __cplusplus
} /* extern "C" */
#endif
```