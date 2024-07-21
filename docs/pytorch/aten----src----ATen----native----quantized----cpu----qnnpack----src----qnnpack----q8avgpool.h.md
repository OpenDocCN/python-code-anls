# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\q8avgpool.h`

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

// 定义宏，声明用于 PyTorch 的 Q8 混合精度平均池化内核函数
#define DECLARE_PYTORCH_Q8MPAVGPOOL_UKERNEL_FUNCTION(fn_name)       \
  PYTORCH_QNNP_INTERNAL void fn_name(                       \
      size_t n,                                             \
      size_t ks,                                            \
      size_t kc,                                            \
      const uint8_t** x,                                    \
      const uint8_t* zero,                                  \
      int32_t* buffer,                                      \
      uint8_t* y,                                           \
      size_t x_increment,                                   \
      size_t y_increment,                                   \
      const union pytorch_qnnp_avgpool_quantization_params* \
          quantization_params);

// 声明具体的 PyTorch Q8 混合精度平均池化内核函数，使用 NEON 指令集
DECLARE_PYTORCH_Q8MPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_mp8x9p8q__neon)
// 声明具体的 PyTorch Q8 混合精度平均池化内核函数，使用 SSE2 指令集
DECLARE_PYTORCH_Q8MPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2)

// 定义宏，声明用于 PyTorch 的 Q8 无填充平均池化内核函数
#define DECLARE_PYTORCH_Q8UPAVGPOOL_UKERNEL_FUNCTION(fn_name)       \
  PYTORCH_QNNP_INTERNAL void fn_name(                       \
      size_t n,                                             \
      size_t ks,                                            \
      size_t kc,                                            \
      const uint8_t** x,                                    \
      const uint8_t* zero,                                  \
      uint8_t* y,                                           \
      size_t x_increment,                                   \
      size_t y_increment,                                   \
      const union pytorch_qnnp_avgpool_quantization_params* \
          quantization_params);

// 声明具体的 PyTorch Q8 无填充平均池化内核函数，使用 NEON 指令集
DECLARE_PYTORCH_Q8UPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_up8x9__neon)
DECLARE_PYTORCH_Q8UPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_up8xm__neon)
// 声明具体的 PyTorch Q8 无填充平均池化内核函数，使用 SSE2 指令集
DECLARE_PYTORCH_Q8UPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_up8x9__sse2)
DECLARE_PYTORCH_Q8UPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_up8xm__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif


注释解释了每个宏的定义以及声明的函数，包括函数的参数和使用的指令集（NEON 或 SSE2）。
```