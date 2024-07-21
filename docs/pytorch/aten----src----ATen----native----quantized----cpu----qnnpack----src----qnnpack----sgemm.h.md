# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\sgemm.h`

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

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

// 定义一个宏，用于声明 PyTorch SGEMM 矩阵乘法的微内核函数
#define DECLARE_PYTORCH_SGEMM_UKERNEL_FUNCTION(fn_name) \
  // 声明函数 fn_name，该函数用于执行矩阵乘法的微内核操作
  PYTORCH_QNNP_INTERNAL void fn_name( \
      size_t mr,                      \
      size_t nr,                      \
      size_t k,                       \
      const float* a,                 \
      size_t a_stride,                \
      const float* w,                 \
      float* c,                       \
      size_t c_stride,                \
      const struct pytorch_qnnp_fp32_clamping_params* clamping_params);

// 声明不同的 PyTorch SGEMM 矩阵乘法的微内核函数，使用 NEON 指令集加速
DECLARE_PYTORCH_SGEMM_UKERNEL_FUNCTION(pytorch_sgemm_ukernel_5x8__neon)
DECLARE_PYTORCH_SGEMM_UKERNEL_FUNCTION(pytorch_sgemm_ukernel_6x8__neon)
DECLARE_PYTORCH_SGEMM_UKERNEL_FUNCTION(pytorch_sgemm_ukernel_6x8__psimd)

#ifdef __cplusplus
} /* extern "C" */
#endif
```