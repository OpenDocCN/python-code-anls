# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\sconv.h`

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

// 定义一个宏声明，用于声明一个 PyTorch SCONV 微核函数
#define DECLARE_PYTORCH_SCONV_UKERNEL_FUNCTION(fn_name) \
  // PyTorch QNNP 内部函数声明，该函数用于实现 SCONV 的计算
  PYTORCH_QNNP_INTERNAL void fn_name(           \
      size_t mr,                                \
      size_t nr,                                \
      size_t kc,                                \
      size_t ks,                                \
      const float** a,                          \
      const float* w,                           \
      float* c,                                 \
      size_t c_stride,                          \
      const struct pytorch_qnnp_fp32_clamping_params* params);

// 使用 C++ 的方式导出函数接口
#ifdef __cplusplus
} /* extern "C" */
#endif

// 定义 PyTorch SCONV 的微核函数，名称为 pytorch_sconv_ukernel_6x8__psimd
DECLARE_PYTORCH_SCONV_UKERNEL_FUNCTION(pytorch_sconv_ukernel_6x8__psimd)
```