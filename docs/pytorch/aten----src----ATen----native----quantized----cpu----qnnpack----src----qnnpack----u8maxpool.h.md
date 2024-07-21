# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\u8maxpool.h`

```
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

// 定义一个宏，声明用于 PyTorch 的 uint8 最大池化微内核函数
#define DECLARE_PYTORCH_U8MAXPOOL_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(                       \
      size_t n,                                             \
      size_t ks,                                            \
      size_t kc,                                            \
      const uint8_t** x,                                    \
      uint8_t* y,                                           \
      size_t x_increment,                                   \
      size_t y_increment,                                   \
      const union pytorch_qnnp_u8_clamping_params* params);

// 声明具体的 PyTorch uint8 最大池化微内核函数，使用 NEON 指令集
DECLARE_PYTORCH_U8MAXPOOL_UKERNEL_FUNCTION(pytorch_u8maxpool_ukernel_16x9p8q__neon)
// 声明具体的 PyTorch uint8 最大池化微内核函数，使用 SSE2 指令集
DECLARE_PYTORCH_U8MAXPOOL_UKERNEL_FUNCTION(pytorch_u8maxpool_ukernel_16x9p8q__sse2)
// 声明具体的 PyTorch uint8 最大池化微内核函数（子 16），使用 NEON 指令集
DECLARE_PYTORCH_U8MAXPOOL_UKERNEL_FUNCTION(pytorch_u8maxpool_ukernel_sub16__neon)
// 声明具体的 PyTorch uint8 最大池化微内核函数（子 16），使用 SSE2 指令集
DECLARE_PYTORCH_U8MAXPOOL_UKERNEL_FUNCTION(pytorch_u8maxpool_ukernel_sub16__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
```