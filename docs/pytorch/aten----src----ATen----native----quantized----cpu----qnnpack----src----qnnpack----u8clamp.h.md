# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\u8clamp.h`

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

// 定义一个宏声明，用于声明两个函数原型，这两个函数执行 U8 类型数据的限制范围操作
#define DECLARE_PYTORCH_U8CLAMP_UKERNEL_FUNCTION(fn_name) \
  // 函数声明，指定函数名为 fn_name，函数返回类型为 void
  PYTORCH_QNNP_INTERNAL void fn_name(             \
      // 函数参数：大小为 n，输入数据 x 是 uint8_t 类型指针，输出数据 y 也是 uint8_t 类型指针，参数 params 是一个结构体指针
      size_t n,                                   \
      const uint8_t* x,                           \
      uint8_t* y,                                 \
      const union pytorch_qnnp_u8_clamping_params* params);

// 声明一个函数原型，函数名为 pytorch_u8clamp_ukernel__neon，执行 NEON SIMD 加速的 U8 限幅内核操作
DECLARE_PYTORCH_U8CLAMP_UKERNEL_FUNCTION(pytorch_u8clamp_ukernel__neon)
// 声明一个函数原型，函数名为 pytorch_u8clamp_ukernel__sse2，执行 SSE2 SIMD 加速的 U8 限幅内核操作
DECLARE_PYTORCH_U8CLAMP_UKERNEL_FUNCTION(pytorch_u8clamp_ukernel__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
```