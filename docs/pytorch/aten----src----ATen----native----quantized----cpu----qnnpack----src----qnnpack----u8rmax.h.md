# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\u8rmax.h`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// 定义了一个预处理宏，用于声明两个函数，实现在不同硬件平台上的特化
#pragma once

// 包含必要的头文件，定义了一些数据类型和函数原型
#include <stddef.h>
#include <stdint.h>

// 包含自定义头文件，这些头文件可能定义了一些通用的宏和函数
#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
// 如果是 C++ 环境，则按照 C 语言的方式进行编译
extern "C" {
#endif

// 定义一个预处理宏，用于声明一个特定函数类型的函数原型
#define DECLARE_PYTORCH_U8RMAX_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL uint8_t fn_name(size_t n, const uint8_t* x);

// 声明两个函数，分别是针对 NEON 和 SSE2 架构的优化函数
DECLARE_PYTORCH_U8RMAX_UKERNEL_FUNCTION(pytorch_u8rmax_ukernel__neon)
DECLARE_PYTORCH_U8RMAX_UKERNEL_FUNCTION(pytorch_u8rmax_ukernel__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
```