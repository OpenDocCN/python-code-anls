# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\x8lut.h`

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

// 定义宏 DECLARE_PYTORCH_X8LUT_UKERNEL_FUNCTION，用于声明一个名为 fn_name 的函数指针类型
#define DECLARE_PYTORCH_X8LUT_UKERNEL_FUNCTION(fn_name) \
  // 声明一个名为 fn_name 的函数，该函数接受三个参数，分别是数组长度 n，输入数组 x，查找表 t 和输出数组 y
  PYTORCH_QNNP_INTERNAL void fn_name(           \
      size_t n, const uint8_t* x, const uint8_t* t, uint8_t* y);

// 使用 C++ 编译器时，将按照 C 语言的方式导出声明
DECLARE_PYTORCH_X8LUT_UKERNEL_FUNCTION(pytorch_x8lut_ukernel__scalar)

#ifdef __cplusplus
} /* extern "C" */
#endif
```