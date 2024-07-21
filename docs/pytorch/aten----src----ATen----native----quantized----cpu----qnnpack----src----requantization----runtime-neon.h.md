# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\runtime-neon.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arm_neon.h>

// 定义一个内联函数，用于从输入向量中减去零点偏置，并返回结果
PYTORCH_QNNP_INLINE uint16x8_t
sub_zero_point(const uint8x8_t va, const uint8x8_t vzp) {
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
  // 如果是运行时量化，则执行运行时量化的减法操作
  return vsubl_u8(va, vzp);
#else
  // 如果是设计时量化，则直接将输入向量中的每个元素零扩展为16位，并返回结果
  return vmovl_u8(va);
#endif
}
```