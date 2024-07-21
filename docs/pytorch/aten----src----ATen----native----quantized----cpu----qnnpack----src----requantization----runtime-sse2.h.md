# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\runtime-sse2.h`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <immintrin.h>

// 定义一个内联函数，用于在 PyTorch QNNPACK 中执行零点减法操作
PYTORCH_QNNP_INLINE __m128i
sub_zero_point(const __m128i va, const __m128i vzp) {
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
  // 如果是运行时量化，则执行零点减法操作
  return _mm_sub_epi16(va, vzp);
#else
  // 如果是设计时量化（即没有运行时量化），则直接返回输入值 va，不做任何操作
  return va;
#endif
}
```