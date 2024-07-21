# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\q8vadd.h`

```py
/*
 * 版权声明和许可信息，说明代码的版权归属和使用许可
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * 该源代码使用 BSD 风格许可证，许可细节在根目录下的 LICENSE 文件中可以找到
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>   // 引入 QNNPACK 库中的通用头文件
#include <qnnpack/params.h>   // 引入 QNNPACK 库中的参数头文件

#ifdef __cplusplus
extern "C" {   // C++ 兼容性声明
#endif

// 定义一个宏，用于声明 PyTorch Q8VADD UKERNEL 函数
#define DECLARE_PYTORCH_Q8VADD_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t n,                                  \
      const uint8_t* a,                          \
      const uint8_t* b,                          \
      uint8_t* y,                                \
      const union pytorch_qnnp_add_quantization_params* quantization_params);

// 声明具体的 PyTorch Q8VADD UKERNEL 函数，包括 NEON 和 SSE2 实现
DECLARE_PYTORCH_Q8VADD_UKERNEL_FUNCTION(pytorch_q8vadd_ukernel__neon)
DECLARE_PYTORCH_Q8VADD_UKERNEL_FUNCTION(pytorch_q8vadd_ukernel__sse2)

#ifdef __cplusplus
} /* extern "C" */   // 结束 C++ 兼容性声明
#endif
```