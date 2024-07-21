# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\hgemm.h`

```py
/*
 * 版权所有（c）Facebook，Inc.及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据在源代码根目录中的LICENSE文件中找到的BSD风格许可证进行许可。
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

// 定义了一个宏，用于声明 PyTorch 中的半精度 GEMM（通用矩阵乘法）微内核函数
#define DECLARE_PYTORCH_HGEMM_UKERNEL_FUNCTION(fn_name) \
  void fn_name(                                 \
      size_t mr,                                \
      size_t nr,                                \
      size_t k,                                 \
      const void* a,                            \
      size_t a_stride,                          \
      const void* w,                            \
      void* c,                                  \
      size_t c_stride,                          \
      const struct pytorch_qnnp_fp16_clamping_params* clamping_params);

// 声明了一个名为 pytorch_hgemm_ukernel_8x8__neonfp16arith 的函数
DECLARE_PYTORCH_HGEMM_UKERNEL_FUNCTION(pytorch_hgemm_ukernel_8x8__neonfp16arith)

// 声明了一个名为 pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith 的函数
DECLARE_PYTORCH_HGEMM_UKERNEL_FUNCTION(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith)

#ifdef __cplusplus
} /* extern "C" */
#endif
```