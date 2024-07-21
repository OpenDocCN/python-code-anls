# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\sdwconv.h`

```py
/*
 * 版权所有 (c) Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据位于源树根目录中的 LICENSE 文件中所述的 BSD 风格许可证许可。
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

// 定义声明 PyTorch 单步深度可分离卷积微内核函数的宏
#define DECLARE_PYTORCH_SUPDWCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(               \
      size_t channels,                              \
      size_t output_width,                          \
      const float** input,                          \
      const float* weights,                         \
      float* output,                                \
      size_t input_stride,                          \
      size_t output_increment,                      \
      const struct pytorch_qnnp_fp32_clamping_params* clamping_params);

// 声明 PyTorch 单步深度可分离卷积微内核函数，针对 4x9 窗口的 PSIMD 实现
DECLARE_PYTORCH_SUPDWCONV_UKERNEL_FUNCTION(pytorch_sdwconv_ukernel_up4x9__psimd)

// 定义声明 PyTorch 多步深度可分离卷积微内核函数的宏
#define DECLARE_PYTORCH_SMPDWCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(               \
      size_t channels,                              \
      size_t output_width,                          \
      const uint8_t** input,                        \
      const void* weights,                          \
      int32_t* buffer,                              \
      uint8_t* output,                              \
      size_t input_stride,                          \
      size_t output_increment,                      \
      const struct pytorch_qnnp_fp32_clamping_params* clamping_params);

#ifdef __cplusplus
} /* extern "C" */
#endif
```