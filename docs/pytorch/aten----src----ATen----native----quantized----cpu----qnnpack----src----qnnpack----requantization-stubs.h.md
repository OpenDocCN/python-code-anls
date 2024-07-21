# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\requantization-stubs.h`

```py
/*
 * 版权所有（c）Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的 LICENSE 文件中的 BSD 风格许可证许可。
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/params.h>

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*pytorch_requantization_function)(
    size_t n,                                   // 函数参数：数据量大小
    const int32_t* input,                       // 函数参数：输入数据数组
    float scale,                                // 函数参数：缩放因子
    uint8_t zero_point,                         // 函数参数：零点
    uint8_t qmin,                               // 函数参数：最小量化值
    uint8_t qmax,                               // 函数参数：最大量化值
    uint8_t* output);                           // 函数参数：输出数据数组

#define DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(fn_name) \
  void fn_name(                                  // 宏定义：声明特定的重量化函数
      size_t n,                                  // 函数参数：数据量大小
      const int32_t* input,                      // 函数参数：输入数据数组
      float scale,                               // 函数参数：缩放因子
      uint8_t zero_point,                        // 函数参数：零点
      uint8_t qmin,                              // 函数参数：最小量化值
      uint8_t qmax,                              // 函数参数：最大量化值
      uint8_t* output);

DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(
    pytorch_qnnp_requantize_precise__scalar_unsigned32)   // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(
    pytorch_qnnp_requantize_precise__scalar_unsigned64)   // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(
    pytorch_qnnp_requantize_precise__scalar_signed64)     // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__sse2)    // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__ssse3)   // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__sse4)    // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__neon)    // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_precise__psimd)   // 声明具体的重量化函数

DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__scalar_lrintf)   // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__scalar_magic)    // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__sse2)           // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__neon)           // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_fp32__psimd)          // 声明具体的重量化函数

DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__scalar)          // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__sse2)            // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__ssse3)           // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__sse4)            // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__neon)            // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_q31__psimd)           // 声明具体的重量化函数

DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__scalar)     // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__sse2)       // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__ssse3)      // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__sse4)       // 声明具体的重量化函数
DECLARE_PYTORCH_REQUANTIZATION_FUNCTION(pytorch_qnnp_requantize_gemmlowp__neon)       // 声明具体的重量化函数

#ifdef __cplusplus
} /* extern "C" */
#endif
```