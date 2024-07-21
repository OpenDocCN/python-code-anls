# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\q8dwconv.h`

```py
/*
 * 版权所有（c）Facebook，Inc.及其关联公司。
 * 保留所有权利。
 *
 * 本源代码在根目录的LICENSE文件中按BSD风格许可证进行许可。
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

// 声明 PyTorch Q8UPDWCONV 的微内核函数，使用 NEON 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_up8x9__neon(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8UPDWCONV 的按通道量化版本的微内核函数，使用 NEON 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_up8x9_per_channel__neon(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8UPDWCONV 的微内核函数，使用 AArch32 NEON 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_up8x9__aarch32_neon(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8UPDWCONV 的按通道量化版本的微内核函数，使用 AArch32 NEON 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8UPDWCONV 的微内核函数，使用 SSE2 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_up8x9__sse2(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8UPDWCONV 的按通道量化版本的微内核函数，使用 SSE2 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8MPDWCONV 的微内核函数，使用 NEON 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_mp8x25__neon(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    int32_t* buffer,                                    // 缓冲区指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8MPDWCONV 的按通道量化版本的微内核函数，使用 NEON 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    int32_t* buffer,                                    // 缓冲区指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8MPDWCONV 的微内核函数，使用 SSE2 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_mp8x25__sse2(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    int32_t* buffer,                                    // 缓冲区指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指针

// 声明 PyTorch Q8MPDWCONV 的按通道量化版本的微内核函数，使用 SSE2 指令集优化
PYTORCH_QNNP_INTERNAL void pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2(
    size_t channels,                                    // 输入通道数
    size_t output_width,                                // 输出宽度
    const uint8_t** input,                              // 输入数据数组
    const void* weights,                                // 权重数据指针
    int32_t* buffer,                                    // 缓冲区指针
    uint8_t* output,                                    // 输出数据指针
    size_t input_stride,                                // 输入跨度
    size_t output_increment,                            // 输出增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params); // 量化参数结构体的指
#define DECLARE_PYTORCH_Q8MPDWCONV_3D_UKERNEL_FUNCTION(fn_name) \  // 宏定义：声明一个 PyTorch 的量化深度可分离三维卷积的微内核函数
  PYTORCH_QNNP_INTERNAL void fn_name(                           \  // 定义一个内部函数 fn_name，接受以下参数：
      size_t channels,                                          \  // 通道数
      size_t output_height,                                     \  // 输出高度
      size_t output_width,                                      \  // 输出宽度
      const uint8_t** input,                                    \  // 输入数据的指针（二维数组）
      const void* weights,                                      \  // 权重数据的指针
      int32_t* buffer,                                          \  // 缓冲区数据的指针
      uint8_t* output,                                          \  // 输出数据的指针
      size_t input_row_stride,                                  \  // 输入行步幅
      size_t input_col_stride,                                  \  // 输入列步幅
      size_t output_increment,                                  \  // 输出增量
      const union pytorch_qnnp_conv_quantization_params* quantization_params);  // PyTorch QNNPACK 卷积量化参数的联合体

DECLARE_PYTORCH_Q8MPDWCONV_3D_UKERNEL_FUNCTION(                 // 声明 PyTorch 量化 Q8MPDWCONV 三维卷积的微内核函数
    pytorch_q8dwconv_ukernel_mp8x27__neon)                      // 使用 NEON 加速的微内核函数声明

DECLARE_PYTORCH_Q8MPDWCONV_3D_UKERNEL_FUNCTION(                 // 声明 PyTorch 量化 Q8MPDWCONV 三维卷积的微内核函数
    pytorch_q8dwconv_ukernel_mp8x27__sse2)                      // 使用 SSE2 加速的微内核函数声明

#ifdef __cplusplus
} /* extern "C" */                                             // C++ 外部接口声明结束
#endif
```