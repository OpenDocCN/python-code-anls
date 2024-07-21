# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\q8gemm_sparse.h`

```
/*
 * 版权声明：Facebook公司及其关联公司保留所有权利。
 *
 * 本源代码采用BSD风格许可证，可以在根目录下的LICENSE文件中找到许可证条款。
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

// 声明稀疏量化8位整数乘法矩阵乘法的微内核函数
#define DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const uint8_t* packed_w,                   \
      const uint32_t* w_row_ptr,                 \
      const uint32_t* w_block_ids_ptr,           \
      uint8_t* c,                                \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

// 定义不同架构的稀疏量化8位整数乘法1x4微内核函数
DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x8__neon)
DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_8x8__neon)

// 定义适用于特定架构的稀疏量化8位整数乘法4x8微内核函数
DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x8__aarch32_neon)

// 定义适用于特定架构的稀疏量化8位整数乘法8x8微内核函数
DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_8x8__aarch64_neon)

// 定义适用于特定架构的稀疏量化8位整数乘法4x4c2微内核函数
DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x4c2__sse2)

// 声明动态量化稀疏8位整数乘法矩阵乘法的微内核函数
#define DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const uint8_t* packed_w,                   \
      const uint32_t* w_row_ptr,                 \
      const uint32_t* w_block_ids_ptr,           \
      const float* b,                            \
      float* c,                                  \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);

// 定义动态量化稀疏8位整数乘法1x4微内核函数
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch64_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2)

// 定义动态量化稀疏8位整数乘法矩阵乘法的打包微内核函数声明
#define DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION( \
    // 定义一个宏函数，用于声明一个特定形式的 QNNP 内部函数，函数名为 fn_name
    // 该函数接受多个参数，包括矩阵块的行数 mr，列数 nr，
    // 两个输入矩阵的 packed 表示为 uint8_t 类型的指针 a_packed 和 packed_w，
    // 用于权重矩阵的行指针和块ID指针 w_row_ptr 和 w_block_ids_ptr，
    // 以及偏置向量 b 和输出向量 c，以及 c 的步进 c_stride
    // 此外还有输出通道索引 output_channel_index 和一个指向动态量化参数结构体的指针 quantization_params
    PYTORCH_QNNP_INTERNAL void fn_name(
        size_t mr,                                          // 矩阵块的行数
        size_t nr,                                          // 矩阵块的列数
        const uint8_t* a_packed,                            // 输入矩阵 A 的压缩表示
        const uint8_t* packed_w,                            // 输入矩阵 W 的压缩表示
        const w_index_dtype* w_row_ptr,                     // 权重矩阵 W 的行指针
        const w_index_dtype* w_block_ids_ptr,               // 权重矩阵 W 的块ID指针
        const float* b,                                     // 偏置向量
        float* c,                                           // 输出向量 C
        size_t c_stride,                                    // 输出向量 C 的步进
        size_t output_channel_index,                        // 输出通道索引
        const struct pytorch_qnnp_conv_dynamic_quantization_params*  // 动态量化参数结构体指针
            quantization_params
    );
// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和4x8打包矩阵，使用uint32_t类型，针对aarch32平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w32__aarch32_neon,
    uint32_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和4x8打包矩阵，使用uint16_t类型，针对aarch32平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w16__aarch32_neon,
    uint16_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和4x8打包矩阵，使用uint8_t类型，针对aarch32平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w8__aarch32_neon,
    uint8_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于8x1输出块和4x8打包矩阵，使用uint32_t类型，针对aarch32平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w32__aarch32_neon,
    uint32_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于8x1输出块和4x8打包矩阵，使用uint16_t类型，针对aarch32平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w16__aarch32_neon,
    uint16_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于8x1输出块和4x8打包矩阵，使用uint8_t类型，针对aarch32平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w8__aarch32_neon,
    uint8_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和8x4打包矩阵，使用uint32_t类型，针对aarch32平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon,
    uint32_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和8x8打包矩阵，使用uint32_t类型，针对aarch64平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w32__aarch64_neon,
    uint32_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和8x8打包矩阵，使用uint16_t类型，针对aarch64平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w16__aarch64_neon,
    uint16_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和8x8打包矩阵，使用uint8_t类型，针对aarch64平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w8__aarch64_neon,
    uint8_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于8x1输出块和8x8打包矩阵，使用uint32_t类型，针对aarch64平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w32__aarch64_neon,
    uint32_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于8x1输出块和8x8打包矩阵，使用uint16_t类型，针对aarch64平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w16__aarch64_neon,
    uint16_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于8x1输出块和8x8打包矩阵，使用uint8_t类型，针对aarch64平台的NEON加速器
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w8__aarch64_neon,
    uint8_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和8x4打包矩阵，使用uint32_t类型，针对SSE2指令集
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2,
    uint32_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用于1x4输出块和8x4打包矩阵，使用uint16_t类型，针对SSE2指令集
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2,
    uint16_t)

// 声明使用动态量化稀疏矩阵A打包的Q8GEMM内核函数，适用
#define DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      const size_t mr,                           \  // 定义宏，声明函数 fn_name，用于解析和打包稀疏矩阵的数据，内部函数
      const size_t K,                            \  // 参数 mr：行向量的长度；参数 K：矩阵的列数
      const uint8_t* a,                          \  // 参数 a：稀疏矩阵的数据指针
      const size_t a_stride,                     \  // 参数 a_stride：矩阵中每行的跨度
      uint8_t* a_packed);                        // 参数 a_packed：打包后的矩阵数据指针

DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon)   // 声明解析和打包稀疏矩阵数据的函数，使用 4x4 大小的 ARM NEON 指令集
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon)   // 声明解析和打包稀疏矩阵数据的函数，使用 8x4 大小的 ARM NEON 指令集
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon)   // 声明解析和打包稀疏矩阵数据的函数，使用 8x4 大小的 ARM AArch64 NEON 指令集
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2)           // 声明解析和打包稀疏矩阵数据的函数，使用 8x4 大小的 SSE2 指令集

#ifdef __cplusplus
} /* extern "C" */
#endif
```