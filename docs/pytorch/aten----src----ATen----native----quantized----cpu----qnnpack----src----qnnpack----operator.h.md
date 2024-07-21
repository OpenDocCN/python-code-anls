# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\operator.h`

```py
/*
 * 版权声明与许可声明
 * 版权所有，Facebook公司及其关联公司保留所有权利
 * 此源代码根目录下的LICENSE文件中包含的BSD风格许可证适用于此源代码
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/requantization.h>

// 定义 PyTorch QNNPACK 的数据格式枚举
enum pytorch_qnnp_format {
  pytorch_qnnp_format_quint8 = 0x02000000,
  pytorch_qnnp_format_float32 = 0x02020202,
  pytorch_qnnp_format_float16 = 0x01010101,
};

// 定义 PyTorch QNNPACK 的微内核类型枚举
enum pytorch_qnnp_ukernel_type {
  pytorch_qnnp_ukernel_type_none = 0,
  pytorch_qnnp_ukernel_type_add,
  pytorch_qnnp_ukernel_type_average_pooling,
  pytorch_qnnp_ukernel_type_channel_shuffle,
  pytorch_qnnp_ukernel_type_clamp,
  pytorch_qnnp_ukernel_type_conv,
  pytorch_qnnp_ukernel_type_dwconv,
  pytorch_qnnp_ukernel_type_gemm,
  pytorch_qnnp_ukernel_type_gemm_sparse_dq,
  pytorch_qnnp_ukernel_type_gemm_prepackA_sparse_dq,
  pytorch_qnnp_ukernel_type_global_average_pooling,
  pytorch_qnnp_ukernel_type_lut,
  pytorch_qnnp_ukernel_type_max_pooling,
  pytorch_qnnp_ukernel_type_softargmax,
  pytorch_qnnp_ukernel_type_xzp_gemm,
};

// 定义稀疏矩阵的结构体
typedef struct {
  union {
    const uint32_t* col_indices_w32;
    const uint16_t* col_indices_w16;
    const uint8_t* col_indices_w8;
  };
  union {
    const uint32_t* row_values_w32;
    const uint16_t* row_values_w16;
    const uint8_t* row_values_w8;
  };
  const uint8_t* values;
  uint32_t row_block_size;
  uint32_t col_block_size;
  enum pytorch_qnnp_sparse_matrix_indices_dtype indices_dtype;
} sparse_matrix_t;

// 定义 PyTorch QNNPACK 运算符的结构体
struct pytorch_qnnp_operator {
  // 输入输出尺寸及相关参数
  size_t batch_size;
  uint32_t input_padding_depth;
  uint32_t input_padding_height;
  uint32_t input_padding_width;
  uint32_t adjustment_height;
  uint32_t adjustment_width;
  uint32_t kernel_depth;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t stride_depth;
  uint32_t stride_height;
  uint32_t stride_width;
  uint32_t dilation_depth;
  uint32_t dilation_height;
  uint32_t dilation_width;
  uint32_t groups;
  size_t group_stride;
  size_t group_channels;
  size_t group_input_channels;
  size_t group_output_channels;
  size_t channels;

  // 输入数据相关信息
  size_t input_depth;
  size_t input_height;
  size_t input_width;
  size_t input_pixel_stride;
  const void* input;
  const void** indirection_buffer;
  void* a_sum;

  // 步长信息
  size_t step_depth;
  size_t step_height;
  size_t step_width;

  // 第二输入信息
  size_t input2_pixel_stride;
  const void* input2;

  // 输出信息
  size_t output_depth;
  size_t output_height;
  size_t output_width;
  size_t output_pixel_stride;
  void* output;

  // 打包权重信息
  void* packed_weights;

  // 输入输出缩放及零点信息
  float input_scale;
  float output_scale;
  uint8_t input_zero_point;
  uint8_t kernel_zero_point;
  uint8_t output_zero_point;
  uint8_t output_min;
  uint8_t output_max;

  // 有效批次尺寸及最后一个输入信息
  size_t valid_batch_size;
  size_t last_input_height;
  size_t last_input_width;
  const void* last_input;

  // 零缓冲区及指针信息
  void* zero_buffer;
  void* zero_pointer;
  void* lookup_table;

  // 重量再量化参数的联合体
  union {
    union pytorch_qnnp_q31_requantization_params requantization_params;
    // 定义结构体，用于存储不同类型的量化参数
    union pytorch_qnnp_conv_quantization_params conv_quantization_params;
    union pytorch_qnnp_add_quantization_params add_quantization_params;
    union pytorch_qnnp_avgpool_quantization_params avgpool_quantization_params;
    union pytorch_qnnp_u8_clamping_params u8_clamping_params;
  };

  // 神经内核类型，用于指示具体使用的神经网络内核
  enum pytorch_qnnp_ukernel_type ukernel_type;
  // 数据格式，表示数据的布局方式或格式
  enum pytorch_qnnp_format format;

  // 是否为每个通道应用不同的量化参数
  bool per_channel;
  // 是否进行转置操作
  bool transpose;

  // 稀疏矩阵支持
  sparse_matrix_t sparse_matrix;
  // 偏置项的指针
  const void* bias;
  // 动态卷积量化参数结构体
  struct pytorch_qnnp_conv_dynamic_quantization_params dynamic_conv_quantization_params;
  // 预打包的矩阵 A 的指针
  uint8_t* prepacked_a;
# 获取卷积操作的输出元素大小的对数（以2为底的对数）
static inline uint32_t pytorch_qnnp_operator_get_log2_output_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  # 从卷积操作的格式字段中提取输出元素大小的对数值
  return (uint32_t)(convolution->format & UINT32_C(0xFF));
}

# 获取卷积操作的输入元素大小的对数（以2为底的对数）
static inline uint32_t pytorch_qnnp_operator_get_log2_input_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  # 从卷积操作的格式字段中提取输入元素大小的对数值
  return (uint32_t)((convolution->format >> 8) & UINT32_C(0xFF));
}

# 获取卷积操作的卷积核元素大小的对数（以2为底的对数）
static inline uint32_t pytorch_qnnp_operator_get_log2_kernel_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  # 从卷积操作的格式字段中提取卷积核元素大小的对数值
  return (uint32_t)((convolution->format >> 16) & UINT32_C(0xFF));
}

# 获取卷积操作的偏置元素大小的对数（以2为底的对数）
static inline uint32_t pytorch_qnnp_operator_get_log2_bias_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  # 从卷积操作的格式字段中提取偏置元素大小的对数值
  return (uint32_t)((convolution->format >> 24) & UINT32_C(0xFF));
}
```