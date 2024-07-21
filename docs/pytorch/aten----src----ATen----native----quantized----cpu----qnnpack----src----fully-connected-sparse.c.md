# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\fully-connected-sparse.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

// 定义一个函数，创建稀疏量化的全连接运算符
enum pytorch_qnnp_status pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const void* kernel_col_indices,
    const void* kernel_row_values,
    const uint8_t* kernel_values,
    const uint32_t kernel_row_block_size,
    const uint32_t kernel_col_block_size,
    enum pytorch_qnnp_sparse_matrix_indices_dtype kernel_indices_dtype,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool use_prepack_kernel,
    pytorch_qnnp_operator_t* fully_connected_out) {

  // 初始化变量
  pytorch_qnnp_operator_t fully_connected = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  // 检查 QNNPACK 是否初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  // 检查是否有不支持的参数
  status = pytorch_qnnp_status_unsupported_parameter;

  // 检查每个输出通道的量化重量化比例因子是否有效
  for (int i = 0; i < output_channels; ++i) {
    if (requantization_scales[i] <= 0.0f ||
        !isnormal(requantization_scales[i])) {
      pytorch_qnnp_log_error(
          "failed to create fully connected operator with %.7g requantization scale: scale must be finite and positive",
          requantization_scales[i]);
      goto error;
    }
  }

  // 分配内存以存储 fully_connected 运算符
  status = pytorch_qnnp_status_out_of_memory;
  fully_connected = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (fully_connected == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  // 检查特定的稀疏模式是否支持，如果不支持则报错
  if (kernel_row_block_size == 8 && kernel_col_block_size == 1) {
    // 这是为了限制 SSE2 上的 8x1 稀疏模式，因为我们还没有实现支持这种模式的 SSE2 内核
    if (pytorch_qnnp_params.q8gemm_sparse_c8x1.packA == NULL) {
      status = pytorch_qnnp_status_invalid_parameter;
      goto error;
    }
  }

  // 设置稀疏矩阵的索引数据类型，并复制索引数据
  fully_connected->sparse_matrix.indices_dtype = kernel_indices_dtype;
  switch (kernel_indices_dtype) {
    case pytorch_qnnp_sparse_matrix_indices_dtype_uint32_t:
      fully_connected->sparse_matrix.col_indices_w32 =
          (const uint32_t*)kernel_col_indices;
      fully_connected->sparse_matrix.row_values_w32 =
          (const uint32_t*)kernel_row_values;
      break;
    // 添加其他索引数据类型的处理逻辑
  }


继续添加更多的注释以覆盖整个代码块。
    case pytorch_qnnp_sparse_matrix_indices_dtype_uint16_t:
      // 如果稀疏矩阵的索引类型是 uint16_t
      fully_connected->sparse_matrix.col_indices_w16 =
          (const uint16_t*)kernel_col_indices;
      fully_connected->sparse_matrix.row_values_w16 =
          (const uint16_t*)kernel_row_values;
      break;
    case pytorch_qnnp_sparse_matrix_indices_dtype_uint8_t:
      // 如果稀疏矩阵的索引类型是 uint8_t
      fully_connected->sparse_matrix.col_indices_w8 =
          (const uint8_t*)kernel_col_indices;
      fully_connected->sparse_matrix.row_values_w8 =
          (const uint8_t*)kernel_row_values;
      break;
    case pytorch_qnnp_sparse_matrix_indices_dtype_invalid:
      // 如果指定了无效的索引类型，返回无效参数状态
      status = pytorch_qnnp_status_invalid_parameter;
      pytorch_qnnp_log_error(
          "Invalid indices dtype specified for qnnpack fully connected sparse");
      // 跳转到错误处理标签
      goto error;
  }

  // 设置稀疏矩阵的值和块大小
  fully_connected->sparse_matrix.values = kernel_values;
  fully_connected->sparse_matrix.row_block_size = kernel_row_block_size;
  fully_connected->sparse_matrix.col_block_size = kernel_col_block_size;

  // 设置全连接层的组信息
  fully_connected->groups = 1;
  fully_connected->group_input_channels = input_channels;
  fully_connected->group_output_channels = output_channels;

  // 设置全连接层的 kernel_zero_point
  fully_connected->kernel_zero_point = kernel_zero_points[0];

  // 设置动态卷积量化参数
  fully_connected->dynamic_conv_quantization_params.input_zero_point =
    input_zero_point;
  fully_connected->dynamic_conv_quantization_params.kernel_zero_points =
    kernel_zero_points;
  fully_connected->dynamic_conv_quantization_params.multipliers =
    requantization_scales;

  // 总是使用基于预打包的 kernel
  fully_connected->ukernel_type = pytorch_qnnp_ukernel_type_gemm_prepackA_sparse_dq;
  // 设置格式为 quint8
  fully_connected->format = pytorch_qnnp_format_quint8;

  // 将 fully_connected 设置为输出参数
  *fully_connected_out = fully_connected;
  // 返回成功状态
  return pytorch_qnnp_status_success;
# 删除指定的 PyTorch QNNPACK 程序运算符对象
pytorch_qnnp_delete_operator(fully_connected);
# 返回函数的执行状态
return status;
}

# 设置稀疏量化整数全连接操作的 QNNPACK 运算符对象
enum pytorch_qnnp_status pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
    pytorch_qnnp_operator_t fully_connected,  # QNNPACK 运算符对象指针
    size_t batch_size,  # 批处理大小
    const uint8_t* input,  # 输入数据的指针（uint8_t 类型）
    size_t input_stride,  # 输入数据的步幅
    const float* bias,  # 偏置数据的指针（float 类型）
    float* output,  # 输出数据的指针（float 类型）
    size_t output_stride) {  # 输出数据的步幅
  if (!pytorch_qnnp_params.initialized) {
    # 如果 QNNPACK 没有正确初始化，则记录错误并返回未初始化状态
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    # 如果批处理大小为 0，则设置运算符对象的批处理大小为 0 并返回成功状态
    fully_connected->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  # 设置运算符对象的批处理大小为 1
  fully_connected->batch_size = 1;
  # 设置输入高度为 batch_size，宽度为 1
  fully_connected->input_height = batch_size;
  fully_connected->input_width = 1;
  # 设置输入数据指针和像素步幅
  fully_connected->input = input;
  fully_connected->input_pixel_stride = input_stride;

  # 设置偏置数据指针
  fully_connected->bias = bias;

  # 设置输出高度为 batch_size，宽度为 1
  fully_connected->output_height = batch_size;
  fully_connected->output_width = 1;
  # 设置输出数据指针和像素步幅
  fully_connected->output = output;
  fully_connected->output_pixel_stride = output_stride;

  # 返回成功状态
  return pytorch_qnnp_status_success;
}
```