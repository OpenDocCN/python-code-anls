# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\max-pooling.c`

```py
/*
 * 版权声明和许可信息
 * Facebook, Inc.及其关联公司保留所有权利。
 *
 * 此源代码根据根目录下的LICENSE文件中的BSD风格许可证进行许可。
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/common.h>
#include <qnnpack/indirection.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/params.h>

// 定义一个静态内联函数，用于计算输出维度
static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t stride_dimension) {
  // 计算有效的卷积核维度
  const size_t effective_kernel_dimension =
      (kernel_dimension - 1) * dilation_dimension + 1;
  // 计算输出维度
  return (padded_input_dimension - effective_kernel_dimension) /
      stride_dimension +
      1;
}

// 定义创建二维最大池化运算符的函数，输入数据以NHWC格式（通道数-高度-宽度）
enum pytorch_qnnp_status pytorch_qnnp_create_max_pooling2d_nhwc_u8(
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* max_pooling_out) {
  // 初始化最大池化运算符和状态
  pytorch_qnnp_operator_t max_pooling = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  // 检查QNNPACK是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    // 若未初始化，则输出错误信息
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_max_pooling2d_nhwc_u8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  // 检查参数有效性
  status = pytorch_qnnp_status_invalid_parameter;

  // 计算池化窗口的大小，检查是否为有效值
  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    // 若池化大小为0，则输出错误信息
    pytorch_qnnp_log_error(
        "failed to create max pooling with %" PRIu32 "x%" PRIu32
        " pooling size: "
        "pooling size dimensions must be non-zero",
        pooling_width,
        pooling_height);
    goto error;
  }

  // 检查是否为1x1的池化窗口，这种情况下池化没有意义
  if (pooling_size == 1) {
    // 输出相关错误信息
    pytorch_qnnp_log_error(
        "failed to create max pooling with 1 pooling element: "
        "1x1 pooling is meaningless");
    goto error;
  }

  // 检查步幅是否为有效值
  if (stride_height == 0 || stride_width == 0) {
    // 输出相关错误信息
    pytorch_qnnp_log_error(
        "failed to create max pooling with %" PRIu32 "x%" PRIu32
        " stride: "
        "stride dimensions must be non-zero",
        stride_width,
        stride_height);
    goto error;
  }

  // 检查扩展（dilation）是否为有效值
  if (dilation_height == 0 || dilation_width == 0) {
    // 输出相关错误信息
    pytorch_qnnp_log_error(
        "failed to create max pooling with %" PRIu32 "x%" PRIu32
        " dilation: "
        "dilation dimensions must be non-zero",
        dilation_width,
        dilation_height);
    goto error;
  }

  // 检查通道数是否为有效值
  if (channels == 0) {
    // 输出相关错误信息
    pytorch_qnnp_log_error(
        "failed to create max pooling with %zu channels: "
        "number of channels must be non-zero",
        channels);
        ```
    // 跳转到错误处理标签，如果分配内存失败
    goto error;
    // 分配内存失败的错误处理代码
    
    status = pytorch_qnnp_status_out_of_memory;
    
    // 分配内存以存储最大池化操作符的结构
    max_pooling = calloc(1, sizeof(struct pytorch_qnnp_operator));
    if (max_pooling == NULL) {
      // 记录错误日志，指出分配内存失败
      pytorch_qnnp_log_error(
          "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
          sizeof(struct pytorch_qnnp_operator));
      // 跳转到错误处理标签
      goto error;
    }
    
    // 设置最大池化操作符的输入填充高度和宽度
    max_pooling->input_padding_height = input_padding_height;
    max_pooling->input_padding_width = input_padding_width;
    
    // 设置最大池化操作符的核高度、宽度、步幅高度、宽度、扩展高度、宽度和通道数
    max_pooling->kernel_height = pooling_height;
    max_pooling->kernel_width = pooling_width;
    max_pooling->stride_height = stride_height;
    max_pooling->stride_width = stride_width;
    max_pooling->dilation_height = dilation_height;
    max_pooling->dilation_width = dilation_width;
    max_pooling->channels = channels;
    
    // 计算并设置最大池化操作符的无符号8位整数饱和参数
    max_pooling->u8_clamping_params =
        pytorch_qnnp_compute_u8_clamping_params(output_min, output_max);
    
    // 设置最大池化操作符的内核类型为最大池化和格式为无符号8位整数
    max_pooling->ukernel_type = pytorch_qnnp_ukernel_type_max_pooling;
    max_pooling->format = pytorch_qnnp_format_quint8;
    
    // 将指向最大池化操作符的指针存储到输出参数max_pooling_out中
    *max_pooling_out = max_pooling;
    // 返回成功状态
    return pytorch_qnnp_status_success;
error:
  // 释放先前分配的最大池化运算符资源
  pytorch_qnnp_delete_operator(max_pooling);
  // 返回当前函数的状态
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
    // 定义最大池化运算符及其参数
    pytorch_qnnp_operator_t max_pooling,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_pixel_stride,
    uint8_t* output,
    size_t output_pixel_stride,
    pthreadpool_t threadpool) {
  // 检查 QNNPACK 是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    // 若未初始化，则记录错误信息并返回未初始化状态
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_max_pooling2d_nhwc_u8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  // 检查批处理大小是否为零
  if (batch_size == 0) {
    // 若为零，设置最大池化运算符的批处理大小为零，并返回成功状态
    max_pooling->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  // 检查输入的宽度或高度是否为零
  if (input_width == 0 || input_height == 0) {
    // 若为零，记录错误信息并返回无效参数状态
    pytorch_qnnp_log_error(
        "failed to setup max pooling with %zux%zu input: input dimensions must be non-zero",
        input_width,
        input_height);
    return pytorch_qnnp_status_invalid_parameter;
  }

  // 设置最大池化运算符的参数
  max_pooling->batch_size = batch_size;
  max_pooling->input_height = input_height;
  max_pooling->input_width = input_width;
  max_pooling->input = input;
  max_pooling->input_pixel_stride = input_pixel_stride;

  // 计算输出的高度和宽度
  max_pooling->output_height = compute_output_dimension(
      input_height + max_pooling->input_padding_height * 2,
      max_pooling->kernel_height,
      max_pooling->dilation_height,
      max_pooling->stride_height);
  max_pooling->output_width = compute_output_dimension(
      input_width + max_pooling->input_padding_width * 2,
      max_pooling->kernel_width,
      max_pooling->dilation_width,
      max_pooling->stride_width);
  max_pooling->output = output;
  max_pooling->output_pixel_stride = output_pixel_stride;

  // 检查是否可以重用之前的输入数据
  size_t valid_batch_size = 0;
  if (input == max_pooling->last_input &&
      input_height == max_pooling->last_input_height &&
      input_width == max_pooling->last_input_width) {
    valid_batch_size = max_pooling->valid_batch_size;
    if (batch_size <= valid_batch_size) {
      return pytorch_qnnp_status_success;
    }
  }

  /* Micro-kernel may read up to (mr - 1) elements after the end of indirection
   * buffer */
  // 定义 micro-kernel 可能读取的缓冲区边界
  const uint32_t mr = pytorch_qnnp_params.u8maxpool.mr;

  // 设置步幅维度的间接缓冲区
  pytorch_qnnp_indirection_set_step_dimensions(max_pooling);
  // 计算所需的间接缓冲区大小
  const size_t indirection_buffer_size = sizeof(void*) *
      ((mr - 1) +
       batch_size * max_pooling->output_height * max_pooling->step_height);

  // 分配间接缓冲区内存
  const void** indirection_buffer = (const void**)realloc(
      max_pooling->indirection_buffer, indirection_buffer_size);
  if (indirection_buffer == NULL) {
    // 分配失败时记录错误信息
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for indirection buffer",
        indirection_buffer_size);
  // 返回内存不足的错误代码
  return pytorch_qnnp_status_out_of_memory;
}

// 将间接访问缓冲区指针设置为指向给定的间接访问缓冲区
max_pooling->indirection_buffer = indirection_buffer;

// 初始化最大池化层的间接访问结构，以支持给定的有效批次大小
pytorch_qnnp_indirection_init_maxpool2d(max_pooling, valid_batch_size);

// 设置最大池化层的最后一次输入、输入高度、输入宽度以及有效批次大小
max_pooling->last_input = input;
max_pooling->last_input_height = input_height;
max_pooling->last_input_width = input_width;
max_pooling->valid_batch_size = max(valid_batch_size, batch_size);

// 返回成功状态代码
return pytorch_qnnp_status_success;
}


注释：


# 这行代码关闭了一个代码块，它与一个开放的 `{` 配对，用于定义一个代码段或控制结构。
```