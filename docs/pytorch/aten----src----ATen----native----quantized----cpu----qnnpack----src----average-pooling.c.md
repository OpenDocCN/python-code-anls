# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\average-pooling.c`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 本源代码受BSD风格许可证保护，详见
 * 源代码根目录下的LICENSE文件。
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

// 计算输出维度的辅助函数
static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t pooling_dimension,
    size_t stride_dimension) {
  return (padded_input_dimension - pooling_dimension) / stride_dimension + 1;
}

// 创建二维NHWC布局量化8位平均池化操作符
enum pytorch_qnnp_status pytorch_qnnp_create_average_pooling2d_nhwc_q8(
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* average_pooling_out) {
  pytorch_qnnp_operator_t average_pooling = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  // 检查QNNPACK是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_average_pooling2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  // 检查参数是否有效
  status = pytorch_qnnp_status_invalid_parameter;

  // 计算池化大小并检查是否为零
  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %" PRIu32 "x%" PRIu32
        " pooling size: "
        "pooling size dimensions must be non-zero",
        pooling_width,
        pooling_height);
    goto error;
  }

  // 检查池化尺寸是否为1，这在平均池化中是无意义的
  if (pooling_size == 1) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with 1 pooling element: "
        "1x1 pooling is meaningless");
    goto error;
  }

  // 检查步长是否为零
  if (stride_height == 0 || stride_width == 0) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %" PRIu32 "x%" PRIu32
        " stride: "
        "stride dimensions must be non-zero",
        stride_width,
        stride_height);
    goto error;
  }

  // 检查通道数是否为零
  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %zu channels: "
        "number of channels must be non-zero",
        channels);
    goto error;
  }

  // 检查输入尺度是否为正常数值
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %.7g input scale: "
        "scale must be finite and positive",
        input_scale);
    goto error;
  }

  // 检查输出尺度是否为正常数值
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
  // 记录错误信息至日志，指出无法创建具有指定输出比例的平均池化操作
  pytorch_qnnp_log_error(
      "failed to create average pooling with %.7g output scale: "
      "scale must be finite and positive",
      output_scale);
  // 跳转到错误处理标签
  goto error;
}

// 设置为不支持的参数状态
status = pytorch_qnnp_status_unsupported_parameter;

// 计算输入输出比例
const float input_output_scale = input_scale / output_scale;
// 检查输入输出比例是否在有效范围内
if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
  // 记录错误信息至日志，指出无法创建具有指定输入输出比例的平均池化操作
  pytorch_qnnp_log_error(
      "failed to create average pooling with %.7g input scale and %.7g output scale: "
      "input-to-output scale ratio (%.7f) must be in [2**-8, 2**8) range",
      input_scale,
      output_scale,
      input_output_scale);
  // 跳转到错误处理标签
  goto error;
}

// 检查池化大小是否超过上限
if (pooling_size >= 16777216) {
  // 记录错误信息至日志，指出无法创建具有指定池化大小的平均池化操作
  pytorch_qnnp_log_error(
      "failed to create average pooling with %" PRIu32 " (%" PRIu32
      "x%" PRIu32
      ") pooling elements: "
      "the number of elements in the pooling area must be below 2**24",
      pooling_size,
      pooling_width,
      pooling_height);
  // 跳转到错误处理标签
  goto error;
}

// 设置为内存不足状态
status = pytorch_qnnp_status_out_of_memory;

// 分配平均池化操作结构的内存空间
average_pooling = calloc(1, sizeof(struct pytorch_qnnp_operator));
if (average_pooling == NULL) {
  // 记录错误信息至日志，指出无法分配平均池化操作结构的内存空间
  pytorch_qnnp_log_error(
      "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
      sizeof(struct pytorch_qnnp_operator));
  // 跳转到错误处理标签
  goto error;
}

// 检查是否需要填充
const bool any_padding = (input_padding_width | input_padding_height) != 0;
const uint32_t kr = pytorch_qnnp_params.q8avgpool.kr;
const uint32_t mr = pytorch_qnnp_params.q8avgpool.mr;
const uint32_t qr = pytorch_qnnp_params.q8avgpool.qr;
if (any_padding || (channels >= kr || (pooling_size - mr) % qr != 0)) {
  // 分配用于零填充的缓冲区内存空间
  void* zero_buffer = malloc(channels);
  if (zero_buffer == NULL) {
    // 记录错误信息至日志，指出无法分配零填充缓冲区的内存空间
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for zero padding", channels);
    // 跳转到错误处理标签
    goto error;
  }
  // 将零填充缓冲区初始化为输入的零点值
  memset(zero_buffer, input_zero_point, channels);
  // 将零填充缓冲区设置为平均池化操作结构的零缓冲区
  average_pooling->zero_buffer = zero_buffer;
  average_pooling->zero_pointer = zero_buffer;
}

// 设置平均池化操作结构的各个属性
average_pooling->input_padding_depth = 0;
average_pooling->input_padding_height = input_padding_height;
average_pooling->input_padding_width = input_padding_width;
average_pooling->kernel_depth = 1;
average_pooling->kernel_height = pooling_height;
average_pooling->kernel_width = pooling_width;
average_pooling->stride_depth = 1;
average_pooling->stride_height = stride_height;
average_pooling->stride_width = stride_width;
average_pooling->dilation_depth = 1;
average_pooling->dilation_height = 1;
average_pooling->dilation_width = 1;
average_pooling->channels = channels;

// 计算池化区域的行数
size_t nrows = pooling_height * pooling_width;
if (channels >= pytorch_qnnp_params.q8avgpool.kr) {
  if (nrows <= mr) {
    nrows = mr;
  } else {
    nrows = round_up(nrows - mr, qr) + mr;
  }


# 结构体或函数的结尾，可能是代码块的结束

  }



  }


# 结构体或函数的结尾，用于结束代码块

  }
  // 释放平均池化操作符占用的内存
  pytorch_qnnp_delete_operator(average_pooling);
  // 返回操作执行的状态
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
    // 平均池化操作符对象
    pytorch_qnnp_operator_t average_pooling,
    // 批处理大小
    size_t batch_size,
    // 输入图像的高度和宽度
    size_t input_height,
    size_t input_width,
    // 输入数据指针
    const uint8_t* input,
    // 输入像素步长
    size_t input_pixel_stride,
    // 输出数据指针
    uint8_t* output,
    // 输出像素步长
    size_t output_pixel_stride,
    // 线程池对象
    pthreadpool_t threadpool) {
  // 如果 QNNPACK 尚未初始化，则报错并返回未初始化状态
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_average_pooling2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  // 如果批处理大小为零，则设置平均池化操作符的批处理大小为零，并返回成功状态
  if (batch_size == 0) {
    average_pooling->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  // 如果输入图像的宽度或高度为零，则报错并返回无效参数状态
  if (input_width == 0 || input_height == 0) {
    pytorch_qnnp_log_error(
        "failed to setup average pooling with %zux%zu input: input dimensions must be non-zero",
        input_width,
        input_height);
    return pytorch_qnnp_status_invalid_parameter;
  }

  // 设置平均池化操作符的属性
  average_pooling->batch_size = batch_size;
  average_pooling->input_depth = 1;
  average_pooling->input_height = input_height;
  average_pooling->input_width = input_width;
  average_pooling->input = input;
  average_pooling->input_pixel_stride = input_pixel_stride;

  // 计算输出图像的高度和宽度，并设置平均池化操作符的输出属性
  average_pooling->output_height = compute_output_dimension(
      input_height + average_pooling->input_padding_height * 2,
      average_pooling->kernel_height,
      average_pooling->stride_height);
  average_pooling->output_width = compute_output_dimension(
      input_width + average_pooling->input_padding_width * 2,
      average_pooling->kernel_width,
      average_pooling->stride_width);
  average_pooling->output_depth = 1;
  average_pooling->output = output;
  average_pooling->output_pixel_stride = output_pixel_stride;

  // 如果输入与上一次设置的输入相同，则使用上次验证的批处理大小
  size_t valid_batch_size = 0;
  if (input == average_pooling->last_input &&
      input_height == average_pooling->last_input_height &&
      input_width == average_pooling->last_input_width) {
    valid_batch_size = average_pooling->valid_batch_size;
    // 如果当前批处理大小小于或等于验证的批处理大小，则直接返回成功状态
    if (batch_size <= valid_batch_size) {
      return pytorch_qnnp_status_success;
    }
  }

  /* 微内核可能会在间接缓冲区结束后读取最多 (mr - 1) 个元素 */
  const uint32_t mr = pytorch_qnnp_params.q8avgpool.mr;

  // 设置步长相关的间接缓冲区尺寸
  pytorch_qnnp_indirection_set_step_dimensions(average_pooling);
  const size_t indirection_buffer_size = sizeof(void*) *
      ((mr - 1) +
       batch_size * average_pooling->output_height *
           average_pooling->step_height);

  // 重新分配间接缓冲区内存
  const void** indirection_buffer = (const void**)realloc(
      average_pooling->indirection_buffer, indirection_buffer_size);
  if (indirection_buffer == NULL) {
    // 如果分配失败，则报错并返回失败状态
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for indirection buffer",
        indirection_buffer_size);
    return pytorch_qnnp_status_out_of_memory;


// 返回内存不足的错误状态码
return pytorch_qnnp_status_out_of_memory;



  }
  average_pooling->indirection_buffer = indirection_buffer;


// 将平均池化层的间接缓冲区设置为给定的间接缓冲区
average_pooling->indirection_buffer = indirection_buffer;



  pytorch_qnnp_indirection_init_dwconv(average_pooling, valid_batch_size);


// 使用给定的有效批次大小初始化深度卷积的间接访问
pytorch_qnnp_indirection_init_dwconv(average_pooling, valid_batch_size);



  average_pooling->last_input = input;
  average_pooling->last_input_height = input_height;
  average_pooling->last_input_width = input_width;
  average_pooling->valid_batch_size = max(valid_batch_size, batch_size);


// 设置平均池化层的最后输入、输入高度、输入宽度和有效批次大小
average_pooling->last_input = input;
average_pooling->last_input_height = input_height;
average_pooling->last_input_width = input_width;
average_pooling->valid_batch_size = max(valid_batch_size, batch_size);



  return pytorch_qnnp_status_success;


// 返回成功状态码
return pytorch_qnnp_status_success;
}



# 结束一个代码块，这里是一个函数或类的结束符号
```