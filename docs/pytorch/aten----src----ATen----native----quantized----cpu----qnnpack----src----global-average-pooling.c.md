# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\global-average-pooling.c`

```
/*
 * 版权所有（C）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录的LICENSE文件中以BSD风格许可证授权。
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

// 创建全局平均池化操作符（对NWC格式的量化8位整数张量）
enum pytorch_qnnp_status pytorch_qnnp_create_global_average_pooling_nwc_q8(
    size_t channels,                 // 输入张量的通道数
    uint8_t input_zero_point,        // 输入张量的零点
    float input_scale,               // 输入张量的缩放因子
    uint8_t output_zero_point,       // 输出张量的零点
    float output_scale,              // 输出张量的缩放因子
    uint8_t output_min,              // 输出张量的最小值
    uint8_t output_max,              // 输出张量的最大值
    uint32_t flags,                  // 操作标志
    pytorch_qnnp_operator_t* global_average_pooling_out) {  // 输出的全局平均池化操作符指针
  pytorch_qnnp_operator_t global_average_pooling_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  // 检查QNNPACK是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_global_average_pooling_nwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  // 检查通道数是否为正
  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create global average pooling operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  // 检查输入缩放因子是否为正常值
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create global average pooling operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  // 检查输出缩放因子是否为正常值
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create global average pooling operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  // 计算输入输出缩放因子比例，并检查其范围
  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    pytorch_qnnp_log_error(
        "failed to create global average pooling operator with %.7g input-to-output scale ratio: "
        "scale ratio must be in [2**-8, 2**8) range",
        input_output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  // 分配并初始化全局平均池化操作符结构体
  global_average_pooling_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (global_average_pooling_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  // 分配用于填充零值的缓冲区
  void* zero_buffer = calloc(channels, sizeof(uint8_t));
  if (zero_buffer == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for zero padding",
        channels * sizeof(uint8_t));
    goto error;
  }
    // 跳转到错误处理标签，用于处理错误情况
    goto error;
  }
  // 将 zero_buffer 分配给 global_average_pooling_op 的 zero_buffer 属性
  global_average_pooling_op->zero_buffer = zero_buffer;
  // 将 zero_buffer 分配给 global_average_pooling_op 的 zero_pointer 属性
  global_average_pooling_op->zero_pointer = zero_buffer;

  // 设置 global_average_pooling_op 的 channels 属性为给定的通道数
  global_average_pooling_op->channels = channels;
  // 设置 global_average_pooling_op 的 input_zero_point 属性为输入的零点
  global_average_pooling_op->input_zero_point = input_zero_point;
  // 设置 global_average_pooling_op 的 output_zero_point 属性为输出的零点
  global_average_pooling_op->output_zero_point = output_zero_point;
  // 设置 global_average_pooling_op 的 input_scale 属性为输入的缩放因子
  global_average_pooling_op->input_scale = input_scale;
  // 设置 global_average_pooling_op 的 output_scale 属性为输出的缩放因子
  global_average_pooling_op->output_scale = output_scale;
  // 设置 global_average_pooling_op 的 output_min 属性为输出的最小值
  global_average_pooling_op->output_min = output_min;
  // 设置 global_average_pooling_op 的 output_max 属性为输出的最大值
  global_average_pooling_op->output_max = output_max;

  // 设置 global_average_pooling_op 的 ukernel_type 属性为全局平均池化的内核类型
  global_average_pooling_op->ukernel_type =
      pytorch_qnnp_ukernel_type_global_average_pooling;
  // 设置 global_average_pooling_op 的 format 属性为 quint8 格式
  global_average_pooling_op->format = pytorch_qnnp_format_quint8;

  // 将 global_average_pooling_op 的地址赋值给 global_average_pooling_out 指向的指针
  *global_average_pooling_out = global_average_pooling_op;
  // 返回成功状态
  return pytorch_qnnp_status_success;
# 删除全局平均池操作符，释放相关资源
error:
  pytorch_qnnp_delete_operator(global_average_pooling_op);
  返回操作状态
  return status;
}

# 设置带有非交叉通道布局的全局平均池化操作符
enum pytorch_qnnp_status pytorch_qnnp_setup_global_average_pooling_nwc_q8(
    # 全局平均池化操作符
    pytorch_qnnp_operator_t global_average_pooling_op,
    # 批处理大小
    size_t batch_size,
    # 输入宽度
    size_t width,
    # 输入数据指针
    const uint8_t* input,
    # 输入步幅
    size_t input_stride,
    # 输出数据指针
    uint8_t* output,
    # 输出步幅
    size_t output_stride) {
  # 若 QNNPACK 未初始化，则记录错误信息并返回未初始化状态
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_global_average_pooling_nwc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  # 若批处理大小为零，则设置操作符的批处理大小为零，并返回成功状态
  if (batch_size == 0) {
    global_average_pooling_op->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  # 若宽度为零，则记录错误信息并返回无效参数状态
  if (width == 0) {
    pytorch_qnnp_log_error(
        "failed to setup global average pooling operator with width %zu: width must be non-zero",
        width);
    return pytorch_qnnp_status_invalid_parameter;
  }

  # 设置操作符的批处理大小和输入宽度
  global_average_pooling_op->batch_size = batch_size;
  global_average_pooling_op->input_width = width;
  global_average_pooling_op->input = input;
  global_average_pooling_op->input_pixel_stride = input_stride;
  global_average_pooling_op->output = output;
  global_average_pooling_op->output_pixel_stride = output_stride;

  # 计算全局平均池化的量化参数
  global_average_pooling_op->avgpool_quantization_params =
      pytorch_qnnp_compute_avgpool_quantization_params(
          -(int32_t)width *
              (int32_t)(uint32_t)global_average_pooling_op->input_zero_point,
          global_average_pooling_op->input_scale /
              (global_average_pooling_op->output_scale * (float)width),
          global_average_pooling_op->output_zero_point,
          global_average_pooling_op->output_min,
          global_average_pooling_op->output_max);

  # 返回成功状态
  return pytorch_qnnp_status_success;
}
```