# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\add.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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

// 定义一个用于创建 QNNPACK 量化加法操作符的函数
enum pytorch_qnnp_status pytorch_qnnp_create_add_nc_q8(
    size_t channels,                  // 输入通道数
    uint8_t a_zero_point,             // 输入张量 A 的零点
    float a_scale,                    // 输入张量 A 的缩放因子
    uint8_t b_zero_point,             // 输入张量 B 的零点
    float b_scale,                    // 输入张量 B 的缩放因子
    uint8_t sum_zero_point,           // 输出张量的零点
    float sum_scale,                  // 输出张量的缩放因子
    uint8_t sum_min,                  // 输出张量的最小值
    uint8_t sum_max,                  // 输出张量的最大值
    uint32_t flags,                   // 操作标志位
    pytorch_qnnp_operator_t* add_out) // 输出的加法操作符指针
{
  pytorch_qnnp_operator_t add_op = NULL; // 初始化 QNNPACK 加法操作符指针
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized; // 初始化操作状态

  // 检查 QNNPACK 是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_add_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  // 检查输入通道数是否为正
  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create add operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  // 检查输入张量 A 的缩放因子是否为正常数值
  if (a_scale <= 0.0f || !isnormal(a_scale)) {
    pytorch_qnnp_log_error(
        "failed to create add operator with %.7g A scale: scale must be finite and positive",
        a_scale);
    goto error;
  }

  // 检查输入张量 B 的缩放因子是否为正常数值
  if (b_scale <= 0.0f || !isnormal(b_scale)) {
    pytorch_qnnp_log_error(
        "failed to create add operator with %.7g B scale: scale must be finite and positive",
        b_scale);
    goto error;
  }

  // 检查输出张量的缩放因子是否为正常数值
  if (sum_scale <= 0.0f || !isnormal(sum_scale)) {
    pytorch_qnnp_log_error(
        "failed to create add operator with %.7g output scale: scale must be finite and positive",
        sum_scale);
    goto error;
  }

  // 检查输出张量的范围是否合法
  if (sum_min >= sum_max) {
    pytorch_qnnp_log_error(
        "failed to create add operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        sum_min,
        sum_max);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  // 计算输入张量 A 到输出张量的缩放比例，并检查其合法性
  const float a_output_scale = a_scale / sum_scale;
  if (a_output_scale < 0x1.0p-14f || a_output_scale >= 0x1.0p+8f) {
    pytorch_qnnp_log_error(
        "failed to create add operator with %.7g A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range",
        a_output_scale);
    goto error;
  }

  // 计算输入张量 B 到输出张量的缩放比例，并检查其合法性
  const float b_output_scale = b_scale / sum_scale;
  if (b_output_scale < 0x1.0p-14f || b_output_scale >= 0x1.0p+8f) {
    pytorch_qnnp_log_error(
        "failed to create add operator with %.7g A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range",
        b_output_scale);


This completes the annotation of the provided C code snippet following the specified guidelines.
  // 跳转到错误处理标签，如果分配 add_op 失败，则执行错误处理
  goto error;
}

// 设置内存耗尽的状态
status = pytorch_qnnp_status_out_of_memory;

// 分配一个 pytorch_qnnp_operator 结构体的内存空间，并检查分配是否成功
add_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
if (add_op == NULL) {
  // 如果分配失败，记录错误信息并跳转到错误处理
  pytorch_qnnp_log_error(
      "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
      sizeof(struct pytorch_qnnp_operator));
  goto error;
}

// 设置 add_op 结构体的 channels 属性
add_op->channels = channels;

// 计算并设置 add_op 结构体的加法量化参数
add_op->add_quantization_params =
    pytorch_qnnp_compute_add_quantization_params(
        a_zero_point,
        b_zero_point,
        sum_zero_point,
        a_scale / sum_scale,
        b_scale / sum_scale,
        sum_min,
        sum_max);

// 设置 add_op 结构体的 ukernel_type 属性为加法类型
add_op->ukernel_type = pytorch_qnnp_ukernel_type_add;

// 设置 add_op 结构体的 format 属性为 8 位无符号整数（quint8）
add_op->format = pytorch_qnnp_format_quint8;

// 将指向 add_op 结构体的指针赋给 add_out 指针所指向的位置
*add_out = add_op;

// 返回成功状态
return pytorch_qnnp_status_success;
// 删除 pytorch_qnnp_delete_operator 函数调用的 add_op 操作符
pytorch_qnnp_delete_operator(add_op);
// 返回当前函数的状态
return status;
}

// 设置一个用于执行 uint8_t 类型输入的加法操作的 QNNPACK 操作符
enum pytorch_qnnp_status pytorch_qnnp_setup_add_nc_q8(
    // 添加操作符指针
    pytorch_qnnp_operator_t add_op,
    // 批处理大小
    size_t batch_size,
    // 第一个输入数据指针
    const uint8_t* a,
    // 第一个输入数据的跨度
    size_t a_stride,
    // 第二个输入数据指针
    const uint8_t* b,
    // 第二个输入数据的跨度
    size_t b_stride,
    // 存放加法结果的数据指针
    uint8_t* sum,
    // 存放加法结果的数据跨度
    size_t sum_stride) {
  
  // 检查 QNNPACK 是否已经初始化
  if (!pytorch_qnnp_params.initialized) {
    // 若未初始化，则记录错误信息
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_add_nc_q8 failed because QNNPACK is not properly initialized");
    // 返回未初始化状态
    return pytorch_qnnp_status_uninitialized;
  }

  // 检查批处理大小是否为 0
  if (batch_size == 0) {
    // 若批处理大小为 0，则设置操作符的批处理大小为 0，并返回成功状态
    add_op->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  // 设置操作符的批处理大小为传入的批处理大小
  add_op->batch_size = batch_size;
  // 设置操作符的第一个输入数据指针
  add_op->input = a;
  // 设置操作符的第一个输入数据的跨度
  add_op->input_pixel_stride = a_stride;
  // 设置操作符的第二个输入数据指针
  add_op->input2 = b;
  // 设置操作符的第二个输入数据的跨度
  add_op->input2_pixel_stride = b_stride;
  // 设置操作符的输出数据指针
  add_op->output = sum;
  // 设置操作符的输出数据的跨度
  add_op->output_pixel_stride = sum_stride;

  // 返回成功状态
  return pytorch_qnnp_status_success;
}
```