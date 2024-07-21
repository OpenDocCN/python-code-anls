# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\channel-shuffle.c`

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
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>
#include <qnnpack/params.h>

// 定义 QNNPACK 中的通道重排操作的创建函数，支持 x8 数据格式
enum pytorch_qnnp_status pytorch_qnnp_create_channel_shuffle_nc_x8(
    size_t groups,                   // 分组数
    size_t group_channels,           // 每个分组的通道数
    uint32_t flags,                  // 标志位
    pytorch_qnnp_operator_t* channel_shuffle_out) {  // 输出的通道重排操作符指针
  pytorch_qnnp_operator_t channel_shuffle_op = NULL;  // 初始化通道重排操作符
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;  // 初始化状态

  // 检查 QNNPACK 是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_channel_shuffle_nc_x8 failed because QNNPACK is not properly initialized");
    goto error;  // 若未初始化，记录错误并跳转到错误处理标签
  }

  status = pytorch_qnnp_status_invalid_parameter;  // 设置状态为参数无效

  // 检查分组数是否小于等于 1，要求至少两个分组
  if (groups <= 1) {
    pytorch_qnnp_log_error(
        "failed to create channel shuffle operator with %zu groups: "
        "at least two groups required",
        groups);
    goto error;  // 记录错误并跳转到错误处理标签
  }

  // 检查每个分组的通道数是否为 0，要求非零值
  if (group_channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create channel shuffle operator with %zu group channels: "
        "number of group channels must be non-zero",
        group_channels);
    goto error;  // 记录错误并跳转到错误处理标签
  }

  status = pytorch_qnnp_status_out_of_memory;  // 设置状态为内存不足

  // 分配内存并初始化通道重排操作符
  channel_shuffle_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (channel_shuffle_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;  // 记录错误并跳转到错误处理标签
  }

  // 设置通道重排操作符的属性
  channel_shuffle_op->groups = groups;  // 设置分组数
  channel_shuffle_op->group_channels = group_channels;  // 设置每个分组的通道数

  channel_shuffle_op->ukernel_type = pytorch_qnnp_ukernel_type_channel_shuffle;  // 设置核函数类型为通道重排
  channel_shuffle_op->format = pytorch_qnnp_format_quint8;  // 设置数据格式为 8 位整数型

  *channel_shuffle_out = channel_shuffle_op;  // 返回创建的通道重排操作符
  return pytorch_qnnp_status_success;  // 返回成功状态

error:
  pytorch_qnnp_delete_operator(channel_shuffle_op);  // 删除创建的通道重排操作符
  return status;  // 返回错误状态
}

// 定义 QNNPACK 中的通道重排操作的设置函数，支持 x8 数据格式
enum pytorch_qnnp_status pytorch_qnnp_setup_channel_shuffle_nc_x8(
    pytorch_qnnp_operator_t channel_shuffle_op,  // 通道重排操作符
    size_t batch_size,                          // 批处理大小
    const uint8_t* input,                       // 输入数据指针
    size_t input_stride,                        // 输入数据的步幅
    uint8_t* output,                            // 输出数据指针
    size_t output_stride) {                     // 输出数据的步幅
  // 检查 QNNPACK 是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_channel_shuffle_nc_x8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;  // 若未初始化，返回未初始化状态
  }

  // 若批处理大小为 0，则设置通道重排操作符的批处理大小为 0，并返回成功状态
  if (batch_size == 0) {
    channel_shuffle_op->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  // 设置通道重排操作符的属性
  channel_shuffle_op->batch_size = batch_size;  // 设置批处理大小
  channel_shuffle_op->input = input;  // 设置输入数据指针
  channel_shuffle_op->input_pixel_stride = input_stride;  // 设置输入数据的步幅
  channel_shuffle_op->output = output;  // 设置输出数据指针
  channel_shuffle_op->output_pixel_stride = output_stride;  // 设置输出数据的步幅

  return pytorch_qnnp_status_success;  // 返回成功状态
}
```