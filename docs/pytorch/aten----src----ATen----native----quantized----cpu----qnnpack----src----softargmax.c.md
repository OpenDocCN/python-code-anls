# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\softargmax.c`

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

// 定义 SoftArgMax 操作的 QNNPACK 接口函数
enum pytorch_qnnp_status pytorch_qnnp_create_softargmax_nc_q8(
    size_t channels,                          // 输入特征图的通道数
    float input_scale,                        // 输入量化的缩放因子
    uint8_t output_zero_point,                // 输出量化的零点
    float output_scale,                       // 输出量化的缩放因子
    uint32_t flags,                           // 操作标志位
    pytorch_qnnp_operator_t* softargmax_out)  // 返回的 SoftArgMax 操作符指针
{
  pytorch_qnnp_operator_t softargmax_op = NULL; // 初始化 SoftArgMax 操作符指针
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized; // 初始化操作状态

  // 检查 QNNPACK 是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_softargmax_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter; // 设置参数无效的状态

  // 检查通道数是否为正数
  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  // 检查输入量化缩放因子是否有效
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  // 检查输出量化缩放因子是否有效
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter; // 设置不支持的参数状态

  // 检查输出量化缩放因子是否为 1/256
  if (output_scale != 0x1.0p-8f) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g output scale: only output scale of 1/256 is supported",
        output_scale);
    goto error;
  }

  // 检查输出量化零点是否为 0
  if (output_zero_point != 0) {
    pytorch_qnnp_log_error(
        "failed to create Soft ArgMax operator with %" PRIu8
        " output zero point: only output zero point of 0 is supported",
        output_zero_point);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory; // 设置内存不足的状态

  // 分配 SoftArgMax 操作符结构体内存
  softargmax_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (softargmax_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  // 分配 SoftArgMax 查找表内存
  softargmax_op->lookup_table = malloc(256 * sizeof(uint32_t));
  if (softargmax_op->lookup_table == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate 256 bytes for Soft ArgMax lookup table");
    goto error;
  }

  uint32_t* lookup_table = softargmax_op->lookup_table; // 获取查找表指针
  const double qscale =
      fmin(((double)UINT32_MAX) / (double)channels, 8388607.0); // 计算量化缩放因子
  // 填充 SoftArgMax 查找表
  for (int32_t i = 0; i < 256; i++) {
    const double scaled_exp_xi =
        qscale * exp((double)(i - 255) * (double)input_scale); // 计算缩放后的指数值
    // 将 uint32_t 类型的索引 i 映射到经过四舍五入的 scaled_exp_xi 的 uint32_t 类型值，并存入查找表中
    lookup_table[(uint32_t)i] = (uint32_t)lrint(scaled_exp_xi);
  }

  // 设置 softargmax_op 结构体中的 channels 字段为指定的通道数
  softargmax_op->channels = channels;

  // 将 softargmax_op 结构体中的 ukernel_type 字段设置为 pytorch_qnnp_ukernel_type_softargmax，表示使用 softargmax 操作
  softargmax_op->ukernel_type = pytorch_qnnp_ukernel_type_softargmax;

  // 将 softargmax_op 结构体中的 format 字段设置为 pytorch_qnnp_format_quint8，表示数据格式为 quint8（8位无符号整数）
  softargmax_op->format = pytorch_qnnp_format_quint8;

  // 将 softargmax_op 结构体的指针赋值给 softargmax_out 指向的位置，完成 softargmax_op 的输出
  *softargmax_out = softargmax_op;

  // 返回成功状态码，表示 softargmax 操作执行成功
  return pytorch_qnnp_status_success;
error:
  pytorch_qnnp_delete_operator(softargmax_op);
  return status;

# 调用函数 `pytorch_qnnp_delete_operator` 删除给定的软最大值操作符 `softargmax_op`，然后返回 `status` 变量的值。


enum pytorch_qnnp_status pytorch_qnnp_setup_softargmax_nc_q8(
    pytorch_qnnp_operator_t softargmax,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {

# 定义一个函数 `pytorch_qnnp_setup_softargmax_nc_q8`，该函数的返回类型是 `enum pytorch_qnnp_status`，接受以下参数：
# - `softargmax`：指向 `pytorch_qnnp_operator_t` 类型的指针，表示软最大值操作符
# - `batch_size`：表示批次大小，类型为 `size_t`
# - `input`：指向 `uint8_t` 类型的常量指针，表示输入数据的起始地址
# - `input_stride`：表示输入数据的步幅，类型为 `size_t`
# - `output`：指向 `uint8_t` 类型的指针，表示输出数据的起始地址
# - `output_stride`：表示输出数据的步幅，类型为 `size_t`


  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_softargmax_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

# 如果 `pytorch_qnnp_params.initialized` 为假，表示 QNNPACK 框架未正确初始化，记录错误日志并返回未初始化状态 `pytorch_qnnp_status_uninitialized`。


  if (batch_size == 0) {
    softargmax->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

# 如果 `batch_size` 等于 0，将 `softargmax` 的 `batch_size` 成员设置为 0，并返回操作成功状态 `pytorch_qnnp_status_success`。


  softargmax->batch_size = batch_size;
  softargmax->input = input;
  softargmax->input_pixel_stride = input_stride;
  softargmax->output = output;
  softargmax->output_pixel_stride = output_stride;

  return pytorch_qnnp_status_success;
}

# 设置 `softargmax` 结构体的成员变量：
# - `batch_size` 设置为 `batch_size`
# - `input` 设置为 `input` 指针
# - `input_pixel_stride` 设置为 `input_stride`
# - `output` 设置为 `output` 指针
# - `output_pixel_stride` 设置为 `output_stride`
# 最后返回操作成功状态 `pytorch_qnnp_status_success`。
```