# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\hardswish.c`

```
/*
 * 版权声明：
 * Facebook, Inc.及其关联公司保留所有权利。
 *
 * 此源代码在根目录下的LICENSE文件中以BSD风格许可证许可
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>

/*
 * 创建一个以量化参数为基础的 Hardswish 运算符
 *
 * 参数：
 * - channels: 通道数
 * - input_zero_point: 输入零点
 * - input_scale: 输入缩放因子
 * - output_zero_point: 输出零点
 * - output_scale: 输出缩放因子
 * - output_min: 输出的最小值
 * - output_max: 输出的最大值
 * - flags: 标志位
 * - hardswish_out: 输出的 Hardswish 运算符指针
 *
 * 返回值：
 * - 返回一个枚举类型的状态码，表示函数执行的结果状态
 */
enum pytorch_qnnp_status pytorch_qnnp_create_hardswish_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* hardswish_out) {
  pytorch_qnnp_operator_t hardswish_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_hardswish_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create Hardswish operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Hardswish operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Hardswish operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    pytorch_qnnp_log_error(
        "failed to create Hardswish operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  hardswish_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (hardswish_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  hardswish_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (hardswish_op->lookup_table == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate 256 bytes for Hardswish lookup table");
    goto error;
  }

  uint8_t* lookup_table = hardswish_op->lookup_table;
  const float scaled_min = (float)(int32_t)output_min;
  const float scaled_max = (float)(int32_t)output_max;
  const float inv_output_scale = 1.0f / output_scale;
  for (int32_t i = 0; i < 256; i++) {
    float x =
        input_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    // 执行 Hardswish 函数，C语言中没有内置的min/max函数
    float x2 = x + 3.0f;
    x2 = x2 > 0.0f ? x2 : 0.0f;
    x2 = x2 < 6.0f ? x2 : 6.0f;
    x2 = x * x2 / 6.0f;
    # 计算 x 的平方乘以 x2 除以 6，结果赋给 x2
    float scaled_hardswish_x = inv_output_scale * x2 + output_zero_point;
    # 计算经过硬切线激活函数缩放后的值，乘以逆输出比例，加上输出零点偏移
    if (scaled_hardswish_x < scaled_min) {
      # 如果 scaled_hardswish_x 小于预定的最小值 scaled_min，则设为 scaled_min
      scaled_hardswish_x = scaled_min;
    }
    if (scaled_hardswish_x > scaled_max) {
      # 如果 scaled_hardswish_x 大于预定的最大值 scaled_max，则设为 scaled_max
      scaled_hardswish_x = scaled_max;
    }
    lookup_table[(uint32_t)i] = (uint8_t)lrintf(scaled_hardswish_x);
    # 将经过取整后的 scaled_hardswish_x 转换为 uint8 类型，并存入查找表中的索引 i 处
  }

  hardswish_op->channels = channels;
  # 设置硬切线操作的通道数为 channels

  hardswish_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
  # 设置硬切线操作的内核类型为查找表内核类型

  hardswish_op->format = pytorch_qnnp_format_quint8;
  # 设置硬切线操作的数据格式为无符号八位整数类型

  *hardswish_out = hardswish_op;
  # 将硬切线操作指针赋给硬切线输出指针

  return pytorch_qnnp_status_success;
  # 返回操作成功的状态码
// 释放 hardswish 操作符占用的内存
pytorch_qnnp_delete_operator(hardswish_op);
// 返回操作的状态，表示设置 hardswish 操作成功或失败
return status;
}

// 设置处理 hardswish 操作的 QNNPACK 算子的参数
enum pytorch_qnnp_status pytorch_qnnp_setup_hardswish_nc_q8(
    // hardswish 操作符对象
    pytorch_qnnp_operator_t hardswish,
    // 批处理大小
    size_t batch_size,
    // 输入数据的指针，以 uint8_t 类型表示
    const uint8_t* input,
    // 输入数据中相邻像素之间的跨度
    size_t input_stride,
    // 输出数据的指针，以 uint8_t 类型表示
    uint8_t* output,
    // 输出数据中相邻像素之间的跨度
    size_t output_stride) {
  // 检查 QNNPACK 是否已经初始化
  if (!pytorch_qnnp_params.initialized) {
    // 若未初始化，则记录错误信息并返回未初始化状态
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_hardswish_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  // 如果批处理大小为 0，则设置操作对象的批处理大小为 0，并返回成功状态
  if (batch_size == 0) {
    hardswish->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  // 设置操作对象的批处理大小、输入数据、输入像素跨度、输出数据和输出像素跨度
  hardswish->batch_size = batch_size;
  hardswish->input = input;
  hardswish->input_pixel_stride = input_stride;
  hardswish->output = output;
  hardswish->output_pixel_stride = output_stride;

  // 返回设置成功状态
  return pytorch_qnnp_status_success;
}
```