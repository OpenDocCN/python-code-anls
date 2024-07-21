# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\clamp.c`

```
/*
 * 以下是 C 语言的 QNNPACK 库函数，用于创建和设置 Clamp 操作符，操作的数据类型为 uint8_t。
 * 这些函数用于限制输入张量的数值范围在指定的最小值和最大值之间。
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
 * 创建一个 Clamp 操作符实例，限制通道内的 uint8_t 输入张量的数值范围。
 *
 * 参数:
 *   channels: 输入张量的通道数
 *   output_min: 输出张量的最小值
 *   output_max: 输出张量的最大值
 *   flags: 额外标志位
 *   clamp_out: 输出，指向创建的 Clamp 操作符的指针
 *
 * 返回:
 *   状态码，表示操作是否成功
 */
enum pytorch_qnnp_status pytorch_qnnp_create_clamp_nc_u8(
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* clamp_out) {
  pytorch_qnnp_operator_t clamp_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_clamp_nc_u8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create Clamp operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (output_min > output_max) {
    pytorch_qnnp_log_error(
        "failed to create Clamp operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  clamp_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (clamp_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  clamp_op->channels = channels;
  clamp_op->u8_clamping_params =
      pytorch_qnnp_compute_u8_clamping_params(output_min, output_max);

  clamp_op->ukernel_type = pytorch_qnnp_ukernel_type_clamp;
  clamp_op->format = pytorch_qnnp_format_quint8;

  *clamp_out = clamp_op;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(clamp_op);
  return status;
}

/*
 * 设置 Clamp 操作符的输入和输出张量以及相关的参数。
 *
 * 参数:
 *   clamp: 要设置的 Clamp 操作符实例
 *   batch_size: 批处理大小
 *   input: 输入张量的指针
 *   input_stride: 输入张量的像素跨度
 *   output: 输出张量的指针
 *   output_stride: 输出张量的像素跨度
 *
 * 返回:
 *   状态码，表示操作是否成功
 */
enum pytorch_qnnp_status pytorch_qnnp_setup_clamp_nc_u8(
    pytorch_qnnp_operator_t clamp,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_clamp_nc_u8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    clamp->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  clamp->batch_size = batch_size;
  clamp->input = input;
  clamp->input_pixel_stride = input_stride;
  clamp->output = output;
  clamp->output_pixel_stride = output_stride;

  return pytorch_qnnp_status_success;
}
```