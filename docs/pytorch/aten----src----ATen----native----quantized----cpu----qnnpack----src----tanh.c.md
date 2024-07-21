# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\tanh.c`

```
/*
 * 版权声明：Facebook, Inc. 及其关联公司保留所有权利。
 *
 * 此源代码在根目录下的 LICENSE 文件中以 BSD 风格许可证授权。
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>

// 定义创建 TanH 操作符的函数，处理 QNNPACK 中 8 位量化的特定需求
enum pytorch_qnnp_status pytorch_qnnp_create_tanh_nc_q8(
    size_t channels,                     // 输入通道数
    uint8_t input_zero_point,            // 输入零点
    float input_scale,                   // 输入缩放因子
    uint8_t output_zero_point,           // 输出零点
    float output_scale,                  // 输出缩放因子
    uint8_t output_min,                  // 输出最小值
    uint8_t output_max,                  // 输出最大值
    uint32_t flags,                      // 标志
    pytorch_qnnp_operator_t* tanh_out) { // 输出 TanH 操作符

  // 初始化局部变量
  pytorch_qnnp_operator_t tanh_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  // 检查 QNNPACK 是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_tanh_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  // 检查通道数是否为正数
  status = pytorch_qnnp_status_invalid_parameter;
  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create TanH operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  // 检查输入缩放因子是否有效
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create TanH operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  // 检查输出缩放因子是否有效
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create TanH operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  // 检查输出范围是否有效
  if (output_min >= output_max) {
    pytorch_qnnp_log_error(
        "failed to create TanH operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  // 检查输出缩放因子是否为 [-1, 1] 范围内的 8 位量化
  status = pytorch_qnnp_status_unsupported_parameter;
  if (output_scale != 0x2.0p-8f) {  // [-1, 1] 范围在 8 位中的表示，等于 2.0 / 256
    pytorch_qnnp_log_error(
        "failed to create TanH operator with %.7g output scale: only output scale of 2/256 is supported",
        output_scale);
    goto error;
  }

  // 检查输出零点是否为 128
  if (output_zero_point != 128) {
    pytorch_qnnp_log_error(
        "failed to create TanH operator with %" PRIu8
        " output zero point: only output zero point of 128 is supported",
        output_zero_point);
    goto error;
  }

  // 分配内存给 TanH 操作符结构体
  status = pytorch_qnnp_status_out_of_memory;
  tanh_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (tanh_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  // 分配内存给 TanH 的查找表
  tanh_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (tanh_op->lookup_table == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate 256 bytes for TanH lookup table");
    // 跳转到错误处理标签，用于在发生错误时执行清理操作并返回错误状态
    goto error;
  }

  // 获取 tanh_op 结构体中的查找表指针
  uint8_t* lookup_table = tanh_op->lookup_table;
  // 将输出的最小值和最大值按照整型方式转换为浮点数，并存储在 scaled_min 和 scaled_max 中
  const float scaled_min = (float)(int32_t)output_min;
  const float scaled_max = (float)(int32_t)output_max;
  // 遍历 0 到 255 的整数
  for (int32_t i = 0; i < 256; i++) {
    // 根据输入的比例因子和零点偏移计算当前值 x
    const float x =
        input_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    /* 缩放 tanh(x) 乘以 1 / 输出比例 = 128.0
       同时，从缩放值中减去零点，因为我们假设是 UINT8 类型
    */
    float scaled_tanh_x = 128.0f * tanhf(x) + 128.0f;
    // 如果缩放后的值小于 scaled_min，则将其设为 scaled_min
    if (scaled_tanh_x < scaled_min) {
      scaled_tanh_x = scaled_min;
    }
    // 如果缩放后的值大于 scaled_max，则将其设为 scaled_max
    if (scaled_tanh_x > scaled_max) {
      scaled_tanh_x = scaled_max;
    }
    // 将缩放后的值四舍五入并转换为 uint8 存储在查找表中
    lookup_table[(uint32_t)i] = (uint8_t)lrintf(scaled_tanh_x);
  }

  // 将通道数存储到 tanh_op 结构体中
  tanh_op->channels = channels;

  // 设置 ukernel_type 为 pytorch_qnnp_ukernel_type_lut
  tanh_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
  // 设置 format 为 pytorch_qnnp_format_quint8
  tanh_op->format = pytorch_qnnp_format_quint8;

  // 将 tanh_op 结构体指针存储到 tanh_out 指向的位置
  *tanh_out = tanh_op;
  // 返回成功状态
  return pytorch_qnnp_status_success;
// 删除 tanh_op 操作符对象
pytorch_qnnp_delete_operator(tanh_op);
// 返回当前函数的状态
return status;
}

// 设置一个处理批量输入的 tanh 运算符，输入和输出是 uint8 类型
enum pytorch_qnnp_status pytorch_qnnp_setup_tanh_nc_q8(
    // tanh 运算符对象指针
    pytorch_qnnp_operator_t tanh,
    // 批量大小
    size_t batch_size,
    // 输入数据的指针
    const uint8_t* input,
    // 输入数据的跨度（每个数据项的大小）
    size_t input_stride,
    // 输出数据的指针
    uint8_t* output,
    // 输出数据的跨度（每个数据项的大小）
    size_t output_stride) {
  // 检查 QNNPACK 是否已经初始化
  if (!pytorch_qnnp_params.initialized) {
    // 如果未初始化，则记录错误信息并返回未初始化状态
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_tanh_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  // 如果批量大小为 0，则设置运算符的批量大小为 0，并返回成功状态
  if (batch_size == 0) {
    tanh->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  // 设置运算符的批量大小、输入指针及其跨度、输出指针及其跨度
  tanh->batch_size = batch_size;
  tanh->input = input;
  tanh->input_pixel_stride = input_stride;
  tanh->output = output;
  tanh->output_pixel_stride = output_stride;

  // 返回成功状态
  return pytorch_qnnp_status_success;
}
```