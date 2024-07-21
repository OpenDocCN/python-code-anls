# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\sigmoid.c`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在源代码根目录中的LICENSE文件中找到的BSD风格许可下授权。
 */

#include <assert.h>  // 包含断言相关的头文件
#include <math.h>    // 包含数学函数相关的头文件
#include <stddef.h>  // 包含与大小相关的头文件
#include <stdint.h>  // 包含整数类型相关的头文件
#include <stdlib.h>  // 包含标准库函数相关的头文件

#include <pytorch_qnnpack.h>  // 包含PyTorch QNNPACK的头文件
#include <qnnpack/log.h>      // 包含QNNPACK日志相关的头文件
#include <qnnpack/operator.h> // 包含QNNPACK操作符相关的头文件

// 定义创建 QNNPACK sigmoid 操作符的函数，处理8位量化的NC格式
enum pytorch_qnnp_status pytorch_qnnp_create_sigmoid_nc_q8(
    size_t channels,               // 通道数
    uint8_t input_zero_point,      // 输入的零点
    float input_scale,             // 输入的缩放因子
    uint8_t output_zero_point,     // 输出的零点
    float output_scale,            // 输出的缩放因子
    uint8_t output_min,            // 输出的最小值
    uint8_t output_max,            // 输出的最大值
    uint32_t flags,                // 标志位
    pytorch_qnnp_operator_t* sigmoid_out) {  // 输出的sigmoid操作符指针
  pytorch_qnnp_operator_t sigmoid_op = NULL;  // 初始化sigmoid操作符为NULL
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;  // 初始化状态为未初始化

  // 检查QNNPACK是否已经初始化，如果没有则报错
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_sigmoid_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;  // 设置状态为无效参数

  // 检查通道数是否为零，如果是则报错
  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create Sigmoid operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  // 检查输入缩放因子是否为正常值，如果不是则报错
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Sigmoid operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  // 检查输出缩放因子是否为正常值，如果不是则报错
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Sigmoid operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  // 检查输出范围最小值是否小于最大值，如果不是则报错
  if (output_min >= output_max) {
    pytorch_qnnp_log_error(
        "failed to create Sigmoid operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;  // 设置状态为不支持的参数

  // 检查输出缩放因子是否为1/256，如果不是则报错
  if (output_scale != 0x1.0p-8f) {
    pytorch_qnnp_log_error(
        "failed to create Sigmoid operator with %.7g output scale: only output scale of 1/256 is supported",
        output_scale);
    goto error;
  }

  // 检查输出零点是否为0，如果不是则报错
  if (output_zero_point != 0) {
    pytorch_qnnp_log_error(
        "failed to create Sigmoid operator with %" PRIu8
        " output zero point: only output zero point of 0 is supported",
        output_zero_point);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;  // 设置状态为内存不足

  // 分配内存以创建sigmoid操作符结构
  sigmoid_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (sigmoid_op == NULL) {  // 检查内存分配是否成功，如果失败则报错
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  // 分配内存以创建256字节的sigmoid查找表
  sigmoid_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (sigmoid_op->lookup_table == NULL) {  // 检查内存分配是否成功，如果失败则报错
    pytorch_qnnp_log_error(
        "failed to allocate 256 bytes for Sigmoid lookup table");
    // 跳转到错误处理标签，如果发生错误
    goto error;
  }

  // 获取 sigmoid_op 结构体中的查找表指针
  uint8_t* lookup_table = sigmoid_op->lookup_table;
  // 计算输出的最小和最大值，并进行浮点数缩放
  const float scaled_min = (float)(int32_t)output_min;
  const float scaled_max = (float)(int32_t)output_max;
  // 遍历输入范围内的所有可能值（0-255）
  for (int32_t i = 0; i < 256; i++) {
    // 计算输入值经过量化后的浮点数值 x
    const float x =
        input_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    /* 将 sigmoid(x) 的结果按照 1 / output scale = 256.0 进行缩放 */
    float scaled_sigmoid_x = 256.0f / (1.0f + expf(-x));
    // 如果缩放后的值小于输出的最小值，则将其调整为最小值
    if (scaled_sigmoid_x < scaled_min) {
      scaled_sigmoid_x = scaled_min;
    }
    // 如果缩放后的值大于输出的最大值，则将其调整为最大值
    if (scaled_sigmoid_x > scaled_max) {
      scaled_sigmoid_x = scaled_max;
    }
    // 将缩放后的 sigmoid(x) 值四舍五入为整数，并存储到查找表中
    lookup_table[(uint32_t)i] = (uint8_t)lrintf(scaled_sigmoid_x);
  }

  // 将通道数设置到 sigmoid_op 结构体中
  sigmoid_op->channels = channels;

  // 将 ukernel 类型设置为 lut 类型
  sigmoid_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
  // 将格式设置为 quint8
  sigmoid_op->format = pytorch_qnnp_format_quint8;

  // 将 sigmoid_op 结构体指针保存到 sigmoid_out 指针指向的位置
  *sigmoid_out = sigmoid_op;
  // 返回成功状态
  return pytorch_qnnp_status_success;
# 调用 QNNPACK 函数删除 sigmoid 操作符
pytorch_qnnp_delete_operator(sigmoid_op);
# 返回函数调用的状态
return status;
}

# 设置 sigmoid 操作符的参数和输入输出数据布局
enum pytorch_qnnp_status pytorch_qnnp_setup_sigmoid_nc_q8(
    # sigmoid 操作符的指针
    pytorch_qnnp_operator_t sigmoid,
    # 批处理大小
    size_t batch_size,
    # 输入数据的指针
    const uint8_t* input,
    # 输入数据的步幅
    size_t input_stride,
    # 输出数据的指针
    uint8_t* output,
    # 输出数据的步幅
    size_t output_stride) {
  # 检查 QNNPACK 是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    # 记录错误日志并返回未初始化状态
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_sigmoid_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  # 如果批处理大小为零，直接设置 sigmoid 操作符的批处理大小为零并返回成功状态
  if (batch_size == 0) {
    sigmoid->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  # 设置 sigmoid 操作符的批处理大小
  sigmoid->batch_size = batch_size;
  # 设置 sigmoid 操作符的输入数据指针
  sigmoid->input = input;
  # 设置 sigmoid 操作符的输入数据步幅
  sigmoid->input_pixel_stride = input_stride;
  # 设置 sigmoid 操作符的输出数据指针
  sigmoid->output = output;
  # 设置 sigmoid 操作符的输出数据步幅
  sigmoid->output_pixel_stride = output_stride;

  # 返回成功状态
  return pytorch_qnnp_status_success;
}
```