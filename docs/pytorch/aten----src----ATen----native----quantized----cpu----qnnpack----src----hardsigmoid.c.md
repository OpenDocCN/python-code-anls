# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\hardsigmoid.c`

```py
/*
 * 版权所有（c）Facebook公司及其附属公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的LICENSE文件中找到的BSD样式许可证进行许可。
 */

#include <assert.h>                 // 引入断言库，用于运行时检查
#include <math.h>                   // 引入数学函数库，如数学运算和函数
#include <stddef.h>                 // 引入stddef.h头文件，定义了一些常用的数据类型和宏
#include <stdint.h>                 // 引入stdint.h头文件，定义了一些标准整数类型
#include <stdlib.h>                 // 引入标准库函数，如内存分配等

#include <pytorch_qnnpack.h>        // 引入PyTorch QNNPACK头文件
#include <qnnpack/log.h>            // 引入QNNPACK日志功能头文件
#include <qnnpack/operator.h>       // 引入QNNPACK运算符头文件

enum pytorch_qnnp_status pytorch_qnnp_create_hardsigmoid_nc_q8(
    size_t channels,                // Hardsigmoid操作器的通道数
    uint8_t input_zero_point,       // 输入的零点
    float input_scale,              // 输入的比例
    uint8_t output_zero_point,      // 输出的零点
    float output_scale,             // 输出的比例
    uint8_t output_min,             // 输出的最小值
    uint8_t output_max,             // 输出的最大值
    uint32_t flags,                 // 标志位
    pytorch_qnnp_operator_t* hardsigmoid_out) {  // 输出的Hardsigmoid操作器指针
  pytorch_qnnp_operator_t hardsigmoid_op = NULL;  // 初始化Hardsigmoid操作器为NULL
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;  // 初始化状态为未初始化

  if (!pytorch_qnnp_params.initialized) {  // 如果QNNPACK未初始化
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_hardsigmoid_nc_q8 failed because QNNPACK is not properly initialized");  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  status = pytorch_qnnp_status_invalid_parameter;  // 设置状态为无效参数

  if (channels == 0) {  // 如果通道数为0
    pytorch_qnnp_log_error(
        "failed to create Hardsigmoid operator with %zu channels: number of channels must be non-zero",
        channels);  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {  // 如果输入比例小于等于0或者不是正常数
    pytorch_qnnp_log_error(
        "failed to create Hardsigmoid operator with %.7g input scale: scale must be finite and positive",
        input_scale);  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {  // 如果输出比例小于等于0或者不是正常数
    pytorch_qnnp_log_error(
        "failed to create Hardsigmoid operator with %.7g output scale: scale must be finite and positive",
        output_scale);  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  if (output_min >= output_max) {  // 如果输出的最小值大于等于最大值
    pytorch_qnnp_log_error(
        "failed to create Hardsigmoid operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  status = pytorch_qnnp_status_unsupported_parameter;  // 设置状态为不支持的参数

  if (output_scale != 0x1.0p-8f) {  // 如果输出比例不等于1/256
    pytorch_qnnp_log_error(
        "failed to create Hardsigmoid operator with %.7g output scale: only output scale of 1/256 is supported",
        output_scale);  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  if (output_zero_point != 0) {  // 如果输出的零点不等于0
    pytorch_qnnp_log_error(
        "failed to create Hardsigmoid operator with %" PRIu8
        " output zero point: only output zero point of 0 is supported",
        output_zero_point);  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  status = pytorch_qnnp_status_out_of_memory;  // 设置状态为内存不足

  hardsigmoid_op = calloc(1, sizeof(struct pytorch_qnnp_operator));  // 分配Hardsigmoid操作器结构的内存空间
  if (hardsigmoid_op == NULL) {  // 如果分配失败
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  hardsigmoid_op->lookup_table = malloc(256 * sizeof(uint8_t));  // 分配查找表的内存空间
  if (hardsigmoid_op->lookup_table == NULL) {  // 如果分配失败

    pytorch_qnnp_log_error(
        "failed to allocate lookup table for Hardsigmoid operator");  // 记录错误日志
    goto error;  // 跳转到错误处理
  }

  // 成功创建Hardsigmoid操作器，设置输出指针
  *hardsigmoid_out = hardsigmoid_op;
  return pytorch_qnnp_status_success;

error:
  // 错误处理，释放已分配的资源
  if (hardsigmoid_op != NULL) {
    if (hardsigmoid_op->lookup_table != NULL) {
      free(hardsigmoid_op->lookup_table);
    }
    free(hardsigmoid_op);
  }
  return status;
}
    // 记录错误信息到日志，指出无法为Hardsigmoid查找表分配256字节的内存空间
    pytorch_qnnp_log_error(
        "failed to allocate 256 bytes for Hardsigmoid lookup table");
    // 转至错误处理部分
    goto error;
  }

  // 获取指向Hardsigmoid操作的查找表的指针
  uint8_t* lookup_table = hardsigmoid_op->lookup_table;
  // 将输出的最小和最大值转换为浮点数并缩放
  const float scaled_min = (float)(int32_t)output_min;
  const float scaled_max = (float)(int32_t)output_max;
  // 输出缩放的倒数
  const float inv_output_scale = 1.0f / output_scale;
  // 遍历256个索引
  for (int32_t i = 0; i < 256; i++) {
    // 计算输入的缩放
    float x =
        input_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    // 应用Hardsigmoid函数，因为C语言没有min/max函数
    float x2 = x + 3.0f;
    x2 = x2 > 0.0f ? x2 : 0.0f;
    x2 = x2 < 6.0f ? x2 : 6.0f;
    x2 = x2 / 6.0f;
    // 缩放后的Hardsigmoid值，并加上输出零点偏移
    float scaled_hardsigmoid_x = inv_output_scale * x2 + output_zero_point;
    // 如果缩放后的值小于输出的最小值，则使用最小值
    if (scaled_hardsigmoid_x < scaled_min) {
      scaled_hardsigmoid_x = scaled_min;
    }
    // 如果缩放后的值大于输出的最大值，则使用最大值
    if (scaled_hardsigmoid_x > scaled_max) {
      scaled_hardsigmoid_x = scaled_max;
    }
    // 将缩放后的Hardsigmoid值四舍五入并转换为uint8存入查找表中
    lookup_table[(uint32_t)i] = (uint8_t)lrintf(scaled_hardsigmoid_x);
  }

  // 设置Hardsigmoid操作的通道数
  hardsigmoid_op->channels = channels;

  // 设置Hardsigmoid操作的ukernel类型为查找表
  hardsigmoid_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
  // 设置Hardsigmoid操作的数据格式为quint8
  hardsigmoid_op->format = pytorch_qnnp_format_quint8;

  // 将Hardsigmoid操作结果指针赋值给输出参数
  *hardsigmoid_out = hardsigmoid_op;
  // 返回成功状态
  return pytorch_qnnp_status_success;
# 删除指定的硬sigmoid运算符
error:
  pytorch_qnnp_delete_operator(hardsigmoid_op);
  # 返回当前状态
  return status;



# 设置硬sigmoid操作符的参数和输入输出数据信息
enum pytorch_qnnp_status pytorch_qnnp_setup_hardsigmoid_nc_q8(
    pytorch_qnnp_operator_t hardsigmoid,  # 硬sigmoid操作符对象
    size_t batch_size,  # 批处理大小
    const uint8_t* input,  # 输入数据的指针
    size_t input_stride,  # 输入数据的跨度
    uint8_t* output,  # 输出数据的指针
    size_t output_stride) {  # 输出数据的跨度
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_hardsigmoid_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;  # 返回未初始化状态
  }

  if (batch_size == 0) {
    hardsigmoid->batch_size = 0;  # 设置批处理大小为0
    return pytorch_qnnp_status_success;  # 返回成功状态
  }

  hardsigmoid->batch_size = batch_size;  # 设置操作符的批处理大小
  hardsigmoid->input = input;  # 设置输入数据的指针
  hardsigmoid->input_pixel_stride = input_stride;  # 设置输入数据的跨度
  hardsigmoid->output = output;  # 设置输出数据的指针
  hardsigmoid->output_pixel_stride = output_stride;  # 设置输出数据的跨度

  return pytorch_qnnp_status_success;  # 返回成功状态
}
```