# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\leaky-relu.c`

```
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 本源代码使用BSD风格许可证授权，许可证可以在源代码根目录下的LICENSE文件中找到。
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>

// 定义QNNPACK中Leaky ReLU算子的创建函数，处理8位量化数据的NC布局
enum pytorch_qnnp_status pytorch_qnnp_create_leaky_relu_nc_q8(
    size_t channels,                          // 输入通道数
    float negative_slope,                     // 负斜率
    uint8_t input_zero_point,                 // 输入零点
    float input_scale,                        // 输入缩放比例
    uint8_t output_zero_point,                // 输出零点
    float output_scale,                       // 输出缩放比例
    uint8_t output_min,                       // 输出最小值
    uint8_t output_max,                       // 输出最大值
    uint32_t flags,                           // 标志位
    pytorch_qnnp_operator_t* leaky_relu_out) // 输出Leaky ReLU算子
{
  pytorch_qnnp_operator_t leaky_relu_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  // 检查QNNPACK是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_leaky_relu_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  // 检查通道数是否为正数
  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  // 检查负斜率是否为有限正数
  if (negative_slope <= 0.0f || !isnormal(negative_slope)) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g negative slope: slope must be finite and positive",
        negative_slope);
    goto error;
  }

  // 负斜率不能大于1.0
  if (negative_slope > 1.0f) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g negative slope: slope must not exceed 1.0",
        negative_slope);
    goto error;
  }

  // 输入缩放比例必须为有限正数
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  // 输出缩放比例必须为有限正数
  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  // 输出范围的最小值必须小于最大值
  if (output_min >= output_max) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  // 计算输入输出的缩放比例
  const float input_output_scale = input_scale / output_scale;

  // 输入输出缩放比例必须在[2**-8, 2**8)范围内
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    pytorch_qnnp_log_error(
        "failed to create Leaky ReLU operator with %.7g input-to-output scale ratio: "
        "scale ratio must be in [2**-8, 2**8) range",
        input_output_scale);


这段代码定义了一个在QNNPACK中处理8位量化数据的Leaky ReLU算子的创建函数，确保参数合法性并进行相应的错误处理和日志记录。
  // 跳转到错误处理标签
  goto error;
}

// 设置内存不足的错误状态
status = pytorch_qnnp_status_out_of_memory;

// 分配内存以存储 Leaky ReLU 运算符的结构体
leaky_relu_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
if (leaky_relu_op == NULL) {
  // 打印内存分配失败的错误信息
  pytorch_qnnp_log_error(
      "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
      sizeof(struct pytorch_qnnp_operator));
  // 跳转到错误处理标签
  goto error;
}

// 分配内存以存储 Leaky ReLU 激活函数的查找表
leaky_relu_op->lookup_table = malloc(256 * sizeof(uint8_t));
if (leaky_relu_op->lookup_table == NULL) {
  // 打印内存分配失败的错误信息
  pytorch_qnnp_log_error(
      "failed to allocate 256 bytes for Leaky ReLU lookup table");
  // 跳转到错误处理标签
  goto error;
}

// 指向查找表的指针
uint8_t* lookup_table = leaky_relu_op->lookup_table;
// 计算缩放后的最小值和最大值与零点的差
const float scaled_min_less_zero_point =
    (float)((int32_t)output_min - (int32_t)output_zero_point);
const float scaled_max_less_zero_point =
    (float)((int32_t)output_max - (int32_t)output_zero_point);
// 填充 Leaky ReLU 查找表
for (int32_t i = 0; i < 256; i++) {
  // 计算输入值的缩放
  const float x =
      input_output_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
  // 应用 Leaky ReLU 函数
  float y = x < 0.0f ? x * negative_slope : x;
  // 约束输出值在指定范围内
  if (y < scaled_min_less_zero_point) {
    y = scaled_min_less_zero_point;
  }
  if (y > scaled_max_less_zero_point) {
    y = scaled_max_less_zero_point;
  }
  // 将计算得到的值保存为查找表中的整数形式
  lookup_table[(uint32_t)i] = (uint8_t)(lrintf(y) + (long)output_zero_point);
}

// 设置 Leaky ReLU 运算符的通道数
leaky_relu_op->channels = channels;

// 指定 Leaky ReLU 运算所使用的 UKernel 类型和数据格式
leaky_relu_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
leaky_relu_op->format = pytorch_qnnp_format_quint8;

// 将创建的 Leaky ReLU 运算符结构体传递给输出指针
*leaky_relu_out = leaky_relu_op;
// 返回成功状态
return pytorch_qnnp_status_success;
# 删除泄漏 ReLU 运算符
pytorch_qnnp_delete_operator(leaky_relu_op);
# 返回状态
return status;
}

# 设置基于非连续数据的泄漏 ReLU 算子
enum pytorch_qnnp_status pytorch_qnnp_setup_leaky_relu_nc_q8(
    # 泄漏 ReLU 算子
    pytorch_qnnp_operator_t leaky_relu,
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
  # 如果 QNNPACK 未初始化
  if (!pytorch_qnnp_params.initialized) {
    # 记录错误日志
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_leaky_relu_nc_q8 failed because QNNPACK is not properly initialized");
    # 返回未初始化状态
    return pytorch_qnnp_status_uninitialized;
  }

  # 如果批处理大小为 0
  if (batch_size == 0) {
    # 设置泄漏 ReLU 算子的批处理大小为 0
    leaky_relu->batch_size = 0;
    # 返回成功状态
    return pytorch_qnnp_status_success;
  }

  # 设置泄漏 ReLU 算子的批处理大小
  leaky_relu->batch_size = batch_size;
  # 设置泄漏 ReLU 算子的输入数据指针
  leaky_relu->input = input;
  # 设置泄漏 ReLU 算子的输入数据步幅
  leaky_relu->input_pixel_stride = input_stride;
  # 设置泄漏 ReLU 算子的输出数据指针
  leaky_relu->output = output;
  # 设置泄漏 ReLU 算子的输出数据步幅
  leaky_relu->output_pixel_stride = output_stride;

  # 返回成功状态
  return pytorch_qnnp_status_success;
}
```