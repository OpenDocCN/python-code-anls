# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\deconvolution.c`

```
/*
 * 版权声明：Facebook公司及其关联公司版权所有。
 * 保留所有权利。
 *
 * 此源代码根据源代码根目录中的LICENSE文件中的BSD风格许可证进行许可。
 */

// 包含标准头文件
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// 包含QNNPACK库头文件
#include <pytorch_qnnpack.h>
#include <qnnpack/indirection.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

// 定义一个静态内联函数，用于计算输出维度
static inline size_t compute_output_dimension(
    size_t input_dimension,
    size_t input_padding_dimension,
    size_t adjustment_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t stride_dimension) {
  // 计算有效的卷积核维度
  const size_t effective_kernel_dimension =
      (kernel_dimension - 1) * dilation_dimension + 1;
  // 根据输入维度、填充维度、调整维度、卷积核维度、扩展维度、步幅维度计算输出维度
  return stride_dimension * (input_dimension - 1) + adjustment_dimension +
      effective_kernel_dimension - input_padding_dimension;
}

// 定义创建二维反卷积（转置卷积）操作符的函数，输入输出数据格式为NHWC，量化为uint8类型
enum pytorch_qnnp_status pytorch_qnnp_create_deconvolution2d_nhwc_q8(
    uint32_t input_padding_height,          // 输入填充高度
    uint32_t input_padding_width,           // 输入填充宽度
    uint32_t adjustment_height,             // 调整高度
    uint32_t adjustment_width,              // 调整宽度
    uint32_t kernel_height,                 // 卷积核高度
    uint32_t kernel_width,                  // 卷积核宽度
    uint32_t stride_height,                 // 步幅高度
    uint32_t stride_width,                  // 步幅宽度
    uint32_t dilation_height,               // 扩展高度
    uint32_t dilation_width,                // 扩展宽度
    uint32_t groups,                        // 组数
    size_t group_input_channels,            // 单组输入通道数
    size_t group_output_channels,           // 单组输出通道数
    uint8_t input_zero_point,               // 输入零点
    const uint8_t* kernel_zero_points,      // 卷积核零点
    const uint8_t* kernel,                  // 卷积核数据
    const int32_t* bias,                    // 偏置数据
    uint8_t output_zero_point,              // 输出零点
    uint8_t output_min,                     // 输出最小值
    uint8_t output_max,                     // 输出最大值
    uint32_t flags,                         // 标志位
    const float* requantization_scales,     // 重量化比例
    pytorch_qnnp_operator_t* deconvolution_out // 输出的反卷积操作符
) {
  pytorch_qnnp_operator_t deconvolution = NULL; // 反卷积操作符初始化为NULL
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized; // 操作符状态未初始化

  // 检查QNNPACK是否已初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_deconvolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter; // 参数无效状态

  // 检查卷积核尺寸是否有效
  if (kernel_width == 0 || kernel_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %" PRIu32 "x%" PRIu32
        " kernel: kernel dimensions must be non-zero",
        kernel_width,
        kernel_height);
    goto error;
  }

  // 检查步幅尺寸是否有效
  if (stride_width == 0 || stride_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %" PRIu32 "x%" PRIu32
        " stride: "
        "stride dimensions must be non-zero",
        stride_width,
        stride_height);
    goto error;
  }

  // 检查扩展尺寸是否有效
  if (dilation_width == 0 || dilation_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %" PRIu32 "x%" PRIu32
        " dilation: "
        "dilation dimensions must be non-zero",
        dilation_width,
        dilation_height);
  // 跳转到错误处理标签，用于处理发生错误的情况
  goto error;
}

// 设置为不支持的参数状态，用于后续检查
status = pytorch_qnnp_status_unsupported_parameter;

// 遍历所有的输出通道组，检查重新量化比例是否有效
for (int i = 0; i < groups * group_output_channels; i++) {
  // 如果重新量化比例小于等于0或者不是正常数（非正常数包括NaN和无穷大），则记录错误并跳转到错误处理
  if (requantization_scales[i] <= 0.0f ||
      !isnormal(requantization_scales[i])) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution operator with %.7g requantization scale for "
        "channel %d scale must be finite and positive",
        requantization_scales[i], i);
    goto error;
  }
}

// 设置为内存不足的状态，用于后续检查
status = pytorch_qnnp_status_out_of_memory;

// 分配内存以存储卷积操作符的结构体
deconvolution = calloc(1, sizeof(struct pytorch_qnnp_operator));
if (deconvolution == NULL) {
  // 如果内存分配失败，则记录错误并跳转到错误处理
  pytorch_qnnp_log_error(
      "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
      sizeof(struct pytorch_qnnp_operator));
  goto error;
}

// 从全局参数中获取卷积的一些参数值
const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
const uint32_t kr = pytorch_qnnp_params.q8conv.kr;

// 计算分组卷积的步长，确保内存对齐
const uint32_t n_stride = (group_output_channels + (nr - 1)) & -nr;
const uint32_t k_stride = (group_input_channels + (kr - 1)) & -kr;
const uint32_t kernel_size = kernel_height * kernel_width;
const size_t packed_group_weights_size =
    (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) * n_stride;

// 分配内存以存储打包的卷积权重数据
deconvolution->packed_weights = malloc(packed_group_weights_size * groups);
if (deconvolution->packed_weights == NULL) {
  // 如果内存分配失败，则记录错误并跳转到错误处理
  pytorch_qnnp_log_error(
      "failed to allocate %zu bytes for packed weights",
      packed_group_weights_size * groups);
  goto error;
}

// 初始化打包的卷积权重数据为指定的零点值
memset(
    deconvolution->packed_weights,
    kernel_zero_points[0],
    packed_group_weights_size * groups);

// 遍历所有的组，为每个组的卷积权重进行打包
for (uint32_t group = 0; group < groups; group++) {
  pytorch_pack_q8deconv_w(
      group_output_channels,
      kernel_size,
      group_input_channels,
      nr,
      kr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        input_zero_point,
        kernel_zero_points[0],
#endif
        kernel +
            group * group_output_channels * kernel_size * group_input_channels,
        bias + group * group_output_channels,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        kernel_zero_points + group * group_output_channels,
#endif
        (void*)((uintptr_t)deconvolution->packed_weights + group * packed_group_weights_size));


#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        // 如果不是运行时量化，则使用预定义的输入和卷积核零点
        input_zero_point,
        kernel_zero_points[0],
#endif
        // 计算卷积核和偏置的指针偏移量，考虑分组卷积的情况
        kernel +
            group * group_output_channels * kernel_size * group_input_channels,
        bias + group * group_output_channels,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        // 如果是运行时量化，则使用动态计算的卷积核零点数组
        kernel_zero_points + group * group_output_channels,
#endif
        // 将卷积层的预打包权重的指针偏移量保存到 deconvolution 结构中
        (void*)((uintptr_t)deconvolution->packed_weights + group * packed_group_weights_size));



  }

  // 计算用于填充输入的零值缓冲区大小和偏移量
  size_t zero_size = sizeof(uint8_t) * k_stride;
  size_t zero_offset = 0;
  if (group_input_channels < 8) {
    // 如果分组输入通道数小于8，则需要额外的零填充
    zero_size += 8;
    zero_offset = 8;
  }

  // 分配零填充缓冲区内存
  void* zero_buffer = malloc(zero_size);
  if (zero_buffer == NULL) {
    // 分配内存失败时输出错误信息并跳转到 error 处理
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for zero padding", zero_size);
    goto error;
  }
  // 将零填充缓冲区初始化为输入零点值
  memset(zero_buffer, input_zero_point, zero_size);
  // 将零填充缓冲区和偏移后的指针保存到 deconvolution 结构中
  deconvolution->zero_buffer = zero_buffer;
  deconvolution->zero_pointer = (void*)((uintptr_t)zero_buffer + zero_offset);

  // 设置 deconvolution 结构的各种属性
  deconvolution->input_padding_height = input_padding_height;
  deconvolution->input_padding_width = input_padding_width;
  deconvolution->adjustment_height = adjustment_height;
  deconvolution->adjustment_width = adjustment_width;

  deconvolution->kernel_height = kernel_height;
  deconvolution->kernel_width = kernel_width;
  deconvolution->stride_height = stride_height;
  deconvolution->stride_width = stride_width;
  deconvolution->dilation_height = dilation_height;
  deconvolution->dilation_width = dilation_width;
  deconvolution->groups = groups;
  deconvolution->group_input_channels = group_input_channels;
  deconvolution->group_output_channels = group_output_channels;

  // 设置卷积核零点值
  deconvolution->kernel_zero_point = kernel_zero_points[0];

  // 计算卷积量化参数
  deconvolution->conv_quantization_params =
      pytorch_qnnp_compute_conv_quantization_params(
          input_zero_point,
          kernel_zero_points,
          requantization_scales,
          output_zero_point,
          output_min,
          output_max);

  // 设置 deconvolution 结构的类型和格式
  deconvolution->ukernel_type = pytorch_qnnp_ukernel_type_conv;
  deconvolution->format = pytorch_qnnp_format_quint8;
  deconvolution->transpose = true;

  // 将创建好的 deconvolution 结构指针保存到输出参数中
  *deconvolution_out = deconvolution;
  // 返回成功状态
  return pytorch_qnnp_status_success;

error:
  // 在出错时释放 deconvolution 操作符的内存，并返回错误状态
  pytorch_qnnp_delete_operator(deconvolution);
  return status;
}


enum pytorch_qnnp_status pytorch_qnnp_setup_deconvolution2d_nhwc_q8(
    pytorch_qnnp_operator_t deconvolution,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_pixel_stride,
    uint8_t* output,
    size_t output_pixel_stride,
    pthreadpool_t threadpool) {
  if (!pytorch_qnnp_params.initialized) {
    // 若 QNNPACK 未正确初始化，则输出错误信息并返回未初始化状态
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_deconvolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    // 如果批处理大小为零，则设置 deconvolution 结构的批处理大小为零并返回成功状态
    deconvolution->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  if (input_width == 0 || input_height == 0) {
    pytorch_qnnp_log_error(
        "failed to setup deconvolution with %zux%zu input: input dimensions must be non-zero",
        input_width,
        input_height);

错误日志记录函数调用，用于记录无法设置反卷积的错误，显示输入维度必须是非零的。


    return pytorch_qnnp_status_invalid_parameter;
  }

如果输入维度无效，直接返回参数无效的状态。


  deconvolution->batch_size = batch_size;
  deconvolution->input_height = input_height;
  deconvolution->input_width = input_width;
  deconvolution->input = input;
  deconvolution->input_pixel_stride = input_pixel_stride;
  deconvolution->output = output;
  deconvolution->output_pixel_stride = output_pixel_stride;

设置反卷积结构体的基本属性，包括批大小、输入高度、宽度、输入数据指针、输入像素步长、输出数据指针、输出像素步长。


  const size_t kernel_height = deconvolution->kernel_height;
  const size_t kernel_width = deconvolution->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t stride_height = deconvolution->stride_height;
  const size_t stride_width = deconvolution->stride_width;

从反卷积结构体中获取卷积核的高度、宽度、步长等参数。


  const size_t output_height = deconvolution->output_height =
      compute_output_dimension(
          input_height,
          deconvolution->input_padding_height * 2,
          deconvolution->adjustment_height,
          kernel_height,
          deconvolution->dilation_height,
          stride_height);

计算输出的高度，并将结果赋给反卷积结构体的输出高度属性。


  const size_t output_width = deconvolution->output_width =
      compute_output_dimension(
          input_width,
          deconvolution->input_padding_width * 2,
          deconvolution->adjustment_width,
          kernel_width,
          deconvolution->dilation_width,
          stride_width);

计算输出的宽度，并将结果赋给反卷积结构体的输出宽度属性。


  const size_t groups = deconvolution->groups;
  const size_t output_size = output_height * output_width;
  const size_t output_tile_size = pytorch_qnnp_params.q8conv.mr;
  const size_t tiled_output_size = round_up(output_size, output_tile_size);

获取反卷积结构体中的组数和输出大小，并计算瓦片化输出大小。


  const size_t indirection_buffer_size =
      sizeof(void*) * batch_size * groups * tiled_output_size * kernel_size;

计算间接缓冲区的大小，以便存储指向数据的指针。


  const void** indirection_buffer = (const void**)realloc(
      deconvolution->indirection_buffer, indirection_buffer_size);

重新分配反卷积结构体中的间接缓冲区，以适应新的大小。


  if (indirection_buffer == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for indirection buffer",
        indirection_buffer_size);
    return pytorch_qnnp_status_out_of_memory;
  }

如果重新分配失败，则记录错误并返回内存不足的状态。


  deconvolution->indirection_buffer = indirection_buffer;

  pytorch_qnnp_indirection_init_deconv2d(
      deconvolution, output_tile_size, tiled_output_size);

将重新分配的间接缓冲区存储到反卷积结构体中，并初始化反卷积的二维间接索引。


  return pytorch_qnnp_status_success;

成功设置反卷积并返回成功的状态。
}



# 这行代码表示一个单独的右花括号 '}'，用于结束一个代码块或语句的闭合。在程序中，花括号通常用于定义函数、循环或条件语句的范围。
```