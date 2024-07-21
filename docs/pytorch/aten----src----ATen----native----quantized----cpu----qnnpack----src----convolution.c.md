# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\convolution.c`

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
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fxdiv.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/common.h>
#include <qnnpack/indirection.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>

// 定义一个静态内联函数，计算输出维度
static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t subsampling_dimension) {
  // 计算有效的卷积核维度
  const size_t effective_kernel_dimension =
      (kernel_dimension - 1) * dilation_dimension + 1;
  // 返回输出维度
  return (padded_input_dimension - effective_kernel_dimension) /
      subsampling_dimension +
      1;
}

/**
 * Not exposed in header file
 */
// 定义一个静态函数，用于创建基于 QNNPACK 的 Q8 卷积操作符
static enum pytorch_qnnp_status pytorch_qnnp_create_convolution_ndhwc_q8(
    uint32_t input_padding_depth,
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t kernel_depth,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_depth,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_depth,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool per_channel,
    pytorch_qnnp_operator_t* convolution_out,
    bool is_2d /* true: 2d, false: 3d */) {
  
  pytorch_qnnp_operator_t convolution = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  // 检查 QNNPACK 是否已经初始化
  if (!pytorch_qnnp_params.initialized) {
    // 如果未初始化，记录错误信息并返回错误状态
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_convolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  // 检查参数是否有效
  status = pytorch_qnnp_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    // 如果卷积核宽度或高度为零，记录错误信息并返回错误状态
    pytorch_qnnp_log_error(
        "failed to create convolution with %" PRIu32 "x%" PRIu32
        " kernel: kernel dimensions must be non-zero",
        kernel_width,
        kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    // 如果子采样宽度或高度为零，记录错误信息并返回错误状态
    pytorch_qnnp_log_error(
        "failed to create convolution with %" PRIu32 "x%" PRIu32
        " subsampling: "
        "subsampling dimensions must be non-zero",
        subsampling_width,
        subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    // 如果膨胀宽度或高度为零，记录错误信息并返回错误状态
    pytorch_qnnp_log_error(
        "failed to create convolution with %" PRIu32 "x%" PRIu32
        " dilation: "
        "dilation dimensions must be non-zero",
        dilation_width,
        dilation_height);
    goto error;
  }



继续上述代码的注释部分。
  pytorch_qnnp_log_error(
      "failed to create convolution with %" PRIu32 "x%" PRIu32
      " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width,
      dilation_height);
  // 记录错误日志，指示无法创建卷积操作，因为膨胀（dilation）维度必须是非零的
  goto error;
}

status = pytorch_qnnp_status_unsupported_parameter;

if (subsampling_height > kernel_height) {
  pytorch_qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32
      " kernel and %" PRIu32 "x%" PRIu32
      " subsampling: "
      "height subsampling is greater than kernel height; subsampling should be performed before the convolution",
      kernel_width,
      kernel_height,
      subsampling_width,
      subsampling_height);
  // 记录信息日志，提示卷积操作的效率问题，说明高度下采样大于卷积核高度，建议在卷积之前进行下采样
}

if (subsampling_width > kernel_width) {
  pytorch_qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32
      " kernel and %" PRIu32 "x%" PRIu32
      " subsampling: "
      "width subsampling is greater than kernel width; subsampling should be performed before the convolution",
      kernel_width,
      kernel_height,
      subsampling_width,
      subsampling_height);
  // 记录信息日志，提示卷积操作的效率问题，说明宽度下采样大于卷积核宽度，建议在卷积之前进行下采样
}

if (input_padding_depth >= kernel_depth) {
  pytorch_qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 "x%" PRIu32
      " kernel and %" PRIu32 "+%" PRIu32
      " depth padding: "
      "input depth padding is greater or equal to kernel depth",
      kernel_depth,
      kernel_height,
      kernel_width,
      input_padding_depth,
      input_padding_depth);
  // 记录信息日志，提示卷积操作的效率问题，说明输入深度填充大于或等于卷积核深度
}

if (input_padding_height >= kernel_height) {
  pytorch_qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 "x%" PRIu32
      " kernel and %" PRIu32 "+%" PRIu32
      " height padding: "
      "input height padding is greater or equal to kernel height",
      kernel_depth,
      kernel_height,
      kernel_width,
      input_padding_height,
      input_padding_height);
  // 记录信息日志，提示卷积操作的效率问题，说明输入高度填充大于或等于卷积核高度
}

if (input_padding_width >= kernel_width) {
  pytorch_qnnp_log_info(
      "inefficiency in convolution with %" PRIu32 "x%" PRIu32 "x%" PRIu32
      " kernel and %" PRIu32 "+%" PRIu32
      " width padding: "
      "input width padding is greater or equal to kernel width",
      kernel_depth,
      kernel_height,
      kernel_width,
      input_padding_width,
      input_padding_width);
  // 记录信息日志，提示卷积操作的效率问题，说明输入宽度填充大于或等于卷积核宽度
}

for (int i = 0; i < groups * group_output_channels; ++i) {
  if (requantization_scales[i] <= 0.0f ||
      !isnormal(requantization_scales[i])) {
    pytorch_qnnp_log_error(
        "failed to create fully connected operator with %.7g requantization scale: scale must be finite and positive",
        requantization_scales[i]);
    // 记录错误日志，指示无法创建全连接操作，因为重新量化比例必须是有限且正数
    goto error;
  }
}

status = pytorch_qnnp_status_out_of_memory;

convolution = calloc(1, sizeof(struct pytorch_qnnp_operator));
if (convolution == NULL) {
    // 记录错误并跳转到错误处理标签，如果分配 pytorch_qnnp_operator 结构体失败
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  // 计算卷积核大小
  const size_t kernel_size = kernel_height * kernel_width * kernel_depth;

  // 判断是否存在填充
  enum pytorch_qnnp_ukernel_type ukernel_type = pytorch_qnnp_ukernel_type_none;
  const bool any_padding =
      (input_padding_depth | input_padding_height | input_padding_width) != 0;

  // 判断是否是深度可分离卷积
  const bool has_depthwise_dimensions =
      (is_2d &&
       ((kernel_height == 3 && kernel_width == 3) ||
        (kernel_height == 5 && kernel_width == 5))) ||
      (!is_2d && kernel_height == 3 && kernel_width == 3 && kernel_depth == 3);
  const bool has_depthwise_grouping =
      group_input_channels == 1 && group_output_channels == 1 && groups > 1;
  if (has_depthwise_dimensions && has_depthwise_grouping) {
    ukernel_type = pytorch_qnnp_ukernel_type_dwconv;  // 设置为深度可分离卷积类型
  } else if (
      kernel_size == 1 && subsampling_height == 1 && subsampling_width == 1 &&
      !any_padding) {
    // 根据条件设置卷积类型
    ukernel_type =
        group_input_channels >= pytorch_qnnp_params.q8conv_xzp.kthreshold
        ? pytorch_qnnp_ukernel_type_xzp_gemm
        : pytorch_qnnp_ukernel_type_gemm;
  } else {
    ukernel_type = pytorch_qnnp_ukernel_type_conv;  // 默认设为普通卷积类型
  }
  size_t zero_size = 0, zero_offset = 0;

  // 根据卷积类型进行分支处理
  switch (ukernel_type) {
    // 处理深度可分离卷积和 per_channel 类型
    case pytorch_qnnp_ukernel_type_dwconv: {
      const uint32_t cr = pytorch_qnnp_params.q8dw9.cr;
      const uint32_t c_stride = (groups + (cr - 1)) & -cr;
      convolution->group_stride = c_stride;  // 设置组间距
      // 计算打包权重后的大小
      const size_t packed_weights_size =
          (sizeof(uint8_t) * kernel_size + sizeof(int32_t)) * c_stride;
      convolution->packed_weights = malloc(packed_weights_size);  // 分配内存
      if (convolution->packed_weights == NULL) {
        // 分配失败，记录错误信息并跳转到错误处理标签
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_weights_size);
        goto error;
      }

      // 根据卷积核大小选择打包函数
      switch (kernel_size) {
        case 9:
          pytorch_pack_q8dw_w(
              kernel_height,
              kernel_width,
              groups,
              cr,
      // 如果不是使用 PYTORCH_QNNPACK_RUNTIME_QUANTIZATION，设置输入和内核的零点
      input_zero_point,
      kernel_zero_points[0],
    // 对于卷积类型为 pytorch_qnnp_ukernel_type_conv 的情况
    case pytorch_qnnp_ukernel_type_conv: {
      // 提取参数中的输出通道数和输入通道数
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      // 计算输出通道数的对齐步长和输入通道数的对齐步长
      const uint32_t n_stride = (group_output_channels + (nr - 1)) & -nr;
      const uint32_t k_stride = (group_input_channels + (kr - 1)) & -kr;

      // 计算每个组的压缩权重大小
      const size_t packed_group_weights_size =
          (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) *
          n_stride;
      // 分配存储压缩权重的内存空间
      convolution->packed_weights = malloc(packed_group_weights_size * groups);
      if (convolution->packed_weights == NULL) {
        // 内存分配失败，记录错误并跳转到错误处理代码
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_group_weights_size * groups);
        goto error;
      }
      // 使用第一个卷积核零点值初始化分配的权重内存空间
      memset(
          convolution->packed_weights,
          kernel_zero_points[0],
          packed_group_weights_size * groups);

      // 根据不同的 ukernel 类型执行不同的操作
      switch (ukernel_type) {
        // 对于 ukernel 类型为 pytorch_qnnp_ukernel_type_gemm
        case pytorch_qnnp_ukernel_type_gemm:
          // 遍历每个组，将权重打包为 Q8 格式的矩阵
          for (uint32_t group = 0; group < groups; group++) {
            pytorch_pack_q8gemm_w(
                group_output_channels,
                group_input_channels,
                nr,
                nr,
                kr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
                input_zero_point,
                kernel_zero_points[0],
#endif
                kernel + group * group_output_channels * group_input_channels,
                bias + group * group_output_channels,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
                kernel_zero_points + group * group_output_channels,
#endif
                (void*)((uintptr_t)convolution->packed_weights + group * packed_group_weights_size));
          }
          break;


        case pytorch_qnnp_ukernel_type_conv:
          // 针对每个 group 进行循环，打包 Q8 卷积权重
          for (uint32_t group = 0; group < groups; group++) {
            pytorch_pack_q8conv_w(
                group_output_channels,
                kernel_size,
                group_input_channels,
                nr,
                kr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
                input_zero_point,
                kernel_zero_points[0],
#endif
                // 计算当前 group 的卷积核偏移量
                kernel +
                    group * group_output_channels * kernel_size *
                        group_input_channels,
                // 计算当前 group 的偏置项偏移量
                bias + group * group_output_channels,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
                // 当运行时量化开启时，获取当前 group 的卷积核零点
                kernel_zero_points + group * group_output_channels,
#endif
                // 计算当前 group 在 packed_weights 中的偏移量
                (void*)((uintptr_t)convolution->packed_weights + group * packed_group_weights_size));
          }
          break;


        default:
          // 不可达代码，指示未知的 ukernel 类型
          PYTORCH_QNNP_UNREACHABLE;
      }

      // 如果输入通道数大于等于 8，则设置零填充尺寸和偏移量
      if (group_input_channels >= 8) {
        zero_size = sizeof(uint8_t) * k_stride;
        zero_offset = 0;
      } else {
        zero_size = sizeof(uint8_t) * k_stride + 8;
        zero_offset = 8;
      }
      break;
    }
    default:
      // 不可达代码，指示未知的 Padding 类型
      PYTORCH_QNNP_UNREACHABLE;
  }


  // 如果需要任何形式的填充
  if (any_padding) {
    // 分配零填充缓冲区
    void* zero_buffer = malloc(zero_size);
    if (zero_buffer == NULL) {
      // 分配失败时记录错误信息并跳转到错误处理
      pytorch_qnnp_log_error(
          "failed to allocate %zu bytes for zero padding", zero_size);
      goto error;
    }
    // 使用输入零点填充零缓冲区
    memset(zero_buffer, input_zero_point, zero_size);
    convolution->zero_buffer = zero_buffer;
    // 设置零指针为零缓冲区偏移量后的位置
    convolution->zero_pointer = (void*)((uintptr_t)zero_buffer + zero_offset);
  }

  // 设置卷积结构体中各参数的值
  convolution->input_padding_depth = input_padding_depth;
  convolution->input_padding_height = input_padding_height;
  convolution->input_padding_width = input_padding_width;
  convolution->kernel_depth = kernel_depth;
  convolution->kernel_height = kernel_height;
  convolution->kernel_width = kernel_width;
  convolution->stride_depth = subsampling_depth;
  convolution->stride_height = subsampling_height;
  convolution->stride_width = subsampling_width;
  convolution->dilation_depth = dilation_depth;
  convolution->dilation_height = dilation_height;
  convolution->dilation_width = dilation_width;
  convolution->groups = groups;
  convolution->group_input_channels = group_input_channels;
  convolution->group_output_channels = group_output_channels;

  // 设置卷积结构体中的卷积核零点
  convolution->kernel_zero_point = kernel_zero_points[0];

  // 如果 ukernel 类型是 xzp_gemm
    # 如果是混合精度量化，计算重新量化参数并分配给卷积操作
    convolution->requantization_params =
        pytorch_qnnp_compute_requantization_params(
            requantization_scales[0], output_zero_point, output_min, output_max);
  } else {
    # 如果不是混合精度量化，计算卷积量化参数并分配给卷积操作
    convolution->conv_quantization_params =
        pytorch_qnnp_compute_conv_quantization_params(
            input_zero_point,
            kernel_zero_points,
            requantization_scales,
            output_zero_point,
            output_min,
            output_max);
  }

  # 设置卷积操作的内核类型
  convolution->ukernel_type = ukernel_type;
  # 设置卷积操作的数据格式为8位无符号整数
  convolution->format = pytorch_qnnp_format_quint8;

  # 设置是否按通道量化
  convolution->per_channel = per_channel;

  # 将构建好的卷积操作结构体赋值给输出指针
  *convolution_out = convolution;
  # 返回成功状态
  return pytorch_qnnp_status_success;
// 调用 pytorch_qnnp_delete_operator 函数来删除给定的卷积操作符对象
pytorch_qnnp_delete_operator(convolution);
// 返回当前函数的执行状态
return status;
}

// 创建 2D NHWC 格式的量化卷积操作符，返回操作状态
enum pytorch_qnnp_status pytorch_qnnp_create_convolution2d_nhwc_q8(
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool per_channel,
    pytorch_qnnp_operator_t* convolution_out) {
  // 调用通用的 N-dimensional NHWC 格式量化卷积创建函数，指定 is_2d 参数为 true
  return pytorch_qnnp_create_convolution_ndhwc_q8(
      0,
      input_padding_height,
      input_padding_width,
      1,
      kernel_height,
      kernel_width,
      1,
      subsampling_height,
      subsampling_width,
      1,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      input_zero_point,
      kernel_zero_points,
      kernel,
      bias,
      output_zero_point,
      output_min,
      output_max,
      flags,
      requantization_scales,
      per_channel,
      convolution_out,
      true /* is_2d? */);
}

// 创建 3D NDHWC 格式的量化卷积操作符，返回操作状态
enum pytorch_qnnp_status pytorch_qnnp_create_convolution3d_ndhwc_q8(
    uint32_t input_padding_depth,
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t kernel_depth,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_depth,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_depth,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool per_channel,
    pytorch_qnnp_operator_t* convolution_out) {
  // 调用通用的 N-dimensional NHWC 格式量化卷积创建函数，指定 is_2d 参数为 false
  return pytorch_qnnp_create_convolution_ndhwc_q8(
      input_padding_depth,
      input_padding_height,
      input_padding_width,
      kernel_depth,
      kernel_height,
      kernel_width,
      subsampling_depth,
      subsampling_height,
      subsampling_width,
      dilation_depth,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      input_zero_point,
      kernel_zero_points,
      kernel,
      bias,
      output_zero_point,
      output_min,
      output_max,
      flags,
      requantization_scales,
      per_channel,
      convolution_out,
      false /* is_2d? */);
}
    // 调用 QNNPACK 库中的函数设置 NDHWC 格式的量化卷积操作
    pytorch_qnnp_operator_t convolution,
    // 卷积操作的描述符
    size_t batch_size,
    // 批量处理的大小
    size_t input_height,
    // 输入图像的高度
    size_t input_width,
    // 输入图像的宽度
    const uint8_t* input,
    // 输入图像的指针，以 uint8_t 类型表示
    size_t input_pixel_stride,
    // 输入图像中像素的跨度（步幅）
    uint8_t* output,
    // 输出图像的指针，以 uint8_t 类型表示
    size_t output_pixel_stride,
    // 输出图像中像素的跨度（步幅）
    pthreadpool_t threadpool) {
    // 线程池对象，用于并行执行操作
      // 调用 QNNPACK 库中的函数设置 NDHWC 格式的量化卷积操作，并返回其结果
      return pytorch_qnnp_setup_convolution_ndhwc_q8(
          convolution,
          // 卷积操作的描述符
          batch_size,
          // 批量处理的大小
          1,
          // 卷积操作的维度为 1（只在时间轴上进行卷积）
          input_height,
          // 输入图像的高度
          input_width,
          // 输入图像的宽度
          input,
          // 输入图像的指针
          input_pixel_stride,
          // 输入图像中像素的跨度（步幅）
          output,
          // 输出图像的指针
          output_pixel_stride,
          // 输出图像中像素的跨度（步幅）
          threadpool);
          // 线程池对象，用于并行执行操作
    }
}
# 结束函数定义

enum pytorch_qnnp_status pytorch_qnnp_setup_convolution_ndhwc_q8(
    # 设置 QNNPACK 卷积操作符，接受以下参数：
    pytorch_qnnp_operator_t convolution,   # QNNPACK 卷积操作符指针
    size_t batch_size,                    # 批处理大小
    size_t input_depth,                   # 输入深度
    size_t input_height,                  # 输入高度
    size_t input_width,                   # 输入宽度
    const uint8_t* input,                 # 输入数据指针
    size_t input_pixel_stride,            # 输入像素步长
    uint8_t* output,                      # 输出数据指针
    size_t output_pixel_stride,           # 输出像素步长
    pthreadpool_t threadpool) {           # 线程池指针

  if (!pytorch_qnnp_params.initialized) {
    # 如果 QNNPACK 没有正确初始化，则记录错误并返回未初始化状态
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_convolution_ndhwc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    # 如果批处理大小为零，则设置卷积操作符的批处理大小为零并返回成功状态
    convolution->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  if (input_width == 0 || input_height == 0 || input_depth == 0) {
    # 如果输入宽度、高度或深度为零，则记录错误并返回无效参数状态
    pytorch_qnnp_log_error(
        "failed to setup convolution with %zux%zux%zu input: input dimensions must be non-zero",
        input_width,
        input_height,
        input_depth);
    return pytorch_qnnp_status_invalid_parameter;
  }

  # 设置卷积操作符的属性
  convolution->batch_size = batch_size;
  convolution->input_depth = input_depth;
  convolution->input_height = input_height;
  convolution->input_width = input_width;
  convolution->input = input;
  convolution->input_pixel_stride = input_pixel_stride;

  # 计算输出的深度、高度和宽度
  convolution->output_depth = compute_output_dimension(
      input_depth + convolution->input_padding_depth * 2,
      convolution->kernel_depth,
      convolution->dilation_depth,
      convolution->stride_depth);
  convolution->output_height = compute_output_dimension(
      input_height + convolution->input_padding_height * 2,
      convolution->kernel_height,
      convolution->dilation_height,
      convolution->stride_height);
  convolution->output_width = compute_output_dimension(
      input_width + convolution->input_padding_width * 2,
      convolution->kernel_width,
      convolution->dilation_width,
      convolution->stride_width);
  convolution->output = output;
  convolution->output_pixel_stride = output_pixel_stride;

  switch (convolution->ukernel_type) {
    case pytorch_qnnp_ukernel_type_gemm:
      /* Convolution maps directly to GEMM and doesn't use indirection buffer */
      return pytorch_qnnp_status_success;
    case pytorch_qnnp_ukernel_type_xzp_gemm: {
      const size_t groups = convolution->groups;
      const size_t input_size = input_depth * input_height * input_width;
      void* a_sum = (void*)realloc(
          convolution->a_sum,
          sizeof(int32_t) * batch_size * groups * input_size);
      if (a_sum == NULL) {
        # 分配行和数据总和所需的内存失败时，记录错误并返回内存不足状态
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for row sum data",
            sizeof(int32_t) * batch_size * groups * input_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      convolution->a_sum = a_sum;
      return pytorch_qnnp_status_success;
    }
    # 对于 pytorch_qnnp_ukernel_type_conv 类型的情况：
    case pytorch_qnnp_ukernel_type_conv: {
      # 从 convolution 结构中获取卷积操作的参数
      const size_t groups = convolution->groups;                 // 卷积操作的分组数
      const size_t kernel_depth = convolution->kernel_depth;     // 卷积核的深度
      const size_t kernel_height = convolution->kernel_height;   // 卷积核的高度
      const size_t kernel_width = convolution->kernel_width;     // 卷积核的宽度
      const size_t kernel_size = kernel_depth * kernel_height * kernel_width;  // 卷积核的总大小
      const size_t output_depth = convolution->output_depth;     // 输出的深度
      const size_t output_height = convolution->output_height;   // 输出的高度
      const size_t output_width = convolution->output_width;     // 输出的宽度
      const size_t output_size = output_depth * output_height * output_width;  // 输出的总大小
      const size_t output_tile_size = pytorch_qnnp_params.q8conv.mr;  // 输出瓦片的大小
      const size_t tiled_output_size = round_up(output_size, output_tile_size);  // 向上取整后的输出大小
      const size_t indirection_buffer_size =
          sizeof(void*) * batch_size * groups * tiled_output_size * kernel_size;  // 间接寻址缓冲区的大小计算

      # 重新分配或者分配 convolution->indirection_buffer 的内存空间
      const void** indirection_buffer = (const void**)realloc(
          convolution->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for indirection buffer",
            indirection_buffer_size);
        return pytorch_qnnp_status_out_of_memory;  // 内存分配失败的错误处理
      }
      convolution->indirection_buffer = indirection_buffer;  // 更新 indirection_buffer 指针

      # 初始化卷积的间接寻址
      pytorch_qnnp_indirection_init_conv3d(
          convolution, output_tile_size, tiled_output_size);
      return pytorch_qnnp_status_success;  // 返回成功状态
    }

    # 对于 pytorch_qnnp_ukernel_type_dwconv 类型的情况：
    case pytorch_qnnp_ukernel_type_dwconv: {
      # 设置深度可分离卷积的步长维度
      pytorch_qnnp_indirection_set_step_dimensions(convolution);

      # 计算间接寻址缓冲区的大小
      const size_t indirection_buffer_size = sizeof(void*) * batch_size *
          convolution->output_depth * convolution->step_depth;

      # 重新分配或者分配 convolution->indirection_buffer 的内存空间
      const void** indirection_buffer = (const void**)realloc(
          convolution->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for indirection buffer",
            indirection_buffer_size);
        return pytorch_qnnp_status_out_of_memory;  // 内存分配失败的错误处理
      }
      convolution->indirection_buffer = indirection_buffer;  // 更新 indirection_buffer 指针

      # 初始化深度可分离卷积的间接寻址
      pytorch_qnnp_indirection_init_dwconv(convolution, 0);
      return pytorch_qnnp_status_success;  // 返回成功状态
    }

    # 默认情况下不应该到达，用于标记不可达的情况
    default:
      PYTORCH_QNNP_UNREACHABLE;
  }
}



# 这行代码表示一个代码块的结束，通常与一个以前的“{”配对，用于结束一个函数、循环或条件语句的定义。
```