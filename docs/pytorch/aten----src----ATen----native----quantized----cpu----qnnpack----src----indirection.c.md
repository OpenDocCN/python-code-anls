# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\indirection.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

#include <fxdiv.h>

#include <qnnpack/indirection.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>

// 初始化卷积操作的间接访问缓冲区
void pytorch_qnnp_indirection_init_conv3d(
    pytorch_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size) {
  
  // 获取操作符中的间接访问缓冲区、输入数据、输入像素步幅、零指针等参数
  const void** indirection_buffer = op->indirection_buffer;
  const void* input = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride;
  const void* zero = op->zero_pointer;
  const size_t groups = op->groups;
  const size_t group_input_channels = op->group_input_channels;
  const size_t batch_size = op->batch_size;
  const size_t input_depth = op->input_depth;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_depth = op->output_depth;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t kernel_depth = op->kernel_depth;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t stride_depth = op->stride_depth;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_depth = op->dilation_depth;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_depth = op->input_padding_depth;
  const size_t input_padding_height = op->input_padding_height;
  const size_t input_padding_width = op->input_padding_width;

  // 计算输出尺寸和卷积核尺寸
  const size_t output_size = output_depth * output_height * output_width;
  const size_t kernel_size = kernel_depth * kernel_height * kernel_width;

  // 初始化用于计算输出坐标的除法器
  const struct fxdiv_divisor_size_t output_yx_divisor =
      fxdiv_init_size_t(output_height * output_width);
  const struct fxdiv_divisor_size_t output_x_divisor =
      fxdiv_init_size_t(output_width);
  
  // 针对每个分组执行操作
  for (size_t group = 0; group < groups; group++) {
    // 在这里填入实现代码
  }
}

// 初始化深度可分离卷积操作的间接访问缓冲区
void pytorch_qnnp_indirection_init_dwconv(
    pytorch_qnnp_operator_t op,
    // 定义函数参数 `batch_start`，表示当前批次的起始索引位置
    size_t batch_start) {
      // 获取操作对象 `op` 的间接缓冲区地址，并赋值给 `indirection_buffer`
      const void** indirection_buffer = op->indirection_buffer;
      // 获取操作对象 `op` 的输入数据地址，并赋值给 `input`
      const void* input = op->input;
      // 获取操作对象 `op` 的输入数据像素步幅，并赋值给 `input_pixel_stride`
      const size_t input_pixel_stride = op->input_pixel_stride;
      // 获取操作对象 `op` 的零指针，并赋值给 `zero`
      const void* zero = op->zero_pointer;
      // 获取操作对象 `op` 的批次大小，并赋值给 `batch_size`
      const size_t batch_size = op->batch_size;
      // 获取操作对象 `op` 的输入数据深度，并赋值给 `input_depth`
      const size_t input_depth = op->input_depth;
      // 获取操作对象 `op` 的输入数据高度，并赋值给 `input_height`
      const size_t input_height = op->input_height;
      // 获取操作对象 `op` 的输入数据宽度，并赋值给 `input_width`
      const size_t input_width = op->input_width;
      // 获取操作对象 `op` 的输出数据深度，并赋值给 `output_depth`
      const size_t output_depth = op->output_depth;
      // 获取操作对象 `op` 的输出数据高度，并赋值给 `output_height`
      const size_t output_height = op->output_height;
      // 获取操作对象 `op` 的输出数据宽度，并赋值给 `output_width`
      const size_t output_width = op->output_width;
      // 获取操作对象 `op` 的卷积核深度，并赋值给 `kernel_depth`
      const size_t kernel_depth = op->kernel_depth;
      // 获取操作对象 `op` 的卷积核高度，并赋值给 `kernel_height`
      const size_t kernel_height = op->kernel_height;
      // 获取操作对象 `op` 的卷积核宽度，并赋值给 `kernel_width`
      const size_t kernel_width = op->kernel_width;
      // 获取操作对象 `op` 的深度步幅，并赋值给 `stride_depth`
      const size_t stride_depth = op->stride_depth;
      // 获取操作对象 `op` 的高度步幅，并赋值给 `stride_height`
      const size_t stride_height = op->stride_height;
      // 获取操作对象 `op` 的宽度步幅，并赋值给 `stride_width`
      const size_t stride_width = op->stride_width;
      // 获取操作对象 `op` 的深度膨胀率，并赋值给 `dilation_depth`
      const size_t dilation_depth = op->dilation_depth;
      // 获取操作对象 `op` 的高度膨胀率，并赋值给 `dilation_height`
      const size_t dilation_height = op->dilation_height;
      // 获取操作对象 `op` 的宽度膨胀率，并赋值给 `dilation_width`
      const size_t dilation_width = op->dilation_width;
      // 获取操作对象 `op` 的输入数据深度填充量，并赋值给 `input_padding_depth`
      const size_t input_padding_depth = op->input_padding_depth;
      // 获取操作对象 `op` 的输入数据高度填充量，并赋值给 `input_padding_height`
      const size_t input_padding_height = op->input_padding_height;
      // 获取操作对象 `op` 的输入数据宽度填充量，并赋值给 `input_padding_width`
      const size_t input_padding_width = op->input_padding_width;
      // 获取操作对象 `op` 的深度步长，并赋值给 `step_depth`
      const size_t step_depth = op->step_depth;
      // 获取操作对象 `op` 的高度步长，并赋值给 `step_height`
      const size_t step_height = op->step_height;
      // 获取操作对象 `op` 的宽度步长，并赋值给 `step_width`
      const size_t step_width = op->step_width;
#define DW_CONV_3D_INDEX(oz, oy, ox, kz, ky, kx)                              \
  /* Output Pixel */                                                          \
  (image * output_depth + oz) * step_depth + /* slice */                      \
  oy * step_height + /* row */                                                \
  ox * step_width * kernel_height * kernel_depth + /* column */               \
  /* Kernel */                                                                \
  kx * kernel_depth * kernel_height + /* column */                            \
  ky * kernel_depth + /* row */                                               \
  kz /* slice */

for (size_t image = batch_start; image < batch_size; image++) {
    // 循环遍历每个图像的批次，从起始位置 `batch_start` 到 `batch_size`
}
}
}

void pytorch_qnnp_indirection_init_deconv2d(
    pytorch_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size) {
  const void** indirection_buffer = op->indirection_buffer;
  const void* input = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride;
  const void* zero = op->zero_pointer;
  const size_t groups = op->groups;
  const size_t group_input_channels = op->group_input_channels;
  const size_t batch_size = op->batch_size;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_height = op->input_padding_height;
  const size_t input_padding_width = op->input_padding_width;

  const size_t output_size = output_height * output_width;
  const size_t kernel_size = kernel_height * kernel_width;

  // 循环遍历每个分组
  for (size_t group = 0; group < groups; group++) {
    // 遍历每个图像（batch 中的每个图像）
    for (size_t image = 0; image < batch_size; image++) {
      // 遍历输出瓦片的起始位置
      for (size_t output_tile_start = 0; output_tile_start < tiled_output_size;
           output_tile_start += output_tile_size) {
        // 遍历每个输出瓦片的偏移量
        for (size_t output_tile_offset = 0;
             output_tile_offset < output_tile_size;
             output_tile_offset++) {
          // 计算当前瓦片在完全铺展输出中的索引
          const size_t tiled_output_index =
              output_tile_start + output_tile_offset;
          // 限制输出索引不超过最大输出大小
          const size_t output_index = min(tiled_output_index, output_size - 1);
          // 计算输出索引对应的 y 坐标和 x 坐标
          const size_t output_y = output_index / output_width;
          const size_t output_x = output_index % output_width;
          // 遍历卷积核的高度
          for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
            // 计算在输入图像中的 y 坐标
            const size_t y =
                output_y + input_padding_height - kernel_y * dilation_height;
            const size_t input_y = y / stride_height;
            // 遍历卷积核的宽度
            for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
              // 计算在输入图像中的 x 坐标
              const size_t x =
                  output_x + input_padding_width - kernel_x * dilation_width;
              const size_t input_x = x / stride_width;
              // 计算在间接缓冲区中的索引
              const size_t index = (group * batch_size + image) *
                      tiled_output_size * kernel_size +
                  output_tile_start * kernel_size +
                  (kernel_y * kernel_width + kernel_x) * output_tile_size +
                  output_tile_offset;
              // 如果坐标在输入图像内并且符合步幅要求
              if (input_y * stride_height == y && input_y < input_height &&
                  input_x * stride_width == x && input_x < input_width) {
                // 将输入像素数据的指针存入间接缓冲区
                indirection_buffer[index] = (char*)input +
                    ((image * input_height + input_y) * input_width + input_x) *
                        input_pixel_stride +
                    group * group_input_channels;
              } else {
                // 否则将零指针存入间接缓冲区
                indirection_buffer[index] = zero;
              }
            }
          }
        }
      }
    }
}

void pytorch_qnnp_indirection_init_maxpool2d(
    pytorch_qnnp_operator_t op,
    size_t batch_start) {
  // 获取操作符中的间接缓冲区、输入数据及其像素步幅等参数
  const void** indirection_buffer = op->indirection_buffer;
  const void* input = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride;
  const size_t batch_size = op->batch_size;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t pooling_height = op->kernel_height;
  const size_t pooling_width = op->kernel_width;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_height = op->input_padding_height;
  const size_t input_padding_width = op->input_padding_width;
  const size_t step_height = op->step_height;
  const size_t step_width = op->step_width;

  // 遍历每个图像及其输出特征图上的每个像素点
  for (size_t image = batch_start; image < batch_size; image++) {
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      for (size_t pooling_y = 0; pooling_y < pooling_height; pooling_y++) {
        // 计算池化窗口在输入上的起始位置，并确保在输入图像内
        const size_t input_y =
            doz(output_y * stride_height + pooling_y * dilation_height,
                input_padding_height);
        const size_t clamped_input_y = min(input_y, input_height - 1);
        for (size_t output_x = 0; output_x < output_width; output_x++) {
          for (size_t pooling_x = 0; pooling_x < pooling_width; pooling_x++) {
            // 计算池化窗口在输入上的起始位置，并确保在输入图像内
            const size_t input_x =
                doz(output_x * stride_width + pooling_x * dilation_width,
                    input_padding_width);
            const size_t clamped_input_x = min(input_x, input_width - 1);
            // 计算存储在间接缓冲区中的索引，用于查找输入数据的位置
            const size_t index =
                (image * output_height + output_y) * step_height +
                output_x * step_width * pooling_height +
                pooling_x * pooling_height + pooling_y;
            // 将输入数据的位置存储在间接缓冲区中
            indirection_buffer[index] = (char*)input +
                ((image * input_height + clamped_input_y) * input_width +
                 clamped_input_x) *
                    input_pixel_stride;
          }
        }
      }
    }
  }
}

void pytorch_qnnp_indirection_set_step_dimensions(pytorch_qnnp_operator_t op) {
  // 获取操作符中的原始核深度及核大小等参数
  const size_t original_kernel_depth = op->kernel_depth;
  const size_t kernel_depth =
      (original_kernel_depth != 0) ? original_kernel_depth : 1;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t kernel_size = kernel_depth * kernel_height * kernel_width;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;

  // 初始化步长宽度为零
  size_t step_width = 0;
  // 根据微内核类型设置步长宽度
  switch (op->ukernel_type) {
    case pytorch_qnnp_ukernel_type_dwconv:
      step_width = op->dilation_width == 1 ? op->stride_width : kernel_width;
      break;
    // 根据不同的池化类型设置步幅宽度
    switch (op->ukernel_type) {
      // 对于平均池化，步幅宽度为操作的步长和卷积核宽度的较小值
      case pytorch_qnnp_ukernel_type_average_pooling:
        step_width = min(op->stride_width, kernel_width);
        break;
      // 对于最大池化，如果膨胀宽度大于1，则步幅宽度为卷积核宽度；否则为步长和卷积核宽度的较小值
      case pytorch_qnnp_ukernel_type_max_pooling:
        step_width = op->dilation_width > 1 ? kernel_width
                                            : min(op->stride_width, kernel_width);
        break;
      // 默认情况下，不可达状态（应该不会执行到这里）
      default:
        PYTORCH_QNNP_UNREACHABLE;
    }

    // 计算步深度（depth）
    const size_t step_height = kernel_size +
        (output_width - 1) * step_width * kernel_height * kernel_depth;

    // 计算步高度（height）
    const size_t step_depth = step_height * output_height;

    // 将计算得到的步深度、步高度和步宽度保存到操作（op）对象中
    op->step_depth = step_depth;
    op->step_height = step_height;
    op->step_width = step_width;
}


注释：


# 这行代码结束了一个代码块，大括号用于表示代码块的开始和结束，这里是结束了一个未命名的代码块
```