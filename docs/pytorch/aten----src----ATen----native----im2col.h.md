# `.\pytorch\aten\src\ATen\native\im2col.h`

```
#pragma once
// 防止头文件重复包含的预处理指令

#include <ATen/core/Tensor.h>
// 包含张量相关的核心头文件
#include <ATen/TensorUtils.h>
// 包含张量工具函数的头文件
#include <ATen/Utils.h>
// 包含ATen库的实用功能头文件
#include <ATen/Parallel.h>
// 包含ATen库的并行处理功能头文件
#include <ATen/native/cpu/utils.h>
// 包含CPU相关的实用函数头文件
#include <c10/util/irange.h>
// 包含C10库中的迭代范围函数头文件

#include <algorithm>
// 包含标准算法库的头文件

namespace at::native {

template <typename T>
static void im2col(
    const T* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_col,
    bool is_channels_last = false) {
  const int64_t height_col = output_height;
  // 列化后的输出高度等于指定的输出高度
  const int64_t width_col = output_width;
  // 列化后的输出宽度等于指定的输出宽度
  const int64_t channels_col = channels * kernel_h * kernel_w;
  // 列化后的通道数等于输入通道数乘以卷积核的高度和宽度

  if (is_channels_last) {
    // 如果是通道在最后的格式
    at::parallel_for(0, height_col * width_col, 0, [&](int64_t begin, int64_t end) {
      int64_t h_col{0}, w_col{0};
      // 初始化列化后的高度和宽度索引
      data_index_init(begin, h_col, height_col, w_col, width_col);

      for (const auto i_col : c10::irange(begin, end)) {
        // 对每个列化后的像素进行遍历
        for (const auto h_offset : c10::irange(kernel_h)) {
          // 对卷积核的高度维度进行遍历
          int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
          // 计算输入图像的高度索引
          for (const auto w_offset : c10::irange(kernel_w)) {
            // 对卷积核的宽度维度进行遍历
            int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
            // 计算输入图像的宽度索引

            const T* slice_im = data_im + (h_im * width + w_im) * channels;
            // 获取输入图像中的切片
            T* slice_col = data_col + (i_col * kernel_h * kernel_w + h_offset * kernel_w + w_offset) * channels;
            // 获取列化后的输出数据中的切片

            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
              std::copy_n(slice_im, channels, slice_col);
              // 如果输入图像切片在有效范围内，则复制数据到列化后的输出中
            } else {
              std::fill_n(slice_col, channels, T(0));
              // 否则，填充列化后的输出切片为零
            }
          }
        }

        // 移动到下一个索引位置
        data_index_step(h_col, height_col, w_col, width_col);
      }
    });
  } else {
    // 如果是通道在前的格式
    at::parallel_for(0, channels_col, 0, [&](int64_t begin, int64_t end) {
      int64_t c_im{0}, h_offset{0}, w_offset{0};
      // 初始化输入通道、卷积核高度偏移和宽度偏移索引
      data_index_init(begin, c_im, channels, h_offset, kernel_h, w_offset, kernel_w);

      for (const auto c_col : c10::irange(begin, end)) {
        // 对每个列化后的通道进行遍历
        for (const auto h_col : c10::irange(height_col)) {
          // 对列化后的输出高度维度进行遍历
          int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
          // 计算输入图像的高度索引
          for (const auto w_col : c10::irange(width_col)) {
            // 对列化后的输出宽度维度进行遍历
            int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
            // 计算输入图像的宽度索引
            data_col[(c_col * height_col + h_col) * width_col + w_col] =
                (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im[(c_im * height + h_im) * width + w_im]
                : static_cast<T>(0);
            // 将输入图像数据复制到列化后的输出数据中，或者在无效范围内填充为零
          }
        }

        // 移动到下一个索引位置
        data_index_step(c_im, channels, h_offset, kernel_h, w_offset, kernel_w);
      }
    });
  }
}

template <typename T>
static void col2im(
    const T* data_col,
    // 使用 std::fill_n 函数将 data_im 数组初始化为 0，数组大小为 height * width * channels
    std::fill_n(data_im, height * width * channels, T(0));

    // 计算输出列的高度和宽度
    const int64_t height_col = output_height;
    const int64_t width_col = output_width;

    // 计算输出列的通道数，等于输入通道数乘以 kernel_h 乘以 kernel_w
    const int64_t channels_col = channels * kernel_h * kernel_w;

    // 如果输入数据是 channels last 格式
    if (is_channels_last) {
        // 遍历输出列的高度
        for (const auto h_col : c10::irange(height_col)) {
            // 遍历输出列的宽度
            for (const auto w_col : c10::irange(width_col)) {
                // 遍历卷积核的高度偏移量
                for (const auto h_offset : c10::irange(kernel_h)) {
                    // 计算输入图像的高度坐标
                    int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
                    // 遍历卷积核的宽度偏移量
                    for (const auto w_offset : c10::irange(kernel_w)) {
                        // 计算输入图像的宽度坐标
                        int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                        // 获取当前切片的输入图像数据起始位置
                        T* slice_im = data_im + (h_im * width + w_im) * channels;
                        // 获取当前切片的输出列数据起始位置
                        const T* slice_col = data_col + ((h_col * width_col + w_col) * kernel_h * kernel_w
                            + h_offset * kernel_w + w_offset) * channels;

                        // 如果坐标在输入图像范围内，则对应位置相加
                        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
                            std::transform(slice_col, slice_col + channels, slice_im, slice_im, std::plus<T>());
                        }
                    }
                }
            }
        }
    } else {
        // 如果输入数据不是 channels last 格式
        // 遍历输出列的通道
        for (const auto c_col : c10::irange(channels_col)) {
            // 计算当前通道在输入图像中的宽度偏移量
            int64_t w_offset = c_col % kernel_w;
            // 计算当前通道在输入图像中的高度偏移量
            int64_t h_offset = (c_col / kernel_w) % kernel_h;
            // 计算当前通道在输入图像中的通道索引
            int64_t c_im = c_col / kernel_h / kernel_w;

            // 遍历输出列的高度
            for (const auto h_col : c10::irange(height_col)) {
                // 计算输入图像的高度坐标
                int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
                // 遍历输出列的宽度
                for (const auto w_col : c10::irange(width_col)) {
                    // 计算输入图像的宽度坐标
                    int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                    // 如果坐标在输入图像范围内，则累加对应位置的值
                    if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
                        data_im[(c_im * height + h_im) * width + w_im] +=
                            data_col[(c_col * height_col + h_col) * width_col + w_col];
                }
            }
        }
    }
}

// 结束 at::native 命名空间
} // namespace at::native
```