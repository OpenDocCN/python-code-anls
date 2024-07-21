# `.\pytorch\aten\src\ATen\native\vol2col.h`

```
#pragma once
// 使用预处理指令#pragma once确保头文件只被编译一次

#include <cstring>
// 包含C标准库中的cstring头文件，提供了字符串操作相关的函数

namespace at::native {

template <typename T>
void vol2col(
    const T* data_vol,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t depth_col,
    const int64_t height_col,
    const int64_t width_col,
    const int64_t kT,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pT,
    const int64_t pH,
    const int64_t pW,
    const int64_t dT,
    const int64_t dH,
    const int64_t dW,
    const int64_t dilationT,
    const int64_t dilationH,
    const int64_t dilationW,
    T* data_col) {
  // 初始化循环变量
  int64_t c, t, h, w;
  // 计算列方向上的通道数
  int64_t channels_col = channels * kT * kernel_height * kernel_width;
  // 循环遍历每一个输出的列向量
  for (c = 0; c < channels_col; ++c) {
    // 计算当前列向量在原始数据中的偏移量
    int64_t w_offset = c % kernel_width;
    int64_t h_offset = (c / kernel_width) % kernel_height;
    int64_t t_offset = (c / kernel_width / kernel_height) % kT;
    int64_t c_vol = c / kT / kernel_height / kernel_width;
    // 循环遍历每一列的深度
    for (t = 0; t < depth_col; ++t) {
      // 计算填充后的深度索引
      int64_t t_pad = t * dT - pT + t_offset * dilationT;
      // 循环遍历每一列的高度
      for (h = 0; h < height_col; ++h) {
        // 计算填充后的高度索引
        int64_t h_pad = h * dH - pH + h_offset * dilationH;
        // 循环遍历每一列的宽度
        for (w = 0; w < width_col; ++w) {
          // 计算填充后的宽度索引
          int64_t w_pad = w * dW - pW + w_offset * dilationW;
          // 判断索引是否在有效范围内，若在则从输入数据中复制值到输出列向量中，否则置零
          if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                data_vol[((c_vol * depth + t_pad) * height + h_pad) * width +
                         w_pad];
          else
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                0;
        }
      }
    }
  }
}

template <typename T>
void col2vol(
    const T* data_col,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t out_depth,
    const int64_t out_height,
    const int64_t out_width,
    const int64_t kT,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pT,
    const int64_t pH,
    const int64_t pW,
    const int64_t dT,
    const int64_t dH,
    const int64_t dW,
    const int64_t dilationT,
    const int64_t dilationH,
    const int64_t dilationW,
    T* data_vol) {
  // 使用memset函数将输出数据初始化为零
  memset(data_vol, 0, sizeof(T) * depth * height * width * channels);
  // 初始化循环变量
  int64_t depth_col = out_depth;
  int64_t height_col = out_height;
  int64_t width_col = out_width;
  int64_t channels_col = channels * kT * kernel_height * kernel_width;
  // 循环遍历每一个输出的列向量
  for (int64_t c = 0; c < channels_col; ++c) {
    // 计算当前列向量在原始数据中的偏移量
    int64_t w_offset = c % kernel_width;
    int64_t h_offset = (c / kernel_width) % kernel_height;
    int64_t t_offset = (c / kernel_width / kernel_height) % kT;
    int64_t c_vol = c / kT / kernel_height / kernel_width;


**继续完成剩余部分的注释**
    // 遍历3维卷积的输出空间中的每个元素
    for (int64_t t = 0; t < depth_col; ++t) {
      // 计算当前输出元素在深度（t）方向上的填充位置
      int64_t t_pad = t * dT - pT + t_offset * dilationT;
      // 遍历高度（h）方向上的每个元素
      for (int64_t h = 0; h < height_col; ++h) {
        // 计算当前输出元素在高度（h）方向上的填充位置
        int64_t h_pad = h * dH - pH + h_offset * dilationH;
        // 遍历宽度（w）方向上的每个元素
        for (int64_t w = 0; w < width_col; ++w) {
          // 计算当前输出元素在宽度（w）方向上的填充位置
          int64_t w_pad = w * dW - pW + w_offset * dilationW;
          // 检查填充后的位置是否在输入数据的有效范围内
          if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            // 将数据列（data_col）中的值累加到数据体（data_vol）中的相应位置
            data_vol
                [((c_vol * depth + t_pad) * height + h_pad) * width + w_pad] +=
                data_col
                    [((c * depth_col + t) * height_col + h) * width_col + w];
        }
      }
    }
  }


这段代码是一个三重循环，用于将一个3D卷积操作的输出数据（data_col）累加到对应的输入数据（data_vol）的指定位置上。
}

} // namespace at::native


// 结束 at::native 命名空间的定义
}
// 结束整个文件的命名空间 at
} // namespace at::native
```