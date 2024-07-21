# `.\pytorch\aten\src\ATen\native\quantized\ConvUtils.h`

```py
#pragma once
#include <ATen/core/List.h>
#include <ATen/native/ConvUtils.h>

namespace at::native::quantized {
namespace {
// MakeConvOutputShape used from both CPU and CUDA libraries
// and exporting symbol from torch_cpu would probably take more storage
// than duplicating implementation which likely be inlined away

// 模板函数，用于生成卷积操作的输出形状，支持不同维度的情况

template <int kSpatialDim>
at::SmallVector<int64_t, kSpatialDim + 2> MakeConvOutputShape(
    int N, // mini-batch，表示批处理中的样本数
    int M, // output channels，表示输出通道数
    const std::array<int64_t, kSpatialDim>& input_image_shape, // 输入图像的形状，对应于不同维度的数组
    const std::vector<int64_t>& kernel, // 卷积核大小的数组
    const torch::List<int64_t>& stride, // 步幅的列表
    const torch::List<int64_t>& padding, // 填充的列表
    const torch::List<int64_t>& dilation); // 膨胀的列表

// 当使用 CUDA 或者 PyTorch QNNPACK 时，特化模板为二维卷积的情况
template <>
at::SmallVector<int64_t, 4> MakeConvOutputShape<2>(
    int N, // mini-batch，表示批处理中的样本数
    int M, // output channels，表示输出通道数
    const std::array<int64_t, 2>& input_image_shape, // 输入图像的二维形状
    const std::vector<int64_t>& kernel, // 卷积核大小的数组
    const at::List<int64_t>& stride, // 步幅的列表
    const at::List<int64_t>& padding, // 填充的列表
    const at::List<int64_t>& dilation) { // 膨胀的列表
  const int H = input_image_shape[0]; // 输入图像的高度
  const int W = input_image_shape[1]; // 输入图像的宽度
  const int64_t Y_H =
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1; // 计算输出图像的高度
  const int64_t Y_W =
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1; // 计算输出图像的宽度
  return {N, M, Y_H, Y_W}; // 返回结果，包括批处理大小、输出通道数以及计算得到的输出图像大小
}

// 当使用 CUDA 或者 PyTorch QNNPACK 时，特化模板为三维卷积的情况
template <>
at::SmallVector<int64_t, 5> MakeConvOutputShape<3>(
    int N, // mini-batch，表示批处理中的样本数
    int M, // output channels，表示输出通道数
    const std::array<int64_t, 3>& input_image_shape, // 输入图像的三维形状
    const std::vector<int64_t>& kernel, // 卷积核大小的数组
    const at::List<int64_t>& stride, // 步幅的列表
    const at::List<int64_t>& padding, // 填充的列表
    const torch::List<int64_t>& dilation) { // 膨胀的列表
  const int D = input_image_shape[0]; // 输入图像的深度
  const int H = input_image_shape[1]; // 输入图像的高度
  const int W = input_image_shape[2]; // 输入图像的宽度
  const int64_t Y_D =
      (D + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1; // 计算输出图像的深度
  const int64_t Y_H =
      (H + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1; // 计算输出图像的高度
  const int64_t Y_W =
      (W + 2 * padding[2] - dilation[2] * (kernel[2] - 1) - 1) / stride[2] + 1; // 计算输出图像的宽度
  return {N, M, Y_D, Y_H, Y_W}; // 返回结果，包括批处理大小、输出通道数以及计算得到的输出图像大小
}

#endif
} // anonymous namespace
} // namespace at::native::quantized
```