# `.\pytorch\aten\src\ATen\native\xnnpack\Engine.h`

```py
#pragma once

#include <ATen/core/Tensor.h>  // 包含 ATen 库中的 Tensor 类定义
#include <limits>  // 包含标准库中的数值极限定义

namespace at::native::xnnpack {

//
// Convolution
//

// 判断是否可以使用 XNNPack 进行二维卷积操作
bool use_convolution2d(
    const Tensor& input,  // 输入张量
    const Tensor& weight,  // 卷积核张量
    const at::OptionalIntArrayRef bias_sizes_opt,  // 可选的偏置大小
    const IntArrayRef padding,  // 填充
    const IntArrayRef stride,  // 步幅
    const IntArrayRef dilation,  // 空洞卷积扩张率
    const int64_t groups,  // 分组数
    const bool transposed);  // 是否是转置卷积

// 执行二维卷积操作
Tensor convolution2d(
    const Tensor& input,  // 输入张量
    const Tensor& weight,  // 卷积核张量
    const Tensor& bias,  // 偏置张量
    const IntArrayRef padding,  // 填充
    const IntArrayRef stride,  // 步幅
    const IntArrayRef dilation,  // 空洞卷积扩张率
    const int64_t groups);  // 分组数

//
// Linear
//

// 判断是否可以使用 XNNPack 进行线性操作
bool use_linear(
  const Tensor& input,  // 输入张量
  const Tensor& weight,  // 权重张量
  const Tensor& bias);  // 偏置张量

// 执行线性操作
Tensor linear(
  const Tensor& input,  // 输入张量
  const Tensor& weight,  // 权重张量
  const Tensor& bias);  // 偏置张量

//
// Max Pooling
//

// 判断是否可以使用 XNNPack 进行最大池化操作
bool use_max_pool2d(
    const Tensor& input,  // 输入张量
    const IntArrayRef kernel,  // 池化核大小
    const IntArrayRef padding,  // 填充
    IntArrayRef stride,  // 步幅
    const IntArrayRef dilation,  // 空洞卷积扩张率
    const bool ceil_mode,  // 是否向上取整模式
    const float output_min = -std::numeric_limits<float>::infinity(),  // 输出最小值，默认为负无穷大
    const float output_max = +std::numeric_limits<float>::infinity());  // 输出最大值，默认为正无穷大

// 执行最大池化操作
Tensor max_pool2d(
    const Tensor& input,  // 输入张量
    const IntArrayRef kernel,  // 池化核大小
    const IntArrayRef padding,  // 填充
    IntArrayRef stride,  // 步幅
    const IntArrayRef dilation,  // 空洞卷积扩张率
    const bool ceil_mode,  // 是否向上取整模式
    const float output_min = -std::numeric_limits<float>::infinity(),  // 输出最小值，默认为负无穷大
    const float output_max = +std::numeric_limits<float>::infinity());  // 输出最大值，默认为正无穷大

//
// Global Average Pooling
//

// 判断是否可以使用 XNNPack 进行全局平均池化操作
bool use_global_average_pool(const Tensor& input);  // 输入张量

// 执行全局平均池化操作
Tensor global_average_pool(const Tensor& input);  // 输入张量

//
// Channel Shuffle
//

// 判断是否可以使用 XNNPack 进行通道重排操作
bool use_channel_shuffle(
    const Tensor& input,  // 输入张量
    const int64_t groups);  // 分组数

// 执行通道重排操作
Tensor channel_shuffle(
    const Tensor& input,  // 输入张量
    const int64_t groups);  // 分组数

//
// Activations
//

// 判断是否可以使用 XNNPack 进行硬切线激活函数操作
bool use_hardswish(const Tensor& input);  // 输入张量

// 执行硬切线激活函数操作
Tensor hardswish(const Tensor& input);  // 输入张量

// 在原地执行硬切线激活函数操作
Tensor& hardswish_(Tensor& input);  // 输入张量

} // namespace at::native::xnnpack
```