# `.\pytorch\aten\src\ATen\native\FractionalMaxPooling.h`

```py
#pragma once
// 引入 ATen 库中的 Tensor 类
#include <ATen/core/Tensor.h>
// 引入 ATen 库中的 TensorUtils.h 文件
#include <ATen/TensorUtils.h>
// 引入 c10/util/irange.h 文件，用于生成范围内的整数序列
#include <c10/util/irange.h>

// 定义 at::native 命名空间
namespace at::native {

// 声明一个模板函数，用于生成间隔序列
template<typename scalar_t>
// 标记函数为内联函数，提高执行效率
inline std::vector<int> generate_intervals(
    scalar_t sample,    // 输入参数：采样值
    int64_t inputSize,  // 输入参数：输入尺寸
    int64_t outputSize, // 输入参数：输出尺寸
    int64_t poolSize) { // 输入参数：池化尺寸

  // 创建一个整数向量，长度为输出尺寸
  std::vector<int> sequence(outputSize);

  // 如果输出尺寸大于1，则执行以下逻辑
  if (outputSize > 1) {
    // 计算 alpha 值，用于确定间隔
    scalar_t alpha = static_cast<scalar_t>(inputSize - poolSize) /
      static_cast<scalar_t>(outputSize - 1);

    // 遍历生成间隔序列
    for (const auto i : c10::irange(outputSize - 1)) {
      sequence[i] =
        static_cast<int>((i + sample) * alpha) - static_cast<int>(sample * alpha);
    }
  }

  // 如果输出尺寸大于0，则处理最后一个元素
  if (outputSize > 0) {
    sequence[outputSize - 1] = inputSize - poolSize;
  }

  // 返回生成的间隔序列
  return sequence;
}

// 声明一个模板函数，用于检查分数最大池化的输入形状
template <int64_t ndim>
// 标记函数为内联函数，提高执行效率
inline void fractional_max_pool_check_shape(
    const Tensor& input,         // 输入参数：输入张量
    const Tensor& randomSamples) { // 输入参数：随机采样张量

  // 检查随机采样张量的数据类型与输入张量相同
  TORCH_CHECK(
      input.scalar_type() == randomSamples.scalar_type(),
      "Expect _random_samples to have the same dtype as input");

  // 获取随机采样张量的维度
  int64_t ndimension = randomSamples.ndimension();
  // 检查随机采样张量的维度为3
  TORCH_CHECK(
      ndimension == 3,
      "Expect _random_samples to have 3 dimensions, got ", ndimension);

  // 获取随机采样张量的尺寸
  int64_t N = randomSamples.size(0);
  int64_t C = randomSamples.size(1);
  int64_t D = randomSamples.size(2);

  int64_t input_batch, input_channel;
  // 根据 ndim 的值选择相应的处理分支
  if (ndim == 2) {
    // 如果是二维分数最大池化，检查输入张量的维度
    if (input.ndimension() == 3) {
      input_batch = 1;
      input_channel = input.size(0);
    } else {
      input_batch = input.size(0);
      input_channel = input.size(1);
    }
  } else {
    // 如果是三维分数最大池化，检查输入张量的维度
    if (input.ndimension() == 4) {
      input_batch = 1;
      input_channel = input.size(0);
    } else {
      input_batch = input.size(0);
      input_channel = input.size(1);
    }
  }

  // 检查随机采样张量的第一维度大小不小于输入张量的批次大小
  TORCH_CHECK(
      N >= input_batch,
      "Expect _random_samples.size(0) no less then input batch size.");
  // 检查随机采样张量的第二维度大小与输入张量的通道数相同
  TORCH_CHECK(
      C == input_channel,
      "Expect _random_samples.size(1) equals to input channel size.");
  // 检查随机采样张量的第三维度大小与 ndim 相同
  TORCH_CHECK(
      D == ndim,
      "Expect _random_samples.size(2) equals to ", ndim, "; got ", D, ".");
}

} // namespace at::native
```