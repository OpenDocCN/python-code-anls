# `.\pytorch\aten\src\ATen\benchmarks\quantize_per_channel.cpp`

```py
// 包含 ATen 库和标准输入输出流的头文件
#include <ATen/ATen.h>
#include <iostream>

// 包含 Google Benchmark 库的头文件
#include <benchmark/benchmark.h>

// 声明一个基准测试函数，用于测试四维张量的逐通道量化（数据是连续存储的）
static void quantize_per_channel_4d_contiguous(benchmark::State& state) {
  // 从基准测试状态中获取批次大小、通道数、高度和宽度
  const size_t batches = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));
  const size_t height = static_cast<size_t>(state.range(2));
  const size_t width = static_cast<size_t>(state.range(3));

  // 创建一个随机初始化的四维张量 a
  at::Tensor a = at::rand({batches, channels, height, width});
  
  // 创建随机初始化的尺度向量 scales 和零点向量 zero_points
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa; // 定义一个用于存储量化后结果的张量 qa
  // 循环执行量化逐通道操作，记录执行时间
  for (auto _ : state) {
    qa = at::native::quantize_per_channel(
        a, scales, zero_points, 1, at::ScalarType::QUInt8);
  }
}

// 声明一个基准测试函数，用于测试四维张量的逐通道量化（通道存储在最后）
static void quantize_per_channel_4d_channels_last(benchmark::State& state) {
  // 从基准测试状态中获取批次大小、通道数、高度和宽度
  const size_t batches = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));
  const size_t height = static_cast<size_t>(state.range(2));
  const size_t width = static_cast<size_t>(state.range(3));

  // 创建一个随机初始化的四维张量 a，通道存储在最后
  at::Tensor a = at::rand(
      {batches, channels, height, width},
      at::TensorOptions().memory_format(at::MemoryFormat::ChannelsLast));
  
  // 创建随机初始化的尺度向量 scales 和零点向量 zero_points
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa; // 定义一个用于存储量化后结果的张量 qa
  // 循环执行量化逐通道操作，记录执行时间
  for (auto _ : state) {
    qa = at::native::quantize_per_channel(
        a, scales, zero_points, 1, at::ScalarType::QUInt8);
  }
}

// 声明一个基准测试函数，用于测试二维张量的逐通道量化
static void quantize_per_channel_2d(benchmark::State& state) {
  // 从基准测试状态中获取通道数和元素数
  const size_t channels = static_cast<size_t>(state.range(0));
  const size_t nelem = static_cast<size_t>(state.range(1));

  // 创建一个随机初始化的二维张量 a
  at::Tensor a = at::rand({channels, nelem});
  
  // 创建随机初始化的尺度向量 scales 和零点向量 zero_points
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa; // 定义一个用于存储量化后结果的张量 qa
  // 循环执行量化逐通道操作，记录执行时间
  for (auto _ : state) {
    qa = at::native::quantize_per_channel(
        a, scales, zero_points, 0, at::ScalarType::QUInt8);
  }
}

// 声明一个生成四维张量大小的辅助函数
static void GenerateSizes4d(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "C", "H", "W"});

  // 生成测试参数：批次数从 16 到 256（每次乘以 2）、通道数从 4 到 256（每次乘以 2）、高度和宽度从 4 到 256（每次乘以 2）
  for (size_t n = 16; n < 256; n *= 2) {
    for (size_t c = 4; c < 256; c *= 2) {
      for (size_t hw = 4; hw < 256; hw *= 2) {
        b->Args({n, c, hw, hw});
      }
    }
  }
}

// 声明一个生成二维张量大小的辅助函数
static void GenerateSizes2d(benchmark::internal::Benchmark* b) {
  b->ArgNames({"C", "N"});

  // 生成测试参数：通道数从 4 到 512（每次乘以 2）、元素数从 4 到 512（每次乘以 2）
  for (size_t c = 4; c < 512; c *= 2) {
    for (size_t n = 4; n < 512; n *= 2) {
      b->Args({c, n});
    }
  }
}

// 应用生成二维张量大小的辅助函数到量化逐通道二维张量基准测试
BENCHMARK(quantize_per_channel_2d)->Apply(GenerateSizes2d);

// 应用生成四维张量大小的辅助函数到量化逐通道四维张量基准测试（连续存储）
BENCHMARK(quantize_per_channel_4d_contiguous)->Apply(GenerateSizes4d);

// 应用生成四维张量大小的辅助函数到量化逐通道四维张量基准测试（通道存储在最后）
BENCHMARK(quantize_per_channel_4d_channels_last)->Apply(GenerateSizes4d);

// 声明主函数入口，运行基准测试
BENCHMARK_MAIN();
```