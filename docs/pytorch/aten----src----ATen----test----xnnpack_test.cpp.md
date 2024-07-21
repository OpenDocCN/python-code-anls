# `.\pytorch\aten\src\ATen\test\xnnpack_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/types.h>  // 引入 PyTorch 的类型定义头文件
#include <torch/utils.h>  // 引入 PyTorch 的实用工具头文件

#include <ATen/native/xnnpack/Common.h>    // 引入 XNNPACK 的公共函数头文件
#include <ATen/native/xnnpack/Engine.h>    // 引入 XNNPACK 的引擎定义头文件
#include <ATen/native/xnnpack/OpContext.h> // 引入 XNNPACK 的操作上下文头文件
#include <ATen/native/xnnpack/Pooling.h>   // 引入 XNNPACK 的池化操作头文件
#include <c10/core/CPUAllocator.h>         // 引入 C10 的 CPU 内存分配器头文件
#include <c10/core/MemoryFormat.h>         // 引入 C10 的内存格式头文件

#include <atomic>             // 引入 C++ 原子操作的头文件
#include <condition_variable> // 引入 C++ 条件变量的头文件
#include <thread>             // 引入 C++ 线程的头文件

#if defined(C10_MOBILE) && defined(USE_XNNPACK)
// 检查相对误差是否符合要求
bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  // 遍历输入张量列表，找出最大值的绝对值
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  // 返回是否满足相对误差小于阈值的条件
  return diff.abs().max().item<float>() < (0.01 + 2e-2 * maxValue);
}

// 检查两个张量是否几乎相等
bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

// 检查两个张量是否完全相等
bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

// 测试 hardswish 函数
void test_hardswish(const at::Tensor& input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_hardswish(input)); // 断言使用了 XNNPACK 的 hardswish 函数
  auto result = at::native::xnnpack::hardswish(input);    // 调用 XNNPACK 的 hardswish 函数
  auto check = almostEqual(expected, result);             // 检查函数输出与预期张量几乎相等
  ASSERT_TRUE(check);                                    // 断言检查结果为真
  ASSERT_TRUE(expected.suggest_memory_format() == input.suggest_memory_format());  // 断言张量内存格式一致
}

// 原位测试 hardswish 函数
void test_hardswish_(at::Tensor input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_hardswish(input)); // 断言使用了 XNNPACK 的 hardswish 函数
  at::native::xnnpack::hardswish_(input);                 // 调用 XNNPACK 的原位 hardswish 函数
  auto check = almostEqual(expected, input);              // 检查原位操作后的张量与预期张量几乎相等
  ASSERT_TRUE(check);                                    // 断言检查结果为真
  ASSERT_TRUE(expected.suggest_memory_format() == input.suggest_memory_format());  // 断言张量内存格式一致
}

// 测试全局平均池化函数
void test_global_average_pool(at::Tensor input, const at::Tensor& expected) {
  ASSERT_TRUE(at::native::xnnpack::use_global_average_pool(input)); // 断言使用了 XNNPACK 的全局平均池化函数
  auto result = at::native::xnnpack::global_average_pool(input);    // 调用 XNNPACK 的全局平均池化函数
  auto check = almostEqual(expected, result);                       // 检查函数输出与预期张量几乎相等
  ASSERT_TRUE(check);                                               // 断言检查结果为真
}

// 测试 XNNPACK 的线性层操作
TEST(TestXNNPackOps, TestLinear) {
  constexpr std::array<int64_t, 2u> input_shape{1, 37};   // 输入张量形状
  constexpr std::array<int64_t, 2u> weight_shape{41, 37}; // 权重张量形状
  constexpr std::array<int64_t, 2u> bias_shape{1, 41};    // 偏置张量形状
  const auto input_cpu =                                   // 在 CPU 上生成随机输入张量
      at::rand(input_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight =                                      // 在 CPU 上生成随机权重张量
      at::rand(weight_shape, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias =                                        // 在 CPU 上生成随机偏置张量
      at::rand(bias_shape, at::device(at::kCPU).dtype(at::kFloat));

  const auto out_cpu = at::linear(input_cpu, weight, bias); // 执行 PyTorch 的线性层操作

  const auto xnnpack_bias = bias.view({41});                // 调整偏置张量视图以适应 XNNPACK
  ASSERT_TRUE(at::native::xnnpack::use_linear(input_cpu, weight, xnnpack_bias));  // 断言使用了 XNNPACK 的线性层函数
  const auto result =                                       // 调用 XNNPACK 的线性层函数
      at::native::xnnpack::linear(input_cpu, weight, xnnpack_bias);

  auto check = almostEqual(out_cpu, result);                // 检查 XNNPACK 结果与 CPU 结果几乎相等
  ASSERT_TRUE(check);                                       // 断言检查结果为真
}
#endif
TEST(TestXNNPackOps, TestMaxPool2d) {
  // 创建具有指定形状和数据类型的随机张量，存储在 CPU 上
  const auto in_cpu =
      at::rand({5, 13, 55, 68}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 对输入张量进行最大池化操作，返回输出张量
  const auto out_cpu =
      at::max_pool2d(in_cpu, {3, 4}, {2, 1}, {1, 1}, {1, 1}, false);
  // 使用 XNNPACK 加速库检查是否可以使用最大池化操作
  ASSERT_TRUE(at::native::xnnpack::use_max_pool2d(
      in_cpu, {3, 4}, {1, 1}, {2, 1}, {1, 1}, false));
  // 使用 XNNPACK 加速库执行最大池化操作，返回结果张量
  const auto result = at::native::xnnpack::max_pool2d(
      in_cpu, {3, 4}, {1, 1}, {2, 1}, {1, 1}, false);

  // 检查两个张量是否几乎相等
  auto check = almostEqual(out_cpu, result);
  // 断言检查结果为真
  ASSERT_TRUE(check);
}

TEST(TestXNNPackOps, TestConvolution2d) {
  // 定义卷积操作的分组数
  constexpr int64_t groups = 1;
  // 定义卷积操作的步幅
  constexpr std::array<int64_t, 2u> stride{2, 2};
  // 定义卷积操作的填充
  constexpr std::array<int64_t, 2u> padding{1, 1};
  // 定义卷积操作的扩展
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  // 定义输入张量的结构体，包括批次数、通道数、宽度和高度
  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{1, 3, 8, 8};

  // 定义权重张量的结构体，包括输出通道数、输入通道数、宽度和高度
  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{1, input.channels, 3, 3};

  // 创建具有指定形状和数据类型的随机输入张量，存储在 CPU 上
  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  // 创建具有指定形状和数据类型的随机权重张量，存储在 CPU 上
  const auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  // 创建具有指定形状和数据类型的随机偏置张量，存储在 CPU 上
  const auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  // 使用 PyTorch 提供的 conv2d 函数执行卷积操作，返回输出张量
  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  // 使用 XNNPACK 加速库检查是否可以使用卷积操作
  ASSERT_TRUE(at::native::xnnpack::use_convolution2d(
      input_cpu,
      weights_cpu,
      weights.output_channels,
      padding,
      stride,
      dilation,
      groups,
      false));
  // 使用 XNNPACK 加速库执行卷积操作，返回结果张量
  const auto result = at::native::xnnpack::convolution2d(
      input_cpu, weights_cpu, bias_cpu, padding, stride, dilation, groups);
  // 检查两个张量是否几乎相等
  auto check = almostEqual(output_cpu, result);
  // 断言检查结果为真
  ASSERT_TRUE(check);
}
TEST(TestXNNPackOps, TestHardSwish) {
  // 定义输入张量，此处为 2x2 的浮点数张量
  auto in = torch::tensor({{1, 1}, {1, 1}}, {torch::kFloat32});
  // 对输入张量进行索引，选择所有行的第一列
  auto in_slice = in.index({"...", 0});

  // 定义输入和预期结果的对，每对包含输入张量和对应的预期结果张量
  std::vector<std::pair<at::Tensor, at::Tensor>> input_result_pairs = {
      // 第一组：输入为 {1, 2, 3, 4, 5}，预期结果为 {0.6667, 1.6667, 3.0000, 4.0000, 5.0000}
      {torch::tensor({1, 2, 3, 4, 5}, {torch::kFloat32}),
       torch::tensor(
           {0.6667, 1.6667, 3.0000, 4.0000, 5.0000}, {torch::kFloat32})},
      // 第二组：输入为 {0.3330}，预期结果为 {0.1850}
      {torch::tensor({0.3330}, {torch::kFloat32}),
       torch::tensor({0.1850}, {torch::kFloat32})},
      // 第三组：输入为 2x3 的张量，预期结果为经过 HardSwish 运算后的张量
      {torch::tensor({{0.4523, 0.8131, 0.9829}, {0.0782, 0.7395, 0.0787}}),
       torch::tensor({{0.2602, 0.5167, 0.6525}, {0.0401, 0.4609, 0.0404}})},
      // 第四组：输入为 in_slice，预期结果为 {0.6667, 0.6667}
      {in_slice, torch::tensor({0.6667, 0.6667}, {torch::kFloat32})},
      // 第五组：输入为 2x2x2x2 的张量，按 ChannelsLast 连续布局
      {torch::tensor({{{{0.4993, 0.3835}, {0.3163, 0.2348}},
                       {{0.4705, 0.4129}, {0.9314, 0.0631}}},
                      {{{0.0030, 0.5656}, {0.1413, 0.1943}},
                       {{0.1380, 0.1985}, {0.2746, 0.8109}}}})
           .contiguous(at::MemoryFormat::ChannelsLast),
       torch::tensor({{{{0.2912, 0.2163}, {0.1748, 0.1266}},
                       {{0.2722, 0.2349}, {0.6103, 0.0322}}},
                      {{{0.0015, 0.3361}, {0.0740, 0.1034}},
                       {{0.0722, 0.1058}, {0.1499, 0.5150}}}})
           .contiguous(at::MemoryFormat::ChannelsLast)}};

  // 对于每个输入结果对，分别调用 test_hardswish 和 test_hardswish_ 进行测试
  for (const auto& input_result : input_result_pairs) {
    test_hardswish(input_result.first, input_result.second);
    test_hardswish_(input_result.first, input_result.second);
  }
}

TEST(TestXNNPackOps, TestConvolution2dMultiThreaded) {
  // 定义卷积操作的分组数为 1
  constexpr int64_t groups = 1;

  // 定义输入张量的结构体，包含批次数、通道数、宽度和高度
  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{1, 3, 8, 8};

  // 定义权重张量的结构体，包含输出通道数、输入通道数、宽度和高度
  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{1, input.channels, 3, 3};

  // 生成随机输入张量，并指定在 CPU 上的数据类型和设备
  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  // 生成随机权重张量，并指定在 CPU 上的数据类型和设备
  auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  // 生成随机偏置张量，并指定在 CPU 上的数据类型和设备
  auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  // 创建 XNNPackConv2dOpContext 上下文对象，包含权重、偏置、步长、填充等信息
  auto context = at::native::xnnpack::XNNPackConv2dOpContext::create_context(
      std::move(weights_cpu), std::move(bias_cpu), {1, 1}, {2, 2}, {1, 1}, groups, c10::nullopt, c10::nullopt);
  // 计数器，用于追踪并发执行的任务数量
  std::atomic<int64_t> count{0};
  // 并行工作线程数目
  int64_t num_workers = 5;
  // 互斥锁，用于线程间同步
  std::mutex lock;
  // 条件变量，用于线程等待和唤醒
  std::condition_variable cond;
  // 定义同步并运行卷积操作的函数，参数为输入张量的高度和宽度
  auto sync_and_run_conv = [&](int64_t h, int64_t w) -> at::Tensor
  {
    // 生成随机输入张量，大小为 1x3xh x w，并指定在 CPU 上的数据类型和设备
    auto input_tensor = at::randn({1, 3, h, w}, at::device(at::kCPU).dtype(at::kFloat));
    // 自增计数值
    int64_t count_val = ++count;
    // 如果当前计数值小于工作线程数目
    if (count_val < num_workers) {
      // 使用互斥锁锁定，确保线程安全
      std::unique_lock<std::mutex> g(lock);
      // 当计数值小于工作线程数目时，线程等待条件变量
      while ((count_val = count.load()) < num_workers) {
        cond.wait(g, [&]() {
            // 在等待过程中，重新获取计数值
            auto new_val = count.load();
            // 返回是否满足条件（计数值大于等于工作线程数目）
            return new_val >= num_workers;
        });
      }
    } else {
      // 如果当前计数值不小于工作线程数目，唤醒所有等待中的线程
      std::unique_lock<std::mutex> g(lock);
      cond.notify_all();
    }
    // 运行上下文中的任务，总共运行30次
    for (int64_t i = 0; i < 30; i++) {
      context->run(input_tensor);
    }
    // 返回上下文中运行输入张量后的结果
    return context->run(input_tensor);
  };

  // 定义一个lambda函数conv，调用sync_and_run_conv进行同步卷积运算
  auto conv = [sync_and_run_conv](int64_t h, int64_t w) -> at::Tensor
  {
    return sync_and_run_conv(h, w);
  };

  // 创建5个线程，分别执行conv函数，传入不同的参数
  std::thread t1(conv, 16, 16);
  std::thread t2(conv, 12, 12);
  std::thread t3(conv, 20, 20);
  std::thread t4(conv, 22, 22);
  std::thread t5(conv, 8, 8);

  // 等待所有线程执行完毕
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
}

TEST(TestXNNPackOps, TestGlobal) {
  // 定义测试用例输入和期望输出的对
  std::vector<std::pair<at::Tensor, at::Tensor>> input_result_pairs = {
      // 第一个测试用例的输入和期望输出
      {torch::tensor(
           {{{{0.0852, 0.7312, 0.9943, 0.7105},
              {0.0956, 0.9072, 0.3124, 0.9362},
              {0.5878, 0.8883, 0.5086, 0.9494}},
             {{0.1056, 0.4968, 0.7740, 0.7593},
              {0.8519, 0.3543, 0.8078, 0.5517},
              {0.1413, 0.4608, 0.1706, 0.0314}}}},
           {torch::kFloat32}),  // 输入张量数据类型为 Float32
       torch::tensor({{{{0.6422}}, {{0.4588}}}}, {torch::kFloat32})},  // 期望输出张量及其数据类型

      // 第二个测试用例的输入和期望输出
      {torch::tensor(
           {{{{0.0280, 0.9073}, {0.2103, 0.5298}},
             {{0.5335, 0.9901}, {0.2902, 0.2955}}},
            {{{0.2363, 0.7024}, {0.7903, 0.8260}},
             {{0.3802, 0.5959}, {0.5749, 0.8855}}}},
           {torch::kFloat32}),  // 输入张量数据类型为 Float32
       torch::tensor(
           {{{{0.4188}}, {{0.5273}}}, {{{0.6388}}, {{0.6091}}}},  // 期望输出张量及其数据类型
           {torch::kFloat32})}
  };

  // 对每个测试用例进行循环测试
  for (const auto& input_result : input_result_pairs) {
    // 调用函数 test_global_average_pool 测试全局平均池化函数
    test_global_average_pool(input_result.first, input_result.second);
  }
}

int main(int argc, char* argv[]) {
  // 设置默认的分配器为移动设备，用于测试复制和非复制情况
  c10::SetCPUAllocator(c10::GetDefaultMobileCPUAllocator(), /*priority*/ 100);
  // 初始化 Google Test 框架
  ::testing::InitGoogleTest(&argc, argv);
  // 运行所有的测试，并返回结果
  return RUN_ALL_TESTS();
}
#endif
```