# `.\pytorch\aten\src\ATen\benchmarks\stateful_conv1d.cpp`

```py
// 包含 Google Benchmark 库的头文件
#include <benchmark/benchmark.h>
// 包含 C10 库中的工具类 irange 的头文件
#include <c10/util/irange.h>
// 包含 Torch 库中 XNNPACK 重写的 JIT passes 的头文件
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
// 包含 Torch 库中自动微分生成的变量工厂的头文件
#include <torch/csrc/autograd/generated/variable_factories.h>
// 包含 Torch 库中 JIT 模块的 API 头文件
#include <torch/csrc/jit/api/module.h>

// 包含 vector 标准库的头文件
#include <vector>

// 定义一个静态函数 stateful_conv1d，该函数用于进行基于状态的一维卷积基准测试
static void stateful_conv1d(benchmark::State& state) {
  // 从基准测试的状态对象 state 中获取参数值并转换为合适的数据类型
  const size_t input_channels = static_cast<size_t>(state.range(0));
  const size_t output_channels = static_cast<size_t>(state.range(1));
  const size_t kernel = static_cast<size_t>(state.range(2));
  const size_t batch_size = static_cast<size_t>(state.range(3));
  const size_t width = static_cast<size_t>(state.range(4));
  const bool optimized = static_cast<bool>(state.range(5));

  // 创建一个名为 m 的 Torch JIT 模块对象
  torch::jit::Module m("m");
  // 向模块 m 注册多个参数：weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, weight_4, bias_4
  m.register_parameter("weight_1", torch::rand({output_channels, input_channels, kernel}), false);
  m.register_parameter("bias_1", torch::rand({output_channels}), false);
  m.register_parameter("weight_2", torch::rand({output_channels, output_channels, kernel}), false);
  m.register_parameter("bias_2", torch::rand({output_channels}), false);
  m.register_parameter("weight_3", torch::rand({output_channels, output_channels, kernel}), false);
  m.register_parameter("bias_3", torch::rand({output_channels}), false);
  m.register_parameter("weight_4", torch::rand({output_channels, output_channels, kernel}), false);
  m.register_parameter("bias_4", torch::rand({output_channels}), false);

  // 定义模块 m 的前向计算，使用了多次一维卷积操作
  m.define(R"(
    def forward(self, x):
      x = torch.conv1d(x, self.weight_1, self.bias_1, 1, 0, 1, 1)
      x = torch.conv1d(x, self.weight_2, self.bias_2, 1, 0, 1, 1)
      x = torch.conv1d(x, self.weight_3, self.bias_3, 1, 0, 1, 1)
      x = torch.conv1d(x, self.weight_4, self.bias_4, 1, 0, 1, 1)
      return x
  )");

  // 创建一个空的输入向量的向量 inputs，其中包含多个随机生成的输入向量
  std::vector<std::vector<torch::jit::IValue>> inputs;
  for (const auto i : c10::irange(10)) {
    inputs.emplace_back(
        {torch::jit::IValue(torch::rand({batch_size, input_channels, width}))});
  }

  // 克隆模块 m 到 m_cloned
  auto m_cloned = m.clone();
  // 对克隆的模块 m_cloned 进行一维卷积转二维卷积的转换
  torch::jit::transformConv1dToConv2d(m_cloned);
  // 对转换后的模块 m_cloned 进行移动端优化
  auto m_optimized = torch::jit::optimizeForMobile(m_cloned);
  // 定义一个 Torch JIT 的 IValue 对象用于存储模型输出
  torch::jit::IValue output;

  // 根据 optimized 的值选择使用哪个模块进行前向计算的基准测试
  if (!optimized) {
    // 非优化模式下的基准测试
    for (auto _ : state) {
      for (const auto& input : inputs) {
        output = m.forward(input);
      }
    }
  } else {
    // 优化模式下的基准测试
    for (auto _ : state) {
      for (const auto& input : inputs) {
        output = m_optimized.forward(input);
      }
    }
  }
}

// 定义一个静态函数 GenerateSizes，用于生成基准测试的参数
static void GenerateSizes(benchmark::internal::Benchmark* b) {
  // 设置基准测试的参数名称
  b->ArgNames({"Input Channels",
               "Output Channels",
               "Kernel",
               "Batch Size",
               "Width",
               "Optimized"});

  // 循环生成不同的输入通道数，每次以当前值乘以 2，直到超过 256
  for (size_t input_channels = 32; input_channels < 256; input_channels *= 2) {
    # 外层循环控制输出通道数，初始为32，每次乘以2，直到达到256
    for (size_t output_channels = 32; output_channels < 256; output_channels *= 2) {
      # 第二层循环迭代卷积核大小，范围在3到7之间
      for (const auto kernel : c10::irange(3, 8)) {
        # 第三层循环控制批量大小，从1到4
        for (const auto batch_size : c10::irange(1, 5)) {
          # 最内层循环控制输入宽度，初始为32，每次乘以2，直到达到256
          for (size_t width = 32; width < 256; width *= 2) {
            # 向基准测试框架添加参数组合，包括输入通道数、输出通道数、卷积核大小、批量大小、输入宽度和一个布尔值标志
            b->Args({input_channels, output_channels, kernel, batch_size, width, true});
            b->Args({input_channels, output_channels, kernel, batch_size, width, false});
          }
        }
      }
    }
  }
}
```