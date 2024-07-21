# `.\pytorch\aten\src\ATen\benchmarks\tensor_add.cpp`

```
// 包含 ATen 库，提供张量操作的支持
#include <ATen/ATen.h>

// 包含 Google Benchmark 库，用于性能基准测试
#include <benchmark/benchmark.h>

// 定义性能测试函数，接受状态对象作为参数
static void tensor_add(benchmark::State& state) {
  // 从状态对象中获取批量大小和通道数作为大小参数
  const size_t batchSize = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  // 创建两个随机初始化的张量 a 和 b
  at::Tensor a = at::rand({batchSize, channels});
  at::Tensor b = at::rand({batchSize, channels});
  at::Tensor c;

  // 迭代执行状态中的基准测试
  for (auto _ : state) {
    // 执行张量加法操作，并将结果保存到张量 c 中
    c = a + b;
  }
}

// 定义生成测试尺寸的函数
static void GenerateSizes(benchmark::internal::Benchmark* b) {
  // 设置参数的名称为 "N" 和 "C"
  b->ArgNames({"N", "C"});

  // 使用指数增长的方式生成不同的尺寸参数
  for (size_t n = 8; n < 1024;) {
    for (size_t c = 8; c < 1024;) {
      // 将当前的 n 和 c 值作为参数添加到基准测试对象中
      b->Args({n, c});
      c *= 2;  // 每次乘以 2，增加通道数
    }
    n *= 2;    // 每次乘以 2，增加批量大小
  }
}

// 应用生成的尺寸参数到 tensor_add 函数上，并进行性能测试
BENCHMARK(tensor_add)->Apply(GenerateSizes);

// 定义主函数作为性能测试入口点
BENCHMARK_MAIN();
```