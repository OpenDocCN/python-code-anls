# `.\pytorch\test\cpp\profiler\perf_events.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <torch/csrc/profiler/events.h>  // 包含 Torch 的性能分析器事件头文件
#include <torch/csrc/profiler/perf.h>    // 包含 Torch 的性能分析器性能头文件

double calc_pi() {
  volatile double pi = 1.0;  // 定义并初始化计算圆周率的变量 pi
  for (int i = 3; i < 100000; i += 2) {  // 计算圆周率的近似值，循环迭代
    pi += (((i + 1) >> 1) % 2) ? 1.0 / i : -1.0 / i;  // 根据 Leibniz 公式计算圆周率的逼近值
  }
  return pi * 4.0;  // 返回计算得到的圆周率值
}

TEST(ProfilerTest, LinuxPerf) {
  torch::profiler::impl::linux_perf::PerfProfiler profiler;  // 创建 Linux 性能分析器对象 profiler

  std::vector<std::string> standard_events(  // 创建标准事件列表，用于配置性能计数器
      std::begin(torch::profiler::ProfilerPerfEvents),  // 开始位置为 Torch 提供的性能事件列表的开始
      std::end(torch::profiler::ProfilerPerfEvents));   // 结束位置为 Torch 提供的性能事件列表的结束
  torch::profiler::perf_counters_t counters;  // 声明性能计数器数组 counters
  counters.resize(standard_events.size(), 0);  // 调整计数器数组大小，初始化为零

  // Use try..catch HACK to check TORCH_CHECK because we don't yet fail
  // gracefully if the syscall were to fail
  try {
    profiler.Configure(standard_events);  // 配置性能分析器使用标准事件

    profiler.Enable();  // 启动性能分析器
    auto pi = calc_pi();  // 计算圆周率，并将结果存储到 pi 中
    profiler.Disable(counters);  // 关闭性能分析器，并将计数结果存储到 counters 中
  } catch (const c10::Error&) {
    // Bail here if something bad happened during the profiling, we don't want
    // to make the test fail
    return;  // 如果在性能分析过程中发生错误，提前退出测试
  } catch (...) {
    // something else went wrong - this should be reported
    ASSERT_EQ(0, 1);  // 如果出现其他未处理的异常，标记测试失败
  }

  // Should have counted something if worked, so lets test that
  // And if it not supported the counters should be zeros.
#if defined(__ANDROID__) || defined(__linux__)
  for (auto counter : counters) {  // 遍历计数器数组
    ASSERT_GT(counter, 0);  // 断言计数器中的值大于零，表明计数成功
  }
#else /* __ANDROID__ || __linux__ */
  for (auto counter : counters) {  // 在非 Android 和 Linux 环境下
    ASSERT_EQ(counter, 0);  // 断言计数器中的值为零，表明计数器未正常工作
  }
#endif /* __ANDROID__ || __linux__ */
}

TEST(ProfilerTest, LinuxPerfNestedDepth) {
  torch::profiler::impl::linux_perf::PerfProfiler profiler;  // 创建 Linux 性能分析器对象 profiler

  // Only monotonically increasing events will work
  std::vector<std::string> standard_events(  // 创建标准事件列表，用于配置性能计数器
      std::begin(torch::profiler::ProfilerPerfEvents),  // 开始位置为 Torch 提供的性能事件列表的开始
      std::end(torch::profiler::ProfilerPerfEvents));   // 结束位置为 Torch 提供的性能事件列表的结束

  torch::profiler::perf_counters_t counters_A;  // 声明性能计数器数组 counters_A
  torch::profiler::perf_counters_t counters_B;  // 声明性能计数器数组 counters_B
  torch::profiler::perf_counters_t counters_C;  // 声明性能计数器数组 counters_C

  counters_A.resize(standard_events.size(), 0);  // 调整计数器数组大小，初始化为零
  counters_B.resize(standard_events.size(), 0);  // 调整计数器数组大小，初始化为零
  counters_C.resize(standard_events.size(), 0);  // 调整计数器数组大小，初始化为零

  // Use try..catch HACK to check TORCH_CHECK because we don't yet fail
  // gracefully if the syscall were to fail
  try {
    profiler.Configure(standard_events);  // 配置性能分析器使用标准事件

    // * = work kernel calc_pi()
    //
    // A --*---+              +--*-- A
    //         |              |
    //         |              |
    //       B +-*--+    +--*-+ B
    //              |    |
    //              |    |
    //            C +-*--+ C
    //

    profiler.Enable();  // 启动性能分析器
    auto A = calc_pi();  // 计算圆周率，并将结果存储到 A 中

    profiler.Enable();  // 启动性能分析器
    auto B = calc_pi();  // 计算圆周率，并将结果存储到 B 中

    profiler.Enable();  // 启动性能分析器
    auto C = calc_pi();  // 计算圆周率，并将结果存储到 C 中
    profiler.Disable(counters_C);  // 关闭性能分析器，并将计数结果存储到 counters_C 中

    auto B2 = calc_pi();  // 再次计算圆周率，并将结果存储到 B2 中
    profiler.Disable(counters_B);  // 关闭性能分析器，并将计数结果存储到 counters_B 中

    auto A2 = calc_pi();  // 再次计算圆周率，并将结果存储到 A2 中
    profiler.Disable(counters_A);  // 关闭性能分析器，并将计数结果存储到 counters_A 中
  } catch (const c10::Error&) {
    // Bail here if something bad happened during the profiling, we don't want
    // to make the test fail
    return;  // 如果在性能分析过程中发生错误，提前退出测试
  } catch (...) {
    // 断言失败，如果执行到这里，说明程序逻辑出现严重问题，需要报告错误
    ASSERT_EQ(0, 1);
  }
// for each counter, assert A > B > C
#if defined(__ANDROID__) || defined(__linux__)
  // 对于每个计数器，检查 A > B > C
  for (auto i = 0; i < standard_events.size(); ++i) {
    // 断言 counters_A[i] 大于 counters_B[i]
    ASSERT_GT(counters_A[i], counters_B[i]);
    // 断言 counters_A[i] 大于 counters_C[i]
    ASSERT_GT(counters_A[i], counters_C[i]);
    // 断言 counters_B[i] 大于 counters_C[i]
    ASSERT_GT(counters_B[i], counters_C[i]);
    // 断言 counters_A[i] 大于 counters_B[i] + counters_C[i]
    ASSERT_GT(counters_A[i], counters_B[i] + counters_C[i]);
  }
#else /* __ANDROID__ || __linux__ */
  // 如果不是 Android 或 Linux 系统，执行以下逻辑
  for (auto i = 0; i < standard_events.size(); ++i) {
    // 断言 counters_A[i] 等于 0
    ASSERT_EQ(counters_A[i], 0);
    // 断言 counters_B[i] 等于 0
    ASSERT_EQ(counters_B[i], 0);
    // 断言 counters_C[i] 等于 0
    ASSERT_EQ(counters_C[i], 0);
  }
#endif /* __ANDROID__ || __linux__ */
}

// 测试用例：ProfilerTest.LinuxPerfNestedMultiple
TEST(ProfilerTest, LinuxPerfNestedMultiple) {
  // 创建 PerfProfiler 对象实例
  torch::profiler::impl::linux_perf::PerfProfiler profiler;

  // 定义需要进行性能分析的标准事件
  std::vector<std::string> standard_events(
      std::begin(torch::profiler::ProfilerPerfEvents),
      std::end(torch::profiler::ProfilerPerfEvents));

  // 定义三个性能计数器数组
  torch::profiler::perf_counters_t counters_A;
  torch::profiler::perf_counters_t counters_B;
  torch::profiler::perf_counters_t counters_C;

  // 初始化计数器数组大小，并全部设为0
  counters_A.resize(standard_events.size(), 0);
  counters_B.resize(standard_events.size(), 0);
  counters_C.resize(standard_events.size(), 0);

  // 尝试配置性能分析器，使用标准事件
  try {
    profiler.Configure(standard_events);

    // 启用性能分析器开始计数
    profiler.Enable();
    auto A1 = calc_pi(); // 计算 A1 阶段的性能指标

    profiler.Enable();
    auto B1 = calc_pi(); // 计算 B1 阶段的性能指标
    auto B2 = calc_pi(); // 计算 B2 阶段的性能指标
    profiler.Disable(counters_B); // 关闭 B 阶段的性能统计，并存入 counters_B

    auto A2 = calc_pi(); // 计算 A2 阶段的性能指标

    profiler.Enable();
    auto C1 = calc_pi(); // 计算 C1 阶段的性能指标
    profiler.Disable(counters_C); // 关闭 C 阶段的性能统计，并存入 counters_C

    auto A3 = calc_pi(); // 计算 A3 阶段的性能指标
    profiler.Disable(counters_A); // 关闭 A 阶段的性能统计，并存入 counters_A
  } catch (const c10::Error&) {
    // 如果在性能分析过程中发生错误，返回，测试不失败
    return;
  } catch (...) {
    // 如果发生其他异常，报告测试失败
    ASSERT_EQ(0, 1);
  }

// for each counter, assert A > B > C
#if defined(__ANDROID__) || defined(__linux__)
  // 对于每个计数器，检查 A > B > C
  for (auto i = 0; i < standard_events.size(); ++i) {
    // 断言 counters_A[i] 大于 counters_B[i]
    ASSERT_GT(counters_A[i], counters_B[i]);
    // 断言 counters_A[i] 大于 counters_C[i]
    ASSERT_GT(counters_A[i], counters_C[i]);
    // 断言 counters_B[i] 大于 counters_C[i]
    ASSERT_GT(counters_B[i], counters_C[i]);
    // 断言 counters_A[i] 大于 counters_B[i] + counters_C[i]
    ASSERT_GT(counters_A[i], counters_B[i] + counters_C[i]);
  }
#else /* __ANDROID__ || __linux__ */
  // 如果不是 Android 或 Linux 系统，执行以下逻辑
  for (auto i = 0; i < standard_events.size(); ++i) {
    // 断言 counters_A[i] 等于 0
    ASSERT_EQ(counters_A[i], 0);
    // 断言 counters_B[i] 等于 0
    ASSERT_EQ(counters_B[i], 0);
    // 断言 counters_C[i] 等于 0
    ASSERT_EQ(counters_C[i], 0);
  }
#endif /* __ANDROID__ || __linux__ */
}
TEST(ProfilerTest, LinuxPerfNestedSingle) {
  // 创建 LinuxPerfNestedSingle 测试用例
  
  // 创建一个 PerfProfiler 对象实例
  torch::profiler::impl::linux_perf::PerfProfiler profiler;

  // 从 ProfilerPerfEvents 中获取标准事件列表
  std::vector<std::string> standard_events(
      std::begin(torch::profiler::ProfilerPerfEvents),
      std::end(torch::profiler::ProfilerPerfEvents));

  // 初始化三个性能计数器数组，大小为标准事件列表的大小，并初始化为零
  torch::profiler::perf_counters_t counters_A;
  torch::profiler::perf_counters_t counters_B;
  torch::profiler::perf_counters_t counters_C;

  counters_A.resize(standard_events.size(), 0);
  counters_B.resize(standard_events.size(), 0);
  counters_C.resize(standard_events.size(), 0);

  // 使用 try..catch 块捕获异常，以检查 TORCH_CHECK 是否失败
  try {
    // 配置性能分析器使用标准事件列表
    profiler.Configure(standard_events);

    // 启用性能分析器多次（这里启用了三次，看起来像是一种保险措施）
    profiler.Enable();
    profiler.Enable();
    profiler.Enable();

    // 执行计算 pi 的函数，并获取返回值 A1
    auto A1 = calc_pi();

    // 停止性能分析器并获取性能计数器数据到 counters_C, counters_B, counters_A
    profiler.Disable(counters_C);
    profiler.Disable(counters_B);
    profiler.Disable(counters_A);
  } catch (const c10::Error&) {
    // 如果在性能分析过程中发生错误，提前结束测试，避免测试失败
    return;
  } catch (...) {
    // 如果发生其他未知错误，报告测试失败
    ASSERT_EQ(0, 1);
  }

  // 对每个性能计数器，验证 A > B > C 的关系
#if defined(__ANDROID__) || defined(__linux__)
  for (auto i = 0; i < standard_events.size(); ++i) {
    ASSERT_GE(counters_A[i], counters_B[i]);
    ASSERT_GE(counters_A[i], counters_C[i]);
    ASSERT_GE(counters_B[i], counters_C[i]);
  }
#else /* __ANDROID__ || __linux__ */
  // 如果不是 Android 或者 Linux 系统，确保所有计数器都是零
  for (auto i = 0; i < standard_events.size(); ++i) {
    ASSERT_EQ(counters_A[i], 0);
    ASSERT_EQ(counters_B[i], 0);
    ASSERT_EQ(counters_C[i], 0);
  }
#endif /* __ANDROID__ || __linux__ */
}
```