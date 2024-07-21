# `.\pytorch\torch\csrc\utils\throughput_benchmark-inl.h`

```
// 预处理指令，指示编译器只包含本头文件一次
#pragma once

// 包含随机数生成、线程管理等标准库头文件
#include <random>
#include <thread>

// 包含 PyTorch 的性能分析和 Python 绑定的相关头文件
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

// 包含 ATen 的并行处理相关头文件
#include <ATen/Parallel.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/irange.h>

// 声明命名空间 torch::throughput_benchmark::detail
namespace torch::throughput_benchmark::detail {

// BenchmarkHelper 类模板的 benchmark 成员函数定义
template <class Input, class Output, class Model>
BenchmarkExecutionStats BenchmarkHelper<Input, Output, Model>::benchmark(
    const BenchmarkConfig& config) const {
  // 检查初始化标志
  CHECK(initialized_);

  // 检查配置中线程数为 1，仅支持由调用方进行的并行化
  TORCH_CHECK(
      config.num_worker_threads == 1,
      "Only parallelization by callers is supported");

  // 输出当前的并行信息
  LOG(INFO) << at::get_parallel_info();

  // 为每个调用线程预先生成输入数据，以避免基准测试运行时的额外开销
  std::vector<std::vector<Input>> thread_inputs(config.num_calling_threads);
  std::vector<size_t> input_iters(config.num_calling_threads);
  {
    // 使用随机设备初始化随机数生成器
    std::random_device seeder;
    std::mt19937 engine(seeder());

    // 检查输入数据集非空
    TORCH_CHECK(
        !inputs_.empty(),
        "Please provide benchmark inputs."
        "Did you forget to call add_input()? ");

    // 均匀分布，用于随机选择输入数据
    std::uniform_int_distribution<int> dist(0, inputs_.size() - 1);

    // 遍历每个调用线程
    for (const auto thread_id : c10::irange(config.num_calling_threads)) {
      // 为每个线程生成 num_iters + num_warmup_iters 次输入数据
      for (const auto i [[maybe_unused]] :
           c10::irange(config.num_iters + config.num_warmup_iters)) {
        thread_inputs[thread_id].push_back(cloneInput(inputs_[dist(engine)]));
      }
      // 初始化输入迭代器
      input_iters[thread_id] = 0;
    }
  }

  // 初始化互斥量和条件变量
  std::mutex m;
  std::condition_variable worker_main_cv;
  std::condition_variable main_worker_cv;

  // 初始化原子计数器和标志变量
  int64_t initialized{0};
  int64_t finished{0};
  bool start{false};
  std::atomic<int64_t> num_attempted_iters{0};

  // 创建线程对象的容器
  std::vector<std::thread> callers;
  callers.reserve(config.num_calling_threads);

  // 获取当前线程的梯度开启状态和局部调度键集
  bool tls_grad_enabled = c10::GradMode::is_enabled();
  c10::impl::LocalDispatchKeySet tls_key_set =
      c10::impl::tls_local_dispatch_key_set();

  // 遍历每个调用线程，创建线程并执行 benchmark 函数
  for (const auto thread_id : c10::irange(config.num_calling_threads)) {
    {
      // 向 callers 向量添加一个 lambda 函数，该函数会被后续的线程执行
      callers.emplace_back([&, thread_id]() {
        // 使用条件变量作为屏障，确保每个线程在开始测量之前执行所需的预热迭代
        c10::GradMode::set_enabled(tls_grad_enabled);
        c10::impl::_force_tls_local_dispatch_key_set(tls_key_set);
    
        // 执行预热迭代，确保线程处于稳定状态
        for (const auto j : c10::irange(config.num_warmup_iters)) {
          (void)j;  // 忽略未使用的变量 j
          runOnce(std::move(thread_inputs[thread_id][input_iters[thread_id]]));
          ++input_iters[thread_id];
        }
    
        {
          std::unique_lock<std::mutex> lock(m);
          ++initialized;  // 增加已初始化线程的计数
          worker_main_cv.notify_one();  // 通知主线程有线程已初始化完成
    
          // 等待 start 标志变为 true，以开始真正的测量
          while (!start) {
            main_worker_cv.wait(lock);
          }
        }
    
        // 输出线程开始前向传播的信息
        LOG(INFO) << "Starting forward thread " << thread_id;
    
        // 执行正式的迭代测量
        while (num_attempted_iters.fetch_add(1) < config.num_iters) {
          runOnce(std::move(thread_inputs[thread_id][input_iters[thread_id]]));
          ++input_iters[thread_id];
        }
    
        {
          std::unique_lock<std::mutex> lock(m);
          ++finished;  // 增加已完成迭代的线程计数
          worker_main_cv.notify_one();  // 通知主线程有线程已完成所有迭代
    
          // 输出线程完成前向传播的信息，包括完成线程数的统计
          LOG(INFO) << "Shutting down forward thread " << thread_id
                    << ". Total number of finished threads: " << finished;
        }
      });
    }
    
    using Clock = std::chrono::high_resolution_clock;
    using RecordProfile = torch::autograd::profiler::RecordProfile;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint start_time;
    
    std::unique_ptr<RecordProfile> profiler_guard;
    {
      std::unique_lock<std::mutex> lock(m);
    
      // 等待所有调用线程都初始化完成
      while (initialized != config.num_calling_threads) {
        worker_main_cv.wait(lock);
      }
    
      // 如果配置了分析器输出路径，则使用 Autograd 分析器并创建记录器
      if (!config.profiler_output_path.empty()) {
        LOG(INFO) << "Using Autograd profiler. Trace will be saved to "
                  << config.profiler_output_path;
        profiler_guard =
            std::make_unique<RecordProfile>(config.profiler_output_path);
      }
    
      // 输出线程开始启动的信息，并标记开始时间
      LOG(INFO) << "Starting threads";
      start = true;
      start_time = Clock::now();
    }
    
    // 唤醒所有等待中的主线程
    main_worker_cv.notify_all();
    
    {
      std::unique_lock<std::mutex> lock(m);
      // 这里继续执行其它操作
    }
  // 等待 worker_main_cv 条件变量，直到 finished 等于 config.num_calling_threads
  worker_main_cv.wait(
      lock, [&]() { return finished == config.num_calling_threads; });
}

// 记录结束时间点
auto end_time = std::chrono::high_resolution_clock::now();

// 重置性能分析器的状态
profiler_guard.reset();

// 输出日志信息，表示基准测试已完成
LOG(INFO) << "Finished benchmark";

// 创建 BenchmarkExecutionStats 对象用于保存基准测试统计数据
BenchmarkExecutionStats stats;

// 计算总运行时间（毫秒），并将纳秒转换为毫秒
// NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
float total_time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          end_time - start_time)
                          .count() /
    1000.0 / 1000.0;

// 计算平均延迟时间（毫秒），使用 config.num_iters 代替 num_attempted_iters，
// 因为它更能代表真正的工作量。每个调用线程的最后尝试迭代并不代表真正的工作（例如运行模型）
stats.latency_avg_ms =
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    total_time_ms * config.num_calling_threads / config.num_iters;

// 将 config.num_iters 存入 stats 对象，表示迭代次数
stats.num_iters = config.num_iters;

// 等待所有调用线程结束
for (auto& t : callers) {
  t.join();
}

// 返回基准测试的统计数据
return stats;
}

} // namespace torch::throughput_benchmark::detail
```