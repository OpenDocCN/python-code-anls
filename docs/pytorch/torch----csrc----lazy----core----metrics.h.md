# `.\pytorch\torch\csrc\lazy\core\metrics.h`

```py
/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/master/third_party/xla_client/metrics.h
 */

#pragma once

#include <atomic>       // Atomic operations library
#include <functional>   // Function objects library
#include <map>          // Map containers library
#include <memory>       // Memory management library
#include <mutex>        // Mutual exclusion primitives library
#include <string>       // String types library
#include <vector>       // Vector containers library

#include <c10/macros/Export.h>  // Export macro for the c10 library

namespace torch {
namespace lazy {

struct TORCH_API Sample {
  Sample() = default;
  Sample(int64_t timestamp_ns, double value)
      : timestamp_ns(timestamp_ns), value(value) {}

  int64_t timestamp_ns = 0;  // Timestamp in nanoseconds
  double value = 0;          // Numeric value associated with the sample
};

using MetricReprFn = std::function<std::string(double)>;  // Type alias for function returning string representation of a double

// Class used to collect time-stamped numeric samples. The samples are stored in
// a circular buffer whose size can be configured at constructor time.
class TORCH_API MetricData {
 public:
  // Creates a new MetricData object with the internal circular buffer storing
  // max_samples samples. The repr_fn argument allow to specify a function which
  // pretty-prints a sample value.
  MetricData(MetricReprFn repr_fn, size_t max_samples);

  // Returns the total values of all the samples being posted to this metric.
  double Accumulator() const;

  size_t TotalSamples() const;

  // Adds a new sample with the given timestamp and value to the metric data.
  void AddSample(int64_t timestamp_ns, double value);

  // Returns a vector with all the current samples, from the oldest to the
  // newer. If accumulator is not nullptr, it will receive the current value of
  // the metrics' accumulator (the sum of all posted values). If total_samples
  // is not nullptr, it will receive the count of the posted values.
  std::vector<Sample> Samples(double* accumulator, size_t* total_samples) const;

  // Returns a string representation of the given value using the repr_fn_ function.
  std::string Repr(double value) const {
    return repr_fn_(value);
  }

  // Resets the metric data, clearing all samples and resetting accumulator.
  void Reset();

  // Checks if the metric data contains any valid samples.
  bool IsValid() const {
    return TotalSamples() > 0;
  }

 private:
  mutable std::mutex lock_;       // Mutex for thread-safe access to metric data
  MetricReprFn repr_fn_;          // Function for representing a double value as a string
  size_t count_ = 0;              // Current count of samples stored
  std::vector<Sample> samples_;   // Vector to store samples
  double accumulator_ = 0.0;      // Accumulated value of all posted samples
};

// Counters are a very lightweight form of metrics which do not need to track
// sample time.
class TORCH_API CounterData {
 public:
  CounterData() : value_(0) {}

  // Adds the given value to the current counter value atomically.
  void AddValue(int64_t value) {
    value_ += value;
  }

  // Returns the current value of the counter.
  int64_t Value() const {
    return value_;
  }

  // Resets the counter value to zero.
  void Reset() {
    value_ = 0;
  }

  // Checks if the counter has a valid (non-zero) value.
  bool IsValid() const {
    return value_ > 0;
  }

 private:
  std::atomic<int64_t> value_;    // Atomic variable to store counter value
};

} // namespace lazy
} // namespace torch
// 定义一个名为 MetricsArena 的类，用于管理指标和计数器的注册和获取
class TORCH_API MetricsArena {
 public:
  // 获取 MetricsArena 的单例对象
  static MetricsArena* Get();

  // 重置所有计数器的计数
  void ResetCounters();

  // 重置所有指标的数据
  void ResetMetrics();

  // 注册一个新的指标到全局的指标集合中
  void RegisterMetric(
      const std::string& name,
      MetricReprFn repr_fn,
      size_t max_samples,
      std::shared_ptr<MetricData>* data);

  // 注册一个新的计数器到全局的计数器集合中
  void RegisterCounter(
      const std::string& name,
      std::shared_ptr<CounterData>* data);

  // 遍历所有指标并执行给定的函数
  void ForEachMetric(
      const std::function<void(const std::string&, MetricData*)>& metric_func);

  // 遍历所有计数器并执行给定的函数
  void ForEachCounter(
      const std::function<void(const std::string&, CounterData*)>&
          counter_func);

  // 获取所有已注册指标的名称列表
  std::vector<std::string> GetMetricNames();

  // 根据名称获取指定的指标数据
  MetricData* GetMetric(const std::string& name);

  // 获取所有已注册计数器的名称列表
  std::vector<std::string> GetCounterNames();

  // 根据名称获取指定的计数器数据
  CounterData* GetCounter(const std::string& name);

 private:
  std::mutex lock_; // 用于保护并发访问的互斥锁
  std::map<std::string, std::shared_ptr<MetricData>> metrics_; // 存储所有指标的映射表
  std::map<std::string, std::shared_ptr<CounterData>> counters_; // 存储所有计数器的映射表
};

// 将 double 值转换为字符串形式
TORCH_API std::string MetricFnValue(double value);
// 将 double 值转换为人类可读的字节表示形式
TORCH_API std::string MetricFnBytes(double value);
// 将 double 值转换为人类可读的时间表示形式，单位为纳秒的 EPOCH 时间
TORCH_API std::string MetricFnTime(double value);

// Metric 类用于表示一个指标，可以添加样本数据并进行统计
class TORCH_API Metric {
 public:
  // 构造函数，初始化指标的名称、数据转换函数和最大样本数
  explicit Metric(
      std::string name,
      MetricReprFn repr_fn = MetricFnValue,
      size_t max_samples = 0);

  // 获取指标的名称
  const std::string& Name() const {
    return name_;
  }

  // 获取指标的累加器值
  double Accumulator() const;

  // 添加一个时间戳和值的样本
  void AddSample(int64_t timestamp_ns, double value);

  // 添加一个值的样本
  void AddSample(double value);

  // 获取所有样本及其累加器值和总样本数
  std::vector<Sample> Samples(double* accumulator, size_t* total_samples) const;

  // 根据值获取其字符串表示形式
  std::string Repr(double value) const;

 private:
  // 获取当前指标的数据对象
  MetricData* GetData() const;

  std::string name_; // 指标名称
  MetricReprFn repr_fn_; // 数据转换函数
  size_t max_samples_; // 最大样本数
  mutable std::shared_ptr<MetricData> data_ptr_; // 指标数据的智能指针
  mutable std::atomic<MetricData*> data_; // 指标数据的原子指针
};

// Counter 类用于表示一个计数器，可以增加或减少其整数值
class TORCH_API Counter {
 public:
  // 构造函数，初始化计数器的名称
  explicit Counter(std::string name);

  // 增加计数器的值
  void AddValue(int64_t value) {
    GetData()->AddValue(value);
  }

  // 获取当前计数器的值
  int64_t Value() const {
    return GetData()->Value();
  }

 private:
  // 获取当前计数器的数据对象
  CounterData* GetData() const;

  std::string name_; // 计数器名称
  mutable std::shared_ptr<CounterData> data_ptr_; // 计数器数据的智能指针
  mutable std::atomic<CounterData*> data_; // 计数器数据的原子指针
};
#define TORCH_LAZY_COUNTER(name, value)        \
  do {                                         \
    // 声明静态计数器指针，并在需要时初始化，计数器名称为给定的name，初始值为value
    static ::torch::lazy::Counter* __counter = \
        new ::torch::lazy::Counter(name);      \
    // 向计数器中添加值value
    __counter->AddValue(value);                \
  } while (0)

#define TORCH_LAZY_FN_COUNTER(ns) TORCH_LAZY_COUNTER(c10::str(ns, __func__), 1)

#define TORCH_LAZY_VALUE_METRIC(name, value)                         \
  do {                                                               \
    // 声明静态指标指针，并在需要时初始化，指标名称为给定的name，指标类型为值类型
    static ::torch::lazy::Metric* __metric =                         \
        new ::torch::lazy::Metric(name, torch::lazy::MetricFnValue); \
    // 向指标中添加值value作为样本
    __metric->AddSample(value);                                      \
  } while (0)

// 创建当前指标统计报告的函数
TORCH_API std::string CreateMetricReport();

// 创建选定指标统计报告的函数
TORCH_API std::string CreateMetricReport(
    const std::vector<std::string>& counter_names,
    const std::vector<std::string>& metric_names);

// 返回当前已注册指标名称的函数
TORCH_API std::vector<std::string> GetMetricNames();

// 检索给定指标的指标数据，如果指标不存在则返回nullptr
TORCH_API MetricData* GetMetric(const std::string& name);

// 返回当前已注册计数器名称的函数
TORCH_API std::vector<std::string> GetCounterNames();

// 检索给定计数器的计数器数据，如果计数器不存在则返回nullptr
TORCH_API CounterData* GetCounter(const std::string& name);

// 检索当前时间的纳秒表示
TORCH_API int64_t NowNs();

// 用于测量给定C++作用域内代码执行时间的工具类
class TORCH_API TimedSection {
 public:
  // 构造函数，初始化计时器对象和起始时间
  explicit TimedSection(Metric* metric) : metric_(metric), start_(NowNs()) {}

  // 析构函数，在对象销毁时计算时间差，并将时间差作为样本添加到指标中
  ~TimedSection() {
    int64_t now = NowNs();
    metric_->AddSample(now, now - start_);
  }

  // 返回已经流逝的时间，单位为秒
  double Elapsed() const {
    return 1e-9 * static_cast<double>(NowNs() - start_);
  }

 private:
  Metric* metric_; // 指向Metric对象的指针
  int64_t start_;  // 记录起始时间的变量
};

// 定义用于创建计时指标和执行时间跟踪的宏
#define TORCH_LAZY_TIMED(name)                                  \
  static torch::lazy::Metric* timed_metric =                    \
      new torch::lazy::Metric(name, torch::lazy::MetricFnTime); \
  torch::lazy::TimedSection timed_section(timed_metric)

// 将计数器和时间跟踪结合使用的宏定义
#define TORCH_LAZY_FN_COUNTER_TIMED_TRACING(ns) \
  TORCH_LAZY_FN_COUNTER(ns);                    \
  TORCH_LAZY_TIMED("LazyTracing")

} // namespace lazy
} // namespace torch
```