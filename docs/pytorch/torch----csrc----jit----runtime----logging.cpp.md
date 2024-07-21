# `.\pytorch\torch\csrc\jit\runtime\logging.cpp`

```
// 包含 Torch JIT 运行时日志记录相关头文件
#include <torch/csrc/jit/runtime/logging.h>

// 包含需要使用的 C++ 标准库头文件
#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

// 定义 torch::jit::logging 命名空间
namespace torch::jit::logging {

// 锁定日志记录器类的方法，用于添加统计值
void LockingLogger::addStatValue(const std::string& stat_name, int64_t val) {
  // 使用互斥锁保护临界区域
  std::unique_lock<std::mutex> lk(m);
  // 获取指定统计名称的原始计数器引用，如果不存在则创建
  auto& raw_counter = raw_counters[stat_name];
  // 增加计数器的总和及计数
  raw_counter.sum += val;
  raw_counter.count++;
}

// 获取指定名称的计数器值
int64_t LockingLogger::getCounterValue(const std::string& name) const {
  // 使用互斥锁保护临界区域
  std::unique_lock<std::mutex> lk(m);
  // 如果计数器不存在，则返回默认值 0
  if (!raw_counters.count(name)) {
    return 0;
  }
  // 获取指定名称的聚合类型，若未指定则默认为 SUM
  AggregationType type =
      agg_types.count(name) ? agg_types.at(name) : AggregationType::SUM;
  // 获取指定名称的原始计数器引用
  const auto& raw_counter = raw_counters.at(name);
  // 根据聚合类型返回相应的值：SUM 或 AVG
  switch (type) {
    case AggregationType::SUM: {
      return raw_counter.sum;
    } break;
    case AggregationType::AVG: {
      return raw_counter.sum / raw_counter.count;
    } break;
  }
  // 如果未知的聚合类型，则抛出运行时错误
  throw std::runtime_error("Unknown aggregation type!");
}

// 设置指定统计名称的聚合类型
void LockingLogger::setAggregationType(
    const std::string& stat_name,
    AggregationType type) {
  // 存储指定统计名称及其对应的聚合类型
  agg_types[stat_name] = type;
}

// 全局日志记录器的原子指针，默认为 NoopLogger 实例
std::atomic<LoggerBase*> global_logger{new NoopLogger()};

// 获取当前全局日志记录器实例的指针
LoggerBase* getLogger() {
  return global_logger.load();
}

// 设置全局日志记录器实例的指针，并返回之前的实例
LoggerBase* setLogger(LoggerBase* logger) {
  LoggerBase* previous = global_logger.load();
  // 原子操作：尝试将全局日志记录器指针设为新值，直到成功
  while (!global_logger.compare_exchange_strong(previous, logger)) {
    previous = global_logger.load();
  }
  return previous;
}

// 返回当前高分辨率时钟的时间点
JITTimePoint timePoint() {
  return JITTimePoint{std::chrono::high_resolution_clock::now()};
}

// 记录从给定时间点到当前时间的持续时间
void recordDurationSince(const std::string& name, const JITTimePoint& tp) {
  auto end = std::chrono::high_resolution_clock::now();
  // 计算时间差，单位为纳秒
  auto seconds = std::chrono::duration<double>(end - tp.point).count() * 1e9;
  // 将计时结果添加到日志记录器中
  logging::getLogger()->addStatValue(name, seconds);
}

} // namespace torch::jit::logging
```