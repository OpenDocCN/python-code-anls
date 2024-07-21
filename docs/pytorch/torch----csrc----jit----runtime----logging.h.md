# `.\pytorch\torch\csrc\jit\runtime\logging.h`

```
#pragma once

#include <chrono>  // 引入时间相关的头文件
#include <mutex>   // 引入互斥锁的头文件
#include <string>  // 引入字符串处理的头文件
#include <unordered_map>  // 引入无序映射的头文件
#include <vector>  // 引入向量的头文件

#include <torch/csrc/Export.h>  // 引入 Torch 导出相关的头文件

namespace torch::jit::logging {  // 进入 torch::jit::logging 命名空间

class LoggerBase {  // 定义 LoggerBase 类
 public:
  TORCH_API virtual void addStatValue(
      const std::string& stat_name,
      int64_t val) = 0;  // 纯虚函数，用于添加统计值
  virtual ~LoggerBase() = default;  // 虚析构函数
};

TORCH_API LoggerBase* getLogger();  // 获取 LoggerBase 对象的函数声明
TORCH_API LoggerBase* setLogger(LoggerBase* logger);  // 设置 LoggerBase 对象的函数声明

// No-op logger. This is the default and is meant to incur almost no runtime
// overhead.
// 无操作日志记录器，用于几乎没有运行时开销的默认情况。

class NoopLogger : public LoggerBase {  // NoopLogger 类继承自 LoggerBase
 public:
  void addStatValue(const std::string& stat_name, int64_t val) override {}  // 重载添加统计值的函数，实现为空操作
  ~NoopLogger() override = default;  // 默认析构函数
};

// Trivial locking logger. Pass in an instance of this to setLogger() to use it.
// This keeps track of the sum of all statistics.
//
// NOTE: this is not written in a scalable way and should probably only be used
// in the single-threaded case or for testing.
// 简单的锁定日志记录器。将此实例传递给 setLogger() 以使用它。
// 这将跟踪所有统计数据的总和。
//
// 注意：这种方式不是按可扩展的方式编写的，可能只适用于单线程情况或测试用途。

class TORCH_API LockingLogger : public LoggerBase {  // LockingLogger 类继承自 LoggerBase
 public:
  void addStatValue(const std::string& stat_name, int64_t val) override;  // 重载添加统计值的函数声明
  virtual int64_t getCounterValue(const std::string& name) const;  // 获取计数器值的虚函数声明
  void setAggregationType(const std::string& stat_name, AggregationType type);  // 设置聚合类型的函数声明
  ~LockingLogger() override = default;  // 默认析构函数

 private:
  mutable std::mutex m;  // 可变的互斥锁对象
  struct RawCounter {  // 内部结构体 RawCounter
    RawCounter() : sum(0), count(0) {}  // 构造函数初始化 sum 和 count
    int64_t sum;  // 总和
    size_t count;  // 计数
  };
  std::unordered_map<std::string, RawCounter> raw_counters;  // 使用字符串作为键的无序映射，存储 RawCounter 对象
  std::unordered_map<std::string, AggregationType> agg_types;  // 使用字符串作为键的无序映射，存储聚合类型对象
};

// Make this struct so the timer internals are opaque to the user.
// 使此结构体对用户不透明，以隐藏计时器的内部细节。
struct JITTimePoint {  // JITTimePoint 结构体定义
  std::chrono::time_point<std::chrono::high_resolution_clock> point;  // 高精度时钟的时间点
};

TORCH_API JITTimePoint timePoint();  // 获取时间点的函数声明
TORCH_API void recordDurationSince(
    const std::string& name,
    const JITTimePoint& tp);  // 记录从指定时间点到当前时间经过的持续时间的函数声明

namespace runtime_counters {  // 进入 runtime_counters 命名空间
constexpr const char* GRAPH_EXECUTORS_CONSTRUCTED =  // 图执行器构造的常量字符串
    "pytorch_runtime.graph_executors_constructed";
constexpr const char* GRAPH_EXECUTOR_INVOCATIONS =  // 图执行器调用的常量字符串
    "pytorch_runtime.graph_executor_invocations";
constexpr const char* EXECUTION_PLAN_CACHE_HIT =  // 执行计划缓存命中的常量字符串
    "pytorch_runtime.execution_plan_cache_hit";
constexpr const char* EXECUTION_PLAN_CACHE_MISS =  // 执行计划缓存未命中的常量字符串
    "pytorch_runtime.execution_plan_cache_miss";

inline std::vector<const char*> allRuntimeCounters() {  // 返回所有运行时计数器的函数定义
  return {  // 返回字符串指针向量
      GRAPH_EXECUTORS_CONSTRUCTED,
      GRAPH_EXECUTOR_INVOCATIONS,
      EXECUTION_PLAN_CACHE_HIT,
      EXECUTION_PLAN_CACHE_MISS};
}

} // namespace runtime_counters  // 退出 runtime_counters 命名空间

} // namespace torch::jit::logging  // 退出 torch::jit::logging 命名空间
```