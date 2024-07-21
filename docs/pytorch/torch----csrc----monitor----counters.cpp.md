# `.\pytorch\torch\csrc\monitor\counters.cpp`

```
// 引入 Torch 的计数器头文件
#include <torch/csrc/monitor/counters.h>

// 引入无序集合的头文件
#include <unordered_set>

// Torch 命名空间
namespace torch {
namespace monitor {

// 根据聚合类型返回对应的名称字符串
const char* aggregationName(Aggregation agg) {
  switch (agg) {
    case Aggregation::NONE:
      return "none";
    case Aggregation::VALUE:
      return "value";
    case Aggregation::MEAN:
      return "mean";
    case Aggregation::COUNT:
      return "count";
    case Aggregation::SUM:
      return "sum";
    case Aggregation::MAX:
      return "max";
    case Aggregation::MIN:
      return "min";
    default:
      throw std::runtime_error(
          "unknown aggregation: " + std::to_string(static_cast<int>(agg)));
  }
}

// 匿名命名空间，定义了用于统计的数据结构 Stats
namespace {
struct Stats {
  std::mutex mu;  // 互斥量，用于保护 Stats 中的数据

  // 用于存储 double 类型统计量的无序集合
  std::unordered_set<Stat<double>*> doubles;
  // 用于存储 int64_t 类型统计量的无序集合
  std::unordered_set<Stat<int64_t>*> int64s;
};

// 获取全局唯一的 Stats 对象实例
Stats& stats() {
  static Stats stats;  // 静态变量，保证唯一实例
  return stats;
}
} // namespace

// detail 命名空间，实现了注册和注销统计量的函数
namespace detail {
// 注册 double 类型的统计量
void registerStat(Stat<double>* stat) {
  std::lock_guard<std::mutex> guard(stats().mu);  // 加锁保护共享数据

  stats().doubles.insert(stat);  // 将统计量指针插入 doubles 集合
}

// 注册 int64_t 类型的统计量
void registerStat(Stat<int64_t>* stat) {
  std::lock_guard<std::mutex> guard(stats().mu);  // 加锁保护共享数据

  stats().int64s.insert(stat);  // 将统计量指针插入 int64s 集合
}

// 注销 double 类型的统计量
void unregisterStat(Stat<double>* stat) {
  std::lock_guard<std::mutex> guard(stats().mu);  // 加锁保护共享数据

  stats().doubles.erase(stat);  // 从 doubles 集合中移除统计量指针
}

// 注销 int64_t 类型的统计量
void unregisterStat(Stat<int64_t>* stat) {
  std::lock_guard<std::mutex> guard(stats().mu);  // 加锁保护共享数据

  stats().int64s.erase(stat);  // 从 int64s 集合中移除统计量指针
}
} // namespace detail

} // namespace monitor
} // namespace torch
```