# `.\pytorch\torch\csrc\monitor\counters.h`

```py
#pragma once
// 预处理指令：确保头文件只被包含一次，避免重复定义

#include <bitset>
// 包含位集（bitset）的头文件，用于处理位标志

#include <mutex>
// 包含互斥锁（mutex）的头文件，用于线程同步

#include <sstream>
// 包含字符串流（stringstream）的头文件，用于字符串操作

#include <unordered_map>
// 包含无序映射（unordered_map）的头文件，用于快速查找和插入键值对

#include <vector>
// 包含向量（vector）的头文件，用于动态数组操作

#include <c10/macros/Macros.h>
// 包含c10宏定义的头文件

#include <torch/csrc/monitor/events.h>
// 包含torch监视器事件相关的头文件

namespace torch {
namespace monitor {

constexpr int NUM_AGGREGATIONS = 7;
// 常量表达式：定义了统计聚合类型的数量

// Aggregation是Stats可能的聚合类型列表。
// 使用位标志以便能够高效地存储。
enum class C10_API_ENUM Aggregation {
  // NONE表示未设置任何聚合。
  NONE = 0,
  // VALUE导出最近设置的值。
  VALUE = 1,
  // MEAN计算窗口内设置值的平均值。如果没有值，则为零。
  MEAN = 2,
  // COUNT跟踪在窗口内设置值的次数。
  COUNT = 3,
  // SUM计算窗口内设置的值的总和。
  SUM = 4,
  // MAX计算窗口内设置值的最大值。如果没有值，则为零。
  MAX = 5,
  // MIN计算窗口内设置值的最小值。如果没有值，则为零。
  MIN = 6,
};

struct TORCH_API AggregationHash {
  // 聚合哈希结构，用于将聚合类型转换为哈希值
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

// aggregationName返回与聚合类型对应的可读名称。
TORCH_API const char* aggregationName(Aggregation agg);

template <typename T>
class Stat;

namespace {
template <typename T>
inline std::bitset<NUM_AGGREGATIONS> merge(T& list) {
  // 在匿名命名空间中定义的函数模板merge，将聚合类型列表转换为位集
  std::bitset<NUM_AGGREGATIONS> a;
  for (Aggregation b : list) {
    a.set(static_cast<int>(b));
  }
  return a;
}
} // namespace

namespace detail {
void TORCH_API registerStat(Stat<double>* stat);
void TORCH_API registerStat(Stat<int64_t>* stat);
void TORCH_API unregisterStat(Stat<double>* stat);
void TORCH_API unregisterStat(Stat<int64_t>* stat);
} // namespace detail

// Stat用于在固定时间间隔内以高效方式计算摘要统计信息。
// Stat将统计信息作为事件记录，每隔`windowSize`时长记录一次。
// 当窗口关闭时，通过事件处理程序将统计信息作为`torch.monitor.Stat`事件记录。
//
// `windowSize`应设置为相对较高的值，以避免记录大量事件。例如：60秒。Stat使用毫秒精度。
//
// 如果设置了maxSamples，则Stat将通过在窗口中发生`maxSamples`次添加调用后丢弃`add`调用，来限制每个窗口内的样本数量。
// 如果未设置maxSamples，则窗口期间的所有`add`调用都将包括在内。
// 这是一个可选字段，使得在样本数量可能变化时，聚合更加直接可比较。
//
// 根据需要记录double和int64_t数据类型的统计信息，因此需要用其中之一对Stat进行模板化。
//
// 当Stat被析构时，即使窗口尚未过去，它也会记录任何剩余数据。
template <typename T>
class Stat {
 private:
  struct Values {
    T value{0};
    T sum{0};
    T min{0};
    T max{0};
    int64_t count{0};
  };

 public:
  // Stat 类的构造函数，接受名称、聚合方式列表、窗口大小和最大样本数作为参数
  Stat(
      std::string name,
      std::initializer_list<Aggregation> aggregations,
      std::chrono::milliseconds windowSize,
      int64_t maxSamples = std::numeric_limits<int64_t>::max())
      : name_(std::move(name)),  // 初始化名称
        aggregations_(merge(aggregations)),  // 合并并初始化聚合方式
        windowSize_(windowSize),  // 初始化窗口大小
        maxSamples_(maxSamples) {  // 初始化最大样本数
    detail::registerStat(this);  // 在统计对象创建时注册到详情模块
  }

  // 另一个构造函数，接受名称、聚合方式向量、窗口大小和最大样本数作为参数
  Stat(
      std::string name,
      std::vector<Aggregation> aggregations,
      std::chrono::milliseconds windowSize,
      int64_t maxSamples = std::numeric_limits<int64_t>::max())
      : name_(std::move(name)),  // 初始化名称
        aggregations_(merge(aggregations)),  // 合并并初始化聚合方式
        windowSize_(windowSize),  // 初始化窗口大小
        maxSamples_(maxSamples) {  // 初始化最大样本数
    detail::registerStat(this);  // 在统计对象创建时注册到详情模块
  }

  // 析构函数，在对象销毁时调用
  virtual ~Stat() {
    {
      // 在销毁时，如果存在未记录的数据，则记录日志
      std::lock_guard<std::mutex> guard(mu_);
      logLocked();
    }
    detail::unregisterStat(this);  // 在详情模块中注销统计对象
  }

  // add 方法，向当前窗口添加值 v
  // 如果已经记录过日志，则直接返回
  void add(T v) {
    std::lock_guard<std::mutex> guard(mu_);
    maybeLogLocked();  // 可能会记录日志

    if (alreadyLogged()) {  // 如果已经记录过日志，则直接返回
      return;
    }

    // 根据聚合方式设置当前值、求和、最大值、最小值以及计数
    if (aggregations_.test(static_cast<int>(Aggregation::VALUE))) {
      current_.value = v;
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MEAN)) ||
        aggregations_.test(static_cast<int>(Aggregation::SUM))) {
      current_.sum += v;
    }

    if (aggregations_.test(static_cast<int>(Aggregation::MAX))) {
      if (current_.max < v || current_.count == 0
    // 如果前一个统计数据的计数为零，则直接返回，不执行后续操作
    if (prev_.count == 0) {
      return;
    }

    // 创建一个事件对象 e，并设置事件名称为 "torch.monitor.Stat"
    Event e;
    e.name = "torch.monitor.Stat";
    // 设置事件的时间戳为当前系统时间
    e.timestamp = std::chrono::system_clock::now();

    // 获取被锁定的统计数据，这里调用了私有成员函数 getLocked()
    auto stats = getLocked();
    // 预留空间以容纳统计数据的大小
    e.data.reserve(stats.size());
    // 遍历统计数据，将每个统计项的名称及其值存入事件的数据字典中
    for (auto& kv : stats) {
      // 构建统计项的键值对的键名，格式为 name_.aggregationName(kv.first)
      std::stringstream key;
      key << name_;  // name_ 是类的成员变量，表示统计对象的名称
      key << ".";
      key << aggregationName(kv.first);  // aggregationName 是一个函数，返回 Aggregation 类型的名称字符串
      e.data[key.str()] = kv.second;  // 将统计项的名称及其值存入事件的数据字典中
    }

    // 记录事件到日志
    logEvent(e);
  }

  // 获取被锁定的统计数据，返回一个无序映射表
  std::unordered_map<Aggregation, T, AggregationHash> getLocked()
      const noexcept {
    std::unordered_map<Aggregation, T, AggregationHash> out;
    // 预留空间以容纳所需数量的元素
    out.reserve(aggregations_.count());

    // 根据设置的聚合标志，将对应的统计数据添加到返回的映射表中
    if (aggregations_.test(static_cast<int>(Aggregation::VALUE))) {
      out.emplace(Aggregation::VALUE, prev_.value);
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MEAN))) {
      // 如果 prev_.count 为零，则将 MEAN 统计项设为 0；否则计算平均值并存入映射表
      if (prev_.count == 0) {
        out.emplace(Aggregation::MEAN, 0);
      } else {
        out.emplace(Aggregation::MEAN, prev_.sum / prev_.count);
      }
    }
    if (aggregations_.test(static_cast<int>(Aggregation::COUNT))) {
      out.emplace(Aggregation::COUNT, prev_.count);
    }
    if (aggregations_.test(static_cast<int>(Aggregation::SUM))) {
      out.emplace(Aggregation::SUM, prev_.sum);
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MAX))) {
      out.emplace(Aggregation::MAX, prev_.max);
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MIN))) {
      out.emplace(Aggregation::MIN, prev_.min);
    }

    return out;  // 返回填充好的统计数据映射表
  }

  // 成员变量定义
  const std::string name_;  // 统计对象的名称
  const std::bitset<NUM_AGGREGATIONS> aggregations_;  // 用于标识需要聚合的项目集合

  std::mutex mu_;  // 互斥锁，用于保护共享数据
  Values current_;  // 当前统计数据
  Values prev_;     // 前一次统计数据

  uint64_t windowId_{0};  // 窗口 ID，用于标识当前窗口
  uint64_t lastLoggedWindowId_{0};  // 上次记录的窗口 ID
  const std::chrono::milliseconds windowSize_;  // 窗口大小，用于统计时间窗口的长度
  const int64_t maxSamples_;  // 最大样本数，限制窗口内可容纳的最大样本数
};
// 结束 monitor 命名空间
} // namespace monitor
// 结束 torch 命名空间
} // namespace torch
```