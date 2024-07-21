# `.\pytorch\test\cpp\monitor\test_counters.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <thread>  // 引入线程相关的头文件

#include <torch/csrc/monitor/counters.h>  // 引入计数器相关的头文件
#include <torch/csrc/monitor/events.h>    // 引入事件相关的头文件

using namespace torch::monitor;  // 使用 torch::monitor 命名空间

TEST(MonitorTest, CounterDouble) {  // 定义测试用例 MonitorTest.CounterDouble
  Stat<double> a{  // 声明一个名为 a 的统计量，类型为 double
      "a",  // 统计量的名称为 "a"
      {Aggregation::MEAN, Aggregation::COUNT},  // 统计的聚合方式为 MEAN 和 COUNT
      std::chrono::milliseconds(100000),  // 统计的时间窗口为 100000 毫秒
      2,  // 统计的样本数量初始值为 2
  };
  a.add(5.0);  // 向统计量 a 中添加数据 5.0
  ASSERT_EQ(a.count(), 1);  // 断言统计量 a 的样本数量为 1
  a.add(6.0);  // 向统计量 a 中再次添加数据 6.0
  ASSERT_EQ(a.count(), 0);  // 断言统计量 a 的样本数量为 0

  auto stats = a.get();  // 获取统计量 a 的统计结果
  std::unordered_map<Aggregation, double, AggregationHash> want = {  // 定义期望的统计结果
      {Aggregation::MEAN, 5.5},  // 平均值为 5.5
      {Aggregation::COUNT, 2.0},  // 样本数量为 2.0
  };
  ASSERT_EQ(stats, want);  // 断言实际的统计结果与期望相同
}

TEST(MonitorTest, CounterInt64Sum) {  // 定义测试用例 MonitorTest.CounterInt64Sum
  Stat<int64_t> a{  // 声明一个名为 a 的统计量，类型为 int64_t
      "a",  // 统计量的名称为 "a"
      {Aggregation::SUM},  // 统计的聚合方式为 SUM
      std::chrono::milliseconds(100000),  // 统计的时间窗口为 100000 毫秒
      2,  // 统计的样本数量初始值为 2
  };
  a.add(5);  // 向统计量 a 中添加数据 5
  a.add(6);  // 向统计量 a 中再次添加数据 6
  auto stats = a.get();  // 获取统计量 a 的统计结果
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {  // 定义期望的统计结果
      {Aggregation::SUM, 11},  // 总和为 11
  };
  ASSERT_EQ(stats, want);  // 断言实际的统计结果与期望相同
}

TEST(MonitorTest, CounterInt64Value) {  // 定义测试用例 MonitorTest.CounterInt64Value
  Stat<int64_t> a{  // 声明一个名为 a 的统计量，类型为 int64_t
      "a",  // 统计量的名称为 "a"
      {Aggregation::VALUE},  // 统计的聚合方式为 VALUE
      std::chrono::milliseconds(100000),  // 统计的时间窗口为 100000 毫秒
      2,  // 统计的样本数量初始值为 2
  };
  a.add(5);  // 向统计量 a 中添加数据 5
  a.add(6);  // 向统计量 a 中再次添加数据 6
  auto stats = a.get();  // 获取统计量 a 的统计结果
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {  // 定义期望的统计结果
      {Aggregation::VALUE, 6},  // 最后一个值为 6
  };
  ASSERT_EQ(stats, want);  // 断言实际的统计结果与期望相同
}

TEST(MonitorTest, CounterInt64Mean) {  // 定义测试用例 MonitorTest.CounterInt64Mean
  Stat<int64_t> a{  // 声明一个名为 a 的统计量，类型为 int64_t
      "a",  // 统计量的名称为 "a"
      {Aggregation::MEAN},  // 统计的聚合方式为 MEAN
      std::chrono::milliseconds(100000),  // 统计的时间窗口为 100000 毫秒
      2,  // 统计的样本数量初始值为 2
  };
  {
    // zero samples case
    auto stats = a.get();  // 获取统计量 a 的统计结果
    std::unordered_map<Aggregation, int64_t, AggregationHash> want = {  // 定义期望的统计结果
        {Aggregation::MEAN, 0},  // 均值为 0
    };
    ASSERT_EQ(stats, want);  // 断言实际的统计结果与期望相同
  }

  a.add(0);  // 向统计量 a 中添加数据 0
  a.add(10);  // 向统计量 a 中再次添加数据 10

  {
    auto stats = a.get();  // 获取统计量 a 的统计结果
    std::unordered_map<Aggregation, int64_t, AggregationHash> want = {  // 定义期望的统计结果
        {Aggregation::MEAN, 5},  // 均值为 5
    };
    ASSERT_EQ(stats, want);  // 断言实际的统计结果与期望相同
  }
}

TEST(MonitorTest, CounterInt64Count) {  // 定义测试用例 MonitorTest.CounterInt64Count
  Stat<int64_t> a{  // 声明一个名为 a 的统计量，类型为 int64_t
      "a",  // 统计量的名称为 "a"
      {Aggregation::COUNT},  // 统计的聚合方式为 COUNT
      std::chrono::milliseconds(100000),  // 统计的时间窗口为 100000 毫秒
      2,  // 统计的样本数量初始值为 2
  };
  ASSERT_EQ(a.count(), 0);  // 断言统计量 a 的样本数量为 0
  a.add(0);  // 向统计量 a 中添加数据 0
  ASSERT_EQ(a.count(), 1);  // 断言统计量 a 的样本数量为 1
  a.add(10);  // 向统计量 a 中再次添加数据 10
  ASSERT_EQ(a.count(), 0);  // 断言统计量 a 的样本数量为 0

  auto stats = a.get();  // 获取统计量 a 的统计结果
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {  // 定义期望的统计结果
      {Aggregation::COUNT, 2},  // 样本数量为 2
  };
  ASSERT_EQ(stats, want);  // 断言实际的统计结果与期望相同
}

TEST(MonitorTest, CounterInt64MinMax) {  // 定义测试用例 MonitorTest.CounterInt64MinMax
  Stat<int64_t> a{  // 声明一个名为 a 的统计量，类型为 int64_t
      "a",  // 统计量的名称为 "a"
      {Aggregation::MIN, Aggregation::MAX},  // 统计的聚合方式为 MIN 和 MAX
      std::chrono::milliseconds(100000),  // 统计的时间窗口为 100000 毫秒
      6,  // 统计的样本数量初始值为 6
  };
  {
    auto stats = a.get();  // 获取统计量 a 的统计结果
    std::unordered_map<Aggregation, int64_t, AggregationHash> want = {  // 定义期望的统计结果
        {Aggregation::MAX, 0},  // 最大值为 0
        {Aggregation::MIN, 0},
TEST(MonitorTest, CounterInt64WindowSize) {
  // 创建一个 int64_t 类型的统计对象 a，名称为 "a"
  // 支持 COUNT 和 SUM 聚合，窗口大小为 100 秒
  Stat<int64_t> a{
      "a",
      {Aggregation::COUNT, Aggregation::SUM},
      std::chrono::milliseconds(100000),
      /*windowSize=*/3,
  };
  // 向统计对象 a 添加值 1
  a.add(1);
  // 向统计对象 a 添加值 2
  a.add(2);
  // 断言统计对象 a 的计数为 2
  ASSERT_EQ(a.count(), 2);
  // 向统计对象 a 添加值 3
  a.add(3);
  // 断言统计对象 a 的计数为 0
  ASSERT_EQ(a.count(), 0);

  // 在窗口中记录最大值后，计数应为 0
  // 向统计对象 a 添加值 4
  a.add(4);
  // 断言统计对象 a 的计数为 0
  ASSERT_EQ(a.count(), 0);

  // 获取统计对象 a 的当前统计数据
  auto stats = a.get();
  // 期望的统计结果，包含 COUNT 和 SUM 两个聚合的值
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
      {Aggregation::COUNT, 3},
      {Aggregation::SUM, 6},
  };
  // 断言实际统计结果与期望结果相同
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64WindowSizeHuge) {
  // 创建一个 int64_t 类型的统计对象 a，名称为 "a"
  // 支持 COUNT 和 SUM 聚合，窗口大小为 10 年
  Stat<int64_t> a{
      "a",
      {Aggregation::COUNT, Aggregation::SUM},
      std::chrono::hours(24 * 365 * 10), // 10 years
      /*windowSize=*/3,
  };
  // 向统计对象 a 添加值 1
  a.add(1);
  // 向统计对象 a 添加值 2
  a.add(2);
  // 断言统计对象 a 的计数为 2
  ASSERT_EQ(a.count(), 2);
  // 向统计对象 a 添加值 3
  a.add(3);
  // 断言统计对象 a 的计数为 0
  ASSERT_EQ(a.count(), 0);

  // 在窗口中记录最大值后，计数应为 0
  // 向统计对象 a 添加值 4
  a.add(4);
  // 断言统计对象 a 的计数为 0
  ASSERT_EQ(a.count(), 0);

  // 获取统计对象 a 的当前统计数据
  auto stats = a.get();
  // 期望的统计结果，包含 COUNT 和 SUM 两个聚合的值
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
      {Aggregation::COUNT, 3},
      {Aggregation::SUM, 6},
  };
  // 断言实际统计结果与期望结果相同
  ASSERT_EQ(stats, want);
}

template <typename T>
struct TestStat : public Stat<T> {
  uint64_t mockWindowId{1};

  TestStat(
      std::string name,
      std::initializer_list<Aggregation> aggregations,
      std::chrono::milliseconds windowSize,
      int64_t maxSamples = std::numeric_limits<int64_t>::max())
      : Stat<T>(name, aggregations, windowSize, maxSamples) {}

  uint64_t currentWindowId() const override {
    return mockWindowId;
  }
};

struct AggregatingEventHandler : public EventHandler {
  std::vector<Event> events;

  void handle(const Event& e) override {
    events.emplace_back(e);
  }
};

template <typename T>
struct HandlerGuard {
  std::shared_ptr<T> handler;

  HandlerGuard() : handler(std::make_shared<T>()) {
    registerEventHandler(handler);
  }

  ~HandlerGuard() {
    unregisterEventHandler(handler);
  }
};

TEST(MonitorTest, Stat) {
  // 创建一个事件处理器的保护对象 guard
  HandlerGuard<AggregatingEventHandler> guard;

  // 创建一个 int64_t 类型的统计对象 a，名称为 "a"
  // 支持 COUNT 和 SUM 聚合，窗口大小为 1 毫秒
  Stat<int64_t> a{
      "a",
      {Aggregation::COUNT, Aggregation::SUM},
      std::chrono::milliseconds(1),
  };
  // 断言事件处理器的事件列表大小为 0
  ASSERT_EQ(guard.handler->events.size(), 0);

  // 向统计对象 a 添加值 1
  a.add(1);
  // 断言统计对象 a 的计数不超过 1
  ASSERT_LE(a.count(), 1);

  // 等待 2 毫秒
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  // 向统计对象 a 添加值 2
  a.add(2);
  // 断言统计对象 a 的计数不超过 1
  ASSERT_LE(a.count(), 1);

  // 断言事件处理器的事件列表大小至少为 1，但不超过 2
  ASSERT_GE(guard.handler->events.size(), 1);
  ASSERT_LE(guard.handler->events.size(), 2);
}
TEST(MonitorTest, StatEvent) {
  // 在测试开始前创建一个 AggregatingEventHandler 的 HandlerGuard 对象
  HandlerGuard<AggregatingEventHandler> guard;

  // 创建一个名为 "a" 的 TestStat 对象，指定统计方式和时间窗口
  TestStat<int64_t> a{
      "a",
      {Aggregation::COUNT, Aggregation::SUM},
      std::chrono::milliseconds(1),
  };
  // 断言初始事件列表为空
  ASSERT_EQ(guard.handler->events.size(), 0);

  // 添加数据点并断言计数是否正确
  a.add(1);
  ASSERT_EQ(a.count(), 1);
  a.add(2);
  ASSERT_EQ(a.count(), 2);
  // 断言事件列表仍为空
  ASSERT_EQ(guard.handler->events.size(), 0);

  // 修改 mockWindowId 属性
  a.mockWindowId = 100;

  // 添加数据点并断言计数是否正确
  a.add(3);
  ASSERT_LE(a.count(), 1);

  // 断言事件列表中有一个事件
  ASSERT_EQ(guard.handler->events.size(), 1);
  // 获取第一个事件并断言其属性
  Event e = guard.handler->events.at(0);
  ASSERT_EQ(e.name, "torch.monitor.Stat");
  ASSERT_NE(e.timestamp, std::chrono::system_clock::time_point{});
  // 断言事件数据与预期数据一致
  std::unordered_map<std::string, data_value_t> data{
      {"a.sum", 3L},
      {"a.count", 2L},
  };
  ASSERT_EQ(e.data, data);
}

TEST(MonitorTest, StatEventDestruction) {
  // 在测试开始前创建一个 AggregatingEventHandler 的 HandlerGuard 对象
  HandlerGuard<AggregatingEventHandler> guard;

  {
    // 在局部作用域内创建一个名为 "a" 的 TestStat 对象，指定统计方式和时间窗口
    TestStat<int64_t> a{
        "a",
        {Aggregation::COUNT, Aggregation::SUM},
        std::chrono::hours(10),
    };
    // 添加数据点并断言计数是否正确
    a.add(1);
    ASSERT_EQ(a.count(), 1);
    // 断言事件列表仍为空
    ASSERT_EQ(guard.handler->events.size(), 0);
  }
  // 断言事件列表中有一个事件
  ASSERT_EQ(guard.handler->events.size(), 1);

  // 获取第一个事件并断言其属性
  Event e = guard.handler->events.at(0);
  ASSERT_EQ(e.name, "torch.monitor.Stat");
  ASSERT_NE(e.timestamp, std::chrono::system_clock::time_point{});
  // 断言事件数据与预期数据一致
  std::unordered_map<std::string, data_value_t> data{
      {"a.sum", 1L},
      {"a.count", 1L},
  };
  ASSERT_EQ(e.data, data);
}
```