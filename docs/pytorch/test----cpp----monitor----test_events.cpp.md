# `.\pytorch\test\cpp\monitor\test_events.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <torch/csrc/monitor/events.h>  // 包含 Torch 监控模块中事件相关的头文件

using namespace torch::monitor;  // 使用 Torch 监控模块的命名空间

struct AggregatingEventHandler : public EventHandler {
  std::vector<Event> events;  // 存储事件的向量

  void handle(const Event& e) override {
    events.emplace_back(e);  // 处理事件的方法，在事件向量中添加新事件
  }
};

TEST(EventsTest, EventHandler) {
  Event e;  // 创建一个事件对象 e

  e.name = "test";  // 设置事件名称为 "test"
  e.timestamp = std::chrono::system_clock::now();  // 获取当前时间戳作为事件时间
  e.data["string"] = "asdf";  // 设置事件的字符串类型数据
  e.data["double"] = 1234.5678;  // 设置事件的双精度浮点数数据
  e.data["int"] = 1234L;  // 设置事件的长整型数据
  e.data["bool"] = true;  // 设置事件的布尔类型数据

  // 记录事件，但不做任何操作
  logEvent(e);

  auto handler = std::make_shared<AggregatingEventHandler>();  // 创建一个聚合事件处理器
  registerEventHandler(handler);  // 注册事件处理器，开始记录事件

  logEvent(e);  // 再次记录事件 e
  ASSERT_EQ(handler->events.size(), 1);  // 断言事件处理器中记录的事件数量为 1
  ASSERT_EQ(e, handler->events.at(0));  // 断言处理器中记录的第一个事件与 e 相同

  unregisterEventHandler(handler);  // 取消注册事件处理器，停止记录事件
  logEvent(e);  // 再次记录事件 e，但此时不会被处理器记录

  // 断言处理器记录的事件数量仍为 1，因为之前已取消注册
  ASSERT_EQ(handler->events.size(), 1);
}
```