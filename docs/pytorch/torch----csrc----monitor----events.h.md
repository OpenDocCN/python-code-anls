# `.\pytorch\torch\csrc\monitor\events.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <chrono>
// 包含时间相关的标准库头文件

#include <memory>
// 包含智能指针相关的标准库头文件

#include <string>
// 包含字符串处理相关的标准库头文件

#include <unordered_map>
// 包含无序映射相关的标准库头文件

#include <c10/macros/Macros.h>
// 包含特定于C10的宏定义头文件

#include <variant>
// 包含变体类型相关的标准库头文件

namespace torch {
namespace monitor {
// 声明命名空间 torch::monitor

// data_value_t 是事件数据值的类型，支持字符串、双精度浮点数、64位整数和布尔值
using data_value_t = std::variant<std::string, double, int64_t, bool>;

// Event 表示单个可以记录到外部追踪器的事件
// 在记录日志时会获取日志锁，因此应尽量少用以避免性能问题
struct TORCH_API Event {
  // name 是事件的名称，是一个静态字符串，用于程序化访问时区分事件类型
  // 类型应该采用完全限定的 Python 风格类名格式
  // 例如: torch.monitor.MonitorEvent
  std::string name;

  // timestamp 是相对于 Unix 纪元时间的时间戳
  std::chrono::system_clock::time_point timestamp;

  // data 包含关于事件的丰富信息。内容是事件特定的，因此在访问数据之前应检查类型以确保符合预期
  //
  // 注意: 这些事件没有版本控制，消费者需要检查字段以确保向后兼容性
  std::unordered_map<std::string, data_value_t> data;
};

// 重载运算符==，用于比较两个 Event 结构是否相等
TORCH_API inline bool operator==(const Event& lhs, const Event& rhs) {
  return lhs.name == rhs.name && lhs.timestamp == rhs.timestamp &&
      lhs.data == rhs.data;
}

// EventHandler 是一个抽象的事件处理程序，可以注册来捕获事件。每次记录事件时，将调用所有处理程序处理事件内容
//
// 注意: 处理程序应避免任何IO、阻塞调用或重计算，因为这可能会阻塞主线程并导致性能问题
class TORCH_API EventHandler {
 public:
  virtual ~EventHandler() = default;

  // handle 需要实现来处理事件。可能从多个线程调用，因此需要线程安全
  virtual void handle(const Event& e) = 0;
};

// logEvent 调用每个已注册的事件处理程序处理事件。此方法可以从多个线程并发调用
TORCH_API void logEvent(const Event& e);

// registerEventHandler 注册一个 EventHandler，以便接收任何记录的事件。通常在程序设置期间注册 EventHandler，并在结束时取消注册
TORCH_API void registerEventHandler(std::shared_ptr<EventHandler> p);

// unregisterEventHandler 取消注册由 shared_ptr 指向的事件处理程序
TORCH_API void unregisterEventHandler(const std::shared_ptr<EventHandler>& p);

} // namespace monitor
} // namespace torch
```