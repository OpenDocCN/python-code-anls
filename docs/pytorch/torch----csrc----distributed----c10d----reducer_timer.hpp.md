# `.\pytorch\torch\csrc\distributed\c10d\reducer_timer.hpp`

```py
#pragma once
// 包含头文件，用于时间近似计算
#include <c10/util/ApproximateClock.h>
// 包含头文件，用于自动微分分析器
#include <torch/csrc/autograd/profiler.h>

// 进入命名空间 c10d
namespace c10d {
// 定义一个常量，表示时间未设置
constexpr int kUnsetTime = -1;

// 定义一个内联函数，返回当前时间的纳秒数
inline int64_t current_time_in_nanos() {
  return c10::getTime();
}

// 定义一个计时器类
class TORCH_API Timer {
 private:
  // 此迭代中前向调用开始时间戳
  int64_t forward_start_time = kUnsetTime;
  // 此迭代中反向计算开始和结束时间戳
  int64_t backward_compute_start_time = kUnsetTime;
  int64_t backward_compute_end_time = kUnsetTime;
  // 此迭代中第一次通信调用开始时间戳
  int64_t backward_comm_start_time = kUnsetTime;
  // 此迭代中最后一次通信调用结束时间戳
  int64_t backward_comm_end_time = kUnsetTime;

 public:
  // 枚举类型，表示计时器中的事件
  enum class Event : uint8_t {
    kForwardStart,
    kBackwardComputeStart,
    kBackwardComputeEnd,
    kBackwardCommStart,
    kBackwardCommEnd,
  };

  // 记录当前事件的发生时间，默认为 CPU 实现
  virtual void record(Event event) {
    // 调用当前时间获取函数，将时间记录到相应的成员变量中
    getTimeRef(event) = current_time_in_nanos();
  }

  // 返回两个事件之间的时间差，单位为纳秒；如果其中一个事件未记录，则返回 nullopt
  virtual std::optional<int64_t> measureDifference(Event start, Event end) = 0;

  // 返回指定事件的时间戳，如果尚未记录，则返回 nullopt
  std::optional<int64_t> getTimestamp(Event event) {
    auto time = getTimeRef(event);
    if (time == kUnsetTime) {
      return c10::nullopt;
    } else {
      return time;
    }
  }

  // 返回与给定事件相关联的时间成员变量的引用
  int64_t& getTimeRef(Event event) {
    switch (event) {
      case Event::kForwardStart:
        return forward_start_time;
      case Event::kBackwardComputeStart:
        return backward_compute_start_time;
      case Event::kBackwardComputeEnd:
        return backward_compute_end_time;
      case Event::kBackwardCommStart:
        return backward_comm_start_time;
      case Event::kBackwardCommEnd:
        return backward_comm_end_time;
      default:
        // 如果传入的事件未知，则断言失败
        TORCH_INTERNAL_ASSERT(false);
    }
  }
};

// 声明一个类型化的注册表，用于存储计时器类的实例
TORCH_DECLARE_TYPED_REGISTRY(
    TimerRegistry,
    c10::DeviceType,
    Timer,
    std::unique_ptr,
    c10::Device);
} // namespace c10d
```