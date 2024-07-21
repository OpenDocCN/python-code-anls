# `.\pytorch\torch\csrc\monitor\instrumentation.cpp`

```
#ifndef FBCODE_CAFFE2
#include <torch/csrc/monitor/instrumentation.h>

#include <chrono>
#include <string_view>

namespace torch {
namespace monitor {

namespace detail {
// 定义一个私有类 WaitCounterImpl，用于内部实现细节
class WaitCounterImpl {};
// 定义一个静态函数 getImpl，返回一个 WaitCounterImpl 实例的引用，根据给定的字符串视图 key
static detail::WaitCounterImpl& getImpl(std::string_view key) {
  // 分配并返回一个新的 WaitCounterImpl 实例
  auto* impl = new detail::WaitCounterImpl();
  return *impl;
}
} // namespace detail

// WaitCounterHandle 构造函数，接受一个字符串视图 key 作为参数
WaitCounterHandle::WaitCounterHandle(std::string_view key)
    : impl_(detail::getImpl(key)) {
  // 实现待添加
}

// 开始计时的函数，接受一个时间点 now 作为参数
void WaitCounterHandle::start(std::chrono::steady_clock::time_point now) {
  // 实现待添加
}

// 停止计时的函数，接受一个时间点 now 作为参数
void WaitCounterHandle::stop(std::chrono::steady_clock::time_point now) {
  // 实现待添加
}

} // namespace monitor
} // namespace torch
#endif
```