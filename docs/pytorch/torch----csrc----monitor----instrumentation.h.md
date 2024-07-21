# `.\pytorch\torch\csrc\monitor\instrumentation.h`

```py
#pragma once
// 声明该头文件只被编译一次

#include <chrono>
// 包含时间库chrono，用于处理时间相关操作
#include <memory>
// 包含内存管理库memory，用于智能指针等内存管理
#include <string>
// 包含字符串库string，用于字符串操作
#include <string_view>
// 包含字符串视图库string_view，用于非拥有式字符串操作

#include <c10/macros/Macros.h>
// 包含c10宏定义的头文件Macros.h
#include <c10/util/ScopeExit.h>
// 包含c10中提供的ScopeExit功能，用于在作用域结束时执行特定操作

namespace torch {
namespace monitor {
namespace detail {
class WaitCounterImpl;
}
// 声明torch::monitor::detail命名空间，定义WaitCounterImpl类

// A handle to a wait counter.
// 等待计数器的句柄类定义
class WaitCounterHandle {
 public:
  explicit WaitCounterHandle(std::string_view key);
  // 显式构造函数，接受string_view类型的参数key，用于初始化等待计数器句柄

  // Starts a waiter
  // 启动一个等待器
  void start(
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now());
  // 开始等待器的计时，可以指定开始时间，默认为当前时间点

  // Stops the waiter. Each start() call should be matched by exactly one stop()
  // call.
  // 停止等待器计时，每次start()调用应该精确匹配一次stop()调用
  void stop(
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now());
  // 停止等待器的计时，可以指定停止时间，默认为当前时间点

 private:
  detail::WaitCounterImpl& impl_;
  // 私有成员变量，引用detail命名空间中的WaitCounterImpl类的实例
};
} // namespace monitor
} // namespace torch

#define STATIC_WAIT_COUNTER(_key)                           \
  []() {                                                    \
    static torch::monitor::WaitCounterHandle handle(#_key); \
    return handle;                                          \
  }()
// 定义一个静态函数对象，用于创建并返回torch::monitor::WaitCounterHandle类型的静态对象，名称为_key

#define STATIC_SCOPED_WAIT_COUNTER(_name)    \
  STATIC_WAIT_COUNTER(_name).start();        \
  auto C10_ANONYMOUS_VARIABLE(SCOPE_GUARD) = \
      c10::make_scope_exit([&]() { STATIC_WAIT_COUNTER(_name).stop(); });
// 定义一个宏，创建一个作用域内的静态等待计数器，并在作用域结束时自动调用stop()方法
```