# `.\pytorch\c10\util\CallOnce.h`

```
#pragma once
// 防止头文件被多次包含

#include <atomic>
// 引入原子操作支持

#include <mutex>
// 引入互斥锁支持

#include <utility>
// 引入 std::forward 和 std::move 等实用工具

#include <c10/macros/Macros.h>
// 引入 c10 宏定义

#include <c10/util/C++17.h>
// 引入 c10 的 C++17 兼容性工具

namespace c10 {

// c10 命名空间内的自定义 call_once 实现，用于避免 std::call_once 中的死锁问题。
// 这里的实现是从 folly 简化过来的，可能具有更高的内存占用。
template <typename Flag, typename F, typename... Args>
inline void call_once(Flag& flag, F&& f, Args&&... args) {
  // 如果标志已经被设置过，则直接返回
  if (C10_LIKELY(flag.test_once())) {
    return;
  }
  // 否则调用慢路径实现
  flag.call_once_slow(std::forward<F>(f), std::forward<Args>(args)...);
}

class once_flag {
 public:
#ifndef _WIN32
  // 在 MSVC 上遇到构建错误。似乎无法在本地复现，因此避免使用 constexpr
  //
  //   C:/actions-runner/_work/pytorch/pytorch\c10/util/CallOnce.h(26): error:
  //   defaulted default constructor cannot be constexpr because the
  //   corresponding implicitly declared default constructor would not be
  //   constexpr 1 error detected in the compilation of
  //   "C:/actions-runner/_work/pytorch/pytorch/aten/src/ATen/cuda/cub.cu".
  constexpr
#endif
      once_flag() noexcept = default;
  // 默认构造函数，不抛出异常

  once_flag(const once_flag&) = delete;
  // 删除拷贝构造函数

  once_flag& operator=(const once_flag&) = delete;
  // 删除拷贝赋值运算符

 private:
  // 以下是私有成员

  template <typename Flag, typename F, typename... Args>
  friend void call_once(Flag& flag, F&& f, Args&&... args);
  // 声明 call_once 为友元函数，以便访问私有成员

  template <typename F, typename... Args>
  void call_once_slow(F&& f, Args&&... args) {
    // 慢路径实现：使用互斥锁保护，确保初始化只执行一次
    std::lock_guard<std::mutex> guard(mutex_);
    if (init_.load(std::memory_order_relaxed)) {
      return;
    }
    // 调用传入的函数对象，并传递参数
    c10::guts::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    // 设置标志为已初始化
    init_.store(true, std::memory_order_release);
  }

  bool test_once() {
    // 测试标志是否已初始化
    return init_.load(std::memory_order_acquire);
  }

  void reset_once() {
    // 重置标志，设置为未初始化状态
    init_.store(false, std::memory_order_release);
  }

 private:
  // 私有成员变量

  std::mutex mutex_;
  // 互斥锁，用于保护初始化过程

  std::atomic<bool> init_{false};
  // 原子布尔变量，表示初始化状态，默认为未初始化
};

} // namespace c10
// 结束 c10 命名空间
```