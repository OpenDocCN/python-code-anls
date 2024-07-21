# `.\pytorch\torch\csrc\lazy\core\multi_wait.cpp`

```
#include <torch/csrc/lazy/core/multi_wait.h>

#include <chrono>
#include <exception>
#include <stdexcept>

namespace torch {
namespace lazy {

// 当前类的实现

void MultiWait::Done() {
  bool notify = false;
  {
    // 使用互斥锁保护临界区
    std::lock_guard<std::mutex> lock(mutex_);
    // 完成的任务计数加一
    completed_count_ += 1;
    // 如果完成的任务数量等于总任务数量，则需要通知等待中的线程
    notify = completed_count_ == count_;
  }
  // 如果需要通知，则唤醒所有等待的线程
  if (notify) {
    cv_.notify_all();
  }
}

void MultiWait::Wait() {
  // 使用独占锁
  std::unique_lock<std::mutex> lock(mutex_);
  // 等待条件变量，直到完成的任务数量达到总任务数量
  cv_.wait(lock, [this] { return completed_count_ >= count_; });
  // 如果有异常被记录，则重新抛出异常
  if (exptr_ != nullptr) {
    std::rethrow_exception(exptr_);
  }
}

void MultiWait::Wait(double wait_seconds) {
  // 使用独占锁
  std::unique_lock<std::mutex> lock(mutex_);
  // 等待条件变量，允许指定的时间段
  if (!cv_.wait_for(lock, std::chrono::duration<double>(wait_seconds), [this] {
        return completed_count_ >= count_;
      })) {
    // 如果超时，则抛出运行时异常
    throw std::runtime_error("Timeout");
  }
  // 如果有异常被记录，则重新抛出异常
  if (exptr_ != nullptr) {
    std::rethrow_exception(exptr_);
  }
}

void MultiWait::Reset(size_t count) {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> lock(mutex_);
  // 重置总任务数量和完成任务数量
  count_ = count;
  completed_count_ = 0;
  // 清空异常指针
  exptr_ = nullptr;
}

std::function<void()> MultiWait::Completer(std::function<void()> func) {
  // 创建一个完成函数的包装器，该函数会调用传入的 func 并完成任务
  auto completer = [this, func = std::move(func)]() { Complete(func); };
  return completer;
}

std::function<void()> MultiWait::Completer(
    std::shared_ptr<MultiWait> mwait,
    std::function<void()> func) {
  // 创建一个完成函数的包装器，该函数会调用传入的 func 并在指定的 mwait 对象上完成任务
  auto completer = [mwait = std::move(mwait), func = std::move(func)]() {
    mwait->Complete(func);
  };
  return completer;
}

void MultiWait::Complete(const std::function<void()>& func) {
  // 尝试执行传入的函数，并在发生异常时记录异常
  try {
    func();
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex_);
    exptr_ = std::current_exception();
  }
  // 完成一个任务
  Done();
}

} // namespace lazy
} // namespace torch
```