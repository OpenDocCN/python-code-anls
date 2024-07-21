# `.\pytorch\torch\csrc\api\include\torch\data\detail\queue.h`

```py
#pragma once

#include <torch/types.h>

#include <c10/util/Exception.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace torch {
namespace data {
namespace detail {

/// A basic locked, blocking MPMC queue.
///
/// Every `push` and `pop` is guarded by a mutex. A condition variable is used
/// to communicate insertion of new elements, such that waiting threads will be
/// woken up if they are currently waiting inside a call to `pop()`.
///
/// Note that this data structure is written specifically for use with the
/// `DataLoader`. Its behavior is tailored to this use case and may not be
/// applicable to more general uses.
template <typename T>
class Queue {
 public:
  /// Pushes a new value to the back of the `Queue` and notifies one thread on
  /// the waiting side about this event.
  void push(T value) {
    {
      // 使用互斥锁保护临界区，将值推入队列的尾部
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(std::move(value));
    }
    // 通知一个正在等待的线程，有新的元素被推入队列
    cv_.notify_one();
  }

  /// Blocks until at least one element is ready to be popped from the front of
  /// the queue. An optional `timeout` in seconds can be used to limit the time
  /// spent waiting for an element. If the wait times out, an exception is
  /// raised.
  T pop(optional<std::chrono::milliseconds> timeout = nullopt) {
    // 获取独占锁
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout) {
      // 如果设置了超时时间，等待直到队列不为空或超时
      if (!cv_.wait_for(
              lock, *timeout, [this] { return !this->queue_.empty(); })) {
        // 超时时抛出异常
        // clang-format off
        AT_ERROR(
            "Timeout in DataLoader queue while waiting for next batch"
            " (timeout was ", timeout->count(), " ms)");
        // clang-format on
      }
    } else {
      // 等待直到队列不为空
      cv_.wait(lock, [this] { return !this->queue_.empty(); });
    }
    // 确保队列不为空
    AT_ASSERT(!queue_.empty());
    // 取出队列的首个元素
    T value = queue_.front();
    // 弹出队列的首个元素
    queue_.pop();
    lock.unlock();
    return value;
  }

  /// Empties the queue and returns the number of elements that were present at
  /// the start of the function. No threads are notified about this event as it
  /// is assumed to be used to drain the queue during shutdown of a
  /// `DataLoader`.
  size_t clear() {
    // 获取互斥锁
    std::lock_guard<std::mutex> lock(this->mutex_);
    // 获取当前队列的大小
    const auto size = queue_.size();
    // 清空队列
    while (!queue_.empty()) {
      queue_.pop();
    }
    return size;
  }

 private:
  std::queue<T> queue_;               // 存储元素的队列
  std::mutex mutex_;                  // 保护队列操作的互斥锁
  std::condition_variable cv_;        // 用于线程间同步的条件变量
};
} // namespace detail
} // namespace data
} // namespace torch
```