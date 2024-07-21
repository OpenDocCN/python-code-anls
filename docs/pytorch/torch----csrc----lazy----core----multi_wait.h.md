# `.\pytorch\torch\csrc\lazy\core\multi_wait.h`

```py
/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/master/third_party/xla_client/multi_wait.h
 */

// 包含必要的头文件
#pragma once

#include <condition_variable>  // 条件变量，用于线程同步
#include <exception>           // 异常处理相关
#include <functional>          // 函数对象相关
#include <memory>              // 内存管理相关
#include <mutex>               // 互斥锁相关

#include <c10/macros/Export.h> // 导出宏定义

namespace torch {
namespace lazy {

// 支持等待多个任务完成的类
class TORCH_API MultiWait {
 public:
  // 构造函数，初始化待完成任务的数量
  explicit MultiWait(size_t count) : count_(count) {}

  // 标记单个任务完成
  void Done();

  // 等待至少 count 个任务完成
  void Wait();

  // 等待至少 count 个任务完成，最多等待 wait_seconds 秒
  void Wait(double wait_seconds);

  // 重置 MultiWait 对象的计数器，并将已完成的计数器清零
  void Reset(size_t count);

  // 创建一个完成者函数对象，用于任务完成时通知 MultiWait 对象
  // 处理异常情况，使用适当的状态值通知 MultiWait
  // 返回一个函数对象，该对象捕获了 MultiWait 的引用，因此返回的函数对象
  // 的生命周期必须确保 MultiWait 引用在其整个生命周期内有效
  std::function<void()> Completer(std::function<void()> func);

  // 类似上述 API，但显式捕获 MultiWait 的共享指针
  static std::function<void()> Completer(
      std::shared_ptr<MultiWait> mwait,
      std::function<void()> func);

 private:
  // 完成任务的内部方法，调用完成者函数
  void Complete(const std::function<void()>& func);

  // 互斥锁，用于保护状态的修改
  std::mutex mutex_;

  // 条件变量，用于线程等待和通知
  std::condition_variable cv_;

  // 需要等待完成的任务数量
  size_t count_ = 0;

  // 已完成的任务数量
  size_t completed_count_ = 0;

  // 异常指针，用于处理任务执行过程中的异常
  std::exception_ptr exptr_;
};

} // namespace lazy
} // namespace torch
```