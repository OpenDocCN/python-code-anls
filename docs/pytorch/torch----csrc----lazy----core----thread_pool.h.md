# `.\pytorch\torch\csrc\lazy\core\thread_pool.h`

```
/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/master/third_party/xla_client/metrics.h
 */

#pragma once

#include <functional>   // 包含函数对象的头文件
#include <memory>       // 包含智能指针的头文件
#include <thread>       // 包含线程操作的头文件

#include <c10/macros/Export.h>   // 导出宏定义

namespace torch {
namespace lazy {

class TORCH_API Completion {    // 定义 TORCH_API 的类 Completion
 public:
  class Data;                   // 前置声明内部类 Data

  explicit Completion(std::shared_ptr<Data> data);   // 构造函数，接受一个共享指针参数

  ~Completion();                // 析构函数

  void Wait();                  // 等待操作完成的方法

 private:
  std::shared_ptr<Data> data_;  // 私有成员变量，共享指针类型的 data_
};

// 调度一个可能等待 IO 或其他事件/条件的闭包函数
TORCH_API void ScheduleIoClosure(std::function<void()> closure);

// 调度一个带有完成对象的闭包函数，该对象可能等待 IO 或其他事件/条件
TORCH_API Completion ScheduleIoClosureWithCompletion(std::function<void()> closure);

} // namespace lazy
} // namespace torch
```