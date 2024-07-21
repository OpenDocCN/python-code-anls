# `.\pytorch\caffe2\utils\threadpool\pthreadpool-cpp.h`

```
#pragma once

#pragma once 表示这是一个预处理器编译指令，用于确保头文件只被编译一次。


#ifdef USE_PTHREADPOOL

#ifdef USE_PTHREADPOOL 检查是否定义了 USE_PTHREADPOOL 宏，用于条件编译。


#ifdef USE_INTERNAL_PTHREADPOOL_IMPL
#include <caffe2/utils/threadpool/pthreadpool.h>
#else
#include <pthreadpool.h>
#endif

#ifdef USE_INTERNAL_PTHREADPOOL_IMPL 检查是否定义了 USE_INTERNAL_PTHREADPOOL_IMPL 宏，根据不同的情况包含不同的头文件，实现条件编译。


#include <functional>
#include <memory>
#include <mutex>

包含标准库头文件 <functional>、<memory> 和 <mutex>，提供函数对象、智能指针和互斥量的支持。


namespace caffe2 {

定义命名空间 caffe2，用于封装类和函数，避免全局命名冲突。


class PThreadPool final {
 public:
  explicit PThreadPool(size_t thread_count);
  ~PThreadPool() = default;

  PThreadPool(const PThreadPool&) = delete;
  PThreadPool& operator=(const PThreadPool&) = delete;

  PThreadPool(PThreadPool&&) = delete;
  PThreadPool& operator=(PThreadPool&&) = delete;

  size_t get_thread_count() const;
  void set_thread_count(size_t thread_count);

  // Run, in parallel, function fn(task_id) over task_id in range [0, range).
  // This function is blocking.  All input is processed by the time it returns.
  void run(const std::function<void(size_t)>& fn, size_t range);

 private:
  friend pthreadpool_t pthreadpool_();

 private:
  mutable std::mutex mutex_;
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_;
};

定义了类 PThreadPool，实现了线程池的基本功能，包括构造函数、析构函数、线程数获取和设置函数、并行运行函数 run，以及私有成员变量 mutex_ 和 threadpool_。


PThreadPool* pthreadpool();

声明函数 pthreadpool()，返回 PThreadPool 类的单例实例，用于 ATen/TH 多线程。


pthreadpool_t pthreadpool_();

声明函数 pthreadpool_()，返回 PThreadPool 的底层实现，用于在内部和外部库之间统一线程处理。


} // namespace caffe2

命名空间 caffe2 的结束标记，确保所有定义都在这个命名空间中。


#endif /* USE_PTHREADPOOL */

结束条件编译块，检查 USE_PTHREADPOOL 宏是否被定义，如果没有定义则结束。
```