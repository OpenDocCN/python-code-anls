# `.\pytorch\caffe2\utils\threadpool\ThreadPool.h`

```py
#ifndef CAFFE2_UTILS_THREADPOOL_H_
#define CAFFE2_UTILS_THREADPOOL_H_

#include "ThreadPoolCommon.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "caffe2/core/common.h"

//
// A work-stealing threadpool loosely based off of pthreadpool
//

namespace caffe2 {

// 声明一个结构体 Task 和一个类 WorkersPool
struct Task;
class WorkersPool;

// 缓存行大小，用于优化缓存性能
constexpr size_t kCacheLineSize = 64;

// 线程池类，基于给定数量的线程进行工作分配
// 注意：kCacheLineSize 对齐仅用于缓存性能，并不严格执行（例如在堆上创建对象时）。
// 因此，为避免未对齐的内在函数，线程池实现不涉及 SSE 指令。
// 注意：alignas 被禁用，因为某些编译器不能同时处理 TORCH_API 和 alignas 标注。
class TORCH_API /*alignas(kCacheLineSize)*/ ThreadPool {
 public:
  // 创建具有指定线程数的线程池的静态方法
  static ThreadPool* createThreadPool(int numThreads);
  // 创建默认线程池的静态方法
  static std::unique_ptr<ThreadPool> defaultThreadPool();
  virtual ~ThreadPool() = default;

  // 返回当前正在使用的线程数
  virtual int getNumThreads() const = 0;
  // 设置线程数
  virtual void setNumThreads(size_t numThreads) = 0;

  // 设置最小工作大小，小于此大小的工作将在主线程上运行
  void setMinWorkSize(size_t size) {
    // 加锁，保护执行互斥区域
    std::lock_guard<std::mutex> guard(executionMutex_);
    minWorkSize_ = size;
  }

  // 获取最小工作大小
  size_t getMinWorkSize() const {
    return minWorkSize_;
  }

  // 在线程池中执行指定范围的任务
  virtual void run(const std::function<void(int, size_t)>& fn, size_t range) = 0;

  // 在访问 Workers Pool 时以线程安全的方式运行任意函数
  virtual void withPool(const std::function<void(WorkersPool*)>& fn) = 0;

 protected:
  // 默认线程数
  static size_t defaultNumThreads_;
  // 执行互斥锁
  mutable std::mutex executionMutex_;
  // 最小工作大小
  size_t minWorkSize_;
};

// 获取默认线程数的静态方法
size_t getDefaultNumThreads();
} // namespace caffe2

#endif // CAFFE2_UTILS_THREADPOOL_H_
```