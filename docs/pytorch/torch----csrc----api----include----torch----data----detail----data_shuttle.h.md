# `.\pytorch\torch\csrc\api\include\torch\data\detail\data_shuttle.h`

```
#pragma once

#include <torch/data/detail/queue.h>  // 包含队列实现的头文件
#include <torch/types.h>              // 包含 Torch 相关类型的头文件

#include <c10/util/Exception.h>  // 包含 C10 异常处理的头文件
#include <c10/util/Optional.h>   // 包含 C10 可选类型的头文件

#include <chrono>      // 包含时间库
#include <utility>     // 包含实用工具

namespace torch {
namespace data {
namespace detail {

/// Encapsulates the full life cycle of DataLoader jobs.
///
/// When a new job is enqueued to the `DataShuttle`, a counter for in-flight
/// jobs is bumped. This job is said to be "in-flight" until its result is
/// popped. Worker threads dequeue jobs as soon as they are available. When a
/// worker finishes a job, it enqueues the result. Only when the main thread
/// dequeues a result is the count of in-flight jobs decremented. When the main
/// thread attempts to dequeue a job but no jobs are in-flight, that means the
/// epoch is complete and `pop_result` returns an empty optional.
template <typename Job, typename Result>
class DataShuttle {
 public:
  /// Pushes a new job. Called by the main thread.
  void push_job(Job job) {
    new_jobs_.push(std::move(job));   // 将新作业推入未在执行队列
    ++in_flight_jobs_;                // 增加在执行作业的计数器
  }

  /// Pushes the result of a job. Called by worker threads.
  void push_result(Result result) {
    results_.push(std::move(result)); // 将作业的结果推入结果队列
  }

  /// Returns the next job, blocking until there is one available. Called by
  /// worker threads.
  Job pop_job() {
    return new_jobs_.pop();           // 弹出下一个作业，阻塞直到有作业可用
  }

  /// Returns the result of a job, or nullopt if all jobs were exhausted. Called
  /// by the main thread.
  optional<Result> pop_result(
      optional<std::chrono::milliseconds> timeout = nullopt) {
    if (in_flight_jobs_ > 0) {
      auto result = results_.pop(timeout);  // 弹出作业的结果，超时后返回空
      --in_flight_jobs_;                   // 减少在执行作业的计数器
      return result;                       // 返回作业的结果
    }
    return nullopt;                        // 如果没有在执行作业，返回空
  }

  /// Discards any jobs that are not yet in flight, and waits for all in-flight
  /// jobs to finish, discarding their result.
  void drain() {
    // 清除所有未执行的作业，以停止进一步的作业调度
    auto number_cleared = new_jobs_.clear();
    in_flight_jobs_ -= number_cleared;    // 减去已清除的未执行作业数量
    // 移除所有未完成的结果
    while (in_flight_jobs_ > 0) {
      pop_result();                       // 弹出所有未完成的作业结果
    }
  }

  /// Returns the number of jobs that are still in progress.
  /// When this number is zero, an epoch is finished.
  size_t in_flight_jobs() const noexcept {
    return in_flight_jobs_;               // 返回正在执行的作业数量
  }

 private:
  /// The queue for jobs that are not yet in flight.
  Queue<Job> new_jobs_;                   // 未执行作业的队列
  /// The number of in-flight jobs.
  /// NOTE: Not atomic because only manipulated by the main thread.
  size_t in_flight_jobs_ = 0;             // 正在执行的作业数量，非原子操作，因为仅主线程操作
  /// The queue for results of finished jobs.
  Queue<Result> results_;                 // 已完成作业结果的队列
};

} // namespace detail
} // namespace data
} // namespace torch
```