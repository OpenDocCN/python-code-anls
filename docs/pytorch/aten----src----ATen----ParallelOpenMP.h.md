# `.\pytorch\aten\src\ATen\ParallelOpenMP.h`

```
#pragma once

#include <algorithm>  // 包含算法标准库，用于使用 std::min
#include <atomic>     // 包含原子操作标准库，用于处理原子操作
#include <cstddef>    // 包含标准库定义的一些常量和类型
#include <exception>  // 包含异常处理标准库，用于异常处理

#ifdef _OPENMP
#define INTRA_OP_PARALLEL

#include <omp.h>      // 如果使用 OpenMP，包含 OpenMP 标准库
#endif

#ifdef _OPENMP
namespace at::internal {  // 声明命名空间 at::internal
template <typename F>
inline void invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& f) {
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;  // 初始化原子标志，用于异常处理
  std::exception_ptr eptr;                      // 异常指针，用于捕获并传播异常

#pragma omp parallel  // 使用 OpenMP 创建并行区块
  {
    // 根据粒度和线程数选择任务数量
    // 由于 GOMP 线程池的 bug，不能使用 num_threads 子句（参见 #32008）
    int64_t num_threads = omp_get_num_threads();  // 获取当前线程数
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));  // 计算任务数上限
    }

    int64_t tid = omp_get_thread_num();  // 获取当前线程的线程 ID
    int64_t chunk_size = divup((end - begin), num_threads);  // 计算每个线程处理的数据块大小
    int64_t begin_tid = begin + tid * chunk_size;  // 计算当前线程处理的起始位置
    if (begin_tid < end) {
      try {
        internal::ThreadIdGuard tid_guard(tid);  // 使用线程 ID 创建线程 ID 保护器
        f(begin_tid, std::min(end, chunk_size + begin_tid));  // 调用传入的函数对象处理数据块
      } catch (...) {
        if (!err_flag.test_and_set()) {  // 捕获异常并设置错误标志
          eptr = std::current_exception();  // 获取当前异常指针
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);  // 如果存在异常，则重新抛出异常
  }
}
} // namespace at::internal  // 结束命名空间声明
#endif // _OPENMP
```