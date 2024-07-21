# `.\pytorch\aten\src\ATen\Parallel-inl.h`

```
#pragma once


// 指令：#pragma once，确保头文件只被包含一次

#include <c10/util/Exception.h>
#include <c10/util/ParallelGuard.h>
#include <c10/util/SmallVector.h>


// 包含相关头文件，用于异常处理、并行保护、小向量的操作

namespace at {


// 进入命名空间 at

template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  // 定义并行循环函数，接受起始位置、结束位置、粒度大小和函数对象 f

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grain_size >= 0);
  // 断言：确保 grain_size 大于等于 0，仅在调试模式下起作用

  if (begin >= end) {
    return;
  }
  // 如果起始位置大于等于结束位置，直接返回，无需执行循环

#ifdef INTRA_OP_PARALLEL
  at::internal::lazy_init_num_threads();
  // 如果定义了 INTRA_OP_PARALLEL，调用初始化线程函数

  const auto numiter = end - begin;
  const bool use_parallel =
      (numiter > grain_size && numiter > 1 && !at::in_parallel_region() &&
       at::get_num_threads() > 1);
  // 确定是否可以使用并行执行：迭代次数大于粒度、大于 1、不在并行区域、线程数大于 1

  if (!use_parallel) {
    internal::ThreadIdGuard tid_guard(0);
    // 如果不能使用并行，创建线程 ID 保护对象

    c10::ParallelGuard guard(true);
    // 并行保护，确保线程安全

    f(begin, end);
    // 执行函数 f，传递起始位置和结束位置
    return;
  }

  internal::invoke_parallel(
      begin, end, grain_size, [&](int64_t begin, int64_t end) {
        c10::ParallelGuard guard(true);
        // 在并行执行期间，确保线程安全

        f(begin, end);
        // 执行函数 f，传递当前分块的起始和结束位置
      });
#else
  internal::ThreadIdGuard tid_guard(0);
  // 如果没有定义 INTRA_OP_PARALLEL，创建线程 ID 保护对象

  c10::ParallelGuard guard(true);
  // 并行保护，确保线程安全

  f(begin, end);
  // 执行函数 f，传递起始位置和结束位置
#endif
}


// 结束并行循环函数模板定义

template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const scalar_t ident,
    const F& f,
    const SF& sf) {
  // 定义并行归约函数模板，接受起始位置、结束位置、粒度大小、初始值、函数对象 f 和归约函数对象 sf

  TORCH_CHECK(grain_size >= 0);
  // 检查：确保 grain_size 大于等于 0

  if (begin >= end) {
    return ident;
  }
  // 如果起始位置大于等于结束位置，直接返回初始值 ident

#ifdef INTRA_OP_PARALLEL
  at::internal::lazy_init_num_threads();
  // 如果定义了 INTRA_OP_PARALLEL，调用初始化线程函数

  const auto max_threads = at::get_num_threads();
  // 获取当前可用的最大线程数

  const bool use_parallel =
      ((end - begin) > grain_size && !at::in_parallel_region() &&
       max_threads > 1);
  // 确定是否可以使用并行执行：范围大于粒度、不在并行区域、线程数大于 1

  if (!use_parallel) {
    internal::ThreadIdGuard tid_guard(0);
    // 如果不能使用并行，创建线程 ID 保护对象

    c10::ParallelGuard guard(true);
    // 并行保护，确保线程安全

    return f(begin, end, ident);
    // 执行归约函数 f，传递起始位置、结束位置和初始值 ident，并返回结果
  }

  c10::SmallVector<scalar_t, 64> results(max_threads, ident);
  // 创建结果向量，大小为最大线程数，初始值为 ident

  internal::invoke_parallel(
      begin,
      end,
      grain_size,
      [&](const int64_t my_begin, const int64_t my_end) {
        const auto tid = at::get_thread_num();
        // 获取当前线程的线程 ID

        c10::ParallelGuard guard(true);
        // 并行保护，确保线程安全

        results[tid] = f(my_begin, my_end, ident);
        // 执行归约函数 f，将结果存入对应线程 ID 的结果位置
      });

  scalar_t result = ident;
  // 初始化归约结果为初始值 ident

  for (auto partial_result : results) {
    result = sf(result, partial_result);
    // 使用归约函数 sf，对所有部分结果进行最终归约
  }

  return result;
  // 返回最终的归约结果
#else
  internal::ThreadIdGuard tid_guard(0);
  // 如果没有定义 INTRA_OP_PARALLEL，创建线程 ID 保护对象

  c10::ParallelGuard guard(true);
  // 并行保护，确保线程安全

  return f(begin, end, ident);
  // 执行归约函数 f，传递起始位置、结束位置和初始值 ident，并返回结果
#endif
}


// 结束并行归约函数模板定义

} // namespace at


// 结束命名空间 at
```