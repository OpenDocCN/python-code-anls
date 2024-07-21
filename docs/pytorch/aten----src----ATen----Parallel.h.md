# `.\pytorch\aten\src\ATen\Parallel.h`

```
#pragma once
#include <ATen/Config.h>
#include <c10/macros/Macros.h>
#include <functional>
#include <string>

namespace at {

// 定义一个内联函数，用于计算 x/y 的上取整
inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// 初始化线程数量
TORCH_API void init_num_threads();

// 设置并行区域中使用的线程数量
TORCH_API void set_num_threads(int);

// 获取可以在并行区域中使用的最大线程数
TORCH_API int get_num_threads();

// 获取当前线程号（从0开始），在并行区域中；在顺序执行区域返回0
TORCH_API int get_thread_num();

// 检查当前代码是否运行在并行区域中
TORCH_API bool in_parallel_region();

namespace internal {

// 第一次并行调用时懒惰地初始化线程数量
inline void lazy_init_num_threads() {
  thread_local bool init = false;
  if (C10_UNLIKELY(!init)) {
    at::init_num_threads();
    init = true;
  }
}

// 设置当前线程的线程号
TORCH_API void set_thread_num(int);

// 线程号保护类，用于在作用域结束时恢复旧的线程号
class TORCH_API ThreadIdGuard {
 public:
  ThreadIdGuard(int new_id) : old_id_(at::get_thread_num()) {
    set_thread_num(new_id);
  }

  ~ThreadIdGuard() {
    set_thread_num(old_id_);
  }

 private:
  int old_id_;
};

} // namespace internal

/*
并行循环

begin: 开始应用用户函数的索引

end: 停止应用用户函数的索引

grain_size: 每个块的元素数量，影响并行化程度

f: 应用于块的用户函数，签名为:
  void f(int64_t begin, int64_t end)

警告：parallel_for 不会将当前线程的线程局部状态复制到工作线程。
这意味着在您的函数体中不能使用张量操作，只能使用数据指针。
*/
template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f);

/*
并行归约

begin: 开始应用归约的索引

end: 停止应用归约的索引

grain_size: 每个块的元素数量，影响中间结果张量的元素数量和并行化程度

ident: 二进制组合函数 sf 的单位元。sf(ident, x) 需要返回 x。

f: 对块进行归约的函数。f 需要具有签名 scalar_t f(int64_t partial_begin, int64_t partial_end, scalar_t identifiy)

sf: 组合两个部分结果的函数 sf。sf 需要具有签名 scalar_t sf(scalar_t x, scalar_t y)

例如，您可能有一个包含10000个元素的张量，并希望将所有元素求和。
使用 grain_size 为 2500 的 parallel_reduce 将会分配一个包含4个元素的中间结果张量。
然后，它将执行您提供的函数"f"，并传递这些块的起始和结束索引（例如0-2499, 2500-4999等）以及组合单位元。
然后，它将结果写入中间结果张量的每个块中。完成后，
*/
template <class F>
inline void parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const typename std::remove_reference<F>::type::value_type& ident,
    const F& f,
    const std::function<typename std::remove_reference<F>::type::value_type(
        typename std::remove_reference<F>::type::value_type,
        typename std::remove_reference<F>::type::value_type)>& sf);

} // namespace at
/*
Defines a template function `parallel_reduce` for parallel computation,
where it divides the range [begin, end) into chunks of size `grain_size`,
and reduces each chunk using function `f`, combining results with `sf`.
The parameter `ident` serves as the identity element for the reduction.
This function is inspired by Intel TBB's approach [1], requiring functions
for subrange accumulation, result combination, and an identity.

Warning: `parallel_reduce` does not propagate thread-local states from
the calling thread to worker threads. This restriction implies that
Tensor operations cannot be safely used within the function body,
only raw data pointers are allowed.

[1] https://software.intel.com/en-us/node/506154
*/
template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t begin,            // Start index of the range to process
    const int64_t end,              // End index (exclusive) of the range to process
    const int64_t grain_size,       // Size of each chunk for parallel processing
    const scalar_t ident,           // Identity element for reduction
    const F& f,                     // Function for subrange accumulation
    const SF& sf);                  // Function for combining two partial results

// Returns a detailed string describing parallelization settings
TORCH_API std::string get_parallel_info();

// Sets number of threads used for inter-op parallelism
TORCH_API void set_num_interop_threads(int);

// Returns the number of threads used for inter-op parallelism
TORCH_API int get_num_interop_threads();

// Launches inter-op parallel task
TORCH_API void launch(std::function<void()> func);

namespace internal {
void launch_no_thread_state(std::function<void()> fn);
} // namespace internal

// Launches intra-op parallel task
TORCH_API void intraop_launch(std::function<void()> func);

// Returns number of intra-op threads used by default
TORCH_API int intraop_default_num_threads();

} // namespace at

#if AT_PARALLEL_OPENMP
#include <ATen/ParallelOpenMP.h> // IWYU pragma: keep
#elif AT_PARALLEL_NATIVE
#include <ATen/ParallelNative.h> // IWYU pragma: keep
#endif

#include <ATen/Parallel-inl.h> // IWYU pragma: keep
```