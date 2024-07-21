# `.\pytorch\aten\src\ATen\ParallelOpenMP.cpp`

```py
#include <ATen/Config.h>
#include <ATen/core/jit_type.h>
#if AT_PARALLEL_OPENMP
#include <ATen/Parallel.h>
#include <ATen/ParallelFuture.h>

#include <atomic>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

namespace at {

#if AT_MKLDNN_ENABLED()
namespace native { namespace mkldnn {
void clear_computation_cache();
}} // namespace native::mkldnn
#endif

namespace {
// 用户设置的线程数，使用原子整数保证多线程安全
std::atomic<int> num_threads{-1};
// 每个线程的线程 ID，使用线程局部存储（thread_local）来保存
thread_local int this_thread_id{0};
} // namespace

// 初始化全局线程数
void init_num_threads() {
  auto nthreads = num_threads.load();
  if (nthreads > 0) {
    // 如果用户已经设置了线程数，则根据设置的线程数初始化
    set_num_threads(nthreads);
  } else {
#if defined(_OPENMP) && AT_MKL_ENABLED() && !AT_MKL_SEQUENTIAL()
    // 如果使用了 MKL 并且使用了 OpenMP，则确保线程数匹配
    // 否则，MKL 和启用了 OpenMP 的功能将会导致性能下降（在 GCC 5.4 中还可能导致内存泄漏）
    omp_set_num_threads(mkl_get_max_threads());
#elif defined(_OPENMP)
    // 使用默认的线程数设置 OpenMP 的线程数
    omp_set_num_threads(intraop_default_num_threads());
#endif
  }
}

// 设置线程数
void set_num_threads(int nthreads) {
  TORCH_CHECK(nthreads > 0, "Expected positive number of threads");
  // 将用户设置的线程数保存到全局原子整数中
  num_threads.store(nthreads);
#ifdef _OPENMP
  // 设置 OpenMP 的线程数
  omp_set_num_threads(nthreads);
#endif
#if AT_MKL_ENABLED()
  // 设置 MKL 的线程数
  mkl_set_num_threads_local(nthreads);

  // 因为 PyTorch 在 MKL 调用外也使用了 OpenMP，
  // 我们希望这个标志为 false，以避免在每次 MKL / 非 MKL 边界处
  // 破坏和重新创建线程，参见 https://github.com/pytorch/pytorch/issues/13757
  mkl_set_dynamic(false);
#endif
#ifdef USE_PTHREADPOOL
  // 因为 PyTorch 在 QNNPACK 中使用了 caffe2::pthreadpool()，
  // 我们需要确保线程池的线程数与设置的线程数匹配
  caffe2::PThreadPool* const pool = caffe2::pthreadpool();
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");
  pool->set_thread_count(nthreads);
#endif
#if AT_MKLDNN_ENABLED()
  // 清理 MKLDNN 的计算缓存
  at::native::mkldnn::clear_computation_cache();
#endif
}

// 获取当前使用的线程数
int get_num_threads() {
#ifdef _OPENMP
  // 惰性初始化线程数
  at::internal::lazy_init_num_threads();
  // 返回当前 OpenMP 的最大线程数
  return omp_get_max_threads();
#else
  // 非 OpenMP 环境下，返回默认的线程数 1
  return 1;
#endif
}

// 获取当前线程的线程 ID
int get_thread_num() {
  return this_thread_id;
}

namespace internal {
// 设置当前线程的线程 ID
void set_thread_num(int id) {
  this_thread_id = id;
}
}

// 判断当前是否处于并行区域
bool in_parallel_region() {
#ifdef _OPENMP
  // 判断当前是否处于 OpenMP 的并行区域
  return omp_in_parallel();
#else
  // 非 OpenMP 环境下，始终返回 false
  return false;
#endif
}

// 在同步内操作中执行给定的函数
void intraop_launch(std::function<void()> func) {
  // 在 OpenMP 情况下，直接执行给定的函数
  func();
}

// 在同步内操作中执行给定的函数，并返回一个 Future 对象
c10::intrusive_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
  // 执行给定的函数
  func();
  // 创建并标记完成一个空的 Future 对象
  auto future = c10::make_intrusive<c10::ivalue::Future>(NoneType::get());
  future->markCompleted();
  return future;
}

} // namespace at
#endif
```