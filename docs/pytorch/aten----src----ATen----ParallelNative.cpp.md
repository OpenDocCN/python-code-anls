# `.\pytorch\aten\src\ATen\ParallelNative.cpp`

```
// 包含 ATen 库的配置文件
#include <ATen/Config.h>

// 如果 AT_PARALLEL_NATIVE 定义为真，则包含相关的并行处理头文件
#if AT_PARALLEL_NATIVE
#include <ATen/Parallel.h>
#include <ATen/ParallelFuture.h>
#include <ATen/PTThreadPool.h>

// 如果不是 C10_MOBILE 平台，包含 C10 核心的线程池和工具头文件
#ifndef C10_MOBILE
#include <c10/core/thread_pool.h>
#include <c10/util/irange.h>
// 如果是 C10_MOBILE 平台，包含 caffe2 的线程池实现头文件
#else
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#endif // C10_MOBILE

// 包含原子操作的头文件
#include <atomic>
#include <utility>

// 如果定义了 _OPENMP，则包含 OpenMP 的头文件
#ifdef _OPENMP
#include <omp.h>
#endif

// 如果 AT_MKL_ENABLED() 返回真，则包含 MKL 的头文件
#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

// ATen 的命名空间
namespace at {

// 匿名命名空间，用于定义局部变量和函数
namespace {

// 用于标记是否当前线程处于并行区域的 thread_local 变量
thread_local bool in_parallel_region_ = false;

// 用于标记并行原语设置的线程编号的 thread_local 变量
thread_local int thread_num_ = 0;

// 设置当前线程是否在并行区域的状态
void _set_in_parallel_region(bool in_region) {
  in_parallel_region_ = in_region;
}

}  // namespace (anonymous)

// ATen 内部命名空间，定义了设置线程编号的函数
namespace internal {

// 设置当前线程的编号
void set_thread_num(int thread_num) {
  thread_num_ = thread_num;
}
}

// 匿名命名空间，定义了取消设置线程编号的函数
namespace {

// 取消当前线程的线程编号设置
void _unset_thread_num() {
  thread_num_ = 0;
}

// 如果不是 C10_MOBILE 平台，定义了线程池中线程数量未设置和已消费的常量
#ifndef C10_MOBILE

const int NOT_SET = -1;
const int CONSUMED = -2;

// 原子变量，用于存储用户设置的线程数量
// NOT_SET -> 正值 -> CONSUMED
// 或
// NOT_SET -> CONSUMED
// 含义：
// - NOT_SET：线程池未初始化，用户未设置值
// - 正值：线程池未初始化，用户设置了值
// - CONSUMED：线程池已初始化
std::atomic<int> num_intraop_threads{NOT_SET};

// 获取用于操作内部任务的线程池对象
TaskThreadPoolBase& _get_intraop_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          /* pool_size */ _num_pool_threads(num_intraop_threads.exchange(CONSUMED)),
          /* create_new */ true); // 为内部操作创建一个独立的线程池
  return *pool;
}

#endif // C10_MOBILE

// 使用线程池在 [0, `range`) 范围内运行 lambda 函数 `fn`
// `fn` 将以 (thread_pool_task_id, task_id) 参数调用
void _run_with_pool(const std::function<void(int, size_t)>& fn, size_t range) {
#ifndef C10_MOBILE
  // 对于范围 [1, range) 中的每个任务 i，在线程池中运行任务
  for (const auto i : c10::irange(1, range)) {
    _get_intraop_pool().run([fn, i]() { fn((int)i, i); });
  }
  // 直接在当前线程上运行第一个任务
  fn(0, 0);
#else
  // 获取用于 caffe2 的线程池对象
  caffe2::PThreadPool* const pool = caffe2::pthreadpool();
  // 断言线程池对象的有效性
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");

  // 在线程池中运行任务，PThreadPool::run() 是阻塞的
  pool->run(
    // PThreadPool::run() 是阻塞的。这个 lambda 的 std::function 引用在 PThreadPool::run() 返回前不能超出作用域。
    [&fn](const size_t task_id) {
      fn(0 /* 未使用 */, task_id);
    }, range);
#endif // C10_MOBILE
}

// RAII 保护区域，支持 in_parallel_region() 和 get_thread_num() API
struct ParallelRegionGuard {
  // 构造函数，用于设置线程编号
  ParallelRegionGuard(int task_id) {
    internal::set_thread_num(task_id);
}
    _set_in_parallel_region(true);
  }

  ~ParallelRegionGuard() {
    _set_in_parallel_region(false);
    _unset_thread_num();
  }



# 进入并行区域的标志设置为 true
_set_in_parallel_region(true);



# 并行区域的析构函数，在对象销毁时执行
~ParallelRegionGuard() {
    # 退出并行区域，将标志设置为 false
    _set_in_parallel_region(false);
    # 清除线程号信息
    _unset_thread_num();
}
};  // 结束命名空间

} // namespace

namespace internal {

inline std::tuple<size_t, size_t> calc_num_tasks_and_chunk_size(
    int64_t begin, int64_t end, int64_t grain_size) {
  if ((end - begin) < grain_size) {
    // 如果任务量小于颗粒大小，返回单任务和剩余任务大小
    return std::make_tuple(1, std::max((int64_t)0, end - begin));
  }
  // 根据颗粒大小和线程数选择任务数量
  size_t chunk_size = divup((end - begin), get_num_threads());
  // 确保每个任务至少有颗粒大小这么多
  chunk_size = std::max((size_t)grain_size, chunk_size);
  // 根据任务数量计算分块大小
  size_t num_tasks = divup((end - begin), chunk_size);
  return std::make_tuple(num_tasks, chunk_size);
}

void invoke_parallel(
  const int64_t begin,
  const int64_t end,
  const int64_t grain_size,
  const std::function<void(int64_t, int64_t)>& f) {
  at::internal::lazy_init_num_threads();

  size_t num_tasks = 0, chunk_size = 0;
  // 计算任务数量和分块大小
  std::tie(num_tasks, chunk_size) =
      internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);

  // 并行执行的状态结构体
  struct {
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;  // 错误标志
    std::exception_ptr eptr;  // 异常指针
    std::mutex mutex;  // 互斥锁
    std::atomic_size_t remaining{0};  // 剩余任务计数
    std::condition_variable cv;  // 条件变量
  } state;

  // 并行任务的执行函数
  auto task = [f, &state, begin, end, chunk_size]
      (int /* unused */, size_t task_id) {
    int64_t local_start = begin + task_id * chunk_size;
    if (local_start < end) {
      int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
      try {
        ParallelRegionGuard guard(task_id);
        f(local_start, local_end);
      } catch (...) {
        // 捕获异常并设置错误标志和异常指针
        if (!state.err_flag.test_and_set()) {
          state.eptr = std::current_exception();
        }
      }
    }
    {
      // 减少剩余任务计数，并在需要时通知完成
      std::unique_lock<std::mutex> lk(state.mutex);
      if (--state.remaining == 0) {
        state.cv.notify_one();
      }
    }
  };
  state.remaining = num_tasks;
  // 使用线程池执行任务
  _run_with_pool(std::move(task), num_tasks);

  // 等待所有任务完成
  {
    std::unique_lock<std::mutex> lk(state.mutex);
    if (state.remaining != 0) {
      state.cv.wait(lk);
    }
  }
  // 如果有异常，重新抛出异常指针
  if (state.eptr) {
    std::rethrow_exception(state.eptr);
  }
}

} // namespace internal

void init_num_threads() {
#ifdef _OPENMP
  omp_set_num_threads(1);  // 设置 OpenMP 线程数为 1
#endif

#if AT_MKL_ENABLED()
  mkl_set_num_threads(1);  // 设置 MKL 线程数为 1
#endif

#ifdef C10_MOBILE
  caffe2::pthreadpool();  // 在移动设备上初始化 pthreadpool
#endif
}

void set_num_threads(int nthreads) {
#ifndef C10_MOBILE
  TORCH_CHECK(nthreads > 0, "Expected positive number of threads");
  int no_value = NOT_SET;
  if (!num_intraop_threads.compare_exchange_strong(no_value, nthreads)) {
    // num_intraop_threads 只能存储正整数或 CONSUMED，
    // 检查请求的大小是否与当前大小相同
    int stored_nthreads = num_intraop_threads.load();
    if (stored_nthreads <= 0) {
      // 加一是因为主线程
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      stored_nthreads = _get_intraop_pool().size() + 1;
    }
    # 如果存储的线程数与给定的线程数不相等时执行以下代码块
    if (stored_nthreads != nthreads) {
      # 输出警告信息，说明不能在并行工作已经开始或在使用本地并行后端时，
      # 在已经调用过 set_num_threads 函数后设置线程数
      TORCH_WARN(
        "Cannot set number of intraop threads "
        "after parallel work has started or after set_num_threads call "
        "when using native parallel backend");
    }
  }
#else
// 获取指向全局线程池的常量指针
caffe2::PThreadPool* const pool = caffe2::pthreadpool();
// 内部断言，确保线程池有效
TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");
// 设置线程池的线程数量
pool->set_thread_count(nthreads);
#endif // C10_MOBILE
}

// 获取当前线程数量的函数
int get_num_threads() {
  // 惰性初始化线程数量
  at::internal::lazy_init_num_threads();
#ifndef C10_MOBILE
  // 不必要地初始化线程池，因为初始化后无法重新调整大小
  int nthreads = num_intraop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == NOT_SET) {
    // 返回默认的线程数量
    return intraop_default_num_threads();
  } else {
    TORCH_INTERNAL_ASSERT(nthreads == CONSUMED);
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    // 返回内部操作池的大小加一
    return _get_intraop_pool().size() + 1;
  }
#else
  // 获取指向全局线程池的常量指针
  caffe2::PThreadPool* const pool = caffe2::pthreadpool();
  // 内部断言，确保线程池有效
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!")
  // 如果处于并行区域中，则返回当前线程数
  return in_parallel_region() ? 1 /* current thread */ : pool->get_thread_count();
#endif // C10_MOBILE
}

// 返回当前线程编号的函数
int get_thread_num() {
  return thread_num_;
}

// 检查当前是否处于并行区域的函数
bool in_parallel_region() {
#ifndef C10_MOBILE
  // 如果在并行区域中或者在内部操作线程数量为 CONSUMED 的情况下且线程池正在运行，则返回 true
  return in_parallel_region_ || (
    num_intraop_threads.load() == CONSUMED &&
    // Needed as intraop_launch() doesn't set in_parallel_region().
    _get_intraop_pool().inThreadPool()
  );
#else
  // 在移动设备上，始终返回 false
  return in_parallel_region_;
#endif // C10_MOBILE
}

// 在内部操作中启动函数的函数，接受一个函数作为参数
void intraop_launch(std::function<void()> func) {
#ifndef C10_MOBILE
  // 如果不在并行区域且当前线程数大于 1，则在内部操作池中运行函数
  if (!in_parallel_region() && get_num_threads() > 1) {
    _get_intraop_pool().run(std::move(func));
  } else {
    // 如果在并行区域中，则直接执行函数
    func();
  }
#else
  // 在移动设备上，直接执行函数
  // TODO: caffe2::PThreadPool 仅提供数据并行的 API，不支持任务并行
  func();
#endif // C10_MOBILE
}

// 在内部操作中启动返回未来对象的函数，接受一个函数作为参数
c10::intrusive_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
#ifndef C10_MOBILE
  // 创建一个空的未来对象
  auto future = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
  // 如果不在并行区域且当前线程数大于 1，则在内部操作池中运行函数，并在运行后标记未来对象为完成状态
  if (!in_parallel_region() && get_num_threads() > 1) {
    _get_intraop_pool().run(
      [func, future]() {
        func();
        future->markCompleted();
      }
    );
  } else {
    // 如果在并行区域中或者只有一个线程，则直接执行函数，并标记未来对象为完成状态
    func();
    future->markCompleted();
  }
  return future;
#else
  // 在移动设备上，直接执行函数，并标记未来对象为完成状态
  // TODO: caffe2::PThreadPool 仅提供数据并行的 API，不支持任务并行
  auto future = c10::make_intrusive<c10::ivalue::Future>(c10::dynT<NoneType>());
  func();
  future->markCompleted();
  return future;
#endif // C10_MOBILE
}

} // namespace at
#endif
```