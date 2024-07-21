# `.\pytorch\aten\src\ATen\ParallelThreadPoolNative.cpp`

```py
// 包含 ATen 库的配置文件
#include <ATen/Config.h>
// 如果支持 OpenMP 或本地并行，则包含并行处理相关的头文件
#if AT_PARALLEL_OPENMP || AT_PARALLEL_NATIVE
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>
#include <ATen/ThreadLocalState.h>

#include <atomic>  // 引入原子操作支持

// ATen 命名空间
namespace at {

// 匿名命名空间中定义的常量
namespace {
const int NOT_SET = -1;  // 表示未设置的常量值
const int CONSUMED = -2;  // 表示已消耗的常量值

// 原子整数，用于存储用户设置的跨操作线程数
std::atomic<int> num_interop_threads{NOT_SET};

// 获取线程池的函数，返回 TaskThreadPoolBase 的引用
TaskThreadPoolBase& get_pool() {
  // 静态局部变量，保证唯一性，存储线程池对象
  static std::shared_ptr<TaskThreadPoolBase> pool =
      // 创建线程池，使用 ThreadPoolRegistry 注册表
      ThreadPoolRegistry()->Create(
          "C10",  // 线程池名称
          /* device_id */ 0,  // 设备 ID，这里为固定值 0
          /* pool_size */ num_interop_threads.exchange(CONSUMED),  // 线程池大小
          /* create_new */ true);  // 是否创建新线程池
  return *pool;  // 返回线程池对象引用
}

// 创建 C10 线程池的工厂函数
std::shared_ptr<TaskThreadPoolBase> create_c10_threadpool(
    int device_id,
    int pool_size,
    bool create_new) {
  // 检查设备 ID 是否为 0
  TORCH_CHECK(device_id == 0);
  // 检查是否创建新的线程池
  TORCH_CHECK(create_new);
  // 返回新创建的 PTThreadPool 对象
  return std::make_shared<PTThreadPool>(pool_size);
}

}  // namespace

// 注册函数，向 ThreadPoolRegistry 注册 C10 线程池创建函数
C10_REGISTER_CREATOR(ThreadPoolRegistry, C10, create_c10_threadpool);

// 设置跨操作线程数的函数
void set_num_interop_threads(int nthreads) {
  // 检查线程数是否为正数
  TORCH_CHECK(nthreads > 0, "Expected positive number of threads");

  int no_value = NOT_SET;
  // 尝试原子地设置跨操作线程数
  TORCH_CHECK(num_interop_threads.compare_exchange_strong(no_value, nthreads),
      "Error: cannot set number of interop threads after parallel work "
      "has started or set_num_interop_threads called");
}

// 获取跨操作线程数的函数
int get_num_interop_threads() {
  // 懒初始化线程数
  at::internal::lazy_init_num_threads();
  int nthreads = num_interop_threads.load();
  if (nthreads > 0) {
    return nthreads;  // 返回用户设置的线程数
  } else if (nthreads == NOT_SET) {
    // 返回默认的线程数
    return TaskThreadPoolBase::defaultNumThreads();
  } else {
    return get_pool().size();  // 返回当前线程池的大小
  }
}

// ATen 内部命名空间中的函数
namespace internal {
// 启动不带线程状态的函数
void launch_no_thread_state(std::function<void()> fn) {
  // 如果定义了 AT_EXPERIMENTAL_SINGLE_THREAD_POOL 宏，则单线程执行
#if AT_EXPERIMENTAL_SINGLE_THREAD_POOL
  intraop_launch(std::move(fn));
#else
  get_pool().run(std::move(fn));  // 否则使用线程池运行函数
#endif
}
}  // namespace internal

// 启动函数，使用线程局部状态和函数作为参数
void launch(std::function<void()> func) {
  // 调用 launch_no_thread_state 函数，传递线程局部状态和函数
  // NOLINTNEXTLINE(modernize-avoid-bind)
  internal::launch_no_thread_state(std::bind([](
    std::function<void()> f, ThreadLocalState thread_locals) {
      ThreadLocalStateGuard guard(std::move(thread_locals));
      f();  // 执行函数
    },
    std::move(func),
    ThreadLocalState()  // 线程局部状态对象
  ));
}

}  // namespace at  // ATen 命名空间结束
#endif  // 结束条件编译指令，检查是否支持并行处理
```