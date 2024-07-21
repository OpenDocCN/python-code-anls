# `.\pytorch\c10\core\thread_pool.cpp`

```py
// 包含C10库中的线程池和日志记录头文件
#include <c10/core/thread_pool.h>
#include <c10/util/Logging.h>

// 如果不是PowerPC或s390x架构，则包含cpuinfo头文件
#if !defined(__powerpc__) && !defined(__s390x__)
#include <cpuinfo.h>
#endif

// C10命名空间
namespace c10 {

// 返回默认线程数的静态方法
size_t TaskThreadPoolBase::defaultNumThreads() {
  size_t num_threads = 0;
  
  // 如果cpuinfo初始化成功
#if !defined(__powerpc__) && !defined(__s390x__)
  if (cpuinfo_initialize()) {
    // 根据cpuinfo术语，核心数是物理核心数，处理器数是虚拟核心数
    // 线程池默认使用物理核心数
    size_t num_cores = cpuinfo_get_cores_count();
    num_threads = cpuinfo_get_processors_count();
    
    // 如果核心数大于0且小于处理器数，则返回核心数
    if (num_cores > 0 && num_cores < num_threads) {
      return num_cores;
    }
    
    // 如果处理器数大于0，则返回处理器数
    if (num_threads > 0) {
      return num_threads;
    }
  }
#endif

  // 使用C++标准库获取硬件并发线程数，如果获取不到则返回1
  num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 1;
  }
  
  // 返回计算得到的线程数
  return num_threads;
}

// 线程池构造函数
ThreadPool::ThreadPool(
    int pool_size,
    int numa_node_id,
    const std::function<void()>& init_thread)
    : threads_(pool_size < 0 ? defaultNumThreads() : pool_size), // 初始化线程池大小
      running_(true), // 设置线程池运行状态为true
      complete_(true), // 设置任务完成状态为true
      available_(threads_.size()), // 设置可用线程数为线程池大小
      total_(threads_.size()), // 设置总线程数为线程池大小
      numa_node_id_(numa_node_id) // 设置NUMA节点ID
{
  // 初始化线程池中的每个线程
  for (std::size_t i = 0; i < threads_.size(); ++i) {
    threads_[i] = std::thread([this, i, init_thread]() {
      // 如果有初始化线程的函数，则调用它
      if (init_thread) {
        init_thread();
      }
      // 调用线程池的主循环方法
      this->main_loop(i);
    });
  }
}

// 线程池析构函数
ThreadPool::~ThreadPool() {
  // 将运行状态设置为false，然后通知所有线程
  {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
    condition_.notify_all();
  }

  // 等待所有线程结束
  for (auto& t : threads_) {
    try {
      t.join();
    } catch (const std::exception&) {
      // 捕获所有异常
    }
  }
}

// 返回线程池中线程的数量
size_t ThreadPool::size() const {
  return threads_.size();
}

// 返回线程池中可用线程的数量
size_t ThreadPool::numAvailable() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return available_;
}

// 检查当前线程是否在线程池中
bool ThreadPool::inThreadPool() const {
  for (auto& thread : threads_) {
    if (thread.get_id() == std::this_thread::get_id()) {
      return true;
    }
  }
  return false;
}

// 向线程池中添加任务
void ThreadPool::run(std::function<void()> func) {
  // 如果线程池为空，抛出运行时错误
  if (threads_.empty()) {
    throw std::runtime_error("No threads to run a task");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  
  // 设置任务并通知条件变量，以便工作线程唤醒并使用任务
  tasks_.emplace(std::move(func));
  complete_ = false;
  condition_.notify_one();
}

// 等待所有工作任务完成
void ThreadPool::waitWorkComplete() {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_.wait(lock, [&]() { return complete_; });
}

// 线程池主循环方法
void ThreadPool::main_loop(std::size_t index) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (running_) {
    // 在任务队列为空且线程池仍在运行时等待条件变量
    condition_.wait(lock, [&]() { return !tasks_.empty() || !running_; });
    
    // 如果线程池不再运行，则跳出循环
    if (!running_) {
      break;
    }

    // 本地复制任务并从队列中移除，这是为了安全地执行任务
    // 以自己的作用域运行任务对象，以便在任务运行后立即销毁。这在函数包含通过 bind 绑定的 shared_ptr 参数时非常有用。
    {
      // 取出队列中的第一个任务元素
      task_element_t tasks = std::move(tasks_.front());
      // 弹出队列中的第一个任务元素
      tasks_.pop();
      // 减少可用线程计数，表示线程不再可用
      --available_;

      // 解锁互斥量
      lock.unlock();

      // 执行任务
      try {
        // 如果任务指定了任务ID，则使用任务ID运行
        if (tasks.run_with_id) {
          tasks.with_id(index);
        } else {
          // 否则执行无ID的任务
          tasks.no_id();
        }
      } catch (const std::exception& e) {
        // 捕获并记录异常信息
        LOG(ERROR) << "Exception in thread pool task: " << e.what();
      } catch (...) {
        // 捕获并记录未知异常信息
        LOG(ERROR) << "Exception in thread pool task: unknown";
      }

      // 在重新获取锁之前销毁任务。因为任务是用户提供的 std::function，
      // 它们在销毁期间可以运行任意代码，包括可能会重新进入调用 ThreadPool 的代码（如果我们持有锁，则会导致死锁）。
    }

    // 重新获取锁，恢复状态
    lock.lock();

    // 增加可用线程计数，表示线程现在可用
    ++available_;
    // 如果任务队列为空，并且所有线程都可用，则标记为已完成
    if (tasks_.empty() && available_ == total_) {
      complete_ = true;
      // 通知一个等待的线程任务已完成
      completed_.notify_one();
    }

    // 故意在尾部保持锁定状态，以便该线程有机会在另一个线程获取锁之前获取新任务。
  } // while running_
}

// C10_DEFINE_SHARED_REGISTRY 宏的使用示例，定义了一个名为 ThreadPoolRegistry 的共享注册表，注册表的键类型为 TaskThreadPoolBase，
// 值类型为 std::tuple<int, int, bool>。
C10_DEFINE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPoolBase,
    int,
    int,
    bool);
// 结束命名空间 c10
} // namespace c10
```