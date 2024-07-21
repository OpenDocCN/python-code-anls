# `.\pytorch\c10\core\thread_pool.h`

```py
#pragma once

#include <atomic>                               // 原子操作
#include <condition_variable>                   // 条件变量
#include <cstddef>                              // 标准库定义的宏
#include <functional>                           // 函数对象
#include <mutex>                                // 互斥量
#include <queue>                                // 队列
#include <thread>                               // 线程
#include <utility>                              // 实用工具
#include <vector>                               // 向量

#include <c10/macros/Export.h>                  // C10 导出宏
#include <c10/util/Registry.h>                  // C10 注册表工具
#include <c10/util/numa.h>                      // C10 NUMA 工具
#include <c10/util/thread_name.h>               // C10 线程名称工具

namespace c10 {

class C10_API TaskThreadPoolBase {
 public:
  virtual void run(std::function<void()> func) = 0;  // 虚函数，运行给定的函数
  virtual size_t size() const = 0;                   // 虚函数，返回线程池大小
  virtual size_t numAvailable() const = 0;           // 虚函数，返回空闲线程数
  virtual bool inThreadPool() const = 0;             // 虚函数，检查当前线程是否在线程池中
  virtual ~TaskThreadPoolBase() noexcept = default;  // 虚析构函数
  static size_t defaultNumThreads();                 // 静态函数，返回默认线程数
};

class C10_API ThreadPool : public c10::TaskThreadPoolBase {
 protected:
  struct task_element_t {
    bool run_with_id;                             // 标记是否使用 ID 运行
    const std::function<void()> no_id;            // 无 ID 的任务函数
    const std::function<void(std::size_t)> with_id; // 带有 ID 的任务函数

    explicit task_element_t(std::function<void()> f)
        : run_with_id(false), no_id(std::move(f)), with_id(nullptr) {}  // 构造函数，初始化无 ID 任务
    explicit task_element_t(std::function<void(std::size_t)> f)
        : run_with_id(true), no_id(nullptr), with_id(std::move(f)) {}   // 构造函数，初始化带 ID 任务
  };

  std::queue<task_element_t> tasks_;               // 任务队列
  std::vector<std::thread> threads_;               // 线程向量
  mutable std::mutex mutex_;                       // 互斥量
  std::condition_variable condition_;              // 条件变量，用于任务通知
  std::condition_variable completed_;              // 条件变量，用于任务完成通知
  std::atomic_bool running_;                       // 原子布尔变量，表示线程池是否运行中
  bool complete_;                                  // 标记所有任务是否完成
  std::size_t available_;                          // 可用线程数
  std::size_t total_;                              // 总线程数
  int numa_node_id_;                               // NUMA 节点 ID

 public:
  ThreadPool() = delete;                          // 禁用默认构造函数

  explicit ThreadPool(
      int pool_size,
      int numa_node_id = -1,
      const std::function<void()>& init_thread = nullptr);  // 构造函数

  ~ThreadPool() override;                         // 析构函数

  size_t size() const override;                   // 返回线程池大小

  size_t numAvailable() const override;           // 返回空闲线程数

  bool inThreadPool() const override;             // 检查当前线程是否在线程池中

  void run(std::function<void()> func) override;  // 运行给定的函数

  template <typename Task>
  void runTaskWithID(Task task) {                 // 运行带有 ID 的任务
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 设置任务并通知条件变量，以便工作线程被唤醒并使用任务
    tasks_.emplace(static_cast<std::function<void(std::size_t)>>(task));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief 等待队列为空
  void waitWorkComplete();

 private:
  // @brief 线程池线程的入口点
  void main_loop(std::size_t index);
};

class C10_API TaskThreadPool : public c10::ThreadPool {
 public:
  explicit TaskThreadPool(int pool_size, int numa_node_id = -1)
      : ThreadPool(pool_size, numa_node_id, [numa_node_id]() {
          setThreadName("CaffeTaskThread");
          NUMABind(numa_node_id);
        }) {}                                    // 构造函数，设置线程名称和 NUMA 绑定
};

C10_DECLARE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPoolBase,
    int,
    int,
    bool);

} // namespace c10
```