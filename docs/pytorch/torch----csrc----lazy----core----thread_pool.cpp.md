# `.\pytorch\torch\csrc\lazy\core\thread_pool.cpp`

```
namespace torch {
namespace lazy {
namespace {

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads) {
    // 构造函数，初始化线程向量并预留空间
    threads_.reserve(num_threads);
    // 循环创建指定数量的线程并添加到线程向量中
    for (const auto i : c10::irange(num_threads)) {
      (void)i; // 抑制未使用变量警告
      threads_.emplace_back([this]() { Worker(); });
    }
  }

  ~ThreadPool() {
    {
      // 析构函数，设置退出标志并通知所有线程
      std::lock_guard<std::mutex> lock(mutex_);
      exiting_ = true;
      cv_.notify_all();
    }
    // 等待所有线程结束
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void Schedule(std::function<void()> closure) {
    // 调度函数，用于提交任务到线程池执行
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // 如果当前等待执行的任务数量少于等待的线程数，直接将任务放入队列
      if (work_.size() < waiting_) {
        work_.emplace_back(std::move(closure));
        lock.unlock();
        cv_.notify_one();
        return;
      }
    }
    // 否则，在单独的线程上调度任务执行
    ScheduleOnThread(std::move(closure));
  }

 private:
  void Worker() {
    // 工作线程函数，循环获取并执行任务
    while (true) {
      std::function<void()> closure = GetWork();
      if (closure == nullptr) {
        break;
      }
      try {
        closure();
      } catch (const std::exception& ex) {
        // 捕获任务执行中的异常，记录异常信息
        TORCH_LAZY_COUNTER("ThreadPoolException", 1);
        LOG(ERROR) << "Exception from running thread pool closure: "
                   << ex.what();
      }
    }
  }

  void ScheduleOnThread(std::function<void()> closure) {
    // 在单独的线程上调度执行任务
    std::thread thread(std::move(closure));
    thread.detach();
  }

  std::function<void()> GetWork() {
    // 获取任务函数，从任务队列中取出任务
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_;
    cv_.wait(lock, [this] { return exiting_ || !work_.empty(); });
    --waiting_;
    if (work_.empty()) {
      return nullptr;
    }
    std::function<void()> closure(std::move(work_.front()));
    work_.pop_front();
    return closure;
  }

  std::vector<std::thread> threads_; // 线程向量
  std::mutex mutex_;                 // 互斥锁
  std::condition_variable cv_;       // 条件变量
  bool exiting_ = false;             // 退出标志
  std::deque<std::function<void()>> work_; // 任务队列
  size_t waiting_ = 0;               // 等待任务的线程数
};

ThreadPool* GetIoThreadPool() {
  // 获取 IO 线程池单例函数，返回静态的线程池指针
  static ThreadPool* pool =
      new ThreadPool(FLAGS_torch_lazy_io_thread_pool_size);
  return pool;
}

} // namespace

class Completion::Data {
 public:
  void Wait() {
    // 等待完成函数，等待任务完成或异常抛出
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return completed_; });
    if (exptr_ != nullptr) {
      std::rethrow_exception(exptr_);
    }
  }

  static std::function<void()> GetCompleter(
      const std::shared_ptr<Data>& data,
      std::function<void()> closure) {
    // 创建一个自动闭包，捕获给定的闭包和数据
    auto closure_wrapper = [closure = std::move(closure), data]() {
      // 异常指针，用于捕获可能发生的异常
      std::exception_ptr exptr;
      try {
        // 执行捕获的闭包
        closure();
      } catch (...) {
        // 捕获异常并存储到异常指针中
        exptr = std::current_exception();
      }
      // 调用数据对象的Complete方法，传递捕获的异常指针
      data->Complete(exptr);
    };
    // 返回创建的闭包
    return closure_wrapper;
  }

 private:
  // Complete方法：处理传入的异常指针，完成后通知条件变量
  void Complete(std::exception_ptr exptr) {
    // 使用互斥锁保护临界区
    std::lock_guard<std::mutex> lock(mutex_);
    // 将异常指针移动到成员变量中
    exptr_ = std::move(exptr);
    // 设置完成标志为true
    completed_ = true;
    // 通知所有等待条件变量的线程
    cv_.notify_all();
  }

  // 互斥锁，保护对共享数据的访问
  std::mutex mutex_;
  // 条件变量，用于线程间的同步通信
  std::condition_variable cv_;
  // 完成标志，指示任务是否已完成
  bool completed_ = false;
  // 异常指针，用于存储在任务执行过程中捕获的异常
  std::exception_ptr exptr_;
};

Completion::Completion(std::shared_ptr<Data> data) : data_(std::move(data)) {}
# 定义 Completion 类的构造函数，接受一个 std::shared_ptr<Data> 参数，并将其移动到成员变量 data_ 中

void Completion::Wait() {
  data_->Wait();
}
# 定义 Wait 方法，调用 data_ 的 Wait 方法，等待完成

void ScheduleIoClosure(std::function<void()> closure) {
  GetIoThreadPool()->Schedule(std::move(closure));
}
# 定义 ScheduleIoClosure 函数，接受一个 std::function<void()> 的闭包参数，
# 调用 GetIoThreadPool() 获取 I/O 线程池的实例，并调度传入的闭包以进行执行

Completion ScheduleIoClosureWithCompletion(std::function<void()> closure) {
  auto data = std::make_shared<Completion::Data>();
  # 创建一个 std::shared_ptr<Completion::Data> 实例 data，通过 std::make_shared 构造
  GetIoThreadPool()->Schedule(
      Completion::Data::GetCompleter(data, std::move(closure)));
  # 调用 GetIoThreadPool() 获取 I/O 线程池的实例，并调度 Completion::Data::GetCompleter 方法的返回值，
  # 传入 data 和闭包 closure，完成调度
  return Completion(std::move(data));
}
# 定义 ScheduleIoClosureWithCompletion 函数，接受一个 std::function<void()> 的闭包参数，
# 创建一个新的 Completion 对象，通过 GetIoThreadPool() 调度闭包，返回该 Completion 对象

} // namespace lazy
} // namespace torch
```