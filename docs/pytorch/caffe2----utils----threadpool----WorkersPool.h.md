# `.\pytorch\caffe2\utils\threadpool\WorkersPool.h`

```
#pragma once
// 使用 #pragma once 确保头文件只被编译一次

#include <atomic>
// 包含原子操作相关的头文件

#include <condition_variable>
// 包含条件变量相关的头文件

#include <thread>
// 包含线程相关的头文件

#include "c10/util/thread_name.h"
// 包含自定义的线程命名工具的头文件

#include <c10/util/irange.h>
// 包含用于生成整数范围的头文件

#include <c10/util/Logging.h>
// 包含日志记录相关的头文件

#if defined(_MSC_VER)
#include <intrin.h>
// 如果是 MSVC 编译器，包含 instrin.h 用于内部函数调用
#endif

namespace caffe2 {

// Uses code derived from gemmlowp,
// https://github.com/google/gemmlowp/blob/6c91e1ed0c2eff1182d804310b92911fe9c18019/internal/multi_thread_gemm.h
// Changes:
// - allocation-free execute()
// - Use RAII where possible.
// - Run the first task on the main thread (since that is the largest task).
// - removed custom allocator.
// - Removed some ifdef's
// - cache-line align Worker.
// - use std::atomic instead of volatile and custom barriers.
// - use std::mutex/std::condition_variable instead of raw pthreads.

constexpr size_t kGEMMLOWPCacheLineSize = 64;
// 定义缓存行的大小为 64 字节

template <typename T>
struct AllocAligned {
  // Allocate a T aligned at an `align` byte address
  // 分配一个对齐到 `align` 字节地址的 T 类型对象

  template <typename... Args>
  static T* alloc(Args&&... args) {
    void* p = nullptr;

#if defined(__ANDROID__)
    p = memalign(kGEMMLOWPCacheLineSize, sizeof(T));
    // 如果是在 Android 平台，使用 memalign 进行内存对齐分配
#elif defined(_MSC_VER)
    p = _aligned_malloc(sizeof(T), kGEMMLOWPCacheLineSize);
    // 如果是在 MSVC 平台，使用 _aligned_malloc 进行内存对齐分配
#else
    auto res = posix_memalign((void**)&p, kGEMMLOWPCacheLineSize, sizeof(T));
    (void)res;
    // 在其他平台使用 posix_memalign 进行内存对齐分配
#endif

    if (p) {
      return new (p) T(std::forward<Args>(args)...);
      // 如果成功分配内存，则在分配的地址上构造对象 T 并返回指针
    }

    return nullptr;
    // 分配失败时返回 nullptr
  }

  // Free a T previously allocated via AllocAligned<T>::alloc()
  // 释放之前由 AllocAligned<T>::alloc() 分配的对象 T

  static void release(T* p) {
    if (p) {
      p->~T();
      // 调用对象 T 的析构函数

#if defined(_MSC_VER)
      _aligned_free((void*)p);
      // 如果是在 MSVC 平台，使用 _aligned_free 释放内存
#else
      free((void*)p);
      // 在其他平台使用 free 释放内存
#endif
    }
  }
};

// Deleter object for unique_ptr for an aligned object
// 用于对齐对象的 unique_ptr 的删除器对象

template <typename T>
struct AlignedDeleter {
  void operator()(T* p) const { AllocAligned<T>::release(p); }
  // 重载函数调用操作符，调用 AllocAligned<T>::release(p) 释放对象
};

// make_unique that guarantees alignment
// 确保对齐的 make_unique 函数模板

template <typename T>
struct MakeAligned {
  template <typename... Args>
  static std::unique_ptr<T, AlignedDeleter<T>> make(Args&&... args) {
    return std::unique_ptr<T, AlignedDeleter<T>>(
        AllocAligned<T>::alloc(std::forward<Args>(args)...));
    // 使用 AllocAligned<T>::alloc 进行对齐内存分配，返回对齐的 unique_ptr
  }
};

const int kMaxBusyWaitNOPs = 32 * 1000 * 1000;
// 定义最大的忙等待操作数

#if defined(_MSC_VER)
#define GEMMLOWP_NOP __nop();
// 如果是在 MSVC 平台，定义 GEMMLOWP_NOP 为 __nop() 汇编指令
#else
#define GEMMLOWP_NOP "nop\n"
// 在其他平台，定义 GEMMLOWP_NOP 为 "nop\n" 汇编指令
#endif

#define GEMMLOWP_STRING_CONCAT_4(X) X X X X
// 定义一个宏，将参数 X 连续拼接四次

#define GEMMLOWP_NOP4 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP)
#define GEMMLOWP_NOP16 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP4)
#define GEMMLOWP_NOP64 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP16)
// 定义多个宏，分别拼接 4、16、64 次的 nop 操作

inline int Do256NOPs() {
#if defined(_MSC_VER)
  GEMMLOWP_NOP64;
#else
  asm volatile(GEMMLOWP_NOP64);
#endif
  return 64;
}
// 定义一个函数，执行 256 次 nop 操作

#undef GEMMLOWP_STRING_CONCAT_4
#undef GEMMLOWP_NOP256
#undef GEMMLOWP_NOP64
#undef GEMMLOWP_NOP16
#undef GEMMLOWP_NOP4
#undef GEMMLOWP_NOP
// 取消定义之前定义的多个宏

// Waits until *var != initial_value.
// 等待 *var 不等于初始值 initial_value

// Returns the new value of *var. The guarantee here is that
// the return value is different from initial_value, and that that
// new value has been taken by *var at some point during the
// execution of this function. There is no guarantee that this is
// 等待 *var 变化，并返回 *var 的新值。保证返回值与 initial_value 不同，
// 并且在函数执行过程中 *var 的新值已经被获取。不能保证这一点
// still the value of *var when this function returns, since *var is
// not assumed to be guarded by any lock.
//
// First does some busy-waiting for a fixed number of no-op cycles,
// then falls back to passive waiting for the given condvar, guarded
// by the given mutex.
//
// The idea of doing some initial busy-waiting is to help get
// better and more consistent multithreading benefits for small GEMM sizes.
// Busy-waiting help ensuring that if we need to wake up soon after having
// started waiting, then we can wake up quickly (as opposed to, say,
// having to wait to be scheduled again by the OS). On the other hand,
// we must still eventually revert to passive waiting for longer waits
// (e.g. worker threads having finished a GEMM and waiting until the next GEMM)
// so as to avoid permanently spinning.
//
template <typename T>
T WaitForVariableChange(std::atomic<T>* var,
                        T initial_value,
                        std::condition_variable* cond,
                        std::mutex* mutex) {
  // If we are on a platform that supports it, spin for some time.
  {
    int nops = 0;
    // First, trivial case where the variable already changed value.
    T new_value = var->load(std::memory_order_relaxed);
    if (new_value != initial_value) {
      std::atomic_thread_fence(std::memory_order_acquire);
      return new_value;
    }
    // Then try busy-waiting.
    while (nops < kMaxBusyWaitNOPs) {
      nops += Do256NOPs();  // Perform 256 no-op operations
      new_value = var->load(std::memory_order_relaxed);
      if (new_value != initial_value) {
        std::atomic_thread_fence(std::memory_order_acquire);
        return new_value;
      }
    }
  }

  // Finally, do real passive waiting.
  {
    std::unique_lock<std::mutex> g(*mutex);  // Acquire a unique lock on the mutex
    T new_value = var->load(std::memory_order_relaxed);
    // Handle spurious wakeups.
    cond->wait(g, [&]() {  // Wait on the condition variable, releasing the mutex until awakened
      new_value = var->load(std::memory_order_relaxed);
      return new_value != initial_value;
    });
    TORCH_DCHECK_NE(static_cast<size_t>(new_value), static_cast<size_t>(initial_value));  // Check that the new value is not equal to the initial value
    return new_value;
  }
}

// A BlockingCounter lets one thread to wait for N events to occur.
// This is how the master thread waits for all the worker threads
// to have finished working.
class BlockingCounter {
 public:
  // Sets/resets the counter; initial_count is the number of
  // decrementing events that the Wait() call will be waiting for.
  void Reset(std::size_t initial_count) {
    std::lock_guard<std::mutex> g(mutex_);  // Acquire a lock guard on the mutex
    TORCH_DCHECK_EQ(count_, 0);  // Ensure that the count is initially zero
    count_ = initial_count;  // Set the count to the initial value
  }

  // Decrements the counter; if the counter hits zero, signals
  // the thread that was waiting for that, and returns true.
  // Otherwise (if the decremented count is still nonzero),
  // returns false.
  bool DecrementCount() {
    const auto count_value = count_.fetch_sub(1, std::memory_order_relaxed) - 1;  // Atomically decrement the counter
    TORCH_DCHECK_GE(count_value, 0);  // Ensure that the decremented count is non-negative
    if (count_value == 0) {
      return true;  // Return true if the counter hits zero
    }
    return false;  // Otherwise, return false
  }
  
 private:
  std::mutex mutex_;  // Mutex for synchronization
  std::size_t count_ = 0;  // Counter variable
};
    // 如果计数器值为零，则通知一个等待中的线程
    if (count_value == 0) {
      // 使用互斥锁保护临界区域
      std::lock_guard<std::mutex> g(mutex_);
      // 发送通知给一个等待中的线程
      cond_.notify_one();
    }
    // 返回当前计数器值是否为零的布尔结果
    bool retval = count_value == 0;
    // 返回布尔结果
    return retval;
  }

  // 等待其他 N 个线程（N 由 Reset() 设置）到达阻塞计数器
  void Wait() {
    // 循环等待，直到计数器值为零
    while (size_t count_value = count_.load(std::memory_order_relaxed)) {
      // 等待条件变量的信号，阻塞直到条件满足
      WaitForVariableChange(&count_, count_value, &cond_, &mutex_);
    }
  }

 private:
  // 条件变量，用于线程同步
  std::condition_variable cond_;
  // 互斥锁，保护共享资源
  std::mutex mutex_;
  // 原子计数器，用于线程安全地更新计数
  std::atomic<std::size_t> count_{0};
};

// A workload for a worker.
struct Task {
  Task() = default;  // 默认构造函数
  virtual ~Task() = default;  // 虚析构函数
  virtual void Run() = 0;  // 纯虚函数，用于执行任务
};

// A worker thread.
class alignas(kGEMMLOWPCacheLineSize) Worker {
 public:
  enum class State : uint8_t {
    ThreadStartup, // 线程启动前的初始状态
    Ready, // 空闲状态，等待分配新的工作
    HasWork, // 有工作要执行
    ExitAsSoonAsPossible // 尽快退出的状态
  };

  explicit Worker(BlockingCounter* counter_to_decrement_when_ready)
      : task_(nullptr),
        state_(State::ThreadStartup),
        counter_to_decrement_when_ready_(counter_to_decrement_when_ready) {
    // 创建线程并启动，线程函数为 Worker::ThreadFunc()
    thread_ = std::make_unique<std::thread>([this]() { this->ThreadFunc(); });
  }

  ~Worker() {
    ChangeState(State::ExitAsSoonAsPossible);  // 在析构函数中请求退出
    thread_->join();  // 等待线程结束
  }

  // 修改状态的方法，可以由工作线程或主线程调用，使用互斥锁保护
  void ChangeState(State new_state) {
    std::lock_guard<std::mutex> g(state_mutex_);
    DCHECK(new_state != state_.load(std::memory_order_relaxed));
    switch (state_.load(std::memory_order_relaxed)) {
    case State::ThreadStartup:
      DCHECK(new_state == State::Ready);
      break;
    case State::Ready:
      DCHECK(new_state == State::HasWork || new_state == State::ExitAsSoonAsPossible);
      break;
    case State::HasWork:
      DCHECK(new_state == State::Ready || new_state == State::ExitAsSoonAsPossible);
      break;
    default:
      abort();
    }
    state_.store(new_state, std::memory_order_relaxed);  // 更新状态
    state_cond_.notify_one();  // 通知等待中的线程状态变化
    if (new_state == State::Ready) {
      counter_to_decrement_when_ready_->DecrementCount();  // 如果切换到 Ready 状态，递减计数器
    }
  }

  // 线程的入口函数
  void ThreadFunc() {
    c10::setThreadName("CaffeWorkersPool");  // 设置线程名称
    ChangeState(State::Ready);  // 初始状态设为 Ready

    // 线程主循环
    while (true) {
      // 获取要处理的状态
      // 在 'Ready' 状态下，什么也不做，只等待状态切换
      State state_to_act_upon =
          WaitForVariableChange(&state_, State::Ready, &state_cond_, &state_mutex_);

      // 现在有状态要处理，执行相应操作
      switch (state_to_act_upon) {
      case State::HasWork:
        // 有工作要做！执行任务并切换回 'Ready' 状态
        DCHECK(task_.load());
        (*task_).Run();
        task_ = nullptr;
        ChangeState(State::Ready);
        break;
      case State::ExitAsSoonAsPossible:
        return;  // 退出线程函数
      default:
        abort();
      }
    }
  }

  static void* ThreadFunc(void* arg) {
    static_cast<Worker*>(arg)->ThreadFunc();  // 静态函数入口，转发到实例方法
    return nullptr;
  }

  // 主线程调用，给该工作线程分配任务
  void StartWork(Task* task) {
    DCHECK(!task_.load());
    task_ = task;  // 分配任务
    DCHECK(state_.load(std::memory_order_acquire) == State::Ready);  // 只能在 Ready 状态下分配任务
    // 切换状态为 HasWork（有工作状态）
    ChangeState(State::HasWork);
  }

 private:
  // 线程的智能指针
  std::unique_ptr<std::thread> thread_;

  // 要处理的任务，使用原子指针以确保多线程安全
  std::atomic<Task*> task_;

  // 状态变化的条件变量，与状态互斥锁一起使用来保护状态更改
  std::condition_variable state_cond_;
  std::mutex state_mutex_;

  // 枚举状态，指示当前工作状态，如工作中、等待工作等
  std::atomic<State> state_;

  // 指向主线程的 BlockingCounter 对象的指针，用于在此工作线程切换到 'Ready' 状态时通知主线程
  BlockingCounter* const counter_to_decrement_when_ready_;
};

// WorkersPool 类的定义开始
class WorkersPool {
 public:
  // 默认构造函数
  WorkersPool() = default;

  // 执行给定任务的方法
  void Execute(const std::vector<std::shared_ptr<Task>>& tasks) {
    // 确保任务列表至少有一个任务
    CAFFE_ENFORCE_GE(tasks.size(), 1);
    
    // 计算工作线程数（除了当前线程外的任务数）
    int workers_count = tasks.size() - 1;
    
    // 创建工作线程池
    CreateWorkers(workers_count);
    
    // 断言确保工作线程数不超过实际可用工作线程数
    TORCH_DCHECK_LE(workers_count, (int)workers_.size());
    
    // 初始化一个计数器，用于等待工作线程完成
    counter_to_decrement_when_ready_.Reset(workers_count);
    
    // 启动除第一个任务外的所有任务
    for (const auto task : c10::irange(1, tasks.size())) {
      workers_[task - 1]->StartWork(tasks[task].get());
    }
    
    // 立即在当前线程上执行第一个任务的工作
    auto& task = tasks.front();
    task->Run();
    
    // 等待上述提交的工作线程完成
    counter_to_decrement_when_ready_.Wait();
  }

 private:
  // 确保工作线程池至少有给定数量的工作线程
  // 如果需要创建新的工作线程，此函数会等待直到新线程就绪
  void CreateWorkers(std::size_t workers_count) {
    if (workers_.size() >= workers_count) {
      return;
    }
    
    // 重置等待计数器以等待新工作线程就绪
    counter_to_decrement_when_ready_.Reset(workers_count - workers_.size());
    
    // 创建新的工作线程直到达到指定数量
    while (workers_.size() < workers_count) {
      workers_.push_back(MakeAligned<Worker>::make(&counter_to_decrement_when_ready_));
    }
    
    // 等待所有新创建的工作线程就绪
    counter_to_decrement_when_ready_.Wait();
  }

  // 禁用复制和赋值操作
  C10_DISABLE_COPY_AND_ASSIGN(WorkersPool);

  // 存储工作线程的列表
  std::vector<std::unique_ptr<Worker, AlignedDeleter<Worker>>> workers_;
  
  // 用于等待工作线程完成的计数器
  BlockingCounter counter_to_decrement_when_ready_;
};
// WorkersPool 类的定义结束
} // namespace caffe2
```