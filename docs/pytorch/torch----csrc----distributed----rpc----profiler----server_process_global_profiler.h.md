# `.\pytorch\torch\csrc\distributed\rpc\profiler\server_process_global_profiler.h`

```
#pragma once

#include <shared_mutex>  // 包含 C++ 标准库中的 shared_mutex 头文件

#include <torch/csrc/autograd/profiler.h>  // 包含 PyTorch 的自动微分模块中的 profiler 头文件

namespace torch {
namespace distributed {
namespace rpc {
namespace profiler {
namespace processglobal {

using namespace torch::autograd::profiler;  // 使用 PyTorch 自动微分模块中的 profiler 命名空间

// Process global profiler state.
//
// This class holds information about a profiling range, from "enable" to
// "disable".
// An instance of this ``State`` will be
// pushed into a global stack, so nested profiling range is supported.
//
// It has 2 members.
// One is ``autograd::profiler::ProfilerConfig``. It's set by user and
// will be copied to thread-local profiler state of RPC threads.
// The other is a container that aggregates recorded
// ``autograd::profiler::Event``s from all thread-local profilers on RPC
// threads.
class State {
 public:
  explicit State(const ProfilerConfig& config) : config_(config) {}  // 构造函数，接受一个 ProfilerConfig 对象的引用，初始化 config_ 成员

  ~State() = default;  // 默认析构函数

  const ProfilerConfig& config() const {  // 返回当前状态的 ProfilerConfig 对象的引用
    return config_;
  }

  void pushResult(thread_event_lists result) {  // 将一个 thread_event_lists 对象推入结果容器中
    std::unique_lock<std::mutex> lock(resultsMutex_);  // 获取互斥锁以操作 results_

    // NB: When a thread wants to push an entry into the this container,
    // main control logic might have exited the process-global profile range.
    results_.emplace_back(std::move(result));  // 将 result 移动到 results_ 容器的末尾
  }

  std::vector<thread_event_lists> results();  // 声明 results 函数，待实现

 private:
  // Each result comes from a profile range. In each profile range, there is a
  // "__profiler_start" marker event that all following events calculate time
  // relative to it, so it's required to call
  // parse_cpu_trace(result) for results of all profile range.
  std::mutex resultsMutex_;  // 互斥锁，用于保护对 results_ 容器的并发访问
  std::vector<thread_event_lists> results_;  // 存储 thread_event_lists 对象的容器
  const ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled);  // 成员变量，存储当前状态的 ProfilerConfig 对象
};

class StateStackEntry;

#if defined(__MACH__)
// Compiler error: 'shared_timed_mutex' is unavailable: introduced in
// macOS 10.12
using mutexType = std::mutex;  // macOS 上的互斥类型为 std::mutex
// Compiler error: 'shared_lock' is unavailable: introduced in
// macOS 10.12
using rLockType = std::unique_lock<std::mutex>;  // macOS 上的读取锁类型为 std::unique_lock<std::mutex>
using wLockType = std::unique_lock<std::mutex>;  // macOS 上的写入锁类型为 std::unique_lock<std::mutex>
#else
using mutexType = std::shared_timed_mutex;  // 其他系统上的互斥类型为 std::shared_timed_mutex
using rLockType = std::shared_lock<std::shared_timed_mutex>;  // 其他系统上的读取锁类型为 std::shared_lock<std::shared_timed_mutex>
using wLockType = std::unique_lock<std::shared_timed_mutex>;  // 其他系统上的写入锁类型为 std::unique_lock<std::shared_timed_mutex>
#endif

// This is the global stack of ``State``s.
TORCH_API extern std::shared_ptr<StateStackEntry> currentStateStackEntryPtr;  // 当前状态堆栈的全局指针
TORCH_API extern mutexType currentStateStackEntryMutex;  // 当前状态堆栈的全局互斥锁

// This class is used to implement a stack of ``State``s.
// It has 2 members.
// One is `prevPtr`, a shared_ptr pointing to previous element in the
// stack.
// The other is ``statePtr``, a shared_ptr pointing to ``State``.
class StateStackEntry {
 public:
  StateStackEntry(
      std::shared_ptr<StateStackEntry> prevPtr,
      std::shared_ptr<State> statePtr)
      : prevPtr_(std::move(prevPtr)), statePtr_(std::move(statePtr)) {}

  // 静态方法：将给定的 State 对象推入状态堆栈范围
  static void pushRange(std::shared_ptr<State> profilerProcessGlobalStatePtr);
  
  // 静态方法：弹出当前状态堆栈范围的 State 对象
  static std::shared_ptr<State> popRange();

  // 静态方法：获取当前状态堆栈的顶部条目
  static std::shared_ptr<StateStackEntry> current() {
    // 使用读锁保护访问当前状态堆栈的顶部条目
    rLockType rlock(currentStateStackEntryMutex);

    return currentStateStackEntryPtr;
  }

  // 返回前一个 StateStackEntry 的共享指针
  std::shared_ptr<StateStackEntry> prevPtr() const {
    return prevPtr_;
  }

  // 返回当前 StateStackEntry 持有的 State 对象的共享指针
  std::shared_ptr<State> statePtr() const {
    return statePtr_;
  }

 private:
  const std::shared_ptr<StateStackEntry> prevPtr_{nullptr}; // 前一个 StateStackEntry 的共享指针，初始化为 nullptr
  const std::shared_ptr<State> statePtr_{nullptr}; // 当前 StateStackEntry 持有的 State 对象的共享指针，初始化为 nullptr
};

// 将结果推送到当前分析范围内的 State 对象，并递归外部分析范围
TORCH_API void pushResultRecursive(
    std::shared_ptr<StateStackEntry> stateStackEntryPtr,
    const thread_event_lists& result);

// 用户可见的 API
//
// 进入服务端全局进程范围的分析范围
// 分析范围可以嵌套，因此可以多次调用此 API。这允许所有 RPC 线程运行服务器端请求回调。
TORCH_API void enableServer(const ProfilerConfig& new_config);
//
// 退出服务端全局进程范围的分析范围
// 分析范围可以嵌套，因此调用此 API 后，分析器可能仍处于启用状态。
// 这允许所有 RPC 线程运行服务器端请求回调。
TORCH_API std::vector<thread_event_lists> disableServer();

} // namespace processglobal
} // namespace profiler
} // namespace rpc
} // namespace distributed
} // namespace torch
```