# `.\pytorch\torch\csrc\distributed\rpc\profiler\server_process_global_profiler.cpp`

```py
// 包含头文件 <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>
#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>

// 命名空间 torch::distributed::rpc::profiler::processglobal
namespace torch {
namespace distributed {
namespace rpc {
namespace profiler {
namespace processglobal {

// 使用 torch::autograd::profiler 命名空间
using namespace torch::autograd::profiler;

// 定义 State 类中的 results 函数，返回当前结果
std::vector<thread_event_lists> State::results() {
  // 锁定 resultsMutex_，确保线程安全
  std::unique_lock<std::mutex> lock(resultsMutex_);

  // 创建副本并交换数据，保证结果原子性
  std::vector<thread_event_lists> results;
  results.swap(results_);
  return results;
}

// 定义 currentStateStackEntryMutex，用于保护共享数据 currentStateStackEntryPtr
mutexType currentStateStackEntryMutex;
// 初始化 currentStateStackEntryPtr 为空指针
std::shared_ptr<StateStackEntry> currentStateStackEntryPtr = nullptr;

// StateStackEntry 类的 pushRange 方法实现
void StateStackEntry::pushRange(
    std::shared_ptr<State> profilerProcessGlobalStatePtr) {
  // 加写锁 currentStateStackEntryMutex，保护共享数据
  wLockType wlock(currentStateStackEntryMutex);

  // 将当前 currentStateStackEntryPtr 设置为前一状态的栈入口
  auto previousStateStackEntryPtr = currentStateStackEntryPtr;
  // 创建新的 StateStackEntry，并设置为当前状态的栈入口
  currentStateStackEntryPtr = std::make_shared<StateStackEntry>(
      previousStateStackEntryPtr, std::move(profilerProcessGlobalStatePtr));
}

// StateStackEntry 类的 popRange 方法实现
std::shared_ptr<State> StateStackEntry::popRange() {
  // 加写锁 currentStateStackEntryMutex，保护共享数据
  wLockType wlock(currentStateStackEntryMutex);

  // 弹出当前 currentStateStackEntryPtr
  auto poppedStateStackEntryPtr = currentStateStackEntryPtr;
  // 断言 poppedStateStackEntryPtr 和其状态指针 statePtr_ 非空
  TORCH_INTERNAL_ASSERT(
      poppedStateStackEntryPtr && poppedStateStackEntryPtr->statePtr_);
  // 将当前 currentStateStackEntryPtr 设置为上一个状态的栈入口
  currentStateStackEntryPtr = poppedStateStackEntryPtr->prevPtr_;
  // 返回弹出状态的状态指针 statePtr_
  return poppedStateStackEntryPtr->statePtr_;
}

// 递归推送结果到 process-global 分析器状态的方法
void pushResultRecursive(
    std::shared_ptr<StateStackEntry> stateStackEntryPtr,
    const thread_event_lists& result) {
  // 当前状态栈入口存在时，逐层推送事件列表到 process-global 分析器状态
  while (stateStackEntryPtr) {
    stateStackEntryPtr->statePtr()->pushResult(result);
    stateStackEntryPtr = stateStackEntryPtr->prevPtr();
  }
}

// 启用服务器端的方法，配置为新的分析器配置
void enableServer(const ProfilerConfig& new_config) {
  // 创建新的分析器状态，并推入状态栈入口
  auto new_state = std::make_shared<State>(new_config);
  StateStackEntry::pushRange(std::move(new_state));
}

// 禁用服务器端的方法，从状态栈中弹出状态并返回结果
std::vector<thread_event_lists> disableServer() {
  // 弹出状态栈中的顶部状态指针，并返回其结果
  auto statePtr = StateStackEntry::popRange();
  return statePtr->results();
}

} // namespace processglobal
} // namespace profiler
} // namespace rpc
} // namespace distributed
} // namespace torch
```