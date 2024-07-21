# `.\pytorch\torch\csrc\autograd\profiler_legacy.cpp`

```
// 包含 Torch 的自动微分模块中的性能分析器的传统版本头文件

#include <torch/csrc/autograd/profiler_legacy.h>

// 包含 Torch 的自动微分功能模块的相关头文件
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>

// 包含 ATen 库的代码模板和操作注册相关头文件
#include <ATen/code_template.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

// 包含标准库的文件操作、互斥锁、字符串和向量相关头文件
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// 包含 ATen 库的性能记录函数、内存分配器和近似时钟相关头文件
#include <ATen/record_function.h>
#include <c10/core/Allocator.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <c10/util/irange.h>

// 包含标准输入输出流库的头文件
#include <iostream>

// 定义 torch::autograd::profiler 命名空间
namespace torch::autograd::profiler {

// 将性能分析器的逻辑分解为以下组件：

// ThreadLocalDebugInfo:
//
// ThreadLocalDebugInfo 是一个线程本地的映射，将槽位映射到调试信息结构体。
// ThreadLocalDebugInfo 会自动在线程边界传播，包括以下情况：
//  - 使用 at::launch 启动异步任务
//  - 执行 JIT continuation
//  - 从前向线程切换到自动微分（后向）线程
//
// ThreadLocalDebugInfo 中的条目由 DebugInfoGuard 管理，
// 可用于添加或覆盖线程本地映射中的条目。当销毁 guard 时，相应的条目会被移除，
// 可能会显示之前设置的相同槽位的值。
//
// 对于异步任务，主线程在启动异步任务之前设置的槽位在异步任务中是共享和可见的。
//
// 另一方面，异步任务对映射的添加或覆盖对主线程不可见，
// 并且主线程的任何修改（包括删除条目）在任务启动后不会对异步任务可见。
//
// 我们使用 ThreadLocalDebugInfo（槽位 PROFILER_STATE）来存储性能分析器的配置，
// 以及在性能分析期间发生的事件列表。
// 每次进入性能分析器（例如进入性能分析上下文管理器/调用 enableConfig）时都会创建 ThreadLocalDebugInfo 的实例，
// 并唯一标识一个性能分析运行。
//
// 我们自动将 ThreadLocalDebugInfo 传播到异步任务，
// 以及通过 JIT continuation 和自动微分线程，因此在开始和结束性能分析之间的所有操作
// （不一定在同一个线程内）都会被记录。
// 除非像嵌套性能分析范围中覆盖性能分析槽位那样（在这种情况下，子范围的事件由嵌套分析器处理）。
//
// 当我们退出性能分析范围（无论是退出性能分析上下文管理器还是调用 disableProfiler）时，
// 我们会移除给定线程本地映射中先前设置的性能分析条目，并整理性能分析结果中的事件。
//
//
// ThreadLocalState:
//
// ThreadLocalState 使用提供的 getter 获取线程本地变量的“快照”。
// 它与 ThreadLocalStateGuard 一起使用，
// 在修改或访问线程本地变量时提供了线程安全的管理和访问机制。
//
// to transfer the snapshot across thread boundary and set the thread local
// values as in the parent task.
//
// Profiler uses ThreadLocalState to propagate profiler's thread local state.
// ThreadLocalState also automatically propagates profiler callbacks.
//
//
// at::RecordFunction and observers
//
// Profiler uses observers mechanism to add a pair of thread local callbacks
// that are executed on a number of predetermined ranges, including:
//  - c10/ATen ops
//  - TorchScript functions/methods
//  - user defined named ranges (see `record_function` python context manager)
//
// Profiler setups a pair of callbacks that record profiling events and save
// them into the thread local profiler struct (ThreadLocalDebugInfo,
// PROFILER_STATE slot)
//
//
// Thus, the overall logic is:
//
// enableProfiler:
//  - checks that profiler is not enabled (otherwise throws)
//  - pushes new ThreadLocalDebugInfo (slot PROFILER_STATE) as the profiler
//    config for the current thread
//  - pushes profiling callbacks for the current thread
//
// disableProfiler:
//  - pops PROFILER_STATE slot from the current ThreadLocalDebugInfo and
//    consolidates events
//  - removes profiling callbacks
//
// ThreadLocalState:
//  - propagates ThreadLocalDebugInfo across threads
//  - propagates profiler callbacks across threads
//
// Profiler callbacks:
//  - get the current profiling state (PROFILER slot in ThreadLocalDebugInfo)
//  - save profiling events into the profiling state
//

namespace {
using torch::profiler::impl::ActiveProfilerType;
using torch::profiler::impl::ProfilerStateBase;

// Definition of a thread-local state for legacy profiler configurations
struct ProfilerLegacyThreadLocalState : public ProfilerStateBase {
  explicit ProfilerLegacyThreadLocalState(
      const torch::profiler::impl::ProfilerConfig& config)
      : ProfilerStateBase(config), remoteProfiledEvents_{c10::nullopt} {}
  ~ProfilerLegacyThreadLocalState() override = default;

  // Retrieve the thread-local instance of ProfilerLegacyThreadLocalState
  static ProfilerLegacyThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::LEGACY);
    return static_cast<ProfilerLegacyThreadLocalState*>(tls);
  }

  // Consolidate thread events into a unified structure
  thread_event_lists consolidate();

  // Mark the beginning or end of a profiling range with optional CUDA inclusion
  void mark(std::string name, bool include_cuda = true);

  // Set or add remote profiled events received from another source
  void setOrAddRemoteProfiledEvents(
      std::vector<LegacyEvent>&& remoteProfiledEvents);

  // Push a new profiling range using at::RecordFunction
  void pushRange(
      const at::RecordFunction& fn,
      const bool record_cuda,
      std::vector<std::vector<int64_t>>&& shapes = {});

  // Pop the current profiling range
  void popRange(const at::RecordFunction& fn, const bool record_cuda);

  // Report memory usage metrics for legacy profiler
  void reportMemoryUsage(
      void* /* unused */,
      int64_t alloc_size,
      size_t /* total_allocated, unused for legacy */,
      size_t /* total_reserved, unused for legacy */,
      c10::Device device) override;

  // Return the type of profiler (always LEGACY for this implementation)
  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::LEGACY;
  }

  // Handle any necessary cleanup when this object is no longer needed
  void leakHandle() {
    // 初始化一个名为 handle_ 的整数变量，赋值为 0
    handle_ = 0;
  }

 protected:
  // 返回与给定线程 ID 关联的事件列表的引用，如果未指定线程 ID，则返回主列表
  RangeEventList& getEventList(
      std::optional<uint64_t> thread_id = std::nullopt);

  // 状态互斥锁，用于保护状态的并发访问
  std::mutex state_mutex_;

  // 映射线程 ID 到其对应的事件列表的无序映射表
  std::unordered_map<uint64_t, std::shared_ptr<RangeEventList>>
      event_lists_map_;

  // 可选的远程分析事件数据的向量的向量
  std::optional<std::vector<std::vector<LegacyEvent>>> remoteProfiledEvents_;
};

// 函数定义：consolidate()，用于合并线程事件列表
thread_event_lists ProfilerLegacyThreadLocalState::consolidate() {
  // 使用互斥锁保护状态访问
  std::lock_guard<std::mutex> g(state_mutex_);
  // 创建存放合并后事件列表的结果对象
  thread_event_lists result;
  // 遍历事件列表映射
  for (auto& kv : event_lists_map_) {
    auto& list = kv.second;
    // 合并当前事件列表，并将结果添加到结果对象中
    result.emplace_back(list->consolidate());
  }
  // 如果存在远程分析事件，则也合并到结果对象中
  if (remoteProfiledEvents_) {
    result.insert(
        result.end(),
        std::make_move_iterator(remoteProfiledEvents_->begin()),
        std::make_move_iterator(remoteProfiledEvents_->end()));
  }
  // 返回合并后的结果对象
  return result;
}

// 函数定义：mark()，用于标记事件
void ProfilerLegacyThreadLocalState::mark(std::string name, bool include_cuda) {
  // 如果分析器已禁用，则直接返回
  if (config_.disabled()) {
    return;
  }
  // 根据分析器状态选择标记方法
  if (config_.state == torch::profiler::impl::ProfilerState::NVTX) {
    // 使用 CUDA stubs 标记事件
    torch::profiler::impl::cudaStubs()->mark(name.c_str());
  } else {
    // 创建 LegacyEvent 对象，记录当前标记事件
    LegacyEvent evt(
        EventKind::Mark,
        at::StringView(std::move(name)),
        at::RecordFunction::currentThreadId(),
        include_cuda &&
            config_.state == torch::profiler::impl::ProfilerState::CUDA);
    // 设置事件节点 ID
    evt.setNodeId(at::RecordFunction::getDefaultNodeId());
    // 将事件记录到事件列表中
    getEventList().record(std::move(evt));
  }
}

// 函数定义：setOrAddRemoteProfiledEvents()，设置或添加远程分析事件
void ProfilerLegacyThreadLocalState::setOrAddRemoteProfiledEvents(
    std::vector<LegacyEvent>&& remoteProfiledEvents) {
  // 加锁以序列化多个回调线程的访问
  std::lock_guard<std::mutex> guard(state_mutex_);
  // 如果已经存在远程分析事件列表，则添加到现有列表中
  if (remoteProfiledEvents_) {
    (*remoteProfiledEvents_).emplace_back(remoteProfiledEvents);
  } else {
    // 否则，创建新的远程分析事件列表
    remoteProfiledEvents_ = {std::move(remoteProfiledEvents)};
  }
}

// 函数定义：pushRange()，推送范围事件
void ProfilerLegacyThreadLocalState::pushRange(
    const at::RecordFunction& fn,
    const bool record_cuda,
    std::vector<std::vector<int64_t>>&& shapes) {
  // 如果分析器已禁用，则直接返回
  if (config_.disabled()) {
    return;
  }
  // 根据分析器状态选择推送范围事件方法
  if (config_.state == torch::profiler::impl::ProfilerState::NVTX) {
    // 使用 CUDA stubs 推送范围事件
    torch::profiler::impl::cudaStubs()->rangePush(
        torch::profiler::impl::getNvtxStr(fn.name(), fn.seqNr(), shapes)
            .c_str());
  } else {
    // 创建 LegacyEvent 对象，记录推送范围事件
    LegacyEvent evt(
        EventKind::PushRange,
        at::StringView(std::string(fn.name())),
        at::RecordFunction::currentThreadId(),
        record_cuda,
        fn.handle(),
        std::move(shapes),
        at::RecordFunction::getDefaultNodeId(),
        fn.isAsync());
    // 设置事件序列号、前向线程 ID 和作用域
    evt.setSequenceNr(fn.seqNr());
    evt.setFwdThreadId(fn.forwardThreadId());
    evt.setScope((uint8_t)fn.scope());
    // 如果配置启用 FLOPS 计算，则设置额外参数和 FLOPS
    if (config_.with_flops) {
      evt.setExtraArgs(torch::profiler::impl::saveExtraArgs(fn));
      evt.setFlops(torch::profiler::impl::computeFlops(
          std::string(fn.name()), evt.extraArgs()));
    }

    // TODO: 将统一 BUILD_LITE_INTERPRETER 和 C10_MOBILE 两个宏的处理
    // 后向节点源范围对应于前向节点
    // TODO: 考虑使用 C++ 堆栈跟踪
    # 检查配置中是否包含堆栈信息，并且当前函数作用域不是反向函数记录
    if (config_.with_stack &&
        fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
      # 准备当前调用堆栈信息
      auto cs =
          torch::profiler::impl::prepareCallstack(jit::currentCallstack());
      # 如果当前调用堆栈信息为空，则准备 Python 调用堆栈信息作为备选
      if (cs.empty()) {
        cs = torch::profiler::impl::prepareCallstack(
            jit::tracer::pythonCallstack());
      }
      # 将堆栈信息转换成字符串，并设置到事件对象中
      evt.setStack(callstackStr(cs));
    }
#endif
// 结束条件编译指令，用于关闭先前的条件编译区段

getEventList().record(std::move(evt));
// 调用 getEventList() 函数获取事件列表的引用，并记录移动构造的 evt 对象

}

void ProfilerLegacyThreadLocalState::popRange(
    const at::RecordFunction& fn,
    const bool record_cuda) {
  // 如果配置为禁用状态，则直接返回
  if (config_.disabled()) {
    return;
  }
  // 如果配置的状态为 NVTX，则调用 CUDA 的范围弹出函数
  if (config_.state == torch::profiler::impl::ProfilerState::NVTX) {
    torch::profiler::impl::cudaStubs()->rangePop();
  } else {
    // 否则，创建 LegacyEvent 对象，表示 PopRange 事件
    // 这里可能在不同线程上调用 RecordFunction 和 popRange，
    // 按照约定，将异步弹出操作放在原始线程上，并在弹出事件中保存当前线程 ID
    LegacyEvent evt(
        EventKind::PopRange,
        at::StringView(""),
        at::RecordFunction::currentThreadId(),
        record_cuda,
        fn.handle());
    evt.setNodeId(at::RecordFunction::getDefaultNodeId());
    // 将事件记录到相应的事件列表中
    getEventList(fn.threadId()).record(std::move(evt));
  }
}

void ProfilerLegacyThreadLocalState::reportMemoryUsage(
    void* /* unused */,
    int64_t alloc_size,
    size_t /* total_allocated, unused for legacy */,
    size_t /* total_reserved, unused for legacy */,
    c10::Device device) {
  // 如果配置要求内存分析且未被禁用，则执行以下代码块
  if (config_.profile_memory && !config_.disabled()) {
    // 获取当前线程的线程 ID
    uint64_t thread_id = at::RecordFunction::currentThreadId();
    // 创建 MemoryAlloc 事件的 LegacyEvent 对象
    LegacyEvent evt(
        EventKind::MemoryAlloc,
        at::StringView(""),
        thread_id,
        config_.state == torch::profiler::impl::ProfilerState::CUDA);
    // 更新事件的内存统计信息
    evt.updateMemoryStats(alloc_size, device);
    // 将事件记录到相应的事件列表中
    getEventList(thread_id).record(std::move(evt));
  }
}

RangeEventList& ProfilerLegacyThreadLocalState::getEventList(
    std::optional<uint64_t> thread_id) {
  // 如果未提供线程 ID，则使用当前线程的线程 ID
  if (!thread_id.has_value()) {
    thread_id = at::RecordFunction::currentThreadId();
  }
  // 初始化事件列表指针为 nullptr
  RangeEventList* list_ptr = nullptr;
  // 使用互斥锁保护状态
  std::lock_guard<std::mutex> guard(state_mutex_);
  // 查找给定线程 ID 对应的事件列表
  auto it = event_lists_map_.find(thread_id.value());
  if (it != event_lists_map_.end()) {
    // 如果找到，则获取事件列表的指针
    list_ptr = it->second.get();
  } else {
    // 如果未找到，则创建一个新的事件列表并存储在映射中
    auto event_list = std::make_shared<RangeEventList>();
    event_lists_map_[thread_id.value()] = event_list;
    list_ptr = event_list.get();
  }
  // 返回事件列表的引用
  return *list_ptr;
}

enum EventIValueIdx {
  KIND = 0,
  NAME,
  THREAD_ID,
  HANDLE,
  NODE_ID,
  CPU_MEM_USAGE,
  CPU_NS,
  CUDA_RECORDED,
  CUDA_MEM_USAGE,
  CUDA_DEVICE,
  CUDA_US,
  SHAPES,
  NUM_EVENT_IVALUE_IDX // 必须是列表中的最后一个元素
};

const std::unordered_set<std::string> disable_cuda_profiling = {
    // 禁用 CUDA 分析的操作列表
    "aten::view",
    "aten::t",
    "aten::transpose",
    "aten::stride",
    "aten::empty",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::as_strided",
    "aten::expand",
    "aten::resize_",
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::slice",
    "aten::_unsafe_view",
    "aten::size"};
void pushProfilingCallbacksLegacy() {
  // 获取当前线程的旧版性能分析器状态指针
  auto registration_state_ptr = ProfilerLegacyThreadLocalState::getTLS();
  // 断言确保性能分析器状态指针不为空
  TORCH_INTERNAL_ASSERT(registration_state_ptr, "Expected profiler state set");
  // 注册线程本地回调函数
  auto handle = at::addThreadLocalCallback(
      // 创建记录函数回调，用于开始性能分析
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            // 获取当前线程的旧版性能分析器状态指针
            auto state_ptr = ProfilerLegacyThreadLocalState::getTLS();
            // 如果状态指针为空或者性能分析被禁用，则返回空
            if (!state_ptr || state_ptr->config().disabled()) {
              return nullptr;
            }
            // 检查是否记录 CUDA 操作
            bool record_cuda = state_ptr->config().state ==
                torch::profiler::impl::ProfilerState::CUDA;
            // 如果记录 CUDA 并且禁用了特定函数的 CUDA 分析，则取消记录 CUDA
            if (record_cuda &&
                disable_cuda_profiling.find(fn.name()) !=
                    disable_cuda_profiling.end()) {
              record_cuda = false;
            }

            // 如果配置要求报告输入形状，则获取输入大小
            if (state_ptr->config().report_input_shapes) {
              auto sizes = torch::profiler::impl::inputSizes(fn);
              // 推入性能分析范围，包括 CUDA 记录和输入大小
              state_ptr->pushRange(fn, record_cuda, std::move(sizes));
            } else {
              // 推入性能分析范围，仅包括 CUDA 记录
              state_ptr->pushRange(fn, record_cuda);
            }

            return nullptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext*) {
            // 获取当前线程的旧版性能分析器状态指针
            auto state_ptr = ProfilerLegacyThreadLocalState::getTLS();
            // 如果状态指针为空或者性能分析被禁用，则返回
            if (!state_ptr || state_ptr->config().disabled()) {
              return;
            }
            // 检查是否记录 CUDA 操作
            bool record_cuda = state_ptr->config().state ==
                torch::profiler::impl::ProfilerState::CUDA;
            // 如果记录 CUDA 并且禁用了特定函数的 CUDA 分析，则取消记录 CUDA
            if (record_cuda &&
                disable_cuda_profiling.find(fn.name()) !=
                    disable_cuda_profiling.end()) {
              record_cuda = false;
            }
            // 弹出性能分析范围，包括 CUDA 记录
            state_ptr->popRange(fn, record_cuda);
          })
          // 设置是否需要输入形状的报告
          .needsInputs(registration_state_ptr->config().report_input_shapes)
          // 设置需要标识符
          .needsIds(true));
  // 设置回调函数句柄到性能分析器状态中
  registration_state_ptr->setCallbackHandle(handle);
}

} // namespace

void enableProfilerLegacy(
    const torch::profiler::impl::ProfilerConfig& new_config) {
  // 检查是否可以使用 NVTX 性能分析器
  TORCH_CHECK(
      new_config.state != torch::profiler::impl::ProfilerState::NVTX ||
          torch::profiler::impl::cudaStubs()->enabled(),
      "Can't use NVTX profiler - PyTorch was compiled without CUDA");

  // 检查是否可以使用 KINETO 性能分析器
  TORCH_CHECK(new_config.state != torch::profiler::impl::ProfilerState::KINETO);

  // 获取当前线程的旧版性能分析器状态指针
  auto state_ptr = ProfilerLegacyThreadLocalState::getTLS();
  // 确保当前线程上没有启用性能分析器
  TORCH_CHECK(!state_ptr, "Profiler is already enabled on this thread");
  // 创建新的性能分析器状态对象并共享给线程调试信息
  auto state = std::make_shared<ProfilerLegacyThreadLocalState>(new_config);
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

  // 注册旧版性能分析器的回调函数
  pushProfilingCallbacksLegacy();

  // 标记性能分析开始点
  state->mark("__start_profile", false);
}

thread_event_lists disableProfilerLegacy(
    // 定义函数，禁用分析器选项可选，其中包含禁用TLS状态和合并选项
    std::optional<ProfilerDisableOptions> profilerDisableOptions) {
      // 如果给定了禁用分析器选项，则获取cleanupTLSState的值；否则默认为true
      auto cleanupTLSState =
          profilerDisableOptions ? profilerDisableOptions->cleanupTLSState : true;
      // 如果给定了禁用分析器选项，则获取consolidate的值；否则默认为true
      auto consolidate =
          profilerDisableOptions ? profilerDisableOptions->consolidate : true;
      // 所有的DebugInfoBase对象都是基于作用域的，应该使用DebugInfoGuard进行管理
      std::shared_ptr<c10::DebugInfoBase> state;
      // 如果需要清理TLS状态，则从TLS中弹出Profiler状态对象；否则仅查看不弹出
      if (cleanupTLSState) {
        state = c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE);
      } else {
        state =
            c10::ThreadLocalDebugInfo::_peek(c10::DebugInfoKind::PROFILER_STATE);
      }
    
      // 将state强制转换为ProfilerLegacyThreadLocalState指针
      auto state_ptr = static_cast<ProfilerLegacyThreadLocalState*>(state.get());
      // 检查state_ptr不为空且分析器配置未禁用，否则抛出异常
      TORCH_CHECK(
          state_ptr && !state_ptr->config().disabled(),
          "Can't disable profiler when it's not running");
    
      // 如果需要清理TLS状态，则调用removeCallback方法；否则调用leakHandle方法
      cleanupTLSState ? state_ptr->removeCallback() : state_ptr->leakHandle();
      // 如果不需要合并或者分析器状态为NVTX，则返回空的thread_event_lists()
      if (!consolidate ||
          state_ptr->config().state == torch::profiler::impl::ProfilerState::NVTX) {
        return thread_event_lists();
      }
    
      // 在Profiler状态对象中标记"__stop_profile"事件，返回合并后的事件列表
      state_ptr->mark("__stop_profile", false);
      // 注意，这会擦除底层事件
      return state_ptr->consolidate();
    }
}

// 添加事件列表到分析器的遗留事件中
void addEventList(std::vector<LegacyEvent>&& profiledEvents) {
  // 获取当前线程的遗留分析器状态指针
  auto state_ptr = ProfilerLegacyThreadLocalState::getTLS();
  // 检查分析器状态指针是否有效
  TORCH_CHECK(state_ptr, "Profiler must be enabled.");
  // 将传入的遗留事件列表设置或添加到远程分析器的遗留事件中
  state_ptr->setOrAddRemoteProfiledEvents(std::move(profiledEvents));
}

// 记录遗留事件，支持记录 CUDA 事件
void LegacyEvent::record(bool record_cuda) {
  // 如果需要记录 CUDA 事件
  if (record_cuda) {
    // 使用 CUDA 子系统记录事件
    torch::profiler::impl::cudaStubs()->record(&device_, &cuda_event, &cpu_ns_);
    return;
  }
  // 否则记录 CPU 时间戳
  cpu_ns_ = c10::getTime();
}

// 从 IValue 中重建 LegacyEvent 对象
/* static */ LegacyEvent LegacyEvent::fromIValue(
    const at::IValue& eventIValue) {
  // 断言确保传入的 IValue 是列表类型
  TORCH_INTERNAL_ASSERT(
      eventIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  // 获取列表中的所有元素
  auto ivalues = eventIValue.toList();
  // 断言确保至少有 NUM_EVENT_IVALUE_IDX 个元素用于重建 LegacyEvent
  TORCH_INTERNAL_ASSERT(
      ivalues.size() >= NUM_EVENT_IVALUE_IDX,
      "Expected at least ",
      NUM_EVENT_IVALUE_IDX,
      " elements to reconstruct LegacyEvent.");

  // 从 ivalues 中重建输入形状
  const auto& shapeListIValue = ivalues.get(EventIValueIdx::SHAPES);
  // 断言确保形状列表也是列表类型
  TORCH_INTERNAL_ASSERT(
      shapeListIValue.isList(),
      "Expected profiler shapes IValue to contain type c10::impl::GenericList.");

  // 获取形状列表中的形状并重建它们
  auto shapeList = shapeListIValue.toList();
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(shapeList.size());
  for (const auto i : c10::irange(shapeList.size())) {
    std::vector<int64_t> s;
    const auto& shapeIValue = shapeList.get(i);
    // 断言确保每个形状元素都是列表类型，包含形状
    TORCH_INTERNAL_ASSERT(
        shapeIValue.isList(),
        "Expected each profiler shape element to contain shapes of type c10::impl::GenericList.")
    auto curShapesList = shapeIValue.toList();
    s.reserve(curShapesList.size());
    for (const auto j : c10::irange(curShapesList.size())) {
      // 将每个形状元素转换为 int64_t 并添加到当前形状向量中
      s.emplace_back(curShapesList.get(j).toInt());
    }
    // 将当前形状向量添加到形状列表中
    shapes.emplace_back(s);
  }

  // 使用重建的数据构造 LegacyEvent 对象
  LegacyEvent evt(
      static_cast<EventKind>(
          ivalues.get(EventIValueIdx::KIND).toInt()), // 事件类型
      at::StringView(ivalues.get(EventIValueIdx::NAME).toStringRef()), // 名称
      ivalues.get(EventIValueIdx::THREAD_ID).toInt(), // 线程 ID
      static_cast<at::RecordFunctionHandle>(
          ivalues.get(EventIValueIdx::HANDLE).toDouble()), // 句柄
      std::move(shapes), // 输入形状
      ivalues.get(EventIValueIdx::NODE_ID).toInt(), // 节点 ID
      true, // 是否远程
      ivalues.get(EventIValueIdx::CPU_MEM_USAGE).toInt(), // CPU 内存使用
      ivalues.get(EventIValueIdx::CPU_NS).toInt(), // CPU 时间戳
      ivalues.get(EventIValueIdx::CUDA_RECORDED).toBool(), // 是否已记录 CUDA
      ivalues.get(EventIValueIdx::CUDA_MEM_USAGE).toInt(), // CUDA 内存使用
      c10::DeviceIndex(
          ivalues.get(EventIValueIdx::CUDA_DEVICE).toInt()), // CUDA 设备索引
      static_cast<double>(
          ivalues.get(EventIValueIdx::CUDA_US).toInt()) // CUDA 使用时间
  );
  return evt;
}
at::IValue LegacyEvent::toIValue() const {
  // 创建一个泛型列表 eventIValueList，元素类型为 at::AnyType
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  // 预留空间以容纳 NUM_EVENT_IVALUE_IDX 个元素
  eventIValueList.reserve(NUM_EVENT_IVALUE_IDX);
  // 将事件的种类 kind_ 转换为 int64_t 类型并加入列表
  eventIValueList.emplace_back(static_cast<int64_t>(kind_));
  // 将事件的名称 name_ 转换为 std::string 类型并加入列表
  eventIValueList.emplace_back(std::string(name_.str()));
  // 将事件的线程 ID thread_id_ 转换为 int64_t 类型并加入列表
  eventIValueList.emplace_back(static_cast<int64_t>(thread_id_));
  // 将事件的句柄 handle_ 转换为 double 类型并加入列表
  eventIValueList.emplace_back(static_cast<double>(handle_));
  // 将事件的节点 ID node_id_ 加入列表
  eventIValueList.emplace_back(node_id_);
  // 将事件的 CPU 内存使用量 cpu_memory_usage_ 加入列表
  eventIValueList.emplace_back(cpu_memory_usage_);
  // 将事件的 CPU 时间戳 cpu_ns_ 加入列表
  eventIValueList.emplace_back(cpu_ns_);
  // CUDA 事件信息
  bool cuda_profiling_enabled = hasCuda();
  // 将 CUDA 是否启用加入列表
  eventIValueList.emplace_back(cuda_profiling_enabled);
  // 将 CUDA 内存使用量 cuda_memory_usage_ 转换为 int64_t 类型并加入列表
  eventIValueList.emplace_back(static_cast<int64_t>(cuda_memory_usage_));
  // 将事件的设备 device_ 加入列表
  eventIValueList.emplace_back(device_);
  // 将事件的 CUDA 时间戳 cuda_us_ 加入列表
  eventIValueList.emplace_back(cuda_us_);
  
  // 形状信息
  // 创建一个泛型列表 shapesList，元素类型为 at::IntType 的列表
  c10::impl::GenericList shapesList =
      c10::impl::GenericList(at::ListType::create(at::IntType::get()));
  // 预留足够空间以容纳 shapes_ 中所有形状的列表
  shapesList.reserve(shapes_.size());
  // 遍历每个形状 shape
  for (const auto& shape : shapes_) {
    // 创建一个新的形状列表 s，元素类型为 at::IntType
    c10::impl::GenericList s = c10::impl::GenericList(at::IntType::get());
    // 预留足够空间以容纳当前形状 shape 的所有元素
    s.reserve(shape.size());
    // 将当前形状 shape 的每个元素加入列表 s
    for (const auto& k : shape) {
      s.emplace_back(k);
    }
    // 将形状列表 s 加入 shapesList
    shapesList.emplace_back(s);
  }
  // 将 shapesList 加入事件值列表 eventIValueList
  eventIValueList.emplace_back(shapesList);
  
  // 返回事件值列表 eventIValueList 封装成的 at::IValue 对象
  return at::IValue(eventIValueList);
}

double LegacyEvent::cudaElapsedUs(const LegacyEvent& e) const {
  // 检查两个事件是否都记录了 CUDA 信息
  TORCH_CHECK(e.hasCuda() && hasCuda(), "Events were not recorded for CUDA");
  // 检查两个事件是否在同一个设备上
  TORCH_CHECK(
      e.device() == device(),
      c10::str(
          "Events are not on the same device: ", e.device(), " vs ", device()));
  // 如果两个事件都是远程事件
  if (isRemote() && e.isRemote()) {
    // 验证 cuda_us_ 是否已正确设置
    TORCH_INTERNAL_ASSERT(cuda_us_ >= 0 && e.cuda_us_ >= 0);
    // 返回两个事件的 CUDA 时间戳差值
    return static_cast<double>(e.cuda_us_ - cuda_us_);
  }
  // 使用 CUDA 存根函数计算两个 CUDA 事件的时间差
  return torch::profiler::impl::cudaStubs()->elapsed(
      &cuda_event, &e.cuda_event);
}

static const at::jit::CodeTemplate event_template(R"(
{
  "name": "${name}",
  "ph": "X",
  "ts": ${ts},
  "dur": ${dur},
  "tid": ${tid},
  "pid": "CPU Functions",
  "args": {}
})");

void writeProfilerEventsToStream(
    std::ostream& out,
    const std::vector<LegacyEvent*>& events) {
  // 检查输出流是否有效
  TORCH_CHECK(out, "Could not open file");
  // 指向首个事件的指针
  LegacyEvent* profiler_start = nullptr;
  // 查找名为 "__start_profile" 的事件作为起始标记
  for (LegacyEvent* e : events) {
    if (0 == strcmp(e->name(), "__start_profile")) {
      profiler_start = e;
      break;
    }
  }
  // 检查是否找到 "__start_profile" 标记
  TORCH_CHECK(profiler_start, "Could not find __start_profile mark");

  // 定义一个用于哈希对的哈希函数对象
  struct PairHash {
    size_t operator()(
        std::pair<at::RecordFunctionHandle, int> p) const noexcept {
      return std::hash<at::RecordFunctionHandle>()(p.first) ^
          std::hash<int64_t>()(p.second);
    }
  };
  // 创建一个无序映射表 events_map，将事件对 (handle, tid) 映射到 LegacyEvent 指针
  std::unordered_map<
      std::pair<at::RecordFunctionHandle, int64_t>,
      LegacyEvent*,
      PairHash>
      events_map;
  // 将 JSON 数组开始符号写入输出流
  out << "[\n";
  // 是否为第一个事件标志
  bool first = true;
  // 遍历所有事件
  for (LegacyEvent* evt : events) {
    // 如果事件类型是 "push"
    if (evt->kindStr() == "push") {
      // 将事件的句柄和节点ID作为键，事件对象作为值存入events_map中
      events_map[std::make_pair(evt->handle(), evt->nodeId())] = evt;
    } else if (evt->kindStr() == "pop") {  // 如果事件类型是 "pop"
      // 如果不是第一次写入输出流，添加逗号和换行符
      if (!first) {
        out << ",\n";
      }
      // 将first标记设为false，表示已经不是第一次写入输出流
      first = false;
      // 查找events_map中与当前事件句柄和节点ID匹配的事件
      auto it = events_map.find(std::make_pair(evt->handle(), evt->nodeId()));
      // 检查是否找到匹配的事件，如果未找到，抛出异常信息 "Unmatched pop event"
      TORCH_CHECK(it != events_map.end(), "Unmatched pop event");
      // 获取与当前事件匹配的起始事件对象
      LegacyEvent* evt_start = it->second;
      // 从events_map中移除当前事件的条目
      events_map.erase(it);

      // 创建一个TemplateEnv对象env，用于构建输出模板所需的环境变量
      at::jit::TemplateEnv env;
      // 将事件起始点的名称添加到env的环境变量中
      env.s("name", evt_start->name());
      // 计算事件起始点到当前事件的CPU时间差，并添加到env的环境变量中
      env.d("ts", profiler_start->cpuElapsedUs(*evt_start));
      // 计算当前事件的CPU执行时间，并添加到env的环境变量中
      env.d("dur", evt_start->cpuElapsedUs(*evt));
      // 将事件起始点的线程ID添加到env的环境变量中
      env.d("tid", evt_start->threadId());
      // 使用env中的数据格式化event_template模板，并将结果写入输出流out中
      out << event_template.format(env);
    }
  }
  // 在输出流的末尾添加右括号，表示结束事件列表
  out << "]\n";
}

// RecordProfile 类的构造函数，接受一个输出流作为参数，初始化成员变量 out_
RecordProfile::RecordProfile(std::ostream& out) : out_(out) {
  // 调用初始化函数
  init();
}

// RecordProfile 类的构造函数，接受一个文件名作为参数，创建一个新的 ofstream 对象，并初始化成员变量 file_ 和 out_
RecordProfile::RecordProfile(const std::string& filename)
    : file_(new std::ofstream(filename)), out_(*file_) {
  // 调用初始化函数
  init();
}

// RecordProfile 类的初始化函数，启用旧版的性能分析器
void RecordProfile::init() {
  enableProfilerLegacy(torch::profiler::impl::ProfilerConfig(
      torch::profiler::impl::ProfilerState::CPU));
}

// RecordProfile 类的析构函数
RecordProfile::~RecordProfile() {
  try {
    // 禁用旧版的性能分析器，并获取事件列表
    thread_event_lists event_lists = disableProfilerLegacy();
    std::vector<LegacyEvent*> events;
    // 将所有事件从列表中提取到一个向量中
    for (auto& l : event_lists) {
      for (auto& e : l) {
        events.push_back(&e);
      }
    }
    // 处理提取的事件
    processEvents(events);
  } catch (const std::exception& e) {
    // 捕获异常并记录日志
    LOG(ERROR) << e.what() << '\n';
  } catch (...) {
    // 捕获未知异常并记录日志
    LOG(ERROR) << "Unknown error" << '\n';
  }
}

// 处理事件的函数，将事件写入到给定的输出流中
void RecordProfile::processEvents(const std::vector<LegacyEvent*>& events) {
  writeProfilerEventsToStream(out_, events);
}

// 命名空间结束符，结束 torch::autograd::profiler 命名空间的定义
} // namespace torch::autograd::profiler
```