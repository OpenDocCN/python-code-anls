# `.\pytorch\torch\csrc\autograd\profiler_kineto.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <string>
// 包含字符串操作相关的标准库头文件
#include <vector>
// 包含向量操作相关的标准库头文件

#include <torch/csrc/profiler/api.h>
// 引入 Torch 的性能分析 API 头文件
#include <torch/csrc/profiler/events.h>
// 引入 Torch 的性能分析事件头文件
#include <torch/csrc/profiler/stubs/base.h>
// 引入 Torch 的性能分析基本存根头文件
#include <torch/csrc/profiler/util.h>
// 引入 Torch 的性能分析实用工具头文件

namespace torch {
// 定义 Torch 命名空间

namespace profiler::impl {
// 定义性能分析实现的命名空间

struct Result;
// 声明 Result 结构体，具体实现未给出

namespace kineto {
// 定义 Kineto 模块的命名空间

struct ActivityTraceWrapper;
// 声明 ActivityTraceWrapper 结构体，具体实现未给出

} // namespace kineto

} // namespace profiler::impl

namespace autograd::profiler {
// 定义自动微分性能分析的命名空间

using experimental_event_t = std::shared_ptr<torch::profiler::impl::Result>;
// 使用 Torch 性能分析结果的共享指针作为实验性事件类型
using extra_meta_t = std::unordered_map<std::string, std::string>;
// 使用无序映射存储额外的元数据，键和值都是字符串类型

struct TORCH_API KinetoEvent {
// 定义 KinetoEvent 结构体，表示来自 Kineto 的事件

  KinetoEvent(
      const std::shared_ptr<const torch::profiler::impl::Result>&,
      const bool verbose);
  // 构造函数，接受性能分析结果的共享指针和详细标志位作为参数

  uint64_t startThreadId() const;
  // 返回开始线程 ID 的方法
  uint64_t endThreadId() const;
  // 返回结束线程 ID 的方法
  uint8_t activityType() const;
  // 返回活动类型的方法
  uint64_t fwdThreadId() const;
  // 返回前向线程 ID 的方法
  bool hasShapes() const;
  // 检查是否存在形状信息的方法
  const c10::ArrayRef<std::vector<int64_t>> shapes() const;
  // 返回形状信息数组的方法
  bool hasTypes() const;
  // 检查是否存在数据类型信息的方法
  const c10::ArrayRef<std::string> dtypes() const;
  // 返回数据类型信息数组的方法
  bool hasConcreteInputs() const;
  // 检查是否存在具体输入信息的方法
  const c10::ArrayRef<c10::IValue> concreteInputs() const;
  // 返回具体输入信息数组的方法
  uint64_t flops() const;
  // 返回浮点操作数的方法
  int64_t sequenceNr() const;
  // 返回序列号的方法
  bool hasStack() const;
  // 检查是否存在堆栈信息的方法
  const c10::ArrayRef<std::string> stack() const;
  // 返回堆栈信息数组的方法
  uint8_t scope() const;
  // 返回作用域的方法
  bool hasModuleHierarchy() const;
  // 检查是否存在模块层次信息的方法
  const c10::ArrayRef<std::string> moduleHierarchy() const;
  // 返回模块层次信息数组的方法
  int64_t debugHandle() const;
  // 返回调试句柄的方法
  std::string name() const;
  // 返回事件名称的方法
  c10::DeviceType deviceType() const;
  // 返回设备类型的方法
  int deviceIndex() const;
  // 返回设备索引的方法
  int64_t nBytes() const;
  // 返回字节数的方法
  uint64_t startNs() const;
  // 返回开始时间的纳秒表示方法
  uint64_t endNs() const;
  // 返回结束时间的纳秒表示方法
  uint64_t durationNs() const;
  // 返回持续时间的纳秒表示方法
  bool isAsync() const;
  // 检查事件是否异步的方法
  uint64_t correlationId() const;
  // 返回相关 ID 的方法
  uint64_t linkedCorrelationId() const;
  // 返回关联相关 ID 的方法
  int64_t deviceResourceId() const;
  // 返回设备资源 ID 的方法
  std::string backend() const;
  // 返回后端信息的方法
  bool isPythonFunction() const;
  // 检查事件是否为 Python 函数的方法
  int64_t cudaElapsedUs() const;
  // 返回 CUDA 执行时间的微秒表示方法
  int64_t privateuse1ElapsedUs() const;
  // 返回私有使用的执行时间的微秒表示方法
  void getPerfEventCounters(torch::profiler::perf_counters_t&) const;
  // 获取性能事件计数器的方法
  extra_meta_t extraMeta() const;
  // 返回额外元数据的方法

 private:
  torch::profiler::impl::ProfilerVoidEventStub fallbackStart() const;
  // 返回开始回退的方法
  torch::profiler::impl::ProfilerVoidEventStub fallbackEnd() const;
  // 返回结束回退的方法

  std::shared_ptr<const torch::profiler::impl::Result> result_;
  // 性能分析结果的共享指针
  std::vector<std::string> python_stack_;
  // Python 堆栈信息的字符串向量

  // 从结果复制字段以便返回 ArrayRef
  std::vector<std::vector<int64_t>> shapes_;
  // 形状信息的二维整数向量
  std::vector<std::string> dtypes_;
  // 数据类型信息的字符串向量
  std::vector<c10::IValue> concrete_inputs_;
  // 具体输入信息的 IValue 向量
};

// 用于合并从 Kineto 直接返回的事件
// 与我们手动创建的事件（如开始/停止标记、内存分配事件）
struct TORCH_API ProfilerResult {
// 定义 ProfilerResult 结构体

  ProfilerResult();
  // 默认构造函数

  ProfilerResult(
      uint64_t start_time,
      std::vector<KinetoEvent> events,
      std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>&&
          trace,
      std::vector<experimental_event_t>&& event_tree);
  // 构造函数，接受开始时间、事件向量、活动跟踪包装器和事件树作为参数

  ~ProfilerResult();
  // 析构函数

  uint64_t trace_start_ns() const {
    return trace_start_ns_;
  }
  // 返回跟踪开始时间的纳秒表示方法

  const std::vector<KinetoEvent>& events() const {
  // 返回事件向量的方法
    return events_;
  }



# 返回私有成员变量 events_ 的值
  const std::vector<experimental_event_t>& event_tree() const {
    # 返回私有成员变量 event_tree_ 的引用
    return event_tree_;
  }

  void save(const std::string& path);



# 保存对象的状态到指定路径，但具体实现未提供
 private:
  # 跟踪开始时间，以纳秒为单位，默认为 0
  uint64_t trace_start_ns_ = 0;
  # 存储事件的向量
  std::vector<KinetoEvent> events_;
  # 跟踪对象的唯一指针
  std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper> trace_;
  # 实验性事件的向量
  std::vector<experimental_event_t> event_tree_;
};

/*
`
};

/*
 * This API is used by backends to record latency of events that
 * happened in the backend but were not visible to pytorch runtime.
 * For example, if part of the model is lowered to a dsp backend, then
 * the execution of that part of the model is delegated to the backend.
 * When backend finishes execution it has an option to provide profiling
 * information (latency only at the moment) corresponding to different operators
 * that were executed in the backend.
 * When such events are recorded by backend using this API, the event
 * records will be collected by active kineto profiler. If no kineto profiler
 * is active then the event is ignored.
 * This provides us with a way to generate all the profiling information
 * for a model regardless of where model (or part of it) executed.
 * @param start_time_us: start time in us of the event
 * @param end_time_us: end time in us of the event
 * @param debug_handle: debug handle to correlate this event/op with
 * model level module/source information
 * @param scope: scope of the event, e.g. LITE_INTERPRETER, RECORD_FN etc.
 * @param event_name: name of the event, e.g. op name
 * @param backend_name: name of the backend where the event took place.
 */
TORCH_API void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,             // 事件开始时间，单位为微秒
    const int64_t end_time_us,               // 事件结束时间，单位为微秒
    const int64_t debug_handle,              // 调试句柄，用于将事件/操作与模型级模块/源信息关联
    const at::RecordScope scope,             // 事件的作用域，例如 LITE_INTERPRETER、RECORD_FN 等
    const std::string& event_name,           // 事件的名称，例如操作名
    const std::string& backend_name);        // 事件发生的后端名称
TORCH_API void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,         // 配置参数，用于设置 profiler 的行为
    const std::set<torch::profiler::impl::ActivityType>& activities, // 指定要捕获的活动类型集合
    const std::unordered_set<at::RecordScope>& scopes = {});      // 可选的事件作用域集合，默认为空

/*
 * Same as enableProfiler but with callback to do post-processing of
 * KinetoEvents.
 * enableProfilerWithEventPostProcess enables profiler to capture
 * specified activities, with specified RecordFunction scope, if any.
 * Additionally, it takes a functor that does in-place post processing of
 * events, e.g. populate stack trace or module hierarchy information lazily
 * using debug_handle.
 * Example usage is with lite interpreter that has recording scope of
 * LITE_INTERPRETER. In this case lite interpreter runtime, records debug
 * handles in RecordFunction, along with other information. Debug handles are
 * eventually passed down to KinetoEvent and recorded as part of the event.
 * KinetoEdgeCPUProfiler, in torch/csrc/jit/mobile/profiler_edge.cpp, enables
 * profiler using post-processing callback, via
 * enableProfilerWithEventPostProcess, that takes these debug handles and
 * generates stack trace and module hierarchy information, once profiling is
 * done.
 */
using post_process_t = std::function<void(
    /*debug_handle */ int64_t,             // 调试句柄，用于关联事件和其上下文信息
    /*jit_stack    */ std::vector<std::string>&, // JIT 调用栈，保存栈帧信息的字符串向量
    /*jit_modules  */ std::vector<std::string>&)>; // JIT 模块，保存模块名称的字符串向量
TORCH_API void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,    // 配置参数，用于设置 profiler 的行为
    const std::set<torch::profiler::impl::ActivityType>& activities,  // 指定要捕获的活动类型集合
    const std::unordered_set<at::RecordScope>& scopes = {}, // 可选的事件作用域集合，默认为空
    const post_process_t& post_process = nullptr);         // 可选的事件后处理回调函数，用于处理事件后的数据
    const std::set<torch::profiler::impl::ActivityType>& activities,
    // activities 是一个常量引用，表示一组 torch 代码分析器的活动类型集合
    post_process_t&& cb,
    // cb 是一个移动语义的后处理函数对象，用来处理分析结果
    const std::unordered_set<at::RecordScope>& scopes = {});
    // scopes 是一个默认为空的无序集合，包含了 at 命名空间中记录范围的对象
/**
 * 禁用分析器的接口函数，返回唯一指针指向分析器结果对象
 */
TORCH_API std::unique_ptr<ProfilerResult> disableProfiler();

/**
 * 准备分析器的接口函数，配置分析器并指定活动类型集合
 * @param config 分析器配置对象
 * @param activities 活动类型集合
 */
TORCH_API void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities);

/**
 * 当 C++ 线程无法控制分析器的启用方式时，例如由不可达的 Python 代码启用，
 * 可调用这些函数将自身测试/加入/解除分析器的收集集合中。若未调用这些函数，
 * 可能出现的症状是“看不到一些子线程的 GPU 事件”。以下是如何使用它们的示例：
 *
 *    using namespace torch::autograd::profiler;
 *    bool enabled = isProfilerEnabledInMainThread();
 *    if (enabled != saved_enabled_state) {
 *      if (enabled) {
 *        enableProfilerInChildThread();
 *      } else {
 *        disableProfilerInChildThread();
 *      }
 *      saved_enabled_state = enabled;
 *    }
 */
TORCH_API bool isProfilerEnabledInMainThread();
TORCH_API void enableProfilerInChildThread();
TORCH_API void disableProfilerInChildThread();

namespace profiler::impl {

/**
 * 向分析器报告 Vulkan 事件的实验性接口函数
 * @param id Vulkan 事件的标识符
 */
TORCH_API void _reportVulkanEventToProfiler(vulkan_id_t id);

} // namespace profiler::impl

} // namespace torch
```