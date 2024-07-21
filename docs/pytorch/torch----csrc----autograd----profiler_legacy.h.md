# `.\pytorch\torch\csrc\autograd\profiler_legacy.h`

```
#pragma once

#include <cstdint>  // 包含整数类型定义的头文件
#include <iostream>  // 输入输出流库
#include <memory>  // 智能指针库
#include <mutex>  // 互斥量库
#include <string>  // 字符串库
#include <vector>  // 向量库

#include <torch/csrc/Export.h>  // Torch导出相关定义
#include <torch/csrc/profiler/api.h>  // 分析器API定义
#include <torch/csrc/profiler/stubs/base.h>  // 分析器基本定义
#include <torch/csrc/profiler/util.h>  // 分析器工具函数定义

namespace torch::autograd {

struct Node;  // 定义节点结构体，但未在此处实现

namespace profiler {

enum class C10_API_ENUM EventKind : uint16_t {
  Mark,  // 标记事件类型
  PushRange,  // 推入范围事件类型
  PopRange,  // 弹出范围事件类型
  MemoryAlloc,  // 内存分配事件类型
};

// To be deprecated, once we switch to Kineto profiling
struct TORCH_API LegacyEvent {
  LegacyEvent(
      EventKind kind,
      at::StringView name,
      uint16_t thread_id,
      bool record_cuda,
      at::RecordFunctionHandle handle = 0,
      std::vector<std::vector<int64_t>>&& shapes = {},
      int64_t node_id = -1,
      bool is_async = false)
      : name_(std::move(name)),  // 初始化事件名称
        kind_(kind),  // 初始化事件类型
        thread_id_(thread_id),  // 初始化线程ID
        handle_(handle),  // 初始化记录函数句柄
        shapes_(std::move(shapes)),  // 初始化形状信息
        node_id_(node_id),  // 初始化节点ID
        is_async_(is_async) {  // 初始化是否异步标志
    record(record_cuda);  // 记录CUDA事件
  }

  // Constructor to be used in conjunction with LegacyEvent::fromIValue.
  LegacyEvent(
      EventKind kind,
      at::StringView name,
      uint16_t thread_id,
      at::RecordFunctionHandle handle,
      std::vector<std::vector<int64_t>>&& shapes,
      int64_t node_id,
      bool is_remote,
      int64_t cpu_memory_usage,
      int64_t cpu_ns,
      bool cuda_recorded,
      int64_t cuda_memory_usage = 0,
      c10::DeviceIndex device = -1,
      double cuda_us = -1)
      : cpu_ns_(cpu_ns),  // 初始化CPU时间
        name_(std::move(name)),  // 初始化事件名称
        kind_(kind),  // 初始化事件类型
        thread_id_(thread_id),  // 初始化线程ID
        handle_(handle),  // 初始化记录函数句柄
        shapes_(std::move(shapes)),  // 初始化形状信息
        cpu_memory_usage_(cpu_memory_usage),  // 初始化CPU内存使用量
        cuda_memory_usage_(cuda_memory_usage),  // 初始化CUDA内存使用量
        device_(device),  // 初始化设备索引
        node_id_(node_id),  // 初始化节点ID
        is_remote_(is_remote),  // 初始化是否远程
        cuda_us_(static_cast<int64_t>(cuda_us)) {  // 初始化CUDA时间
    // Sanity check values that were deserialized
    TORCH_INTERNAL_ASSERT(cpu_ns_ > 0);  // 检查CPU时间值大于0
    if (cuda_recorded) {  // 如果记录了CUDA事件
      TORCH_INTERNAL_ASSERT(device_ >= 0);  // 检查设备索引大于等于0
      TORCH_INTERNAL_ASSERT(cuda_us_ >= 0);  // 检查CUDA时间大于等于0
    }
  }

  // Returns IValues corresponding to event structure, to be used for
  // serialization.
  at::IValue toIValue() const;  // 返回与事件结构对应的IValue，用于序列化

  // Reconstructs an event from IValues given by toIValue.
  static LegacyEvent fromIValue(const at::IValue& eventIValue);  // 从给定的IValue重建事件

  void record(bool record_cuda);  // 记录事件，根据是否记录CUDA

  std::string kindStr() const {  // 返回事件类型的字符串表示
    switch (kind_) {
      case EventKind::Mark:
        return "mark";  // 标记事件类型字符串
      case EventKind::PushRange:
        return "push";  // 推入范围事件类型字符串
      case EventKind::PopRange:
        return "pop";  // 弹出范围事件类型字符串
      case EventKind::MemoryAlloc:
        return "memory_alloc";  // 内存分配事件类型字符串
    }
    throw std::runtime_error("unknown event kind");  // 抛出未知事件类型的运行时错误
  }

  EventKind kind() const {  // 返回事件类型
    return kind_;
  }

  const char* name() const {  // 返回事件名称
    return name_.str();
  }

  uint64_t threadId() const {  // 返回线程ID
    return thread_id_;
  }

  std::vector<std::vector<int64_t>> shapes() const {  // 返回形状信息向量
  // 返回 shapes_ 成员变量的值
  return shapes_;
}

// 计算给定 LegacyEvent 对象 e 的 CPU 时间差，单位为微秒
double cpuElapsedUs(const LegacyEvent& e) const {
  return static_cast<double>(e.cpu_ns_ - cpu_ns_) / (1000.0);
}

// 设置 CPU 时间，单位为微秒
void setCpuUs(int64_t cpu_us) {
  cpu_ns_ = cpu_us * 1000;
}

// 获取 CPU 时间，单位为微秒
double cpuUs() const {
  return static_cast<double>(cpu_ns_) / (1000.0);
}

// 声明 CUDA 时间差计算函数，实际实现未提供

// 检查是否有 CUDA 事件或者是远程事件且指定了设备
bool hasCuda() const {
  return cuda_event != nullptr || (isRemote() && device_ != -1);
}

// 返回事件的设备索引
c10::DeviceIndex device() const {
  return device_;
}

// 更新内存统计信息，根据设备类型分配内存大小
void updateMemoryStats(int64_t alloc_size, c10::Device device) {
  if (device.is_cuda() || device.type() == c10::DeviceType::HIP) {
    cuda_memory_usage_ = alloc_size;
  } else if (
      device.is_cpu() || device.type() == c10::DeviceType::MKLDNN ||
      device.type() == c10::DeviceType::IDEEP) {
    cpu_memory_usage_ = alloc_size;
  } else {
    LOG(WARNING) << "Unsupported memory profiling device: " << device;
  }
}

// 获取 CPU 内存使用量
int64_t cpuMemoryUsage() const {
  return cpu_memory_usage_;
}

// 获取 CUDA 内存使用量
int64_t cudaMemoryUsage() const {
  return cuda_memory_usage_;
}

// 返回记录函数句柄
at::RecordFunctionHandle handle() const {
  return handle_;
}

// 返回事件的节点 ID
int64_t nodeId() const {
  return node_id_;
}

// 设置事件的节点 ID
void setNodeId(int64_t node_id) {
  node_id_ = node_id;
}

// 设置事件的名称
void setName(at::StringView newName_) {
  name_ = std::move(newName_);
}

// 检查事件是否为远程事件
bool isRemote() const {
  return is_remote_;
}

// 设置 CUDA 时间，单位为微秒
void setCudaUs(int64_t cuda_us) {
  cuda_us_ = cuda_us;
}

// 设置事件的序列号
void setSequenceNr(int64_t sequence_nr) {
  sequence_nr_ = sequence_nr;
}

// 获取事件的序列号
int64_t sequenceNr() const {
  return sequence_nr_;
}

// 设置事件的关联 ID
void setCorrelationId(uint64_t correlation_id) {
  correlation_id_ = correlation_id;
}

// 获取事件的关联 ID
uint64_t correlationId() const {
  return correlation_id_;
}

// 返回事件的调用栈
const std::vector<std::string>& stack() const {
  return stack_;
}

// 设置事件的调用栈
void setStack(const std::vector<std::string>& stack) {
  stack_ = stack;
}

// 返回前向线程 ID
uint64_t fwdThreadId() const {
  return fwd_thread_id_;
}

// 设置前向线程 ID
void setFwdThreadId(uint64_t fwd_thread_id) {
  fwd_thread_id_ = fwd_thread_id;
}

// 返回事件的作用域
uint8_t scope() const {
  return scope_;
}

// 设置事件的作用域
void setScope(uint8_t scope) {
  scope_ = scope;
}

// 返回额外参数字典
const std::unordered_map<std::string, c10::IValue>& extraArgs() const {
  return extra_args_;
}

// 设置额外参数字典
void setExtraArgs(std::unordered_map<std::string, c10::IValue>&& save_args) {
  extra_args_ = std::move(save_args);
}

// 返回事件的浮点运算次数
uint64_t flops() {
  return flops_;
}

// 检查事件是否是异步的
bool isAsync() {
  return is_async_;
}

// 设置事件的浮点运算次数
void setFlops(uint64_t flops) {
    // 将传入的 flops 值赋给 flops_ 成员变量
    flops_ = flops;
  }

 private:
  // signed to allow for negative intervals, initialized for safety.
  // cpu_ns_ 表示 CPU 时间，初始值为 0，用于记录时间间隔，采用有符号整数以支持负数间隔
  int64_t cpu_ns_ = 0;
  // name_ 表示事件的名称，用于记录事件的描述信息
  at::StringView name_;
  // kind_ 表示事件的类型，记录事件的种类
  EventKind kind_;
  // thread_id_ 表示事件所在线程的唯一标识
  uint64_t thread_id_;
  // fwd_thread_id_ 表示前向线程的唯一标识，默认为 0
  uint64_t fwd_thread_id_{0};
  // handle_ 表示记录函数的句柄，初始化为 0
  at::RecordFunctionHandle handle_{0};
  // shapes_ 表示操作的输入和输出张量的形状信息的容器
  std::vector<std::vector<int64_t>> shapes_;
  // cpu_memory_usage_ 表示 CPU 内存使用量，初始值为 0
  int64_t cpu_memory_usage_ = 0;
  // cuda_memory_usage_ 表示 CUDA 内存使用量，初始值为 0
  int64_t cuda_memory_usage_ = 0;
  // device_ 表示事件涉及的设备索引，初始化为 -1
  c10::DeviceIndex device_ = -1;
  // cuda_event 表示 CUDA 事件的存根，初始化为 nullptr
  torch::profiler::impl::ProfilerVoidEventStub cuda_event = nullptr;
  // node_id_ 表示事件所属节点的标识，初始化为 0
  int64_t node_id_ = 0;
  // is_remote_ 表示事件是否是远程事件，初始化为 false
  bool is_remote_ = false;
  // cuda_us_ 表示 CUDA 时间的微秒数，初始化为 -1
  int64_t cuda_us_ = -1;
  // sequence_nr_ 表示事件的序列号，初始化为 -1
  int64_t sequence_nr_ = -1;
  // is_async_ 表示事件是否是异步的，初始化为 false
  bool is_async_ = false;

  // stack_ 表示事件调用堆栈的字符串列表
  std::vector<std::string> stack_;
  // scope_ 表示事件的作用域，初始化为 0
  uint8_t scope_{0};
  // correlation_id_ 表示事件的相关标识，初始化为 0
  uint64_t correlation_id_{0};
  // extra_args_ 表示用于计算操作 FLOPS 的额外参数的无序映射
  // flops_ 表示操作的浮点运算次数（FLOPS），初始化为 0
  std::unordered_map<std::string, c10::IValue> extra_args_;
  uint64_t flops_ = 0;
};

// 定义一个结构体 RangeEventList，用于存储固定大小的向量链表，以避免在性能分析事件中
// std::vector 的重新分配操作耗费大量时间。
struct RangeEventList {
  // 构造函数，初始化 events_ 向量的预留容量为 kReservedCapacity
  RangeEventList() {
    events_.reserve(kReservedCapacity);
  }

  // 记录函数模板，使用参数包 Args&& args 来记录事件
  template <typename... Args>
  void record(Args&&... args) {
    std::lock_guard<std::mutex> guard(mutex_);
    // 向 events_ 向量尾部添加一个新元素，元素值由传入的参数 args 初始化
    events_.emplace_back(std::forward<Args>(args)...);
  }

  // 合并函数，将所有已记录的事件转移至新的 std::vector<LegacyEvent> 对象中
  std::vector<LegacyEvent> consolidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<LegacyEvent> result;
    // 将 events_ 向量的所有元素转移至 result 向量的开头
    result.insert(
        result.begin(),
        std::make_move_iterator(events_.begin()),
        std::make_move_iterator(events_.end()));
    // 清空 events_ 向量
    events_.erase(events_.begin(), events_.end());
    return result;
  }

  // 返回 events_ 向量的当前大小
  size_t size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
  }

 private:
  // 用于在多线程访问同一 RangeEventList 实例时进行序列化访问的互斥量
  std::mutex mutex_;
  // 存储 LegacyEvent 对象的向量，用于记录事件
  std::vector<LegacyEvent> events_;

  // 预留的 events_ 向量的初始容量
  static const size_t kReservedCapacity = 1024;
};

// 用于控制 disableProfiler 选项设置的结构体
struct TORCH_API ProfilerDisableOptions {
  ProfilerDisableOptions() = default;
  // 构造函数，根据传入的参数设置 cleanupTLSState 和 consolidate 成员变量
  ProfilerDisableOptions(bool shouldCleanupTLSState, bool shouldConsolidate)
      : cleanupTLSState(shouldCleanupTLSState),
        consolidate(shouldConsolidate) {}
  // 是否应该清理线程本地的分析器状态，如 ThreadLocalDebugInfo 和线程本地的 RecordFunction 回调
  bool cleanupTLSState = true;
  // 是否应该合并当前记录的所有分析事件。如果为 false，则不会合并，其他线程可以继续向事件列表中写入数据。
  bool consolidate = true;
};

// 注意：分析器模式是线程本地的，会自动在线程边界（如 at::launch 任务中）传播
TORCH_API void enableProfilerLegacy(
    const torch::profiler::impl::ProfilerConfig&);

// 使用 std::vector<std::vector<LegacyEvent>> 类型定义线程事件列表别名 thread_event_lists
using thread_event_lists = std::vector<std::vector<LegacyEvent>>;

// 关闭分析器的传统接口函数，可选参数 profilerDisableOptions 用于设置禁用分析器的选项
TORCH_API thread_event_lists disableProfilerLegacy(
    std::optional<ProfilerDisableOptions> profilerDisableOptions =
        c10::nullopt);

// 将 profiledEvents 添加到当前线程本地记录的事件列表中，每个事件将由 fromNodeId 指定的节点 ID 标记
TORCH_API void addEventList(std::vector<LegacyEvent>&& profiledEvents);

// 将分析事件写入流中的函数
TORCH_API void writeProfilerEventsToStream(
    std::ostream& out,
    const std::vector<LegacyEvent*>& events);

// 用法：
//   {
//     RecordProfile guard("filename.trace");
//     // 要进行分析的代码
//   }
// 然后在 chrome://tracing 中打开 filename.trace 文件进行分析
struct TORCH_API RecordProfile {
  // 构造函数，初始化 RecordProfile 对象时，接收一个输出流作为参数
  RecordProfile(std::ostream& out);
  // 构造函数，初始化 RecordProfile 对象时，接收一个文件名作为参数
  RecordProfile(const std::string& filename);

  // 析构函数，用于清理资源或完成其他清理工作
  ~RecordProfile();

 private:
  // 初始化函数，用于初始化 RecordProfile 对象的私有成员
  void init();
  // 文件输出流的唯一指针，用于记录分析事件
  std::unique_ptr<std::ofstream> file_;
  // 输出流的引用，用于记录分析事件
  std::ostream& out_;
  // 处理事件的私有成员函数，将事件处理结果写入文件或输出流
  void processEvents(const std::vector<LegacyEvent*>& events);
};
// 定义一个结构体 TLSLegacyProfilerGuard，用于启用旧版分析器，并接收一个可选的回调函数处理结果。用法如下：
// {
//   TLSLegacyProfilerGuard g([](thread_event_lists profilerResults) {
//     // 处理分析器结果
//   });
//   需要进行分析的代码块
// }
struct TORCH_API TLSLegacyProfilerGuard {
  explicit TLSLegacyProfilerGuard(
      const torch::profiler::impl::ProfilerConfig& cfg,
      std::optional<std::function<void(const thread_event_lists&)>>
          resultCallback = c10::nullopt,
      std::optional<ProfilerDisableOptions> profilerDisableOptions =
          c10::nullopt)
      : cb_(std::move(resultCallback)),
        profilerDisableOptions_(profilerDisableOptions) {
    // 启用旧版分析器
    enableProfilerLegacy(cfg);
  }
  ~TLSLegacyProfilerGuard() {
    // 禁用旧版分析器，并获取事件列表
    thread_event_lists event_lists =
        disableProfilerLegacy(profilerDisableOptions_);
    // 如果有回调函数，则尝试处理分析器事件列表
    if (cb_) {
      try {
        (*cb_)(event_lists);
      } catch (const std::exception& e) {
        // 捕获异常并记录错误日志
        LOG(ERROR) << "Got error processing profiler events: " << e.what();
      }
    }
  }

 private:
  std::optional<std::function<void(const thread_event_lists&)>> cb_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  // 可选的禁用选项，用于配置禁用分析器
  const std::optional<ProfilerDisableOptions> profilerDisableOptions_;
};

// 命名空间声明结束：profiler
} // namespace profiler
// 命名空间声明结束：torch::autograd
} // namespace torch::autograd
```