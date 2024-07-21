# `.\pytorch\torch\csrc\jit\mobile\profiler_edge.cpp`

```py
// 包含 C10 核心分配器头文件
#include <c10/core/Allocator.h>
// 包含 C10 异常处理头文件
#include <c10/util/Exception.h>
// 包含 C10 overloaded 实用工具
#include <c10/util/overloaded.h>
// 包含 Torch 移动端 JIT 框架的性能分析器边缘相关头文件
#include <torch/csrc/jit/mobile/profiler_edge.h>
// 包含字符串操作相关头文件
#include <string>
// 包含向量操作相关头文件
#include <vector>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {
// 移动端命名空间
namespace mobile {

// 线程本地变量，用于存储边缘 CPU 分析器的指针
thread_local KinetoEdgeCPUProfiler* tls_edge_profiler{nullptr};

// 构造函数定义，初始化 KinetoEdgeCPUProfiler 对象
KinetoEdgeCPUProfiler::KinetoEdgeCPUProfiler(
    const torch::jit::mobile::Module& m,                  // 输入模块的引用
    const std::string& fname,                             // 跟踪文件名
    const bool report_input_shapes,                        // 是否报告输入形状
    const bool profile_memory,                             // 是否分析内存
    const bool with_stack,                                 // 是否包含堆栈信息
    const bool with_flops,                                 // 是否计算浮点操作
    const bool with_modules,                               // 是否包含模块信息
    std::vector<std::string> events,                       // 性能事件名称向量
    const bool adjust_vulkan_timestamps)                   // 是否调整 Vulkan 时间戳
    : m_(m), trace_file_name_(fname) {
  
  // 实验性配置对象，用于设置硬件计数器
  torch::profiler::impl::ExperimentalConfig experimental_config;
  
  // 如果有指定事件，设置性能事件列表
  if (events.size()) {
    experimental_config.performance_events = std::move(events);
  }

  // 调整 Vulkan 时间戳以便与 CPU 事件时间对齐
  experimental_config.adjust_timestamps = adjust_vulkan_timestamps;

  // 分析器配置对象，设置为 Kineto 分析器状态
  torch::profiler::impl::ProfilerConfig config(
      torch::profiler::impl::ProfilerState::KINETO,
      report_input_shapes,
      profile_memory,
      with_stack,
      with_flops,
      with_modules,
      experimental_config);
  
  // 准备启动分析器，指定活动类型为 CPU
  torch::autograd::profiler::prepareProfiler(
      config, {torch::autograd::profiler::ActivityType::CPU});

  // 如果需要处理模块或者堆栈信息
  if (with_modules || with_stack) {
    // 定义事件后处理函数
    auto post_processing = [this, with_stack, with_modules](
                               int64_t debug_handle,
                               std::vector<std::string>& jit_stack,
                               std::vector<std::string>& jit_modules) {
      std::string no_debug_info("Model was not saved with debug information");
      if (with_modules) {
        // 如果需要模块信息，获取模块层级结构
        jit_modules = std::vector<std::string>(
            {this->m_.hasDebugHandles()
                 ? this->m_.getModuleHierarchy(debug_handle)
                 : no_debug_info});
      } else if (with_stack) {
        // 如果需要堆栈信息，获取调用堆栈
        jit_stack = std::vector<std::string>(
            {this->m_.hasDebugHandles() ? this->m_.getCallStack(debug_handle)
                                        : no_debug_info});
      }
    };

    // 启用分析器并注册事件后处理函数，指定活动类型为 CPU
    torch::autograd::profiler::enableProfilerWithEventPostProcess(
        config,
        {torch::autograd::profiler::ActivityType::CPU},
        post_processing,
        {at::RecordScope::LITE_INTERPRETER});
  } else {
    // 启用分析器，指定活动类型为 CPU
    torch::autograd::profiler::enableProfiler(
        config,
        {torch::autograd::profiler::ActivityType::CPU},
        {at::RecordScope::LITE_INTERPRETER});
  }

  // 设置跟踪文件名
  trace_file_name_ = fname;
  
  // 检查是否已经有正在运行的边缘分析器
  TORCH_CHECK(
      tls_edge_profiler == nullptr, "Edge profiler is already profiling.")
  
  // 将当前对象指针存储在线程本地变量中
  tls_edge_profiler = this;
}

// 命名空间结束
} // namespace mobile
} // namespace jit
} // namespace torch
// 记录后端内存事件到分析器中
void KinetoEdgeCPUProfiler::recordBackendMemoryEvent(
    void* ptr,                             // 指向分配内存的指针
    int64_t alloc_size,                    // 分配内存的大小
    size_t total_allocated,                // 总共已分配的内存
    size_t total_reserved,                 // 总共保留的内存
    c10::Device device) {                  // 分配内存的设备

  // 调用 C10 库函数，向分析器报告内存使用情况
  c10::reportMemoryUsageToProfiler(
      ptr, alloc_size, total_allocated, total_reserved, device);
}

// 记录后端事件到活动 Kineto 分析器中
void KinetoEdgeCPUProfiler::recordBackendEvent(
    const int64_t start_time_us,           // 事件开始时间（微秒）
    const int64_t end_time_us,             // 事件结束时间（微秒）
    const int64_t debug_handle,            // 调试句柄
    const std::string& event_name,         // 事件名称
    const std::string& backend_name) {     // 后端名称

  // 调用 Torch 自动微分库的分析器函数，向活动的 Kineto 分析器报告后端事件
  torch::autograd::profiler::reportBackendEventToActiveKinetoProfiler(
      start_time_us,
      end_time_us,
      debug_handle,
      at::RecordScope::LITE_INTERPRETER,   // 记录范围为轻量级解释器
      event_name,
      backend_name);
}

// 禁用分析器，并返回分析结果的唯一指针
const std::unique_ptr<torch::autograd::profiler::ProfilerResult>&
KinetoEdgeCPUProfiler::disableProfiler() {

  // 检查分析器结果是否为空，如果不为空则抛出异常
  TORCH_CHECK(
      !profiler_result_,
      "KinetoEdgeCPUProfiler already disabled. "
      "To get list of events use getProfilerResults()");

  // 调用 Torch 自动微分库的禁用分析器函数，并将结果赋给成员变量
  profiler_result_ = torch::autograd::profiler::disableProfiler();
  
  // 返回禁用后的分析结果指针
  return profiler_result_;
}

// 获取分析结果的唯一指针
const std::unique_ptr<torch::autograd::profiler::ProfilerResult>&
KinetoEdgeCPUProfiler::getProfilerResult() {

  // 检查分析器结果是否为空，如果为空则抛出异常
  TORCH_CHECK(
      profiler_result_,
      "KinetoEdgeCPUProfiler has not been disabled. "
      "use disableProfiler() API first, which returns the ProfilerResult.");

  // 返回分析结果的唯一指针
  return profiler_result_;
}

// 析构函数，负责释放资源和保存追踪文件（如果已指定）
KinetoEdgeCPUProfiler::~KinetoEdgeCPUProfiler() {

  // 如果追踪文件名不为空
  if (!trace_file_name_.empty()) {

    // 如果分析器结果指针不为空，则保存分析结果到指定的追踪文件
    if (profiler_result_) {
      profiler_result_->save(trace_file_name_);
    } else {
      // 否则，禁用分析器并保存分析结果到指定的追踪文件
      torch::autograd::profiler::disableProfiler()->save(trace_file_name_);
    }
  }

  // 清空 TLS 变量，释放 Edge 分析器的资源
  tls_edge_profiler = nullptr;
}

// 获取当前线程的 Edge 分析器实例指针
KinetoEdgeCPUProfiler* getCurrentEdgeProfiler() {
  return tls_edge_profiler;
}

} // namespace mobile
} // namespace jit
} // namespace torch
```