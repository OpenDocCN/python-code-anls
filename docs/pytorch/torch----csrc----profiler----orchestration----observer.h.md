# `.\pytorch\torch\csrc\profiler\orchestration\observer.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/record_function.h>
// 包含 ATen 库的记录功能头文件

#include <torch/csrc/Export.h>
// 包含 Torch 库的导出定义头文件

#include <utility>
// 包含 C++ 标准库中的实用工具

namespace torch {
namespace profiler {
namespace impl {

// ----------------------------------------------------------------------------
// -- Profiler Config ---------------------------------------------------------
// ----------------------------------------------------------------------------
// 枚举类型定义，表示活动类型
enum class C10_API_ENUM ActivityType {
  CPU = 0, // CPU 相关活动
  XPU,     // XPU 内核，运行时
  CUDA,    // CUDA 内核，运行时
  MTIA,    // MTIA 内核，运行时
  PrivateUse1, // PrivateUse1 内核，运行时
  NUM_KINETO_ACTIVITIES, // 必须是最后一个
};

// 枚举类型定义，表示分析器状态
enum class C10_API_ENUM ProfilerState {
  Disabled = 0,                // 禁用状态
  CPU,                         // 仅 CPU 分析
  CUDA,                        // CPU + CUDA 事件
  NVTX,                        // 仅发出 NVTX 标记
  ITT,                         // 仅发出 ITT 标记
  PRIVATEUSE1,                 // 仅发出 PRIVATEUSE1 标记
  KINETO,                      // 使用 libkineto
  KINETO_GPU_FALLBACK,         // 当 CUPTI 不可用时使用 CUDA 事件
  KINETO_PRIVATEUSE1_FALLBACK, // 使用 PrivateUse1 事件
  KINETO_ONDEMAND,             // 在需求模式下运行分析器
  NUM_PROFILER_STATES,         // 必须是最后一个
};

// 枚举类型定义，表示活动分析器类型
enum class C10_API_ENUM ActiveProfilerType {
  NONE = 0,     // 无分析器
  LEGACY,       // 传统分析器
  KINETO,       // Kineto 分析器
  NVTX,         // NVTX 分析器
  ITT,          // ITT 分析器
  PRIVATEUSE1   // PrivateUse1 分析器
};

// 实验性配置结构体，用于配置分析器
struct TORCH_API ExperimentalConfig {
  ExperimentalConfig(
      std::vector<std::string> profiler_metrics = {},       // 分析器指标
      bool profiler_measure_per_kernel = false,             // 是否每个内核测量
      bool verbose = false,                                 // 是否详细输出
      std::vector<std::string> performance_events = {},     // 性能事件列表
      bool enable_cuda_sync_events = false,                 // 启用 CUDA 同步事件
      bool adjust_timestamps = false);                      // 是否调整时间戳

  explicit operator bool() const;  // 显示类型转换运算符

  std::vector<std::string> profiler_metrics;               // 分析器指标列表
  bool profiler_measure_per_kernel;                        // 是否每个内核测量
  bool verbose;                                            // 是否详细输出
  std::vector<std::string> performance_events;             // 性能事件列表
  bool enable_cuda_sync_events;                            // 启用 CUDA 同步事件
  bool adjust_timestamps;                                  // 是否调整时间戳
  /*
   * For CUDA profiling mode, enable adding CUDA synchronization events
   * that expose CUDA device, stream and event synchronization activities.
   * This feature is new and currently disabled by default.
   */
  /*
   * 用于 CUDA 分析模式，启用添加 CUDA 同步事件，展示 CUDA 设备、流和事件同步活动。
   * 此功能是新功能，默认情况下处于禁用状态。
   */
};
// 定义了一个名为 ProfilerConfig 的结构体，用于配置性能分析器的各项参数
struct TORCH_API ProfilerConfig {
  // 构造函数，初始化配置参数
  ProfilerConfig(
      ProfilerState state,  // 分析器的状态
      bool report_input_shapes = false,  // 是否报告输入形状，默认为 false
      bool profile_memory = false,  // 是否进行内存分析，默认为 false
      bool with_stack = false,  // 是否包含调用栈信息，默认为 false
      bool with_flops = false,  // 是否包含 FLOPs 统计信息，默认为 false
      bool with_modules = false,  // 是否包含模块信息，默认为 false
      ExperimentalConfig experimental_config = ExperimentalConfig());  // 实验性配置，默认构造函数创建

  // 返回分析器是否被禁用
  bool disabled() const;
  // 返回分析器是否为全局模式
  bool global() const;

  // 分析器的状态
  ProfilerState state;
  // 实验性配置
  ExperimentalConfig experimental_config;
  // 是否报告输入形状
  bool report_input_shapes;
  // 是否进行内存分析
  bool profile_memory;
  // 是否包含调用栈信息
  bool with_stack;
  // 是否包含 FLOPs 统计信息
  bool with_flops;
  // 是否包含模块信息
  bool with_modules;

  // 序列化到 IValue 的方法
  at::IValue toIValue() const;
  // 从 IValue 中反序列化得到 ProfilerConfig 对象的静态方法
  static ProfilerConfig fromIValue(const at::IValue& profilerConfigIValue);
};

// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------

// ProfilerStateBase 类继承自 c10::MemoryReportingInfoBase 类，是性能分析器的基类
struct TORCH_API ProfilerStateBase : public c10::MemoryReportingInfoBase {
  // 构造函数，初始化性能分析器状态基类的配置参数
  explicit ProfilerStateBase(ProfilerConfig config);
  // 析构函数，用于清理资源
  ~ProfilerStateBase() override;

  // 获取当前线程的性能分析器状态对象的静态方法
  static ProfilerStateBase* get(bool global);
  // 获取当前线程的性能分析器状态对象的静态方法，自动判断是否全局
  static ProfilerStateBase* get() {
    auto* out = get(/*global=*/true);
    return out ? out : get(/*global=*/false);
  }

  // 将性能分析器状态对象推入堆栈的静态方法
  static void push(std::shared_ptr<ProfilerStateBase>&& state);

  // 从堆栈中弹出性能分析器状态对象的静态方法，自动判断是否全局
  static std::shared_ptr<ProfilerStateBase> pop(bool global);
  static std::shared_ptr<ProfilerStateBase> pop() {
    auto out = pop(/*global=*/true);
    return out ? std::move(out) : pop(/*global=*/false);
  }

  // 返回当前性能分析器状态对象的配置信息
  const ProfilerConfig& config() const {
    return config_;
  }

  // 设置回调函数句柄
  void setCallbackHandle(at::CallbackHandle handle);
  // 移除回调函数句柄
  void removeCallback();

  // 是否启用内存分析的重写方法
  bool memoryProfilingEnabled() const override {
    return config_.profile_memory;
  }

  // 纯虚函数，子类需要实现，返回活跃的性能分析器类型
  virtual ActiveProfilerType profilerType() = 0;

 protected:
  // 互斥量，用于保护状态数据的访问
  std::mutex state_mutex_;
  // 配置对象，包含性能分析器的配置信息，默认状态为 Disabled
  ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled);
  // 回调函数句柄
  at::CallbackHandle handle_ = 0;
};

// 注：以下内容仅适用于当前线程本地的活跃性能分析器。
// 命名空间 impl 中定义了性能分析器相关的接口和实现
namespace impl {
// 返回当前线程的性能分析器是否启用
TORCH_API bool profilerEnabled();
// 返回当前线程的活跃性能分析器类型
TORCH_API ActiveProfilerType profilerType();
// 返回当前线程的性能分析器配置对象
TORCH_API ProfilerConfig getProfilerConfig();

} // namespace impl
} // namespace profiler
} // namespace torch
```