# `.\pytorch\torch\csrc\profiler\orchestration\observer.cpp`

```py
// 引入 Torch 的性能分析模块中的观察器头文件
#include <torch/csrc/profiler/orchestration/observer.h>

// 引入 Torch 的性能分析模块中的实用工具头文件
#include <torch/csrc/profiler/util.h>

// 引入 C++ 标准库中的实用工具
#include <utility>

// 定义 Torch 命名空间
namespace torch {
namespace profiler {
namespace impl {

// 使用别名 GlobalManager 表示 ProfilerStateBase 的全局状态管理器
using GlobalManager = GlobalStateManager<ProfilerStateBase>;

// ----------------------------------------------------------------------------
// -- Profiler Config ---------------------------------------------------------
// ----------------------------------------------------------------------------

// 构造函数：初始化 ExperimentalConfig 对象
ExperimentalConfig::ExperimentalConfig(
    std::vector<std::string> profiler_metrics,                    // 性能指标列表
    bool profiler_measure_per_kernel,                             // 是否按照内核测量
    bool verbose,                                                 // 是否详细输出
    std::vector<std::string> performance_events,                  // 性能事件列表
    bool enable_cuda_sync_events,                                 // 是否启用 CUDA 同步事件
    bool adjust_timestamps)                                       // 是否调整时间戳
    : profiler_metrics{std::move(profiler_metrics)},              // 初始化性能指标
      profiler_measure_per_kernel{profiler_measure_per_kernel},   // 初始化按内核测量选项
      verbose{verbose},                                           // 初始化详细输出选项
      performance_events(std::move(performance_events)),          // 初始化性能事件列表
      enable_cuda_sync_events{enable_cuda_sync_events},           // 初始化 CUDA 同步事件选项
      adjust_timestamps{adjust_timestamps} {}                     // 初始化时间戳调整选项

// 转换运算符：将 ExperimentalConfig 转换为布尔值，判断是否有效
/*explicit*/ ExperimentalConfig::operator bool() const {
  return !profiler_metrics.empty();  // 返回性能指标列表是否非空的布尔值
}

// 构造函数：初始化 ProfilerConfig 对象
ProfilerConfig::ProfilerConfig(
    ProfilerState state,                                           // 性能分析器状态
    bool report_input_shapes,                                      // 是否报告输入形状
    bool profile_memory,                                           // 是否分析内存
    bool with_stack,                                               // 是否记录堆栈信息
    bool with_flops,                                               // 是否记录浮点操作
    bool with_modules,                                             // 是否记录模块信息
    ExperimentalConfig experimental_config)                        // 实验性配置
    : state{state},                                                // 初始化性能分析器状态
      experimental_config{std::move(experimental_config)},         // 初始化实验性配置
      report_input_shapes{report_input_shapes},                    // 初始化报告输入形状选项
      profile_memory{profile_memory},                              // 初始化内存分析选项
      with_stack{with_stack},                                      // 初始化记录堆栈信息选项
      with_flops{with_flops},                                      // 初始化记录浮点操作选项
      with_modules{with_modules} {}                                // 初始化记录模块信息选项

// 方法：检查性能分析器是否禁用
bool ProfilerConfig::disabled() const {
  return state == torch::profiler::impl::ProfilerState::Disabled;  // 返回性能分析器状态是否为 Disabled
}

// 方法：检查是否使用全局性能分析器
bool ProfilerConfig::global() const {
  return state == torch::profiler::impl::ProfilerState::KINETO_ONDEMAND;  // 返回性能分析器状态是否为 KINETO_ONDEMAND
}

// 命名空间：定义枚举变量 ProfilerIValueIdx
namespace {
enum ProfilerIValueIdx {
  STATE = 0,                // 状态索引
  REPORT_INPUT_SHAPES,      // 报告输入形状索引
  PROFILE_MEMORY,           // 内存分析索引
  NUM_PROFILER_CFG_IVALUE_IDX // 必须是列表中的最后一个元素
};
} // namespace

// 方法：将 ProfilerConfig 转换为 IValue 对象
at::IValue ProfilerConfig::toIValue() const {
  // 创建泛型列表 eventIValueList，其中存储各种性能分析器配置项
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  eventIValueList.reserve(NUM_PROFILER_CFG_IVALUE_IDX);  // 预留列表空间
  eventIValueList.emplace_back(static_cast<int64_t>(state));  // 添加状态索引
  eventIValueList.emplace_back(report_input_shapes);           // 添加报告输入形状选项
  eventIValueList.emplace_back(profile_memory);                // 添加内存分析选项
  return eventIValueList;                                      // 返回配置项列表
}

// 静态方法：从 IValue 对象创建 ProfilerConfig 对象
ProfilerConfig ProfilerConfig::fromIValue(
    // 使用 const 引用传入的 profilerConfigIValue 参数来构建 ProfilerConfig 对象
    const at::IValue& profilerConfigIValue) {
  
  // 断言 profilerConfigIValue 必须是一个列表类型的 IValue
  TORCH_INTERNAL_ASSERT(
      profilerConfigIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  
  // 将 profilerConfigIValue 转换为标准库 vector
  auto ivalues = profilerConfigIValue.toList();
  
  // 断言 ivalues 的大小必须等于 NUM_PROFILER_CFG_IVALUE_IDX
  TORCH_INTERNAL_ASSERT(
      ivalues.size() == NUM_PROFILER_CFG_IVALUE_IDX,
      c10::str(
          "Expected exactly ",
          NUM_PROFILER_CFG_IVALUE_IDX,
          " ivalues to resconstruct ProfilerConfig."));
  
  // 使用从 ivalues 中提取的数据重构 ProfilerConfig 对象并返回
  return ProfilerConfig(
      static_cast<ProfilerState>(ivalues.get(ProfilerIValueIdx::STATE).toInt()),
      ivalues.get(ProfilerIValueIdx::REPORT_INPUT_SHAPES).toBool(),
      ivalues.get(ProfilerIValueIdx::PROFILE_MEMORY).toBool());
// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------

// 显式构造函数，接受一个 ProfilerConfig 对象作为参数，继承自 MemoryReportingInfoBase
/*explicit*/ ProfilerStateBase::ProfilerStateBase(ProfilerConfig config)
    : c10::MemoryReportingInfoBase(), config_(std::move(config)) {}

// 析构函数
ProfilerStateBase::~ProfilerStateBase() {
  // 如果 handle_ 不为空，则移除回调函数
  if (handle_) {
    auto handle = handle_;
    removeCallback();
    // 断言，检测是否有泄漏的回调句柄
    SOFT_ASSERT(false, "Leaked callback handle: ", handle);
  }
}

// 静态方法，根据 global 参数获取 ProfilerStateBase 对象
/*static*/ ProfilerStateBase* ProfilerStateBase::get(bool global) {
  auto* out = global
      ? GlobalManager::get()
      : static_cast<ProfilerStateBase*>(
            c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE));
  // 断言，确保获取的对象不为空，并且其配置中的全局标志与 global 参数一致
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!out || out->config().global() == global);
  return out;
}

// 静态方法，将给定的 ProfilerStateBase 对象 state 推入相应的管理器中
/*static*/ void ProfilerStateBase::push(
    std::shared_ptr<ProfilerStateBase>&& state) {
  // 断言，确保传入的状态对象不为空
  TORCH_INTERNAL_ASSERT(state != nullptr);
  // 如果状态对象是全局的，则推入全局管理器中
  if (state->config().global()) {
    GlobalManager::push(std::move(state));
  } else {
    // 否则推入线程本地的调试信息中
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
  }
}

// 匿名命名空间，用于实现线程本地状态对象的弹出操作
namespace {
std::shared_ptr<ProfilerStateBase> popTLS() {
  // 如果存在活动的线程本地分析器，则返回其对象；否则返回空指针
  // 如果存在活动分析器但不是顶层的 DebugInfoBase，则调用 _pop 会抛出异常
  // TODO(robieta): make `noexcept` version.
  return c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE)
      ? std::static_pointer_cast<ProfilerStateBase>(
            c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE))
      : nullptr;
}
} // namespace

// 静态方法，根据 global 参数弹出 ProfilerStateBase 对象
/*static*/ std::shared_ptr<ProfilerStateBase> ProfilerStateBase::pop(
    bool global) {
  // 根据 global 参数选择弹出方式，并返回弹出的对象
  auto out = global ? GlobalManager::pop() : popTLS();
  // 断言，确保弹出的对象不为空，并且其配置中的全局标志与 global 参数一致
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!out || out->config().global() == global);
  return out;
}

// 设置回调句柄的方法
void ProfilerStateBase::setCallbackHandle(at::CallbackHandle handle) {
  // 如果已经存在回调句柄，则先移除现有的回调函数
  if (handle_) {
    at::removeCallback(handle_);
    // 断言，检测是否存在已注册的回调句柄，移除以避免泄漏
    SOFT_ASSERT(
        false,
        "ProfilerStateBase already has a registered callback. "
        "Removing to avoid leaked callback.");
  }
  // 设置新的回调句柄
  handle_ = handle;
}

// 移除回调函数的方法
void ProfilerStateBase::removeCallback() {
  // 如果存在回调句柄，则先移除注册的回调函数，然后将 handle_ 置为 0
  if (handle_) {
    at::removeCallback(handle_);
    handle_ = 0;
  }
}

// 检查分析器是否启用的函数
bool profilerEnabled() {
  // 获取当前线程的 ProfilerStateBase 对象指针
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  // 返回是否存在且未被禁用
  return state_ptr && !state_ptr->config().disabled();
}

// 获取当前分析器类型的函数
TORCH_API ActiveProfilerType profilerType() {
  // 获取当前线程的 ProfilerStateBase 对象指针
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  // 如果对象为空，则返回 NONE 类型；否则返回其当前的分析器类型
  return state_ptr == nullptr ? ActiveProfilerType::NONE
                              : state_ptr->profilerType();
}
// 定义函数 getProfilerConfig，返回类型为 torch::profiler::impl::ProfilerConfig
torch::profiler::impl::ProfilerConfig getProfilerConfig() {
  // 获取非全局的分析器状态指针
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  // 检查状态指针是否有效，否则抛出错误信息
  TORCH_CHECK(
      state_ptr,
      "Tried to access profiler config, but profiler is not enabled!");
  // 返回状态指针对应的配置信息
  return state_ptr->config();
}

// 命名空间结束符号
} // namespace impl
// 命名空间结束符号
} // namespace profiler
// 命名空间结束符号
} // namespace torch
```