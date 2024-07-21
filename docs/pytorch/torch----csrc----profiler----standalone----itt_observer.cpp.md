# `.\pytorch\torch\csrc\profiler\standalone\itt_observer.cpp`

```py
#include <torch/csrc/profiler/standalone/itt_observer.h>

#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

namespace torch::profiler::impl {

// ITTThreadLocalState 结构体，继承自 ProfilerStateBase，用于管理 ITT 专用的线程本地状态
struct ITTThreadLocalState : ProfilerStateBase {
  explicit ITTThreadLocalState(const ProfilerConfig& config)
      : ProfilerStateBase(config) {
    // 在这个上下文中只有 `report_input_shapes` 参数是有意义的，其他参数必须为 false
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  // 虚析构函数，供基类调用
  ~ITTThreadLocalState() override = default;

  // 返回活跃的分析器类型为 ITT
  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::ITT;
  }

  // 不实现内存使用报告的方法
  void reportMemoryUsage(void*, int64_t, size_t, size_t, c10::Device) override {
  }

  // 静态方法，获取线程本地 ITTThreadLocalState 实例
  static ITTThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    // 断言调试模式下，获取到的实例为空或者是 ITTThreadLocalState 类型
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::ITT);
    return static_cast<ITTThreadLocalState*>(tls);
  }
};

// 模板函数 enterITT，进入 ITT 分析的上下文，根据 report_input_shapes 参数决定是否推入范围
template <bool report_input_shapes>
std::unique_ptr<at::ObserverContext> enterITT(const at::RecordFunction& fn) {
  if (ITTThreadLocalState::getTLS() != nullptr) {
    torch::profiler::impl::ittStubs()->rangePush(fn.name());
  }
  return nullptr;
}

// 推入 ITT 回调函数的配置和作用域
void pushITTCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  // 检查 ITT 分析器是否启用，否则抛出异常
  TORCH_CHECK(
      torch::profiler::impl::ittStubs()->enabled(),
      "Can't use ITT profiler - PyTorch was compiled without ITT");

  // 推入调试信息，设置为 ITTThreadLocalState 的实例
  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<ITTThreadLocalState>(config));

  // 获取当前线程的 ITTThreadLocalState 实例
  auto state_ptr = ITTThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  // 添加线程本地回调函数，用于记录函数的输入形状
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          state_ptr->config().report_input_shapes
              ? &enterITT</*report_input_shapes=*/true>
              : &enterITT</*report_input_shapes=*/false>,
          [](const at::RecordFunction&, at::ObserverContext*) {
            torch::profiler::impl::ittStubs()->rangePop();
          })
          .needsInputs(config.report_input_shapes)
          .scopes(scopes));
  state_ptr->setCallbackHandle(handle);
}

} // namespace torch::profiler::impl
```