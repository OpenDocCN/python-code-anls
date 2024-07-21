# `.\pytorch\aten\src\ATen\core\PythonFallbackKernel.cpp`

```
// 引入 TorchDispatchModeTLS.h 中的 Torch 调度模式 TLS 功能
#include <c10/core/impl/TorchDispatchModeTLS.h>
// 引入 PythonDispatcherTLS.h 中的 Python 调度器 TLS 功能
#include <c10/core/impl/PythonDispatcherTLS.h>
// 引入 PythonFallbackKernel.h 中的 Python 回退内核
#include <ATen/core/PythonFallbackKernel.h>
// 引入 SafePyObject.h 中的安全 Python 对象
#include <c10/core/SafePyObject.h>

// 匿名命名空间，限定符内部使用
namespace {

// 此 TLS 用于跟踪调度程序的状态，以便在调用 Python 后能够恢复它。
// 具有以下不变性：
//  - 在执行 Python 代码时必须为空。
//  - 即使在多次调度器调用后不返回到 Python，也应该只设置一次。
// 为了实现这一点，我们确保 tls 默认为空，并在调用用户 torch_dispatch 或此调用后返回 Python 时再次清空。

thread_local std::optional<c10::impl::LocalDispatchKeySet> tls_on_entry;

// 获取安全的 tls_on_entry，确保其值存在
c10::impl::LocalDispatchKeySet safe_get_tls_on_entry() {
  TORCH_CHECK(tls_on_entry.has_value(), "Accessing torch dispatch state outside of '__torch_dispatch__' "
              "is not allowed.");
  return tls_on_entry.value();
}

// Python key 之后的所有键集合
constexpr c10::DispatchKeySet after_Python_keyset = c10::DispatchKeySet(c10::DispatchKeySet::FULL) ^
  (c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Python) |
   c10::DispatchKeySet(c10::DispatchKey::Python));

// 此保护假设 tls_on_entry 具有值。
struct StashTLSOnEntryGuard {
public:
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  StashTLSOnEntryGuard(): saved_(tls_on_entry.value()) {
    tls_on_entry = c10::nullopt;
  }

  ~StashTLSOnEntryGuard() {
    TORCH_INTERNAL_ASSERT(!tls_on_entry.has_value());
    tls_on_entry = saved_;
  }

private:
  c10::impl::LocalDispatchKeySet saved_;
};

// Python 回退函数，处理操作符和堆栈
void pythonFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
  // 排除 Python key 之后的调度键集合的调度器保护
  c10::impl::ExcludeDispatchKeyGuard guard(after_Python_keyset);

  // 如果 Torch 调度模式处于活动状态，则使用其 PyInterpreter 进行调度
  const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
  if (mode_stack_len > 0) {
    const auto& cur_torch_dispatch_mode_state = c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
    cur_torch_dispatch_mode_state->pyinterpreter()->dispatch(op, stack);
    return;
  }

  // 否则，在 Tensor 上找到一个 PyInterpreter
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  // 可以安全地在第一个 Tensor 上分发，具有 pyobj_interpreter，
  // 而无需检查任何参数的解释器，因为当我们实际运行 dispatch() 时，
  // 我们将在该解释器的上下文中提取 PyObjects，并确保所有人都在同一个解释器上。
  for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
    // 检查 ivalue 是否为 Tensor 类型
    if (ivalue.isTensor()) {
      // 如果是 Tensor 类型，则获取其 Python 对象的解释器指针
      auto* interpreter = ivalue.unsafeToTensorImpl()->pyobj_slot()->pyobj_interpreter();
      // 如果解释器存在，则调用解释器的 dispatch 方法，并直接返回
      if (interpreter) {
        (*interpreter)->dispatch(op, stack);
        return;
      }
    } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
      // 如果 ivalue 是 TensorList 或 OptionalTensorList 类型
      // 注意：使用 toListRef 以避免增加引用计数（toTensorListRef 方法不存在）
      for (const auto& nv : ivalue.toListRef()) {
        // 遍历 Tensor 列表的每个元素 nv
        // 如果当前元素为 None，则跳过
        if (nv.isNone()) {
          continue;
        }
        // 否则，获取当前元素的 Python 对象的解释器指针
        auto* interpreter = nv.unsafeToTensorImpl()->pyobj_slot()->pyobj_interpreter();
        // 如果解释器存在，则调用解释器的 dispatch 方法，并直接返回
        if (interpreter) {
          (*interpreter)->dispatch(op, stack);
          return;
        }
      }
    }
  }
  // 如果程序执行到这里，则表示没有找到任何具有 PyInterpreter 的参数（可能没有 Tensor 类型的参数）
  TORCH_INTERNAL_ASSERT(0, "Hit Python dispatch key but no arguments had PyInterpreter (no tensor args?)");
}

// 定义一个函数 pythonDispatcherFallback，用于处理来自 PythonDispatcher 的调度操作
void pythonDispatcherFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // 获取当前线程的 PythonDispatcherTLS 状态
  auto* state = c10::impl::PythonDispatcherTLS::get_state();
  // 断言确保 PythonDispatcherTLS 状态存在，若不存在则输出错误信息
  TORCH_INTERNAL_ASSERT(state, "Hit PythonDispatcher dispatch key but PythonDispatcherTLS was not set");
  // 调用 PythonDispatcher 对象的操作方法，处理操作并更新栈数据
  (*state)->python_dispatcher(op, dispatch_keys.remove(c10::DispatchKey::PythonDispatcher), stack);
}

// 定义函数 pythonTLSSnapshotFallback，用于处理来自 PythonTLSSnapshot 的调度操作
void pythonTLSSnapshotFallback(const c10::OperatorHandle &op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // 允许 TLS 可能已经设置
  // 这表示有多个调用进入分发器，而不是源自 Python 代码
  // 下面的保护将正确忽略这样的调用
  at::impl::MaybeSetTLSOnEntryGuard guard;
  // 通过 redispatchBoxed 方法重新调度操作，只保留 PythonTLSSnapshot 以后的调度键
  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PythonTLSSnapshot), stack);
}

// 定义函数 preDispatchFallback，处理 PreDispatch 的调度操作
void preDispatchFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // 通过 redispatchBoxed 方法重新调度操作，只保留 PreDispatch 以后的调度键
  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PreDispatch), stack);
}

} // 匿名命名空间

// 包含了 at::impl 命名空间下的定义
namespace at::impl {

// 实现 RestorePythonTLSSnapshot 类的构造函数
RestorePythonTLSSnapshot::RestorePythonTLSSnapshot() : saved_(safe_get_tls_on_entry()), guard_(safe_get_tls_on_entry()) {
  // 将 tls_on_entry 设为无值状态
  tls_on_entry = c10::nullopt;
}

// 实现 RestorePythonTLSSnapshot 类的析构函数
RestorePythonTLSSnapshot::~RestorePythonTLSSnapshot() {
  // 断言 tls_on_entry 不应有值
  TORCH_INTERNAL_ASSERT(!tls_on_entry.has_value());
  // 恢复 tls_on_entry 的保存状态
  tls_on_entry = saved_;
}

// 实现 MaybeSetTLSOnEntryGuard 类的构造函数
MaybeSetTLSOnEntryGuard::MaybeSetTLSOnEntryGuard() {
  // 如果 tls_on_entry 已有值，则设置 value_set_ 为 false
  if (tls_on_entry.has_value()) {
    value_set_ = false;
  } else {
    // 否则设置 value_set_ 为 true，并保存当前的 tls_on_entry 状态
    value_set_ = true;
    tls_on_entry = c10::impl::tls_local_dispatch_key_set();
  }
}

// 实现 MaybeSetTLSOnEntryGuard 类的析构函数
MaybeSetTLSOnEntryGuard::~MaybeSetTLSOnEntryGuard() {
  // 如果 value_set_ 为 true，则断言 tls_on_entry 必须有值，并将其设为无值状态
  if (value_set_) {
    TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
    tls_on_entry = c10::nullopt;
  }
}

} // namespace at::impl

// 注册 Torch 库的实现，为 Python 模块
TORCH_LIBRARY_IMPL(_, Python, m) {
  // 注册 pythonFallback 函数为回退函数
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallback>());
}

// 注册 Torch 库的实现，为 PythonDispatcher 模块
TORCH_LIBRARY_IMPL(_, PythonDispatcher, m) {
  // 注册 pythonDispatcherFallback 函数为回退函数
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonDispatcherFallback>());
}

// 注册 Torch 库的实现，为 PythonTLSSnapshot 模块
TORCH_LIBRARY_IMPL(_, PythonTLSSnapshot, m) {
  // 注册 pythonTLSSnapshotFallback 函数为回退函数
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonTLSSnapshotFallback>());
}
# 在 Torch 库实现中注册预调度（PreDispatch）功能
TORCH_LIBRARY_IMPL(_, PreDispatch, m) {
    # 将预调度的回退函数注册为 Torch C++ 函数的封装
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&preDispatchFallback>());
}
```