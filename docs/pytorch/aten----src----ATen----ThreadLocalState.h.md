# `.\pytorch\aten\src\ATen\ThreadLocalState.h`

```py
#pragma once

#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <ATen/FuncTorchTLS.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/ThreadLocalPythonObjects.h>
#include <ATen/record_function.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace at {

// Thread local state contains values that are preserved across
// thread boundaries (e.g. at::launch/JIT fork, autograd).
// Note at::parallel_for doesn't preserve TLS across thread boundaries.
class TORCH_API ThreadLocalState {
 public:
  // Constructor that initializes the thread local state
  ThreadLocalState();

  // Sets the gradient mode flag in the current thread's state
  void set_grad_mode(bool enabled);

  // Sets the multithreading enabled flag in the current thread's state
  void set_multithreading_enabled(bool enabled);

  // Sets the thread local state variables in the current thread
  static void setThreadLocalState(const ThreadLocalState& state);

 private:
  // Holds the set of local dispatch keys for the current thread
  c10::impl::LocalDispatchKeySet dispatch_key_;

  // Thread-local debug information shared across the current thread
  std::shared_ptr<c10::ThreadLocalDebugInfo> debug_info_;

  // Thread-local state for recording function calls
  RecordFunctionTLS rf_tls_;

  // Thread-local state for out-of-tree functorch operations
  std::shared_ptr<const functorch::FuncTorchTLSBase> functorch_tls_;

  // Thread-local state for Autograd
  AutogradState autograd_tls_;

  // Thread-local state for Torch dispatch mode
  c10::impl::TorchDispatchModeTLS torch_dispatch_mode_state_;

  // Thread-local state for Python dispatcher
  c10::impl::PyInterpreter* python_dispatcher_state_;

  // Thread-local state for Python __torch_function__ customization
  at::impl::PythonTorchFunctionTLS python_torch_function_state_;

  // Thread-local state for saved tensor default hooks
  at::impl::SavedTensorDefaultHooksTLS saved_tensors_default_hooks_state_;

  // State flag for functionalization reapplying views
  bool functionalization_reapply_views_state_;

  // Thread-local storage for arbitrary Python objects registered via hooks
  at::impl::ThreadLocalPythonObjects saved_objects_;

  friend class ThreadLocalStateGuard;
};

// Guard to set and reset the thread local state
class TORCH_API ThreadLocalStateGuard {
 public:
  // ThreadLocalStateGuard 类的构造函数，接收一个 ThreadLocalState 对象作为参数
  explicit ThreadLocalStateGuard(const ThreadLocalState& state)
      // 初始化 prev_state_ 为默认构造的 ThreadLocalState 对象
      : prev_state_(ThreadLocalState()) {
    // 将给定的 state 跨线程边界设置
    ThreadLocalState::setThreadLocalState(state);
  }

  // ThreadLocalStateGuard 类的析构函数
  ~ThreadLocalStateGuard() {
    // 恢复之前设置的变量状态
    ThreadLocalState::setThreadLocalState(prev_state_);
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  // 之前设置的 ThreadLocalState 对象
  const ThreadLocalState prev_state_;
};

// 模板函数 wrapPropagateTLSState，接收一个回调函数 callback 作为参数
template <typename T>
auto wrapPropagateTLSState(T callback) {
  // 返回一个 lambda 表达式，捕获 tls_state 和 callback
  return [tls_state = ThreadLocalState(),
          callback = std::move(callback)](auto&&... args) {
    // 在当前作用域内创建 ThreadLocalStateGuard 对象 g，传入 tls_state
    ThreadLocalStateGuard g(tls_state);
    // 传播 callback() 返回的值
    return callback(std::forward<decltype(args)>(args)...);
  };
}

// at 命名空间结束
} // namespace at
```