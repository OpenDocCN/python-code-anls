# `.\pytorch\aten\src\ATen\ThreadLocalState.cpp`

```py
// 包含 ATen 库中的 ThreadLocalState.h 头文件

#include <ATen/ThreadLocalState.h>

// 如果不是 XPLAT 构建并且不是 C10_MOBILE，才包含 grad_mode.h 头文件
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/grad_mode.h>
#endif

// 包含 ATen 库中的 record_function.h、SavedTensorHooks.h 和 FunctionalTensorWrapper.h 头文件
#include <ATen/record_function.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/FunctionalTensorWrapper.h>

// ATen 命名空间
namespace at {

// 构造函数 ThreadLocalState 的定义
ThreadLocalState::ThreadLocalState()
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()), // 初始化 dispatch_key_ 使用 c10 的本地调度键设置函数
      debug_info_(c10::ThreadLocalDebugInfo::current()), // 初始化 debug_info_ 使用当前线程的调试信息
      rf_tls_(at::get_record_function_tls_()), // 初始化 rf_tls_ 使用 ATen 的记录函数线程本地存储
      functorch_tls_(functorch::getCopyOfFuncTorchTLS()), // 初始化 functorch_tls_ 使用 functorch 的功能 Torch 线程本地存储的副本
      autograd_tls_(c10::AutogradState::get_tls_state()), // 初始化 autograd_tls_ 使用 c10 的自动求导状态的线程本地存储
      torch_dispatch_mode_state_(c10::impl::TorchDispatchModeTLS::get_state()), // 初始化 torch_dispatch_mode_state_ 使用 c10 的 Torch 调度模式 TLS 状态
      python_dispatcher_state_(c10::impl::PythonDispatcherTLS::get_state()), // 初始化 python_dispatcher_state_ 使用 c10 的 Python 分发器 TLS 状态
      python_torch_function_state_(at::impl::PythonTorchFunctionTLS::get_state()), // 初始化 python_torch_function_state_ 使用 ATen 的 Python Torch 函数 TLS 状态
      saved_tensors_default_hooks_state_(at::SavedTensorDefaultHooks::get_tls_state()), // 初始化 saved_tensors_default_hooks_state_ 使用 ATen 的保存张量默认钩子线程本地存储
      functionalization_reapply_views_state_(at::functionalization::impl::getFunctionalizationReapplyViewsTLS()), // 初始化 functionalization_reapply_views_state_ 使用 ATen 的功能化重应用视图 TLS 状态
      saved_objects_(at::impl::ThreadLocalPythonObjects::get_state()) {} // 初始化 saved_objects_ 使用 ATen 的线程本地 Python 对象状态

// 设置梯度模式的成员函数定义
void ThreadLocalState::set_grad_mode(bool enabled) {
  autograd_tls_.set_grad_mode(enabled); // 使用 autograd_tls_ 设置自动求导的梯度模式
}

// 设置多线程启用状态的成员函数定义
void ThreadLocalState::set_multithreading_enabled(bool enabled) {
  autograd_tls_.set_multithreading_enabled(enabled); // 使用 autograd_tls_ 设置自动求导的多线程启用状态
}

// 静态函数 setThreadLocalState 的定义，设置线程本地状态
/* static */
void ThreadLocalState::setThreadLocalState(
    const ThreadLocalState& state) {
  // 注意，只有在这个函数中设置推理模式 TLS 是允许的，因为我们总是同时恢复调度键集 TLS
  c10::AutogradState::set_tls_state(state.autograd_tls_); // 设置自动求导状态的 TLS

  c10::impl::TorchDispatchModeTLS::set_state(state.torch_dispatch_mode_state_); // 设置 Torch 调度模式 TLS 的状态

  at::impl::PythonTorchFunctionTLS::set_state(state.python_torch_function_state_); // 设置 Python Torch 函数 TLS 的状态

  at::set_record_function_tls_(state.rf_tls_); // 设置记录函数 TLS

  at::SavedTensorDefaultHooks::set_tls_state(state.saved_tensors_default_hooks_state_); // 设置保存张量默认钩子 TLS 的状态

  c10::impl::PythonDispatcherTLS::set_state(state.python_dispatcher_state_); // 设置 Python 分发器 TLS 的状态

  c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_); // 强制设置当前调试信息

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_); // 强制设置本地调度键集

  functorch::setFuncTorchTLS(state.functorch_tls_); // 设置 functorch 的功能 Torch TLS

  at::functionalization::impl::setFunctionalizationReapplyViewsTLS(state.functionalization_reapply_views_state_); // 设置功能化重应用视图 TLS

  at::impl::ThreadLocalPythonObjects::set_state(state.saved_objects_); // 设置线程本地 Python 对象的状态
}

} // namespace at
```