# `.\pytorch\c10\core\InferenceMode.h`

```
#pragma once

#include <c10/core/AutogradState.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/macros/Export.h>

namespace c10 {

// A RAII, thread local (!) guard that enables or disables inference mode upon
// construction, and sets it back to the original value upon destruction.
struct C10_API InferenceMode {
  // Note [Expected TLS state in InferenceMode]:
  //   InferenceMode: ADInplaceOrView not in
  //   raw_local_dispatch_key_set.included(),
  //                  Autograd in raw_local_dispatch_key_set.excluded()
  //                  GradMode is disabled.
  //   NormalMode: ADInplaceOrView in raw_local_dispatch_key_set.included(),
  //               Autograd not in raw_local_dispatch_key_set.excluded()
  //               GradMode is enabled by default unless toggled manually
  //               through other APIs, e.g. NoGradGuard.
  //
  // Invariant:
  // - ADInplaceOrView is never in the excluded set
  // - Autograd is never in the included set
  // - Setting InferenceMode will set GradMode accordingly, but not vice versa.
  //
  //  1. Why do we put ADInplaceOrView in included set outside InferenceMode?
  //
  //     Inplace update to inference tensor outside InferenceMode is not
  //     allowed. See Note [Inplace update inference tensor] for more details.
  //     Without going through ADInplaceOrView kernel, we cannot throw error
  //     for `inference_tensor.add_(1)` case.
  //
  // 2. Why not put ADInplaceOrView in the excluded set inside InferenceMode?
  //
  //    For example:
  //    torch::Tensor a = torch::ones({1, 2, 3}).set_requires_grad(true);
  //    torch::Tensor k = a + 2;
  //    {
  //      c10::InferenceMode guard(true);
  //      k.add_(2);
  //    }
  //    `k.add_(2)` still need to go through ADInplaceOrView kernel so that it's
  //    prepared for future autograd.
  //
  // 3. Why does setting InferenceMode also set GradMode?
  //
  //    This is required since InferenceMode is a faster and more restrictive
  //    version of NoGradGuard. All runtime checks using GradMode::is_enabled()
  //    are applicable to InferenceMode as well, e.g.
  //    `tensorTypeInCurrentExecutionContext` in interpreter.cpp.
  
  // 构造函数，根据传入的 enabled 参数设置推断模式
  InferenceMode(bool enabled = true)
      : prev_mode(AutogradState::get_tls_state()),  // 保存当前的自动求导状态
        prev_keyset(c10::impl::tls_local_dispatch_key_set()) {  // 保存当前的调度键集合
    // 根据 enabled 参数设置自动求导状态和推断模式状态
    AutogradState::set_tls_state(AutogradState(
        /* grad_mode */ !enabled,
        /* inference_mode */ enabled,
        /* fw_grad_mode */ !enabled,
        /* multithreading_enabled*/ !enabled));
    
    // 根据 enabled 参数更新调度键集合，移除或添加 ADInplaceOrView 调度键
    DispatchKeySet included = enabled
        ? prev_keyset.included_.remove(c10::DispatchKey::ADInplaceOrView)
        : prev_keyset.included_.add(c10::DispatchKey::ADInplaceOrView);
    // 根据当前是否启用，确定更新后的排除调度键集合
    DispatchKeySet excluded = enabled
        ? (prev_keyset.excluded_ | c10::autograd_dispatch_keyset)
        : (prev_keyset.excluded_ - c10::autograd_dispatch_keyset);
    // 创建当前键集合对象，用于管理本地调度键集合
    c10::impl::PODLocalDispatchKeySet cur_keyset{};
    // 设置当前键集合包含的调度键
    cur_keyset.set_included(included);
    // 设置当前键集合排除的调度键
    cur_keyset.set_excluded(excluded);
    // 强制将当前键集合设置为线程本地的调度键集合
    c10::impl::_force_tls_local_dispatch_key_set(cur_keyset);
  }

  // 析构函数，用于恢复推断模式之前的自动求导状态及调度键集合
  ~InferenceMode() {
    // 恢复线程本地存储的自动求导状态为推断模式之前的状态
    AutogradState::set_tls_state(prev_mode);
    // 恢复线程本地存储的调度键集合为推断模式之前的状态
    c10::impl::_force_tls_local_dispatch_key_set(prev_keyset);
  }

  // 静态方法，用于检查推断模式是否已启用
  static bool is_enabled();

 private:
  // 保存推断模式切换前的自动求导状态
  AutogradState prev_mode;
  // 保存推断模式切换前的本地调度键集合
  c10::impl::LocalDispatchKeySet prev_keyset;
};
// 结束命名空间 c10
} // namespace c10
```