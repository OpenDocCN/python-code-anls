# `.\pytorch\aten\src\ATen\TracerMode.h`

```py
#pragma once
// 一次性指令，确保头文件只被包含一次

#include <c10/core/impl/LocalDispatchKeySet.h>
// 引入 LocalDispatchKeySet.h，用于本地分发键集合
#include <c10/macros/Export.h>
// 引入 Export.h，包含了导出宏定义
#include <c10/macros/Macros.h>
// 引入 Macros.h，包含了各种宏定义

// NOTE [Tracing Mode Switches]
//
// Historically, tracing function was controlled by two switches:
//
// - `AutoDispatchBelowADInplaceOrView` guard
//
//    历史上，跟踪功能由两个开关控制：
//
//    - `AutoDispatchBelowADInplaceOrView` 守卫
//
//       跟踪函数曾经在 `VariableType_*.cpp` 中由脚本生成，与自动微分函数共享
//       相同的 `Autograd` 分发键。因此，在将跟踪功能从 VariableType 中移出之前，
//       `AutoDispatchBelowADInplaceOrView` 守卫还可以通过禁用 `Autograd` 分发来
//       禁用跟踪。
//
// - `setTracingState()` API in `torch/csrc/jit/frontend/tracer.h`
//
//    它在 TLS 中存储跟踪数据于 `TracingState` 对象中。如果 TLS 中的 `TracingState`
//    对象为 `null`，则跟踪被暂停。
//
//    `TracingState` 对象在 `tracer::trace()` 中创建，是跟踪函数的主要入口。它在
//    生成的 VariableType（现在是 TraceType）中临时设置为 `null`，以跳过中间操作
//    的跟踪（被其他操作调用的操作）。中间操作调用完成后，它会被设置回原始的
//    `TracingState` 对象。
//
//    `TracingState` 对象在 TLS 中也可以通过其在 `python_tracer.cpp` 中的 Python
//    绑定和 `get/setTracingState()` C++ API 读取/写入，这些 API 也暴露为 `TORCH_API`。
//
// 自从跟踪函数从 VariableType 中移出后，引入了两个新的开关：
//
// - `tracer::impl::set_dispatch_enabled()` API
//
//    不同于默认包含在分发键集中的特殊 `Autograd` 分发键，`Tracer` 分发键默认为关闭。
//    可以通过这个新的 API 切换分发开关。
//
// - `tracer::impl::NoTracerDispatchMode` 守卫
//
//    它用于覆盖跟踪从 VariableType 移出后的 `AutoDispatchBelowADInplaceOrView` 的旧语义。
//
// 在将跟踪函数从 VariableType 中移出之前，满足以下条件时启用跟踪：
//
//    1) TLS 中的 `TracingState` 对象不为 `null`；
//       - 要么在 `tracer::trace()` 的执行范围内，或者
//       - 调用 `setTracingState()` 时传入非 `null` 对象。
//    2) 不在 `AutoDispatchBelowADInplaceOrView` 的作用域内；
//
// 之后：
//
//    1) TLS 中的 `TracingState` 对象不为 `null`；
//    2) 调用了 `tracer::impl::set_dispatch_enabled(true)`；
//    3) 不在 `tracer::impl::NonDispatchGuard` 的作用域内；
//
// [TODOs]
//
// - `setTracingState()` v.s. `tracer::impl::set_dispatch_enabled()`
//
//   目前 `set_dispatch_enabled()` 在 `setTracingState()` 中设置/取消设置，
//   以保持与以前完全相同的语义 - 同时保留这两个开关是令人困惑的。我们应该考虑简化/
//   限制暴露的 `setTracingState()` Python/C++ API（以及调用它的其他 API），以便
//   可以统一这两者。
//
// - `AutoDispatchBelowADInplaceOrView` v.s.
// `tracer::impl::NoTracerDispatchMode`
//
//   We don't need to always set both guards together to keep semantics
//   unchanged. For the following use cases of `AutoDispatchBelowADInplaceOrView`,
//   we don't need to set the new tracer guard:
//
//   * Script-generated VariableType kernels. The guard is not necessary as
//     tracing is already disabled explicitly by `setTracingState(null)` in
//     generated TraceType kernels - we could keep it as is or use the new guard
//     instead.
//
//   * Custom ops. Will be handled by fallback kernel for `Tracer`.
//
//   * Functions that are not likely to be called in a tracing context (no Python
//     binding / not an operator), e.g., all mobile forward() wrappers, test
//     binaries, and etc.
//
//   * Where new threads are spawned, e.g., ATen/native/ConvolutionMM2d.cpp.
//     It's not necessary as tracing is off by default.
//
//   For the rest of cases, we might need to have both:
//
//   * Functions that might be reachable from eager mode Python (especially
//     factory methods), e.g.:
//     `internal_new_from_data()` in `torch/csrc/utils/tensor_new.cpp`.
//     Without the new guard, it will add `aten::empty` to the traced graph.
//
//   * Some manually maintained functions, e.g.:
//     `torch/csrc/autograd/VariableTypeManual.cpp`.
//     Set the new guard if it's not obvious whether `setTracingState(null)`
//     has been called before it reaches the `AutoDispatchBelowADInplaceOrView`
//     guard.
//
//   We might need to tweak the usage of the new guard to optimize/fix things.
//   It should only affect the correctness of the tracing function because the
//   guard is essentially a no-op when the master `setTracingState()` switch is
//   off.
//
// TODO: move this from `at::` to `jit::torch::` after
// `aten/src/ATen/cpp_custom_type_hack.h` is removed.
//
// Defines the namespace `at::tracer::impl` which contains utility functions and
// structures related to tracing.
namespace at::tracer::impl {

// Defines a static inline function `is_dispatch_enabled()` which checks whether
// tracing dispatch is enabled by verifying that `Tracer` dispatch key is included
// and not excluded.
static inline bool is_dispatch_enabled() {
  return c10::impl::tls_is_dispatch_key_included(at::DispatchKey::Tracer) &&
      !c10::impl::tls_is_dispatch_key_excluded(at::DispatchKey::Tracer);
}

// Defines a static inline function `set_dispatch_enabled()` which enables or
// disables tracing dispatch based on the `enabled` parameter. It internally checks
// and asserts that tracing is not being enabled within the scope of
// `NoTracerDispatchMode`.
static inline void set_dispatch_enabled(bool enabled) {
  TORCH_INTERNAL_ASSERT(
      !c10::impl::tls_is_dispatch_key_excluded(at::DispatchKey::Tracer),
      "Cannot enable tracing within the scope of NoTracerDispatchMode!");
  c10::impl::tls_set_dispatch_key_included(at::DispatchKey::Tracer, enabled);
}

// Defines a structure `NoTracerDispatchMode` which includes an
// `ExcludeDispatchKeyGuard` guard to exclude `Tracer` dispatch key from dispatch.
struct NoTracerDispatchMode {
  c10::impl::ExcludeDispatchKeyGuard guard_{at::DispatchKey::Tracer};
};

} // namespace at::tracer::impl
```