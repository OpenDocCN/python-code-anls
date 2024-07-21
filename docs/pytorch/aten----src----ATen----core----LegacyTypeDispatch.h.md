# `.\pytorch\aten\src\ATen\core\LegacyTypeDispatch.h`

```
#pragma once

// ATen 中用于分派运算符的传统机制是 Type 对象，它本质上是一个巨大的虚拟分派表，
// 用于动态分派我们支持的每个操作。
//
// 这已经被 ATenDispatch 废弃，未来会转向 c10 分派器。
// TODO: 清理这里剩余的内容

#include <c10/core/impl/LocalDispatchKeySet.h>

namespace at {

// 一个 RAII 风格的、线程局部的 (!) 守卫，用于禁用对变量处理器的分派。
//
// NOTE [ 在类型分派中将变量视为非变量 ]
//
// AutoDispatchBelowAutograd 的作用是什么？简短的答案是，它导致 ATen 函数的分派
// 走向非变量实现，绕过自动求导处理（以及分析和跟踪）。
//
// 要理解为什么需要这个守卫，有助于了解 Variable 实现背后的历史。
// 以前，Variables 是在 Tensors 的包装器上实现的；因此处理 Variable 涉及解包底层的 Tensor，
// 然后在 /那个/ 操作上调用底层基础操作。
//
// 然而，在 Variable/Tensor 合并后，不再存在解包张量的概念了。
// 如果在 VariableType 处理程序内再次调用相同的变量操作，会再次分派到 VariableType，
// 这不是我们想要的结果。
//
// 上述问题的解决方案是添加 `at::AutoDispatchBelowAutograd`，启用它将导致 `legacyTensorType()`
// 和 `getType()` 始终返回非变量类型，即使在调用它们的张量是一个变量时也是如此。

/* Note [AutoDispatchBelowAutograd]
 * AutoDispatchBelowAutograd 是 **仅限内部使用** 的，应用于内核实现和定制的 C++ 内核。
 * 如果你正在寻找一个在推理模式下运行工作负载的守卫，请使用用户可见的 c10::InferenceMode RAII。
 * 在过去，用户代码中曾使用 AutoDispatchBelowAutograd（或其旧版本 AutoNonVariableTypeMode）
 * 来处理仅推理工作负载，这在某些边缘情况下潜在风险，可能会静默产生错误结果。例如：
 * ```
 *  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
 *  torch::Tensor out = s * s;
 *  {
 *    at::AutoDispatchBelowAutograd guard;
 *    s.add_(1);  // 跳过对 `s` 的版本更新。
 *  }
 *  // 错误的梯度！s.grad() 现在是使用 `s` 在原地更新后的值计算的。
 *  out.backward(torch::ones_like(out));
 * ```
 * 用户应该在这里使用 `c10::InferenceMode`，这样它会正确地抛出错误，说 "用于梯度计算的变量之一已被修改"。
 */
struct TORCH_API AutoDispatchBelowAutograd {
  // 构造函数，初始化 autograd_guard_ 为 c10::autograd_dispatch_keyset
  AutoDispatchBelowAutograd() :
    autograd_guard_(c10::autograd_dispatch_keyset) {
  }

  // 禁用所有自动求导分派键
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

// TODO: AutoNonVariableTypeMode 应该在发布 1.10 版本中移除。
// 定义一个结构体 AutoNonVariableTypeMode，用于管理非变量类型模式
struct TORCH_API AutoNonVariableTypeMode {
  // 构造函数，可选择启用或禁用非变量类型模式，默认启用
  AutoNonVariableTypeMode(bool enabled = true) :
    // 初始化 autograd_guard_，排除所有 autograd 分发键
    autograd_guard_(c10::autograd_dispatch_keyset) {
    // 发出一次性警告，表明 AutoNonVariableTypeMode 已弃用，并将在 1.10 版本中移除
    TORCH_WARN_ONCE("AutoNonVariableTypeMode is deprecated and will be removed in 1.10 release. "
        "For kernel implementations please use AutoDispatchBelowADInplaceOrView instead, "
        "If you are looking for a user facing API to enable running your inference-only "
        "workload, please use c10::InferenceMode. Using AutoDispatchBelowADInplaceOrView in user code "
        "is under risk of producing silent wrong result in some edge cases. "
        "See Note [AutoDispatchBelowAutograd] for more details.");
    // 内部断言，确保 enabled 参数为 true
    TORCH_INTERNAL_ASSERT(enabled);
  }

  // 禁用所有 autograd 分发键的守卫对象
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

// 定义一个结构体 AutoDispatchSkipFunctionalize
struct TORCH_API AutoDispatchSkipFunctionalize {
  // 构造函数，初始化 dispatch_key_guard_，排除 Functionalize 分发键
  AutoDispatchSkipFunctionalize() :
    dispatch_key_guard_(c10::DispatchKeySet(c10::DispatchKey::Functionalize)) {
  }
  // 禁用 Functionalize 分发键的守卫对象
  c10::impl::ExcludeDispatchKeyGuard dispatch_key_guard_;
};

/* Note [AutoDispatchBelowADInplaceOrView]
 * AutoDispatchBelowADInplaceOrView 等同于在将 inplace 和 view 操作从 VariableType 核心中分离出来之前的 AutoNonVariableTypeMode。
 * 注意，此守卫在 VariableType 核心中用于功能操作以及 ADInplaceOrView 核心中用于 inplace/view 操作，以强制执行不变量：
 *   一旦进入某个操作的 VariableType/ADInplaceOrView 核心，
 *   直到完成当前操作，不会再返回到同一分发键的其他核心。
 */
struct TORCH_API AutoDispatchBelowADInplaceOrView {
  // 构造函数，初始化 dispatch_key_guard_，排除 Autograd 和 ADInplaceOrView 分发键
  AutoDispatchBelowADInplaceOrView() :
    dispatch_key_guard_(c10::autograd_dispatch_keyset_with_ADInplaceOrView) {
  }
  // 禁用 Autograd 和 ADInplaceOrView 分发键的守卫对象
  c10::impl::ExcludeDispatchKeyGuard dispatch_key_guard_;
};
// 命名空间结束
} // namespace at
```