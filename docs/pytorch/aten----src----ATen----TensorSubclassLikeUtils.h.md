# `.\pytorch\aten\src\ATen\TensorSubclassLikeUtils.h`

```py
#pragma once
// 预处理指令：确保头文件只被包含一次

#include <ATen/core/List.h>
// 包含 ATen 库中的 List 头文件

#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 Tensor 头文件

#include <c10/core/impl/TorchDispatchModeTLS.h>
// 包含 c10 库中的 TorchDispatchModeTLS 实现头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果未定义 AT_PER_OPERATOR_HEADERS，包含 ATen 库中的 Functions 头文件
#else
#include <ATen/ops/equal.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，包含 ATen 库中的 equal 头文件
#endif

namespace at {
// 进入 at 命名空间

// Note [Tensor-subclass-like Tensors]
// Tensor-subclass-like 的定义如下：
// - 是 Tensor 的子类（通过 Python 中的 __torch_dispatch__ 或在 C++ 中扩展 TensorImpl）
// - 其他与 Tensor 子类具有相同风险的对象。例如，许多 Tensor 子类没有存储，
//   而元 Tensor 也没有存储，因此元 Tensor 属于此类别。
//
// 我们应确保 PyTorch 内部支持 Tensor-subclass-like 对象。特别是，Tensor-subclass-like
// 对象在两类操作中存在问题，这些操作对 Tensor 子类也同样存在问题：
// 1. 因为某些 Tensor 子类没有存储，.item() 或 .data_ptr() 调用可能不合适。
// 2. 某些原地操作可能会消除 Tensor 子类的类型标记。例如：
//    >>> torch.zeros(input.sizes(), grad.options()).diag().copy_(input)
//    如果 input 是 Tensor 子类，则上述操作要么会出错，要么会返回一个常规的非 Tensor 子类 Tensor！

constexpr auto kFunctorchWrappedTensors = DispatchKeySet(
    {DispatchKey::FuncTorchGradWrapper,
     DispatchKey::FuncTorchBatched,
     DispatchKey::Functionalize});
// 定义一个包含特定调度键的常量集合，用于 Functorch 封装的 Tensor

constexpr auto kTensorSubclassLike =
    kFunctorchWrappedTensors |
    DispatchKeySet(
        {// 警告：不要将组合的后端组件 + 功能键放在此处，否则可能会始终匹配功能键，
         // 无论后端组件如何
         DispatchKey::Batched,
         DispatchKey::Sparse,
         DispatchKey::SparseCsr,
         DispatchKey::Python}) |
    DispatchKeySet(BackendComponent::MetaBit);
// 定义一个包含 Tensor-subclass-like 对象调度键的常量集合，结合了不同的调度键和后端组件

inline bool isTensorSubclassLike(const Tensor& tensor) {
  // 内联函数：检查给定的 tensor 是否属于 Tensor-subclass-like 对象
  if (c10::impl::dispatch_mode_enabled())
    // 如果调度模式已启用，返回 true
    return true;
  auto key_set = tensor.unsafeGetTensorImpl()->key_set();
  // 获取 tensor 的关键键集合
  return !(key_set & kTensorSubclassLike).empty();
  // 返回 tensor 的关键键集合与 Tensor-subclass-like 常量集合的交集是否为空
}

inline bool areAnyTensorSubclassLike(TensorList tensors) {
  // 内联函数：检查给定的 Tensor 列表中是否有任何一个属于 Tensor-subclass-like 对象
  if (c10::impl::dispatch_mode_enabled())
    // 如果调度模式已启用，返回 true
    return true;
  return std::any_of(tensors.begin(), tensors.end(), isTensorSubclassLike);
  // 使用 std::any_of 检查列表中是否有任何一个 tensor 是 Tensor-subclass-like 对象
}

inline bool areAnyOptionalTensorSubclassLike(
    const c10::List<std::optional<Tensor>>& tensors) {
  // 内联函数：检查给定的可选 Tensor 列表中是否有任何一个属于 Tensor-subclass-like 对象
  if (c10::impl::dispatch_mode_enabled())
    // 如果调度模式已启用，返回 true
    return true;
  return std::any_of(
      tensors.begin(), tensors.end(), [](const optional<Tensor>& opt_tensor) {
        return (
            opt_tensor.has_value() && isTensorSubclassLike(opt_tensor.value()));
        // 使用 lambda 函数检查列表中是否有任何一个非空且属于 Tensor-subclass-like 对象的可选 tensor
      });
}

// Helper function to deal testing truthfulness of a scalar tensor
// in a Composite Compliant manner.
// NOTE: This function expects a scalar tensor of boolean dtype.
// Eg.
// Non-Composite Compliant Pattern : (t == 0).all().item<bool>()
// Composite Compliant Patter : is_salar_tensor_true((t == 0).all())
// 辅助函数：以 Composite Compliant 方式测试标量张量的真实性。
// 内联函数，用于检查给定的张量是否为标量张量并且其值为 true
inline bool is_scalar_tensor_true(const Tensor& t) {
  // 内部断言：确保张量的维度为 0，即为标量
  TORCH_INTERNAL_ASSERT(t.dim() == 0)
  // 内部断言：确保张量的标量类型为布尔类型
  TORCH_INTERNAL_ASSERT(t.scalar_type() == kBool)
  // 返回一个布尔值，指示张量是否等于一个相同设备和数据类型的标量值为 true 的张量
  return at::equal(t, t.new_ones({}, t.options()));
}

} // namespace at
```