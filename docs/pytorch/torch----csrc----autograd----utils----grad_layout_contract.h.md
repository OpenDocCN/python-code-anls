# `.\pytorch\torch\csrc\autograd\utils\grad_layout_contract.h`

```
#pragma once

#include <ATen/Tensor.h>

namespace torch {
namespace autograd {
namespace utils {

// Helper functions to enforce the "Gradient Layout Contract" described in
// torch/csrc/autograd/functions/accumulate_grad.h.

// Checks if grad obeys the contract with variable.
inline bool obeys_layout_contract(
    const at::Tensor& grad,
    const at::Tensor& variable) {
  // 断言 grad 不是稀疏张量
  TORCH_INTERNAL_ASSERT(!grad.is_sparse());
  // 断言 grad 不是稀疏 CSR 格式张量
  TORCH_INTERNAL_ASSERT(!grad.is_sparse_csr());
  // 断言 variable 不是稀疏 CSR 格式张量
  TORCH_INTERNAL_ASSERT(!variable.is_sparse_csr());

  // NOLINTNEXTLINE(bugprone-branch-clone)
  // 如果 variable 是嵌套张量
  if (variable.is_nested()) {
    // TODO: 嵌套张量目前没有 detach 的实现。当前的嵌套张量实现可能确实遵循梯度约定，应返回 true，但这可能会在未来改变
    return false;
  } else if (variable.is_sparse()) {
    // 稀疏布局不适用于梯度布局约定
    return false;
  } else if (variable.is_non_overlapping_and_dense()) {
    // 仅检查非重叠且密集张量的布局约定
    const auto& grad_sizes = grad.sym_sizes();
    const auto& grad_strides = grad.sym_strides();
    const auto& variable_strides = variable.sym_strides();
    for (const auto idx : c10::irange(grad_sizes.size())) {
      if (grad_sizes[idx] != 1) {
        // 对于大小不为 1 的维度，只考虑步长
        if (grad_strides[idx] != variable_strides[idx]) {
          return false;
        }
      } else {
        // 这应该不需要，但我们在将张量存储在内部之前没有检查张量是否有视图。
        // 对于大小为 1 的 0 步长张量实际上是像 cat 这样操作的视图
        // TODO: 在 accumulateGrad 函数中实际检测视图，以便完全不考虑此张量
        if (grad_strides[idx] == 0) {
          return false;
        }
      }
    }
    return true;
  } else {
    // 如果 grad 是连续的，并符合指定的内存格式，返回 true
    return grad.is_contiguous(at::MemoryFormat::Contiguous);
  }
}

// Creates a clone of new_grad that obeys the contract with variable.
// The clone should attach to new_grad's history if GradMode::is_enabled().
inline at::Tensor clone_obey_contract(
    const at::Tensor& new_grad,
    const at::Tensor& variable) {
  if (variable.is_non_overlapping_and_dense()) {
    // (1)
    // 这个看起来有点冒险的序列是否在 GradMode::is_enabled() 时将结果附加到 new_grad 的历史记录？ 是的，@alband 表示应该这样做。
    return std::move(new_grad
                         .new_empty_strided_symint(
                             variable.sym_sizes(),
                             variable.sym_strides(),
                             variable.options().memory_format(c10::nullopt))
                         .copy_(new_grad));
  } else {
    // (2)
    // 创建一个符合指定内存格式的 new_grad 的克隆张量
    return new_grad.clone(at::MemoryFormat::Contiguous);
  }
}

} // namespace utils
} // namespace autograd
} // namespace torch
```