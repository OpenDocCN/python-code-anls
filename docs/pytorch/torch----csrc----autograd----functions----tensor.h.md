# `.\pytorch\torch\csrc\autograd\functions\tensor.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/Export.h>
// 导入 Torch 库导出相关功能

#include <torch/csrc/autograd/function.h>
// 导入 Torch 自动求导功能中的函数定义

#include <torch/csrc/autograd/variable.h>
// 导入 Torch 自动求导功能中的变量定义

#include <ATen/TensorGeometry.h>
// 导入 ATen 库中与张量几何相关的功能

#include <ATen/core/DeprecatedTypeProperties.h>
// 导入 ATen 库中已弃用的类型属性功能

#include <c10/util/Optional.h>
// 导入 C10 库中的可选类型支持

#include <cstdint>
// 导入标准整数类型支持

#include <memory>
// 导入内存管理相关功能

namespace torch {
namespace autograd {

struct TORCH_API CopyBackwards : public Node {
  // CopyBackwards 结构体，继承自 Node

  variable_list apply(variable_list&& grads) override;
  // 重写 Node 类的 apply 方法，接收变量列表并返回变量列表

  void compiled_args(CompiledNodeArgs& args) override;
  // 重写 Node 类的 compiled_args 方法，接收编译后的节点参数对象的引用

  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  // 重写 Node 类的 apply_with_saved 方法，接收输入变量列表和保存的交换变量对象的引用

  at::TensorOptions src_options;
  // 源张量选项对象

};

// Note [View + Inplace update for base tensor]
//
// This note covers a few important topics related to view + inplace handling.
//   - It explains what is the CopySlices Node and why we need it.
//   - It explains the considerations on what is saved for backward in
//   CopySlices.
//   - It explains why we need to sometimes change the exec_info of the current
//   backward
//
// What is CopySlices?
// ~~~~~~~~~~~~~~~~~~~
//
// We support autograd with inplace mutation; e.g., if you write x.mul_(2)
// the autograd will work as if you now had multiple Tensors under the hood and
// you did
//   x = t.clone()
//   x0 = x
//   x1 = x0 * 2
//   x = x1
// As you can see here, after this operation, x.grad_fn now points to x1.grad_fn
// (the MulBackward node) and this node points to x's original grad_fn (which is
// also x0.grad_fn). It is important to keep in mind that after the inplace,
// there is no Tensor object that represents the x0 state anymore. But the graph
// for it is still around in autograd (in case x was used before being modified
// inplace). See Example 1 in
// https://docs.google.com/drawings/d/1-T5DyYfChMX1ONQkY-zU-hj_ayQ2zmA5CBOKDWqvEhE
// We call this rebasing the history of the Tensor.
//
// Now, a difficult situation is what happens if x is a differentiable view
// of a base b.
//   b = t.clone()
//   x = b.select(0, 0)
//   x *= 2
// With the same approach as above, this will become
//   b = t.clone()
//   x = b.select(0, 0)
//   b0 = b
//   x0 = x
//   x1 = x0 * 2
//   b1 = b0.select_scatter(x1, 0, 0)
//   x2 = b1.select(0, 0)
//   x = x2
//   b = b1
// As you can see here, not only we need to modify x's grad_fn, we also need to
// modify the one from b. We also need to ensure that the new grad_fn on x is
// linked to b's new grad_fn. The chain the select_scatter, multiplication and
// select is what CopySlices does, all wrapped into a single Node.
//
// See Example 1 in
// https://docs.google.com/drawings/d/1-T5DyYfChMX1ONQkY-zU-hj_ayQ2zmA5CBOKDWqvEhE
//
// What do we need to save in CopySlices to run backward?
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// We need to perform grad_view = fn(grad_view), but out-of-place.
// view_fn_ is an optional function saved in DifferentiableViewMeta
// from forward pass, so that we can recover we when as_strided is not
// supported. It preserves the invariants:
//   view = view_fn_(base)
//   grad_view = view_fn_(grad_base)
//
// When as_strided is supported (e.g. strided CPU/CUDA Tensors), view_fn_
// is empty and we save TensorGeometry(view) instead.
// With the TensorGeometry information we can use `as_strided` call which
// is more efficient to recover views in backward.
//
// For example:
//   view_1 = view_op_1(base)
//   view_2 = view_op_2(view_1)
//   ...
//   view_n = view_op_n(view_n-1)
//   view_n = inplace_op(view_n)
//
// In CPU/CUDA case where we support efficient as_strided implementation,
// grad_view_n can be calculated through 1 step.
//
//   grad_view_n = grad_base.as_strided(view_sizes, view_strides, view_offset);
//
// But in XLA backend where we don't have full support of as_strided,
// it has to save a chained lambda function view_fn_, to exactly
// replay how the view was done in forward.
//
//   view_fn_ = view_op_n(...(view_op_2(view_op_1())))
//   grad_view_n = view_fn_(grad_base)
//
// This chain view_fn_ works as long as forward view ops are implemented,
// e.g XLA simulates view without a real Storage behind Tensor, but it's less
// efficient than the as_strided one so we should be careful to only use it when
// necessary.
//
//   - For CPU/CUDA we save TensorGeometry of both base and view tensors,
//     That's all we need to pass into as_strided.
//     E.g. int[] sizes, int[] strides, and int storage_offset.
//   - For XLA we use view_fn_, which captures all forward view op arguments
//     by **value**.
//     E.g for at::narrow, int dim, int start, in length are saved.
//
// Theoretically we could also save Tensor `view` in CopySlices Node, but
// it's far more expensive than what we currently save.
//   1. We cannot afford keeping large tensors alive to recover views only.
//   2. There are inplace checks when Tensors are loaded back to make sure
//      they haven't been changed (including size metadata).
// So saving metadata like TensorGeometry/view arguments is much better
// because it is minimal information needed to recover views, as well as it
// allows the user to modify the original Tensor without preventing the
// backward pass from running.
//
// Why do we manually change exec_info in the apply?
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Using the same example as before,
//   b = t.clone()
//   x = b.select(0, 0)
//   x *= y
//
// You can see the visualization at
// https://docs.google.com/drawings/d/1Bx-Hcz-zlIv7PabQqnPhUIVIs9F8WWi48svqMsAUMFs
// which contains the wrapped MulBackward Node and show what it links to.
// Since a backward can happen between any subset of the inputs (t and y) and
// outputs (o, x, b). It is possible to get into a state where CopySlices's 0th
// next function (CloneBackward) needs gradient but MulBackward's 0th next
// function (SelectBackward) is not. This happens if you do autograd.grad
// between x and t for example.
// 将 SelectBackward 标记为需要梯度，以便在执行 MulBackward 时实际计算第0个输入的梯度。
//
// 对于所有其他的 next functions，它们始终是共享的（在 apply 代码中进行了断言），因此它们不需要额外的操作。

// 查看当一个就地操作发生时，我们对视图张量进行的视图+就地更新的操作，参见注释 [View + Inplace update for view tensor]。
struct TORCH_API CopySlices : public Node {
  CopySlices(
      const Variable& base_var,        // 基本变量
      at::TensorGeometry view_,        // 视图张量的几何属性
      std::unique_ptr<ViewFunc> view_fn_,  // 视图函数的唯一指针
      std::shared_ptr<Node> fn_);      // 共享节点的指针

  // 在 apply/apply_with_saved 之间的通用代码
  template <typename T>
  variable_list apply_impl(variable_list&& inputs, const T& call_fn);

  // 应用操作到输入上，返回变量列表
  variable_list apply(variable_list&& inputs) override;

  // 释放变量
  void release_variables() override;

  // 编译参数
  void compiled_args(CompiledNodeArgs& args) override;

  // 应用操作到带有保存的输入上，使用 SwapSavedVariables
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  at::TensorGeometry base;   // 基础张量的几何属性
  // view 和 view_fn 是冗余的，如果可用，将使用 view_fn。
  // 参见注释 [View + Inplace update for base tensor] 获取详细信息。
  at::TensorGeometry view;   // 视图张量的几何属性
  std::unique_ptr<ViewFunc> view_fn;   // 视图函数的唯一指针
  std::shared_ptr<Node> fn;   // 共享节点的指针
};

} // namespace autograd
} // namespace torch
```