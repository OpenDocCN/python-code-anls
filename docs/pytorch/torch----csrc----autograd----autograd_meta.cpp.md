# `.\pytorch\torch\csrc\autograd\autograd_meta.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，限定仅允许方法操作符

#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
// 引入头文件，包括C10库的irange和torch自动求导的variable

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 如果未定义AT_PER_OPERATOR_HEADERS宏，则引入ATen的功能函数头文件
#else
#include <ATen/ops/_has_same_storage_numel.h>
#include <ATen/ops/_new_zeros_with_same_feature_meta.h>
#include <ATen/ops/zeros.h>
// 如果定义了AT_PER_OPERATOR_HEADERS宏，则引入ATen的其他操作头文件
#endif

namespace torch {
namespace autograd {

using at::Tensor;
// 使用ATen命名空间中的Tensor类型

// [Forward Grad View/inplace]
// 前向梯度视图/原地操作
// 重要的是允许视图和原地操作与双重张量一起工作。
// 这些操作应该计算正确的梯度或引发用户友好的错误信息。

// 当所有张量都是双重张量时，基本情况如下：
//     # 有：
//     #   foo是一个不是视图的双重张量
//     #   bar是一个适当大小的不是视图的双重张量（取决于情况）
//
//     # 情况1：无视图
//     foo.copy_(bar)
//
//     # 情况2：有视图，从视图传播到基础张量
//     view = foo[0]
//     view.copy_(bar)
//
//     # 情况3：有视图，从基础张量传播到视图
//     view = foo[0]
//     foo.copy_(bar)
//
//     # 在所有情况下，foo的前向梯度必须得到适当更新。
//     # 在第二和第三种情况下，视图的前向梯度必须与它们共同部分的foo的一致。
//
// 所有这些情况都可以通过前向梯度上的以下布局约束来处理：
//   - 一个张量及其前向梯度（对所有级别）必须具有相同的元数据（大小、步幅、共轭/负位和存储偏移）。
//   - 视图操作必须创建一个前向梯度，该前向梯度是基础前向梯度的视图。
//   - 原地操作必须原地修改输入的前向梯度。
//
// 此布局约束在下面的set_fw_grad函数中得到保证。

// 当非双重张量与双重张量交互时，会出现更复杂的情况。
// 最重要的两种情况是：
//
//     # 有：
//     #   foo是一个不是视图的常规张量
//     #   bar是一个适当大小的不是视图的双重张量（取决于情况）
//
//     # 情况4：对视图的更改必须传播到其基础张量
//     view = foo[0]
//     # 此时view仍然是常规张量
//     view.copy_(bar)
//     # 现在view和foo都是适当的双重张量，具有相应的前向梯度
//
//     # 情况5：对基础张量的更改必须传播到所有其视图
//     view = foo[0]
//     # 此时view仍然是常规张量
//     base.copy_(bar)
//     # 现在view和foo都是适当的双重张量，具有相应的前向梯度
//
//     # 注意：还有一个情况6涉及对视图的更改传播到其他视图，但它完全由上述两种情况描述，并在本讨论中被跳过。
//
// 情况4由set_fw_grad函数处理，通过适当设置前向梯度来处理视图的前向梯度。
// base if needed. Case 5 is handled in fw_grad by reading the forward grad from
// the base if needed.

namespace utils {

// Enforcing that the metadata between the primal and tangent are same has two
// goals:
// - When properties of the primal are checked in composite op's to determine
//   control flow, the code path decided upon is also reasonable for the tangent
// - Make sure that when the same as_strided is applied to both primal and
//   and tangent, it behaves similarly.
//
// We do that by checking:
//   1) the storages have same properties: size and conj/neg-ness
//   2) the same indices refer to the same elements in storage
//      (we are more strict than necessary here to satisfy the goal 1)
bool has_same_meta(const Variable& base, const Variable& other) {
  if (!base.defined() || !other.defined()) {
    return false;
  }
  // 1) The storages have the same properties
  if (!at::_has_same_storage_numel(base, other)) {
    return false;
  }
  if (base.is_conj() != other.is_conj() || base.is_neg() != other.is_neg()) {
    return false;
  }

  // Technically dim and size belong as part of (2), so we shouldn't really care
  // if a zero-numel tensor violates these. But since these properties
  // (unlike offset and strides) often determine control flow in composite ops
  // it is useful to enforce that they match for primal and tangent here so
  // nothing funny happens later (See goal 1).
  if (base.dim() != other.dim()) {
    return false;
  }
  for (const auto i : c10::irange(base.dim())) {
    if (base.sym_sizes()[i] != other.sym_sizes()[i]) {
      return false;
    }
  }

  // The check below will always be vacuously true for 0-element tensors
  if (base.sym_numel() == 0 && other.sym_numel() == 0) {
    return true;
  }

  // 2) The same indices refer to the same elements in storage
  if (base.sym_storage_offset() != other.sym_storage_offset()) {
    return false;
  }

  for (const auto i : c10::irange(base.dim())) {
    if (base.sym_strides()[i] != other.sym_strides()[i] &&
        base.sym_sizes()[i] != 1 && base.sym_sizes()[i] != 0) {
      return false;
    }
  }
  return true;
}

} // namespace utils

// This function is will ensure that the fw_grad_ is properly a view of the base
// for inplace ops on Tensors that do not have forward grad originally.
void AutogradMeta::set_fw_grad(
    const at::TensorBase& new_grad_base,
    const at::TensorBase& self_base,
    uint64_t level,
    bool is_inplace_op) {
  TORCH_CHECK(
      !new_grad_base._fw_grad(level).defined(),
      "Setting a forward grad that "
      "itself has a forward gradient at the same level",
      level,
      " is not supported.");
  TORCH_INTERNAL_ASSERT(
      (new_grad_base.is_floating_point() || new_grad_base.is_complex()) &&
          (self_base.is_floating_point() || self_base.is_complex()),
      "Expected both tensor and its forward grad to be floating point or complex");
  // Lazy initialization
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // 如果 fw_grad_ 为空指针，则创建一个新的 ForwardGrad 对象并赋给 fw_grad_
    if (!fw_grad_) {
      fw_grad_ = std::make_shared<ForwardGrad>();
    }
  }
  // 如果 fw_grad_ 中已经包含了指定的 level
  if (fw_grad_->contains(level)) {
    // 设置 forward grad 只允许在它是一个空操作的情况下重新设置
    // 允许这种情况是为了简化对原地操作生成代码的编写
    TORCH_INTERNAL_ASSERT(
        new_grad_base.defined(),
        "Cannot set a forward grad that is an undefined Tensor. Use "
        "_fw_primal(level) to get a new Tensor with this forward grad unset.");

    TORCH_INTERNAL_ASSERT(
        is_inplace_op,
        "Only inplace operations can re-set the forward grad of a Tensor that "
        "already has one.");

    TORCH_INTERNAL_ASSERT(
        fw_grad_->value(level).is_same(new_grad_base),
        "Cannot set a value of a forward grad if it "
        "already exists. Inplace operations should modify it inplace.");
  } else {
    // TODO(alband) remove this spurious version counter bump
    // 创建一个新的 Tensor 对象 new_grad，并使用 new_grad_base 初始化它
    Tensor new_grad(new_grad_base);
    // 创建对 self_base 的 OptionalTensorRef 引用，并将其赋给 self_ref
    at::OptionalTensorRef self_ref(self_base);
    // 使用 self_ref 获取 self 对象的引用
    const Tensor& self = *self_ref;

    // 检查 new_grad 和 self 的尺寸是否相同
    TORCH_CHECK(
        self.is_same_size(new_grad),
        "Trying to set a forward gradient that has a different size than that "
        "of the original Tensor, this is not supported. Tensor is of size ",
        self.sizes(),
        " while the given "
        "forward gradient is of size ",
        new_grad.sizes(),
        ".");
    // 检查是否为原地操作且当前对象为视图
    if (is_inplace_op && is_view_) {
      auto this_view_meta = static_cast<DifferentiableViewMeta*>(this);

      // 对于没有前向梯度的原地操作视图，将梯度传播到基本张量，并确保新的梯度也是该基本张量梯度的视图。
      // 这保证了上面提到的 [Forward Grad View/inplace] 的第四种情况能够正常工作。
      // 主要操作如下：
      //   - 检查基本张量是否已经有梯度
      //   - 如果没有，为其设置一个全零的前向梯度
      //   - 获取基本张量的前向梯度的视图
      //   - 将给定的新梯度复制到该视图中
      //   - 使用该视图作为新的 new_grad
      if (this_view_meta->has_fw_view()) {
        auto& view_info = this_view_meta->get_forward_view();
        auto& base = view_info.base_;

        if (!base._fw_grad(level).defined()) {
          // 在此处强制使用相同的元数据，以确保以下视图操作始终有效
          Tensor new_base_fw_grad;
          if (utils::has_same_meta(new_grad, base) &&
              utils::has_same_meta(new_grad, self)) {
            // TODO: 扩展到当 new_grad 的基础存储可以被重用时的特殊情况。
            new_base_fw_grad = new_grad;
          } else {
            new_base_fw_grad =
                at::_new_zeros_with_same_feature_meta(new_grad, base);
            new_base_fw_grad._set_conj(base.is_conj());
            new_base_fw_grad._set_neg(base.is_neg());

            // 更新 new_grad 为基本张量的视图
            Tensor new_fw_grad_value;
            if (view_info.has_view_fn()) {
              new_fw_grad_value = view_info.view_fn()(new_base_fw_grad);
            } else {
              new_fw_grad_value = new_base_fw_grad.as_strided(
                  self.sizes(), self.strides(), self.storage_offset());
            }

            new_fw_grad_value.copy_(new_grad);
            new_grad = new_fw_grad_value;
          }

          base._set_fw_grad(new_base_fw_grad, level, /* is_inplace_op */ false);
        }
      }
    }

    // 强制执行基本布局约束
    // 如果新梯度和当前对象的元数据不相同
    if (!utils::has_same_meta(new_grad, self)) {
      if (is_view_) {
        auto this_view_meta = static_cast<DifferentiableViewMeta*>(this);
        // 断言：预期前向可微视图操作的输出，其切线与原始张量具有相同的布局
        TORCH_INTERNAL_ASSERT(
            !this_view_meta->has_fw_view(),
            "Expected the output of forward differentiable view operations to have the tangent have the same layout as primal")
      }
      // 创建一个新的张量，具有与 new_grad 相同的特征元数据
      auto res = at::_new_zeros_with_same_feature_meta(new_grad, self);
      res._set_conj(self.is_conj());
      res._set_neg(self.is_neg());
      res.copy_(new_grad);
      new_grad = res;
    }

    // 设置当前级别的前向梯度为 new_grad
    fw_grad_->set_value(new_grad, level);
}
// AutogradMeta 类的 fw_grad 方法，用于获取前向梯度
const Variable& AutogradMeta::fw_grad(
    uint64_t level,
    const at::TensorBase& self) const {
  // 检查是否禁用了前向自动微分
  if (!c10::AutogradState::get_tls_state().get_fw_grad_mode()) {
    // 如果禁用了前向自动微分，则返回未定义的梯度
    return ForwardGrad::undef_grad();
  }

  // 确保并发调用 fw_grad() 的“读取”是线程安全的
  std::lock_guard<std::mutex> lock(mutex_);

  // 获取直接的前向梯度
  const auto& direct_fw_grad =
      fw_grad_ ? fw_grad_->value(level) : ForwardGrad::undef_grad();

  if (!direct_fw_grad.defined() && is_view_) {
    // 对于没有前向梯度的视图，检查它们的基张量是否由原位操作定义了梯度
    auto const_view_meta =
        static_cast<const torch::autograd::DifferentiableViewMeta*>(this);
    // 这样做是安全的，因为我们只修改 fw_grad_，而且这个字段在所有方法中都被正确锁定
    if (const_view_meta->has_fw_view()) {
      const auto& view_info = const_view_meta->get_forward_view();
      const auto& base = view_info.base_;

      const auto& base_val = base._fw_grad(level);
      if (base_val.defined()) {
        // 惰性初始化 fw_grad_
        const_view_meta->fw_grad_ = std::make_shared<ForwardGrad>();

        Variable new_val;
        if (view_info.has_view_fn()) {
          new_val = view_info.view_fn()(base_val);
        } else {
          new_val = base_val.as_strided(
              self.sizes(), self.strides(), self.storage_offset());
        }

        const_view_meta->fw_grad_->set_value(new_val, level);
        return const_view_meta->fw_grad_->value(level);
      }
    }
  }
  return direct_fw_grad;
}

} // namespace autograd
} // namespace torch
```