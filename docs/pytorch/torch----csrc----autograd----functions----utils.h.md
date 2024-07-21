# `.\pytorch\torch\csrc\autograd\functions\utils.h`

```py
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/InferenceMode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/core/Tensor.h>

#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace autograd {

using function_constructor = std::function<std::shared_ptr<Node>(edge_list&&)>;

/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
TORCH_API variable_list wrap_outputs(
    const variable_list& inputs,
    tensor_list&& outputs,
    const function_constructor& ctr);

/**
 * Checks that inputs contains exactly `args` items and that the first
 * `required_args` items are not nullptr. If not specified, `required_args` defaults to `args`.
 */
TORCH_API void check_input_variables(
    const char* name,
    const variable_list& inputs,
    int args,
    int required_args = -1,
    bool allow_undefined = false);

/**
 * Functor to determine if any of the tensors in the arguments have requires_grad set.
 * Used in compute_requires_grad.
 */
struct ComputeRequiresGrad : IterArgs<ComputeRequiresGrad> {
  bool out = false;
  using IterArgs<ComputeRequiresGrad>::operator();
  void operator()(const at::Tensor& tensor) {
    const auto& var = static_cast<const Variable&>(tensor);
    if (var.defined() && var.requires_grad()) {
      out = true;
    }
  }
  void operator()(const std::optional<at::Tensor>& tensor) {
    if (tensor.has_value()) {
      (*this)(*tensor);
    }
  }
  bool short_circuit() {
    return out;
  }
};

/**
 * Template function to check if any of the input tensors require gradients.
 */
template <typename... Args>
inline bool compute_requires_grad(Args&&... args) {
  if (!GradMode::is_enabled()) {
    return false;
  }
  return ComputeRequiresGrad().apply(std::forward<Args>(args)...).out;
}

/**
 * Sets the history of a single tensor variable with its associated gradient function.
 * Throws an error if grad_fn is nullptr and variable is defined.
 */
inline void set_history(
    const at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  TORCH_CHECK(grad_fn != nullptr);
  if (variable.defined()) {
    // If the codegen triggers this, you most likely want to add your newly
    // added function to the DONT_REQUIRE_DERIVATIVE list in
    // tools/autograd/gen_variable_type.py
    TORCH_INTERNAL_ASSERT(isDifferentiableType(variable.scalar_type()));
    auto output_nr = grad_fn->add_input_metadata(variable);
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}

/**
 * Sets the history of a vector of tensor variables with their associated gradient function.
 */
inline void set_history(
    const std::vector<Variable>& variables,
    const std::shared_ptr<Node>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}

/**
 * Checks if the forward gradient of an optional tensor is defined.
 */
inline bool isFwGradDefined(const std::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() && t->_fw_grad(/*level */ 0).defined();
}

/**
 * Checks if the forward gradients of any tensors in a list are defined.
 */
inline bool isFwGradDefinedTensorList(const at::ITensorListRef& variables) {
  bool ret = false;
  for (auto& variable : variables) {
    ret |= isFwGradDefined(variable);
  }
  return ret;
}

/**
 * Overload for checking if the forward gradients of any tensors in a list are defined.
 */
inline bool isFwGradDefinedTensorList(
    // The function signature is intentionally left incomplete here,
    // waiting for further code completion.
    const c10::List<std::optional<at::Tensor>>& li) {
```  
# 接受一个常量引用参数 `li`，该参数是一个 `c10::List` 类型的对象，存储了包含可选的 `at::Tensor` 对象。


  bool ret = false;
```py  
# 初始化一个布尔变量 `ret`，设置为 `false`。


  for (auto i : c10::irange(li.size())) {
```  
# 使用 `c10::irange` 迭代 `li` 的长度，生成一个序列用于遍历 `li` 中的元素。


    auto t = li.get(i);
```py  
# 获取 `li` 中索引为 `i` 的元素，并将其存储在变量 `t` 中。


    ret |= (t.has_value() && isFwGradDefined(t.value()));
```  
# 将 `ret` 更新为其当前值与 `(t.has_value() && isFwGradDefined(t.value()))` 的逻辑或结果。


  }
```py  
# 结束 `for` 循环。


  return ret;
```  
# 返回布尔变量 `ret` 的值作为函数的结果。
}
// 结束 autograd 命名空间
} // namespace autograd
// 结束 torch 命名空间
} // namespace torch
```