# `.\pytorch\torch\csrc\autograd\functions\basic_ops.cpp`

```py
#include <torch/csrc/autograd/functions/basic_ops.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

#include <ATen/ATen.h>

#include <memory>
#include <utility>

namespace torch {
namespace autograd {

// 实现 Error 类的 apply 方法，抛出运行时错误
auto Error::apply(variable_list&& inputs) -> variable_list {
  throw std::runtime_error(msg);
}

// 实现 Error 类的 compiled_args 方法，用于在收集过程中抛出错误，阻止图的编译
void Error::compiled_args(CompiledNodeArgs& args) {
  apply(variable_list());
}

// 实现 Error 类的 apply_with_saved 方法，未定义，断言不可达
variable_list Error::apply_with_saved(
    const variable_list& inputs,
    SwapSavedVariables& saved) {
  TORCH_INTERNAL_ASSERT(false, "unreachable");
}

// 实现 DelayedError 类的 apply 方法，生成输出的 tensor_list，并在异常处理函数中构造 Error 对象
auto DelayedError::apply(variable_list&& inputs) -> variable_list {
  tensor_list outputs;
  outputs.reserve(inputs.size());
  for (auto& var : inputs) {
    // FIXME: share version counters
    outputs.emplace_back(var.defined() ? var.tensor_data() : at::Tensor());
  }
  return wrap_outputs(inputs, std::move(outputs), [&](edge_list&& next_edges) {
    return std::make_shared<Error>(msg, std::move(next_edges));
  });
}

// 实现 UndefinedGrad 类的 apply 方法，生成输出的 tensor_list，并在异常处理函数中构造 UndefinedGradBackward 对象
auto UndefinedGrad::apply(variable_list&& inputs) -> variable_list {
  tensor_list outputs;
  outputs.reserve(inputs.size());
  for (auto& var : inputs) {
    outputs.emplace_back(
        var.defined() ? var.clone().tensor_data() : at::Tensor());
  }
  return wrap_outputs(inputs, std::move(outputs), [&](edge_list&& next_edges) {
    return std::make_shared<UndefinedGradBackward>(std::move(next_edges));
  });
}

// 实现 UndefinedGradBackward 类的 apply 方法，生成输入梯度 tensor_list
auto UndefinedGradBackward::apply(variable_list&& output_grads)
    -> variable_list {
  tensor_list input_grads;
  input_grads.reserve(output_grads.size());
  for (auto& grad : output_grads) {
    (void)grad; // Suppress unused variable warning
    input_grads.emplace_back();
  }
  return input_grads;
}

// 实现 Identity 类的 apply 方法，直接返回输入的梯度
auto Identity::apply(variable_list&& grads) -> variable_list {
  return std::move(grads);
}

// 实现 GraphRoot 类的 compiled_args 方法，收集输出变量列表
void GraphRoot::compiled_args(CompiledNodeArgs& args) {
  args.collect(outputs);
}

// 实现 GraphRoot 类的 apply_with_saved 方法，保存前和保存后的操作，并返回输出变量列表
variable_list GraphRoot::apply_with_saved(
    const variable_list& inputs,
    SwapSavedVariables& saved) {
  saved.before(outputs);
  variable_list result(outputs);
  saved.after(outputs);
  return result;
}

} // namespace autograd
} // namespace torch
```