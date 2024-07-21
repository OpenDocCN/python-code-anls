# `.\pytorch\torch\csrc\autograd\utils\lambda_post_hook.h`

```
#pragma once

#include <torch/csrc/autograd/function_hook.h>

namespace torch {
namespace autograd {
namespace utils {

// 将 lambda 函数转换为 torch::autograd::FunctionPostHook。
class LambdaPostHook : public torch::autograd::FunctionPostHook {
  using variable_list = std::vector<torch::autograd::Variable>;
  using fn_type =
      std::function<variable_list(const variable_list&, const variable_list&)>;
  using compiled_fn_type = std::function<void(CompiledNodeArgs&)>;

 public:
  // lambda 函数接收自动求导函数的输出和输入作为参数，并可以通过返回新的输出来修改自动求导函数的输出。
  /* implicit */ LambdaPostHook(fn_type fn) : fn_(std::move(fn)) {}

  // 构造函数允许传入已编译的 lambda 函数，以便执行优化后的操作。
  LambdaPostHook(fn_type fn, compiled_fn_type compiled_fn)
      : fn_(std::move(fn)), compiled_fn_(std::move(compiled_fn)) {}

  // 重载运算符 ()，执行 lambda 函数并返回其结果。
  variable_list operator()(
      const variable_list& outputs,
      const variable_list& inputs) override {
    return fn_(outputs, inputs);
  }

  // 实现接口函数，用于执行已编译的节点参数。
  void compiled_args(CompiledNodeArgs& args) override {}

 protected:
  // 存储 lambda 函数和已编译的函数。
  std::function<variable_list(const variable_list&, const variable_list&)> fn_;
  compiled_fn_type compiled_fn_;
};

} // namespace utils
} // namespace autograd
} // namespace torch
```