# `.\pytorch\torch\csrc\autograd\functions\accumulate_grad.cpp`

```
// 包含头文件，用于自动求导模块中的梯度累积操作
#include <torch/csrc/autograd/functions/accumulate_grad.h>

// 包含用于分发调度的调度器头文件
#include <ATen/core/dispatch/Dispatcher.h>
// 包含自动求导模块中的基本操作函数头文件
#include <torch/csrc/autograd/functions/basic_ops.h>
// 包含自动求导模块中的张量操作函数头文件
#include <torch/csrc/autograd/functions/tensor.h>
// 包含自动求导模块中的实用函数头文件
#include <torch/csrc/autograd/functions/utils.h>
// 包含自动求导模块中的梯度模式头文件
#include <torch/csrc/autograd/grad_mode.h>
// 包含自动求导模块中的变量头文件
#include <torch/csrc/autograd/variable.h>
// 包含编译自动求导模块的头文件
#include <torch/csrc/dynamo/compiled_autograd.h>

// 包含标准整数类型的头文件
#include <cstdint>
// 包含标准异常处理的头文件
#include <stdexcept>
// 包含标准实用工具的头文件
#include <utility>

// 命名空间 torch 下的 autograd 命名空间
namespace torch {
namespace autograd {

// AccumulateGrad 类的构造函数，设置 sequence_nr 为最大值，确保在反向传播时尽早调用
AccumulateGrad::AccumulateGrad(Variable variable_)
    : Node(/*sequence_nr=*/UINT64_MAX), variable(std::move(variable_)) {
  // 添加输入变量的元数据到节点中
  add_input_metadata(variable);
}

// 应用函数 apply，接收梯度列表并进行处理
auto AccumulateGrad::apply(variable_list&& grads) -> variable_list {
  // 检查输入变量列表的有效性
  check_input_variables("AccumulateGrad", grads, 1, 0);

  // 如果梯度未定义，则返回空列表
  if (!grads[0].defined())
    return {};
  // 如果变量已有梯度函数，则抛出逻辑错误异常
  if (variable.grad_fn())
    throw std::logic_error(
        "leaf variable has been moved into the graph interior");
  // 如果变量不需要梯度，则返回空列表
  if (!variable.requires_grad())
    return {};

  // 使用 std::move(grads[0]) 避免增加引用计数
  at::Tensor new_grad = std::move(grads[0]);

  // 获取 mutex_ 锁，保护变量的线程安全，确保 AccumulateGrad 不会因为来自不同线程的共享变量而竞争更新梯度
  std::lock_guard<std::mutex> lock(mutex_);

  // 获取变量的可变梯度，并将新的梯度累积到其中
  at::Tensor& grad = variable.mutable_grad();
  accumulateGrad(
      variable,
      grad,
      new_grad,
      1 + !post_hooks().empty() /* num_expected_refs */,
      [&grad](at::Tensor&& grad_update) { grad = std::move(grad_update); });

  // 调用张量后梯度累积的钩子函数
  auto& hook = tensor_post_acc_grad_hooks();
  if (hook != nullptr) {
    (*hook)(variable);
  }

  // 返回空的变量列表
  return variable_list();
}

// 编译自动求导节点参数的函数，收集变量和其梯度
void AccumulateGrad::compiled_args(CompiledNodeArgs& args) {
  // 如果变量已定义并且需要梯度，则收集变量及其梯度
  if (args.cond(variable.defined() && variable.requires_grad())) {
    args.collect(variable);
    args.collect(variable.grad());
  }
  // 调用张量后梯度累积的钩子函数的编译参数收集
  auto& hook = tensor_post_acc_grad_hooks();
  if (hook != nullptr) {
    hook->compiled_args(args);
  }
}

// 应用函数 apply_with_saved，接收梯度列表及保存的变量，并进行处理
variable_list AccumulateGrad::apply_with_saved(
    const variable_list& grads,
    SwapSavedVariables& saved) {
  // 如果变量未定义或不需要梯度，或者梯度未定义，则返回空列表
  if (!(variable.defined() && variable.requires_grad()) ||
      !grads[0].defined()) {
    // 返回变量列表
    return variable_list();
  }
  // 断言：确保变量没有梯度函数，并且梯度列表的大小为1
  TORCH_INTERNAL_ASSERT(!variable.grad_fn() && grads.size() == 1);
  // 创建变量的副本
  at::Tensor variable_copy = variable;
  // 获取变量的梯度副本
  at::Tensor grad_copy = variable.grad();
  // 在保存点"saved"中记录变量和其梯度的状态
  saved.before(variable_copy);
  saved.before(grad_copy);
  // 设置变量的可变梯度为梯度副本
  variable_copy.mutable_grad() = grad_copy;
  // 操作是故意静态的
  // 查找并调用名为"inductor::accumulate_grad_"的静态分发操作
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("inductor::accumulate_grad_", "")
                       .typed<void(const at::Tensor&, const at::Tensor&)>();
  // 调用操作，累积梯度到变量的副本上
  op.call(variable_copy, grads[0]);
  // 获取张量后梯度计算后的钩子
  auto& hook = tensor_post_acc_grad_hooks();
  // 如果钩子不为空，应用保存的状态到变量的副本上
  if (hook != nullptr) {
    hook->apply_with_saved(variable_copy, saved);
  }
  // 在保存点"saved"中记录变量和其梯度的状态
  saved.after(variable_copy);
  saved.after(grad_copy);

  // 返回变量列表
  return variable_list();
}  // 结束 autograd 命名空间
}  // 结束 torch 命名空间
```