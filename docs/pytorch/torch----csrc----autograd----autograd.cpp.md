# `.\pytorch\torch\csrc\autograd\autograd.cpp`

```
// 包含 Torch 自动求导模块的头文件
#include <torch/csrc/autograd/autograd.h>
// 包含 Torch 变量相关的头文件
#include <torch/csrc/autograd/variable.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来选择不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/ones_like.h>
#endif

// 包含 Torch 自动求导模块的其他必要头文件
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>

// 包含 C10 库的实用工具头文件
#include <c10/util/irange.h>

// Torch 的命名空间
namespace torch {
namespace autograd {

// 注意事项提到的一些注释
// NB: This code duplicates existing logic at torch/autograd/__init__.py and
// torch._C._EngineBase.run_backward in torch/csrc/autograd/python_engine.cpp
// This is a purely C++ API for Autograd without any dependencies on python
// it can be exposed in PyTorch C++ API and TorchScript. We will need to
// maintain the logic equality of this file and the python file together if one
// changes.
// TODO: Make the Python API above to just call this C++ API.

// 定义静态函数 _make_grads，用于生成梯度列表
static variable_list _make_grads(
    const variable_list& outputs,     // 输入参数：输出变量列表
    const variable_list& grad_outputs) {  // 输入参数：梯度输出列表
  size_t num_tensors = outputs.size();   // 输出变量列表的大小
  size_t num_gradients = grad_outputs.size();  // 梯度输出列表的大小
  variable_list new_grads;  // 新的梯度列表
  new_grads.reserve(num_tensors);  // 预留空间以容纳输出变量数量的梯度

  // 如果梯度输出列表为空
  if (grad_outputs.empty()) {
    // 遍历每个输出变量
    for (const Variable& output : outputs) {
      // 如果当前输出变量需要梯度计算
      if (output.requires_grad()) {
        // 检查输出变量是否为标量
        TORCH_CHECK(
            output.numel() == 1,
            "grad can be implicitly created only for scalar outputs");
        // 检查输出变量是否为浮点数类型
        TORCH_CHECK(
            c10::isFloatingType(output.scalar_type()),
            "grad can be computed only for real scalar outputs but got ",
            output.scalar_type());
        // 创建一个与输出变量相同形状的全为1的张量作为梯度
        new_grads.emplace_back(
            at::ones_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
      }
    }
  } else {
    // 如果梯度输出列表不为空，检查输出变量和梯度输出的数量是否相等
    TORCH_CHECK(
        num_tensors == num_gradients,
        "got ",
        num_tensors,
        " tensors and ",
        num_gradients,
        " gradients");
    // 对于每个输出张量进行迭代处理
    for (const auto i : c10::irange(outputs.size())) {
      // 获取当前输出张量和对应的梯度张量
      const Variable& output = outputs[i];
      const Variable& grad_output = grad_outputs[i];
      // 如果梯度张量未定义
      if (!grad_output.defined()) {
        // 如果当前输出张量需要计算梯度
        if (output.requires_grad()) {
          // 检查当前输出张量是否为标量
          TORCH_CHECK(
              output.numel() == 1,
              "grad can be implicitly created only for scalar outputs");
          // 检查当前输出张量是否为浮点数类型
          TORCH_CHECK(
              c10::isFloatingType(output.scalar_type()),
              "grad can be computed only for real scalar outputs but got ",
              output.scalar_type());
          // 创建一个与当前输出张量形状相同的全为1的张量作为梯度
          new_grads.emplace_back(
              at::ones_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
        }
      } else {
        // 如果梯度张量已定义，检查梯度张量和输出张量的复数性质是否匹配
        TORCH_CHECK(
            grad_output.is_complex() == output.is_complex(),
            "For complex Tensors, both grad_output and output are required ",
            "to have the same dtype. Mismatch in dtype: grad_output[",
            grad_output,
            "] has a dtype of ",
            grad_output.scalar_type(),
            " and output[",
            output,
            "] has a dtype of ",
            output.scalar_type(),
            ".");
        // 将梯度张量加入新的梯度列表中
        // grad_output 已定义，直接添加到 new_grads 中
        new_grads.emplace_back(grad_output);
      }
    }
  }
  // 返回新的梯度张量列表
  return new_grads;
}

// 后向传播函数，计算梯度并返回梯度变量列表
static variable_list run_backward(
    const variable_list& outputs,         // 输出变量列表
    const variable_list& grad_outputs,    // 梯度输出列表
    bool keep_graph,                     // 是否保留计算图
    bool create_graph,                   // 是否创建计算图
    const variable_list& inputs,         // 输入变量列表
    bool allow_unused,                   // 是否允许未使用的变量
    bool accumulate_grad) {              // 是否累积梯度

  size_t num_tensors = outputs.size();   // 输出变量的数量
  edge_list roots;                      // 根节点列表
  roots.reserve(num_tensors);           // 预留空间以容纳输出变量数量的根节点
  for (const auto i : c10::irange(num_tensors)) {  // 遍历输出变量列表
    const Variable& output = outputs[i];   // 获取当前输出变量
    auto gradient_edge = impl::gradient_edge(output);  // 获取输出变量的梯度边
    TORCH_CHECK(
        gradient_edge.function,
        "element ",
        i,
        " of tensors does not require grad and does not have a grad_fn");  // 检查梯度边的函数是否存在
    roots.push_back(std::move(gradient_edge));   // 将梯度边添加到根节点列表中
  }

  edge_list output_edges;   // 输出边列表
  if (!inputs.empty()) {    // 如果输入列表不为空
    size_t num_inputs = inputs.size();   // 输入变量的数量
    output_edges.reserve(num_inputs);   // 预留空间以容纳输入变量数量的输出边
    for (const auto i : c10::irange(num_inputs)) {   // 遍历输入变量列表
      const Variable& input = inputs[i];   // 获取当前输入变量
      const auto output_nr = input.output_nr();   // 获取输入变量的输出编号
      auto grad_fn = input.grad_fn();   // 获取输入变量的梯度函数
      if (!grad_fn) {
        grad_fn = impl::try_get_grad_accumulator(input);   // 尝试获取输入变量的梯度累加器
      }
      if (accumulate_grad) {
        input.retain_grad();   // 如果需要累积梯度，则保持输入变量的梯度
      }
      TORCH_CHECK(
          input.requires_grad(),
          "element ",
          i,
          " of the input tensors does not require grad");   // 检查输入变量是否需要梯度
      if (!grad_fn) {
        // See NOTE [ Autograd Unreachable Input ] for details
        output_edges.emplace_back(std::make_shared<Identity>(), 0);   // 如果没有梯度函数，则添加一个 Identity 边
      } else {
        output_edges.emplace_back(grad_fn, output_nr);   // 否则添加输入变量的梯度函数和输出编号
      }
    }
  }

  variable_list grad_inputs = Engine::get_default_engine().execute(
      roots,
      grad_outputs,
      keep_graph,
      create_graph,
      accumulate_grad,
      output_edges);   // 使用默认引擎执行反向传播计算梯度

  // 根据 allow_unused 标志检查 grad_inputs 是否包含 None
  if (!inputs.empty() && !allow_unused) {   // 如果输入列表不为空且不允许未使用的变量
    size_t num_inputs = inputs.size();   // 输入变量的数量
    for (const auto i : c10::irange(num_inputs)) {   // 遍历输入变量列表
      TORCH_CHECK(
          grad_inputs[i].defined(),
          "element ",
          i,
          "of the "
          "differentiated Tensors appears to not have been used "
          "in the graph. Set allow_unused=True if this is the "
          "desired behavior.");   // 检查梯度是否已定义，如果未使用则抛出错误信息
    }
  }
  
  return grad_inputs;   // 返回计算得到的梯度变量列表
}

// 反向传播函数，计算梯度并更新参数
void backward(
    const variable_list& tensors,   // 张量列表
    const variable_list& grad_tensors,   // 梯度张量列表
    std::optional<bool> retain_graph,   // 是否保留计算图的可选标志
    bool create_graph,   // 是否创建计算图
    const variable_list& inputs) {   // 输入变量列表

  variable_list gradients = _make_grads(tensors, grad_tensors);   // 生成梯度列表
  if (!retain_graph) {
    retain_graph = create_graph;   // 如果未指定保留计算图，则与 create_graph 一致
  }
  run_backward(
      tensors,
      gradients,
      retain_graph.value(),   // 获取保留计算图的值
      create_graph,   // 是否创建计算图
      inputs,
      /*allow_unused=*/true,   // 是否允许未使用的变量
      /*accumulate_grad=*/true);   // 是否累积梯度
}

// 计算梯度函数，返回计算得到的梯度变量列表
variable_list grad(
    const variable_list& outputs,   // 输出变量列表
    const variable_list& inputs,   // 输入变量列表
    const variable_list& grad_outputs,   // 梯度输出列表
    std::optional<bool> retain_graph,   // 是否保留计算图的可选标志
    bool create_graph,   // 是否创建计算图
    bool allow_unused) {
  // 调用 _make_grads 函数，根据给定的输出和梯度输出创建梯度变量列表
  variable_list gradients = _make_grads(outputs, grad_outputs);
  // 如果不需要保留计算图，则根据 create_graph 的值来确定是否保留
  if (!retain_graph) {
    retain_graph = create_graph;
  }
  // 调用 run_backward 函数进行反向传播计算梯度
  // 参数依次为：输出(outputs)，梯度(gradients)，是否保留计算图(retain_graph的值)，是否创建新计算图(create_graph)，输入(inputs)，是否允许未使用的变量(allow_unused)，是否累积梯度(accumulate_grad=false)
  return run_backward(
      outputs,
      gradients,
      retain_graph.value(),
      create_graph,
      inputs,
      allow_unused,
      /*accumulate_grad=*/false);
}
}

namespace forward_ad {

// 定义一个名为enter_dual_level的函数，返回类型为uint64_t
uint64_t enter_dual_level() {
    // 调用ForwardADLevel命名空间的get_next_idx函数，并返回其结果
    return ForwardADLevel::get_next_idx();
}

// 定义一个名为exit_dual_level的函数，参数为uint64_t类型的level，返回类型为void
void exit_dual_level(uint64_t level) {
    // 调用ForwardADLevel命名空间的release_idx函数，传入level作为参数
    ForwardADLevel::release_idx(level);
}

} // namespace forward_ad

} // namespace autograd
} // namespace torch
```