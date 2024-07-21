# `.\pytorch\torch\csrc\jit\passes\requires_grad_analysis.cpp`

```
// 引入 Torch JIT 中的相关头文件和命名空间
#include <torch/csrc/jit/passes/requires_grad_analysis.h>

// 引入 ATen 和 C10 中的类型和工具类头文件
#include <ATen/core/jit_type.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>

// 引入标准库中的向量容器
#include <vector>

// Torch JIT 命名空间
namespace torch {
namespace jit {

// 匿名命名空间，定义一些内部使用的函数和数据结构

// 获取节点中值的 requires_grad 属性
bool getRequiresGrad(Value* value) {
  return value->requires_grad();
}

// 设置节点中值的 requires_grad 属性
void setRequiresGrad(Value* value, bool req_value) {
  // 如果节点类型是 TensorType，则更新其 requires_grad 属性
  if (auto type = value->type()->cast<TensorType>()) {
    value->setType(type->withRequiresGrad(req_value));
  }
}

// 批量设置输出值的 requires_grad 属性
void setRequiresGrad(
    at::ArrayRef<Value*> outputs,
    const std::vector<bool>& values) {
  AT_ASSERT(outputs.size() == values.size());
  // 遍历所有输出值，逐个设置 requires_grad 属性
  for (const auto i : c10::irange(values.size())) {
    setRequiresGrad(outputs[i], values[i]);
  }
}

// 批量设置节点输出的 requires_grad 属性
void setRequiresGrad(Node* node, const std::vector<bool>& values) {
  // 调用上面的函数设置节点的所有输出值的 requires_grad 属性
  setRequiresGrad(node->outputs(), values);
}

// 对两个 bool 向量进行按位或操作
std::vector<bool> bitwiseOr(std::vector<bool> a, const std::vector<bool>& b) {
  AT_ASSERT(a.size() == b.size());
  // 遍历两个向量，对对应位置的元素进行按位或操作
  for (const auto i : c10::irange(a.size())) {
    a[i] = a[i] || b[i];
  }
  return a;
}

// 处理简单节点的 requires_grad 传播
void PropagateRequiresGradSimpleNode(Node* node) {
  // 定义比较操作的运算符集合
  static const OperatorSet comparison_ops = {
      "aten::lt(Tensor self, Tensor other) -> Tensor",
      "aten::le(Tensor self, Tensor other) -> Tensor",
      "aten::gt(Tensor self, Tensor other) -> Tensor",
      "aten::ge(Tensor self, Tensor other) -> Tensor",
      "aten::eq(Tensor self, Tensor other) -> Tensor",
      "aten::ne(Tensor self, Tensor other) -> Tensor",
      "aten::lt(Tensor self, Scalar other) -> Tensor",
      "aten::le(Tensor self, Scalar other) -> Tensor",
      "aten::gt(Tensor self, Scalar other) -> Tensor",
      "aten::ge(Tensor self, Scalar other) -> Tensor",
      "aten::eq(Tensor self, Scalar other) -> Tensor",
      "aten::ne(Tensor self, Scalar other) -> Tensor",
  };

  // 根据节点所属的运算符集合进行不同的 requires_grad 设置
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (node->isMemberOf(comparison_ops)) {
    return setRequiresGrad(node->output(), false); // 比较操作结果不需要梯度
  } else if (node->matches(
                 "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
    return setRequiresGrad(node->output(), node->input(0)->requires_grad()); // type_as 操作保持输入的梯度属性
  } else if (node->matches("aten::detach(Tensor(a) self) -> Tensor(a)")) {
    return setRequiresGrad(node->output(), false); // detach 操作使得输出不需要梯度
  } else if (node->kind() == aten::tensor) {
    // 处理 tensor 构造函数的 requires_grad 参数
    if (auto grad_index =
            node->schema().argumentIndexWithName("requires_grad")) {
      if (auto const_arg = constant_as<bool>(node->inputs().at(*grad_index))) {
        return setRequiresGrad(node->output(), *const_arg); // 根据参数值设置 requires_grad
      }
    }
    // 对于 tensor 构造函数，根据数据类型判断是否可微分
    if (auto type = node->output()->type()->cast<TensorType>()) {
      if (type->scalarType()) {
        setRequiresGrad(
            node->output(),
            autograd::isDifferentiableType(*type->scalarType()));
      }
    }
    return;
  }

  // 获取节点的输入和输出
  auto inputs = node->inputs();
  auto outputs = node->outputs();

  // 判断是否需要设置梯度要求
  bool should_require =
      std::any_of(inputs.begin(), inputs.end(), getRequiresGrad);

  // 遍历节点的输出
  for (Value* output : outputs) {
    // 检查输出是否是张量类型
    if (auto type = output->type()->cast<TensorType>()) {
      // 如果输出有标量类型
      if (type->scalarType()) {
        // 设置是否需要梯度要求，条件是输入节点需要梯度，并且标量类型可微分
        setRequiresGrad(
            output,
            should_require &&
                autograd::isDifferentiableType(*type->scalarType()));
      }
    }
  }
// 定义一个函数，用于递归地传播节点是否需要梯度的信息
void PropagateRequiresGrad(Block* block);

// 实现节点是否需要梯度传播的函数
void PropagateRequiresGrad(Node* node) {
  // 如果节点是条件语句 (prim::If)
  if (node->kind() == prim::If) {
    // 获取节点的两个分支块
    auto blocks = node->blocks();
    auto true_block = blocks.at(0);
    auto false_block = blocks.at(1);

    // 递归调用以传播真实现和虚拟实现的梯度要求
    PropagateRequiresGrad(true_block);
    PropagateRequiresGrad(false_block);

    // 计算真实现和虚拟实现的输出是否需要梯度，通过按位或操作合并结果
    auto outputs_require = bitwiseOr(
        fmap(true_block->outputs(), getRequiresGrad),
        fmap(false_block->outputs(), getRequiresGrad));
    // 设置当前节点的输出是否需要梯度
    setRequiresGrad(node, outputs_require);
  } else if (node->kind() == prim::Loop) {  // 如果节点是循环语句 (prim::Loop)
    // 获取循环体块
    auto body = node->blocks().at(0);
    // 根据节点输入中的梯度需求初始化循环输入的梯度需求
    std::vector<bool> loop_inputs_require =
        fmap(node->inputs().slice(2), getRequiresGrad);
    std::vector<bool> body_inputs_require = loop_inputs_require;
    std::vector<bool> body_outputs_require(node->outputs().size(), false);

    std::vector<bool> new_body_inputs_require = body_inputs_require;
    std::vector<bool> new_body_outputs_require = body_outputs_require;

    // 持续迭代直到结果收敛
    do {
      body_inputs_require = new_body_inputs_require;
      body_outputs_require = new_body_outputs_require;

      // 计算新的循环体输入梯度需求
      new_body_inputs_require =
          bitwiseOr(body_inputs_require, body_outputs_require);
      // 设置循环体参数节点的输出是否需要梯度
      setRequiresGrad(
          body->param_node()->outputs().slice(1), new_body_inputs_require);
      // 递归调用以传播循环体内部的梯度要求
      PropagateRequiresGrad(body);
      // 计算新的循环体输出梯度需求
      new_body_outputs_require =
          fmap(body->return_node()->inputs().slice(1), getRequiresGrad);
    } while (new_body_inputs_require != body_inputs_require ||
             new_body_outputs_require != body_outputs_require);

    // 设置当前节点的输出是否需要梯度，包括循环体输出和循环输入
    setRequiresGrad(node, bitwiseOr(body_outputs_require, loop_inputs_require));
  } else {
    // 对于其他类型的节点，简化的梯度传播处理
    PropagateRequiresGradSimpleNode(node);
  }
}

// 对块中的每个节点递归调用节点梯度传播函数
void PropagateRequiresGrad(Block* block) {
  for (Node* node : block->nodes()) {
    PropagateRequiresGrad(node);
  }
}

// 匿名命名空间结束
} // anonymous namespace

// 函数入口，传播图的梯度需求
void PropagateRequiresGrad(std::shared_ptr<Graph>& graph) {
  PropagateRequiresGrad(graph->block());
}

// 结束定义 torch 的 jit 命名空间
} // namespace jit
// 结束定义 torch 命名空间
} // namespace torch
```