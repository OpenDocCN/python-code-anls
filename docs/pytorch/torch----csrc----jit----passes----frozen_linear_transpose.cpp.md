# `.\pytorch\torch\csrc\jit\passes\frozen_linear_transpose.cpp`

```
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/transpose.h>
#endif

#include <iostream>
#include <utility>

namespace torch {
namespace jit {
namespace {

using Tensor = at::Tensor;

// 定义一个用于转置线性操作的类
class TransposeFrozenLinear {
 public:
  // 构造函数，接收一个图形对象指针作为参数
  TransposeFrozenLinear(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  // 执行转置操作
  bool run() {
    // 不能在迭代过程中删除节点
    DepthFirstGraphNodeIterator graph_it(graph_);

    // 迭代图中的每个节点
    for (auto next_node = graph_it.next(); next_node != nullptr;) {
      Node* node = next_node;
      next_node = graph_it.next();

      // 如果节点是常量线性操作，则替换为矩阵乘法
      if (is_constant_linear_op(node)) {
        replace_linear_with_matmul(node);
      }
    }
    return graph_modified_;
  }

  // 判断节点是否是常量线性操作
  bool is_constant_linear_op(Node* node) {
    // 如果节点的类型不是 aten::linear，则返回 false
    if (node->kind() != aten::linear) {
      return false;
    }

    // 这也过滤掉线性操作的 out-variant
    return !nonConstantParameters(node);
  }

  // 替换线性操作为矩阵乘法
  void replace_linear_with_matmul(Node* node) {
    // 标记图已修改
    graph_modified_ = true;
    Node* matmul = nullptr;

    {
      // 在节点处设置插入点保护
      WithInsertPoint insert_guard(node);
      auto weight = node->namedInput("weight");

      // 获取权重张量并转置
      Tensor weight_tensor = constant_as<Tensor>(weight).value();
      Tensor weight_t_tensor = at::transpose(weight_tensor, 1, 0)
                                   .clone(at::MemoryFormat::Contiguous);
      // 将转置后的权重张量插入图中
      Value* weight_t = graph_->insertConstant(std::move(weight_t_tensor));
      // 创建一个矩阵乘法节点，并插入到当前节点之后
      matmul = graph_->create(aten::matmul, {node->inputs()[0], weight_t});
      matmul->insertAfter(node);
    }

    // 处理偏置（如果存在的话）
    {
      // 在矩阵乘法节点处设置插入点保护
      WithInsertPoint insert_guard(matmul);
      auto bias = node->namedInput("bias");
      // 如果偏置是 None 类型，则直接替换所有用途为矩阵乘法结果
      if (bias->type() == NoneType::get()) {
        node->replaceAllUsesWith(matmul);
      } else {
        // 创建一个偏置倍数节点，并进行加法操作
        Value* bias_scale = graph_->insertConstant(1);
        Node* bias_result =
            graph_->create(aten::add, {matmul->output(), bias, bias_scale});
        bias_result->insertAfter(matmul);
        node->replaceAllUsesWith(bias_result);
      }
      // 删除原线性操作节点
      node->destroy();
    }
  };

  // 处理块及其子块（未实现）
  void handleBlockAndSubblocks(Block* block) {}

 private:
  std::shared_ptr<Graph> graph_;
  bool graph_modified_ = false;
};

} // namespace

// 导出的函数，用于对图执行冻结线性转置操作
TORCH_API bool FrozenLinearTranspose(std::shared_ptr<Graph>& graph) {
  TransposeFrozenLinear transposeWeight(graph);
  GRAPH_DUMP("Before FrozenLinearTranspose", graph);
  bool changed = transposeWeight.run();
  if (changed) {
    GRAPH_DUMP("After FrozenLinearTranspose", graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch
```