# `.\pytorch\torch\csrc\jit\passes\frozen_linear_folding.cpp`

```
// 引入 Torch 的头文件，用于定义常量、IR 结构和图优化的函数
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_linear_bn.h>
#include <torch/csrc/jit/passes/frozen_linear_folding.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>

// 根据编译器定义，选择性引入 ATen 库的函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#endif

// Torch 的命名空间
namespace torch {
namespace jit {

// 匿名命名空间，用于实现局部函数或者变量的封装
namespace {

// 定义别名 Tensor 为 ATen 的 Tensor 类型
using Tensor = at::Tensor;

// 判断节点是否为支持的线性节点（目前仅支持 aten::linear 类型）
bool supportedLinearNode(Node* n) {
  if (n->kind() == aten::linear) {
    return true;
  } else {
    return false;
  }
}

// 递归函数，用于在图的基本块中折叠冻结的线性批归一化节点
bool FoldFrozenLinearBatchnorm(Block* b) {
  bool graph_modified = false;
  // 遍历基本块中的每个节点
  for (Node* n : b->nodes()) {
    // 如果节点包含子块，递归调用折叠函数
    for (Block* block : n->blocks()) {
      graph_modified |= FoldFrozenLinearBatchnorm(block);
    }
    // 处理当前节点的操作
    // 这里需要补充具体的操作内容
  }
  // 返回图是否被修改的标志
  return graph_modified;
}

} // namespace

// 对外接口函数，用于在整个图中折叠冻结的线性批归一化节点
bool FoldFrozenLinearBatchnorm(std::shared_ptr<Graph>& graph) {
  // 调用基本块级别的折叠函数
  bool graph_modified = FoldFrozenLinearBatchnorm(graph->block());
  // 执行死代码消除的优化
  EliminateDeadCode(graph);
  // 返回图是否被修改的标志
  return graph_modified;
}

} // namespace jit
} // namespace torch
```