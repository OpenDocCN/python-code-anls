# `.\pytorch\torch\csrc\jit\codegen\onednn\decompose_silu.cpp`

```
// 包含 Torch 的相关头文件，用于 OneDNN 的 SILU 函数分解
#include <torch/csrc/jit/codegen/onednn/decompose_silu.h>
#include <torch/csrc/jit/codegen/onednn/operator.h>

// 包含 ATen 的代码模板
#include <ATen/code_template.h>
// 包含 Torch 的消除死代码优化 Pass
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 包含 Torch 的子图重写 Pass
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {
// Fuser 命名空间
namespace fuser {
// OneDNN 命名空间
namespace onednn {

// 判断节点是否应该被分解为 SILU 函数
static bool shouldDecomposeSilu(Node* node) {
  // 如果节点不是 silu 类型，则不分解
  if (node->kind() != aten::silu) {
    return false;
  }
  // 获取输入到 SILU 函数的节点
  auto inputToSilu = node->input(0)->node();
  // 如果输入节点是卷积操作
  if (inputToSilu->kind() == aten::_convolution) {
    // TODO: 当桥接支持 ConvTranspose 后，删除此处的转置检查
    // 获取第六个参数的布尔值，表示是否转置
    bool transposed = Operator::Bool(inputToSilu, 6);
    // 如果不是转置，则可以分解 SILU 函数
    return !transposed;
  }
  // 如果输入节点是线性操作，则可以分解 SILU 函数
  if (inputToSilu->kind() == aten::linear) {
    return true;
  }
  // 其他情况不分解 SILU 函数
  return false;
}

// 对单个节点进行 SILU 函数分解
static void DecomposeSilu(Node* node) {
  // 如果节点应该被分解为 SILU 函数
  if (shouldDecomposeSilu(node)) {
    // 获取输入节点的张量类型
    auto dtype = node->input(0)->type()->expect<TensorType>();

    // 在节点的插入点创建一个新的图
    WithInsertPoint guard(node);
    auto g = node->owningGraph();

    // 插入 sigmoid 操作，并设置其类型为输入节点的张量类型
    auto sigmoid = g->insert(aten::sigmoid, {node->input(0)});
    sigmoid->setType(dtype);

    // 插入乘法操作，计算 sigmoid(node->input(0)) * node->input(0)，并设置类型为输入节点的张量类型
    auto mul = g->insert(aten::mul, {sigmoid, node->input(0)});
    mul->setType(dtype);

    // 替换原节点输出的所有使用为新插入的乘法操作
    node->output()->replaceAllUsesWith(mul);
  }
}

// 递归地在整个块中进行 SILU 函数分解
static void DecomposeSilu(Block* block) {
  for (auto node : block->nodes()) {
    // 递归处理每个节点的子块
    for (auto sub : node->blocks()) {
      DecomposeSilu(sub);
    }

    // 如果当前节点是 SILU 操作，则对其进行分解
    if (node->kind() == aten::silu) {
      DecomposeSilu(node);
    }
  }
}

// 对给定图中的所有节点执行 SILU 函数分解，并进行死代码消除优化
void DecomposeSiluForLLGA(std::shared_ptr<Graph>& graph) {
  // 对图的主块执行 SILU 函数分解
  DecomposeSilu(graph->block());
  // 执行死代码消除优化
  EliminateDeadCode(graph);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```