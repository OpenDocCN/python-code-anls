# `.\pytorch\torch\csrc\jit\passes\remove_exceptions.cpp`

```
// 引入Torch库中的头文件，用于异常处理和常量优化
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/remove_exceptions.h>

// 引入Torch库中的日志记录模块的头文件
#include <torch/csrc/jit/jit_log.h>

// 定义在torch命名空间内的jit命名空间
namespace torch {
namespace jit {

// 判断给定的基本块中是否存在抛出异常的操作
static bool certainlyThrows(Block* block) {
  // 遍历基本块中的每个节点
  for (Node* n : block->nodes()) {
    // 如果节点的类型是 prim::RaiseException，表示存在异常抛出
    if (n->kind() == prim::RaiseException) {
      return true;
    }
  }
  return false;  // 基本块中不存在抛出异常的操作
}

// 递归函数，用于消除图中所有基本块中的异常处理操作
static void EliminateExceptions(Block* block) {
  auto graph = block->owningGraph();
  // 在图中插入常量 false，表示布尔值 false
  Value* false_const = graph->insertConstant(IValue(false));
  // 在图中插入常量 true，表示布尔值 true
  Value* true_const = graph->insertConstant(IValue(true));
  // 遍历基本块中的每个节点
  for (Node* n : block->nodes()) {
    // 如果节点的类型是 prim::If
    if (n->kind() == prim::If) {
      // 获取 if 节点的两个分支块
      Block* true_block = n->blocks()[0];
      Block* false_block = n->blocks()[1];
      // 如果 true 分支块中存在抛出异常的操作
      if (certainlyThrows(true_block)) {
        // 将 if 节点的条件输入替换为 false_const
        n->input(0)->replaceAllUsesWith(false_const);
      } else if (certainlyThrows(false_block)) {
        // 如果 false 分支块中存在抛出异常的操作，则将条件输入替换为 true_const
        n->input(0)->replaceAllUsesWith(true_const);
      }
    }

    // 递归调用，处理当前节点内的所有子块
    for (Block* subblock : n->blocks()) {
      EliminateExceptions(subblock);
    }
  }
}

// 对给定的图进行异常消除处理
void EliminateExceptions(std::shared_ptr<Graph>& graph) {
  // 打印处理前的图状态信息
  GRAPH_DUMP("Before EliminateExceptions: ", graph);
  // 对图中的根基本块进行异常消除处理
  EliminateExceptions(graph->block());
  // 执行常量传播优化
  ConstantPropagation(graph);
  // 执行常量池优化
  ConstantPooling(graph);
  // 打印处理后的图状态信息
  GRAPH_DUMP("After EliminateExceptions: ", graph);
}

} // namespace jit
} // namespace torch
```