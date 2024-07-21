# `.\pytorch\torch\csrc\jit\frontend\inline_loop_condition.cpp`

```
// 包含必要的头文件
#include <functional>
#include <memory>
#include <string>

// 引入 Torch 的相关头文件
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/inline_loop_condition.h>
#include <torch/csrc/jit/ir/ir.h>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 将给定节点之前的块插入到指定节点之前
void InlineBlockBeforeNode(Node* before_node, Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto block_node = *it++;
    block_node->moveBefore(before_node);
  }
}

// 内联循环条件的函数，用于优化循环结构
// 初始的循环节点如下所示：
// Loop(max_trip_count)
//    block0(loop_counter) {
//      <body>
//    }
//    block1 {
//      <loop condition>
//      -> (condition)
//    }
// 在此函数中，我们内联循环条件，将循环转换为以下形式：
// Loop(max_trip_count, start_condition)
//    block0(loop_counter, loop_carried_block*) {
//      <body>
//       BlockExit(continue_condition, loop_carried_block*)
//    }
static void inlineLoopCondition(Node* n) {
  // 获取循环体和前置块
  Block* body_block = n->blocks().at(0);
  auto pre_header = n->blocks().at(1);

  // 创建临时块并移动前置块的内容到临时块
  auto temp_block = n->addBlock();
  temp_block->cloneFrom(pre_header, [](Value* v) { return v; });
  InlineBlockBeforeNode(n, temp_block);

  // 将临时块的输出作为循环的起始条件
  n->insertInput(/*start_condition_index*/ 1, temp_block->outputs().at(0));
  n->eraseBlock(2); // 删除原始的前置块

  // 内联循环体的返回节点到前置块
  InlineBlockBeforeNode(body_block->return_node(), pre_header);
  body_block->return_node()->insertInput(0, pre_header->outputs().at(0));
  n->eraseBlock(1); // 删除原始的循环条件块
}

// 递归地内联循环条件，处理整个图中的所有节点
static void inlineLoopCondition(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      inlineLoopCondition(b);
    }
    if (n->kind() == prim::Loop) {
      inlineLoopCondition(n);
    }
  }
}

// 公共接口函数，用于开始内联循环条件的处理过程
void InlineLoopCondition(std::shared_ptr<Graph>& graph) {
  inlineLoopCondition(graph->block());
}

} // namespace torch::jit
```