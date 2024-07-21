# `.\pytorch\torch\csrc\jit\passes\clear_profiling.cpp`

```py
// 包含清除分析信息的头文件
#include <torch/csrc/jit/passes/clear_profiling.h>

// 包含 JIT 日志的头文件
#include <torch/csrc/jit/jit_log.h>

// 命名空间 torch 中的 JIT 模块
namespace torch {
namespace jit {

// 清除图中输入节点的分析信息
void unprofileGraphInputs(const std::shared_ptr<Graph>& graph) {
  // 遍历图的输入节点
  for (auto i : graph->inputs()) {
    // 如果节点类型是 TensorType 的子类型
    if (i->type()->isSubtypeOf(*TensorType::get())) {
      // 更新节点类型为未形状化的类型
      i->setType(unshapedType(i->type()));
    }
  }
}

// 清除基本块及其嵌套块中的分析信息
void unprofileBlock(Block* start_block) {
  // 使用堆栈追踪待处理的块
  std::vector<Block*> stack;
  stack.push_back(start_block);

  // 迭代处理堆栈中的块
  while (!stack.empty()) {
    // 弹出堆栈顶部的块
    Block* block = stack.back();
    stack.pop_back();

    // 遍历块中的节点
    for (auto n : block->nodes()) {
      // 遍历节点的输出
      for (auto o : n->outputs()) {
        // 如果输出类型是 TensorType 的子类型
        if (o->type()->isSubtypeOf(*TensorType::get())) {
          // 更新输出类型为未形状化的类型
          o->setType(unshapedType(o->type()));
        }
      }
      // 将节点的嵌套块加入堆栈末尾
      stack.insert(stack.end(), n->blocks().begin(), n->blocks().end());
    }
  }
}

// 清除图中的分析信息，并在清除后进行图的打印输出
void ClearProfilingInformation(const std::shared_ptr<Graph>& graph) {
  // 清除图的输入节点的分析信息
  unprofileGraphInputs(graph);
  // 清除图的基本块及其嵌套块的分析信息
  unprofileBlock(graph->block());
  // 在控制台输出清除分析信息后的图的结构
  GRAPH_DUMP("After ClearProfilingInformation: ", graph);
}

} // namespace jit
} // namespace torch
```