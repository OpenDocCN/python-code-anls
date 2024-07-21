# `.\pytorch\torch\csrc\jit\passes\clear_undefinedness.cpp`

```
// 包含 Torch 中用于清除未定义值的 Pass 所需的头文件
#include <torch/csrc/jit/passes/clear_undefinedness.h>

// 包含 Torch JIT 日志记录的头文件
#include <torch/csrc/jit/jit_log.h>

// 定义 Torch JIT 的命名空间
namespace torch {
namespace jit {

// 清除未定义值的辅助函数，处理单个值
static void clearUndefinedness(Value* o) {
  // 如果值的类型是张量类型
  if (o->type()->kind() == TensorType::Kind) {
    // 将该值的类型设置为通用张量类型
    o->setType(TensorType::get());
  } else if (
      // 如果值的类型是列表类型，并且列表中元素的类型是张量类型
      o->type()->kind() == ListType::Kind &&
      o->type()->expectRef<ListType>().getElementType()->kind() ==
          TensorType::Kind) {
    // 将该值的类型设置为张量类型的列表类型
    o->setType(ListType::create(TensorType::get()));
  }
}

// 清除未定义值的辅助函数，处理整个基本块
static void clearUndefinedness(Block* block) {
  // 遍历基本块中的每一个节点
  for (auto n : block->nodes()) {
    // 处理节点的每一个输出值
    for (auto o : n->outputs()) {
      clearUndefinedness(o);
    }
    // 递归处理节点内部的每一个子块
    for (auto ib : n->blocks()) {
      clearUndefinedness(ib);
    }
  }
}

// 清除未定义值的入口函数，处理整个图
void ClearUndefinedness(const std::shared_ptr<Graph>& graph) {
  // 处理图的每一个输入值
  for (auto i : graph->inputs()) {
    clearUndefinedness(i);
  }
  // 清除整个图的未定义值
  clearUndefinedness(graph->block());
  // 记录清除未定义值后的图结构
  GRAPH_DUMP("After removeUndefinedness: ", graph);
}

} // namespace jit
} // namespace torch
```