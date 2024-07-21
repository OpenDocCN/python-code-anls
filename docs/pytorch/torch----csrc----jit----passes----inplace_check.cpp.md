# `.\pytorch\torch\csrc\jit\passes\inplace_check.cpp`

```
// 包含 Torch 库中的头文件，用于 JIT pass 的原位操作检查
#include <torch/csrc/jit/passes/inplace_check.h>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {

// 静态函数，用于检查指定基本块中的原位操作
static void CheckInplace(Block* block) {
  // 遍历基本块中的每个节点
  for (auto node : block->nodes()) {
    // 如果节点是 PythonOp 类型且具有 inplace 属性
    if (node->kind() == prim::PythonOp && node->hasAttribute(attr::inplace)) {
      // 如果 inplace 属性为真
      if (node->i(attr::inplace)) {
        // 抛出运行时错误，说明 JIT 不支持此处的原位操作
        throw std::runtime_error(
            std::string("inplace ") + static_cast<PythonOp*>(node)->name() +
            " not supported in the JIT");
      }
    }
  }
}

// 函数重载，检查图中所有基本块的原位操作
void CheckInplace(std::shared_ptr<Graph>& graph) {
  // 调用上面定义的 CheckInplace 函数，传入图的根基本块
  CheckInplace(graph->block());
}

} // namespace jit
} // namespace torch
```