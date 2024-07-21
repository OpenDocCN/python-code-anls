# `.\pytorch\torch\csrc\jit\passes\lower_grad_of.cpp`

```py
#include <torch/csrc/jit/passes/lower_grad_of.h>
// 包含 LowerGradOf 所需的头文件

#include <torch/csrc/jit/jit_log.h>
// 包含 JIT 编译器的日志头文件

namespace torch {
namespace jit {

void LowerGradOf(Graph& g) {
  // 遍历计算图中的每个节点
  for (auto it = g.nodes().begin(); it != g.nodes().end(); ++it) {
    // 如果当前节点的类型是 prim::GradOf
    if (it->kind() == prim::GradOf) {
      // 设置插入点为当前节点 it
      WithInsertPoint guard(*it);
      
      // 插入一个 prim::AutogradAnyNonZero 节点作为条件
      auto cond = g.insertNode(g.create(prim::AutogradAnyNonZero, it->inputs()))
                      ->output()
                      ->setType(IntType::get());
      
      // 插入一个 prim::If 节点，根据条件选择执行的块数
      auto if_stat =
          g.insertNode(g.create(prim::If, {cond}, it->outputs().size()));
      
      // 将当前 GradOf 节点的第一个块克隆到 if_stat 的第一个块中
      if_stat->addBlock()->cloneFrom(
          it->blocks().at(0), [](Value* v) { return v; });
      
      // 添加 else 块，并生成 autograd zero tensors
      auto else_block = if_stat->addBlock();
      auto undef = g.createAutogradZero()
                       ->insertBefore(else_block->return_node())
                       ->output();
      
      // 将 autograd zero tensors 注册为 else 块的输出
      for (size_t i = 0; i < it->outputs().size(); ++i) {
        else_block->registerOutput(undef);
        if_stat->outputs().at(i)->copyMetadata(it->outputs().at(i));
      }
      
      // 更新计算图中节点的替换和销毁
      GRAPH_UPDATE("Replacing ", getHeader(*it), " with ", getHeader(if_stat));
      it->replaceAllUsesWith(if_stat);
      it.destroyCurrent();
    }
  }
}

} // namespace jit
} // namespace torch
```