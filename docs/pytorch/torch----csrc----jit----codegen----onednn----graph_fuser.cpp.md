# `.\pytorch\torch\csrc\jit\codegen\onednn\graph_fuser.cpp`

```
// 引入 Torch 库中的头文件，包括了 OneDNN 的图形融合器相关功能
#include <torch/csrc/jit/codegen/onednn/graph_fuser.h>
// 引入 Torch 库中的头文件，包括了 OneDNN 的图形助手相关功能
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
// 引入 Torch 库中的头文件，包括了别名分析的实现
#include <torch/csrc/jit/ir/alias_analysis.h>
// 引入 Torch 库中的头文件，包括了公共子表达式消除的实现
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
// 引入 Torch 库中的头文件，包括了死代码消除的实现
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 引入 Torch 库中的头文件，包括了子图实用工具函数
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// Torch 的命名空间
namespace torch {
// JIT 编译器的命名空间
namespace jit {
// Fuser 模块的命名空间
namespace fuser {
// OneDNN 的命名空间
namespace onednn {

// 创建 LLGA 子图的函数，接收一个共享指针类型的图对象作为参数
void CreateLlgaSubgraphs(std::shared_ptr<Graph>& graph) {
  // 使用图对象创建别名数据库
  AliasDb db(graph);
  // 创建图重写器对象，用于重写图的块，传递图和别名数据库作为参数
  GraphRewriter graphRewriter(graph->block(), graph, db);
  
  // 通过递归构建所有子图来积累 LLGA 子图，同时在建立子图的过程中保持别名数据库的正确性
  graphRewriter.buildupSubgraphs();
  // 清理和合并小的子图，以便清理子图和取消内联自动微分子图时保持别名数据库的正确性
  graphRewriter.cleanupSubgraphs();
  
  // 全局运行一次公共子表达式消除，以消除内联子图可能导致的重复计算
  EliminateCommonSubexpression(graph);
  // 在图中运行一次死代码消除，以消除无用的计算节点
  EliminateDeadCode(graph);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```