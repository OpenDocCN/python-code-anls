# `.\pytorch\torch\csrc\jit\codegen\onednn\graph_fuser.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
// 引入用于 OneDNN 图形辅助功能的头文件
#include <torch/csrc/jit/ir/ir.h>
// 引入 Torch JIT IR 的头文件

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct WorkBlock : public std::pair<Node*, Node*> {
  using pair::pair;

  Node* begin() {
    return this->first;
  }
  // 返回 WorkBlock 的起始节点

  Node* end() {
    return this->second;
  }
  // 返回 WorkBlock 的结束节点
};

class GraphRewriter {
 public:
  GraphRewriter(Block* block, std::shared_ptr<Graph> graph, AliasDb& aliasDb)
      : block_(block),
        graph_(std::move(graph)),
        aliasDb_(aliasDb),
        llgaHelper_(graph_) {}
  // 构造函数，初始化 GraphRewriter 对象

  void cleanupSubgraphs();
  // 清理子图的方法声明

  void buildupSubgraphs();
  // 建立子图的方法声明

 private:
  Block* block_;
  // 指向 Block 的指针成员变量
  std::shared_ptr<Graph> graph_;
  // 指向 Graph 的共享指针成员变量
  AliasDb& aliasDb_;
  // 别名数据库的引用
  LlgaGraphHelper llgaHelper_;
  // 用于 OneDNN 图形辅助功能的对象
  std::vector<WorkBlock> buildWorkBlocks();
  // 构建 WorkBlock 向量的方法声明
  std::pair<graph_node_list::iterator, bool> scanNode(
      Node* consumer,
      graph_node_list::iterator workblock_begin);
  // 扫描节点的方法声明，返回迭代器和布尔值对
  std::optional<Node*> tryMerge(Node* consumer, Node* producer);
  // 尝试合并节点的方法声明，返回节点的可选指针
};

// This pass creates the subgraphs for oneDNN Graph Fusion Nodes.
// Its code-structure has been vastly inspired from
// torch/csrc/jit/passes/create_autodiff_subgraphs.cpp
void CreateLlgaSubgraphs(std::shared_ptr<Graph>& graph);
// 创建 OneDNN 图形融合节点的子图的函数声明

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```