# `.\pytorch\torch\csrc\jit\passes\utils\subgraph_utils.h`

```py
#pragma once

# 预处理指令，确保头文件只被编译一次，防止重复包含


#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>

# 包含 Torch 库的头文件，用于导出符号、别名分析以及中间表示的操作


namespace torch {
namespace jit {

# 命名空间 torch::jit 开始


// Utilities for dealing with nodes that contain subgraphs.
//
// They handle the complexity of editing inputs/outputs as you merge nodes in
// and out of subgraphs.
namespace SubgraphUtils {

# SubgraphUtils 命名空间，提供处理包含子图节点的实用工具函数
# 这些函数处理将节点合并进出子图时的输入/输出编辑复杂性


// Create a new subgraph node that contains only `n`. The new subgraph will have
// `subgraphKind` as its type.
//
// `n` is destroyed.
//
// Returns the new subgraph node.
TORCH_API Node* createSingletonSubgraph(Node* n, Symbol subgraphKind);

# 创建一个只包含 `n` 的新子图节点，新子图将以 `subgraphKind` 作为其类型
# `n` 节点将被销毁
# 返回新的子图节点


// Creates a new subgraph that only contains `n`, amd updates the new outputs
// of the subgraph to have the aliasing properties of the original `n` outputs
TORCH_API Node* createSingletonSubgraphAndUpdateAliasing(
    Node* to_merge,
    Symbol subgraphKind,
    AliasDb& db);

# 创建一个只包含 `n` 的新子图，并更新子图的新输出，使其具有原始 `n` 输出的别名特性
# `to_merge` 节点将被合并
# `AliasDb& db` 别名数据库的引用


// Merge a node into a subgraph node. If `toMerge` is also a subgraph, the
// subgraphs are merged.
// If `destroyNode` is true `toMerge` is destroyed.
// An optional argument 'vmap' could be used to retrieve value mappings.
// Values will be mapped to their new subgraph values
TORCH_API void mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    bool destroyNode = true);

# 将一个节点合并到一个子图节点中。如果 `toMerge` 也是一个子图，则子图将被合并
# 如果 `destroyNode` 为 true，则销毁 `toMerge` 节点
# 可选参数 'vmap' 可用于检索值映射
# 值将映射到它们的新子图值


// Merges a node into a subgraph node, and updates the new outputs of the
// subgraph to have the aliasing properties of the corresponding `to_merge`
// outputs
TORCH_API void mergeNodeIntoSubgraphAndUpdateAliasing(
    Node* to_merge,
    Node* subgraphNode,
    AliasDb& db);

# 将一个节点合并到一个子图节点中，并更新子图的新输出，使其具有相应 `to_merge` 输出的别名属性
# `to_merge` 节点将被合并
# `AliasDb& db` 别名数据库的引用


TORCH_API std::vector<Node*> unmergeAliasedOutputs(
    Node* subgraphNode,
    AliasDb& db);

# 分离一个子图节点的别名输出
# `subgraphNode` 子图节点
# `AliasDb& db` 别名数据库的引用
# 返回包含已分离节点的向量


// Move nodes from a subgraph node to the outer graph.
// `subgraphNode` is destroyed.
TORCH_API void unmergeSubgraph(Node* subgraphNode);

# 将子图节点中的节点移动到外部图中
# 销毁 `subgraphNode` 子图节点


// Move `node_to_unmerge` and its descendants after `subgraphNode`
// promotes any dependencies of `node_to_unmerge` to subgraphNode outputs
TORCH_API void unmergeNode(Node* node_to_unmerge, Node* subgraphNode);

# 将 `node_to_unmerge` 及其后代节点移动到 `subgraphNode` 之后
# 将 `node_to_unmerge` 的任何依赖推广到 `subgraphNode` 的输出


TORCH_API bool unmergeOutputsAlisingInputs(Node* subgraphNode);

# 分离子图节点的别名输出，输入保持不变
# `subgraphNode` 子图节点


TORCH_API bool unmergeAliasedOutputs(Node* subgraphNode);

# 分离子图节点的别名输出
# `subgraphNode` 子图节点


// Convenience function
std::shared_ptr<Graph> getSubgraph(Node* n);

# 方便函数，获取节点 `n` 的子图


TORCH_API std::string generateNameForGraph(
    const std::shared_ptr<Graph>& graph,
    size_t maxlen = 40,
    const std::string& prefix = "fused");

# 为给定图 `graph` 生成一个名称，可指定最大长度和前缀


} // namespace SubgraphUtils
} // namespace jit
} // namespace torch

# SubgraphUtils 命名空间结束
# jit 命名空间结束
# torch 命名空间结束
```