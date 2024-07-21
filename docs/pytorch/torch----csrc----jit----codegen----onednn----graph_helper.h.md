# `.\pytorch\torch\csrc\jit\codegen\onednn\graph_helper.h`

```
#pragma once
// 防止头文件多重包含

#include <oneapi/dnnl/dnnl_graph.hpp>
// 包含 oneDNN 图形编程接口头文件

#include <torch/csrc/jit/codegen/onednn/operator.h>
// 包含 Torch 的 OneDNN 操作符头文件

#include <torch/csrc/jit/ir/alias_analysis.h>
// 包含 Torch 的别名分析头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的中间表示(IR)头文件

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

#define STRIDED_LAYOUT 0
// 定义分块映射类型的分块布局常量：分块布局方式为分块映射

#define OPAQUE_LAYOUT 1
// 定义分块映射类型的分块布局常量：分块布局方式为不透明映射

struct OpPartitionMap {
  void add(uint64_t opId, uint64_t partitionId) {
    opmap_[opId] = partitionId;
  }
  // 向操作符映射表中添加操作符 ID 和分块 ID 的映射关系

  void add(Node* n, uint64_t partitionId) {
    add(Operator::getId(n), partitionId);
  }
  // 向操作符映射表中添加节点对应的操作符 ID 和分块 ID 的映射关系

  bool has(uint64_t opId) {
    return opmap_.count(opId) > 0;
  }
  // 检查是否存在给定操作符 ID 的映射关系

  bool has(Node* n) {
    return has(Operator::getId(n));
  }
  // 检查是否存在给定节点对应的操作符 ID 的映射关系

  uint64_t get(uint64_t opId) {
    return opmap_[opId];
  }
  // 获取给定操作符 ID 对应的分块 ID

  uint64_t get(Node* n) {
    auto opId = Operator::getId(n);
    TORCH_CHECK(
        has(opId),
        "Node ",
        n->kind().toQualString(),
        " does not belong to any LLGA partition");
    return get(opId);
  }
  // 获取给定节点对应的操作符 ID，然后根据此 ID 获取分块 ID，并进行错误检查

 private:
  std::unordered_map<uint64_t, uint64_t> opmap_;
  // 操作符 ID 到分块 ID 的映射表
};

class LlgaGraphHelper {
 public:
  LlgaGraphHelper(
      const std::shared_ptr<Graph>& graph,
      dnnl::graph::partition::policy policy =
          dnnl::graph::partition::policy::fusion);
  // 构造函数：初始化 LLGA 图形辅助类，接收图形和分块策略作为参数

  bool shouldMerge(Node* toMerge, Node* subgraph);
  // 判断是否应该将指定节点合并到子图中

  bool shouldConsiderForMerge(Node* node);
  // 判断节点是否应考虑合并

  bool checkForSingleOpPartition(Node* node);
  // 检查节点是否为单操作符分块

  Node* createSingletonSubgraph(Node* n, AliasDb& db);
  // 创建包含单一节点的子图

  void mergeNodeIntoSubgraph(Node* toMerge, Node* subgraphNode, AliasDb& db);
  // 将节点合并到子图中

  void unmergeIfAnyNodeIsMissing(Node* subgraphNode);
  // 如果任何节点丢失，则取消合并

  static bool isLlgaSubgraph(const Node* node);
  // 静态方法：判断给定节点是否为 LLGA 子图

  Operator makeEltwiseOp(Node* node, dnnl::graph::op::kind kind);
  // 根据节点和操作种类创建元素操作符

  Operator makeBinaryOp(Node* node, dnnl::graph::op::kind kind);
  // 根据节点和操作种类创建二元操作符

  std::vector<dnnl::graph::partition> getPartitions() const;
  // 获取分块列表

  std::map<size_t, Value*> getTensorIdToValue() const;
  // 获取张量 ID 到值的映射

  Operator createOperator(Node* node);
  // 创建操作符

 private:
  size_t countSupportedOps(const std::shared_ptr<Graph>& graph) const;
  // 计算图中支持的操作数

  std::unique_ptr<dnnl::graph::graph> dnnl_graph_ = nullptr;
  // 指向 OneDNN 图形的独特指针

  std::unique_ptr<torch::jit::AliasDb> aliasDb_ = nullptr;
  // 指向 Torch 别名数据库的独特指针

  OpPartitionMap opToOwningPartition_;
  // 操作符到所属分块的映射

  std::vector<dnnl::graph::partition> partitions_;
  // 分块列表

  std::map<size_t, Value*> tensorIdToValue_;
  // 张量 ID 到 Torch 值的映射表
};

class LlgaNodeWrapper {
 public:
  LlgaNodeWrapper(const Node* node);
  // 构造函数：初始化 LLGA 节点包装器

  void setOpaqueLayout(size_t offset);
  // 设置不透明布局的偏移量

  bool useOpaqueLayout(size_t offset) const;
  // 使用不透明布局的偏移量

  friend class LlgaGraphHelper;
  // 将 LLGA 图形辅助类声明为友元类

 private:
  Node* n;
  // 节点指针
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
```