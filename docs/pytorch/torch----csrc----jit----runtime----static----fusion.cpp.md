# `.\pytorch\torch\csrc\jit\runtime\static\fusion.cpp`

```
#include <torch/csrc/jit/runtime/static/fusion.h>

#include <ATen/core/symbol.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>

namespace torch::jit {

void createFusionGroups(Block* block, AliasDb* aliasDb, size_t min_size);

void fuseStaticSubgraphs(std::shared_ptr<Graph> graph, size_t min_size) {
  // 内联所有函数调用
  Inline(*graph);
  // 用复制替换节点
  ReplaceWithCopy(graph);
  // 可能用复制替换节点
  ReplaceWithMaybeCopy(graph);
  // 常量传播
  ConstantPropagation(graph);
  // 规范化图形
  Canonicalize(graph);
  // 常量传播
  ConstantPropagation(graph);
  // 移除张量变异
  RemoveTensorMutation(graph);
  // 常量传播
  ConstantPropagation(graph);
  // 消除死代码
  EliminateDeadCode(graph);
  // 创建别名数据库
  auto aliasDb = std::make_unique<AliasDb>(graph);
  // 创建融合组
  createFusionGroups(graph->block(), aliasDb.get(), min_size);
  // 常量池
  ConstantPooling(graph);
  // 常量传播
  ConstantPropagation(graph);
  // 消除死代码
  torch::jit::EliminateDeadCode(graph);
}

// 创建静态子图运行时
static Operation createStaticSubgraphRuntime(const Node* node) {
  // 获取子图
  auto g = node->g(attr::Subgraph);
  // 创建静态模块
  auto module = std::make_shared<torch::jit::StaticModule>(g);
  // 获取输入数量
  auto num_inputs = module->num_inputs();
  return [module, num_inputs](Stack& stack) {
    // 记录函数调用
    RECORD_FUNCTION("Static Runtime", std::vector<c10::IValue>());
    // 获取输入
    auto inps = torch::jit::last(stack, num_inputs);
    // 调用模块
    auto outputs = (*module)(inps.vec(), {});
    // 丢弃输入
    torch::jit::drop(stack, num_inputs);

    if (module->num_outputs() > 1) {
      // 如果输出数量大于1，将每个输出推入栈中
      for (auto& o : outputs.toTupleRef().elements()) {
        push_one(stack, std::move(o));
      }
    } else {
      // 否则将输出推入栈中
      push_one(stack, std::move(outputs));
    }
    return 0;
  };
}

// 注册静态子图操作符
RegisterOperators StaticSubgraphOps({torch::jit::Operator(
    prim::StaticSubgraph,
    createStaticSubgraphRuntime,
    AliasAnalysisKind::INTERNAL_SPECIAL_CASE)});

// 定义宏，用于检查条件是否满足
#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

// 检查节点是否可以处理
static bool canHandle(Node* node) {
  for (Value* input : node->inputs()) {
    // 检查输入是否为张量类型
    bool is_tensor = !!input->type()->cast<TensorType>();
    // 检查输入是否为列表类型
    auto list_type = input->type()->cast<ListType>();
    bool is_list = list_type && list_type->getElementType()->cast<TupleType>();
    // 检查输入是否为元组类型
    auto tuple_type = input->type()->cast<TupleType>();
    // 定义一个 lambda 函数，用于检查节点是否为元组
    bool is_tuple = [&]() -> bool {
      // 如果 tuple_type 为空指针，则不是元组
      if (!tuple_type) {
        return false;
      }
      // 遍历元组类型的每个元素
      for (auto& t : tuple_type->elements()) {
        // 如果某个元素不是 TensorType 类型，则不是元组
        if (!t->cast<TensorType>()) {
          return false;
        }
      }
      // 如果所有元素都是 TensorType 类型，则是元组
      return true;
    }();
    
    // 如果节点既不是张量，也不是列表或元组
    if (!(is_tensor || is_list || is_tuple)) {
      // 如果输入节点不是常量节点，则返回 false
      if (input->node()->kind() != prim::Constant) {
        return false;
      }
    }

  // 获取节点的种类
  auto kind = node->kind();
  // 如果节点的种类是基本类型操作符
  if (kind.is_prim()) {
    // 要求节点的种类必须是元组构造、列表构造或静态子图
    REQ(kind == prim::TupleConstruct || kind == prim::ListConstruct ||
        kind == prim::StaticSubgraph);
    
    // 如果节点是元组构造或列表构造
    if (kind == prim::TupleConstruct || kind == prim::ListConstruct) {
      // 遍历节点的所有输入
      for (Value* input : node->inputs()) {
        // 如果输入不是 TensorType 类型，则返回 false
        if (!input->type()->cast<TensorType>()) {
          return false;
        }
      }
    }
    // 如果以上条件都满足，则返回 true
    return true;
  }

  // TODO 添加 "canRunNatively" 一旦内存管理经过审计

  // 如果节点不是基本类型操作符，则检查是否可以原生运行该操作
  return getOutOfPlaceOperation(node) != nullptr;
}

static bool canMerge(Node* consumer, Node* producer, AliasDb* aliasDb) {
    // 仅在同一个基本块内进行融合
    REQ(consumer->owningBlock() == producer->owningBlock());

    // 符号检查
    REQ(canHandle(producer) || producer->kind() == prim::StaticSubgraph);
    TORCH_INTERNAL_ASSERT(
        consumer->kind() == prim::StaticSubgraph || canHandle(consumer));

    // 别名检查
    REQ(aliasDb->couldMoveBeforeTopologically(producer, consumer));

    // 返回别名的操作只能在这是唯一使用的情况下折叠
    if (producer->kind() == aten::slice || producer->kind() == aten::unsqueeze ||
        producer->kind() == prim::ConstantChunk) {
        for (auto& use : producer->output(0)->uses()) {
            REQ(use.user == consumer);
        }
    }

    return true;
}

static Node* getOrCreateStaticSubgraph(Node* n, AliasDb* aliasDb) {
    if (n->hasAttribute(attr::Subgraph) && n->kind() == prim::StaticSubgraph) {
        return n;
    }
    GRAPH_UPDATE("Creating a static subgraph::Group node from: ", *n);
    return SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
        n, prim::StaticSubgraph, *aliasDb);
}

static value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* b) {
    value_list result;
    for (auto i : inputs) {
        if (i->node()->owningBlock() == b) {
            result.push_back(i);
        }
    }
    // 按照逆拓扑顺序排序
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
        return a->node()->isAfter(b->node());
    });
    return result;
}

static void debugDumpFusionGroup(const std::string& msg, Node* n) {
    // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
    GRAPH_DEBUG(msg, *n);
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    if (n->kind() == prim::StaticSubgraph) {
        GRAPH_DEBUG(*n->g(attr::Subgraph));
    }
}

static std::optional<Node*> tryMerge(
    Node* fusion_group,
    Node* to_merge,
    AliasDb* aliasDb) {
    if (!canMerge(fusion_group, to_merge, aliasDb)) {
        return c10::nullopt;
    }

    std::vector<Node*> nodes_to_merge = {to_merge};

    if (to_merge->kind() == aten::cat) {
        Node* listconstruct = to_merge->input(0)->node();
        nodes_to_merge.push_back(listconstruct);
    }

    // 首先，尝试将所有要融合的节点移动到融合组旁边
    Node* move_point = fusion_group;
    for (auto n : nodes_to_merge) {
        GRAPH_UPDATE("Trying to move node next to fusion group: ", getHeader(n));
        if (!aliasDb->moveBeforeTopologicallyValid(n, move_point)) {
            GRAPH_UPDATE("Failed to move because of AliasDb checks!");
            return c10::nullopt;
        }
        move_point = n;
    }

    // 现在所有要融合的节点都已移动到融合组旁边，因此我们可以安全地将它们合并到融合组的子图中
    fusion_group = getOrCreateStaticSubgraph(fusion_group, aliasDb);

    for (auto n : nodes_to_merge) {
        GRAPH_UPDATE("Merging ", getHeader(n));
        SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
            n, fusion_group, *aliasDb);
    }
    return fusion_group;
}
}

static std::pair<graph_node_list::iterator, bool> createFusionGroup(
    Node* fusion_node,
    AliasDb* aliasDb) {
  // 调用函数以获取或创建静态子图，并更新 fusion_node 指针
  fusion_node = getOrCreateStaticSubgraph(fusion_node, aliasDb);

  // 输出调试信息，指示开始迭代地将输入节点合并到融合组中
  GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
  // 对融合节点的输入进行反向拓扑排序
  auto inputs =
      sortReverseTopological(fusion_node->inputs(), fusion_node->owningBlock());
  for (auto input : inputs) {
    // 输出调试信息，显示当前的融合组以及正在尝试合并的输入节点
    debugDumpFusionGroup("Current fusion group: ", fusion_node);
    GRAPH_DEBUG("Trying to merge: ", *input->node());
    if (auto maybe_fusion_group =
            tryMerge(fusion_node, input->node(), aliasDb)) {
      // 如果成功合并，则重新扫描新组以寻找更多合并机会，并返回迭代器和 true 表示有变化
      return std::make_pair(
          maybe_fusion_group.value()->reverseIterator(), true);
    }
  }

  // 返回 fusion_node 迭代器的下一个位置和 false 表示没有变化
  return std::make_pair(++fusion_node->reverseIterator(), false);
}

static std::pair<graph_node_list::iterator, bool> scanNode(
    Node* n,
    AliasDb* aliasDb) {
  // 输出调试信息，指示正在考虑的节点
  GRAPH_DEBUG("Considering node:", *n);

  // 如果节点无法处理，则直接返回下一个节点的迭代器和 false
  if (!canHandle(n)) {
    return std::make_pair(++n->reverseIterator(), false);
  }

  // 否则，调用 createFusionGroup 函数来尝试创建融合组
  return createFusionGroup(n, aliasDb);
}

static bool inlineIfTooSmall(Node* n, size_t min_size) {
  // 如果节点的类型不是静态子图，则返回 false
  if (n->kind() != prim::StaticSubgraph) {
    return false;
  }
  // 获取子图，并计算其中节点的数量
  auto subgraph = SubgraphUtils::getSubgraph(n);
  size_t num_nodes = std::distance(
      subgraph->block()->nodes().begin(), subgraph->block()->nodes().end());
  // 如果节点数量小于最小大小要求，则输出更新信息并解除子图的合并状态，返回 true
  if (num_nodes < min_size) {
    GRAPH_UPDATE("Fusion group is too small, unmerging: ", *n);
    SubgraphUtils::unmergeSubgraph(n);
    return true;
  }
  // 对子图进行常量池操作和常量传播
  ConstantPooling(subgraph);
  ConstantPropagation(subgraph);
  return false;
}

static void inlineSmallFusionGroups(Block* block, size_t min_size) {
  // 遍历块中的每个节点，并对其进行处理
  for (Node* n : block->nodes()) {
    // 对每个节点包含的子块递归调用 inlineSmallFusionGroups 函数
    for (Block* b : n->blocks()) {
      inlineSmallFusionGroups(b, min_size);
    }
    // 对当前节点尝试进行大小检查和内联处理
    inlineIfTooSmall(n, min_size);
  }
}

void createFusionGroups(Block* block, AliasDb* aliasDb, size_t min_size) {
  // 初始化一个标志来指示是否有任何变化
  bool any_changed = true;
  // 当有变化时，继续处理节点
  while (any_changed) {
    any_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      bool changed;
      // 扫描并尝试合并节点，更新 any_changed 标志
      std::tie(it, changed) = scanNode(*it, aliasDb);
      any_changed |= changed;
    }
  }

  // 对每个节点包含的子块递归调用 createFusionGroups 函数
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      createFusionGroups(b, aliasDb, min_size);
    }
  }

  // 尝试合并相邻的融合组，以处理未依赖于彼此的融合组的合并
  std::vector<Node*> initial_fusion_groups;
  // 收集所有的初始融合组
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::StaticSubgraph) {
      initial_fusion_groups.push_back(n);
  }
}

Node* prev_fusion_group =
    !initial_fusion_groups.empty() ? initial_fusion_groups[0] : nullptr;

for (const auto i : c10::irange(1, initial_fusion_groups.size())) {
  // 尝试将刚创建的融合组合并到前一个融合组中。
  // 如果合并失败，则将前一个融合组放入fusion_groups向量中，此后不再处理它。
  // 如果合并成功，将合并后的组保存为“前一个”融合组，以便尝试将下一个组合并到它中。

  Node* fusion_group = initial_fusion_groups[i];
  debugDumpFusionGroup(
      "Trying to merge into the previous fusion group: ", prev_fusion_group);
  if (auto merged_fusion_group =
          tryMerge(prev_fusion_group, fusion_group, aliasDb)) {
    prev_fusion_group = *merged_fusion_group;
    debugDumpFusionGroup(
        "Successfully merged into the previous fusion group: ",
        prev_fusion_group);
  } else {
    GRAPH_DEBUG("Cannot merge into the previous fusion group");
    prev_fusion_group = fusion_group;
  }
}
inlineSmallFusionGroups(block, min_size);
}

static void inlineFallbackGraphs(std::shared_ptr<Graph> graph) {
  // 创建一个深度优先遍历图的节点迭代器
  DepthFirstGraphNodeIterator it(graph);

  Node* n = nullptr;
  // 迭代器遍历图中的每个节点
  while ((n = it.next()) != nullptr) {
    // 检查节点是否为 prim::FallbackGraph 类型
    if (n->kind() == prim::FallbackGraph) {
      // 如果是，调用工具类函数解开子图
      SubgraphUtils::unmergeSubgraph(n);
    }
  }
}

void performTensorExprFusion(
    std::shared_ptr<Graph> graph,
    std::vector<IValue> sample_inputs) {
  // 启用支持动态形状的 TensorExpr 融合
  setTensorExprDynamicShapeFusionEnabled(true);
  // 输出融合前的图结构信息
  GRAPH_DEBUG("Graph before tracing: ", *graph);
  // 对图进行追踪，生成追踪后的图
  auto traced_graph = TraceGraph(graph, sample_inputs);
  // 输出追踪后的图结构信息
  GRAPH_DEBUG("Graph after tracing: ", *traced_graph);
  // 对追踪后的图进行 TensorExpr 融合
  FuseTensorExprs(
      traced_graph,
      /*min_group_size*/ 2,
      /*add_composed_op*/ true,
      /*fuse_to_dynamic_shapes*/ true);
  // 移除原始图的 Tensor 类型特化
  RemoveTensorTypeSpecializations(graph);
  // 内联处理所有的回退图
  inlineFallbackGraphs(traced_graph);
  // 清空原始图的块
  graph->block()->clear();
  // 克隆融合后的图的块到原始图的块中
  graph->block()->cloneFrom(traced_graph->block(), nullptr);
  // 输出融合后的图结构信息
  GRAPH_DUMP("Graph after fusion: ", graph);
}

} // namespace torch::jit
```