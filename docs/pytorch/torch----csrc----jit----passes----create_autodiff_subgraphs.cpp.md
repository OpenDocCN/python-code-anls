# `.\pytorch\torch\csrc\jit\passes\create_autodiff_subgraphs.cpp`

```
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/remove_redundant_profiles.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/autodiff.h>

namespace torch {
namespace jit {

namespace {

// 定义一个结构体 WorkBlock，继承自 std::pair<Node*, Node*>
struct WorkBlock : public std::pair<Node*, Node*> {
  using pair::pair;

  // 返回第一个节点指针
  Node* begin() {
    return this->first;
  }
  // 返回第二个节点指针
  Node* end() {
    return this->second;
  }
};

// 定义一个子类 SubgraphSlicer，用于处理子图切片
class SubgraphSlicer {
 public:
  // 构造函数，初始化 SubgraphSlicer 对象
  SubgraphSlicer(
      Block* block,
      std::shared_ptr<Graph> graph,
      size_t minSubgraphSize,
      AliasDb& aliasDb,
      std::vector<Node*>& diff_nodes)
      : block_(block),
        graph_(std::move(graph)),
        minSubgraphSize_(minSubgraphSize),
        aliasDb_(aliasDb),
        diff_nodes_(diff_nodes) {}

  // 运行切片处理逻辑
  void run() {
    // 在构建自动微分子图时，我们需要保持 alias db 的正确性，但是在取消内联自动微分子图时，难以保持正确性。
    // 首先递归构建所有子图，然后递归清理和拆分小子图。
    buildupSubgraphs();
    GRAPH_DUMP("before unfuseAliasedOutputs", graph_);
    unfuseAliasedOutputs(block_);
    cleanupSubgraphs();
    // 运行全局的公共子表达式消除，消除内联子图可能导致的重复。
    EliminateCommonSubexpression(graph_);
  }

  // 清理子图
  void cleanupSubgraphs() {
    auto curNode = *block_->nodes().rbegin();
    while (curNode != *block_->nodes().rend()) {
      // 保存前一个节点，因为我们可能会在下一个块中删除 curNode
      auto prevNode = curNode->prev();
      if (curNode->kind() == prim::DifferentiableGraph) {
        // 内联节点可能导致一些子表达式重新出现在子图中（例如，重复复制常量将生成冗余的 prim::Constants）。
        // 运行 CSE 清理它们。
        EliminateCommonSubexpression(curNode->g(attr::Subgraph));

        // 如果子图太小，无法内联，则将其添加到 diff_nodes_ 中
        if (!inlineIfTooSmall(curNode)) {
          diff_nodes_.push_back(curNode);
        }
      }
      curNode = prevNode;
    }

    // 递归处理当前节点的所有块
    for (Node* n : block_->nodes()) {
      for (Block* b : n->blocks()) {
        SubgraphSlicer(b, graph_, minSubgraphSize_, aliasDb_, diff_nodes_)
            .cleanupSubgraphs();
      }
    }
  }

  // 构建子图
  void buildupSubgraphs() {
    // 我们需要多次运行切片器以获取所有合并机会。
    // 这是因为 moveBeforeTopologicalValid 可能会重新排列节点，使其在当前迭代点之后。
    // 为了正确考虑这些节点进行合并，我们需要运行该过程，直到不再有更改为止。
    //
    // 示例:
    //   c = f(a, b)
    //
    // 在此示例中，f 可能会在当前节点之后被重新排序，我们需要考虑这种情况以进行合并。
    //
    while (moveBeforeTopologicalValid(block_)) {
      // 继续移动节点直到拓扑排序有效
    }
  }

 private:
  Block* block_;                           // 当前块指针
  std::shared_ptr<Graph> graph_;           // 图指针
  size_t minSubgraphSize_;                 // 最小子图大小
  AliasDb& aliasDb_;                       // 别名数据库引用
  std::vector<Node*>& diff_nodes_;         // 不同节点的向量引用
};

} // namespace
} // namespace jit
} // namespace torch
    // 构建工作块列表，用于划分不同的工作单元
    auto workblocks = buildWorkBlocks();
    // 遍历每个工作块
    for (auto& workblock : workblocks) {
      bool any_changed = true;
      // 当存在改变时循环执行
      while (any_changed) {
        any_changed = false;
        // 从工作块末尾向开头反向遍历节点
        for (auto it = workblock.end()->reverseIterator();
             it != workblock.begin()->reverseIterator();) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool changed;
          // 执行节点扫描操作，并获取是否有改变的标志
          std::tie(it, changed) = scanNode(*it);
          any_changed |= changed; // 更新改变标志
        }
      }
    }

    // 递归构建子图
    for (Node* n : block_->nodes()) {
      // 遍历节点的每个子块
      for (auto subBlock : n->blocks()) {
        // 通过子图切片器构建子图
        SubgraphSlicer(
            subBlock, graph_, minSubgraphSize_, aliasDb_, diff_nodes_)
            .buildupSubgraphs();
      }
    }
  }

 private:
  // 解除别名输出
  void unfuseAliasedOutputs(Block* b) {
    bool any_changed = true;
    // 当存在改变时循环执行
    while (any_changed) {
      any_changed = false;
      // 反向遍历节点，以便跳过可能在当前 DifferentiableGraph 后解除融合的节点
      for (auto n : b->nodes().reverse()) {
        if (n->kind() == prim::DifferentiableGraph) {
          // DifferentiableGraph 中的别名输出必须解除融合
          // 注意，使用 |= 以确保 unfuseAliasedOutputs 不会短路
          any_changed |= SubgraphUtils::unmergeAliasedOutputs(n);
          any_changed |= SubgraphUtils::unmergeOutputsAlisingInputs(n);
          // 调试输出，显示是否有改变以及相关子图的信息
          GRAPH_DEBUG(
              "any_changed on ",
              any_changed,
              " ",
              n->g(attr::Subgraph)->toString(false));
        }
      }
    }

    // 递归处理每个节点的每个子块
    for (Node* n : b->nodes()) {
      for (Block* ib : n->blocks()) {
        unfuseAliasedOutputs(ib);
      }
    }
  }

  // 构建工作块的向量
  std::vector<WorkBlock> buildWorkBlocks() {
    // [workblocks]
    // IR 中有许多节点无法重新排序，例如 prim::Bailout
    // 如果节点 N 被两个无法重新排序的节点 A 和 B 包围，则从 N 创建的可微分子图只能包含 (A, B) 中的节点
    // A 到 B 之间的节点代表子图切片器要处理的一个工作块
    // 通过提前创建这些工作块，我们避免每次 scanNode 返回时重新遍历整个图块，并且可以避免在不包含 >= minSubgraphSize_ 可微分节点的工作块中尝试创建可微分子图
    Node* end_bound_node = block_->return_node();
    Node* curr = end_bound_node->prev();

    std::vector<WorkBlock> worklist;
    size_t differentiable_nodes = 0;
    // 当前节点不是目标块的参数节点时执行循环
    while (curr != block_->param_node()) {
      // 将应考虑合并的不同iable节点计数加一
      differentiable_nodes += shouldConsiderForMerge(curr);

      // 不能在具有副作用的节点周围重新排序
      if (curr->hasSideEffects()) {
        // 如果不同iable节点足够多以创建不同iable子图
        if (differentiable_nodes >= minSubgraphSize_) {
          // 将当前节点和结束绑定节点添加到工作列表中
          worklist.emplace_back(curr, end_bound_node);
        }
        // 重置不同iable节点计数，并设置结束绑定节点为当前节点
        differentiable_nodes = 0;
        end_bound_node = curr;
      }
      // 将当前节点更新为前一个节点
      curr = curr->prev();
    }

    // 如果不同iable节点数量达到最小子图大小要求，将当前节点和结束绑定节点添加到工作列表中
    if (differentiable_nodes >= minSubgraphSize_) {
      worklist.emplace_back(curr, end_bound_node);
    }

    // 返回工作列表
    return worklist;
  }

  // 如果该节点的子图小于指定的最小大小，则将其内联到外部图中
  //
  // 如果发生内联，则返回true，否则返回false
  bool inlineIfTooSmall(Node* n) {
    AT_ASSERT(n->kind() == prim::DifferentiableGraph);
    // 获取节点的子图
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t i = 0;
    // 遍历子图中的节点
    for (auto it = subgraph->nodes().begin(); it != subgraph->nodes().end();
         ++it) {
      // 统计非未执行操作的节点数
      i += !it->notExecutedOp();
      // 如果节点数超过或等于最小子图大小，返回false
      if (i >= minSubgraphSize_) {
        return false;
      }
    }

    // 取消子图的合并
    SubgraphUtils::unmergeSubgraph(n);
    // 返回true表示已执行内联
    return true;
  }

  // 对输入的值进行反向拓扑排序
  value_list sortReverseTopological(ArrayRef<Value*> inputs) {
    value_list result;
    // 遍历输入的值
    for (auto i : inputs) {
      // 如果值所在节点的块等于当前块，将值添加到结果列表中
      if (i->node()->owningBlock() == block_) {
        result.push_back(i);
      }
    }
    // 按照反向拓扑顺序排序结果列表
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    // 返回排序后的结果列表
    return result;
  }

  // 判断节点是否为视图操作
  bool isViewOp(Node* n) {
    switch (n->kind()) {
      // 如果节点的种类是以下视图操作之一，返回true
      case aten::view:
      case aten::view_as:
      case aten::reshape:
      case aten::reshape_as:
      case aten::transpose:
      case aten::expand:
      case aten::expand_as:
        return true;
    }
    // 否则返回false
    return false;
  }

  // 判断是否应该考虑将节点合并到不同iable子图中
  bool shouldConsiderForMerge(Node* node) {
    // 如果节点已经在合并过程中
    if (node->kind() == prim::DifferentiableGraph) {
      return true;
    }
    // 如果节点是常量节点，不考虑合并
    if (node->kind() == prim::Constant) {
      return false;
    }

    // 视图操作作为不同iable子图的输出可能会导致不正确的微分，暂时不包括在子图中
    if (isViewOp(node)) {
      return false;
    }

    // 判断节点是否是可微分的
    return isDifferentiable(node);
  }

  // 扫描消费者节点
  std::pair<graph_node_list::iterator, bool> scanNode(Node* consumer) {
    // 如果应考虑将当前节点（consumer）与其输入节点合并
    if (shouldConsiderForMerge(consumer)) {
      // 如果当前节点的类型不是 DifferentiableGraph
      if (consumer->kind() != prim::DifferentiableGraph) {
        // 创建一个单节点子图，并更新别名信息，将当前节点转换为 DifferentiableGraph 类型
        consumer = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
            consumer, prim::DifferentiableGraph, aliasDb_);
      }
      // 对当前节点的输入节点进行反向拓扑排序
      auto inputs = sortReverseTopological(consumer->inputs());
      // 遍历当前节点的输入节点
      for (auto input : inputs) {
        // 尝试将当前节点与输入节点进行合并
        if (auto group = tryMerge(consumer, input->node())) {
          // 如果合并成功，新组（group）的输入可能已经改变，因此重新扫描新组以寻找更多合并机会
          return std::make_pair(group.value()->reverseIterator(), true);
        }
      }
    }

    // 如果不需要进行合并或者无法成功合并，则返回当前节点的下一个节点迭代器以及失败标志
    return std::make_pair(++consumer->reverseIterator(), false);
  }

  // 尝试将 producer 节点合并到 consumer 节点中。如果成功，销毁 producer 并返回 consumer 组
  std::optional<Node*> tryMerge(Node* consumer, Node* producer) {
    // 断言 consumer 节点的类型为 DifferentiableGraph
    AT_ASSERT(consumer->kind() == prim::DifferentiableGraph);
    // 检查是否可以将 producer 合并到 consumer 中，并在拓扑上移动 producer
    bool canMerge = shouldConsiderForMerge(producer) &&
        aliasDb_.moveBeforeTopologicallyValid(producer, consumer);

    // 如果无法合并，则返回空 optional
    if (!canMerge) {
      return c10::nullopt;
    }

    // 将 producer 节点合并到 consumer 子图中，并更新别名信息
    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        producer, consumer, aliasDb_);
    // 返回成功合并后的 consumer 节点
    return consumer;
  }

  // 指向当前块的指针
  Block* block_;
  // 共享指针，指向当前图
  std::shared_ptr<Graph> graph_;
  // 子图的最小大小
  size_t minSubgraphSize_;
  // 别名数据库的引用
  AliasDb& aliasDb_;
  // 可变节点的引用向量
  std::vector<Node*>& diff_nodes_;
};

// 获取 Profile 节点的 requires_grad 属性的可选值
std::optional<bool> getProfileNodeRequiresGrad(Node* n) {
  // 断言节点的类型为 prim::profile
  TORCH_INTERNAL_ASSERT(n->kind() == prim::profile);
  // 如果节点没有属性 attr::profiled_type，则返回空
  if (!n->hasAttribute(attr::profiled_type)) {
    return c10::nullopt;
  }
  // 获取节点的 profiled_type 属性
  auto& type = n->ty(attr::profiled_type);
  // 如果节点的类型不是 TensorType，则返回空
  if (type->castRaw<TensorType>() == nullptr) {
    return c10::nullopt;
  }
  // 返回 TensorType 的 requiresGrad 属性值
  return type->expectRef<TensorType>().requiresGrad();
}

// 上下文映射，用于跟踪节点的上下文信息
struct ContextMapping {
  std::vector<const Node*> ctx_stack_;  // 上下文节点堆栈
  std::unordered_map<const Node*, const Node*> node_to_ctx_;  // 节点到上下文节点的映射

  // 处理单个节点，更新节点到上下文的映射关系
  void processNode(Node* n) {
    node_to_ctx_[n] = ctx_stack_.back();

    // 如果节点是 prim::Enter，则将其添加到上下文堆栈
    if (n->kind() == prim::Enter) {
      ctx_stack_.push_back(n);
    }
    // 如果节点是 prim::Exit，则从上下文堆栈中弹出
    else if (n->kind() == prim::Exit) {
      ctx_stack_.pop_back();
    }
  }

  // 递归处理节点的块
  void processBlock(Block* block) {
    for (Node* n : block->nodes()) {
      processNode(n);  // 处理当前节点
      for (Block* b : n->blocks()) {
        processBlock(b);  // 递归处理子块
      }
      // 如果节点是 prim::DifferentiableGraph，则进一步处理其子图
      if (n->kind() == prim::DifferentiableGraph) {
        const auto& subgraph = n->g(attr::Subgraph);
        processBlock(subgraph->block());
      }
    }
  }

  // 构造函数，初始化上下文映射
  ContextMapping(const std::shared_ptr<Graph>& graph) {
    ctx_stack_.push_back(nullptr);  // 初始时堆栈添加一个空指针作为根上下文
    processBlock(graph->block());   // 处理整个图的节点
  }

  // 获取节点对应的上下文节点
  const Node* get(const Node* n) const {
    auto it = node_to_ctx_.find(n);
    // 断言确保能够找到节点对应的上下文节点
    TORCH_INTERNAL_ASSERT(
        it != node_to_ctx_.end(),
        "Cannot find node in node-to-context mapping.");
    return it->second;
  }

  // 检查映射中是否包含节点
  bool has(const Node* n) const {
    return node_to_ctx_.find(n) != node_to_ctx_.end();
  }
};

// 查找输出的 requires_grad 属性是否在指定上下文映射中
std::optional<bool> findRequiresGradForOutput(
    Node* diff_graph,
    Value* output,
    const ContextMapping& ctx_mapping) {
  for (auto& use : output->uses()) {
    // [仅考虑相同上下文中的 profile 节点]
    // 如果使用节点存在于不同的上下文中，则忽略 profiled 的使用
    // 例如，在 no_grad() 上下文中的 profile 节点将记录错误的 requires_grad 信息
    if (ctx_mapping.has(use.user) &&
        ctx_mapping.get(use.user) != ctx_mapping.get(diff_graph)) {
      continue;
    }

    // 如果使用节点是 prim::profile 类型
    if (use.user->kind() == prim::profile) {
      std::optional<bool> req_grad_use;
      // 获取 profile 节点的 requires_grad 属性值
      if ((req_grad_use = getProfileNodeRequiresGrad(use.user)).has_value()) {
        return req_grad_use.value();  // 返回 requires_grad 属性值
      }
    }

    // 可能 profile 节点已经被吸收到可微图中
    // 检查 use 所引用的节点是否是 DifferentiableGraph 类型
    if (use.user->kind() == prim::DifferentiableGraph) {
      // 获取 DifferentiableGraph 的子图属性
      const auto& dg = use.user->g(attr::Subgraph);
      // 针对该图输入的所有使用，查找 profile 节点。
      Value* dg_value = dg->inputs()[use.offset];
      for (auto& dg_use : dg_value->uses()) {
        // 只考虑与当前上下文相同的 profile 节点
        if (ctx_mapping.has(dg_use.user) &&
            ctx_mapping.get(dg_use.user) != ctx_mapping.get(diff_graph)) {
          continue;
        }

        // 如果使用节点是 profile 类型
        if (dg_use.user->kind() == prim::profile) {
          // 获取 profile 节点是否需要梯度的信息
          std::optional<bool> req_grad_use;
          // 检查是否成功获取到需要梯度的信息，并返回其值
          if ((req_grad_use = getProfileNodeRequiresGrad(dg_use.user))
                  .has_value()) {
            return req_grad_use.value();
          }
        }
      }
    }
  }

  // 如果没有找到符合条件的 profile 节点，返回空的 optional<bool>
  return c10::nullopt;
} // anonymous namespace

// 创建自动微分子图。根据给定的阈值，对图进行子图切片，收集自动微分节点。
std::vector<Node*> CreateAutodiffSubgraphs(
    const std::shared_ptr<Graph>& graph,
    size_t threshold) {
  // 创建别名数据库，用于处理别名和依赖关系。
  AliasDb db(graph);
  // 调试输出：在创建自动微分子图之前，打印当前图的信息。
  GRAPH_DEBUG("Before creating autodiff subgraphs", *graph);
  // 运行子图切片器，创建自动微分子图。
  SubgraphSlicer(graph->block(), graph, threshold, db, diff_nodes).run();
  // 调试输出：在创建自动微分子图之后，打印当前图的信息。
  GRAPH_DEBUG("After creating autodiff subgraphs", *graph);
  // 在输出节点上添加 requires_grad 属性。
  AddRequiresGradOnOutputNodes(graph);
  // 调试输出：打印 diff_nodes 的大小。
  GRAPH_DEBUG("diff_nodes.size() ", diff_nodes.size());
  // 返回自动微分节点的向量。
  return diff_nodes;
}

// 命名空间：jit
namespace jit {
// 命名空间：torch
namespace torch {
```