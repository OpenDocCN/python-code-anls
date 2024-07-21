# `.\pytorch\torch\csrc\jit\passes\utils\subgraph_utils.cpp`

```
// 包含头文件 torch/csrc/jit/passes/utils/subgraph_utils.h，用于子图工具函数
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 包含头文件 torch/csrc/jit/passes/canonicalize.h，用于规范化函数
#include <torch/csrc/jit/passes/canonicalize.h>

// 包含头文件 ATen/core/symbol.h，用于符号定义
#include <ATen/core/symbol.h>

// 包含头文件 c10/util/StringUtil.h，用于字符串工具函数
#include <c10/util/StringUtil.h>

// 包含头文件 c10/util/irange.h，用于范围迭代器
#include <c10/util/irange.h>

// 包含头文件 torch/csrc/jit/jit_log.h，用于 JIT 日志
#include <torch/csrc/jit/jit_log.h>

// 包含头文件 utility，用于通用工具函数
#include <utility>

// 命名空间 torch::jit::SubgraphUtils 内部的匿名命名空间开始
namespace torch {
namespace jit {
namespace SubgraphUtils {
namespace {

// 检查节点是否具有子图属性的函数
bool hasSubgraph(Node* n) {
  return n->hasAttribute(attr::Subgraph);
}

// 收集值的最后使用的函数，返回值是包含可选 Use 对象的向量
std::vector<std::optional<const Use>> gatherLastUses(
    at::ArrayRef<Value*> values) {
  // 使用 fmap 函数对 values 进行映射，返回每个值的最后使用的可选 Use 对象
  return fmap(values, [&](Value* v) -> std::optional<const Use> {
    // 调用 firstOrLastUse 函数查找值的最后使用，find_first 参数设为 false
    return firstOrLastUse(v, /*find_first*/ false);
  });
}

// ValueMapper 结构体，用于在节点合并到子图时保持节点输出的别名特性
struct ValueMapper {
  // 构造函数，初始化 ValueMapper 对象
  // to_merge 是要合并到子图的节点，db 是别名数据库，existing_subgraph 是存在的子图节点（可选）
  ValueMapper(
      Node* to_merge,
      AliasDb& db,
      std::optional<Node*> existing_subgraph) {
    // 收集要合并节点的输出的最后使用
    last_uses_ = gatherLastUses(to_merge->outputs());
    // 如果存在目标子图节点，则收集其输出的最后使用
    if (existing_subgraph) {
      existing_last_uses_ = gatherLastUses((*existing_subgraph)->outputs());
    }
    // 设置插入点为 to_merge 所在图的最后一个节点之前
    WithInsertPoint guard(to_merge);
    auto g = to_merge->owningGraph();
    // 在图中插入一个未初始化的节点，用于临时存储节点合并前的别名属性
    placeholder_node_ = g->insertNode(g->create(prim::Uninitialized, 0));
    // 遍历要合并节点的每个输出
    for (size_t i = 0; i < to_merge->outputs().size(); ++i) {
      Value* existing = to_merge->outputs().at(i);
      // 插入一个新的输出到占位节点中，并复制原输出的元数据
      Value* new_value = placeholder_node_->insertOutput(i)->copyMetadata(
          to_merge->outputs().at(i));
      // 使用别名数据库 db 替换原输出 existing 为新输出 new_value
      db.replaceWithNewValue(existing, new_value);
    }
  }

  // 检查两个 Use 对象是否相等的函数
  bool usesEqual(const Use& a, const Use& b) {
    return a.user == b.user && a.offset == b.offset;
  }

  // 复制节点的别名属性到新节点的函数
  void copyAliasing(Node* merged_node, AliasDb& db) {
    // 获取合并节点的新输出
    auto new_outputs = merged_node->outputs();
    // 遍历新输出向量中的每个值 v
    for (Value* v : new_outputs) {
      // 查找 v 的最后一个使用位置
      auto maybe_last_use = firstOrLastUse(v, /*find_first*/ false);
      // 如果找不到使用位置，则断言失败，表示不应将其添加为输出
      TORCH_INTERNAL_ASSERT(maybe_last_use);
      const Use last_use = *maybe_last_use;

      // 检查最后一个使用位置是否属于已存在的子图输出，如果是，则跳过
      bool is_existing_value = false;
      for (size_t i = 0; i < existing_last_uses_.size() && !is_existing_value;
           ++i) {
        // 判断是否存在且使用位置相同
        is_existing_value = existing_last_uses_[i].has_value() &&
            usesEqual(*existing_last_uses_[i], last_use);
      }
      if (is_existing_value) {
        continue;
      }

      // 在 last_uses_ 中查找与 last_use 相同的使用位置
      size_t i = 0;
      while (i < last_uses_.size() && last_uses_.at(i).has_value() &&
             !usesEqual(*last_uses_.at(i), last_use)) {
        ++i;
      }
      // 断言找到了对应的位置
      TORCH_INTERNAL_ASSERT(i != last_uses_.size());
      // 使用 db 替换占位节点的输出值为当前值 v
      db.replaceWithNewValue(placeholder_node_->outputs().at(i), v);
    }
    // 销毁占位节点
    placeholder_node_->destroy();
  }

  // 存储每个值的最后一个使用位置的可选值数组
  std::vector<std::optional<const Use>> last_uses_;
  // 存储已存在的子图输出的最后一个使用位置的可选值数组
  std::vector<std::optional<const Use>> existing_last_uses_;
  // 指向占位节点的指针
  Node* placeholder_node_;
};

// 执行子图合并并更新别名信息
Node* executeSubgraphMergeAndUpdateAliasing(
    Node* to_merge,  // 要合并的节点
    std::optional<Node*> existing,  // 可选的已存在节点
    AliasDb& db,  // 别名数据库的引用
    const std::function<Node*(void)>& merge_fn) {  // 执行合并的函数回调
  // 当将节点合并到子图时，新子图的输出将具有与原始节点输出相同的别名属性。
  // 在这里，我们创建一个占位节点，将别名属性转移到占位节点，执行合并操作，
  // 并将别名属性转移到相应的融合组输出。
  ValueMapper vm(to_merge, db, existing);  // 创建值映射器对象
  Node* fusion_group = merge_fn();  // 执行合并操作，获取融合组节点
  vm.copyAliasing(fusion_group, db);  // 将别名属性从原始节点转移到融合组节点
  return fusion_group;  // 返回融合组节点
}

// 合并两个子图中的节点。节点将合并到 `mergeTo` 中，而 `mergeFrom` 将被销毁。
void mergeSubgraph(Node* mergeTo, Node* mergeFrom) {
  bool merge_from_is_after = mergeFrom->isAfter(mergeTo);  // 检查是否在 `mergeTo` 之后
  Node* nodeBeforeMergeFrom = mergeFrom->prev();  // 获取 `mergeFrom` 的前一个节点
  Node* nodeAfterMergeFrom = mergeFrom->next();  // 获取 `mergeFrom` 的后一个节点

  unmergeSubgraph(mergeFrom);  // 解除 `mergeFrom` 的子图合并状态

  graph_node_list_iterator end_it;  // 结束迭代器
  graph_node_list_iterator it;  // 迭代器

  if (merge_from_is_after) {
    it = nodeBeforeMergeFrom->iterator();  // 从 `nodeBeforeMergeFrom` 的迭代器开始
    end_it = nodeAfterMergeFrom->iterator();  // 到 `nodeAfterMergeFrom` 的迭代器结束
  } else {
    end_it = nodeBeforeMergeFrom->reverseIterator();  // 从 `nodeBeforeMergeFrom` 的反向迭代器开始
    it = nodeAfterMergeFrom->reverseIterator();  // 到 `nodeAfterMergeFrom` 的反向迭代器结束
  }
  ++it;  // 向前移动迭代器

  std::vector<Node*> merged_nodes;  // 存储已合并的节点
  while (it != end_it) {  // 迭代节点列表
    Node* node = *it;
    ++it;
    mergeNodeIntoSubgraph(node, mergeTo);  // 将节点合并到 `mergeTo` 中
  }
}

struct topo_cmp_value {
  bool operator()(Value* a, Value* b) const {
    if (a->node() == b->node()) {  // 如果节点相同
      return a->unique() < b->unique();  // 按照唯一标识排序
    }
    return a->node()->isBefore(b->node());  // 按照节点的先后顺序排序
  }
};

struct topo_cmp_node {
  bool operator()(Node* a, Node* b) const {
    return a->isBefore(b);  // 按照节点的先后顺序排序
  }
};

// 收集需要解除融合的节点
void collectNodesToUnfuse(Node* start, std::set<Node*, topo_cmp_node>& s) {
  if (start->kind() == prim::Return || start->kind() == prim::Param) {
    GRAPH_DEBUG("reached the param or return node", getHeader(start));  // 调试信息，达到参数或返回节点
    return;
  }

  if (s.count(start) != 0) {
    // 已访问过，无需再访问子节点
    return;
  }

  GRAPH_DEBUG("collectNodesToUnfuse: inserting node ", getHeader(start));  // 调试信息，插入节点
  s.insert(start);  // 将节点插入集合

  for (auto o : start->outputs()) {
    for (auto use : o->uses()) {
      collectNodesToUnfuse(use.user, s);  // 递归收集使用节点
    }
  }
}

// 构建带有别名集的向量
std::vector<std::set<Value*, topo_cmp_value>> buildAliasedSets(
    std::shared_ptr<Graph> subgraph) {
  auto outputs = subgraph->outputs();  // 获取子图的输出节点
  AliasDb alias_db(std::move(subgraph));  // 创建别名数据库
  TORCH_INTERNAL_ASSERT(outputs.size() > 1);  // 断言，确保输出节点数量大于1
  std::vector<std::set<Value*, topo_cmp_value>> res;  // 存储结果的向量
  for (auto o : outputs) {  // 遍历输出节点
    auto grouped = false;  // 是否已分组标志
    // 对结果集 res 中的每个集合 s 进行迭代
    for (auto& s : res) {
      // 获取集合 s 中的第一个元素 os
      auto os = *s.begin();
      // 查询 alias_db 看是否可能包含 os 的别名 o
      auto aliased = alias_db.mayContainAlias(os, o);
      // 输出调试信息，比较 o 和 os 的调试名称及结果 aliased
      GRAPH_DEBUG(
          "comparing %",
          o->debugName(),
          " with %",
          os->debugName(),
          " result ",
          aliased);
      // 如果 o 和 os 有别名关系
      if (aliased) {
        // 将 o 插入集合 s 中
        s.insert(o);
        // 输出调试信息，将 o 和 os 分组
        GRAPH_DEBUG("Grouping %", o->debugName(), " with %", os->debugName());
        // 标记已分组为真
        grouped = true;
      }
    }
    // 如果没有找到任何已有分组，则将 o 作为一个新组加入结果集 res
    if (!grouped) {
      res.push_back({o});
    }
  }
  // 返回更新后的结果集 res
  return res;
} // 结束当前函数

} // 结束命名空间

std::shared_ptr<Graph> getSubgraph(Node* n) {
  return n->g(attr::Subgraph); // 返回节点 `n` 的子图
}

void unmergeSubgraph(Node* subgraphNode) {
  // 内联子图，替换节点输出的使用，并销毁节点
  auto outerGraph = subgraphNode->owningGraph(); // 获取子图节点所在的外部图
  WithInsertPoint guard(subgraphNode); // 设置插入点为子图节点
  const auto subgraphOutputs = insertGraph(
      *outerGraph, *getSubgraph(subgraphNode), subgraphNode->inputs()); // 将子图插入外部图
  AT_ASSERT(subgraphOutputs.size() >= subgraphNode->outputs().size()); // 断言确保子图输出的数量不少于子图节点的输出数量
  for (size_t i = 0; i < subgraphNode->outputs().size(); ++i) {
    subgraphNode->outputs()[i]->replaceAllUsesWith(subgraphOutputs[i]); // 替换所有使用子图节点输出的值
  }
  subgraphNode->destroy(); // 销毁子图节点
}

static void collectNestedUses(
    std::unordered_set<Value*>& closed_over_values,
    std::unordered_set<Value*>& new_values,
    std::unordered_map<Value*, Value*>& externalValuesMap,
    Node* input_node) {
  for (auto input : input_node->inputs()) {
    if (externalValuesMap.count(input) == 0 && new_values.count(input) == 0) {
      closed_over_values.insert(input); // 将未记录过的输入节点插入闭包值集合
    }
  }
  if (input_node->kind() == prim::If) {
    for (Block* block : input_node->blocks()) {
      for (Node* node : block->nodes()) {
        collectNestedUses(
            closed_over_values, new_values, externalValuesMap, node); // 递归收集条件语句块中的使用
      }
      for (Value* v : block->outputs()) {
        if (externalValuesMap.count(v) == 0 && new_values.count(v) == 0) {
          closed_over_values.insert(v); // 将未记录过的条件语句块输出插入闭包值集合
        }
      }
    }
  } else if (input_node->kind() == prim::Loop) {
    for (Value* v : input_node->inputs()) {
      if (externalValuesMap.count(v) == 0 && new_values.count(v) == 0) {
        closed_over_values.insert(v); // 将未记录过的循环节点输入插入闭包值集合
      }
    }
    Block* block = input_node->blocks().at(0);
    for (Value* v : block->inputs()) {
      new_values.insert(v); // 将循环块的输入添加到新值集合
    }
    for (Node* node : block->nodes()) {
      collectNestedUses(
          closed_over_values, new_values, externalValuesMap, node); // 递归收集循环块中的使用
    }
  } else if (!input_node->blocks().empty()) {
    TORCH_INTERNAL_ASSERT(false, input_node, " kind not handled yet"); // 报告未处理的节点类型错误
  }
  for (Value* output : input_node->outputs()) {
    new_values.insert(output); // 将节点的输出添加到新值集合
  }
}

static std::unordered_set<Value*> closedOverValues(
    Node* toMerge,
    std::unordered_map<Value*, Value*>& externalValuesMap) {
  std::unordered_set<Value*> closed_over_values;
  std::unordered_set<Value*> new_values;
  collectNestedUses(closed_over_values, new_values, externalValuesMap, toMerge); // 收集节点 `toMerge` 的闭包值
  return closed_over_values; // 返回闭包值集合
}

void mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    bool destroyNode) {
  AT_ASSERT(hasSubgraph(subgraphNode) && toMerge != subgraphNode); // 断言确保子图节点包含子图，并且要合并的节点不是子图节点本身
  if (hasSubgraph(toMerge)) {
  // 返回合并子图的结果节点
  return mergeSubgraph(subgraphNode, toMerge);
}

// 获取子图
auto subgraph = getSubgraph(subgraphNode);

// 从周围图中的值到子图中输入/输出的映射
std::unordered_map<Value*, Value*> externalValuesMap;

// 断言：子图节点的输入数量与子图的输入数量相同
AT_ASSERT(subgraphNode->inputs().size() == subgraph->inputs().size());
size_t idx = 0;
// 建立输入值的映射关系
for (auto input : subgraphNode->inputs()) {
  externalValuesMap[input] = subgraph->inputs()[idx];
  idx++;
}

// 建立输出值的映射关系
for (size_t i = 0; i < subgraphNode->outputs().size(); ++i) {
  externalValuesMap[subgraphNode->outputs().at(i)] =
      subgraph->outputs().at(i);
}

// 如果还没有将 n 的输入添加到组的输入列表中，则添加
bool merging_node_after_subgraph = toMerge->isAfter(subgraphNode);
Node* guard_node = merging_node_after_subgraph ? *subgraph->nodes().end()
                                               : *subgraph->nodes().begin();
WithInsertPoint guard(guard_node);

// 获取受闭包值影响的一组值
std::unordered_set<Value*> closedValues =
    closedOverValues(toMerge, externalValuesMap);

// 目前有下游使用依赖于图输入的固定顺序，TODO: 移除
std::vector<Value*> orderedClosedValues;
std::unordered_set<Value*> orderedSeenValues;
// 遍历 toMerge 的输入值
for (Value* input : toMerge->inputs()) {
  orderedClosedValues.push_back(input);
  orderedSeenValues.insert(input);
}
// 将闭包值添加到有序闭包值列表中
for (Value* closedValue : closedValues) {
  if (!orderedSeenValues.count(closedValue)) {
    orderedClosedValues.push_back(closedValue);
    orderedSeenValues.insert(closedValue);
  }
}

// 对于有序闭包值列表中的每个输入值
for (auto input : orderedClosedValues) {
  if (externalValuesMap.count(input) == 0) {
    // 如果是常量值，克隆到子图中，而不是直接引用，以便进行更多的优化
    if (auto value = toIValue(input)) {
      auto nv = subgraph->insertConstant(*value);
      nv->copyMetadata(input);
      externalValuesMap[input] = nv;
    } else {
      // 通常情况下，这是一个常规输入，因此将其注册到组节点和内部子图中
      subgraphNode->addInput(input);
      auto inputToGraph = subgraph->addInput();
      inputToGraph->copyMetadata(input);
      externalValuesMap[input] = inputToGraph;
    }
  }
}

// 将节点合并到子图中
auto mergedNode = subgraph->insertNode(subgraph->createClone(
    toMerge, [&](Value* v) { return externalValuesMap[v]; }));

// 如果不是在子图节点之后合并节点
if (!merging_node_after_subgraph) {
  // 如果 n 的输出是 `group` 的输入，则移除它们，因为我们刚刚合并了 n
  //
  // 即，
  // x = f(w); group(x, y, z) 变成 group(w, y, z).
  // x, y, z = f(w); group(x, y, z) 变成 group(w).
  auto inputs = subgraphNode->inputs();
  // 遍历要合并节点的输出
  for (size_t i = 0; i < toMerge->outputs().size(); ++i) {
    // 查找当前输出是否在输入列表中
    auto it = std::find(inputs.begin(), inputs.end(), toMerge->outputs()[i]);
    // 如果找到了当前输出在输入列表中
    if (it != inputs.end()) {
      // 计算当前输出在输入列表中的位置
      size_t p = it - inputs.begin();
      // 从子图节点中移除对应位置的输入
      subgraphNode->removeInput(p);
      // 替换子图中相应位置的输入节点为合并节点的输出
      subgraph->inputs()[p]->replaceAllUsesWith(mergedNode->outputs()[i]);
      // 在子图中删除对应位置的输入
      subgraph->eraseInput(p);
    }
  }
}

// 将合并节点的输出添加到组节点和内部子图的输出中
for (const auto i : c10::irange(toMerge->outputs().size())) {
  // 获取原始输出和新输出
  auto oldOutput = toMerge->outputs()[i];
  auto newOutput = mergedNode->outputs()[i];
  // 在子图中注册新输出
  subgraph->registerOutput(newOutput);
  // 在组节点中添加一个输出，复制原始输出的元数据
  auto groupOutput = subgraphNode->addOutput();
  groupOutput->copyMetadata(oldOutput);
  // 替换原始输出的所有使用为组节点的输出
  oldOutput->replaceAllUsesWith(groupOutput);
}

// 如果需要销毁节点，则现在删除原始节点
if (destroyNode) {
  toMerge->destroy();
}

// 在销毁 `toMerge` 前等待修剪子图输出，
// 因为销毁 `toMerge` 可能导致子图输出不再被使用
const auto hasUsesOutsideSubgraph = [&](Value* v) {
  return std::any_of(
      v->uses().cbegin(), v->uses().cend(), [&](const Use& use) {
        return use.user->isAfter(subgraphNode);
      });
};

// 逆向遍历子图节点的输出
for (int64_t i = subgraphNode->outputs().size() - 1; i >= 0; i--) {
  // 如果子图输出在子图节点之后没有任何使用
  if (!hasUsesOutsideSubgraph(subgraphNode->outputs().at(i))) {
    // 从子图节点和子图中删除对应位置的输出
    subgraphNode->eraseOutput(i);
    subgraph->eraseOutput(i);
  }
}
}

// 创建一个单节点子图，将节点插入该子图中，并返回该子图节点
Node* createSingletonSubgraph(Node* n, Symbol subgraphKind) {
  // 获取节点所属的计算图
  auto graph = n->owningGraph();
  // 创建一个指定类型的子图
  auto subgraph = graph->create(subgraphKind, 0);
  // 设置子图的属性，关联到当前作用域的新图形对象
  subgraph->g_(attr::Subgraph, std::make_shared<Graph>(graph->current_scope()));
  // 将子图插入到节点之前
  subgraph->insertBefore(n);
  // 将节点合并到子图中
  mergeNodeIntoSubgraph(n, subgraph);
  // 返回创建的子图节点
  return subgraph;
}

// 将节点合并到子图中，并更新别名信息
void mergeNodeIntoSubgraphAndUpdateAliasing(
    Node* to_merge,
    Node* subgraphNode,
    AliasDb& db) {
  // 执行节点合并并更新别名信息
  executeSubgraphMergeAndUpdateAliasing(to_merge, subgraphNode, db, [&]() {
    // 合并节点到子图中
    mergeNodeIntoSubgraph(to_merge, subgraphNode);
    // 返回更新后的子图节点
    return subgraphNode;
  });
}

// 创建一个单节点子图并更新别名信息
Node* createSingletonSubgraphAndUpdateAliasing(
    Node* to_merge,
    Symbol subgraphKind,
    AliasDb& db) {
  // 执行子图合并并更新别名信息
  return executeSubgraphMergeAndUpdateAliasing(
      to_merge, c10::nullopt, db, [&]() {
        // 创建一个指定类型的单节点子图
        return createSingletonSubgraph(to_merge, subgraphKind);
      });
}

// 对子图节点执行输出解绑和输入关联解绑操作
bool unmergeOutputsAlisingInputs(Node* subgraphNode) {
  // 输出解绑和输入关联解绑操作的调试信息
  GRAPH_DEBUG("unfuseOutputsAlisingInputs on ", getHeader(subgraphNode));
  // 获取子图对象
  auto subgraph = subgraphNode->g(attr::Subgraph);
  // 创建别名数据库对象
  AliasDb alias_db(subgraph);

  // 用于存储需要解绑的节点集合
  std::set<Node*, topo_cmp_node> nodes;
  // 遍历子图的输出
  for (auto o : subgraph->outputs()) {
    // 如果输出可能与子图的输入存在别名关系
    if (alias_db.mayContainAlias(o, subgraph->inputs())) {
      // 收集需要解绑的节点
      collectNodesToUnfuse(o->node(), nodes);
    }
  }

  // 按照反向拓扑顺序执行解绑操作
  for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
    // 解绑节点
    SubgraphUtils::unmergeNode(*it, subgraphNode);
  }

  // 返回是否有节点被解绑
  return !nodes.empty();
}

// 解绑子图节点的别名输出
bool unmergeAliasedOutputs(Node* subgraphNode) {
  // 解绑别名输出的调试信息
  GRAPH_DEBUG("unfuseAliasedOutputs on ", getHeader(subgraphNode));
  // 如果子图的输出少于2个，直接返回false
  if (subgraphNode->outputs().size() < 2) {
    return false;
  }

  // 获取子图对象
  auto subgraph = subgraphNode->g(attr::Subgraph);
  // 调试输出子图信息
  GRAPH_DUMP("unfuseAliasedOutputs Subgraph ", subgraph);
  // 构建别名集合
  auto sets = buildAliasedSets(std::move(subgraph));
  // 调试输出别名集合的大小
  GRAPH_DEBUG("buildAliasedSets sets.size() = ", sets.size());

  // 用于存储需要解绑的节点集合
  std::set<Node*, topo_cmp_node> nodes;

  // 遍历别名集合
  for (auto i : c10::irange(sets.size())) {
    // 如果某个集合的大小小于等于1，则跳过
    if (sets[i].size() <= 1) {
      GRAPH_DEBUG(
          "Set ",
          i,
          " with leader ",
          (*(sets[i].begin()))->debugName(),
          " size = ",
          sets[i].size());
      continue;
    }

    // 对于具有至少两个别名输出的集合
    auto it = ++sets[i].begin();
    // 遍历集合中的节点，跳过拓扑顺序最早的节点
    while (it != sets[i].end()) {
      GRAPH_DEBUG(
          "root aliased value ", (*it)->debugName(), " node ", *(*it)->node());
      // 收集需要解绑的节点
      collectNodesToUnfuse((*it)->node(), nodes);
      it++;
    }
  }

  // 按照反向拓扑顺序执行解绑操作
  for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
    // 解绑节点
    unmergeNode(*it, subgraphNode);
  }

  // 返回是否有节点被解绑
  return !nodes.empty();
}
void unmergeNode(Node* n, Node* subgraphNode) {
  // 收集输出索引
  GRAPH_DEBUG("unfuseNode node ", getHeader(n));
  auto subgraph = n->owningGraph();

  std::set<Value*> node_outputs(n->outputs().begin(), n->outputs().end());
  std::set<size_t> output_indices;
  std::set<Value*> node_inputs(n->inputs().begin(), n->inputs().end());

  std::unordered_map<Value*, Value*> local_map;
  auto env = [&](Value* v) {
    auto it = local_map.find(v);
    if (it != local_map.end()) {
      return it->second;
    }
    TORCH_INTERNAL_ASSERT(
        false,
        "all inputs should've been mapped. Couldn't map %",
        v->debugName());
    return v;
  };

  for (auto i : c10::irange(subgraph->outputs().size())) {
    if (node_outputs.count(subgraph->outputs().at(i)) != 0) {
      output_indices.insert(i);
    }

    if (node_inputs.count(subgraph->outputs().at(i)) != 0) {
      GRAPH_DEBUG(
          "output %",
          subgraph->outputs().at(i)->debugName(),
          " is already subgraph's output");
      GRAPH_DEBUG(
          "Mapping %",
          subgraph->outputs().at(i)->debugName(),
          " to %",
          subgraphNode->outputs().at(i)->debugName());
      local_map[subgraph->outputs().at(i)] = subgraphNode->outputs().at(i);
      node_inputs.erase(subgraph->outputs().at(i));
    }
  }

  WithInsertPoint wip(subgraphNode->next());

  // 这些节点输入需要添加到子图的输出中
  // 将它们放入 vmap
  for (auto ni : node_inputs) {
    if (local_map.count(ni) != 0) {
      // 如果 `n` 使用常量节点的两个或更多输出，则可能会发生这种情况
      // 并且我们已经将常量克隆到外部图中并映射了其输出
      continue;
    }

    Value* sno = nullptr;
    if (ni->node()->kind() == prim::Constant) {
      auto copy = subgraphNode->owningGraph()->createClone(ni->node(), env);
      subgraphNode->owningGraph()->insertNode(copy);
      // 如果我们有一个多输出常量，映射其余的输出
      // 这样当我们克隆 `n` 时，`n` 的克隆将使用此常量克隆的输出
      for (auto i : c10::irange(n->outputs().size())) {
        GRAPH_DEBUG(
            "Mapping %",
            ni->node()->output(i)->debugName(),
            " to %",
            copy->output(i)->debugName());
        local_map[ni->node()->output(i)] = copy->output(i);
      }
    } else {
      subgraph->registerOutput(ni);
      sno = subgraphNode->addOutput();
      sno->setType(ni->type());
      GRAPH_DEBUG("Mapping %", ni->debugName(), " to %", sno->debugName());
      local_map[ni] = sno;
    }
  }

  auto copy = subgraphNode->owningGraph()->createClone(n, env);
  GRAPH_DEBUG("copy ", *copy);

  for (auto i : c10::irange(n->outputs().size())) {
    auto oo = n->outputs()[i];
    auto no = copy->outputs()[i];
    no->copyMetadata(oo);
    GRAPH_DEBUG("Mapping %", oo->debugName(), " to %", no->debugName());
    local_map[oo] = no;

# 将 `local_map` 中的键 `oo` 映射到值 `no`


  }

  subgraphNode->owningGraph()->insertNode(copy);

# 将 `copy` 节点插入到 `subgraphNode` 所属的图中


  for (auto it = output_indices.rbegin(); it != output_indices.rend(); it++) {
    auto replace_val = local_map[subgraph->outputs().at(*it)];
    subgraphNode->outputs().at(*it)->replaceAllUsesWith(replace_val);
    subgraphNode->eraseOutput(*it);
    subgraph->eraseOutput(*it);
  }

# 反向遍历 `output_indices`，替换子图 `subgraphNode` 的输出
# 使用 `local_map` 中的映射值替换原输出值，并从子图和子图节点的输出中移除对应索引的输出


  n->destroy();

# 销毁节点 `n`
}

// 定义静态函数：根据给定的最大长度截断字符串，并加入哈希后返回
static std::string truncateStrWithHash(const std::string& s, size_t maxlen) {
  // 如果字符串长度小于等于最大长度，直接返回原字符串
  if (s.size() <= maxlen) {
    return s;
  }
  // 计算字符串的哈希值，并转换为字符串形式
  std::string hash_str = std::to_string(c10::hash<std::string>{}(s));
  // 如果哈希值加上 '_' 可以放入到最大长度内，则相应地截断原始字符串
  // 使得包含哈希值的最终字符串长度不超过最大长度。如果不可能，则至少
  // 截断原始字符串到最大长度，并将哈希值追加到其后。
  size_t trunc_len =
      (maxlen > hash_str.size() + 1) ? (maxlen - hash_str.size() - 1) : maxlen;
  // 创建字符串流用于构建截断后的字符串
  std::stringstream truncated;
  // 截取原始字符串到指定长度
  truncated << s.substr(0, trunc_len);
  // 在截断后的字符串末尾加入 '_' 和哈希值
  truncated << "_" << hash_str;
  // 返回构建好的截断字符串
  return truncated.str();
}

// 生成用于图形命名的字符串，结合给定的前缀和图形节点信息
std::string generateNameForGraph(
    const std::shared_ptr<Graph>& graph,
    size_t maxlen,
    const std::string& prefix) {
  // 创建字符串流以构建最终的图形名称
  std::stringstream graph_name;
  // 将前缀添加到图形名称中
  graph_name << prefix;
  // 遍历图形中的每个节点
  for (Node* node : graph->nodes()) {
    // 如果节点类型不是 ATen 类型，则跳过
    if (!node->kind().is_aten()) {
      continue;
    }
    // 将节点的非限定字符串形式（Unqualified String）添加到图形名称中
    graph_name << "_" << node->kind().toUnqualString();
  }
  // 调用截断字符串并添加哈希的函数，返回最终的图形名称
  return truncateStrWithHash(graph_name.str(), maxlen);
}

// 结束命名空间声明
} // namespace SubgraphUtils
} // namespace jit
} // namespace torch
```