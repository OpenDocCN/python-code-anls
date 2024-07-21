# `.\pytorch\torch\csrc\jit\passes\remove_mutation.cpp`

```py
// 包含 Torch 中移除变异的相关头文件
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>

// 定义 torch 命名空间和 jit 命名空间
namespace torch {
namespace jit {

// 移除列表变异的方法，返回是否成功
bool MutationRemover::removeListMutation() {
  // 调用 RemoveListMutation 函数处理当前图的块，并返回结果
  return RemoveListMutation(graph_->block());
}

// 移除张量变异的方法，返回是否成功
bool MutationRemover::removeTensorMutation() {
  // 调用 RemoveTensorMutation 函数处理当前图的块，并返回结果
  return RemoveTensorMutation(graph_->block());
}

// 检查值是否具有副作用或别名的方法
bool MutationRemover::hasSideEffectOrAlias(Value* v, AliasDb* aliasDb) {
  // 获取值关联的节点
  Node* n = v->node();
  
  // 检查节点是否包含未处理的子块、子图或具有副作用，或者是图输入节点
  bool unhandled_node = !n->blocks().empty() ||
      n->hasAttribute(attr::Subgraph) || n->hasSideEffects() ||
      (v->node()->kind() == prim::Param);

  // 如果节点不可能与其输入发生别名，说明其是唯一的
  bool mayAliasInputs = (v->node()->kind() != prim::ListConstruct) &&
      aliasDb->mayContainAlias(v->node()->inputs(), v);
  
  // 返回是否有未处理的节点、可能有别名输入或者是图输入节点的任意情况
  return unhandled_node || mayAliasInputs || (v->node()->kind() == prim::Param);
}

// 创建特殊映射操作节点的方法
Node* MutationRemover::createSpecialMappedOp(Node* n) {
  // 在节点 n 的插入点上创建新节点的保护区域
  WithInsertPoint guard(n);
  
  // 获取节点的输入
  auto inputs = n->inputs();
  
  // 声明新节点的指针
  Node* new_node;

  // 根据节点 n 的不同类型执行不同的操作
  if (n->matches(
          "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")) {
    // 如果节点匹配 fill_ 操作，插入 dtype 节点，并创建新的 full_like 节点
    auto dtype = graph_->insert(prim::dtype, {inputs.at(0)});
    new_node = graph_
                   ->insert(
                       aten::full_like,
                       {inputs.at(0), inputs.at(1)},
                       {NamedValue("dtype", dtype)})
                   ->node();
    new_node->copyMetadata(n);
    new_node->output()->setType(n->output()->type());
  } else if (n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)")) {
    // 如果节点匹配 zero_ 操作，插入 zeros_like 节点
    new_node = graph_->insert(aten::zeros_like, {n->inputs().at(0)})->node();
  } else if (
      n->matches(
          "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")) {
    // 如果节点匹配 normal_ 操作，插入多个参数节点，并创建新的 normal 节点
    auto size = graph_->insert(aten::size, {n->inputs().at(0)});
    auto dtype = graph_->insert(prim::dtype, {n->inputs().at(0)});
    auto layout = graph_->insert(prim::layout, {n->inputs().at(0)});
    auto device = graph_->insert(prim::device, {n->inputs().at(0)});
    auto pin_memory = graph_->insert(aten::is_pinned, {n->inputs().at(0)});
    auto generator = graph_->insertConstant(IValue());
    new_node = graph_->insertNode(graph_->create(
        aten::normal,
        {n->inputs().at(1),
         n->inputs().at(2),
         size,
         generator,
         dtype,
         layout,
         device,
         pin_memory}));
  } else {
    // 如果没有匹配的操作，抛出内部断言错误
    TORCH_INTERNAL_ASSERT(false);
  }
  
  // 复制原节点的元数据到新节点
  new_node->copyMetadata(n);
  
  // 设置新节点的输出类型与原节点一致
  new_node->output()->setType(n->output()->type());
  
  // 返回创建的新节点
  return new_node;
}

// 结束 jit 命名空间
} // namespace jit
} // namespace torch
static bool removableSetItem(Node* n) {
  // 检查节点类型是否为 _set_item，且第二个输入是常量
  if (n->kind() != aten::_set_item ||
      n->input(1)->node()->kind() != prim::Constant) {
    return false;
  }
  // 检查第一个输入节点是否为 ListConstruct
  if (n->inputs().at(0)->node()->kind() != prim::ListConstruct) {
    return false;
  }
  auto li_node = n->inputs().at(0)->node();
  int64_t index = *constant_as<int64_t>(n->input(1));
  // 处理负索引，转换为正索引
  if (index < 0) {
    index += li_node->inputs().size();
  }
  auto li_len = static_cast<int64_t>(li_node->inputs().size());
  // 检查索引是否在列表长度范围内
  return index < li_len && index >= 0;
}

bool MutationRemover::listMutationFollowingListConstruct(Node* n) {
  // 检查节点类型是否为 append、insert 或 removableSetItem 函数返回 true，
  // 且第一个输入节点是 ListConstruct
  return (
      (n->kind() == aten::append ||
       (n->kind() == aten::insert &&
        n->inputs().at(1)->node()->kind() == prim::Constant) ||
       (removableSetItem(n))) &&
      n->inputs().at(0)->node()->kind() == prim::ListConstruct);
}

bool MutationRemover::tryMakeCreationAndMutationAtomic(
    Value* mutated_value,
    Node* mutating_op) {
  // 只有当被修改的值在图中没有副作用或别名时，才能移除其变异操作
  if (hasSideEffectOrAlias(mutated_value, getOrCreateAliasDb())) {
    return false;
  }

  // 确保创建张量和其后续变异是一个原子操作
  return getOrCreateAliasDb()->moveBeforeTopologicallyValid(
      mutated_value->node(), mutating_op);
}

bool MutationRemover::tryMakeUnaliasedIfOutputAndMutationAtomic(
    Value* mutated_value,
    Node* mutating_op) {
  // 只有当被修改的值的节点类型是 prim::If 时才考虑
  if (mutated_value->node()->kind() != prim::If) {
    return false;
  }

  auto if_node = mutated_value->node();
  auto offset = mutated_value->offset();
  auto true_value = if_node->blocks().at(0)->outputs().at(offset);
  auto false_value = if_node->blocks().at(1)->outputs().at(offset);

  // 检查 true_value 和 false_value 是否有多于一个使用，并且它们在图中没有副作用或别名
  if (true_value->uses().size() > 1 || false_value->uses().size() > 1) {
    return false;
  }

  if (hasSideEffectOrAlias(true_value, getOrCreateAliasDb()) ||
      hasSideEffectOrAlias(false_value, getOrCreateAliasDb())) {
    return false;
  }

  // 将 if 节点和变异操作节点移动到拓扑排序中的合适位置
  return getOrCreateAliasDb()->moveBeforeTopologicallyValid(
      if_node, mutating_op);
}

bool MutationRemover::RemoveListMutation(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    for (Block* sub_block : node->blocks()) {
      changed |= RemoveListMutation(sub_block);
    }

    // 检查当前节点是否符合移除列表变异的条件
    if (!listMutationFollowingListConstruct(node)) {
      continue;
    }

    Value* mutated_value = node->inputs().at(0);
    // 尝试将创建和变异操作合并为一个原子操作
    if (!tryMakeCreationAndMutationAtomic(mutated_value, node)) {
      continue;
    }

    changed = true;
    // 重写类似以下操作：
    // x = {v0}
    // x.append(v1)（或者 x.insert(0, v1)）
    // 为：
    // x = {v0, v1}（或者 x = {v1, v0}）
    // 可以从写入别名数据库列表中移除 x.append。
    // 所有其他别名属性仍然有效。
    Node* list_construct = mutated_value->node();
    // 根据节点类型进行不同的操作
    switch (node->kind()) {
      case aten::append:
        // 将节点的第二个输入添加到列表构造中
        list_construct->addInput(node->inputs().at(1));
        break;
      case aten::insert: {
        // 获取插入位置和列表长度
        int pos = toIValue(node->inputs().at(1))->toInt();
        int size = list_construct->inputs().size();
        // 如果插入位置为负数，则转换为标准位置
        if (pos < 0) {
          pos = std::max(pos + size, 0);
        }
        // 如果插入位置超出当前列表长度，则等同于追加
        pos = std::min(pos, size);
        // 在指定位置插入节点的第三个输入
        list_construct->insertInput(pos, node->inputs().at(2));
        break;
      }
      case aten::_set_item: {
        // 获取设置位置和列表长度
        int pos = toIValue(node->inputs().at(1))->toInt();
        int size = list_construct->inputs().size();
        // 如果设置位置为负数，则转换为标准位置
        if (pos < 0) {
          pos = std::max(pos + size, 0);
        }
        // 替换指定位置的输入为节点的第三个输入
        list_construct->replaceInput(pos, node->input(2));
        break;
      }
      default:
        // 抛出内部断言错误
        TORCH_INTERNAL_ASSERT(false);
    }

    // 处理节点输出的使用链和别名
    bool has_output = (!node->outputs().empty());
    if (has_output) {
      // 替换节点输出的所有使用为变异值
      node->output()->replaceAllUsesWith(mutated_value);
      // 从别名数据库的写入索引中删除节点
      getOrCreateAliasDb()->writeIndex_->erase(node);
    }

    // 销毁节点
    node->destroy();

    // TODO: 不严格需要重置写入缓存，根据模型评估
    // 重建写入位置索引
    getOrCreateAliasDb()->buildWrittenToLocationsIndex();
  }

  // 返回是否发生了改变
  return changed;
}



// 移除给定块中的张量变异操作
bool MutationRemover::RemoveTensorMutation(Block* block) {
  // 记录是否有变化
  bool changed = false;
  // 遍历该块中的每个节点
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    // 递归处理子块中的节点，记录是否有变化
    for (Block* sub_block : node->blocks()) {
      changed |= RemoveTensorMutation(sub_block);
    }

    // 如果设置了变异过滤器
    if (mutation_filter_) {
      const auto& mutation_filter = *mutation_filter_;
      // 如果节点不符合变异过滤器的条件，则跳过处理
      if (!mutation_filter(node)) {
        continue;
      }
    }

    // TODO: out op variants
    // 如果节点不是就地操作的变体，则跳过处理
    if (!inplaceOpVariant(node)) {
      continue;
    }

    // 获取被变异的值
    Value* mutated_value = node->inputs().at(0);
    // 尝试使创建和变异操作是原子的，如果失败则跳过
    if (!tryMakeCreationAndMutationAtomic(mutated_value, node) &&
        !tryMakeUnaliasedIfOutputAndMutationAtomic(mutated_value, node)) {
      continue;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* new_node;
    // 如果是特殊映射的操作，则创建特殊映射的节点
    if (isSpecialMappedOp(node)) {
      new_node = createSpecialMappedOp(node);
    } else {
      // 否则创建相同符号的新节点
      auto schema_name = node->schema().name();
      auto new_schema = schema_name.substr(0, schema_name.size() - 1);
      new_node = graph_->create(Symbol::fromQualString(new_schema), 1);
      new_node->copyMetadata(node);
      new_node->insertBefore(node);
      for (Value* input : node->inputs()) {
        new_node->addInput(input);
      }
      new_node->output()->setType(node->output()->type());

      // 如果新节点没有操作符，则销毁节点并跳过处理
      if (!new_node->maybeOperator()) {
        new_node->destroy();
        continue;
      }
    }

    // 标记发生了变化
    changed = true;
    // 替换所有使用被变异值的节点为新节点的输出
    mutated_value->replaceAllUsesAfterNodeWith(node, new_node->output());
    node->output()->replaceAllUsesWith(new_node->output());

    // 重写类似以下的操作序列：
    // x = torch.zeros()
    // x.add_(1)
    // x.add_(2)
    // 为：
    // x = torch.zeros()
    // x0 = x.add(1)
    // x0.add_(2)
    // 确保 x0 与原始 x 具有相同的别名关系
    // 替换 x 的内存 DAG 元素为 x0，以避免重建整个别名数据库
    getOrCreateAliasDb()->replaceWithNewValue(
        mutated_value, new_node->output());

    // 确保可变类型都有一个内存 DAG 元素，因此重新为 x 创建别名数据库元素
    getOrCreateAliasDb()->createValue(mutated_value);

    // 从别名数据库的写入列表中擦除被销毁节点
    getOrCreateAliasDb()->writeIndex_->erase(node);
    // 销毁节点
    node->destroy();

    // 移除变异操作后，写入缓存变得陈旧
    // TODO: 不一定需要重置写入缓存，需要在模型上进行评估
    getOrCreateAliasDb()->buildWrittenToLocationsIndex();
  }

  // 返回是否有任何变化
  return changed;
}

// 判断节点是否是就地操作的变体
bool MutationRemover::inplaceOpVariant(Node* n) {
  // 如果节点不是 ATen 操作，返回 false
  if (!n->kind().is_aten()) {
    return false;
  }

  // 如果节点是特殊映射的操作，则返回 true
  if (isSpecialMappedOp(n)) {
  // 返回 true，表示可以原地操作
  return true;
}

// 获取节点的操作符名称
auto name = n->schema().name();
// 检查是否是原地操作（操作符名称以'_'结尾）
bool inplace_op = name.at(name.size() - 1) == '_';
if (!inplace_op) {
  // 如果不是原地操作，返回 false
  return false;
}

// 需要根据模式进行别名分析
auto op = n->maybeOperator();
if (!op) {
  // 如果无法获取操作符，返回 false
  return false;
}
if (op->aliasAnalysisKind() != AliasAnalysisKind::FROM_SCHEMA) {
  // 如果操作符的别名分析类型不符合要求，返回 false
  return false;
}

// 所有的原地操作在当前实现中都只有一个被修改和返回的输入
// 检查是否符合这一条件，否则可能有奇怪的语义
if (n->outputs().size() != 1 || n->inputs().empty()) {
  // 如果输出不止一个或者输入为空，返回 false
  return false;
}
auto inputs = n->inputs();
// 检查第一个输入是否被写入别名数据库
if (!getOrCreateAliasDb()->writesToAlias(n, {inputs.at(0)}) ||
    // 检查其他输入是否被写入别名数据库
    getOrCreateAliasDb()->writesToAlias(
        n, {inputs.slice(1).begin(), inputs.slice(1).end()})) {
  // 如果有输入未被正确写入别名数据库，返回 false
  return false;
}

// 去除操作符名称末尾的'_'，得到新的模式名称
auto new_schema = name.substr(0, name.size() - 1);
// 检查是否存在使用新模式名称的操作符
return !getAllOperatorsFor(Symbol::fromQualString(new_schema)).empty();
}

// 从图中移除列表变异操作
bool RemoveListMutation(const std::shared_ptr<Graph>& graph) {
  // 创建 MutationRemover 对象，并传入图对象
  MutationRemover mr(graph);
  // 调用 MutationRemover 对象的 removeListMutation 方法，返回结果
  return mr.removeListMutation();
}

// 从图中移除张量变异操作
bool RemoveTensorMutation(
    const std::shared_ptr<Graph>& graph,
    std::optional<std::function<bool(Node*)>> mutation_filter) {
  // 创建 MutationRemover 对象，并传入图对象和变异过滤器
  MutationRemover mr(graph, std::move(mutation_filter));
  // 调用 MutationRemover 对象的 removeTensorMutation 方法，返回结果
  return mr.removeTensorMutation();
}

// 静态常量 unordered_set，包含激活操作符的符号集合
static const std::unordered_set<Symbol> activation_ops = []() {
  std::unordered_set<Symbol> target_ops;
  // 遍历激活类型提升映射
  for (const auto& iter : activation_type_promotion_mapping) {
    // 构造操作符名称，并插入到目标操作符集合中
    std::string name = std::string(iter.first.toQualString()) + "_";
    target_ops.insert(Symbol::fromQualString(name));
  }
  // 返回填充完成的目标操作符集合
  return target_ops;
}();

// 将原地操作转换为函数式激活操作
bool InplaceToFunctionalActivation(const std::shared_ptr<Graph>& graph) {
  // 调用 RemoveTensorMutation 函数，传入图对象和激活操作符过滤器
  return RemoveTensorMutation(graph, [](Node* node) {
    // 检查节点的操作符是否在激活操作符集合中
    return activation_ops.count(node->kind()) != 0;
  });
}

// 命名空间声明结束
} // namespace jit
} // namespace torch
```