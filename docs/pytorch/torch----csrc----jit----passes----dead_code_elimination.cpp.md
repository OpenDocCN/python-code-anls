# `.\pytorch\torch\csrc\jit\passes\dead_code_elimination.cpp`

```
    // 如果节点已经标记过，返回 false，表示未标记新内容
    if (marked_.count(node)) {
      return false;
    }

    // 标记该节点为已标记
    marked_.insert(node);

    // 遍历节点的输出值
    for (const auto& output : node->outputs()) {
      // 如果输出值没有使用，则跳过
      if (!output->hasUses()) {
        continue;
      }

      // 标记所有使用该输出值的节点为活跃节点
      for (const auto& use : output->uses()) {
        mark(use.user);
      }
    }

    return true;
  }

  // 标记给定块中的所有活跃节点
  void mark(Block* block) {
    for (Node* node : block->nodes()) {
      // 调用 markReturnNode 标记块中的每个节点
      markReturnNode(node);
    }
  }

  // 清除不再需要的节点和值
  void sweep(Block* block, bool recurse) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      if (!marked_.count(node)) {
        it = block->eraseNode(it);
      } else {
        ++it;
      }
    }

    // 递归处理子块
    if (recurse) {
      for (Block* sb : block->blocks()) {
        sweep(sb, recurse);
      }
    }
  }

  // 标记节点的输出为活跃状态，并递归标记使用该输出的节点
  void mark(Node* node) {
    for (const auto& output : node->outputs()) {
      if (!output->hasUses()) {
        continue;
      }
      for (const auto& use : output->uses()) {
        mark(use.user);
      }
    }
    liveValues_.insert(node->outputs().begin(), node->outputs().end());
  }

  DCESideEffectPolicy sideEffectPolicy_;
  std::shared_ptr<Graph> graph_;
  std::function<void(const std::unordered_set<const Value*>&)> deleteCallback_;
  bool useAliasDb_;

  // 已标记为活跃的节点集合
  std::unordered_set<Node*> marked_;

  // 活跃的值集合
  std::unordered_set<const Value*> liveValues_;
};

} // namespace jit
} // namespace torch
    // 确保节点所在的块是返回节点自身
    AT_ASSERT(node->owningBlock()->return_node() == node);
    // 获取外部节点，即包含当前块的节点
    auto outerNode = node->owningBlock()->owningNode();
    // 如果外部节点为空或者外部节点是反向节点，则表示当前节点在图的顶层返回块，将节点标记为使用
    if (outerNode == nullptr || outerNode->kind() == prim::Reverse) {
      // 如果没有外部节点，我们在处理图的顶层返回块。我们认为所有图输出都是“使用的”，因此正常标记此节点。
      return mark(node);
    }

    // 收集实际存活的所有输入
    if (outerNode->kind() == prim::Loop ||
        outerNode->kind() == c10::onnx::Loop) {
      // 特殊处理循环中的依赖关系
      auto loop = LoopView(outerNode);
      for (const auto i : c10::irange(loop.carriedOutputs().size())) {
        if (outerNode->kind() == c10::onnx::Loop) {
          // 对于 ONNX 循环的特殊处理
          // 循环体中携带的输入和输出数量不同
          // 不能简单地通过相同的索引将它们映射到一起
          liveValues_.insert(loop.bodyCarriedOutputs().at(i));
          continue;
        }
        auto innerInput = loop.bodyCarriedInputs().at(i);
        auto innerOutput = loop.bodyCarriedOutputs().at(i);
        auto outerOutput = loop.carriedOutputs().at(i);
        // 如果外部输出已经存在于 liveValues_ 中，或者内部输入有使用，则将内部输出标记为存活
        if (liveValues_.count(outerOutput) || innerInput->hasUses()) {
          liveValues_.insert(innerOutput);
        }
      }

      // 同样标记循环的下一个条件为存活，因为它将在循环体内使用
      liveValues_.insert(loop.nextCond());
    } else {
      // 确保外部节点的输出数量与节点的输入数量相等
      AT_ASSERT(outerNode->outputs().size() == node->inputs().size());
      for (const auto i : c10::irange(outerNode->outputs().size())) {
        auto innerOutput = node->inputs()[i];
        auto outerOutput = outerNode->outputs()[i];
        // 如果外部输出已经存在于 liveValues_ 中，则将节点的相应输入标记为存活
        if (liveValues_.count(outerOutput)) {
          liveValues_.insert(innerOutput);
        }
      }
    }

    // 将当前节点标记为已处理
    marked_.insert(node);
    // 返回 true 表示节点已被标记
    return true;
  }

  // 循环是特殊情况，因为我们需要使其收敛
  // 考虑以下循环：
  //   for i in range(3):
  //     tot += a[0][0]
  //     b = a[0]
  //     b[0] += 1
  //   print(tot)
  //
  // 如果我们只处理循环块一次，我们会得出结论 `b[0]` 和 `b` 是死变量，
  // 即使 `b[0] += 1` 修改了一个存活的内存位置（因为 `b[0]` 是 `a` 的别名）。
  // 即 `a` 在下一次迭代中被用于计算 `tot`
  //
  // 我们需要使用 `a` 是存活的信息重新标记循环，并重复直到不再标记新内容。
  //
  // 返回 true 当且仅当这标记了我们之前未标记的内容。
  bool markLoop(Node* node) {
    // 断言节点的类型是 prim::Loop
    TORCH_INTERNAL_ASSERT(node->kind() == prim::Loop);
    // 单次迭代循环块是否标记了新内容？
    // 如果为 false，表示已经收敛
    bool marked = false;
    // 是否曾经标记了新内容？
    bool anyMarked = false;
    // 循环直到不再标记新内容为止
    do {
      marked = mark(node->blocks().at(0));
      anyMarked |= marked;
    } while (marked);
  // 返回 true，当且仅当此处标记了我们之前没有标记的东西。
  bool mark(Block* block) {
    bool anyMarked = false;
    // 标记所有具有副作用的节点。
    for (auto node : block->nodes()) {
      if (sideEffectPolicy_ ==
              DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS &&
          hasSideEffects(node)) {
        anyMarked |= mark(node);
      }
    }

    // 初始化标记返回节点
    anyMarked |= markReturnNode(block->return_node());

    // 反向遍历节点列表
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); ++it) {
      auto node = *it;
      if (node->kind() == prim::Loop) {
        // 循环的特殊情况处理，参见 markLoop 中的注释。
        anyMarked |= markLoop(node);
      } else {
        // 其他具有子块的节点正常标记。
        for (auto subBlock : node->blocks()) {
          anyMarked |= mark(subBlock);
        }
      }
      anyMarked |= markIfLive(node);
    }
    return anyMarked;
  }

  // 如果我们输出或写入到活跃内存位置，标记此节点
  // 返回 true，当且仅当此处标记了我们之前没有标记的东西。
  bool markIfLive(Node* node) {
    for (const auto output : node->outputs()) {
      if (liveValues_.count(output)) {
        return mark(node);
      }
    }

    if (useAliasDb_) {
      if (getOrCreateAliasDb()->writesToAlias(node, liveValues_)) {
        return mark(node);
      }
    }

    return false;
  }

  // 标记此节点为活跃，并将此节点的输入和别名添加到活跃值集合中。
  // 返回 true，当且仅当此处标记了我们之前没有标记的东西。
  bool mark(Node* node) {
    if (marked_.count(node)) {
      return false;
    }

    marked_.insert(node);

    // 标记此节点区块链中的所有节点（因为如果它们包含一个活跃节点，则拥有节点也被认为是活跃的）
    auto curNode = node;
    while (curNode) {
      if (!curNode->owningBlock()) {
        break;
      }

      mark(curNode);
      curNode = curNode->owningBlock()->owningNode();
    }

    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    for (const auto input : node->inputs()) {
      if (liveValues_.count(input)) {
        continue;
      }
      liveValues_.insert(input);
    }
    return true;
  }
    // 遍历节点集合，使用迭代器访问每个节点
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      // 在递归之前执行这些操作是为了发现用于计算输出的块中的死代码
      // 移除当前节点中的死块输出
      removeDeadBlockOutputs(node);
      // 移除当前节点中的死循环输出
      removeDeadLoopOutputs(node);
      // 如果设置了递归标志，则对当前节点的所有块进行扫描
      if (recurse) {
        for (Block* block : node->blocks()) {
          sweep(block, true);
        }
      }
      // 注意：检查节点是否被标记过或者是否有使用它的地方。AD 图并非完全有效，
      // 因为 grad_desc.f 中的一个节点可能在 reverse_block 中被使用。
      // Reverse_block 在分离到 grad_desc.df 之前被内联到 grad_desc.f 中。
      if (!(marked_.count(node) || node->hasUses())) {
        // 如果节点既没有被标记过，也没有被使用，则输出信息并移除该节点
        GRAPH_UPDATE(
            "Node ",
            it->kind().toQualString(),
            " which outputs ",
            (!node->outputs().empty() ? node->outputs().at(0)->debugName()
                                      : "n/a"),
            " will be removed");
        it.destroyCurrent(); // 销毁当前节点
      }
    }
  }

  // 检查节点是否存在未跟踪的突变
  bool hasUntrackedMutation(Node* node) {
    if (!useAliasDb_) {
      // 如果没有别名信息，则所有可变操作都有未知效果，无法消除
      if (node->kind() == prim::SetAttr) {
        // SetAttr 是特例：它没有模式(schema)，但是有未跟踪的突变
        return true;
      }
      // 在导出到 ONNX 时调用 EliminateDeadCode，但有时会传递无效的 aten 操作符。
      // 因此我们调用 maybeSchema 来处理节点没有有效模式的情况
      auto schema = node->maybeSchema();
      return schema && schema->is_mutable(); // 检查模式是否可变
    } else {
      // 如果有别名信息，则使用别名数据库来检查节点是否写入通配符
      return getOrCreateAliasDb()->writesToWildcard(node);
    }
  }

  // 检查节点是否具有副作用
  bool hasSideEffects(Node* node) {
    auto it = memo_.find(node);
    if (it != memo_.end())
      return it->second;
    // 检查节点本身是否有副作用，或者它的任何块中的任何节点是否有副作用，或者它是否有未跟踪的突变
    bool has_side_effects = node->hasSideEffects() ||
        std::any_of(node->blocks().begin(),
                    node->blocks().end(),
                    [&](Block* b) {
                      return std::any_of(
                          b->nodes().begin(), b->nodes().end(), [&](Node* n) {
                            return hasSideEffects(n); // 递归检查块中的节点
                          });
                    }) ||
        hasUntrackedMutation(node);

    memo_.emplace(node, has_side_effects);
    return has_side_effects;
  }

  // 移除节点中的死块输出
  void removeDeadBlockOutputs(Node* node) {
    // 如果节点不是 prim::If 或 prim::GradOf 类型，则直接返回
    if (node->kind() != prim::If && node->kind() != prim::GradOf) {
      return;
    }
  for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
    // 从节点的输出数量开始遍历，逆序处理每个输出
    size_t i = i_1 - 1;
    // 获取当前输出的索引，注意索引是从后往前的
    if (!node->outputs().at(i)->hasUses()) {
      // 检查当前输出是否没有被使用
      GRAPH_UPDATE(
          "Dead ",
          i,
          "-th output ",
          node->outputs().at(i)->debugName(),
          " of node ",
          node->kind().toQualString(),
          " will be removed");
      // 记录日志：将被移除的节点的输出信息
      node->eraseOutput(i);
      // 移除节点的指定输出
      for (Block* b : node->blocks()) {
        // 遍历节点的所有块
        GRAPH_UPDATE(
            "\tCorresponding block output ",
            b->outputs().at(i)->debugName(),
            " will be removed");
        // 记录日志：将被移除的块的输出信息
        b->eraseOutput(i);
        // 移除块的指定输出
      }
    }
  }
}

void removeDeadLoopOutputs(Node* node) {
  if (node->kind() != prim::Loop)
    // 如果节点不是循环节点，则直接返回
    return;
  auto loop_body = node->blocks().at(0);
  // 获取循环节点的第一个块
  auto loop_input_offset = 2; // loop carried deps 在输入列表中的偏移量
  auto loop_body_offset =
      1; // 块输入/输出中循环中承载的依赖项的偏移量

  for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
    // 从节点的输出数量开始遍历，逆序处理每个输出
    size_t i = i_1 - 1;
    // 获取当前输出的索引，注意索引是从后往前的
    if (!node->outputs().at(i)->hasUses() &&
        !loop_body->inputs().at(loop_body_offset + i)->hasUses()) {
      // 检查当前输出是否没有被使用，并且循环块的对应输入也没有被使用
      logDeadLoopOutputs(node, i, loop_input_offset, loop_body_offset);
      // 记录循环输出的日志信息
      node->eraseOutput(i);
      // 移除节点的指定输出
      node->removeInput(loop_input_offset + i);
      // 移除节点的指定输入
      loop_body->eraseInput(loop_body_offset + i);
      // 移除循环块的指定输入
      loop_body->eraseOutput(loop_body_offset + i);
      // 移除循环块的指定输出
    }
  }
}

void logDeadLoopOutputs(
    Node* node,
    size_t i,
    size_t loop_input_offset,
    size_t loop_body_offset) {
  auto loop_body = node->blocks().at(0);
  // 获取循环节点的第一个块
  GRAPH_UPDATE(
      "Dead ",
      loop_input_offset + i,
      "-th input ",
      node->inputs().at(i)->debugName(),
      " will be removed");
  // 记录日志：将被移除的循环节点的输入信息
  GRAPH_UPDATE(
      "Dead ",
      i,
      "-th output ",
      node->outputs().at(i)->debugName(),
      " will be removed");
  // 记录日志：将被移除的循环节点的输出信息
  GRAPH_UPDATE(
      "\tDead block input ",
      loop_body->inputs().at(loop_body_offset + i)->debugName(),
      "at offset ",
      loop_body_offset + i,
      " will be removed");
  // 记录日志：将被移除的循环块的输入信息
  GRAPH_UPDATE(
      "\tDead block output ",
      loop_body->outputs().at(loop_body_offset + i)->debugName(),
      "at offset ",
      loop_body_offset + i,
      " will be removed");
  // 记录日志：将被移除的循环块的输出信息
}

AliasDb* getOrCreateAliasDb() {
  if (!aliasDb_) {
    aliasDb_ = std::make_unique<AliasDb>(graph_);
  }
  return aliasDb_.get();
}

DCESideEffectPolicy sideEffectPolicy_;

std::shared_ptr<Graph> graph_;
bool useAliasDb_ = false;
// 懒初始化
std::unique_ptr<AliasDb> aliasDb_ = nullptr;
// 懒初始化的别名数据库
std::unordered_map<Node*, bool> memo_;
// 用于存储节点及其标记信息的映射
std::unordered_set<Node*> marked_;
// 用于存储被标记的节点的集合
std::unordered_set<const Value*> liveValues_;
// 存储活跃值的集合
std::function<void(const std::unordered_set<const Value*>&)> deleteCallback_ =
    [](const std::unordered_set<const Value*>&) {};
// 删除回调函数的初始化
};

// 函数定义：消除死代码
void EliminateDeadCode(
    const std::shared_ptr<Graph>& graph,  // 输入参数：图的智能指针
    DCESideEffectPolicy sideEffectPolicy) {  // 输入参数：死代码消除的副作用策略
  // 创建死代码消除器对象，并运行消除死代码的操作
  DeadCodeEliminator(graph, sideEffectPolicy)
      .run(graph->block(), /*recurse=*/true);  // 在图的起始块上运行死代码消除，支持递归处理
  // 输出调试信息：消除死代码后的图
  GRAPH_DUMP("After EliminateDeadCode: ", graph);
}

// 函数定义：消除死代码
void EliminateDeadCode(
    Block* block,  // 输入参数：待处理的块
    bool recurse,  // 输入参数：是否递归处理子块的标志
    DCESideEffectPolicy sideEffectPolicy) {  // 输入参数：死代码消除的副作用策略
  // 创建死代码消除器对象，并运行消除死代码的操作
  DeadCodeEliminator(sideEffectPolicy).run(block, recurse);
}

// 函数定义：消除死代码
void EliminateDeadCode(
    Block* block,  // 输入参数：待处理的块
    std::function<void(const std::unordered_set<const Value*>&)> cb,  // 输入参数：删除回调函数
    DCESideEffectPolicy sideEffectPolicy) {  // 输入参数：死代码消除的副作用策略
  // 创建死代码消除器对象
  DeadCodeEliminator eliminator(sideEffectPolicy);
  // 设置死代码消除器的删除回调函数
  eliminator.setDeleteCallback(std::move(cb));
  // 运行死代码消除的操作，在给定的块上，支持递归处理子块
  eliminator.run(block, /*recurse=*/true);
}

// 命名空间结束标记：jit
} // namespace jit
// 命名空间结束标记：torch
} // namespace torch
```