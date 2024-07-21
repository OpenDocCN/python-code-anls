# `.\pytorch\torch\csrc\jit\passes\create_functional_graphs.cpp`

```py
    /*
    反向迭代块，并创建功能图形。
    遇到非功能节点时跳过，否则尝试将功能节点合并到当前功能子图中。
    如果无法合并到当前功能子图节点，则开始一个新的功能子图组。
    */
    bool changed = false;  // 用于跟踪是否有任何更改发生
    std::vector<Node*> functional_graph_nodes;  // 存储功能图节点的向量

    // 在返回节点之前创建一个带有子图的功能图节点
    Node* functional_subgraph_node =
        graph_->createWithSubgraph(prim::FunctionalGraph)
            ->insertBefore(block->return_node());
    // 反向迭代器，从块的末尾向开头遍历节点
    auto reverse_iter = block->nodes().reverse();
    // 存储图输出的值的向量
    std::vector<Value*> graph_outputs;
    for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
      Node* n = *it++;

      // constants get copied into the graph
      // 如果节点是常量或者是功能子图节点本身，则跳过处理
      if (n->kind() == prim::Constant || n == functional_subgraph_node) {
        continue;
      }

      // if `n` is functional, all of its blocks will be merged into the
      // new functional subgraph, so we only need to recurse if it is not
      // functional
      // 如果 `n` 是功能节点，则所有其块将合并到新的功能子图中，因此如果不是功能节点，则需要递归处理其块
      if (!functional_nodes_.count(n)) {
        for (Block* b : n->blocks()) {
          auto block_changed = CreateFunctionalGraphsImpl(b);
          changed = block_changed && changed;
        }
        continue;
      }

      // if `n` represents a functional graph and the current functional subgraph is empty,
      // replace the subgraph with `n`
      // 如果 `n` 表示一个功能图，并且当前功能子图为空，则用 `n` 替换功能子图节点
      if (n->kind() == prim::FunctionalGraph &&
          isEmptyFunctionalGraph(functional_subgraph_node)) {
        functional_subgraph_node->destroy();
        functional_subgraph_node = n;
        continue;
      }

      // mark that changes have been made
      changed = true;
      // move `n` before `functional_subgraph_node` if it's topologically valid,
      // otherwise create a new functional subgraph with `n`
      // 如果可以在拓扑上下文中将 `n` 移动到 `functional_subgraph_node` 之前，则移动它，
      // 否则创建一个新的带有 `n` 的功能子图
      if (aliasDb_->moveBeforeTopologicallyValid(n, functional_subgraph_node)) {
        SubgraphUtils::mergeNodeIntoSubgraph(n, functional_subgraph_node);
      } else {
        functional_graph_nodes.emplace_back(functional_subgraph_node);
        functional_subgraph_node =
            graph_->createWithSubgraph(prim::FunctionalGraph)->insertAfter(n);
        SubgraphUtils::mergeNodeIntoSubgraph(n, functional_subgraph_node);
      }
    }
    // add the final functional subgraph node to the list
    // 将最终的功能子图节点添加到列表中
    functional_graph_nodes.emplace_back(functional_subgraph_node);

    // for each functional node in the graph, attempt inlining and constant pooling
    // 对于图中的每个功能节点，尝试内联和常量池操作
    for (Node* functional_node : functional_graph_nodes) {
      if (!inlineIfTooSmall(functional_node)) {
        ConstantPooling(functional_node->g(attr::Subgraph));
      }
    }
    // indicate whether changes were made during the analysis
    // 返回分析过程中是否有变化发生
    return changed;
  }

  bool AnalyzeFunctionalSubset(Node* n) {
    // TODO: clarify hasSideEffects, isNondeterministic
    bool is_functional_node = true;

    // Functional Graphs are not responsible for maintaining aliasing
    // relationships. If an output of a functional graph escapes scope
    // or is mutated then we might change semantics of the program if
    // aliasing relationships are changed.
    // We don't allow any node in the functional graph to output a value
    // that escapes scope or is mutated, and we don't allow any mutating nodes
    // into the graph.
    // - allow functional graphs to have at most one value that can escape scope
    // - allow outputs which alias the wildcard set but do not "re-escape"
    // 分析功能子集节点，检查是否满足功能节点条件，不允许节点输出可能逃逸作用域或被修改的值
    for (Value* v : n->outputs()) {
      bool has_writers = aliasDb_->hasWriters(v);
      bool escapes_scope = aliasDb_->escapesScope(v);
      if (has_writers) {
        mutated_values_.insert(v);
      }
      is_functional_node = is_functional_node && !escapes_scope && !has_writers;
    }

    // recursively analyze each block in the node
    // 递归分析节点中的每个块
    for (Block* block : n->blocks()) {
      auto functional_block = AnalyzeFunctionalSubset(block);
      is_functional_node = is_functional_node && functional_block;
    }

    // ensure the node itself is not mutable
    // 确保节点本身不可变
    is_functional_node = is_functional_node && !aliasDb_->isMutable(n);
    // if the node satisfies all conditions, mark it as a functional node
    // 如果节点满足所有条件，则将其标记为功能节点
    if (is_functional_node) {
      functional_nodes_.insert(n);
    }
    return is_functional_node;
  }

  void AnalyzeFunctionalSubset(at::ArrayRef<Block*> blocks) {
    // 对传入的每个 Block 进行功能子集分析
    for (Block* block : blocks) {
      AnalyzeFunctionalSubset(block);
    }
  }

  bool AnalyzeFunctionalSubset(Block* block) {
    bool is_functional_block = true;
    // 对 Block 的输入值进行迭代，将有写入操作的值添加到 mutated_values_ 集合中
    for (Value* v : block->inputs()) {
      bool has_writers = aliasDb_->hasWriters(v);
      if (has_writers) {
        mutated_values_.insert(v);
      }
    }
    // 分析 Block 中的每个节点
    for (Node* n : block->nodes()) {
      // 递归调用 AnalyzeFunctionalSubset 分析节点 n 的功能子集
      bool functional = AnalyzeFunctionalSubset(n);
      // 更新当前 Block 是否为功能块的状态
      is_functional_block = is_functional_block && functional;
    }
    // 返回当前 Block 是否为功能块的状态
    return is_functional_block;
  }

  std::unordered_set<Node*> functional_nodes_;
  std::unordered_set<Value*> mutated_values_;
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  size_t minSubgraphSize_ = 6;
};

// 函数：递归地内联功能图形
void InlineFunctionalGraphs(Block* block) {
  // 遍历当前块中的所有节点
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    // 获取当前节点指针，并将迭代器向前移动
    Node* n = *it;
    it++;
    // 遍历当前节点包含的子块，并递归地调用内联功能图形函数
    for (Block* b : n->blocks()) {
      InlineFunctionalGraphs(b);
    }
    // 如果当前节点是功能图形节点
    if (n->kind() == prim::FunctionalGraph) {
      // 解除子图的合并操作
      SubgraphUtils::unmergeSubgraph(n);
    }
  }
}

// namespace 结束

} // namespace

// 函数：创建功能图形
void CreateFunctionalGraphs(const std::shared_ptr<Graph>& graph) {
  // 运行常量池化，以提升常量值
  ConstantPooling(graph);
  // 使用 FunctionalGraphSlicer 对象处理图形
  FunctionalGraphSlicer func(graph);
  func.run();
  // 再次运行常量池化，处理创建功能子图和反内联操作产生的多余常量
  ConstantPooling(graph);
}

// 函数：内联功能图形
void InlineFunctionalGraphs(const std::shared_ptr<Graph>& graph) {
  // 调用递归内联功能图形函数，从图的根块开始
  InlineFunctionalGraphs(graph->block());
}

// namespace jit 结束
} // namespace torch
```