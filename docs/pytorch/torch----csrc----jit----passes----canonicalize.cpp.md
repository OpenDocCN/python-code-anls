# `.\pytorch\torch\csrc\jit\passes\canonicalize.cpp`

```
// 引入 Torch 的 JIT 模块中的标准头文件和类
#include <torch/csrc/jit/passes/canonicalize.h>

// 引入 C++ 标准库头文件
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir_views.h>

// Torch 的命名空间
namespace torch {
namespace jit {

// 对图进行规范化，重新编号以确保所有结构等价的图具有相同的编号。
// keep_unique_names: 如果为 false，则通过移除唯一名称来规范化唯一名称，并用普通值名称替换它们。
//                    如果为 true，则忽略具有唯一名称的值。
std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph,
    bool keep_unique_names) {
  // 创建一个新的图，与原图的当前作用域相同
  auto r = std::make_shared<Graph>(graph->current_scope());

  // 映射旧图中值到新图中值的环境
  std::unordered_map<Value*, Value*> rn_env;

  // 重命名函数，用于在新图中创建节点时更新值的映射关系
  auto rn_fn = [&](Value* v) { return rn_env.at(v); };

  // 复制原图的输入节点到新图中，并建立映射关系
  for (auto* input : graph->inputs()) {
    auto* r_input = r->addInput();
    r_input->copyMetadata(input);
    if (!keep_unique_names)
      r_input->setDebugName("");
    rn_env[input] = r_input;
  }

  // 复制原图的节点到新图中，并更新节点间的连接关系及属性
  for (auto* node : graph->nodes()) {
    auto* r_node = r->createClone(node, rn_fn);

    // 如果不保留唯一名称，则清空新节点的调试名称
    if (!keep_unique_names) {
      for (auto* output : r_node->outputs()) {
        output->setDebugName("");
      }
    }

    // 将新节点添加到新图中
    r->appendNode(r_node);

    // 更新映射环境，将原节点的输出值映射到新节点的输出值
    auto outputs = node->outputs();
    auto r_outputs = r_node->outputs();
    for (const auto i : c10::irange(outputs.size())) {
      rn_env[outputs.at(i)] = r_outputs.at(i);
    }

    // 如果节点包含子图属性，则递归规范化子图
    if (node->hasAttribute(attr::Subgraph)) {
      r_node->g_(
          attr::Subgraph,
          Canonicalize(node->g(attr::Subgraph), keep_unique_names));
    }
  }

  // 将原图的输出节点注册到新图中
  for (auto* output : graph->outputs()) {
    r->registerOutput(rn_fn(output));
  }

  // 返回规范化后的新图
  return r;
}

// 获取块在其拥有节点中的索引
static size_t blockIndex(const Block* b) {
  auto n = b->owningNode();
  AT_ASSERT(n);
  for (size_t i = 0; i < n->blocks().size(); ++i) {
    if (n->blocks()[i] == b) {
      return i;
    }
  }
  AT_ASSERT(false);
}

/*
 * 确定节点的规范顺序。
 * 如果 n1 和 n2 在同一个块中，则先出现的节点排在后面。
 * 如果 n1 和 n2 分别在 if 节点的不同块中，则 true 块中的块排在前面。
 * 如果 n1 包含 n2，则 n1 在 n2 之前。这个属性确保在图的转储中先出现的节点排在后面。
 * 注意：这不是拓扑排序索引。在拓扑上，if 节点不同块中的两个节点并不互为 < 或 >。
 */
static bool isBefore(Node* n1, Node* n2) {
  // 不允许使用相同的节点作为参数调用
  AT_ASSERT(n1 != n2);

  // 计算从 Graph 块到 n1 和 n2 所在块的距离
  size_t d_1 = n1->blocksFromGraphBlock();
  size_t d_2 = n2->blocksFromGraphBlock();

  // 从 n1 开始向上遍历块，直到找到 n2，确定 n1 是否在 n2 之前
  for (; d_1 > d_2; --d_1) {
    n1 = n1->owningBlock()->owningNode();
    if (n1 == n2) {
      return false;
    }
  }

  // 从 n2 开始向上遍历块，直到找到 n1，确定 n1 是否在 n2 之前
  for (; d_2 > d_1; --d_2) {
    n2 = n2->owningBlock()->owningNode();
    if (n2 == n1) {
      return true;
    }
  }

  // 如果没有找到相互包含的情况，说明节点之间不存在顺序关系
  AT_ASSERT(false);
}
  }
}

// 现在它们距离图块的块数相同，
// 递归向上，检查它们是否在同一个块上
while (true) {
  // 检查两个节点是否属于同一个块
  if (n1->owningBlock() == n2->owningBlock()) {
    // 如果在同一个块内，比较节点在块内的顺序
    return n1->isBefore(n2);
  }

  // 获取上一级的块的所有者节点
  auto new_n1 = n1->owningBlock()->owningNode();
  auto new_n2 = n2->owningBlock()->owningNode();

  // 断言新的所有者节点不为空
  AT_ASSERT(new_n1 != nullptr);
  AT_ASSERT(new_n2 != nullptr);

  // 如果新的所有者节点相同，则取早期块中的节点
  if (new_n1 == new_n2) {
    auto index_1 = blockIndex(n1->owningBlock());
    auto index_2 = blockIndex(n2->owningBlock());
    return index_1 < index_2;
  }

  // 更新节点为新的所有者节点，继续迭代检查
  n1 = new_n1;
  n2 = new_n2;
}
}

// 比较两个 Use 对象是否是按照规定顺序的静态函数
static bool isBefore(const Use& a, const Use& b) {
  // 如果两个 Use 对象的用户节点相同，则按偏移量排序
  if (a.user == b.user) {
    return a.offset < b.offset;
  }

  // 否则调用重载函数 isBefore，递归比较用户节点的顺序
  return isBefore(a.user, b.user);
}

// 判断两个 Use 对象是否是按照规定顺序的静态函数
static bool isAfter(const Use& a, const Use& b) {
  // 如果两个 Use 对象的用户节点和偏移量均相同，则返回 false
  if (a.user == b.user && a.offset == b.offset) {
    return false;
  }
  // 否则调用 isBefore 函数，判断是否 a 在 b 之后
  return !isBefore(a, b);
}

// 判断两个 Use 对象是否按照指定顺序，并根据 checking_before 参数确定是按前还是按后顺序
bool isBeforeOrAfter(const Use& a, const Use& b, bool checking_before) {
  // 根据 checking_before 参数调用 isBefore 或 isAfter 函数判断顺序
  return checking_before ? isBefore(a, b) : isAfter(a, b);
}

// 查找值的第一个或最后一个使用位置，并返回其 Use 对象的可选类型
std::optional<const Use> firstOrLastUse(Value* v, bool find_first) {
  // 如果值 v 没有使用位置，则返回空
  if (v->uses().empty()) {
    return c10::nullopt;
  }
  // 获取第一个使用位置的 Use 对象作为极限使用位置
  Use extreme_use = v->uses()[0];
  // 遍历值 v 的所有使用位置
  for (size_t i = 1; i < v->uses().size(); ++i) {
    auto n_use = v->uses()[i];
    // 如果找到更合适的使用位置，则更新极限使用位置
    if (!isBeforeOrAfter(extreme_use, n_use, find_first)) {
      extreme_use = n_use;
    }
  }

  // 返回极限使用位置的 Use 对象
  return extreme_use;
}

// 根据值的数组排序其索引，依据值的第一个使用位置的规范顺序
static std::vector<std::optional<const Use>> gatherFirstUses(
    at::ArrayRef<Value*> values) {
  // 使用 fmap 函数遍历每个值，获取其第一个使用位置的可选类型 Use 对象
  return fmap(values, [&](Value* v) -> std::optional<const Use> {
    return firstOrLastUse(v, true);
  });
}

// 对值的数组进行排序，并返回其索引
static std::vector<size_t> sort_indexes(at::ArrayRef<Value*> values) {
  // 初始化原始索引位置
  std::vector<size_t> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);  // 填充索引从 0 开始递增的值

  // 获取每个值的第一个使用位置
  std::vector<std::optional<const Use>> first_uses = gatherFirstUses(values);

  // 根据值的第一个使用位置的规范顺序对索引进行排序
  std::sort(idx.begin(), idx.end(), [&first_uses](size_t i1, size_t i2) {
    // 如果两个值都没有使用位置，则保持原始顺序
    if (first_uses[i1] == c10::nullopt && first_uses[i2] == c10::nullopt) {
      return i1 < i2;
    }
    // 如果其中一个值没有使用位置，则另一个值优先
    if (first_uses[i1] == c10::nullopt) {
      return false;
    } else if (first_uses[i2] == c10::nullopt) {
      return true;
    }

    // 否则比较两个值的第一个使用位置的顺序
    auto fst_v1 = *first_uses[i1];
    auto fst_v2 = *first_uses[i2];
    return isBefore(fst_v1, fst_v2);
  });

  // 返回排序后的索引
  return idx;
}

// 标准化循环节点的输出顺序
static void CanonicalizeLoopOutputs(Node* n) {
  // 获取排序后的输出索引
  auto new_indices = sort_indexes(n->outputs());
  // 使用 LoopView 类重新排列循环节点的输出
  LoopView(n).permuteLoopCarried(new_indices);
}

// 标准化条件节点的输出顺序
static void CanonicalizeIfOutputs(Node* n) {
  // 获取排序后的输出索引
  auto new_indices = sort_indexes(n->outputs());
  // 使用 IfView 类重新排列条件节点的输出
  IfView(n).permuteOutputs(new_indices);
}

// 递归标准化节点块及其内部节点的输出顺序
static void CanonicalizeOutputs(Block* block) {
  // 逆序迭代节点块中的每个节点，因为节点的输出顺序依赖于图中其后的值使用
  for (Node* n : block->nodes().reverse()) {
    switch (n->kind()) {
      case prim::Loop: {
        // 标准化循环节点的输出顺序
        CanonicalizeLoopOutputs(n);
      } break;
      case prim::If: {
        // 标准化条件节点的输出顺序
        CanonicalizeIfOutputs(n);
      } break;
    }
    // 递归标准化节点的块
    for (Block* b : n->blocks()) {
      CanonicalizeOutputs(b);
    }
  }
}
// 标准化图的控制流节点输出。我们在 ir_emitter.cpp 中的编译第一次通过后执行此操作，
// 以解决在控制流节点中添加输出后引起的抖动问题。
void CanonicalizeOutputs(std::shared_ptr<Graph>& graph) {
  // 调用 CanonicalizeOutputs 函数处理图的块级别结构
  CanonicalizeOutputs(graph->block());
}
} // namespace jit
} // namespace torch
```