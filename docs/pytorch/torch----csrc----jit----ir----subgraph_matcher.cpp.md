# `.\pytorch\torch\csrc\jit\ir\subgraph_matcher.cpp`

```
/**
 * \brief Include necessary header files for the C10 library and Torch JIT.
 */
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>

#include <regex>  // Include regex library for regular expressions
#include <stack>  // Include stack library for stack data structure

namespace torch::jit {
namespace {

/**
 * \brief A class implementing an API for comparing subgraphs.
 */
class SubgraphMatcher {
 public:
  explicit SubgraphMatcher(const Graph& pattern) : pattern_(pattern) {}

  /**
   * \brief Compare matchGraph with the part of the graph denoted by a node \p ANCHOR.
   *
   * The anchor node would be compared against the deepest node in the
   * match-graph. A node is considered matching if its number of inputs/outputs
   * is the same as in the corresponding matchGraph node, its type is the same,
   * and all nodes producing input-values also match.
   */
  bool matchesSubgraphFromAnchorNode(Node* anchor);

  /** \brief Return match map for nodes. */
  std::unordered_map<const Node*, Node*> nodes_map() const {
    return nodes_map_;
  }

  /** \brief Return match map for values. */
  std::unordered_map<const Value*, Value*> values_map() const {
    return values_map_;
  }

 private:
  /**
   * \brief Check if the given Value \p v is an input to the node it belongs to.
   */
  static bool isInput(const Value* v);

  /**
   * \brief Check if the given Value \p v is an output from the node it belongs to.
   */
  static bool isOutput(const Value* v);

  /**
   * \brief Match two Values, v1 from the pattern graph and v2 from the actual graph.
   *
   * Values are considered matching if:
   * 1) The nodes defining them match.
   * 2) They have the same number of uses, except for entry or exit nodes.
   */
  bool matchValues(const Value* v1, Value* v2);

  /**
   * \brief Match two Nodes, n1 from the pattern graph and n2 from the actual graph.
   *
   * Nodes are considered matching if:
   * - Their number of inputs and outputs match.
   * - Their types match.
   * - All nodes producing input-values also match.
   */
  bool matchNodes(const Node* n1, Node* n2);

  /**
   * \brief Match attributes of two Nodes, n1 from the pattern graph and n2 from the actual graph.
   */
  bool matchAttributes(const Node* n1, Node* n2);

  std::unordered_map<const Node*, Node*> nodes_map_;   ///< Map for matched nodes
  std::unordered_map<const Value*, Value*> values_map_; ///< Map for matched values

  const Graph& pattern_; ///< Reference to the pattern graph
  const Node* anchor_ = nullptr; ///< Anchor node for subgraph matching
};

/**
 * \brief Verify that the provided pattern graph \p pattern is valid.
 *
 * This function checks that the pattern graph has a single block and
 * verifies that nodes in the pattern don't alias.
 */
bool patternGraphIsValid(const Graph& pattern) {
  // Verify that pattern graph has a single block.
  for (const Node* n : pattern.nodes()) {
    if (!n->blocks().empty()) {
      return false;
    }
  }

  // TODO: Verify that nodes in the pattern don't alias.
  return true;
}

bool SubgraphMatcher::isInput(const Value* v) {
  return v->node()->kind() == prim::Param;
}

bool SubgraphMatcher::isOutput(const Value* v) {
  for (const Value* output : v->owningGraph()->outputs()) {
    if (v == output) {
      return true;
    }
  }
  return false;
}

/**
 * \brief Compare two Values, v1 from the pattern and v2 from the actual graph.
 *
 * Values are considered matching if:
 * - The nodes defining them match.
 * - They have the same number of uses, except for entry or exit nodes.
 */
bool SubgraphMatcher::matchValues(const Value* v1, Value* v2) {
  // Check if we've already visited these values.
  if (values_map_.count(v1)) {
    if (values_map_.at(v1) != v2) {
      GRAPH_DEBUG(
          "Values %",
          v1->debugName(),
          " and %",
          v2->debugName(),
          " did not match because %",
          v1->debugName(),
          " has already been matched with %",
          values_map_.at(v1)->debugName(),
          ".\n");
      return false;
    }
  }
    // 返回 true，表示节点匹配成功
    return true;
  }

  // 当 V2 是 ANCHOR 时，我们比较现有值；当 V1->node 是 PARAM 时，我们比较进入值。
  // 在这两种情况下，它们的使用次数不需要相同。
  if (v1->uses().size() != v2->uses().size() && !isOutput(v1) && !isInput(v1)) {
    // 如果两个值的使用次数不同，并且 V1 不是输出也不是输入节点，则匹配失败。
    GRAPH_DEBUG(
        "Values %",
        v1->debugName(),
        " and %",
        v2->debugName(),
        " did not match because number of their uses is different.\n");
    return false;
  }

  // 将值添加到映射中，然后调用 matchNodes，以避免无限递归。
  GRAPH_DEBUG(
      "Values %", v1->debugName(), " and %", v2->debugName(), " matched.\n");
  // 将 v1 和 v2 添加到值映射中，表示它们匹配成功。
  values_map_[v1] = v2;
  // 递归调用 matchNodes，比较两个节点的子节点是否匹配。
  return matchNodes(v1->node(), v2->node());
}

bool SubgraphMatcher::matchAttributes(const Node* n1, Node* n2) {
  // 检查两个节点的属性数量是否相同
  if (n1->numAttributes() != n2->numAttributes()) {
    GRAPH_DEBUG("Nodes did not match in number attributes:\n", *n1, *n2);
    return false;
  }
  // 遍历节点 n1 的所有属性名
  for (const Symbol& attr_name : n1->attributeNames()) {
    // 检查属性类型是否匹配
    if (n1->kindOf(attr_name) != n2->kindOf(attr_name)) {
      GRAPH_DEBUG(
          "Nodes did not match because type of attribute '",
          attr_name.toQualString(),
          "' did not match:\n",
          *n1,
          *n2);
      return false;
    }
    // 根据属性类型进行比较
    switch (n1->kindOf(attr_name)) {
      case AttributeKind::s: // 字符串类型属性比较
        if (!std::regex_match(n2->s(attr_name), std::regex(n1->s(attr_name)))) {
          GRAPH_DEBUG(
              "Nodes did not match because attribute '",
              attr_name.toQualString(),
              "' did not match: ",
              n1->s(attr_name),
              " != ",
              n2->s(attr_name),
              " \n",
              *n1,
              *n2);
          return false;
        }
        break;
      case AttributeKind::c: // 字符类型属性比较
        if (n1->c(attr_name) != n2->c(attr_name)) {
          GRAPH_DEBUG(
              "Nodes did not match because attribute '",
              attr_name.toQualString(),
              "' did not match:",
              n1->c(attr_name),
              " != ",
              n2->c(attr_name),
              " \n",
              *n1,
              *n2);
          return false;
        }
        break;
      case AttributeKind::f: // 浮点数类型属性比较
        if (n1->f(attr_name) != n2->f(attr_name)) {
          GRAPH_DEBUG(
              "Nodes did not match because attribute '",
              attr_name.toQualString(),
              "' did not match:",
              n1->f(attr_name),
              " != ",
              n2->f(attr_name),
              " \n",
              *n1,
              *n2);
          return false;
        }
        break;
      case AttributeKind::i: // 整数类型属性比较
        if (n1->i(attr_name) != n2->i(attr_name)) {
          GRAPH_DEBUG(
              "Nodes did not match because attribute '",
              attr_name.toQualString(),
              "' did not match:",
              n1->i(attr_name),
              " != ",
              n2->i(attr_name),
              " \n",
              *n1,
              *n2);
          return false;
        }
        break;
      default: {
        // 不支持的其他属性类型
        GRAPH_DEBUG(
            "Nodes did not match because type of attribute '",
            attr_name.toQualString(),
            "' is not supported.\n",
            *n1,
            *n2);
        return false;
      }
    }
  }
  // 所有属性匹配成功，返回 true
  return true;
}

// 检查字符串是否以指定后缀结尾
static bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
      0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}
/**
 * Compare two Nodes. N1 is from pattern, N2 is from the actual graph.
 *
 * The nodes are considered matching if:
 * 1) N1 and N2 are of the same kind.
 * 2) Number of inputs and outputs is the same.
 * 3) All input and output values match.
 *
 * A special case is when N1 is PARAM - this is considered outside the pattern,
 * so it matches everything.
 */
bool SubgraphMatcher::matchNodes(const Node* n1, Node* n2) {
  // Check if we've already visited these nodes.
  if (nodes_map_.count(n1)) {
    // Return true if N1 has already been mapped to N2.
    return nodes_map_.at(n1) == n2;
  }

  // Param node in pattern graph matches everything.
  if (n1->kind() == prim::Param) {
    // Debugging message indicating nodes matched due to PARAM node.
    GRAPH_DEBUG("Nodes matched:\n", *n1, *n2);
    return true;
  }

  // We don't allow matches to span across blocks, so check if N2 is in the same
  // block as the first (anchor) node.
  if (n2->owningBlock() != anchor_->owningBlock()) {
    // Debugging message indicating nodes did not match due to different blocks.
    GRAPH_DEBUG(
        "Nodes did not match because it is in the different block:\n",
        *n1,
        *n2);
    return false;
  }

  // Special handling for matching modules
  if (n1->kind() == Symbol::fromQualString("match::module")) {
    if (n2->kind() == prim::GetAttr) {
      if (!n1->hasAttributeS("name")) {
        // Debugging message indicating nodes did not match due to missing 'name' attribute.
        GRAPH_DEBUG(
            "Nodes did not match because special node match::module does not have 'name' attribute:\n",
            *n1,
            *n2);
        return false;
      }
      auto t = n2->output()->type()->expect<ClassType>();
      auto real_typename = t->name()->qualifiedName();
      auto pattern_typename = n1->s(attr::name);
      if (!endsWith(real_typename, pattern_typename)) {
        // Debugging message indicating nodes did not match due to different module types.
        GRAPH_DEBUG(
            "Nodes did not match because expected module type is different:\n");
        GRAPH_DEBUG("  actualtype:    ", real_typename, "\n");
        GRAPH_DEBUG("  expected type: ", pattern_typename, "\n");
        GRAPH_DEBUG("Nodes:", *n1, *n2);
        return false;
      }
    }
  } else {
    // General case: check if N1 and N2 have different kinds or mismatch in IO sizes.
    if (n1->kind() != n2->kind() ||
        n1->outputs().size() != n2->outputs().size() ||
        n1->inputs().size() != n2->inputs().size()) {
      // Debugging message indicating nodes did not match in kind or IO sizes.
      GRAPH_DEBUG(
          "Nodes did not match in their kind or number of inputs/outputs:\n",
          *n1,
          *n2);
      return false;
    }
    // Check if node attributes match.
    if (!matchAttributes(n1, n2)) {
      return false;
    }
  }

  // Add nodes to the map before calling matchValues to avoid infinite
  // recursion.
  nodes_map_[n1] = n2;
  // Recursively match output and input values of N1 and N2.
  for (const auto i : c10::irange(n1->outputs().size())) {
    if (!matchValues(n1->outputs()[i], n2->outputs()[i])) {
      return false;
    }
  }
  for (const auto i : c10::irange(n1->inputs().size())) {
    if (!matchValues(n1->inputs()[i], n2->inputs()[i])) {
      return false;
    }
  }

  // Debugging message indicating nodes matched successfully.
  GRAPH_DEBUG("Nodes matched:\n", *n1, *n2);
  return true;
}

/**
 * Recursively try to match pattern with the actual graph starting from the
 * exiting node in the pattern and anchor node in the actual graph.
 */
// 开始从新的锚点节点进行子图匹配，记录日志并输出锚点节点的详细信息
GRAPH_UPDATE("Starting match from a new anchor: ", *anchor);
// 清空节点映射和数值映射，准备重新进行匹配
nodes_map_.clear();
values_map_.clear();
// 将当前锚点设为给定的 anchor 节点
anchor_ = anchor;

// 从模式的最后一个节点获取底部节点，并将其指向其输入的第一个节点
const Node* bottom_node = *(pattern_.nodes().end());
bottom_node = bottom_node->input(0)->node();

// 如果无法将底部节点与锚点节点匹配，则返回匹配失败
if (!matchNodes(bottom_node, anchor)) {
  return false;
}

// 确保模式中所有的输出都在值映射中存在，否则断言失败
for (const Value* output : pattern_.outputs()) {
  AT_ASSERT(values_map_.count(output));
}

// 记录日志，表示模式匹配成功
GRAPH_UPDATE("Pattern matched!\n");
return true;
}

} // unnamed namespace

// 子图匹配的主要入口点。
std::vector<Match> findPatternMatches(const Graph& pattern, Graph& graph) {
  // 断言确保模式图有效
  AT_ASSERT(patternGraphIsValid(pattern));
  // 输出模式图和目标图的详细结构
  GRAPH_DUMP("Pattern graph: ", &pattern);
  GRAPH_DUMP("Target graph: ", &graph);

  // 创建子图匹配器对象，并初始化
  SubgraphMatcher m(pattern);
  std::vector<Match> matches;
  std::stack<Block*> blocks_to_visit;

  // 遍历图中所有节点（包括子块中的节点），尝试为每个节点匹配模式
  blocks_to_visit.push(graph.block());
  while (!blocks_to_visit.empty()) {
    Block* block = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : block->nodes()) {
      // 如果从当前节点开始能够匹配到子图，将匹配的结果存入 matches 中
      if (m.matchesSubgraphFromAnchorNode(n)) {
        matches.push_back({n, m.nodes_map(), m.values_map()});
      }
      // 将当前节点的所有子块加入待访问列表
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  // 返回所有匹配到的结果
  return matches;
}

} // namespace torch::jit
```