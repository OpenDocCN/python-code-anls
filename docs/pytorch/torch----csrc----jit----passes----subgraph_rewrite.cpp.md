# `.\pytorch\torch\csrc\jit\passes\subgraph_rewrite.cpp`

```py
// 包含 Torch 的 JIT 子图重写相关头文件
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

// 包含 Torch 的 IR 解析器相关头文件
#include <torch/csrc/jit/ir/irparser.h>
// 包含 Torch 的子图匹配器相关头文件
#include <torch/csrc/jit/ir/subgraph_matcher.h>

// 包含 C10 库的实用工具，用于生成范围内的整数序列
#include <c10/util/irange.h>

// 引入 C++ 标准库的实用工具
#include <utility>

// Torch 的命名空间
namespace torch {
namespace jit {

// 匿名命名空间，用于定义局部函数和变量
namespace {
// 更新源范围和调用堆栈指针的函数，用于将匹配到的模式图中的源范围和调用堆栈信息更新到替换图中的节点
void update_source_range_and_cs_ptr(
    const std::set<const Node*>& input_nodes,  // 输入节点的集合
    const Match& m,  // 匹配对象，包含模式节点到匹配节点的映射
    std::unordered_map<Node*, Node*>& pattern_node_map) {  // 模式节点到替换节点的映射表
  // 遍历模式节点到替换节点的映射表
  for (auto& it : pattern_node_map) {
    Node* replacement_node = it.first;   // 替换节点
    Node* pattern_node = it.second;      // 模式节点
    // 如果模式节点不在输入节点集合中
    if (!input_nodes.count(pattern_node)) {
      Node* orig_node = m.nodes_map.at(pattern_node);  // 获取模式节点在匹配中的原始节点
      replacement_node->setSourceRange(orig_node->sourceRange());  // 更新替换节点的源范围
      if (orig_node->callstack()) {
        replacement_node->setCallStack(orig_node->callstack().value());  // 更新替换节点的调用堆栈信息
      }
    }
  }
}
} // namespace

// 注册默认模式的函数
void SubgraphRewriter::RegisterDefaultPatterns() {
  // TODO: 添加实际的模式（如 Conv-Relu）
  RegisterRewritePattern(
      R"IR(
graph(%x, %w, %b):
  %c = aten::conv(%x, %w, %b)
  %r = aten::relu(%c)
  return (%r))IR",
      R"IR(
graph(%x, %w, %b):
  %r = aten::convrelu(%x, %w, %b)
  return (%r))IR",
      {{"r", "c"}});  // 注册 Conv-Relu 模式的重写规则
}

// 注册重写模式的函数，接受模式字符串、替换字符串和值名称对的向量
void SubgraphRewriter::RegisterRewritePattern(
    const std::string& pattern,  // 模式字符串
    const std::string& replacement,  // 替换字符串
    const std::vector<std::pair<std::string, std::string>>& value_name_pairs) {  // 值名称对的向量
  std::unordered_map<std::string, std::string> value_name_map(
      value_name_pairs.begin(), value_name_pairs.end());  // 根据值名称对创建映射表
  RewritePatternDescr d = {pattern, replacement, std::move(value_name_map)};  // 创建重写模式描述对象
  patterns_.push_back(std::move(d));  // 将重写模式描述对象添加到模式列表中
}

// 在模块上运行的函数，接受模块对象作为参数
Module SubgraphRewriter::runOnModule(const Module& module) {
  nodes_to_delete_.clear();  // 清空待删除节点列表
  // 遍历模块中的每个方法
  for (const auto& m : module.get_methods()) {
    auto g = toGraphFunction(m.function()).graph();  // 将方法转换为图形函数，获取其图对象
    runOnGraph(g);  // 在图对象上运行子图重写
  }
  return module;  // 返回处理后的模块对象
}

// 在图对象上运行子图重写的函数，接受图对象和匹配过滤器的向量作为参数
void SubgraphRewriter::runOnGraph(
    std::shared_ptr<Graph>& graph,  // 图对象的智能指针
    const std::vector<MatchFilter>& filters) {  // 匹配过滤器的向量
  // 遍历每个注册的重写模式
  for (const RewritePatternDescr& pattern : patterns_) {
    rewriteSinglePatternOnGraph(graph, pattern, filters);  // 在图对象上应用单个重写模式
  }
}

// 在图对象上应用单个重写模式的函数，接受图对象、重写模式描述对象和匹配过滤器的向量作为参数
void SubgraphRewriter::rewriteSinglePatternOnGraph(
    std::shared_ptr<Graph>& graph,  // 图对象的智能指针
    const RewritePatternDescr& pattern,  // 重写模式描述对象
    const std::vector<MatchFilter>& filters) {  // 匹配过滤器的向量
  // 此处实现省略，用以应用单个重写模式在图对象上
}
    const std::vector<MatchFilter>& filters) {
  // 重写映射，用于存储需要替换的值和替换后的值的映射关系
  std::unordered_map<Value*, Value*> rewrite_map;
  // 需要重写的值的列表
  std::vector<Value*> values_to_rewrite;

  // 创建模式图对象，并解析模式的 IR 表示到模式图中
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  parseIR(pattern.pattern, &pattern_graph, vmap);

  // 创建替换图对象，并解析替换的 IR 表示到替换图中
  Graph replacement_graph;
  std::unordered_map<std::string, Value*> vmap_replacement;
  parseIR(pattern.replacement, &replacement_graph, vmap_replacement);

  // 首先构建节点指针之间的映射关系
  // 这将替换图中的节点映射到模式图中的节点
  // 使用 value_name_map 将替换模式中的值名称映射到模式中的值名称
  std::unordered_map<Node*, Node*> pattern_node_map;
  // 用于存储模式输入节点的集合
  std::set<const Node*> pattern_input_nodes;
  for (auto& it : vmap_replacement) {
    const auto& replacement_value_name = it.first;
    Node* replacement_value_node = it.second->node();
    if (pattern.value_name_map.count(replacement_value_name)) {
      const auto& pattern_value_name =
          pattern.value_name_map.at(replacement_value_name);
      // 检查模式图中是否存在该值
      TORCH_CHECK(
          vmap.count(pattern_value_name),
          "Value must be found in the replacement graph.");
      Node* pattern_value_node = vmap.at(pattern_value_name)->node();
      pattern_node_map.emplace(replacement_value_node, pattern_value_node);
    }
  }

  // 查找模式图在当前图中的所有匹配
  const auto& matches = findPatternMatches(pattern_graph, *graph);
  // 遍历每一个匹配
  for (const Match& match : matches) {
    // 检查是否通过所有的过滤器，如果没有通过则跳过这个匹配
    if (!std::all_of(filters.begin(), filters.end(), [&](const MatchFilter& f) {
          return f(match, vmap);
        })) {
      continue;
    }
    // 匹配可能会重叠，如果与之前的匹配重叠，则跳过这个匹配
    if (overlapsWithPreviousMatches(&match)) {
      continue;
    }

    // 确定替换子图的输入和输出值，以及替换子图应该插入的位置
    Node* ins_point = nullptr;
    std::vector<Value*> inputs, outputs;
    // 遍历模式图的输入节点
    for (Value* v : pattern_graph.inputs()) {
      // 获取当前匹配中的值对应的输入值
      Value* input = match.values_map.at(v);
      if (!ins_point || ins_point->isBefore(input->node())) {
        ins_point = input->node();
      }
      inputs.push_back(input);
    }
    AT_ASSERT(ins_point);

    // 检查我们选择的插入点是否在所有输出值的使用之前
    bool ins_point_before_uses = true;
    // 遍历模式图的输出节点
    for (Value* v : pattern_graph.outputs()) {
      // 获取当前匹配中的值对应的输出值
      Value* output = match.values_map.at(v);
      outputs.push_back(match.values_map.at(v));

      // 检查输出值的每一个使用情况是否在插入点之前
      for (const Use& u : output->uses()) {
        if (u.user->isBefore(ins_point)) {
          ins_point_before_uses = false;
          break;
        }
      }
    }

    // 如果插入点在所有输出值的使用之前，则可以进行替换
    if (!ins_point_before_uses) {
      continue;
    }

    // 在重写图之前，更新源范围和调用堆栈
    // 更新替换模式图中的源范围和控制流指针，以便重写后的图形具有更新的信息
    update_source_range_and_cs_ptr(
        pattern_input_nodes, match, pattern_node_map);



    // 在插入点的下一个位置插入替换子图的克隆。
    // `inputs` 向量包含我们作为新子图的入站值使用的值，
    // 我们将得到 `new_outputs` 向量，其中包含此新子图生成的值，
    // 然后我们将使用新的输出重写旧的输出。
    WithInsertPoint insert_point(ins_point->next());
    std::vector<Value*> new_outputs =
        insertGraph(*graph, replacement_graph, inputs);



    // 记录所有计划的重写
    AT_ASSERT(outputs.size() == new_outputs.size());
    for (const auto idx : c10::irange(outputs.size())) {
      values_to_rewrite.push_back(outputs[idx]);
      rewrite_map[outputs[idx]] =
          new_outputs[idx]->setType(outputs[idx]->type());
    }



    // 记录所有计划的删除
    for (Node* pattern_n : pattern_graph.nodes()) {
      if (match.nodes_map.count(pattern_n)) {
        Node* n = match.nodes_map.at(pattern_n);
        nodes_to_delete_.insert(n);
      }
    }
  }



  // 执行计划的重写
  for (auto v : values_to_rewrite) {
    v->replaceAllUsesWith(rewrite_map.at(v));
  }



  // 执行计划的删除
  for (auto n : nodes_to_delete_) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete_) {
    n->destroy();
  }
  nodes_to_delete_.clear();
}

bool SubgraphRewriter::overlapsWithPreviousMatches(const Match* match) {
  // 遍历给定匹配对象的节点映射
  for (auto n : match->nodes_map) {
    // 如果节点映射中的节点ID存在于nodes_to_delete_集合中，则存在重叠匹配
    if (nodes_to_delete_.count(n.second)) {
      return true;
    }
  }
  // 没有发现重叠匹配
  return false;
}

Module PatternBasedRewrite(const Module& module) {
  // TODO: 深拷贝模块（尚未实现）
  SubgraphRewriter subgraph_rewriter;
  // 注册默认的模式匹配规则
  subgraph_rewriter.RegisterDefaultPatterns();
  // 对给定的模块执行重写操作并返回
  return subgraph_rewriter.runOnModule(module);
}

} // namespace jit
} // namespace torch
```