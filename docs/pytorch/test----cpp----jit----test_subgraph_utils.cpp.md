# `.\pytorch\test\cpp\jit\test_subgraph_utils.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include "test/cpp/jit/test_utils.h"  // 包含测试工具函数的头文件

#include <torch/csrc/jit/testing/file_check.h>  // 包含文件检查工具的头文件
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"  // 包含公共子表达式消除的头文件
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"  // 包含子图工具函数的头文件

namespace torch {
namespace jit {

TEST(SubgraphUtilsTest, Basic) {  // 定义基础测试用例 SubgraphUtilsTest.Basic
  auto graph = build_lstm();  // 构建一个 LSTM 图

  EliminateCommonSubexpression(graph);  // 对图进行公共子表达式消除

  std::vector<Node*> originalNodes(  // 创建原始节点列表
      graph->nodes().begin(), graph->nodes().end());

  for (bool reverse_iterate : {true, false}) {  // 迭代两次，分别为正向和反向迭代
    // Merge everything into a single subgraph
    bool first = true;  // 标记是否是第一次迭代
    Node* subgraph;  // 定义子图节点指针
    auto it =
        reverse_iterate ? graph->nodes().rbegin() : graph->nodes().begin();  // 根据迭代方向选择起始迭代点
    auto end = reverse_iterate ? graph->nodes().rend() : graph->nodes().end();  // 根据迭代方向选择终止迭代点
    for (; it != end;) {
      if (first) {  // 如果是第一次迭代
        subgraph = SubgraphUtils::createSingletonSubgraph(
            *it, prim::DifferentiableGraph);  // 创建单个节点的子图，并指定为可微图
        it = reverse_iterate ? ++subgraph->reverseIterator()  // 更新迭代器位置
                             : ++subgraph->iterator();
        first = false;  // 标记第一次迭代完成
      }

      SubgraphUtils::mergeNodeIntoSubgraph(*it, subgraph);  // 将当前节点合并到子图中
      it = reverse_iterate ? ++subgraph->reverseIterator()  // 更新迭代器位置
                           : ++subgraph->iterator();
    }

    // Unmerge and compare with original node listing
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    SubgraphUtils::unmergeSubgraph(subgraph);  // 解除子图的合并操作
    EliminateCommonSubexpression(graph);  // 再次进行公共子表达式消除

    std::vector<Node*> newNodes(graph->nodes().begin(), graph->nodes().end());  // 获取更新后的节点列表
    ASSERT_EQ(originalNodes.size(), newNodes.size());  // 断言原始节点列表与更新后的节点列表长度相等
  }
}

TEST(SubgraphUtilsTest, MergeSubgraphs) {  // 定义合并子图测试用例 SubgraphUtilsTest.MergeSubgraphs
  auto graph = std::make_shared<Graph>();  // 创建一个新的图对象
  std::unordered_map<std::string, Value*> parse_map;  // 定义用于解析的映射表
  parseIR(  // 解析输入的 IR 表达式，构建图结构
      R"IR(
graph(%a : Tensor, %b : Tensor, %c : Tensor):
  %x : Tensor = aten::sigmoid(%a)
  %y : Tensor = aten::mul(%a, %b)
  %p : Tensor = aten::div(%c, %b)
  %q1 : Tensor = aten::mul(%p, %a)
  %q2 : Tensor = aten::tanh(%q1)
  %q3 : Tensor = aten::tanh(%q2)
  %q4 : Tensor = aten::tanh(%q3)
  %q5 : Tensor = aten::hardsigmoid(%q4)
  return (%x, %y, %q5))IR",
      &*graph,
      parse_map);

  std::vector<Node*> originalNodes(  // 创建原始节点列表
      graph->nodes().begin(), graph->nodes().end());

  for (bool reverse_merge : {true, false}) {  // 迭代两次，分别为正向和反向合并
    // Merge everything into two adjacent subgraphs
    Node* graph1 = SubgraphUtils::createSingletonSubgraph(
        *graph->nodes().begin(), prim::DifferentiableGraph);  // 创建第一个单节点子图，并指定为可微图
    while (true) {
      Node* next = graph1->next();  // 获取下一个节点
      if (next->kind() == aten::tanh) {  // 如果下一个节点的操作是 tanh
        break;  // 停止合并
      }
      SubgraphUtils::mergeNodeIntoSubgraph(next, graph1);  // 将当前节点合并到第一个子图中
    }
    Node* graph2 = SubgraphUtils::createSingletonSubgraph(
        graph1->next(), prim::DifferentiableGraph);  // 创建第二个单节点子图，并指定为可微图
    while (graph2->next() != *graph->nodes().end()) {  // 当第二个子图的下一个节点不是图的最后一个节点时
      SubgraphUtils::mergeNodeIntoSubgraph(graph2->next(), graph2);  // 将下一个节点合并到第二个子图中
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* subgraph;
    // 如果需要进行反向合并
    if (reverse_merge) {
      // 将 graph2 中的节点合并到 graph1 中
      SubgraphUtils::mergeNodeIntoSubgraph(graph2, graph1);
      // 将 subgraph 指向 graph1
      subgraph = graph1;
    } else {
      // 将 graph1 中的节点合并到 graph2 中
      SubgraphUtils::mergeNodeIntoSubgraph(graph1, graph2);
      // 将 subgraph 指向 graph2
      subgraph = graph2;
    }
    // 定义并运行一个文件检查 lambda 函数，对 subgraph 进行静态分析
    auto run_file_check = [](std::shared_ptr<Graph> graph) {
      // 对 graph 进行静态检查
      graph->lint();
      // 使用 FileCheck 进行检查，验证包含特定操作的图结构
      testing::FileCheck()
          .check("aten::sigmoid")  // 检查是否包含 aten::sigmoid 操作
          ->check("aten::mul")     // 检查是否包含 aten::mul 操作
          ->check("aten::div")     // 检查是否包含 aten::div 操作
          ->check("aten::mul")     // 再次检查是否包含 aten::mul 操作
          ->check_count("aten::tanh", 3)  // 检查 aten::tanh 操作出现次数是否为 3
          ->check("aten::hardsigmoid")    // 检查是否包含 aten::hardsigmoid 操作
          ->run(*graph);  // 运行检查器，对 graph 执行检查
    };
    // 对 subgraph 的子图进行文件检查
    run_file_check(subgraph->g(attr::Subgraph));

    // 取消合并并与原始节点列表进行比较
    SubgraphUtils::unmergeSubgraph(subgraph);  // 取消对 subgraph 的合并操作
    EliminateCommonSubexpression(graph);       // 消除图中的公共子表达式
    run_file_check(graph);  // 对 graph 进行文件检查

    // 将 graph 的节点复制到新节点向量中
    std::vector<Node*> newNodes(graph->nodes().begin(), graph->nodes().end());
    // 断言新节点数量与原始节点数量相等
    ASSERT_EQ(originalNodes.size(), newNodes.size());
  }
TEST(SubgraphUtilsTest, GraphName) {
  // 创建一个共享指针指向 Graph 对象
  auto graph = std::make_shared<Graph>();

  // 创建一个无序映射，用于解析 IR 中的参数
  std::unordered_map<std::string, Value*> parse_map;

  // 调用 parseIR 函数，解析给定的 IR 字符串，并将结果填充到 graph 和 parse_map 中
  parseIR(
      R"IR(
graph(%a : Tensor, %b : Tensor, %c : Tensor):
  %x : Tensor = aten::tanh(%a)
  %y : Tensor = aten::mul(%a, %b)
  %p : Tensor = aten::div(%c, %b)
  %q1 : Tensor = aten::mul(%p, %a)
  %q2 : Tensor = aten::tanh(%q1)
  %q3 : Tensor = aten::tanh(%q2)
  %q4 : Tensor = aten::tanh(%q3)
  %q5 : Tensor = aten::tanh(%q4)
  return (%x, %y, %q5))IR",
      &*graph,
      parse_map);

  // 期望的完整图名称
  std::string ref_full_name = "graph_tanh_mul_div_mul_tanh_tanh_tanh_tanh";

  // 调用 SubgraphUtils::generateNameForGraph 函数，生成图的名称，并检查其与参考名称是否相等
  std::string full_name =
      SubgraphUtils::generateNameForGraph(graph, 80, "graph");
  ASSERT_EQ(full_name, ref_full_name);

  // 调用 SubgraphUtils::generateNameForGraph 函数，生成被截断的图名称，并检查其长度是否小于或等于参考名称的长度
  std::string truncated_name =
      SubgraphUtils::generateNameForGraph(graph, 10, "graph");
  ASSERT_LE(truncated_name.size(), ref_full_name.size());
}

// 结束命名空间 jit
} // namespace jit

// 结束命名空间 torch
} // namespace torch
```