# `.\pytorch\test\cpp\jit\test_subgraph_rewriter.cpp`

```
// 引入 Google Test 框架的头文件
#include <gtest/gtest.h>

// 引入用于测试的实用工具头文件
#include <test/cpp/jit/test_utils.h>

// 引入子图匹配器头文件
#include <torch/csrc/jit/ir/subgraph_matcher.h>

// 引入子图重写通行证头文件
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

// 引入文件检查工具头文件
#include <torch/csrc/jit/testing/file_check.h>

// 命名空间声明：torch::jit
namespace torch {
namespace jit {

// 使用测试命名空间
using namespace testing;

// 定义子图重写测试类 SubgraphRewriterTest
TEST(SubgraphRewriterTest, FilterMatch) {
  // 创建一个共享指针指向空图形对象
  auto graph = std::make_shared<Graph>();

  // 解析 IR 文本，并将其构建成图形
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b : int = prim::Constant[value=1]()
  %c = c::ccc(%a, %b)
  return (%c))IR",
      graph.get());

  // 定义模式字符串
  std::string pattern = R"IR(
graph(%a, %b):
  %c = c::ccc(%a, %b)
  return (%c))IR";

  // 创建模式图形对象
  Graph pattern_graph;
  // 定义值映射
  std::unordered_map<std::string, Value*> vmap;

  // 解析模式字符串，并将其构建成模式图形
  parseIR(pattern, &pattern_graph, vmap);

  // 定义一个函数对象，用于检查匹配中 'b' 是否为常数节点
  auto b_is_constant = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto b_node = match_vmap.at(vmap.at("b"))->node();
    return b_node->kind() == prim::Constant;
  };

  // 定义一个函数对象，用于检查匹配中 'b' 是否为整数值为 1
  auto b_is_one = [](const Match& match,
                     const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto b_val = toIValue(match_vmap.at(vmap.at("b")));
    return b_val && b_val->isInt() && b_val->toInt() == 1;
  };

  // 定义一个函数对象，用于检查匹配中 'b' 是否为整数值为 2
  auto b_is_two = [](const Match& match,
                     const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto b_val = toIValue(match_vmap.at(vmap.at("b")));
    return b_val && b_val->isInt() && b_val->toInt() == 2;
  };

  // 定义替换模式字符串
  std::string replacement = R"IR(
graph(%a, %b):
  %d = d::ddd(%a, %b)
  return (%d))IR";

  // 创建子图重写器对象
  SubgraphRewriter rewriter;
  // 注册重写模式
  rewriter.RegisterRewritePattern(pattern, replacement);

  // 测试用例：当 'b' 是常数时，匹配成功
  {
    auto g = graph->copy();
    rewriter.runOnGraph(g, b_is_constant);
    FileCheck().check("d::ddd")->check_not("c::ccc")->run(*g);
  }

  // 测试用例：当 'b' 是常数且值为 1 时，匹配成功
  {
    auto g = graph->copy();
    rewriter.runOnGraph(g, {b_is_constant, b_is_one});
    FileCheck().check("d::ddd")->check_not("c::ccc")->run(*g);
  }

  // 测试用例：当 'b' 是常数但值不为 2 时，匹配失败
  {
    auto g = graph->copy();
    rewriter.runOnGraph(g, {b_is_constant, b_is_two});
    FileCheck().check("c::ccc")->check_not("d::ddd")->run(*g);
  }
}

// 定义另一个子图重写测试类 SubgraphRewriterTest，测试不匹配情况
TEST(SubgraphRewriterTest, FilterNoMatch) {
  // 创建一个共享指针指向空图形对象
  auto graph = std::make_shared<Graph>();
  
  // 解析 IR 文本，并将其构建成图形
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = prim::Constant[value=1]()
  %c = c::ccc(%a, %b)
  return (%c))IR",
      graph.get());

  // 定义模式字符串
  std::string pattern = R"IR(
graph(%a, %b):
  %c = c::ccc(%a, %b)
  return (%c))IR";

  // 创建模式图形对象
  Graph pattern_graph;
  // 定义值映射
  std::unordered_map<std::string, Value*> vmap;

  // 解析模式字符串，并将其构建成模式图形
  parseIR(pattern, &pattern_graph, vmap);

  // 定义一个过滤器函数对象，该过滤器函数没有具体实现
    // 从 match_vmap 中获取与 vmap 中 "b" 对应的值，然后获取其 node()
    auto b_node = match_vmap.at(vmap.at("b"))->node();
    // 如果 b_node 的类型不是 prim::Assign，则不会匹配，因此不会执行重写
    // 返回 b_node 的类型是否为 prim::Assign
    return b_node->kind() == prim::Assign;
  };

  // 创建一个包含 IR（Intermediate Representation）代码的字符串作为替换
  std::string replacement = R"IR(
// 定义测试函数 SubgraphRewriterTest，测试多输出情况
TEST(SubgraphRewriterTest, MultiOutput) {
  {
    // 创建一个新的图形对象 graph
    auto graph = std::make_shared<Graph>();

    // 解析以下 IR，填充到图形对象中
    // 这段 IR 描述了一个包含多个输出的图形模式重写示例
    parseIR(
        R"IR(
graph(%0, %1):
  %a1, %a2 = a::aaa(%0, %1)
  %b = b::bbb(%a1)
  %c = c::ccc(%b)

  %x1, %x2 = a::aaa(%c, %a2)
  %y = b::bbb(%x1)
  %z = d::ddd(%y)
  return (%z))IR",
        graph.get());

    // 定义要匹配和替换的模式和替换字符串
    std::string pattern = R"IR(
graph(%0, %1):
  %a1, %a2 = a::aaa(%0, %1)
  %b = b::bbb(%a1)
  return (%b, %a2))IR";

    std::string replacement = R"IR(
graph(%a, %b):
  %x, %y = ab::ababab(%a, %b)
  return (%x, %y))IR";

    // 创建重写器对象
    SubgraphRewriter rewriter;
    // 注册模式和替换规则
    rewriter.RegisterRewritePattern(pattern, replacement);

    // 复制当前图形对象，并在其上运行重写器
    auto g = graph->copy();
    rewriter.runOnGraph(g);
    // 检查替换结果
    FileCheck().check("ab::ababab")->check("ab::ababab")->run(*g);
  }

  // ...以下两个相似的测试案例略
}
    graph(%a, %c):
      %d, %b = db::fused(%a, %c)
      return (%d, %b))IR";


    # 定义一个名为 graph 的函数，接受参数 %a 和 %c
    # 调用 db::fused 函数，将 %a 和 %c 作为参数传入，得到返回值 %d 和 %b
    # 返回一个包含 %d 和 %b 的元组
    return (%d, %b))IR";



    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(pattern, replacement);


    # 创建 SubgraphRewriter 对象 rewriter
    # 注册重写模式，使用给定的 pattern 和 replacement
    rewriter.RegisterRewritePattern(pattern, replacement);



    auto g = graph->copy();
    rewriter.runOnGraph(g);


    # 复制输入的 graph，并将其赋值给变量 g
    # 在复制的图 g 上运行重写器 rewriter
    rewriter.runOnGraph(g);



    // We should not perform the replacement on the given graph due to data
    // dependency constraints: the output %b is used in %e, which precedes one
    // def of the input %c.


    // 由于数据依赖约束，我们不应该在给定的图上执行替换操作：
    // 输出 %b 在 %e 中被使用，而 %e 在输入 %c 的定义之前被使用。



    FileCheck().check_not("db::fused")->run(*g);
  }


    // 使用 FileCheck 对象检查，在处理后的图 g 上，不应包含字符串 "db::fused"
    FileCheck().check_not("db::fused")->run(*g);
  }
TEST(SubgraphRewriterTest, OutputType) {
  // 定义匹配模式字符串，使用原始字符串字面量标识符R"IR()"，允许多行字符串
  std::string pattern = R"IR(
graph(%a, %b):
  %c = c::ccc(%a, %b)
  return (%c))IR";
  // 创建图对象
  Graph pattern_graph;
  // 创建值映射的无序映射表
  std::unordered_map<std::string, Value*> vmap;

  // 解析匹配模式字符串为图形表示，填充值映射
  parseIR(pattern, &pattern_graph, vmap);

  // 定义lambda函数b_is_constant，用于检查匹配是否为常量b
  auto b_is_constant = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto b_node = match_vmap.at(vmap.at("b"))->node();
    return b_node->kind() == prim::Constant;
  };

  // 定义替换模式字符串，使用原始字符串字面量标识符R"IR()"
  std::string replacement = R"IR(
graph(%a, %b):
  %d = d::ddd(%a, %b)
  return (%d))IR";

  // 创建子图重写器对象
  SubgraphRewriter rewriter;
  // 注册替换模式
  rewriter.RegisterRewritePattern(pattern, replacement);

  {
    // 创建共享指针指向新的图对象
    auto graph = std::make_shared<Graph>();

    // 解析IR字符串为图形表示，使用传入的图对象填充
    parseIR(
        R"IR(
  graph(%0):
    %a : Float(10, 20) = a::aaa(%0)
    %b : int = prim::Constant[value=1]()
    %c : Float(10, 20) = c::ccc(%a, %b)
    return (%c))IR",
        graph.get());

    // 在图上运行子图重写器，应用b_is_constant函数进行匹配
    rewriter.runOnGraph(graph, b_is_constant);

    // 使用FileCheck对象验证输出，检查替换结果是否符合预期
    FileCheck()
        .check("Float(10, 20) = d::ddd")
        ->check_not("c::ccc")
        ->run(*graph);
  }

  {
    // 创建共享指针指向新的图对象
    auto graph = std::make_shared<Graph>();

    // 解析IR字符串为图形表示，使用传入的图对象填充
    parseIR(
        R"IR(
  graph(%0):
    %a = a::aaa(%0)
    %b : int = prim::Constant[value=1]()
    %c = c::ccc(%a, %b)
    return (%c))IR",
        graph.get());

    // 在图上运行子图重写器，应用b_is_constant函数进行匹配
    rewriter.runOnGraph(graph, b_is_constant);

    // 使用FileCheck对象验证输出，检查替换结果是否符合预期
    FileCheck().check("Tensor = d::ddd")->check_not("c::ccc")->run(*graph);
  }
}

} // namespace jit
} // namespace torch
```