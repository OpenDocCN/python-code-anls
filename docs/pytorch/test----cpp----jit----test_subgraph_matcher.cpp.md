# `.\pytorch\test\cpp\jit\test_subgraph_matcher.cpp`

```
// 引入 Google Test 的测试框架
#include <gtest/gtest.h>

// 引入自定义的测试工具和 Torch 的子图匹配相关头文件
#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

// Torch 命名空间
namespace torch {
namespace jit {

// 测试用例 SubgraphMatcherTest.Trivial1
TEST(SubgraphMatcherTest, Trivial1) {
  // 创建空的图 graph 和 pattern
  Graph graph, pattern;

  // 解析 graph 的 IR 表示并加载到 graph 中
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  return (%a))IR",
      &graph);

  // 解析 pattern 的 IR 表示并加载到 pattern 中
  parseIR(
      R"IR(
graph(%0):
  %x = a::aaa(%0)
  return (%x))IR",
      &pattern);

  // 使用子图匹配器查找匹配，断言找到的匹配结果非空
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

// 测试用例 SubgraphMatcherTest.Trivial2
TEST(SubgraphMatcherTest, Trivial2) {
  // 创建空的图 graph
  Graph graph;
  // 添加输入节点 g_in
  auto* g_in = graph.addInput();
  // 插入一个 tanh 节点 g_tanh，并将 g_in 作为输入
  auto* g_tanh = graph.insertNode(graph.create(aten::tanh, /*num_outputs =*/1));
  g_tanh->addInput(g_in);
  // 将 g_tanh 的输出注册为 graph 的输出
  graph.registerOutput(g_tanh->output());

  // 创建空的图 pattern
  Graph pattern;
  // 添加输入节点 p_in
  auto* p_in = pattern.addInput();
  // 插入一个 tanh 节点 p_tanh，并将 p_in 作为输入
  auto* p_tanh =
      pattern.insertNode(pattern.create(aten::tanh, /*num_outputs =*/1));
  p_tanh->addInput(p_in);
  // 将 p_tanh 的输出注册为 pattern 的输出
  pattern.registerOutput(p_tanh->output());

  // 使用子图匹配器查找匹配，断言找到的匹配结果数量为 1
  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  // 遍历匹配结果中的每一个匹配 m，进行断言验证
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(p_in) == g_in);
    AT_ASSERT(m.values_map.at(p_tanh->output()) == g_tanh->output());
    AT_ASSERT(m.nodes_map.at(p_tanh) == g_tanh);
  }
}

// 测试用例 SubgraphMatcherTest.Trivial3
TEST(SubgraphMatcherTest, Trivial3) {
  // 创建空的图 graph 和 pattern
  Graph graph, pattern;

  // 解析 graph 的 IR 表示并加载到 graph 中
  parseIR(
      R"IR(
graph(%0):
  %a = a::a(%0)
  %b = a::b(%0)
  %c = a::c(%a, %b)
  return (%c))IR",
      &graph);

  // 解析 pattern 的 IR 表示并加载到 pattern 中
  parseIR(
      R"IR(
graph(%a, %b):
  %c = a::c(%a, %b)
  return (%c))IR",
      &pattern);

  // 使用子图匹配器查找匹配，断言找到的匹配结果非空
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

// 测试用例 SubgraphMatcherTest.Trivial4
TEST(SubgraphMatcherTest, Trivial4) {
  // 创建空的图 graph
  Graph graph;
  // 添加两个输入节点 g_in0 和 g_in1
  auto* g_in0 = graph.addInput();
  auto* g_in1 = graph.addInput();
  // 插入一个乘法节点 g_mul，并将 g_in0 和 g_in1 作为输入
  auto* g_mul = graph.insertNode(graph.create(aten::mul, /*num_outputs =*/1));
  g_mul->addInput(g_in0);
  g_mul->addInput(g_in1);
  // 将 g_mul 的输出注册为 graph 的输出
  graph.registerOutput(g_mul->output());

  // 创建空的图 pattern
  Graph pattern;
  // 添加两个输入节点 p_in0 和 p_in1
  auto* p_in0 = pattern.addInput();
  auto* p_in1 = pattern.addInput();
  // 插入一个乘法节点 p_mul，并将 p_in0 和 p_in1 作为输入
  auto* p_mul =
      pattern.insertNode(pattern.create(aten::mul, /*num_outputs =*/1));
  p_mul->addInput(p_in0);
  p_mul->addInput(p_in1);
  // 将 p_mul 的输出注册为 pattern 的输出
  pattern.registerOutput(p_mul->output());

  // 使用子图匹配器查找匹配，断言找到的匹配结果数量为 1
  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  // 遍历匹配结果中的每一个匹配 m，进行断言验证
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(p_in0) == g_in0);
    AT_ASSERT(m.values_map.at(p_in1) == g_in1);
    AT_ASSERT(m.values_map.at(p_mul->output()) == g_mul->output());
    AT_ASSERT(m.nodes_map.at(p_mul) == g_mul);
  }
}

// 测试用例 SubgraphMatcherTest.Linear1
TEST(SubgraphMatcherTest, Linear1) {
  // 创建空的图 graph 和 pattern
  Graph graph, pattern;

  // 解析 graph 的 IR 表示并加载到 graph 中
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = b::bbb(%a)
  %c = c::ccc(%b)
  %d = d::ddd(%c)
  %a = a::aaa(%0)
  return (%d))IR",
      &graph);

  // 解析 pattern 的 IR 表示并加载到 pattern 中
  parseIR(
      R"IR(
graph(%0):
  %x = b::bbb(%0)
  %y = c::ccc(%x)
  return (%y))IR",
      &pattern);

  // 使用子图匹配器查找匹配，断言找到的匹配结果非空
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

// 关闭命名空间 torch::jit
} // namespace jit
} // namespace torch
TEST(SubgraphMatcherTest, Linear2) {
  // 创建一个空的计算图
  Graph graph;
  // 向计算图中添加一个输入节点，并返回指向该节点的指针
  auto* g_in = graph.addInput();

  // 在计算图中插入一个 tanh 操作节点，并返回指向该节点的指针
  auto* g_tanh = graph.insertNode(graph.create(aten::tanh, /*num_outputs =*/1));
  // 将前面创建的输入节点作为该 tanh 节点的输入
  g_tanh->addInput(g_in);

  // 在计算图中再次插入一个 tanh 操作节点，并返回指向该节点的指针
  auto* g_tanh2 =
      graph.insertNode(graph.create(aten::tanh, /*num_outputs =*/1));
  // 将第一个 tanh 节点的输出作为第二个 tanh 节点的输入
  g_tanh2->addInput(g_tanh->output());

  // 在计算图中注册第二个 tanh 节点的输出作为计算图的输出
  graph.registerOutput(g_tanh2->output());

  // 创建一个空的模式图
  Graph pattern;
  // 向模式图中添加一个输入节点，并返回指向该节点的指针
  auto* p_in = pattern.addInput();

  // 在模式图中插入一个 tanh 操作节点，并返回指向该节点的指针
  auto* p_tanh =
      pattern.insertNode(pattern.create(aten::tanh, /*num_outputs =*/1));
  // 将模式图的输入节点作为该 tanh 节点的输入
  p_tanh->addInput(p_in);

  // 在模式图中再次插入一个 tanh 操作节点，并返回指向该节点的指针
  auto* p_tanh2 =
      pattern.insertNode(pattern.create(aten::tanh, /*num_outputs =*/1));
  // 将第一个 tanh 节点的输出作为第二个 tanh 节点的输入
  p_tanh2->addInput(p_tanh->output());

  // 在模式图中注册第二个 tanh 节点的输出作为模式图的输出
  pattern.registerOutput(p_tanh2->output());

  // 在图形中查找模式匹配，返回匹配结果
  auto matches = findPatternMatches(pattern, graph);
  // 断言匹配结果的大小为1
  AT_ASSERT(matches.size() == 1);
  // 遍历每个匹配，进行断言验证
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(p_in) == g_in);
    AT_ASSERT(m.values_map.at(p_tanh->output()) == g_tanh->output());
    AT_ASSERT(m.values_map.at(p_tanh2->output()) == g_tanh2->output());
    AT_ASSERT(m.nodes_map.at(p_tanh) == g_tanh);
    AT_ASSERT(m.nodes_map.at(p_tanh2) == g_tanh2);
  }
}

/**
 * Test diamond pattern:
 *
 *     ooo
 *      |
 *     aaa
 *    /   \
 *  bbb   ccc
 *     \ /
 *     ddd
 *      |
 *     eee
 */
TEST(SubgraphMatcherTest, Diamond1) {
  // 创建三个空的计算图：graph, pattern1, pattern2
  Graph graph, pattern1, pattern2;
  // 解析给定的 IR 表示的计算图并将其存储在 graph 中
  parseIR(
      R"IR(
graph(%0):
  %o = o::ooo(%0)
  %a = a::aaa(%o)
  %b = b::bbb(%a)
  %c = c::ccc(%a)
  %d = d::ddd(%b, %c)
  %e = e::eee(%d)
  return (%e))IR",
      &graph);

  // 解析给定的 IR 表示的计算图并将其存储在 pattern1 中
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = b::bbb(%a)
  %c = c::ccc(%a)
  %d = d::ddd(%b, %c)
  return (%d))IR",
      &pattern1);
  // 断言模式1在计算图中存在匹配项
  AT_ASSERT(!findPatternMatches(pattern1, graph).empty());

  // 解析给定的 IR 表示的计算图并将其存储在 pattern2 中
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %c = c::ccc(%a)
  %b = b::bbb(%a)
  %d = d::ddd(%b, %c)
  return (%d))IR",
      &pattern2);
  // 断言模式2在计算图中存在匹配项
  AT_ASSERT(!findPatternMatches(pattern2, graph).empty());
}

/**
 * Test diamond pattern:
 *
 *     i0
 *      |
 *    chunk
 *    /   \
 * os[0] os[1]
 *     \ /
 *      *
 *      |
 *      o1
 */
# 定义一个名为 Diamond2 的测试用例，用于测试子图匹配功能
TEST(SubgraphMatcherTest, Diamond2) {
  # 创建一个空图 graph
  Graph graph;
  # 向图中添加一个输入节点，并将其指针保存到变量 g_in 中
  auto* g_in = graph.addInput();

  # 在图中插入一个 ConstantChunk 节点，并设置属性 chunks=2, dim=0
  auto* g_chunk =
      graph.insertNode(graph.create(prim::ConstantChunk, /*num_outputs =*/2));
  g_chunk->i_(attr::chunks, 2)->i_(attr::dim, 0);
  # 将 g_in 作为 g_chunk 的输入
  g_chunk->addInput(g_in);

  # 在图中插入一个乘法节点，并将 ConstantChunk 节点的输出作为输入
  auto* g_mul = graph.insertNode(graph.create(aten::mul, /*num_outputs =*/1));
  g_mul->addInput(g_chunk->outputs()[0]);
  g_mul->addInput(g_chunk->outputs()[1]);
  # 将 g_mul 的输出注册为图的输出
  graph.registerOutput(g_mul->output());

  # 定义一个名为 pattern 的空图
  Graph pattern;
  # 向 pattern 图中添加一个输入节点，并将其指针保存到变量 p_in 中
  auto* p_in = pattern.addInput();
  # 在 pattern 图中插入一个 ConstantChunk 节点，并设置属性 chunks=2, dim=0
  auto* p_chunk = pattern.insertNode(
      pattern.create(prim::ConstantChunk, /*num_outputs =*/2));
  p_chunk->i_(attr::chunks, 2)->i_(attr::dim, 0);
  # 将 p_in 作为 p_chunk 的输入
  p_chunk->addInput(p_in);

  # 在 pattern 图中插入一个乘法节点，并将 ConstantChunk 节点的输出作为输入
  auto* p_mul =
      pattern.insertNode(pattern.create(aten::mul, /*num_outputs =*/1));
  p_mul->addInput(p_chunk->outputs()[0]);
  p_mul->addInput(p_chunk->outputs()[1]);
  # 将 p_mul 的输出注册为图的输出
  pattern.registerOutput(p_mul->output());

  # 使用 findPatternMatches 函数在 graph 图中查找与 pattern 图匹配的子图
  auto matches = findPatternMatches(pattern, graph);
  # 断言找到的匹配数为 1
  AT_ASSERT(matches.size() == 1);
  # 遍历所有匹配结果
  for (const Match& m : matches) {
    # 断言匹配结果中 p_in 对应的值为 g_in
    AT_ASSERT(m.values_map.at(p_in) == g_in);
    # 断言匹配结果中 p_chunk 的第一个输出对应的值为 g_chunk 的第一个输出
    AT_ASSERT(m.values_map.at(p_chunk->outputs()[0]) == g_chunk->outputs()[0]);
    # 断言匹配结果中 p_chunk 的第二个输出对应的值为 g_chunk 的第二个输出
    AT_ASSERT(m.values_map.at(p_chunk->outputs()[1]) == g_chunk->outputs()[1]);
    # 断言匹配结果中 p_mul 的输出对应的值为 g_mul 的输出
    AT_ASSERT(m.values_map.at(p_mul->output()) == g_mul->output());
    # 断言匹配结果中 p_mul 节点对应的指针为 g_mul
    AT_ASSERT(m.nodes_map.at(p_mul) == g_mul);
  }
}

# 定义一个名为 XPattern 的测试用例，用于测试复杂图模式的匹配
TEST(SubgraphMatcherTest, XPattern) {
  # 创建空图 graph 和 pattern
  Graph graph, pattern;
  # 解析输入 IR，并将其加载到 graph 图中
  parseIR(
      R"IR(
graph(%0, %1):
  %b = b::bbb(%0)
  %c = c::ccc(%1)
  %x = x::xxx(%b, %c)
  %e = e::eee(%x)
  %f = f::fff(%x)
  %g = g::ggg(%e, %f)
  return (%g))IR",
      &graph);
  # 解析输入 IR，并将其加载到 pattern 图中
  parseIR(
      R"IR(
graph(%0, %1):
  %b = b::bbb(%0)
  %c = c::ccc(%1)
  %x = x::xxx(%b, %c)
  %e = e::eee(%x)
  %f = f::fff(%x)
  %g = g::ggg(%e, %f)
  return (%g))IR",
      &pattern);
  # 断言找到的匹配结果不为空
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

# 定义一个名为 MultipleMatches 的测试用例，用于测试多个匹配的情况
TEST(SubgraphMatcherTest, MultipleMatches) {
  # 创建空图 graph 和 pattern
  Graph graph, pattern;
  # 解析输入 IR，并将其加载到 graph 图中
  parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  %t3 = a::aaa(%t2)
  %t4 = a::aaa(%t3)
  return (%t4))IR",
      &graph);
  # 解析输入 IR，并将其加载到 pattern 图中
  parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  return (%t1))IR",
      &pattern);
  # 使用 findPatternMatches 函数在 graph 图中查找与 pattern 图匹配的子图
  auto matches = findPatternMatches(pattern, graph);
  # 断言找到的匹配数为 4
  AT_ASSERT(matches.size() == 4);
}

# 定义一个名为 OverlappingMatches 的测试用例，用于测试重叠匹配的情况
TEST(SubgraphMatcherTest, OverlappingMatches) {
  # 创建空图 graph 和 pattern
  Graph graph, pattern;
  # 解析输入 IR，并将其加载到 graph 图中
  parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  %t3 = a::aaa(%t2)
  %t4 = a::aaa(%t3)
  return (%t4))IR",
      &graph);
  # 解析输入 IR，并将其加载到 pattern 图中
  parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  return (%t2))IR",
      &pattern);
  # 使用 findPatternMatches 函数在 graph 图中查找与 pattern 图匹配的子图
  auto matches = findPatternMatches(pattern, graph);
  # 断言找到的匹配数为 3
  AT_ASSERT(matches.size() == 3);
}

# 定义一个名为 MatchInBasicBlocks1 的测试用例，用于测试基本块内匹配的情况
TEST(SubgraphMatcherTest, MatchInBasicBlocks1) {
  # 创建一个空图 graph
  Graph graph;
  # 解析输入 IR，并将其加载到 graph 图中
  parseIR(
      R"IR(
graph(%a, %b, %c):
  %d = aten::mul(%a, %b)
  %x = prim::If(%c)
    block0():
      %x1 = aten::mul(%a, %d)
      -> (%x1)
      )IR",
      &graph);
}
    # 定义一个函数块 block1，进行一些计算操作
    block1():
      # 执行张量的乘法操作，将结果存储在 %x2 变量中
      %x2 = aten::mul(%b, %d)
      -> (%x2)
  return (%x))IR",
      &graph);

  # 确保匹配操作不会跨越基本块边界
  # 创建一个名为 pattern0 的图形对象
  Graph pattern0;
  # 解析所提供的内联汇编代码，并存储在 pattern0 中
  parseIR(
      R"IR(
// 导入的代码库提供的图形处理函数，用于模式匹配
TEST(SubgraphMatcherTest, MatchInBasicBlocks2) {
  // 创建一个空的图形对象
  Graph graph;
  // 解析包含特定IR格式的字符串，填充图形对象
  parseIR(
      R"IR(
graph(%a, %b):
  %x = my::mul(%a, %b)
  %y = my::node_with_subblock()
    block0():
      %z = my::mul(%b, %x)
      -> (%z)
  return (%y))IR",
      &graph);

  // 创建用于匹配的模式图形对象
  Graph pattern0;
  // 解析包含特定IR格式的字符串，填充模式图形对象
  parseIR(
      R"IR(
graph(%x, %y):
  %z = my::mul(%x, %y)
  return (%z))IR",
      &pattern0);
  // 断言模式在图形中的匹配次数为2
  AT_ASSERT(findPatternMatches(pattern0, graph).size() == 2);

  // 创建另一个用于匹配的模式图形对象
  Graph pattern1;
  // 解析包含特定IR格式的字符串，填充模式图形对象
  parseIR(
      R"IR(
graph(%x, %y):
  %u = my::mul(%x, %y)
  %v = my::mul(%y, %u)
  return (%v))IR",
      &pattern1);
  // 断言模式在图形中的匹配次数为0
  AT_ASSERT(findPatternMatches(pattern1, graph).size() == 0);
}
// 定义一个名为 graph 的函数，参数为 %x
graph(%x):
  // 调用 my::op1 函数处理参数 %x，将结果赋给 %y
  %y = my::op1(%x)
  // 调用 my::op2 函数处理参数 %x，将结果赋给 %z
  %z = my::op2(%x)
  // 返回 %y 和 %z 的元组
  return (%y, %z))IR",
      &graph);

  // 解析一个带有子块的 IR
  parseIR(
      R"IR(
graph(%x):
  // 调用 my::node_with_subblock 函数，并将结果赋给 %y
  %y = my::node_with_subblock()
    // 定义 block0 子块
    block0():
      // 调用 my::op 函数处理参数 %x，将结果赋给 %z
      %z = my::op(%x)
      // 返回 %z
      -> (%z)
  // 返回 %y
  return (%y))IR",
      &pattern1);
  // 不支持带有子块的模式
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(findPatternMatches(pattern1, graph));

  // 解析另一个 IR
  parseIR(
      R"IR(
graph(%x):
  // 调用 my::op1 函数处理参数 %x，将结果赋给 %y
  %y = my::op1(%x)
  // 调用 my::op2 函数处理参数 %x，将结果赋给 %z
  %z = my::op2(%x)
  // 返回 %y 和 %z 的元组
  return (%y, %z))IR",
      &pattern2);
  // 不支持多输出模式，因为不是从第一个输出开始遍历整个模式 (`%z = ...` 没有被访问)。
  // 见 subgraph_matcher.h 中的注释 "Multi-output Patterns"。
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(findPatternMatches(pattern2, graph));
}

// 测试 SubgraphMatcher 的多输出模式
TEST(SubgraphMatcherTest, MultiOutput) {
  {
    // 创建图和模式对象
    Graph graph, pattern;
    // 解析一个 IR
    parseIR(
        R"IR(
graph(%0):
  // 调用 a::aaa 函数处理参数 %0，将结果赋给 %a
  %a = a::aaa(%0)
  // 调用 b::bbb 函数处理参数 %a，将结果赋给 %b
  %b = b::bbb(%a)
  // 调用 c::ccc 函数处理参数 %a 和 %b，将结果赋给 %c
  %c = c::ccc(%a, %b)
  // 再次调用 a::aaa 函数处理参数 %c，将结果赋给 %x
  %x = a::aaa(%c)
  // 调用 b::bbb 函数处理参数 %x，将结果赋给 %y
  %y = b::bbb(%x)
  // 调用 d::ddd 函数处理参数 %x 和 %y，将结果赋给 %z
  %z = d::ddd(%x, %y)
  // 返回 %y
  return (%y))IR",
        &graph);
    // 解析另一个 IR 作为模式
    parseIR(
        R"IR(
graph(%0):
  // 调用 a::aaa 函数处理参数 %0，将结果赋给 %a
  %a = a::aaa(%0)
  // 调用 b::bbb 函数处理参数 %a，将结果赋给 %b
  %b = b::bbb(%a)
  // 返回 %b 和 %a 的元组
  return (%b, %a))IR",
        &pattern);
    // 断言在图中找到的模式匹配数量为 2
    AT_ASSERT(findPatternMatches(pattern, graph).size() == 2);
  }
  {
    // 创建图和模式对象
    Graph graph, pattern;
    // 解析一个 IR
    parseIR(
        R"IR(
graph(%0, %1):
  // 调用 a::aaa 函数处理参数 %0 和 %1，将结果分别赋给 %a1 和 %a2
  %a1, %a2 = a::aaa(%0, %1)
  // 调用 b::bbb 函数处理参数 %a1，将结果赋给 %b
  %b = b::bbb(%a1)
  // 调用 c::ccc 函数处理参数 %b，将结果赋给 %c
  %c = c::ccc(%b)

  // 再次调用 a::aaa 函数处理参数 %c 和 %a2，将结果分别赋给 %x1 和 %x2
  %x1, %x2 = a::aaa(%c, %a2)
  // 调用 b::bbb 函数处理参数 %x1，将结果赋给 %y
  %y = b::bbb(%x1)
  // 调用 d::ddd 函数处理参数 %y，将结果赋给 %z
  %z = d::ddd(%y)
  // 返回 %z
  return (%z))IR",
        &graph);
    // 解析另一个 IR 作为模式
    parseIR(
        R"IR(
graph(%0, %1):
  // 调用 a::aaa 函数处理参数 %0 和 %1，将结果分别赋给 %a1 和 %a2
  %a1, %a2 = a::aaa(%0, %1)
  // 调用 b::bbb 函数处理参数 %a1，将结果赋给 %b
  %b = b::bbb(%a1)
  // 返回 %b 和 %a2 的元组
  return (%b, %a2))IR",
        &pattern);
    // 断言在图中找到的模式匹配数量为 2
    AT_ASSERT(findPatternMatches(pattern, graph).size() == 2);
  }
}

} // namespace jit
} // namespace torch
```