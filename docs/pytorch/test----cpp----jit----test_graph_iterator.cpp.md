# `.\pytorch\test\cpp\jit\test_graph_iterator.cpp`

```
// 包含必要的头文件：iostream 用于标准输入输出操作，sstream 用于字符串流操作，string 用于字符串操作
// gtest.h 是 Google 测试框架的头文件，用于编写和运行 C++ 单元测试
#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

// 引入测试框架的辅助函数和 Torch 相关的头文件
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/torch.h>

// 命名空间 torch::jit 下的声明
namespace torch {
namespace jit {

/**
 * 反转一个无序映射。
 */
template <typename K, typename V>
std::unordered_map<V, K> invert_map(std::unordered_map<K, V>& map) {
  // 创建一个空的反转映射表
  std::unordered_map<V, K> inverted;
  // 使用 lambda 表达式遍历输入的映射表，将键值对反转后插入 inverted 中
  std::for_each(map.begin(), map.end(), [&inverted](const std::pair<K, V>& p) {
    inverted.insert(std::make_pair(p.second, p.first));
  });
  // 返回反转后的映射表
  return inverted;
}

/**
 * 使用 DepthFirstGraphNodeIterator 遍历图形，并返回包含原始名称的数组。
 */
std::vector<std::string> traverse_depth_first(
    std::string graph_string,   // 输入的图形字符串
    int max_count = 100) {      // 最大遍历节点数，默认为 100
  auto graph = std::make_shared<Graph>();   // 创建一个共享指针指向 Graph 对象
  std::unordered_map<std::string, Value*> vmap;  // 创建一个空的字符串到 Value 指针的映射表
  torch::jit::parseIR(graph_string, graph.get(), vmap);   // 解析输入的图形字符串并填充 Graph 和 vmap
  auto get_name = invert_map(vmap);   // 反转 vmap，得到一个从 Value 指针到字符串的映射表

  std::vector<std::string> result;    // 创建一个空的字符串数组，用于存储遍历结果
  DepthFirstGraphNodeIterator graph_it(graph);   // 创建 DepthFirstGraphNodeIterator 对象并绑定到 graph
  Node* node = graph_it.next();   // 获取第一个节点
  int count = 0;    // 计数器初始化为 0
  while (node && count < max_count) {   // 循环遍历节点直到达到最大节点数或者节点为空
    std::stringstream buffer;   // 创建一个字符串流 buffer
    std::vector<const torch::jit::Node*> vec;   // 创建一个空的 Node 指针数组
    node->print(buffer, 0, &vec, false, true, true, false);   // 将节点信息打印到 buffer
    result.push_back(buffer.str());   // 将 buffer 转换为字符串并添加到 result 数组中
    node = graph_it.next();   // 获取下一个节点
    ++count;   // 计数器加一
  }
  // 返回遍历结果数组
  return result;
}

/** 
 * 检查迭代顺序是否与期望的顺序匹配。
 */
void assert_ordering(
    std::vector<std::string> actual,   // 实际的遍历结果数组
    std::initializer_list<std::string> expected_list) {   // 期望的顺序列表
  auto expected = std::vector<std::string>(expected_list);   // 将期望的顺序列表转换为字符串数组
  ASSERT_EQ(expected.size(), actual.size())   // 断言实际结果和期望结果数组长度相等
      << "Got " << actual.size() << " elements (" << actual << ")"
      << " expected " << expected.size() << " elements (" << expected << ")";
  for (unsigned i = 0; i < expected.size(); i++) {   // 循环遍历每个位置
    ASSERT_EQ(expected[i], actual[i])   // 断言实际结果和期望结果在当前位置上相等
        << "Difference at index " << i << " in " << actual << " (expected "
        << actual << ")";
  }
}

// 定义单元测试类 GraphIteratorTest，测试深度优先图形遍历
TEST(GraphIteratorTest, ConstantReturnGraph) {
  // 定义一个常量返回图形的字符串表示
  const auto graph_string = R"IR(
      graph():
        %1 : int = prim::Constant[value=0]()
        return (%1))IR";
  auto graph = std::make_shared<Graph>();   // 创建一个共享指针指向 Graph 对象
  torch::jit::parseIR(graph_string, graph.get());   // 解析图形字符串并填充 Graph 对象
  DepthFirstGraphNodeIterator graph_it(graph);   // 创建 DepthFirstGraphNodeIterator 对象并绑定到 graph
  ASSERT_EQ(graph_it.next()->kind(), prim::Constant);   // 断言第一个节点的类型是 prim::Constant
  ASSERT_EQ(graph_it.next(), nullptr);   // 断言下一个节点为空
}

// 定义单元测试类 GraphIteratorTest，测试带有参数的图形
TEST(GraphIteratorTest, GraphWithParameters) {
  // 定义一个带有参数的图形的字符串表示
  const auto graph_string = R"IR(
      graph(%0 : Double(2)):
        %1 : int = prim::Constant[value=0]()
        return (%0))IR";
  auto ordering = traverse_depth_first(graph_string);   // 使用深度优先遍历获取遍历顺序
  assert_ordering(ordering, {"%1 : int = prim::Constant[value=0]()"});   // 断言遍历顺序符合预期
}

// 定义单元测试类 GraphIteratorTest，测试带有 If 语句的图形
TEST(GraphIteratorTest, GraphWithIf) {
  // 定义一个带有 If 语句的图形的字符串表示
  const auto graph_string = R"IR(
      // 此处省略部分内容
// 定义名为 graph 的函数，参数为一个 Tensor 类型的变量 %a
graph(%a : Tensor):
  // 创建一个常量整数 %a，其值为 30
  %a : int = prim::Constant[value=30]()
  // 创建一个常量整数 %b，其值为 10
  %b : int = prim::Constant[value=10]()
  // 对 %a 进行布尔类型转换，结果存储在 %c 中
  %c : bool = aten::Bool(%a)
  // 根据 %c 的值进行条件判断
  %d : int = prim::If(%c)
    block0():  // 如果条件为真，则执行 block0
      -> (%a)  // 返回 %a 的值
    block1():  // 如果条件为假，则执行 block1
      -> (%b)  // 返回 %b 的值
  // 创建一个常量整数 %e，其值为 20
  %e : int = prim::Constant[value=20]()
  // 返回条件判断后的结果 %d
  return (%d)
)IR";
  // 对 graph_string 进行深度优先遍历，生成 ordering
  auto ordering = traverse_depth_first(graph_string);
  // 断言 ordering 是否与预期的顺序匹配
  assert_ordering(
      ordering,
      {"%1 : int = prim::Constant[value=30]()",
       "%2 : int = prim::Constant[value=10]()",
       "%3 : bool = aten::Bool(%1)",
       "%4 : int = prim::If(%3)",
       "%5 : int = prim::Constant[value=20]()"});
}

// 测试带有嵌套条件的图形
TEST(GraphIteratorTest, GraphWithNestedIf) {
  // 定义一个名为 graph 的函数，带有两个 Tensor 类型的参数 %a.1 和 %b.1
  const auto graph_string = R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  // 创建一个常量整数 %2，其值为 10
  %2 : int = prim::Constant[value=10]()
  // 创建一个常量整数 %3，其值为 20
  %3 : int = prim::Constant[value=20]()
  // 创建一个常量整数 %4，其值为 30
  %4 : int = prim::Constant[value=30]()
  // 创建一个常量整数 %5，其值为 40
  %5 : int = prim::Constant[value=40]()
  // 对 %a.1 进行布尔类型转换，结果存储在 %6 中
  %6 : bool = aten::Bool(%a.1)
  // 根据 %6 的值进行条件判断
  %7 : int = prim::If(%6)
    block0():  // 如果条件为真，则执行 block0
      // 对 %b.1 进行布尔类型转换，结果存储在 %8 中
      %8 : bool = aten::Bool(%b.1)
      // 根据 %8 的值进行条件判断
      %9 : int = prim::If(%8)
        block0():  // 如果条件为真，则执行 block0
          -> (%2)  // 返回 %2 的值
        block1():  // 如果条件为假，则执行 block1
          -> (%3)  // 返回 %3 的值
      -> (%9)  // 返回条件判断后的结果 %9
    block1():  // 如果条件为假，则执行 block1
      // 对 %b.1 进行布尔类型转换，结果存储在 %10 中
      %10 : bool = aten::Bool(%b.1)
      // 根据 %10 的值进行条件判断
      %11 : int = prim::If(%10)
        block0():  // 如果条件为真，则执行 block0
          -> (%4)  // 返回 %4 的值
        block1():  // 如果条件为假，则执行 block1
          -> (%5)  // 返回 %5 的值
      -> (%11)  // 返回条件判断后的结果 %11
  // 对 %b.1 进行布尔类型转换，结果存储在 %8 中
  %8 : bool = aten::Bool(%b.1)
  // 根据 %8 的值进行条件判断
  %9 : int = prim::If(%8)
    block0():  // 如果条件为真，则执行 block0
      -> (%2)  // 返回 %2 的值
    block1():  // 如果条件为假，则执行 block1
      -> (%3)  // 返回 %3 的值
  // 对 %b.1 进行布尔类型转换，结果存储在 %10 中
  %10 : bool = aten::Bool(%b.1)
  // 根据 %10 的值进行条件判断
  %11 : int = prim::If(%10)
    block0():  // 如果条件为真，则执行 block0
      -> (%4)  // 返回 %4 的值
    block1():  // 如果条件为假，则执行 block1
      -> (%5)  // 返回 %5 的值
  // 返回条件判断后的结果 %7
  return (%7)
)IR";
  // 对 graph_string 进行深度优先遍历，生成 ordering
  auto ordering = traverse_depth_first(graph_string);
  // 断言 ordering 是否与预期的顺序匹配
  assert_ordering(
      ordering,
      {"%2 : int = prim::Constant[value=10]()",
       "%3 : int = prim::Constant[value=20]()",
       "%4 : int = prim::Constant[value=30]()",
       "%5 : int = prim::Constant[value=40]()",
       "%6 : bool = aten::Bool(%a.1)",
       "%7 : int = prim::If(%6)",
       "%8 : bool = aten::Bool(%b.1)",
       "%9 : int = prim::If(%8)",
       "%10 : bool = aten::Bool(%b.1)",
       "%11 : int = prim::If(%10)",
       "%12 : bool = aten::Bool(%b.1)",
       "%13 : int = prim::If(%12)",
       "%14 : bool = aten::Bool(%b.1)",
       "%15 : int = prim::If(%14)"});
}

// 测试带有循环的图形
TEST(GraphIteratorTest, GraphWithLoop) {
  // 定义一个名为 graph 的函数，带有一个 Tensor 类型的参数 %a.1
  const auto graph_string = R"IR(
graph(%a.1 : Tensor):
  // 创建一个常量布尔 %1，其值为 1
  %1 : bool = prim::Constant[value=1]()
  // 创建一个常量整数 %2，其值为 10
  %2 : int = prim::Constant[value=10]()
  // 创建一个常量整数 %3，其值为 1
  %3 : int = prim::Constant[value=1]()
  // 创建一个循环，迭代次数由 %2 决定，循环体内使用 %a.1 和 %3 进行加法操作
  %4 : Tensor = prim::Loop(%2, %1, %a.1)
    block0(%i : int, %b.9 : Tensor):
      %5 : Tensor = aten::add_(%b.9, %3, %3)
      -> (%1, %5)
  // 创建一个循环，迭代次数由 %2 决定，循环体内返回结果 %4
  %6 : Tensor = prim::Loop(%2, %1, %a.1)
    block0(%i : int, %b.9 : Tensor):
      -> (%1, %4)
  // 返回循环结束后的结果 %6
  return (%6)
// 结束命名空间 `torch`，进入 `jit` 命名空间
namespace torch {
namespace jit {

// 定义一个 C++ 字符串 `graph_string`，包含一些神经网络图的描述信息
auto graph_string = R"IR";
  
// 对图进行深度优先遍历，返回遍历顺序的排序结果
auto ordering = traverse_depth_first(graph_string);

// 断言排序结果 `ordering` 符合预期的顺序
assert_ordering(
    ordering,
    {"%1 : bool = prim::Constant[value=1]()",
     "%2 : int = prim::Constant[value=10]()",
     "%3 : int = prim::Constant[value=1]()",
     "%4 : Tensor = prim::Loop(%2, %1, %a.1)",
     "%7 : Tensor = aten::add_(%b.10, %3, %3)",
     "%8 : Tensor = prim::Loop(%2, %1, %a.1)"});

} // namespace jit
} // namespace torch
```