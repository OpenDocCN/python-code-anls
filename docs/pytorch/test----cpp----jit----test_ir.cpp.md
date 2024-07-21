# `.\pytorch\test\cpp\jit\test_ir.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <test/cpp/jit/test_utils.h>  // 包含测试工具的头文件
#include <torch/csrc/jit/ir/irparser.h>  // 包含解析 IR 的头文件

namespace torch {
namespace jit {

TEST(IRTest, Attributes) {  // 定义 IR 属性测试的单元测试
  Graph g;  // 创建一个图对象 g

  auto one = attr::alpha;  // 设置一个属性 one
  auto two = attr::device;  // 设置一个属性 two
  auto three = attr::end;  // 设置一个属性 three
  auto four = attr::perm;  // 设置一个属性 four

  Node* n = g.create(Symbol::fromQualString("foo::bar"));  // 在图 g 中创建一个名为 "foo::bar" 的节点 n
  Node& attr = *n;  // 引用节点 n 作为 attr

  attr.f_(one, 3.4)->i_(two, 5)->s_(three, "what");  // 设置节点 attr 的属性：float 类型的 one 为 3.4，整数类型的 two 为 5，字符串类型的 three 为 "what"

  ASSERT_EQ(attr.f(one), 3.4);  // 断言节点 attr 的属性 one 的 float 值为 3.4
  ASSERT_EQ(attr.s(three), "what");  // 断言节点 attr 的属性 three 的字符串值为 "what"
  ASSERT_EQ(attr.i(two), 5);  // 断言节点 attr 的属性 two 的整数值为 5

  attr.s_(one, "no");  // 将节点 attr 的属性 one 的字符串值设置为 "no"
  ASSERT_EQ(attr.s(one), "no");  // 断言节点 attr 的属性 one 的字符串值为 "no"

  ASSERT_TRUE(attr.hasAttribute(three));  // 断言节点 attr 是否具有属性 three
  ASSERT_TRUE(!attr.hasAttribute(four));  // 断言节点 attr 是否不具有属性 four

  attr.ss_(two, {"hi", "now"});  // 将节点 attr 的属性 two 的字符串集合设置为 {"hi", "now"}
  ASSERT_EQ(attr.ss(two).at(1), "now");  // 断言节点 attr 的属性 two 的字符串集合中索引为 1 的元素为 "now"

  Node* n2 = g.create(Symbol::fromQualString("foo::baz"));  // 在图 g 中创建一个名为 "foo::baz" 的节点 n2
  Node& attr2 = *n2;  // 引用节点 n2 作为 attr2

  attr2.copyAttributes(attr);  // 复制节点 attr 的所有属性给节点 attr2
  ASSERT_EQ(attr2.s(one), "no");  // 断言节点 attr2 的属性 one 的字符串值与节点 attr 的相同

  attr2.f_(one, 5);  // 将节点 attr2 的属性 one 的 float 值设置为 5
  ASSERT_EQ(attr.s(one), "no");  // 断言节点 attr 的属性 one 的字符串值仍为 "no"
  ASSERT_EQ(attr2.f(one), 5);  // 断言节点 attr2 的属性 one 的 float 值为 5
}

TEST(IRTest, Blocks) {  // 定义 IR 块测试的单元测试
  auto g = std::make_shared<Graph>();  // 创建一个共享指针指向的图对象 g

  const auto graph_string = R"IR(
    graph(%a : Tensor,
          %b : Tensor,
          %c : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::add(%a, %b, %2)
      %5 : Tensor = prim::If(%c)
        block0():
          %6 : int = prim::Constant[value=1]()
          %7 : Tensor = aten::add(%3, %3, %6)
          -> (%7)
        block1():
          %8 : int = prim::Constant[value=1]()
          %9 : Tensor = aten::add(%b, %3, %8)
          %10 : int = prim::Constant[value=1]()
          %11 : Tensor = aten::add(%9, %3, %10)
          -> (%11)
      %12 : int = prim::Constant[value=1]()
      %13 : Tensor = aten::add(%5, %3, %12)
      return (%13))IR";

  torch::jit::parseIR(graph_string, g.get());  // 解析 IR 字符串并将结果存入图对象 g

  g->lint();  // 检查图 g 的一致性

  testing::FileCheck()
      .check("add")  // 检查是否包含 "add"
      ->check("prim::If")  // 检查是否包含 "prim::If"
      ->check("block0")  // 检查是否包含 "block0"
      ->check("aten::add")  // 检查是否包含 "aten::add"
      ->check("block1")  // 检查是否包含 "block1"
      ->check_count("aten::add", 3)  // 检查 "aten::add" 的出现次数为 3
      ->run(*g);  // 在图 g 上运行测试文件检查

  // Removes block0 of the conditional
  for (auto* node : g->block()->nodes()) {  // 遍历图 g 的节点
    if (node->kind() == prim::If) {  // 如果节点的类型为 prim::If
      node->eraseBlock(0);  // 删除条件语句的 block0
      break;
    }
  }

  testing::FileCheck()
      .check("add")  // 检查是否包含 "add"
      ->check("prim::If")  // 检查是否包含 "prim::If"
      ->check("block0")  // 检查是否包含 "block0"
      ->check_not("block")  // 检查是否不包含 "block"
      ->run(*g);  // 在图 g 上运行测试文件检查

  g->lint();  // 再次检查图 g 的一致性

  // test recursive copy of blocks works
  auto g2 = g->copy();  // 复制图 g 到图 g2
  testing::FileCheck()
      .check("add")  // 检查是否包含 "add"
      ->check("prim::If")  // 检查是否包含 "prim::If"
      ->check("block0")  // 检查是否包含 "block0"
      ->check_not("block")  // 检查是否不包含 "block"
      ->run(*g2);  // 在图 g2 上运行测试文件检查
}

TEST(IRTest, CommonAncestor) {  // 定义 IR 公共祖先测试的单元测试
  std::string input_str = R"(
graph(%x : Tensor,
      %a.1 : bool,
      %b.1 : bool,
      %c.1 : bool):
  %4 : int = prim::If(%a.1)
    block0():
      %5 : int = prim::If(%b.1)
        block0():
          %6 : int = prim::Constant[value=2]()
          -> (%6)
        block1():
          %7 : int = prim::Constant[value=3]()
          -> (%7)
      -> (%5)
)";
  // 输入的 IR 字符串

  // 注意：由于篇幅原因，这里没有完全注释所有的代码，但保证每个单元测试的核心功能都有注释
}
    def block1():
        # 定义一个条件分支，根据 %c.1 的值进行判断
        %8 : int = prim::If(%c.1)
            block0():
                # 如果条件为真，返回常量值 4
                %9 : int = prim::Constant[value=4]()
                -> (%9)
            block1():
                # 如果条件为假，返回常量值 5
                %10 : int = prim::Constant[value=5]()
                -> (%10)
        # 返回条件分支的结果
        -> (%8)
    # 返回结果 %4，但上述代码未定义 %4，可能为整体代码块的一部分
    return (%4)
)";
  
  // 创建一个空的 Torch JIT 图形对象
  torch::jit::Graph g;
  // 创建一个映射，用于将变量名映射到 Torch JIT 值的指针
  std::unordered_map<std::string, torch::jit::Value*> name_to_value;
  // 解析输入的 IR 字符串，将解析结果填充到图形对象和映射中
  torch::jit::parseIR(input_str, &g, name_to_value);

  // 定义要测试的值的名称列表
  std::vector<std::string> value_names{"6", "7", "9", "10"};
  // 将值的名称列表转换为无序集合，以便进行快速查找
  std::unordered_set<std::string> value_names_set(
      value_names.begin(), value_names.end());

  /* clang-format off */
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  // 定义一个参考的块图，表示两个值之间的关系
  int ref_blocks_from_graph[4][4] = {
    /* (6, 6), (6, 7), (6, 9), (6, 10) */
    {   2,     1,      0,      0        },
    /* (7, 6), (7, 7), (7, 9), (7, 10) */
    {   1,     2,      0,      0        },
    /* (9, 6), (9, 7), (9, 9), (9, 10) */
    {   0,     0,      2,      1,       },
    /* (10, 6),(10, 7),(10, 9),(10, 10) */
    {   0,     0,      1,      2        }
  };
  /* clang-format on */

  // 遍历要测试的值的名称列表
  for (size_t i = 0; i < value_names.size(); ++i) {
    // 获取第一个值的 Torch JIT 对象指针
    torch::jit::Value* i_val = name_to_value[value_names[i]];
    // 再次遍历要测试的值的名称列表
    for (size_t j = 0; j < value_names.size(); ++j) {
      // 获取第二个值的 Torch JIT 对象指针
      torch::jit::Value* j_val = name_to_value[value_names[j]];
      // 查找两个值的节点的共同祖先块
      torch::jit::Block* common_ancestor =
          i_val->node()->findCommonAncestorBlockWith(j_val->node());
      // 获取共同祖先块中从图形块到参数节点块的距离
      int blocks_from_graph_block =
          common_ancestor->param_node()->blocksFromGraphBlock();
      // 断言共同祖先块的距离与参考块图中的预期值相等
      ASSERT_EQ(blocks_from_graph_block, ref_blocks_from_graph[i][j]);
    }
  }
}
TEST(IRTest, OperatorMap) {
  // 创建一个 OperatorMap 实例
  OperatorMap<int> op_map;
  // 定义多个操作符文字描述
  const char* literal1 =
      "aten::dropout(Tensor input, float p, bool train) -> Tensor";
  const char* literal2 =
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor";
  const char* literal3 =
      "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor";
  const char* literal4 =
      "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor";
  const char* literal5 =
      "aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor";
  const char* literal6 =
      "aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor";
  // 获取每个操作符对应的 Operator 实例
  std::shared_ptr<Operator> op1 = getOperatorForLiteral(literal1);
  std::shared_ptr<Operator> op2 = getOperatorForLiteral(literal2);
  std::shared_ptr<Operator> op3 = getOperatorForLiteral(literal3);
  std::shared_ptr<Operator> op4 = getOperatorForLiteral(literal4);
  std::shared_ptr<Operator> op5 = getOperatorForLiteral(literal5);
  std::shared_ptr<Operator> op6 = getOperatorForLiteral(literal6);
  // 将 Operator 实例插入到 OperatorMap 中，附带一个整数值
  op_map.insert(op1, 1);
  op_map.insert({{op2, 2}, {op3, 3}});
  op_map.insert({{op4, 4}, {op5, 5}});
  op_map.insert(op6, 6);
  // 断言各个操作符是否存在于 OperatorMap 中
  ASSERT_TRUE(op_map.contains(*op1));
  ASSERT_TRUE(op_map.contains(*op2));
  ASSERT_TRUE(op_map.contains(*op3));
  ASSERT_TRUE(op_map.contains(*op4));
  ASSERT_TRUE(op_map.contains(*op5));
  ASSERT_TRUE(op_map.contains(*op6));
  // 从 OperatorMap 中移除部分 Operator 实例
  op_map.erase(op6);
  op_map.erase(op3);
  op_map.erase(op1);
  // 再次断言被移除的操作符是否不在 OperatorMap 中
  ASSERT_FALSE(op_map.contains(*op1));
  ASSERT_FALSE(op_map.contains(*op3));
  ASSERT_FALSE(op_map.contains(*op6));
  // 重新插入一个操作符，并再次断言其存在于 OperatorMap 中
  op_map.insert(op1, 1);
  ASSERT_TRUE(op_map.contains(*op1));
  // 查找操作符对应的整数值，并断言其存在
  std::optional<int> o1 = op_map.find(*op1);
  ASSERT_TRUE(o1.has_value());
  std::optional<int> o2 = op_map.find(*op2);
  ASSERT_TRUE(o2.has_value());
  std::optional<int> o3 = op_map.find(*op3);
  ASSERT_FALSE(o3.has_value());
  std::optional<int> o4 = op_map.find(*op4);
  ASSERT_TRUE(o4.has_value());
  std::optional<int> o5 = op_map.find(*op5);
  ASSERT_TRUE(o5.has_value());
  std::optional<int> o6 = op_map.find(*op6);
  ASSERT_FALSE(o6.has_value());
}

} // namespace jit
} // namespace torch
```