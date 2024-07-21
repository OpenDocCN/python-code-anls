# `.\pytorch\test\cpp\jit\test_op_replacement.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <test/cpp/jit/test_utils.h>  // 包含测试实用工具的头文件
#include <torch/csrc/jit/operator_upgraders/upgraders.h>  // 包含操作升级器的头文件
#include <torch/csrc/jit/operator_upgraders/version_map.h>  // 包含版本映射的头文件
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>  // 包含旧操作替换的头文件
#include <memory>  // 包含智能指针的头文件

namespace torch {
namespace jit {

// 定义一个用于存储升级器的无序映射，键为字符串，值也为字符串
std::unordered_map<std::string, std::string> test_upgraders(
    {{"_test_serialization_subcmul_0_2", R"IR(graph(%self.1 : Tensor,
                                                    %other.1 : Tensor,
                                                    %alpha.1 : Union(float, int)):
                                                %7 : int = prim::Constant[value=1]()
                                                %6 : Tensor = aten::mul(%self.1, %alpha.1) # torch/jit/operator_upgraders.py:18:20
                                                %8 : Tensor = aten::sub(%other.1, %6, %7) # torch/jit/operator_upgraders.py:18:11
                                                return (%8))IR"},
     {"div_Tensor_0_3", R"IR(graph(%self.1 : Tensor,
                                  %other.1 : Tensor):
                            %32 : str = prim::Constant[value="trunc"]()
                            %6 : bool = prim::Constant[value=1]()
                            %4 : bool = aten::is_floating_point(%self.1)
                            %11 : bool = prim::If(%4)
                                block0():
                                    -> (%6)
                                block1():
                                    %9 : bool = aten::is_floating_point(%other.1)
                                    -> (%9)
                            %35 : Tensor = prim::If(%11)
                                block0():
                                    %36 : Tensor = aten::div(%self.1, %other.1)
                                    -> (%36)
                                block1():
                                    %37 : Tensor = aten::div(%self.1, %other.1, %32)
                                    -> (%37)
                            return (%35))IR"}});

// 定义测试用例 OpReplacementTest，测试旧操作替换
TEST(OpReplacementTest, ReplaceDivInSimpleFunction) {
  // 定义一个包含 IR 的字符串表示的计算图
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor):
            %2 : Tensor = aten::add(%0, %1)
            %3 : Tensor  = aten::div(%2, %1)
            return (%3))IR";
  
  auto g = std::make_shared<Graph>();  // 创建一个共享指针指向图对象
  test_only_populate_upgraders(test_upgraders);  // 调用测试工具函数，加载升级器
  torch::jit::parseIR(graph_string, g.get());  // 解析 IR 字符串并加载到图对象 g 中
  g->set_op_version(2);  // 设置操作版本为 2
  ReplaceOldOperatorsWithUpgraders(g);  // 使用升级器替换图中的旧操作
  testing::FileCheck()
      .check("prim::If")  // 检查是否包含 prim::If
      ->check_count("aten::div(%2, %1)", 1, /*exactly=*/true)  // 检查特定模式的操作出现次数
      ->check_count("aten::div(%2, %1, %4)", 1, /*exactly=*/true)  // 检查特定模式的操作出现次数
      ->run(*g);  // 在图 g 上运行检查
}

}  // namespace jit
}  // namespace torch
# 在 OpReplacementTest 测试中，替换简单函数中的两个操作符
TEST(OpReplacementTest, ReplaceTwoOpsInSimpleFunction) {
  # 定义图形的字符串表示，包含两个输入张量 %0 和 %1
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor):
            %2 : Tensor = aten::add(%0, %1)  # 执行张量相加操作 %0 + %1，结果存储在 %2 中
            %3 : Tensor  = aten::div(%2, %1)  # 执行张量相除操作 %2 / %1，结果存储在 %3 中
            %4 : int = prim::Constant[value=1]()  # 创建常量值为 1 的整数，存储在 %4 中
            %5: Tensor = aten::_test_serialization_subcmul(%0, %1, %4)  # 调用自定义函数 _test_serialization_subcmul，传入 %0, %1 和 %4
            return (%3, %5))IR";  # 返回张量 %3 和函数调用的结果 %5

  # 创建一个共享的图对象 g
  auto g = std::make_shared<Graph>();
  
  # 调用特定函数以填充升级器列表 test_upgraders
  test_only_populate_upgraders(test_upgraders);
  
  # 定义一个要添加的升级器条目 test_entry
  UpgraderEntry test_entry{
      3,
      "_test_serialization_subcmul_0_2",
      "aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=2) -> Tensor"};
  
  # 仅添加一个入口到特定函数 "aten::_test_serialization_subcmul"
  test_only_add_entry("aten::_test_serialization_subcmul", test_entry);
  
  # 解析图形字符串 graph_string，并将其加载到图对象 g 中
  torch::jit::parseIR(graph_string, g.get());
  
  # 设置图对象的操作版本为 2
  g->set_op_version(2);
  
  # 替换图中旧的操作符为升级器
  ReplaceOldOperatorsWithUpgraders(g);
  
  # 运行文件检查以验证替换操作
  testing::FileCheck()
      .check("prim::If")  # 检查是否存在 prim::If
      ->check_count("aten::div", 2, /*exactly=*/true)  # 检查 aten::div 操作的确切出现次数为 2
      ->run(*g);  # 在图对象 g 上运行文件检查

  # 仅移除特定函数 "aten::_test_serialization_subcmul" 的入口
  test_only_remove_entry("aten::_test_serialization_subcmul");
  
  # 仅移除升级器列表 test_upgraders 中的所有升级器
  test_only_remove_upgraders(test_upgraders);
}

# 在 OpReplacementTest 测试中，替换嵌套函数中的除法操作符
TEST(OpReplacementTest, ReplaceDivInNestedFunction) {
  # 定义图形的字符串表示，包含三个输入 %0, %1 和 %8
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor,
              %8 : bool):
            %9 : bool = prim::Constant[value=1]()  # 创建常量值为 1 的布尔值，存储在 %9 中
            %7 : bool = prim::If(%8)  # 根据输入 %8 执行条件判断
                block0():  # 条件为 false 时执行的块
                    -> (%9)  # 返回 %9
                block1():  # 条件为 true 时执行的块
                    %2 : Tensor = aten::add(%0, %1)  # 执行张量相加操作 %0 + %1，结果存储在 %2 中
                    %3 : Tensor  = aten::div(%2, %1)  # 执行张量相除操作 %2 / %1，结果存储在 %3 中
                    %4 : Tensor = aten::add(%3, %0)  # 执行张量相加操作 %3 + %0，结果存储在 %4 中
                    %10 : bool = aten::is_floating_point(%4)  # 判断张量 %4 是否为浮点数
                    -> (%10)  # 返回 %10
            return (%7))IR";  # 返回块执行的结果 %7

  # 创建一个共享的图对象 g
  auto g = std::make_shared<Graph>();
  
  # 调用特定函数以填充升级器列表 test_upgraders
  test_only_populate_upgraders(test_upgraders);
  
  # 解析图形字符串 graph_string，并将其加载到图对象 g 中
  torch::jit::parseIR(graph_string, g.get());
  
  # 设置图对象的操作版本为 2
  g->set_op_version(2);
  
  # 替换图中旧的操作符为升级器
  ReplaceOldOperatorsWithUpgraders(g);
  
  # 运行文件检查以验证替换操作
  testing::FileCheck()
      .check("prim::If")  # 检查是否存在 prim::If
      ->check_count("aten::add", 2, false)  # 检查 aten::add 操作的出现次数为 2，忽略顺序
      ->run(*g);  # 在图对象 g 上运行文件检查

  # 运行文件检查以验证替换操作
  testing::FileCheck()
      .check("prim::If")  # 检查是否存在 prim::If
      ->check_count("aten::div", 2, false)  # 检查 aten::div 操作的出现次数为 2，忽略顺序
      ->run(*g);  # 在图对象 g 上运行文件检查
  
  # 仅移除升级器列表 test_upgraders 中的所有升级器
  test_only_remove_upgraders(test_upgraders);
}
TEST(OpReplacementTest, ReplaceTestSubcmulInSimpleFunction) {
  // 定义一个包含IR代码的字符串，表示一个简单的计算图
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor):
            %3 : int = prim::Constant[value=1]()
            %2 : Tensor = aten::_test_serialization_subcmul(%0, %1, %3)
            return (%2))IR";
  
  // 创建一个空的计算图对象
  auto g = std::make_shared<Graph>();

  // 调用测试函数，向升级器列表中添加指定的升级器
  test_only_populate_upgraders(test_upgraders);

  // 定义一个升级器条目对象
  UpgraderEntry test_entry{
      3,
      "_test_serialization_subcmul_0_2",
      "aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=2) -> Tensor"};
  
  // 仅向升级器中添加指定的条目
  test_only_add_entry("aten::_test_serialization_subcmul", test_entry);

  // 解析IR字符串，将其转换为计算图g的表示形式
  torch::jit::parseIR(graph_string, g.get());
  
  // 设置计算图的操作版本为2
  g->set_op_version(2);
  
  // 替换计算图中旧的运算符为升级器
  ReplaceOldOperatorsWithUpgraders(g);
  
  // 使用文件检查工具，验证计算图中"aten::mul"操作的出现次数为1
  testing::FileCheck().check_count("aten::mul", 1, false)->run(*g);

  // 使用文件检查工具，验证计算图中"aten::sub"操作的出现次数为1
  testing::FileCheck().check_count("aten::sub", 1, false)->run(*g);

  // 清除测试用的升级器列表
  test_only_remove_upgraders(test_upgraders);
  
  // 移除指定的升级器条目
  test_only_remove_entry("aten::_test_serialization_subcmul");
}

} // namespace jit
} // namespace torch
```