# `.\pytorch\test\cpp\jit\test_constant_pooling.cpp`

```py
#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {

// 定义 ConstantPoolingTest 类中的测试案例 Int
TEST(ConstantPoolingTest, Int) {
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析输入的 IR 表达式并将其添加到图形中
  parseIR(
      R"IR(
graph():
  %8 : int = prim::Constant[value=1]()   // 创建一个整数常量节点 %8，值为 1
  %10 : int = prim::Constant[value=1]()  // 创建一个整数常量节点 %10，值为 1
  return (%8, %10)                      // 返回节点 %8 和 %10
      )IR",
      &*graph);
  // 对图形进行常量池优化
  ConstantPooling(graph);
  // 运行文件检查，验证是否有正好一个 prim::Constant 节点
  testing::FileCheck()
      .check_count("prim::Constant", 1, /*exactly*/ true)
      ->run(*graph);
}

// 定义 ConstantPoolingTest 类中的测试案例 PoolingAcrossBlocks
TEST(ConstantPoolingTest, PoolingAcrossBlocks) {
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析输入的 IR 表达式并将其添加到图形中
  parseIR(
      R"IR(
graph(%cond : Tensor):
  %a : str = prim::Constant[value="bcd"]()      // 创建一个字符串常量节点 %a，值为 "bcd"
  %3 : bool = aten::Bool(%cond)
  %b : str = prim::If(%3)
    block0():
      %b.1 : str = prim::Constant[value="abc"]()  // 在 block0 中创建字符串常量节点 %b.1，值为 "abc"
      -> (%b.1)
    block1():
      %b.2 : str = prim::Constant[value="abc"]()  // 在 block1 中创建字符串常量节点 %b.2，值为 "abc"
      -> (%b.2)
  %7 : (str, str) = prim::TupleConstruct(%a, %b)   // 构造元组 %7，包含常量 %a 和 %b
  return (%7)
      )IR",
      &*graph);
  // 对图形进行常量池优化
  ConstantPooling(graph);
  // 运行文件检查，验证是否有正好一个值为 "abc" 和一个值为 "bcd" 的 prim::Constant 节点
  testing::FileCheck()
      .check_count("prim::Constant[value=\"abc\"]", 1, /*exactly*/ true)
      ->check_count("prim::Constant[value=\"bcd\"]", 1, /*exactly*/ true)
      ->run(*graph);
}

// 定义 ConstantPoolingTest 类中的测试案例 PoolingDifferentDevices
TEST(ConstantPoolingTest, PoolingDifferentDevices) {
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析输入的 IR 表达式并将其添加到图形中
  parseIR(
      R"IR(
graph():
  %2 : int = prim::Constant[value=2]()      // 创建一个整数常量节点 %2，值为 2
  %1 : int = prim::Constant[value=1]()      // 创建一个整数常量节点 %1，值为 1
  %5 : int? = prim::Constant()              // 创建一个空值整数常量节点 %5
  %7 : Device? = prim::Constant()           // 创建一个空值设备常量节点 %7
  %15: bool = prim::Constant[value=0]()     // 创建一个布尔常量节点 %15，值为 false
  %10 : int = prim::Constant[value=6]()     // 创建一个整数常量节点 %10，值为 6
  %3 : int[] = prim::ListConstruct(%1, %2)  // 创建一个整数列表常量节点 %3，包含值 %1 和 %2
  %x : Tensor = aten::tensor(%3, %5, %7, %15) // 创建一个张量常量节点 %x，使用列表 %3 和其他常量
  %y : Tensor = aten::tensor(%3, %10, %7, %15) // 创建一个张量常量节点 %y，使用列表 %3 和其他常量
  %9 : int[] = prim::ListConstruct(%1, %2)  // 创建一个整数列表常量节点 %9，包含值 %1 和 %2
  %z : Tensor = aten::tensor(%9, %10, %7, %15) // 创建一个张量常量节点 %z，使用列表 %9 和其他常量
  prim::Print(%x, %y, %z)                   // 打印张量 %x, %y, %z 的信息
  return (%1)                               // 返回常量 %1
      )IR",
      &*graph);
  // 对图形进行常量传播优化
  ConstantPropagation(graph);
  // 对图形进行常量池优化
  ConstantPooling(graph);
  // 运行文件检查，验证是否有正好一个 Float 类型和一个 Long 类型的 prim::Constant 节点
  testing::FileCheck()
      .check_count(
          "Float(2, strides=[1], requires_grad=0, device=cpu) = prim::Constant",
          1,
          /*exactly*/ true)
      ->check_count(
          "Long(2, strides=[1], requires_grad=0, device=cpu) = prim::Constant",
          1,
          /*exactly*/ true)
      ->run(*graph);
}

// 定义 ConstantPoolingTest 类中的测试案例 DictConstantPooling
TEST(ConstantPoolingTest, DictConstantPooling) {
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析输入的 IR 表达式并将其添加到图形中
  parseIR(
      R"IR(
# 定义一个名为 graph 的函数，没有参数
graph():
  # %0 是一个整数常量节点，值为 1，位于 test/elias.py 文件的第 6 行第 9 列
  %0 : int = prim::Constant[value=1]() # test/elias.py:6:9
  # %1 是一个整数常量节点，值为 2，位于 test/elias.py 文件的第 6 行第 12 列
  %1 : int = prim::Constant[value=2]() # test/elias.py:6:12
  # %a.1 是一个整数到整数的字典构造节点，包含一个键值对 (0 -> 1)
  %a.1 : Dict(int, int) = prim::DictConstruct(%0, %1)
  # %b.1 是一个整数到整数的字典构造节点，包含一个键值对 (1 -> 1)
  %b.1 : Dict(int, int) = prim::DictConstruct(%1, %1)
  # 返回两个字典构造节点 %a.1 和 %b.1
  return (%a.1, %b.1)
  )IR",
  # 将 graph 传递给 ConstantPropagation 函数进行常量传播优化
  &*graph);
  # 将 graph 传递给 ConstantPooling 函数进行常量池化优化
  ConstantPropagation(graph);
  ConstantPooling(graph);
  # 运行测试，检查 prim::Constant 出现的确切次数为 2
  testing::FileCheck()
      .check_count(
          "Dict(int, int) = prim::Constant",
          2,
          /*exactly*/ true)
      ->run(*graph);
}
} // namespace jit
} // namespace torch
```