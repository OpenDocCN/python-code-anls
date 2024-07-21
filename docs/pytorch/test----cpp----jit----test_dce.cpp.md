# `.\pytorch\test\cpp\jit\test_dce.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/csrc/jit/ir/irparser.h>  // 引入 Torch JIT IR 解析器的头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>  // 引入 Torch JIT 死代码消除的头文件
#include <torch/csrc/jit/testing/file_check.h>  // 引入 Torch JIT 文件检查工具的头文件

namespace torch {
namespace jit {
TEST(EliminateDeadCodeTest, Basic) {  // 定义名为 EliminateDeadCodeTest 的测试用例
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针指向新建的图对象

  // Consider the following loop:
  //   for i in range(3):
  //     tot += a[0][0]
  //     b = a[0]
  //     b[0] += 1
  //   print(tot)
  // We want to check that b[0] and b are properly marked as live and thus not
  // DCE'd.
  
  // 定义包含 IR 的字符串，描述了一个包含循环的计算图
  const std::string input =
      R"IR(
graph():
  %48 : None = prim::Constant()  // 创建一个常量节点
  %50 : bool = prim::Constant[value=1]()  // 创建一个布尔常量节点
  %0 : int = prim::Constant[value=2]()  // 创建一个整数常量节点
  %12 : int = prim::Constant[value=1]()  // 创建一个整数常量节点
  %24 : int = prim::Constant[value=3]()  // 创建一个整数常量节点
  %31 : int = prim::Constant[value=0]()  // 创建一个整数常量节点
  %2 : int[] = prim::ListConstruct(%0, %0)  // 创建一个整数数组节点
  %a.1 : Tensor = prim::MakeTestTensor()  // 创建一个测试张量节点
  %14 : int[] = prim::ListConstruct(%12)  // 创建一个整数数组节点
  %tot.1 : Tensor = prim::MakeTestTensor()  // 创建一个测试张量节点
  %tot : Tensor = prim::Loop(%24, %50, %tot.1)  // 创建一个循环节点
    block0(%i : int, %tot.6 : Tensor):
      %33 : Tensor = aten::select(%a.1, %31, %31)  // 从张量中选择元素节点
      %35 : Tensor = aten::select(%33, %31, %31)  // 从张量中选择元素节点
      # CHECK: add_
      %tot.3 : Tensor = aten::add_(%tot.6, %35, %12)  // 张量相加节点
      %b.1 : Tensor = aten::select(%a.1, %31, %31)  // 从张量中选择元素节点
      %44 : Tensor = aten::select(%b.1, %31, %31)  // 从张量中选择元素节点
      # CHECK: add_
      %46 : Tensor = aten::add_(%44, %12, %12)  // 张量相加节点
      -> (%50, %tot.3)  // 返回节点的输出
  return (%tot)  // 返回计算图的输出
)IR";
  
  parseIR(input, graph.get());  // 解析输入的 IR 字符串到图中
  EliminateDeadCode(graph);  // 执行死代码消除的操作

  // Check that dead code elimin
  testing::FileCheck().run(input, *graph);  // 运行文件检查工具，验证消除死代码的效果
}
} // namespace jit
} // namespace torch
```