# `.\pytorch\test\cpp\jit\test_add_if_then_else.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <test/cpp/jit/test_utils.h>  // 引入测试工具函数的头文件
#include <torch/csrc/jit/ir/irparser.h>  // 引入 IR 解析器的头文件
#include <torch/csrc/jit/passes/add_if_then_else.h>  // 引入添加 IfThenElse Pass 的头文件

namespace torch {
namespace jit {

TEST(AddIfThenElseOpTest, AddIfThenElseOpSimple) {  // 定义 AddIfThenElseOpTest 测试用例
  const auto src = R"IR(  // 定义一个包含 IR 的字符串常量
        graph(%cond: bool, %a: Tensor, %b: Tensor):
            %result: Tensor = prim::If(%cond)  // 如果条件 %cond 满足，则执行 If 分支
                block0():  // If 分支的第一个块
                    -> (%a)  // 返回张量 %a
                block1():  // If 分支的第二个块
                    -> (%b)  // 返回张量 %b
            return (%result)  // 返回最终结果张量 %result
    )IR";

  auto graph = std::make_shared<Graph>();  // 创建一个共享指针指向 Graph 对象
  parseIR(src, graph.get());  // 解析 IR 字符串并填充到 graph 对象中
  EXPECT_TRUE(AddIfThenElseOp(graph));  // 断言添加 IfThenElseOp 成功

  testing::FileCheck()
      .check_count("= prim::IfThenElse", 1, /*exactly*/ true)  // 检查 IR 中出现 IfThenElse 的确切次数为 1
      ->check_count("= prim::If", 0, /*exactly*/ true)  // 检查 IR 中出现 If 的确切次数为 0
      ->run(*graph);  // 在 graph 上运行文件检查器
}

TEST(AddIfThenElseOpTest, NoIfThenElseOpMultipleOutputs) {  // 定义 NoIfThenElseOpMultipleOutputs 测试用例
  const auto src = R"IR(  // 定义一个包含 IR 的字符串常量
        graph(%cond: bool, %a: Tensor, %b: Tensor):
            %result1: Tensor, %result2: Tensor = prim::If(%cond)  // 如果条件 %cond 满足，则执行 If 分支
                block0():  // If 分支的第一个块
                    -> (%a, %b)  // 返回张量 %a 和 %b
                block1():  // If 分支的第二个块
                    -> (%b, %a)  // 返回张量 %b 和 %a
            return (%result1, %result2)  // 返回最终结果张量 %result1 和 %result2
    )IR";

  auto graph = std::make_shared<Graph>();  // 创建一个共享指针指向 Graph 对象
  parseIR(src, graph.get());  // 解析 IR 字符串并填充到 graph 对象中
  EXPECT_FALSE(AddIfThenElseOp(graph));  // 断言添加 IfThenElseOp 失败

  testing::FileCheck()
      .check_count("= prim::IfThenElse", 0, /*exactly*/ true)  // 检查 IR 中出现 IfThenElse 的确切次数为 0
      ->check_count("= prim::If", 1, /*exactly*/ true)  // 检查 IR 中出现 If 的确切次数为 1
      ->run(*graph);  // 在 graph 上运行文件检查器
}

} // namespace jit
} // namespace torch
```