# `.\pytorch\test\cpp\jit\test_cleanup_passes.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <torch/csrc/jit/frontend/ir_emitter.h>  // 包含 Torch JIT 前端的 IR emitter 头文件
#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch JIT 中 IR 的头文件
#include <torch/csrc/jit/ir/irparser.h>  // 包含 Torch JIT 中 IR 解析器的头文件
#include <torch/csrc/jit/testing/file_check.h>  // 包含 Torch JIT 测试中的文件检查头文件

namespace torch {
namespace jit {

TEST(CleanupPassTest, Basic) {  // 定义名为 CleanupPassTest 的测试套件，测试基本功能
  // Tests stability of clean up passes when dealing with constant pooling
  // and constant propagation.
  // 测试清理传递（cleanup passes）在处理常量池和常量传播时的稳定性。

  auto graph = std::make_shared<Graph>();  // 创建一个共享指针 graph，指向一个新的 Graph 对象
  parseIR(  // 调用 parseIR 函数，解析下面提供的 IR 字符串，将结果存储到 graph 中
      R"IR(
graph(%cond.1 : Tensor,
      %suffix.1 : str):
  %3 : bool = aten::Bool(%cond.1) # o.py:6:7
  %25 : str = prim::If(%3) # o.py:6:4
    block0():
      %a.1 : str = prim::Constant[value="same string"]()
      %b.1 : str = prim::Constant[value=" with a twist"]()
      %7 : str = aten::add(%a.1, %b.1)
      %11 : str = aten::add(%7, %suffix.1) # o.py:10:15
      -> (%11)
    block1():
      %c.1 : str = prim::Constant[value="same string"]()
      %d.1 : str = prim::Constant[value=" with a twist"]()
      %12 : str = aten::add(%c.1, %d.1)
      -> (%12)
  return (%25)
  )IR",
      &*graph);  // 将 IR 字符串解析到 graph 中

  runCleanupPasses(graph);  // 运行清理传递函数，处理 graph 中的优化和清理操作

  testing::FileCheck()  // 创建一个文件检查对象
      .check_count(  // 检查特定模式出现的次数
          "prim::Constant[value=\"same string with a twist\"]",  // 检查图中是否存在特定的常量字符串
          1,  // 期望的出现次数
          /*exactly=*/true)  // 确保精确匹配次数
      ->run(*graph);  // 在 graph 上运行文件检查，验证是否符合预期模式

  auto graph_after_pass_once = graph->toString();  // 获取第一次优化后的 graph 字符串表示
  runCleanupPasses(graph);  // 再次运行清理传递函数
  auto graph_after_pass_twice = graph->toString();  // 获取第二次优化后的 graph 字符串表示
  ASSERT_EQ(graph_after_pass_once, graph_after_pass_twice);  // 断言第一次和第二次优化后的 graph 字符串相等
}

}  // namespace jit
}  // namespace torch
```