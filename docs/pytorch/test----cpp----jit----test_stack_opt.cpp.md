# `.\pytorch\test\cpp\jit\test_stack_opt.cpp`

```
// 包含 ATen 库函数头文件
#include <ATen/Functions.h>
// 包含 Google Test 测试框架头文件
#include <gtest/gtest.h>

// 包含 JIT 测试工具函数头文件
#include <test/cpp/jit/test_utils.h>
// 包含 IR 解析器头文件
#include <torch/csrc/jit/ir/irparser.h>
// 包含变长操作的头文件
#include <torch/csrc/jit/passes/variadic_ops.h>
// 包含解释器头文件
#include <torch/csrc/jit/runtime/interpreter.h>
// 包含文件检查工具头文件
#include <torch/csrc/jit/testing/file_check.h>

// 命名空间 torch::jit 下的测试套件 StackOptTest
namespace torch {
namespace jit {

// 定义 StackOptTest 测试套件中的 UseVariadicStack 测试用例
TEST(StackOptTest, UseVariadicStack) {
  // 创建一个空的计算图对象
  auto graph = std::make_shared<Graph>();

  // 定义输入的 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56),
              %3: Float(56, 56, 56),
              %4: Float(56, 56, 56),
              %5: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1, %2, %3, %4, %5)
          %stack : Float(5, 56, 56, 56) = aten::stack(%input, %10)
          return (%stack)
      )IR";

  // 解析输入的 IR 字符串并将结果存储到 graph 中
  parseIR(input, graph.get());

  // 创建包含随机张量的输入向量
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};
  
  // 运行未优化的图并获取输出结果
  auto orig_outputs = runGraph(graph, inputs);

  // 断言优化过程中是否使用了变长堆栈操作
  ASSERT_TRUE(UseVariadicStack(graph));

  // 执行图的静态分析
  graph->lint();

  // 再次运行优化后的图并获取输出结果
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化后的输出与原始输出是否完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 使用文件检查工具验证替换后的图是否符合预期
  // 替换 `aten::stack` 为 `prim::VarStack` 后，期望的图结构如下：
  //
  //  graph(%0 : ...,
  //        %1 : ...):
  //    %zero : int = prim:Constant[value=0]()
  //    %varstack : Tensor = prim::VarStack(%0, %1, %2, %3, %4, %5, %zero)
  //    return (%varstack)
  testing::FileCheck()
      .check_count("= prim::VarStack(", 1, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

// 结束命名空间 torch::jit
} // namespace jit
// 结束命名空间 torch
} // namespace torch
// 定义名为 StackOptTest 的测试用例，测试变参堆栈替换多个情况
TEST(StackOptTest, UseVariadicStackReplaceMultiple) {
  // 创建一个共享指针指向图对象
  auto graph = std::make_shared<Graph>();

  // 定义输入的 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56),
              %3: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input1 : Tensor[] = prim::ListConstruct(%0, %1)
          %stack1 : Float(4, 56, 56, 56) = aten::stack(%input1, %10)
          %input2 : Tensor[] = prim::ListConstruct(%2, %3)
          %stack2 : Float(4, 56, 56, 56) = aten::stack(%input2, %10)
          return (%stack1, %stack2)
      )IR";

  // 解析 IR 字符串并将结果存入图中
  parseIR(input, graph.get());

  // 创建包含四个随机张量的输入向量
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};

  // 运行图并将原始输出保存在 orig_outputs 中
  auto orig_outputs = runGraph(graph, inputs);

  // 断言使用变参堆栈优化函数 UseVariadicStack 返回真
  ASSERT_TRUE(UseVariadicStack(graph));

  // 对图进行静态分析
  graph->lint();

  // 再次运行图并将优化后的输出保存在 opt_outputs 中
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化前后输出张量完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 完全堆栈优化后，应该得到以下图形式：
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...,
  //        %3 : ....):
  //    %zero : int = prim:Constant[value=0]()
  //    %varcat1 : Tensor = prim::VarStack(%0, %1, %zero)
  //    %varcat2 : Tensor = prim::VarStack(%2, %3, %zero)
  //    return (%varcat1, %varcat2)
  testing::FileCheck()
      .check_count("= prim::VarStack(", 2, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}
TEST(StackOptTest, UseVariadicStackWithMultipleListUses) {
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();

  // 定义输入字符串，包含IR表示的计算图
  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56)):
          %2 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %stack : Float(2, 56, 56, 56) = aten::stack(%input, %2)
          return (%stack, %input)
      )IR";

  // 解析输入的IR字符串，构建计算图
  parseIR(input, graph.get());

  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU), at::rand({56, 56, 56}, at::kCPU)};

  // 运行原始的计算图，获取原始输出
  auto orig_outputs = runGraph(graph, inputs);

  // 断言优化函数 UseVariadicStack 返回真值
  ASSERT_TRUE(UseVariadicStack(graph));

  // 对图进行lint操作，确保图的正确性
  graph->lint();

  // 再次运行优化后的计算图，获取优化后的输出
  auto opt_outputs = runGraph(graph, inputs);

  // 断言原始输出与优化输出完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 替换后的计算图中，aten::stack应该被替换为prim::VarStack，验证替换后的图形结构
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= prim::VarStack(", 1, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(StackOptTest, UseVariadicStackWithListMutationAfterCat) {
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();

  // 定义输入字符串，包含IR表示的计算图
  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %stack : Float(3, 56, 56, 56) = aten::stack(%input, %10)
          %11 : Tensor = aten::append(%input, %2)
          return (%stack, %input)
      )IR";

  // 解析输入的IR字符串，构建计算图
  parseIR(input, graph.get());

  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};

  // 运行原始的计算图，获取原始输出
  auto orig_outputs = runGraph(graph, inputs);

  // 断言优化函数 UseVariadicStack 返回真值
  ASSERT_TRUE(UseVariadicStack(graph));

  // 对图进行lint操作，确保图的正确性
  graph->lint();

  // 再次运行优化后的计算图，获取优化后的输出
  auto opt_outputs = runGraph(graph, inputs);

  // 断言原始输出与优化输出完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 在aten::stack操作之后对输入列表进行了变异，因此应该将其替换为prim::VarStack，
  // 验证替换后的图形结构
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= prim::VarStack(", 1, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->run(*graph);
}
TEST(StackOptTest, UseVariadicStackWithListMutationBeforeCat) {
  // 创建一个共享的图对象
  auto graph = std::make_shared<Graph>();

  // 定义输入的 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %11 : Tensor = aten::append(%input, %2)
          %stack : Float(3, 56, 56, 56) = aten::stack(%input, %10)
          return (%stack)
      )IR";

  // 解析输入的 IR 字符串并将其添加到图中
  parseIR(input, graph.get());

  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};
  
  // 运行未优化的图，并保存原始输出
  auto orig_outputs = runGraph(graph, inputs);

  {
    // 断言未使用 VariadicStack 优化
    ASSERT_FALSE(UseVariadicStack(graph));
    // 对图进行 lint 检查
    graph->lint();
    // 运行优化后的图，并保存优化后的输出
    auto opt_outputs = runGraph(graph, inputs);
    // 断言优化后的输出与原始输出完全相等

    // 由于 prim::ListConstruct 在 aten::stack 之前被修改，因此不应发生任何转换
    testing::FileCheck()
        .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::stack(", 1, /*exactly*/ true)
        ->check_count("= prim::VarStack(", 0, /*exactly*/ true)
        ->run(*graph);
  }

  {
    // 断言移除列表修改并使用 VariadicStack
    ASSERT_TRUE(RemoveListMutationAndUseVariadicStack(graph));
    // 对图进行 lint 检查
    graph->lint();
    // 再次运行优化后的图，并保存优化后的输出
    auto opt_outputs = runGraph(graph, inputs);
    // 断言优化后的输出与原始输出完全相等

    // 必须移除列表的修改，并将 aten::stack 操作替换为 prim::VarStack 操作
    // 优化后的图应该类似于以下结构：
    //
    //  graph(%0 : ...,
    //        %1 : ...,
    //        %2 : ...):
    //    %3 : int = prim:Constant[value=0]()
    //    %7 : Tensor = prim::VarStack(%0, %1, %2, %3)
    //    return (%7)
    testing::FileCheck()
        .check_count("= prim::VarStack(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
        ->check_count("= aten::stack(", 0, /*exactly*/ true)
        ->run(*graph);
  }
}
TEST(StackOptTest, UseVariadicStackWithMultipleListMutations) {
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针指向图形对象

  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56),
              %3: Float(56, 56, 56),
              %4: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()  // 定义整数常量
          %input : Tensor[] = prim::ListConstruct(%0, %1)  // 构造张量列表
          %stack.1 : Float(5, 56, 56, 56) = aten::stack(%input, %10)  // 使用 ATen 的 stack 操作，将列表中的张量堆叠
          %11 : Tensor = aten::append(%input, %2)  // 在列表中追加张量
          %stack.2 : Float(5, 56, 56, 56) = aten::stack(%input, %10)  // 再次使用 stack 操作堆叠更新后的列表
          %12 : Tensor = aten::append(%input, %3)  // 继续在列表中追加张量
          %stack.3 : Float(5, 56, 56, 56) = aten::stack(%input, %10)  // 再次使用 stack 操作堆叠更新后的列表
          %13 : Tensor = aten::append(%input, %4)  // 继续在列表中追加张量
          %stack.4 : Float(5, 56, 56, 56) = aten::stack(%input, %10)  // 最后一次使用 stack 操作堆叠更新后的列表
          return (%stack.1, %stack.2, %stack.3, %stack.4)  // 返回所有堆叠后的张量
      )IR";
  parseIR(input, graph.get());  // 解析输入的 IR（Intermediate Representation），将其添加到图形中
  std::vector<at::Tensor> inputs = {  // 创建张量输入向量
      at::rand({56, 56, 56}, at::kCPU),  // 生成随机张量
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);  // 运行图形，获取原始输出

  ASSERT_TRUE(RemoveListMutationAndUseVariadicStack(graph));  // 断言：移除列表变异并使用可变堆栈
  graph->lint();  // 对图形进行检查
  auto opt_outputs = runGraph(graph, inputs);  // 运行优化后的图形，获取优化输出
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));  // 断言：优化前后输出应相等

  // All the mutations of the list must be removed and the `aten::stack` ops
  // must be replaced with `prim::VarStack` ops in the graph. The transformed
  // graph should look like the following:
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...,
  //        %3 : ...,
  //        %4 : ...):
  //    %10 : int = prim:Constant[value=0]()
  //    %5 : Tensor = prim::VarStack(%0, %1, %10)
  //    %6 : Tensor = prim::VarStack(%0, %1, %2, %10)
  //    %7 : Tensor = prim::VarStack(%0, %1, %2, %3, %10)
  //    %8 : Tensor = prim::VarStack(%0, %1, %2, %3, %4, %10)
  //    return (%5, %6, %7, %8)
  testing::FileCheck()
      .check_count("= prim::VarStack(", 4, /*exactly*/ true)  // 断言：确保变换后有四个 prim::VarStack 操作
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)  // 断言：确保列表构造操作被移除
      ->check_count("= aten::stack(", 0, /*exactly*/ true)  // 断言：确保所有 aten::stack 操作被移除
      ->run(*graph);  // 在图形上运行检查
}

} // namespace jit
} // namespace torch
```