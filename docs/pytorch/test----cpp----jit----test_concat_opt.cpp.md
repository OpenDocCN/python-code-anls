# `.\pytorch\test\cpp\jit\test_concat_opt.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 ATen 库的函数头文件
#include <ATen/Functions.h>

// 包含用于测试的实用函数的头文件
#include <test/cpp/jit/test_utils.h>

// 包含解析器相关的头文件
#include <torch/csrc/jit/ir/irparser.h>

// 包含拼接优化相关的头文件
#include <torch/csrc/jit/passes/concat_opt.h>

// 包含可变操作相关的头文件
#include <torch/csrc/jit/passes/variadic_ops.h>

// 包含解释器相关的头文件
#include <torch/csrc/jit/runtime/interpreter.h>

// 包含用于文件检查的头文件
#include <torch/csrc/jit/testing/file_check.h>

// 定义命名空间 torch::jit 内的测试套件 ConcatOptTest
namespace torch {
namespace jit {

// 定义名为 SimpleCommonInputsEliminationPrefix 的测试用例
TEST(ConcatOptTest, SimpleCommonInputsEliminationPrefix) {
  // 创建一个共享指针指向新建的图对象
  auto graph = std::make_shared<Graph>();

  // 定义输入的 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %5)
          %concat.3 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %5)
          %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
          return (%res)
      )IR";

  // 解析输入的 IR 字符串到图对象
  parseIR(input, graph.get());

  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};

  // 运行原始图并获取输出结果
  auto orig_outputs = runGraph(graph, inputs);

  // 执行拼接优化，检查是否成功
  ASSERT_TRUE(EliminateConcatCommonInputs(graph));

  // 检查图的合法性
  graph->lint();

  // 重新运行优化后的图并获取输出结果
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化前后的输出结果完全相同
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 输出优化后的图的结构，用于后续的文件检查
  // Graph after EliminateConcatCommonInputs:
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim::Constant[value=0]()
  //    %4 : Tensor = prim::VarConcat(%0, %1, %3)
  //    %7 : Tensor = prim::VarConcat(%4, %2, %3) // UPDATED
  //    %8 : Tensor[] = prim::ListConstruct(%4, %7)
  //    return (%8)
  testing::FileCheck()
      .check_count("= prim::VarConcat(%0, %1, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%4, %2, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(%4, %7)", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

} // namespace jit
} // namespace torch
// 定义测试函数，测试消除共同输入的简单情况
TEST(ConcatOptTest, SimpleCommonInputsEliminationSuffix) {
  // 创建共享指针指向图对象
  auto graph = std::make_shared<Graph>();

  // 定义输入的 IR 表示
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%1, %2, %5)
          %concat.3 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %5)
          %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
          return (%res)
      )IR";

  // 解析 IR 表示并将其添加到图中
  parseIR(input, graph.get());

  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};

  // 运行原始图并获取输出结果
  auto orig_outputs = runGraph(graph, inputs);

  // 断言消除共同输入后的操作成功
  ASSERT_TRUE(EliminateConcatCommonInputs(graph));

  // 对图进行检查
  graph->lint();

  // 再次运行图并获取优化后的输出结果
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化前后输出结果一致
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 输出消除共同输入后的图形状和操作信息
  // Graph after EliminateConcatCommonInputs:
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim::Constant[value=0]()
  //    %4 : Tensor = prim::VarConcat(%1, %2, %3)
  //    %7 : Tensor = prim::VarConcat(%0, %4, %3) // UPDATED
  //    %8 : Tensor[] = prim::ListConstruct(%4, %7)
  //    return (%8)

  // 使用 FileCheck 工具验证图形上的特定操作和数量
  testing::FileCheck()
      .check_count("= prim::VarConcat(%1, %2, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%0, %4, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(%4, %7)", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}
// 定义测试用例 `ConcatOptTest.CommonInputsEliminationWithDifferentOrderInputs`
TEST(ConcatOptTest, CommonInputsEliminationWithDifferentOrderInputs) {
  // 创建一个共享的图对象
  auto graph = std::make_shared<Graph>();

  // 定义输入 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()

          #CHECK: prim::VarConcat
          %concat.1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %5)

          #CHECK: prim::VarConcat
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%1, %0, %2, %5)

          #CHECK: prim::ListConstruct
          %res : Tensor[] = prim::ListConstruct(%concat.1, %concat.2)
          return (%res)
      )IR";

  // 解析输入 IR 字符串并将其添加到图中
  parseIR(input, graph.get());

  // 创建输入张量向量
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};

  // 运行未优化的图计算，获取原始输出
  auto orig_outputs = runGraph(graph, inputs);

  // 断言：消除公共输入的优化不应该成功
  ASSERT_FALSE(EliminateConcatCommonInputs(graph));

  // 对图进行检查
  graph->lint();

  // 再次运行优化后的图计算，获取优化后的输出
  auto opt_outputs = runGraph(graph, inputs);

  // 断言：优化前后的输出应该完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 在这种情况下不应有任何优化发生，因为 `cat` 的输入顺序不同
  // 使用 FileCheck 工具验证输入 IR 在图中的匹配情况
  testing::FileCheck().run(input, *graph);
}
// 定义一个名为 `MoreCommonInputsElimination` 的测试用例函数，属于 `ConcatOptTest` 测试集合
TEST(ConcatOptTest, MoreCommonInputsElimination) {
  // 创建一个指向图对象的共享指针 `graph`
  auto graph = std::make_shared<Graph>();

  // 定义包含 IR（Intermediate Representation）代码的字符串 `input`
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %4: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()
          // 创建 `prim::VarConcat` 操作，将 `%0` 和 `%1` 合并成 `%concat.1`
          %concat.1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %5)
          // 创建 `prim::VarConcat` 操作，将 `%0`、`%1` 和 `%2` 合并成 `%concat.2`
          %concat.2 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %5)
          // 创建 `prim::VarConcat` 操作，将 `%0`、`%1`、`%2` 和 `%3` 合并成 `%concat.3`
          %concat.3 : Float(160, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %3, %5)
          // 创建 `prim::VarConcat` 操作，将 `%0`、`%1`、`%2`、`%3` 和 `%4` 合并成 `%concat.4`
          %concat.4 : Float(192, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %3, %4, %5)
          // 创建 `prim::ListConstruct` 操作，构造一个包含所有 `prim::VarConcat` 结果的列表 `%res`
          %res : Tensor[] = prim::ListConstruct(%concat.1, %concat.2, %concat.3, %concat.4)
          // 返回结果列表 `%res`
          return (%res)
      )IR";

  // 解析 IR 字符串 `input` 并将结果存入图对象 `graph`
  parseIR(input, graph.get());

  // 创建一个包含随机张量的输入向量 `inputs`
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};

  // 运行图 `graph`，使用 `inputs` 作为输入，获取原始输出 `orig_outputs`
  auto orig_outputs = runGraph(graph, inputs);

  // 断言：调用消除共同输入的优化函数 `EliminateConcatCommonInputs` 返回真
  ASSERT_TRUE(EliminateConcatCommonInputs(graph));

  // 对图 `graph` 进行 lint（静态分析）
  graph->lint();

  // 再次运行图 `graph`，使用相同的 `inputs`，获取优化后的输出 `opt_outputs`
  auto opt_outputs = runGraph(graph, inputs);

  // 断言：原始输出 `orig_outputs` 和优化输出 `opt_outputs` 必须完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 使用 FileCheck 进行多项检查：
  // 1. 检查 `prim::VarConcat(%0, %1, %5)` 出现的次数为 1 次
  // 2. 检查 `prim::VarConcat(%6, %2, %5)` 出现的次数为 1 次
  // 3. 检查 `prim::VarConcat(%11, %3, %5)` 出现的次数为 1 次
  // 4. 检查 `prim::VarConcat(%12, %4, %5)` 出现的次数为 1 次
  // 5. 检查 `aten::cat(` 不出现
  // 最后执行检查操作，并使用 `graph` 作为其参数
  testing::FileCheck()
      .check_count("= prim::VarConcat(%0, %1, %5)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%6, %2, %5)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%11, %3, %5)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%12, %4, %5)", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}
TEST(ConcatOptTest, ExpandConcat) {
    // 创建一个新的图形对象，用于表示计算图
    auto graph = std::make_shared<Graph>();

    // 定义包含优化前计算图的字符串形式
    const std::string input =
        R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %2 : int = prim::Constant[value=0]()
          %3 : float = prim::Constant[value=0.5]()
          %4 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%0, %3)
          %5 : Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%1, %3)
          %input : Tensor[] = prim::ListConstruct(%4, %5)
          %concat : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %2)
          return (%concat)
      )IR";

    // 解析输入的 IR 字符串，将计算图构建到 graph 对象中
    parseIR(input, graph.get());

    // 创建输入数据张量
    std::vector<at::Tensor> inputs = {
        at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};

    // 运行初始计算图，获取初始输出
    auto orig_outputs = runGraph(graph, inputs);

    // 执行拼接优化和冗余消除操作
    ExpandConcatAndEliminateRedundancy(graph);

    // 对优化后的计算图进行检查
    graph->lint();

    // 再次运行优化后的计算图，获取优化后的输出
    auto opt_outputs = runGraph(graph, inputs);

    // 使用断言检查优化前后输出是否完全相等
    ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

    // 输出最终优化后的计算图预期结构
    //
    //  graph(%0 : ...,
    //        %1 : ...):
    //    ...
    //    %4 : Tensor = aten::clamp_max(...)
    //    %5 : Tensor = aten::clamp_max(...)
    //    %13 : int[] = prim::ListConstruct(...)
    //    %14 : Tensor = aten::empty(%13, ...)    // concat buffer
    //    %17 : Tensor = aten::slice(%14, ...)    // slice for %4
    //    %18 : Tensor = aten::copy_(%17, %4)
    //    %20 : Tensor = aten::slice(%14, ...)    // slice for %5
    //    %21 : Tensor = aten::copy_(%20, %5)
    //    return (%14)
    testing::FileCheck()
        .check_count("= aten::cat(", 0, /*exactly*/ true)
        ->check_count("= aten::clamp_max(", 2, /*exactly*/ true)
        ->check_count("= aten::empty(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 1, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 1, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 0, /*exactly*/ true)
        ->run(*graph);
}
TEST(ConcatOptTest, ConcatWithoutResultShape) {
  // 创建一个空的计算图
  auto graph = std::make_shared<Graph>();

  // 定义输入 IR 字符串，描述了两个 Float 类型的输入张量
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %2 : int = prim::Constant[value=0]()
          %3 : float = prim::Constant[value=0.5]()
          # CHECK: clamp_max
          %4 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%0, %3)
          # CHECK: clamp_max
          %5 : Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%1, %3)
          # CHECK: prim::ListConstruct
          %6 : Tensor[] = prim::ListConstruct(%4, %5)
          # CHECK: aten::cat
          %7 : Tensor = aten::cat(%6, %2)
          return (%7)
      )IR";

  // 解析输入的 IR 字符串并将图形化对象附加到图形中
  parseIR(input, graph.get());

  // 创建两个随机张量作为输入
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};

  // 运行未优化的计算图并获取原始输出
  auto orig_outputs = runGraph(graph, inputs);

  // 执行扩展拼接和消除冗余优化
  ExpandConcatAndEliminateRedundancy(graph);

  // 对优化后的图进行检查
  graph->lint();

  // 再次运行优化后的图并获取输出
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化前后的输出是否完全相同
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 在这种情况下不应进行任何优化，因为 `aten::cat` 的输出形状未知。
  testing::FileCheck().run(input, *graph);
}

TEST(ConcatOptTest, ConcatWithoutInputShape) {
  // 创建一个空的计算图
  auto graph = std::make_shared<Graph>();

  // 定义输入 IR 字符串，描述了两个 Float 类型的输入张量
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %2 : int = prim::Constant[value=0]()
          %3 : float = prim::Constant[value=0.5]()
          # CHECK: clamp_max
          %4 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%0, %3)
          # CHECK: clamp_max
          %5 : Tensor = aten::clamp_max(%1, %3)
          # CHECK: prim::ListConstruct
          %6 : Tensor[] = prim::ListConstruct(%4, %5)
          # CHECK: aten::cat
          %7 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%6, %2)
          return (%7)
      )IR";

  // 解析输入的 IR 字符串并将图形化对象附加到图形中
  parseIR(input, graph.get());

  // 创建两个随机张量作为输入
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};

  // 运行未优化的计算图并获取原始输出
  auto orig_outputs = runGraph(graph, inputs);

  // 执行扩展拼接和消除冗余优化
  ExpandConcatAndEliminateRedundancy(graph);

  // 对优化后的图进行检查
  graph->lint();

  // 再次运行优化后的图并获取输出
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化前后的输出是否完全相同
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 在这种情况下不应进行任何优化，因为 `aten::cat` 的输入张量 %5 的形状未知。
  testing::FileCheck().run(input, *graph);
}
# 定义一个测试用例 TEST(ConcatOptTest, UseVariadicCat)，用于测试使用可变参数进行连接操作的优化

auto graph = std::make_shared<Graph>();
# 创建一个名为 graph 的共享指针，用于存储图结构

const std::string input =
    R"IR(
      graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
            %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
            %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
            %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
            %4: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
            %5: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
        %10 : int = prim::Constant[value=0]()
        %input : Tensor[] = prim::ListConstruct(%0, %1, %2, %3, %4, %5)
        %concat : Float(224, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
        return (%concat)
    )IR";
# 定义输入的 IR（Intermediate Representation），描述了一个计算图的结构和操作

parseIR(input, graph.get());
# 解析输入的 IR 并将其加载到 graph 中

std::vector<at::Tensor> inputs = {
    at::rand({64, 56, 56}, at::kCPU),
    at::rand({32, 56, 56}, at::kCPU),
    at::rand({32, 56, 56}, at::kCPU),
    at::rand({32, 56, 56}, at::kCPU),
    at::rand({32, 56, 56}, at::kCPU),
    at::rand({32, 56, 56}, at::kCPU)};
# 创建一个包含随机张量的输入向量

auto orig_outputs = runGraph(graph, inputs);
# 运行图 graph，并将 inputs 作为输入，获得原始输出结果

ASSERT_TRUE(UseVariadicCat(graph));
# 断言使用可变参数连接操作的优化函数 UseVariadicCat 返回 true

graph->lint();
# 对图进行检查，确保其结构正确

auto opt_outputs = runGraph(graph, inputs);
# 再次运行图 graph，获得优化后的输出结果

ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));
# 断言优化前后的输出结果相等

// After replacing `aten::cat` with `prim::VarConcat` we should have the
// following graph:
//
//  graph(%0 : ...,
//        %1 : ...):
//    %zero : int = prim:Constant[value=0]()
//    %varcat : Tensor = prim::VarConcat(%0, %1, %2, %3, %4, %5, %zero)
//    return (%varcat)
# 描述替换 aten::cat 操作为 prim::VarConcat 操作后的期望图结构和操作

testing::FileCheck()
    .check_count("= prim::VarConcat(", 1, /*exactly*/ true)
    ->check_count("= aten::cat(", 0, /*exactly*/ true)
    ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
    ->run(*graph);
# 使用 FileCheck 进行图的检查，确保 prim::VarConcat 出现了一次，
# 而 aten::cat 和 prim::ListConstruct 都没有出现，确保优化操作正确实施
TEST(OptimizeConcatTest, UseVariadicCatReplaceMultiple) {
  // 创建一个共享的图对象
  auto graph = std::make_shared<Graph>();

  // 定义输入的 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input1 : Tensor[] = prim::ListConstruct(%0, %1)
          %concat1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input1, %10)
          %input2 : Tensor[] = prim::ListConstruct(%2, %3)
          %concat2 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input2, %10)
          return (%concat1, %concat2)
      )IR";

  // 解析 IR 字符串并构建图
  parseIR(input, graph.get());

  // 构建输入张量的向量
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};

  // 运行原始图以获取输出
  auto orig_outputs = runGraph(graph, inputs);

  // 使用优化函数进行变量拼接优化，并进行断言验证
  ASSERT_TRUE(UseVariadicCat(graph));

  // 对图进行 lint 检查
  graph->lint();

  // 再次运行优化后的图以获取输出
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化前后输出的张量是否完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 对优化后的图进行进一步检查，确保优化成功
  // 在完全拼接优化后，期望图如下所示：
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...,
  //        %3 : ....):
  //    %zero : int = prim:Constant[value=0]()
  //    %varcat1 : Tensor = prim::VarConcat(%0, %1, %zero)
  //    %varcat2 : Tensor = prim::VarConcat(%2, %3, %zero)
  //    return (%varcat1, %varcat2)
  testing::FileCheck()
      .check_count("= prim::VarConcat(", 2, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}
// 定义测试函数 `ConcatOptTest.UseVariadicCatWithMultipleListUses`
TEST(ConcatOptTest, UseVariadicCatWithMultipleListUses) {
  // 创建一个指向 Graph 对象的智能指针
  auto graph = std::make_shared<Graph>();

  // 定义输入字符串，包含表示计算图的 IR（中间表示）
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %2 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %concat : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %2)
          return (%concat, %input)
      )IR";

  // 解析输入字符串并将结果存储到 graph 对象中
  parseIR(input, graph.get());

  // 创建一个包含两个随机张量的向量
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};

  // 运行原始图，并获取输出结果
  auto orig_outputs = runGraph(graph, inputs);

  // 断言优化过程中是否成功使用了 `UseVariadicCat` 优化函数
  ASSERT_TRUE(UseVariadicCat(graph));

  // 对图进行静态检查
  graph->lint();

  // 再次运行优化后的图，并获取输出结果
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化后的输出结果与原始输出结果完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 在将 `aten::cat` 替换为 `prim::VarConcat` 后，检查图的变化
  // 应该得到如下图：
  //
  //  graph(%0 : ...,
  //        %1 : ...):
  //    %zero : int = prim:Constant[value=0]()
  //    %input : Tensor[] = prim::ListConstruct(%0, %1)
  //    %varcat : Tensor = prim::VarConcat(%0, %1, %zero)
  //    return (%varcat, %input)
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}
TEST(ConcatOptTest, UseVariadicCatWithListMutationAfterCat) {
  // 创建一个共享的图形对象
  auto graph = std::make_shared<Graph>();

  // 定义输入的 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %concat : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          %11 : Tensor = aten::append(%input, %2)
          return (%concat, %input)
      )IR";

  // 解析 IR 字符串并更新到图形对象中
  parseIR(input, graph.get());

  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};

  // 运行未优化的图形并获取原始输出
  auto orig_outputs = runGraph(graph, inputs);

  // 断言使用 variadic cat 后的优化结果
  ASSERT_TRUE(UseVariadicCat(graph));

  // 执行图形的静态分析
  graph->lint();

  // 再次运行优化后的图形并获取输出
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化后的输出与原始输出相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 对变换后的图形结构进行检查和验证
  // 期望转换后的图形如下所示：
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim:Constant[value=0]()
  //    %4 : Tensor[] = prim::ListConstruct(%0, %1)
  //    %7 : Tensor = prim::VarConcat(%0, %1, %3)
  //    %6 : Tensor = aten::append(%4, %2)
  //    return (%7, %4)
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOptTest, UseVariadicCatWithListMutationBeforeCat) {
  // 创建一个共享的图形对象
  auto graph = std::make_shared<Graph>();

  // 定义输入的 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %11 : Tensor = aten::append(%input, %2)
          %concat : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          return (%concat)
      )IR";

  // 解析 IR 字符串并更新到图形对象中
  parseIR(input, graph.get());

  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};

  // 运行未优化的图形并获取原始输出
  auto orig_outputs = runGraph(graph, inputs);

  // 断言不使用 variadic cat 后的优化结果
  {
    ASSERT_FALSE(UseVariadicCat(graph));

    // 执行图形的静态分析
    graph->lint();

    // 再次运行优化后的图形并获取输出
    auto opt_outputs = runGraph(graph, inputs);

    // 断言优化后的输出与原始输出相等
    ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));
  }
}
  // 不应有任何转换发生，因为 `prim::ListConstruct` 在 `aten::cat` 之前被修改。
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(", 0, /*exactly*/ true)
      ->run(*graph);
}

{
  // 断言移除列表的变异并使用可变参数的 `aten::cat` 操作。
  ASSERT_TRUE(RemoveListMutationAndUseVariadicCat(graph));
  // 对图进行静态检查
  graph->lint();
  // 运行经过优化的图，并获取输出结果
  auto opt_outputs = runGraph(graph, inputs);
  // 断言优化后的输出与原始输出完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 列表的变异必须被移除，并且图中的 `aten::cat` 操作必须被替换为 `prim::VarConcat` 操作。
  // 转换后的图应如下所示：
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim:Constant[value=0]()
  //    %7 : Tensor = prim::VarConcat(%0, %1, %2, %3)
  //    return (%7)
  testing::FileCheck()
      .check_count("= prim::VarConcat(", 1, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}
TEST(ConcatOptTest, UseVariadicCatWithMultipleListMutations) {
  // 创建一个共享指针指向图对象
  auto graph = std::make_shared<Graph>();

  // 定义包含IR的输入字符串
  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %4: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %concat.1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          %11 : Tensor = aten::append(%input, %2)
          %concat.2 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          %12 : Tensor = aten::append(%input, %3)
          %concat.3 : Float(160, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          %13 : Tensor = aten::append(%input, %4)
          %concat.4 : Float(192, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          return (%concat.1, %concat.2, %concat.3, %concat.4)
      )IR";

  // 解析IR并将其添加到图中
  parseIR(input, graph.get());

  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};

  // 运行原始图，并保存输出结果
  auto orig_outputs = runGraph(graph, inputs);

  // 断言移除列表变异和使用变长cat操作后的图变换成功
  ASSERT_TRUE(RemoveListMutationAndUseVariadicCat(graph));

  // 对图进行lint检查
  graph->lint();

  // 运行优化后的图，并保存输出结果
  auto opt_outputs = runGraph(graph, inputs);

  // 断言优化后的输出与原始输出完全相等
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 所有列表的变异操作必须被移除，aten::cat操作必须被替换为prim::VarConcat操作。
  // 转换后的图应如下所示：
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...,
  //        %3 : ...,
  //        %4 : ...):
  //    %10 : int = prim:Constant[value=0]()
  //    %5 : Tensor = prim::VarConcat(%0, %1, %10)
  //    %6 : Tensor = prim::VarConcat(%0, %1, %2, %10)
  //    %7 : Tensor = prim::VarConcat(%0, %1, %2, %3, %10)
  //    %8 : Tensor = prim::VarConcat(%0, %1, %2, %3, %4, %10)
  //    return (%5, %6, %7, %8)
  testing::FileCheck()
      .check_count("= prim::VarConcat(", 4, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}
    // 创建一个共享指针指向 Graph 对象
    auto graph = std::make_shared<Graph>();
    
    // 定义包含 IR 表达式的字符串
    const std::string input =
        R"IR(
          graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
                %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
                %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
            %5 : int = prim::Constant[value=0]()
    
            %features.2 : Tensor[] = prim::ListConstruct(%0, %1)
            %6 : Tensor[] = aten::append(%features.2, %2)
            %concat.2 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)
    
            %7 : Tensor[] = aten::append(%features.2, %0)
            %concat.3 : Float(160, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)
    
            %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
            return (%res)
        )IR";
    
    // 解析 IR 字符串并将结果存储在 graph 中
    parseIR(input, graph.get());
    
    // 创建包含三个随机张量的输入向量
    std::vector<at::Tensor> inputs = {
        at::rand({64, 56, 56}, at::kCPU),
        at::rand({32, 56, 56}, at::kCPU),
        at::rand({32, 56, 56}, at::kCPU)};
    
    // 运行原始图并存储输出
    auto orig_outputs = runGraph(graph, inputs);
    
    // 断言，确保成功应用了以下优化：
    //     * 移除列表修改
    //     * 使用可变参数的 cat 函数
    //     * 消除公共输入
    ASSERT_TRUE(RemoveListMutationAndUseVariadicCat(graph));
    ASSERT_TRUE(EliminateConcatCommonInputs(graph));
    
    // 对图进行静态分析
    graph->lint();
    
    // 再次运行优化后的图并存储输出
    auto opt_outputs = runGraph(graph, inputs);
    
    // 断言，确保优化前后的输出一致
    ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));
    
    // 输出测试结果，验证优化后的图的正确性
    // 在执行以下优化后：
    //     * 移除列表修改
    //     * 使用可变参数的 cat 函数
    //     * 消除公共输入
    // 应该得到如下图形：
    //
    //  graph(%0 : ...,
    //        %1 : ...,
    //        %2 : ...):
    //    %3 : int = prim::Constant[value=0]()
    //    %10 : Tensor = prim::VarConcat(%0, %1, %2, %3)
    //    %12 : Tensor = prim::VarConcat(%10, %0, %3) // 更新后
    //    %8 : Tensor[] = prim::ListConstruct(%10, %12)
    //    return (%8)
    testing::FileCheck()
        .check_count("= prim::VarConcat(%0, %1, %2, %3)", 1, /*exactly*/ true)
        ->check_count("= prim::VarConcat(%10, %0, %3)", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(%10, %12)", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 0, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
        ->run(*graph);
TEST(ConcatOpt, CombineConcatsSimpleCase) {
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 定义输入的 IR 表示
  const std::string input =
      R"IR(
        graph(%0: Tensor):
          %dim : int = prim::Constant[value=0]()
          %input.1 : Tensor[] = prim::ListConstruct(%0, %0)
          %concat.1 : Tensor = aten::cat(%input.1, %dim)
          %input.2 : Tensor[] = prim::ListConstruct(%concat.1, %0)
          %concat.2 : Tensor = aten::cat(%input.2, %dim)
          return (%concat.2)
      )IR";
  // 解析输入的 IR 表示并将其添加到图中
  parseIR(input, graph.get());
  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {at::rand({1})};
  // 运行图形对象，获取原始输出结果
  auto orig_outputs = runGraph(graph, inputs);

  // 断言优化操作 CombineConcats 的执行成功
  ASSERT_TRUE(CombineConcats(graph));
  // 对图形对象进行静态分析
  graph->lint();
  // 再次运行图形对象，获取优化后的输出结果
  auto opt_outputs = runGraph(graph, inputs);
  // 断言优化前后输出结果的一致性
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 执行 CombineConcats 优化后的预期图形状态:
  //  graph(%0 : Tensor):
  //    %dim : int = prim::Constant[value=0]()
  //    %input : Tensor[] = prim::ListConstruct(%0, %0, %0)
  //    %concat : Tensor = aten::cat(%input, %dim)
  //    return (%concat)
  testing::FileCheck()
      .check_count("prim::ListConstruct", 1, /*exactly*/ true)
      ->check_count("aten::cat", 1, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOpt, CombineConcatsLongChain) {
  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 定义输入的 IR 表示
  const std::string input =
      R"IR(
        graph(%0: Tensor, %1 : Tensor):
          %dim : int = prim::Constant[value=0]()
          %input.1 : Tensor[] = prim::ListConstruct(%0, %0)
          %concat.1 : Tensor = aten::cat(%input.1, %dim)
          %input.2 : Tensor[] = prim::ListConstruct(%1, %concat.1, %1)
          %concat.2 : Tensor = aten::cat(%input.2, %dim)
          %input.3 : Tensor[] = prim::ListConstruct(%0, %concat.2, %0)
          %concat.3 : Tensor = aten::cat(%input.3, %dim)
          return (%concat.3)
      )IR";
  // 解析输入的 IR 表示并将其添加到图中
  parseIR(input, graph.get());
  // 创建输入张量列表
  std::vector<at::Tensor> inputs = {at::rand({1}), at::randn({1})};
  // 运行图形对象，获取原始输出结果
  auto orig_outputs = runGraph(graph, inputs);

  // 断言优化操作 CombineConcats 的执行成功
  ASSERT_TRUE(CombineConcats(graph));
  // 对图形对象进行静态分析
  graph->lint();
  // 再次运行图形对象，获取优化后的输出结果
  auto opt_outputs = runGraph(graph, inputs);
  // 断言优化前后输出结果的一致性
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // 执行 CombineConcats 优化后的预期图形状态:
  //  graph(%0 : Tensor):
  //    %dim : int = prim::Constant[value=0]()
  //    %input : Tensor[] = prim::ListConstruct(%0, %1, %0, %0, %1, %0)
  //    %concat : Tensor = aten::cat(%input, %dim)
  //    return (%concat)
  testing::FileCheck()
      .check_count("prim::ListConstruct", 1, /*exactly*/ true)
      ->check_count("aten::cat", 1, /*exactly*/ true)
      ->run(*graph);
}
TEST(ConcatOpt, CombineConcatsMutation) {
  // 创建一个共享的图形对象
  auto graph = std::make_shared<Graph>();
  // 定义输入的 IR 字符串
  const std::string input =
      R"IR(
        graph(%0: Tensor, %1 : Tensor):
          %dim : int = prim::Constant[value=0]()
          %input.1 : Tensor[] = prim::ListConstruct(%0, %0)
          %concat.1 : Tensor = aten::cat(%input.1, %dim)
          %input.2 : Tensor[] = prim::ListConstruct(%1, %concat.1, %1)
          %input.3 : Tensor[] = aten::append(%input.2, %0)
          %concat.2 : Tensor = aten::cat(%input.2, %dim)
          return (%concat.2)
      )IR";
  // 解析 IR 字符串并将其添加到图形对象中
  parseIR(input, graph.get());
  // 创建输入张量的向量
  std::vector<at::Tensor> inputs = {at::rand({1}), at::randn({1})};
  // 由于 aten::append 没有修改，断言应该为假
  ASSERT_FALSE(CombineConcats(graph));
}

} // namespace jit
} // namespace torch
```