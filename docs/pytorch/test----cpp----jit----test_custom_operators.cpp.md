# `.\pytorch\test\cpp\jit\test_custom_operators.cpp`

```py
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <test/cpp/jit/test_utils.h> // 引入测试工具函数的头文件

#include <torch/csrc/jit/ir/alias_analysis.h> // 引入别名分析的头文件
#include <torch/csrc/jit/ir/irparser.h> // 引入 IR 解析器的头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h> // 引入死代码消除的头文件
#include <torch/csrc/jit/runtime/custom_operator.h> // 引入自定义运算符的运行时支持头文件
#include <torch/csrc/jit/runtime/register_ops_utils.h> // 引入操作注册工具函数的头文件
#include <torch/jit.h> // 引入 Torch 脚本模块的头文件

namespace torch {
namespace jit {

TEST(CustomOperatorTest, InferredSchema) { // 自定义运算符推断模式的单元测试
  torch::RegisterOperators reg( // 注册自定义运算符 "foo::bar"
      "foo::bar", [](double a, at::Tensor b) { return a + b; });
  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foo::bar")); // 获取注册的运算符列表
  ASSERT_EQ(ops.size(), 1); // 断言运算符列表中有且仅有一个运算符

  auto& op = ops.front(); // 获取第一个运算符
  ASSERT_EQ(op->schema().name(), "foo::bar"); // 断言运算符的名称为 "foo::bar"

  ASSERT_EQ(op->schema().arguments().size(), 2); // 断言运算符的参数数量为2
  ASSERT_EQ(op->schema().arguments()[0].name(), "_0"); // 断言第一个参数的名称为 "_0"
  ASSERT_EQ(op->schema().arguments()[0].type()->kind(), TypeKind::FloatType); // 断言第一个参数的类型为浮点型
  ASSERT_EQ(op->schema().arguments()[1].name(), "_1"); // 断言第二个参数的名称为 "_1"
  ASSERT_EQ(op->schema().arguments()[1].type()->kind(), TypeKind::TensorType); // 断言第二个参数的类型为张量型

  ASSERT_EQ(op->schema().returns()[0].type()->kind(), TypeKind::TensorType); // 断言运算符的返回类型为张量型

  Stack stack; // 创建操作堆栈
  push(stack, 2.0f, at::ones(5)); // 压入堆栈中的参数值
  op->getOperation()(stack); // 调用运算符的操作函数
  at::Tensor output; // 定义输出张量
  pop(stack, output); // 从堆栈中弹出结果到输出张量

  ASSERT_TRUE(output.allclose(at::full(5, 3.0f))); // 断言输出张量与预期张量相近
}

TEST(CustomOperatorTest, ExplicitSchema) { // 自定义运算符显式模式的单元测试
  torch::RegisterOperators reg( // 注册具有显式模式的自定义运算符 "foo::bar_with_schema"
      "foo::bar_with_schema(float a, Tensor b) -> Tensor",
      [](double a, at::Tensor b) { return a + b; });

  auto& ops =
      getAllOperatorsFor(Symbol::fromQualString("foo::bar_with_schema")); // 获取注册的运算符列表
  ASSERT_EQ(ops.size(), 1); // 断言运算符列表中有且仅有一个运算符

  auto& op = ops.front(); // 获取第一个运算符
  ASSERT_EQ(op->schema().name(), "foo::bar_with_schema"); // 断言运算符的名称为 "foo::bar_with_schema"

  ASSERT_EQ(op->schema().arguments().size(), 2); // 断言运算符的参数数量为2
  ASSERT_EQ(op->schema().arguments()[0].name(), "a"); // 断言第一个参数的名称为 "a"
  ASSERT_EQ(op->schema().arguments()[0].type()->kind(), TypeKind::FloatType); // 断言第一个参数的类型为浮点型
  ASSERT_EQ(op->schema().arguments()[1].name(), "b"); // 断言第二个参数的名称为 "b"
  ASSERT_EQ(op->schema().arguments()[1].type()->kind(), TypeKind::TensorType); // 断言第二个参数的类型为张量型

  ASSERT_EQ(op->schema().returns().size(), 1); // 断言运算符的返回值数量为1
  ASSERT_EQ(op->schema().returns()[0].type()->kind(), TypeKind::TensorType); // 断言运算符的返回类型为张量型

  Stack stack; // 创建操作堆栈
  push(stack, 2.0f, at::ones(5)); // 压入堆栈中的参数值
  op->getOperation()(stack); // 调用运算符的操作函数
  at::Tensor output; // 定义输出张量
  pop(stack, output); // 从堆栈中弹出结果到输出张量

  ASSERT_TRUE(output.allclose(at::full(5, 3.0f))); // 断言输出张量与预期张量相近
}

} // namespace jit
} // namespace torch
TEST(CustomOperatorTest, ListParameters) {
  // 检查列表的使用情况。
  torch::RegisterOperators reg(
      // 注册自定义运算符 "foo::lists"，接受多种类型的列表作为参数并返回浮点数列表。
      "foo::lists(int[] ints, float[] floats, complex[] complexdoubles, Tensor[] tensors) -> float[]",
      [](torch::List<int64_t> ints,
         torch::List<double> floats,
         torch::List<c10::complex<double>> complexdoubles,
         torch::List<at::Tensor> tensors) { return floats; });

  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foo::lists"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "foo::lists");

  ASSERT_EQ(op->schema().arguments().size(), 4);
  ASSERT_EQ(op->schema().arguments()[0].name(), "ints");
  ASSERT_TRUE(
      op->schema().arguments()[0].type()->isSubtypeOf(*ListType::ofInts()));
  ASSERT_EQ(op->schema().arguments()[1].name(), "floats");
  ASSERT_TRUE(
      op->schema().arguments()[1].type()->isSubtypeOf(*ListType::ofFloats()));
  ASSERT_EQ(op->schema().arguments()[2].name(), "complexdoubles");
  ASSERT_TRUE(op->schema().arguments()[2].type()->isSubtypeOf(
      *ListType::ofComplexDoubles()));
  ASSERT_EQ(op->schema().arguments()[3].name(), "tensors");
  ASSERT_TRUE(
      op->schema().arguments()[3].type()->isSubtypeOf(*ListType::ofTensors()));

  ASSERT_EQ(op->schema().returns().size(), 1);
  ASSERT_TRUE(
      op->schema().returns()[0].type()->isSubtypeOf(*ListType::ofFloats()));

  Stack stack;
  push(stack, c10::List<int64_t>({1, 2}));
  push(stack, c10::List<double>({1.0, 2.0}));
  push(
      stack,
      c10::List<c10::complex<double>>(
          {c10::complex<double>(2.4, -5.5), c10::complex<double>(-1.3, 2)}));
  push(stack, c10::List<at::Tensor>({at::ones(5)}));
  op->getOperation()(stack);
  c10::List<double> output;
  pop(stack, output);

  ASSERT_EQ(output.size(), 2);
  ASSERT_EQ(output.get(0), 1.0);
  ASSERT_EQ(output.get(1), 2.0);
}

TEST(CustomOperatorTest, ListParameters2) {
  torch::RegisterOperators reg(
      // 注册另一个自定义运算符 "foo::lists2"，接受张量列表作为参数并返回张量列表。
      "foo::lists2(Tensor[] tensors) -> Tensor[]",
      [](torch::List<at::Tensor> tensors) { return tensors; });

  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foo::lists2"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "foo::lists2");

  ASSERT_EQ(op->schema().arguments().size(), 1);
  ASSERT_EQ(op->schema().arguments()[0].name(), "tensors");
  ASSERT_TRUE(
      op->schema().arguments()[0].type()->isSubtypeOf(*ListType::ofTensors()));

  ASSERT_EQ(op->schema().returns().size(), 1);
  ASSERT_TRUE(
      op->schema().returns()[0].type()->isSubtypeOf(*ListType::ofTensors()));

  Stack stack;
  push(stack, c10::List<at::Tensor>({at::ones(5)}));
  op->getOperation()(stack);
  c10::List<at::Tensor> output;
  pop(stack, output);

  ASSERT_EQ(output.size(), 1);
  ASSERT_TRUE(output.get(0).allclose(at::ones(5)));
}


这段代码包含了两个测试案例，分别注册和测试了两个自定义的Torch运算符。注释解释了每行代码的作用，确保代码的每个部分都得到了清晰的说明。
TEST(CustomOperatorTest, Aliasing) {
  // 在测试中注册自定义操作符 "foo::aliasing"
  torch::RegisterOperators reg(
      "foo::aliasing", [](at::Tensor a, at::Tensor b) -> at::Tensor {
        // 自定义操作符执行张量 a 和 b 的原位加法操作
        a.add_(b);
        // 返回操作后的张量 a
        return a;
      });
  // 获取所有符号为 "foo::aliasing" 的操作符
  getAllOperatorsFor(Symbol::fromQualString("foo::aliasing"));

  {
    // 创建一个共享的计算图对象
    auto graph = std::make_shared<Graph>();
    // 解析以下 IR 表示的图结构
    parseIR(
        R"IR(
graph(%x: Tensor, %y: Tensor):
  %ret : Tensor = foo::aliasing(%x, %y)
  return (%ret)
  )IR",
        graph.get());

    // 获取操作节点
    auto opNode = *graph->block()->nodes().begin();

    // 创建别名数据库对象，用于分析张量别名
    AliasDb aliasDb(graph);
    // 遍历操作节点的每个输入
    for (const auto input : opNode->inputs()) {
      // 自定义操作符会写入所有输入的别名
      ASSERT_TRUE(aliasDb.writesToAlias(opNode, {input}));
      // 输出应该是通配符，因此与所有输入都可能有别名
      ASSERT_TRUE(aliasDb.mayAlias(opNode->output(), input));
    }
  }
  {
    // 死代码消除不应该移除自定义操作符
    auto graph = std::make_shared<Graph>();
    const auto text = R"IR(
graph(%x: Tensor, %y: Tensor):
  # CHECK: foo::aliasing
  %ret : Tensor = foo::aliasing(%x, %y)
  return (%x)
  )IR";
    // 解析图结构文本
    parseIR(text, graph.get());
    // 执行死代码消除
    EliminateDeadCode(graph);

    // 使用 FileCheck 检查特定文本是否存在于修改后的图中
    testing::FileCheck().run(text, *graph);
  }
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
// 定义操作列表的静态常量字符串
static constexpr char op_list[] = "foofoo::bar.template;foo::another";
// 定义宏以根据条件选择架构中的名称
#define TORCH_SELECTIVE_NAME_IN_SCHEMA(l, n)                                   \
  torch::detail::SelectiveStr<c10::impl::op_allowlist_contains_name_in_schema( \
      l, n)>(n)

TEST(TestCustomOperator, OperatorGeneratorUndeclared) {
  // 尝试注册一个在操作列表 op_list 中不存在的操作名
  // 预期结果：该操作名不会被注册
  torch::jit::RegisterOperators reg({OperatorGenerator(
      TORCH_SELECTIVE_NAME_IN_SCHEMA(
          op_list, "foofoo::not_exist(float a, Tensor b) -> Tensor"),
      [](Stack& stack) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        // 操作符执行的 Lambda 函数，将栈顶的 float 和 Tensor 弹出并求和
        double a;
        at::Tensor b;
        pop(stack, a, b);
        push(stack, a + b);
      },
      aliasAnalysisFromSchema())});

  // 获取所有符号为 "foofoo::not_exist" 的操作符
  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foofoo::not_exist"));
  // 断言操作符数量为 0
  ASSERT_EQ(ops.size(), 0);
}
TEST(TestCustomOperator, OperatorGeneratorBasic) {
  // 测试自定义操作符生成器的基本功能

  // 注册操作符，使用 OperatorGenerator 匿名函数创建操作符，并通过名称验证注册
  torch::jit::RegisterOperators reg({
      OperatorGenerator(
          TORCH_SELECTIVE_NAME_IN_SCHEMA(
              op_list, "foofoo::bar.template(float a, Tensor b) -> Tensor"),
          [](Stack& stack) {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            double a;
            at::Tensor b;
            // 从栈中弹出参数 a 和 b
            pop(stack, a, b);
            // 将 a 和 b 相加，结果推入栈中
            push(stack, a + b);
          },
          aliasAnalysisFromSchema())
  });

  // 获取所有符号为 "foofoo::bar" 的操作符
  auto& ops = getAllOperatorsFor(Symbol::fromQualString("foofoo::bar"));
  ASSERT_EQ(ops.size(), 1);

  // 验证第一个操作符的名称为 "foofoo::bar"
  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "foofoo::bar");

  // 验证操作符的参数列表为两个
  ASSERT_EQ(op->schema().arguments().size(), 2);
  ASSERT_EQ(op->schema().arguments()[0].name(), "a");
  ASSERT_EQ(op->schema().arguments()[0].type()->kind(), TypeKind::FloatType);
  ASSERT_EQ(op->schema().arguments()[1].name(), "b");
  ASSERT_EQ(op->schema().arguments()[1].type()->kind(), TypeKind::TensorType);

  // 验证操作符的返回值列表为一个 Tensor 类型
  ASSERT_EQ(op->schema().returns().size(), 1);
  ASSERT_EQ(op->schema().returns()[0].type()->kind(), TypeKind::TensorType);

  // 准备测试栈
  Stack stack;
  // 将浮点数 2.0 和一个全为 1 的 Tensor 推入栈中
  push(stack, 2.0f, at::ones(5));
  // 调用操作符的操作函数
  op->getOperation()(stack);
  // 从栈中弹出输出 Tensor
  at::Tensor output;
  pop(stack, output);

  // 验证输出 Tensor 是否与全为 3 的 Tensor 相似
  ASSERT_TRUE(output.allclose(at::full(5, 3.0f)));
}

} // namespace jit
} // namespace torch
```