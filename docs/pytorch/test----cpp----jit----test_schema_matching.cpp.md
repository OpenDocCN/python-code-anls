# `.\pytorch\test\cpp\jit\test_schema_matching.cpp`

```py
#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {

// 定义 SchemaMatchingTest 测试套件中的 VarType 测试用例
TEST(SchemaMatchingTest, VarType) {
  // 注册自定义运算符 "aten::test_vartype(t[] a, t b) -> (t)"
  RegisterOperators reg({
      Operator(
          "aten::test_vartype(t[] a, t b) -> (t)",
          [](Stack& stack) {
            c10::List<double> list;
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            double a;
            // 从堆栈中弹出参数并赋给 list 和 a
            pop(stack, list, a);
            // 将 a 推入堆栈作为结果
            push(stack, a);
          },
          c10::AliasAnalysisKind::FROM_SCHEMA),
  });
  
  // 创建一个名为 m 的模块
  Module m("m");
  // 定义一个 Torch 脚本函数
  m.define(R"(
      def test(self):
        a = (1.0, 2.0)
        return torch.test_vartype(a, 2.0)
    )");
  
  // 运行定义的 test 函数并获取结果
  auto result = m.run_method("test");
  // 内部断言，确保结果转换为 double 后等于 2.0
  TORCH_INTERNAL_ASSERT(result.toDouble() == 2.0);

  // 定义一个包含错误 Torch 脚本的字符串常量
  const std::string error_example = R"JIT(
      def test_2(self):
          a = (1.0, 2.0)
          non_float = (1, 1)
          return torch.test_vartype(a, non_float)
    )JIT";

  std::string err = "";
  try {
    // 尝试定义包含错误的 Torch 脚本
    m.define(error_example);
  } catch (const std::exception& e) {
    // 捕获异常并将错误信息保存到 err 变量
    err = e.what();
  }
  // 内部断言，确保错误信息中包含特定的字符串
  TORCH_INTERNAL_ASSERT(
      err.find("previously matched to type") != std::string::npos);
}

// 定义 SchemaMatchingTest 测试套件中的 VarType2 测试用例
TEST(SchemaMatchingTest, VarType2) {
  // 注册自定义运算符 "aten::test_vartype2(t a, t[] b) -> (t[])"
  RegisterOperators reg({
      Operator(
          "aten::test_vartype2(t a, t[] b) -> (t[])",
          [](Stack& stack) {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            double a;
            c10::List<double> list;
            // 从堆栈中弹出参数并赋给 a 和 list
            pop(stack, a, list);
            // 将 a 推入堆栈作为结果
            push(stack, a);
          },
          AliasAnalysisKind::FROM_SCHEMA),
  });

  // 创建一个名为 m 的模块
  Module m("m");
  // 定义一个 Torch 脚本函数
  m.define(R"JIT(
      def test(self):
          a = (1.0, 2.0)
          return torch.test_vartype2(3.0, a)
    )JIT");

  // 运行定义的 test 函数并获取结果
  auto result = m.run_method("test");
  // 内部断言，确保结果转换为 double 后等于 3.0
  TORCH_INTERNAL_ASSERT(result.toDouble() == 3.0);

  // 定义一个包含错误 Torch 脚本的静态常量字符串
  static const auto error_exam2 = R"JIT(
      def test_2(self):
          a = (1, 2)
          return torch.test_vartype2(3.0, a)
    )JIT";

  std::string err = "";
  try {
    // 尝试定义包含错误的 Torch 脚本
    m.define(error_exam2);
  } catch (const std::exception& e) {
    // 捕获异常并将错误信息保存到 err 变量
    err = e.what();
  }
  // 内部断言，确保错误信息中包含特定的字符串
  TORCH_INTERNAL_ASSERT(
      err.find("previously matched to type") != std::string::npos);
}

} // namespace jit
} // namespace torch
```