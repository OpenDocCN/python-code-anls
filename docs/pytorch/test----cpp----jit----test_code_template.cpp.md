# `.\pytorch\test\cpp\jit\test_code_template.cpp`

```py
// 包含 Google Test 的头文件
#include <gtest/gtest.h>

// 包含 ATen 库的代码模板和测试实用工具的头文件
#include <ATen/code_template.h>
#include <test/cpp/jit/test_utils.h>

// 声明命名空间 torch::jit
namespace torch {
namespace jit {

// 定义静态常量 ct，使用 ATen 库的代码模板
static const auto ct = at::jit::CodeTemplate(R"(
  int foo($args) {

      $bar
          $bar
      $a+$b
  }
  int commatest(int a${,stuff})
  int notest(int a${,empty,})
  )");

// 定义期望的 ct 字符串
static const auto ct_expect = R"(
  int foo(hi, 8) {

      what
      on many
      lines...
      7
          what
          on many
          lines...
          7
      3+4
  }
  int commatest(int a, things..., others)
  int notest(int a)
  )";

// 测试代码模板的复制行为
TEST(TestCodeTemplate, Copying) {
  // 创建模板环境 e
  at::jit::TemplateEnv e;
  // 设置字符串变量 "hi" 为 "foo"
  e.s("hi", "foo");
  // 设置字符串数组变量 "what" 包含 {"is", "this"}
  e.v("what", {"is", "this"});
  // 使用 e 创建 c，并复制 e 的内容
  at::jit::TemplateEnv c(e);
  // 修改 c 中的 "hi" 为 "foo2"
  c.s("hi", "foo2");
  // 断言 e 中 "hi" 为 "foo"
  ASSERT_EQ(e.s("hi"), "foo");
  // 断言 c 中 "hi" 为 "foo2"
  ASSERT_EQ(c.s("hi"), "foo2");
  // 断言 e 中的 "what" 数组的第一个元素为 "is"
  ASSERT_EQ(e.v("what")[0], "is");
}

// 测试代码模板的格式化功能
TEST(TestCodeTemplate, Formatting) {
  // 创建模板环境 e
  at::jit::TemplateEnv e;
  // 设置字符串数组变量 "args" 为 {"hi", "8"}
  e.v("args", {"hi", "8"});
  // 设置字符串数组变量 "bar" 为 {"what\non many\nlines...", "7"}
  e.v("bar", {"what\non many\nlines...", "7"});
  // 设置字符串变量 "a" 为 "3"
  e.s("a", "3");
  // 设置字符串变量 "b" 为 "4"
  e.s("b", "4");
  // 设置字符串数组变量 "stuff" 为 {"things...", "others"}
  e.v("stuff", {"things...", "others"});
  // 设置空字符串数组变量 "empty"
  e.v("empty", {});
  // 使用模板环境 e 格式化代码模板 ct，生成字符串 s
  auto s = ct.format(e);
  // 比较生成的字符串 s 和预期的 ct_expect 字符串是否相等
  ASSERT_EQ(s, ct_expect);
}

} // namespace jit
} // namespace torch
```