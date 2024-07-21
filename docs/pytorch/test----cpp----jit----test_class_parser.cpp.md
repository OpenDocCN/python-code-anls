# `.\pytorch\test\cpp\jit\test_class_parser.cpp`

```py
#include <gtest/gtest.h>

#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>

// 声明命名空间 torch::jit
namespace torch {
namespace jit {

// 定义测试用的源代码字符串
constexpr c10::string_view testSource = R"JIT(
  class FooTest:
    def __init__(self, x):
      self.x = x

    def get_x(self):
      return self.x

    an_attribute : Tensor
)JIT";

// 定义测试类 ClassParserTest，测试解析器的基本功能
TEST(ClassParserTest, Basic) {
  // 创建解析器对象，并传入测试源代码
  Parser p(std::make_shared<Source>(testSource));

  // 定义存储解析结果的容器
  std::vector<Def> definitions;
  std::vector<Resolver> resolvers;

  // 解析类定义并存储
  const auto classDef = ClassDef(p.parseClass());

  // 断言解析器达到了预期的结束标记
  p.lexer().expect(TK_EOF);

  // 断言类名正确解析为 "FooTest"
  ASSERT_EQ(classDef.name().name(), "FooTest");

  // 断言类体中定义了三个成员函数
  ASSERT_EQ(classDef.body().size(), 3);

  // 断言第一个成员函数为 "__init__"
  ASSERT_EQ(Def(classDef.body()[0]).name().name(), "__init__");

  // 断言第二个成员函数为 "get_x"
  ASSERT_EQ(Def(classDef.body()[1]).name().name(), "get_x");

  // 断言第三个成员变量为 "an_attribute"
  ASSERT_EQ(
      Var(Assign(classDef.body()[2]).lhs()).name().name(), "an_attribute");

  // 断言第三个成员变量没有赋值表达式
  ASSERT_FALSE(Assign(classDef.body()[2]).rhs().present());

  // 断言第三个成员变量有类型声明
  ASSERT_TRUE(Assign(classDef.body()[2]).type().present());
}

} // namespace jit
} // namespace torch
```