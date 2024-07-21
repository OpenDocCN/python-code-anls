# `.\pytorch\aten\src\ATen\test\operators_test.cpp`

```
// 引入 Google 测试框架的头文件
#include <gtest/gtest.h>

// 引入 PyTorch 的 ATen 头文件
#include <ATen/ATen.h>
#include <ATen/Operators.h>

// 使用 ATen 命名空间
using namespace at;

// 定义模板函数 pass_through_wrapper，用于包装函数调用
template <class F, F Func, class Output, class... Args>
Output pass_through_wrapper(Args... args) {
  return Func(std::forward<Args>(args)...);
}

// 定义测试用例 OperatorsTest.TestFunctionDecltype
TEST(OperatorsTest, TestFunctionDecltype) {
  // 创建两个大小为 5x5 的随机张量 a 和 b
  Tensor a = at::randn({5, 5});
  Tensor b = at::randn({5, 5});
  // 计算预期结果
  auto expected = a * b;

  // 使用 pass_through_wrapper 调用 ATEN_FN2(mul, Tensor) 函数
  auto result = pass_through_wrapper<
    decltype(&ATEN_FN2(mul, Tensor)), &ATEN_FN2(mul, Tensor),
    Tensor, const Tensor&, const Tensor&>(a, b);
  // 断言结果与预期结果相近
  ASSERT_TRUE(at::allclose(result, a * b));
}

// 定义测试用例 OperatorsTest.TestMethodOnlyDecltype
TEST(OperatorsTest, TestMethodOnlyDecltype) {
  // 创建两个大小为 5x5 的随机张量 a 和 b
  Tensor a = at::randn({5, 5});
  Tensor b = at::randn({5, 5});
  // 计算预期结果
  auto expected = a * b;

  // NB: add_ overloads are guaranteed to be method-only
  // 因为张量 API 的工作方式，add_ 重载保证只能作为方法调用
  // 使用 pass_through_wrapper 调用 ATEN_FN2(mul_, Tensor) 函数
  auto& result = pass_through_wrapper<
    decltype(&ATEN_FN2(mul_, Tensor)), &ATEN_FN2(mul_, Tensor),
    Tensor&, Tensor&, const Tensor&>(a, b);
  // 断言结果与预期结果相近
  ASSERT_TRUE(at::allclose(result, expected));
}

// 定义测试用例 OperatorsTest.Test_ATEN_FN
TEST(OperatorsTest, Test_ATEN_FN) {
  // 创建大小为 5x5 的随机张量 a
  Tensor a = at::rand({5, 5});

  // 使用 pass_through_wrapper 调用 ATEN_FN(sin) 函数
  auto result = pass_through_wrapper<
    decltype(&ATEN_FN(sin)), &ATEN_FN(sin),
    Tensor, const Tensor&>(a);
  // 断言结果与 a.sin() 的结果相近
  ASSERT_TRUE(at::allclose(result, a.sin()));
}

// 定义测试用例 OperatorsTest.TestOutVariantIsFaithful
TEST(OperatorsTest, TestOutVariantIsFaithful) {
  // 创建大小为 5x5 的随机张量 a 和空张量 b
  Tensor a = at::rand({5, 5});
  Tensor b = at::empty({5, 5});

  // 使用 pass_through_wrapper 调用 ATEN_FN2(sin, out) 函数
  auto& result = pass_through_wrapper<
    decltype(&ATEN_FN2(sin, out)), &ATEN_FN2(sin, out),
    Tensor&, const Tensor&, Tensor&>(a, b);
  // 断言结果与 a.sin() 的结果相近
  ASSERT_TRUE(at::allclose(result, a.sin()));
}
```