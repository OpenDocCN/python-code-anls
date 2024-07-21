# `.\pytorch\aten\src\ATen\test\wrapdim_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/ATen.h>  // 引入 PyTorch C++ 扩展库的头文件

using namespace at;  // 使用 at 命名空间，简化对 ATen 库中函数和类的访问

void TestSimpleCase(DeprecatedTypeProperties& T) {
  auto a = randn({2, 3, 4, 5}, T);  // 生成一个指定形状的随机张量 a
  ASSERT_TRUE(a.prod(-4).equal(a.prod(0)));  // 断言：在指定维度上求积的结果相等
  ASSERT_TRUE(a.prod(3).equal(a.prod(-1)));  // 断言：在指定维度上求积的结果相等
}

void TestExpressionSpecification(DeprecatedTypeProperties& T) {
  auto a = randn({2, 3, 4, 5}, T);  // 生成一个指定形状的随机张量 a
  ASSERT_TRUE(a.unsqueeze(-5).equal(a.unsqueeze(0)));  // 断言：在指定维度上展开的结果相等
  ASSERT_TRUE(a.unsqueeze(4).equal(a.unsqueeze(-1)));  // 断言：在指定维度上展开的结果相等

  // 可以展开标量
  auto b = randn({}, T);  // 生成一个标量 b
  ASSERT_TRUE(b.unsqueeze(0).equal(b.unsqueeze(-1)));  // 断言：在指定维度上展开的结果相等
}

void TestEmptyTensor(DeprecatedTypeProperties& T) {
  auto a = randn(0, T);  // 生成一个形状为空的随机张量 a
  ASSERT_TRUE(a.prod(0).equal(at::ones({}, T)));  // 断言：在指定维度上求积的结果等于一个全为 1 的张量
}

void TestScalarVs1Dim1Size(DeprecatedTypeProperties& T) {
  auto a = randn(1, T);  // 生成一个形状为 [1] 的随机张量 a
  ASSERT_TRUE(a.prod(0).equal(a.prod(-1)));  // 断言：在指定维度上求积的结果相等
  a.resize_({});  // 将张量 a 重置为标量
  ASSERT_EQ(a.dim(), 0);  // 断言：张量的维度为 0
  ASSERT_TRUE(a.prod(0).equal(a.prod(-1)));  // 断言：在指定维度上求积的结果相等
}

TEST(TestWrapdim, TestWrapdim) {
  manual_seed(123);  // 设置随机数种子为 123
  DeprecatedTypeProperties& T = CPU(kFloat);  // 创建一个 CPU 上的浮点类型属性

  TestSimpleCase(T);  // 执行测试函数 TestSimpleCase
  TestEmptyTensor(T);  // 执行测试函数 TestEmptyTensor
  TestScalarVs1Dim1Size(T);  // 执行测试函数 TestScalarVs1Dim1Size
  TestExpressionSpecification(T);  // 执行测试函数 TestExpressionSpecification
}
```