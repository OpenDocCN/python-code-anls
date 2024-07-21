# `.\pytorch\aten\src\ATen\test\broadcast_test.cpp`

```
// 引入 Google Test 框架的头文件
#include <gtest/gtest.h>

// 引入 ATen 库的头文件
#include <ATen/ATen.h>

// 使用 at 命名空间
using namespace at;

// 测试空张量无法扩展的情况
void TestEmptyTensor(DeprecatedTypeProperties& T) {
  // 创建一个空的张量
  auto empty = randn({0}, T);
  // 检查是否会抛出异常，当试图对空张量进行扩展时
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(empty.expand({3}));
}

// 测试带有两个参数的 out-place 函数
void TestOut2Basic(DeprecatedTypeProperties& T) {
  // 创建两个张量 a 和 b
  auto a = randn({3, 1}, T);
  auto b = randn({5}, T);
  std::vector<int64_t> expanded_sizes = {3, 5};
  // 检查两个相同形状的张量相加是否等于它们各自扩展后相加的结果
  ASSERT_TRUE(
      (a + b).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes)));
}

// 测试带有标量的 out-place 函数
void TestOut2WithScalar(DeprecatedTypeProperties& T) {
  // 创建一个标量和一个张量 b
  auto aScalar = ones({}, T);
  auto b = randn({3, 5}, T);
  // 检查标量与张量 b 相加后，是否等于将标量和 b 分别扩展后相加的结果
  ASSERT_TRUE(
      (aScalar + b).equal(aScalar.expand(b.sizes()) + b.expand(b.sizes())));
}

// 测试旧的回退行为是否会产生错误
void TestOut2OldFallback(DeprecatedTypeProperties& T) {
  // 创建两个形状不匹配的张量 a 和 b
  auto a = randn({3, 5}, T);
  auto b = randn({5, 3}, T);
  // 检查是否会抛出异常，当试图对形状不匹配的张量进行相加时
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(a + b);
}

// 测试带有不匹配大小的张量
void TestOut2MismatchedSizes(DeprecatedTypeProperties& T) {
  // 创建两个形状不匹配的张量 a 和 b
  auto a = randn({3, 5}, T);
  auto b = randn({7, 5}, T);
  // 检查是否会抛出异常，当试图对形状不匹配的张量进行相加时
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(a + b);
}

// 测试带有三个参数的 out-place 函数
void TestOut3Basic(DeprecatedTypeProperties& T) {
  // 创建三个张量 a、b 和 c
  auto a = randn({3, 1, 1}, T);
  auto b = randn({1, 2, 1}, T);
  auto c = randn({1, 1, 5}, T);
  std::vector<int64_t> expanded_sizes = {3, 2, 5};
  // 检查三个相同形状的张量相加是否等于它们各自扩展后相加的结果
  ASSERT_TRUE((a + b + c).equal(
      a.expand(expanded_sizes) + b.expand(expanded_sizes) +
      c.expand(expanded_sizes)));
}

// 测试带有标量的 out-place 函数
void TestOut3WithScalar(DeprecatedTypeProperties& T) {
  // 创建一个张量标量和两个张量 b 和 c
  auto aTensorScalar = ones({}, T);
  auto b = randn({3, 2, 1}, T);
  auto c = randn({1, 2, 5}, T);
  std::vector<int64_t> expanded_sizes = {3, 2, 5};
  // 检查标量与张量 b 和 c 进行逐元素相乘加后的结果，是否等于将标量和 b、c 扩展后进行相乘加的结果
  ASSERT_TRUE(aTensorScalar.addcmul(b, c).equal(
      aTensorScalar.expand(expanded_sizes)
          .addcmul(b.expand(expanded_sizes), c.expand(expanded_sizes))));
}

// 测试旧的回退行为是否会产生错误
void TestOut3OldFallback(DeprecatedTypeProperties& T) {
  // 创建三个形状不匹配的张量 a、b 和 c
  auto a = randn({3, 2, 5}, T);
  auto b = randn({2, 3, 5}, T);
  auto c = randn({5, 3, 2}, T);
  // 检查是否会抛出异常，当试图对形状不匹配的张量进行 addcmul 操作时
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(a.addcmul(b, c));
}

// 测试带有不匹配大小的张量
void TestOut3MismatchedSizes(DeprecatedTypeProperties& T) {
  // 创建三个形状不匹配的张量 a、b 和 c
  auto a = randn({3, 2, 5}, T);
  auto b = randn({2, 3, 5}, T);
  auto c = randn({5, 5, 5}, T);
  // 检查是否会抛出异常，当试图对形状不匹配的张量进行 addcmul 操作时
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(a.addcmul(b, c));
}

// 测试带有两个参数的 in-place 函数
void TestIn2Basic(DeprecatedTypeProperties& T) {
  // 创建两个张量 a 和 b
  auto a = randn({3, 5}, T);
  auto b = randn({3, 1}, T);
  // 检查两个相同形状的张量相加后，是否等于其中一个张量加上另一个张量扩展后的结果
  ASSERT_TRUE((a + b).equal(a + b.expand({3, 5})));
}

// 测试带有标量
// 对输入张量和标量进行加法操作，并使用断言验证结果是否与使用扩展后的标量进行加法的结果相等
void TestIn2WithScalar(DeprecatedTypeProperties& T) {
  auto a = randn({3, 5}, T);  // 生成一个大小为 (3, 5) 的随机张量 a
  auto bScalar = ones({}, T);  // 生成一个标量张量 bScalar，其元素为 1
  // 断言：a 和 bScalar 的加法结果与 a 和 bScalar 扩展后的张量的加法结果相等
  ASSERT_TRUE((a + bScalar).equal(a + bScalar.expand(a.sizes())));
}

// 错误示例：需要对 inplace 参数进行扩展操作
void TestIn2ExpandError(DeprecatedTypeProperties& T) {
  auto a = randn({1, 5}, T);  // 生成一个大小为 (1, 5) 的随机张量 a
  auto b = randn({3, 1}, T);  // 生成一个大小为 (3, 1) 的随机张量 b
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言：尝试对 a 执行 inplace 加法操作，期望抛出异常
  ASSERT_ANY_THROW(a.add_(b));
}

// 使用三个参数进行 inplace 操作的函数
void TestIn3Basic(DeprecatedTypeProperties& T) {
  auto a = randn({3, 5, 2}, T);  // 生成一个大小为 (3, 5, 2) 的随机张量 a
  auto b = randn({3, 1, 2}, T);  // 生成一个大小为 (3, 1, 2) 的随机张量 b
  auto c = randn({1, 5, 1}, T);  // 生成一个大小为 (1, 5, 1) 的随机张量 c
  auto aClone = a.clone();       // 复制张量 a
  // 断言：使用三个参数进行 inplace 乘加操作的结果与使用扩展后的参数进行乘加操作的结果相等
  ASSERT_TRUE(a.addcmul_(b, c).equal(
      aClone.addcmul_(b.expand(a.sizes()), c.expand(a.sizes()))));
}

// 使用标量与三个参数进行 inplace 操作
void TestIn3WithScalar(DeprecatedTypeProperties& T) {
  auto a = randn({3, 5, 2}, T);  // 生成一个大小为 (3, 5, 2) 的随机张量 a
  auto b = randn({3, 1, 2}, T);  // 生成一个大小为 (3, 1, 2) 的随机张量 b
  auto c = randn({1, 5, 1}, T);  // 生成一个大小为 (1, 5, 1) 的随机张量 c
  auto aClone = a.clone();       // 复制张量 a
  auto bScalar = ones({}, T);    // 生成一个标量张量 bScalar，其元素为 1
  // 断言：使用标量和三个参数进行 inplace 乘加操作的结果与使用扩展后的参数进行乘加操作的结果相等
  ASSERT_TRUE(a.addcmul_(bScalar, c)
                  .equal(aClone.addcmul_(
                      bScalar.expand(a.sizes()), c.expand(a.sizes()))));
}

// 错误示例：需要对 inplace 参数进行扩展操作
void TestIn3ExpandError(DeprecatedTypeProperties& T) {
  auto a = randn({1, 3, 5}, T);  // 生成一个大小为 (1, 3, 5) 的随机张量 a
  auto b = randn({4, 1, 1}, T);  // 生成一个大小为 (4, 1, 1) 的随机张量 b
  auto c = randn({1, 3, 1}, T);  // 生成一个大小为 (1, 3, 1) 的随机张量 c
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言：尝试对 a 执行 inplace 乘加操作，期望抛出异常
  ASSERT_ANY_THROW(a.addcmul_(b, c));
}

// 显式指定维度进行矩阵乘加操作
void TestExplicitDimBasic(DeprecatedTypeProperties& T) {
  auto a = randn({1}, T);    // 生成一个大小为 (1) 的随机张量 a
  auto b = randn({5, 3}, T); // 生成一个大小为 (5, 3) 的随机张量 b
  auto c = randn({3, 7}, T); // 生成一个大小为 (3, 7) 的随机张量 c
  // 断言：使用显式指定的维度进行 inplace 矩阵乘加操作的结果与使用扩展后的参数进行乘加操作的结果相等
  ASSERT_TRUE(a.addmm(b, c).equal(a.expand({5, 7}).addmm(b, c)));
}

// 使用标量进行显式维度矩阵乘加操作
void TestExplicitDimWithScalar(DeprecatedTypeProperties& T) {
  auto a = randn({1}, T);         // 生成一个大小为 (1) 的随机张量 a
  auto b = randn({5, 3}, T);      // 生成一个大小为 (5, 3) 的随机张量 b
  auto c = randn({3, 7}, T);      // 生成一个大小为 (3, 7) 的随机张量 c
  Tensor aScalar = ones({}, T);   // 生成一个标量张量 aScalar，其元素为 1
  // 断言：使用标量和显式指定的维度进行 inplace 矩阵乘加操作的结果与使用扩展后的参数进行乘加操作的结果相等
  ASSERT_TRUE(aScalar.addmm(b, c).equal(aScalar.expand({5, 7}).addmm(b, c)));
}

// 错误示例：参数尺寸不匹配，无法进行乘加操作
void TestExplicitDimWithMismatchedSizes(DeprecatedTypeProperties& T) {
  auto b = randn({5, 3}, T);  // 生成一个大小为 (5, 3) 的随机张量 b
  auto c = randn({3, 7}, T);  // 生成一个大小为 (3, 7) 的随机张量 c
  auto a = randn({3, 3}, T);  // 生成一个大小为 (3, 3) 的随机张量 a
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言：尝试对 a 执行显式维度矩阵乘加操作，期望抛出异常
  ASSERT_ANY_THROW(a.addmm(b, c));
}
```