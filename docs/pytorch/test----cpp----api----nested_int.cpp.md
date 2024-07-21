# `.\pytorch\test\cpp\api\nested_int.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/core/NestedIntSymNodeImpl.h>  // 包含了 NestedIntSymNodeImpl 类的头文件
#include <c10/core/SymInt.h>  // 包含了 SymInt 类的头文件
#include <c10/core/SymNodeImpl.h>  // 包含了 SymNodeImpl 类的头文件
#include <torch/torch.h>  // 包含了 PyTorch 的头文件

#include <test/cpp/api/support.h>  // 包含了测试支持函数的头文件

TEST(NestedIntTest, Comparisons) {  // 定义一个测试案例 NestedIntTest，测试比较操作
  auto a = c10::SymInt(  // 创建 SymInt 对象 a
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(1, 1)));  // 使用 NestedIntSymNodeImpl 创建 SymNode，再创建 SymInt
  auto b = c10::SymInt(  // 创建 SymInt 对象 b
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(1, 1)));  // 使用 NestedIntSymNodeImpl 创建 SymNode，再创建 SymInt
  auto c = c10::SymInt(  // 创建 SymInt 对象 c
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(2, 1)));  // 使用 NestedIntSymNodeImpl 创建 SymNode，再创建 SymInt
  auto d = c10::SymInt(3);  // 创建 SymInt 对象 d，直接传入整数值

  ASSERT_TRUE(a == a);  // 断言 a 等于 a
  ASSERT_TRUE(a == b);  // 断言 a 等于 b
  ASSERT_FALSE(a != a);  // 断言 a 不等于 a 为假
  ASSERT_FALSE(a != b);  // 断言 a 不等于 b 为假
  ASSERT_FALSE(a == c);  // 断言 a 等于 c 为假
  ASSERT_TRUE(a != c);  // 断言 a 不等于 c

  ASSERT_FALSE(a == d);  // 断言 a 等于 d 为假
  ASSERT_TRUE(a != d);  // 断言 a 不等于 d
  ASSERT_FALSE(d == a);  // 断言 d 等于 a 为假
  ASSERT_TRUE(d != a);  // 断言 d 不等于 a

  // ge
  ASSERT_TRUE(a >= a);  // 断言 a 大于等于 a
  ASSERT_TRUE(a >= b);  // 断言 a 大于等于 b
  ASSERT_TRUE(b >= a);  // 断言 b 大于等于 a
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a >= c), c10::Error);  // 预期抛出异常，因为 a 不大于等于 c
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c >= a), c10::Error);  // 预期抛出异常，因为 c 不大于等于 a
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c >= 3), c10::Error);  // 预期抛出异常，因为 c 不大于等于 3
  ASSERT_TRUE(c >= 2);  // 断言 c 大于等于 2
  ASSERT_TRUE(c >= 1);  // 断言 c 大于等于 1
  ASSERT_FALSE(1 >= c);  // 断言 1 大于等于 c 为假

  // lt
  ASSERT_FALSE(a < a);  // 断言 a 小于 a 为假
  ASSERT_FALSE(a < b);  // 断言 a 小于 b 为假
  ASSERT_FALSE(b < a);  // 断言 b 小于 a 为假
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a < c), c10::Error);  // 预期抛出异常，因为 a 不小于 c
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c < a), c10::Error);  // 预期抛出异常，因为 c 不小于 a
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(3 < a), c10::Error);  // 预期抛出异常，因为 3 不小于 a
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(2 < a), c10::Error);  // 预期抛出异常，因为 2 不小于 a
  ASSERT_TRUE(1 < a);  // 断言 1 小于 a

  // le
  ASSERT_TRUE(a <= a);  // 断言 a 小于等于 a
  ASSERT_TRUE(b <= a);  // 断言 b 小于等于 a
  ASSERT_TRUE(a <= b);  // 断言 a 小于等于 b
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a <= c), c10::Error);  // 预期抛出异常，因为 a 不小于等于 c
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c <= a), c10::Error);  // 预期抛出异常，因为 c 不小于等于 a
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(3 <= c), c10::Error);  // 预期抛出异常，因为 3 不小于等于 c
  ASSERT_TRUE(2 <= c);  // 断言 2 小于等于 c
  ASSERT_TRUE(1 <= c);  // 断言 1 小于等于 c
  ASSERT_FALSE(c <= 1);  // 断言 c 不小于等于 1

  // gt
  ASSERT_FALSE(a > a);  // 断言 a 大于 a 为假
  ASSERT_FALSE(b > a);  // 断言 b 大于 a 为假
  ASSERT_FALSE(a > b);  // 断言 a 大于 b 为假
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > c), c10::Error);  // 预期抛出异常，因为 a 不大于 c
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c > a), c10::Error);  // 预期抛出异常，因为 c 不大于 a
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > 3), c10::Error);  // 预期抛出异常，因为 a 不大于 3
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > 2), c10::Error);  // 预期抛出异常，因为 a 不大于 2
  ASSERT_TRUE(a > 1);  // 断言 a 大于 1
}
TEST(NestedIntTest, WithFactor) {
  // 创建一个包含值为1和因子为5的NestedIntSymNodeImpl对象，并使用其创建SymNode对象，再封装为SymInt对象a
  auto a = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(1, 5)));
  // 创建一个包含值为1和因子为10的NestedIntSymNodeImpl对象，并使用其创建SymNode对象，再封装为SymInt对象b
  auto b = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(1, 10)));
  // 断言a不等于b
  ASSERT_FALSE(a == b);
  // 断言a不大于等于b
  ASSERT_FALSE(a >= b);
  // 断言b大于等于a
  ASSERT_TRUE(b >= a);
  // 断言a小于等于b
  ASSERT_TRUE(a <= b);
  // 断言b不小于等于a
  ASSERT_FALSE(b <= a);
  // 断言a不等于b
  ASSERT_TRUE(a != b);
  // 断言a乘以2等于b
  ASSERT_TRUE(a * 2 == b);
  // 断言a乘以3大于等于b
  ASSERT_TRUE(a * 3 >= b);
  // 断言a乘以2等于2乘以a
  ASSERT_TRUE(a * 2 == 2 * a);
}
```