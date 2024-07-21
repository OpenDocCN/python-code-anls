# `.\pytorch\c10\test\core\SymInt_test.cpp`

```py
#include <gtest/gtest.h>

#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Macros.h>

using namespace c10;

#ifndef C10_MOBILE
// 定义静态函数，用于检查 SymInt 对象与整数值的关系
static void check(int64_t value) {
  // 创建 SymInt 对象 i，用给定的整数值初始化
  const auto i = SymInt(value);
  // 断言 SymInt 对象 i 可能表示的整数值等于给定的值
  EXPECT_EQ(i.maybe_as_int(), c10::make_optional(value));
}

// 单元测试 SymInt 类的 ConcreteInts 方法
TEST(SymIntTest, ConcreteInts) {
  // 检查 SymInt 对象能否正确处理 INT64_MAX
  check(INT64_MAX);
  // 检查 SymInt 对象能否正确处理 0
  check(0);
  // 检查 SymInt 对象能否正确处理 -1
  check(-1);
  // 检查 SymInt 对象能否正确处理一个极小的负数
  check(-4611686018427387904LL);
  // 检查 SymInt 对象能否正确处理 INT64_MIN
  check(INT64_MIN);
}

// 单元测试 SymInt 类的 CheckRange 方法
TEST(SymIntTest, CheckRange) {
  // 断言 SymInt::check_range 方法返回 false，检查 INT64_MIN 是否在有效范围内
  EXPECT_FALSE(SymInt::check_range(INT64_MIN));
}

#if !C10_UBSAN_ENABLED
// 由于 signed-integer-overflow UBSAN 检查失败，此测试可能会失败
// 单元测试 SymInt 类的 Overflows 方法
TEST(SymIntTest, Overflows) {
  // 创建 SymInt 对象 x，并对其进行操作以验证溢出
  const auto x = SymInt(INT64_MAX);
  // 断言 -(x + 1) 不等于 0，验证溢出的情况
  EXPECT_NE(-(x + 1), 0);

  // 创建 SymInt 对象 y，并对其进行操作以验证溢出
  const auto y = SymInt(INT64_MIN);
  // 断言 -y 不等于 0，验证溢出的情况
  EXPECT_NE(-y, 0);
  // 断言 0 - y 不等于 0，验证溢出的情况
  EXPECT_NE(0 - y, 0);
}
#endif

#endif
```