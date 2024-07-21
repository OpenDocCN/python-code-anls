# `.\pytorch\test\cpp\aoti_abi_check\test_dtype.cpp`

```
#include <gtest/gtest.h>

#include <c10/util/BFloat16-math.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>

namespace torch {
namespace aot_inductor {

// 定义单元测试类 TestDtype，测试 c10::BFloat16 类型
TEST(TestDtype, TestBFloat16) {
  // 创建并初始化 c10::BFloat16 类型变量 a, b
  c10::BFloat16 a = 1.0f;
  c10::BFloat16 b = 2.0f;
  // 创建并初始化 c10::BFloat16 类型变量 add, sub, mul, div
  c10::BFloat16 add = 3.0f;
  c10::BFloat16 sub = -1.0f;
  c10::BFloat16 mul = 2.0f;
  c10::BFloat16 div = 0.5f;

  // 验证加法操作的期望结果
  EXPECT_EQ(a + b, add);
  // 验证减法操作的期望结果
  EXPECT_EQ(a - b, sub);
  // 验证乘法操作的期望结果
  EXPECT_EQ(a * b, mul);
  // 验证除法操作的期望结果
  EXPECT_EQ(a / b, div);
}

// 定义单元测试类 TestDtype，测试 c10::Float8_e4m3fn 类型
TEST(TestDtype, TestFloat8_e4m3fn) {
  // 创建并初始化 c10::Float8_e4m3fn 类型变量 a, b
  c10::Float8_e4m3fn a = 1.0f;
  c10::Float8_e4m3fn b = 2.0f;
  // 创建并初始化 c10::Float8_e4m3fn 类型变量 add, sub, mul, div
  c10::Float8_e4m3fn add = 3.0f;
  c10::Float8_e4m3fn sub = -1.0f;
  c10::Float8_e4m3fn mul = 2.0f;
  c10::Float8_e4m3fn div = 0.5f;

  // 验证加法操作的期望结果
  EXPECT_EQ(a + b, add);
  // 验证减法操作的期望结果
  EXPECT_EQ(a - b, sub);
  // 验证乘法操作的期望结果
  EXPECT_EQ(a * b, mul);
  // 验证除法操作的期望结果
  EXPECT_EQ(a / b, div);
}

// 定义单元测试类 TestDtype，测试 c10::Float8_e4m3fnuz 类型
TEST(TestDtype, TestFloat8_e4m3fuz) {
  // 创建并初始化 c10::Float8_e4m3fnuz 类型变量 a, b
  c10::Float8_e4m3fnuz a = 1.0f;
  c10::Float8_e4m3fnuz b = 2.0f;
  // 创建并初始化 c10::Float8_e4m3fnuz 类型变量 add, sub, mul, div
  c10::Float8_e4m3fnuz add = 3.0f;
  c10::Float8_e4m3fnuz sub = -1.0f;
  c10::Float8_e4m3fnuz mul = 2.0f;
  c10::Float8_e4m3fnuz div = 0.5f;

  // 验证加法操作的期望结果
  EXPECT_EQ(a + b, add);
  // 验证减法操作的期望结果
  EXPECT_EQ(a - b, sub);
  // 验证乘法操作的期望结果
  EXPECT_EQ(a * b, mul);
  // 验证除法操作的期望结果
  EXPECT_EQ(a / b, div);
}

// 定义单元测试类 TestDtype，测试 c10::Float8_e5m2 类型
TEST(TestDtype, TestFloat8_e5m2) {
  // 创建并初始化 c10::Float8_e5m2 类型变量 a, b
  c10::Float8_e5m2 a = 1.0f;
  c10::Float8_e5m2 b = 2.0f;
  // 创建并初始化 c10::Float8_e5m2 类型变量 add, sub, mul, div
  c10::Float8_e5m2 add = 3.0f;
  c10::Float8_e5m2 sub = -1.0f;
  c10::Float8_e5m2 mul = 2.0f;
  c10::Float8_e5m2 div = 0.5f;

  // 验证加法操作的期望结果
  EXPECT_EQ(a + b, add);
  // 验证减法操作的期望结果
  EXPECT_EQ(a - b, sub);
  // 验证乘法操作的期望结果
  EXPECT_EQ(a * b, mul);
  // 验证除法操作的期望结果
  EXPECT_EQ(a / b, div);
}

// 定义单元测试类 TestDtype，测试 c10::Float8_e5m2fnuz 类型
TEST(TestDtype, TestFloat8_e5m2fnuz) {
  // 创建并初始化 c10::Float8_e5m2fnuz 类型变量 a, b
  c10::Float8_e5m2fnuz a = 1.0f;
  c10::Float8_e5m2fnuz b = 2.0f;
  // 创建并初始化 c10::Float8_e5m2fnuz 类型变量 add, sub, mul, div
  c10::Float8_e5m2fnuz add = 3.0f;
  c10::Float8_e5m2fnuz sub = -1.0f;
  c10::Float8_e5m2fnuz mul = 2.0f;
  c10::Float8_e5m2fnuz div = 0.5f;

  // 验证加法操作的期望结果
  EXPECT_EQ(a + b, add);
  // 验证减法操作的期望结果
  EXPECT_EQ(a - b, sub);
  // 验证乘法操作的期望结果
  EXPECT_EQ(a * b, mul);
  // 验证除法操作的期望结果
  EXPECT_EQ(a / b, div);
}

// 定义单元测试类 TestDtype，测试 c10::Half 类型
TEST(TestDtype, TestHalf) {
  // 创建并初始化 c10::Half 类型变量 a, b
  c10::Half a = 1.0f;
  c10::Half b = 2.0f;
  // 创建并初始化 c10::Half 类型变量 add, sub, mul, div
  c10::Half add = 3.0f;
  c10::Half sub = -1.0f;
  c10::Half mul = 2.0f;
  c10::Half div = 0.5f;

  // 验证加法操作的期望结果
  EXPECT_EQ(a + b, add);
  // 验证减法操作的期望结果
  EXPECT_EQ(a - b, sub);
  // 验证乘法操作的期望结果
  EXPECT_EQ(a * b, mul);
  // 验证除法操作的期望结果
  EXPECT_EQ(a / b, div);
}

} // namespace aot_inductor
} // namespace torch
```