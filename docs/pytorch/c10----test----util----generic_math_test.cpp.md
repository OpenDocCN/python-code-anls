# `.\pytorch\c10\test\util\generic_math_test.cpp`

```
// 依赖声明，包含 C10 的通用数学库
#include <c10/util/generic_math.h>

// Google 测试框架的头文件
#include <gtest/gtest.h>

// C++ 标准数学库头文件
#include <cmath>

// 使用测试命名空间
using namespace ::testing;

// 测试用例：GenericMathTest，测试整数和浮点数的地板除法函数
TEST(GenericMathTest, div_floor_test) {
  // 测试浮点数地板除法，除数为0时返回无穷大
  EXPECT_EQ(c10::div_floor_floating(5., 0.), INFINITY);
  // 测试浮点数地板除法，正常情况下返回期望的整数结果
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., 2.), 2.);
  // 测试浮点数地板除法，负除数情况下返回期望的整数结果
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., -2.), -3.);
  // 测试整数地板除法，正常情况下返回期望的整数结果
  EXPECT_EQ(c10::div_floor_integer(5, 2), 2);
  // 测试整数地板除法，负除数情况下返回期望的整数结果
  EXPECT_EQ(c10::div_floor_integer(5, -2), -3);
}
```