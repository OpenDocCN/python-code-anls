# `.\pytorch\test\cpp\aoti_abi_check\test_math.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <ATen/NumericUtils.h>  // 包含 ATen 数字工具库的头文件
#include <c10/util/generic_math.h>  // 包含 c10 通用数学库的头文件
#include <cmath>  // 包含数学函数的头文件
namespace torch {
namespace aot_inductor {

TEST(TestMath, TestDivFloor) {
  EXPECT_EQ(c10::div_floor_floating(5., 0.), INFINITY);  // 测试浮点数除法向下取整，对于除以0的情况期望得到无穷大
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., 2.), 2.);  // 测试浮点数除法向下取整，对于除数为2的情况期望结果为2
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., -2.), -3.);  // 测试浮点数除法向下取整，对于除数为-2的情况期望结果为-3
  EXPECT_EQ(c10::div_floor_integer(5, 2), 2);  // 测试整数除法向下取整，对于5除以2期望结果为2
  EXPECT_EQ(c10::div_floor_integer(5, -2), -3);  // 测试整数除法向下取整，对于5除以-2期望结果为-3
}

TEST(TestMath, TestNan) {
  EXPECT_FALSE(at::_isnan(1.0));  // 测试是否不是 NaN，对于1.0期望结果为假
  EXPECT_TRUE(at::_isnan(std::nan("")));  // 测试是否是 NaN，传入空字符串以创建 NaN，期望结果为真
}

} // namespace aot_inductor
} // namespace torch
```