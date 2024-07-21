# `.\pytorch\c10\test\util\complex_math_test.cpp`

```
#include <gtest/gtest.h>
# 包含 Google Test 框架的头文件

#define C10_DEFINE_TEST(a, b) TEST(a, b)
# 定义宏 C10_DEFINE_TEST(a, b)，用于定义一个 Google Test 测试用例

#define C10_ASSERT_NEAR(a, b, tol) ASSERT_NEAR(a, b, tol)
# 定义宏 C10_ASSERT_NEAR(a, b, tol)，用于在测试中断言两个数值近似相等

#include <c10/test/util/complex_math_test_common.h>
# 包含用于复杂数学测试的 C10 库的头文件
```