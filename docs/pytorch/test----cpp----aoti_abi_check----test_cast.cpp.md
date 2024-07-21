# `.\pytorch\test\cpp\aoti_abi_check\test_cast.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <c10/util/TypeCast.h>  // 包含类型转换相关的头文件
#include <c10/util/bit_cast.h>  // 包含位级别类型转换相关的头文件

namespace torch {
namespace aot_inductor {

TEST(TestCast, TestConvert) {  // 定义名为 TestConvert 的测试用例
  c10::BFloat16 a = 3.0f;  // 声明并初始化一个 c10::BFloat16 类型的变量 a，值为 3.0f
  c10::Half b = 3.0f;  // 声明并初始化一个 c10::Half 类型的变量 b，值为 3.0f

  EXPECT_EQ(c10::convert<c10::Half>(a), b);  // 断言：将 a 转换为 c10::Half 类型后应等于 b
  EXPECT_EQ(a, c10::convert<c10::BFloat16>(b));  // 断言：将 b 转换为 c10::BFloat16 类型后应等于 a
}

TEST(TestCast, TestBitcast) {  // 定义名为 TestBitcast 的测试用例
  c10::BFloat16 a = 3.0f;  // 声明并初始化一个 c10::BFloat16 类型的变量 a，值为 3.0f
  c10::Half b = 3.0f;  // 声明并初始化一个 c10::Half 类型的变量 b，值为 3.0f

  EXPECT_EQ(c10::bit_cast<c10::BFloat16>(c10::bit_cast<c10::Half>(a)), a);  // 断言：a 经过两次 bit_cast 后应等于原始的 a
  EXPECT_EQ(c10::bit_cast<c10::Half>(c10::bit_cast<c10::BFloat16>(b)), b);  // 断言：b 经过两次 bit_cast 后应等于原始的 b
}

} // namespace aot_inductor
} // namespace torch
```