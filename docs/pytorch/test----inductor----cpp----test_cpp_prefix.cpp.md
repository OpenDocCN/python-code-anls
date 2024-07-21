# `.\pytorch\test\inductor\cpp\test_cpp_prefix.cpp`

```py
// 导入相对路径的 C++ 前缀头文件
#include "../../torchinductor/codegen/cpp_prefix.h"
// 导入 Google 测试框架的头文件
#include <gtest/gtest.h>

// 定义测试套件 testCppPrefix，并测试 atomic_add 函数对整数的原子加法操作
TEST(testCppPrefix, testAtomicAddInt) {
  // 定义整数变量 x，并初始化为 0
  int x = 0;
  // 调用 atomic_add 函数，对 x 执行原子加法操作，增加 100
  atomic_add(&x, 100);
  // 验证 x 的值是否等于 100
  EXPECT_EQ(x, 100);
}

// 定义测试套件 testCppPrefix，并测试 atomic_add 函数对浮点数的原子加法操作
TEST(testCppPrefix, testAtomicAddFloat) {
  // 定义浮点数变量 x，并初始化为 0.0
  float x = 0.0f;
  // 调用 atomic_add 函数，对 x 执行原子加法操作，增加 100.0
  atomic_add(&x, 100.0f);
  // 验证 x 的值是否等于 100.0
  EXPECT_EQ(x, 100.0f);
}

// 定义测试套件 testCppPrefix，并测试 atomic_add 函数对 int64_t 类型的原子加法操作
TEST(testCppPrefix, testAtomicAddI64) {
  // 定义 int64_t 类型变量 x 和 y，并初始化分别为 0 和 100
  int64_t x = 0.0;
  int64_t y = 100.0;
  // 调用 atomic_add 函数，对 x 执行原子加法操作，增加 y 的值
  atomic_add(&x, y);
  // 验证 x 的值是否等于 100
  EXPECT_EQ(x, 100);
}
```