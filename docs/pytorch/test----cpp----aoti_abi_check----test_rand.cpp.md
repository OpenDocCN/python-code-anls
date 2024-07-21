# `.\pytorch\test\cpp\aoti_abi_check\test_rand.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <ATen/core/PhiloxRNGEngine.h>  // 包含 PyTorch 的随机数生成引擎头文件

#include <cstdint>       // 包含标准整数类型的头文件
#include <iostream>      // 包含标准输入输出流的头文件
namespace torch {       // 进入 torch 命名空间
namespace aot_inductor {  // 进入 aot_inductor 命名空间

int64_t randint64_cpu(   // 定义返回 int64_t 类型的函数 randint64_cpu，接受四个参数
    uint32_t seed,       // 第一个参数：种子值，无符号 32 位整数
    uint32_t offset,     // 第二个参数：偏移量，无符号 32 位整数
    int64_t low,         // 第三个参数：下界，有符号 64 位整数
    int64_t high) {      // 第四个参数：上界，有符号 64 位整数

  auto gen = at::Philox4_32(seed, 0, offset);  // 使用 Philox4_32 创建随机数生成器
  uint64_t r0 = gen();  // 生成随机数 r0
  uint64_t r1 = gen();  // 生成随机数 r1
  uint64_t result = r0 | (r1 << 32);  // 将 r0 和 r1 组合成 64 位随机数
  return static_cast<int64_t>(result % (high - low)) + low;  // 返回处于指定范围内的随机整数
}

TEST(TestRand, TestRandn) {  // 定义名为 TestRand 的测试套件，包含 TestRandn 测试用例
  at::Philox4_32 engine_1(1, 0, 0);  // 创建 Philox4_32 随机数生成器 engine_1
  float a = engine_1.randn(10);  // 使用 engine_1 生成服从标准正态分布的浮点数 a
  at::Philox4_32 engine_2(1, 0, 0);  // 创建另一个 Philox4_32 随机数生成器 engine_2
  float b = engine_2.randn(10);  // 使用 engine_2 生成服从标准正态分布的浮点数 b

  EXPECT_EQ(a, b);  // 断言 a 和 b 的值相等
}

TEST(TestRand, TestRandint64) {  // 定义名为 TestRand 的测试套件，包含 TestRandint64 测试用例
  int64_t a = randint64_cpu(0xffffffff, 100, 0, INT64_MAX);  // 调用 randint64_cpu 函数生成随机整数 a
  int64_t b = randint64_cpu(0xffffffff, 100, 0, INT64_MAX);  // 再次调用 randint64_cpu 函数生成随机整数 b

  EXPECT_EQ(a, b);  // 断言 a 和 b 的值相等
}

} // namespace aot_inductor  // 结束 aot_inductor 命名空间
} // namespace torch  // 结束 torch 命名空间
```