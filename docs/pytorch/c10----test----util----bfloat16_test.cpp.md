# `.\pytorch\c10\test\util\bfloat16_test.cpp`

```py
// 包含BFloat16类型定义和相关数学函数头文件
#include <c10/util/BFloat16.h>
#include <c10/util/BFloat16-math.h>
#include <c10/util/irange.h>
// 包含Google测试框架的头文件
#include <gtest/gtest.h>

// 匿名命名空间，定义测试用例
namespace {

// 根据给定的符号、指数和尾数构造32位浮点数表示的函数
float float_from_bytes(uint32_t sign, uint32_t exponent, uint32_t fraction) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t bytes;
  bytes = 0;
  bytes |= sign;
  bytes <<= 8;
  bytes |= exponent;
  bytes <<= 23;
  bytes |= fraction;

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float res;
  // 将构造好的32位表示的浮点数拷贝到res中
  std::memcpy(&res, &bytes, sizeof(res));
  return res;
}

// 测试BFloat16类型的转换和逆转换
TEST(BFloat16Conversion, FloatToBFloat16AndBack) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  float in[100];
  // 初始化in数组
  for (const auto i : c10::irange(100)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers)
    in[i] = i + 1.25;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  c10::BFloat16 bfloats[100];
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  float out[100];

  // 遍历输入数组，进行BFloat16转换和逆转换，并验证误差
  for (const auto i : c10::irange(100)) {
    // 将浮点数转换为BFloat16格式
    bfloats[i].x = c10::detail::bits_from_f32(in[i]);
    // 将BFloat16格式转换回浮点数格式
    out[i] = c10::detail::f32_from_bits(bfloats[i].x);

    // 验证相对误差是否小于1/(2^7)，因为BFloat16有7位尾数
    EXPECT_LE(std::fabs(out[i] - in[i]) / in[i], 1.0 / 128);
  }
}

// 测试BFloat16类型的转换和逆转换（舍入到最近偶数）
TEST(BFloat16Conversion, FloatToBFloat16RNEAndBack) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  float in[100];
  // 初始化in数组
  for (const auto i : c10::irange(100)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers)
    in[i] = i + 1.25;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  c10::BFloat16 bfloats[100];
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  float out[100];

  // 遍历输入数组，进行BFloat16转换和逆转换（舍入到最近偶数），并验证误差
  for (const auto i : c10::irange(100)) {
    // 将浮点数舍入为最近偶数的BFloat16格式
    bfloats[i].x = c10::detail::round_to_nearest_even(in[i]);
    // 将BFloat16格式转换回浮点数格式
    out[i] = c10::detail::f32_from_bits(bfloats[i].x);

    // 验证相对误差是否小于1/(2^7)，因为BFloat16有7位尾数
    EXPECT_LE(std::fabs(out[i] - in[i]) / in[i], 1.0 / 128);
  }
}

// 测试NaN值的处理
TEST(BFloat16Conversion, NaN) {
  // 创建一个NaN的浮点数
  float inNaN = float_from_bytes(0, 0xFF, 0x7FFFFF);
  // 验证inNaN是NaN
  EXPECT_TRUE(std::isnan(inNaN));

  // 将NaN的浮点数转换为BFloat16类型
  c10::BFloat16 a = c10::BFloat16(inNaN);
  // 从BFloat16格式转换回浮点数
  float out = c10::detail::f32_from_bits(a.x);

  // 验证转换后的浮点数仍然是NaN
  EXPECT_TRUE(std::isnan(out));
}

} // namespace
TEST(BFloat16Conversion, Inf) {
  // 创建一个浮点数表示正无穷大
  float inInf = float_from_bytes(0, 0xFF, 0);
  // 断言输入值为正无穷大
  EXPECT_TRUE(std::isinf(inInf));

  // 将浮点数转换为BFloat16类型
  c10::BFloat16 a = c10::BFloat16(inInf);
  // 从BFloat16类型获取浮点数表示
  float out = c10::detail::f32_from_bits(a.x);

  // 断言输出值为正无穷大
  EXPECT_TRUE(std::isinf(out));
}

TEST(BFloat16Conversion, SmallestDenormal) {
  // 获取浮点数表示的最小非零次正规化数
  float in = std::numeric_limits<float>::denorm_min();
  c10::BFloat16 a = c10::BFloat16(in);
  // 从BFloat16类型获取浮点数表示
  float out = c10::detail::f32_from_bits(a.x);

  // 断言输入值与输出值相等
  EXPECT_FLOAT_EQ(in, out);
}

TEST(BFloat16Math, Addition) {
  // 这个测试验证了在加法后如果只改变了浮点数的前7位尾数，应该不会失去精度。

  // 输入的浮点数表示
  // S | 指数 | 尾数
  // 0 | 10000000 | 10010000000000000000000 = 3.125
  float input = float_from_bytes(0, 0, 0x40480000);

  // 预期的浮点数表示
  // S | 指数 | 尾数
  // 0 | 10000001 | 10010000000000000000000 = 6.25
  float expected = float_from_bytes(0, 0, 0x40c80000);

  // 创建一个BFloat16对象
  c10::BFloat16 b;
  // 将输入浮点数表示转换为BFloat16的位表示
  b.x = c10::detail::bits_from_f32(input);
  // BFloat16对象自加
  b = b + b;

  // 从BFloat16类型获取浮点数表示
  float res = c10::detail::f32_from_bits(b.x);
  // 断言结果与预期相等
  EXPECT_EQ(res, expected);
}

TEST(BFloat16Math, Subtraction) {
  // 这个测试验证了在减法后如果只改变了浮点数的前7位尾数，应该不会失去精度。

  // 输入的浮点数表示
  // S | 指数 | 尾数
  // 0 | 10000001 | 11101000000000000000000 = 7.625
  float input = float_from_bytes(0, 0, 0x40f40000);

  // 预期的浮点数表示
  // S | 指数 | 尾数
  // 0 | 10000000 | 01010000000000000000000 = 2.625
  float expected = float_from_bytes(0, 0, 0x40280000);

  // 创建一个BFloat16对象
  c10::BFloat16 b;
  // 将输入浮点数表示转换为BFloat16的位表示
  b.x = c10::detail::bits_from_f32(input);
  // BFloat16对象自减
  b = b - 5;

  // 从BFloat16类型获取浮点数表示
  float res = c10::detail::f32_from_bits(b.x);
  // 断言结果与预期相等
  EXPECT_EQ(res, expected);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BFloat16Math, NextAfterZero) {
  // 创建一个BFloat16对象表示零
  const c10::BFloat16 zero{0};

  auto check_nextafter =
      [](c10::BFloat16 from, c10::BFloat16 to, c10::BFloat16 expected) {
        // 调用std::nextafter函数，获取实际值
        c10::BFloat16 actual = std::nextafter(from, to);
        // 检查实际值与预期值是否位相等
        ASSERT_EQ(actual.x ^ expected.x, uint16_t{0});
      };
  // 检查从零到零的下一个值
  check_nextafter(zero, zero, /*expected=*/zero);
  // 检查从零到负零的下一个值
  check_nextafter(zero, -zero, /*expected=*/-zero);
  // 检查从负零到零的下一个值
  check_nextafter(-zero, zero, /*expected=*/zero);
  // 检查从负零到负零的下一个值
  check_nextafter(-zero, -zero, /*expected=*/-zero);
}

float BinaryToFloat(uint32_t bytes) {
  // 创建一个浮点数表示，通过将字节拷贝到浮点数内存中
  float res;
  std::memcpy(&res, &bytes, sizeof(res));
  return res;
}

struct BFloat16TestParam {
  uint32_t input;   // 输入的32位无符号整数
  uint16_t rne;     // RNE（Round to Nearest Even，最近偶数舍入）的16位无符号整数
};

class BFloat16Test : public ::testing::Test,
                     public ::testing::WithParamInterface<BFloat16TestParam> {};
// 定义一个测试用例 BFloat16RNETest，使用 Google Test 框架的 TEST_P 宏
TEST_P(BFloat16Test, BFloat16RNETest) {
    // 从测试参数中获取输入值，并将二进制表示转换为单精度浮点数
    float value = BinaryToFloat(GetParam().input);
    // 调用 c10::detail 命名空间中的函数，将浮点数 value 近似为最接近的偶数表示，并转换为无符号 16 位整数
    uint16_t rounded = c10::detail::round_to_nearest_even(value);
    // 验证预期值 GetParam().rne 与 rounded 是否相等
    EXPECT_EQ(GetParam().rne, rounded);
}

// 实例化 BFloat16Test 测试套件，使用 Google Test 框架的 INSTANTIATE_TEST_SUITE_P 宏
INSTANTIATE_TEST_SUITE_P(
    // 测试实例命名为 BFloat16TestInstantiation
    BFloat16TestInstantiation,
    // 使用 BFloat16Test 测试类作为被实例化的测试
    BFloat16Test,
    // 提供一组测试参数，每组包含一个输入值和一个期望的输出值
    ::testing::Values(
        BFloat16TestParam{0x3F848000, 0x3F84},
        BFloat16TestParam{0x3F848010, 0x3F85},
        BFloat16TestParam{0x3F850000, 0x3F85},
        BFloat16TestParam{0x3F858000, 0x3F86},
        BFloat16TestParam{0x3FFF8000, 0x4000}));

// 结束命名空间定义
} // namespace
```