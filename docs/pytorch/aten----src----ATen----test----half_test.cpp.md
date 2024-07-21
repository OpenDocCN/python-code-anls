# `.\pytorch\aten\src\ATen\test\half_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/ATen.h>  // 引入 ATen 库
#include <ATen/test/test_assert.h>  // 引入 ATen 的测试断言头文件
#include <cmath>  // 引入数学函数库
#include <iostream>  // 引入输入输出流库
#include <limits>  // 引入数值极限库
#include <sstream>  // 引入字符串流库
#include <type_traits>  // 引入类型特性库

using namespace at;  // 使用 at 命名空间

// 定义 Half 类型的单元测试 TestHalf，测试算术运算
TEST(TestHalf, Arithmetic) {
  Half zero = 0;  // 创建 Half 类型变量 zero，并赋值为 0
  Half one = 1;  // 创建 Half 类型变量 one，并赋值为 1

  // 断言：zero 加 one 等于 one
  ASSERT_EQ(zero + one, one);
  // 断言：zero 加 zero 等于 zero
  ASSERT_EQ(zero + zero, zero);
  // 断言：zero 乘 one 等于 zero
  ASSERT_EQ(zero * one, zero);
  // 断言：one 乘 one 等于 one
  ASSERT_EQ(one * one, one);
  // 断言：one 除以 one 等于 one
  ASSERT_EQ(one / one, one);
  // 断言：one 减 one 等于 zero
  ASSERT_EQ(one - one, zero);
  // 断言：one 减 zero 等于 one
  ASSERT_EQ(one - zero, one);
  // 断言：zero 减 one 等于 -one
  ASSERT_EQ(zero - one, -one);
  // 断言：one 加 one 等于 Half(2)
  ASSERT_EQ(one + one, Half(2));
  // 断言：one 加 one 等于 2
  ASSERT_EQ(one + one, 2);
}

// 定义 Half 类型的单元测试 TestHalf，测试比较操作
TEST(TestHalf, Comparisions) {
  Half zero = 0;  // 创建 Half 类型变量 zero，并赋值为 0
  Half one = 1;  // 创建 Half 类型变量 one，并赋值为 1

  // 断言：zero 小于 one
  ASSERT_LT(zero, one);
  // 断言：zero 小于 1
  ASSERT_LT(zero, 1);
  // 断言：1 大于 zero
  ASSERT_GT(1, zero);
  // 断言：0 大于等于 zero
  ASSERT_GE(0, zero);
  // 断言：0 不等于 one
  ASSERT_NE(0, one);
  // 断言：zero 等于 0
  ASSERT_EQ(zero, 0);
  // 断言：zero 等于 zero
  ASSERT_EQ(zero, zero);
  // 断言：zero 等于 -zero
  ASSERT_EQ(zero, -zero);
}

// 定义 Half 类型的单元测试 TestHalf，测试类型转换
TEST(TestHalf, Cast) {
  Half value = 1.5f;  // 创建 Half 类型变量 value，并赋值为 1.5

  // 断言：value 转换为 int 等于 1
  ASSERT_EQ((int)value, 1);
  // 断言：value 转换为 short 等于 1
  ASSERT_EQ((short)value, 1);
  // 断言：value 转换为 long long 等于 1LL
  ASSERT_EQ((long long)value, 1LL);
  // 断言：value 转换为 float 等于 1.5f
  ASSERT_EQ((float)value, 1.5f);
  // 断言：value 转换为 double 等于 1.5
  ASSERT_EQ((double)value, 1.5);
  // 断言：value 转换为 bool 等于 true
  ASSERT_EQ((bool)value, true);
  // 断言：Half(0.0f) 转换为 bool 等于 false
  ASSERT_EQ((bool)Half(0.0f), false);
}

// 定义 Half 类型的单元测试 TestHalf，测试构造函数
TEST(TestHalf, Construction) {
  // 断言：Half((short)3) 等于 Half(3.0f)
  ASSERT_EQ(Half((short)3), Half(3.0f));
  // 断言：Half((unsigned short)3) 等于 Half(3.0f)
  ASSERT_EQ(Half((unsigned short)3), Half(3.0f));
  // 断言：Half(3) 等于 Half(3.0f)
  ASSERT_EQ(Half(3), Half(3.0f));
  // 断言：Half(3U) 等于 Half(3.0f)
  ASSERT_EQ(Half(3U), Half(3.0f));
  // 断言：Half(3LL) 等于 Half(3.0f)
  ASSERT_EQ(Half(3LL), Half(3.0f));
  // 断言：Half(3ULL) 等于 Half(3.0f)
  ASSERT_EQ(Half(3ULL), Half(3.0f));
  // 断言：Half(3.5) 等于 Half(3.5f)
  ASSERT_EQ(Half(3.5), Half(3.5f));
}

// 定义将 Half 类型转换为字符串的辅助函数
static std::string to_string(const Half& h) {
  std::stringstream ss;  // 创建字符串流 ss
  ss << h;  // 将 Half 类型 h 插入到 ss 中
  return ss.str();  // 返回 ss 的字符串表示
}

// 定义 Half 类型的单元测试 TestHalf，测试 Half 类型转换为字符串
TEST(TestHalf, Half2String) {
  // 断言：to_string(Half(3.5f)) 等于 "3.5"
  ASSERT_EQ(to_string(Half(3.5f)), "3.5");
  // 断言：to_string(Half(-100.0f)) 等于 "-100"
  ASSERT_EQ(to_string(Half(-100.0f)), "-100");
}

// 定义 Half 类型的单元测试 TestHalf，测试 Half 类型的数值极限
TEST(TestHalf, HalfNumericLimits) {
  using limits = std::numeric_limits<Half>;  // 创建 Half 类型的数值极限对象 limits

  // 断言：limits::lowest() 等于 -65504.0f
  ASSERT_EQ(limits::lowest(), -65504.0f);
  // 断言：limits::max() 等于 65504.0f
  ASSERT_EQ(limits::max(), 65504.0f);
  // 断言：limits::min() 大于 0
  ASSERT_GT(limits::min(), 0);
  // 断言：limits::min() 小于 1
  ASSERT_LT(limits::min(), 1);
  // 断言：limits::denorm_min() 大于 0
  ASSERT_GT(limits::denorm_min(), 0);
  // 断言：limits::denorm_min() 的一半等于 0
  ASSERT_EQ(limits::denorm_min() / 2, 0);
  // 断言：limits::infinity() 等于 std::numeric_limits<float>::infinity()
  ASSERT_EQ(limits::infinity(), std::numeric_limits<float>::infinity());
  // 断言：limits::quiet_NaN() 不等于自身
  ASSERT_NE(limits::quiet_NaN(), limits::quiet_NaN());
  // 断言：limits::signaling_NaN() 不等于自身
  ASSERT_NE(limits::signaling_NaN(), limits::signaling_NaN());
}

// 检查 numeric_limits<Half> 成员的声明类型是否与 numeric_limits<float> 上的声明类型相同
#define ASSERT_SAME_TYPE(name)                         \
  static_assert(                                       \
      std::is_same_v<                                  \
          decltype(std::numeric_limits<Half>::name),   \
          decltype(std::numeric_limits<float>::name)>, \
      "decltype(" #name ") differs")

// 断言：验证各个 numeric_limits<Half> 成员的声明类型与 numeric_limits<float> 上的相同
ASSERT_SAME_TYPE(is_specialized);
ASSERT_SAME_TYPE(is_signed);
ASSERT_SAME_TYPE(is_integer);
ASSERT_SAME_TYPE(is_exact);
ASSERT_SAME_TYPE(has_infinity);
ASSERT_SAME_TYPE(has_quiet_NaN);
ASSERT
// 断言确保变量类型与预期类型相同：digits
ASSERT_SAME_TYPE(digits);

// 断言确保变量类型与预期类型相同：digits10
ASSERT_SAME_TYPE(digits10);

// 断言确保变量类型与预期类型相同：max_digits10
ASSERT_SAME_TYPE(max_digits10);

// 断言确保变量类型与预期类型相同：radix
ASSERT_SAME_TYPE(radix);

// 断言确保变量类型与预期类型相同：min_exponent
ASSERT_SAME_TYPE(min_exponent);

// 断言确保变量类型与预期类型相同：min_exponent10
ASSERT_SAME_TYPE(min_exponent10);

// 断言确保变量类型与预期类型相同：max_exponent
ASSERT_SAME_TYPE(max_exponent);

// 断言确保变量类型与预期类型相同：max_exponent10
ASSERT_SAME_TYPE(max_exponent10);

// 断言确保变量类型与预期类型相同：traps
ASSERT_SAME_TYPE(traps);

// 断言确保变量类型与预期类型相同：tinyness_before
ASSERT_SAME_TYPE(tinyness_before);

// 单元测试函数，测试半精度浮点数在常见数学函数上的行为
TEST(TestHalf, CommonMath) {
    // 如果处于调试模式，设置阈值
#ifndef NDEBUG
    float threshold = 0.00001;
#endif

    // 断言半精度浮点数和单精度浮点数在 lgamma 函数上的差异小于等于阈值
    assert(std::abs(std::lgamma(Half(10.0)) - std::lgamma(10.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 exp 函数上的差异小于等于阈值
    assert(std::abs(std::exp(Half(1.0)) - std::exp(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 log 函数上的差异小于等于阈值
    assert(std::abs(std::log(Half(1.0)) - std::log(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 log10 函数上的差异小于等于阈值
    assert(std::abs(std::log10(Half(1000.0)) - std::log10(1000.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 log1p 函数上的差异小于等于阈值
    assert(std::abs(std::log1p(Half(0.0)) - std::log1p(0.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 log2 函数上的差异小于等于阈值
    assert(std::abs(std::log2(Half(1000.0)) - std::log2(1000.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 expm1 函数上的差异小于等于阈值
    assert(std::abs(std::expm1(Half(1.0)) - std::expm1(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 cos 函数上的差异小于等于阈值
    assert(std::abs(std::cos(Half(0.0)) - std::cos(0.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 sin 函数上的差异小于等于阈值
    assert(std::abs(std::sin(Half(0.0)) - std::sin(0.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 sqrt 函数上的差异小于等于阈值
    assert(std::abs(std::sqrt(Half(100.0)) - std::sqrt(100.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 ceil 函数上的差异小于等于阈值
    assert(std::abs(std::ceil(Half(2.4)) - std::ceil(2.4f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 floor 函数上的差异小于等于阈值
    assert(std::abs(std::floor(Half(2.7)) - std::floor(2.7f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 trunc 函数上的差异小于等于阈值
    assert(std::abs(std::trunc(Half(2.7)) - std::trunc(2.7f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 acos 函数上的差异小于等于阈值
    assert(std::abs(std::acos(Half(-1.0)) - std::acos(-1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 cosh 函数上的差异小于等于阈值
    assert(std::abs(std::cosh(Half(1.0)) - std::cosh(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 acosh 函数上的差异小于等于阈值
    assert(std::abs(std::acosh(Half(1.0)) - std::acosh(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 asin 函数上的差异小于等于阈值
    assert(std::abs(std::asin(Half(1.0)) - std::asin(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 sinh 函数上的差异小于等于阈值
    assert(std::abs(std::sinh(Half(1.0)) - std::sinh(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 asinh 函数上的差异小于等于阈值
    assert(std::abs(std::asinh(Half(1.0)) - std::asinh(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 tan 函数上的差异小于等于阈值
    assert(std::abs(std::tan(Half(0.0)) - std::tan(0.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 atan 函数上的差异小于等于阈值
    assert(std::abs(std::atan(Half(1.0)) - std::atan(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 tanh 函数上的差异小于等于阈值
    assert(std::abs(std::tanh(Half(1.0)) - std::tanh(1.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 erf 函数上的差异小于等于阈值
    assert(std::abs(std::erf(Half(10.0)) - std::erf(10.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点数在 erfc 函数上的差异小于等于阈值
    assert(std::abs(std::erfc(Half(10.0)) - std::erfc(10.0f)) <= threshold);

    // 断言半精度浮点数和单精度浮点
TEST(TestHalf, ComplexHalf) {
  // 定义一个 Half 类型的实数部分，赋值为 3.0f
  Half real = 3.0f;
  // 定义一个 Half 类型的虚数部分，赋值为 -10.0f
  Half imag = -10.0f;
  // 使用实数和虚数部分创建一个复数对象
  auto complex = c10::complex<Half>(real, imag);
  // 断言复数对象的实部等于预期的实数部分
  ASSERT_EQ(complex.real(), real);
  // 断言复数对象的虚部等于预期的虚数部分
  ASSERT_EQ(complex.imag(), imag);
}
```