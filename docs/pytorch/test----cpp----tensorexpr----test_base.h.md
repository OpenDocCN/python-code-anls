# `.\pytorch\test\cpp\tensorexpr\test_base.h`

```py
#pragma once
// 如果定义了 USE_GTEST 宏，则包含 GTest 测试框架的相关头文件
#if defined(USE_GTEST)
#include <gtest/gtest.h>
#include <test/cpp/common/support.h>
// 否则，包含 <cmath> 和一些 Torch 相关的异常处理和断言头文件
#else
#include <cmath>
#include "c10/util/Exception.h"
#include "test/cpp/tensorexpr/gtest_assert_float_eq.h"

// 定义 ASSERT_EQ 宏，用于断言两个值相等
#define ASSERT_EQ(x, y, ...) TORCH_INTERNAL_ASSERT((x) == (y), __VA_ARGS__)
// 定义 ASSERT_FLOAT_EQ 宏，用于断言两个浮点数近似相等
#define ASSERT_FLOAT_EQ(x, y, ...) \
  TORCH_INTERNAL_ASSERT(AlmostEquals((x), (y)), __VA_ARGS__)
// 定义 ASSERT_NE 宏，用于断言两个值不相等
#define ASSERT_NE(x, y, ...) TORCH_INTERNAL_ASSERT((x) != (y), __VA_ARGS__)
// 定义 ASSERT_GT 宏，用于断言前一个值大于后一个值
#define ASSERT_GT(x, y, ...) TORCH_INTERNAL_ASSERT((x) > (y), __VA_ARGS__)
// 定义 ASSERT_GE 宏，用于断言前一个值大于等于后一个值
#define ASSERT_GE(x, y, ...) TORCH_INTERNAL_ASSERT((x) >= (y), __VA_ARGS__)
// 定义 ASSERT_LT 宏，用于断言前一个值小于后一个值
#define ASSERT_LT(x, y, ...) TORCH_INTERNAL_ASSERT((x) < (y), __VA_ARGS__)
// 定义 ASSERT_LE 宏，用于断言前一个值小于等于后一个值
#define ASSERT_LE(x, y, ...) TORCH_INTERNAL_ASSERT((x) <= (y), __VA_ARGS__)

// 定义 ASSERT_NEAR 宏，用于断言两个值在一定误差范围内近似相等
#define ASSERT_NEAR(x, y, a, ...) \
  TORCH_INTERNAL_ASSERT(std::fabs((x) - (y)) < (a), __VA_ARGS__)

// 定义 ASSERT_TRUE 宏，用于断言表达式为真
#define ASSERT_TRUE TORCH_INTERNAL_ASSERT
// 定义 ASSERT_FALSE 宏，用于断言表达式为假
#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))
// 定义 ASSERT_THROWS_WITH 宏，用于断言某个语句抛出异常，并且异常信息包含指定的子字符串
#define ASSERT_THROWS_WITH(statement, substring)                         \
  try {                                                                  \
    (void)statement;                                                     \
    ASSERT_TRUE(false);                                                  \
  } catch (const std::exception& e) {                                    \
    ASSERT_NE(std::string(e.what()).find(substring), std::string::npos); \
  }
// 定义 ASSERT_ANY_THROW 宏，用于断言某个语句抛出任何异常
#define ASSERT_ANY_THROW(statement)     \
  {                                     \
    bool threw = false;                 \
    try {                               \
      (void)statement;                  \
    } catch (const std::exception& e) { \
      threw = true;                     \
    }                                   \
    ASSERT_TRUE(threw);                 \
  }

#endif // defined(USE_GTEST)

// 命名空间 torch::jit::tensorexpr 下的模板函数 ExpectAllNear，用于断言两个容器中的元素在指定的误差范围内近似相等
namespace torch {
namespace jit {
namespace tensorexpr {

// 模板函数，用于断言两个容器的所有元素在指定的误差范围内近似相等
template <typename U, typename V>
void ExpectAllNear(
    const std::vector<U>& v1,
    const std::vector<U>& v2,
    V threshold,
    const std::string& name = "") {
  // 断言两个容器的大小相等
  ASSERT_EQ(v1.size(), v2.size());
  // 遍历容器元素，断言每对元素在指定的误差范围内近似相等
  for (size_t i = 0; i < v1.size(); i++) {
    ASSERT_NEAR(v1[i], v2[i], threshold);
  }
}

// 模板函数，用于断言容器中所有元素与指定值在指定的误差范围内近似相等
template <typename U, typename V>
void ExpectAllNear(
    const std::vector<U>& vec,
    const U& val,
    V threshold,
    const std::string& name = "") {
  // 遍历容器元素，断言每个元素与指定值在指定的误差范围内近似相等
  for (size_t i = 0; i < vec.size(); i++) {
    ASSERT_NEAR(vec[i], val, threshold);
  }
}

// 静态模板函数，用于断言容器中所有元素与指定值相等
template <typename T>
static void assertAllEqual(const std::vector<T>& vec, const T& val) {
  // 遍历容器元素，断言每个元素与指定值相等
  for (auto const& elt : vec) {
    ASSERT_EQ(elt, val);
  }
}

// 静态模板函数，用于断言两个容器中所有元素相等
template <typename T>
static void assertAllEqual(const std::vector<T>& v1, const std::vector<T>& v2) {
  // 断言两个容器的大小相等
  ASSERT_EQ(v1.size(), v2.size());
  // 遍历容器元素，断言每对元素相等
  for (size_t i = 0; i < v1.size(); ++i) {
    ASSERT_EQ(v1[i], v2[i]);
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```