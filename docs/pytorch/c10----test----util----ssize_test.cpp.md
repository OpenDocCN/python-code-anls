# `.\pytorch\c10\test\util\ssize_test.cpp`

```
// 包含 ssize.h 文件，该文件提供了 ssize 函数的定义
#include <c10/util/ssize.h>

// 包含 Google Mock 和 Google Test 的头文件
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// 包含必要的标准库头文件
#include <cstdint>
#include <limits>

// 定义 c10 命名空间
namespace c10 {
// 定义匿名命名空间，用于隐藏类和函数的实现细节
namespace {

// 模板类 Container，用于存储特定类型的大小信息
template <typename size_type_>
class Container {
 public:
  // 使用别名定义 size_type 为 size_type_，便于后续引用
  using size_type = size_type_;

  // 构造函数，显式地初始化容器的大小
  constexpr explicit Container(size_type size) : size_(size) {}

  // 返回容器的大小
  constexpr auto size() const noexcept -> size_type {
    return size_;
  }

 private:
  // 私有成员变量，存储容器的大小
  size_type size_;
};

// 单元测试定义，验证 ssize 函数对 size_t 类型的处理是否正确
TEST(ssizeTest, size_t) {
  // 使用 ssize 函数计算容器的大小，并断言其结果与预期相等
  ASSERT_THAT(ssize(Container(std::size_t{3})), testing::Eq(std::ptrdiff_t{3}));
}

// 单元测试定义，验证 ssize 函数对 size_t 类型溢出情况的处理
TEST(ssizeTest, size_t_overflow) {
  // 如果处于发布模式，则跳过该测试，并输出相关信息
#if defined(NDEBUG)
  GTEST_SKIP() << "Only valid if assert is enabled." << std::endl;
#endif

  // 计算 std::ptrdiff_t 类型的最大值
  constexpr auto ptrdiff_t_max =
      std::size_t{std::numeric_limits<std::ptrdiff_t>::max()};
  // 静态断言，确保 ptrdiff_t_max 小于 std::size_t 的最大值，避免溢出
  static_assert(ptrdiff_t_max < std::numeric_limits<std::size_t>::max());
  // 断言调用 ssize 函数会抛出 c10::Error 异常，以验证溢出情况得到正确处理
  EXPECT_THROW(ssize(Container(ptrdiff_t_max + 1)), c10::Error);
}

// 单元测试定义，验证 ssize 函数对小容器类型提升为 std::ptrdiff_t 的情况
TEST(ssizeTest, small_container_promotes_to_ptrdiff_t) {
  // 调用 ssize 函数计算有符号大小，并使用静态断言确保其类型为 std::ptrdiff_t
  auto signed_size = ssize(Container(std::uint16_t{3}));
  static_assert(std::is_same_v<decltype(signed_size), std::ptrdiff_t>);
  // 断言 ssize 函数计算结果与预期相等
  ASSERT_THAT(signed_size, testing::Eq(3));
}

// 单元测试定义，验证 ssize 函数在 32 位平台上提升为 64 位整数的情况
TEST(ssizeTest, promotes_to_64_bit_on_32_bit_platform) {
  // 如果不是 32 位平台，则跳过该测试，并输出相关信息
  if (sizeof(std::intptr_t) != 4) {
    GTEST_SKIP() << "Only valid in 64-bits." << std::endl;
  }

  // 调用 ssize 函数计算有符号大小，并使用静态断言确保其类型为 std::int64_t
  auto signed_size = ssize(Container(std::uint64_t{3}));
  static_assert(std::is_same_v<decltype(signed_size), std::int64_t>);
  // 断言 ssize 函数计算结果与预期相等
  ASSERT_THAT(signed_size, testing::Eq(3));
}

} // namespace
} // namespace c10
```