# `.\pytorch\c10\test\util\exception_test.cpp`

```
#include <c10/util/Exception.h>
#include <gtest/gtest.h>
#include <stdexcept>

using c10::Error; // 使用 c10 命名空间中的 Error 类

namespace {

template <class Functor>
inline void expectThrowsEq(Functor&& functor, const char* expectedMessage) {
  try {
    std::forward<Functor>(functor)(); // 调用传入的 functor
  } catch (const Error& e) {
    EXPECT_STREQ(e.what_without_backtrace(), expectedMessage); // 检查异常信息是否符合预期
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \""
                << expectedMessage << "\" but didn't throw"; // 如果没有抛出异常，则测试失败
}
} // namespace

TEST(ExceptionTest, TORCH_INTERNAL_ASSERT_DEBUG_ONLY) {
#ifdef NDEBUG
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false)); // 在 NDEBUG 模式下，不应抛出异常
  // Does nothing - `throw ...` should not be evaluated
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (throw std::runtime_error("I'm throwing..."), true))); // 在 NDEBUG 模式下，应该不会评估 throw 语句
#else
  ASSERT_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false), c10::Error); // 在非 NDEBUG 模式下，应抛出 c10::Error 异常
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(true)); // 在非 NDEBUG 模式下，不应抛出异常
#endif
}

// On these platforms there's no assert
#if !defined(__ANDROID__) && !defined(__APPLE__)
TEST(ExceptionTest, CUDA_KERNEL_ASSERT) {
  // This function always throws even in NDEBUG mode
  ASSERT_DEATH_IF_SUPPORTED({ CUDA_KERNEL_ASSERT(false); }, "Assert"); // 在所有平台上，即使在 NDEBUG 模式下，也应该抛出异常
}
#endif

TEST(WarningTest, JustPrintWarning) {
  TORCH_WARN("I'm a warning"); // 打印警告信息
}

TEST(ExceptionTest, ErrorFormatting) {
  expectThrowsEq(
      []() { TORCH_CHECK(false, "This is invalid"); }, "This is invalid"); // 检查 TORCH_CHECK 是否抛出了预期的异常信息

  expectThrowsEq(
      []() {
        try {
          TORCH_CHECK(false, "This is invalid");
        } catch (Error& e) {
          TORCH_RETHROW(e, "While checking X"); // 捕获异常并重新抛出，添加额外的上下文信息
        }
      },
      "This is invalid (While checking X)");

  expectThrowsEq(
      []() {
        try {
          try {
            TORCH_CHECK(false, "This is invalid");
          } catch (Error& e) {
            TORCH_RETHROW(e, "While checking X"); // 多层嵌套的异常捕获和重新抛出，添加多个上下文信息
          }
        } catch (Error& e) {
          TORCH_RETHROW(e, "While checking Y");
        }
      },
      R"msg(This is invalid
  While checking X
  While checking Y)msg"); // 检查多层异常捕获和重新抛出是否输出了预期的多行信息
}

static int assertionArgumentCounter = 0;
static int getAssertionArgument() {
  return ++assertionArgumentCounter; // 返回递增的 assertionArgumentCounter 值
}

static void failCheck() {
  TORCH_CHECK(false, "message ", getAssertionArgument()); // 使用 getAssertionArgument 函数的返回值作为消息的一部分，检查是否抛出异常
}

static void failInternalAssert() {
  TORCH_INTERNAL_ASSERT(false, "message ", getAssertionArgument()); // 使用 getAssertionArgument 函数的返回值作为消息的一部分，检查是否抛出内部断言异常
}

TEST(ExceptionTest, DontCallArgumentFunctionsTwiceOnFailure) {
  assertionArgumentCounter = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failCheck()); // 检查 TORCH_CHECK 是否只调用了一次参数函数，而不是两次
  EXPECT_EQ(assertionArgumentCounter, 1) << "TORCH_CHECK called argument twice";

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failInternalAssert()); // 检查 TORCH_INTERNAL_ASSERT 是否只调用了一次参数函数，而不是两次
  EXPECT_EQ(assertionArgumentCounter, 2)
      << "TORCH_INTERNAL_ASSERT called argument twice";
}
```