# `.\pytorch\test\cpp\lazy\test_util.cpp`

```py
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件

#include <exception> // 包含异常处理相关的头文件

#include <torch/csrc/lazy/core/util.h> // 包含 Torch 懒加载模块的工具函数头文件

namespace torch {
namespace lazy {

TEST(UtilTest, ExceptionCleanup) {
  std::exception_ptr exception; // 声明一个异常指针，用于捕获异常

  EXPECT_EQ(exception, nullptr); // 断言异常指针初始为空

  {
    // 创建 ExceptionCleanup 对象，用于在作用域结束时捕获异常
    ExceptionCleanup cleanup(
        [&](std::exception_ptr&& e) { exception = std::move(e); });

    // 设置异常状态为运行时错误，并交由 cleanup 对象处理
    cleanup.SetStatus(std::make_exception_ptr(std::runtime_error("Oops!")));
  }
  EXPECT_NE(exception, nullptr); // 断言异常指针不为空，说明异常已被捕获

  try {
    // 重新抛出捕获的异常，并进行断言检查异常的文本内容
    std::rethrow_exception(exception);
  } catch (const std::exception& e) {
    EXPECT_STREQ(e.what(), "Oops!"); // 断言捕获的异常文本为 "Oops!"
  }

  exception = nullptr; // 重置异常指针为 nullptr

  {
    // 创建 ExceptionCleanup 对象，用于处理异常，但在后续调用 release 后不捕获异常
    ExceptionCleanup cleanup(
        [&](std::exception_ptr&& e) { exception = std::move(e); });

    // 设置异常状态为空异常，并交由 cleanup 对象处理
    cleanup.SetStatus(std::make_exception_ptr(std::runtime_error("")));
    cleanup.Release(); // 释放 cleanup 对象，不捕获异常
  }
  EXPECT_EQ(exception, nullptr); // 断言异常指针为空，说明未捕获到异常
}

TEST(UtilTest, MaybeRef) {
  std::string storage("String storage"); // 创建一个字符串对象

  // 创建 MaybeRef 对象，引用 storage 对象，并断言其未存储
  MaybeRef<std::string> refStorage(storage);
  EXPECT_FALSE(refStorage.IsStored());

  // 断言 MaybeRef 对象的引用内容与原始字符串对象相同
  EXPECT_EQ(*refStorage, storage);

  // 创建 MaybeRef 对象，引用一个临时字符串对象，并断言其已存储
  MaybeRef<std::string> effStorage(std::string("Vanishing"));
  EXPECT_TRUE(effStorage.IsStored());

  // 断言 MaybeRef 对象的引用内容与初始化时的字符串相同
  EXPECT_EQ(*effStorage, "Vanishing");
}

TEST(UtilTest, Iota) {
  // 调用 Iota 函数生成一个整数类型的空数组，断言其为空
  auto result = Iota<int>(0);
  EXPECT_TRUE(result.empty());

  // 再次调用 Iota 函数生成一个包含一个整数的数组，进行相应的断言检查
  result = Iota<int>(1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], 0);

  // 再次调用 Iota 函数生成一个包含两个整数的数组，进行相应的断言检查
  result = Iota<int>(2);
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 1);

  // 再次调用 Iota 函数生成一个包含三个整数的数组，从 1 开始，步长为 3，进行相应的断言检查
  result = Iota<int>(3, 1, 3);
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 4);
  EXPECT_EQ(result[2], 7);
}

} // namespace lazy
} // namespace torch
```