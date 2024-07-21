# `.\pytorch\test\cpp\common\support.h`

```
#pragma once

#include <c10/util/Exception.h>  // 包含了异常处理相关的头文件

#include <gtest/gtest.h>  // 包含了 Google 测试框架的头文件

#include <stdexcept>  // 包含了标准异常类的头文件
#include <string>     // 包含了使用字符串的相关函数和类的头文件

namespace torch {
namespace test {

#define ASSERT_THROWS_WITH(statement, substring)                        \
  {                                                                     \
    std::string assert_throws_with_error_message;                       \
    try {                                                               \
      (void)statement;                                                  \
      // 如果 statement 没有抛出异常，则测试失败并打印相关信息
      FAIL() << "Expected statement `" #statement                       \
                "` to throw an exception, but it did not";              \
    } catch (const c10::Error& e) {                                     \
      // 如果捕获到 c10::Error 异常，则获取异常信息，不包括回溯信息
      assert_throws_with_error_message = e.what_without_backtrace();    \
    } catch (const std::exception& e) {                                 \
      // 如果捕获到标准异常，则获取异常信息
      assert_throws_with_error_message = e.what();                      \
    }                                                                   \
    // 检查捕获到的异常信息中是否包含指定的子串 substring
    if (assert_throws_with_error_message.find(substring) ==             \
        std::string::npos) {                                            \
      // 如果不包含指定的子串，则测试失败并打印相关信息
      FAIL() << "Error message \"" << assert_throws_with_error_message  \
             << "\" did not contain expected substring \"" << substring \
             << "\"";                                                   \
    }                                                                   \
  }

} // namespace test
} // namespace torch
```