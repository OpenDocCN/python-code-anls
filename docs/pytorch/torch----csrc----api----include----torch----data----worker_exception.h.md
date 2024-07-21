# `.\pytorch\torch\csrc\api\include\torch\data\worker_exception.h`

```
#pragma once

#include <exception>  // 包含异常处理的头文件
#include <string>     // 包含字符串处理的头文件
#include <utility>    // 包含实用工具的头文件

namespace torch {
namespace data {

/// An exception thrown when a DataLoader's worker thread throws an exception,
/// which is caught. A `WorkerException` stores an `exception_ptr` to the
/// original exception thrown in the worker thread.
/// 当DataLoader的工作线程抛出异常并被捕获时抛出的异常。`WorkerException`存储了指向
/// 工作线程中抛出的原始异常的`exception_ptr`。
struct WorkerException : public std::exception {
  /// Constructs a `WorkerException` from an `exception_ptr`.
  /// 从`exception_ptr`构造一个`WorkerException`。
  explicit WorkerException(std::exception_ptr original)
      : original_exception(std::move(original)),
        message("Caught exception in DataLoader worker thread.") {
    try {
      std::rethrow_exception(original_exception);  // 重新抛出原始异常以获得异常信息
    } catch (std::exception& e) {
      message += " Original message: ";   // 添加消息前缀
      message += e.what();                // 添加异常的具体信息
    }
  }

  const char* what() const noexcept override {
    return message.c_str();  // 返回异常消息的C风格字符串
  }

  /// The original exception thrown in the worker thread.
  /// 工作线程中抛出的原始异常。
  std::exception_ptr original_exception;

  /// This exception's message (not the original exception's message).
  /// 此异常的消息（不是原始异常的消息）。
  std::string message;
};

} // namespace data
} // namespace torch
```