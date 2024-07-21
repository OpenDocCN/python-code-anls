# `.\pytorch\torch\csrc\profiler\unwind\unwind_error.h`

```
#pragma once
#include <c10/util/Optional.h>   // 包含可选类型的头文件
#include <fmt/format.h>          // 包含格式化输出的头文件
#include <stdexcept>             // 包含标准异常类的头文件

namespace torch::unwind {

struct UnwindError : public std::runtime_error {  // 定义 UnwindError 结构体，继承自 std::runtime_error
  using std::runtime_error::runtime_error;        // 使用基类构造函数
};

#define UNWIND_CHECK(cond, fmtstring, ...)                          \
  do {                                                              \
    if (!(cond)) {                                                  \
      throw unwind::UnwindError(fmt::format(                        \
          "{}:{}: " fmtstring, __FILE__, __LINE__, ##__VA_ARGS__)); \
    }                                                               \
  } while (0)  // 定义宏 UNWIND_CHECK，如果条件不满足则抛出 UnwindError 异常

// #define LOG_INFO(...) fmt::print(__VA_ARGS__)
#define LOG_INFO(...)  // 定义宏 LOG_INFO，目前被注释掉

// #define PRINT_INST(...) LOG_INFO(__VA_ARGS__)
#define PRINT_INST(...)  // 定义宏 PRINT_INST，目前被注释掉

// #define PRINT_LINE_TABLE(...) LOG_INFO(__VA_ARGS__)
#define PRINT_LINE_TABLE(...)  // 定义宏 PRINT_LINE_TABLE，目前被注释掉

using std::optional; // NOLINT，使用 std 命名空间中的 optional 类型

} // namespace torch::unwind  // torch::unwind 命名空间结束
```