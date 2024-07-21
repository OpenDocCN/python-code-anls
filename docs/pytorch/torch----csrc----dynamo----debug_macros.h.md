# `.\pytorch\torch\csrc\dynamo\debug_macros.h`

```py
#pragma once
// 单次包含头文件防止重复包含

#include <stdio.h>
// 包含标准输入输出头文件

#ifdef _WIN32
#define unlikely(x) (x)
#else
#define unlikely(x) __builtin_expect((x), 0)
#endif
// 根据操作系统定义 unlikely 宏，用于优化分支预测

#define NULL_CHECK(val)                                         \
  if (unlikely((val) == NULL)) {                                \
    fprintf(stderr, "NULL ERROR: %s:%d\n", __FILE__, __LINE__); \
    PyErr_Print();                                              \
    abort();                                                    \
  } else {                                                      \
  }
// 定义 NULL_CHECK 宏，检查指针是否为空，打印错误信息并终止程序

// CHECK might be previously declared
#undef CHECK
#define CHECK(cond)                                                     \
  if (unlikely(!(cond))) {                                              \
    fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__); \
    abort();                                                            \
  } else {                                                              \
  }
// 定义 CHECK 宏，检查条件是否满足，如果不满足则打印失败信息并终止程序

// Uncomment next line to print debug message
// #define TORCHDYNAMO_DEBUG 1
#ifdef TORCHDYNAMO_DEBUG
// 如果定义了 TORCHDYNAMO_DEBUG 宏，则启用以下调试宏定义

#define DEBUG_CHECK(cond) CHECK(cond)
// 调试版本的 CHECK 宏

#define DEBUG_NULL_CHECK(val) NULL_CHECK(val)
// 调试版本的 NULL_CHECK 宏

#define DEBUG_TRACE(msg, ...) \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__, __VA_ARGS__)
// 带参数的调试输出宏，输出函数名、行号和格式化消息

#define DEBUG_TRACE0(msg) \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__)
// 不带参数的调试输出宏，输出函数名、行号和消息

#else
// 如果未定义 TORCHDYNAMO_DEBUG 宏，则禁用以下调试宏定义

#define DEBUG_CHECK(cond)
// 空的调试版本 CHECK 宏

#define DEBUG_NULL_CHECK(val)
// 空的调试版本 NULL_CHECK 宏

#define DEBUG_TRACE(msg, ...)
// 空的调试版本带参数调试输出宏

#define DEBUG_TRACE0(msg)
// 空的调试版本不带参数调试输出宏

#endif
```