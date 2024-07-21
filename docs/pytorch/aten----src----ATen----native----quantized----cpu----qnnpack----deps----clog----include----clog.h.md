# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\deps\clog\include\clog.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>

// 定义日志级别常量
#define CLOG_NONE 0
#define CLOG_FATAL 1
#define CLOG_ERROR 2
#define CLOG_WARNING 3
#define CLOG_INFO 4
#define CLOG_DEBUG 5

// 定义 CLOG_VISIBILITY 宏，用于声明函数的可见性
#ifndef CLOG_VISIBILITY
#if defined(__ELF__)
#define CLOG_VISIBILITY __attribute__((__visibility__("internal")))
#elif defined(__MACH__)
#define CLOG_VISIBILITY __attribute__((__visibility__("hidden")))
#else
#define CLOG_VISIBILITY
#endif
#endif

// 定义 CLOG_ARGUMENTS_FORMAT 宏，用于指定函数参数格式
#ifndef CLOG_ARGUMENTS_FORMAT
#if defined(__GNUC__)
#define CLOG_ARGUMENTS_FORMAT __attribute__((__format__(__printf__, 1, 2)))
#else
#define CLOG_ARGUMENTS_FORMAT
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 声明输出调试信息的函数，参数为模块名、格式化字符串和可变参数列表
CLOG_VISIBILITY void clog_vlog_debug(
    const char* module,
    const char* format,
    va_list args);
// 声明输出信息日志的函数，参数为模块名、格式化字符串和可变参数列表
CLOG_VISIBILITY void clog_vlog_info(
    const char* module,
    const char* format,
    va_list args);
// 声明输出警告信息的函数，参数为模块名、格式化字符串和可变参数列表
CLOG_VISIBILITY void clog_vlog_warning(
    const char* module,
    const char* format,
    va_list args);
// 声明输出错误信息的函数，参数为模块名、格式化字符串和可变参数列表
CLOG_VISIBILITY void clog_vlog_error(
    const char* module,
    const char* format,
    va_list args);
// 声明输出致命错误信息的函数，参数为模块名、格式化字符串和可变参数列表
CLOG_VISIBILITY void clog_vlog_fatal(
    const char* module,
    const char* format,
    va_list args);

// 定义调试日志输出宏，参数为输出函数名称、模块名和日志级别
#define CLOG_DEFINE_LOG_DEBUG(log_debug_function_name, module, level)   \
  CLOG_ARGUMENTS_FORMAT                                                 \
  inline static void log_debug_function_name(const char* format, ...) { \
    // 检查日志级别是否达到调试级别，若达到则输出调试信息
    if (level >= CLOG_DEBUG) {                                          \
      va_list args;                                                     \
      va_start(args, format);                                           \
      clog_vlog_debug(module, format, args);                            \
      va_end(args);                                                     \
    }                                                                   \
  }

// 定义信息日志输出宏，参数为输出函数名称、模块名和日志级别
#define CLOG_DEFINE_LOG_INFO(log_info_function_name, module, level)    \
  CLOG_ARGUMENTS_FORMAT                                                \
  inline static void log_info_function_name(const char* format, ...) { \
    // 检查日志级别是否达到信息级别，若达到则输出信息日志
    if (level >= CLOG_INFO) {                                          \
      va_list args;                                                    \
      va_start(args, format);                                          \
      clog_vlog_info(module, format, args);                            \
      va_end(args);                                                    \
    }                                                                  \
  }

// 定义警告日志输出宏，参数为输出函数名称、模块名和日志级别
#define CLOG_DEFINE_LOG_WARNING(log_warning_function_name, module, level) \
  CLOG_ARGUMENTS_FORMAT                                                   \
  inline static void log_warning_function_name(const char* format, ...) { \
    // 检查日志级别是否达到警告级别，若达到则输出警告信息

#define CLOG_DEFINE_LOG_WARNING(log_warning_function_name, module, level) \
  CLOG_ARGUMENTS_FORMAT                                                   \
  inline static void log_warning_function_name(const char* format, ...) { \
    // 检查日志级别是否达到警告级别，若达到则输出警告信息
    if (level >= CLOG_WARNING) {                                          \
      va_list args;                                                       \
      va_start(args, format);                                             \
      clog_vlog_warning(module, format, args);                            \
      va_end(args);                                                       \
    }                                                                     \
  }


这段代码定义了一系列用于输出不同级别日志的宏和函数，确保在不同条件下根据日志级别输出相应的信息，利用了可变参数列表和预处理宏来实现灵活的日志输出控制。
    // 如果日志级别大于等于警告级别 CLOG_WARNING，则执行以下代码块
    if (level >= CLOG_WARNING) {                                          
      // 声明 va_list 类型的变量 args，用于存储可变参数列表
      va_list args;                                                       
      // 初始化可变参数列表 args，以 format 参数为基础
      va_start(args, format);                                             
      // 调用 clog_vlog_warning 函数，传递模块名称 module、格式化字符串 format 和可变参数列表 args
      clog_vlog_warning(module, format, args);                            
      // 结束使用可变参数列表 args
      va_end(args);                                                       
    }                                                                     
  }
# 定义宏 CLOG_DEFINE_LOG_ERROR，用于定义记录错误日志的函数
#define CLOG_DEFINE_LOG_ERROR(log_error_function_name, module, level)   \
  CLOG_ARGUMENTS_FORMAT                                                 \
  // 内联静态函数，用于记录错误日志，接受格式化字符串和可变参数列表
  inline static void log_error_function_name(const char* format, ...) { \
    // 如果日志级别大于等于错误级别，则记录错误日志
    if (level >= CLOG_ERROR) {                                          \
      // 声明可变参数列表并开始解析
      va_list args;                                                     \
      va_start(args, format);                                           \
      // 调用具体的错误日志记录函数，传递模块名、格式化字符串和参数列表
      clog_vlog_error(module, format, args);                            \
      // 结束可变参数列表的解析
      va_end(args);                                                     \
    }                                                                   \
  }

# 定义宏 CLOG_DEFINE_LOG_FATAL，用于定义记录致命错误日志的函数
#define CLOG_DEFINE_LOG_FATAL(log_fatal_function_name, module, level)   \
  CLOG_ARGUMENTS_FORMAT                                                 \
  // 内联静态函数，用于记录致命错误日志，接受格式化字符串和可变参数列表
  inline static void log_fatal_function_name(const char* format, ...) { \
    // 如果日志级别大于等于致命错误级别，则记录致命错误日志
    if (level >= CLOG_FATAL) {                                          \
      // 声明可变参数列表并开始解析
      va_list args;                                                     \
      va_start(args, format);                                           \
      // 调用具体的致命错误日志记录函数，传递模块名、格式化字符串和参数列表
      clog_vlog_fatal(module, format, args);                            \
      // 结束可变参数列表的解析
      va_end(args);                                                     \
    }                                                                   \
    // 终止程序执行
    abort();                                                            \
  }

#ifdef __cplusplus
} /* extern "C" */
#endif
```