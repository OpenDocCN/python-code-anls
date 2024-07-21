# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\deps\clog\src\clog.c`

```py
/*
 * 版权声明：
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * 本源代码在其根目录下的LICENSE文件中采用BSD风格许可证授权
 */

#include <stdarg.h>     // 包含可变参数列表支持
#include <stdio.h>      // 包含标准输入输出函数
#include <stdlib.h>     // 包含标准库函数
#include <string.h>     // 包含字符串处理函数
#ifdef _WIN32
#include <windows.h>    // Windows系统特定的头文件
#else
#include <unistd.h>     // Unix-like系统特定的头文件
#endif
#ifdef __ANDROID__
#include <android/log.h> // Android系统特定的日志头文件
#endif

#ifndef CLOG_LOG_TO_STDIO
#ifdef __ANDROID__
#define CLOG_LOG_TO_STDIO 0 // 如果是在Android平台且未定义CLOG_LOG_TO_STDIO，则将其设置为0
#else
#define CLOG_LOG_TO_STDIO 1 // 在其他平台，默认将CLOG_LOG_TO_STDIO设置为1
#endif
#endif

#include <clog.h>   // 包含CLOG库的头文件

/* Messages up to this size are formatted entirely on-stack, and don't allocate
 * heap memory */
#define CLOG_STACK_BUFFER_SIZE 1024 // 定义栈缓冲区大小为1024字节

#define CLOG_FATAL_PREFIX "Fatal error: " // 定义致命错误的前缀字符串常量
#define CLOG_FATAL_PREFIX_LENGTH 13      // 定义致命错误前缀字符串的长度
#define CLOG_FATAL_PREFIX_FORMAT "Fatal error in %s: " // 定义带模块信息的致命错误前缀格式化字符串
#define CLOG_ERROR_PREFIX "Error: "       // 定义错误的前缀字符串常量
#define CLOG_ERROR_PREFIX_LENGTH 7        // 定义错误前缀字符串的长度
#define CLOG_ERROR_PREFIX_FORMAT "Error in %s: " // 定义带模块信息的错误前缀格式化字符串
#define CLOG_WARNING_PREFIX "Warning: "   // 定义警告的前缀字符串常量
#define CLOG_WARNING_PREFIX_LENGTH 9      // 定义警告前缀字符串的长度
#define CLOG_WARNING_PREFIX_FORMAT "Warning in %s: " // 定义带模块信息的警告前缀格式化字符串
#define CLOG_INFO_PREFIX "Note: "         // 定义提示信息的前缀字符串常量
#define CLOG_INFO_PREFIX_LENGTH 6         // 定义提示信息前缀字符串的长度
#define CLOG_INFO_PREFIX_FORMAT "Note (%s): " // 定义带模块信息的提示信息前缀格式化字符串
#define CLOG_DEBUG_PREFIX "Debug: "       // 定义调试信息的前缀字符串常量
#define CLOG_DEBUG_PREFIX_LENGTH 7        // 定义调试信息前缀字符串的长度
#define CLOG_DEBUG_PREFIX_FORMAT "Debug (%s): " // 定义带模块信息的调试信息前缀格式化字符串
#define CLOG_SUFFIX_LENGTH 1              // 定义后缀字符串的长度

void clog_vlog_fatal(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  __android_log_vprint(ANDROID_LOG_FATAL, module, format, args); // 如果在Android平台且未设置CLOG_LOG_TO_STDIO，则使用Android日志系统输出致命错误信息
#else
  char stack_buffer[CLOG_STACK_BUFFER_SIZE]; // 定义栈缓冲区数组
  char* heap_buffer = NULL; // 声明堆缓冲区指针，并初始化为空
  char* out_buffer = &stack_buffer[0]; // 设置输出缓冲区指针指向栈缓冲区的起始位置

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  va_list args_copy;
  va_copy(args_copy, args); // 复制参数列表args到args_copy，以备第二次vsnprintf调用需要使用

  int prefix_chars = CLOG_FATAL_PREFIX_LENGTH; // 初始化前缀字符数为致命错误前缀的长度
  if (module == NULL) {
    memcpy(stack_buffer, CLOG_FATAL_PREFIX, CLOG_FATAL_PREFIX_LENGTH); // 如果模块信息为空，则将致命错误前缀复制到栈缓冲区中
  } else {
    prefix_chars = snprintf(
        stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_FATAL_PREFIX_FORMAT, module); // 否则，根据格式化字符串和模块信息格式化生成致命错误前缀，并计算生成的字符数
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      prefix_chars = 0; // 如果生成前缀时出现格式错误，则将前缀字符数置为0，忽略前缀，继续执行
    }
  }

  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    format_chars = vsnprintf(NULL, 0, format, args); // 如果前缀字符数加上后缀长度超出了栈缓冲区的大小，则需要使用堆缓冲区来存储格式化后的字符串
  } else {
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args); // 否则，在栈缓冲区剩余空间中格式化生成消息内容
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    goto cleanup; // 如果消息格式化过程中出现错误，则忽略此消息，直接跳转到清理步骤
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* 在堆上分配一个缓冲区，并使用 vsnprintf 将内容写入该缓冲区 */
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    /* 检查内存分配是否成功 */
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    /* 如果前缀字符超出了栈上缓冲区的大小，重新将前缀格式化到堆上的缓冲区 */
    snprintf(
        heap_buffer,
        prefix_chars + 1 /* '\0' 结尾符 */,
        CLOG_FATAL_PREFIX_FORMAT,
        module);
    /* 否则，将栈上缓冲区中的预格式化前缀复制到堆上的缓冲区 */
    memcpy(heap_buffer, stack_buffer, prefix_chars);

    /* 使用 vsnprintf 将格式化的消息内容写入堆上缓冲区 */
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);

    /* 将堆上缓冲区的指针赋给输出缓冲区 */
    out_buffer = heap_buffer;
  }
  /* 在输出缓冲区中添加换行符 */
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  // 如果在 Windows 平台下
  DWORD bytes_written;
  // 使用 WriteFile 将 out_buffer 中指定长度的数据写入标准错误输出，并获取写入的字节数
  WriteFile(
      GetStdHandle(STD_ERROR_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  // 如果不在 Windows 平台下
  // 将 out_buffer 中指定长度的数据写入 STDERR_FILENO（标准错误输出）
  write(
      STDERR_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  // 释放 heap_buffer 内存
  free(heap_buffer);
  // 结束变长参数的使用
  va_end(args_copy);
#endif
}

void clog_vlog_error(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  // 如果在 Android 平台且不将日志输出到标准IO流
  // 使用 __android_log_vprint 输出 Android 平台的日志错误信息
  __android_log_vprint(ANDROID_LOG_ERROR, module, format, args);
#else
  // 在其他平台下或者需要将日志输出到标准IO流时
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  // 第一次调用 vsnprintf 会破坏 args，因此需要在需要第二次调用 vsnprintf 时使用 args 的副本
  va_list args_copy;
  va_copy(args_copy, args);

  int prefix_chars = CLOG_ERROR_PREFIX_LENGTH;
  if (module == NULL) {
    // 如果 module 为 NULL，则拷贝错误信息前缀到 stack_buffer
    memcpy(stack_buffer, CLOG_ERROR_PREFIX, CLOG_ERROR_PREFIX_LENGTH);
  } else {
    // 否则，使用 snprintf 格式化 module 和错误信息前缀，将结果存储到 stack_buffer，并获取前缀字符数
    prefix_chars = snprintf(
        stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_ERROR_PREFIX_FORMAT, module);
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      // 如果前缀格式化出错，则跳过前缀并继续执行
      prefix_chars = 0;
    }
  }

  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    // 如果前缀加后缀的长度超过了栈上缓冲区的大小，则需要使用堆上的缓冲区
    // 不尝试将字符串格式化到栈上缓冲区
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    // 否则，将格式化后的信息存储到 stack_buffer 的合适位置，并获取格式化后字符的数量
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    // 如果信息格式化出错，则静默忽略该消息
    goto cleanup;
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    // 如果前缀加格式化后的信息加后缀的长度超过了栈上缓冲区的大小
    // 分配一个堆上的缓冲区，并将格式化后的信息存储到该缓冲区
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
       * buffer */
      // 如果前缀没有适合栈上缓冲区的大小，将前缀重新格式化到堆上缓冲区
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_ERROR_PREFIX_FORMAT,
          module);
    } else {
      /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
      // 否则，将预先格式化的前缀从栈上缓冲区复制到堆上缓冲区
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    out_buffer = heap_buffer;
  }
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  // 如果在 Windows 平台下
  DWORD bytes_written;
  // 将错误信息写入标准错误输出，获取写入字节数
  WriteFile(
      GetStdHandle(STD_ERROR_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  // 如果不在 Windows 平台下
  // 将错误信息写入标准错误输出
  write(
      STDERR_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  // 释放堆缓冲区
  free(heap_buffer);
  // 结束变参列表的使用
  va_end(args_copy);
#endif
}

void clog_vlog_warning(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  // 如果在 Android 下并且不要求将日志输出到标准输入输出
  __android_log_vprint(ANDROID_LOG_WARN, module, format, args);
#else
  // 在其他平台下或者需要将日志输出到标准输入输出
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  // 第一次调用 vsnprintf 会破坏 args，因此需要复制一份以备可能需要第二次调用
  va_list args_copy;
  va_copy(args_copy, args);

  // 设置日志前缀的长度为警告信息的长度
  int prefix_chars = CLOG_WARNING_PREFIX_LENGTH;
  // 如果模块名为空
  if (module == NULL) {
    // 将预定义的警告信息前缀复制到栈缓冲区
    memcpy(stack_buffer, CLOG_WARNING_PREFIX, CLOG_WARNING_PREFIX_LENGTH);
  } else {
    // 否则，根据模块名格式化警告信息前缀并计算长度
    prefix_chars = snprintf(
        stack_buffer,
        CLOG_STACK_BUFFER_SIZE,
        CLOG_WARNING_PREFIX_FORMAT,
        module);
    // 如果格式化失败，可能是前缀修改导致的格式错误，跳过前缀，继续处理
    if (prefix_chars < 0) {
      prefix_chars = 0;
    }
  }

  // 记录格式化后的消息的长度
  int format_chars;
  // 如果栈缓冲区无法容纳前缀和后缀，则使用堆缓冲区
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    // 使用 vsnprintf 计算消息格式化后的长度，但不实际格式化内容
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    // 在栈缓冲区中格式化消息，计算格式化后的长度
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  // 如果格式化失败，静默地忽略这条消息
  if (format_chars < 0) {
    goto cleanup;
  }
  // 如果栈缓冲区无法容纳前缀、消息和后缀，则需要使用堆缓冲区
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    // 分配堆缓冲区，并格式化消息内容到堆缓冲区
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    // 如果分配失败，则进行清理操作
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    // 如果前缀无法在栈上容纳，则重新格式化前缀到堆缓冲区
    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_WARNING_PREFIX_FORMAT,
          module);
    } else {
      // 否则，将栈缓冲区中的前缀复制到堆缓冲区
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    // 使用复制的参数列表在堆缓冲区中格式化消息内容
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    // 将输出缓冲指向堆缓冲区
    out_buffer = heap_buffer;
  }
  // 在输出缓冲区末尾添加换行符
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  // 如果在 Windows 下编译
  DWORD bytes_written;
  // 使用 WriteFile 函数写入数据到标准错误输出句柄
  WriteFile(
      GetStdHandle(STD_ERROR_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  // 在非 Windows 环境下
  // 使用 write 函数写入数据到标准错误文件描述符
  write(
      STDERR_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  // 清理操作，释放堆内存
  free(heap_buffer);
  // 结束变长参数的使用
  va_end(args_copy);
#endif
}

void clog_vlog_info(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  // 如果在 Android 环境下，并且不将日志输出到标准输入输出
  // 使用 Android 平台的日志打印函数
  __android_log_vprint(ANDROID_LOG_INFO, module, format, args);
#else
  // 在其他环境下
  // 声明一个栈上的字符数组作为输出缓冲区
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  // 第一次调用 vsnprintf 会损坏 args，因此需要复制一份 args
  va_list args_copy;
  va_copy(args_copy, args);

  // 计算日志前缀的长度
  int prefix_chars = CLOG_INFO_PREFIX_LENGTH;
  if (module == NULL) {
    // 如果模块名为空，则将默认的日志前缀复制到栈上的缓冲区中
    memcpy(stack_buffer, CLOG_INFO_PREFIX, CLOG_INFO_PREFIX_LENGTH);
  } else {
    // 否则根据模块名格式化日志前缀并计算前缀字符数
    prefix_chars = snprintf(
        stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_INFO_PREFIX_FORMAT, module);
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      // 如果格式化日志前缀时出错，则跳过前缀，继续执行
      prefix_chars = 0;
    }
  }

  // 计算格式化后消息体的长度
  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    // 如果日志前缀加后缀长度超过栈上缓冲区大小，则需要使用堆上的缓冲区
    // 此时只计算格式化后消息体的长度，不在栈上缓冲区中尝试格式化字符串
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    // 否则在栈上缓冲区中格式化消息体，并计算其长度
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    // 如果格式化消息体时出错，则静默地忽略这条消息
    goto cleanup;
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    // 如果日志前缀加消息体长度加后缀超过栈上缓冲区大小，则需要在堆上分配缓冲区
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
       * buffer */
      // 如果日志前缀没有适合栈上缓冲区的空间，则重新格式化到堆上缓冲区
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_INFO_PREFIX_FORMAT,
          module);
    } else {
      /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
      // 否则将已经格式化好的日志前缀复制到堆上缓冲区
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    // 将格式化后的消息体追加到堆上缓冲区
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    out_buffer = heap_buffer;
  }
  // 在输出缓冲区的结尾添加换行符
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  // 如果在 Windows 平台下
  DWORD bytes_written;
  // 使用 Windows API 写入数据到标准输出
  WriteFile(
      GetStdHandle(STD_OUTPUT_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  // 如果不在 Windows 平台下，使用 POSIX API 写入数据到标准输出
  write(
      STDOUT_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  // 清理工作：释放动态分配的堆内存
  free(heap_buffer);
  // 结束可变参数列表的使用
  va_end(args_copy);
#endif
}

void clog_vlog_debug(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  // 在 Android 平台上，通过 Android 日志系统输出调试信息
  __android_log_vprint(ANDROID_LOG_DEBUG, module, format, args);
#else
  // 定义栈上的缓冲区和堆上的缓冲区指针
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  // 由于第一次调用 vsnprintf 会破坏 args，因此需要创建一个 args 的副本
  va_list args_copy;
  va_copy(args_copy, args);

  // 计算调试信息前缀的长度
  int prefix_chars = CLOG_DEBUG_PREFIX_LENGTH;
  if (module == NULL) {
    // 如果模块名为空，使用默认的调试信息前缀
    memcpy(stack_buffer, CLOG_DEBUG_PREFIX, CLOG_DEBUG_PREFIX_LENGTH);
  } else {
    // 根据模块名格式化调试信息前缀，并获取其长度
    prefix_chars = snprintf(
        stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_DEBUG_PREFIX_FORMAT, module);
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      // 如果前缀格式化错误，则跳过前缀，继续处理
      prefix_chars = 0;
    }
  }

  // 计算格式化输出的字符数
  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    // 如果前缀和后缀总长度超过栈上缓冲区大小，则需要使用堆上的缓冲区
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    // 否则，将格式化输出到栈上的缓冲区中
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    // 如果格式化消息时出现错误，则静默忽略这条消息
    goto cleanup;
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    // 如果输出总长度超过栈上缓冲区大小，则分配堆上的缓冲区
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
       * buffer */
      // 如果前缀不适合栈上缓冲区，则重新格式化到堆上的缓冲区
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_DEBUG_PREFIX_FORMAT,
          module);
    } else {
      /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
      // 将预格式化的前缀从栈上缓冲区复制到堆上的缓冲区
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    // 将格式化输出到堆上的缓冲区
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    out_buffer = heap_buffer;
  }
  // 在输出缓冲区的结尾添加换行符
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  // 如果是在 Windows 平台下
  DWORD bytes_written;
  // 调用 Windows API 的 WriteFile 函数，向标准输出句柄写入数据
  WriteFile(
      GetStdHandle(STD_OUTPUT_HANDLE),  // 获取标准输出句柄
      out_buffer,                       // 要写入的数据缓冲区
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,  // 要写入的字节数
      &bytes_written,                   // 实际写入的字节数将被写入此变量
      NULL);                            // 不使用 overlapped I/O

#else
  // 如果不是在 Windows 平台下
  // 调用 POSIX 标准的 write 函数，向标准输出写入数据
  write(
      STDOUT_FILENO,                    // 标准输出文件描述符
      out_buffer,                       // 要写入的数据缓冲区
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);  // 要写入的字节数
#endif

cleanup:
  // 释放动态分配的堆内存
  free(heap_buffer);
  // 结束使用变长参数列表
  va_end(args_copy);
#endif
}
```