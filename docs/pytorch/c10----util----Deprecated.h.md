# `.\pytorch\c10\util\Deprecated.h`

```py
#pragma once

/**
 * This file provides portable macros for marking declarations
 * as deprecated.  You should generally use C10_DEPRECATED,
 * except when marking 'using' declarations as deprecated,
 * in which case you should use C10_DEFINE_DEPRECATED_USING
 * (due to portability concerns).
 */

// NB: __cplusplus doesn't work for MSVC, so for now MSVC always uses
// the "__declspec(deprecated)" implementation and not the C++14
// "[[deprecated]]" attribute. We tried enabling "[[deprecated]]" for C++14 on
// MSVC, but ran into issues with some older MSVC versions.

// Define C10_DEPRECATED and C10_DEPRECATED_MESSAGE based on compiler support

#if (defined(__cplusplus) && __cplusplus >= 201402L)
// C++14 and later supports [[deprecated]]
#define C10_DEPRECATED [[deprecated]]
#define C10_DEPRECATED_MESSAGE(message) [[deprecated(message)]]
#elif defined(__GNUC__)
// GCC supports __attribute__((deprecated))
#define C10_DEPRECATED __attribute__((deprecated))
// TODO Is there some way to implement this?
#define C10_DEPRECATED_MESSAGE(message) __attribute__((deprecated))
#elif defined(_MSC_VER)
// MSVC supports __declspec(deprecated)
#define C10_DEPRECATED __declspec(deprecated)
#define C10_DEPRECATED_MESSAGE(message) __declspec(deprecated(message))
#else
// Compiler warning for unsupported compiler
#warning "You need to implement C10_DEPRECATED for this compiler"
#define C10_DEPRECATED
#endif

// Define C10_DEFINE_DEPRECATED_USING based on compiler support

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated) && !defined(__CUDACC__)
// Use [[deprecated]] for compilers supporting the attribute
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName [[deprecated]] = TypeThingy;
#endif
#endif

#if defined(_MSC_VER)
#if defined(__CUDACC__)
// For nvcc on Windows, define using without deprecated attribute due to compatibility
#if defined(C10_DEFINE_DEPRECATED_USING)
#undef C10_DEFINE_DEPRECATED_USING
#endif
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName = TypeThingy;
#else
// For other MSVC scenarios, choose between [[deprecated]] and __declspec(deprecated)
#ifndef C10_DEFINE_DEPRECATED_USING
#if defined(_MSVC_LANG) && _MSVC_LANG >= 201402L
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName [[deprecated]] = TypeThingy;
#else
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName = __declspec(deprecated) TypeThingy;
#endif
#endif
#endif
#endif

#if !defined(C10_DEFINE_DEPRECATED_USING) && defined(__GNUC__)
// GCC specific handling for deprecated attribute due to nvcc compatibility issues
// 如果当前编译环境不是 CUDA，且使用的是不支持我们之前检查到的 C++14 语法的 GCC 编译器（即 __GNUC__ < 5）
#if !defined(__CUDACC__)
// 定义一个使用了已废弃标记的类型别名宏，将 TypeName 映射到 TypeThingy
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName __attribute__((deprecated)) = TypeThingy;
#else
// 如果是 CUDA 或者是支持 C++14 语法的 GCC 编译器（即 __GNUC__ >= 5）
// 定义一个普通的类型别名宏，将 TypeName 映射到 TypeThingy
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName = TypeThingy;
#endif
#endif

// 如果 C10_DEFINE_DEPRECATED_USING 宏未被定义
#if !defined(C10_DEFINE_DEPRECATED_USING)
// 输出警告消息，提示需要为当前编译器实现 C10_DEFINE_DEPRECATED_USING 宏
#warning "You need to implement C10_DEFINE_DEPRECATED_USING for this compiler"
// 定义一个空的 C10_DEFINE_DEPRECATED_USING 宏，以避免编译错误
#define C10_DEFINE_DEPRECATED_USING
#endif
```