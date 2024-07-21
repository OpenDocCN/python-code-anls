# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\common.h`

```py
/*
 * 版权声明：
 * 版权所有（c）Facebook, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码使用 BSD 风格许可证授权，详见源代码根目录下的 LICENSE 文件。
 */

#pragma once

#if defined(__GNUC__)
// 如果是 GCC 编译器
#if defined(__clang__) || (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 5)
// 如果是 Clang 编译器，或者 GCC 版本大于等于 4.5
#define PYTORCH_QNNP_UNREACHABLE \
  do {                           \
    __builtin_unreachable();     \
  } while (0)
#else
// 其他情况，使用内置陷阱函数
#define PYTORCH_QNNP_UNREACHABLE \
  do {                           \
    __builtin_trap();            \
  } while (0)
#endif
#elif defined(_MSC_VER)
// 如果是 MSVC 编译器
#define PYTORCH_QNNP_UNREACHABLE __assume(0)
#else
// 其他平台或编译器，不执行任何操作
#define PYTORCH_QNNP_UNREACHABLE \
  do {                           \
  } while (0)
#endif

#if defined(_MSC_VER)
// 如果是 MSVC 编译器，使用 __declspec(align(alignment)) 进行对齐
#define PYTORCH_QNNP_ALIGN(alignment) __declspec(align(alignment))
#else
// 其他情况，使用 __attribute__((__aligned__(alignment))) 进行对齐
#define PYTORCH_QNNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#endif

// 计算数组元素个数的宏定义
#define PYTORCH_QNNP_COUNT_OF(array) (sizeof(array) / sizeof(0 [array]))

#if defined(__GNUC__)
// 如果是 GCC 编译器，使用 __builtin_expect 来优化条件判断
#define PYTORCH_QNNP_LIKELY(condition) (__builtin_expect(!!(condition), 1))
#define PYTORCH_QNNP_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
// 其他情况，直接返回条件判断结果
#define PYTORCH_QNNP_LIKELY(condition) (!!(condition))
#define PYTORCH_QNNP_UNLIKELY(condition) (!!(condition))
#endif

#if defined(__GNUC__)
// 如果是 GCC 编译器，定义内联函数并标记为 __always_inline
#define PYTORCH_QNNP_INLINE inline __attribute__((__always_inline__))
#else
// 其他情况，只定义为内联函数
#define PYTORCH_QNNP_INLINE inline
#endif

#ifndef PYTORCH_QNNP_INTERNAL
#if defined(__ELF__)
// 如果是 ELF 格式的目标文件，设置为 internal 可见性
#define PYTORCH_QNNP_INTERNAL __attribute__((__visibility__("internal")))
#elif defined(__MACH__)
// 如果是 macOS 或 iOS 平台，设置为 hidden 可见性
#define PYTORCH_QNNP_INTERNAL __attribute__((__visibility__("hidden")))
#else
// 其他情况，不设置可见性
#define PYTORCH_QNNP_INTERNAL
#endif
#endif

#ifndef PYTORCH_QNNP_PRIVATE
#if defined(__ELF__)
// 如果是 ELF 格式的目标文件，设置为 hidden 可见性
#define PYTORCH_QNNP_PRIVATE __attribute__((__visibility__("hidden")))
#elif defined(__MACH__)
// 如果是 macOS 或 iOS 平台，设置为 hidden 可见性
#define PYTORCH_QNNP_PRIVATE __attribute__((__visibility__("hidden")))
#else
// 其他情况，不设置可见性
#define PYTORCH_QNNP_PRIVATE
#endif
#endif

#if defined(_MSC_VER)
// 如果是 MSVC 编译器，定义 RESTRICT_STATIC 和 restrict 为空
#define RESTRICT_STATIC
#define restrict
#else
// 其他情况，定义 RESTRICT_STATIC 为 restrict static
#define RESTRICT_STATIC restrict static
#endif

#if defined(_MSC_VER)
// 如果是 MSVC 编译器，定义 __builtin_prefetch 为空
#define __builtin_prefetch
#endif

#if defined(__GNUC__)
// 如果是 GCC 编译器，定义 PYTORCH_QNNP_UNALIGNED 为 __attribute__((__aligned__(1)))
#define PYTORCH_QNNP_UNALIGNED __attribute__((__aligned__(1)))
#elif defined(_MSC_VER)
// 如果是 MSVC 编译器，根据平台不同进行处理
  #if defined(_M_IX86)
    // 如果是 x86 平台，不进行对齐
    #define PYTORCH_QNNP_UNALIGNED
  #else
    // 其他情况，定义为 __unaligned
    #define PYTORCH_QNNP_UNALIGNED __unaligned
  #endif
#else
// 其他平台需要特定的实现来定义 PYTORCH_QNNP_UNALIGNED
#error "Platform-specific implementation of PYTORCH_QNNP_UNALIGNED required"
#endif
```