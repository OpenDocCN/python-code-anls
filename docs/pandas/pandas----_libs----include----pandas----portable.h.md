# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\portable.h`

```
/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.
*/

// 如果之前未定义 strcasecmp 宏，则根据 _MSC_VER 定义定义为 _stricmp 函数
#if defined(_MSC_VER)
#define strcasecmp(s1, s2) _stricmp(s1, s2)
#endif

// GH-23516 - 解决了与 locale 相关的性能问题
// 基于 MUSL libc 的实现，其许可证在 LICENSES/MUSL_LICENSE
#define isdigit_ascii(c) (((unsigned)(c) - '0') < 10u)
#define getdigit_ascii(c, default)                                             \
  (isdigit_ascii(c) ? ((int)((c) - '0')) : default)
#define isspace_ascii(c) (((c) == ' ') || (((unsigned)(c) - '\t') < 5))
#define toupper_ascii(c) ((((unsigned)(c) - 'a') < 26) ? ((c) & 0x5f) : (c))
#define tolower_ascii(c) ((((unsigned)(c) - 'A') < 26) ? ((c) | 0x20) : (c))

// 如果在 Windows 平台，则定义 PD_FALLTHROUGH 为空语句，用于 switch 语句的 fallthrough
#if defined(_WIN32)
#define PD_FALLTHROUGH                                                         \
  do {                                                                         \
  } while (0) /* fallthrough */
// 如果编译器支持 __fallthrough__ 属性，则定义 PD_FALLTHROUGH 为 __fallthrough__ 属性
#elif __has_attribute(__fallthrough__)
#define PD_FALLTHROUGH __attribute__((__fallthrough__))
// 否则定义 PD_FALLTHROUGH 为空语句，用于 switch 语句的 fallthrough
#else
#define PD_FALLTHROUGH                                                         \
  do {                                                                         \
  } while (0) /* fallthrough */
#endif
```