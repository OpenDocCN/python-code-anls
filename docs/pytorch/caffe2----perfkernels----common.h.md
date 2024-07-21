# `.\pytorch\caffe2\perfkernels\common.h`

```
// !!!! PLEASE READ !!!!
// Minimize (transitively) included headers from _avx*.cc because some of the
// functions defined in the headers compiled with platform dependent compiler
// options can be reused by other translation units generating illegal
// instruction run-time error.

// Common utilities for writing performance kernels and easy dispatching of
// different backends.

/*
The general workflow shall be as follows, say we want to
implement a functionality called void foo(int a, float b).

In foo.h, do:
   void foo(int a, float b);

In foo_avx512.cc, do:
   void foo__avx512(int a, float b) {
     [actual avx512 implementation]
   }

In foo_avx2.cc, do:
   void foo__avx2(int a, float b) {
     [actual avx2 implementation]
   }

In foo_avx.cc, do:
   void foo__avx(int a, float b) {
     [actual avx implementation]
   }

In foo.cc, do:
   // The base implementation should *always* be provided.
   void foo__base(int a, float b) {
     [base, possibly slow implementation]
   }
   decltype(foo__base) foo__avx512;
   decltype(foo__base) foo__avx2;
   decltype(foo__base) foo__avx;
   void foo(int a, float b) {
     // You should always order things by their preference, faster
     // implementations earlier in the function.
     AVX512_DO(foo, a, b);
     AVX2_DO(foo, a, b);
     AVX_DO(foo, a, b);
     BASE_DO(foo, a, b);
   }
*/

// Details: this functionality basically covers the cases for both build time
// and run time architecture support.
//
// During build time:
//    The build system should provide flags CAFFE2_PERF_WITH_AVX512,
//    CAFFE2_PERF_WITH_AVX2, and CAFFE2_PERF_WITH_AVX that corresponds to the
//    __AVX512F__, __AVX512DQ__, __AVX512VL__, __AVX2__, and __AVX__ flags the
//    compiler provides. Note that we do not use the compiler flags but rely on
//    the build system flags, because the common files (like foo.cc above) will
//    always be built without __AVX512F__, __AVX512DQ__, __AVX512VL__, __AVX2__
//    and __AVX__.
// During run time:
//    we use cpuinfo to identify cpu support and run the proper functions.

#pragma once

#if defined(CAFFE2_PERF_WITH_AVX512) || defined(CAFFE2_PERF_WITH_AVX2) \
     || defined(CAFFE2_PERF_WITH_AVX)
#include <cpuinfo.h>
#endif

// DO macros: these should be used in your entry function, similar to foo()
// above, that routes implementations based on CPU capability.

#define BASE_DO(funcname, ...) return funcname##__base(__VA_ARGS__);

#ifdef CAFFE2_PERF_WITH_AVX512
#define AVX512_DO(funcname, ...)                                   \
  {                                                                \
    static const bool isDo = cpuinfo_initialize() &&               \
        cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512dq() && \
        cpuinfo_has_x86_avx512vl();                                \
    if (isDo) {                                                    \
      return funcname##__avx512(__VA_ARGS__);                      \
    }                                                              \
  }
#else
#define AVX512_DO(funcname, ...) /* AVX512 not enabled */
#endif

#ifdef CAFFE2_PERF_WITH_AVX2
#define AVX2_DO(funcname, ...)                                     \
  {                                                                \
    static const bool isDo = cpuinfo_initialize() &&               \
        cpuinfo_has_x86_avx2();                                    \
    if (isDo) {                                                    \
      return funcname##__avx2(__VA_ARGS__);                        \
    }                                                              \
  }
#else
#define AVX2_DO(funcname, ...) /* AVX2 not enabled */
#endif

#ifdef CAFFE2_PERF_WITH_AVX
#define AVX_DO(funcname, ...)                                      \
  {                                                                \
    static const bool isDo = cpuinfo_initialize() &&               \
        cpuinfo_has_x86_avx();                                     \
    if (isDo) {                                                    \
      return funcname##__avx(__VA_ARGS__);                         \
    }                                                              \
  }
#else
#define AVX_DO(funcname, ...) /* AVX not enabled */
#endif
    }                                                              \
  }


注释：


    # 这段代码片段似乎是一处格式化或者语法修复。它通过右括号和花括号的排列来修复代码结构或者语法错误。
    # 在某些编程语言中，这种用法可能是为了避免语法错误或者为了将多行代码合并为一行。
    # 在大多数情况下，这不是一个标准的语法结构，可能是由于代码编辑或复制粘贴过程中的临时行为。
    # 该片段的具体作用需要根据上下文来确认。
#ifdef CAFFE2_PERF_WITH_AVX512
// 如果定义了 CAFFE2_PERF_WITH_AVX512 宏，则定义 AVX512_DO 宏
#define AVX512_DO(funcname, ...)
#endif // CAFFE2_PERF_WITH_AVX512

#ifdef CAFFE2_PERF_WITH_AVX2
// 如果定义了 CAFFE2_PERF_WITH_AVX2 宏，则定义 AVX2_DO 和 AVX2_FMA_DO 宏
#define AVX2_DO(funcname, ...)                                               \
  {                                                                          \
    // 静态变量 isDo 标志是否满足 AVX2 要求，包括 CPU 信息的初始化和 AVX2 的支持
    static const bool isDo = cpuinfo_initialize() && cpuinfo_has_x86_avx2(); \
    // 如果满足 AVX2 要求，则调用对应的 AVX2 版本的函数，并返回结果
    if (isDo) {                                                              \
      return funcname##__avx2(__VA_ARGS__);                                  \
    }                                                                        \
  }

#define AVX2_FMA_DO(funcname, ...)                                             \
  {                                                                            \
    // 静态变量 isDo 标志是否满足 AVX2 + FMA3 要求，包括 CPU 信息的初始化、AVX2 和 FMA3 的支持
    static const bool isDo = cpuinfo_initialize() && cpuinfo_has_x86_avx2() && \
        cpuinfo_has_x86_fma3();                                                \
    // 如果满足 AVX2 + FMA3 要求，则调用对应的 AVX2 + FMA3 版本的函数，并返回结果
    if (isDo) {                                                                \
      return funcname##__avx2_fma(__VA_ARGS__);                                \
    }                                                                          \
  }
#else // CAFFE2_PERF_WITH_AVX2
// 如果未定义 CAFFE2_PERF_WITH_AVX2 宏，则定义 AVX2_DO 和 AVX2_FMA_DO 宏为空
#define AVX2_DO(funcname, ...)
#define AVX2_FMA_DO(funcname, ...)
#endif // CAFFE2_PERF_WITH_AVX2

#ifdef CAFFE2_PERF_WITH_AVX
// 如果定义了 CAFFE2_PERF_WITH_AVX 宏，则定义 AVX_DO 和 AVX_F16C_DO 宏
#define AVX_DO(funcname, ...)                                               \
  {                                                                         \
    // 静态变量 isDo 标志是否满足 AVX 要求，包括 CPU 信息的初始化和 AVX 的支持
    static const bool isDo = cpuinfo_initialize() && cpuinfo_has_x86_avx(); \
    // 如果满足 AVX 要求，则调用对应的 AVX 版本的函数，并返回结果
    if (isDo) {                                                             \
      return funcname##__avx(__VA_ARGS__);                                  \
    }                                                                       \
  }

#define AVX_F16C_DO(funcname, ...)                                            \
  {                                                                           \
    // 静态变量 isDo 标志是否满足 AVX + F16C 要求，包括 CPU 信息的初始化、AVX 和 F16C 的支持
    static const bool isDo = cpuinfo_initialize() && cpuinfo_has_x86_avx() && \
        cpuinfo_has_x86_f16c();                                               \
    // 如果满足 AVX + F16C 要求，则调用对应的 AVX + F16C 版本的函数，并返回结果
    if (isDo) {                                                               \
      return funcname##__avx_f16c(__VA_ARGS__);                               \
    }                                                                         \
  }
#else // CAFFE2_PERF_WITH_AVX
// 如果未定义 CAFFE2_PERF_WITH_AVX 宏，则定义 AVX_DO 和 AVX_F16C_DO 宏为空
#define AVX_DO(funcname, ...)
#define AVX_F16C_DO(funcname, ...)
#endif // CAFFE2_PERF_WITH_AVX
```