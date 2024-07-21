# `.\pytorch\aten\src\ATen\native\cpu\Intrinsics.h`

```py
#pragma once
// 如果使用的是 Clang 编译器，并且目标是 x86/x86-64 架构
#if defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
/* Clang-compatible compiler, targeting x86/x86-64 */
// 包含 x86 平台的内部函数定义头文件
#include <x86intrin.h>
// 如果使用的是 Microsoft C/C++ 编译器
#elif defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
// 包含 Microsoft 内部函数定义头文件
#include <intrin.h>
// 对于早于等于 1900 版本的 MSC 编译器，定义 _mm256_extract_epi64 宏
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) (((uint64_t*)&X)[Y])
#endif
// 如果使用的是 GCC 编译器，并且目标是 x86/x86-64 架构
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
// 包含 x86 平台的内部函数定义头文件
#include <x86intrin.h>
// 如果使用的是 GCC 编译器，并且目标是 ARM 并且支持 NEON
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
// 包含 ARM NEON 技术的内部函数定义头文件
#include <arm_neon.h>
// 如果使用的是 GCC 编译器，并且目标是 ARM 并且支持 WMMX
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
// 包含 ARM WMMX 技术的内部函数定义头文件
#include <mmintrin.h>
// 如果使用的是 XLC 或者 GCC 编译器，并且目标是 PowerPC 并且支持 VMX/VSX
#elif (defined(__GNUC__) || defined(__xlC__)) && \
    (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
// 包含 PowerPC VMX/VSX 技术的内部函数定义头文件
#include <altivec.h>
// 我们需要取消定义 <altivec.h> 中的一些标记，以避免与 C++ 类型产生冲突
#undef bool
#undef vector
#undef pixel
// 如果使用的是 GCC 编译器，并且目标是 PowerPC 并且支持 SPE
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
// 包含 PowerPC SPE 技术的内部函数定义头文件
#include <spe.h>
#endif
```