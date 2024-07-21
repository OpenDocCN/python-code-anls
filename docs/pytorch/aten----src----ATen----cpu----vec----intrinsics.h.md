# `.\pytorch\aten\src\ATen\cpu\vec\intrinsics.h`

```
#pragma once
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* 如果是 GCC 或者兼容的 clang 编译器，并且目标平台是 x86/x86-64 */
#include <x86intrin.h>
#elif defined(__clang__) && (defined(__ARM_NEON__) || defined(__aarch64__))
/* 如果是兼容 clang 的编译器，并且目标平台是 ARM Neon */
#include <arm_neon.h>
#elif defined(_MSC_VER)
/* 如果是 Microsoft C/C++ 兼容的编译器 */
#include <intrin.h>
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) (_mm_extract_epi64(_mm256_extractf128_si256(X, Y >> 1), Y % 2))
#define _mm256_extract_epi32(X, Y) (_mm_extract_epi32(_mm256_extractf128_si256(X, Y >> 2), Y % 4))
#define _mm256_extract_epi16(X, Y) (_mm_extract_epi16(_mm256_extractf128_si256(X, Y >> 3), Y % 8))
#define _mm256_extract_epi8(X, Y) (_mm_extract_epi8(_mm256_extractf128_si256(X, Y >> 4), Y % 16))
#endif
#elif defined(__GNUC__) && (defined(__ARM_NEON__) || defined(__aarch64__))
/* 如果是 GCC 兼容的编译器，并且目标平台是 ARM Neon */
#include <arm_neon.h>
#if defined (MISSING_ARM_VLD1)
#include <ATen/cpu/vec/vec256/missing_vld1_neon.h>
#elif defined (MISSING_ARM_VST1)
#include <ATen/cpu/vec/vec256/missing_vst1_neon.h>
#endif
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* 如果是 GCC 兼容的编译器，并且目标平台是 ARM WMMX */
#include <mmintrin.h>
#elif defined(__s390x__)
// targets Z/architecture
// we will include vecintrin later
#elif (defined(__GNUC__) || defined(__xlC__)) &&                               \
        (defined(__VEC__) || defined(__ALTIVEC__))
/* 如果是 XLC 或者 GCC 兼容的编译器，并且目标平台是 PowerPC VMX/VSX */
#include <altivec.h>
/* 需要取消 <altivec.h> 定义的这些标记以避免与 C++ 类型冲突 => 仍然可以使用 __bool/__vector */
#undef bool
#undef vector
#undef pixel
#elif defined(__GNUC__) && defined(__SPE__)
/* 如果是 GCC 兼容的编译器，并且目标平台是 PowerPC SPE */
#include <spe.h>
#endif
```