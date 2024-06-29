# `.\numpy\numpy\_core\src\common\simd\sse\utils.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_UTILS_H
#define _NPY_SIMD_SSE_UTILS_H

#if !defined(__x86_64__) && !defined(_M_X64)
// 如果不是 x86_64 架构，则定义 npyv128_cvtsi128_si64 函数
NPY_FINLINE npy_int64 npyv128_cvtsi128_si64(__m128i a)
{
    // 创建一个 16 字节对齐的整数数组 idx 来存储 __m128i a 的内容
    npy_int64 NPY_DECL_ALIGNED(16) idx[2];
    // 将 __m128i a 的内容存储到 idx 数组中
    _mm_store_si128((__m128i *)idx, a);
    // 返回 idx 数组的第一个元素
    return idx[0];
}
#else
    // 如果是 x86_64 架构，则直接使用 SSE 提供的 _mm_cvtsi128_si64 函数
    #define npyv128_cvtsi128_si64 _mm_cvtsi128_si64
#endif

#endif // _NPY_SIMD_SSE_UTILS_H
```