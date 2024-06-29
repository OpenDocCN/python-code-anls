# `.\numpy\numpy\_core\src\common\simd\sse\reorder.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_REORDER_H
#define _NPY_SIMD_SSE_REORDER_H

// 定义宏：将两个向量的低部分合并
#define npyv_combinel_u8  _mm_unpacklo_epi64
#define npyv_combinel_s8  _mm_unpacklo_epi64
#define npyv_combinel_u16 _mm_unpacklo_epi64
#define npyv_combinel_s16 _mm_unpacklo_epi64
#define npyv_combinel_u32 _mm_unpacklo_epi64
#define npyv_combinel_s32 _mm_unpacklo_epi64
#define npyv_combinel_u64 _mm_unpacklo_epi64
#define npyv_combinel_s64 _mm_unpacklo_epi64
// 将两个单精度浮点数向量的低部分合并
#define npyv_combinel_f32(A, B) _mm_castsi128_ps(_mm_unpacklo_epi64(_mm_castps_si128(A), _mm_castps_si128(B)))
// 将两个双精度浮点数向量的低部分合并
#define npyv_combinel_f64 _mm_unpacklo_pd

// 定义宏：将两个向量的高部分合并
#define npyv_combineh_u8  _mm_unpackhi_epi64
#define npyv_combineh_s8  _mm_unpackhi_epi64
#define npyv_combineh_u16 _mm_unpackhi_epi64
#define npyv_combineh_s16 _mm_unpackhi_epi64
#define npyv_combineh_u32 _mm_unpackhi_epi64
#define npyv_combineh_s32 _mm_unpackhi_epi64
#define npyv_combineh_u64 _mm_unpackhi_epi64
#define npyv_combineh_s64 _mm_unpackhi_epi64
// 将两个单精度浮点数向量的高部分合并
#define npyv_combineh_f32(A, B) _mm_castsi128_ps(_mm_unpackhi_epi64(_mm_castps_si128(A), _mm_castps_si128(B)))
// 将两个双精度浮点数向量的高部分合并
#define npyv_combineh_f64 _mm_unpackhi_pd

// 定义函数：从两个整型向量中提取低部分和高部分组成一个结构体
NPY_FINLINE npyv_m128ix2 npyv__combine(__m128i a, __m128i b)
{
    npyv_m128ix2 r;
    r.val[0] = npyv_combinel_u8(a, b); // 提取 a 和 b 的低部分组合成 r 的第一个值
    r.val[1] = npyv_combineh_u8(a, b); // 提取 a 和 b 的高部分组合成 r 的第二个值
    return r; // 返回包含低部分和高部分的结构体
}

// 定义函数：从两个单精度浮点数向量中提取低部分和高部分组成一个结构体
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m128 a, __m128 b)
{
    npyv_f32x2 r;
    r.val[0] = npyv_combinel_f32(a, b); // 提取 a 和 b 的低部分组合成 r 的第一个值
    r.val[1] = npyv_combineh_f32(a, b); // 提取 a 和 b 的高部分组合成 r 的第二个值
    return r; // 返回包含低部分和高部分的结构体
}

// 定义函数：从两个双精度浮点数向量中提取低部分和高部分组成一个结构体
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m128d a, __m128d b)
{
    npyv_f64x2 r;
    r.val[0] = npyv_combinel_f64(a, b); // 提取 a 和 b 的低部分组合成 r 的第一个值
    r.val[1] = npyv_combineh_f64(a, b); // 提取 a 和 b 的高部分组合成 r 的第二个值
    return r; // 返回包含低部分和高部分的结构体
}

// 定义宏：从两个向量中提取低部分和高部分组合成一个结构体的函数模板
#define NPYV_IMPL_SSE_ZIP(T_VEC, SFX, INTR_SFX)            \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b) \
    {                                                      \
        T_VEC##x2 r;                                       \
        r.val[0] = _mm_unpacklo_##INTR_SFX(a, b);          // 提取 a 和 b 的低部分组合成 r 的第一个值
        r.val[1] = _mm_unpackhi_##INTR_SFX(a, b);          // 提取 a 和 b 的高部分组合成 r 的第二个值
        return r;                                          // 返回包含低部分和高部分的结构体
    }

// 实现各种数据类型的函数模板，用于从两个向量中提取低部分和高部分组合成一个结构体
NPYV_IMPL_SSE_ZIP(npyv_u8,  u8,  epi8)
NPYV_IMPL_SSE_ZIP(npyv_s8,  s8,  epi8)
NPYV_IMPL_SSE_ZIP(npyv_u16, u16, epi16)
NPYV_IMPL_SSE_ZIP(npyv_s16, s16, epi16)
NPYV_IMPL_SSE_ZIP(npyv_u32, u32, epi32)
NPYV_IMPL_SSE_ZIP(npyv_s32, s32, epi32)
NPYV_IMPL_SSE_ZIP(npyv_u64, u64, epi64)
NPYV_IMPL_SSE_ZIP(npyv_s64, s64, epi64)
NPYV_IMPL_SSE_ZIP(npyv_f32, f32, ps)
NPYV_IMPL_SSE_ZIP(npyv_f64, f64, pd)

#endif // _NPY_SIMD_SSE_REORDER_H
// 将两个向量解交错
NPY_FINLINE npyv_u8x2 npyv_unzip_u8(npyv_u8 ab0, npyv_u8 ab1)
{
#ifdef NPY_HAVE_SSSE3
    // 创建用于解交错的索引向量
    const __m128i idx = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
    );
    // 使用索引向量解交错第一个输入向量
    __m128i abl = _mm_shuffle_epi8(ab0, idx);
    // 使用索引向量解交错第二个输入向量
    __m128i abh = _mm_shuffle_epi8(ab1, idx);
    // 结合解交错后的两部分向量为一个新的向量
    return npyv_combine_u8(abl, abh);
#else
    // 如果没有 SSSE3 支持，则手动解交错
    __m128i ab_083b = _mm_unpacklo_epi8(ab0, ab1);
    __m128i ab_4c6e = _mm_unpackhi_epi8(ab0, ab1);
    __m128i ab_048c = _mm_unpacklo_epi8(ab_083b, ab_4c6e);
    __m128i ab_36be = _mm_unpackhi_epi8(ab_083b, ab_4c6e);
    __m128i ab_0346 = _mm_unpacklo_epi8(ab_048c, ab_36be);
    __m128i ab_8bc8 = _mm_unpackhi_epi8(ab_048c, ab_36be);
    // 创建结果向量结构体
    npyv_u8x2 r;
    r.val[0] = _mm_unpacklo_epi8(ab_0346, ab_8bc8);
    r.val[1] = _mm_unpackhi_epi8(ab_0346, ab_8bc8);
    return r;
#endif
}
#define npyv_unzip_s8 npyv_unzip_u8

// 解交错两个 uint16_t 向量
NPY_FINLINE npyv_u16x2 npyv_unzip_u16(npyv_u16 ab0, npyv_u16 ab1)
{
#ifdef NPY_HAVE_SSSE3
    // 创建用于解交错的索引向量
    const __m128i idx = _mm_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
    );
    // 使用索引向量解交错第一个输入向量
    __m128i abl = _mm_shuffle_epi8(ab0, idx);
    // 使用索引向量解交错第二个输入向量
    __m128i abh = _mm_shuffle_epi8(ab1, idx);
    // 结合解交错后的两部分向量为一个新的向量
    return npyv_combine_u16(abl, abh);
#else
    // 如果没有 SSSE3 支持，则手动解交错
    __m128i ab_0415 = _mm_unpacklo_epi16(ab0, ab1);
    __m128i ab_263f = _mm_unpackhi_epi16(ab0, ab1);
    __m128i ab_0246 = _mm_unpacklo_epi16(ab_0415, ab_263f);
    __m128i ab_135f = _mm_unpackhi_epi16(ab_0415, ab_263f);
    // 创建结果向量结构体
    npyv_u16x2 r;
    r.val[0] = _mm_unpacklo_epi16(ab_0246, ab_135f);
    r.val[1] = _mm_unpackhi_epi16(ab_0246, ab_135f);
    return r;
#endif
}
#define npyv_unzip_s16 npyv_unzip_u16

// 解交错两个 uint32_t 向量
NPY_FINLINE npyv_u32x2 npyv_unzip_u32(npyv_u32 ab0, npyv_u32 ab1)
{
    // 使用预定义的掩码进行解交错操作
    __m128i abl = _mm_shuffle_epi32(ab0, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i abh = _mm_shuffle_epi32(ab1, _MM_SHUFFLE(3, 1, 2, 0));
    // 结合解交错后的两部分向量为一个新的向量
    return npyv_combine_u32(abl, abh);
}
#define npyv_unzip_s32 npyv_unzip_u32

// 直接将两个 uint64_t 向量结合成一个
NPY_FINLINE npyv_u64x2 npyv_unzip_u64(npyv_u64 ab0, npyv_u64 ab1)
{
    return npyv_combine_u64(ab0, ab1);
}
#define npyv_unzip_s64 npyv_unzip_u64

// 解交错两个 float 向量
NPY_FINLINE npyv_f32x2 npyv_unzip_f32(npyv_f32 ab0, npyv_f32 ab1)
{
    // 使用预定义的掩码进行解交错操作
    npyv_f32x2 r;
    r.val[0] = _mm_shuffle_ps(ab0, ab1, _MM_SHUFFLE(2, 0, 2, 0));
    r.val[1] = _mm_shuffle_ps(ab0, ab1, _MM_SHUFFLE(3, 1, 3, 1));
    return r;
}

// 直接将两个 double 向量结合成一个
NPY_FINLINE npyv_f64x2 npyv_unzip_f64(npyv_f64 ab0, npyv_f64 ab1)
{
    return npyv_combine_f64(ab0, ab1);
}

// 反转每个 64 位通道中的元素顺序
NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
#ifdef NPY_HAVE_SSSE3
    // 创建用于反转的索引向量
    const __m128i idx = _mm_setr_epi8(
        6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9
    );
    // 使用索引向量反转输入向量
    return _mm_shuffle_epi8(a, idx);
#else
    // 如果没有 SSSE3 支持，则手动反转
    __m128i lo = _mm_shufflelo_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm_shufflehi_epi16(lo, _MM_SHUFFLE(0, 1, 2, 3));
#endif
}
#define npyv_rev64_s16 npyv_rev64_u16

// 反转每个 64 位通道中的元素顺序
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
#ifdef NPY_HAVE_SSSE3
    // 创建用于反转的索引向量
    const __m128i idx = _mm_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
    );
    // 使用索引向量反转输入向量
    return _mm_shuffle_epi8(a, idx);
#else
    // 如果没有 SSSE3 支持，则手动反转
    __m128i lo = _mm_shufflelo_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm_shufflehi_epi16(lo, _MM_SHUFFLE(0, 1, 2, 3));
#endif
}
    );
    # 调用 SSE2 指令集中的 _mm_shuffle_epi8 函数，并传入参数 a 和 idx
    return _mm_shuffle_epi8(a, idx);
    # 返回 _mm_shuffle_epi8 函数的结果
#else
    __m128i rev16 = npyv_rev64_u16(a);
    // 使用 npyv_rev64_u16 函数对参数 a 进行逆序操作，返回结果存入 rev16 中
    // 交换 8 位的对
    return _mm_or_si128(_mm_slli_epi16(rev16, 8), _mm_srli_epi16(rev16, 8));
#endif
}
#define npyv_rev64_s8 npyv_rev64_u8

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    // 使用 SSE 指令对参数 a 进行 32 位元素的逆序操作
    return _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1));
}
#define npyv_rev64_s32 npyv_rev64_u32

NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{
    // 使用 SSE 指令对参数 a 进行 32 位浮点数元素的逆序操作
    return _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
}

// 对每个 128 位 lane 的元素进行按照指定索引的置换操作
#define npyv_permi128_u32(A, E0, E1, E2, E3) \
    _mm_shuffle_epi32(A, _MM_SHUFFLE(E3, E2, E1, E0))

#define npyv_permi128_s32 npyv_permi128_u32

// 对每个 128 位 lane 的元素进行按照指定索引的置换操作，针对 64 位无符号整数
#define npyv_permi128_u64(A, E0, E1) \
    _mm_shuffle_epi32(A, _MM_SHUFFLE(((E1)<<1)+1, ((E1)<<1), ((E0)<<1)+1, ((E0)<<1)))

#define npyv_permi128_s64 npyv_permi128_u64

// 对每个 128 位 lane 的元素进行按照指定索引的置换操作，针对 32 位浮点数
#define npyv_permi128_f32(A, E0, E1, E2, E3) \
    _mm_shuffle_ps(A, A, _MM_SHUFFLE(E3, E2, E1, E0))

// 对每个 128 位 lane 的元素进行按照指定索引的置换操作，针对 64 位浮点数
#define npyv_permi128_f64(A, E0, E1) \
    _mm_shuffle_pd(A, A, _MM_SHUFFLE2(E1, E0))

#endif // _NPY_SIMD_SSE_REORDER_H


注释完成。
```