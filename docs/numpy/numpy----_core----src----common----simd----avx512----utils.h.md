# `.\numpy\numpy\_core\src\common\simd\avx512\utils.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_UTILS_H
#define _NPY_SIMD_AVX512_UTILS_H

// 定义将__m512i类型转换为__m256i类型的宏
#define npyv512_lower_si256 _mm512_castsi512_si256
// 定义将__m512类型转换为__m256类型的宏
#define npyv512_lower_ps256 _mm512_castps512_ps256
// 定义将__m512d类型转换为__m256d类型的宏
#define npyv512_lower_pd256 _mm512_castpd512_pd256

// 定义从__m512i类型中提取高128位__m256i类型的宏
#define npyv512_higher_si256(A) _mm512_extracti64x4_epi64(A, 1)
// 定义从__m512d类型中提取高128位__m256d类型的宏
#define npyv512_higher_pd256(A) _mm512_extractf64x4_pd(A, 1)

#ifdef NPY_HAVE_AVX512DQ
    // 如果支持 AVX512DQ，则定义从__m512类型中提取高256位__m256类型的宏
    #define npyv512_higher_ps256(A) _mm512_extractf32x8_ps(A, 1)
#else
    // 如果不支持 AVX512DQ，则通过组合操作从__m512类型中提取高256位__m256类型
    #define npyv512_higher_ps256(A) \
        _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(A), 1))
#endif

// 定义将两个__m256i类型合并为一个__m512i类型的宏
#define npyv512_combine_si256(A, B) _mm512_inserti64x4(_mm512_castsi256_si512(A), B, 1)
// 定义将两个__m256d类型合并为一个__m512d类型的宏
#define npyv512_combine_pd256(A, B) _mm512_insertf64x4(_mm512_castpd256_pd512(A), B, 1)

#ifdef NPY_HAVE_AVX512DQ
    // 如果支持 AVX512DQ，则定义将两个__m256类型合并为一个__m512类型的宏
    #define npyv512_combine_ps256(A, B) _mm512_insertf32x8(_mm512_castps256_ps512(A), B, 1)
#else
    // 如果不支持 AVX512DQ，则通过组合操作将两个__m256类型合并为一个__m512类型
    #define npyv512_combine_ps256(A, B) \
        _mm512_castsi512_ps(npyv512_combine_si256(_mm256_castps_si256(A), _mm256_castps_si256(B)))
#endif

// 定义宏，用于从AVX2转换到AVX512的单参数函数实现，返回__m512i类型
#define NPYV_IMPL_AVX512_FROM_AVX2_1ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512i FN_NAME(__m512i a)               \
    {                                                    \
        __m256i l_a  = npyv512_lower_si256(a);           \
        __m256i h_a  = npyv512_higher_si256(a);          \
        l_a = INTRIN(l_a);                               \
        h_a = INTRIN(h_a);                               \
        return npyv512_combine_si256(l_a, h_a);          \
    }

// 定义宏，用于从AVX2转换到AVX512的单参数函数实现，返回__m512类型
#define NPYV_IMPL_AVX512_FROM_AVX2_PS_1ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512 FN_NAME(__m512 a)                    \
    {                                                       \
        __m256 l_a  = npyv512_lower_ps256(a);               \
        __m256 h_a  = npyv512_higher_ps256(a);              \
        l_a = INTRIN(l_a);                                  \
        h_a = INTRIN(h_a);                                  \
        return npyv512_combine_ps256(l_a, h_a);             \
    }

// 定义宏，用于从AVX2转换到AVX512的单参数函数实现，返回__m512d类型
#define NPYV_IMPL_AVX512_FROM_AVX2_PD_1ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512d FN_NAME(__m512d a)                  \
    {                                                       \
        __m256d l_a  = npyv512_lower_pd256(a);              \
        __m256d h_a  = npyv512_higher_pd256(a);             \
        l_a = INTRIN(l_a);                                  \
        h_a = INTRIN(h_a);                                  \
        return npyv512_combine_pd256(l_a, h_a);             \
    }

// 定义宏，用于从AVX2转换到AVX512的双参数函数实现，返回__m512i类型
#define NPYV_IMPL_AVX512_FROM_AVX2_2ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512i FN_NAME(__m512i a, __m512i b)    \
    {
        # 提取给定 AVX-512 512位向量 a 的低256位部分
        __m256i l_a  = npyv512_lower_si256(a);
        # 提取给定 AVX-512 512位向量 a 的高256位部分
        __m256i h_a  = npyv512_higher_si256(a);
        # 提取给定 AVX-512 512位向量 b 的低256位部分
        __m256i l_b  = npyv512_lower_si256(b);
        # 提取给定 AVX-512 512位向量 b 的高256位部分
        __m256i h_b  = npyv512_higher_si256(b);
        # 对 l_a 和 l_b 进行特定的 INTRIN 操作，结果保存在 l_a 中
        l_a = INTRIN(l_a, l_b);
        # 对 h_a 和 h_b 进行特定的 INTRIN 操作，结果保存在 h_a 中
        h_a = INTRIN(h_a, h_b);
        # 将修改后的 l_a 和 h_a 合并成一个 AVX-512 512位向量，并返回结果
        return npyv512_combine_si256(l_a, h_a);
    }
#define NPYV_IMPL_AVX512_FROM_SI512_PS_2ARG(FN_NAME, INTRIN) \
    // 定义一个内联函数 FN_NAME，接受两个 __m512 类型的参数 a 和 b，返回结果为 __m512 类型
    NPY_FINLINE __m512 FN_NAME(__m512 a, __m512 b)           \
    {                                                        \
        // 将 a 和 b 转换为 __m512i 类型，然后调用 INTRIN 进行处理，最后将结果转换回 __m512 类型并返回
        return _mm512_castsi512_ps(INTRIN(                   \
            _mm512_castps_si512(a), _mm512_castps_si512(b)   \
        ));                                                  \
    }

#define NPYV_IMPL_AVX512_FROM_SI512_PD_2ARG(FN_NAME, INTRIN) \
    // 定义一个内联函数 FN_NAME，接受两个 __m512d 类型的参数 a 和 b，返回结果为 __m512d 类型
    NPY_FINLINE __m512d FN_NAME(__m512d a, __m512d b)        \
    {                                                        \
        // 将 a 和 b 转换为 __m512i 类型，然后调用 INTRIN 进行处理，最后将结果转换回 __m512d 类型并返回
        return _mm512_castsi512_pd(INTRIN(                   \
            _mm512_castpd_si512(a), _mm512_castpd_si512(b)   \
        ));                                                  \
    }

#ifndef NPY_HAVE_AVX512BW
    // 如果没有 AVX512BW 扩展，则使用 AVX2 的 _mm256_packs_epi16 作为 npyv512_packs_epi16 的定义
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv512_packs_epi16,  _mm256_packs_epi16)
#else
    // 否则直接定义 npyv512_packs_epi16 为 _mm512_packs_epi16
    #define npyv512_packs_epi16 _mm512_packs_epi16
#endif

NPY_FINLINE __m256i npyv512_pack_lo_hi(__m512i a) {
    // 提取 a 的低 256 位和高 256 位分别存入 lo 和 hi
    __m256i lo = npyv512_lower_si256(a);
    __m256i hi = npyv512_higher_si256(a);
    // 使用 _mm256_packs_epi32 将 lo 和 hi 中的每对相邻元素进行有符号 16 位整数打包操作，返回结果
    return _mm256_packs_epi32(lo, hi);
}

#endif // _NPY_SIMD_AVX512_UTILS_H
```