# `.\numpy\numpy\_core\src\common\simd\avx512\reorder.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_REORDER_H
#define _NPY_SIMD_AVX512_REORDER_H

// 定义宏：将两个向量的低部分合并
#define npyv_combinel_u8(A, B) _mm512_inserti64x4(A, _mm512_castsi512_si256(B), 1)
#define npyv_combinel_s8  npyv_combinel_u8
#define npyv_combinel_u16 npyv_combinel_u8
#define npyv_combinel_s16 npyv_combinel_u8
#define npyv_combinel_u32 npyv_combinel_u8
#define npyv_combinel_s32 npyv_combinel_u8
#define npyv_combinel_u64 npyv_combinel_u8
#define npyv_combinel_s64 npyv_combinel_u8
#define npyv_combinel_f64(A, B) _mm512_insertf64x4(A, _mm512_castpd512_pd256(B), 1)
#ifdef NPY_HAVE_AVX512DQ
    // 定义宏：将两个单精度浮点数向量的低部分合并
    #define npyv_combinel_f32(A, B) \
        _mm512_insertf32x8(A, _mm512_castps512_ps256(B), 1)
#else
    // 定义宏：将两个单精度浮点数向量的低部分合并（替代实现）
    #define npyv_combinel_f32(A, B) \
        _mm512_castsi512_ps(npyv_combinel_u8(_mm512_castps_si512(A), _mm512_castps_si512(B)))
#endif

// 定义宏：将两个向量的高部分合并
#define npyv_combineh_u8(A, B) _mm512_inserti64x4(B, _mm512_extracti64x4_epi64(A, 1), 0)
#define npyv_combineh_s8  npyv_combineh_u8
#define npyv_combineh_u16 npyv_combineh_u8
#define npyv_combineh_s16 npyv_combineh_u8
#define npyv_combineh_u32 npyv_combineh_u8
#define npyv_combineh_s32 npyv_combineh_u8
#define npyv_combineh_u64 npyv_combineh_u8
#define npyv_combineh_s64 npyv_combineh_u8
#define npyv_combineh_f64(A, B) _mm512_insertf64x4(B, _mm512_extractf64x4_pd(A, 1), 0)
#ifdef NPY_HAVE_AVX512DQ
    // 定义宏：将两个单精度浮点数向量的高部分合并
    #define npyv_combineh_f32(A, B) \
        _mm512_insertf32x8(B, _mm512_extractf32x8_ps(A, 1), 0)
#else
    // 定义宏：将两个单精度浮点数向量的高部分合并（替代实现）
    #define npyv_combineh_f32(A, B) \
        _mm512_castsi512_ps(npyv_combineh_u8(_mm512_castps_si512(A), _mm512_castps_si512(B)))
#endif

// 定义函数：从两个整型向量 a 和 b 中组合得到一个 m512ix2 结构体
NPY_FINLINE npyv_m512ix2 npyv__combine(__m512i a, __m512i b)
{
    npyv_m512ix2 r;
    // 将向量 a 和 b 的低部分合并
    r.val[0] = npyv_combinel_u8(a, b);
    // 将向量 a 和 b 的高部分合并
    r.val[1] = npyv_combineh_u8(a, b);
    return r;
}

// 定义函数：从两个单精度浮点数向量 a 和 b 中组合得到一个 f32x2 结构体
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m512 a, __m512 b)
{
    npyv_f32x2 r;
    // 将单精度浮点数向量 a 和 b 的低部分合并
    r.val[0] = npyv_combinel_f32(a, b);
    // 将单精度浮点数向量 a 和 b 的高部分合并
    r.val[1] = npyv_combineh_f32(a, b);
    return r;
}

// 定义函数：从两个双精度浮点数向量 a 和 b 中组合得到一个 f64x2 结构体
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m512d a, __m512d b)
{
    npyv_f64x2 r;
    // 将双精度浮点数向量 a 和 b 的低部分合并
    r.val[0] = npyv_combinel_f64(a, b);
    // 将双精度浮点数向量 a 和 b 的高部分合并
    r.val[1] = npyv_combineh_f64(a, b);
    return r;
}

// 定义宏：将两个整型向量的低部分合并，实际调用 npyv__combine 函数
#define npyv_combine_u8  npyv__combine
#define npyv_combine_s8  npyv__combine
#define npyv_combine_u16 npyv__combine
#define npyv_combine_s16 npyv__combine
#define npyv_combine_u32 npyv__combine
#define npyv_combine_s32 npyv__combine
#define npyv_combine_u64 npyv__combine
#define npyv_combine_s64 npyv__combine

// 定义宏：插入两个向量的低部分，根据 AVX512BW 的可用性选择实现
#ifndef NPY_HAVE_AVX512BW
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv__unpacklo_epi8,  _mm256_unpacklo_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv__unpackhi_epi8,  _mm256_unpackhi_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv__unpacklo_epi16, _mm256_unpacklo_epi16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv__unpackhi_epi16, _mm256_unpackhi_epi16)
#endif

#endif // _NPY_SIMD_AVX512_REORDER_H
# 定义一个内联函数，用于将两个 __m512i 类型的寄存器按特定规则进行元素混合并返回结果
NPY_FINLINE npyv_u64x2 npyv_zip_u64(__m512i a, __m512i b)
{
    # 声明一个存放结果的结构体
    npyv_u64x2 r;
    # 将第一个结果元素混合计算并存入结果结构体的第一个成员中
    r.val[0] = _mm512_permutex2var_epi64(a, npyv_set_u64(0, 8, 1, 9, 2, 10, 3, 11), b);
    # 将第二个结果元素混合计算并存入结果结构体的第二个成员中
    r.val[1] = _mm512_permutex2var_epi64(a, npyv_set_u64(4, 12, 5, 13, 6, 14, 7, 15), b);
    # 返回混合后的结果结构体
    return r;
}
# 定义一个宏，用于有符号 64 位整数的元素混合，实际上调用了无符号版本的混合函数
#define npyv_zip_s64 npyv_zip_u64

# 定义一个内联函数，用于将两个 __m512i 类型的寄存器按特定规则进行元素混合并返回结果
NPY_FINLINE npyv_u8x2 npyv_zip_u8(__m512i a, __m512i b)
{
    # 声明一个存放结果的结构体
    npyv_u8x2 r;
    # 如果支持 AVX512VBMI 指令集，则按照特定的字节顺序混合计算结果
#ifdef NPY_HAVE_AVX512VBMI
    r.val[0] = _mm512_permutex2var_epi8(a,
        npyv_set_u8(0,  64, 1,  65, 2,  66, 3,  67, 4,  68, 5,  69, 6,  70, 7,  71,
                    8,  72, 9,  73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79,
                    16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87,
                    24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95), b);
    r.val[1] = _mm512_permutex2var_epi8(a,
        npyv_set_u8(32, 96,  33, 97,  34, 98,  35, 99,  36, 100, 37, 101, 38, 102, 39, 103,
                    40, 104, 41, 105, 42, 106, 43, 107, 44, 108, 45, 109, 46, 110, 47, 111,
                    48, 112, 49, 113, 50, 114, 51, 115, 52, 116, 53, 117, 54, 118, 55, 119,
                    56, 120, 57, 121, 58, 122, 59, 123, 60, 124, 61, 125, 62, 126, 63, 127), b);
#else
    #ifdef NPY_HAVE_AVX512BW
    # 否则，根据 AVX512BW 指令集进行字节的拆分和混合
    __m512i ab0 = _mm512_unpacklo_epi8(a, b);
    __m512i ab1 = _mm512_unpackhi_epi8(a, b);
    #else
    # 如果以上指令集都不支持，则调用其他适配的函数进行低位和高位的字节拆分
    __m512i ab0 = npyv__unpacklo_epi8(a, b);
    __m512i ab1 = npyv__unpackhi_epi8(a, b);
    #endif
    # 将拆分后的结果进行 64 位整数的混合计算
    r.val[0] = _mm512_permutex2var_epi64(ab0, npyv_set_u64(0, 1, 8, 9, 2, 3, 10, 11), ab1);
    r.val[1] = _mm512_permutex2var_epi64(ab0, npyv_set_u64(4, 5, 12, 13, 6, 7, 14, 15), ab1);
#endif
    # 返回混合后的结果结构体
    return r;
}
# 定义一个宏，用于有符号 8 位整数的元素混合，实际上调用了无符号版本的混合函数
#define npyv_zip_s8 npyv_zip_u8

# 定义一个内联函数，用于将两个 __m512i 类型的寄存器按特定规则进行元素混合并返回结果
NPY_FINLINE npyv_u16x2 npyv_zip_u16(__m512i a, __m512i b)
{
    # 声明一个存放结果的结构体
    npyv_u16x2 r;
    # 如果支持 AVX512BW 指令集，则按照特定的 16 位整数顺序混合计算结果
#ifdef NPY_HAVE_AVX512BW
    r.val[0] = _mm512_permutex2var_epi16(a,
        npyv_set_u16(0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                     8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47), b);
    r.val[1] = _mm512_permutex2var_epi16(a,
        npyv_set_u16(16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
                     24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63), b);
#else
    # 否则，调用其他适配的函数进行低位和高位的 16 位整数的拆分
    __m512i ab0 = npyv__unpacklo_epi16(a, b);
    __m512i ab1 = npyv__unpackhi_epi16(a, b);
    # 将拆分后的结果进行 64 位整数的混合计算
    r.val[0] = _mm512_permutex2var_epi64(ab0, npyv_set_u64(0, 1, 8, 9, 2, 3, 10, 11), ab1);
    r.val[1] = _mm512_permutex2var_epi64(ab0, npyv_set_u64(4, 5, 12, 13, 6, 7, 14, 15), ab1);
#endif
    # 返回混合后的结果结构体
    return r;
}
# 定义一个宏，用于有符号 16 位整数的元素混合，实际上调用了无符号版本的混合函数
#define npyv_zip_s16 npyv_zip_u16

# 定义一个内联函数，用于将两个 __m512i 类型的寄存器按特定规则进行元素混合并返回结果
NPY_FINLINE npyv_u32x2 npyv_zip_u32(__m512i a, __m512i b)
{
    # 声明一个存放
// 定义一个内联函数，用于将两个 __m512 类型的向量 a 和 b 进行压缩合并成一个 npyv_f32x2 结构体返回
NPY_FINLINE npyv_f32x2 npyv_zip_f32(__m512 a, __m512 b)
{
    // 声明一个 npyv_f32x2 结构体 r，用于存储结果
    npyv_f32x2 r;
    // 使用 _mm512_permutex2var_ps 函数，根据给定的索引将向量 a 和 b 进行混合，结果存入 r.val[0]
    r.val[0] = _mm512_permutex2var_ps(a,
        npyv_set_u32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23), b);
    // 同上，将向量 a 和 b 进行混合，结果存入 r.val[1]
    r.val[1] = _mm512_permutex2var_ps(a,
        npyv_set_u32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31), b);
    // 返回结果结构体 r
    return r;
}

// 定义一个内联函数，用于将两个 __m512d 类型的双精度向量 a 和 b 进行压缩合并成一个 npyv_f64x2 结构体返回
NPY_FINLINE npyv_f64x2 npyv_zip_f64(__m512d a, __m512d b)
{
    // 声明一个 npyv_f64x2 结构体 r，用于存储结果
    npyv_f64x2 r;
    // 使用 _mm512_permutex2var_pd 函数，根据给定的索引将双精度向量 a 和 b 进行混合，结果存入 r.val[0]
    r.val[0] = _mm512_permutex2var_pd(a, npyv_set_u64(0, 8, 1, 9, 2, 10, 3, 11), b);
    // 同上，将双精度向量 a 和 b 进行混合，结果存入 r.val[1]
    r.val[1] = _mm512_permutex2var_pd(a, npyv_set_u64(4, 12, 5, 13, 6, 14, 7, 15), b);
    // 返回结果结构体 r
    return r;
}

// 定义一个内联函数，用于将两个 npyv_u8 类型的向量 ab0 和 ab1 进行解交错，结果存入 npyv_u8x2 结构体返回
// 如果支持 AVX512VBMI 指令集，则使用 _mm512_permutex2var_epi8 函数进行解交错，否则根据是否支持 AVX512BW 选择相应的处理方式
NPY_FINLINE npyv_u8x2 npyv_unzip_u8(npyv_u8 ab0, npyv_u8 ab1)
{
    // 声明一个 npyv_u8x2 结构体 r，用于存储解交错后的结果
    npyv_u8x2 r;
#ifdef NPY_HAVE_AVX512VBMI
    // 如果支持 AVX512VBMI，则定义解交错所需的索引 idx_a 和 idx_b
    const __m512i idx_a = npyv_set_u8(
        0,  2,  4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,  26,  28,  30,
        32, 34, 36,  38,  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62,
        64, 66, 68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,
        96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126
    );
    const __m512i idx_b = npyv_set_u8(
        1,  3,  5,   7,   9,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
        33, 35, 37,  39,  41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63,
        65, 67, 69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89,  91,  93,  95,
        97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127
    );
    // 使用 _mm512_permutex2var_epi8 函数，根据 idx_a 和 idx_b 进行解交错，结果存入 r.val[0] 和 r.val[1]
    r.val[0] = _mm512_permutex2var_epi8(ab0, idx_a, ab1);
    r.val[1] = _mm512_permutex2var_epi8(ab0, idx_b, ab1);
#else
    // 如果不支持 AVX512VBMI，根据是否支持 AVX512BW 选择相应的处理方式
    #ifdef NPY_HAVE_AVX512BW
        // 如果支持 AVX512BW，则定义解交错所需的 idx
        const __m512i idx = npyv_set_u8(
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
        );
        // 使用 _mm512_shuffle_epi8 函数，根据 idx 进行解交错，结果存入 abl 和 abh
        __m512i abl = _mm512_shuffle_epi8(ab0, idx);
        __m512i abh = _mm512_shuffle_epi8(ab1, idx);
    #else
        // 如果以上都不支持，则使用 AVX2 指令集处理
        const __m256i idx = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
        );
        // 使用 _mm256_shuffle_epi8 函数，根据 idx 对 ab0 和 ab1 进行解交错，结果存入对应的低位和高位部分
        __m256i abl_lo = _mm256_shuffle_epi8(npyv512_lower_si256(ab0),  idx);
        __m256i abl_hi = _mm256_shuffle_epi
    return r;


# 返回变量 r 的值作为函数的结果
#define npyv_unzip_s8 npyv_unzip_u8



NPY_FINLINE npyv_u16x2 npyv_unzip_u16(npyv_u16 ab0, npyv_u16 ab1)
{
    npyv_u16x2 r;
#ifdef NPY_HAVE_AVX512BW
    // 如果支持 AVX-512 BW 指令集，则定义两个索引向量 idx_a 和 idx_b
    const __m512i idx_a = npyv_set_u16(
        0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62
    );
    const __m512i idx_b = npyv_set_u16(
        1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63
    );
    // 使用 permute 操作按照 idx_a 和 idx_b 重新排列输入的 ab0 和 ab1，存储到结果 r 中
    r.val[0] = _mm512_permutex2var_epi16(ab0, idx_a, ab1);
    r.val[1] = _mm512_permutex2var_epi16(ab0, idx_b, ab1);
#else
    // 如果不支持 AVX-512 BW 指令集，则定义一个字节级别的索引向量 idx
    const __m256i idx = _mm256_setr_epi8(
        0,1, 4,5, 8,9, 12,13, 2,3, 6,7, 10,11, 14,15,
        0,1, 4,5, 8,9, 12,13, 2,3, 6,7, 10,11, 14,15
    );
    // 对输入向量 ab0 和 ab1 进行字节级别的 shuffle 操作，以得到结果 r
    __m256i abl_lo = _mm256_shuffle_epi8(npyv512_lower_si256(ab0),  idx);
    __m256i abl_hi = _mm256_shuffle_epi8(npyv512_higher_si256(ab0), idx);
    __m256i abh_lo = _mm256_shuffle_epi8(npyv512_lower_si256(ab1),  idx);
    __m256i abh_hi = _mm256_shuffle_epi8(npyv512_higher_si256(ab1), idx);
    __m512i abl = npyv512_combine_si256(abl_lo, abl_hi);
    __m512i abh = npyv512_combine_si256(abh_lo, abh_hi);

    // 定义两个 64 位整数型的索引向量 idx_a 和 idx_b
    const __m512i idx_a = npyv_set_u64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i idx_b = npyv_set_u64(1, 3, 5, 7, 9, 11, 13, 15);
    // 使用 permute 操作按照 idx_a 和 idx_b 重新排列 abl 和 abh，存储到结果 r 中
    r.val[0] = _mm512_permutex2var_epi64(abl, idx_a, abh);
    r.val[1] = _mm512_permutex2var_epi64(abl, idx_b, abh);
#endif
    return r;
}



#define npyv_unzip_s16 npyv_unzip_u16



NPY_FINLINE npyv_u32x2 npyv_unzip_u32(npyv_u32 ab0, npyv_u32 ab1)
{
    // 定义两个 32 位整数型的索引向量 idx_a 和 idx_b
    const __m512i idx_a = npyv_set_u32(
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    );
    const __m512i idx_b = npyv_set_u32(
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
    );
    npyv_u32x2 r;
    // 使用 permute 操作按照 idx_a 和 idx_b 重新排列输入的 ab0 和 ab1，存储到结果 r 中
    r.val[0] = _mm512_permutex2var_epi32(ab0, idx_a, ab1);
    r.val[1] = _mm512_permutex2var_epi32(ab0, idx_b, ab1);
    return r;
}



#define npyv_unzip_s32 npyv_unzip_u32



NPY_FINLINE npyv_u64x2 npyv_unzip_u64(npyv_u64 ab0, npyv_u64 ab1)
{
    // 定义两个 64 位整数型的索引向量 idx_a 和 idx_b
    const __m512i idx_a = npyv_set_u64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i idx_b = npyv_set_u64(1, 3, 5, 7, 9, 11, 13, 15);
    npyv_u64x2 r;
    // 使用 permute 操作按照 idx_a 和 idx_b 重新排列输入的 ab0 和 ab1，存储到结果 r 中
    r.val[0] = _mm512_permutex2var_epi64(ab0, idx_a, ab1);
    r.val[1] = _mm512_permutex2var_epi64(ab0, idx_b, ab1);
    return r;
}



#define npyv_unzip_s64 npyv_unzip_u64



NPY_FINLINE npyv_f32x2 npyv_unzip_f32(npyv_f32 ab0, npyv_f32 ab1)
{
    // 定义两个 32 位整数型的索引向量 idx_a 和 idx_b
    const __m512i idx_a = npyv_set_u32(
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    );
    const __m512i idx_b = npyv_set_u32(
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
    );
    npyv_f32x2 r;
    // 使用 permute 操作按照 idx_a 和 idx_b 重新排列输入的 ab0 和 ab1，存储到结果 r 中
    r.val[0] = _mm512_permutex2var_ps(ab0, idx_a, ab1);
    r.val[1] = _mm512_permutex2var_ps(ab0, idx_b, ab1);
    return r;
}



NPY_FINLINE npyv_f64x2 npyv_unzip_f64(npyv_f64 ab0, npyv_f64 ab1)
{
    // 定义两个 64 位整数型的索引向量 idx_a
    const __m512i idx_a = npyv_set_u64(0, 2, 4, 6, 8, 10, 12, 14);
    npyv_f64x2 r;
    // 使用 permute 操作按照 idx_a 重新排列输入的 ab0 和 ab1，存储到结果 r 中
    r.val[0] = _mm512_permutex2var_pd(ab0, idx_a, ab1);
    return r;
}
    // 创建一个包含固定顺序的无符号64位整数的__m512i类型变量，用于索引操作
    const __m512i idx_b = npyv_set_u64(1, 3, 5, 7, 9, 11, 13, 15);
    // 创建一个双精度浮点数向量变量r，包含两个元素
    npyv_f64x2 r;
    // 使用_mm512_permutex2var_pd函数对两个双精度浮点数向量ab0和ab1进行按索引混合操作，并将结果存入r的第一个元素
    r.val[0] = _mm512_permutex2var_pd(ab0, idx_a, ab1);
    // 使用_mm512_permutex2var_pd函数对两个双精度浮点数向量ab0和ab1进行按固定索引顺序（idx_b）混合操作，并将结果存入r的第二个元素
    r.val[1] = _mm512_permutex2var_pd(ab0, idx_b, ab1);
    // 返回双精度浮点数向量r
    return r;
#ifdef NPY_HAVE_AVX512BW
    // 如果编译器支持 AVX-512BW 指令集，则使用 AVX-512 实现反转每个 64 位 lane 的操作
    const __m512i idx = npyv_set_u8(
        // 创建 AVX-512 需要的索引，用于反转每个 64 位 lane 中的元素
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8
    );
    // 使用 AVX-512 指令集中的 shuffle 操作进行反转操作
    return _mm512_shuffle_epi8(a, idx);
#else
    // 如果编译器不支持 AVX-512BW 指令集，则使用 AVX2 实现反转每个 64 位 lane 的操作
    const __m256i idx = _mm256_setr_epi8(
        // 创建 AVX2 需要的索引，用于反转每个 64 位 lane 中的元素
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
    );
    // 分别对高低 256 位执行 shuffle 操作，然后将结果合并
    __m256i lo = _mm256_shuffle_epi8(npyv512_lower_si256(a),  idx);
    __m256i hi = _mm256_shuffle_epi8(npyv512_higher_si256(a), idx);
    return npyv512_combine_si256(lo, hi);
#endif
}
```