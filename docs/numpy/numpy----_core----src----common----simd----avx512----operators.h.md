# `.\numpy\numpy\_core\src\common\simd\avx512\operators.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_OPERATORS_H
#define _NPY_SIMD_AVX512_OPERATORS_H

#include "conversion.h" // tobits

/***************************
 * Shifting
 ***************************/

// left
#ifdef NPY_HAVE_AVX512BW
    #define npyv_shl_u16(A, C) _mm512_sll_epi16(A, _mm_cvtsi32_si128(C))
#else
    // 定义 AVX512BW 不可用时的位移操作函数，参数为 512 位整数向量和整数常量
    #define NPYV_IMPL_AVX512_SHIFT(FN, INTRIN)          \
        NPY_FINLINE __m512i npyv_##FN(__m512i a, int c) \
        {                                               \
            // 将 512 位向量拆分为两个 256 位向量
            __m256i l  = npyv512_lower_si256(a);        \
            __m256i h  = npyv512_higher_si256(a);       \
            // 将整数常量转换为 128 位的数据类型
            __m128i cv = _mm_cvtsi32_si128(c);          \
            // 对低位和高位向量分别进行指定的位移操作
            l = _mm256_##INTRIN(l, cv);                 \
            h = _mm256_##INTRIN(h, cv);                 \
            // 将结果重新组合成一个 512 位向量返回
            return npyv512_combine_si256(l, h);         \
        }

    // 定义具体的位移函数 npyv_shl_u16，使用 AVX2 指令集的 sll_epi16 函数
    NPYV_IMPL_AVX512_SHIFT(shl_u16, sll_epi16)
#endif
#define npyv_shl_s16 npyv_shl_u16
// 定义其他整数类型的左移操作宏，参数为向量和整数常量，使用对应的 AVX512 指令
#define npyv_shl_u32(A, C) _mm512_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s32(A, C) _mm512_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_u64(A, C) _mm512_sll_epi64(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s64(A, C) _mm512_sll_epi64(A, _mm_cvtsi32_si128(C))

// left by an immediate constant
#ifdef NPY_HAVE_AVX512BW
    // 如果支持 AVX512BW，则使用对应的 slli_epi16 函数进行左移操作
    #define npyv_shli_u16 _mm512_slli_epi16
#else
    // 否则使用之前定义的 npyv_shl_u16 宏进行左移操作
    #define npyv_shli_u16 npyv_shl_u16
#endif
// 定义其他整数类型的按常量左移操作宏，使用对应的 AVX512 指令
#define npyv_shli_s16  npyv_shl_u16
#define npyv_shli_u32 _mm512_slli_epi32
#define npyv_shli_s32 _mm512_slli_epi32
#define npyv_shli_u64 _mm512_slli_epi64
#define npyv_shli_s64 _mm512_slli_epi64

// right
#ifdef NPY_HAVE_AVX512BW
    // 如果支持 AVX512BW，则使用对应的 srl_epi16 和 sra_epi16 函数进行右移操作
    #define npyv_shr_u16(A, C) _mm512_srl_epi16(A, _mm_cvtsi32_si128(C))
    #define npyv_shr_s16(A, C) _mm512_sra_epi16(A, _mm_cvtsi32_si128(C))
#else
    // 否则使用自定义的宏 NPYV_IMPL_AVX512_SHIFT 定义的右移函数
    NPYV_IMPL_AVX512_SHIFT(shr_u16, srl_epi16)
    NPYV_IMPL_AVX512_SHIFT(shr_s16, sra_epi16)
#endif
// 定义其他整数类型的右移操作宏，使用对应的 AVX512 指令
#define npyv_shr_u32(A, C) _mm512_srl_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s32(A, C) _mm512_sra_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_u64(A, C) _mm512_srl_epi64(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s64(A, C) _mm512_sra_epi64(A, _mm_cvtsi32_si128(C))

// right by an immediate constant
#ifdef NPY_HAVE_AVX512BW
    // 如果支持 AVX512BW，则使用对应的 srli_epi16 和 srai_epi16 函数进行按常量右移操作
    #define npyv_shri_u16 _mm512_srli_epi16
    #define npyv_shri_s16 _mm512_srai_epi16
#else
    // 否则使用之前定义的 npyv_shr_u16 和 npyv_shr_s16 宏进行右移操作
    #define npyv_shri_u16 npyv_shr_u16
    #define npyv_shri_s16 npyv_shr_s16
#endif
// 定义其他整数类型的按常量右移操作宏，使用对应的 AVX512 指令
#define npyv_shri_u32 _mm512_srli_epi32
#define npyv_shri_s32 _mm512_srai_epi32
#define npyv_shri_u64 _mm512_srli_epi64
#define npyv_shri_s64 _mm512_srai_epi64

/***************************
 * Logical
 ***************************/

// AND
// 定义各种整数类型的按位与操作宏，使用对应的 AVX512 指令
#define npyv_and_u8  _mm512_and_si512
#define npyv_and_s8  _mm512_and_si512
#define npyv_and_u16 _mm512_and_si512
#define npyv_and_s16 _mm512_and_si512
#define npyv_and_u32 _mm512_and_si512
#define npyv_and_s32 _mm512_and_si512
#define npyv_and_u64 _mm512_and_si512
#define npyv_and_s64 _mm512_and_si512
#ifdef NPY_HAVE_AVX512DQ
    # 定义宏，用于执行 AVX-512 指令集的单精度浮点数按位与操作
    #define npyv_and_f32 _mm512_and_ps
    # 定义宏，用于执行 AVX-512 指令集的双精度浮点数按位与操作
    #define npyv_and_f64 _mm512_and_pd
#else
    NPYV_IMPL_AVX512_FROM_SI512_PS_2ARG(npyv_and_f32, _mm512_and_si512)
    // 定义 AVX512 浮点数按位与操作宏，使用 _mm512_and_si512 实现
    NPYV_IMPL_AVX512_FROM_SI512_PD_2ARG(npyv_and_f64, _mm512_and_si512)
    // 定义 AVX512 双精度浮点数按位与操作宏，使用 _mm512_and_si512 实现
#endif

// OR
#define npyv_or_u8  _mm512_or_si512
// 定义 AVX512 无符号 8 位整数按位或操作宏，使用 _mm512_or_si512
#define npyv_or_s8  _mm512_or_si512
// 定义 AVX512 有符号 8 位整数按位或操作宏，使用 _mm512_or_si512
#define npyv_or_u16 _mm512_or_si512
// 定义 AVX512 无符号 16 位整数按位或操作宏，使用 _mm512_or_si512
#define npyv_or_s16 _mm512_or_si512
// 定义 AVX512 有符号 16 位整数按位或操作宏，使用 _mm512_or_si512
#define npyv_or_u32 _mm512_or_si512
// 定义 AVX512 无符号 32 位整数按位或操作宏，使用 _mm512_or_si512
#define npyv_or_s32 _mm512_or_si512
// 定义 AVX512 有符号 32 位整数按位或操作宏，使用 _mm512_or_si512
#define npyv_or_u64 _mm512_or_si512
// 定义 AVX512 无符号 64 位整数按位或操作宏，使用 _mm512_or_si512
#define npyv_or_s64 _mm512_or_si512
// 定义 AVX512 有符号 64 位整数按位或操作宏，使用 _mm512_or_si512
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_or_f32 _mm512_or_ps
    // 如果支持 AVX512DQ，定义 AVX512 单精度浮点数按位或操作宏，使用 _mm512_or_ps
    #define npyv_or_f64 _mm512_or_pd
    // 如果支持 AVX512DQ，定义 AVX512 双精度浮点数按位或操作宏，使用 _mm512_or_pd
#else
    NPYV_IMPL_AVX512_FROM_SI512_PS_2ARG(npyv_or_f32, _mm512_or_si512)
    // 否则，定义 AVX512 浮点数按位或操作宏，使用 _mm512_or_si512 实现
    NPYV_IMPL_AVX512_FROM_SI512_PD_2ARG(npyv_or_f64, _mm512_or_si512)
    // 否则，定义 AVX512 双精度浮点数按位或操作宏，使用 _mm512_or_si512 实现
#endif

// XOR
#define npyv_xor_u8  _mm512_xor_si512
// 定义 AVX512 无符号 8 位整数按位异或操作宏，使用 _mm512_xor_si512
#define npyv_xor_s8  _mm512_xor_si512
// 定义 AVX512 有符号 8 位整数按位异或操作宏，使用 _mm512_xor_si512
#define npyv_xor_u16 _mm512_xor_si512
// 定义 AVX512 无符号 16 位整数按位异或操作宏，使用 _mm512_xor_si512
#define npyv_xor_s16 _mm512_xor_si512
// 定义 AVX512 有符号 16 位整数按位异或操作宏，使用 _mm512_xor_si512
#define npyv_xor_u32 _mm512_xor_si512
// 定义 AVX512 无符号 32 位整数按位异或操作宏，使用 _mm512_xor_si512
#define npyv_xor_s32 _mm512_xor_si512
// 定义 AVX512 有符号 32 位整数按位异或操作宏，使用 _mm512_xor_si512
#define npyv_xor_u64 _mm512_xor_si512
// 定义 AVX512 无符号 64 位整数按位异或操作宏，使用 _mm512_xor_si512
#define npyv_xor_s64 _mm512_xor_si512
// 定义 AVX512 有符号 64 位整数按位异或操作宏，使用 _mm512_xor_si512
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_xor_f32 _mm512_xor_ps
    // 如果支持 AVX512DQ，定义 AVX512 单精度浮点数按位异或操作宏，使用 _mm512_xor_ps
    #define npyv_xor_f64 _mm512_xor_pd
    // 如果支持 AVX512DQ，定义 AVX512 双精度浮点数按位异或操作宏，使用 _mm512_xor_pd
#else
    NPYV_IMPL_AVX512_FROM_SI512_PS_2ARG(npyv_xor_f32, _mm512_xor_si512)
    // 否则，定义 AVX512 浮点数按位异或操作宏，使用 _mm512_xor_si512 实现
    NPYV_IMPL_AVX512_FROM_SI512_PD_2ARG(npyv_xor_f64, _mm512_xor_si512)
    // 否则，定义 AVX512 双精度浮点数按位异或操作宏，使用 _mm512_xor_si512 实现
#endif

// NOT
#define npyv_not_u8(A) _mm512_xor_si512(A, _mm512_set1_epi32(-1))
// 定义 AVX512 无符号 8 位整数按位取反操作宏，使用 _mm512_xor_si512(A, _mm512_set1_epi32(-1))
#define npyv_not_s8  npyv_not_u8
// 定义 AVX512 有符号 8 位整数按位取反操作宏，与无符号 8 位整数取反相同
#define npyv_not_u16 npyv_not_u8
// 定义 AVX512 无符号 16 位整数按位取反操作宏，与无符号 8 位整数取反相同
#define npyv_not_s16 npyv_not_u8
// 定义 AVX512 有符号 16 位整数按位取反操作宏，与无符号 8 位整数取反相同
#define npyv_not_u32 npyv_not_u8
// 定义 AVX512 无符号 32 位整数按位取反操作宏，与无符号 8 位整数取反相同
#define npyv_not_s32 npyv_not_u8
// 定义 AVX512 有符号 32 位整数按位取反操作宏，与无符号 8 位整数取反相同
#define npyv_not_u64 npyv_not_u8
// 定义 AVX512 无符号 64 位整数按位取反操作宏，与无符号 8 位整数取反相同
#define npyv_not_s64 npyv_not_u8
// 定义 AVX512 有符号 64 位整数按位取反操作宏，与无符号 8 位整数取反相同
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_not_f32(A) _mm512_xor_ps(A, _mm512_castsi512_ps(_mm512_set1_epi32(-1)))
    // 如果支持 AVX512DQ，定义 AVX512 单精度浮点数按位取反操作宏，使用 _mm512_xor_ps 实现
    #define npyv_not_f64(A) _mm512_xor_pd(A, _mm512_castsi512_pd(_mm512_set1_epi32(-1)))
    // 如果支持 AVX512DQ，定义 AVX512 双精度浮点数按位取反操作宏，使用 _mm512_xor_pd 实现
#else
    #define npyv_not_f32(A) _mm512_castsi512_ps(npyv_not_u32(_mm512_castps_si512(A)))
    // 否则，定义 AVX512 单精度浮点数按位取
    # 定义一个内联函数，对两个8位向量进行按位异或操作，并返回结果
    NPY_FINLINE npyv_b8  npyv_xor_b8(npyv_b8 a, npyv_b8 b)
    { return a ^ b; }
    
    # 定义一个内联函数，对两个16位向量进行按位异或操作，并返回结果
    NPY_FINLINE npyv_b16 npyv_xor_b16(npyv_b16 a, npyv_b16 b)
    { return a ^ b; }
    
    # 定义一个内联函数，对一个8位向量进行按位取反操作，并返回结果
    NPY_FINLINE npyv_b8  npyv_not_b8(npyv_b8 a)
    { return ~a; }
    
    # 定义一个内联函数，对一个16位向量进行按位取反操作，并返回结果
    NPY_FINLINE npyv_b16 npyv_not_b16(npyv_b16 a)
    { return ~a; }
    
    # 定义一个内联函数，对一个8位向量与另一个8位向量按位取反的结果进行按位与操作，并返回结果
    NPY_FINLINE npyv_b8  npyv_andc_b8(npyv_b8 a, npyv_b8 b)
    { return a & (~b); }
    
    # 定义一个内联函数，对一个8位向量与另一个8位向量按位取反的结果进行按位或操作，并返回结果
    NPY_FINLINE npyv_b8  npyv_orc_b8(npyv_b8 a, npyv_b8 b)
    { return a | (~b); }
    
    # 定义一个内联函数，对两个8位向量进行按位异或操作后再按位取反，并返回结果
    NPY_FINLINE npyv_b8  npyv_xnor_b8(npyv_b8 a, npyv_b8 b)
    { return ~(a ^ b); }
#else
    // 定义 AVX512 下的位与操作宏
    #define npyv_and_b8  _mm512_and_si512
    // 定义 AVX512 下的位与操作宏
    #define npyv_and_b16 _mm512_and_si512
    // 定义 AVX512 下的位或操作宏
    #define npyv_or_b8   _mm512_or_si512
    // 定义 AVX512 下的位或操作宏
    #define npyv_or_b16  _mm512_or_si512
    // 定义 AVX512 下的位异或操作宏
    #define npyv_xor_b8  _mm512_xor_si512
    // 定义 AVX512 下的位异或操作宏
    #define npyv_xor_b16 _mm512_xor_si512
    // 定义 AVX512 下的位非操作宏
    #define npyv_not_b8  npyv_not_u8
    // 定义 AVX512 下的位非操作宏
    #define npyv_not_b16 npyv_not_u8
    // 定义 AVX512 下的位与非操作宏
    #define npyv_andc_b8(A, B) _mm512_andnot_si512(B, A)
    // 定义 AVX512 下的位或非操作宏
    #define npyv_orc_b8(A, B) npyv_or_b8(npyv_not_b8(B), A)
    // 定义 AVX512 下的位异或非操作宏
    #define npyv_xnor_b8(A, B) npyv_not_b8(npyv_xor_b8(A, B))
#endif

#define npyv_and_b32 _mm512_kand
#define npyv_or_b32  _mm512_kor
#define npyv_xor_b32 _mm512_kxor
#define npyv_not_b32 _mm512_knot

#ifdef NPY_HAVE_AVX512DQ_MASK
    // 定义 AVX512 DQ 指令集下的位与操作宏
    #define npyv_and_b64 _kand_mask8
    // 定义 AVX512 DQ 指令集下的位或操作宏
    #define npyv_or_b64  _kor_mask8
    // 定义 AVX512 DQ 指令集下的位异或操作宏
    #define npyv_xor_b64 _kxor_mask8
    // 定义 AVX512 DQ 指令集下的位非操作宏
    #define npyv_not_b64 _knot_mask8
#else
    // 如果不支持 AVX512 DQ 指令集，定义 64 位位与操作函数
    NPY_FINLINE npyv_b64 npyv_and_b64(npyv_b64 a, npyv_b64 b)
    { return (npyv_b64)_mm512_kand((npyv_b32)a, (npyv_b32)b); }
    // 如果不支持 AVX512 DQ 指令集，定义 64 位位或操作函数
    NPY_FINLINE npyv_b64 npyv_or_b64(npyv_b64 a, npyv_b64 b)
    { return (npyv_b64)_mm512_kor((npyv_b32)a, (npyv_b32)b); }
    // 如果不支持 AVX512 DQ 指令集，定义 64 位位异或操作函数
    NPY_FINLINE npyv_b64 npyv_xor_b64(npyv_b64 a, npyv_b64 b)
    { return (npyv_b64)_mm512_kxor((npyv_b32)a, (npyv_b32)b); }
    // 如果不支持 AVX512 DQ 指令集，定义 64 位位非操作函数
    NPY_FINLINE npyv_b64 npyv_not_b64(npyv_b64 a)
    { return (npyv_b64)_mm512_knot((npyv_b32)a); }
#endif

/***************************
 * Comparison
 ***************************/

// int Equal
#ifdef NPY_HAVE_AVX512BW
    // 定义 AVX512 BW 指令集下的无符号 8 位相等比较操作宏
    #define npyv_cmpeq_u8  _mm512_cmpeq_epu8_mask
    // 定义 AVX512 BW 指令集下的有符号 8 位相等比较操作宏
    #define npyv_cmpeq_s8  _mm512_cmpeq_epi8_mask
    // 定义 AVX512 BW 指令集下的无符号 16 位相等比较操作宏
    #define npyv_cmpeq_u16 _mm512_cmpeq_epu16_mask
    // 定义 AVX512 BW 指令集下的有符号 16 位相等比较操作宏
    #define npyv_cmpeq_s16 _mm512_cmpeq_epi16_mask
#else
    // 如果不支持 AVX512 BW 指令集，从 AVX2 转换定义无符号 8 位相等比较操作函数
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_cmpeq_u8,  _mm256_cmpeq_epi8)
    // 如果不支持 AVX512 BW 指令集，从 AVX2 转换定义无符号 16 位相等比较操作函数
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_cmpeq_u16, _mm256_cmpeq_epi16)
    // 如果不支持 AVX512 BW 指令集，定义有符号 8 位相等比较操作宏
    #define npyv_cmpeq_s8  npyv_cmpeq_u8
    // 如果不支持 AVX512 BW 指令集，定义有符号 16 位相等比较操作宏
    #define npyv_cmpeq_s16 npyv_cmpeq_u16
#endif
// 定义 AVX512 下的无符号 32 位相等比较操作宏
#define npyv_cmpeq_u32 _mm512_cmpeq_epu32_mask
// 定义 AVX512 下的有符号 32 位相等比较操作宏
#define npyv_cmpeq_s32 _mm512_cmpeq_epi32_mask
// 定义 AVX512 下的无符号 64 位相等比较操作宏
#define npyv_cmpeq_u64 _mm512_cmpeq_epu64_mask
// 定义 AVX512 下的有符号 64 位相等比较操作宏
#define npyv_cmpeq_s64 _mm512_cmpeq_epi64_mask

// int not equal
#ifdef NPY_HAVE_AVX512BW
    // 定义 AVX512 BW 指令集下的无符号 8 位不相等比较操作宏
    #define npyv_cmpneq_u8  _mm512_cmpneq_epu8_mask
    // 定义 AVX512 BW 指令集下的有符号 8 位不相等比较操作宏
    #define npyv_cmpneq_s8  _mm512_cmpneq_epi8_mask
    // 定义 AVX512 BW 指令集下的无符号 16 位不相等比较操作宏
    #define npyv_cmpneq_u16 _mm512_cmpneq_epu16_mask
    // 定义 AVX512 BW 指令集下的有符号 16 位不相等比较操作宏
    #define npyv_cmpneq_s16 _mm512_cmpneq_epi16_mask
#else
    // 如果不支持 AVX512 BW 指令集，定义无符号 8 位不相等比较操作宏
    #define npyv_cmpneq_u8(A, B) npyv_not_u8(npyv_cmpeq_u8(A, B))
    // 如果不支持 AVX512 BW 指令集，定义无符号 16 位不相等比较操作宏
    #define npyv_cmpneq_u16(A, B) npyv_not_u16(npyv_cmpeq_u16(A, B))
    // 如果不支持 AVX512 BW 指令集，定义有符号 8 位不相等比较操作宏
    #define npyv_cmpne
    # 使用 AVX2 指令集中的 _mm256_cmpgt_epi8 函数实现 npyv_cmpgt_s8 函数
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_cmpgt_s8,  _mm256_cmpgt_epi8)
    
    # 使用 AVX2 指令集中的 _mm256_cmpgt_epi16 函数实现 npyv_cmpgt_s16 函数
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_cmpgt_s16, _mm256_cmpgt_epi16)
    
    # 定义一个内联函数 npyv_cmpgt_u8，用于比较两个 __m512i 类型的无符号整数向量 a 和 b
    NPY_FINLINE __m512i npyv_cmpgt_u8(__m512i a, __m512i b)
    {
        # 创建一个常量向量 sbit，其每个元素为 0x80808080，用于处理无符号整数的比较
        const __m512i sbit = _mm512_set1_epi32(0x80808080);
        # 返回对 a 和 b 分别按位异或后再传给 npyv_cmpgt_s8 处理的结果
        return npyv_cmpgt_s8(_mm512_xor_si512(a, sbit), _mm512_xor_si512(b, sbit));
    }
    
    # 定义一个内联函数 npyv_cmpgt_u16，用于比较两个 __m512i 类型的无符号整数向量 a 和 b
    NPY_FINLINE __m512i npyv_cmpgt_u16(__m512i a, __m512i b)
    {
        # 创建一个常量向量 sbit，其每个元素为 0x80008000，用于处理无符号整数的比较
        const __m512i sbit = _mm512_set1_epi32(0x80008000);
        # 返回对 a 和 b 分别按位异或后再传给 npyv_cmpgt_s16 处理的结果
        return npyv_cmpgt_s16(_mm512_xor_si512(a, sbit), _mm512_xor_si512(b, sbit));
    }
#endif
#define npyv_cmpgt_u32 _mm512_cmpgt_epu32_mask
#define npyv_cmpgt_s32 _mm512_cmpgt_epi32_mask
#define npyv_cmpgt_u64 _mm512_cmpgt_epu64_mask
#define npyv_cmpgt_s64 _mm512_cmpgt_epi64_mask

// 定义无符号和有符号整数的 AVX-512 指令，用于比较大于的掩码生成

#ifdef NPY_HAVE_AVX512BW
    #define npyv_cmpge_u8  _mm512_cmpge_epu8_mask
    #define npyv_cmpge_s8  _mm512_cmpge_epi8_mask
    #define npyv_cmpge_u16 _mm512_cmpge_epu16_mask
    #define npyv_cmpge_s16 _mm512_cmpge_epi16_mask
#else
    #define npyv_cmpge_u8(A, B)  npyv_not_u8(npyv_cmpgt_u8(B, A))
    #define npyv_cmpge_s8(A, B)  npyv_not_s8(npyv_cmpgt_s8(B, A))
    #define npyv_cmpge_u16(A, B) npyv_not_u16(npyv_cmpgt_u16(B, A))
    #define npyv_cmpge_s16(A, B) npyv_not_s16(npyv_cmpgt_s16(B, A))
#endif

// 如果支持 AVX-512BW 指令集，则定义无符号和有符号整数的大于等于掩码生成指令，否则用大于生成掩码并取反生成大于等于掩码

#define npyv_cmpge_u32 _mm512_cmpge_epu32_mask
#define npyv_cmpge_s32 _mm512_cmpge_epi32_mask
#define npyv_cmpge_u64 _mm512_cmpge_epu64_mask
#define npyv_cmpge_s64 _mm512_cmpge_epi64_mask

// 定义无符号和有符号整数的 AVX-512 指令，用于比较大于等于的掩码生成

#define npyv_cmplt_u8(A, B)  npyv_cmpgt_u8(B, A)
#define npyv_cmplt_s8(A, B)  npyv_cmpgt_s8(B, A)
#define npyv_cmplt_u16(A, B) npyv_cmpgt_u16(B, A)
#define npyv_cmplt_s16(A, B) npyv_cmpgt_s16(B, A)
#define npyv_cmplt_u32(A, B) npyv_cmpgt_u32(B, A)
#define npyv_cmplt_s32(A, B) npyv_cmpgt_s32(B, A)
#define npyv_cmplt_u64(A, B) npyv_cmpgt_u64(B, A)
#define npyv_cmplt_s64(A, B) npyv_cmpgt_s64(B, A)

// 定义无符号和有符号整数的 AVX-512 指令，用于比较小于的掩码生成

#define npyv_cmple_u8(A, B)  npyv_cmpge_u8(B, A)
#define npyv_cmple_s8(A, B)  npyv_cmpge_s8(B, A)
#define npyv_cmple_u16(A, B) npyv_cmpge_u16(B, A)
#define npyv_cmple_s16(A, B) npyv_cmpge_s16(B, A)
#define npyv_cmple_u32(A, B) npyv_cmpge_u32(B, A)
#define npyv_cmple_s32(A, B) npyv_cmpge_s32(B, A)
#define npyv_cmple_u64(A, B) npyv_cmpge_u64(B, A)
#define npyv_cmple_s64(A, B) npyv_cmpge_s64(B, A)

// 定义无符号和有符号整数的 AVX-512 指令，用于比较小于等于的掩码生成

// precision comparison
#define npyv_cmpeq_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_EQ_OQ)
#define npyv_cmpeq_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_EQ_OQ)
#define npyv_cmpneq_f32(A, B) _mm512_cmp_ps_mask(A, B, _CMP_NEQ_UQ)
#define npyv_cmpneq_f64(A, B) _mm512_cmp_pd_mask(A, B, _CMP_NEQ_UQ)
#define npyv_cmplt_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_LT_OQ)
#define npyv_cmplt_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_LT_OQ)
#define npyv_cmple_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_LE_OQ)
#define npyv_cmple_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_LE_OQ)
#define npyv_cmpgt_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_GT_OQ)
#define npyv_cmpgt_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_GT_OQ)
#define npyv_cmpge_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_GE_OQ)
#define npyv_cmpge_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_GE_OQ)

// 定义单精度和双精度浮点数的 AVX-512 指令，用于比较相等、不相等、小于、小于等于、大于、大于等于的掩码生成

// check special cases
NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
{ return _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q); }
NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
{ return _mm512_cmp_pd_mask(a, a, _CMP_ORD_Q); }

// 检查特殊情况，返回单精度和双精度浮点数向量中非 NaN 元素的掩码

// Test cross all vector lanes
// any: returns true if any of the elements is not equal to zero
// all: returns true if all elements are not equal to zero
# 定义 AVX512 SIMD 操作的任意和所有函数模板

#define NPYV_IMPL_AVX512_ANYALL(SFX, MASK)        \
    # 定义检查是否存在非零元素的函数，返回值为布尔类型
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a) \
    { return npyv_tobits_##SFX(a) != 0; }         \
    # 定义检查是否所有元素都等于给定掩码值的函数，返回值为布尔类型
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a) \
    { return npyv_tobits_##SFX(a) == MASK; }

# 使用模板定义具体的 AVX512 SIMD 操作函数

NPYV_IMPL_AVX512_ANYALL(b8,  0xffffffffffffffffull)
NPYV_IMPL_AVX512_ANYALL(b16, 0xfffffffful)
NPYV_IMPL_AVX512_ANYALL(b32, 0xffff)
NPYV_IMPL_AVX512_ANYALL(b64, 0xff)

# 取消定义 AVX512 SIMD 操作的任意和所有函数模板

#undef NPYV_IMPL_AVX512_ANYALL

# 重新定义 AVX512 SIMD 操作的任意和所有函数模板，支持不同的数据类型 SFX 和 BSFX，以及不同的掩码 MASK

#define NPYV_IMPL_AVX512_ANYALL(SFX, BSFX, MASK)   \
    # 定义检查是否存在非零元素的函数，返回值为布尔类型
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)  \
    {                                              \
        return npyv_tobits_##BSFX(                 \
            npyv_cmpeq_##SFX(a, npyv_zero_##SFX()) \
        ) != MASK;                                 \
    }                                              \
    # 定义检查是否所有元素都等于给定掩码值的函数，返回值为布尔类型
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)  \
    {                                              \
        return npyv_tobits_##BSFX(                 \
            npyv_cmpeq_##SFX(a, npyv_zero_##SFX()) \
        ) == 0;                                    \
    }

# 使用模板定义具体的 AVX512 SIMD 操作函数

NPYV_IMPL_AVX512_ANYALL(u8,  b8,  0xffffffffffffffffull)
NPYV_IMPL_AVX512_ANYALL(s8,  b8,  0xffffffffffffffffull)
NPYV_IMPL_AVX512_ANYALL(u16, b16, 0xfffffffful)
NPYV_IMPL_AVX512_ANYALL(s16, b16, 0xfffffffful)
NPYV_IMPL_AVX512_ANYALL(u32, b32, 0xffff)
NPYV_IMPL_AVX512_ANYALL(s32, b32, 0xffff)
NPYV_IMPL_AVX512_ANYALL(u64, b64, 0xff)
NPYV_IMPL_AVX512_ANYALL(s64, b64, 0xff)
NPYV_IMPL_AVX512_ANYALL(f32, b32, 0xffff)
NPYV_IMPL_AVX512_ANYALL(f64, b64, 0xff)

# 取消定义 AVX512 SIMD 操作的任意和所有函数模板

#undef NPYV_IMPL_AVX512_ANYALL

#endif // _NPY_SIMD_AVX512_OPERATORS_H
```