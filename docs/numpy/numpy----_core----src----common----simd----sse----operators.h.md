# `.\numpy\numpy\_core\src\common\simd\sse\operators.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_OPERATORS_H
#define _NPY_SIMD_SSE_OPERATORS_H

/***************************
 * Shifting
 ***************************/

// left
// 定义宏，将 SSE 寄存器 A 中的每个元素左移 C 位
#define npyv_shl_u16(A, C) _mm_sll_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s16(A, C) _mm_sll_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shl_u32(A, C) _mm_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s32(A, C) _mm_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_u64(A, C) _mm_sll_epi64(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s64(A, C) _mm_sll_epi64(A, _mm_cvtsi32_si128(C))

// left by an immediate constant
// 定义宏，将 SSE 寄存器 A 中的每个元素左移常数 C 位
#define npyv_shli_u16 _mm_slli_epi16
#define npyv_shli_s16 _mm_slli_epi16
#define npyv_shli_u32 _mm_slli_epi32
#define npyv_shli_s32 _mm_slli_epi32
#define npyv_shli_u64 _mm_slli_epi64
#define npyv_shli_s64 _mm_slli_epi64

// right
// 定义宏，将 SSE 寄存器 A 中的每个元素右移 C 位（逻辑右移）
#define npyv_shr_u16(A, C) _mm_srl_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s16(A, C) _mm_sra_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shr_u32(A, C) _mm_srl_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s32(A, C) _mm_sra_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_u64(A, C) _mm_srl_epi64(A, _mm_cvtsi32_si128(C))

// 定义一个特化的右移宏，处理 64 位有符号整数的右移操作
NPY_FINLINE __m128i npyv_shr_s64(__m128i a, int c)
{
    const __m128i sbit = npyv_setall_s64(0x8000000000000000);
    const __m128i cv   = _mm_cvtsi32_si128(c);
    __m128i r = _mm_srl_epi64(_mm_add_epi64(a, sbit), cv);
    return _mm_sub_epi64(r, _mm_srl_epi64(sbit, cv));
}

// Right by an immediate constant
// 定义宏，将 SSE 寄存器 A 中的每个元素右移常数 C 位（逻辑右移）
#define npyv_shri_u16 _mm_srli_epi16
#define npyv_shri_s16 _mm_srai_epi16
#define npyv_shri_u32 _mm_srli_epi32
#define npyv_shri_s32 _mm_srai_epi32
#define npyv_shri_u64 _mm_srli_epi64
#define npyv_shri_s64  npyv_shr_s64

/***************************
 * Logical
 ***************************/

// AND
// 定义宏，将 SSE 寄存器 A 和 B 中的每个元素按位与
#define npyv_and_u8  _mm_and_si128
#define npyv_and_s8  _mm_and_si128
#define npyv_and_u16 _mm_and_si128
#define npyv_and_s16 _mm_and_si128
#define npyv_and_u32 _mm_and_si128
#define npyv_and_s32 _mm_and_si128
#define npyv_and_u64 _mm_and_si128
#define npyv_and_s64 _mm_and_si128
#define npyv_and_f32 _mm_and_ps
#define npyv_and_f64 _mm_and_pd
#define npyv_and_b8  _mm_and_si128
#define npyv_and_b16 _mm_and_si128
#define npyv_and_b32 _mm_and_si128
#define npyv_and_b64 _mm_and_si128

// OR
// 定义宏，将 SSE 寄存器 A 和 B 中的每个元素按位或
#define npyv_or_u8  _mm_or_si128
#define npyv_or_s8  _mm_or_si128
#define npyv_or_u16 _mm_or_si128
#define npyv_or_s16 _mm_or_si128
#define npyv_or_u32 _mm_or_si128
#define npyv_or_s32 _mm_or_si128
#define npyv_or_u64 _mm_or_si128
#define npyv_or_s64 _mm_or_si128
#define npyv_or_f32 _mm_or_ps
#define npyv_or_f64 _mm_or_pd
#define npyv_or_b8  _mm_or_si128
#define npyv_or_b16 _mm_or_si128
#define npyv_or_b32 _mm_or_si128
#define npyv_or_b64 _mm_or_si128

// XOR
// 定义宏，将 SSE 寄存器 A 和 B 中的每个元素按位异或
#define npyv_xor_u8  _mm_xor_si128
#define npyv_xor_s8  _mm_xor_si128
#define npyv_xor_u16 _mm_xor_si128
#define npyv_xor_s16 _mm_xor_si128
#define npyv_xor_u32 _mm_xor_si128
#define npyv_xor_s32 _mm_xor_si128



#define npyv_xor_u64 _mm_xor_si128
// 定义按位异或操作宏，对于各种数据类型使用对应的 SSE 指令进行操作
#define npyv_xor_u64 _mm_xor_si128
#define npyv_xor_s64 _mm_xor_si128
#define npyv_xor_f32 _mm_xor_ps
#define npyv_xor_f64 _mm_xor_pd
#define npyv_xor_b8  _mm_xor_si128
#define npyv_xor_b16 _mm_xor_si128
#define npyv_xor_b32 _mm_xor_si128
#define npyv_xor_b64 _mm_xor_si128

// 定义按位取反操作宏，根据数据类型使用对应的 SSE 指令进行操作
#define npyv_not_u8(A) _mm_xor_si128(A, _mm_set1_epi32(-1))
#define npyv_not_s8  npyv_not_u8
#define npyv_not_u16 npyv_not_u8
#define npyv_not_s16 npyv_not_u8
#define npyv_not_u32 npyv_not_u8
#define npyv_not_s32 npyv_not_u8
#define npyv_not_u64 npyv_not_u8
#define npyv_not_s64 npyv_not_u8
#define npyv_not_f32(A) _mm_xor_ps(A, _mm_castsi128_ps(_mm_set1_epi32(-1)))
#define npyv_not_f64(A) _mm_xor_pd(A, _mm_castsi128_pd(_mm_set1_epi32(-1)))
#define npyv_not_b8  npyv_not_u8
#define npyv_not_b16 npyv_not_u8
#define npyv_not_b32 npyv_not_u8
#define npyv_not_b64 npyv_not_u8

// 定义按位与非操作宏，根据数据类型使用对应的 SSE 指令进行操作
#define npyv_andc_u8(A, B) _mm_andnot_si128(B, A)
#define npyv_andc_b8(A, B) _mm_andnot_si128(B, A)
// 定义按位或非操作宏，使用按位非和按位或操作来实现
#define npyv_orc_b8(A, B) npyv_or_b8(npyv_not_b8(B), A)
// 定义按位异或非操作宏，使用比较相等操作和按位异或操作来实现
#define npyv_xnor_b8 _mm_cmpeq_epi8

/***************************
 * Comparison
 ***************************/

// 定义整数相等比较操作宏，根据数据类型使用对应的 SSE 指令进行操作
#define npyv_cmpeq_u8  _mm_cmpeq_epi8
#define npyv_cmpeq_s8  _mm_cmpeq_epi8
#define npyv_cmpeq_u16 _mm_cmpeq_epi16
#define npyv_cmpeq_s16 _mm_cmpeq_epi16
#define npyv_cmpeq_u32 _mm_cmpeq_epi32
#define npyv_cmpeq_s32 _mm_cmpeq_epi32
#define npyv_cmpeq_s64  npyv_cmpeq_u64

#ifdef NPY_HAVE_SSE41
    // 如果支持 SSE4.1，定义整数相等比较操作宏使用 SSE 指令
    #define npyv_cmpeq_u64 _mm_cmpeq_epi64
#else
    // 否则，定义整数相等比较操作函数，使用基本的 SSE 指令来实现
    NPY_FINLINE __m128i npyv_cmpeq_u64(__m128i a, __m128i b)
    {
        __m128i cmpeq = _mm_cmpeq_epi32(a, b);
        __m128i cmpeq_h = _mm_srli_epi64(cmpeq, 32);
        __m128i test = _mm_and_si128(cmpeq, cmpeq_h);
        return _mm_shuffle_epi32(test, _MM_SHUFFLE(2, 2, 0, 0));
    }
#endif

// 定义整数不等比较操作宏，根据数据类型和支持的指令集使用对应的比较和非操作宏
#ifdef NPY_HAVE_XOP
    #define npyv_cmpneq_u8  _mm_comneq_epi8
    #define npyv_cmpneq_u16 _mm_comneq_epi16
    #define npyv_cmpneq_u32 _mm_comneq_epi32
    #define npyv_cmpneq_u64 _mm_comneq_epi64
#else
    #define npyv_cmpneq_u8(A, B)  npyv_not_u8(npyv_cmpeq_u8(A, B))
    #define npyv_cmpneq_u16(A, B) npyv_not_u16(npyv_cmpeq_u16(A, B))
    #define npyv_cmpneq_u32(A, B) npyv_not_u32(npyv_cmpeq_u32(A, B))
    #define npyv_cmpneq_u64(A, B) npyv_not_u64(npyv_cmpeq_u64(A, B))
#endif
#define npyv_cmpneq_s8  npyv_cmpneq_u8
#define npyv_cmpneq_s16 npyv_cmpneq_u16
#define npyv_cmpneq_s32 npyv_cmpneq_u32
#define npyv_cmpneq_s64 npyv_cmpneq_u64

// 定义有符号大于比较操作宏，根据数据类型和支持的指令集使用对应的 SSE 指令进行操作
#define npyv_cmpgt_s8  _mm_cmpgt_epi8
#define npyv_cmpgt_s16 _mm_cmpgt_epi16
#define npyv_cmpgt_s32 _mm_cmpgt_epi32

#ifdef NPY_HAVE_SSE42
    // 如果支持 SSE4.2，定义有符号大于比较操作宏使用 SSE 指令
    #define npyv_cmpgt_s64 _mm_cmpgt_epi64
#else
    // 否则，定义有符号大于比较操作函数，使用基本的 SSE 指令来实现
    NPY_FINLINE __m128i npyv_cmpgt_s64(__m128i a, __m128i b)
    {
        // 计算向量 b 减去向量 a 的结果
        __m128i sub = _mm_sub_epi64(b, a);
        // 计算 a 和 b 各位不同的比特位
        __m128i nsame_sbit = _mm_xor_si128(a, b);
        // 如果 nsame_sbit 不为零，则返回 b，否则返回 sub
        __m128i test = _mm_xor_si128(sub, _mm_and_si128(_mm_xor_si128(sub, b), nsame_sbit));
        // 扩展比特位，将 test 右移31位后的高位扩展到低位，并使用特定的顺序重新排列
        __m128i extend_sbit = _mm_shuffle_epi32(_mm_srai_epi32(test, 31), _MM_SHUFFLE(3, 3, 1, 1));
        // 返回最终结果 extend_sbit
        return extend_sbit;
    }
#endif

#ifdef NPY_HAVE_XOP
    #define npyv_cmpge_s8  _mm_comge_epi8
    #define npyv_cmpge_s16 _mm_comge_epi16
    #define npyv_cmpge_s32 _mm_comge_epi32
    #define npyv_cmpge_s64 _mm_comge_epi64
#else
    // 定义有符号整数比较大于等于操作，使用逻辑反操作得到
    #define npyv_cmpge_s8(A, B)  npyv_not_s8(_mm_cmpgt_epi8(B, A))
    #define npyv_cmpge_s16(A, B) npyv_not_s16(_mm_cmpgt_epi16(B, A))
    #define npyv_cmpge_s32(A, B) npyv_not_s32(_mm_cmpgt_epi32(B, A))
    #define npyv_cmpge_s64(A, B) npyv_not_s64(npyv_cmpgt_s64(B, A))
#endif

#ifdef NPY_HAVE_XOP
    #define npyv_cmpgt_u8  _mm_comgt_epu8
    #define npyv_cmpgt_u16 _mm_comgt_epu16
    #define npyv_cmpgt_u32 _mm_comgt_epu32
    #define npyv_cmpgt_u64 _mm_comgt_epu64
#else
    // 定义无符号整数大于操作，通过异或运算实现
    #define NPYV_IMPL_SSE_UNSIGNED_GT(LEN, SIGN)                     \
        NPY_FINLINE __m128i npyv_cmpgt_u##LEN(__m128i a, __m128i b)  \
        {                                                            \
            const __m128i sbit = _mm_set1_epi32(SIGN);               \
            return _mm_cmpgt_epi##LEN(                               \
                _mm_xor_si128(a, sbit), _mm_xor_si128(b, sbit)       \
            );                                                       \
        }

    NPYV_IMPL_SSE_UNSIGNED_GT(8,  0x80808080)
    NPYV_IMPL_SSE_UNSIGNED_GT(16, 0x80008000)
    NPYV_IMPL_SSE_UNSIGNED_GT(32, 0x80000000)

    // 定义64位无符号整数大于操作，通过异或运算实现
    NPY_FINLINE __m128i npyv_cmpgt_u64(__m128i a, __m128i b)
    {
        const __m128i sbit = npyv_setall_s64(0x8000000000000000);
        return npyv_cmpgt_s64(_mm_xor_si128(a, sbit), _mm_xor_si128(b, sbit));
    }
#endif

#ifdef NPY_HAVE_XOP
    #define npyv_cmpge_u8  _mm_comge_epu8
    #define npyv_cmpge_u16 _mm_comge_epu16
    #define npyv_cmpge_u32 _mm_comge_epu32
    #define npyv_cmpge_u64 _mm_comge_epu64
#else
    // 定义无符号整数大于等于操作，通过比较和最大值计算得到
    NPY_FINLINE __m128i npyv_cmpge_u8(__m128i a, __m128i b)
    { return _mm_cmpeq_epi8(a, _mm_max_epu8(a, b)); }
    
    // 在 SSE4.1 支持下，定义更高精度的无符号整数大于等于操作
    #ifdef NPY_HAVE_SSE41
        NPY_FINLINE __m128i npyv_cmpge_u16(__m128i a, __m128i b)
        { return _mm_cmpeq_epi16(a, _mm_max_epu16(a, b)); }
        NPY_FINLINE __m128i npyv_cmpge_u32(__m128i a, __m128i b)
        { return _mm_cmpeq_epi32(a, _mm_max_epu32(a, b)); }
    #else
        // 对于没有 SSE4.1 支持的情况，通过比较和逻辑非得到结果
        #define npyv_cmpge_u16(A, B) _mm_cmpeq_epi16(_mm_subs_epu16(B, A), _mm_setzero_si128())
        #define npyv_cmpge_u32(A, B) npyv_not_u32(npyv_cmpgt_u32(B, A))
    #endif
    // 定义64位无符号整数大于等于操作，通过逻辑非得到
    #define npyv_cmpge_u64(A, B) npyv_not_u64(npyv_cmpgt_u64(B, A))
#endif

// 定义各种类型的小于操作，通过大于操作和参数位置调换得到
#define npyv_cmplt_u8(A, B)  npyv_cmpgt_u8(B, A)
#define npyv_cmplt_s8(A, B)  npyv_cmpgt_s8(B, A)
#define npyv_cmplt_u16(A, B) npyv_cmpgt_u16(B, A)
#define npyv_cmplt_s16(A, B) npyv_cmpgt_s16(B, A)
#define npyv_cmplt_u32(A, B) npyv_cmpgt_u32(B, A)
#define npyv_cmplt_s32(A, B) npyv_cmpgt_s32(B, A)
#define npyv_cmplt_u64(A, B) npyv_cmpgt_u64(B, A)
#define npyv_cmplt_s64(A, B) npyv_cmpgt_s64(B, A)

// 定义各种类型的小于等于操作，通过大于等于操作和参数位置调换得到
#define npyv_cmple_u8(A, B)  npyv_cmpge_u8(B, A)
// 小于等于比较，返回值为将 B 与 A 进行大于等于比较的结果
#define npyv_cmple_s8(A, B)  npyv_cmpge_s8(B, A)
// 无符号 16 位整数小于等于比较，返回值为将 B 与 A 进行大于等于比较的结果
#define npyv_cmple_u16(A, B) npyv_cmpge_u16(B, A)
// 有符号 16 位整数小于等于比较，返回值为将 B 与 A 进行大于等于比较的结果
#define npyv_cmple_s16(A, B) npyv_cmpge_s16(B, A)
// 无符号 32 位整数小于等于比较，返回值为将 B 与 A 进行大于等于比较的结果
#define npyv_cmple_u32(A, B) npyv_cmpge_u32(B, A)
// 有符号 32 位整数小于等于比较，返回值为将 B 与 A 进行大于等于比较的结果
#define npyv_cmple_s32(A, B) npyv_cmpge_s32(B, A)
// 无符号 64 位整数小于等于比较，返回值为将 B 与 A 进行大于等于比较的结果
#define npyv_cmple_u64(A, B) npyv_cmpge_u64(B, A)
// 有符号 64 位整数小于等于比较，返回值为将 B 与 A 进行大于等于比较的结果
#define npyv_cmple_s64(A, B) npyv_cmpge_s64(B, A)

// 单精度浮点数等于比较，返回值为将 a 与 b 进行等于比较的结果
#define npyv_cmpeq_f32(a, b)  _mm_castps_si128(_mm_cmpeq_ps(a, b))
// 双精度浮点数等于比较，返回值为将 a 与 b 进行等于比较的结果
#define npyv_cmpeq_f64(a, b)  _mm_castpd_si128(_mm_cmpeq_pd(a, b))
// 单精度浮点数非等于比较，返回值为将 a 与 b 进行非等于比较的结果
#define npyv_cmpneq_f32(a, b) _mm_castps_si128(_mm_cmpneq_ps(a, b))
// 双精度浮点数非等于比较，返回值为将 a 与 b 进行非等于比较的结果
#define npyv_cmpneq_f64(a, b) _mm_castpd_si128(_mm_cmpneq_pd(a, b))
// 单精度浮点数小于比较，返回值为将 a 与 b 进行小于比较的结果
#define npyv_cmplt_f32(a, b)  _mm_castps_si128(_mm_cmplt_ps(a, b))
// 双精度浮点数小于比较，返回值为将 a 与 b 进行小于比较的结果
#define npyv_cmplt_f64(a, b)  _mm_castpd_si128(_mm_cmplt_pd(a, b))
// 单精度浮点数小于等于比较，返回值为将 a 与 b 进行小于等于比较的结果
#define npyv_cmple_f32(a, b)  _mm_castps_si128(_mm_cmple_ps(a, b))
// 双精度浮点数小于等于比较，返回值为将 a 与 b 进行小于等于比较的结果
#define npyv_cmple_f64(a, b)  _mm_castpd_si128(_mm_cmple_pd(a, b))
// 单精度浮点数大于比较，返回值为将 a 与 b 进行大于比较的结果
#define npyv_cmpgt_f32(a, b)  _mm_castps_si128(_mm_cmpgt_ps(a, b))
// 双精度浮点数大于比较，返回值为将 a 与 b 进行大于比较的结果
#define npyv_cmpgt_f64(a, b)  _mm_castpd_si128(_mm_cmpgt_pd(a, b))
// 单精度浮点数大于等于比较，返回值为将 a 与 b 进行大于等于比较的结果
#define npyv_cmpge_f32(a, b)  _mm_castps_si128(_mm_cmpge_ps(a, b))
// 双精度浮点数大于等于比较，返回值为将 a 与 b 进行大于等于比较的结果
#define npyv_cmpge_f64(a, b)  _mm_castpd_si128(_mm_cmpge_pd(a, b))

// 检查特殊情况
// 检查单精度浮点数向量中是否有 NaN，返回值为将 a 与 a 进行非 NaN 比较的结果
NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
{ return _mm_castps_si128(_mm_cmpord_ps(a, a)); }
// 检查双精度浮点数向量中是否有 NaN，返回值为将 a 与 a 进行非 NaN 比较的结果
NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
{ return _mm_castpd_si128(_mm_cmpord_pd(a, a)); }

// 测试跨所有向量通道
// any: 如果任何元素不等于零，则返回 true
// all: 如果所有元素均不等于零，则返回 true
#define NPYV_IMPL_SSE_ANYALL(SFX, MSFX, TSFX, MASK)                 \
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)                   \
    { return _mm_movemask_##MSFX(                                   \
        _mm_cmpeq_##TSFX(a, npyv_zero_##SFX())                      \
    ) != MASK; }                                                    \
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)                   \
    { return _mm_movemask_##MSFX(                                   \
        _mm_cmpeq_##TSFX(a, npyv_zero_##SFX())                      \
    ) == 0; }
// 无符号 8 位整数的任意和全部比较实现
NPYV_IMPL_SSE_ANYALL(u8,  epi8, epi8, 0xffff)
// 有符号 8 位整数的任意和全部比较实现
NPYV_IMPL_SSE_ANYALL(s8,  epi8, epi8, 0xffff)
// 无符号 16 位整数的任意和全部比较实现
NPYV_IMPL_SSE_ANYALL(u16, epi8, epi16, 0xffff)
// 有符号 16 位整数的任意和全部比较实现
NPYV_IMPL_SSE_ANYALL(s16, epi8, epi16, 0xffff)
// 无符号 32 位整数的任意和全部比较实现
NPYV_IMPL_SSE_ANYALL(u32, epi8, epi32, 0xffff)
// 有符号 32 位整数的任意和全部比较实现
NPYV_IMPL_SSE_ANYALL(s32, epi8, epi32, 0xffff)
#ifdef NPY_HAVE_SSE41
    // 无符号 64 位整数的任意和全部比较实现（如果支持 SSE4.1）
    NPYV_IMPL_SSE_ANYALL(u64, epi8, epi64, 0xffff)
    NPYV_IMPL_SSE_ANYALL(s64, epi8, epi64, 0xffff)



# 使用 SSE 指令集实现向量操作，判断64位有符号整数中每个元素是否满足条件
NPYV_IMPL_SSE_ANYALL(s64, epi8, epi64, 0xffff)


这行代码调用了一个宏 `NPYV_IMPL_SSE_ANYALL`，利用 SSE 指令集对 64 位有符号整数向量进行操作，它的作用是判断每个元素是否满足特定条件（这里是检查每个元素的低位16位是否全部为1）。
#else
    // 定义函数 npyv_any_u64，检查是否存在非全零的无符号64位整数向量 a
    NPY_FINLINE bool npyv_any_u64(npyv_u64 a)
    {
        // 判断向量 a 中是否有元素等于全零向量 npyv_zero_u64()，返回结果的逻辑非为真
        return _mm_movemask_epi8(
            _mm_cmpeq_epi32(a, npyv_zero_u64())
        ) != 0xffff;
    }
    // 定义函数 npyv_all_u64，检查是否所有的无符号64位整数向量 a 中的元素都为全零
    NPY_FINLINE bool npyv_all_u64(npyv_u64 a)
    {
        // 将向量 a 中的每个元素与全零向量 npyv_zero_u64() 比较，得到结果向量 a
        a = _mm_cmpeq_epi32(a, npyv_zero_u64());
        // 将结果向量 a 与自身进行按位与，并使用特定的整数序列进行打乱
        a = _mm_and_si128(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1)));
        // 检查向量 a 是否所有元素都为全零，返回结果的逻辑非为真
        return _mm_movemask_epi8(a) == 0;
    }
    // 定义宏 npyv_any_s64，使其与 npyv_any_u64 具有相同的定义
    #define npyv_any_s64 npyv_any_u64
    // 定义宏 npyv_all_s64，使其与 npyv_all_u64 具有相同的定义
    #define npyv_all_s64 npyv_all_u64
#endif

// 宏 NPYV_IMPL_SSE_ANYALL 的具体实现，用于实现 SSE 下的任意/所有功能
NPYV_IMPL_SSE_ANYALL(f32, ps, ps, 0xf)
NPYV_IMPL_SSE_ANYALL(f64, pd, pd, 0x3)
// 取消定义宏 NPYV_IMPL_SSE_ANYALL
#undef NPYV_IMPL_SSE_ANYALL

#endif // _NPY_SIMD_SSE_OPERATORS_H
```