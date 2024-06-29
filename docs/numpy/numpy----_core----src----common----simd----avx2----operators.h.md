# `.\numpy\numpy\_core\src\common\simd\avx2\operators.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_OPERATORS_H
#define _NPY_SIMD_AVX2_OPERATORS_H

/***************************
 * Shifting
 ***************************/

// 定义 AVX2 下的无符号 16 位整数左移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shl_u16(A, C) _mm256_sll_epi16(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的有符号 16 位整数左移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shl_s16(A, C) _mm256_sll_epi16(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的无符号 32 位整数左移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shl_u32(A, C) _mm256_sll_epi32(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的有符号 32 位整数左移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shl_s32(A, C) _mm256_sll_epi32(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的无符号 64 位整数左移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shl_u64(A, C) _mm256_sll_epi64(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的有符号 64 位整数左移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shl_s64(A, C) _mm256_sll_epi64(A, _mm_cvtsi32_si128(C))

// 左移动作，移位常数为立即数
#define npyv_shli_u16 _mm256_slli_epi16
#define npyv_shli_s16 _mm256_slli_epi16
#define npyv_shli_u32 _mm256_slli_epi32
#define npyv_shli_s32 _mm256_slli_epi32
#define npyv_shli_u64 _mm256_slli_epi64
#define npyv_shli_s64 _mm256_slli_epi64

// 定义 AVX2 下的无符号 16 位整数右移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shr_u16(A, C) _mm256_srl_epi16(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的有符号 16 位整数右移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shr_s16(A, C) _mm256_sra_epi16(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的无符号 32 位整数右移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shr_u32(A, C) _mm256_srl_epi32(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的有符号 32 位整数右移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shr_s32(A, C) _mm256_sra_epi32(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的无符号 64 位整数右移操作，参数 A 为输入向量，C 为移位常数
#define npyv_shr_u64(A, C) _mm256_srl_epi64(A, _mm_cvtsi32_si128(C))

// 定义 AVX2 下的有符号 64 位整数右移操作，参数 A 为输入向量，C 为移位常数
NPY_FINLINE __m256i npyv_shr_s64(__m256i a, int c)
{
    // 定义常量 sbit 为 0x8000000000000000 的 256 位整数向量
    const __m256i sbit = _mm256_set1_epi64x(0x8000000000000000);
    // 将整数常数 c 转换为 128 位整数向量
    const __m128i c64  = _mm_cvtsi32_si128(c);
    // 计算右移结果，包括符号位的扩展
    __m256i r = _mm256_srl_epi64(_mm256_add_epi64(a, sbit), c64);
    // 还原符号位
    return _mm256_sub_epi64(r, _mm256_srl_epi64(sbit, c64));
}

// 右移动作，移位常数为立即数
#define npyv_shri_u16 _mm256_srli_epi16
#define npyv_shri_s16 _mm256_srai_epi16
#define npyv_shri_u32 _mm256_srli_epi32
#define npyv_shri_s32 _mm256_srai_epi32
#define npyv_shri_u64 _mm256_srli_epi64
#define npyv_shri_s64  npyv_shr_s64

/***************************
 * Logical
 ***************************/

// 逻辑 AND 操作，参数类型为无符号 8 位整数
#define npyv_and_u8  _mm256_and_si256

// 逻辑 AND 操作，参数类型为有符号 8 位整数
#define npyv_and_s8  _mm256_and_si256

// 逻辑 AND 操作，参数类型为无符号 16 位整数
#define npyv_and_u16 _mm256_and_si256

// 逻辑 AND 操作，参数类型为有符号 16 位整数
#define npyv_and_s16 _mm256_and_si256

// 逻辑 AND 操作，参数类型为无符号 32 位整数
#define npyv_and_u32 _mm256_and_si256

// 逻辑 AND 操作，参数类型为有符号 32 位整数
#define npyv_and_s32 _mm256_and_si256

// 逻辑 AND 操作，参数类型为无符号 64 位整数
#define npyv_and_u64 _mm256_and_si256

// 逻辑 AND 操作，参数类型为有符号 64 位整数
#define npyv_and_s64 _mm256_and_si256

// 逻辑 AND 操作，参数类型为单精度浮点数
#define npyv_and_f32 _mm256_and_ps

// 逻辑 AND 操作，参数类型为双精度浮点数
#define npyv_and_f64 _mm256_and_pd

// 逻辑 AND 操作，参数类型为布尔值 8 位
#define npyv_and_b8  _mm256_and_si256

// 逻辑 AND 操作，参数类型为布尔值 16 位
#define npyv_and_b16 _mm256_and_si256

// 逻辑 AND 操作，参数类型为布尔值 32 位
#define npyv_and_b32 _mm256_and_si256

// 逻辑 AND 操作，参数类型为布尔值 64 位
#define npyv_and_b64 _mm256_and_si256

// 逻辑 OR 操作，参数类型为无符号 8 位整数
#define npyv_or_u8  _mm256_or_si256

// 逻辑 OR 操作，参数类型为有符号 8 位整数
#define npyv_or_s8  _mm256_or_si256

// 逻辑 OR 操作，参数类型为无符号 16 位整数
#define npyv_or_u16 _mm256_or_si256

// 逻辑 OR 操作，参数类型为有符号 16 位整数
#define npyv_or_s16 _mm256_or_si256

// 逻辑 OR 操作，参数类型为无符号 32 位整数
#define npyv_or_u32 _mm256_or_si256

// 逻辑 OR 操作，参数类型为有符号 32 位整数
#define npyv_or_s32 _mm256_or_si256

// 逻辑 OR 操作，参数类型为无符号 64 位整数
#define npyv_or_u64 _mm256_or_si256

// 逻辑 OR 操作
// 定义按位异或操作，用于不同数据类型的操作
#define npyv_xor_s8  _mm256_xor_si256
#define npyv_xor_u16 _mm256_xor_si256
#define npyv_xor_s16 _mm256_xor_si256
#define npyv_xor_u32 _mm256_xor_si256
#define npyv_xor_s32 _mm256_xor_si256
#define npyv_xor_u64 _mm256_xor_si256
#define npyv_xor_s64 _mm256_xor_si256
#define npyv_xor_f32 _mm256_xor_ps
#define npyv_xor_f64 _mm256_xor_pd
#define npyv_xor_b8  _mm256_xor_si256
#define npyv_xor_b16 _mm256_xor_si256
#define npyv_xor_b32 _mm256_xor_si256
#define npyv_xor_b64 _mm256_xor_si256

// NOT 操作的宏定义
#define npyv_not_u8(A) _mm256_xor_si256(A, _mm256_set1_epi32(-1))
#define npyv_not_s8  npyv_not_u8
#define npyv_not_u16 npyv_not_u8
#define npyv_not_s16 npyv_not_u8
#define npyv_not_u32 npyv_not_u8
#define npyv_not_s32 npyv_not_u8
#define npyv_not_u64 npyv_not_u8
#define npyv_not_s64 npyv_not_u8
#define npyv_not_f32(A) _mm256_xor_ps(A, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))
#define npyv_not_f64(A) _mm256_xor_pd(A, _mm256_castsi256_pd(_mm256_set1_epi32(-1)))
#define npyv_not_b8  npyv_not_u8
#define npyv_not_b16 npyv_not_u8
#define npyv_not_b32 npyv_not_u8
#define npyv_not_b64 npyv_not_u8

// ANDC, ORC 和 XNOR 操作的宏定义
#define npyv_andc_u8(A, B) _mm256_andnot_si256(B, A)
#define npyv_andc_b8(A, B) _mm256_andnot_si256(B, A)
#define npyv_orc_b8(A, B) npyv_or_b8(npyv_not_b8(B), A)
#define npyv_xnor_b8 _mm256_cmpeq_epi8

/***************************
 * Comparison
 ***************************/

// 整数相等比较操作的宏定义
#define npyv_cmpeq_u8  _mm256_cmpeq_epi8
#define npyv_cmpeq_s8  _mm256_cmpeq_epi8
#define npyv_cmpeq_u16 _mm256_cmpeq_epi16
#define npyv_cmpeq_s16 _mm256_cmpeq_epi16
#define npyv_cmpeq_u32 _mm256_cmpeq_epi32
#define npyv_cmpeq_s32 _mm256_cmpeq_epi32
#define npyv_cmpeq_u64 _mm256_cmpeq_epi64
#define npyv_cmpeq_s64 _mm256_cmpeq_epi64

// 整数不相等比较操作的宏定义
#define npyv_cmpneq_u8(A, B) npyv_not_u8(_mm256_cmpeq_epi8(A, B))
#define npyv_cmpneq_s8 npyv_cmpneq_u8
#define npyv_cmpneq_u16(A, B) npyv_not_u16(_mm256_cmpeq_epi16(A, B))
#define npyv_cmpneq_s16 npyv_cmpneq_u16
#define npyv_cmpneq_u32(A, B) npyv_not_u32(_mm256_cmpeq_epi32(A, B))
#define npyv_cmpneq_s32 npyv_cmpneq_u32
#define npyv_cmpneq_u64(A, B) npyv_not_u64(_mm256_cmpeq_epi64(A, B))
#define npyv_cmpneq_s64 npyv_cmpneq_u64

// 有符号大于比较操作的宏定义
#define npyv_cmpgt_s8  _mm256_cmpgt_epi8
#define npyv_cmpgt_s16 _mm256_cmpgt_epi16
#define npyv_cmpgt_s32 _mm256_cmpgt_epi32
#define npyv_cmpgt_s64 _mm256_cmpgt_epi64

// 有符号大于等于比较操作的宏定义
#define npyv_cmpge_s8(A, B)  npyv_not_s8(_mm256_cmpgt_epi8(B, A))
#define npyv_cmpge_s16(A, B) npyv_not_s16(_mm256_cmpgt_epi16(B, A))
#define npyv_cmpge_s32(A, B) npyv_not_s32(_mm256_cmpgt_epi32(B, A))
#define npyv_cmpge_s64(A, B) npyv_not_s64(_mm256_cmpgt_epi64(B, A))

// 以下是一个未完成的宏定义，用于无符号大于比较操作
#define NPYV_IMPL_AVX2_UNSIGNED_GT(LEN, SIGN)                    \
    NPY_FINLINE __m256i npyv_cmpgt_u##LEN(__m256i a, __m256i b)  \
    {                                                            \
        const __m256i sbit = _mm256_set1_epi32(SIGN);            \
        # 创建一个包含全部元素为常数SIGN的256位整数向量sbit
        return _mm256_cmpgt_epi##LEN(                            \
            _mm256_xor_si256(a, sbit), _mm256_xor_si256(b, sbit) \
        );                                                       \
        # 使用AVX2指令集中的_mm256_cmpgt_epi##LEN函数，比较经过位异或运算后的两个256位整数向量a和b，
        # 返回结果是一个256位整数向量，每个元素是比较的结果（1表示a对应元素大于b对应元素，0表示否）
    }
// 使用 AVX2 指令集实现无符号大于比较操作，对每个操作数为8位整数的向量进行比较
NPYV_IMPL_AVX2_UNSIGNED_GT(8,  0x80808080)
// 使用 AVX2 指令集实现无符号大于比较操作，对每个操作数为16位整数的向量进行比较
NPYV_IMPL_AVX2_UNSIGNED_GT(16, 0x80008000)
// 使用 AVX2 指令集实现无符号大于比较操作，对每个操作数为32位整数的向量进行比较
NPYV_IMPL_AVX2_UNSIGNED_GT(32, 0x80000000)

// 定义函数 npyv_cmpgt_u64，用于比较两个64位整数向量 a 和 b 的大于关系
NPY_FINLINE __m256i npyv_cmpgt_u64(__m256i a, __m256i b)
{
    // 定义掩码向量 sbit，用于对比特反转，用于处理符号位的比较
    const __m256i sbit = _mm256_set1_epi64x(0x8000000000000000);
    // 返回对 a 和 b 的符号反转后的比较结果
    return _mm256_cmpgt_epi64(_mm256_xor_si256(a, sbit), _mm256_xor_si256(b, sbit));
}

// 定义函数 npyv_cmpge_u8，用于比较两个8位无符号整数向量 a 和 b 的大于等于关系
NPY_FINLINE __m256i npyv_cmpge_u8(__m256i a, __m256i b)
{ return _mm256_cmpeq_epi8(a, _mm256_max_epu8(a, b)); }

// 定义函数 npyv_cmpge_u16，用于比较两个16位无符号整数向量 a 和 b 的大于等于关系
NPY_FINLINE __m256i npyv_cmpge_u16(__m256i a, __m256i b)
{ return _mm256_cmpeq_epi16(a, _mm256_max_epu16(a, b)); }

// 定义函数 npyv_cmpge_u32，用于比较两个32位无符号整数向量 a 和 b 的大于等于关系
NPY_FINLINE __m256i npyv_cmpge_u32(__m256i a, __m256i b)
{ return _mm256_cmpeq_epi32(a, _mm256_max_epu32(a, b)); }

// 定义宏 npyv_cmpge_u64，用于比较两个64位无符号整数向量 A 和 B 的大于等于关系
#define npyv_cmpge_u64(A, B) npyv_not_u64(npyv_cmpgt_u64(B, A))

// 定义一系列宏，用于实现不同类型的小于比较操作，分别对应不同的整数类型（有符号和无符号）

// 定义一系列宏，用于实现不同类型的小于等于比较操作，分别对应不同的整数类型（有符号和无符号）

// 定义一系列宏，用于实现浮点数的精确比较操作，包括等于、不等于、小于、小于等于、大于、大于等于

// 定义函数 npyv_notnan_f32，用于检查32位浮点数向量 a 中的元素是否都不是 NaN
NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
{ return _mm256_castps_si256(_mm256_cmp_ps(a, a, _CMP_ORD_Q)); }

// 定义函数 npyv_notnan_f64，用于检查64位浮点数向量 a 中的元素是否都不是 NaN
NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
{ return _mm256_castpd_si256(_mm256_cmp_pd(a, a, _CMP_ORD_Q)); }

// 测试跨所有向量通道的情况
// any: 如果任意一个元素不等于零，则返回 true
// 定义宏 NPYV_IMPL_AVX2_ANYALL(SFX)，用于生成 AVX2 指令集的任意/全部判断函数
#define NPYV_IMPL_AVX2_ANYALL(SFX)                     \
    // 内联函数，判断 AVX2 数据类型为 SFX 的向量是否存在非零元素
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)      \
    {                                                  \
        // 使用 AVX2 指令 movemask 判断向量是否存在不等于零的元素
        return _mm256_movemask_epi8(                   \
            npyv_cmpeq_##SFX(a, npyv_zero_##SFX())     \
        ) != -1;                                       \
    }                                                  \
    // 内联函数，判断 AVX2 数据类型为 SFX 的向量是否所有元素都等于零
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)      \
    {                                                  \
        // 使用 AVX2 指令 movemask 判断向量是否所有元素都等于零
        return _mm256_movemask_epi8(                   \
            npyv_cmpeq_##SFX(a, npyv_zero_##SFX())     \
        ) == 0;                                        \
    }

// 生成 AVX2 指令集的不同数据类型的任意/全部判断函数
NPYV_IMPL_AVX2_ANYALL(b8)
NPYV_IMPL_AVX2_ANYALL(b16)
NPYV_IMPL_AVX2_ANYALL(b32)
NPYV_IMPL_AVX2_ANYALL(b64)

// 取消定义宏 NPYV_IMPL_AVX2_ANYALL，避免宏定义冲突
#undef NPYV_IMPL_AVX2_ANYALL

// 重新定义宏 NPYV_IMPL_AVX2_ANYALL(SFX)，用于生成 AVX2 指令集的任意/全部判断函数
#define NPYV_IMPL_AVX2_ANYALL(SFX)                                         \
    // 内联函数，判断 AVX2 数据类型为 SFX 的向量是否存在非零元素            \
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)                          \
    {                                                                       \
        // 使用 AVX2 指令 movemask 判断向量是否存在不等于零的元素            \
        return _mm256_movemask_##XSFX(                                      \
            _mm256_cmp_##XSFX(a, npyv_zero_##SFX(), _CMP_EQ_OQ)             \
        ) != MASK;                                                          \
    }                                                                       \
    // 内联函数，判断 AVX2 数据类型为 SFX 的向量是否所有元素都等于零         \
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)                          \
    {                                                                       \
        // 使用 AVX2 指令 movemask 判断向量是否所有元素都等于零               \
        return _mm256_movemask_##XSFX(                                      \
            _mm256_cmp_##XSFX(a, npyv_zero_##SFX(), _CMP_EQ_OQ)             \
        ) == 0;                                                             \
    }

// 生成 AVX2 指令集的不同数据类型的任意/全部判断函数，使用不同的比较器 XSFX 和 MASK
NPYV_IMPL_AVX2_ANYALL(u8,  ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(s8,  ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(u16, ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(s16, ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(u32, ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(s32, ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(u64, ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(s64, ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(f32, pd, 0xf)
NPYV_IMPL_AVX2_ANYALL(f64, pd, 0xf)

// 取消定义宏 NPYV_IMPL_AVX2_ANYALL，避免宏定义冲突
#undef NPYV_IMPL_AVX2_ANYALL

// 结束条件，关闭头文件 _NPY_SIMD_AVX2_OPERATORS_H 的声明
#endif // _NPY_SIMD_AVX2_OPERATORS_H
```