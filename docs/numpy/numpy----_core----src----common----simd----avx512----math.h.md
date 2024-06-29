# `.\numpy\numpy\_core\src\common\simd\avx512\math.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_MATH_H
#define _NPY_SIMD_AVX512_MATH_H

/***************************
 * Elementary
 ***************************/

// Square root functions for AVX512, operating on vectors of single and double precision floating point numbers
#define npyv_sqrt_f32 _mm512_sqrt_ps
#define npyv_sqrt_f64 _mm512_sqrt_pd

// Reciprocal functions for AVX512, computing the reciprocal of each element in a vector
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm512_div_ps(_mm512_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm512_div_pd(_mm512_set1_pd(1.0), a); }

// Absolute value functions for AVX512, computing absolute values of vectors of single and double precision floating point numbers
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
#if 0 // def NPY_HAVE_AVX512DQ
    return _mm512_range_ps(a, a, 8);
#else
    return npyv_and_f32(
        a, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff))
    );
#endif
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
#if 0 // def NPY_HAVE_AVX512DQ
    return _mm512_range_pd(a, a, 8);
#else
    return npyv_and_f64(
        a, _mm512_castsi512_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
#endif
}

// Square functions for AVX512, computing element-wise squares of vectors of single and double precision floating point numbers
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm512_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm512_mul_pd(a, a); }

// Maximum functions for AVX512, computing element-wise maximums of vectors of single and double precision floating point numbers
#define npyv_max_f32 _mm512_max_ps
#define npyv_max_f64 _mm512_max_pd

// Maximum with propagation of NaNs for single precision floating point numbers in AVX512
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_max_ps(a, nn, a, b);
}
// Maximum with propagation of NaNs for double precision floating point numbers in AVX512
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_max_pd(a, nn, a, b);
}

// Maximum with NaN handling for single precision floating point numbers in AVX512
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
    return _mm512_mask_max_ps(a, nn, a, b);
}
// Maximum with NaN handling for double precision floating point numbers in AVX512
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(a, a, _CMP_ORD_Q);
    return _mm512_mask_max_pd(a, nn, a, b);
}

// Maximum functions for integer types in AVX512, using unsigned and signed 8, 16, 32, and 64-bit integers
#ifdef NPY_HAVE_AVX512BW
    #define npyv_max_u8 _mm512_max_epu8
    #define npyv_max_s8 _mm512_max_epi8
    #define npyv_max_u16 _mm512_max_epu16
    #define npyv_max_s16 _mm512_max_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_u8, _mm256_max_epu8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_s8, _mm256_max_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_u16, _mm256_max_epu16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_s16, _mm256_max_epi16)
#endif
#define npyv_max_u32 _mm512_max_epu32
#define npyv_max_s32 _mm512_max_epi32
#define npyv_max_u64 _mm512_max_epu64
#define npyv_max_s64 _mm512_max_epi64

// Minimum functions for AVX512, computing element-wise minimums of vectors of single precision floating point numbers
#define npyv_min_f32 _mm512_min_ps

#endif // _NPY_SIMD_AVX512_MATH_H
#define npyv_min_f64 _mm512_min_pd
// 定义宏，将 _mm512_min_pd 重命名为 npyv_min_f64，用于执行双精度浮点数的最小值操作

// 返回 a 和 b 中每个对应元素的最小值，支持 IEEE 浮点算术（IEC 60559）
// - 如果其中一个向量包含 NaN，则设置另一个向量对应元素的值
// - 只有当两个对应元素都为 NaN 时，才设置 NaN
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);
    // 使用掩码 nn，将 a 和 b 中不是 NaN 的元素对应位置的最小值组成新向量返回
    return _mm512_mask_min_ps(a, nn, a, b);
}

// 返回 a 和 b 中每个对应元素的最小值，支持 IEEE 双精度浮点数算术（IEC 60559）
// - 如果其中一个向量包含 NaN，则设置另一个向量对应元素的值
// - 只有当两个对应元素都为 NaN 时，才设置 NaN
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(b, b, _CMP_ORD_Q);
    // 使用掩码 nn，将 a 和 b 中不是 NaN 的元素对应位置的最小值组成新向量返回
    return _mm512_mask_min_pd(a, nn, a, b);
}

// 返回 a 和 b 中每个对应元素的最小值，支持 IEEE 浮点算术（IEC 60559），传播 NaN
// - 如果任何对应元素为 NaN，则将该位置设置为 NaN
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
    // 使用掩码 nn，将 a 和 b 中 a 不是 NaN 的元素对应位置的最小值组成新向量返回
    return _mm512_mask_min_ps(a, nn, a, b);
}

// 返回 a 和 b 中每个对应元素的最小值，支持 IEEE 双精度浮点数算术（IEC 60559），传播 NaN
// - 如果任何对应元素为 NaN，则将该位置设置为 NaN
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(a, a, _CMP_ORD_Q);
    // 使用掩码 nn，将 a 和 b 中 a 不是 NaN 的元素对应位置的最小值组成新向量返回
    return _mm512_mask_min_pd(a, nn, a, b);
}

// 返回 a 和 b 中每个对应元素的最小值，支持整数操作
#ifdef NPY_HAVE_AVX512BW
    #define npyv_min_u8 _mm512_min_epu8
    #define npyv_min_s8 _mm512_min_epi8
    #define npyv_min_u16 _mm512_min_epu16
    #define npyv_min_s16 _mm512_min_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_u8, _mm256_min_epu8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_s8, _mm256_min_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_u16, _mm256_min_epu16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_s16, _mm256_min_epi16)
#endif

#define npyv_min_u32 _mm512_min_epu32
#define npyv_min_s32 _mm512_min_epi32
#define npyv_min_u64 _mm512_min_epu64
#define npyv_min_s64 _mm512_min_epi64

#ifdef NPY_HAVE_AVX512F_REDUCE
    #define npyv_reduce_min_u32 _mm512_reduce_min_epu32
    #define npyv_reduce_min_s32 _mm512_reduce_min_epi32
    #define npyv_reduce_min_u64 _mm512_reduce_min_epu64
    #define npyv_reduce_min_s64 _mm512_reduce_min_epi64
    #define npyv_reduce_min_f32 _mm512_reduce_min_ps
    #define npyv_reduce_min_f64 _mm512_reduce_min_pd
    #define npyv_reduce_max_u32 _mm512_reduce_max_epu32
    #define npyv_reduce_max_s32 _mm512_reduce_max_epi32
    #define npyv_reduce_max_u64 _mm512_reduce_max_epu64
    #define npyv_reduce_max_s64 _mm512_reduce_max_epi64
    #define npyv_reduce_max_f32 _mm512_reduce_max_ps
    #define npyv_reduce_max_f64 _mm512_reduce_max_pd
#else
    // 32位和64位的减少最小和最大值的实现
    #define NPY_IMPL_AVX512_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                              \
        // 定义 AVX-512 下的最小值和最大值归约函数，参数分别为数据类型、指令类型、向量指令类型
        NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m512i a)                              \
        {                                                                                      \
            // 将512位整型寄存器拆分为两个256位整型寄存器
            __m256i v256 = _mm256_##VINTRIN##32(npyv512_lower_si256(a),                        \
                    npyv512_higher_si256(a));                                                  \
            // 从256位整型寄存器中抽取低128位整型寄存器
            __m128i v128 = _mm_##VINTRIN##32(_mm256_castsi256_si128(v256),                     \
                    _mm256_extracti128_si256(v256, 1));                                        \
            // 使用 Shuffle 指令对128位整型寄存器进行重排，得到64位整型寄存器
            __m128i v64 =  _mm_##VINTRIN##32(v128, _mm_shuffle_epi32(v128,                     \
                        (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));                              \
            // 再次使用 Shuffle 指令对64位整型寄存器进行重排，得到32位整型寄存器
            __m128i v32 = _mm_##VINTRIN##32(v64, _mm_shuffle_epi32(v64,                        \
                        (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                              \
            // 将最终结果转换为32位整数并返回
            return (STYPE##32)_mm_cvtsi128_si32(v32);                                          \
        }                                                                                      \
        // 定义 AVX-512 下的最小值和最大值归约函数，参数分别为数据类型、指令类型、向量指令类型
        NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m512i a)                              \
        {                                                                                      \
            // 使用 Shuffle 指令对512位整型寄存器进行重排，得到256位整型寄存器
            __m512i v256 = _mm512_##VINTRIN##64(a,                                             \
                    _mm512_shuffle_i64x2(a, a, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));       \
            // 再次使用 Shuffle 指令对256位整型寄存器进行重排，得到128位整型寄存器
            __m512i v128 = _mm512_##VINTRIN##64(v256,                                          \
                    _mm512_shuffle_i64x2(v256, v256, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1))); \
            // 使用 Shuffle 指令对128位整型寄存器进行重排，得到64位整型寄存器
            __m512i v64  = _mm512_##VINTRIN##64(v128,                                          \
                    _mm512_shuffle_epi32(v128, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));       \
            // 提取64位整型寄存器中的第一个元素，并将结果转换为64位整数返回
            return (STYPE##64)npyv_extract0_u64(v64);                                          \
        }
    
    // 定义 AVX-512 下的最小值和最大值归约函数的具体实现
    NPY_IMPL_AVX512_REDUCE_MINMAX(npy_uint, min_u, min_epu)
    NPY_IMPL_AVX512_REDUCE_MINMAX(npy_int,  min_s, min_epi)
    NPY_IMPL_AVX512_REDUCE_MINMAX(npy_uint, max_u, max_epu)
    NPY_IMPL_AVX512_REDUCE_MINMAX(npy_int,  max_s, max_epi)
    #undef NPY_IMPL_AVX512_REDUCE_MINMAX
    // 取消宏定义 NPY_IMPL_AVX512_REDUCE_MINMAX，并注释说明为 ps & pd 的最小值和最大值归约
    #define NPY_IMPL_AVX512_REDUCE_MINMAX(INTRIN)                                         \
        // 定义 AVX512 下的最小值和最大值归约函数，使用指定的 INTRIN 操作
        NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                          \
        {                                                                                 \
            // 使用 AVX2 指令集下的 _mm256_##INTRIN##_ps 函数，对 a 进行归约操作
            __m256 v256 = _mm256_##INTRIN##_ps(                                           \
                    npyv512_lower_ps256(a), npyv512_higher_ps256(a));                     \
            // 将结果从 AVX256 转换为 AVX128，继续使用 AVX 指令进行归约
            __m128 v128 = _mm_##INTRIN##_ps(                                              \
                    _mm256_castps256_ps128(v256), _mm256_extractf128_ps(v256, 1));        \
            // 在 AVX128 中进行进一步的归约操作
            __m128 v64 =  _mm_##INTRIN##_ps(v128,                                         \
                    _mm_shuffle_ps(v128, v128, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));  \
            // 继续使用 AVX 指令在更低精度下进行最后的归约操作
            __m128 v32 = _mm_##INTRIN##_ps(v64,                                           \
                    _mm_shuffle_ps(v64, v64, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));    \
            // 返回最终结果，将 AVX32 结果转换为 float 类型
            return _mm_cvtss_f32(v32);                                                    \
        }                                                                                 \
        // 定义 AVX512 下的双精度最小值和最大值归约函数，使用指定的 INTRIN 操作
        NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)                         \
        {                                                                                 \
            // 使用 AVX2 指令集下的 _mm256_##INTRIN##_pd 函数，对 a 进行归约操作
            __m256d v256 = _mm256_##INTRIN##_pd(                                          \
                    npyv512_lower_pd256(a), npyv512_higher_pd256(a));                     \
            // 将结果从 AVX256 转换为 AVX128，继续使用 AVX 指令进行归约
            __m128d v128 = _mm_##INTRIN##_pd(                                             \
                    _mm256_castpd256_pd128(v256), _mm256_extractf128_pd(v256, 1));        \
            // 在 AVX128 中进行进一步的归约操作
            __m128d v64 =  _mm_##INTRIN##_pd(v128,                                        \
                    _mm_shuffle_pd(v128, v128, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));  \
            // 返回最终结果，将 AVX64 结果转换为 double 类型
            return _mm_cvtsd_f64(v64);                                                    \
        }
    // 取消定义 NPY_IMPL_AVX512_REDUCE_MINMAX 宏
    #undef NPY_IMPL_AVX512_REDUCE_MINMAX
#ifndef

#if 指令，用于条件编译，检查是否已经定义了指定的宏，如果没有则编译后续代码。


#define NPY_IMPL_AVX512_REDUCE_MINMAX(INTRIN, INF, INF64)           \

定义宏 NPY_IMPL_AVX512_REDUCE_MINMAX，带有三个参数：INTRIN、INF、INF64。


NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)       \

定义内联函数 npyv_reduce_##INTRIN##p_f32，接受一个参数 npyv_f32 类型的 a，返回 float 类型的值。


{

函数体开始。


npyv_b32 notnan = npyv_notnan_f32(a);                       

声明并初始化变量 notnan，用于存储对 a 应用 npyv_notnan_f32 函数的结果。


if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                  

如果 notnan 中不存在任何 true 值（即没有非 NaN 的元素），执行以下代码块。


return _mm_cvtss_f32(_mm512_castps512_ps128(a));        

将 a 强制转换为 128 位宽度的单精度浮点数，并返回其值。


}

条件语句结束。


a = npyv_select_f32(notnan, a,                              
        npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));    

通过 npyv_select_f32 函数，根据 notnan 的条件选择将 a 中的元素保留或替换为 INF 的值。


return npyv_reduce_##INTRIN##_f32(a);                       

调用 npyv_reduce_##INTRIN##_f32 函数，返回处理后的 a 的值。


}

函数体结束。

依此类推，对于每一个函数，需要按照类似的方式注释解释每一行代码的作用，确保完整理解每个函数的实现和功能。
// 定义 AVX512 指令的宏，用于实现最小值和最大值的归约操作
#define NPY_IMPL_AVX512_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                                               \
    // 定义内联函数，对 __m512i 类型的数据进行最小值或最大值的归约操作，返回 STYPE##16 类型的结果
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m512i a)                                               \
    {                                                                                                       \
        // 将 __m512i 类型数据的低 256 位和高 256 位分别提取为 __m256i 类型
        __m256i v256 = _mm256_##VINTRIN##16(npyv512_lower_si256(a), npyv512_higher_si256(a));               \
        // 从 __m256i 中提取低 128 位和高 128 位，执行相同的指令操作
        __m128i v128 = _mm_##VINTRIN##16(_mm256_castsi256_si128(v256), _mm256_extracti128_si256(v256, 1));  \
        // 将低 128 位的数据向高位扩展
        __m128i v64 =  _mm_##VINTRIN##16(v128, _mm_shuffle_epi32(v128,                                      \
                   (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));                                                \
        // 继续向高位扩展为 32 位
        __m128i v32 = _mm_##VINTRIN##16(v64, _mm_shuffle_epi32(v64,                                         \
                   (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                                                \
        // 再次向高位扩展为 16 位
        __m128i v16 = _mm_##VINTRIN##16(v32, _mm_shufflelo_epi16(v32,                                       \
                   (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                                                \
        // 将 16 位转换为 32 位，并返回最终结果
        return (STYPE##16)_mm_cvtsi128_si32(v16);                                                           \
    }                                                                                                       \
    // 定义内联函数，对 __m512i 类型的数据进行最小值或最大值的归约操作，返回 STYPE##8 类型的结果
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m512i a)                                                 \
    {                                                                                                       \
        // 将 __m512i 类型数据的低 256 位和高 256 位分别提取为 __m256i 类型
        __m256i v256 = _mm256_##VINTRIN##8(npyv512_lower_si256(a), npyv512_higher_si256(a));                \
        // 从 __m256i 中提取低 128 位和高 128 位，执行相同的指令操作
        __m128i v128 = _mm_##VINTRIN##8(_mm256_castsi256_si128(v256), _mm256_extracti128_si256(v256, 1));   \
        // 将低 128 位的数据向高位扩展
        __m128i v64 =  _mm_##VINTRIN##8(v128, _mm_shuffle_epi32(v128,                                       \
                    (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));                                               \
        // 继续向高位扩展为 32 位
        __m128i v32 = _mm_##VINTRIN##8(v64, _mm_shuffle_epi32(v64,                                          \
                    (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                                               \
        // 再次向高位扩展为 16 位
        __m128i v16 = _mm_##VINTRIN##8(v32, _mm_shufflelo_epi16(v32,                                        \
                    (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                                               \
        // 将 16 位向高位扩展为 8 位，并返回最终结果
        __m128i v8 = _mm_##VINTRIN##8(v16, _mm_srli_epi16(v16, 8));                                         \
        return (STYPE##16)_mm_cvtsi128_si32(v8);                                                            \
    }
// 定义 AVX512 指令集下的最小最大值归约操作，针对无符号整数，使用最小值操作宏
NPY_IMPL_AVX512_REDUCE_MINMAX(npy_uint, min_u, min_epu)
// 定义 AVX512 指令集下的最小最大值归约操作，针对有符号整数，使用最小值操作宏
NPY_IMPL_AVX512_REDUCE_MINMAX(npy_int,  min_s, min_epi)
// 定义 AVX512 指令集下的最小最大值归约操作，针对无符号整数，使用最大值操作宏
NPY_IMPL_AVX512_REDUCE_MINMAX(npy_uint, max_u, max_epu)
// 定义 AVX512 指令集下的最小最大值归约操作，针对有符号整数，使用最大值操作宏
NPY_IMPL_AVX512_REDUCE_MINMAX(npy_int,  max_s, max_epi)
// 取消前面定义的 AVX512 最小最大值归约操作宏
#undef NPY_IMPL_AVX512_REDUCE_MINMAX

// 定义向最近偶数整数舍入的 AVX512 单精度浮点数运算宏
#define npyv_rint_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_NEAREST_INT)
// 定义向最近偶数整数舍入的 AVX512 双精度浮点数运算宏
#define npyv_rint_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_NEAREST_INT)

// 定义向正无穷大舍入的 AVX512 单精度浮点数运算宏
#define npyv_ceil_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_POS_INF)
// 定义向正无穷大舍入的 AVX512 双精度浮点数运算宏
#define npyv_ceil_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_POS_INF)

// 定义向零舍入的 AVX512 单精度浮点数运算宏
#define npyv_trunc_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_ZERO)
// 定义向零舍入的 AVX512 双精度浮点数运算宏
#define npyv_trunc_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_ZERO)

// 定义向负无穷大舍入的 AVX512 单精度浮点数运算宏
#define npyv_floor_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_NEG_INF)
// 定义向负无穷大舍入的 AVX512 双精度浮点数运算宏
#define npyv_floor_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_NEG_INF)

// 结束 _NPY_SIMD_AVX512_MATH_H 文件的条件编译
#endif // _NPY_SIMD_AVX512_MATH_H
```