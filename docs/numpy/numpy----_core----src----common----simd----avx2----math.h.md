# `.\numpy\numpy\_core\src\common\simd\avx2\math.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_MATH_H
#define _NPY_SIMD_AVX2_MATH_H
/***************************
 * Elementary
 ***************************/
// 平方根函数定义为 AVX2 指令集的平方根计算函数
#define npyv_sqrt_f32 _mm256_sqrt_ps
#define npyv_sqrt_f64 _mm256_sqrt_pd

// 倒数函数，对单精度浮点数向量进行定义
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm256_div_ps(_mm256_set1_ps(1.0f), a); }
// 倒数函数，对双精度浮点数向量进行定义
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm256_div_pd(_mm256_set1_pd(1.0), a); }

// 绝对值函数，对单精度浮点数向量进行定义
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
    return _mm256_and_ps(
        a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))
    );
}
// 绝对值函数，对双精度浮点数向量进行定义
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
    return _mm256_and_pd(
        a, _mm256_castsi256_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
}

// 平方函数，对单精度浮点数向量进行定义
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm256_mul_ps(a, a); }
// 平方函数，对双精度浮点数向量进行定义
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm256_mul_pd(a, a); }

// 最大值函数，直接映射为 AVX2 指令集的最大值函数，不保证处理 NaN
#define npyv_max_f32 _mm256_max_ps
#define npyv_max_f64 _mm256_max_pd
// 最大值函数，支持 IEEE 浮点数算术（IEC 60559）
// - 如果其中一个向量包含 NaN，则将另一个向量对应元素设置为 NaN
// - 仅当对应的两个元素都是 NaN 时，才设置 NaN
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __m256 nn  = _mm256_cmp_ps(b, b, _CMP_ORD_Q);
    __m256 max = _mm256_max_ps(a, b);
    return _mm256_blendv_ps(a, max, nn);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __m256d nn  = _mm256_cmp_pd(b, b, _CMP_ORD_Q);
    __m256d max = _mm256_max_pd(a, b);
    return _mm256_blendv_pd(a, max, nn);
}
// 最大值函数，传播 NaN
// 如果任何对应元素是 NaN，则设置 NaN
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    __m256 nn  = _mm256_cmp_ps(a, a, _CMP_ORD_Q);
    __m256 max = _mm256_max_ps(a, b);
    return _mm256_blendv_ps(a, max, nn);
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    __m256d nn  = _mm256_cmp_pd(a, a, _CMP_ORD_Q);
    __m256d max = _mm256_max_pd(a, b);
    return _mm256_blendv_pd(a, max, nn);
}

// 最大值函数，整数操作
#define npyv_max_u8 _mm256_max_epu8
#define npyv_max_s8 _mm256_max_epi8
#define npyv_max_u16 _mm256_max_epu16
#define npyv_max_s16 _mm256_max_epi16
#define npyv_max_u32 _mm256_max_epu32
#define npyv_max_s32 _mm256_max_epi32
// 对于无符号 64 位整数，定义最大值函数
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    return _mm256_blendv_epi8(b, a, npyv_cmpgt_u64(a, b));
}
// 对于有符号 64 位整数，定义最大值函数
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi64(a, b));
}

// 最小值函数，直接映射为 AVX2 指令集的最小值函数，不保证处理 NaN
#define npyv_min_f32 _mm256_min_ps
#define npyv_min_f64 _mm256_min_pd
// 最小值函数，支持 IEEE 浮点数算术（IEC 60559）
// - 如果其中一个向量包含 NaN，则将另一个向量对应元素设置为 NaN
// 返回两个 __m256 向量中对应元素的最小值，如果元素对应位置其中一个为 NaN，则结果也是 NaN
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    // 检查向量 b 中的元素是否为有序（非 NaN）
    __m256 nn  = _mm256_cmp_ps(b, b, _CMP_ORD_Q);
    // 计算两个向量对应元素的最小值
    __m256 min = _mm256_min_ps(a, b);
    // 根据 nn 向量的结果进行选择，如果对应位置 b 的元素为 NaN，则选择 a 中的元素
    return _mm256_blendv_ps(a, min, nn);
}

// 返回两个 __m256d 向量中对应元素的最小值，如果元素对应位置其中一个为 NaN，则结果也是 NaN
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    // 检查向量 b 中的元素是否为有序（非 NaN）
    __m256d nn  = _mm256_cmp_pd(b, b, _CMP_ORD_Q);
    // 计算两个向量对应元素的最小值
    __m256d min = _mm256_min_pd(a, b);
    // 根据 nn 向量的结果进行选择，如果对应位置 b 的元素为 NaN，则选择 a 中的元素
    return _mm256_blendv_pd(a, min, nn);
}

// 返回两个 __m256 向量中对应元素的最小值，如果任一对应位置元素为 NaN，则结果也是 NaN
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    // 检查向量 a 中的元素是否为有序（非 NaN）
    __m256 nn  = _mm256_cmp_ps(a, a, _CMP_ORD_Q);
    // 计算两个向量对应元素的最小值
    __m256 min = _mm256_min_ps(a, b);
    // 根据 nn 向量的结果进行选择，如果对应位置 a 的元素为 NaN，则选择 b 中的元素
    return _mm256_blendv_ps(a, min, nn);
}

// 返回两个 __m256d 向量中对应元素的最小值，如果任一对应位置元素为 NaN，则结果也是 NaN
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    // 检查向量 a 中的元素是否为有序（非 NaN）
    __m256d nn  = _mm256_cmp_pd(a, a, _CMP_ORD_Q);
    // 计算两个向量对应元素的最小值
    __m256d min = _mm256_min_pd(a, b);
    // 根据 nn 向量的结果进行选择，如果对应位置 a 的元素为 NaN，则选择 b 中的元素
    return _mm256_blendv_pd(a, min, nn);
}

// 返回两个 __m256i 向量中对应元素的最小值，用于无符号 64 位整数
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    // 使用 npyv_cmplt_u64 函数比较 a 和 b 的元素，生成选择向量，选择较小的元素
    return _mm256_blendv_epi8(b, a, npyv_cmplt_u64(a, b));
}

// 返回两个 __m256i 向量中对应元素的最小值，用于有符号 64 位整数
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    // 使用 _mm256_cmpgt_epi64 函数比较 a 和 b 的元素，生成选择向量，选择较小的元素
    return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi64(a, b));
}

// 实现 AVX2 指令集下的最小值和最大值归约操作，用于 32 位和 64 位数据类型
#define NPY_IMPL_AVX2_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                                              \
    NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m256i a)                                            \
    {                                                                                                    \
        // 将 __m256i 向量 a 转换为 __m128i 向量 v128，并且将两个 128 位块进行指定方式合并
        __m128i v128 = _mm_##VINTRIN##32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));     \
        // 继续归约到 64 位块
        __m128i v64 =  _mm_##VINTRIN##32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(0, 0, 3, 2)));        \
        // 继续归约到 32 位块
        __m128i v32 = _mm_##VINTRIN##32(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));           \
        // 将结果转换为 32 位整数
        return (STYPE##32)_mm_cvtsi128_si32(v32);                                                        \
    }                                                                                                    \
    NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m256i a)                                            \
    {                                                                                                    \
        // 使用 npyv_##INTRIN##64 函数将 a 向量与自身进行指定方式合并
        __m256i v128 = npyv_##INTRIN##64(a, _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(0, 0, 0, 1)));   \
        // 进行 64 位块的归约操作
        __m256i v64  = npyv_##INTRIN##64(v128, _mm256_shuffle_epi32(v128, _MM_SHUFFLE(0, 0, 3, 2)));     \
        // 提取归约结果中的第一个 64 位无符号整数
        return (STYPE##64)npyv_extract0_u64(v64);                                                        \
    }
// 定义 AVX2 指令集下的最小值归约操作宏，支持无符号和有符号 32 位整数
NPY_IMPL_AVX2_REDUCE_MINMAX(npy_uint, min_u, min_epu)
// 定义 AVX2 指令集下的最小值归约操作宏，支持无符号和有符号 64 位整数
NPY_IMPL_AVX2_REDUCE_MINMAX(npy_int,  min_s, min_epi)
# 定义 AVX2 指令集下的最小值和最大值归约操作宏，针对无符号整数类型
NPY_IMPL_AVX2_REDUCE_MINMAX(npy_uint, max_u, max_epu)

# 定义 AVX2 指令集下的最小值和最大值归约操作宏，针对有符号整数类型
NPY_IMPL_AVX2_REDUCE_MINMAX(npy_int, max_s, max_epi)

# 取消之前定义的 AVX2 指令集下的最小值和最大值归约操作宏
#undef NPY_IMPL_AVX2_REDUCE_MINMAX

# 定义 AVX2 指令集下的最小值和最大值归约操作宏，用于单精度浮点数和双精度浮点数
#define NPY_IMPL_AVX2_REDUCE_MINMAX(INTRIN, INF, INF64)                                              \
    # 定义单精度浮点数的归约函数，将 AVX2 矢量中的四个 float 值降维到一个 float 值
    NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                                         \
    {                                                                                                \
        # 提取 AVX2 矢量的低128位和高128位，执行指定的最小值或最大值归约操作
        __m128 v128 = _mm_##INTRIN##_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));     \
        # 对结果继续归约，取得最终的 float 值
        __m128 v64 =  _mm_##INTRIN##_ps(v128, _mm_shuffle_ps(v128, v128, _MM_SHUFFLE(0, 0, 3, 2)));  \
        __m128 v32 = _mm_##INTRIN##_ps(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1)));      \
        return _mm_cvtss_f32(v32);                                                                   \
    }                                                                                                \
    # 定义双精度浮点数的归约函数，将 AVX2 矢量中的四个 double 值降维到一个 double 值
    NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)                                        \
    {                                                                                                \
        # 提取 AVX2 矢量的低128位和高128位，执行指定的最小值或最大值归约操作
        __m128d v128 = _mm_##INTRIN##_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));    \
        # 对结果继续归约，取得最终的 double 值
        __m128d v64 =  _mm_##INTRIN##_pd(v128, _mm_shuffle_pd(v128, v128, _MM_SHUFFLE(0, 0, 0, 1))); \
        return _mm_cvtsd_f64(v64);                                                                   \
    }                                                                                                \
    # 定义单精度浮点数的条件归约函数，处理 NaN 值的情况
    NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)                                        \
    {                                                                                                \
        # 判断是否存在非 NaN 的元素
        npyv_b32 notnan = npyv_notnan_f32(a);                                                        \
        if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                                                   \
            # 若不存在非 NaN 元素，则直接返回矢量的第一个 float 值
            return _mm_cvtss_f32(_mm256_castps256_ps128(a));                                         \
        }                                                                                            \
        # 选取非 NaN 元素，若不存在非 NaN 元素，则使用指定的 INF 值
        a = npyv_select_f32(notnan, a, npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));              \
        # 对选取的值执行最小值或最大值归约操作
        return npyv_reduce_##INTRIN##_f32(a);                                                        \
    }                                                                                                \
    # 定义双精度浮点数的条件归约函数，处理 NaN 值的情况
    NPY_FINLINE double npyv_reduce_##INTRIN##p_f64(npyv_f64 a)                                       \
    {                                                                                                \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                        \  // 检查双精度浮点向量a中的每个元素是否不是NaN
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {                                                   \  // 如果向量a中没有任何元素不是NaN，则执行以下操作
            return _mm_cvtsd_f64(_mm256_castpd256_pd128(a));                                         \  // 将AVX双精度向量a转换为128位SIMD寄存器后取其低64位转换为标量返回
        }                                                                                            \
        a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(npyv_setall_u64(INF64)));            \  // 根据notnan向量选择a中不是NaN的元素，将NaN元素替换为正无穷大(INF64)
        return npyv_reduce_##INTRIN##_f64(a);                                                        \  // 调用指定指令集的双精度浮点向量a的规约函数，并返回结果
    }                                                                                                \
    NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)                                        \  // 使用指定指令集INTRIN对浮点向量a进行规约的函数定义
    {                                                                                                \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                        \  // 检查单精度浮点向量a中的每个元素是否不是NaN
        if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {                                                   \  // 如果向量a中存在任何NaN元素，则执行以下操作
            const union { npy_uint32 i; float f;} pnan = {0x7fc00000UL};                             \  // 创建一个union将NaN表示为单精度浮点数的结构
            return pnan.f;                                                                           \  // 返回NaN作为单精度浮点数
        }                                                                                            \
        return npyv_reduce_##INTRIN##_f32(a);                                                        \  // 调用指定指令集的单精度浮点向量a的规约函数，并返回结果
    }                                                                                                \
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)                                       \  // 使用指定指令集INTRIN对双精度浮点向量a进行规约的函数定义
    {                                                                                                \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                        \  // 检查双精度浮点向量a中的每个元素是否不是NaN
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {                                                   \  // 如果向量a中存在任何NaN元素，则执行以下操作
            const union { npy_uint64 i; double d;} pnan = {0x7ff8000000000000ull};                   \  // 创建一个union将NaN表示为双精度浮点数的结构
            return pnan.d;                                                                           \  // 返回NaN作为双精度浮点数
        }                                                                                            \
        return npyv_reduce_##INTRIN##_f64(a);                                                        \  // 调用指定指令集的双精度浮点向量a的规约函数，并返回结果
    }
// 定义 AVX2 指令集下的最小值和最大值函数，用于处理特定浮点数的边界值
NPY_IMPL_AVX2_REDUCE_MINMAX(min, 0x7f800000, 0x7ff0000000000000)
NPY_IMPL_AVX2_REDUCE_MINMAX(max, 0xff800000, 0xfff0000000000000)
// 取消定义 NPY_IMPL_AVX2_REDUCE_MINMAX 宏，结束对 AVX2 指令集下最小值和最大值函数的定义

// 定义 AVX256 指令集下的函数，用于对 8 位和 16 位整数执行最小值和最大值的归约操作
#define NPY_IMPL_AVX256_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                                        \
    // 归约 16 位整数数组的最小值或最大值
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m256i a)                                        \
    {                                                                                                \
        __m128i v128 = _mm_##VINTRIN##16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)); \
        __m128i v64 =  _mm_##VINTRIN##16(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(0, 0, 3, 2)));    \
        __m128i v32 = _mm_##VINTRIN##16(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));       \
        __m128i v16 = _mm_##VINTRIN##16(v32, _mm_shufflelo_epi16(v32, _MM_SHUFFLE(0, 0, 0, 1)));     \
        return (STYPE##16)_mm_cvtsi128_si32(v16);                                                    \
    }                                                                                                \
    // 归约 8 位整数数组的最小值或最大值
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m256i a)                                          \
    {                                                                                                \
        __m128i v128 = _mm_##VINTRIN##8(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));  \
        __m128i v64 =  _mm_##VINTRIN##8(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(0, 0, 3, 2)));     \
        __m128i v32 = _mm_##VINTRIN##8(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));        \
        __m128i v16 = _mm_##VINTRIN##8(v32, _mm_shufflelo_epi16(v32, _MM_SHUFFLE(0, 0, 0, 1)));      \
        __m128i v8 = _mm_##VINTRIN##8(v16, _mm_srli_epi16(v16, 8));                                  \
        return (STYPE##16)_mm_cvtsi128_si32(v8);                                                     \
    }
// 定义 AVX256 指令集下的最小值和最大值函数，针对无符号整数和有符号整数
NPY_IMPL_AVX256_REDUCE_MINMAX(npy_uint, min_u, min_epu)
NPY_IMPL_AVX256_REDUCE_MINMAX(npy_int,  min_s, min_epi)
NPY_IMPL_AVX256_REDUCE_MINMAX(npy_uint, max_u, max_epu)
NPY_IMPL_AVX256_REDUCE_MINMAX(npy_int,  max_s, max_epi)
// 取消定义 NPY_IMPL_AVX256_REDUCE_MINMAX 宏，结束对 AVX256 指令集下最小值和最大值函数的定义

// 定义对浮点数进行四舍五入到最近偶数的宏
#define npyv_rint_f32(A) _mm256_round_ps(A, _MM_FROUND_TO_NEAREST_INT)
#define npyv_rint_f64(A) _mm256_round_pd(A, _MM_FROUND_TO_NEAREST_INT)

// 定义对浮点数进行向上取整的宏
#define npyv_ceil_f32 _mm256_ceil_ps
#define npyv_ceil_f64 _mm256_ceil_pd

// 定义对浮点数进行截断取整的宏
#define npyv_trunc_f32(A) _mm256_round_ps(A, _MM_FROUND_TO_ZERO)
#define npyv_trunc_f64(A) _mm256_round_pd(A, _MM_FROUND_TO_ZERO)

// 定义对浮点数进行向下取整的宏
#define npyv_floor_f32 _mm256_floor_ps
#define npyv_floor_f64 _mm256_floor_pd

#endif // _NPY_SIMD_AVX2_MATH_H


这些宏和函数定义在一个 C/C++ 头文件中，主要用于处理 AVX2 和 AVX256 指令集下的向量化操作，包括整数的最小值和最大值归约，以及浮点数的四舍五入、向上取整、截断和向下取整操作。
```