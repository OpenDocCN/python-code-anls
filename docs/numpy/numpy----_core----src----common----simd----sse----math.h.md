# `.\numpy\numpy\_core\src\common\simd\sse\math.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_MATH_H
#define _NPY_SIMD_SSE_MATH_H
/***************************
 * Elementary
 ***************************/
// 平方根函数定义
#define npyv_sqrt_f32 _mm_sqrt_ps
#define npyv_sqrt_f64 _mm_sqrt_pd

// 倒数函数定义
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm_div_ps(_mm_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm_div_pd(_mm_set1_pd(1.0), a); }

// 绝对值函数定义
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
    return _mm_and_ps(
        a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))
    );
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
    return _mm_and_pd(
        a, _mm_castsi128_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
}

// 平方函数定义
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm_mul_pd(a, a); }

// 最大值函数定义，直接映射，不保证处理 NaN
#define npyv_max_f32 _mm_max_ps
#define npyv_max_f64 _mm_max_pd

// 最大值函数，支持 IEEE 浮点算术（IEC 60559）
// - 如果其中一个向量包含 NaN，则设置另一个向量对应元素
// - 只有当两个对应元素都是 NaN 时，才设置 NaN
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __m128i nn = npyv_notnan_f32(b);
    __m128 max = _mm_max_ps(a, b);
    return npyv_select_f32(nn, max, a);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i nn  = npyv_notnan_f64(b);
    __m128d max = _mm_max_pd(a, b);
    return npyv_select_f64(nn, max, a);
}
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    __m128i nn = npyv_notnan_f32(a);
    __m128 max = _mm_max_ps(a, b);
    return npyv_select_f32(nn, max, a);
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i nn  = npyv_notnan_f64(a);
    __m128d max = _mm_max_pd(a, b);
    return npyv_select_f64(nn, max, a);
}

// 最大值函数，整数操作
#ifdef NPY_HAVE_SSE41
    #define npyv_max_s8 _mm_max_epi8
    #define npyv_max_u16 _mm_max_epu16
    #define npyv_max_u32 _mm_max_epu32
    #define npyv_max_s32 _mm_max_epi32
#else
    NPY_FINLINE npyv_s8 npyv_max_s8(npyv_s8 a, npyv_s8 b)
    {
        return npyv_select_s8(npyv_cmpgt_s8(a, b), a, b);
    }
    NPY_FINLINE npyv_u16 npyv_max_u16(npyv_u16 a, npyv_u16 b)
    {
        return npyv_select_u16(npyv_cmpgt_u16(a, b), a, b);
    }
    NPY_FINLINE npyv_u32 npyv_max_u32(npyv_u32 a, npyv_u32 b)
    {
        return npyv_select_u32(npyv_cmpgt_u32(a, b), a, b);
    }
    NPY_FINLINE npyv_s32 npyv_max_s32(npyv_s32 a, npyv_s32 b)
    {
        return npyv_select_s32(npyv_cmpgt_s32(a, b), a, b);
    }
#endif
#define npyv_max_u8 _mm_max_epu8
#define npyv_max_s16 _mm_max_epi16
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    return npyv_select_u64(npyv_cmpgt_u64(a, b), a, b);
}
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    # 使用 SIMD 指令进行比较，选取 a 和 b 中对应元素中的大值
    return npyv_select_s64(npyv_cmpgt_s64(a, b), a, b);
// Minimum, natively mapping with no guarantees to handle NaN.
// 定义了几个宏用于执行单精度和双精度浮点数的最小值操作
#define npyv_min_f32 _mm_min_ps
#define npyv_min_f64 _mm_min_pd

// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
// 实现了对单精度浮点数向量的最小值操作，支持 IEEE 浮点算术（IEC 60559），
// - 如果两个向量中的一个包含 NaN，则设置另一个向量对应元素的值
// - 只有当两个对应的元素都是 NaN 时，才会设置为 NaN
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    // 使用 nopyv_notnan_f32 函数获取非 NaN 的掩码
    __m128i nn = npyv_notnan_f32(b);
    // 使用 SSE 指令计算 a 和 b 向量中的最小值
    __m128 min = _mm_min_ps(a, b);
    // 根据非 NaN 的掩码选择最小值或原始值来创建新的向量
    return npyv_select_f32(nn, min, a);
}

// 类似于上述函数，实现了对双精度浮点数向量的最小值操作
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i nn  = npyv_notnan_f64(b);
    __m128d min = _mm_min_pd(a, b);
    return npyv_select_f64(nn, min, a);
}

// 实现了对单精度浮点数向量的最小值操作
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    __m128i nn = npyv_notnan_f32(a);
    __m128 min = _mm_min_ps(a, b);
    return npyv_select_f32(nn, min, a);
}

// 实现了对双精度浮点数向量的最小值操作
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i nn  = npyv_notnan_f64(a);
    __m128d min = _mm_min_pd(a, b);
    return npyv_select_f64(nn, min, a);
}

// Minimum, integer operations
#ifdef NPY_HAVE_SSE41
    // 使用 SSE4.1 指令集提供的最小值操作宏
    #define npyv_min_s8 _mm_min_epi8
    #define npyv_min_u16 _mm_min_epu16
    #define npyv_min_u32 _mm_min_epu32
    #define npyv_min_s32 _mm_min_epi32
#else
    // 对于不支持 SSE4.1 的平台，实现了对应整数类型的最小值操作
    NPY_FINLINE npyv_s8 npyv_min_s8(npyv_s8 a, npyv_s8 b)
    {
        // 使用条件选择函数实现 s8 类型向量的最小值操作
        return npyv_select_s8(npyv_cmplt_s8(a, b), a, b);
    }

    NPY_FINLINE npyv_u16 npyv_min_u16(npyv_u16 a, npyv_u16 b)
    {
        // 使用条件选择函数实现 u16 类型向量的最小值操作
        return npyv_select_u16(npyv_cmplt_u16(a, b), a, b);
    }

    NPY_FINLINE npyv_u32 npyv_min_u32(npyv_u32 a, npyv_u32 b)
    {
        // 使用条件选择函数实现 u32 类型向量的最小值操作
        return npyv_select_u32(npyv_cmplt_u32(a, b), a, b);
    }

    NPY_FINLINE npyv_s32 npyv_min_s32(npyv_s32 a, npyv_s32 b)
    {
        // 使用条件选择函数实现 s32 类型向量的最小值操作
        return npyv_select_s32(npyv_cmplt_s32(a, b), a, b);
    }
#endif

// 使用 SSE2 指令集提供的最小值操作宏
#define npyv_min_u8 _mm_min_epu8
#define npyv_min_s16 _mm_min_epi16

// 实现了对 u64 类型向量的最小值操作
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    // 使用条件选择函数实现 u64 类型向量的最小值操作
    return npyv_select_u64(npyv_cmplt_u64(a, b), a, b);
}

// 实现了对 s64 类型向量的最小值操作
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    // 使用条件选择函数实现 s64 类型向量的最小值操作
    return npyv_select_s64(npyv_cmplt_s64(a, b), a, b);
}

// reduce min&max for 32&64-bits
// 定义了宏用于实现对 32 位和 64 位数据类型的最小和最大值缩减操作
#define NPY_IMPL_SSE_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                                     \
    NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m128i a)                                  \
    {                                                                                          \
        // 使用 SSE 指令集进行数据重排和比较，得到最小值
        __m128i v64 =  npyv_##INTRIN##32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));    \
        // 再次重排和比较，得到最小值
        __m128i v32 = npyv_##INTRIN##32(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1))); \
        // 将结果转换为对应的 32 位整数并返回
        return (STYPE##32)_mm_cvtsi128_si32(v32);                                              \
    }                                                                                          \
    NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m128i a)                                  \
    {
        # 创建一个名为 v64 的变量，使用 npyv_##INTRIN##64 函数生成一个 64 位的数据向量，
        # 这个函数以 a 作为输入，通过对 a 进行特定的位操作和洗牌生成结果。
        __m128i v64  = npyv_##INTRIN##64(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));

        # 将 v64 中的数据提取出来，并转换成 STYPE##64 类型的值，然后作为函数的返回值返回。
        return (STYPE##64)npyv_extract0_u64(v64);
    }


这段代码中，使用了一些宏定义和 SIMD（单指令多数据流）指令，用于处理数据向量化操作，具体的 INTRIN 和 STYPE 是根据具体的上下文和宏定义来确定的。
// 定义宏 NPY_IMPL_SSE_REDUCE_MINMAX，用于生成不同类型和函数的 SSE 指令实现
NPY_IMPL_SSE_REDUCE_MINMAX(npy_uint, min_u, min_epu)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_int,  min_s, min_epi)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_uint, max_u, max_epu)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_int,  max_s, max_epi)
// 取消宏定义 NPY_IMPL_SSE_REDUCE_MINMAX

// 宏重新定义 NPY_IMPL_SSE_REDUCE_MINMAX，用于生成 SSE 指令实现
#define NPY_IMPL_SSE_REDUCE_MINMAX(INTRIN, INF, INF64)                                          \
    // 定义单精度浮点数的 SSE 指令实现，用于计算最小值或最大值
    NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                                    \
    {                                                                                           \
        // 将向量 a 按指定方式重新排列，得到 64 位浮点数向量 v64
        __m128 v64 =  _mm_##INTRIN##_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 3, 2)));      \
        // 将 v64 按指定方式重新排列，得到 32 位浮点数向量 v32
        __m128 v32 = _mm_##INTRIN##_ps(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1))); \
        // 返回 v32 的第一个元素作为浮点数结果
        return _mm_cvtss_f32(v32);                                                              \
    }                                                                                           \
    // 定义双精度浮点数的 SSE 指令实现，用于计算最小值或最大值
    NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)                                   \
    {                                                                                           \
        // 将向量 a 按指定方式重新排列，得到 64 位双精度浮点数向量 v64
        __m128d v64 = _mm_##INTRIN##_pd(a, _mm_shuffle_pd(a, a, _MM_SHUFFLE(0, 0, 0, 1)));      \
        // 返回 v64 的第一个元素作为双精度浮点数结果
        return _mm_cvtsd_f64(v64);                                                              \
    }                                                                                           \
    // 定义单精度浮点数的 SSE 指令实现，用于计算带有处理 NaN 的最小值或最大值
    NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)                                   \
    {                                                                                           \
        // 获取 a 中非 NaN 的掩码
        npyv_b32 notnan = npyv_notnan_f32(a);                                                   \
        // 如果所有元素都是 NaN，则返回 a 的第一个元素
        if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                                              \
            return _mm_cvtss_f32(a);                                                            \
        }                                                                                       \
        // 选取非 NaN 的元素或者用 INF 替换 NaN 的元素
        a = npyv_select_f32(notnan, a, npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));         \
        // 调用 npyv_reduce_##INTRIN##_f32 计算最终结果
        return npyv_reduce_##INTRIN##_f32(a);                                                   \
    }                                                                                           \
    // 定义双精度浮点数的 SSE 指令实现，用于计算带有处理 NaN 的最小值或最大值
    NPY_FINLINE double npyv_reduce_##INTRIN##p_f64(npyv_f64 a)                                  \
    {
        # 使用 SIMD 向量操作，检查双精度浮点向量 a 中的非 NaN 元素
        npyv_b64 notnan = npyv_notnan_f64(a);
        # 如果向量 a 中没有非 NaN 元素，则返回向量中的第一个元素作为结果
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {
            return _mm_cvtsd_f64(a);
        }
        # 将向量 a 中非 NaN 元素保留，将 NaN 元素替换为正无穷，并重新解释为双精度浮点数向量
        a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(npyv_setall_u64(INF64)));
        # 对非 NaN 元素进行指定的归约操作 INTRIN，并返回结果
        return npyv_reduce_##INTRIN##_f64(a);
    }
    NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)
    {
        # 使用 SIMD 向量操作，检查单精度浮点向量 a 中的非 NaN 元素
        npyv_b32 notnan = npyv_notnan_f32(a);
        # 如果向量 a 中有 NaN 元素，则返回特定的 NaN 值
        if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {
            const union { npy_uint32 i; float f;} pnan = {0x7fc00000UL};
            return pnan.f;
        }
        # 对非 NaN 元素进行指定的归约操作 INTRIN，并返回结果
        return npyv_reduce_##INTRIN##_f32(a);
    }
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)
    {
        # 使用 SIMD 向量操作，检查双精度浮点向量 a 中的非 NaN 元素
        npyv_b64 notnan = npyv_notnan_f64(a);
        # 如果向量 a 中有 NaN 元素，则返回特定的 NaN 值
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {
            const union { npy_uint64 i; double d;} pnan = {0x7ff8000000000000ull};
            return pnan.d;
        }
        # 对非 NaN 元素进行指定的归约操作 INTRIN，并返回结果
        return npyv_reduce_##INTRIN##_f64(a);
    }
// 定义 SSE 指令的宏，用于实现最小值和最大值的规约操作
#define NPY_IMPL_SSE_REDUCE_MINMAX(STYPE, INTRIN)                                                    \
    // 定义内联函数，用于减少 SSE 寄存器中 16 位数据的最小值或最大值
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m128i a)                                        \
    {                                                                                                \
        // 将输入的 128 位整数按指定方式重组成 64 位整数
        __m128i v64 =  npyv_##INTRIN##16(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));          \
        // 进一步重组以得到 32 位整数
        __m128i v32 = npyv_##INTRIN##16(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));       \
        // 最后得到 16 位整数，返回其结果
        __m128i v16 = npyv_##INTRIN##16(v32, _mm_shufflelo_epi16(v32, _MM_SHUFFLE(0, 0, 0, 1)));     \
        return (STYPE##16)_mm_cvtsi128_si32(v16);                                                    \
    }                                                                                                \
    // 定义内联函数，用于减少 SSE 寄存器中 8 位数据的最小值或最大值
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m128i a)                                          \
    {                                                                                                \
        // 将输入的 128 位整数按指定方式重组成 64 位整数
        __m128i v64 =  npyv_##INTRIN##8(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));           \
        // 进一步重组以得到 32 位整数
        __m128i v32 = npyv_##INTRIN##8(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));        \
        // 再得到 16 位整数
        __m128i v16 = npyv_##INTRIN##8(v32, _mm_shufflelo_epi16(v32, _MM_SHUFFLE(0, 0, 0, 1)));      \
        // 最后得到 8 位整数，通过逻辑右移实现除以 2
        __m128i v8 = npyv_##INTRIN##8(v16, _mm_srli_epi16(v16, 8));                                  \
        return (STYPE##16)_mm_cvtsi128_si32(v8);                                                     \
    }

// 使用宏定义实现不同数据类型的最小值和最大值规约操作
NPY_IMPL_SSE_REDUCE_MINMAX(npy_uint, min_u)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_int,  min_s)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_uint, max_u)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_int,  max_s)
#undef NPY_IMPL_SSE_REDUCE_MINMAX

// 如果支持 SSE4.1 指令集，则使用 SSE4.1 的指令执行浮点数四舍五入到最近偶数
NPY_FINLINE npyv_f32 npyv_rint_f32(npyv_f32 a)
{
#ifdef NPY_HAVE_SSE41
    // 使用 SSE4.1 的指令实现浮点数的四舍五入到最近偶数
    return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
#else
    // 如果不支持 SSE4.1 指令集，则使用以下代码实现浮点数的四舍五入到最近偶数
    // 设置一个 32 位浮点数全为 -0.0f 的向量
    const __m128 szero = _mm_set1_ps(-0.0f);
    // 设置一个 32 位整数表示指数掩码为 0xff000000
    const __m128i exp_mask = _mm_set1_epi32(0xff000000);

    // 生成一个掩码，用于检测非无限值
    __m128i nfinite_mask = _mm_slli_epi32(_mm_castps_si128(a), 1);
    nfinite_mask = _mm_and_si128(nfinite_mask, exp_mask);
    nfinite_mask = _mm_cmpeq_epi32(nfinite_mask, exp_mask);

    // 消除 NaN 和 Inf，以避免无效的浮点错误
    __m128 x = _mm_xor_ps(a, _mm_castsi128_ps(nfinite_mask));
    // 将浮点数转换为整数进行四舍五入
    __m128i roundi = _mm_cvtps_epi32(x);
    // 将整数转换回浮点数
    __m128 round = _mm_cvtepi32_ps(roundi);
    // 处理带符号的零
    round = _mm_or_ps(round, _mm_and_ps(a, szero));
    // 如果溢出则返回原始值
    __m128i overflow_mask = _mm_cmpeq_epi32(roundi, _mm_castps_si128(szero));
    // 如果溢出或者非有限值，则返回原始值，否则返回四舍五入后的结果
    return npyv_select_f32(_mm_or_si128(nfinite_mask, overflow_mask), a, round);
#endif
}
NPY_FINLINE npyv_f64 npyv_rint_f64(npyv_f64 a)
{
#ifdef NPY_HAVE_SSE41
    // 如果支持 SSE4.1 指令集，则使用 SSE 指令进行向最近整数舍入并返回结果
    return _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
#else
    // 如果不支持 SSE4.1 指令集，则进行手动实现舍入操作

    // 设置常量 -0.0 的向量
    const __m128d szero = _mm_set1_pd(-0.0);
    // 设置常量 2^52 的向量
    const __m128d two_power_52 = _mm_set1_pd(0x10000000000000);

    // 计算出 NaN 掩码，用于排除 NaN 值，避免在比较操作中出现无效的浮点错误
    __m128d nan_mask = _mm_cmpunord_pd(a, a);

    // 计算绝对值向量，并处理 NaN 值
    __m128d abs_x = npyv_abs_f64(_mm_xor_pd(nan_mask, a));

    // 执行舍入操作，加上魔法数 2^52
    __m128d round = _mm_sub_pd(_mm_add_pd(two_power_52, abs_x), two_power_52);

    // 复制原始值的符号到舍入结果中
    round = _mm_or_pd(round, _mm_and_pd(a, szero));

    // 如果 |a| >= 2^52 或者 a 是 NaN，则返回原始值 a；否则返回舍入结果 round
    __m128d mask = _mm_cmpge_pd(abs_x, two_power_52);
    mask = _mm_or_pd(mask, nan_mask);
    return npyv_select_f64(_mm_castpd_si128(mask), a, round);
#endif
}

// ceil
#ifdef NPY_HAVE_SSE41
    // 如果支持 SSE4.1 指令集，则使用 SSE 指令进行向上取整并返回结果
    #define npyv_ceil_f32 _mm_ceil_ps
    #define npyv_ceil_f64 _mm_ceil_pd
#else
    // 如果不支持 SSE4.1 指令集，则手动实现单精度浮点数向上取整操作
    NPY_FINLINE npyv_f32 npyv_ceil_f32(npyv_f32 a)
    {
        const __m128 one = _mm_set1_ps(1.0f);
        const __m128 szero = _mm_set1_ps(-0.0f);
        const __m128i exp_mask = _mm_set1_epi32(0xff000000);

        // 计算非无穷大的掩码，用于排除 NaN 和 Inf 值，避免在比较操作中出现无效的浮点错误
        __m128i nfinite_mask = _mm_slli_epi32(_mm_castps_si128(a), 1);
        nfinite_mask = _mm_and_si128(nfinite_mask, exp_mask);
        nfinite_mask = _mm_cmpeq_epi32(nfinite_mask, exp_mask);

        // 处理 NaN 和 Inf 值，将它们的位反转，并与原始值异或，以排除这些值
        __m128 x = _mm_xor_ps(a, _mm_castsi128_ps(nfinite_mask));

        // 执行舍入操作，并将结果转换为整数
        __m128i roundi = _mm_cvtps_epi32(x);

        // 将整数结果转换回单精度浮点数
        __m128 round = _mm_cvtepi32_ps(roundi);

        // 执行向上取整操作
        __m128 ceil = _mm_add_ps(round, _mm_and_ps(_mm_cmplt_ps(round, x), one));

        // 将符号位从原始值复制到向上取整结果中
        ceil = _mm_or_ps(ceil, _mm_and_ps(a, szero));

        // 如果溢出，返回原始值 a；否则返回向上取整结果 ceil
        __m128i overflow_mask = _mm_cmpeq_epi32(roundi, _mm_castps_si128(szero));
        return npyv_select_f32(_mm_or_si128(nfinite_mask, overflow_mask), a, ceil);
    }

    // 手动实现双精度浮点数向上取整操作
    NPY_FINLINE npyv_f64 npyv_ceil_f64(npyv_f64 a)
    {
        // 创建一个包含值为1.0的双精度浮点数向量
        const __m128d one = _mm_set1_pd(1.0);
        // 创建一个包含值为-0.0的双精度浮点数向量
        const __m128d szero = _mm_set1_pd(-0.0);
        // 创建一个包含值为2^52的双精度浮点数向量
        const __m128d two_power_52 = _mm_set1_pd(0x10000000000000);
        
        // 创建一个向量，其元素为a中的NaN的掩码
        __m128d nan_mask = _mm_cmpunord_pd(a, a);
        
        // 通过异或操作去除NaN，以避免在后续的比较操作中出现无效的浮点数错误
        __m128d x = _mm_xor_pd(nan_mask, a);
        
        // 计算x的绝对值向量
        __m128d abs_x = npyv_abs_f64(x);
        
        // 计算x的符号向量
        __m128d sign_x = _mm_and_pd(x, szero);
        
        // 使用魔术数2^52进行四舍五入
        // 假设MXCSR寄存器已设置为四舍五入模式
        __m128d round = _mm_sub_pd(_mm_add_pd(two_power_52, abs_x), two_power_52);
        
        // 根据x的符号向量进行符号位的设置
        round = _mm_or_pd(round, sign_x);
        
        // 执行向上取整操作
        __m128d ceil = _mm_add_pd(round, _mm_and_pd(_mm_cmplt_pd(round, x), one));
        
        // 保持0.0的符号
        ceil = _mm_or_pd(ceil, sign_x);
        
        // 如果|a| >= 2^52 或 a == NaN，则返回a，否则返回ceil
        __m128d mask = _mm_cmpge_pd(abs_x, two_power_52);
        mask = _mm_or_pd(mask, nan_mask);
        
        // 根据掩码选择返回a或ceil的元素
        return npyv_select_f64(_mm_castpd_si128(mask), a, ceil);
    }
// 如果定义了 NPY_HAVE_SSE41，则定义宏 npyv_floor_f32 为 _mm_floor_ps，即使用 SSE 指令集的单精度浮点数向下取整操作
// 如果定义了 NPY_HAVE_SSE41，则定义宏 npyv_floor_f64 为 _mm_floor_pd，即使用 SSE 指令集的双精度浮点数向下取整操作
#ifdef NPY_HAVE_SSE41
    #define npyv_floor_f32 _mm_floor_ps
    #define npyv_floor_f64 _mm_floor_pd
#else
    // 否则，定义 npyv_floor_f32 函数，对单精度浮点数 a 进行向下取整操作
    NPY_FINLINE npyv_f32 npyv_floor_f32(npyv_f32 a)
    {
        // 设置一个单精度浮点数 -0.0 的向量
        const __m128 szero = _mm_set1_ps(-0.0f);
        // 设置一个整型向量，其高位字节为 0xff000000
        const __m128i exp_mask = _mm_set1_epi32(0xff000000);

        // 创建一个掩码，用于标识 a 中有限数字部分的位置
        __m128i nfinite_mask = _mm_slli_epi32(_mm_castps_si128(a), 1);
        nfinite_mask = _mm_and_si128(nfinite_mask, exp_mask);
        nfinite_mask = _mm_cmpeq_epi32(nfinite_mask, exp_mask);

        // 消除 NaN 和无穷大，以避免无效的浮点数错误
        __m128 x = _mm_xor_ps(a, _mm_castsi128_ps(nfinite_mask));
        // 将浮点数 x 向下取整到最接近的整数，得到整型向量
        __m128i trunci = _mm_cvttps_epi32(x);
        // 将整型向量转换回单精度浮点数向量
        __m128 trunc = _mm_cvtepi32_ps(trunci);
        // 将负零保留为负零，例如 -0.5 变为 -0.0
        trunc = _mm_or_ps(trunc, _mm_and_ps(a, szero));
        // 如果溢出则返回原始值 a
        __m128i overflow_mask = _mm_cmpeq_epi32(trunci, _mm_castps_si128(szero));
        // 如果 a 溢出或不是有限数字，则返回 a；否则返回向下取整后的值
        return npyv_select_f32(_mm_or_si128(nfinite_mask, overflow_mask), a, trunc);
    }

    // 否则，定义 npyv_floor_f64 函数，对双精度浮点数 a 进行向下取整操作
    NPY_FINLINE npyv_f64 npyv_floor_f64(npyv_f64 a)
    {
        // 设置一个双精度浮点数 1.0 的向量
        const __m128d one = _mm_set1_pd(1.0);
        // 设置一个双精度浮点数 -0.0 的向量
        const __m128d szero = _mm_set1_pd(-0.0);
        // 设置一个双精度浮点数 2^52 的向量
        const __m128d two_power_52 = _mm_set1_pd(0x10000000000000);
        // 创建一个掩码，用于标识 a 中 NaN 的位置
        __m128d nan_mask = _mm_cmpunord_pd(a, a);
        // 消除 NaN，以避免无效的浮点数错误在 cmpge 内
        __m128d abs_x = npyv_abs_f64(_mm_xor_pd(nan_mask, a));
        // 将 abs_x 向下取整，通过加上 2^52 的魔数来实现
        // 假设 MXCSR 寄存器已设置为舍入
        __m128d abs_round = _mm_sub_pd(_mm_add_pd(two_power_52, abs_x), two_power_52);
        // 计算需要减去的值，以确保正确的向下取整
        __m128d subtrahend = _mm_and_pd(_mm_cmpgt_pd(abs_round, abs_x), one);
        // 进行向下取整操作
        __m128d trunc = _mm_sub_pd(abs_round, subtrahend);
        // 赋予结果相同的符号
        trunc = _mm_or_pd(trunc, _mm_and_pd(a, szero));
        // 如果 |a| >= 2^52 或者 a 是 NaN，则返回 a；否则返回向下取整后的值
        __m128d mask = _mm_cmpge_pd(abs_x, two_power_52);
        mask = _mm_or_pd(mask, nan_mask);
        return npyv_select_f64(_mm_castpd_si128(mask), a, trunc);
    }
#endif
#endif
    {
        // 定义一个包含单精度浮点数 1.0 的 SSE 寄存器
        const __m128 one = _mm_set1_ps(1.0f);
        // 定义一个包含单精度浮点数 -0.0 的 SSE 寄存器
        const __m128 szero = _mm_set1_ps(-0.0f);
        // 定义一个包含 0xff000000 的 SSE 寄存器，用于提取指数部分
        const __m128i exp_mask = _mm_set1_epi32(0xff000000);
    
        // 创建一个掩码，用于标识非无穷的浮点数（finite）
        __m128i nfinite_mask = _mm_slli_epi32(_mm_castps_si128(a), 1);
                nfinite_mask = _mm_and_si128(nfinite_mask, exp_mask);
                nfinite_mask = _mm_cmpeq_epi32(nfinite_mask, exp_mask);
    
        // 消除 NaN 和无穷大，以避免无效的浮点数错误
        __m128 x = _mm_xor_ps(a, _mm_castsi128_ps(nfinite_mask));
        // 将浮点数转换为整数并进行舍入
        __m128i roundi = _mm_cvtps_epi32(x);
        // 将整数舍入结果转换回浮点数
        __m128 round = _mm_cvtepi32_ps(roundi);
        // 计算向下取整结果
        __m128 floor = _mm_sub_ps(round, _mm_and_ps(_mm_cmpgt_ps(round, x), one));
        // 考虑到带符号的零
        floor = _mm_or_ps(floor, _mm_and_ps(a, szero));
        // 如果发生溢出，则返回原始值 a
        __m128i overflow_mask = _mm_cmpeq_epi32(roundi, _mm_castps_si128(szero));
        // 如果数值溢出或者是非有限浮点数，则返回 a，否则返回 floor 的结果
        return npyv_select_f32(_mm_or_si128(nfinite_mask, overflow_mask), a, floor);
    }
    NPY_FINLINE npyv_f64 npyv_floor_f64(npyv_f64 a)
    {
        // 定义一个包含双精度浮点数 1.0 的 SSE 寄存器
        const __m128d one = _mm_set1_pd(1.0);
        // 定义一个包含双精度浮点数 -0.0 的 SSE 寄存器
        const __m128d szero = _mm_set1_pd(-0.0);
        // 定义一个包含 2^52 的双精度浮点数的 SSE 寄存器
        const __m128d two_power_52 = _mm_set1_pd(0x10000000000000);
        // 创建一个 NaN 掩码，用于标识 NaN 值
        __m128d nan_mask = _mm_cmpunord_pd(a, a);
        // 消除 NaN 以避免在 cmpge 内部出现无效的浮点数错误
        __m128d x = _mm_xor_pd(nan_mask, a);
        // 计算绝对值
        __m128d abs_x = npyv_abs_f64(x);
        // 提取符号位
        __m128d sign_x = _mm_and_pd(x, szero);
        // 通过加上魔法数 2^52 进行舍入
        // 假设 MXCSR 寄存器已设置为四舍五入
        __m128d round = _mm_sub_pd(_mm_add_pd(two_power_52, abs_x), two_power_52);
        // 复制符号位
        round = _mm_or_pd(round, sign_x);
        // 计算向下取整结果
        __m128d floor = _mm_sub_pd(round, _mm_and_pd(_mm_cmpgt_pd(round, x), one));
        // 如果 |a| >= 2^52 或者 a 是 NaN，则返回 a；否则返回 floor 的结果
        __m128d mask = _mm_cmpge_pd(abs_x, two_power_52);
               mask = _mm_or_pd(mask, nan_mask);
        return npyv_select_f64(_mm_castpd_si128(mask), a, floor);
    }
#endif // NPY_HAVE_SSE41



// 结束条件：如果定义了 NPY_HAVE_SSE41 宏，则结束当前代码块
#endif // _NPY_SIMD_SSE_MATH_H



// 结束条件：结束 _NPY_SIMD_SSE_MATH_H 头文件的条件编译块
```