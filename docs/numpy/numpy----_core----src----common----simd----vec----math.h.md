# `.\numpy\numpy\_core\src\common\simd\vec\math.h`

```py
#ifndef NPY_SIMD
    // 如果未定义 NPY_SIMD 宏，则输出错误消息并终止编译
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_MATH_H
#define _NPY_SIMD_VEC_MATH_H
/***************************
 * Elementary
 ***************************/
// 平方根函数定义
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则使用向量化平方根函数 vec_sqrt
    #define npyv_sqrt_f32 vec_sqrt
#endif
// 双精度浮点数平方根函数定义
#define npyv_sqrt_f64 vec_sqrt

// 倒数函数定义
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则定义单精度浮点数的倒数函数 npyv_recip_f32
    NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
    {
        // 定义常量 one 为所有元素为 1.0f 的向量
        const npyv_f32 one = npyv_setall_f32(1.0f);
        // 返回 one 与 a 的逐元素除法结果
        return vec_div(one, a);
    }
#endif
// 双精度浮点数的倒数函数定义
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{
    // 定义常量 one 为所有元素为 1.0 的向量
    const npyv_f64 one = npyv_setall_f64(1.0);
    // 返回 one 与 a 的逐元素除法结果
    return vec_div(one, a);
}

// 绝对值函数定义
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则使用向量化绝对值函数 vec_abs
    #define npyv_abs_f32 vec_abs
#endif
// 双精度浮点数绝对值函数定义
#define npyv_abs_f64 vec_abs

// 平方函数定义
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则定义单精度浮点数的平方函数 npyv_square_f32
    NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
    { return vec_mul(a, a); } // 返回 a 与自身逐元素相乘的结果，即平方
#endif
// 双精度浮点数的平方函数定义
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return vec_mul(a, a); } // 返回 a 与自身逐元素相乘的结果，即平方

// 最大值函数定义，不保证处理 NaN
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则使用向量化最大值函数 vec_max
    #define npyv_max_f32 vec_max
#endif
// 双精度浮点数最大值函数定义，不保证处理 NaN
#define npyv_max_f64 vec_max
// 最大值函数定义，支持 IEEE 浮点算术 (IEC 60559)
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则定义单精度浮点数的最大值函数 npyv_maxp_f32
    #define npyv_maxp_f32 vec_max
#endif
// 对于不同架构，选择是否使用向量化最大值函数，如 vfmindb & vfmaxdb appears in zarch12
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    // 如果定义了 NPY_HAVE_VXE 或 NPY_HAVE_VSX，使用向量化最大值函数 vec_max
    #define npyv_maxp_f64 vec_max
#else
    // 否则，在没有特定支持的架构下，使用自定义最大值函数 npyv_maxp_f64
    NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
    {
        // 定义非 NaN 掩码
        npyv_b64 nn_a = npyv_notnan_f64(a);
        npyv_b64 nn_b = npyv_notnan_f64(b);
        // 返回根据非 NaN 掩码选择的最大值
        return vec_max(vec_sel(b, a, nn_a), vec_sel(a, b, nn_b));
    }
#endif
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则定义单精度浮点数的非 NaN 最大值函数 npyv_maxn_f32
    NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
    {
        // 定义非 NaN 掩码
        npyv_b32 nn_a = npyv_notnan_f32(a);
        npyv_b32 nn_b = npyv_notnan_f32(b);
        // 计算最大值并根据非 NaN 掩码选择
        npyv_f32 max = vec_max(a, b);
        return vec_sel(b, vec_sel(a, max, nn_a), nn_b);
    }
#endif
// 双精度浮点数的非 NaN 最大值函数定义
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    // 定义非 NaN 掩码
    npyv_b64 nn_a = npyv_notnan_f64(a);
    npyv_b64 nn_b = npyv_notnan_f64(b);
    // 计算最大值并根据非 NaN 掩码选择
    npyv_f64 max = vec_max(a, b);
    return vec_sel(b, vec_sel(a, max, nn_a), nn_b);
}

// 最大值函数定义，整数运算
#define npyv_max_u8 vec_max
#define npyv_max_s8 vec_max
#define npyv_max_u16 vec_max
#define npyv_max_s16 vec_max
#define npyv_max_u32 vec_max
#define npyv_max_s32 vec_max
#define npyv_max_u64 vec_max
#define npyv_max_s64 vec_max

// 最小值函数定义，不保证处理 NaN
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则使用向量化最小值函数 vec_min
    #define npyv_min_f32 vec_min
#endif
// 双精度浮点数最小值函数定义，不保证处理 NaN
#define npyv_min_f64 vec_min
// 最小值函数定义，支持 IEEE 浮点算术 (IEC 60559)
#if NPY_SIMD_F32
    // 如果支持单精度浮点运算，则定义单精度浮点数的最小值函数 npyv_minp_f32
    #define npyv_minp_f32 vec_min
#endif
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    // 如果定义了 NPY_HAVE_VXE 或 NPY_HAVE_VSX，使用向量化最小值函数 vec_min
    #define npyv_minp_f64 vec_min
#else
    // 定义一个内联函数 npyv_minp_f64，用于计算两个 npyv_f64 类型向量 a 和 b 的最小值
    NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
    {
        // 获取向量 a 中非 NaN 元素的掩码
        npyv_b64 nn_a = npyv_notnan_f64(a);
        // 获取向量 b 中非 NaN 元素的掩码
        npyv_b64 nn_b = npyv_notnan_f64(b);
        // 使用掩码 nn_a 和 nn_b，选择 a 和 b 中的对应元素形成新的向量，以确保不考虑 NaN 值
        return vec_min(vec_sel(b, a, nn_a), vec_sel(a, b, nn_b));
    }
#ifdef
#if NPY_SIMD_F32
    // 如果编译器支持单精度 SIMD 指令集，则定义以下函数
    NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
    {
        // 获取非 NaN 的元素掩码
        npyv_b32 nn_a = npyv_notnan_f32(a);
        npyv_b32 nn_b = npyv_notnan_f32(b);
        // 计算向量 a 和 b 的最小值
        npyv_f32 min = vec_min(a, b);
        // 根据 nn_b 选择 b 或者 a 的最小值
        return vec_sel(b, vec_sel(a, min, nn_a), nn_b);
    }
#endif

// 定义双精度版本的最小值函数
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    // 获取双精度向量 a 和 b 中非 NaN 的掩码
    npyv_b64 nn_a = npyv_notnan_f64(a);
    npyv_b64 nn_b = npyv_notnan_f64(b);
    // 计算向量 a 和 b 的最小值
    npyv_f64 min = vec_min(a, b);
    // 根据 nn_b 选择 b 或者 a 的最小值
    return vec_sel(b, vec_sel(a, min, nn_a), nn_b);
}

// Minimum, integer operations
// 定义各整数类型的最小值宏
#define npyv_min_u8 vec_min
#define npyv_min_s8 vec_min
#define npyv_min_u16 vec_min
#define npyv_min_s16 vec_min
#define npyv_min_u32 vec_min
#define npyv_min_s32 vec_min
#define npyv_min_u64 vec_min
#define npyv_min_s64 vec_min

// 定义通用的向量减少操作宏，用于计算最小值和最大值
#define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, STYPE, SFX)                  \
    NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \
    {                                                                   \
        // 使用 SIMD 指令实现向量的最小/最大值减少操作
        npyv_##SFX r = vec_##INTRIN(a, vec_sld(a, a, 8));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 4));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 2));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 1));               \
        // 提取结果向量中的第一个元素作为最终的减少结果
        return (npy_##STYPE)vec_extract(r, 0);                          \
    }

// 定义各种整数类型和操作的最小值和最大值向量减少操作
NPY_IMPL_VEC_REDUCE_MINMAX(min, uint8, u8)
NPY_IMPL_VEC_REDUCE_MINMAX(max, uint8, u8)
NPY_IMPL_VEC_REDUCE_MINMAX(min, int8, s8)
NPY_IMPL_VEC_REDUCE_MINMAX(max, int8, s8)
#undef NPY_IMPL_VEC_REDUCE_MINMAX

NPY_IMPL_VEC_REDUCE_MINMAX(min, uint16, u16)
NPY_IMPL_VEC_REDUCE_MINMAX(max, uint16, u16)
NPY_IMPL_VEC_REDUCE_MINMAX(min, int16, s16)
NPY_IMPL_VEC_REDUCE_MINMAX(max, int16, s16)
#undef NPY_IMPL_VEC_REDUCE_MINMAX

NPY_IMPL_VEC_REDUCE_MINMAX(min, uint32, u32)
NPY_IMPL_VEC_REDUCE_MINMAX(max, uint32, u32)
NPY_IMPL_VEC_REDUCE_MINMAX(min, int32, s32)
NPY_IMPL_VEC_REDUCE_MINMAX(max, int32, s32)
#undef NPY_IMPL_VEC_REDUCE_MINMAX
#define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, STYPE, SFX)                  \
    NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \  // 定义一个模板宏，用于生成特定类型和后缀的向量归约函数
    {                                                                   \
        npyv_##SFX r = vec_##INTRIN(a, vec_sld(a, a, 8));               \  // 使用指定的向量指令进行归约操作
        return (npy_##STYPE)vec_extract(r, 0);                          \  // 从归约结果向量中提取第一个元素作为结果并返回
        (void)r;                                                         \  // 使用 void 来避免编译器警告未使用变量的提示
    }

NPY_IMPL_VEC_REDUCE_MINMAX(min, uint64, u64)   // 实例化模板宏，生成 unsigned 64 位整数的最小值归约函数
NPY_IMPL_VEC_REDUCE_MINMAX(max, uint64, u64)   // 实例化模板宏，生成 unsigned 64 位整数的最大值归约函数
NPY_IMPL_VEC_REDUCE_MINMAX(min, int64, s64)    // 实例化模板宏，生成 signed 64 位整数的最小值归约函数
NPY_IMPL_VEC_REDUCE_MINMAX(max, int64, s64)    // 实例化模板宏，生成 signed 64 位整数的最大值归约函数

#undef NPY_IMPL_VEC_REDUCE_MINMAX  // 取消宏定义，清理预处理器环境

#if NPY_SIMD_F32
    #define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, INF)                   \  // 定义一个模板宏，用于生成特定浮点数类型的向量归约函数
        NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)      \  // 内联函数，生成 float 类型的向量归约函数
        {                                                             \
            npyv_f32 r = vec_##INTRIN(a, vec_sld(a, a, 8));           \  // 使用指定的向量指令进行归约操作
                     r = vec_##INTRIN(r, vec_sld(r, r, 4));           \  // 进行额外的归约操作
            return vec_extract(r, 0);                                 \  // 从归约结果向量中提取第一个元素作为结果并返回
        }                                                             \
        NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)     \  // 内联函数，用于处理特殊情况下的浮点数归约
        {                                                             \
            return npyv_reduce_##INTRIN##_f32(a);                     \  // 直接调用上面定义的 float 归约函数
        }                                                             \
        NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)     \  // 内联函数，用于处理特殊情况下的浮点数归约
        {                                                             \
            npyv_b32 notnan = npyv_notnan_f32(a);                     \  // 检查是否有 NaN 值
            if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {                \  // 如果有 NaN 值，则处理特殊情况
                const union { npy_uint32 i; float f;}                 \  // 定义联合体以便直接使用位操作
                pnan = {0x7fc00000UL};                                \  // 定义 NaN 值的位表示
                return pnan.f;                                        \  // 返回 NaN 值
            }                                                         \
            return npyv_reduce_##INTRIN##_f32(a);                     \  // 否则，调用上面定义的 float 归约函数
        }
    NPY_IMPL_VEC_REDUCE_MINMAX(min, 0x7f800000)  // 实例化模板宏，生成 float 类型的最小值归约函数
    NPY_IMPL_VEC_REDUCE_MINMAX(max, 0xff800000)  // 实例化模板宏，生成 float 类型的最大值归约函数
    #undef NPY_IMPL_VEC_REDUCE_MINMAX  // 取消宏定义，清理预处理器环境
#endif // NPY_SIMD_F32

#define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, INF)                   \
    NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)     \  // 定义一个模板宏，用于生成 double 类型的向量归约函数
    {                                                             \
        npyv_f64 r = vec_##INTRIN(a, vec_sld(a, a, 8));           \  // 使用指定的向量指令进行归约操作
        return vec_extract(r, 0);                                 \  // 从归约结果向量中提取第一个元素作为结果并返回
        (void)r;                                                  \  // 使用 void 来避免编译器警告未使用变量的提示
    }                                                             \
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)    \  // 内联函数，用于处理特殊情况下的 double 归约
    {                                                             \
        npyv_b64 notnan = npyv_notnan_f64(a);                     \
        // 使用 SIMD 指令检查向量中每个元素是否为非 NaN 的浮点数
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {                \
            // 如果向量中存在 NaN 值，则返回一个预定义的 NaN 值
            const union { npy_uint64 i; double f;}                \
            pnan = {0x7ff8000000000000ull};                       \
            return pnan.f;                                        \
        }                                                         \
        // 对向量中的浮点数进行约简操作，使用给定的 INTRIN 指令集
        return npyv_reduce_##INTRIN##_f64(a);                     \
    }
// 定义一个宏，实现向量化操作以计算最小值和最大值，使用 IEEE 754 double 标准的值
NPY_IMPL_VEC_REDUCE_MINMAX(min, 0x7ff0000000000000)
// 定义一个宏，实现向量化操作以计算最小值和最大值，使用 IEEE 754 double 标准的值
NPY_IMPL_VEC_REDUCE_MINMAX(max, 0xfff0000000000000)
// 取消前面定义的 NPY_IMPL_VEC_REDUCE_MINMAX 宏
#undef NPY_IMPL_VEC_REDUCE_MINMAX

// 如果定义了 NPY_HAVE_VXE 或者 NPY_HAVE_VSX，则使用相同的向量化操作来实现最小值和最大值的降维
#define npyv_reduce_minp_f64 npyv_reduce_min_f64
#define npyv_reduce_maxp_f64 npyv_reduce_max_f64
#else
// 如果没有定义 NPY_HAVE_VXE 和 NPY_HAVE_VSX，则定义以下两个函数
NPY_FINLINE double npyv_reduce_minp_f64(npyv_f64 a)
{
    // 检查向量 a 中非 NaN 元素，若不存在，返回向量中的第一个元素
    npyv_b64 notnan = npyv_notnan_f64(a);
    if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {
        return vec_extract(a, 0);
    }
    // 将 NaN 元素替换为 IEEE 754 double 标准中的最小值，并继续进行最小值的降维操作
    a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(
                npyv_setall_u64(0x7ff0000000000000)));
    return npyv_reduce_min_f64(a);
}
// 类似于上面的函数，但是用于计算最大值的降维操作
NPY_FINLINE double npyv_reduce_maxp_f64(npyv_f64 a)
{
    npyv_b64 notnan = npyv_notnan_f64(a);
    if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {
        return vec_extract(a, 0);
    }
    a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(
                npyv_setall_u64(0xfff0000000000000)));
    return npyv_reduce_max_f64(a);
}
#endif

// 定义一个宏，实现向量化操作以进行向最近偶数的整数舍入
#define npyv_rint_f64 vec_rint
// 定义一个宏，实现向量化操作以进行向正无穷大的舍入
#define npyv_ceil_f64 vec_ceil
// 定义一个宏，实现向量化操作以进行向零的舍入
#define npyv_trunc_f64 vec_trunc
// 定义一个宏，实现向量化操作以进行向负无穷大的舍入
#define npyv_floor_f64 vec_floor

// 如果定义了 NPY_SIMD_F32，则同时定义以下几个宏，使用单精度浮点数操作
#if NPY_SIMD_F32
    #define npyv_rint_f32 vec_rint
    #define npyv_ceil_f32 vec_ceil
    #define npyv_trunc_f32 vec_trunc
    #define npyv_floor_f32 vec_floor
#endif

#endif // _NPY_SIMD_VEC_MATH_H
```