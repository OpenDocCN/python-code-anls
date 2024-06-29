# `.\numpy\numpy\_core\src\common\simd\neon\math.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_MATH_H
#define _NPY_SIMD_NEON_MATH_H

/***************************
 * Elementary
 ***************************/

// Absolute
#define npyv_abs_f32 vabsq_f32
#define npyv_abs_f64 vabsq_f64

// Square
// 定义计算单精度浮点数向量平方的函数
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return vmulq_f32(a, a); }
#if NPY_SIMD_F64
    // 定义计算双精度浮点数向量平方的函数
    NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
    { return vmulq_f64(a, a); }
#endif

// Square root
#if NPY_SIMD_F64
    // 对双精度浮点数向量进行平方根计算
    #define npyv_sqrt_f32 vsqrtq_f32
    #define npyv_sqrt_f64 vsqrtq_f64
#else
    // 基于 ARM 文档，参考 https://developer.arm.com/documentation/dui0204/j/CIHDIACI
    // 定义计算单精度浮点数向量平方根的函数
    NPY_FINLINE npyv_f32 npyv_sqrt_f32(npyv_f32 a)
    {
        // 定义常量
        const npyv_f32 zero = vdupq_n_f32(0.0f);
        const npyv_u32 pinf = vdupq_n_u32(0x7f800000);
        // 检查是否为零或无穷大
        npyv_u32 is_zero = vceqq_f32(a, zero), is_inf = vceqq_u32(vreinterpretq_u32_f32(a), pinf);
        // 防止浮点数除零错误
        npyv_f32 guard_byz = vbslq_f32(is_zero, vreinterpretq_f32_u32(pinf), a);
        // 估算 (1/√a)
        npyv_f32 rsqrte = vrsqrteq_f32(guard_byz);
        /**
         * 牛顿-拉弗森迭代法:
         *  x[n+1] = x[n] * (3-d * (x[n]*x[n]) )/2)
         * 当 x0 是应用于 d 的 VRSQRTE 的结果时，收敛到 (1/√d)。
         *
         * 注意：至少需要 3 次迭代以提高精度
         */
        rsqrte = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, rsqrte), rsqrte), rsqrte);
        rsqrte = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, rsqrte), rsqrte), rsqrte);
        rsqrte = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, rsqrte), rsqrte), rsqrte);
        // a * (1/√a)
        npyv_f32 sqrt = vmulq_f32(a, rsqrte);
        // 如果 a 是零，则返回零；如果 a 是正无穷大，则返回正无穷大
        return vbslq_f32(vorrq_u32(is_zero, is_inf), a, sqrt);
    }
#endif // NPY_SIMD_F64

// Reciprocal
// 定义计算单精度浮点数向量倒数的函数
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{
#if NPY_SIMD_F64
    // 定义计算双精度浮点数向量倒数的函数
    const npyv_f32 one = vdupq_n_f32(1.0f);
    return npyv_div_f32(one, a);
#else
    // 使用 VRECPE 应用于 d 的结果 x0，收敛到 (1/d) 的牛顿-拉弗森迭代法:
    npyv_f32 recipe = vrecpeq_f32(a);
    recipe = vmulq_f32(vrecpsq_f32(a, recipe), recipe);
    recipe = vmulq_f32(vrecpsq_f32(a, recipe), recipe);
    recipe = vmulq_f32(vrecpsq_f32(a, recipe), recipe);
    return recipe;
#endif
}
#if NPY_SIMD_F64
    // 定义计算双精度浮点数向量倒数的函数
    NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
    {
        const npyv_f64 one = vdupq_n_f64(1.0);
        return npyv_div_f64(one, a);
    }
#endif // NPY_SIMD_F64

// Maximum, natively mapping with no guarantees to handle NaN.
// 定义计算单精度浮点数向量最大值的宏，无法保证处理 NaN
#define npyv_max_f32 vmaxq_f32
#define npyv_max_f64 vmaxq_f64
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// 支持 IEEE 浮点数算术（IEC 60559）

#endif // _NPY_SIMD_NEON_MATH_H
#ifdef NPY_HAVE_ASIMD
    // 如果使用 ASIMD 指令集，定义 npyv_maxp_f32 为 vmaxnmq_f32 函数
    #define npyv_maxp_f32 vmaxnmq_f32
#else
    // 如果未使用 ASIMD 指令集，定义 npyv_maxp_f32 函数
    NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
    {
        // 使用 vceqq_f32 函数比较 a 是否为 NaN，结果存储在 nn_a 中
        npyv_u32 nn_a = vceqq_f32(a, a);
        // 使用 vceqq_f32 函数比较 b 是否为 NaN，结果存储在 nn_b 中
        npyv_u32 nn_b = vceqq_f32(b, b);
        // 返回根据 nn_a 和 nn_b 条件选择的最大值向量
        return vmaxq_f32(vbslq_f32(nn_a, a, b), vbslq_f32(nn_b, b, a));
    }
#endif

// 定义 npyv_maxn_f32 函数为 vmaxq_f32 函数
// 最大化函数，传播 NaN
// 如果任意对应的元素是 NaN，则设置 NaN
#define npyv_maxn_f32 vmaxq_f32

#if NPY_SIMD_F64
    // 如果支持双精度 SIMD 计算
    // 定义 npyv_maxp_f64 函数为 vmaxnmq_f64 函数
    #define npyv_maxp_f64 vmaxnmq_f64
    // 定义 npyv_maxn_f64 函数为 vmaxq_f64 函数
    #define npyv_maxn_f64 vmaxq_f64
#endif // NPY_SIMD_F64

// 最大化函数，整数操作
#define npyv_max_u8 vmaxq_u8
#define npyv_max_s8 vmaxq_s8
#define npyv_max_u16 vmaxq_u16
#define npyv_max_s16 vmaxq_s16
#define npyv_max_u32 vmaxq_u32
#define npyv_max_s32 vmaxq_s32

// 定义 npyv_max_u64 函数
// 返回 a 和 b 中每个元素的最大值
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    // 使用 vbslq_u64 函数根据 npyv_cmpgt_u64(a, b) 的结果选择 a 或 b 中的元素
    return vbslq_u64(npyv_cmpgt_u64(a, b), a, b);
}

// 定义 npyv_max_s64 函数
// 返回 a 和 b 中每个元素的最大值
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    // 使用 vbslq_s64 函数根据 npyv_cmpgt_s64(a, b) 的结果选择 a 或 b 中的元素
    return vbslq_s64(npyv_cmpgt_s64(a, b), a, b);
}

// 最小化函数，本地映射，不保证处理 NaN
#define npyv_min_f32 vminq_f32
#define npyv_min_f64 vminq_f64

// 最小化函数，支持 IEEE 浮点运算 (IEC 60559)
// - 如果两个向量中的一个包含 NaN，则设置另一个向量中相应的元素
// - 只有当两个对应元素都是 NaN 时，才设置 NaN
#ifdef NPY_HAVE_ASIMD
    // 如果使用 ASIMD 指令集，定义 npyv_minp_f32 为 vminnmq_f32 函数
    #define npyv_minp_f32 vminnmq_f32
#else
    // 如果未使用 ASIMD 指令集，定义 npyv_minp_f32 函数
    NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
    {
        // 使用 vceqq_f32 函数比较 a 是否为 NaN，结果存储在 nn_a 中
        npyv_u32 nn_a = vceqq_f32(a, a);
        // 使用 vceqq_f32 函数比较 b 是否为 NaN，结果存储在 nn_b 中
        npyv_u32 nn_b = vceqq_f32(b, b);
        // 返回根据 nn_a 和 nn_b 条件选择的最小值向量
        return vminq_f32(vbslq_f32(nn_a, a, b), vbslq_f32(nn_b, b, a));
    }
#endif

// 定义 npyv_minn_f32 函数为 vminq_f32 函数
// 最小化函数，传播 NaN
// 如果任意对应的元素是 NaN，则设置 NaN
#define npyv_minn_f32 vminq_f32

#if NPY_SIMD_F64
    // 如果支持双精度 SIMD 计算
    // 定义 npyv_minp_f64 函数为 vminnmq_f64 函数
    #define npyv_minp_f64 vminnmq_f64
    // 定义 npyv_minn_f64 函数为 vminq_f64 函数
    #define npyv_minn_f64 vminq_f64
#endif // NPY_SIMD_F64

// 最小化函数，整数操作
#define npyv_min_u8 vminq_u8
#define npyv_min_s8 vminq_s8
#define npyv_min_u16 vminq_u16
#define npyv_min_s16 vminq_s16
#define npyv_min_u32 vminq_u32
#define npyv_min_s32 vminq_s32

// 定义 npyv_min_u64 函数
// 返回 a 和 b 中每个元素的最小值
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    // 使用 vbslq_u64 函数根据 npyv_cmplt_u64(a, b) 的结果选择 a 或 b 中的元素
    return vbslq_u64(npyv_cmplt_u64(a, b), a, b);
}

// 定义 npyv_min_s64 函数
// 返回 a 和 b 中每个元素的最小值
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    // 使用 vbslq_s64 函数根据 npyv_cmplt_s64(a, b) 的结果选择 a 或 b 中的元素
    return vbslq_s64(npyv_cmplt_s64(a, b), a, b);
}

// 减少所有数据类型的最小/最大值
#if NPY_SIMD_F64
    // 如果支持双精度 SIMD 计算
    #define npyv_reduce_max_u8 vmaxvq_u8
    #define npyv_reduce_max_s8 vmaxvq_s8
    #define npyv_reduce_max_u16 vmaxvq_u16
    #define npyv_reduce_max_s16 vmaxvq_s16
    #define npyv_reduce_max_u32 vmaxvq_u32
    #define npyv_reduce_max_s32 vmaxvq_s32

    #define npyv_reduce_max_f32 vmaxvq_f32
    #define npyv_reduce_max_f64 vmaxvq_f64
    #define npyv_reduce_maxn_f32 vmaxvq_f32
    #define npyv_reduce_maxn_f64 vmaxvq_f64
    #define npyv_reduce_maxp_f32 vmaxnmvq_f32
#endif
    #define npyv_reduce_maxp_f64 vmaxnmvq_f64
    
    定义了一个宏 `npyv_reduce_maxp_f64`，用于表示将向量中的浮点数类型的元素进行最大值约简。
    
    
    #define npyv_reduce_min_u8 vminvq_u8
    
    定义了一个宏 `npyv_reduce_min_u8`，用于表示将向量中的无符号8位整数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_min_s8 vminvq_s8
    
    定义了一个宏 `npyv_reduce_min_s8`，用于表示将向量中的有符号8位整数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_min_u16 vminvq_u16
    
    定义了一个宏 `npyv_reduce_min_u16`，用于表示将向量中的无符号16位整数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_min_s16 vminvq_s16
    
    定义了一个宏 `npyv_reduce_min_s16`，用于表示将向量中的有符号16位整数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_min_u32 vminvq_u32
    
    定义了一个宏 `npyv_reduce_min_u32`，用于表示将向量中的无符号32位整数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_min_s32 vminvq_s32
    
    定义了一个宏 `npyv_reduce_min_s32`，用于表示将向量中的有符号32位整数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_min_f32 vminvq_f32
    
    定义了一个宏 `npyv_reduce_min_f32`，用于表示将向量中的单精度浮点数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_min_f64 vminvq_f64
    
    定义了一个宏 `npyv_reduce_min_f64`，用于表示将向量中的双精度浮点数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_minn_f32 vminvq_f32
    
    定义了一个宏 `npyv_reduce_minn_f32`，用于表示将向量中的单精度浮点数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_minn_f64 vminvq_f64
    
    定义了一个宏 `npyv_reduce_minn_f64`，用于表示将向量中的双精度浮点数类型的元素进行最小值约简。
    
    
    #define npyv_reduce_minp_f32 vminnmvq_f32
    
    定义了一个宏 `npyv_reduce_minp_f32`，用于表示将向量中的单精度浮点数类型的元素进行带负最小值约简。
    
    
    #define npyv_reduce_minp_f64 vminnmvq_f64
    
    定义了一个宏 `npyv_reduce_minp_f64`，用于表示将向量中的双精度浮点数类型的元素进行带负最小值约简。
#else
    // 定义 NEON 求最小值和最大值的宏函数
    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX)                            \
        // 实现 NEON 向量的最小值或最大值求解函数
        NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)             \
        {                                                                              \
            // 使用 NEON 指令求取向量的最小值或最大值
            STYPE##x8_t r = vp##INTRIN##_##SFX(vget_low_##SFX(a), vget_high_##SFX(a)); \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
            // 返回结果向量中的第一个元素作为标量结果
            return (npy_##STYPE)vget_lane_##SFX(r, 0);                                 \
        }
    // 实例化 uint8 类型的最小值和最大值求解函数
    NPY_IMPL_NEON_REDUCE_MINMAX(min, uint8, u8)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, uint8, u8)
    // 实例化 int8 类型的最小值和最大值求解函数
    NPY_IMPL_NEON_REDUCE_MINMAX(min, int8, s8)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, int8, s8)
    // 取消宏定义 NPY_IMPL_NEON_REDUCE_MINMAX
    #undef NPY_IMPL_NEON_REDUCE_MINMAX

    // 定义 NEON 求最小值和最大值的宏函数
    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX)                            \
        // 实现 NEON 向量的最小值或最大值求解函数
        NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)             \
        {                                                                              \
            // 使用 NEON 指令求取向量的最小值或最大值
            STYPE##x4_t r = vp##INTRIN##_##SFX(vget_low_##SFX(a), vget_high_##SFX(a)); \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
            // 返回结果向量中的第一个元素作为标量结果
            return (npy_##STYPE)vget_lane_##SFX(r, 0);                                 \
        }
    // 实例化 uint16 类型的最小值和最大值求解函数
    NPY_IMPL_NEON_REDUCE_MINMAX(min, uint16, u16)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, uint16, u16)
    // 实例化 int16 类型的最小值和最大值求解函数
    NPY_IMPL_NEON_REDUCE_MINMAX(min, int16, s16)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, int16, s16)
    // 取消宏定义 NPY_IMPL_NEON_REDUCE_MINMAX
    #undef NPY_IMPL_NEON_REDUCE_MINMAX

    // 定义 NEON 求最小值和最大值的宏函数
    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX)                            \
        // 实现 NEON 向量的最小值或最大值求解函数
        NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)             \
        {                                                                              \
            // 使用 NEON 指令求取向量的最小值或最大值
            STYPE##x2_t r = vp##INTRIN##_##SFX(vget_low_##SFX(a), vget_high_##SFX(a)); \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
            // 返回结果向量中的第一个元素作为标量结果
            return (npy_##STYPE)vget_lane_##SFX(r, 0);                                 \
        }
    // 实例化 uint32 类型的最小值和最大值求解函数
    NPY_IMPL_NEON_REDUCE_MINMAX(min, uint32, u32)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, uint32, u32)
    // 实例化 int32 类型的最小值和最大值求解函数
    NPY_IMPL_NEON_REDUCE_MINMAX(min, int32, s32)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, int32, s32)
    // 取消宏定义 NPY_IMPL_NEON_REDUCE_MINMAX
    #undef NPY_IMPL_NEON_REDUCE_MINMAX
    // 定义宏 NPY_IMPL_NEON_REDUCE_MINMAX，用于实现 NEON 指令集的最小值和最大值归约函数
    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, INF)                             \
        // 定义内联函数，用于将一个 npyv_f32 向量归约为一个 float 最小值或最大值                    \
        NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                 \
        {                                                                        \
            // 使用 NEON 指令将输入向量 a 的低位和高位部分合并，得到一个 float32x2_t 结果向量 \
            float32x2_t r = vp##INTRIN##_f32(vget_low_f32(a), vget_high_f32(a));\
            // 使用 NEON 指令再次对合并结果向量 r 进行 INTRIN 运算，得到最终归约结果向量       \
            r = vp##INTRIN##_f32(r, r);                                          \
            // 返回最终结果向量的第一个元素，即最小值或最大值                                 \
            return vget_lane_f32(r, 0);                                          \
        }                                                                        \
        // 定义内联函数，用于将一个 npyv_f32 向量归约为一个 float 最小值或最大值，忽略 NaN 值     \
        NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)                \
        {                                                                        \
            // 获取非 NaN 元素的掩码                                               \
            npyv_b32 notnan = npyv_notnan_f32(a);                                \
            // 如果向量中所有元素均为 NaN，则直接返回第一个元素的值                        \
            if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                           \
                return vgetq_lane_f32(a, 0);                                     \
            }                                                                    \
            // 使用掩码选择非 NaN 元素，将 NaN 元素替换为 INF （表示无穷大）              \
            a = npyv_select_f32(notnan, a,                                       \
                    npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));             \
            // 调用相应的归约函数处理已处理过 NaN 的向量 a，返回最终的最小值或最大值             \
            return npyv_reduce_##INTRIN##_f32(a);                                \
        }                                                                        \
        // 定义内联函数，用于将一个 npyv_f32 向量归约为一个 float 最小值或最大值，无论是否有 NaN 值 \
        NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)                \
        {                                                                        \
            // 直接调用归约函数处理向量 a，返回最终的最小值或最大值                           \
            return npyv_reduce_##INTRIN##_f32(a);                                \
        }
    // 取消宏定义 NPY_IMPL_NEON_REDUCE_MINMAX 的定义
    #undef NPY_IMPL_NEON_REDUCE_MINMAX
#ifdef NPY_SIMD_F64
    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX, OP)       \
        NPY_FINLINE STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \
        {                                                             \
            // 提取 NEON 向量 a 的低部分并将其转换为 STYPE 类型
            STYPE al = (STYPE)vget_low_##SFX(a);                      \
            // 提取 NEON 向量 a 的高部分并将其转换为 STYPE 类型
            STYPE ah = (STYPE)vget_high_##SFX(a);                     \
            // 返回 al 和 ah 中较大或较小的值，根据 OP 参数指定的比较操作符
            return al OP ah ? al : ah;                                \
        }
#endif // NPY_SIMD_F64

// 定义 NEON 实现的最大值和最小值归约函数
#define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX, OP)       \
    NPY_FINLINE STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \
    {                                                             \
        // 提取 NEON 向量 a 的低部分并将其转换为 STYPE 类型
        STYPE al = (STYPE)vget_low_##SFX(a);                      \
        // 提取 NEON 向量 a 的高部分并将其转换为 STYPE 类型
        STYPE ah = (STYPE)vget_high_##SFX(a);                     \
        // 返回 al 和 ah 中较大或较小的值，根据 OP 参数指定的比较操作符
        return al OP ah ? al : ah;                                \
    }

// 调用宏定义来生成具体的函数实现，用于不同的数据类型和操作符
NPY_IMPL_NEON_REDUCE_MINMAX(max, npy_uint64, u64, >)
NPY_IMPL_NEON_REDUCE_MINMAX(max, npy_int64,  s64, >)
NPY_IMPL_NEON_REDUCE_MINMAX(min, npy_uint64, u64, <)
NPY_IMPL_NEON_REDUCE_MINMAX(min, npy_int64,  s64, <)
#undef NPY_IMPL_NEON_REDUCE_MINMAX

// round to nearest integer even
NPY_FINLINE npyv_f32 npyv_rint_f32(npyv_f32 a)
{
#ifdef NPY_HAVE_ASIMD
    // 使用 NEON 指令 vrndnq_f32 对向量 a 进行舍入到最近的偶数整数
    return vrndnq_f32(a);
#else
    // ARMv7 NEON 仅支持浮点数到整数的截断转换。
    // 使用一个魔术技巧，添加 1.5 * 2^23 来实现舍入到最近的偶数整数，
    // 然后减去这个魔术数以得到整数部分。

    // 创建一个常数向量，内容为 -0.0f 的无符号整数表示
    const npyv_u32 szero = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
    // 计算向量 a 的符号位掩码
    const npyv_u32 sign_mask = vandq_u32(vreinterpretq_u32_f32(a), szero);
    // 创建一个常数向量，内容为 2^23 的浮点数表示
    const npyv_f32 two_power_23 = vdupq_n_f32(8388608.0); // 2^23
    // 创建一个常数向量，内容为 1.5 * 2^23 的浮点数表示
    const npyv_f32 two_power_23h = vdupq_n_f32(12582912.0f); // 1.5 * 2^23
    // 创建一个向量，用于消除 NaN 值，避免无效的浮点错误
    npyv_u32 nnan_mask = vceqq_f32(a, a);
    // 计算向量 a 的绝对值
    npyv_f32 abs_x = vabsq_f32(vreinterpretq_f32_u32(vandq_u32(nnan_mask, vreinterpretq_u32_f32(a))));
    // 执行舍入操作，通过添加魔术数 1.5 * 2^23
    npyv_f32 round = vsubq_f32(vaddq_f32(two_power_23h, abs_x), two_power_23h);
    // 使用符号掩码来进行符号位复制
    round = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(round), sign_mask ));
    // 如果 |a| >= 2^23 或者 a 是 NaN，则返回 a；否则返回 round
    npyv_u32 mask = vcleq_f32(abs_x, two_power_23);
             mask = vandq_u32(mask, nnan_mask);
    return vbslq_f32(mask, round, a);
#endif
}

// 如果 NPY_SIMD_F64 定义了，则使用 vrndnq_f64 来定义 npyv_rint_f64 函数
#if NPY_SIMD_F64
    #define npyv_rint_f64 vrndnq_f64
#endif // NPY_SIMD_F64

// 如果 NPY_HAVE_ASIMD 定义了，则使用 vrndpq_f32 来定义 npyv_ceil_f32 函数
#ifdef NPY_HAVE_ASIMD
    #define npyv_ceil_f32 vrndpq_f32
#else
    // 否则，定义一个函数 npyv_ceil_f32，实现向上取整操作
    NPY_FINLINE npyv_f32 npyv_ceil_f32(npyv_f32 a)
    {
        // 创建常量 one，其值为单精度浮点数 1.0 的转换后的无符号整数表示
        const npyv_u32 one = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
        // 创建常量 szero，其值为单精度浮点数 -0.0 的转换后的无符号整数表示
        const npyv_u32 szero = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
        // 创建 sign_mask，通过将浮点数 a 转换为无符号整数，并与 szero 相与得到
        const npyv_u32 sign_mask = vandq_u32(vreinterpretq_u32_f32(a), szero);
        // 创建常量 two_power_23，其值为单精度浮点数 8388608.0 的复制
        const npyv_f32 two_power_23 = vdupq_n_f32(8388608.0); // 2^23
        // 创建常量 two_power_23h，其值为单精度浮点数 12582912.0 的复制
        const npyv_f32 two_power_23h = vdupq_n_f32(12582912.0f); // 1.5 * 2^23
        
        // 创建 nnan_mask，检查 a 是否等于自身（排除 NaN 值）
        npyv_u32 nnan_mask = vceqq_f32(a, a);
        // 将 nnan_mask 与 a 进行位与操作，得到 x，用于消除 NaN 值以避免无效的浮点数错误
        npyv_f32 x = vreinterpretq_f32_u32(vandq_u32(nnan_mask, vreinterpretq_u32_f32(a)));
        
        // 计算 x 的绝对值 abs_x
        npyv_f32 abs_x = vabsq_f32(x);
        
        // 使用魔数 1.5 * 2^23 进行四舍五入
        npyv_f32 round = vsubq_f32(vaddq_f32(two_power_23h, abs_x), two_power_23h);
        
        // 将 round 的符号位设置为与 a 相同的符号
        round = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(round), sign_mask));
        
        // 对 round 进行向上取整操作，考虑了有符号零值的情况
        npyv_f32 ceil = vaddq_f32(round, vreinterpretq_f32_u32(
            vandq_u32(vcltq_f32(round, x), one))
        );
        
        // 将 ceil 的符号位设置为与 a 相同的符号
        ceil = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(ceil), sign_mask));
        
        // 如果 |a| >= 2^23 或者 a 为 NaN，则返回 a；否则返回 ceil
        npyv_u32 mask = vcleq_f32(abs_x, two_power_23);
        mask = vandq_u32(mask, nnan_mask);
        
        return vbslq_f32(mask, ceil, a);
    }
#endif
#if NPY_SIMD_F64
    #define npyv_ceil_f64 vrndpq_f64
#endif // NPY_SIMD_F64

// trunc
#ifdef NPY_HAVE_ASIMD
    #define npyv_trunc_f32 vrndq_f32
#else
    // 定义在没有 ASIMD 支持时，用于截断操作的函数 npyv_trunc_f32
    NPY_FINLINE npyv_f32 npyv_trunc_f32(npyv_f32 a)
    {
        // 定义常量 max_int 为 0x7fffffff
        const npyv_s32 max_int = vdupq_n_s32(0x7fffffff);
        // 定义常量 exp_mask 为 0xff000000
        const npyv_u32 exp_mask = vdupq_n_u32(0xff000000);
        // 定义常量 szero 为 -0.0f 的整数表示
        const npyv_s32 szero = vreinterpretq_s32_f32(vdupq_n_f32(-0.0f));
        // 创建 sign_mask，用于提取输入参数 a 的符号位
        const npyv_u32 sign_mask = vandq_u32(
            vreinterpretq_u32_f32(a), vreinterpretq_u32_s32(szero));

        // 创建 nfinite_mask，用于检测 a 是否为有限值
        npyv_u32 nfinite_mask = vshlq_n_u32(vreinterpretq_u32_f32(a), 1);
                 nfinite_mask = vandq_u32(nfinite_mask, exp_mask);
                 nfinite_mask = vceqq_u32(nfinite_mask, exp_mask);
        // 消除 NaN 和 inf，避免无效的浮点错误
        npyv_f32 x = vreinterpretq_f32_u32(
            veorq_u32(nfinite_mask, vreinterpretq_u32_f32(a)));
        /**
         * 在 armv7 上，vcvtq.f32 处理特殊情况如下：
         *  NaN 返回 0
         * +inf 或超出范围 返回 0x80000000(-0.0f)
         * -inf 或超出范围 返回 0x7fffffff(nan)
         */
        // 将 x 转为整数类型
        npyv_s32 trunci = vcvtq_s32_f32(x);
        // 将整数类型再转回浮点数类型
        npyv_f32 trunc = vcvtq_f32_s32(trunci);
        // 根据符号位，保留有符号零，例如 -0.5 -> -0.0
        trunc = vreinterpretq_f32_u32(
            vorrq_u32(vreinterpretq_u32_f32(trunc), sign_mask));
        // 如果溢出，则返回原始参数 a
        npyv_u32 overflow_mask = vorrq_u32(
            vceqq_s32(trunci, szero), vceqq_s32(trunci, max_int)
        );
        // 如果溢出或非有限值，则返回原始参数 a，否则返回截断后的值
        return vbslq_f32(vorrq_u32(nfinite_mask, overflow_mask), a, trunc);
   }
#endif
#if NPY_SIMD_F64
    #define npyv_trunc_f64 vrndq_f64
#endif // NPY_SIMD_F64

// floor
#ifdef NPY_HAVE_ASIMD
    #define npyv_floor_f32 vrndmq_f32
#else
    // 在没有 ASIMD 支持时，定义用于向下取整操作的函数 npyv_floor_f32
    NPY_FINLINE npyv_f32 npyv_floor_f32(npyv_f32 a)
    {
        // 创建一个常量，其值为单精度浮点数 1.0 对应的无符号整数形式
        const npyv_u32 one = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
        // 创建一个常量，其值为单精度浮点数 -0.0 对应的无符号整数形式
        const npyv_u32 szero = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
        // 通过按位与操作，生成一个用于标记符号位的掩码
        const npyv_u32 sign_mask = vandq_u32(vreinterpretq_u32_f32(a), szero);
        // 创建一个常量，其值为单精度浮点数 2^23 对应的向量形式
        const npyv_f32 two_power_23 = vdupq_n_f32(8388608.0); // 2^23
        // 创建一个常量，其值为单精度浮点数 1.5 * 2^23 对应的向量形式
        const npyv_f32 two_power_23h = vdupq_n_f32(12582912.0f); // 1.5 * 2^23
    
        // 创建一个掩码，用于消除 NaN 值，以避免无效的浮点错误
        npyv_u32 nnan_mask = vceqq_f32(a, a);
        // 通过按位与操作，提取绝对值形式的浮点数向量
        npyv_f32 x = vreinterpretq_f32_u32(vandq_u32(nnan_mask, vreinterpretq_u32_f32(a)));
    
        // 计算绝对值的浮点数向量
        npyv_f32 abs_x = vabsq_f32(x);
        // 通过加上魔数 1.5 * 2^23 来进行四舍五入
        npyv_f32 round = vsubq_f32(vaddq_f32(two_power_23h, abs_x), two_power_23h);
    
        // 执行拷贝符号操作
        round = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(round), sign_mask));
    
        // 计算向下取整的浮点数向量
        npyv_f32 floor = vsubq_f32(round, vreinterpretq_f32_u32(
            vandq_u32(vcgtq_f32(round, x), one)
        ));
    
        // 尊重带符号零的特性
        floor = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(floor), sign_mask));
    
        // 如果 |a| >= 2^23 或者 a 是 NaN，则返回 a；否则返回 floor
        npyv_u32 mask = vcleq_f32(abs_x, two_power_23);
                 mask = vandq_u32(mask, nnan_mask);
        return vbslq_f32(mask, floor, a);
    }
#if defined(NPY_HAVE_ASIMD)
    // 如果定义了 NPY_HAVE_ASIMD 宏，则执行以下代码块
    #if NPY_SIMD_F64
        // 如果定义了 NPY_SIMD_F64 宏，则定义 npyv_floor_f64 宏为 vrndmq_f64
        #define npyv_floor_f64 vrndmq_f64
    #endif // NPY_SIMD_F64
#endif // NPY_HAVE_ASIMD
// 结束对 NPY_HAVE_ASIMD 宏的条件编译

#endif // _NPY_SIMD_NEON_MATH_H
// 结束对 _NPY_SIMD_NEON_MATH_H 头文件的条件编译
```