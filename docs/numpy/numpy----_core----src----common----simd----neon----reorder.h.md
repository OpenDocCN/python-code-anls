# `.\numpy\numpy\_core\src\common\simd\neon\reorder.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_REORDER_H
#define _NPY_SIMD_NEON_REORDER_H

// 定义 __aarch64__ 情况下的向量操作宏，用于合并两个向量的低部分元素
#ifdef __aarch64__
    #define npyv_combinel_u8(A, B)  vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(A), vreinterpretq_u64_u8(B)))
    #define npyv_combinel_s8(A, B)  vreinterpretq_s8_u64(vzip1q_u64(vreinterpretq_u64_s8(A), vreinterpretq_u64_s8(B)))
    #define npyv_combinel_u16(A, B) vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u16(A), vreinterpretq_u64_u16(B)))
    #define npyv_combinel_s16(A, B) vreinterpretq_s16_u64(vzip1q_u64(vreinterpretq_u64_s16(A), vreinterpretq_u64_s16(B)))
    #define npyv_combinel_u32(A, B) vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(A), vreinterpretq_u64_u32(B)))
    #define npyv_combinel_s32(A, B) vreinterpretq_s32_u64(vzip1q_u64(vreinterpretq_u64_s32(A), vreinterpretq_u64_s32(B)))
    #define npyv_combinel_u64       vzip1q_u64
    #define npyv_combinel_s64       vzip1q_s64
    #define npyv_combinel_f32(A, B) vreinterpretq_f32_u64(vzip1q_u64(vreinterpretq_u64_f32(A), vreinterpretq_u64_f32(B)))
    #define npyv_combinel_f64       vzip1q_f64
#else
    // 定义非 __aarch64__ 情况下的向量操作宏，用于合并两个向量的低部分元素
    #define npyv_combinel_u8(A, B)  vcombine_u8(vget_low_u8(A), vget_low_u8(B))
    #define npyv_combinel_s8(A, B)  vcombine_s8(vget_low_s8(A), vget_low_s8(B))
    #define npyv_combinel_u16(A, B) vcombine_u16(vget_low_u16(A), vget_low_u16(B))
    #define npyv_combinel_s16(A, B) vcombine_s16(vget_low_s16(A), vget_low_s16(B))
    #define npyv_combinel_u32(A, B) vcombine_u32(vget_low_u32(A), vget_low_u32(B))
    #define npyv_combinel_s32(A, B) vcombine_s32(vget_low_s32(A), vget_low_s32(B))
    #define npyv_combinel_u64(A, B) vcombine_u64(vget_low_u64(A), vget_low_u64(B))
    #define npyv_combinel_s64(A, B) vcombine_s64(vget_low_s64(A), vget_low_s64(B))
    #define npyv_combinel_f32(A, B) vcombine_f32(vget_low_f32(A), vget_low_f32(B))
#endif

// 定义 __aarch64__ 情况下的向量操作宏，用于合并两个向量的高部分元素
#ifdef __aarch64__
    #define npyv_combineh_u8(A, B)  vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(A), vreinterpretq_u64_u8(B)))
    #define npyv_combineh_s8(A, B)  vreinterpretq_s8_u64(vzip2q_u64(vreinterpretq_u64_s8(A), vreinterpretq_u64_s8(B)))
    #define npyv_combineh_u16(A, B) vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u16(A), vreinterpretq_u64_u16(B)))
    #define npyv_combineh_s16(A, B) vreinterpretq_s16_u64(vzip2q_u64(vreinterpretq_u64_s16(A), vreinterpretq_u64_s16(B)))
    #define npyv_combineh_u32(A, B) vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(A), vreinterpretq_u64_u32(B)))
    #define npyv_combineh_s32(A, B) vreinterpretq_s32_u64(vzip2q_u64(vreinterpretq_u64_s32(A), vreinterpretq_u64_s32(B)))
    #define npyv_combineh_u64       vzip2q_u64
    #define npyv_combineh_s64       vzip2q_s64
    #define npyv_combineh_f32(A, B) vreinterpretq_f32_u64(vzip2q_u64(vreinterpretq_u64_f32(A), vreinterpretq_u64_f32(B)))
    #define npyv_combineh_f64       vzip2q_f64
#else
    // 定义非 __aarch64__ 情况下的向量操作宏，用于合并两个向量的高部分元素
#endif

#endif
    # 定义宏函数 `npyv_combineh_u8(A, B)`，将两个 uint8x8_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_u8(A, B)  vcombine_u8(vget_high_u8(A), vget_high_u8(B))
    
    # 定义宏函数 `npyv_combineh_s8(A, B)`，将两个 int8x8_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_s8(A, B)  vcombine_s8(vget_high_s8(A), vget_high_s8(B))
    
    # 定义宏函数 `npyv_combineh_u16(A, B)`，将两个 uint16x4_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_u16(A, B) vcombine_u16(vget_high_u16(A), vget_high_u16(B))
    
    # 定义宏函数 `npyv_combineh_s16(A, B)`，将两个 int16x4_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_s16(A, B) vcombine_s16(vget_high_s16(A), vget_high_s16(B))
    
    # 定义宏函数 `npyv_combineh_u32(A, B)`，将两个 uint32x2_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_u32(A, B) vcombine_u32(vget_high_u32(A), vget_high_u32(B))
    
    # 定义宏函数 `npyv_combineh_s32(A, B)`，将两个 int32x2_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_s32(A, B) vcombine_s32(vget_high_s32(A), vget_high_s32(B))
    
    # 定义宏函数 `npyv_combineh_u64(A, B)`，将两个 uint64x1_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_u64(A, B) vcombine_u64(vget_high_u64(A), vget_high_u64(B))
    
    # 定义宏函数 `npyv_combineh_s64(A, B)`，将两个 int64x1_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_s64(A, B) vcombine_s64(vget_high_s64(A), vget_high_s64(B))
    
    # 定义宏函数 `npyv_combineh_f32(A, B)`，将两个 float32x2_t 类型的寄存器 A 和 B 的高位元素组合成一个新的寄存器
    #define npyv_combineh_f32(A, B) vcombine_f32(vget_high_f32(A), vget_high_f32(B))
// 定义宏函数 NPYV_IMPL_NEON_COMBINE，用于将两个给定类型 T_VEC 的向量合并成一个类型为 T_VEC##x2 的结构体
#define NPYV_IMPL_NEON_COMBINE(T_VEC, SFX)                     \
    // 内联函数，将两个类型为 T_VEC 的向量 a 和 b 合并
    NPY_FINLINE T_VEC##x2 npyv_combine_##SFX(T_VEC a, T_VEC b) \
    {                                                          \
        // 创建类型为 T_VEC##x2 的变量 r，其中 val[0] 是通过 npyv_combinel_SFX 函数合并 a 和 b 的结果
        r.val[0] = NPY_CAT(npyv_combinel_, SFX)(a, b);         \
        // val[1] 是通过 npyv_combineh_SFX 函数合并 a 和 b 的结果
        r.val[1] = NPY_CAT(npyv_combineh_, SFX)(a, b);         \
        // 返回合并后的结果结构体 r
        return r;                                              \
    }

// 根据给定类型和后缀 SFX 实例化 NPYV_IMPL_NEON_COMBINE 宏
NPYV_IMPL_NEON_COMBINE(npyv_u8,  u8)
NPYV_IMPL_NEON_COMBINE(npyv_s8,  s8)
NPYV_IMPL_NEON_COMBINE(npyv_u16, u16)
NPYV_IMPL_NEON_COMBINE(npyv_s16, s16)
NPYV_IMPL_NEON_COMBINE(npyv_u32, u32)
NPYV_IMPL_NEON_COMBINE(npyv_s32, s32)
NPYV_IMPL_NEON_COMBINE(npyv_u64, u64)
NPYV_IMPL_NEON_COMBINE(npyv_s64, s64)
NPYV_IMPL_NEON_COMBINE(npyv_f32, f32)

// 根据宏 __aarch64__ 的定义条件编译以下部分代码块
#ifdef __aarch64__
    // 定义宏函数 NPYV_IMPL_NEON_ZIP，用于对给定类型 T_VEC 的向量执行交织（interleave）和解交织（deinterleave）操作
    #define NPYV_IMPL_NEON_ZIP(T_VEC, SFX)                       \
        // 内联函数，将两个类型为 T_VEC 的向量 a 和 b 进行交织操作
        NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)   \
        {                                                        \
            // 创建类型为 T_VEC##x2 的变量 r，其中 val[0] 是通过 vzip1q_SFX(a, b) 进行的交织操作
            r.val[0] = vzip1q_##SFX(a, b);                       \
            // val[1] 是通过 vzip2q_SFX(a, b) 进行的交织操作
            r.val[1] = vzip2q_##SFX(a, b);                       \
            // 返回交织后的结果结构体 r
            return r;                                            \
        }                                                        \
        // 内联函数，将两个类型为 T_VEC 的向量 a 和 b 进行解交织操作
        NPY_FINLINE T_VEC##x2 npyv_unzip_##SFX(T_VEC a, T_VEC b) \
        {                                                        \
            // 创建类型为 T_VEC##x2 的变量 r，其中 val[0] 是通过 vuzp1q_SFX(a, b) 进行的解交织操作
            r.val[0] = vuzp1q_##SFX(a, b);                       \
            // val[1] 是通过 vuzp2q_SFX(a, b) 进行的解交织操作
            r.val[1] = vuzp2q_##SFX(a, b);                       \
            // 返回解交织后的结果结构体 r
            return r;                                            \
        }
// 若未定义 __aarch64__ 宏，则编译以下代码块
#else
    // 定义宏函数 NPYV_IMPL_NEON_ZIP，用于对给定类型 T_VEC 的向量执行交织（interleave）和解交织（deinterleave）操作
    #define NPYV_IMPL_NEON_ZIP(T_VEC, SFX)                       \
        // 内联函数，将两个类型为 T_VEC 的向量 a 和 b 进行交织操作
        NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)   \
        { return vzipq_##SFX(a, b); }                            \
        // 内联函数，将两个类型为 T_VEC 的向量 a 和 b 进行解交织操作
        NPY_FINLINE T_VEC##x2 npyv_unzip_##SFX(T_VEC a, T_VEC b) \
        { return vuzpq_##SFX(a, b); }
#endif

// 根据给定类型和后缀 SFX 实例化 NPYV_IMPL_NEON_ZIP 宏
NPYV_IMPL_NEON_ZIP(npyv_u8,  u8)
NPYV_IMPL_NEON_ZIP(npyv_s8,  s8)
NPYV_IMPL_NEON_ZIP(npyv_u16, u16)
NPYV_IMPL_NEON_ZIP(npyv_s16, s16)
NPYV_IMPL_NEON_ZIP(npyv_u32, u32)
NPYV_IMPL_NEON_ZIP(npyv_s32, s32)
NPYV_IMPL_NEON_ZIP(npyv_f32, f32)

// 定义一系列宏函数，将不同类型的向量直接映射到其对应的合并和解交织函数
#define npyv_zip_u64 npyv_combine_u64
#define npyv_zip_s64 npyv_combine_s64
#define npyv_zip_f64 npyv_combine_f64
#define npyv_unzip_u64 npyv_combine_u64
#define npyv_unzip_s64 npyv_combine_s64
#define npyv_unzip_f64 npyv_combine_f64

// 定义一系列宏函数，用于反转每个 64 位通道中的元素顺序
#define npyv_rev64_u8  vrev64q_u8
#define npyv_rev64_s8  vrev64q_s8
#define npyv_rev64_u16 vrev64q_u16
#define npyv_rev64_s16 vrev64q_s16
#define npyv_rev64_u32 vrev64q_u32
#define npyv_rev64_s32 vrev64q_s32
#define npyv_rev64_f32 vrev64q_f32
// 定义宏 npyv_rev64_f32 用于反转 NEON 寄存器中 64 位浮点数元素的顺序

// 根据不同的编译器预处理指令，定义宏 npyv_permi128_u32，用于对 NEON 寄存器中的 128 位整数型数据进行按指定索引重新排列
#ifdef __clang__
    #define npyv_permi128_u32(A, E0, E1, E2, E3) \
        __builtin_shufflevector(A, A, E0, E1, E2, E3)
// 对于 Clang 编译器，使用 __builtin_shufflevector 进行元素重新排列
#elif defined(__GNUC__)
    #define npyv_permi128_u32(A, E0, E1, E2, E3) \
        __builtin_shuffle(A, npyv_set_u32(E0, E1, E2, E3))
// 对于 GCC 编译器，使用 __builtin_shuffle 结合 npyv_set_u32 宏进行元素重新排列
#else
    #define npyv_permi128_u32(A, E0, E1, E2, E3)          \
        npyv_set_u32(                                     \
            vgetq_lane_u32(A, E0), vgetq_lane_u32(A, E1), \
            vgetq_lane_u32(A, E2), vgetq_lane_u32(A, E3)  \
        )
// 对于其他编译器，使用 vgetq_lane_u32 获取指定索引处的 32 位整数，并通过 npyv_set_u32 宏进行元素重新排列
    #define npyv_permi128_s32(A, E0, E1, E2, E3)          \
        npyv_set_s32(                                     \
            vgetq_lane_s32(A, E0), vgetq_lane_s32(A, E1), \
            vgetq_lane_s32(A, E2), vgetq_lane_s32(A, E3)  \
        )
    #define npyv_permi128_f32(A, E0, E1, E2, E3)          \
        npyv_set_f32(                                     \
            vgetq_lane_f32(A, E0), vgetq_lane_f32(A, E1), \
            vgetq_lane_f32(A, E2), vgetq_lane_f32(A, E3)  \
        )
#endif

#if defined(__clang__) || defined(__GNUC__)
    // 对于 Clang 和 GCC 编译器，定义整数和浮点数的 128 位元素重排宏
    #define npyv_permi128_s32 npyv_permi128_u32
    #define npyv_permi128_f32 npyv_permi128_u32
#endif

// 根据不同的编译器预处理指令，定义宏 npyv_permi128_u64，用于对 NEON 寄存器中的 128 位无符号长整型数据进行按指定索引重新排列
#ifdef __clang__
    #define npyv_permi128_u64(A, E0, E1) \
        __builtin_shufflevector(A, A, E0, E1)
// 对于 Clang 编译器，使用 __builtin_shufflevector 进行元素重新排列
#elif defined(__GNUC__)
    #define npyv_permi128_u64(A, E0, E1) \
        __builtin_shuffle(A, npyv_set_u64(E0, E1))
// 对于 GCC 编译器，使用 __builtin_shuffle 结合 npyv_set_u64 宏进行元素重新排列
#else
    #define npyv_permi128_u64(A, E0, E1)                  \
        npyv_set_u64(                                     \
            vgetq_lane_u64(A, E0), vgetq_lane_u64(A, E1)  \
        )
    #define npyv_permi128_s64(A, E0, E1)                  \
        npyv_set_s64(                                     \
            vgetq_lane_s64(A, E0), vgetq_lane_s64(A, E1)  \
        )
    #define npyv_permi128_f64(A, E0, E1)                  \
        npyv_set_f64(                                     \
            vgetq_lane_f64(A, E0), vgetq_lane_f64(A, E1)  \
        )
#endif

#if defined(__clang__) || defined(__GNUC__)
    // 对于 Clang 和 GCC 编译器，定义整数和浮点数的 128 位元素重排宏
    #define npyv_permi128_s64 npyv_permi128_u64
    #define npyv_permi128_f64 npyv_permi128_u64
#endif

#if !NPY_SIMD_F64
    // 如果不支持双精度浮点数 SIMD 操作，则取消定义 npyv_permi128_f64 宏
    #undef npyv_permi128_f64
#endif

#endif // _NPY_SIMD_NEON_REORDER_H
// 结束 _NPY_SIMD_NEON_REORDER_H 文件的条件编译
```