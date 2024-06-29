# `.\numpy\numpy\_core\src\common\simd\neon\misc.h`

```py
#ifndef NPY_SIMD
    // 如果未定义 NPY_SIMD 宏，则报错，因为此文件不可作为独立头文件使用
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_MISC_H
#define _NPY_SIMD_NEON_MISC_H

// 定义一个函数-like 宏，用于返回所有通道为零的 uint8x16_t 向量
#define npyv_zero_u8()  vreinterpretq_u8_s32(npyv_zero_s32())
// 定义一个函数-like 宏，用于返回所有通道为零的 int8x16_t 向量
#define npyv_zero_s8()  vreinterpretq_s8_s32(npyv_zero_s32())
// 定义一个函数-like 宏，用于返回所有通道为零的 uint16x8_t 向量
#define npyv_zero_u16() vreinterpretq_u16_s32(npyv_zero_s32())
// 定义一个函数-like 宏，用于返回所有通道为零的 int16x8_t 向量
#define npyv_zero_s16() vreinterpretq_s16_s32(npyv_zero_s32())
// 定义一个函数-like 宏，用于返回所有通道为零的 uint32x4_t 向量
#define npyv_zero_u32() vdupq_n_u32((unsigned)0)
// 定义一个函数-like 宏，用于返回所有通道为零的 int32x4_t 向量
#define npyv_zero_s32() vdupq_n_s32((int)0)
// 定义一个函数-like 宏，用于返回所有通道为零的 uint64x2_t 向量
#define npyv_zero_u64() vreinterpretq_u64_s32(npyv_zero_s32())
// 定义一个函数-like 宏，用于返回所有通道为零的 int64x2_t 向量
#define npyv_zero_s64() vreinterpretq_s64_s32(npyv_zero_s32())
// 定义一个函数-like 宏，用于返回所有通道为零的 float32x4_t 向量
#define npyv_zero_f32() vdupq_n_f32(0.0f)
// 定义一个函数-like 宏，用于返回所有通道为零的 float64x2_t 向量
#define npyv_zero_f64() vdupq_n_f64(0.0)

// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 uint8x16_t 向量
#define npyv_setall_u8  vdupq_n_u8
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 int8x16_t 向量
#define npyv_setall_s8  vdupq_n_s8
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 uint16x8_t 向量
#define npyv_setall_u16 vdupq_n_u16
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 int16x8_t 向量
#define npyv_setall_s16 vdupq_n_s16
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 uint32x4_t 向量
#define npyv_setall_u32 vdupq_n_u32
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 int32x4_t 向量
#define npyv_setall_s32 vdupq_n_s32
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 uint64x2_t 向量
#define npyv_setall_u64 vdupq_n_u64
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 int64x2_t 向量
#define npyv_setall_s64 vdupq_n_s64
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 float32x4_t 向量
#define npyv_setall_f32 vdupq_n_f32
// 定义一个函数-like 宏，用于返回所有通道设置为指定值的 float64x2_t 向量
#define npyv_setall_f64 vdupq_n_f64

// 定义一个函数-like 宏，根据填充值 FILL 和其他参数，返回指定类型和值的向量
// 这些宏适用于不同大小的数据类型，并且具有变长参数列表
#if defined(__clang__) || defined(__GNUC__)
    #define npyv_setf_u8(FILL, ...)  ((uint8x16_t){NPYV__SET_FILL_16(uint8_t, FILL, __VA_ARGS__)})
    #define npyv_setf_s8(FILL, ...)  ((int8x16_t){NPYV__SET_FILL_16(int8_t, FILL, __VA_ARGS__)})
    #define npyv_setf_u16(FILL, ...) ((uint16x8_t){NPYV__SET_FILL_8(uint16_t, FILL, __VA_ARGS__)})
    #define npyv_setf_s16(FILL, ...) ((int16x8_t){NPYV__SET_FILL_8(int16_t, FILL, __VA_ARGS__)})
    #define npyv_setf_u32(FILL, ...) ((uint32x4_t){NPYV__SET_FILL_4(uint32_t, FILL, __VA_ARGS__)})
    #define npyv_setf_s32(FILL, ...) ((int32x4_t){NPYV__SET_FILL_4(int32_t, FILL, __VA_ARGS__)})
    #define npyv_setf_u64(FILL, ...) ((uint64x2_t){NPYV__SET_FILL_2(uint64_t, FILL, __VA_ARGS__)})
    #define npyv_setf_s64(FILL, ...) ((int64x2_t){NPYV__SET_FILL_2(int64_t, FILL, __VA_ARGS__)})
    #define npyv_setf_f32(FILL, ...) ((float32x4_t){NPYV__SET_FILL_4(float, FILL, __VA_ARGS__)})
    #if NPY_SIMD_F64
        #define npyv_setf_f64(FILL, ...) ((float64x2_t){NPYV__SET_FILL_2(double, FILL, __VA_ARGS__)})
    #endif
#else
    // 对于不支持变长参数的编译器，定义一个函数，手动设置向量的每个通道值，并返回向量
    NPY_FINLINE uint8x16_t npyv__set_u8(npy_uint8 i0, npy_uint8 i1, npy_uint8 i2, npy_uint8 i3,
        npy_uint8 i4, npy_uint8 i5, npy_uint8 i6, npy_uint8 i7, npy_uint8 i8, npy_uint8 i9,
        npy_uint8 i10, npy_uint8 i11, npy_uint8 i12, npy_uint8 i13, npy_uint8 i14, npy_uint8 i15)
    {
        // 创建一个数据数组，存储指定的每个通道的值
        const uint8_t NPY_DECL_ALIGNED(16) data[16] = {
            i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15
        };
        // 使用 vld1q_u8 函数加载数据数组，返回一个 uint8x16_t 向量
        return vld1q_u8(data);
    }
#endif

#endif // _NPY_SIMD_NEON_MISC_H
    # 定义一个内联函数，用于创建一个包含8个int8_t元素的向量
    NPY_FINLINE int8x16_t npyv__set_s8(npy_int8 i0, npy_int8 i1, npy_int8 i2, npy_int8 i3,
        npy_int8 i4, npy_int8 i5, npy_int8 i6, npy_int8 i7, npy_int8 i8, npy_int8 i9,
        npy_int8 i10, npy_int8 i11, npy_int8 i12, npy_int8 i13, npy_int8 i14, npy_int8 i15)
    {
        # 创建一个包含16个int8_t元素的常量数组
        const int8_t NPY_DECL_ALIGNED(16) data[16] = {
            i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15
        };
        # 使用ARM NEON指令加载16字节对齐的数据并返回一个int8x16_t类型的向量
        return vld1q_s8(data);
    }
    
    # 定义一个内联函数，用于创建一个包含8个uint16_t元素的向量
    NPY_FINLINE uint16x8_t npyv__set_u16(npy_uint16 i0, npy_uint16 i1, npy_uint16 i2, npy_uint16 i3,
        npy_uint16 i4, npy_uint16 i5, npy_uint16 i6, npy_uint16 i7)
    {
        # 创建一个包含8个uint16_t元素的常量数组
        const uint16_t NPY_DECL_ALIGNED(16) data[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
        # 使用ARM NEON指令加载16字节对齐的数据并返回一个uint16x8_t类型的向量
        return vld1q_u16(data);
    }
    
    # 定义一个内联函数，用于创建一个包含8个int16_t元素的向量
    NPY_FINLINE int16x8_t npyv__set_s16(npy_int16 i0, npy_int16 i1, npy_int16 i2, npy_int16 i3,
        npy_int16 i4, npy_int16 i5, npy_int16 i6, npy_int16 i7)
    {
        # 创建一个包含8个int16_t元素的常量数组
        const int16_t NPY_DECL_ALIGNED(16) data[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
        # 使用ARM NEON指令加载16字节对齐的数据并返回一个int16x8_t类型的向量
        return vld1q_s16(data);
    }
    
    # 定义一个内联函数，用于创建一个包含4个uint32_t元素的向量
    NPY_FINLINE uint32x4_t npyv__set_u32(npy_uint32 i0, npy_uint32 i1, npy_uint32 i2, npy_uint32 i3)
    {
        # 创建一个包含4个uint32_t元素的常量数组
        const uint32_t NPY_DECL_ALIGNED(16) data[4] = {i0, i1, i2, i3};
        # 使用ARM NEON指令加载16字节对齐的数据并返回一个uint32x4_t类型的向量
        return vld1q_u32(data);
    }
    
    # 定义一个内联函数，用于创建一个包含4个int32_t元素的向量
    NPY_FINLINE int32x4_t npyv__set_s32(npy_int32 i0, npy_int32 i1, npy_int32 i2, npy_int32 i3)
    {
        # 创建一个包含4个int32_t元素的常量数组
        const int32_t NPY_DECL_ALIGNED(16) data[4] = {i0, i1, i2, i3};
        # 使用ARM NEON指令加载16字节对齐的数据并返回一个int32x4_t类型的向量
        return vld1q_s32(data);
    }
    
    # 定义一个内联函数，用于创建一个包含2个uint64_t元素的向量
    NPY_FINLINE uint64x2_t npyv__set_u64(npy_uint64 i0, npy_uint64 i1)
    {
        # 创建一个包含2个uint64_t元素的常量数组
        const uint64_t NPY_DECL_ALIGNED(16) data[2] = {i0, i1};
        # 使用ARM NEON指令加载16字节对齐的数据并返回一个uint64x2_t类型的向量
        return vld1q_u64(data);
    }
    
    # 定义一个内联函数，用于创建一个包含2个int64_t元素的向量
    NPY_FINLINE int64x2_t npyv__set_s64(npy_int64 i0, npy_int64 i1)
    {
        # 创建一个包含2个int64_t元素的常量数组
        const int64_t NPY_DECL_ALIGNED(16) data[2] = {i0, i1};
        # 使用ARM NEON指令加载16字节对齐的数据并返回一个int64x2_t类型的向量
        return vld1q_s64(data);
    }
    
    # 定义一个内联函数，用于创建一个包含4个float元素的向量
    NPY_FINLINE float32x4_t npyv__set_f32(float i0, float i1, float i2, float i3)
    {
        # 创建一个包含4个float元素的常量数组
        const float NPY_DECL_ALIGNED(16) data[4] = {i0, i1, i2, i3};
        # 使用ARM NEON指令加载16字节对齐的数据并返回一个float32x4_t类型的向量
        return vld1q_f32(data);
    }
    
    # 如果支持双精度浮点运算（NPY_SIMD_F64宏定义为真），定义一个内联函数，用于创建一个包含2个double元素的向量
    #if NPY_SIMD_F64
        NPY_FINLINE float64x2_t npyv__set_f64(double i0, double i1)
        {
            # 创建一个包含2个double元素的常量数组
            const double NPY_DECL_ALIGNED(16) data[2] = {i0, i1};
            # 使用ARM NEON指令加载16字节对齐的数据并返回一个float64x2_t类型的向量
            return vld1q_f64(data);
        }
    #endif
    
    # 定义一个宏，用于创建一个包含8个uint8_t元素的向量，通过调用npyv__set_u8函数
    #define npyv_setf_u8(FILL, ...)  npyv__set_u8(NPYV__SET_FILL_16(npy_uint8, FILL, __VA_ARGS__))
    # 定义一个宏，用于创建一个包含8个int8_t元素的向量，通过调用npyv__set_s8函数
    #define npyv_setf_s8(FILL, ...)  npyv__set_s8(NPYV__SET_FILL_16(npy_int8, FILL, __VA_ARGS__))
    # 定义一个宏，用于创建一个包含8个uint16_t元素的向量，通过调用npyv__set_u16函数
    #define npyv_setf_u16(FILL, ...) npyv__set_u16(NPYV__SET_FILL_8(npy_uint16, FILL, __VA_ARGS__))
    # 定义一个宏，用于创建一个包含8个int16_t元素的向量，通过调用npyv__set_s16函数
    #define npyv_setf_s16(FILL, ...) npyv__set_s16(NPYV__SET_FILL_8(npy_int16, FILL, __VA_ARGS__))
    # 定义一个宏，用于创建一个包含4个uint32_t元素的向量，通过调用npyv__set_u32函数
    #define npyv_setf_u32(FILL, ...) npyv__set_u32(NPYV__SET_FILL_4(npy_uint32, FILL, __VA_ARGS__))
    # 定义一个宏，用于创建一个包含4个int32_t元素的向量，通过调用npyv__set_s32函数
    #define npyv_set
    #define npyv_setf_s64(FILL, ...) npyv__set_s64(NPYV__SET_FILL_2(npy_int64, FILL, __VA_ARGS__))
    # 定义一个宏，用于设置长度为2的npy_int64类型向量，填充值为FILL和可变参数列表
    
    #define npyv_setf_f32(FILL, ...) npyv__set_f32(NPYV__SET_FILL_4(float, FILL, __VA_ARGS__))
    # 定义一个宏，用于设置长度为4的float类型向量，填充值为FILL和可变参数列表
    
    #if NPY_SIMD_F64
        #define npyv_setf_f64(FILL, ...) npyv__set_f64(NPYV__SET_FILL_2(double, FILL, __VA_ARGS__))
    #endif
    # 如果NPY_SIMD_F64为真（即双精度浮点数支持可用），则定义一个宏，用于设置长度为2的double类型向量，填充值为FILL和可变参数列表
// 宏定义：设置每个向量的特定值，并将其余的所有向量设为零
#define npyv_set_u8(...)  npyv_setf_u8(0,  __VA_ARGS__)
#define npyv_set_s8(...)  npyv_setf_s8(0,  __VA_ARGS__)
#define npyv_set_u16(...) npyv_setf_u16(0, __VA_ARGS__)
#define npyv_set_s16(...) npyv_setf_s16(0, __VA_ARGS__)
#define npyv_set_u32(...) npyv_setf_u32(0, __VA_ARGS__)
#define npyv_set_s32(...) npyv_setf_s32(0, __VA_ARGS__)
#define npyv_set_u64(...) npyv_setf_u64(0, __VA_ARGS__)
#define npyv_set_s64(...) npyv_setf_s64(0, __VA_ARGS__)
#define npyv_set_f32(...) npyv_setf_f32(0, __VA_ARGS__)
#define npyv_set_f64(...) npyv_setf_f64(0, __VA_ARGS__)

// 宏定义：按每个向量的每个元素选择
#define npyv_select_u8  vbslq_u8
#define npyv_select_s8  vbslq_s8
#define npyv_select_u16 vbslq_u16
#define npyv_select_s16 vbslq_s16
#define npyv_select_u32 vbslq_u32
#define npyv_select_s32 vbslq_s32
#define npyv_select_u64 vbslq_u64
#define npyv_select_s64 vbslq_s64
#define npyv_select_f32 vbslq_f32
#define npyv_select_f64 vbslq_f64

// 宏定义：提取第一个向量的第一个元素
#define npyv_extract0_u8(A) ((npy_uint8)vgetq_lane_u8(A, 0))
#define npyv_extract0_s8(A) ((npy_int8)vgetq_lane_s8(A, 0))
#define npyv_extract0_u16(A) ((npy_uint16)vgetq_lane_u16(A, 0))
#define npyv_extract0_s16(A) ((npy_int16)vgetq_lane_s16(A, 0))
#define npyv_extract0_u32(A) ((npy_uint32)vgetq_lane_u32(A, 0))
#define npyv_extract0_s32(A) ((npy_int32)vgetq_lane_s32(A, 0))
#define npyv_extract0_u64(A) ((npy_uint64)vgetq_lane_u64(A, 0))
#define npyv_extract0_s64(A) ((npy_int64)vgetq_lane_s64(A, 0))
#define npyv_extract0_f32(A) vgetq_lane_f32(A, 0)
#define npyv_extract0_f64(A) vgetq_lane_f64(A, 0)

// 宏定义：重新解释数据类型为另一种数据类型
#define npyv_reinterpret_u8_u8(X) X
#define npyv_reinterpret_u8_s8  vreinterpretq_u8_s8
#define npyv_reinterpret_u8_u16 vreinterpretq_u8_u16
#define npyv_reinterpret_u8_s16 vreinterpretq_u8_s16
#define npyv_reinterpret_u8_u32 vreinterpretq_u8_u32
#define npyv_reinterpret_u8_s32 vreinterpretq_u8_s32
#define npyv_reinterpret_u8_u64 vreinterpretq_u8_u64
#define npyv_reinterpret_u8_s64 vreinterpretq_u8_s64
#define npyv_reinterpret_u8_f32 vreinterpretq_u8_f32
#define npyv_reinterpret_u8_f64 vreinterpretq_u8_f64

#define npyv_reinterpret_s8_s8(X) X
#define npyv_reinterpret_s8_u8  vreinterpretq_s8_u8
#define npyv_reinterpret_s8_u16 vreinterpretq_s8_u16
#define npyv_reinterpret_s8_s16 vreinterpretq_s8_s16
#define npyv_reinterpret_s8_u32 vreinterpretq_s8_u32
#define npyv_reinterpret_s8_s32 vreinterpretq_s8_s32
#define npyv_reinterpret_s8_u64 vreinterpretq_s8_u64
#define npyv_reinterpret_s8_s64 vreinterpretq_s8_s64
#define npyv_reinterpret_s8_f32 vreinterpretq_s8_f32
#define npyv_reinterpret_s8_f64 vreinterpretq_s8_f64

#define npyv_reinterpret_u16_u16(X) X
#define npyv_reinterpret_u16_u8  vreinterpretq_u16_u8
#define npyv_reinterpret_u16_s8  vreinterpretq_u16_s8
#define npyv_reinterpret_u16_s16 vreinterpretq_u16_s16
#define npyv_reinterpret_u16_u32 vreinterpretq_u16_u32
# 定义宏 npyv_reinterpret_u16_s32，用于将 16 位无符号整数向量重新解释为 32 位有符号整数向量
#define npyv_reinterpret_u16_s32 vreinterpretq_u16_s32
# 定义宏 npyv_reinterpret_u16_u64，用于将 16 位无符号整数向量重新解释为 64 位无符号整数向量
#define npyv_reinterpret_u16_u64 vreinterpretq_u16_u64
# 定义宏 npyv_reinterpret_u16_s64，用于将 16 位无符号整数向量重新解释为 64 位有符号整数向量
#define npyv_reinterpret_u16_s64 vreinterpretq_u16_s64
# 定义宏 npyv_reinterpret_u16_f32，用于将 16 位无符号整数向量重新解释为 32 位浮点数向量
#define npyv_reinterpret_u16_f32 vreinterpretq_u16_f32
# 定义宏 npyv_reinterpret_u16_f64，用于将 16 位无符号整数向量重新解释为 64 位浮点数向量
#define npyv_reinterpret_u16_f64 vreinterpretq_u16_f64

# 定义宏 npyv_reinterpret_s16_s16，用于将 16 位有符号整数向量重新解释为 16 位有符号整数向量（无需重新解释）
#define npyv_reinterpret_s16_s16(X) X
# 定义宏 npyv_reinterpret_s16_u8，用于将 16 位有符号整数向量重新解释为 8 位无符号整数向量
#define npyv_reinterpret_s16_u8  vreinterpretq_s16_u8
# 定义宏 npyv_reinterpret_s16_s8，用于将 16 位有符号整数向量重新解释为 8 位有符号整数向量
#define npyv_reinterpret_s16_s8  vreinterpretq_s16_s8
# 定义宏 npyv_reinterpret_s16_u16，用于将 16 位有符号整数向量重新解释为 16 位无符号整数向量
#define npyv_reinterpret_s16_u16 vreinterpretq_s16_u16
# 定义宏 npyv_reinterpret_s16_u32，用于将 16 位有符号整数向量重新解释为 32 位无符号整数向量
#define npyv_reinterpret_s16_u32 vreinterpretq_s16_u32
# 定义宏 npyv_reinterpret_s16_s32，用于将 16 位有符号整数向量重新解释为 32 位有符号整数向量
#define npyv_reinterpret_s16_s32 vreinterpretq_s16_s32
# 定义宏 npyv_reinterpret_s16_u64，用于将 16 位有符号整数向量重新解释为 64 位无符号整数向量
#define npyv_reinterpret_s16_u64 vreinterpretq_s16_u64
# 定义宏 npyv_reinterpret_s16_s64，用于将 16 位有符号整数向量重新解释为 64 位有符号整数向量
#define npyv_reinterpret_s16_s64 vreinterpretq_s16_s64
# 定义宏 npyv_reinterpret_s16_f32，用于将 16 位有符号整数向量重新解释为 32 位浮点数向量
#define npyv_reinterpret_s16_f32 vreinterpretq_s16_f32
# 定义宏 npyv_reinterpret_s16_f64，用于将 16 位有符号整数向量重新解释为 64 位浮点数向量
#define npyv_reinterpret_s16_f64 vreinterpretq_s16_f64

# 定义宏 npyv_reinterpret_u32_u32，用于将 32 位无符号整数向量重新解释为 32 位无符号整数向量（无需重新解释）
#define npyv_reinterpret_u32_u32(X) X
# 定义宏 npyv_reinterpret_u32_u8，用于将 32 位无符号整数向量重新解释为 8 位无符号整数向量
#define npyv_reinterpret_u32_u8  vreinterpretq_u32_u8
# 定义宏 npyv_reinterpret_u32_s8，用于将 32 位无符号整数向量重新解释为 8 位有符号整数向量
#define npyv_reinterpret_u32_s8  vreinterpretq_u32_s8
# 定义宏 npyv_reinterpret_u32_u16，用于将 32 位无符号整数向量重新解释为 16 位无符号整数向量
#define npyv_reinterpret_u32_u16 vreinterpretq_u32_u16
# 定义宏 npyv_reinterpret_u32_s16，用于将 32 位无符号整数向量重新解释为 16 位有符号整数向量
#define npyv_reinterpret_u32_s16 vreinterpretq_u32_s16
# 定义宏 npyv_reinterpret_u32_s32，用于将 32 位无符号整数向量重新解释为 32 位有符号整数向量
#define npyv_reinterpret_u32_s32 vreinterpretq_u32_s32
# 定义宏 npyv_reinterpret_u32_u64，用于将 32 位无符号整数向量重新解释为 64 位无符号整数向量
#define npyv_reinterpret_u32_u64 vreinterpretq_u32_u64
# 定义宏 npyv_reinterpret_u32_s64，用于将 32 位无符号整数向量重新解释为 64 位有符号整数向量
#define npyv_reinterpret_u32_s64 vreinterpretq_u32_s64
# 定义宏 npyv_reinterpret_u32_f32，用于将 32 位无符号整数向量重新解释为 32 位浮点数向量
#define npyv_reinterpret_u32_f32 vreinterpretq_u32_f32
# 定义宏 npyv_reinterpret_u32_f64，用于将 32 位无符号整数向量重新解释为 64 位浮点数向量
#define npyv_reinterpret_u32_f64 vreinterpretq_u32_f64

# 定义宏 npyv_reinterpret_s32_s32，用于将 32 位有符号整数向量重新解释为 32 位有符号整数向量（无需重新解释）
#define npyv_reinterpret_s32_s32(X) X
# 定义宏 npyv_reinterpret_s32_u8，用于将 32 位有符号整数向量重新解释为 8 位无符号整数向量
#define npyv_reinterpret_s32_u8  vreinterpretq_s32_u8
# 定义宏 npyv_reinterpret_s32_s8，用于将 32 位有符号整数向量重新解释为 8 位有符号整数向量
#define npyv_reinterpret_s32_s8  vreinterpretq_s32_s8
# 定义宏 npyv_reinterpret_s32_u16，用于将 32 位有符号整数向量重新解释为 16 位无符号整数向量
#define npyv_reinterpret_s32_u16 vreinterpretq_s32_u16
# 定义宏 npyv_reinterpret_s32_s16，用于将 32 位有符号整数向量重新解释为 16 位有符号整数向量
#define npyv_reinterpret_s32_s16 vreinterpretq_s32_s16
# 定义宏 npyv_reinterpret_s32_u32，用于将 32 位有符号整数向量重新解释为 32 位无符号整数向量
#define npyv_reinterpret_s32_u32 vreinterpretq_s32_u32
# 定义宏 npyv_reinterpret_s32_u64，用于将 32 位有符号整数向量重新解释为 64 位无符号整数向量
#define npyv_reinterpret_s32
// 定义一系列宏，用于将不同类型的数据重新解释为 float32 向量
#define npyv_reinterpret_f32_u8  vreinterpretq_f32_u8
#define npyv_reinterpret_f32_s8  vreinterpretq_f32_s8
#define npyv_reinterpret_f32_u16 vreinterpretq_f32_u16
#define npyv_reinterpret_f32_s16 vreinterpretq_f32_s16
#define npyv_reinterpret_f32_u32 vreinterpretq_f32_u32
#define npyv_reinterpret_f32_s32 vreinterpretq_f32_s32
#define npyv_reinterpret_f32_u64 vreinterpretq_f32_u64
#define npyv_reinterpret_f32_s64 vreinterpretq_f32_s64
#define npyv_reinterpret_f32_f64 vreinterpretq_f32_f64

// 定义一个宏，用于将相同类型的 float64 数据重新解释为 float64 向量
#define npyv_reinterpret_f64_f64(X) X
#define npyv_reinterpret_f64_u8  vreinterpretq_f64_u8
#define npyv_reinterpret_f64_s8  vreinterpretq_f64_s8
#define npyv_reinterpret_f64_u16 vreinterpretq_f64_u16
#define npyv_reinterpret_f64_s16 vreinterpretq_f64_s16
#define npyv_reinterpret_f64_u32 vreinterpretq_f64_u32
#define npyv_reinterpret_f64_s32 vreinterpretq_f64_s32
#define npyv_reinterpret_f64_u64 vreinterpretq_f64_u64
#define npyv_reinterpret_f64_s64 vreinterpretq_f64_s64
#define npyv_reinterpret_f64_f32 vreinterpretq_f64_f32

// AVX2/AVX512 架构下才需要执行的清理操作宏
#define npyv_cleanup() ((void)0)

#endif // _NPY_SIMD_NEON_MISC_H
```