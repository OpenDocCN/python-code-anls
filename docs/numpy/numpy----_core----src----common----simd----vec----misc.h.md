# `.\numpy\numpy\_core\src\common\simd\vec\misc.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif
// 如果 NPY_SIMD 宏未定义，则报错，表示此头文件不可单独使用

#ifndef _NPY_SIMD_VEC_MISC_H
#define _NPY_SIMD_VEC_MISC_H
// 如果 _NPY_SIMD_VEC_MISC_H 宏未定义，则定义它，防止头文件重复包含

// 定义各种数据类型的零向量宏
#define npyv_zero_u8()  ((npyv_u8)   npyv_setall_s32(0))
#define npyv_zero_s8()  ((npyv_s8)   npyv_setall_s32(0))
#define npyv_zero_u16() ((npyv_u16)  npyv_setall_s32(0))
#define npyv_zero_s16() ((npyv_s16)  npyv_setall_s32(0))
#define npyv_zero_u32() npyv_setall_u32(0)
#define npyv_zero_s32() npyv_setall_s32(0)
#define npyv_zero_u64() ((npyv_u64) npyv_setall_s32(0))
#define npyv_zero_s64() ((npyv_s64) npyv_setall_s32(0))
#if NPY_SIMD_F32
    #define npyv_zero_f32() npyv_setall_f32(0.0f)
#endif
#define npyv_zero_f64() npyv_setall_f64(0.0)
// 定义了返回各种数据类型的零向量的宏，分别使用相应的 npyv_setall_* 函数设置全零值

// 定义设置全向量相同值的宏，使用了不同数据类型的向量宏
#define npyv_setall_u8(VAL)  NPYV_IMPL_VEC_SPLTB(npyv_u8,  (unsigned char)(VAL))
#define npyv_setall_s8(VAL)  NPYV_IMPL_VEC_SPLTB(npyv_s8,  (signed char)(VAL))
#define npyv_setall_u16(VAL) NPYV_IMPL_VEC_SPLTH(npyv_u16, (unsigned short)(VAL))
#define npyv_setall_s16(VAL) NPYV_IMPL_VEC_SPLTH(npyv_s16, (short)(VAL))
#define npyv_setall_u32(VAL) NPYV_IMPL_VEC_SPLTW(npyv_u32, (unsigned int)(VAL))
#define npyv_setall_s32(VAL) NPYV_IMPL_VEC_SPLTW(npyv_s32, (int)(VAL))
#if NPY_SIMD_F32
    #define npyv_setall_f32(VAL) NPYV_IMPL_VEC_SPLTW(npyv_f32, (VAL))
#endif
#define npyv_setall_u64(VAL) NPYV_IMPL_VEC_SPLTD(npyv_u64, (npy_uint64)(VAL))
#define npyv_setall_s64(VAL) NPYV_IMPL_VEC_SPLTD(npyv_s64, (npy_int64)(VAL))
#define npyv_setall_f64(VAL) NPYV_IMPL_VEC_SPLTD(npyv_f64, VAL)
// 定义了设置全向量为相同值的宏，使用了 NPYV_IMPL_VEC_SPLTB、NPYV_IMPL_VEC_SPLTH、NPYV_IMPL_VEC_SPLTW、NPYV_IMPL_VEC_SPLTD 宏

// 定义设置向量中各个通道的值的宏
#define npyv_setf_u8(FILL, ...)  ((npyv_u8){NPYV__SET_FILL_16(unsigned char, FILL, __VA_ARGS__)})
#define npyv_setf_s8(FILL, ...)  ((npyv_s8){NPYV__SET_FILL_16(signed char, FILL, __VA_ARGS__)})
#define npyv_setf_u16(FILL, ...) ((npyv_u16){NPYV__SET_FILL_8(unsigned short, FILL, __VA_ARGS__)})
#define npyv_setf_s16(FILL, ...) ((npyv_s16){NPYV__SET_FILL_8(short, FILL, __VA_ARGS__)})
#define npyv_setf_u32(FILL, ...) ((npyv_u32){NPYV__SET_FILL_4(unsigned int, FILL, __VA_ARGS__)})
#define npyv_setf_s32(FILL, ...) ((npyv_s32){NPYV__SET_FILL_4(int, FILL, __VA_ARGS__)})
#define npyv_setf_u64(FILL, ...) ((npyv_u64){NPYV__SET_FILL_2(npy_uint64, FILL, __VA_ARGS__)})
#define npyv_setf_s64(FILL, ...) ((npyv_s64){NPYV__SET_FILL_2(npy_int64, FILL, __VA_ARGS__)})
#if NPY_SIMD_F32
    #define npyv_setf_f32(FILL, ...) ((npyv_f32){NPYV__SET_FILL_4(float, FILL, __VA_ARGS__)})
#endif
#define npyv_setf_f64(FILL, ...) ((npyv_f64){NPYV__SET_FILL_2(double, FILL, __VA_ARGS__)})
// 定义了设置向量中各个通道值的宏，分别使用 NPYV__SET_FILL_16、NPYV__SET_FILL_8、NPYV__SET_FILL_4 宏

#endif
// 结束 _NPY_SIMD_VEC_MISC_H 宏的定义
// 将所有剩余的向量通道设置为零
#define npyv_set_u8(...)  npyv_setf_u8(0,  __VA_ARGS__)
#define npyv_set_s8(...)  npyv_setf_s8(0,  __VA_ARGS__)
#define npyv_set_u16(...) npyv_setf_u16(0, __VA_ARGS__)
#define npyv_set_s16(...) npyv_setf_s16(0, __VA_ARGS__)
#define npyv_set_u32(...) npyv_setf_u32(0, __VA_ARGS__)
#define npyv_set_s32(...) npyv_setf_s32(0, __VA_ARGS__)
#define npyv_set_u64(...) npyv_setf_u64(0, __VA_ARGS__)
#define npyv_set_s64(...) npyv_setf_s64(0, __VA_ARGS__)
#if NPY_SIMD_F32
    #define npyv_set_f32(...) npyv_setf_f32(0, __VA_ARGS__)
#endif
#define npyv_set_f64(...) npyv_setf_f64(0, __VA_ARGS__)

// 按通道选择
#define npyv_select_u8(MASK, A, B) vec_sel(B, A, MASK)
#define npyv_select_s8  npyv_select_u8
#define npyv_select_u16 npyv_select_u8
#define npyv_select_s16 npyv_select_u8
#define npyv_select_u32 npyv_select_u8
#define npyv_select_s32 npyv_select_u8
#define npyv_select_u64 npyv_select_u8
#define npyv_select_s64 npyv_select_u8
#if NPY_SIMD_F32
    #define npyv_select_f32 npyv_select_u8
#endif
#define npyv_select_f64 npyv_select_u8

// 提取第一个向量通道的值
#define npyv_extract0_u8(A) ((npy_uint8)vec_extract(A, 0))
#define npyv_extract0_s8(A) ((npy_int8)vec_extract(A, 0))
#define npyv_extract0_u16(A) ((npy_uint16)vec_extract(A, 0))
#define npyv_extract0_s16(A) ((npy_int16)vec_extract(A, 0))
#define npyv_extract0_u32(A) ((npy_uint32)vec_extract(A, 0))
#define npyv_extract0_s32(A) ((npy_int32)vec_extract(A, 0))
#define npyv_extract0_u64(A) ((npy_uint64)vec_extract(A, 0))
#define npyv_extract0_s64(A) ((npy_int64)vec_extract(A, 0))
#if NPY_SIMD_F32
    #define npyv_extract0_f32(A) vec_extract(A, 0)
#endif
#define npyv_extract0_f64(A) vec_extract(A, 0)

// 重新解释类型
#define npyv_reinterpret_u8_u8(X) X
#define npyv_reinterpret_u8_s8(X) ((npyv_u8)X)
#define npyv_reinterpret_u8_u16 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_s16 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_u32 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_s32 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_u64 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_s64 npyv_reinterpret_u8_s8
#if NPY_SIMD_F32
    #define npyv_reinterpret_u8_f32 npyv_reinterpret_u8_s8
#endif
#define npyv_reinterpret_u8_f64 npyv_reinterpret_u8_s8

#define npyv_reinterpret_s8_s8(X) X
#define npyv_reinterpret_s8_u8(X) ((npyv_s8)X)
#define npyv_reinterpret_s8_u16 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_s16 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_u32 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_s32 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_u64 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_s64 npyv_reinterpret_s8_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_s8_f32 npyv_reinterpret_s8_u8
#endif
#define npyv_reinterpret_s8_f64 npyv_reinterpret_s8_u8

#define npyv_reinterpret_u16_u16(X) X
#define npyv_reinterpret_u16_u8(X) ((npyv_u16)X)
#define npyv_reinterpret_u16_s8  npyv_reinterpret_u16_u8
#define npyv_reinterpret_u16_s16 npyv_reinterpret_u16_u8
// 定义将 unsigned 16 位整数重新解释为 signed 16 位整数的宏，实际上使用 unsigned 8 位整数宏
#define npyv_reinterpret_u16_u32 npyv_reinterpret_u16_u8
// 定义将 unsigned 16 位整数重新解释为 unsigned 32 位整数的宏，实际上使用 unsigned 8 位整数宏
#define npyv_reinterpret_u16_s32 npyv_reinterpret_u16_u8
// 定义将 unsigned 16 位整数重新解释为 signed 32 位整数的宏，实际上使用 unsigned 8 位整数宏
#define npyv_reinterpret_u16_u64 npyv_reinterpret_u16_u8
// 定义将 unsigned 16 位整数重新解释为 unsigned 64 位整数的宏，实际上使用 unsigned 8 位整数宏
#define npyv_reinterpret_u16_s64 npyv_reinterpret_u16_u8
// 定义将 unsigned 16 位整数重新解释为 signed 64 位整数的宏，实际上使用 unsigned 8 位整数宏
#if NPY_SIMD_F32
    #define npyv_reinterpret_u16_f32 npyv_reinterpret_u16_u8
#endif
// 如果支持单精度浮点运算，定义将 unsigned 16 位整数重新解释为单精度浮点数的宏，实际上使用 unsigned 8 位整数宏
#define npyv_reinterpret_u16_f64 npyv_reinterpret_u16_u8

#define npyv_reinterpret_s16_s16(X) X
// 定义将 signed 16 位整数重新解释为 signed 16 位整数的宏，直接返回参数 X
#define npyv_reinterpret_s16_u8(X) ((npyv_s16)X)
// 定义将 signed 16 位整数重新解释为 unsigned 8 位整数的宏，进行类型转换为 signed 16 位整数
#define npyv_reinterpret_s16_s8  npyv_reinterpret_s16_u8
// 定义将 signed 16 位整数重新解释为 signed 8 位整数的宏，实际上使用 signed 16 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_s16_u16 npyv_reinterpret_s16_u8
// 定义将 signed 16 位整数重新解释为 unsigned 16 位整数的宏，实际上使用 signed 16 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_s16_u32 npyv_reinterpret_s16_u8
// 定义将 signed 16 位整数重新解释为 unsigned 32 位整数的宏，实际上使用 signed 16 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_s16_s32 npyv_reinterpret_s16_u8
// 定义将 signed 16 位整数重新解释为 signed 32 位整数的宏，实际上使用 signed 16 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_s16_u64 npyv_reinterpret_s16_u8
// 定义将 signed 16 位整数重新解释为 unsigned 64 位整数的宏，实际上使用 signed 16 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_s16_s64 npyv_reinterpret_s16_u8
// 定义将 signed 16 位整数重新解释为 signed 64 位整数的宏，实际上使用 signed 16 位整数转为 unsigned 8 位整数宏
#if NPY_SIMD_F32
    #define npyv_reinterpret_s16_f32 npyv_reinterpret_s16_u8
#endif
// 如果支持单精度浮点运算，定义将 signed 16 位整数重新解释为单精度浮点数的宏，实际上使用 signed 16 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_s16_f64 npyv_reinterpret_s16_u8

#define npyv_reinterpret_u32_u32(X) X
// 定义将 unsigned 32 位整数重新解释为 unsigned 32 位整数的宏，直接返回参数 X
#define npyv_reinterpret_u32_u8(X) ((npyv_u32)X)
// 定义将 unsigned 32 位整数重新解释为 unsigned 8 位整数的宏，进行类型转换为 unsigned 32 位整数
#define npyv_reinterpret_u32_s8  npyv_reinterpret_u32_u8
// 定义将 unsigned 32 位整数重新解释为 signed 8 位整数的宏，实际上使用 unsigned 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_u32_u16 npyv_reinterpret_u32_u8
// 定义将 unsigned 32 位整数重新解释为 unsigned 16 位整数的宏，实际上使用 unsigned 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_u32_s16 npyv_reinterpret_u32_u8
// 定义将 unsigned 32 位整数重新解释为 signed 16 位整数的宏，实际上使用 unsigned 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_u32_s32 npyv_reinterpret_u32_u8
// 定义将 unsigned 32 位整数重新解释为 signed 32 位整数的宏，实际上使用 unsigned 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_u32_u64 npyv_reinterpret_u32_u8
// 定义将 unsigned 32 位整数重新解释为 unsigned 64 位整数的宏，实际上使用 unsigned 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_u32_s64 npyv_reinterpret_u32_u8
// 定义将 unsigned 32 位整数重新解释为 signed 64 位整数的宏，实际上使用 unsigned 32 位整数转为 unsigned 8 位整数宏
#if NPY_SIMD_F32
    #define npyv_reinterpret_u32_f32 npyv_reinterpret_u32_u8
#endif
// 如果支持单精度浮点运算，定义将 unsigned 32 位整数重新解释为单精度浮点数的宏，实际上使用 unsigned 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_u32_f64 npyv_reinterpret_u32_u8

#define npyv_reinterpret_s32_s32(X) X
// 定义将 signed 32 位整数重新解释为 signed 32 位整数的宏，直接返回参数 X
#define npyv_reinterpret_s32_u8(X) ((npyv_s32)X)
// 定义将 signed 32 位整数重新解释为 unsigned 8 位整数的宏，进行类型转换为 signed 32 位整数
#define npyv_reinterpret_s32_s8  npyv_reinterpret_s32_u8
// 定义将 signed 32 位整数重新解释为 signed 8 位整数的宏，实际上使用 signed 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_s32_u16 npyv_reinterpret_s32_u8
// 定义将 signed 32 位整数重新解释为 unsigned 16 位整数的宏，实际上使用 signed 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret_s32_s16 npyv_reinterpret_s32_u8
// 定义将 signed 32 位整数重新解释为 signed 16 位整数的宏，实际上使用 signed 32 位整数转为 unsigned 8 位整数宏
#define npyv_reinterpret
#define npyv_reinterpret_s64_u32 npyv_reinterpret_s64_u8
#define npyv_reinterpret_s64_s32 npyv_reinterpret_s64_u8
#define npyv_reinterpret_s64_u64 npyv_reinterpret_s64_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_s64_f32 npyv_reinterpret_s64_u8
#endif
#define npyv_reinterpret_s64_f64 npyv_reinterpret_s64_u8
```