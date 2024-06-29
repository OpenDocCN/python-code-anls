# `.\numpy\numpy\_core\src\common\simd\neon\neon.h`

```py
#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"
#endif

// 定义 SIMD 宽度为 128 bits
#define NPY_SIMD 128
// 定义 SIMD 宽度为 16 bytes
#define NPY_SIMD_WIDTH 16
// 定义支持单精度浮点数操作
#define NPY_SIMD_F32 1

#ifdef __aarch64__
    // 如果是 ARM64 架构，定义支持双精度浮点数操作
    #define NPY_SIMD_F64 1
#else
    // 否则，不支持双精度浮点数操作
    #define NPY_SIMD_F64 0
#endif

#ifdef NPY_HAVE_NEON_VFPV4
    // 如果支持 NEON 指令集的 FMA3 指令，设置为原生支持
    #define NPY_SIMD_FMA3 1  // native support
#else
    // 否则，使用硬件模拟的方式支持 FMA3 指令
    #define NPY_SIMD_FMA3 0  // HW emulated
#endif

// 定义 SIMD 架构为小端模式
#define NPY_SIMD_BIGENDIAN 0
// 定义 SIMD 比较信号的支持
#define NPY_SIMD_CMPSIGNAL 1

// 下面是各种数据类型的 SIMD 向量定义
typedef uint8x16_t  npyv_u8;
typedef int8x16_t   npyv_s8;
typedef uint16x8_t  npyv_u16;
typedef int16x8_t   npyv_s16;
typedef uint32x4_t  npyv_u32;
typedef int32x4_t   npyv_s32;
typedef uint64x2_t  npyv_u64;
typedef int64x2_t   npyv_s64;
typedef float32x4_t npyv_f32;

#if NPY_SIMD_F64
typedef float64x2_t npyv_f64;
#endif

typedef uint8x16_t  npyv_b8;
typedef uint16x8_t  npyv_b16;
typedef uint32x4_t  npyv_b32;
typedef uint64x2_t  npyv_b64;

// 各种 SIMD 向量的多重结构定义
typedef uint8x16x2_t  npyv_u8x2;
typedef int8x16x2_t   npyv_s8x2;
typedef uint16x8x2_t  npyv_u16x2;
typedef int16x8x2_t   npyv_s16x2;
typedef uint32x4x2_t  npyv_u32x2;
typedef int32x4x2_t   npyv_s32x2;
typedef uint64x2x2_t  npyv_u64x2;
typedef int64x2x2_t   npyv_s64x2;
typedef float32x4x2_t npyv_f32x2;

#if NPY_SIMD_F64
typedef float64x2x2_t npyv_f64x2;
#endif

typedef uint8x16x3_t  npyv_u8x3;
typedef int8x16x3_t   npyv_s8x3;
typedef uint16x8x3_t  npyv_u16x3;
typedef int16x8x3_t   npyv_s16x3;
typedef uint32x4x3_t  npyv_u32x3;
typedef int32x4x3_t   npyv_s32x3;
typedef uint64x2x3_t  npyv_u64x3;
typedef int64x2x3_t   npyv_s64x3;
typedef float32x4x3_t npyv_f32x3;

#if NPY_SIMD_F64
typedef float64x2x3_t npyv_f64x3;
#endif

// 各种数据类型 SIMD 向量的通道数定义
#define npyv_nlanes_u8  16
#define npyv_nlanes_s8  16
#define npyv_nlanes_u16 8
#define npyv_nlanes_s16 8
#define npyv_nlanes_u32 4
#define npyv_nlanes_s32 4
#define npyv_nlanes_u64 2
#define npyv_nlanes_s64 2
#define npyv_nlanes_f32 4
#define npyv_nlanes_f64 2

// 包含 SIMD 相关的头文件
#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
#include "math.h"
```