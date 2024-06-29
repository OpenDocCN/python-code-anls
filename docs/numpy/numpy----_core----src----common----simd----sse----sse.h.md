# `.\numpy\numpy\_core\src\common\simd\sse\sse.h`

```py
#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"  // 如果没有包含在主头文件中，则报错
#endif

#define NPY_SIMD 128  // 定义 SIMD 宽度为 128 bits
#define NPY_SIMD_WIDTH 16  // SIMD 的宽度为 16 bytes
#define NPY_SIMD_F32 1  // 支持单精度浮点运算
#define NPY_SIMD_F64 1  // 支持双精度浮点运算
#if defined(NPY_HAVE_FMA3) || defined(NPY_HAVE_FMA4)
    #define NPY_SIMD_FMA3 1  // 使用原生的 FMA3 指令集支持
#else
    #define NPY_SIMD_FMA3 0  // 使用快速模拟的 FMA3
#endif
#define NPY_SIMD_BIGENDIAN 0  // SIMD 不支持大端模式
#define NPY_SIMD_CMPSIGNAL 1  // SIMD 支持比较信号处理

typedef __m128i npyv_u8;  // 定义无符号 8-bit 整数 SIMD 类型
typedef __m128i npyv_s8;  // 定义有符号 8-bit 整数 SIMD 类型
typedef __m128i npyv_u16;  // 定义无符号 16-bit 整数 SIMD 类型
typedef __m128i npyv_s16;  // 定义有符号 16-bit 整数 SIMD 类型
typedef __m128i npyv_u32;  // 定义无符号 32-bit 整数 SIMD 类型
typedef __m128i npyv_s32;  // 定义有符号 32-bit 整数 SIMD 类型
typedef __m128i npyv_u64;  // 定义无符号 64-bit 整数 SIMD 类型
typedef __m128i npyv_s64;  // 定义有符号 64-bit 整数 SIMD 类型
typedef __m128  npyv_f32;  // 定义单精度浮点数 SIMD 类型
typedef __m128d npyv_f64;  // 定义双精度浮点数 SIMD 类型

typedef __m128i npyv_b8;   // 定义 8-bit 布尔类型 SIMD 类型
typedef __m128i npyv_b16;  // 定义 16-bit 布尔类型 SIMD 类型
typedef __m128i npyv_b32;  // 定义 32-bit 布尔类型 SIMD 类型
typedef __m128i npyv_b64;  // 定义 64-bit 布尔类型 SIMD 类型

typedef struct { __m128i val[2]; } npyv_m128ix2;  // 定义包含两个 __m128i 的 SIMD 结构体
typedef npyv_m128ix2 npyv_u8x2;  // 定义两个 __m128i 的无符号 8-bit 整数 SIMD 结构体
typedef npyv_m128ix2 npyv_s8x2;  // 定义两个 __m128i 的有符号 8-bit 整数 SIMD 结构体
typedef npyv_m128ix2 npyv_u16x2;  // 定义两个 __m128i 的无符号 16-bit 整数 SIMD 结构体
typedef npyv_m128ix2 npyv_s16x2;  // 定义两个 __m128i 的有符号 16-bit 整数 SIMD 结构体
typedef npyv_m128ix2 npyv_u32x2;  // 定义两个 __m128i 的无符号 32-bit 整数 SIMD 结构体
typedef npyv_m128ix2 npyv_s32x2;  // 定义两个 __m128i 的有符号 32-bit 整数 SIMD 结构体
typedef npyv_m128ix2 npyv_u64x2;  // 定义两个 __m128i 的无符号 64-bit 整数 SIMD 结构体
typedef npyv_m128ix2 npyv_s64x2;  // 定义两个 __m128i 的有符号 64-bit 整数 SIMD 结构体

typedef struct { __m128i val[3]; } npyv_m128ix3;  // 定义包含三个 __m128i 的 SIMD 结构体
typedef npyv_m128ix3 npyv_u8x3;  // 定义三个 __m128i 的无符号 8-bit 整数 SIMD 结构体
typedef npyv_m128ix3 npyv_s8x3;  // 定义三个 __m128i 的有符号 8-bit 整数 SIMD 结构体
typedef npyv_m128ix3 npyv_u16x3;  // 定义三个 __m128i 的无符号 16-bit 整数 SIMD 结构体
typedef npyv_m128ix3 npyv_s16x3;  // 定义三个 __m128i 的有符号 16-bit 整数 SIMD 结构体
typedef npyv_m128ix3 npyv_u32x3;  // 定义三个 __m128i 的无符号 32-bit 整数 SIMD 结构体
typedef npyv_m128ix3 npyv_s32x3;  // 定义三个 __m128i 的有符号 32-bit 整数 SIMD 结构体
typedef npyv_m128ix3 npyv_u64x3;  // 定义三个 __m128i 的无符号 64-bit 整数 SIMD 结构体
typedef npyv_m128ix3 npyv_s64x3;  // 定义三个 __m128i 的有符号 64-bit 整数 SIMD 结构体

typedef struct { __m128  val[2]; } npyv_f32x2;  // 定义包含两个 __m128 的单精度浮点数 SIMD 结构体
typedef struct { __m128d val[2]; } npyv_f64x2;  // 定义包含两个 __m128d 的双精度浮点数 SIMD 结构体
typedef struct { __m128  val[3]; } npyv_f32x3;  // 定义包含三个 __m128 的单精度浮点数 SIMD 结构体
typedef struct { __m128d val[3]; } npyv_f64x3;  // 定义包含三个 __m128d 的双精度浮点数 SIMD 结构体

#define npyv_nlanes_u8  16  // 定义无符号 8-bit 整数 SIMD 向量的长度
#define npyv_nlanes_s8  16  // 定义有符号 8-bit 整数 SIMD 向量的长度
#define npyv_nlanes_u16 8   // 定义无符号 16-bit 整数 SIMD 向量的长度
#define npyv_nlanes_s16 8   // 定义有符号 16-bit 整数 SIMD 向量的长度
#define npyv_nlanes_u32 4   // 定义无符号 32-bit 整数 SIMD 向量的长度
#define npyv_nlanes_s32 4   // 定义有符号 32-bit 整数 SIMD 向量的长度
#define npyv_nlanes_u64 2   // 定义无符号 64-bit 整数 SIMD 向量的长度
#define npyv_nlanes_s64 2   // 定义有符号 64-bit 整数 SIMD 向量的长度
#define npyv_nlanes_f32 4   // 定义单精度浮点数 SIMD 向量的长度
#define npyv_nlanes_f64 2   // 定义双精度浮点数 SIMD 向量的长度

#include "utils.h"  // 包含实用工具函数的头文件
#include "memory.h"  // 包含内存操作相关的头文件
#include "misc.h"  // 包含杂项功能的头文件
#include "reorder.h"  // 包含重新排序相关的头文件
#include "operators.h"  // 包含运算符重载相关的头文件
#include "conversion.h"  // 包含类型转换相关的头文件
#include "arithmetic.h"  // 包含算术运算相关的头文件
#include "math.h"
```