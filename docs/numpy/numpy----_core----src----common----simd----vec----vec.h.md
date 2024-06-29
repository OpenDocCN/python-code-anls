# `.\numpy\numpy\_core\src\common\simd\vec\vec.h`

```
/**
 * This block of code defines SIMD (Single Instruction, Multiple Data) types and macros
 * tailored for different SIMD architectures and compilers.
 */

#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"
#endif

// Check if either VX(zarch11) or VSX2(Power8/ISA2.07) support is required
#if !defined(NPY_HAVE_VX) && !defined(NPY_HAVE_VSX2)
    #error "require minimum support VX(zarch11) or VSX2(Power8/ISA2.07)"
#endif

// Check for VSX support and ensure it's only used in little-endian mode
#if defined(NPY_HAVE_VSX) && !defined(__LITTLE_ENDIAN__)
    #error "VSX support doesn't cover big-endian mode yet, only zarch."
#endif

// Check for VX(zarch) support and restrict usage to big-endian mode
#if defined(NPY_HAVE_VX) && defined(__LITTLE_ENDIAN__)
    #error "VX(zarch) support doesn't cover little-endian mode."
#endif

// Suppress certain GCC warnings for older versions (<= 7) that might affect SIMD intrinsics
#if defined(__GNUC__) && __GNUC__ <= 7
    /**
      * GCC <= 7 produces ambiguous warning caused by -Werror=maybe-uninitialized,
      * when certain intrinsics involved. `vec_ld` is one of them but it seemed to work fine,
      * and suppressing the warning wouldn't affect its functionality.
      */
    #pragma GCC diagnostic ignored "-Wuninitialized"
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

// Define SIMD configuration constants
#define NPY_SIMD 128
#define NPY_SIMD_WIDTH 16
#define NPY_SIMD_F64 1

// Define NPY_SIMD_F32 based on whether NPY_HAVE_VXE or NPY_HAVE_VSX is defined
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    #define NPY_SIMD_F32 1
#else
    #define NPY_SIMD_F32 0
#endif

// Enable native FMA3 support
#define NPY_SIMD_FMA3 1 // native support

// Define endianness and comparison signal macros based on architecture support
#ifdef NPY_HAVE_VX
    #define NPY_SIMD_BIGENDIAN 1
    #define NPY_SIMD_CMPSIGNAL 0
#else
    #define NPY_SIMD_BIGENDIAN 0
    #define NPY_SIMD_CMPSIGNAL 1
#endif

// Define SIMD vector types for various data types
typedef __vector unsigned char      npyv_u8;
typedef __vector signed char        npyv_s8;
typedef __vector unsigned short     npyv_u16;
typedef __vector signed short       npyv_s16;
typedef __vector unsigned int       npyv_u32;
typedef __vector signed int         npyv_s32;
typedef __vector unsigned long long npyv_u64;
typedef __vector signed long long   npyv_s64;
#if NPY_SIMD_F32
typedef __vector float              npyv_f32;
#endif
typedef __vector double             npyv_f64;

// Define SIMD vector types for 2-element structures
typedef struct { npyv_u8  val[2]; } npyv_u8x2;
typedef struct { npyv_s8  val[2]; } npyv_s8x2;
typedef struct { npyv_u16 val[2]; } npyv_u16x2;
typedef struct { npyv_s16 val[2]; } npyv_s16x2;
typedef struct { npyv_u32 val[2]; } npyv_u32x2;
typedef struct { npyv_s32 val[2]; } npyv_s32x2;
typedef struct { npyv_u64 val[2]; } npyv_u64x2;
typedef struct { npyv_s64 val[2]; } npyv_s64x2;
#if NPY_SIMD_F32
typedef struct { npyv_f32 val[2]; } npyv_f32x2;
#endif
typedef struct { npyv_f64 val[2]; } npyv_f64x2;

// Define SIMD vector types for 3-element structures
typedef struct { npyv_u8  val[3]; } npyv_u8x3;
typedef struct { npyv_s8  val[3]; } npyv_s8x3;
typedef struct { npyv_u16 val[3]; } npyv_u16x3;
typedef struct { npyv_s16 val[3]; } npyv_s16x3;
typedef struct { npyv_u32 val[3]; } npyv_u32x3;
typedef struct { npyv_s32 val[3]; } npyv_s32x3;
typedef struct { npyv_u64 val[3]; } npyv_u64x3;
typedef struct { npyv_s64 val[3]; } npyv_s64x3;
#if NPY_SIMD_F32
typedef struct { npyv_f32 val[3]; } npyv_f32x3;
#endif
typedef struct { npyv_f64 val[3]; } npyv_f64x3;

// Define number of lanes for each SIMD vector type
#define npyv_nlanes_u8  16
#define npyv_nlanes_s8  16
#define npyv_nlanes_u16 8
#define npyv_nlanes_s16 8
// 定义不同类型的 SIMD 向量的长度
#define npyv_nlanes_u32 4  // 无符号整数型向量长度为 4
#define npyv_nlanes_s32 4  // 有符号整数型向量长度为 4
#define npyv_nlanes_u64 2  // 无符号长整型向量长度为 2
#define npyv_nlanes_s64 2  // 有符号长整型向量长度为 2
#define npyv_nlanes_f32 4  // 单精度浮点数型向量长度为 4
#define npyv_nlanes_f64 2  // 双精度浮点数型向量长度为 2

// 使用 __vector __bool 与 typedef 结合会导致模糊错误，此处进行定义以避免
#define npyv_b8  __vector __bool char    // 8 位布尔型向量
#define npyv_b16 __vector __bool short   // 16 位布尔型向量
#define npyv_b32 __vector __bool int     // 32 位布尔型向量
#define npyv_b64 __vector __bool long long  // 64 位布尔型向量

// 包含各种工具、内存、杂项、重新排序、操作符、类型转换、算术运算和数学函数的头文件
#include "utils.h"
#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
#include "math.h"
```