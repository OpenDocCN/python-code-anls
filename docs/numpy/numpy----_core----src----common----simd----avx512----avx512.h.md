# `.\numpy\numpy\_core\src\common\simd\avx512\avx512.h`

```
#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"
#endif
// 如果未定义 _NPY_SIMD_H_，则输出错误信息，不是独立头文件
#define NPY_SIMD 512
// 定义 NPY_SIMD 为 512，表示 SIMD 宽度为 512 比特
#define NPY_SIMD_WIDTH 64
// 定义 NPY_SIMD_WIDTH 为 64，表示每个 SIMD 寄存器的宽度为 64 比特
#define NPY_SIMD_F32 1
// 定义 NPY_SIMD_F32 为 1，表示支持单精度浮点数的 SIMD 操作
#define NPY_SIMD_F64 1
// 定义 NPY_SIMD_F64 为 1，表示支持双精度浮点数的 SIMD 操作
#define NPY_SIMD_FMA3 1 // native support
// 定义 NPY_SIMD_FMA3 为 1，表示本机支持 Fused Multiply-Add (FMA3) 指令
#define NPY_SIMD_BIGENDIAN 0
// 定义 NPY_SIMD_BIGENDIAN 为 0，表示 SIMD 环境不是大端序
#define NPY_SIMD_CMPSIGNAL 0
// 定义 NPY_SIMD_CMPSIGNAL 为 0，表示 SIMD 比较信号不支持

// 以下两个宏定义允许使用 _mm512_i32gather_* 和 _mm512_i32scatter_*，限制为内存加载和存储的最大步长
#define NPY_SIMD_MAXLOAD_STRIDE32  (0x7fffffff / 16)
#define NPY_SIMD_MAXSTORE_STRIDE32 (0x7fffffff / 16)

typedef __m512i npyv_u8;
typedef __m512i npyv_s8;
typedef __m512i npyv_u16;
typedef __m512i npyv_s16;
typedef __m512i npyv_u32;
typedef __m512i npyv_s32;
typedef __m512i npyv_u64;
typedef __m512i npyv_s64;
typedef __m512  npyv_f32;
typedef __m512d npyv_f64;

#ifdef NPY_HAVE_AVX512BW
// 如果定义了 NPY_HAVE_AVX512BW，使用 AVX-512BW 指令集
typedef __mmask64 npyv_b8;
typedef __mmask32 npyv_b16;
#else
// 如果未定义 NPY_HAVE_AVX512BW，使用普通的 AVX-512 整型寄存器
typedef __m512i npyv_b8;
typedef __m512i npyv_b16;
#endif
// 定义 AVX-512 下的掩码类型，根据 NPY_HAVE_AVX512BW 的定义选择 __mmask* 或 __m512i

typedef __mmask16 npyv_b32;
typedef __mmask8  npyv_b64;

typedef struct { __m512i val[2]; } npyv_m512ix2;
typedef npyv_m512ix2 npyv_u8x2;
typedef npyv_m512ix2 npyv_s8x2;
typedef npyv_m512ix2 npyv_u16x2;
typedef npyv_m512ix2 npyv_s16x2;
typedef npyv_m512ix2 npyv_u32x2;
typedef npyv_m512ix2 npyv_s32x2;
typedef npyv_m512ix2 npyv_u64x2;
typedef npyv_m512ix2 npyv_s64x2;

typedef struct { __m512i val[3]; } npyv_m512ix3;
typedef npyv_m512ix3 npyv_u8x3;
typedef npyv_m512ix3 npyv_s8x3;
typedef npyv_m512ix3 npyv_u16x3;
typedef npyv_m512ix3 npyv_s16x3;
typedef npyv_m512ix3 npyv_u32x3;
typedef npyv_m512ix3 npyv_s32x3;
typedef npyv_m512ix3 npyv_u64x3;
typedef npyv_m512ix3 npyv_s64x3;

typedef struct { __m512  val[2]; } npyv_f32x2;
typedef struct { __m512d val[2]; } npyv_f64x2;
typedef struct { __m512  val[3]; } npyv_f32x3;
typedef struct { __m512d val[3]; } npyv_f64x3;

#define npyv_nlanes_u8  64
#define npyv_nlanes_s8  64
#define npyv_nlanes_u16 32
#define npyv_nlanes_s16 32
#define npyv_nlanes_u32 16
#define npyv_nlanes_s32 16
#define npyv_nlanes_u64 8
#define npyv_nlanes_s64 8
#define npyv_nlanes_f32 16
#define npyv_nlanes_f64 8

#include "utils.h"
#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
#include "math.h"
#include "maskop.h"
```