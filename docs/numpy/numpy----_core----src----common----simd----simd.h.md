# `.\numpy\numpy\_core\src\common\simd\simd.h`

```py
#ifndef _NPY_SIMD_H_
#define _NPY_SIMD_H_
/**
 * the NumPy C SIMD vectorization interface "NPYV" are types and functions intended
 * to simplify vectorization of code on different platforms, currently supports
 * the following SIMD extensions SSE, AVX2, AVX512, VSX and NEON.
 *
 * TODO: Add an independent sphinx doc.
*/

#include "numpy/npy_common.h"
#ifndef __cplusplus
    #include <stdbool.h>
#endif

#include "npy_cpu_dispatch.h"
#include "simd_utils.h"

#ifdef __cplusplus
extern "C" {
#endif
/*
 * clang commit an aggressive optimization behaviour when flag `-ftrapping-math`
 * isn't fully supported that's present at -O1 or greater. When partially loading a
 * vector register for a operations that requires to fill up the remaining lanes
 * with certain value for example divide operation needs to fill the remaining value
 * with non-zero integer to avoid fp exception divide-by-zero.
 * clang optimizer notices that the entire register is not needed for the store
 * and optimizes out the fill of non-zero integer to the remaining
 * elements. As workaround we mark the returned register with `volatile`
 * followed by symmetric operand operation e.g. `or`
 * to convince the compiler that the entire vector is needed.
 */
#if defined(__clang__) && !defined(NPY_HAVE_CLANG_FPSTRICT)
    #define NPY_SIMD_GUARD_PARTIAL_LOAD 1
#else
    #define NPY_SIMD_GUARD_PARTIAL_LOAD 0
#endif

#if defined(_MSC_VER) && defined(_M_IX86)
/*
 * Avoid using any of the following intrinsics with MSVC 32-bit,
 * even if they are apparently work on newer versions.
 * They had bad impact on the generated instructions,
 * sometimes the compiler deal with them without the respect
 * of 32-bit mode which lead to crush due to execute 64-bit
 * instructions and other times generate bad emulated instructions.
 */
    #undef _mm512_set1_epi64
    #undef _mm256_set1_epi64x
    #undef _mm_set1_epi64x
    #undef _mm512_setr_epi64x
    #undef _mm256_setr_epi64x
    #undef _mm_setr_epi64x
    #undef _mm512_set_epi64x
    #undef _mm256_set_epi64x
    #undef _mm_set_epi64x
#endif

// lane type by intrin suffix
typedef npy_uint8  npyv_lanetype_u8;    // 定义无符号 8 位整数的 SIMD 向量元素类型
typedef npy_int8   npyv_lanetype_s8;    // 定义有符号 8 位整数的 SIMD 向量元素类型
typedef npy_uint16 npyv_lanetype_u16;   // 定义无符号 16 位整数的 SIMD 向量元素类型
typedef npy_int16  npyv_lanetype_s16;   // 定义有符号 16 位整数的 SIMD 向量元素类型
typedef npy_uint32 npyv_lanetype_u32;   // 定义无符号 32 位整数的 SIMD 向量元素类型
typedef npy_int32  npyv_lanetype_s32;   // 定义有符号 32 位整数的 SIMD 向量元素类型
typedef npy_uint64 npyv_lanetype_u64;   // 定义无符号 64 位整数的 SIMD 向量元素类型
typedef npy_int64  npyv_lanetype_s64;   // 定义有符号 64 位整数的 SIMD 向量元素类型
typedef float      npyv_lanetype_f32;   // 定义单精度浮点数的 SIMD 向量元素类型
typedef double     npyv_lanetype_f64;   // 定义双精度浮点数的 SIMD 向量元素类型

#if defined(NPY_HAVE_AVX512F) && !defined(NPY_SIMD_FORCE_256) && !defined(NPY_SIMD_FORCE_128)
    #include "avx512/avx512.h"
#elif defined(NPY_HAVE_AVX2) && !defined(NPY_SIMD_FORCE_128)
    #include "avx2/avx2.h"
#elif defined(NPY_HAVE_SSE2)
    #include "sse/sse.h"
#endif

// TODO: Add support for VSX(2.06) and BE Mode for VSX
#if defined(NPY_HAVE_VX) || (defined(NPY_HAVE_VSX2) && defined(__LITTLE_ENDIAN__))
    #include "vec/vec.h"
#endif

#ifdef NPY_HAVE_NEON
    #include "neon/neon.h"
#endif

#ifndef NPY_SIMD
    /// 定义：如果没有可用的SIMD扩展，NPY_SIMD为0，否则为SIMD的位宽（以比特为单位）。
    #define NPY_SIMD 0
    /// 定义：如果没有可用的SIMD扩展，NPY_SIMD_WIDTH为0，否则为SIMD的位宽（以字节为单位）。
    #define NPY_SIMD_WIDTH 0
    /// 定义：如果启用的SIMD扩展支持单精度浮点数，则为1，否则为0。
    #define NPY_SIMD_F32 0
    /// 定义：如果启用的SIMD扩展支持双精度浮点数，则为1，否则为0。
    #define NPY_SIMD_F64 0
    /// 定义：如果启用的SIMD扩展支持本地FMA（Fused Multiply-Add）操作，则为1，否则为0。
    /// 注意：即使不支持FMA指令集，仍然会模拟（快速）FMA操作，但在精度要求高时不应使用。
    #define NPY_SIMD_FMA3 0
    /// 定义：如果启用的SIMD扩展在大端模式下运行，则为1，否则为0。
    #define NPY_SIMD_BIGENDIAN 0
    /// 定义：如果支持的比较指令集（lt, le, gt, ge）在处理静默NaN时引发浮点无效异常，则为1，否则为0。
    #define NPY_SIMD_CMPSIGNAL 0
#ifndef _NPY_SIMD_H_

// 如果 _NPY_SIMD_H_ 未定义，则执行以下代码，防止头文件重复包含


#endif

// 结束条件，确保头文件在多次包含时不会被重复定义


#if !defined(NPY_HAVE_AVX512F) && NPY_SIMD && NPY_SIMD < 512
    #include "emulate_maskop.h"
#endif

// 如果未定义 NPY_HAVE_AVX512F 并且 NPY_SIMD 为真且小于 512，则包含 emulate_maskop.h 头文件


#if NPY_SIMD
    #include "intdiv.h"
#endif

// 如果 NPY_SIMD 为真，则包含 intdiv.h 头文件


/**
 * Some SIMD extensions currently(AVX2, AVX512F) require (de facto)
 * a maximum number of strides sizes when dealing with non-contiguous memory access.
 *
 * Therefore the following functions must be used to check the maximum
 * acceptable limit of strides before using any of non-contiguous load/store intrinsics.
 *
 * For instance:
 *  npy_intp ld_stride = step[0] / sizeof(float);
 *  npy_intp st_stride = step[1] / sizeof(float);
 *
 *  if (npyv_loadable_stride_f32(ld_stride) && npyv_storable_stride_f32(st_stride)) {
 *      for (;;)
 *          npyv_f32 a = npyv_loadn_f32(ld_pointer, ld_stride);
 *          // ...
 *          npyv_storen_f32(st_pointer, st_stride, a);
 *  }
 *  else {
 *      for (;;)
 *          // C scalars
 *  }
 */

// 以下是一段注释，描述了一些 SIMD 扩展（如 AVX2, AVX512F）在处理非连续内存访问时的最大步长限制，以及使用非连续加载/存储指令前必须使用的函数。


#ifndef NPY_SIMD_MAXLOAD_STRIDE32
    #define NPY_SIMD_MAXLOAD_STRIDE32 0
#endif
#ifndef NPY_SIMD_MAXSTORE_STRIDE32
    #define NPY_SIMD_MAXSTORE_STRIDE32 0
#endif
#ifndef NPY_SIMD_MAXLOAD_STRIDE64
    #define NPY_SIMD_MAXLOAD_STRIDE64 0
#endif
#ifndef NPY_SIMD_MAXSTORE_STRIDE64
    #define NPY_SIMD_MAXSTORE_STRIDE64 0
#endif

// 如果未定义这些宏，则定义它们并初始化为 0，用于限制 SIMD 加载和存储的最大步长


#define NPYV_IMPL_MAXSTRIDE(SFX, MAXLOAD, MAXSTORE) \
    NPY_FINLINE int npyv_loadable_stride_##SFX(npy_intp stride) \
    { return MAXLOAD > 0 ? llabs(stride) <= MAXLOAD : 1; } \
    NPY_FINLINE int npyv_storable_stride_##SFX(npy_intp stride) \
    { return MAXSTORE > 0 ? llabs(stride) <= MAXSTORE : 1; }

// 定义一个宏，生成两个内联函数，用于检查给定步长是否在指定的最大加载或存储步长范围内


#if NPY_SIMD
    NPYV_IMPL_MAXSTRIDE(u32, NPY_SIMD_MAXLOAD_STRIDE32, NPY_SIMD_MAXSTORE_STRIDE32)
    NPYV_IMPL_MAXSTRIDE(s32, NPY_SIMD_MAXLOAD_STRIDE32, NPY_SIMD_MAXSTORE_STRIDE32)
    NPYV_IMPL_MAXSTRIDE(f32, NPY_SIMD_MAXLOAD_STRIDE32, NPY_SIMD_MAXSTORE_STRIDE32)
    NPYV_IMPL_MAXSTRIDE(u64, NPY_SIMD_MAXLOAD_STRIDE64, NPY_SIMD_MAXSTORE_STRIDE64)
    NPYV_IMPL_MAXSTRIDE(s64, NPY_SIMD_MAXLOAD_STRIDE64, NPY_SIMD_MAXSTORE_STRIDE64)
#endif

// 如果 NPY_SIMD 为真，则实例化各种数据类型（u32, s32, f32, u64, s64）的加载和存储步长检查函数


#if NPY_SIMD_F64
    NPYV_IMPL_MAXSTRIDE(f64, NPY_SIMD_MAXLOAD_STRIDE64, NPY_SIMD_MAXSTORE_STRIDE64)
#endif

// 如果支持双精度 SIMD（NPY_SIMD_F64 为真），则实例化 f64 类型的加载和存储步长检查函数


#ifdef __cplusplus
}
#endif
#endif // _NPY_SIMD_H_

// 如果是 C++ 环境，则关闭外部 C 链接，并结束头文件保护符 `#ifndef _NPY_SIMD_H_`
```