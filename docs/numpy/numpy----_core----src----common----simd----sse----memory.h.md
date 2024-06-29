# `.\numpy\numpy\_core\src\common\simd\sse\memory.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_MEMORY_H
#define _NPY_SIMD_SSE_MEMORY_H

#include "misc.h"

/***************************
 * load/store
 ***************************/

// stream load
#ifdef NPY_HAVE_SSE41
    #define npyv__loads(PTR) _mm_stream_load_si128((__m128i *)(PTR))
#else
    #define npyv__loads(PTR) _mm_load_si128((const __m128i *)(PTR))
#endif

// 定义 SSE 内存操作的模板宏，参数为数据类型 CTYPE 和后缀 SFX
#define NPYV_IMPL_SSE_MEM_INT(CTYPE, SFX)                                    \
    // 加载未对齐的数据并转换为 SSE 寄存器格式
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)                 \
    { return _mm_loadu_si128((const __m128i*)ptr); }                         \
    // 加载对齐的数据并转换为 SSE 寄存器格式
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)                \
    { return _mm_load_si128((const __m128i*)ptr); }                          \
    // 使用 stream 方式加载数据，根据是否支持 SSE4.1 决定具体实现
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)                \
    { return npyv__loads(ptr); }                                             \
    // 加载 64 位数据的低位部分并转换为 SSE 寄存器格式
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)                \
    { return _mm_loadl_epi64((const __m128i*)ptr); }                         \
    // 存储 SSE 寄存器数据到未对齐的内存位置
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)            \
    { _mm_storeu_si128((__m128i*)ptr, vec); }                                \
    // 存储 SSE 寄存器数据到对齐的内存位置
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_store_si128((__m128i*)ptr, vec); }                                 \
    // 使用 stream 方式存储 SSE 寄存器数据到内存位置
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_stream_si128((__m128i*)ptr, vec); }                                \
    // 存储 SSE 寄存器数据的低位部分到内存位置
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_storel_epi64((__m128i *)ptr, vec); }                               \
    // 存储 SSE 寄存器数据的高位部分到内存位置，使用 _mm_unpackhi_epi64 实现
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_storel_epi64((__m128i *)ptr, _mm_unpackhi_epi64(vec, vec)); }

// 各种数据类型的 SSE 内存操作宏的具体实现
NPYV_IMPL_SSE_MEM_INT(npy_uint8,  u8)
NPYV_IMPL_SSE_MEM_INT(npy_int8,   s8)
NPYV_IMPL_SSE_MEM_INT(npy_uint16, u16)
NPYV_IMPL_SSE_MEM_INT(npy_int16,  s16)
NPYV_IMPL_SSE_MEM_INT(npy_uint32, u32)
NPYV_IMPL_SSE_MEM_INT(npy_int32,  s32)
NPYV_IMPL_SSE_MEM_INT(npy_uint64, u64)
NPYV_IMPL_SSE_MEM_INT(npy_int64,  s64)

// unaligned load
#define npyv_load_f32 _mm_loadu_ps
#define npyv_load_f64 _mm_loadu_pd
// aligned load
#define npyv_loada_f32 _mm_load_ps
#define npyv_loada_f64 _mm_load_pd
// load lower part
// 加载浮点数低位部分并转换为 SSE 寄存器格式，使用 _mm_castsi128_ps 转换为 __m128 类型
#define npyv_loadl_f32(PTR) _mm_castsi128_ps(npyv_loadl_u32((const npy_uint32*)(PTR)))
#define npyv_loadl_f64(PTR) _mm_castsi128_pd(npyv_loadl_u32((const npy_uint32*)(PTR)))
// stream load
// 使用 stream 方式加载浮点数并转换为 SSE 寄存器格式，使用 _mm_castsi128_ps 转换为 __m128 类型
#define npyv_loads_f32(PTR) _mm_castsi128_ps(npyv__loads(PTR))
#define npyv_loads_f64(PTR) _mm_castsi128_pd(npyv__loads(PTR))
// unaligned store
#define npyv_store_f32 _mm_storeu_ps
#define npyv_store_f64 _mm_storeu_pd
// aligned store
#define npyv_storea_f32 _mm_store_ps
#define npyv_storea_f64 _mm_store_pd
// stream store
#define npyv_stores_f32 _mm_stream_ps
#define npyv_stores_f64 _mm_stream_pd

#endif // _NPY_SIMD_SSE_MEMORY_H
// 存储向量低部分为 32 位整数
#define npyv_storel_f32(PTR, VEC) _mm_storel_epi64((__m128i*)(PTR), _mm_castps_si128(VEC));

// 存储向量低部分为 64 位双精度浮点数
#define npyv_storel_f64(PTR, VEC) _mm_storel_epi64((__m128i*)(PTR), _mm_castpd_si128(VEC));

// 存储向量高部分为 32 位单精度浮点数
#define npyv_storeh_f32(PTR, VEC) npyv_storeh_u32((npy_uint32*)(PTR), _mm_castps_si128(VEC))

// 存储向量高部分为 64 位双精度浮点数
#define npyv_storeh_f64(PTR, VEC) npyv_storeh_u32((npy_uint32*)(PTR), _mm_castpd_si128(VEC))

/***************************
 * 非连续加载
 ***************************/

//// 32 位整数加载
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    // 将首地址的整数加载到 xmm 寄存器中
    __m128i a = _mm_cvtsi32_si128(*ptr);
#ifdef NPY_HAVE_SSE41
    // 使用 SSE4.1 插入指令加载后续整数到 xmm 寄存器
    a = _mm_insert_epi32(a, ptr[stride],   1);
    a = _mm_insert_epi32(a, ptr[stride*2], 2);
    a = _mm_insert_epi32(a, ptr[stride*3], 3);
#else
    // 使用非 SSE4.1 插入方法加载后续整数到 xmm 寄存器
    __m128i a1 = _mm_cvtsi32_si128(ptr[stride]);
    __m128i a2 = _mm_cvtsi32_si128(ptr[stride*2]);
    __m128i a3 = _mm_cvtsi32_si128(ptr[stride*3]);
    a = _mm_unpacklo_epi32(a, a1);
    a = _mm_unpacklo_epi64(a, _mm_unpacklo_epi32(a2, a3));
#endif
    return a;
}

// 无符号 32 位整数加载，实际调用有符号加载函数
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{ return npyv_loadn_s32((const npy_int32*)ptr, stride); }

// 单精度浮点数加载，转换整数加载函数返回结果为浮点数
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return _mm_castsi128_ps(npyv_loadn_s32((const npy_int32*)ptr, stride)); }

//// 64 位双精度浮点数加载
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return _mm_loadh_pd(npyv_loadl_f64(ptr), ptr + stride); }

// 无符号 64 位整数加载，实际调用双精度加载函数
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return _mm_castpd_si128(npyv_loadn_f64((const double*)ptr, stride)); }

// 有符号 64 位整数加载，实际调用双精度加载函数
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return _mm_castpd_si128(npyv_loadn_f64((const double*)ptr, stride)); }

//// 64 位加载，步长为 32 位整数
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{
    // 加载低部分双精度浮点数到 xmm 寄存器
    __m128d r = _mm_loadh_pd(
        npyv_loadl_f64((const double*)ptr), (const double*)(ptr + stride)
    );
    return _mm_castpd_ps(r);  // 转换为单精度浮点数返回
}

// 无符号 32 位整数加载，实际调用单精度加载函数
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return _mm_castps_si128(npyv_loadn2_f32((const float*)ptr, stride)); }

// 有符号 32 位整数加载，实际调用单精度加载函数
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return _mm_castps_si128(npyv_loadn2_f32((const float*)ptr, stride)); }

//// 128 位加载，步长为 64 位双精度浮点数
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{ (void)stride; return npyv_load_f64(ptr); }  // 直接加载双精度浮点数

// 无符号 64 位整数加载，实际调用双精度加载函数
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ (void)stride; return npyv_load_u64(ptr); }

// 有符号 64 位整数加载，实际调用双精度加载函数
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ (void)stride; return npyv_load_s64(ptr); }

/***************************
 * 非连续存储
 ***************************/

//// 32 位整数存储
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{
    // 存储 xmm 寄存器中整数的低部分到目标地址
    ptr[stride * 0] = _mm_cvtsi128_si32(a);
#ifdef NPY_HAVE_SSE41
    # 将寄存器 `a` 中的第 1 个整数提取并存储到指针数组 `ptr` 的指定位置
    ptr[stride * 1] = _mm_extract_epi32(a, 1);
    # 将寄存器 `a` 中的第 2 个整数提取并存储到指针数组 `ptr` 的指定位置
    ptr[stride * 2] = _mm_extract_epi32(a, 2);
    # 将寄存器 `a` 中的第 3 个整数提取并存储到指针数组 `ptr` 的指定位置
    ptr[stride * 3] = _mm_extract_epi32(a, 3);
//// 64
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{
    // 使用 SSE2 指令将双精度浮点数向量 a 存储到 ptr 和 ptr + stride 处
    _mm_storel_pd(ptr, a);  // 将向量 a 的低64位存储到 ptr
    _mm_storeh_pd(ptr + stride, a);  // 将向量 a 的高64位存储到 ptr + stride
}

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    // 使用 SSE2 指令将32位无符号整数向量 a 存储到 ptr 和 ptr + stride 处
    _mm_storel_pd((double*)ptr, _mm_castsi128_pd(a));  // 将向量 a 转换为双精度浮点数后存储到 ptr
    _mm_storeh_pd((double*)(ptr + stride), _mm_castsi128_pd(a));  // 将向量 a 转换为双精度浮点数后存储到 ptr + stride
}

/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);  // 断言 nlane 大于 0，确保加载的长度有效
    #ifndef NPY_HAVE_SSE41
        const short *wptr = (const short*)ptr;  // 如果没有 SSE4.1 支持，将 ptr 视为 short 指针
    #endif
    const __m128i vfill = npyv_setall_s32(fill);  // 使用 SSE2 指令生成填充值 fill 的向量 vfill
    __m128i a;
    switch(nlane) {
        case 2:
            a = _mm_castpd_si128(
                _mm_loadl_pd(_mm_castsi128_pd(vfill), (double*)ptr)
            );
            break;
    #ifdef NPY_HAVE_SSE41
        case 1:
            a = _mm_insert_epi32(vfill, ptr[0], 0);  // 使用 SSE4.1 指令在 vfill 中插入第一个整数值
            break;
        case 3:
            a = _mm_loadl_epi64((const __m128i*)ptr);  // 使用 SSE2 指令加载 ptr 所指向的64位数据到向量 a
            a = _mm_insert_epi32(a, ptr[2], 2);  // 在向量 a 中的第2个位置插入 ptr[2] 的值
            a = _mm_insert_epi32(a, fill, 3);  // 在向量 a 中的第3个位置插入 fill 的值
            break;
    #else
        case 1:
            a = _mm_insert_epi16(vfill, wptr[0], 0);  // 使用 SSE2 指令在 vfill 中插入第一个短整数值
    a = _mm_insert_epi16(a, wptr[1], 1);
    # 在__m128i类型变量a中的索引1位置插入wptr数组的第2个元素，返回结果赋给a

    break;
    # 跳出当前循环或者switch语句的执行，结束当前的case分支

    case 3:
        a = _mm_loadl_epi64((const __m128i*)ptr);
        # 将ptr指向的内存中的64位数据加载到a中的低64位，高位填充为0
        a = _mm_unpacklo_epi64(a, vfill);
        # 将a中的64位数据与vfill中的数据进行交错组合，返回结果赋给a
        a = _mm_insert_epi16(a, wptr[4], 4);
        # 在a中的索引4位置插入wptr数组的第5个元素，返回结果赋给a
        a = _mm_insert_epi16(a, wptr[5], 5);
        # 在a中的索引5位置插入wptr数组的第6个元素，返回结果赋给a
        break;
    # 结束当前case分支的执行

#endif // NPY_HAVE_SSE41
    default:
        return npyv_load_s32(ptr);
    # 如果没有满足case条件的情况，返回调用npyv_load_s32(ptr)的结果，作为默认操作

#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // We use a variable marked 'volatile' to convince the compiler that
    // the entire vector is needed.
    volatile __m128i workaround = a;
    # 使用一个标记为'volatile'的变量workaround来告知编译器整个向量a是必需的
    // avoid optimizing it out
    a = _mm_or_si128(workaround, a);
    # 将workaround和a进行按位或操作，结果存储回a，以确保a不被优化掉
#endif
    return a;
    # 返回变量a作为函数的结果
// 在非常量数组指针ptr指向的地址上，根据整型元素读取数据，直到填充nlane数量的32位整型数据
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    // 断言保证加载的元素数量大于0
    assert(nlane > 0);
    // 使用指定的32位整数填充创建一个__m128i类型的向量vfill
    __m128i vfill = npyv_setall_s32(fill);
    // 如果不支持SSE4.1指令集
    #ifndef NPY_HAVE_SSE41
        // 将ptr转换为short型指针wptr
        const short *wptr = (const short*)ptr;
    #endif
    // 根据nlane的数量进行switch选择
    switch(nlane) {


这段代码的主要作用是根据传入的指针和参数，加载一定数量的32位整数数据到SSE向量中，支持非连续的部分加载。
    #ifdef NPY_HAVE_SSE41
        // 如果定义了 NPY_HAVE_SSE41 宏，则执行以下代码块
        case 3:
            // 将 ptr[stride*2] 插入到 vfill 的第 2 个位置
            vfill = _mm_insert_epi32(vfill, ptr[stride*2], 2);
        case 2:
            // 将 ptr[stride] 插入到 vfill 的第 1 个位置
            vfill = _mm_insert_epi32(vfill, ptr[stride], 1);
        case 1:
            // 将 ptr[0] 插入到 vfill 的第 0 个位置
            vfill = _mm_insert_epi32(vfill, ptr[0], 0);
            // 跳出 switch 语句
            break;
    #else
        // 如果未定义 NPY_HAVE_SSE41 宏，则执行以下代码块
        case 3:
            // 将 ptr[stride*2] 和 vfill 进行低位展开，组成新的 vfill
            vfill = _mm_unpacklo_epi32(_mm_cvtsi32_si128(ptr[stride*2]), vfill);
        case 2:
            // 将 ptr[0] 和 ptr[stride] 以及当前的 vfill 进行展开操作，组成新的 vfill
            vfill = _mm_unpacklo_epi64(_mm_unpacklo_epi32(
                _mm_cvtsi32_si128(*ptr), _mm_cvtsi32_si128(ptr[stride])
            ), vfill);
            // 跳出 switch 语句
            break;
        case 1:
            // 将 wptr[0] 插入到 vfill 的第 0 个位置
            vfill = _mm_insert_epi16(vfill, wptr[0], 0);
            // 将 wptr[1] 插入到 vfill 的第 1 个位置
            vfill = _mm_insert_epi16(vfill, wptr[1], 1);
            // 跳出 switch 语句
            break;
    #endif // NPY_HAVE_SSE41
    // 如果不匹配上述任何 case，执行默认情况
    default:
        // 调用 npyv_loadn_s32 函数加载 ptr 和 stride 所指示的数据
        return npyv_loadn_s32(ptr, stride);
    } // switch 结束
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则执行以下代码块
    volatile __m128i workaround = vfill;
    // 创建一个 volatile 的 __m128i 变量 workaround，用 vfill 初始化它
    vfill = _mm_or_si128(workaround, vfill);
    // 使用逻辑或操作符将 workaround 和 vfill 合并，并将结果赋给 vfill
#endif
    // 返回 vfill 变量的值
    return vfill;
}
// 填充剩余的通道为零
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    // 断言 nlane 大于 0
    switch(nlane) {
    case 1:
        // 当 nlane 为 1 时执行以下代码块
        return _mm_cvtsi32_si128(ptr[0]);
        // 使用 ptr[0] 创建一个 __m128i 类型的变量并返回
    case 2:;
        // 当 nlane 为 2 时执行以下代码块
        {
            // 创建一个局部的 npyv_s32 类型变量 a，并用 ptr[0] 初始化它
            npyv_s32 a = _mm_cvtsi32_si128(ptr[0]);
    #ifdef NPY_HAVE_SSE41
            // 如果定义了 NPY_HAVE_SSE41 宏，则执行以下代码块
            return _mm_insert_epi32(a, ptr[stride], 1);
            // 使用 _mm_insert_epi32 在 a 的第二个位置插入 ptr[stride]，并返回结果
    #else
            // 如果未定义 NPY_HAVE_SSE41 宏，则执行以下代码块
            return _mm_unpacklo_epi32(a, _mm_cvtsi32_si128(ptr[stride]));
            // 使用 _mm_unpacklo_epi32 将 a 和 ptr[stride] 的数据组合，并返回结果
    #endif // NPY_HAVE_SSE41
        }
    case 3:
        // 当 nlane 为 3 时执行以下代码块
        {
            // 创建一个局部的 npyv_s32 类型变量 a，并用 ptr[0] 初始化它
            npyv_s32 a = _mm_cvtsi32_si128(ptr[0]);
    #ifdef NPY_HAVE_SSE41
            // 如果定义了 NPY_HAVE_SSE41 宏，则执行以下代码块
            a = _mm_insert_epi32(a, ptr[stride], 1);
            // 使用 _mm_insert_epi32 在 a 的第二个位置插入 ptr[stride]
            a = _mm_insert_epi32(a, ptr[stride*2], 2);
            // 使用 _mm_insert_epi32 在 a 的第三个位置插入 ptr[stride*2]
            return a;
            // 返回已经填充好数据的 a
    #else
            // 如果未定义 NPY_HAVE_SSE41 宏，则执行以下代码块
            a = _mm_unpacklo_epi32(a, _mm_cvtsi32_si128(ptr[stride]));
            // 使用 _mm_unpacklo_epi32 将 a 和 ptr[stride] 的数据组合
            a = _mm_unpacklo_epi64(a, _mm_cvtsi32_si128(ptr[stride*2]));
            // 使用 _mm_unpacklo_epi64 将 a 和 ptr[stride*2] 的数据组合
            return a;
            // 返回已经填充好数据的 a
    #endif // NPY_HAVE_SSE41
        }
    default:
        // 默认情况下
        return npyv_loadn_s32(ptr, stride);
        // 调用 npyv_loadn_s32 函数并返回结果
    }
}

//// 64
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    // 断言 nlane 大于 0
    if (nlane == 1) {
        // 如果 nlane 等于 1，则执行以下代码块
        return npyv_load_till_s64(ptr, 1, fill);
        // 调用 npyv_load_till_s64 函数并返回结果
    }
    return npyv_loadn_s64(ptr, stride);
    // 调用 npyv_loadn_s64 函数并返回结果
}
// 填充剩余的通道为零
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    // 断言 nlane 大于 0
    if (nlane == 1) {
        // 如果 nlane 等于 1，则执行以下代码块
        return _mm_loadl_epi64((const __m128i*)ptr);
        // 使用 _mm_loadl_epi64 加载 ptr 所指向的数据，并返回结果
    }
    return npyv_loadn_s64(ptr, stride);
    // 调用 npyv_loadn_s64 函数并返回结果
}

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_s32 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    // 断言 nlane 大于 0
    if (nlane == 1) {
        // 如果 nlane 等于 1，则执行以下代码块
        const __m128i vfill = npyv_set_s32(0, 0, fill_lo, fill_hi);
        // 使用 npyv_set_s32 函数创建一个 __m128i 类型的 vfill 变量
        __m128i a = _mm_castpd_si128(
            _mm_loadl_pd(_mm_castsi128_pd(vfill), (double*)ptr)
        );
        // 使用 _mm_loadl_pd 和 _mm_castpd_si128 加载数据到 a 变量
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则执行以下代码块
        volatile __m128i workaround = a;
        // 创建一个 volatile 的 __m128i 变量 workaround，用 a 初始化它
        a = _mm_or_si128(workaround, a);
        // 使用逻辑或操作符将 workaround 和 a 合并，并将结果赋给 a
    #endif
        // 返回 a 变量的值
        return a;
    }
    return npyv_loadn2_s32(ptr, stride);
    // 调用 npyv_loadn2_s32 函数并返回结果
}
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    // 断言 nlane 大于 0
    if (nlane == 1) {
        // 如果 nlane 等于 1，则执行以下代码块
        return _mm_loadl_epi64((const __m128i*)ptr);
        // 使用 _mm_loadl_epi64 加载 ptr 所指向的数据，并返回结果
    }
    return npyv_loadn2_s32(ptr, stride);
    // 调用 npyv_loadn2_s32 函数并返回结果
}

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_s64 npyv_loadn2_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                                                  npy_int64 fill_lo, npy_int64 fill_hi)
{ assert(nlane > 0); (void)stride; (void)nlane; (void)fill_lo; (void)fill_hi; return npyv_load_s64(ptr); }
// 断言 nlane 大于 0，并调用 npyv_load_s64 函数并返回结果
//// 32
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    // 确保 nlane 大于 0，即至少有一个元素需要存储
    assert(nlane > 0);
    
    // 根据 nlane 的值选择不同的存储方式
    switch(nlane) {
    case 1:
        // 当 nlane 为 1 时，将 a 的低位 32 位存储到 ptr 指向的位置
        *ptr = _mm_cvtsi128_si32(a);
        break;
    case 2:
        // 当 nlane 为 2 时，将 a 的低位 64 位存储到 ptr 指向的位置
        _mm_storel_epi64((__m128i *)ptr, a);
        break;
    case 3:
        // 当 nlane 为 3 时，将 a 的低位 64 位存储到 ptr 指向的位置
        _mm_storel_epi64((__m128i *)ptr, a);
    #ifdef NPY_HAVE_SSE41
        // 如果支持 SSE4.1，则额外将 a 的第三个 32 位整数存储到 ptr 指向的位置
        ptr[2] = _mm_extract_epi32(a, 2);
    #else
        // 如果不支持 SSE4.1，则通过重新排列获取第三个 32 位整数并存储到 ptr 指向的位置
        ptr[2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 2)));
    #endif
        break;
    default:
        // 对于其他情况，调用通用的存储函数 npyv_store_s32 将 a 中的所有数据存储到 ptr 指向的位置
        npyv_store_s32(ptr, a);
    }
}

//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    // 确保 nlane 大于 0，即至少有一个元素需要存储
    assert(nlane > 0);
    
    // 当 nlane 为 1 时，将 a 的低位 64 位存储到 ptr 指向的位置
    if (nlane == 1) {
        _mm_storel_epi64((__m128i *)ptr, a);
        return;
    }
    
    // 对于其他情况，调用通用的存储函数 npyv_store_s64 将 a 中的所有数据存储到 ptr 指向的位置
    npyv_store_s64(ptr, a);
}

//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    // 确保 nlane 大于 0，即至少有一个元素需要存储
    assert(nlane > 0);
    // 调用 npyv_store_till_s64 将 a 中的数据存储到 ptr 指向的位置
    npyv_store_till_s64((npy_int64*)ptr, nlane, a);
}

//// 128-bit nlane
NPY_FINLINE void npyv_store2_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    // 确保 nlane 大于 0，即至少有一个元素需要存储
    assert(nlane > 0);
    // 调用通用的存储函数 npyv_store_s64 将 a 中的所有数据存储到 ptr 指向的位置
    npyv_store_s64(ptr, a);
}

/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    // 确保 nlane 大于 0，即至少有一个元素需要存储
    assert(nlane > 0);
    
    // 将 a 的低位 32 位整数存储到 ptr 指向的位置（根据 stride 和索引计算存储位置）
    ptr[stride*0] = _mm_cvtsi128_si32(a);
    
    // 根据 nlane 的值选择不同的存储方式
    switch(nlane) {
    case 1:
        return;
#ifdef NPY_HAVE_SSE41
    case 2:
        // 如果支持 SSE4.1，则将 a 的第二个 32 位整数存储到 ptr 指向的位置
        ptr[stride*1] = _mm_extract_epi32(a, 1);
        return;
    case 3:
        // 如果支持 SSE4.1，则将 a 的第二和第三个 32 位整数存储到 ptr 指向的位置
        ptr[stride*1] = _mm_extract_epi32(a, 1);
        ptr[stride*2] = _mm_extract_epi32(a, 2);
        return;
    default:
        // 如果支持 SSE4.1，则将 a 的第二到第四个 32 位整数存储到 ptr 指向的位置
        ptr[stride*1] = _mm_extract_epi32(a, 1);
        ptr[stride*2] = _mm_extract_epi32(a, 2);
        ptr[stride*3] = _mm_extract_epi32(a, 3);
#else
    case 2:
        // 如果不支持 SSE4.1，则通过重新排列获取第二个 32 位整数并存储到 ptr 指向的位置
        ptr[stride*1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 1)));
        return;
    case 3:
        // 如果不支持 SSE4.1，则通过重新排列获取第二和第三个 32 位整数并存储到 ptr 指向的位置
        ptr[stride*1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 1)));
        ptr[stride*2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 2)));
        return;
    default:
        // 如果不支持 SSE4.1，则通过重新排列获取第二到第四个 32 位整数并存储到 ptr 指向的位置
        ptr[stride*1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 1)));
        ptr[stride*2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 2)));
        ptr[stride*3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 3)));
#endif
    }
}

//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    // 确保 nlane 大于 0，即至少有一个元素需要存储
    assert(nlane > 0);
    
    // 当 nlane 为 1 时，将 a 的低位 64 位存储到 ptr 指向的位置
    if (nlane == 1) {
        _mm_storel_epi64((__m128i *)ptr, a);
        return;
    }
    
    // 调用通用的存储函数 npyv_storen_s64 将 a 中的所有数据按照指定的 stride 存储到 ptr 指向的位置
    npyv_storen_s64(ptr, stride, a);
}
//// 64-bit store over 32-bit stride
// 定义了一个函数 npyv_storen2_till_s32，用于将 64 位整数存储到 32 位步长位置
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    // 断言确保 nlane 大于 0
    assert(nlane > 0);
    // 将向量 a 的低位存储到 ptr 指向的位置
    npyv_storel_s32(ptr, a);
    // 如果 nlane 大于 1，则将向量 a 的高位存储到 ptr + stride 指向的位置
    if (nlane > 1) {
        npyv_storeh_s32(ptr + stride, a);
    }
}

//// 128-bit store over 64-bit stride
// 定义了一个函数 npyv_storen2_till_s64，用于将 128 位整数存储到 64 位步长位置
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    // 断言确保 nlane 大于 0
    assert(nlane > 0);
    // 忽略 stride 和 nlane 的值，将向量 a 存储到 ptr 指向的位置
    (void)stride; (void)nlane; npyv_store_s64(ptr, a);
}

/*****************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via casting
 *****************************************************************/
// 定义了一系列宏，实现了通过类型转换来进行部分加载/存储操作，支持 u32/f32/u64/f64 等类型

#define NPYV_IMPL_SSE_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                      \
    // 实现了加载函数 npyv_load_till_##F_SFX，用于部分加载 F_SFX 类型数据
    NPY_FINLINE npyv_##F_SFX npyv_load_till_##F_SFX                                         \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_lanetype_##F_SFX fill)         \
    {                                                                                       \
        // 定义一个联合体 pun，用于将 fill 转换成目标类型 T_SFX
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        // 返回将转换后的填充值 pun.to_##T_SFX 传递给 npyv_load_till_##T_SFX 函数的结果
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(                   \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX                       \
        ));                                                                                 \
    }                                                                                       \
    // 实现了加载函数 npyv_loadn_till_##F_SFX，用于部分加载 F_SFX 类型数据，支持步长操作
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill)                                                            \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        // 使用联合体 pun 将 fill 转换为 from_##F_SFX 类型的数据
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun.to_##T_SFX               \
        ));                                                                                 \
    }                                                                                       \
    // 以非常规方式加载 F_SFX 类型的向量，直接从指针 ptr 中读取 nlane 个元素
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        // 将加载到的数据重新解释为 T_SFX 类型的向量
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    // 以非常规方式加载 F_SFX 类型的向量，并且在末尾填充零值
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        // 将加载到的数据重新解释为 T_SFX 类型的向量
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    // 以非常规方式存储 F_SFX 类型的向量数据
    NPY_FINLINE void npyv_store_till_##F_SFX                                                \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        // 将 F_SFX 类型的向量 a 重新解释为 T_SFX 类型后存储到 ptr 指向的内存中
        npyv_store_till_##T_SFX(                                                            \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \

        npyv_store_till_##F_SFX  函数的说明如上ultimate
    }                                                                                       \
    // 定义 NPY_FINLINE 宏展开后的函数 npyv_storen_till_##F_SFX
    NPY_FINLINE void npyv_storen_till_##F_SFX                                               \
    // 函数签名：将 npyv_lanetype_##F_SFX 类型的向量 a 存储到 ptr 指向的内存中，步长为 stride，处理的向量数为 nlane
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        // 调用宏 npyv_storen_till_##T_SFX 将 a 向量重新解释为 npyv_##T_SFX 类型，然后存储到 ptr 中
        npyv_storen_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }
// 定义宏 NPYV_IMPL_SSE_REST_PARTIAL_TYPES，用于生成特定数据类型和操作类型的函数实现
NPYV_IMPL_SSE_REST_PARTIAL_TYPES(u32, s32)
// 生成 npyv_u32 类型的 SSE 函数实现，处理 s32 类型操作
NPYV_IMPL_SSE_REST_PARTIAL_TYPES(f32, s32)
// 生成 npyv_f32 类型的 SSE 函数实现，处理 s32 类型操作
NPYV_IMPL_SSE_REST_PARTIAL_TYPES(u64, s64)
// 生成 npyv_u64 类型的 SSE 函数实现，处理 s64 类型操作
NPYV_IMPL_SSE_REST_PARTIAL_TYPES(f64, s64)
// 生成 npyv_f64 类型的 SSE 函数实现，处理 s64 类型操作

// 定义宏 NPYV_IMPL_SSE_REST_PARTIAL_TYPES_PAIR，用于生成两种数据类型组合的函数实现
#define NPYV_IMPL_SSE_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                 \
    // 声明 npyv_load2_till_##F_SFX 函数，加载指定类型的数据直到指定数量，并填充指定的低位和高位值
    NPY_FINLINE npyv_##F_SFX npyv_load2_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                                     \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        // 定义联合体 pun 用于类型转换
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        // 定义联合体变量 pun_lo 和 pun_hi，分别填充低位和高位的值
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        // 返回根据填充值转换后的结果，使用 npyv_load2_till_##T_SFX 函数加载数据
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun_lo.to_##T_SFX, pun_hi.to_##T_SFX \
        ));                                                                                 \
    }                                                                                       \
    // 声明 npyv_loadn2_till_##F_SFX 函数，加载指定类型的数据直到指定数量，带有步长，并填充指定的低位和高位值
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_till_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_till_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun_lo.to_##T_SFX,           \
            pun_hi.to_##T_SFX                                                               \
        ));                                                                                 \
    }                                                                                       \

这段代码定义了一个函数，用于加载并重新解释 SIMD 向量数据类型。根据不同的输入类型转换加载，以及指定的填充值。


    NPY_FINLINE npyv_##F_SFX npyv_load2_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \

这段代码定义了一个函数，用于加载指定数量的 SIMD 向量数据类型并清零剩余部分。


    NPY_FINLINE npyv_##F_SFX npyv_loadn2_tillz_##F_SFX                                      \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_tillz_##T_SFX(                \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \

这段代码定义了一个函数，用于加载带有步长的 SIMD 向量数据类型，并清零剩余部分。


    NPY_FINLINE void npyv_store2_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \

这段代码定义了一个函数，用于存储指定数量的 SIMD 向量数据类型。
    {
        # 调用宏 npyv_store2_till_##T_SFX，用于将向量 a 的数据按类型 T_SFX 解释后存储到指针 ptr 所指向的内存中，处理 nlane 个元素
        npyv_store2_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    
        # 定义内联函数 npyv_storen2_till_##F_SFX，用于将向量 a 的数据按类型 F_SFX 解释后存储到 ptr 指向的内存中，处理步长为 stride，nlane 个元素
        NPY_FINLINE void npyv_storen2_till_##F_SFX
        (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)
        {
            # 调用宏 npyv_storen2_till_##T_SFX，将向量 a 的数据按类型 T_SFX 解释后存储到 ptr 指向的内存中，处理步长为 stride，nlane 个元素
            npyv_storen2_till_##T_SFX(
                (npyv_lanetype_##T_SFX *)ptr, stride, nlane,
                npyv_reinterpret_##T_SFX##_##F_SFX(a)
            );
        }
    }
// 定义一个宏，用于生成特定类型的 SIMD 操作函数，实现加载和存储的互相关联
#define NPYV_IMPL_SSE_REST_PARTIAL_TYPES_PAIR(TYPE1, TYPE2) \
    // 实现类型为 TYPE1 和 TYPE2 的 SSE 操作的特定函数

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/

// 两个通道的内存交织加载和连续存储

// 定义一个宏，用于实现给定类型的内存交织加载和存储操作
#define NPYV_IMPL_SSE_MEM_INTERLEAVE(SFX, ZSFX) \
    // 内存交织加载操作，将连续存储的数据分别加载到两个向量中
    NPY_FINLINE npyv_##ZSFX##x2 npyv_zip_##ZSFX(npyv_##SFX, npyv_##SFX); \
    // 内存解交织操作，将两个向量的数据合并为一个向量
    NPY_FINLINE npyv_##ZSFX##x2 npyv_unzip_##ZSFX(npyv_##SFX, npyv_##SFX); \
    // 加载两个连续存储的数据块，分别存储到两个向量中
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2( \
        const npyv_lanetype_##SFX *ptr \
    ) { \
        return npyv_unzip_##ZSFX( \
            npyv_load_##SFX(ptr), npyv_load_##SFX(ptr+npyv_nlanes_##SFX) \
        ); \
    } \
    // 将两个向量的数据分别存储到连续存储块中
    NPY_FINLINE void npyv_store_##SFX##x2( \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v \
    ) { \
        npyv_##SFX##x2 zip = npyv_zip_##ZSFX(v.val[0], v.val[1]); \
        npyv_store_##SFX(ptr, zip.val[0]); \
        npyv_store_##SFX(ptr + npyv_nlanes_##SFX, zip.val[1]); \
    }

// 以下是根据上述宏定义生成的具体类型的内存交织加载和存储操作函数

NPYV_IMPL_SSE_MEM_INTERLEAVE(u8, u8)
NPYV_IMPL_SSE_MEM_INTERLEAVE(s8, u8)
NPYV_IMPL_SSE_MEM_INTERLEAVE(u16, u16)
NPYV_IMPL_SSE_MEM_INTERLEAVE(s16, u16)
NPYV_IMPL_SSE_MEM_INTERLEAVE(u32, u32)
NPYV_IMPL_SSE_MEM_INTERLEAVE(s32, u32)
NPYV_IMPL_SSE_MEM_INTERLEAVE(u64, u64)
NPYV_IMPL_SSE_MEM_INTERLEAVE(s64, u64)
NPYV_IMPL_SSE_MEM_INTERLEAVE(f32, f32)
NPYV_IMPL_SSE_MEM_INTERLEAVE(f64, f64)

/*********************************
 * Lookup table
 *********************************/

// 使用向量作为索引查找包含 32 个 float32 元素的表格

// 返回根据索引 idx 查找的 float32 类型的表格中的值组成的向量
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{
    // 提取 idx 向量的第一个元素作为整数索引 i0
    const int i0 = _mm_cvtsi128_si32(idx);
#ifdef NPY_HAVE_SSE41
    // 使用 SSE4.1 的函数从 idx 向量中提取其他整数索引 i1, i2, i3
    const int i1 = _mm_extract_epi32(idx, 1);
    const int i2 = _mm_extract_epi32(idx, 2);
    const int i3 = _mm_extract_epi32(idx, 3);
#else
    // 使用 SSE2 的函数从 idx 向量中提取其他整数索引 i1, i2, i3
    const int i1 = _mm_extract_epi16(idx, 2);
    const int i2 = _mm_extract_epi16(idx, 4);
    const int i3 = _mm_extract_epi16(idx, 6);
#endif
    // 返回根据索引从表格中取出的四个 float32 值组成的向量
    return npyv_set_f32(table[i0], table[i1], table[i2], table[i3]);
}

// 返回根据索引 idx 查找的 uint32 类型的表格中的值组成的向量
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float*)table, idx)); }

// 返回根据索引 idx 查找的 int32 类型的表格中的值组成的向量
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
// 使用给定的索引向表中查找对应的双精度浮点数，返回一个双精度向量
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{
    // 从 idx 中提取第一个索引 i0，并转换为整数
    const int i0 = _mm_cvtsi128_si32(idx);
#ifdef NPY_HAVE_SSE41
    // 如果支持 SSE4.1，则从 idx 中提取第二个索引 i1（32位），否则从 idx 中提取第二个索引 i1（16位）
    const int i1 = _mm_extract_epi32(idx, 2);
#else
    const int i1 = _mm_extract_epi16(idx, 4);
#endif
    // 使用 i0 和 i1 作为索引，从表中获取两个双精度浮点数，并将它们组合成一个双精度向量返回
    return npyv_set_f64(table[i0], table[i1]);
}

// 使用给定的表和索引查找表中的64位无符号整数，并将结果重新解释为双精度浮点数向量返回
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }

// 使用给定的表和索引查找表中的64位有符号整数，并将结果重新解释为双精度浮点数向量返回
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }
```