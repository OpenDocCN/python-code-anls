# `.\numpy\numpy\_core\src\common\simd\avx2\memory.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#include "misc.h"

#ifndef _NPY_SIMD_AVX2_MEMORY_H
#define _NPY_SIMD_AVX2_MEMORY_H

/***************************
 * load/store
 ***************************/

// 定义 AVX2 模式下不同整数类型的加载和存储操作

#define NPYV_IMPL_AVX2_MEM_INT(CTYPE, SFX)                                   \
    // 加载未对齐的整型数据到 AVX2 向量
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)                 \
    { return _mm256_loadu_si256((const __m256i*)ptr); }                      \
    // 加载对齐的整型数据到 AVX2 向量
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)                \
    { return _mm256_load_si256((const __m256i*)ptr); }                       \
    // 流式加载整型数据到 AVX2 向量
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)                \
    { return _mm256_stream_load_si256((const __m256i*)ptr); }                \
    // 加载整型数据的低128位到 AVX2 向量
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)                \
    { return _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)ptr)); } \
    // 存储 AVX2 向量数据到未对齐的整型地址
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)            \
    { _mm256_storeu_si256((__m256i*)ptr, vec); }                             \
    // 存储 AVX2 向量数据到对齐的整型地址
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm256_store_si256((__m256i*)ptr, vec); }                              \
    // 流式存储 AVX2 向量数据到整型地址
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm256_stream_si256((__m256i*)ptr, vec); }                             \
    // 存储 AVX2 向量低128位数据到未对齐的整型地址
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_storeu_si128((__m128i*)(ptr), _mm256_castsi256_si128(vec)); }      \
    // 存储 AVX2 向量数据的高128位到未对齐的整型地址
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_storeu_si128((__m128i*)(ptr), _mm256_extracti128_si256(vec, 1)); }

// 使用宏定义来展开 AVX2 不同类型整数加载和存储函数
NPYV_IMPL_AVX2_MEM_INT(npy_uint8,  u8)
NPYV_IMPL_AVX2_MEM_INT(npy_int8,   s8)
NPYV_IMPL_AVX2_MEM_INT(npy_uint16, u16)
NPYV_IMPL_AVX2_MEM_INT(npy_int16,  s16)
NPYV_IMPL_AVX2_MEM_INT(npy_uint32, u32)
NPYV_IMPL_AVX2_MEM_INT(npy_int32,  s32)
NPYV_IMPL_AVX2_MEM_INT(npy_uint64, u64)
NPYV_IMPL_AVX2_MEM_INT(npy_int64,  s64)

// 未对齐加载单精度浮点数
#define npyv_load_f32 _mm256_loadu_ps
// 对齐加载单精度浮点数
#define npyv_loada_f32 _mm256_load_ps
// 流式加载单精度浮点数
#define npyv_loads_f32(PTR) \
    _mm256_castsi256_ps(_mm256_stream_load_si256((const __m256i*)(PTR)))
// 加载单精度浮点数低部分
#define npyv_loadl_f32(PTR) _mm256_castps128_ps256(_mm_loadu_ps(PTR))
// 未对齐存储单精度浮点数
#define npyv_store_f32 _mm256_storeu_ps
// 对齐存储单精度浮点数
#define npyv_storea_f32 _mm256_store_ps
// 流式存储单精度浮点数
#define npyv_stores_f32 _mm256_stream_ps
// 存储单精度浮点数低部分
#define npyv_storel_f32(PTR, VEC) _mm_storeu_ps(PTR, _mm256_castps256_ps128(VEC))

// 未对齐加载双精度浮点数
#define npyv_load_f64 _mm256_loadu_pd
// 对齐加载双精度浮点数
#define npyv_loada_f64 _mm256_load_pd
// 流式加载双精度浮点数
#define npyv_loads_f64(PTR) \
    _mm256_castsi256_pd(_mm256_stream_load_si256((const __m256i*)(PTR)))
// 加载双精度浮点数低部分
#define npyv_loadl_f64(PTR) _mm256_castpd128_pd256(_mm_loadu_pd(PTR))
// 未对齐存储双精度浮点数
#define npyv_store_f64 _mm256_storeu_pd
// 对齐存储双精度浮点数
#define npyv_storea_f64 _mm256_store_pd
// 流式存储双精度浮点数
#define npyv_stores_f64 _mm256_stream_pd
#define npyv_storel_f64(PTR, VEC) _mm_storeu_pd(PTR, _mm256_castpd256_pd128(VEC))
// 将 AVX2 双精度向量 VEC 的低128位存储到 PTR 指向的内存地址中

#define npyv_storeh_f32(PTR, VEC) _mm_storeu_ps(PTR, _mm256_extractf128_ps(VEC, 1))
// 将 AVX2 单精度向量 VEC 的高128位存储到 PTR 指向的内存地址中

#define npyv_storeh_f64(PTR, VEC) _mm_storeu_pd(PTR, _mm256_extractf128_pd(VEC, 1))
// 将 AVX2 双精度向量 VEC 的高128位存储到 PTR 指向的内存地址中

/***************************
 * Non-contiguous Load
 ***************************/

//// 32
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    // 设置步长为0到7的整数向量 steps
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    // 使用步长向量 steps 和给定的 stride 计算索引 idx
    const __m256i idx = _mm256_mullo_epi32(_mm256_set1_epi32((int)stride), steps);
    // 使用索引 idx 从内存地址 ptr 加载32位整数数据，并返回 AVX2 32位无符号整数向量
    return _mm256_i32gather_epi32((const int*)ptr, idx, 4);
}

NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    // 调用 npyv_loadn_u32 将结果强制转换为有符号整数向量
    return npyv_loadn_u32((const npy_uint32*)ptr, stride);
}

NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{
    // 调用 npyv_loadn_u32 将结果强制转换为单精度浮点数向量
    return _mm256_castsi256_ps(npyv_loadn_u32((const npy_uint32*)ptr, stride));
}

//// 64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{
    // 加载 ptr 指向的双精度数据到 a0 和 a2
    __m128d a0 = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)ptr));
    __m128d a2 = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*2)));
    // 加载 ptr + stride 和 ptr + stride*3 的双精度数据到 a01 和 a23
    __m128d a01 = _mm_loadh_pd(a0, ptr + stride);
    __m128d a23 = _mm_loadh_pd(a2, ptr + stride*3);
    // 将 a01 和 a23 合并成 AVX2 双精度向量并返回
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(a01), a23, 1);
}

NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{
    // 调用 npyv_loadn_f64 将结果强制转换为无符号64位整数向量
    return _mm256_castpd_si256(npyv_loadn_f64((const double*)ptr, stride));
}

NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{
    // 调用 npyv_loadn_f64 将结果强制转换为有符号64位整数向量
    return _mm256_castpd_si256(npyv_loadn_f64((const double*)ptr, stride));
}

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{
    // 加载 ptr 指向的单精度数据到 a0 和 a2
    __m128d a0 = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)ptr));
    __m128d a2 = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*2)));
    // 加载 ptr + stride 和 ptr + stride*3 的双精度数据到 a01 和 a23
    __m128d a01 = _mm_loadh_pd(a0, (const double*)(ptr + stride));
    __m128d a23 = _mm_loadh_pd(a2, (const double*)(ptr + stride*3));
    // 将 a01 和 a23 合并成 AVX2 单精度向量并返回
    return _mm256_castpd_ps(_mm256_insertf128_pd(_mm256_castpd128_pd256(a01), a23, 1));
}

NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{
    // 调用 npyv_loadn2_f32 将结果强制转换为无符号32位整数向量
    return _mm256_castps_si256(npyv_loadn2_f32((const float*)ptr, stride));
}

NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{
    // 调用 npyv_loadn2_f32 将结果强制转换为有符号32位整数向量
    return _mm256_castps_si256(npyv_loadn2_f32((const float*)ptr, stride));
}

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{
    // 调用 npyv_loadl_f64 加载低64位数据到 AVX2 双精度向量 a
    __m256d a = npyv_loadl_f64(ptr);
    // 加载 ptr + stride 的双精度数据到 AVX 单精度向量并将其插入 a 的第二个128位
    return _mm256_insertf128_pd(a, _mm_loadu_pd(ptr + stride), 1);
}

NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{
    // 调用 npyv_loadn2_f64 将结果强制转换为无符号64位整数向量
    return _mm256_castpd_si256(npyv_loadn2_f64((const double*)ptr, stride));
}

NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{
    // 调用 npyv_loadn2_f64 将结果强制转换为有符号64位整数向量
    return _mm256_castpd_si256(npyv_loadn2_f64((const double*)ptr, stride));
}
/***************************
 * Non-contiguous Store
 ***************************/

//// 32
// 将 256 位整数向量 a 存储到 npy_int32 类型指针数组 ptr 中，按照指定的步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{
    // 将 256 位整数向量 a 转换为两个 128 位整数向量 a0 和 a1
    __m128i a0 = _mm256_castsi256_si128(a);
    __m128i a1 = _mm256_extracti128_si256(a, 1);
    
    // 分别将 a0 的各分量存储到 ptr 中不同的位置，根据步长 stride 进行间隔存储
    ptr[stride * 0] = _mm_cvtsi128_si32(a0);
    ptr[stride * 1] = _mm_extract_epi32(a0, 1);
    ptr[stride * 2] = _mm_extract_epi32(a0, 2);
    ptr[stride * 3] = _mm_extract_epi32(a0, 3);
    // 同样地，将 a1 的各分量存储到 ptr 中不同的位置，根据步长 stride 进行间隔存储
    ptr[stride * 4] = _mm_cvtsi128_si32(a1);
    ptr[stride * 5] = _mm_extract_epi32(a1, 1);
    ptr[stride * 6] = _mm_extract_epi32(a1, 2);
    ptr[stride * 7] = _mm_extract_epi32(a1, 3);
}

// 将 256 位无符号整数向量 a 存储到 npy_uint32 类型指针数组 ptr 中，按照指定的步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ 
    // 调用 npyv_storen_s32 函数，将 a 视为 npyv_s32 处理，转为 npy_int32* 类型存储
    npyv_storen_s32((npy_int32*)ptr, stride, a); 
}

// 将 256 位单精度浮点数向量 a 存储到 float 类型指针数组 ptr 中，按照指定的步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ 
    // 调用 npyv_storen_s32 函数，将 a 转为 npyv_s32 处理后存储，转为 npy_int32* 类型存储
    npyv_storen_s32((npy_int32*)ptr, stride, _mm256_castps_si256(a)); 
}

//// 64
// 将 256 位双精度浮点数向量 a 存储到 double 类型指针数组 ptr 中，按照指定的步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{
    // 将 256 位双精度浮点数向量 a 转换为两个 128 位双精度浮点数向量 a0 和 a1
    __m128d a0 = _mm256_castpd256_pd128(a);
    __m128d a1 = _mm256_extractf128_pd(a, 1);
    
    // 将 a0 的低64位存储到 ptr 中的不同位置，根据步长 stride 进行间隔存储
    _mm_storel_pd(ptr + stride * 0, a0);
    _mm_storeh_pd(ptr + stride * 1, a0);
    // 将 a1 的低64位存储到 ptr 中的不同位置，根据步长 stride 进行间隔存储
    _mm_storel_pd(ptr + stride * 2, a1);
    _mm_storeh_pd(ptr + stride * 3, a1);
}

// 将 256 位无符号整数向量 a 存储到 npy_uint64 类型指针数组 ptr 中，按照指定的步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ 
    // 调用 npyv_storen_f64 函数，将 a 视为 npyv_f64 处理，转为 double* 类型存储
    npyv_storen_f64((double*)ptr, stride, _mm256_castsi256_pd(a)); 
}

// 将 256 位有符号整数向量 a 存储到 npy_int64 类型指针数组 ptr 中，按照指定的步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ 
    // 调用 npyv_storen_f64 函数，将 a 视为 npyv_f64 处理，转为 double* 类型存储
    npyv_storen_f64((double*)ptr, stride, _mm256_castsi256_pd(a)); 
}

//// 64-bit store over 32-bit stride
// 将 256 位无符号整数向量 a 存储到 npy_uint32 类型指针数组 ptr 中，按照指定的 32 位步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    // 将 256 位无符号整数向量 a 转换为两个 128 位双精度浮点数向量 a0 和 a1
    __m128d a0 = _mm256_castpd256_pd128(_mm256_castsi256_pd(a));
    __m128d a1 = _mm256_extractf128_pd(_mm256_castsi256_pd(a), 1);
    
    // 将 a0 的低64位存储到 ptr 中的不同位置，根据步长 stride 进行间隔存储
    _mm_storel_pd((double*)ptr, a0);
    _mm_storeh_pd((double*)(ptr + stride), a0);
    // 将 a1 的低64位存储到 ptr 中的不同位置，根据步长 stride 进行间隔存储
    _mm_storel_pd((double*)(ptr + stride*2), a1);
    _mm_storeh_pd((double*)(ptr + stride*3), a1);
}

// 将 256 位有符号整数向量 a 存储到 npy_int32 类型指针数组 ptr 中，按照指定的 32 位步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ 
    // 调用 npyv_storen2_u32 函数，将 a 视为 npyv_u32 处理，转为 npy_uint32* 类型存储
    npyv_storen2_u32((npy_uint32*)ptr, stride, a); 
}

// 将 256 位单精度浮点数向量 a 存储到 float 类型指针数组 ptr 中，按照指定的 32 位步长 stride 进行非连续存储
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ 
    // 调用 npyv_storen2_u32 函数，将 a 转为 npyv_u32 处理后存储，转为 npy_uint32* 类型存储
    npyv_storen2_u32((npy_uint32*)ptr, stride, _mm256_castps_si256(a)); 
}

//// 128-bit store over 64-bit stride
// 将 256 位无符号整数向量 a
//// 64
// 加载直到填充 64 位有符号整数向量
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    // 断言：向量长度大于 0
    assert(nlane > 0);
    // 设置一个包含 fill 值的 AVX 寄存器
    const __m256i vfill = npyv_setall_s64(fill);
    // 设置步长寄存器为 (0, 1, 2, 3)
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    // 设置向量长度寄存器
    __m256i vnlane  = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    // 比较向量长度与步长，生成掩码
    __m256i mask    = _mm256_cmpgt_epi64(vnlane, steps);
    // 使用掩码从内存中加载数据到 AVX 寄存器
    __m256i payload = _mm256_maskload_epi64((const long long*)ptr, mask);
    // 根据掩码进行混合，用 fill 值填充未加载的部分
    __m256i ret     = _mm256_blendv_epi8(vfill, payload, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果启用部分加载保护，使用 volatile 变量执行一个空操作
    volatile __m256i workaround = ret;
    // 或运算，确保在部分加载情况下返回正确的结果
    ret = _mm256_or_si256(workaround, ret);
#endif
    // 返回加载的 AVX 寄存器
    return ret;
}

// 填充零直到剩余的向量长度
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    // 断言：向量长度大于 0
    assert(nlane > 0);
    // 设置步长寄存器为 (0, 1, 2, 3)
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    // 设置向量长度寄存器
    __m256i vnlane  = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    // 比较向量长度与步长，生成掩码
    __m256i mask    = _mm256_cmpgt_epi64(vnlane, steps);
    // 使用掩码从内存中加载数据到 AVX 寄存器
    __m256i ret     = _mm256_maskload_epi64((const long long*)ptr, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果启用部分加载保护，使用 volatile 变量执行一个空操作
    volatile __m256i workaround = ret;
    // 或运算，确保在部分加载情况下返回正确的结果
    ret = _mm256_or_si256(workaround, ret);
#endif
    // 返回加载的 AVX 寄存器
    return ret;
}

//// 64-bit nlane
// 加载直到填充 64 位整数向量，扩展为两倍的长度
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    // 断言：向量长度大于 0
    assert(nlane > 0);
    // 设置包含填充值的 AVX 寄存器
    const __m256i vfill = npyv_set_s32(
        fill_lo, fill_hi, fill_lo, fill_hi,
        fill_lo, fill_hi, fill_lo, fill_hi
    );
    // 设置步长寄存器为 (0, 1, 2, 3)
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    // 设置向量长度寄存器
    __m256i vnlane  = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    // 比较向量长度与步长，生成掩码
    __m256i mask    = _mm256_cmpgt_epi64(vnlane, steps);
    // 使用掩码从内存中加载数据到 AVX 寄存器
    __m256i payload = _mm256_maskload_epi64((const long long*)ptr, mask);
    // 根据掩码进行混合，用填充值填充未加载的部分
    __m256i ret     = _mm256_blendv_epi8(vfill, payload, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则执行以下操作
    volatile __m256i workaround = ret;
    // 使用 volatile 变量 workaround 来避免编译器优化，确保正确加载数据
    ret = _mm256_or_si256(workaround, ret);
    // 将 workaround 和 ret 进行按位或操作，可能是为了处理特定的加载问题
#endif
    // 返回加载的结果向量
    return ret;
}
// 填充零到剩余的通道
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

/// 128-bit nlane
NPY_FINLINE npyv_u64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    // 断言确保通道数大于0
    assert(nlane > 0);
    // 计算掩码 m，用于部分加载时控制处理的通道
    npy_int64 m  = -((npy_int64)(nlane > 1));
    // 设置一个掩码，根据 m 来选择加载的通道
    __m256i mask = npyv_set_s64(-1, -1, m, m);
    // 使用掩码加载数据到 ret 向量
    __m256i ret  = _mm256_maskload_epi64((const long long*)ptr, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则执行以下操作
    volatile __m256i workaround = ret;
    // 使用 volatile 变量 workaround 来避免编译器优化，确保正确加载数据
    ret = _mm256_or_si256(workaround, ret);
    // 将 workaround 和 ret 进行按位或操作，可能是为了处理特定的加载问题
#endif
    // 返回加载的结果向量
    return ret;
}
// 填充指定值到剩余的通道
NPY_FINLINE npyv_u64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{
    // 设置一个向量填充值 vfill
    const __m256i vfill = npyv_set_s64(0, 0, fill_lo, fill_hi);
    // 计算掩码 m，用于部分加载时控制处理的通道
    npy_int64 m     = -((npy_int64)(nlane > 1));
    // 设置一个掩码，根据 m 来选择加载的通道
    __m256i mask    = npyv_set_s64(-1, -1, m, m);
    // 使用掩码加载数据到 payload 向量
    __m256i payload = _mm256_maskload_epi64((const long long*)ptr, mask);
    // 将指定值和加载的数据根据掩码进行混合
    __m256i ret     = _mm256_blendv_epi8(vfill, payload, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则执行以下操作
    volatile __m256i workaround = ret;
    // 使用 volatile 变量 workaround 来避免编译器优化，确保正确加载数据
    ret = _mm256_or_si256(workaround, ret);
    // 将 workaround 和 ret 进行按位或操作，可能是为了处理特定的加载问题
#endif
    // 返回加载的结果向量
    return ret;
}
/*********************************
 * Non-contiguous partial load
 *********************************/
//// 32
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    // 断言确保通道数大于0
    assert(nlane > 0);
    // 断言确保步长绝对值不超过最大加载步长
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    // 设置一个向量填充值 vfill
    const __m256i vfill = _mm256_set1_epi32(fill);
    // 设置步长向量 steps
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    // 计算加载数据的索引
    const __m256i idx   = _mm256_mullo_epi32(_mm256_set1_epi32((int)stride), steps);
    // 设置一个通道数向量 vnlane，如果通道数大于8，则设置为8，否则设置为 nlane
    __m256i vnlane      = _mm256_set1_epi32(nlane > 8 ? 8 : (int)nlane);
    // 比较通道数和步长，生成掩码
    __m256i mask        = _mm256_cmpgt_epi32(vnlane, steps);
    // 使用掩码从指针 ptr 处加载数据到 ret 向量
    __m256i ret         = _mm256_mask_i32gather_epi32(vfill, (const int*)ptr, idx, mask, 4);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则执行以下操作
    volatile __m256i workaround = ret;
    // 使用 volatile 变量 workaround 来避免编译器优化，确保正确加载数据
    ret = _mm256_or_si256(workaround, ret);
    // 将 workaround 和 ret 进行按位或操作，可能是为了处理特定的加载问题
#endif
    // 返回加载的结果向量
    return ret;
}
// 填充零到剩余的通道
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }
//// 64
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    // 断言确保通道数大于0
    assert(nlane > 0);
    // 设置一个向量填充值 vfill
    const __m256i vfill = npyv_setall_s64(fill);
    // 设置加载数据的索引 idx
    const __m256i idx   = npyv_set_s64(0, 1*stride, 2*stride, 3*stride);
    // 设置步长向量 steps
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    // 设置一个通道数向量 vnlane，如果通道数大于4，则设置为4，否则设置为 nlane
    __m256i vnlane = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    // 比较通道数和步长，生成掩码
    __m256i mask   = _mm256_cmpgt_epi64(vnlane, steps);
    // 使用掩码从指针 ptr 处加载数据到 ret 向量
    __m256i ret    = _mm256_mask_i64gather_epi64(vfill, (const long long*)ptr, idx, mask, 8);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则执行以下操作
    volatile __m256i workaround = ret;
    // 使用 volatile 变量 workaround 来避免编译器优化，确保正确加载数据
    ret = _mm256_or_si256(workaround, ret);
    // 将 workaround 和 ret 进行按位或操作，可能是为了处理特定的加载问题
#endif
    // 返回加载的结果向量
    return ret;
}
    // 创建一个 volatile 类型为 __m256i 的变量 workaround，并将 ret 的值赋给它
    volatile __m256i workaround = ret;
    // 使用 _mm256_or_si256 函数对 workaround 和 ret 进行按位或操作，并将结果赋给 ret
    ret = _mm256_or_si256(workaround, ret);
//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_s64 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{
    // 断言确保加载的向量长度大于0
    assert(nlane > 0);
    
    // 创建包含填充值的 AVX 寄存器
    const __m256i vfill = npyv_set_s32(
        fill_lo, fill_hi, fill_lo, fill_hi,
        fill_lo, fill_hi, fill_lo, fill_hi
    );
    
    // 创建包含步长的 AVX 寄存器
    const __m256i idx   = npyv_set_s64(0, 1*stride, 2*stride, 3*stride);
    
    // 创建包含步数的 AVX 寄存器
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    
    // 设置加载的向量长度
    __m256i vnlane = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    
    // 创建用于掩码比较的 AVX 寄存器
    __m256i mask   = _mm256_cmpgt_epi64(vnlane, steps);
    
    // 使用掩码从指针 ptr 处加载数据到 AVX 寄存器 ret
    __m256i ret    = _mm256_mask_i64gather_epi64(vfill, (const long long*)ptr, idx, mask, 4);
    
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果开启了部分加载保护，则进行一个额外的 volatile 操作
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    
    // 返回加载的 AVX 寄存器 ret
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn2_till_s32(ptr, stride, nlane, 0, 0); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_s64 npyv_loadn2_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                                          npy_int64 fill_lo, npy_int64 fill_hi)
{
    // 断言确保加载的向量长度大于0
    assert(nlane > 0);
    
    // 加载低64位数据到 AVX 寄存器 a
    __m256i a = npyv_loadl_s64(ptr);
    
#if defined(_MSC_VER) && defined(_M_IX86)
    // 如果是 MSVC 编译器和 x86 平台，则按照 32位 int 大小设置填充值
    __m128i fill =_mm_setr_epi32(
        (int)fill_lo, (int)(fill_lo >> 32),
        (int)fill_hi, (int)(fill_hi >> 32)
    );
#else
    // 否则，按照 64位 int 大小设置填充值
    __m128i fill = _mm_set_epi64x(fill_hi, fill_lo);
#endif
    
    // 如果向量长度大于1，则从 ptr + stride 处加载数据到 AVX 寄存器 b；否则使用 fill 作为数据源
    __m128i b = nlane > 1 ? _mm_loadu_si128((const __m128i*)(ptr + stride)) : fill;
    
    // 将低128位从 b 插入到 a 的高128位，形成结果 ret
    __m256i ret = _mm256_inserti128_si256(a, b, 1);
    
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果开启了部分加载保护，则进行一个额外的 volatile 操作
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    
    // 返回加载的 AVX 寄存器 ret
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_loadn2_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn2_till_s64(ptr, stride, nlane, 0, 0); }

/*********************************
 * Partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    // 断言确保存储的向量长度大于0
    assert(nlane > 0);
    
    // 创建包含步长的 AVX 寄存器
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    
    // 设置存储的向量长度
    __m256i vnlane = _mm256_set1_epi32(nlane > 8 ? 8 : (int)nlane);
    
    // 创建用于掩码比较的 AVX 寄存器
    __m256i mask   = _mm256_cmpgt_epi32(vnlane, steps);
    
    // 使用掩码将 AVX 寄存器 a 中的数据存储到指针 ptr 处
    _mm256_maskstore_epi32((int*)ptr, mask, a);
}
//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    // 断言确保存储的向量长度大于0
    assert(nlane > 0);
    
    // 创建包含步长的 AVX 寄存器
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    
    // 设置存储的向量长度
    __m256i vnlane = npyv_setall_s64(nlane > 8 ? 8 : (int)nlane);
    
    // 创建用于掩码比较的 AVX 寄存器
    __m256i mask   = _mm256_cmpgt_epi64(vnlane, steps);
    # 使用 AVX2 指令集的 _mm256_maskstore_epi64 函数，将数据 a 根据掩码 mask 存储到内存地址 ptr 指向的位置
    _mm256_maskstore_epi64((long long*)ptr, mask, a);
/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
// 将长度为32位的向量部分存储到非连续地址的内存中
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    // 确保存储的长度大于0
    assert(nlane > 0);
    
    // 将256位整型向量a转换为两个128位整型向量a0和a1
    __m128i a0 = _mm256_castsi256_si128(a);
    __m128i a1 = _mm256_extracti128_si256(a, 1);

    // 根据nlane的不同值，选择不同数量的元素存储到内存中，间隔为stride
    ptr[stride*0] = _mm_extract_epi32(a0, 0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        return;
    case 3:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        return;
    case 4:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        return;
    case 5:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        ptr[stride*4] = _mm_extract_epi32(a1, 0);
        return;
    case 6:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        ptr[stride*4] = _mm_extract_epi32(a1, 0);
        ptr[stride*5] = _mm_extract_epi32(a1, 1);
        return;
    case 7:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        ptr[stride*4] = _mm_extract_epi32(a1, 0);
        ptr[stride*5] = _mm_extract_epi32(a1, 1);
        ptr[stride*6] = _mm_extract_epi32(a1, 2);
        return;
    }
}
    // 根据 switch 语句的默认情况执行以下操作
    default:
        // 将 a0 的第 1 个 32 位整数提取出来，并存入 ptr 数组的对应位置
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        // 将 a0 的第 2 个 32 位整数提取出来，并存入 ptr 数组的对应位置
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        // 将 a0 的第 3 个 32 位整数提取出来，并存入 ptr 数组的对应位置
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        // 将 a1 的第 0 个 32 位整数提取出来，并存入 ptr 数组的对应位置
        ptr[stride*4] = _mm_extract_epi32(a1, 0);
        // 将 a1 的第 1 个 32 位整数提取出来，并存入 ptr 数组的对应位置
        ptr[stride*5] = _mm_extract_epi32(a1, 1);
        // 将 a1 的第 2 个 32 位整数提取出来，并存入 ptr 数组的对应位置
        ptr[stride*6] = _mm_extract_epi32(a1, 2);
        // 将 a1 的第 3 个 32 位整数提取出来，并存入 ptr 数组的对应位置
        ptr[stride*7] = _mm_extract_epi32(a1, 3);
    }
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    // 将 256 位整数向量 a 转换为两个 128 位双精度浮点向量
    __m128d a0 = _mm256_castpd256_pd128(_mm256_castsi256_pd(a));
    __m128d a1 = _mm256_extractf128_pd(_mm256_castsi256_pd(a), 1);

    // 将指针 ptr 强制转换为双精度浮点数指针
    double *dptr = (double*)ptr;
    // 存储 a0 的低位部分到 dptr
    _mm_storel_pd(dptr, a0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        // 如果 nlane 为 2，则存储 a0 的高位部分到 dptr + stride * 1
        _mm_storeh_pd(dptr + stride * 1, a0);
        return;
    case 3:
        // 如果 nlane 为 3，则存储 a0 的高位部分到 dptr + stride * 1，
        // 存储 a1 的低位部分到 dptr + stride * 2
        _mm_storeh_pd(dptr + stride * 1, a0);
        _mm_storel_pd(dptr + stride * 2, a1);
        return;
    default:
        // 对于其他情况，存储 a0 的高位部分到 dptr + stride * 1，
        // 存储 a1 的低位部分到 dptr + stride * 2，
        // 存储 a1 的高位部分到 dptr + stride * 3
        _mm_storeh_pd(dptr + stride * 1, a0);
        _mm_storel_pd(dptr + stride * 2, a1);
        _mm_storeh_pd(dptr + stride * 3, a1);
    }
}

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    // 将 256 位整数向量 a 转换为两个 128 位双精度浮点向量
    __m128d a0 = _mm256_castpd256_pd128(_mm256_castsi256_pd(a));
    __m128d a1 = _mm256_extractf128_pd(_mm256_castsi256_pd(a), 1);

    // 存储 a0 的低位部分到双精度浮点数指针 ptr
    _mm_storel_pd((double*)ptr, a0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        // 如果 nlane 为 2，则存储 a0 的高位部分到双精度浮点数指针 ptr + stride * 1
        _mm_storeh_pd((double*)(ptr + stride * 1), a0);
        return;
    case 3:
        // 如果 nlane 为 3，则存储 a0 的高位部分到双精度浮点数指针 ptr + stride * 1，
        // 存储 a1 的低位部分到双精度浮点数指针 ptr + stride * 2
        _mm_storeh_pd((double*)(ptr + stride * 1), a0);
        _mm_storel_pd((double*)(ptr + stride * 2), a1);
        return;
    default:
        // 对于其他情况，存储 a0 的高位部分到双精度浮点数指针 ptr + stride * 1，
        // 存储 a1 的低位部分到双精度浮点数指针 ptr + stride * 2，
        // 存储 a1 的高位部分到双精度浮点数指针 ptr + stride * 3
        _mm_storeh_pd((double*)(ptr + stride * 1), a0);
        _mm_storel_pd((double*)(ptr + stride * 2), a1);
        _mm_storeh_pd((double*)(ptr + stride * 3), a1);
    }
}

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    // 调用 npyv_storel_s64 存储 a 的低位部分到 ptr
    npyv_storel_s64(ptr, a);
    if (nlane > 1) {
        // 如果 nlane 大于 1，则调用 npyv_storeh_s64 存储 a 的高位部分到 ptr + stride
        npyv_storeh_s64(ptr + stride, a);
    }
}
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        // 使用联合体 pun 将填充值 fill 转换为目标类型 T_SFX 的数据
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(                   \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX                       \
        ));                                                                                 \
    }                                                                                       \
    // 加载并重新解释数据为类型为 T_SFX 的向量，并返回
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill)                                                            \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        // 使用联合体 pun 将填充值 fill 转换为目标类型 T_SFX 的数据
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun.to_##T_SFX               \
        ));                                                                                 \
    }                                                                                       \
    // 加载指定步长和数量的数据，并重新解释为类型为 T_SFX 的向量，并返回
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        // 加载指定数量的数据，并重新解释为类型为 T_SFX 的向量，并返回
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    // 加载直到遇到零的数据，并重新解释为类型为 T_SFX 的向量，并返回
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX                                       \
    # 定义一个宏，用于将给定类型 F_SFX 的指针数组重新解释为类型 T_SFX 的向量数组，并加载直到遇到零元素的向量。
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)
    {
        # 调用 npyv_loadn_tillz_##T_SFX 函数，加载从指针 ptr 开始，步长为 stride 的 nlane 个向量，直到遇到零元素。
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane
        ));
    }
    
    # 定义一个内联函数，将类型 F_SFX 的向量数组 a 存储到类型 F_SFX 的指针数组 ptr 中，存储直到达到 nlane 个向量。
    NPY_FINLINE void npyv_store_till_##F_SFX
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)
    {
        # 调用 npyv_reinterpret_##T_SFX##_##F_SFX 函数，将类型 F_SFX 的向量数组 a 重新解释为类型 T_SFX 的向量数组，然后存储到 ptr 中，存储直到达到 nlane 个向量。
        npyv_store_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
    
    # 定义一个内联函数，将类型 F_SFX 的向量数组 a 存储到类型 F_SFX 的指针数组 ptr 中，使用步长 stride，存储直到达到 nlane 个向量。
    NPY_FINLINE void npyv_storen_till_##F_SFX
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)
    {
        # 调用 npyv_reinterpret_##T_SFX##_##F_SFX 函数，将类型 F_SFX 的向量数组 a 重新解释为类型 T_SFX 的向量数组，然后使用步长 stride 存储到 ptr 中，存储直到达到 nlane 个向量。
        npyv_storen_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
// 定义宏，用于生成一组加载和存储函数的实现，支持AVX2并处理部分数据类型

// 定义加载函数的宏
#define NPYV_IMPL_AVX2_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                \
    // 定义加载函数，加载AVX2向量类型 F_SFX，处理数据类型为 T_SFX
    NPY_FINLINE npyv_##F_SFX npyv_load2_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                                     \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        // 定义联合体 pun，用于类型转换
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        // 定义 pun_lo 和 pun_hi，用于分别存储 fill_lo 和 fill_hi
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        // 调用类型转换宏 npyv_reinterpret_##F_SFX##_##T_SFX，将加载函数转换为 T_SFX 类型
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun_lo.to_##T_SFX, pun_hi.to_##T_SFX \
        ));                                                                                 \
    }                                                                                       \
    // 定义加载函数，加载 AVX2 向量类型 F_SFX 的一对值，处理数据类型为 T_SFX
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
        // 调用 reinterpret 函数将填充值重新解释为目标类型，并加载对应的向量数据
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_till_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun_lo.to_##T_SFX,           \
            pun_hi.to_##T_SFX                                                               \
        ));                                                                                 \
    }                                                                                       \
    // 加载非零填充的前两个向量并将它们转换为目标类型
    NPY_FINLINE npyv_##F_SFX npyv_load2_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        // 调用 reinterpret 函数将加载的数据重新解释为目标类型并返回
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    // 加载非零填充的前两个向量并将它们转换为目标类型
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_tillz_##F_SFX                                      \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        // 调用 reinterpret 函数将加载的数据重新解释为目标类型并返回
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_tillz_##T_SFX(                \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    // 存储到目标类型的前两个向量数据
    NPY_FINLINE void npyv_store2_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        npyv_store2_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen2_till_##F_SFX                                              \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen2_till_##T_SFX(                                                          \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }



{                                                                                       \
    npyv_store2_till_##T_SFX(                                                           \
        (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
        npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
    );                                                                                  \
}


这段代码片段是一个宏定义和一个内联函数的实现。它们用于向特定类型的向量寄存器（vector register）存储数据。


NPY_FINLINE void npyv_storen2_till_##F_SFX                                              \
(npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
{


这是一个内联函数的声明，用于存储多个向量元素到内存中，其中：
- `npyv_storen2_till_##F_SFX` 是函数名的宏展开形式，根据 `F_SFX` 定义生成不同的函数名。
- `npyv_lanetype_##F_SFX` 是一个特定类型的向量元素的数据类型。
- `ptr` 是指向目标内存区域的指针。
- `stride` 是步长，表示内存中相邻元素之间的间隔。
- `nlane` 是要存储的向量元素数量。
- `npyv_##F_SFX a` 是要存储的向量数据。


    npyv_storen2_till_##T_SFX(                                                          \
        (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
        npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
    );                                                                                  \
}


这里是函数实现的具体内容，调用了另一个宏展开的函数 `npyv_storen2_till_##T_SFX`，这个函数实际上是将一个类型的向量 `a` 存储为另一种类型 `T_SFX` 的向量数据。

整体来说，这段代码片段通过宏定义和内联函数实现了向不同类型的向量寄存器存储数据的功能，利用了预处理宏的特性来生成不同类型的函数名和函数实现。
// 宏定义，用于生成AVX2指令集的加载/存储解交错函数对，操作的数据类型为u32和s32
#define NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(u32, s32) \
    // 省略部分实现细节，生成u32和s32类型对应的AVX2指令集函数

// 依据宏定义生成AVX2指令集的加载/存储解交错函数对，操作的数据类型为f32和s32
#define NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(f32, s32) \
    // 省略部分实现细节，生成f32和s32类型对应的AVX2指令集函数

// 依据宏定义生成AVX2指令集的加载/存储解交错函数对，操作的数据类型为u64和s64
#define NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(u64, s64) \
    // 省略部分实现细节，生成u64和s64类型对应的AVX2指令集函数

// 依据宏定义生成AVX2指令集的加载/存储解交错函数对，操作的数据类型为f64和s64
#define NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(f64, s64) \
    // 省略部分实现细节，生成f64和s64类型对应的AVX2指令集函数

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/

// 定义AVX2指令集的加载/存储解交错函数宏，操作数据类型为SFX和ZSFX
#define NPYV_IMPL_AVX2_MEM_INTERLEAVE(SFX, ZSFX)                             \
    // 内联函数，将类型为npyv_ZSFX的向量解交错为类型为npyv_SFX的双通道向量
    NPY_FINLINE npyv_##ZSFX##x2 npyv_zip_##ZSFX(npyv_##SFX, npyv_##SFX);   \
    // 内联函数，将类型为npyv_SFX的双通道向量互连为类型为npyv_ZSFX的向量
    NPY_FINLINE npyv_##ZSFX##x2 npyv_unzip_##ZSFX(npyv_##SFX, npyv_##SFX); \
    // 内联函数，从内存中加载类型为npyv_lanetype_SFX的双通道向量
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                          \
        const npyv_lanetype_##SFX *ptr                                       \
    ) {                                                                      \
        // 返回解交错后的双通道向量
        return npyv_unzip_##ZSFX(                                            \
            // 调用npyv_load_##SFX函数加载ptr和ptr+npyv_nlanes_##SFX位置的数据
            npyv_load_##SFX(ptr), npyv_load_##SFX(ptr+npyv_nlanes_##SFX)     \
        );                                                                   \
    }                                                                        \
    // 内联函数，将类型为npyv_SFX的双通道向量存储到内存中
    NPY_FINLINE void npyv_store_##SFX##x2(                                   \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                           \
    ) {                                                                      \
        // 将双通道向量v进行互连并存储到ptr和ptr+npyv_nlanes_##SFX位置
        npyv_##SFX##x2 zip = npyv_zip_##ZSFX(v.val[0], v.val[1]);            \
        npyv_store_##SFX(ptr, zip.val[0]);                                   \
        npyv_store_##SFX(ptr + npyv_nlanes_##SFX, zip.val[1]);               \
    }

// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为u8和u8
NPYV_IMPL_AVX2_MEM_INTERLEAVE(u8, u8)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为s8和u8
NPYV_IMPL_AVX2_MEM_INTERLEAVE(s8, u8)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为u16和u16
NPYV_IMPL_AVX2_MEM_INTERLEAVE(u16, u16)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为s16和u16
NPYV_IMPL_AVX2_MEM_INTERLEAVE(s16, u16)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为u32和u32
NPYV_IMPL_AVX2_MEM_INTERLEAVE(u32, u32)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为s32和u32
NPYV_IMPL_AVX2_MEM_INTERLEAVE(s32, u32)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为u64和u64
NPYV_IMPL_AVX2_MEM_INTERLEAVE(u64, u64)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为s64和u64
NPYV_IMPL_AVX2_MEM_INTERLEAVE(s64, u64)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为f32和f32
NPYV_IMPL_AVX2_MEM_INTERLEAVE(f32, f32)
// 使用宏定义生成AVX2指令集的加载/存储解交错函数对，操作数据类型为f64和f64
NPYV_IMPL_AVX2_MEM_INTERLEAVE(f64, f64)

/*********************************
 * Lookup tables
 *********************************/

// 使用向量作为索引，从包含32个float32元素的表中查找值
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{ return _mm256_i32gather_ps(table, idx, 4); }

// 使用向量作为索引，从包含32个uint32元素的表中查找值，并将结果重新解释为npyv_u32
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float*)table, idx)); }

// 使用向量作为索引，从包含32个int32元素的表中查找值，并将结果重新解释为npyv_s32
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_f32(npyv_lut32_f32((const float*)table, idx)); }

// 使用向量作为索引，从包含16个float64元素的表中查找值
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{ return _mm256_i64gather_pd(table, idx, 8); }

// 使用向量作为索引，从包含16个uint64元素的表中查找值，并将结果重新解释为npyv_u64
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
# 返回一个64位浮点数向量，其内容由表格中索引为idx的16个元素查找后重解释为64位浮点数得到
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }
# 返回一个64位有符号整数向量，其内容由表格中索引为idx的16个元素查找后重解释为64位有符号整数得到
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_AVX2_MEMORY_H
```