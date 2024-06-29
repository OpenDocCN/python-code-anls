# `.\numpy\numpy\_core\src\common\simd\avx512\memory.h`

```py
#ifndef NPY_SIMD
    // 如果没有定义 NPY_SIMD 宏，则输出错误信息 "Not a standalone header"
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_MEMORY_H
#define _NPY_SIMD_AVX512_MEMORY_H

#include "misc.h"

/***************************
 * load/store
 ***************************/

#if defined(__GNUC__)
    // 如果使用 GCC 编译器，因为 GCC 期望指针参数类型为 `void*` 而不是 `const void*`，会引发大量警告，
    // 所以使用 `_mm512_stream_load_si512` 进行加载操作
    #define npyv__loads(PTR) _mm512_stream_load_si512((__m512i*)(PTR))
#else
    // 对于其他编译器，正常使用 `_mm512_stream_load_si512` 加载操作
    #define npyv__loads(PTR) _mm512_stream_load_si512((const __m512i*)(PTR))
#endif

#if defined(_MSC_VER) && defined(_M_IX86)
    // 解决 MSVC(32位) 的溢出 bug，详见 https://developercommunity.visualstudio.com/content/problem/911872/u.html
    NPY_FINLINE __m512i npyv__loadl(const __m256i *ptr)
    {
        // 使用 `_mm256_loadu_si256` 加载 `ptr` 所指向的内存，并将结果插入到 `_mm512_castsi256_si512` 的结果中返回
        __m256i a = _mm256_loadu_si256(ptr);
        return _mm512_inserti64x4(_mm512_castsi256_si512(a), a, 0);
    }
#else
    // 对于其他情况，使用 `_mm256_loadu_si256` 加载 `PTR` 所指向的内存，然后使用 `_mm512_castsi256_si512` 转换返回结果
    #define npyv__loadl(PTR) \
        _mm512_castsi256_si512(_mm256_loadu_si256(PTR))
#endif

// 定义 AVX-512 内存操作的宏实现，包括不同类型的加载和存储操作
#define NPYV_IMPL_AVX512_MEM_INT(CTYPE, SFX)                                 \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)                 \
    { return _mm512_loadu_si512((const __m512i*)ptr); }                      \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)                \
    { return _mm512_load_si512((const __m512i*)ptr); }                       \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)                \
    { return npyv__loads(ptr); }                                             \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)                \
    { return npyv__loadl((const __m256i *)ptr); }                            \
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)            \
    { _mm512_storeu_si512((__m512i*)ptr, vec); }                             \
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm512_store_si512((__m512i*)ptr, vec); }                              \
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm512_stream_si512((__m512i*)ptr, vec); }                             \
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm256_storeu_si256((__m256i*)ptr, npyv512_lower_si256(vec)); }        \
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm256_storeu_si256((__m256i*)(ptr), npyv512_higher_si256(vec)); }

// 定义不同整数类型的 AVX-512 内存操作实现
NPYV_IMPL_AVX512_MEM_INT(npy_uint8,  u8)
NPYV_IMPL_AVX512_MEM_INT(npy_int8,   s8)
NPYV_IMPL_AVX512_MEM_INT(npy_uint16, u16)
NPYV_IMPL_AVX512_MEM_INT(npy_int16,  s16)
NPYV_IMPL_AVX512_MEM_INT(npy_uint32, u32)
NPYV_IMPL_AVX512_MEM_INT(npy_int32,  s32)
NPYV_IMPL_AVX512_MEM_INT(npy_uint64, u64)
NPYV_IMPL_AVX512_MEM_INT(npy_int64,  s64)

// 不对齐加载操作宏定义
#define npyv_load_f32(PTR) _mm512_loadu_ps((const __m512*)(PTR))
#define npyv_load_f64(PTR) _mm512_loadu_pd((const __m512d*)(PTR))
// 对齐加载操作

#endif // _NPY_SIMD_AVX512_MEMORY_H
// 定义宏：以 32 位浮点数精度加载数据到 AVX-512 向量，使用非对齐方式
#define npyv_loada_f32(PTR) _mm512_load_ps((const __m512*)(PTR))
// 定义宏：以 64 位浮点数精度加载数据到 AVX-512 向量，使用非对齐方式
#define npyv_loada_f64(PTR) _mm512_load_pd((const __m512d*)(PTR))

// 宏条件编译：加载低位部分的数据，根据编译器和架构的不同使用不同的实现
#if defined(_MSC_VER) && defined(_M_IX86)
    #define npyv_loadl_f32(PTR) _mm512_castsi512_ps(npyv__loadl((const __m256i *)(PTR)))
    #define npyv_loadl_f64(PTR) _mm512_castsi512_pd(npyv__loadl((const __m256i *)(PTR)))
#else
    #define npyv_loadl_f32(PTR) _mm512_castps256_ps512(_mm256_loadu_ps(PTR))
    #define npyv_loadl_f64(PTR) _mm512_castpd256_pd512(_mm256_loadu_pd(PTR))
#endif

// 定义宏：使用流加载方式加载数据到 AVX-512 向量，适合连续加载操作
#define npyv_loads_f32(PTR) _mm512_castsi512_ps(npyv__loads(PTR))
#define npyv_loads_f64(PTR) _mm512_castsi512_pd(npyv__loads(PTR))

// 定义宏：以非对齐方式将 AVX-512 向量中的数据存储到内存
#define npyv_store_f32 _mm512_storeu_ps
#define npyv_store_f64 _mm512_storeu_pd

// 定义宏：以对齐方式将 AVX-512 向量中的数据存储到内存
#define npyv_storea_f32 _mm512_store_ps
#define npyv_storea_f64 _mm512_store_pd

// 定义宏：使用流存储方式将 AVX-512 向量中的数据存储到内存
#define npyv_stores_f32 _mm512_stream_ps
#define npyv_stores_f64 _mm512_stream_pd

// 定义宏：存储 AVX-512 向量的低位部分到内存
#define npyv_storel_f32(PTR, VEC) _mm256_storeu_ps(PTR, npyv512_lower_ps256(VEC))
#define npyv_storel_f64(PTR, VEC) _mm256_storeu_pd(PTR, npyv512_lower_pd256(VEC))

// 定义宏：存储 AVX-512 向量的高位部分到内存
#define npyv_storeh_f32(PTR, VEC) _mm256_storeu_ps(PTR, npyv512_higher_ps256(VEC))
#define npyv_storeh_f64(PTR, VEC) _mm256_storeu_pd(PTR, npyv512_higher_pd256(VEC))

/***************************
 * 非连续加载操作
 ***************************/

//// 32 位整数加载
// 加载非连续的 32 位无符号整数到 AVX-512 向量
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{
    // 断言：步长的绝对值不超过 NPY_SIMD_MAXLOAD_STRIDE32
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    // 设置步长向量
    const __m512i steps = npyv_set_s32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    // 计算索引
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32((int)stride));
    // 使用步长向量从内存中加载数据到 AVX-512 向量
    return _mm512_i32gather_epi32(idx, (const __m512i*)ptr, 4);
}

// 加载非连续的 32 位有符号整数到 AVX-512 向量
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{ return npyv_loadn_u32((const npy_uint32*)ptr, stride); }

// 加载非连续的 32 位浮点数到 AVX-512 向量
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return _mm512_castsi512_ps(npyv_loadn_u32((const npy_uint32*)ptr, stride)); }

//// 64 位整数加载
// 加载非连续的 64 位无符号整数到 AVX-512 向量
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{
    // 设置索引向量
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    // 使用索引向量从内存中加载数据到 AVX-512 向量
    return _mm512_i64gather_epi64(idx, (const __m512i*)ptr, 8);
}

// 加载非连续的 64 位有符号整数到 AVX-512 向量
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return npyv_loadn_u64((const npy_uint64*)ptr, stride); }

// 加载非连续的 64 位双精度浮点数到 AVX-512 向量
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return _mm512_castsi512_pd(npyv_loadn_u64((const npy_uint64*)ptr, stride)); }

//// 64 位整数加载（通过 32 位步长）
// 加载非连续的 32 位无符号整数到 AVX-512 向量的高位
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{
    // 使用 AVX 指令集加载第二部分数据到 AVX-512 向量的低位
    __m128d a = _mm_loadh_pd(
        _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)ptr)),
        (const double*)(ptr + stride)
    );
    //
    # 加载两个 double 值到寄存器 __m128d 中，其中高位值来自 ptr + stride*2 处的数据，低位值来自 ptr + stride*3 处的数据
    __m128d b = _mm_loadh_pd(
        _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*2))),
        (const double*)(ptr + stride*3)
    );
    
    # 加载两个 double 值到寄存器 __m128d 中，其中高位值来自 ptr + stride*4 处的数据，低位值来自 ptr + stride*5 处的数据
    __m128d c = _mm_loadh_pd(
        _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*4))),
        (const double*)(ptr + stride*5)
    );
    
    # 加载两个 double 值到寄存器 __m128d 中，其中高位值来自 ptr + stride*6 处的数据，低位值来自 ptr + stride*7 处的数据
    __m128d d = _mm_loadh_pd(
        _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*6))),
        (const double*)(ptr + stride*7)
    );
    
    # 组合四个 __m128d 寄存器成一个 __m512i 寄存器，通过调用 npyv512_combine_pd256 函数并转换为 __m512i 类型返回
    return _mm512_castpd_si512(npyv512_combine_pd256(
        # 将两个 __m128d 寄存器 a 和 b 合并为一个 __m256d 寄存器，并在高位插入 b 寄存器的值
        _mm256_insertf128_pd(_mm256_castpd128_pd256(a), b, 1),
        # 将两个 __m128d 寄存器 c 和 d 合并为一个 __m256d 寄存器，并在高位插入 d 寄存器的值
        _mm256_insertf128_pd(_mm256_castpd128_pd256(c), d, 1)
    ));
//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    // 将输入的512位整数向量转换为两个256位双精度浮点向量
    __m256d lo = _mm512_castpd512_pd256(_mm512_castsi512_pd(a));
    // 提取512位整数向量的高128位并转换为256位双精度浮点向量
    __m256d hi = _mm512_extractf64x4_pd(_mm512_castsi512_pd(a), 1);
    // 从lo中提取低128位双精度浮点数并存储到ptr的0号元素位置
    __m128d e0 = _mm256_castpd256_pd128(lo);
    // 从lo中提取高128位双精度浮点数并存储到ptr的1号元素位置
    __m128d e1 = _mm256_extractf128_pd(lo, 1);
    // 从hi中提取低128位双精度浮点数并存储到ptr的2号元素位置
    __m128d e2 = _mm256_castpd256_pd128(hi);
    // 从hi中提取高128位双精度浮点数并存储到ptr的3号元素位置
    __m128d e3 = _mm256_extractf128_pd(hi, 1);
    // 使用单精度浮点数存储ptr的0号位置
    _mm_storel_pd((double*)(ptr + stride * 0), e0);
    // 使用单精度浮点数存储ptr的1号位置
    _mm_storeh_pd((double*)(ptr + stride * 1), e0);
    // 使用单精度浮点数存储ptr的2号位置
    _mm_storel_pd((double*)(ptr + stride * 2), e1);
}
    // 使用 SSE2 指令集中的 _mm_storeh_pd 函数，将双精度浮点数 e1 的高64位存储到 ptr + stride * 3 处
    _mm_storeh_pd((double*)(ptr + stride * 3), e1);
    // 使用 SSE2 指令集中的 _mm_storel_pd 函数，将双精度浮点数 e2 的低64位存储到 ptr + stride * 4 处
    _mm_storel_pd((double*)(ptr + stride * 4), e2);
    // 使用 SSE2 指令集中的 _mm_storeh_pd 函数，将双精度浮点数 e2 的高64位存储到 ptr + stride * 5 处
    _mm_storeh_pd((double*)(ptr + stride * 5), e2);
    // 使用 SSE2 指令集中的 _mm_storel_pd 函数，将双精度浮点数 e3 的低64位存储到 ptr + stride * 6 处
    _mm_storel_pd((double*)(ptr + stride * 6), e3);
    // 使用 SSE2 指令集中的 _mm_storeh_pd 函数，将双精度浮点数 e3 的高64位存储到 ptr + stride * 7 处
    _mm_storeh_pd((double*)(ptr + stride * 7), e3);
//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    // 将512位无符号整数向量a分成低256位和高256位
    __m256i lo = npyv512_lower_si256(a);
    __m256i hi = npyv512_higher_si256(a);

    // 从lo中提取128位元素e0和e1，从hi中提取128位元素e2和e3
    __m128i e0 = _mm256_castsi256_si128(lo);
    __m128i e1 = _mm256_extracti128_si256(lo, 1);
    __m128i e2 = _mm256_castsi256_si128(hi);
    __m128i e3 = _mm256_extracti128_si256(hi, 1);

    // 将128位元素e0、e1、e2、e3分别存储到ptr的不同偏移位置
    _mm_storeu_si128((__m128i*)(ptr + stride * 0), e0);
    _mm_storeu_si128((__m128i*)(ptr + stride * 1), e1);
    _mm_storeu_si128((__m128i*)(ptr + stride * 2), e2);
    _mm_storeu_si128((__m128i*)(ptr + stride * 3), e3);
}

/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    // 断言nlane大于0，即加载的元素数量大于0
    assert(nlane > 0);

    // 设置填充值为vfill
    const __m512i vfill = _mm512_set1_epi32(fill);

    // 计算掩码，mask为-1或者(1 << nlane) - 1，取决于nlane是否大于15
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;

    // 使用掩码加载元素到ret中，ptr强制转换为__m512i类型
    __m512i ret = _mm512_mask_loadu_epi32(vfill, mask, (const __m512i*)ptr);

    // 如果定义了NPY_SIMD_GUARD_PARTIAL_LOAD，执行下面的工作区绕过
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif

    return ret;
}

// fill zero to rest lanes
//// 32
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
    // 断言nlane大于0，即加载的元素数量大于0
    assert(nlane > 0);

    // 计算掩码，mask为-1或者(1 << nlane) - 1，取决于nlane是否大于15
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;

    // 使用掩码加载元素到ret中，ptr强制转换为__m512i类型，加载时填充零
    __m512i ret = _mm512_maskz_loadu_epi32(mask, (const __m512i*)ptr);

    // 如果定义了NPY_SIMD_GUARD_PARTIAL_LOAD，执行下面的工作区绕过
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif

    return ret;
}

//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    // 断言nlane大于0，即加载的元素数量大于0
    assert(nlane > 0);

    // 设置填充值为vfill
    const __m512i vfill = npyv_setall_s64(fill);

    // 计算掩码，mask为-1或者(1 << nlane) - 1，取决于nlane是否大于7
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;

    // 使用掩码加载元素到ret中，ptr强制转换为__m512i类型
    __m512i ret = _mm512_mask_loadu_epi64(vfill, mask, (const __m512i*)ptr);

    // 如果定义了NPY_SIMD_GUARD_PARTIAL_LOAD，执行下面的工作区绕过
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif

    return ret;
}

// fill zero to rest lanes
//// 64
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    // 断言nlane大于0，即加载的元素数量大于0
    assert(nlane > 0);

    // 计算掩码，mask为-1或者(1 << nlane) - 1，取决于nlane是否大于7
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;

    // 使用掩码加载元素到ret中，ptr强制转换为__m512i类型，加载时填充零
    __m512i ret = _mm512_maskz_loadu_epi64(mask, (const __m512i*)ptr);

    // 如果定义了NPY_SIMD_GUARD_PARTIAL_LOAD，执行下面的工作区绕过
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif

    return ret;
}
//// 64-bit nlane
// 定义一个函数，加载指定数量的 32 位有符号整数到 SIMD 向量中，同时使用指定的值填充未加载的部分
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    // 断言确保加载的 lane 数大于 0
    assert(nlane > 0);
    // 创建一个包含填充值的 512 位整数向量，顺序为 fill_hi, fill_lo, fill_hi, fill_lo
    const __m512i vfill = _mm512_set4_epi32(fill_hi, fill_lo, fill_hi, fill_lo);
    // 根据 nlane 的大小设置掩码，如果 nlane 大于 7，则掩码为全 1，否则为低 nlane 位为 1
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    // 使用掩码从指针处加载 64 位整数数据到 512 位整数向量 ret 中
    __m512i ret = _mm512_mask_loadu_epi64(vfill, mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，执行偏移量加载后的补救措施
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    // 返回加载后的向量 ret
    return ret;
}
// 使用零值填充未加载的 lane
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

//// 128-bit nlane
// 定义一个函数，加载指定数量的 64 位有符号整数到 SIMD 向量中，同时使用指定的值填充未加载的部分
NPY_FINLINE npyv_u64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{
    // 断言确保加载的 lane 数大于 0
    assert(nlane > 0);
    // 创建一个包含填充值的 512 位整数向量，顺序为 fill_hi, fill_lo, fill_hi, fill_lo
    const __m512i vfill = _mm512_set4_epi64(fill_hi, fill_lo, fill_hi, fill_lo);
    // 根据 nlane 的大小设置掩码，如果 nlane 大于 3，则掩码为全 1，否则为低 nlane*2 位为 1
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;
    // 使用掩码从指针处加载 64 位整数数据到 512 位整数向量 ret 中
    __m512i ret = _mm512_mask_loadu_epi64(vfill, mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，执行偏移量加载后的补救措施
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    // 返回加载后的向量 ret
    return ret;
}
// 使用零值填充未加载的 lane
NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    // 断言确保加载的 lane 数大于 0
    assert(nlane > 0);
    // 根据 nlane 的大小设置掩码，如果 nlane 大于 3，则掩码为全 1，否则为低 nlane*2 位为 1
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;
    // 使用掩码从指针处加载零填充的 64 位整数数据到 512 位整数向量 ret 中
    __m512i ret = _mm512_maskz_loadu_epi64(mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，执行偏移量加载后的补救措施
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    // 返回加载后的向量 ret
    return ret;
}
/*********************************
 * Non-contiguous partial load
 *********************************/
//// 32
// 定义一个函数，以非连续的方式加载指定数量的 32 位有符号整数到 SIMD 向量中，同时使用指定的值填充未加载的部分
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    // 断言确保加载的 lane 数大于 0
    assert(nlane > 0);
    // 断言确保步长的绝对值不超过 NPY_SIMD_MAXLOAD_STRIDE32
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    // 创建一个顺序为 0 到 15 的 512 位整数向量 steps
    const __m512i steps = npyv_set_s32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    // 使用步长和给定的步长值计算索引
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32((int)stride));
    // 创建一个包含填充值的 512 位整数向量 vfill
    const __m512i vfill = _mm512_set1_epi32(fill);
    // 根据 nlane 的大小设置掩码，如果 nlane 大于 15，则掩码为全 1，否则为低 nlane 位为 1
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    // 使用掩码从 ptr 指针处按照 4 字节步长加载 32 位整数数据到 512 位整数向量 ret 中
    __m512i ret = _mm512_mask_i32gather_epi32(vfill, mask, idx, (const __m512i*)ptr, 4);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，执行偏移量加载后的补救措施
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    // 返回加载后的向量 ret
    return ret;
}
// 使用零值填充未加载的 lane
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }

//// 64
// 定义一个函数，以非连续的方式加载指定数量的 64 位有符号整数到 SIMD 向量中，同时使用指定的值填充未加载的部分
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    // 断言确保加载的 lane 数大于 0
    assert(nlane > 0);
    // 断言确保步长的绝对值不超过 NPY_SIMD_MAXLOAD_STRIDE64
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE64);
    //cpp
    // 创建一个包含给定索引值的 __m512i 向量，这些索引是以步长 stride 递增的
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    
    // 创建一个所有元素均为 fill 的 __m512i 向量
    const __m512i vfill = npyv_setall_s64(fill);
    
    // 根据 nlane 的值创建一个掩码，如果 nlane 大于 15，则掩码设置为全1；否则设置为 (1 << nlane) - 1
    const __mmask8 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    
    // 使用 _mm512_mask_i64gather_epi64 函数从 ptr 指向的内存中根据 idx 向量的索引收集数据到 ret 向量中，每次收集 8 个元素
    __m512i ret = _mm512_mask_i64gather_epi64(vfill, mask, idx, (const __m512i*)ptr, 8);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则使用 workaround 来处理 ret
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    // 返回 ret 变量的值
    return ret;
}
// 使用零值填充其余的通道

//// 64-bit load over 32-bit stride
// 以 64 位加载数据，步长为 32 位
NPY_FINLINE npyv_s64 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    // 创建包含索引的 __m512i 对象，每个索引乘以步长用于加载数据
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    // 创建包含填充值的 __m512i 对象
    const __m512i vfill = _mm512_set4_epi32(fill_hi, fill_lo, fill_hi, fill_lo);
    // 根据 nlane 的大小设置掩码，以便决定加载多少数据
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    // 使用 gather 操作加载数据到 ret 变量中
    __m512i ret = _mm512_mask_i64gather_epi64(vfill, mask, idx, (const __m512i*)ptr, 4);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则使用 workaround 处理 ret
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    // 返回加载的结果
    return ret;
}
// 使用零值填充其余的通道

//// 128-bit load over 64-bit stride
// 以 128 位加载数据，步长为 64 位
NPY_FINLINE npyv_s64 npyv_loadn2_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                                                  npy_int64 fill_lo, npy_int64 fill_hi)
{
    assert(nlane > 0);
    // 创建包含索引的 __m512i 对象，每个索引乘以步长用于加载数据
    const __m512i idx = npyv_set_s64(
       0,        1,          stride,   stride+1,
       stride*2, stride*2+1, stride*3, stride*3+1
    );
    // 根据 nlane 的大小设置掩码，以便决定加载多少数据
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;
    // 创建包含填充值的 __m512i 对象
    const __m512i vfill = _mm512_set4_epi64(fill_hi, fill_lo, fill_hi, fill_lo);
    // 使用 gather 操作加载数据到 ret 变量中
    __m512i ret = _mm512_mask_i64gather_epi64(vfill, mask, idx, (const __m512i*)ptr, 8);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则使用 workaround 处理 ret
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    // 返回加载的结果
    return ret;
}
// 使用零值填充其余的通道

/*********************************
 * Partial store
 *********************************/
//// 32
// 存储至少 32 位数据
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    // 根据 nlane 的大小设置掩码，以便决定存储多少数据
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    // 使用掩码进行存储操作
    _mm512_mask_storeu_epi32((__m512i*)ptr, mask, a);
}
//// 64
// 存储至少 64 位数据
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    // 根据 nlane 的大小设置掩码，以便决定存储多少数据
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    // 使用掩码进行存储操作
    _mm512_mask_storeu_epi64((__m512i*)ptr, mask, a);
}

//// 64-bit nlane
// 存储至少 64 位数据
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    // 根据 nlane 的大小设置掩码，以便决定存储多少数据
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    # 使用512位无符号整数向内存位置ptr按mask掩码存储a的值
    _mm512_mask_storeu_epi64((__m512i*)ptr, mask, a);
/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    // 确保要存储的元素数量大于0
    assert(nlane > 0);
    // 确保步长的绝对值不超过最大存储步长限制
    assert(llabs(stride) <= NPY_SIMD_MAXSTORE_STRIDE32);

    // 创建一个步长数组
    const __m512i steps = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );

    // 计算实际存储时的索引，使用给定的步长
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32((int)stride));

    // 根据要存储的元素数量，创建一个掩码
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;

    // 使用掩码进行非连续的整数存储
    _mm512_mask_i32scatter_epi32((__m512i*)ptr, mask, idx, a, 4);
}

//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    // 确保要存储的元素数量大于0
    assert(nlane > 0);

    // 创建一个步长数组
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );

    // 根据要存储的元素数量，创建一个掩码
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;

    // 使用掩码进行非连续的64位整数存储
    _mm512_mask_i64scatter_epi64((__m512i*)ptr, mask, idx, a, 8);
}

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    // 确保要存储的元素数量大于0
    assert(nlane > 0);

    // 创建一个步长数组
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );

    // 根据要存储的元素数量，创建一个掩码
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;

    // 使用掩码进行64位整数存储，但每个元素占4字节步长
    _mm512_mask_i64scatter_epi64((__m512i*)ptr, mask, idx, a, 4);
}

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    // 确保要存储的元素数量大于0
    assert(nlane > 0);

    // 创建一个步长数组
    const __m512i idx = npyv_set_s64(
        0,        1,            stride,   stride+1,
        2*stride, 2*stride+1, 3*stride, 3*stride+1
    );

    // 根据要存储的元素数量，创建一个掩码
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;

    // 使用掩码进行64位整数存储，每个元素占8字节步长
    _mm512_mask_i64scatter_epi64((__m512i*)ptr, mask, idx, a, 8);
}
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \  // 声明一个联合体 pun，其中包含类型为 npyv_lanetype_##F_SFX 的 from_##F_SFX 成员
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \  // 联合体中还包含类型为 npyv_lanetype_##T_SFX 的 to_##T_SFX 成员
        } pun;                                                                              \  // 声明 pun 作为联合体变量
        pun.from_##F_SFX = fill;                                                            \  // 将 fill 赋值给联合体 pun 中的 from_##F_SFX 成员
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(                   \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX                       \  // 调用 npyv_load_till_##T_SFX 函数，传递指针 ptr、nlane 和联合体 pun 中的 to_##T_SFX 成员
        ));                                                                                 \  // 返回 npyv_reinterpret_##F_SFX##_##T_SFX 的结果
    }                                                                                       \  // 结束函数定义
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX                                        \  // 定义一个内联函数 npyv_loadn_till_##F_SFX，返回类型为 npyv_##F_SFX
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \  // 函数参数：指向 npyv_lanetype_##F_SFX 类型的指针 ptr，步长 stride，数量 nlane，以及填充值 fill
     npyv_lanetype_##F_SFX fill)                                                            \  // 填充值 fill 的类型为 npyv_lanetype_##F_SFX
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \  // 声明一个联合体 pun，包含类型为 npyv_lanetype_##F_SFX 的 from_##F_SFX 成员
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \  // 联合体中还包含类型为 npyv_lanetype_##T_SFX 的 to_##T_SFX 成员
        } pun;                                                                              \  // 声明 pun 作为联合体变量
        pun.from_##F_SFX = fill;                                                            \  // 将 fill 赋值给联合体 pun 中的 from_##F_SFX 成员
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun.to_##T_SFX               \  // 调用 npyv_loadn_till_##T_SFX 函数，传递指针 ptr、stride、nlane 和联合体 pun 中的 to_##T_SFX 成员
        ));                                                                                 \  // 返回 npyv_reinterpret_##F_SFX##_##T_SFX 的结果
    }                                                                                       \  // 结束函数定义
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX                                        \  // 定义一个内联函数 npyv_load_tillz_##F_SFX，返回类型为 npyv_##F_SFX
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \  // 函数参数：指向 npyv_lanetype_##F_SFX 类型的指针 ptr，数量 nlane
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \  // 调用 npyv_load_tillz_##T_SFX 函数，传递指针 ptr 和 nlane
        ));                                                                                 \  // 返回 npyv_reinterpret_##F_SFX##_##T_SFX 的结果
    }                                                                                       \  // 结束函数定义
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX                                       \
    # 定义一个宏，用于将类型为 F_SFX 的向量数据指针重新解释为类型为 T_SFX 的向量数据，并加载直到遇到零
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)
    {
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane
        ));
    }
    
    # 定义一个内联函数，用于将类型为 F_SFX 的向量数据 a 存储到类型为 F_SFX 的指针数组 ptr 的前 nlane 个位置
    NPY_FINLINE void npyv_store_till_##F_SFX
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)
    {
        # 将类型为 F_SFX 的向量数据 a 重新解释为类型为 T_SFX 的向量数据，并存储到 ptr 数组中前 nlane 个位置
        npyv_store_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
    
    # 定义一个内联函数，用于将类型为 F_SFX 的向量数据 a 存储到类型为 F_SFX 的指针数组 ptr 中，带有步长 stride，并存储到前 nlane 个位置
    NPY_FINLINE void npyv_storen_till_##F_SFX
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)
    {
        # 将类型为 F_SFX 的向量数据 a 重新解释为类型为 T_SFX 的向量数据，并存储到 ptr 数组中，带有步长 stride，并存储到前 nlane 个位置
        npyv_storen_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
// 定义AVX-512加速部分类型的宏，用于u32和s32类型
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES(u32, s32)
// 定义AVX-512加速部分类型的宏，用于f32和s32类型
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES(f32, s32)
// 定义AVX-512加速部分类型的宏，用于u64和s64类型
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES(u64, s64)
// 定义AVX-512加速部分类型的宏，用于f64和s64类型

// 128位/64位步长（双元素加载/存储）的宏定义
#define NPYV_IMPL_AVX512_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                              \
    // 内部联合结构用于类型转换
    NPY_FINLINE npyv_##F_SFX npyv_load2_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                                     \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        // 联合结构定义
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        // 低位和高位数据的联合结构变量
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        // 将填充的低位和高位数据存入联合结构中
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        // 调用加载函数，将转换后的数据作为参数传递
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun_lo.to_##T_SFX, pun_hi.to_##T_SFX \
        ));                                                                                 \
    }                                                                                       \
    // 内部联合结构用于类型转换
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

    // 定义一个内联函数 npyv_load2_tillz_##F_SFX，加载并转换数据为目标类型，填充未使用的部分为零
    NPY_FINLINE npyv_##F_SFX npyv_load2_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        // 调用相应的加载函数，将输入指针的数据加载并转换为目标类型
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \

    // 定义一个内联函数 npyv_loadn2_tillz_##F_SFX，加载并转换数据为目标类型，带步长和指定数量
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_tillz_##F_SFX                                      \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        // 调用相应的加载函数，将输入指针的数据加载并转换为目标类型，带有指定步长和数量
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_tillz_##T_SFX(                \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \

    // 定义一个内联函数 npyv_store2_till_##F_SFX，存储数据到目标指针，填充未使用的部分
    NPY_FINLINE void npyv_store2_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {
        npyv_store2_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
    NPY_FINLINE void npyv_storen2_till_##F_SFX
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)
    {
        npyv_storen2_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
    
    
    
    {
        // 使用 `npyv_store2_till_##T_SFX` 宏将数据 `a` 解释为 `T_SFX` 类型后存储到 `ptr` 指向的地址，存储 `nlane` 个元素
        npyv_store2_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
    
    // 使用 `npyv_storen2_till_##F_SFX` 函数将数据 `a` 解释为 `T_SFX` 类型后存储到 `ptr` 指向的地址，每隔 `stride` 存储一个元素，存储 `nlane` 个元素
    NPY_FINLINE void npyv_storen2_till_##F_SFX
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)
    {
        npyv_storen2_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
// 宏定义，用于生成 AVX-512 指令集下的加载和存储操作的代码片段，支持两种数据类型的成对操作
#define NPYV_IMPL_AVX512_REST_PARTIAL_TYPES_PAIR(TYPE1, TYPE2) \
    // 在 AVX-512 指令集下实现 TYPE1 和 TYPE2 数据类型的特定操作

// 宏定义，实现 AVX-512 指令集下的内存交织加载和存储操作
#define NPYV_IMPL_AVX512_MEM_INTERLEAVE(SFX, ZSFX)                            \
    // 定义函数 npyv_zip_##ZSFX 和 npyv_unzip_##ZSFX，用于两通道的交织操作
    NPY_FINLINE npyv_##ZSFX##x2 npyv_zip_##ZSFX(npyv_##ZSFX, npyv_##ZSFX);    \
    NPY_FINLINE npyv_##ZSFX##x2 npyv_unzip_##ZSFX(npyv_##ZSFX, npyv_##ZSFX);  \
    // 定义函数 npyv_load_##SFX##x2，从内存中加载两个连续通道的数据
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                           \
        const npyv_lanetype_##SFX *ptr                                        \
    ) {                                                                       \
        return npyv_unzip_##ZSFX(                                             \
            npyv_load_##SFX(ptr), npyv_load_##SFX(ptr+npyv_nlanes_##SFX)      \
        );                                                                    \
    }                                                                         \
    // 定义函数 npyv_store_##SFX##x2，将两个通道的数据连续存储到内存中
    NPY_FINLINE void npyv_store_##SFX##x2(                                    \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                            \
    ) {                                                                       \
        npyv_##SFX##x2 zip = npyv_zip_##ZSFX(v.val[0], v.val[1]);             \
        npyv_store_##SFX(ptr, zip.val[0]);                                    \
        npyv_store_##SFX(ptr + npyv_nlanes_##SFX, zip.val[1]);                \
    }

// 使用宏定义生成 AVX-512 指令集下的内存交织加载和存储操作的具体实现
NPYV_IMPL_AVX512_MEM_INTERLEAVE(u8, u8)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(s8, u8)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(u16, u16)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(s16, u16)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(u32, u32)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(s32, u32)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(u64, u64)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(s64, u64)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(f32, f32)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(f64, f64)

/**************************************************
 * Lookup table
 *************************************************/

// 使用向量作为索引来访问包含 32 个 float32 元素的查找表
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{
    // 加载表中的前 16 个元素到向量 table0
    const npyv_f32 table0 = npyv_load_f32(table);
    // 加载表中的后 16 个元素到向量 table1
    const npyv_f32 table1 = npyv_load_f32(table + 16);
    // 使用 _mm512_permutex2var_ps 函数按照 idx 的指定顺序对 table0 和 table1 进行混洗操作
    return _mm512_permutex2var_ps(table0, idx, table1);
}

// 使用向量作为索引来访问包含 32 个元素的 uint32 查找表，并将结果转换为 float32 向量返回
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float*)table, idx)); }

// 使用向量作为索引来访问包含 32 个元素的 int32 查找表，并将结果转换为 float32 向量返回
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_f32(npyv_lut32_f32((const float*)table, idx)); }

// 使用向量作为索引来访问包含 16 个元素的 float64 查找表
// （此处代码截断，未完成）
    # 加载一个包含双精度浮点数的表格到向量table0中
    const npyv_f64 table0 = npyv_load_f64(table);
    # 从表格中的第8个位置开始加载双精度浮点数到向量table1中
    const npyv_f64 table1 = npyv_load_f64(table + 8);
    # 使用掩码向量idx对table0和table1中的数据进行混洗，并返回结果
    return _mm512_permutex2var_pd(table0, idx, table1);
// 返回一个npyv_u64类型的向量，其中根据给定的索引从64位整数表中查找值
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }

// 返回一个npyv_s64类型的向量，其中根据给定的索引从64位有符号整数表中查找值
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_AVX512_MEMORY_H
```