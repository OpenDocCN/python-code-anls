# `.\numpy\numpy\_core\src\common\simd\sse\misc.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_MISC_H
#define _NPY_SIMD_SSE_MISC_H

// 定义一个函数宏，用于创建所有通道为零的128位整数向量
#define npyv_zero_u8  _mm_setzero_si128
#define npyv_zero_s8  _mm_setzero_si128
#define npyv_zero_u16 _mm_setzero_si128
#define npyv_zero_s16 _mm_setzero_si128
#define npyv_zero_u32 _mm_setzero_si128
#define npyv_zero_s32 _mm_setzero_si128
#define npyv_zero_u64 _mm_setzero_si128
#define npyv_zero_s64 _mm_setzero_si128
// 定义一个函数宏，用于创建所有通道为零的128位单精度浮点数向量
#define npyv_zero_f32 _mm_setzero_ps
#define npyv_zero_f64 _mm_setzero_pd

// 定义一个函数宏，用于创建所有通道为指定值的8位整数向量
#define npyv_setall_u8(VAL)  _mm_set1_epi8((char)(VAL))
#define npyv_setall_s8(VAL)  _mm_set1_epi8((char)(VAL))
#define npyv_setall_u16(VAL) _mm_set1_epi16((short)(VAL))
#define npyv_setall_s16(VAL) _mm_set1_epi16((short)(VAL))
#define npyv_setall_u32(VAL) _mm_set1_epi32((int)(VAL))
#define npyv_setall_s32(VAL) _mm_set1_epi32((int)(VAL))
// 定义一个函数宏，用于创建所有通道为指定值的单精度浮点数向量
#define npyv_setall_f32 _mm_set1_ps
#define npyv_setall_f64 _mm_set1_pd

// 定义一个内联函数，设置所有通道为指定64位无符号整数的向量
NPY_FINLINE __m128i npyv__setr_epi64(npy_int64 i0, npy_int64 i1);

// 定义一个函数，创建所有通道为指定64位无符号整数的向量
NPY_FINLINE npyv_u64 npyv_setall_u64(npy_uint64 a)
{
#if defined(_MSC_VER) && defined(_M_IX86)
    return npyv__setr_epi64((npy_int64)a, (npy_int64)a);
#else
    return _mm_set1_epi64x((npy_int64)a);
#endif
}

// 定义一个函数，创建所有通道为指定64位有符号整数的向量
NPY_FINLINE npyv_s64 npyv_setall_s64(npy_int64 a)
{
#if defined(_MSC_VER) && defined(_M_IX86)
    return npyv__setr_epi64(a, a);
#else
    return _mm_set1_epi64x((npy_int64)a);
#endif
}

/**
 * 创建具有特定值的每个通道的向量，并为其余通道设置特定值
 *
 * 由 NPYV__SET_FILL_* 生成的参数，如果 _mm_setr_* 被定义为宏，则不会展开。
 */
NPY_FINLINE __m128i npyv__setr_epi8(
    char i0, char i1, char i2,  char i3,  char i4,  char i5,  char i6,  char i7,
    char i8, char i9, char i10, char i11, char i12, char i13, char i14, char i15)
{
    return _mm_setr_epi8(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
}

// 定义一个内联函数，创建具有特定值的每个通道的16位有符号整数向量
NPY_FINLINE __m128i npyv__setr_epi16(short i0, short i1, short i2, short i3, short i4, short i5,
                                     short i6, short i7)
{
    return _mm_setr_epi16(i0, i1, i2, i3, i4, i5, i6, i7);
}

// 定义一个内联函数，创建具有特定值的每个通道的32位有符号整数向量
NPY_FINLINE __m128i npyv__setr_epi32(int i0, int i1, int i2, int i3)
{
    return _mm_setr_epi32(i0, i1, i2, i3);
}

// 定义一个内联函数，创建具有特定值的每个通道的64位有符号整数向量
NPY_FINLINE __m128i npyv__setr_epi64(npy_int64 i0, npy_int64 i1)
{
#if defined(_MSC_VER) && defined(_M_IX86)
    return _mm_setr_epi32((int)i0, (int)(i0 >> 32), (int)i1, (int)(i1 >> 32));
#else
    return _mm_set_epi64x(i1, i0);
#endif
}

// 定义一个内联函数，创建具有特定值的每个通道的32位单精度浮点数向量
NPY_FINLINE __m128 npyv__setr_ps(float i0, float i1, float i2, float i3)
{
    return _mm_setr_ps(i0, i1, i2, i3);
}

// 定义一个内联函数，创建具有特定值的每个通道的64位双精度浮点数向量
NPY_FINLINE __m128d npyv__setr_pd(double i0, double i1)
{
    return _mm_setr_pd(i0, i1);
}

// 定义一个函数宏，创建所有通道为指定填充值的8位整数向量
#define npyv_setf_u8(FILL, ...)  npyv__setr_epi8(NPYV__SET_FILL_16(char, FILL, __VA_ARGS__))
// 定义一个函数宏，创建所有通道为指定填充值的8位整数向量
#define npyv_setf_s8(FILL, ...)  npyv__setr_epi8(NPYV__SET_FILL_16(char, FILL, __VA_ARGS__))

#endif // _NPY_SIMD_SSE_MISC_H
// 定义宏以设置特定数据类型和填充值的向量，并生成适当的函数调用
#define npyv_setf_u16(FILL, ...) npyv__setr_epi16(NPYV__SET_FILL_8(short, FILL, __VA_ARGS__))
#define npyv_setf_s16(FILL, ...) npyv__setr_epi16(NPYV__SET_FILL_8(short, FILL, __VA_ARGS__))
#define npyv_setf_u32(FILL, ...) npyv__setr_epi32(NPYV__SET_FILL_4(int, FILL, __VA_ARGS__))
#define npyv_setf_s32(FILL, ...) npyv__setr_epi32(NPYV__SET_FILL_4(int, FILL, __VA_ARGS__))
#define npyv_setf_u64(FILL, ...) npyv__setr_epi64(NPYV__SET_FILL_2(npy_int64, FILL, __VA_ARGS__))
#define npyv_setf_s64(FILL, ...) npyv__setr_epi64(NPYV__SET_FILL_2(npy_int64, FILL, __VA_ARGS__))
#define npyv_setf_f32(FILL, ...) npyv__setr_ps(NPYV__SET_FILL_4(float, FILL, __VA_ARGS__))
#define npyv_setf_f64(FILL, ...) npyv__setr_pd(NPYV__SET_FILL_2(double, FILL, __VA_ARGS__))

// 定义宏以设置无符号8位整数向量的每个通道，未指定通道使用零填充
#define npyv_set_u8(...)  npyv_setf_u8(0,  __VA_ARGS__)
// 定义宏以设置有符号8位整数向量的每个通道，未指定通道使用零填充
#define npyv_set_s8(...)  npyv_setf_s8(0,  __VA_ARGS__)
// 定义宏以设置无符号16位整数向量的每个通道，未指定通道使用零填充
#define npyv_set_u16(...) npyv_setf_u16(0, __VA_ARGS__)
// 定义宏以设置有符号16位整数向量的每个通道，未指定通道使用零填充
#define npyv_set_s16(...) npyv_setf_s16(0, __VA_ARGS__)
// 定义宏以设置无符号32位整数向量的每个通道，未指定通道使用零填充
#define npyv_set_u32(...) npyv_setf_u32(0, __VA_ARGS__)
// 定义宏以设置有符号32位整数向量的每个通道，未指定通道使用零填充
#define npyv_set_s32(...) npyv_setf_s32(0, __VA_ARGS__)
// 定义宏以设置无符号64位整数向量的每个通道，未指定通道使用零填充
#define npyv_set_u64(...) npyv_setf_u64(0, __VA_ARGS__)
// 定义宏以设置有符号64位整数向量的每个通道，未指定通道使用零填充
#define npyv_set_s64(...) npyv_setf_s64(0, __VA_ARGS__)
// 定义宏以设置单精度浮点数向量的每个通道，未指定通道使用零填充
#define npyv_set_f32(...) npyv_setf_f32(0, __VA_ARGS__)
// 定义宏以设置双精度浮点数向量的每个通道，未指定通道使用零填充
#define npyv_set_f64(...) npyv_setf_f64(0, __VA_ARGS__)

// 如果支持 SSE4.1 指令集，则使用 SSE4.1 的混合向量选择指令
#ifdef NPY_HAVE_SSE41
    #define npyv_select_u8(MASK, A, B)  _mm_blendv_epi8(B, A, MASK)
    #define npyv_select_f32(MASK, A, B) _mm_blendv_ps(B, A, _mm_castsi128_ps(MASK))
    #define npyv_select_f64(MASK, A, B) _mm_blendv_pd(B, A, _mm_castsi128_pd(MASK))
// 否则，定义非内联函数以实现无 SSE4.1 支持的混合向量选择
#else
    NPY_FINLINE __m128i npyv_select_u8(__m128i mask, __m128i a, __m128i b)
    { return _mm_xor_si128(b, _mm_and_si128(_mm_xor_si128(b, a), mask)); }
    NPY_FINLINE __m128 npyv_select_f32(__m128i mask, __m128 a, __m128 b)
    { return _mm_xor_ps(b, _mm_and_ps(_mm_xor_ps(b, a), _mm_castsi128_ps(mask))); }
    NPY_FINLINE __m128d npyv_select_f64(__m128i mask, __m128d a, __m128d b)
    { return _mm_xor_pd(b, _mm_and_pd(_mm_xor_pd(b, a), _mm_castsi128_pd(mask))); }
#endif

// 为不同数据类型的整数向量选择定义宏
#define npyv_select_s8  npyv_select_u8
#define npyv_select_u16 npyv_select_u8
#define npyv_select_s16 npyv_select_u8
#define npyv_select_u32 npyv_select_u8
#define npyv_select_s32 npyv_select_u8
#define npyv_select_u64 npyv_select_u8
#define npyv_select_s64 npyv_select_u8

// 提取向量的第一个通道数据并转换为指定的数据类型
#define npyv_extract0_u8(A) ((npy_uint8)_mm_cvtsi128_si32(A))
#define npyv_extract0_s8(A) ((npy_int8)_mm_cvtsi128_si32(A))
#define npyv_extract0_u16(A) ((npy_uint16)_mm_cvtsi128_si32(A))
#define npyv_extract0_s16(A) ((npy_int16)_mm_cvtsi128_si32(A))
#define npyv_extract0_u32(A) ((npy_uint32)_mm_cvtsi128_si32(A))
#define npyv_extract0_s32(A) ((npy_int32)_mm_cvtsi128_si32(A))
#define npyv_extract0_u64(A) ((npy_uint64)npyv128_cvtsi128_si64(A))
#define npyv_extract0_s64(A) ((npy_int64)npyv128_cvtsi128_si64(A))
// 定义宏，用于从 X 中提取 float32 的值，_mm_cvtss_f32 是 SSE 指令
#define npyv_extract0_f32 _mm_cvtss_f32
// 定义宏，用于从 X 中提取 float64 的值，_mm_cvtsd_f64 是 SSE2 指令
#define npyv_extract0_f64 _mm_cvtsd_f64

// 定义宏，用于将 X 重新解释为 uint8_t 类型
#define npyv_reinterpret_u8_u8(X)  X
// 定义宏，用于将 X 重新解释为 int8_t 类型
#define npyv_reinterpret_u8_s8(X)  X
#define npyv_reinterpret_u8_u16(X) X
#define npyv_reinterpret_u8_s16(X) X
#define npyv_reinterpret_u8_u32(X) X
#define npyv_reinterpret_u8_s32(X) X
#define npyv_reinterpret_u8_u64(X) X
#define npyv_reinterpret_u8_s64(X) X
// 定义宏，用于将 X 重新解释为 float32 类型，_mm_castps_si128 是 SSE 指令
#define npyv_reinterpret_u8_f32 _mm_castps_si128
// 定义宏，用于将 X 重新解释为 float64 类型，_mm_castpd_si128 是 SSE2 指令
#define npyv_reinterpret_u8_f64 _mm_castpd_si128

#define npyv_reinterpret_s8_s8(X)  X
#define npyv_reinterpret_s8_u8(X)  X
#define npyv_reinterpret_s8_u16(X) X
#define npyv_reinterpret_s8_s16(X) X
#define npyv_reinterpret_s8_u32(X) X
#define npyv_reinterpret_s8_s32(X) X
#define npyv_reinterpret_s8_u64(X) X
#define npyv_reinterpret_s8_s64(X) X
#define npyv_reinterpret_s8_f32 _mm_castps_si128
#define npyv_reinterpret_s8_f64 _mm_castpd_si128

#define npyv_reinterpret_u16_u16(X) X
#define npyv_reinterpret_u16_u8(X)  X
#define npyv_reinterpret_u16_s8(X)  X
#define npyv_reinterpret_u16_s16(X) X
#define npyv_reinterpret_u16_u32(X) X
#define npyv_reinterpret_u16_s32(X) X
#define npyv_reinterpret_u16_u64(X) X
#define npyv_reinterpret_u16_s64(X) X
#define npyv_reinterpret_u16_f32 _mm_castps_si128
#define npyv_reinterpret_u16_f64 _mm_castpd_si128

#define npyv_reinterpret_s16_s16(X) X
#define npyv_reinterpret_s16_u8(X)  X
#define npyv_reinterpret_s16_s8(X)  X
#define npyv_reinterpret_s16_u16(X) X
#define npyv_reinterpret_s16_u32(X) X
#define npyv_reinterpret_s16_s32(X) X
#define npyv_reinterpret_s16_u64(X) X
#define npyv_reinterpret_s16_s64(X) X
#define npyv_reinterpret_s16_f32 _mm_castps_si128
#define npyv_reinterpret_s16_f64 _mm_castpd_si128

#define npyv_reinterpret_u32_u32(X) X
#define npyv_reinterpret_u32_u8(X)  X
#define npyv_reinterpret_u32_s8(X)  X
#define npyv_reinterpret_u32_u16(X) X
#define npyv_reinterpret_u32_s16(X) X
#define npyv_reinterpret_u32_s32(X) X
#define npyv_reinterpret_u32_u64(X) X
#define npyv_reinterpret_u32_s64(X) X
#define npyv_reinterpret_u32_f32 _mm_castps_si128
#define npyv_reinterpret_u32_f64 _mm_castpd_si128

#define npyv_reinterpret_s32_s32(X) X
#define npyv_reinterpret_s32_u8(X)  X
#define npyv_reinterpret_s32_s8(X)  X
#define npyv_reinterpret_s32_u16(X) X
#define npyv_reinterpret_s32_s16(X) X
#define npyv_reinterpret_s32_u32(X) X
#define npyv_reinterpret_s32_u64(X) X
#define npyv_reinterpret_s32_s64(X) X
#define npyv_reinterpret_s32_f32 _mm_castps_si128
#define npyv_reinterpret_s32_f64 _mm_castpd_si128

#define npyv_reinterpret_u64_u64(X) X
#define npyv_reinterpret_u64_u8(X)  X
#define npyv_reinterpret_u64_s8(X)  X
#define npyv_reinterpret_u64_u16(X) X
#define npyv_reinterpret_u64_s16(X) X
#define npyv_reinterpret_u64_u32(X) X
#define npyv_reinterpret_u64_s32(X) X
#define npyv_reinterpret_u64_s64(X) X
#define npyv_reinterpret_u64_f32 _mm_castps_si128
#define npyv_reinterpret_u64_f64 _mm_castpd_si128

#define npyv_reinterpret_s64_s64(X) X
#define npyv_reinterpret_s64_u8(X)  X
// 定义了一系列宏，用于将不同类型数据重新解释为 64 位有符号整数或者浮点数，具体实现为直接返回传入的参数 X。
#define npyv_reinterpret_s64_s8(X)  X
#define npyv_reinterpret_s64_u16(X) X
#define npyv_reinterpret_s64_s16(X) X
#define npyv_reinterpret_s64_u32(X) X
#define npyv_reinterpret_s64_s32(X) X
#define npyv_reinterpret_s64_u64(X) X
#define npyv_reinterpret_s64_f32 _mm_castps_si128
#define npyv_reinterpret_s64_f64 _mm_castpd_si128

// 定义了一系列宏，用于将不同类型数据重新解释为 32 位浮点数，具体实现为使用 SSE 指令进行转换。
#define npyv_reinterpret_f32_f32(X) X
#define npyv_reinterpret_f32_u8  _mm_castsi128_ps
#define npyv_reinterpret_f32_s8  _mm_castsi128_ps
#define npyv_reinterpret_f32_u16 _mm_castsi128_ps
#define npyv_reinterpret_f32_s16 _mm_castsi128_ps
#define npyv_reinterpret_f32_u32 _mm_castsi128_ps
#define npyv_reinterpret_f32_s32 _mm_castsi128_ps
#define npyv_reinterpret_f32_u64 _mm_castsi128_ps
#define npyv_reinterpret_f32_s64 _mm_castsi128_ps
#define npyv_reinterpret_f32_f64 _mm_castpd_ps

// 定义了一系列宏，用于将不同类型数据重新解释为 64 位浮点数，具体实现为使用 SSE 指令进行转换。
#define npyv_reinterpret_f64_f64(X) X
#define npyv_reinterpret_f64_u8  _mm_castsi128_pd
#define npyv_reinterpret_f64_s8  _mm_castsi128_pd
#define npyv_reinterpret_f64_u16 _mm_castsi128_pd
#define npyv_reinterpret_f64_s16 _mm_castsi128_pd
#define npyv_reinterpret_f64_u32 _mm_castsi128_pd
#define npyv_reinterpret_f64_s32 _mm_castsi128_pd
#define npyv_reinterpret_f64_u64 _mm_castsi128_pd
#define npyv_reinterpret_f64_s64 _mm_castsi128_pd
#define npyv_reinterpret_f64_f32 _mm_castps_pd

// 仅在 AVX2/AVX512 环境下需要调用的宏，此处定义为空操作。
#define npyv_cleanup() ((void)0)

// 头文件结束的标记，用于防止头文件重复包含。
#endif // _NPY_SIMD_SSE_MISC_H
```