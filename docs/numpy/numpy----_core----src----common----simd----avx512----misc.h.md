# `.\numpy\numpy\_core\src\common\simd\avx512\misc.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_MISC_H
#define _NPY_SIMD_AVX512_MISC_H

// 定义了一系列宏用于将 AVX-512 向量的所有元素设置为零
#define npyv_zero_u8  _mm512_setzero_si512
#define npyv_zero_s8  _mm512_setzero_si512
#define npyv_zero_u16 _mm512_setzero_si512
#define npyv_zero_s16 _mm512_setzero_si512
#define npyv_zero_u32 _mm512_setzero_si512
#define npyv_zero_s32 _mm512_setzero_si512
#define npyv_zero_u64 _mm512_setzero_si512
#define npyv_zero_s64 _mm512_setzero_si512
#define npyv_zero_f32 _mm512_setzero_ps
#define npyv_zero_f64 _mm512_setzero_pd

// 定义了一系列宏用于将 AVX-512 向量的所有元素设置为相同的值
#define npyv_setall_u8(VAL)  _mm512_set1_epi8((char)VAL)
#define npyv_setall_s8(VAL)  _mm512_set1_epi8((char)VAL)
#define npyv_setall_u16(VAL) _mm512_set1_epi16((short)VAL)
#define npyv_setall_s16(VAL) _mm512_set1_epi16((short)VAL)
#define npyv_setall_u32(VAL) _mm512_set1_epi32((int)VAL)
#define npyv_setall_s32(VAL) _mm512_set1_epi32(VAL)
#define npyv_setall_f32(VAL) _mm512_set1_ps(VAL)
#define npyv_setall_f64(VAL) _mm512_set1_pd(VAL)

// 在一些编译器中缺少了 _mm512_set_epi8 和 _mm512_set_epi16 函数，这里定义了一个特定值的向量函数
NPY_FINLINE __m512i npyv__setr_epi64(
    npy_int64, npy_int64, npy_int64, npy_int64,
    npy_int64, npy_int64, npy_int64, npy_int64
);
// 设置所有 AVX-512 64 位无符号整数向量的元素为同一个值
NPY_FINLINE npyv_u64 npyv_setall_u64(npy_uint64 a)
{
    npy_int64 ai = (npy_int64)a;
#if defined(_MSC_VER) && defined(_M_IX86)
    return npyv__setr_epi64(ai, ai, ai, ai, ai, ai, ai, ai);
#else
    return _mm512_set1_epi64(ai);
#endif
}
// 设置所有 AVX-512 64 位有符号整数向量的元素为同一个值
NPY_FINLINE npyv_s64 npyv_setall_s64(npy_int64 a)
{
#if defined(_MSC_VER) && defined(_M_IX86)
    return npyv__setr_epi64(a, a, a, a, a, a, a, a);
#else
    return _mm512_set1_epi64(a);
#endif
}
/**
 * 设置 AVX-512 8 位整数向量的每个通道为特定值，并将其余通道设置为特定值
 *
 * 在许多编译器中缺少了 _mm512_set_epi8 和 _mm512_set_epi16 函数
 */
NPY_FINLINE __m512i npyv__setr_epi8(
    char i0,  char i1,  char i2,  char i3,  char i4,  char i5,  char i6,  char i7,
    char i8,  char i9,  char i10, char i11, char i12, char i13, char i14, char i15,
    char i16, char i17, char i18, char i19, char i20, char i21, char i22, char i23,
    char i24, char i25, char i26, char i27, char i28, char i29, char i30, char i31,
    char i32, char i33, char i34, char i35, char i36, char i37, char i38, char i39,
    char i40, char i41, char i42, char i43, char i44, char i45, char i46, char i47,
    char i48, char i49, char i50, char i51, char i52, char i53, char i54, char i55,
    char i56, char i57, char i58, char i59, char i60, char i61, char i62, char i63)
{
    // 将输入的字符数组按照 AVX-512 要求对齐后加载为向量
    const char NPY_DECL_ALIGNED(64) data[64] = {
        i0,  i1,  i2,  i3,  i4,  i5,  i6,  i7,  i8,  i9,  i10, i11, i12, i13, i14, i15,
        i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31,
        i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43, i44, i45, i46, i47,
        i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58, i59, i60, i61, i62, i63
    };
    // 加载数据到 AVX-512 整数向量
    return _mm512_load_si512((const void*)data);
}

#endif // _NPY_SIMD_AVX512_MISC_H
NPY_FINLINE __m512i npyv__setr_epi16(
    short i0,  short i1,  short i2,  short i3,  short i4,  short i5,  short i6,  short i7,
    short i8,  short i9,  short i10, short i11, short i12, short i13, short i14, short i15,
    short i16, short i17, short i18, short i19, short i20, short i21, short i22, short i23,
    short i24, short i25, short i26, short i27, short i28, short i29, short i30, short i31)
{
    // 创建包含32个元素的数组，每个元素是输入的短整型参数
    const short NPY_DECL_ALIGNED(64) data[32] = {
        i0,  i1,  i2,  i3,  i4,  i5,  i6,  i7,  i8,  i9,  i10, i11, i12, i13, i14, i15,
        i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31
    };
    // 使用 AVX-512 指令集加载包含在数组中的数据到 __m512i 类型的寄存器
    return _mm512_load_si512((const void*)data);
}

// 如果 _mm512_setr_* 被定义为宏，则由于宏不会展开生成的参数。
NPY_FINLINE __m512i npyv__setr_epi32(
    int i0, int i1, int i2,  int i3,  int i4,  int i5,  int i6,  int i7,
    int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15)
{
    // 使用输入的整型参数创建一个 __m512i 类型的寄存器
    return _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
}

// 使用输入的 64 位整型参数创建一个 __m512i 类型的寄存器
NPY_FINLINE __m512i npyv__setr_epi64(npy_int64 i0, npy_int64 i1, npy_int64 i2, npy_int64 i3,
                                     npy_int64 i4, npy_int64 i5, npy_int64 i6, npy_int64 i7)
{
    // 如果编译器为 MSVC 且目标平台为 x86，则使用 _mm512_setr_epi32 创建 __m512i 类型的寄存器
    // 否则，使用 _mm512_setr_epi64 创建 __m512i 类型的寄存器
#if defined(_MSC_VER) && defined(_M_IX86)
    return _mm512_setr_epi32(
        (int)i0, (int)(i0 >> 32), (int)i1, (int)(i1 >> 32),
        (int)i2, (int)(i2 >> 32), (int)i3, (int)(i3 >> 32),
        (int)i4, (int)(i4 >> 32), (int)i5, (int)(i5 >> 32),
        (int)i6, (int)(i6 >> 32), (int)i7, (int)(i7 >> 32)
    );
#else
    return _mm512_setr_epi64(i0, i1, i2, i3, i4, i5, i6, i7);
#endif
}

// 使用输入的单精度浮点参数创建一个 __m512 类型的寄存器
NPY_FINLINE __m512 npyv__setr_ps(
    float i0, float i1, float i2,  float i3,  float i4,  float i5,  float i6,  float i7,
    float i8, float i9, float i10, float i11, float i12, float i13, float i14, float i15)
{
    // 使用 _mm512_setr_ps 创建一个 __m512 类型的寄存器，包含输入的单精度浮点参数
    return _mm512_setr_ps(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
}

// 使用输入的双精度浮点参数创建一个 __m512d 类型的寄存器
NPY_FINLINE __m512d npyv__setr_pd(double i0, double i1, double i2, double i3,
                                  double i4, double i5, double i6, double i7)
{
    // 使用 _mm512_setr_pd 创建一个 __m512d 类型的寄存器，包含输入的双精度浮点参数
    return _mm512_setr_pd(i0, i1, i2, i3, i4, i5, i6, i7);
}

// 宏定义，用于将填充值 FILL 和可变参数展开为对应的 npyv__setr_epi8 函数调用
#define npyv_setf_u8(FILL, ...)  npyv__setr_epi8(NPYV__SET_FILL_64(char, FILL, __VA_ARGS__))
#define npyv_setf_s8(FILL, ...)  npyv__setr_epi8(NPYV__SET_FILL_64(char, FILL, __VA_ARGS__))
#define npyv_setf_u16(FILL, ...) npyv__setr_epi16(NPYV__SET_FILL_32(short, FILL, __VA_ARGS__))
#define npyv_setf_s16(FILL, ...) npyv__setr_epi16(NPYV__SET_FILL_32(short, FILL, __VA_ARGS__))
#define npyv_setf_u32(FILL, ...) npyv__setr_epi32(NPYV__SET_FILL_16(int, FILL, __VA_ARGS__))
#define npyv_setf_s32(FILL, ...) npyv__setr_epi32(NPYV__SET_FILL_16(int, FILL, __VA_ARGS__))
#define npyv_setf_u64(FILL, ...) npyv__setr_epi64(NPYV__SET_FILL_8(npy_int64, FILL, __VA_ARGS__))
#define npyv_setf_s64(FILL, ...) npyv__setr_epi64(NPYV__SET_FILL_8(npy_int64, FILL, __VA_ARGS__))
// 定义宏 npyv_setf_f32，用于设置单精度浮点数向量中的值，调用 npyv__setr_ps 宏来实现
#define npyv_setf_f32(FILL, ...) npyv__setr_ps(NPYV__SET_FILL_16(float, FILL, __VA_ARGS__))
// 定义宏 npyv_setf_f64，用于设置双精度浮点数向量中的值，调用 npyv__setr_pd 宏来实现
#define npyv_setf_f64(FILL, ...) npyv__setr_pd(NPYV__SET_FILL_8(double, FILL, __VA_ARGS__))

// 定义宏 npyv_set_u8，将 npyv_setf_u8 宏调用为设置无符号8位整数向量中的值，0表示其余所有通道为0
#define npyv_set_u8(...)  npyv_setf_u8(0,  __VA_ARGS__)
// 定义宏 npyv_set_s8，将 npyv_setf_s8 宏调用为设置有符号8位整数向量中的值，0表示其余所有通道为0
#define npyv_set_s8(...)  npyv_setf_s8(0,  __VA_ARGS__)
// 定义宏 npyv_set_u16，将 npyv_setf_u16 宏调用为设置无符号16位整数向量中的值，0表示其余所有通道为0
#define npyv_set_u16(...) npyv_setf_u16(0, __VA_ARGS__)
// 定义宏 npyv_set_s16，将 npyv_setf_s16 宏调用为设置有符号16位整数向量中的值，0表示其余所有通道为0
#define npyv_set_s16(...) npyv_setf_s16(0, __VA_ARGS__)
// 定义宏 npyv_set_u32，将 npyv_setf_u32 宏调用为设置无符号32位整数向量中的值，0表示其余所有通道为0
#define npyv_set_u32(...) npyv_setf_u32(0, __VA_ARGS__)
// 定义宏 npyv_set_s32，将 npyv_setf_s32 宏调用为设置有符号32位整数向量中的值，0表示其余所有通道为0
#define npyv_set_s32(...) npyv_setf_s32(0, __VA_ARGS__)
// 定义宏 npyv_set_u64，将 npyv_setf_u64 宏调用为设置无符号64位整数向量中的值，0表示其余所有通道为0
#define npyv_set_u64(...) npyv_setf_u64(0, __VA_ARGS__)
// 定义宏 npyv_set_s64，将 npyv_setf_s64 宏调用为设置有符号64位整数向量中的值，0表示其余所有通道为0
#define npyv_set_s64(...) npyv_setf_s64(0, __VA_ARGS__)
// 定义宏 npyv_set_f32，将 npyv_setf_f32 宏调用为设置单精度浮点数向量中的值，0表示其余所有通道为0
#define npyv_set_f32(...) npyv_setf_f32(0, __VA_ARGS__)
// 定义宏 npyv_set_f64，将 npyv_setf_f64 宏调用为设置双精度浮点数向量中的值，0表示其余所有通道为0
#define npyv_set_f64(...) npyv_setf_f64(0, __VA_ARGS__)

// 根据是否支持 AVX512BW 指令集选择不同的宏实现
#ifdef NPY_HAVE_AVX512BW
    // 如果支持 AVX512BW，定义宏 npyv_select_u8，使用 _mm512_mask_blend_epi8 实现按位选择无符号8位整数向量
    #define npyv_select_u8(MASK, A, B)  _mm512_mask_blend_epi8(MASK,  B, A)
    // 定义宏 npyv_select_u16，使用 _mm512_mask_blend_epi16 实现按位选择无符号16位整数向量
    #define npyv_select_u16(MASK, A, B) _mm512_mask_blend_epi16(MASK, B, A)
#else
    // 如果不支持 AVX512BW，定义函数 npyv_select_u8，使用 _mm512_xor_si512 和 _mm512_and_si512 实现按位选择无符号8位整数向量
    NPY_FINLINE __m512i npyv_select_u8(__m512i mask, __m512i a, __m512i b)
    { return _mm512_xor_si512(b, _mm512_and_si512(_mm512_xor_si512(b, a), mask)); }
    // 定义宏 npyv_select_u16，与 npyv_select_u8 相同
    #define npyv_select_u16 npyv_select_u8
#endif
// 定义宏 npyv_select_s8，与 npyv_select_u8 相同
#define npyv_select_s8  npyv_select_u8
// 定义宏 npyv_select_s16，与 npyv_select_u16 相同
#define npyv_select_s16 npyv_select_u16
// 定义宏 npyv_select_u32，使用 _mm512_mask_blend_epi32 实现按位选择无符号32位整数向量
#define npyv_select_u32(MASK, A, B) _mm512_mask_blend_epi32(MASK, B, A)
// 定义宏 npyv_select_s32，与 npyv_select_u32 相同
#define npyv_select_s32 npyv_select_u32
// 定义宏 npyv_select_u64，使用 _mm512_mask_blend_epi64 实现按位选择无符号64位整数向量
#define npyv_select_u64(MASK, A, B) _mm512_mask_blend_epi64(MASK, B, A)
// 定义宏 npyv_select_s64，与 npyv_select_u64 相同
#define npyv_select_s64 npyv_select_u64
// 定义宏 npyv_select_f32，使用 _mm512_mask_blend_ps 实现按位选择单精度浮点数向量
#define npyv_select_f32(MASK, A, B) _mm512_mask_blend_ps(MASK, B, A)
// 定义宏 npyv_select_f64，使用 _mm512_mask_blend_pd 实现按位选择双精度浮点数向量
#define npyv_select_f64(MASK, A, B) _mm512_mask_blend_pd(MASK, B, A)

// 提取第一个向量的第一个通道值
#define npyv_extract0_u8(A) ((npy_uint8)_mm_cvtsi128_si32(_mm512_castsi512_si128(A)))
#define npyv_extract0_s8(A) ((npy_int8)_mm_cvtsi128_si32(_mm512_castsi512_si128(A)))
#define npyv_extract0_u16(A) ((npy_uint16)_mm_cvtsi128_si32(_mm512_castsi512_si128(A)))
#define npyv_extract0_s16(A) ((npy_int16)_mm_cvtsi128_si32(_mm512_castsi512_si128(A)))
#define npyv_extract0_u32(A) ((npy_uint32)_mm_cvtsi128_si32(_mm512_castsi512_si128(A)))
#define npyv_extract0_s32(A) ((npy_int32)_mm_cvtsi128_si32(_mm512_castsi512_si128(A)))
#define npyv_extract0_u64(A) ((npy_uint64)npyv128_cvtsi128_si64(_mm512_castsi512_si128(A)))
#define npyv_extract0_s64(A) ((npy_int64)npyv128_cvtsi128_si64(_mm512_castsi512_si128(A)))
#define npyv_extract0_f32(A) _mm_cvtss_f32(_mm512_castps512_ps128(A))
#define npyv_extract0_f64(A) _mm_cvtsd_f64(_mm512_castpd512_pd128(A))

// 重新解释宏，将输入参数重新解释为无符号8位整数向量
#define npyv_reinterpret_u8_u8(X)  X
// 将输入参数重新解释为有符号8位整数向量
#define npyv_reinterpret_u8_s8(X)  X
# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s8_s8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s8_u8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s8_u16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s8_s16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s8_u32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s8_s32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s8_u64(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s8_s64(X) X

# 宏定义，将输入参数按 512 位转换为单精度浮点数向量
#define npyv_reinterpret_s8_f32 _mm512_castps_si512

# 宏定义，将输入参数按 512 位转换为双精度浮点数向量
#define npyv_reinterpret_s8_f64 _mm512_castpd_si512

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u16_u16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u16_u8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u16_s8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u16_s16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u16_u32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u16_s32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u16_u64(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u16_s64(X) X

# 宏定义，将输入参数按 512 位转换为单精度浮点数向量
#define npyv_reinterpret_u16_f32 _mm512_castps_si512

# 宏定义，将输入参数按 512 位转换为双精度浮点数向量
#define npyv_reinterpret_u16_f64 _mm512_castpd_si512

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s16_s16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s16_u8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s16_s8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s16_u16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s16_u32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s16_s32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s16_u64(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s16_s64(X) X

# 宏定义，将输入参数按 512 位转换为单精度浮点数向量
#define npyv_reinterpret_s16_f32 _mm512_castps_si512

# 宏定义，将输入参数按 512 位转换为双精度浮点数向量
#define npyv_reinterpret_s16_f64 _mm512_castpd_si512

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u32_u32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u32_u8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u32_s8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u32_u16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u32_s16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u32_s32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u32_u64(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u32_s64(X) X

# 宏定义，将输入参数按 512 位转换为单精度浮点数向量
#define npyv_reinterpret_u32_f32 _mm512_castps_si512

# 宏定义，将输入参数按 512 位转换为双精度浮点数向量
#define npyv_reinterpret_u32_f64 _mm512_castpd_si512

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s32_s32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s32_u8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s32_s8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s32_u16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s32_s16(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s32_u32(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s32_u64(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_s32_s64(X) X

# 宏定义，将输入参数按 512 位转换为单精度浮点数向量
#define npyv_reinterpret_s32_f32 _mm512_castps_si512

# 宏定义，将输入参数按 512 位转换为双精度浮点数向量
#define npyv_reinterpret_s32_f64 _mm512_castpd_si512

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u64_u64(X) X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define npyv_reinterpret_u64_u8(X)  X

# 宏定义，将输入参数直接作为输出，不做任何类型转换
#define np
#define npyv_reinterpret_f32_s8  _mm512_castsi512_ps
// 定义将 512 位整数转换为单精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f32_u16 _mm512_castsi512_ps
// 定义将 512 位整数转换为单精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f32_s16 _mm512_castsi512_ps
// 定义将 512 位整数转换为单精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f32_u32 _mm512_castsi512_ps
// 定义将 512 位整数转换为单精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f32_s32 _mm512_castsi512_ps
// 定义将 512 位整数转换为单精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f32_u64 _mm512_castsi512_ps
// 定义将 512 位整数转换为单精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f32_s64 _mm512_castsi512_ps
// 定义将 512 位整数转换为单精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f32_f64 _mm512_castpd_ps
// 定义将 512 位双精度浮点数转换为单精度浮点数的宏，使用 AVX-512 指令集

#define npyv_reinterpret_f64_f64(X) X
// 定义将双精度浮点数保持不变的宏
#define npyv_reinterpret_f64_u8  _mm512_castsi512_pd
// 定义将 512 位整数转换为双精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f64_s8  _mm512_castsi512_pd
// 定义将 512 位整数转换为双精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f64_u16 _mm512_castsi512_pd
// 定义将 512 位整数转换为双精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f64_s16 _mm512_castsi512_pd
// 定义将 512 位整数转换为双精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f64_u32 _mm512_castsi512_pd
// 定义将 512 位整数转换为双精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f64_s32 _mm512_castsi512_pd
// 定义将 512 位整数转换为双精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f64_u64 _mm512_castsi512_pd
// 定义将 512 位整数转换为双精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f64_s64 _mm512_castsi512_pd
// 定义将 512 位整数转换为双精度浮点数的宏，使用 AVX-512 指令集
#define npyv_reinterpret_f64_f32 _mm512_castps_pd
// 定义将单精度浮点数转换为双精度浮点数的宏，使用 AVX-512 指令集

#ifdef NPY_HAVE_AVX512_KNL
    #define npyv_cleanup() ((void)0)
#else
    #define npyv_cleanup _mm256_zeroall
#endif
// 如果定义了 NPY_HAVE_AVX512_KNL，则定义 npyv_cleanup 为空操作；否则定义为 _mm256_zeroall，该函数清空 AVX 寄存器

#endif // _NPY_SIMD_AVX512_MISC_H
// 结束条件编译指令，用于保护头文件内容不被重复引入
```