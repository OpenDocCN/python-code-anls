# `.\numpy\numpy\_core\src\common\simd\avx2\reorder.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_REORDER_H
#define _NPY_SIMD_AVX2_REORDER_H

// 定义宏：将两个向量的低部分组合起来
#define npyv_combinel_u8(A, B) _mm256_permute2x128_si256(A, B, 0x20)
#define npyv_combinel_s8  npyv_combinel_u8
#define npyv_combinel_u16 npyv_combinel_u8
#define npyv_combinel_s16 npyv_combinel_u8
#define npyv_combinel_u32 npyv_combinel_u8
#define npyv_combinel_s32 npyv_combinel_u8
#define npyv_combinel_u64 npyv_combinel_u8
#define npyv_combinel_s64 npyv_combinel_u8
#define npyv_combinel_f32(A, B) _mm256_permute2f128_ps(A, B, 0x20)
#define npyv_combinel_f64(A, B) _mm256_permute2f128_pd(A, B, 0x20)

// 定义宏：将两个向量的高部分组合起来
#define npyv_combineh_u8(A, B) _mm256_permute2x128_si256(A, B, 0x31)
#define npyv_combineh_s8  npyv_combineh_u8
#define npyv_combineh_u16 npyv_combineh_u8
#define npyv_combineh_s16 npyv_combineh_u8
#define npyv_combineh_u32 npyv_combineh_u8
#define npyv_combineh_s32 npyv_combineh_u8
#define npyv_combineh_u64 npyv_combineh_u8
#define npyv_combineh_s64 npyv_combineh_u8
#define npyv_combineh_f32(A, B) _mm256_permute2f128_ps(A, B, 0x31)
#define npyv_combineh_f64(A, B) _mm256_permute2f128_pd(A, B, 0x31)

// 定义函数：将两个__m256i类型的向量a和b的低高部分分别组合起来，返回一个npyv_m256ix2结构体
NPY_FINLINE npyv_m256ix2 npyv__combine(__m256i a, __m256i b)
{
    npyv_m256ix2 r;
    // 将a和b的低高部分组合成a1b0
    __m256i a1b0 = _mm256_permute2x128_si256(a, b, 0x21);
    // 使用_blend_epi32混合a和a1b0的高低部分，将结果存入r.val[0]
    r.val[0] = _mm256_blend_epi32(a, a1b0, 0xF0);
    // 使用_blend_epi32混合b和a1b0的高低部分，将结果存入r.val[1]
    r.val[1] = _mm256_blend_epi32(b, a1b0, 0xF);
    return r;
}

// 定义函数：将两个__m256类型的向量a和b的低高部分分别组合起来，返回一个npyv_f32x2结构体
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m256 a, __m256 b)
{
    npyv_f32x2 r;
    // 将a和b的低高部分组合成a1b0
    __m256 a1b0 = _mm256_permute2f128_ps(a, b, 0x21);
    // 使用_blend_ps混合a和a1b0的高低部分，将结果存入r.val[0]
    r.val[0] = _mm256_blend_ps(a, a1b0, 0xF0);
    // 使用_blend_ps混合b和a1b0的高低部分，将结果存入r.val[1]
    r.val[1] = _mm256_blend_ps(b, a1b0, 0xF);
    return r;
}

// 定义函数：将两个__m256d类型的向量a和b的低高部分分别组合起来，返回一个npyv_f64x2结构体
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m256d a, __m256d b)
{
    npyv_f64x2 r;
    // 将a和b的低高部分组合成a1b0
    __m256d a1b0 = _mm256_permute2f128_pd(a, b, 0x21);
    // 使用_blend_pd混合a和a1b0的高低部分，将结果存入r.val[0]
    r.val[0] = _mm256_blend_pd(a, a1b0, 0xC);
    // 使用_blend_pd混合b和a1b0的高低部分，将结果存入r.val[1]
    r.val[1] = _mm256_blend_pd(b, a1b0, 0x3);
    return r;
}

// 定义宏：实现AVX2下的ZIP操作，将向量a和b的低高部分交织在一起，返回一个T_VEC##x2结构体
#define NPYV_IMPL_AVX2_ZIP_U(T_VEC, LEN)                    \
    NPY_FINLINE T_VEC##x2 npyv_zip_u##LEN(T_VEC a, T_VEC b) \
    {                                                       \
        // 将向量a和b的低部分和高部分分别解包并交织在一起，得到ab0和ab1
        __m256i ab0 = _mm256_unpacklo_epi##LEN(a, b);       \
        __m256i ab1 = _mm256_unpackhi_epi##LEN(a, b);       \
        // 使用npyv__combine将ab0和ab1的低高部分组合在一起，返回结果
        return npyv__combine(ab0, ab1);                     \
    }

// 为不同长度的T_VEC类型定义具体的ZIP函数实现
NPYV_IMPL_AVX2_ZIP_U(npyv_u8,  8)
NPYV_IMPL_AVX2_ZIP_U(npyv_u16, 16)
NPYV_IMPL_AVX2_ZIP_U(npyv_u32, 32)
NPYV_IMPL_AVX2_ZIP_U(npyv_u64, 64)
#define npyv_zip_s8  npyv_zip_u8
#define npyv_zip_s16 npyv_zip_u16
#define npyv_zip_s32 npyv_zip_u32

#endif // _NPY_SIMD_AVX2_REORDER_H
// 定义一个宏，将有符号64位整数向无符号64位整数的转换别名为npv_zip_s64
#define npyv_zip_s64 npyv_zip_u64

// 定义一个内联函数，将两个256位单精度浮点数向量a和b解交错
NPY_FINLINE npyv_f32x2 npyv_zip_f32(__m256 a, __m256 b)
{
    // 解交错操作，取出a和b的低128位和高128位，分别放入ab0和ab1
    __m256 ab0 = _mm256_unpacklo_ps(a, b);
    __m256 ab1 = _mm256_unpackhi_ps(a, b);
    // 调用npv_combine_f32函数将ab0和ab1组合成一个新的256位单精度浮点数向量
    return npyv_combine_f32(ab0, ab1);
}

// 定义一个内联函数，将两个256位双精度浮点数向量a和b解交错
NPY_FINLINE npyv_f64x2 npyv_zip_f64(__m256d a, __m256d b)
{
    // 解交错操作，取出a和b的低128位和高128位，分别放入ab0和ab1
    __m256d ab0 = _mm256_unpacklo_pd(a, b);
    __m256d ab1 = _mm256_unpackhi_pd(a, b);
    // 调用npv_combine_f64函数将ab0和ab1组合成一个新的256位双精度浮点数向量
    return npyv_combine_f64(ab0, ab1);
}

// 定义一个内联函数，将两个8位无符号整数向量ab0和ab1进行解插值
NPY_FINLINE npyv_u8x2 npyv_unzip_u8(npyv_u8 ab0, npyv_u8 ab1)
{
    // 创建一个常量256位整数向量idx，其排列顺序按照指定的索引
    const __m256i idx = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
    );
    // 对ab0和ab1按照idx的顺序进行字节级重排
    __m256i ab_03 = _mm256_shuffle_epi8(ab0, idx);
    __m256i ab_12 = _mm256_shuffle_epi8(ab1, idx);
    // 调用npyv_combine_u8将ab_03和ab_12组合成一个新的8位无符号整数向量对
    npyv_u8x2 ab_lh = npyv_combine_u8(ab_03, ab_12);
    npyv_u8x2 r;
    // 将ab_lh的val[0]和val[1]分别按64位整数进行解交错，存入r.val[0]和r.val[1]
    r.val[0] = _mm256_unpacklo_epi64(ab_lh.val[0], ab_lh.val[1]);
    r.val[1] = _mm256_unpackhi_epi64(ab_lh.val[0], ab_lh.val[1]);
    return r;
}
// 定义一个宏，将有符号8位整数向无符号8位整数的转换别名为npv_unzip_s8
#define npyv_unzip_s8 npyv_unzip_u8

// 定义一个内联函数，将两个16位无符号整数向量ab0和ab1进行解插值
NPY_FINLINE npyv_u16x2 npyv_unzip_u16(npyv_u16 ab0, npyv_u16 ab1)
{
    // 创建一个常量256位整数向量idx，其排列顺序按照指定的索引
    const __m256i idx = _mm256_setr_epi8(
        0,1, 4,5, 8,9, 12,13, 2,3, 6,7, 10,11, 14,15,
        0,1, 4,5, 8,9, 12,13, 2,3, 6,7, 10,11, 14,15
    );
    // 对ab0和ab1按照idx的顺序进行字节级重排
    __m256i ab_03 = _mm256_shuffle_epi8(ab0, idx);
    __m256i ab_12 = _mm256_shuffle_epi8(ab1, idx);
    // 调用npyv_combine_u16将ab_03和ab_12组合成一个新的16位无符号整数向量对
    npyv_u16x2 ab_lh = npyv_combine_u16(ab_03, ab_12);
    npyv_u16x2 r;
    // 将ab_lh的val[0]和val[1]分别按64位整数进行解交错，存入r.val[0]和r.val[1]
    r.val[0] = _mm256_unpacklo_epi64(ab_lh.val[0], ab_lh.val[1]);
    r.val[1] = _mm256_unpackhi_epi64(ab_lh.val[0], ab_lh.val[1]);
    return r;
}
// 定义一个宏，将有符号16位整数向无符号16位整数的转换别名为npv_unzip_s16
#define npyv_unzip_s16 npyv_unzip_u16

// 定义一个内联函数，将两个32位无符号整数向量ab0和ab1进行解插值
NPY_FINLINE npyv_u32x2 npyv_unzip_u32(npyv_u32 ab0, npyv_u32 ab1)
{
    // 创建一个常量256位整数向量idx，按照指定的索引对ab0和ab1进行32位整数级的重排
    const __m256i idx = npyv_set_u32(0, 2, 4, 6, 1, 3, 5, 7);
    __m256i abl = _mm256_permutevar8x32_epi32(ab0, idx);
    __m256i abh = _mm256_permutevar8x32_epi32(ab1, idx);
    // 调用npyv_combine_u32将abl和abh组合成一个新的32位无符号整数向量对
    return npyv_combine_u32(abl, abh);
}
// 定义一个宏，将有符号32位整数向无符号32位整数的转换别名为npv_unzip_s32
#define npyv_unzip_s32 npyv_unzip_u32

// 定义一个内联函数，将两个64位无符号整数向量ab0和ab1进行解插值
NPY_FINLINE npyv_u64x2 npyv_unzip_u64(npyv_u64 ab0, npyv_u64 ab1)
{
    // 调用npyv_combine_u64将ab0和ab1组合成一个新的64位无符号整数向量对
    npyv_u64x2 ab_lh = npyv_combine_u64(ab0, ab1);
    npyv_u64x2 r;
    // 将ab_lh的val[0]和val[1]分别按64位整数进行解交错，存入r.val[0]和r.val[1]
    r.val[0] = _mm256_unpacklo_epi64(ab_lh.val[0], ab_lh.val[1]);
    r.val[1] = _mm256_unpackhi_epi64(ab_lh.val[0], ab_lh.val[1]);
    return r;
}
// 定义一个宏，将有符号64位整数向无符号64位整数的转换别名为npv_unzip_s64
#define npyv_unzip_s64 npyv_unzip_u64

// 定义一个内联函数，将两个32位单精度浮点数向量ab0和ab1进行解插值
NPY_FINLINE npyv_f32x2 npyv_unzip_f32(npyv_f32 ab
    # 创建一个 256 位整数类型的常量 idx，使用 _mm256_setr_epi8 函数设置它的值
    # 这个常量用于定义一个特定的字节重排顺序，具体顺序如下：
    # 从左到右分两行设置，每行包含 16 个字节索引，第一行从 7 到 0，第二行从 15 到 8
    const __m256i idx = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8
    );
    # 使用 AVX2 指令集中的 _mm256_shuffle_epi8 函数，按照 idx 的指定顺序对参数 a 进行字节重排
    return _mm256_shuffle_epi8(a, idx);
// 定义宏 npyv_rev64_s8，将其映射到 npyv_rev64_u8 宏
#define npyv_rev64_s8 npyv_rev64_u8

// 定义函数 npyv_rev64_u16，反转参数中的 16 位元素顺序
NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
    // 创建一个常量 __m256i 类型的索引，用于反转 16 位元素的顺序
    const __m256i idx = _mm256_setr_epi8(
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9,
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9
    );
    // 使用 AVX2 指令集中的 _mm256_shuffle_epi8 对 a 应用 idx 索引进行元素反转
    return _mm256_shuffle_epi8(a, idx);
}
// 定义宏 npyv_rev64_s16，将其映射到 npyv_rev64_u16 宏

// 定义函数 npyv_rev64_u32，反转参数中的 32 位元素顺序
NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    // 使用 AVX2 指令集中的 _mm256_shuffle_epi32 对 a 进行元素反转，模式为 _MM_SHUFFLE(2, 3, 0, 1)
    return _mm256_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1));
}
// 定义宏 npyv_rev64_s32，将其映射到 npyv_rev64_u32 宏

// 定义函数 npyv_rev64_f32，反转参数中的 32 位浮点元素顺序
NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{
    // 使用 AVX 指令集中的 _mm256_shuffle_ps 对 a 进行浮点元素反转，模式为 _MM_SHUFFLE(2, 3, 0, 1)
    return _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
}

// Permuting the elements of each 128-bit lane by immediate index for
// each element.

// 定义宏 npyv_permi128_u32，根据指定的索引对 128 位数据进行元素排列
#define npyv_permi128_u32(A, E0, E1, E2, E3) \
    // 使用 AVX2 指令集中的 _mm256_shuffle_epi32 对 A 进行元素排列，模式由参数 E0, E1, E2, E3 确定

// 定义宏 npyv_permi128_s32，将其映射到 npyv_permi128_u32 宏

// 定义宏 npyv_permi128_u64，根据指定的索引对 128 位数据进行元素排列
#define npyv_permi128_u64(A, E0, E1) \
    // 使用 AVX2 指令集中的 _mm256_shuffle_epi32 对 A 进行元素排列，模式由参数 E0, E1 确定

// 定义宏 npyv_permi128_s64，将其映射到 npyv_permi128_u64 宏

// 定义宏 npyv_permi128_f32，根据指定的索引对 128 位数据进行浮点元素排列
#define npyv_permi128_f32(A, E0, E1, E2, E3) \
    // 使用 AVX 指令集中的 _mm256_permute_ps 对 A 进行浮点元素排列，模式由参数 E0, E1, E2, E3 确定

// 定义宏 npyv_permi128_f64，根据指定的索引对 128 位数据进行双精度浮点元素排列
#define npyv_permi128_f64(A, E0, E1) \
    // 使用 AVX 指令集中的 _mm256_permute_pd 对 A 进行双精度浮点元素排列，模式由参数 E0, E1 确定

// 结束条件，结束条件编译器的处理，确保头文件内容不会被重复包含
#endif // _NPY_SIMD_AVX2_REORDER_H
```