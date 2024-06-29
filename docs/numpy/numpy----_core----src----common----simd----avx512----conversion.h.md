# `.\numpy\numpy\_core\src\common\simd\avx512\conversion.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_CVT_H
#define _NPY_SIMD_AVX512_CVT_H

// 如果未定义NPY_SIMD，抛出错误信息，要求不要单独使用此头文件
#ifdef NPY_HAVE_AVX512BW
    // 如果支持AVX512BW，则使用AVX512指令集中的_mm512_movm_epi8函数进行无符号8位整数向量到掩码的转换
    #define npyv_cvt_u8_b8  _mm512_movm_epi8
    // 如果支持AVX512BW，则使用AVX512指令集中的_mm512_movm_epi16函数进行无符号16位整数向量到掩码的转换
    #define npyv_cvt_u16_b16 _mm512_movm_epi16
#else
    // 否则，定义宏npv_cvt_u8_b8和npyv_cvt_u16_b16分别为输入和输出相等
    #define npyv_cvt_u8_b8(BL) BL
    #define npyv_cvt_u16_b16(BL) BL
#endif
// npyv_cvt_s8_b8和npyv_cvt_s16_b16分别为npv_cvt_u8_b8和npyv_cvt_u16_b16的别名
#define npyv_cvt_s8_b8  npyv_cvt_u8_b8
#define npyv_cvt_s16_b16 npyv_cvt_u16_b16

#ifdef NPY_HAVE_AVX512DQ
    // 如果支持AVX512DQ，则使用AVX512指令集中的_mm512_movm_epi32函数进行无符号32位整数向量到掩码的转换
    #define npyv_cvt_u32_b32 _mm512_movm_epi32
    // 如果支持AVX512DQ，则使用AVX512指令集中的_mm512_movm_epi64函数进行无符号64位整数向量到掩码的转换
    #define npyv_cvt_u64_b64 _mm512_movm_epi64
#else
    // 否则，定义宏npyv_cvt_u32_b32和npyv_cvt_u64_b64为使用BL和指定值-1作为输入
    #define npyv_cvt_u32_b32(BL) _mm512_maskz_set1_epi32(BL, (int)-1)
    #define npyv_cvt_u64_b64(BL) _mm512_maskz_set1_epi64(BL, (npy_int64)-1)
#endif
// npyv_cvt_s32_b32和npyv_cvt_s64_b64分别为npyv_cvt_u32_b32和npyv_cvt_u64_b64的别名
#define npyv_cvt_s32_b32 npyv_cvt_u32_b32
#define npyv_cvt_s64_b64 npyv_cvt_u64_b64
// 定义npv_cvt_f32_b32和npv_cvt_f64_b64分别为_mm512_castsi512_ps和npyv_cvt_u32_b32和npyv_cvt_u64_b64的别名
#define npyv_cvt_f32_b32(BL) _mm512_castsi512_ps(npyv_cvt_u32_b32(BL))
#define npyv_cvt_f64_b64(BL) _mm512_castsi512_pd(npyv_cvt_u64_b64(BL))

// 将整数向量转换为掩码
#ifdef NPY_HAVE_AVX512BW
    // 如果支持AVX512BW，则使用AVX512指令集中的_mm512_movepi8_mask和_mm512_movepi16_mask分别进行8位和16位整数向量到掩码的转换
    #define npyv_cvt_b8_u8 _mm512_movepi8_mask
    #define npyv_cvt_b16_u16 _mm512_movepi16_mask
#else
    // 否则，定义宏npyv_cvt_b8_u8和npyv_cvt_b16_u16为输入
    #define npyv_cvt_b8_u8(A)  A
    #define npyv_cvt_b16_u16(A) A
#endif
// npyv_cvt_b8_s8和npyv_cvt_b16_s16分别为npyv_cvt_b8_u8和npyv_cvt_b16_u16的别名
#define npyv_cvt_b8_s8  npyv_cvt_b8_u8
#define npyv_cvt_b16_s16 npyv_cvt_b16_u16

#ifdef NPY_HAVE_AVX512DQ
    // 如果支持AVX512DQ，则使用AVX512指令集中的_mm512_movepi32_mask和_mm512_movepi64_mask分别进行32位和64位整数向量到掩码的转换
    #define npyv_cvt_b32_u32 _mm512_movepi32_mask
    #define npyv_cvt_b64_u64 _mm512_movepi64_mask
#else
    // 否则，定义宏npyv_cvt_b32_u32和npyv_cvt_b64_u64为使用A和_mm512_setzero_si512作为输入
    #define npyv_cvt_b32_u32(A) _mm512_cmpneq_epu32_mask(A, _mm512_setzero_si512())
    #define npyv_cvt_b64_u64(A) _mm512_cmpneq_epu64_mask(A, _mm512_setzero_si512())
#endif
// npyv_cvt_b32_s32和npyv_cvt_b64_s64分别为npyv_cvt_b32_u32和npyv_cvt_b64_u64的别名
#define npyv_cvt_b32_s32 npyv_cvt_b32_u32
#define npyv_cvt_b64_s64 npyv_cvt_b64_u64
// 定义npv_cvt_b32_f32和npv_cvt_b64_f64分别为npyv_cvt_b32_u32和npyv_cvt_b64_u64的别名
#define npyv_cvt_b32_f32(A) npyv_cvt_b32_u32(_mm512_castps_si512(A))
#define npyv_cvt_b64_f64(A) npyv_cvt_b64_u64(_mm512_castpd_si512(A))

// 扩展函数
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data)
{
    npyv_u16x2 r;
    // 获取数据的低256位和高256位
    __m256i lo = npyv512_lower_si256(data);
    __m256i hi = npyv512_higher_si256(data);
#ifdef NPY_HAVE_AVX512BW
    // 如果支持AVX512BW，则将低256位和高256位的无符号8位整数转换为无符号16位整数，分别存入r的第一个和第二个元素
    r.val[0] = _mm512_cvtepu8_epi16(lo);
    r.val[1] = _mm512_cvtepu8_epi16(hi);
#else
    // 否则，将低256位和高256位的无符号8位整数先分别转换为无符号16位整数，然后组合成256位，分别存入r的第一个和第二个元素
    __m256i loelo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo));
    __m256i loehi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(lo, 1));
    __m256i hielo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi));
    __m256i hiehi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(hi, 1));
    r.val[0] = npyv512_combine_si256(loelo, loehi);
    r.val[1] = npyv512_combine_si256(hielo, hiehi);
#endif
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data)
{
    npyv_u32x2 r;
    // 获取数据的低256位和高256位
    __m256i lo = npyv512_lower_si256(data);
    __m256i hi = npyv512_higher_si256(data);
#ifdef NPY_HAVE_AVX512BW
    // 如果支持AVX512BW，则将低256位和高256位的
    # 将 AVX2 寄存器中的低位 128 位转换为整数型寄存器 __m256i
    __m256i hielo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi));
    # 将 AVX2 寄存器中的高位 128 位转换为整数型寄存器 __m256i
    __m256i hiehi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi, 1));
    # 将两个 256 位 AVX2 寄存器的低位和高位合并为一个 512 位 AVX-512 寄存器
    r.val[0] = npyv512_combine_si256(loelo, loehi);
    # 将两个 256 位 AVX2 寄存器的低位和高位合并为一个 512 位 AVX-512 寄存器
    r.val[1] = npyv512_combine_si256(hielo, hiehi);
#endif
    return r;
}

// 将两个16位布尔值打包成一个8位布尔向量
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
#ifdef NPY_HAVE_AVX512BW
    // 使用AVX-512指令集进行位解压缩和打包
    return _mm512_kunpackd((__mmask64)b, (__mmask64)a);
#else
    // 创建索引以重新排列16位整数元素
    const __m512i idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    // 使用AVX-512指令集将两个16位整数向量打包成8位整数向量
    return _mm512_permutexvar_epi64(idx, npyv512_packs_epi16(a, b));
#endif
}

// 将四个32位布尔向量打包成一个8位布尔向量
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
#ifdef NPY_HAVE_AVX512BW
    // 使用AVX-512指令集进行位解压缩和打包
    __mmask32 ab = _mm512_kunpackw((__mmask32)b, (__mmask32)a);
    __mmask32 cd = _mm512_kunpackw((__mmask32)d, (__mmask32)c);
    // 调用上一函数，将两个32位布尔向量打包成一个8位布尔向量
    return npyv_pack_b8_b16(ab, cd);
#else
    // 创建索引以重新排列32位整数元素
    const __m512i idx = _mm512_setr_epi32(
        0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    // 将四个32位整数向量打包成两个16位整数向量
    __m256i ta = npyv512_pack_lo_hi(npyv_cvt_u32_b32(a));
    __m256i tb = npyv512_pack_lo_hi(npyv_cvt_u32_b32(b));
    __m256i tc = npyv512_pack_lo_hi(npyv_cvt_u32_b32(c));
    __m256i td = npyv512_pack_lo_hi(npyv_cvt_u32_b32(d));
    // 将两个16位整数向量打包成8位整数向量
    __m256i ab = _mm256_packs_epi16(ta, tb);
    __m256i cd = _mm256_packs_epi16(tc, td);
    // 将两个8位整数向量打包成一个8位整数向量
    __m512i abcd = npyv512_combine_si256(ab, cd);
    // 使用AVX-512指令集根据索引重新排列元素
    return _mm512_permutexvar_epi32(idx, abcd);
#endif
}

// 将八个64位布尔向量打包成一个8位布尔向量
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    // 使用AVX-512指令集进行位解压缩和打包
    __mmask16 ab = _mm512_kunpackb((__mmask16)b, (__mmask16)a);
    __mmask16 cd = _mm512_kunpackb((__mmask16)d, (__mmask16)c);
    __mmask16 ef = _mm512_kunpackb((__mmask16)f, (__mmask16)e);
    __mmask16 gh = _mm512_kunpackb((__mmask16)h, (__mmask16)g);
    // 调用上一函数，将四个16位布尔向量打包成一个8位布尔向量
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}
/*
 * A compiler bug workaround on Intel Compiler Classic.
 * The bug manifests specifically when the
 * scalar result of _cvtmask64_u64 is compared against the constant -1. This
 * comparison uniquely triggers a bug under conditions of equality (==) or
 * inequality (!=) checks, which are typically used in reduction operations like
 * np.logical_or.
 *
 * The underlying issue arises from the compiler's optimizer. When the last
 * vector comparison instruction operates on zmm, the optimizer erroneously
 * emits a duplicate of this instruction but on the lower half register ymm. It
 * then performs a bitwise XOR operation between the mask produced by this
 * duplicated instruction and the mask from the original comparison instruction.
 * This erroneous behavior leads to incorrect results.
 *
 * See https://github.com/numpy/numpy/issues/26197#issuecomment-2056750975
 */
#ifdef __INTEL_COMPILER
// 使用volatile修饰符以解决Intel编译器经典版本上的编译器错误
#define NPYV__VOLATILE_CVTMASK64 volatile
#else
// 在非Intel编译器上不使用volatile修饰符
#define NPYV__VOLATILE_CVTMASK64
#endif
// 将布尔向量转换为整数位域
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a) {
#ifdef NPY_HAVE_AVX512BW_MASK
    // 将布尔向量转换为64位整数
    npy_uint64 NPYV__VOLATILE_CVTMASK64 t = (npy_uint64)_cvtmask64_u64(a);
    return t;
#elif defined(NPY_HAVE_AVX512BW)
    # 如果定义了 NPY_HAVE_AVX512BW，则执行以下代码段
    npy_uint64 NPYV__VOLATILE_CVTMASK64 t = (npy_uint64)a;
    return t;
#else
    # 否则执行以下代码段
    int mask_lo = _mm256_movemask_epi8(npyv512_lower_si256(a));
    int mask_hi = _mm256_movemask_epi8(npyv512_higher_si256(a));
    return (unsigned)mask_lo | ((npy_uint64)(unsigned)mask_hi << 32);
#endif
}
#undef NPYV__VOLATILE_CVTMASK64

NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
#ifdef NPY_HAVE_AVX512BW_MASK
    # 如果定义了 NPY_HAVE_AVX512BW_MASK，则执行以下代码段
    return (npy_uint32)_cvtmask32_u32(a);
#elif defined(NPY_HAVE_AVX512BW)
    # 如果定义了 NPY_HAVE_AVX512BW，则执行以下代码段
    return (npy_uint32)a;
#else
    # 否则执行以下代码段
    __m256i pack = _mm256_packs_epi16(
        npyv512_lower_si256(a), npyv512_higher_si256(a)
    );
    return (npy_uint32)_mm256_movemask_epi8(_mm256_permute4x64_epi64(pack, _MM_SHUFFLE(3, 1, 2, 0)));
#endif
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{ return (npy_uint16)a; }
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
#ifdef NPY_HAVE_AVX512DQ_MASK
    # 如果定义了 NPY_HAVE_AVX512DQ_MASK，则执行以下代码段
    return _cvtmask8_u32(a);
#else
    # 否则执行以下代码段
    return (npy_uint8)a;
#endif
}

// round to nearest integer (assuming even)
#define npyv_round_s32_f32 _mm512_cvtps_epi32
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    // 将 a 和 b 转换为整数（向最近的偶数舍入）
    __m256i lo = _mm512_cvtpd_epi32(a), hi = _mm512_cvtpd_epi32(b);
    // 将两个 __m256i 类型的变量合并成一个 npyv_s32 类型的变量并返回
    return npyv512_combine_si256(lo, hi);
}

#endif // _NPY_SIMD_AVX512_CVT_H
```