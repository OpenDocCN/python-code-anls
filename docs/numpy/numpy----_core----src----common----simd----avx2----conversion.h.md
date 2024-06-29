# `.\numpy\numpy\_core\src\common\simd\avx2\conversion.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_CVT_H
#define _NPY_SIMD_AVX2_CVT_H

// 将掩码类型转换为整数类型
#define npyv_cvt_u8_b8(A)   A
#define npyv_cvt_s8_b8(A)   A
#define npyv_cvt_u16_b16(A) A
#define npyv_cvt_s16_b16(A) A
#define npyv_cvt_u32_b32(A) A
#define npyv_cvt_s32_b32(A) A
#define npyv_cvt_u64_b64(A) A
#define npyv_cvt_s64_b64(A) A
#define npyv_cvt_f32_b32 _mm256_castsi256_ps
#define npyv_cvt_f64_b64 _mm256_castsi256_pd

// 将整数类型转换为掩码类型
#define npyv_cvt_b8_u8(BL)   BL
#define npyv_cvt_b8_s8(BL)   BL
#define npyv_cvt_b16_u16(BL) BL
#define npyv_cvt_b16_s16(BL) BL
#define npyv_cvt_b32_u32(BL) BL
#define npyv_cvt_b32_s32(BL) BL
#define npyv_cvt_b64_u64(BL) BL
#define npyv_cvt_b64_s64(BL) BL
#define npyv_cvt_b32_f32 _mm256_castps_si256
#define npyv_cvt_b64_f64 _mm256_castpd_si256

// 将布尔向量转换为整数位字段
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{ return (npy_uint32)_mm256_movemask_epi8(a); }

NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    // 将两个 16 位布尔向量打包成一个 8 位布尔向量
    __m128i pack = _mm_packs_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    return (npy_uint16)_mm_movemask_epi8(pack);
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{ return (npy_uint8)_mm256_movemask_ps(_mm256_castsi256_ps(a)); }
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{ return (npy_uint8)_mm256_movemask_pd(_mm256_castsi256_pd(a)); }

// 将 8 位无符号整数向量扩展为两个 16 位无符号整数向量
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data) {
    npyv_u16x2 r;
    r.val[0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(data));
    r.val[1] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 1));
    return r;
}

// 将 16 位无符号整数向量扩展为两个 32 位无符号整数向量
NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data) {
    npyv_u32x2 r;
    r.val[0] = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(data));
    r.val[1] = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(data, 1));
    return r;
}

// 将两个 16 位布尔向量打包成一个 8 位布尔向量
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    __m256i ab = _mm256_packs_epi16(a, b);
    return npyv256_shuffle_odd(ab);
}

// 将四个 32 位布尔向量打包成一个 8 位布尔向量
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    __m256i ab = _mm256_packs_epi32(a, b);
    __m256i cd = _mm256_packs_epi32(c, d);
    __m256i abcd = npyv_pack_b8_b16(ab, cd);
    return _mm256_shuffle_epi32(abcd, _MM_SHUFFLE(3, 1, 2, 0));
}

// 将八个 64 位布尔向量打包成一个 8 位布尔向量
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    __m256i ab = _mm256_packs_epi32(a, b);
    __m256i cd = _mm256_packs_epi32(c, d);
    __m256i ef = _mm256_packs_epi32(e, f);
    __m256i gh = _mm256_packs_epi32(g, h);
    __m256i abcd = _mm256_packs_epi32(ab, cd);
    __m256i efgh = _mm256_packs_epi32(ef, gh);
    # 使用 npyv256_shuffle_odd 函数对 abcd 和 efgh 进行奇数位置元素的重新排列，并将结果打包为 16 位整数
    __m256i all  = npyv256_shuffle_odd(_mm256_packs_epi16(abcd, efgh));
    
    # 使用 _mm256_alignr_epi8 函数从 all 中提取后 128 位，并将其作为新的 all，实现向右移动 8 个字节
    __m256i rev128 = _mm256_alignr_epi8(all, all, 8);
    
    # 使用 _mm256_unpacklo_epi16 函数从 all 和 rev128 中解压低位的 16 位整数并返回结果
    return _mm256_unpacklo_epi16(all, rev128);
// 结束宏定义部分

// 定义将双精度浮点数向最接近的整数四舍五入（假设四舍六入五成双）的函数，返回整型向量
#define npyv_round_s32_f32 _mm256_cvtps_epi32

// 对双精度浮点数向量进行四舍五入转换为32位有符号整数向量的函数
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    // 将 a 和 b 各自转换为128位整数向量
    __m128i lo = _mm256_cvtpd_epi32(a), hi = _mm256_cvtpd_epi32(b);
    // 将 lo 和 hi 合并成256位整数向量，lo 在低128位，hi 在高128位
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// 结束条件编译指令
#endif // _NPY_SIMD_AVX2_CVT_H
```