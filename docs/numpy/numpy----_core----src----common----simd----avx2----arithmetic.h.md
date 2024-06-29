# `.\numpy\numpy\_core\src\common\simd\avx2\arithmetic.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_ARITHMETIC_H
#define _NPY_SIMD_AVX2_ARITHMETIC_H

#include "../sse/utils.h"
/***************************
 * Addition
 ***************************/
// 定义 AVX2 指令集下的非饱和加法操作宏
#define npyv_add_u8  _mm256_add_epi8
#define npyv_add_s8  _mm256_add_epi8
#define npyv_add_u16 _mm256_add_epi16
#define npyv_add_s16 _mm256_add_epi16
#define npyv_add_u32 _mm256_add_epi32
#define npyv_add_s32 _mm256_add_epi32
#define npyv_add_u64 _mm256_add_epi64
#define npyv_add_s64 _mm256_add_epi64
#define npyv_add_f32 _mm256_add_ps
#define npyv_add_f64 _mm256_add_pd

// 定义 AVX2 指令集下的饱和加法操作宏
#define npyv_adds_u8  _mm256_adds_epu8
#define npyv_adds_s8  _mm256_adds_epi8
#define npyv_adds_u16 _mm256_adds_epu16
#define npyv_adds_s16 _mm256_adds_epi16
// TODO: 其他类型待实现 Packs intrins

/***************************
 * Subtraction
 ***************************/
// 定义 AVX2 指令集下的非饱和减法操作宏
#define npyv_sub_u8  _mm256_sub_epi8
#define npyv_sub_s8  _mm256_sub_epi8
#define npyv_sub_u16 _mm256_sub_epi16
#define npyv_sub_s16 _mm256_sub_epi16
#define npyv_sub_u32 _mm256_sub_epi32
#define npyv_sub_s32 _mm256_sub_epi32
#define npyv_sub_u64 _mm256_sub_epi64
#define npyv_sub_s64 _mm256_sub_epi64
#define npyv_sub_f32 _mm256_sub_ps
#define npyv_sub_f64 _mm256_sub_pd

// 定义 AVX2 指令集下的饱和减法操作宏
#define npyv_subs_u8  _mm256_subs_epu8
#define npyv_subs_s8  _mm256_subs_epi8
#define npyv_subs_u16 _mm256_subs_epu16
#define npyv_subs_s16 _mm256_subs_epi16
// TODO: 其他类型待实现 Packs intrins

/***************************
 * Multiplication
 ***************************/
// 定义 AVX2 指令集下的非饱和乘法操作宏
#define npyv_mul_u8  npyv256_mul_u8
#define npyv_mul_s8  npyv_mul_u8
#define npyv_mul_u16 _mm256_mullo_epi16
#define npyv_mul_s16 _mm256_mullo_epi16
#define npyv_mul_u32 _mm256_mullo_epi32
#define npyv_mul_s32 _mm256_mullo_epi32
#define npyv_mul_f32 _mm256_mul_ps
#define npyv_mul_f64 _mm256_mul_pd

// 饱和乘法操作宏待实现 Packs intrins

/***************************
 * Integer Division
 ***************************/
// 查看 simd/intdiv.h 以获取更多细节
// 对每个无符号 8 位元素进行预计算的除法运算
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const __m256i bmask = _mm256_set1_epi32(0x00FF00FF);
    const __m128i shf1  = _mm256_castsi256_si128(divisor.val[1]);
    const __m128i shf2  = _mm256_castsi256_si128(divisor.val[2]);
    const __m256i shf1b = _mm256_set1_epi8(0xFFU >> _mm_cvtsi128_si32(shf1));
    const __m256i shf2b = _mm256_set1_epi8(0xFFU >> _mm_cvtsi128_si32(shf2));
    // 高位无符号乘法
    __m256i mulhi_even  = _mm256_mullo_epi16(_mm256_and_si256(a, bmask), divisor.val[0]);
            mulhi_even  = _mm256_srli_epi16(mulhi_even, 8);
    __m256i mulhi_odd   = _mm256_mullo_epi16(_mm256_srli_epi16(a, 8), divisor.val[0]);
    __m256i mulhi       = _mm256_blendv_epi8(mulhi_odd, mulhi_even, bmask);
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    # 使用 AVX2 指令集中的 _mm256_sub_epi8 函数计算向量 a 减去向量 mulhi 的每个元素差，存储在向量 q 中
    __m256i q = _mm256_sub_epi8(a, mulhi);
    # 对向量 q 中的每个元素执行右移 16 位操作，然后与 shf1 按位与，结果存储回向量 q
    q = _mm256_and_si256(_mm256_srl_epi16(q, shf1), shf1b);
    # 将向量 mulhi 与向量 q 中的每个元素相加，结果存储回向量 q
    q = _mm256_add_epi8(mulhi, q);
    # 对向量 q 中的每个元素执行右移 16 位操作，然后与 shf2 按位与，结果存储回向量 q
    q = _mm256_and_si256(_mm256_srl_epi16(q, shf2), shf2b);
    # 返回计算结果的向量 q
    return q;
// 使用 SIMD 指令集进行有符号 8 位元素除法，以预先计算的除数为基准（向零舍入）
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor);
// 使用 SIMD 指令集进行有符号 8 位元素除法，以预先计算的除数为基准
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    // 创建一个掩码，用于混合选择不同的结果
    const __m256i bmask = _mm256_set1_epi32(0x00FF00FF);
    // 以特定方法处理溢出，而不是使用 _mm256_cvtepi8_epi16/_mm256_packs_epi16
    __m256i divc_even = npyv_divc_s16(_mm256_srai_epi16(_mm256_slli_epi16(a, 8), 8), divisor);
    __m256i divc_odd  = npyv_divc_s16(_mm256_srai_epi16(a, 8), divisor);
            divc_odd  = _mm256_slli_epi16(divc_odd, 8);
    // 根据掩码混合偶数和奇数位置的结果，返回最终结果
    return _mm256_blendv_epi8(divc_odd, divc_even, bmask);
}

// 使用 SIMD 指令集进行无符号 16 位元素除法，以预先计算的除数为基准
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    // 从 divisor 中提取所需的移位操作数
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    const __m128i shf2 = _mm256_castsi256_si128(divisor.val[2]);
    // 高位部分的无符号乘法
    __m256i mulhi      = _mm256_mulhi_epu16(a, divisor.val[0]);
    // 计算 floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q          = _mm256_sub_epi16(a, mulhi);
            q          = _mm256_srl_epi16(q, shf1);
            q          = _mm256_add_epi16(mulhi, q);
            q          = _mm256_srl_epi16(q, shf2);
    // 返回计算结果
    return q;
}

// 使用 SIMD 指令集进行有符号 16 位元素除法，以预先计算的除数为基准（向零舍入）
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    // 从 divisor 中提取所需的移位操作数
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    // 高位部分的有符号乘法
    __m256i mulhi      = _mm256_mulhi_epi16(a, divisor.val[0]);
    // 计算 trunc(a/d) = ((a + mulhi) >> sh1) - XSIGN(a)
    // 其中 XSIGN(a) 表示 a 的符号位扩展
    __m256i q          = _mm256_sra_epi16(_mm256_add_epi16(a, mulhi), shf1);
            q          = _mm256_sub_epi16(q, _mm256_srai_epi16(a, 15));
            q          = _mm256_sub_epi16(_mm256_xor_si256(q, divisor.val[2]), divisor.val[2]);
    // 返回计算结果
    return q;
}

// 使用 SIMD 指令集进行无符号 32 位元素除法，以预先计算的除数为基准
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    // 从 divisor 中提取所需的移位操作数
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    const __m128i shf2 = _mm256_castsi256_si128(divisor.val[2]);
    // 高位部分的无符号乘法
    __m256i mulhi_even = _mm256_srli_epi64(_mm256_mul_epu32(a, divisor.val[0]), 32);
    __m256i mulhi_odd  = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), divisor.val[0]);
    __m256i mulhi      = _mm256_blend_epi32(mulhi_even, mulhi_odd, 0xAA);
    // 计算 floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q          = _mm256_sub_epi32(a, mulhi);
            q          = _mm256_srl_epi32(q, shf1);
            q          = _mm256_add_epi32(mulhi, q);
            q          = _mm256_srl_epi32(q, shf2);
    // 返回计算结果
    return q;
}
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    // 提取除数结构体中的第二个成员作为移位掩码
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    // 计算偶数位置的乘法高位
    __m256i mulhi_even = _mm256_srli_epi64(_mm256_mul_epi32(a, divisor.val[0]), 32);
    // 计算奇数位置的乘法高位
    __m256i mulhi_odd  = _mm256_mul_epi32(_mm256_srli_epi64(a, 32), divisor.val[0]);
    // 合并偶数和奇数位置的乘法高位
    __m256i mulhi      = _mm256_blend_epi32(mulhi_even, mulhi_odd, 0xAA);
    // 计算商 q = ((a + mulhi) >> sh1) - XSIGN(a)
    // 其中 XSIGN(a) 是 a 的符号扩展
    __m256i q          = _mm256_sra_epi32(_mm256_add_epi32(a, mulhi), shf1);
            q          = _mm256_sub_epi32(q, _mm256_srai_epi32(a, 31));
            q          = _mm256_sub_epi32(_mm256_xor_si256(q, divisor.val[2]), divisor.val[2]);
    return q;
}

// 返回无符号 64 位乘法的高 64 位结果
// 参考 https://stackoverflow.com/a/28827013
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{
    // 定义一个掩码，用于提取低 32 位
    __m256i lomask = npyv_setall_s64(0xffffffff);
    // 将 a 向右移动 32 位，得到高位部分
    __m256i a_hi   = _mm256_srli_epi64(a, 32);        // a0l, a0h, a1l, a1h
    // 将 b 向右移动 32 位，得到高位部分
    __m256i b_hi   = _mm256_srli_epi64(b, 32);        // b0l, b0h, b1l, b1h
    // 计算部分乘积
    __m256i w0     = _mm256_mul_epu32(a, b);          // a0l*b0l, a1l*b1l
    __m256i w1     = _mm256_mul_epu32(a, b_hi);       // a0l*b0h, a1l*b1h
    __m256i w2     = _mm256_mul_epu32(a_hi, b);       // a0h*b0l, a1h*b0l
    __m256i w3     = _mm256_mul_epu32(a_hi, b_hi);    // a0h*b0h, a1h*b1h
    // 求和部分乘积
    __m256i w0h    = _mm256_srli_epi64(w0, 32);
    __m256i s1     = _mm256_add_epi64(w1, w0h);
    __m256i s1l    = _mm256_and_si256(s1, lomask);
    __m256i s1h    = _mm256_srli_epi64(s1, 32);

    __m256i s2     = _mm256_add_epi64(w2, s1l);
    __m256i s2h    = _mm256_srli_epi64(s2, 32);

    __m256i hi     = _mm256_add_epi64(w3, s1h);
            hi     = _mm256_add_epi64(hi, s2h);
    return hi;
}

// 按除数逐个除每个无符号 64 位元素
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // 提取除数结构体中的第二个和第三个成员作为移位掩码
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    const __m128i shf2 = _mm256_castsi256_si128(divisor.val[2]);
    // 计算无符号乘法的高位
    __m256i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q          = _mm256_sub_epi64(a, mulhi);
            q          = _mm256_srl_epi64(q, shf1);
            q          = _mm256_add_epi64(mulhi, q);
            q          = _mm256_srl_epi64(q, shf2);
    return q;
}

// 按除数逐个除每个有符号 64 位元素（向零舍入）
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    // 提取除数结构体中的第二个成员作为移位掩码
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    // 计算无符号乘法的高位
    __m256i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // 将无符号乘法结果转换为有符号数的高位乘法
    // 计算符号位向量，标识 a 和 divisor 的符号情况
    __m256i asign      = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a);
    __m256i msign      = _mm256_cmpgt_epi64(_mm256_setzero_si256(), divisor.val[0]);

    // 计算 m_asign 和 a_msign，用于减去 mulhi 中相应的值
    __m256i m_asign    = _mm256_and_si256(divisor.val[0], asign);
    __m256i a_msign    = _mm256_and_si256(a, msign);

    // 减去 mulhi 中需要的值，以计算正确的 mulhi 值
    mulhi              = _mm256_sub_epi64(mulhi, m_asign);
    mulhi              = _mm256_sub_epi64(mulhi, a_msign);

    // 计算商 q，先加上 mulhi，再进行右移操作
    __m256i q          = _mm256_add_epi64(a, mulhi);
    const __m256i sigb = npyv_setall_s64(1LL << 63);  // 创建一个符号位的掩码
    q                  = _mm256_srl_epi64(_mm256_add_epi64(q, sigb), shf1);  // 模拟算术右移操作
    q                  = _mm256_sub_epi64(q, _mm256_srl_epi64(sigb, shf1));

    // 调整 q，使其正确表示除法的结果
    q                  = _mm256_sub_epi64(q, asign);  // 减去 a 的符号位
    q                  = _mm256_sub_epi64(_mm256_xor_si256(q, divisor.val[2]), divisor.val[2]);

    // 返回计算得到的商 q
    return q;
/***************************
 * Division
 ***************************/
// 定义单精度浮点数和双精度浮点数的向量除法指令
#define npyv_div_f32 _mm256_div_ps
#define npyv_div_f64 _mm256_div_pd

/***************************
 * FUSED
 ***************************/
#ifdef NPY_HAVE_FMA3
    // 使用 FMA3 指令集的乘加操作，a*b + c
    #define npyv_muladd_f32 _mm256_fmadd_ps
    #define npyv_muladd_f64 _mm256_fmadd_pd
    // 使用 FMA3 指令集的乘减操作，a*b - c
    #define npyv_mulsub_f32 _mm256_fmsub_ps
    #define npyv_mulsub_f64 _mm256_fmsub_pd
    // 使用 FMA3 指令集的负乘加操作，-(a*b) + c
    #define npyv_nmuladd_f32 _mm256_fnmadd_ps
    #define npyv_nmuladd_f64 _mm256_fnmadd_pd
    // 使用 FMA3 指令集的负乘减操作，-(a*b) - c
    #define npyv_nmulsub_f32 _mm256_fnmsub_ps
    #define npyv_nmulsub_f64 _mm256_fnmsub_pd
    // 使用 FMA3 指令集的乘加减混合操作，(a * b) -+ c
    #define npyv_muladdsub_f32 _mm256_fmaddsub_ps
    #define npyv_muladdsub_f64 _mm256_fmaddsub_pd
#else
    // 如果没有 FMA3 指令集，定义纯粹的乘加、乘减、负乘加、负乘减、乘加减混合操作
    // multiply and add, a*b + c
    NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_add_f32(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_muladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_add_f64(npyv_mul_f64(a, b), c); }
    // multiply and subtract, a*b - c
    NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_sub_f32(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_mulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_sub_f64(npyv_mul_f64(a, b), c); }
    // negate multiply and add, -(a*b) + c
    NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_sub_f32(c, npyv_mul_f32(a, b)); }
    NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_sub_f64(c, npyv_mul_f64(a, b)); }
    // negate multiply and subtract, -(a*b) - c
    NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    {
        // 使用 XOR 指令对 a 取反
        npyv_f32 neg_a = npyv_xor_f32(a, npyv_setall_f32(-0.0f));
        return npyv_sub_f32(npyv_mul_f32(neg_a, b), c);
    }
    NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        // 使用 XOR 指令对 a 取反
        npyv_f64 neg_a = npyv_xor_f64(a, npyv_setall_f64(-0.0));
        return npyv_sub_f64(npyv_mul_f64(neg_a, b), c);
    }
    // multiply, add for odd elements and subtract even elements.
    // (a * b) -+ c
    NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return _mm256_addsub_ps(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return _mm256_addsub_pd(npyv_mul_f64(a, b), c); }

#endif // !NPY_HAVE_FMA3

/***************************
 * Summation
 ***************************/
// 对向量进行求和操作
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    // 对向量 a 进行两次横向加法操作
    __m256i s0 = _mm256_hadd_epi32(a, a);
    s0 = _mm256_hadd_epi32(s0, s0);
    // 从一个 256 位的寄存器 s0 中提取第二个 128 位子寄存器 s1
    __m128i s1 = _mm256_extracti128_si256(s0, 1);
    // 将 s0 的低 128 位子寄存器与 s1 相加，并将结果存储回 s1
    s1 = _mm_add_epi32(_mm256_castsi256_si128(s0), s1);
    // 将 s1 转换为一个整数，并返回结果
    return _mm_cvtsi128_si32(s1);
// 定义一个内联函数，计算无符号64位整数向量的总和
NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    // 创建一个256位整数向量，将a和a的逆序排列相加
    __m256i two = _mm256_add_epi64(a, _mm256_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
    // 将256位整数向量two的低128位转换为128位整数向量one，并加上其高128位
    __m128i one = _mm_add_epi64(_mm256_castsi256_si128(two), _mm256_extracti128_si256(two, 1));
    // 将128位整数向量one转换为64位无符号整数并返回
    return (npy_uint64)npyv128_cvtsi128_si64(one);
}

// 定义一个内联函数，计算单精度浮点向量的总和
NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    // 对256位单精度浮点向量a进行水平加法
    __m256 sum_halves = _mm256_hadd_ps(a, a);
    sum_halves = _mm256_hadd_ps(sum_halves, sum_halves);
    // 将256位浮点向量sum_halves拆分为两个128位向量lo和hi，并将它们相加
    __m128 lo = _mm256_castps256_ps128(sum_halves);
    __m128 hi = _mm256_extractf128_ps(sum_halves, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    // 返回128位向量sum中的单精度浮点数
    return _mm_cvtss_f32(sum);
}

// 定义一个内联函数，计算双精度浮点向量的总和
NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
    // 对256位双精度浮点向量a进行水平加法
    __m256d sum_halves = _mm256_hadd_pd(a, a);
    // 将256位双精度向量sum_halves拆分为两个128位向量lo和hi，并将它们相加
    __m128d lo = _mm256_castpd256_pd128(sum_halves);
    __m128d hi = _mm256_extractf128_pd(sum_halves, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    // 返回128位向量sum中的双精度浮点数
    return _mm_cvtsd_f64(sum);
}

// 定义一个内联函数，对无符号8位整数向量进行求和累加
// 使用256位整数向量a计算无符号16位整数的累加和
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    // 使用256位整数向量a计算无符号8位整数的累加和，存储到256位整数向量four
    __m256i four = _mm256_sad_epu8(a, _mm256_setzero_si256());
    // 将256位整数向量four拆分为两个128位整数向量，然后进行累加
    __m128i two  = _mm_add_epi16(_mm256_castsi256_si128(four), _mm256_extracti128_si256(four, 1));
    __m128i one  = _mm_add_epi16(two, _mm_unpackhi_epi64(two, two));
    // 将128位整数向量one转换为16位无符号整数并返回
    return (npy_uint16)_mm_cvtsi128_si32(one);
}

// 定义一个内联函数，对无符号16位整数向量进行求和累加
NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    // 创建一个常量掩码，用于提取偶数位
    const npyv_u16 even_mask = _mm256_set1_epi32(0x0000FFFF);
    // 使用256位整数向量a按位与掩码，得到偶数位
    __m256i even  = _mm256_and_si256(a, even_mask);
    // 将256位整数向量a右移16位，得到奇数位
    __m256i odd   = _mm256_srli_epi32(a, 16);
    // 将偶数位和奇数位相加得到累加和，并返回
    __m256i eight = _mm256_add_epi32(even, odd);
    return npyv_sum_u32(eight);
}

#endif // _NPY_SIMD_AVX2_ARITHMETIC_H
```