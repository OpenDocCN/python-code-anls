# `.\numpy\numpy\_core\src\common\simd\sse\arithmetic.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_ARITHMETIC_H
#define _NPY_SIMD_SSE_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// 非饱和加法
#define npyv_add_u8  _mm_add_epi8
#define npyv_add_s8  _mm_add_epi8
#define npyv_add_u16 _mm_add_epi16
#define npyv_add_s16 _mm_add_epi16
#define npyv_add_u32 _mm_add_epi32
#define npyv_add_s32 _mm_add_epi32
#define npyv_add_u64 _mm_add_epi64
#define npyv_add_s64 _mm_add_epi64
#define npyv_add_f32 _mm_add_ps
#define npyv_add_f64 _mm_add_pd

// 饱和加法
#define npyv_adds_u8  _mm_adds_epu8
#define npyv_adds_s8  _mm_adds_epi8
#define npyv_adds_u16 _mm_adds_epu16
#define npyv_adds_s16 _mm_adds_epi16
// TODO: 实现 Packs intrins 后继续添加

/***************************
 * Subtraction
 ***************************/
// 非饱和减法
#define npyv_sub_u8  _mm_sub_epi8
#define npyv_sub_s8  _mm_sub_epi8
#define npyv_sub_u16 _mm_sub_epi16
#define npyv_sub_s16 _mm_sub_epi16
#define npyv_sub_u32 _mm_sub_epi32
#define npyv_sub_s32 _mm_sub_epi32
#define npyv_sub_u64 _mm_sub_epi64
#define npyv_sub_s64 _mm_sub_epi64
#define npyv_sub_f32 _mm_sub_ps
#define npyv_sub_f64 _mm_sub_pd

// 饱和减法
#define npyv_subs_u8  _mm_subs_epu8
#define npyv_subs_s8  _mm_subs_epi8
#define npyv_subs_u16 _mm_subs_epu16
#define npyv_subs_s16 _mm_subs_epi16
// TODO: 实现 Packs intrins 后继续添加

/***************************
 * Multiplication
 ***************************/
// 非饱和乘法（8位无符号整数特化）
NPY_FINLINE __m128i npyv_mul_u8(__m128i a, __m128i b)
{
    // 构造掩码，用于选择乘法结果的偶数位置字节
    const __m128i mask = _mm_set1_epi32(0xFF00FF00);
    // 偶数位置乘法结果
    __m128i even = _mm_mullo_epi16(a, b);
    // 奇数位置乘法结果
    __m128i odd  = _mm_mullo_epi16(_mm_srai_epi16(a, 8), _mm_srai_epi16(b, 8));
            odd  = _mm_slli_epi16(odd, 8);
    // 选择最终结果
    return npyv_select_u8(mask, odd, even);
}
#define npyv_mul_s8  npyv_mul_u8
#define npyv_mul_u16 _mm_mullo_epi16
#define npyv_mul_s16 _mm_mullo_epi16

#ifdef NPY_HAVE_SSE41
    #define npyv_mul_u32 _mm_mullo_epi32
#else
    // 32位无符号整数乘法（未实现 SSE4.1 的情况下）
    NPY_FINLINE __m128i npyv_mul_u32(__m128i a, __m128i b)
    {
        // 偶数位置乘法结果
        __m128i even = _mm_mul_epu32(a, b);
        // 奇数位置乘法结果
        __m128i odd  = _mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32));
        // 合并低32位和高32位结果
        __m128i low  = _mm_unpacklo_epi32(even, odd);
        __m128i high = _mm_unpackhi_epi32(even, odd);
        return _mm_unpacklo_epi64(low, high);
    }
#endif // NPY_HAVE_SSE41
#define npyv_mul_s32 npyv_mul_u32
// TODO: 模拟64位整数乘法

#define npyv_mul_f32 _mm_mul_ps
#define npyv_mul_f64 _mm_mul_pd

// 饱和乘法
// TODO: 实现 Packs intrins 后继续添加

/***************************
 * Integer Division
 ***************************/
// 参见 simd/intdiv.h 以获取更多说明
// 将每个无符号8位元素除以预计算的除数
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    // 构造掩码，用于选择除法结果的低位字节
    const __m128i bmask = _mm_set1_epi32(0x00FF00FF);
    // 根据除数的第二个元素设置移位量
    const __m128i shf1b = _mm_set1_epi8(0xFFU >> _mm_cvtsi128_si32(divisor.val[1]));
    // 设置 shf2b 为一个包含 divisor.val[2] 的低 8 位的掩码
    const __m128i shf2b = _mm_set1_epi8(0xFFU >> _mm_cvtsi128_si32(divisor.val[2]));

    // 计算偶数位置的高位乘积：a 与 bmask 按位与，再与 divisor.val[0] 相乘
    __m128i mulhi_even  = _mm_mullo_epi16(_mm_and_si128(a, bmask), divisor.val[0]);

    // 计算奇数位置的高位乘积：a 右移 8 位，再与 divisor.val[0] 相乘
    __m128i mulhi_odd   = _mm_mullo_epi16(_mm_srli_epi16(a, 8), divisor.val[0]);

    // 将偶数和奇数位置的高位乘积结果右移 8 位
    mulhi_even  = _mm_srli_epi16(mulhi_even, 8);

    // 使用 bmask 选择偶数位置或奇数位置的高位乘积结果
    __m128i mulhi       = npyv_select_u8(bmask, mulhi_even, mulhi_odd);

    // 计算 floor(a/d) = (mulhi + ((a - mulhi) >> sh1)) >> sh2
    __m128i q           = _mm_sub_epi8(a, mulhi);  // a - mulhi
    q           = _mm_and_si128(_mm_srl_epi16(q, divisor.val[1]), shf1b);  // (a - mulhi) >> sh1
    q           = _mm_add_epi8(mulhi, q);  // mulhi + ((a - mulhi) >> sh1)
    q           = _mm_and_si128(_mm_srl_epi16(q, divisor.val[2]), shf2b);  // ((mulhi + ((a - mulhi) >> sh1)) >> sh2)

    // 返回计算结果 q
    return q;
// 结束 numpPy 内联函数 npyv_divc_s8，将每个有符号 8 位元素除以预先计算的除数（向零舍入）
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    // 创建一个掩码，用于选择每个 8 位元素的低位部分
    const __m128i bmask = _mm_set1_epi32(0x00FF00FF);
    // 使用 npyv_divc_s16 函数分别计算偶数和奇数位元素的除法
    __m128i divc_even = npyv_divc_s16(_mm_srai_epi16(_mm_slli_epi16(a, 8), 8), divisor);
    __m128i divc_odd  = npyv_divc_s16(_mm_srai_epi16(a, 8), divisor);
    // 将奇数位元素左移 8 位，再与偶数位元素结合，形成结果
    divc_odd  = _mm_slli_epi16(divc_odd, 8);
    return npyv_select_u8(bmask, divc_even, divc_odd);
}

// 按照预先计算的除数，将每个无符号 16 位元素进行除法
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    // 使用无符号乘法的高位部分
    __m128i mulhi = _mm_mulhi_epu16(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q     = _mm_sub_epi16(a, mulhi);
            q     = _mm_srl_epi16(q, divisor.val[1]);
            q     = _mm_add_epi16(mulhi, q);
            q     = _mm_srl_epi16(q, divisor.val[2]);
    return  q;
}

// 按照预先计算的除数，将每个有符号 16 位元素进行除法（向零舍入）
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    // 使用有符号乘法的高位部分
    __m128i mulhi = _mm_mulhi_epi16(a, divisor.val[0]);
    // q = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    __m128i q     = _mm_sra_epi16(_mm_add_epi16(a, mulhi), divisor.val[1]);
            q     = _mm_sub_epi16(q, _mm_srai_epi16(a, 15));
            q     = _mm_sub_epi16(_mm_xor_si128(q, divisor.val[2]), divisor.val[2]);
    return  q;
}

// 按照预先计算的除数，将每个无符号 32 位元素进行除法
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    // 使用无符号乘法的高位部分，分别计算偶数和奇数位元素的乘积
    __m128i mulhi_even = _mm_srli_epi64(_mm_mul_epu32(a, divisor.val[0]), 32);
    __m128i mulhi_odd  = _mm_mul_epu32(_mm_srli_epi64(a, 32), divisor.val[0]);
#ifdef NPY_HAVE_SSE41
    // 使用 SSE4.1 提供的指令混合偶数和奇数位元素的乘积
    __m128i mulhi      = _mm_blend_epi16(mulhi_even, mulhi_odd, 0xCC);
#else
    // 使用非 SSE4.1 的方式混合偶数和奇数位元素的乘积
    __m128i mask_13    = _mm_setr_epi32(0, -1, 0, -1);
           mulhi_odd   = _mm_and_si128(mulhi_odd, mask_13);
    __m128i mulhi      = _mm_or_si128(mulhi_even, mulhi_odd);
#endif
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q          = _mm_sub_epi32(a, mulhi);
            q          = _mm_srl_epi32(q, divisor.val[1]);
            q          = _mm_add_epi32(mulhi, q);
            q          = _mm_srl_epi32(q, divisor.val[2]);
    return  q;
}

// 按照预先计算的除数，将每个有符号 32 位元素进行除法（向零舍入）
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    // 取每个有符号 32 位元素的符号
    __m128i asign      = _mm_srai_epi32(a, 31);
#ifdef NPY_HAVE_SSE41
    // 使用 SSE4.1 提供的指令，计算有符号乘法的高位部分
    __m128i mulhi_even = _mm_srli_epi64(_mm_mul_epi32(a, divisor.val[0]), 32);
    # 使用 SSE2 指令集中的 `_mm_srli_epi64` 对寄存器 `a` 进行 64 位向右移位操作，然后使用 `_mm_mul_epi32` 计算结果与寄存器 `divisor.val[0]` 的乘积。
    __m128i mulhi_odd  = _mm_mul_epi32(_mm_srli_epi64(a, 32), divisor.val[0]);
    
    # 使用 SSE2 指令集中的 `_mm_blend_epi16` 将 `mulhi_even` 和 `mulhi_odd` 的结果根据掩码 `0xCC` 进行混合，生成 `mulhi` 寄存器。
    __m128i mulhi      = _mm_blend_epi16(mulhi_even, mulhi_odd, 0xCC);
#else  // not SSE4.1
    // 如果不支持 SSE4.1，执行以下代码段

    // 计算无符号乘法的高位部分
    __m128i mulhi_even = _mm_srli_epi64(_mm_mul_epu32(a, divisor.val[0]), 32);
    // 计算偶数索引位置的乘法高位
    __m128i mulhi_odd  = _mm_mul_epu32(_mm_srli_epi64(a, 32), divisor.val[0]);
    // 创建掩码以选择奇数索引位置的乘法高位结果
    __m128i mask_13    = _mm_setr_epi32(0, -1, 0, -1);
            mulhi_odd  = _mm_and_si128(mulhi_odd, mask_13);
    // 合并偶数和奇数索引位置的乘法高位结果
    __m128i mulhi      = _mm_or_si128(mulhi_even, mulhi_odd);
    // 将无符号乘法结果转换为带符号的高位乘法
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
    const __m128i msign= _mm_srai_epi32(divisor.val[0], 31);
    // 对 msign 和 a 进行按位与操作
    __m128i m_asign    = _mm_and_si128(divisor.val[0], asign);
    __m128i a_msign    = _mm_and_si128(a, msign);
            mulhi      = _mm_sub_epi32(mulhi, m_asign);
            mulhi      = _mm_sub_epi32(mulhi, a_msign);
#endif
    // 计算商 q = ((a + mulhi) >> sh1) - XSIGN(a)
    __m128i q          = _mm_sra_epi32(_mm_add_epi32(a, mulhi), divisor.val[1]);
            q          = _mm_sub_epi32(q, asign);
            q          = _mm_sub_epi32(_mm_xor_si128(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// 返回无符号 64 位乘法的高 64 位结果
// 参考：https://stackoverflow.com/a/28827013
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{
    __m128i lomask = npyv_setall_s64(0xffffffff);
    // 将 a 向右移动 32 位，得到高位部分
    __m128i a_hi   = _mm_srli_epi64(a, 32);        // a0l, a0h, a1l, a1h
    // 将 b 向右移动 32 位，得到高位部分
    __m128i b_hi   = _mm_srli_epi64(b, 32);        // b0l, b0h, b1l, b1h
    // 计算部分乘积
    __m128i w0     = _mm_mul_epu32(a, b);          // a0l*b0l, a1l*b1l
    __m128i w1     = _mm_mul_epu32(a, b_hi);       // a0l*b0h, a1l*b1h
    __m128i w2     = _mm_mul_epu32(a_hi, b);       // a0h*b0l, a1h*b0l
    __m128i w3     = _mm_mul_epu32(a_hi, b_hi);    // a0h*b0h, a1h*b1h
    // 求和部分乘积
    __m128i w0h    = _mm_srli_epi64(w0, 32);
    __m128i s1     = _mm_add_epi64(w1, w0h);
    __m128i s1l    = _mm_and_si128(s1, lomask);
    __m128i s1h    = _mm_srli_epi64(s1, 32);

    __m128i s2     = _mm_add_epi64(w2, s1l);
    __m128i s2h    = _mm_srli_epi64(s2, 32);

    __m128i hi     = _mm_add_epi64(w3, s1h);
            hi     = _mm_add_epi64(hi, s2h);
    return hi;
}
// 每个无符号 64 位元素除以预先计算的除数
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // 计算无符号乘法的高位部分
    __m128i mulhi = npyv__mullhi_u64(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q     = _mm_sub_epi64(a, mulhi);
            q     = _mm_srl_epi64(q, divisor.val[1]);
            q     = _mm_add_epi64(mulhi, q);
            q     = _mm_srl_epi64(q, divisor.val[2]);
    return  q;
}
// 每个有符号 64 位元素除以预先计算的除数（向零舍入）
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    // 计算无符号乘法的高位部分
    __m128i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // 使用函数 npyv__mullhi_u64 对 a 和 divisor.val[0] 进行无符号64位乘法，并将结果存储在 mulhi 中
    // 这一步计算得到的是 a 与 divisor.val[0] 的乘积的高64位部分
    
    // convert unsigned to signed high multiplication
    // 将无符号乘法转换为有符号乘法的高位运算
    
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
    // 如果 a 小于 0，则将 m 加到 mulhi 上；如果 m 小于 0，则将 a 加到 mulhi 上。
    // 这段代码可能用于在无符号乘法的基础上进行有符号调整，确保乘法结果在有符号整数环境下的正确性。
#ifdef NPY_HAVE_SSE42
    // 如果支持 SSE4.2，则使用比较运算寄存器来设置除数的符号位
    const __m128i msign= _mm_cmpgt_epi64(_mm_setzero_si128(), divisor.val[0]);
    // 使用比较运算寄存器来设置被除数 a 的符号位
    __m128i asign      = _mm_cmpgt_epi64(_mm_setzero_si128(), a);
#else
    // 如果不支持 SSE4.2，则通过移位和 shuffle 操作设置除数的符号位
    const __m128i msign= _mm_srai_epi32(_mm_shuffle_epi32(divisor.val[0], _MM_SHUFFLE(3, 3, 1, 1)), 31);
    // 通过移位和 shuffle 操作设置被除数 a 的符号位
    __m128i asign      = _mm_srai_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1)), 31);
#endif
    // 计算除数和被除数符号位的按位与结果
    __m128i m_asign    = _mm_and_si128(divisor.val[0], asign);
    // 计算被除数和除数符号位的按位与结果
    __m128i a_msign    = _mm_and_si128(a, msign);
    // 对 mulhi 执行符号位修正
    mulhi              = _mm_sub_epi64(mulhi, m_asign);
    mulhi              = _mm_sub_epi64(mulhi, a_msign);
    // 计算商 q，即 (a + mulhi) >> sh，其中 sh 是移位参数
    __m128i q          = _mm_add_epi64(a, mulhi);
    // 模拟算术右移操作
    const __m128i sigb = npyv_setall_s64(1LL << 63);
    q                  = _mm_srl_epi64(_mm_add_epi64(q, sigb), divisor.val[1]);
    q                  = _mm_sub_epi64(q, _mm_srl_epi64(sigb, divisor.val[1]));
    // 修正商 q，即 q = q - XSIGN(a)
    q                  = _mm_sub_epi64(q, asign);
    // 执行截断操作，即 trunc(a/d) = (q ^ dsign) - dsign
    q                  = _mm_sub_epi64(_mm_xor_si128(q, divisor.val[2]), divisor.val[2]);
    return q;
}
/***************************
 * Division
 ***************************/
// TODO: emulate integer division
#define npyv_div_f32 _mm_div_ps
#define npyv_div_f64 _mm_div_pd
/***************************
 * FUSED
 ***************************/
#ifdef NPY_HAVE_FMA3
    // multiply and add, a*b + c
    #define npyv_muladd_f32 _mm_fmadd_ps
    #define npyv_muladd_f64 _mm_fmadd_pd
    // multiply and subtract, a*b - c
    #define npyv_mulsub_f32 _mm_fmsub_ps
    #define npyv_mulsub_f64 _mm_fmsub_pd
    // negate multiply and add, -(a*b) + c
    #define npyv_nmuladd_f32 _mm_fnmadd_ps
    #define npyv_nmuladd_f64 _mm_fnmadd_pd
    // negate multiply and subtract, -(a*b) - c
    #define npyv_nmulsub_f32 _mm_fnmsub_ps
    #define npyv_nmulsub_f64 _mm_fnmsub_pd
    // multiply, add for odd elements and subtract even elements.
    // (a * b) -+ c
    #define npyv_muladdsub_f32 _mm_fmaddsub_ps
    #define npyv_muladdsub_f64 _mm_fmaddsub_pd
#elif defined(NPY_HAVE_FMA4)
    // multiply and add, a*b + c
    #define npyv_muladd_f32 _mm_macc_ps
    #define npyv_muladd_f64 _mm_macc_pd
    // multiply and subtract, a*b - c
    #define npyv_mulsub_f32 _mm_msub_ps
    #define npyv_mulsub_f64 _mm_msub_pd
    // negate multiply and add, -(a*b) + c
    #define npyv_nmuladd_f32 _mm_nmacc_ps
    #define npyv_nmuladd_f64 _mm_nmacc_pd
    // multiply, add for odd elements and subtract even elements.
    // (a * b) -+ c
    #define npyv_muladdsub_f32 _mm_maddsub_ps
    #define npyv_muladdsub_f64 _mm_maddsub_pd
#else
    // 如果不支持 FMA 指令集，则定义一些函数来模拟乘加和乘减操作
    // multiply and add, a*b + c
    NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_add_f32(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_muladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_add_f64(npyv_mul_f64(a, b), c); }
#endif
    // 定义一个内联函数，实现浮点数向量 a 和 b 的乘法，再减去向量 c 的结果
    NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_sub_f32(npyv_mul_f32(a, b), c); }
    
    // 定义一个内联函数，实现双精度浮点数向量 a 和 b 的乘法，再减去向量 c 的结果
    NPY_FINLINE npyv_f64 npyv_mulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_sub_f64(npyv_mul_f64(a, b), c); }
    
    // 定义一个内联函数，实现浮点数向量 a 和 b 的乘法的相反数，再加上向量 c 的结果
    NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_sub_f32(c, npyv_mul_f32(a, b)); }
    
    // 定义一个内联函数，实现双精度浮点数向量 a 和 b 的乘法的相反数，再加上向量 c 的结果
    NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_sub_f64(c, npyv_mul_f64(a, b)); }
    
    // 定义一个内联函数，实现浮点数向量 a 和 b 的乘法结果加上向量 c 的结果（奇数元素加，偶数元素减）
    // (a * b) -+ c
    NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    {
        // 计算 a 和 b 的乘积
        npyv_f32 m = npyv_mul_f32(a, b);
    #ifdef NPY_HAVE_SSE3
        // 如果支持 SSE3 指令集，则使用 SSE3 中的加减操作
        return _mm_addsub_ps(m, c);
    #else
        // 如果不支持 SSE3，则手动实现加减操作
        const npyv_f32 msign = npyv_set_f32(-0.0f, 0.0f, -0.0f, 0.0f);
        return npyv_add_f32(m, npyv_xor_f32(msign, c));
    #endif
    }
    
    // 定义一个内联函数，实现双精度浮点数向量 a 和 b 的乘法结果加上向量 c 的结果（奇数元素加，偶数元素减）
    NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        // 计算 a 和 b 的乘积
        npyv_f64 m = npyv_mul_f64(a, b);
    #ifdef NPY_HAVE_SSE3
        // 如果支持 SSE3 指令集，则使用 SSE3 中的加减操作
        return _mm_addsub_pd(m, c);
    #else
        // 如果不支持 SSE3，则手动实现加减操作
        const npyv_f64 msign = npyv_set_f64(-0.0, 0.0);
        return npyv_add_f64(m, npyv_xor_f64(msign, c));
    #endif
    }
#endif // NPY_HAVE_FMA3
#ifndef NPY_HAVE_FMA3 // for FMA4 and NON-FMA3
    // negate multiply and subtract, -(a*b) - c
    // 定义一个内联函数，用于实现 -(a*b) - c 的操作
    NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    {
        // 计算 a 的负数
        npyv_f32 neg_a = npyv_xor_f32(a, npyv_setall_f32(-0.0f));
        // 返回 -(a*b) - c 的结果
        return npyv_sub_f32(npyv_mul_f32(neg_a, b), c);
    }
    // 定义一个内联函数，用于实现 -(a*b) - c 的操作，针对双精度浮点数
    NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        // 计算 a 的负数
        npyv_f64 neg_a = npyv_xor_f64(a, npyv_setall_f64(-0.0));
        // 返回 -(a*b) - c 的结果
        return npyv_sub_f64(npyv_mul_f64(neg_a, b), c);
    }
#endif // !NPY_HAVE_FMA3

/***************************
 * Summation
 ***************************/
// reduce sum across vector
// 对无符号 32 位整数向量进行求和
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    // 将向量 a 按元素相加，并将结果保存在临时变量 t 中
    __m128i t = _mm_add_epi32(a, _mm_srli_si128(a, 8));
    t = _mm_add_epi32(t, _mm_srli_si128(t, 4));
    // 将临时变量 t 的最低元素转换为 unsigned int 类型并返回
    return (unsigned)_mm_cvtsi128_si32(t);
}

// 对无符号 64 位整数向量进行求和
NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    // 将向量 a 的前后两个元素按位相加，并将结果保存在临时变量 one 中
    __m128i one = _mm_add_epi64(a, _mm_unpackhi_epi64(a, a));
    // 将临时变量 one 的元素转换为 unsigned long long 类型并返回
    return (npy_uint64)npyv128_cvtsi128_si64(one);
}

// 对单精度浮点数向量进行求和
NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
#ifdef NPY_HAVE_SSE3
    // 使用 SSE3 指令集实现单精度浮点数向量的求和
    __m128 sum_halves = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(_mm_hadd_ps(sum_halves, sum_halves));
#else
    // 使用传统 SSE 指令集实现单精度浮点数向量的求和
    __m128 t1 = _mm_movehl_ps(a, a);
    __m128 t2 = _mm_add_ps(a, t1);
    __m128 t3 = _mm_shuffle_ps(t2, t2, 1);
    __m128 t4 = _mm_add_ss(t2, t3);
    return _mm_cvtss_f32(t4);
#endif
}

// 对双精度浮点数向量进行求和
NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
#ifdef NPY_HAVE_SSE3
    // 使用 SSE3 指令集实现双精度浮点数向量的求和
    return _mm_cvtsd_f64(_mm_hadd_pd(a, a));
#else
    // 使用传统 SSE 指令集实现双精度浮点数向量的求和
    return _mm_cvtsd_f64(_mm_add_pd(a, _mm_unpackhi_pd(a, a)));
#endif
}

// expand the source vector and performs sum reduce
// 对无符号 8 位整数向量进行求和
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    // 将向量 a 的所有元素累加到一个 16 位整数向量 two 中
    __m128i two = _mm_sad_epu8(a, _mm_setzero_si128());
    // 将向量 two 的前后两个元素按位相加，并将结果保存在临时变量 one 中
    __m128i one = _mm_add_epi16(two, _mm_unpackhi_epi64(two, two));
    // 将临时变量 one 的元素转换为 unsigned short 类型并返回
    return (npy_uint16)_mm_cvtsi128_si32(one);
}

// 对无符号 16 位整数向量进行求和
NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    // 创建一个掩码，用于提取 a 中的偶数位元素
    const __m128i even_mask = _mm_set1_epi32(0x0000FFFF);
    // 将 a 中的偶数位元素提取到向量 even 中
    __m128i even = _mm_and_si128(a, even_mask);
    // 将 a 中的奇数位元素向右移动 16 位，并与 even 相加得到 four
    __m128i odd  = _mm_srli_epi32(a, 16);
    __m128i four = _mm_add_epi32(even, odd);
    // 调用 npyv_sum_u32 函数对 four 进行求和，并返回结果
    return npyv_sum_u32(four);
}

#endif // _NPY_SIMD_SSE_ARITHMETIC_H
```