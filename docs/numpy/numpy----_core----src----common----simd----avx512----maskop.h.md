# `.\numpy\numpy\_core\src\common\simd\avx512\maskop.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header, use simd/simd.h instead"
#endif

#ifndef _NPY_SIMD_AVX512_MASKOP_H
#define _NPY_SIMD_AVX512_MASKOP_H

/**
 * Implements conditional addition and subtraction.
 * e.g. npyv_ifadd_f32(m, a, b, c) -> m ? a + b : c
 * e.g. npyv_ifsub_f32(m, a, b, c) -> m ? a - b : c
 */

// 定义 AVX512 下的条件加法和减法操作
#define NPYV_IMPL_AVX512_EMULATE_MASK_ADDSUB(SFX, BSFX)       \
    NPY_FINLINE npyv_##SFX npyv_ifadd_##SFX                   \
    (npyv_##BSFX m, npyv_##SFX a, npyv_##SFX b, npyv_##SFX c) \
    {                                                         \
        // 执行向量加法 a + b
        npyv_##SFX add = npyv_add_##SFX(a, b);                \
        // 根据掩码 m 选择返回加法结果 add 或者 c
        return npyv_select_##SFX(m, add, c);                  \
    }                                                         \
    NPY_FINLINE npyv_##SFX npyv_ifsub_##SFX                   \
    (npyv_##BSFX m, npyv_##SFX a, npyv_##SFX b, npyv_##SFX c) \
    {                                                         \
        // 执行向量减法 a - b
        npyv_##SFX sub = npyv_sub_##SFX(a, b);                \
        // 根据掩码 m 选择返回减法结果 sub 或者 c
        return npyv_select_##SFX(m, sub, c);                  \
    }

// 定义 AVX512 下的条件加法和减法操作，使用 AVX512 指令
#define NPYV_IMPL_AVX512_MASK_ADDSUB(SFX, BSFX, ZSFX)          \
    NPY_FINLINE npyv_##SFX npyv_ifadd_##SFX                    \
    (npyv_##BSFX m, npyv_##SFX a, npyv_##SFX b, npyv_##SFX c)  \
    { return _mm512_mask_add_##ZSFX(c, m, a, b); }             \
    NPY_FINLINE npyv_##SFX npyv_ifsub_##SFX                    \
    (npyv_##BSFX m, npyv_##SFX a, npyv_##SFX b, npyv_##SFX c)  \
    { return _mm512_mask_sub_##ZSFX(c, m, a, b); }

#ifdef NPY_HAVE_AVX512BW
    // 如果支持 AVX512BW，则使用 AVX512 指令实现 u8, s8, u16, s16 类型的条件加减法
    NPYV_IMPL_AVX512_MASK_ADDSUB(u8,  b8,  epi8)
    NPYV_IMPL_AVX512_MASK_ADDSUB(s8,  b8,  epi8)
    NPYV_IMPL_AVX512_MASK_ADDSUB(u16, b16, epi16)
    NPYV_IMPL_AVX512_MASK_ADDSUB(s16, b16, epi16)
#else
    // 如果不支持 AVX512BW，则使用仿真方式实现 u8, s8, u16, s16 类型的条件加减法
    NPYV_IMPL_AVX512_EMULATE_MASK_ADDSUB(u8,  b8)
    NPYV_IMPL_AVX512_EMULATE_MASK_ADDSUB(s8,  b8)
    NPYV_IMPL_AVX512_EMULATE_MASK_ADDSUB(u16, b16)
    NPYV_IMPL_AVX512_EMULATE_MASK_ADDSUB(s16, b16)
#endif

// 使用 AVX512 指令实现 u32, s32, u64, s64, f32, f64 类型的条件加减法
NPYV_IMPL_AVX512_MASK_ADDSUB(u32, b32, epi32)
NPYV_IMPL_AVX512_MASK_ADDSUB(s32, b32, epi32)
NPYV_IMPL_AVX512_MASK_ADDSUB(u64, b64, epi64)
NPYV_IMPL_AVX512_MASK_ADDSUB(s64, b64, epi64)
NPYV_IMPL_AVX512_MASK_ADDSUB(f32, b32, ps)
NPYV_IMPL_AVX512_MASK_ADDSUB(f64, b64, pd)

// 使用 AVX512 指令实现条件除法，m ? a / b : c
NPY_FINLINE npyv_f32 npyv_ifdiv_f32(npyv_b32 m, npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return _mm512_mask_div_ps(c, m, a, b); }
// 使用 AVX512 指令实现条件除法，m ? a / b : 0
NPY_FINLINE npyv_f32 npyv_ifdivz_f32(npyv_b32 m, npyv_f32 a, npyv_f32 b)
{ return _mm512_maskz_div_ps(m, a, b); }
// 使用 AVX512 指令实现条件除法，m ? a / b : c
NPY_FINLINE npyv_f64 npyv_ifdiv_f64(npyv_b32 m, npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return _mm512_mask_div_pd(c, m, a, b); }
// 使用 AVX512 指令实现条件除法，m ? a / b : 0
NPY_FINLINE npyv_f64 npyv_ifdivz_f64(npyv_b32 m, npyv_f64 a, npyv_f64 b)
{ return _mm512_maskz_div_pd(m, a, b); }

#endif // _NPY_SIMD_AVX512_MASKOP_H
```