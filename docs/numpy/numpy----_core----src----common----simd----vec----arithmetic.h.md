# `.\numpy\numpy\_core\src\common\simd\vec\arithmetic.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_ARITHMETIC_H
#define _NPY_SIMD_VEC_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// 定义不饱和加法操作宏
#define npyv_add_u8  vec_add
#define npyv_add_s8  vec_add
#define npyv_add_u16 vec_add
#define npyv_add_s16 vec_add
#define npyv_add_u32 vec_add
#define npyv_add_s32 vec_add
#define npyv_add_u64 vec_add
#define npyv_add_s64 vec_add
#if NPY_SIMD_F32
#define npyv_add_f32 vec_add
#endif
#define npyv_add_f64 vec_add

// 饱和加法的条件编译分支
#ifdef NPY_HAVE_VX
    // 定义使用 VX 扩展的饱和加法函数实现
    #define NPYV_IMPL_VX_ADDS(SFX, PSFX) \
        NPY_FINLINE npyv_##SFX npyv_adds_##SFX(npyv_##SFX a, npyv_##SFX b)\
        {                                                                 \
            return vec_pack##PSFX(                                        \
                vec_add(vec_unpackh(a), vec_unpackh(b)),                  \
                vec_add(vec_unpackl(a), vec_unpackl(b))                   \
            );                                                            \
        }

    // 各数据类型的饱和加法宏定义
    NPYV_IMPL_VX_ADDS(u8, su)
    NPYV_IMPL_VX_ADDS(s8, s)
    NPYV_IMPL_VX_ADDS(u16, su)
    NPYV_IMPL_VX_ADDS(s16, s)
#else // VSX
    // 如果没有 VX 扩展，则使用标准的饱和加法宏定义
    #define npyv_adds_u8  vec_adds
    #define npyv_adds_s8  vec_adds
    #define npyv_adds_u16 vec_adds
    #define npyv_adds_s16 vec_adds
#endif

/***************************
 * Subtraction
 ***************************/
// 定义不饱和减法操作宏
#define npyv_sub_u8  vec_sub
#define npyv_sub_s8  vec_sub
#define npyv_sub_u16 vec_sub
#define npyv_sub_s16 vec_sub
#define npyv_sub_u32 vec_sub
#define npyv_sub_s32 vec_sub
#define npyv_sub_u64 vec_sub
#define npyv_sub_s64 vec_sub
#if NPY_SIMD_F32
#define npyv_sub_f32 vec_sub
#endif
#define npyv_sub_f64 vec_sub

// 饱和减法的条件编译分支
#ifdef NPY_HAVE_VX
    // 定义使用 VX 扩展的饱和减法函数实现
    #define NPYV_IMPL_VX_SUBS(SFX, PSFX)                                  \
        NPY_FINLINE npyv_##SFX npyv_subs_##SFX(npyv_##SFX a, npyv_##SFX b)\
        {                                                                 \
            return vec_pack##PSFX(                                        \
                vec_sub(vec_unpackh(a), vec_unpackh(b)),                  \
                vec_sub(vec_unpackl(a), vec_unpackl(b))                   \
            );                                                            \
        }

    // 各数据类型的饱和减法宏定义
    NPYV_IMPL_VX_SUBS(u8, su)
    NPYV_IMPL_VX_SUBS(s8, s)
    NPYV_IMPL_VX_SUBS(u16, su)
    NPYV_IMPL_VX_SUBS(s16, s)
#else // VSX
    // 如果没有 VX 扩展，则使用标准的饱和减法宏定义
    #define npyv_subs_u8  vec_subs
    #define npyv_subs_s8  vec_subs
    #define npyv_subs_u16 vec_subs
    #define npyv_subs_s16 vec_subs
#endif

/***************************
 * Multiplication
 ***************************/
// 定义不饱和乘法操作宏
// 在 GCC 6 及以下版本中，vec_mul 仅支持某些精度和长长整型
#if defined(NPY_HAVE_VSX) && defined(__GNUC__) && __GNUC__ < 7
    // 定义宏 NPYV_IMPL_VSX_MUL，生成用于乘法操作的函数
    #define NPYV_IMPL_VSX_MUL(T_VEC, SFX, ...)              \
        // 定义内联函数 npyv_mul_##SFX，用于向量类型 T_VEC，执行乘法操作
        NPY_FINLINE T_VEC npyv_mul_##SFX(T_VEC a, T_VEC b)  \
        {                                                   \
            // 创建一个常量向量 ev_od，包含给定位置的乘法结果索引
            const npyv_u8 ev_od = {__VA_ARGS__};            \
            // 使用 vec_perm 函数对乘法的偶数和奇数位置结果进行排列
            return vec_perm(                                \
                // 使用 vec_mule 执行乘法，生成偶数位置的结果
                (T_VEC)vec_mule(a, b),                      \
                // 使用 vec_mulo 执行乘法，生成奇数位置的结果，并根据 ev_od 排列
                (T_VEC)vec_mulo(a, b), ev_od                \
            );                                              \
        }

    // 使用宏定义的函数，生成不同类型的乘法函数并调用
    NPYV_IMPL_VSX_MUL(npyv_u8,  u8,  0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30)
    NPYV_IMPL_VSX_MUL(npyv_s8,  s8,  0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30)
    NPYV_IMPL_VSX_MUL(npyv_u16, u16, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29)
    NPYV_IMPL_VSX_MUL(npyv_s16, s16, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29)

    // 宏 NPYV_IMPL_VSX_MUL_32，定义用于 32 位整数的乘法操作函数
    // 使用 asm 内联汇编执行 vmuluwm 指令，对无符号或有符号整数进行乘法操作
    #define NPYV_IMPL_VSX_MUL_32(T_VEC, SFX)                \
        NPY_FINLINE T_VEC npyv_mul_##SFX(T_VEC a, T_VEC b)  \
        {                                                   \
            T_VEC ret;                                      \
            __asm__ __volatile__(                           \
                // 使用 vmuluwm 指令，将 a 和 b 进行乘法，结果存储到 ret
                "vmuluwm %0,%1,%2" :                        \
                "=v" (ret) : "v" (a), "v" (b)               \
            );                                              \
            return ret;                                     \
        }

    // 生成不同类型的 32 位整数乘法函数并调用
    NPYV_IMPL_VSX_MUL_32(npyv_u32, u32)
    NPYV_IMPL_VSX_MUL_32(npyv_s32, s32)
#else
#define npyv_mul_u8  vec_mul
#define npyv_mul_s8  vec_mul
#define npyv_mul_u16 vec_mul
#define npyv_mul_s16 vec_mul
#define npyv_mul_u32 vec_mul
#define npyv_mul_s32 vec_mul
#endif
#if NPY_SIMD_F32
#define npyv_mul_f32 vec_mul
#endif
#define npyv_mul_f64 vec_mul


// 如果未定义 NPY_SIMD_F32，则定义以下整数类型的乘法宏为 vec_mul
// 这些宏分别用于无符号和有符号的 8、16、32 位整数乘法
#else
#define npyv_mul_u8  vec_mul
#define npyv_mul_s8  vec_mul
#define npyv_mul_u16 vec_mul
#define npyv_mul_s16 vec_mul
#define npyv_mul_u32 vec_mul
#define npyv_mul_s32 vec_mul
#endif

// 如果定义了 NPY_SIMD_F32，则定义单精度浮点数乘法宏为 vec_mul
#if NPY_SIMD_F32
#define npyv_mul_f32 vec_mul
#endif

// 定义双精度浮点数乘法宏为 vec_mul
#define npyv_mul_f64 vec_mul

/***************************
 * Integer Division
 ***************************/

// 查看 simd/intdiv.h 以获取更多详细信息
// 对每个无符号 8 位元素进行除法，除数预先计算
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
#ifdef NPY_HAVE_VX
    // 使用向量化指令集 VX 执行无符号 8 位整数乘法高位计算
    npyv_u8  mulhi    = vec_mulh(a, divisor.val[0]);
#else // VSX
    // VSX 指令集下的替代实现
    const npyv_u8 mergeo_perm = {
        1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
    };
    // 执行无符号 8 位整数乘法的高位部分
    npyv_u16 mul_even = vec_mule(a, divisor.val[0]);
    npyv_u16 mul_odd  = vec_mulo(a, divisor.val[0]);
    npyv_u8  mulhi    = (npyv_u8)vec_perm(mul_even, mul_odd, mergeo_perm);
#endif
    // 计算 floor(a/d) 的近似值
    // floor(a/d)     = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    npyv_u8 q         = vec_sub(a, mulhi);
            q         = vec_sr(q, divisor.val[1]);
            q         = vec_add(mulhi, q);
            q         = vec_sr(q, divisor.val[2]);
    return  q;
}

// 对每个有符号 8 位元素进行除法，除数预先计算
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
#ifdef NPY_HAVE_VX
    // 使用向量化指令集 VX 执行有符号 8 位整数乘法高位计算
    npyv_s8  mulhi    = vec_mulh(a, divisor.val[0]);
#else
    // VSX 指令集下的替代实现
    const npyv_u8 mergeo_perm = {
        1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
    };
    // 执行有符号 8 位整数乘法的高位部分
    npyv_s16 mul_even = vec_mule(a, divisor.val[0]);
    npyv_s16 mul_odd  = vec_mulo(a, divisor.val[0]);
    npyv_s8  mulhi    = (npyv_s8)vec_perm(mul_even, mul_odd, mergeo_perm);
#endif
    // 计算 trunc(a/d) 的近似值
    // q              = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)     = (q ^ dsign) - dsign
    npyv_s8 q         = vec_sra_s8(vec_add(a, mulhi), (npyv_u8)divisor.val[1]);
            q         = vec_sub(q, vec_sra_s8(a, npyv_setall_u8(7)));
            q         = vec_sub(vec_xor(q, divisor.val[2]), divisor.val[2]);
    return  q;
}

// 对每个无符号 16 位元素进行除法，除数预先计算
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
#ifdef NPY_HAVE_VX
    // 使用向量化指令集 VX 执行无符号 16 位整数乘法高位计算
    npyv_u16 mulhi    = vec_mulh(a, divisor.val[0]);
#else // VSX
    // VSX 指令集下的替代实现
    const npyv_u8 mergeo_perm = {
        2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
    };
    // 执行无符号 16 位整数乘法的高位部分
    npyv_u32 mul_even = vec_mule(a, divisor.val[0]);
    npyv_u32 mul_odd  = vec_mulo(a, divisor.val[0]);
    npyv_u16 mulhi    = (npyv_u16)vec_perm(mul_even, mul_odd, mergeo_perm);
#endif
    // 计算 floor(a/d) 的近似值
    // floor(a/d)     = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    npyv_u16 q        = vec_sub(a, mulhi);
             q        = vec_sr(q, divisor.val[1]);
             q        = vec_add(mulhi, q);
             q        = vec_sr(q, divisor.val[2]);
    return   q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
#ifdef NPY_HAVE_VX
    // Perform vectorized high part of signed multiplication using VX instructions
    npyv_s16 mulhi    = vec_mulh(a, divisor.val[0]);
#else // VSX
    // Permutation pattern to merge high parts from vec_mule and vec_mulo results
    const npyv_u8 mergeo_perm = {
        2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
    };
    // Compute high parts of signed multiplication separately for even and odd lanes
    npyv_s32 mul_even = vec_mule(a, divisor.val[0]);
    npyv_s32 mul_odd  = vec_mulo(a, divisor.val[0]);
    // Combine high parts using permutation to get the result for each lane
    npyv_s16 mulhi    = (npyv_s16)vec_perm(mul_even, mul_odd, mergeo_perm);
#endif
    // Calculate quotient using divisor and shift parameters
    npyv_s16 q        = vec_sra_s16(vec_add(a, mulhi), (npyv_u16)divisor.val[1]);
    // Adjust quotient by subtracting the sign of the input (XOR with all 1s) and divisor
             q        = vec_sub(q, vec_sra_s16(a, npyv_setall_u16(15)));
             q        = vec_sub(vec_xor(q, divisor.val[2]), divisor.val[2]);
    return   q;
}

// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
#if defined(NPY_HAVE_VSX4) || defined(NPY_HAVE_VX)
    // Perform vectorized high part of unsigned multiplication using VX or VSX4 instructions
    npyv_u32 mulhi    = vec_mulh(a, divisor.val[0]);
#else // VSX
    #if defined(__GNUC__) && __GNUC__ < 8
        // Doubleword integer wide multiplication using assembly for older GCC versions
        npyv_u64 mul_even, mul_odd;
        __asm__ ("vmulouw %0,%1,%2" : "=v" (mul_even) : "v" (a), "v" (divisor.val[0]));
        __asm__ ("vmuleuw %0,%1,%2" : "=v" (mul_odd)  : "v" (a), "v" (divisor.val[0]));
    #else
        // Doubleword integer wide multiplication using vec_mule and vec_mulo for GCC >= 8
        npyv_u64 mul_even = vec_mule(a, divisor.val[0]);
        npyv_u64 mul_odd  = vec_mulo(a, divisor.val[0]);
    #endif
    // Combine high parts of unsigned multiplication results into 32-bit lanes
    npyv_u32 mulhi    = vec_mergeo((npyv_u32)mul_even, (npyv_u32)mul_odd);
#endif
    // Calculate floor(x/d) using shifts and addition
    npyv_u32 q        = vec_sub(a, mulhi);
             q        = vec_sr(q, divisor.val[1]);
             q        = vec_add(mulhi, q);
             q        = vec_sr(q, divisor.val[2]);
    return   q;
}

// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
#if defined(NPY_HAVE_VSX4) || defined(NPY_HAVE_VX)
    // Perform vectorized high part of signed multiplication using VX or VSX4 instructions
    npyv_s32 mulhi    = vec_mulh(a, divisor.val[0]);
#else
    #if defined(__GNUC__) && __GNUC__ < 8
        // Doubleword integer wide multiplication using assembly for older GCC versions
        npyv_s64 mul_even, mul_odd;
        __asm__ ("vmulosw %0,%1,%2" : "=v" (mul_even) : "v" (a), "v" (divisor.val[0]));
        __asm__ ("vmulesw %0,%1,%2" : "=v" (mul_odd)  : "v" (a), "v" (divisor.val[0]));
    #else
        // Doubleword integer wide multiplication using vec_mule and vec_mulo for GCC >= 8
        npyv_s64 mul_even = vec_mule(a, divisor.val[0]);
        npyv_s64 mul_odd  = vec_mulo(a, divisor.val[0]);
    #endif
    // Combine high parts of signed multiplication results into 32-bit lanes
    npyv_s32 mulhi    = vec_mergeo((npyv_s32)mul_even, (npyv_s32)mul_odd);
#endif
    // Calculate quotient using divisor and shift parameters
    npyv_s32 q        = vec_sub(a, mulhi);
             q        = vec_sr(q, divisor.val[1]);
             q        = vec_add(mulhi, q);
             q        = vec_sr(q, divisor.val[2]);
    return   q;
}
    // 计算有符号乘法的高位结果
    npyv_s32 mulhi = vec_mergeo((npyv_s32)mul_even, (npyv_s32)mul_odd);
// q              = ((a + mulhi) >> sh1) - XSIGN(a)
// 对 a 加上 mulhi 后右移 sh1 位，再减去 XSIGN(a) 的结果赋值给 q
npyv_s32 q        = vec_sra_s32(vec_add(a, mulhi), (npyv_u32)divisor.val[1]);
         q        = vec_sub(q, vec_sra_s32(a, npyv_setall_u32(31)));
         q        = vec_sub(vec_xor(q, divisor.val[2]), divisor.val[2]);
// 返回 q
return   q;
}
// divide each unsigned 64-bit element by a precomputed divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
#if defined(NPY_HAVE_VSX4)
// 如果定义了 NPY_HAVE_VSX4，则使用 vec_div 进行向量除法操作
return vec_div(a, divisor.val[0]);
#else
// 否则，从 divisor 中提取第一个元素 d，并将 a 的每个元素除以 d 得到结果向量
const npy_uint64 d = vec_extract(divisor.val[0], 0);
return npyv_set_u64(vec_extract(a, 0) / d, vec_extract(a, 1) / d);
#endif
}
// divide each signed 64-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
// 检查是否有可能溢出的条件，如果 a 是 -2^63 并且 divisor 的标志位为真，使用 divisor 的第一个元素作为除数 d
npyv_b64 overflow = npyv_and_b64(vec_cmpeq(a, npyv_setall_s64(-1LL << 63)), (npyv_b64)divisor.val[1]);
npyv_s64 d = vec_sel(divisor.val[0], npyv_setall_s64(1), overflow);
// 返回 a 向量中每个元素除以 d 的结果向量
return vec_div(a, d);
}
/***************************
 * Division
 ***************************/
#if NPY_SIMD_F32
#define npyv_div_f32 vec_div
#endif
#define npyv_div_f64 vec_div

/***************************
 * FUSED
 ***************************/
// multiply and add, a*b + c
#define npyv_muladd_f64 vec_madd
// multiply and subtract, a*b - c
#define npyv_mulsub_f64 vec_msub
#if NPY_SIMD_F32
#define npyv_muladd_f32 vec_madd
#define npyv_mulsub_f32 vec_msub
#endif
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
// negate multiply and add, -(a*b) + c
#define npyv_nmuladd_f32 vec_nmsub // 等同于 -(a*b - c)
#define npyv_nmuladd_f64 vec_nmsub
// negate multiply and subtract, -(a*b) - c
#define npyv_nmulsub_f64 vec_nmadd
#define npyv_nmulsub_f32 vec_nmadd // 等同于 -(a*b + c)
#else
NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return vec_neg(vec_msub(a, b, c)); }
NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return vec_neg(vec_madd(a, b, c)); }
#endif
// multiply, add for odd elements and subtract even elements.
// (a * b) -+ c
#if NPY_SIMD_F32
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{
// 使用 msign 进行乘法和减法，a*b - (msign ^ c)
const npyv_f32 msign = npyv_set_f32(-0.0f, 0.0f, -0.0f, 0.0f);
return npyv_muladd_f32(a, b, npyv_xor_f32(msign, c));
}
#endif
NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{
// 使用 msign 进行乘法和减法，a*b - (msign ^ c)
const npyv_f64 msign = npyv_set_f64(-0.0, 0.0);
return npyv_muladd_f64(a, b, npyv_xor_f64(msign, c));
}
/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
#ifdef NPY_HAVE_VX
// 如果定义了 NPY_HAVE_VX，则使用 vec_sum_u128 在零向量 zero 上对 a 进行求和并提取结果的第二部分
const npyv_u64 zero = npyv_zero_u64();
return vec_extract((npyv_u64)vec_sum_u128(a, zero), 1);
#else
// 否则，使用 vec_add 对 a 进行求和，并从结果中提取第一个元素作为最终的求和结果
return vec_extract(vec_add(a, vec_mergel(a, a)), 0);
#endif
}
// 定义一个内联函数，用于计算给定的 npyv_u32 向量的总和
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
#ifdef NPY_HAVE_VX
    // 如果支持 VX 指令集，创建一个零向量 npyv_u32 zero，并使用 vec_sum_u128 计算向量 a 和 zero 的总和，然后提取第三个元素作为结果
    const npyv_u32 zero = npyv_zero_u32();
    return vec_extract((npyv_u32)vec_sum_u128(a, zero), 3);
#else
    // 如果不支持 VX 指令集，计算向量 a 的一系列操作，返回第一个元素作为结果
    const npyv_u32 rs = vec_add(a, vec_sld(a, a, 8));
    return vec_extract(vec_add(rs, vec_sld(rs, rs, 4)), 0);
#endif
}

#if NPY_SIMD_F32
// 如果支持单精度浮点数 SIMD 计算，定义一个内联函数计算 npyv_f32 向量的总和
NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    // 将向量 a 与其自身的高低部分相加得到 sum，然后返回 sum 的前两个元素相加的结果
    npyv_f32 sum = vec_add(a, npyv_combineh_f32(a, a));
    return vec_extract(sum, 0) + vec_extract(sum, 1);
    (void)sum; // 防止编译器警告
}
#endif

// 定义一个内联函数，计算 npyv_f64 向量的总和
NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
    // 直接返回向量 a 的第一个和第二个元素相加的结果
    return vec_extract(a, 0) + vec_extract(a, 1);
}

// 扩展源向量并执行求和操作
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
#ifdef NPY_HAVE_VX
    // 如果支持 VX 指令集，创建一个零向量 npyv_u8 zero，计算向量 a 和 zero 的四个元素的和，然后将结果作为 npyv_u32 传递给 npyv_sum_u32 函数，最后转换为 npy_uint16 类型返回
    const npyv_u8 zero = npyv_zero_u8();
    npyv_u32 sum4 = vec_sum4(a, zero);
    return (npy_uint16)npyv_sum_u32(sum4);
#else
    // 如果不支持 VX 指令集，使用向量 a 和 zero 的扩展值计算向量 a 的总和，最后提取第四个元素作为结果返回
    const npyv_u32 zero = npyv_zero_u32();
    npyv_u32 four = vec_sum4s(a, zero);
    npyv_s32 one  = vec_sums((npyv_s32)four, (npyv_s32)zero);
    return (npy_uint16)vec_extract(one, 3);
    (void)one; // 防止编译器警告
#endif
}

// 定义一个内联函数，计算 npyv_u16 向量的总和
NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
#ifdef NPY_HAVE_VX
    // 如果支持 VX 指令集，计算向量 a 的两个和，将结果作为 npyv_u64 传递给 npyv_sum_u64 函数，最后转换为 npy_uint32 类型返回
    npyv_u64 sum = vec_sum2(a, npyv_zero_u16());
    return (npy_uint32)npyv_sum_u64(sum);
#else // VSX
    // 如果不支持 VX 指令集，使用向量 a 的扩展值计算 a 的总和，最后提取第四个元素作为结果返回
    const npyv_s32 zero = npyv_zero_s32();
    npyv_u32x2 eight = npyv_expand_u32_u16(a);
    npyv_u32   four  = vec_add(eight.val[0], eight.val[1]);
    npyv_s32   one   = vec_sums((npyv_s32)four, zero);
    return (npy_uint32)vec_extract(one, 3);
    (void)one; // 防止编译器警告
#endif
}

#endif // _NPY_SIMD_VEC_ARITHMETIC_H
```