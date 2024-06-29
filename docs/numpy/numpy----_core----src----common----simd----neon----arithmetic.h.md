# `.\numpy\numpy\_core\src\common\simd\neon\arithmetic.h`

```py
#ifndef NPY_SIMD
    // 如果未定义 NPY_SIMD，则抛出错误消息 "Not a standalone header"
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_ARITHMETIC_H
    // 如果未定义 _NPY_SIMD_NEON_ARITHMETIC_H，则开始定义该头文件
#define _NPY_SIMD_NEON_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
// 定义各种数据类型的无饱和加法操作
#define npyv_add_u8  vaddq_u8
#define npyv_add_s8  vaddq_s8
#define npyv_add_u16 vaddq_u16
#define npyv_add_s16 vaddq_s16
#define npyv_add_u32 vaddq_u32
#define npyv_add_s32 vaddq_s32
#define npyv_add_u64 vaddq_u64
#define npyv_add_s64 vaddq_s64
#define npyv_add_f32 vaddq_f32
#define npyv_add_f64 vaddq_f64

// saturated
// 定义各种数据类型的有饱和加法操作
#define npyv_adds_u8  vqaddq_u8
#define npyv_adds_s8  vqaddq_s8
#define npyv_adds_u16 vqaddq_u16
#define npyv_adds_s16 vqaddq_s16

/***************************
 * Subtraction
 ***************************/
// non-saturated
// 定义各种数据类型的无饱和减法操作
#define npyv_sub_u8  vsubq_u8
#define npyv_sub_s8  vsubq_s8
#define npyv_sub_u16 vsubq_u16
#define npyv_sub_s16 vsubq_s16
#define npyv_sub_u32 vsubq_u32
#define npyv_sub_s32 vsubq_s32
#define npyv_sub_u64 vsubq_u64
#define npyv_sub_s64 vsubq_s64
#define npyv_sub_f32 vsubq_f32
#define npyv_sub_f64 vsubq_f64

// saturated
// 定义各种数据类型的有饱和减法操作
#define npyv_subs_u8  vqsubq_u8
#define npyv_subs_s8  vqsubq_s8
#define npyv_subs_u16 vqsubq_u16
#define npyv_subs_s16 vqsubq_s16

/***************************
 * Multiplication
 ***************************/
// non-saturated
// 定义各种数据类型的无饱和乘法操作
#define npyv_mul_u8  vmulq_u8
#define npyv_mul_s8  vmulq_s8
#define npyv_mul_u16 vmulq_u16
#define npyv_mul_s16 vmulq_s16
#define npyv_mul_u32 vmulq_u32
#define npyv_mul_s32 vmulq_s32
#define npyv_mul_f32 vmulq_f32
#define npyv_mul_f64 vmulq_f64

/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
// 对每个无符号8位元素进行除法运算，除数为预先计算的值
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const uint8x8_t mulc_lo = vget_low_u8(divisor.val[0]);
    // high part of unsigned multiplication
    // 无符号乘法的高位部分
    uint16x8_t mull_lo  = vmull_u8(vget_low_u8(a), mulc_lo);
#if NPY_SIMD_F64
    uint16x8_t mull_hi  = vmull_high_u8(a, divisor.val[0]);
    // get the high unsigned bytes
    // 获取高位的无符号字节
    uint8x16_t mulhi    = vuzp2q_u8(vreinterpretq_u8_u16(mull_lo), vreinterpretq_u8_u16(mull_hi));
#else
    const uint8x8_t mulc_hi = vget_high_u8(divisor.val[0]);
    uint16x8_t mull_hi  = vmull_u8(vget_high_u8(a), mulc_hi);
    uint8x16_t mulhi    = vuzpq_u8(vreinterpretq_u8_u16(mull_lo), vreinterpretq_u8_u16(mull_hi)).val[1];
#endif
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    // 计算 a/d 的下整值，通过位移和加法
    uint8x16_t q        = vsubq_u8(a, mulhi);
               q        = vshlq_u8(q, vreinterpretq_s8_u8(divisor.val[1]));
               q        = vaddq_u8(mulhi, q);
               q        = vshlq_u8(q, vreinterpretq_s8_u8(divisor.val[2]));
    return q;
}
// divide each signed 8-bit element by a precomputed divisor (round towards zero)
// 对每个有符号8位元素进行除法运算（向零舍入）
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    const int8x8_t mulc_lo = vget_low_s8(divisor.val[0]);
    // 使用 Neon 指令 vmull_s8 对两个 int8x8_t 类型的寄存器进行有符号乘法运算，得到低位结果
    int16x8_t mull_lo  = vmull_s8(vget_low_s8(a), mulc_lo);
#if NPY_SIMD_F64
    // 如果定义了 NPY_SIMD_F64 宏，则使用64位SIMD指令进行操作

    // 对a和divisor.val[0]的高位进行有符号8位乘法，并获取高16位结果
    int16x8_t mull_hi  = vmull_high_s8(a, divisor.val[0]);

    // 交错打包两个有符号8位向量，从mull_lo和mull_hi中获取高8位
    int8x16_t mulhi    = vuzp2q_s8(vreinterpretq_s8_s16(mull_lo), vreinterpretq_s8_s16(mull_hi));
#else
    // 如果未定义 NPY_SIMD_F64 宏，则执行以下代码块

    // 获取divisor.val[0]的高8位
    const int8x8_t mulc_hi = vget_high_s8(divisor.val[0]);

    // 对a的高8位和mulc_hi进行有符号8位乘法
    int16x8_t mull_hi  = vmull_s8(vget_high_s8(a), mulc_hi);

    // 从mull_lo和mull_hi中交错打包出第二个向量
    int8x16_t mulhi    = vuzpq_s8(vreinterpretq_s8_s16(mull_lo), vreinterpretq_s8_s16(mull_hi)).val[1];
#endif

// q               = ((a + mulhi) >> sh1) - XSIGN(a)
// 计算q，将a和mulhi相加后右移sh1位，再减去a的符号位
int8x16_t q        = vshlq_s8(vaddq_s8(a, mulhi), divisor.val[1]);
          q        = vsubq_s8(q, vshrq_n_s8(a, 7));
          q        = vsubq_s8(veorq_s8(q, divisor.val[2]), divisor.val[2]);

return q;
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    // 获取divisor.val[0]的低16位
    const uint16x4_t mulc_lo = vget_low_u16(divisor.val[0]);

    // 对a和mulc_lo进行无符号16位乘法
    uint32x4_t mull_lo  = vmull_u16(vget_low_u16(a), mulc_lo);

#if NPY_SIMD_F64
    // 如果定义了 NPY_SIMD_F64 宏，则执行以下代码块

    // 对a和divisor.val[0]的高16位进行无符号16位乘法
    uint32x4_t mull_hi  = vmull_high_u16(a, divisor.val[0]);

    // 交错打包两个无符号16位向量，从mull_lo和mull_hi中获取高16位
    uint16x8_t mulhi    = vuzp2q_u16(vreinterpretq_u16_u32(mull_lo), vreinterpretq_u16_u32(mull_hi));
#else
    // 如果未定义 NPY_SIMD_F64 宏，则执行以下代码块

    // 获取divisor.val[0]的高16位
    const uint16x4_t mulc_hi = vget_high_u16(divisor.val[0]);

    // 对a的高16位和mulc_hi进行无符号16位乘法
    uint32x4_t mull_hi  = vmull_u16(vget_high_u16(a), mulc_hi);

    // 从mull_lo和mull_hi中交错打包出第二个向量
    uint16x8_t mulhi    = vuzpq_u16(vreinterpretq_u16_u32(mull_lo), vreinterpretq_u16_u32(mull_hi)).val[1];
#endif

// floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
// 计算floor(a/d)，首先计算a-mulhi的右移sh1位后加上mulhi，再左移sh2位
uint16x8_t q        = vsubq_u16(a, mulhi);
           q        = vshlq_u16(q, vreinterpretq_s16_u16(divisor.val[1]));
           q        = vaddq_u16(mulhi, q);
           q        = vshlq_u16(q, vreinterpretq_s16_u16(divisor.val[2]));

return q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    // 获取divisor.val[0]的低16位
    const int16x4_t mulc_lo = vget_low_s16(divisor.val[0]);

    // 对a和mulc_lo进行有符号16位乘法
    int32x4_t mull_lo  = vmull_s16(vget_low_s16(a), mulc_lo);

#if NPY_SIMD_F64
    // 如果定义了 NPY_SIMD_F64 宏，则执行以下代码块

    // 对a和divisor.val[0]的高16位进行有符号16位乘法
    int32x4_t mull_hi  = vmull_high_s16(a, divisor.val[0]);

    // 交错打包两个有符号16位向量，从mull_lo和mull_hi中获取高16位
    int16x8_t mulhi    = vuzp2q_s16(vreinterpretq_s16_s32(mull_lo), vreinterpretq_s16_s32(mull_hi));
#else
    // 如果未定义 NPY_SIMD_F64 宏，则执行以下代码块

    // 获取divisor.val[0]的高16位
    const int16x4_t mulc_hi = vget_high_s16(divisor.val[0]);

    // 对a的高16位和mulc_hi进行有符号16位乘法
    int32x4_t mull_hi  = vmull_s16(vget_high_s16(a), mulc_hi);

    // 从mull_lo和mull_hi中交错打包出第二个向量
    int16x8_t mulhi    = vuzpq_s16(vreinterpretq_s16_s32(mull_lo), vreinterpretq_s16_s32(mull_hi)).val[1];
#endif

// q               = ((a + mulhi) >> sh1) - XSIGN(a)
// 计算q，将a和mulhi相加后右移sh1位，再减去a的符号位
int16x8_t q        = vshlq_s16(vaddq_s16(a, mulhi), divisor.val[1]);
          q        = vsubq_s16(q, vshrq_n_s16(a, 15));
          q        = vsubq_s16(veorq_s16(q, divisor.val[2]), divisor.val[2]);

return q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    // 获取除数的低位部分
    const uint32x2_t mulc_lo = vget_low_u32(divisor.val[0]);
    // 对 a 的低位部分和 mulc_lo 进行无符号乘法，得到64位结果
    uint64x2_t mull_lo  = vmull_u32(vget_low_u32(a), mulc_lo);
#if NPY_SIMD_F64
    // 对 a 和除数的低位部分进行无符号高位乘法
    uint64x2_t mull_hi  = vmull_high_u32(a, divisor.val[0]);
    // 将乘法结果进行交错排列，得到高位无符号32位结果
    uint32x4_t mulhi    = vuzp2q_u32(vreinterpretq_u32_u64(mull_lo), vreinterpretq_u32_u64(mull_hi));
#else
    // 获取除数的高位部分
    const uint32x2_t mulc_hi = vget_high_u32(divisor.val[0]);
    // 对 a 的高位部分和 mulc_hi 进行无符号乘法，得到64位结果
    uint64x2_t mull_hi  = vmull_u32(vget_high_u32(a), mulc_hi);
    // 将乘法结果进行交错排列，得到高位无符号32位结果
    uint32x4_t mulhi    = vuzpq_u32(vreinterpretq_u32_u64(mull_lo), vreinterpretq_u32_u64(mull_hi)).val[1];
#endif
    // 计算商，使用预先计算的移位因子 divisor.val[1] 和 divisor.val[2]
    uint32x4_t q        =  vsubq_u32(a, mulhi);
               q        =  vshlq_u32(q, vreinterpretq_s32_u32(divisor.val[1]));
               q        =  vaddq_u32(mulhi, q);
               q        =  vshlq_u32(q, vreinterpretq_s32_u32(divisor.val[2]));
    return q;
}

// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    // 获取除数的低位部分
    const int32x2_t mulc_lo = vget_low_s32(divisor.val[0]);
    // 对 a 的低位部分和 mulc_lo 进行有符号乘法，得到64位结果
    int64x2_t mull_lo  = vmull_s32(vget_low_s32(a), mulc_lo);
#if NPY_SIMD_F64
    // 对 a 和除数的低位部分进行有符号高位乘法
    int64x2_t mull_hi  = vmull_high_s32(a, divisor.val[0]);
    // 将乘法结果进行交错排列，得到高位有符号32位结果
    int32x4_t mulhi    = vuzp2q_s32(vreinterpretq_s32_s64(mull_lo), vreinterpretq_s32_s64(mull_hi));
#else
    // 获取除数的高位部分
    const int32x2_t mulc_hi = vget_high_s32(divisor.val[0]);
    // 对 a 的高位部分和 mulc_hi 进行有符号乘法，得到64位结果
    int64x2_t mull_hi  = vmull_s32(vget_high_s32(a), mulc_hi);
    // 将乘法结果进行交错排列，得到高位有符号32位结果
    int32x4_t mulhi    = vuzpq_s32(vreinterpretq_s32_s64(mull_lo), vreinterpretq_s32_s64(mull_hi)).val[1];
#endif
    // 计算商，使用预先计算的移位因子 divisor.val[1] 和 divisor.val[2]
    int32x4_t q        = vshlq_s32(vaddq_s32(a, mulhi), divisor.val[1]);
              q        = vsubq_s32(q, vshrq_n_s32(a, 31));
              q        = vsubq_s32(veorq_s32(q, divisor.val[2]), divisor.val[2]);
    return q;
}

// divide each unsigned 64-bit element by a divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // 获取除数的第一个元素作为64位整数
    const uint64_t d = vgetq_lane_u64(divisor.val[0], 0);
    // 将 a 的每个64位元素除以 d，返回结果
    return npyv_set_u64(vgetq_lane_u64(a, 0) / d, vgetq_lane_u64(a, 1) / d);
}

// returns the high 64 bits of signed 64-bit multiplication
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    // 获取除数的第一个元素作为64位整数
    const int64_t d = vgetq_lane_s64(divisor.val[0], 0);
    // 将 a 的每个64位元素除以 d，返回结果
    return npyv_set_s64(vgetq_lane_s64(a, 0) / d, vgetq_lane_s64(a, 1) / d);
}
/***************************
 * Division
 ***************************/
#if NPY_SIMD_F64
    // 如果定义了 NPY_SIMD_F64，则使用内置的单精度浮点除法
    #define npyv_div_f32 vdivq_f32
#else
    // 否则定义一个函数 npyv_div_f32，实现单精度浮点除法
    NPY_FINLINE npyv_f32 npyv_div_f32(npyv_f32 a, npyv_f32 b)
    {
        // 基于 ARM 文档，参见 https://developer.arm.com/documentation/dui0204/j/CIHDIACI
        // 估算 b 的倒数
        npyv_f32 recipe = vrecpeq_f32(b);
        /**
         * 牛顿-拉弗森迭代法：
         *  x[n+1] = x[n] * (2 - d * x[n])
         * 当 x0 是应用于 d 的 VRECPE 结果时，收敛到 (1/d)。
         *
         * 注意：至少需要 3 次迭代以提高精度。
         */
        recipe = vmulq_f32(vrecpsq_f32(b, recipe), recipe);
        recipe = vmulq_f32(vrecpsq_f32(b, recipe), recipe);
        recipe = vmulq_f32(vrecpsq_f32(b, recipe), recipe);
        // 返回 a/b = a * recip(b) 的结果
        return vmulq_f32(a, recipe);
    }
#ifdef NPY_HAVE_NEON_VFPV4 // FMA
    // multiply and add, a*b + c
    NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vfmaq_f32(c, a, b); }
    // multiply and subtract, a*b - c
    NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vfmaq_f32(vnegq_f32(c), a, b); }
    // negate multiply and add, -(a*b) + c
    NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vfmsq_f32(c, a, b); }
    // negate multiply and subtract, -(a*b) - c
    NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vfmsq_f32(vnegq_f32(c), a, b); }
#else
    // multiply and add, a*b + c
    NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vmlaq_f32(c, a, b); }
    // multiply and subtract, a*b - c
    NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vmlaq_f32(vnegq_f32(c), a, b); }
    // negate multiply and add, -(a*b) + c
    NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vmlsq_f32(c, a, b); }
    // negate multiply and subtract, -(a*b) - c
    NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vmlsq_f32(vnegq_f32(c), a, b); }
#endif

// multiply, add for odd elements and subtract even elements.
// (a * b) -+ c
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{
    // Create a mask for selecting odd and even elements
    const npyv_f32 msign = npyv_set_f32(-0.0f, 0.0f, -0.0f, 0.0f);
    // XOR operation to toggle the sign bit, achieving -(c)
    // Perform fused multiply-add or fused multiply-subtract based on the mask
    return npyv_muladd_f32(a, b, npyv_xor_f32(msign, c));
}

#if NPY_SIMD_F64
    // F64 versions of fused operations for systems supporting SIMD with 64-bit floats
    NPY_FINLINE npyv_f64 npyv_muladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return vfmaq_f64(c, a, b); }
    NPY_FINLINE npyv_f64 npyv_mulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return vfmaq_f64(vnegq_f64(c), a, b); }
    NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return vfmsq_f64(c, a, b); }
    NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return vfmsq_f64(vnegq_f64(c), a, b); }
    // Multiply, add for odd elements and subtract even elements for F64
    NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        // Mask for selecting odd and even elements
        const npyv_f64 msign = npyv_set_f64(-0.0, 0.0);
        // XOR operation to toggle the sign bit, achieving -(c)
        // Perform fused multiply-add or fused multiply-subtract based on the mask
        return npyv_muladd_f64(a, b, npyv_xor_f64(msign, c));
    }
#endif // NPY_SIMD_F64

// Reduce sum across vector
#if NPY_SIMD_F64
    // SIMD operations for floating point 64-bit sums
    #define npyv_sum_u32 vaddvq_u32
    #define npyv_sum_u64 vaddvq_u64
    #define npyv_sum_f32 vaddvq_f32
    #define npyv_sum_f64 vaddvq_f64
#else
    // Summation for 64-bit unsigned integers
    NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
    {
        // Extract and sum low and high parts of the vector
        return vget_lane_u64(vadd_u64(vget_low_u64(a), vget_high_u64(a)), 0);
    }
    // Summation for 32-bit unsigned integers
    NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
    // 使用 NEON 指令，将输入向量 a 的低32位和高32位分别相加，生成新的向量 a0
    uint32x2_t a0 = vpadd_u32(vget_low_u32(a), vget_high_u32(a));

    // 使用 NEON 指令，将向量 a0 和输入向量 a 的高32位分别相加，返回相加结果的第0个元素，转换为无符号整数并返回
    return (unsigned)vget_lane_u32(vpadd_u32(a0, vget_high_u32(a)), 0);
}

// 定义一个内联函数，计算给定的浮点型 NEON 向量 a 中所有元素的总和
NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    // 使用 NEON 指令，将向量 a 中的高32位和低32位分别相加，得到结果向量 r
    float32x2_t r = vadd_f32(vget_high_f32(a), vget_low_f32(a));

    // 使用 NEON 指令，将结果向量 r 和自身相加，再取结果向量的第0个元素作为返回值，即向量中所有元素的总和
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#ifdef

// 如果定义了 NPY_SIMD_F64 宏，则使用 vaddlvq_u8 和 vaddlvq_u16 宏来进行向量内部求和操作
#if NPY_SIMD_F64
    #define npyv_sumup_u8  vaddlvq_u8     // 定义向量内部无符号8位整数求和宏
    #define npyv_sumup_u16 vaddlvq_u16    // 定义向量内部无符号16位整数求和宏
#else
    // 如果未定义 NPY_SIMD_F64 宏，则定义以下两个函数进行向量内部求和操作

    // 对无符号8位整数向量进行求和，返回一个16位无符号整数
    NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
    {
        uint32x4_t t0 = vpaddlq_u16(vpaddlq_u8(a));     // 将每个8位整数加倍扩展到16位，然后两两相加
        uint32x2_t t1 = vpadd_u32(vget_low_u32(t0), vget_high_u32(t0));  // 横向求和16位结果的低32位和高32位
        return vget_lane_u32(vpadd_u32(t1, t1), 0);      // 将低32位和高32位再次相加并返回最低32位结果
    }

    // 对无符号16位整数向量进行求和，返回一个32位无符号整数
    NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
    {
        uint32x4_t t0 = vpaddlq_u16(a);                  // 将每个16位整数向量两两相加
        uint32x2_t t1 = vpadd_u32(vget_low_u32(t0), vget_high_u32(t0));  // 横向求和16位结果的低32位和高32位
        return vget_lane_u32(vpadd_u32(t1, t1), 0);      // 将低32位和高32位再次相加并返回最低32位结果
    }
#endif

#endif // _NPY_SIMD_NEON_ARITHMETIC_H
```