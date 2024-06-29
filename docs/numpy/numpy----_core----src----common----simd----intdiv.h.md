# `.\numpy\numpy\_core\src\common\simd\intdiv.h`

```
/**
 * This header implements `npyv_divisor_*` intrinsics used for computing the parameters
 * of fast integer division, while division intrinsics `npyv_divc_*` are defined in
 * {extension}/arithmetic.h.
 */
#ifndef NPY_SIMD
    #error "Not a standalone header, use simd/simd.h instead"
#endif
#ifndef _NPY_SIMD_INTDIV_H
#define _NPY_SIMD_INTDIV_H
/**
 * bit-scan reverse for non-zeros. returns the index of the highest set bit.
 * equivalent to floor(log2(a))
 */
#ifdef _MSC_VER
    #include <intrin.h> // _BitScanReverse
#endif
/**
 * Inline function to find the index of the highest set bit in a 32-bit unsigned integer.
 * Uses compiler-specific intrinsics or assembly for efficient implementation.
 */
NPY_FINLINE unsigned npyv__bitscan_revnz_u32(npy_uint32 a)
{
    assert(a > 0); // Ensure input 'a' is non-zero due to the use of __builtin_clz
#if defined(NPY_HAVE_SSE2) && defined(_MSC_VER)
    unsigned long rl;
    (void)_BitScanReverse(&rl, (unsigned long)a); // Use _BitScanReverse for MSC compiler
    r = (unsigned)rl;
#elif defined(NPY_HAVE_SSE2) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)) \
    &&  (defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64))
    __asm__("bsr %1, %0" : "=r" (r) : "r"(a)); // Use inline assembly for GCC, Clang, or Intel Compiler on x86/x86_64
#elif defined(__GNUC__) || defined(__clang__)
    r = 31 - __builtin_clz(a); // Fallback to built-in function __builtin_clz for other architectures
#else
    r = 0; // Default to 0 if no specific implementation found
    while (a >>= 1) {
        r++;
    }
#endif
    return r; // Return the index of the highest set bit
}
/**
 * Inline function to find the index of the highest set bit in a 64-bit unsigned integer.
 * Uses compiler-specific intrinsics or assembly for efficient implementation.
 */
NPY_FINLINE unsigned npyv__bitscan_revnz_u64(npy_uint64 a)
{
    assert(a > 0); // Ensure input 'a' is non-zero due to the use of __builtin_clzll
#if defined(_M_AMD64) && defined(_MSC_VER)
    unsigned long rl;
    (void)_BitScanReverse64(&rl, a); // Use _BitScanReverse64 for MSC compiler on AMD64
    return (unsigned)rl;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER))
    npy_uint64 r;
    __asm__("bsrq %1, %0" : "=r"(r) : "r"(a)); // Use inline assembly for GCC, Clang, or Intel Compiler on x86_64
    return (unsigned)r;
#elif defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(a); // Fallback to built-in function __builtin_clzll for other architectures
#else
    npy_uint64 a_hi = a >> 32;
    if (a_hi == 0) {
        return npyv__bitscan_revnz_u32((npy_uint32)a); // Handle the upper 32 bits if they are zero
    }
    return 32 + npyv__bitscan_revnz_u32((npy_uint32)a_hi); // Calculate the index of highest set bit for 64-bit 'a'
#endif
}
/**
 * Inline function to divide a 128-bit unsigned integer by a 64-bit divisor,
 * returning the quotient.
 *
 * This function ensures the divisor is greater than 1.
 */
NPY_FINLINE npy_uint64 npyv__divh128_u64(npy_uint64 high, npy_uint64 divisor)
{
    assert(divisor > 1); // Ensure divisor is greater than 1 for valid division
    npy_uint64 quotient;
#if defined(_M_X64) && defined(_MSC_VER) && _MSC_VER >= 1920 && !defined(__clang__)
    npy_uint64 remainder;
    quotient = _udiv128(high, 0, divisor, &remainder); // Use _udiv128 for MSC compiler versions >= 1920
    (void)remainder;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER))
    __asm__("divq %[d]" : "=a"(quotient) : [d] "r"(divisor), "a"(0), "d"(high)); // Use inline assembly for GCC, Clang, or Intel Compiler on x86_64
#elif defined(__SIZEOF_INT128__)
    quotient = (npy_uint64)((((__uint128_t)high) << 64) / divisor); // Use __uint128_t for 128-bit integers if available
#else
    /**
     * Minified version based on Donald Knuth’s Algorithm D (Division of nonnegative integers),
     * and Generic implementation in Hacker’s Delight.
     *
     * See https://skanthak.homepage.t-online.de/division.html
     * with respect to the license of the Hacker's Delight book
     * (https://web.archive.org/web/20190408122508/http://www.hackersdelight.org/permissions.htm)
     */
    // 计算使得除数归一化的位移量
    unsigned ldz = 63 - npyv__bitscan_revnz_u64(divisor);
    // 对除数进行归一化处理
    divisor <<= ldz;
    high    <<= ldz;
    // 将除数分解为两个32位的数字
    npy_uint32 divisor_hi  = divisor >> 32;
    npy_uint32 divisor_lo  = divisor & 0xFFFFFFFF;
    // 计算高位商数数字
    npy_uint64 quotient_hi = high / divisor_hi;
    npy_uint64 remainder   = high - divisor_hi * quotient_hi;
    npy_uint64 base32      = 1ULL << 32;
    // 使用循环计算更低位的商数数字
    while (quotient_hi >= base32 || quotient_hi*divisor_lo > base32*remainder) {
        --quotient_hi;
        remainder += divisor_hi;
        if (remainder >= base32) {
            break;
        }
    }
    // 计算被除数的数字对
    npy_uint64 dividend_pairs = base32*high - divisor*quotient_hi;
    // 计算较低零的第二个商数数字
    npy_uint32 quotient_lo = (npy_uint32)(dividend_pairs / divisor_hi);
    quotient = base32*quotient_hi + quotient_lo;
#endif
    return quotient;
}
// Initializing divisor parameters for unsigned 8-bit division
NPY_FINLINE npyv_u8x3 npyv_divisor_u8(npy_uint8 d)
{
    unsigned l, l2, sh1, sh2, m;
    switch (d) {
    case 0: // LCOV_EXCL_LINE
        // 处理可能的除零情况，对于 x86 架构，GCC 插入 `ud2` 指令以替代
        // 让硬件/CPU 陷入的行为，这会导致非法指令异常。
        // 'volatile' 应该抑制此行为，允许我们引发硬件/CPU 算术异常。
        m = sh1 = sh2 = 1 / ((npy_uint8 volatile *)&d)[0];
        break;
    case 1:
        m = 1; sh1 = sh2 = 0;
        break;
    case 2:
        m = 1; sh1 = 1; sh2 = 0;
        break;
    default:
        l   = npyv__bitscan_revnz_u32(d - 1) + 1;  // 计算 ceil(log2(d))
        l2  = (npy_uint8)(1 << l);                 // 2^l，若 l = 8 则溢出为 0
        m   = ((npy_uint16)((l2 - d) << 8)) / d + 1; // 计算乘数
        sh1 = 1;  sh2 = l - 1;                     // 计算位移量
    }
    npyv_u8x3 divisor;
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[0] = npyv_setall_u16(m);
    divisor.val[1] = npyv_set_u8(sh1);
    divisor.val[2] = npyv_set_u8(sh2);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    divisor.val[0] = npyv_setall_u8(m);
    divisor.val[1] = npyv_setall_u8(sh1);
    divisor.val[2] = npyv_setall_u8(sh2);
#elif defined(NPY_HAVE_NEON)
    divisor.val[0] = npyv_setall_u8(m);
    divisor.val[1] = npyv_reinterpret_u8_s8(npyv_setall_s8(-sh1));
    divisor.val[2] = npyv_reinterpret_u8_s8(npyv_setall_s8(-sh2));
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    return divisor;
}
// Initializing divisor parameters for signed 8-bit division
NPY_FINLINE npyv_s16x3 npyv_divisor_s16(npy_int16 d);
NPY_FINLINE npyv_s8x3 npyv_divisor_s8(npy_int8 d)
{
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    npyv_s16x3 p = npyv_divisor_s16(d);
    npyv_s8x3 r;
    r.val[0] = npyv_reinterpret_s8_s16(p.val[0]);
    r.val[1] = npyv_reinterpret_s8_s16(p.val[1]);
    r.val[2] = npyv_reinterpret_s8_s16(p.val[2]);
    return r;
#else
    int d1 = abs(d);
    int sh, m;
    if (d1 > 1) {
        sh = (int)npyv__bitscan_revnz_u32(d1-1); // 计算 ceil(log2(abs(d))) - 1
        m = (1 << (8 + sh)) / d1 + 1;            // 计算乘数
    }
    else if (d1 == 1) {
        sh = 0; m = 1;
    }
    else {
        // 对于 d == 0，引发算术异常
        sh = m = 1 / ((npy_int8 volatile *)&d)[0]; // LCOV_EXCL_LINE
    }
    npyv_s8x3 divisor;
    divisor.val[0] = npyv_setall_s8(m);
    divisor.val[2] = npyv_setall_s8(d < 0 ? -1 : 0);
    #if defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
        divisor.val[1] = npyv_setall_s8(sh);
    #elif defined(NPY_HAVE_NEON)
        divisor.val[1] = npyv_setall_s8(-sh);
    #else
        #error "please initialize the shifting operand for the new architecture"
    #endif
    return divisor;
#endif
}
// Initializing divisor parameters for unsigned 16-bit division
NPY_FINLINE npyv_u16x3 npyv_divisor_u16(npy_uint16 d)
{
    unsigned l, l2, sh1, sh2, m;
    switch (d) {
    case 0: // LCOV_EXCL_LINE
        // 若 d 等于 0，抛出算术异常
        m = sh1 = sh2 = 1 / ((npy_uint16 volatile *)&d)[0];
        break;
    case 1:
        m = 1; sh1 = sh2 = 0;
        break;
    case 2:
        m = 1; sh1 = 1; sh2 = 0;
        break;
    default:
        l   = npyv__bitscan_revnz_u32(d - 1) + 1; // 计算 ceil(log2(d))
        l2  = (npy_uint16)(1 << l);               // 2^l，若 l = 16 则溢出为 0
        m   = ((l2 - d) << 16) / d + 1;           // 计算乘数
        sh1 = 1;  sh2 = l - 1;                    // 计算移位数
    }
    npyv_u16x3 divisor;
    divisor.val[0] = npyv_setall_u16(m);
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[1] = npyv_set_u16(sh1);
    divisor.val[2] = npyv_set_u16(sh2);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    divisor.val[1] = npyv_setall_u16(sh1);
    divisor.val[2] = npyv_setall_u16(sh2);
#elif defined(NPY_HAVE_NEON)
    divisor.val[1] = npyv_reinterpret_u16_s16(npyv_setall_s16(-sh1));
    divisor.val[2] = npyv_reinterpret_u16_s16(npyv_setall_s16(-sh2));
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    return divisor;
}



// 为有符号 16 位整数除法初始化除数参数
NPY_FINLINE npyv_s16x3 npyv_divisor_s16(npy_int16 d)
{
    int d1 = abs(d);
    int sh, m;
    if (d1 > 1) {
        sh = (int)npyv__bitscan_revnz_u32(d1 - 1); // 计算 ceil(log2(abs(d))) - 1
        m = (1 << (16 + sh)) / d1 + 1;             // 计算乘数
    }
    else if (d1 == 1) {
        sh = 0; m = 1;
    }
    else {
        // 若 d 等于 0，抛出算术异常
        sh = m = 1 / ((npy_int16 volatile *)&d)[0]; // LCOV_EXCL_LINE
    }
    npyv_s16x3 divisor;
    divisor.val[0] = npyv_setall_s16(m);
    divisor.val[2] = npyv_setall_s16(d < 0 ? -1 : 0); // 设置除数的符号
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[1] = npyv_set_s16(sh);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    divisor.val[1] = npyv_setall_s16(sh);
#elif defined(NPY_HAVE_NEON)
    divisor.val[1] = npyv_setall_s16(-sh);
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    return divisor;
}



// 为无符号 32 位整数除法初始化除数参数
NPY_FINLINE npyv_u32x3 npyv_divisor_u32(npy_uint32 d)
{
    npy_uint32 l, l2, sh1, sh2, m;
    switch (d) {
    case 0: // LCOV_EXCL_LINE
        // 若 d 等于 0，抛出算术异常
        m = sh1 = sh2 = 1 / ((npy_uint32 volatile *)&d)[0]; // LCOV_EXCL_LINE
        break;
    case 1:
        m = 1; sh1 = sh2 = 0;
        break;
    case 2:
        m = 1; sh1 = 1; sh2 = 0;
        break;


这些注释解释了每个函数和代码段的作用，确保了读者能理解其背后的逻辑和功能。
    # 对于默认情况下的处理分支（switch-case结构），计算最小位数l，即ceil(log2(d))
    l   = npyv__bitscan_revnz_u32(d - 1) + 1;     // ceil(log2(d))
    # 计算2^l，注意如果l=32可能会导致溢出为0
    l2  = (npy_uint32)(1ULL << l);                // 2^l, overflow to 0 if l = 32
    # 计算乘数m，使用(l2 - d) * 2^32 / d + 1得到的结果
    m   = ((npy_uint64)(l2 - d) << 32) / d + 1;   // multiplier
    # 设置sh1为1，sh2为l - 1，这是用于后续的位移操作的位移计数
    sh1 = 1;  sh2 = l - 1;                        // shift counts
}
# 创建一个包含三个32位无符号整数的向量，所有元素初始化为m
npyv_u32x3 divisor;
divisor.val[0] = npyv_setall_u32(m);
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    // 如果支持 SSE2 或者更高版本的 SIMD 指令集，设置除数的第二和第三元素为 sh1 和 sh2
    divisor.val[1] = npyv_set_u32(sh1);
    divisor.val[2] = npyv_set_u32(sh2);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    // 如果支持 VSX2 或者 VX 指令集，设置除数的第二和第三元素为 sh1 和 sh2 的全局设定
    divisor.val[1] = npyv_setall_u32(sh1);
    divisor.val[2] = npyv_setall_u32(sh2);
#elif defined(NPY_HAVE_NEON)
    // 如果支持 NEON 指令集，设置除数的第二和第三元素为 -sh1 和 -sh2 的重新解释的无符号整数
    divisor.val[1] = npyv_reinterpret_u32_s32(npyv_setall_s32(-sh1));
    divisor.val[2] = npyv_reinterpret_u32_s32(npyv_setall_s32(-sh2));
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    // 返回初始化后的除数
    return divisor;
}
// 初始化有符号 32 位整数除法的除数参数
NPY_FINLINE npyv_s32x3 npyv_divisor_s32(npy_int32 d)
{
    npy_int32 d1 = abs(d);
    npy_int32 sh, m;
    // 处理绝对值溢出的情况
    if ((npy_uint32)d == 0x80000000U) {
        m = 0x80000001;
        sh = 30;
    }
    else if (d1 > 1) {
        // 计算 d1 的对数向上取整减去 1，作为 sh 的值
        sh = npyv__bitscan_revnz_u32(d1 - 1); // ceil(log2(abs(d))) - 1
        // 计算乘数 m
        m =  (1ULL << (32 + sh)) / d1 + 1;    // multiplier
    }
    else if (d1 == 1) {
        sh = 0; m = 1;
    }
    else {
        // 对于 d == 0，抛出算术异常
        sh = m = 1 / ((npy_int32 volatile *)&d)[0]; // LCOV_EXCL_LINE
    }
    npyv_s32x3 divisor;
    // 设置除数的第一个元素为 m
    divisor.val[0] = npyv_setall_s32(m);
    // 设置除数的第三个元素为 d 的符号
    divisor.val[2] = npyv_setall_s32(d < 0 ? -1 : 0); // sign of divisor
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    // 如果支持 SSE2 或者更高版本的 SIMD 指令集，设置除数的第二元素为 sh
    divisor.val[1] = npyv_set_s32(sh);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    // 如果支持 VSX2 或者 VX 指令集，设置除数的第二元素为 sh 的全局设定
    divisor.val[1] = npyv_setall_s32(sh);
#elif defined(NPY_HAVE_NEON)
    // 如果支持 NEON 指令集，设置除数的第二元素为 -sh 的全局设定
    divisor.val[1] = npyv_setall_s32(-sh);
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    // 返回初始化后的除数
    return divisor;
}
// 初始化无符号 64 位整数除法的除数参数
NPY_FINLINE npyv_u64x3 npyv_divisor_u64(npy_uint64 d)
{
    npyv_u64x3 divisor;
#if defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX) || defined(NPY_HAVE_NEON)
    // 如果支持 VSX2、VX 或者 NEON 指令集，设置除数的第一个元素为 d 的全局设定
    divisor.val[0] = npyv_setall_u64(d);
#else
    npy_uint64 l, l2, sh1, sh2, m;
    switch (d) {
    case 0: // LCOV_EXCL_LINE
        // 对于 d == 0，抛出算术异常
        m = sh1 = sh2 = 1 / ((npy_uint64 volatile *)&d)[0]; // LCOV_EXCL_LINE
        break;
    case 1:
        m = 1; sh1 = sh2 = 0;
        break;
    case 2:
        m = 1; sh1 = 1; sh2 = 0;
        break;
    default:
        // 计算 d 的对数向上取整加 1，作为 l
        l = npyv__bitscan_revnz_u64(d - 1) + 1;      // ceil(log2(d))
        // 计算 2^l，作为 l2
        l2 = l < 64 ? 1ULL << l : 0;                 // 2^l
        // 计算乘数 m
        m = npyv__divh128_u64(l2 - d, d) + 1;        // multiplier
        // 设置移位计数 sh1 和 sh2
        sh1 = 1;  sh2 = l - 1;                       // shift counts
    }
    // 设置除数的第一个元素为 m
    divisor.val[0] = npyv_setall_u64(m);
    #ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
        // 如果支持 SSE2 或者更高版本的 SIMD 指令集，设置除数的第二和第三元素为 sh1 和 sh2
        divisor.val[1] = npyv_set_u64(sh1);
        divisor.val[2] = npyv_set_u64(sh2);
    #else
        #error "please initialize the shifting operand for the new architecture"
    #endif
#endif
    // 返回初始化后的除数
    return divisor;
}
// 初始化有符号 64 位整数除法的除数参数
NPY_FINLINE npyv_s64x3 npyv_divisor_s64(npy_int64 d)
{
    // 定义一个名为 divisor 的变量，其类型为 npyv_s64x3
    npyv_s64x3 divisor;
#if defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX) || defined(NPY_HAVE_NEON)
    // 设置除数向量的第一个元素为 d 的值
    divisor.val[0] = npyv_setall_s64(d);
    // 如果 d 是 -1，则设置除数向量的第二个元素为全 1，否则为全 0
    divisor.val[1] = npyv_cvt_s64_b64(
        npyv_cmpeq_s64(npyv_setall_s64(-1), divisor.val[0])
    );
#else
    npy_int64 d1 = llabs(d);
    npy_int64 sh, m;
    // 处理 abs(d) 溢出的情况
    if ((npy_uint64)d == 0x8000000000000000ULL) {
        m = 0x8000000000000001LL; // 设置特定溢出情况下的修正值
        sh = 62; // 对应的位移量
    }
    else if (d1 > 1) {
        sh = npyv__bitscan_revnz_u64(d1 - 1);       // 计算 ceil(log2(abs(d))) - 1
        m  = npyv__divh128_u64(1ULL << sh, d1) + 1; // 计算乘法因子
    }
    else if (d1 == 1) {
        sh = 0; m = 1; // 处理 d = 1 的情况
    }
    else {
        // 对于 d == 0，抛出算术异常
        sh = m = 1 / ((npy_int64 volatile *)&d)[0]; // LCOV_EXCL_LINE
        // 上面的语句标记为不计算覆盖率，处理 d = 0 的特殊情况
    }
    // 设置除数向量的第一个元素为 m
    divisor.val[0] = npyv_setall_s64(m);
    // 设置除数向量的第二个元素为 d 的符号位，如果 d < 0 则为 -1，否则为 0
    divisor.val[2] = npyv_setall_s64(d < 0 ? -1 : 0);  // 符号位
    #ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[1] = npyv_set_s64(sh); // 设置除数向量的第三个元素为 sh
    #else
        #error "please initialize the shifting operand for the new architecture"
    #endif
#endif
    // 返回填充好的 divisor 结构体
    return divisor;
}

#endif // _NPY_SIMD_INTDIV_H
```