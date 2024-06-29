# `.\numpy\numpy\_core\src\npymath\halffloat.cpp`

```
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/*
 * 如果设置为1，转换过程中会尝试在需要时触发浮点系统的下溢、上溢和无效异常。
 */
#define NPY_HALF_GENERATE_OVERFLOW 1
#define NPY_HALF_GENERATE_INVALID 1

#include "numpy/halffloat.h"

#include "common.hpp"
/*
 ********************************************************************
 *                   HALF-PRECISION ROUTINES                        *
 ********************************************************************
 */

using namespace np;

/*
 * 将半精度浮点数转换为单精度浮点数
 */
float npy_half_to_float(npy_half h)
{
    return static_cast<float>(Half::FromBits(h));
}

/*
 * 将半精度浮点数转换为双精度浮点数
 */
double npy_half_to_double(npy_half h)
{
    return static_cast<double>(Half::FromBits(h));
}

/*
 * 将单精度浮点数转换为半精度浮点数
 */
npy_half npy_float_to_half(float f)
{
    return Half(f).Bits();
}

/*
 * 将双精度浮点数转换为半精度浮点数
 */
npy_half npy_double_to_half(double d)
{
    return Half(d).Bits();
}

/*
 * 判断半精度浮点数是否为零
 */
int npy_half_iszero(npy_half h)
{
    return (h&0x7fff) == 0;
}

/*
 * 判断半精度浮点数是否为NaN
 */
int npy_half_isnan(npy_half h)
{
    return Half::FromBits(h).IsNaN();
}

/*
 * 判断半精度浮点数是否为无穷大
 */
int npy_half_isinf(npy_half h)
{
    return ((h&0x7fffu) == 0x7c00u);
}

/*
 * 判断半精度浮点数是否为有限数
 */
int npy_half_isfinite(npy_half h)
{
    return ((h&0x7c00u) != 0x7c00u);
}

/*
 * 判断半精度浮点数的符号位是否为1（负数）
 */
int npy_half_signbit(npy_half h)
{
    return (h&0x8000u) != 0;
}

/*
 * 计算两个半精度浮点数之间的距离
 */
npy_half npy_half_spacing(npy_half h)
{
    npy_half ret;
    npy_uint16 h_exp = h&0x7c00u;
    npy_uint16 h_sig = h&0x03ffu;
    if (h_exp == 0x7c00u) {
#if NPY_HALF_GENERATE_INVALID
        npy_set_floatstatus_invalid();
#endif
        ret = NPY_HALF_NAN;
    } else if (h == 0x7bffu) {
#if NPY_HALF_GENERATE_OVERFLOW
        npy_set_floatstatus_overflow();
#endif
        ret = NPY_HALF_PINF;
    } else if ((h&0x8000u) && h_sig == 0) { /* 负数边界情况 */
        if (h_exp > 0x2c00u) { /* 如果结果是规格化的 */
            ret = h_exp - 0x2c00u;
        } else if(h_exp > 0x0400u) { /* 结果是次规格化的，但不是最小的 */
            ret = 1 << ((h_exp >> 10) - 2);
        } else {
            ret = 0x0001u; /* 最小的次规格化半精度浮点数 */
        }
    } else if (h_exp > 0x2800u) { /* 如果结果仍然是规格化的 */
        ret = h_exp - 0x2800u;
    } else if (h_exp > 0x0400u) { /* 结果是次规格化的，但不是最小的 */
        ret = 1 << ((h_exp >> 10) - 1);
    } else {
        ret = 0x0001u;
    }

    return ret;
}

/*
 * 返回一个半精度浮点数，其值与x相同，但其符号与y相同
 */
npy_half npy_half_copysign(npy_half x, npy_half y)
{
    return (x&0x7fffu) | (y&0x8000u);
}

/*
 * 返回x和y之间的下一个半精度浮点数
 */
npy_half npy_half_nextafter(npy_half x, npy_half y)
{
    npy_half ret;

    if (npy_half_isnan(x) || npy_half_isnan(y)) {
        ret = NPY_HALF_NAN;
    } else if (npy_half_eq_nonan(x, y)) {
        ret = x;
    } else if (npy_half_iszero(x)) {
        ret = (y&0x8000u) + 1; /* 最小的次规格化半精度浮点数 */
    } else if (!(x&0x8000u)) { /* x > 0 */
        if ((npy_int16)x > (npy_int16)y) { /* x > y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    } else {
        // 如果 y 的符号位为 0 或者 x 的绝对值大于 y 的绝对值，则 x < y
        if (!(y&0x8000u) || (x&0x7fffu) > (y&0x7fffu)) { /* x < y */
            // 如果 x < y，则返回 x 减去 1
            ret = x-1;
        } else {
            // 否则返回 x 加上 1
            ret = x+1;
        }
    }
#if NPY_HALF_GENERATE_OVERFLOW
    // 如果结果溢出且输入值不是无穷大，则设置浮点数状态为溢出
    if (npy_half_isinf(ret) && npy_half_isfinite(x)) {
        npy_set_floatstatus_overflow();
    }
#endif

    // 返回计算结果
    return ret;
}

// 比较两个非 NaN 的 npy_half 类型的值是否相等
int npy_half_eq_nonan(npy_half h1, npy_half h2)
{
    return Half::FromBits(h1).Equal(Half::FromBits(h2));
}

// 比较两个 npy_half 类型的值是否相等
int npy_half_eq(npy_half h1, npy_half h2)
{
    return Half::FromBits(h1) == Half::FromBits(h2);
}

// 比较两个 npy_half 类型的值是否不相等
int npy_half_ne(npy_half h1, npy_half h2)
{
    return Half::FromBits(h1) != Half::FromBits(h2);
}

// 比较两个非 NaN 的 npy_half 类型的值是否 h1 小于 h2
int npy_half_lt_nonan(npy_half h1, npy_half h2)
{
    return Half::FromBits(h1).Less(Half::FromBits(h2));
}

// 比较两个 npy_half 类型的值是否 h1 小于 h2
int npy_half_lt(npy_half h1, npy_half h2)
{
    return Half::FromBits(h1) < Half::FromBits(h2);
}

// 比较两个 npy_half 类型的值是否 h1 大于 h2
int npy_half_gt(npy_half h1, npy_half h2)
{
    // 转换为 h2 < h1 的比较形式
    return npy_half_lt(h2, h1);
}

// 比较两个非 NaN 的 npy_half 类型的值是否 h1 小于等于 h2
int npy_half_le_nonan(npy_half h1, npy_half h2)
{
    return Half::FromBits(h1).LessEqual(Half::FromBits(h2));
}

// 比较两个 npy_half 类型的值是否 h1 小于等于 h2
int npy_half_le(npy_half h1, npy_half h2)
{
    return Half::FromBits(h1) <= Half::FromBits(h2);
}

// 比较两个 npy_half 类型的值是否 h1 大于等于 h2
int npy_half_ge(npy_half h1, npy_half h2)
{
    // 转换为 h2 <= h1 的比较形式
    return npy_half_le(h2, h1);
}

// 计算 npy_half 类型的 h1 除以 h2 的商和余数
npy_half npy_half_divmod(npy_half h1, npy_half h2, npy_half *modulus)
{
    float fh1 = npy_half_to_float(h1);
    float fh2 = npy_half_to_float(h2);
    float div, mod;

    // 执行浮点数的除法和取余操作
    div = npy_divmodf(fh1, fh2, &mod);
    // 将取余结果转换为 npy_half 类型
    *modulus = npy_float_to_half(mod);
    // 将商的结果转换为 npy_half 类型并返回
    return npy_float_to_half(div);
}


/*
 ********************************************************************
 *                     BIT-LEVEL CONVERSIONS                        *
 ********************************************************************
 */

// 将 npy_uint32 类型的浮点数位表示转换为 npy_uint16 类型的半精度浮点数位表示
npy_uint16 npy_floatbits_to_halfbits(npy_uint32 f)
{
    if constexpr (Half::kNativeConversion<float>) {
        // 如果支持本地浮点数转换，则执行转换
        return BitCast<uint16_t>(Half(BitCast<float>(f)));
    }
    else {
        // 否则使用私有方法执行转换
        return half_private::FromFloatBits(f);
    }
}

// 将 npy_uint64 类型的双精度浮点数位表示转换为 npy_uint16 类型的半精度浮点数位表示
npy_uint16 npy_doublebits_to_halfbits(npy_uint64 d)
{
    if constexpr (Half::kNativeConversion<double>) {
        // 如果支持本地浮点数转换，则执行转换
        return BitCast<uint16_t>(Half(BitCast<double>(d)));
    }
    else {
        // 否则使用私有方法执行转换
        return half_private::FromDoubleBits(d);
    }
}

// 将 npy_uint16 类型的半精度浮点数位表示转换为 npy_uint32 类型的浮点数位表示
npy_uint32 npy_halfbits_to_floatbits(npy_uint16 h)
{
    if constexpr (Half::kNativeConversion<float>) {
        // 如果支持本地浮点数转换，则执行转换
        return BitCast<uint32_t>(static_cast<float>(Half::FromBits(h)));
    }
    else {
        // 否则使用私有方法执行转换
        return half_private::ToFloatBits(h);
    }
}

// 将 npy_uint16 类型的半精度浮点数位表示转换为 npy_uint64 类型的双精度浮点数位表示
npy_uint64 npy_halfbits_to_doublebits(npy_uint16 h)
{
    if constexpr (Half::kNativeConversion<double>) {
        // 如果支持本地浮点数转换，则执行转换
        return BitCast<uint64_t>(static_cast<double>(Half::FromBits(h)));
    }
    else {
        // 否则使用私有方法执行转换
        return half_private::ToDoubleBits(h);
    }
}
```