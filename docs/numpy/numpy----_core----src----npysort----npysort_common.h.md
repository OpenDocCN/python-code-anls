# `.\numpy\numpy\_core\src\npysort\npysort_common.h`

```py
#ifndef __NPY_SORT_COMMON_H__
#define __NPY_SORT_COMMON_H__

#include <stdlib.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>
#include "dtypemeta.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 *****************************************************************************
 **                        SWAP MACROS                                      **
 *****************************************************************************
 */

// 布尔类型的交换宏，交换两个变量的值
#define BOOL_SWAP(a,b) {npy_bool tmp = (b); (b)=(a); (a) = tmp;}
// 字节类型的交换宏，交换两个变量的值
#define BYTE_SWAP(a,b) {npy_byte tmp = (b); (b)=(a); (a) = tmp;}
// 无符号字节类型的交换宏，交换两个变量的值
#define UBYTE_SWAP(a,b) {npy_ubyte tmp = (b); (b)=(a); (a) = tmp;}
// 短整型的交换宏，交换两个变量的值
#define SHORT_SWAP(a,b) {npy_short tmp = (b); (b)=(a); (a) = tmp;}
// 无符号短整型的交换宏，交换两个变量的值
#define USHORT_SWAP(a,b) {npy_ushort tmp = (b); (b)=(a); (a) = tmp;}
// 整型的交换宏，交换两个变量的值
#define INT_SWAP(a,b) {npy_int tmp = (b); (b)=(a); (a) = tmp;}
// 无符号整型的交换宏，交换两个变量的值
#define UINT_SWAP(a,b) {npy_uint tmp = (b); (b)=(a); (a) = tmp;}
// 长整型的交换宏，交换两个变量的值
#define LONG_SWAP(a,b) {npy_long tmp = (b); (b)=(a); (a) = tmp;}
// 无符号长整型的交换宏，交换两个变量的值
#define ULONG_SWAP(a,b) {npy_ulong tmp = (b); (b)=(a); (a) = tmp;}
// 长长整型的交换宏，交换两个变量的值
#define LONGLONG_SWAP(a,b) {npy_longlong tmp = (b); (b)=(a); (a) = tmp;}
// 无符号长长整型的交换宏，交换两个变量的值
#define ULONGLONG_SWAP(a,b) {npy_ulonglong tmp = (b); (b)=(a); (a) = tmp;}
// 半精度浮点型的交换宏，交换两个变量的值
#define HALF_SWAP(a,b) {npy_half tmp = (b); (b)=(a); (a) = tmp;}
// 单精度浮点型的交换宏，交换两个变量的值
#define FLOAT_SWAP(a,b) {npy_float tmp = (b); (b)=(a); (a) = tmp;}
// 双精度浮点型的交换宏，交换两个变量的值
#define DOUBLE_SWAP(a,b) {npy_double tmp = (b); (b)=(a); (a) = tmp;}
// 长双精度浮点型的交换宏，交换两个变量的值
#define LONGDOUBLE_SWAP(a,b) {npy_longdouble tmp = (b); (b)=(a); (a) = tmp;}
// 复数单精度浮点型的交换宏，交换两个变量的值
#define CFLOAT_SWAP(a,b) {npy_cfloat tmp = (b); (b)=(a); (a) = tmp;}
// 复数双精度浮点型的交换宏，交换两个变量的值
#define CDOUBLE_SWAP(a,b) {npy_cdouble tmp = (b); (b)=(a); (a) = tmp;}
// 复数长双精度浮点型的交换宏，交换两个变量的值
#define CLONGDOUBLE_SWAP(a,b) {npy_clongdouble tmp = (b); (b)=(a); (a) = tmp;}
// 日期时间类型的交换宏，交换两个变量的值
#define DATETIME_SWAP(a,b) {npy_datetime tmp = (b); (b)=(a); (a) = tmp;}
// 时间增量类型的交换宏，交换两个变量的值
#define TIMEDELTA_SWAP(a,b) {npy_timedelta tmp = (b); (b)=(a); (a) = tmp;}

/* Need this for the argsort functions */
// 整型指针类型的交换宏，交换两个变量的值
#define INTP_SWAP(a,b) {npy_intp tmp = (b); (b)=(a); (a) = tmp;}

/*
 *****************************************************************************
 **                        COMPARISON FUNCTIONS                             **
 *****************************************************************************
 */

// 布尔类型的小于比较函数
static inline int
BOOL_LT(npy_bool a, npy_bool b)
{
    return a < b;
}

// 字节类型的小于比较函数
static inline int
BYTE_LT(npy_byte a, npy_byte b)
{
    return a < b;
}

// 无符号字节类型的小于比较函数
static inline int
UBYTE_LT(npy_ubyte a, npy_ubyte b)
{
    return a < b;
}

// 短整型的小于比较函数
static inline int
SHORT_LT(npy_short a, npy_short b)
{
    return a < b;
}

// 无符号短整型的小于比较函数
static inline int
USHORT_LT(npy_ushort a, npy_ushort b)
{
    return a < b;
}

// 整型的小于比较函数
static inline int
INT_LT(npy_int a, npy_int b)
{
    return a < b;
}

// 无符号整型的小于比较函数
static inline int
UINT_LT(npy_uint a, npy_uint b)
{
    return a < b;
}

// 长整型的小于比较函数
static inline int
LONG_LT(npy_long a, npy_long b)
{
    return a < b;
}

// 无符号长整型的小于比较函数
static inline int
ULONG_LT(npy_ulong a, npy_ulong b)
{
    return a < b;
}

// 长长整型的小于比较函数
static inline int
LONGLONG_LT(npy_longlong a, npy_longlong b)
{
    return a < b;
}

// 无符号长长整型的小于比较函数
static inline int
ULONGLONG_LT(npy_ulonglong a, npy_ulonglong b)
{
    return a < b;
}
ULONGLONG_LT(npy_ulonglong a, npy_ulonglong b)
{
    // 检查无符号长长整型数 a 是否小于 b，并返回比较结果
    return a < b;
}


static inline int
FLOAT_LT(npy_float a, npy_float b)
{
    // 检查单精度浮点数 a 是否小于 b，或者 a 是 NaN 且 b 不是 NaN，则返回真
    return a < b || (b != b && a == a);
}


static inline int
DOUBLE_LT(npy_double a, npy_double b)
{
    // 检查双精度浮点数 a 是否小于 b，或者 a 是 NaN 且 b 不是 NaN，则返回真
    return a < b || (b != b && a == a);
}


static inline int
LONGDOUBLE_LT(npy_longdouble a, npy_longdouble b)
{
    // 检查长双精度浮点数 a 是否小于 b，或者 a 是 NaN 且 b 不是 NaN，则返回真
    return a < b || (b != b && a == a);
}


static inline int
_npy_half_isnan(npy_half h)
{
    // 检查半精度浮点数 h 是否为 NaN
    return ((h&0x7c00u) == 0x7c00u) && ((h&0x03ffu) != 0x0000u);
}


static inline int
_npy_half_lt_nonan(npy_half h1, npy_half h2)
{
    if (h1&0x8000u) {
        if (h2&0x8000u) {
            // 如果 h1 和 h2 都是负数，比较它们的绝对值大小
            return (h1&0x7fffu) > (h2&0x7fffu);
        }
        else {
            // 如果 h1 是负数而 h2 是非负数，检查零值情况，否则返回假
            return (h1 != 0x8000u) || (h2 != 0x0000u);
        }
    }
    else {
        if (h2&0x8000u) {
            // 如果 h1 是非负数而 h2 是负数，返回假
            return 0;
        }
        else {
            // 如果 h1 和 h2 都是非负数，比较它们的绝对值大小
            return (h1&0x7fffu) < (h2&0x7fffu);
        }
    }
}


static inline int
HALF_LT(npy_half a, npy_half b)
{
    int ret;

    // 检查 b 是否为 NaN
    if (_npy_half_isnan(b)) {
        // 如果 b 是 NaN，则 a 必须不是 NaN 才返回真
        ret = !_npy_half_isnan(a);
    }
    else {
        // 如果 b 不是 NaN，则比较 a 和 b 的大小，但需要确保 a 不是 NaN
        ret = !_npy_half_isnan(a) && _npy_half_lt_nonan(a, b);
    }

    return ret;
}

/*
 * For inline functions SUN recommends not using a return in the then part
 * of an if statement. It's a SUN compiler thing, so assign the return value
 * to a variable instead.
 */
static inline int
CFLOAT_LT(npy_cfloat a, npy_cfloat b)
{
    int ret;

    // 比较复数浮点数 a 和 b 的实部
    if (npy_crealf(a) < npy_crealf(b)) {
        // 如果 a 的实部小于 b 的实部，则检查它们的虚部是否相等或者有 NaN
        ret = npy_cimagf(a) == npy_cimagf(a) || npy_cimagf(b) != npy_cimagf(b);
    }
    else if (npy_crealf(a) > npy_crealf(b)) {
        // 如果 a 的实部大于 b 的实部，则检查它们的虚部是否有 NaN，并且 a 的虚部等于其自身
        ret = npy_cimagf(b) != npy_cimagf(b) && npy_cimagf(a) == npy_cimagf(a);
    }
    else if (npy_crealf(a) == npy_crealf(b) || (npy_crealf(a) != npy_crealf(a) && npy_crealf(b) != npy_crealf(b))) {
        // 如果 a 和 b 的实部相等，或者它们的实部都是 NaN，则比较它们的虚部
        ret =  npy_cimagf(a) < npy_cimagf(b) || (npy_cimagf(b) != npy_cimagf(b) && npy_cimagf(a) == npy_cimagf(a));
    }
    else {
        // 否则，只有 b 的实部是 NaN 时才返回真
        ret = npy_crealf(b) != npy_crealf(b);
    }

    return ret;
}


static inline int
CDOUBLE_LT(npy_cdouble a, npy_cdouble b)
{
    int ret;

    // 比较复数双精度浮点数 a 和 b 的实部
    if (npy_creal(a) < npy_creal(b)) {
        // 如果 a 的实部小于 b 的实部，则检查它们的虚部是否相等或者有 NaN
        ret = npy_cimag(a) == npy_cimag(a) || npy_cimag(b) != npy_cimag(b);
    }
    else if (npy_creal(a) > npy_creal(b)) {
        // 如果 a 的实部大于 b 的实部，则检查它们的虚部是否有 NaN，并且 a 的虚部等于其自身
        ret = npy_cimag(b) != npy_cimag(b) && npy_cimag(a) == npy_cimag(a);
    }
    else if (npy_creal(a) == npy_creal(b) || (npy_creal(a) != npy_creal(a) && npy_creal(b) != npy_creal(b))) {
        // 如果 a 和 b 的实部相等，或者它们的实部都是 NaN，则比较它们的虚部
        ret =  npy_cimag(a) < npy_cimag(b) || (npy_cimag(b) != npy_cimag(b) && npy_cimag(a) == npy_cimag(a));
    }
    else {
        // 否则，只有 b 的实部是 NaN 时才返回真
        ret = npy_creal(b) != npy_creal(b);
    }

    return ret;
}


static inline int
CLONGDOUBLE_LT(npy_clongdouble a, npy_clongdouble b)
{
    int ret;

    // 比较复数长双精度浮点数 a 和 b 的实部
    if (npy_creall(a) < npy_creall(b)) {
        // 如果 a 的实部小于 b 的实部，则检查它们的虚部是否相等或者有 NaN
        ret = npy_cimagl(a) == npy_cimagl(a) || npy_cimagl(b) != npy_cimagl(b);
    }
    # 如果复数 a 的实部大于复数 b 的实部
    else if (npy_creall(a) > npy_creall(b)) {
        # 则返回 b 的虚部不等于自身且 a 的虚部等于自身的逻辑值
        ret = npy_cimagl(b) != npy_cimagl(b) && npy_cimagl(a) == npy_cimagl(a);
    }
    # 如果复数 a 的实部等于复数 b 的实部，或者都不是数字
    else if (npy_creall(a) == npy_creall(b) || (npy_creall(a) != npy_creall(a) && npy_creall(b) != npy_creall(b))) {
        # 则返回 a 的虚部小于 b 的虚部，或者 b 的虚部不等于自身且 a 的虚部等于自身的逻辑值
        ret =  npy_cimagl(a) < npy_cimagl(b) || (npy_cimagl(b) != npy_cimagl(b) && npy_cimagl(a) == npy_cimagl(a));
    }
    # 否则
    else {
        # 返回 b 的实部不等于自身的逻辑值
        ret = npy_creall(b) != npy_creall(b);
    }

    # 返回计算得到的结果
    return ret;
static inline void
STRING_COPY(char *s1, char const*s2, size_t len)
{
    // 使用 memcpy 函数将 s2 指向的字符串复制到 s1 指向的内存中，长度为 len
    memcpy(s1, s2, len);
}


static inline void
STRING_SWAP(char *s1, char *s2, size_t len)
{
    // 使用循环逐个交换 s1 和 s2 指向的字符串中的字符，长度为 len
    while(len--) {
        const char t = *s1;
        *s1++ = *s2;
        *s2++ = t;
    }
}


static inline int
STRING_LT(const char *s1, const char *s2, size_t len)
{
    // 将 s1 和 s2 强制转换为无符号字符指针
    const unsigned char *c1 = (const unsigned char *)s1;
    const unsigned char *c2 = (const unsigned char *)s2;
    size_t i;
    int ret = 0;

    // 逐个比较两个字符串的每个字符，直到达到 len 长度或者找到不同字符
    for (i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            // 如果发现第一个不同的字符，根据其大小确定返回值
            ret = c1[i] < c2[i];
            break;
        }
    }
    return ret;
}


static inline void
UNICODE_COPY(npy_ucs4 *s1, npy_ucs4 const *s2, size_t len)
{
    // 使用循环将 s2 指向的 unicode 字符串复制到 s1 指向的内存中，长度为 len
    while(len--) {
        *s1++ = *s2++;
    }
}


static inline void
UNICODE_SWAP(npy_ucs4 *s1, npy_ucs4 *s2, size_t len)
{
    // 使用循环逐个交换 s1 和 s2 指向的 unicode 字符串中的字符，长度为 len
    while(len--) {
        const npy_ucs4 t = *s1;
        *s1++ = *s2;
        *s2++ = t;
    }
}


static inline int
UNICODE_LT(const npy_ucs4 *s1, const npy_ucs4 *s2, size_t len)
{
    size_t i;
    int ret = 0;

    // 逐个比较两个 unicode 字符串的每个字符，直到达到 len 长度或者找到不同字符
    for (i = 0; i < len; ++i) {
        if (s1[i] != s2[i]) {
            // 如果发现第一个不同的字符，根据其大小确定返回值
            ret = s1[i] < s2[i];
            break;
        }
    }
    return ret;
}


static inline int
DATETIME_LT(npy_datetime a, npy_datetime b)
{
    // 如果 a 是特殊值 NPY_DATETIME_NAT，则认为不小于 b
    if (a == NPY_DATETIME_NAT) {
        return 0;
    }

    // 如果 b 是特殊值 NPY_DATETIME_NAT，则认为 a 比 b 小
    if (b == NPY_DATETIME_NAT) {
        return 1;
    }

    // 否则直接比较 a 和 b 的大小
    return a < b;
}


static inline int
TIMEDELTA_LT(npy_timedelta a, npy_timedelta b)
{
    // 如果 a 是特殊值 NPY_DATETIME_NAT，则认为不小于 b
    if (a == NPY_DATETIME_NAT) {
        return 0;
    }

    // 如果 b 是特殊值 NPY_DATETIME_NAT，则认为 a 比 b 小
    if (b == NPY_DATETIME_NAT) {
        return 1;
    }

    // 否则直接比较 a 和 b 的大小
    return a < b;
}


static inline void
GENERIC_COPY(char *a, char *b, size_t len)
{
    // 使用 memcpy 函数将 b 指向的数据复制到 a 指向的内存中，长度为 len
    memcpy(a, b, len);
}


static inline void
GENERIC_SWAP(char *a, char *b, size_t len)
{
    // 使用循环逐个交换 a 和 b 指向的数据中的字节，长度为 len
    while(len--) {
        const char t = *a;
        *a++ = *b;
        *b++ = t;
    }
}
```