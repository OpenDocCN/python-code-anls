# `.\numpy\numpy\_core\src\npymath\ieee754.cpp`

```
/*
 * -*- c -*-
 *
 * Low-level routines related to IEEE-754 format
 */

/*
 * vim:syntax=c
 *
 * Low-level routines related to IEEE-754 format
 */

/*
 * The below code is provided for compilers which do not yet provide C11
 * compatibility (gcc 4.5 and older)
 */
#include "numpy/utils.h"

#include "npy_math_common.h"
#include "npy_math_private.h"

/*
 * FIXME: There is a lot of redundancy between _next* and npy_nextafter*.
 * refactor this at some point
 *
 * p >= 0, return x + nulp
 * p < 0, return x - nulp
 */
static double
_next(double x, int p)
{
    volatile double t;
    npy_int32 hx, hy, ix;
    npy_uint32 lx;

    EXTRACT_WORDS(hx, lx, x);
    ix = hx & 0x7fffffff; /* |x| */

    if (((ix >= 0x7ff00000) && ((ix - 0x7ff00000) | lx) != 0)) /* x is nan */
        return x;
    if ((ix | lx) == 0) { /* x == 0 */
        if (p >= 0) {
            INSERT_WORDS(x, 0x0, 1); /* return +minsubnormal */
        }
        else {
            INSERT_WORDS(x, 0x80000000, 1); /* return -minsubnormal */
        }
        t = x * x;
        if (t == x)
            return t;
        else
            return x; /* raise underflow flag */
    }
    if (p < 0) { /* x -= ulp */
        if (lx == 0)
            hx -= 1;
        lx -= 1;
    }
    else { /* x += ulp */
        lx += 1;
        if (lx == 0)
            hx += 1;
    }
    hy = hx & 0x7ff00000;
    if (hy >= 0x7ff00000)
        return x + x;      /* overflow  */
    if (hy < 0x00100000) { /* underflow */
        t = x * x;
        if (t != x) { /* raise underflow flag */
            INSERT_WORDS(x, hx, lx);
            return x;
        }
    }
    INSERT_WORDS(x, hx, lx);
    return x;
}

/*
 * FIXME: There is a lot of redundancy between _next* and npy_nextafter*.
 * refactor this at some point
 *
 * p >= 0, return x + nulp
 * p < 0, return x - nulp
 */
static float
_next(float x, int p)
{
    volatile float t;
    npy_int32 hx, hy, ix;

    GET_FLOAT_WORD(hx, x);
    ix = hx & 0x7fffffff; /* |x| */

    if ((ix > 0x7f800000)) /* x is nan */
        return x;
    if (ix == 0) { /* x == 0 */
        if (p >= 0) {
            SET_FLOAT_WORD(x, 0x0 | 1); /* return +minsubnormal */
        }
        else {
            SET_FLOAT_WORD(x, 0x80000000 | 1); /* return -minsubnormal */
        }
        t = x * x;
        if (t == x)
            return t;
        else
            return x; /* raise underflow flag */
    }
    if (p < 0) { /* x -= ulp */
        hx -= 1;
    }
    else { /* x += ulp */
        hx += 1;
    }
    hy = hx & 0x7f800000;
    if (hy >= 0x7f800000)
        return x + x;      /* overflow  */
    if (hy < 0x00800000) { /* underflow */
        t = x * x;
        if (t != x) { /* raise underflow flag */
            SET_FLOAT_WORD(x, hx);
            return x;
        }
    }
    SET_FLOAT_WORD(x, hx);
    return x;
}

#if defined(HAVE_LDOUBLE_DOUBLE_DOUBLE_BE) || \
        defined(HAVE_LDOUBLE_DOUBLE_DOUBLE_LE)

/*
 * FIXME: this is ugly and untested. The asm part only works with gcc, and we
 * should consolidate the GET_LDOUBLE* / SET_LDOUBLE macros
 */
#define math_opt_barrier(x)    \
    ({                         \
        __typeof(x) __x = x;   \  # 定义一个临时变量 __x，类型与 x 相同，用于存储 x 的值
        __asm("" : "+m"(__x)); \  # 使用内联汇编语句，操作 __x 变量，"+m" 表示将 __x 视为内存输入输出操作数
        __x;                   \  # 返回 __x 变量的值
    })
/* 定义宏 math_force_eval(x)，强制评估 x 的值 */
#define math_force_eval(x) __asm __volatile("" : : "m"(x))

/* 定义联合体 ieee854_long_double_shape_type，用于处理 IEEE 854 格式的长双精度浮点数 */
typedef union {
    npy_longdouble value;                 /* 长双精度浮点数值 */
    struct {
        npy_uint64 msw;                  /* 高位 64 位整数部分 */
        npy_uint64 lsw;                  /* 低位 64 位整数部分 */
    } parts64;
    struct {
        npy_uint32 w0, w1, w2, w3;       /* 32 位整数部分分解 */
    } parts32;
} ieee854_long_double_shape_type;

/* 从长双精度浮点数中获取两个 64 位整数 */
#define GET_LDOUBLE_WORDS64(ix0, ix1, d)     \
    do {                                     \
        ieee854_long_double_shape_type qw_u; \
        qw_u.value = (d);                    \
        (ix0) = qw_u.parts64.msw;            \
        (ix1) = qw_u.parts64.lsw;            \
    } while (0)

/* 将两个 64 位整数设置为长双精度浮点数 */
#define SET_LDOUBLE_WORDS64(d, ix0, ix1)     \
    do {                                     \
        ieee854_long_double_shape_type qw_u; \
        qw_u.parts64.msw = (ix0);            \
        qw_u.parts64.lsw = (ix1);            \
        (d) = qw_u.value;                    \
    } while (0)

/* 静态函数 _next，用于实现浮点数运算 */
static long double
_next(long double x, int p)
{
    npy_int64 hx, ihx, ilx;                 /* 高位和低位整数部分 */
    npy_uint64 lx;                          /* 无符号长整数部分 */
    npy_longdouble u;                       /* 长双精度浮点数 u */
    const npy_longdouble eps = exp2l(-105.); /* 0x1.0000000000000p-105L，用于计算精度 */

    GET_LDOUBLE_WORDS64(hx, lx, x);         /* 获取长双精度浮点数 x 的整数部分 */

    ihx = hx & 0x7fffffffffffffffLL;        /* 取 x 的绝对值的整数部分 */
    ilx = lx & 0x7fffffffffffffffLL;        /* 取 x 的绝对值的低位整数部分 */

    if (((ihx & 0x7ff0000000000000LL) == 0x7ff0000000000000LL) &&
        ((ihx & 0x000fffffffffffffLL) != 0)) {
        return x;                           /* 如果 x 是 NaN，则返回 x */
    }
    if (ihx == 0 && ilx == 0) {              /* 如果 x 等于 0 */
        SET_LDOUBLE_WORDS64(x, p, 0ULL);    /* 设置 x 为 +-minsubnormal */
        u = x * x;
        if (u == x) {
            return u;                       /* 如果 u 等于 x，则返回 u */
        }
        else {
            return x;                       /* 否则返回 x */
        }
    }

    if (p < 0) {                            /* 如果 p 小于 0 */
        if ((hx == 0xffefffffffffffffLL) && (lx == 0xfc8ffffffffffffeLL))
            return x + x;                   /* 如果溢出，返回 -inf */
        if (hx >= 0x7ff0000000000000LL) {   /* 如果 x 是无穷大 */
            SET_LDOUBLE_WORDS64(u, 0x7fefffffffffffffLL, 0x7c8ffffffffffffeLL);
            return u;                       /* 返回特定值 u */
        }
        if (ihx <= 0x0360000000000000LL) {  /* 如果 x 小于等于 LDBL_MIN */
            u = math_opt_barrier(x);        /* 对 x 进行数学优化屏障 */
            x -= LDBL_TRUE_MIN;             /* 减去 LDBL_TRUE_MIN */
            if (ihx < 0x0360000000000000LL || (hx > 0 && (npy_int64)lx <= 0) ||
                (hx < 0 && (npy_int64)lx > 1)) {
                u = u * u;
                math_force_eval(u);         /* 强制评估 u，触发下溢标志 */
            }
            return x;                       /* 返回 x */
        }
        if (ihx < 0x06a0000000000000LL) {   /* 如果 ulp 会变成非规格化数 */
            SET_LDOUBLE_WORDS64(u, (hx & 0x7ff0000000000000LL), 0ULL);
            u *= eps;                       /* u 乘以 eps */
        }
        else
            SET_LDOUBLE_WORDS64(
                    u, (hx & 0x7ff0000000000000LL) - 0x0690000000000000LL,
                    0ULL);
        return x - u;                       /* 返回 x - u */
    }


这段代码是关于处理长双精度浮点数的宏定义和函数实现，主要用于浮点数的精确计算和特殊情况处理。
    else { /* p >= 0, x += ulp */
        # 如果 p >= 0，意味着需要增加 ulp 到 x 上

        if ((hx == 0x7fefffffffffffffLL) && (lx == 0x7c8ffffffffffffeLL))
            # 如果 x 是正无穷大，返回 +inf
            return x + x; /* overflow, return +inf */

        if ((npy_uint64)hx >= 0xfff0000000000000ULL) {
            # 如果 x 是 NaN 或者无穷大，设置返回值为 NaN
            SET_LDOUBLE_WORDS64(u, 0xffefffffffffffffLL, 0xfc8ffffffffffffeLL);
            return u;
        }

        if (ihx <= 0x0360000000000000LL) { /* x <= LDBL_MIN */
            # 如果 x 小于等于最小长双精度浮点数 LDBL_MIN

            u = math_opt_barrier(x);
            # 对 x 进行优化处理，可能是某种数学优化栅栏
            x += LDBL_TRUE_MIN;
            # 将 x 增加真实的最小长双精度浮点数

            if (ihx < 0x0360000000000000LL ||
                (hx > 0 && (npy_int64)lx < 0 && lx != 0x8000000000000001LL) ||
                (hx < 0 && (npy_int64)lx >= 0)) {
                # 如果 x 小于 LDBL_MIN 或者 (x 为正数且低位为负数且不是 -0.0L) 或者 (x 为负数且低位为正数)
                
                u = u * u;
                # 对 u 进行平方
                math_force_eval(u); /* raise underflow flag */
                # 强制评估 u，引发下溢标志
            }

            if (x == 0.0L)
                # 处理负的 LDBL_TRUE_MIN 情况
                x = -0.0L;

            return x;
            # 返回 x
        }

        if (ihx < 0x06a0000000000000LL) { /* ulp will denormal */
            # 如果 ulp 会导致非正常数（denormal）

            SET_LDOUBLE_WORDS64(u, (hx & 0x7ff0000000000000LL), 0ULL);
            # 将 u 设置为与 hx 的前 52 位相同，后面 52 位为 0
            u *= eps;
            # 将 u 乘以 eps
        }
        else {
            SET_LDOUBLE_WORDS64(
                    u, (hx & 0x7ff0000000000000LL) - 0x0690000000000000LL,
                    0ULL);
            # 将 u 设置为 hx 的前 52 位减去 0x0690000000000000LL，后面 52 位为 0
        }

        return x + u;
        # 返回 x 加上 u
    }
}
#else
static long double
_next(long double x, int p)
{
    volatile npy_longdouble t;  // 声明一个 volatile 类型的长双精浮点数 t，用于临时存储计算结果
    union IEEEl2bitsrep ux;     // 声明一个联合体 ux，用于存储长双精浮点数 x 的位表示

    ux.e = x;   // 将长双精浮点数 x 的值赋给联合体 ux 的位表示

    if ((GET_LDOUBLE_EXP(ux) == 0x7fff &&
         ((GET_LDOUBLE_MANH(ux) & ~LDBL_NBIT) | GET_LDOUBLE_MANL(ux)) != 0)) {
        return ux.e; /* x is nan */  // 如果 x 是 NaN，则直接返回 x
    }
    if (ux.e == 0.0) {
        SET_LDOUBLE_MANH(ux, 0); /* return +-minsubnormal */  // 设置 ux 的尾数部分为 +-minsubnormal
        SET_LDOUBLE_MANL(ux, 1);  // 将 ux 的尾数部分的低位设置为 1
        if (p >= 0) {
            SET_LDOUBLE_SIGN(ux, 0);  // 如果 p 大于等于 0，则设置 ux 的符号位为正号
        }
        else {
            SET_LDOUBLE_SIGN(ux, 1);  // 否则，设置 ux 的符号位为负号
        }
        t = ux.e * ux.e;  // 计算 ux.e 的平方，并将结果存储在 t 中
        if (t == ux.e) {
            return t;  // 如果 t 等于 ux.e，则返回 t
        }
        else {
            return ux.e; /* raise underflow flag */  // 否则，返回 ux.e，并可能引发下溢标志
        }
    }
    if (p < 0) { /* x -= ulp */  // 如果 p 小于 0，则表示 x 减去最后一个有效位
        if (GET_LDOUBLE_MANL(ux) == 0) {
            if ((GET_LDOUBLE_MANH(ux) & ~LDBL_NBIT) == 0) {
                SET_LDOUBLE_EXP(ux, GET_LDOUBLE_EXP(ux) - 1);  // 如果 ux 的尾数部分低位为 0，并且尾数部分高位去除掉了 LDBL_NBIT，那么将指数部分减 1
            }
            SET_LDOUBLE_MANH(ux, (GET_LDOUBLE_MANH(ux) - 1) |
                                         (GET_LDOUBLE_MANH(ux) & LDBL_NBIT));  // 将 ux 的尾数部分高位减 1，并保留 LDBL_NBIT
        }
        SET_LDOUBLE_MANL(ux, GET_LDOUBLE_MANL(ux) - 1);  // 将 ux 的尾数部分低位减 1
    }
    else { /* x += ulp */  // 否则，表示 x 加上最后一个有效位
        SET_LDOUBLE_MANL(ux, GET_LDOUBLE_MANL(ux) + 1);  // 将 ux 的尾数部分低位加 1
        if (GET_LDOUBLE_MANL(ux) == 0) {
            SET_LDOUBLE_MANH(ux, (GET_LDOUBLE_MANH(ux) + 1) |
                                         (GET_LDOUBLE_MANH(ux) & LDBL_NBIT));  // 如果 ux 的尾数部分低位变为 0，将 ux 的尾数部分高位加 1，并保留 LDBL_NBIT
            if ((GET_LDOUBLE_MANH(ux) & ~LDBL_NBIT) == 0) {
                SET_LDOUBLE_EXP(ux, GET_LDOUBLE_EXP(ux) + 1);  // 如果 ux 的尾数部分高位去除掉了 LDBL_NBIT，则将指数部分加 1
            }
        }
    }
    if (GET_LDOUBLE_EXP(ux) == 0x7fff) {
        return ux.e + ux.e; /* overflow  */  // 如果 ux 的指数部分为 0x7fff，则返回 ux.e 的双倍，并可能引发溢出
    }
    if (GET_LDOUBLE_EXP(ux) == 0) { /* underflow */  // 如果 ux 的指数部分为 0，则表示发生下溢
        if (LDBL_NBIT) {
            SET_LDOUBLE_MANH(ux, GET_LDOUBLE_MANH(ux) & ~LDBL_NBIT);  // 如果 LDBL_NBIT 为真，则清除 ux 的尾数部分高位的 LDBL_NBIT
        }
        t = ux.e * ux.e;
        if (t != ux.e) { /* raise underflow flag */  // 如果 t 不等于 ux.e，则可能引发下溢标志
            return ux.e;
        }
    }

    return ux.e;  // 返回计算后的长双精浮点数值 ux.e
}
#endif

namespace {
template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float> {
    static const npy_float nan;  // 定义 float 类型的特殊值 nan
};
const npy_float numeric_limits<float>::nan = NPY_NANF;  // 初始化 float 类型的 nan

template <>
struct numeric_limits<double> {
    static const npy_double nan;  // 定义 double 类型的特殊值 nan
};
const npy_double numeric_limits<double>::nan = NPY_NAN;  // 初始化 double 类型的 nan

template <>
struct numeric_limits<long double> {
    static const npy_longdouble nan;  // 定义 long double 类型的特殊值 nan
};
const npy_longdouble numeric_limits<long double>::nan = NPY_NANL;  // 初始化 long double 类型的 nan
}  // namespace

template <typename type>
static type
_npy_spacing(type x)
{
    /* XXX: npy isnan/isinf may be optimized by bit twiddling */  // 提示可能通过位操作对 npy 的 isnan/isinf 进行优化
    if (npy_isinf(x)) {
        return numeric_limits<type>::nan;  // 如果 x 是无穷大，则返回该类型的 nan 值
    }

    return _next(x, 1) - x;  // 否则，返回 x 的下一个值减去 x 本身的差值
}

/*
 * Instantiation of C interface
 */
extern "C" {
npy_float
npy_spacingf(npy_float x)
{
    return _npy_spacing(x);  // 返回 float 类型的 npy_spacing 函数调用结果
}
npy_double
npy_spacing(npy_double x)
{
    return _npy_spacing(x);  // 返回 double 类型的 npy_spacing 函数调用结果
}
npy_longdouble
npy_spacingl(npy_longdouble x)
{
    return _npy_spacing(x);  // 返回 long double 类型的 npy_spacing 函数调用结果
}
}

extern "C" int
npy_clear_floatstatus()
{
    char x = 0;  // 声明一个 char 类型的变量 x，并初始化为 0
    # 调用函数 npy_clear_floatstatus_barrier，传入参数 x，并返回其结果。
    return npy_clear_floatstatus_barrier(&x);
extern "C" int
npy_get_floatstatus()
{
    // 定义一个字符变量 x 并初始化为 0
    char x = 0;
    // 调用 npy_get_floatstatus_barrier 函数，传入 x 的地址作为参数，并返回结果
    return npy_get_floatstatus_barrier(&x);
}


/* 
 * 用于浮点错误处理的通用 C99 代码。这些函数主要是因为 `fenv.h` 在 C89 中未标准化，
 * 因此它们提供了更好的可移植性。在 C99/C++11 中应该不再需要这些，可以直接从 `fenv.h` 中使用更多功能。
 */
#include <fenv.h>

/*
 * 根据 C99 标准，如果 FE_DIVBYZERO 等未支持，则可能不会提供。在这种情况下，NumPy 将无法正确报告这些错误，
 * 但我们仍应允许编译（无论测试是否通过）。通过在本地定义它们为 0，我们使它们成为无操作。例如，`musl` 仍定义所有这些函数（作为无操作）：
 *     https://git.musl-libc.org/cgit/musl/tree/src/fenv/fenv.c
 * 并在其测试中执行类似的替换：
 *     http://nsz.repo.hu/git/?p=libc-test;a=blob;f=src/common/mtest.h;h=706c1ba23ea8989b17a2f72ed1a919e187c06b6a;hb=HEAD#l30
 */
#ifndef FE_DIVBYZERO
    #define FE_DIVBYZERO 0
#endif
#ifndef FE_OVERFLOW
    #define FE_OVERFLOW 0
#endif
#ifndef FE_UNDERFLOW
    #define FE_UNDERFLOW 0
#endif
#ifndef FE_INVALID
    #define FE_INVALID 0
#endif


extern "C" int
npy_get_floatstatus_barrier(char *param)
{
    // 使用 fetestexcept 函数获取当前浮点异常状态并存储在 fpstatus 中
    int fpstatus = fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW |
                                FE_INVALID);
    /*
     * 通过使用 volatile，编译器无法重新排序此调用
     */
    // 如果传入的 param 不为 NULL，则声明一个 volatile 的字符变量 c，并使用 param 转换后的指针来初始化它
    if (param != NULL) {
        volatile char NPY_UNUSED(c) = *(char *)param;
    }

    // 根据 fpstatus 的位掩码，返回相应的浮点异常错误码
    return ((FE_DIVBYZERO & fpstatus) ? NPY_FPE_DIVIDEBYZERO : 0) |
           ((FE_OVERFLOW & fpstatus) ? NPY_FPE_OVERFLOW : 0) |
           ((FE_UNDERFLOW & fpstatus) ? NPY_FPE_UNDERFLOW : 0) |
           ((FE_INVALID & fpstatus) ? NPY_FPE_INVALID : 0);
}

extern "C" int
npy_clear_floatstatus_barrier(char *param)
{
    /* 在 x86 上测试浮点状态比清除状态快 50-100 倍 */
    // 调用 npy_get_floatstatus_barrier 函数获取当前浮点异常状态并存储在 fpstatus 中
    int fpstatus = npy_get_floatstatus_barrier(param);
    // 如果 fpstatus 不为 0，则调用 feclearexcept 函数清除 FE_DIVBYZERO、FE_OVERFLOW、FE_UNDERFLOW 和 FE_INVALID 异常
    if (fpstatus != 0) {
        feclearexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    }

    // 返回 fpstatus
    return fpstatus;
}

extern "C" void
npy_set_floatstatus_divbyzero(void)
{
    // 调用 feraiseexcept 函数，设置 FE_DIVBYZERO 异常
    feraiseexcept(FE_DIVBYZERO);
}

extern "C" void
npy_set_floatstatus_overflow(void)
{
    // 调用 feraiseexcept 函数，设置 FE_OVERFLOW 异常
    feraiseexcept(FE_OVERFLOW);
}

extern "C" void
npy_set_floatstatus_underflow(void)
{
    // 调用 feraiseexcept 函数，设置 FE_UNDERFLOW 异常
    feraiseexcept(FE_UNDERFLOW);
}

extern "C" void
npy_set_floatstatus_invalid(void)
{
    // 调用 feraiseexcept 函数，设置 FE_INVALID 异常
    feraiseexcept(FE_INVALID);
}
```