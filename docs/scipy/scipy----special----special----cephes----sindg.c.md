# `D:\src\scipysrc\scipy\scipy\special\special\cephes\sindg.c`

```
/*
 *
 *     Circular sine of angle in degrees
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, sindg();
 *
 * y = sindg( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Range reduction is into intervals of 45 degrees.
 *
 * Two polynomial approximating functions are employed.
 * Between 0 and pi/4 the sine is approximated by
 *      x  +  x**3 P(x**2).
 * Between pi/4 and pi/2 the cosine is represented as
 *      1  -  x**2 P(x**2).
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain      # trials      peak         rms
 *    IEEE      +-1000       30000      2.3e-16      5.6e-17
 *
 * ERROR MESSAGES:
 *
 *   message           condition        value returned
 * sindg total loss   x > 1.0e14 (IEEE)     0.0
 *
 */
/*
 *
 *    Circular cosine of angle in degrees
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, cosdg();
 *
 * y = cosdg( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Range reduction is into intervals of 45 degrees.
 *
 * Two polynomial approximating functions are employed.
 * Between 0 and pi/4 the cosine is approximated by
 *      1  -  x**2 P(x**2).
 * Between pi/4 and pi/2 the sine is represented as
 *      x  +  x**3 P(x**2).
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain      # trials      peak         rms
 *    IEEE     +-1000        30000       2.1e-16     5.7e-17
 *  See also sin().
 *
 */

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1985, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

#include "mconf.h"

static double sincof[] = {
    1.58962301572218447952E-10,
    -2.50507477628503540135E-8,
    2.75573136213856773549E-6,
    -1.98412698295895384658E-4,
    8.33333333332211858862E-3,
    -1.66666666666666307295E-1
};

static double coscof[] = {
    1.13678171382044553091E-11,
    -2.08758833757683644217E-9,
    2.75573155429816611547E-7,
    -2.48015872936186303776E-5,
    1.38888888888806666760E-3,
    -4.16666666666666348141E-2,
    4.99999999999999999798E-1
};

static double PI180 = 1.74532925199432957692E-2;    /* pi/180 */
static double lossth = 1.0e14;

double sindg(double x)
{
    double y, z, zz;
    int j, sign;

    /* make argument positive but save the sign */
    sign = 1;
    if (x < 0) {
        x = -x;
        sign = -1;
    }

    if (x > lossth) {
        sf_error("sindg", SF_ERROR_NO_RESULT, NULL);
        return (0.0);
    }

    y = floor(x / 45.0);    /* integer part of x/M_PI_4 */

    /* strip high bits of integer part to prevent integer overflow */
    z = ldexp(y, -4);
    z = floor(z);        /* integer part of y/8 */
    z = y - ldexp(z, 4);    /* y - 16 * (y/16) */

    j = z;            /* convert to integer for tests on the phase angle */
    /* map zeros to origin */
    if (j & 1) {
        j += 1;
        y += 1.0;
    }
    j = j & 07;            /* octant modulo 360 degrees */
    /* reflect in x axis */
    if (j > 3) {
    sign = -sign;  # 反转符号变量sign的值，相当于 sign = -1 * sign;
    j -= 4;  # 将变量j减去4

    z = x - y * 45.0;  # 计算 x mod 45 度的余数，将结果赋给变量z
    z *= PI180;  # 将z乘以PI/180，将角度转换为弧度，并更新z的值
    zz = z * z;  # 计算z的平方，将结果赋给变量zz

    if ((j == 1) or (j == 2)):
        # 如果j等于1或2，则执行以下操作
        y = 1.0 - zz * polevl(zz, coscof, 6);
    else:
        # 如果j不等于1或2，则执行以下操作
        y = z + z * (zz * polevl(zz, sincof, 5));

    if (sign < 0):
        # 如果sign小于0，则执行以下操作
        y = -y;  # 将y取负值

    return (y);  # 返回变量y作为函数的结果
}

/* 函数：计算角度 x 的余弦值，x 单位为度 */
double cosdg(double x)
{
    double y, z, zz;    /* 定义变量 y, z, zz，用于中间计算 */
    int j, sign;        /* 定义整型变量 j, sign，用于存储整数和符号 */

    /* 使角度参数 x 变为正数 */
    sign = 1;
    if (x < 0)
        x = -x;

    /* 若角度超出 lossth，返回错误并返回 0.0 */
    if (x > lossth) {
        sf_error("cosdg", SF_ERROR_NO_RESULT, NULL);
        return (0.0);
    }

    y = floor(x / 45.0);    /* 求 x 除以 45 的整数部分 y */
    z = ldexp(y, -4);       /* z = y / 16 */
    z = floor(z);           /* z 取整，即 y / 8 的整数部分 */
    z = y - ldexp(z, 4);    /* z = y - 16 * (y / 16) */

    /* 取整数和小数部分模八的余数 */
    j = z;
    if (j & 1) {            /* 如果 j 是奇数，将 j 加一，同时 y 加一 */
        j += 1;
        y += 1.0;
    }
    j = j & 07;             /* j 取模 8 的余数 */
    if (j > 3) {            /* 如果 j 大于 3，j 减去 4，并改变 sign 符号 */
        j -= 4;
        sign = -sign;
    }

    if (j > 1)              /* 如果 j 大于 1，改变 sign 符号 */
        sign = -sign;

    z = x - y * 45.0;       /* 计算 x 对 45 取模的余数，单位为度 */
    z *= PI180;             /* 将余数乘以 PI/180 转换为弧度 */

    zz = z * z;             /* zz = z 的平方 */

    if ((j == 1) || (j == 2)) {
        y = z + z * (zz * polevl(zz, sincof, 5));    /* 根据 j 的值选择不同的多项式计算 sin */
    }
    else {
        y = 1.0 - zz * polevl(zz, coscof, 6);        /* 根据 j 的值选择不同的多项式计算 cos */
    }

    if (sign < 0)           /* 如果 sign 小于 0，将 y 取负 */
        y = -y;

    return (y);             /* 返回计算得到的 cos 值 */
}


/* 函数：将角度、分、秒转换为弧度 */
/* 1 角秒对应的弧度值 = 4.848136811095359935899141023579479759563533023727e-6 */
static double P64800 =
    4.848136811095359935899141023579479759563533023727e-6;

double radian(double d, double m, double s)
{
    return (((d * 60.0 + m) * 60.0 + s) * P64800);    /* 计算角度、分、秒对应的弧度值并返回 */
}
```