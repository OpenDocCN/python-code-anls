# `D:\src\scipysrc\scipy\scipy\special\special\cephes\const.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 *
 * Since we support only IEEE-754 floating point numbers, conditional logic
 * supporting other arithmetic types has been removed.
 */



/*
 *
 *
 *                                                   const.c
 *
 *     Globally declared constants
 *
 *
 *
 * SYNOPSIS:
 *
 * extern double nameofconstant;
 *
 *
 *
 *
 * DESCRIPTION:
 *
 * This file contains a number of mathematical constants and
 * also some needed size parameters of the computer arithmetic.
 * The values are supplied as arrays of hexadecimal integers
 * for IEEE arithmetic, and in a normal decimal scientific notation for
 * other machines.  The particular notation used is determined
 * by a symbol (IBMPC, or UNK) defined in the include file
 * mconf.h.
 *
 * The default size parameters are as follows.
 *
 * For UNK mode:
 * MACHEP =  1.38777878078144567553E-17       2**-56
 * MAXLOG =  8.8029691931113054295988E1       log(2**127)
 * MINLOG = -8.872283911167299960540E1        log(2**-128)
 *
 * For IEEE arithmetic (IBMPC):
 * MACHEP =  1.11022302462515654042E-16       2**-53
 * MAXLOG =  7.09782712893383996843E2         log(2**1024)
 * MINLOG = -7.08396418532264106224E2         log(2**-1022)
 *
 * The global symbols for mathematical constants are
 * SQ2OPI =  7.9788456080286535587989E-1      sqrt( 2/pi )
 * LOGSQ2 =  3.46573590279972654709E-1        log(2)/2
 * THPIO4 =  2.35619449019234492885           3*pi/4
 *
 * These lists are subject to change.
 */
/*                                                     const.c */



/*
 * Cephes Math Library Release 2.3:  March, 1995
 * Copyright 1984, 1995 by Stephen L. Moshier
 */
#pragma once
    // 命名空间 detail 包含一些常量的定义，用于数学计算和常数处理
    
    // 最大迭代次数
    constexpr std::uint64_t MAXITER = 500;
    // 机器精度，约为 2**-53
    constexpr double MACHEP = 1.11022302462515654042E-16;
    // 双精度浮点数最大值的自然对数
    constexpr double MAXLOG = 7.09782712893383996732E2;
    // 2**-1022 的自然对数
    constexpr double MINLOG = -7.451332191019412076235E2;
    // 1/pi 的平方根
    constexpr double SQRT1OPI = 5.64189583547756286948E-1;
    // 2/pi 的平方根
    constexpr double SQRT2OPI = 7.9788456080286535587989E-1;
    // 2pi 的平方根
    constexpr double SQRT2PI = 0.79788456080286535587989;
    // log(2)/2
    constexpr double LOGSQ2 = 3.46573590279972654709E-1;
    // 3*pi/4
    constexpr double THPIO4 = 2.35619449019234492885;
    // sqrt(3)
    constexpr double SQRT3 = 1.732050807568877293527;
    // pi/180
    constexpr double PI180 = 1.74532925199432957692E-2;
    // sqrt(pi)
    constexpr double SQRTPI = 2.50662827463100050242E0;
    // log(pi)
    constexpr double LOGPI = 1.14472988584940017414;
    // Gamma 函数的最大有效参数值
    constexpr double MAXGAM = 171.624376956302725;
    // log(sqrt(pi))
    constexpr double LOGSQRT2PI = 0.9189385332046727;
    
    // 下面两个常量由 SciPy 开发者添加
    // 欧拉常数
    constexpr double SCIPY_EULER = 0.577215664901532860606512090082402431;
    // 长双精度下的 e
    constexpr long double SCIPY_El = 2.718281828459045235360287471352662498L;
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```