# `D:\src\scipysrc\scipy\scipy\special\special\cephes\j0.h`

```
/*
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
 */

/* Note: all coefficients satisfy the relative error criterion
 * except YP, YQ which are designed for absolute error. */

// 指定 Cephes 数学库的版本信息和版权声明

/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

// 代码由 SciPy 开发者在 2024 年翻译成 C++，原始的版权信息如下。

/*                                                     j0.c
 *
 *     Bessel function of order zero
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, j0();
 *
 * y = j0( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns Bessel function of order zero of the argument.
 *
 * The domain is divided into the intervals [0, 5] and
 * (5, infinity). In the first interval the following rational
 * approximation is used:
 *
 *
 *        2         2
 * (w - r  ) (w - r  ) P (w) / Q (w)
 *       1         2    3       8
 *
 *            2
 * where w = x  and the two r's are zeros of the function.
 *
 * In the second interval, the Hankel asymptotic expansion
 * is employed with two rational functions of degree 6/6
 * and 7/7.
 *
 *
 *
 * ACCURACY:
 *
 *                      Absolute error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0, 30       60000       4.2e-16     1.1e-16
 *
 */

// j0.c: 计算零阶贝塞尔函数 J0(x) 的函数实现

/*                            y0.c
 *
 *    Bessel function of the second kind, order zero
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, y0();
 *
 * y = y0( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns Bessel function of the second kind, of order
 * zero, of the argument.
 *
 * The domain is divided into the intervals [0, 5] and
 * (5, infinity). In the first interval a rational approximation
 * R(x) is employed to compute
 *   y0(x)  = R(x)  +   2 * log(x) * j0(x) / M_PI.
 * Thus a call to j0() is required.
 *
 * In the second interval, the Hankel asymptotic expansion
 * is employed with two rational functions of degree 6/6
 * and 7/7.
 *
 *
 *
 * ACCURACY:
 *
 *  Absolute error, when y0(x) < 1; else relative error:
 *
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0, 30       30000       1.3e-15     1.6e-16
 *
 */

// y0.c: 计算零阶第二类贝塞尔函数 Y0(x) 的函数实现
    // 声明并初始化 Bessel 函数 J0 的系数数组 j0_PP，长度为 7
    constexpr double j0_PP[7] = {
        7.96936729297347051624E-4, 8.28352392107440799803E-2, 1.23953371646414299388E0,  5.44725003058768775090E0,
        8.74716500199817011941E0,  5.30324038235394892183E0,  9.99999999999999997821E-1,
    };
    
    // 声明并初始化 Bessel 函数 J0 的系数数组 j0_PQ，长度为 7
    constexpr double j0_PQ[7] = {
        9.24408810558863637013E-4, 8.56288474354474431428E-2, 1.25352743901058953537E0, 5.47097740330417105182E0,
        8.76190883237069594232E0,  5.30605288235394617618E0,  1.00000000000000000218E0,
    };
    
    // 声明并初始化 Bessel 函数 J0 的系数数组 j0_QP，长度为 8
    constexpr double j0_QP[8] = {
        -1.13663838898469149931E-2, -1.28252718670509318512E0, -1.95539544257735972385E1, -9.32060152123768231369E1,
        -1.77681167980488050595E2,  -1.47077505154951170175E2, -5.14105326766599330220E1, -6.05014350600728481186E0,
    };
    
    // 声明并初始化 Bessel 函数 J0 的系数数组 j0_QQ，长度为 7
    constexpr double j0_QQ[7] = {
        /*  1.00000000000000000000E0, */
        6.43178256118178023184E1, 8.56430025976980587198E2, 3.88240183605401609683E3, 7.24046774195652478189E3,
        5.93072701187316984827E3, 2.06209331660327847417E3, 2.42005740240291393179E2,
    };
    
    // 声明并初始化 Bessel 函数 J0 的系数数组 j0_YP，长度为 8
    constexpr double j0_YP[8] = {
        1.55924367855235737965E4,   -1.46639295903971606143E7,  5.43526477051876500413E9,
        -9.82136065717911466409E11, 8.75906394395366999549E13,  -3.46628303384729719441E15,
        4.42733268572569800351E16,  -1.84950800436986690637E16,
    };
    
    // 声明并初始化 Bessel 函数 J0 的系数数组 j0_YQ，长度为 7
    constexpr double j0_YQ[7] = {
        /* 1.00000000000000000000E0, */
        1.04128353664259848412E3,  6.26107330137134956842E5,  2.68919633393814121987E8,  8.64002487103935000337E10,
        2.02979612750105546709E13, 3.17157752842975028269E15, 2.50596256172653059228E17,
    };
    
    // 声明并初始化 Bessel 函数 J0 的系数 j0_DR1
    constexpr double j0_DR1 = 5.78318596294678452118E0;
    
    // 声明并初始化 Bessel 函数 J0 的系数 j0_DR2
    constexpr double j0_DR2 = 3.04712623436620863991E1;
    
    // 声明并初始化 Bessel 函数 J0 的系数数组 j0_RP，长度为 4
    constexpr double j0_RP[4] = {
        -4.79443220978201773821E9,
        1.95617491946556577543E12,
        -2.49248344360967716204E14,
        9.70862251047306323952E15,
    };
    
    // 声明并初始化 Bessel 函数 J0 的系数数组 j0_RQ，长度为 8
    constexpr double j0_RQ[8] = {
        /* 1.00000000000000000000E0, */
        4.99563147152651017219E2,  1.73785401676374683123E5,  4.84409658339962045305E7,  1.11855537045356834862E10,
        2.11277520115489217587E12, 3.10518229857422583814E14, 3.18121955943204943306E16, 1.71086294081043136091E18,
    };
    # 定义一个内联函数，计算第一类零阶贝塞尔函数 J0(x)
    SPECFUN_HOST_DEVICE inline double j0(double x) {
        double w, z, p, q, xn;

        // 如果 x 小于 0，则取其相反数，保证 x 为非负值
        if (x < 0) {
            x = -x;
        }

        // 如果 x 小于等于 5.0，使用近似计算贝塞尔函数 J0(x)
        if (x <= 5.0) {
            z = x * x;
            // 如果 x 很接近于零，返回 1.0 - z / 4.0 的近似值
            if (x < 1.0e-5) {
                return (1.0 - z / 4.0);
            }

            // 使用多项式近似计算 J0(x)
            p = (z - detail::j0_DR1) * (z - detail::j0_DR2);
            p = p * polevl(z, detail::j0_RP, 3) / p1evl(z, detail::j0_RQ, 8);
            return (p);
        }

        // 如果 x 大于 5.0，采用另一种近似方法计算 J0(x)
        w = 5.0 / x;
        q = 25.0 / (x * x);
        p = polevl(q, detail::j0_PP, 6) / polevl(q, detail::j0_PQ, 6);
        q = polevl(q, detail::j0_QP, 7) / p1evl(q, detail::j0_QQ, 7);
        xn = x - M_PI_4;
        // 计算 J0(x) 的近似值，考虑余弦和正弦函数
        p = p * std::cos(xn) - w * q * std::sin(xn);
        // 返回 J0(x) 的最终近似值
        return (p * detail::SQRT2OPI / std::sqrt(x));
    }

    /*                                                     y0() 2  */
    /* 贝塞尔函数第二类零阶，order zero */

    /* 使用有理数逼近系数 YP[]，YQ[] 来计算。
     * 计算的函数是 y0(x) - 2 * log(x) * j0(x) / M_PI，
     * 在 x = 0 时的值为 2 * ( log(0.5) + EUL ) / M_PI
     * = 0.073804295108687225.
     */

    // 定义一个内联函数，计算贝塞尔函数第二类零阶 Y0(x)
    SPECFUN_HOST_DEVICE inline double y0(double x) {
        double w, z, p, q, xn;

        // 如果 x 小于等于 5.0，采用近似方法计算 Y0(x)
        if (x <= 5.0) {
            // 如果 x 等于 0.0，设置错误并返回负无穷大
            if (x == 0.0) {
                set_error("y0", SF_ERROR_SINGULAR, NULL);
                return -std::numeric_limits<double>::infinity();
            } else if (x < 0.0) {
                // 如果 x 小于 0.0，设置域错误并返回 NaN
                set_error("y0", SF_ERROR_DOMAIN, NULL);
                return std::numeric_limits<double>::quiet_NaN();
            }
            z = x * x;
            // 使用多项式逼近计算 Y0(x)
            w = polevl(z, detail::j0_YP, 7) / p1evl(z, detail::j0_YQ, 7);
            // 添加修正项，包含对 J0(x) 的调整
            w += M_2_PI * std::log(x) * j0(x);
            // 返回 Y0(x) 的近似值
            return (w);
        }

        // 如果 x 大于 5.0，采用另一种近似方法计算 Y0(x)
        w = 5.0 / x;
        z = 25.0 / (x * x);
        p = polevl(z, detail::j0_PP, 6) / polevl(z, detail::j0_PQ, 6);
        q = polevl(z, detail::j0_QP, 7) / p1evl(z, detail::j0_QQ, 7);
        xn = x - M_PI_4;
        // 计算 Y0(x) 的近似值，考虑正弦和余弦函数
        p = p * std::sin(xn) + w * q * std::cos(xn);
        // 返回 Y0(x) 的最终近似值
        return (p * detail::SQRT2OPI / std::sqrt(x));
    }
} // end of namespace cephes
} // end of namespace special
```