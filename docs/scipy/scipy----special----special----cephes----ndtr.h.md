# `D:\src\scipysrc\scipy\scipy\special\special\cephes\ndtr.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     ndtr.c
 *
 *     Normal distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, ndtr();
 *
 * y = ndtr( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the area under the Gaussian probability density
 * function, integrated from minus infinity to x:
 *
 *                            x
 *                             -
 *                   1        | |          2
 *    ndtr(x)  = ---------    |    exp( - t /2 ) dt
 *               sqrt(2pi)  | |
 *                           -
 *                          -inf.
 *
 *             =  ( 1 + erf(z) ) / 2
 *             =  erfc(z) / 2
 *
 * where z = x/sqrt(2). Computation is via the functions
 * erf and erfc.
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     -13,0        30000       3.4e-14     6.7e-15
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition         value returned
 * erfc underflow    x > 37.519379347       0.0
 *
 */
/*                            erf.c
 *
 *    Error function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, erf();
 *
 * y = erf( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * The integral is
 *
 *                           x
 *                            -
 *                 2         | |          2
 *   erf(x)  =  --------     |    exp( - t  ) dt.
 *              sqrt(pi)   | |
 *                          -
 *                           0
 *
 * For 0 <= |x| < 1, erf(x) = x * P4(x**2)/Q5(x**2); otherwise
 * erf(x) = 1 - erfc(x).
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,1         30000       3.7e-16     1.0e-16
 *
 */
/*                            erfc.c
 *
 *    Complementary error function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, erfc();
 *
 * y = erfc( x );
 *
 *
 *
 * DESCRIPTION:
 *
 *
 *  1 - erf(x) =
 *
 *                           inf.
 *                             -
 *                  2         | |          2
 *   erfc(x)  =  --------     |    exp( - t  ) dt
 *               sqrt(pi)   | |
 *                           -
 *                            x
 *
 *
 * For small x, erfc(x) = 1 - erf(x); otherwise rational
 * approximations are computed.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,26.6417   30000       5.7e-14     1.5e-14
 */

/*
 * Cephes Math Library Release 2.2:  June, 1992
 * Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"

#include "const.h"
#include "polevl.h"

namespace special {
namespace cephes {
    namespace detail {
    
        // 累积分布函数 (Normal Distribution) 的计算所需的系数数组
        constexpr double ndtr_P[] = {2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0,
                                     4.86371970985681366614E1,   1.96520832956077098242E2,  5.26445194995477358631E2,
                                     9.34528527171957607540E2,   1.02755188689515710272E3,  5.57535335369399327526E2};
    
        // 累积分布函数 (Normal Distribution) 的计算所需的系数数组
        constexpr double ndtr_Q[] = {
            /* 1.00000000000000000000E0, */
            1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2, 9.75708501743205489753E2,
            1.82390916687909736289E3, 2.24633760818710981792E3, 1.65666309194161350182E3, 5.57535340817727675546E2};
    
        // 累积分布函数 (Normal Distribution) 的计算所需的系数数组
        constexpr double ndtr_R[] = {5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0,
                                     6.16021097993053585195E0,  7.40974269950448939160E0, 2.97886665372100240670E0};
    
        // 累积分布函数 (Normal Distribution) 的计算所需的系数数组
        constexpr double ndtr_S[] = {
            /* 1.00000000000000000000E0, */
            2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1,
            1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0};
    
        // 累积分布函数 (Normal Distribution) 的计算所需的系数数组
        constexpr double ndtr_T[] = {9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3,
                                     7.00332514112805075473E3, 5.55923013010394962768E4};
    
        // 累积分布函数 (Normal Distribution) 的计算所需的系数数组
        constexpr double ndtr_U[] = {
            /* 1.00000000000000000000E0, */
            3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3, 2.26290000613890934246E4,
            4.92673942608635921086E4};
    
        // 逆正态分布函数的阈值
        constexpr double ndtri_UTHRESH = 37.519379347;
    
    } // namespace detail
    
    // 计算误差函数 (Error Function)
    SPECFUN_HOST_DEVICE inline double erf(double x);
    
    // 计算补余误差函数 (Complementary Error Function)
    SPECFUN_HOST_DEVICE inline double erfc(double a) {
        double p, q, x, y, z;
    
        // 如果输入参数是 NaN，则设置错误并返回 NaN
        if (std::isnan(a)) {
            set_error("erfc", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        }
    
        // 如果输入参数小于 0，则取其相反数
        if (a < 0.0) {
            x = -a;
        } else {
            x = a;
        }
    
        // 如果输入参数小于 1，则通过误差函数 erf(a) 计算补余误差函数 erfc(a)
        if (x < 1.0) {
            return 1.0 - erf(a);
        }
    
        // 计算 z = -a^2
        z = -a * a;
    
        // 如果 z 的值小于 -MAXLOG，跳转到 under 标签处理
        if (z < -detail::MAXLOG) {
            goto under;
        }
    
        // 计算 z 的指数函数
        z = std::exp(z);
    
        // 根据 x 的值选择合适的系数数组进行计算
        if (x < 8.0) {
            p = polevl(x, detail::ndtr_P, 8);
            q = p1evl(x, detail::ndtr_Q, 8);
        } else {
            p = polevl(x, detail::ndtr_R, 5);
            q = p1evl(x, detail::ndtr_S, 6);
        }
        y = (z * p) / q;
    
        // 如果输入参数 a 小于 0，则返回 2 - y，否则直接返回 y
        if (a < 0) {
            y = 2.0 - y;
        }
    
        // 如果 y 不等于 0，则返回 y；否则跳转到 under 标签处理
        if (y != 0.0) {
            return y;
        }
    
    under:
        // 设置错误并返回对应的值
        set_error("erfc", SF_ERROR_UNDERFLOW, NULL);
        if (a < 0) {
            return 2.0;
        } else {
            return 0.0;
        }
    }
    # 定义一个内联函数 erf，计算误差函数的值
    SPECFUN_HOST_DEVICE inline double erf(double x) {
        double y, z;

        # 检查输入是否为 NaN，如果是则设置错误并返回 NaN
        if (std::isnan(x)) {
            set_error("erf", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        }

        # 如果 x 小于 0，则利用 erf(-x) 的对称性负值来计算 erf(x)
        if (x < 0.0) {
            return -erf(-x);
        }

        # 当 x 绝对值大于 1 时，利用 erfc 函数的关系计算 erf(x) 的值
        if (std::abs(x) > 1.0) {
            return (1.0 - erfc(x));
        }

        # 对于其他情况，计算 erf(x) 的近似值
        z = x * x;
        y = x * polevl(z, detail::ndtr_T, 4) / p1evl(z, detail::ndtr_U, 5);
        return y;
    }

    # 定义一个内联函数 ndtr，计算标准正态分布函数的值
    SPECFUN_HOST_DEVICE inline double ndtr(double a) {
        double x, y, z;

        # 检查输入是否为 NaN，如果是则设置错误并返回 NaN
        if (std::isnan(a)) {
            set_error("ndtr", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        }

        # 将输入 a 转换为标准正态分布的变量 x
        x = a * M_SQRT1_2;
        z = std::abs(x);

        # 根据 x 的绝对值大小选择不同的计算路径
        if (z < M_SQRT1_2) {
            # 当 |x| < sqrt(1/2) 时，使用 erf 函数计算标准正态分布函数值
            y = 0.5 + 0.5 * erf(x);
        } else {
            # 当 |x| >= sqrt(1/2) 时，使用 erfc 函数计算标准正态分布函数值
            y = 0.5 * erfc(z);
            if (x > 0) {
                y = 1.0 - y;
            }
        }

        return y;
    }
} // 结束 cephes 命名空间
} // 结束 special 命名空间
```