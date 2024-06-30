# `D:\src\scipysrc\scipy\scipy\special\special\cephes\igami.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * (C) Copyright John Maddock 2006.
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "gamma.h"
#include "igam.h"
#include "polevl.h"

namespace special {
namespace cephes {

    } // namespace detail

    // 声明 igamci 函数，用于计算修正的不完全 Gamma 函数的逆
    SPECFUN_HOST_DEVICE inline double igamci(double a, double q);

    // 定义 igami 函数，用于计算不完全 Gamma 函数的逆
    SPECFUN_HOST_DEVICE inline double igami(double a, double p) {
        int i;
        double x, fac, f_fp, fpp_fp;

        // 检查输入是否为 NaN
        if (std::isnan(a) || std::isnan(p)) {
            return std::numeric_limits<double>::quiet_NaN();
        } else if ((a < 0) || (p < 0) || (p > 1)) {  // 检查参数范围
            set_error("gammaincinv", SF_ERROR_DOMAIN, NULL);
        } else if (p == 0.0) {  // 特殊情况：当 p 为 0 时，返回结果为 0
            return 0.0;
        } else if (p == 1.0) {  // 特殊情况：当 p 为 1 时，返回结果为正无穷大
            return std::numeric_limits<double>::infinity();
        } else if (p > 0.9) {  // 当 p 大于 0.9 时，调用 igamci 函数
            return igamci(a, 1 - p);
        }

        // 使用详细函数 find_inverse_gamma 寻找初始估计值 x
        x = detail::find_inverse_gamma(a, p, 1 - p);

        /* Halley's method */
        // 使用 Halley 方法进行迭代求解
        for (i = 0; i < 3; i++) {
            fac = detail::igam_fac(a, x);
            if (fac == 0.0) {
                return x;
            }
            f_fp = (igam(a, x) - p) * x / fac;
            // 计算一阶导数与二阶导数的比值
            fpp_fp = -1.0 + (a - 1) / x;
            if (std::isinf(fpp_fp)) {
                // 在溢出的情况下回退到牛顿方法
                x = x - f_fp;
            } else {
                // 使用 Halley 方法更新 x 的值
                x = x - f_fp / (1.0 - 0.5 * f_fp * fpp_fp);
            }
        }

        return x;
    }

    // 定义 igamci 函数，用于计算修正的不完全 Gamma 函数的逆
    SPECFUN_HOST_DEVICE inline double igamci(double a, double q) {
        int i;
        double x, fac, f_fp, fpp_fp;

        // 检查输入是否为 NaN
        if (std::isnan(a) || std::isnan(q)) {
            return std::numeric_limits<double>::quiet_NaN();
        } else if ((a < 0.0) || (q < 0.0) || (q > 1.0)) {  // 检查参数范围
            set_error("gammainccinv", SF_ERROR_DOMAIN, NULL);
        } else if (q == 0.0) {  // 特殊情况：当 q 为 0 时，返回结果为正无穷大
            return std::numeric_limits<double>::infinity();
        } else if (q == 1.0) {  // 特殊情况：当 q 为 1 时，返回结果为 0
            return 0.0;
        } else if (q > 0.9) {  // 当 q 大于 0.9 时，调用 igami 函数
            return igami(a, 1 - q);
        }

        // 使用详细函数 find_inverse_gamma 寻找初始估计值 x
        x = detail::find_inverse_gamma(a, 1 - q, q);
        // 使用 Halley 方法进行迭代求解
        for (i = 0; i < 3; i++) {
            fac = detail::igam_fac(a, x);
            if (fac == 0.0) {
                return x;
            }
            f_fp = (igamc(a, x) - q) * x / (-fac);
            fpp_fp = -1.0 + (a - 1) / x;
            if (std::isinf(fpp_fp)) {
                // 在溢出的情况下回退到牛顿方法
                x = x - f_fp;
            } else {
                // 使用 Halley 方法更新 x 的值
                x = x - f_fp / (1.0 - 0.5 * f_fp * fpp_fp);
            }
        }

        return x;
    }

} // namespace cephes
} // namespace special
```