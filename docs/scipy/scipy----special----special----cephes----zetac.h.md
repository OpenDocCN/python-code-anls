# `D:\src\scipysrc\scipy\scipy\special\special\cephes\zetac.h`

```
/*
 * Cephes Math Library Release 2.1:  January, 1989
 * Copyright 1984, 1987, 1989 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#pragma once

#include "../config.h"

#include "const.h"
#include "lanczos.h"
#include "polevl.h"
#include "zeta.h"

namespace special {
namespace cephes {

    /*
     * Riemann zeta function, minus one
     */
    // 定义函数 zetac，计算 Riemann zeta 函数的补函数 zetac(x)
    SPECFUN_HOST_DEVICE inline double zetac(double x) {
        // 如果输入是 NaN，直接返回 NaN
        if (std::isnan(x)) {
            return x;
        }
        // 如果输入是负无穷大，返回 quiet NaN
        else if (x == -std::numeric_limits<double>::infinity()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 如果输入在 -0.01 到 0 之间，使用 detail::zetac_smallneg 函数计算并返回结果
        else if (x < 0.0 && x > -0.01) {
            return detail::zetac_smallneg(x);
        }
        // 如果输入小于 0，使用 zeta_reflection 函数计算 Riemann zeta 函数的反函数结果，并减去 1
        else if (x < 0.0) {
            return detail::zeta_reflection(-x) - 1;
        }
        // 对于其他情况（即 x >= 0），使用 detail::zetac_positive 函数计算 zetac(x) 的正值部分并返回
        else {
            return detail::zetac_positive(x);
        }
    }

    /*
     * Riemann zeta function
     */
    // 定义函数 riemann_zeta，计算 Riemann zeta 函数
    SPECFUN_HOST_DEVICE inline double riemann_zeta(double x) {
        // 如果输入是 NaN，直接返回 NaN
        if (std::isnan(x)) {
            return x;
        }
        // 如果输入是负无穷大，返回 quiet NaN
        else if (x == -std::numeric_limits<double>::infinity()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // 如果输入在 -0.01 到 0 之间，使用 detail::zetac_smallneg 函数计算并返回结果，加上 1
        else if (x < 0.0 && x > -0.01) {
            return 1 + detail::zetac_smallneg(x);
        }
        // 如果输入小于 0，使用 zeta_reflection 函数计算 Riemann zeta 函数的反函数结果
        else if (x < 0.0) {
            return detail::zeta_reflection(-x);
        }
        // 对于其他情况（即 x >= 0），使用 detail::zetac_positive 函数计算 zetac(x) 的正值部分并加上 1 返回
        else {
            return 1 + detail::zetac_positive(x);
        }
    }

} // namespace cephes
} // namespace special
```