# `D:\src\scipysrc\scipy\scipy\special\special\cephes\spence.h`

```
/*
 * C++版本的spence函数，由SciPy开发者在2024年翻译。
 * 原始代码的版权信息和头部信息如下方所示。
 */

/*
 * spence.c
 * 对数积分函数
 *
 * SYNOPSIS:
 * double x, y, spence();
 * y = spence( x );
 *
 * DESCRIPTION:
 * 计算积分
 *             x
 *             -
 *            | | log t
 * spence(x) = |   ----- dt
 *           | |   t - 1
 *             -
 *             1
 * 当 x >= 0 时有效。在区间(0.5, 1.5)使用有理逼近计算积分。
 * 对于 1/x 和 1-x，使用转换公式处理超出基本展开范围的情况。
 *
 * ACCURACY:
 * 相对误差:
 *    IEEE      0,4         30000       3.9e-15     5.4e-16
 */

/*
 * Cephes Math Library Release 2.1:  January, 1989
 * 版权所有 1985, 1987, 1989 Stephen L. Moshier
 * 直接查询地址: 30 Frost Street, Cambridge, MA 02140
 */

#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "polevl.h"

namespace special {
namespace cephes {

    namespace detail {

        constexpr double spence_A[8] = {
            4.65128586073990045278E-5, 7.31589045238094711071E-3, 1.33847639578309018650E-1, 8.79691311754530315341E-1,
            2.71149851196553469920E0,  4.25697156008121755724E0,  3.29771340985225106936E0,  1.00000000000000000126E0,
        };

        constexpr double spence_B[8] = {
            6.90990488912553276999E-4, 2.54043763932544379113E-2, 2.82974860602568089943E-1, 1.41172597751831069617E0,
            3.63800533345137075418E0,  5.03278880143316990390E0,  3.54771340985225096217E0,  9.99999999999999998740E-1,
        };

    } // namespace detail

    // 计算对数积分函数 spence(x)
    SPECFUN_HOST_DEVICE inline double spence(double x) {
        double w, y, z;
        int flag;

        // 如果 x < 0，则设置错误并返回 NaN
        if (x < 0.0) {
            set_error("spence", SF_ERROR_DOMAIN, NULL);
            return (std::numeric_limits<double>::quiet_NaN());
        }

        // 如果 x == 1.0，则直接返回 0.0
        if (x == 1.0) {
            return (0.0);
        }

        // 如果 x == 0.0，则返回常数 M_PI * M_PI / 6.0
        if (x == 0.0) {
            return (M_PI * M_PI / 6.0);
        }

        flag = 0;

        // 如果 x > 2.0，则转换 x = 1.0 / x，同时标记使用了转换
        if (x > 2.0) {
            x = 1.0 / x;
            flag |= 2;
        }

        // 根据 x 的值进行不同情况的处理
        if (x > 1.5) {
            // 当 x > 1.5 时，计算 w = (1.0 / x) - 1.0，并标记使用了转换
            w = (1.0 / x) - 1.0;
            flag |= 2;
        } else if (x < 0.5) {
            // 当 x < 0.5 时，计算 w = -x，并标记使用了特殊情况
            w = -x;
            flag |= 1;
        } else {
            // 其他情况下，计算 w = x - 1.0
            w = x - 1.0;
        }

        // 使用 detail::spence_A 和 detail::spence_B 中的系数进行多项式逼近计算
        y = -w * polevl(w, detail::spence_A, 7) / polevl(w, detail::spence_B, 7);

        // 根据标志位进行修正
        if (flag & 1) {
            y = (M_PI * M_PI) / 6.0 - std::log(x) * std::log(1.0 - x) - y;
        }

        if (flag & 2) {
            z = std::log(x);
            y = -0.5 * z * z - y;
        }

        // 返回计算结果
        return (y);
    }

} // namespace cephes
} // namespace special
```