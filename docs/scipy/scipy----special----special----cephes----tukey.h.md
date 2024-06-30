# `D:\src\scipysrc\scipy\scipy\special\special\cephes\tukey.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/* Compute the CDF of the Tukey-Lambda distribution
 * using a bracketing search with special checks
 *
 * The PPF of the Tukey-lambda distribution is
 * G(p) = (p**lam + (1-p)**lam) / lam
 *
 * Author:  Travis Oliphant
 */

#pragma once

#include "../config.h"

namespace special {
namespace cephes {

    namespace detail {

        // 定义常量：小值阈值和精度阈值，最大迭代次数
        constexpr double tukey_SMALLVAL = 1e-4;
        constexpr double tukey_EPS = 1.0e-14;
        constexpr int tukey_MAXCOUNT = 60;

    } // namespace detail

    // Tukey-lambda分布的累积分布函数
    SPECFUN_HOST_DEVICE inline double tukeylambdacdf(double x, double lmbda) {
        double pmin, pmid, pmax, plow, phigh, xeval;
        int count;

        // 如果输入参数x或lmbda是NaN，则返回NaN
        if (std::isnan(x) || std::isnan(lmbda)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // 计算常量xeval
        xeval = 1.0 / lmbda;
        // 根据lmbda的正负情况处理边界情况
        if (lmbda > 0.0) {
            if (x <= (-xeval)) {
                return 0.0;
            }
            if (x >= xeval) {
                return 1.0;
            }
        }

        // 当lmbda接近0时的特殊情况处理
        if ((-detail::tukey_SMALLVAL < lmbda) && (lmbda < detail::tukey_SMALLVAL)) {
            if (x >= 0) {
                return 1.0 / (1.0 + std::exp(-x));
            } else {
                return std::exp(x) / (1.0 + std::exp(x));
            }
        }

        // 初始化二分搜索的初始参数
        pmin = 0.0;
        pmid = 0.5;
        pmax = 1.0;
        plow = pmin;
        phigh = pmax;
        count = 0;

        // 进行二分搜索迭代
        while ((count < detail::tukey_MAXCOUNT) && (std::abs(pmid - plow) > detail::tukey_EPS)) {
            // 计算当前中值处的函数值
            xeval = (std::pow(pmid, lmbda) - std::pow(1.0 - pmid, lmbda)) / lmbda;
            // 如果函数值与目标值x相等，则返回中值pmid
            if (xeval == x) {
                return pmid;
            }
            // 根据函数值与目标值x的大小关系更新搜索区间
            if (xeval > x) {
                phigh = pmid;
                pmid = (pmid + plow) / 2.0;
            } else {
                plow = pmid;
                pmid = (pmid + phigh) / 2.0;
            }
            // 更新迭代次数
            count++;
        }
        // 返回最终计算得到的累积分布函数值pmid
        return pmid;
    }

} // namespace cephes
} // namespace special
```