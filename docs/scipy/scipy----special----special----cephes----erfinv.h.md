# `D:\src\scipysrc\scipy\scipy\special\special\cephes\erfinv.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "ndtri.h"

namespace special {
namespace cephes {

    /*
     * Inverse of the error function.
     *
     * Computes the inverse of the error function on the restricted domain
     * -1 < y < 1. This restriction ensures the existence of a unique result
     * such that erf(erfinv(y)) = y.
     */
    SPECFUN_HOST_DEVICE inline double erfinv(double y) {
        // 定义域下界和上界
        constexpr double domain_lb = -1;
        constexpr double domain_ub = 1;

        // 阈值，用于判断是否使用泰勒展开
        constexpr double thresh = 1e-7;

        /*
         * For small arguments, use the Taylor expansion
         * erf(y) = 2/\sqrt{\pi} (y - y^3 / 3 + O(y^5)),    y\to 0
         * where we only retain the linear term.
         * Otherwise, y + 1 loses precision for |y| << 1.
         */
        if ((-thresh < y) && (y < thresh)) {
            // 对于较小的参数，使用泰勒展开的线性项
            return y / M_2_SQRTPI;
        }
        // 在定义域内使用逆正态分布函数
        if ((domain_lb < y) && (y < domain_ub)) {
            return ndtri(0.5 * (y + 1)) * M_SQRT1_2;
        } else if (y == domain_lb) {
            // 返回负无穷大
            return -std::numeric_limits<double>::infinity();
        } else if (y == domain_ub) {
            // 返回正无穷大
            return std::numeric_limits<double>::infinity();
        } else if (std::isnan(y)) {
            // 如果输入是 NaN，则设置错误并返回原始值
            set_error("erfinv", SF_ERROR_DOMAIN, NULL);
            return y;
        } else {
            // 其他情况设置错误并返回 NaN
            set_error("erfinv", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    /*
     * Inverse of the complementary error function.
     *
     * Computes the inverse of the complimentary error function on the restricted
     * domain 0 < y < 2. This restriction ensures the existence of a unique result
     * such that erfc(erfcinv(y)) = y.
     */
    SPECFUN_HOST_DEVICE inline double erfcinv(double y) {
        // 定义域下界和上界
        constexpr double domain_lb = 0;
        constexpr double domain_ub = 2;

        // 在定义域内使用逆正态分布函数
        if ((domain_lb < y) && (y < domain_ub)) {
            return -ndtri(0.5 * y) * M_SQRT1_2;
        } else if (y == domain_lb) {
            // 返回正无穷大
            return std::numeric_limits<double>::infinity();
        } else if (y == domain_ub) {
            // 返回负无穷大
            return -std::numeric_limits<double>::infinity();
        } else if (std::isnan(y)) {
            // 如果输入是 NaN，则设置错误并返回原始值
            set_error("erfcinv", SF_ERROR_DOMAIN, NULL);
            return y;
        } else {
            // 其他情况设置错误并返回 NaN
            set_error("erfcinv", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

} // namespace cephes
} // namespace special
```