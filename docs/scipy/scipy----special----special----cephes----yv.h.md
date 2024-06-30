# `D:\src\scipysrc\scipy\scipy\special\special\cephes\yv.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * Cephes Math Library Release 2.8: June, 2000
 * Copyright 1984, 1987, 2000 by Stephen L. Moshier
 */
#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "jv.h"
#include "yn.h"

namespace special {
namespace cephes {

    /*
     * Bessel function of noninteger order
     */
    SPECFUN_HOST_DEVICE inline double yv(double v, double x) {
        double y, t;
        int n;

        // 将参数 v 转换为整数 n
        n = v;
        // 如果 v 转换后与原来相等，则调用整数阶贝塞尔函数 yn
        if (n == v) {
            y = yn(n, x);
            return (y);
        } else if (v == std::floor(v)) {
            /* 分母为零的情况 */
            // 设置错误并返回 NaN
            set_error("yv", SF_ERROR_DOMAIN, NULL);
            return std::numeric_limits<double>::quiet_NaN();
        }

        // 计算常量 M_PI 乘以 v
        t = M_PI * v;
        // 计算贝塞尔函数 Jv(v, x) 的余弦与 Jv(-v, x) 之差除以正弦 t
        y = (std::cos(t) * jv(v, x) - jv(-v, x)) / std::sin(t);

        // 如果 y 是无穷大
        if (std::isinf(y)) {
            if (v > 0) {
                // 如果 v 大于 0，设置溢出错误并返回负无穷
                set_error("yv", SF_ERROR_OVERFLOW, NULL);
                return -std::numeric_limits<double>::infinity();
            } else if (v < -1e10) {
                /* 这里数值上无法确定是 +inf 还是 -inf */
                // 设置域错误并返回 NaN
                set_error("yv", SF_ERROR_DOMAIN, NULL);
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        return (y);
    }
} // namespace cephes
} // namespace special
```