# `D:\src\scipysrc\scipy\scipy\special\special\cephes\besselpoly.h`

```
/* 被SciPy开发者于2024年翻译成C++的代码。
 *
 * 这部分不属于原始的cephes库。
 */
#pragma once

#include "../config.h"
#include "gamma.h"

namespace special {
namespace cephes {
    namespace detail {

        // 定义常量 besselpoly_EPS，用于控制迭代精度
        constexpr double besselpoly_EPS = 1.0e-17;
    }

    // 在特殊函数命名空间中定义 besselpoly 函数，计算贝塞尔多项式
    SPECFUN_HOST_DEVICE inline double besselpoly(double a, double lambda, double nu) {

        int m, factor = 0;
        double Sm, relerr, Sol;
        double sum = 0.0;

        // 对 a = 0.0 进行特殊处理
        if (a == 0.0) {
            if (nu == 0.0) {
                return 1.0 / (lambda + 1);
            } else {
                return 0.0;
            }
        }

        // 对负数和整数 nu 进行特殊处理
        if ((nu < 0) && (std::floor(nu) == nu)) {
            nu = -nu;
            factor = static_cast<int>(nu) % 2;
        }

        // 初始化 Sm，使用指数和伽玛函数计算贝塞尔多项式初始值
        Sm = std::exp(nu * std::log(a)) / (Gamma(nu + 1) * (lambda + nu + 1));
        m = 0;
        // 迭代计算贝塞尔多项式的各项直到满足精度要求或达到最大迭代次数
        do {
            sum += Sm;
            Sol = Sm;
            Sm *= -a * a * (lambda + nu + 1 + 2 * m) / ((nu + m + 1) * (m + 1) * (lambda + nu + 1 + 2 * m + 2));
            m++;
            relerr = std::abs((Sm - Sol) / Sm);
        } while (relerr > detail::besselpoly_EPS && m < 1000);

        // 根据 factor 的值返回最终计算结果，负数的情况需加负号
        if (!factor)
            return sum;
        else
            return -sum;
    }
} // namespace cephes
} // namespace special
```