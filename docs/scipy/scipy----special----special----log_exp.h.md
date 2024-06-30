# `D:\src\scipysrc\scipy\scipy\special\special\log_exp.h`

```
#pragma once
// 防止头文件重复包含的预处理指令

#include <cmath>
// 包含数学函数库

#include "config.h"
// 包含自定义配置文件

namespace special {

template <typename T>
T expit(T x) {
    // 计算 logistic sigmoid 函数 expit(x) = 1 / (1 + exp(-x))
    return 1 / (1 + std::exp(-x));
};

inline double exprel(double x) {
    // 如果 x 的绝对值小于 double 类型的最小正数，则返回 1
    if (std::abs(x) < std::numeric_limits<double>::epsilon()) {
        return 1;
    }

    // 如果 x 大于 717（接近 log(DBL_MAX)），返回正无穷
    if (x > 717) {
        return std::numeric_limits<double>::infinity();
    }

    // 计算 exprel(x) = expm1(x) / x
    return std::expm1(x) / x;
}

inline float exprel(float x) { return exprel(static_cast<double>(x)); }
// exprel 函数的重载，用于处理 float 类型参数，转换为 double 处理

template <typename T>
T logit(T x) {
    // 计算 logit 函数，即 log(x / (1 - x))
    return std::log(x / (1 - x));
};

//
// The logistic sigmoid function 'expit' is
//
//     S(x) = 1/(1 + exp(-x))     = exp(x)/(exp(x) + 1)
//
// so
//
// log S(x) = -log(1 + exp(-x))   = x - log(exp(x) + 1)
//          = -log1p(exp(-x))     = x - log1p(exp(x))
//
// By using -log1p(exp(-x)) for x >= 0 and x - log1p(exp(x))
// for x < 0, we extend the range of x values for which we
// obtain accurate results (compared to the naive implementation
// log(expit(x))).
//
template <typename T>
T log_expit(T x) {
    // 如果 x 小于 0，使用 x - log1p(exp(x)) 的方式计算
    if (x < 0) {
        return x - std::log1p(std::exp(x));
    }

    // 否则使用 -log1p(exp(-x)) 的方式计算
    return -std::log1p(std::exp(-x));
};

} // namespace special
// 结束 special 命名空间声明
```