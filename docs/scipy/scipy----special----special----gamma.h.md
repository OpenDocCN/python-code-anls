# `D:\src\scipysrc\scipy\scipy\special\special\gamma.h`

```
#pragma once

#include "cephes/gamma.h"  // 包含 Gamma 函数的头文件
#include "loggamma.h"      // 包含 loggamma 函数的头文件

namespace special {

template <typename T>
SPECFUN_HOST_DEVICE T gamma(T x) {
    return cephes::Gamma(x);  // 调用 Cephes 库中的 Gamma 函数计算 Gamma(x)
}

template <typename T>
SPECFUN_HOST_DEVICE T gammaln(T x) {
    return cephes::lgam(x);   // 调用 Cephes 库中的 lgam 函数计算 ln(Gamma(x))
}

SPECFUN_HOST_DEVICE inline std::complex<double> gamma(std::complex<double> z) {
    // 使用 loggamma 计算 Gamma(z)
    if (z.real() <= 0 && z == std::floor(z.real())) {
        // 处理极点情况，设置错误并返回 NaN
        set_error("gamma", SF_ERROR_SINGULAR, NULL);
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    return std::exp(loggamma(z));  // 返回 exp(loggamma(z))，即 Gamma(z)
}

SPECFUN_HOST_DEVICE inline std::complex<float> gamma(std::complex<float> z) {
    return static_cast<std::complex<float>>(gamma(static_cast<std::complex<double>>(z)));
    // 对于复数 z，调用上面定义的 double 类型版本的 gamma 函数，然后转换为 float 类型返回
}

} // namespace special
```