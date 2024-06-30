# `D:\src\scipysrc\scipy\scipy\special\special\zlog1.h`

```
/* Translated from Cython into C++ by SciPy developers in 2023.
 *
 * Original author: Josh Wilson, 2016.
 */

#pragma once

#include "config.h"

namespace special {
namespace detail {

    // 计算对数，特别关注在接近1时的精确性。我们自己实现这个函数，因为一些系统（特别是Travis CI机器）在这个区域内表现较弱。
    SPECFUN_HOST_DEVICE inline std::complex<double> zlog1(std::complex<double> z) {
        // 初始化系数和结果
        std::complex<double> coeff = -1.0;
        std::complex<double> res = 0.0;

        // 如果 z 接近于 1，使用自定义的近似计算方法
        if (std::abs(z - 1.0) > 0.1) {
            return std::log(z);
        }

        // 从 z 减去 1
        z -= 1.0;
        // 迭代求解级数展开直到足够精确或达到最大迭代次数
        for (int n = 1; n < 17; n++) {
            coeff *= -z;
            res += coeff / static_cast<double>(n);
            // 检查是否达到所需精度
            if (std::abs(res / coeff) < std::numeric_limits<double>::epsilon()) {
                break;
            }
        }
        // 返回计算结果
        return res;
    }
} // namespace detail
} // namespace special
```