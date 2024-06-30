# `D:\src\scipysrc\scipy\scipy\special\special\cephes\lanczos.h`

```
/*
 * 版权声明及许可协议信息，引用 Boost 软件许可证 Version 1.0。
 * 可以自由使用、修改和分发，详见 LICENSE_1_0.txt 或 https://www.boost.org/LICENSE_1_0.txt
 */

/*
 * lanczos.h 和 lanczos.c 基于 Boost 的 lanczos.hpp 而来
 *
 * Scipy 修改:
 * - 2016年6月22日: 移除所有与双精度无关的代码，并为在 Cephes 中使用而转换为 C 语言。
 *   注意，系数的顺序已反转，以匹配 polevl 的行为。
 */

/*
 * 每个 N 对应的最优 G 值及理论误差界限来自于
 * https://web.viu.ca/pughg/phdThesis/phdThesis.pdf
 *
 * 常数使用 Godfrey 描述的方法计算，
 * 参考 https://my.fit.edu/~gabdo/gamma.txt，并由 Toth 在 https://www.rskey.org/gamma.htm 使用 1000 位精度的 NTL::RR 进行详细说明。
 */

/*
 * N=13 时的 Lanczos 系数，对应 G=6.024680040776729583740234375
 * 最大实验误差（使用任意精度算术）为 1.196214e-17
 * 使用 Microsoft Visual C++ version 8.0 on Win32 编译于 2006年3月23日
 * 用于双精度计算。
 */

#pragma once

#include "../config.h" // 引入上级目录中的 config.h
#include "polevl.h" // 引入 polevl.h 中的函数和定义

namespace special {
namespace cephes {

    namespace detail {

    } // namespace detail

    constexpr double lanczos_g = 6.024680040776729583740234375; // Lanczos 方法中的常数 G
    SPECFUN_HOST_DEVICE double lanczos_sum_expg_scaled(double x) {
        return ratevl(x, detail::lanczos_sum_expg_scaled_num, 12, detail::lanczos_sum_expg_scaled_denom, 12);
        // 使用 ratevl 函数计算经过缩放的指数和的 Lanczos 近似求解
    }
} // namespace cephes
} // namespace special
```