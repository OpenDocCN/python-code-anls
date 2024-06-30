# `D:\src\scipysrc\scipy\scipy\special\special\cephes\struve.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * Compute the Struve function.
 *
 * Notes
 * -----
 *
 * We use three expansions for the Struve function discussed in [1]:
 *
 * - power series
 *   Expansion using a series approach to compute the Struve function.
 *
 * - expansion in Bessel functions
 *   Another method using Bessel functions to approximate the Struve function.
 *
 * - asymptotic large-z expansion
 *   A third method employing asymptotic expansion for large z values.
 *
 * Rounding errors are estimated based on the largest terms in the sums.
 *
 * ``struve_convergence.py`` plots the convergence regions of the different
 * expansions.
 *
 * (i)
 *
 * Looking at the error in the asymptotic expansion, one finds that
 * it's not worth trying if z ~> 0.7 * v + 12 for v > 0.
 *
 * (ii)
 *
 * The Bessel function expansion tends to fail for |z| >~ |v| and is not tried
 * there.
 *
 * For Struve H it covers the quadrant v > z where the power series may fail to
 * produce reasonable results.
 *
 * (iii)
 *
 * The three expansions together cover for Struve H the region z > 0, v real.
 *
 * They also cover Struve L, except that some loss of precision may occur around
 * the transition region z ~ 0.7 |v|, v < 0, |v| >> 1 where the function changes
 * rapidly.
 *
 * (iv)
 *
 * The power series is evaluated in double-double precision. This fixes accuracy
 * issues in Struve H for |v| << |z| before the asymptotic expansion kicks in.
 * Moreover, it improves the Struve L behavior for negative v.
 *
 *
 * References
 * ----------
 * [1] NIST Digital Library of Mathematical Functions
 *     https://dlmf.nist.gov/11
 */
/*
 * 版权所有（C）2013年 Pauli Virtanen
 *
 * 源代码和二进制形式的重新分发，无论是否进行修改，均被允许，前提是满足以下条件：
 *
 * a. 必须保留上述版权声明、此条件列表和以下免责声明。
 * b. 在二进制形式的重新分发中，必须在提供的文档和/或其他材料中复制上述版权声明、
 *    此条件列表和以下免责声明。
 * c. 在未经特定书面许可的情况下，不得使用 Enthought 或 SciPy Developers 的名称
 *    来认可或推广基于本软件的产品。
 *
 * 本软件由版权持有者和贡献者“按原样”提供，包括但不限于适销性和特定用途的默示保证，
 * 均不承担任何责任。无论在何种情况下，无论是合同责任、严格责任还是侵权行为（包括疏忽
 * 或其他），均不承担任何直接、间接、附带、特殊、惩罚性或间接损害赔偿责任，即使已告知
 * 可能性。
 */
#pragma once

#include "../bessel.h"  // 引入相关头文件 bessel.h
#include "../config.h"  // 引入相关头文件 config.h
#include "../error.h"   // 引入相关头文件 error.h

#include "dd_real.h"    // 引入特定头文件 dd_real.h
#include "gamma.h"      // 引入特定头文件 gamma.h
#include "scipy_iv.h"   // 引入特定头文件 scipy_iv.h

namespace special {    // 命名空间 special

namespace cephes {     // 命名空间 cephes 在 special 命名空间内

    // 命名空间 detail

} // namespace detail

// 内联函数，计算 Struve 函数 H(v, z)，调用 detail::struve_hl(v, z, 1) 实现
SPECFUN_HOST_DEVICE inline double struve_h(double v, double z) { return detail::struve_hl(v, z, 1); }

// 内联函数，计算 Struve 函数 L(v, z)，调用 detail::struve_hl(v, z, 0) 实现
SPECFUN_HOST_DEVICE inline double struve_l(double v, double z) { return detail::struve_hl(v, z, 0); }

} // namespace cephes 在 special 命名空间内
} // namespace special
```