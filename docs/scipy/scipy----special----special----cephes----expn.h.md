# `D:\src\scipysrc\scipy\scipy\special\special\cephes\expn.h`

```
/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*                                                     expn.c
 *
 *             Exponential integral En
 *
 *
 *
 * SYNOPSIS:
 *
 * int n;
 * double x, y, expn();
 *
 * y = expn( n, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Evaluates the exponential integral
 *
 *                 inf.
 *                   -
 *                  | |   -xt
 *                  |    e
 *      E (x)  =    |    ----  dt.
 *       n          |      n
 *                | |     t
 *                 -
 *                  1
 *
 *
 * Both n and x must be nonnegative.
 *
 * The routine employs either a power series, a continued
 * fraction, or an asymptotic formula depending on the
 * relative values of n and x.
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0, 30       10000       1.7e-15     3.6e-16
 *
 */

/*                                                     expn.c  */

/* Cephes Math Library Release 1.1:  March, 1985
 * Copyright 1985 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140 */

/* Sources
 * [1] NIST, "The Digital Library of Mathematical Functions", dlmf.nist.gov
 */

/* Scipy changes:
 * - 09-10-2016: improved asymptotic expansion for large n
 */

#pragma once

#include "../config.h"
#include "../error.h"
#include "const.h"
#include "gamma.h"
#include "polevl.h"

namespace special {
namespace cephes {

    } // namespace detail

    }

} // namespace cephes
} // namespace special


注释：
这段代码是C++的头文件，用于计算指数积分En。它包含了描述函数用途、输入参数要求、精度以及实现背后的算法选择的详细信息。这些信息来自于Cephes数学库的文档和SciPy开发者的改进注释。
```