# `D:\src\scipysrc\scipy\scipy\special\special\cephes\ellie.h`

```
/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987, 1993 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
/* Copyright 2014, Eric W. Moore */

/*     Incomplete elliptic integral of second kind     */
#pragma once

#include "../config.h"
#include "const.h"
#include "ellpe.h"
#include "unity.h"

namespace special {
namespace cephes {
    // 返回不完全第二类椭圆积分
    double ellie(double phi, double m) {
        double sn, cn, dn, E, M, temp, a, em, ellpk();
        int i, npio2, sign;

        if (m <= 0.0 || m > 1.0) {
            // 如果 m 不在有效范围内，则返回 NaN
            return (NAN);
        }

        if (phi > PIO2) {
            // 如果 phi 大于 PI/2，则返回 NaN
            return (NAN);
        }

        if (phi < -PIO2) {
            // 如果 phi 小于 -PI/2，则返回 NaN
            return (NAN);
        }

        if (m == 0.0) {
            // 如果 m 为 0，则返回 phi
            return (phi);
        }

        sign = 1;

        if (phi < 0.0) {
            // 如果 phi 小于 0，则取反号并记录
            phi = -phi;
            sign = -1;
        }

        npio2 = (int)(phi * M_1_PI + 0.5);
        a = phi - npio2 * PIO2;
        sn = sin(a);
        cn = cos(a);
        dn = 1.0 - m * sn * sn;
        em = 1.0 - m;
        M = 1.0;
        E = 1.0;

        for (i = 0; i < 50; i++) {
            temp = em;
            em += dn;
            M = 0.5 * (M + temp);
            dn = sqrt(temp * dn);
            temp = sn;
            sn = 0.5 * (sn + cn);
            cn = 0.5 * (temp - cn);
            temp = sn * ellpe(M);
            temp *= temp;
            E += temp;
            if (dn <= MACHEP) {
                // 当 dn 达到精度要求时退出循环
                break;
            }
        }

        temp = npio2;
        if (sign < 0) {
            // 如果 sign 小于 0，则取相反数
            temp = -temp;
        }
        temp += npio2 * E;
        // 返回计算结果
        return (temp);
    }

} // namespace cephes
} // namespace special
```