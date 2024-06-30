# `D:\src\scipysrc\scipy\scipy\special\special\airy.h`

```
    // 计算系数初始化
    int k, km, km2, kmax;
    double ck[52], dk[52];
    // 取绝对值
    double xa, xq, xm, fx, r, gx, df, dg, sai, sad, sbi, sbd, xp1, xr2;
    // 变量初始化
    double xf, rp, xar, xe, xr1, xcs, xss, ssa, sda, ssb, sdb;

    // 定义精度常量和数学常数
    const double eps = 1.0e-15;
    const double pi = 3.141592653589793;
    const double c1 = 0.355028053887817;
    const double c2 = 0.258819403792807;
    const double sr3 = 1.732050807568877;

    // 设置 xm 的初值
    km2 = 0;
    xa = fabs(x);
    xq = sqrt(xa);
    xm = 8.0;
    if (x > 0.0)
        xm = 5.0;

    // 处理特殊情况 x == 0.0
    if (x == 0.0) {
        *ai = c1;
        *bi = sr3 * c1;
        *ad = -c2;
        *bd = sr3 * c2;
        return;
    }

    // 当 xa <= xm 时的计算
    fx = 1.0;
    r = 1.0;
    // 计算 Ai(x) 的级数展开
    for (k = 1; k <= 40; k++) {
        r = r * x / (3.0 * k) * x / (3.0 * k - 1.0) * x;
        fx += r;
        // 判断级数是否收敛
        if (fabs(r) < fabs(fx) * eps)
            break;
    }

    gx = x;
    r = x;
    // 计算 Bi(x) 的级数展开
    for (k = 1; k <= 40; k++) {
        r = r * x / (3.0 * k) * x / (3.0 * k + 1.0) * x;
        gx += r;
        // 判断级数是否收敛
        if (fabs(r) < fabs(gx) * eps)
            break;
    }

    // 计算 Ai(x), Bi(x), Ai'(x), Bi'(x) 的值
    *ai = c1 * fx - c2 * gx;
    *bi = sr3 * (c1 * fx + c2 * gx);

    df = 0.5 * x * x;
    r = df;
    // 计算 Ai'(x) 的级数展开
    for (k = 1; k <= 40; k++) {
        r = r * x / (3.0 * k) * x / (3.0 * k + 2.0) * x;
        df += r;
        // 判断级数是否收敛
        if (fabs(r) < fabs(df) * eps)
            break;
    }

    dg = 1.0;
    r = 1.0;
    // 计算 Bi'(x) 的级数展开
    for (k = 1; k <= 40; k++) {
        r = r * x / (3.0 * k) * x / (3.0 * k - 2.0) * x;
        dg += r;
        // 判断级数是否收敛
        if (fabs(r) < fabs(dg) * eps)
            break;
    }

    *ad = c1 * df - c2 * dg;
    *bd = sr3 * (c1 * df + c2 * dg);
    } else {
        // 计算截断点，使得在渐近展开中余项大小为 epsilon。X<0 分支需要尽快执行，以提高 AIRYZO 的效率
        km = (int) (24.5 - xa);
        if (xa < 6.0)
            km = 14;
        if (xa > 15.0)
            km = 10;

        if (x > 0.0) {
            kmax = km;
        } else {
            // 根据 Xa 的大小选择截断点，使得在渐近展开中余项大小为 epsilon
            if (xa > 70.0)
                km = 3;
            if (xa > 500.0)
                km = 2;
            if (xa > 1000.0)
                km = 1;

            // 为 Xa 大于特定值时重新设置截断点
            km2 = km;
            if (xa > 150.0)
                km2 = 1;
            if (xa > 3000.0)
                km2 = 0;
            kmax = 2 * km + 1;
        }
        // 计算 xe、xr1、xar 和 xf 的值
        xe = xa * xq / 1.5;
        xr1 = 1.0 / xe;
        xar = 1.0 / xq;
        xf = sqrt(xar);
        rp = 0.5641895835477563;
        r = 1.0;
        // 计算系数 ck 和 dk
        for (k = 1; k <= kmax; k++) {
            r = r * (6.0 * k - 1.0) / 216.0 * (6.0 * k - 3.0) / k * (6.0 * k - 5.0) / (2.0 * k - 1.0);
            ck[k - 1] = r;
            dk[k - 1] = -(6.0 * k + 1.0) / (6.0 * k - 1.0) * r;
        }

        if (x > 0.0) {
            // 计算 X > 0 的情况下的 sai、sad、sbi 和 sbd
            sai = 1.0;
            sad = 1.0;
            r = 1.0;
            for (k = 1; k <= km; k++) {
                r *= -xr1;
                sai += ck[k - 1] * r;
                sad += dk[k - 1] * r;
            }
            sbi = 1.0;
            sbd = 1.0;
            r = 1.0;
            for (k = 1; k <= km; k++) {
                r *= xr1;
                sbi += ck[k - 1] * r;
                sbd += dk[k - 1] * r;
            }
            xp1 = exp(-xe);
            *ai = 0.5 * rp * xf * xp1 * sai;
            *bi = rp * xf / xp1 * sbi;
            *ad = -0.5 * rp / xf * xp1 * sad;
            *bd = rp / xf / xp1 * sbd;
        } else {
            // 计算 X <= 0 的情况下的 ssa、sda、ssb 和 sdb
            xcs = cos(xe + pi / 4.0);
            xss = sin(xe + pi / 4.0);
            ssa = 1.0;
            sda = 1.0;
            r = 1.0;
            xr2 = 1.0 / (xe * xe);
            for (k = 1; k <= km; k++) {
                r *= -xr2;
                ssa += ck[2 * k - 1] * r;
                sda += dk[2 * k - 1] * r;
            }
            ssb = ck[0] * xr1;
            sdb = dk[0] * xr1;
            r = xr1;
            for (k = 1; k <= km2; k++) {
                r *= -xr2;
                ssb += ck[2 * k] * r;
                sdb += dk[2 * k] * r;
            }

            *ai = rp * xf * (xss * ssa - xcs * ssb);
            *bi = rp * xf * (xcs * ssa + xss * ssb);
            *ad = -rp / xf * (xcs * sda + xss * sdb);
            *bd = rp / xf * (xss * sda - xcs * sdb);
        }
    }
    return;
}



inline void airyzo(int nt, int kf, double *xa, double *xb, double *xc, double *xd) {
    // ========================================================
    // Purpose: Compute the first NT zeros of Airy functions
    //          Ai(x) and Ai'(x), a and a', and the associated
    //          values of Ai(a') and Ai'(a); and the first NT
    //          zeros of Airy functions Bi(x) and Bi'(x), b and
    //          b', and the associated values of Bi(b') and
    //          Bi'(b)
    // Input :  NT    --- Total number of zeros
    //          KF    --- Function code
    //                    KF=1 for Ai(x) and Ai'(x)
    //                    KF=2 for Bi(x) and Bi'(x)
    // Output:  XA(m) --- a, the m-th zero of Ai(x) or
    //                    b, the m-th zero of Bi(x)
    //          XB(m) --- a', the m-th zero of Ai'(x) or
    //                    b', the m-th zero of Bi'(x)
    //          XC(m) --- Ai(a') or Bi(b')
    //          XD(m) --- Ai'(a) or Bi'(b)
    //                    ( m --- Serial number of zeros )
    // Routine called: AIRYB for computing Airy functions and
    //                 their derivatives
    // =======================================================

    const double pi = 3.141592653589793;
    int i;
    double rt = 0.0, rt0, u = 0.0, u1 = 0.0, x, ai, bi, ad, bd, err;

    for (i = 1; i <= nt; ++i) {
        rt0 = 0.0;
        if (kf == 1) {
            u = 3.0 * pi * (4.0 * i - 1) / 8.0;
            u1 = 1 / (u * u);
        } else if (kf == 2) {
            if (i == 1) {
                rt0 = -1.17371;
            } else {
                u = 3.0 * pi * (4.0 * i - 3.0) / 8.0;
                u1 = 1 / (u * u);
            }
        }

        if (rt0 == 0) {
            // DLMF 9.9.18: Compute initial estimate of zero using asymptotic series
            rt0 = -pow(u * u, 1.0 / 3.0) *
                  (1.0 +
                   u1 * (5.0 / 48.0 + u1 * (-5.0 / 36.0 + u1 * (77125.0 / 82944.0 + u1 * (-108056875.0 / 6967296.0)))));
        }

        // Refine the estimate of the zero using Newton-Raphson iteration
        while (1) {
            x = rt0;
            airyb(x, &ai, &bi, &ad, &bd);

            if (kf == 1) {
                rt = rt0 - ai / ad;
            } else if (kf == 2) {
                rt = rt0 - bi / bd;
            }

            err = fabs((rt - rt0) / rt);
            if (err <= 1.0e-12) {
                break;  // Convergence criteria met, exit loop
            } else {
                rt0 = rt;  // Update the estimate for further iteration
            }
        }

        // Store the computed zeros and their associated values
        xa[i - 1] = rt;
        if (err > 1.0e-14) {
            airyb(rt, &ai, &bi, &ad, &bd);
        }

        if (kf == 1) {
            xd[i - 1] = ad;
        } else if (kf == 2) {
            xd[i - 1] = bd;
        }
    }



}



// End of the airyzo function
    # 循环处理每个索引 i，从 1 到 nt
    for (i = 1; i <= nt; ++i) {
        rt0 = 0.0;

        # 如果 kf 等于 1
        if (kf == 1) {
            # 如果当前索引 i 等于 1，设置 rt0 为固定值 -1.01879
            if (i == 1) {
                rt0 = -1.01879;
            } else {
                # 否则，根据公式计算变量 u 和 u1
                u = 3.0 * pi * (4.0 * i - 3.0) / 8.0;
                u1 = 1 / (u * u);
            }
        } else if (kf == 2) {
            # 如果 kf 等于 2
            if (i == 1) {
                # 如果当前索引 i 等于 1，设置 rt0 为固定值 -2.29444
                rt0 = -2.29444;
            } else {
                # 否则，根据公式计算变量 u 和 u1
                u = 3.0 * pi * (4.0 * i - 1.0) / 8.0;
                u1 = 1 / (u * u);
            }
        }

        # 如果 rt0 仍然为初始值 0
        if (rt0 == 0) {
            // DLMF 9.9.19
            # 根据 DLMF 9.9.19 计算 rt0 的值，使用多项式表达式
            rt0 = -pow(u * u, 1.0 / 3.0) *
                  (1.0 + u1 * (-7.0 / 48.0 +
                               u1 * (35.0 / 288.0 + u1 * (-181223.0 / 207360.0 + u1 * (18683371.0 / 1244160.0)))));
        }

        # 迭代计算直到满足误差条件
        while (1) {
            # 将当前的 rt0 存入 x
            x = rt0;
            # 调用 airyb 函数，计算 ai, bi, ad, bd
            airyb(x, &ai, &bi, &ad, &bd);

            # 根据 kf 的值更新 rt 的值
            if (kf == 1) {
                rt = rt0 - ad / (ai * x);
            } else if (kf == 2) {
                rt = rt0 - bd / (bi * x);
            }

            # 计算当前的相对误差 err
            err = fabs((rt - rt0) / rt);
            # 如果误差小于等于 1.0e-12，跳出循环
            if (err <= 1.0e-12) {
                break;
            } else {
                # 否则，更新 rt0 继续迭代
                rt0 = rt;
            }
        }
        
        # 将计算得到的 rt 存入数组 xb 的第 i-1 个位置
        xb[i - 1] = rt;

        # 如果误差大于 1.0e-14，再次调用 airyb 函数计算 ai, bi, ad, bd
        if (err > 1.0e-14) {
            airyb(rt, &ai, &bi, &ad, &bd);
        }

        # 根据 kf 的值更新数组 xc 的第 i-1 个位置
        if (kf == 1) {
            xc[i - 1] = ai;
        } else if (kf == 2) {
            xc[i - 1] = bi;
        }
    }
    # 函数返回，结束
    return;
# 定义了一个模板函数 airy，计算 Airy 函数的值及其导数。
# 对于复数 z，函数返回 ai, aip, bi, bip，分别是 Airy 函数的第一类、第一类导数、第二类和第二类导数。

template <typename T>
void airy(std::complex<T> z, std::complex<T> &ai, std::complex<T> &aip, std::complex<T> &bi, std::complex<T> &bip) {
    # 初始化 id 为 0，ierr 为 0，kode 为 1
    int id = 0;
    int ierr = 0;
    int kode = 1;
    int nz;

    # 计算第一类 Airy 函数 ai，并设置错误和 NaN 值
    ai = amos::airy(z, id, kode, &nz, &ierr);
    set_error_and_nan("airy:", ierr_to_sferr(nz, ierr), ai);

    # 设置 nz 为 0，计算第二类 Airy 函数 bi，并设置错误和 NaN 值
    nz = 0;
    bi = amos::biry(z, id, kode, &ierr);
    set_error_and_nan("airy:", ierr_to_sferr(nz, ierr), bi);

    # 将 id 设置为 1，计算第一类 Airy 函数的导数 aip，并设置错误和 NaN 值
    id = 1;
    aip = amos::airy(z, id, kode, &nz, &ierr);
    set_error_and_nan("airy:", ierr_to_sferr(nz, ierr), aip);

    # 再次将 nz 设置为 0，计算第二类 Airy 函数的导数 bip，并设置错误和 NaN 值
    nz = 0;
    bip = amos::biry(z, id, kode, &ierr);
    set_error_and_nan("airy:", ierr_to_sferr(nz, ierr), bip);
}

# 定义了一个模板函数 airye，计算指数缩放后的 Airy 函数值及其导数。
# 对于复数 z，函数返回 ai, aip, bi, bip，分别是指数缩放后的 Airy 函数的第一类、第一类导数、第二类和第二类导数。

template <typename T>
void airye(std::complex<T> z, std::complex<T> &ai, std::complex<T> &aip, std::complex<T> &bi, std::complex<T> &bip) {
    # 初始化 id 为 0，kode 为 2（表示指数缩放），nz 和 ierr
    int id = 0;
    int kode = 2; /* Exponential scaling */
    int nz, ierr;

    # 计算指数缩放后的第一类 Airy 函数 ai，并设置错误和 NaN 值
    ai = amos::airy(z, id, kode, &nz, &ierr);
    set_error_and_nan("airye:", ierr_to_sferr(nz, ierr), ai);

    # 将 nz 设置为 0，计算指数缩放后的第二类 Airy 函数 bi，并设置错误和 NaN 值
    nz = 0;
    bi = amos::biry(z, id, kode, &ierr);
    set_error_and_nan("airye:", ierr_to_sferr(nz, ierr), bi);

    # 将 id 设置为 1，计算指数缩放后的第一类 Airy 函数的导数 aip，并设置错误和 NaN 值
    id = 1;
    aip = amos::airy(z, id, kode, &nz, &ierr);
    set_error_and_nan("airye:", ierr_to_sferr(nz, ierr), aip);

    # 再次将 nz 设置为 0，计算指数缩放后的第二类 Airy 函数的导数 bip，并设置错误和 NaN 值
    nz = 0;
    bip = amos::biry(z, id, kode, &ierr);
    set_error_and_nan("airye:", ierr_to_sferr(nz, ierr), bip);
}

# 定义了一个模板函数 airye，处理实数类型 z 的 Airy 函数值及其导数。
# 对于实数 z，函数返回 ai, aip, bi, bip，分别是 Airy 函数的第一类、第一类导数、第二类和第二类导数。

template <typename T>
void airye(T z, T &ai, T &aip, T &bi, T &bip) {
    # 初始化 id 为 0，kode 为 2（表示指数缩放），nz 和 ierr
    int id = 0;
    int kode = 2; /* Exponential scaling */
    int nz, ierr;
    std::complex<T> cai, caip, cbi, cbip;

    # 设置复数部分为 NaN
    cai.real(NAN);
    cai.imag(NAN);
    cbi.real(NAN);
    cbi.imag(NAN);
    caip.real(NAN);
    caip.imag(NAN);
    cbip.real(NAN);
    cbip.real(NAN);

    # 如果 z 小于 0，则将 ai 设置为 NaN
    if (z < 0) {
        ai = NAN;
    } else {
        # 计算第一类 Airy 函数 cai，并设置错误和 NaN 值
        cai = amos::airy(z, id, kode, &nz, &ierr);
        set_error_and_nan("airye:", ierr_to_sferr(nz, ierr), cai);
        ai = std::real(cai);
    }

    # 将 nz 设置为 0，计算第二类 Airy 函数 cbi，并设置错误和 NaN 值
    nz = 0;
    cbi = amos::biry(z, id, kode, &ierr);
    set_error_and_nan("airye:", ierr_to_sferr(nz, ierr), cbi);
    bi = std::real(cbi);

    # 将 id 设置为 1，如果 z 小于 0，则将 aip 设置为 NaN
    id = 1;
    if (z < 0) {
        aip = NAN;
    } else {
        # 计算第一类 Airy 函数的导数 caip，并设置错误和 NaN 值
        caip = amos::airy(z, id, kode, &nz, &ierr);
        set_error_and_nan("airye:", ierr_to_sferr(nz, ierr), caip);
        aip = std::real(caip);
    }

    # 将 nz 设置为 0，计算第二类 Airy 函数的导数 cbip，并设置错误和 NaN 值
    nz = 0;
    cbip = amos::biry(z, id, kode, &ierr);
    set_error_and_nan("airye:", ierr_to_sferr(nz, ierr), cbip);
    bip = std::real(cbip);
}

# 定义了一个模板函数 airy，根据参数 x 计算 Airy 函数值及其导数。
# 对于实数 x，函数返回 ai, aip, bi, bip，分别是 Airy 函数的第一类、第一类导数、第二类和第二类导数。

template <typename T>
void airy(T x, T &ai, T &aip, T &bi, T &bip) {
    /* 对于小参数，使用 Cephes 库，因为速度稍快。
     * 对于大参数，使用 AMOS 库，因为更精确。
     */
    if (x < -10 || x > 10) {
        # 如果 x 大于 10 或小于 -10，则将 x 转换为复数，并计算 Airy 函数及其导数
        std::complex<T> zai, zaip, zbi, zbip;
        airy(std::complex(x), zai, zaip, zbi, zbip);
        ai = std::real(zai);
        aip = std::real(zaip);
        bi = std::real(zbi);
        bip = std::real(zbip);
    } else {
        # 对于其他情况，使用 Cephes 库计算 Airy 函数及其导数
        cephes::airy(x, &ai, &aip, &bi, &bip);
    }
}
void itairy(T x, T &apt, T &bpt, T &ant, T &bnt) {
    // 检查 x 的符号位
    bool x_signbit = std::signbit(x);
    // 如果 x 是负数，则取其相反数以便在函数调用中使用正数计算
    if (x_signbit) {
        x = -x;
    }

    // 调用 detail 命名空间中的 itairy 函数，计算 Airy 函数的特殊值
    detail::itairy(x, apt, bpt, ant, bnt);
    
    // 如果 x 是负数，需要对计算结果进行符号调整
    if (x_signbit) { /* negative limit -- switch signs and roles */
        // 交换 apt 和 ant 的值，并将它们的符号反转
        T tmp = apt;
        apt = -ant;
        ant = -tmp;

        // 交换 bpt 和 bnt 的值，并将它们的符号反转
        tmp = bpt;
        bpt = -bnt;
        bnt = -tmp;
    }
}

} // namespace special
```