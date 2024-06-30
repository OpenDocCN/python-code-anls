# `D:\src\scipysrc\scipy\scipy\special\special\par_cyl.h`

```
#pragma once
// 包含特殊函数库的头文件
#include "specfun/specfun.h"

namespace special {
namespace detail {

    // 声明一个模板函数 dvsa，计算小参数情况下的抛物柱函数 Dv(x)
    template <typename T>
    T dvsa(T x, T va) {

        // ===================================================
        // Purpose: Compute parabolic cylinder function Dv(x)
        //          for small argument
        // Input:   x  --- Argument
        //          va --- Order
        // Output:  PD --- Dv(x)
        // Routine called: GAMMA2 for computing Г(x)
        // ===================================================

        int m; // 循环计数器
        T ep, a0, va0, pd, ga0, g1, g0, r, vm, gm, r1, vt; // 定义临时变量

        const T pi = 3.141592653589793; // 圆周率常量
        const T eps = 1.0e-15; // 误差限
        const T sq2 = sqrt(2); // 根号2

        ep = exp(-0.25 * x * x); // 计算 exp(-0.25 * x^2)
        va0 = 0.5 * (1.0 - va); // 计算 0.5 * (1.0 - va)
        if (va == 0.0) { // 如果 va 等于 0
            pd = ep; // 结果为 ep
        } else {
            if (x == 0.0) { // 如果 x 等于 0
                if ((va0 <= 0.0) && (va0 == (int) va0)) { // 如果 va0 小于等于 0 且 va0 等于其整数部分
                    pd = 0.0; // 结果为 0.0
                } else {
                    ga0 = specfun::gamma2(va0); // 调用 gamma2 函数计算 Γ(va0)
                    pd = sqrt(pi) / (pow(2.0, -0.5 * va) * ga0); // 计算 sqrt(pi) / (2^(-0.5 * va) * Γ(va0))
                }
            } else {
                g1 = specfun::gamma2(-va); // 调用 gamma2 函数计算 Γ(-va)
                a0 = pow(2.0, -0.5 * va - 1.0) * ep / g1; // 计算 2^(-0.5 * va - 1.0) * ep / Γ(-va)
                vt = -0.5 * va; // 计算 -0.5 * va
                g0 = specfun::gamma2(vt); // 调用 gamma2 函数计算 Γ(-0.5 * va)
                pd = g0; // 将结果赋值给 pd
                r = 1.0; // 初始化 r 为 1.0
                for (m = 1; m <= 250; m++) { // 循环计算
                    vm = 0.5 * (m - va); // 计算 0.5 * (m - va)
                    gm = specfun::gamma2(vm); // 调用 gamma2 函数计算 Γ(vm)
                    r = -r * sq2 * x / m; // 更新 r 的值
                    r1 = gm * r; // 计算 gm * r
                    pd += r1; // 更新 pd 的值
                    if (fabs(r1) < fabs(pd) * eps) { // 如果 r1 的绝对值小于 pd 的绝对值乘以误差限
                        break; // 跳出循环
                    }
                }
                pd *= a0; // 最终结果乘以 a0
            }
        }
        return pd; // 返回计算结果 pd
    }

    // 继续定义下一个模板函数
    template <typename T>
    T dvla(T x, T va) {
        // ====================================================
        // Purpose: Compute parabolic cylinder functions Dv(x)
        //          for large argument
        // Input:   x  --- Argument
        //          va --- Order
        // Output:  PD --- Dv(x)
        // Routines called:
        //       (1) VVLA for computing Vv(x) for large |x|
        //       (2) GAMMA2 for computing Г(x)
        // ====================================================

        int k;
        T ep, a0, gl, pd, r, vl, x1;

        const T pi = 3.141592653589793;
        const T eps = 1.0e-12;
        
        // Compute exp(-.25 * x * x)
        ep = exp(-.25 * x * x);
        
        // Compute pow(fabs(x), va) * ep
        a0 = pow(fabs(x), va) * ep;
        
        r = 1.0;
        pd = 1.0;

        // Loop to compute Dv(x) using series expansion
        for (k = 1; k <= 16; k++) {
            r = -0.5 * r * (2.0 * k - va - 1.0) * (2.0 * k - va - 2.0) / (k * x * x);
            pd += r;
            // Check convergence criteria
            if (fabs(r / pd) < eps) {
                break;
            }
        }
        
        pd *= a0;

        // Adjust if x < 0.0 using Vv(x) and Г(x)
        if (x < 0.0) {
            x1 = -x;
            vl = vvla(x1, va);
            gl = specfun::gamma2(-va);
            pd = pi * vl / gl + cos(pi * va) * pd;
        }
        
        // Return the computed Dv(x)
        return pd;
    }

    template <typename T>
    T vvla(T x, T va) {
        // ===================================================
        // Purpose: Compute parabolic cylinder function Vv(x)
        //          for large argument
        // Input:   x  --- Argument
        //          va --- Order
        // Output:  PV --- Vv(x)
        // Routines called:
        //       (1) DVLA for computing Dv(x) for large |x|
        //       (2) GAMMA2 for computing Г(x)
        // ===================================================

        int k;
        T pv, qe, a0, r, x1, gl, dsl, pdl;

        const T pi = 3.141592653589793;
        const T eps = 1e-12;

        // Compute exp(0.25 * x * x)
        qe = exp(0.25 * x * x);
        
        // Compute pow(fabs(x), -va - 1) * sqrt(2.0 / pi) * qe
        a0 = pow(fabs(x), -va - 1) * sqrt(2.0 / pi) * qe;
        
        r = 1.0;
        pv = 1.0;

        // Loop to compute Vv(x) using series expansion
        for (k = 1; k <= 18; k++) {
            r = 0.5 * r * (2.0 * k + va - 1.0) * (2.0 * k + va) / (k * x * x);
            pv += r;
            // Check convergence criteria
            if (fabs(r / pv) < eps) {
                break;
            }
        }
        
        pv *= a0;

        // Adjust if x < 0.0 using Dv(x) and Г(x)
        if (x < 0.0) {
            x1 = -x;
            pdl = dvla(x1, va);
            gl = specfun::gamma2(-va);
            dsl = sin(pi * va) * sin(pi * va);
            pv = dsl * gl / pi * pdl - cos(pi * va) * pv;
        }
        
        // Return the computed Vv(x)
        return pv;
    }

    template <typename T>
    // ===================================================
    // Purpose: Compute parabolic cylinder function Vv(x)
    //          for small argument
    // Input:   x  --- Argument
    //          va --- Order
    // Output:  PV --- Vv(x)
    // Routine called : GAMMA2 for computing Г(x)
    // ===================================================
    T vvsa(T x, T va) {
        // 声明变量和系数
        T a0, fac, g1, ga0, gm, gw, r, r1, sq2, sv, sv0, v1, vb0, vm, pv;

        // 定义常量
        const T eps = 1.0e-15;
        const T pi = 3.141592653589793;
        
        // 计算指数函数的值
        const T ep = exp(-0.25 * x * x);
        // 计算 va0
        T va0 = 1.0 + 0.5 * va;

        // 处理特殊情况：x = 0.0
        if (x == 0.0) {
            // 判断 va0 是否满足条件
            if (((va0 <= 0.0) && (va0 == (int) va0)) || va == 0.0) {
                // 如果条件满足，设置 PV 为 0.0
                pv = 0.0;
            } else {
                // 否则计算 vb0 和 sv0
                vb0 = -0.5 * va;
                sv0 = sin(va0 * pi);
                // 计算 ga0
                ga0 = specfun::gamma2(va0);
                // 计算 PV
                pv = pow(2.0, vb0) * sv0 / ga0;
            }
        } else {
            // 当 x 不等于 0 时的计算过程
            sq2 = sqrt(2.0);
            // 计算 a0
            a0 = pow(2.0, -0.5 * va) * ep / (2.0 * pi);
            // 计算 sv
            sv = sin(-(va + 0.5) * pi);
            // 计算 v1
            v1 = -0.5 * va;
            // 调用 gamma2 计算 g1
            g1 = specfun::gamma2(v1);
            // 初始化 PV
            pv = (sv + 1.0) * g1;
            // 初始化 r 和 fac
            r = 1.0;
            fac = 1.0;

            // 开始迭代计算
            for (int m = 1; m <= 250; m++) {
                // 计算 vm 和 gm
                vm = 0.5 * (m - va);
                gm = specfun::gamma2(vm);
                // 更新 r 和 fac
                r = r * sq2 * x / m;
                fac = -fac;
                // 计算 gw
                gw = fac * sv + 1.0;
                // 计算 r1
                r1 = gw * r * gm;
                // 更新 PV
                pv += r1;
                // 检查收敛条件
                if ((fabs(r1 / pv) < eps) && (gw != 0.0)) {
                    break;
                }
            }
            // 最终乘以 a0 得到 PV
            pv *= a0;
        }
        // 返回计算结果 PV
        return pv;
    }
} // namespace detail

/*
 * 如果 x > 0，则返回 w1f 和 w1d。否则将 x = abs(x)，并返回 w2f 和 -w2d。
 */
template <typename T>
void pbwa(T a, T x, T &wf, T &wd) {
    int flag = 0;
    T w1f = 0.0, w1d = 0.0, w2f = 0.0, w2d = 0.0;

    if (x < -5 || x > 5 || a < -5 || a > 5) {
        /*
         * Zhang 和 Jin 的实现仅使用 Taylor 级数；
         * 在精度范围之外返回 NaN。
         */
        wf = std::numeric_limits<T>::quiet_NaN();
        wd = std::numeric_limits<T>::quiet_NaN();
        set_error("pbwa", SF_ERROR_LOSS, NULL);
    } else {
        if (x < 0) {
            x = -x;
            flag = 1;
        }
        detail::pbwa(a, x, &w1f, &w1d, &w2f, &w2d);
        if (flag) {
            wf = w2f;
            wd = -w2d;
        } else {
            wf = w1f;
            wd = w1d;
        }
    }
}

template <typename T>
void pbdv(T v, T x, T &pdf, T &pdd) {
    T *dv;
    T *dp;
    int num;

    if (isnan(v) || isnan(x)) {
        pdf = std::numeric_limits<T>::quiet_NaN();
        pdd = std::numeric_limits<T>::quiet_NaN();
    } else {
        /* 注意：在 specfun.f:PBDV 中 DV/DP 的索引从 0 开始，因此要加 2 */
        num = std::abs((int) v) + 2;
        dv = (T *) malloc(sizeof(T) * 2 * num);
        if (dv == NULL) {
            set_error("pbdv", SF_ERROR_OTHER, "memory allocation error");
            pdf = std::numeric_limits<T>::quiet_NaN();
            pdd = std::numeric_limits<T>::quiet_NaN();
        } else {
            dp = dv + num;
            detail::pbdv(x, v, dv, dp, &pdf, &pdd);
            free(dv);
        }
    }
}

template <typename T>
void pbvv(T v, T x, T &pvf, T &pvd) {
    T *vv;
    T *vp;
    int num;

    if (isnan(v) || isnan(x)) {
        pvf = std::numeric_limits<T>::quiet_NaN();
        pvd = std::numeric_limits<T>::quiet_NaN();
    } else {
        /* 注意：在 specfun.f:PBVV 中 VV/VP 的索引从 0 开始，因此要加 2 */
        num = std::abs((int) v) + 2;
        vv = (T *) malloc(sizeof(T) * 2 * num);
        if (vv == NULL) {
            set_error("pbvv", SF_ERROR_OTHER, "memory allocation error");
            pvf = std::numeric_limits<T>::quiet_NaN();
            pvd = std::numeric_limits<T>::quiet_NaN();
        } else {
            vp = vv + num;
            detail::pbvv(x, v, vv, vp, &pvf, &pvd);
            free(vv);
        }
    }
}

} // namespace special
```