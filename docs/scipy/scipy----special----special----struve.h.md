# `D:\src\scipysrc\scipy\scipy\special\special\struve.h`

```
#pragma once

#include "config.h"

namespace special {
namespace detail {

    // ===========================================================
    // Purpose: Evaluate the integral H0(t)/t with respect to t
    //          from x to infinity
    // Input :  x   --- Lower limit  ( x ≥ 0 )
    // Output:  TTH --- Integration of H0(t)/t from x to infinity
    // ===========================================================
    inline double itth0(double x) {
        int k;
        double f0, g0, r, s, t, tth, tty, xt;
        const double pi = 3.141592653589793;

        // Initialize variables
        s = 1.0;
        r = 1.0;

        // Case when x < 24.5
        if (x < 24.5) {
            // Series expansion for the integral
            for (k = 1; k < 61; k++) {
                r = -r * x * x * (2.0 * k - 1.0) / pow(2.0 * k + 1.0, 3);
                s += r;
                // Break condition for series convergence
                if (fabs(r) < fabs(s) * 1.0e-12) {
                    break;
                }
            }
            // Calculate the integral
            tth = pi / 2.0 - 2.0 / pi * x * s;
        } else {
            // Series expansion for large x
            for (k = 1; k < 11; k++) {
                r = -r * pow(2.0 * k - 1.0, 3) / ((2.0 * k + 1.0) * x * x);
                s += r;
                // Break condition for series convergence
                if (fabs(r) < fabs(s) * 1.0e-12) {
                    break;
                }
            }
            // Calculate the integral
            tth = 2.0 / (pi * x) * s;
            // Additional terms for large x
            t = 8.0 / x;
            xt = x + 0.25 * pi;
            f0 = (((((0.18118e-2 * t - 0.91909e-2) * t + 0.017033) * t - 0.9394e-3) * t - 0.051445) * t - 0.11e-5) * t +
                 0.7978846;
            g0 =
                (((((-0.23731e-2 * t + 0.59842e-2) * t + 0.24437e-2) * t - 0.0233178) * t + 0.595e-4) * t + 0.1620695) *
                t;
            tty = (f0 * sin(xt) - g0 * cos(xt)) / (sqrt(x) * x);
            // Combine contributions from series and additional terms
            tth = tth + tty;
        }
        // Return the computed integral
        return tth;
    }
    // ===================================================
    // Purpose: Evaluate the integral of Struve function
    //          H0(t) with respect to t from 0 and x
    // Input :  x   --- Upper limit  ( x ≥ 0 )
    // Output:  TH0 --- Integration of H0(t) from 0 and x
    // ===================================================

    // 定义函数 itsh0，计算 Struve 函数 H0(t) 在区间 [0, x] 上的积分值
    inline double itsh0(double x) {

        int k;
        double a[25], a0, a1, af, bf, bg, r, rd, s, s0, th0, ty, xp;
        const double pi = 3.141592653589793;
        const double el = 0.57721566490153;

        // 初始条件设定
        r = 1.0;

        // 根据 x 的大小分别计算 Struve 函数 H0(t) 的积分
        if (x <= 30.0) {
            s = 0.5;
            // 使用迭代方法计算积分
            for (k = 1; k < 101; k++) {
                rd = 1.0;
                if (k == 1) {
                    rd = 0.5;
                }
                // 更新积分项
                r = -r * rd * k / (k + 1.0) * pow(x / (2.0 * k + 1.0), 2);
                s += r;
                // 判断迭代是否收敛
                if (fabs(r) < fabs(s) * 1.0e-12) {
                    break;
                }
            }
            // 计算积分结果
            th0 = 2.0 / pi * x * x * s;
        } else {
            s = 1.0;
            // 使用不同的迭代公式计算积分
            for (k = 1; k < 13; k++) {
                r = -r * k / (k + 1.0) * pow((2.0 * k + 1.0) / x, 2);
                s += r;
                // 判断迭代是否收敛
                if (fabs(r) < fabs(s) * 1.0e-12) {
                    break;
                }
            }
            // 计算积分结果的另一部分
            s0 = s / (pi * x * x) + 2.0 / pi * (log(2.0 * x) + el);
            a0 = 1.0;
            a1 = 5.0 / 8.0;
            a[0] = a1;
            // 计算系数数组 a[k]
            for (k = 1; k < 21; k++) {
                af = ((1.5 * (k + 0.5) * (k + 5.0 / 6.0) * a1 - 0.5 * (k + 0.5) * (k + 0.5) * (k - 0.5) * a0)) /
                     (k + 1.0);
                a[k] = af;
                a0 = a1;
                a1 = af;
            }
            bf = 1.0;
            r = 1.0;
            // 计算 bf 的值
            for (k = 1; k < 11; k++) {
                r = -r / (x * x);
                bf += a[2 * k - 1] * r;
            }
            bg = a[0] * x;
            r = 1.0 / x;
            // 计算 bg 的值
            for (k = 1; k < 10; k++) {
                r = -r / (x * x);
                bg += a[2 * k] * r;
            }
            // 计算 ty 的值
            xp = x + 0.25 * pi;
            ty = sqrt(2.0 / (pi * x)) * (bg * cos(xp) - bf * sin(xp));
            // 计算积分结果
            th0 = ty + s0;
        }
        // 返回积分结果
        return th0;
    }
    // ===========================================================
    // Purpose: Evaluate the integral of modified Struve function
    //          L0(t) with respect to t from 0 to x
    // Input :  x   --- Upper limit  ( x ≥ 0 )
    // Output:  TL0 --- Integration of L0(t) from 0 to x
    // ===========================================================

    // 声明整型变量 k 和双精度浮点数数组 a、a0、a1、af、r、rd、s、s0、ti、tl0
    // 声明常量 pi 和 el
    int k;
    double a[18], a0, a1, af, r, rd, s, s0, ti, tl0;
    const double pi = 3.141592653589793;
    const double el = 0.57721566490153;
    
    // 初始化 r 为 1.0
    r = 1.0;
    
    // 如果 x 小于等于 20.0，执行以下语句块
    if (x <= 20.0) {
        // 初始化 s 为 0.5
        s = 0.5;
        
        // 循环计算求和直到满足精度条件或达到循环上限
        for (k = 1; k < 101; k++) {
            // 初始化 rd 为 1.0，如果是第一次循环将 rd 设置为 0.5
            rd = 1.0;
            if (k == 1) {
                rd = 0.5;
            }
            // 更新 r 的值，进行累积乘法运算
            r = r * rd * k / (k + 1.0) * pow(x / (2.0 * k + 1.0), 2);
            // 更新 s 的值，进行累积加法运算
            s += r;
            // 如果相对误差小于给定精度则退出循环
            if (fabs(r / s) < 1.0e-12) {
                break;
            }
        }
        // 计算并赋值 tl0
        tl0 = 2.0 / pi * x * x * s;
    } else {
        // 如果 x 大于 20.0，执行以下语句块
        // 初始化 s 为 1.0
        s = 1.0;
        
        // 循环计算求和直到满足精度条件或达到循环上限
        for (k = 1; k < 11; k++) {
            // 更新 r 的值，进行累积乘法运算
            r = r * k / (k + 1.0) * pow((2.0 * k + 1.0) / x, 2);
            // 更新 s 的值，进行累积加法运算
            s += r;
            // 如果相对误差小于给定精度则退出循环
            if (fabs(r / s) < 1.0e-12) {
                break;
            }
        }
        
        // 计算 s0 的值
        s0 = -s / (pi * x * x) + 2.0 / pi * (log(2.0 * x) + el);
        
        // 初始化 a0 为 1.0，a1 为 5.0 / 8.0
        a0 = 1.0;
        a1 = 5.0 / 8.0;
        // 初始化数组 a 的第一个元素为 a1
        a[0] = a1;
        
        // 循环计算数组 a 中的元素值
        for (k = 1; k < 11; k++) {
            // 计算 af 的值
            af = ((1.5 * (k + .50) * (k + 5.0 / 6.0) * a1 - 0.5 * pow(k + 0.5, 2) * (k - 0.5) * a0)) / (k + 1.0);
            // 将 af 赋值给数组 a 的当前元素
            a[k] = af;
            // 更新 a0 和 a1 的值
            a0 = a1;
            a1 = af;
        }
        
        // 初始化 ti 为 1.0，r 为 1.0
        ti = 1.0;
        r = 1.0;
        
        // 循环计算 ti 的值
        for (k = 1; k < 11; k++) {
            // 更新 r 的值
            r = r / x;
            // 更新 ti 的值
            ti += a[k - 1] * r;
        }
        
        // 计算并赋值 tl0
        tl0 = ti / sqrt(2 * pi * x) * exp(x) + s0;
    }
    
    // 返回计算结果 tl0
    return tl0;
}
} // namespace detail

// 结束 detail 命名空间的定义


template <typename T>
T itstruve0(T x) {
    if (x < 0) {
        x = -x;
    }

    T out = detail::itsh0(x);
    SPECFUN_CONVINF("itstruve0", out);
    return out;
}

// 计算第一类斯特鲁夫函数 \( \mathrm{H}_0(x) \)，如果输入 \( x \) 是负数，取其绝对值后计算
// 输出计算结果，并使用宏 `SPECFUN_CONVINF` 进行转换和信息记录


template <typename T>
T it2struve0(T x) {
    int flag = 0;

    if (x < 0) {
        x = -x;
        flag = 1;
    }

    T out = detail::itth0(x);
    SPECFUN_CONVINF("it2struve0", out);
    if (flag) {
        out = M_PI - out;
    }
    return out;
}

// 计算第二类斯特鲁夫函数 \( \mathrm{L}_0(x) \)，如果输入 \( x \) 是负数，取其绝对值后计算
// 输出计算结果，并使用宏 `SPECFUN_CONVINF` 进行转换和信息记录；如果输入是负数，则进行调整以符合定义域


template <typename T>
T itmodstruve0(T x) {
    if (x < 0) {
        x = -x;
    }

    T out = detail::itsl0(x);
    SPECFUN_CONVINF("itmodstruve0", out);
    return out;
}

} // namespace special

// 计算修正的第一类斯特鲁夫函数 \( \mathrm{L}_0(x) \)，如果输入 \( x \) 是负数，取其绝对值后计算
// 输出计算结果，并使用宏 `SPECFUN_CONVINF` 进行转换和信息记录
// 结束 special 命名空间的定义
```