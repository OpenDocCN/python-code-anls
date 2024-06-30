# `D:\src\scipysrc\scipy\scipy\special\special\mathieu.h`

```
#pragma once
// 声明特殊函数的命名空间
#include "specfun/specfun.h"

namespace special {

// 定义 Mathieu 函数的特征值函数
template <typename T>
T sem_cva(T m, T q);

// 定义 Mathieu 函数的函数 sem
template <typename T>
void sem(T m, T q, T x, T &csf, T &csd);

/* Mathieu functions */

// 定义 Mathieu 函数的特征值函数 cem_cva
// 参数 m: Mathieu 方程的参数 m
// 参数 q: Mathieu 方程的参数 q
template <typename T>
T cem_cva(T m, T q) {
    int int_m, kd = 1;

    // 如果 m 不是非负整数，设置错误并返回 NaN
    if ((m < 0) || (m != floor(m))) {
        set_error("cem_cva", SF_ERROR_DOMAIN, NULL);
        return std::numeric_limits<T>::quiet_NaN();
    }
    int_m = (int) m;

    // 如果 q 小于 0
    if (q < 0) {
        /* https://dlmf.nist.gov/28.2#E26 */
        // 如果 m 是偶数，返回 cem_cva(m, -q)
        if (int_m % 2 == 0) {
            return cem_cva(m, -q);
        } else {
            // 如果 m 是奇数，返回 sem_cva(m, -q)
            return sem_cva(m, -q);
        }
    }

    // 如果 m 是奇数，设置 kd 为 2
    if (int_m % 2) {
        kd = 2;
    }
    // 调用 specfun::cva2 函数计算特征值，返回结果
    return specfun::cva2(kd, int_m, q);
}

// 定义 Mathieu 函数的特征值函数 sem_cva
// 参数 m: Mathieu 方程的参数 m
// 参数 q: Mathieu 方程的参数 q
template <typename T>
T sem_cva(T m, T q) {
    int int_m, kd = 4;

    // 如果 m 小于等于 0 或者 m 不是非负整数，设置错误并返回 NaN
    if ((m <= 0) || (m != floor(m))) {
        set_error("cem_cva", SF_ERROR_DOMAIN, NULL);
        return std::numeric_limits<T>::quiet_NaN();
    }
    int_m = (int) m;

    // 如果 q 小于 0
    if (q < 0) {
        /* https://dlmf.nist.gov/28.2#E26 */
        // 如果 m 是偶数，返回 sem_cva(m, -q)
        if (int_m % 2 == 0) {
            return sem_cva(m, -q);
        } else {
            // 如果 m 是奇数，返回 cem_cva(m, -q)
            return cem_cva(m, -q);
        }
    }

    // 如果 m 是奇数，设置 kd 为 3
    if (int_m % 2) {
        kd = 3;
    }
    // 调用 specfun::cva2 函数计算特征值，返回结果
    return specfun::cva2(kd, int_m, q);
}

/* Mathieu functions */

// 定义 Mathieu 函数 cem
// 参数 m: Mathieu 方程的参数 m
// 参数 q: Mathieu 方程的参数 q
// 参数 x: 自变量 x
// 参数 csf: 返回的函数值 csf
// 参数 csd: 返回的导数值 csd
template <typename T>
void cem(T m, T q, T x, T &csf, T &csd) {
    int int_m, kf = 1, sgn;
    T f = 0.0, d = 0.0;

    // 如果 m 小于 0 或者 m 不是非负整数，设置 csf 和 csd 为 NaN，并设置错误
    if ((m < 0) || (m != floor(m))) {
        csf = std::numeric_limits<T>::quiet_NaN();
        csd = std::numeric_limits<T>::quiet_NaN();
        set_error("cem", SF_ERROR_DOMAIN, NULL);
    } else {
        int_m = (int) m;

        // 如果 q 小于 0
        if (q < 0) {
            /* https://dlmf.nist.gov/28.2#E34 */
            // 如果 m 是偶数
            if (int_m % 2 == 0) {
                sgn = ((int_m / 2) % 2 == 0) ? 1 : -1;
                // 递归调用 cem 函数
                cem(m, -q, 90 - x, f, d);
                // 设置 csf 和 csd 的值
                csf = sgn * f;
                csd = -sgn * d;
            } else {
                sgn = ((int_m / 2) % 2 == 0) ? 1 : -1;
                // 调用 sem 函数
                sem(m, -q, 90 - x, f, d);
                // 设置 csf 和 csd 的值
                csf = sgn * f;
                csd = -sgn * d;
            }
        } else {
            // 调用 specfun::mtu0 函数计算 Mathieu 函数的值
            specfun::mtu0(kf, int_m, q, x, &csf, &csd);
        }
    }
}

// 定义 Mathieu 函数 sem
// 参数 m: Mathieu 方程的参数 m
// 参数 q: Mathieu 方程的参数 q
// 参数 x: 自变量 x
// 参数 csf: 返回的函数值 csf
// 参数 csd: 返回的导数值 csd
template <typename T>
void sem(T m, T q, T x, T &csf, T &csd) {
    int int_m, kf = 2, sgn;
    T f = 0.0, d = 0.0;

    // 如果 m 小于 0 或者 m 不是非负整数，设置 csf 和 csd 为 NaN，并设置错误
    if ((m < 0) || (m != floor(m))) {
        csf = std::numeric_limits<T>::quiet_NaN();
        csd = std::numeric_limits<T>::quiet_NaN();
        set_error("sem", SF_ERROR_DOMAIN, NULL);
    } else {
        // 将浮点数 m 转换为整数 int_m
        int_m = (int) m;
        // 如果 int_m 为 0，则设置 csf 和 csd 为 0
        if (int_m == 0) {
            csf = 0;
            csd = 0;
        } else if (q < 0) {
            // 如果 q 小于 0，根据 int_m 的奇偶性确定 sgn 的值
            // 参考：https://dlmf.nist.gov/28.2#E34
            if (int_m % 2 == 0) {
                sgn = ((int_m / 2) % 2 == 0) ? -1 : 1;
                // 调用 sem 函数，并使用特定参数计算结果 csf 和 csd
                sem(m, -q, 90 - x, f, d);
                csf = sgn * f;
                csd = -sgn * d;
            } else {
                sgn = ((int_m / 2) % 2 == 0) ? 1 : -1;
                // 调用 cem 函数，并使用特定参数计算结果 csf 和 csd
                cem(m, -q, 90 - x, f, d);
                csf = sgn * f;
                csd = -sgn * d;
            }
        } else {
            // 当 q 不小于 0 时，调用 specfun::mtu0 函数计算结果 csf 和 csd
            specfun::mtu0(kf, int_m, q, x, &csf, &csd);
        }
    }
// 结束 special 命名空间

template <typename T>
void mcm1(T m, T q, T x, T &f1r, T &d1r) {
    // 定义整数变量 int_m，kf 和 kc 初始化为 1
    int int_m, kf = 1, kc = 1;
    // 定义浮点数变量 f2r 和 d2r，并初始化为 0.0

    if ((m < 0) || (m != floor(m)) || (q < 0)) {
        // 如果 m 小于 0 或者 m 不是整数或者 q 小于 0，则设置 f1r 和 d1r 为 NaN
        f1r = std::numeric_limits<T>::quiet_NaN();
        d1r = std::numeric_limits<T>::quiet_NaN();
        // 调用 set_error 函数，设置错误信息为 "mcm1"，错误域为 SF_ERROR_DOMAIN，没有额外数据
        set_error("mcm1", SF_ERROR_DOMAIN, NULL);
    } else {
        // 将 m 转换为整数并赋值给 int_m
        int_m = (int) m;
        // 调用 specfun::mtu12 函数，传递 kf、kc、int_m、q、x，以及 f1r、d1r、f2r、d2r 的指针
        specfun::mtu12(kf, kc, int_m, q, x, &f1r, &d1r, &f2r, &d2r);
    }
}

template <typename T>
void msm1(T m, T q, T x, T &f1r, T &d1r) {
    // 定义整数变量 int_m，kf 初始化为 2，kc 初始化为 1
    int int_m, kf = 2, kc = 1;
    // 定义浮点数变量 f2r 和 d2r，并初始化为 0.0

    if ((m < 1) || (m != floor(m)) || (q < 0)) {
        // 如果 m 小于 1 或者 m 不是整数或者 q 小于 0，则设置 f1r 和 d1r 为 NaN
        f1r = std::numeric_limits<T>::quiet_NaN();
        d1r = std::numeric_limits<T>::quiet_NaN();
        // 调用 set_error 函数，设置错误信息为 "msm1"，错误域为 SF_ERROR_DOMAIN，没有额外数据
        set_error("msm1", SF_ERROR_DOMAIN, NULL);
    } else {
        // 将 m 转换为整数并赋值给 int_m
        int_m = (int) m;
        // 调用 specfun::mtu12 函数，传递 kf、kc、int_m、q、x，以及 f1r、d1r、f2r、d2r 的指针
        specfun::mtu12(kf, kc, int_m, q, x, &f1r, &d1r, &f2r, &d2r);
    }
}

template <typename T>
void mcm2(T m, T q, T x, T &f2r, T &d2r) {
    // 定义整数变量 int_m，kf 初始化为 1，kc 初始化为 2
    int int_m, kf = 1, kc = 2;
    // 定义浮点数变量 f1r 和 d1r，并初始化为 0.0

    if ((m < 0) || (m != floor(m)) || (q < 0)) {
        // 如果 m 小于 0 或者 m 不是整数或者 q 小于 0，则设置 f2r 和 d2r 为 NaN
        f2r = std::numeric_limits<T>::quiet_NaN();
        d2r = std::numeric_limits<T>::quiet_NaN();
        // 调用 set_error 函数，设置错误信息为 "mcm2"，错误域为 SF_ERROR_DOMAIN，没有额外数据
        set_error("mcm2", SF_ERROR_DOMAIN, NULL);
    } else {
        // 将 m 转换为整数并赋值给 int_m
        int_m = (int) m;
        // 调用 specfun::mtu12 函数，传递 kf、kc、int_m、q、x，以及 f1r、d1r、f2r、d2r 的指针
        specfun::mtu12(kf, kc, int_m, q, x, &f1r, &d1r, &f2r, &d2r);
    }
}

template <typename T>
void msm2(T m, T q, T x, T &f2r, T &d2r) {
    // 定义整数变量 int_m，kf 和 kc 初始化为 2
    int int_m, kf = 2, kc = 2;
    // 定义浮点数变量 f1r 和 d1r，并初始化为 0.0

    if ((m < 1) || (m != floor(m)) || (q < 0)) {
        // 如果 m 小于 1 或者 m 不是整数或者 q 小于 0，则设置 f2r 和 d2r 为 NaN
        f2r = std::numeric_limits<T>::quiet_NaN();
        d2r = std::numeric_limits<T>::quiet_NaN();
        // 调用 set_error 函数，设置错误信息为 "msm2"，错误域为 SF_ERROR_DOMAIN，没有额外数据
        set_error("msm2", SF_ERROR_DOMAIN, NULL);
    } else {
        // 将 m 转换为整数并赋值给 int_m
        int_m = (int) m;
        // 调用 specfun::mtu12 函数，传递 kf、kc、int_m、q、x，以及 f1r、d1r、f2r、d2r 的指针
        specfun::mtu12(kf, kc, int_m, q, x, &f1r, &d1r, &f2r, &d2r);
    }
}
```