# `D:\src\scipysrc\scipy\scipy\special\special\bessel.h`

```
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include "amos.h"
// 包含自定义头文件 amos.h，用于特殊函数计算

#include "cephes/jv.h"
// 包含 cephes 库中的 jv 头文件，提供贝塞尔函数计算

#include "cephes/scipy_iv.h"
// 包含 cephes 库中的 scipy_iv 头文件，提供修正的贝塞尔函数计算

#include "cephes/yv.h"
// 包含 cephes 库中的 yv 头文件，提供贝塞尔函数计算

#include "error.h"
// 包含自定义的错误处理头文件 error.h

#include "specfun.h"
// 包含自定义的特殊函数头文件 specfun.h

#include "trig.h"
// 包含自定义的三角函数头文件 trig.h

extern "C" double cephes_iv(double v, double x);
// 声明外部 C 函数 cephes_iv，用于计算修正的贝塞尔函数 I_v(x)

namespace special {
namespace detail {

    template <typename T>
    std::complex<T> rotate(std::complex<T> z, T v) {
        // 函数模板 rotate：将复数 z 绕原点旋转角度 v 乘 π
        T c = cospi(v);
        // 计算 cos(π*v)
        T s = sinpi(v);
        // 计算 sin(π*v)

        return {c * std::real(z) - s * std::imag(z), s * std::real(z) + c * std::imag(z)};
        // 返回旋转后的复数
    }

    template <typename T>
    std::complex<T> rotate_jy(std::complex<T> j, std::complex<T> y, T v) {
        // 函数模板 rotate_jy：将复数 j 和 y 绕原点旋转角度 v 乘 π
        T c = cospi(v);
        // 计算 cos(π*v)
        T s = sinpi(v);
        // 计算 sin(π*v)

        return c * j - s * y;
        // 返回旋转后的复数 j 和 y 的乘积
    }

    template <typename T>
    int reflect_jy(std::complex<T> *jy, T v) {
        // 函数模板 reflect_jy：根据 v 的整数性质反射复数 *jy
        /* NB: Y_v may be huge near negative integers -- so handle exact
         *     integers carefully
         */
        int i;
        if (v != floor(v))
            return 0;
        // 若 v 不是整数，则返回 0

        i = v - 16384.0 * floor(v / 16384.0);
        // 计算 i = v - 16384.0 * floor(v / 16384.0)

        if (i & 1) {
            *jy = -(*jy);
            // 如果 i 是奇数，反转复数 *jy 的符号
        }
        return 1;
        // 返回 1 表示操作成功
    }

    template <typename T>
    int reflect_i(std::complex<T> *ik, T v) {
        // 函数模板 reflect_i：根据 v 的整数性质反射复数 *ik
        if (v != floor(v)) {
            return 0;
            // 若 v 不是整数，则返回 0 表示失败
        }

        return 1; /* I is symmetric for integer v */
        // 对于整数 v，函数 I 对称，返回 1 表示成功
    }

    template <typename T>
    std::complex<T> rotate_i(std::complex<T> i, std::complex<T> k, T v) {
        // 函数模板 rotate_i：将复数 i 绕原点旋转角度 v 乘 π，并加上复数 k
        T s = std::sin(v * M_PI) * (2 / M_PI);
        // 计算 sin(v * π) * (2 / π)

        return i + s * k;
        // 返回旋转后的复数
    }

    template <typename T>
    void itika(T x, T *ti, T *tk) {
        // =======================================================
        // Purpose: Integrate modified Bessel functions I0(t) and
        //          K0(t) with respect to t from 0 to x
        // Input :  x  --- Upper limit of the integral  ( x ≥ 0 )
        // Output:  TI --- Integration of I0(t) from 0 to x
        //          TK --- Integration of K0(t) from 0 to x
        // =======================================================

        int k;
        T rc1, rc2, e0, b1, b2, rs, r, tw, x2;
        static const T a[10] = {
            0.625,
            1.0078125,
            2.5927734375,
            9.1868591308594,
            4.1567974090576e+1,
            2.2919635891914e+2,
            1.491504060477e+3,
            1.1192354495579e+4,
            9.515939374212e+4,
            9.0412425769041e+5
        };
        const T pi = 3.141592653589793;
        const T el = 0.5772156649015329;

        // Handle the special case when x is zero
        if (x == 0.0) {
            *ti = 0.0;
            *tk = 0.0;
            return;
        } else if (x < 20.0) {  // Use series expansion for small x
            x2 = x * x;
            *ti = 1.0;
            r = 1.0;
            // Iterate to compute the integral of I0(t)
            for (k = 1; k <= 50; k++) {
                r = 0.25 * r * (2 * k - 1.0) / (2 * k + 1.0) / (k * k) * x2;
                *ti += r;
                // Check convergence
                if (fabs(r / (*ti)) < 1.0e-12) {
                    break;
                }
            }
            *ti *= x;  // Scale by x
        } else {  // Use asymptotic expansion for large x
            x2 = 0.0;
            *ti = 1.0;
            // Iterate using precomputed coefficients for large x
            for (k = 1; k <= 10; k++) {
                r = r / x;
                *ti += a[k - 1] * r;
            }
            rc1 = 1.0 / sqrt(2.0 * pi * x);
            *ti = rc1 * exp(x) * (*ti);  // Compute using asymptotic formula
        }

        // Compute the integral of K0(t)
        if (x < 12.0) {  // Use series expansion for small x
            e0 = el + log(x / 2.0);
            b1 = 1.0 - e0;
            b2 = 0.0;
            rs = 0.0;
            r = 1.0;
            tw = 0.0;
            // Iterate to compute the integral of K0(t)
            for (k = 1; k <= 50; k++) {
                r = 0.25 * r * (2 * k - 1.0) / (2 * k + 1.0) / (k * k) * x2;
                b1 += r * (1.0 / (2 * k + 1) - e0);
                rs += 1.0 / k;
                b2 += r * rs;
                *tk = b1 + b2;
                // Check convergence
                if (fabs((*tk - tw) / (*tk)) < 1.0e-12) {
                    break;
                }
                tw = *tk;
            }
            *tk *= x;  // Scale by x
        } else {  // Use asymptotic expansion for large x
            *tk = 1.0;
            // Iterate using precomputed coefficients for large x
            for (k = 1; k <= 10; k++) {
                r = -r / x;
                *tk = *tk + a[k - 1] * r;
            }
            rc2 = sqrt(pi / (2.0 * x));
            *tk = pi / 2.0 - rc2 * (*tk) * exp(-x);  // Compute using asymptotic formula
        }
        return;
    }
    // 定义一个模板函数 itjya，计算特殊函数 J(x) 和 Y(x) 的近似值
    void itjya(T x, T *tj, T *ty) {
        int k;  // 定义整型变量 k，用于循环计数
        T a[18], a0, a1, af, bf, bg, r, r2, rc, rs, ty1, ty2, x2, xp;  // 声明多个变量用于计算特殊函数的近似值
        const T pi = 3.141592653589793;  // 定义常量 pi，代表圆周率
        const T el = 0.5772156649015329;  // 定义常量 el，代表欧拉常数
        const T eps = 1.0e-12;  // 定义常量 eps，用于控制近似精度
    
        // 如果 x 等于 0.0，直接返回 J(x) 和 Y(x) 均为 0.0
        if (x == 0.0) {
            *tj = 0.0;
            *ty = 0.0;
        } else if (x <= 20.0) {  // 如果 x 小于等于 20.0，使用近似方法计算 J(x) 和 Y(x)
            x2 = x * x;  // 计算 x 的平方
            *tj = x;  // 初值 J(x) 设为 x
            r = x;  // r 初值设为 x
            // 开始循环计算 J(x) 的近似值
            for (k = 1; k < 61; k++) {
                r = -0.25 * r * (2.0 * k - 1.0) / (2.0 * k + 1.0) / (k * k) * x2;  // 计算下一项近似值
                *tj += r;  // 累加得到 J(x) 的近似值
                if (fabs(r) < fabs(*tj) * eps) {  // 如果近似值足够小，则退出循环
                    break;
                }
            }
            // 计算 Y(x) 的第一部分近似值
            ty1 = (el + log(x / 2.0)) * (*tj);
            rs = 0.0;
            ty2 = 1.0;
            r = 1.0;
            // 开始循环计算 Y(x) 的第二部分近似值
            for (k = 1; k < 61; k++) {
                r = -0.25 * r * (2.0 * k - 1.0) / (2.0 * k + 1.0) / (k * k) * x2;  // 计算下一项近似值
                rs += 1.0 / k;
                r2 = r * (rs + 1.0 / (2.0 * k + 1.0));
                ty2 += r2;  // 累加得到 Y(x) 的第二部分近似值
                if (fabs(r2) < fabs(ty2) * eps) {  // 如果近似值足够小，则退出循环
                    break;
                }
            }
            *ty = (ty1 - x * ty2) * 2.0 / pi;  // 计算 Y(x) 的最终近似值
        } else {  // 如果 x 大于 20.0，使用另一种近似方法计算 J(x) 和 Y(x)
            a0 = 1.0;
            a1 = 5.0 / 8.0;
            a[0] = a1;
    
            // 计算系数 a[k] 的循环
            for (int k = 1; k <= 16; k++) {
                af = ((1.5 * (k + 0.5) * (k + 5.0 / 6.0) * a1 - 0.5 * (k + 0.5) * (k + 0.5) * (k - 0.5) * a0)) /
                     (k + 1.0);
                a[k] = af;  // 存储系数 a[k]
                a0 = a1;
                a1 = af;
            }
            bf = 1.0;
            r = 1.0;
            // 计算 J(x) 的另一部分近似值
            for (int k = 1; k <= 8; k++) {
                r = -r / (x * x);
                bf += a[2 * k - 1] * r;
            }
            bg = a[0] / x;
            r = 1.0 / x;
            // 计算 Y(x) 的另一部分近似值
            for (int k = 1; k <= 8; k++) {
                r = -1.0 / (x * x);
                bg += a[2 * k] * r;
            }
            xp = x + 0.25 * pi;
            rc = sqrt(2.0 / (pi * x));
            // 计算 J(x) 和 Y(x) 的最终近似值
            *tj = 1.0 - rc * (bf * cos(xp) + bg * sin(xp));
            *ty = rc * (bg * cos(xp) - bf * sin(xp));
        }
        return;  // 函数返回
    }
    
    template <typename T>
    void ittika(T x, T *tti, T *ttk) {
        // =========================================================
        // Purpose: Integrate [I0(t)-1]/t with respect to t from 0
        //          to x, and K0(t)/t with respect to t from x to ∞
        // Input :  x   --- Variable in the limits  ( x ≥ 0 )
        // Output:  TTI --- Integration of [I0(t)-1]/t from 0 to x
        //          TTK --- Integration of K0(t)/t from x to ∞
        // =========================================================
    
        int k;
        T b1, e0, r, r2, rc, rs;
        const T pi = 3.141592653589793;
        const T el = 0.5772156649015329;
        static const T c[8] = {1.625,           4.1328125,       1.45380859375,   6.553353881835,
                               3.6066157150269, 2.3448727161884, 1.7588273098916, 1.4950639538279};
    
        // Handle special case where x is exactly 0
        if (x == 0.0) {
            *tti = 0.0;
            *ttk = 1.0e+300;  // Set ttk to a large value
            return;
        }
    
        // Case where x < 40.0
        if (x < 40.0) {
            *tti = 1.0;
            r = 1.0;
            // Series expansion for small x
            for (int k = 2; k <= 50; k++) {
                r = 0.25 * r * (k - 1.0) / (k * k * k) * x * x;
                *tti += r;
                if (fabs(r / (*tti)) < 1.0e-12) {
                    break;
                }
            }
            *tti *= 0.125 * x * x;
        } else {
            *tti = 1.0;
            r = 1.0;
            // Series expansion for large x
            for (int k = 1; k <= 8; k++) {
                r = r / x;
                *tti += c[k - 1] * r;
            }
            rc = x * sqrt(2.0 * pi * x);
            *tti = (*tti) * exp(x) / rc;
        }
    
        // Case where x <= 12.0
        if (x <= 12.0) {
            e0 = (0.5 * log(x / 2.0) + el) * log(x / 2.0) + pi * pi / 24.0 + 0.5 * el * el;
            b1 = 1.5 - (el + log(x / 2.0));
            rs = 1.0;
            r = 1.0;
            // Series expansion for small x (again)
            for (k = 2; k <= 50; k++) {
                r = 0.25 * r * (k - 1.0) / (k * k * k) * x * x;
                rs += 1.0 / k;
                r2 = r * (rs + 1.0 / (2.0 * k) - (el + log(x / 2.0)));
                b1 += r2;
                if (fabs(r2 / b1) < 1.0e-12) {
                    break;
                }
            }
            *ttk = e0 - 0.125 * x * x * b1;
        } else {
            *ttk = 1.0;
            r = 1.0;
            // Series expansion for large x (again)
            for (k = 1; k <= 8; k++) {
                r = -r / x;
                *ttk += c[k - 1] * r;
            }
            rc = x * sqrt(2.0 / (pi * x));
            *ttk = (*ttk) * exp(-x) / rc;
        }
    
        return;
    }
} // namespace detail

/* Integrals of bessel functions */

/* int(j0(t),t=0..x) */
/* int(y0(t),t=0..x) */

template <typename T>
void it1j0y0(T x, T &j0int, T &y0int) {
    int flag = 0; // 标志位，用于标记是否对 x 进行了取负操作

    if (x < 0) {
        x = -x; // 如果 x 小于 0，则取其绝对值
        flag = 1; // 设置标志位为 1，表示进行了取负操作
    }
    detail::itjya(x, &j0int, &y0int); // 调用 detail 命名空间中的函数计算 j0int 和 y0int
    if (flag) {
        j0int = -j0int; // 如果进行了取负操作，则对 j0int 取负
        y0int = std::numeric_limits<T>::quiet_NaN(); /* domain error */ // 如果取负后，y0int 设为 NaN，表示域错误
    }
}

/* int((1-j0(t))/t,t=0..x) */
/* int(y0(t)/t,t=x..inf) */

template <typename T>
void it2j0y0(T x, T &j0int, T &y0int) {
    int flag = 0; // 标志位，用于标记是否对 x 进行了取负操作

    if (x < 0) {
        x = -x; // 如果 x 小于 0，则取其绝对值
        flag = 1; // 设置标志位为 1，表示进行了取负操作
    }
    detail::ittjya(x, &j0int, &y0int); // 调用 detail 命名空间中的函数计算 j0int 和 y0int
    if (flag) {
        y0int = std::numeric_limits<T>::quiet_NaN(); /* domain error */ // 如果进行了取负操作，则将 y0int 设为 NaN，表示域错误
    }
}

/* Integrals of modified bessel functions */

template <typename T>
void it1i0k0(T x, T &i0int, T &k0int) {
    int flag = 0; // 标志位，用于标记是否对 x 进行了取负操作

    if (x < 0) {
        x = -x; // 如果 x 小于 0，则取其绝对值
        flag = 1; // 设置标志位为 1，表示进行了取负操作
    }
    detail::itika(x, &i0int, &k0int); // 调用 detail 命名空间中的函数计算 i0int 和 k0int
    if (flag) {
        i0int = -i0int; // 如果进行了取负操作，则对 i0int 取负
        k0int = std::numeric_limits<T>::quiet_NaN(); /* domain error */ // 如果取负后，k0int 设为 NaN，表示域错误
    }
}

template <typename T>
void it2i0k0(T x, T &i0int, T &k0int) {
    int flag = 0; // 标志位，用于标记是否对 x 进行了取负操作

    if (x < 0) {
        x = -x; // 如果 x 小于 0，则取其绝对值
        flag = 1; // 设置标志位为 1，表示进行了取负操作
    }
    detail::ittika(x, &i0int, &k0int); // 调用 detail 命名空间中的函数计算 i0int 和 k0int
    if (flag) {
        k0int = std::numeric_limits<T>::quiet_NaN(); /* domain error */ // 如果进行了取负操作，则将 k0int 设为 NaN，表示域错误
    }
}

template <typename T, typename OutputVec1, typename OutputVec2>
void rctj(T x, int *nm, OutputVec1 rj, OutputVec2 dj) {

    // ========================================================
    // Purpose: Compute Riccati-Bessel functions of the first
    //          kind and their derivatives
    // Input:   x --- Argument of Riccati-Bessel function
    //          n --- Order of jn(x)  ( n = 0,1,2,... )
    // Output:  RJ(n) --- x·jn(x)
    //          DJ(n) --- [x·jn(x)]'
    //          NM --- Highest order computed
    // Routines called:
    //          MSTA1 and MSTA2 for computing the starting
    //          point for backward recurrence
    // ========================================================

    int n = rj.extent(0) - 1; // 计算输出向量 rj 的长度，并减去 1，得到 n

    int k, m;
    T cs, f, f0, f1, rj0, rj1;

    *nm = n; // 将 nm 设置为 n
    if (fabs(x) < 1.0e-100) { // 如果 x 的绝对值小于 1.0e-100
        for (int k = 0; k <= n; k++) {
            rj[k] = 0.0; // 将 rj 数组中的每个元素都设为 0
            dj[k] = 0.0; // 将 dj 数组中的每个元素都设为 0
        }
        dj[0] = 1.0; // 设置 dj 的第一个元素为 1.0
        return; // 返回，结束函数
    }
    rj[0] = sin(x); // 计算 rj 的第一个元素为 sin(x)
    rj[1] = rj[0] / x - cos(x); // 计算 rj 的第二个元素
    rj0 = rj[0]; // 保存 rj 的第一个元素
    rj1 = rj[1]; // 保存 rj 的第二个元素
    cs = 0.0; // 初始化 cs
    f = 0.0; // 初始化 f

    if (n >= 2) {
        m = specfun::msta1(x, 200); // 调用 specfun 命名空间中的函数计算 m
        if (m < n) {
            *nm = m; // 如果 m 小于 n，则将 nm 设置为 m
        } else {
            m = specfun::msta2(x, n, 15); // 否则调用 specfun 命名空间中的函数计算 m
        }

        f0 = 0.0; // 初始化 f0
        f1 = 1.0e-100; // 初始化 f1

        for (k = m; k >= 0; k--) {
            f = (2.0 * k + 3.0) * f1 / x - f0; // 计算 f
            if (k <= *nm) {
                rj[k] = f; // 将计算得到的 f 存入 rj 中
            }
            f0 = f1; // 更新 f0
            f1 = f; // 更新 f1
        }
        cs = (fabs(rj0) > fabs(rj1) ? rj0 / f : rj1 / f0); // 计算 cs
        for (k = 0; k <= *nm; k++) {
            rj[k] = cs * rj[k]; // 将 rj 中的每个元素乘以 cs
        }
    }
    # 计算并存储第一个导数项，即 cos(x)
    dj[0] = cos(x);
    # 计算并存储其余导数项，根据递推公式 -k * rj[k] / x + rj[k - 1]
    for (int k = 1; k <= *nm; k++) {
        dj[k] = -k * rj[k] / x + rj[k - 1];
    }
}

template <typename T, typename OutputVec1, typename OutputVec2>
void rctj(T x, OutputVec1 rj, OutputVec2 dj) {
    // 调用 rctj 函数，传入指针 nm、输出向量 rj 和 dj
    int nm;
    rctj(x, &nm, rj, dj);
}

template <typename T, typename OutputVec1, typename OutputVec2>
void rcty(T x, int *nm, OutputVec1 ry, OutputVec2 dy) {

    // ========================================================
    // Purpose: Compute Riccati-Bessel functions of the second
    //          kind and their derivatives
    // Input:   x --- Argument of Riccati-Bessel function
    //          n --- Order of yn(x)
    // Output:  RY(n) --- x·yn(x)
    //          DY(n) --- [x·yn(x)]'
    //          NM --- Highest order computed
    // ========================================================

    // 计算 ry 向量的长度
    int n = ry.extent(0) - 1;

    int k;
    T rf0, rf1, rf2;
    *nm = n;

    // 如果 x 的值非常小，直接设定 ry 和 dy 向量的值为极限值，并返回
    if (x < 1.0e-60) {
        for (k = 0; k <= n; k++) {
            ry[k] = -1.0e+300;
            dy[k] = 1.0e+300;
        }
        ry[0] = -1.0;
        dy[0] = 0.0;
        return;
    }

    // 计算 ry[0] 和 ry[1] 的值
    ry[0] = -cos(x);
    ry[1] = ry[0] / x - sin(x);
    rf0 = ry[0];
    rf1 = ry[1];

    // 使用递推关系计算 ry 中剩余元素的值
    for (k = 2; k <= n; k++) {
        rf2 = (2.0 * k - 1.0) * rf1 / x - rf0;
        if (fabs(rf2) > 1.0e+300) {
            break;
        }
        ry[k] = rf2;
        rf0 = rf1;
        rf1 = rf2;
    }

    // 更新 nm，计算 dy[0] 和 dy 中剩余元素的值
    *nm = k - 1;
    dy[0] = sin(x);
    for (k = 1; k <= *nm; k++) {
        dy[k] = -k * ry[k] / x + ry[k - 1];
    }
    return;
}

template <typename T, typename OutputVec1, typename OutputVec2>
void rcty(T x, OutputVec1 ry, OutputVec2 dy) {
    // 调用 rcty 函数，传入指针 nm、输出向量 ry 和 dy
    int nm;
    rcty(x, &nm, ry, dy);
}

inline std::complex<double> cyl_bessel_je(double v, std::complex<double> z) {
    // 初始化变量
    int n = 1;
    int kode = 2;
    int nz, ierr;
    int sign = 1;
    std::complex<double> cy_j, cy_y;

    // 将复数部分设为 NaN
    cy_j.real(NAN);
    cy_j.imag(NAN);
    cy_y.real(NAN);
    cy_y.imag(NAN);

    // 如果输入参数中包含 NaN，则返回 cy_j
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy_j;
    }

    // 如果 v 小于 0，修改 sign，并处理 cy_j
    if (v < 0) {
        v = -v;
        sign = -1;
    }

    // 计算 besj，设置错误信息，若 sign 为 -1 则处理 besy 和旋转
    nz = amos::besj(z, v, kode, n, &cy_j, &ierr);
    set_error_and_nan("jve:", ierr_to_sferr(nz, ierr), cy_j);
    if (sign == -1) {
        if (!detail::reflect_jy(&cy_j, v)) {
            nz = amos::besy(z, v, kode, n, &cy_y, &ierr);
            set_error_and_nan("jve(yve):", ierr_to_sferr(nz, ierr), cy_y);
            cy_j = detail::rotate_jy(cy_j, cy_y, v);
        }
    }
    return cy_j;
}

inline std::complex<float> cyl_bessel_je(float v, std::complex<float> x) {
    // 将浮点数 v 和复数 x 转换为双精度，再调用 cyl_bessel_je 函数
    return static_cast<std::complex<float>>(cyl_bessel_je(static_cast<double>(v), static_cast<std::complex<double>>(x))
    );
}

template <typename T>
T cyl_bessel_je(T v, T x) {
    // 如果 v 不是整数且 x 小于 0，则返回 T 类型的 NaN
    if (v != floor(v) && x < 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 返回 cyl_bessel_je 的实部
    return std::real(cyl_bessel_je(v, std::complex(x)));
}

inline std::complex<double> cyl_bessel_ye(double v, std::complex<double> z) {
    // 初始化变量
    int n = 1;
    int kode = 2;
    int nz, ierr;
    int sign = 1;
    std::complex<double> cy_y, cy_j;
    # 将 cy_j 和 cy_y 的实部和虚部设置为 NaN
    cy_j.real(NAN);
    cy_j.imag(NAN);
    cy_y.real(NAN);
    cy_y.imag(NAN);

    # 如果 v、z 的实部或虚部有任何一个是 NaN，则返回 cy_y
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy_y;
    }
    
    # 如果 v 小于 0，则将 v 取绝对值，并设置 sign 为 -1
    if (v < 0) {
        v = -v;
        sign = -1;
    }

    # 调用 amos 库中的 besy 函数计算贝塞尔函数 Y_v(z) 的值，将结果存入 cy_y，并返回 nz 值
    nz = amos::besy(z, v, kode, n, &cy_y, &ierr);
    
    # 根据 ierr 的值设定错误信息并将 cy_y 设为 NaN
    set_error_and_nan("yve:", ierr_to_sferr(nz, ierr), cy_y);

    # 如果 ierr 为 2，并且 z 的实部大于等于 0，虚部等于 0，则表示发生了溢出
    if (ierr == 2) {
        if (z.real() >= 0 && z.imag() == 0) {
            /* overflow */
            # 将 cy_y 的实部设为正无穷，虚部设为 0
            cy_y.real(INFINITY);
            cy_y.imag(0);
        }
    }

    # 如果 sign 等于 -1
    if (sign == -1) {
        # 如果 detail::reflect_jy 函数返回 false，计算贝塞尔函数 J_v(z) 的值，并将结果存入 cy_j
        if (!detail::reflect_jy(&cy_y, v)) {
            nz = amos::besj(z, v, kode, n, &cy_j, &ierr);
            # 根据 ierr 的值设定错误信息并将 cy_j 设为 NaN
            set_error_and_nan("yv(jv):", ierr_to_sferr(nz, ierr), cy_j);
            # 将 cy_y 和 cy_j 组合旋转得到新的 cy_y
            cy_y = detail::rotate_jy(cy_y, cy_j, -v);
        }
    }

    # 返回最终计算得到的 cy_y
    return cy_y;
}

// 计算复数第二类修正贝塞尔函数 Y_v(x) 的模板函数，返回 std::complex<float> 类型
inline std::complex<float> cyl_bessel_ye(float v, std::complex<float> x) {
    // 调用双精度版本的 cyl_bessel_ye，并将结果转换为单精度复数
    return static_cast<std::complex<float>>(cyl_bessel_ye(static_cast<double>(v), static_cast<std::complex<double>>(x))
    );
}

// 计算修正贝塞尔函数 Y_v(x) 的模板函数，返回 T 类型
template <typename T>
T cyl_bessel_ye(T v, T x) {
    // 如果 x 小于 0，则返回 NaN
    if (x < 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 调用双精度版本的 cyl_bessel_ye，并返回其实部
    return std::real(cyl_bessel_ye(v, std::complex(x)));
}

// 计算复数第一类修正贝塞尔函数 I_v(x) 的模板函数，返回 std::complex<double> 类型
inline std::complex<double> cyl_bessel_ie(double v, std::complex<double> z) {
    int n = 1;
    int kode = 2;
    int sign = 1;
    int nz, ierr;
    std::complex<double> cy, cy_k;

    cy.real(NAN); // 设置 cy 的实部为 NaN
    cy.imag(NAN); // 设置 cy 的虚部为 NaN
    cy_k.real(NAN); // 设置 cy_k 的实部为 NaN
    cy_k.imag(NAN); // 设置 cy_k 的虚部为 NaN

    // 如果 v 或 z 的实部或虚部是 NaN，则返回 cy
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy;
    }
    // 如果 v 小于 0，则取其绝对值，并设置 sign 为 -1
    if (v < 0) {
        v = -v;
        sign = -1;
    }
    // 调用 amos::besi 计算 I_v(z)，并返回 nz 和 ierr
    nz = amos::besi(z, v, kode, n, &cy, &ierr);
    // 设置错误并返回 NaN
    set_error_and_nan("ive:", ierr_to_sferr(nz, ierr), cy);

    // 如果 sign 为 -1
    if (sign == -1) {
        // 如果 detail::reflect_i 返回 false
        if (!detail::reflect_i(&cy, v)) {
            // 调用 amos::besk 计算 K_v(z)，并返回 nz 和 cy_k
            nz = amos::besk(z, v, kode, n, &cy_k, &ierr);
            // 设置错误并返回 NaN
            set_error_and_nan("ive(kv):", ierr_to_sferr(nz, ierr), cy_k);
            // 调整尺度以匹配 zbesi
            cy_k = detail::rotate(cy_k, -z.imag() / M_PI);
            // 如果 z 的实部大于 0
            if (z.real() > 0) {
                cy_k.real(cy_k.real() * exp(-2 * z.real()));
                cy_k.imag(cy_k.imag() * exp(-2 * z.real()));
            }
            // v -> -v，调用 detail::rotate_i
            cy = detail::rotate_i(cy, cy_k, v);
        }
    }

    return cy;
}

// 计算复数第一类修正贝塞尔函数 I_v(x) 的模板函数，返回 std::complex<float> 类型
inline std::complex<float> cyl_bessel_ie(float v, std::complex<float> x) {
    // 调用双精度版本的 cyl_bessel_ie，并将结果转换为单精度复数
    return static_cast<std::complex<float>>(cyl_bessel_ie(static_cast<double>(v), static_cast<std::complex<double>>(x))
    );
}

// 计算修正贝塞尔函数 K_v(x) 的模板函数，返回 std::complex<double> 类型
template <typename T>
T cyl_bessel_ie(T v, T x) {
    // 如果 v 不是整数且 x 小于 0，则返回 NaN
    if (v != floor(v) && x < 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 调用双精度版本的 cyl_bessel_ie，并返回其实部
    return std::real(cyl_bessel_ie(v, std::complex(x)));
}

// 计算复数第二类修正贝塞尔函数 K_v(x) 的模板函数，返回 std::complex<double> 类型
inline std::complex<double> cyl_bessel_ke(double v, std::complex<double> z) {
    std::complex<double> cy{NAN, NAN}; // 创建复数 cy，初始化实部和虚部为 NaN
    // 如果 v 或 z 的实部或虚部是 NaN，则返回 cy
    if (isnan(v) || isnan(std::real(z)) || isnan(std::imag(z))) {
        return cy;
    }

    // 如果 v 小于 0，则取其绝对值
    if (v < 0) {
        // K_v == K_{-v} 即使对于非整数 v
        v = -v;
    }

    int n = 1;
    int kode = 2;
    int ierr;
    // 调用 amos::besk 计算 K_v(z)，并返回 nz 和 cy
    int nz = amos::besk(z, v, kode, n, &cy, &ierr);
    // 设置错误并返回 NaN
    set_error_and_nan("kve:", ierr_to_sferr(nz, ierr), cy);
    // 如果 ierr 等于 2
    if (ierr == 2) {
        // 如果 z 的实部大于等于 0 并且 z 的虚部等于 0，则溢出
        if (std::real(z) >= 0 && std::imag(z) == 0) {
            // 设置 cy 为无穷大
            cy = INFINITY;
        }
    }

    return cy;
}

// 计算复数第二类修正贝塞尔函数 K_v(x) 的模板函数，返回 std::complex<float> 类型
inline std::complex<float> cyl_bessel_ke(float v, std::complex<float> x) {
    // 调用双精度版本的 cyl_bessel_ke，并将结果转换为单精度复数
    return static_cast<std::complex<float>>(cyl_bessel_ke(static_cast<double>(v), static_cast<std::complex<double>>(x))
    );
}

// 计算修正贝塞尔函数 K_v(x) 的模板函数，返回 T 类型
template <typename T>
T cyl_bessel_ke(T v, T x) {
    // 如果 x 小于 0，则返回 NaN
    if (x < 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 如果 x 等于 0，则返回正无穷
    if (x == 0) {
        return std::numeric_limits<T>::infinity();
    }

    // 调用双精度版本的 cyl_bessel_ke，并返回其实部
    return std::real(cyl_bessel_ke(v, std::complex(x)));
}
inline std::complex<double> cyl_hankel_1e(double v, std::complex<double> z) {
    // 初始化一些变量
    int n = 1;
    int kode = 2;
    int m = 1;
    int nz, ierr;
    int sign = 1;
    // 定义复数变量 cy，并将其实部和虚部置为 NaN
    std::complex<double> cy;
    cy.real(NAN);
    cy.imag(NAN);

    // 检查输入参数是否包含 NaN，如果有则返回初始值 cy
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy;
    }

    // 如果 v 小于 0，取其绝对值并修改 sign 标志
    if (v < 0) {
        v = -v;
        sign = -1;
    }

    // 调用外部函数 amos::besh 计算 Hankel 函数的值，返回非零结果码 nz 和错误码 ierr
    nz = amos::besh(z, v, kode, m, n, &cy, &ierr);
    // 根据 ierr 设置错误信息和 cy 的值
    set_error_and_nan("hankel1e:", ierr_to_sferr(nz, ierr), cy);

    // 如果 sign 为 -1，则对 cy 进行旋转变换
    if (sign == -1) {
        cy = detail::rotate(cy, v);
    }

    // 返回计算结果 cy
    return cy;
}

inline std::complex<float> cyl_hankel_1e(float v, std::complex<float> z) {
    // 调用双精度版本 cyl_hankel_1e，将结果转换为单精度复数并返回
    return static_cast<std::complex<float>>(cyl_hankel_1e(static_cast<double>(v), static_cast<std::complex<double>>(z)));
}

inline std::complex<double> cyl_hankel_2e(double v, std::complex<double> z) {
    // 初始化一些变量
    int n = 1;
    int kode = 2;
    int m = 2;
    int nz, ierr;
    int sign = 1;

    // 定义复数变量 cy，并将其实部和虚部置为 NaN
    std::complex<double> cy{NAN, NAN};

    // 检查输入参数是否包含 NaN，如果有则返回初始值 cy
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy;
    }

    // 如果 v 小于 0，取其绝对值并修改 sign 标志
    if (v < 0) {
        v = -v;
        sign = -1;
    }

    // 调用外部函数 amos::besh 计算 Hankel 函数的值，返回非零结果码 nz 和错误码 ierr
    nz = amos::besh(z, v, kode, m, n, &cy, &ierr);
    // 根据 ierr 设置错误信息和 cy 的值
    set_error_and_nan("hankel2e:", ierr_to_sferr(nz, ierr), cy);

    // 如果 sign 为 -1，则对 cy 进行旋转变换
    if (sign == -1) {
        cy = detail::rotate(cy, -v);
    }

    // 返回计算结果 cy
    return cy;
}

inline std::complex<float> cyl_hankel_2e(float v, std::complex<float> z) {
    // 调用双精度版本 cyl_hankel_2e，将结果转换为单精度复数并返回
    return static_cast<std::complex<float>>(cyl_hankel_2e(static_cast<double>(v), static_cast<std::complex<double>>(z)));
}

inline std::complex<double> cyl_bessel_j(double v, std::complex<double> z) {
    // 初始化一些变量
    int n = 1;
    int kode = 1;
    int nz, ierr;
    int sign = 1;
    // 定义复数变量 cy_j 和 cy_y，并将其实部和虚部置为 NaN
    std::complex<double> cy_j, cy_y;
    cy_j.real(NAN);
    cy_j.imag(NAN);
    cy_y.real(NAN);
    cy_y.imag(NAN);

    // 检查输入参数是否包含 NaN，如果有则返回 cy_j
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy_j;
    }

    // 如果 v 小于 0，取其绝对值并修改 sign 标志
    if (v < 0) {
        v = -v;
        sign = -1;
    }

    // 调用外部函数 amos::besj 计算第一类贝塞尔函数的值，返回非零结果码 nz 和错误码 ierr
    nz = amos::besj(z, v, kode, n, &cy_j, &ierr);
    // 根据 ierr 设置错误信息和 cy_j 的值
    set_error_and_nan("jv:", ierr_to_sferr(nz, ierr), cy_j);

    // 如果 ierr 等于 2，表示溢出，调用 cyl_bessel_je 函数重新计算 cy_j，并设置为无穷大
    if (ierr == 2) {
        /* overflow */
        cy_j = cyl_bessel_je(v, z);
        cy_j.real(cy_j.real() * INFINITY);
        cy_j.imag(cy_j.imag() * INFINITY);
    }

    // 如果 sign 为 -1，且 detail::reflect_jy 返回 false，计算第二类贝塞尔函数的值 cy_y，并进行旋转变换
    if (sign == -1) {
        if (!detail::reflect_jy(&cy_j, v)) {
            nz = amos::besy(z, v, kode, n, &cy_y, &ierr);
            set_error_and_nan("jv(yv):", ierr_to_sferr(nz, ierr), cy_y);
            cy_j = detail::rotate_jy(cy_j, cy_y, v);
        }
    }

    // 返回计算结果 cy_j
    return cy_j;
}

inline std::complex<float> cyl_bessel_j(float v, std::complex<float> x) {
    // 调用双精度版本 cyl_bessel_j，将结果转换为单精度复数并返回
    return static_cast<std::complex<float>>(cyl_bessel_j(static_cast<double>(v), static_cast<std::complex<double>>(x)));
}

template <typename T>
T cyl_bessel_j(T v, T x) {
    // 如果 v 不是整数且 x 小于 0，则设置错误并返回 NaN
    if (v != static_cast<int>(v) && x < 0) {
        set_error("jv", SF_ERROR_DOMAIN, NULL);
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 调用双精度版本 cyl_bessel_j，并返回结果
    std::complex<T> res = cyl_bessel_j(v, std::complex(x));
    # 如果实部不等于自身，说明结果为 NaN，可能是由于溢出导致的
    if (std::real(res) != std::real(res)) {
        # 如果出现 NaN，返回 cephes::jv(v, x) 的计算结果
        /* AMOS returned NaN, possibly due to overflow */
        return cephes::jv(v, x);
    }

    # 如果没有出现 NaN，返回结果的实部
    return std::real(res);
}

// 计算复数参数的圆柱贝塞尔函数 Y(v, z)
inline std::complex<double> cyl_bessel_y(double v, std::complex<double> z) {
    // 初始化变量
    int n = 1;
    int kode = 1;
    int nz, ierr;
    int sign = 1;
    std::complex<double> cy_y, cy_j;

    // 将结果初始化为 NaN
    cy_j.real(NAN);
    cy_j.imag(NAN);
    cy_y.real(NAN);
    cy_y.imag(NAN);

    // 检查参数是否包含 NaN，若是则直接返回 NaN
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy_y;
    }
    // 处理负数的 v
    if (v < 0) {
        v = -v;
        sign = -1;
    }

    // 处理 z 为零的情况
    if (z.real() == 0 && z.imag() == 0) {
        /* overflow */
        // 设置 cy_y 为负无穷，表示溢出
        cy_y.real(-INFINITY);
        cy_y.imag(0);
        // 设置错误信息
        set_error("yv", SF_ERROR_OVERFLOW, NULL);
    } else {
        // 调用 amos 库计算贝塞尔函数，返回结果存入 cy_y，并设置错误码
        nz = amos::besy(z, v, kode, n, &cy_y, &ierr);
        // 设置错误信息及处理 NaN
        set_error_and_nan("yv:", ierr_to_sferr(nz, ierr), cy_y);
        // 若 ierr 为 2，处理特殊情况
        if (ierr == 2) {
            if (z.real() >= 0 && z.imag() == 0) {
                /* overflow */
                // 设置 cy_y 为负无穷，表示溢出
                cy_y.real(-INFINITY);
                cy_y.imag(0);
            }
        }
    }

    // 若 sign 为 -1，处理反射及调用 amos 库计算贝塞尔函数
    if (sign == -1) {
        // 反射处理，调用 amos 库计算贝塞尔函数，并设置错误信息
        if (!detail::reflect_jy(&cy_y, v)) {
            nz = amos::besj(z, v, kode, n, &cy_j, &ierr);
            // 调用 amos 库计算贝塞尔函数，并设置错误信息
            set_error_and_nan("yv(jv):", ierr_to_sferr(nz, ierr), cy_j);
            // 进行旋转处理
            cy_y = detail::rotate_jy(cy_y, cy_j, -v);
        }
    }
    // 返回计算结果 cy_y
    return cy_y;
}

// 计算复数参数的圆柱贝塞尔函数 Y(v, x)，类型为 float
inline std::complex<float> cyl_bessel_y(float v, std::complex<float> x) {
    // 转换参数类型并调用双精度版本的函数
    return static_cast<std::complex<float>>(cyl_bessel_y(static_cast<double>(v), static_cast<std::complex<double>>(x)));
}

// 模板函数，计算圆柱贝塞尔函数 Y(v, x)，支持各种数值类型 T
template <typename T>
T cyl_bessel_y(T v, T x) {
    // 处理 x 小于 0 的情况
    if (x < 0) {
        // 设置错误信息
        set_error("yv", SF_ERROR_DOMAIN, nullptr);
        // 返回 NaN
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 计算圆柱贝塞尔函数 Y(v, x) 的结果，并处理返回值
    std::complex<T> res = cyl_bessel_y(v, std::complex(x));
    // 检查结果是否为实数
    if (std::real(res) != std::real(res)) {
        // 调用 cephes 库计算圆柱贝塞尔函数 Y(v, x) 的结果
        return cephes::yv(v, x);
    }

    // 返回实部结果
    return std::real(res);
}

// 计算双精度浮点数参数的圆柱贝塞尔函数 I(v, x)
inline double cyl_bessel_i(double v, double x) { return cephes::iv(v, x); }

// 计算单精度浮点数参数的圆柱贝塞尔函数 I(v, x)，转换为双精度后调用
inline float cyl_bessel_i(float v, float x) { return cyl_bessel_i(static_cast<double>(v), static_cast<double>(x)); }

// 计算复数参数的圆柱贝塞尔函数 I(v, z)
inline std::complex<double> cyl_bessel_i(double v, std::complex<double> z) {
    // 初始化变量
    int n = 1;
    int kode = 1;
    int sign = 1;
    int nz, ierr;
    std::complex<double> cy, cy_k;

    // 将结果初始化为 NaN
    cy.real(NAN);
    cy.imag(NAN);
    cy_k.real(NAN);
    cy_k.imag(NAN);

    // 检查参数是否包含 NaN，若是则直接返回 NaN
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy;
    }
    // 处理负数的 v
    if (v < 0) {
        v = -v;
        sign = -1;
    }
    // 调用 amos 库计算贝塞尔函数，返回结果存入 cy，并设置错误码
    nz = amos::besi(z, v, kode, n, &cy, &ierr);
    // 设置错误信息及处理 NaN
    set_error_and_nan("iv:", ierr_to_sferr(nz, ierr), cy);
    // 若 ierr 为 2，处理特殊情况
    if (ierr == 2) {
        /* overflow */
        // 处理溢出情况
        if (z.imag() == 0 && (z.real() >= 0 || v == floor(v))) {
            if (z.real() < 0 && v / 2 != floor(v / 2))
                cy.real(-INFINITY);
            else
                cy.real(INFINITY);
            cy.imag(0);
        } else {
            // 调用 cyl_bessel_ie 处理
            cy = cyl_bessel_ie(v * sign, z);
            cy.real(cy.real() * INFINITY);
            cy.imag(cy.imag() * INFINITY);
        }
    }
    // 如果 sign 等于 -1，则执行以下代码块
    if (sign == -1) {
        // 调用 detail 命名空间中的 reflect_i 函数，将 v 的值传入，如果返回 false，则执行以下代码块
        if (!detail::reflect_i(&cy, v)) {
            // 调用 amos 命名空间中的 besk 函数，传入 z, v, kode, n 以及指向 cy_k 和 ierr 的指针，将返回值赋给 nz
            nz = amos::besk(z, v, kode, n, &cy_k, &ierr);
            // 调用 set_error_and_nan 函数，设置错误信息，传入 "iv(kv):", ierr_to_sferr(nz, ierr) 的返回值和 cy_k
            set_error_and_nan("iv(kv):", ierr_to_sferr(nz, ierr), cy_k);
            // 调用 detail 命名空间中的 rotate_i 函数，传入 cy, cy_k, v，并将返回值赋给 cy
            cy = detail::rotate_i(cy, cy_k, v);
        }
    }

    // 返回 cy 变量的值
    return cy;
}

// 计算复数阶第一类汉克尔函数的贝塞尔函数，参数类型为 float
inline std::complex<float> cyl_bessel_i(float v, std::complex<float> x) {
    // 调用双精度版本的函数，将参数转换为双精度复数类型
    return static_cast<std::complex<float>>(cyl_bessel_i(static_cast<double>(v), static_cast<std::complex<double>>(x)));
}

// 计算复数阶第二类汉克尔函数的贝塞尔函数，参数类型为 double
inline std::complex<double> cyl_bessel_k(double v, std::complex<double> z) {
    // 初始化复数类型的结果
    std::complex<double> cy(NAN, NAN);
    // 如果输入包含 NaN，则返回初始值
    if (std::isnan(v) || std::isnan(std::real(z)) || isnan(std::imag(z))) {
        return cy;
    }

    // 如果阶数 v 是负数，则转换为其正数
    if (v < 0) {
        /* K_v == K_{-v} even for non-integer v */
        v = -v;
    }

    // 设置参数和返回值的初始化
    int n = 1;
    int kode = 1;
    int ierr;
    int nz = amos::besk(z, v, kode, n, &cy, &ierr);
    // 设置错误状态并返回 NaN 值
    set_error_and_nan("kv:", ierr_to_sferr(nz, ierr), cy);
    // 如果错误状态为 2
    if (ierr == 2) {
        // 如果实部大于等于 0 且虚部为 0，则表示溢出
        if (std::real(z) >= 0 && std::imag(z) == 0) {
            /* overflow */
            cy = INFINITY;
        }
    }

    // 返回计算结果
    return cy;
}

// 计算复数阶第二类汉克尔函数的贝塞尔函数，参数类型为 float
inline std::complex<float> cyl_bessel_k(float v, std::complex<float> x) {
    // 调用双精度版本的函数，将参数转换为双精度复数类型
    return static_cast<std::complex<float>>(cyl_bessel_k(static_cast<double>(v), static_cast<std::complex<double>>(x)));
}

// 通用模板函数，计算复数阶第二类汉克尔函数的贝塞尔函数
template <typename T>
T cyl_bessel_k(T v, T z) {
    // 如果 z 小于 0，则返回 quiet NaN
    if (z < 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 如果 z 等于 0，则返回正无穷大
    if (z == 0) {
        return std::numeric_limits<T>::infinity();
    }

    // 如果 z 大于 710 * (1 + |v|)，则返回 0
    if (z > 710 * (1 + std::abs(v))) {
        /* Underflow. See uniform expansion https://dlmf.nist.gov/10.41
         * This condition is not a strict bound (it can underflow earlier),
         * rather, we are here working around a restriction in AMOS.
         */
        return 0;
    }

    // 调用双精度版本的函数，计算结果的实部
    return std::real(cyl_bessel_k(v, std::complex(z)));
}

// 计算复数阶第一类汉克尔函数的汉克尔函数，参数类型为 double
inline std::complex<double> cyl_hankel_1(double v, std::complex<double> z) {
    // 初始化参数和返回值的相关变量
    int n = 1;
    int kode = 1;
    int m = 1;
    int nz, ierr;
    int sign = 1;
    std::complex<double> cy;

    // 设置初始值为 NaN
    cy.real(NAN);
    cy.imag(NAN);

    // 如果输入包含 NaN，则返回初始值
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy;
    }

    // 如果阶数 v 是负数，则转换为其正数
    if (v < 0) {
        v = -v;
        sign = -1;
    }

    // 调用 AMOS 库计算汉克尔函数，并获取相关错误码和结果
    nz = amos::besh(z, v, kode, m, n, &cy, &ierr);
    // 设置错误状态并返回 NaN 值
    set_error_and_nan("hankel1:", ierr_to_sferr(nz, ierr), cy);
    // 如果阶数为负，则对结果进行旋转变换
    if (sign == -1) {
        cy = detail::rotate(cy, v);
    }
    // 返回计算结果
    return cy;
}

// 计算复数阶第一类汉克尔函数的汉克尔函数，参数类型为 float
inline std::complex<float> cyl_hankel_1(float v, std::complex<float> z) {
    // 调用双精度版本的函数，将参数转换为双精度复数类型
    return static_cast<std::complex<float>>(cyl_hankel_1(static_cast<double>(v), static_cast<std::complex<double>>(z)));
}

// 计算复数阶第二类汉克尔函数的汉克尔函数，参数类型为 double
inline std::complex<double> cyl_hankel_2(double v, std::complex<double> z) {
    // 初始化参数和返回值的相关变量
    int n = 1;
    int kode = 1;
    int m = 2;
    int nz, ierr;
    int sign = 1;
    std::complex<double> cy;

    // 设置初始值为 NaN
    cy.real(NAN);
    cy.imag(NAN);

    // 如果输入包含 NaN，则返回初始值
    if (isnan(v) || isnan(z.real()) || isnan(z.imag())) {
        return cy;
    }

    // 如果阶数 v 是负数，则转换为其正数
    if (v < 0) {
        v = -v;
        sign = -1;
    }

    // 调用 AMOS 库计算汉克尔函数，并获取相关错误码和结果
    nz = amos::besh(z, v, kode, m, n, &cy, &ierr);
    // 设置错误状态并返回 NaN 值
    set_error_and_nan("hankel2:", ierr_to_sferr(nz, ierr), cy);
    // 如果阶数为负，则对结果进行旋转变换
    if (sign == -1) {
        cy = detail::rotate(cy, -v);
    }
    // 返回计算结果
    return cy;
}

// 计算复数阶第二类汉克尔函数的汉克尔函数，参数类型为 float
inline std::complex<float> cyl_hankel_2(float v, std::complex<float> z) {
    # 将变量 v 强制转换为 double 类型，并调用 cyl_hankel_2 函数计算修正第二类 Hankel 函数，参数是变量 z 强制转换为 std::complex<double> 类型后的值。
    # 将计算结果强制转换为 std::complex<float> 类型后返回。
    return static_cast<std::complex<float>>(cyl_hankel_2(static_cast<double>(v), static_cast<std::complex<double>>(z)));
}

} // namespace special



// 关闭特定命名空间 "special"
}
```