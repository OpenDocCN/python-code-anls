# `D:\src\scipysrc\scipy\scipy\special\special\fresnel.h`

```
#pragma once


// 使用 pragma once 指令，确保头文件只被编译一次，避免重复包含


#include "config.h"


// 包含名为 config.h 的头文件，用于引入特定的配置信息或者宏定义


namespace special {
namespace detail {


// 进入命名空间 special，嵌套命名空间 detail，用于组织和限定特定的功能或实现细节
    // =========================================================
    // Purpose: Compute complex Fresnel integral C(z) and C'(z)
    // Input :  z --- Argument of C(z)
    // Output:  ZF --- C(z)
    //          ZD --- C'(z)
    // =========================================================

    inline void cfc(std::complex<double> z, std::complex<double> *zf, std::complex<double> *zd) {

        int k, m;
        double wa0, wa;
        std::complex<double> c, cr, cf, cf0, cf1, cg, d;
        const double eps = 1.0e-14;
        const double pi = 3.141592653589793;

        double w0 = std::abs(z);
        std::complex<double> zp = 0.5 * pi * z * z;
        std::complex<double> zp2 = zp * zp;
        std::complex<double> z0 = 0.0;

        // Check if z is zero
        if (z == z0) {
            c = z0;
        } else if (w0 <= 2.5) {
            // Compute C(z) using series expansion for small w0
            cr = z;
            c = cr;
            wa0 = 0.0;
            for (k = 1; k <= 80; k++) {
                cr = -0.5 * cr * (4.0 * k - 3.0) / static_cast<double>(k) / (2.0 * k - 1.0) / (4.0 * k + 1.0) * zp2;
                c += cr;
                wa = std::abs(c);
                // Convergence check
                if ((fabs((wa - wa0) / wa) < eps) && (k > 10)) {
                    *zf = c;
                    *zd = std::cos(0.5 * pi * z * z);
                    return;
                }
                wa0 = wa;
            }
        } else if ((w0 > 2.5) && (w0 < 4.5)) {
            // Compute C(z) using continued fraction for medium w0
            m = 85;
            c = z0;
            cf1 = z0;
            cf0 = 1.0e-100;
            for (k = m; k >= 0; k--) {
                cf = (2.0 * k + 3.0) * cf0 / zp - cf1;
                if (k % 2 == 0) {
                    c += cf;
                }
                cf1 = cf0;
                cf0 = cf;
            }
            c *= 2.0 / (pi * z) * std::sin(zp) / cf;
        } else {
            // Compute C(z) using quadrant selection and series for large w0
            if ((z.imag() > -z.real()) && (z.imag() <= z.real())) {
                // right quadrant
                d = 0.5;
            } else if ((z.imag() > z.real()) && (z.imag() >= -z.real())) {
                // upper quadrant
                d = std::complex<double>(0, 0.5);
            } else if ((z.imag() < -z.real()) && (z.imag() >= z.real())) {
                // left quadrant
                d = -0.5;
            } else {
                d = std::complex<double>(0, -0.5);
            }
            // Compute C(z) using special case of the series
            cr = 1.0;
            cf = 1.0;
            for (k = 1; k <= 20; k++) {
                cr = -0.25 * cr * (4.0 * k - 1.0) * (4.0 * k - 3.0) / zp2;
                cf += cr;
            }
            // Compute C'(z) using a different series
            cr = 1.0 / (pi * z * z);
            cg = cr;
            for (k = 1; k <= 12; k++) {
                cr = -0.25 * cr * (4.0 * k + 1.0) * (4.0 * k - 1.0) / zp2;
                cg += cr;
            }
            c = d + (cf * std::sin(zp) - cg * std::cos(zp)) / (pi * z);
        }
        // Store results
        *zf = c;
        *zd = std::cos(0.5 * pi * z * z);
        return;
    }
} // namespace detail

/* Fresnel integrals of complex numbers */

// 计算复数的 Fresnel 积分函数 C(z) 和 S(z) 的零点，使用修改的牛顿迭代法
inline void cfresnl(std::complex<double> z, std::complex<double> *zfs, std::complex<double> *zfc) {
    std::complex<double> zfd;

    // 调用 detail 命名空间中的 cfs 函数，计算 S(z) 的积分值和导数
    detail::cfs(z, zfs, &zfd);
    // 调用 detail 命名空间中的 cfc 函数，计算 C(z) 的积分值和导数
    detail::cfc(z, zfc, &zfd);
}

// 模板函数，计算修改后的 Fresnel 积分 F+(x) 和 K+(x)
template <typename T>
void modified_fresnel_plus(T x, std::complex<T> &Fplus, std::complex<T> &Kplus) {
    // 调用 detail 命名空间中的 ffk 函数，计算修改后的 Fresnel 积分 F+(x) 和 K+(x)
    detail::ffk(0, x, Fplus, Kplus);
}

// 模板函数，计算修改后的 Fresnel 积分 F-(x) 和 K-(x)
template <typename T>
void modified_fresnel_minus(T x, std::complex<T> &Fminus, std::complex<T> &Kminus) {
    // 调用 detail 命名空间中的 ffk 函数，计算修改后的 Fresnel 积分 F-(x) 和 K-(x)
    detail::ffk(1, x, Fminus, Kminus);
}

// 计算 Fresnel 积分 C(z) 或 S(z) 的复数零点，使用修改的牛顿迭代法
inline void fcszo(int kf, int nt, std::complex<double> *zo) {

    // ===============================================================
    // Purpose: Compute the complex zeros of Fresnel integral C(z)
    //          or S(z) using modified Newton's iteration method
    // Input :  KF  --- Function code
    //                  KF=1 for C(z) or KF=2 for S(z)
    //          NT  --- Total number of zeros
    // Output:  ZO(L) --- L-th zero of C(z) or S(z)
    // Routines called:
    //      (1) CFC for computing Fresnel integral C(z)
    //      (2) CFS for computing Fresnel integral S(z)
    // ==============================================================

    int it;
    double psq, px, py, w, w0;
    std::complex<double> z, zp, zf, zd, zfd, zgd, zq, zw;
    const double pi = 3.141592653589793;
    psq = 0.0;
    w = 0.0;

    for (int nr = 1; nr <= nt; ++nr) {
        if (kf == 1)
            psq = sqrt(4.0 * nr - 1.0);
        if (kf == 2)
            psq = 2.0 * sqrt(nr);

        px = psq - log(pi * psq) / (pi * pi * psq * psq * psq);
        py = log(pi * psq) / (pi * psq);
        z = std::complex<double>(px, py);

        if (kf == 2) {
            if (nr == 2) {
                z = std::complex<double>(2.8334, 0.2443);
            }
            if (nr == 3) {
                z = std::complex<double>(3.4674, 0.2185);
            }
            if (nr == 4) {
                z = std::complex<double>(4.0025, 0.2008);
            }
        }

        it = 0;
        do {
            it++;
            if (kf == 1) {
                // 调用 detail 命名空间中的 cfc 函数，计算 C(z) 的积分值和导数
                detail::cfc(z, &zf, &zd);
            }
            if (kf == 2) {
                // 调用 detail 命名空间中的 cfs 函数，计算 S(z) 的积分值和导数
                detail::cfs(z, &zf, &zd);
            }

            zp = 1.0;
            for (int i = 1; i < nr; i++)
                zp *= (z - zo[i - 1]);

            zfd = zf / zp;
            zq = 0.0;
            for (int i = 1; i < nr; i++) {
                zw = 1.0;
                for (int j = 1; j < nr; j++) {
                    if (j == i) {
                        continue;
                    }
                    zw *= (z - zo[j - 1]);
                }
                zq += zw;
            }
            zgd = (zd - zq * zfd) / zp;
            z -= zfd / zgd;
            w0 = w;
            w = std::abs(z);
        } while ((it <= 50) && (fabs((w - w0) / w) > 1.0e-12));
        zo[nr - 1] = z;
    }
    return;
}

} // namespace special
```