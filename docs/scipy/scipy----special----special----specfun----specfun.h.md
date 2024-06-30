# `D:\src\scipysrc\scipy\scipy\special\special\specfun\specfun.h`

```
/*
 * This file accompanied with the header file specfun.h is a partial
 * C translation of the Fortran code by Zhang and Jin following
 * original description:
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *       COMPUTATION OF SPECIAL FUNCTIONS
 *
 *          Shanjie Zhang and Jianming Jin
 *
 *       Copyrighted but permission granted to use code in programs.
 *       Buy their book:
 *
 *          Shanjie Zhang, Jianming Jin,
 *          Computation of Special Functions,
 *          Wiley, 1996,
 *          ISBN: 0-471-11963-6,
 *          LC: QA351.C45.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 *       Scipy changes:
 *       - Compiled into a single source file and changed REAL To DBLE throughout.
 *       - Changed according to ERRATA.
 *       - Changed GAMMA to GAMMA2 and PSI to PSI_SPEC to avoid potential conflicts.
 *       - Made functions return sf_error codes in ISFER variables instead
 *         of printing warnings. The codes are
 *         - SF_ERROR_OK        = 0: no error
 *         - SF_ERROR_SINGULAR  = 1: singularity encountered
 *         - SF_ERROR_UNDERFLOW = 2: floating point underflow
 *         - SF_ERROR_OVERFLOW  = 3: floating point overflow
 *         - SF_ERROR_SLOW      = 4: too many iterations required
 *         - SF_ERROR_LOSS      = 5: loss of precision
 *         - SF_ERROR_NO_RESULT = 6: no result obtained
 *         - SF_ERROR_DOMAIN    = 7: out of domain
 *         - SF_ERROR_ARG       = 8: invalid input parameter
 *         - SF_ERROR_OTHER     = 9: unclassified error
 *       - Improved initial guesses for roots in JYZO.
 */
/*
 * 版权所有 (C) 2024 SciPy 开发者
 *
 * 源代码和二进制形式的重新分发，在遵循以下条件的前提下，无论是否进行修改，
 * 都是允许的：
 *
 * a. 源代码的重新分发必须保留上述版权声明、此条件列表和以下免责声明。
 * b. 二进制形式的重新分发必须在随附分发的文档或其他材料中重现上述版权声明、
 *    此条件列表和以下免责声明。
 * c. 未经特定书面许可，不得使用 SciPy 开发者的名称来认可或推广由本软件衍生的产品。
 *
 * 本软件由版权所有者和贡献者按 "现状" 提供，不提供任何明示或暗示的担保，
 * 包括但不限于适销性和特定用途适用性的暗示担保。在任何情况下，版权所有者或
 * 贡献者对任何直接、间接、偶然、特殊、示范性或后果性损害（包括但不限于
 * 采购替代商品或服务；使用数据、利润或业务中断）不承担责任，无论是因合同责
 * 任、严格责任还是因侵权（包括疏忽或其他方式）引起的，即使事先已告知
 * 可能性。
 */

#pragma once

#include "../config.h"

namespace special {
namespace specfun {

// 定义特殊函数的命名空间

void airyb(double, double*, double*, double*, double*);
// 声明 airy B 函数，接受五个参数：输入参数，三个输出参数

void bjndd(double, int, double *, double *, double *);
// 声明 Bessel 函数的导数，接受三个输入参数和三个输出参数

void cerzo(int, std::complex<double> *);
// 声明一种复数特殊函数，接受整数和一个复数数组作为参数

void cyzo(int, int, int, std::complex<double>*, std::complex<double> *);
// 声明一种复数特殊函数，接受三个整数和两个复数数组作为参数

void cerf(std::complex<double>, std::complex<double> *, std::complex<double> *);
// 声明一种复数特殊函数，接受一个复数和两个复数指针作为参数

std::complex<double> cgama(std::complex<double>, int);
// 声明一种复数伽玛函数，接受一个复数和一个整数作为参数，返回复数结果

double chgubi(double, double, double, int *);
// 声明一个双精度变换函数，接受三个双精度数和一个整数指针作为参数，返回一个双精度数

double chguit(double, double, double, int *);
// 声明一个双精度变换函数，接受三个双精度数和一个整数指针作为参数，返回一个双精度数

double chgul(double, double, double, int *);
// 声明一个双精度变换函数，接受三个双精度数和一个整数指针作为参数，返回一个双精度数

double chgus(double, double, double, int *);
// 声明一个双精度变换函数，接受三个双精度数和一个整数指针作为参数，返回一个双精度数

void cpbdn(int, std::complex<double>, std::complex<double> *, std::complex<double> *);
// 声明一种复数特殊函数，接受一个整数、一个复数和两个复数指针作为参数

std::complex<double> cpdla(int, std::complex<double>);
// 声明一种复数特殊函数，接受一个整数和一个复数作为参数，返回一个复数

std::complex<double> cpdsa(int, std::complex<double>);
// 声明一种复数特殊函数，接受一个整数和一个复数作为参数，返回一个复数

double cv0(double, double, double);
// 声明一个双精度函数，接受三个双精度数作为参数，返回一个双精度数

double cvf(int, int, double, double, int);
// 声明一个双精度函数，接受两个整数和三个双精度数作为参数，返回一个双精度数

double cvql(int, int, double);
// 声明一个双精度函数，接受两个整数和一个双精度数作为参数，返回一个双精度数

double cvqm(int, double);
// 声明一个双精度函数，接受一个整数和一个双精度数作为参数，返回一个双精度数

double gaih(double);
// 声明一个双精度函数，接受一个双精度数作为参数，返回一个双精度数

double gam0(double);
// 声明一个双精度函数，接受一个双精度数作为参数，返回一个双精度数

double gamma2(double);
// 声明一个双精度函数，接受一个双精度数作为参数，返回一个双精度数

template <typename T>
void jynbh(int, int, T, int *, T *, T *);
// 声明一个模板函数，接受两个整数、一个模板类型参数和三个指针作为参数

void jyndd(int, double, double *, double *, double *, double *, double *, double *);
// 声明一个函数，接受一个整数、一个双精度数和六个双精度数指针作为参数

double lpmv0(double, int, double);
// 声明一个双精度函数，接受一个双精度数和两个整数作为参数，返回一个双精度数

int msta1(double, int);
// 声明一个整数函数，接受一个双精度数和一个整数作为参数，返回一个整数

int msta2(double, int, int);
// 声明一个整数函数，接受两个双精度数和一个整数作为参数，返回一个整数

double psi_spec(double);
// 声明一个双精度函数，接受一个双精度数作为参数，返回一个双精度数

double refine(int, int, double, double);
// 声明一个双精度函数，接受两个整数和两个双精度数作为参数，返回一个双精度数

template <typename T>
void sckb(int, int, T, T *, T *);
// 声明一个模板函数，接受两个整数、一个模板类型参数和两个指针作为参数

template <typename T>
void sdmn(int, int, T, T, int, T *);
// 声明一个模板函数，接受两个整数、两个模板类型参数、一个整数和一个指针作为参数

template <typename T>
*/
template <typename T>
void aswfa(T x, int m, int n, T c, int kd, T cv, T *s1f, T *s1d) {
    // ===========================================================
    // Purpose: Compute the prolate and oblate spheroidal angular
    //          functions of the first kind and their derivatives
    // Input :  m  --- Mode parameter,  m = 0,1,2,...
    //          n  --- Mode parameter,  n = m,m+1,...
    //          c  --- Spheroidal parameter
    //          x  --- Argument of angular function, |x| < 1.0
    //          KD --- Function code
    //                 KD=1 for prolate;  KD=-1 for oblate
    //          cv --- Characteristic value
    // Output:  S1F --- Angular function of the first kind
    //          S1D --- Derivative of the angular function of
    //                  the first kind
    // Routine called:
    //          sdmn for computing derivative coefficients df
    //          sckb for computing expansion coefficients ck
    // ===========================================================

    int ip, k, nm, nm2;
    T a0, d0, d1, r, su1, su2, x0, x1;
    T *ck = (T *) calloc(200, sizeof(T));  // Allocate memory for expansion coefficients
    T *df = (T *) calloc(200, sizeof(T));  // Allocate memory for derivative coefficients
    const T eps = 1e-14;  // Machine epsilon

    x0 = x;
    x = fabs(x);  // Ensure x is non-negative

    ip = ((n-m) % 2 == 0 ? 0 : 1);  // Determine parity of (n-m)
    nm = 40 + (int)((n-m)/2 + c);  // Compute nm based on mode parameters m, n, and spheroidal parameter c
    nm2 = nm/2 - 2;

    sdmn(m, n, c, cv, kd, df);  // Compute derivative coefficients df
    sckb(m, n, c, df, ck);  // Compute expansion coefficients ck

    x1 = 1.0 - x*x;

    if ((m == 0) && (x1 == 0.0)) {
        a0 = 1.0;  // Special case handling
    } else {
        a0 = pow(x1, 0.5*m);  // Compute power of x1
    }

    su1 = ck[0];  // Initialize sum for angular function
    for (k = 1; k <= nm2; k++) {
        r = ck[k]*pow(x1, k);  // Compute term for sum
        su1 += r;  // Accumulate sum
        if ((k >= 10) && (fabs(r/su1) < eps)) { break; }  // Check convergence
    }

    *s1f = a0*pow(x, ip)*su1;  // Compute angular function S1F

    if (x == 1.0) {
        // Special cases when x = 1.0
        if (m == 0) {
            *s1d = ip*ck[0] - 2.0*ck[1];
        } else if (m == 1) {
            *s1d = -1e100;  // Arbitrary large negative value
        } else if (m == 2) {
            *s1d = -2.0*ck[0];
        } else if (m >= 3) {
            *s1d = 0.0;
        }
    } else {
        // General case
        d0 = ip - m/x1*pow(x, ip+1.0);
        d1 = -2.0*a0*pow(x, ip+1.0);
        su2 = ck[1];  // Initialize sum for derivative
        for (k = 2; k <= nm2; k++) {
            r = k*ck[k]*pow(x1, (k-1.0));  // Compute term for derivative sum
            su2 += r;  // Accumulate sum
            if ((k >= 10) && (fabs(r/su2) < eps)) { break; }  // Check convergence
        }
        *s1d = d0*a0*su1 + d1*su2;  // Compute derivative of angular function S1D
    }

    if ((x0 < 0.0) && (ip == 0)) { *s1d = -*s1d; }  // Adjust sign if necessary
    if ((x0 < 0.0) && (ip == 1)) { *s1f = -*s1f; }  // Adjust sign if necessary

    x = x0;  // Restore original value of x
    free(ck); free(df);  // Free allocated memory
    return;  // Return from function
}
    # 循环计算贝塞尔函数的系数 bn[m]，m 从 4 开始，每次增加 2，直到 n+1
    for ( m = 4; m < (n+1); m += 2) {
        # 计算贝塞尔函数的前导系数 r1
        r1 = -r1 * (m-1)*m/(tpi*tpi);
        # 初始化 r2 为 1.0
        r2 = 1.0;
        # 循环计算贝塞尔函数的主体部分，k 从 2 开始，直到 10001
        for (k = 2; k < 10001; k++) {
            # 计算当前项 s
            s = pow(1.0/k, m);
            # 累加到 r2 中
            r2 += s;
            # 如果当前项 s 的值小于 1e-15，跳出循环
            if (s < 1e-15) { break; }
        }
        # 将计算得到的 bn[m] 存入结果数组中
        bn[m] = r1*r2;
    }
    # 函数无返回值，直接返回
    return;
inline void bjndd(double x, int n, double *bj, double *dj, double *fj) {
    // =====================================================
    // Purpose: Compute Bessel functions Jn(x) and their
    //          first and second derivatives ( 0 <= n <= 100)
    // Input:   x ---  Argument of Jn(x)  ( x ≥ 0 )
    //          n ---  Order of Jn(x)
    // Output:  BJ(n+1) ---  Jn(x)
    //          DJ(n+1) ---  Jn'(x)
    //          FJ(n+1) ---  Jn"(x)
    // =====================================================

    int k, m, mt;
    double bs = 0.0, f = 0.0, f0 = 0.0, f1 = 1e-35;

    // Determine suitable upper limit for summation
    for (m = 1; m < 901; m++) {
        mt = (int)(0.5*log10(6.28*m)-m*log10(1.36*fabs(x)/m));
        if (mt > 20) { break; }
    }
    if (m == 901) { m -= 1; }

    // Compute Bessel function and derivatives
    for (k = m; k > -1; k--) {
        f = 2.0*(k+1.0)*f1/x - f0;
        if (k <= n) { bj[k] = f; }
        if (k % 2 == 0) { bs += 2.0*f; }
        f0 = f1;
        f1 = f;
    }

    // Normalize Bessel function values
    for (k = 0; k < (n+1); k++) {
        bj[k] /= (bs - f);
    }

    // Compute derivatives Jn'(x) and Jn"(x)
    dj[0] = -bj[1];
    fj[0] = -bj[0] - dj[0]/x;
    for (k = 1; k < (n+1); k++) {
        dj[k] = bj[k-1] - k*bj[k]/x;
        fj[k] = (k*k/(x*x)-1.0)*bj[k] - dj[k]/x;
    }

    return;
}
    // 如果ip为1，则执行以下代码块
    } else if (ip == 1) {
        // 初始化sw为0.0
        sw = 0.0;
        // 对于k从0到n2-1的循环
        for (k = 0; k < n2; k++) {
            // 初始化s1为0.0
            s1 = 0.0;
            // 计算i1的值，即k-m+1
            i1 = k - m + 1;

            // 对于i从i1到nm的循环
            for (int i = i1; i < nm + 1; i++) {
                // 如果i小于0，则跳过本次循环
                if (i < 0) { continue; }
                // 初始化r1为1.0
                r1 = 1.0;
                // 对于j从1到k的循环
                for (j = 1; j <= k; j++) {
                    // 计算r1的值
                    r1 = r1 * (i + m - j) / (1.0 * j);
                }
                // 如果i大于0
                if (i > 0) {
                    // 更新s1的值
                    s1 += ck[i - 1] * (2.0 * i + m - 1) * r1;
                }
                // 更新s1的值
                s1 -= ck[i] * (2.0 * i + m) * r1;

                // 如果fabs(s1 - sw)小于fabs(s1) * eps，则退出循环
                if (fabs(s1 - sw) < fabs(s1) * eps) { break; }
                // 更新sw的值
                sw = s1;
            }
            // 计算bk[k]的值
            bk[k] = qt * s1;
        }
    }

    // 更新w[0]和bk[0]的值
    w[0] /= v[0];
    bk[0] /= v[0];

    // 对于k从2到n2的循环
    for (k = 2; k <= n2; k++) {
        // 计算t的值
        t = v[k - 1] - w[k - 2] * u[k - 1];
        // 更新w[k-1]的值
        w[k - 1] /= t;
        // 更新bk[k-1]的值
        bk[k - 1] = (bk[k - 1] - bk[k - 2] * u[k - 1]) / t;
    }

    // 对于k从n2-1到1的递减循环
    for (k = n2 - 1; k >= 1; k--) {
        // 更新bk[k-1]的值
        bk[k - 1] -= w[k - 1] * bk[k];
    }
    // 释放动态分配的数组u, v, w
    free(u); free(v); free(w);
    // 返回
    return;
}

inline void cerf(std::complex<double> z, std::complex<double> *cer, std::complex<double> *cder) {

    // ==========================================================
    // Purpose: Compute complex Error function erf(z) & erf'(z)
    // Input:   z   --- Complex argument of erf(z)
    // Output:  cer --- erf(z)
    //          cder --- erf'(z)
    // ==========================================================

    int k;
    double c0, cs, er0, er, er1, ei1, er2, ei2, err, eri, r, ss, w, w1, w2;
    const double eps = 1.0e-12;
    const double pi = 3.141592653589793;

    double x = z.real();
    double y = z.imag();
    double x2 = x * x;

    // Check if the real part of z is less than or equal to 3.5
    if (x <= 3.5) {
        er = 1.0;
        r = 1.0;
        w = 0.0;

        // Compute the series expansion for erf(z)
        for (k = 1; k <= 100; k++) {
            r = r * x2 / (k + 0.5);
            er += r;
            if (fabs(er - w) <= eps * fabs(er))
                break;
            w = er;
        }

        // Compute the prefactor and final result for erf(z)
        c0 = 2.0 / sqrt(pi) * x * exp(-x2);
        er0 = c0 * er;
        *cer = er0;
    } else {
        er = 1.0;
        r = 1.0;

        // Compute the series expansion for large x (>3.5)
        for (k = 1; k <= 12; k++) {
            r = -r * (k - 0.5) / x2;
            er += r;
        }

        // Compute the prefactor and final result for erf(z)
        c0 = exp(-x2) / (x * sqrt(pi));
        er0 = 1.0 - c0 * er;
        *cer = er0;
    }

    // Handle the imaginary part of z
    if (y == 0.0) {
        err = cer->real();
        eri = 0.0;
        *cer = std::complex<double>(err, eri);
    } else {
        cs = cos(2.0 * x * y);
        ss = sin(2.0 * x * y);
        er1 = exp(-x2) * (1.0 - cs) / (2.0 * pi * x);
        ei1 = exp(-x2) * ss / (2.0 * pi * x);
        er2 = 0.0;
        w1 = 0.0;

        // Compute the second term in the imaginary part of erf(z)
        for (int n = 1; n <= 100; n++) {
            er2 += exp(-0.25 * n * n) / (n * n + 4.0 * x2) * (2.0 * x - 2.0 * x * cosh(n * y) * cs + n * sinh(n * y) * ss);
            if (fabs((er2 - w1) / er2) < eps)
                break;
            w1 = er2;
        }

        // Compute the prefactor and final imaginary part of erf(z)
        c0 = 2.0 * exp(-x2) / pi;
        err = cer->real() + er1 + c0 * er2;
        ei2 = 0.0;
        w2 = 0.0;

        // Compute the third term in the imaginary part of erf(z)
        for (int n = 1; n <= 100; n++) {
            ei2 += exp(-0.25 * n * n) / (n * n + 4.0 * x2) * (2.0 * x * cosh(n * y) * ss + n * sinh(n * y) * cs);
            if (fabs((ei2 - w2) / ei2) < eps)
                break;
            w2 = ei2;
        }
        *cer = std::complex<double>(err, ei1 + c0 * ei2);
    }
    // Compute the derivative of erf(z)
    *cder = 2.0 / sqrt(pi) * std::exp(-z*z);

}


inline std::complex<double> cerror(std::complex<double> z) {

    // ====================================================
    // Purpose: Compute error function erf(z) for a complex
    //          argument (z=x+iy)
    // Input :  z   --- Complex argument
    // Output:  cer --- erf(z)
    // ====================================================

    int k;
    std::complex<double> cer, cl, cr, cs, z1;
    std::complex<double> c0 = std::exp(-z*z);
    const double sqpi = 1.7724538509055160273;
    z1 = z;

    // Adjust z1 if the real part of z is negative
    if (z.real() < 0.0) { z1 = -z; }
    // 如果 z 的绝对值小于等于 4.36，则使用Taylor级数展开计算CER函数
    // Taylor级数的截断误差大约为 ~ R*R * EPSILON * R**(2 R**2) / (2 R**2 Gamma(R**2 + 1/2))
    // 其中 R = 4.36，EPSILON为机器精度
    if (std::abs(z) <= 4.36) {
        // 初始化cs和cr为z1，用于累加Taylor级数的部分和
        cs = z1;
        cr = z1;
        // 循环计算Taylor级数的部分和，最多进行120次迭代
        for (k = 1; k < 121; k++) {
            // 计算下一项的cr，并累加到cs中
            cr = cr*(z1*z1) / (k+0.5);
            cs += cr;
            // 检查截断误差是否满足精度要求，如果满足则终止循环
            if (std::abs(cr/cs) < 1e-15) { break; }
        }
        // 计算最终结果cer，这里乘以2.0*c0/sqpi
        cer = 2.0*c0*cs/sqpi;
    } else {
        // 如果 z 的绝对值大于 4.36，则使用渐近级数计算CER函数
        // 初始化cl和cr为1.0/z1，用于累加渐近级数的部分和
        cl = 1.0 / z1;
        cr = cl;
        // 循环计算渐近级数的部分和，最多进行20次迭代
        for (k = 1; k < 21; k++) {
            // 计算下一项的cr，并累加到cl中
            cr = -cr*(k-0.5) / (z1*z1);
            cl += cr;
            // 检查截断误差是否满足精度要求，如果满足则终止循环
            if (std::abs(cr/cl) < 1e-15) { break; }
        }
        // 计算最终结果cer，这里乘以1.0-c0/sqpi
        cer = 1.0 - c0*cl/sqpi;
    }
    // 如果 z 的实部小于 0，则将cer取负值
    if (z.real() < 0.0) { cer = -cer; }
    // 返回计算得到的cer作为函数结果
    return cer;
}

// ===============================================================
// Purpose : Evaluate the complex zeros of error function erf(z)
//           using the modified Newton's iteration method
// Input :   NT --- Total number of zeros
// Output:   ZO(L) --- L-th zero of erf(z), L=1,2,...,NT
// Routine called: CERF for computing erf(z) and erf'(z)
// ===============================================================
inline void cerzo(int nt, std::complex<double> *zo) {

    int i, j, nr, it = 0;
    double pu, pv, px, py, w0;
    std::complex<double> z, zf, zd, zp, zw, zq, zfd, zgd;
    double w = 0.0;
    const double pi = 3.141592653589793;

    for (nr = 1; nr <= nt; nr++) {
        pu = sqrt(pi * (4.0 * nr - 0.5));
        pv = pi * sqrt(2.0 * nr - 0.25);
        px = 0.5 * pu - 0.5 * log(pv) / pu;
        py = 0.5 * pu + 0.5 * log(pv) / pu;
        z = std::complex<double>(px, py);
        it = 0;

        // Perform Newton's iteration to find the zeros of erf(z)
        do {
            it++;
            // Compute erf(z) and its derivative erf'(z)
            cerf(z, &zf, &zd);
            zp = 1.0;

            // Compute the product (z - zo[i]) for i = 1 to nr-1
            for (i = 1; i < nr; i++) {
                zp *= (z - zo[i - 1]);
            }
            zfd = zf / zp;

            zq = 0.0;
            // Compute the sum of products (z - zo[j]) for j = 1 to nr-1
            for (i = 1; i < nr; i++) {
                zw = 1.0;
                for (j = 1; j < nr; j++) {
                    if (j == i) continue;
                    zw *= (z - zo[j - 1]);
                }
                zq += zw;
            }
            // Compute the derivative of zfd divided by zgd
            zgd = (zd - zq * zfd) / zp;
            // Update z using Newton's method
            z -= zfd / zgd;
            w0 = w;
            w = std::abs(z);
        } while ((it <= 50) && (fabs((w - w0) / w) > 1.0e-11));

        // Store the computed zero in zo[nr-1]
        zo[nr - 1] = z;
    }
    return;
}

// ===================================================
// Purpose: Compute confluent hypergeometric function
//          M(a,b,z) with real parameters a, b and a
//          complex argument z
// Input :  a --- Parameter
//          b --- Parameter
//          z --- Complex argument
// Output:  CHG --- M(a,b,z)
// Routine called: CGAMA for computing complex ln[Г(x)]
// ===================================================
inline std::complex<double> cchg(double a, double b, std::complex<double> z) {

    int i, j, k, la, m, n, nl, ns;
    double a0, a1, phi, x0, x, y;
    std::complex<double> cfac, cg1, cg2, cg3, chg, chg1, chg2, chw, cr, cr1, cr2, cs1,
                   cs2, crg, cy0, cy1, z0;
    const double pi = 3.141592653589793;
    const std::complex<double> ci(0.0, 1.0);

    // Initialize variables
    a0 = a;
    a1 = a;
    z0 = z;
    cy0 = 0.0;
    cy1 = 0.0;

    // Check special cases and return early if conditions are met
    if ((b == 0.0) || (b == -(int)fabs(b))) { return 1e300; }
    if ((a == 0.0) || (z == 0.0)) { return 1.0; }
    if (a == -1.0) { return 1.0 - z/b; }
    if (a == b) { return std::exp(z); }
    if (a - b == 1.0) { return (1.0 + z/b)*std::exp(z); }
    if ((a == 1.0) && (b == 2.0)) { return (std::exp(z)-1.0) / z; }

    // Further computations...


这段代码实现了两个内联函数 `cerzo` 和 `cchg`，分别用于计算误差函数的复零点和复变量的超几何函数。
    // 如果 a 是整数且小于 0，则执行以下操作
    if ((a == (int)a) && (a < 0.0)) {
        // 取 a 的绝对值，将其赋给 m
        m = (int)(-a);
        // 初始化 cr 和 chg 为 1.0
        cr = 1.0;
        chg = 1.0;
        // 循环计算特定条件下的累积值
        for (k = 1; k < (m+1); k++) {
            // 计算累积乘积，更新 cr
            cr = cr * (a+k-1.0)/static_cast<double>(k)/(b+k-1.0)*z;
            // 累加至 chg
            chg += cr;
        }
    } else {
        // 获取 z 的实部赋给 x0
        x0 = z.real();
        // 如果 x0 小于 0.0，则执行以下操作
        if (x0 < 0.0) {
            // 调整参数 a 和 z 的值
            a = b-a;
            a0 = a;
            z = -z;
        }
        // 初始化 nl 和 la 为 0
        nl = 0;
        la = 0;
        // 如果 a 大于等于 2.0，则执行以下操作
        if (a >= 2.0) {
            // 设置 nl 为 1，la 设置为 a 的整数部分
            nl = 1;
            la = (int)a;
            // 调整参数 a 的值
            a -= la + 1;
        }
        // 初始化 ns 为 0
        ns = 0;
        // 对 nl 的每一项执行循环
        for (n = 0; n < (nl+1); n++) {
            // 如果 a0 大于等于 2.0，则增加 a 的值
            if (a0 >= 2.0) { a += 1.0; }
            // 如果满足指定条件，则执行以下操作
            if ((std::abs(z) < 20.0+fabs(b)) || (a < 0.0)) {
                // 初始化 chg 为 1.0
                chg = 1.0;
                // 初始化 chw 为 0.0
                chw = 0.0;
                // 初始化 crg 为 1.0
                crg = 1.0;
                // 执行循环，计算累积值直到满足条件
                for (j = 1; j < 501; j++) {
                    // 计算累积乘积，更新 crg
                    crg = crg * (a+j-1.0)/(j*(b+j-1.0))*z;
                    // 累加至 chg
                    chg += crg;
                    // 如果达到足够精度，则终止循环
                    if (std::abs((chg-chw)/chg) < 1e-15) { break; }
                    // 更新 chw
                    chw = chg;
                }
            } else {
                // 初始化 y 为 0.0
                y = 0.0;
                // 计算 gamma 函数的值
                cg1 = cgama(a, 0);
                cg2 = cgama(b, 0);
                cg3 = cgama(b-a, 0);
                cs1 = 1.0;
                cs2 = 1.0;
                cr1 = 1.0;
                cr2 = 1.0;
                // 执行循环计算累积值
                for (i = 1; i <= 8; i++) {
                    // 计算累积乘积，更新 cr1 和 cr2
                    cr1 = -cr1 * (a+i-1.0)*(a-b+i)/(z*static_cast<double>(i));
                    cr2 = cr2 * (b-a+i-1.0)*(i-a)/(z*static_cast<double>(i));
                    // 累加至 cs1 和 cs2
                    cs1 += cr1;
                    cs2 += cr2;
                }
                // 获取 z 的实部和虚部
                x = z.real();
                y = z.imag();
                // 根据条件设置 phi 的值
                if ((x == 0.0) && (y >= 0.0)) {
                    phi = 0.5*pi;
                } else if ((x == 0.0) && (y <= 0.0)) {
                    phi = -0.5*pi;
                } else {
                    phi = atan(y/x);
                }
                // 根据 phi 的值设置 ns 的值
                if ((phi > -0.5*pi) && (phi < 1.5*pi)) { ns = 1; }
                if ((phi > -1.5*pi) && (phi <= -0.5*pi)) { ns = -1; }
                // 计算 cfac 的值
                cfac = std::exp(static_cast<double>(ns)*ci*pi*a);
                if (y == 0.0) { cfac = cos(pi*a); }
                // 计算 chg1 和 chg2 的值
                chg1 = std::exp(cg2-cg3)*std::pow(z, -a)*cfac*cs1;
                chg2 = std::exp(cg2-cg1+z)*std::pow(z, a-b)*cs2;
                // 计算 chg 的值
                chg = chg1 + chg2;
            }
            // 如果是第一次循环，则将 chg 赋给 cy0
            if (n == 0) { cy0 = chg; }
            // 如果是第二次循环，则将 chg 赋给 cy1
            if (n == 1) { cy1 = chg; }
        }
        // 如果 a0 大于等于 2.0，则执行以下操作
        if (a0 >= 2.0) {
            // 对 la 次执行循环计算 chg 的值
            for (i = 1; i < la; i++) {
                chg = ((2.0*a-b+z)*cy1 + (b-a)*cy0)/a;
                // 更新 cy0 和 cy1 的值
                cy0 = cy1;
                cy1 = chg;
                // 增加 a 的值
                a += 1.0;
            }
        }
        // 如果 x0 小于 0.0，则乘以 std::exp(-z)
        if (x0 < 0.0) { chg *= std::exp(-z); }
    }
    // 将 a1 和 z0 的值分别赋给 a 和 z
    a = a1;
    z = z0;
    // 返回计算得到的 chg 的值
    return chg;
}



// =========================================================
// Purpose: Compute the gamma function Г(z) or ln[Г(z)]
//          for a complex argument
// Input :  z  --- Complex argument
//          kf --- Function code
//                 kf=0 for ln[Г(z)]
//                 kf=1 for Г(z)
// Output:  g  --- ln[Г(z)] or Г(z)
// ========================================================

inline std::complex<double> cgama(std::complex<double> z, int kf) {

    std::complex<double> g, z1;
    double az0, az1, gi, gi1, gr, gr1, t, th, th1, th2, sr, si, x0, xx, yy;
    int j, k, na;
    const double pi = 3.141592653589793;
    static const double a[10] = {
        8.333333333333333e-02, -2.777777777777778e-03,
        7.936507936507937e-04, -5.952380952380952e-04,
        8.417508417508418e-04, -1.917526917526918e-03,
        6.410256410256410e-03, -2.955065359477124e-02,
        1.796443723688307e-01, -1.392432216905900e+00
    };
    xx = z.real();
    yy = z.imag();
    // Check if z is purely real and less than or equal to 0
    // Return a large number if true to prevent invalid computation
    if ((yy == 0.0) && (xx <= 0.0) && (xx == (int)xx)) {
        return 1e300;
    } else if (xx < 0.0) {
        // Adjust z and xx, yy if xx is negative
        z1 = z;
        z = -z;
        xx = -xx;
        yy = -yy;
    } else {
        z1 = std::complex<double>(xx, 0.0);
    }
    x0 = xx;
    na = 0;
    // Adjust x0 and na if xx is less than or equal to 7
    if (xx <= 7.0) {
        na = (int)(7 - xx);
        x0 = xx + na;
    }
    az0 = std::abs(std::complex<double>(x0, yy));
    th = atan(yy / x0);
    // Compute gr and gi using series expansion and logarithmic functions
    gr = (x0 - 0.5)*log(az0) - th*yy - x0 + 0.5*log(2.0*pi);
    gi = th*(x0 - 0.5) + yy*log(az0) - yy;
    for (k = 1; k < 11; k++) {
        t = pow(az0, 1-2*k);
        gr += a[k - 1]*t*cos((2.0*k - 1.0)*th);
        gi += -a[k - 1]*t*sin((2.0*k - 1.0)*th);
    }
    // Adjust gr and gi if xx is less than or equal to 7
    if (xx <= 7.0) {
        gr1 = 0.0;
        gi1 = 0.0;
        for (j = 0; j < na; j++) {
            gr1 += 0.5*log(pow(xx + j, 2) + yy*yy);
            gi1 += atan(yy/(xx + j));
        }
        gr -= gr1;
        gi -= gi1;
    }
    // Adjust gr and gi if z1.real() < 0
    if (z1.real() < 0.0) {
        az0 = std::abs(z);
        th1 = atan(yy/xx);
        sr = -sin(pi*xx)*cosh(pi*yy);
        si = -cos(pi*xx)*sinh(pi*yy);
        az1 = std::abs(std::complex<double>(sr, si));
        th2 = atan(si/sr);
        if (sr < 0.0) {
            th2 += pi;
        }
        gr = log(pi/(az0*az1)) - gr;
        gi = - th1 - th2 - gi;
        z = z1;
    }
    // Return g as exp(gr)*exp(i*gi) or (gr + i*gi) based on kf
    if (kf == 1) {
        g = exp(gr)*std::complex<double>(cos(gi), sin(gi));
    } else {
        g = std::complex<double>(gr, gi);
    }
    return g;
}



// ===================================================
// Purpose: Compute confluent hypergeometric function
//          M(a,b,x)
// Input  : a  --- Parameter
//          b  --- Parameter ( b <> 0,-1,-2,... )
//          x  --- Argument
// Output:  HG --- M(a,b,x)
// Routine called: CGAMA for computing complex ln[Г(x)]
// ===================================================

inline double chgm(double x, double a, double b) {

    int i, j, la, n, nl;
    // 初始化变量和常量
    double a0 = a, a1 = a, x0 = x, y0, y1, hg1, hg2, r1, r2, rg, xg, sum1, sum2;
    std::complex<double> cta, ctb, ctba;
    const double pi = 3.141592653589793;
    double hg = 0.0;

    // DLMF 13.2.39：根据 DLMF 13.2.39 条款调整参数 a 和 x 的值
    if (x < 0.0) {
        a = b - a;
        a0 = a;
        x = fabs(x);
    }

    // 初始化计数器和变量
    int nl = 0;
    int la = 0;

    // 根据参数 a 的值设置 nl 和 la 的值，为 DLMF 13.3.1 做准备
    if (a >= 2.0) {
        nl = 1;
        la = (int)a;
        a -= la + 1;
    }

    // 初始化 y0 和 y1
    y0 = 0.0;
    y1 = 0.0;

    // 循环计算 n 的值
    for (int n = 0; n < (nl + 1); n++) {
        // 如果 a0 大于等于 2.0，增加 a 的值
        if (a0 >= 2.0) {
            a += 1.0;
        }

        // 判断条件，根据不同情况计算超几何函数 hg
        if ((x <= 30.0 + fabs(b)) || (a < 0.0)) {
            // 计算条件：x 较小或者 a 小于 0
            hg = 1.0;
            rg = 1.0;

            // 计算超几何级数
            for (int j = 1; j < 501; j++) {
                rg = rg * (a + j - 1.0) / (j * (b + j - 1.0)) * x;
                hg += rg;

                // 判断收敛条件
                if ((hg != 0.0) && (fabs(rg / hg) < 1e-15)) {
                    // 应用 DLMF 13.2.39（参见上文）
                    if (x0 < 0.0) {
                        hg *= exp(x0);
                    }
                    break;
                }
            }
        } else {
            // 计算条件：x 较大，应用 DLMF 13.7.2 & 13.2.4，SUM2 对应第一项和
            cta = cgama(a, 0);
            ctb = cgama(b, 0);
            xg = b - a;
            ctba = cgama(xg, 0);
            sum1 = 1.0;
            sum2 = 1.0;
            r1 = 1.0;
            r2 = 1.0;

            // 计算两个求和项的级数
            for (int i = 1; i < 9; i++) {
                r1 = -r1 * (a + i - 1.0) * (a - b + i) / (x * i);
                r2 = -r2 * (b - a + i - 1.0) * (a - i) / (x * i);
                sum1 += r1;
                sum2 += r2;
            }

            // 根据 x0 的值选择计算方式，应用 DLMF 13.2.39（参见上文）
            if (x0 >= 0.0) {
                hg1 = (std::exp(ctb - ctba)).real() * pow(x, -a) * cos(pi * a) * sum1;
                hg2 = (std::exp(ctb - cta + x)).real() * pow(x, a - b) * sum2;
            } else {
                hg1 = (std::exp(ctb - ctba + x0)).real() * pow(x, -a) * cos(pi * a) * sum1;
                hg2 = (std::exp(ctb - cta)).real() * pow(x, a - b) * sum2;
            }

            hg = hg1 + hg2;
        }

        // 25：保存计算结果到 y0 和 y1 中
        if (n == 0) {
            y0 = hg;
        }
        if (n == 1) {
            y1 = hg;
        }
    }

    // 应用 DLMF 13.3.1，如果 a0 大于等于 2.0
    if (a0 >= 2.0) {
        for (int i = 1; i < la; i++) {
            hg = ((2.0 * a - b + x) * y1 + (b - a) * y0) / a;
            y0 = y1;
            y1 = hg;
            a += 1.0;
        }
    }

    // 恢复原始的 a 和 x 的值，并返回超几何函数 hg
    a = a1;
    x = x0;
    return hg;
    // =======================================================
    // Purpose: Compute the confluent hypergeometric function
    //          U(a,b,x) with integer b ( b = ±1,±2,... )
    // Input  : a  --- Parameter
    //          b  --- Parameter
    //          x  --- Argument
    // Output:  HU --- U(a,b,x)
    //          ID --- Estimated number of significant digits
    // Routines called:
    //      (1) GAMMA2 for computing gamma function Г(x)
    //      (2) PSI_SPEC for computing psi function
    // =======================================================

    int id1, id2, j, k, m, n;
    double a0, a1, a2, da1, da2, db1, db2, ga, ga1, h0, hm1, hm2, hm3,\
           hmax, hmin, hu, hu1, hu2, hw, ps, r, rn, rn1, s0, s1, s2,\
           sa, sb, ua, ub;
    const double el = 0.5772156649015329;

    // Initialize the significant digits estimate to a default value
    *id = -100;

    // Declaration of variables used in various parts of the function
    // These variables are used for intermediate computations and results
    # 计算 fabs(b-1) 的整数部分并赋值给 n
    n = (int)fabs(b-1);
    # 初始化 rn1 和 rn 为 1.0
    rn1 = 1.0;
    rn = 1.0;
    # 循环计算阶乘 rn = n!
    for (j = 1; j <= n; j++) {
        rn *= j;
        # 当 j 等于 n-1 时，将 rn 的值赋给 rn1
        if (j == n-1) {
            rn1 = rn;
        }
    }
    # 计算 psi_spec(a) 并赋给 ps
    ps = psi_spec(a);
    # 计算 gamma2(a) 并赋给 ga
    ga = gamma2(a);
    # 根据 b 的值分支处理
    if (b > 0.0) {
        # 对于 b 大于 0 的情况
        a0 = a;
        a1 = a - n;
        a2 = a1;
        # 计算 gamma2(a1) 并赋给 ga1
        ga1 = gamma2(a1);
        # 计算 ua 和 ub
        ua = pow(-1, n-1) / (rn * ga1);
        ub = rn1 / ga * pow(x, -n);
    } else {
        # 对于 b 小于等于 0 的情况
        a0 = a + n;
        a1 = a0;
        a2 = a;
        # 计算 gamma2(a1) 并赋给 ga1
        ga1 = gamma2(a1);
        # 计算 ua 和 ub
        ua = pow(-1, n-1) / (rn * ga) * pow(x, n);
        ub = rn1 / ga1;
    }
    # 初始化 hm1, r, hmax, hmin, h0
    hm1 = 1.0;
    r = 1.0;
    hmax = 0.0;
    hmin = 1e300;
    h0 = 0.0;
    # 循环迭代求和直到满足条件或达到最大迭代次数
    for (k = 1; k <= 150; k++) {
        # 更新 r
        r = r * (a0 + k - 1) * x / ((n + k) * k);
        # 更新 hm1
        hm1 += r;
        # 计算当前 hm1 的绝对值
        hu1 = fabs(hm1);

        # 更新 hmax 和 hmin
        if (hu1 > hmax) {
            hmax = hu1;
        }
        if (hu1 < hmin) {
            hmin = hu1;
        }
        # 如果满足收敛条件，退出循环
        if (fabs(hm1 - h0) < fabs(hm1) * 1.0e-15) { break; }
        # 更新 h0
        h0 = hm1;
    }

    # 计算 log10(hmax) 并赋给 da1
    da1 = log10(hmax);
    # 初始化 da2
    da2 = 0;

    # 如果 hmin 不等于 0，计算 log10(hmin) 并赋给 da2
    if (hmin != 0) {
        da2 = log10(hmin);
    }

    # 计算 id 并赋值给 *id
    *id = 15 - (int)fabs(da1 - da2);
    # 更新 hm1
    hm1 *= log(x);
    # 初始化 s0
    s0 = 0;

    # 循环计算 s0
    for (m = 1; m <= n; m++) {
        # 根据 b 的正负分支处理
        if (b >= 0) {
            s0 -= 1.0 / m;
        }
        if (b < 0) {
            s0 += (1.0 - a) / (m * (a + m - 1));
        }
    }

    # 初始化 hm2
    hm2 = ps + 2 * el + s0;
    # 初始化 r, hmax, hmin
    r = 1;
    hmax = 0;
    hmin = 1.0e+300;

    # 循环迭代求和直到满足条件或达到最大迭代次数
    for (k = 1; k <= 150; k++) {
        # 初始化 s1, s2
        s1 = 0;
        s2 = 0;

        # 根据 b 的正负分支处理
        if (b > 0) {
            # 计算 s1
            for (m = 1; m <= k; m++) {
                s1 -= (m + 2 * a - 2) / (m * (m + a - 1));
            }
            # 计算 s2
            for (m = 1; m <= n; m++) {
                s2 += 1.0 / (k + m);
            }
        } else {
            # 计算 s1
            for (m = 1; m <= k + n; m++) {
                s1 += (1.0 - a) / (m * (m + a - 1));
            }
            # 计算 s2
            for (m = 1; m <= k; m++) {
                s2 += 1.0 / m;
            }
        }

        # 计算 hw
        hw = 2 * el + ps + s1 - s2;
        # 更新 r
        r = r * (a0 + k - 1) * x / ((n + k) * k);
        # 更新 hm2
        hm2 += r * hw;
        # 计算当前 hm2 的绝对值
        hu2 = fabs(hm2);

        # 更新 hmax 和 hmin
        if (hu2 > hmax) {
            hmax = hu2;
        }
        if (hu2 < hmin) {
            hmin = hu2;
        }

        # 如果满足收敛条件，退出循环
        if (fabs((hm2 - h0) / hm2) < 1.0e-15) {
            break;
        }

        # 更新 h0
        h0 = hm2;
    }

    # 计算 log10(hmax) 并赋给 db1
    db1 = log10(hmax);
    # 初始化 db2
    db2 = 0.0;
    # 如果 hmin 不等于 0，计算 log10(hmin) 并赋给 db2
    if (hmin != 0.0) { db2 = log10(hmin); }
    # 计算 id1 并赋给 id1
    id1 = 15 - (int)fabs(db1 - db2);
    # 如果 id1 小于 *id，更新 *id
    if (id1 < *id) { *id = id1; }
    # 初始化 hm3
    hm3 = 1.0;
    # 根据 n 的值分支处理
    if (n == 0) { hm3 = 0.0; }
    # 初始化 r
    r = 1.0;
    # 循环迭代求和直到满足条件
    for (k = 1; k < n; k++) {
        # 更新 r
        r = r * (a2 + k - 1.0) / ((k - n)*k)*x;
        # 更新 hm3
        hm3 += r;
    }
    # 计算 sa 和 sb
    sa = ua*(hm1 + hm2);
    sb = ub*hm3;
    # 计算 hu
    hu = sa + sb;
    # 初始化 id2
    id2 = 0;
    # 如果 sa 不等于 0，计算 log10(fabs(sa)) 并赋给 id1
    if (sa != 0.0) { id1 = (int)(log10(fabs(sa))); }
    # 如果 hu 不等于 0，计算 log10(fabs(hu)) 并赋给 id2
    if (hu != 0.0) { id2 = (int)(log10(fabs(hu))); }
    # 如果 sa*sb 小于 0，更新 *id
    if (sa*sb < 0.0) { *id -= abs(id1-id2); }
    # 返回 hu
    return hu;
    // 声明变量和数组
    int k, j, m;
    double a1, b1, c, d, f1, f2, g, ga, hu, hu0, hu1, hu2, s, t1, t2, t3, t4;
    // Gaussian-Legendre积分节点和权重，用于积分计算
    static const double t[30] = {
        0.259597723012478e-01, 0.778093339495366e-01, 0.129449135396945e+00, 0.180739964873425e+00,
        0.231543551376029e+00, 0.281722937423262e+00, 0.331142848268448e+00, 0.379670056576798e+00,
        0.427173741583078e+00, 0.473525841761707e+00, 0.518601400058570e+00, 0.562278900753945e+00,
        0.604440597048510e+00, 0.644972828489477e+00, 0.683766327381356e+00, 0.720716513355730e+00,
        0.755723775306586e+00, 0.788693739932264e+00, 0.819537526162146e+00, 0.848171984785930e+00,
        0.874519922646898e+00, 0.898510310810046e+00, 0.920078476177628e+00, 0.939166276116423e+00,
        0.955722255839996e+00, 0.969701788765053e+00, 0.981067201752598e+00, 0.989787895222222e+00,
        0.995840525118838e+00, 0.999210123227436e+00
    };
    static const double w[30] = {
        0.519078776312206e-01, 0.517679431749102e-01, 0.514884515009810e-01, 0.510701560698557e-01,
        0.505141845325094e-01, 0.498220356905502e-01, 0.489955754557568e-01, 0.480370318199712e-01,
        0.469489888489122e-01, 0.457343797161145e-01, 0.443964787957872e-01, 0.429388928359356e-01,
        0.413655512355848e-01, 0.396806954523808e-01, 0.378888675692434e-01, 0.359948980510845e-01,
        0.340038927249464e-01, 0.319212190192963e-01, 0.297524915007890e-01, 0.275035567499248e-01,
        0.251804776215213e-01, 0.227895169439978e-01, 0.203371207294572e-01, 0.178299010142074e-01,
        0.152746185967848e-01, 0.126781664768159e-01, 0.100475571822880e-01, 0.738993116334531e-02,
        0.471272992695363e-02, 0.202681196887362e-02
    };
    // 估计有效数字的初始值
    *id = 9;
    // DLMF 13.4.4, integration up to C=12/X
    // 根据公式计算一些中间值
    a1 = a - 1.0;
    b1 = b - a - 1.0;
    c = 12.0 / x;
    hu0 = 0.0;
    // 循环计算积分，m从10到100，步长为5
    for (m = 10; m <= 100; m += 5) {
        hu1 = 0.0;
        g = 0.5 * c / m;  // 计算步长 g
        d = g;  // 初始化 d 为 g
        // 内层循环，计算积分
        for (j = 1; j < (m + 1); j++) {
            s = 0.0;
            // 对积分区间进行30点 Gauss-Legendre 求积
            for (k = 1; k <= 30; k++) {
                t1 = d + g * t[k-1];
                t2 = d - g * t[k-1];
                // 计算被积函数 f1 和 f2
                f1 = exp(-x*t1) * pow(t1, a1) * pow(1.0 + t1, b1);
                f2 = exp(-x*t2) * pow(t2, a1) * pow(1.0 + t2, b1);
                // 计算积分点的加权和
                s += w[k-1] * (f1 + f2);
            }
            // 计算积分 hu1
            hu1 += s * g;
            d += 2.0 * g;  // 更新 d
        }
        // 判断是否满足精度要求，如果满足则退出循环
        if (fabs(1.0 - hu0 / hu1) < 1.0e-9) { break; }
        hu0 = hu1;  // 更新 hu0
    }

    ga = gamma2(a);  // 计算 gamma2(a)
    hu1 /= ga;  // 归一化 hu1

    // 第二个积分计算，使用 m 从2到10的循环
    for (m = 2; m <= 10; m += 2) {
        hu2 = 0.0;
        g = 0.5 / m;  // 计算步长 g
        d = g;  // 初始化 d 为 g
        // 内层循环，计算积分
        for (j = 1; j <= m; j++) {
            s = 0.0;
            // 对积分区间进行30点 Gauss-Legendre 求积
            for (k = 1; k <= 30; k++) {
                t1 = d + g * t[k-1];
                t2 = d - g * t[k-1];
                // 计算新的变量 t3 和 t4
                t3 = c / (1.0 - t1);
                t4 = c / (1.0 - t2);
                // 计算被积函数 f1 和 f2
                f1 = t3 * t3 / c * exp(-x * t3) * pow(t3, a1) * pow(1.0 + t3, b1);
                f2 = t4 * t4 / c * exp(-x * t4) * pow(t4, a1) * pow(1.0 + t4, b1);
                // 计算积分点的加权和
                s += w[k-1] * (f1 + f2);
            }
            // 计算积分 hu2
            hu2 += s * g;
            d += 2.0 * g;  // 更新 d
        }
        // 判断是否满足精度要求，如果满足则退出循环
        if (fabs(1.0 - hu0 / hu2) < 1.0e-9) { break; }
        hu0 = hu2;  // 更新 hu0
    }

    ga = gamma2(a);  // 计算 gamma2(a)
    hu2 /= ga;  // 归一化 hu2

    hu = hu1 + hu2;  // 计算最终的积分结果 hu
    return hu;  // 返回积分结果 hu
}

// 定义一个内联函数，计算大参数 x 下的广义超几何函数 U(a,b,x)
inline double chgul(double x, double a, double b, int *id) {

    // =======================================================
    // Purpose: Compute the confluent hypergeometric function
    //          U(a,b,x) for large argument x
    // Input  : a  --- Parameter
    //          b  --- Parameter
    //          x  --- Argument
    // Output:  HU --- U(a,b,x)
    //          ID --- Estimated number of significant digits
    // =======================================================

    int il1, il2, k, nm;
    double aa, hu, r, r0 = 0.0, ra = 0.0;

    *id = -100;  // 设定初始值为 -100
    aa = a - b + 1.0;  // 计算 aa = a - b + 1.0
    il1 = (a == (int)a) && (a <= 0.0);  // 判断是否满足条件 il1
    il2 = (aa == (int)aa) && (aa <= 0.0);  // 判断是否满足条件 il2
    nm = 0;  // 初始化 nm 为 0
    if (il1) { nm = (int)fabs(a); }  // 若 il1 成立，则 nm 取整数部分的绝对值
    if (il2) { nm = (int)fabs(aa); }  // 若 il2 成立，则 nm 取整数部分的绝对值

    // IL1: DLMF 13.2.7 with k=-s-a
    // IL2: DLMF 13.2.8
    if (il1 || il2) {
        // 如果 il1 或 il2 成立，执行以下内容
        hu = 1.0;  // 初始化 hu 为 1.0
        r = 1.0;   // 初始化 r 为 1.0
        for (k = 1; k <= nm; k++) {
            r = -r*(a + k - 1.0)*(a - b + k) / (k*x);  // 更新 r 的值
            hu += r;  // 更新 hu 的值
        }
        hu *= pow(x, -a);  // 计算 hu 的最终值
        *id = 10;  // 设定 ID 为 10
    } else {
        // 如果 il1 和 il2 都不成立，执行以下内容
        hu = 1.0;  // 初始化 hu 为 1.0
        r = 1.0;   // 初始化 r 为 1.0
        for (k = 1; k <= 25; k++) {
            r = -r*(a + k - 1.0)*(a - b + k) / (k*x);  // 更新 r 的值
            ra = fabs(r);  // 计算 r 的绝对值
            // 判断终止条件
            if (((k > 5) && (ra >= r0)) || (ra < 1e-15)) { break; }
            r0 = ra;  // 更新 r0 的值
            hu += r;  // 更新 hu 的值
        }
        *id = (int)fabs(log10(ra));  // 计算 ID 的值
        hu *= pow(x, -a);  // 计算 hu 的最终值
    }
    return hu;  // 返回计算结果 hu
}

// 定义一个内联函数，计算小参数 x 下的广义超几何函数 U(a,b,x)
inline double chgus(double x, double a, double b, int *id) {

    // ======================================================
    // Purpose: Compute confluent hypergeometric function
    //          U(a,b,x) for small argument x
    // Input  : a  --- Parameter
    //          b  --- Parameter ( b <> 0,-1,-2,...)
    //          x  --- Argument
    // Output:  HU --- U(a,b,x)
    //          ID --- Estimated number of significant digits
    // Routine called: GAMMA2 for computing gamma function
    // ======================================================

    // DLMF 13.2.42 with prefactors rewritten according to
    // DLMF 5.5.3, M(a, b, x) with DLMF 13.2.2
    int j;
    double d1, d2, ga, gb, gab, gb2, h0, hmax, hmin, hu, hu0, hua, r1, r2;
    const double pi = 3.141592653589793;

    *id = 100;  // 设定初始值为 100
    ga = gamma2(a);  // 计算 gamma2(a)
    gb = gamma2(b);  // 计算 gamma2(b)
    gab = gamma2(1.0 + a - b);  // 计算 gamma2(1.0 + a - b)
    gb2 = gamma2(2.0 - b);  // 计算 gamma2(2.0 - b)
    hu0 = pi / sin(pi*b);  // 计算 hu0
    r1 = hu0 / (gab*gb);  // 计算 r1
    r2 = hu0*pow(x, 1.0 - b) / (ga*gb2);  // 计算 r2
    hu = r1 - r2;  // 计算 hu 的初值
    hmax = 0.0;  // 初始化 hmax 为 0.0
    hmin = 1e300;  // 初始化 hmin 为一个较大的数值
    h0 = 0.0;  // 初始化 h0 为 0.0
    for (j = 1; j < 151; j++) {
        r1 = r1*(a + j - 1.0) / (j*(b + j - 1.0))*x;  // 更新 r1 的值
        r2 = r2*(a - b + j) / (j*(1.0 - b + j))*x;  // 更新 r2 的值
        hu += r1 - r2;  // 更新 hu 的值
        hua = fabs(hu);  // 计算 hu 的绝对值
        if (hua > hmax) { hmax = hua; }  // 更新 hmax 的值
        if (hua < hmin) { hmin = hua; }  // 更新 hmin 的值
        // 判断终止条件
        if (fabs(hu - h0) < fabs(hu)*1e-15) { break; }
        h0 = hu;  // 更新 h0 的值
    }
    d1 = log10(hmax);  // 计算 d1
    d2 = 0.0;  // 初始化 d2 为 0.0
    if (hmin != 0.0) { d2 = log10(hmin); }  // 若 hmin 非零，则计算 d2
    *id = 15 - (d1 - d2 < 0 ? d2 - d1 : d1 - d2);  // 计算 ID 的值
    return hu;  // 返回计算结果 hu
}
// ==================================================
// Purpose: Compute the parabolic cylinder function Dn(z)
//          and its derivative Dn'(z) for a complex argument
// Input:   z --- Complex argument of Dn(z)
//          n --- Order of Dn(z) ( n=0,±1,±2,… )
// Output:  CPB(|n|) --- Dn(z)
//          CPD(|n|) --- Dn'(z)
// Routines called:
//      (1) CPDSA for computing Dn(z) for a small |z|
//      (2) CPDLA for computing Dn(z) for a large |z|
// ==================================================

inline void cpbdn(int n, std::complex<double> z, std::complex<double> *cpb, std::complex<double> *cpd) {

    int n0, n1, nm1;
    double a0, x;
    std::complex<double> ca0, cf, cf0, cf1, cfa, cfb, cs0, z1;
    const double pi = 3.141592653589793;

    x = z.real();  // Extract the real part of z
    a0 = std::abs(z);  // Compute the absolute value of z
    ca0 = std::exp(-0.25 * z * conj(z));  // Compute exp(-0.25*z*conj(z))
    n0 = 0;

    if (n >= 0) {  // If n is non-negative
        cf0 = ca0;
        cf1 = z * ca0;

        cpb[0] = cf0;  // Initialize CPB(|n|) with Dn(z)
        cpb[1] = cf1;  // Initialize CPB(|n|) with Dn'(z)

        for (int k = 2; k <= n; ++k) {
            cf = z * cf1 - (k - 1.0) * cf0;  // Recurrence relation for Dn(z)
            cpb[k] = cf;  // Store Dn(z) in CPB(|n|)
            cf0 = cf1;
            cf1 = cf;
        }
    } else {  // If n is negative
        n0 = -n;

        if (x <= 0.0 || a0 == 0.0) {  // If z is non-positive or |z| is zero
            cf0 = ca0;
            cpb[0] = cf0;  // Initialize CPB(|n|) with Dn(z)

            z1 = -z;  // Compute -z

            if (a0 <= 7.0) {
                cpb[1] = cpdsa(-1, z1);  // Compute Dn'(z) using CPDSA for small |z|
            } else {
                cpb[1] = cpdla(-1, z1);  // Compute Dn'(z) using CPDLA for large |z|
            }

            cf1 = std::sqrt(2.0 * pi) / ca0 - cpb[1];  // Compute Dn(z) using the asymptotic formula
            cpb[1] = cf1;  // Store Dn(z) in CPB(|n|)

            for (int k = 2; k < n0; ++k) {
                cf = (-z * cf1 + cf0) / (k - 1.0);  // Recurrence relation for Dn(z)
                cpb[k] = cf;  // Store Dn(z) in CPB(|n|)
                cf0 = cf1;
                cf1 = cf;
            }
        } else if (a0 <= 3.0) {  // If |z| is less than or equal to 3.0
            cpb[n0] = cpdsa(-n0, z);  // Compute Dn(z) using CPDSA
            n1 = n0 + 1;
            cpb[n1] = cpdsa(-n1, z);  // Compute Dn(z) using CPDSA

            nm1 = n0 - 1;
            for (int k = nm1; k >= 0; --k) {
                cf = z * cpb[n0] + (k + 1.0) * cpb[n1];  // Recurrence relation for Dn(z)
                cpb[k] = cf;  // Store Dn(z) in CPB(|n|)
                cpb[n1] = cpb[n0];
                cpb[n0] = cf;
            }
        } else {  // If |z| is greater than 3.0
            int m = 100 + abs(n);
            cfa = 0.0;
            cfb = 1.0e-30;

            for (int k = m; k >= 0; --k) {
                cf = z * cfb + (k + 1.0) * cfa;  // Recurrence relation for Dn(z)

                if (k <= n0) {
                    cpb[k] = cf;  // Store Dn(z) in CPB(|n|)
                }

                cfa = cfb;
                cfb = cf;
            }

            cs0 = ca0 / cfb;  // Compute scaling factor

            for (int k = 0; k <= n0; ++k) {
                cpb[k] = cs0 * cpb[k];  // Scale Dn(z) and store in CPB(|n|)
            }
        }
    }

    cpd[0] = -0.5 * z * cpb[0];  // Compute Dn'(z) for n=0

    if (n >= 0) {  // If n is non-negative
        for (int k = 1; k <= n; ++k) {
            cpd[k] = -0.5 * z * cpb[k] + static_cast<double>(k) * cpb[k - 1];  // Recurrence relation for Dn'(z)
        }
    } else {  // If n is negative
        for (int k = 1; k < n0; ++k) {
            cpd[k] = 0.5 * z * cpb[k] - cpb[k - 1];  // Recurrence relation for Dn'(z)
        }
    }
}
    // 初始化循环变量 k
    int k;
    // 定义复数类型的变量 cb0, cr, cdn
    std::complex<double> cb0, cr, cdn;

    // 计算 cb0 = z^n * exp(-0.25*z*z)
    cb0 = std::pow(z, n) * std::exp(-0.25 * z * z);
    
    // 初始化 cr = 1.0
    cr = 1.0;
    
    // 初始化 cdn = 1.0
    cdn = 1.0;

    // 循环计算复杂的递推关系，最多执行 16 次
    for (k = 1; k <= 16; k++) {
        // 更新 cr 根据递推关系
        cr = -0.5 * cr * (2.0 * k - n - 1.0) * (2.0 * k - n - 2.0) / (static_cast<double>(k) * z * z);
        
        // 累加到 cdn
        cdn += cr;
        
        // 如果 cr 的绝对值小于 cdn 的绝对值乘以 1e-12，则跳出循环
        if (std::abs(cr) < std::abs(cdn) * 1e-12) { break; }
    }
    
    // 返回 cdn * cb0 作为结果
    return cdn * cb0;
}

// 定义一个内联函数，计算复数抛物柱函数 Dn(z) 的小参数近似
// 输入参数:
//   z   --- D(z) 的复数参数
//   n   --- D(z) 的阶数 (n = 0,-1,-2,...)
// 输出:
//   CDN --- Dn(z)
// 调用的函数:
//   GAIH 用于计算 Г(x)，其中 x=n/2 (n=1,2,...)
inline std::complex<double> cpdsa(int n, std::complex<double> z) {

    // 常量定义
    const double eps = 1.0e-15;
    const double pi = 3.141592653589793;
    const double sq2 = sqrt(2.0);

    // 变量声明
    int m;
    double va0, pd, vm, vt, xn;
    std::complex<double> ca0, cb0, cdn, cr, cdw, g0, g1, ga0, gm;

    // 计算 ca0，即 exp(-0.25 * z * z)
    ca0 = std::exp(-0.25 * z * z);

    // 计算 va0，即 0.5 * (1.0 - n)
    va0 = 0.5 * (1.0 - n);

    // 处理 n == 0 的情况
    if (n == 0.0) {
        cdn = ca0;
    } else {
        // 处理 z 的绝对值为 0 的情况
        if (std::abs(z) == 0.0) {
            if ((va0 <= 0.0) && (va0 == (int)va0)) {
                cdn = 0.0;
            } else {
                // 调用 gaih(va0) 计算结果 ga0
                ga0 = gaih(va0);
                // 计算 pd
                pd = sqrt(pi) / (pow(2.0, -0.5 * n) * ga0.real());
                cdn = pd;
            }
        } else {
            // 处理一般情况下的计算
            xn = -n;
            // 调用 gaih(xn) 计算结果 g1
            g1 = gaih(xn);
            // 计算 cb0
            cb0 = pow(2.0, -0.5 * n - 1.0) * ca0 / g1;
            // 计算 vt
            vt = -0.5 * n;
            // 调用 gaih(vt) 计算结果 g0
            g0 = gaih(vt);
            cdn = g0;
            cr = std::complex<double>(1.0, 0.0);

            // 迭代计算直到满足条件或达到最大迭代次数 (250)
            for (m = 1; m <= 250; m++) {
                vm = 0.5 * (m - n);
                // 调用 gaih(vm) 计算结果 gm
                gm = gaih(vm);
                cr = -cr * sq2 * z / static_cast<double>(m);
                cdw = gm * cr;
                cdn += cdw;
                // 判断是否满足迭代终止条件
                if (std::abs(cdw) < std::abs(cdn) * eps) {
                    break;
                }
            }
            cdn *= cb0;
        }
    }
    // 返回结果 cdn
    return cdn;
}

// 定义一个内联函数，计算 Mathieu 函数的初始特征值 A0
// 输入参数:
//   kd  --- 
//   m   --- Mathieu 函数的阶数
//   q   --- Mathieu 函数的参数
// 输出:
//   A0 --- 初始特征值
// 调用的函数:
//   CVQM 用于计算 q ≤ 3*m 的初始特征值
//   CVQL 用于计算 q ≥ m*m 的初始特征值
inline double cv0(double kd, double m, double q) {

    // 常量定义
    double a0 = 0.0, q2 = q * q;

    // 处理 m == 0 的情况
    if (m == 0) {
        // 根据 q 的大小选择计算方式
        if (q <= 1.0) {
            a0 = (((0.0036392 * q2 - 0.0125868) * q2 + 0.0546875) * q2 - 0.5) * q2;
        } else if (q <= 10.0) {
            a0 = ((3.999267e-3 * q - 9.638957e-2) * q - 0.88297) * q + 0.5542818;
        } else {
            // 调用 cvql(kd, m, q) 计算结果
            a0 = cvql(kd, m, q);
        }

            a0 = cvql(kd, m, q);
        }
    }

    // 返回结果 a0
    return a0;
}
    } else if (m == 1) {
        // 如果 m 等于 1，则根据 q 和 kd 的值计算 a0
        if ((q <= 1.0) && (kd == 2)) {
            // 根据特定公式计算 a0
            a0 = (((-6.51e-4 * q - 0.015625) * q - 0.125) * q + 1.0) * q + 1.0;
        } else if (q <= 1.0 && kd == 3) {
            // 根据特定公式计算 a0
            a0 = (((-6.51e-4 * q + 0.015625) * q - 0.125) * q - 1.0) * q + 1.0;
        } else if (q <= 10.0 && kd == 2) {
            // 根据特定公式计算 a0
            a0 = (((-4.94603e-4 * q + 1.92917e-2) * q - 0.3089229) * q + 1.33372) * q + 0.811752;
        } else if (q <= 10.0 && kd == 3) {
            // 根据特定公式计算 a0
            a0 = ((1.971096e-3 * q - 5.482465e-2) * q - 1.152218) * q + 1.10427;
        } else {
            // 如果条件不满足上述任何一个，调用 cvql 函数计算 a0
            a0 = cvql(kd, m, q);
        }
    } else if (m == 2) {
        // 如果 m 等于 2，则根据 q 和 kd 的值计算 a0
        if (q <= 1.0 && kd == 1) {
            // 根据特定公式计算 a0
            a0 = (((-0.0036391 * q2 + 0.0125888) * q2 - 0.0551939) * q2 + 0.416667) * q2 + 4.0;
        } else if (q <= 1.0 && kd == 4) {
            // 根据特定公式计算 a0
            a0 = (0.0003617 * q2 - 0.0833333) * q2 + 4.0;
        } else if (q <= 15.0 && kd == 1) {
            // 根据特定公式计算 a0
            a0 = (((3.200972e-4 * q - 8.667445e-3) * q - 1.829032e-4) * q + 0.9919999) * q + 3.3290504;
        } else if (q <= 10.0 && kd == 4) {
            // 根据特定公式计算 a0
            a0 = ((2.38446e-3 * q - 0.08725329) * q - 4.732542e-3) * q + 4.00909;
        } else {
            // 如果条件不满足上述任何一个，调用 cvql 函数计算 a0
            a0 = cvql(kd, m, q);
        }
    } else if (m == 3) {
        // 如果 m 等于 3，则根据 q 和 kd 的值计算 a0
        if (q <= 1.0 && kd == 2) {
            // 根据特定公式计算 a0
            a0 = (((6.348e-4 * q + 0.015625) * q + 0.0625) * q2 + 9.0);
        } else if (q <= 1.0 && kd == 3) {
            // 根据特定公式计算 a0
            a0 = (((6.348e-4 * q - 0.015625) * q + 0.0625) * q2 + 9.0);
        } else if (q <= 20.0 && kd == 2) {
            // 根据特定公式计算 a0
            a0 = (((3.035731e-4 * q - 1.453021e-2) * q + 0.19069602) * q - 0.1039356) * q + 8.9449274;
        } else if (q <= 15.0 && kd == 3) {
            // 根据特定公式计算 a0
            a0 = ((9.369364e-5 * q - 0.03569325) * q + 0.2689874) * q + 8.771735;
        } else {
            // 如果条件不满足上述任何一个，调用 cvql 函数计算 a0
            a0 = cvql(kd, m, q);
        }
    } else if (m == 4) {
        // 如果 m 等于 4，则根据 q 和 kd 的值计算 a0
        if (q <= 1.0 && kd == 1) {
            // 根据特定公式计算 a0
            a0 = ((-2.1e-6 * q2 + 5.012e-4) * q2 + 0.0333333) * q2 + 16.0;
        } else if (q <= 1.0 && kd == 4) {
            // 根据特定公式计算 a0
            a0 = ((3.7e-6 * q2 - 3.669e-4) * q2 + 0.0333333) * q2 + 16.0;
        } else if (q <= 25.0 && kd == 1) {
            // 根据特定公式计算 a0
            a0 = (((1.076676e-4 * q - 7.9684875e-3) * q + 0.17344854) * q - 0.5924058) * q + 16.620847;
        } else if (q <= 20.0 && kd == 4) {
            // 根据特定公式计算 a0
            a0 = ((-7.08719e-4 * q + 3.8216144e-3) * q + 0.1907493) * q + 15.744;
        } else {
            // 如果条件不满足上述任何一个，调用 cvql 函数计算 a0
            a0 = cvql(kd, m, q);
        }
    } else if (m == 5) {
        // 如果 m 等于 5，则根据 q 和 kd 的值计算 a0
        if (q <= 1.0 && kd == 2) {
            // 根据特定公式计算 a0
            a0 = ((6.8e-6 * q + 1.42e-5) * q2 + 0.0208333) * q2 + 25.0;
        } else if (q <= 1.0 && kd == 3) {
            // 根据特定公式计算 a0
            a0 = ((-6.8e-6 * q + 1.42e-5) * q2 + 0.0208333) * q2 + 25.0;
        } else if (q <= 35.0 && kd == 2) {
            // 根据特定公式计算 a0
            a0 = (((2.238231e-5 * q - 2.983416e-3) * q + 0.10706975) * q - 0.600205) * q + 25.93515;
        } else if (q <= 25.0 && kd == 3) {
            // 根据特定公式计算 a0
            a0 = ((-7.425364e-4 * q + 2.18225e-2) * q + 4.16399e-2) * q + 24.897;
        } else {
            // 如果条件不满足上述任何一个，调用 cvql 函数计算 a0
            a0 = cvql(kd, m, q);
        }
    } else if (m == 6) {
        # 根据给定的条件判断，如果 m 等于 6，则执行以下逻辑
        if (q <= 1.0) {
            # 如果 q 小于等于 1.0，计算并赋值给 a0
            a0 = (4e-6 * q ** 2 + 0.0142857) * q ** 2 + 36.0;
        } else if (q <= 40.0 && kd == 1) {
            # 如果 q 小于等于 40.0 并且 kd 等于 1，计算并赋值给 a0
            a0 = (((-1.66846e-5 * q + 4.80263e-4) * q + 2.53998e-2) * q - 0.181233) * q + 36.423;
        } else if (q <= 35.0 && kd == 4) {
            # 如果 q 小于等于 35.0 并且 kd 等于 4，计算并赋值给 a0
            a0 = ((-4.57146e-4 * q + 2.16609e-2) * q - 2.349616e-2) * q + 35.99251;
        } else {
            # 其他情况调用函数 cvql(kd, m, q)，并将结果赋值给 a0
            a0 = cvql(kd, m, q);
        }
    } else if (m == 7) {
        # 根据给定的条件判断，如果 m 等于 7，则执行以下逻辑
        if (q <= 10.0) {
            # 如果 q 小于等于 10.0，调用函数 cvqm(m, q)，并将结果赋值给 a0
            a0 = cvqm(m, q);
        } else if (q <= 50.0 && kd == 2) {
            # 如果 q 小于等于 50.0 并且 kd 等于 2，计算并赋值给 a0
            a0 = (((-1.411114e-5 * q + 9.730514e-4) * q - 3.097887e-3) * q + 3.533597e-2) * q + 49.0547;
        } else if (q <= 40.0 && kd == 3) {
            # 如果 q 小于等于 40.0 并且 kd 等于 3，计算并赋值给 a0
            a0 = ((-3.043872e-4 * q + 2.05511e-2) * q - 9.16292e-2) * q + 49.19035;
        } else {
            # 其他情况调用函数 cvql(kd, m, q)，并将结果赋值给 a0
            a0 = cvql(kd, m, q);
        }
    } else if (m >= 8) {
        # 根据给定的条件判断，如果 m 大于等于 8，则执行以下逻辑
        if (q <= 3*m) {
            # 如果 q 小于等于 3*m，调用函数 cvqm(m, q)，并将结果赋值给 a0
            a0 = cvqm(m, q);
        } else if (q > m * m) {
            # 如果 q 大于 m*m，调用函数 cvql(kd, m, q)，并将结果赋值给 a0
            a0 = cvql(kd, m, q);
        } else {
            # 根据 m 和 kd 的不同组合计算并赋值给 a0
            if (m == 8 && kd == 1) {
                a0 = (((8.634308e-6 * q - 2.100289e-3) * q + 0.169072) * q - 4.64336) * q + 109.4211;
            } else if (m == 8 && kd == 4) {
                a0 = ((-6.7842e-5 * q + 2.2057e-3) * q + 0.48296) * q + 56.59;
            } else if (m == 9 && kd == 2) {
                a0 = (((2.906435e-6 * q - 1.019893e-3) * q + 0.1101965) * q - 3.821851) * q + 127.6098;
            } else if (m == 9 && kd == 3) {
                a0 = ((-9.577289e-5 * q + 0.01043839) * q + 0.06588934) * q + 78.0198;
            } else if (m == 10 && kd == 1) {
                a0 = (((5.44927e-7 * q - 3.926119e-4) * q + 0.0612099) * q - 2.600805) * q + 138.1923;
            } else if (m == 10 && kd == 4) {
                a0 = ((-7.660143e-5 * q + 0.01132506) * q - 0.09746023) * q + 99.29494;
            } else if (m == 11 && kd == 2) {
                a0 = (((-5.67615e-7 * q + 7.152722e-6) * q + 0.01920291) * q - 1.081583) * q + 140.88;
            } else if (m == 11 && kd == 3) {
                a0 = ((-6.310551e-5 * q + 0.0119247) * q - 0.2681195) * q + 123.667;
            } else if (m == 12 && kd == 1) {
                a0 = (((-2.38351e-7 * q - 2.90139e-5) * q + 0.02023088) * q - 1.289) * q + 171.2723;
            } else if (m == 12 && kd == 4) {
                a0 = (((3.08902e-7 * q - 1.577869e-4) * q + 0.0247911) * q - 1.05454) * q + 161.471;
            }
        }
    }
    # 返回计算得到的值 a0
    return a0;
inline double cvf(int kd, int m, double q, double a, int mj) {

    // ======================================================
    // Purpose: Compute the value of F for characteristic
    //          equation of Mathieu functions
    // Input :  m --- Order of Mathieu functions
    //          q --- Parameter of Mathieu functions
    //          a --- Characteristic value of Mathieu functions
    //          mj --- Index of the characteristic value
    //                 computation (1 or 2)
    //          KD --- Case code
    //                 KD=1 for cem(x,q)  ( m = 0,2,4,...)
    //                 KD=2 for cem(x,q)  ( m = 1,3,5,...)
    //                 KD=3 for sem(x,q)  ( m = 1,3,5,...)
    //                 KD=4 for sem(x,q)  ( m = 2,4,6,...)
    // Output:  F --- Value of the characteristic equation
    // Routines called:
    //       (1) CVFQ for computing F for Mathieu functions
    // ======================================================

    double f;

    // Call CVFQ to compute the value of F for Mathieu functions
    f = cvfq(kd, m, q, a, mj);

    return f;
}



}


注释：
- `cvf` 函数用于计算 Mathieu 函数特征方程的 F 值。
- 输入参数包括 `m`（Mathieu 函数的阶数）、`q`（Mathieu 函数的参数）、`a`（Mathieu 函数的特征值）、`mj`（特征值计算的索引）、`KD`（用于区分不同 Mathieu 函数类型的编码）。
- 返回值 `F` 表示特征方程的值。
- 函数内部调用了 `cvfq` 函数来计算 Mathieu 函数的 F 值。
    // 初始化变量和参数设置
    int j, ic = m / 2, l = 0, l0 = 0, j0 = 2;  // 定义整型变量和初始化ic、l、l0、j0
    int jf = ic;  // 设置jf等于ic
    double t0 = 0.0, t1 = 0.0, t2 = 0.0, b = a, f;  // 定义双精度浮点变量和初始化t0、t1、t2、b、f

    // 根据条件设置j0和l0的值
    if (kd == 1) {
        j0 = 3;
        l0 = 2;
    }
    // 如果kd为2或3，设置l为1
    if ((kd == 2) || (kd == 3)) { l = 1; }
    // 如果kd为4，设置jf为ic-1
    if (kd == 4) { jf = ic-1; }

    // 循环计算t1的值，逆序迭代j从mj到ic+1
    for (j = mj; j >= ic + 1; j--) {
        t1 = -q*q / (pow(2.0*j + l, 2) - b + t1);  // 使用q、j、l、b更新t1
    }

    // 根据m的值执行条件分支
    if (m <= 2) {
        // 若kd为1且m为0，更新t1
        if ((kd == 1) && (m == 0)) { t1 += t1; }
        // 若kd为1且m为2，更新t1
        if ((kd == 1) && (m == 2)) { t1 = -2.0*q*q / (4.0 - b + t1) - 4.0; }
        // 若kd为2且m为1，更新t1
        if ((kd == 2) && (m == 1)) { t1 += q; }
        // 若kd为3且m为1，更新t1
        if ((kd == 3) && (m == 1)) { t1 -= q; }
    } else {
        // 根据kd的值计算t0的值
        if (kd == 1) { t0 = 4.0 - b + 2.0*q*q / b; }
        if (kd == 2) { t0 = 1.0 - b + q; }
        if (kd == 3) { t0 = 1.0 - b - q; }
        if (kd == 4) { t0 = 4.0 - b; }
        // 根据kd的值计算t2的值，迭代j从j0到jf
        t2 = -q*q / t0;
        for (j = j0; j <= jf; j++) {
            t2 = -q*q / (pow(2.0*j - l - l0, 2.0) - b + t2);  // 使用q、j、l、l0、b更新t2
        }
    }
    // 计算并返回f的值
    f = pow(2.0*ic + l, 2) + t1 + t2 - b;
    return f;
// 定义一个内联函数，计算Mathieu函数的特征值A0，适用于q ≥ 3m的情况
// 输入：m --- Mathieu函数的阶数
//      q --- Mathieu函数的参数
// 输出：A0 --- 初始特征值
inline double cvql(int kd, int m, double q) {

    // 初始化变量
    double a0, w, w2, w3, w4, w6, d1, d2, d3, d4, c1, p1, p2, cv1, cv2;

    w = 0.0;
    // 根据kd的不同取值计算w的值
    if ((kd == 1) || (kd == 2)) { w = 2.0*m + 1.0; }
    if ((kd == 3) || (kd == 4)) { w = 2.0*m - 1.0; }
    
    // 计算w的幂次方
    w2 = w*w;
    w3 = w*w2;
    w4 = w2*w2;
    w6 = w2*w4;
    
    // 计算d1, d2, d3, d4的值
    d1 = 5.0 + 34.0 / w2 + 9.0 / w4;
    d2 = (33.0 + 410.0 / w2 + 405.0 / w4) / w;
    d3 = (63.0 + 1260.0 / w2 + 2943.0 / w4 + 486.0 / w6) / w2;
    d4 = (527.0 + 15617.0 / w2 + 69001.0 / w4 + 41607.0 / w6) / w3;
    
    // 初始化常数c1和计算p2, p1的值
    c1 = 128.0;
    p2 = q / w4;
    p1 = sqrt(p2);
    
    // 计算cv1和cv2的值
    cv1 = -2.0*q + 2.0*w*sqrt(q) - (w2 + 1.0) / 8.0;
    cv2 = (w + 3.0 / w) + d1 / (32.0 * p1) + d2 / (8.0 * c1 * p2);
    cv2 = cv2 + d3 / (64.0 * c1 * p1 * p2) + d4 / (16.0 * c1 * c1 * p2 * p2);
    
    // 计算A0的值
    a0 = cv1 - cv2 / (c1 * p1);
    
    // 返回计算结果A0
    return a0;
}

// 定义一个内联函数，计算Mathieu函数的特征值A0，适用于q ≤ m*m的情况
// 输入：m --- Mathieu函数的阶数
//      q --- Mathieu函数的参数
// 输出：A0 --- 初始特征值
inline double cvqm(int m, double q) {

    // 初始化变量
    double hm1, hm3, hm5, a0;
    
    // 计算hm1, hm3, hm5的值
    hm1 = 0.5 * q / (m * m - 1.0);
    hm3 = 0.25 * pow(hm1, 3) / (m * m - 4.0);
    hm5 = hm1 * hm3 * q / ((m * m - 1.0) * (m * m - 9.0));
    
    // 计算A0的值
    a0 = m * m + q * (hm1 + (5.0 * m * m + 7.0) * hm3 + (9.0 * pow(m, 4) + 58.0 * m * m + 29.0) * hm5);
    
    // 返回计算结果A0
    return a0;
}

// 定义一个内联函数，计算复贝塞尔函数Y0(z), Y1(z)及其导数
// 输入：z  --- 复数参数
//       kf --- 函数选择代码
//             kf=0 为ZF=Y0(z)和ZD=Y0'(z)
//             kf=1 为ZF=Y1(z)和ZD=Y1'(z)
//             kf=2 为ZF=Y1'(z)和ZD=Y1''(z)
// 输出：zf --- Y0(z)或Y1(z)或Y1'(z)
//       zd --- Y0'(z)或Y1'(z)或Y1''(z)
inline void cy01(int kf, std::complex<double> z, std::complex<double> *zf, std::complex<double> *zd) {

    // 初始化变量
    int k, k0;
    double a0, w0, w1;
    std::complex<double> cr, cp, cp0, cq0, cu, cp1, cq1, cbj0, cbj1,\
                   cby0, cby1, cdy0, cdy1, cs, ct1, ct2, z1, z2;

    // 定义常数和复数单位
    const double pi = 3.141592653589793;
    const double el = 0.5772156649015329;
    const double rp2 = 2.0 / pi;
    const std::complex<double> ci(0.0, 1.0);
    // 定义常量数组 a 包含12个双精度浮点数
    static const double a[12] = {-0.703125e-01,        0.112152099609375, -0.5725014209747314,
                                  0.6074042001273483, -0.1100171402692467, 0.3038090510922384,
                                  -0.1188384262567832, 0.6252951493434797, -0.4259392165047669,
                                  0.3646840080706556, -0.3833534661393944, 0.4854014686852901};

    // 定义常量数组 b 包含12个双精度浮点数
    static const double b[12] = { 0.732421875e-01, -0.2271080017089844, 0.1727727502584457,
                                 -0.2438052969955606, 0.5513358961220206, -0.1825775547429318,
                                  0.8328593040162893, -0.5006958953198893, 0.3836255180230433,
                                 -0.3649010818849833, 0.4218971570284096, -0.5827244631566907};

    // 定义常量数组 a1 包含12个双精度浮点数
    static const double a1[12] = { 0.1171875,          -0.144195556640625,   0.6765925884246826,
                                  -0.6883914268109947,  0.1215978918765359, -0.3302272294480852,
                                   0.1276412726461746, -0.6656367718817688,  0.4502786003050393,
                                  -0.3833857520742790,  0.4011838599133198, -0.5060568503314727};

    // 定义常量数组 b1 包含12个双精度浮点数
    static const double b1[12] = {-0.1025390625,        0.2775764465332031, -0.1993531733751297,
                                   0.2724882731126854, -0.6038440767050702,  0.1971837591223663,
                                  -0.8902978767070678,  0.5310411010968522, -0.4043620325107754,
                                   0.3827011346598605, -0.4406481417852278,  0.6065091351222699};

    // 计算 z 的绝对值
    a0 = std::abs(z);

    // z 的实部赋值给 z1
    z1 = z;

    // z 的平方赋值给 z2
    z2 = z * z;

    // 如果 a0 为零
    if (a0 == 0.0) {
        // 初始化复数对象 cbj0, cbj1, cby0, cby1, cdy0, cdy1
        cbj0 = std::complex<double>(1.0, 0.0);
        cbj1 = std::complex<double>(0.0, 0.0);
        cby0 = std::complex<double>(-1e300, 0.0);
        cby1 = std::complex<double>(-1e300, 0.0);
        cdy0 = std::complex<double>( 1e300, 0.0);
        cdy1 = std::complex<double>( 1e300, 0.0);

        // 根据 kf 的值选择设置 zf 和 zd 的值
        if (kf == 0) {
            *zf = cby0;
            *zd = cdy0;
        } else if (kf == 1) {
            *zf = cby1;
            *zd = cdy1;
        } else if (kf == 2) {
            *zf = cdy1;
            *zd = -cdy1 / z - (1.0 - 1.0 / (z * z)) * cby1;
        }
        // 函数返回
        return;
    }

    // 如果 z 的实部小于零，则将 z1 变为 z 的相反数
    if (z.real() < 0.0) {
        z1 = -z;
    }
    // 如果 a0 <= 12.0，执行以下代码块
    if (a0 <= 12.0) {
        // 初始化第一个贝塞尔函数 J0 的初值为 1.0
        cbj0 = std::complex<double>(1.0, 0.0);
        // 初始化 cr 为复数 1.0
        cr = std::complex<double>(1.0, 0.0);
        // 计算贝塞尔函数 J0 的级数展开直到收敛或达到最大迭代次数 40
        for (k = 1; k <= 40; k++) {
            // 更新 cr，计算级数项，使用 static_cast<double> 强制类型转换
            cr = -0.25 * cr * z2 / static_cast<double>(k * k);
            // 更新 cbj0，累加级数项
            cbj0 += cr;
            // 如果级数项的绝对值小于当前总和的 1.0e-15 倍，则停止迭代
            if (std::abs(cr) < std::abs(cbj0) * 1.0e-15) break;
        }

        // 初始化第一个贝塞尔函数 J1 的初值为 1.0
        cbj1 = std::complex<double>(1.0, 0.0);
        // 重新初始化 cr 为复数 1.0
        cr = std::complex<double>(1.0, 0.0);
        // 计算贝塞尔函数 J1 的级数展开直到收敛或达到最大迭代次数 40
        for (k = 1; k <= 40; k++) {
            // 更新 cr，计算级数项
            cr = -0.25 * cr * z2 / (k * (k + 1.0));
            // 更新 cbj1，累加级数项
            cbj1 += cr;
            // 如果级数项的绝对值小于当前总和的 1.0e-15 倍，则停止迭代
            if (std::abs(cr) < std::abs(cbj1) * 1.0e-15) break;
        }

        // 计算 cbj1 的最终系数
        cbj1 *= 0.5 * z1;
        
        // 初始化 w0 为 0.0
        w0 = 0.0;
        // 重新初始化 cr 和 cs 为复数 1.0 和 0.0
        cr = std::complex<double>(1.0, 0.0);
        cs = std::complex<double>(0.0, 0.0);
        // 计算复杂的级数和，直到收敛或达到最大迭代次数 40
        for (k = 1; k <= 40; k++) {
            // 更新 w0，累加级数和的分母部分
            w0 += 1.0 / k;
            // 更新 cr，计算级数项
            cr = -0.25 * cr / static_cast<double>(k * k) * z2;
            // 计算 cp，级数项与 w0 的乘积
            cp = cr * w0;
            // 更新 cs，累加级数项
            cs += cp;
            // 如果级数项的绝对值小于当前总和的 1.0e-15 倍，则停止迭代
            if (std::abs(cp) < std::abs(cs) * 1.0e-15) break;
        }

        // 计算第一个修正的贝塞尔函数 Y0
        cby0 = rp2 * (std::log(z1 / 2.0) + el) * cbj0 - rp2 * cs;
        
        // 初始化 w1 为 0.0
        w1 = 0.0;
        // 重新初始化 cr 和 cs 为复数 1.0
        cr = 1.0;
        cs = 1.0;
        // 计算复杂的级数和，直到收敛或达到最大迭代次数 40
        for (k = 1; k <= 40; k++) {
            // 更新 w1，累加级数和的分母部分
            w1 += 1.0 / k;
            // 更新 cr，计算级数项
            cr = -0.25 * cr / static_cast<double>(k * (k + 1)) * z2;
            // 计算 cp，级数项与 w1 的乘积
            cp = cr * (2.0 * w1 + 1.0 / (k + 1.0));
            // 更新 cs，累加级数项
            cs += cp;
            // 如果级数项的绝对值小于当前总和的 1.0e-15 倍，则停止迭代
            if (std::abs(cp) < std::abs(cs) * 1.0e-15) break;
        }

        // 计算第二个修正的贝塞尔函数 Y1
        cby1 = rp2 * ((std::log(z1 / 2.0) + el) * cbj1 - 1.0 / z1 - 0.25 * z1 * cs);
    } else {
        // 如果 a0 > 12.0，选择适当的 k0 值
        k0 = 12;
        if (a0 >= 35.0) k0 = 10;
        if (a0 >= 50.0) k0 = 8;

        // 计算 cp0，第一个多项式级数
        ct1 = z1 - 0.25 * pi;
        cp0 = 1.0;
        for (k = 1; k <= k0; k++) {
            cp0 += a[k - 1] * pow(z1, -2 * k);
        }

        // 计算 cq0，第二个多项式级数
        cq0 = -0.125 / z1;
        for (k = 1; k <= k0; k++)
            cq0 += b[k - 1] * pow(z1, -2 * k - 1);

        // 计算 cbj0 和 cby0，使用多项式展开计算
        cu = std::sqrt(rp2 / z1);
        cbj0 = cu * (cp0 * cos(ct1) - cq0 * sin(ct1));
        cby0 = cu * (cp0 * sin(ct1) + cq0 * cos(ct1));

        // 计算 cp1，第三个多项式级数
        ct2 = z1 - 0.75 * pi;
        cp1 = 1.0;
        for (k = 1; k <= k0; k++)
            cp1 += a1[k - 1] * pow(z1, -2 * k);

        // 计算 cq1，第四个多项式级数
        cq1 = 0.375 / z1;
        for (k = 1; k <= k0; k++) {
            cq1 = cq1 + b1[k - 1] * pow(z1, -2 * k - 1);
        }

        // 计算 cbj1 和 cby1，使用多项式展开计算
        cbj1 = cu * (cp1 * cos(ct2) - cq1 * sin(ct2));
        cby1 = cu * (cp1 * sin(ct2) + cq1 * cos(ct2));
    }

    // 如果 z 的实部小于 0.0，对 cbj0、cbj1、cby0 和 cby1 进行修正
    if (z.real() < 0.0) {
        if (z.imag() < 0.0) cby0 = cby0 - 2.0 * ci * cbj0;
        if (z.imag() > 0.0) cby0 = cby0 + 2.0 * ci * cbj0;
        if (z.imag() < 0.0) cby1 = -(cby1 - 2.0 * ci * cbj1);
        if (z.imag() > 0.0) cby1 = -(cby1 + 2.0 * ci * cbj1);
        // 对 cbj1 取负号
        cbj1 = -cbj1;
    }

    // 计算 cdy0 和 cdy1
    cdy0 = -cby1;
    cdy1 = cby0 - 1.0 / z * cby1;

    // 根据 kf 的值选择返回值
    if (kf == 0) {
        *zf = cby0;
        *zd = cdy0
// Compute the complex zeros of Bessel functions Y0(z), Y1(z), and Y1'(z)
// using modified Newton's iteration method.
// This function calculates the zeros and their associated values at the zeros.
inline void cyzo(int nt, int kf, int kc, std::complex<double> *zo, std::complex<double> *zv) {
    // Initialize variables for iteration
    int i, it, j, nr;
    double x, h, w, y, w0;
    std::complex<double> z, zf, zd, zfd, zgd, zp, zq, zw;

    // Initialize some variables to zero
    x = 0.0;
    y = 0.0;
    h = 0.0;

    // Set initial values based on choice of complex or real roots
    if (kc == 0) {
        x = -2.4;
        y = 0.54;
        h = 3.14;
    } else if (kc == 1) {
        x = 0.89;
        y = 0.0;
        h = -3.14;
    }

    // Adjust x based on function choice
    if (kf == 1) {
        x = -0.503;
    }

    if (kf == 2) {
        x = 0.577;
    }

    // Initialize z as a complex number
    z = std::complex<double>(x, y);
    w = 0.0;

    // Iteratively compute zeros and their associated values
    for (nr = 1; nr <= nt; nr++) {
        // Adjust z for subsequent iterations
        if (nr > 1) {
            z = zo[nr - 2] - h;
        }
        it = 0;

        // Perform Newton's iteration until convergence or maximum iterations
        do {
            it += 1;
            // Compute values of the Bessel function and its derivative
            cy01(kf, z, &zf, &zd);

            // Compute zfd as the derivative of the Bessel function over the product of (z - zo[i])
            zp = 1.0;
            for (i = 1; i < nr; i++) {
                zp *= (z - zo[i - 1]);
            }
            zfd = zf / zp;

            // Compute zq as the sum of products (z - zo[j]) for all j ≠ i
            zq = 0.0;
            for (i = 1; i < nr; i++) {
                zw = 1.0;
                for (j = 1; j < nr; j++) {
                    if (j == i) { continue; }
                    zw *= (z - zo[j - 1]);
                }
                zq += zw;
            }

            // Compute zgd as (zd - zq * zfd) / zp
            zgd = (zd - zq * zfd) / zp;

            // Update z using Newton's method
            z -= zfd / zgd;

            // Update w0 and w to check for convergence
            w0 = w;
            w = std::abs(z);

        } while ((it <= 50) && (fabs((w - w0) / w) > 1.0e-12));

        // Store computed zero back into zo array
        zo[nr - 1] = z;
    }

    // Compute values of the Bessel function or its derivative at the computed zeros
    for (i = 1; i <= nt; i++) {
        z = zo[i - 1];
        if ((kf == 0) || (kf == 2)) {
            cy01(1, z, &zf, &zd);
            zv[i - 1] = zf;
        } else if (kf == 1) {
            cy01(0, z, &zf, &zd);
            zv[i - 1] = zf;
        }
    }
    return;
}
    # 如果 x 等于 0.0，则直接设定 e1 为一个非常大的数值，避免除以零错误
    if (x == 0.0) {
        e1 = 1e300;
    # 如果 x 在 (0.0, 1.0] 范围内
    } else if (x <= 1.0) {
        e1 = 1.0;  # 初始化 e1
        r = 1.0;   # 初始化 r
        # 开始计算级数展开
        for (k = 1; k < 26; k++) {
            r = -r*k*x/pow(k+1.0, 2);  # 计算下一个级数项
            e1 += r;  # 累加到 e1 中
            # 如果级数项 r 的绝对值小于 e1 绝对值的 1e-15 倍，则停止循环
            if (fabs(r) <= fabs(e1)*1e-15) { break; }
        }
        e1 = -ga - log(x) + x*e1;  # 计算最终结果 e1
    # 如果 x 大于 1.0
    } else {
        m = 20 + (int)(80.0/x);  # 计算 m 的值
        t0 = 0.0;  # 初始化 t0
        # 计算另一种级数展开的初始值
        for (k = m; k > 0; k--) {
            t0 = k / (1.0 + k / (x+t0));
        }
        t = 1.0 / (x + t0);  # 计算 t 的值
        e1 = exp(-x)*t;  # 计算最终结果 e1
    }
    return e1;  # 返回计算结果 e1
}

template <typename T>
std::complex<T> e1z(std::complex<T> z) {

    // ====================================================
    // Purpose: Compute complex exponential integral E1(z)
    // Input :  z   --- Argument of E1(z)
    // Output:  CE1 --- E1(z)
    // ====================================================

    const T pi = 3.141592653589793;
    const T el = 0.5772156649015328;
    int k;
    std::complex<T> ce1, cr, zc, zd, zdc;
    T x = z.real();
    T a0 = std::abs(z);
    // Continued fraction converges slowly near negative real axis,
    // so use power series in a wedge around it until radius 40.0
    T xt = -2.0*fabs(z.imag());

    if (a0 == 0.0) { return 1e300; }
    if ((a0 < 5.0) || ((x < xt) && (a0 < 40.0))) {
        // Power series
        ce1 = 1.0;
        cr = 1.0;
        for (k = 1; k < 501; k++) {
            cr = -cr*z*static_cast<T>(k / std::pow(k + 1, 2));
            ce1 += cr;
            if (std::abs(cr) < std::abs(ce1)*1e-15) { break; }
        }
        if ((x <= 0.0) && (z.imag() == 0.0)) {
            //Careful on the branch cut -- use the sign of the imaginary part
            // to get the right sign on the factor if pi.
            ce1 = -el - std::log(-z) + z*ce1 - copysign(pi, z.imag())*std::complex<T>(0.0, 1.0);
        } else {
            ce1 = -el - std::log(z) + z*ce1;
        }
    } else {
        // Continued fraction https://dlmf.nist.gov/6.9
        //                  1     1     1     2     2     3     3
        // E1 = exp(-z) * ----- ----- ----- ----- ----- ----- ----- ...
        //                Z +   1 +   Z +   1 +   Z +   1 +   Z +
        zc = 0.0;
        zd = static_cast<T>(1) / z;
        zdc = zd;
        zc += zdc;
        for (k = 1; k < 501; k++) {
            zd = static_cast<T>(1) / (zd*static_cast<T>(k) + static_cast<T>(1));
            zdc *= (zd - static_cast<T>(1));
            zc += zdc;

            zd = static_cast<T>(1) / (zd*static_cast<T>(k) + z);
            zdc *= (z*zd - static_cast<T>(1));
            zc += zdc;
            if ((std::abs(zdc) <= std::abs(zc)*1e-15) && (k > 20)) { break; }
        }
        ce1 = std::exp(-z)*zc;
        if ((x <= 0.0) && (z.imag() == 0.0)) {
            ce1 -= pi*std::complex<T>(0.0, 1.0);
        }
    }
    return ce1;
}


template <typename T>
T eix(T x) {

    // ============================================
    // Purpose: Compute exponential integral Ei(x)
    // Input :  x  --- Argument of Ei(x)
    // Output:  EI --- Ei(x)
    // ============================================

    const T ga = 0.5772156649015328;
    T ei, r;

    if (x == 0.0) {
        ei = -1.0e+300;
    } else if (x < 0) {
        // Compute the modified exponential integral E1(x) for negative x
        ei = -e1xb(-x);
   `
// 如果 x 的绝对值小于等于 40.0，使用以 x=0 为中心的幂级数展开计算 ei
} else if (fabs(x) <= 40.0) {
    // 初始化 ei 和 r
    ei = 1.0;
    r = 1.0;

    // 开始计算幂级数
    for (int k = 1; k <= 100; k++) {
        r = r * k * x / ((k + 1.0) * (k + 1.0));
        ei += r;
        // 检查误差是否足够小，如果是则退出循环
        if (fabs(r / ei) <= 1.0e-15) { break; }
    }
    // 计算最终的 ei 值
    ei = ga + log(x) + x * ei;
} else {
    // 如果 x 的绝对值大于 40.0，使用渐近展开计算 ei
    // 初始化 ei 和 r
    ei = 1.0;
    r = 1.0;

    // 开始计算渐近展开
    for (int k = 1; k <= 20; k++) {
        r = r * k / x;
        ei += r;
    }
    // 计算最终的 ei 值
    ei = exp(x) / x * ei;
}
// 返回最终计算得到的 ei 值
return ei;
}

template <typename T>
std::complex<T> eixz(std::complex<T> z) {

    // ============================================
    // Purpose: Compute exponential integral Ei(x)
    // Input :  x  --- Complex argument of Ei(x)
    // Output:  EI --- Ei(x)
    // ============================================

    std::complex<T> cei;
    const T pi = 3.141592653589793;
    cei = - e1z(-z);  // 计算 Ei(x) 的负值
    if (z.imag() > 0.0) {
        cei += std::complex<T>(0.0, pi);  // 如果 z 的虚部大于 0，加上 πi
    } else if (z.imag() < 0.0 ) {
        cei -= std::complex<T>(0.0, pi);  // 如果 z 的虚部小于 0，减去 πi
    } else {
        if (z.real() > 0.0) {
            cei += std::complex<T>(0.0, copysign(pi, z.imag()));  // 如果 z 的实部大于 0，加上 πi 的符号（根据虚部符号确定正负）
        }
    }
    return cei;  // 返回计算结果 cei
}


inline void eulerb(int n, double *en) {

    // ======================================
    // Purpose: Compute Euler number En
    // Input :  n --- Serial number
    // Output:  EN(n) --- En
    // ======================================

    int k, m, isgn;
    double r1, r2, s;
    const double hpi = 2.0 / 3.141592653589793;
    en[0] = 1.0;  // 设置初始值 E_0 = 1
    en[2] = -1.0;  // 设置 E_2 = -1
    r1 = -4.0*pow(hpi, 3);  // 计算常数 -4 * (π/2)^3
    for (m = 4; m <= n; m += 2) {
        r1 = -r1 * (m-1) * m * hpi * hpi;  // 更新 r1 的值
        r2 = 1.0;
        isgn = 1;
        for (k = 3; k <= 1000; k += 2) {
            isgn = -isgn;
            s = pow(1.0 / k, m + 1);
            r2 += isgn * s;  // 计算 r2 的累加值
            if (s < 1e-15) { break; }  // 如果 s 很小则退出循环
        }
        en[m] = r1*r2;  // 计算并存储 En 的值
    }
    return;  // 返回
}


template <typename T>
void fcoef(int kd, int m, T q, T a, T *fc) {

    // =====================================================
    // Purpose: Compute expansion coefficients for Mathieu
    //          functions and modified Mathieu functions
    // Input :  m  --- Order of Mathieu functions
    //          q  --- Parameter of Mathieu functions
    //          KD --- Case code
    //                 KD=1 for cem(x,q)  ( m = 0,2,4,...)
    //                 KD=2 for cem(x,q)  ( m = 1,3,5,...)
    //                 KD=3 for sem(x,q)  ( m = 1,3,5,...)
    //                 KD=4 for sem(x,q)  ( m = 2,4,6,...)
    //          A  --- Characteristic value of Mathieu
    //                 functions for given m and q
    // Output:  FC(k) --- Expansion coefficients of Mathieu
    //                 functions ( k= 1,2,...,KM )
    //                 FC(1),FC(2),FC(3),... correspond to
    //                 A0,A2,A4,... for KD=1 case, A1,A3,
    //                 A5,... for KD=2 case, B1,B3,B5,...
    //                 for KD=3 case and B2,B4,B6,... for
    //                 KD=4 case
    // =====================================================

    int i, k, j, jm = 0, km, kb;
    T f1, fnan, qm, s, f, u, v, f2, f3, sp, ss, s0;

    for (i = 0; i < 251; ++i) { fc[i] = 0.0; }  // 初始化 fc 数组为零
}
    if (fabs(q) <= 1.0e-7) {
        // 如果 q 的绝对值小于或等于 1.0e-7，则进行以下处理（Abramowitz & Stegun 20.2.27-28）
        if (kd == 1) {
            // 如果 kd 等于 1，则根据规则计算 jm
            jm = m / 2 + 1;
        } else if ((kd == 2) || (kd == 3)) {
            // 如果 kd 等于 2 或 3，则根据规则计算 jm
            jm = (m - 1) / 2 + 1;
        } else if (kd == 4) {
            // 如果 kd 等于 4，则根据规则计算 jm
            jm = m / 2;
        }

        // 如果 jm + 1 大于 251，则处理溢出情况
        if (jm + 1 > 251) {
            fnan = NAN;
            // 将数组 fc 的前 251 个元素赋值为 NaN
            for (i = 0; i < 251; ++i) {
                fc[i] = fnan;
            }
            return;
        }
        // 使用最简单的展开方式继续处理
        if (kd == 1 || kd == 2) {
            // 根据不同的 m 值填充数组 fc
            if (m == 0) {
                fc[0] = 1.0 / sqrt(2.0);
                fc[1] = -q / (2.0 * sqrt(2.0));
            } else if (m == 1) {
                fc[0] = 1.0;
                fc[1] = -q / 8.0;
            } else if (m == 2) {
                fc[0] = q / 4.0;
                fc[1] = 1.0;
                fc[2] = -q / 12.0;
            } else {
                fc[jm - 1] = 1.0;
                fc[jm] = -q / (4.0 * (m + 1));
                fc[jm - 2] = q / (4.0 * (m - 1));
            }
        } else if (kd == 3 || kd == 4) {
            // 根据不同的 m 值填充数组 fc
            if (m == 1) {
                fc[0] = 1.0;
                fc[1] = -q / 8.0;
            } else if (m == 2) {
                fc[0] = 1.0;
                fc[1] = -q / 12.0;
            } else {
                fc[jm - 1] = 1.0;
                fc[jm] = -q / (4.0 * (m + 1));
                fc[jm - 2] = q / (4.0 * (m - 1));
            }
        }
        return;
    } else if (q <= 1.0) {
        // 如果 q 小于或等于 1.0，则根据一组公式计算 qm
        qm = 7.5 + 56.1 * sqrt(q) - 134.7 * q + 90.7 * sqrt(q) * q;
    } else {
        // 如果 q 大于 1.0，则根据另一组公式计算 qm
        qm = 17.0 + 3.1 * sqrt(q) - 0.126 * q + 0.0037 * sqrt(q) * q;
    }

    // 根据计算得到的 qm 和 m 计算 km
    km = (int)(qm + 0.5 * m);

    // 如果 km 大于 251，则处理溢出情况
    if (km > 251) {
        // 将数组 fc 的前 251 个元素赋值为 NaN
        for (i = 0; i < 251; ++i) {
            fc[i] = NAN;
        }
        return;
    }

    // 初始化变量 kb, s, f, u, fc[km-1], f2 的值
    kb = 0;
    s = 0.0;
    f = 1.0e-100;
    u = 0.0;
    fc[km - 1] = 0.0;
    f2 = 0.0;
    # 如果 kd 等于 1，则执行以下操作
    if (kd == 1) {
        # 从 km 循环到 3
        for (k = km; k >= 3; k--) {
            v = u;
            u = f;
            # 计算新的 f 值
            f = (a - 4.0 * k * k) * u / q - v;

            # 如果 f 的绝对值小于 fc[k] 的绝对值
            if (fabs(f) < fabs(fc[k])) {
                kb = k;
                fc[0] = 1.0e-100;
                sp = 0.0;
                f3 = fc[k];
                fc[1] = a / q * fc[0];
                fc[2] = (a - 4.0) * fc[1] / q - 2.0 * fc[0];
                u = fc[1];
                f1 = fc[2];

                # 从 3 到 kb 进行循环
                for (i = 3; i <= kb; i++) {
                    v = u;
                    u = f1;
                    # 计算新的 fc[i] 值
                    f1 = (a - 4.0 * (i - 1.0) * (i - 1.0)) * u / q - v;
                    fc[i] = f1;

                    # 如果 i 等于 kb，则记录 f2 值
                    if (i == kb) { f2 = f1; }
                    # 如果 i 不等于 kb，则累加 sp 值
                    if (i != kb) { sp += f1*f1; }
                }

                # 计算 sp 的最终值
                sp += 2.0*fc[0]*fc[0] + fc[1]*fc[1] + fc[2]*fc[2];
                # 计算 ss 的值
                ss = s + sp * (f3 / f2) * (f3 / f2);
                # 计算 s0 的值
                s0 = sqrt(1.0 / ss);

                # 从 1 到 km 进行循环
                for (j = 1; j <= km; j++) {
                    # 如果 j 小于等于 kb + 1，则更新 fc[j-1] 的值
                    if (j <= kb + 1) {
                        fc[j - 1] = s0 * fc[j-1] * f3 / f2;
                    } else {
                        # 否则，更新 fc[j-1] 的值
                        fc[j - 1] *= s0;
                    }
                }
                # 如果 fc[0] 小于 0.0，则将 fc 数组中的元素取负值
                if (fc[0] < 0.0) { for (j = 0; j < km; j++) { fc[j] = -fc[j]; } }
                # 结束函数执行
                return;
            } else {
                # 否则，更新 fc[k-1] 的值并累加 s 的值
                fc[k - 1] = f;
                s += f*f;
            }
        }

        # 更新 fc[1] 和 fc[0] 的值
        fc[1] = q * fc[2] / (a - 4.0 - 2.0 * q * q / a);
        fc[0] = q / a * fc[1];
        # 更新 s 的值
        s += 2.0 * fc[0] * fc[0] + fc[1] * fc[1];
        # 计算 s0 的值
        s0 = sqrt(1.0 / s);

        # 从 1 到 km 进行循环
        for (k = 1; k <= km; k++) {
            # 更新 fc[k-1] 的值
            fc[k - 1] *= s0;
        }
    } else if ((kd == 2) || (kd == 3)) {
        # 如果 kd 等于 2 或者 3，则执行以下操作
        for (k = km; k >= 3; k--) {
            v = u;
            u = f;
            # 计算新的 f 值
            f = (a - (2.0 * k - 1) * (2.0 * k - 1)) * u / q - v;

            # 如果 f 的绝对值大于等于 fc[k-1] 的绝对值
            if (fabs(f) >= fabs(fc[k - 1])) {
                # 更新 fc[k-2] 的值并累加 s 的值
                fc[k - 2] = f;
                s += f * f;
            } else {
                # 否则，更新 kb 和 f3 的值，并跳转到标签 L45
                kb = k;
                f3 = fc[k - 1];
                goto L45;
            }
        }

        # 计算 fc[0] 的值
        fc[0] = q / (a - 1.0 - pow(-1, kd) * q) * fc[1];
        # 累加 s 的值
        s += fc[0] * fc[0];
        # 计算 s0 的值
        s0 = sqrt(1.0 / s);

        # 从 1 到 km 进行循环
        for (k = 1; k <= km; k++) {
            # 更新 fc[k-1] 的值
            fc[k - 1] *= s0;
        }
        # 如果 fc[0] 小于 0.0，则将 fc 数组中的元素取负值
        if (fc[0] < 0.0) { for (j = 0; j < km; j++) { fc[j] = -fc[j]; } }
        # 结束函数执行
        return;
L45:
        // 初始化数组元素
        fc[0] = 1.0e-100;
        // 计算数组第二个元素的值
        fc[1] = (a - 1.0 - pow(-1, kd) * q) / q * fc[0];
        // 初始化变量
        sp = 0.0;
        // 设置初始值
        u = fc[0];
        // 设置初始值
        f1 = fc[1];

        // 循环计算数组的其余元素
        for (i = 2; i <= kb - 1; i++) {
            v = u;
            u = f1;
            // 计算当前元素的值
            f1 = (a - (2.0 * i - 1) * (2.0 * i - 1)) * u / q - v;

            // 如果不是最后一个元素，更新数组并计算 sp
            if (i != kb - 1) {
                fc[i] = f1;
                sp += f1 * f1;
            } else {
                f2 = f1;
            }
        }

        // 计算 sp 的最终值
        sp += fc[0] * fc[0] + fc[1] * fc[1];
        // 更新 ss 的值
        ss = s + sp * (f3 / f2) * (f3 / f2);
        // 计算 s0 的值
        s0 = sqrt(1.0 / ss);

        // 更新数组中的值
        for (j = 1; j <= km; j++) {
            if (j < kb) {
                fc[j - 1] *= s0 * f3 / f2;
            }

            if (j >= kb) {
                fc[j - 1] *= s0;
            }
        }

    } else if (kd == 4) {
        // 当 kd 等于 4 时的计算过程
        for (k = km; k >= 3; k--) {
            v = u;
            u = f;
            // 计算 f 的值
            f = (a - 4.0 * k * k) * u / q - v;

            // 根据条件更新数组元素
            if (fabs(f) >= fabs(fc[k])) {
                fc[k - 2] = f;
                s += f*f;
            } else {
                kb = k;
                f3 = fc[k - 1];
                // 跳转到标签 L70
                goto L70;
            }
        }

        // 计算数组第一个元素的值
        fc[0] = q / (a - 4.0) * fc[1];
        // 更新 s 的值
        s += fc[0] * fc[0];
        // 计算 s0 的值
        s0 = sqrt(1.0 / s);

        // 更新数组中的值
        for (k = 1; k <= km; k++) {
            fc[k - 1] *= s0;
        }
        // 如果数组第一个元素小于 0，取反数组中的所有元素
        if (fc[0] < 0.0) { for (j = 0; j < km; j++) { fc[j] = -fc[j]; } }
        return;
L70:
        // 标签 L70：初始化数组元素
        fc[0] = 1.0e-100;
        // 计算数组第二个元素的值
        fc[1] = (a - 4.0) / q * fc[0];
        // 初始化变量
        sp = 0.0;
        // 设置初始值
        u = fc[0];
        // 设置初始值
        f1 = fc[1];

        // 循环计算数组的其余元素
        for (i = 2; i <= kb - 1; i++) {
            v = u;
            u = f1;
            // 计算当前元素的值
            f1 = (a - 4.0 * i * i) * u / q - v;

            // 如果不是最后一个元素，更新数组并计算 sp
            if (i != kb - 1) {
                fc[i] = f1;
                sp = sp + f1 * f1;
            } else {
                f2 = f1;
            }
        }

        // 计算 sp 的最终值
        sp += fc[0] * fc[0] + fc[1] * fc[1];
        // 更新 ss 的值
        ss = s + sp * (f3 / f2) * (f3 / f2);
        // 计算 s0 的值
        s0 = sqrt(1.0 / ss);

        // 更新数组中的值
        for (j = 1; j <= km; j++) {
            if (j < kb) {
                fc[j - 1] *= s0 * f3 / f2;
            } else {
                fc[j - 1] *= s0;
            }
        }
    }
    // 如果数组第一个元素小于 0，取反数组中的所有元素
    if (fc[0] < 0.0) { for (j = 0; j < km; j++) { fc[j] = -fc[j]; } }
    // 返回函数
    return;
}


inline double gaih(double x) {

    // =====================================================
    // Purpose: Compute gamma function Г(x)
    // Input :  x  --- Argument of Г(x), x = n/2, n=1,2,…
    // Output:  GA --- Г(x)
    // =====================================================

    int k, m;
    const double pi = 3.141592653589793;
    double ga = 0.0;

    // 如果 x 是正整数
    if ((x == (int)x) && (x > 0.0)) {
        ga = 1.0;
        m = (int)(x - 1.0);
        // 计算阶乘
        for (k = 2; k < (m+1); k++) {
            ga *= k;
        }
    // 如果 x 是半整数
    } else if (((x+0.5) == (int)(x+0.5)) && (x > 0.0)) {
        m = (int)x;
        ga = sqrt(pi);
        // 计算半整数的伽玛函数
        for (k = 1; k < (m+1); k++) {
            ga *= 0.5*(2.0*k - 1.0);
        }
    } else {
        ga = NAN;
    }
    // 返回计算结果
    return ga;
}
// ================================================
// Purpose: Compute gamma function Г(x) for x in the range |x| ≤ 1
// Input :  x  --- Argument of Г(x)
// Output:  GA --- Г(x)
// ================================================
inline double gam0(double x) {
    double gr;
    // Coefficients for the series approximation of gamma function
    static const double g[25] = {
         1.0e0,
         0.5772156649015329e0, -0.6558780715202538e0, -0.420026350340952e-1, 0.1665386113822915e0,
        -0.421977345555443e-1, -0.96219715278770e-2, 0.72189432466630e-2, -0.11651675918591e-2,
        -0.2152416741149e-3, 0.1280502823882e-3, -0.201348547807e-4, -0.12504934821e-5,
         0.11330272320e-5, -0.2056338417e-6, 0.61160950e-8, 0.50020075e-8, -0.11812746e-8,
         0.1043427e-9, 0.77823e-11, -0.36968e-11, 0.51e-12, -0.206e-13, -0.54e-14, 0.14e-14
    };
    gr = g[24];
    // Compute the gamma function using the series approximation
    for (int k = 23; k >= 0; k--) {
        gr = gr*x + g[k];
    }
    // Return the reciprocal of the computed value
    return 1.0 / (gr * x);
}


// ==================================================
// Purpose: Compute gamma function Г(x) for x ≠ 0, -1, -2, ...
// Input :  x  --- Argument of Г(x)
// Output:  GA --- Г(x)
// ==================================================
inline double gamma2(double x) {
    double ga, gr, r, z;
    int k, m;
    const double pi = 3.141592653589793;
    // Coefficients for the series approximation of gamma function
    static const double g[26] = {
        1.0000000000000000e+00,  0.5772156649015329e+00, -0.6558780715202538e+00, -0.4200263503409520e-01,
        0.1665386113822915e+00, -0.4219773455554430e-01, -0.9621971527877000e-02,  0.7218943246663000e-02,
       -0.1165167591859100e-02, -0.2152416741149000e-03,  0.1280502823882000e-03, -0.2013485478070000e-04,
       -0.1250493482100000e-05,  0.1133027232000000e-05, -0.2056338417000000e-06,  0.6116095000000000e-08,
        0.5002007500000000e-08, -0.1181274600000000e-08,  0.1043427000000000e-09,  0.7782300000000000e-11,
       -0.3696800000000000e-11,  0.5100000000000000e-12, -0.2060000000000000e-13, -0.5400000000000000e-14,
        0.1400000000000000e-14,  0.1000000000000000e-15
    };
    
    // Special case when x is an integer
    if (x == (int)x) {
        if (x > 0.0) {
            ga = 1.0;
            m = (int)(x - 1);
            // Compute factorial for positive integers
            for  (k = 2; k < (m+1); k++) {
                ga *= k;
            }
        } else {
            ga = 1e300; // Avoid overflow for non-positive integers
        }
    } else {
        r = 1.0;
        // Handling for |x| > 1
        if (fabs(x) > 1.0) {
            z = fabs(x);
            m = (int)z;
            // Compute the product (z-1)*(z-2)*...*(z-m)
            for  (k = 1; k < (m+1); k++) {
                r *= (z-k);
            }
            z -= m;
        } else {
            z = x;
        }
        
        // Series approximation for gamma function
        gr = g[25];
        for ( k = 25; k > 0; k--) {
            gr *= z;
            gr += g[k-1];
        }
        
        // Compute the gamma function value
        ga = 1.0 / (gr*z);
        
        // Adjust the result for |x| > 1 and negative x
        if (fabs(x) > 1.0) {
            ga *= r;
            if (x < 0.0) {
                ga = -pi / (x*ga*sin(pi*x));
            }
        }
    }
    
    // Return the computed gamma function value
    return ga;
}


template <typename T>
inline void gmn(int m, int n, T c, T x, T *bk, T *gf, T *gd) {
    // ===========================================================
    // Purpose: Compute gmn(-ic,ix) and its derivative for oblate
    //          radial functions with a small argument
    // ===========================================================

    // 声明整型变量ip, k, nm，以及浮点型变量xm, gf0, gw, gd0, gd1
    int ip, k, nm;
    T xm, gf0, gw, gd0, gd1;
    // 定义精度常量eps
    const T eps = 1.0e-14;
    // 根据条件判断确定ip的值
    ip = ((n - m) % 2 == 0 ? 0 : 1);
    // 计算nm的值
    nm = 25 + (int)(0.5 * (n - m) + c);
    // 计算xm，即pow(1.0 + x * x, -0.5 * m)
    xm = pow(1.0 + x * x, -0.5 * m);
    // 初始化gf0和gw为0
    gf0 = 0.0;
    gw = 0.0;

    // 开始循环，计算gf0
    for (k = 1; k <= nm; k++) {
        // 累加计算gf0
        gf0 += bk[k - 1] * pow(x, 2.0 * k - 2.0);
        // 检查收敛条件，并在满足条件且k >= 10时退出循环
        if ((fabs((gf0 - gw) / gf0) < eps) && (k >= 10)) { break; }
        // 更新gw为当前的gf0
        gw = gf0;
    }

    // 计算*gf，即结果的一部分
    *gf = xm * gf0 * pow(x, 1 - ip);
    // 计算gd1
    gd1 = -m * x / (1.0 + x * x) * (*gf);
    // 初始化gd0为0
    gd0 = 0.0;

    // 开始循环，计算gd0
    for (k = 1; k < nm; ++k) {
        // 根据ip的值选择不同的计算方式
        if (ip == 0) {
            gd0 += (2.0 * k - 1.0) * bk[k - 1] * pow(x, 2.0 * k - 2.0);
        } else {
            gd0 += 2.0 * k * bk[k - 1] * pow(x, 2.0 * k - 1.0);
        }
        // 检查收敛条件，并在满足条件且k >= 10时退出循环
        if ((fabs((gd0 - gw) / gd0) < eps) && (k >= 10)) { break; }
        // 更新gw为当前的gd0
        gw = gd0;
    }
    // 计算*gd，即结果的另一部分
    *gd = gd1 + xm * gd0;
// ======================================================
// Purpose: Compute the hypergeometric function for a
//          complex argument, F(a,b,c,z)
// Input :  a --- Parameter
//          b --- Parameter
//          c --- Parameter,  c <> 0,-1,-2,...
//          z --- Complex argument
// Output:  ZHF --- F(a,b,c,z)
//          ISFER --- Error flag
// Routines called:
//      (1) GAMMA2 for computing gamma function
//      (2) PSI_SPEC for computing psi function
// ======================================================

// Initialize local variables
int L0 = 0, L1 = 0, L2 = 0, L3 = 0, L4 = 0, L5 = 0, L6 = 0;
int j, k=1, m, mab, mcab, nca, ncb, nm;
double a0, aa, bb, ca, cb, g0, g1, g2, g3, ga, gab, gam, gabc, gb, gba, gbm, gc, gcab,\
       gca, gcb, gm, pa, pac, pb, pca, rk1, rk2, rm, sp0, sm, sp, sq, sj1, sj2, w0, ws;
std::complex<double> z00, z1, zc0, zc1, zf0, zf1, zhf = 0.0, zp, zr, zp0, zr0, zr1, zw = 0.0;

// Extract real and imaginary parts of complex z
double x = z.real();
double y = z.imag();
double eps = 1e-15;
double pi = 3.141592653589793;
double el = 0.5772156649015329;
*isfer = 0;

// Check conditions for special cases L0-L6
if ((c == (int)c) && (c < 0.0)) { L0 = 1; }
if ((fabs(1 - x) < eps) && (y == 0.0) && (c-a-b <= 0.0)) { L1 = 1; }
if ((std::abs(z+1.0) < eps) && (fabs(c-a+b - 1.0) < eps)) { L2 = 1; }
if ((a == (int)a) && (a < 0.0)) { L3 = 1; }
if ((b == (int)b) && (b < 0.0)) { L4 = 1; }
if (((c-a) == (int)(c-a)) && (c-a <= 0.0)) { L5 = 1; }
if (((c-b) == (int)(c-b)) && (c-b <= 0.0)) { L6 = 1; }
aa = a;
bb = b;
a0 = std::abs(z);

// Adjust epsilon if necessary
if (a0 > 0.95) { eps = 1e-8; }

// Handle special cases identified by L0-L6
if (L0 || L1) {
    *isfer = 3;
    return 0.0;
}

if ((a0 == 0.0) || (a == 0.0) || (b == 0.0)) {
    zhf = 1.0;
} else if ((z == 1.0) && (c-a-b > 0.0)) {
    // Compute hypergeometric function for z = 1.0 and c-a-b > 0
    gc = gamma2(c);
    gcab = gamma2(c-a-b);
    gca = gamma2(c-a);
    gcb = gamma2(c-b);
    zhf = gc * gcab / (gca * gcb);
} else if (L2) {
    // Compute hypergeometric function for special case identified by L2
    g0 = sqrt(pi) * pow(2.0, -a);
    g1 = gamma2(c);
    g2 = gamma2(1.0 + 0.5*a - b);
    g3 = gamma2(0.5 + 0.5*a);
    zhf = g0 * g1 / (g2 * g3);
} else if (L3 || L4) {
    // Compute hypergeometric function for negative integer parameters a or b
    if (L3) { nm = (int)fabs(a); }
    if (L4) { nm = (int)fabs(b); }
    zhf = 1.0;
    zr = 1.0;
    for (k = 1; k < (nm+1); k++) {
        zr = zr * (c-a+k-1.0) * (c-b+k-1.0) / (k * (c+k-1.0)) * z;
        zhf += zr;
    }
} else if (L5 || L6) {
    // Compute hypergeometric function for negative integer parameters c-a or c-b
    if (L5) { nm = (int)fabs(c-a); }
    if (L6) { nm = (int)fabs(c-b); }
    zhf = 1.0;
    zr = 1.0;
    for (k = 1; k < (nm+1); k++) {
        zr = zr * (c-a+k-1.0) * (c-b+k-1.0) / (k * (c+k-1.0)) * z;
        zhf += zr;
    }
    zhf *= std::pow(1.0 - z, c-a-b);
}

// Restore original values of a and b
a = aa;
b = bb;

// Check if iteration count k exceeded 150
if (k > 150) { *isfer = 5; }

// Return computed hypergeometric function value
return zhf;
}
    // 定义整型变量和双精度浮点型变量
    int i, j, k, L, L0, L1, L2, mm, nm;
    double x, x0, x1, x2, xm;

    // 分配内存给 p1 数组，大小为 70 个整型元素
    int* p1 = (int *) calloc(70, sizeof(int));
    // 在 specfun.f 中，我们使用单个数组而不是分开的三个数组，
    // 并使用指针算术来访问。它们的使用方式基本上是一次性的，
    // 因此不会使代码复杂化。

    // 分配内存给 mnzoc 数组，大小为 211 个双精度浮点型元素
    // 在 specfun.f 中，ZOC 和 ZOC 数组是从 0 开始索引的
    double* mnzoc = (double *) calloc(211, sizeof(double));

    // 分配内存给 bdfj 数组，大小为 303 个双精度浮点型元素
    // 在 specfun.f 中，BJ, DJ 和 FJ 数组分别有 101 个元素
    double* bdfj = (double *) calloc(303, sizeof(double));

    // 将 x 初始化为 0
    x = 0;

    // 根据 nt 的值选择不同的计算公式和参数设置
    if (nt < 600) {
        // 计算 xm, nm 和 mm 的值
        xm = -1.0 + 2.248485 * sqrt(nt) - 0.0159382 * nt + 3.208775e-4 * pow(nt, 1.5);
        nm = (int)(14.5 + 0.05875 * nt);
        mm = (int)(0.02 * nt) + 6;
    } else {
        // 计算 xm, nm 和 mm 的值
        xm = 5.0 + 1.445389 * sqrt(nt) + 0.01889876 * nt - 2.147763e-4 * pow(nt, 1.5);
        nm = (int)(27.8 + 0.0327 * nt);
        mm = (int)(0.01088 * nt) + 10;
    }

    // 初始化 L0 的值为 0
    L0 = 0;
    /* 45 LOOP */
    for (i = 1; i < (nm+1); i++) {
        x1 = 0.407658 + 0.4795504*sqrt(i-1.0) + 0.983618*(i-1);
        x2 = 1.99535  + 0.8333883*sqrt(i-1.0) + 0.984584*(i-1);
        L1 = 0;

        /* 开始循环处理每个 i 值 */
        for (j = 1; j < (mm+1); j++) {
            /* 如果不是第一个循环，则设置 x 为 x1 */
            if ((i != 1) || (j != 1)) {
                x = x1;
                /* 迭代求解 x 的值，直到满足精度要求 */
                do
                {
                    bjndd(x, i, &bdfj[0], &bdfj[101], &bdfj[202]);
                    x0 = x;
                    x -= bdfj[100+i]/bdfj[201+i];
                    /* 如果 x1 大于 xm，则跳转到标签 L20 */
                    if (x1 > xm) { goto L20; }
                } while (fabs(x-x0) > 1e-10);
            }
            /* 更新 L1 的值 */
            L1 += 1;
            mnzoc[69 + L1] = i-1;                    /* 将 i-1 存入数组 mnzoc 的特定位置 */
            mnzoc[L1-1] = j;                         /* 将 j 存入数组 mnzoc 的特定位置 */
            /* 如果 i 等于 1，则在数组 mnzoc 中的 L1-1 位置存入 j-1 */
            if (i == 1) { mnzoc[L1 - 1] = j-1; }
            /* 在数组 p1 的 L1-1 位置存入 1 */
            p1[L1-1] = 1;
            mnzoc[140+L1] = x;                       /* 将 x 存入数组 mnzoc 的特定位置 */
            /* 根据 i 的大小，更新 x1 的值 */
            if (i <= 15) {
                x1 = x + 3.057 + 0.0122*(i-1) + (1.555 + 0.41575*(i-1))/pow(j+1, 2.0);
            } else {
                x1 = x + 2.918 + 0.01924*(i-1) + (6.26 + 0.13205*(i-1))/pow(j+1, 2.0);
            }
void jynb(int n, T x, int *nm, T *bj, T *dj, T *by, T *dy) {
    // =====================================================
    // Purpose: Compute Bessel functions Jn(x), Yn(x) and
    //          their derivatives
    // Input :  x --- Argument of Jn(x) and Yn(x) ( x ≥ 0 )
    //          n --- Order of Jn(x) and Yn(x)
    // Output:  BJ(n) --- Jn(x)
    //          DJ(n) --- Jn'(x)
    //          BY(n) --- Yn(x)
    //          DY(n) --- Yn'(x)
    //          NM --- Highest order computed
    // Routines called:
    //          JYNBH to calculate the Jn and Yn
    // =====================================================

    int k;
    // 调用 jynbh 函数计算 Bessel 函数 Jn(x) 和 Yn(x)，以及它们的导数
    jynbh(n, 0, x, nm, bj, by);

    // 根据微分公式计算导数
    if (x < 1.0e-100) {
        // 当 x 接近 0 时的特殊处理
        for (k = 0; k <= n; k++) {
            dj[k] = 0.0;
            dy[k] = 1.0e+300;
        }
        dj[1] = 0.5;
    } else {
        // 计算 Jn'(x)
        dj[0] = -bj[1];
        for (k = 1; k <= *nm; k++) {
            dj[k] = bj[k - 1] - k / x * bj[k];
        }

        // 计算 Yn'(x)
        dy[0] = -by[1];
        for (k = 1; k <= *nm; k++) {
            dy[k] = by[k - 1] - k * by[k] / x;
        }
    }
    return;
}
// Purpose: Compute Bessel functions Jn(x), Yn(x)
// Input :  x --- Argument of Jn(x) and Yn(x) ( x ≥ 0 )
//          n --- Highest order of Jn(x) and Yn(x) computed  ( n ≥ 0 )
//          nmin -- Lowest order computed  ( nmin ≥ 0 )
// Output:  BJ(n-NMIN) --- Jn(x)   ; if indexing starts at 0
//          BY(n-NMIN) --- Yn(x)   ; if indexing starts at 0
//          NM --- Highest order computed
// Routines called:
//          MSTA1 and MSTA2 to calculate the starting
//          point for backward recurrence
void jynbh(int n, int nmin, T x, int *nm, T *bj, T *by) {
    
    int k, m, ky;
    T pi = 3.141592653589793;
    T r2p = 0.63661977236758;
    T bs, s0, su, sv, f2, f1, f;
    T bj0, bj1, ec, by0, by1, bjk, byk;
    T p0, q0, cu, t1, p1, q1, t2;
    
    // Initialize the highest order computed to n
    *nm = n;

    // Check if x is very small
    if (x < 1.0e-100) {
        // Compute Bessel functions values
        for (k = nmin; k <= n; k++) {
            bj[k - nmin] = 0.0;
            by[k - nmin] = -1.0e+300; // Set Yn(x) to a very large negative number
        }

        // Special case for nmin == 0
        if (nmin == 0) {
            bj[0] = 1.0; // Set J0(x) = 1.0
        }
        return; // Exit function
    }

    // Condition for using backward recurrence method
    if ((x <= 300.0) || (n > (int)(0.9 * x))) {
        // Backward recurrence for Jn
        if (n == 0) {
            *nm = 1; // Adjust highest order computed to 1 for n = 0 case
        }

        // Calculate starting point for backward recurrence using MSTA1
        m = msta1(x, 200);
        if (m < *nm) {
            *nm = m; // Update highest order computed if m is smaller
        } else {
            m = msta2(x, *nm, 15); // Calculate starting point using MSTA2
        }

        // Initialize variables for backward recurrence
        bs = 0.0;
        su = 0.0;
        sv = 0.0;
        f2 = 0.0;
        f1 = 1.0e-100;
        f = 0.0;

        // Perform backward recurrence to compute Bessel functions
        for (k = m; k >= 0; k--) {
            f = 2.0*(k + 1.0)/x*f1 - f2;
            if ((k <= *nm) && (k >= nmin)) {
                bj[k - nmin] = f; // Store Jn(x) values
            }
            if (k == 2 * (int)(k / 2) && k != 0) {
                bs += 2.0 * f; // Sum for further computation
                su += pow(-1, k / 2) * f / k; // Sum for further computation
            } else if (k > 1) {
                sv += pow(-1, k / 2) * k / (k * k - 1.0) * f; // Sum for further computation
            }
            f2 = f1;
            f1 = f;
        }
        s0 = bs + f; // Final sum value

        // Normalize Bessel function values
        for (k = nmin; k <= *nm; k++) {
            bj[k - nmin] /= s0;
        }

        // Estimates for Yn at start of recurrence
        bj0 = f1 / s0;
        bj1 = f2 / s0;
        ec = log(x / 2.0) + 0.5772156649015329; // Euler's constant
        by0 = r2p * (ec * bj0 - 4.0*su/s0); // Estimate Y0(x)
        by1 = r2p * ((ec - 1.0)*bj1 - bj0/x - 4.0*sv/s0); // Estimate Y1(x)

        // Assign estimated values to BY array
        if (0 >= nmin) { by[0 - nmin] = by0; }
        if (1 >= nmin) { by[1 - nmin] = by1; }
        ky = 2; // Initialize ky to 2
    } else {
        // Hankel expansion
        // 计算 t1，为 x 减去 π/4
        t1 = x - 0.25*pi;
        // 初始化 p0 和 q0
        p0 = 1.0;
        q0 = -0.125/x;

        // 计算 p0 和 q0 的值，进行 Hankel 展开
        for (k = 1; k <= 4; k++) {
            p0 += a[k - 1] * pow(x,-2*k);
            q0 += b[k - 1] * pow(x, -2*k - 1);
        }

        // 计算 cu，为 sqrt(r2p / x)
        cu = sqrt(r2p / x);
        // 计算 bj0 和 by0
        bj0 = cu * (p0*cos(t1) - q0*sin(t1));
        by0 = cu * (p0*sin(t1) + q0*cos(t1));

        // 如果 0 大于等于 nmin，将 bj0 和 by0 存入对应数组
        if (0 >= nmin) {
            bj[0 - nmin] = bj0;
            by[0 - nmin] = by0;
        }

        // 计算 t2，为 x 减去 0.75*pi
        t2 = x - 0.75*pi;
        // 初始化 p1 和 q1
        p1 = 1.0;
        q1 = 0.375/x;

        // 计算 p1 和 q1 的值，进行 Hankel 展开
        for (k = 1; k <= 4; k++) {
            p1 += a1[k - 1] * pow(x, -2*k);
            q1 += b1[k - 1] * pow(x, -2*k - 1);
        }

        // 计算 bj1 和 by1
        bj1 = cu * (p1*cos(t2) - q1*sin(t2));
        by1 = cu * (p1*sin(t2) + q1*cos(t2));

        // 如果 1 大于等于 nmin，将 bj1 和 by1 存入对应数组
        if (1 >= nmin) {
            bj[1 - nmin] = bj1;
            by[1 - nmin] = by1;
        }

        // 计算 bj 的后续元素，使用前向递推公式
        for (k = 2; k <= *nm; k++) {
            bjk = 2.0*(k - 1.0)/x*bj1 - bj0;
            // 如果 k 大于等于 nmin，将 bjk 存入 bj 数组
            if (k >= nmin) { bj[k - nmin] = bjk; }
            bj0 = bj1;
            bj1 = bjk;
        }
        // 设置 ky 为 2
        ky = 2;
    }
    // 对 Yn 进行前向递推计算
    for (k = ky; k <= *nm; k++) {
        byk = 2.0 * (k - 1.0) * by1 / x - by0;

        // 如果 k 大于等于 nmin，将 byk 存入 by 数组
        if (k >= nmin)
            by[k - nmin] = byk;

        by0 = by1;
        by1 = byk;
    }
inline void jyndd(int n, double x, double *bjn, double *djn, double *fjn, double *byn, double *dyn, double *fyn) {
    // ===========================================================
    // purpose: compute bessel functions jn(x) and yn(x), and
    //          their first and second derivatives
    // input:   x   ---  argument of jn(x) and yn(x) ( x > 0 )
    //          n   ---  order of jn(x) and yn(x)
    // output:  bjn ---  jn(x)
    //          djn ---  jn'(x)
    //          fjn ---  jn"(x)
    //          byn ---  yn(x)
    //          dyn ---  yn'(x)
    //          fyn ---  yn"(x)
    // routines called:
    //          jynbh to compute jn and yn
    // ===========================================================

    int nm = 0;
    double bj[2], by[2];

    // Call jynbh to compute Bessel functions jn(x) and yn(x)
    jynbh(n+1, n, x, &nm, bj, by);
    // compute derivatives by differentiation formulas
    *bjn = bj[0];  // jn(x)
    *byn = by[0];  // yn(x)
    *djn = -bj[1] + n*bj[0]/x;  // jn'(x)
    *dyn = -by[1] + n*by[0]/x;  // yn'(x)
    *fjn = (n*n/(x*x) - 1.0)*(*bjn) - (*djn)/x;  // jn"(x)
    *fyn = (n*n/(x*x) - 1.0)*(*byn) - (*dyn)/x;  // yn"(x)
    return;
}

inline void jyzo(int n, int nt, double *rj0, double *rj1, double *ry0, double *ry1) {
    // ======================================================
    // Purpose: Compute the zeros of Bessel functions Jn(x),
    //          Yn(x), and their derivatives
    // Input :  n  --- Order of Bessel functions  (n >= 0)
    //          NT --- Number of zeros (roots)
    // Output:  RJ0(L) --- L-th zero of Jn(x),  L=1,2,...,NT
    //          RJ1(L) --- L-th zero of Jn'(x), L=1,2,...,NT
    //          RY0(L) --- L-th zero of Yn(x),  L=1,2,...,NT
    //          RY1(L) --- L-th zero of Yn'(x), L=1,2,...,NT
    // Routine called: JYNDD for computing Jn(x), Yn(x), and
    //                 their first and second derivatives
    // ======================================================

    /*
     * SciPy Note:
     * See GH-18859 for additional changes done by SciPy for
     * better initial condition selection in Newton iteration
     */

    int L;
    double b, h, x, x0, bjn, djn, fjn, byn, dyn, fyn;
    const double pi = 3.141592653589793;
    // -- Newton method for j_{N,L}
    // initial guess for j_{N,1}
    if (n == 0) {
        x = 2.4;
    } else {
        // Initial guess for j_{N,1} based on specific formula
        x = n + 1.85576*pow(n, 0.33333) + 1.03315/ pow(n, 0.33333);
    }
    // iterate using Newton's method to find the zeros
    L = 0;
L10:
    x0 = x;
    jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);  // Compute Bessel functions and derivatives
    x -= bjn/djn;  // Newton's method iteration step
    if (fabs(x - x0) > 1e-11) { goto L10; }  // Convergence criterion for Newton's method

    L += 1;
    rj0[L - 1] = x;  // Store the L-th zero of Jn(x)
    // initial guess for j_{N,L+1}
    if (L == 1) {
        if (n == 0) {
            x = 5.52;
        } else {
            // Initial guess for j_{N,2} based on specific expansion and coefficients
            x = n + 3.24460 * pow(n, 0.33333) + 3.15824 / pow(n, 0.33333);
        }
    }
    } else {
        // 当根的增长近似为线性时 (https://dlmf.nist.gov/10.21#E19)
        x = rj0[L - 1] + (rj0[L - 1] - rj0[L - 2]);
    }
    if (L <= (n + 10)) {
        // 调用 jyndd 函数计算 Bessel 函数及其导数的值
        jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
        // 计算角度 h
        h = atan(fabs(djn) / sqrt(fabs(fjn * bjn)));
        // 计算系数 b
        b = -djn / (bjn * atan(h));
        // 根据 Newton 方法迭代计算 x 的值
        x -= (h - pi/2) / b;
    }

    if (L < nt) { goto L10; }

    // -- 对 j_{N,L+1}' 使用 Newton 方法
    if (n == 0) {
        // 当 n 等于 0 时，设定初始值为 3.8317
        x = 3.8317;
    } else {
        // 使用公式计算初始值 x (https://dlmf.nist.gov/10.21#E40)
        x = n + 0.80861 * pow(n, 0.33333) + 0.07249 / pow(n, 0.33333);
    }
    // 迭代过程的起始点设为 L=0
    L=0;
    L15:
        // Store the current value of x into x0
        x0 = x;
        // Calculate derivatives and other factors related to function jyndd
        jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
        // Update x using Newton's method
        x -= djn / fjn;
        // Check convergence condition for Newton's method
        if (fabs(x - x0) > 1e-11) goto L15;
        // Increment L by 1
        L += 1;
        // Store the current value of x into rj1 array
        rj1[L - 1] = x;
        // Check if L is less than nt to decide whether to continue iterating
        if (L < nt) {
            // Adjust x using the next step in the sequence
            x = rj1[L - 1] + (rj0[L] - rj0[L - 1]);
            goto L15;
        }

        // -- Newton method for y_{N,L}
        // Initial guess for y_{N,1} depending on the value of n
        if (n == 0) {
            x = 0.89357697;
        } else {
            // Calculate initial guess using a specific formula
            x = n + 0.93158 * pow(n, 0.33333) + 0.26035 / pow(n, 0.33333);
        }
        // Reset L to 0 for the next iteration
        L = 0;
    L20:
        // Store the current value of x into x0
        x0 = x;
        // Calculate derivatives and other factors related to function jyndd
        jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
        // Update x using Newton's method
        x -= byn / dyn;
        // Check convergence condition for Newton's method
        if (fabs(x - x0) > 1.0e-11) goto L20;
        // Increment L by 1
        L += 1;
        // Store the current value of x into ry0 array
        ry0[L - 1] = x;
        // Determine initial guess for y_{N,L+1} depending on L
        if (L == 1) {
            if (n == 0) {
                x = 3.957678419314858;
            } else {
                // Calculate initial guess using specific expansions and coefficients
                x = n + 2.59626 * pow(n, 0.33333) + 2.022183 / pow(n, 0.33333);
            }
        } else {
            // Approximate next root using a linear growth pattern
            x = ry0[L - 1] + (ry0[L - 1] - ry0[L - 2]);
        }
        // Perform additional iterations if L is less than or equal to n+10
        if (L <= n + 10) {
            // Calculate derivatives and adjust x using additional conditions
            jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
            h = atan(fabs(dyn) / sqrt(fabs(fyn * byn)));
            b = -dyn / (byn * tan(h));
            x -= (h - pi / 2) / b;
        }
        // Check if L is less than nt to decide whether to continue iterating
        if (L < nt) goto L20;

        // -- Newton method for y_{N,L+1}'
        // Initial guess for y_{N,L+1}' depending on the value of n
        if (n == 0) {
            x = 2.67257;
        } else {
            // Calculate initial guess using a specific formula
            x = n + 1.8211 * pow(n, 0.33333) + 0.94001 / pow(n, 0.33333);
        }
        // Reset L to 0 for the next iteration
        L = 0;
    L25:
        // Store the current value of x into x0
        x0 = x;
        // Calculate derivatives and other factors related to function jyndd
        jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
        // Update x using Newton's method
        x -= dyn / fyn;
        // Check convergence condition for Newton's method
        if (fabs(x - x0) > 1.0e-11) goto L25;
        // Increment L by 1
        L += 1;
        // Store the current value of x into ry1 array
        ry1[L - 1] = x;
        // Check if L is less than nt to decide whether to continue iterating
        if (L < nt) {
            // Adjust x using the next step in the sequence
            x = ry1[L - 1] + (ry0[L] - ry0[L - 1]);
            goto L25;
        }
        // Exit the function
        return;
    }

    template <typename T>
    inline void kmn(int m, int n, T c, T cv, int kd, T *df, T *dn, T *ck1, T *ck2) {

        // ===================================================
        // Purpose: Compute the expansion coefficients of the
        //          prolate and oblate spheroidal functions
        //          and joining factors
        // ===================================================

        int nm, nn, ip, k, i, l, j;
        T cs, gk0, gk1, gk2, gk3, t, r, dnp, su0, sw, r1, r2, r3, sa0, r4, r5, g0, sb0;
        // Calculate nm and nn based on input parameters
        nm = 25 + (int)(0.5 * (n - m) + c);
        nn = nm + m;
        // Allocate memory for arrays u, v, w, tp, and rk
        T *u = (T *)malloc((nn + 4) * sizeof(T));
        T *v = (T *)malloc((nn + 4) * sizeof(T));
        T *w = (T *)malloc((nn + 4) * sizeof(T));
        T *tp = (T *)malloc((nn + 4) * sizeof(T));
        T *rk = (T *)malloc((nn + 4) * sizeof(T));

        // Set a small value for epsilon
        const T eps = 1.0e-14;

        // Compute cs, ck1, and ck2 based on input parameters
        cs = c * c * kd;
        *ck1 = 0.0;
        *ck2 = 0.0;

        // Determine the parity of (n - m) to set ip
        ip = ((n - m) % 2 == 0 ? 0 : 1);
        k = 0;
    for (i = 1; i <= nn + 3; i++) {
        // 计算 k 的值，根据 ip 的条件选择不同的公式
        k = (ip == 0 ? -2 * (i - 1) : -(2 * i - 3));
        // 计算 gk0, gk1, gk2, gk3 四个变量的值
        gk0 = 2.0 * m + k;
        gk1 = (m + k) * (m + k + 1.0);
        gk2 = 2.0 * (m + k) - 1.0;
        gk3 = 2.0 * (m + k) + 3.0;

        // 计算 u[i-1], v[i-1], w[i-1] 三个数组的元素值
        u[i - 1] = gk0 * (gk0 - 1.0) * cs / (gk2 * (gk2 + 2.0));
        v[i - 1] = gk1 - cv + (2.0 * (gk1 - m * m) - 1.0) * cs / (gk2 * gk3);
        w[i - 1] = (k + 1.0) * (k + 2.0) * cs / ((gk2 + 2.0) * gk3);
    }

    for (k = 1; k <= m; k++) {
        // 初始化 t 的值为 v[m]
        t = v[m];
        // 计算 rk[k-1] 的值
        for (l = 0; l <= m - k - 1; l++)
            t = v[m - l - 1] - w[m - l] * u[m - l - 1] / t;

        rk[k - 1] = -u[k - 1] / t;
    }

    // 初始化 r 的值为 1.0
    r = 1.0;
    // 计算 dn 数组的值
    for (k = 1; k <= m; k++) {
        r = r * rk[k - 1];
        dn[k - 1] = df[0] * r;
    }

    // 初始化 tp[nn-1] 的值为 v[nn]
    tp[nn - 1] = v[nn];

    for (k = nn - 1; k >= m + 1; k--) {
        // 计算 tp[k-1] 的值
        tp[k - 1] = v[k] - w[k + 1] * u[k] / tp[k];

        if (k > m + 1)
            // 计算 rk[k-1] 的值
            rk[k - 1] = -u[k - 1] / tp[k - 1];
    }

    // 计算 dnp 的值
    dnp = (m == 0 ? df[0] : dn[m - 1]);
    dn[m] = pow(-1, ip) * dnp * cs / ((2.0 * m - 1.0) * (2.0 * m + 1.0 - 4.0 * ip) * tp[m]);

    for (k = m + 2; k <= nn; k++)
        // 更新 dn 数组的值
        dn[k - 1] *= rk[k - 1];

    // 初始化 r1 的值为 1.0
    r1 = 1.0;
    // 计算 su0 的值
    for (j = 1; j <= (n + m + ip) / 2; j++) {
        r1 = r1*(j + 0.5 * (n + m + ip));
    }
    // 初始化 r 的值为 1.0
    r = 1.0;
    // 计算 r 的阶乘
    for (j = 1; j <= 2 * m + ip; ++j){
        r *= j;
    }
    // 计算 su0 的值
    su0 = r * df[0];
    // 初始化 sw 的值为 0.0
    sw = 0.0;

    for (k = 2; k <= nm; ++k) {
        // 计算 su0 的值
        r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5);
        su0 = su0 + r * df[k - 1];
        // 判断是否满足收敛条件，若满足则退出循环
        if (k > (n - m) / 2 && fabs((su0 - sw) / su0) < eps) { break; }
        sw = su0;
    }

    if (kd != 1) {
        // 初始化 r2 的值为 1.0
        r2 = 1.0;

        for (j = 1; j <= m; ++j)
            // 计算 r2 的值
            r2 = 2.0 * c * r2 * j;

        // 初始化 r3 的值为 1.0
        r3 = 1.0;

        for (j = 1; j <= (n - m - ip) / 2; ++j)
            // 计算 r3 的值
            r3 = r3 * j;

        // 计算 sa0 的值
        sa0 = (2.0 * (m + ip) + 1.0) * r1 / (pow(2.0, n) * pow(c, ip) * r2 * r3 * df[0]);
        *ck1 = sa0 * su0;

        if (kd == -1) {
            free(u); free(v); free(w); free(tp); free(rk);
            return;
        }
    }

    // 初始化 r4 的值为 1.0
    r4 = 1.0;
    // 计算 r4 的阶乘
    for (j = 1; j <= (n - m - ip) / 2; ++j) {
        r4 *= 4.0 * j;
    }
    // 初始化 r5 的值为 1.0
    r5 = 1.0;
    // 计算 r5 的值
    for (j = 1; j <= m; ++j)
        r5 = r5 * (j + m) / c;

    // 计算 g0 的值
    g0 = dn[m - 1];

    if (m == 0)
        g0 = df[0];

    // 计算 sb0 的值
    sb0 = (ip + 1.0) * pow(c, ip + 1) / (2.0 * ip * (m - 2.0) + 1.0) / (2.0 * m - 1.0);
    *ck2 = pow(-1, ip) * sb0 * r4 * r5 * g0 / r1 * su0;

    // 释放动态分配的内存
    free(u); free(v); free(w); free(tp); free(rk);
    return;
inline void lamn(int n, double x, int *nm, double *bl, double *dl) {

    // =========================================================
    // Purpose: Compute lambda functions and their derivatives
    // Input:   x --- Argument of lambda function
    //          n --- Order of lambda function
    // Output:  BL(n) --- Lambda function of order n
    //          DL(n) --- Derivative of lambda function
    //          NM --- Highest order computed
    // Routines called:
    //          MSTA1 and MSTA2 for computing the start
    //          point for backward recurrence
    // =========================================================

    int i, k, m;
    double bk, r, uk, bs, f, f0, f1, bg, r0, x2;

    *nm = n;  // 设置最高计算阶数为 n

    // 处理 x 接近零的情况
    if (fabs(x) < 1e-100) {
        for (k = 0; k <= n; k++) {
            bl[k] = 0.0;  // 将 BL 数组初始化为 0
            dl[k] = 0.0;  // 将 DL 数组初始化为 0
        }
        bl[0] = 1.0;  // 当 k=0 时，BL(0) 设置为 1.0
        dl[1] = 0.5;  // 当 k=1 时，DL(1) 设置为 0.5
        return;  // 返回
    }

    // 处理 x 小于等于 12.0 的情况
    if (x <= 12.0) {
        x2 = x * x;  // 计算 x 的平方
        for (k = 0; k <= n; k++) {
            bk = 1.0;  // 初始化 bk 为 1.0
            r = 1.0;  // 初始化 r 为 1.0
            // 进行逆向递推的计算，最多进行 50 次迭代
            for (i = 1; i <= 50; i++) {
                r = -0.25 * r * x2 / (i * (i + k));  // 递推公式
                bk += r;  // 累加结果到 bk

                // 如果 r 的绝对值小于 bk 的绝对值乘以 1.0e-15，则停止迭代
                if (fabs(r) < fabs(bk) * 1.0e-15) { break; }
            }
            bl[k] = bk;  // 将计算得到的 bk 存入 BL 数组
            if (k >= 1) {
                dl[k - 1] = -0.5 * x / k * bk;  // 计算 DL 数组
            }
        }
        uk = 1.0;  // 初始化 uk 为 1.0
        r = 1.0;  // 重置 r 为 1.0
        // 进行逆向递推的计算，最多进行 50 次迭代
        for (i = 1; i <= 50; i++) {
            r = -0.25 * r * x2 / (i * (i + n + 1.0));  // 递推公式
            uk += r;  // 累加结果到 uk

            // 如果 r 的绝对值小于 uk 的绝对值乘以 1.0e-15，则停止迭代
            if (fabs(r) < fabs(uk) * 1.0e-15) { break; }
        }
        dl[n] = -0.5 * x / (n + 1.0) * uk;  // 计算 DL 数组
        return;  // 返回
    }

    // 处理 n=0 的情况
    if (n == 0) {
        *nm = 1;  // 设置最高计算阶数为 1
    }

    // 调用 MSTA1 函数计算起始点 m
    m = msta1(x, 200);
    // 如果 m 小于当前计算的最高阶数，则更新最高阶数
    if (m < *nm) {
        *nm = m;
    } else {
        // 否则调用 MSTA2 函数计算起始点 m
        m = msta2(x, *nm, 15);
    }

    bs = 0.0;  // 初始化 bs 为 0.0
    f = 0.0;  // 初始化 f 为 0.0
    f0 = 0.0;  // 初始化 f0 为 0.0
    f1 = 1e-100;  // 初始化 f1 为 1e-100
    // 从 m 开始逆向计算
    for (k = m; k >= 0; k--) {
        f = 2.0 * (k + 1.0) * f1 / x - f0;  // 计算 f
        if (k <= *nm) {
            bl[k] = f;  // 将计算得到的 f 存入 BL 数组
        }
        if (k % 2 == 0) {
            bs += 2.0 * f;  // 计算 bs
        }
        f0 = f1;  // 更新 f0
        f1 = f;  // 更新 f1
    }
    bg = bs - f;  // 计算 bg
    for (k = 0; k <= *nm; k++) {
        bl[k] /= bg;  // 对 BL 数组进行归一化
    }
    r0 = 1.0;  // 初始化 r0 为 1.0
    for (k = 1; k <= *nm; k++) {
        r0 = 2.0 * r0 * k / x;  // 计算 r0
        bl[k] *= r0;  // 更新 BL 数组
    }
    dl[0] = -0.5 * x * bl[1];  // 计算 DL 数组的第一个元素
    for (k = 1; k <= *nm; k++) {
        dl[k] = 2.0 * k / x * (bl[k - 1] - bl[k]);  // 计算 DL 数组
    }
    return;  // 返回
}
    // 根据公式设置变量和常量，用于计算贝塞尔函数的递归过程和gamma函数
    // =========================================================

    int i, n, k, j, k0, m;  // 定义整型变量 i, n, k, j, k0, m
    double cs, ga, fac, r0, f0, f1, f2, f, xk, vv;  // 定义双精度浮点型变量 cs, ga, fac, r0, f0, f1, f2, f, xk, vv
    double x2, v0, vk, bk, r, uk, qx, px, rp, a0, ck, sk, bjv0, bjv1;  // 定义双精度浮点型变量 x2, v0, vk, bk, r, uk, qx, px, rp, a0, ck, sk, bjv0, bjv1
    const double pi = 3.141592653589793;  // 定义常量 pi，值为圆周率
    const double rp2 = 0.63661977236758;  // 定义常量 rp2，值为 sqrt(2/pi)

    x = fabs(x);  // 将 x 取绝对值
    x2 = x * x;  // 计算 x 的平方
    n = (int)v;  // 将 v 转换为整型，赋值给 n
    v0 = v - n;  // 计算 v 的小数部分，赋值给 v0
    *vm = v;  // 将 v 赋值给指针变量 vm 所指向的位置

    if (x <= 12.0) {  // 如果 x 小于等于 12.0
        for (k = 0; k <= n; k++) {  // 循环从 k = 0 到 k <= n
            vk = v0 + k;  // 计算 vk，即 v0 + k
            bk = 1.0;  // 初始化 bk 为 1.0
            r = 1.0;  // 初始化 r 为 1.0

            for (i = 1; i <= 50; i++) {  // 内循环 i 从 1 到 50
                r = -0.25 * r * x2 / (i * (i + vk));  // 更新 r 的值
                bk = bk + r;  // 更新 bk 的值

                if (fabs(r) < fabs(bk) * 1.0e-15)  // 如果 r 的绝对值小于 bk 的绝对值乘以 1.0e-15
                    break;  // 跳出内循环
            }
            vl[k] = bk;  // 将 bk 赋值给 vl[k]

            uk = 1.0;  // 初始化 uk 为 1.0
            r = 1.0;  // 初始化 r 为 1.0

            for (i = 1; i <= 50; i++) {  // 再次内循环 i 从 1 到 50
                r = -0.25 * r * x2 / (i * (i + vk + 1.0));  // 更新 r 的值
                uk = uk + r;  // 更新 uk 的值

                if (fabs(r) < fabs(uk) * 1.0e-15)  // 如果 r 的绝对值小于 uk 的绝对值乘以 1.0e-15
                    break;  // 跳出内循环
            }
            dl[k] = -0.5 * x / (vk + 1.0) * uk;  // 将计算结果赋值给 dl[k]
        }
        return;  // 返回
    }

    k0 = (x >= 50.0) ? 8 : ((x >= 35.0) ? 10 : 11);  // 根据 x 的大小确定 k0 的值
    bjv0 = 0.0;  // 初始化 bjv0 为 0.0
    bjv1 = 0.0;  // 初始化 bjv1 为 0.0

    for (j = 0; j <= 1; j++) {  // 循环 j 从 0 到 1
        vv = 4.0 * (j + v0) * (j + v0);  // 计算 vv 的值
        px = 1.0;  // 初始化 px 为 1.0
        rp = 1.0;  // 初始化 rp 为 1.0

        for (k = 1; k <= k0; k++) {  // 内循环 k 从 1 到 k0
            rp = -0.78125e-2 * rp * (vv - pow((4.0 * k - 3.0), 2.0)) * (vv - pow((4.0 * k - 1.0), 2.0)) / (k * (2.0 * k - 1.0) * x2);  // 更新 rp 的值
            px += rp;  // 更新 px 的值
        }

        qx = 1.0;  // 初始化 qx 为 1.0
        rp = 1.0;  // 初始化 rp 为 1.0

        for (k = 1; k <= k0; k++) {  // 再次内循环 k 从 1 到 k0
            rp = -0.78125e-2 * rp * (vv - pow((4.0 * k - 1.0), 2.0)) * (vv - pow((4.0 * k + 1.0), 2.0)) / (k * (2.0 * k + 1.0) * x2);  // 更新 rp 的值
            qx += rp;  // 更新 qx 的值
        }

        qx = 0.125 * (vv - 1.0) * qx / x;  // 计算 qx 的值
        xk = x - (0.5 * (j + v0) + 0.25) * pi;  // 计算 xk 的值
        a0 = sqrt(rp2 / x);  // 计算 a0 的值
        ck = cos(xk);  // 计算 ck 的值
        sk = sin(xk);  // 计算 sk 的值

        if (j == 0) bjv0 = a0 * (px * ck - qx * sk);  // 如果 j 等于 0，则更新 bjv0 的值
        if (j == 1) bjv1 = a0 * (px * ck - qx * sk);  // 如果 j 等于 1，则更新 bjv1 的值
    }

    if (v0 == 0.0) {  // 如果 v0 等于 0.0
        ga = 1.0;  // 设置 ga 为 1.0
    } else {
        ga = gam0(v0);  // 否则调用 gam0 函数计算 gamma 函数值，并赋给 ga
        ga *= v0;  // 将 ga 乘以 v0
    }

    fac = pow(2.0 / x, v0) * ga;  // 计算 fac 的值
    vl[0] = bjv0;  // 将 bjv0 赋给 vl[0]
    dl[0] = -bjv1 + v0 / x * bjv0;  // 计算 dl[0] 的值
    vl[1] = bjv1;  // 将 bjv1 赋给 vl[1]
    dl[1] = bjv0 - (1.0 + v0) / x * bjv1;  // 计算 dl[1] 的值
    r0 = 2.0 * (1.0 + v0) / x;  // 计算 r0 的值

    if (n <= 1) {  // 如果 n 小于等于 1
        vl[0] *= fac;  // 更新 vl[0] 的值
        dl[0] = fac * dl[0] - v0 / x * vl[0];  // 更新 dl[0] 的值
        vl[1] *= fac * r0;  // 更新 vl[1] 的值
        dl[1] = fac * r0 * dl[1] - (1.0 + v0) / x * vl[1];  // 更新 dl[1] 的值
        return;  // 返回
    }

    if (n >= 2 && n <= (int
    } else if (n >= 2) {
        // 如果 n 大于等于 2，则执行以下操作
        m = msta1(x, 200);
        // 调用 msta1 函数计算 m 的值，参数为 x 和 200
        if (m < n) {
            // 如果 m 小于 n，则将 n 更新为 m
            n = m;
        } else {
            // 否则，调用 msta2 函数计算 m 的值，参数为 x, n, 15
            m = msta2(x, n, 15);
        }

        // 初始化 f, f2, f1 为浮点数
        f = 0.0;
        f2 = 0.0;
        f1 = 1.0e-100;

        // 逆序循环，从 m 到 0
        for (k = m; k >= 0; k--) {
            // 计算 f 的值
            f = 2.0 * (v0 + k + 1.0) / x * f1 - f2;
            // 如果 k 小于等于 n，则将 vl[k] 赋值为 f
            if (k <= n) vl[k] = f;
            // 更新 f2 和 f1 的值
            f2 = f1;
            f1 = f;
        }

        // 初始化 cs 为 0.0
        cs = 0.0;
        // 比较 fabs(bjv0) 和 fabs(bjv1) 的绝对值大小
        if (fabs(bjv0) > fabs(bjv1)) {
            // 如果 fabs(bjv0) 大于 fabs(bjv1)，则计算 cs 的值
            cs = bjv0 / f;
        } else {
            // 否则，计算 cs 的值
            cs = bjv1 / f2;
        }

        // 循环，对 vl[k] 应用 cs 的倍数
        for (k = 0; k <= n; k++) {
            vl[k] *= cs;
        }
    }

    // 将 vl[0] 乘以 fac
    vl[0] *= fac;
    // 循环，对 vl[j] 应用 fac * r0 的倍数，并计算 dl[j-1] 的值
    for (j = 1; j <= n; j++) {
        vl[j] *= fac * r0;
        dl[j - 1] = -0.5 * x / (j + v0) * vl[j];
        r0 = 2.0 * (j + v0 + 1) / x * r0;
    }

    // 计算 dl[n] 的值
    dl[n] = 2.0 * (v0 + n) * (vl[n - 1] - vl[n]) / x;
    // 将 *vm 的值设为 n + v0
    *vm = n + v0;
    // 返回
    return;
    int mx, neg_m, nv, j;
    double vx, pmv, v0, p0, p1, g1, g2;

    // =======================================================
    // Purpose: Compute the associated Legendre function
    //          Pmv(x) with an integer order and an arbitrary
    //          degree v, using recursion for large degrees
    // Input :  x   --- Argument of Pm(x)  ( -1 ≤ x ≤ 1 )
    //          m   --- Order of Pmv(x)
    //          v   --- Degree of Pmv(x)
    // Output:  PMV --- Pmv(x)
    // Routine called:  LPMV0
    // =======================================================

    // Initialize variables
    if ((x == -1.0) && (v != (int)v)) {
        // Handle special cases when x = -1 and v is not an integer
        if (m == 0) {
            pmv = -1e300;
        } else {
            pmv = 1e300;
        }
        return pmv;  // Return extreme value for special case
    }

    vx = v;
    mx = m;
    // DLMF 14.9.5
    if (v < 0) { vx = -vx -1.0; }  // Adjust vx if v is negative
    neg_m = 0;
    # 如果指数 m 小于 0，则需要特殊处理
    if (m < 0) {
        # 如果 (vx+m+1) 大于 0 或者 vx 不是整数，则设置 neg_m 为 1，并且取 mx 的相反数
        if (((vx+m+1) > 0) || (vx != (int)vx)) {
            neg_m = 1;
            mx = -m;
        } else {
            // 如果不符合 DLMF 14.9.3 中描述的情况，则返回 NaN
            return NAN;
        }
    }
    # 将 vx 转换为整数并赋给 nv
    nv = (int)vx;
    # 计算 v0，即 vx 减去 nv 的小数部分
    v0 = vx - nv;
    # 如果 nv 大于 2 并且大于 mx，则按照 AMS 8.5.3 / DLMF 14.10.3 进行向上递归处理
    if ((nv > 2) && (nv > mx)) {
        // 在度数上进行向上递归，根据 AMS 8.5.3 / DLMF 14.10.3
        p0 = lpmv0(v0+mx, mx, x);
        p1 = lpmv0(v0+mx+1, mx, x);
        pmv = p1;
        # 从 mx+2 到 nv 循环计算 pmv
        for (j = mx+2; j <= nv; j++) {
            pmv = ((2*(v0+j)-1)*x*p1 - (v0+j-1+mx)*p0) / (v0+j-mx);
            p0 = p1;
            p1 = pmv;
        }
    } else {
        # 否则直接使用 lpmv0 函数计算 pmv
        pmv = lpmv0(vx, mx, x);
    }
    # 如果 neg_m 不为 0 并且 pmv 的绝对值小于 1.e300，则按照 DLMF 14.9.3 进行修正
    if ((neg_m != 0) && (fabs(pmv) < 1.e300)) {
        // 根据 DLMF 14.9.3 进行修正
        g1 = gamma2(vx-mx+1);
        g2 = gamma2(vx+mx+1);
        pmv = pmv * g1 / g2 * pow(-1, mx);
    }
    # 返回计算得到的 pmv 值
    return pmv;
    // 定义整型变量 j, k, nv，以及双精度浮点变量 c0, v0, vs, pa, pss, pv0, pmv, r, r0, r1, r2, s, s0, s1, s2, qr, rg, xq
    int j, k, nv;
    double c0, v0, vs, pa, pss, pv0, pmv, r, r0, r1, r2, s, s0, s1, s2, qr, rg, xq;

    // 定义常量 pi, el, eps，并赋值
    const double pi = 3.141592653589793;
    const double el = 0.5772156649015329;
    const double eps = 1e-14;

    // 将 v 转换为整数部分 nv 和小数部分 v0
    nv = (int)v;
    v0 = v - nv;

    // 如果 x 等于 -1.0 且 v 不等于 nv，则返回特定的极限值
    if (x == -1.0 && v != nv) {
        if (m == 0)
            return -1.0e+300;
        if (m != 0)
            return 1.0e+300;
    }

    // 初始化系数 c0 为 1.0
    c0 = 1.0;

    // 如果 m 不等于 0，则计算相关系数 rg 和 r0
    if (m != 0) {
        rg = v * (v + m);
        for (j = 1; j <= m - 1; j++) {
            rg *= (v * v - j * j);
        }
        xq = sqrt(1.0 - x*x);
        r0 = 1.0;
        for (j = 1; j <= m; j++) {
            r0 = 0.5*r0*xq/j;
        }
        c0 = r0*rg;
    }

    // 如果 v0 等于 0.0，使用 DLMF 等式计算 pmv
    if (v0 == 0.0) {
        // DLMF 14.3.4, 14.7.17, 15.2.4
        pmv = 1.0;
        r = 1.0;
        for (k = 1; k <= nv - m; k++) {
            r = 0.5 * r * (-nv + m + k - 1.0) * (nv + m + k) / (k * (k + m)) * (1.0 + x);
            pmv += r;
        }
        // 返回结果，使用幂函数计算结果
        return pow(-1, nv)*c0*pmv;
    } else {
        if (x >= -0.35) {
            // 使用 DLMF 14.3.4, 15.2.1 中的算法计算概率密度函数值
            pmv = 1.0;
            r = 1.0;
            for (k = 1; k <= 100; k++) {
                // 计算递归关系以及累积概率密度函数值
                r = 0.5 * r * (-v + m + k - 1.0) * (v + m + k) / (k * (m + k)) * (1.0 - x);
                pmv += r;
                // 当达到一定迭代次数或者达到精度要求时停止迭代
                if (k > 12 && fabs(r / pmv) < eps) { break; }
            }
            // 返回计算得到的概率密度函数值
            return pow(-1, m)*c0*pmv;
        } else {
            // 使用 DLMF 14.3.5, 15.8.10 中的算法计算概率密度函数值
            vs = sin(v * pi) / pi;
            pv0 = 0.0;
            if (m != 0) {
                // 计算初始项 pv0
                qr = sqrt((1.0 - x) / (1.0 + x));
                r2 = 1.0;
                for (j = 1; j <= m; j++) {
                    r2 *= qr * j;
                }
                s0 = 1.0;
                r1 = 1.0;
                for (k = 1; k <= m - 1; k++) {
                    // 计算 s0 的递推关系
                    r1 = 0.5 * r1 * (-v + k - 1) * (v + k) / (k * (k - m)) * (1.0 + x);
                    s0 += r1;
                }
                // 计算 pv0 的值
                pv0 = -vs * r2 / m * s0;
            }

            // 计算 pa
            pa = 2.0 * (psi_spec(v) + el) + pi / tan(pi * v) + 1.0 / v;
            s1 = 0.0;
            for (j = 1; j <= m; j++) {
                // 计算 s1 的累加项
                s1 += (j * j + v * v) / (j * (j * j - v * v));
            }
            // 计算初始的 pmv 值
            pmv = pa + s1 - 1.0 / (m - v) + log(0.5 * (1.0 + x));
            r = 1.0;
            for (k = 1; k <= 100; k++) {
                // 计算递推关系以及累积概率密度函数值 pmv
                r = 0.5 * r * (-v + m + k - 1.0) * (v + m + k) / (k * (k + m)) * (1.0 + x);
                s = 0.0;
                for (j = 1; j <= m; j++)
                    // 计算 s 的累加项
                    s += ((k + j) * (k + j) + v * v) / ((k + j) * ((k + j) * (k + j) - v * v));

                s2 = 0.0;
                for (j = 1; j <= k; j++)
                    // 计算 s2 的递推关系
                    s2 = s2 + 1.0 / (j * (j * j - v * v));

                // 计算 pss 和 r2
                pss = pa + s + 2.0 * v * v * s2 - 1.0 / (m + k - v) + log(0.5 * (1.0 + x));
                r2 = pss * r;
                // 累积 pmv，并检查是否满足精度要求
                pmv += r2;
                if (fabs(r2 / pmv) < eps) { break; }
            }
            // 返回计算得到的概率密度函数值
            return pv0 + pmv * vs * c0;
        }
    }
    // 定义变量 l, ls, k, km 以及其他需要的变量
    int l, ls, k, km;
    // 定义变量 xq, q0, q00, q10, q01, q11, qf0, qf1, qm0, qm1, qg0, qg1, qh0, qh1, qh2, qmk, q0l, q1l, qf2, val

    // 初始化 val 为 0.0
    T xq, q0, q00, q10, q01, q11, qf0, qf1, qm0, qm1, qg0, qg1, qh0, qh1,\
           qh2, qmk, q0l, q1l, qf2, val;

    // 如果 x 的绝对值等于 1.0，则将 val 设置为极大值 1e300
    val = 0.0;
    if (fabs(x) == 1.0) { val = 1e300; }
    
    // 初始化数组 qm 和 qd，将每个元素都设置为 val
    for (k = 0; k <= n; k++) {
        qm[k] = val;
        qd[k] = val;
    }

    // 如果 x 的绝对值等于 1.0，则直接返回
    if (fabs(x) == 1.0) {
        return;
    }
    
    // ls 根据 x 的绝对值判断取值为 -1 或 1
    ls = (fabs(x) > 1.0 ? -1 : 1);

    // 计算 xq 为 sqrt(ls*(1.0 - x*x))
    xq = sqrt(ls*(1.0 - x*x));
    
    // 计算 q0 为 0.5 * log(fabs((x + 1.0) / (x - 1.0)))
    q0 = 0.5 * log(fabs((x + 1.0) / (x - 1.0)));
    q00 = q0;
    q10 = -1.0 / xq;
    q01 = x*q0 - 1.0;
    q11 = -ls*xq*(q0 + x / (1.0 - x*x));
    qf0 = q00;
    qf1 = q10;
    qm0 = 0.0;
    qm1 = 0.0;

    // 计算 Qmn(x) 和 Qmn'(x) 的递归过程
    for (k = 2; k <= m; k++) {
        qm0 = -2.0 * (k-1.0) / xq * x * qf1 - ls * (k-1.0) * (2.0 - k) * qf0;
        qf0 = qf1;
        qf1 = qm0;
    }

    // 根据 m 的值，设置 Qmn(x) 的初始值
    if (m == 0) {
        qm0 = q00;
    }
    if (m == 1) {
        qm0 = q10;
    }

    // 将 Qmn(x) 的值存入 qm 数组中
    qm[0] = qm0;
    # 检查 x 的绝对值是否小于 1.0001
    if (fabs(x) < 1.0001) {
        # 如果 m 等于 0 并且 n 大于 0，则执行以下操作
        if ((m == 0) && (n > 0)) {
            # 初始化 qf0 和 qf1 为预设值 q00 和 q01
            qf0 = q00;
            qf1 = q01;
            # 循环计算从 k=2 到 n 的 qm 数组元素
            for (k = 2; k <= n; k++) {
                # 使用递推公式计算 qf2
                qf2 = ((2.0 * k - 1.0) * x * qf1 - (k - 1.0) * qf0) / k;
                # 将计算结果赋值给 qm 数组相应位置
                qm[k] = qf2;
                # 更新 qf0 和 qf1 的值
                qf0 = qf1;
                qf1 = qf2;
            }
        }

        # 初始化 qg0 和 qg1 为预设值 q01 和 q11
        qg0 = q01;
        qg1 = q11;
        # 循环计算从 k=2 到 m 的 qm1 值
        for (k = 2; k <= m; k++) {
            # 使用递推公式计算 qm1
            qm1 = -2.0 * (k - 1.0) / xq * x * qg1 - ls * k * (3.0 - k) * qg0;
            # 更新 qg0 和 qg1 的值
            qg0 = qg1;
            qg1 = qm1;
        }

        # 如果 m 等于 0，则将 qm1 设置为 q01
        if (m == 0) {
            qm1 = q01;
        }
        # 如果 m 等于 1，则将 qm1 设置为 q11
        if (m == 1) {
            qm1 = q11;
        }

        # 将 qm1 赋值给 qm 数组的第一个元素
        qm[1] = qm1;

        # 如果 m 等于 1 并且 n 大于 1，则执行以下操作
        if ((m == 1) && (n > 1)) {
            # 初始化 qh0 和 qh1 为预设值 q10 和 q11
            qh0 = q10;
            qh1 = q11;
            # 循环计算从 k=2 到 n 的 qm 数组元素
            for (k = 2; k <= n; k++) {
                # 使用递推公式计算 qh2
                qh2 = ((2.0 * k - 1.0) * x * qh1 - k * qh0) / (k - 1.0);
                # 将计算结果赋值给 qm 数组相应位置
                qm[k] = qh2;
                # 更新 qh0 和 qh1 的值
                qh0 = qh1;
                qh1 = qh2;
            }
        } else if (m >= 2) {
            # 初始化 qg0、qg1、qh0、qh1 为预设值 q00、q01、q10、q11
            qg0 = q00;
            qg1 = q01;
            qh0 = q10;
            qh1 = q11;
            # 初始化 qmk 为 0.0
            qmk = 0.0;
            # 循环计算从 l=2 到 n 的 qm 数组元素
            for (l = 2; l <= n; l++) {
                # 使用递推公式计算 q0l 和 q1l
                q0l = ((2.0 * l - 1.0) * x * qg1 - (l - 1.0) * qg0) / l;
                q1l = ((2.0 * l - 1.0) * x * qh1 - l * qh0) / (l - 1.0);
                qf0 = q0l;
                qf1 = q1l;
                # 循环计算从 k=2 到 m 的 qmk 值
                for (k = 2; k <= m; k++) {
                    qmk = -2.0 * (k - 1.0) / xq * x * qf1 - ls * (k + l - 1.0) * (l + 2.0 - k) * qf0;
                    # 更新 qf0 和 qf1 的值
                    qf0 = qf1;
                    qf1 = qmk;
                }
                # 将 qmk 赋值给 qm 数组相应位置
                qm[l] = qmk;
                # 更新 qg0、qg1、qh0、qh1 的值
                qg0 = qg1;
                qg1 = q0l;
                qh0 = qh1;
                qh1 = q1l;
            }
        }
    } else {
        # 如果 x 的绝对值大于 1.1，则将 km 设置为 40 + m + n
        if (fabs(x) > 1.1) {
            km = 40 + m + n;
        }
        # 否则，根据公式计算 km 的值
        else {
            km = (40 + m + n) * (int)(-1.0 - 1.8 * log(x - 1.0));
        }
        # 初始化 qf2 和 qf1 为 0.0 和 1.0
        qf2 = 0.0;
        qf1 = 1.0;
        # 倒序循环计算从 k=km 到 0 的 qm 数组元素
        for (k = km; k >= 0; k--) {
            # 使用递推公式计算 qf0
            qf0 = ((2.0 * k + 3.0) * x * qf1 - (k + 2.0 - m) * qf2) / (k + m + 1.0);
            # 如果 k 小于等于 n，则将计算结果赋值给 qm 数组相应位置
            if (k <= n) {
                qm[k] = qf0;
            }
            # 更新 qf2 和 qf1 的值
            qf2 = qf1;
            qf1 = qf0;
        }
        # 循环计算从 k=0 到 n 的 qm 数组元素，并乘以 qm0/qf0
        for (k = 0; k <= n; k++) {
            qm[k] = qm[k] * qm0 / qf0;
        }
    }

    # 如果 x 的绝对值小于 1.0，则将 qm 数组每个元素乘以 (-1)^m
    if (fabs(x) < 1.0) {
        for (k = 0; k <= n; k++) {
            qm[k] = pow(-1, m) * qm[k];
        }
    }

    # 计算 qd[0]，使用递推公式
    qd[0] = ((1.0 - m) * qm[1] - x * qm[0]) / (x*x - 1.0);
    # 循环计算从 k=1 到 n 的 qd 数组元素，使用递推公式
    for (k = 1; k <= n; k++) {
        qd[k] = (k * x * qm[k] - (k + m) * qm[k-1]) / (x*x - 1.0);
    }
    # 函数返回，结束执行
    return;
}

inline int msta1(double x, int mp) {

    // ===================================================
    // Purpose: Determine the starting point for backward
    //          recurrence such that the magnitude of
    //          Jn(x) at that point is about 10^(-MP)
    // Input :  x     --- Argument of Jn(x)
    //          MP    --- Value of magnitude
    // Output:  MSTA1 --- Starting point
    // ===================================================

    int it, nn, n0, n1;
    double a0, f, f0, f1;

    a0 = fabs(x);  // Calculate absolute value of x
    n0 = (int)(1.1*a0) + 1;  // Initial approximation of starting point
    f0 = 0.5*log10(6.28*n0) - n0*log10(1.36*a0/n0)- mp;  // Initial function value
    n1 = n0 + 5;  // Increment the initial point
    f1 = 0.5*log10(6.28*n1) - n1*log10(1.36*a0/n1) - mp;  // Function value at incremented point
    for (it = 1; it <= 20; it++) {  // Iterate to refine the starting point
        nn = n1 - (n1 - n0) / (1.0 - f0/f1);  // Newton-Raphson iteration
        f = 0.5*log10(6.28*nn) - nn*log10(1.36*a0/nn) - mp;  // Function value at current point
        if (abs(nn-n1) < 1) { break; }  // Check convergence
        n0 = n1;  // Update points for next iteration
        f0 = f1;
        n1 = nn;
        f1 = f;
    }
    return nn;  // Return refined starting point
}


inline int msta2(double x, int n, int mp) {

    // ===================================================
    // Purpose: Determine the starting point for backward
    //          recurrence such that all Jn(x) has MP
    //          significant digits
    // Input :  x  --- Argument of Jn(x)
    //          n  --- Order of Jn(x)
    //          MP --- Significant digit
    // Output:  MSTA2 --- Starting point
    // ===================================================

    int it, n0, n1, nn;
    double a0, hmp, ejn, obj, f, f0, f1;

    a0 = fabs(x);  // Calculate absolute value of x
    hmp = 0.5*mp;  // Half of significant digits
    ejn = 0.5*log10(6.28*n) - n*log10(1.36*a0/n);  // Estimated significant digits of Jn(x)
    if (ejn <= hmp ) {
        obj = mp;  // Objective value if condition met
        n0 = (int)(1.1*a0) + 1;  // Initial approximation of starting point
    } else {
        obj = hmp + ejn;  // Objective value otherwise
        n0 = n;  // Start with order n
    }
    f0 = 0.5*log10(6.28*n0) - n0*log10(1.36*a0/n0) - obj;  // Initial function value
    n1 = n0 + 5;  // Increment the initial point
    f1 = 0.5*log10(6.28*n1) - n1*log10(1.36*a0/n1) - obj;  // Function value at incremented point
    for (it = 1; it <= 20; it++) {  // Iterate to refine the starting point
        nn = n1 - (n1 - n0) / (1.0 - f0/f1);  // Newton-Raphson iteration
        f = 0.5*log10(6.28*nn) - nn*log10(1.36*a0/nn) - obj;  // Function value at current point
        if (abs(nn-n1) < 1) { break; }  // Check convergence
        n0 = n1;  // Update points for next iteration
        f0 = f1;
        n1 = nn;
        f1 = f;
    }
    return nn + 10;  // Return refined starting point plus offset
}


template <typename T>
void mtu0(int kf, int m, T q, T x, T *csf, T *csd) {

    // ===============================================================
    // Purpose: Compute Mathieu functions cem(x,q) and sem(x,q)
    //          and their derivatives ( q ≥ 0 )
    // Input :  KF  --- Function code
    //                  KF=1 for computing cem(x,q) and cem'(x,q)
    //                  KF=2 for computing sem(x,q) and sem'(x,q)
    //          m   --- Order of Mathieu functions
    //          q   --- Parameter of Mathieu functions
    //          x   --- Argument of Mathieu functions (in degrees)
    // Output:  CSF --- cem(x,q) or sem(x,q)
    //          CSD --- cem'x,q) or sem'x,q)
    // Routines called:
    //      (1) CVA2 for computing the characteristic values
    //      (2) FCOEF for computing the expansion coefficients
    // ===============================================================
    // 初始化变量和常量
    int kd = 0, km = 0, ic, k;  // 定义整数变量 kd, km, ic, k
    T a, qm, xr;  // 定义泛型类型变量 a, qm, xr
    const T eps = 1.0e-14;  // 定义常量 eps，用于比较精度
    const T rd = 1.74532925199433e-2;  // 定义常量 rd，表示弧度转换因子

    // 根据条件设置 kd 的值
    if (kf == 1 && m == 2 * (int)(m / 2)) { kd = 1; }
    if (kf == 1 && m != 2 * (int)(m / 2)) { kd = 2; }
    if (kf == 2 && m != 2 * (int)(m / 2)) { kd = 3; }
    if (kf == 2 && m == 2 * (int)(m / 2)) { kd = 4; }

    // 调用函数 cva2 计算得到变量 a
    a = cva2(kd, m, q);

    // 根据 q 的值设置 qm
    if (q <= 1.0) {
        qm = 7.5 + 56.1 * sqrt(q) - 134.7 * q + 90.7 * sqrt(q) * q;
    } else {
        qm = 17.0 + 3.1 * sqrt(q) - 0.126 * q + 0.0037 * sqrt(q) * q;
    }

    // 计算变量 km 的值
    km = (int)(qm + 0.5 * m);

    // 如果 km 大于 251，则返回 NaN
    if (km > 251) {
        *csf = NAN;
        *csd = NAN;
        return;
    }

    // 分配内存给数组 fg，长度为 251，并调用函数 fcoef 计算得到 fg 的值
    T *fg = (T *) calloc(251, sizeof(T));
    fcoef(kd, m, q, a, fg);

    // 计算变量 ic 的值，并将 x 转换为弧度单位 xr
    ic = (int)(m / 2) + 1;
    xr = x * rd;

    // 初始化 csf 和 csd 的值
    *csf = 0.0;
    // 计算 csf 的值，根据 kd 的不同情况选择不同的计算方式
    for (k = 1; k <= km; k++) {
        if (kd == 1) {
            *csf += fg[k - 1] * cos((2*k - 2) * xr);
        } else if (kd == 2) {
            *csf += fg[k - 1] * cos((2*k - 1) * xr);
        } else if (kd == 3) {
            *csf += fg[k - 1] * sin((2*k - 1) * xr);
        } else if (kd == 4) {
            *csf += fg[k - 1] * sin(2*k*xr);
        }
        // 当 k 大于等于 ic 且 fg[k] 的绝对值小于 csf 的绝对值乘以 eps 时，跳出循环
        if ((k >= ic) && (fabs(fg[k]) < fabs(*csf) * eps)) {
            break;
        }
    }

    // 初始化 csd 的值
    *csd = 0.0;
    // 计算 csd 的值，根据 kd 的不同情况选择不同的计算方式
    for (k = 1; k <= km; k++) {
        if (kd == 1) {
            *csd -= (2*k - 2) * fg[k - 1] * sin((2*k - 2) * xr);
        } else if (kd == 2) {
            *csd -= (2*k - 1) * fg[k - 1] * sin((2*k - 1) * xr);
        } else if (kd == 3) {
            *csd += (2*k - 1) * fg[k - 1] * cos((2*k - 1) * xr);
        } else if (kd == 4) {
            *csd += 2.0 * k * fg[k - 1] * cos(2*k*xr);
        }
        // 当 k 大于等于 ic 且 fg[k-1] 的绝对值小于 csd 的绝对值乘以 eps 时，跳出循环
        if ((k >= ic) && (fabs(fg[k - 1]) < fabs(*csd) * eps)) {
            break;
        }
    }

    // 释放动态分配的数组内存
    free(fg);
    return;
    // 开始定义和实现函数 mtu12，计算修改后的马修函数及其导数
    // 使用模板进行泛型编程，支持不同类型的输入 T
template <typename T>
void mtu12(int kf, int kc, int m, T q, T x, T *f1r, T *d1r, T *f2r, T *d2r) {

    // ==============================================================
    // Purpose: Compute modified Mathieu functions of the first and
    //          second kinds, Mcm(1)(2)(x,q) and Msm(1)(2)(x,q),
    //          and their derivatives
    // Input:   KF --- Function code
    //                 KF=1 for computing Mcm(x,q)
    //                 KF=2 for computing Msm(x,q)
    //          KC --- Function Code
    //                 KC=1 for computing the first kind
    //                 KC=2 for computing the second kind
    //                      or Msm(2)(x,q) and Msm(2)'(x,q)
    //                 KC=3 for computing both the first
    //                      and second kinds
    //          m  --- Order of Mathieu functions
    //          q  --- Parameter of Mathieu functions ( q ≥ 0 )
    //          x  --- Argument of Mathieu functions
    // Output:  F1R --- Mcm(1)(x,q) or Msm(1)(x,q)
    //          D1R --- Derivative of Mcm(1)(x,q) or Msm(1)(x,q)
    //          F2R --- Mcm(2)(x,q) or Msm(2)(x,q)
    //          D2R --- Derivative of Mcm(2)(x,q) or Msm(2)(x,q)
    // Routines called:
    //      (1) CVA2 for computing the characteristic values
    //      (2) FCOEF for computing expansion coefficients
    //      (3) JYNB for computing Jn(x), Yn(x) and their
    //          derivatives
    // ==============================================================

    // 设置一个极小值
    T eps = 1.0e-14;
    // 定义一些局部变量
    T a, qm, c1, c2, u1, u2, w1, w2;
    int kd, km, ic, k, nm = 0;

    // 根据不同的 kf 和 m 的奇偶性来确定 kd
    if ((kf == 1) && (m % 2 == 0)) { kd = 1; }
    if ((kf == 1) && (m % 2 != 0)) { kd = 2; }
    if ((kf == 2) && (m % 2 != 0)) { kd = 3; }
    if ((kf == 2) && (m % 2 == 0)) { kd = 4; }

    // 调用 CVA2 函数计算特征值
    a = cva2(kd, m, q);

    // 根据 q 的大小选择不同的计算方式
    if (q <= 1.0) {
        qm = 7.5 + 56.1 * sqrt(q) - 134.7 * q + 90.7 * sqrt(q) * q;
    } else {
        qm = 17.0 + 3.1 * sqrt(q) - 0.126 * q + 0.0037 * sqrt(q) * q;
    }

    // 计算 km 的值
    km = (int)(qm + 0.5 * m);
    // 如果 km 大于等于 251，返回 NAN
    if (km >= 251) {
        *f1r = NAN;
        *d1r = NAN;
        *f2r = NAN;
        *d2r = NAN;
        return;
    }

    // 分配内存给一些数组
    T *fg = (T *) calloc(251, sizeof(T));
    T *bj1 = (T *) calloc(252, sizeof(T));
    T *dj1 = (T *) calloc(252, sizeof(T));
    T *bj2 = (T *) calloc(252, sizeof(T));
    T *dj2 = (T *) calloc(252, sizeof(T));
    T *by1 = (T *) calloc(252, sizeof(T));
    T *dy1 = (T *) calloc(252, sizeof(T));
    T *by2 = (T *) calloc(252, sizeof(T));
    T *dy2 = (T *) calloc(252, sizeof(T));

    // 调用 FCOEF 函数计算展开系数
    fcoef(kd, m, q, a, fg);
    ic = (int)(m / 2) + 1;
    if (kd == 4) { ic = m / 2; }

    // 计算一些中间变量
    c1 = exp(-x);
    c2 = exp(x);
    u1 = sqrt(q) * c1;
    u2 = sqrt(q) * c2;
    // 调用 JYNB 函数计算 Bessel 函数及其导数
    jynb(km+1, u1, &nm, bj1, dj1, by1, dy1);
    jynb(km+1, u2, &nm, bj2, dj2, by2, dy2);
    // 初始化 w1 和 w2
    w1 = 0.0;
    w2 = 0.0;
    # 如果 kc 不等于 2，则执行以下代码块
    if (kc != 2) {
        # 将 *f1r 设置为 0.0
        *f1r = 0.0;
        # 循环遍历 k 从 1 到 km
        for (k = 1; k <= km; k++) {
            # 根据 kd 的不同取值执行不同的计算
            if (kd == 1) {
                # 计算 *f1r，根据公式计算加权和
                *f1r += pow(-1, ic + k) * fg[k - 1] * bj1[k - 1] * bj2[k - 1];
            } else if (kd == 2 || kd == 3) {
                # 计算 *f1r，根据公式计算加权和
                *f1r += pow(-1, ic + k) * fg[k - 1] * (bj1[k - 1] * bj2[k] + pow(-1, kd) * bj1[k] * bj2[k - 1]);
            } else {
                # 计算 *f1r，根据公式计算加权和
                *f1r += pow(-1, ic + k) * fg[k - 1] * (bj1[k - 1] * bj2[k + 1] - bj1[k + 1] * bj2[k - 1]);
            }

            # 如果 k 大于等于 5 并且 *f1r 与 w1 的差的绝对值小于 *f1r 的绝对值乘以 eps，则跳出循环
            if (k >= 5 && fabs(*f1r - w1) < fabs(*f1r) * eps) { break; }
            # 更新 w1 为当前的 *f1r
            w1 = *f1r;
        }

        # 将 *f1r 除以 fg[0]
        *f1r /= fg[0];

        # 将 *d1r 设置为 0.0
        *d1r = 0.0;
        # 循环遍历 k 从 1 到 km
        for (k = 1; k <= km; k++) {
            # 根据 kd 的不同取值执行不同的计算
            if (kd == 1) {
                # 计算 *d1r，根据公式计算加权和
                *d1r += pow(-1, ic + k) * fg[k - 1] * (c2 * bj1[k - 1] * dj2[k - 1] - c1 * dj1[k - 1] * bj2[k - 1]);
            } else if (kd == 2 || kd == 3) {
                # 计算 *d1r，根据公式计算加权和
                *d1r += pow(-1, ic + k) * fg[k - 1] * (c2 * (bj1[k - 1] * dj2[k] + pow(-1, kd) * bj1[k] * dj2[k - 1])
                        - c1 * (dj1[k - 1] * bj2[k] + pow(-1, kd) * dj1[k] * bj2[k - 1]));
            } else {
                # 计算 *d1r，根据公式计算加权和
                *d1r += pow(-1, ic + k) * fg[k - 1] * (c2 * (bj1[k - 1] * dj2[k + 1] - bj1[k + 1] * dj2[k - 1])
                        - c1 * (dj1[k - 1] * bj2[k + 1] - dj1[k + 1] * bj2[k - 1]));
            }

            # 如果 k 大于等于 5 并且 *d1r 与 w2 的差的绝对值小于 *d1r 的绝对值乘以 eps，则跳出循环
            if (k >= 5 && fabs(*d1r - w2) < fabs(*d1r) * eps) { break; }
            # 更新 w2 为当前的 *d1r
            w2 = *d1r;
        }
        # 将 *d1r 乘以 sqrt(q) 再除以 fg[0]
        *d1r *= sqrt(q) / fg[0];

        # 如果 kc 等于 1，则释放内存并返回
        if (kc == 1) {
            free(fg);
            free(bj1); free(dj1); free(bj2); free(dj2);
            free(by1); free(dy1); free(by2); free(dy2);
            return;
        }
    }

    # 将 *f2r 设置为 0.0
    *f2r = 0.0;
    # 循环遍历 k 从 1 到 km
    for (k = 1; k <= km; k++) {
        # 根据 kd 的不同取值执行不同的计算
        if (kd == 1) {
            # 计算 *f2r，根据公式计算加权和
            *f2r += pow(-1, ic + k) * fg[k - 1] * bj1[k - 1] * by2[k - 1];
        } else if (kd == 2 || kd == 3) {
            # 计算 *f2r，根据公式计算加权和
            *f2r += pow(-1, ic + k) * fg[k - 1] * (bj1[k - 1] * by2[k] + pow(-1, kd) * bj1[k] * by2[k - 1]);
        } else {
            # 计算 *f2r，根据公式计算加权和
            *f2r += pow(-1, ic + k) * fg[k - 1] * (bj1[k - 1] * by2[k + 1] - bj1[k + 1] * by2[k - 1]);
        }

        # 如果 k 大于等于 5 并且 *f2r 与 w1 的差的绝对值小于 *f2r 的绝对值乘以 eps，则跳出循环
        if (k >= 5 && fabs(*f2r - w1) < fabs(*f2r) * eps) { break; }
        # 更新 w1 为当前的 *f2r
        w1 = *f2r;
    }
    # 将 *f2r 除以 fg[0]
    *f2r /= fg[0];

    # 将 *d2r 设置为 0.0
    *d2r = 0.0;
    # 循环遍历 k 从 1 到 km
    for (k = 1; k <= km; k++) {
        # 根据 kd 的不同取值执行不同的计算
        if (kd == 1) {
            # 计算 *d2r，根据公式计算加权和
            *d2r += pow(-1, ic + k) * fg[k - 1] * (c2 * bj1[k - 1] * dy2[k - 1] - c1 * dj1[k - 1] * by2[k - 1]);
        } else if (kd == 2 || kd == 3) {
            # 计算 *d2r，根据公式计算加权和
            *d2r += pow(-1, ic + k) * fg[k - 1] * (c2 * (bj1[k - 1] * dy2[k] + pow(-1, kd) * bj1[k] * dy2[k - 1])
                    - c1 * (dj1[k - 1] * by2[k] + pow(-1, kd) * dj1[k] * by2[k - 1]));
        } else {
            # 计算 *d2r，根据公式计算加权和
            *d2r += pow(-1, ic + k) * fg[k - 1] * (c2 * (bj1[k - 1] * dy2[k + 1] - bj1[k + 1] * dy2[k - 1])
                    - c1 * (dj1[k - 1] * by2[k + 1] - dj1[k + 1] * by2[k - 1]));
        }

        # 如果 k 大于等于 5 并且 *d2r 与 w2 的差的绝对值小于 *d2r 的绝对值乘以 eps，则跳出
    # 释放内存：释放指针 bj1 所指向的内存块
    free(bj1);
    # 释放内存：释放指针 dj1 所指向的内存块
    free(dj1);
    # 释放内存：释放指针 bj2 所指向的内存块
    free(bj2);
    # 释放内存：释放指针 dj2 所指向的内存块
    free(dj2);
    # 释放内存：释放指针 by1 所指向的内存块
    free(by1);
    # 释放内存：释放指针 dy1 所指向的内存块
    free(dy1);
    # 释放内存：释放指针 by2 所指向的内存块
    free(by2);
    # 释放内存：释放指针 dy2 所指向的内存块
    free(dy2);
    # 函数返回，结束函数执行
    return;
}


inline double psi_spec(double x) {

    // ======================================
    // Purpose: Compute Psi function
    // Input :  x  --- Argument of psi(x)
    // Output:  PS --- psi(x)
    // ======================================

    int k, n;
    double ps, s = 0.0, x2, xa = fabs(x);
    const double pi = 3.141592653589793;
    const double el = 0.5772156649015329;
    static const double a[8] = {
        -0.8333333333333e-01,
         0.83333333333333333e-02,
        -0.39682539682539683e-02,
         0.41666666666666667e-02,
        -0.75757575757575758e-02,
         0.21092796092796093e-01,
        -0.83333333333333333e-01,
         0.4432598039215686
    };

    if ((x == (int)x) && (x <= 0.0)) {
        return 1e300;  // 返回一个极大值，表示无穷大
    } else if (xa == (int)xa) {
        n = (int)xa;
        for (k = 1; k < n; k++) {
            s += 1.0 / k;  // 计算调和级数部分
        }
        ps = -el + s;  // 计算 psi(x) 的近似值
    } else if ((xa + 0.5) == (int)(xa + 0.5)) {
        n = (int)(xa - 0.5);
        for (k = 1; k < (n+1); k++) {
            s += 1.0 / (2.0*k - 1.0);  // 计算调和级数部分
        }
        ps = -el + 2.0*s - 1.386294361119891;  /* 2*log(2) */  // 计算 psi(x) 的近似值
    } else {
        if (xa < 10.0) {
            n = (10.0 - (int)xa);
            for (k = 0; k < n; k++) {
                s += 1.0 / (xa + k);  // 计算调和级数部分
            }
            xa += n;
        }
        x2 = 1.0 / (xa*xa);
        ps = log(xa) - 0.5 / xa;
        ps += x2*(((((((a[7]*x2+a[6])*x2+a[5])*x2+a[4])*x2+a[3])*x2+a[2])*x2+a[1])*x2+a[0];  // 计算 psi(x) 的近似值
        ps -= s;  // 调整 psi(x) 的近似值
    }
    if (x < 0.0) {
        ps -= pi*cos(pi*x)/sin(pi*x) + 1.0 / x;  // 若 x < 0 ，修正 psi(x) 的近似值
    }
    return ps;  // 返回最终 psi(x) 的值
}


template <typename T>
void qstar(int m, int n, T c, T ck1, T *ck, T *qs, T *qt) {
    int ip, i, l, k;
    T r, s, sk, qs0;
    T *ap = (T *) malloc(200*sizeof(T));
    ip = ((n - m) == 2 * ((n - m) / 2) ? 0 : 1);  // 根据条件设置 ip 的值
    r = 1.0 / pow(ck[0], 2);  // 计算 r 的值
    ap[0] = r;  // 初始化 ap[0]

    for (i = 1; i <= m; i++) {
        s = 0.0;
        for (l = 1; l <= i; l++) {
            sk = 0.0;
            for (k = 0; k <= l; k++)
                sk += ck[k] * ck[l - k];  // 计算 sk 的值
            s += sk * ap[i - l];  // 计算 s 的值
        }
        ap[i] = -r * s;  // 计算 ap[i] 的值
    }
    qs0 = ap[m - 1];  // 设置 qs0 的初值

    for (l = 1; l < m; l++) {
        r = 1.0;
        for (k = 1; k <= l; ++k) {
            r = r * (2.0 * k + ip) * (2.0 * k - 1.0 + ip) / pow(2.0 * k, 2);  // 计算 r 的值
        }
        qs0 += ap[m - l] * r;  // 计算 qs0 的值
    }
    *qs = pow(-1, ip) * (ck1) * (ck1 * qs0) / c;  // 计算 qs 的值
    *qt = -2.0 / (ck1) * (*qs);  // 计算 qt 的值
    free(ap);  // 释放动态分配的内存
    return;  // 结束函数
}


inline double refine(int kd, int m, double q, double a) {

    // =====================================================
    // Purpose: calculate the accurate characteristic value
    //          by the secant method
    // Input :  m --- Order of Mathieu functions
    //          q --- Parameter of Mathieu functions
    //          A --- Initial characteristic value
    // Output:  A --- Refineed characteristic value
    // Routine called:  CVF for computing the value of F for
    //                  characteristic equation
    // 初始化迭代变量和初始值
    int it, mj;
    double x1, x0, x, f, f0, f1;
    const double eps = 1e-14;

    // 计算增加后的 mj 值
    mj = 10 + m;
    // 设置初始值 x0，并计算其对应的函数值 f0
    x0 = a;
    f0 = cvf(kd, m, q, x0, mj);
    // 设置初始值 x1，并计算其对应的函数值 f1
    x1 = 1.002 * a;
    f1 = cvf(kd, m, q, x1, mj);

    // 开始迭代过程，最多迭代 100 次
    for (it = 1; it <= 100; it++) {
        // 增加 mj 的值
        mj += 1;
        // 计算当前迭代的新值 x
        x = x1 - (x1 - x0) / (1.0 - f0 / f1);
        // 计算当前 x 对应的函数值 f
        f = cvf(kd, m, q, x, mj);
        // 判断迭代是否收敛或者已达到精度要求
        if ((fabs(1.0 - x1 / x) < eps) || (f == 0.0)) { 
            // 若满足条件则跳出循环
            break; 
        }
        // 更新迭代变量和函数值
        x0 = x1;
        f0 = f1;
        x1 = x;
        f1 = f;
    }

    // 返回最终迭代得到的解 x
    return x;
// 分配内存以存储展开系数和函数值
T *ck = (T *) calloc(200, sizeof(T));
T *dj = (T *) calloc(252, sizeof(T));
T *sj = (T *) calloc(252, sizeof(T));
const T eps = 1.0e-14;

// 计算 nm1，它是 (n - m) / 2 的整数部分
int nm1 = (int)((n - m) / 2);
// 计算 ip，如果 n - m 等于 2 * nm1 则为 0，否则为 1
int ip = (n - m == 2 * nm1 ? 0 : 1);
// 计算 nm，其中包含 n, m, c 的函数计算次数
int nm = 25 + nm1 + (int)c;
// 根据 m 和 nm 计算 reg 的值
T reg = (m + nm > 80 ? 1.0e-200 : 1.0);
T r0 = reg;

// 计算 r0 的初始值
for (j = 1; j <= 2 * m + ip; ++j) {
    r0 *= j;
}
T r = r0;
// 计算初始的 suc 值
T suc = r * df[0];
T sw = 0.0;

// 主循环计算展开系数
for (k = 2; k <= nm; k++) {
    r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5);
    suc += r * df[k - 1];
    // 如果达到精度要求，则提前结束循环
    if ((k > nm1) && (fabs(suc - sw) < fabs(suc) * eps)) { break; }
    sw = suc;
}

// 当 x 等于 0 时的特殊处理
if (x == 0.0) {
    // 调用 SCKB 计算展开系数 ck
    sckb(m, n, c, df, ck);

    // 计算展开系数的和 sum
    T sum = 0.0;
    T sw1 = 0.0;
    for (j = 1; j <= nm; ++j) {
        sum += ck[j - 1];
        // 如果达到精度要求，则提前结束循环
        if (fabs(sum - sw1) < fabs(sum) * eps) { break; }
        sw1 = sum;
    }

    // 计算 r1 的初始值
    T r1 = 1.0;
    for (j = 1; j <= (n + m + ip) / 2; j++) {
        r1 = r1 * (j + 0.5 * (n + m + ip));
    }

    // 计算 r2 的初始值
    T r2 = 1.0;
    for (j = 1; j <= m; j++) {
        r2 *= 2.0 * c * j;
    }

    // 计算 r3 的初始值
    T r3 = 1.0;
    for (j = 1; j <= (n - m - ip) / 2; j++) {
        r3 *= j;
    }
    // 计算 sa0
    T sa0 = (2.0 * (m + ip) + 1.0) * r1 / (pow(2.0, n) * pow(c, ip) * r2 * r3);

    // 根据 ip 的值计算 *r1f 和 *r1d
    if (ip == 0) {
        *r1f = sum / (sa0 * suc) * df[0] * reg;
        *r1d = 0.0;
    } else if (ip == 1) {
        *r1f = 0.0;
        *r1d = sum / (sa0 * suc) * df[0] * reg;
    }
    // 释放分配的内存
    free(ck); free(dj); free(sj);
    return;
}

// 计算 cx
T cx = c * x;
int nm2 = 2 * nm + m;
// 调用 sphj 计算 sj 和 dj
sphj(cx, nm2, &nm2, sj, dj);

// 计算 a0
T a0 = pow(1.0 - kd / (x * x), 0.5 * m) / suc;
*r1f = 0.0;
sw = 0.0;
int lg = 0;

// 计算 *r1f
for (k = 1; k <= nm; ++k) {
    int l = 2 * k + m - n - 2 + ip;
    lg = (l % 4 == 0 ? 1 : -1);

    if (k == 1) {
        r = r0;
    } else {
        r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5);
    }

    int np = m + 2 * k - 2 + ip;
    *r1f += lg * r * df[k - 1] * sj[np];

    // 如果达到精度要求，则提前结束循环
    if ((k > nm1) && (fabs(*r1f - sw) < fabs(*r1f) * eps)) { break; }
    sw = *r1f;
}

// 计算 *r1f 的最终值
*r1f *= a0;
// 计算 b0
T b0 = kd * m / pow(x, 3.0) / (1.0 - kd / (x * x)) * (*r1f);

// 初始化 sud 和 sw
T sud = 0.0;
sw = 0.0;
    # 循环计算求和表达式的每一项
    for (k = 1; k <= nm; k++) {
        # 计算当前项对应的 l 值
        l = 2 * k + m - n - 2 + ip;
        # 计算当前项的符号 lg，根据 l 是否能被 4 整除确定符号
        lg = (l % 4 == 0 ? 1 : -1);

        # 根据当前 k 的值选择不同的 r 值
        if (k == 1) {
            r = r0;
        } else {
            # 根据前一项的 r 值计算当前项的 r 值
            r = r * (m + k - 1.0) * (m + k + ip - 1.5) / (k - 1.0) / (k + ip - 1.5);
        }

        # 计算当前项对应的 np 值
        np = m + 2 * k - 2 + ip;
        # 计算当前项的贡献 sud，累加到总和 sud 上
        sud = sud + lg * r * df[k - 1] * dj[np];

        # 如果满足条件，跳出循环
        if ((k > nm1) && (fabs(sud - sw) < fabs(sud) * eps)) { break; }
        # 更新 sw 为当前 sud 的值
        sw = sud;
    }
    # 计算最终结果 r1d
    *r1d = b0 + a0 * c * sud;
    # 释放动态分配的内存
    free(ck); free(dj); free(sj);
    # 函数无返回值，直接返回
    return;
# 函数定义，计算第二类椭圆和椭圆函数的径向函数
template <typename T>
inline void rmn2so(int m, int n, T c, T x, T cv, int kd, T *df, T *r2f, T *r2d) {

    // =============================================================
    // Purpose: Compute prolate and oblate spheroidal radial
    //          functions of the second kind for given m, n,
    //          c, x, cv, and kd
    // =============================================================


这段代码是一个模板函数，用于计算给定参数 m, n, c, x, cv, kd 的第二类椭圆和椭圆函数的径向函数。
    // 定义整数变量 nm, ip, j，以及浮点数变量 ck1, ck2, r1f, r1d, qs, qt, sum, sw, gf, gd, h0
    int nm, ip, j;
    T ck1, ck2, r1f, r1d, qs, qt, sum, sw, gf, gd, h0;
    // 定义常量 eps 和 pi
    const T eps = 1.0e-14;
    const T pi = 3.141592653589793;

    // 如果 df[0] 的绝对值小于等于 1.0e-280
    if (fabs(df[0]) <= 1.0e-280) {
        // 将 *r2f 和 *r2d 设置为极大值，然后返回
        *r2f = 1.0e+300;
        *r2d = 1.0e+300;
        return;
    }
    
    // 分配并初始化大小为 200 的数组 bk, ck, dn
    T *bk = (T *) calloc(200, sizeof(double));
    T *ck = (T *) calloc(200, sizeof(double));
    T *dn = (T *) calloc(200, sizeof(double));

    // 计算 nm 和 ip 的值
    nm = 25 + (int)((n - m) / 2 + c);
    ip = (n - m) % 2;
    
    // 调用各个函数计算所需的值
    // (1) 计算展开系数 ck
    sckb(m, n, c, df, ck);
    // (2) 计算连接因子 ck1 和 ck2
    kmn(m, n, c, cv, kd, df, dn, &ck1, &ck2);
    // (3) 计算因子 qs 和 qt
    qstar(m, n, c, ck1, ck, &qs, &qt);
    // (4) 计算展开系数 bk
    cbk(m, n, c, cv, qt, ck, bk);

    // 如果 x 等于 0.0
    if (x == 0.0) {
        // 初始化 sum 和 sw
        sum = 0.0;
        sw = 0.0;

        // 计算 sum 的累积值，直到收敛或者达到迭代次数 nm
        for (j = 0; j < nm; ++j) {
            sum += ck[j];
            // 如果累积的变化量小于阈值，则停止累积
            if (fabs(sum - sw) < fabs(sum) * eps) { break; }
            sw = sum;
        }

        // 根据 ip 的奇偶性分别计算 r1f 和 r1d
        if (ip == 0) {
            r1f = sum / ck1;
            *r2f = -0.5 * pi * qs * r1f;  // 计算并存储 r2f
            *r2d = qs * r1f + bk[0];      // 计算并存储 r2d
        } else {
            r1d = sum / ck1;
            *r2f = bk[0];                 // 计算并存储 r2f
            *r2d = -0.5 * pi * qs * r1d;  // 计算并存储 r2d
        }
    } else {
        // 如果 x 不等于 0.0，则计算 gmn, r1f, r1d, h0
        gmn(m, n, c, x, bk, &gf, &gd);
        rmn1(m, n, c, x, kd, df, &r1f, &r1d);
        h0 = atan(x) - 0.5 * pi;
        *r2f = qs * r1f * h0 + gf;     // 计算并存储 r2f
        *r2d = qs * (r1d * h0 + r1f / (1.0 + x * x)) + gd;  // 计算并存储 r2d
    }
    
    // 释放分配的内存空间
    free(bk); free(ck); free(dn);
    return;
    // 开始定义模板函数 rmn2sp，用于计算特定条件下的褐矮星形的球形辐射函数
    // 参数解释：
    //     m: 球形辐射函数中的一个参数
    //     n: 球形辐射函数中的一个参数
    //     c: 常数，影响球形辐射函数的计算
    //     x: 参数，影响球形辐射函数的计算
    //     cv: 常数，影响球形辐射函数的计算
    //     kd: 整数，影响球形辐射函数的计算
    //     df: 指向结果数组的指针，保存计算结果
    //     r2f: 指向结果数组的指针，保存计算结果
    //     r2d: 指向结果数组的指针，保存计算结果
template <typename T>
void rmn2sp(int m, int n, T c, T x, T cv, int kd, T *df, T *r2f, T *r2d) {

    // ======================================================
    // Purpose: Compute prolate spheroidal radial function
    //          of the second kind with a small argument
    // Routines called:
    //      (1) LPMNS for computing the associated Legendre
    //          functions of the first kind
    //      (2) LQMNS for computing the associated Legendre
    //          functions of the second kind
    //      (3) KMN for computing expansion coefficients
    //          and joining factors
    // ======================================================

    int k, j, j1, j2, l1, ki, nm3, sum, sdm;
    T ip, nm1, nm, nm2, su0, sw, sd0, su1, sd1, sd2, ga, r1, r2, r3,\
           sf, gb, spl, gc, sd, r4, spd1, spd2, su2, ck1, ck2;

    // 分配内存用于存储计算中间结果的数组
    T *pm = (T *) malloc(252*sizeof(T));
    T *pd = (T *) malloc(252*sizeof(T));
    T *qm = (T *) malloc(252*sizeof(T));
    T *qd = (T *) malloc(252*sizeof(T));
    T *dn = (T *) malloc(201*sizeof(T));
    const T eps = 1.0e-14;

    // 计算一些中间变量
    nm1 = (n - m) / 2;
    nm = 25.0 + nm1 + c;
    nm2 = 2 * nm + m;
    ip = (n - m) % 2;

    // 调用 KMN 函数计算系数和连接因子
    kmn(m, n, c, cv, kd, df, dn, &ck1, &ck2);

    // 调用 LPMNS 函数计算第一类关联Legendre函数
    lpmns(m, nm2, x, pm, pd);

    // 调用 LQMNS 函数计算第二类关联Legendre函数
    lqmns(m, nm2, x, qm, qd);

    // 初始化累加器变量
    su0 = 0.0;
    sw = 0.0;

    // 计算第一个累加和
    for (k = 1; k <= nm; k++) {
        j = 2 * k - 2 + m + ip;
        su0 += df[k - 1] * qm[j - 1];
        if ((k > nm1) && (fabs(su0 - sw) < fabs(su0) * eps)) { break; }
        sw = su0;
    }

    // 初始化第二个累加和
    sd0 = 0.0;
    for (k = 1; k <= nm; k++) {
        j = 2 * k - 2 + m + ip;
        sd0 += df[k - 1] * qd[j - 1];
        if (k > nm1 && fabs(sd0 - sw) < fabs(sd0) * eps) { break; }
        sw = sd0;
    }

    // 初始化第三个累加和
    su1 = 0.0;
    sd1 = 0.0;
    for (k = 1; k <= m; k++) {
        j = m - 2 * k + ip;
        if (j < 0) {
            j = -j - 1;
        }
        su1 += dn[k - 1] * qm[j - 1];
        sd1 += dn[k - 1] * qd[j - 1];
    }

    // 计算 ga 的值
    ga = pow((x - 1.0) / (x + 1.0), 0.5 * m);
    // 循环，k 从 1 到 m
    for (k = 1; k <= m; k++) {
        // 计算 j
        j = m - 2 * k + ip;
        // 如果 j >= 0，跳过当前循环
        if (j >= 0) { continue; }
        // 如果 j < 0，修正 j 的值
        if (j < 0) { j = -j - 1; }
        // 初始化 r1
        r1 = 1.0;
        // 计算 r1 的乘积
        for (j1 = 0; j1 < j; j1++) {
            r1 *= (m + j1);
        }
        // 初始化 r2
        r2 = 1.0;
        // 计算 r2 的乘积
        for (j2 = 1; j2 <= (m - j - 2); j2++) {
            r2 *= j2;
        }
        // 初始化 r3 和 sf
        r3 = 1.0;
        sf = 1.0;
        // 计算 r3 的值和 sf 的累加
        for (l1 = 1; l1 <= j; l1++) {
            r3 = 0.5 * r3 * (-j + l1 - 1.0) * (j + l1) / ((m + l1) * l1) * (1.0 - x);
            sf += r3;
        }
        // 如果 m - j >= 2，计算 gb
        if (m - j >= 2) {
            gb = (m - j - 1.0) * r2;
        }
        // 如果 m - j <= 1，设置 gb 为 1.0
        if (m - j <= 1) {
            gb = 1.0;
        }
        // 计算 spl
        spl = r1 * ga * gb * sf;
        // 计算 su1
        su1 += pow(-1, (j + m)) * dn[k-1] * spl;
        // 计算 spd1
        spd1 = m / (x * x - 1.0) * spl;
        // 计算 gc
        gc = 0.5 * j * (j + 1.0) / (m + 1.0);
        // 初始化 sd 和 r4
        sd = 1.0;
        r4 = 1.0;
        // 计算 sd 的累加和 r4 的值
        for (l1 = 1; l1 <= j - 1; l1++) {
            r4 = 0.5 * r4 * (-j + l1) * (j + l1 + 1.0) / ((m + l1 + 1.0) * l1) * (1.0 - x);
            sd += r4;
        }
        // 计算 spd2
        spd2 = r1 * ga * gb * gc * sd;
        // 计算 sd1
        sd1 += pow(-1, (j + m)) * dn[k - 1] * (spd1 + spd2);
    }
    // 初始化 su2、ki 和 nm3
    su2 = 0.0;
    ki = (2 * m + 1 + ip) / 2;
    nm3 = nm + ki;
    // 循环，k 从 ki 到 nm3
    for (k = ki; k <= nm3; k++) {
        // 计算 j
        j = 2 * k - 1 - m - ip;
        // 计算 su2 的累加
        su2 += dn[k - 1] * pm[j - 1];
        // 如果 j > m 且满足条件，跳出循环
        if ((j > m) && (fabs(su2 - sw) < fabs(su2) * eps)) { break; }
        // 更新 sw
        sw = su2;
    }
    // 初始化 sd2
    sd2 = 0.0;
    // 循环，k 从 ki 到 nm3-1
    for (k = ki; k < nm3; k++) {
        // 计算 j
        j = 2 * k - 1 - m - ip;
        // 计算 sd2 的累加
        sd2 += dn[k - 1] * pd[j - 1];
        // 如果 j > m 且满足条件，跳出循环
        if (j > m && fabs(sd2 - sw) < fabs(sd2) * eps) { break; }
        // 更新 sw
        sw = sd2;
    }
    // 计算 sum 和 sdm
    sum = su0 + su1 + su2;
    sdm = sd0 + sd1 + sd2;
    // 计算 *r2f 和 *r2d
    *r2f = sum / ck2;
    *r2d = sdm / ck2;
    // 释放内存
    free(pm); free(pd); free(qm); free(qd); free(dn);
    // 函数返回
    return;
template <typename T>
inline void rswfp(int m, int n, T c, T x, T cv, int kf, T *r1f, T *r1d, T *r2f, T *r2d) {
    // ==============================================================
    // Purpose: Compute prolate spheroid radial functions of the
    //          first and second kinds, and their derivatives
    // Input :  m  --- Mode parameter, m = 0,1,2,...
    //          n  --- Mode parameter, n = m,m+1,m+2,...
    //          c  --- Spheroidal parameter
    //          x  --- Argument of radial function ( x > 1.0 )
    //          cv --- Characteristic value
    //          KF --- Function code
    //                 KF=1 for the first kind
    //                 KF=2 for the second kind
    //                 KF=3 for both the first and second kinds
    // Output:  R1F --- Radial function of the first kind
    //          R1D --- Derivative of the radial function of
    //                  the first kind
    //          R2F --- Radial function of the second kind
    //          R2D --- Derivative of the radial function of
    //                  the second kind
    // Routines called:
    //      (1) SDMN for computing expansion coefficients dk
    //      (2) RMN1 for computing prolate and oblate radial
    //          functions of the first kind
    //      (3) RMN2L for computing prolate and oblate radial
    //          functions of the second kind for a large argument
    //      (4) RMN2SP for computing the prolate radial function
    //          of the second kind for a small argument
    // ==============================================================

    // Allocate memory for expansion coefficients
    T *df = (T *) malloc(200*sizeof(double));
    int id, kd = 1;

    // Compute expansion coefficients using SDMN function
    sdmn(m, n, c, cv, kd, df);

    // Compute prolate radial function of the first kind and its derivative
    if (kf != 2) {
        rmn1(m, n, c, x, kd, df, r1f, r1d);
    }

    // Compute prolate radial function of the second kind and its derivative
    // for a large argument, and check if additional computation is needed
    if (kf > 1) {
        rmn2l(m, n, c, x, kd, df, r2f, r2d, &id);
        if (id > -8) {
            rmn2sp(m, n, c, x, cv, kd, df, r2f, r2d);
        }
    }

    // Free allocated memory
    free(df);
    return;
}



template <typename T>
void rswfo(int m, int n, T c, T x, T cv, int kf, T *r1f, T *r1d, T *r2f, T *r2d) {
    // ==========================================================
    // Purpose: Compute oblate radial functions of the first
    //          and second kinds, and their derivatives
    // Input :  m  --- Mode parameter,  m = 0,1,2,...
    //          n  --- Mode parameter,  n = m,m+1,m+2,...
    //          c  --- Spheroidal parameter
    //          x  --- Argument (x ≥ 0)
    //          cv --- Characteristic value
    //          KF --- Function code
    //                 KF=1 for the first kind
    //                 KF=2 for the second kind
    //                 KF=3 for both the first and second kinds
    // Output:  R1F --- Radial function of the first kind
    //          R1D --- Derivative of the radial function of
    //                  the first kind
    //          R2F --- Radial function of the second kind
    //          R2D --- Derivative of the radial function of
    //                  the second kind
    // 分配内存以存储 200 个类型为 T 的元素的数组，用于存储中间计算结果
    T *df = (T *) malloc(200*sizeof(T));
    // 定义变量 id 和 kd，并初始化 kd 为 -1
    int id, kd = -1;

    // 调用 SDMN 函数计算扩展系数 dk
    sdmn(m, n, c, cv, kd, df);

    // 如果 kf 不等于 2，则执行下列语句块
    if (kf != 2) {
        // 调用 RMN1 函数计算第一类椭球坐标系中的径向函数和其导数
        rmn1(m, n, c, x, kd, df, r1f, r1d);
    }
    // 如果 kf 大于 1，则执行下列语句块
    if (kf > 1) {
        // 初始化 id 为 10
        id = 10;
        // 如果 x 大于 1e-8，则调用 RMN2L 函数计算大参数情况下的第二类椭球坐标系径向函数和其导数
        if (x > 1e-8) {
            rmn2l(m, n, c, x, kd, df, r2f, r2d, &id);
        }
        // 如果 id 大于 -1，则调用 RMN2SO 函数计算小参数情况下的第二类椭球坐标系的径向函数和其导数
        if (id > -1) {
            rmn2so(m, n, c, x, cv, kd, df, r2f, r2d);
        }
    }
    // 释放先前分配的内存空间 df
    free(df);
    // 函数结束，返回
    return;
}

template <typename T>
void sckb(int m, int n, T c, T *df, T *ck) {

    // ======================================================
    // Purpose: Compute the expansion coefficients of the
    //          prolate and oblate spheroidal functions
    // Input :  m  --- Mode parameter
    //          n  --- Mode parameter
    //          c  --- Spheroidal parameter
    //          DF(k) --- Expansion coefficients dk
    // Output:  CK(k) --- Expansion coefficients ck;
    //                    CK(1), CK(2), ... correspond to
    //                    c0, c2, ...
    // ======================================================

    int i, ip, i1, i2, k, nm;
    T reg, fac, sw, r, d1, d2, d3, sum, r1;

    if (c <= 1.0e-10) {
        c = 1.0e-10;
    }
    nm = 25 + (int)(0.5 * (n - m) + c);
    ip = (n - m) % 2;
    reg = ((m + nm) > 80 ? 1.0e-200 : 1.0);
    fac = -pow(0.5, m);
    sw = 0.0;

    for (k = 0; k < nm; k++) {
        fac = -fac;
        i1 = 2 * k + ip + 1;
        r = reg;

        for (i = i1; i <= i1 + 2 * m - 1; i++) {
            r *= i;
        }
        i2 = k + m + ip;
        for (i = i2; i <= i2 + k - 1; i++) {
            r *= (i + 0.5);
        }
        sum = r * df[k];
        for (i = k + 1; i <= nm; i++) {
            d1 = 2.0 * i + ip;
            d2 = 2.0 * m + d1;
            d3 = i + m + ip - 0.5;
            r = r * d2 * (d2 - 1.0) * i * (d3 + k) / (d1 * (d1 - 1.0) * (i - k) * d3);
            sum += r * df[i];
            if (fabs(sw - sum) < fabs(sum) * 1.0e-14) { break; }
            sw = sum;
        }
        r1 = reg;
        for (i = 2; i <= m + k; i++) { r1 *= i; }
        ck[k] = fac * sum / r1;
    }
}

template <typename T>
void sdmn(int m, int n, T c, T cv, int kd, T *df) {

    // =====================================================
    // Purpose: Compute the expansion coefficients of the
    //          prolate and oblate spheroidal functions, dk
    // Input :  m  --- Mode parameter
    //          n  --- Mode parameter
    //          c  --- Spheroidal parameter
    //          cv --- Characteristic value
    //          KD --- Function code
    //                 KD=1 for prolate; KD=-1 for oblate
    // Output:  DF(k) --- Expansion coefficients dk;
    //                    DF(1), DF(2), ... correspond to
    //                    d0, d2, ... for even n-m and d1,
    //                    d3, ... for odd n-m
    // =====================================================

    int nm, ip, k, kb;
    T cs, dk0, dk1, dk2, d2k, f, fs, f1, f0, fl, f2, su1,\
           su2, sw, r1, r3, r4, s0;

    nm = 25 + (int)(0.5 * (n - m) + c);

    if (c < 1e-10) {
        // Initialize DF array with zeros and set specific element
        for (int i = 1; i <= nm; ++i) {
            df[i-1] = 0.0;
        }
        df[(n - m) / 2] = 1.0;
        return;
    }

    // Allocate memory for arrays a, d, g
    T *a = (T *) calloc(nm + 2, sizeof(double));
    T *d = (T *) calloc(nm + 2, sizeof(double));
    T *g = (T *) calloc(nm + 2, sizeof(double));
    cs = c*c*kd;
    ip = (n - m) % 2;
    // 循环计算系数数组 a, d, g 中的每个元素
    for (int i = 1; i <= nm + 2; ++i) {
        // 根据 ip 的值确定 k 的计算方式
        k = (ip == 0 ? 2 * (i - 1) : 2 * i - 1);

        // 计算数组中的索引值
        dk0 = m + k;
        dk1 = m + k + 1;
        dk2 = 2 * (m + k);
        d2k = 2 * m + k;

        // 计算并存储系数 a[i-1], d[i-1], g[i-1]
        a[i - 1] = (d2k + 2.0) * (d2k + 1.0) / ((dk2 + 3.0) * (dk2 + 5.0)) * cs;
        d[i - 1] = dk0 * dk1 + (2.0 * dk0 * dk1 - 2.0 * m * m - 1.0) / ((dk2 - 1.0) * (dk2 + 3.0)) * cs;
        g[i - 1] = k * (k - 1.0) / ((dk2 - 3.0) * (dk2 - 1.0)) * cs;
    }

    // 初始化一些变量
    fs = 1.0;
    f1 = 0.0;
    f0 = 1.0e-100;
    kb = 0;
    df[nm] = 0.0;
    fl = 0.0;

    // 反向循环计算 df 数组的值
    for (int k = nm; k >= 1; k--) {
        // 计算当前 df[k-1] 的值
        f = -((d[k] - cv) * f0 + a[k] * f1) / g[k];

        // 根据条件更新 df 数组的值
        if (fabs(f) > fabs(df[k])) {
            df[k-1] = f;
            f1 = f0;
            f0 = f;

            // 如果 f 的绝对值过大，进行数值稳定处理
            if (fabs(f) > 1.0e+100) {
                for (int k1 = k; k1 <= nm; k1++)
                    df[k1 - 1] *= 1.0e-100;
                f1 *= 1.0e-100;
                f0 *= 1.0e-100;
            }
        } else {
            // 更新 kb, fl, df[0] 等变量的值
            kb = k;
            fl = df[k];
            f1 = 1.0e-100;
            f2 = -((d[0] - cv) / a[0]) * f1;
            df[0] = f1;

            // 根据 kb 的不同情况更新 fs 和 df 数组
            if (kb == 1) {
                fs = f2;
            } else if (kb == 2) {
                df[1] = f2;
                fs = -((d[1] - cv) * f2 + g[1] * f1) / a[1];
            } else {
                df[1] = f2;

                // 循环更新 df 数组的值
                for (int j = 3; j <= kb + 1; j++) {
                    f = -((d[j - 2] - cv) * f2 + g[j - 2] * f1) / a[j - 2];
                    if (j <= kb) {
                        df[j-1] = f;
                    }
                    // 如果 f 的绝对值过大，进行数值稳定处理
                    if (fabs(f) > 1.0e+100) {
                        for (int k1 = 1; k1 <= j; k1++) {
                            df[k1 - 1] *= 1.0e-100;
                        }
                        f *= 1.0e-100;
                        f2 *= 1.0e-100;
                    }
                    f1 = f2;
                    f2 = f;
                }
                fs = f;
            }
            break;
        }
    }

    // 初始化变量 su1 和 r1
    su1 = 0.0;
    r1 = 1.0;

    // 计算 su1 的值
    for (int j = m + ip + 1; j <= 2 * (m + ip); j++) {
        r1 *= j;
    }
    su1 = df[0] * r1;

    // 计算 su1 的累加值
    for (int k = 2; k <= kb; k++) {
        r1 = -r1 * (k + m + ip - 1.5) / (k - 1.0);
        su1 += r1 * df[k - 1];
    }

    // 初始化变量 su2 和 sw
    su2 = 0.0;
    sw = 0.0;

    // 计算 su2 的值
    for (int k = kb + 1; k <= nm; k++) {
        if (k != 1) {
            r1 = -r1 * (k + m + ip - 1.5) / (k - 1.0);
        }
        su2 += r1 * df[k - 1];

        // 检查 sw 和 su2 的差值是否足够小，以确定是否结束循环
        if (fabs(sw - su2) < fabs(su2) * 1.0e-14) { break; }
        sw = su2;
    }

    // 初始化变量 r3 和 r4
    r3 = 1.0;

    // 计算 r3 的值
    for (int j = 1; j <= (m + n + ip) / 2; j++) {
        r3 *= (j + 0.5 * (n + m + ip));
    }
    r4 = 1.0;

    // 计算 r4 的值
    for (int j = 1; j <= (n - m - ip) / 2; j++) {
        r4 *= -4.0 * j;
    }

    // 计算 s0 的值
    s0 = r3 / (fl * (su1 / fs) + su2) / r4;

    // 更新 df 数组的值
    for (int k = 1; k <= kb; ++k) {
        df[k - 1] *= fl / fs * s0;
    }
    for (int k = kb + 1; k <= nm; ++k) {
        df[k - 1] *= s0;
    }

    // 释放动态分配的内存并返回
    free(a); free(d); free(g);
    return;
    // TODO: Following array sizes should be decided dynamically
    
    // 分配大小为300的动态数组a，用于存储T类型的数据
    T *a = (T *) calloc(300, sizeof(T));
    
    // 分配大小为100的动态数组b，用于存储T类型的数据
    T *b = (T *) calloc(100, sizeof(T));
    
    // 分配大小为100的动态数组cv0，用于存储T类型的数据
    T *cv0 = (T *) calloc(100, sizeof(T));
    
    // 分配大小为300的动态数组d，用于存储T类型的数据
    T *d = (T *) calloc(300, sizeof(T));
    
    // 分配大小为300的动态数组e，用于存储T类型的数据
    T *e = (T *) calloc(300, sizeof(T));
    
    // 分配大小为300的动态数组f，用于存储T类型的数据
    T *f = (T *) calloc(300, sizeof(T));
    
    // 分配大小为300的动态数组g，用于存储T类型的数据
    T *g = (T *) calloc(300, sizeof(T));
    
    // 分配大小为100的动态数组h，用于存储T类型的数据
    T *h = (T *) calloc(100, sizeof(T));
    
    // 计算 icm 的值，icm = (n-m+2)/2
    icm = (n-m+2)/2;
    
    // 计算 nm 的值，nm = 10 + (int)(0.5*(n-m)+c)
    nm = 10 + (int)(0.5*(n-m)+c);
    
    // 计算 cs 的值，cs = c*c*kd
    cs = c*c*kd;
    
    // 初始化 k 为 0
    k = 0;
    // 外层循环控制计算两组系数的过程，l = 0 和 l = 1 分别代表两组不同的计算
    for (l = 0; l <= 1; l++) {
        // 内层循环，计算当前组中每个系数的值
        for (i = 1; i <= nm; i++) {
            // 根据 l 的不同，计算当前系数的索引 k
            k = (l == 0 ? 2*(i - 1) : 2*i - 1);
            // 计算相关的索引值
            dk0 = m + k;
            dk1 = m + k + 1;
            dk2 = 2*(m + k);
            d2k = 2*m + k;
            // 计算数组 a 中的值
            a[i-1] = (d2k+2.0)*(d2k+1.0)/((dk2+3.0)*(dk2+5.0))*cs;
            // 计算数组 d 中的值
            d[i-1] = dk0*dk1+(2.0*dk0*dk1-2.0*m*m-1.0)/((dk2-1.0)*(dk2+3.0))*cs;
            // 计算数组 g 中的值
            g[i-1] = k*(k-1.0)/((dk2-3.0)*(dk2-1.0))*cs;
        }
        // 计算数组 e 和 f 的值
        for (k = 2; k <= nm; k++) {
            e[k-1] = sqrt(a[k-2]*g[k-1]);
            f[k-1] = e[k-1]*e[k-1];
        }
        // 初始化数组 f 和 e 的第一个元素为 0
        f[0] = 0.0;
        e[0] = 0.0;
        // 计算 xa 和 xb 的值
        xa = d[nm-1] + fabs(e[nm-1]);
        xb = d[nm-1] - fabs(e[nm-1]);
        // 计算 nm1 的值
        nm1 = nm-1;
        // 更新 xa 和 xb 的值
        for (i = 1; i <= nm1; i++) {
            t = fabs(e[i-1])+fabs(e[i]);
            t1 = d[i-1] + t;
            if (xa < t1) { xa = t1; }
            t1 = d[i-1] - t;
            if (t1 < xb) { xb = t1; }
        }
        // 将计算得到的 xa 和 xb 存入数组 b 和 h
        for (i = 1; i <= icm; i++) {
            b[i-1] = xa;
            h[i-1] = xb;
        }
        // 进行排序和调整数组 b 和 h 的值
        for (k = 1; k <= icm; k++) {
            for (k1 = k; k1 <= icm; k1++) {
                if (b[k1-1] < b[k-1]) {
                    b[k-1] = b[k1-1];
                    break;
                }
            }
            if (k != 1) {
                if(h[k-1] < h[k-2]) { h[k-1] = h[k-2]; }
            }
            // 使用二分法计算 cv0 数组的值
            while (1) {
                x1 = (b[k-1]+h[k-1])/2.0;
                cv0[k-1] = x1;
                // 判断是否满足精度要求
                if (fabs((b[k-1] - h[k-1])/x1) < 1e-14) { break; }
                j = 0;
                s = 1.0;
                // 计算二分法的迭代过程
                for (i = 1; i <= nm; i++) {
                    if (s == 0.0) { s += 1e-30; }
                    t = f[i-1]/s;
                    s = d[i-1] - t - x1;
                    if (s < 0.0) { j += 1; }
                }
                // 根据迭代结果更新边界值 b 和 h
                if (j < k) {
                    h[k-1] = x1;
                } else {
                    b[k-1] = x1;
                    if (j >= icm) {
                        b[icm - 1] = x1;
                    } else {
                        if (h[j] < x1) { h[j] = x1; }
                        if (x1 < b[j-1]) { b[j-1] = x1; }
                    }
                }
            }
            cv0[k-1] = x1;
            // 根据 l 的值更新 eg 数组
            if (l == 0) eg[2*k-2] = cv0[k-1];
            if (l == 1) eg[2*k-1] = cv0[k-1];
        }
    }
    // 将最终计算结果存入 cv
    *cv = eg[n-m];
    // 释放动态分配的内存
    free(a); free(b); free(cv0); free(d); free(e); free(f); free(g); free(h);
    // 函数执行完毕，返回
    return;
}

template <typename T>
void sphj(T x, int n, int *nm, T *sj, T *dj) {
    // MODIFIED to ALLOW N=0 CASE (ALSO IN SPHY)
    //
    // =======================================================
    // Purpose: Compute spherical Bessel functions jn(x) and
    //          their derivatives
    // Input :  x --- Argument of jn(x)
    //          n --- Order of jn(x)  ( n = 0,1,… )
    // Output:  SJ(n) --- jn(x)
    //          DJ(n) --- jn'(x)
    //          NM --- Highest order computed
    // Routines called:
    //          MSTA1 and MSTA2 for computing the starting
    //          point for backward recurrence
    // =======================================================

    int k, m;
    T cs, f, f0, f1, sa, sb;

    *nm = n;
    // Check if x is very close to zero
    if (fabs(x) < 1e-100) {
        // Initialize sj and dj to zero
        for (k = 0; k <= n; k++) {
            sj[k] = 0.0;
            dj[k] = 0.0;
        }
        // Set initial values for sj and dj when n=0
        sj[0] = 1.0;
        if (n > 0) {
            dj[1] = 1.0 / 3.0;
        }
        return;
    }
    // Compute initial values of sj[0] and dj[0]
    sj[0] = sin(x)/x;
    dj[0] = (cos(x) - sin(x)/x)/x;
    // Return if n < 1
    if (n < 1) {
        return;
    }
    // Compute sj[1] and dj[1]
    sj[1] = (sj[0] - cos(x))/x;
    // Compute higher orders of sj and dj when n >= 2
    if (n >= 2) {
        sa = sj[0];
        sb = sj[1];
        // Compute starting index for backward recurrence
        m = msta1(x, 200);
        if (m < n) {
            *nm = m;
        } else {
            m = msta2(x, n, 15);
        }
        f = 0.0;
        f0 = 0.0;
        f1 = 1e-100;
        // Compute sj[k] using backward recurrence
        for (k = m; k >= 0; k--) {
            f = (2.0*k + 3.0)*f1/x - f0;
            if (k <= *nm) { sj[k] = f; }
            f0 = f1;
            f1 = f;
        }
        // Determine scaling factor cs and apply it to sj
        cs = (fabs(sa) > fabs(sb) ? sa/f : sb/f0);
        for (k = 0; k <= *nm; k++) {
            sj[k] *= cs;
        }
    }
    // Compute dj[k] for k > 0
    for (k = 1; k <= *nm; k++) {
        dj[k] = sj[k - 1] - (k + 1.0)*sj[k]/x;
    }
    return;
}


template <typename T>
inline void sphy(T x, int n, int *nm, T *sy, T *dy) {
    // ======================================================
    // Purpose: Compute spherical Bessel functions yn(x) and
    //          their derivatives
    // Input :  x --- Argument of yn(x) ( x ≥ 0 )
    //          n --- Order of yn(x) ( n = 0,1,… )
    // Output:  SY(n) --- yn(x)
    //          DY(n) --- yn'(x)
    //          NM --- Highest order computed
    // ======================================================

    T f, f0, f1;

    // Check if x is very small
    if (x < 1.0e-60) {
        // Initialize sy and dy to extreme values
        for (int k = 0; k <= n; ++k) {
            sy[k] = -1.0e300;
            dy[k] = 1.0e300;
        }
        *nm = n;
        return;
    }
    // Compute initial values of sy[0] and dy[0]
    sy[0] = -cos(x) / x;
    f0 = sy[0];
    dy[0] = (sin(x) + cos(x) / x) / x;

    // Return if n < 1
    if (n < 1) {
        *nm = n;
        return;
    }

    // Compute sy[1] and higher orders of sy and dy
    sy[1] = (sy[0] - sin(x)) / x;
    f1 = sy[1];

    for (int k = 2; k <= n; k++) {
        f = ((2.0 * k - 1.0) * f1 / x) - f0;
        sy[k] = f;
        // Check for overflow in sy[k]
        if (fabs(f) >= 1.0e300) {
            *nm = k - 1;
            return;
        }
        f0 = f1;
        f1 = f;
    }
    // Update *nm and compute dy[k] for k > 0
    *nm = n - 1;
    for (int k = 1; k <= *nm; k++) {
        dy[k] = sy[k - 1] - (k + 1.0) * sy[k] / x;
    }
    return;
}
```