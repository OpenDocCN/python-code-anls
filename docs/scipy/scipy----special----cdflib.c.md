# `D:\src\scipysrc\scipy\scipy\special\cdflib.c`

```
/*
 *
 * This file accompanied with the header file cdflib.h is a C rewrite of
 * the Fortran code with the following as its original description:
 *
 * Cumulative distribution functions, inverses and parameters for
 * Beta, Binomial, Chi-square, noncentral Chi-square, F, noncentral F, Gamma,
 * negative Binomial, Normal, Poisson, Student's t distributions.
 * It uses various TOMS algorithms and Abramowitz & Stegun, also Bus Dekker
 * zero-finding algorithm.
 *
 * The original Fortran code can be found at Netlib
 * https://www.netlib.org/random/
 *
 *
 * References
 * ----------
 *
 *  J. C. P. Bus, T. J. Dekker, Two Efficient Algorithms with Guaranteed
 *  Convergence for Finding a Zero of a Function, ACM Trans. Math. Software 1:4
 *  (1975) 330-345, DOI:10.1145/355656.355659
 *
 *  M. Abramowitz and I. A. Stegun (Eds.) (1964) Handbook of Mathematical
 *  Functions with Formulas, Graphs, and Mathematical Tables. National Bureau
 *  of Standards Applied Mathematics Series, U.S. Government Printing Office,
 *  Washington, D.C..
 *
 */

/*
 *
 * Copyright (C) 2024 SciPy developers
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * b. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * c. Names of the SciPy Developers may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "cdflib.h"

// 定义圆周率常量 PI
static const double PI = 3.1415926535897932384626433832795028841971693993751;

// 定义结构体 DinvrState，包含了用于反函数求解的状态参数
struct DinvrState
{
    double absstp;  // 绝对步长
    double abstol;  // 绝对容差
    double big;     // 大数
    double fbig;    // 大函数值
    double fx;      // 函数值
    double fsmall;  // 小函数值
    double relstp;  // 相对步长
    double reltol;  // 相对容差
    double small;   // 小数
    int status;     // 状态码
    double step;    // 步长
    double stpmul;  // 步长倍数
    double x;       // 自变量 x
    double xhi;     // x 的高值
    double xlb;     // x 下界
    double xlo;     // x 的低值
    double xsave;   // 保存的 x
    // 声明并定义一个双精度浮点型变量 xub，用于存储某个数值
    double xub;
    
    // 声明并定义一个双精度浮点型变量 yy，用于存储某个数值
    double yy;
    
    // 声明并定义一个双精度浮点型变量 zx，用于存储某个数值
    double zx;
    
    // 声明并定义一个双精度浮点型变量 zy，用于存储某个数值
    double zy;
    
    // 声明并定义一个双精度浮点型变量 zz，用于存储某个数值
    double zz;
    
    // 声明并定义一个整型变量 next_state，用于存储某个状态值
    int next_state;
    
    // 声明并定义一个整型变量 qbdd，用于存储某个状态值
    int qbdd;
    
    // 声明并定义一个整型变量 qcond，用于存储某个条件值
    int qcond;
    
    // 声明并定义一个整型变量 qdum1，用于存储某个占位值
    int qdum1;
    
    // 声明并定义一个整型变量 qdum2，用于存储某个占位值
    int qdum2;
    
    // 声明并定义一个整型变量 qhi，用于存储某个高值
    int qhi;
    
    // 声明并定义一个整型变量 qleft，用于存储某个左值
    int qleft;
    
    // 声明并定义一个整型变量 qincr，用于存储某个增量值
    int qincr;
    
    // 声明并定义一个整型变量 qlim，用于存储某个限制值
    int qlim;
    
    // 声明并定义一个整型变量 qok，用于存储某个确定值
    int qok;
    
    // 声明并定义一个整型变量 qup，用于存储某个上值
    int qup;
};

// 结构体定义，包含了一系列双精度浮点数和整数成员，用于表示状态和参数
struct DzrorState
{
    double a;        // 区间的左端点
    double atol;     // 绝对误差容限
    double b;        // 区间的右端点
    double c;        // 计算过程中的临时变量
    double d;        // 计算过程中的临时变量
    double fa;       // 左端点处的函数值
    double fb;       // 右端点处的函数值
    double fc;       // 中点处的函数值
    double fd;       // 计算过程中的临时变量
    double fda;      // 计算过程中的临时变量
    double fdb;      // 计算过程中的临时变量
    double fx;       // 当前点处的函数值
    double m;        // 区间的中点
    double mb;       // 计算过程中的临时变量
    double p;        // 计算过程中的临时变量
    double q;        // 计算过程中的临时变量
    double tol;      // 容限
    double rtol;     // 相对误差容限
    double w;        // 计算过程中的临时变量
    double xhi;      // 区间的上限
    double xlo;      // 区间的下限
    double x;        // 当前点
    int ext;         // 扩展标志
    int status;      // 状态标志
    int next_state;  // 下一个状态
    int first;       // 初始标志
    int qrzero;      // 零标志
    int qleft;       // 左标志
    int qhi;         // 高标志
};

// 包含三个常量，分别代表双精度浮点数的机器精度、最小正数和最大数
static const double spmpar[3] = {
    2.220446049250313e-16,     // np.finfo(np.float64).eps
    2.2250738585072014e-308,   // np.finfo(np.float64).tiny
    1.7976931348623157e+308,   // np.finfo(np.float64).max
};

// 下面是一系列函数的声明，每个函数都用于数学计算或统计分布函数
// 函数的具体实现在其他地方定义，这里只进行声明

static double algdiv(double, double);
static double alngam(double);
static double alnrel(double);
static double apser(double, double, double, double);
static double basym(double, double, double, double);
static double bcorr(double, double);
static double betaln(double, double);
static double bfrac(double, double, double, double, double, double);
static struct TupleDI bgrat(double, double, double, double, double, double);
static double bpser(double, double, double, double);
static struct TupleDDI bratio(double, double, double, double);
static double brcmp1(int, double, double, double, double);
static double brcomp(double, double, double, double);
static double bup(double, double, double, double, int, double);
static struct TupleDD cumbet(double, double, double, double);
static struct TupleDD cumbin(double, double, double, double);
static struct TupleDD cumchi(double, double);
static struct TupleDD cumchn(double, double, double);
static struct TupleDD cumf(double, double, double);
static struct TupleDDI cumfnc(double, double, double, double);
static struct TupleDD cumgam(double, double);
static struct TupleDD cumnbn(double, double, double, double);
static struct TupleDD cumnor(double);
static struct TupleDD cumpoi(double, double);
static struct TupleDD cumt(double, double);
static struct TupleDD cumtnc(double, double, double);
static double devlpl(double *, int, double);
static double dinvnr(double, double);
static void dinvr(struct DinvrState *, struct DzrorState *);
static double dt1(double, double, double);
static void dzror(struct DzrorState *);
static double cdflib_erf(double);
static double erfc1(int, double);
static double esum(int, double);
static double fpser(double, double, double, double);
static double gam1(double);
static struct TupleDI gaminv(double, double, double, double);
static double gaminv_helper_30(double, double, double, double);
static double gamln(double);
static double gamln1(double);
static double cdflib_gamma(double);
static struct TupleDD grat1(double, double, double, double);
static struct TupleDD gratio(double, double, int);
static double gsumln(double, double);
static double psi(double);
static double rcomp(double, double);
static double rexp(double);
static double rlog(double);
static double rlog1(double);
static double stvaln(double);

// 计算 ln(gamma(b)/gamma(a+b)) 当 b >= 8
//
// 在这个算法中，del(x) 是由以下公式定义的函数：
// Ln(gamma(x)) = (x - 0.5)*ln(x) - x + 0.5*ln(2*pi) + del(x)。
double algdiv(double a, double b)
{
    double c, d, h, s11, s3, s5, s7, s9, t, u, v, w, x, x2;
    double carr[6] = {0.833333333333333e-01, -0.277777777760991e-02,
                      0.793650666825390e-03, -0.595202931351870e-03,
                      0.837308034031215e-03, -0.165322962780713e-02};

    // 如果 a > b，则进行以下计算
    if (a > b) {
        h = b / a;
        c = 1./(1. + h);
        x = h/(1. + h);
        d = a + (b - 0.5);
    } else { // 如果 a <= b，则进行以下计算
        h = a / b;
        c = h/(1. + h);
        x = 1./(1. + h);
        d = b + (a - 0.5);
    }
    
    // 设置 sn = (1 - x**n)/(1 - x)
    x2 = x*x;
    s3 = 1. + (x + x2);
    s5 = 1. + (x + x2*s3);
    s7 = 1. + (x + x2*s5);
    s9 = 1. + (x + x2*s7);
    s11 = 1. + (x + x2*s9);

    // 设置 w = del(b) - del(a + b)
    t = pow((1. / b), 2);
    w = (((((carr[5]*s11
            )*t + carr[4]*s9
           )*t + carr[3]*s7
          )*t + carr[2]*s5
         )*t + carr[1]*s3
        )*t + carr[0];
    w *= c / b;

    // 合并结果
    u = d * alnrel(a / b);
    v = a * (log(b) - 1.);
    return (u > v ? (w - v) - u : (w - u) - v);
}

// 返回 gamma 函数的自然对数的双精度值
double alngam(double x)
{
    double prod, xx, result, offset;
    int i, n;
    double scoefn[9] = {0.62003838007127258804e2, 0.36036772530024836321e2,
                        0.20782472531792126786e2, 0.6338067999387272343e1,
                        0.215994312846059073e1, 0.3980671310203570498e0,
                        0.1093115956710439502e0, 0.92381945590275995e-2,
                        0.29737866448101651e-2};
    double scoefd[4] = {0.62003838007126989331e2, 0.9822521104713994894e1,
                        -0.8906016659497461257e1, 0.1000000000000000000e1};
    double coef[5] = {0.83333333333333023564e-1, -0.27777777768818808e-2,
                      0.79365006754279e-3, -0.594997310889e-3, 0.8065880899e-3};

    // 如果 x <= 6.0，则使用递归将 x 减少到 3 以下，
    // 然后应用 hart 等人的有理逼近公式 5236
    // 如果 x > 6.0，则使用递归将 x 增加到至少 12，
    // 然后应用相同来源的公式 5423
    //
    // X 是双精度值，表示要返回其缩放对数 gamma 的值
}
    // 如果 x 小于等于 6.0，则计算 gamma 函数的对数值
    if (x <= 6.0) {
        // 初始化乘积为 1.0，并将 xx 设为 x
        prod = 1.0;
        xx = x;

        // 如果 x 大于 3.0，则循环直到 xx 小于等于 3.0，累乘计算阶乘
        if (x > 3.0) {
            while (xx > 3.0) {
                xx -= 1.0;
                prod *= xx;
            }
        }

        // 如果 x 小于 2.0，则循环直到 xx 大于等于 2.0，计算逆阶乘
        if (x < 2.0) {
            while (xx < 2.0) {
                prod /= xx;
                xx += 1.0;
            }
        }

        // 计算分子与分母多项式的比值，用于逼近 gamma(x)
        result = devlpl(scoefn, 9, xx - 2.) / devlpl(scoefd, 4, xx - 2.);
        // 计算 gamma(x) 的有理逼近值的自然对数
        return log(result * prod);
    }

    // 计算 offset 作为 log(2*PI)/2
    offset = 0.5*log(2.*PI);

    // 如果 x 小于等于 12.0，则调整 x 至至少为 12，并修正 offset
    if (x <= 12.0) {
        // 计算需要调整的整数部分 n
        n = (int)(12. - x);
        if (n > 0) {
            // 如果 n 大于 0，则计算从 x 到 x+n 的连乘积，并调整 offset
            prod = 1.0;
            for (i = 0; i < n; i++) {
                prod *= x + i;
            }
            offset -= log(prod);
            xx = x + n;
        } else {
            // 如果 n 不大于 0，则保持 xx 为 x
            xx = x;
        }
    } else {
        // 如果 x 大于 12.0，则保持 xx 为 x
        xx = x;
    }

    // 计算幂级数的结果
    result = devlpl(coef, 5, pow((1./xx), 2)) / xx;
    result += offset + (xx - 0.5)*log(xx) - xx;
    return result;
double basym(double a, double b, double lmbda, double eps)
{
    //    Asymptotic expansion for ix(a,b) for large a and b.
    //    Lambda = (a + b)*y - b  and eps is the tolerance used.
    //    It is assumed that lambda is nonnegative and that
    //    a and b are greater than or equal to 15.

    double a0[21] = { 0.0 };  // 初步设定为0的数组a0，大小为21
    double b0[21] = { 0.0 };  // 初步设定为0的数组b0，大小为21
    double c[21] = { 0.0 };   // 初步设定为0的数组c，大小为21
    double d[21] = { 0.0 };   // 初步设定为0的数组d，大小为21
    double bsum, dsum, f, h, h2, hn, j0, j1, r, r0, r1, s, ssum;
    double t, t0, t1, u, w, w0, z, z0, z2, zn, znm1;
    double e0 = 2. / sqrt(PI);  // e0为2除以π的平方根的常量
    double e1 = pow(2.0, (-3./2.));  // e1为2的-3/2次方
    int i, imj, j, m, mmj, n, num;

    // ****** Num is the maximum value that n can take in the do loop
    //        ending at statement 50. It is required that num be even.
    //        The arrays a0, b0, c, d have dimension num + 1.
    num = 20;  // num设定为20，是n在do循环中的最大值

    if (a < b) {
        h = a / b;  // h为a除以b的比率
        r0 = 1./(1.+h);  // r0为1除以1加h的比率
        r1 = (b - a) / b;  // r1为b减去a除以b的比率
        w0 = 1. / sqrt(a * (1. + h));  // w0为1除以a乘以1加h的平方根
    } else {
        h = b / a;  // 否则，h为b除以a的比率
        r0 = 1./(1.+h);  // r0为1除以1加h的比率
        r1 = (b - a) / a;  // r1为b减去a除以a的比率
        w0 = 1. / sqrt(b * (1. + h));  // w0为1除以b乘以1加h的平方根
    }
    f = a*rlog1(-lmbda/a) + b*rlog1(lmbda/b);  // f为a乘以rlog1(-lmbda除以a)加上b乘以rlog1(lmbda除以b)
    t = exp(-f);  // t为e的-f次方
    if (t == 0.0) { return 0.0; }  // 如果t为0，则返回0.0
    z0 = sqrt(f);  // z0为f的平方根
    z = 0.5*(z0/e1);  // z为z0的一半除以e1
    z2 = f + f;  // z2为f加上f的和

    a0[0] = (2./3.)*r1;  // a0的第一个元素为2除以3乘以r1
    c[0] = -0.5*a0[0];  // c的第一个元素为-0.5乘以a0的第一个元素
    d[0] = -c[0];  // d的第一个元素为-c的第一个元素
    j0 = (0.5/e0)*erfc1(1, z0);  // j0为0.5除以e0乘以erfc1函数的返回值，参数为1和z0
    j1 = e1;  // j1为e1
    ssum = j0 + d[0]*w0*j1;  // ssum为j0加上d的第一个元素乘以w0乘以j1的和

    s = 1.0;  // s设定为1.0
    h2 = h*h;  // h2为h的平方
    hn = 1.0;  // hn设定为1.0
    w = w0;  // w设定为w0
    znm1 = z;  // znm1设定为z
    zn = z2;  // zn设定为z2
}
    # 循环计算偶数索引 n 对应的值，n 从 2 开始到 num 结束，每次增加 2
    for (n = 2; n <= num; n += 2) {
        # hn 乘以 h2，更新 hn 的值
        hn *= h2;
        # 计算 a0[n-1] 的值，存入 a0 数组中
        a0[n-1] = 2. * r0 * (1. + h * hn) / (n + 2.);
        # 将 hn 加到 s 上，更新 s 的值
        s += hn;
        # 计算 a0[n] 的值，存入 a0 数组中
        a0[n] = 2. * r1 * s / (n + 3.);

        # 内层循环，计算 b0 数组的值
        for (i = n; i <= n + 1; i++) {
            # 计算 r 的值
            r = -0.5 * (i + 1.);
            # 计算 b0[0] 的值，存入 b0 数组中
            b0[0] = r * a0[0];

            # 第二层循环，计算 b0 数组的其他值
            for (m = 2; m <= i; m++) {
                # 初始化 bsum
                bsum = 0.0;
                # 第三层循环，计算 bsum 的值
                for (j = 1; j < m; j++) {
                    # 计算 mmj
                    mmj = m - j;
                    # 更新 bsum 的值
                    bsum += (j * r - mmj) * a0[j - 1] * b0[mmj - 1];
                }
                # 计算 b0[m-1] 的值，存入 b0 数组中
                b0[m - 1] = r * a0[m - 1] + bsum / m;
            }
            # 计算 c[i-1] 的值，存入 c 数组中
            c[i - 1] = b0[i - 1] / (i + 1.);
            # 初始化 dsum
            dsum = 0.0;

            # 第四层循环，计算 dsum 的值
            for (j = 1; j < i; j++) {
                # 计算 imj
                imj = i - j;
                # 更新 dsum 的值
                dsum += d[imj - 1] * c[j - 1];
            }
            # 计算 d[i-1] 的值，存入 d 数组中
            d[i - 1] = -(dsum + c[i - 1]);
        }
        # 更新 j0 和 j1 的值
        j0 = e1 * znm1 + (n - 1.) * j0;
        j1 = e1 * zn + n * j1;
        # 更新 znm1 和 zn 的值
        znm1 *= z2;
        zn *= z2;
        # 更新 w 的值
        w *= w0;
        # 计算 t0 和 t1 的值
        t0 = d[n - 1] * w * j0;
        w *= w0;
        t1 = d[n] * w * j1;
        # 更新 ssum 的值
        ssum += t0 + t1;
        # 检查是否满足终止条件，如果满足则跳出循环
        if ((fabs(t0) + fabs(t1)) <= eps * ssum) { break; }
    }
    # 计算 u 的值
    u = exp(-bcorr(a, b));
    # 返回结果 e0*t*u*ssum
    return e0 * t * u * ssum;
}

// 定义一个函数 bcorr，计算 beta 分布中修正系数的值
double bcorr(double a0, double b0)
{
    // Evaluation of del(a0) + del(b0) - del(a0 + b0) where
    // ln(gamma(a)) = (a - 0.5)*ln(a) - a + 0.5*ln(2*pi) + del(a).
    // It is assumed that a0 >= 8 And b0 >= 8.

    // 声明变量
    double a, b, c, h, s11, s3, s5, s7, s9, t, w, x, x2;
    double carr[6] = {0.833333333333333e-01, -0.277777777760991e-02,
                      0.793650666825390e-03, -0.595202931351870e-03,
                      0.837308034031215e-03, -0.165322962780713e-02};

    // 选取 a 和 b 的较小值和较大值
    a = fmin(a0, b0);
    b = fmax(a0, b0);
    // 计算 h、c 和 x
    h = a / b;
    c = h / (1. + h);
    x = 1. / (1. + h);
    x2 = x * x;
    // 设置 sn = (1 - x**n)/(1 - x)
    s3 = 1. + (x + x2);
    s5 = 1. + (x + x2 * s3);
    s7 = 1. + (x + x2 * s5);
    s9 = 1. + (x + x2 * s7);
    s11 = 1. + (x + x2 * s9);
    // 设置 w = del(b) - del(a + b)
    t = pow((1. / b), 2);
    w = (((((carr[5] * s11
            )*t + carr[4] * s9
           )*t + carr[3] * s7
          )*t + carr[2] * s5
         )*t + carr[1] * s3
        )*t + carr[0];
    w *= c / b;
    // 计算 del(a) + w
    t = pow((1. / a), 2);
    return ((((((carr[5]) * t + carr[4]
               )*t + carr[3]
              )*t + carr[2]
             )*t + carr[1]
            )*t + carr[0]
           ) / a + w;
}

// 定义一个函数 betaln，计算 beta 函数的对数值
double betaln(double a0, double b0)
{
    // Evaluation of the logarithm of the beta function

    // 声明变量
    double a, b, c, h, u, v, w, z;
    double e = 0.918938533204673;
    int i, n;

    // 选取 a 和 b 的较小值和较大值
    a = fmin(a0, b0);
    b = fmax(a0, b0);

    // 如果 a >= 8.0，使用 bcorr 计算修正系数 w
    if (a >= 8.0) {
        w = bcorr(a, b);
        h = a / b;
        c = h / (1. + h);
        u = -(a - 0.5) * log(c);
        v = b * alnrel(h);
        if (u > v) {
            return (((-0.5 * log(b) + e) + w) - v) - u;
        } else {
            return (((-0.5 * log(b) + e) + w) - u) - v;
        }
    }

    // 如果 a < 1，根据 b 的大小分情况计算
    if (a < 1) {
        if (b > 8) {
            return gamln(a) + algdiv(a, b);
        } else {
            return gamln(a) + (gamln(b) - gamln(a + b));
        }
    }

    // 如果 a <= 2，根据 b 的大小分情况计算
    if (a <= 2) {
        if (b <= 2) {
            return gamln(a) + gamln(b) - gsumln(a, b);
        }
        if (b >= 8) {
            return gamln(a) + algdiv(a, b);
        }
        w = 0.;
    }

    // 如果 a > 2，根据 b 的大小分情况计算
    if (a > 2) {
        if (b <= 1000) {
            n = (int)(a - 1.);
            w = 1.;
            for (i = 0; i < n; i++) {
                a -= 1.0;
                h = a / b;
                w *= h / (1. + h);
            }
            w = log(w);
            if (b >= 8.0) {
                return w + gamln(a) + algdiv(a, b);
            }
        } else {
            n = (int)(a - 1.);
            w = 1.0;
            for (i = 0; i < n; i++) {
                a -= 1.0;
                w *= a / (1. + (a / b));
            }
            return (log(w) - n * log(b)) + (gamln(a) + algdiv(a, b));
        }
    }

    // 计算 b - 1 的整数部分
    n = (int)(b - 1.);
    z = 1.0;
    // 计算 beta 函数对数的剩余部分
    for (i = 0; i < n; i++) {
        b -= 1.0;
        z *= b / (a + b);
    }
    return w + log(z) + (gamln(a) + gamln(b) - gsumln(a, b));
}
    // 定义需要用到的变量
    double alpha, beta, e, r0, t, w, result;
    // 计算常数 c, c0, c1
    double c = 1. + lambda;
    double c0 = b / a;
    double c1 = 1. + (1. / a);
    // 计算 yp1, n, p, s, an, bn, anp1, bnp1, r 的初始值
    double yp1 = y + 1.;
    double n = 0.;
    double p = 1.;
    double s = a + 1.;
    double an = 0.;
    double bn = 1.;
    double anp1 = 1.;
    double bnp1 = c / c1;
    double r = c1 / c;
    
    // 调用 brcomp 函数计算 brcomp(a, b, x, y) 的结果并存入 result
    result = brcomp(a, b, x, y);
    
    // 如果 result 为 0，直接返回 0
    if (result == 0.0) { return 0; }
    
    // 开始计算连分数展开
    while (1) {
        // 更新 n
        n += 1.0;
        // 计算 t, w, e, alpha, beta
        t = n / a;
        w = n * (b - n) * x;
        e = a / s;
        alpha = (p * (p + c0) * e * e) * (w * x);
        e = (1. + t) / (c1 + t + t);
        beta = n + (w / s) + e * (c + n * yp1);
        // 更新 p, s
        p = 1. + t;
        s += 2.;
        // 更新 an, bn, anp1, bnp1
        t = alpha * an + beta * anp1;
        an = anp1;
        anp1 = t;
        t = alpha * bn + beta * bnp1;
        bn = bnp1;
        bnp1 = t;
        // 更新 r0, r
        r0 = r;
        r = anp1 / bnp1;
        // 检查是否满足精度条件，如果满足则结束循环
        if (!(fabs(r - r0) > eps * r)) { break; }
        // 重新缩放 an, bn, anp1, bnp1
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;
    }
    
    // 返回结果 result 乘以最终的 r
    return result * r;
}


struct TupleDI bgrat(double a, double b, double x , double y, double w, double eps)
{
    // Asymptotic expansion for ix(a,b) when a is larger than b.
    // The result of the expansion is added to w. It is assumed
    // that a >= 15 And b <= 1.  Eps is the tolerance used.
    // Ierr is a variable that reports the status of the results.

    double bp2n, cn, coef, dj, j, l, n2, q, r, s, ssum, t, t2, u, v;
    double c[30] = { 0.0 };
    double d[30] = { 0.0 };
    double bm1 = (b - 0.5) - 0.5;
    double nu = a + bm1*0.5;
    double lnx = (y > 0.375 ? log(x) : alnrel(-y));
    double z = -nu*lnx;
    int i, n;

    if ((b*z) == 0.0) { return (struct TupleDI){.d1 = w, .i1 = 1}; }

    // COMPUTATION OF THE EXPANSION
    // SET R = EXP(-Z)*Z**B/GAMMA(B)
    r = b * (1. + gam1(b)) * exp(b*log(z));
    r *= exp(a*lnx) * exp(0.5*bm1*lnx);
    u = algdiv(b, a) + b*log(nu);
    u = r*exp(-u);
    if (u == 0.0) { return (struct TupleDI){.d1 = w, .i1 = 1}; }

    struct TupleDD ret = grat1(b, z, r, eps);
    q = ret.d2;
    v = 0.25 * pow((1 / nu), 2);
    t2 = 0.25*lnx*lnx;
    l = w / u;
    j = q / r;
    ssum = j;
    t = 1.;
    cn = 1.;
    n2 = 0.;

    for (n = 1; n <= 30; n++) {
        bp2n = b + n2;
        j = (bp2n*(bp2n + 1.)*j + (z + bp2n + 1.)*t)*v;
        n2 += 2.;
        t *= t2;
        cn *= 1/(n2*(n2 + 1.));
        c[n-1] = cn;
        s = 0.0;
        if (n > 1) {
            coef = b - n;
            for (i = 1; i < n; i++) {
                s += coef*c[i-1]*d[n-i-1];
                coef += b;
            }
        }
        d[n-1] = bm1*cn + s/n;
        dj = d[n-1]*j;
        ssum += dj;
        if (ssum <= 0.) { return (struct TupleDI){.d1 = w, .i1 = 1}; }
        if (!(fabs(dj) > eps*(ssum+l))) { break; }
    }
    return (struct TupleDI){.d1 = w + u*ssum, .i1 = 0};
}


double bpser(double a, double b, double x, double eps)
{
    // Power series expansion for evaluating ix(a,b) when b <= 1
    // Or b*x <= 0.7.  Eps is the tolerance used.

    double a0, apb, b0, c, n, ssum, t, tol, u, w, z;
    int i, m;
    double result = 0.0;

    if (x == 0.0) { return 0.0; }

    // Compute the factor x**a/(a*beta(a,b))
    a0 = fmin(a, b);
    # 如果参数 a0 小于 1.0，则进行以下计算
    if (a0 < 1.0) {
        # 计算 a 和 b 的较大值，并赋给 b0
        b0 = fmax(a, b);
        # 如果 b0 小于等于 1.0，则执行以下计算
        if (b0 <= 1.0) {
            # 计算 x 的 a 次方，并赋给 result
            result = pow(x, a);
            # 如果 result 等于 0.0，则返回 0.0
            if (result == 0.0) { return 0.0; }

            # 计算 a + b，并赋给 apb
            apb = a + b;
            # 如果 apb 大于 1.0，则执行以下计算
            if (apb > 1.0) {
                # 计算 u 和 z 的值
                u = a + b - 1.0;
                z = (1. + gam1(u)) / apb;
            } else {
                # 否则，直接计算 z 的值
                z = 1. + gam1(apb);
            }
            # 计算 c 的值
            c = (1. + gam1(a)) * (1. + gam1(b)) / z;
            # 更新 result 的值
            result *= c * (b / apb);

        } else if (b0 < 8) {
            # 计算 u 的值
            u = gamln1(a0);
            # 将 b0 - 1.0 转换为整数，并赋给 m
            m = (int)(b0 - 1.0);
            # 如果 m 大于 0，则执行以下循环
            if (m > 0) {
                c = 1.0;
                # 循环计算 c 的值
                for (i = 0; i < m; i++) {
                    b0 -= 1.0;
                    c *= (b0 / (a0 + b0));
                }
                # 更新 u 的值
                u += log(c);
            }
            # 计算 z 的值
            z = a*log(x) - u;
            # 更新 b0 的值
            b0 -= 1.0;
            # 计算 apb 的值
            apb = a0 + b0;
            # 如果 apb 大于 1.0，则执行以下计算
            if (apb > 1.0) {
                u = a0 + b0 - 1.0;
                t = (1. + gam1(u)) / apb;
            } else {
                t = 1. + gam1(apb);
            }
            # 更新 result 的值
            result = exp(z) * (a0 / a) * (1. + gam1(b0)) / t;

        } else if (b0 >= 8) {
            # 计算 u 的值
            u = gamln1(a0) + algdiv(a0, b0);
            # 计算 z 的值
            z = a*log(x) - u;
            # 更新 result 的值
            result = (a0 / a) * exp(z);
        }
    } else {
        # 否则，计算 z 的值
        z = a*log(x) - betaln(a, b);
        # 计算 result 的值
        result = exp(z) / a;
    }
    # 如果 result 为 0. 或者 a 小于等于 0.1*eps，则返回 result
    if ((result == 0.) || (a <= 0.1*eps)) { return result; }
    
    # 计算系列求和的初始值
    ssum = 0.0;
    n = 0;
    c = 1.0;
    # 计算容差的值
    tol = eps / a;
    # 进行循环，计算系列求和的值
    while (1) {
        n += 1.0;
        c *= (0.5 + (0.5 - b/n))*x;
        w = c / (a + n);
        ssum += w;
        # 如果 w 的绝对值不大于 tol，则退出循环
        if (!(fabs(w) > tol)) { break; }
    }
    # 返回最终的结果值
    return result * (1. + a*ssum);
struct TupleDDI bratio(double a, double b, double x, double y)
{
    // Evaluation of the incomplete beta function Ix(a,b)
    //
    // It is assumed that a and b are nonnegative, and that x <= 1
    // And y = 1 - x. Bratio assigns w and w1 the values
    //
    // w  = Ix(a,b)
    // w1 = 1 - Ix(a,b)
    //
    // Ierr is a variable that reports the status of the results.
    // If no input errors are detected then ierr is set to 0 and
    // w and w1 are computed. Otherwise, if an error is detected,
    // then w and w1 are assigned the value 0 and ierr is set to
    // one of the following values ...
    //
    // Ierr = 1  if a or b is negative
    // ierr = 2  if a = b = 0
    // ierr = 3  if x < 0 Or x > 1
    // Ierr = 4  if y < 0 Or y > 1
    // Ierr = 5  if x + y .ne. 1
    // Ierr = 6  if x = a = 0
    // ierr = 7  if y = b = 0

    double a0, b0, lmbda, x0, y0;
    int ind, n;
    double w = 0.;
    double w1 = 0.;
    double eps = fmax(spmpar[0], 1e-15);
    struct TupleDI bgratret;

    // Check for specific error conditions and return appropriate values
    if ((a < 0.) || (b < 0.)) { return (struct TupleDDI){.d1 = w, .d2 = w1, .i1 = 1}; }
    else if ((a == 0.) && (b == 0.)) { return (struct TupleDDI){.d1 = w, .d2 = w1, .i1 = 2}; }
    else if ((x < 0.) || (x > 1.)) { return (struct TupleDDI){.d1 = w, .d2 = w1, .i1 = 3}; }
    else if ((y < 0.) || (y > 1.)) { return (struct TupleDDI){.d1 = w, .d2 = w1, .i1 = 4}; }
    else if (fabs(((x+y)-0.5) - 0.5) > 3.*eps) { return (struct TupleDDI){.d1 = w, .d2 = w1, .i1 = 5}; }
    else if (x == 0.) {
        if (a == 0) {
            return (struct TupleDDI){.d1 = w, .d2 = w1, .i1 = 6};
        } else {
            return (struct TupleDDI){.d1 = 0.0, .d2 = 1.0, .i1 = 0};
        }
    } else if (y == 0.) {
        if (b == 0) {
            return (struct TupleDDI){.d1 = w, .d2 = w1, .i1 = 7};
        } else {
            return (struct TupleDDI){.d1 = 1.0, .d2 = 0.0, .i1 = 0};
        }
    } else if (a == 0.) { return (struct TupleDDI){.d1 = 1.0, .d2 = 0.0, .i1 = 0}; }
    else if (b == 0.) { return (struct TupleDDI){.d1 = 0.0, .d2 = 1.0, .i1 = 0}; }
    else if (fmax(a, b) < 1e-3*eps) { return (struct TupleDDI){.d1 = b/(a+b), .d2 = a/(a+b), .i1 = 0}; }

    // Initialize variables for iteration
    ind = 0;
    a0 = a;
    b0 = b;
    x0 = x;
    y0 = y;
    } else {
        // 计算 lambda 值
        lmbda = (a > b ? ((a + b)*y - b) : (a - (a + b)*x));
        // 如果 lambda 小于 0，调整参数和值
        if (lmbda < 0.0) {
            ind = 1;
            a0 = b;
            b0 = a;
            x0 = y;
            y0 = x;
            lmbda = fabs(lmbda);
        }
        // 如果 b0 小于 40，并且满足条件，则调用 bpser 函数
        if ((b0 < 40.) && (b0*x0 <= 0.7)) {
            w = bpser(a0, b0, x0, eps);
            w1 = 0.5 + (0.5 - w);
            // 返回结果结构体 TupleDDI
            return (struct TupleDDI){.d1 = (ind == 0? w : w1), .d2 = (ind == 0? w1 : w), .i1 = 0};
        }
        // 如果 b0 小于 40，则处理 b0 的整数部分和小数部分
        if (b0 < 40.) {
            n = (int)b0;
            b0 -= n;
            if (b0 == 0.) {
                n -= 1;
                b0 = 1.0;
            }
            // 调用 bup 函数计算 w
            w = bup(b0, a0, y0, x0, n, eps);
            // 如果 x0 小于等于 0.7，再调用 bpser 函数
            if (x0 <= 0.7) {
                w += bpser(a0, b0, x0, eps);
                w1 = 0.5 + (0.5 - w);
                // 返回结果结构体 TupleDDI
                return (struct TupleDDI){.d1 = (ind == 0? w : w1), .d2 = (ind == 0? w1 : w), .i1 = 0};
            } else {
                // 如果 a0 小于等于 15，增加 n 并调用 bup 函数
                if (a0 <= 15.) {
                    n = 20;
                    w += bup(a0, b0, x0, y0, n, eps);
                    a0 += n;
                }
                // 调用 bgrat 函数计算结果
                bgratret = bgrat(a0, b0, x0, y0, w, 15.*eps);
                w = bgratret.d1;
                w1 = 0.5 + (0.5 - w);
                // 返回结果结构体 TupleDDI
                return (struct TupleDDI){.d1 = (ind == 0? w : w1), .d2 = (ind == 0? w1 : w), .i1 = 0};
            }
        }
        // 如果 a0 大于 b0，根据条件选择调用 bfrac 函数或 basym 函数
        if (a0 > b0) {
            if ((b0 <= 100) || (lmbda > 0.03*b0)) {
                w = bfrac(a0, b0, x0, y0, lmbda, 15.0*eps);
                w1 = 0.5 + (0.5 - w);
                // 返回结果结构体 TupleDDI
                return (struct TupleDDI){.d1 = (ind == 0? w : w1), .d2 = (ind == 0? w1 : w), .i1 = 0};
            } else {
                // 否则调用 basym 函数
                w = basym(a0, b0, lmbda, 100.*eps);
                w1 = 0.5 + (0.5 - w);
                // 返回结果结构体 TupleDDI
                return (struct TupleDDI){.d1 = (ind == 0? w : w1), .d2 = (ind == 0? w1 : w), .i1 = 0};
            }
        }
        // 处理特殊情况，调用 bfrac 函数或 basym 函数
        if ((a0 <= 100.) || (lmbda > 0.03*a0)) {
            w = bfrac(a0, b0, x0, y0, lmbda, 15.0*eps);
            w1 = 0.5 + (0.5 - w);
            // 返回结果结构体 TupleDDI
            return (struct TupleDDI){.d1 = (ind == 0? w : w1), .d2 = (ind == 0? w1 : w), .i1 = 0};
        }
        // 否则调用 basym 函数
        w = basym(a0, b0, lmbda, 100.*eps);
        w1 = 0.5 + (0.5 - w);
        // 返回结果结构体 TupleDDI
        return (struct TupleDDI){.d1 = (ind == 0? w : w1), .d2 = (ind == 0? w1 : w), .i1 = 0};
    }
    //    Evaluation of x**a*y**b/beta(a,b)

    // 初始化变量
    double a0, apb, b0, c, e, h, lmbda, lnx, lny, t, u, v, x0, y0, z;
    // 定义常数
    double const r2pi = 1. / sqrt(2 * PI);
    // 循环计数器
    int i, n;

    // 检查是否有一个输入为零，若有则直接返回0
    if ((x == 0.) || (y == 0.)) {
        return 0.;
    }
    // 计算 a0 为 a 和 b 中的较小值
    a0 = fmin(a, b);
    # 如果 a0 大于等于 8.0
    if (a0 >= 8.) {
        # 如果 a 大于 b
        if (a > b) {
            # 计算 h 为 b 除以 a 的结果
            h = b / a;
            # 计算 x0 和 y0
            x0 = 1. / (1. + h);
            y0 = h / (1. + h);
            # 计算 lmbda
            lmbda = (a + b)*y - b;
        } else {
            # 计算 h 为 a 除以 b 的结果
            h = a / b;
            # 计算 x0 和 y0
            x0 = h / (1. + h);
            y0 = 1. / (1. + h);
            # 计算 lmbda
            lmbda = a - (a + b)*x;
        }
        # 计算 e
        e = -lmbda / a;
        # 如果 fabs(e) 大于 0.6
        if (fabs(e) > 0.6) {
            # 计算 u
            u = e - log(x / x0);
        } else {
            # 调用 rlog1(e) 计算 u
            u = rlog1(e);
        }
        # 计算 e
        e = lmbda / b;
        # 如果 fabs(e) 大于 0.6
        if (fabs(e) > 0.6) {
            # 计算 v
            v = e - log(y / y0);
        } else {
            # 调用 rlog1(e) 计算 v
            v = rlog1(e);
        }

        # 计算 z
        z = exp(-(a*u + b*v));
        # 返回结果
        return r2pi*sqrt(b*x0)*z*exp(-bcorr(a, b));
    }
    # 如果 x 小于等于 0.375
    if (x <= 0.375) {
        # 计算 lnx 和 lny
        lnx = log(x);
        lny = alnrel(-x);
    } else {
        # 如果 y 大于 0.375
        lnx = (y > 0.375 ? log(x) : alnrel(-y));
        lny = log(y);
    }
    # 计算 z
    z = a*lnx + b*lny;
    # 如果 a0 大于等于 1.0
    if (a0 >= 1.0) {
        # 减去 betaln(a, b) 的结果，并返回 exp(z)
        z -= betaln(a, b);
        return exp(z);
    }

    # 计算 b0
    b0 = fmax(a, b);
    # 如果 b0 大于等于 8.0
    if (b0 >= 8.0) {
        # 计算 u
        u = gamln1(a0) + algdiv(a0, b0);
        return a0*exp(z - u);
    }

    # 如果 b0 大于 1.0
    if (b0 > 1.0) {
        # 计算 u
        u = gamln1(a0);
        # 计算 n
        n = (int)(b0 - 1.);
        # 如果 n 大于等于 1
        if (n >= 1) {
            # 初始化 c
            c = 1.0;
            # 循环计算 c
            for (i = 0; i < n; i++) {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            # 更新 u
            u += log(c);
        }
        # 更新 z
        z -= u;
        b0 -= 1.0;
        apb = a0 + b0;

        # 如果 apb 大于 1.0
        if (apb > 1.0) {
            # 计算 u 和 t
            u = a0 + b0 - 1.0;
            t = (1. + gam1(u)) / apb;
        } else {
            # 计算 t
            t = 1. + gam1(apb);
        }
        # 返回结果
        return a0*exp(z)*(1. + gam1(b0)) / t;
    }
    # 如果 exp(z) 等于 0.
    if (exp(z) == 0.) {
        # 返回 0.0
        return 0.0;
    }

    # 计算 apb 和 t
    apb = a + b;
    t = exp(z);
    # 如果 apb 大于 1.0
    if (apb > 1.0) {
        # 计算 u 和 z
        u = a + b - 1.0;
        z = (1. + gam1(u)) / apb;
    } else {
        # 计算 z
        z = 1. + gam1(apb);
    }
    # 计算 c
    c = (1. + gam1(a)) * (1. + gam1(b)) / z;
    # 返回结果
    return t * (a0*c) / (1. + a0 / b0);
// 双精度函数，计算变量 a, b, x, y, n, eps 对应的函数 bup 的差值
double bup(double a, double b, double x, double y, int n, double eps)
{
    // 计算 a + b 的值
    double apb = a + b;
    // 计算 a + 1 的值
    double ap1 = a + 1.;
    // 初始化变量 d 为 1
    double d = 1.;
    // 声明变量 r, t, w, result 用于存储结果
    double r, t, w, result;
    // 声明变量 i, nm1
    int i, nm1;
    // 初始化 k 和 mu
    int k = 0;
    int mu = 0;

    // 若 n 为正整数并且 a < 1 或者 apb < 1.1*ap1，则执行以下代码块
    // 获取缩放因子 exp(-mu) 和 exp(mu)*(x**a*y**b/beta(a,b))/a
    if (!((n == 1) || (a < 1) || (apb < 1.1*ap1))) {
        // 设置 mu 为 708
        mu = 708;
        // 将 mu 赋值给 t
        t = mu;
        // 计算 exp(-708) 并赋值给 d
        d = exp(-708);
    }

    // 调用函数 brcmp1(mu, a, b, x, y)，计算结果除以 a，并赋值给 result
    result = brcmp1(mu, a, b, x, y) / a;
    // 若 n 为 1 或者 result 为 0，则直接返回 result
    if ((n == 1) || (result == 0.)) {
        return result;
    }
    // 将 n - 1 赋值给 nm1
    nm1 = n - 1;
    // 将 d 赋值给 w
    w = d;

    // 如果 b <= 1，则执行以下代码块
    // 50
    for (i = 0; i < n-1; i++) {
        // 更新 d 的值
        d *= ((apb + i) / (ap1 + i))*x;
        // 更新 w 的值
        w += d;
        // 如果 d <= eps*w，则跳出循环
        if (d <= eps*w) { break; }
    }
    // 返回 result*w 的值
    return result*w;

    // 如果 y > 1e-4，则执行以下代码块
    // 20
    r = (b - 1.)*x/y - a;
    if (r < 1.) {
        // 50
        for (i = 0; i < n-1; i++) {
            // 更新 d 的值
            d *= ((apb + i) / (ap1 + i))*x;
            // 更新 w 的值
            w += d;
            // 如果 d <= eps*w，则跳出循环
            if (d <= eps*w) { break; }
        }
        // 返回 result*w 的值
        return result*w;
    }
    // 将 nm1 赋值给 k
    k = nm1;
    // 将 nm1 赋值给 t
    t = nm1;
    // 若 r < t，则将 r 转换为整数赋值给 k
    if (r < t) {
        k = (int)r;
    }

    // 30
    // 添加级数的递增项
    for (i = 0; i < k; i++) {
        // 更新 d 的值
        d *= ((apb + i) / (ap1 + i))*x;
        // 更新 w 的值
        w += d;
    }

    // 若 k 等于 nm1，则返回 result*w 的值
    if (k == nm1) {
        return result*w;
    }

    // 50
    // 添加级数的剩余项
    for (i = k; i < n-1; i++) {
        // 更新 d 的值
        d *= ((apb + i) / (ap1 + i))*x;
        // 更新 w 的值
        w += d;
        // 如果 d <= eps*w，则跳出循环
        if (d <= eps*w) { break; }
    }
    // 返回 result*w 的值
    return result*w;
}
    // X <--> Upper limit of integration of beta density.
    //        Input range: [0,1].
    //        Search range: [0,1]
    //        DOUBLE PRECISION X

    // Y <--> 1-X.
    //        Input range: [0,1].
    //        Search range: [0,1]
    //        X + Y = 1.0.
    //        DOUBLE PRECISION Y

    // A <--> The first parameter of the beta density.
    //        Input range: (0, +infinity).
    //        Search range: [1D-100,1D100]
    //        DOUBLE PRECISION A

    // B <--> The second parameter of the beta density.
    //        Input range: (0, +infinity).
    //        Search range: [1D-100,1D100]
    //        DOUBLE PRECISION B

    // STATUS <-- 0 if calculation completed correctly
    //           -I if input parameter number I is out of range
    //            1 if answer appears to be lower than lowest
    //              search bound
    //            2 if answer appears to be higher than greatest
    //              search bound
    //            3 if P + Q .ne. 1
    //            4 if X + Y .ne. 1
    //        INTEGER STATUS

    // BOUND <-- Undefined if STATUS is 0
    //           Bound exceeded by parameter number I if STATUS
    //           is negative.
    //           Lower search bound if STATUS is 1.
    //           Upper search bound if STATUS is 2.


    //                              Method


    // Cumulative distribution function  (P)  is calculated directly by
    // code associated with the following reference.

    // DiDinato, A. R. and Morris,  A.   H.  Algorithm 708: Significant
    // Digit Computation of the Incomplete  Beta  Function Ratios.  ACM
    // Trans. Math.  Softw. 18 (1993), 360-373.

    // Computation of other parameters involve a search for a value that
    // produces  the desired  value  of P.   The search relies  on  the
    // monotonicity of P with the other parameter.


    //                              Note


    // The beta density is proportional to
    //           t^(A-1) * (1-t)^(B-1)

    //**********************************************************************
struct TupleDDID cdfbet_which1(double x, double y, double a, double b)
{
    // 声明结构体变量 betret 用于存储函数返回值
    struct TupleDD betret;
    // 声明结构体变量 ret，并初始化其字段 i1 为 0
    struct TupleDDID ret = {0};
    // 检查 x 是否不在 [0, 1] 范围内
    if (!((0 <= x) && (x <= 1.0))) {
        // 如果不在范围内，设置返回结构体的 i1 为 -1，并根据 x 的值设置 d3 字段为 0 或 1
        ret.i1 = -1;
        ret.d3 = (!(x > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 y 是否不在 [0, 1] 范围内
    if (!((0 <= y) && (y <= 1.0))) {
        // 如果不在范围内，设置返回结构体的 i1 为 -2，并根据 y 的值设置 d3 字段为 0 或 1
        ret.i1 = -2;
        ret.d3 = (!(y > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 a 是否不大于 0
    if (!(0 < a)) {
        // 如果不大于 0，返回一个带有特定字段值的结构体
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }
    // 检查 b 是否不大于 0
    if (!(0 < b)) {
        // 如果不大于 0，返回一个带有特定字段值的结构体
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4, .d3 = 0.0};
    }
    // 检查条件 fabs(x+y) - 0.5 - 0.5 是否大于 3*spmpar[0]
    if (((fabs(x+y)-0.5)-0.5) > 3*spmpar[0]) {
        // 如果满足条件，返回一个带有特定字段值的结构体
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = 4,
                                  .d3 = (x+y < 0 ? 0.0 : 1.0)};
    }

    // 调用函数 cumbet 计算并返回结果
    betret = cumbet(x, y, a, b);
    // 返回一个带有特定字段值的结构体
    return (struct TupleDDID){.d1 = betret.d1, .d2 = betret.d2, .i1 = 0, .d3 = 0.0};
}


struct TupleDDID cdfbet_which2(double p, double q, double a, double b)
{
    // 声明变量
    double xx, yy;
    // 设置公差和绝对容差的初始值
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q，结果赋给变量 qporq
    int qporq = (p <= q);
    // 声明状态结构体 DZ，并初始化部分字段
    DzrorState DZ = {0};

    // 设置 DZ 结构体的初始值
    DZ.xlo = 0.0;
    DZ.xhi = 1.0;
    DZ.atol = atol;
    DZ.rtol = tol;
    DZ.x = 0.0;
    DZ.b = 0.0;
    // 声明结构体变量 betret 用于存储函数返回值
    struct TupleDD betret;
    // 声明结构体变量 ret，并初始化其字段 i1 为 0
    struct TupleDDID ret = {0};

    // 检查 p 是否不在 [0, 1] 范围内
    if (!((0 <= p) && (p <= 1.0))) {
        // 如果不在范围内，设置返回结构体的 i1 为 -1，并根据 p 的值设置 d3 字段为 0 或 1
        ret.i1 = -1;
        ret.d3 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 是否不在 [0, 1] 范围内
    if (!((0 <= q) && (q <= 1.0))) {
        // 如果不在范围内，设置返回结构体的 i1 为 -2，并根据 q 的值设置 d3 字段为 0 或 1
        ret.i1 = -2;
        ret.d3 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 a 是否不大于 0
    if (!(0 < a)) {
        // 如果不大于 0，返回一个带有特定字段值的结构体
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }
    // 检查 b 是否不大于 0
    if (!(0 < b)) {
        // 如果不大于 0，返回一个带有特定字段值的结构体
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4, .d3 = 0.0};
    }
    // 检查条件 fabs(p+q) - 0.5 - 0.5 是否大于 3*spmpar[0]
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        // 如果满足条件，返回一个带有特定字段值的结构体
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = 3,
                                  .d3 = (p+q < 0 ? 0.0 : 1.0)};
    }

    // 根据 qporq 的值选择不同的计算路径
    if (qporq) {
        // 初始化 DZ，并循环计算，直到满足条件
        dzror(&DZ);
        yy = 1.0 - DZ.x;
        while (DZ.status == 1) {
            betret = cumbet(DZ.x, yy, a, b);
            DZ.fx = betret.d1 - p;
            dzror(&DZ);
            yy = 1.0 - DZ.x;
        }
        xx = DZ.x;
    } else {
        // 初始化 DZ，并循环计算，直到满足条件
        dzror(&DZ);
        xx = 1.0 - DZ.x;
        while (DZ.status == 1) {
            betret = cumbet(xx, DZ.x, a, b);
            DZ.fx = betret.d2 - q;
            dzror(&DZ);
            xx = 1.0 - DZ.x;
        }
        yy = DZ.x;
    }

    // 检查 DZ 的状态，根据不同状态设置返回结构体的字段值
    if (DZ.status == -1) {
        ret.d1 = xx;
        ret.d2 = yy;
        ret.i1 = (DZ.qleft ? 1 : 2);
        ret.d3 = (DZ.qleft ? 0 : 1);
        return ret;
    } else {
        ret.d1 = xx;
        ret.d2 = yy;
        return ret;
    }
}


struct TupleDID cdfbet_which3(double p, double q, double x, double y, double b)
{
    // 设置公差和绝对容差的初始值
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q，结果赋给变量 qporq
    int qporq = (p <= q);
    // 声明状态结构体 DS 和 DZ，并初始化部分字段
    DinvrState DS
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.0;
    struct TupleDD betret;
    struct TupleDID ret = {0};

    // 检查参数 p 的有效性，确保在 [0, 1] 范围内
    if (!((0 <= p) && (p <= 1.0))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查参数 q 的有效性，确保在 [0, 1] 范围内
    if (!((0 <= q) && (q <= 1.0))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查参数 x 的有效性，确保在 [0, 1] 范围内
    if (!((0 <= x) && (x <= 1.0))) {
        ret.i1 = -3;
        ret.d2 = (!(x > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查参数 y 的有效性，确保在 [0, 1] 范围内
    if (!((0 <= y) && (y <= 1.0))) {
        ret.i1 = -4;
        ret.d2 = (!(y > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查参数 b 是否大于零
    if (!(0 < b)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -5};
    }
    // 检查 p+q 是否超出指定范围
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3, .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }
    // 检查 x+y 是否超出指定范围
    if (((fabs(x+y)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 4, .d2 = (x+y < 0 ? 0.0 : 1.0)};
    }

    // 调用 dinvr 函数处理 DS 和 DZ 结构
    dinvr(&DS, &DZ);
    // 循环执行，直到 DS 的状态不为 1
    while (DS.status == 1) {
        // 调用 cumbet 函数计算 betret 结构
        betret = cumbet(x, y, DS.x, b);
        // 更新 DS.fx 的值
        DS.fx = (qporq ? betret.d1 - p : betret.d2 - q);
        // 再次调用 dinvr 函数
        dinvr(&DS, &DZ);
    }

    // 根据 DS 的最终状态，设置返回结构 ret 的值
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 1e-100 : 1e100);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
    // 设置数值容差
    double tol = 1e-10;
    // 设置绝对容差
    double atol = 1e-50;
    // 确定 qporq 标志，用于比较 p 和 q
    int qporq = (p <= q);
    // 初始化 DinvrState 结构体 DS
    DinvrState DS = {0};
    // 初始化 DzrorState 结构体 DZ
    DzrorState DZ = {0};

    // 设置 DS 的小值
    DS.small = 1e-100;
    // 设置 DS 的大值
    DS.big = 1e100;
    // 设置 DS 的绝对步长
    DS.absstp = 0.5;
    // 设置 DS 的相对步长
    DS.relstp = 0.5;
    // 设置 DS 的步长倍增因子
    DS.stpmul = 5.;
    // 设置 DS 的绝对容差
    DS.abstol = atol;
    // 设置 DS 的相对容差
    DS.reltol = tol;
    // 设置 DS 的初始值 x
    DS.x = 5.0;
    // 初始化返回结构体 TupleDD betret
    struct TupleDD betret;
    // 初始化返回结构体 TupleDID ret
    struct TupleDID ret = {0};

    // 检查 p 的范围，若不在 [0, 1] 内则返回错误码和条件判断结果
    if (!((0 <= p) && (p <= 1.0))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 的范围，若不在 [0, 1] 内则返回错误码和条件判断结果
    if (!((0 <= q) && (q <= 1.0))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 x 的范围，若不在 [0, 1] 内则返回错误码和条件判断结果
    if (!((0 <= x) && (x <= 1.0))) {
        ret.i1 = -3;
        ret.d2 = (!(x > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 y 的范围，若不在 [0, 1] 内则返回错误码和条件判断结果
    if (!((0 <= y) && (y <= 1.0))) {
        ret.i1 = -4;
        ret.d2 = (!(y > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 a 是否大于 0，若不是则返回错误码
    if (!(0 < a)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -5};
    }
    // 检查参数的合法性，若不满足条件则返回错误码
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3, .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }
    // 检查参数的合法性，若不满足条件则返回错误码
    if (((fabs(x+y)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 4, .d2 = (x+y < 0 ? 0.0 : 1.0)};
    }

    // 调用 dinvr 函数计算结果
    dinvr(&DS, &DZ);
    // 当 DS 的状态为 1 时循环执行以下代码
    while (DS.status == 1) {
        // 调用 cumbet 函数计算累积二项分布的参数
        betret = cumbet(x, y, a, DS.x);
        // 根据 qporq 的值更新 DS 的 fx 字段
        DS.fx = (qporq ? betret.d1 - p : betret.d2 - q);
        // 再次调用 dinvr 函数更新 DS 的状态
        dinvr(&DS, &DZ);
    }

    // 根据 DS 的状态返回不同的结果
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 1e-100 : 1e100);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}


注释部分详细解释了每行代码的作用，包括变量初始化、条件检查、函数调用以及返回结果的处理。
    //                    DOUBLE PRECISION Q
    //
    //     S <--> The number of successes observed.
    //            Input range: [0, XN]
    //            Search range: [0, XN]
    //                    DOUBLE PRECISION S
    //
    //     XN  <--> The number of binomial trials.
    //              Input range: (0, +infinity).
    //              Search range: [1E-100, 1E100]
    //                    DOUBLE PRECISION XN
    //
    //     PR  <--> The probability of success in each binomial trial.
    //              Input range: [0,1].
    //              Search range: [0,1]
    //                    DOUBLE PRECISION PR
    //
    //     OMPR  <--> 1-PR
    //              Input range: [0,1].
    //              Search range: [0,1]
    //              PR + OMPR = 1.0
    //                    DOUBLE PRECISION OMPR
    //
    //     STATUS <-- 0 if calculation completed correctly
    //               -I if input parameter number I is out of range
    //                1 if answer appears to be lower than lowest
    //                  search bound
    //                2 if answer appears to be higher than greatest
    //                  search bound
    //                3 if P + Q .ne. 1
    //                4 if PR + OMPR .ne. 1
    //                    INTEGER STATUS
    //
    //     BOUND <-- Undefined if STATUS is 0
    //
    //               Bound exceeded by parameter number I if STATUS
    //               is negative.
    //
    //               Lower search bound if STATUS is 1.
    //
    //               Upper search bound if STATUS is 2.
    //
    //
    //                              Method
    //
    //
    //     Formula  26.5.24    of   Abramowitz  and    Stegun,  Handbook   of
    //     Mathematical   Functions (1966) is   used  to reduce the  binomial
    //     distribution  to  the  cumulative incomplete    beta distribution.
    //
    //     Computation of other parameters involve a search for a value that
    //     produces  the desired  value  of P.   The search relies  on  the
    //     monotinicity of P with the other parameter.
    //
    //
    //**********************************************************************
// 根据给定的参数计算二项分布的累积分布函数值及其相关信息
struct TupleDDID cdfbin_which1(double s, double xn, double pr, double ompr)
{
    // 检查 s 是否在合理范围内，若不在则返回默认值
    if (!((0 <= s) && (s <= xn))) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1, .d3 = 0.0};
    }
    // 检查 xn 是否大于零，若不是则返回默认值
    if (!(0 < xn)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }
    // 检查 pr 是否在 [0, 1) 范围内，若不在则返回默认值
    if (!((0 <= pr) && (pr < 1))) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3,
                                  .d3 = (pr > 0.0 ? 0.0 : 1.0)};
    }
    // 检查 ompr 是否在 [0, 1) 范围内，若不在则返回默认值
    if (!((0 <= ompr) && (ompr < 1))) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4,
                                  .d3 = (ompr > 0.0 ? 0.0 : 1.0)};
    }
    // 检查是否满足特定条件，若不满足则返回默认值
    if (((fabs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = 4,
                                  .d3 = (pr+ompr < 0 ? 0.0 : 1.0)};
    }
    // 计算二项分布的累积分布函数并返回结果
    struct TupleDD res = cumbin(s, xn, pr, ompr);
    return (struct TupleDDID){.d1 = res.d1, .d2 = res.d2, .i1 = 0, .d3 = 0.0};
}


// 根据给定的参数计算二项分布的逆累积分布函数值及其相关信息
struct TupleDID cdfbin_which2(double p, double q, double xn, double pr, double ompr)
{
    // 定义精度相关的参数
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否在 [0, 1] 范围内，若不在则返回默认值
    int qporq = (p <= q);
    DinvrState DS = {0};
    DzrorState DZ = {0};

    DS.small = 0.0;
    DS.big = xn;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = xn / 2.0;
    struct TupleDD binret;
    struct TupleDID ret = {0};

    // 检查 p 是否在 [0, 1] 范围内，若不在则返回默认值
    if (!((0 <= p) && (p <= 1.0))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 是否在 [0, 1] 范围内，若不在则返回默认值
    if (!((0 <= q) && (q <= 1.0))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 xn 是否大于零，若不是则返回默认值
    if (!(0 < xn)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 检查 pr 是否在 [0, 1] 范围内，若不在则返回默认值
    if (!((0 <= pr) && (pr <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .i1 = -4, .d2 = (pr > 0.0 ? 0.0 : 1.0)};
    }
    // 检查 ompr 是否在 [0, 1) 范围内，若不在则返回默认值
    if (!((0 <= ompr) && (ompr < 1))) {
        return (struct TupleDID){.d1 = 0.0, .i1 = -5, .d2 = (ompr > 0.0 ? 0.0 : 1.0)};
    }
    // 检查是否满足特定条件，若不满足则返回默认值
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3, .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }
    // 检查是否满足特定条件，若不满足则返回默认值
    if (((fabs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 4, .d2 = (pr+ompr < 0 ? 0.0 : 1.0)};
    }
    // 计算二项分布的逆累积分布函数并返回结果
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        binret = cumbin(DS.x, xn, pr, ompr);
        DS.fx = (qporq ? binret.d1 - p : binret.d2 - q);
        dinvr(&DS, &DZ);
    }

    // 根据逆累积分布函数计算结果设置返回值
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : xn);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}


// 根据给定的参数计算二项分布的逆累积分布函数值及其相关信息
struct TupleDID cdfbin_which3(double p, double q, double s, double pr, double ompr)
{
    // 定义精度相关的参数
    double tol = 1e-10;
    double atol = 1e-50;
    int qporq = (p <= q);
    DinvrState DS = {0};
    DzrorState DZ = {0};

    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.0;
    struct TupleDD binret;
    struct TupleDID ret = {0};

    // 检查参数 p 的范围是否有效，若无效则返回相应的错误码和默认值
    if (!((0 <= p) && (p <= 1.0))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查参数 q 的范围是否有效，若无效则返回相应的错误码和默认值
    if (!((0 <= q) && (q <= 1.0))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查参数 s 的范围是否有效，若无效则返回相应的错误码和默认值
    if (!(0 <= s)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 检查参数 pr 的范围是否有效，若无效则返回相应的错误码和默认值
    if (!((0 <= pr) && (pr <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .i1 = -4, .d2 = (pr > 0.0 ? 0.0 : 1.0)};
    }
    // 检查参数 ompr 的范围是否有效，若无效则返回相应的错误码和默认值
    if (!((0 <= ompr) && (ompr < 1))) {
        return (struct TupleDID){.d1 = 0.0, .i1 = -5, .d2 = (ompr > 0.0 ? 0.0 : 1.0)};
    }
    // 检查 p+q 是否超出范围，若超出则返回相应的错误码和默认值
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3, .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }
    // 检查 pr+ompr 是否超出范围，若超出则返回相应的错误码和默认值
    if (((fabs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 4, .d2 = (pr+ompr < 0 ? 0.0 : 1.0)};
    }

    // 计算和迭代过程，直到 DS.status 不再为 1
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        // 调用 cumbin 函数计算二项累积分布函数值
        binret = cumbin(s, DS.x, pr, ompr);
        // 更新 DS.fx 的值，根据 qporq 判断是 binret.d1 - p 还是 binret.d2 - q
        DS.fx = (qporq ? binret.d1 - p : binret.d2 - q);
        // 继续迭代
        dinvr(&DS, &DZ);
    }

    // 根据 DS.status 的值返回相应的结果
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 1e-100 : 1e100);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}


// 结构体 TupleDDID 的函数 cdfbin_which4，用于计算二项分布的累积分布函数
struct TupleDDID cdfbin_which4(double p, double q, double s, double xn)
{
    // 声明变量 pr 和 ompr，以及容差和绝对容差
    double pr, ompr;
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q，结果保存在 qporq 变量中
    int qporq = (p <= q);
    // 初始化 DzrorState 结构体 DZ，设定其各项值
    DzrorState DZ = {0};

    // 设定 DZ 结构体的上下界和容差值
    DZ.xlo = 0.0;
    DZ.xhi = 1.0;
    DZ.atol = atol;
    DZ.rtol = tol;
    DZ.x = 0.0;
    DZ.b = 0.0;
    // 声明返回值结构体 TupleDD 和 TupleDDID
    struct TupleDD binret;
    struct TupleDDID ret = {0};

    // 检查 p 是否在 [0, 1] 范围内，若不是则返回特定错误码和值
    if (!((0 <= p) && (p <= 1.0))) {
        ret.i1 = -1;
        ret.d3 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 是否在 [0, 1] 范围内，若不是则返回特定错误码和值
    if (!((0 <= q) && (q <= 1.0))) {
        ret.i1 = -2;
        ret.d3 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 s 是否在 [0, xn] 范围内，若不是则返回特定错误码和值
    if (!((0 <= s) && (s <= xn))) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }
    // 检查 xn 是否大于 0，若不是则返回特定错误码和值
    if (!(0 < xn)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4, .d3 = 0.0};
    }
    // 检查 fabs(p+q)-0.5 是否大于 3*spmpar[0] 的值，若是则返回特定错误码和值
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = 3,
                                  .d3 = (p+q < 0 ? 0.0 : 1.0)};
    }

    // 根据 qporq 的值选择不同的计算路径
    if (qporq) {
        // 调用 dzror 函数处理 DZ 结构体，然后计算 ompr 和 pr
        dzror(&DZ);
        ompr = 1.0 - DZ.x;
        while (DZ.status == 1) {
            // 调用 cumbin 函数计算二项分布的累积分布函数值
            binret = cumbin(s, xn, DZ.x, ompr);
            // 更新 DZ 结构体的 fx 值，并再次调用 dzror 函数
            DZ.fx = binret.d1 - p;
            dzror(&DZ);
            ompr = 1.0 - DZ.x;
        }
        pr = DZ.x;
    } else {
        // 调用 dzror 函数处理 DZ 结构体，然后计算 pr 和 ompr
        dzror(&DZ);
        pr = 1.0 - DZ.x;
        while (DZ.status == 1) {
            // 调用 cumbin 函数计算二项分布的累积分布函数值
            binret = cumbin(s, xn, pr, DZ.x);
            // 更新 DZ 结构体的 fx 值，并再次调用 dzror 函数
            DZ.fx = binret.d2 - q;
            dzror(&DZ);
            pr = 1.0 - DZ.x;
        }
        ompr = DZ.x;
    }

    // 根据 DZ 结构体的状态值返回不同的 TupleDDID 结构体
    if (DZ.status == -1) {
        ret.d1 = pr;
        ret.d2 = ompr;
        ret.i1 = (DZ.qleft ? 1 : 2);
        ret.d3 = (DZ.qleft ? 0 : 1);
        return ret;
    } else {
        ret.d1 = pr;
        ret.d2 = ompr;
        return ret;
    }
}


    // 累积分布函数
    // CHI-Square 分布
    //
    //
    // 函数
    //
    //
    // 计算卡方分布的任一参数，给定其他参数的值。
    //
    //
    // 参数
    //
    //
    // WHICH --> 整数，指示从下面三个参数中计算哪一个。
    //           合法范围: 1..3
    //           iwhich = 1 : 从 X 和 DF 计算 P 和 Q
    //           iwhich = 2 : 从 P, Q 和 DF 计算 X
    //           iwhich = 3 : 从 P, Q 和 X 计算 DF
    //                    INTEGER WHICH
    //
    // P <--> 卡方分布的积分从 0 到 X 的值。
    //         输入范围: [0, 1]。
    //                    DOUBLE PRECISION P
    //
    // Q <--> 1-P。
    //         输入范围: (0, 1]。
    //         P + Q = 1.0。
    //                    DOUBLE PRECISION Q
    //
    // X <--> 非中心卡方分布的上限
    //            chi-square distribution.
    //            Input range: [0, +infinity).
    //            Search range: [0,1E100]
    //                    DOUBLE PRECISION X
    //
    //     DF <--> Degrees of freedom of the
    //             chi-square distribution.
    //             Input range: (0, +infinity).
    //             Search range: [ 1E-100, 1E100]
    //                    DOUBLE PRECISION DF
    //
    //     STATUS <-- 0 if calculation completed correctly
    //               -I if input parameter number I is out of range
    //                1 if answer appears to be lower than lowest
    //                  search bound
    //                2 if answer appears to be higher than greatest
    //                  search bound
    //                3 if P + Q .ne. 1
    //               10 indicates error returned from cumgam.  See
    //                  references in cdfgam
    //                    INTEGER STATUS
    //
    //     BOUND <-- Undefined if STATUS is 0
    //
    //               Bound exceeded by parameter number I if STATUS
    //               is negative.
    //
    //               Lower search bound if STATUS is 1.
    //
    //               Upper search bound if STATUS is 2.
    //
    //
    //                              Method
    //
    //
    //     Formula    26.4.19   of Abramowitz  and     Stegun, Handbook  of
    //     Mathematical Functions   (1966) is used   to reduce the chisqure
    //     distribution to the incomplete distribution.
    //
    //     Computation of other parameters involve a search for a value that
    //     produces  the desired  value  of P.   The search relies  on  the
    //     monotinicity of P with the other parameter.
    //
    //**********************************************************************
struct TupleDDID cdfchi_which1(double x, double df)
{
    // 定义结构体变量 chiret 来存储累积卡方分布函数的结果
    struct TupleDD chiret;
    
    // 检查 x 是否大于等于 0，若不是则返回错误代码和默认值
    if (!(0 <= x)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1, .d3 = 0.0};
    }
    
    // 检查 df 是否大于等于 0，若不是则返回错误代码和默认值
    if (!(0 <= df)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }
    
    // 调用 cumchi 函数计算累积卡方分布函数的值
    chiret = cumchi(x, df);
    
    // 返回结果结构体，其中 i1 为 0，d3 为默认值 0.0
    return (struct TupleDDID){.d1 = chiret.d1, .d2 = chiret.d2, .i1 = 0, .d3 = 0.0};
}


struct TupleDID cdfchi_which2(double p, double q, double df)
{
    // 定义误差容限 tol 和 atol
    double tol = 1e-10;
    double atol = 1e-50;
    
    // 判断 p 是否在 [0, 1] 范围内，若不是返回错误代码和 p 是否大于 0 的结果
    int qporq = (p <= q);
    int porq = (qporq ? p : q);
    DinvrState DS = {0};
    DzrorState DZ = {0};

    // 初始化 DS 结构体的参数
    DS.small = 0.0;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    struct TupleDD chiret;
    struct TupleDID ret = {0};

    // 检查 p 是否在 [0, 1] 范围内，若不是返回错误代码和 p 是否大于 0 的结果
    if (!((0 <= p) && (p <= 1.0))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    
    // 检查 q 是否在 (0, 1] 范围内，若不是返回错误代码和 q 是否大于 0 的结果
    if (!((0 < q) && (q <= 1.0))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    
    // 检查 df 是否大于等于 0，若不是返回错误代码和默认值
    if (!(0 <= df)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    
    // 检查表达式是否超出范围，若超出返回错误代码和是否小于 0 的结果
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3,
                                 .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }

    // 调用 dinvr 函数并进入循环，直到 DS.status 不为 1
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        // 调用 cumchi 函数计算累积卡方分布函数的值
        chiret = cumchi(DS.x, df);
        
        // 根据 qporq 的值计算 DS.fx，并检查是否满足条件，若满足则返回结果
        DS.fx = (qporq ? chiret.d1 - p : chiret.d2 - q);
        if (DS.fx + porq <= 1.5) {
            return (struct TupleDID){.d1 = DS.x, .i1 = 10, .d2 = 0.0};
        }
        
        // 再次调用 dinvr 函数
        dinvr(&DS, &DZ);
    }

    // 根据 DS.status 的值返回不同的结果
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e100);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}


struct TupleDID cdfchi_which3(double p, double q, double x)
{
    // 定义误差容限 tol 和 atol
    double tol = 1e-10;
    double atol = 1e-50;
    
    // 判断 p 是否在 [0, 1] 范围内，若不是返回错误代码和 p 是否大于 0 的结果
    int qporq = (p <= q);
    int porq = (qporq ? p : q);
    DinvrState DS = {0};
    DzrorState DZ = {0};

    // 初始化 DS 结构体的参数
    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    struct TupleDD chiret;
    struct TupleDID ret = {0};

    // 检查 p 是否在 [0, 1] 范围内，若不是返回错误代码和 p 是否大于 0 的结果
    if (!((0 <= p) && (p <= 1.0))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    
    // 检查 q 是否在 (0, 1] 范围内，若不是返回错误代码和 q 是否大于 0 的结果
    if (!((0 < q) && (q <= 1.0))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    
    // 检查 x 是否大于等于 0，若不是返回错误代码和默认值
    if (!(0 <= x)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    
    // 检查表达式是否超出范围，若超出返回错误代码和是否小于 0 的结果
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3,
                                 .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }

    // 调用 dinvr 函数并进入循环，直到 DS.status 不为 1
    dinvr(&DS, &DZ);
    # 当 DS 的状态为 1 时进入循环
    while (DS.status == 1) {
        # 计算累积卡方分布函数值
        chiret = cumchi(x, DS.x);
        # 根据 qporq 的值选择不同的表达式计算 DS.fx
        DS.fx = (qporq ? chiret.d1 - p : chiret.d2 - q);
        # 如果 DS.fx 加上 porq 大于 1.5，则返回一个特定的结构体
        if (DS.fx + porq > 1.5) {
            return (struct TupleDID){.d1 = DS.x, .i1 = 10, .d2 = 0.0};
        }
        # 更新 DS 和 DZ 的状态
        dinvr(&DS, &DZ);
    }

    # 如果 DS 的状态为 -1
    if (DS.status == -1) {
        # 设置返回结构体 ret 的各个字段值
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 1e-100 : 1e100);
        # 返回设置好的结构体 ret
        return ret;
    } else {
        # 如果 DS 的状态不为 -1，则只设置返回结构体 ret 的 d1 字段
        ret.d1 = DS.x;
        # 返回设置好的结构体 ret
        return ret;
    }
    //               Cumulative Distribution Function
    //               Non-central Chi-Square
    //
    //
    //                              Function
    //
    //
    //     Calculates any one parameter of the non-central chi-square
    //     distribution given values for the others.
    //
    //
    //                              Arguments
    //
    //
    //     WHICH --> Integer indicating which of the next three argument
    //               values is to be calculated from the others.
    //               Input range: 1..4
    //               iwhich = 1 : Calculate P and Q from X and DF
    //               iwhich = 2 : Calculate X from P,DF and PNONC
    //               iwhich = 3 : Calculate DF from P,X and PNONC
    //               iwhich = 3 : Calculate PNONC from P,X and DF
    //                    INTEGER WHICH
    //
    //     P <--> The integral from 0 to X of the non-central chi-square
    //            distribution.
    //            Input range: [0, 1-1E-16).
    //                    DOUBLE PRECISION P
    //
    //     Q <--> 1-P.
    //            Q is not used by this subroutine and is only included
    //            for similarity with other cdf* routines.
    //                    DOUBLE PRECISION Q
    //
    //     X <--> Upper limit of integration of the non-central
    //            chi-square distribution.
    //            Input range: [0, +infinity).
    //            Search range: [0,1E300]
    //                    DOUBLE PRECISION X
    //
    //     DF <--> Degrees of freedom of the non-central
    //             chi-square distribution.
    //             Input range: (0, +infinity).
    //             Search range: [ 1E-300, 1E300]
    //                    DOUBLE PRECISION DF
    //
    //     PNONC <--> Non-centrality parameter of the non-central
    //                chi-square distribution.
    //                Input range: [0, +infinity).
    //                Search range: [0,1E4]
    //                    DOUBLE PRECISION PNONC
    //
    //     STATUS <-- 0 if calculation completed correctly
    //               -I if input parameter number I is out of range
    //                1 if answer appears to be lower than lowest
    //                  search bound
    //                2 if answer appears to be higher than greatest
    //                  search bound
    //                    INTEGER STATUS
    //
    //     BOUND <-- Undefined if STATUS is 0
    //
    //               Bound exceeded by parameter number I if STATUS
    //               is negative.
    //
    //               Lower search bound if STATUS is 1.
    //
    //               Upper search bound if STATUS is 2.
    //
    //
    //                              Method
    //
    //
    //     Formula  26.4.25   of   Abramowitz   and   Stegun,  Handbook  of
    //     Mathematical  Functions (1966) is used to compute the cumulative
    //     distribution function.
    //
    // 计算其他参数的过程涉及搜索一个值，使得 P 的值达到期望的数值。
    // 这个搜索依赖于 P 随其他参数的单调性。

    // 警告：
    // 此例程所需的计算时间与非中心参数（PNONC）成正比。
    // 非常大的 PNONC 值可能会消耗巨大的计算资源。因此搜索范围被限制在 1e9 内。

    //**********************************************************************
// 定义一个函数 cdfchn_which1，返回一个结构体 TupleDDID
struct TupleDDID cdfchn_which1(double x, double df, double pnonc)
{
    // 将 x 限制在 spmpar[2] 的最小值内
    x = fmin(x, spmpar[2]);
    // 将 df 限制在 spmpar[2] 的最小值内
    df = fmin(df, spmpar[2]);
    // 将 pnonc 限制在 1e9 内
    pnonc = fmin(pnonc, 1e9);

    // 声明一个结构体 TupleDD 类型的变量 chnret
    struct TupleDD chnret;
    
    // 如果 x 不大于等于 0，则返回一个特定结构体
    if (!(0 <= x)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1, .d3 = 0.0};
    }
    // 如果 df 不大于等于 0，则返回一个特定结构体
    if (!(0 <= df)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }
    // 如果 pnonc 不大于等于 0，则返回一个特定结构体
    if (!(0 <= pnonc)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }

    // 调用 cumchn 函数计算 chnret 的值
    chnret = cumchn(x, df, pnonc);
    // 返回一个结构体 TupleDDID，其中 d1 和 d2 值来自 chnret，i1 为 0，d3 为 0.0
    return (struct TupleDDID){.d1 = chnret.d1, .d2 = chnret.d2, .i1 = 0, .d3 = 0.0};
}


// 定义一个函数 cdfchn_which2，返回一个结构体 TupleDID
struct TupleDID cdfchn_which2(double p, double df, double pnonc)
{
    // 定义容差和状态结构体 DS 和 DZ，并初始化其中的值
    double tol = 1e-10;
    double atol = 1e-50;
    DinvrState DS = {0};
    DzrorState DZ = {0};

    DS.small = 0.;
    DS.big = 1e300;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    
    // 声明一个结构体 TupleDD 和 TupleDID 类型的变量
    struct TupleDD chnret;
    struct TupleDID ret = {0};

    // 将 df 限制在 spmpar[2] 的最小值内
    df = fmin(df, spmpar[2]);
    // 将 pnonc 限制在 1.e9 内
    pnonc = fmin(pnonc, 1.e9);

    // 如果 p 不在 [0, 1 - 1e-16] 区间内，则返回一个特定结构体
    if (!((0 <= p) && (p <= (1. - 1e-16)))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : (1. - 1e-16));
        return ret;
    }
    // 如果 df 不大于等于 0，则返回一个特定结构体
    if (!(0 <= df)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2};
    }
    // 如果 pnonc 不大于等于 0，则返回一个特定结构体
    if (!(0 <= pnonc)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }

    // 调用 dinvr 函数处理 DS 和 DZ 结构体
    dinvr(&DS, &DZ);
    // 当 DS.status 为 1 时循环执行以下代码
    while (DS.status == 1) {
        // 调用 cumchn 函数计算 chnret 的值
        chnret = cumchn(DS.x, df, pnonc);
        // DS.fx 为 chnret.d1 - p
        DS.fx = chnret.d1 - p;
        // 继续调用 dinvr 函数处理 DS 和 DZ 结构体
        dinvr(&DS, &DZ);
    }

    // 如果 DS.status 为 -1，则返回特定的 TupleDID 结构体
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e300);
        return ret;
    } else {
        // 否则返回一个包含 DS.x 的 TupleDID 结构体
        ret.d1 = DS.x;
        return ret;
    }
}


// 定义一个函数 cdfchn_which3，返回一个结构体 TupleDID
struct TupleDID cdfchn_which3(double p, double x, double pnonc)
{
    // 定义容差和状态结构体 DS 和 DZ，并初始化其中的值
    double tol = 1e-10;
    double atol = 1e-50;
    DinvrState DS = {0};
    DzrorState DZ = {0};

    DS.small = 1e-300;
    DS.big = 1e300;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.0;
    
    // 声明一个结构体 TupleDD 和 TupleDID 类型的变量
    struct TupleDD chnret;
    struct TupleDID ret = {0};

    // 将 x 限制在 spmpar[2] 的最小值内
    x = fmin(x, spmpar[2]);
    // 将 pnonc 限制在 1.e9 内
    pnonc = fmin(pnonc, 1.e9);

    // 如果 p 不在 [0, 1] 区间内，则返回一个特定结构体
    if (!((0 <= p) && (p <= 1))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : (1. - 1e-16));
        return ret;
    }
    // 如果 x 不大于等于 0，则返回一个特定结构体
    if (!(0 <= x)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2};
    }
    // 如果 pnonc 不大于等于 0，则返回一个特定结构体
    if (!(0 <= pnonc)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }

    // 调用 dinvr 函数处理 DS 和 DZ 结构体
    dinvr(&DS, &DZ);
    // 当 DS.status 为 1 时循环执行以下代码
    while (DS.status == 1) {
        // 调用 cumchn 函数计算 chnret 的值
        chnret = cumchn(x, DS.x, pnonc);
        // DS.fx 为 chnret.d1 - p
        DS.fx = chnret.d1 - p;
        // 继续调用 dinvr 函数处理 DS 和 DZ 结构体
        dinvr(&DS, &DZ);
    }

    // 如果 DS.status 为 -1，则返回特定的 TupleDID 结构体
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 1e-300 : 1e300);
        return ret;
    } else {
        // 否则返回一个包含 DS.x 的
    // 设置浮点数的容差和绝对容差阈值
    double tol = 1e-10;
    double atol = 1e-50;

    // 初始化 DinvrState 和 DzrorState 结构体，均赋初值为零
    DinvrState DS = {0};
    DzrorState DZ = {0};

    // 设置 DS 结构体的各个参数值
    DS.small = 0.0;
    DS.big = 1e9;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.0;

    // 声明结构体 TupleDD 变量 chnret
    struct TupleDD chnret;

    // 声明并初始化结构体 TupleDID 变量 ret，将其整型字段 i1 初始化为零
    struct TupleDID ret = {0};

    // 限制变量 x 和 df 的取值不超过 spmpar[2]
    x = fmin(x, spmpar[2]);
    df = fmin(df, spmpar[2]);

    // 检查概率值 p 是否在区间 [0, 1] 内，若不在则返回错误码和修正值
    if (!((0 <= p) && (p <= 1))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : (1. - 1e-16));
        return ret;
    }

    // 检查 x 是否非负，若不是则返回错误码
    if (!(0 <= x)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2};
    }

    // 检查 df 是否非负，若不是则返回错误码
    if (!(0 <= df)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }

    // 调用 dinvr 函数计算初始状态下 DS 和 DZ 结构体的值
    dinvr(&DS, &DZ);

    // 循环直到 DS 结构体的状态不为 1
    while (DS.status == 1) {
        // 调用 cumchn 函数计算 chnret 结构体的值
        chnret = cumchn(x, df, DS.x);
        // 更新 DS 结构体中的 fx 值
        DS.fx = chnret.d1 - p;
        // 再次调用 dinvr 函数更新 DS 和 DZ 结构体的值
        dinvr(&DS, &DZ);
    }

    // 根据 DS 结构体的状态值返回相应的结果
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e9);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
    //               Cumulative Distribution Function
    //               F distribution
    //
    //
    //                              Function
    //
    //
    //     Calculates any one parameter of the F distribution
    //     given values for the others.
    //
    //
    //                              Arguments
    //
    //
    //     WHICH --> Integer indicating which of the next four argument
    //               values is to be calculated from the others.
    //               Legal range: 1..4
    //               iwhich = 1 : Calculate P and Q from F,DFN and DFD
    //               iwhich = 2 : Calculate F from P,Q,DFN and DFD
    //               iwhich = 3 : Calculate DFN from P,Q,F and DFD
    //               iwhich = 4 : Calculate DFD from P,Q,F and DFN
    //                    INTEGER WHICH
    //
    //       P <--> The integral from 0 to F of the f-density.
    //              Input range: [0,1].
    //                    DOUBLE PRECISION P
    //
    //       Q <--> 1-P.
    //              Input range: (0, 1].
    //              P + Q = 1.0.
    //                    DOUBLE PRECISION Q
    //
    //       F <--> Upper limit of integration of the f-density.
    //              Input range: [0, +infinity).
    //              Search range: [0,1E100]
    //                    DOUBLE PRECISION F
    //
    //     DFN < --> Degrees of freedom of the numerator sum of squares.
    //               Input range: (0, +infinity).
    //               Search range: [ 1E-100, 1E100]
    //                    DOUBLE PRECISION DFN
    //
    //     DFD < --> Degrees of freedom of the denominator sum of squares.
    //               Input range: (0, +infinity).
    //               Search range: [ 1E-100, 1E100]
    //                    DOUBLE PRECISION DFD
    //
    //     STATUS <-- 0 if calculation completed correctly
    //               -I if input parameter number I is out of range
    //                1 if answer appears to be lower than lowest
    //                  search bound
    //                2 if answer appears to be higher than greatest
    //                  search bound
    //                3 if P + Q .ne. 1
    //                    INTEGER STATUS
    //
    //     BOUND <-- Undefined if STATUS is 0
    //
    //               Bound exceeded by parameter number I if STATUS
    //               is negative.
    //
    //               Lower search bound if STATUS is 1.
    //
    //               Upper search bound if STATUS is 2.
    //
    //
    //                              Method
    //
    //
    //     Formula   26.6.2   of   Abramowitz   and   Stegun,  Handbook  of
    //     Mathematical  Functions (1966) is used to reduce the computation
    //     of the  cumulative  distribution function for the  F  variate to
    //     that of an incomplete beta.
    //
    //     Computation of other parameters involve a search for a value that
    # 以下是多行注释，用于说明下面代码的一些注意事项和警告
    # 生成所需的 P 值。搜索依赖于 P 与另一个参数的单调性。
    #
    #                              警告
    #
    #     累积 F 分布的值在自由度上未必单调递增。因此可能存在给定 CDF 值的两个解。
    #     此程序假设单调性并将找到其中的一个任意解。
    #
    #**********************************************************************
struct TupleDDID cdff_which1(double f, double dfn, double dfd)
{
    // 声明返回值结构体
    struct TupleDD fret;
    // 检查 f 是否大于等于 0
    if (!(0 <= f)) {
        // 返回错误信息和标识符
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1, .d3 = 0.0};
    }
    // 检查 dfn 是否大于 0
    if (!(0 < dfn)) {
        // 返回错误信息和标识符
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }
    // 检查 dfd 是否大于 0
    if (!(0 < dfd)) {
        // 返回错误信息和标识符
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }

    // 调用 cumf 函数计算累积分布函数的值
    fret = cumf(f, dfn, dfd);
    // 返回结构体，将结果赋给结构体字段并设置其余字段为默认值
    return (struct TupleDDID){.d1 = fret.d1, .d2 = fret.d2, .i1 = 0, .d3 = 0.0};
}


struct TupleDID cdff_which2(double p, double q, double dfn, double dfd)
{
    // 设置公差值
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q
    int qporq = (p <= q);
    // 初始化状态结构体
    DinvrState DS = {0};
    DzrorState DZ = {0};

    // 设置状态结构体的初始值
    DS.small = 0.;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    // 声明返回值结构体
    struct TupleDD fret;
    struct TupleDID ret = {0};

    // 检查 p 是否在 [0, 1] 区间内
    if (!((0 <= p) && (p <= 1.0))) {
        // 返回错误信息和标识符
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 是否在 (0, 1] 区间内
    if (!((0 < q) && (q <= 1.0))) {
        // 返回错误信息和标识符
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 dfn 是否大于 0
    if (!(0 < dfn)) {
        // 返回错误信息和标识符
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 检查 dfd 是否大于 0
    if (!(0 < dfd)) {
        // 返回错误信息和标识符
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    // 检查 p 和 q 的组合是否超过阈值
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        // 返回错误信息和标识符
        return (struct TupleDID){.d1 = 0.0, .i1 = 3,
                                 .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }

    // 调用 dinvr 函数进行求解
    dinvr(&DS, &DZ);
    // 循环执行，直到状态不为 1
    while (DS.status == 1) {
        // 调用 cumf 函数计算累积分布函数的值
        fret = cumf(DS.x, dfn, dfd);
        // 更新 DS 结构体的 fx 字段
        DS.fx = (qporq ? fret.d1 - p : fret.d2 - q);
        // 再次调用 dinvr 函数进行求解
        dinvr(&DS, &DZ);
    }

    // 检查最终状态
    if (DS.status == -1) {
        // 返回求解结果和状态信息
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e100);
        return ret;
    } else {
        // 返回求解结果和状态信息
        ret.d1 = DS.x;
        return ret;
    }
}


struct TupleDID cdff_which3(double p, double q, double f, double dfd)
{
    // 设置公差值
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q
    int qporq = (p <= q);
    // 初始化状态结构体
    DinvrState DS = {0};
    DzrorState DZ = {0};

    // 设置状态结构体的初始值
    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    // 声明返回值结构体
    struct TupleDD fret;
    struct TupleDID ret = {0};

    // 检查 p 是否在 [0, 1] 区间内
    if (!((0 <= p) && (p <= 1.0))) {
        // 返回错误信息和标识符
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 是否在 (0, 1] 区间内
    if (!((0 < q) && (q <= 1.0))) {
        // 返回错误信息和标识符
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 f 是否大于等于 0
    if (!(0 <= f)) {
        // 返回错误信息和标识符
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 检查 dfd 是否大于 0
    if (!(0 < dfd)) {
        // 返回错误信息和标识符
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    // 检查 p 和 q 的组合是否超过阈值
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        // 返回错误信息和标识符
        return (struct TupleDID){.d1 = 0.0, .i1 = 3,
                                 .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }

    // 调用 dinvr 函数进行求解
    dinvr(&DS, &DZ);
    // 返回结构体，将结果赋给结构体字段并设置其余字段为默认值
    # 当 DS 的状态为 1 时，执行以下循环
    while (DS.status == 1) {
        # 计算累积分布函数的值，并将结果赋给 fret
        fret = cumf(f, DS.x, dfd);
        # 根据 qporq 的布尔值选择不同的计算结果，并将其赋给 DS.fx
        DS.fx = (qporq ? fret.d1 - p : fret.d2 - q);
        # 调用函数 dinvr 处理 DS 和 DZ 对象
        dinvr(&DS, &DZ);
    }
    
    # 如果 DS 的状态为 -1，则执行以下逻辑
    if (DS.status == -1) {
        # 设置返回结构体 ret 的第一个浮点数字段为 DS.x
        ret.d1 = DS.x;
        # 设置返回结构体 ret 的整数字段为 1（如果 DS.qleft 为真）或 2（如果 DS.qleft 为假）
        ret.i1 = (DS.qleft ? 1 : 2);
        # 设置返回结构体 ret 的第二个浮点数字段为 1e-100（如果 DS.qleft 为真）或 1e100（如果 DS.qleft 为假）
        ret.d2 = (DS.qleft ? 1e-100 : 1e100);
        # 返回 ret 结构体
        return ret;
    } else {
        # 如果 DS 的状态不是 -1，则执行以下逻辑
        # 设置返回结构体 ret 的第一个浮点数字段为 DS.x
        ret.d1 = DS.x;
        # 返回 ret 结构体
        return ret;
    }
struct TupleDID cdff_which4(double p, double q, double f, double dfn)
{
    // 设置容差值和绝对容差值
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q
    int qporq = (p <= q);
    // 初始化逆变换状态结构体
    DinvrState DS = {0};
    // 初始化零值状态结构体
    DzrorState DZ = {0};

    // 设置逆变换状态的小值和大值
    DS.small = 1e-100;
    DS.big = 1e100;
    // 设置逆变换状态的绝对步长和相对步长
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    // 设置逆变换状态的步长倍数和容差值
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    // 初始化返回的双精度浮点数和整型结构体
    struct TupleDD fret;
    struct TupleDID ret = {0};

    // 检查 p 的范围是否合法
    if (!((0 <= p) && (p <= 1.0))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 的范围是否合法
    if (!((0 < q) && (q <= 1.0))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 f 的范围是否合法
    if (!(0 <= f)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 检查 dfn 的范围是否合法
    if (!(0 < dfn)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    // 检查非中心分布 F 的条件
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3,
                                 .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }

    // 调用逆变换函数 dinvr 初始化状态
    dinvr(&DS, &DZ);
    // 循环直到逆变换状态不为 1
    while (DS.status == 1) {
        // 计算累积分布函数
        fret = cumf(f, dfn, DS.x);
        // 根据 qporq 的值计算 DS.fx
        DS.fx = (qporq ? fret.d1 - p : fret.d2 - q);
        // 继续调用逆变换函数 dinvr
        dinvr(&DS, &DZ);
    }

    // 处理逆变换状态为 -1 的情况
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 1e-100 : 1e100);
        return ret;
    } else {
        // 返回状态值和 DS.x
        ret.d1 = DS.x;
        return ret;
    }
}


注释部分详细解释了每行代码的功能和意图，确保了代码的可读性和理解性。
    // DFN < --> 分子平方和的自由度。
    //         输入范围：(0, +infinity)。
    //         搜索范围：[ 1E-100, 1E100]
    //         双精度变量 DFN
    //
    // DFD < --> 分母平方和的自由度。
    //         必须在范围内：(0, +infinity)。
    //         输入范围：(0, +infinity)。
    //         搜索范围：[ 1E-100, 1E100]
    //         双精度变量 DFD
    //
    // PNONC <-> 非中心参数
    //         输入范围：[0,infinity)
    //         搜索范围：[0,1E4]
    //         双精度变量 PHONC
    //
    // STATUS <-- 0 如果计算完成
    //           -I 如果输入参数编号 I 超出范围
    //            1 如果答案似乎低于最低搜索边界
    //            2 如果答案似乎高于最大搜索边界
    //            3 如果 P + Q ≠ 1
    //            整数变量 STATUS
    //
    // BOUND <-- 如果 STATUS 是 0，则未定义
    //           如果 STATUS 是负数，则参数编号 I 超出界限
    //           如果 STATUS 是 1，则是下限搜索边界
    //           如果 STATUS 是 2，则是上限搜索边界
    //
    //
    //                              方法
    //
    //
    // Abramowitz 和 Stegun 的数学函数手册（1966年）中使用公式 26.6.20 计算累积分布函数。
    //
    // 计算其他参数涉及搜索一个产生所需 P 值的参数值。搜索依赖于 P 随其他参数的单调性。
    //
    //                            警告
    //
    // 此程序所需的计算时间与非中心参数（PNONC）成正比。非常大的参数值可能会消耗巨大的计算资源。这就是为什么搜索范围被限制在10,000的原因。
    //
    //                            警告
    //
    // 累积非中心 F 分布的值在自由度上不一定是单调的。因此可能存在两个值提供给定的CDF值。这个程序假设单调性，并将找到这两个值中的任意一个。
    //
    //**********************************************************************
# 计算给定的 f 值对应的累积分布函数，返回一个结构体 TupleDDID
struct TupleDDID cdffnc_which1(double f, double dfn, double dfd, double phonc)
{
    struct TupleDDI fncret;
    # 检查 f 是否小于等于 0，如果是则返回特定的错误结构体
    if (!(0 <= f)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1, .d3 = 0.0};
    }
    # 检查 dfn 是否小于等于 0，如果是则返回特定的错误结构体
    if (!(0 < dfn)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }
    # 检查 dfd 是否小于等于 0，如果是则返回特定的错误结构体
    if (!(0 < dfd)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }
    # 检查 phonc 是否小于 0，如果是则返回特定的错误结构体
    if (!(0 <= phonc)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4, .d3 = 0.0};
    }
    # 调用 cumfnc 函数计算累积分布函数，并将结果保存在 fncret 中
    fncret = cumfnc(f, dfn, dfd, phonc);
    # 如果 fncret 中的 i1 不等于 0，则返回具有特定标志的结构体
    if (fncret.i1 != 0) {
        return (struct TupleDDID){.d1 = fncret.d1, .d2 = fncret.d2, .i1 = 10, .d3 = 0.0};
    } else {
        return (struct TupleDDID){.d1 = fncret.d1, .d2 = fncret.d2, .i1 = 0, .d3 = 0.0};
    }
}



# 计算给定的 p 值对应的累积分布函数，返回一个结构体 TupleDID
struct TupleDID cdffnc_which2(double p, double q, double dfn, double dfd, double phonc)
{
    double tol = 1e-10;
    double atol = 1e-50;
    DinvrState DS = {0};
    DzrorState DZ = {0};

    DS.small = 0.;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    struct TupleDDI fncret;
    struct TupleDID ret = {0};

    # 检查 p 是否在区间 [0, 1 - 1e-16] 内，如果不是则返回特定的错误结构体
    if (!((0 <= p) && (p <= (1. - 1e-16)))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : (1. - 1e-16));
        return ret;
    }
    # 检查 dfn 是否小于等于 0，如果是则返回特定的错误结构体
    if (!(0 < dfn)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    # 检查 dfd 是否小于等于 0，如果是则返回特定的错误结构体
    if (!(0 < dfd)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    # 检查 phonc 是否小于 0，如果是则返回特定的错误结构体
    if (!(0 <= phonc)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -5};
    }

    # 调用 dinvr 函数来计算特定的数值状态 DS 和 DZ
    dinvr(&DS, &DZ);
    # 在 DS 状态为 1 时循环，直到状态改变
    while (DS.status == 1) {
        # 调用 cumfnc 函数计算累积分布函数，保存结果在 fncret 中
        fncret = cumfnc(DS.x, dfn, dfd, phonc);
        # 计算 DS.fx，并检查 fncret 中的 i1 是否不为 0
        DS.fx = fncret.d1 - p;
        if (fncret.i1 != 0) {
            return (struct TupleDID){.d1 = DS.x, .d2 = 0.0, .i1 = 10};
        }
        # 调用 dinvr 函数更新 DS 和 DZ
        dinvr(&DS, &DZ);
    }

    # 检查 DS 的最终状态，根据结果返回特定的结构体
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e100);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}



# 计算给定的 p 值对应的累积分布函数，返回一个结构体 TupleDID
struct TupleDID cdffnc_which3(double p, double q, double f, double dfd, double phonc)
{
    double tol = 1e-10;
    double atol = 1e-50;
    DinvrState DS = {0};
    DzrorState DZ = {0};

    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    struct TupleDDI fncret;
    struct TupleDID ret = {0};

    # 检查 p 是否在区间 [0, 1 - 1e-16] 内，如果不是则返回特定的错误结构体
    if (!((0 <= p) && (p <= (1. - 1e-16)))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : (1. - 1e-16));
        return ret;
    }
    # 检查 f 是否小于等于 0，如果是则返回特定的错误结构体
    if (!(0 <= f)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    # 检查 dfd 是否小于等于 0，如果是则返回特定的错误结构体
    if (!(0 < dfd)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    # 检查 phonc 是否小于 0，如果是则返回特定的错误结构体
    if (!(0 <= phonc)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -5};
    }

    # 调用 dinvr 函数来计算特定的数值状态 DS 和 DZ
    dinvr(&DS, &DZ);
    # 在 DS 状态为 1 时循环，直到状态改变
    while (DS.status == 1) {
        # 调用 cumfnc 函数计算累积分布函数，保存结果在 fncret 中
        fncret = cumfnc(DS.x, dfn, dfd, phonc);
        DS.fx = fncret.d1 - p;
        # 如果 fncret 中的 i1 不等于 0，则返回具有
    // 当 DS 的状态为 1 时执行循环
    while (DS.status == 1) {
        // 调用 cumfnc 函数计算函数值及其导数，返回结果存入 fncret
        fncret = cumfnc(f, DS.x, dfd, phonc);
        // 计算 DS.fx，即函数值的导数减去 p
        DS.fx = fncret.d1 - p;
        // 如果 fncret.i1 不为 0，则返回一个 TupleDID 结构体，表示计算失败的情况
        if (fncret.i1 != 0) {
            return (struct TupleDID){.d1 = DS.x, .d2 = 0.0, .i1 = 10};
        }
        // 更新 DS 和 DZ 的状态
        dinvr(&DS, &DZ);
    }

    // 当 DS 的状态为 -1 时执行以下代码块
    if (DS.status == -1) {
        // 设置返回结构体 ret 的成员变量
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 1e-100 : 1e100);
        // 返回设置好的结构体 ret
        return ret;
    } else {
        // 当 DS 的状态不为 -1 时，设置返回结构体 ret 的成员变量
        ret.d1 = DS.x;
        // 返回设置好的结构体 ret
        return ret;
    }
    // Cumulative Distribution Function
    // GAMma Distribution
    //
    //
    // Function
    //
    //
    // Calculates any one parameter of the gamma
    // distribution given values for the others.
    //
    //
    // Arguments
    //
    //
    //     WHICH --> Integer indicating which of the next four argument
    //               values is to be calculated from the others.
    //               Legal range: 1..4
    //               iwhich = 1 : Calculate P and Q from X,SHAPE and SCALE
    //               iwhich = 2 : Calculate X from P,Q,SHAPE and SCALE
    //               iwhich = 3 : Calculate SHAPE from P,Q,X and SCALE
    //               iwhich = 4 : Calculate SCALE from P,Q,X and SHAPE
    //                    INTEGER WHICH
    //
    //     P <--> The integral from 0 to X of the gamma density.
    //            Input range: [0,1].
    //                    DOUBLE PRECISION P
    //
    //     Q <--> 1-P.
    //            Input range: (0, 1].
    //            P + Q = 1.0.
    //                    DOUBLE PRECISION Q
    //
    //
    //     X <--> The upper limit of integration of the gamma density.
    //            Input range: [0, +infinity).
    //            Search range: [0,1E100]
    //                    DOUBLE PRECISION X
    //
    //     SHAPE <--> The shape parameter of the gamma density.
    //                Input range: (0, +infinity).
    //                Search range: [1E-100,1E100]
    //                  DOUBLE PRECISION SHAPE
    //
    //
    //     SCALE <--> The scale parameter of the gamma density.
    //                Input range: (0, +infinity).
    //                Search range: (1E-100,1E100]
    //                   DOUBLE PRECISION SCALE
    //
    //     STATUS <-- 0 if calculation completed correctly
    //               -I if input parameter number I is out of range
    //                1 if answer appears to be lower than lowest
    //                  search bound
    //                2 if answer appears to be higher than greatest
    //                  search bound
    //                3 if P + Q .ne. 1
    //                10 if the gamma or inverse gamma routine cannot
    //                   compute the answer.  Usually happens only for
    //                   X and SHAPE very large (gt 1E10 or more)
    //                    INTEGER STATUS
    //
    //     BOUND <-- Undefined if STATUS is 0
    //
    //               Bound exceeded by parameter number I if STATUS
    //               is negative.
    //
    //               Lower search bound if STATUS is 1.
    //
    //               Upper search bound if STATUS is 2.
    //
    //
    //                              Method
    //
    //
    //     Cumulative distribution function (P) is calculated directly by
    //     the code associated with:
    //
    //     DiDinato, A. R. and Morris, A. H. Computation of the  incomplete
    //     gamma function  ratios  and their  inverse.   ACM  Trans.  Math.
    //     Softw. 12 (1986), 377-393.
    //
    //     Computation of other parameters involve a search for a value that
    //     produces  the desired  value  of P.   The search relies  on  the
    //     monotonicity of P with the other parameter.
    //
    //  The gamma density is proportional to
    //    T**(SHAPE - 1) * EXP(- SCALE * T)
    //
    //  This comment describes the form of the gamma probability density function (PDF),
    //  which is proportional to T raised to the power of (SHAPE - 1) multiplied by
    //  the exponential function of (- SCALE * T).
    //
    //  It indicates the mathematical expression representing the probability density
    //  of a gamma-distributed random variable T, where SHAPE is the shape parameter
    //  and SCALE is the scale parameter of the gamma distribution.
    //
    //**********************************************************************
// 定义一个返回结构为 TupleDDID 的函数 cdfgam_which1，计算累积分布函数的特定值
struct TupleDDID cdfgam_which1(double x, double shape, double scale)
{
    // 声明一个结构体变量 gamret，用于存储函数 cumgam 返回的结果
    struct TupleDD gamret;
    // 如果 x 不大于等于 0，则返回一个标志错误的结构体
    if (!(0 <= x)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1, .d3 = 0.0};
    }
    // 如果 shape 不大于 0，则返回一个标志错误的结构体
    if (!(0 < shape)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }
    // 如果 scale 不大于 0，则返回一个标志错误的结构体
    if (!(0 < scale)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }
    // 计算 x * scale，然后调用 cumgam 函数，将结果存储在 gamret 中
    gamret = cumgam(x * scale, shape);
    // 如果 gamret.d1 大于等于 1.5，则返回一个带有特定标志的结构体
    if (gamret.d1 >= 1.5) {
        return (struct TupleDDID){.d1 = gamret.d1, .d2 = gamret.d2, .i1 = 10, .d3 = 0.0};
    } else {
        // 否则返回一个默认标志的结构体
        return (struct TupleDDID){.d1 = gamret.d1, .d2 = gamret.d2, .i1 = 0, .d3 = 0.0};
    }
}

// 定义一个返回结构为 TupleDID 的函数 cdfgam_which2，计算累积分布函数的特定值
struct TupleDID cdfgam_which2(double p, double q, double shape, double scale)
{
    // 声明一个结构体变量 invret，用于存储函数 gaminv 返回的结果
    struct TupleDI invret;
    // 如果 p 不在 [0, 1] 的范围内，则返回一个标志错误的结构体
    if (!((0 <= p) && (p <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1};
    }
    // 如果 q 不在 (0, 1] 的范围内，则返回一个标志错误的结构体
    if (!((0 < q) && (q <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2};
    }
    // 如果 shape 不大于 0，则返回一个标志错误的结构体
    if (!(0 < shape)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 如果 scale 不大于 0，则返回一个标志错误的结构体
    if (!(0 < scale)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    // 如果条件 ((fabs(p+q)-0.5)-0.5) > 3*spmpar[0] 满足，则返回一个特定的结构体
    if (((fabs(p + q) - 0.5) - 0.5) > 3 * spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3, .d2 = (p + q < 0 ? 0.0 : 1.0)};
    }
    // 调用 gaminv 函数，将结果存储在 invret 中
    invret = gaminv(shape, p, q, -1);
    // 如果 gaminv 返回的结果小于 0，则返回一个带有特定标志的结构体
    if (invret.i1 < 0) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 10, .d2 = 0.0};
    } else {
        // 否则返回一个默认标志的结构体，将 d1 缩放为 invret.d1 / scale
        return (struct TupleDID){.d1 = invret.d1 / scale, .i1 = 0, .d2 = 0.0};
    }
}

// 定义一个返回结构为 TupleDID 的函数 cdfgam_which3，计算累积分布函数的特定值
struct TupleDID cdfgam_which3(double p, double q, double x, double scale)
{
    // 计算 x * scale，并定义一些常量和状态变量
    double xscale = x * scale;
    double tol = 1e-10;
    double atol = 1e-50;
    int qporq = p <= q;
    DinvrState DS = {0};
    DzrorState DZ = {0};

    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.0;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.0;
    // 声明两个结构体变量 gamret 和 ret，用于存储函数返回结果
    struct TupleDD gamret;
    struct TupleDID ret = {0};

    // 如果 p 不在 [0, 1] 的范围内，则返回一个标志错误的结构体
    if (!((0 <= p) && (p <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1};
    }
    // 如果 q 不在 (0, 1] 的范围内，则返回一个标志错误的结构体
    if (!((0 < q) && (q <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2};
    }
    // 如果 x 不大于等于 0，则返回一个标志错误的结构体
    if (!(0 <= x)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 如果 scale 不大于 0，则返回一个标志错误的结构体
    if (!(0 < scale)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    // 如果条件 ((fabs(p+q)-0.5)-0.5) > 3*spmpar[0] 满足，则返回一个特定的结构体
    if (((fabs(p + q) - 0.5) - 0.5) > 3 * spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3, .d2 = (p + q < 0 ? 0.0 : 1.0)};
    }

    // 调用 dinvr 函数，并在状态变量 DS 为 1 时循环执行以下操作
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        // 调用 cumgam 函数，将结果存储在 gamret 中
        gamret = cumgam(xscale, DS.x);
        // 根据 qporq 的值和 gamret 的结果，判断返回哪个特定的结构体
        if (((qporq) && (gamret.d1 > 1.5)) || (!(qporq) && (gamret.d2 > 1.5))) {
            return (struct TupleDID){.d1 = DS.x, .d2 = 0.0, .i1 = 10};
        }
        // 再次调用 dinvr 函数，更新状态变量 DS
        dinvr
    # 如果 DS 的状态为 -1，则执行以下操作
    if (DS.status == -1) {
        # 设置返回值的第一个属性为 DS 的 x 值
        ret.d1 = DS.x;
        # 设置返回值的第二个属性，如果 DS 的 qleft 为真，则为 1；否则为 2
        ret.i1 = (DS.qleft ? 1 : 2);
        # 设置返回值的第三个属性，如果 DS 的 qleft 为真，则为 0.0；否则为 1e100
        ret.d2 = (DS.qleft ? 0.0 : 1e100);
        # 返回设置好的返回值 ret
        return ret;
    } else {
        # 如果 DS 的状态不为 -1，则执行以下操作
        # 设置返回值的第一个属性为 DS 的 x 值
        ret.d1 = DS.x;
        # 返回设置好的返回值 ret
        return ret;
    }
    // 结构体函数，计算负二项分布的累积分布函数
    // 参数：
    //     p - 累积到 S 的负二项分布的概率，范围 [0, 1]
    //     q - 1 - p，范围 (0, 1]
    //     x - 成功之前的失败数上限，范围 [0, +∞)
    //     shape - 成功的形状参数，范围 (0, +∞)
    // 返回值：
    //     结构体 TupleDID，包含三个字段：
    //         .d1 - 计算的结果
    //         .d2 - 预留字段，未使用
    //         .i1 - 错误代码，0 表示成功，负数表示错误码
    struct TupleDID cdfgam_which4(double p, double q, double x, double shape)
    {
        struct TupleDI invret;

        // 检查参数边界，如果不符合要求则返回相应的错误码和空结果
        if (!((0 <= p) && (p <= 1))) {
            return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1};
        }
        if (!((0 < q) && (q <= 1))) {
            return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2};
        }
        if (!(0 <= x)) {
            return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
        }
        if (!(0 < shape)) {
            return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
        }
        
        // 进行进一步的条件检查
        if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
            return (struct TupleDID){.d1 = 0.0, .i1 = 3, .d2 = (p+q < 0 ? 0.0 : 1.0)};
        }

        // 调用 gaminv 函数计算逆 Gamma 分布，将结果存储在 invret 中
        invret = gaminv(shape, p, q, -1);

        // 检查 gaminv 返回的错误码，如果小于 0 则返回相应的错误码和空结果，否则计算并返回结果
        if (invret.i1 < 0) {
            return (struct TupleDID){.d1 = 0.0, .i1 = 10, .d2 = 0.0};
        } else {
            return (struct TupleDID){.d1 = invret.d1/x, .i1 = 0, .d2 = 0.0};
        }
    }
    //              Search range: [0, 1E100]
    //                    DOUBLE PRECISION XN
    //
    //     PR  <--> The probability of success in each binomial trial.
    //              Input range: [0,1].
    //              Search range: [0,1].
    //                    DOUBLE PRECISION PR
    //
    //     OMPR  <--> 1-PR
    //              Input range: [0,1].
    //              Search range: [0,1]
    //              PR + OMPR = 1.0
    //                    DOUBLE PRECISION OMPR
    //
    //     STATUS <-- 0 if calculation completed correctly
    //               -I if input parameter number I is out of range
    //                1 if answer appears to be lower than lowest
    //                  search bound
    //                2 if answer appears to be higher than greatest
    //                  search bound
    //                3 if P + Q .ne. 1
    //                4 if PR + OMPR .ne. 1
    //                    INTEGER STATUS
    //
    //     BOUND <-- Undefined if STATUS is 0
    //
    //               Bound exceeded by parameter number I if STATUS
    //               is negative.
    //
    //               Lower search bound if STATUS is 1.
    //
    //               Upper search bound if STATUS is 2.
    //
    //
    //                              Method
    //
    //
    //     Formula   26.5.26   of   Abramowitz  and  Stegun,  Handbook   of
    //     Mathematical Functions (1966) is used  to  reduce calculation of
    //     the cumulative distribution  function to that of  an  incomplete
    //     beta.
    //
    //     Computation of other parameters involve a search for a value that
    //     produces  the desired  value  of P.   The search relies  on  the
    //     monotinicity of P with the other parameter.
    //
    //
    //**********************************************************************
struct TupleDID cdfnbn_which3(double p, double q, double s, double pr, double ompr)
{
    double tol = 1e-10;  // 定义容差值为 1e-10
    double atol = 1e-50;  // 定义绝对容差值为 1e-50
    int qporq = (p <= q);  // 判断 p 是否小于等于 q，结果存储在 qporq 中
    DinvrState DS = {0};  // 初始化 DinvrState 结构体 DS，全部成员赋值为 0
    DzrorState DZ = {0};  // 初始化 DzrorState 结构体 DZ，全部成员赋值为 0

    DS.small = 0.;  // 设置 DS 结构体的 small 成员为 0.0
    DS.big = 1e100;  // 设置 DS 结构体的 big 成员为 1e100
    DS.absstp = 0.5;  // 设置 DS 结构体的 absstp 成员为 0.5
    DS.relstp = 0.5;  // 设置 DS 结构体的 relstp 成员为 0.5
    DS.stpmul = 5.;  // 设置 DS 结构体的 stpmul 成员为 5.0

    // 初始化 DS 结构体的 x 成员为 5.0
    DS.x = 5.;

    struct TupleDD nbnret;  // 声明 TupleDD 结构体变量 nbnret
    struct TupleDID ret = {0};  // 初始化 TupleDID 结构体 ret，全部成员赋值为 0

    // 如果 p 不在 [0, 1] 范围内，返回错误码 -1
    if (!((0 <= p) && (p <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1};
    }
    // 如果 q 不在 (0, 1] 范围内，返回错误码 -2
    if (!((0 < q) && (q <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2};
    }
    // 如果 s 不大于等于 0，返回错误码 -3
    if (!(0 <= s)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 如果 pr 不在 [0, 1] 范围内，返回错误码 -4
    if (!((0 <= pr) && (pr <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    // 如果 ompr 不在 [0, 1] 范围内，返回错误码 -5
    if (!((0 <= ompr) && (ompr <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -5};
    }
    // 如果 fabs(p+q)-1.0 大于 3 * spmpar[0]，返回错误码 3
    if (((fabs(p+q)-1.0)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3,
                                 .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }
    // 如果 fabs(pr+ompr)-1.0 大于 3 * spmpar[0]，返回错误码 4
    if (((fabs(pr+ompr)-1.0)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 4,
                                 .d2 = (pr+ompr < 0 ? 0.0 : 1.0)};
    }

    // 调用 dinvr 函数，使用 DS 和 DZ 结构体进行计算
    dinvr(&DS, &DZ);

    // 当 DS 结构体的 status 成员为 1 时执行循环
    while (DS.status == 1) {
        // 调用 cumnbn 函数计算结果，并根据 qporq 的值选择计算方式
        nbnret = cumnbn(DS.x, s, pr, ompr);
        DS.fx = (qporq ? nbnret.d1 - p : nbnret.d2 - q);
        // 继续调用 dinvr 函数，更新 DS 结构体状态
        dinvr(&DS, &DZ);
    }

    // 如果 DS 结构体的 status 等于 -1，设置 ret 结构体的各成员值
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e100);
        return ret;
    } else {
        // 否则，设置 ret 结构体的部分成员值
        ret.d1 = DS.x;
        return ret;
    }
}
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    struct TupleDD nbnret;
    struct TupleDID ret = {0};

    // 检查参数 p 是否在 [0, 1] 范围内，否则返回错误码 -1
    if (!((0 <= p) && (p <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1};
    }
    // 检查参数 q 是否在 (0, 1] 范围内，否则返回错误码 -2
    if (!((0 < q) && (q <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2};
    }
    // 检查参数 s 是否非负，否则返回错误码 -3
    if (!(0 <= s)) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3};
    }
    // 检查参数 pr 是否在 [0, 1] 范围内，否则返回错误码 -4
    if (!((0 <= pr) && (pr <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4};
    }
    // 检查参数 ompr 是否在 [0, 1] 范围内，否则返回错误码 -5
    if (!((0 <= ompr) && (ompr <= 1))) {
        return (struct TupleDID){.d1 = 0.0, .d2 = 0.0, .i1 = -5};
    }
    // 检查 p + q 是否满足一定条件，否则返回错误码 3 或 4
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 3,
                                 .d2 = (p+q < 0 ? 0.0 : 1.0)};
    }
    // 检查 pr + ompr 是否满足一定条件，否则返回错误码 3 或 4
    if (((fabs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDID){.d1 = 0.0, .i1 = 4,
                                 .d2 = (pr+ompr < 0 ? 0.0 : 1.0)};
    }
    // 初始化迭代求解函数 dinvr，直到 DS.status 不再为 1
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        // 计算累积负二项分布函数的结果
        nbnret = cumnbn(s, DS.x, pr, ompr);
        // 根据 qporq 的值更新 DS.fx
        DS.fx = (qporq ? nbnret.d1 - p : nbnret.d2 - q);
        // 继续迭代求解函数 dinvr
        dinvr(&DS, &DZ);
    }

    // 根据 DS.status 的值返回不同的结构体 ret
    if (DS.status == -1) {
        // DS.status 为 -1 时，设置 ret 结构体的值
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e100);
        return ret;
    } else {
        // DS.status 不为 -1 时，设置 ret 结构体的值
        ret.d1 = DS.x;
        return ret;
    }
}


// 结构体函数 cdfnbn_which4，计算负二项分布的累积分布函数
struct TupleDDID cdfnbn_which4(double p, double q, double s, double xn)
{
    // 声明局部变量 pr, ompr
    double pr, ompr;
    // 定义误差容限 tol 和极小绝对误差 atol
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q 的布尔值，用于后续判断
    int qporq = (p <= q);
    // 初始化 DZ 结构体
    DzrorState DZ = {0};

    // 初始化 DZ 结构体的各个成员变量
    DZ.xlo = 0.;
    DZ.xhi = 1.;
    DZ.atol = atol;
    DZ.rtol = tol;
    DZ.x = 0.;
    DZ.b = 0.;

    // 声明结构体变量 nbnret 和 ret，分别用于存储函数 cumnbn 的返回值
    struct TupleDD nbnret;
    struct TupleDDID ret = {0};

    // 检查 p 的取值范围，如果不在 [0, 1] 内则返回特定结构体
    if (!((0 <= p) && (p <= 1))) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1,
                                  .d3 = (p > 0 ? 0.0 : 1.0)};
    }
    // 检查 q 的取值范围，如果不在 (0, 1] 内则返回特定结构体
    if (!((0 < q) && (q <= 1))) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2,
                                  .d3 = (1 > 0 ? 0.0 : 1.0)};
    }
    // 检查 s 的取值范围，如果小于 0 则返回特定结构体
    if (!(0 <= s)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }
    // 检查 xn 的取值范围，如果小于 0 则返回特定结构体
    if (!(0 <= xn)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -4, .d3 = 0.0};
    }
    // 检查 |p + q - 0.5 - 0.5| 是否大于 3*spmpar[0]，如果是则返回特定结构体
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = 4,
                                 .d3 = (p+q < 0 ? 0.0 : 1.0)};
    }

    // 根据 qporq 的值选择不同的处理路径
    if (qporq) {
        // 调用 dzror 函数，进行根据 DZ 结构体处理
        dzror(&DZ);
        // 计算 ompr 的值
        ompr = 1.0 - DZ.x;
        // 循环直到 DZ 的状态不为 1
        while (DZ.status == 1) {
            // 调用 cumnbn 函数，计算 nbnret 的值
            nbnret = cumnbn(s, xn, DZ.x, ompr);
            // 更新 DZ.fx 的值
            DZ.fx = nbnret.d1 - p;
            // 再次调用 dzror 函数
            dzror(&DZ);
            // 更新 ompr 的值
            ompr = 1.0 - DZ.x;
        }
        // 将 pr 设置为 DZ.x 的值
        pr = DZ.x;
    } else {
        // 调用 dzror 函数，进行根据 DZ 结构体处理
        dzror(&DZ);
        // 计算 pr 的值
        pr = 1.0 - DZ.x;
        // 循环直到 DZ 的状态不为 1
        while (DZ.status == 1) {
            // 调用 cumnbn 函数，计算 nbnret 的值
            nbnret = cumnbn(s, xn, pr, DZ.x);
            // 更新 DZ.fx 的值
            DZ.fx = nbnret.d2 - q;
            // 再次调用 dzror 函数
            dzror(&DZ);
            // 更新 pr 的值
            pr = 1.0 - DZ.x;
        }
        // 将 ompr 设置为 DZ.x 的值
        ompr = DZ.x;
    }

    // 根据 DZ 的状态值进行返回不同的结构体
    if (DZ.status == -1) {
        ret.d1 = pr;
        ret.d2 = ompr;
        ret.i1 = (DZ.qleft ? 1 : 2);
        ret.d3 = (DZ.qleft ? 0 : 1);
        return ret;
    } else {
        ret.d1 = pr;
        ret.d2 = ompr;
        return ret;
    }
}


// Cumulative Distribution Function
// NORmal distribution
//
//
// Function
//
//
// Calculates any one parameter of the normal
// distribution given values for the others.
//
//
// Arguments
//
//
// WHICH  --> Integer indicating  which of the  next  parameter
// values is to be calculated using values  of the others.
// Legal range: 1..4
//               iwhich = 1 : Calculate P and Q from X,MEAN and SD
//               iwhich = 2 : Calculate X from P,Q,MEAN and SD
//               iwhich = 3 : Calculate MEAN from P,Q,X and SD
//               iwhich = 4 : Calculate SD from P,Q,X and MEAN
//                    INTEGER WHICH
//
// P <--> The integral from -infinity to X of the normal density.
//        Input range: (0,1].
//        DOUBLE PRECISION P
//
// Q <--> 1-P.
//        Input range: (0, 1].
//        P + Q = 1.0.
    //                    DOUBLE PRECISION Q
    //
    //     X < --> Upper limit of integration of the normal-density.
    //             Input range: ( -infinity, +infinity)
    //                    DOUBLE PRECISION X
    //
    //     MEAN <--> The mean of the normal density.
    //               Input range: (-infinity, +infinity)
    //                    DOUBLE PRECISION MEAN
    //
    //     SD <--> Standard Deviation of the normal density.
    //             Input range: (0, +infinity).
    //                    DOUBLE PRECISION SD
    //
    //     STATUS <-- 0 if calculation completed correctly
    //               -I if input parameter number I is out of range
    //                1 if answer appears to be lower than lowest
    //                  search bound
    //                2 if answer appears to be higher than greatest
    //                  search bound
    //                3 if P + Q .ne. 1
    //                    INTEGER STATUS
    //
    //     BOUND <-- Undefined if STATUS is 0
    //
    //               Bound exceeded by parameter number I if STATUS
    //               is negative.
    //
    //               Lower search bound if STATUS is 1.
    //
    //               Upper search bound if STATUS is 2.
    //
    //
    //                              Method
    //
    //
    //
    //
    //     A slightly modified version of ANORM from
    //
    //     Cody, W.D. (1993). "ALGORITHM 715: SPECFUN - A Portabel FORTRAN
    //     Package of Special Function Routines and Test Drivers"
    //     acm Transactions on Mathematical Software. 19, 22-32.
    //
    //     is used to calculate the cumulative standard normal distribution.
    //
    //     The rational functions from pages  90-95  of Kennedy and Gentle,
    //     Statistical  Computing,  Marcel  Dekker, NY,  1980 are  used  as
    //     starting values to Newton's Iterations which compute the inverse
    //     standard normal.  Therefore no  searches  are necessary for  any
    //     parameter.
    //
    //     For X < -15, the asymptotic expansion for the normal is used  as
    //     the starting value in finding the inverse standard normal.
    //     This is formula 26.2.12 of Abramowitz and Stegun.
    //
    //
    //                              Note
    //
    //
    //      The normal density is proportional to
    //      exp( - 0.5 * (( X - MEAN)/SD)**2)
    //
    //
    //**********************************************************************
// Calculate P and Q from S and XLAM for the Poisson distribution
// Legal range for WHICH: 1..3
// iwhich = 1 : Calculate P and Q from S and XLAM
// iwhich = 2 : Calculate S from P, Q, and XLAM
// iwhich = 3 : Calculate XLAM from P, Q, and S
// Arguments:
//    INTEGER WHICH
//        Integer indicating which argument value is to be calculated from the others.
//    DOUBLE PRECISION P
//        The cumulation from 0 to S of the Poisson density. Input range: [0, 1].
//    DOUBLE PRECISION Q
//        1 - P. Input range: (0, 1]. P + Q = 1.0.
//    DOUBLE PRECISION S
//        Upper limit of cumulation of the Poisson. Input range: [0, +infinity).
//        Search range: [0, 1E100]
//    DOUBLE PRECISION XLAM
//        Mean of the Poisson distribution. Input range: [0, +infinity).
//        Search range: [0, 1E100]
// Returns:
//    INTEGER STATUS
//        0 if calculation completed correctly
//        -I if input parameter number I is out of range
//        1 if answer appears to be lower than lowest search bound
//        2 if answer appears to be higher than greatest search bound
//        3 if P + Q != 1
//    Undefined if STATUS is 0

struct TupleDDID cdfnor_which1(double x, double mean, double sd)
{
    // Check if standard deviation is positive
    if (!(sd > 0.0)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3, .d3 = 0.0};
    }
    // Calculate z-score
    double z = (x - mean) / sd;
    // Calculate cumulative normal distribution
    struct TupleDD res = cumnor(z);
    return (struct TupleDDID){.d1 = res.d1, .d2 = res.d2, .i1 = 0, .d3 = 0.0};
}

struct TupleDID cdfnor_which2(double p, double q, double mean, double sd)
{
    // Check if standard deviation is positive
    if (!(sd > 0.0)) {
        return (struct TupleDID){.d1 = 0.0, .i1 = -4, .d2 = 0.0};
    }
    // Calculate z-score
    double z = dinvnr(p, q);
    // Calculate mean-adjusted value
    return (struct TupleDID){.d1 = sd * z + mean, .i1 = 0, .d2 = 0.0};
}

struct TupleDID cdfnor_which3(double p, double q, double x, double sd)
{
    // Check if standard deviation is positive
    if (!(sd > 0.0)) {
        return (struct TupleDID){.d1 = 0.0, .i1 = -4, .d2 = 0.0};
    }
    // Calculate z-score
    double z = dinvnr(p, q);
    // Calculate x adjusted by z-score
    return (struct TupleDID){.d1 = x - sd * z, .i1 = 0, .d2 = 0.0};
}

struct TupleDID cdfnor_which4(double p, double q, double x, double mean)
{
    // Calculate z-score
    double z = dinvnr(p, q);
    // Calculate x standardized by mean
    return (struct TupleDID){.d1 = (x - mean) / z, .i1 = 0, .d2 = 0.0};
}
    //
    //               根据参数的情况，该函数用于计算不同的边界值。
    //               如果 STATUS 为负数，根据参数编号 I，表示超出了边界。
    //               如果 STATUS 为 1，表示下限搜索边界。
    //               如果 STATUS 为 2，表示上限搜索边界。
    //
    //
    //                              方法
    //
    //
    //     该函数使用 Abramowitz 和 Stegun 的《数学函数手册》（1966年）中的公式
    //     26.4.21 来将累积分布函数的计算简化为计算卡方分布，进而计算不完全伽玛函数。
    //
    //     直接计算累积分布函数（P）的值。
    //     计算其他参数涉及搜索使得 P 达到期望值的参数值。搜索依赖于 P 随另一参数的单调性。
    //
    //
    //**********************************************************************
struct TupleDDID cdfpoi_which1(double s, double xlam)
{
    // 检查 s 是否为非负数，若不是则返回一个特定的结构体
    if (!(s >= 0.0)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1, .d3 = 0.0};
    }
    // 检查 xlam 是否为非负数，若不是则返回一个特定的结构体
    if (!(xlam >= 0.0)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }
    // 调用 cumpoi 函数计算累积泊松分布，并将结果转为 TupleDD 结构体
    struct TupleDD res = cumpoi(s, xlam);
    // 将 TupleDD 结构体的部分值组成 TupleDDID 结构体并返回
    return (struct TupleDDID){.d1 = res.d1, .d2 = res.d2, .i1 = 0, .d3 = 0.0};
}

struct TupleDID cdfpoi_which2(double p, double q, double xlam)
{
    // 定义误差容限和绝对误差容限
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否在 [0, 1] 区间内，若不是则返回特定的结构体
    int qporq = (p <= q);
    DinvrState DS = {0};
    DzrorState DZ = {0};

    // 初始化数值求解状态结构体 DS
    DS.small = 0.;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.0;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.0;
    struct TupleDD poiret;
    struct TupleDID ret = {0};

    // 检查 p 是否在 [0, 1] 区间内，若不是则返回特定的结构体
    if (!((0 <= p) && (p <= 1))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 是否在 (0, 1] 区间内，若不是则返回特定的结构体
    if (!((0 < q) && (q <= 1))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 xlam 是否为非负数，若不是则返回特定的结构体
    if (!(xlam >= 0.0)) {
        ret.i1 = -3;
        return ret;
    }
    // 判断条件是否满足，若满足则返回特定的结构体
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        ret.i1 = 3;
        ret.d2 = (p+q < 0 ? 0.0 : 1.0);
        return ret;
    }

    // 当 xlam 较小时和 p 较小时，返回特定的结构体
    if ((xlam < 0.01) && (p < 0.975)) {
        // 对于足够小的 xlam 和 p，结果为 0
        return ret;
    }

    // 使用数值求解函数 dinvr 和循环求解，直至求解状态不再为 1
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        // 计算累积泊松分布并更新 DS.fx
        poiret = cumpoi(DS.x, xlam);
        DS.fx = (qporq ? poiret.d1 - p : poiret.d2 - q);
        dinvr(&DS, &DZ);
    }

    // 根据 DS 状态返回不同的结构体
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e100);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}

struct TupleDID cdfpoi_which3(double p, double q, double s)
{
    // 定义误差容限和绝对误差容限
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否在 [0, 1] 区间内，若不是则返回特定的结构体
    int qporq = (p <= q);
    DinvrState DS = {0};
    DzrorState DZ = {0};

    // 初始化数值求解状态结构体 DS
    DS.small = 0.0;
    DS.big = 1e300;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.0;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.0;
    struct TupleDD poiret;
    struct TupleDID ret = {0};

    // 检查 p 是否在 [0, 1] 区间内，若不是则返回特定的结构体
    if (!((0 <= p) && (p <= 1))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 q 是否在 (0, 1] 区间内，若不是则返回特定的结构体
    if (!((0 < q) && (q <= 1))) {
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 检查 s 是否为非负数，若不是则返回特定的结构体
    if (!(s >= 0.0)) {
        ret.i1 = -3;
        return ret;
    }
    // 判断条件是否满足，若满足则返回特定的结构体
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        ret.i1 = 3;
        ret.d2 = (p+q < 0 ? 0.0 : 1.0);
        return ret;
    }

    // 使用数值求解函数 dinvr 和循环求解，直至求解状态不再为 1
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        // 计算累积泊松分布并更新 DS.fx
        poiret = cumpoi(s, DS.x);
        DS.fx = (qporq ? poiret.d1 - p : poiret.d2 - q);
        dinvr(&DS, &DZ);
    }

    // 根据 DS 状态返回不同的结构体
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0.0 : 1e100);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}
    } else {
        // 如果条件不满足，则执行这个代码块
        // 将 DS.x 赋值给 ret.d1
        ret.d1 = DS.x;
        // 返回 ret 结构体
        return ret;
    }
// 结构体 TupleDDID 表示函数 cdft_which1 的返回类型，包含四个字段：d1, d2, i1, d3
struct TupleDDID cdft_which1(double t, double df)
{
    // 如果自由度 df 不大于 0，则返回一个结构体，其中 d1, d2, d3 置为 0.0，i1 置为 -2
    if (!(df > 0.0)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }
    # 调用 cumt 函数计算结果，并将返回的结构体 TupleDD 存储在 res 中
    struct TupleDD res = cumt(t, df);
    # 创建一个新的结构体 TupleDDID，初始化其中的字段值
    return (struct TupleDDID){.d1 = res.d1, .d2 = res.d2, .i1 = 0, .d3 = 0.0};
}

// 结构体定义，表示返回的元组，包含一个整数和一个双精度浮点数
struct TupleDID cdft_which2(double p, double q, double df)
{
    // 定义公差和绝对公差的值
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q，结果保存在 qporq 中
    int qporq = (p <= q);
    // 定义并初始化 DinvrState 结构体
    DinvrState DS = {0};
    // 定义并初始化 DzrorState 结构体
    DzrorState DZ = {0};

    // 设置 DS 结构体的小值和大值
    DS.small = -1.e100;
    DS.big = 1.e100;
    // 设置 DS 结构体的绝对步长和相对步长
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    // 设置 DS 结构体的步长倍数
    DS.stpmul = 5.;
    // 设置 DS 结构体的绝对公差和相对公差
    DS.abstol = atol;
    DS.reltol = tol;
    // 调用 dt1 函数计算 DS 结构体的 x 值
    DS.x = dt1(p, q, df);
    // 定义返回的元组 tret 和 ret，初始化为 0
    struct TupleDD tret;
    struct TupleDID ret = {0};

    // 如果 p 不在 [0, 1] 范围内
    if (!((0 <= p) && (p <= 1))) {
        // 返回错误码 -1 和一个浮点数结果
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 如果 q 不在 [0, 1] 范围内
    if (!((0 <= q) && (q <= 1))) {
        // 返回错误码 -2 和一个浮点数结果
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 如果 df 不大于 0
    if (!(df > 0.0)) {
        // 返回错误码 -3
        ret.i1 = -3;
        return ret;
    }
    // 如果 fabs(p + q) - 0.5 - 0.5 大于 3*spmpar[0]
    if (((fabs(p + q) - 0.5) - 0.5) > 3 * spmpar[0]) {
        // 返回错误码 3 和一个浮点数结果
        ret.i1 = 3;
        ret.d2 = (p + q < 0 ? 0.0 : 1.0);
        return ret;
    }

    // 调用 dinvr 函数，更新 DS 和 DZ 结构体
    dinvr(&DS, &DZ);
    // 循环直到 DS 状态不为 1
    while (DS.status == 1) {
        // 调用 cumt 函数计算 tret
        tret = cumt(DS.x, df);
        // 更新 DS 结构体的 fx 值
        DS.fx = (qporq ? tret.d1 - p : tret.d2 - q);
        // 再次调用 dinvr 函数，更新 DS 和 DZ 结构体
        dinvr(&DS, &DZ);
    }

    // 如果 DS 状态为 -1
    if (DS.status == -1) {
        // 返回 DS 结构体的 x 值、一个整数和一个浮点数结果
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? -1e100 : 1e100);
        return ret;
    } else {
        // 否则返回 DS 结构体的 x 值和一个整数结果
        ret.d1 = DS.x;
        return ret;
    }
}

// 结构体定义，表示返回的元组，包含一个整数和一个双精度浮点数
struct TupleDID cdft_which3(double p, double q, double t)
{
    // 定义公差和绝对公差的值
    double tol = 1e-10;
    double atol = 1e-50;
    // 判断 p 是否小于等于 q，结果保存在 qporq 中
    int qporq = (p <= q);
    // 定义并初始化 DinvrState 结构体
    DinvrState DS = {0};
    // 定义并初始化 DzrorState 结构体
    DzrorState DZ = {0};
    // 定义返回的元组 tret 和 ret，初始化为 0
    struct TupleDD tret;
    struct TupleDID ret = {0};

    // 设置 DS 结构体的小值和大值
    DS.small = 1e-100;
    DS.big = 1e10;
    // 设置 DS 结构体的绝对步长和相对步长
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    // 设置 DS 结构体的步长倍数
    DS.stpmul = 5.0;
    // 设置 DS 结构体的绝对公差和相对公差
    DS.abstol = atol;
    DS.reltol = tol;
    // 设置 DS 结构体的 x 值
    DS.x = 5.0;

    // 如果 p 不在 [0, 1] 范围内
    if (!((0 <= p) && (p <= 1))) {
        // 返回错误码 -1 和一个浮点数结果
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 如果 q 不在 [0, 1] 范围内
    if (!((0 <= q) && (q <= 1))) {
        // 返回错误码 -2 和一个浮点数结果
        ret.i1 = -2;
        ret.d2 = (!(q > 0.0) ? 0.0 : 1.0);
        return ret;
    }
    // 如果 fabs(p + q) - 0.5 - 0.5 大于 3*spmpar[0]
    if (((fabs(p + q) - 0.5) - 0.5) > 3 * spmpar[0]) {
        // 返回错误码 3 和一个浮点数结果
        ret.i1 = 3;
        ret.d2 = (p + q < 0 ? 0.0 : 1.0);
        return ret;
    }

    // 调用 dinvr 函数，更新 DS 和 DZ 结构体
    dinvr(&DS, &DZ);
    // 循环直到 DS 状态不为 1
    while (DS.status == 1) {
        // 调用 cumt 函数计算 tret
        tret = cumt(t, DS.x);
        // 更新 DS 结构体的 fx 值
        DS.fx = (qporq ? tret.d1 - p : tret.d2 - q);
        // 再次调用 dinvr 函数，更新 DS 和 DZ 结构体
        dinvr(&DS, &DZ);
    }
    // 如果 DS 状态为 -1
    if (DS.status == -1) {
        // 返回 DS 结构体的 x 值、一个整数和一个浮点数结果
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? -1e100 : 1e10);
        return ret;
    } else {
        // 否则返回 DS 结构体的 x 值和一个整数结果
        ret.d1 = DS.x;
        return ret;
    }
}

// 注释：累积分布函数
// 注释：非中心 t 分布
// 注释：函数
// 注释：根据其他参数计算非中心 t 分布的某个参数。
// 注释：参数
// 注释：WHICH --> 整数，指示要从其他参数计算的参数。
// 注释：合法范围：1..3
    //               iwhich = 1 : Calculate P and Q from T,DF,PNONC
    //               iwhich = 2 : Calculate T from P,Q,DF,PNONC
    //               iwhich = 3 : Calculate DF from P,Q,T
    //               iwhich = 4 : Calculate PNONC from P,Q,DF,T
    //                    INTEGER WHICH
    //
    //        P <--> The integral from -infinity to t of the noncentral t-density
    //              Input range: (0,1].
    //                    DOUBLE PRECISION P
    //
    //     Q <--> 1-P.
    //            Input range: (0, 1].
    //            P + Q = 1.0.
    //                    DOUBLE PRECISION Q
    //
    //        T <--> Upper limit of integration of the noncentral t-density.
    //               Input range: ( -infinity, +infinity).
    //               Search range: [ -1E100, 1E100 ]
    //                    DOUBLE PRECISION T
    //
    //        DF <--> Degrees of freedom of the noncentral t-distribution.
    //                Input range: (0 , +infinity).
    //                Search range: [1e-100, 1E10]
    //                    DOUBLE PRECISION DF
    //
    //     PNONC <--> Noncentrality parameter of the noncentral t-distribution.
    //                Input range: [-1e6, 1E6].
    //
    //     STATUS <-- 0 if calculation completed correctly
    //               -I if input parameter number I is out of range
    //                1 if answer appears to be lower than lowest
    //                  search bound
    //                2 if answer appears to be higher than greatest
    //                  search bound
    //                3 if P + Q .ne. 1
    //                    INTEGER STATUS
    //
    //     BOUND <-- Undefined if STATUS is 0
    //
    //               Bound exceeded by parameter number I if STATUS
    //               is negative.
    //
    //               Lower search bound if STATUS is 1.
    //
    //               Upper search bound if STATUS is 2.
    //
    //                                Method
    //
    //     Upper tail    of  the  cumulative  noncentral t is calculated using
    //     formulae  from page 532  of Johnson, Kotz,  Balakrishnan, Continuous
    //     Univariate Distributions, Vol 2, 2nd Edition.  Wiley (1995)
    //
    //     Computation of other parameters involve a search for a value that
    //     produces  the desired  value  of P.   The search relies  on  the
    //     monotonicity of P with the other parameter.
    //
    //***********************************************************************
struct TupleDDID cdftnc_which1(double t, double df, double pnonc)
{
    // 检查 t 是否是 NaN，若是则返回特定的错误元组
    if (!(t == t)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -1, .d3 = 0.0};
    }
    // 检查 df 是否大于 0，若不是则返回特定的错误元组
    if (!(df > 0.0)) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -2, .d3 = 0.0};
    }

    // 将 df 限制在 1.e10 内
    df = fmin(df, 1.e10);
    // 将 t 限制在 [-spmpar[2], spmpar[2]] 范围内
    t = fmax(fmin(t, spmpar[2]), -spmpar[2]);

    // 检查 pnonc 是否在 [-1e6, 1e6] 范围内，若不是则返回特定的错误元组
    if (!((-1.e6 <= pnonc) && (pnonc <= 1.e6))) {
        return (struct TupleDDID){.d1 = 0.0, .d2 = 0.0, .i1 = -3,
         .d3 = (pnonc > -1e6 ? 1.0e6 : -1.0e6)};
    }

    // 调用 cumtnc 函数计算累积 t 分布的函数值，并返回结果元组的部分
    struct TupleDD res = cumtnc(t, df, pnonc);
    // 返回符合 cdftnc_which1 函数签名的结果元组
    return (struct TupleDDID){.d1 = res.d1, .d2 = res.d2, .i1 = 0, .d3 = 0.0};
}


struct TupleDID cdftnc_which2(double p, double q, double df, double pnonc)
{
    // 设置数值容差和状态结构体
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0};
    DzrorState DZ = {0};

    // 设置 DS 结构体的初始参数
    DS.small = -1.e100;
    DS.big = 1.e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;
    struct TupleDD tncret;
    struct TupleDID ret = {0};

    // 检查 p 是否在 [0, 1 - 1e-16] 范围内，若不是则返回特定的错误元组
    if (!((0 <= p) && (p <= (1. - 1e-16)))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : (1. - 1e-16));
        return ret;
    }

    // 检查 df 是否大于 0，若不是则返回特定的错误元组
    if (!(df > 0.)) {
        ret.i1 = -3;
        return ret;
    }
    // 将 df 限制在 1.e10 内
    df = fmin(df, 1.e10);

    // 检查 pnonc 是否在 [-1e6, 1e6] 范围内，若不是则返回特定的错误元组
    if (!((-1e6 <= pnonc) && (pnonc <= 1e6))) {
        ret.i1 = -4;
        ret.d2 = (pnonc > -1e6 ? 1e6 : -1e6);
        return ret;
    }

    // 检查是否满足附加条件，若不满足则返回特定的错误元组
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        ret.i1 = 3;
        ret.d2 = (p+q < 0 ? 0.0 : 1.0);
        return ret;
    }

    // 调用 dinvr 函数求解 DS 结构体中的 x 值，用于累积 t 分布的逆函数计算
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        // 调用 cumtnc 函数计算累积 t 分布的函数值，并更新 DS 结构体中的 fx 值
        tncret = cumtnc(DS.x, df, pnonc);
        DS.fx = tncret.d1 - p;
        // 继续求解 DS 结构体中的 x 值，直至满足退出条件
        dinvr(&DS, &DZ);
    }
    // 根据 DS 结构体的最终状态返回结果元组
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? -1e100 : 1e100);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}


struct TupleDID cdftnc_which3(double p, double q, double t, double pnonc)
{
    // 设置数值容差和状态结构体
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0};
    DzrorState DZ = {0};
    struct TupleDD tncret;
    struct TupleDID ret = {0};

    // 设置 DS 结构体的初始参数
    DS.small = 1e-100;
    DS.big = 1.e10;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    // 检查 p 是否在 [0, 1 - 1e-16] 范围内，若不是则返回特定的错误元组
    if (!((0 <= p) && (p <= (1. - 1e-16)))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : (1. - 1e-16));
        return ret;
    }

    // 将 t 限制在 [-spmpar[2], spmpar[2]] 范围内
    t = fmax(fmin(t, spmpar[2]), -spmpar[2]);
    // 检查 t 是否是 NaN，若是则返回特定的错误元组
    if (!(t == t)) {
        ret.i1 = -3;
        return ret;
    }

    // 检查 pnonc 是否在 [-1e6, 1e6] 范围内，若不是则返回特定的错误元组
    if (!((-1e6 <= pnonc) && (pnonc <= 1e6))) {
        ret.i1 = -4;
        ret.d2 = (pnonc > -1e6 ? 1e6 : -1e6);
        return ret;
    }

    // 检查是否满足附加条件，若不满足则返回特定的错误元组
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        ret.i1 = 3;
        ret.d2 = (p+q < 0 ? 0.0 : 1.0);
        return ret;
    }

    // 调用 dinvr 函数求解 DS 结构体中的 x 值，用于累积 t 分布的逆函数计算
    dinvr(&DS, &DZ);
    # 当 DS 的状态为 1 时执行循环
    while (DS.status == 1) {
        # 调用 cumtnc 函数计算累积概率，返回结果包含在 tncret 中
        tncret = cumtnc(t, DS.x, pnonc);
        # 计算 DS.fx 的值，作为 tncret.d1 减去 p 的结果
        DS.fx = tncret.d1 - p;
        # 调用 dinvr 函数，更新 DS 和 DZ 的值
        dinvr(&DS, &DZ);
    }
    # 如果 DS 的状态为 -1，则执行以下语句块
    if (DS.status == -1) {
        # 设置 ret 的成员变量 d1 为 DS.x 的值
        ret.d1 = DS.x;
        # 设置 ret 的成员变量 i1，如果 DS.qleft 为真则为 1，否则为 2
        ret.i1 = (DS.qleft ? 1 : 2);
        # 设置 ret 的成员变量 d2，如果 DS.qleft 为真则为 -1e100，否则为 1e100
        ret.d2 = (DS.qleft ? -1e100 : 1e100);
        # 返回 ret 结构体
        return ret;
    } else {
        # 如果 DS 的状态不为 -1，则执行以下语句块
        # 设置 ret 的成员变量 d1 为 DS.x 的值
        ret.d1 = DS.x;
        # 返回 ret 结构体
        return ret;
    }
struct TupleDID cdftnc_which4(double p, double q, double t, double df)
{
    // 定义容差和绝对容差
    double tol = 1e-8;
    double atol = 1e-50;
    
    // 初始化状态结构体
    DinvrState DS = {0};
    DzrorState DZ = {0};
    
    // 定义返回的结构体
    struct TupleDD tncret;
    struct TupleDID ret = {0};

    // 设置状态结构体的参数
    DS.small = -1.e6;
    DS.big = 1.e6;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    // 检查参数 p 的范围
    if (!((0 <= p) && (p <= (1. - 1e-16)))) {
        ret.i1 = -1;
        ret.d2 = (!(p > 0.0) ? 0.0 : (1. - 1e-16));
        return ret;
    }

    // 检查 t 是否为 NaN
    if (!(t == t)) {
        ret.i1 = -3;
        return ret;
    }

    // 检查 df 是否大于 0
    if (!(df > 0.)) {
        ret.i1 = -4;
        return ret;
    }

    // 检查 p + q 的范围是否超过阈值
    if (((fabs(p+q)-0.5)-0.5) > 3*spmpar[0]) {
        ret.i1 = 3;
        ret.d2 = (p+q < 0 ? 0.0 : 1.0);
        return ret;
    }

    // 约束 t 的范围
    t = fmax(fmin(t, spmpar[2]), -spmpar[2]);

    // 约束 df 的范围
    df = fmin(df, 1.e10);

    // 调用数值求逆和寻根函数
    dinvr(&DS, &DZ);
    while (DS.status == 1) {
        // 计算累积 t 分布的函数值
        tncret = cumtnc(t, df, DS.x);
        DS.fx = tncret.d1 - p;
        dinvr(&DS, &DZ);
    }

    // 根据寻根函数的状态返回结果
    if (DS.status == -1) {
        ret.d1 = DS.x;
        ret.i1 = (DS.qleft ? 1 : 2);
        ret.d2 = (DS.qleft ? 0 : 1e6);
        return ret;
    } else {
        ret.d1 = DS.x;
        return ret;
    }
}
    # 如果 x 小于等于 0.0，则返回一个结构体 TupleDD，其第一个元素为 0.0，第二个元素为 1.0
    if (x <= 0.0) {
        return (struct TupleDD){.d1 = 0.0, .d2 = 1.0};
    }
    # 如果 y 小于等于 0.0，则返回一个结构体 TupleDD，其第一个元素为 1.0，第二个元素为 0.0
    if (y <= 0.0) {
        return (struct TupleDD){.d1 = 1.0, .d2 = 0.0};
    }
    # 调用函数 bratio(a, b, x, y)，返回结构体 TupleDDI，并将其第一个和第二个元素分别赋值给新的结构体 TupleDD 的第一个和第二个元素
    struct TupleDDI res = bratio(a, b, x, y);
    # 返回一个结构体 TupleDD，其第一个元素为 res 结构体的第一个元素，第二个元素为 res 结构体的第二个元素
    return (struct TupleDD){.d1 = res.d1, .d2 = res.d2};
struct TupleDD cumbin(double s, double xn, double pr, double ompr)
{
    // CUmulative BINomial distribution
    //
    //
    // Function:
    //
    //
    // Returns the probability of 0 to S successes in XN binomial
    // trials, each of which has a probability of success, PBIN.
    //
    //
    // Arguments:
    //
    //
    // S    --> The upper limit of cumulation of the binomial distribution.
    //           S is DOUBLE PRECISION
    //
    // XN   --> The number of binomial trials.
    //           XN is DOUBLE PRECISION
    //
    // PBIN --> The probability of success in each binomial trial.
    //           PBIN is DOUBLE PRECISION
    //
    // OMPR --> 1 - PBIN
    //           OMPR is DOUBLE PRECISION
    //
    // CUM  <-- Cumulative binomial distribution.
    //           CUM is DOUBLE PRECISION
    //
    // CCUM <-- Compliment of Cumulative binomial distribution.
    //           CCUM is DOUBLE PRECISION
    //
    //
    // Method:
    //
    //
    // Formula 26.5.24 of Abramowitz and Stegun, Handbook of
    // Mathematical Functions (1966) is used to reduce the binomial
    // distribution to the cumulative beta distribution.

    double cum, ccum;
    // Check if S is not less than XN
    if (!(s < xn)) {
        // If S is not less than XN, return (1.0, 0.0) as TupleDD
        return (struct TupleDD){.d1 = 1.0, .d2 = 0.0};
        cum = 1.0;  // This line is never reached due to the early return
        ccum = 0.0; // This line is never reached due to the early return
    } else {
        // Otherwise, calculate cumulative beta distribution using cumbet function
        struct TupleDD res = cumbet(pr, ompr, s + 1.0, xn - s);
        // Swap result and return as TupleDD
        return (struct TupleDD){.d1 = res.d2, .d2 = res.d1};
    }
}
    // 调用不完全伽马函数（CUMGAM）
    //
    // 返回不完全伽马函数的计算结果，参数为 0.5*x 和 0.5*df
    return cumgam(0.5*x, 0.5*df);
    // 如果 x 不大于 0，则返回一个结构体，其中 d1 为 0.0，d2 为 1.0
    if (!(x > 0.)) {
        return (struct TupleDD){.d1 = 0.0, .d2 = 1.0};
    }
    // 如果 pnonc 不大于 1e-10，则调用 cumchi 函数并返回其结果
    if (!(pnonc > 1e-10)) {
        return cumchi(x, df);
    }
    // 计算 xnonc 为 pnonc 的一半
    xnonc = pnonc/2.0;
    // 将 xnonc 转换为整数类型，并在其为零时设置为 1
    icent = (int)xnonc;
    if (icent == 0) {
        icent = 1;
    }
    // 计算 chid2 为 x 的一半
    chid2 = x / 2.0;

    // 计算对数阶乘 lfact
    lfact = alngam(icent + 1);
    // 计算 lcntwt 和 centwt
    lcntwt = -xnonc + icent*log(xnonc) - lfact;
    centwt = exp(lcntwt);

    // 调用 cumchi 函数计算累积分布，并将结果存储在 res 结构体中
    struct TupleDD res = cumchi(x, df + 2.*icent);
    pcent = res.d1;
    // 计算 dfd2
    dfd2 = (df + 2.*icent)/2.;
    // 计算对数阶乘 lfact
    lfact = alngam(1. + dfd2);
    // 计算 lcntaj
    lcntaj = dfd2*log(chid2) - chid2 - lfact;
    centaj = exp(lcntaj);
    # 计算 lcntaj 的指数，赋值给 centaj

    ssum = centwt*pcent;
    # 计算 centwt 和 pcent 的乘积，赋值给 ssum

    sumadj = 0.;
    # 初始化 sumadj 为 0

    adj = centaj;
    # 将 centaj 的值赋给 adj

    wt = centwt;
    # 将 centwt 的值赋给 wt

    i = icent;
    # 将 icent 的值赋给 i

    while (1) {
        dfd2 = (df + 2.*i)/2.;
        # 计算 df + 2*i 的结果除以 2，赋值给 dfd2

        adj *= dfd2 / chid2;
        # 将 adj 乘以 dfd2/chid2 的结果，更新 adj

        sumadj += adj;
        # 将 adj 加到 sumadj 上

        pterm = pcent + sumadj;
        # 将 pcent 和 sumadj 的和赋给 pterm

        wt *= (i / xnonc);
        # 将 wt 乘以 i/xnonc 的结果，更新 wt

        term = wt*pterm;
        # 计算 wt 和 pterm 的乘积，赋值给 term

        ssum += term;
        # 将 term 加到 ssum 上

        i -= 1;
        # 将 i 减 1

        if ((!((ssum >= abstol) && (term >= eps*ssum))) || (i == 0)) { break; }
        # 如果 ssum 大于等于 abstol 并且 term 大于等于 eps*ssum，或者 i 等于 0，则退出循环
    }

    sumadj = centaj;
    # 将 centaj 的值赋给 sumadj

    adj = centaj;
    # 将 centaj 的值赋给 adj

    wt = centwt;
    # 将 centwt 的值赋给 wt

    i = icent;
    # 将 icent 的值赋给 i

    while (1) {
        wt *= (xnonc / (i+1.));
        # 将 wt 乘以 xnonc/(i+1) 的结果，更新 wt

        pterm = pcent - sumadj;
        # 将 pcent 减去 sumadj 的结果，赋值给 pterm

        term = wt*pterm;
        # 计算 wt 和 pterm 的乘积，赋值给 term

        ssum += term;
        # 将 term 加到 ssum 上

        i += 1;
        # 将 i 加 1

        dfd2 = (df + 2.*i)/2.0;
        # 计算 df + 2*i 的结果除以 2，赋值给 dfd2

        adj *= chid2/dfd2;
        # 将 adj 乘以 chid2/dfd2 的结果，更新 adj

        sumadj += adj;
        # 将 adj 加到 sumadj 上

        if (!((ssum >= abstol) && (term >= eps*ssum))) { break; }
        # 如果 ssum 不大于等于 abstol 或者 term 不大于等于 eps*ssum，则退出循环
    }

    return (struct TupleDD){.d1  = ssum, .d2 = 0.5 + (0.5 - ssum)};
    # 返回一个结构体 TupleDD，其中 d1 为 ssum，d2 为 0.5 + (0.5 - ssum)
struct TupleDD cumf(double f, double dfn, double dfd)
{
    // CUMulative F distribution
    //
    // Function:
    //
    // Computes the integral from 0 to F of the f-density with DFN
    // and DFD degrees of freedom.
    //
    // Arguments:
    //
    // F --> Upper limit of integration of the f-density.
    //        F is DOUBLE PRECISION
    //
    // DFN --> Degrees of freedom of the numerator sum of squares.
    //        DFN is DOUBLE PRECISION
    //
    // DFD --> Degrees of freedom of the denominator sum of squares.
    //        DFD is DOUBLE PRECISION
    //
    // Returns:
    // CUM <-- Cumulative f distribution.
    //        CUM is DOUBLE PRECISION
    //
    // Method:
    //
    // Formula 26.5.28 of Abramowitz and Stegun is used to reduce
    // the cumulative F to a cumulative beta distribution.
    //
    // Note:
    //
    // If F is less than or equal to 0, returns (0.0, 1.0).

    double dsum, prod, xx, yy;

    if (f <= 0.0) {
        return (struct TupleDD){.d1 = 0.0, .d2 = 1.0};
    }
    prod = dfn * f;
    dsum = dfd + prod;
    xx = dfd / dsum;
    if (xx > 0.5) {
        yy = prod / dsum;
        xx = 1. - yy;
    } else {
        yy = 1. - xx;
    }
    struct TupleDDI res = bratio(dfd * 0.5, dfn * 0.5, xx, yy);
    return (struct TupleDD){.d1 = res.d2, .d2 = res.d1};
}



struct TupleDDI cumfnc(double f, double dfn, double dfd, double pnonc)
{
    // F -NON- -C-ENTRAL F DISTRIBUTION
    //
    // Function:
    //
    // Computes noncentral F distribution with DFN and DFD
    // degrees of freedom and noncentrality parameter PNONC.
    //
    // Arguments:
    //
    // X --> Upper limit of integration of noncentral F in equation
    //
    // DFN --> Degrees of freedom of numerator
    //
    // DFD --> Degrees of freedom of denominator
    //
    // PNONC --> Noncentrality parameter.
    //
    // Returns:
    // CUM <-- Cumulative noncentral F distribution
    //
    // Method:
    //
    // Uses formula 26.6.20 of reference for infinite series.
    // Series is calculated backward and forward from J = LAMBDA/2
    // (this is the term with the largest Poisson weight) until
    // the convergence criterion is met.
}
    // 初始化变量，用于计算不完全贝塔函数的值
    double dsum, prod, xx, yy, adn, aup, b;
    double betdn, betup, centwt, dnterm, ssum;
    double upterm, xmult, xnonc;
    // 设定精度控制参数
    double eps = 1e-4;
    double abstol = 1e-300;
    // 状态变量初始化
    int status = 0;
    int i, icent;

    // 检查输入条件，若条件不满足则返回特定结果
    if (!(f > 0.0)) {
        return (struct TupleDDI){.d1 = 0.0, .d2 = 1.0, .i1 = status};
    }

    // 检查非中心参数是否过小，若是则调用cumf函数计算累积分布函数
    if (!(pnonc > 1e-10)) {
        struct TupleDD res = cumf(f, dfn, dfd);
        return (struct TupleDDI){.d1 = res.d1, .d2 = res.d2, .i1=status};
    }

    // 计算非中心参数的一半，并检查其是否为整数
    xnonc = pnonc / 2.0;
    icent = (int)xnonc;
    if (fabs(xnonc - icent) >= 1.0) {
        return (struct TupleDDI){.d1 = 0.0, .d2 = 0.0, .i1 = 1};
    }
    // 若icent为0，则设为1
    if (icent == 0) {
        icent = 1;
    }

    // 计算中心权重
    centwt = exp(-xnonc + icent*log(xnonc) - alngam(icent + 1));
    prod = dfn * f;
    dsum = dfd + prod;
    yy = dfd / dsum;
    if (yy > 0.5) {
        xx = prod / dsum;
        yy = 1. - xx;
    } else {
        xx = 1. - yy;
    }

    // 调用bratio函数计算不完全贝塔函数
    struct TupleDDI resi = bratio(dfn*0.5 + icent, dfd*0.5, xx, yy);
    betdn = resi.d1;
    adn = dfn/2. + icent;
    aup = adn;
    b = dfd / 2.0;
    betup = betdn;
    ssum = centwt*betdn;

    xmult = centwt;
    i = icent;

    // 计算不完全贝塔函数的一项
    if (adn < 2.0) {
        dnterm = exp(alngam(adn+b) - alngam(adn+1.) - alngam(b) +
                     adn*log(xx) + b*log(yy));
    } else {
        dnterm = exp(-betaln(adn, b) - log(adn) + adn*log(xx) +
                     b*log(yy));
    }

    // 迭代计算直到满足终止条件
    while (((ssum >= abstol) && (xmult*betdn >= eps*ssum)) && (i > 0)) {
        xmult *= (i/xnonc);
        i -= 1;
        adn -= 1;
        dnterm *= (adn + 1) / ((adn + b)*xx);
        betdn += dnterm;
        ssum += xmult*betdn;
    }

    i = icent + 1;
    xmult = centwt;

    // 计算另一种情况下的不完全贝塔函数的一项
    if ((aup - 1 + b) == 0) {
        upterm = exp(-alngam(aup) - alngam(b) + (aup-1)*log(xx) + b*log(yy));
    } else {
        if (aup < 2) {
            upterm = exp(alngam(aup-1+b) - alngam(aup)
                         - alngam(b) + (aup-1)*log(xx) + b*log(yy));
        } else {
            // 对于较大的aup，使用修正的表达式以避免问题
            upterm = exp(-betaln(aup-1, b) - log(aup-1) +
                         (aup-1)*log(xx) + b*log(yy));
        }
    }
    # 进入循环，计算 xmult 的乘积
    while (1) {
        xmult *= xnonc / i;
        # 增加 i 的值
        i += 1;
        # 增加 aup 的值
        aup += 1;
        # 计算 upterm 的乘积
        upterm *= (aup + b - 2.)*xx/(aup - 1.);
        # 减去 upterm 的值
        betup -= upterm;
        # 计算 ssum 的累加
        ssum += xmult*betup;
        # 检查终止条件，如果不满足则退出循环
        if (!((ssum >= abstol) && (xmult*betup >= eps*ssum))) { break; }
    }

    # 返回一个结构体 TupleDDI，包括 ssum、0.5 + (0.5 - ssum) 和 status 的值
    return (struct TupleDDI){.d1 = ssum, .d2 = 0.5 + (0.5 - ssum), .i1 = status};
// 结构体定义，表示返回两个 double 类型的值
struct TupleDD cumgam(double x, double a)
{
    // Double precision cUMulative incomplete GAMma distribution
    // 计算累积不完全 Gamma 分布函数的值

    // 如果 x 大于 0，调用 gratio 函数计算不完全 Gamma 函数值
    if (x > 0.0) {
        return gratio(a, x, 0);
    } else {
        // 如果 x 不大于 0，返回一个结构体，第一个值为 0.0，第二个值为 1.0
        return (struct TupleDD){.d1 = 0.0, .d2 = 1.0};
    }
}


struct TupleDD cumnbn(double s, double xn, double pr, double ompr)
{
    // CUmulative Negative BINomial distribution
    // 计算累积负二项分布函数的值

    // 暂无具体实现方法描述，可能在后续代码中调用其他函数来实现具体计算
}
    # 使用Abramowitz和Stegun的《数学函数手册》（1966年）中的公式26.5.26，
    # 将负二项分布转化为累积贝塔分布。
    return cumbet(pr, ompr, xn, s+1.)
struct TupleDD cumnor(double x)
{
    // Function: Computes the cumulative normal distribution.
    // x: Upper limit of integration (DOUBLE PRECISION).
    // Returns: Cumulative normal distribution value (DOUBLE PRECISION).
    //         Compliment of Cumulative normal distribution (DOUBLE PRECISION).

    // Renaming of function ANORM from Cody, W.D. (1993) "ALGORITHM 715: SPECFUN..."
    // with modifications for ccum and machine constants.

    // This function evaluates the normal distribution function:
    //         / x
    //  P(x) = |   -t*t/2
    //         |  e       dt
    //         /-oo
    // The main computation uses rational Chebyshev approximations
    // derived from "Rational Chebyshev approximations for the error function"
    // by W. J. Cody, Math. Comp., 1969, 631-637.
    // The program uses rational functions approximating the normal distribution
    // function to at least 18 significant decimal digits.

    double a[5] = {2.2352520354606839287e00, 1.6102823106855587881e02,
                   1.0676894854603709582e03, 1.8154981253343561249e04,
                   6.5682337918207449113e-2};
    double b[4] = {4.7202581904688241870e01, 9.7609855173777669322e02,
                   1.0260932208618978205e04, 4.5507789335026729956e04};
    double c[9] = {3.9894151208813466764e-1, 8.8831497943883759412e00,
                   9.3506656132177855979e01, 5.9727027639480026226e02,
                   2.4945375852903726711e03, 6.8481904505362823326e03,
                   1.1602651437647350124e04, 9.8427148383839780218e03,
                   1.0765576773720192317e-8};
    // 定义包含8个预设值的双精度数组d
    double d[8] = {2.2266688044328115691e01, 2.3538790178262499861e02,
                   1.5193775994075548050e03, 6.4855582982667607550e03,
                   1.8615571640885098091e04, 3.4900952721145977266e04,
                   3.8912003286093271411e04, 1.9685429676859990727e04};
    
    // 定义包含6个预设值的双精度数组p
    double p[6] = {2.1589853405795699e-01, 1.274011611602473639e-01,
                   2.2235277870649807e-02, 1.421619193227893466e-03,
                   2.9112874951168792e-05, 2.307344176494017303e-02};
    
    // 定义包含5个预设值的双精度数组q
    double q[5] = {1.28426009614491121e00, 4.68238212480865118e-01,
                   6.59881378689285515e-02, 3.78239633202758244e-03,
                   7.29751555083966205e-05};
    
    // 根据spmpar数组的第一个元素计算eps值
    double eps = spmpar[0] * 0.5;
    
    // 将spmpar数组的第二个元素赋给tiny
    double tiny = spmpar[1];
    
    // 计算x的绝对值，并赋给变量y
    double y = fabs(x);
    
    // 设定阈值为0.66291
    double threshold = 0.66291;
    
    // 声明变量dl, result, xden, xnum, xsq, ccum
    double dl, result, xden, xnum, xsq, ccum;
    
    // 声明整型变量i
    int i;
    
    // 如果y小于等于阈值threshold
    if (y <= threshold) {
        // 计算xsq为y的平方，或者0和y*eps中的较大者的平方
        xsq = y > eps ? x * x : 0;
        
        // 初始化xnum为a[4] * xsq
        xnum = a[4] * xsq;
        
        // 初始化xden为xsq
        xden = xsq;
        
        // 循环计算xnum和xden的值，基于a和b数组的前三个元素
        for (int i = 0; i < 3; i++) {
            xnum += a[i];
            xnum *= xsq;
            xden += b[i];
            xden *= xsq;
        }
        
        // 计算result为x乘以(xnum + a[3])除以(xden + b[3])
        result = x * (xnum + a[3]) / (xden + b[3]);
        
        // 计算ccum为0.5减去result
        ccum = 0.5 - result;
        
        // 将result加上0.5
        result += 0.5;
    } else if (y < sqrt(32)) {
        // 如果y大于threshold但小于sqrt(32)
        
        // 初始化xnum为c[8]乘以y
        xnum = c[8] * y;
        
        // 初始化xden为y
        xden = y;
        
        // 循环计算xnum和xden的值，基于c和d数组的前七个元素
        for (i = 0; i < 7; i++) {
            xnum += c[i];
            xnum *= y;
            xden += d[i];
            xden *= y;
        }
        
        // 计算result为(xnum + c[7])除以(xden + d[7])
        result = (xnum + c[7]) / (xden + d[7]);
        
        // 计算xsq为y的整数部分
        xsq = (int)(y * 1.6) / 1.6;
        
        // 计算dl为(y - xsq)乘以(y + xsq)
        dl = (y - xsq) * (y + xsq);
        
        // 计算result乘以exp(-xsq的平方除以2)和exp(-0.5乘以dl)
        result *= exp(-xsq * xsq * 0.5) * exp(-0.5 * dl);
        
        // 计算ccum为1减去result
        ccum = 1.0 - result;
    } else {
        // 如果y大于等于sqrt(32)
        
        // 初始化result为0
        result = 0.0;
        
        // 计算xsq为1除以x的平方
        xsq = (1.0 / x / x);
        
        // 初始化xnum为p[5]乘以xsq
        xnum = p[5] * xsq;
        
        // 初始化xden为xsq
        xden = xsq;
        
        // 循环计算xnum和xden的值，基于p和q数组的前四个元素
        for (i = 0; i < 4; i++) {
            xnum += p[i];
            xnum *= xsq;
            xden += q[i];
            xden *= xsq;
        }
        
        // 计算result为xsq乘以(xnum + p[4])除以(xden + q[4])
        result = xsq * (xnum + p[4]) / (xden + q[4]);
        
        // 计算result为(sqrt(1除以(2乘以PI))减去result)除以y
        result = (sqrt(1.0 / (2.0 * PI)) - result) / y;
        
        // 计算xsq为x的整数部分
        xsq = (int)(x * 1.6) / 1.6;
        
        // 计算dl为(x - xsq)乘以(x + xsq)
        dl = (x - xsq) * (x + xsq);
        
        // 计算result乘以exp(-xsq的平方除以2)和exp(-0.5乘以dl)
        result *= exp(-xsq * xsq * 0.5) * exp(-0.5 * dl);
        
        // 计算ccum为1减去result
        ccum = 1.0 - result;
    }
    
    // 如果x大于0，则交换result和ccum的值
    if (x > 0) {
        double tmp = result;
        result = ccum;
        ccum = tmp;
    }
    
    // 如果result小于tiny，则将result设为0
    if (result < tiny) {
        result = 0.0;
    }
    
    // 如果ccum小于tiny，则将ccum设为0
    if (ccum < tiny) {
        ccum = 0.0;
    }
    
    // 返回一个包含result和ccum的结构体TupleDD
    return (struct TupleDD){.d1 = result, .d2 = ccum};
// CUMulative Non-Central T-distribution
//
//
//                            Function
//
//
//    Computes the integral from -infinity to T of the non-central t-density.
//
//
//                            Arguments
//
//
//    T --> Upper limit of integration of the non-central t-density.
//                                                T is DOUBLE PRECISION
//
//    DF --> Degrees of freedom of the non-central t-distribution.
//                                                DF is DOUBLE PRECISION
//
//    PNONC --> Non-centrality parameter of the non-central t-distribution.
//                                                PNONC is DOUBLE PRECISION
//
//    CUM <-- Cumulative non-central t-distribution.
//                                                CUM is DOUBLE PRECISION
//
//    CCUM <-- Compliment of Cumulative non-central t-distribution.
//                                                CCUM is DOUBLE PRECISION
//
//
//                            Method
//
//
//    Uses the method described in Abramowitz and Stegun, Handbook of Mathematical Functions,
//    involving the incomplete beta function to compute the cumulative non-central t-distribution.
//

struct TupleDD cumtnc(double t, double df, double pnonc)
{
    double a, oma, tt, dfptt, xx, yy, cum, ccum;

    tt = t * t;
    dfptt = df + tt;
    xx = df / dfptt;
    yy = tt / dfptt;
    struct TupleDD res = cumbet(xx, yy, 0.5 * df, 0.5);
    a = res.d1;
    oma = res.d2;
    if (t > 0.0) {
        ccum = 0.5 * a;
        cum = oma + ccum;
    } else {
        cum = 0.5 * a;
        ccum = oma + cum;
    }
    return (struct TupleDD){.d1 = cum, .d2 = ccum};
}
    //
    //    Computes the integral from -infinity to T of the non-central
    //    t-density.
    //
    //
    //                            Arguments
    //
    //
    //    T --> Upper limit of integration of the non-central t-density.
    //                                                T is DOUBLE PRECISION
    //
    //    DF --> Degrees of freedom of the non-central t-distribution.
    //                                                DF is DOUBLE PRECISION
    //
    //    PNONC --> Non-centrality parameter of the non-central t distribution.
    //                                                PNONC is DOUBLE PRECISION
    //
    //    CUM <-- Cumulative t-distribution.
    //                                                CCUM is DOUBLE PRECISION
    //
    //    CCUM <-- Complement of Cumulative t-distribution.
    //                                                CCUM is DOUBLE PRECISION
    //
    //
    //                            Method
    //
    //    Upper tail of the cumulative noncentral t using
    //    formulae from page 532 of Johnson, Kotz, Balakrishnan, Continuous
    //    Univariate Distributions, Vol 2, 2nd Edition. Wiley (1995)
    //
    //    This implementation starts the calculation at i = lambda,
    //    which is near the largest Di. It then sums forward and backward.
    double alghdf, b, bb, bbcent, bcent, cent, cum, ccum, d, dcent;
    double dpnonc, dum1, dum2, e, ecent, lmbda, lnomx, lnx, omx;
    double pnonc2, s, scent, ss, sscent, t2, term, tt, twoi, x;
    double xi, xlnd, xlne;
    double conv = 1.e-7;
    double tiny = 1.e-10;
    int qrevs;

    // Check if the absolute value of non-centrality parameter is very small
    if (fabs(pnonc) <= tiny) { return cumt(t, df); }

    // Determine if t is negative (qrevs = 1) or positive (qrevs = 0)
    qrevs = t < 0.0;
    tt = (qrevs ? -t : t);
    dpnonc = (qrevs ? -pnonc : pnonc);
    pnonc2 = pow(dpnonc, 2);
    t2 = pow(tt, 2);

    // If tt (absolute value of t) is very small, return cumulative normal distribution at -pnonc
    if (fabs(tt) <= tiny) { return cumnor(-pnonc); }

    // Calculate lambda as half of pnonc squared
    lmbda = 0.5 * pnonc2;
    x = df / (df + t2);
    omx = 1.0 - x;
    lnx = log(x);
    lnomx = log(omx);
    alghdf = gamln(0.5 * df);

    // Determine cent as the maximum of floor(lmbda) and 1
    cent = fmax(floor(lmbda), 1.0);

    // Compute d=T(2i) in log space and offset by exp(-lambda)
    xlnd = cent * log(lmbda) - gamln(cent + 1.0) - lmbda;
    dcent = exp(xlnd);

    // Compute e=t(2i+1) in log space offset by exp(-lambda)
    xlne = (cent + 0.5) * log(lmbda) - gamln(cent + 1.5) - lmbda;
    ecent = exp(xlne);

    // Adjust ecent if dpnonc is negative
    if (dpnonc < 0.0) {
        ecent = -ecent;
    }

    // Compute bcent=B(2*cent)
    struct TupleDDI res1 = bratio(0.5 * df, cent + 0.5, x, omx);
    bcent = res1.d1;
    dum1 = res1.d2;

    // Compute bbcent=B(2*cent+1)
    struct TupleDDI res2 = bratio(0.5 * df, cent + 1.0, x, omx);
    bbcent = res2.d1;
    dum2 = res2.d2;

    // If both bcent and bbcent are essentially zero, indicating t is effectively infinite
    if ((bbcent + bcent) < tiny) {
        if (qrevs) {
            return (struct TupleDD){.d1 = 0.0, .d2 = 1.0};
        } else {
            return (struct TupleDD){.d1 = 1.0, .d2 = 0.0};
        }
    }
    // 如果 dum1 和 dum2 的和小于 tiny，返回一个值
    // 这里的返回值是由 cumnor 函数返回的，参数是 -pnonc
    if ((dum1 + dum2) < tiny) { return cumnor(-pnonc); }

    // 计算 ccum 的第一项是 D*B + E*BB
    ccum = dcent*bcent + ecent*bbcent;

    // 计算 scent = B(2*(cent+1)) - B(2*cent))
    scent = exp(gamln(0.5*df + cent + 0.5) - gamln(cent + 1.5) - alghdf
                + 0.5*df*lnx + (cent + 0.5)*lnomx);

    // 计算 sscent = B(2*cent+3) - B(2*cent+1)
    sscent = exp(gamln(0.5*df + cent + 1.) - gamln(cent + 2.) - alghdf
                 + 0.5*df*lnx + (cent + 1.)*lnomx);

    // 初始化变量
    xi = cent + 1.0;
    twoi = 2.*xi;
    d = dcent;
    e = ecent;
    b = bcent;
    s = scent;
    bb = bbcent;
    ss = sscent;

    // 开始迭代循环
    while (1) {
        // 更新变量
        b += s;
        bb += ss;
        d = (lmbda/xi)*d;
        e = (lmbda/(xi + 0.5))*e;
        // 计算当前 term 的值
        term = d*b + e*bb;
        // 更新 ccum
        ccum += term;
        // 更新 s 和 ss
        s *= omx*(df + twoi - 1.)/(twoi + 1.);
        ss *= omx*(df + twoi)/(twoi + 2.);
        // 更新 xi 和 twoi
        xi += 1.0;
        twoi = 2.0*xi;
        // 检查终止条件
        if (fabs(term) <= conv*ccum) { break; }
    }

    // 初始化变量
    xi = cent;
    twoi = 2.0*xi;
    d = dcent;
    e = ecent;
    b = bcent;
    bb = bbcent;
    s = scent*(1.0 + twoi)/((df + twoi - 1.)*omx);
    ss = sscent*(2.0 + twoi)/((df + twoi)*omx);

    // 开始迭代循环
    while (1) {
        // 更新变量
        b -= s;
        bb -= ss;
        d *= (xi / lmbda);
        e *= (xi + 0.5)/lmbda;
        // 计算当前 term 的值
        term = d*b + e*bb;
        // 更新 ccum
        ccum += term;
        // 更新 xi
        xi -= 1.0;
        // 检查终止条件
        if (xi < 0.5) { break; }
        twoi = 2.0*xi;
        s *= (1. + twoi) / ((df + twoi - 1.)*omx);
        ss *= (2. + twoi) / ((df + twoi)*omx);
        if (fabs(term) <= conv*ccum) { break; }
    }

    // 由于舍入误差，ccum 可能不在 [0, 1] 区间内
    // 强制将其限制在 [0, 1] 区间内
    if (qrevs) {
        cum = fmax(fmin(0.5*ccum, 1.), 0.);
        ccum = fmax(fmin(1.-cum, 1.), 0.);
    } else {
        ccum = fmax(fmin(0.5*ccum, 1.), 0.);
        cum = fmax(fmin(1.-ccum, 1.), 0.);
    }

    // 返回结构体 TupleDD，其中 d1 是 cum，d2 是 ccum
    return (struct TupleDD){.d1 = cum, .d2 = ccum};
double devlpl(double *a, int n, double x)
{
    // Double precision EVALuate a PoLynomial at X
    // Evaluate a polynomial defined by coefficients in array A of length N at point X.

    double temp = a[n-1];   // Initialize temp with the highest degree coefficient
    int i;

    for (i = n - 2; i >= 0; i--) {
        temp = a[i] + temp*x;   // Horner's method to evaluate polynomial
    }
    return temp;   // Return the evaluated polynomial value at X
}


double dinvnr(double p, double q)
{
    // Double precision NoRmal distribution INVerse
    // Compute the inverse of the cumulative normal distribution function for probability P.

    int i, maxit = 100;
    double eps = 1e-13;
    double r2pi = sqrt(1. / (2.*PI));
    double strtx, xcur, cum, pp, dx;

    pp = (p > q ? q : p);   // Determine the smaller probability
    strtx = stvaln(pp);     // Get a starting value using some function stvaln(pp)
    xcur = strtx;

    // Iteratively refine the estimate using Newton's method
    for (i = 0; i < maxit; i++) {
        struct TupleDD res = cumnor(xcur);   // Compute cumulative normal distribution
        cum = res.d1;
        dx = (cum - pp) / (r2pi * exp(-0.5*xcur*xcur));   // Newton's method iteration
        xcur -= dx;
        if (fabs(dx / xcur) < eps) {
            return (p > q ? -xcur : xcur);   // Return the final estimate
        }
    }
    return (p > q ? -strtx : strtx);   // Return the starting value if convergence fails
}


void dinvr(DinvrState *S, DzrorState *DZ)
{
    // Double precision bounds the zero of the function and invokes zror
    // Bounds the function and uses ZROR to find its zero.

    // This function typically involves reverse communication for zero finding.
    // The details of its operation are context-dependent and not fully specified here.
}
    // STATUS: 当前状态码，用于表示函数调用的结果
    INTEGER STATUS
    
    // X: 函数 F(X) 所需计算的输入值
    DOUBLE PRECISION X
    
    // FX: 当 STATUS = 1 时返回的 F(X) 的计算结果
    DOUBLE PRECISION FX
    
    // QLEFT: 如果 QMFINV 返回 .FALSE.，则表示步进搜索在 SMALL 处未能成功终止
    // QLEFT 是一个逻辑值，为 .TRUE. 或 .FALSE.
    LOGICAL QLEFT
    
    // QHI: 如果 QMFINV 返回 .FALSE.，则表示在搜索终止时 F(X) > Y 或 F(X) < Y
    // QHI 是一个逻辑值，为 .TRUE. 或 .FALSE.
    LOGICAL QHI
    // 进入无限循环，处理状态机的各个状态
    while (1) {
        // 如果下一个状态是0
        if (S->next_state == 0) {
            // 检查 S->x 是否在 S->small 和 S->big 之间，设置 S->qcond
            S->qcond = ((S->small <= S->x) && (S->x <= S->big));
            // 如果 S->qcond 不成立
            if (!(S->qcond)) {
                // 将状态设置为 -2 并返回
                S->status = -2;
                return;
            }
            // 保存当前的 S->x 值到 S->xsave
            S->xsave = S->x;
            // 将 S->x 设置为 S->small
            S->x = S->small;
            // 设置下一个状态为 10
            S->next_state = 10;
            // 设置状态为 1 并返回
            S->status = 1;
            return;
        } else if (S->next_state == 10) {
            // 设置 S->fsmall 为 S->fx
            S->fsmall = S->fx;
            // 将 S->x 设置为 S->big
            S->x = S->big;
            // 设置下一个状态为 20
            S->next_state = 20;
            // 设置状态为 1 并返回
            S->status = 1;
            return;
        } else if (S->next_state == 20) {
            // 设置 S->fbig 为 S->fx
            S->fbig = S->fx;
            // 检查 S->fbig 是否大于 S->fsmall，设置 S->qincr
            S->qincr = (S->fbig > S->fsmall);
            // 设置状态为 -1
            S->status = -1;
            // 如果 S->qincr 不成立
            if (!(S->qincr)) {
                // 50
                // 如果 S->fsmall 大于等于 0.0
                if (S->fsmall >= 0.0) {
                    // 60
                    // 如果 S->fbig 不小于等于 0.0
                    if (!(S->fbig <= 0.0)) {
                        // 设置 S->qleft 和 S->qhi 的值并返回
                        S->qleft = 0;
                        S->qhi = 1;
                        return;
                    }
                } else {
                    // 设置 S->qleft 和 S->qhi 的值并返回
                    S->qleft = 1;
                    S->qhi = 0;
                    return;
                }
            } else {
                // 如果 S->fsmall 小于等于 0.0
                if (S->fsmall <= 0.0) {
                    // 30
                    // 如果 S->fbig 不大于等于 0.0
                    if (!(S->fbig >= 0.0)) {
                        // 设置 S->qleft 和 S->qhi 的值并返回
                        S->qleft = 0;
                        S->qhi = 0;
                        return;
                    }
                } else {
                    // 设置 S->qleft 和 S->qhi 的值并返回
                    S->qleft = 1;
                    S->qhi = 1;
                    return;
                }
            }
            // 设置状态为 1
            S->status = 1;
            // 恢复 S->x 的值为之前保存的 S->xsave
            S->x = S->xsave;
            // 计算步长并设置下一个状态为 90
            S->step = fmax(S->absstp, S->relstp*fabs(S->x));
            S->next_state = 90;
            return;
        } else if (S->next_state == 90) {
            // 设置 S->yy 为 S->fx
            S->yy = S->fx;
            // 如果 S->yy 等于 0.0
            if (S->yy == 0.0) {
                // 设置状态为 0，并标记 S->qok 为 1 并返回
                S->status = 0;
                S->qok = 1;
                return;
            }
            // 设置下一个状态为 100
            S->next_state = 100;
            // 继续下一次循环
    // 处理需要向上步进的情况
    } else if (S->next_state == 100) {
        // 设置是否需要步进的标志 qup
        S->qup = (S->qincr & (S->yy < 0.)) || ((!(S->qincr)) && (S->yy > 0.));
        if (S->qup) {
            // 更新搜索范围上限和下限
            S->xlb = S->xsave;
            S->xub = fmin(S->xlb + S->step, S->big);
            S->next_state = 120;
        } else {
            // 处理状态 170
            S->xub = S->xsave;
            S->xlb = fmax(S->xub - S->step, S->small);
            S->next_state = 190;
        }
    } else if (S->next_state == 120) {
        // 设置当前搜索点为搜索范围的上限，状态置为 130
        S->x = S->xub;
        S->status = 1;
        S->next_state = 130;
        return;
    } else if (S->next_state == 130) {
        // 记录当前函数值，设置是否需要步进的标志 qbdd
        S->yy = S->fx;
        S->qbdd = (S->qincr & (S->yy >= 0.)) || ((!(S->qincr)) && (S->yy <= 0.));
        // 判断是否达到搜索边界或者函数变号
        S->qlim = (S->xub >= S->big);
        S->qcond = ((S->qbdd) || (S->qlim));
        if (S->qcond) {
            S->next_state = 150;
        } else {
            // 更新步长，更新搜索范围上限和下限，状态置为 120
            S->step *= S->stpmul;
            S->xlb = S->xub;
            S->xub = fmin(S->xlb + S->step, S->big);
            S->next_state = 120;
        }
    } else if (S->next_state == 150) {
        // 处理状态 150
        if (S->qlim & (!(S->qbdd))) {
            // 达到搜索边界且未发现函数变号，设置状态为 -1，返回搜索结果
            S->status = -1;
            S->qleft = 0;
            S->qhi = (!(S->qincr));
            S->x = S->big;
            return;
        } else {
            S->next_state = 240;
        }
    } else if (S->next_state == 190) {
        // 处理需要向下步进的情况
        S->x = S->xlb;
        S->status = 1;
        S->next_state = 200;
        return;
    // 处理需要向下步进的情况
    } else if (S->next_state == 200) {
        // 记录当前函数值，设置是否需要步进的标志 qbdd
        S->yy = S->fx;
        S->qbdd = (((S->qincr) && (S->yy <= 0.0)) || ((!(S->qincr)) && (S->yy >= 0.0)));
        // 判断是否达到搜索边界或者函数变号
        S->qlim = ((S->xlb) <= (S->small));
        S->qcond = ((S->qbdd) || (S->qlim));
        if (S->qcond) {
            S->next_state = 220;
        } else {
            // 更新步长，更新搜索范围上限和下限，状态置为 190
            S->step *= S->stpmul;
            S->xub = S->xlb;
            S->xlb = fmax(S->xub - S->step, S->small);
            S->next_state = 190;
        }
    } else if (S->next_state == 220) {
        // 处理状态 220
        if ((S->qlim) && (!(S->qbdd))) {
            // 达到搜索边界且未发现函数变号，设置状态为 -1，返回搜索结果
            S->status = -1;
            S->qleft = 1;
            S->qhi = S->qincr;
            S->x = S->small;
            return;
        } else {
            S->next_state = 240;
        }
    // 如果程序执行到这里，说明 xlb 和 xub 界定了函数 f 的零点位置。
        } else if (S->next_state == 240) {
            // 使用问题中提供的 xub 覆盖给定的 DZ 结构体中的 xhi
            DZ->xhi = S->xub;
            // 使用问题中提供的 xlb 覆盖给定的 DZ 结构体中的 xlo
            DZ->xlo = S->xlb;
            // 使用问题中提供的 abstol 覆盖给定的 DZ 结构体中的 atol
            DZ->atol = S->abstol;
            // 使用问题中提供的 reltol 覆盖给定的 DZ 结构体中的 rtol
            DZ->rtol = S->reltol;
            // 使用 xlb 初始化给定的 DZ 结构体中的 x
            DZ->x = S->xlb;
            // 使用 xlb 初始化给定的 DZ 结构体中的 b
            DZ->b = S->xlb;
            // 设置状态机的下一个状态为 250
            S->next_state = 250;
        } else if (S->next_state == 250) {
            // 调用 dzror 函数处理 DZ 结构体
            dzror(DZ);
            // 如果处理后 DZ 的状态为 1
            if (DZ->status == 1) {
                // 将状态机的下一个状态设置为 260
                S->next_state = 260;
                // 设置状态机的状态为 1
                S->status = 1;
                // 将状态机的 x 设置为 DZ 结构体中的 x
                S->x = DZ->x;
                // 函数返回
                return;
            } else {
                // 将状态机的 x 设置为 DZ 结构体中的 xlo
                S->x = DZ->xlo;
                // 设置状态机的状态为 0
                S->status = 0;
                // 函数返回
                return;
            }
        } else if (S->next_state == 260) {
            // 使用当前状态机的 fx 属性覆盖 DZ 结构体中的 fx
            DZ->fx = S->fx;
            // 将状态机的下一个状态设置为 250
            S->next_state = 250;
        } else {
    // 错误状态，不应该执行到这里
            // 设置状态机的状态为 -9999 表示错误状态
            S->status = -9999;
            // 函数返回
            return;
        }
    }
double dt1(double p, double q, double df)
{
    //    Double precision Initialize Approximation to
    //        INVerse of the cumulative T distribution
    //
    //
    //                            Function
    //
    //
    //    Returns  the  inverse   of  the T   distribution   function, i.e.,
    //    the integral from 0 to INVT of the T density is P. This is an
    //    initial approximation
    //
    //
    //                            Arguments
    //
    //
    //    P --> The p-value whose inverse from the T distribution is
    //        desired.
    //                P is DOUBLE PRECISION
    //
    //    Q --> 1-P.
    //                Q is DOUBLE PRECISION
    //
    //    DF --> Degrees of freedom of the T distribution.
    //                DF is DOUBLE PRECISION

    // Declare local variables
    double ssum, term, x, xx;
    double denpow = 1.0;
    
    // Coefficients for the approximation
    double coef[4][5] = {{1., 1., 0., 0., 0.},
                         {3., 16., 5., 0., 0.},
                         {-15., 17., 19., 3., 0.},
                         {-945., -1920., 1482., 776., 79.}};
    
    // Denominators for the approximation
    double denom[4] = {4., 96., 384.0, 92160.0};
    
    // Degree of polynomials
    int i, ideg[4] = {2, 3, 4, 5};

    // Compute initial values
    x = fabs(dinvnr(p, q)); // Calculate absolute value of normal deviate
    xx = x*x;
    ssum = x;

    // Loop to compute the approximation
    for (i = 0; i < 4; i++) {
        term = (devlpl(coef[i], ideg[i], xx)) * x;
        denpow *= df;
        ssum += term / (denpow * denom[i]);
    }

    // Return the computed value
    return (p >= 0.5 ? ssum : -ssum); // Return ssum with sign based on p
}

void dzror(DzrorState *S)
{
    //    Double precision ZeRo of a function -- Reverse Communication
    //
    //
    //                            Function
    //
    //
    //    Performs the zero finding.  STZROR must have been called before
    //    this routine in order to set its parameters.
    //
    //
    //                            Arguments
    //
    //
    //    STATUS <--> At the beginning of a zero finding problem, STATUS
    //                should be set to 0 and ZROR invoked.  (The value
    //                of other parameters will be ignored on this call.)
    //
    //                When ZROR needs the function evaluated, it will set
    //                STATUS to 1 and return.  The value of the function
    //                should be set in FX and ZROR again called without
    //                changing any of its other parameters.
    //
    //                When ZROR has finished without error, it will return
    //                with STATUS 0.  In that case (XLO,XHI) bound the answe
    //
    //                If ZROR finds an error (which implies that F(XLO)-Y an
    //                F(XHI)-Y have the same sign, it returns STATUS -1.  In
    //                this case, XLO and XHI are undefined.
    //                        INTEGER STATUS
    //
    //    X <-- The value of X at which F(X) is to be evaluated.
    //                        DOUBLE PRECISION X
    //
    //    FX --> The value of F(X) calculated when ZROR returns with
    //        STATUS = 1.

    // Implementation of zero finding using reverse communication
}
    //                        DOUBLE PRECISION FX
    //
    //    XLO <-- When ZROR returns with STATUS = 0, XLO bounds the
    //            inverval in X containing the solution below.
    //                        DOUBLE PRECISION XLO
    //
    //    XHI <-- When ZROR returns with STATUS = 0, XHI bounds the
    //            inverval in X containing the solution above.
    //                        DOUBLE PRECISION XHI
    //
    //    QLEFT <-- .TRUE. if the stepping search terminated unsuccessfully
    //            at XLO.  If it is .FALSE. the search terminated
    //            unsuccessfully at XHI.
    //                QLEFT is LOGICAL
    //
    //    QHI <-- .TRUE. if F(X) > Y at the termination of the
    //            search and .FALSE. if F(X) < Y at the
    //            termination of the search.
    //                QHI is LOGICAL
//    Evaluation of the complementary error function
//    计算互补误差函数

double erfc1(int ind, double x)
{
//    Evaluation of the complementary error function
//    评估互补误差函数

//    Erfc1(ind,x) = erfc(x)            if ind = 0
//    If ind = 0, return erfc(x)

//    Erfc1(ind,x) = exp(x*x)*erfc(x)   otherwise
//    Otherwise, return exp(x*x)*erfc(x)

    double ax, bot, t, top, result;
    // Initialize variables
    double c = 0.564189583547756;
    // Constant c for the error function
    double a[5] = {.771058495001320e-04, -.133733772997339e-02,
                   .323076579225834e-01, .479137145607681e-01,
                   .128379167095513e+00};
    // Coefficients for the polynomial approximation of the error function
    double b[3] = {.301048631703895e-02, .538971687740286e-01,
                   .375795757275549e+00};
    // Coefficients for the polynomial approximation of the error function
    // 定义存储系数的数组，用于计算近似值
    double p[8] = {-1.36864857382717e-07, 5.64195517478974e-01,
                   7.21175825088309e+00, 4.31622272220567e+01,
                   1.52989285046940e+02, 3.39320816734344e+02,
                   4.51918953711873e+02, 3.00459261020162e+02};
    double q[8] = {1.00000000000000e+00, 1.27827273196294e+01,
                   7.70001529352295e+01, 2.77585444743988e+02,
                   6.38980264465631e+02, 9.31354094850610e+02,
                   7.90950925327898e+02, 3.00459260956983e+02};
    double r[5] = {2.10144126479064e+00, 2.62370141675169e+01,
                   2.13688200555087e+01, 4.65807828718470e+00,
                   2.82094791773523e-01};
    double s[4] = {9.41537750555460e+01, 1.87114811799590e+02,
                   9.90191814623914e+01, 1.80124575948747e+01};

    // 若 x <= -5.6，则返回条件成立时的结果，否则返回另一结果
    if (x <= -5.6) { return (ind == 0 ? 2.0 : (2*exp(x*x))); }

    // 检查是否需要返回零值，条件是 ind 为 0 且 x 大于 26.64
    // sqrt(log(np.finfo(np.float64).max)) ~= 26.64
    if ((ind == 0) && (x > 26.64))  { return 0.0; }

    // 计算绝对值
    ax = fabs(x);

    // 如果 ax 小于等于 0.5，则使用多项式近似计算结果
    if (ax <= 0.5) {
        t = x*x;
        // 计算多项式的分子部分
        top = (((((a[0])*t+a[1])*t+a[2])*t+a[3])*t+a[4]) + 1.0;
        // 计算多项式的分母部分
        bot = (((b[0])*t+b[1])*t+b[2])*t + 1.0;
        // 计算结果并返回，根据 ind 决定是否乘以 exp(t)
        result = 0.5 + (0.5 - x*(top/bot));
        return (ind == 0 ? result : result*exp(t));
    } else if ((0.5 < ax) && (ax <= 4.0)) {
        // 如果 ax 在 0.5 到 4.0 之间，则使用 p 和 q 数组进行近似计算
        // 计算多项式的分子部分
        top = (((((((p[0]
                    )*ax+p[1]
                   )*ax+p[2]
                  )*ax+p[3]
                 )*ax+p[4]
                )*ax+p[5]
               )*ax+p[6]
              )*ax + p[7];
        // 计算多项式的分母部分
        bot = (((((((q[0]
                  )*ax+q[1]
                 )*ax+q[2]
                )*ax+q[3]
               )*ax+q[4]
              )*ax+q[5]
             )*ax+q[6])*ax + q[7];
        // 计算结果
        result = top / bot;
    } else {
        // 如果 ax 大于 4.0，则使用 r 和 s 数组进行近似计算
        t = pow(1 / x, 2);
        // 计算多项式的分子部分
        top = (((r[0]*t+r[1])*t+r[2])*t+r[3])*t + r[4];
        // 计算多项式的分母部分
        bot = (((s[0]*t+s[1])*t+s[2])*t+s[3])*t + 1.0;
        // 计算结果，并在最后除以 ax
        result = (c - t*(top/bot)) / ax;
    }
    
    // 根据 ind 决定是否乘以 exp(-(x*x))，然后返回最终结果
    if (ind == 0) {
        result *= exp(-(x*x));
        return (x < 0 ? (2.0 - result) : result);
    } else {
        return (x < 0 ? (2.0*exp(x*x) - result) : result);
    }
double esum(int mu, double x)
{
    //    Evaluation of exp(mu + x)

    if (x > 0.0) {
        if ((mu > 0.) || (mu + x < 0)) {
            return exp(mu)*exp(x);  // 如果 x > 0 且 (mu > 0 或 mu + x < 0)，返回 exp(mu) * exp(x)
        } else {
            return exp(mu + x);  // 如果 x > 0 且 !(mu > 0 或 mu + x < 0)，返回 exp(mu + x)
        }
    } else {
        if ((mu < 0.) || (mu + x > 0.)) {
            return exp(mu)*exp(x);  // 如果 x <= 0 且 (mu < 0 或 mu + x > 0)，返回 exp(mu) * exp(x)
        } else {
            return exp(mu + x);  // 如果 x <= 0 且 !(mu < 0 或 mu + x > 0)，返回 exp(mu + x)
        }
    }
}


double fpser(double a, double b, double x, double eps)
{
    //           Evaluation of i_x(a,b)
    //
    //    for b < Min(eps,eps*a) and x <= 0.5.

    double an, c, s, t, tol, result = 1.0;

    if (!(a <= 1e-3*eps)){
        result = 0.0;  // 如果 !(a <= 1e-3*eps)，将 result 置为 0
        t = a*log(x);  // 计算 t = a * log(x)
        if (t < -708.) {return result;}  // 如果 t < -708，直接返回 result
        result = exp(t);  // 否则 result = exp(t)
    }
    //  Note that 1/Beta(a,b) = b
    result *= (b / a);  // 结果乘以 b / a
    tol = eps /a;  // 计算公差 tol = eps / a
    an = a + 1.0;
    t = x;
    s = t / an;
    while (1) {
        an += 1;
        t *= x;
        c = t / an;
        s += c;
        if (!(fabs(c) > tol)) { break; }  // 当 |c| <= tol 时退出循环
    }
    return result*(1. + a*s);  // 返回结果乘以 (1 + a*s)
}


double gam1(double a)
{
    //    Computation of 1/gamma(a+1) - 1  for -0.5 <= A <= 1.5

    double bot, d, t, top, w;
    double p[7] = {.577215664901533e+00, -.409078193005776e+00,
                   -.230975380857675e+00, .597275330452234e-01,
                   .766968181649490e-02, -.514889771323592e-02,
                   .589597428611429e-03};
    double q[5] = {.100000000000000e+01, .427569613095214e+00,
                   .158451672430138e+00, .261132021441447e-01,
                   .423244297896961e-02};
    double r[9] = {-.422784335098468e+00, -.771330383816272e+00,
                   -.244757765222226e+00, .118378989872749e+00,
                   .930357293360349e-03, -.118290993445146e-01,
                   .223047661158249e-02, .266505979058923e-03,
                   -.132674909766242e-03};
    double s[2] = {.273076135303957e+00, .559398236957378e-01};

    d = a - 0.5;
    t = (d > 0 ? d - 0.5 : a);  // 如果 d > 0，t = d - 0.5；否则 t = a

    if (t == 0.0) { return 0.0; }  // 如果 t == 0，直接返回 0

    if (t < 0) {
        top = ((((((((r[8]
                     )*t+r[7]
                    )*t+r[6]
                   )*t+r[5]
                  )*t+r[4]
                 )*t+r[3]
                )*t+r[2]
               )*t+r[1]
              )*t + r[0];
        bot = (s[1]*t + s[0])*t + 1.0;
        w = top / bot;
        if (d > 0.0) {
            return t*w/a;  // 如果 d > 0，返回 t*w/a
        } else {
            return a * ((w + 0.5) + 0.5);  // 否则返回 a * ((w + 0.5) + 0.5)
        }
    }
    top = ((((((p[6]
               )*t+p[5]
              )*t+p[4]
             )*t+p[3]
            )*t+p[2]
           )*t+p[1]
          )*t + p[0];
    bot = ((((q[4]
             )*t+q[3]
            )*t+q[2]
           )*t+q[1]
          )*t + 1.0;
    w = top / bot;
    if (d > 0.0) {
        return (t/a) * ((w - 0.5) - 0.5);  // 如果 d > 0，返回 (t/a) * ((w - 0.5) - 0.5)
    } else {
        return a * w;  // 否则返回 a * w
    }
}


struct TupleDI gaminv(double a, double p, double q, double x0)
{
    //        INVERSE INCOMPLETE GAMMA RATIO FUNCTION
    //
    //    GIVEN POSITIVE A, AND NONEGATIVE P AND Q WHERE P + Q = 1.
    //    THEN X IS COMPUTED WHERE P(A,X) = P AND Q(A,X) = Q. SCHRODER
    //    ITERATION IS EMPLOYED. THE ROUTINE ATTEMPTS TO COMPUTE X
    //    TO 10 SIGNIFICANT DIGITS IF THIS IS POSSIBLE FOR THE
    //    PARTICULAR COMPUTER ARITHMETIC BEING USED.
    //
    //                    ------------
    //
    //    X IS A VARIABLE. IF P = 0 THEN X IS ASSIGNED THE VALUE 0,
    //    AND IF Q = 0 THEN X IS SET TO THE LARGEST FLOATING POINT
    //    NUMBER AVAILABLE. OTHERWISE, GAMINV ATTEMPTS TO OBTAIN
    //    A SOLUTION FOR P(A,X) = P AND Q(A,X) = Q. IF THE ROUTINE
    //    IS SUCCESSFUL THEN THE SOLUTION IS STORED IN X.
    //
    //    X0 IS AN OPTIONAL INITIAL APPROXIMATION FOR X. IF THE USER
    //    DOES NOT WISH TO SUPPLY AN INITIAL APPROXIMATION, THEN SET
    //    X0 <= 0.
    //
    //    IERR IS A VARIABLE THAT REPORTS THE STATUS OF THE RESULTS.
    //    WHEN THE ROUTINE TERMINATES, IERR HAS ONE OF THE FOLLOWING
    //    VALUES ...
    //
    //    IERR =  0    THE SOLUTION WAS OBTAINED. ITERATION WAS
    //                NOT USED.
    //    IERR>0    THE SOLUTION WAS OBTAINED. IERR ITERATIONS
    //                WERE PERFORMED.
    //    IERR = -2    (INPUT ERROR) A <= 0
    //    IERR = -3    NO SOLUTION WAS OBTAINED. THE RATIO Q/A
    //                IS TOO LARGE.
    //    IERR = -4    (INPUT ERROR) P + Q .NE. 1
    //    IERR = -6    20 ITERATIONS WERE PERFORMED. THE MOST
    //                RECENT VALUE OBTAINED FOR X IS GIVEN.
    //                THIS CANNOT OCCUR IF X0 <= 0.
    //    IERR = -7    ITERATION FAILED. NO VALUE IS GIVEN FOR X.
    //                THIS MAY OCCUR WHEN X IS APPROXIMATELY 0.
    //    IERR = -8    A VALUE FOR X HAS BEEN OBTAINED, BUT THE
    //                ROUTINE IS NOT CERTAIN OF ITS ACCURACY.
    //                ITERATION CANNOT BE PERFORMED IN THIS
    //                CASE. IF X0 <= 0, THIS CAN OCCUR ONLY
    //                WHEN P OR Q IS APPROXIMATELY 0. IF X0 IS
    //                POSITIVE THEN THIS CAN OCCUR WHEN A IS
    //                EXCEEDINGLY CLOSE TO X AND A IS EXTREMELY
    //                LARGE (SAY A >= 1.E20).
    
    double act_val, ap1, ap2, ap3, apn, b, bot, d, g, h;
    double pn, qg, qn, r, rta, s, s2, ssum, t, top, tol_val, u, w, y, z;
    double am1 = 0.;
    double ln10 = log(10);  // 自然对数的底数 e 的对数，即 ln(10)
    double c = 0.57721566490153286060651209008;  // 欧拉常数 γ
    double tol = 1e-5;  // 公差，用于控制数值精度
    double e = spmpar[0];  // 计算机中浮点运算的机器精度
    double e2 = 2*e;  // 机器精度的两倍
    double amax = 0.4e-10 / (e*e);  // 最大绝对误差的上限
    double xmin = spmpar[1];  // 可接受的最小 x 值
    double xmax = spmpar[2];  // 可接受的最大 x 值
    double xn = x0;  // 初始近似值 x0
    double x = 0.;  // 存储解 x
    int ierr = 0;  // 返回状态码，表示计算结果的状态
    int iop = (e > 1e-10 ? 1 : 0);  // 根据机器精度设置操作码
    int use_p = (p > 0.5 ? 0 : 1);  // 根据参数 p 决定是否使用 p
    int skip140 = 0;  // 控制流程跳转
    double arr[4] = {3.31125922108741, 11.6616720288968,
                     4.28342155967104, 0.213623493715853};  // 常数数组
    double barr[4] = {6.61053765625462, 6.40691597760039,
                      1.27364489782223, 0.036117081018842};  // 常数数组
    double eps0[2] = {1e-10, 1e-8};  // 机器精度数组
    double amin[2] = {500., 100.};  // 最小值数组
    // 定义两个长度为2的数组，分别初始化为极小值
    double bmin[2] = {1e-28, 1e-13};
    double dmin[2] = {1e-6, 1e-4};
    double emin[2] = {2e-3, 6e-3};
    // 从数组eps0中取出特定索引iop处的值赋给变量eps
    double eps = eps0[iop];

    // 如果a不大于0，返回一个结构体TupleDI，其中d1为x，i1为-2
    if (!(a > 0.)){return (struct TupleDI){.d1=x, .i1=-2};}
    // 计算变量t的值
    t = p + q - 1.0;
    // 如果fabs(t)不在e的范围内，返回一个结构体TupleDI，其中d1为x，i1为-4
    if (!(fabs(t) <= e)) {return (struct TupleDI){.d1=x, .i1=-4};}
    // 将变量ierr置为0
    ierr = 0;
    // 如果p等于0，返回一个结构体TupleDI，其中d1为x，i1为0
    if (p == 0.0) {return (struct TupleDI){.d1=x, .i1=0};}
    // 如果q等于0，返回一个结构体TupleDI，其中d1为xmax，i1为0
    if (q == 0.0) {return (struct TupleDI){.d1=xmax, .i1=0};}
    // 如果a等于1，根据条件返回一个结构体TupleDI，d1的值根据条件表达式计算得出，i1为0
    if (a == 1.) {
        return (struct TupleDI){.d1=(!(q >= 0.9)? -log(q) : -alnrel(-p)), .i1=0};
    }
    // 如果x0大于0
    if (x0 > 0.0) {
        // 根据p的值确定use_p的值
        use_p = (p > 0.5 ? 0 : 1);
        // 计算am1的值
        am1 = (a - 0.5) - 0.5;
        // 如果条件满足，返回一个结构体TupleDI，其中d1为xn，i1为-8
        if ((use_p ? p : q) <= 1.e10*xmin) {return (struct TupleDI){.d1=xn, .i1=-8};}
    } else if (a <= 1.0) {
        // 计算g的值
        g = cdflib_gamma(a + 1.);
        // 计算qg的值
        qg = q*g;
        // 如果qg等于0，返回一个结构体TupleDI，其中d1为xmax，i1为-8
        if (qg == 0.0) {return (struct TupleDI){.d1=xmax, .i1=-8};}
        // 计算b的值
        b = qg / a;

        // 如果条件满足，进入条件判断
        if ((qg > 0.6*a) || (((a >= 0.3) || (b < 0.35)) && (b >= 0.45))) {
            // 根据b*q的值进入分支
            if (b*q > 1.e-8) {
    // 50
                // 根据p的值进入条件分支，计算xn的值
                if (p <= 0.9) {
                    xn = exp(log(p*g)/a);
                } else {
    // 60
                    xn = exp((alnrel(-q) + gamln1(a))/a);
                }
            } else {
    // 40
                // 计算xn的值
                xn = exp(-(q/a + c));
            }
    // 70
            // 如果xn等于0，返回一个结构体TupleDI，其中d1为x，i1为-3
            if (xn == 0.0) {return (struct TupleDI){.d1=x, .i1=-3};}
            // 计算t的值
            t = 0.5 + (0.5 - xn/(a + 1.));
            // 更新xn的值
            xn /= t;
            // 设定use_p的值为1
            use_p = 1;
            // 计算am1的值
            am1 = (a - 0.5) - 0.5;
            // 如果条件满足，返回一个结构体TupleDI，其中d1为xn，i1为-8
            if ((use_p ? p : q) <= 1.e10*xmin) {return (struct TupleDI){.d1=xn, .i1=-8};}
        } else if ((a >= 0.3) || (b < 0.35)) {
    // 10
            // 如果b等于0，返回一个结构体TupleDI，其中d1为xmax，i1为-8
            if (b == 0.0) {return (struct TupleDI){.d1=xmax, .i1=-8};}
            // 计算y的值
            y = -log(b);
            // 计算s的值
            s = 0.5 + (0.5 - a);
            // 计算z的值
            z = log(y);
            // 计算t的值
            t = y - s*z;
            // 根据b的值进入条件判断
            if (b < 0.15) {
    // 20
                // 如果b小于等于0.01，计算xn的值
                if (b <= 0.01) {
                    xn = gaminv_helper_30(a, s, y, z);
                    // 根据条件返回一个结构体TupleDI，其中d1为xn，i1为0
                    if ((a <= 1.) || (b > bmin[iop])) {
                        return (struct TupleDI){.d1=xn, .i1=0};
                    }
                }
                // 计算u的值
                u = ((t+2.*(3.-a))*t + (2.-a)* (3.-a))/((t+ (5.-a))*t+2.);
                // 计算xn的值
                xn = y - s*log(t) - log(u);
            } else {
                // 计算xn的值
                xn = y - s*log(t)-log(1.+s/(t+1.));
            }
            // 220
            // 设定use_p的值为0
            use_p = 0;
            // 计算am1的值
            am1 = (a - 0.5) - 0.5;
            // 如果q小于等于1.e10*xmin，返回一个结构体TupleDI，其中d1为xn，i1为-8
            if (q <= 1.e10*xmin) {return (struct TupleDI){.d1=xn, .i1=-8};}
        } else {
            // 计算t的值
            t = exp(-(b+c));
            // 计算u的值
            u = t*exp(t);
            // 计算xn的值
            xn = t*exp(u);

            // 160
            // 设定use_p的值为1
            use_p = 1;
            // 计算am1的值
            am1 = (a - 0.5) - 0.5;
            // 如果p小于等于1.e10*xmin，返回一个结构体TupleDI，其中d1为xn，i1为-8
            if (p <= 1.e10*xmin) {return (struct TupleDI){.d1=xn, .i1=-8};}
        }

    }
    // Schroder迭代使用P或Q
    for (ierr = 0; ierr < 20; ierr++)
    {
        // 如果 a 大于 amax，则执行以下操作
        if (a > amax) {
            // 计算 d 的值
            d = 0.5 + (0.5 - xn / a);
            // 如果 d 的绝对值小于等于 e2，则返回一个特定的结构体元组
            if (fabs(d) <= e2) {return (struct TupleDI){.d1=xn, .i1=-8};}
        }
        // 调用 gratio 函数，获取 pn 和 qn 的值
        struct TupleDD pnqn = gratio(a, xn, 0);
        pn = pnqn.d1; // 设置 pn 的值
        qn = pnqn.d2; // 设置 qn 的值
        // 如果 pn 或 qn 为 0，则返回一个特定的结构体元组
        if ((pn == 0.) || (qn == 0.)) {return (struct TupleDI){.d1=xn, .i1=-8};}
        // 计算 r 的值
        r = rcomp(a, xn);
        // 如果 r 的值为 0，则返回一个特定的结构体元组
        if (r == 0.) {return (struct TupleDI){.d1=xn, .i1=-8};}
        // 根据 use_p 的条件选择计算 t 的不同方式
        t =  (use_p ? (pn-p)/r : (q-qn)/r);
        // 计算 w 的值
        w = 0.5 * (am1 - xn);
    
        // 如果 t 的绝对值小于等于 0.1，并且 w*t 的绝对值也小于等于 0.1
        if ((fabs(t) <= 0.1) && (fabs(w*t) <= 0.1)) {
            // 计算 h 的值
            h = t * (1. + w*t);
            // 计算 x 的值
            x = xn * (1. - h);
            // 如果 x 小于等于 0，则返回一个特定的结构体元组
            if (x <= 0.) {return (struct TupleDI){.d1=x, .i1=-7};}
            // 如果 w 的绝对值大于等于 1，并且 fabs(w)*t*t 小于等于 eps，则返回一个特定的结构体元组
            if ((fabs(w) >= 1.) && (fabs(w)*t*t <= eps)) {
                return (struct TupleDI){.d1=x, .i1=ierr};
            }
            // 计算 d 的绝对值
            d = fabs(h);
        } else {
            // 计算 x 的值
            x = xn * (1. - t);
            // 如果 x 小于等于 0，则返回一个特定的结构体元组
            if (x <= 0.) {return (struct TupleDI){.d1=x, .i1=-7};}
            // 计算 d 的绝对值
            d = fabs(t);
        }
    
        // 更新 xn 的值为 x
        xn = x;
    
        // 如果 d 小于等于 tol
        if (d <= tol) {
            // 计算 act_val 和 tol_val 的值
            act_val = (use_p ? fabs(pn-p) : fabs(qn-q));
            tol_val = tol*(use_p ? p : q);
            // 如果 d 小于等于 eps，或者 act_val 小于等于 tol_val，则返回一个特定的结构体元组
            if ((d <= eps) || (act_val <= tol_val)){
                return (struct TupleDI){.d1=x, .i1=ierr};
            }
        }
    }
    // 默认返回一个特定的结构体元组
    return (struct TupleDI){.d1=x, .i1=-6};
double gaminv_helper_30(double a, double s, double y, double z)
{
    // 定义局部变量用于存储中间计算结果
    double c1, c2, c3, c4, c5;
    // 计算 c1 到 c5 的值，用于后续的数学运算
    c1 = -s*z;
    c2 = -s*(1. + c1);
    c3 = s*((0.5*c1+ (2.-a))*c1 + (2.5-1.5*a));
    c4 = -s*(((c1/3. + (2.5-1.5*a))*c1 + ((a-6.)*a+7.))*c1 + ((11.*a-46.)*a+47.)/6.);
    c5 = -s*((((-c1/4.+ (11.*a-17.)/6.
               )*c1+ ((-3.*a+13.)*a-13.)
              )*c1 + 0.5*(((2.*a-25.)*a+72.)*a-61.)
             )*c1+ (((25.*a-195.)*a+477.)*a-379.)/12.);
    // 返回最终的数值计算结果
    return ((((c5/y+c4)/y+c3)/y+c2)/y+c1) + y;
}



double gamln(double a)
{
    // 对于正值 a，计算 ln(gamma(a)) 的值

    // 定义常量和变量
    double t, w, d = .418938533204673;
    int i,n;
    // 预定义的常量数组
    const double c[6] = {.833333333333333e-01, -.277777777760991e-02,
                         .793650666825390e-03, -.595202931351870e-03,
                         .837308034031215e-03, -.165322962780713e-02};

    // 根据不同情况计算 ln(gamma(a)) 的值
    if (a <= 0.8) { return gamln1(a) - log(a); }

    if (a <= 2.25) {
        // 对于较小的 a，进行特定的变换计算
        t = (a-0.5) - 0.5;
        return gamln1(t);
    }

    if (a < 10) {
        // 对于较小的 a，通过递推的方式计算 ln(gamma(a))
        n = (int)(a - 1.25);
        t = a;
        w = 1.0;
        for (i = 0; i < n; i++)
        {
            t -= 1.0;
            w *= t;
        }
        return gamln1(t-1.) + log(w);
    }

    // 对于较大的 a，使用另一种近似计算公式
    t = pow(1/a, 2);
    w = (((((c[5]*t+c[4])*t+c[3])*t+c[2])*t+c[1])*t+c[0])/a;
    return (d + w) + (a-0.5)*(log(a) - 1.);
}



double gamln1(double a)
{
    // 对于 -0.2 <= A <= 1.25 的范围内，计算 ln(gamma(1 + a)) 的值

    // 定义局部变量
    double bot, top, w, x;

    // 预定义的常量数组
    const double p[7] = { .577215664901533e+00,  .844203922187225e+00,
                         -.168860593646662e+00, -.780427615533591e+00,
                         -.402055799310489e+00, -.673562214325671e-01,
                         -.271935708322958e-02};
    const double q[6] = {.288743195473681e+01, .312755088914843e+01,
                         .156875193295039e+01, .361951990101499e+00,
                         .325038868253937e-01, .667465618796164e-03};
    const double r[6] = {.422784335098467e+00, .848044614534529e+00,
                         .565221050691933e+00, .156513060486551e+00,
                         .170502484022650e-01, .497958207639485e-03};
    const double s[5] = {.124313399877507e+01, .548042109832463e+00,
                         .101552187439830e+00, .713309612391000e-02,
                         .116165475989616e-03};

    // 对于不同情况，计算 ln(gamma(1 + a)) 的近似值
    if (a < 0.6) {
        top = ((((((p[6]
                   )*a+p[5]
                  )*a+p[4]
                 )*a+p[3]
                )*a+p[2]
               )*a+p[1]
              )*a+p[0];
        bot = ((((((q[5]
                   )*a+q[4]
                  )*a+q[3]
                 )*a+q[2]
                )*a+q[1]
               )*a+q[0]
              )*a+1.;
        w = top/bot;
        return -a*w;


这些代码段中的注释解释了每个函数和变量的作用，以及在函数中发生的主要计算步骤。
    } else {
        // 计算 x = (a - 0.5) - 0.5
        x = (a - 0.5) - 0.5;
        // 计算顶部系数多项式 top = (((((r[5]*x + r[4])*x + r[3])*x + r[2])*x + r[1])*x + r[0])
        top = (((((r[5] * x + r[4]) * x + r[3]) * x + r[2]) * x + r[1]) * x + r[0];
        // 计算底部系数多项式 bot = (((((s[4]*x + s[3])*x + s[2])*x + s[1])*x + s[0])*x + 1.)
        bot = (((((s[4] * x + s[3]) * x + s[2]) * x + s[1]) * x + s[0]) * x + 1.;
        // 计算 w = top / bot
        w = top / bot;
        // 返回 x * w 作为结果
        return x * w;
    }
}



double cdflib_gamma(double a)
{
    // Gamma 函数的实际参数求值
    //
    // 当 Gamma 函数无法计算时，Gamma(a) 被赋值为 0.

    double bot, g, lnx, t, top, w, z, result;
    int i, j, m, n;
    double s = 0.0;
    double d = 0.5*(log(2.*PI) - 1);
    double x = a;
    double p[7] = {.539637273585445e-03, .261939260042690e-02,
                   .204493667594920e-01, .730981088720487e-01,
                   .279648642639792e+00, .553413866010467e+00,
                   1.0};
    double q[7] = {-.832979206704073e-03, .470059485860584e-02,
                   .225211131035340e-01, -.170458969313360e+00,
                   -.567902761974940e-01, .113062953091122e+01,
                   1.0};
    double r[5] = {.820756370353826e-03, -.595156336428591e-03,
                   .793650663183693e-03, -.277777777770481e-02,
                   .833333333333333e-01};

    result = 0.0;
    if (fabs(a) < 15) {
        t = 1.0;
        m = (int)(a) - 1;
        if (m > 0) {
            for (j = 0; j < m; j++)
            {
                x -= 1.0;
                t *= x;
            }
            x -= 1.0;
        } else if (m == 0) {
            x -= 1.0;
        } else {
            t = a;
            if (a <= 0.) {
                m = -m - 1;
                if (m != 0.) {
                    for (j = 0; j < m; j++)
                    {
                        x += 1.0;
                        t *= x;
                    }
                }
                x += 0.5;
                x += 0.5;
                t *= x;

                if (t == 0.) { return result; }
            }
            if (fabs(t) < 1e-30) {
                if (fabs(t)*spmpar[2] <= 1.0001) { return result; }
                return 1./t;
            }
        }
        top = p[0];
        bot = q[0];
        for (i = 1; i < 7; i++)
        {
            top *= x;
            top += p[i];
            bot *= x;
            bot += q[i];
        }
        result = top / bot;
        return (a < 1.0 ? result/t : result*t);
    }

    if (fabs(a) >= 1.e3) { return result; }

    if (a <= 0.0) {
        x = -a;
        n = (int)x;
        t = x - n;
        if (t > 0.9) {
            t = 1. - t;
        }
        s = sin(PI*t) / PI;
        if (n % 2 == 0) {
            s = -s;
        }
        if (s == 0.0) { return result; }
    }
    t = pow(1 / x, 2);
    g = ((((r[0]*t+r[1])*t+r[2])*t+r[3])*t+r[4]) / x;
    lnx = log(x);
    z = x;
    g = (d + g) + (z -0.5)*(lnx - 1.);
    w = g;
    t = g - w;
    if (w > 0.99999*709) { return result; }
    result = exp(w)*(1. + t);
    return (a < 0.0 ? (1. / (result * s)) / x : result);
}



struct TupleDD grat1(double a, double x, double r, double eps)
{
    // 不完全 Gamma 函数比率函数 p(a,x) 和 q(a,x) 的求值
    //
    // 假定 a <= 1. Eps 是要使用的容差值。
    // 输入参数 r 的值为 e**(-x)*x**a/gamma(a).
    
    double a2n, a2nm1, am0, an, an0, b2n, b2nm1, c, cma, g, h, j, l;
    double p, q, ssum, t, tol, w, z;

    // 如果 a*x 等于 0
    if (a*x == 0.) {
        // 如果 x 大于 a，则返回 (1.0, 0.0)
        if (x > a) {
            return (struct TupleDD){.d1=1.0, .d2=0.0};
        } else { // 否则返回 (0.0, 1.0)
            return (struct TupleDD){.d1=0.0, .d2=1.0};
        }
    }

    // 如果 a 等于 0.5
    if (a == 0.5) {
        // 如果 x 小于 0.25，则计算 sqrt(x) 的误差函数并返回 (cdflib_erf(sqrt(x)), 0.5 + (0.5 - p))
        if (x < 0.25) {
            p = cdflib_erf(sqrt(x));
            return (struct TupleDD){.d1=p, .d2=0.5 + (0.5 - p)};
        } else { // 否则计算 sqrt(x) 的余误差函数并返回 (0.5 + (0.5 - q), q)
            q = erfc1(0, sqrt(x));
            return (struct TupleDD){.d1=0.5 + (0.5 - q), .d2=q};
        }
    }

    // 如果 x 小于 1.1
    if (x < 1.1) {
        //
        // Taylor series for p(a,x)/x**a
        //
        an = 3.0;
        c = x;
        ssum = x / (a + 3.);
        tol = 0.1*eps / (a + 1.);
        // 开始循环直到条件不满足
        while (1) {
            an += 1;
            c *= -(x / an);
            t = c / (a + an);
            ssum += t;
            // 如果误差 t 的绝对值小于等于 tol，则退出循环
            if (fabs(t) <= tol) { break; }
        }
        // 计算 p(a,x) 的近似值 j
        j = a*x*((ssum/6. - 0.5/(a+2.))*x + 1./(a+1.));

        // 计算 z = a * log(x) 和 gamma 函数的补函数 h
        z = a * log(x);
        h = gam1(a);
        g = 1. + h;

        // 如果条件满足，则计算 p 和 q，并返回 (p, q)
        if (((x >= 0.25) && (a >= x /2.59)) || ((x < 0.25) && (z <= -0.13394))) {
            w = exp(z);
            p = w*g*(0.5 + (0.5 - j));
            q = 0.5 + (0.5 - p);
            return (struct TupleDD){.d1=p, .d2=q};
        } else { // 否则计算 l = rexp(z)，并根据 q 的值返回 (1.0, 0.0) 或者 (p, q)
            l = rexp(z);
            w = 0.5 + (0.5 + l);
            q = (w*j - l)*g - h;
            if (q < 0.0) {
                return (struct TupleDD){.d1=1.0, .d2=0.0};
            }
            p = 0.5 + (0.5 - q);
            return (struct TupleDD){.d1=p, .d2=q};
        }
    }
    
    // 连分数展开
    a2nm1 = 1.0;
    a2n = 1.0;
    b2nm1 = x;
    b2n = x + (1. - a);
    c = 1.0;
    // 开始循环直到条件不满足
    while (1) {
        a2nm1 = x*a2n + c*a2nm1;
        b2nm1 = x*b2n + c*b2nm1;
        am0 = a2nm1/b2nm1;
        c = c + 1.;
        cma = c - a;
        a2n = a2nm1 + cma*a2n;
        b2n = b2nm1 + cma*b2n;
        an0 = a2n/b2n;
        // 如果绝对误差 fabs(an0-am0) 小于 eps*an0，则退出循环
        if (!(fabs(an0-am0) >= eps*an0)) { break; }
    }
    // 计算 q = r*an0 和 p = 0.5 + (0.5 - q)，并返回 (p, q)
    q = r*an0;
    p = 0.5 + (0.5 - q);
    return (struct TupleDD){.d1=p, .d2=q};
    //    Evaluation of the incomplete gamma ratio functions
    //                    P(a,x) and Q(a,x)
    //
    //                    ----------
    //
    //    It is assumed that a and x are nonnegative, where a and x
    //    Are not both 0.
    //
    //    Ans and qans are variables. Gratio assigns ans the value
    //    P(a,x) and qans the value q(a,x). Ind may be any integer.
    //    If ind = 0 then the user is requesting as much accuracy as
    //    Possible (up to 14 significant digits). Otherwise, if
    //    Ind = 1 then accuracy is requested to within 1 unit of the
    //    6-Th significant digit, and if ind .ne. 0,1 Then accuracy
    //    Is requested to within 1 unit of the 3rd significant digit.
    //
    //    Error return ...
    //    Ans is assigned the value 2 when a or x is negative,
    //    When a*x = 0, or when p(a,x) and q(a,x) are indeterminant.
    //    P(a,x) and q(a,x) are computationally indeterminant when
    //    X is exceedingly close to a and a is extremely large.
    struct TupleDD gratio(double a, double x, int ind)
    {
        double d10 = -.185185185185185e-02;  // Coefficient initialization
        double d20 = .413359788359788e-02;
        double d30 = .649434156378601e-03;
        double d40 = -.861888290916712e-03;
        double d50 = -.336798553366358e-03;
        double d60 = .531307936463992e-03;
        double d70 = .344367606892378e-03;
        double alog10 = log(10);  // Logarithm of 10
        double rt2pi = sqrt(1. / (2.*PI));  // Square root of 1 / (2 * PI)
        double rtpi = sqrt(PI);  // Square root of PI
        double eps = spmpar[0];  // Machine epsilon value
        double acc, a2n, a2nm1, am0, amn, an, an0, ans, apn, b2n, b2nm1;
        double c, c0, c1, c2, c3, c4, c5, c6, cma, e0, g, h, j, l, qans, r;
        double rta, rtx, s, ssum, t, t1, tol, twoa, u, w, x0, y, z;
        int i, m, n, last_entry;

        double wk[20] = {0.0};  // Array for workspace initialization
        double acc0[3] = {5.e-15, 5.e-7, 5.e-4};  // Accuracy thresholds
        double big[3] = {20., 14., 10.};  // Big values for various calculations
        double e00[3] = {.00025, .025, .14};  // Values for E00
        double x00[3] = {31., 17., 9.7};  // Values for X00
        double d0[13] = {.833333333333333e-01, -.148148148148148e-01,
                         .115740740740741e-02, .352733686067019e-03,
                         -.178755144032922e-03, .391926317852244e-04,
                         -.218544851067999e-05, -.185406221071516e-05,
                         .829671134095309e-06, -.176659527368261e-06,
                         .670785354340150e-08, .102618097842403e-07,
                         -.438203601845335e-08};  // Coefficients for d0
        double d1[12] = {-.347222222222222e-02, .264550264550265e-02,
                         -.990226337448560e-03, .205761316872428e-03,
                         -.401877572016461e-06, -.180985503344900e-04,
                         .764916091608111e-05, -.161209008945634e-05,
                         .464712780280743e-08, .137863344691572e-06,
                         -.575254560351770e-07, .119516285997781e-07};  // Coefficients for d1

        // Function body continues...
    // 初始化一些预先定义的数组，这些数组用于后续的数学运算
    double d2[10] = {-.268132716049383e-02, .771604938271605e-03,
                     .200938786008230e-05, -.107366532263652e-03,
                     .529234488291201e-04, -.127606351886187e-04,
                     .342357873409614e-07, .137219573090629e-05,
                     -.629899213838006e-06, .142806142060642e-06};
    double d3[8] = {.229472093621399e-03, -.469189494395256e-03,
                    .267720632062839e-03, -.756180167188398e-04,
                    -.239650511386730e-06, .110826541153473e-04,
                    -.567495282699160e-05, .142309007324359e-05};
    double d4[6] = {.784039221720067e-03, -.299072480303190e-03,
                    -.146384525788434e-05, .664149821546512e-04,
                    -.396836504717943e-04, .113757269706784e-04};
    double d5[4] = {-.697281375836586e-04, .277275324495939e-03,
                    -.199325705161888e-03, .679778047793721e-04};
    double d6[2] = {-.592166437353694e-03, .270878209671804e-03};

    // 检查 a 和 x 是否小于零，若是，则返回特定的结构体 TupleDD
    if ((!(a >= 0.0)) || (!(x >= 0.0))) {
        return (struct TupleDD){.d1=2.0, .d2=0.0};
    }
    // 检查 a 和 x 是否同时为零，若是，则返回特定的结构体 TupleDD
    if ((a == 0.0) && (x == 0.0)) {
        return (struct TupleDD){.d1=2.0, .d2=0.0};
    }

    // 检查 a 与 x 的乘积是否为零，若是，则根据 x 与 a 的大小关系返回不同的结构体 TupleDD
    if (a*x == 0.0) {
        if (x > a) {
            return (struct TupleDD){.d1=1.0, .d2=0.0};
        } else {
            return (struct TupleDD){.d1=0.0, .d2=1.0};
        }
    }

    // 检查 ind 是否不为 0 或 1，若不是，则将 ind 设为 2
    if ((!(ind == 0)) && (!(ind == 1))) {
        ind = 2;
    }

    // 初始化一些变量，使用 fmax 函数确保 acc 大于等于 acc0[ind] 和 eps 中的较大值
    acc = fmax(acc0[ind], eps);
    e0 = e00[ind];
    x0 = x00[ind];

    // 如果 a 等于 0.5，则执行以下逻辑
    if (a == 0.5) {
        // 390
        // 根据 x 的值执行不同的数学运算，返回结果结构体 TupleDD
        if (x >= 0.5) {
            qans = erfc1(0, sqrt(x));
            ans = 0.5 + (0.5 - qans);
        } else {
            ans = cdflib_erf(sqrt(x));
            qans = 0.5 + (0.5 - ans);
        }
        return (struct TupleDD){.d1=ans, .d2=qans};
    }

    // 如果 x 小于 1.1，则执行以下逻辑
    if (x < 1.1) {
        // 160
        // TAYLOR SERIES FOR P(A,X)/X**A
        //
        // 初始化变量，使用 while 循环计算特定数学公式的近似值
        an = 3.0;
        c = x;
        ssum = x / (a + 3.);
        tol = 3.*acc / (a + 1.);
        while (1) {
            an += 1.0;
            c *= -(x / an);
            t = c / (a + an);
            ssum += t;
            // 判断是否达到精度要求，若是则退出循环
            if (!(fabs(t) > tol)) { break; }
        }
        j = a*x*((ssum / 6. - 0.5 / (a + 2.))*x + 1./(a + 1.));
        z = a*log(x);
        h = gam1(a);
        g = 1. + h;

        // 根据不同的条件选择不同的数学计算方法，返回结果结构体 TupleDD
        if (((x < 0.25) && (z > -0.13394)) || (a < x/2.59)) {
            l = rexp(z);
            w = 0.5 + (0.5 + l);
            qans = (w*j - l)*g - h;
            if (qans < 0.) {
                return (struct TupleDD){.d1=1.0, .d2=0.0};
            }
            ans = 0.5 + (0.5 - qans);
        } else {
            w = exp(z);
            ans = w*g*(0.5 + (0.5 - j));
            qans = 0.5 + (0.5 - ans);
        }
        return (struct TupleDD){.d1=ans, .d2=qans};
    }

    // 若前面的条件都不符合，则执行以下逻辑
    t1 = a*log(x) - x;
    u = a*exp(t1);
    if (u == 0.0) {
        return (struct TupleDD){.d1=1.0, .d2=0.0};
    }

    r = u * (1. + gam1(a));
    // 250
    // CONTINUED FRACTION EXPANSION
    //
    tol = (5.0*eps > acc ? 5.0*eps : acc);
    # 设置容差(tol)，如果5倍的机器精度eps大于acc，则tol为5倍eps，否则tol为acc

    a2nm1 = 1.0;
    # 初始化 a2nm1 为 1.0

    a2n = 1.0;
    # 初始化 a2n 为 1.0

    b2nm1 = x;
    # 初始化 b2nm1 为 x

    b2n = x + (1.0 - a);
    # 初始化 b2n 为 x + (1.0 - a)

    c = 1.0;
    # 初始化 c 为 1.0

    while (1) {
        # 进入无限循环

        a2nm1 = x*a2n + c*a2nm1;
        # 更新 a2nm1：a2nm1 = x * a2n + c * a2nm1

        b2nm1 = x*b2n + c*b2nm1;
        # 更新 b2nm1：b2nm1 = x * b2n + c * b2nm1

        am0 = a2nm1/b2nm1;
        # 计算 am0：am0 = a2nm1 / b2nm1

        c += 1.0;
        # c 自增 1.0

        cma = c - a;
        # 计算 cma：cma = c - a

        a2n = a2nm1 + cma*a2n;
        # 更新 a2n：a2n = a2nm1 + cma * a2n

        b2n = b2nm1 + cma*b2n;
        # 更新 b2n：b2n = b2nm1 + cma * b2n

        an0 = a2n/b2n;
        # 计算 an0：an0 = a2n / b2n

        if (!(fabs(an0-am0) >= tol*an0)) { break; }
        # 如果 |an0 - am0| 不大于 tol * an0，则跳出循环
    }

    qans = r*an0;
    # 计算 qans：qans = r * an0

    ans = 0.5 + (0.5 - qans);
    # 计算 ans：ans = 0.5 + (0.5 - qans)

    return (struct TupleDD){.d1=ans, .d2=qans};
    # 返回一个结构体 TupleDD，包含 ans 和 qans 作为其成员
double psi(double xx)
{
    //                    Evaluation of the digamma function
    //
    //                          -----------
    //
    //    Psi(xx) is assigned the value 0 when the digamma function cannot
    //    be computed.
    //
    //    The main computation involves evaluation of rational chebyshev
    //    approximations published in math. Comp. 27, 123-127(1973) By
    //    cody, strecok and thacher.
    //
    //    ----------------------------------------------------------------
    //    Psi was written at Argonne National Laboratory for the FUNPACK
    //    package of special function subroutines. Psi was modified by
    //    A.H. Morris (nswc).

    double aug, den, dx0, sgn, upper, w, x, xmax1, xmx0, xsmall, z;
    double p1[7] = {0.895385022981970e-02, 0.477762828042627e+01,
                    0.142441585084029e+03, 0.118645200713425e+04,
                    0.363351846806499e+04, 0.413810161269013e+04,
                    0.130560269827897e+04};
    double q1[6] = {0.448452573429826e+02, 0.520752771467162e+03,
                    0.221000799247830e+04, 0.364127349079381e+04,
                    0.190831076596300e+04, 0.691091682714533e-05};
    double p2[4] = {-0.212940445131011e+01, -0.701677227766759e+01,
                    -0.448616543918019e+01, -0.648157123766197e+00};
    double q2[4] = {0.322703493791143e+02, 0.892920700481861e+02,
                    0.546117738103215e+02, 0.777788548522962e+01};
    int nq, i;
    dx0 = 1.461632144968362341262659542325721325;
    xmax1 = 4503599627370496.0;
    xsmall = 1e-9;
    x = xx;
    aug = 0.0;

    if (x < 0.5) {
        // Case when x is less than 0.5
        if (fabs(x) <= xsmall) {
            // Handle special case when x is close to zero
            if (x == 0.) {
                return 0.0;  // Return 0 when x is exactly 0
            }
            aug = -1./x;  // Compute augmentation for small x
        } else {
            // Compute augmentation using series approximation
            w = -x;
            sgn = PI / 4;
            if (w <= 0.) {
                w = -w;
                sgn = -sgn;
            }
            if (w >= xmax1) {
                return 0.0;  // Return 0 if w exceeds maximum value
            }
            w -= (int)w;
            nq = (int)(w*4.0);
            w = 4.*(w - 0.25*nq);

            if (nq % 2 == 1) {
                w = 1. - w;
            }
            z = (PI / 4.)*w;

            if ((nq / 2) % 2 == 1) {
                sgn = -sgn;
            }
            if ((((nq + 1) / 2) % 2) == 1) {
                aug = sgn * (tan(z)*4.);  // Compute augmentation using tangent
            } else {
                if (z == 0.) {
                    return 0.0;
                }
                aug = sgn * (4./tan(z));  // Compute augmentation using tangent
            }
        }
        x = 1 - x;
    }
    # 如果输入值 x 小于等于 3.0，则执行以下代码块
    if (x <= 3.0) {
        # 分母初始化为 x
        den = x;
        # 分子初始化为 p1[0] * x
        upper = p1[0]*x;
        # 循环计算 den 和 upper
        for (i = 0; i < 5; i++)
        {
            # 更新 den
            den = (den + q1[i])*x;
            # 更新 upper
            upper = (upper + p1[i+1])*x;
        }
        # 计算最终的 den，即 (upper + p1[6]) / (den + q1[5])
        den = (upper + p1[6]) / (den + q1[5]);
        # 计算 xmx0，即 x - dx0
        xmx0 = x - dx0;
        # 返回结果 den * xmx0 + aug
        return (den * xmx0) + aug;
    } else {
        # 如果输入值 x 大于 3.0，则执行以下代码块
        # 检查 x 是否小于 xmax1
        if (x < xmax1) {
            # 计算 w，即 1 / (x*x)
            w = 1. / (x*x);
            # 分母初始化为 w
            den = w;
            # 分子初始化为 p2[0] * w
            upper = p2[0]*w;

            # 循环计算 den 和 upper
            for (i = 0; i < 3; i++) {
                # 更新 den
                den = (den + q2[i])*w;
                # 更新 upper
                upper = (upper + p2[i+1])*w;
            }
            # 更新 aug，即 aug += upper / (den + q2[3]) - 0.5/x
            aug += upper / (den + q2[3]) - 0.5/x;
        }
        # 返回结果 aug + log(x)
        return aug + log(x);
    }
// 开始定义函数 stvaln，用于计算正态分布的逆函数的初始值
double stvaln(double p)
{
    // 这个函数是为了计算正态分布逆函数的起始值
    // 使用牛顿-拉夫逊方法进行计算

    // 声明函数使用的变量和常量
    double a = .914041914819518e-09;
    double b = .238082361044469e-01;
    double p0 = .333333333333333;
    double p1 = -.224696413112536;
    double p2 = .620886815375787e-02;
    double q0 = -.499999999085958;
    double q1 = .107141568980644;
    double q2 = -.119041179760821e-01;
    double q3 = .595130811860248e-03;
    double r2pi = sqrt(1. / (2.0 * PI));

    // 声明函数内部使用的变量
    double t, t1, u, w, w1;

    // 根据输入参数 p 的绝对值大小，选择不同的计算路径
    if (fabs(p) <= 0.15) {
        // 当 p 的绝对值小于等于 0.15 时的计算路径
        t = p * ((p1 * p + p0) * p + 1.) / ((((q3 * p + q2) * p + q1) * p + q0) * p + 1.);
        return t;
    } else {
        // 当 p 的绝对值大于 0.15 时的计算路径
        u = exp(p);
        if (p > 0.) {
            w = u * (0.5 + (0.5 - 1. / u));
        } else {
            w = (u - 0.5) - 0.5;
        }
        return w;
    }
}
    // 定义两个双精度浮点数变量 y 和 z
    double y, z;
    // 定义用于有理函数计算的系数数组 xnum 和 xden
    double xnum[5] = {-0.322232431088, -1.000000000000, -0.342242088547,
                      -0.204231210245e-1, -0.453642210148e-4};
    double xden[5]  = {0.993484626060e-1, 0.588581570495, 0.531103462366,
                       0.103537752850, 0.38560700634e-2};
    // 根据概率 p 计算 z 值，如果 p 大于 0.5，使用 1 - p，否则使用 p
    z = (p > 0.5 ? 1.0 - p : p);
    // 计算 y 值，y = sqrt(-2.0 * log(z))
    y = sqrt(-2.0 * log(z));
    // 根据 Kennedy 和 Gentle 的有理函数方法计算 z 值
    z = y + (devlpl(xnum, 5, y) / devlpl(xden, 5, y));
    // 如果 p 大于 0.5，返回 z，否则返回 -z
    return (p > 0.5 ? z : -z);
}


注释：

# 这行代码表示一个代码块的结束，对应于某个函数或条件语句的结尾。
```