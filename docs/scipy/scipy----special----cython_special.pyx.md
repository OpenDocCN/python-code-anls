# `D:\src\scipysrc\scipy\scipy\special\cython_special.pyx`

```
"""
.. highlight:: cython

Cython API for special functions
================================

Scalar, typed versions of many of the functions in ``scipy.special``
can be accessed directly from Cython; the complete list is given
below. Functions are overloaded using Cython fused types so their
names match their Python counterpart. The module follows the following
conventions:

- If a function's Python counterpart returns multiple values, then the
  function returns its outputs via pointers in the final arguments.
- If a function's Python counterpart returns a single value, then the
  function's output is returned directly.

The module is usable from Cython via::

    cimport scipy.special.cython_special

Error handling
--------------

Functions can indicate an error by returning ``nan``; however they
cannot emit warnings like their counterparts in ``scipy.special``.

Available functions
-------------------

- :py:func:`~scipy.special.voigt_profile`::

        double voigt_profile(double, double, double)

    Computes the Voigt profile function for given parameters.

- :py:func:`~scipy.special.agm`::

        double agm(double, double)

    Computes the arithmetic-geometric mean of two numbers.

- :py:func:`~scipy.special.airy`::

        void airy(double, double *, double *, double *, double *)
        void airy(double complex, double complex *, double complex *, double complex *, double complex *)

    Computes Airy functions Ai and Bi for real and complex arguments.

- :py:func:`~scipy.special.airye`::

        void airye(double complex, double complex *, double complex *, double complex *, double complex *)
        void airye(double, double *, double *, double *, double *)

    Computes exponentially scaled Airy functions Ai and Bi for real and complex arguments.

- :py:func:`~scipy.special.bdtr`::

        double bdtr(double, double, double)
        double bdtr(double, long, double)

    Computes the cumulative distribution function of the binomial distribution.

- :py:func:`~scipy.special.bdtrc`::

        double bdtrc(double, double, double)
        double bdtrc(double, long, double)

    Computes the complement of the cumulative distribution function of the binomial distribution.

- :py:func:`~scipy.special.bdtri`::

        double bdtri(double, double, double)
        double bdtri(double, long, double)

    Computes the inverse of the cumulative distribution function of the binomial distribution.

- :py:func:`~scipy.special.bdtrik`::

        double bdtrik(double, double, double)

    Computes the iterative solution of the cumulative distribution function of the binomial distribution.

- :py:func:`~scipy.special.bdtrin`::

        double bdtrin(double, double, double)

    Computes the integral of the cumulative distribution function of the binomial distribution.

- :py:func:`~scipy.special.bei`::

        double bei(double)

    Computes the Kelvin function bei(x).

- :py:func:`~scipy.special.beip`::

        double beip(double)

    Computes the Kelvin function bei'(x).

- :py:func:`~scipy.special.ber`::

        double ber(double)

    Computes the Kelvin function ber(x).

- :py:func:`~scipy.special.berp`::

        double berp(double)

    Computes the Kelvin function ber'(x).

- :py:func:`~scipy.special.besselpoly`::

        double besselpoly(double, double, double)

    Computes the generalized Bessel polynomial.

- :py:func:`~scipy.special.beta`::

        double beta(double, double)

    Computes the beta function.

- :py:func:`~scipy.special.betainc`::

        float betainc(float, float, float)
        double betainc(double, double, double)

    Computes the incomplete beta function.

- :py:func:`~scipy.special.betaincc`::

        float betaincc(float, float, float)
        double betaincc(double, double, double)

    Computes the complement of the incomplete beta function.

- :py:func:`~scipy.special.betaincinv`::

        float betaincinv(float, float, float)
        double betaincinv(double, double, double)

    Computes the inverse of the incomplete beta function.

"""
# 计算不完全贝塔函数的逆函数
float betainccinv(float, float, float)
double betainccinv(double, double, double)

# 计算对数贝塔函数
double betaln(double, double)

# 计算二项式系数
double binom(double, double)

# 计算Box-Cox变换
double boxcox(double, double)

# 计算Box-Cox变换（增加1后）
double boxcox1p(double, double)

# 计算贝塔分布的累积分布函数的值
double btdtr(double, double, double)

# 计算贝塔分布的累积分布函数的逆函数
double btdtri(double, double, double)

# 计算贝塔分布的累积分布函数的对数
double btdtria(double, double, double)

# 计算贝塔分布的累积分布函数的逆对数
double btdtrib(double, double, double)

# 计算立方根
double cbrt(double)

# 计算卡方分布的累积分布函数的值
double chdtr(double, double)

# 计算卡方分布的补充累积分布函数的值
double chdtrc(double, double)

# 计算卡方分布的逆函数
double chdtri(double, double)

# 计算卡方分布的逆函数的一种变体
double chdtriv(double, double)

# 计算非中心卡方分布的累积分布函数的值
double chndtr(double, double, double)

# 计算非中心卡方分布的自由度
double chndtridf(double, double, double)

# 计算非中心卡方分布的增量累积分布函数的值
double chndtrinc(double, double, double)

# 计算非中心卡方分布的逆函数
double chndtrix(double, double, double)

# 计算角度的余弦值（度数制）
double cosdg(double)

# 计算 x - 1 的余弦值
double cosm1(double)

# 计算角度的余切值（度数制）
double cotdg(double)

# 计算 Dawson 函数
double dawsn(double)
double complex dawsn(double complex)

# 计算椭圆积分第二类
double ellipe(double)

# 计算不完全椭圆积分第二类
double ellipeinc(double, double)

# 计算椭圆函数 Jacobi 的正弦、余弦、振幅和模
void ellipj(double, double, double *, double *, double *, double *)

# 计算完全椭圆积分第二类
double ellipkinc(double, double)

# 计算椭圆积分第二类的增量
double ellipkm1(double)

# 计算完全椭圆积分第二类
double ellipk(double)

# 计算完全椭圆积分第一类
double elliprc(double, double)
double complex elliprc(double complex, double complex)

# 计算椭圆积分第三类
double elliprd(double, double, double)
double complex elliprd(double complex, double complex, double complex)

# 计算椭圆积分第一类
double elliprf(double, double, double)
double complex elliprf(double complex, double complex, double complex)

# 计算椭圆积分第四类
double elliprg(double, double, double)
double complex elliprg(double complex, double complex, double complex)
// 计算 Jacobi 椭圆函数 R_J(a, b, c, x)，返回双精度复数值
double complex elliprj(double complex a, double complex b, double complex c, double complex x)

// 计算熵函数 entropy(x)，返回双精度值
double entr(double x)

// 计算误差函数 erf(z)，其中 z 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex erf(double complex z)
double erf(double z)

// 计算余误差函数 erfc(z)，其中 z 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex erfc(double complex z)
double erfc(double z)

// 计算调整的余误差函数 erfcx(x)，其中 x 可以是双精度复数或双精度数值，返回双精度值或双精度复数值
double erfcx(double x)
double complex erfcx(double complex x)

// 计算调整的误差函数 erfi(x)，其中 x 可以是双精度复数或双精度数值，返回双精度值或双精度复数值
double erfi(double x)
double complex erfi(double complex x)

// 计算误差函数 erf 的反函数，其中 x 是单精度或双精度数值，返回单精度或双精度值
float erfinv(float x)
double erfinv(double x)

// 计算调整的余误差函数 erfc 的反函数，其中 x 是双精度数值，返回双精度值
double erfcinv(double x)

// 计算 Chebyshev 多项式类型 C_n(x)，其中 n 是长整型，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_chebyc(double n, double complex x)
double eval_chebyc(double n, double x)
double eval_chebyc(long n, double x)

// 计算 Chebyshev 多项式类型 S_n(x)，其中 n 是长整型，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_chebys(double n, double complex x)
double eval_chebys(double n, double x)
double eval_chebys(long n, double x)

// 计算 Chebyshev 多项式类型 T_n(x)，其中 n 是长整型，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_chebyt(double n, double complex x)
double eval_chebyt(double n, double x)
double eval_chebyt(long n, double x)

// 计算 Chebyshev 多项式类型 U_n(x)，其中 n 是长整型，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_chebyu(double n, double complex x)
double eval_chebyu(double n, double x)
double eval_chebyu(long n, double x)

// 计算 Gegenbauer 多项式类型 C^(alpha)_n(x)，其中 alpha 和 n 是双精度数值，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_gegenbauer(double alpha, double n, double complex x)
double eval_gegenbauer(double alpha, double n, double x)
double eval_gegenbauer(long alpha, double n, double x)

// 计算广义 Laguerre 多项式 L_n^(alpha)(x)，其中 alpha 和 n 是双精度数值，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_genlaguerre(double alpha, double n, double complex x)
double eval_genlaguerre(double alpha, double n, double x)
double eval_genlaguerre(long alpha, double n, double x)

// 计算 Hermite 多项式 H_n(x)，其中 n 是长整型，x 是双精度数值，返回双精度值
double eval_hermite(long n, double x)

// 计算归一化的 Hermite 多项式 H_n(x)，其中 n 是长整型，x 是双精度数值，返回双精度值
double eval_hermitenorm(long n, double x)

// 计算 Jacobi 多项式 P^(alpha, beta)_n(x)，其中 alpha、beta 和 n 是双精度数值，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_jacobi(double alpha, double beta, double n, double complex x)
double eval_jacobi(double alpha, double beta, double n, double x)
double eval_jacobi(long alpha, double beta, double n, double x)

// 计算 Laguerre 多项式 L_n(x)，其中 n 是长整型，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_laguerre(double n, double complex x)
double eval_laguerre(double n, double x)
double eval_laguerre(long n, double x)

// 计算 Legendre 多项式 P_n(x)，其中 n 是长整型，x 可以是双精度复数或双精度数值，返回双精度复数值或双精度值
double complex eval_legendre(double n, double complex x)
double eval_legendre(double n, double x)
double eval_legendre(long n, double x)
// 计算第一类 Chebyshev 多项式的值
double complex eval_sh_chebyt(double, double complex)
// 计算第一类 Chebyshev 多项式的值
double eval_sh_chebyt(double, double)
// 计算第一类 Chebyshev 多项式的值
double eval_sh_chebyt(long, double)

// 计算第二类 Chebyshev 多项式的值
double complex eval_sh_chebyu(double, double complex)
// 计算第二类 Chebyshev 多项式的值
double eval_sh_chebyu(double, double)
// 计算第二类 Chebyshev 多项式的值
double eval_sh_chebyu(long, double)

// 计算 Jacobi 多项式的值
double complex eval_sh_jacobi(double, double, double, double complex)
// 计算 Jacobi 多项式的值
double eval_sh_jacobi(double, double, double, double)
// 计算 Jacobi 多项式的值
double eval_sh_jacobi(long, double, double, double)

// 计算 Legendre 多项式的值
double complex eval_sh_legendre(double, double complex)
// 计算 Legendre 多项式的值
double eval_sh_legendre(double, double)
// 计算 Legendre 多项式的值
double eval_sh_legendre(long, double)

// 计算指数积分函数的复数形式
double complex exp1(double complex)
// 计算指数积分函数的实数形式
double exp1(double)

// 计算 10 的指数幂
double exp10(double)

// 计算 2 的指数幂
double exp2(double)

// 计算指数积分函数的复数形式
double complex expi(double complex)
// 计算指数积分函数的实数形式
double expi(double)

// 计算 Logistic 函数的值
double expit(double)
// 计算 Logistic 函数的值
float expit(float)
// 计算 Logistic 函数的值
long double expit(long double)

// 计算指数函数减一的复数形式
double complex expm1(double complex)
// 计算指数函数减一的实数形式
double expm1(double)

// 计算修正 Bessel 函数的值
double expn(double, double)
// 计算修正 Bessel 函数的值
double expn(long, double)

// 计算指数相对误差
double exprel(double)

// 计算上不完全贝塞尔函数
double fdtr(double, double, double)

// 计算上完全贝塞尔函数的补函数
double fdtrc(double, double, double)

// 计算上完全贝塞尔函数的逆函数
double fdtri(double, double, double)

// 计算上完全贝塞尔函数的导数
double fdtridfd(double, double, double)

// 计算 Fresnel 积分的实部和虚部
void fresnel(double, double *, double *)
// 计算 Fresnel 积分的实部和虚部
void fresnel(double complex, double complex *, double complex *)

// 计算 Gamma 函数的复数形式
double complex gamma(double complex)
// 计算 Gamma 函数的实数形式
double gamma(double)

// 计算不完全 Gamma 函数
double gammainc(double, double)

// 计算 Gamma 函数的补函数
double gammaincc(double, double)

// 计算 Gamma 函数的补函数的逆函数
double gammainccinv(double, double)

// 计算不完全 Gamma 函数的逆函数
double gammaincinv(double, double)

// 计算 Gamma 函数的自然对数
double gammaln(double)

// 计算 Gamma 函数的符号
double gammasgn(double)

// 计算下不完全贝塞尔函数
double gdtr(double, double, double)

// 计算下完全贝塞尔函数的补函数
double gdtrc(double, double, double)

// 计算下完全贝塞尔函数的逆函数
double gdtria(double, double, double)
// 计算 Generalized Distributions Triangular 和其导数之一
double gdtrib(double, double, double)

// 计算 Generalized Distributions Triangular 和其导数之一
double gdtrix(double, double, double)

// 计算第一类修正 Hankel 函数 H1^{(1)}(x)
double complex hankel1(double, double complex)

// 计算第一类修正 Hankel 函数 H1^{(1)}(x)
double complex hankel1e(double, double complex)

// 计算第二类修正 Hankel 函数 H1^{(2)}(x)
double complex hankel2(double, double complex)

// 计算第二类修正 Hankel 函数 H1^{(2)}(x)
double complex hankel2e(double, double complex)

// 计算 Huber 函数
double huber(double, double)

// 计算超几何函数 0F1
double complex hyp0f1(double, double complex)
// 计算超几何函数 0F1
double hyp0f1(double, double)

// 计算超几何函数 1F1
double hyp1f1(double, double, double)
// 计算超几何函数 1F1
double complex hyp1f1(double, double, double complex)

// 计算超几何函数 2F1
double hyp2f1(double, double, double, double)
// 计算超几何函数 2F1
double complex hyp2f1(double, double, double, double complex)

// 计算 Tricomi 超几何函数 U(a, b, x)
double hyperu(double, double, double)

// 计算第一类修正 Bessel 函数 I0(x)
double i0(double)

// 计算第一类修正 Bessel 函数 I0e(x)
double i0e(double)

// 计算第一类修正 Bessel 函数 I1(x)
double i1(double)

// 计算第一类修正 Bessel 函数 I1e(x)
double i1e(double)

// 计算 Box-Cox 反变换
double inv_boxcox(double, double)

// 计算 Box-Cox 反变换
double inv_boxcox1p(double, double)

// 计算对应于修改 Struve 函数 H0'(x) 的 I0 和 K0 的积分
void it2i0k0(double, double *, double *)

// 计算对应于修改 Struve 函数 H0'(x) 的 J0 和 Y0 的积分
void it2j0y0(double, double *, double *)

// 计算修改 Struve 函数 H0'(x)
double it2struve0(double)

// 计算 Airy 函数 Ai 和 Bi 及其导数
void itairy(double, double *, double *, double *, double *)

// 计算修正 Bessel 函数 I0 和 K0 的积分
void iti0k0(double, double *, double *)

// 计算 Bessel 函数 J0 和 Y0 的积分
void itj0y0(double, double *, double *)

// 计算修改 Struve 函数 H0'(x) 的模
double itmodstruve0(double)

// 计算 Struve 函数 H0'(x)
double itstruve0(double)

// 计算修正 Bessel 函数 Iv(x)
double complex iv(double, double complex)
// 计算修正 Bessel 函数 Iv(x)
double iv(double, double)

// 计算修正 Bessel 函数 I've(x)
double complex ive(double, double complex)
// 计算修正 Bessel 函数 I've(x)
double ive(double, double)

// 计算 Bessel 函数 J0(x)
double j0(double)

// 计算 Bessel 函数 J1(x)
double j1(double)

// 计算 Bessel 函数 Jv(x)
double complex jv(double, double complex)
// 计算 Bessel 函数 Jv(x)
double jv(double, double)

// 计算 Bessel 函数 J've(x)
double complex jve(double, double complex)
// 计算 Bessel 函数 J've(x)
double jve(double, double)

// 计算修正 Bessel 函数 K0(x)
double k0(double)

// 计算修正 Bessel 函数 K0e(x)
double k0e(double)
// 计算第一类修改的贝塞尔函数 K1(x)，返回双精度浮点数
double k1(double);

// 计算第一类修改的贝塞尔函数 K1e(x)，返回双精度浮点数
double k1e(double);

// 计算开尔文函数 kei(x)，返回双精度浮点数
double kei(double);

// 计算开尔文函数的导数 keip(x)，返回双精度浮点数
double keip(double);

// 计算开尔文函数 kelvin(a, x, c, ci)，其中 x 是复数，a、c、ci 是复数指针
void kelvin(double, double complex *, double complex *, double complex *, double complex *);

// 计算第二类修改的贝塞尔函数 Ker(x)，返回双精度浮点数
double ker(double);

// 计算第二类修改的贝塞尔函数的导数 Kerp(x)，返回双精度浮点数
double kerp(double);

// 计算 KL 散度 kl_div(x, y)，返回双精度浮点数
double kl_div(double, double);

// 计算第 n 阶修改的贝塞尔函数 Kn(x) 或 Kn(n, x)，返回双精度浮点数
double kn(double, double);
double kn(long, double);

// 计算科尔莫哥洛夫函数的逆 kolmogi(p)，返回双精度浮点数
double kolmogi(double);

// 计算科尔莫哥洛夫分布函数 kolmogorov(x)，返回双精度浮点数
double kolmogorov(double);

// 计算贝塞尔函数第二类 kv(v, z) 或 kv(v, x)，其中 v 是复数，z 或 x 是双精度浮点数
double complex kv(double, double complex);
double kv(double, double);

// 计算修改的贝塞尔函数第二类 kve(v, z) 或 kve(v, x)，其中 v 是复数，z 或 x 是双精度浮点数
double complex kve(double, double complex);
double kve(double, double);

// 计算 log(1 + x) 的复数版本 log1p(z) 或 实数版本 log1p(x)，返回双精度浮点数
double complex log1p(double complex);
double log1p(double);

// 计算逻辑 sigmoid 函数的对数版本 log_expit(x)，其中 x 是双精度浮点数、单精度浮点数或长双精度浮点数，返回双精度浮点数
double log_expit(double);
float log_expit(float);
long double log_expit(long double);

// 计算标准正态分布的对数版本 log_ndtr(x) 或 log_ndtr(z)，其中 x 是双精度浮点数、复数版本 z，返回双精度浮点数
double log_ndtr(double);
double complex log_ndtr(double complex);

// 计算伽马函数的对数版本 loggamma(x) 或 loggamma(z)，其中 x 是双精度浮点数、复数版本 z，返回双精度浮点数
double loggamma(double);
double complex loggamma(double complex);

// 计算逻辑函数的逆 logit(x)，其中 x 是双精度浮点数、单精度浮点数或长双精度浮点数，返回双精度浮点数
double logit(double);
float logit(float);
long double logit(long double);

// 计算连带勒让德函数的复数版本 lpmv(m, n, z)，其中 m、n 是双精度浮点数，z 是复数，返回双精度浮点数
double lpmv(double, double, double);

// 计算马修函数 Mathieu_A(m, q)，返回双精度浮点数
double mathieu_a(double, double);

// 计算马修函数 Mathieu_B(m, q)，返回双精度浮点数
double mathieu_b(double, double);

// 计算马修函数 Mathieu_CEM(m, q, x, cem1, cem2)，其中 x 是双精度浮点数，cem1、cem2 是复数指针
void mathieu_cem(double, double, double, double *, double *);

// 计算调整后的马修函数 Mathieu_MODCEM1(m, q, x, cem1, cem2)，其中 x 是双精度浮点数，cem1、cem2 是复数指针
void mathieu_modcem1(double, double, double, double *, double *);

// 计算调整后的马修函数 Mathieu_MODCEM2(m, q, x, cem1, cem2)，其中 x 是双精度浮点数，cem1、cem2 是复数指针
void mathieu_modcem2(double, double, double, double *, double *);

// 计算调整后的马修函数 Mathieu_MODSEM1(m, q, x, sem1, sem2)，其中 x 是双精度浮点数，sem1、sem2 是复数指针
void mathieu_modsem1(double, double, double, double *, double *);

// 计算调整后的马修函数 Mathieu_MODSEM2(m, q, x, sem1, sem2)，其中 x 是双精度浮点数，sem1、sem2 是复数指针
void mathieu_modsem2(double, double, double, double *, double *);

// 计算马修函数 Mathieu_SEM(m, q, x, sem1, sem2)，其中 x 是双精度浮点数，sem1、sem2 是复数指针
void mathieu_sem(double, double, double, double *, double *);

// 计算调整后的费涅尔函数 ModFresnelM(x, fm1, fm2)，其中 x 是双精度浮点数，fm1、fm2 是复数指针
void modfresnelm(double, double complex *, double complex *);

// 计算调整后的费涅尔函数 ModFresnelP(x, fp1, fp2)，其中 x 是双精度浮点数，fp1、fp2 是复数指针
void modfresnelp(double, double complex *, double complex *);

// 计算修改的斯特鲁夫函数 ModStruve(v, x)，返回双精度浮点数
double modstruve(double, double);
// 计算负二项分布的累积分布函数（CDF）
double nbdtr(double, double, double)
// 计算负二项分布的累积分布函数（CDF）
double nbdtr(long, long, double)

// 计算负二项分布的补充累积分布函数（1 - CDF）
double nbdtrc(double, double, double)
// 计算负二项分布的补充累积分布函数（1 - CDF）
double nbdtrc(long, long, double)

// 计算负二项分布的反函数（CDF 的逆函数）
double nbdtri(double, double, double)
// 计算负二项分布的反函数（CDF 的逆函数）
double nbdtri(long, long, double)

// 计算负二项分布的 k-分位数
double nbdtrik(double, double, double)

// 计算负二项分布的 n-分位数
double nbdtrin(double, double, double)

// 计算非中心 F 分布的累积分布函数（CDF）
double ncfdtr(double, double, double, double)

// 计算非中心 F 分布的反函数（CDF 的逆函数）
double ncfdtri(double, double, double, double)

// 计算非中心 F 分布的自由度 dfd
double ncfdtridfd(double, double, double, double)

// 计算非中心 F 分布的非中心参数 dfn
double ncfdtridfn(double, double, double, double)

// 计算非中心 F 分布的互补累积分布函数（1 - CDF）
double ncfdtrinc(double, double, double, double)

// 计算非中心 t 分布的累积分布函数（CDF）
double nctdtr(double, double, double)

// 计算非中心 t 分布的自由度
double nctdtridf(double, double, double)

// 计算非中心 t 分布的互补累积分布函数（1 - CDF）
double nctdtrinc(double, double, double)

// 计算非中心 t 分布的反函数（CDF 的逆函数）
double nctdtrit(double, double, double)

// 计算正态分布的累积分布函数（CDF）
double complex ndtr(double complex)
double ndtr(double)

// 计算正态分布的反函数（CDF 的逆函数）
double ndtri(double)

// 计算正态分布的修正值
double nrdtrimn(double, double, double)

// 计算正态分布的标准差
double nrdtrisd(double, double, double)

// 计算球坐标系的角度
void obl_ang1(double, double, double, double, double *, double *)

// 计算球坐标系的角度（复数版本）
void obl_ang1_cv(double, double, double, double, double, double *, double *)

// 计算球坐标系的弧度
double obl_cv(double, double, double)

// 计算球坐标系的半径
void obl_rad1(double, double, double, double, double *, double *)

// 计算球坐标系的半径（复数版本）
void obl_rad1_cv(double, double, double, double, double, double *, double *)

// 计算球坐标系的第二个半径
void obl_rad2(double, double, double, double, double *, double *)

// 计算球坐标系的第二个半径（复数版本）
void obl_rad2_cv(double, double, double, double, double, double *, double *)

// 计算Owens T函数的值
double owens_t(double, double)

// 计算参数的变化
void pbdv(double, double, double *, double *)

// 计算变化函数
void pbvv(double, double, double *, double *)

// 计算计算数值积分
void pbwa(double, double, double *, double *)

// 计算概率分布的积分
double pdtr(double, double)
// 计算正态分布的累积分布函数的补函数（1 - CDF）
double pdtrc(double, double)

// 计算正态分布的分位数函数的补函数（逆函数）
double pdtri(double, double)
// 计算二项分布的分位数函数的补函数（逆函数）
double pdtri(long, double)

// 计算卡方分布的累积分布函数的补函数（1 - CDF）
double pdtrik(double, double)

// 计算 Pochhammer 符号的值
double poch(double, double)

// 计算指数函数减一的值
float powm1(float, float)
double powm1(double, double)

// 计算球面角的三维投影
void pro_ang1(double, double, double, double, double *, double *)

// 计算球面角的三维投影（带验证）
void pro_ang1_cv(double, double, double, double, double, double *, double *)

// 计算椭圆积分的一种形式
double pro_cv(double, double, double)

// 计算径向函数的一维投影
void pro_rad1(double, double, double, double, double *, double *)

// 计算径向函数的一维投影（带验证）
void pro_rad1_cv(double, double, double, double, double, double *, double *)

// 计算径向函数的二维投影
void pro_rad2(double, double, double, double, double *, double *)

// 计算径向函数的二维投影（带验证）
void pro_rad2_cv(double, double, double, double, double, double *, double *)

// 计算 Pseudo-Huber 损失函数的值
double pseudo_huber(double, double)

// 计算 Psi 函数的值
double complex psi(double complex)
double psi(double)

// 将角度转换为弧度
double radian(double, double, double)

// 计算相对熵（KL 散度）
double rel_entr(double, double)

// 计算 Gamma 函数的值
double complex rgamma(double complex)
double rgamma(double)

// 对浮点数进行四舍五入
double round(double)

// 计算反正弦积分的值
void shichi(double complex, double complex *, double complex *)
void shichi(double, double *, double *)

// 计算正弦积分的值
void sici(double complex, double complex *, double complex *)
void sici(double, double *, double *)

// 计算角度的正弦值
double sindg(double)

// 计算斯米尔诺夫分布函数的值
double smirnov(double, double)
double smirnov(long, double)

// 计算斯米尔诺夫分布函数的逆函数
double smirnovi(double, double)
double smirnovi(long, double)

// 计算斯彭斯（Spence）函数的值
double complex spence(double complex)
double spence(double)

// 计算球谐函数的值
double complex sph_harm(double, double, double, double)
double complex sph_harm(long, long, double, double)

// 计算 Student t 分布的累积分布函数
double stdtr(double, double)

// 计算 Student t 分布的分位数函数
double stdtridf(double, double)

// 计算 Student t 分布的逆函数
double stdtrit(double, double)
# 导入数学库中的 NaN 常量
from libc.math cimport NAN

# 导入 NumPy 中的特定数据类型
from numpy cimport npy_float, npy_double, npy_longdouble, npy_cdouble, npy_int, npy_long
# 从头文件 "numpy/ufuncobject.h" 中导入 PyUFunc_getfperr 函数的声明
cdef extern from "numpy/ufuncobject.h":
    int PyUFunc_getfperr() nogil

# 定义一个公共的 Cython 函数 wrap_PyUFunc_getfperr，用于调用 PyUFunc_getfperr 函数，
# 在确保 PyUFunc_API 数组已初始化的情况下调用，以避免干扰 UNIQUE_SYMBOL 宏定义。
cdef public int wrap_PyUFunc_getfperr() noexcept nogil:
    """
    Call PyUFunc_getfperr in a context where PyUFunc_API array is initialized;
    this avoids messing with the UNIQUE_SYMBOL #defines
    """
    return PyUFunc_getfperr()

# 从当前目录的 _complexstuff 模块中导入全部内容
from . cimport _complexstuff

# 从 scipy.special._ufuncs_cxx 模块中导入全部内容
cimport scipy.special._ufuncs_cxx

# 从 scipy.special 模块中导入 _ufuncs 对象
from scipy.special import _ufuncs

# 定义 long double 类型的别名 long_double
ctypedef long double long_double

# 定义 float complex 类型的别名 float_complex
ctypedef float complex float_complex

# 定义 double complex 类型的别名 double_complex
ctypedef double complex double_complex

# 定义 long double complex 类型的别名 long_double_complex
ctypedef long double complex long_double_complex

# 从头文件 "special_wrappers.h" 中外部导入以下函数声明，并使用 nogil 语义
cdef extern from r"special_wrappers.h":
    # 定义 _func_gammaln_wrap 函数，参数为 double 类型，别名为 "gammaln_wrap"
    double _func_gammaln_wrap "gammaln_wrap"(double) nogil

    # 定义 special_bei 函数，参数为 double 类型，别名为 "special_bei"
    double special_bei(double) nogil

    # 定义 special_beip 函数，参数为 double 类型，别名为 "special_beip"
    double special_beip(double) nogil

    # 定义 special_ber 函数，参数为 double 类型，别名为 "special_ber"
    double special_ber(double) nogil

    # 定义 special_berp 函数，参数为 double 类型，别名为 "special_berp"
    double special_berp(double) nogil

    # 定义 special_kei 函数，参数为 npy_double 类型，别名为 "special_kei"
    npy_double special_kei(npy_double) nogil

    # 定义 special_keip 函数，参数为 npy_double 类型，别名为 "special_keip"
    npy_double special_keip(npy_double) nogil

    # 定义 special_ckelvin 函数，参数为 npy_double, npy_cdouble *, npy_cdouble *, npy_cdouble *, npy_cdouble *，
    # 别名为 "special_ckelvin"
    void special_ckelvin(npy_double, npy_cdouble *, npy_cdouble *, npy_cdouble *, npy_cdouble *) nogil

    # 定义 special_ker 函数，参数为 npy_double 类型，别名为 "special_ker"
    npy_double special_ker(npy_double) nogil

    # 定义 special_kerp 函数，参数为 double 类型，别名为 "special_kerp"
    double special_kerp(double) nogil

    # 定义 _func_cem_cva_wrap 函数，参数为 npy_double, npy_double 类型，别名为 "cem_cva_wrap"
    npy_double _func_cem_cva_wrap "cem_cva_wrap"(npy_double, npy_double) nogil

    # 定义 _func_sem_cva_wrap 函数，参数为 npy_double, npy_double 类型，别名为 "sem_cva_wrap"
    npy_double _func_sem_cva_wrap "sem_cva_wrap"(npy_double, npy_double) nogil

    # 定义 _func_cem_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double *, npy_double *，
    # 别名为 "cem_wrap"
    void _func_cem_wrap "cem_wrap"(npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil

    # 定义 _func_mcm1_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double *, npy_double *，
    # 别名为 "mcm1_wrap"
    void _func_mcm1_wrap "mcm1_wrap"(npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil

    # 定义 _func_mcm2_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double *, npy_double *，
    # 别名为 "mcm2_wrap"
    void _func_mcm2_wrap "mcm2_wrap"(npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil

    # 定义 _func_msm1_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double *, npy_double *，
    # 别名为 "msm1_wrap"
    void _func_msm1_wrap "msm1_wrap"(npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil

    # 定义 _func_msm2_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double *, npy_double *，
    # 别名为 "msm2_wrap"
    void _func_msm2_wrap "msm2_wrap"(npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil

    # 定义 _func_sem_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double *, npy_double *，
    # 别名为 "sem_wrap"
    void _func_sem_wrap "sem_wrap"(npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil

    # 定义 _func_modified_fresnel_minus_wrap 函数，参数为 npy_double, npy_cdouble *, npy_cdouble *，
    # 别名为 "modified_fresnel_minus_wrap"
    void _func_modified_fresnel_minus_wrap "modified_fresnel_minus_wrap"(npy_double, npy_cdouble *, npy_cdouble *) nogil

    # 定义 _func_modified_fresnel_plus_wrap 函数，参数为 npy_double, npy_cdouble *, npy_cdouble *，
    # 别名为 "modified_fresnel_plus_wrap"
    void _func_modified_fresnel_plus_wrap "modified_fresnel_plus_wrap"(npy_double, npy_cdouble *, npy_cdouble *) nogil

    # 定义 _func_oblate_aswfa_nocv_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double, npy_double *，
    # 别名为 "oblate_aswfa_nocv_wrap"
    npy_double _func_oblate_aswfa_nocv_wrap "oblate_aswfa_nocv_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double *) nogil

    # 定义 _func_oblate_aswfa_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double, npy_double, npy_double *, npy_double *，
    # 别名为 "oblate_aswfa_wrap"
    void _func_oblate_aswfa_wrap "oblate_aswfa_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil

    # 定义 _func_oblate_segv_wrap 函数，参数为 npy_double, npy_double, npy_double 类型，别名为 "oblate_segv_wrap"
    npy_double _func_oblate_segv_wrap "oblate_segv_wrap"(npy_double, npy_double, npy_double) nogil

    # 定义 _func_oblate_radial1_nocv_wrap 函数，参数为 npy_double, npy_double, npy_double, npy_double, npy_double *，
    # 别名为 "oblate_radial1_nocv_wrap"
    npy_double _func_oblate_radial1_nocv_wrap "oblate_radial1_nocv_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double *) nogil

    # 定义 _func_oblate_radial1_wrap 函数，参数为 npy_double, np
    # 声明一个接受多个参数的函数 _func_oblate_radial2_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_oblate_radial2_wrap "oblate_radial2_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受多个参数的函数 _func_prolate_aswfa_nocv_wrap，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double _func_prolate_aswfa_nocv_wrap "prolate_aswfa_nocv_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double *) nogil
    
    # 声明一个接受多个参数的函数 _func_prolate_aswfa_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_prolate_aswfa_wrap "prolate_aswfa_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受多个参数的函数 _func_prolate_segv_wrap，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double _func_prolate_segv_wrap "prolate_segv_wrap"(npy_double, npy_double, npy_double) nogil
    
    # 声明一个接受多个参数的函数 _func_prolate_radial1_nocv_wrap，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double _func_prolate_radial1_nocv_wrap "prolate_radial1_nocv_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double *) nogil
    
    # 声明一个接受多个参数的函数 _func_prolate_radial1_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_prolate_radial1_wrap "prolate_radial1_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受多个参数的函数 _func_prolate_radial2_nocv_wrap，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double _func_prolate_radial2_nocv_wrap "prolate_radial2_nocv_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double *) nogil
    
    # 声明一个接受多个参数的函数 _func_prolate_radial2_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_prolate_radial2_wrap "prolate_radial2_wrap"(npy_double, npy_double, npy_double, npy_double, npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个复数参数的函数 special_cexp1，返回值类型为 npy_cdouble，使用 nogil 来指示 GIL 释放
    npy_cdouble special_cexp1(npy_cdouble) nogil
    
    # 声明一个接受一个浮点数参数的函数 special_exp1，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double special_exp1(npy_double) nogil
    
    # 声明一个接受一个复数参数的函数 special_cexpi，返回值类型为 npy_cdouble，使用 nogil 来指示 GIL 释放
    npy_cdouble special_cexpi(npy_cdouble) nogil
    
    # 声明一个接受一个浮点数参数的函数 special_expi，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double special_expi(npy_double) nogil
    
    # 声明一个接受一个浮点数参数的函数 _func_it2i0k0_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_it2i0k0_wrap "it2i0k0_wrap"(npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个浮点数参数的函数 _func_it2j0y0_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_it2j0y0_wrap "it2j0y0_wrap"(npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个浮点数参数的函数 special_it2struve0，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double special_it2struve0(npy_double) nogil
    
    # 声明一个接受多个参数的函数 special_itairy，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void special_itairy(npy_double, npy_double *, npy_double *, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个浮点数参数的函数 _func_it1i0k0_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_it1i0k0_wrap "it1i0k0_wrap"(npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个浮点数参数的函数 _func_it1j0y0_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_it1j0y0_wrap "it1j0y0_wrap"(npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个浮点数参数的函数 special_itmodstruve0，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double special_itmodstruve0(npy_double) nogil
    
    # 声明一个接受一个浮点数参数的函数 special_itstruve0，返回值类型为 npy_double，使用 nogil 来指示 GIL 释放
    npy_double special_itstruve0(npy_double) nogil
    
    # 声明一个接受两个浮点数参数的函数 _func_pbdv_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_pbdv_wrap "pbdv_wrap"(npy_double, npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受两个浮点数参数的函数 _func_pbvv_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_pbvv_wrap "pbvv_wrap"(npy_double, npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受两个浮点数参数的函数 _func_pbwa_wrap，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void _func_pbwa_wrap "pbwa_wrap"(npy_double, npy_double, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个复数和一个复数指针参数的函数 _func_cfresnl_wrap，返回值类型为 npy_int，使用 nogil 来指示 GIL 释放
    npy_int _func_cfresnl_wrap "cfresnl_wrap"(npy_cdouble, npy_cdouble *, npy_cdouble *) nogil
    
    # 声明一个接受一个浮点数参数的函数 special_airy，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void special_airy(npy_double, npy_double *, npy_double *, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个复数参数的函数 special_cairy，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void special_cairy(npy_cdouble, npy_cdouble *, npy_cdouble *, npy_cdouble *, npy_cdouble *) nogil
    
    # 声明一个接受一个浮点数参数的函数 special_airye，返回值类型为 void，使用 nogil 来指示 GIL 释放
    void special_airye(npy_double, npy_double *, npy_double *, npy_double *, npy_double *) nogil
    
    # 声明一个接受一个
    # 定义一个特殊函数 special_binom，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL（全局解释器锁）的影响
    
    # 定义一个特殊函数 special_digamma，接受一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_cdigamma，接受一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_cyl_bessel_j，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_ccyl_bessel_j，接受一个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_cyl_bessel_je，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_ccyl_bessel_je，接受一个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_cyl_bessel_y，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_ccyl_bessel_y，接受一个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_cyl_bessel_ye，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_ccyl_bessel_ye，接受一个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_cyl_bessel_k_int，接受一个 npy_int 和一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_cyl_bessel_k，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_ccyl_bessel_k，接受一个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_cyl_bessel_ke，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_ccyl_bessel_ke，接受一个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_cyl_bessel_i，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_ccyl_bessel_i，接受一个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_cyl_bessel_ie，接受两个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_ccyl_bessel_ie，接受一个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_exprel，接受一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_gamma，接受一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_cgamma，接受一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义三个函数 special_expitf, special_expit, special_expitl，分别接受 npy_float, npy_double, npy_longdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义三个函数 special_log_expitf, special_log_expit, special_log_expitl，分别接受 npy_float, npy_double, npy_longdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义三个函数 special_logitf, special_logit, special_logitl，分别接受 npy_float, npy_double, npy_longdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_loggamma，接受一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_cloggamma，接受一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_hyp2f1，接受四个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_chyp2f1，接受三个 npy_double 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_rgamma，接受一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_crgamma，接受一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_sph_bessel_j，接受一个 npy_long 和一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_csph_bessel_j，接受一个 npy_long 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_sph_bessel_j_jac，接受一个 npy_long 和一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_csph_bessel_j_jac，接受一个 npy_long 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_sph_bessel_y，接受一个 npy_long 和一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_csph_bessel_y，接受一个 npy_long 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_sph_bessel_y_jac，接受一个 npy_long 和一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_csph_bessel_y_jac，接受一个 npy_long 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    
    # 定义一个特殊函数 special_sph_bessel_i，接受一个 npy_long 和一个 npy_double 类型的参数，使用 nogil 以确保没有 GIL 的影响
    # 定义一个特殊函数 special_csph_bessel_i，接受一个 npy_long 和一个 npy_cdouble 类型的参数，使用 nogil 以确保没有 GIL 的影响
    npy_long special_sph_bessel_i_jac(npy_long, npy_double) nogil
        # 定义一个函数 special_sph_bessel_i_jac，返回一个 npy_long 类型的值，接受一个 npy_long 和一个 npy_double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_cdouble special_csph_bessel_i_jac(npy_long, npy_cdouble) nogil
        # 定义一个函数 special_csph_bessel_i_jac，返回一个 npy_cdouble 类型的值，接受一个 npy_long 和一个 npy_cdouble 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_long special_sph_bessel_k(npy_long, npy_double) nogil
        # 定义一个函数 special_sph_bessel_k，返回一个 npy_long 类型的值，接受一个 npy_long 和一个 npy_double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_cdouble special_csph_bessel_k(npy_long, npy_cdouble) nogil
        # 定义一个函数 special_csph_bessel_k，返回一个 npy_cdouble 类型的值，接受一个 npy_long 和一个 npy_cdouble 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_long special_sph_bessel_k_jac(npy_long, npy_double) nogil
        # 定义一个函数 special_sph_bessel_k_jac，返回一个 npy_long 类型的值，接受一个 npy_long 和一个 npy_double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_cdouble special_csph_bessel_k_jac(npy_long, npy_cdouble) nogil
        # 定义一个函数 special_csph_bessel_k_jac，返回一个 npy_cdouble 类型的值，接受一个 npy_long 和一个 npy_cdouble 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_cdouble special_sph_harm(npy_long, npy_long, npy_double, npy_double) nogil
        # 定义一个函数 special_sph_harm，返回一个 npy_cdouble 类型的值，接受两个 npy_long 类型和两个 npy_double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_cdouble special_sph_harm_unsafe(npy_double, npy_double, npy_double, npy_double) nogil
        # 定义一个函数 special_sph_harm_unsafe，返回一个 npy_cdouble 类型的值，接受四个 npy_double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double _func_cephes_iv_wrap "cephes_iv_wrap"(double, double) nogil
        # 定义一个函数 _func_cephes_iv_wrap，返回一个 double 类型的值，命名为 "cephes_iv_wrap"，接受两个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_double special_wright_bessel(npy_double, npy_double, npy_double) nogil
        # 定义一个函数 special_wright_bessel，返回一个 npy_double 类型的值，接受三个 npy_double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    npy_double special_log_wright_bessel(npy_double, npy_double, npy_double) nogil
        # 定义一个函数 special_log_wright_bessel，返回一个 npy_double 类型的值，接受三个 npy_double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double special_ellipk(double m) nogil
        # 定义一个函数 special_ellipk，返回一个 double 类型的值，接受一个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_besselpoly(double a, double lmbda, double nu) nogil
        # 定义一个函数 cephes_besselpoly，返回一个 double 类型的值，接受三个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_beta(double a, double b) nogil
        # 定义一个函数 cephes_beta，返回一个 double 类型的值，接受两个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_chdtr(double df, double x) nogil
        # 定义一个函数 cephes_chdtr，返回一个 double 类型的值，接受两个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_chdtrc(double df, double x) nogil
        # 定义一个函数 cephes_chdtrc，返回一个 double 类型的值，接受两个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_chdtri(double df, double y) nogil
        # 定义一个函数 cephes_chdtri，返回一个 double 类型的值，接受两个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_chdtrc(double df, double x) nogil
        # 定义一个函数 cephes_chdtrc，返回一个 double 类型的值，接受两个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_chdtri(double df, double y) nogil
        # 定义一个函数 cephes_chdtri，返回一个 double 类型的值，接受两个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_lbeta(double a, double b) nogil
        # 定义一个函数 cephes_lbeta，返回一个 double 类型的值，接受两个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_sinpi(double x) nogil
        # 定义一个函数 cephes_sinpi，返回一个 double 类型的值，接受一个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_cospi(double x) nogil
        # 定义一个函数 cephes_cospi，返回一个 double 类型的值，接受一个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_cbrt(double x) nogil
        # 定义一个函数 cephes_cbrt，返回一个 double 类型的值，接受一个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_Gamma(double x) nogil
        # 定义一个函数 cephes_Gamma，返回一个 double 类型的值，接受一个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_gammasgn(double x) nogil
        # 定义一个函数 cephes_gammasgn，返回一个 double 类型的值，接受一个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_hyp2f1(double a, double b, double c, double x) nogil
        # 定义一个函数 cephes_hyp2f1，返回一个 double 类型的值，接受四个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_i0(double x) nogil
        # 定义一个函数 cephes_i0，返回一个 double 类型的值，接受一个 double 类型的参数，且在函数体内不使用全局解释器锁（nogil）。
    
    double cephes_i0e(double x) nogil
        # 定义一个函数 cephes_i0e
    # 计算 Zeta 函数的值
    double cephes_zeta(double x, double q) nogil
    
    # 计算 Zeta 补函数的值
    double cephes_zetac(double x) nogil
    
    # 计算 Riemann Zeta 函数的值
    double cephes_riemann_zeta(double x) nogil
    
    # 计算 log(1+x) 的值
    double cephes_log1p(double x) nogil
    
    # 计算 log(1+x)-x 的值
    double cephes_log1pmx(double x) nogil
    
    # 计算 lgamma(1+x) 的值
    double cephes_lgam1p(double x) nogil
    
    # 计算 expm1(x) 的值
    double cephes_expm1(double x) nogil
    
    # 计算 cos(x)-1 的值
    double cephes_cosm1(double x) nogil
    
    # 计算指数函数的负幂
    double cephes_expn(int n, double x) nogil
    
    # 计算椭圆积分的第二类和第三类
    double cephes_ellpe(double x) nogil
    double cephes_ellpk(double x) nogil
    double cephes_ellie(double phi, double m) nogil
    double cephes_ellik(double phi, double m) nogil
    
    # 计算正弦、余弦、正切和余切的度数值
    double cephes_sindg(double x) nogil
    double cephes_cosdg(double x) nogil
    double cephes_tandg(double x) nogil
    double cephes_cotdg(double x) nogil
    
    # 将度数转换为弧度
    double cephes_radian(double d, double m, double s) nogil
    
    # 计算正态分布的逆累积分布函数
    double cephes_ndtri(double x) nogil
    
    # 计算贝塔分布的累积分布函数和其逆函数
    double cephes_bdtr(double k, int n, double p) nogil
    double cephes_bdtri(double k, int n, double y) nogil
    double cephes_bdtrc(double k, int n, double p) nogil
    double cephes_btdtri(double aa, double bb, double yy0) nogil
    double cephes_btdtr(double a, double b, double x) nogil
    
    # 计算互补误差函数的逆函数
    double cephes_erfcinv(double y) nogil
    
    # 计算 10 的指数函数和 2 的指数函数
    double cephes_exp10(double x) nogil
    double cephes_exp2(double x) nogil
    
    # 计算 F 分布的累积分布函数和其逆函数
    double cephes_fdtr(double a, double b, double x) nogil
    double cephes_fdtrc(double a, double b, double x) nogil
    double cephes_fdtri(double a, double b, double y) nogil
    
    # 计算 gamma 分布的累积分布函数和互补分布函数
    double cephes_gdtr(double a, double b, double x) nogil
    double cephes_gdtrc(double a, double b, double x) nogil
    
    # 计算 Owen's T 函数值
    double cephes_owens_t(double h, double a) nogil
    
    # 计算负二项分布的累积分布函数和互补分布函数
    double cephes_nbdtr(int k, int n, double p) nogil
    double cephes_nbdtrc(int k, int n, double p) nogil
    double cephes_nbdtri(int k, int n, double p) nogil
    
    # 计算 Poisson 分布的累积分布函数和互补分布函数
    double cephes_pdtr(double k, double m) nogil
    double cephes_pdtrc(double k, double m) nogil
    double cephes_pdtri(int k, double y) nogil
    
    # 计算浮点数的四舍五入值
    double cephes_round(double x) nogil
    
    # 计算斯佩恩斯函数的值
    double cephes_spence(double x) nogil
    
    # 计算 Tukey Lambda 分布的累积分布函数
    double cephes_tukeylambdacdf(double x, double lmbda) nogil
    
    # 计算斯特鲁夫函数 H 和 L 的值
    double cephes_struve_h(double v, double z) nogil
    double cephes_struve_l(double v, double z) nogil
# 导入AGM函数并定义其类型及变量，使用Cython的cimport语法
from ._agm cimport agm as _func_agm
ctypedef double _proto_agm_t(double, double) noexcept nogil
cdef _proto_agm_t *_proto_agm_t_var = &_func_agm

# 导入bdtr_unsafe函数并定义其类型及变量，使用Cython的cimport语法
from ._legacy cimport bdtr_unsafe as _func_bdtr_unsafe
ctypedef double _proto_bdtr_unsafe_t(double, double, double) noexcept nogil
cdef _proto_bdtr_unsafe_t *_proto_bdtr_unsafe_t_var = &_func_bdtr_unsafe

# 导入bdtrc_unsafe函数并定义其类型及变量，使用Cython的cimport语法
from ._legacy cimport bdtrc_unsafe as _func_bdtrc_unsafe
ctypedef double _proto_bdtrc_unsafe_t(double, double, double) noexcept nogil
cdef _proto_bdtrc_unsafe_t *_proto_bdtrc_unsafe_t_var = &_func_bdtrc_unsafe

# 导入bdtri_unsafe函数并定义其类型及变量，使用Cython的cimport语法
from ._legacy cimport bdtri_unsafe as _func_bdtri_unsafe
ctypedef double _proto_bdtri_unsafe_t(double, double, double) noexcept nogil
cdef _proto_bdtri_unsafe_t *_proto_bdtri_unsafe_t_var = &_func_bdtri_unsafe

# 导入bdtrik函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport bdtrik as _func_bdtrik
ctypedef double _proto_bdtrik_t(double, double, double) noexcept nogil
cdef _proto_bdtrik_t *_proto_bdtrik_t_var = &_func_bdtrik

# 导入bdtrin函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport bdtrin as _func_bdtrin
ctypedef double _proto_bdtrin_t(double, double, double) noexcept nogil
cdef _proto_bdtrin_t *_proto_bdtrin_t_var = &_func_bdtrin

# 导入boxcox函数并定义其类型及变量，使用Cython的cimport语法
from ._boxcox cimport boxcox as _func_boxcox
ctypedef double _proto_boxcox_t(double, double) noexcept nogil
cdef _proto_boxcox_t *_proto_boxcox_t_var = &_func_boxcox

# 导入boxcox1p函数并定义其类型及变量，使用Cython的cimport语法
from ._boxcox cimport boxcox1p as _func_boxcox1p
ctypedef double _proto_boxcox1p_t(double, double) noexcept nogil
cdef _proto_boxcox1p_t *_proto_boxcox1p_t_var = &_func_boxcox1p

# 导入btdtria函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport btdtria as _func_btdtria
ctypedef double _proto_btdtria_t(double, double, double) noexcept nogil
cdef _proto_btdtria_t *_proto_btdtria_t_var = &_func_btdtria

# 导入btdtrib函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport btdtrib as _func_btdtrib
ctypedef double _proto_btdtrib_t(double, double, double) noexcept nogil
cdef _proto_btdtrib_t *_proto_btdtrib_t_var = &_func_btdtrib

# 导入chdtriv函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport chdtriv as _func_chdtriv
ctypedef double _proto_chdtriv_t(double, double) noexcept nogil
cdef _proto_chdtriv_t *_proto_chdtriv_t_var = &_func_chdtriv

# 导入chndtr函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport chndtr as _func_chndtr
ctypedef double _proto_chndtr_t(double, double, double) noexcept nogil
cdef _proto_chndtr_t *_proto_chndtr_t_var = &_func_chndtr

# 导入chndtridf函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport chndtridf as _func_chndtridf
ctypedef double _proto_chndtridf_t(double, double, double) noexcept nogil
cdef _proto_chndtridf_t *_proto_chndtridf_t_var = &_func_chndtridf

# 导入chndtrinc函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport chndtrinc as _func_chndtrinc
ctypedef double _proto_chndtrinc_t(double, double, double) noexcept nogil
cdef _proto_chndtrinc_t *_proto_chndtrinc_t_var = &_func_chndtrinc

# 导入chndtrix函数并定义其类型及变量，使用Cython的cimport语法
from ._cdflib_wrappers cimport chndtrix as _func_chndtrix
ctypedef double _proto_chndtrix_t(double, double, double) noexcept nogil
cdef _proto_chndtrix_t *_proto_chndtrix_t_var = &_func_chndtrix

# 从外部头文件_r_ufuncs_defs.h中引入cephes_ellpj_wrap函数的声明，使用Cython的cdef extern语法
cdef extern from r"_ufuncs_defs.h":
    cdef npy_int _func_cephes_ellpj_wrap "cephes_ellpj_wrap"(npy_double, npy_double, npy_double *, npy_double *, npy_double *, npy_double *)nogil
# 从外部头文件 "_ufuncs_defs.h" 中导入名为 _func_ellik 的函数，其参数和返回值类型均为 npy_double，且在无全局解锁时执行
cdef extern from r"_ufuncs_defs.h":
    cdef npy_double _func_ellik "ellik"(npy_double, npy_double) nogil

# 从模块 "_ellipk" 中导入 ellipk 函数，并重命名为 _func_ellipk
from ._ellipk cimport ellipk as _func_ellipk

# 定义类型别名 _proto_ellipk_t，表示一个接受 double 类型参数并返回 double 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double _proto_ellipk_t(double) noexcept nogil

# 创建指针变量 _proto_ellipk_t_var，指向 _func_ellipk 函数
cdef _proto_ellipk_t *_proto_ellipk_t_var = &_func_ellipk

# 从模块 "_convex_analysis" 中导入 entr 函数，并重命名为 _func_entr
from .convex_analysis cimport entr as _func_entr

# 定义类型别名 _proto_entr_t，表示一个接受 double 类型参数并返回 double 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double _proto_entr_t(double) noexcept nogil

# 创建指针变量 _proto_entr_t_var，指向 _func_entr 函数
cdef _proto_entr_t *_proto_entr_t_var = &_func_entr

# 从模块 "orthogonal_eval" 中导入 eval_chebyc 函数，并重命名为 _func_eval_chebyc
from .orthogonal_eval cimport eval_chebyc as _func_eval_chebyc

# 定义类型别名 _proto_eval_chebyc_double_complex__t，表示一个接受 double 和 double complex 类型参数并返回 double complex 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double complex _proto_eval_chebyc_double_complex__t(double, double complex) noexcept nogil

# 创建指针变量 _proto_eval_chebyc_double_complex__t_var，指向 _func_eval_chebyc 函数的 double complex 版本
cdef _proto_eval_chebyc_double_complex__t *_proto_eval_chebyc_double_complex__t_var = &_func_eval_chebyc[double_complex]

# 从模块 "orthogonal_eval" 中导入 eval_chebyc 函数，并重命名为 _func_eval_chebyc
from .orthogonal_eval cimport eval_chebyc as _func_eval_chebyc

# 定义类型别名 _proto_eval_chebyc_double__t，表示一个接受 double 类型参数并返回 double 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double _proto_eval_chebyc_double__t(double, double) noexcept nogil

# 创建指针变量 _proto_eval_chebyc_double__t_var，指向 _func_eval_chebyc 函数的 double 版本
cdef _proto_eval_chebyc_double__t *_proto_eval_chebyc_double__t_var = &_func_eval_chebyc[double]

# 从模块 "orthogonal_eval" 中导入 eval_chebyc_l 函数，并重命名为 _func_eval_chebyc_l
from .orthogonal_eval cimport eval_chebyc_l as _func_eval_chebyc_l

# 定义类型别名 _proto_eval_chebyc_l_t，表示一个接受 long 和 double 类型参数并返回 double 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double _proto_eval_chebyc_l_t(long, double) noexcept nogil

# 创建指针变量 _proto_eval_chebyc_l_t_var，指向 _func_eval_chebyc_l 函数
cdef _proto_eval_chebyc_l_t *_proto_eval_chebyc_l_t_var = &_func_eval_chebyc_l

# 从模块 "orthogonal_eval" 中导入 eval_chebys 函数，并重命名为 _func_eval_chebys
from .orthogonal_eval cimport eval_chebys as _func_eval_chebys

# 定义类型别名 _proto_eval_chebys_double_complex__t，表示一个接受 double 和 double complex 类型参数并返回 double complex 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double complex _proto_eval_chebys_double_complex__t(double, double complex) noexcept nogil

# 创建指针变量 _proto_eval_chebys_double_complex__t_var，指向 _func_eval_chebys 函数的 double complex 版本
cdef _proto_eval_chebys_double_complex__t *_proto_eval_chebys_double_complex__t_var = &_func_eval_chebys[double_complex]

# 从模块 "orthogonal_eval" 中导入 eval_chebys 函数，并重命名为 _func_eval_chebys
from .orthogonal_eval cimport eval_chebys as _func_eval_chebys

# 定义类型别名 _proto_eval_chebys_double__t，表示一个接受 double 类型参数并返回 double 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double _proto_eval_chebys_double__t(double, double) noexcept nogil

# 创建指针变量 _proto_eval_chebys_double__t_var，指向 _func_eval_chebys 函数的 double 版本
cdef _proto_eval_chebys_double__t *_proto_eval_chebys_double__t_var = &_func_eval_chebys[double]

# 从模块 "orthogonal_eval" 中导入 eval_chebys_l 函数，并重命名为 _func_eval_chebys_l
from .orthogonal_eval cimport eval_chebys_l as _func_eval_chebys_l

# 定义类型别名 _proto_eval_chebys_l_t，表示一个接受 long 和 double 类型参数并返回 double 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double _proto_eval_chebys_l_t(long, double) noexcept nogil

# 创建指针变量 _proto_eval_chebys_l_t_var，指向 _func_eval_chebys_l 函数
cdef _proto_eval_chebys_l_t *_proto_eval_chebys_l_t_var = &_func_eval_chebys_l

# 从模块 "orthogonal_eval" 中导入 eval_chebyt 函数，并重命名为 _func_eval_chebyt
from .orthogonal_eval cimport eval_chebyt as _func_eval_chebyt

# 定义类型别名 _proto_eval_chebyt_double_complex__t，表示一个接受 double 和 double complex 类型参数并返回 double complex 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double complex _proto_eval_chebyt_double_complex__t(double, double complex) noexcept nogil

# 创建指针变量 _proto_eval_chebyt_double_complex__t_var，指向 _func_eval_chebyt 函数的 double complex 版本
cdef _proto_eval_chebyt_double_complex__t *_proto_eval_chebyt_double_complex__t_var = &_func_eval_chebyt[double_complex]

# 从模块 "orthogonal_eval" 中导入 eval_chebyt 函数，并重命名为 _func_eval_chebyt
from .orthogonal_eval cimport eval_chebyt as _func_eval_chebyt

# 定义类型别名 _proto_eval_chebyt_double__t，表示一个接受 double 类型参数并返回 double 类型结果的函数指针类型，函数是无异常、无全局解锁
ctypedef double _proto_eval_chebyt_double__t(double, double) noexcept nogil

# 创建指针变量 _proto_eval_chebyt_double__t_var，指向 _func_eval_chebyt 函数的 double 版本
cdef _proto_eval_chebyt_double__t *_proto_eval_chebyt_double__t_var = &_func_eval_chebyt[double]

# 从模块 "orthogonal_eval" 中导入 eval_chebyt_l 函数，并重命名为 _func_eval_chebyt_l
from .orthogonal_eval cimport eval_chebyt_l as _func_eval_chebyt_l

# 定义类型别名 _proto_eval_chebyt_l_t，表示一个接受 long 和 double 类型参数并返回 double 类型结果的函数指针类型，函数是无异常、无全局解锁
ct
# 导入所需的 C 语言函数定义和类型定义

# 定义一个 C 语言函数类型 _proto_eval_chebyu_double__t，接受两个 double 类型参数，且无异常，不使用全局解锁
ctypedef double _proto_eval_chebyu_double__t(double, double) noexcept nogil

# 创建一个指向 _func_eval_chebyu[double] 的指针变量 _proto_eval_chebyu_double__t_var
cdef _proto_eval_chebyu_double__t *_proto_eval_chebyu_double__t_var = &_func_eval_chebyu[double]

# 从 orthogonal_eval 模块中导入 eval_chebyu_l 函数，并为其创建一个别名 _func_eval_chebyu_l
from .orthogonal_eval cimport eval_chebyu_l as _func_eval_chebyu_l

# 定义一个 C 语言函数类型 _proto_eval_chebyu_l_t，接受一个 long 和一个 double 类型参数，且无异常，不使用全局解锁
ctypedef double _proto_eval_chebyu_l_t(long, double) noexcept nogil

# 创建一个指向 _func_eval_chebyu_l 的指针变量 _proto_eval_chebyu_l_t_var
cdef _proto_eval_chebyu_l_t *_proto_eval_chebyu_l_t_var = &_func_eval_chebyu_l

# 从 orthogonal_eval 模块中导入 eval_gegenbauer 函数，并为其创建一个别名 _func_eval_gegenbauer
from .orthogonal_eval cimport eval_gegenbauer as _func_eval_gegenbauer

# 定义一个 C 语言函数类型 _proto_eval_gegenbauer_double_complex__t，接受三个参数（double, double, double complex），且无异常，不使用全局解锁
ctypedef double complex _proto_eval_gegenbauer_double_complex__t(double, double, double complex) noexcept nogil

# 创建一个指向 _func_eval_gegenbauer[double_complex] 的指针变量 _proto_eval_gegenbauer_double_complex__t_var
cdef _proto_eval_gegenbauer_double_complex__t *_proto_eval_gegenbauer_double_complex__t_var = &_func_eval_gegenbauer[double_complex]

# 再次从 orthogonal_eval 模块中导入 eval_gegenbauer 函数，无需创建别名，直接导入
from .orthogonal_eval cimport eval_gegenbauer as _func_eval_gegenbauer

# 定义一个 C 语言函数类型 _proto_eval_gegenbauer_double__t，接受三个 double 类型参数，且无异常，不使用全局解锁
ctypedef double _proto_eval_gegenbauer_double__t(double, double, double) noexcept nogil

# 创建一个指向 _func_eval_gegenbauer[double] 的指针变量 _proto_eval_gegenbauer_double__t_var
cdef _proto_eval_gegenbauer_double__t *_proto_eval_gegenbauer_double__t_var = &_func_eval_gegenbauer[double]

# 从 orthogonal_eval 模块中导入 eval_gegenbauer_l 函数，并为其创建一个别名 _func_eval_gegenbauer_l
from .orthogonal_eval cimport eval_gegenbauer_l as _func_eval_gegenbauer_l

# 定义一个 C 语言函数类型 _proto_eval_gegenbauer_l_t，接受两个 long 和一个 double 类型参数，且无异常，不使用全局解锁
ctypedef double _proto_eval_gegenbauer_l_t(long, double, double) noexcept nogil

# 创建一个指向 _func_eval_gegenbauer_l 的指针变量 _proto_eval_gegenbauer_l_t_var
cdef _proto_eval_gegenbauer_l_t *_proto_eval_gegenbauer_l_t_var = &_func_eval_gegenbauer_l

# 从 orthogonal_eval 模块中导入 eval_genlaguerre 函数，并为其创建一个别名 _func_eval_genlaguerre
from .orthogonal_eval cimport eval_genlaguerre as _func_eval_genlaguerre

# 定义一个 C 语言函数类型 _proto_eval_genlaguerre_double_complex__t，接受三个参数（double, double, double complex），且无异常，不使用全局解锁
ctypedef double complex _proto_eval_genlaguerre_double_complex__t(double, double, double complex) noexcept nogil

# 创建一个指向 _func_eval_genlaguerre[double_complex] 的指针变量 _proto_eval_genlaguerre_double_complex__t_var
cdef _proto_eval_genlaguerre_double_complex__t *_proto_eval_genlaguerre_double_complex__t_var = &_func_eval_genlaguerre[double_complex]

# 再次从 orthogonal_eval 模块中导入 eval_genlaguerre 函数，无需创建别名，直接导入
from .orthogonal_eval cimport eval_genlaguerre as _func_eval_genlaguerre

# 定义一个 C 语言函数类型 _proto_eval_genlaguerre_double__t，接受三个 double 类型参数，且无异常，不使用全局解锁
ctypedef double _proto_eval_genlaguerre_double__t(double, double, double) noexcept nogil

# 创建一个指向 _func_eval_genlaguerre[double] 的指针变量 _proto_eval_genlaguerre_double__t_var
cdef _proto_eval_genlaguerre_double__t *_proto_eval_genlaguerre_double__t_var = &_func_eval_genlaguerre[double]

# 从 orthogonal_eval 模块中导入 eval_genlaguerre_l 函数，并为其创建一个别名 _func_eval_genlaguerre_l
from .orthogonal_eval cimport eval_genlaguerre_l as _func_eval_genlaguerre_l

# 定义一个 C 语言函数类型 _proto_eval_genlaguerre_l_t，接受两个 long 和一个 double 类型参数，且无异常，不使用全局解锁
ctypedef double _proto_eval_genlaguerre_l_t(long, double, double) noexcept nogil

# 创建一个指向 _func_eval_genlaguerre_l 的指针变量 _proto_eval_genlaguerre_l_t_var
cdef _proto_eval_genlaguerre_l_t *_proto_eval_genlaguerre_l_t_var = &_func_eval_genlaguerre_l

# 从 orthogonal_eval 模块中导入 eval_hermite 函数，并为其创建一个别名 _func_eval_hermite
from .orthogonal_eval cimport eval_hermite as _func_eval_hermite

# 定义一个 C 语言函数类型 _proto_eval_hermite_t，接受一个 long 和一个 double 类型参数，且无异常，不使用全局解锁
ctypedef double _proto_eval_hermite_t(long, double) noexcept nogil

# 创建一个指向 _func_eval_hermite 的指针变量 _proto_eval_hermite_t_var
cdef _proto_eval_hermite_t *_proto_eval_hermite_t_var = &_func_eval_hermite

# 从 orthogonal_eval 模块中导入 eval_hermitenorm 函数，并为其创建一个别名 _func_eval_hermitenorm
from .orthogonal_eval cimport eval_hermitenorm as _func_eval_hermitenorm

# 定义一个 C 语言函数类型 _proto_eval_hermitenorm_t，接受一个 long 和一个 double 类型参数，且无异常，不使用全局解锁
ctypedef double _proto_eval_hermitenorm_t(long, double) noexcept nogil

# 创建一个指向 _func_eval_hermitenorm 的指针变量 _proto_eval_hermitenorm_t_var
cdef _proto_eval_hermitenorm_t *_proto_eval_hermitenorm_t_var = &_func_eval_hermitenorm

# 从 orthogonal_eval 模块中导入 eval_jacobi 函数，并为其创建一个别名 _func_eval_jacobi
from .orthogonal_eval cimport eval_jacobi as _func_eval_jacobi

# 定义一个 C 语言函数类型 _proto_eval_jacobi_double_complex__t，接受四个参数（double, double, double, double complex），且无异常，不使用全局解锁
ctypedef double complex _proto_eval_jacobi_double_complex__t(double, double, double, double complex) noexcept nogil

# 创建一个指向 _func_eval_jacobi[double_complex] 的指针变量 _proto_eval_jacobi_double_complex__t_var
cdef _proto_eval_jacobi_double_complex__t *_proto_eval_jacobi_double_complex__t_var = &_func_eval_jacobi[double_complex]

# 再次从 orthogonal_eval
# 导入外部定义的 C 函数 _func_eval_jacobi[double]，并将其赋值给 _proto_eval_jacobi_double__t_var 变量
cdef _proto_eval_jacobi_double__t *_proto_eval_jacobi_double__t_var = &_func_eval_jacobi[double]

# 从外部模块中导入 eval_jacobi_l 函数，并将其命名为 _func_eval_jacobi_l
from .orthogonal_eval cimport eval_jacobi_l as _func_eval_jacobi_l

# 定义 eval_jacobi_l 函数类型 _proto_eval_jacobi_l_t，并声明 _proto_eval_jacobi_l_t_var 指向 _func_eval_jacobi_l
ctypedef double _proto_eval_jacobi_l_t(long, double, double, double) noexcept nogil
cdef _proto_eval_jacobi_l_t *_proto_eval_jacobi_l_t_var = &_func_eval_jacobi_l

# 从外部模块中导入 eval_laguerre 函数，并将其命名为 _func_eval_laguerre
from .orthogonal_eval cimport eval_laguerre as _func_eval_laguerre

# 定义 eval_laguerre 函数类型 _proto_eval_laguerre_double_complex__t，并声明 _proto_eval_laguerre_double_complex__t_var 指向 _func_eval_laguerre[double_complex]
ctypedef double complex _proto_eval_laguerre_double_complex__t(double, double complex) noexcept nogil
cdef _proto_eval_laguerre_double_complex__t *_proto_eval_laguerre_double_complex__t_var = &_func_eval_laguerre[double_complex]

# 再次导入 eval_laguerre 函数，为下一个函数声明做准备
from .orthogonal_eval cimport eval_laguerre as _func_eval_laguerre

# 定义 eval_laguerre 函数类型 _proto_eval_laguerre_double__t，并声明 _proto_eval_laguerre_double__t_var 指向 _func_eval_laguerre[double]
ctypedef double _proto_eval_laguerre_double__t(double, double) noexcept nogil
cdef _proto_eval_laguerre_double__t *_proto_eval_laguerre_double__t_var = &_func_eval_laguerre[double]

# 从外部模块中导入 eval_laguerre_l 函数，并将其命名为 _func_eval_laguerre_l
from .orthogonal_eval cimport eval_laguerre_l as _func_eval_laguerre_l

# 定义 eval_laguerre_l 函数类型 _proto_eval_laguerre_l_t，并声明 _proto_eval_laguerre_l_t_var 指向 _func_eval_laguerre_l
ctypedef double _proto_eval_laguerre_l_t(long, double) noexcept nogil
cdef _proto_eval_laguerre_l_t *_proto_eval_laguerre_l_t_var = &_func_eval_laguerre_l

# 从外部模块中导入 eval_legendre 函数，并将其命名为 _func_eval_legendre
from .orthogonal_eval cimport eval_legendre as _func_eval_legendre

# 定义 eval_legendre 函数类型 _proto_eval_legendre_double_complex__t，并声明 _proto_eval_legendre_double_complex__t_var 指向 _func_eval_legendre[double_complex]
ctypedef double complex _proto_eval_legendre_double_complex__t(double, double complex) noexcept nogil
cdef _proto_eval_legendre_double_complex__t *_proto_eval_legendre_double_complex__t_var = &_func_eval_legendre[double_complex]

# 再次导入 eval_legendre 函数，为下一个函数声明做准备
from .orthogonal_eval cimport eval_legendre as _func_eval_legendre

# 定义 eval_legendre 函数类型 _proto_eval_legendre_double__t，并声明 _proto_eval_legendre_double__t_var 指向 _func_eval_legendre[double]
ctypedef double _proto_eval_legendre_double__t(double, double) noexcept nogil
cdef _proto_eval_legendre_double__t *_proto_eval_legendre_double__t_var = &_func_eval_legendre[double]

# 从外部模块中导入 eval_legendre_l 函数，并将其命名为 _func_eval_legendre_l
from .orthogonal_eval cimport eval_legendre_l as _func_eval_legendre_l

# 定义 eval_legendre_l 函数类型 _proto_eval_legendre_l_t，并声明 _proto_eval_legendre_l_t_var 指向 _func_eval_legendre_l
ctypedef double _proto_eval_legendre_l_t(long, double) noexcept nogil
cdef _proto_eval_legendre_l_t *_proto_eval_legendre_l_t_var = &_func_eval_legendre_l

# 从外部模块中导入 eval_sh_chebyt 函数，并将其命名为 _func_eval_sh_chebyt
from .orthogonal_eval cimport eval_sh_chebyt as _func_eval_sh_chebyt

# 定义 eval_sh_chebyt 函数类型 _proto_eval_sh_chebyt_double_complex__t，并声明 _proto_eval_sh_chebyt_double_complex__t_var 指向 _func_eval_sh_chebyt[double_complex]
ctypedef double complex _proto_eval_sh_chebyt_double_complex__t(double, double complex) noexcept nogil
cdef _proto_eval_sh_chebyt_double_complex__t *_proto_eval_sh_chebyt_double_complex__t_var = &_func_eval_sh_chebyt[double_complex]

# 再次导入 eval_sh_chebyt 函数，为下一个函数声明做准备
from .orthogonal_eval cimport eval_sh_chebyt as _func_eval_sh_chebyt

# 定义 eval_sh_chebyt 函数类型 _proto_eval_sh_chebyt_double__t，并声明 _proto_eval_sh_chebyt_double__t_var 指向 _func_eval_sh_chebyt[double]
ctypedef double _proto_eval_sh_chebyt_double__t(double, double) noexcept nogil
cdef _proto_eval_sh_chebyt_double__t *_proto_eval_sh_chebyt_double__t_var = &_func_eval_sh_chebyt[double]

# 从外部模块中导入 eval_sh_chebyt_l 函数，并将其命名为 _func_eval_sh_chebyt_l
from .orthogonal_eval cimport eval_sh_chebyt_l as _func_eval_sh_chebyt_l

# 定义 eval_sh_chebyt_l 函数类型 _proto_eval_sh_chebyt_l_t，并声明 _proto_eval_sh_chebyt_l_t_var 指向 _func_eval_sh_chebyt_l
ctypedef double _proto_eval_sh_chebyt_l_t(long, double) noexcept nogil
cdef _proto_eval_sh_chebyt_l_t *_proto_eval_sh_chebyt_l_t_var = &_func_eval_sh_chebyt_l

# 从外部模块中导入 eval_sh_chebyu 函数，并将其命名为 _func_eval_sh_chebyu
from .orthogonal_eval cimport eval_sh_chebyu as _func_eval_sh_chebyu

# 定义 eval_sh_chebyu 函数类型 _proto_eval_sh_chebyu_double_complex__t，并声明 _proto_eval_sh_chebyu_double_complex__t_var 指向 _func_eval_sh_chebyu[double_complex]
ctypedef double complex _proto_eval_sh_chebyu_double_complex__t(double, double complex) noexcept nogil
cdef _proto_eval_sh_chebyu_double_complex__t *_proto_eval_sh_chebyu_double_complex__t_var = &_func_eval_sh_chebyu[double_complex]
# 导入自定义的评估函数 eval_sh_chebyu，并将其命名为 _func_eval_sh_chebyu
from .orthogonal_eval cimport eval_sh_chebyu as _func_eval_sh_chebyu
# 定义一个函数原型 _proto_eval_sh_chebyu_double__t，接受两个 double 类型参数，并且不引发异常，在无 GIL 环境下操作
ctypedef double _proto_eval_sh_chebyu_double__t(double, double) noexcept nogil
# 将 _func_eval_sh_chebyu 的 double 版本的指针赋给 _proto_eval_sh_chebyu_double__t_var
cdef _proto_eval_sh_chebyu_double__t *_proto_eval_sh_chebyu_double__t_var = &_func_eval_sh_chebyu[double]

# 导入自定义的评估函数 eval_sh_chebyu_l，命名为 _func_eval_sh_chebyu_l
from .orthogonal_eval cimport eval_sh_chebyu_l as _func_eval_sh_chebyu_l
# 定义一个函数原型 _proto_eval_sh_chebyu_l_t，接受一个 long 和一个 double 类型参数，并且不引发异常，在无 GIL 环境下操作
ctypedef double _proto_eval_sh_chebyu_l_t(long, double) noexcept nogil
# 将 _func_eval_sh_chebyu_l 的指针赋给 _proto_eval_sh_chebyu_l_t_var
cdef _proto_eval_sh_chebyu_l_t *_proto_eval_sh_chebyu_l_t_var = &_func_eval_sh_chebyu_l

# 导入自定义的评估函数 eval_sh_jacobi，命名为 _func_eval_sh_jacobi
from .orthogonal_eval cimport eval_sh_jacobi as _func_eval_sh_jacobi
# 定义一个复数版本的函数原型 _proto_eval_sh_jacobi_double_complex__t，接受四个参数（double, double, double, double complex），并且不引发异常，在无 GIL 环境下操作
ctypedef double complex _proto_eval_sh_jacobi_double_complex__t(double, double, double, double complex) noexcept nogil
# 将 _func_eval_sh_jacobi 的 double complex 版本的指针赋给 _proto_eval_sh_jacobi_double_complex__t_var
cdef _proto_eval_sh_jacobi_double_complex__t *_proto_eval_sh_jacobi_double_complex__t_var = &_func_eval_sh_jacobi[double_complex]

# 导入自定义的评估函数 eval_sh_jacobi，命名为 _func_eval_sh_jacobi
from .orthogonal_eval cimport eval_sh_jacobi as _func_eval_sh_jacobi
# 定义一个函数原型 _proto_eval_sh_jacobi_double__t，接受四个 double 类型参数，并且不引发异常，在无 GIL 环境下操作
ctypedef double _proto_eval_sh_jacobi_double__t(double, double, double, double) noexcept nogil
# 将 _func_eval_sh_jacobi 的 double 版本的指针赋给 _proto_eval_sh_jacobi_double__t_var
cdef _proto_eval_sh_jacobi_double__t *_proto_eval_sh_jacobi_double__t_var = &_func_eval_sh_jacobi[double]

# 导入自定义的评估函数 eval_sh_jacobi_l，命名为 _func_eval_sh_jacobi_l
from .orthogonal_eval cimport eval_sh_jacobi_l as _func_eval_sh_jacobi_l
# 定义一个函数原型 _proto_eval_sh_jacobi_l_t，接受一个 long 和三个 double 类型参数，并且不引发异常，在无 GIL 环境下操作
ctypedef double _proto_eval_sh_jacobi_l_t(long, double, double, double) noexcept nogil
# 将 _func_eval_sh_jacobi_l 的指针赋给 _proto_eval_sh_jacobi_l_t_var
cdef _proto_eval_sh_jacobi_l_t *_proto_eval_sh_jacobi_l_t_var = &_func_eval_sh_jacobi_l

# 导入自定义的评估函数 eval_sh_legendre，命名为 _func_eval_sh_legendre
from .orthogonal_eval cimport eval_sh_legendre as _func_eval_sh_legendre
# 定义一个复数版本的函数原型 _proto_eval_sh_legendre_double_complex__t，接受两个参数（double, double complex），并且不引发异常，在无 GIL 环境下操作
ctypedef double complex _proto_eval_sh_legendre_double_complex__t(double, double complex) noexcept nogil
# 将 _func_eval_sh_legendre 的 double complex 版本的指针赋给 _proto_eval_sh_legendre_double_complex__t_var
cdef _proto_eval_sh_legendre_double_complex__t *_proto_eval_sh_legendre_double_complex__t_var = &_func_eval_sh_legendre[double_complex]

# 导入自定义的评估函数 eval_sh_legendre，命名为 _func_eval_sh_legendre
from .orthogonal_eval cimport eval_sh_legendre as _func_eval_sh_legendre
# 定义一个函数原型 _proto_eval_sh_legendre_double__t，接受两个 double 类型参数，并且不引发异常，在无 GIL 环境下操作
ctypedef double _proto_eval_sh_legendre_double__t(double, double) noexcept nogil
# 将 _func_eval_sh_legendre 的 double 版本的指针赋给 _proto_eval_sh_legendre_double__t_var
cdef _proto_eval_sh_legendre_double__t *_proto_eval_sh_legendre_double__t_var = &_func_eval_sh_legendre[double]

# 导入自定义的评估函数 eval_sh_legendre_l，命名为 _func_eval_sh_legendre_l
from .orthogonal_eval cimport eval_sh_legendre_l as _func_eval_sh_legendre_l
# 定义一个函数原型 _proto_eval_sh_legendre_l_t，接受一个 long 和一个 double 类型参数，并且不引发异常，在无 GIL 环境下操作
ctypedef double _proto_eval_sh_legendre_l_t(long, double) noexcept nogil
# 将 _func_eval_sh_legendre_l 的指针赋给 _proto_eval_sh_legendre_l_t_var
cdef _proto_eval_sh_legendre_l_t *_proto_eval_sh_legendre_l_t_var = &_func_eval_sh_legendre_l

# 导入自定义的函数 cexpm1，命名为 _func_cexpm1
from ._cunity cimport cexpm1 as _func_cexpm1
# 定义一个复数版本的函数原型 _proto_cexpm1_t，接受一个复数参数，并且不引发异常，在无 GIL 环境下操作
ctypedef double complex _proto_cexpm1_t(double complex) noexcept nogil
# 将 _func_cexpm1 的指针赋给 _proto_cexpm1_t_var
cdef _proto_cexpm1_t *_proto_cexpm1_t_var = &_func_cexpm1

# 导入自定义的函数 expn_unsafe，命名为 _func_expn_unsafe
from ._legacy cimport expn_unsafe as _func_expn_unsafe
# 定义一个函数原型 _proto_expn_unsafe_t，接受两个 double 类型参数，并且不引发异常，在无 GIL 环境下操作
ctypedef double _proto_expn_unsafe_t(double, double) noexcept nogil
# 将 _func_expn_unsafe 的指针赋给 _proto_expn_unsafe_t_var
cdef _proto_expn_unsafe_t *_proto_expn_unsafe_t_var = &_func_expn_unsafe

# 从 _ufuncs_defs.h 中的外部引入 _func_expn 函数的定义
cdef extern from r"_ufuncs_defs.h":
    cdef npy_double _func_expn "expn"(npy_int, npy_double) nogil

# 从 _ufuncs_defs.h 中的外部引入 _func_fdtr 函数的定义
cdef extern from r"_ufuncs_defs.h":
    cdef npy_double _func_fdtr "fdtr"(npy_double, npy_double, npy_double) nogil

# 从 _ufuncs_defs.h 中的外部引入 _func_fdtrc 函数的定义
cdef extern from r"_ufuncs_defs.h":
    cdef npy_double _func_fdtrc "fdtrc"(npy_double, npy_double, npy_double) nogil

# 从 _ufuncs_defs.h 中的外部引入 _func_fdtri 函数的定义
cdef extern from r"_ufuncs_defs.h":
    cdef npy_double _func_fdtri
# 导入 C 语言的函数接口和类型定义模块
from ._cdflib_wrappers cimport fdtridfd as _func_fdtridfd
# 定义 fdtridfd 函数的原型类型为双精度浮点数，接受三个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto_fdtridfd_t(double, double, double) noexcept nogil
# 定义 _proto_fdtridfd_t_var 变量为 fdtridfd 函数的指针
cdef _proto_fdtridfd_t *_proto_fdtridfd_t_var = &_func_fdtridfd

# 从 _ufuncs_defs.h 头文件中导入 cephes_fresnl_wrap 函数的接口
cdef extern from r"_ufuncs_defs.h":
    # 定义 cephes_fresnl_wrap 函数的原型，接受一个双精度浮点数和两个指向双精度浮点数的指针参数，无异常处理，无全局解锁
    cdef npy_int _func_cephes_fresnl_wrap "cephes_fresnl_wrap"(npy_double, npy_double *, npy_double *)nogil

# 从 ._cdflib_wrappers 模块中导入 gdtria 函数
from ._cdflib_wrappers cimport gdtria as _func_gdtria
# 定义 gdtria 函数的原型类型为双精度浮点数，接受三个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto_gdtria_t(double, double, double) noexcept nogil
# 定义 _proto_gdtria_t_var 变量为 gdtria 函数的指针
cdef _proto_gdtria_t *_proto_gdtria_t_var = &_func_gdtria

# 从 ._cdflib_wrappers 模块中导入 gdtrib 函数
from ._cdflib_wrappers cimport gdtrib as _func_gdtrib
# 定义 gdtrib 函数的原型类型为双精度浮点数，接受三个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto_gdtrib_t(double, double, double) noexcept nogil
# 定义 _proto_gdtrib_t_var 变量为 gdtrib 函数的指针
cdef _proto_gdtrib_t *_proto_gdtrib_t_var = &_func_gdtrib

# 从 ._cdflib_wrappers 模块中导入 gdtrix 函数
from ._cdflib_wrappers cimport gdtrix as _func_gdtrix
# 定义 gdtrix 函数的原型类型为双精度浮点数，接受三个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto_gdtrix_t(double, double, double) noexcept nogil
# 定义 _proto_gdtrix_t_var 变量为 gdtrix 函数的指针
cdef _proto_gdtrix_t *_proto_gdtrix_t_var = &_func_gdtrix

# 从 ._convex_analysis 模块中导入 huber 函数
from ._convex_analysis cimport huber as _func_huber
# 定义 huber 函数的原型类型为双精度浮点数，接受两个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto_huber_t(double, double) noexcept nogil
# 定义 _proto_huber_t_var 变量为 huber 函数的指针
cdef _proto_huber_t *_proto_huber_t_var = &_func_huber

# 从 ._hyp0f1 模块中导入 _hyp0f1_cmplx 函数
from ._hyp0f1 cimport _hyp0f1_cmplx as _func__hyp0f1_cmplx
# 定义 _hyp0f1_cmplx 函数的原型类型为复数双精度浮点数，接受一个双精度浮点数和一个复数双精度浮点数参数，无异常处理，无全局解锁
ctypedef double complex _proto__hyp0f1_cmplx_t(double, double complex) noexcept nogil
# 定义 _proto__hyp0f1_cmplx_t_var 变量为 _hyp0f1_cmplx 函数的指针
cdef _proto__hyp0f1_cmplx_t *_proto__hyp0f1_cmplx_t_var = &_func__hyp0f1_cmplx

# 从 ._hyp0f1 模块中导入 _hyp0f1_real 函数
from ._hyp0f1 cimport _hyp0f1_real as _func__hyp0f1_real
# 定义 _hyp0f1_real 函数的原型类型为双精度浮点数，接受两个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto__hyp0f1_real_t(double, double) noexcept nogil
# 定义 _proto__hyp0f1_real_t_var 变量为 _hyp0f1_real 函数的指针
cdef _proto__hyp0f1_real_t *_proto__hyp0f1_real_t_var = &_func__hyp0f1_real

# 从 r"_ufuncs_defs.h" 头文件中导入 chyp1f1_wrap 函数的接口
cdef extern from r"_ufuncs_defs.h":
    # 定义 chyp1f1_wrap 函数的原型，接受一个双精度浮点数和两个复数双精度浮点数参数，无异常处理，无全局解锁
    cdef npy_cdouble _func_chyp1f1_wrap "chyp1f1_wrap"(npy_double, npy_double, npy_cdouble)nogil

# 从 ._hypergeometric 模块中导入 hyperu 函数
from ._hypergeometric cimport hyperu as _func_hyperu
# 定义 hyperu 函数的原型类型为双精度浮点数，接受三个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto_hyperu_t(double, double, double) noexcept nogil
# 定义 _proto_hyperu_t_var 变量为 hyperu 函数的指针
cdef _proto_hyperu_t *_proto_hyperu_t_var = &_func_hyperu

# 从 ._boxcox 模块中导入 inv_boxcox 函数
from ._boxcox cimport inv_boxcox as _func_inv_boxcox
# 定义 inv_boxcox 函数的原型类型为双精度浮点数，接受两个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto_inv_boxcox_t(double, double) noexcept nogil
# 定义 _proto_inv_boxcox_t_var 变量为 inv_boxcox 函数的指针
cdef _proto_inv_boxcox_t *_proto_inv_boxcox_t_var = &_func_inv_boxcox

# 从 ._boxcox 模块中导入 inv_boxcox1p 函数
from ._boxcox cimport inv_boxcox1p as _func_inv_boxcox1p
# 定义 inv_boxcox1p 函数的原型类型为双精度浮点数，接受两个双精度浮点数参数，无异常处理，无全局解锁
ctypedef double _proto_inv_boxcox1p_t(double, double) noexcept nogil
# 定义 _proto_inv_boxcox1p_t_var 变量为 inv_boxcox1p 函数的指针
cdef _proto_inv_boxcox1p_t *_proto_inv_boxcox1p_t_var = &_func_inv_boxcox1p

# 从 r"_ufuncs_defs.h" 头文件中导入 j0 函数的接口
cdef extern from r"_ufuncs_defs.h":
    # 定义 j0 函数的原型，接受一个双精度浮点数参数，无异常处理，无全局解锁
    cdef npy_double _func_j0 "j0"(npy_double)nogil

# 从 r"_ufuncs_defs.h" 头文件
# 定义了一个双精度浮点数类型的函数指针 _proto_kn_unsafe_t，接受两个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_kn_unsafe_t(double, double) noexcept nogil
# 将 _func_kn_unsafe 函数的地址赋值给 _proto_kn_unsafe_t_var 变量
cdef _proto_kn_unsafe_t *_proto_kn_unsafe_t_var = &_func_kn_unsafe

# 从 _cunity 模块中导入 clog1p 函数，并重命名为 _func_clog1p
from ._cunity cimport clog1p as _func_clog1p
# 定义了一个复数类型的函数指针 _proto_clog1p_t，接受一个复数参数，不使用全局解释器锁（GIL）
ctypedef double complex _proto_clog1p_t(double complex) noexcept nogil
# 将 _func_clog1p 函数的地址赋值给 _proto_clog1p_t_var 变量
cdef _proto_clog1p_t *_proto_clog1p_t_var = &_func_clog1p

# 从 _ufuncs_defs.h 头文件中导入 pmv_wrap 函数声明，定义为在没有全局解释器锁（GIL）的情况下接受三个 npy_double 类型参数并返回 npy_double 类型
cdef extern from r"_ufuncs_defs.h":
    cdef npy_double _func_pmv_wrap "pmv_wrap"(npy_double, npy_double, npy_double) nogil

# 从 _legacy 模块中导入 nbdtr_unsafe 函数，并重命名为 _func_nbdtr_unsafe
from ._legacy cimport nbdtr_unsafe as _func_nbdtr_unsafe
# 定义了一个双精度浮点数类型的函数指针 _proto_nbdtr_unsafe_t，接受三个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_nbdtr_unsafe_t(double, double, double) noexcept nogil
# 将 _func_nbdtr_unsafe 函数的地址赋值给 _proto_nbdtr_unsafe_t_var 变量
cdef _proto_nbdtr_unsafe_t *_proto_nbdtr_unsafe_t_var = &_func_nbdtr_unsafe

# 从 _legacy 模块中导入 nbdtrc_unsafe 函数，并重命名为 _func_nbdtrc_unsafe
from ._legacy cimport nbdtrc_unsafe as _func_nbdtrc_unsafe
# 定义了一个双精度浮点数类型的函数指针 _proto_nbdtrc_unsafe_t，接受三个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_nbdtrc_unsafe_t(double, double, double) noexcept nogil
# 将 _func_nbdtrc_unsafe 函数的地址赋值给 _proto_nbdtrc_unsafe_t_var 变量
cdef _proto_nbdtrc_unsafe_t *_proto_nbdtrc_unsafe_t_var = &_func_nbdtrc_unsafe

# 从 _legacy 模块中导入 nbdtri_unsafe 函数，并重命名为 _func_nbdtri_unsafe
from ._legacy cimport nbdtri_unsafe as _func_nbdtri_unsafe
# 定义了一个双精度浮点数类型的函数指针 _proto_nbdtri_unsafe_t，接受三个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_nbdtri_unsafe_t(double, double, double) noexcept nogil
# 将 _func_nbdtri_unsafe 函数的地址赋值给 _proto_nbdtri_unsafe_t_var 变量
cdef _proto_nbdtri_unsafe_t *_proto_nbdtri_unsafe_t_var = &_func_nbdtri_unsafe

# 从 _cdflib_wrappers 模块中导入 nbdtrik 函数，并重命名为 _func_nbdtrik
from ._cdflib_wrappers cimport nbdtrik as _func_nbdtrik
# 定义了一个双精度浮点数类型的函数指针 _proto_nbdtrik_t，接受三个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_nbdtrik_t(double, double, double) noexcept nogil
# 将 _func_nbdtrik 函数的地址赋值给 _proto_nbdtrik_t_var 变量
cdef _proto_nbdtrik_t *_proto_nbdtrik_t_var = &_func_nbdtrik

# 从 _cdflib_wrappers 模块中导入 nbdtrin 函数，并重命名为 _func_nbdtrin
from ._cdflib_wrappers cimport nbdtrin as _func_nbdtrin
# 定义了一个双精度浮点数类型的函数指针 _proto_nbdtrin_t，接受三个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_nbdtrin_t(double, double, double) noexcept nogil
# 将 _func_nbdtrin 函数的地址赋值给 _proto_nbdtrin_t_var 变量
cdef _proto_nbdtrin_t *_proto_nbdtrin_t_var = &_func_nbdtrin

# 从 _cdflib_wrappers 模块中导入 ncfdtr 函数，并重命名为 _func_ncfdtr
from ._cdflib_wrappers cimport ncfdtr as _func_ncfdtr
# 定义了一个双精度浮点数类型的函数指针 _proto_ncfdtr_t，接受四个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_ncfdtr_t(double, double, double, double) noexcept nogil
# 将 _func_ncfdtr 函数的地址赋值给 _proto_ncfdtr_t_var 变量
cdef _proto_ncfdtr_t *_proto_ncfdtr_t_var = &_func_ncfdtr

# 从 _cdflib_wrappers 模块中导入 ncfdtri 函数，并重命名为 _func_ncfdtri
from ._cdflib_wrappers cimport ncfdtri as _func_ncfdtri
# 定义了一个双精度浮点数类型的函数指针 _proto_ncfdtri_t，接受四个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_ncfdtri_t(double, double, double, double) noexcept nogil
# 将 _func_ncfdtri 函数的地址赋值给 _proto_ncfdtri_t_var 变量
cdef _proto_ncfdtri_t *_proto_ncfdtri_t_var = &_func_ncfdtri

# 从 _cdflib_wrappers 模块中导入 ncfdtridfd 函数，并重命名为 _func_ncfdtridfd
from ._cdflib_wrappers cimport ncfdtridfd as _func_ncfdtridfd
# 定义了一个双精度浮点数类型的函数指针 _proto_ncfdtridfd_t，接受四个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_ncfdtridfd_t(double, double, double, double) noexcept nogil
# 将 _func_ncfdtridfd 函数的地址赋值给 _proto_ncfdtridfd_t_var 变量
cdef _proto_ncfdtridfd_t *_proto_ncfdtridfd_t_var = &_func_ncfdtridfd

# 从 _cdflib_wrappers 模块中导入 ncfdtridfn 函数，并重命名为 _func_ncfdtridfn
from ._cdflib_wrappers cimport ncfdtridfn as _func_ncfdtridfn
# 定义了一个双精度浮点数类型的函数指针 _proto_ncfdtridfn_t，接受四个 double 类型参数，不使用全局解释器锁（GIL）
ctypedef double _proto_ncfdtridfn_t(double, double, double, double) noexcept nogil
# 将 _func_ncfdtridfn 函数的地址赋值给 _proto_ncfdtridfn_t_var 变量
cdef _proto_ncfdtridfn_t *_proto_nc
# 导入 C 扩展函数 _func_nctdtrinc 作为 _proto_nctdtrinc_t_var 变量
cdef _proto_nctdtrinc_t *_proto_nctdtrinc_t_var = &_func_nctdtrinc

# 从 _cdflib_wrappers 中导入 C 扩展函数 nctdtrit 并命名为 _func_nctdtrit
from ._cdflib_wrappers cimport nctdtrit as _func_nctdtrit

# 定义 C 函数类型 _proto_nctdtrit_t，接受三个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_nctdtrit_t(double, double, double) noexcept nogil

# 将 _func_nctdtrit 函数赋值给 _proto_nctdtrit_t_var 变量
cdef _proto_nctdtrit_t *_proto_nctdtrit_t_var = &_func_nctdtrit

# 从 _cdflib_wrappers 中导入 C 扩展函数 nrdtrimn 并命名为 _func_nrdtrimn
from ._cdflib_wrappers cimport nrdtrimn as _func_nrdtrimn

# 定义 C 函数类型 _proto_nrdtrimn_t，接受三个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_nrdtrimn_t(double, double, double) noexcept nogil

# 将 _func_nrdtrimn 函数赋值给 _proto_nrdtrimn_t_var 变量
cdef _proto_nrdtrimn_t *_proto_nrdtrimn_t_var = &_func_nrdtrimn

# 从 _cdflib_wrappers 中导入 C 扩展函数 nrdtrisd 并命名为 _func_nrdtrisd
from ._cdflib_wrappers cimport nrdtrisd as _func_nrdtrisd

# 定义 C 函数类型 _proto_nrdtrisd_t，接受三个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_nrdtrisd_t(double, double, double) noexcept nogil

# 将 _func_nrdtrisd 函数赋值给 _proto_nrdtrisd_t_var 变量
cdef _proto_nrdtrisd_t *_proto_nrdtrisd_t_var = &_func_nrdtrisd

# 从 _legacy 中导入 C 扩展函数 pdtri_unsafe 并命名为 _func_pdtri_unsafe
from ._legacy cimport pdtri_unsafe as _func_pdtri_unsafe

# 定义 C 函数类型 _proto_pdtri_unsafe_t，接受两个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_pdtri_unsafe_t(double, double) noexcept nogil

# 将 _func_pdtri_unsafe 函数赋值给 _proto_pdtri_unsafe_t_var 变量
cdef _proto_pdtri_unsafe_t *_proto_pdtri_unsafe_t_var = &_func_pdtri_unsafe

# 从 _cdflib_wrappers 中导入 C 扩展函数 pdtrik 并命名为 _func_pdtrik
from ._cdflib_wrappers cimport pdtrik as _func_pdtrik

# 定义 C 函数类型 _proto_pdtrik_t，接受两个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_pdtrik_t(double, double) noexcept nogil

# 将 _func_pdtrik 函数赋值给 _proto_pdtrik_t_var 变量
cdef _proto_pdtrik_t *_proto_pdtrik_t_var = &_func_pdtrik

# 从 _convex_analysis 中导入 C 扩展函数 pseudo_huber 并命名为 _func_pseudo_huber
from ._convex_analysis cimport pseudo_huber as _func_pseudo_huber

# 定义 C 函数类型 _proto_pseudo_huber_t，接受两个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_pseudo_huber_t(double, double) noexcept nogil

# 将 _func_pseudo_huber 函数赋值给 _proto_pseudo_huber_t_var 变量
cdef _proto_pseudo_huber_t *_proto_pseudo_huber_t_var = &_func_pseudo_huber

# 从 _convex_analysis 中导入 C 扩展函数 rel_entr 并命名为 _func_rel_entr
from ._convex_analysis cimport rel_entr as _func_rel_entr

# 定义 C 函数类型 _proto_rel_entr_t，接受两个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_rel_entr_t(double, double) noexcept nogil

# 将 _func_rel_entr 函数赋值给 _proto_rel_entr_t_var 变量
cdef _proto_rel_entr_t *_proto_rel_entr_t_var = &_func_rel_entr

# 从 _sici 中导入 C 扩展函数 cshichi 并命名为 _func_cshichi
from ._sici cimport cshichi as _func_cshichi

# 定义 C 函数类型 _proto_cshichi_t，接受一个复数和两个复数指针参数，不抛出异常，不使用 GIL
ctypedef int _proto_cshichi_t(double complex, double complex *, double complex *) noexcept nogil

# 将 _func_cshichi 函数赋值给 _proto_cshichi_t_var 变量
cdef _proto_cshichi_t *_proto_cshichi_t_var = &_func_cshichi

# 从 _sici 中导入 C 扩展函数 csici 并命名为 _func_csici
from ._sici cimport csici as _func_csici

# 定义 C 函数类型 _proto_csici_t，接受一个复数和两个复数指针参数，不抛出异常，不使用 GIL
ctypedef int _proto_csici_t(double complex, double complex *, double complex *) noexcept nogil

# 将 _func_csici 函数赋值给 _proto_csici_t_var 变量
cdef _proto_csici_t *_proto_csici_t_var = &_func_csici

# 从 _legacy 中导入 C 扩展函数 smirnov_unsafe 并命名为 _func_smirnov_unsafe
from ._legacy cimport smirnov_unsafe as _func_smirnov_unsafe

# 定义 C 函数类型 _proto_smirnov_unsafe_t，接受两个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_smirnov_unsafe_t(double, double) noexcept nogil

# 将 _func_smirnov_unsafe 函数赋值给 _proto_smirnov_unsafe_t_var 变量
cdef _proto_smirnov_unsafe_t *_proto_smirnov_unsafe_t_var = &_func_smirnov_unsafe

# 从 _legacy 中导入 C 扩展函数 smirnovi_unsafe 并命名为 _func_smirnovi_unsafe
from ._legacy cimport smirnovi_unsafe as _func_smirnovi_unsafe

# 定义 C 函数类型 _proto_smirnovi_unsafe_t，接受两个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_smirnovi_unsafe_t(double, double) noexcept nogil

# 将 _func_smirnovi_unsafe 函数赋值给 _proto_smirnovi_unsafe_t_var 变量
cdef _proto_smirnovi_unsafe_t *_proto_smirnovi_unsafe_t_var = &_func_smirnovi_unsafe

# 从 _spence 中导入 C 扩展函数 cspence 并命名为 _func_cspence
from ._spence cimport cspence as _func_cspence

# 定义 C 函数类型 _proto_cspence_t，接受一个复数参数，不抛出异常，不使用 GIL
ctypedef double complex _proto_cspence_t(double complex) noexcept nogil

# 将 _func_cspence 函数赋值给 _proto_cspence_t_var 变量
cdef _proto_cspence_t *_proto_cspence_t_var = &_func_cspence

# 从 _cdflib_wrappers 中导入 C 扩展函数 stdtr 并命名为 _func_stdtr
from ._cdflib_wrappers cimport stdtr as _func_stdtr

# 定义 C 函数类型 _proto_stdtr_t，接受两个 double 类型参数，不抛出异常，不使用 GIL
ctypedef double _proto_std
# 导入 C 函数签名并声明一个指针变量 _proto_stdtridf_t_var，指向 _func_stdtridf 函数
ctypedef double _proto_stdtridf_t(double, double) noexcept nogil
cdef _proto_stdtridf_t *_proto_stdtridf_t_var = &_func_stdtridf

# 从 _cdflib_wrappers 模块中导入 stdtrit 函数作为 _func_stdtrit
from ._cdflib_wrappers cimport stdtrit as _func_stdtrit

# 导入 C 函数签名并声明一个指针变量 _proto_stdtrit_t_var，指向 _func_stdtrit 函数
ctypedef double _proto_stdtrit_t(double, double) noexcept nogil
cdef _proto_stdtrit_t *_proto_stdtrit_t_var = &_func_stdtrit

# 从 _xlogy 模块中导入 xlog1py 函数作为 _func_xlog1py
from ._xlogy cimport xlog1py as _func_xlog1py

# 导入 C 函数签名并声明一个指针变量 _proto_xlog1py_double__t_var，指向 _func_xlog1py[double] 函数
ctypedef double _proto_xlog1py_double__t(double, double) noexcept nogil
cdef _proto_xlog1py_double__t *_proto_xlog1py_double__t_var = &_func_xlog1py[double]

# 从 _xlogy 模块中导入 xlog1py 函数作为 _func_xlog1py
from ._xlogy cimport xlog1py as _func_xlog1py

# 导入 C 函数签名并声明一个指针变量 _proto_xlog1py_double_complex__t_var，指向 _func_xlog1py[double_complex] 函数
ctypedef double complex _proto_xlog1py_double_complex__t(double complex, double complex) noexcept nogil
cdef _proto_xlog1py_double_complex__t *_proto_xlog1py_double_complex__t_var = &_func_xlog1py[double_complex]

# 从 _xlogy 模块中导入 xlogy 函数作为 _func_xlogy
from ._xlogy cimport xlogy as _func_xlogy

# 导入 C 函数签名并声明一个指针变量 _proto_xlogy_double__t_var，指向 _func_xlogy[double] 函数
ctypedef double _proto_xlogy_double__t(double, double) noexcept nogil
cdef _proto_xlogy_double__t *_proto_xlogy_double__t_var = &_func_xlogy[double]

# 从 _xlogy 模块中导入 xlogy 函数作为 _func_xlogy
from ._xlogy cimport xlogy as _func_xlogy

# 导入 C 函数签名并声明一个指针变量 _proto_xlogy_double_complex__t_var，指向 _func_xlogy[double_complex] 函数
ctypedef double complex _proto_xlogy_double_complex__t(double complex, double complex) noexcept nogil
cdef _proto_xlogy_double_complex__t *_proto_xlogy_double_complex__t_var = &_func_xlogy[double_complex]

# 从 _legacy 模块中导入 yn_unsafe 函数作为 _func_yn_unsafe
from ._legacy cimport yn_unsafe as _func_yn_unsafe

# 导入 C 函数签名并声明一个指针变量 _proto_yn_unsafe_t_var，指向 _func_yn_unsafe 函数
ctypedef double _proto_yn_unsafe_t(double, double) noexcept nogil
cdef _proto_yn_unsafe_t *_proto_yn_unsafe_t_var = &_func_yn_unsafe

# 从 _ndtri_exp 模块中导入 ndtri_exp 函数作为 _func_ndtri_exp
from ._ndtri_exp cimport ndtri_exp as _func_ndtri_exp

# 导入 C 函数签名并声明一个指针变量 _proto_ndtri_exp_t_var，指向 _func_ndtri_exp 函数
ctypedef double _proto_ndtri_exp_t(double) noexcept nogil
cdef _proto_ndtri_exp_t *_proto_ndtri_exp_t_var = &_func_ndtri_exp

# 定义一个 Cython 函数 voigt_profile，调用 scipy.special._ufuncs_cxx._export_faddeeva_voigt_profile 函数
# 用于计算 Voigt profile 函数
cpdef double voigt_profile(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.voigt_profile"""
    return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_faddeeva_voigt_profile)(x0, x1, x2)

# 定义一个 Cython 函数 agm，调用 _func_agm 函数，计算算术几何平均数
cpdef double agm(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.agm"""
    return _func_agm(x0, x1)

# 定义一个 Cython 辅助函数 airy，实现特定条件下的 Airy 函数计算
cdef void airy(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1, Dd_number_t *y2, Dd_number_t *y3) noexcept nogil:
    """See the documentation for scipy.special.airy"""
    cdef npy_cdouble tmp0
    cdef npy_cdouble tmp1
    cdef npy_cdouble tmp2
    cdef npy_cdouble tmp3
    # 根据不同的数值类型调用不同的 Airy 函数实现
    if Dd_number_t is double:
        special_airy(x0, y0, y1, y2, y3)
    elif Dd_number_t is double_complex:
        special_cairy(_complexstuff.npy_cdouble_from_double_complex(x0), &tmp0, &tmp1, &tmp2, &tmp3)
        y0[0] = _complexstuff.double_complex_from_npy_cdouble(tmp0)
        y1[0] = _complexstuff.double_complex_from_npy_cdouble(tmp1)
        y2[0] = _complexstuff.double_complex_from_npy_cdouble(tmp2)
        y3[0] = _complexstuff.double_complex_from_npy_cdouble(tmp3)
    else:
        # 如果条件不满足（即 Dd_number_t 不是 double_complex 类型），执行以下代码块
        if Dd_number_t is double_complex:
            # 如果 Dd_number_t 是 double_complex 类型，则以下代码不执行，因为条件不满足
            y0[0] = NAN
            y1[0] = NAN
            y2[0] = NAN
            y3[0] = NAN
        else:
            # 如果 Dd_number_t 不是 double_complex 类型，则执行以下代码块
            y0[0] = NAN
            y1[0] = NAN
            y2[0] = NAN
            y3[0] = NAN
# 定义一个函数 _airy_pywrap，接受一个 Dd_number_t 类型的参数 x0
def _airy_pywrap(Dd_number_t x0):
    # 声明四个变量，都是 Dd_number_t 类型
    cdef Dd_number_t y0
    cdef Dd_number_t y1
    cdef Dd_number_t y2
    cdef Dd_number_t y3
    # 调用 airy 函数，将结果存储在 y0, y1, y2, y3 中
    airy(x0, &y0, &y1, &y2, &y3)
    # 返回 y0, y1, y2, y3 四个值
    return y0, y1, y2, y3

# 声明一个内联函数 airye，接受五个参数，它们都是 Dd_number_t 类型的指针，并且不抛出异常，没有 GIL
cdef void airye(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1, Dd_number_t *y2, Dd_number_t *y3) noexcept nogil:
    """See the documentation for scipy.special.airye"""
    # 声明四个临时变量，都是 npy_cdouble 类型
    cdef npy_cdouble tmp0
    cdef npy_cdouble tmp1
    cdef npy_cdouble tmp2
    cdef npy_cdouble tmp3
    # 如果 Dd_number_t 是 double_complex 类型
    if Dd_number_t is double_complex:
        # 调用 special_cairye 函数，将结果存储在 tmp0, tmp1, tmp2, tmp3 中
        special_cairye(_complexstuff.npy_cdouble_from_double_complex(x0), &tmp0, &tmp1, &tmp2, &tmp3)
        # 将 tmp0, tmp1, tmp2, tmp3 转换为 double_complex 类型，存储在 y0, y1, y2, y3 中
        y0[0] = _complexstuff.double_complex_from_npy_cdouble(tmp0)
        y1[0] = _complexstuff.double_complex_from_npy_cdouble(tmp1)
        y2[0] = _complexstuff.double_complex_from_npy_cdouble(tmp2)
        y3[0] = _complexstuff.double_complex_from_npy_cdouble(tmp3)
    # 否则如果 Dd_number_t 是 double 类型
    elif Dd_number_t is double:
        # 调用 special_airye 函数，将结果存储在 y0, y1, y2, y3 中
        special_airye(x0, y0, y1, y2, y3)
    else:
        # 如果 Dd_number_t 不是 double 或 double_complex 类型，则将 y0, y1, y2, y3 设置为 NAN
        y0[0] = NAN
        y1[0] = NAN
        y2[0] = NAN
        y3[0] = NAN

# 定义一个函数 _airye_pywrap，接受一个 Dd_number_t 类型的参数 x0
def _airye_pywrap(Dd_number_t x0):
    # 声明四个变量，都是 Dd_number_t 类型
    cdef Dd_number_t y0
    cdef Dd_number_t y1
    cdef Dd_number_t y2
    cdef Dd_number_t y3
    # 调用 airye 函数，将结果存储在 y0, y1, y2, y3 中
    airye(x0, &y0, &y1, &y2, &y3)
    # 返回 y0, y1, y2, y3 四个值
    return y0, y1, y2, y3

# 声明一个 Cython Python 可调用的函数 bdtr，接受三个参数，并且不抛出异常，没有 GIL
cpdef double bdtr(double x0, dl_number_t x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.bdtr"""
    # 如果 dl_number_t 是 double 类型
    if dl_number_t is double:
        # 调用 _func_bdtr_unsafe 函数，返回结果
        return _func_bdtr_unsafe(x0, x1, x2)
    # 否则如果 dl_number_t 是 long 类型
    elif dl_number_t is long:
        # 调用 cephes_bdtr 函数，返回结果
        return cephes_bdtr(x0, x1, x2)
    else:
        # 如果 dl_number_t 不是 double 或 long 类型，则返回 NAN
        return NAN

# 声明一个 Cython Python 可调用的函数 bdtrc，接受三个参数，并且不抛出异常，没有 GIL
cpdef double bdtrc(double x0, dl_number_t x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.bdtrc"""
    # 如果 dl_number_t 是 double 类型
    if dl_number_t is double:
        # 调用 _func_bdtrc_unsafe 函数，返回结果
        return _func_bdtrc_unsafe(x0, x1, x2)
    # 否则如果 dl_number_t 是 long 类型
    elif dl_number_t is long:
        # 调用 cephes_bdtrc 函数，返回结果
        return cephes_bdtrc(x0, x1, x2)
    else:
        # 如果 dl_number_t 不是 double 或 long 类型，则返回 NAN
        return NAN

# 声明一个 Cython Python 可调用的函数 bdtri，接受三个参数，并且不抛出异常，没有 GIL
cpdef double bdtri(double x0, dl_number_t x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.bdtri"""
    # 如果 dl_number_t 是 double 类型
    if dl_number_t is double:
        # 调用 _func_bdtri_unsafe 函数，返回结果
        return _func_bdtri_unsafe(x0, x1, x2)
    # 否则如果 dl_number_t 是 long 类型
    elif dl_number_t is long:
        # 调用 cephes_bdtri 函数，返回结果
        return cephes_bdtri(x0, x1, x2)
    else:
        # 如果 dl_number_t 不是 double 或 long 类型，则返回 NAN
        return NAN

# 声明一个 Cython Python 可调用的函数 bdtrik，接受三个 double 类型参数，并且不抛出异常，没有 GIL
cpdef double bdtrik(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.bdtrik"""
    # 调用 _func_bdtrik 函数，返回结果
    return _func_bdtrik(x0, x1, x2)

# 声明一个 Cython Python 可调用的函数 bdtrin，接受三个 double 类型参数，并且不抛出异常，没有 GIL
cpdef double bdtrin(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.bdtrin"""
    # 调用 _func_bdtrin 函数，返回结果
    return _func_bdtrin(x0, x1, x2)

# 声明一个 Cython Python 可调用的函数 bei，接受一个 double 类型参数，并且不抛出异常，没有 GIL
cpdef double bei(double x0) noexcept nogil:
    """See the documentation for scipy.special.bei"""
    # 调用 special_bei 函数，返回结果
    return special_bei(x0)

# 声明一个 Cython Python 可调用的函数 beip，接受一个 double 类型参数，并且不抛出异常，没有 GIL
cpdef double beip(double x0) noexcept nogil:
    """See the documentation for scipy.special.beip"""
    # 调用 special_beip 函数，返回结果
    return special_beip(x0)

# 声明一个 Cython Python 可调用的函数 ber，接受一个 double 类型参数，并且不抛出异常，没有 GIL
cpdef double ber(double x0) noexcept nogil:
    """See the documentation for scipy.special.ber"""
    # 返回特殊的 BER 编码结果，参数为 x0
    return special_ber(x0)
// 调用 Scipy 库中的特殊函数 scipy.special.berp，计算 berp 函数的值
cpdef double berp(double x0) noexcept nogil:
    """See the documentation for scipy.special.berp"""
    return special_berp(x0)

// 调用 Scipy 库中的特殊函数 scipy.special.besselpoly，计算 besselpoly 函数的值
cpdef double besselpoly(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.besselpoly"""
    return cephes_besselpoly(x0, x1, x2)

// 调用 Scipy 库中的特殊函数 scipy.special.beta，计算 beta 函数的值
cpdef double beta(double x0, double x1) noexcept nogil:
    return cephes_beta(x0, x1)

// 调用 Scipy 库中的特殊函数 scipy.special.betainc，计算 betainc 函数的值
cpdef df_number_t betainc(df_number_t x0, df_number_t x1, df_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.betainc"""
    // 根据输入类型选择合适的底层 C++ 函数进行计算
    if df_number_t is float:
        return (<float(*)(float, float, float) noexcept nogil>scipy.special._ufuncs_cxx._export_ibeta_float)(x0, x1, x2)
    elif df_number_t is double:
        return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_ibeta_double)(x0, x1, x2)
    else:
        // 如果输入类型不支持，则返回 NaN
        if df_number_t is double:
            return NAN
        else:
            return NAN

// 调用 Scipy 库中的特殊函数 scipy.special.betaincc，计算 betaincc 函数的值
cpdef df_number_t betaincc(df_number_t x0, df_number_t x1, df_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.betaincc"""
    // 根据输入类型选择合适的底层 C++ 函数进行计算
    if df_number_t is float:
        return (<float(*)(float, float, float) noexcept nogil>scipy.special._ufuncs_cxx._export_ibetac_float)(x0, x1, x2)
    elif df_number_t is double:
        return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_ibetac_double)(x0, x1, x2)
    else:
        // 如果输入类型不支持，则返回 NaN
        if df_number_t is double:
            return NAN
        else:
            return NAN

// 调用 Scipy 库中的特殊函数 scipy.special.betaincinv，计算 betaincinv 函数的值
cpdef df_number_t betaincinv(df_number_t x0, df_number_t x1, df_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.betaincinv"""
    // 根据输入类型选择合适的底层 C++ 函数进行计算
    if df_number_t is float:
        return (<float(*)(float, float, float) noexcept nogil>scipy.special._ufuncs_cxx._export_ibeta_inv_float)(x0, x1, x2)
    elif df_number_t is double:
        return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_ibeta_inv_double)(x0, x1, x2)
    else:
        // 如果输入类型不支持，则返回 NaN
        if df_number_t is double:
            return NAN
        else:
            return NAN

// 调用 Scipy 库中的特殊函数 scipy.special.betainccinv，计算 betainccinv 函数的值
cpdef df_number_t betainccinv(df_number_t x0, df_number_t x1, df_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.betainccinv"""
    // 根据输入类型选择合适的底层 C++ 函数进行计算
    if df_number_t is float:
        return (<float(*)(float, float, float) noexcept nogil>scipy.special._ufuncs_cxx._export_ibetac_inv_float)(x0, x1, x2)
    elif df_number_t is double:
        return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_ibetac_inv_double)(x0, x1, x2)
    else:
        // 如果输入类型不支持，则返回 NaN
        if df_number_t is double:
            return NAN
        else:
            return NAN

// 调用 Scipy 库中的特殊函数 scipy.special.betaln，计算 betaln 函数的值
cpdef double betaln(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.betaln"""
    return cephes_lbeta(x0, x1)

// 调用 Scipy 库中的特殊函数 scipy.special.binom，计算 binom 函数的值
cpdef double binom(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.binom"""
    return special_binom(x0, x1)
cpdef double boxcox(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.boxcox"""
    return _func_boxcox(x0, x1)

cpdef double boxcox1p(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.boxcox1p"""
    return _func_boxcox1p(x0, x1)

cpdef double btdtr(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.btdtr"""
    return cephes_btdtr(x0, x1, x2)

cpdef double btdtri(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.btdtri"""
    return cephes_btdtri(x0, x1, x2)

cpdef double btdtria(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.btdtria"""
    return _func_btdtria(x0, x1, x2)

cpdef double btdtrib(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.btdtrib"""
    return _func_btdtrib(x0, x1, x2)

cpdef double cbrt(double x0) noexcept nogil:
    """See the documentation for scipy.special.cbrt"""
    return cephes_cbrt(x0)

cpdef double chdtr(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.chdtr"""
    return cephes_chdtr(x0, x1)

cpdef double chdtrc(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.chdtrc"""
    return cephes_chdtrc(x0, x1)

cpdef double chdtri(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.chdtri"""
    return cephes_chdtri(x0, x1)

cpdef double chdtriv(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.chdtriv"""
    return _func_chdtriv(x0, x1)

cpdef double chndtr(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.chndtr"""
    return _func_chndtr(x0, x1, x2)

cpdef double chndtridf(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.chndtridf"""
    return _func_chndtridf(x0, x1, x2)

cpdef double chndtrinc(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.chndtrinc"""
    return _func_chndtrinc(x0, x1, x2)

cpdef double chndtrix(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.chndtrix"""
    return _func_chndtrix(x0, x1, x2)

cpdef double cosdg(double x0) noexcept nogil:
    """See the documentation for scipy.special.cosdg"""
    return cephes_cosdg(x0)

cpdef double cosm1(double x0) noexcept nogil:
    """See the documentation for scipy.special.cosm1"""
    return cephes_cosm1(x0)

cpdef double cotdg(double x0) noexcept nogil:
    """See the documentation for scipy.special.cotdg"""
    return cephes_cotdg(x0)

cpdef Dd_number_t dawsn(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.dawsn"""
    if Dd_number_t is double:
        return (<double(*)(double) noexcept nogil>scipy.special._ufuncs_cxx._export_faddeeva_dawsn)(x0)
    # 如果 Dd_number_t 类型是 double_complex，则执行以下逻辑
    elif Dd_number_t is double_complex:
        # 调用 C++ 函数，返回 Faddeeva 函数的特定处理结果
        return (<double complex(*)(double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_faddeeva_dawsn_complex)(x0)
    # 如果不是 double_complex 类型，则执行以下逻辑
    else:
        # 如果 Dd_number_t 类型是 double_complex，则返回 Not a Number (NaN)
        if Dd_number_t is double_complex:
            return NAN
        # 如果不是 double_complex 类型，则同样返回 Not a Number (NaN)
        else:
            return NAN
# 使用 CPython 语法定义一个 C 函数，返回 scipy.special.ellipe 的值，无异常处理，无 GIL
cpdef double ellipe(double x0) noexcept nogil:
    """See the documentation for scipy.special.ellipe"""
    return cephes_ellpe(x0)

# 使用 CPython 语法定义一个 C 函数，返回 scipy.special.ellipeinc 的值，无异常处理，无 GIL
cpdef double ellipeinc(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.ellipeinc"""
    return cephes_ellie(x0, x1)

# 使用 CPython 语法定义一个 C 函数，计算 scipy.special.ellipj，将结果通过指针输出，无异常处理，无 GIL
cdef void ellipj(double x0, double x1, double *y0, double *y1, double *y2, double *y3) noexcept nogil:
    """See the documentation for scipy.special.ellipj"""
    _func_cephes_ellpj_wrap(x0, x1, y0, y1, y2, y3)

# Python 封装函数，调用 C 函数 ellipj，返回其计算结果
def _ellipj_pywrap(double x0, double x1):
    cdef double y0
    cdef double y1
    cdef double y2
    cdef double y3
    ellipj(x0, x1, &y0, &y1, &y2, &y3)
    return y0, y1, y2, y3

# 使用 CPython 语法定义一个 C 函数，返回 scipy.special.ellipkinc 的值，无异常处理，无 GIL
cpdef double ellipkinc(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.ellipkinc"""
    return cephes_ellik(x0, x1)

# 使用 CPython 语法定义一个 C 函数，返回 scipy.special.ellipkm1 的值，无异常处理，无 GIL
cpdef double ellipkm1(double x0) noexcept nogil:
    """See the documentation for scipy.special.ellipkm1"""
    return cephes_ellpk(x0)

# 使用 CPython 语法定义一个 C 函数，返回 scipy.special.ellipk 的值，无异常处理，无 GIL
cpdef double ellipk(double x0) noexcept nogil:
    """See the documentation for scipy.special.ellipk"""
    return special_ellipk(x0)

# 使用 CPython 语法定义一个 C++ 函数，返回 scipy.special.elliprc 的值，根据类型选择合适的函数并处理异常，无 GIL
cpdef Dd_number_t elliprc(Dd_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.elliprc"""
    if Dd_number_t is double:
        return (<double(*)(double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_fellint_RC)(x0, x1)
    elif Dd_number_t is double_complex:
        return (<double complex(*)(double complex, double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_cellint_RC)(x0, x1)
    else:
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 使用 CPython 语法定义一个 C++ 函数，返回 scipy.special.elliprd 的值，根据类型选择合适的函数并处理异常，无 GIL
cpdef Dd_number_t elliprd(Dd_number_t x0, Dd_number_t x1, Dd_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.elliprd"""
    if Dd_number_t is double:
        return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_fellint_RD)(x0, x1, x2)
    elif Dd_number_t is double_complex:
        return (<double complex(*)(double complex, double complex, double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_cellint_RD)(x0, x1, x2)
    else:
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 使用 CPython 语法定义一个 C++ 函数，返回 scipy.special.elliprf 的值，根据类型选择合适的函数并处理异常，无 GIL
cpdef Dd_number_t elliprf(Dd_number_t x0, Dd_number_t x1, Dd_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.elliprf"""
    if Dd_number_t is double:
        return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_fellint_RF)(x0, x1, x2)
    elif Dd_number_t is double_complex:
        return (<double complex(*)(double complex, double complex, double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_cellint_RF)(x0, x1, x2)
    else:
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 使用 CPython 语法定义一个 C++ 函数，返回 scipy.special.elliprg 的值，根据类型选择合适的函数并处理异常，无 GIL
cpdef Dd_number_t elliprg(Dd_number_t x0, Dd_number_t x1, Dd_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.elliprg"""
    # 如果 Dd_number_t 的类型是 double，则调用相应的双精度函数，返回结果
    if Dd_number_t is double:
        return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_fellint_RG)(x0, x1, x2)
    # 如果 Dd_number_t 的类型是 double_complex，则调用相应的复双精度函数，返回结果
    elif Dd_number_t is double_complex:
        return (<double complex(*)(double complex, double complex, double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_cellint_RG)(x0, x1, x2)
    else:
        # 如果 Dd_number_t 不是 double_complex，则返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN
cpdef Dd_number_t elliprj(Dd_number_t x0, Dd_number_t x1, Dd_number_t x2, Dd_number_t x3) noexcept nogil:
    """See the documentation for scipy.special.elliprj"""
    如果 Dd_number_t 是 double 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_fellint_RJ，传入参数 x0, x1, x2, x3，并返回结果
    elif Dd_number_t 是 double_complex 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_cellint_RJ，传入参数 x0, x1, x2, x3，并返回结果
    else:
        如果 Dd_number_t 是 double_complex 类型：
            返回 NaN
        else:
            返回 NaN

cpdef double entr(double x0) noexcept nogil:
    """See the documentation for scipy.special.entr"""
    调用 C++ 函数 _func_entr，传入参数 x0，并返回结果

cpdef Dd_number_t erf(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.erf"""
    如果 Dd_number_t 是 double_complex 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_faddeeva_erf，传入参数 x0，并返回结果
    elif Dd_number_t 是 double 类型：
        调用 cephes_erf 函数，传入参数 x0，并返回结果
    else:
        如果 Dd_number_t 是 double_complex 类型：
            返回 NaN
        else:
            返回 NaN

cpdef Dd_number_t erfc(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.erfc"""
    如果 Dd_number_t 是 double_complex 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_faddeeva_erfc_complex，传入参数 x0，并返回结果
    elif Dd_number_t 是 double 类型：
        调用 cephes_erfc 函数，传入参数 x0，并返回结果
    else:
        如果 Dd_number_t 是 double_complex 类型：
            返回 NaN
        else:
            返回 NaN

cpdef Dd_number_t erfcx(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.erfcx"""
    如果 Dd_number_t 是 double 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_faddeeva_erfcx，传入参数 x0，并返回结果
    elif Dd_number_t 是 double_complex 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_faddeeva_erfcx_complex，传入参数 x0，并返回结果
    else:
        如果 Dd_number_t 是 double_complex 类型：
            返回 NaN
        else:
            返回 NaN

cpdef Dd_number_t erfi(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.erfi"""
    如果 Dd_number_t 是 double 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_faddeeva_erfi，传入参数 x0，并返回结果
    elif Dd_number_t 是 double_complex 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_faddeeva_erfi_complex，传入参数 x0，并返回结果
    else:
        如果 Dd_number_t 是 double_complex 类型：
            返回 NaN
        else:
            返回 NaN

cpdef df_number_t erfinv(df_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.erfinv"""
    如果 df_number_t 是 float 类型：
        调用 C++ 函数 scipy.special._ufuncs_cxx._export_erfinv_float，传入参数 x0，并返回结果
    # 如果 df_number_t 是 double 类型
    elif df_number_t is double:
        # 调用 C++ 函数导出的 erfinv 函数，并返回结果
        return (<double(*)(double) noexcept nogil>scipy.special._ufuncs_cxx._export_erfinv_double)(x0)
    # 如果 df_number_t 不是 double 类型
    else:
        # 如果 df_number_t 是 double 类型，返回 NaN
        if df_number_t is double:
            return NAN
        # 如果 df_number_t 不是 double 类型，也返回 NaN
        else:
            return NAN
# 定义一个 CPython 函数，返回 scipy.special.erfcinv 的文档注释所描述的功能
cpdef double erfcinv(double x0) noexcept nogil:
    return cephes_erfcinv(x0)

# 定义一个 CPython 函数，根据参数类型选择不同的特定函数来评估 Chebyshev 多项式
cpdef Dd_number_t eval_chebyc(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_chebyc"""
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_chebyc[double_complex](x0, x1)
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_chebyc[double](x0, x1)
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_chebyc_l(x0, x1)
    else:
        # 处理未知类型情况，返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个 CPython 函数，根据参数类型选择不同的特定函数来评估 Chebyshev S 多项式
cpdef Dd_number_t eval_chebys(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_chebys"""
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_chebys[double_complex](x0, x1)
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_chebys[double](x0, x1)
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_chebys_l(x0, x1)
    else:
        # 处理未知类型情况，返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个 CPython 函数，根据参数类型选择不同的特定函数来评估 Chebyshev T 多项式
cpdef Dd_number_t eval_chebyt(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_chebyt"""
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_chebyt[double_complex](x0, x1)
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_chebyt[double](x0, x1)
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_chebyt_l(x0, x1)
    else:
        # 处理未知类型情况，返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个 CPython 函数，根据参数类型选择不同的特定函数来评估 Chebyshev U 多项式
cpdef Dd_number_t eval_chebyu(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_chebyu"""
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_chebyu[double_complex](x0, x1)
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_chebyu[double](x0, x1)
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_chebyu_l(x0, x1)
    else:
        # 处理未知类型情况，返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个 CPython 函数，根据参数类型选择不同的特定函数来评估 Gegenbauer 多项式
cpdef Dd_number_t eval_gegenbauer(dl_number_t x0, double x1, Dd_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.eval_gegenbauer"""
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_gegenbauer[double_complex](x0, x1, x2)
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_gegenbauer[double](x0, x1, x2)
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_gegenbauer_l(x0, x1, x2)
    else:
        # 如果条件不满足第一个条件，则进入这个分支
        if Dd_number_t is double_complex:
            # 如果 Dd_number_t 的类型是 double_complex，则返回 NaN
            return NAN
        else:
            # 如果 Dd_number_t 的类型不是 double_complex，则也返回 NaN
            return NAN
# 根据输入的参数类型和函数功能选择合适的 eval_genlaguerre 函数进行计算并返回结果
cpdef Dd_number_t eval_genlaguerre(dl_number_t x0, double x1, Dd_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.eval_genlaguerre"""
    # 如果 dl_number_t 是 double 而 Dd_number_t 是 double_complex，调用相应的复数类型函数
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_genlaguerre[double_complex](x0, x1, x2)
    # 如果 dl_number_t 是 double 而 Dd_number_t 是 double，调用相应的双精度浮点数函数
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_genlaguerre[double](x0, x1, x2)
    # 如果 dl_number_t 是 long 而 Dd_number_t 是 double，调用相应的长整型和双精度浮点数函数
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_genlaguerre_l(x0, x1, x2)
    else:
        # 如果 Dd_number_t 是 double_complex，则返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            # 否则返回 NaN
            return NAN

# 根据输入的参数调用 eval_hermite 函数进行计算并返回结果
cpdef double eval_hermite(long x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.eval_hermite"""
    return _func_eval_hermite(x0, x1)

# 根据输入的参数调用 eval_hermitenorm 函数进行计算并返回结果
cpdef double eval_hermitenorm(long x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.eval_hermitenorm"""
    return _func_eval_hermitenorm(x0, x1)

# 根据输入的参数类型和函数功能选择合适的 eval_jacobi 函数进行计算并返回结果
cpdef Dd_number_t eval_jacobi(dl_number_t x0, double x1, double x2, Dd_number_t x3) noexcept nogil:
    """See the documentation for scipy.special.eval_jacobi"""
    # 如果 dl_number_t 是 double 而 Dd_number_t 是 double_complex，调用相应的复数类型函数
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_jacobi[double_complex](x0, x1, x2, x3)
    # 如果 dl_number_t 是 double 而 Dd_number_t 是 double，调用相应的双精度浮点数函数
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_jacobi[double](x0, x1, x2, x3)
    # 如果 dl_number_t 是 long 而 Dd_number_t 是 double，调用相应的长整型和双精度浮点数函数
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_jacobi_l(x0, x1, x2, x3)
    else:
        # 如果 Dd_number_t 是 double_complex，则返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            # 否则返回 NaN
            return NAN

# 根据输入的参数类型和函数功能选择合适的 eval_laguerre 函数进行计算并返回结果
cpdef Dd_number_t eval_laguerre(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_laguerre"""
    # 如果 dl_number_t 是 double 而 Dd_number_t 是 double_complex，调用相应的复数类型函数
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_laguerre[double_complex](x0, x1)
    # 如果 dl_number_t 是 double 而 Dd_number_t 是 double，调用相应的双精度浮点数函数
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_laguerre[double](x0, x1)
    # 如果 dl_number_t 是 long 而 Dd_number_t 是 double，调用相应的长整型和双精度浮点数函数
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_laguerre_l(x0, x1)
    else:
        # 如果 Dd_number_t 是 double_complex，则返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            # 否则返回 NaN
            return NAN

# 根据输入的参数类型和函数功能选择合适的 eval_legendre 函数进行计算并返回结果
cpdef Dd_number_t eval_legendre(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_legendre"""
    # 如果 dl_number_t 是 double 而 Dd_number_t 是 double_complex，调用相应的复数类型函数
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_legendre[double_complex](x0, x1)
    # 如果 dl_number_t 是 double 而 Dd_number_t 是 double，调用相应的双精度浮点数函数
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_legendre[double](x0, x1)
    # 如果 dl_number_t 是 long 而 Dd_number_t 是 double，调用相应的长整型和双精度浮点数函数
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_legendre_l(x0, x1)
    else:
        # 如果 Dd_number_t 是 double_complex，则返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            # 否则返回 NaN
            return NAN

# 根据输入的参数类型和函数功能选择合适的 eval_sh_chebyt 函数进行计算并返回结果
cpdef Dd_number_t eval_sh_chebyt(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_sh_chebyt"""
    # 检查 dl_number_t 是否为 double 类型，Dd_number_t 是否为 double_complex 类型
    if dl_number_t is double and Dd_number_t is double_complex:
        # 如果满足条件，调用 _func_eval_sh_chebyt[double_complex] 函数并返回结果
        return _func_eval_sh_chebyt[double_complex](x0, x1)
    # 检查 dl_number_t 是否为 double 类型，Dd_number_t 是否为 double 类型
    elif dl_number_t is double and Dd_number_t is double:
        # 如果满足条件，调用 _func_eval_sh_chebyt[double] 函数并返回结果
        return _func_eval_sh_chebyt[double](x0, x1)
    # 检查 dl_number_t 是否为 long 类型，Dd_number_t 是否为 double 类型
    elif dl_number_t is long and Dd_number_t is double:
        # 如果满足条件，调用 _func_eval_sh_chebyt_l 函数并返回结果
        return _func_eval_sh_chebyt_l(x0, x1)
    else:
        # 如果 Dd_number_t 是 double_complex 类型，返回 NAN（不是数字的数字）
        if Dd_number_t is double_complex:
            return NAN
        else:
            # 如果不满足以上条件，也返回 NAN
            return NAN
# 定义一个函数 eval_sh_chebyu，用于计算 Chebyshev 超几何函数的值
# 参数 x0 是 dl_number_t 类型，参数 x1 是 Dd_number_t 类型
# 函数声明使用了 Cython 的 cpdef 关键字，表明该函数可以作为 C 函数调用，并且可以在不需要全局解释器锁的情况下执行
# 函数声明使用了 noexcept 关键字，表明该函数不会抛出异常
# 函数声明使用了 nogil 关键字，表明该函数在执行时不需要全局解释器锁
cpdef Dd_number_t eval_sh_chebyu(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_sh_chebyu"""
    # 根据 dl_number_t 和 Dd_number_t 的类型分支执行不同的计算函数
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_sh_chebyu[double_complex](x0, x1)
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_sh_chebyu[double](x0, x1)
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_sh_chebyu_l(x0, x1)
    else:
        # 处理无法识别的参数类型组合，返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个函数 eval_sh_jacobi，用于计算 Jacobi 超几何函数的值
# 参数 x0 是 dl_number_t 类型，参数 x1、x2、x3 是 double 类型，参数 x3 是 Dd_number_t 类型
# 函数声明使用了 Cython 的 cpdef 关键字，表明该函数可以作为 C 函数调用，并且可以在不需要全局解释器锁的情况下执行
# 函数声明使用了 noexcept 关键字，表明该函数不会抛出异常
# 函数声明使用了 nogil 关键字，表明该函数在执行时不需要全局解释器锁
cpdef Dd_number_t eval_sh_jacobi(dl_number_t x0, double x1, double x2, Dd_number_t x3) noexcept nogil:
    """See the documentation for scipy.special.eval_sh_jacobi"""
    # 根据 dl_number_t 和 Dd_number_t 的类型分支执行不同的计算函数
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_sh_jacobi[double_complex](x0, x1, x2, x3)
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_sh_jacobi[double](x0, x1, x2, x3)
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_sh_jacobi_l(x0, x1, x2, x3)
    else:
        # 处理无法识别的参数类型组合，返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个函数 eval_sh_legendre，用于计算 Legendre 超几何函数的值
# 参数 x0 是 dl_number_t 类型，参数 x1 是 Dd_number_t 类型
# 函数声明使用了 Cython 的 cpdef 关键字，表明该函数可以作为 C 函数调用，并且可以在不需要全局解释器锁的情况下执行
# 函数声明使用了 noexcept 关键字，表明该函数不会抛出异常
# 函数声明使用了 nogil 关键字，表明该函数在执行时不需要全局解释器锁
cpdef Dd_number_t eval_sh_legendre(dl_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.eval_sh_legendre"""
    # 根据 dl_number_t 和 Dd_number_t 的类型分支执行不同的计算函数
    if dl_number_t is double and Dd_number_t is double_complex:
        return _func_eval_sh_legendre[double_complex](x0, x1)
    elif dl_number_t is double and Dd_number_t is double:
        return _func_eval_sh_legendre[double](x0, x1)
    elif dl_number_t is long and Dd_number_t is double:
        return _func_eval_sh_legendre_l(x0, x1)
    else:
        # 处理无法识别的参数类型组合，返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个函数 exp1，用于计算指数积分函数 exp1 的值
# 参数 x0 是 Dd_number_t 类型
# 函数声明使用了 Cython 的 cpdef 关键字，表明该函数可以作为 C 函数调用，并且可以在不需要全局解释器锁的情况下执行
# 函数声明使用了 noexcept 关键字，表明该函数不会抛出异常
# 函数声明使用了 nogil 关键字，表明该函数在执行时不需要全局解释器锁
cpdef Dd_number_t exp1(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.exp1"""
    # 根据 Dd_number_t 的类型分支执行不同的计算函数
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_cexp1(_complexstuff.npy_cdouble_from_double_complex(x0)))
    elif Dd_number_t is double:
        return special_exp1(x0)
    else:
        # 处理无法识别的参数类型组合，返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个函数 exp10，用于计算以 10 为底的指数函数 exp10 的值
# 参数 x0 是 double 类型
# 函数声明使用了 Cython 的 cpdef 关键字，表明该函数可以作为 C 函数调用，并且可以在不需要全局解释器锁的情况下执行
# 函数声明使用了 noexcept 关键字，表明该函数不会抛出异常
# 函数声明使用了 nogil 关键字，表明该函数在执行时不需要全局解释器锁
cpdef double exp10(double x0) noexcept nogil:
    """See the documentation for scipy.special.exp10"""
    # 调用底层 C 函数计算 exp10
    return cephes_exp10(x0)

# 定义一个函数 exp2，用于计算以 2 为底的指数函数 exp2 的值
# 参数 x0 是 double 类型
# 函数声明使用了 Cython 的 cpdef 关键字，表明该函数可以作为 C 函数调用，并且可以在不需要全局解释器锁的情况下执行
# 函数声明使用了 noexcept 关键字，表明该函数不会抛出异常
# 函数声明使用了 nogil 关键字，表明该函数在执行时不需要全局解释器锁
cpdef double exp2(double x0) noexcept nogil:
    """See the documentation for scipy.special.exp2"""
    # 调用底层 C 函数计算 exp2
    return cephes_exp2(x0)

# 定义一个函数 expi，用于计算复指数函数 expi 的值
# 参数 x0 是 Dd_number_t 类型
# 函数声明使用了 Cython 的 cpdef 关键字，表明该函数可以作为 C 函数调用，并且可以在不需要全局解释器锁的情况下执行
# 函数声明使用了 noexcept 关键字，表明该函数不会抛出异常
# 函数声明使用了 nogil 关键字，表明该函数在执行时不需要全局解释器锁
cpdef Dd_number_t expi(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.expi"""
    # 根据 Dd_number_t 的类型分支执行不同的计算函数
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_cexpi(_complexstuff.npy_cdouble_from_double_complex(x0)))
    elif Dd_number_t is double:
        return special_expi(x0)
    else:
        # 如果条件不满足第一个条件（即 Dd_number_t 不是 float_complex 类型）
        if Dd_number_t is double_complex:
            # 如果 Dd_number_t 是 double_complex 类型，返回 NAN (Not a Number)
            return NAN
        else:
            # 如果 Dd_number_t 不是 double_complex 类型，也返回 NAN
            return NAN
# 定义一个函数 expit，用于计算 sigmoid 函数的值，具体实现参考 scipy.special.expit 的文档
cpdef dfg_number_t expit(dfg_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.expit"""
    # 根据数据类型 dfg_number_t 的不同分支执行不同的计算函数
    if dfg_number_t is double:
        return special_expit(x0)
    elif dfg_number_t is float:
        return special_expitf(x0)
    elif dfg_number_t is long_double:
        return special_expitl(x0)
    else:
        # 对于未知的数据类型，返回 NaN
        if dfg_number_t is double:
            return NAN
        elif dfg_number_t is float:
            return NAN
        else:
            return NAN

# 定义一个函数 expm1，用于计算 exp(x) - 1 的值，具体实现参考 scipy.special.expm1 的文档
cpdef Dd_number_t expm1(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.expm1"""
    # 根据数据类型 Dd_number_t 的不同分支执行不同的计算函数
    if Dd_number_t is double_complex:
        return _func_cexpm1(x0)
    elif Dd_number_t is double:
        return cephes_expm1(x0)
    else:
        # 对于未知的数据类型，返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个函数 expn，用于计算特定的指数积分函数 expn(x0, x1)，具体实现参考 scipy.special.expn 的文档
cpdef double expn(dl_number_t x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.expn"""
    # 根据数据类型 dl_number_t 的不同分支执行不同的计算函数
    if dl_number_t is double:
        return _func_expn_unsafe(x0, x1)
    elif dl_number_t is long:
        return cephes_expn(x0, x1)
    else:
        # 对于未知的数据类型，返回 NaN
        return NAN

# 定义一个函数 exprel，用于计算 exp(x0) - 1 的值，具体实现参考 scipy.special.exprel 的文档
cpdef double exprel(double x0) noexcept nogil:
    """See the documentation for scipy.special.exprel"""
    # 直接调用特定的计算函数 special_exprel
    return special_exprel(x0)

# 定义一个函数 fdtr，用于计算 F 分布的累积分布函数，具体实现参考 scipy.special.fdtr 的文档
cpdef double fdtr(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.fdtr"""
    # 直接调用特定的计算函数 cephes_fdtr
    return cephes_fdtr(x0, x1, x2)

# 定义一个函数 fdtrc，用于计算 F 分布的补充累积分布函数，具体实现参考 scipy.special.fdtrc 的文档
cpdef double fdtrc(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.fdtrc"""
    # 直接调用特定的计算函数 cephes_fdtrc
    return cephes_fdtrc(x0, x1, x2)

# 定义一个函数 fdtri，用于计算 F 分布的反函数，具体实现参考 scipy.special.fdtri 的文档
cpdef double fdtri(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.fdtri"""
    # 直接调用特定的计算函数 cephes_fdtri
    return cephes_fdtri(x0, x1, x2)

# 定义一个函数 fdtridfd，用于计算 F 分布的分母的导数，具体实现参考 scipy.special.fdtridfd 的文档
cpdef double fdtridfd(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.fdtridfd"""
    # 直接调用特定的计算函数 _func_fdtridfd
    return _func_fdtridfd(x0, x1, x2)

# 定义一个函数 fresnel，用于计算 Fresnel 积分，具体实现参考 scipy.special.fresnel 的文档
cdef void fresnel(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1) noexcept nogil:
    """See the documentation for scipy.special.fresnel"""
    cdef npy_cdouble tmp0
    cdef npy_cdouble tmp1
    # 根据数据类型 Dd_number_t 的不同分支执行不同的计算函数，并将结果写入 y0 和 y1
    if Dd_number_t is double:
        _func_cephes_fresnl_wrap(x0, y0, y1)
    elif Dd_number_t is double_complex:
        _func_cfresnl_wrap(_complexstuff.npy_cdouble_from_double_complex(x0), &tmp0, &tmp1)
        y0[0] = _complexstuff.double_complex_from_npy_cdouble(tmp0)
        y1[0] = _complexstuff.double_complex_from_npy_cdouble(tmp1)
    else:
        # 对于未知的数据类型，将 y0 和 y1 设为 NaN
        if Dd_number_t is double_complex:
            y0[0] = NAN
            y1[0] = NAN
        else:
            y0[0] = NAN
            y1[0] = NAN

# 定义一个函数 _fresnel_pywrap，用于包装 fresnel 函数的调用，并返回结果 y0, y1
def _fresnel_pywrap(Dd_number_t x0):
    cdef Dd_number_t y0
    cdef Dd_number_t y1
    # 调用 fresnel 函数计算结果
    fresnel(x0, &y0, &y1)
    return y0, y1

# 定义一个函数 gamma，用于计算 gamma 函数的值，具体实现参考 scipy.special.gamma 的文档
cpdef Dd_number_t gamma(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.gamma"""
    # 直接调用特定的计算函数 scipy.special.gamma
    # 如果 Dd_number_t 的类型是 double_complex
    if Dd_number_t is double_complex:
        # 调用 _complexstuff.double_complex_from_npy_cdouble 函数，传入 special_cgamma 函数处理后的 _complexstuff.npy_cdouble_from_double_complex(x0) 结果作为参数
        return _complexstuff.double_complex_from_npy_cdouble(special_cgamma(_complexstuff.npy_cdouble_from_double_complex(x0)))
    # 如果 Dd_number_t 的类型是 double
    elif Dd_number_t is double:
        # 调用 special_gamma 函数，传入 x0 作为参数
        return special_gamma(x0)
    else:
        # 如果 Dd_number_t 既不是 double_complex 也不是 double，则进入这个分支
        # 返回特殊值 NAN 表示不是一个数字（Not A Number）
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN
# 调用 cephes 库中的 igam 函数来计算不完全 Gamma 函数值，参见 scipy.special.gammainc 的文档
cpdef double gammainc(double x0, double x1) noexcept nogil:
    return cephes_igam(x0, x1)

# 调用 cephes 库中的 igamc 函数来计算补充不完全 Gamma 函数值，参见 scipy.special.gammaincc 的文档
cpdef double gammaincc(double x0, double x1) noexcept nogil:
    return cephes_igamc(x0, x1)

# 调用 cephes 库中的 igamci 函数来计算不完全 Gamma 函数的反函数，参见 scipy.special.gammainccinv 的文档
cpdef double gammainccinv(double x0, double x1) noexcept nogil:
    return cephes_igamci(x0, x1)

# 调用 cephes 库中的 igami 函数来计算完全逆 Gamma 函数，参见 scipy.special.gammaincinv 的文档
cpdef double gammaincinv(double x0, double x1) noexcept nogil:
    return cephes_igami(x0, x1)

# 调用 _func_gammaln_wrap 函数来计算 Gamma 函数的自然对数，参见 scipy.special.gammaln 的文档
cpdef double gammaln(double x0) noexcept nogil:
    return _func_gammaln_wrap(x0)

# 调用 cephes 库中的 gammasgn 函数来计算 Gamma 函数的符号，参见 scipy.special.gammasgn 的文档
cpdef double gammasgn(double x0) noexcept nogil:
    return cephes_gammasgn(x0)

# 调用 cephes 库中的 gdtr 函数来计算 Gamma 分布的累积分布函数，参见 scipy.special.gdtr 的文档
cpdef double gdtr(double x0, double x1, double x2) noexcept nogil:
    return cephes_gdtr(x0, x1, x2)

# 调用 cephes 库中的 gdtrc 函数来计算 Gamma 分布的补充累积分布函数，参见 scipy.special.gdtrc 的文档
cpdef double gdtrc(double x0, double x1, double x2) noexcept nogil:
    return cephes_gdtrc(x0, x1, x2)

# 调用 _func_gdtria 函数来计算 Gamma 分布的逆累积分布函数，参见 scipy.special.gdtria 的文档
cpdef double gdtria(double x0, double x1, double x2) noexcept nogil:
    return _func_gdtria(x0, x1, x2)

# 调用 _func_gdtrib 函数来计算 Gamma 分布的逆补充累积分布函数，参见 scipy.special.gdtrib 的文档
cpdef double gdtrib(double x0, double x1, double x2) noexcept nogil:
    return _func_gdtrib(x0, x1, x2)

# 调用 _func_gdtrix 函数来计算 Gamma 分布的逆累积分布函数的衍生函数，参见 scipy.special.gdtrix 的文档
cpdef double gdtrix(double x0, double x1, double x2) noexcept nogil:
    return _func_gdtrix(x0, x1, x2)

# 调用 special_ccyl_hankel_1 函数来计算第一类 Hankel 函数，参见 scipy.special.hankel1 的文档
cpdef double complex hankel1(double x0, double complex x1) noexcept nogil:
    return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_hankel_1(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))

# 调用 special_ccyl_hankel_1e 函数来计算第一类修改 Hankel 函数，参见 scipy.special.hankel1e 的文档
cpdef double complex hankel1e(double x0, double complex x1) noexcept nogil:
    return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_hankel_1e(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))

# 调用 special_ccyl_hankel_2 函数来计算第二类 Hankel 函数，参见 scipy.special.hankel2 的文档
cpdef double complex hankel2(double x0, double complex x1) noexcept nogil:
    return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_hankel_2(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))

# 调用 special_ccyl_hankel_2e 函数来计算第二类修改 Hankel 函数，参见 scipy.special.hankel2e 的文档
cpdef double complex hankel2e(double x0, double complex x1) noexcept nogil:
    return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_hankel_2e(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))

# 调用 _func_huber 函数来计算 Huber 函数，参见 scipy.special.huber 的文档
cpdef double huber(double x0, double x1) noexcept nogil:
    return _func_huber(x0, x1)
# 定义一个Cython函数，计算特殊函数 hyp0f1
cpdef Dd_number_t hyp0f1(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.hyp0f1"""
    # 如果 Dd_number_t 是 double_complex 类型，则调用对应的复数计算函数
    if Dd_number_t is double_complex:
        return _func__hyp0f1_cmplx(x0, x1)
    # 如果 Dd_number_t 是 double 类型，则调用对应的实数计算函数
    elif Dd_number_t is double:
        return _func__hyp0f1_real(x0, x1)
    else:
        # 否则返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个Cython函数，计算特殊函数 hyp1f1
cpdef Dd_number_t hyp1f1(double x0, double x1, Dd_number_t x2) noexcept nogil:
    """See the documentation for scipy.special.hyp1f1"""
    # 如果 Dd_number_t 是 double 类型，则调用特定的双精度函数
    if Dd_number_t is double:
        return (<double(*)(double, double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_hyp1f1_double)(x0, x1, x2)
    # 如果 Dd_number_t 是 double_complex 类型，则进行复杂计算转换后调用
    elif Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(_func_chyp1f1_wrap(x0, x1, _complexstuff.npy_cdouble_from_double_complex(x2)))
    else:
        # 否则返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个Cython函数，计算特殊函数 hyp2f1
cpdef Dd_number_t hyp2f1(double x0, double x1, double x2, Dd_number_t x3) noexcept nogil:
    """See the documentation for scipy.special.hyp2f1"""
    # 如果 Dd_number_t 是 double 类型，则调用特殊的 hyp2f1 计算函数
    if Dd_number_t is double:
        return special_hyp2f1(x0, x1, x2, x3)
    # 如果 Dd_number_t 是 double_complex 类型，则进行复杂计算转换后调用
    elif Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_chyp2f1(x0, x1, x2, _complexstuff.npy_cdouble_from_double_complex(x3)))
    else:
        # 否则返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个Cython函数，计算特殊函数 hyperu
cpdef double hyperu(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.hyperu"""
    # 调用 _func_hyperu 函数计算 hyperu
    return _func_hyperu(x0, x1, x2)

# 定义一个Cython函数，计算特殊函数 i0
cpdef double i0(double x0) noexcept nogil:
    """See the documentation for scipy.special.i0"""
    # 调用 cephes_i0 函数计算 i0
    return cephes_i0(x0)

# 定义一个Cython函数，计算特殊函数 i0e
cpdef double i0e(double x0) noexcept nogil:
    """See the documentation for scipy.special.i0e"""
    # 调用 cephes_i0e 函数计算 i0e
    return cephes_i0e(x0)

# 定义一个Cython函数，计算特殊函数 i1
cpdef double i1(double x0) noexcept nogil:
    """See the documentation for scipy.special.i1"""
    # 调用 cephes_i1 函数计算 i1
    return cephes_i1(x0)

# 定义一个Cython函数，计算特殊函数 i1e
cpdef double i1e(double x0) noexcept nogil:
    """See the documentation for scipy.special.i1e"""
    # 调用 cephes_i1e 函数计算 i1e
    return cephes_i1e(x0)

# 定义一个Cython函数，计算特殊函数 inv_boxcox
cpdef double inv_boxcox(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.inv_boxcox"""
    # 调用 _func_inv_boxcox 函数计算 inv_boxcox
    return _func_inv_boxcox(x0, x1)

# 定义一个Cython函数，计算特殊函数 inv_boxcox1p
cpdef double inv_boxcox1p(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.inv_boxcox1p"""
    # 调用 _func_inv_boxcox1p 函数计算 inv_boxcox1p
    return _func_inv_boxcox1p(x0, x1)

# 定义一个Cython函数，调用 C 库中的 it2i0k0 函数
cdef void it2i0k0(double x0, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.it2i0k0"""
    _func_it2i0k0_wrap(x0, y0, y1)

# 定义一个 Python 包装函数，调用 it2i0k0 函数并返回结果
def _it2i0k0_pywrap(double x0):
    cdef double y0
    cdef double y1
    it2i0k0(x0, &y0, &y1)
    return y0, y1

# 定义一个Cython函数，调用 C 库中的 it2j0y0 函数
cdef void it2j0y0(double x0, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.it2j0y0"""
    _func_it2j0y0_wrap(x0, y0, y1)

# 定义一个 Python 包装函数，调用 it2j0y0 函数并返回结果
def _it2j0y0_pywrap(double x0):
    cdef double y0
    cdef double y1
    # 调用函数 it2j0y0，并传入参数 x0，将返回值分别存入变量 y0 和 y1
    it2j0y0(x0, &y0, &y1)
    # 返回 y0 和 y1 作为函数的结果
    return y0, y1
# 调用 Cython 版本的特殊函数 it2struve0，返回 struve 函数的值
cpdef double it2struve0(double x0) noexcept nogil:
    """See the documentation for scipy.special.it2struve0"""
    return special_it2struve0(x0)

# 调用 Cython 版本的特殊函数 itairy，计算 Airy 函数的值
cdef void itairy(double x0, double *y0, double *y1, double *y2, double *y3) noexcept nogil:
    """See the documentation for scipy.special.itairy"""
    special_itairy(x0, y0, y1, y2, y3)

# Python 包装函数，调用 itairy 函数计算 Airy 函数，并返回结果
def _itairy_pywrap(double x0):
    cdef double y0
    cdef double y1
    cdef double y2
    cdef double y3
    itairy(x0, &y0, &y1, &y2, &y3)
    return y0, y1, y2, y3

# 调用 Cython 版本的特殊函数 iti0k0，计算第一类修正 Bessel 函数的值
cdef void iti0k0(double x0, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.iti0k0"""
    _func_it1i0k0_wrap(x0, y0, y1)

# Python 包装函数，调用 iti0k0 函数计算第一类修正 Bessel 函数，并返回结果
def _iti0k0_pywrap(double x0):
    cdef double y0
    cdef double y1
    iti0k0(x0, &y0, &y1)
    return y0, y1

# 调用 Cython 版本的特殊函数 itj0y0，计算第一类和第二类 Bessel 函数的值
cdef void itj0y0(double x0, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.itj0y0"""
    _func_it1j0y0_wrap(x0, y0, y1)

# Python 包装函数，调用 itj0y0 函数计算第一类和第二类 Bessel 函数，并返回结果
def _itj0y0_pywrap(double x0):
    cdef double y0
    cdef double y1
    itj0y0(x0, &y0, &y1)
    return y0, y1

# 调用 Cython 版本的特殊函数 itmodstruve0，返回修正 Struve 函数的值
cpdef double itmodstruve0(double x0) noexcept nogil:
    """See the documentation for scipy.special.itmodstruve0"""
    return special_itmodstruve0(x0)

# 调用 Cython 版本的特殊函数 itstruve0，返回 Struve 函数的值
cpdef double itstruve0(double x0) noexcept nogil:
    """See the documentation for scipy.special.itstruve0"""
    return special_itstruve0(x0)

# 调用 Cython 版本的特殊函数 iv，计算修正 Bessel 函数的值
cpdef Dd_number_t iv(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.iv"""
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_bessel_i(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))
    elif Dd_number_t is double:
        return special_cyl_bessel_i(x0, x1)
    else:
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 调用 Cython 版本的特殊函数 ive，计算修正 Bessel 函数的指数型
cpdef Dd_number_t ive(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.ive"""
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_bessel_ie(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))
    elif Dd_number_t is double:
        return special_cyl_bessel_ie(x0, x1)
    else:
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 调用 Cython 版本的特殊函数 j0，计算第一类 Bessel 函数的值
cpdef double j0(double x0) noexcept nogil:
    """See the documentation for scipy.special.j0"""
    return cephes_j0(x0)

# 调用 Cython 版本的特殊函数 j1，计算第一类 Bessel 函数的值
cpdef double j1(double x0) noexcept nogil:
    """See the documentation for scipy.special.j1"""
    return cephes_j1(x0)

# 调用 Cython 版本的特殊函数 jv，计算第一类 Bessel 函数的值
cpdef Dd_number_t jv(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.jv"""
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_bessel_j(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))
    elif Dd_number_t is double:
        return special_cyl_bessel_j(x0, x1)
    # 如果上述条件不满足，则执行以下代码块
    else:
        # 如果 Dd_number_t 的类型是 double_complex
        if Dd_number_t is double_complex:
            # 返回浮点数表示的非数值（Not a Number）
            return NAN
        else:
            # 如果 Dd_number_t 的类型不是 double_complex，则返回浮点数表示的非数值（Not a Number）
            return NAN
# 使用 Cython 声明一个函数 jve，用于计算特定类型数值和 double 的 Bessel 函数 J
cpdef Dd_number_t jve(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.jve"""
    # 如果 Dd_number_t 是 double_complex 类型，则调用特定的复数 Bessel 函数计算方法
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_bessel_je(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))
    # 如果 Dd_number_t 是 double 类型，则直接调用普通的 Bessel 函数计算方法
    elif Dd_number_t is double:
        return special_cyl_bessel_je(x0, x1)
    else:
        # 如果 Dd_number_t 不是上述两种类型，则返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 使用 Cython 声明一个函数 k0，用于计算第一类修改 Bessel 函数 K
cpdef double k0(double x0) noexcept nogil:
    """See the documentation for scipy.special.k0"""
    # 调用 C 库中的函数计算第一类修改 Bessel 函数 K
    return cephes_k0(x0)

# 使用 Cython 声明一个函数 k0e，用于计算修改 Bessel 函数 K 的指数形式
cpdef double k0e(double x0) noexcept nogil:
    """See the documentation for scipy.special.k0e"""
    # 调用 C 库中的函数计算修改 Bessel 函数 K 的指数形式
    return cephes_k0e(x0)

# 使用 Cython 声明一个函数 k1，用于计算第二类修改 Bessel 函数 K
cpdef double k1(double x0) noexcept nogil:
    """See the documentation for scipy.special.k1"""
    # 调用 C 库中的函数计算第二类修改 Bessel 函数 K
    return cephes_k1(x0)

# 使用 Cython 声明一个函数 k1e，用于计算第二类修改 Bessel 函数 K 的指数形式
cpdef double k1e(double x0) noexcept nogil:
    """See the documentation for scipy.special.k1e"""
    # 调用 C 库中的函数计算第二类修改 Bessel 函数 K 的指数形式
    return cephes_k1e(x0)

# 使用 Cython 声明一个函数 kei，用于计算指数整数 Bessel 函数 I
cpdef double kei(double x0) noexcept nogil:
    """See the documentation for scipy.special.kei"""
    # 调用 C 库中的函数计算指数整数 Bessel 函数 I
    return special_kei(x0)

# 使用 Cython 声明一个函数 keip，用于计算指数整数 Bessel 函数 I 的导数
cpdef double keip(double x0) noexcept nogil:
    """See the documentation for scipy.special.keip"""
    # 调用 C 库中的函数计算指数整数 Bessel 函数 I 的导数
    return special_keip(x0)

# 使用 Cython 声明一个函数 kelvin，用于计算开尔文函数的四个复数解
cdef void kelvin(double x0, double complex *y0, double complex *y1, double complex *y2, double complex *y3) noexcept nogil:
    """See the documentation for scipy.special.kelvin"""
    # 调用 C 库中的函数计算开尔文函数，并将结果存储在指定的复数指针中
    cdef npy_cdouble tmp0
    cdef npy_cdouble tmp1
    cdef npy_cdouble tmp2
    cdef npy_cdouble tmp3
    special_ckelvin(x0, &tmp0, &tmp1, &tmp2, &tmp3)
    # 将复数结果转换为 Cython 的 double_complex 类型并存储在指定变量中
    y0[0] = _complexstuff.double_complex_from_npy_cdouble(tmp0)
    y1[0] = _complexstuff.double_complex_from_npy_cdouble(tmp1)
    y2[0] = _complexstuff.double_complex_from_npy_cdouble(tmp2)
    y3[0] = _complexstuff.double_complex_from_npy_cdouble(tmp3)

# 使用 Cython 声明一个函数 _kelvin_pywrap，用于包装开尔文函数的调用
def _kelvin_pywrap(double x0):
    # 声明需要存储复数解的变量
    cdef double complex y0
    cdef double complex y1
    cdef double complex y2
    cdef double complex y3
    # 调用开尔文函数计算四个复数解，并将结果存储在指定变量中
    kelvin(x0, &y0, &y1, &y2, &y3)
    # 返回四个复数解的元组
    return y0, y1, y2, y3

# 使用 Cython 声明一个函数 ker，用于计算第一类修改 Bessel 函数 K 的缩放形式
cpdef double ker(double x0) noexcept nogil:
    """See the documentation for scipy.special.ker"""
    # 调用 C 库中的函数计算第一类修改 Bessel 函数 K 的缩放形式
    return special_ker(x0)

# 使用 Cython 声明一个函数 kerp，用于计算第二类修改 Bessel 函数 K 的缩放形式
cpdef double kerp(double x0) noexcept nogil:
    """See the documentation for scipy.special.kerp"""
    # 调用 C 库中的函数计算第二类修改 Bessel 函数 K 的缩放形式
    return special_kerp(x0)

# 使用 Cython 声明一个函数 kl_div，用于计算 Kullback-Leibler 散度
cpdef double kl_div(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.kl_div"""
    # 调用 C 库中的函数计算 Kullback-Leibler 散度
    return _func_kl_div(x0, x1)

# 使用 Cython 声明一个函数 kn，用于计算第 n 阶循环 Bessel 函数
cpdef double kn(dl_number_t x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.kn"""
    # 根据 dl_number_t 类型的不同，调用相应的循环 Bessel 函数计算方法
    if dl_number_t is double:
        return _func_kn_unsafe(x0, x1)
    elif dl_number_t is long:
        return special_cyl_bessel_k_int(x0, x1)
    else:
        return NAN

# 使用 Cython 声明一个函数 kolmogi，用于计算 Kolmogorov 函数的反函数
cpdef double kolmogi(double x0) noexcept nogil:
    """See the documentation for scipy.special.kolmogi"""
    # 调用 C 库中的函数计算 Kolmogorov 函数的反函数
    return cephes_kolmogi(x0)

# 使用 Cython 声明一个函数 kolmogorov，用于计算 Kolmogorov 分布函数
cpdef double kolmogorov(double x0) noexcept nogil:
    """See the documentation for scipy.special.kolmogorov"""
    # 调用 cephes_kolmogorov 函数，并返回其结果
    return cephes_kolmogorov(x0)
# 根据给定的参数和类型，计算修正的贝塞尔函数值
cpdef Dd_number_t kv(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.kv"""
    # 如果输入类型是双精度复数，则调用特定的复数贝塞尔函数计算
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_bessel_k(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))
    # 如果输入类型是双精度数，则调用特定的贝塞尔函数计算
    elif Dd_number_t is double:
        return special_cyl_bessel_k(x0, x1)
    else:
        # 其他情况返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 根据给定的参数和类型，计算修正的贝塞尔函数的另一种形式的值
cpdef Dd_number_t kve(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.kve"""
    # 如果输入类型是双精度复数，则调用特定的复数贝塞尔函数计算
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_bessel_ke(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))
    # 如果输入类型是双精度数，则调用特定的贝塞尔函数计算
    elif Dd_number_t is double:
        return special_cyl_bessel_ke(x0, x1)
    else:
        # 其他情况返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 根据给定的参数和类型，计算修正的对数函数的值
cpdef Dd_number_t log1p(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.log1p"""
    # 如果输入类型是双精度复数，则调用特定的复数对数函数计算
    if Dd_number_t is double_complex:
        return _func_clog1p(x0)
    # 如果输入类型是双精度数，则调用特定的对数函数计算
    elif Dd_number_t is double:
        return cephes_log1p(x0)
    else:
        # 其他情况返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 根据给定的参数类型，计算逻辑 sigmoid 函数的对数
cpdef dfg_number_t log_expit(dfg_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.log_expit"""
    # 根据输入类型选择适当的特定 log_expit 函数进行计算
    if dfg_number_t is double:
        return special_log_expit(x0)
    elif dfg_number_t is float:
        return special_log_expitf(x0)
    elif dfg_number_t is long_double:
        return special_log_expitl(x0)
    else:
        # 处理不支持的类型，返回 NAN
        if dfg_number_t is double:
            return NAN
        elif dfg_number_t is float:
            return NAN
        else:
            return NAN

# 根据给定的参数和类型，计算修正的正态分布的对数累积分布函数值
cpdef Dd_number_t log_ndtr(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.log_ndtr"""
    # 如果输入类型是双精度数，则调用特定的 double 类型 log_ndtr 函数计算
    if Dd_number_t is double:
        return (<double(*)(double) noexcept nogil>scipy.special._ufuncs_cxx._export_faddeeva_log_ndtr)(x0)
    # 如果输入类型是双精度复数，则调用特定的 double complex 类型 log_ndtr 函数计算
    elif Dd_number_t is double_complex:
        return (<double complex(*)(double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_faddeeva_log_ndtr_complex)(x0)
    else:
        # 其他情况返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 根据给定的参数和类型，计算 gamma 函数的对数
cpdef Dd_number_t loggamma(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.loggamma"""
    # 如果输入类型是双精度数，则调用特定的 loggamma 函数计算
    if Dd_number_t is double:
        return special_loggamma(x0)
    # 如果输入类型是双精度复数，则调用特定的复数 loggamma 函数计算
    elif Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_cloggamma(_complexstuff.npy_cdouble_from_double_complex(x0)))
    else:
        # 其他情况返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 根据给定的参数类型，计算逻辑 sigmoid 函数的对数
cpdef dfg_number_t logit(dfg_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.logit"""
    # 检查变量 dfq_number_t 是否为双精度浮点数类型（应为 'double'）
    if dfg_number_t is double:
        # 如果是双精度浮点数类型，则调用 special_logit 函数处理 x0 并返回结果
        return special_logit(x0)
    # 检查变量 dfq_number_t 是否为单精度浮点数类型（应为 'float'）
    elif dfg_number_t is float:
        # 如果是单精度浮点数类型，则调用 special_logitf 函数处理 x0 并返回结果
        return special_logitf(x0)
    # 检查变量 dfq_number_t 是否为长双精度浮点数类型（应为 'long_double'）
    elif dfg_number_t is long_double:
        # 如果是长双精度浮点数类型，则调用 special_logitl 函数处理 x0 并返回结果
        return special_logitl(x0)
    else:
        # 如果 dfq_number_t 不是已知的浮点数类型，返回 NAN（Not A Number）
        if dfg_number_t is double:
            # 如果是双精度浮点数类型，则返回 NAN
            return NAN
        elif dfg_number_t is float:
            # 如果是单精度浮点数类型，则返回 NAN
            return NAN
        else:
            # 如果既不是双精度也不是单精度浮点数类型，则返回 NAN
            return NAN
# 调用 C 语言编写的函数 _func_pmv_wrap，将参数 x0, x1, x2 传递给它并返回结果
cpdef double lpmv(double x0, double x1, double x2) noexcept nogil:
    return _func_pmv_wrap(x0, x1, x2)

# 调用 C 语言编写的函数 _func_cem_cva_wrap，将参数 x0, x1 传递给它并返回结果
cpdef double mathieu_a(double x0, double x1) noexcept nogil:
    return _func_cem_cva_wrap(x0, x1)

# 调用 C 语言编写的函数 _func_sem_cva_wrap，将参数 x0, x1 传递给它并返回结果
cpdef double mathieu_b(double x0, double x1) noexcept nogil:
    return _func_sem_cva_wrap(x0, x1)

# 调用 C 语言编写的函数 _func_cem_wrap，将参数 x0, x1, x2 传递给它，并通过指针 y0, y1 返回结果
cdef void mathieu_cem(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil:
    _func_cem_wrap(x0, x1, x2, y0, y1)

# Python 函数，调用 mathieu_cem 函数获得 y0, y1 的值，并作为元组返回
def _mathieu_cem_pywrap(double x0, double x1, double x2):
    cdef double y0
    cdef double y1
    mathieu_cem(x0, x1, x2, &y0, &y1)
    return y0, y1

# 调用 C 语言编写的函数 _func_mcm1_wrap，将参数 x0, x1, x2 传递给它，并通过指针 y0, y1 返回结果
cdef void mathieu_modcem1(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil:
    _func_mcm1_wrap(x0, x1, x2, y0, y1)

# Python 函数，调用 mathieu_modcem1 函数获得 y0, y1 的值，并作为元组返回
def _mathieu_modcem1_pywrap(double x0, double x1, double x2):
    cdef double y0
    cdef double y1
    mathieu_modcem1(x0, x1, x2, &y0, &y1)
    return y0, y1

# 调用 C 语言编写的函数 _func_mcm2_wrap，将参数 x0, x1, x2 传递给它，并通过指针 y0, y1 返回结果
cdef void mathieu_modcem2(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil:
    _func_mcm2_wrap(x0, x1, x2, y0, y1)

# Python 函数，调用 mathieu_modcem2 函数获得 y0, y1 的值，并作为元组返回
def _mathieu_modcem2_pywrap(double x0, double x1, double x2):
    cdef double y0
    cdef double y1
    mathieu_modcem2(x0, x1, x2, &y0, &y1)
    return y0, y1

# 调用 C 语言编写的函数 _func_msm1_wrap，将参数 x0, x1, x2 传递给它，并通过指针 y0, y1 返回结果
cdef void mathieu_modsem1(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil:
    _func_msm1_wrap(x0, x1, x2, y0, y1)

# Python 函数，调用 mathieu_modsem1 函数获得 y0, y1 的值，并作为元组返回
def _mathieu_modsem1_pywrap(double x0, double x1, double x2):
    cdef double y0
    cdef double y1
    mathieu_modsem1(x0, x1, x2, &y0, &y1)
    return y0, y1

# 调用 C 语言编写的函数 _func_msm2_wrap，将参数 x0, x1, x2 传递给它，并通过指针 y0, y1 返回结果
cdef void mathieu_modsem2(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil:
    _func_msm2_wrap(x0, x1, x2, y0, y1)

# Python 函数，调用 mathieu_modsem2 函数获得 y0, y1 的值，并作为元组返回
def _mathieu_modsem2_pywrap(double x0, double x1, double x2):
    cdef double y0
    cdef double y1
    mathieu_modsem2(x0, x1, x2, &y0, &y1)
    return y0, y1

# 调用 C 语言编写的函数 _func_sem_wrap，将参数 x0, x1, x2 传递给它，并通过指针 y0, y1 返回结果
cdef void mathieu_sem(double x0, double x1, double x2, double *y0, double *y1) noexcept nogil:
    _func_sem_wrap(x0, x1, x2, y0, y1)

# Python 函数，调用 mathieu_sem 函数获得 y0, y1 的值，并作为元组返回
def _mathieu_sem_pywrap(double x0, double x1, double x2):
    cdef double y0
    cdef double y1
    mathieu_sem(x0, x1, x2, &y0, &y1)
    return y0, y1

# 调用 C 语言编写的函数 _func_modified_fresnel_minus_wrap，将参数 x0 传递给它，并通过指针 y0, y1 返回结果
cdef void modfresnelm(double x0, double complex *y0, double complex *y1) noexcept nogil:
    cdef npy_cdouble tmp0
    cdef npy_cdouble tmp1
    _func_modified_fresnel_minus_wrap(x0, &tmp0, &tmp1)
    y0[0] = _complexstuff.double_complex_from_npy_cdouble(tmp0)
    # 将 tmp1 转换为 NumPy 的复数类型 cdouble，并将结果赋值给 y1 的第一个元素
    y1[0] = _complexstuff.double_complex_from_npy_cdouble(tmp1)
def _modfresnelm_pywrap(double x0):
    # 定义两个复数变量
    cdef double complex y0
    cdef double complex y1
    # 调用 modfresnelm 函数，将结果存入 y0 和 y1 中
    modfresnelm(x0, &y0, &y1)
    # 返回结果 y0 和 y1
    return y0, y1

cdef void modfresnelp(double x0, double complex *y0, double complex *y1) noexcept nogil:
    """See the documentation for scipy.special.modfresnelp"""
    # 定义两个复数变量 tmp0 和 tmp1
    cdef npy_cdouble tmp0
    cdef npy_cdouble tmp1
    # 调用 _func_modified_fresnel_plus_wrap 函数，将结果存入 tmp0 和 tmp1 中
    _func_modified_fresnel_plus_wrap(x0, &tmp0, &tmp1)
    # 将 tmp0 和 tmp1 转换为 double complex 类型后存入 y0[0] 和 y1[0] 中
    y0[0] = _complexstuff.double_complex_from_npy_cdouble(tmp0)
    y1[0] = _complexstuff.double_complex_from_npy_cdouble(tmp1)

def _modfresnelp_pywrap(double x0):
    # 定义两个复数变量
    cdef double complex y0
    cdef double complex y1
    # 调用 modfresnelp 函数，将结果存入 y0 和 y1 中
    modfresnelp(x0, &y0, &y1)
    # 返回结果 y0 和 y1
    return y0, y1

cpdef double modstruve(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.modstruve"""
    # 调用 cephes_struve_l 函数，返回结果
    return cephes_struve_l(x0, x1)

cpdef double nbdtr(dl_number_t x0, dl_number_t x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.nbdtr"""
    # 根据 dl_number_t 的类型选择不同的函数进行计算，并返回结果
    if dl_number_t is double:
        return _func_nbdtr_unsafe(x0, x1, x2)
    elif dl_number_t is long:
        return cephes_nbdtr(x0, x1, x2)
    else:
        return NAN

cpdef double nbdtrc(dl_number_t x0, dl_number_t x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.nbdtrc"""
    # 根据 dl_number_t 的类型选择不同的函数进行计算，并返回结果
    if dl_number_t is double:
        return _func_nbdtrc_unsafe(x0, x1, x2)
    elif dl_number_t is long:
        return cephes_nbdtrc(x0, x1, x2)
    else:
        return NAN

cpdef double nbdtri(dl_number_t x0, dl_number_t x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.nbdtri"""
    # 根据 dl_number_t 的类型选择不同的函数进行计算，并返回结果
    if dl_number_t is double:
        return _func_nbdtri_unsafe(x0, x1, x2)
    elif dl_number_t is long:
        return cephes_nbdtri(x0, x1, x2)
    else:
        return NAN

cpdef double nbdtrik(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.nbdtrik"""
    # 调用 _func_nbdtrik 函数，返回结果
    return _func_nbdtrik(x0, x1, x2)

cpdef double nbdtrin(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.nbdtrin"""
    # 调用 _func_nbdtrin 函数，返回结果
    return _func_nbdtrin(x0, x1, x2)

cpdef double ncfdtr(double x0, double x1, double x2, double x3) noexcept nogil:
    """See the documentation for scipy.special.ncfdtr"""
    # 调用 _func_ncfdtr 函数，返回结果
    return _func_ncfdtr(x0, x1, x2, x3)

cpdef double ncfdtri(double x0, double x1, double x2, double x3) noexcept nogil:
    """See the documentation for scipy.special.ncfdtri"""
    # 调用 _func_ncfdtri 函数，返回结果
    return _func_ncfdtri(x0, x1, x2, x3)

cpdef double ncfdtridfd(double x0, double x1, double x2, double x3) noexcept nogil:
    """See the documentation for scipy.special.ncfdtridfd"""
    # 调用 _func_ncfdtridfd 函数，返回结果
    return _func_ncfdtridfd(x0, x1, x2, x3)

cpdef double ncfdtridfn(double x0, double x1, double x2, double x3) noexcept nogil:
    """See the documentation for scipy.special.ncfdtridfn"""
    # 调用 _func_ncfdtridfn 函数，返回结果
    return _func_ncfdtridfn(x0, x1, x2, x3)

cpdef double ncfdtrinc(double x0, double x1, double x2, double x3) noexcept nogil:
    """See the documentation for scipy.special.ncfdtrinc"""
    # TODO: 补充此处的函数说明
    # 调用 _func_ncfdtrinc 函数，返回结果
    return _func_ncfdtrinc(x0, x1, x2, x3)
    # 调用函数 _func_ncfdtrinc，并返回其结果
    return _func_ncfdtrinc(x0, x1, x2, x3)
cpdef double nctdtr(double x0, double x1, double x2) noexcept nogil:
    """调用 C++ 实现的函数 _func_nctdtr，计算非中心 t 分布累积分布函数的值"""
    return _func_nctdtr(x0, x1, x2)

cpdef double nctdtridf(double x0, double x1, double x2) noexcept nogil:
    """调用 C++ 实现的函数 _func_nctdtridf，计算非中心 t 分布的逆累积分布函数的值"""
    return _func_nctdtridf(x0, x1, x2)

cpdef double nctdtrinc(double x0, double x1, double x2) noexcept nogil:
    """调用 C++ 实现的函数 _func_nctdtrinc，计算非中心 t 分布的增量函数的值"""
    return _func_nctdtrinc(x0, x1, x2)

cpdef double nctdtrit(double x0, double x1, double x2) noexcept nogil:
    """调用 C++ 实现的函数 _func_nctdtrit，计算非中心 t 分布的积分函数的值"""
    return _func_nctdtrit(x0, x1, x2)

cpdef Dd_number_t ndtr(Dd_number_t x0) noexcept nogil:
    """根据输入的数据类型选择相应的正态分布函数计算"""
    if Dd_number_t is double_complex:
        return (<double complex(*)(double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_faddeeva_ndtr)(x0)
    elif Dd_number_t is double:
        return cephes_ndtr(x0)
    else:
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

cpdef double ndtri(double x0) noexcept nogil:
    """调用 C 实现的 cephes 库函数，计算正态分布的逆累积分布函数的值"""
    return cephes_ndtri(x0)

cpdef double nrdtrimn(double x0, double x1, double x2) noexcept nogil:
    """调用 C++ 实现的函数 _func_nrdtrimn，计算正态分布的尾部修剪的平均值函数的值"""
    return _func_nrdtrimn(x0, x1, x2)

cpdef double nrdtrisd(double x0, double x1, double x2) noexcept nogil:
    """调用 C++ 实现的函数 _func_nrdtrisd，计算正态分布的尾部修剪的标准差函数的值"""
    return _func_nrdtrisd(x0, x1, x2)

cdef void obl_ang1(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil:
    """调用 C++ 实现的函数 _func_oblate_aswfa_nocv_wrap，计算椭圆角函数的第一类"""
    y0[0] = _func_oblate_aswfa_nocv_wrap(x0, x1, x2, x3, y1)

def _obl_ang1_pywrap(double x0, double x1, double x2, double x3):
    """Python 封装函数，调用 obl_ang1 函数并返回结果"""
    cdef double y0
    cdef double y1
    obl_ang1(x0, x1, x2, x3, &y0, &y1)
    return y0, y1

cdef void obl_ang1_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil:
    """调用 C++ 实现的函数 _func_oblate_aswfa_wrap，计算带有圆顶的椭圆角函数的第一类"""
    _func_oblate_aswfa_wrap(x0, x1, x2, x3, x4, y0, y1)

def _obl_ang1_cv_pywrap(double x0, double x1, double x2, double x3, double x4):
    """Python 封装函数，调用 obl_ang1_cv 函数并返回结果"""
    cdef double y0
    cdef double y1
    obl_ang1_cv(x0, x1, x2, x3, x4, &y0, &y1)
    return y0, y1

cpdef double obl_cv(double x0, double x1, double x2) noexcept nogil:
    """调用 C++ 实现的函数 _func_oblate_segv_wrap，计算椭圆形的体积"""
    return _func_oblate_segv_wrap(x0, x1, x2)

cdef void obl_rad1(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil:
    """调用 C++ 实现的函数 _func_oblate_radial1_nocv_wrap，计算椭圆形的径向函数的第一类"""
    y0[0] = _func_oblate_radial1_nocv_wrap(x0, x1, x2, x3, y1)

def _obl_rad1_pywrap(double x0, double x1, double x2, double x3):
    """Python 封装函数，调用 obl_rad1 函数并返回结果"""
    cdef double y0
    cdef double y1
    obl_rad1(x0, x1, x2, x3, &y0, &y1)
    return y0, y1
# 定义一个 C 语言风格的函数 obl_rad1_cv，接受五个 double 类型参数，无异常处理，不使用全局解释器锁（GIL）
cdef void obl_rad1_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.obl_rad1_cv"""
    # 调用内部函数 _func_oblate_radial1_wrap 处理传入的参数，计算结果并存储在 y0 和 y1 中
    _func_oblate_radial1_wrap(x0, x1, x2, x3, x4, y0, y1)

# 定义一个 Python 包装函数 _obl_rad1_cv_pywrap，接受五个 double 类型参数，调用 obl_rad1_cv 并返回结果
def _obl_rad1_cv_pywrap(double x0, double x1, double x2, double x3, double x4):
    cdef double y0
    cdef double y1
    # 调用 obl_rad1_cv 函数计算结果，并将结果存储在 y0 和 y1 中
    obl_rad1_cv(x0, x1, x2, x3, x4, &y0, &y1)
    # 返回计算结果 y0 和 y1
    return y0, y1

# 定义一个 C 语言风格的函数 obl_rad2，接受四个 double 类型参数和两个 double 类型指针，无异常处理，不使用全局解释器锁（GIL）
cdef void obl_rad2(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.obl_rad2"""
    # 调用内部函数 _func_oblate_radial2_nocv_wrap 处理传入的参数，计算结果并存储在 y1 中，y0 作为返回值
    y0[0] = _func_oblate_radial2_nocv_wrap(x0, x1, x2, x3, y1)

# 定义一个 Python 包装函数 _obl_rad2_pywrap，接受四个 double 类型参数，调用 obl_rad2 并返回结果
def _obl_rad2_pywrap(double x0, double x1, double x2, double x3):
    cdef double y0
    cdef double y1
    # 调用 obl_rad2 函数计算结果，并将结果存储在 y0 和 y1 中
    obl_rad2(x0, x1, x2, x3, &y0, &y1)
    # 返回计算结果 y0 和 y1
    return y0, y1

# 定义一个 C 语言风格的函数 obl_rad2_cv，接受五个 double 类型参数和两个 double 类型指针，无异常处理，不使用全局解释器锁（GIL）
cdef void obl_rad2_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.obl_rad2_cv"""
    # 调用内部函数 _func_oblate_radial2_wrap 处理传入的参数，计算结果并存储在 y0 和 y1 中
    _func_oblate_radial2_wrap(x0, x1, x2, x3, x4, y0, y1)

# 定义一个 Python 包装函数 _obl_rad2_cv_pywrap，接受五个 double 类型参数，调用 obl_rad2_cv 并返回结果
def _obl_rad2_cv_pywrap(double x0, double x1, double x2, double x3, double x4):
    cdef double y0
    cdef double y1
    # 调用 obl_rad2_cv 函数计算结果，并将结果存储在 y0 和 y1 中
    obl_rad2_cv(x0, x1, x2, x3, x4, &y0, &y1)
    # 返回计算结果 y0 和 y1
    return y0, y1

# 定义一个 Cython 包装函数 owens_t，接受两个 double 类型参数，无异常处理，不使用全局解释器锁（GIL）
cpdef double owens_t(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.owens_t"""
    # 调用 C 库函数 cephes_owens_t 计算 Owens T 函数的值并返回
    return cephes_owens_t(x0, x1)

# 定义一个 C 语言风格的函数 pbdv，接受两个 double 类型参数和两个 double 类型指针，无异常处理，不使用全局解释器锁（GIL）
cdef void pbdv(double x0, double x1, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pbdv"""
    # 调用内部函数 _func_pbdv_wrap 处理传入的参数，计算结果并存储在 y0 和 y1 中
    _func_pbdv_wrap(x0, x1, y0, y1)

# 定义一个 Python 包装函数 _pbdv_pywrap，接受两个 double 类型参数，调用 pbdv 并返回结果
def _pbdv_pywrap(double x0, double x1):
    cdef double y0
    cdef double y1
    # 调用 pbdv 函数计算结果，并将结果存储在 y0 和 y1 中
    pbdv(x0, x1, &y0, &y1)
    # 返回计算结果 y0 和 y1
    return y0, y1

# 定义一个 C 语言风格的函数 pbvv，接受两个 double 类型参数和两个 double 类型指针，无异常处理，不使用全局解释器锁（GIL）
cdef void pbvv(double x0, double x1, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pbvv"""
    # 调用内部函数 _func_pbvv_wrap 处理传入的参数，计算结果并存储在 y0 和 y1 中
    _func_pbvv_wrap(x0, x1, y0, y1)

# 定义一个 Python 包装函数 _pbvv_pywrap，接受两个 double 类型参数，调用 pbvv 并返回结果
def _pbvv_pywrap(double x0, double x1):
    cdef double y0
    cdef double y1
    # 调用 pbvv 函数计算结果，并将结果存储在 y0 和 y1 中
    pbvv(x0, x1, &y0, &y1)
    # 返回计算结果 y0 和 y1
    return y0, y1

# 定义一个 C 语言风格的函数 pbwa，接受两个 double 类型参数和两个 double 类型指针，无异常处理，不使用全局解释器锁（GIL）
cdef void pbwa(double x0, double x1, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pbwa"""
    # 调用内部函数 _func_pbwa_wrap 处理传入的参数，计算结果并存储在 y0 和 y1 中
    _func_pbwa_wrap(x0, x1, y0, y1)

# 定义一个 Python 包装函数 _pbwa_pywrap，接受两个 double 类型参数，调用 pbwa 并返回结果
def _pbwa_pywrap(double x0, double x1):
    cdef double y0
    cdef double y1
    # 调用 pbwa 函数计算结果，并将结果存储在 y0 和 y1 中
    pbwa(x0, x1, &y0, &y1)
    # 返回计算结果 y0 和 y1
    return y0, y1

# 定义一个 Cython 包装函数 pdtr，接受两个 double 类型参数，无异常处理，不使用全局解释器锁（GIL）
cpdef double pdtr(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.pdtr"""
    # 调用 C 库函数 cephes_pdtr 计算累积分布函数值并返回
    return cephes_pdtr
# 定义一个 Cython 函数，用于计算泊松函数的值，参见 scipy.special.poch 的文档
cpdef double poch(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.poch"""
    return cephes_poch(x0, x1)

# 定义一个 Cython 函数，用于计算 x0^x1 - 1，根据 df_number_t 类型选择对应的函数进行计算
cpdef df_number_t powm1(df_number_t x0, df_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.powm1"""
    if df_number_t is float:
        return (<float(*)(float, float) noexcept nogil>scipy.special._ufuncs_cxx._export_powm1_float)(x0, x1)
    elif df_number_t is double:
        return (<double(*)(double, double) noexcept nogil>scipy.special._ufuncs_cxx._export_powm1_double)(x0, x1)
    else:
        # 如果 df_number_t 不是 float 或 double 类型，则返回 NaN
        if df_number_t is double:
            return NAN
        else:
            return NAN

# 定义一个 Cython 函数，用于计算 pro_ang1 的结果并通过指针返回两个值
cdef void pro_ang1(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pro_ang1"""
    y0[0] = _func_prolate_aswfa_nocv_wrap(x0, x1, x2, x3, y1)

# 定义一个 Python 包装函数，调用 pro_ang1 函数并返回其结果
def _pro_ang1_pywrap(double x0, double x1, double x2, double x3):
    cdef double y0
    cdef double y1
    pro_ang1(x0, x1, x2, x3, &y0, &y1)
    return y0, y1

# 定义一个 Cython 函数，用于计算 pro_ang1_cv 的结果并通过指针返回两个值
cdef void pro_ang1_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pro_ang1_cv"""
    _func_prolate_aswfa_wrap(x0, x1, x2, x3, x4, y0, y1)

# 定义一个 Python 包装函数，调用 pro_ang1_cv 函数并返回其结果
def _pro_ang1_cv_pywrap(double x0, double x1, double x2, double x3, double x4):
    cdef double y0
    cdef double y1
    pro_ang1_cv(x0, x1, x2, x3, x4, &y0, &y1)
    return y0, y1

# 定义一个 Cython 函数，用于计算 pro_cv 的结果
cpdef double pro_cv(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.pro_cv"""
    return _func_prolate_segv_wrap(x0, x1, x2)

# 定义一个 Cython 函数，用于计算 pro_rad1 的结果并通过指针返回两个值
cdef void pro_rad1(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pro_rad1"""
    y0[0] = _func_prolate_radial1_nocv_wrap(x0, x1, x2, x3, y1)

# 定义一个 Python 包装函数，调用 pro_rad1 函数并返回其结果
def _pro_rad1_pywrap(double x0, double x1, double x2, double x3):
    cdef double y0
    cdef double y1
    pro_rad1(x0, x1, x2, x3, &y0, &y1)
    return y0, y1

# 定义一个 Cython 函数，用于计算 pro_rad1_cv 的结果并通过指针返回两个值
cdef void pro_rad1_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pro_rad1_cv"""
    _func_prolate_radial1_wrap(x0, x1, x2, x3, x4, y0, y1)

# 定义一个 Python 包装函数，调用 pro_rad1_cv 函数并返回其结果
def _pro_rad1_cv_pywrap(double x0, double x1, double x2, double x3, double x4):
    cdef double y0
    cdef double y1
    pro_rad1_cv(x0, x1, x2, x3, x4, &y0, &y1)
    return y0, y1

# 定义一个 Cython 函数，用于计算 pro_rad2 的结果并通过指针返回两个值
cdef void pro_rad2(double x0, double x1, double x2, double x3, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pro_rad2"""
    y0[0] = _func_prolate_radial2_nocv_wrap(x0, x1, x2, x3, y1)

# 定义一个 Python 包装函数，调用 pro_rad2 函数并返回其结果
def _pro_rad2_pywrap(double x0, double x1, double x2, double x3):
    cdef double y0
    cdef double y1
    pro_rad2(x0, x1, x2, x3, &y0, &y1)
    return y0, y1

# 定义一个 Cython 函数，用于计算 pro_rad2_cv 的结果并通过指针返回两个值
cdef void pro_rad2_cv(double x0, double x1, double x2, double x3, double x4, double *y0, double *y1) noexcept nogil:
    """See the documentation for scipy.special.pro_rad2_cv"""
    # _func_prolate_radial2_wrap 是一个计算 pro_rad2_cv 的 C 函数的封装
    _func_prolate_radial2_wrap(x0, x1, x2, x3, x4, y0, y1)
    # 调用名为 _func_prolate_radial2_wrap 的函数，传入参数 x0, x1, x2, x3, x4, y0, y1
    """See the documentation for scipy.special.pro_rad2_cv"""
    _func_prolate_radial2_wrap(x0, x1, x2, x3, x4, y0, y1)
# 定义一个Cython函数 _pro_rad2_cv_pywrap，接受五个 double 类型参数，并返回两个 double 类型的值
def _pro_rad2_cv_pywrap(double x0, double x1, double x2, double x3, double x4):
    # 声明两个局部变量 y0 和 y1，类型为 double
    cdef double y0
    cdef double y1
    # 调用 C 函数 pro_rad2_cv，传入参数 x0 到 x4，并将计算结果存储到 y0 和 y1 中
    pro_rad2_cv(x0, x1, x2, x3, x4, &y0, &y1)
    # 返回 y0 和 y1 的值作为函数的结果
    return y0, y1

# 定义一个 Cython 函数 pseudo_huber，接受两个 double 类型参数，并返回一个 double 类型的值
cpdef double pseudo_huber(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.pseudo_huber"""
    # 调用内部函数 _func_pseudo_huber，将 x0 和 x1 作为参数传入，并返回结果
    return _func_pseudo_huber(x0, x1)

# 定义一个 Cython 函数 psi，接受一个 Dd_number_t 类型参数，并返回一个 Dd_number_t 类型的值
cpdef Dd_number_t psi(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.psi"""
    # 根据 Dd_number_t 的类型（double 或 double_complex）执行不同的操作
    if Dd_number_t is double_complex:
        # 如果是 double_complex 类型，执行复杂数相关计算，并返回结果
        return _complexstuff.double_complex_from_npy_cdouble(special_cdigamma(_complexstuff.npy_cdouble_from_double_complex(x0)))
    elif Dd_number_t is double:
        # 如果是 double 类型，调用特殊函数 special_digamma 计算，并返回结果
        return special_digamma(x0)
    else:
        # 其他类型的 Dd_number_t 返回 NAN（不是一个数字）
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个 Cython 函数 radian，接受三个 double 类型参数，并返回一个 double 类型的值
cpdef double radian(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.radian"""
    # 调用 C 函数 cephes_radian，传入参数 x0、x1 和 x2，并返回计算结果
    return cephes_radian(x0, x1, x2)

# 定义一个 Cython 函数 rel_entr，接受两个 double 类型参数，并返回一个 double 类型的值
cpdef double rel_entr(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.rel_entr"""
    # 调用内部函数 _func_rel_entr，将 x0 和 x1 作为参数传入，并返回结果
    return _func_rel_entr(x0, x1)

# 定义一个 Cython 函数 rgamma，接受一个 Dd_number_t 类型参数，并返回一个 Dd_number_t 类型的值
cpdef Dd_number_t rgamma(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.rgamma"""
    # 根据 Dd_number_t 的类型（double 或 double_complex）执行不同的操作
    if Dd_number_t is double_complex:
        # 如果是 double_complex 类型，执行复杂数相关计算，并返回结果
        return _complexstuff.double_complex_from_npy_cdouble(special_crgamma(_complexstuff.npy_cdouble_from_double_complex(x0))) 
    elif Dd_number_t is double:
        # 如果是 double 类型，调用特殊函数 special_rgamma 计算，并返回结果
        return special_rgamma(x0)
    else:
        # 其他类型的 Dd_number_t 返回 NAN（不是一个数字）
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个 Cython 函数 round，接受一个 double 类型参数，并返回一个 double 类型的值
cpdef double round(double x0) noexcept nogil:
    """See the documentation for scipy.special.round"""
    # 调用 C 函数 cephes_round，传入参数 x0，并返回计算结果
    return cephes_round(x0)

# 定义一个 Cython 函数 shichi，接受一个 Dd_number_t 类型参数和两个 Dd_number_t 类型的指针作为输出
# 在不使用全局解释器锁的情况下（nogil），根据 Dd_number_t 类型执行不同的操作
cdef void shichi(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1) noexcept nogil:
    """See the documentation for scipy.special.shichi"""
    if Dd_number_t is double_complex:
        # 如果是 double_complex 类型，调用复杂数相关函数计算结果，并存储到 y0 和 y1 中
        _func_cshichi(x0, y0, y1)
    elif Dd_number_t is double:
        # 如果是 double 类型，调用 C 函数 _func_cephes_shichi_wrap 计算结果，并存储到 y0 和 y1 中
        _func_cephes_shichi_wrap(x0, y0, y1)
    else:
        # 其他类型的 Dd_number_t，将 y0 和 y1 的值设置为 NAN
        if Dd_number_t is double_complex:
            y0[0] = NAN
            y1[0] = NAN
        else:
            y0[0] = NAN
            y1[0] = NAN

# 定义一个 Python 包装函数 _shichi_pywrap，接受一个 Dd_number_t 类型参数，并返回两个 Dd_number_t 类型的值
def _shichi_pywrap(Dd_number_t x0):
    # 声明两个局部变量 y0 和 y1，类型为 Dd_number_t
    cdef Dd_number_t y0
    cdef Dd_number_t y1
    # 调用 Cython 函数 shichi，传入 x0 和 y0、y1 的地址作为参数，并获取计算结果
    shichi(x0, &y0, &y1)
    # 返回 y0 和 y1 的值作为函数的结果
    return y0, y1

# 定义一个 Cython 函数 sici，接受一个 Dd_number_t 类型参数和两个 Dd_number_t 类型的指针作为输出
# 在不使用全局解释器锁的情况下（nogil），根据 Dd_number_t 类型执行不同的操作
cdef void sici(Dd_number_t x0, Dd_number_t *y0, Dd_number_t *y1) noexcept nogil:
    """See the documentation for scipy.special.sici"""
    if Dd_number_t is double_complex:
        # 如果是 double_complex 类型，调用复杂数相关函数计算结果，并存储到 y0 和 y1 中
        _func_csici(x0, y0, y1)
    elif Dd_number_t is double:
        # 如果是 double 类型，调用 C 函数 _func_cephes_sici_wrap 计算结果，并存储到 y0 和 y1 中
        _func_cephes_sici_wrap(x0, y0, y1)
    else:
        # 其他类型的 Dd_number_t，将 y0 和 y1 的值设置为 NAN
        if Dd_number_t is double_complex:
            y0[0] = NAN
            y1[0] = NAN
        else:
            y0[0] = NAN
            y1[0] = NAN

# 定义一个 Python 包装函数 _sici_pywrap，接受一个 Dd_number_t 类型参数，并返回两个 Dd_number_t 类型的值
def _sici_pywrap(Dd_number_t x0):
    # 声明两个局部变量 y0 和 y1，类型为 Dd_number_t
    cdef Dd_number_t y0
    cdef Dd_number_t y1
    # 调用 Cython 函数 sici，传入 x0 和 y0、y1 的地址作为参数，并获取计算结果
    sici(x0, &y0, &y1)
    # 返回 y0 和 y1 的值作为函数的结果
    return y0, y1

# 定义一个 Cython 函数 sindg，接受一个
cpdef double smirnov(dl_number_t x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.smirnov"""
    # 根据参数类型选择对应的函数进行计算
    if dl_number_t is double:
        return _func_smirnov_unsafe(x0, x1)
    elif dl_number_t is long:
        return cephes_smirnov(x0, x1)
    else:
        return NAN

cpdef double smirnovi(dl_number_t x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.smirnovi"""
    # 根据参数类型选择对应的函数进行计算
    if dl_number_t is double:
        return _func_smirnovi_unsafe(x0, x1)
    elif dl_number_t is long:
        return cephes_smirnovi(x0, x1)
    else:
        return NAN

cpdef Dd_number_t spence(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.spence"""
    # 根据参数类型选择对应的函数进行计算
    if Dd_number_t is double_complex:
        return _func_cspence(x0)
    elif Dd_number_t is double:
        return cephes_spence(x0)
    else:
        # 若参数类型不符合预期，返回 NaN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

cpdef double complex sph_harm(dl_number_t x0, dl_number_t x1, double x2, double x3) noexcept nogil:
    """See the documentation for scipy.special.sph_harm"""
    # 根据参数类型选择对应的函数进行计算，并转换为复数类型返回
    if dl_number_t is double:
        return _complexstuff.double_complex_from_npy_cdouble(special_sph_harm_unsafe(x0, x1, x2, x3))
    elif dl_number_t is long:
        return _complexstuff.double_complex_from_npy_cdouble(special_sph_harm(x0, x1, x2, x3))
    else:
        return NAN

cpdef double stdtr(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.stdtr"""
    # 调用特定函数计算结果并返回
    return _func_stdtr(x0, x1)

cpdef double stdtridf(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.stdtridf"""
    # 调用特定函数计算结果并返回
    return _func_stdtridf(x0, x1)

cpdef double stdtrit(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.stdtrit"""
    # 调用特定函数计算结果并返回
    return _func_stdtrit(x0, x1)

cpdef double struve(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.struve"""
    # 调用特定函数计算结果并返回
    return cephes_struve_h(x0, x1)

cpdef double tandg(double x0) noexcept nogil:
    """See the documentation for scipy.special.tandg"""
    # 调用特定函数计算结果并返回
    return cephes_tandg(x0)

cpdef double tklmbda(double x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.tklmbda"""
    # 调用特定函数计算结果并返回
    return cephes_tukeylambdacdf(x0, x1)

cpdef double complex wofz(double complex x0) noexcept nogil:
    """See the documentation for scipy.special.wofz"""
    # 调用特定函数计算结果并返回
    return (<double complex(*)(double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_faddeeva_w)(x0)

cpdef Dd_number_t wrightomega(Dd_number_t x0) noexcept nogil:
    """See the documentation for scipy.special.wrightomega"""
    # 根据参数类型选择对应的函数进行计算
    if Dd_number_t is double_complex:
        return (<double complex(*)(double complex) noexcept nogil>scipy.special._ufuncs_cxx._export_wrightomega)(x0)
    elif Dd_number_t is double:
        return (<double(*)(double) noexcept nogil>scipy.special._ufuncs_cxx._export_wrightomega_real)(x0)
    else:
        # 如果条件不满足第一个条件（Dd_number_t 是 double_complex 类型），则执行以下操作
        if Dd_number_t is double_complex:
            # 返回 NaN（Not a Number）
            return NAN
        else:
            # 如果条件不满足第二个条件（Dd_number_t 不是 double_complex 类型），则执行以下操作
            # 返回 NaN（Not a Number）
            return NAN
# 定义一个 Cython 函数 xlog1py，用于计算 scipy.special.xlog1py 函数的值
cpdef Dd_number_t xlog1py(Dd_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.xlog1py"""
    # 如果 Dd_number_t 类型是 double，则调用对应的函数 _func_xlog1py[double]
    if Dd_number_t is double:
        return _func_xlog1py[double](x0, x1)
    # 如果 Dd_number_t 类型是 double_complex，则调用对应的函数 _func_xlog1py[double_complex]
    elif Dd_number_t is double_complex:
        return _func_xlog1py[double_complex](x0, x1)
    else:
        # 如果类型不是上述两种情况，则返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个 Cython 函数 xlogy，用于计算 scipy.special.xlogy 函数的值
cpdef Dd_number_t xlogy(Dd_number_t x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.xlogy"""
    # 如果 Dd_number_t 类型是 double，则调用对应的函数 _func_xlogy[double]
    if Dd_number_t is double:
        return _func_xlogy[double](x0, x1)
    # 如果 Dd_number_t 类型是 double_complex，则调用对应的函数 _func_xlogy[double_complex]
    elif Dd_number_t is double_complex:
        return _func_xlogy[double_complex](x0, x1)
    else:
        # 如果类型不是上述两种情况，则返回 NAN
        if Dd_number_t is double_complex:
            return NAN
        else:
            return NAN

# 定义一个 Cython 函数 y0，用于计算 scipy.special.y0 函数的值
cpdef double y0(double x0) noexcept nogil:
    """See the documentation for scipy.special.y0"""
    # 调用 cephes_y0 函数计算 y0 函数的值并返回
    return cephes_y0(x0)

# 定义一个 Cython 函数 y1，用于计算 scipy.special.y1 函数的值
cpdef double y1(double x0) noexcept nogil:
    """See the documentation for scipy.special.y1"""
    # 调用 cephes_y1 函数计算 y1 函数的值并返回
    return cephes_y1(x0)

# 定义一个 Cython 函数 yn，用于计算 scipy.special.yn 函数的值
cpdef double yn(dl_number_t x0, double x1) noexcept nogil:
    """See the documentation for scipy.special.yn"""
    # 如果 dl_number_t 类型是 double，则调用 _func_yn_unsafe 函数计算 yn 函数的值
    if dl_number_t is double:
        return _func_yn_unsafe(x0, x1)
    # 如果 dl_number_t 类型是 long，则调用 cephes_yn 函数计算 yn 函数的值
    elif dl_number_t is long:
        return cephes_yn(x0, x1)
    else:
        # 如果类型不是上述两种情况，则返回 NAN
        return NAN

# 定义一个 Cython 函数 yv，用于计算 scipy.special.yv 函数的值
cpdef Dd_number_t yv(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.yv"""
    # 如果 Dd_number_t 类型是 double_complex，则调用 special_ccyl_bessel_y 函数计算 yv 函数的值
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_bessel_y(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))
    # 如果 Dd_number_t 类型是 double，则调用 special_cyl_bessel_y 函数计算 yv 函数的值
    elif Dd_number_t is double:
        return special_cyl_bessel_y(x0, x1)
    else:
        # 如果类型不是上述两种情况，则返回 NAN
        return NAN

# 定义一个 Cython 函数 yve，用于计算 scipy.special.yve 函数的值
cpdef Dd_number_t yve(double x0, Dd_number_t x1) noexcept nogil:
    """See the documentation for scipy.special.yve"""
    # 如果 Dd_number_t 类型是 double_complex，则调用 special_ccyl_bessel_ye 函数计算 yve 函数的值
    if Dd_number_t is double_complex:
        return _complexstuff.double_complex_from_npy_cdouble(special_ccyl_bessel_ye(x0, _complexstuff.npy_cdouble_from_double_complex(x1)))
    # 如果 Dd_number_t 类型是 double，则调用 special_cyl_bessel_ye 函数计算 yve 函数的值
    elif Dd_number_t is double:
        return special_cyl_bessel_ye(x0, x1)
    else:
        # 如果类型不是上述两种情况，则返回 NAN
        return NAN

# 定义一个 Cython 函数 zetac，用于计算 scipy.special.zetac 函数的值
cpdef double zetac(double x0) noexcept nogil:
    """See the documentation for scipy.special.zetac"""
    # 调用 cephes_zetac 函数计算 zetac 函数的值并返回
    return cephes_zetac(x0)

# 定义一个 Cython 函数 wright_bessel，用于计算 scipy.special.wright_bessel 函数的值
cpdef double wright_bessel(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.wright_bessel"""
    # 调用 special_wright_bessel 函数计算 wright_bessel 函数的值并返回
    return special_wright_bessel(x0, x1, x2)

# 定义一个 Cython 函数 log_wright_bessel，用于计算 scipy.special.log_wright_bessel 函数的值
cpdef double log_wright_bessel(double x0, double x1, double x2) noexcept nogil:
    """See the documentation for scipy.special.log_wright_bessel"""
    # 调用 special_log_wright_bessel 函数计算 log_wright_bessel 函数的值并返回
    return special_log_wright_bessel(x0, x1, x2)

# 定义一个 Cython 函数 ndtri_exp，用于计算 scipy.special.ndtri_exp 函数的值
cpdef double ndtri_exp(double x0) noexcept nogil:
    """See the documentation for scipy.special.ndtri_exp"""
    # 调用 _func_ndtri_exp 函数计算 ndtri_exp 函数的值并返回
    return _func_ndtri_exp(x0)
# 定义了一个Cython函数，用于计算球谐函数的第一类Bessel函数
cpdef number_t spherical_jn(long n, number_t z, bint derivative=0) noexcept nogil:
    """See the documentation for scipy.special.spherical_jn"""
    # 如果需要求导数
    if derivative:
        # 如果number_t是双精度浮点数
        if number_t is double:
            # 调用特殊函数库计算球谐函数的第一类Bessel函数的导数
            return special_sph_bessel_j_jac(n, z)
        else:
            # 调用复数特殊函数库将复数转换成双精度复数并计算球谐函数的第一类Bessel函数的导数
            return _complexstuff.double_complex_from_npy_cdouble(special_csph_bessel_j_jac(n, _complexstuff.npy_cdouble_from_double_complex(z)))

    # 如果不需要求导数
    if number_t is double:
        # 直接调用特殊函数库计算球谐函数的第一类Bessel函数
        return special_sph_bessel_j(n, z)
    else:
        # 调用复数特殊函数库将复数转换成双精度复数并计算球谐函数的第一类Bessel函数
        return _complexstuff.double_complex_from_npy_cdouble(special_csph_bessel_j(n, _complexstuff.npy_cdouble_from_double_complex(z)))

# 定义了一个Cython函数，用于计算球谐函数的第二类Bessel函数
cpdef number_t spherical_yn(long n, number_t z, bint derivative=0) noexcept nogil:
    """See the documentation for scipy.special.spherical_yn"""
    # 如果需要求导数
    if derivative:
        # 如果number_t是双精度浮点数
        if number_t is double:
            # 调用特殊函数库计算球谐函数的第二类Bessel函数的导数
            return special_sph_bessel_y_jac(n, z)
        else:
            # 调用复数特殊函数库将复数转换成双精度复数并计算球谐函数的第二类Bessel函数的导数
            return _complexstuff.double_complex_from_npy_cdouble(special_csph_bessel_y_jac(n, _complexstuff.npy_cdouble_from_double_complex(z)))

    # 如果不需要求导数
    if number_t is double:
        # 直接调用特殊函数库计算球谐函数的第二类Bessel函数
        return special_sph_bessel_y(n, z)
    else:
        # 调用复数特殊函数库将复数转换成双精度复数并计算球谐函数的第二类Bessel函数
        return _complexstuff.double_complex_from_npy_cdouble(special_csph_bessel_y(n, _complexstuff.npy_cdouble_from_double_complex(z)))

# 定义了一个Cython函数，用于计算球谐函数的修正第一类Bessel函数
cpdef number_t spherical_in(long n, number_t z, bint derivative=0) noexcept nogil:
    """See the documentation for scipy.special.spherical_in"""
    # 如果需要求导数
    if derivative:
        # 如果number_t是双精度浮点数
        if number_t is double:
            # 调用特殊函数库计算球谐函数的修正第一类Bessel函数的导数
            return special_sph_bessel_i_jac(n, z)
        else:
            # 调用复数特殊函数库将复数转换成双精度复数并计算球谐函数的修正第一类Bessel函数的导数
            return _complexstuff.double_complex_from_npy_cdouble(special_csph_bessel_i_jac(n, _complexstuff.npy_cdouble_from_double_complex(z)))

    # 如果不需要求导数
    if number_t is double:
        # 直接调用特殊函数库计算球谐函数的修正第一类Bessel函数
        return special_sph_bessel_i(n, z)
    else:
        # 调用复数特殊函数库将复数转换成双精度复数并计算球谐函数的修正第一类Bessel函数
        return _complexstuff.double_complex_from_npy_cdouble(special_csph_bessel_i(n, _complexstuff.npy_cdouble_from_double_complex(z)))

# 定义了一个Cython函数，用于计算球谐函数的修正第二类Bessel函数
cpdef number_t spherical_kn(long n, number_t z, bint derivative=0) noexcept nogil:
    """See the documentation for scipy.special.spherical_kn"""
    # 如果需要求导数
    if derivative:
        # 如果number_t是双精度浮点数
        if number_t is double:
            # 调用特殊函数库计算球谐函数的修正第二类Bessel函数的导数
            return special_sph_bessel_k_jac(n, z)
        else:
            # 调用复数特殊函数库将复数转换成双精度复数并计算球谐函数的修正第二类Bessel函数的导数
            return _complexstuff.double_complex_from_npy_cdouble(special_csph_bessel_k_jac(n, _complexstuff.npy_cdouble_from_double_complex(z)))

    # 如果不需要求导数
    if number_t is double:
        # 直接调用特殊函数库计算球谐函数的修正第二类Bessel函数
        return special_sph_bessel_k(n, z)
    else:
        # 调用复数特殊函数库将复数转换成双精度复数并计算球谐函数的修正第二类Bessel函数
        return _complexstuff.double_complex_from_npy_cdouble(special_csph_bessel_k(n, _complexstuff.npy_cdouble_from_double_complex(z)))

# 定义了一个Python函数，用于在Python中使用特殊函数库计算Airy函数
def _bench_airy_d_py(int N, double x0):
    cdef int n
    # 循环调用特殊函数库计算Airy函数
    for n in range(N):
        _ufuncs.airy(x0)

# 定义了一个Cython函数，用于在Cython中使用特殊函数库计算Airy函数
def _bench_airy_d_cy(int N, double x0):
    cdef int n
    cdef double y0
    cdef double y1
    cdef double y2
    cdef double y3
    # 循环调用特殊函数库计算Airy函数，结果存储在指针指向的变量中
    for n in range(N):
        airy(x0, &y0, &y1, &y2, &y3)

# 定义了一个Python函数，用于在Python中使用特殊函数库计算复数形式的Airy函数
def _bench_airy_D_py(int N, double complex x0):
    cdef int n
    # 循环调用特殊函数库计算复数形式的Airy函数
    for n in range(N):
        _ufuncs.airy(x0)

# 定义了一个Cython函数，用于在Cython中使用特殊函数库计算复数形式的Airy函数
def _bench_airy_D_cy(int N, double complex x0):
    cdef int n
    cdef double complex y0
    # 定义三个复数类型的变量 y1, y2, y3
    cdef double complex y1
    cdef double complex y2
    cdef double complex y3
    
    # 使用循环迭代范围为 N 的整数 n，其中 N 为预先定义的变量
    for n in range(N):
        # 调用 airy 函数，计算 Airy 函数及其导数的值，并通过指针将结果存入 y0, y1, y2, y3 中
        airy(x0, &y0, &y1, &y2, &y3)
def _bench_beta_dd_py(int N, double x0, double x1):
    cdef int n
    for n in range(N):
        _ufuncs.beta(x0, x1)


    # 使用 Cython 声明的 _ufuncs.beta 函数，计算 Beta 函数值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.beta(x0, x1)



def _bench_beta_dd_cy(int N, double x0, double x1):
    cdef int n
    for n in range(N):
        beta(x0, x1)


    # 使用 Cython 生成的 beta 函数，计算 Beta 函数值，重复 N 次
    cdef int n
    for n in range(N):
        beta(x0, x1)



def _bench_erf_d_py(int N, double x0):
    cdef int n
    for n in range(N):
        _ufuncs.erf(x0)


    # 使用 Cython 声明的 _ufuncs.erf 函数，计算误差函数值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.erf(x0)



def _bench_erf_d_cy(int N, double x0):
    cdef int n
    for n in range(N):
        erf(x0)


    # 使用 Cython 生成的 erf 函数，计算误差函数值，重复 N 次
    cdef int n
    for n in range(N):
        erf(x0)



def _bench_erf_D_py(int N, double complex x0):
    cdef int n
    for n in range(N):
        _ufuncs.erf(x0)


    # 使用 Cython 声明的 _ufuncs.erf 函数，计算复数参数的误差函数值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.erf(x0)



def _bench_erf_D_cy(int N, double complex x0):
    cdef int n
    for n in range(N):
        erf(x0)


    # 使用 Cython 生成的 erf 函数，计算复数参数的误差函数值，重复 N 次
    cdef int n
    for n in range(N):
        erf(x0)



def _bench_exprel_d_py(int N, double x0):
    cdef int n
    for n in range(N):
        _ufuncs.exprel(x0)


    # 使用 Cython 声明的 _ufuncs.exprel 函数，计算指数相对误差函数值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.exprel(x0)



def _bench_exprel_d_cy(int N, double x0):
    cdef int n
    for n in range(N):
        exprel(x0)


    # 使用 Cython 生成的 exprel 函数，计算指数相对误差函数值，重复 N 次
    cdef int n
    for n in range(N):
        exprel(x0)



def _bench_gamma_d_py(int N, double x0):
    cdef int n
    for n in range(N):
        _ufuncs.gamma(x0)


    # 使用 Cython 声明的 _ufuncs.gamma 函数，计算 Gamma 函数值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.gamma(x0)



def _bench_gamma_d_cy(int N, double x0):
    cdef int n
    for n in range(N):
        gamma(x0)


    # 使用 Cython 生成的 gamma 函数，计算 Gamma 函数值，重复 N 次
    cdef int n
    for n in range(N):
        gamma(x0)



def _bench_gamma_D_py(int N, double complex x0):
    cdef int n
    for n in range(N):
        _ufuncs.gamma(x0)


    # 使用 Cython 声明的 _ufuncs.gamma 函数，计算复数参数的 Gamma 函数值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.gamma(x0)



def _bench_gamma_D_cy(int N, double complex x0):
    cdef int n
    for n in range(N):
        gamma(x0)


    # 使用 Cython 生成的 gamma 函数，计算复数参数的 Gamma 函数值，重复 N 次
    cdef int n
    for n in range(N):
        gamma(x0)



def _bench_jv_dd_py(int N, double x0, double x1):
    cdef int n
    for n in range(N):
        _ufuncs.jv(x0, x1)


    # 使用 Cython 声明的 _ufuncs.jv 函数，计算贝塞尔函数 J_v(x0, x1) 的值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.jv(x0, x1)



def _bench_jv_dd_cy(int N, double x0, double x1):
    cdef int n
    for n in range(N):
        jv(x0, x1)


    # 使用 Cython 生成的 jv 函数，计算贝塞尔函数 J_v(x0, x1) 的值，重复 N 次
    cdef int n
    for n in range(N):
        jv(x0, x1)



def _bench_jv_dD_py(int N, double x0, double complex x1):
    cdef int n
    for n in range(N):
        _ufuncs.jv(x0, x1)


    # 使用 Cython 声明的 _ufuncs.jv 函数，计算复数参数的贝塞尔函数 J_v(x0, x1) 的值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.jv(x0, x1)



def _bench_jv_dD_cy(int N, double x0, double complex x1):
    cdef int n
    for n in range(N):
        jv(x0, x1)


    # 使用 Cython 生成的 jv 函数，计算复数参数的贝塞尔函数 J_v(x0, x1) 的值，重复 N 次
    cdef int n
    for n in range(N):
        jv(x0, x1)



def _bench_loggamma_D_py(int N, double complex x0):
    cdef int n
    for n in range(N):
        _ufuncs.loggamma(x0)


    # 使用 Cython 声明的 _ufuncs.loggamma 函数，计算复数参数的对数 Gamma 函数值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.loggamma(x0)



def _bench_loggamma_D_cy(int N, double complex x0):
    cdef int n
    for n in range(N):
        loggamma(x0)


    # 使用 Cython 生成的 loggamma 函数，计算复数参数的对数 Gamma 函数值，重复 N 次
    cdef int n
    for n in range(N):
        loggamma(x0)



def _bench_logit_d_py(int N, double x0):
    cdef int n
    for n in range(N):
        _ufuncs.logit(x0)


    # 使用 Cython 声明的 _ufuncs.logit 函数，计算 Logit 函数值，重复 N 次
    cdef int n
    for n in range(N):
        _ufuncs.logit(x0)



def _bench_logit_d_cy(int N, double x0):
    cdef int n
    for n in range(N):
        logit(x0)


    # 使用 Cython 生成的 logit 函数，计算 Logit 函数值，重
```