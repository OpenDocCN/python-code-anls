# `D:\src\scipysrc\scipy\scipy\special\__init__.py`

```
"""
========================================
Special functions (:mod:`scipy.special`)
========================================

.. currentmodule:: scipy.special

Almost all of the functions below accept NumPy arrays as input
arguments as well as single numbers. This means they follow
broadcasting and automatic array-looping rules. Technically,
they are `NumPy universal functions
<https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-basics>`_.
Functions which do not accept NumPy arrays are marked by a warning
in the section description.

.. seealso::

   `scipy.special.cython_special` -- Typed Cython versions of special functions


Error handling
==============

Errors are handled by returning NaNs or other appropriate values.
Some of the special function routines can emit warnings or raise
exceptions when an error occurs. By default this is disabled; to
query and control the current error handling state the following
functions are provided.

.. autosummary::
   :toctree: generated/

   geterr                 -- Get the current way of handling special-function errors.
   seterr                 -- Set how special-function errors are handled.
   errstate               -- Context manager for special-function error handling.
   SpecialFunctionWarning -- Warning that can be emitted by special functions.
   SpecialFunctionError   -- Exception that can be raised by special functions.

Available functions
===================

Airy functions
--------------

.. autosummary::
   :toctree: generated/

   airy     -- Airy functions and their derivatives.
   airye    -- Exponentially scaled Airy functions and their derivatives.
   ai_zeros -- Compute `nt` zeros and values of the Airy function Ai and its derivative.
   bi_zeros -- Compute `nt` zeros and values of the Airy function Bi and its derivative.
   itairy   -- Integrals of Airy functions


Elliptic functions and integrals
--------------------------------

.. autosummary::
   :toctree: generated/

   ellipj    -- Jacobian elliptic functions.
   ellipk    -- Complete elliptic integral of the first kind.
   ellipkm1  -- Complete elliptic integral of the first kind around `m` = 1.
   ellipkinc -- Incomplete elliptic integral of the first kind.
   ellipe    -- Complete elliptic integral of the second kind.
   ellipeinc -- Incomplete elliptic integral of the second kind.
   elliprc   -- Degenerate symmetric integral RC.
   elliprd   -- Symmetric elliptic integral of the second kind.
   elliprf   -- Completely-symmetric elliptic integral of the first kind.
   elliprg   -- Completely-symmetric elliptic integral of the second kind.
   elliprj   -- Symmetric elliptic integral of the third kind.

Bessel functions
----------------

# This block of comments documents various special mathematical functions provided by the scipy.special module.
# Bessel 函数的第一类实数阶和复数参数
jv                -- Bessel function of the first kind of real order and \
                        complex argument.
# 指数尺度下的第一类 Bessel 函数
jve               -- Exponentially scaled Bessel function of order `v`.
# Bessel 函数的第二类整数阶和实数参数
yn                -- Bessel function of the second kind of integer order and \
                        real argument.
# Bessel 函数的第二类实数阶和复数参数
yv                -- Bessel function of the second kind of real order and \
                        complex argument.
# 指数尺度下的第二类 Bessel 函数
yve               -- Exponentially scaled Bessel function of the second kind \
                        of real order.
# 修改 Bessel 函数的第二类整数阶 `n`
kn                -- Modified Bessel function of the second kind of integer \
                        order `n`
# 修改 Bessel 函数的第二类实数阶 `v`
kv                -- Modified Bessel function of the second kind of real order \
                        `v`
# 指数尺度下的修改 Bessel 函数的第二类
kve               -- Exponentially scaled modified Bessel function of the \
                        second kind.
# 修改 Bessel 函数的第一类实数阶
iv                -- Modified Bessel function of the first kind of real order.
# 指数尺度下的修改 Bessel 函数的第一类
ive               -- Exponentially scaled modified Bessel function of the \
                        first kind.
# 第一类 Hankel 函数的第一类
hankel1           -- Hankel function of the first kind.
# 指数尺度下的第一类 Hankel 函数的第一类
hankel1e          -- Exponentially scaled Hankel function of the first kind.
# 第一类 Hankel 函数的第二类
hankel2           -- Hankel function of the second kind.
# 指数尺度下的第二类 Hankel 函数的第二类
hankel2e          -- Exponentially scaled Hankel function of the second kind.
# Wright 的广义 Bessel 函数
wright_bessel     -- Wright's generalized Bessel function.
# Wright 的广义 Bessel 函数的对数
log_wright_bessel -- Logarithm of Wright's generalized Bessel function.

The following function does not accept NumPy arrays (it is not a
universal function):

# Jahnke-Emden Lambda 函数，Lambdav(x)
lmbda -- Jahnke-Emden Lambda function, Lambdav(x).

Zeros of Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^

The following functions do not accept NumPy arrays (they are not
universal functions):

# 计算整数阶 Bessel 函数 Jn 和 Jn' 的零点
jnjnp_zeros -- Compute zeros of integer-order Bessel functions Jn and Jn'.
# 计算 Bessel 函数 Jn(x), Jn'(x), Yn(x), 和 Yn'(x) 的 nt 个零点
jnyn_zeros  -- Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).
# 计算整数阶 Bessel 函数 Jn(x) 的零点
jn_zeros    -- Compute zeros of integer-order Bessel function Jn(x).
# 计算整数阶 Bessel 函数导数 Jn'(x) 的零点
jnp_zeros   -- Compute zeros of integer-order Bessel function derivative Jn'(x).
# 计算整数阶 Bessel 函数 Yn(x) 的零点
yn_zeros    -- Compute zeros of integer-order Bessel function Yn(x).
# 计算整数阶 Bessel 函数导数 Yn'(x) 的零点
ynp_zeros   -- Compute zeros of integer-order Bessel function derivative Yn'(x).
# 计算 Bessel 函数 Y0(z) 和其在每个零点处的导数的 nt 个零点
y0_zeros    -- Compute nt zeros of Bessel function Y0(z), and derivative at each zero.
# 计算 Bessel 函数 Y1(z) 和其在每个零点处的导数的 nt 个零点
y1_zeros    -- Compute nt zeros of Bessel function Y1(z), and derivative at each zero.
# 计算 Bessel 导数 Y1'(z) 和其值的 nt 个零点
y1p_zeros   -- Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.

Faster versions of common Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Bessel functions of the first kind of order 0.
j0  -- Bessel function of the first kind of order 0.

# Bessel functions of the first kind of order 1.
j1  -- Bessel function of the first kind of order 1.

# Bessel functions of the second kind of order 0.
y0  -- Bessel function of the second kind of order 0.

# Bessel functions of the second kind of order 1.
y1  -- Bessel function of the second kind of order 1.

# Modified Bessel function of order 0.
i0  -- Modified Bessel function of order 0.

# Exponentially scaled modified Bessel function of order 0.
i0e -- Exponentially scaled modified Bessel function of order 0.

# Modified Bessel function of order 1.
i1  -- Modified Bessel function of order 1.

# Exponentially scaled modified Bessel function of order 1.
i1e -- Exponentially scaled modified Bessel function of order 1.

# Modified Bessel function of the second kind of order 0, K0.
k0  -- Modified Bessel function of the second kind of order 0, :math:`K_0`.

# Exponentially scaled modified Bessel function K of order 0.
k0e -- Exponentially scaled modified Bessel function K of order 0

# Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.
k1  -- Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.

# Exponentially scaled modified Bessel function K of order 1.
k1e -- Exponentially scaled modified Bessel function K of order 1.

# Integrals of Bessel functions
itj0y0     -- Integrals of Bessel functions of order 0.
it2j0y0    -- Integrals related to Bessel functions of order 0.
iti0k0     -- Integrals of modified Bessel functions of order 0.
it2i0k0    -- Integrals related to modified Bessel functions of order 0.
besselpoly -- Weighted integral of a Bessel function.

# Derivatives of Bessel functions
jvp  -- Compute nth derivative of Bessel function Jv(z) with respect to `z`.
yvp  -- Compute nth derivative of Bessel function Yv(z) with respect to `z`.
kvp  -- Compute nth derivative of real-order modified Bessel function Kv(z).
ivp  -- Compute nth derivative of modified Bessel function Iv(z) with respect to `z`.
h1vp -- Compute nth derivative of Hankel function H1v(z) with respect to `z`.
h2vp -- Compute nth derivative of Hankel function H2v(z) with respect to `z`.

# Spherical Bessel functions
spherical_jn -- Spherical Bessel function of the first kind or its derivative.
spherical_yn -- Spherical Bessel function of the second kind or its derivative.
spherical_in -- Modified spherical Bessel function of the first kind or its derivative.
spherical_kn -- Modified spherical Bessel function of the second kind or its derivative.

# Riccati-Bessel functions
# These functions do not accept NumPy arrays and are not universal functions
riccati_jn -- Compute Ricatti-Bessel function of the first kind and its derivative.
riccati_yn -- Compute Ricatti-Bessel function of the second kind and its derivative.

# Struve functions
struve       -- Struve function.
modstruve    -- Modified Struve function.
itstruve0    -- Integral of the Struve function of order 0.
it2struve0   -- Integral related to the Struve function of order 0.
itmodstruve0 -- Integral of the modified Struve function of order 0.
# Binomial distribution functions

# Cumulative distribution function of the binomial distribution.
bdtr

# Survival function of the binomial distribution.
bdtrc

# Inverse function to `bdtr` with respect to `p`.
bdtri

# Inverse function to `bdtr` with respect to `k`.
bdtrik

# Inverse function to `bdtr` with respect to `n`.
bdtrin

# Beta distribution functions

# Cumulative distribution function of the beta distribution.
btdtr

# The `p`-th quantile of the beta distribution.
btdtri

# Inverse of `btdtr` with respect to `a`.
btdtria

# Alias for `btdtria(a, p, x)`.
btdtrib

# F distribution functions

# F cumulative distribution function.
fdtr

# F survival function.
fdtrc

# The `p`-th quantile of the F-distribution.
fdtri

# Inverse cumulative distribution function of the F-distribution.
fdtridfd

# Gamma distribution functions

# Gamma distribution cumulative distribution function.
gdtr

# Gamma distribution survival function.
gdtrc

# Inverse of `gdtr` with respect to `a`.
gdtria

# Inverse of `gdtr` with respect to `b`.
gdtrib

# Inverse of `gdtr` with respect to `x`.
gdtrix

# Negative binomial distribution functions

# Negative binomial cumulative distribution function.
nbdtr

# Negative binomial survival function.
nbdtrc

# Inverse of `nbdtr` with respect to `p`.
nbdtri

# Inverse of `nbdtr` with respect to `k`.
nbdtrik

# Inverse of `nbdtr` with respect to `n`.
nbdtrin

# Noncentral F distribution functions

# Cumulative distribution function of the non-central F distribution.
ncfdtr

# Calculate degrees of freedom (denominator) for the noncentral F-distribution.
ncfdtridfd

# Calculate degrees of freedom (numerator) for the noncentral F-distribution.
ncfdtridfn

# Inverse cumulative distribution function of the non-central F distribution.
ncfdtri

# Calculate non-centrality parameter for non-central F distribution.
ncfdtrinc

# Noncentral t distribution functions

# Cumulative distribution function of the non-central `t` distribution.
nctdtr

# Calculate degrees of freedom for non-central t distribution.
nctdtridf

# Inverse cumulative distribution function of the non-central t distribution.
nctdtrit

# Calculate non-centrality parameter for non-central t distribution.
nctdtrinc

# Normal distribution functions (Note: Missing section in the provided code snippet)
# 计算给定其他参数的正态分布均值
nrdtrimn     -- Calculate mean of normal distribution given other params.

# 计算给定其他参数的正态分布标准差
nrdtrisd     -- Calculate standard deviation of normal distribution given other params.

# 正态分布的累积分布函数
ndtr         -- Normal cumulative distribution function.

# 正态分布累积分布函数的对数
log_ndtr     -- Logarithm of normal cumulative distribution function.

# `ndtr` 的反函数，输入为累积分布函数的值 x
ndtri        -- Inverse of `ndtr` vs x.

# `log_ndtr` 的反函数，输入为累积分布函数的值 x
ndtri_exp    -- Inverse of `log_ndtr` vs x.

Poisson 分布
^^^^^^^^^^^^

# 泊松分布的累积分布函数
pdtr         -- Poisson cumulative distribution function.

# 泊松分布的生存函数
pdtrc        -- Poisson survival function.

# `pdtr` 的反函数，输入为参数 m
pdtri        -- Inverse to `pdtr` vs m.

# `pdtr` 的反函数，输入为参数 k
pdtrik       -- Inverse to `pdtr` vs k.

Student t 分布
^^^^^^^^^^^^^

# 学生 t 分布的累积分布函数
stdtr        -- Student t distribution cumulative distribution function.

# `stdtr` 的反函数，输入为自由度 df
stdtridf     -- Inverse of `stdtr` vs df.

# `stdtr` 的反函数，输入为 t 值
stdtrit      -- Inverse of `stdtr` vs `t`.

Chi 方分布
^^^^^^^^^^

# 卡方分布的累积分布函数
chdtr        -- Chi square cumulative distribution function.

# 卡方分布的生存函数
chdtrc       -- Chi square survival function.

# `chdtrc` 的反函数
chdtri       -- Inverse to `chdtrc`.

# `chdtr` 的反函数，输入为参数 v
chdtriv      -- Inverse to `chdtr` vs `v`.

非中心卡方分布
^^^^^^^^^^^^^^

# 非中心卡方分布的累积分布函数
chndtr       -- Non-central chi square cumulative distribution function.

# `chndtr` 的反函数，输入为自由度 df
chndtridf    -- Inverse to `chndtr` vs `df`.

# `chndtr` 的反函数，输入为非中心参数 nc
chndtrinc    -- Inverse to `chndtr` vs `nc`.

# `chndtr` 的反函数，输入为变量 x
chndtrix     -- Inverse to `chndtr` vs `x`.

Kolmogorov 分布
^^^^^^^^^^^^^^^

# Kolmogorov-Smirnov 补充累积分布函数
smirnov      -- Kolmogorov-Smirnov complementary cumulative distribution function.

# `smirnov` 的反函数
smirnovi     -- Inverse to `smirnov`.

# Kolmogorov 分布的补充累积分布函数
kolmogorov   -- Complementary cumulative distribution function of Kolmogorov distribution.

# `kolmogorov` 的反函数
kolmogi      -- Inverse function to `kolmogorov`.

Box-Cox 变换
^^^^^^^^^^^^^

# 计算 Box-Cox 变换
boxcox       -- Compute the Box-Cox transformation.

# 计算 1 + `x` 的 Box-Cox 变换
boxcox1p     -- Compute the Box-Cox transformation of 1 + `x`.

# 计算 Box-Cox 变换的逆变换
inv_boxcox   -- Compute the inverse of the Box-Cox transformation.

# 计算 1 + `x` 的 Box-Cox 变换的逆变换
inv_boxcox1p -- Compute the inverse of the Box-Cox transformation.

Sigmoid 函数
^^^^^^^^^^^^

# ndarrays 的 Logit 函数
logit        -- Logit ufunc for ndarrays.

# Logistic sigmoid 函数
expit        -- Logistic sigmoid function.

# Logistic sigmoid 函数的对数
log_expit    -- Logarithm of the logistic sigmoid function.

Miscellaneous
^^^^^^^^^^^^^

# Tukey-Lambda 累积分布函数
tklmbda      -- Tukey-Lambda cumulative distribution function.

# Owen's T 函数
owens_t      -- Owen's T Function.
# Elementwise function for computing entropy.
entr         -- Elementwise function for computing entropy.

# Elementwise function for computing relative entropy.
rel_entr     -- Elementwise function for computing relative entropy.

# Elementwise function for computing Kullback-Leibler divergence.
kl_div       -- Elementwise function for computing Kullback-Leibler divergence.

# Huber loss function.
huber        -- Huber loss function.

# Pseudo-Huber loss function.
pseudo_huber -- Pseudo-Huber loss function.

# Gamma function.
gamma        -- Gamma function.

# Logarithm of the absolute value of the Gamma function for real inputs.
gammaln      -- Logarithm of the absolute value of the Gamma function for real inputs.

# Principal branch of the logarithm of the Gamma function.
loggamma     -- Principal branch of the logarithm of the Gamma function.

# Sign of the gamma function.
gammasgn     -- Sign of the gamma function.

# Regularized lower incomplete gamma function.
gammainc     -- Regularized lower incomplete gamma function.

# Inverse to `gammainc`.
gammaincinv  -- Inverse to `gammainc`.

# Regularized upper incomplete gamma function.
gammaincc    -- Regularized upper incomplete gamma function.

# Inverse to `gammaincc`.
gammainccinv -- Inverse to `gammaincc`.

# Beta function.
beta         -- Beta function.

# Natural logarithm of absolute value of beta function.
betaln       -- Natural logarithm of absolute value of beta function.

# Incomplete beta integral.
betainc      -- Incomplete beta integral.

# Complemented incomplete beta integral.
betaincc     -- Complemented incomplete beta integral.

# Inverse function to beta integral.
betaincinv   -- Inverse function to beta integral.

# Inverse of the complemented incomplete beta integral.
betainccinv  -- Inverse of the complemented incomplete beta integral.

# The digamma function.
psi          -- The digamma function.

# Gamma function inverted.
rgamma       -- Gamma function inverted.

# Polygamma function n.
polygamma    -- Polygamma function n.

# Returns the log of multivariate gamma, also sometimes called the generalized gamma.
multigammaln -- Returns the log of multivariate gamma, also sometimes called the generalized gamma.

# psi(x[, out]).
digamma      -- psi(x[, out]).

# Rising factorial (z)_m.
poch         -- Rising factorial (z)_m.

# Returns the error function of complex argument.
erf           -- Returns the error function of complex argument.

# Complementary error function, ``1 - erf(x)``.
erfc          -- Complementary error function, ``1 - erf(x)``.

# Scaled complementary error function, ``exp(x**2) * erfc(x)``.
erfcx         -- Scaled complementary error function, ``exp(x**2) * erfc(x)``.

# Imaginary error function, ``-i erf(i z)``.
erfi          -- Imaginary error function, ``-i erf(i z)``.

# Inverse function for erf.
erfinv        -- Inverse function for erf.

# Inverse function for erfc.
erfcinv       -- Inverse function for erfc.

# Faddeeva function.
wofz          -- Faddeeva function.

# Dawson's integral.
dawsn         -- Dawson's integral.

# Fresnel sin and cos integrals.
fresnel       -- Fresnel sin and cos integrals.

# Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).
fresnel_zeros -- Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).

# Modified Fresnel positive integrals.
modfresnelp   -- Modified Fresnel positive integrals.

# Modified Fresnel negative integrals.
modfresnelm   -- Modified Fresnel negative integrals.

# Voigt profile.
voigt_profile -- Voigt profile.

# Compute nt complex zeros of error function erf(z).
erf_zeros      -- Compute nt complex zeros of error function erf(z).

# Compute nt complex zeros of cosine Fresnel integral C(z).
fresnelc_zeros -- Compute nt complex zeros of cosine Fresnel integral C(z).

# Compute nt complex zeros of sine Fresnel integral S(z).
fresnels_zeros -- Compute nt complex zeros of sine Fresnel integral S(z).

# Associated Legendre function of integer order and real degree.
lpmv     -- Associated Legendre function of integer order and real degree.

# Compute spherical harmonics.
sph_harm -- Compute spherical harmonics.
# Associated Legendre function of the first kind for complex arguments.
clpmn

# Legendre function of the first kind.
lpn

# Legendre function of the second kind.
lqn

# Sequence of associated Legendre functions of the first kind.
lpmn

# Sequence of associated Legendre functions of the second kind.
lqmn

# Ellipsoidal harmonic functions E^p_n(l).
ellip_harm

# Ellipsoidal harmonic functions F^p_n(l).
ellip_harm_2

# Ellipsoidal harmonic normalization constants gamma^p_n.
ellip_normal

# Compute the generalized (associated) Laguerre polynomial of degree n and order k.
assoc_laguerre

# Evaluate Legendre polynomial at a point.
eval_legendre

# Evaluate Chebyshev polynomial of the first kind at a point.
eval_chebyt

# Evaluate Chebyshev polynomial of the second kind at a point.
eval_chebyu

# Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a point.
eval_chebyc

# Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a point.
eval_chebys

# Evaluate Jacobi polynomial at a point.
eval_jacobi

# Evaluate Laguerre polynomial at a point.
eval_laguerre

# Evaluate generalized Laguerre polynomial at a point.
eval_genlaguerre

# Evaluate physicist's Hermite polynomial at a point.
eval_hermite

# Evaluate probabilist's (normalized) Hermite polynomial at a point.
eval_hermitenorm

# Evaluate Gegenbauer polynomial at a point.
eval_gegenbauer

# Evaluate shifted Legendre polynomial at a point.
eval_sh_legendre

# Evaluate shifted Chebyshev polynomial of the first kind at a point.
eval_sh_chebyt

# Evaluate shifted Chebyshev polynomial of the second kind at a point.
eval_sh_chebyu

# Evaluate shifted Jacobi polynomial at a point.
eval_sh_jacobi
.. autosummary::
   :toctree: generated/

   roots_legendre    -- Gauss-Legendre quadrature.
   roots_chebyt      -- Gauss-Chebyshev (first kind) quadrature.
   roots_chebyu      -- Gauss-Chebyshev (second kind) quadrature.
   roots_chebyc      -- Gauss-Chebyshev (first kind) quadrature.
   roots_chebys      -- Gauss-Chebyshev (second kind) quadrature.
   roots_jacobi      -- Gauss-Jacobi quadrature.
   roots_laguerre    -- Gauss-Laguerre quadrature.
   roots_genlaguerre -- Gauss-generalized Laguerre quadrature.
   roots_hermite     -- Gauss-Hermite (physicist's) quadrature.
   roots_hermitenorm -- Gauss-Hermite (statistician's) quadrature.
   roots_gegenbauer  -- Gauss-Gegenbauer quadrature.
   roots_sh_legendre -- Gauss-Legendre (shifted) quadrature.
   roots_sh_chebyt   -- Gauss-Chebyshev (first kind, shifted) quadrature.
   roots_sh_chebyu   -- Gauss-Chebyshev (second kind, shifted) quadrature.
   roots_sh_jacobi   -- Gauss-Jacobi (shifted) quadrature.

# 下面的函数依次返回`orthopoly1d`对象中的多项式系数，该对象类似于 `numpy.poly1d`。
# `orthopoly1d` 类还有一个 `weights` 属性，返回相应形式的高斯积分的根、权重和总权重。
# 这些以一个 `n x 3` 数组的形式返回，根在第一列，权重在第二列，总权重在最后一列。
# 注意，当进行算术运算时，`orthopoly1d` 对象会转换为 `~numpy.poly1d`，丢失原始正交多项式的信息。
# 自动摘要如下：

.. autosummary::
   :toctree: generated/

   legendre    -- Legendre polynomial.
   chebyt      -- Chebyshev polynomial of the first kind.
   chebyu      -- Chebyshev polynomial of the second kind.
   chebyc      -- Chebyshev polynomial of the first kind on :math:`[-2, 2]`.
   chebys      -- Chebyshev polynomial of the second kind on :math:`[-2, 2]`.
   jacobi      -- Jacobi polynomial.
   laguerre    -- Laguerre polynomial.
   genlaguerre -- Generalized (associated) Laguerre polynomial.
   hermite     -- Physicist's Hermite polynomial.
   hermitenorm -- Normalized (probabilist's) Hermite polynomial.
   gegenbauer  -- Gegenbauer (ultraspherical) polynomial.
   sh_legendre -- Shifted Legendre polynomial.
   sh_chebyt   -- Shifted Chebyshev polynomial of the first kind.
   sh_chebyu   -- Shifted Chebyshev polynomial of the second kind.
   sh_jacobi   -- Shifted Jacobi polynomial.

# 警告：
# 使用多项式系数计算高阶多项式（大约 `order > 20`）的值是数值不稳定的。
# 若要计算多项式值，应使用 `eval_*` 函数。
# Gauss hypergeometric function 2F1(a, b; c; z)
hyp2f1 -- Gauss hypergeometric function 2F1(a, b; c; z).

# Confluent hypergeometric function 1F1(a, b; x)
hyp1f1 -- Confluent hypergeometric function 1F1(a, b; x).

# Confluent hypergeometric function U(a, b, x) of the second kind
hyperu -- Confluent hypergeometric function U(a, b, x) of the second kind.

# Confluent hypergeometric limit function 0F1
hyp0f1 -- Confluent hypergeometric limit function 0F1.


# Parabolic cylinder function D
pbdv -- Parabolic cylinder function D.

# Parabolic cylinder function V
pbvv -- Parabolic cylinder function V.

# Parabolic cylinder function W
pbwa -- Parabolic cylinder function W.


# Parabolic cylinder functions Dv(x) and derivatives
pbdv_seq -- Parabolic cylinder functions Dv(x) and derivatives.

# Parabolic cylinder functions Vv(x) and derivatives
pbvv_seq -- Parabolic cylinder functions Vv(x) and derivatives.

# Parabolic cylinder functions Dn(z) and derivatives
pbdn_seq -- Parabolic cylinder functions Dn(z) and derivatives.


# Characteristic value of even Mathieu functions
mathieu_a -- Characteristic value of even Mathieu functions.

# Characteristic value of odd Mathieu functions
mathieu_b -- Characteristic value of odd Mathieu functions.


# Fourier coefficients for even Mathieu and modified Mathieu functions
mathieu_even_coef -- Fourier coefficients for even Mathieu and modified Mathieu functions.

# Fourier coefficients for odd Mathieu and modified Mathieu functions
mathieu_odd_coef  -- Fourier coefficients for odd Mathieu and modified Mathieu functions.


# Even Mathieu function and its derivative
mathieu_cem     -- Even Mathieu function and its derivative.

# Odd Mathieu function and its derivative
mathieu_sem     -- Odd Mathieu function and its derivative.

# Even modified Mathieu function of the first kind and its derivative
mathieu_modcem1 -- Even modified Mathieu function of the first kind and its derivative.

# Even modified Mathieu function of the second kind and its derivative
mathieu_modcem2 -- Even modified Mathieu function of the second kind and its derivative.

# Odd modified Mathieu function of the first kind and its derivative
mathieu_modsem1 -- Odd modified Mathieu function of the first kind and its derivative.

# Odd modified Mathieu function of the second kind and its derivative
mathieu_modsem2 -- Odd modified Mathieu function of the second kind and its derivative.


# Prolate spheroidal angular function of the first kind and its derivative
pro_ang1   -- Prolate spheroidal angular function of the first kind and its derivative.

# Prolate spheroidal radial function of the first kind and its derivative
pro_rad1   -- Prolate spheroidal radial function of the first kind and its derivative.

# Prolate spheroidal radial function of the second kind and its derivative
pro_rad2   -- Prolate spheroidal radial function of the second kind and its derivative.

# Oblate spheroidal angular function of the first kind and its derivative
obl_ang1   -- Oblate spheroidal angular function of the first kind and its derivative.

# Oblate spheroidal radial function of the first kind and its derivative
obl_rad1   -- Oblate spheroidal radial function of the first kind and its derivative.

# Oblate spheroidal radial function of the second kind and its derivative
obl_rad2   -- Oblate spheroidal radial function of the second kind and its derivative.

# Characteristic value of prolate spheroidal function
pro_cv     -- Characteristic value of prolate spheroidal function.

# Characteristic value of oblate spheroidal function
obl_cv     -- Characteristic value of oblate spheroidal function.

# Characteristic values for prolate spheroidal wave functions
pro_cv_seq -- Characteristic values for prolate spheroidal wave functions.

# Characteristic values for oblate spheroidal wave functions
obl_cv_seq -- Characteristic values for oblate spheroidal wave functions.
# The following functions require pre-computed characteristic value:

# Importing specific functions related to prolate and oblate spheroidal angular and radial functions with precomputed characteristic values.
.. autosummary::
   :toctree: generated/

   pro_ang1_cv -- Prolate spheroidal angular function pro_ang1 for precomputed characteristic value.
   pro_rad1_cv -- Prolate spheroidal radial function pro_rad1 for precomputed characteristic value.
   pro_rad2_cv -- Prolate spheroidal radial function pro_rad2 for precomputed characteristic value.
   obl_ang1_cv -- Oblate spheroidal angular function obl_ang1 for precomputed characteristic value.
   obl_rad1_cv -- Oblate spheroidal radial function obl_rad1 for precomputed characteristic value.
   obl_rad2_cv -- Oblate spheroidal radial function obl_rad2 for precomputed characteristic value.

# Kelvin functions
# Importing various Kelvin functions for complex numbers and their zeros computation.
----------------
.. autosummary::
   :toctree: generated/

   kelvin       -- Kelvin functions as complex numbers.
   kelvin_zeros -- Compute nt zeros of all Kelvin functions.
   ber          -- Kelvin function ber.
   bei          -- Kelvin function bei
   berp         -- Derivative of the Kelvin function `ber`.
   beip         -- Derivative of the Kelvin function `bei`.
   ker          -- Kelvin function ker.
   kei          -- Kelvin function ker.
   kerp         -- Derivative of the Kelvin function ker.
   keip         -- Derivative of the Kelvin function kei.

# The following functions do not accept NumPy arrays (they are not
# universal functions):

# Importing functions that do not accept NumPy arrays, specifying zeros computation for each Kelvin function.
.. autosummary::
   :toctree: generated/

   ber_zeros  -- Compute nt zeros of the Kelvin function ber(x).
   bei_zeros  -- Compute nt zeros of the Kelvin function bei(x).
   berp_zeros -- Compute nt zeros of the Kelvin function ber'(x).
   beip_zeros -- Compute nt zeros of the Kelvin function bei'(x).
   ker_zeros  -- Compute nt zeros of the Kelvin function ker(x).
   kei_zeros  -- Compute nt zeros of the Kelvin function kei(x).
   kerp_zeros -- Compute nt zeros of the Kelvin function ker'(x).
   keip_zeros -- Compute nt zeros of the Kelvin function kei'(x).

# Combinatorics
# Importing functions related to combinatorics: combination, permutation, and Stirling numbers of the second kind.
-------------

.. autosummary::
   :toctree: generated/

   comb -- The number of combinations of N things taken k at a time.
   perm -- Permutations of N things taken k at a time, i.e., k-permutations of N.
   stirling2 -- Stirling numbers of the second kind.

# Lambert W and related functions
# Importing Lambert W function and Wright Omega function.
-------------------------------

.. autosummary::
   :toctree: generated/

   lambertw    -- Lambert W function.
   wrightomega -- Wright Omega function.

# Other special functions
# No additional functions are listed here.
-----------------------
import os  # 导入标准库 os，用于操作操作系统相关功能
import warnings  # 导入标准库 warnings，用于处理警告信息


def _load_libsf_error_state():
    """Load libsf_error_state.dll shared library on Windows

    libsf_error_state manages shared state used by
    ``scipy.special.seterr`` and ``scipy.special.geterr`` so that these
    can work consistently between special functions provided by different
    extension modules. This shared library is installed in scipy/special
    alongside this __init__.py file. Due to lack of rpath support, Windows
    cannot find shared libraries installed within wheels. To circumvent this,
    we pre-load ``lib_sf_error_state.dll`` when on Windows.

    The logic for this function was borrowed from the function ``make_init``
    in `scipy/tools/openblas_support.py`:
    https://github.com/scipy/scipy/blob/bb92c8014e21052e7dde67a76b28214dd1dcb94a/tools/openblas_support.py#L239-L274
    """
    # 检查操作系统类型是否为 Windows
    if os.name == "nt":
        # 尝试导入 WinDLL 类型，用于加载动态链接库
        try:
            from ctypes import WinDLL
            # 获取当前脚本文件的目录路径
            basedir = os.path.dirname(__file__)
        # 如果出现异常则忽略，不做处理
        except:  # noqa: E722
            pass
        # 如果成功导入 WinDLL 类型
        else:
            # 构建动态链接库文件的完整路径
            dll_path = os.path.join(basedir, "libsf_error_state.dll")
            # 如果该路径下的动态链接库文件存在
            if os.path.exists(dll_path):
                # 使用 WinDLL 加载动态链接库文件
                WinDLL(dll_path)
# 载入用于库错误状态的函数
_load_libsf_error_state()

# 导入特殊函数警告和错误类
from ._sf_error import SpecialFunctionWarning, SpecialFunctionError

# 导入 _ufuncs 模块及其所有内容
from . import _ufuncs
from ._ufuncs import *

# 从 _support_alternative_backends 导入一些函数定义，以添加数组 API 支持
from ._support_alternative_backends import (
    log_ndtr, ndtr, ndtri, erf, erfc, i0, i0e, i1, i1e, gammaln,
    gammainc, gammaincc, logit, expit, entr, rel_entr, xlogy,
    chdtr, chdtrc, betainc, betaincc, stdtr)

# 导入 _basic 模块及其所有内容
from . import _basic
from ._basic import *

# 导入 _logsumexp 模块的 logsumexp、softmax 和 log_softmax 函数
from ._logsumexp import logsumexp, softmax, log_softmax

# 导入 _orthogonal 模块及其所有内容
from . import _orthogonal
from ._orthogonal import *

# 导入 _spfun_stats 模块的 multigammaln 函数
from ._spfun_stats import multigammaln

# 导入 _ellip_harm 模块的 ellip_harm、ellip_harm_2 和 ellip_normal 函数
from ._ellip_harm import (
    ellip_harm,
    ellip_harm_2,
    ellip_normal
)

# 导入 _lambertw 模块的 lambertw 函数
from ._lambertw import lambertw

# 导入 _spherical_bessel 模块的 spherical_jn、spherical_yn、spherical_in 和 spherical_kn 函数
from ._spherical_bessel import (
    spherical_jn,
    spherical_yn,
    spherical_in,
    spherical_kn
)

# 弃用的命名空间，将在 v2.0.0 中移除
from . import add_newdocs, basic, orthogonal, specfun, sf_error, spfun_stats

# 将 _ufuncs、_basic 和 _orthogonal 模块的 __all__ 属性合并为 __all__ 属性
__all__ = _ufuncs.__all__ + _basic.__all__ + _orthogonal.__all__
__all__ += [
    'SpecialFunctionWarning',
    'SpecialFunctionError',
    'logsumexp',
    'softmax',
    'log_softmax',
    'multigammaln',
    'ellip_harm',
    'ellip_harm_2',
    'ellip_normal',
    'lambertw',
    'spherical_jn',
    'spherical_yn',
    'spherical_in',
    'spherical_kn',
]

# 从 scipy._lib._testutils 模块导入 PytestTester 类并用 __name__ 初始化 test 对象
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)

# 删除 PytestTester 类的引用，清理命名空间
del PytestTester

# 弃用警告消息字符串
_depr_msg = ('\nThis function was deprecated in SciPy 1.12.0, and will be '
             'removed in SciPy 1.14.0.  Use scipy.special.{} instead.')

# 定义 btdtr 函数，引发弃用警告，然后调用 _ufuncs 模块的 btdtr 函数
def btdtr(*args, **kwargs):  # type: ignore [no-redef]
    warnings.warn(_depr_msg.format('betainc'), category=DeprecationWarning,
                  stacklevel=2)
    return _ufuncs.btdtr(*args, **kwargs)

# 设置 btdtr 函数的文档字符串为 _ufuncs 模块的 btdtr 函数的文档字符串
btdtr.__doc__ = _ufuncs.btdtr.__doc__  # type: ignore [misc]

# 定义 btdtri 函数，引发弃用警告，然后调用 _ufuncs 模块的 btdtri 函数
def btdtri(*args, **kwargs):  # type: ignore [no-redef]
    warnings.warn(_depr_msg.format('betaincinv'), category=DeprecationWarning,
                  stacklevel=2)
    return _ufuncs.btdtri(*args, **kwargs)

# 设置 btdtri 函数的文档字符串为 _ufuncs 模块的 btdtri 函数的文档字符串
btdtri.__doc__ = _ufuncs.btdtri.__doc__  # type: ignore [misc]

# 定义 _get_include 函数，返回当前文件所在目录的路径
def _get_include():
    """This function is for development purposes only.

    This function could disappear or its behavior could change at any time.
    """
    import os
    return os.path.dirname(__file__)
```