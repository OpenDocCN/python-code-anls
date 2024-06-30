# `D:\src\scipysrc\scipy\scipy\odr\__init__.py`

```
"""
=================================================
Orthogonal distance regression (:mod:`scipy.odr`)
=================================================

.. currentmodule:: scipy.odr

Package Content
===============

.. autosummary::
   :toctree: generated/

   Data          -- The data to fit.
   RealData      -- Data with weights as actual std. dev.s and/or covariances.
   Model         -- Stores information about the function to be fit.
   ODR           -- Gathers all info & manages the main fitting routine.
   Output        -- Result from the fit.
   odr           -- Low-level function for ODR.

   OdrWarning    -- Warning about potential problems when running ODR.
   OdrError      -- Error exception.
   OdrStop       -- Stop exception.

   polynomial    -- Factory function for a general polynomial model.
   exponential   -- Exponential model
   multilinear   -- Arbitrary-dimensional linear model
   unilinear     -- Univariate linear model
   quadratic     -- Quadratic model

Usage information
=================

Introduction
------------

Why Orthogonal Distance Regression (ODR)? Sometimes one has
measurement errors in the explanatory (a.k.a., "independent")
variable(s), not just the response (a.k.a., "dependent") variable(s).
Ordinary Least Squares (OLS) fitting procedures treat the data for
explanatory variables as fixed, i.e., not subject to error of any kind.
Furthermore, OLS procedures require that the response variables be an
explicit function of the explanatory variables; sometimes making the
equation explicit is impractical and/or introduces errors.  ODR can
handle both of these cases with ease, and can even reduce to the OLS
case if that is sufficient for the problem.

ODRPACK is a FORTRAN-77 library for performing ODR with possibly
non-linear fitting functions. It uses a modified trust-region
Levenberg-Marquardt-type algorithm [1]_ to estimate the function
parameters.  The fitting functions are provided by Python functions
operating on NumPy arrays. The required derivatives may be provided
by Python functions as well, or may be estimated numerically. ODRPACK
can do explicit or implicit ODR fits, or it can do OLS. Input and
output variables may be multidimensional. Weights can be provided to
account for different variances of the observations, and even
covariances between dimensions of the variables.

The `scipy.odr` package offers an object-oriented interface to
ODRPACK, in addition to the low-level `odr` function.

Additional background information about ODRPACK can be found in the
`ODRPACK User's Guide
<https://docs.scipy.org/doc/external/odrpack_guide.pdf>`_, reading
which is recommended.

Basic usage
-----------
"""
# 定义一个用于拟合的函数。
def f(B, x):
    '''Linear function y = m*x + b'''
    # B 是参数向量。
    # x 是当前 x 值的数组。
    # x 的格式与传递给 Data 或 RealData 的 x 相同。
    #
    # 返回一个与传递给 Data 或 RealData 的 y 相同格式的数组。
    return B[0]*x + B[1]

# 创建一个 Model 对象，用于表示要拟合的模型是 f 函数。
linear = Model(f)

# 创建一个 Data 或 RealData 实例，用于存储数据和权重信息。
mydata = Data(x, y, wd=1./power(sx,2), we=1./power(sy,2))
# 或者，当实际协方差已知时，使用 RealData。
# mydata = RealData(x, y, sx=sx, sy=sy)

# 使用给定的数据、模型和初始参数估计实例化 ODR 对象。
myodr = ODR(mydata, linear, beta0=[1., 2.])

# 运行拟合过程。
myoutput = myodr.run()

# 打印拟合结果。
myoutput.pprint()
```