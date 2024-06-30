# `D:\src\scipysrc\scipy\scipy\stats\tests\test_continuous_fit_censored.py`

```
# 导入所需的库
import numpy as np
from numpy.testing import assert_allclose

# 导入用于优化的函数
from scipy.optimize import fmin
# 导入需要拟合的概率分布
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
                         gumbel_l, gumbel_r, invgauss, invweibull, laplace,
                         logistic, lognorm, nct, ncx2, norm, weibull_max,
                         weibull_min)

# 定义一个优化器函数，用于提高参数估计的精确度
def optimizer(func, x0, args=(), disp=0):
    return fmin(func, x0, args=args, disp=disp, xtol=1e-12, ftol=1e-12)

# 测试拟合 beta 分布的形状参数到区间截断数据
def test_beta():
    """
    Test fitting beta shape parameters to interval-censored data.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(0.10, 0.50, 0.75, 0.80),
    +                    right=c(0.20, 0.55, 0.90, 0.95))
    > result = fitdistcens(data, 'beta', control=list(reltol=1e-14))

    > result
    Fitting of the distribution ' beta ' on censored data by maximum likelihood
    Parameters:
           estimate
    shape1 1.419941
    shape2 1.027066
    > result$sd
       shape1    shape2
    0.9914177 0.6866565
    """
    # 创建包含区间截断数据的 CensoredData 对象
    data = CensoredData(interval=[[0.10, 0.20],
                                  [0.50, 0.55],
                                  [0.75, 0.90],
                                  [0.80, 0.95]])

    # 对于此测试，仅拟合形状参数；位置和尺度被固定
    # 使用定义的 optimizer 函数优化拟合过程
    a, b, loc, scale = beta.fit(data, floc=0, fscale=1, optimizer=optimizer)

    # 断言拟合的参数值与预期值接近
    assert_allclose(a, 1.419941, rtol=5e-6)
    assert_allclose(b, 1.027066, rtol=5e-6)
    assert loc == 0
    assert scale == 1


# 测试拟合 Cauchy 分布到右截断数据
def test_cauchy_right_censored():
    """
    Test fitting the Cauchy distribution to right-censored data.

    Calculation in R, with two values not censored [1, 10] and
    one right-censored value [30].

    > library(fitdistrplus)
    > data <- data.frame(left=c(1, 10, 30), right=c(1, 10, NA))
    > result = fitdistcens(data, 'cauchy', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' cauchy ' on censored data by maximum
    likelihood
    Parameters:
             estimate
    location 7.100001
    scale    7.455866
    """
    # 创建包含右截断数据的 CensoredData 对象
    data = CensoredData(uncensored=[1, 10], right=[30])
    
    # 使用定义的 optimizer 函数优化 Cauchy 分布的拟合过程
    loc, scale = cauchy.fit(data, optimizer=optimizer)
    
    # 断言拟合的位置和尺度参数值与预期值接近
    assert_allclose(loc, 7.10001, rtol=5e-6)
    assert_allclose(scale, 7.455866, rtol=5e-6)


# 测试拟合 Cauchy 分布到混合截断数据
def test_cauchy_mixed():
    """
    Test fitting the Cauchy distribution to data with mixed censoring.

    Calculation in R, with:
    * two values not censored [1, 10],
    * one left-censored [1],
    * one right-censored [30], and
    * one interval-censored [[4, 8]].

    > library(fitdistrplus)
    > data <- data.frame(left=c(NA, 1, 4, 10, 30), right=c(1, 1, 8, 10, NA))
    > result = fitdistcens(data, 'cauchy', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' cauchy ' on censored data by maximum
    likelihood
    """
    # 创建包含混合截断数据的 CensoredData 对象
    data = CensoredData(left=[1], uncensored=[1, 10], right=[30], interval=[[4, 8]])
    
    # 使用定义的 optimizer 函数优化 Cauchy 分布的拟合过程
    loc, scale = cauchy.fit(data, optimizer=optimizer)
    Parameters:
             estimate
    location 4.605150
    scale    5.900852
    """
    # 创建一个包含被截断数据的对象，其中未截断的数据是[1, 10]，左截断数据是[1]，右截断数据是[30]，区间是[[4, 8]]
    data = CensoredData(uncensored=[1, 10], left=[1], right=[30],
                        interval=[[4, 8]])
    # 使用优化器拟合数据，返回 Cauchy 分布的位置参数 loc 和尺度参数 scale
    loc, scale = cauchy.fit(data, optimizer=optimizer)
    # 断言位置参数 loc 等于预期值 4.605150，允许的相对误差是 5e-6
    assert_allclose(loc, 4.605150, rtol=5e-6)
    # 断言尺度参数 scale 等于预期值 5.900852，允许的相对误差是 5e-6
    assert_allclose(scale, 5.900852, rtol=5e-6)
# 定义一个用于测试混合数据的卡方分布拟合的函数
def test_chi2_mixed():
    """
    Test fitting just the shape parameter (df) of chi2 to mixed data.

    Calculation in R, with:
    * two values not censored [1, 10],
    * one left-censored [1],
    * one right-censored [30], and
    * one interval-censored [[4, 8]].

    > library(fitdistrplus)
    > data <- data.frame(left=c(NA, 1, 4, 10, 30), right=c(1, 1, 8, 10, NA))
    > result = fitdistcens(data, 'chisq', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' chisq ' on censored data by maximum
    likelihood
    Parameters:
             estimate
    df 5.060329
    """
    # 创建一个混合数据对象，包含未被审查的值、左审查的值、右审查的值和区间审查的值
    data = CensoredData(uncensored=[1, 10], left=[1], right=[30],
                        interval=[[4, 8]])
    # 使用卡方分布拟合数据，返回拟合参数 df（自由度）、loc（位置参数）、scale（尺度参数）
    df, loc, scale = chi2.fit(data, floc=0, fscale=1, optimizer=optimizer)
    # 断言 df 的值接近于预期值 5.060329，允许的相对误差是 5e-6
    assert_allclose(df, 5.060329, rtol=5e-6)
    # 断言 loc 的值为 0
    assert loc == 0
    # 断言 scale 的值为 1
    assert scale == 1


# 定义一个用于测试指数分布右审查数据拟合的函数
def test_expon_right_censored():
    """
    For the exponential distribution with loc=0, the exact solution for
    fitting n uncensored points x[0]...x[n-1] and m right-censored points
    x[n]..x[n+m-1] is

        scale = sum(x)/n

    That is, divide the sum of all the values (not censored and
    right-censored) by the number of uncensored values.  (See, for example,
    https://en.wikipedia.org/wiki/Censoring_(statistics)#Likelihood.)

    The second derivative of the log-likelihood function is

        n/scale**2 - 2*sum(x)/scale**3

    from which the estimate of the standard error can be computed.

    -----

    Calculation in R, for reference only. The R results are not
    used in the test.

    > library(fitdistrplus)
    > dexps <- function(x, scale) {
    +     return(dexp(x, 1/scale))
    + }
    > pexps <- function(q, scale) {
    +     return(pexp(q, 1/scale))
    + }
    > left <- c(1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15,
    +                                     16, 16, 20, 20, 21, 22)
    > right <- c(1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15,
    +                                     NA, NA, NA, NA, NA, NA)
    > result = fitdistcens(data, 'exps', start=list(scale=mean(data$left)),
    +                      control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' exps ' on censored data by maximum likelihood
    Parameters:
          estimate
    scale    19.85
    > result$sd
       scale
    6.277119
    """
    # 这些数据包含了 10 个未被审查的值和 6 个右审查的值。
    obs = [1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15, 16, 16, 20, 20, 21, 22]
    # 创建一个布尔列表，指示每个观察值是否被审查
    cens = [False]*10 + [True]*6
    # 使用右审查的数据创建一个 CensoredData 对象
    data = CensoredData.right_censored(obs, cens)

    # 使用指数分布拟合数据，返回拟合参数 loc（位置参数）、scale（尺度参数）
    loc, scale = expon.fit(data, floc=0, optimizer=optimizer)

    # 断言 loc 的值为 0
    assert loc == 0
    # 使用分析解来计算期望值。这是未被审查的值和右审查的值的总和除以未被审查值的数量。
    n = len(data) - data.num_censored()
    total = data._uncensored.sum() + data._right.sum()
    expected = total / n
    # 断言 scale 的值接近于期望值，允许的绝对误差是 1e-8
    assert_allclose(scale, expected, 1e-8)
# 定义一个测试函数，用于拟合具有右截尾值的 gamma 分布。
def test_gamma_right_censored():
    """
    Fit gamma shape and scale to data with one right-censored value.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, 25.0),
    +                    right=c(2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, NA))
    > result = fitdistcens(data, 'gamma', start=list(shape=1, scale=10),
    +                      control=list(reltol=1e-13))
    > result
    Fitting of the distribution ' gamma ' on censored data by maximum
      likelihood
    Parameters:
          estimate
    shape 1.447623
    scale 8.360197
    > result$sd
        shape     scale
    0.7053086 5.1016531
    """

    # 创建一个右截尾数据对象，最后一个值是右截尾。
    x = CensoredData.right_censored([2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0,
                                     25.0],
                                    [0]*7 + [1])

    # 使用 gamma 分布拟合数据，返回参数 shape、loc 和 scale
    a, loc, scale = gamma.fit(x, floc=0, optimizer=optimizer)

    # 断言拟合的 shape 参数接近给定值 1.447623，相对误差不超过 5e-6
    assert_allclose(a, 1.447623, rtol=5e-6)
    # 断言 loc 参数为 0
    assert loc == 0
    # 断言拟合的 scale 参数接近给定值 8.360197，相对误差不超过 5e-6
    assert_allclose(scale, 8.360197, rtol=5e-6)


# 定义一个测试函数，用于拟合 gumbel 分布。
def test_gumbel():
    """
    Fit gumbel_l and gumbel_r to censored data.

    This R calculation should match gumbel_r.

    > library(evd)
    > library(fitdistrplus)
    > data = data.frame(left=c(0, 2, 3, 9, 10, 10),
    +                   right=c(1, 2, 3, 9, NA, NA))
    > result = fitdistcens(data, 'gumbel',
    +                      control=list(reltol=1e-14),
    +                      start=list(loc=4, scale=5))
    > result
    Fitting of the distribution ' gumbel ' on censored data by maximum
    likelihood
    Parameters:
          estimate
    loc   4.487853
    scale 4.843640
    """

    # 第一个值是区间截尾，最后两个值是右截尾。
    uncensored = np.array([2, 3, 9])
    right = np.array([10, 10])
    interval = np.array([[0, 1]])
    # 创建一个包含截尾数据的对象
    data = CensoredData(uncensored, right=right, interval=interval)

    # 使用 gumbel_r 分布拟合数据，返回参数 loc 和 scale
    loc, scale = gumbel_r.fit(data, optimizer=optimizer)
    # 断言拟合的 loc 参数接近给定值 4.487853，相对误差不超过 5e-6
    assert_allclose(loc, 4.487853, rtol=5e-6)
    # 断言拟合的 scale 参数接近给定值 4.843640，相对误差不超过 5e-6
    assert_allclose(scale, 4.843640, rtol=5e-6)

    # 对数据取负并反转区间，使用 gumbel_l 分布进行测试。
    data2 = CensoredData(-uncensored, left=-right,
                         interval=-interval[:, ::-1])
    # 拟合数据2的 gumbel_l 分布应该得到与上述相同的结果，但 loc 取负。
    loc2, scale2 = gumbel_l.fit(data2, optimizer=optimizer)
    # 断言拟合的 loc2 参数接近给定值 -4.487853，相对误差不超过 5e-6
    assert_allclose(loc2, -4.487853, rtol=5e-6)
    # 断言拟合的 scale2 参数接近给定值 4.843640，相对误差不超过 5e-6
    assert_allclose(scale2, 4.843640, rtol=5e-6)


# 定义一个测试函数，用于拟合 invgauss 分布的形状参数。
def test_invgauss():
    """
    Fit just the shape parameter of invgauss to data with one value
    left-censored and one value right-censored.

    Calculation in R; using a fixed dispersion parameter amounts to fixing
    the scale to be 1.

    > library(statmod)
    > library(fitdistrplus)
    > left <- c(NA, 0.4813096, 0.5571880, 0.5132463, 0.3801414, 0.5904386,
    +           0.4822340, 0.3478597, 3, 0.7191797, 1.5810902, 0.4442299)
    """

    # 此处省略了具体的 R 计算步骤和结果，因为对于 invgauss 分布，只需要拟合形状参数即可。
    # 在此不需要进行具体数值的断言测试。
    # 左边观测数据
    x = [0.4813096, 0.5571880, 0.5132463, 0.3801414,
         0.5904386, 0.4822340, 0.3478597, 0.7191797,
         1.5810902, 0.4442299]
    # 创建包含左右截断数据的 CensoredData 对象
    data = CensoredData(uncensored=x, left=[0.15], right=[3])
    
    # 仅拟合形状参数（mu），loc 设置为 0，scale 设置为 1，并返回拟合的 mu、loc 和 scale
    mu, loc, scale = invgauss.fit(data, floc=0, fscale=1, optimizer=optimizer)
    
    # 使用 assert_allclose 检查拟合的 mu 是否接近于预期值 0.853469，相对容差为 5e-5
    assert_allclose(mu, 0.853469, rtol=5e-5)
    # 断言 loc 应为 0
    assert loc == 0
    # 断言 scale 应为 1
    assert scale == 1
    
    # 拟合形状和比例尺参数（mu 和 scale），loc 设置为 0，并返回拟合的 mu、loc 和 scale
    mu, loc, scale = invgauss.fit(data, floc=0, optimizer=optimizer)
    
    # 使用 assert_allclose 检查拟合的 mu 是否接近于预期值 1.066716，相对容差为 5e-5
    assert_allclose(mu, 1.066716, rtol=5e-5)
    # 断言 loc 应为 0
    assert loc == 0
    # 使用 assert_allclose 检查拟合的 scale 是否接近于预期值 0.8155701，相对容差为 5e-5
    assert_allclose(scale, 0.8155701, rtol=5e-5)
# 定义一个函数用于测试逆威布尔分布拟合带有截尾数据的情况
def test_invweibull():
    """
    Fit invweibull to censored data.

    Here is the calculation in R.  The 'frechet' distribution from the evd
    package matches SciPy's invweibull distribution.  The `loc` parameter
    is fixed at 0.

    > library(evd)
    > library(fitdistrplus)
    > data = data.frame(left=c(0, 2, 3, 9, 10, 10),
    +                   right=c(1, 2, 3, 9, NA, NA))
    > result = fitdistcens(data, 'frechet',
    +                      control=list(reltol=1e-14),
    +                      start=list(loc=4, scale=5))
    > result
    Fitting of the distribution ' frechet ' on censored data by maximum
    likelihood
    Parameters:
           estimate
    scale 2.7902200
    shape 0.6379845
    Fixed parameters:
        value
    loc     0
    """
    # 在R数据中，第一个值是区间截尾，最后两个是右截尾，其余未被截尾。
    # 创建一个CensoredData对象，包含未截尾数据和右截尾数据的信息
    data = CensoredData(uncensored=[2, 3, 9], right=[10, 10],
                        interval=[[0, 1]])
    # 使用invweibull.fit函数拟合数据，固定floc参数为0，并返回拟合结果中的参数
    c, loc, scale = invweibull.fit(data, floc=0, optimizer=optimizer)
    # 断言拟合出的形状参数c接近于0.6379845，允许的相对误差为5e-6
    assert_allclose(c, 0.6379845, rtol=5e-6)
    # 断言拟合出的位置参数loc等于0
    assert loc == 0
    # 断言拟合出的尺度参数scale接近于2.7902200，允许的相对误差为5e-6
    assert_allclose(scale, 2.7902200, rtol=5e-6)


def test_laplace():
    """
    Fit the Laplace distribution to left- and right-censored data.

    Calculation in R:

    > library(fitdistrplus)
    > dlaplace <- function(x, location=0, scale=1) {
    +     return(0.5*exp(-abs((x - location)/scale))/scale)
    + }
    > plaplace <- function(q, location=0, scale=1) {
    +     z <- (q - location)/scale
    +     s <- sign(z)
    +     f <- -s*0.5*exp(-abs(z)) + (s+1)/2
    +     return(f)
    + }
    > left <- c(NA, -41.564, 50.0, 15.7384, 50.0, 10.0452, -2.0684,
    +           -19.5399, 50.0,   9.0005, 27.1227, 4.3113, -3.7372,
    +           25.3111, 14.7987,  34.0887,  50.0, 42.8496, 18.5862,
    +           32.8921, 9.0448, -27.4591, NA, 19.5083, -9.7199)
    > right <- c(-50.0, -41.564,  NA, 15.7384, NA, 10.0452, -2.0684,
    +            -19.5399, NA, 9.0005, 27.1227, 4.3113, -3.7372,
    +            25.3111, 14.7987, 34.0887, NA,  42.8496, 18.5862,
    +            32.8921, 9.0448, -27.4591, -50.0, 19.5083, -9.7199)
    > data <- data.frame(left=left, right=right)
    > result <- fitdistcens(data, 'laplace', start=list(location=10, scale=10),
    +                       control=list(reltol=1e-13))
    > result
    Fitting of the distribution ' laplace ' on censored data by maximum
      likelihood
    Parameters:
             estimate
    location 14.79870
    scale    30.93601
    > result$sd
         location     scale
    0.1758864 7.0972125
    """
    # 值-50是左截尾，值50是右截尾。
    # 创建一个NumPy数组，包含左右截尾数据
    obs = np.array([-50.0, -41.564, 50.0, 15.7384, 50.0, 10.0452, -2.0684,
                    -19.5399, 50.0, 9.0005, 27.1227, 4.3113, -3.7372,
                    25.3111, 14.7987, 34.0887, 50.0, 42.8496, 18.5862,
                    32.8921, 9.0448, -27.4591, -50.0, 19.5083, -9.7199])
    # 从观测数据中选择所有不等于 -50.0 和 50 的值，存入 x 变量
    x = obs[(obs != -50.0) & (obs != 50)]
    # 从观测数据中选择所有等于 -50.0 的值，存入 left 变量
    left = obs[obs == -50.0]
    # 从观测数据中选择所有等于 50 的值，存入 right 变量
    right = obs[obs == 50.0]
    # 使用 CensoredData 类创建一个包含未被截断数据（uncensored）、左截断数据（left）和右截断数据（right）的对象，存入 data 变量
    data = CensoredData(uncensored=x, left=left, right=right)
    # 使用拉普拉斯分布拟合数据，估计位置参数 loc 和尺度参数 scale，使用给定的优化器
    loc, scale = laplace.fit(data, loc=10, scale=10, optimizer=optimizer)
    # 断言 loc 的值接近于 14.79870，相对误差容忍度为 5e-6
    assert_allclose(loc, 14.79870, rtol=5e-6)
    # 断言 scale 的值接近于 30.93601，相对误差容忍度为 5e-6
    assert_allclose(scale, 30.93601, rtol=5e-6)
# 定义一个用于测试 logistic 分布拟合的函数
def test_logistic():
    """
    Fit the logistic distribution to left-censored data.

    Calculation in R:
    > library(fitdistrplus)
    > left = c(13.5401, 37.4235, 11.906 , 13.998 ,  NA    ,  0.4023,  NA    ,
    +          10.9044, 21.0629,  9.6985,  NA    , 12.9016, 39.164 , 34.6396,
    +          NA    , 20.3665, 16.5889, 18.0952, 45.3818, 35.3306,  8.4949,
    +          3.4041,  NA    ,  7.2828, 37.1265,  6.5969, 17.6868, 17.4977,
    +          16.3391, 36.0541)
    > right = c(13.5401, 37.4235, 11.906 , 13.998 ,  0.    ,  0.4023,  0.    ,
    +           10.9044, 21.0629,  9.6985,  0.    , 12.9016, 39.164 , 34.6396,
    +           0.    , 20.3665, 16.5889, 18.0952, 45.3818, 35.3306,  8.4949,
    +           3.4041,  0.    ,  7.2828, 37.1265,  6.5969, 17.6868, 17.4977,
    +           16.3391, 36.0541)
    > data = data.frame(left=left, right=right)
    > result = fitdistcens(data, 'logis', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' logis ' on censored data by maximum
      likelihood
    Parameters:
              estimate
    location 14.633459
    scale     9.232736
    > result$sd
    location    scale
    2.931505 1.546879
    """
    # Values that are zero are left-censored; the true values are less than 0.
    # 定义左截尾数据，其中零表示数据左截尾，实际值小于0
    x = np.array([13.5401, 37.4235, 11.906, 13.998, 0.0, 0.4023, 0.0, 10.9044,
                  21.0629, 9.6985, 0.0, 12.9016, 39.164, 34.6396, 0.0, 20.3665,
                  16.5889, 18.0952, 45.3818, 35.3306, 8.4949, 3.4041, 0.0,
                  7.2828, 37.1265, 6.5969, 17.6868, 17.4977, 16.3391,
                  36.0541])
    # 创建左截尾数据对象
    data = CensoredData.left_censored(x, censored=(x == 0))
    # 拟合 logistic 分布，返回位置参数 loc 和尺度参数 scale
    loc, scale = logistic.fit(data, optimizer=optimizer)
    # 断言位置参数 loc 接近 14.633459，相对误差不超过 5e-7
    assert_allclose(loc, 14.633459, rtol=5e-7)
    # 断言尺度参数 scale 接近 9.232736，相对误差不超过 5e-6
    assert_allclose(scale, 9.232736, rtol=5e-6)


def test_lognorm():
    """
    Ref: https://math.montana.edu/jobo/st528/documents/relc.pdf

    The data is the locomotive control time to failure example that starts
    on page 8.  That's the 8th page in the PDF; the page number shown in
    the text is 270).
    The document includes SAS output for the data.
    """
    # These are the uncensored measurements.  There are also 59 right-censored
    # measurements where the lower bound is 135.
    # 定义未截尾的测量数据，同时包括 59 个右截尾数据，下限为 135
    miles_to_fail = [22.5, 37.5, 46.0, 48.5, 51.5, 53.0, 54.5, 57.5, 66.5,
                     68.0, 69.5, 76.5, 77.0, 78.5, 80.0, 81.5, 82.0, 83.0,
                     84.0, 91.5, 93.5, 102.5, 107.0, 108.5, 112.5, 113.5,
                     116.0, 117.0, 118.5, 119.0, 120.0, 122.5, 123.0, 127.5,
                     131.0, 132.5, 134.0]
    # 创建右截尾数据对象
    data = CensoredData.right_censored(miles_to_fail + [135]*59,
                                       [0]*len(miles_to_fail) + [1]*59)
    # 拟合 lognorm 分布，返回标准差 sigma，位置参数 loc 和尺度参数 scale
    sigma, loc, scale = lognorm.fit(data, floc=0)
    # 断言位置参数 loc 为 0
    assert loc == 0
    # 将 lognorm 参数转换为底层正态分布的 mu 和 sigma
    mu = np.log(scale)
    # 断言检查：验证计算出的 mu 是否接近预期值 5.1169，相对容差为 5e-4
    assert_allclose(mu, 5.1169, rtol=5e-4)
    # 断言检查：验证计算出的 sigma 是否接近预期值 0.7055，相对容差为 5e-3
    assert_allclose(sigma, 0.7055, rtol=5e-3)
# 测试拟合非中心 t 分布到被截尾数据的情况

def test_nct():
    """
    Test fitting the noncentral t distribution to censored data.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(1, 2, 3, 5, 8, 10, 25, 25),
    +                    right=c(1, 2, 3, 5, 8, 10, NA, NA))
    > result = fitdistcens(data, 't', control=list(reltol=1e-14),
    +                      start=list(df=1, ncp=2))
    > result
    Fitting of the distribution ' t ' on censored data by maximum likelihood
    Parameters:
         estimate
    df  0.5432336
    ncp 2.8893565

    """
    # 创建一个带有右截尾的 CensoredData 对象
    data = CensoredData.right_censored([1, 2, 3, 5, 8, 10, 25, 25],
                                       [0, 0, 0, 0, 0, 0, 1, 1])
    # 使用 np.errstate 设置错误状态为忽略，待 gh-14901 问题解决后移除此上下文
    with np.errstate(over='ignore'):
        # 对数据进行非中心 t 分布的拟合，只拟合形状参数 df 和 nc；loc 和 scale 被固定
        df, nc, loc, scale = nct.fit(data, floc=0, fscale=1,
                                     optimizer=optimizer)
    # 断言拟合后的 df 和 nc 与预期值非常接近
    assert_allclose(df, 0.5432336, rtol=5e-6)
    assert_allclose(nc, 2.8893565, rtol=5e-6)
    # 断言 loc 和 scale 分别为 0 和 1
    assert loc == 0
    assert scale == 1


def test_ncx2():
    """
    Test fitting the shape parameters (df, ncp) of ncx2 to mixed data.

    Calculation in R, with
    * 5 not censored values [2.7, 0.2, 6.5, 0.4, 0.1],
    * 1 interval-censored value [[0.6, 1.0]], and
    * 2 right-censored values [8, 8].

    > library(fitdistrplus)
    > data <- data.frame(left=c(2.7, 0.2, 6.5, 0.4, 0.1, 0.6, 8, 8),
    +                    right=c(2.7, 0.2, 6.5, 0.4, 0.1, 1.0, NA, NA))
    > result = fitdistcens(data, 'chisq', control=list(reltol=1e-14),
    +                      start=list(df=1, ncp=2))
    > result
    Fitting of the distribution ' chisq ' on censored data by maximum
    likelihood
    Parameters:
        estimate
    df  1.052871
    ncp 2.362934
    """
    # 创建包含不截尾值、右截尾值和区间截尾值的 CensoredData 对象
    data = CensoredData(uncensored=[2.7, 0.2, 6.5, 0.4, 0.1], right=[8, 8],
                        interval=[[0.6, 1.0]])
    # 使用 np.errstate 设置错误状态为忽略，待 gh-14901 问题解决后移除此上下文
    with np.errstate(over='ignore'):
        # 对数据进行 ncx2 分布的拟合，只拟合形状参数 df 和 ncp；loc 和 scale 被固定
        df, ncp, loc, scale = ncx2.fit(data, floc=0, fscale=1,
                                       optimizer=optimizer)
    # 断言拟合后的 df 和 ncp 与预期值非常接近
    assert_allclose(df, 1.052871, rtol=5e-6)
    assert_allclose(ncp, 2.362934, rtol=5e-6)
    # 断言 loc 和 scale 分别为 0 和 1
    assert loc == 0
    assert scale == 1


def test_norm():
    """
    Test fitting the normal distribution to interval-censored data.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(0.10, 0.50, 0.75, 0.80),
    +                    right=c(0.20, 0.55, 0.90, 0.95))
    > result = fitdistcens(data, 'norm', control=list(reltol=1e-14))

    > result
    Fitting of the distribution ' norm ' on censored data by maximum likelihood
    Parameters:
          estimate
    mean 0.5919990
    sd   0.2868042
    > result$sd
         mean        sd
    0.1444432 0.1029451
    """
    # 创建一个名为 `data` 的对象，其中包含已经被审查或加密处理过的数据，这些数据以区间的形式给出
    data = CensoredData(interval=[[0.10, 0.20],
                                  [0.50, 0.55],
                                  [0.75, 0.90],
                                  [0.80, 0.95]])
    
    # 使用正态分布去拟合给定的 `data` 数据，并通过 `optimizer` 参数进行优化
    loc, scale = norm.fit(data, optimizer=optimizer)
    
    # 断言检查拟合后的均值 `loc` 和标准差 `scale` 是否与给定值非常接近，允许的相对误差为 5e-6
    assert_allclose(loc, 0.5919990, rtol=5e-6)
    assert_allclose(scale, 0.2868042, rtol=5e-6)
def test_weibull_censored1():
    # Ref: http://www.ams.sunysb.edu/~zhu/ams588/Lecture_3_likelihood.pdf
    # 生存时间数据，'*' 表示右截尾（右侧未观察到的数据）
    s = "3,5,6*,8,10*,11*,15,20*,22,23,27*,29,32,35,40,26,28,33*,21,24*"
    
    # 解析生存时间和截尾信息
    times, cens = zip(*[(float(t[0]), len(t) == 2)
                        for t in [w.split('*') for w in s.split(',')]])
    
    # 创建右截尾的数据对象
    data = CensoredData.right_censored(times, cens)

    # 对 Weibull 分布的最小值进行拟合
    c, loc, scale = weibull_min.fit(data, floc=0)

    # 预期结果来自参考文献
    assert_allclose(c, 2.149, rtol=1e-3)
    assert loc == 0
    assert_allclose(scale, 28.99, rtol=1e-3)

    # 翻转数据的符号，并使截尾数据成为左截尾。当我们对翻转后的数据进行
    # Weibull 最大值拟合时，应该得到相同的参数。
    data2 = CensoredData.left_censored(-np.array(times), cens)

    # 对 Weibull 分布的最大值进行拟合
    c2, loc2, scale2 = weibull_max.fit(data2, floc=0)

    assert_allclose(c2, 2.149, rtol=1e-3)
    assert loc2 == 0
    assert_allclose(scale2, 28.99, rtol=1e-3)


def test_weibull_min_sas1():
    # Data and SAS results from
    #   https://support.sas.com/documentation/cdl/en/qcug/63922/HTML/default/
    #         viewer.htm#qcug_reliability_sect004.htm

    # SAS 结果和数据来源于上述链接
    text = """
           450 0    460 1   1150 0   1150 0   1560 1
          1600 0   1660 1   1850 1   1850 1   1850 1
          1850 1   1850 1   2030 1   2030 1   2030 1
          2070 0   2070 0   2080 0   2200 1   3000 1
          3000 1   3000 1   3000 1   3100 0   3200 1
          3450 0   3750 1   3750 1   4150 1   4150 1
          4150 1   4150 1   4300 1   4300 1   4300 1
          4300 1   4600 0   4850 1   4850 1   4850 1
          4850 1   5000 1   5000 1   5000 1   6100 1
          6100 0   6100 1   6100 1   6300 1   6450 1
          6450 1   6700 1   7450 1   7800 1   7800 1
          8100 1   8100 1   8200 1   8500 1   8500 1
          8500 1   8750 1   8750 0   8750 1   9400 1
          9900 1  10100 1  10100 1  10100 1  11500 1
    """

    # 解析生存时间和截尾信息
    life, cens = np.array([int(w) for w in text.split()]).reshape(-1, 2).T
    life = life / 1000.0

    # 创建右截尾的数据对象
    data = CensoredData.right_censored(life, cens)

    # 对 Weibull 分布的最小值进行拟合
    c, loc, scale = weibull_min.fit(data, floc=0, optimizer=optimizer)
    assert_allclose(c, 1.0584, rtol=1e-4)
    assert_allclose(scale, 26.2968, rtol=1e-5)
    assert loc == 0


def test_weibull_min_sas2():
    # http://support.sas.com/documentation/cdl/en/ormpug/67517/HTML/default/
    #      viewer.htm#ormpug_nlpsolver_examples06.htm

    # 最后两个值是右截尾的
    days = np.array([143, 164, 188, 188, 190, 192, 206, 209, 213, 216, 220,
                     227, 230, 234, 246, 265, 304, 216, 244])

    # 创建右截尾的数据对象
    data = CensoredData.right_censored(days, [0] * (len(days) - 2) + [1] * 2)

    # 对 Weibull 分布的最小值进行拟合
    c, loc, scale = weibull_min.fit(data, 1, loc=100, scale=100,
                                    optimizer=optimizer)

    assert_allclose(c, 2.7112, rtol=5e-4)
    assert_allclose(loc, 122.03, rtol=5e-4)
    assert_allclose(scale, 108.37, rtol=5e-4)
```