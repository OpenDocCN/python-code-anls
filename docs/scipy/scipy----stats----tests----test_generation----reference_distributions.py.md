# `D:\src\scipysrc\scipy\scipy\stats\tests\test_generation\reference_distributions.py`

```
# 导入 numpy 库，并使用 np 作为别名
import numpy as np
# 导入 mpmath 库
import mpmath
# 从 mpmath 库中导入 mp 对象
from mpmath import mp

# 定义一个名为 ReferenceDistribution 的类
class ReferenceDistribution:
    """Minimalist distribution infrastructure for generating reference data.

    The purpose is to generate reference values for unit tests of SciPy
    distribution accuracy and robustness.

    Handles array input with standard broadcasting rules, and method
    implementations are easily compared against their mathematical definitions.
    No attempt is made to handle edge cases or be fast, and arbitrary precision
    arithmetic is trusted for accuracy rather than making the method
    implementations "smart".

    Notes
    -----

    In this infrastructure, distributions families are classes, and
    fully-specified distributions (i.e. with definite values of all family
    parameters) are instances of these classes. Typically, the public methods
    accept as input only the argument at which the at which the function is to
    be evaluated. Unlike SciPy distributions, they never accept values of
    distribution family shape, location, or scale parameters. A few
    other parameters are noteworthy:

    - All methods accept `dtype` to control the output data type. The default
      is `np.float64`, but `object` or `mp.mpf` may be
      specified to output the full `mpf`.
    - `ppf`/`isf` accept a `guess` because they use a scalar rootfinder
      to invert the `cdf`/`sf`. This is passed directly into the `x0` method
      of `mpmath.findroot`; see its documentation for details.
    - moment accepts `order`, an integer that specifies the order of the (raw)
      moment, and `center`, which is the value about which the moment is
      taken. The default is to calculate the mean and use it to calculate
      central moments; passing ``0`` results in a noncentral moment. For
      efficiency, the mean can be passed explicitly if it is already known.

    Follow the example of SkewNormal to generate new reference distributions,
    overriding only `__init__` and `_pdf`*. Use the reference distributions to
    generate reference values for unit tests of SciPy distribution method
    precision and robustness (e.g. for extreme arguments). If the a SciPy
    methods implementation is independent and yet the output matches reference
    values generated with this infrastructure, it is unlikely that the SciPy
    and reference values are both inaccurate.

    * If the SciPy output *doesn't* match and the cause appears to be
    inaccuracy of the reference values (e.g. due to numerical issues that
    mpmath's arbitrary precision arithmetic doesn't handle), then it may be
    appropriate to override a method of the reference distribution rather than
    relying on the generic implementation. Otherwise, hesitate to override
    methods: the generic implementations are mathematically correct and easy
    to verify, whereas an override introduces many possibilities of mistakes,
    requires more time to write, and requires more time to review.
    """
    pass  # 这里是类的定义结束，目前类还没有任何方法或属性，只有一个文档字符串作为说明
    # 初始化方法，用于设置对象属性并检查精度设置
    def __init__(self, **kwargs):
        try:
            # 检查是否已经设置了 mpmath.dps，如果是则抛出异常
            if mpmath.dps is not None:
                message = ("`mpmath.dps` has been assigned. This is not "
                           "intended usage; instead, assign the desired "
                           "precision to `mpmath.mp.dps` (e.g. `from mpmath "
                           "import mp; mp.dps = 50.`)")
                raise RuntimeError(message)
        except AttributeError:
            # 如果 mpmath 模块没有 dps 属性，将其设为 None
            mpmath.dps = None

        # 检查 mpmath.mp.dps 是否小于等于 15，如果是则抛出异常
        if mp.dps <= 15:
            message = ("`mpmath.mp.dps <= 15`. Set a higher precision (e.g."
                       "`50`) to use this distribution.")
            raise RuntimeError(message)

        # 将传入的关键字参数转换为 _params 字典，其中值是转换为 mpmath.mpf 数组的结果
        self._params = {key:self._make_mpf_array(val)
                        for key, val in kwargs.items()}

    # 将输入 x 转换为 mpmath.mpf 数组
    def _make_mpf_array(self, x):
        shape = np.shape(x)
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.asarray([mp.mpf(xi) for xi in x]).reshape(shape)[()]

    # 概率密度函数（PDF），需要在子类中实现
    def _pdf(self, x):
        raise NotImplementedError("_pdf must be overridden.")

    # 累积分布函数（CDF），根据参数计算分布函数值
    def _cdf(self, x, **kwargs):
        # 如果当前对象的 _cdf 方法是基类的 _cdf 方法，并且 _sf 方法不是基类的 _sf 方法，则计算并返回 1 - _sf(x)
        if ((self._cdf.__func__ is ReferenceDistribution._cdf)
                and (self._sf.__func__ is not ReferenceDistribution._sf)):
            return mp.one - self._sf(x, **kwargs)

        # 获取分布的支持区间 (a, b)
        a, b = self._support(**kwargs)
        # 使用数值积分计算从 a 到 x 的累积分布函数值
        res = mp.quad(lambda x: self._pdf(x, **kwargs), (a, x))
        # 如果计算的结果大于等于 0.5，则返回 1 - _sf(x)，否则返回计算的结果
        res = res if res < 0.5 else mp.one - self._sf(x, **kwargs)
        return res

    # 生存函数（SF），计算 1 - CDF(x)
    def _sf(self, x, **kwargs):
        # 如果当前对象的 _sf 方法是基类的 _sf 方法，并且 _cdf 方法不是基类的 _cdf 方法，则计算并返回 1 - _cdf(x)
        if ((self._sf.__func__ is ReferenceDistribution._sf)
                and (self._cdf.__func__ is not ReferenceDistribution._cdf)):
            return mp.one - self._cdf(x, **kwargs)

        # 获取分布的支持区间 (a, b)
        a, b = self._support(**kwargs)
        # 使用数值积分计算从 x 到 b 的生存函数值
        res = mp.quad(lambda x: self._pdf(x, **kwargs), (x, b))
        # 如果计算的结果大于等于 0.5，则返回 1 - _cdf(x)，否则返回计算的结果
        res = res if res < 0.5 else mp.one - self._cdf(x, **kwargs)
        return res

    # 百分点函数（PPF），计算给定概率 p 的分布的反函数值
    def _ppf(self, p, guess=0, **kwargs):
        # 如果当前对象的 _ppf 方法是基类的 _ppf 方法，并且 _isf 方法不是基类的 _isf 方法，则计算并返回 _isf(1 - p)
        if ((self._ppf.__func__ is ReferenceDistribution._ppf)
                and (self._isf.__func__ is not ReferenceDistribution._isf)):
            return self._isf(mp.one - p, guess, **kwargs)
        
        # 定义求解方程的函数，返回 _cdf(x) - p
        def f(x):
            return self._cdf(x, **kwargs) - p
        # 使用数值方法找到使得 f(x) 接近 0 的 x 值，作为百分点函数的结果
        return mp.findroot(f, guess)

    # 逆生存函数（ISF），计算给定概率 p 的生存函数的反函数值
    def _isf(self, p, guess=0, **kwargs):
        # 如果当前对象的 _isf 方法是基类的 _isf 方法，并且 _ppf 方法不是基类的 _ppf 方法，则计算并返回 _ppf(1 - p)
        if ((self._isf.__func__ is ReferenceDistribution._isf)
                and (self._ppf.__func__ is not ReferenceDistribution._ppf)):
            return self._ppf(mp.one - p, guess, **kwargs)
        
        # 定义求解方程的函数，返回 _sf(x) - p
        def f(x):
            return self._sf(x, **kwargs) - p
        # 使用数值方法找到使得 f(x) 接近 0 的 x 值，作为逆生存函数的结果
        return mp.findroot(f, guess)

    # 对数概率密度函数（Log PDF），计算取对数后的概率密度函数值
    def _logpdf(self, x, **kwargs):
        return mp.log(self._pdf(x, **kwargs))

    # 对数累积分布函数（Log CDF），计算取对数后的累积分布函数值
    def _logcdf(self, x, **kwargs):
        return mp.log(self._cdf(x, **kwargs))
    # 计算对数生存函数，即对生存函数的对数进行计算
    def _logsf(self, x, **kwargs):
        return mp.log(self._sf(x, **kwargs))

    # 返回分布的支持区间，这里默认为负无穷到正无穷
    def _support(self, **kwargs):
        return -mp.inf, mp.inf

    # 计算分布的熵
    def _entropy(self, **kwargs):
        # 定义积分被积函数
        def integrand(x):
            logpdf = self._logpdf(x, **kwargs)  # 获取对数概率密度函数
            pdf = mp.exp(logpdf)  # 计算概率密度函数
            return -pdf * logpdf  # 返回被积函数值

        a, b = self._support(**kwargs)  # 获取支持区间
        return mp.quad(integrand, (a, b))  # 对被积函数进行数值积分

    # 计算分布的均值
    def _mean(self, **kwargs):
        return self._moment(order=1, center=0, **kwargs)  # 使用矩函数计算一阶矩（均值）

    # 计算分布的方差
    def _var(self, **kwargs):
        mu = self._mean(**kwargs)  # 获取均值
        return self._moment(order=2, center=mu, **kwargs)  # 使用矩函数计算二阶矩（方差）

    # 计算分布的偏度
    def _skew(self, **kwargs):
        mu = self._mean(**kwargs)  # 获取均值
        u2 = self._moment(order=2, center=mu, **kwargs)  # 获取二阶矩（方差）
        sigma = mp.sqrt(u2)  # 计算标准差
        u3 = self._moment(order=3, center=mu, **kwargs)  # 获取三阶矩
        return u3 / sigma ** 3  # 计算偏度

    # 计算分布的峰度
    def _kurtosis(self, **kwargs):
        mu = self._mean(**kwargs)  # 获取均值
        u2 = self._moment(order=2, center=mu, **kwargs)  # 获取二阶矩（方差）
        u4 = self._moment(order=4, center=mu, **kwargs)  # 获取四阶矩
        return u4 / u2 ** 2 - 3  # 计算峰度

    # 计算矩函数的值
    def _moment(self, order, center, **kwargs):
        # 定义积分被积函数
        def integrand(x):
            return self._pdf(x, **kwargs) * (x - center) ** order  # 返回被积函数值

        if center is None:  # 如果中心值未指定，则使用均值作为中心值
            center = self._mean(**kwargs)

        a, b = self._support(**kwargs)  # 获取支持区间
        return mp.quad(integrand, (a, b))  # 对被积函数进行数值积分

    # 计算概率密度函数的值
    def pdf(self, x, dtype=np.float64):
        fun = np.vectorize(self._pdf)  # 向量化概率密度函数
        x = self._make_mpf_array(x)  # 将输入数组转换为多精度浮点数数组
        res = fun(x, **self._params)  # 计算向量化概率密度函数的值
        return np.asarray(res, dtype=dtype)[()]  # 将结果转换为指定数据类型并返回

    # 计算累积分布函数的值
    def cdf(self, x, dtype=np.float64):
        fun = np.vectorize(self._cdf)  # 向量化累积分布函数
        x = self._make_mpf_array(x)  # 将输入数组转换为多精度浮点数数组
        res = fun(x, **self._params)  # 计算向量化累积分布函数的值
        return np.asarray(res, dtype=dtype)[()]  # 将结果转换为指定数据类型并返回

    # 计算生存函数的值
    def sf(self, x, dtype=np.float64):
        fun = np.vectorize(self._sf)  # 向量化生存函数
        x = self._make_mpf_array(x)  # 将输入数组转换为多精度浮点数数组
        res = fun(x, **self._params)  # 计算向量化生存函数的值
        return np.asarray(res, dtype=dtype)[()]  # 将结果转换为指定数据类型并返回

    # 计算分位点函数的值
    def ppf(self, x, guess=0, dtype=np.float64):
        fun = np.vectorize(self._ppf, excluded={1})  # 向量化分位点函数，排除第二个参数（猜测值）
        x = self._make_mpf_array(x)  # 将输入数组转换为多精度浮点数数组
        res = fun(x, guess, **self._params)  # 计算向量化分位点函数的值
        return np.asarray(res, dtype=dtype)[()]  # 将结果转换为指定数据类型并返回

    # 计算逆生存函数的值
    def isf(self, x, guess=0, dtype=np.float64):
        fun = np.vectorize(self._isf, excluded={1})  # 向量化逆生存函数，排除第二个参数（猜测值）
        x = self._make_mpf_array(x)  # 将输入数组转换为多精度浮点数数组
        res = fun(x, guess, **self._params)  # 计算向量化逆生存函数的值
        return np.asarray(res, dtype=dtype)[()]  # 将结果转换为指定数据类型并返回

    # 计算对数概率密度函数的值
    def logpdf(self, x, dtype=np.float64):
        fun = np.vectorize(self._logpdf)  # 向量化对数概率密度函数
        x = self._make_mpf_array(x)  # 将输入数组转换为多精度浮点数数组
        res = fun(x, **self._params)  # 计算向量化对数概率密度函数的值
        return np.asarray(res, dtype=dtype)[()]  # 将结果转换为指定数据类型并返回

    # 计算对数累积分布函数的值
    def logcdf(self, x, dtype=np.float64):
        fun = np.vectorize(self._logcdf)  # 向量化对数累积分布函数
        x = self._make_mpf_array(x)  # 将输入数组转换为多精度浮点数数组
        res = fun(x, **self._params)  # 计算向量化对数累积分布函数的值
        return np.asarray(res, dtype=dtype)[()]  # 将结果转换为指定数据类型并返回
    # 计算对数生存函数值的方法
    def logsf(self, x, dtype=np.float64):
        # 使用 np.vectorize 将 _logsf 方法向量化
        fun = np.vectorize(self._logsf)
        # 将输入参数 x 转换为多精度浮点数数组
        x = self._make_mpf_array(x)
        # 调用向量化后的函数计算结果
        res = fun(x, **self._params)
        # 将结果转换为指定数据类型并返回
        return np.asarray(res, dtype=dtype)[()]

    # 计算分布支持的方法
    def support(self, dtype=np.float64):
        # 使用 np.vectorize 将 _support 方法向量化
        fun = np.vectorize(self._support)
        # 调用向量化后的函数计算结果
        res = fun(**self._params)
        # 将结果转换为指定数据类型并返回
        return np.asarray(res, dtype=dtype)[()]

    # 计算分布熵的方法
    def entropy(self, dtype=np.float64):
        # 使用 np.vectorize 将 _entropy 方法向量化
        fun = np.vectorize(self._entropy)
        # 调用向量化后的函数计算结果
        res = fun(**self._params)
        # 将结果转换为指定数据类型并返回
        return np.asarray(res, dtype=dtype)[()]

    # 计算分布期望的方法
    def mean(self, dtype=np.float64):
        # 使用 np.vectorize 将 _mean 方法向量化
        fun = np.vectorize(self._mean)
        # 调用向量化后的函数计算结果
        res = fun(**self._params)
        # 将结果转换为指定数据类型并返回
        return np.asarray(res, dtype=dtype)[()]

    # 计算分布方差的方法
    def var(self, dtype=np.float64):
        # 使用 np.vectorize 将 _var 方法向量化
        fun = np.vectorize(self._var)
        # 调用向量化后的函数计算结果
        res = fun(**self._params)
        # 将结果转换为指定数据类型并返回
        return np.asarray(res, dtype=dtype)[()]

    # 计算分布偏度的方法
    def skew(self, dtype=np.float64):
        # 使用 np.vectorize 将 _skew 方法向量化
        fun = np.vectorize(self._skew)
        # 调用向量化后的函数计算结果
        res = fun(**self._params)
        # 将结果转换为指定数据类型并返回
        return np.asarray(res, dtype=dtype)[()]

    # 计算分布峰度的方法
    def kurtosis(self, dtype=np.float64):
        # 使用 np.vectorize 将 _kurtosis 方法向量化
        fun = np.vectorize(self._kurtosis)
        # 调用向量化后的函数计算结果
        res = fun(**self._params)
        # 将结果转换为指定数据类型并返回
        return np.asarray(res, dtype=dtype)[()]

    # 计算分布矩的方法
    def moment(self, order, center=None, dtype=np.float64):
        # 使用 np.vectorize 将 _moment 方法向量化
        fun = np.vectorize(self._moment)
        # 将输入参数 order 转换为多精度浮点数数组
        order = self._make_mpf_array(order)
        # 调用向量化后的函数计算结果
        res = fun(order, **self._params)
        # 将结果转换为指定数据类型并返回
        return np.asarray(res, dtype=dtype)[()]
class SkewNormal(ReferenceDistribution):
    """Reference implementation of the SkewNormal distribution.

    Follow the example here to generate new reference distributions.
    Use the reference distributions to generate reference values of
    distributions functions. For now, copy-paste the output into unit
    tests. Full code to generate reference values does not need to be
    included as a comment in the test; just refer to the reference
    distribution used and the settings (e.g. mp.dps=50).
    """

    def __init__(self, *, a):
        # 初始化方法，设置分布的参数a。使用关键字参数避免位置参数的歧义。
        super().__init__(a=a)

    def _support(self, a):
        # 如果分布的支持范围是实数线的子集，则重写_support方法来指定支持范围。
        return -mp.inf, mp.inf

    def _pdf(self, x, a):
        # 计算概率密度函数（PDF），尽量按照学术参考文献的描述实现。使用mpmath保证精度，
        # 不考虑速度问题。重要的是验证PDF与参考文献的一致性。
        return 2 * mp.npdf(x) * mp.ncdf(a * x)

    # 避免重写其他方法，除非通用实现被认为不准确（例如由于数值困难）或者太慢。
    # 为什么要避免？少写代码，少审核代码，保证实现中没有错误（例如错误的公式）。

class BetaPrime(ReferenceDistribution):

    def __init__(self, *, a, b):
        # 初始化方法，设置分布的参数a和b。
        super().__init__(a=a, b=b)

    def _support(self, **kwargs):
        # 如果分布的支持范围不是整个实数线，则重写_support方法来指定支持范围。
        return mp.zero, mp.inf

    def _logpdf(self, x, a, b):
        # 计算对数概率密度函数（log PDF），按照学术参考文献尽量实现。
        return (a - mp.one)*mp.log(x) - (a + b)*mp.log1p(x) - mp.log(mp.beta(a, b))

    def _pdf(self, x, a, b):
        # 计算概率密度函数（PDF），通过_logpdf方法计算后取指数。
        return mp.exp(self._logpdf(x=x, a=a, b=b))

    def _sf(self, x, a, b):
        # 计算生存函数（Survival function），即1减去累积分布函数。
        return 1.0 - mp.betainc(a, b, 0, x/(1+x), regularized=True)


class Burr(ReferenceDistribution):

    def __init__(self, *, c, d):
        # 初始化方法，设置分布的参数c和d。
        super().__init__(c=c, d=d)

    def _support(self, c, d):
        # 设置分布的支持范围，通常为从0到正无穷。
        return 0, mp.inf

    def _pdf(self, x, c, d):
        # 计算概率密度函数（PDF），按照Burr分布的公式实现。
        return c * d * x ** (-c - 1) * (1 + x ** (-c)) ** (-d - 1)

    def _ppf(self, p, guess, c, d):
        # 计算百分点函数（Percent point function），逆推出累积分布函数的自变量。
        return (p**(-1.0/d) - 1)**(-1.0/c)


class LogLaplace(ReferenceDistribution):
    # 在这里继续实现LogLaplace分布的类定义和方法实现。
    # 初始化函数，使用关键字参数 c 初始化父类
    def __init__(self, *, c):
        # 调用父类的初始化方法，传入参数 c
        super().__init__(c=c)

    # 支持函数，返回固定的元组 (0, 无穷大)
    def _support(self, c):
        return 0, mp.inf

    # 概率密度函数，根据输入参数 x 和 c 计算概率密度
    def _pdf(self, x, c):
        # 如果 x 小于 1
        if x < mp.one:
            # 返回 c / 2 * x^(c - 1) 的结果
            return c / 2 * x**(c - mp.one)
        else:
            # 返回 c / 2 * x^(-c - 1) 的结果
            return c / 2 * x**(-c - mp.one)

    # 百分点函数，根据输入参数 q、guess 和 c 计算分位点
    def _ppf(self, q, guess, c):
        # 如果 q 小于 0.5
        if q < 0.5:
            # 返回 (2 * q)^(1 / c) 的结果
            return (2.0 * q)**(mp.one / c)
        else:
            # 返回 (2 * (1 - q))^(-1 / c) 的结果
            return (2 * (mp.one - q))**(-mp.one / c)
class LogNormal(ReferenceDistribution):
    # LogNormal 类，继承自 ReferenceDistribution 类

    def __init__(self, *, s):
        # LogNormal 类的初始化方法，接受参数 s
        super().__init__(s=s)
        # 调用父类 ReferenceDistribution 的初始化方法，传入参数 s

    def _support(self, s):
        # 返回支持范围，始终返回 (0, 无穷大)
        return 0, mp.inf

    def _pdf(self, x, s):
        # 计算概率密度函数（Probability Density Function, PDF）的值
        return (
            mp.one / (s * x * mp.sqrt(2 * mp.pi))
            * mp.exp(-mp.one / 2 * (mp.log(x) / s)**2)
        )
        # LogNormal 分布的概率密度函数公式

    def _cdf(self, x, s):
        # 计算累积分布函数（Cumulative Distribution Function, CDF）的值
        return mp.ncdf(mp.log(x) / s)
        # 使用 mp.ncdf 计算 LogNormal 分布的累积分布函数


class Normal(ReferenceDistribution):
    # Normal 类，继承自 ReferenceDistribution 类

    def _pdf(self, x):
        # 计算标准正态分布的概率密度函数值
        return mp.npdf(x)
        # 使用 mp.npdf 计算标准正态分布的概率密度函数


class NormInvGauss(ReferenceDistribution):
    # NormInvGauss 类，继承自 ReferenceDistribution 类

    def __init__(self, *, alpha, beta):
        # NormInvGauss 类的初始化方法，接受参数 alpha 和 beta
        super().__init__(alpha=alpha, beta=beta)
        # 调用父类 ReferenceDistribution 的初始化方法，传入参数 alpha 和 beta

    def _pdf(self, x, alpha, beta):
        # 计算逆高斯分布的概率密度函数值
        # 实现参考 https://www.jstor.org/stable/4616433
        # 方程式 2.1 - 2.3
        q = mp.sqrt(1 + x**2)
        a = mp.pi**-1 * alpha * mp.exp(mp.sqrt(alpha**2 - beta**2))
        return a * q**-1 * mp.besselk(1, alpha*q) * mp.exp(beta*x)
        # 使用给定的公式计算逆高斯分布的概率密度函数


class Pearson3(ReferenceDistribution):
    # Pearson3 类，继承自 ReferenceDistribution 类

    def __init__(self, *, skew):
        # Pearson3 类的初始化方法，接受参数 skew
        super().__init__(skew=skew)
        # 调用父类 ReferenceDistribution 的初始化方法，传入参数 skew

    def _pdf(self, x, skew):
        # 计算 Pearson 类型 III 分布的概率密度函数值
        b = 2 / skew
        a = b**2
        c = -b
        res = abs(b)/mp.gamma(a) * (b*(x-c))**(a-1) * mp.exp(-b*(x-c))
        return res if abs(res.real) == res else 0
        # 使用给定的公式计算 Pearson 类型 III 分布的概率密度函数


class StudentT(ReferenceDistribution):
    # StudentT 类，继承自 ReferenceDistribution 类

    def __init(self, *, df):
        # StudentT 类的初始化方法，接受参数 df
        super().__init__(df=df)
        # 调用父类 ReferenceDistribution 的初始化方法，传入参数 df

    def _pdf(self, x, df):
        # 计算学生 t 分布的概率密度函数值
        return (mp.gamma((df + mp.one)/2)/(mp.sqrt(df * mp.pi) * mp.gamma(df/2))
                * (mp.one + x*x/df)**(-(df + mp.one)/2))
        # 使用给定的公式计算学生 t 分布的概率密度函数


class TruncExpon(ReferenceDistribution):
    # TruncExpon 类，继承自 ReferenceDistribution 类

    def __init__(self, *, b):
        # TruncExpon 类的初始化方法，接受参数 b
        super().__init__(b=b)
        # 调用父类 ReferenceDistribution 的初始化方法，传入参数 b

    def _support(self, b):
        # 返回支持范围，从 0 到 b
        return 0, b

    def _pdf(self, x, b):
        # 计算截断指数分布的概率密度函数值
        return -mp.exp(-x)/mp.expm1(-b)
        # 使用给定的公式计算截断指数分布的概率密度函数

    def _sf(self, x, b):
        # 计算截断指数分布的生存函数值（1 - CDF）
        return (mp.exp(-b) - mp.exp(-x))/mp.expm1(-b)
        # 使用给定的公式计算截断指数分布的生存函数
```