# `D:\src\scipysrc\sympy\sympy\stats\joint_rv_types.py`

```
# 从 sympy.concrete.products 模块导入 Product 类
# 用于处理乘积表达式的类
from sympy.concrete.products import Product
# 从 sympy.concrete.summations 模块导入 Sum 类
# 用于处理求和表达式的类
from sympy.concrete.summations import Sum
# 从 sympy.core.add 模块导入 Add 类
# 用于表示加法表达式的类
from sympy.core.add import Add
# 从 sympy.core.function 模块导入 Lambda 类
# 用于表示匿名函数的类
from sympy.core.function import Lambda
# 从 sympy.core.mul 模块导入 Mul 类
# 用于表示乘法表达式的类
from sympy.core.mul import Mul
# 从 sympy.core.numbers 模块导入 Integer, Rational, pi 类
# 用于表示整数、有理数和常数 pi 的类
from sympy.core.numbers import (Integer, Rational, pi)
# 从 sympy.core.power 模块导入 Pow 类
# 用于表示幂次表达式的类
from sympy.core.power import Pow
# 从 sympy.core.relational 模块导入 Eq 类
# 用于表示相等关系的类
from sympy.core.relational import Eq
# 从 sympy.core.singleton 模块导入 S 类
# 用于表示单例对象的类
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Symbol, symbols 类
# 用于表示符号和符号集合的类
from sympy.core.symbol import (Symbol, symbols)
# 从 sympy.core.sympify 模块导入 sympify 函数
# 用于将字符串或其他表示转换为 sympy 对象的函数
from sympy.core.sympify import sympify
# 从 sympy.functions.combinatorial.factorials 模块导入 rf, factorial 函数
# 用于计算阶乘和双重阶乘的函数
from sympy.functions.combinatorial.factorials import (rf, factorial)
# 从 sympy.functions.elementary.exponential 模块导入 exp 函数
# 用于计算指数函数的函数
from sympy.functions.elementary.exponential import exp
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数
# 用于计算平方根的函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise 类
# 用于表示分段函数的类
from sympy.functions.elementary.piecewise import Piecewise
# 从 sympy.functions.special.bessel 模块导入 besselk 函数
# 用于计算贝塞尔函数的函数
from sympy.functions.special.bessel import besselk
# 从 sympy.functions.special.gamma_functions 模块导入 gamma 函数
# 用于计算伽玛函数的函数
from sympy.functions.special.gamma_functions import gamma
# 从 sympy.matrices.dense 模块导入 Matrix, ones 类
# 用于表示稠密矩阵和单位矩阵的类
from sympy.matrices.dense import (Matrix, ones)
# 从 sympy.sets.fancysets 模块导入 Range 类
# 用于表示数值范围的类
from sympy.sets.fancysets import Range
# 从 sympy.sets.sets 模块导入 Intersection, Interval 类
# 用于表示集合交和区间的类
from sympy.sets.sets import (Intersection, Interval)
# 从 sympy.tensor.indexed 模块导入 Indexed, IndexedBase 类
# 用于表示索引和索引基的类
from sympy.tensor.indexed import (Indexed, IndexedBase)
# 从 sympy.matrices 模块导入 ImmutableMatrix, MatrixSymbol 类
# 用于表示不可变矩阵和矩阵符号的类
from sympy.matrices import ImmutableMatrix, MatrixSymbol
# 从 sympy.matrices.expressions.determinant 模块导入 det 函数
# 用于计算矩阵行列式的函数
from sympy.matrices.expressions.determinant import det
# 从 sympy.matrices.expressions.matexpr 模块导入 MatrixElement 类
# 用于表示矩阵元素的类
from sympy.matrices.expressions.matexpr import MatrixElement
# 从 sympy.stats.joint_rv 模块导入 JointDistribution, JointPSpace, MarginalDistribution 类
# 用于表示联合随机变量、联合概率空间和边缘分布的类
from sympy.stats.joint_rv import JointDistribution, JointPSpace, MarginalDistribution
# 从 sympy.stats.rv 模块导入 _value_check, random_symbols 函数
# 用于检查值和生成随机符号的函数
from sympy.stats.rv import _value_check, random_symbols

# __all__ 列表，包含模块中需要导出的公开对象名称
__all__ = ['JointRV',
'MultivariateNormal',
'MultivariateLaplace',
'Dirichlet',
'GeneralizedMultivariateLogGamma',
'GeneralizedMultivariateLogGammaOmega',
'Multinomial',
'MultivariateBeta',
'MultivariateEwens',
'MultivariateT',
'NegativeMultinomial',
'NormalGamma'
]

# 定义函数 multivariate_rv
def multivariate_rv(cls, sym, *args):
    # 将 args 中的每个元素转换为 sympy 对象
    args = list(map(sympify, args))
    # 使用参数创建一个分布对象
    dist = cls(*args)
    # 获取分布对象的参数
    args = dist.args
    # 检查分布的有效性
    dist.check(*args)
    # 返回联合概率空间对象
    return JointPSpace(sym, dist).value


# 定义函数 marginal_distribution
def marginal_distribution(rv, *indices):
    """
    Marginal distribution function of a joint random variable.

    Parameters
    ==========

    rv : A random variable with a joint probability distribution.
    indices : Component indices or the indexed random symbol
        for which the joint distribution is to be calculated

    Returns
    =======

    A Lambda expression in `sym`.

    Examples
    ========

    >>> from sympy.stats import MultivariateNormal, marginal_distribution
    >>> m = MultivariateNormal('X', [1, 2], [[2, 1], [1, 2]])
    >>> marginal_distribution(m, m[0])(1)
    1/(2*sqrt(pi))

    """
    # 将 indices 转换为列表
    indices = list(indices)
    # 如果 indices 中的元素是 Indexed 对象，则取出索引值
    for i in range(len(indices)):
        if isinstance(indices[i], Indexed):
            indices[i] = indices[i].args[1]
    # 获取随机变量的概率空间对象
    prob_space = rv.pspace
    # 如果 indices
    # 定义一个方法 `set`，用于返回对象实例的 `args` 属性的第二个元素
    def set(self):
        return self.args[1]
# 定义一个函数，用于创建一个联合随机变量，其中每个组件是连续的。
def JointRV(symbol, pdf, _set=None):
    """
    Create a Joint Random Variable where each of its component is continuous,
    given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    pdf : A PDF in terms of indexed symbols of the symbol given
        as the first argument

    NOTE
    ====

    As of now, the set for each component for a ``JointRV`` is
    equal to the set of all integers, which cannot be changed.

    Examples
    ========

    >>> from sympy import exp, pi, Indexed, S
    >>> from sympy.stats import density, JointRV
    >>> x1, x2 = (Indexed('x', i) for i in (1, 2))
    >>> pdf = exp(-x1**2/2 + x1 - x2**2/2 - S(1)/2)/(2*pi)
    >>> N1 = JointRV('x', pdf) #Multivariate Normal distribution
    >>> density(N1)(1, 2)
    exp(-2)/(2*pi)

    Returns
    =======

    RandomSymbol

    """
    # TODO: Add support for sets provided by the user

    # 将 symbol 转换为 sympy 符号对象
    symbol = sympify(symbol)

    # 从 pdf 的自由符号中筛选出符合条件的索引符号
    syms = [i for i in pdf.free_symbols if isinstance(i, Indexed)
            and i.base == IndexedBase(symbol)]

    # 根据索引的参数对符号进行排序，并转换为元组
    syms = tuple(sorted(syms, key=lambda index: index.args[1]))

    # 设置联合随机变量的集合为实数集的 k 次幂，其中 k 是符号数量
    _set = S.Reals**len(syms)

    # 将 pdf 包装为 Lambda 函数
    pdf = Lambda(syms, pdf)

    # 使用自定义的 pdf 和集合创建联合分布对象
    dist = JointDistributionHandmade(pdf, _set)

    # 通过 JointPSpace 创建联合随机变量的值
    jrv = JointPSpace(symbol, dist).value

    # 获取 pdf 中的随机符号
    rvs = random_symbols(pdf)

    # 如果 pdf 中存在随机符号，则计算边缘分布并返回其值
    if len(rvs) != 0:
        dist = MarginalDistribution(dist, (jrv,))
        return JointPSpace(symbol, dist).value

    # 否则直接返回联合随机变量的值
    return jrv

#-------------------------------------------------------------------------------
# Multivariate Normal distribution ---------------------------------------------

class MultivariateNormalDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')

    is_Continuous=True

    @property
    def set(self):
        # 返回多元正态分布的定义域，即实数集的 k 次幂，其中 k 是均值向量的维度
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma):
        # 检查均值向量和协方差矩阵的尺寸是否匹配
        _value_check(mu.shape[0] == sigma.shape[0],
            "Size of the mean vector and covariance matrix are incorrect.")

        # 检查协方差矩阵是否半正定
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_semidefinite,
                "The covariance matrix must be positive semi definite. ")

    def pdf(self, *args):
        # 获取均值向量和协方差矩阵
        mu, sigma = self.mu, self.sigma
        k = mu.shape[0]

        # 根据参数长度和类型初始化参数向量 x
        if len(args) == 1 and args[0].is_Matrix:
            args = args[0]
        else:
            args = ImmutableMatrix(args)
        x = args - mu

        # 计算多元正态分布的概率密度函数
        density = S.One / sqrt((2*pi)**(k) * det(sigma)) * exp(
            Rational(-1, 2) * x.transpose() * (sigma.inv() * x))

        # 返回密度矩阵的元素 (0, 0)
        return MatrixElement(density, 0, 0)
    # 定义一个方法 `_marginal_distribution`，计算给定索引下的边缘分布函数
    def _marginal_distribution(self, indices, sym):
        # 创建不可变矩阵 `sym`，包含了给定索引对应的符号
        sym = ImmutableMatrix([Indexed(sym, i) for i in indices])
        # 复制当前对象的均值向量 `_mu` 和协方差矩阵 `_sigma`
        _mu, _sigma = self.mu, self.sigma
        # 获取均值向量的维度 `k`
        k = self.mu.shape[0]
        # 遍历 `k` 维度的均值向量
        for i in range(k):
            # 如果当前索引 `i` 不在给定的索引集合 `indices` 中
            if i not in indices:
                # 从 `_mu` 中删除第 `i` 行
                _mu = _mu.row_del(i)
                # 从 `_sigma` 中删除第 `i` 列
                _sigma = _sigma.col_del(i)
                # 再从 `_sigma` 中删除第 `i` 行
                _sigma = _sigma.row_del(i)
        # 返回一个 Lambda 表达式，计算边缘分布函数
        return Lambda(tuple(sym), S.One/sqrt((2*pi)**(len(_mu))*det(_sigma))*exp(
            Rational(-1, 2)*(_mu - sym).transpose()*(_sigma.inv()*\
                (_mu - sym)))[0])
# 创建一个多元正态分布的随机变量
def MultivariateNormal(name, mu, sigma):
    r"""
    Creates a continuous random variable with Multivariate Normal
    Distribution.

    The density of the multivariate normal distribution can be found at [1].

    Parameters
    ==========

    mu : List representing the mean or the mean vector
        均值或均值向量的列表表示
    sigma : Positive semidefinite square matrix
        Represents covariance Matrix.
        如果 `\sigma` 是非可逆的，则当前仅支持抽样

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import MultivariateNormal, density, marginal_distribution
    >>> from sympy import symbols, MatrixSymbol
    >>> X = MultivariateNormal('X', [3, 4], [[2, 1], [1, 2]])
    >>> y, z = symbols('y z')
    >>> density(X)(y, z)
    sqrt(3)*exp(-y**2/3 + y*z/3 + 2*y/3 - z**2/3 + 5*z/3 - 13/3)/(6*pi)
    >>> density(X)(1, 2)
    sqrt(3)*exp(-4/3)/(6*pi)
    >>> marginal_distribution(X, X[1])(y)
    exp(-(y - 4)**2/4)/(2*sqrt(pi))
    >>> marginal_distribution(X, X[0])(y)
    exp(-(y - 3)**2/4)/(2*sqrt(pi))

    The example below shows that it is also possible to use
    symbolic parameters to define the MultivariateNormal class.

    >>> n = symbols('n', integer=True, positive=True)
    >>> Sg = MatrixSymbol('Sg', n, n)
    >>> mu = MatrixSymbol('mu', n, 1)
    >>> obs = MatrixSymbol('obs', n, 1)
    >>> X = MultivariateNormal('X', mu, Sg)

    The density of a multivariate normal can be
    calculated using a matrix argument, as shown below.

    >>> density(X)(obs)
    (exp(((1/2)*mu.T - (1/2)*obs.T)*Sg**(-1)*(-mu + obs))/sqrt((2*pi)**n*Determinant(Sg)))[0, 0]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    """
    # 调用 multivariate_rv 函数创建多元正态分布的随机变量
    return multivariate_rv(MultivariateNormalDistribution, name, mu, sigma)


#-------------------------------------------------------------------------------
# Multivariate Laplace distribution --------------------------------------------

# 多元拉普拉斯分布类，继承自 JointDistribution
class MultivariateLaplaceDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')
    is_Continuous=True

    @property
    # 设置属性，返回实数空间的 k 次幂
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    # 检查参数的静态方法，确保均值向量和协方差矩阵的大小正确
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0],
                     "Size of the mean vector and covariance matrix are incorrect.")
        # 检查协方差矩阵是否为正定
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite,
                         "The covariance matrix must be positive definite. ")
    # 定义一个方法pdf，接受可变数量的参数(*args)
    def pdf(self, *args):
        # 从self对象中获取mu和sigma属性
        mu, sigma = self.mu, self.sigma
        # 计算mu的转置
        mu_T = mu.transpose()
        # 创建一个大小为mu的行数的单位矩阵k
        k = S(mu.shape[0])
        # 计算sigma的逆矩阵
        sigma_inv = sigma.inv()
        # 将传入的参数args转换为不可变的矩阵ImmutableMatrix
        args = ImmutableMatrix(args)
        # 计算args的转置
        args_T = args.transpose()
        # 计算x的值
        x = (mu_T * sigma_inv * mu)[0]
        # 计算y的值
        y = (args_T * sigma_inv * args)[0]
        # 计算v的值
        v = 1 - k / 2
        # 返回pdf函数的结果，该结果是一个复杂的数学表达式
        return (2 * (y / (2 + x)) ** (v / 2) * besselk(v, sqrt((2 + x) * y)) *
                exp((args_T * sigma_inv * mu)[0]) /
                ((2 * pi) ** (k / 2) * sqrt(det(sigma))))
# 创建一个表示多元拉普拉斯分布的随机变量
def MultivariateLaplace(name, mu, sigma):
    """
    Creates a continuous random variable with Multivariate Laplace
    Distribution.

    The density of the multivariate Laplace distribution can be found at [1].

    Parameters
    ==========

    name : str
        Identifier for the random variable.
    mu : List
        Mean vector.
    sigma : Matrix
        Positive definite covariance matrix.

    Returns
    =======

    RandomSymbol
        Random variable following Multivariate Laplace distribution.

    Examples
    ========

    >>> from sympy.stats import MultivariateLaplace, density
    >>> from sympy import symbols
    >>> y, z = symbols('y z')
    >>> X = MultivariateLaplace('X', [2, 4], [[3, 1], [1, 3]])
    >>> density(X)(y, z)
    sqrt(2)*exp(y/4 + 5*z/4)*besselk(0, sqrt(15*y*(3*y/8 - z/8)/2 + 15*z*(-y/8 + 3*z/8)/2))/(4*pi)
    >>> density(X)(1, 2)
    sqrt(2)*exp(11/4)*besselk(0, sqrt(165)/4)/(4*pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution

    """
    # 调用 multivariate_rv 函数来创建多元拉普拉斯分布的随机变量
    return multivariate_rv(MultivariateLaplaceDistribution, name, mu, sigma)

#-------------------------------------------------------------------------------
# 多元 t 分布 -----------------------------------------------------------

class MultivariateTDistribution(JointDistribution):
    _argnames = ('mu', 'shape_mat', 'dof')
    is_Continuous=True

    @property
    def set(self):
        # 获取向量 mu 的维度
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma, v):
        # 检查位置向量和形状矩阵的大小是否匹配
        _value_check(mu.shape[0] == sigma.shape[0],
                     "Size of the location vector and shape matrix are incorrect.")
        # 检查协方差矩阵是否为正定
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite,
                         "The shape matrix must be positive definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.shape_mat
        v = S(self.dof)
        k = S(mu.shape[0])
        # 计算协方差矩阵的逆矩阵
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        x = args - mu
        # 计算多元 t 分布的概率密度函数
        return gamma((k + v)/2)/(gamma(v/2)*(v*pi)**(k/2)*sqrt(det(sigma)))\
        *(1 + 1/v*(x.transpose()*sigma_inv*x)[0])**((-v - k)/2)

# 创建多元 t 分布的随机变量
def MultivariateT(syms, mu, sigma, v):
    """
    Creates a joint random variable with multivariate T-distribution.

    Parameters
    ==========

    syms : A symbol/str
        For identifying the random variable.
    mu : A list/matrix
        Representing the location vector
    sigma : The shape matrix for the distribution
    v : Degrees of freedom

    Examples
    ========

    >>> from sympy.stats import density, MultivariateT
    >>> from sympy import Symbol

    >>> x = Symbol("x")
    >>> X = MultivariateT("x", [1, 1], [[1, 0], [0, 1]], 2)

    >>> density(X)(1, 2)
    2/(9*pi)

    Returns
    =======

    RandomSymbol
        Random variable following Multivariate T-distribution.

    """
    # 调用 multivariate_rv 函数来创建多元 t 分布的随机变量
    return multivariate_rv(MultivariateTDistribution, syms, mu, sigma, v)

#-------------------------------------------------------------------------------
# Multivariate Normal Gamma distribution ---------------------------------------

class NormalGammaDistribution(JointDistribution):
    # 定义多变量正态-伽马分布类，继承自联合分布类

    _argnames = ('mu', 'lamda', 'alpha', 'beta')
    # 类参数列表包括均值mu, lambda, alpha, beta
    is_Continuous=True
    # 设置为连续型分布

    @staticmethod
    def check(mu, lamda, alpha, beta):
        # 静态方法，用于检查分布参数的合法性
        _value_check(mu.is_real, "Location must be real.")
        # 检查均值mu必须为实数
        _value_check(lamda > 0, "Lambda must be positive")
        # 检查lambda必须为正数
        _value_check(alpha > 0, "alpha must be positive")
        # 检查alpha必须为正数
        _value_check(beta > 0, "beta must be positive")
        # 检查beta必须为正数

    @property
    def set(self):
        # 定义属性set，表示随机变量的取值范围
        return S.Reals*Interval(0, S.Infinity)

    def pdf(self, x, tau):
        # 定义概率密度函数pdf，接受参数x和tau
        beta, alpha, lamda = self.beta, self.alpha, self.lamda
        # 从类属性中获取参数beta, alpha, lambda
        mu = self.mu
        # 获取均值mu

        return beta**alpha*sqrt(lamda)/(gamma(alpha)*sqrt(2*pi))*\
        tau**(alpha - S.Half)*exp(-1*beta*tau)*\
        exp(-1*(lamda*tau*(x - mu)**2)/S(2))
        # 返回多变量正态-伽马分布的概率密度函数值

    def _marginal_distribution(self, indices, *sym):
        # 私有方法，计算边缘分布
        if len(indices) == 2:
            return self.pdf(*sym)
            # 对于二维边缘分布，返回概率密度函数值
        if indices[0] == 0:
            # 对于关于x的边缘分布，返回非标准化的学生t分布
            x = sym[0]
            v, mu, sigma = self.alpha - S.Half, self.mu, \
                S(self.beta)/(self.lamda * self.alpha)
            return Lambda(sym, gamma((v + 1)/2)/(gamma(v/2)*sqrt(pi*v)*sigma)*\
                (1 + 1/v*((x - mu)/sigma)**2)**((-v -1)/2))
            # 返回Lambda函数，表示x的边缘分布的概率密度函数
        # 对于关于tau的边缘分布，返回Gamma分布
        from sympy.stats.crv_types import GammaDistribution
        return Lambda(sym, GammaDistribution(self.alpha, self.beta)(sym[0]))
        # 返回Lambda函数，表示tau的边缘分布的概率密度函数

def NormalGamma(sym, mu, lamda, alpha, beta):
    """
    创建具有多变量正态-伽马分布的双变量联合随机变量。

    Parameters
    ==========

    sym : 符号/字符串
        用于标识随机变量。
    mu : 实数
        正态分布的均值
    lamda : 正整数
        联合分布的参数
    alpha : 正整数
        联合分布的参数
    beta : 正整数
        联合分布的参数

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, NormalGamma
    >>> from sympy import symbols

    >>> X = NormalGamma('x', 0, 1, 2, 3)
    >>> y, z = symbols('y z')

    >>> density(X)(y, z)
    9*sqrt(2)*z**(3/2)*exp(-3*z)*exp(-y**2*z/2)/(2*sqrt(pi))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal-gamma_distribution

    """
    return multivariate_rv(NormalGammaDistribution, sym, mu, lamda, alpha, beta)

#-------------------------------------------------------------------------------
# Multivariate Beta/Dirichlet distribution -------------------------------------

class MultivariateBetaDistribution(JointDistribution):
    # 定义多变量贝塔/Dirichlet分布类，继承自联合分布类

    _argnames = ('alpha',)
    # 类参数列表包括alpha

    is_Continuous = True
    # 设置为连续型分布

    @staticmethod
    # 定义一个检查函数，用于验证参数 alpha 至少包含两个类别
    def check(alpha):
        # 调用 _value_check 函数，检查是否传入了至少两个类别
        _value_check(len(alpha) >= 2, "At least two categories should be passed.")
        # 遍历 alpha 中的每个值，检查其是否为正数
        for a_k in alpha:
            _value_check((a_k > 0) != False, "Each concentration parameter should be positive.")

    # 属性装饰器，返回一个 k 维区间 [0, 1]^k
    @property
    def set(self):
        # 获取 alpha 列表的长度 k
        k = len(self.alpha)
        # 返回一个 k 维区间 [0, 1]^k
        return Interval(0, 1)**k

    # 定义一个概率密度函数，接受可变数量的符号参数
    def pdf(self, *syms):
        # 将 self.alpha 赋值给局部变量 alpha
        alpha = self.alpha
        # 计算 B 常数，分子部分是从 alpha 列表中每个元素取 gamma 函数后的乘积，分母部分是 gamma 函数应用于 alpha 总和
        B = Mul.fromiter(map(gamma, alpha))/gamma(Add(*alpha))
        # 返回概率密度函数的计算结果，每个符号参数的乘积除以 B
        return Mul.fromiter(sym**(a_k - 1) for a_k, sym in zip(alpha, syms))/B
# 定义一个名为 MultivariateBeta 的函数，用于创建具有狄利克雷/多变量贝塔分布的连续随机变量
def MultivariateBeta(syms, *alpha):
    """
    Creates a continuous random variable with Dirichlet/Multivariate Beta
    Distribution.

    The density of the Dirichlet distribution can be found at [1].

    Parameters
    ==========

    alpha : Positive real numbers
        Signifies concentration numbers.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, MultivariateBeta, marginal_distribution
    >>> from sympy import Symbol
    >>> a1 = Symbol('a1', positive=True)
    >>> a2 = Symbol('a2', positive=True)
    >>> B = MultivariateBeta('B', [a1, a2])
    >>> C = MultivariateBeta('C', a1, a2)
    >>> x = Symbol('x')
    >>> y = Symbol('y')
    >>> density(B)(x, y)
    x**(a1 - 1)*y**(a2 - 1)*gamma(a1 + a2)/(gamma(a1)*gamma(a2))
    >>> marginal_distribution(C, C[0])(x)
    x**(a1 - 1)*gamma(a1 + a2)/(a2*gamma(a1)*gamma(a2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_distribution
    .. [2] https://mathworld.wolfram.com/DirichletDistribution.html

    """
    # 如果 alpha 的第一个元素不是列表，则将其转换为包含这些元素的元组
    if not isinstance(alpha[0], list):
        alpha = (list(alpha),)
    # 调用 multivariate_rv 函数，使用 MultivariateBetaDistribution 类生成多变量随机变量
    return multivariate_rv(MultivariateBetaDistribution, syms, alpha[0])

# 将 MultivariateBeta 赋值给变量 Dirichlet
Dirichlet = MultivariateBeta

#-------------------------------------------------------------------------------
# Multivariate Ewens distribution ----------------------------------------------

# 定义 MultivariateEwensDistribution 类，继承自 JointDistribution
class MultivariateEwensDistribution(JointDistribution):

    _argnames = ('n', 'theta')
    is_Discrete = True  # 表示该分布是离散的
    is_Continuous = False  # 表示该分布不是连续的

    # 静态方法 check，用于检查参数 n 和 theta 的有效性
    @staticmethod
    def check(n, theta):
        _value_check((n > 0),
                        "sample size should be positive integer.")  # 检查样本大小是否为正整数
        _value_check(theta.is_positive, "mutation rate should be positive.")  # 检查突变率是否为正数

    # 属性方法 set，返回该分布的取值集合
    @property
    def set(self):
        # 如果 n 不是整数，则创建一个符号 i，并返回一个符合条件的集合
        if not isinstance(self.n, Integer):
            i = Symbol('i', integer=True, positive=True)
            return Product(Intersection(S.Naturals0, Interval(0, self.n//i)),
                                    (i, 1, self.n))
        # 如果 n 是整数，则创建一个包含所有可能取值的集合
        prod_set = Range(0, self.n + 1)
        for i in range(2, self.n + 1):
            prod_set *= Range(0, self.n//i + 1)
        return prod_set.flatten()
    # 定义一个名为 pdf 的方法，接受任意数量的参数 syms
    def pdf(self, *syms):
        # 从对象的属性中获取 n 和 theta
        n, theta = self.n, self.theta
        # 检查 self.n 是否是 Integer 类型
        condi = isinstance(self.n, Integer)
        # 如果 syms 的第一个元素不是 IndexedBase 对象且 self.n 不是 Integer 类型，则抛出数值错误
        if not (isinstance(syms[0], IndexedBase) or condi):
            raise ValueError("Please use IndexedBase object for syms as "
                                "the dimension is symbolic")
        # 计算第一个术语，factorial(n) 除以 rf(theta, n)
        term_1 = factorial(n)/rf(theta, n)
        # 如果 self.n 是 Integer 类型
        if condi:
            # 计算第二个术语，使用 Mul.fromiter 方法生成一个乘积表达式
            term_2 = Mul.fromiter(theta**syms[j]/((j+1)**syms[j]*factorial(syms[j]))
                                    for j in range(n))
            # 设置条件，判断 (k + 1) * syms[k] 的和是否等于 n
            cond = Eq(sum((k + 1)*syms[k] for k in range(n)), n)
            # 返回一个分段函数，条件为 cond，返回值为 term_1 * term_2，否则返回 0
            return Piecewise((term_1 * term_2, cond), (0, True))
        # 如果 self.n 不是 Integer 类型
        syms = syms[0]
        # 定义 j 和 k 符号，均为正整数
        j, k = symbols('j, k', positive=True, integer=True)
        # 计算第二个术语，使用 Product 表达式计算乘积
        term_2 = Product(theta**syms[j]/((j+1)**syms[j]*factorial(syms[j])),
                            (j, 0, n - 1))
        # 设置条件，判断 (k + 1) * syms[k] 的和是否等于 n
        cond = Eq(Sum((k + 1)*syms[k], (k, 0, n - 1)), n)
        # 返回一个分段函数，条件为 cond，返回值为 term_1 * term_2，否则返回 0
        return Piecewise((term_1 * term_2, cond), (0, True))
def MultivariateEwens(syms, n, theta):
    """
    创建具有多变量 Ewens 分布的离散随机变量。

    可以在 [1] 中找到该分布的密度函数。

    Parameters
    ==========

    syms : 符号或标识每个分量的符号列表、元组或集合
    n : 正整数
        样本大小或考虑其分区的整数
    theta : 正实数
        突变率

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, marginal_distribution, MultivariateEwens
    >>> from sympy import Symbol
    >>> a1 = Symbol('a1', positive=True)
    >>> a2 = Symbol('a2', positive=True)
    >>> ed = MultivariateEwens('E', 2, 1)
    >>> density(ed)(a1, a2)
    Piecewise((1/(2**a2*factorial(a1)*factorial(a2)), Eq(a1 + 2*a2, 2)), (0, True))
    >>> marginal_distribution(ed, ed[0])(a1)
    Piecewise((1/factorial(a1), Eq(a1, 2)), (0, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ewens%27s_sampling_formula
    .. [2] https://www.jstor.org/stable/24780825
    """
    return multivariate_rv(MultivariateEwensDistribution, syms, n, theta)

#-------------------------------------------------------------------------------
# Generalized Multivariate Log Gamma distribution ------------------------------

class GeneralizedMultivariateLogGammaDistribution(JointDistribution):
    """
    广义多变量对数 Gamma 分布的类。

    Parameters
    ==========

    delta : 范围为 [0, 1] 的常数
    v : 正实数
    lamda : 正数向量
        每个分量必须为正数
    mu : 正数向量
        每个分量必须为正数

    Methods
    =======

    check(delta, v, l, mu) :
        检查参数的有效性，确保它们符合分布的要求。

    set :
        返回定义域为实数集上的 n 维空间。

    pdf(*y) :
        返回给定参数下的概率密度函数值。

    References
    ==========

    .. [1] PDF can be found at [reference]
    """
    
    _argnames = ('delta', 'v', 'lamda', 'mu')
    is_Continuous=True

    def check(self, delta, v, l, mu):
        """
        检查参数的有效性。

        Parameters
        ==========

        delta : [0, 1] 范围内的常数
        v : 正实数
        l : 正数向量
        mu : 正数向量

        Raises
        ======

        ValueError
            如果参数不符合分布的要求。

        """
        _value_check((delta >= 0, delta <= 1), "delta must be in range [0, 1].")
        _value_check((v > 0), "v must be positive")
        for lk in l:
            _value_check((lk > 0), "lamda must be a positive vector.")
        for muk in mu:
            _value_check((muk > 0), "mu must be a positive vector.")
        _value_check(len(l) > 1,"the distribution should have at least"
                                " two random variables.")

    @property
    def set(self):
        """
        返回定义域为实数集上的 n 维空间。

        Returns
        =======

        S.Reals**len(self.lamda)
        """
        return S.Reals**len(self.lamda)

    def pdf(self, *y):
        """
        计算给定参数下的概率密度函数值。

        Parameters
        ==========

        *y : 数字或符号
            分布的参数

        Returns
        =======

        表达式
        """
        d, v, l, mu = self.delta, self.v, self.lamda, self.mu
        n = Symbol('n', negative=False, integer=True)
        k = len(l)
        sterm1 = Pow((1 - d), n)/\
                ((gamma(v + n)**(k - 1))*gamma(v)*gamma(n + 1))
        sterm2 = Mul.fromiter(mui*li**(-v - n) for mui, li in zip(mu, l))
        term1 = sterm1 * sterm2
        sterm3 = (v + n) * sum(mui * yi for mui, yi in zip(mu, y))
        sterm4 = sum(exp(mui * yi)/li for (mui, yi, li) in zip(mu, y, l))
        term2 = exp(sterm3 - sterm4)
        return Pow(d, v) * Sum(term1 * term2, (n, 0, S.Infinity))

def GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu):
    """
    创建具有广义多变量对数 Gamma 分布的联合随机变量。

    可以在 [1] 中找到该分布的联合概率密度函数。

    Parameters
    ==========

    syms : 符号或标识每个分量的符号列表、元组或集合
    delta : [0, 1] 范围内的常数
    v : 正实数
    lamda : 正数向量
        每个分量必须为正数
    mu : 正数向量
        每个分量必须为正数

    Returns
    =======

    RandomSymbol
    """
    # 定义一个函数，创建并返回多元随机变量的实例，基于广义多元对数伽玛分布
    def GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu):
        """
        Parameters
        ==========
    
        syms : List of symbols
            符号列表，用于定义随机变量
        delta : Real number
            常数 delta，通常用于分布中的参数
        v : Positive real number
            正实数 v，用于分布中的参数
        lamda : List of positive real numbers
            正实数列表 lamda，用于分布中的参数
        mu : List of positive real numbers
            正实数列表 mu，用于分布中的参数
    
        Returns
        =======
    
        RandomSymbol
            返回一个随机符号（随机变量的实例）
    
        Examples
        ========
    
        示例代码，展示如何使用 GeneralizedMultivariateLogGamma 分布：
    
        >>> from sympy.stats import density
        >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGamma
        >>> from sympy import symbols, S
        >>> v = 1
        >>> l, mu = [1, 1, 1], [1, 1, 1]
        >>> d = S.Half
        >>> y = symbols('y_1:4', positive=True)
        >>> Gd = GeneralizedMultivariateLogGamma('G', d, v, l, mu)
        >>> density(Gd)(y[0], y[1], y[2])
        Sum(exp((n + 1)*(y_1 + y_2 + y_3) - exp(y_1) - exp(y_2) -
        exp(y_3))/(2**n*gamma(n + 1)**3), (n, 0, oo))/2
    
        References
        ==========
    
        相关文献参考：
        - [1] https://en.wikipedia.org/wiki/Generalized_multivariate_log-gamma_distribution
        - [2] https://www.researchgate.net/publication/234137346_On_a_multivariate_log-gamma_distribution_and_the_use_of_the_distribution_in_the_Bayesian_analysis
    
        Note
        ====
    
        注意事项：
        如果 GeneralizedMultivariateLogGamma 名称太长，可以使用下面的简写方式：
    
        >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGamma as GMVLG
        >>> Gd = GMVLG('G', d, v, l, mu)
    
        如果要传递矩阵 omega 而不是常数 delta，请使用 GeneralizedMultivariateLogGammaOmega。
        """
        return multivariate_rv(GeneralizedMultivariateLogGammaDistribution,
                                syms, delta, v, lamda, mu)
# 定义一个名为 GeneralizedMultivariateLogGammaOmega 的函数，扩展自 GeneralizedMultivariateLogGamma。
def GeneralizedMultivariateLogGammaOmega(syms, omega, v, lamda, mu):
    """
    Extends GeneralizedMultivariateLogGamma.

    Parameters
    ==========

    syms : list/tuple/set of symbols
        用于标识每个组件的符号列表/元组/集合
    omega : A square matrix
           方阵，每个元素必须是相关系数的平方根的绝对值
    v : Positive real number
        正实数
    lamda : List of positive real numbers
            正实数列表
    mu : List of positive real numbers
         正实数列表

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density
    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaOmega
    >>> from sympy import Matrix, symbols, S
    >>> omega = Matrix([[1, S.Half, S.Half], [S.Half, 1, S.Half], [S.Half, S.Half, 1]])
    >>> v = 1
    >>> l, mu = [1, 1, 1], [1, 1, 1]
    >>> G = GeneralizedMultivariateLogGammaOmega('G', omega, v, l, mu)
    >>> y = symbols('y_1:4', positive=True)
    >>> density(G)(y[0], y[1], y[2])
    sqrt(2)*Sum((1 - sqrt(2)/2)**n*exp((n + 1)*(y_1 + y_2 + y_3) - exp(y_1) -
    exp(y_2) - exp(y_3))/gamma(n + 1)**3, (n, 0, oo))/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalized_multivariate_log-gamma_distribution
    .. [2] https://www.researchgate.net/publication/234137346_On_a_multivariate_log-gamma_distribution_and_the_use_of_the_distribution_in_the_Bayesian_analysis

    Notes
    =====

    If the GeneralizedMultivariateLogGammaOmega is too long to type use,

    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaOmega as GMVLGO
    >>> G = GMVLGO('G', omega, v, l, mu)

    """
    # 检查 omega 是方阵且是 Matrix 对象
    _value_check((omega.is_square, isinstance(omega, Matrix)), "omega must be a"
                                                            " square matrix")
    # 检查 omega 中所有值都在 0 到 1 之间（包括边界值）
    for val in omega.values():
        _value_check((val >= 0, val <= 1),
            "all values in matrix must be between 0 and 1(both inclusive).")
    # 检查 omega 的对角线元素全为 1
    _value_check(omega.diagonal().equals(ones(1, omega.shape[0])),
                    "all the elements of diagonal should be 1.")
    # 检查 lamda 和 mu 的长度与 omega 的行数相同
    _value_check((omega.shape[0] == len(lamda), len(lamda) == len(mu)),
                    "lamda, mu should be of same length and omega should "
                    " be of shape (length of lamda, length of mu)")
    # 检查 lamda 的长度大于 1，即分布至少包含两个随机变量
    _value_check(len(lamda) > 1,"the distribution should have at least"
                            " two random variables.")
    # 计算 delta，即 omega 行列式的 (len(lamda) - 1) 次方根
    delta = Pow(Rational(omega.det()), Rational(1, len(lamda) - 1))
    # 返回 GeneralizedMultivariateLogGamma 函数的结果
    return GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu)


#-------------------------------------------------------------------------------
# Multinomial distribution -----------------------------------------------------

class MultinomialDistribution(JointDistribution):
    # Multinomial 分布类，继承自 JointDistribution

    _argnames = ('n', 'p')
    is_Continuous=False
    is_Discrete = True

    @staticmethod
    # 静态方法定义开始
    # 定义检查函数，验证输入参数的合法性
    def check(n, p):
        # 检查试验次数是否为正整数
        _value_check(n > 0,
                        "number of trials must be a positive integer")
        # 遍历每个概率值，确保它们在 [0, 1] 范围内
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1),
                        "probability must be in range [0, 1]")
        # 检查所有概率值的总和是否为 1
        _value_check(Eq(sum(p), 1),
                        "probabilities must sum to 1")

    # 定义类属性方法，返回一个集合，表示可能的离散概率分布取值集合
    @property
    def set(self):
        return Intersection(S.Naturals0, Interval(0, self.n))**len(self.p)

    # 定义概率质量函数（probability density function），计算离散随机变量的概率
    def pdf(self, *x):
        # 获取类的参数 n 和 p
        n, p = self.n, self.p
        # 计算概率质量函数的第一项：n 阶乘除以各 x_k 阶乘的乘积
        term_1 = factorial(n)/Mul.fromiter(factorial(x_k) for x_k in x)
        # 计算概率质量函数的第二项：各 p_k 的 x_k 次幂的乘积
        term_2 = Mul.fromiter(p_k**x_k for p_k, x_k in zip(p, x))
        # 返回分段函数：在总和为 n 时返回第一项与第二项的乘积，否则返回 0
        return Piecewise((term_1 * term_2, Eq(sum(x), n)), (0, True))
# 定义一个函数，创建具有多项分布的离散随机变量
def Multinomial(syms, n, *p):
    """
    Creates a discrete random variable with Multinomial Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n : Positive integer
        Represents number of trials
    p : List of event probabilities
        Must be in the range of $[0, 1]$.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, Multinomial, marginal_distribution
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1, x2, x3', nonnegative=True, integer=True)
    >>> p1, p2, p3 = symbols('p1, p2, p3', positive=True)
    >>> M = Multinomial('M', 3, p1, p2, p3)
    >>> density(M)(x1, x2, x3)
    Piecewise((6*p1**x1*p2**x2*p3**x3/(factorial(x1)*factorial(x2)*factorial(x3)),
    Eq(x1 + x2 + x3, 3)), (0, True))
    >>> marginal_distribution(M, M[0])(x1).subs(x1, 1)
    3*p1*p2**2 + 6*p1*p2*p3 + 3*p1*p3**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multinomial_distribution
    .. [2] https://mathworld.wolfram.com/MultinomialDistribution.html

    """
    # 如果概率列表中的第一个元素不是列表，将其转换为元组
    if not isinstance(p[0], list):
        p = (list(p), )
    # 调用 multivariate_rv 函数创建多变量随机变量，使用 MultinomialDistribution 作为分布类型
    return multivariate_rv(MultinomialDistribution, syms, n, p[0])

#-------------------------------------------------------------------------------
# Negative Multinomial Distribution --------------------------------------------

# 定义负多项式分布的类，继承自 JointDistribution 类
class NegativeMultinomialDistribution(JointDistribution):

    _argnames = ('k0', 'p')
    is_Continuous=False
    is_Discrete = True

    @staticmethod
    def check(k0, p):
        # 检查失败次数 k0 必须是正整数
        _value_check(k0 > 0,
                        "number of failures must be a positive integer")
        # 检查每个概率值 p_k 必须在 [0, 1] 范围内
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1),
                        "probability must be in range [0, 1].")
        # 检查所有概率之和不能大于 1
        _value_check(sum(p) <= 1,
                        "success probabilities must not be greater than 1.")

    @property
    def set(self):
        # 返回 k 的取值范围为 [0, ∞) 的多维空间
        return Range(0, S.Infinity)**len(self.p)

    def pdf(self, *k):
        k0, p = self.k0, self.p
        # 计算概率密度函数的第一部分
        term_1 = (gamma(k0 + sum(k))*(1 - sum(p))**k0)/gamma(k0)
        # 计算概率密度函数的第二部分
        term_2 = Mul.fromiter(pi**ki/factorial(ki) for pi, ki in zip(p, k))
        return term_1 * term_2

# 定义函数，创建具有负多项式分布的离散随机变量
def NegativeMultinomial(syms, k0, *p):
    """
    Creates a discrete random variable with Negative Multinomial Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    k0 : positive integer
        Represents number of failures before the experiment is stopped
    p : List of event probabilities
        Must be in the range of $[0, 1]$

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, NegativeMultinomial, marginal_distribution
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1, x2, x3', nonnegative=True, integer=True)
    >>> p1, p2, p3 = symbols('p1, p2, p3', positive=True)
    >>> NM = NegativeMultinomial('NM', 5, p1, p2, p3)
    >>> density(NM)(x1, x2, x3)
    p1**x1*p2**x2*p3**x3*gamma(k0 + x1 + x2 + x3)/(gamma(k0)*factorial(x1)*factorial(x2)*factorial(x3))
    >>> marginal_distribution(NM, NM[0])(x1).subs(x1, 1)
    p1*p2**2 + 2*p1*p2*p3 + p1*p3**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Negative_multinomial_distribution
    """
    # 如果 p 的第一个元素不是列表，则将其转换为包含单个列表的元组
    if not isinstance(p[0], list):
        p = (list(p), )
    # 调用 multivariate_rv 函数，使用 NegativeMultinomialDistribution 多变量随机变量类创建随机变量
    return multivariate_rv(NegativeMultinomialDistribution, syms, k0, p[0])
```