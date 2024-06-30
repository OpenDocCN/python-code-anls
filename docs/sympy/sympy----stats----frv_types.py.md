# `D:\src\scipysrc\sympy\sympy\stats\frv_types.py`

```
"""
Finite Discrete Random Variables - Prebuilt variable types

Contains
========
FiniteRV
DiscreteUniform
Die
Bernoulli
Coin
Binomial
BetaBinomial
Hypergeometric
Rademacher
IdealSoliton
RobustSoliton
"""


from sympy.core.cache import cacheit  # 导入缓存函数
from sympy.core.function import Lambda  # 导入 Lambda 函数
from sympy.core.numbers import (Integer, Rational)  # 导入整数和有理数类
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt)  # 导入关系运算符类
from sympy.core.singleton import S  # 导入 SymPy 单例对象 S
from sympy.core.symbol import (Dummy, Symbol)  # 导入虚拟符号和符号类
from sympy.core.sympify import sympify  # 导入 sympify 函数
from sympy.functions.combinatorial.factorials import binomial  # 导入二项式系数函数
from sympy.functions.elementary.exponential import log  # 导入对数函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.logic.boolalg import Or  # 导入逻辑或运算符
from sympy.sets.contains import Contains  # 导入包含运算
from sympy.sets.fancysets import Range  # 导入范围集合
from sympy.sets.sets import (Intersection, Interval)  # 导入交集和区间集合
from sympy.functions.special.beta_functions import beta as beta_fn  # 导入贝塔函数
from sympy.stats.frv import (SingleFiniteDistribution,  # 导入单一有限分布类和空间类
                             SingleFinitePSpace)
from sympy.stats.rv import _value_check, Density, is_random  # 导入随机变量值检查函数、密度函数和随机性检查函数
from sympy.utilities.iterables import multiset  # 导入多重集合函数
from sympy.utilities.misc import filldedent  # 导入填充删除的工具函数


__all__ = ['FiniteRV',
'DiscreteUniform',
'Die',
'Bernoulli',
'Coin',
'Binomial',
'BetaBinomial',
'Hypergeometric',
'Rademacher',
'IdealSoliton',
'RobustSoliton',
]

def rv(name, cls, *args, **kwargs):
    args = list(map(sympify, args))  # 将所有参数转换为 SymPy 表达式
    dist = cls(*args)  # 使用参数创建分布对象
    if kwargs.pop('check', True):
        dist.check(*args)  # 检查分布的有效性
    pspace = SingleFinitePSpace(name, dist)  # 创建单一有限概率空间对象
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(name, CompoundDistribution(dist))  # 如果参数中有随机变量，则创建复合概率空间对象
    return pspace.value  # 返回随机变量的值

class FiniteDistributionHandmade(SingleFiniteDistribution):

    @property
    def dict(self):
        return self.args[0]  # 返回分布对象的参数字典

    def pmf(self, x):
        x = Symbol('x')  # 创建符号 x
        return Lambda(x, Piecewise(*(
            [(v, Eq(k, x)) for k, v in self.dict.items()] + [(S.Zero, True)])))  # 返回符号 x 的分布函数

    @property
    def set(self):
        return set(self.dict.keys())  # 返回分布对象的值集合

    @staticmethod
    def check(density):
        for p in density.values():
            _value_check((p >= 0, p <= 1),
                        "Probability at a point must be between 0 and 1.")  # 检查概率密度值是否在 [0, 1] 区间内
        val = sum(density.values())
        _value_check(Eq(val, 1) != S.false, "Total Probability must be 1.")  # 检查总概率是否为 1

def FiniteRV(name, density, **kwargs):
    r"""
    Create a Finite Random Variable given a dict representing the density.

    Parameters
    ==========

    name : Symbol
        Represents name of the random variable.
    density : dict
        Dictionary containing the pdf of finite distribution
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.

    Examples
    ========

    >>> from sympy.stats import FiniteRV, P, E
    # 创建一个字典 `density`，用于表示离散随机变量的概率密度函数
    density = {0: .1, 1: .2, 2: .3, 3: .4}
    
    # 使用概率密度函数 `density` 创建一个离散随机变量对象 `X`
    X = FiniteRV('X', density)
    
    # 计算随机变量 `X` 的期望值（数学期望）
    E(X)
    # 返回值应为 2.00000000000000，表示 `X` 的期望值为 2.0
    
    # 计算随机变量 `X` 大于等于 2 的概率
    P(X >= 2)
    # 返回值应为 0.700000000000000，表示 `X >= 2` 的概率为 0.7
    
    Returns
    =======
    
    RandomSymbol
    
    """
    # 在 `kwargs` 参数中设置 'check' 键，默认为 False，如果存在 'check' 则弹出并用该值替换，默认为 False
    kwargs['check'] = kwargs.pop('check', False)
    # 使用参数调用 `rv` 函数，返回一个随机变量对象
    return rv(name, FiniteDistributionHandmade, density, **kwargs)
class DiscreteUniformDistribution(SingleFiniteDistribution):

    @staticmethod
    def check(*args):
        # 检查输入参数是否有重复
        # 不使用 _value_check，因为这里有一个建议给用户
        if len(set(args)) != len(args):
            # 将参数转换为多重集合
            weights = multiset(args)
            # 参数个数
            n = Integer(len(args))
            # 对于每个元素，计算其权重
            for k in weights:
                weights[k] /= n
            # 抛出值错误，提醒用户输入应为集合，如果需要每个元素不同权重的分布，使用以下形式：
            raise ValueError(filldedent("""
                Repeated args detected but set expected. For a
                distribution having different weights for each
                item use the following:""") + (
                '\nS("FiniteRV(%s, %s)")' % ("'X'", weights)))

    @property
    def p(self):
        # 返回每个元素的概率，假设所有元素等概率
        return Rational(1, len(self.args))

    @property  # type: ignore
    @cacheit
    def dict(self):
        # 返回一个字典，键为元素集合，值为各元素的概率
        return dict.fromkeys(self.set, self.p)

    @property
    def set(self):
        # 返回元素的集合
        return set(self.args)

    def pmf(self, x):
        # 如果 x 在元素集合中，则返回其概率
        if x in self.args:
            return self.p
        else:
            return S.Zero


def DiscreteUniform(name, items):
    r"""
    创建一个代表输入集合的均匀分布的有限随机变量。

    Parameters
    ==========

    items : list/tuple
        要进行均匀分布的项目集合

    Examples
    ========

    >>> from sympy.stats import DiscreteUniform, density
    >>> from sympy import symbols

    >>> X = DiscreteUniform('X', symbols('a b c')) # a, b, c 均等概率分布
    >>> density(X).dict
    {a: 1/3, b: 1/3, c: 1/3}

    >>> Y = DiscreteUniform('Y', list(range(5))) # 在范围内的均匀分布
    >>> density(Y).dict
    {0: 1/5, 1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    .. [2] https://mathworld.wolfram.com/DiscreteUniformDistribution.html

    """
    return rv(name, DiscreteUniformDistribution, *items)


class DieDistribution(SingleFiniteDistribution):
    _argnames = ('sides',)

    @staticmethod
    def check(sides):
        # 检查骰子面数是否为正整数
        _value_check((sides.is_positive, sides.is_integer),
                    "number of sides must be a positive integer.")

    @property
    def is_symbolic(self):
        # 判断骰子面数是否为符号表达式
        return not self.sides.is_number

    @property
    def high(self):
        # 返回骰子面数
        return self.sides

    @property
    def low(self):
        # 返回最小的骰子面数，即1
        return S.One

    @property
    def set(self):
        # 如果骰子面数为符号表达式，则返回0到骰子面数之间的自然数与给定区间的交集
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.sides))
        # 否则返回从1到骰子面数的整数集合
        return set(map(Integer, range(1, self.sides + 1)))

    def pmf(self, x):
        # 将 x 转换为 sympy 表达式
        x = sympify(x)
        # 如果 x 不是数字、符号或随机符号，则抛出值错误
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        # 定义条件，x 应为介于1到骰子面数之间的整数
        cond = Ge(x, 1) & Le(x, self.sides) & Contains(x, S.Integers)
        # 返回概率分布函数，如果 x 符合条件则返回1/骰子面数，否则返回0
        return Piecewise((S.One/self.sides, cond), (S.Zero, True))
# 定义一个函数 Die，用于创建表示公平骰子的有限随机变量

def Die(name, sides=6):
    """
    Create a Finite Random Variable representing a fair die.

    Parameters
    ==========

    sides : Integer
        Represents the number of sides of the Die, by default is 6

    Examples
    ========

    >>> from sympy.stats import Die, density
    >>> from sympy import Symbol

    >>> D6 = Die('D6', 6) # Six sided Die
    >>> density(D6).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}

    >>> D4 = Die('D4', 4) # Four sided Die
    >>> density(D4).dict
    {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}

    >>> n = Symbol('n', positive=True, integer=True)
    >>> Dn = Die('Dn', n) # n sided Die
    >>> density(Dn).dict
    Density(DieDistribution(n))
    >>> density(Dn).dict.subs(n, 4).doit()
    {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}

    Returns
    =======

    RandomSymbol
    """

    # 调用 rv 函数创建随机变量对象，使用 DieDistribution 分布和指定的面数 sides
    return rv(name, DieDistribution, sides)


# 定义一个类 BernoulliDistribution，表示伯努利分布的单一有限分布
class BernoulliDistribution(SingleFiniteDistribution):
    _argnames = ('p', 'succ', 'fail')

    @staticmethod
    def check(p, succ, fail):
        # 检查参数 p 是否在区间 [0, 1] 内
        _value_check((p >= 0, p <= 1),
                    "p should be in range [0, 1].")

    @property
    def set(self):
        # 返回伯努利分布的成功和失败事件的集合
        return {self.succ, self.fail}

    def pmf(self, x):
        # 如果成功和失败事件是符号或字符串类型，则返回对应概率的分段函数
        if isinstance(self.succ, Symbol) and isinstance(self.fail, Symbol):
            return Piecewise((self.p, x == self.succ),
                             (1 - self.p, x == self.fail),
                             (S.Zero, True))
        # 否则返回对应概率的分段函数
        return Piecewise((self.p, Eq(x, self.succ)),
                         (1 - self.p, Eq(x, self.fail)),
                         (S.Zero, True))


# 定义函数 Bernoulli，用于创建表示伯努利过程的有限随机变量
def Bernoulli(name, p, succ=1, fail=0):
    """
    Create a Finite Random Variable representing a Bernoulli process.

    Parameters
    ==========

    p : Rational number between 0 and 1
       Represents probability of success
    succ : Integer/symbol/string
       Represents event of success
    fail : Integer/symbol/string
       Represents event of failure

    Examples
    ========

    >>> from sympy.stats import Bernoulli, density
    >>> from sympy import S

    >>> X = Bernoulli('X', S(3)/4) # 1-0 Bernoulli variable, probability = 3/4
    >>> density(X).dict
    {0: 1/4, 1: 3/4}

    >>> X = Bernoulli('X', S.Half, 'Heads', 'Tails') # A fair coin toss
    >>> density(X).dict
    {Heads: 1/2, Tails: 1/2}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_distribution
    .. [2] https://mathworld.wolfram.com/BernoulliDistribution.html

    """

    # 调用 rv 函数创建随机变量对象，使用 BernoulliDistribution 分布和指定的参数
    return rv(name, BernoulliDistribution, p, succ, fail)


# 定义函数 Coin，用于创建表示硬币抛掷的有限随机变量
def Coin(name, p=S.Half):
    """
    Create a Finite Random Variable representing a Coin toss.

    Parameters
    ==========

    p : Rational Number between 0 and 1
      Represents probability of getting "Heads", by default is Half

    Examples
    ========

    >>> from sympy.stats import Coin, density
    >>> from sympy import Rational

    >>> C = Coin('C') # A fair coin toss

    Returns
    =======

    RandomSymbol
    """

    # 调用 rv 函数创建随机变量对象，使用 BernoulliDistribution 分布和默认的概率 p
    return rv(name, BernoulliDistribution, p)
    # 返回一个随机变量，表示一个硬币投掷的随机变量
    return rv(name, BernoulliDistribution, p, 'H', 'T')
# 定义二项分布的类，继承自SingleFiniteDistribution类
class BinomialDistribution(SingleFiniteDistribution):
    # 定义参数名元组，包括'n', 'p', 'succ', 'fail'
    _argnames = ('n', 'p', 'succ', 'fail')

    # 静态方法：验证参数'n', 'p', 'succ', 'fail'的合法性
    @staticmethod
    def check(n, p, succ, fail):
        # 调用_value_check函数，检查'n'必须是非负整数
        _value_check((n.is_integer, n.is_nonnegative),
                    "'n' must be nonnegative integer.")
        # 调用_value_check函数，检查'p'必须在[0, 1]范围内
        _value_check((p <= 1, p >= 0),
                    "p should be in range [0, 1].")

    # 返回属性：分布的上限为'n'
    @property
    def high(self):
        return self.n

    # 返回属性：分布的下限为0
    @property
    def low(self):
        return S.Zero

    # 返回属性：判断'n'是否为符号表达式
    @property
    def is_symbolic(self):
        return not self.n.is_number

    # 返回属性：返回分布的取值集合
    @property
    def set(self):
        if self.is_symbolic:
            # 如果'n'是符号表达式，返回非负整数和区间[0, n]的交集
            return Intersection(S.Naturals0, Interval(0, self.n))
        # 如果'n'不是符号表达式，返回分布字典的键集合
        return set(self.dict.keys())

    # 概率质量函数，返回给定参数'x'的概率质量值
    def pmf(self, x):
        n, p = self.n, self.p
        x = sympify(x)
        # 如果'x'不是数值、符号或随机符号，抛出值错误异常
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        # 定义条件，判断'x'是否在[0, n]、是否为整数
        cond = Ge(x, 0) & Le(x, n) & Contains(x, S.Integers)
        # 返回分段函数，表示二项分布的概率质量函数
        return Piecewise((binomial(n, x) * p**x * (1 - p)**(n - x), cond), (S.Zero, True))

    # 返回属性：分布的字典表示
    @property  # type: ignore
    @cacheit
    def dict(self):
        if self.is_symbolic:
            # 如果'n'是符号表达式，返回Density对象
            return Density(self)
        # 如果'n'不是符号表达式，返回以k为键，pmf(k)为值的字典
        return {k*self.succ + (self.n-k)*self.fail: self.pmf(k)
                    for k in range(0, self.n + 1)}


# 定义函数Binomial，创建表示二项分布的有限随机变量
def Binomial(name, n, p, succ=1, fail=0):
    r"""
    Create a Finite Random Variable representing a binomial distribution.

    Parameters
    ==========

    n : Positive Integer
      表示试验次数
    p : Rational Number between 0 and 1
      表示成功的概率
    succ : Integer/symbol/string
      表示成功的事件，默认为1
    fail : Integer/symbol/string
      表示失败的事件，默认为0

    Examples
    ========

    >>> from sympy.stats import Binomial, density
    >>> from sympy import S, Symbol

    >>> X = Binomial('X', 4, S.Half) # 四次"硬币翻转"
    >>> density(X).dict
    {0: 1/16, 1: 1/4, 2: 3/8, 3: 1/4, 4: 1/16}

    >>> n = Symbol('n', positive=True, integer=True)
    >>> p = Symbol('p', positive=True)
    >>> X = Binomial('X', n, S.Half) # n次"硬币翻转"
    >>> density(X).dict
    Density(BinomialDistribution(n, 1/2, 1, 0))
    >>> density(X).dict.subs(n, 4).doit()
    {0: 1/16, 1: 1/4, 2: 3/8, 3: 1/4, 4: 1/16}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Binomial_distribution
    .. [2] https://mathworld.wolfram.com/BinomialDistribution.html

    """

    return rv(name, BinomialDistribution, n, p, succ, fail)

#-------------------------------------------------------------------------------
# Beta-binomial distribution ----------------------------------------------------------

# 定义Beta二项分布类，继承自SingleFiniteDistribution类
class BetaBinomialDistribution(SingleFiniteDistribution):
    # 定义参数名元组，包括'n', 'alpha', 'beta'
    _argnames = ('n', 'alpha', 'beta')

    # @staticmethod静态方法定义
    # 定义一个函数 check，用于检查参数 n, alpha, beta 是否符合要求
    def check(n, alpha, beta):
        # 调用 _value_check 函数，检查 n 是否是非负整数
        _value_check((n.is_integer, n.is_nonnegative),
                     "'n' must be nonnegative integer. n = %s." % str(n))
        # 调用 _value_check 函数，检查 alpha 是否大于 0
        _value_check((alpha > 0),
                     "'alpha' must be: alpha > 0 . alpha = %s" % str(alpha))
        # 调用 _value_check 函数，检查 beta 是否大于 0
        _value_check((beta > 0),
                     "'beta' must be: beta > 0 . beta = %s" % str(beta))

    # 定义 high 属性，返回对象自身的 n 属性值
    @property
    def high(self):
        return self.n

    # 定义 low 属性，返回 S.Zero，即符号表达式的零值
    @property
    def low(self):
        return S.Zero

    # 定义 is_symbolic 属性，判断对象自身的 n 属性是否为符号表达式（非数字）
    @property
    def is_symbolic(self):
        return not self.n.is_number

    # 定义 set 属性，根据对象自身的 is_symbolic 属性值返回不同的集合
    @property
    def set(self):
        if self.is_symbolic:
            # 如果是符号表达式，返回自然数的非负整数集合与区间 [0, self.n] 的交集
            return Intersection(S.Naturals0, Interval(0, self.n))
        else:
            # 如果不是符号表达式，返回整数范围 0 到 self.n 的集合
            return set(map(Integer, range(self.n + 1)))

    # 定义 pmf 方法，计算二项分布的概率质量函数值
    def pmf(self, k):
        # 提取对象自身的 n, alpha, beta 属性值
        n, a, b = self.n, self.alpha, self.beta
        # 计算并返回二项分布的概率质量函数值
        return binomial(n, k) * beta_fn(k + a, n - k + b) / beta_fn(a, b)
def BetaBinomial(name, n, alpha, beta):
    r"""
    Create a Finite Random Variable representing a Beta-binomial distribution.

    Parameters
    ==========

    n : Positive Integer
      Represents number of trials
    alpha : Real positive number
    beta : Real positive number

    Examples
    ========

    >>> from sympy.stats import BetaBinomial, density

    >>> X = BetaBinomial('X', 2, 1, 1)
    >>> density(X).dict
    {0: 1/3, 1: 2*beta(2, 2), 2: 1/3}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution
    .. [2] https://mathworld.wolfram.com/BetaBinomialDistribution.html

    """

    # 调用 rv 函数生成 Beta-binomial 分布的随机变量
    return rv(name, BetaBinomialDistribution, n, alpha, beta)


class HypergeometricDistribution(SingleFiniteDistribution):
    _argnames = ('N', 'm', 'n')

    @staticmethod
    def check(n, N, m):
        # 检查参数 N，确保其为非负整数
        _value_check((N.is_integer, N.is_nonnegative),
                     "'N' must be nonnegative integer. N = %s." % str(N))
        # 检查参数 n，确保其为非负整数
        _value_check((n.is_integer, n.is_nonnegative),
                     "'n' must be nonnegative integer. n = %s." % str(n))
        # 检查参数 m，确保其为非负整数
        _value_check((m.is_integer, m.is_nonnegative),
                     "'m' must be nonnegative integer. m = %s." % str(m))

    @property
    def is_symbolic(self):
        # 如果 N, m, n 中有任何一个是符号表达式，则返回 True
        return not all(x.is_number for x in (self.N, self.m, self.n))

    @property
    def high(self):
        # 返回高端值，使用 Piecewise 条件判断
        return Piecewise((self.n, Lt(self.n, self.m) != False), (self.m, True))

    @property
    def low(self):
        # 返回低端值，使用 Piecewise 条件判断
        return Piecewise((0, Gt(0, self.n + self.m - self.N) != False), (self.n + self.m - self.N, True))

    @property
    def set(self):
        N, m, n = self.N, self.m, self.n
        # 如果参数是符号表达式，则返回其可能取值的交集
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(self.low, self.high))
        # 否则返回实际取值范围的集合
        return set(range(max(0, n + m - N), min(n, m) + 1))

    def pmf(self, k):
        N, m, n = self.N, self.m, self.n
        # 计算超几何分布的概率质量函数值
        return S(binomial(m, k) * binomial(N - m, n - k))/binomial(N, n)


def Hypergeometric(name, N, m, n):
    r"""
    Create a Finite Random Variable representing a hypergeometric distribution.

    Parameters
    ==========

    N : Positive Integer
      Represents finite population of size N.
    m : Positive Integer
      Represents number of trials with required feature.
    n : Positive Integer
      Represents numbers of draws.


    Examples
    ========

    >>> from sympy.stats import Hypergeometric, density

    >>> X = Hypergeometric('X', 10, 5, 3) # 10 marbles, 5 white (success), 3 draws
    >>> density(X).dict
    {0: 1/12, 1: 5/12, 2: 5/12, 3: 1/12}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hypergeometric_distribution
    .. [2] https://mathworld.wolfram.com/HypergeometricDistribution.html

    """
    # 调用 rv 函数生成 Hypergeometric 分布的随机变量
    return rv(name, HypergeometricDistribution, N, m, n)


class RademacherDistribution(SingleFiniteDistribution):

    @property
    def is_symbolic(self):
        # 返回是否为符号表达式的布尔值
        pass
    # 定义一个方法 `set`，返回一个包含元素 `-1` 和 `1` 的集合
    def set(self):
        return {-1, 1}

    # 定义一个属性 `pmf`，表示概率质量函数（Probability Mass Function）
    @property
    def pmf(self):
        # 创建一个符号变量 `k`
        k = Dummy('k')
        # 返回一个 Lambda 函数，接受变量 `k`，根据 `k` 的取值返回不同的结果：
        # 当 `k` 等于 `-1` 或 `1` 时返回 `S.Half`，否则返回 `S.Zero`
        return Lambda(k, Piecewise((S.Half, Or(Eq(k, -1), Eq(k, 1))), (S.Zero, True)))
def Rademacher(name):
    r"""
    Create a Finite Random Variable representing a Rademacher distribution.

    Examples
    ========

    >>> from sympy.stats import Rademacher, density

    >>> X = Rademacher('X')
    >>> density(X).dict
    {-1: 1/2, 1: 1/2}

    Returns
    =======

    RandomSymbol

    See Also
    ========

    sympy.stats.Bernoulli

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rademacher_distribution

    """
    # 使用自定义的随机变量函数rv创建一个Rademacher分布的随机变量
    return rv(name, RademacherDistribution)

class IdealSolitonDistribution(SingleFiniteDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
         _value_check(k.is_integer and k.is_positive,
                    "'k' must be a positive integer.")

    @property
    def low(self):
        # 返回最小值为1
        return S.One

    @property
    def high(self):
        # 返回最大值为k
        return self.k

    @property
    def set(self):
        # 返回一个包含从1到k的整数集合
        return set(map(Integer, range(1, self.k + 1)))

    @property # type: ignore
    @cacheit
    def dict(self):
        if self.k.is_Symbol:
            # 如果k是符号，则返回一个Density对象
            return Density(self)
        # 否则生成一个Ideal Soliton分布的概率质量函数字典
        d = {1: Rational(1, self.k)}
        d.update({i: Rational(1, i*(i - 1)) for i in range(2, self.k + 1)})
        return d

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            # 如果x不是数字、符号或随机符号则引发错误
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        # 返回Ideal Soliton分布的概率质量函数
        return Piecewise((1/self.k, cond1), (1/(x*(x - 1)), cond2), (S.Zero, True))

def IdealSoliton(name, k):
    r"""
    Create a Finite Random Variable of Ideal Soliton Distribution

    Parameters
    ==========

    k : Positive Integer
        Represents the number of input symbols in an LT (Luby Transform) code.

    Examples
    ========

    >>> from sympy.stats import IdealSoliton, density, P, E
    >>> sol = IdealSoliton('sol', 5)
    >>> density(sol).dict
    {1: 1/5, 2: 1/2, 3: 1/6, 4: 1/12, 5: 1/20}
    >>> density(sol).set
    {1, 2, 3, 4, 5}

    >>> from sympy import Symbol
    >>> k = Symbol('k', positive=True, integer=True)
    >>> sol = IdealSoliton('sol', k)
    >>> density(sol).dict
    Density(IdealSolitonDistribution(k))
    >>> density(sol).dict.subs(k, 10).doit()
    {1: 1/10, 2: 1/2, 3: 1/6, 4: 1/12, 5: 1/20, 6: 1/30, 7: 1/42, 8: 1/56, 9: 1/72, 10: 1/90}

    >>> E(sol.subs(k, 10))
    7381/2520

    >>> P(sol.subs(k, 4) > 2)
    1/4

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Soliton_distribution#Ideal_distribution
    .. [2] https://pages.cs.wisc.edu/~suman/courses/740/papers/luby02lt.pdf

    """
    # 使用自定义的随机变量函数rv创建一个Ideal Soliton分布的随机变量
    return rv(name, IdealSolitonDistribution, k)

class RobustSolitonDistribution(SingleFiniteDistribution):
    _argnames= ('k', 'delta', 'c')

    @staticmethod
    # 检查参数 k、delta、c 是否符合要求
    def check(k, delta, c):
        # 检查 k 是否为正整数
        _value_check(k.is_integer and k.is_positive,
                    "'k' must be a positive integer")
        # 检查 delta 是否为 (0,1) 区间内的实数
        _value_check(Gt(delta, 0) and Le(delta, 1),
                    "'delta' must be a real number in the interval (0,1)")
        # 检查 c 是否为正实数
        _value_check(c.is_positive,
                    "'c' must be a positive real number.")

    # 计算属性 R
    @property
    def R(self):
        # 返回计算得到的 R 值，使用公式 c * log(k/delta) * sqrt(k)
        return self.c * log(self.k/self.delta) * self.k**0.5

    # 计算属性 Z
    @property
    def Z(self):
        # 初始化变量 z 为 0
        z = 0
        # 循环计算 Z 的求和部分，范围是 1 到 round(k/R)
        for i in Range(1, round(self.k/self.R)):
            z += (1/i)
        # 加上 log(R/delta) 部分的值
        z += log(self.R/self.delta)
        # 返回计算得到的 Z 值
        return 1 + z * self.R/self.k

    # 返回属性 low 的值为 S.One
    @property
    def low(self):
        return S.One

    # 返回属性 high 的值为 k
    @property
    def high(self):
        return self.k

    # 返回属性 set 的值为从 1 到 k 的整数集合
    @property
    def set(self):
        return set(map(Integer, range(1, self.k + 1)))

    # 返回属性 is_symbolic 的值，判断 k、c、delta 是否有非数字的情况
    @property
    def is_symbolic(self):
        return not (self.k.is_number and self.c.is_number and self.delta.is_number)

    # 定义 pmf 方法，计算离散概率质量函数
    def pmf(self, x):
        # 将 x 转换成符号表示
        x = sympify(x)
        # 如果 x 不是数字、符号或随机符号，则抛出 ValueError
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))

        # 定义 rho 的分段函数
        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        rho = Piecewise((Rational(1, self.k), cond1), (Rational(1, x*(x-1)), cond2), (S.Zero, True))

        # 定义 tau 的分段函数
        cond1 = Ge(x, 1) & Le(x, round(self.k/self.R)-1)
        cond2 = Eq(x, round(self.k/self.R))
        tau = Piecewise((self.R/(self.k * x), cond1), (self.R * log(self.R/self.delta)/self.k, cond2), (S.Zero, True))

        # 返回 pmf 的计算结果，即 (rho + tau)/Z
        return (rho + tau)/self.Z
# 定义一个函数 RobustSoliton，用于创建 Robust Soliton 分布的有限随机变量
def RobustSoliton(name, k, delta, c):
    r'''
    Create a Finite Random Variable of Robust Soliton Distribution

    Parameters
    ==========

    k : Positive Integer
        Represents the number of input symbols in an LT (Luby Transform) code.
    delta : Positive Rational Number
            Represents the failure probability. Must be in the interval (0,1).
    c : Positive Rational Number
        Constant of proportionality. Values close to 1 are recommended

    Examples
    ========

    >>> from sympy.stats import RobustSoliton, density, P, E
    >>> robSol = RobustSoliton('robSol', 5, 0.5, 0.01)
    >>> density(robSol).dict
    {1: 0.204253668152708, 2: 0.490631107897393, 3: 0.165210624506162, 4: 0.0834387731899302, 5: 0.0505633404760675}
    >>> density(robSol).set
    {1, 2, 3, 4, 5}

    >>> from sympy import Symbol
    >>> k = Symbol('k', positive=True, integer=True)
    >>> c = Symbol('c', positive=True)
    >>> robSol = RobustSoliton('robSol', k, 0.5, c)
    >>> density(robSol).dict
    Density(RobustSolitonDistribution(k, 0.5, c))
    >>> density(robSol).dict.subs(k, 10).subs(c, 0.03).doit()
    {1: 0.116641095387194, 2: 0.467045731687165, 3: 0.159984123349381, 4: 0.0821431680681869, 5: 0.0505765646770100,
    6: 0.0345781523420719, 7: 0.0253132820710503, 8: 0.0194459129233227, 9: 0.0154831166726115, 10: 0.0126733075238887}

    >>> E(robSol.subs(k, 10).subs(c, 0.05))
    2.91358846104106

    >>> P(robSol.subs(k, 4).subs(c, 0.1) > 2)
    0.243650614389834

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Soliton_distribution#Robust_distribution
    .. [2] https://www.inference.org.uk/mackay/itprnn/ps/588.596.pdf
    .. [3] https://pages.cs.wisc.edu/~suman/courses/740/papers/luby02lt.pdf

    '''
    # 调用 rv 函数，返回一个随机符号，基于 RobustSolitonDistribution 分布
    return rv(name, RobustSolitonDistribution, k, delta, c)
```