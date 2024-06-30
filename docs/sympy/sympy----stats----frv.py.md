# `D:\src\scipysrc\sympy\sympy\stats\frv.py`

```
"""
Finite Discrete Random Variables Module

See Also
========
sympy.stats.frv_types
sympy.stats.rv
sympy.stats.crv
"""
# 导入 itertools 中的 product 函数
from itertools import product

# 导入 SymPy 库中各种需要的模块和类
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (I, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
from sympy.sets.sets import Intersection
from sympy.core.containers import Dict
from sympy.core.logic import Logic
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet
from sympy.stats.rv import (RandomDomain, ProductDomain, ConditionalDomain,
                            PSpace, IndependentProductPSpace, SinglePSpace, random_symbols,
                            sumsets, rv_subs, NamedArgsMixin, Density, Distribution)

# 定义一个类 FiniteDensity，继承自 dict
class FiniteDensity(dict):
    """
    A domain with Finite Density.
    """
    
    # 实现类的可调用行为，根据传入的 item 返回对应的密度值或者 0
    def __call__(self, item):
        """
        Make instance of a class callable.

        If item belongs to current instance of a class, return it.

        Otherwise, return 0.
        """
        item = sympify(item)
        if item in self:
            return self[item]
        else:
            return 0

    # 返回当前实例的字典表示
    @property
    def dict(self):
        """
        Return item as dictionary.
        """
        return dict(self)

# 定义一个类 FiniteDomain，继承自 RandomDomain
class FiniteDomain(RandomDomain):
    """
    A domain with discrete finite support

    Represented using a FiniteSet.
    """
    
    # 设置属性 is_Finite 为 True
    is_Finite = True

    # 返回所有符号构成的 FiniteSet
    @property
    def symbols(self):
        return FiniteSet(sym for sym, val in self.elements)

    # 返回所有元素（即符号和对应值组成的元组）构成的列表
    @property
    def elements(self):
        return self.args[0]

    # 返回当前实例的字典表示，其中每个元素都被转换为 Dict 类型
    @property
    def dict(self):
        return FiniteSet(*[Dict(dict(el)) for el in self.elements])

    # 检查某个元素是否属于当前实例的支持集合中
    def __contains__(self, other):
        return other in self.elements

    # 返回支持集合的迭代器
    def __iter__(self):
        return self.elements.__iter__()

    # 将支持集合表示为布尔表达式
    def as_boolean(self):
        return Or(*[And(*[Eq(sym, val) for sym, val in item]) for item in self])

# 定义一个类 SingleFiniteDomain，继承自 FiniteDomain
class SingleFiniteDomain(FiniteDomain):
    """
    A FiniteDomain over a single symbol/set

    Example: The possibilities of a *single* die roll.
    """
    
    # 构造函数，接受一个符号和一个集合，如果集合不是 FiniteSet 或 Intersection 类型，则转换为 FiniteSet 类型
    def __new__(cls, symbol, set):
        if not isinstance(set, FiniteSet) and \
            not isinstance(set, Intersection):
            set = FiniteSet(*set)
        return Basic.__new__(cls, symbol, set)

    # 返回符号属性
    @property
    def symbol(self):
        return self.args[0]

    # 返回仅包含当前符号的 FiniteSet
    @property
    def symbols(self):
        return FiniteSet(self.symbol)

    # 返回集合属性
    @property
    def set(self):
        return self.args[1]

    # 返回当前实例的字典表示
    @property
    def dict(self):
        return FiniteSet(*[Dict(dict(el)) for el in self.elements])
    # 返回一个包含所有元素的有限集合，每个元素是一个包含符号和集合元素的 frozenset
    def elements(self):
        return FiniteSet(*[frozenset(((self.symbol, elem), )) for elem in self.set])
    
    # 定义迭代器方法，返回一个生成器对象，每个元素是一个包含符号和集合元素的 frozenset
    def __iter__(self):
        return (frozenset(((self.symbol, elem),)) for elem in self.set)
    
    # 判断给定的元素是否存在于对象中
    def __contains__(self, other):
        # 解包其他对象的第一个 frozenset，获取符号和值
        sym, val = tuple(other)[0]
        # 返回符号相同且值存在于对象集合中的布尔值
        return sym == self.symbol and val in self.set
class ProductFiniteDomain(ProductDomain, FiniteDomain):
    """
    A Finite domain consisting of several other FiniteDomains
    
    Example: The possibilities of the rolls of three independent dice
    """

    def __iter__(self):
        # 使用product函数生成多个域的笛卡尔积的迭代器
        proditer = product(*self.domains)
        # 返回每个笛卡尔积元组的和集合的迭代器
        return (sumsets(items) for items in proditer)

    @property
    def elements(self):
        # 返回当前有限域的所有元素作为一个有限集
        return FiniteSet(*self)


class ConditionalFiniteDomain(ConditionalDomain, ProductFiniteDomain):
    """
    A FiniteDomain that has been restricted by a condition
    
    Example: The possibilities of a die roll under the condition that the
    roll is even.
    """

    def __new__(cls, domain, condition):
        """
        Create a new instance of ConditionalFiniteDomain class
        """
        # 如果条件为真，直接返回域本身
        if condition is True:
            return domain
        # 否则，将条件应用于随机变量替换函数rv_subs
        cond = rv_subs(condition)
        return Basic.__new__(cls, domain, cond)

    def _test(self, elem):
        """
        Test the value. If value is boolean, return it. If value is equality
        relational (two objects are equal), return it with left-hand side
        being equal to right-hand side. Otherwise, raise ValueError exception.
        """
        # 用字典elem替换条件中的符号，得到值val
        val = self.condition.xreplace(dict(elem))
        # 如果val是True或False，直接返回
        if val in [True, False]:
            return val
        # 如果val是等式，返回左右两边是否相等的布尔值
        elif val.is_Equality:
            return val.lhs == val.rhs
        # 否则，抛出值错误异常
        raise ValueError("Undecidable if %s" % str(val))

    def __contains__(self, other):
        # 检查other是否在完全域中，并且通过测试函数_test
        return other in self.fulldomain and self._test(other)

    def __iter__(self):
        # 返回完全域中通过测试函数_test的元素的迭代器
        return (elem for elem in self.fulldomain if self._test(elem))

    @property
    def set(self):
        # 如果完全域是单一有限域，则返回满足条件的元素的有限集
        if isinstance(self.fulldomain, SingleFiniteDomain):
            return FiniteSet(*[elem for elem in self.fulldomain.set
                               if frozenset(((self.fulldomain.symbol, elem),)) in self])
        else:
            # 否则，抛出未实现错误
            raise NotImplementedError(
                "Not implemented on multi-dimensional conditional domain")

    def as_boolean(self):
        # 将有限域转换为布尔表达式
        return FiniteDomain.as_boolean(self)


class SingleFiniteDistribution(Distribution, NamedArgsMixin):
    def __new__(cls, *args):
        # 将输入参数映射为符号表达式列表，并创建新实例
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        # 检查参数，但未实现具体功能
        pass

    @property  # type: ignore
    @cacheit
    def dict(self):
        # 如果是符号表达式，返回密度分布，否则返回由概率质量函数pmf生成的字典
        if self.is_symbolic:
            return Density(self)
        return {k: self.pmf(k) for k in self.set}

    def pmf(self, *args):  # to be overridden by specific distribution
        # 抛出未实现错误，需被具体分布覆盖
        raise NotImplementedError()

    @property
    def set(self):  # to be overridden by specific distribution
        # 抛出未实现错误，需被具体分布覆盖
        raise NotImplementedError()

    values = property(lambda self: self.dict.values)
    items = property(lambda self: self.dict.items)
    is_symbolic = property(lambda self: False)
    __iter__ = property(lambda self: self.dict.__iter__)
    __getitem__ = property(lambda self: self.dict.__getitem__)
    # 定义一个特殊方法 __call__()，使得对象可以像函数一样被调用
    def __call__(self, *args):
        # 调用对象的 pmf 方法，并将所有参数传递给它
        return self.pmf(*args)
    
    # 定义一个特殊方法 __contains__()，用于检查对象是否包含另一个对象
    def __contains__(self, other):
        # 检查参数 other 是否存在于对象的 set 属性中，并返回结果
        return other in self.set
#=============================================
#=========  Probability Space  ===============
#=============================================

# 定义有限概率空间类，继承自PSpace类
class FinitePSpace(PSpace):
    """
    A Finite Probability Space

    Represents the probabilities of a finite number of events.
    """
    is_Finite = True  # 设置类属性is_Finite为True，表示这是一个有限概率空间

    # 定义__new__方法，用于创建新的实例
    def __new__(cls, domain, density):
        # 将density字典中的键和值都转换成sympy符号
        density = {sympify(key): sympify(val)
                   for key, val in density.items()}
        public_density = Dict(density)  # 使用sympy的Dict函数创建density的公共副本

        # 调用父类PSpace的__new__方法创建对象
        obj = PSpace.__new__(cls, domain, public_density)
        obj._density = density  # 将密度函数存储在对象的_density属性中
        return obj

    # 定义prob_of方法，用于计算给定元素的概率
    def prob_of(self, elem):
        elem = sympify(elem)  # 将elem转换为sympy符号
        density = self._density  # 获取存储在对象中的密度函数
        # 如果密度函数的键的第一个元素是FiniteSet类型，则返回对应元素的概率，否则返回0
        if isinstance(list(density.keys())[0], FiniteSet):
            return density.get(elem, S.Zero)
        return density.get(tuple(elem)[0][1], S.Zero)

    # 定义where方法，用于返回满足条件的条件有限域
    def where(self, condition):
        assert all(r.symbol in self.symbols for r in random_symbols(condition))
        return ConditionalFiniteDomain(self.domain, condition)

    # 定义compute_density方法，计算表达式的密度函数
    def compute_density(self, expr):
        expr = rv_subs(expr, self.values)  # 替换表达式中的随机变量
        d = FiniteDensity()  # 创建一个空的有限密度对象
        # 遍历有限概率空间的所有元素
        for elem in self.domain:
            val = expr.xreplace(dict(elem))  # 替换表达式中的元素
            prob = self.prob_of(elem)  # 获取元素的概率
            d[val] = d.get(val, S.Zero) + prob  # 更新密度对象中对应值的概率
        return d

    # 使用缓存装饰器@cacheit，定义compute_cdf方法，计算累积分布函数
    @cacheit
    def compute_cdf(self, expr):
        d = self.compute_density(expr)  # 计算表达式的密度函数
        cum_prob = S.Zero  # 初始化累积概率为0
        cdf = []  # 创建一个空的累积分布函数列表
        # 遍历排序后的密度函数中的键
        for key in sorted(d):
            prob = d[key]  # 获取当前键的概率
            cum_prob += prob  # 更新累积概率
            cdf.append((key, cum_prob))  # 将当前键和累积概率添加到累积分布函数列表中

        return dict(cdf)  # 返回累积分布函数的字典表示

    # 使用缓存装饰器@cacheit，定义sorted_cdf方法，返回排序后的累积分布函数
    @cacheit
    def sorted_cdf(self, expr, python_float=False):
        cdf = self.compute_cdf(expr)  # 计算累积分布函数
        items = list(cdf.items())  # 将累积分布函数转换为列表
        # 根据累积概率排序累积分布函数的项目
        sorted_items = sorted(items, key=lambda val_cumprob: val_cumprob[1])
        if python_float:
            # 如果指定要转换为Python浮点数，则转换累积概率的值为浮点数
            sorted_items = [(v, float(cum_prob))
                            for v, cum_prob in sorted_items]
        return sorted_items  # 返回排序后的累积分布函数

    # 使用缓存装饰器@cacheit，定义compute_characteristic_function方法，计算特征函数
    @cacheit
    def compute_characteristic_function(self, expr):
        d = self.compute_density(expr)  # 计算表达式的密度函数
        t = Dummy('t', real=True)  # 创建一个实数虚拟变量t

        # 返回特征函数，使用Lambda函数表示
        return Lambda(t, sum(exp(I*k*t)*v for k,v in d.items()))

    # 使用缓存装饰器@cacheit，定义compute_moment_generating_function方法，计算动差生成函数
    @cacheit
    def compute_moment_generating_function(self, expr):
        d = self.compute_density(expr)  # 计算表达式的密度函数
        t = Dummy('t', real=True)  # 创建一个实数虚拟变量t

        # 返回动差生成函数，使用Lambda函数表示
        return Lambda(t, sum(exp(k*t)*v for k,v in d.items()))
    # 计算随机变量表达式的期望值
    def compute_expectation(self, expr, rvs=None, **kwargs):
        # 如果未指定随机变量集合，则使用默认的 self.values
        rvs = rvs or self.values
        # 将表达式中的随机变量替换为其对应的值
        expr = rv_subs(expr, rvs)
        # 计算每个取值的概率
        probs = [self.prob_of(elem) for elem in self.domain]
        # 根据表达式的类型，生成布尔值列表
        if isinstance(expr, (Logic, Relational)):
            parse_domain = [tuple(elem)[0][1] for elem in self.domain]
            bools = [expr.xreplace(dict(elem)) for elem in self.domain]
        else:
            parse_domain = [expr.xreplace(dict(elem)) for elem in self.domain]
            bools = [True for elem in self.domain]
        # 返回期望值的计算结果，使用 Piecewise 对结果进行分段定义
        return sum(Piecewise((prob * elem, blv), (S.Zero, True))
                for prob, elem, blv in zip(probs, parse_domain, bools))

    # 计算表达式的分位数
    def compute_quantile(self, expr):
        # 计算表达式的累积分布函数
        cdf = self.compute_cdf(expr)
        # 定义一个实数变量 p
        p = Dummy('p', real=True)
        # 设置分段函数定义集合
        set = ((nan, (p < 0) | (p > 1)),)
        # 根据累积分布函数的值设置分段条件
        for key, value in cdf.items():
            set = set + ((key, p <= value), )
        # 返回一个 Lambda 函数，计算表达式的分位数
        return Lambda(p, Piecewise(*set))

    # 计算条件概率
    def probability(self, condition):
        # 提取条件中的符号集合
        cond_symbols = frozenset(rs.symbol for rs in random_symbols(condition))
        # 替换条件中的随机变量
        cond = rv_subs(condition)
        # 检查条件中的符号是否都属于当前随机变量空间的符号集合
        if not cond_symbols.issubset(self.symbols):
            raise ValueError("Cannot compare foreign random symbols, %s"
                             %(str(cond_symbols - self.symbols)))
        # 如果条件是关系表达式且涉及的随机变量不属于当前域的自由符号集合
        if isinstance(condition, Relational) and \
            (not cond.free_symbols.issubset(self.domain.free_symbols)):
            # 提取关系表达式的左侧或右侧符号作为随机变量
            rv = condition.lhs if isinstance(condition.rhs, Symbol) else condition.rhs
            # 返回条件概率的计算结果，使用 Piecewise 对结果进行分段定义
            return sum(Piecewise(
                       (self.prob_of(elem), condition.subs(rv, list(elem)[0][1])),
                       (S.Zero, True)) for elem in self.domain)
        # 返回满足条件的随机变量取值的概率之和
        return sympify(sum(self.prob_of(elem) for elem in self.where(condition)))

    # 计算条件空间
    def conditional_space(self, condition):
        # 根据条件筛选出符合条件的随机变量域
        domain = self.where(condition)
        # 计算条件概率
        prob = self.probability(condition)
        # 根据条件空间和条件概率计算密度函数
        density = {key: val / prob
                for key, val in self._density.items() if domain._test(key)}
        # 返回带有条件密度的有限概率空间对象
        return FinitePSpace(domain, density)

    # 采样方法，返回随机变量到实现值的映射字典
    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        # 使用指定的库进行随机抽样，返回随机变量到实现值的映射字典
        return {self.value: self.distribution.sample(size, library, seed)}
class SingleFinitePSpace(SinglePSpace, FinitePSpace):
    """
    A single finite probability space

    Represents the probabilities of a set of random events that can be
    attributed to a single variable/symbol.

    This class is implemented by many of the standard FiniteRV types such as
    Die, Bernoulli, Coin, etc....
    """

    @property
    def domain(self):
        # 返回由单一有限域对象构成的域
        return SingleFiniteDomain(self.symbol, self.distribution.set)

    @property
    def _is_symbolic(self):
        """
        Helper property to check if the distribution
        of the random variable is having symbolic
        dimension.
        """
        # 检查随机变量的分布是否具有符号维度
        return self.distribution.is_symbolic

    @property
    def distribution(self):
        # 返回该随机变量的分布
        return self.args[1]

    def pmf(self, expr):
        # 返回分布函数的概率质量函数
        return self.distribution.pmf(expr)

    @property  # type: ignore
    @cacheit
    def _density(self):
        # 返回密度函数的字典表示
        return {FiniteSet((self.symbol, val)): prob
                    for val, prob in self.distribution.dict.items()}

    @cacheit
    def compute_characteristic_function(self, expr):
        # 计算特征函数
        if self._is_symbolic:
            d = self.compute_density(expr)
            t = Dummy('t', real=True)
            ki = Dummy('ki')
            # 返回特征函数作为 lambda 表达式
            return Lambda(t, Sum(d(ki)*exp(I*ki*t), (ki, self.args[1].low, self.args[1].high)))
        expr = rv_subs(expr, self.values)
        # 返回有限概率空间对象的特征函数
        return FinitePSpace(self.domain, self.distribution).compute_characteristic_function(expr)

    @cacheit
    def compute_moment_generating_function(self, expr):
        # 计算动差生成函数
        if self._is_symbolic:
            d = self.compute_density(expr)
            t = Dummy('t', real=True)
            ki = Dummy('ki')
            # 返回动差生成函数作为 lambda 表达式
            return Lambda(t, Sum(d(ki)*exp(ki*t), (ki, self.args[1].low, self.args[1].high)))
        expr = rv_subs(expr, self.values)
        # 返回有限概率空间对象的动差生成函数
        return FinitePSpace(self.domain, self.distribution).compute_moment_generating_function(expr)

    def compute_quantile(self, expr):
        # 计算分位数
        if self._is_symbolic:
            # 如果是符号维度的随机变量，抛出未实现错误
            raise NotImplementedError("Computing quantile for random variables "
            "with symbolic dimension because the bounds of searching the required "
            "value is undetermined.")
        expr = rv_subs(expr, self.values)
        # 返回有限概率空间对象的分位数计算结果
        return FinitePSpace(self.domain, self.distribution).compute_quantile(expr)

    def compute_density(self, expr):
        # 计算密度函数
        if self._is_symbolic:
            # 如果是符号维度的随机变量
            rv = list(random_symbols(expr))[0]
            k = Dummy('k', integer=True)
            cond = True if not isinstance(expr, (Relational, Logic)) \
                     else expr.subs(rv, k)
            # 返回密度函数作为 lambda 表达式
            return Lambda(k,
            Piecewise((self.pmf(k), And(k >= self.args[1].low,
            k <= self.args[1].high, cond)), (S.Zero, True)))
        expr = rv_subs(expr, self.values)
        # 返回有限概率空间对象的密度函数
        return FinitePSpace(self.domain, self.distribution).compute_density(expr)
    def compute_cdf(self, expr):
        # 如果随机变量是符号化的
        if self._is_symbolic:
            # 计算给定表达式的密度函数
            d = self.compute_density(expr)
            # 创建虚拟变量 k 和 ki
            k = Dummy('k')
            ki = Dummy('ki')
            # 返回一个 Lambda 函数，表示累积分布函数 (CDF)
            return Lambda(k, Sum(d(ki), (ki, self.args[1].low, k)))
        
        # 如果不是符号化的，则进行替换处理
        expr = rv_subs(expr, self.values)
        # 返回有限概率空间对象的计算累积分布函数结果
        return FinitePSpace(self.domain, self.distribution).compute_cdf(expr)

    def compute_expectation(self, expr, rvs=None, **kwargs):
        # 如果随机变量是符号化的
        if self._is_symbolic:
            # 从表达式中提取随机符号变量
            rv = random_symbols(expr)[0]
            # 创建整数虚拟变量 k
            k = Dummy('k', integer=True)
            # 替换表达式中的随机变量为 k
            expr = expr.subs(rv, k)
            # 确定是否为真条件
            cond = True if not isinstance(expr, (Relational, Logic)) \
                    else expr
            # 创建期望函数表达式
            func = self.pmf(k) * k if cond != True else self.pmf(k) * expr
            # 返回求和结果
            return Sum(Piecewise((func, cond), (S.Zero, True)),
                (k, self.distribution.low, self.distribution.high)).doit()

        # 如果不是符号化的，则进行符号化处理并返回有限概率空间对象的期望值计算结果
        expr = _sympify(expr)
        expr = rv_subs(expr, rvs)
        return FinitePSpace(self.domain, self.distribution).compute_expectation(expr, rvs, **kwargs)

    def probability(self, condition):
        # 如果随机变量是符号化的
        if self._is_symbolic:
            # 抛出未实现的错误，因为当前不支持符号尺寸分布的概率查询
            raise NotImplementedError("Currently, probability queries are not "
            "supported for random variables with symbolic sized distributions.")
        
        # 进行随机变量替换并返回有限概率空间对象的概率计算结果
        condition = rv_subs(condition)
        return FinitePSpace(self.domain, self.distribution).probability(condition)

    def conditional_space(self, condition):
        """
        This method is used for transferring the
        computation to probability method because
        conditional space of random variables with
        symbolic dimensions is currently not possible.
        """
        # 如果随机变量是符号化的
        if self._is_symbolic:
            # 当前方法仅仅返回自身，没有进一步操作
            self
        
        # 获取满足条件的域
        domain = self.where(condition)
        # 计算条件概率
        prob = self.probability(condition)
        # 根据条件概率调整密度函数
        density = {key: val / prob
                for key, val in self._density.items() if domain._test(key)}
        # 返回有限概率空间对象，其密度函数已调整为符合条件的结果
        return FinitePSpace(domain, density)
# 定义一个类，继承自 IndependentProductPSpace 和 FinitePSpace 类
class ProductFinitePSpace(IndependentProductPSpace, FinitePSpace):
    """
    A collection of several independent finite probability spaces
    """

    # 返回该类的 domain 属性，即所有空间的产品域
    @property
    def domain(self):
        return ProductFiniteDomain(*[space.domain for space in self.spaces])

    # 返回该类的 _density 属性，即所有空间的密度函数的乘积
    @property  # type: ignore
    @cacheit
    def _density(self):
        # 使用 product 函数对所有空间的密度函数字典的迭代器进行迭代
        proditer = product(*[iter(space._density.items())
            for space in self.spaces])
        d = {}
        # 遍历迭代器中的项目
        for items in proditer:
            elems, probs = list(zip(*items))
            # 将元素进行求和操作
            elem = sumsets(elems)
            # 将概率进行乘积操作
            prob = Mul(*probs)
            d[elem] = d.get(elem, S.Zero) + prob
        # 返回组合后的密度函数字典
        return Dict(d)

    # 返回该类的 density 属性，即所有空间的密度函数的字典
    @property  # type: ignore
    @cacheit
    def density(self):
        return Dict(self._density)

    # 计算给定条件下的概率
    def probability(self, condition):
        return FinitePSpace.probability(self, condition)

    # 计算给定表达式的密度函数
    def compute_density(self, expr):
        return FinitePSpace.compute_density(self, expr)
```