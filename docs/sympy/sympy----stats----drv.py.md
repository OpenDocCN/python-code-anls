# `D:\src\scipysrc\sympy\sympy\stats\drv.py`

```
from sympy.concrete.summations import (Sum, summation)
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.polys.polytools import poly
from sympy.series.series import series

from sympy.polys.polyerrors import PolynomialError
from sympy.stats.crv import reduce_rational_inequalities_wrap
from sympy.stats.rv import (NamedArgsMixin, SinglePSpace, SingleDomain,
                            random_symbols, PSpace, ConditionalDomain, RandomDomain,
                            ProductDomain, Distribution)
from sympy.stats.symbolic_probability import Probability
from sympy.sets.fancysets import Range, FiniteSet
from sympy.sets.sets import Union
from sympy.sets.contains import Contains
from sympy.utilities import filldedent
from sympy.core.sympify import _sympify

class DiscreteDistribution(Distribution):
    def __call__(self, *args):
        return self.pdf(*args)

class SingleDiscreteDistribution(DiscreteDistribution, NamedArgsMixin):
    """ Discrete distribution of a single variable.

    Serves as superclass for PoissonDistribution etc....

    Provides methods for pdf, cdf, and sampling

    See Also:
        sympy.stats.crv_types.*
    """

    set = S.Integers

    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    @cacheit
    def compute_cdf(self, **kwargs):
        """ Compute the CDF from the PDF.

        Returns a Lambda.
        """
        x = symbols('x', integer=True, cls=Dummy)
        z = symbols('z', real=True, cls=Dummy)
        left_bound = self.set.inf

        # CDF is integral of PDF from left bound to z
        pdf = self.pdf(x)
        cdf = summation(pdf, (x, left_bound, floor(z)), **kwargs)
        # Ensure that CDF left of left_bound is zero using Piecewise function
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    def _cdf(self, x):
        return None

    def cdf(self, x, **kwargs):
        """ Cumulative density function """
        if not kwargs:
            cdf = self._cdf(x)
            if cdf is not None:
                return cdf
        return self.compute_cdf(**kwargs)(x)

    @cacheit
    # 计算特征函数从概率密度函数（PDF）中得出的特征函数。
    # 返回一个 Lambda 函数。
    def compute_characteristic_function(self, **kwargs):
        x, t = symbols('x, t', real=True, cls=Dummy)
        # 获取概率密度函数（PDF）
        pdf = self.pdf(x)
        # 计算特征函数
        cf = summation(exp(I*t*x)*pdf, (x, self.set.inf, self.set.sup))
        return Lambda(t, cf)

    # 私有方法：特征函数
    def _characteristic_function(self, t):
        return None

    # 特征函数方法，接受参数 t 和可选参数 kwargs
    def characteristic_function(self, t, **kwargs):
        """ Characteristic function """
        if not kwargs:
            # 调用私有方法 _characteristic_function 获取特征函数
            cf = self._characteristic_function(t)
            if cf is not None:
                return cf
        # 如果没有直接返回特征函数，则计算并返回特征函数
        return self.compute_characteristic_function(**kwargs)(t)

    # 使用缓存装饰器，计算生成函数
    @cacheit
    def compute_moment_generating_function(self, **kwargs):
        t = Dummy('t', real=True)
        x = Dummy('x', integer=True)
        # 获取概率密度函数（PDF）
        pdf = self.pdf(x)
        # 计算生成函数
        mgf = summation(exp(t*x)*pdf, (x, self.set.inf, self.set.sup))
        return Lambda(t, mgf)

    # 私有方法：生成函数
    def _moment_generating_function(self, t):
        return None

    # 生成函数方法，接受参数 t 和可选参数 kwargs
    def moment_generating_function(self, t, **kwargs):
        if not kwargs:
            # 调用私有方法 _moment_generating_function 获取生成函数
            mgf = self._moment_generating_function(t)
            if mgf is not None:
                return mgf
        # 如果没有直接返回生成函数，则计算并返回生成函数
        return self.compute_moment_generating_function(**kwargs)(t)

    # 使用缓存装饰器，计算分位数
    @cacheit
    def compute_quantile(self, **kwargs):
        """ Compute the Quantile from the PDF.

        Returns a Lambda.
        """
        x = Dummy('x', integer=True)
        p = Dummy('p', real=True)
        left_bound = self.set.inf
        # 获取概率密度函数（PDF）
        pdf = self.pdf(x)
        # 计算累积分布函数（CDF）
        cdf = summation(pdf, (x, left_bound, x), **kwargs)
        set = ((x, p <= cdf), )
        # 返回分位数作为 Lambda 函数
        return Lambda(p, Piecewise(*set))

    # 私有方法：分位数
    def _quantile(self, x):
        return None

    # 分位数方法，接受参数 x 和可选参数 kwargs
    def quantile(self, x, **kwargs):
        """ Cumulative density function """
        if not kwargs:
            # 调用私有方法 _quantile 获取分位数
            quantile = self._quantile(x)
            if quantile is not None:
                return quantile
        # 如果没有直接返回分位数，则计算并返回分位数
        return self.compute_quantile(**kwargs)(x)

    # 计算期望值的方法，表达式为 expr，变量为 var
    # evaluate 参数用于控制是否进行表达式的求值
    def expectation(self, expr, var, evaluate=True, **kwargs):
        """ Expectation of expression over distribution """
        # 如果需要进行求值
        if evaluate:
            try:
                # 尝试将表达式转换为多项式
                p = poly(expr, var)

                # 创建虚拟变量 t
                t = Dummy('t', real=True)

                # 获取生成函数（MGF）
                mgf = self.moment_generating_function(t)
                # 获取多项式的阶数
                deg = p.degree()
                # 计算生成函数的泰勒展开
                taylor = poly(series(mgf, t, 0, deg + 1).removeO(), t)
                result = 0
                # 计算期望值
                for k in range(deg+1):
                    result += p.coeff_monomial(var ** k) * taylor.coeff_monomial(t ** k) * factorial(k)

                return result

            except PolynomialError:
                # 如果表达式不能转换为多项式，则计算表达式关于变量 var 的期望值
                return summation(expr * self.pdf(var),
                                 (var, self.set.inf, self.set.sup), **kwargs)

        else:
            # 如果不需要进行求值，则直接计算表达式关于变量 var 的期望值
            return Sum(expr * self.pdf(var),
                         (var, self.set.inf, self.set.sup), **kwargs)
    # 定义一个特殊方法 __call__，使得实例对象可以像函数一样被调用
    def __call__(self, *args):
        # 调用实例对象的 pdf 方法，并将参数 args 传递给它，返回其结果
        return self.pdf(*args)
class DiscreteDomain(RandomDomain):
    """
    A domain with discrete support with step size one.
    Represented using symbols and Range.
    """
    # 设置一个标志表示这是一个离散域
    is_Discrete = True

class SingleDiscreteDomain(DiscreteDomain, SingleDomain):
    def as_boolean(self):
        # 将域转换为布尔值条件：包含特定符号且在给定集合中
        return Contains(self.symbol, self.set)


class ConditionalDiscreteDomain(DiscreteDomain, ConditionalDomain):
    """
    Domain with discrete support of step size one, that is restricted by
    some condition.
    """
    @property
    def set(self):
        # 获取符号集合，如果有多于一个符号则抛出未实现错误
        rv = self.symbols
        if len(self.symbols) > 1:
            raise NotImplementedError(filldedent('''
                Multivariate conditional domains are not yet implemented.'''))
        rv = list(rv)[0]
        # 对符号应用条件，然后与整个域的集合求交集
        return reduce_rational_inequalities_wrap(self.condition,
            rv).intersect(self.fulldomain.set)


class DiscretePSpace(PSpace):
    is_real = True
    is_Discrete = True

    @property
    def pdf(self):
        # 返回概率密度函数，使用域的符号作为参数
        return self.density(*self.symbols)

    def where(self, condition):
        # 根据条件返回符号变量，确保所有变量都在当前域的符号集中
        rvs = random_symbols(condition)
        assert all(r.symbol in self.symbols for r in rvs)
        if len(rvs) > 1:
            raise NotImplementedError(filldedent('''Multivariate discrete
            random variables are not yet supported.'''))
        # 根据条件缩小域的范围，并创建单一离散域
        conditional_domain = reduce_rational_inequalities_wrap(condition,
            rvs[0])
        conditional_domain = conditional_domain.intersect(self.domain.set)
        return SingleDiscreteDomain(rvs[0].symbol, conditional_domain)

    def probability(self, condition):
        # 概率计算，处理条件的补集情况
        complement = isinstance(condition, Ne)
        if complement:
            condition = Eq(condition.args[0], condition.args[1])
        try:
            # 获取条件下的域，并根据条件返回相应概率
            _domain = self.where(condition).set
            if condition == False or _domain is S.EmptySet:
                return S.Zero
            if condition == True or _domain == self.domain.set:
                return S.One
            prob = self.eval_prob(_domain)
        except NotImplementedError:
            # 处理未实现的情况，使用密度函数进行估算
            from sympy.stats.rv import density
            expr = condition.lhs - condition.rhs
            dens = density(expr)
            if not isinstance(dens, DiscreteDistribution):
                from sympy.stats.drv_types import DiscreteDistributionHandmade
                dens = DiscreteDistributionHandmade(dens)
            z = Dummy('z', real=True)
            space = SingleDiscretePSpace(z, dens)
            prob = space.probability(condition.__class__(space.value, 0))
        if prob is None:
            prob = Probability(condition)
        return prob if not complement else S.One - prob
    # 计算概率密度函数在给定域上的期望值
    def eval_prob(self, _domain):
        # 获取符号集合中的第一个符号
        sym = list(self.symbols)[0]
        # 如果给定域是一个区间（Range）对象
        if isinstance(_domain, Range):
            # 声明一个整数符号 n
            n = symbols('n', integer=True)
            # 解构区间的起始值、结束值和步长
            inf, sup, step = (r for r in _domain.args)
            # 替换概率密度函数中的符号 sym 为 n*step
            summand = ((self.pdf).replace(
              sym, n*step))
            # 对 summand 进行求和，从 inf/step 到 (sup)/step - 1，并求出其值
            rv = summation(summand,
                (n, inf/step, (sup)/step - 1)).doit()
            return rv
        # 如果给定域是一个有限集（FiniteSet）对象
        elif isinstance(_domain, FiniteSet):
            # 以符号 sym 和概率密度函数 self.pdf 创建 Lambda 函数 pdf
            pdf = Lambda(sym, self.pdf)
            # 对给定域中每个元素 x 计算 pdf(x) 的和
            rv = sum(pdf(x) for x in _domain)
            return rv
        # 如果给定域是一个联合集合（Union）对象
        elif isinstance(_domain, Union):
            # 对联合集合中每个部分调用 eval_prob 函数并求和
            rv = sum(self.eval_prob(x) for x in _domain.args)
            return rv

    # 返回符合条件的离散概率空间
    def conditional_space(self, condition):
        # XXX: 将集合转换为元组。Lambda 函数对顺序很敏感，所以应该从集合开始……
        # 创建一个 Lambda 函数，表示密度函数，接受元组形式的符号作为参数
        density = Lambda(tuple(self.symbols), self.pdf/self.probability(condition))
        # 使用 self.values 中的符号与它们的符号标识符来替换条件中的符号
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        # 创建一个条件离散域（ConditionalDiscreteDomain），基于当前域和给定的条件
        domain = ConditionalDiscreteDomain(self.domain, condition)
        # 返回一个新的离散概率空间（DiscretePSpace），基于上述域和密度函数
        return DiscretePSpace(domain, density)
class ProductDiscreteDomain(ProductDomain, DiscreteDomain):
    # 继承自ProductDomain和DiscreteDomain类的ProductDiscreteDomain类

    def as_boolean(self):
        # 将该离散域的所有子域转换为布尔表达式并返回
        return And(*[domain.as_boolean for domain in self.domains])

class SingleDiscretePSpace(DiscretePSpace, SinglePSpace):
    """ Discrete probability space over a single univariate variable """
    # 单变量离散概率空间的定义，继承自DiscretePSpace和SinglePSpace类
    is_real = True
    # 设置is_real属性为True，表示实数域

    @property
    def set(self):
        # 返回分布的集合
        return self.distribution.set

    @property
    def domain(self):
        # 返回单一离散域对象，由符号和集合构成
        return SingleDiscreteDomain(self.symbol, self.set)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method.

        Returns dictionary mapping RandomSymbol to realization value.
        """
        # 内部抽样方法，返回将随机符号映射到实现值的字典
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}

    def compute_expectation(self, expr, rvs=None, evaluate=True, **kwargs):
        # 计算期望值的方法
        rvs = rvs or (self.value,)
        # 如果未指定随机变量集合，则默认为self.value
        if self.value not in rvs:
            # 如果self.value不在随机变量集合中，则直接返回表达式
            return expr

        expr = _sympify(expr)
        # 将表达式转换为SymPy表达式
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})
        # 使用随机变量的符号替换表达式中的随机变量

        x = self.value.symbol
        # 获取self.value的符号
        try:
            return self.distribution.expectation(expr, x, evaluate=evaluate, **kwargs)
            # 尝试计算分布的期望值
        except NotImplementedError:
            return Sum(expr * self.pdf, (x, self.set.inf, self.set.sup), **kwargs)
            # 如果计算期望值的方法未实现，则返回表达式乘以概率密度函数的积分

    def compute_cdf(self, expr, **kwargs):
        # 计算累积分布函数的方法
        if expr == self.value:
            x = Dummy("x", real=True)
            # 创建一个实数域的虚拟变量x
            return Lambda(x, self.distribution.cdf(x, **kwargs))
            # 返回一个函数，该函数计算分布的累积分布函数
        else:
            raise NotImplementedError()
            # 如果表达式不等于self.value，则抛出未实现的错误

    def compute_density(self, expr, **kwargs):
        # 计算密度函数的方法
        if expr == self.value:
            return self.distribution
            # 如果表达式等于self.value，则返回分布对象的密度函数
        raise NotImplementedError()
        # 如果表达式不等于self.value，则抛出未实现的错误

    def compute_characteristic_function(self, expr, **kwargs):
        # 计算特征函数的方法
        if expr == self.value:
            t = Dummy("t", real=True)
            # 创建一个实数域的虚拟变量t
            return Lambda(t, self.distribution.characteristic_function(t, **kwargs))
            # 返回一个函数，该函数计算分布的特征函数
        else:
            raise NotImplementedError()
            # 如果表达式不等于self.value，则抛出未实现的错误

    def compute_moment_generating_function(self, expr, **kwargs):
        # 计算矩生成函数的方法
        if expr == self.value:
            t = Dummy("t", real=True)
            # 创建一个实数域的虚拟变量t
            return Lambda(t, self.distribution.moment_generating_function(t, **kwargs))
            # 返回一个函数，该函数计算分布的矩生成函数
        else:
            raise NotImplementedError()
            # 如果表达式不等于self.value，则抛出未实现的错误

    def compute_quantile(self, expr, **kwargs):
        # 计算分位数的方法
        if expr == self.value:
            p = Dummy("p", real=True)
            # 创建一个实数域的虚拟变量p
            return Lambda(p, self.distribution.quantile(p, **kwargs))
            # 返回一个函数，该函数计算分布的分位数
        else:
            raise NotImplementedError()
            # 如果表达式不等于self.value，则抛出未实现的错误
```