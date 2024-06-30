# `D:\src\scipysrc\sympy\sympy\stats\crv.py`

```
# 导入所需的 SymPy 模块和类
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda, PoleError
from sympy.core.numbers import (I, nan, oo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import poly
from sympy.series.series import series
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Union)
from sympy.solvers.solveset import solveset
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy.stats.rv import (RandomDomain, SingleDomain, ConditionalDomain, is_random,
        ProductDomain, PSpace, SinglePSpace, random_symbols, NamedArgsMixin, Distribution)

# 定义连续随机变量域的基类 ContinuousDomain，继承自 RandomDomain
class ContinuousDomain(RandomDomain):
    """
    A domain with continuous support

    Represented using symbols and Intervals.
    """
    # 设置属性 is_Continuous 为 True，表示该域为连续域
    is_Continuous = True

    # 定义一个抽象方法，子类需要实现具体功能
    def as_boolean(self):
        raise NotImplementedError("Not Implemented for generic Domains")

# 定义单变量连续域的类 SingleContinuousDomain，继承自 ContinuousDomain 和 SingleDomain
class SingleContinuousDomain(ContinuousDomain, SingleDomain):
    """
    A univariate domain with continuous support

    Represented using a single symbol and interval.
    """
    # 计算期望的方法，接受表达式和变量参数
    def compute_expectation(self, expr, variables=None, **kwargs):
        # 如果未提供变量，则默认为域中的符号
        if variables is None:
            variables = self.symbols
        # 如果变量为空，则直接返回表达式
        if not variables:
            return expr
        # 如果给定的变量与域中的符号不相等，则引发 ValueError
        if frozenset(variables) != frozenset(self.symbols):
            raise ValueError("Values should be equal")
        # 假设只有区间，计算表达式的积分期望
        return Integral(expr, (self.symbol, self.set), **kwargs)

    # 将域表示为布尔表达式的方法
    def as_boolean(self):
        return self.set.as_relational(self.symbol)

# 定义多变量连续域的类 ProductContinuousDomain，继承自 ProductDomain 和 ContinuousDomain
class ProductContinuousDomain(ProductDomain, ContinuousDomain):
    """
    A collection of independent domains with continuous support
    """

    # 计算期望的方法，接受表达式和变量参数
    def compute_expectation(self, expr, variables=None, **kwargs):
        # 如果未提供变量，则默认为域中的符号
        if variables is None:
            variables = self.symbols
        # 对每个域进行循环，计算表达式的期望
        for domain in self.domains:
            # 获取域中与变量集交集的符号
            domain_vars = frozenset(variables) & frozenset(domain.symbols)
            # 如果存在这样的变量，则继续递归计算期望
            if domain_vars:
                expr = domain.compute_expectation(expr, domain_vars, **kwargs)
        # 返回计算后的表达式
        return expr

    # 将域表示为布尔表达式的方法
    def as_boolean(self):
        return And(*[domain.as_boolean() for domain in self.domains])

# 定义条件连续域的类 ConditionalContinuousDomain，继承自 ContinuousDomain 和 ConditionalDomain
class ConditionalContinuousDomain(ContinuousDomain, ConditionalDomain):
    """
    A domain with continuous support that has been further restricted by a
    """
    condition such as $x > 3$.
    """

    # 计算期望值的方法，给定表达式 expr 和变量 variables（默认为 None）
    def compute_expectation(self, expr, variables=None, **kwargs):
        # 如果未指定变量，则默认为对象的符号集合 self.symbols
        if variables is None:
            variables = self.symbols
        # 如果变量为空集合，则直接返回表达式 expr
        if not variables:
            return expr
        
        # 提取完整的积分表达式
        fullintgrl = self.fulldomain.compute_expectation(expr, variables)
        # 将积分表达式分离为被积函数和积分限制
        integrand, limits = fullintgrl.function, list(fullintgrl.limits)

        # 初始化条件列表，起始包含对象自身的条件 self.condition
        conditions = [self.condition]
        while conditions:
            cond = conditions.pop()
            # 如果条件是布尔类型
            if cond.is_Boolean:
                # 如果是 And 类型的条件，则将其所有子条件加入条件列表
                if isinstance(cond, And):
                    conditions.extend(cond.args)
                # 如果是 Or 类型的条件，则抛出未实现的异常
                elif isinstance(cond, Or):
                    raise NotImplementedError("Or not implemented here")
            # 如果条件是关系型
            elif cond.is_Relational:
                # 如果是相等关系，向被积函数 integrand 添加适当的 Delta 函数
                if cond.is_Equality:
                    integrand *= DiracDelta(cond.lhs - cond.rhs)
                else:
                    # 获取条件中涉及的自由符号，并与对象的符号集合取交集
                    symbols = cond.free_symbols & set(self.symbols)
                    # 如果涉及多个符号，则抛出未实现的异常
                    if len(symbols) != 1:
                        raise NotImplementedError(
                            "Multivariate Inequalities not yet implemented")
                    # 只能处理单个符号的不等式，获取该符号
                    symbol = symbols.pop()
                    # 在积分限制列表 limits 中找到涉及该符号的限制
                    for i, limit in enumerate(limits):
                        if limit[0] == symbol:
                            # 将条件转换为类似 [0, oo] 的区间
                            cintvl = reduce_rational_inequalities_wrap(
                                cond, symbol)
                            # 将限制转换为类似 [-oo, oo] 的区间
                            lintvl = Interval(limit[1], limit[2])
                            # 将它们相交以得到 [0, oo] 的区间
                            intvl = cintvl.intersect(lintvl)
                            # 将更新后的限制放回 limits 列表中
                            limits[i] = (symbol, intvl.left, intvl.right)
            else:
                # 如果条件既不是关系型也不是布尔型，则抛出类型错误
                raise TypeError(
                    "Condition %s is not a relational or Boolean" % cond)

        # 构造并返回积分对象，包括被积函数、积分限制和其他关键字参数
        return Integral(integrand, *limits, **kwargs)

    # 将对象表示为布尔类型，返回对象的全域部分与条件的 And 运算结果
    def as_boolean(self):
        return And(self.fulldomain.as_boolean(), self.condition)

    # 属性方法，返回条件域的集合表示
    @property
    def set(self):
        # 如果符号集合长度为 1，则返回全域部分与条件约束的交集
        if len(self.symbols) == 1:
            return (self.fulldomain.set & reduce_rational_inequalities_wrap(
                self.condition, tuple(self.symbols)[0]))
        else:
            # 否则抛出未实现的异常，不支持多变量条件域的集合表示
            raise NotImplementedError(
                "Set of Conditional Domain not Implemented")
class ContinuousDistribution(Distribution):
    def __call__(self, *args):
        return self.pdf(*args)


# 定义连续分布类，继承自Distribution类，并重载调用运算符
# 其中，*args是可变数量的参数，调用pdf方法并返回结果



class SingleContinuousDistribution(ContinuousDistribution, NamedArgsMixin):
    """ Continuous distribution of a single variable.

    Explanation
    ===========

    Serves as superclass for Normal/Exponential/UniformDistribution etc....

    Represented by parameters for each of the specific classes.  E.g
    NormalDistribution is represented by a mean and standard deviation.

    Provides methods for pdf, cdf, and sampling.

    See Also
    ========

    sympy.stats.crv_types.*
    """

    set = Interval(-oo, oo)


# 定义单变量连续分布类，继承自ContinuousDistribution类和NamedArgsMixin类
# 该类表示单变量的连续分布

# 类说明：
# - 作为Normal/Exponential/UniformDistribution等特定类的超类
# - 每个具体类都用参数表示，例如NormalDistribution由均值和标准差表示
# - 提供pdf、cdf和抽样方法

# 参见：
# - sympy.stats.crv_types.*

# 类属性：
# - set表示定义在负无穷到正无穷的区间Interval(-oo, oo)



    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)


# 重载构造方法__new__
# 将传入的参数列表args映射为sympify处理后的列表
# 通过Basic类的__new__方法创建一个新的实例



    @staticmethod
    def check(*args):
        pass


# 静态方法check，接受任意数量的参数args
# 方法体为空，即不执行任何操作



    @cacheit
    def compute_cdf(self, **kwargs):
        """ Compute the CDF from the PDF.

        Returns a Lambda.
        """
        x, z = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.set.start

        # CDF is integral of PDF from left bound to z
        pdf = self.pdf(x)
        cdf = integrate(pdf.doit(), (x, left_bound, z), **kwargs)
        # CDF Ensure that CDF left of left_bound is zero
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)


# 计算累积分布函数CDF，基于概率密度函数PDF

# 方法说明：
# - 返回一个Lambda表达式

# 参数：
# - **kwargs：额外的积分参数

# 局部变量：
# - x, z：定义为符号变量，实部为真，类为Dummy
# - left_bound：设置为self.set的起始值

# 方法体：
# - 获取自身的pdf函数并计算其结果
# - 计算pdf从left_bound到z的积分，传递**kwargs参数
# - 使用Piecewise函数确保left_bound左侧的CDF为零
# - 返回Lambda函数，参数为z，表达式为cdf



    def _cdf(self, x):
        return None


# 私有方法_cdf，接受参数x
# 直接返回None，表示不执行任何操作



    def cdf(self, x, **kwargs):
        """ Cumulative density function """
        if len(kwargs) == 0:
            cdf = self._cdf(x)
            if cdf is not None:
                return cdf
        return self.compute_cdf(**kwargs)(x)


# 累积分布函数cdf，接受参数x和**kwargs

# 方法说明：
# - 如果kwargs中没有额外的参数，则调用私有方法_cdf(x)
# - 如果_cdf(x)返回非空值，则直接返回该值
# - 否则，调用compute_cdf(**kwargs)计算CDF，并将x作为参数传递给结果Lambda函数



    @cacheit
    def compute_characteristic_function(self, **kwargs):
        """ Compute the characteristic function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        cf = integrate(exp(I*t*x)*pdf, (x, self.set))
        return Lambda(t, cf)


# 计算特征函数，基于概率密度函数PDF

# 方法说明：
# - 返回一个Lambda表达式

# 参数：
# - **kwargs：额外的积分参数

# 局部变量：
# - x, t：定义为符号变量，实部为真，类为Dummy

# 方法体：
# - 获取自身的pdf函数并计算其结果
# - 计算exp(I*t*x)*pdf关于x在self.set范围内的积分
# - 返回Lambda函数，参数为t，表达式为cf



    def _characteristic_function(self, t):
        return None


# 私有方法_characteristic_function，接受参数t
# 直接返回None，表示不执行任何操作



    def characteristic_function(self, t, **kwargs):
        """ Characteristic function """
        if len(kwargs) == 0:
            cf = self._characteristic_function(t)
            if cf is not None:
                return cf
        return self.compute_characteristic_function(**kwargs)(t)


# 特征函数characteristic_function，接受参数t和**kwargs

# 方法说明：
# - 如果kwargs中没有额外的参数，则调用私有方法_characteristic_function(t)
# - 如果_characteristic_function(t)返回非空值，则直接返回该值
# - 否则，调用compute_characteristic_function(**kwargs)计算特征函数，并将t作为参数传递给结果Lambda函数



    @cacheit
    def compute_moment_generating_function(self, **kwargs):
        """ Compute the moment generating function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        mgf = integrate(exp(t * x) * pdf, (x, self.set))
        return Lambda(t, mgf)


# 计算矩生成函数，基于概率密度函数PDF

# 方法说明：
# - 返回一个Lambda表达式

# 参数：
# - **kwargs：额外的积分参数

# 局部变量：
# - x, t：定义为符号变量，实部为真，类为Dummy

# 方法体：
# - 获取自身的pdf函数并计算其结果
# - 计算exp(t * x) * pdf关于x在self.set范围内的积分
# - 返回Lambda函数，参数为t，表达式为mgf



    def _moment_generating_function(self, t):
        return None


# 私有方法_moment_generating_function，接受参数t
# 直接返回None，表示不执行任何操作


```    
    # 计算矩生成函数
    def moment_generating_function(self, t, **kwargs):
        """ Moment generating function """
        # 如果没有额外参数，直接调用内部的矩生成函数计算
        if not kwargs:
            mgf = self._moment_generating_function(t)
            if mgf is not None:
                return mgf
        # 否则调用指定参数下的矩生成函数计算
        return self.compute_moment_generating_function(**kwargs)(t)

    # 计算分布期望
    def expectation(self, expr, var, evaluate=True, **kwargs):
        """ Expectation of expression over distribution """
        if evaluate:
            try:
                # 尝试将表达式转化为多项式
                p = poly(expr, var)
                if p.is_zero:
                    return S.Zero
                # 定义一个实数虚拟变量t
                t = Dummy('t', real=True)
                # 计算矩生成函数
                mgf = self._moment_generating_function(t)
                if mgf is None:
                    # 如果无法计算矩生成函数，则通过积分计算期望
                    return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
                # 计算多项式的阶数
                deg = p.degree()
                # 对矩生成函数进行泰勒展开
                taylor = poly(series(mgf, t, 0, deg + 1).removeO(), t)
                result = 0
                # 计算期望
                for k in range(deg+1):
                    result += p.coeff_monomial(var ** k) * taylor.coeff_monomial(t ** k) * factorial(k)
                return result
            except PolynomialError:
                # 如果表达式无法转化为多项式，则通过积分计算期望
                return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
        else:
            # 如果不评估表达式，则直接进行积分计算期望
            return Integral(expr * self.pdf(var), (var, self.set), **kwargs)

    # 计算分位数
    @cacheit
    def compute_quantile(self, **kwargs):
        """ Compute the Quantile from the PDF.

        Returns a Lambda.
        """
        # 定义实数虚拟变量x和p
        x, p = symbols('x, p', real=True, cls=Dummy)
        # 获取分布的起始值
        left_bound = self.set.start

        # 计算概率密度函数和累积分布函数
        pdf = self.pdf(x)
        cdf = integrate(pdf, (x, left_bound, x), **kwargs)
        # 求解分位数
        quantile = solveset(cdf - p, x, self.set)
        # 返回一个Lambda函数
        return Lambda(p, Piecewise((quantile, (p >= 0) & (p <= 1) ), (nan, True)))

    # 内部方法，计算分位数
    def _quantile(self, x):
        return None

    # 计算分位数
    def quantile(self, x, **kwargs):
        """ Cumulative density function """
        # 如果没有额外参数，直接调用内部的分位数计算方法
        if len(kwargs) == 0:
            quantile = self._quantile(x)
            if quantile is not None:
                return quantile
        # 否则调用指定参数下的分位数计算方法
        return self.compute_quantile(**kwargs)(x)
class ContinuousPSpace(PSpace):
    """ Continuous Probability Space

    Represents the likelihood of an event space defined over a continuum.

    Represented with a ContinuousDomain and a PDF (Lambda-Like)
    """

    is_Continuous = True  # 标记这个概率空间是连续型的
    is_real = True  # 标记这个概率空间是实数域的

    @property
    def pdf(self):
        return self.density(*self.domain.symbols)
        # 返回这个概率空间的概率密度函数

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        if rvs is None:
            rvs = self.values  # 如果没有指定随机变量，使用默认的值

        else:
            rvs = frozenset(rvs)  # 将随机变量转换为不可变集合

        expr = expr.xreplace({rv: rv.symbol for rv in rvs})
        # 将表达式中的随机变量替换为它们的符号表示

        domain_symbols = frozenset(rv.symbol for rv in rvs)
        # 提取表达式中的符号，并转换为不可变集合

        return self.domain.compute_expectation(self.pdf * expr,
                domain_symbols, **kwargs)
        # 计算期望，使用概率密度函数乘以表达式

    def compute_density(self, expr, **kwargs):
        # 常见情况下的密度函数 Density(X)，其中 X 是 self.values 中的一个
        if expr in self.values:
            # 将所有其他随机符号从密度中边缘化出去
            randomsymbols = tuple(set(self.values) - frozenset([expr]))
            symbols = tuple(rs.symbol for rs in randomsymbols)
            pdf = self.domain.compute_expectation(self.pdf, symbols, **kwargs)
            return Lambda(expr.symbol, pdf)
            # 返回一个 Lambda 表达式，表示密度函数

        z = Dummy('z', real=True)
        return Lambda(z, self.compute_expectation(DiracDelta(expr - z), **kwargs))
        # 返回一个 Lambda 表达式，表示密度函数

    @cacheit
    def compute_cdf(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise ValueError(
                "CDF not well defined on multivariate expressions")
        # 如果定义域不是区间，则抛出错误，因为在多变量表达式上无法定义CDF

        d = self.compute_density(expr, **kwargs)
        x, z = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.domain.set.start
        # 获取定义域的左边界

        # CDF 是从左边界到 z 的概率密度函数的积分
        cdf = integrate(d(x), (x, left_bound, z), **kwargs)
        # 确保左边界左侧的CDF为零
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)
        # 返回一个 Lambda 表达式，表示累积分布函数

    @cacheit
    def compute_characteristic_function(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise NotImplementedError("Characteristic function of multivariate expressions not implemented")
        # 如果定义域不是区间，则抛出未实现错误，因为多变量表达式的特征函数尚未实现

        d = self.compute_density(expr, **kwargs)
        x, t = symbols('x, t', real=True, cls=Dummy)
        cf = integrate(exp(I*t*x)*d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, cf)
        # 返回一个 Lambda 表达式，表示特征函数

    @cacheit
    def compute_moment_generating_function(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise NotImplementedError("Moment generating function of multivariate expressions not implemented")
        # 如果定义域不是区间，则抛出未实现错误，因为多变量表达式的矩生成函数尚未实现

        d = self.compute_density(expr, **kwargs)
        x, t = symbols('x, t', real=True, cls=Dummy)
        mgf = integrate(exp(t * x) * d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, mgf)
        # 返回一个 Lambda 表达式，表示矩生成函数
    # 定义一个方法来计算给定表达式的分位数函数
    def compute_quantile(self, expr, **kwargs):
        # 如果定义域不是区间，则抛出数值错误
        if not self.domain.set.is_Interval:
            raise ValueError(
                "Quantile not well defined on multivariate expressions")

        # 计算给定表达式的累积分布函数
        d = self.compute_cdf(expr, **kwargs)
        
        # 定义实数虚拟变量 x 和正数虚拟变量 p
        x = Dummy('x', real=True)
        p = Dummy('p', positive=True)

        # 解方程 d(x) - p = 0，求解得到分位数
        quantile = solveset(d(x) - p, x, self.set)

        # 返回一个 Lambda 函数，表示分位数函数
        return Lambda(p, quantile)

    # 定义一个方法来计算给定条件下的概率
    def probability(self, condition, **kwargs):
        # 定义实数虚拟变量 z
        z = Dummy('z', real=True)
        
        # 初始化条件反转标志
        cond_inv = False
        
        # 如果条件是 Ne 类型（不等式），转换为相等条件，并标记条件反转
        if isinstance(condition, Ne):
            condition = Eq(condition.args[0], condition.args[1])
            cond_inv = True
        
        # 尝试使用 where 方法获取条件的定义域
        try:
            domain = self.where(condition)
            
            # 在 self.values 中找到对应的随机变量 rv
            rv = [rv for rv in self.values if rv.symbol == domain.symbol][0]
            
            # 计算随机变量 rv 的概率密度函数
            pdf = self.compute_density(rv, **kwargs)
            
            # 如果 domain 是空集或有限集，则返回 0 或 1，视条件反转而定
            if domain.set is S.EmptySet or isinstance(domain.set, FiniteSet):
                return S.Zero if not cond_inv else S.One
            
            # 如果 domain 是 Union 类型，则对每个子区间进行积分求和
            if isinstance(domain.set, Union):
                return sum(
                     Integral(pdf(z), (z, subset), **kwargs) for subset in
                     domain.set.args if isinstance(subset, Interval))
            
            # 否则，在特定域内积分计算概率
            return Integral(pdf(z), (z, domain.set), **kwargs)

        # 处理未实现的情况，通常转化为单变量情况处理
        except NotImplementedError:
            from sympy.stats.rv import density
            
            # 提取条件的左右表达式，计算其密度
            expr = condition.lhs - condition.rhs
            if not is_random(expr):
                dens = self.density
                comp = condition.rhs
            else:
                dens = density(expr, **kwargs)
                comp = 0
            
            # 如果密度不是连续分布，则尝试用手工连续分布处理
            if not isinstance(dens, ContinuousDistribution):
                from sympy.stats.crv_types import ContinuousDistributionHandmade
                dens = ContinuousDistributionHandmade(dens, set=self.domain.set)
            
            # 将问题转化为单变量情况，计算概率
            space = SingleContinuousPSpace(z, dens)
            result = space.probability(condition.__class__(space.value, comp))
            
            # 返回结果或其反转，视条件反转而定
            return result if not cond_inv else S.One - result

    # 定义一个方法来根据条件获取定义域
    def where(self, condition):
        # 提取条件中的随机符号集合
        rvs = frozenset(random_symbols(condition))
        
        # 如果不止一个连续随机变量，则抛出未实现错误
        if not (len(rvs) == 1 and rvs.issubset(self.values)):
            raise NotImplementedError(
                "Multiple continuous random variables not supported")
        
        # 提取唯一的随机变量 rv
        rv = tuple(rvs)[0]
        
        # 使用 reduce_rational_inequalities_wrap 方法计算合并的有理不等式
        interval = reduce_rational_inequalities_wrap(condition, rv)
        
        # 将合并的不等式与定义域进行交集运算
        interval = interval.intersect(self.domain.set)
        
        # 返回单一连续定义域对象
        return SingleContinuousDomain(rv.symbol, interval)
    # 定义一个方法，用于在给定条件下创建条件连续域
    def conditional_space(self, condition, normalize=True, **kwargs):
        # 替换条件中的随机变量为其符号表示，确保符号和数学表达式一致
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        
        # 创建一个基于给定条件的条件连续域对象
        domain = ConditionalContinuousDomain(self.domain, condition)
        
        # 如果需要归一化
        if normalize:
            # 创建变量的克隆，以确保嵌套积分中的变量与外部的变量不同
            # 这确保它们分别且按正确的顺序进行评估
            replacement = {rv: Dummy(str(rv)) for rv in self.symbols}
            
            # 计算条件连续域上的期望值
            norm = domain.compute_expectation(self.pdf, **kwargs)
            
            # 将概率密度函数除以归一化常数，并使用替换变量进行符号替换
            pdf = self.pdf / norm.xreplace(replacement)
            
            # 创建一个 Lambda 函数，参数是条件连续域的符号元组，表达式是概率密度函数
            density = Lambda(tuple(domain.symbols), pdf)
        
        # 返回一个连续概率空间对象，其中包含条件连续域和概率密度函数
        return ContinuousPSpace(domain, density)
class SingleContinuousPSpace(ContinuousPSpace, SinglePSpace):
    """
    A continuous probability space over a single univariate variable.
    
    These consist of a Symbol and a SingleContinuousDistribution
    
    This class is normally accessed through the various random variable
    functions, Normal, Exponential, Uniform, etc....
    """

    @property
    def set(self):
        # 返回分布的定义域
        return self.distribution.set

    @property
    def domain(self):
        # 返回单变量连续域
        return SingleContinuousDomain(sympify(self.symbol), self.set)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method.

        Returns dictionary mapping RandomSymbol to realization value.
        """
        # 返回一个字典，将随机符号映射到样本值
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        rvs = rvs or (self.value,)
        if self.value not in rvs:
            return expr
        
        expr = _sympify(expr)
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})

        x = self.value.symbol
        try:
            # 计算期望值
            return self.distribution.expectation(expr, x, evaluate=evaluate, **kwargs)
        except PoleError:
            # 如果发生极点错误，则返回积分表达式
            return Integral(expr * self.pdf, (x, self.set), **kwargs)

    def compute_cdf(self, expr, **kwargs):
        if expr == self.value:
            # 如果表达式与值相同，则返回分布的累积分布函数的 Lambda 表达式
            z = Dummy("z", real=True)
            return Lambda(z, self.distribution.cdf(z, **kwargs))
        else:
            # 否则调用基类的计算累积分布函数方法
            return ContinuousPSpace.compute_cdf(self, expr, **kwargs)

    def compute_characteristic_function(self, expr, **kwargs):
        if expr == self.value:
            # 如果表达式与值相同，则返回分布的特征函数的 Lambda 表达式
            t = Dummy("t", real=True)
            return Lambda(t, self.distribution.characteristic_function(t, **kwargs))
        else:
            # 否则调用基类的计算特征函数方法
            return ContinuousPSpace.compute_characteristic_function(self, expr, **kwargs)

    def compute_moment_generating_function(self, expr, **kwargs):
        if expr == self.value:
            # 如果表达式与值相同，则返回分布的动差生成函数的 Lambda 表达式
            t = Dummy("t", real=True)
            return Lambda(t, self.distribution.moment_generating_function(t, **kwargs))
        else:
            # 否则调用基类的计算动差生成函数方法
            return ContinuousPSpace.compute_moment_generating_function(self, expr, **kwargs)

    def compute_density(self, expr, **kwargs):
        # https://en.wikipedia.org/wiki/Random_variable#Functions_of_random_variables
        if expr == self.value:
            # 如果表达式与值相同，则返回分布的密度函数
            return self.density
        
        # 定义一个实数 Dummy 变量 y
        y = Dummy('y', real=True)
        
        # 解方程 expr - y = 0，以获得随机变量 self.value 的解集
        gs = solveset(expr - y, self.value, S.Reals)

        # 处理解集类型为 Intersection 的情况
        if isinstance(gs, Intersection):
            if len(gs.args) == 2 and gs.args[0] is S.Reals:
                gs = gs.args[1]
        
        # 如果解集不是有限集，则抛出 ValueError
        if not gs.is_FiniteSet:
            raise ValueError("Can not solve %s for %s" % (expr, self.value))
        
        # 计算随机变量 self.value 的密度函数
        fx = self.compute_density(self.value)
        fy = sum(fx(g) * abs(g.diff(y)) for g in gs)
        
        # 返回 Lambda 表达式，表示随机变量 expr 的密度函数
        return Lambda(y, fy)
    # 定义一个方法 compute_quantile，它接受一个表达式 expr 和其他关键字参数
    def compute_quantile(self, expr, **kwargs):
        # 如果表达式 expr 等于当前对象的值 self.value
        if expr == self.value:
            # 创建一个实数域的虚拟符号 p
            p = Dummy("p", real=True)
            # 返回一个 Lambda 函数，该函数接受 p 作为参数，并调用 self.distribution 的 quantile 方法计算分位数
            return Lambda(p, self.distribution.quantile(p, **kwargs))
        else:
            # 否则，调用父类 ContinuousPSpace 的 compute_quantile 方法处理表达式 expr 和其他关键字参数
            return ContinuousPSpace.compute_quantile(self, expr, **kwargs)
# 定义一个函数来简化不等式条件，返回条件约束后的结果
def _reduce_inequalities(conditions, var, **kwargs):
    try:
        # 调用具体的不等式约化函数，返回约化后的结果
        return reduce_rational_inequalities(conditions, var, **kwargs)
    except PolynomialError:
        # 如果出现多项式错误，抛出值错误并给出失败的条件信息
        raise ValueError("Reduction of condition failed %s\n" % conditions[0])


# 定义一个函数来处理不等式条件的包装，针对不同类型的条件做处理
def reduce_rational_inequalities_wrap(condition, var):
    # 如果条件是关系型的
    if condition.is_Relational:
        # 将条件转化为列表形式，然后调用不等式简化函数进行处理
        return _reduce_inequalities([[condition]], var, relational=False)
    
    # 如果条件是或的逻辑关系
    if isinstance(condition, Or):
        # 对每个子条件分别进行不等式简化，并使用 Union 函数组合结果
        return Union(*[_reduce_inequalities([[arg]], var, relational=False)
            for arg in condition.args])
    
    # 如果条件是与的逻辑关系
    if isinstance(condition, And):
        # 对每个子条件进行不等式简化，然后取交集得到最终结果
        intervals = [_reduce_inequalities([[arg]], var, relational=False)
            for arg in condition.args]
        I = intervals[0]
        for i in intervals:
            I = I.intersect(i)
        return I
```