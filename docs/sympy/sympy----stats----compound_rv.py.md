# `D:\src\scipysrc\sympy\sympy\stats\compound_rv.py`

```
from sympy.concrete.summations import Sum  # 导入 Sum 类，用于处理求和符号
from sympy.core.basic import Basic  # 导入 Basic 类，基础符号类
from sympy.core.function import Lambda  # 导入 Lambda 函数，用于创建匿名函数
from sympy.core.symbol import Dummy  # 导入 Dummy 符号类，用于创建虚拟符号
from sympy.integrals.integrals import Integral  # 导入 Integral 类，处理积分
from sympy.stats.rv import (NamedArgsMixin, random_symbols, _symbol_converter,
                        PSpace, RandomSymbol, is_random, Distribution)  # 导入多个随机变量相关的类和函数
from sympy.stats.crv import ContinuousDistribution, SingleContinuousPSpace  # 导入连续分布和单一连续概率空间类
from sympy.stats.drv import DiscreteDistribution, SingleDiscretePSpace  # 导入离散分布和单一离散概率空间类
from sympy.stats.frv import SingleFiniteDistribution, SingleFinitePSpace  # 导入有限分布和单一有限概率空间类
from sympy.stats.crv_types import ContinuousDistributionHandmade  # 导入手工连续分布类型
from sympy.stats.drv_types import DiscreteDistributionHandmade  # 导入手工离散分布类型
from sympy.stats.frv_types import FiniteDistributionHandmade  # 导入手工有限分布类型


class CompoundPSpace(PSpace):
    """
    A temporary Probability Space for the Compound Distribution. After
    Marginalization, this returns the corresponding Probability Space of the
    parent distribution.
    """

    def __new__(cls, s, distribution):
        s = _symbol_converter(s)  # 将符号转换为统一格式
        if isinstance(distribution, ContinuousDistribution):
            return SingleContinuousPSpace(s, distribution)  # 如果分布是连续分布，则返回单一连续概率空间对象
        if isinstance(distribution, DiscreteDistribution):
            return SingleDiscretePSpace(s, distribution)  # 如果分布是离散分布，则返回单一离散概率空间对象
        if isinstance(distribution, SingleFiniteDistribution):
            return SingleFinitePSpace(s, distribution)  # 如果分布是有限分布，则返回单一有限概率空间对象
        if not isinstance(distribution, CompoundDistribution):
            raise ValueError("%s should be an isinstance of "
                        "CompoundDistribution"%(distribution))  # 如果不是复合分布类型，则引发值错误异常
        return Basic.__new__(cls, s, distribution)

    @property
    def value(self):
        return RandomSymbol(self.symbol, self)  # 返回一个随机符号对象

    @property
    def symbol(self):
        return self.args[0]  # 返回概率空间的符号

    @property
    def is_Continuous(self):
        return self.distribution.is_Continuous  # 返回是否为连续分布

    @property
    def is_Finite(self):
        return self.distribution.is_Finite  # 返回是否为有限分布

    @property
    def is_Discrete(self):
        return self.distribution.is_Discrete  # 返回是否为离散分布

    @property
    def distribution(self):
        return self.args[1]  # 返回概率空间的分布对象

    @property
    def pdf(self):
        return self.distribution.pdf(self.symbol)  # 返回概率密度函数

    @property
    def set(self):
        return self.distribution.set  # 返回分布的集合

    @property
    def domain(self):
        return self._get_newpspace().domain  # 返回新概率空间的域

    def _get_newpspace(self, evaluate=False):
        x = Dummy('x')  # 创建虚拟符号 x
        parent_dist = self.distribution.args[0]  # 获取父分布
        func = Lambda(x, self.distribution.pdf(x, evaluate))  # 创建以 x 为参数的概率密度函数
        new_pspace = self._transform_pspace(self.symbol, parent_dist, func)  # 获取新的概率空间对象
        if new_pspace is not None:
            return new_pspace  # 如果新概率空间对象存在，则返回它
        message = ("Compound Distribution for %s is not implemented yet" % str(parent_dist))  # 构造未实现错误消息
        raise NotImplementedError(message)  # 抛出未实现错误异常
    def _transform_pspace(self, sym, dist, pdf):
        """
        This function transforms the probability space (pspace) associated with a given distribution.
        It creates a new pspace using a custom pdf (probability density function) based on the symbolic variable sym.

        Parameters:
        sym (symbolic variable): Symbolic variable associated with the distribution.
        dist (Distribution object): Original distribution object to transform.
        pdf (function): Custom probability density function used for transformation.

        Returns:
        SingleContinuousPSpace or SingleDiscretePSpace or SingleFinitePSpace: New pspace object based on the type of original distribution.

        Raises:
        TypeError: If the distribution type is not recognized.

        """
        pdf = Lambda(sym, pdf(sym))  # Define a Lambda function using sym and pdf
        _set = dist.set  # Retrieve the set of values from the distribution

        if isinstance(dist, ContinuousDistribution):
            # Return a new pspace for continuous distributions using a handmade ContinuousDistribution
            return SingleContinuousPSpace(sym, ContinuousDistributionHandmade(pdf, _set))
        elif isinstance(dist, DiscreteDistribution):
            # Return a new pspace for discrete distributions using a handmade DiscreteDistribution
            return SingleDiscretePSpace(sym, DiscreteDistributionHandmade(pdf, _set))
        elif isinstance(dist, SingleFiniteDistribution):
            # Create a density dictionary for finite distributions and return a SingleFinitePSpace
            dens = {k: pdf(k) for k in _set}
            return SingleFinitePSpace(sym, FiniteDistributionHandmade(dens))

    def compute_density(self, expr, *, compound_evaluate=True, **kwargs):
        """
        Compute the probability density function (pdf) for a given expression in the new probability space (pspace).

        Parameters:
        expr (SymPy expression): Expression for which pdf is computed.
        compound_evaluate (bool): Option to perform compound evaluation.
        **kwargs: Additional keyword arguments passed to pspace's compute_density method.

        Returns:
        SymPy expression: Result of computing pdf in the new pspace.

        """
        new_pspace = self._get_newpspace(compound_evaluate)  # Get the new pspace
        expr = expr.subs({self.value: new_pspace.value})  # Substitute value into the expression
        return new_pspace.compute_density(expr, **kwargs)  # Compute density in the new pspace

    def compute_cdf(self, expr, *, compound_evaluate=True, **kwargs):
        """
        Compute the cumulative distribution function (CDF) for a given expression in the new probability space (pspace).

        Parameters:
        expr (SymPy expression): Expression for which CDF is computed.
        compound_evaluate (bool): Option to perform compound evaluation.
        **kwargs: Additional keyword arguments passed to pspace's compute_cdf method.

        Returns:
        SymPy expression: Result of computing CDF in the new pspace.

        """
        new_pspace = self._get_newpspace(compound_evaluate)  # Get the new pspace
        expr = expr.subs({self.value: new_pspace.value})  # Substitute value into the expression
        return new_pspace.compute_cdf(expr, **kwargs)  # Compute CDF in the new pspace

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        """
        Compute the expectation of an expression or random variables in the new probability space (pspace).

        Parameters:
        expr (SymPy expression): Expression for which expectation is computed.
        rvs (SymPy expression or None): Random variables for which expectation is computed.
        evaluate (bool): Option to evaluate the expectation.
        **kwargs: Additional keyword arguments passed to pspace's compute_expectation method.

        Returns:
        SymPy expression: Result of computing expectation in the new pspace.

        """
        new_pspace = self._get_newpspace(evaluate)  # Get the new pspace
        expr = expr.subs({self.value: new_pspace.value})  # Substitute value into the expression
        if rvs:
            rvs = rvs.subs({self.value: new_pspace.value})  # Substitute value into random variables if provided
        if isinstance(new_pspace, SingleFinitePSpace):
            return new_pspace.compute_expectation(expr, rvs, **kwargs)  # Compute expectation for finite pspace
        return new_pspace.compute_expectation(expr, rvs, evaluate, **kwargs)  # Compute expectation for other pspace

    def probability(self, condition, *, compound_evaluate=True, **kwargs):
        """
        Compute the probability of a given condition in the new probability space (pspace).

        Parameters:
        condition (SymPy expression): Condition for which probability is computed.
        compound_evaluate (bool): Option to perform compound evaluation.
        **kwargs: Additional keyword arguments passed to pspace's probability method.

        Returns:
        float: Result of computing probability in the new pspace.

        """
        new_pspace = self._get_newpspace(compound_evaluate)  # Get the new pspace
        condition = condition.subs({self.value: new_pspace.value})  # Substitute value into the condition
        return new_pspace.probability(condition)  # Compute probability in the new pspace

    def conditional_space(self, condition, *, compound_evaluate=True, **kwargs):
        """
        Compute the conditional probability space given a condition in the new probability space (pspace).

        Parameters:
        condition (SymPy expression): Condition for which conditional pspace is computed.
        compound_evaluate (bool): Option to perform compound evaluation.
        **kwargs: Additional keyword arguments passed to pspace's conditional_space method.

        Returns:
        ProbabilitySpace: Conditional pspace object based on the condition in the new pspace.

        """
        new_pspace = self._get_newpspace(compound_evaluate)  # Get the new pspace
        condition = condition.subs({self.value: new_pspace.value})  # Substitute value into the condition
        return new_pspace.conditional_space(condition)  # Compute conditional pspace in the new pspace
# 定义一个名为 CompoundDistribution 的类，继承自 Distribution 和 NamedArgsMixin。
class CompoundDistribution(Distribution, NamedArgsMixin):
    """
    Class for Compound Distributions.

    Parameters
    ==========

    dist : Distribution
        Distribution must contain a random parameter
    """

    # 构造方法，创建一个新的 CompoundDistribution 实例
    def __new__(cls, dist):
        # 检查 dist 是否属于 ContinuousDistribution、SingleFiniteDistribution 或 DiscreteDistribution 类型之一
        if not isinstance(dist, (ContinuousDistribution,
                SingleFiniteDistribution, DiscreteDistribution)):
            # 如果不是上述类型，抛出未实现错误，提醒用户该类型的 Compound Distribution 尚未实现
            message = "Compound Distribution for %s is not implemented yet" % str(dist)
            raise NotImplementedError(message)
        
        # 调用 _compound_check 方法，检查 dist 是否包含随机参数
        if not cls._compound_check(dist):
            # 如果不包含随机参数，直接返回 dist
            return dist
        
        # 否则，调用基类 Basic 的构造方法，创建一个新的 CompoundDistribution 实例
        return Basic.__new__(cls, dist)

    @property
    def set(self):
        # 返回 CompoundDistribution 对象中 dist 的 set 属性值
        return self.args[0].set

    @property
    def is_Continuous(self):
        # 返回 dist 是否为 ContinuousDistribution 类型
        return isinstance(self.args[0], ContinuousDistribution)

    @property
    def is_Finite(self):
        # 返回 dist 是否为 SingleFiniteDistribution 类型
        return isinstance(self.args[0], SingleFiniteDistribution)

    @property
    def is_Discrete(self):
        # 返回 dist 是否为 DiscreteDistribution 类型
        return isinstance(self.args[0], DiscreteDistribution)

    def pdf(self, x, evaluate=False):
        # 获取 dist
        dist = self.args[0]
        # 找到 dist 中的随机变量列表
        randoms = [rv for rv in dist.args if is_random(rv)]
        
        # 如果 dist 是 SingleFiniteDistribution 类型
        if isinstance(dist, SingleFiniteDistribution):
            # 创建一个整数类型的虚拟变量 y
            y = Dummy('y', integer=True, negative=False)
            # 计算 dist 的概率质量函数（pmf）
            expr = dist.pmf(y)
        else:
            # 否则，创建一个虚拟变量 y
            y = Dummy('y')
            # 计算 dist 的概率密度函数（pdf）
            expr = dist.pdf(y)
        
        # 对每个随机变量进行边缘化处理
        for rv in randoms:
            expr = self._marginalise(expr, rv, evaluate)
        
        # 返回 Lambda 函数，将 x 作为参数应用于 expr
        return Lambda(y, expr)(x)

    def _marginalise(self, expr, rv, evaluate):
        # 获取 rv 的概率空间分布和概率密度函数或概率质量函数
        if isinstance(rv.pspace.distribution, SingleFiniteDistribution):
            rv_dens = rv.pspace.distribution.pmf(rv)
        else:
            rv_dens = rv.pspace.distribution.pdf(rv)
        
        # 获取 rv 的定义域
        rv_dom = rv.pspace.domain.set
        
        # 如果 rv 的概率空间是离散的或有限的
        if rv.pspace.is_Discrete or rv.pspace.is_Finite:
            # 计算和式，乘以 rv_dens
            expr = Sum(expr * rv_dens, (rv, rv_dom._inf,
                    rv_dom._sup))
        else:
            # 否则，计算积分，乘以 rv_dens
            expr = Integral(expr * rv_dens, (rv, rv_dom._inf,
                    rv_dom._sup))
        
        # 如果 evaluate 为 True，则对表达式进行求值
        if evaluate:
            return expr.doit()
        
        # 否则，返回未求值的表达式
        return expr

    @classmethod
    def _compound_check(self, dist):
        """
        Checks if the given distribution contains random parameters.
        """
        # 初始化随机变量列表
        randoms = []
        # 遍历 dist 的每个参数
        for arg in dist.args:
            # 将每个参数中的随机变量添加到 randoms 列表中
            randoms.extend(random_symbols(arg))
        
        # 如果随机变量列表的长度为 0，返回 False；否则返回 True
        if len(randoms) == 0:
            return False
        return True
```