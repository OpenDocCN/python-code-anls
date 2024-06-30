# `D:\src\scipysrc\sympy\sympy\stats\rv.py`

```
"""
Main Random Variables Module

Defines abstract random variable type.
Contains interfaces for probability space object (PSpace) as well as standard
operators, P, E, sample, density, where, quantile

See Also
========

sympy.stats.crv
sympy.stats.frv
sympy.stats.rv_interface
"""

# 导入必要的模块和函数
from __future__ import annotations
from functools import singledispatch
from math import prod

# 导入 SymPy 核心模块和类
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.indexed import Indexed
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet, ProductSet, Intersection
from sympy.solvers.solveset import solveset
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable

# 设置必要的 doctest 要求
__doctest_requires__ = {('sample',): ['scipy']}

# 定义一个符号 x
x = Symbol('x')

# 定义 singledispatch 装饰器函数，用于判断对象是否是随机变量
@singledispatch
def is_random(x):
    return False

# 注册 Basic 类型的 is_random 函数，递归地检查自由符号中是否包含随机变量
@is_random.register(Basic)
def _(x):
    atoms = x.free_symbols
    return any(is_random(i) for i in atoms)

# 定义 RandomDomain 类，表示随机变量的域
class RandomDomain(Basic):
    """
    Represents a set of variables and the values which they can take.

    See Also
    ========

    sympy.stats.crv.ContinuousDomain
    sympy.stats.frv.FiniteDomain
    """

    is_ProductDomain = False
    is_Finite = False
    is_Continuous = False
    is_Discrete = False

    def __new__(cls, symbols, *args):
        symbols = FiniteSet(*symbols)
        return Basic.__new__(cls, symbols, *args)

    @property
    def symbols(self):
        return self.args[0]

    @property
    def set(self):
        return self.args[1]

    def __contains__(self, other):
        raise NotImplementedError()

    def compute_expectation(self, expr):
        raise NotImplementedError()

# 定义 SingleDomain 类，表示单个变量及其域
class SingleDomain(RandomDomain):
    """
    A single variable and its domain.

    See Also
    ========

    sympy.stats.crv.SingleContinuousDomain
    sympy.stats.frv.SingleFiniteDomain
    """
    def __new__(cls, symbol, set):
        assert symbol.is_Symbol
        return Basic.__new__(cls, symbol, set)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def symbols(self):
        return FiniteSet(self.symbol)
    # 检查对象是否包含特定元素的特殊方法实现
    def __contains__(self, other):
        # 如果 other 不是长度为1的元组，则返回 False
        if len(other) != 1:
            return False
        # 将 other 转换为元组，并解包为符号 sym 和值 val
        sym, val = tuple(other)[0]
        # 返回判断条件：self 的 symbol 属性等于 sym，并且 val 存在于 self 的 set 属性中
        return self.symbol == sym and val in self.set
    def __new__(cls, s, distribution):
        # 将符号转换为内部表示
        s = _symbol_converter(s)
        # 创建一个新的对象实例
        return Basic.__new__(cls, s, distribution)

    @property
    def value(self):
        # 返回一个关联到该概率空间的随机符号
        return RandomSymbol(self.symbol, self)

    @property
    def symbol(self):
        # 返回该概率空间的符号
        return self.args[0]

    @property
    def distribution(self):
        # 返回该概率空间的分布
        return self.args[1]
    # 返回第二个参数，假设是一个分布对象
    def distribution(self):
        return self.args[1]
    
    # 属性装饰器，用于获取当前符号的概率密度函数（PDF）
    def pdf(self):
        # 调用分布对象的 pdf 方法，传入当前符号作为参数，返回该符号的概率密度函数值
        return self.distribution.pdf(self.symbol)
    """
    Random Indexed Symbols represent indexed objects within ProbabilitySpaces in SymPy Expressions.
    They can be either Indexed or Function objects, associated with a ProbabilitySpace determined
    by the pspace property.

    Explanation
    ===========

    Random Indexed Symbols inherit properties from RandomSymbol and extend them to handle indexed
    or function-based symbols within probability spaces.

    Random Indexed Symbols contain pspace and symbol properties.
    The pspace property points to the represented Probability Space.
    The symbol property represents either an Indexed or Function object used within that probability space.

    Properties inherited from RandomSymbol:
    - is_finite: True, indicating it is finite
    - is_symbol: True, indicating it represents a symbol
    - is_Atom: True, indicating it is an atomic symbol
    - _diff_wrt: True, indicating it can be differentiated with respect to its symbol
    - pspace: Property returning the associated Probability Space object
    - symbol: Property returning the represented symbol object
    - name: Property returning the name of the symbol

    Methods inherited from RandomSymbol:
    - _eval_is_positive(): Evaluates if the symbol is positive
    - _eval_is_integer(): Evaluates if the symbol is an integer
    - _eval_is_real(): Evaluates if the symbol or its pspace is real
    - is_commutative: Property indicating if the symbol is commutative
    - free_symbols: Returns a set containing the symbol itself

    Methods specific to Random Indexed Symbol:
    - name: Returns the string representation of the symbol

    An object of the RandomIndexedSymbol type should almost never be created directly by the user.
    They are typically generated by calling convenience functions like Normal, Exponential, etc.
    """
    # 定义一个方法 `key`，用于返回对象的关键信息
    def key(self):
        # 如果对象的符号是一个索引对象
        if isinstance(self.symbol, Indexed):
            # 返回索引对象的第二个参数作为关键信息
            return self.symbol.args[1]
        # 如果对象的符号是一个函数对象
        elif isinstance(self.symbol, Function):
            # 返回函数对象的第一个参数作为关键信息
            return self.symbol.args[0]

    # 定义一个属性 `free_symbols`，返回对象的自由符号集合
    @property
    def free_symbols(self):
        # 获取对象关键信息的自由符号集合
        if self.key.free_symbols:
            free_syms = self.key.free_symbols
            # 将当前对象加入到自由符号集合中
            free_syms.add(self)
            return free_syms
        # 如果关键信息没有自由符号，则返回包含当前对象的集合
        return {self}

    # 定义一个属性 `pspace`，返回对象的第二个参数
    @property
    def pspace(self):
        # 直接返回对象的第二个参数
        return self.args[1]
class RandomMatrixSymbol(RandomSymbol, MatrixSymbol): # type: ignore
    # RandomMatrixSymbol 类继承自 RandomSymbol 和 MatrixSymbol 类，忽略类型检查
    def __new__(cls, symbol, n, m, pspace=None):
        # 调用父类 Basic 的构造函数，创建一个新的实例对象
        n, m = _sympify(n), _sympify(m)
        # 将 n 和 m 转换为 SymPy 符号对象
        symbol = _symbol_converter(symbol)
        # 将 symbol 转换为内部符号表示
        if pspace is None:
            # 如果没有指定概率空间 pspace，则使用默认值 PSpace()
            pspace = PSpace()
        return Basic.__new__(cls, symbol, n, m, pspace)

    symbol = property(lambda self: self.args[0])
    # 定义 symbol 属性，返回实例对象的第一个参数
    pspace = property(lambda self: self.args[3])
    # 定义 pspace 属性，返回实例对象的第四个参数


class ProductPSpace(PSpace):
    """
    Abstract class for representing probability spaces with multiple random
    variables.

    See Also
    ========

    sympy.stats.rv.IndependentProductPSpace
    sympy.stats.joint_rv.JointPSpace
    """
    pass
    # ProductPSpace 类，用于表示具有多个随机变量的概率空间的抽象类


class IndependentProductPSpace(ProductPSpace):
    """
    A probability space resulting from the merger of two independent probability
    spaces.

    Often created using the function, pspace.
    """

    def __new__(cls, *spaces):
        # 创建一个新的 IndependentProductPSpace 实例对象，合并多个独立概率空间
        rs_space_dict = {}
        for space in spaces:
            for value in space.values:
                rs_space_dict[value] = space

        symbols = FiniteSet(*[val.symbol for val in rs_space_dict.keys()])

        # 检查是否有重叠的随机变量
        from sympy.stats.joint_rv import MarginalDistribution
        from sympy.stats.compound_rv import CompoundDistribution
        if len(symbols) < sum(len(space.symbols) for space in spaces if not
         isinstance(space.distribution, (
            CompoundDistribution, MarginalDistribution))):
            raise ValueError("Overlapping Random Variables")

        # 如果所有空间都是有限空间，则使用 ProductFinitePSpace 类
        if all(space.is_Finite for space in spaces):
            from sympy.stats.frv import ProductFinitePSpace
            cls = ProductFinitePSpace

        # 创建一个 Basic 类的新实例对象
        obj = Basic.__new__(cls, *FiniteSet(*spaces))

        return obj

    @property
    def pdf(self):
        # 返回所有空间 pdf 的乘积，并替换为符号表示
        p = Mul(*[space.pdf for space in self.spaces])
        return p.subs({rv: rv.symbol for rv in self.values})

    @property
    def rs_space_dict(self):
        # 返回一个字典，将每个值映射到其所在的空间
        d = {}
        for space in self.spaces:
            for value in space.values:
                d[value] = space
        return d

    @property
    def symbols(self):
        # 返回所有随机变量的符号的有限集合
        return FiniteSet(*[val.symbol for val in self.rs_space_dict.keys()])

    @property
    def spaces(self):
        # 返回所有空间的有限集合
        return FiniteSet(*self.args)

    @property
    def values(self):
        # 返回所有空间值的总和集合
        return sumsets(space.values for space in self.spaces)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        # 计算表达式的期望值，根据给定的随机变量集合 rvs
        rvs = rvs or self.values
        rvs = frozenset(rvs)
        for space in self.spaces:
            expr = space.compute_expectation(expr, rvs & space.values, evaluate=False, **kwargs)
        if evaluate and hasattr(expr, 'doit'):
            return expr.doit(**kwargs)
        return expr

    @property
    def domain(self):
        # 返回空间的乘积域
        return ProductDomain(*[space.domain for space in self.spaces])

    @property
    # 抛出未实现错误，因为 ProductSpaces 中不可用密度
    def density(self):
        raise NotImplementedError("Density not available for ProductSpaces")

    # 对于每个子空间，使用给定的参数从中抽取样本，并返回结果字典
    def sample(self, size=(), library='scipy', seed=None):
        return {k: v for space in self.spaces
            for k, v in space.sample(size=size, library=library, seed=seed).items()}

    # 计算给定条件的概率
    def probability(self, condition, **kwargs):
        cond_inv = False
        if isinstance(condition, Ne):
            # 如果条件是不等式，则转换为等式
            condition = Eq(condition.args[0], condition.args[1])
            cond_inv = True
        elif isinstance(condition, And): # 它们是独立的
            # 如果条件是与逻辑，则返回所有子条件的乘积的概率
            return Mul(*[self.probability(arg) for arg in condition.args])
        elif isinstance(condition, Or): # 它们是独立的
            # 如果条件是或逻辑，则返回所有子条件的和的概率
            return Add(*[self.probability(arg) for arg in condition.args])
        
        # 计算表达式的左右差异
        expr = condition.lhs - condition.rhs
        # 获取表达式中的随机符号
        rvs = random_symbols(expr)
        # 计算表达式的密度函数
        dens = self.compute_density(expr)
        
        # 如果表达式中有任何连续随机变量
        if any(pspace(rv).is_Continuous for rv in rvs):
            from sympy.stats.crv import SingleContinuousPSpace
            from sympy.stats.crv_types import ContinuousDistributionHandmade
            
            # 如果表达式在已知值中
            if expr in self.values:
                # 对密度函数进行边际化，去除其他随机符号
                randomsymbols = tuple(set(self.values) - frozenset([expr]))
                symbols = tuple(rs.symbol for rs in randomsymbols)
                pdf = self.domain.integrate(self.pdf, symbols, **kwargs)
                return Lambda(expr.symbol, pdf)
            
            # 将密度函数处理为连续分布的手工制作类型
            dens = ContinuousDistributionHandmade(dens)
            z = Dummy('z', real=True)
            space = SingleContinuousPSpace(z, dens)
            # 计算给定条件的概率
            result = space.probability(condition.__class__(space.value, 0))
        else:
            from sympy.stats.drv import SingleDiscretePSpace
            from sympy.stats.drv_types import DiscreteDistributionHandmade
            
            # 将密度函数处理为离散分布的手工制作类型
            dens = DiscreteDistributionHandmade(dens)
            z = Dummy('z', integer=True)
            space = SingleDiscretePSpace(z, dens)
            # 计算给定条件的概率
            result = space.probability(condition.__class__(space.value, 0))
        
        # 如果条件是反转的，则返回 1 减去结果
        return result if not cond_inv else S.One - result

    # 计算给定表达式的密度函数
    def compute_density(self, expr, **kwargs):
        # 获取表达式中的随机符号
        rvs = random_symbols(expr)
        
        # 如果表达式中有任何连续随机变量
        if any(pspace(rv).is_Continuous for rv in rvs):
            z = Dummy('z', real=True)
            # 使用直接δ函数计算期望来计算密度函数
            expr = self.compute_expectation(DiracDelta(expr - z), **kwargs)
        else:
            z = Dummy('z', integer=True)
            # 使用Kronecker δ函数计算期望来计算密度函数
            expr = self.compute_expectation(KroneckerDelta(expr, z), **kwargs)
        
        # 返回以 z 为参数的 Lambda 表达式，表示表达式的密度函数
        return Lambda(z, expr)

    # 抛出值错误，因为多变量表达式上的CDF定义不明确
    def compute_cdf(self, expr, **kwargs):
        raise ValueError("CDF not well defined on multivariate expressions")
    # 定义一个方法，用于生成符合条件的随机符号
    def conditional_space(self, condition, normalize=True, **kwargs):
        # 从条件中获取随机符号的列表
        rvs = random_symbols(condition)
        # 将条件中的随机变量替换为其符号表示
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        # 获取每个随机变量的概率空间对象列表
        pspaces = [pspace(rv) for rv in rvs]
        
        # 如果存在连续分布的随机变量，则使用连续概率空间类和条件连续域
        if any(ps.is_Continuous for ps in pspaces):
            from sympy.stats.crv import (ConditionalContinuousDomain,
                ContinuousPSpace)
            space = ContinuousPSpace  # 使用连续概率空间类
            domain = ConditionalContinuousDomain(self.domain, condition)  # 创建条件连续域对象
        # 如果存在离散分布的随机变量，则使用离散概率空间类和条件离散域
        elif any(ps.is_Discrete for ps in pspaces):
            from sympy.stats.drv import (ConditionalDiscreteDomain,
                DiscretePSpace)
            space = DiscretePSpace  # 使用离散概率空间类
            domain = ConditionalDiscreteDomain(self.domain, condition)  # 创建条件离散域对象
        # 如果所有随机变量都是有限的，则使用有限概率空间类
        elif all(ps.is_Finite for ps in pspaces):
            from sympy.stats.frv import FinitePSpace
            return FinitePSpace.conditional_space(self, condition)  # 调用有限概率空间类的条件空间方法
        
        # 如果需要进行归一化处理
        if normalize:
            # 创建用于替换的虚拟符号字典
            replacement  = {rv: Dummy(str(rv)) for rv in self.symbols}
            # 计算概率密度函数的归一化系数
            norm = domain.compute_expectation(self.pdf, **kwargs)
            # 归一化概率密度函数
            pdf = self.pdf / norm.xreplace(replacement)
            # 创建概率密度函数的 Lambda 表达式
            # XXX: 将符号从集合转换为元组。Lambda 表达式对顺序敏感，因此这里不能从集合开始...
            density = Lambda(tuple(domain.symbols), pdf)

        # 返回创建的概率空间对象和其对应的概率密度函数
        return space(domain, density)
class ProductDomain(RandomDomain):
    """
    A domain resulting from the merger of two independent domains.

    See Also
    ========
    sympy.stats.crv.ProductContinuousDomain
    sympy.stats.frv.ProductFiniteDomain
    """
    is_ProductDomain = True  # 标记此类为ProductDomain类的实例

    def __new__(cls, *domains):
        # Flatten any product of products
        domains2 = []
        for domain in domains:
            if not domain.is_ProductDomain:
                domains2.append(domain)  # 如果domain不是ProductDomain，则直接添加到domains2中
            else:
                domains2.extend(domain.domains)  # 如果是ProductDomain，则扩展domains2为其包含的所有子域
        domains2 = FiniteSet(*domains2)  # 将domains2转换为一个FiniteSet集合

        if all(domain.is_Finite for domain in domains2):
            from sympy.stats.frv import ProductFiniteDomain
            cls = ProductFiniteDomain  # 如果所有的domain都是Finite，则选择ProductFiniteDomain类
        if all(domain.is_Continuous for domain in domains2):
            from sympy.stats.crv import ProductContinuousDomain
            cls = ProductContinuousDomain  # 如果所有的domain都是Continuous，则选择ProductContinuousDomain类
        if all(domain.is_Discrete for domain in domains2):
            from sympy.stats.drv import ProductDiscreteDomain
            cls = ProductDiscreteDomain  # 如果所有的domain都是Discrete，则选择ProductDiscreteDomain类

        return Basic.__new__(cls, *domains2)  # 调用父类Basic的构造方法创建一个新实例

    @property
    def sym_domain_dict(self):
        return {symbol: domain for domain in self.domains
                                 for symbol in domain.symbols}  # 返回一个字典，将每个符号映射到其对应的域

    @property
    def symbols(self):
        return FiniteSet(*[sym for domain in self.domains
                               for sym in domain.symbols])  # 返回所有域中的符号组成的FiniteSet集合

    @property
    def domains(self):
        return self.args  # 返回此ProductDomain实例的所有参数（域）

    @property
    def set(self):
        return ProductSet(*(domain.set for domain in self.domains))  # 返回此ProductDomain实例的所有域的乘积集合

    def __contains__(self, other):
        # Split event into each subdomain
        for domain in self.domains:
            # Collect the parts of this event which associate to this domain
            elem = frozenset([item for item in other
                              if sympify(domain.symbols.contains(item[0]))
                              is S.true])  # 使用domain中的符号检查other中的元素，并形成一个frozenset集合
            # Test this sub-event
            if elem not in domain:
                return False  # 如果elem不在domain中，则返回False
        # All subevents passed
        return True  # 所有子事件都通过，则返回True

    def as_boolean(self):
        return And(*[domain.as_boolean() for domain in self.domains])  # 返回所有域的布尔表达式的And逻辑与


def random_symbols(expr):
    """
    Returns all RandomSymbols within a SymPy Expression.
    """
    atoms = getattr(expr, 'atoms', None)
    if atoms is not None:
        comp = lambda rv: rv.symbol.name
        l = list(atoms(RandomSymbol))  # 获取表达式中的所有RandomSymbol
        return sorted(l, key=comp)  # 返回排序后的RandomSymbol列表
    else:
        return []


def pspace(expr):
    """
    Returns the underlying Probability Space of a random expression.

    For internal use.

    Examples
    ========

    >>> from sympy.stats import pspace, Normal
    >>> X = Normal('X', 0, 1)
    >>> pspace(2*X + 1) == X.pspace
    True
    """
    expr = sympify(expr)
    if isinstance(expr, RandomSymbol) and expr.pspace is not None:
        return expr.pspace  # 如果表达式是RandomSymbol类型且有概率空间，则返回其概率空间
    # 检查表达式中是否包含 RandomMatrixSymbol 类型的符号
    if expr.has(RandomMatrixSymbol):
        # 获取表达式中第一个 RandomMatrixSymbol 符号
        rm = list(expr.atoms(RandomMatrixSymbol))[0]
        # 返回该符号的概率空间
        return rm.pspace

    # 获取表达式中所有的随机变量符号
    rvs = random_symbols(expr)
    # 如果没有找到随机变量符号，则抛出异常
    if not rvs:
        raise ValueError("Expression containing Random Variable expected, not %s" % (expr))

    # 如果所有的随机变量符号都属于同一个概率空间，返回这个概率空间
    if all(rv.pspace == rvs[0].pspace for rv in rvs):
        return rvs[0].pspace

    # 导入相关的概率空间类
    from sympy.stats.compound_rv import CompoundPSpace
    from sympy.stats.stochastic_process import StochasticPSpace

    # 遍历所有的随机变量符号，如果其中有符号的概率空间属于 CompoundPSpace 或 StochasticPSpace，则返回该概率空间
    for rv in rvs:
        if isinstance(rv.pspace, (CompoundPSpace, StochasticPSpace)):
            return rv.pspace

    # 如果以上条件都不满足，则创建一个独立乘积空间，包含所有随机变量符号的概率空间
    return IndependentProductPSpace(*[rv.pspace for rv in rvs])
def sumsets(sets):
    """
    Union of sets

    Parameters:
    sets -- list of sets to be unioned

    Returns:
    frozenset -- union of all sets as a frozenset
    """
    return frozenset().union(*sets)


def rs_swap(a, b):
    """
    Build a dictionary to swap RandomSymbols based on their underlying symbol.

    Parameters:
    a -- collection of random variables
    b -- collection of random variables

    Returns:
    dict -- dictionary mapping RVs in 'a' to RVs in 'b'
    """
    d = {}
    for rsa in a:
        # Find matching symbol in 'b' and create mapping
        d[rsa] = [rsb for rsb in b if rsa.symbol == rsb.symbol][0]
    return d


def given(expr, condition=None, **kwargs):
    r""" Conditional Random Expression.

    Explanation
    ===========

    From a random expression and a condition on that expression creates a new
    probability space from the condition and returns the same expression on that
    conditional probability space.

    Examples
    ========

    >>> from sympy.stats import given, density, Die
    >>> X = Die('X', 6)
    >>> Y = given(X, X > 3)
    >>> density(Y).dict
    {4: 1/3, 5: 1/3, 6: 1/3}

    Following convention, if the condition is a random symbol then that symbol
    is considered fixed.

    >>> from sympy.stats import Normal
    >>> from sympy import pprint
    >>> from sympy.abc import z

    >>> X = Normal('X', 0, 1)
    >>> Y = Normal('Y', 0, 1)
    >>> pprint(density(X + Y, Y)(z), use_unicode=False)
                    2
           -(-Y + z)
           -----------
      ___       2
    \/ 2 *e
    ------------------
             ____
         2*\/ pi

    Parameters:
    expr -- random expression to condition upon
    condition -- condition under which to condition 'expr'

    Returns:
    sympy expression -- conditional random expression based on the given condition
    """
    if not is_random(condition) or pspace_independent(expr, condition):
        return expr

    if isinstance(condition, RandomSymbol):
        condition = Eq(condition, condition.symbol)

    condsymbols = random_symbols(condition)
    if (isinstance(condition, Eq) and len(condsymbols) == 1 and
        not isinstance(pspace(expr).domain, ConditionalDomain)):
        rv = tuple(condsymbols)[0]

        results = solveset(condition, rv)
        if isinstance(results, Intersection) and S.Reals in results.args:
            results = list(results.args[1])

        sums = 0
        for res in results:
            temp = expr.subs(rv, res)
            if temp == True:
                return True
            if temp != False:
                # XXX: This seems nonsensical but preserves existing behaviour
                # after the change that Relational is no longer a subclass of
                # Expr. Here expr is sometimes Relational and sometimes Expr
                # but we are trying to add them with +=. This needs to be
                # fixed somehow.
                if sums == 0 and isinstance(expr, Relational):
                    sums = expr.subs(rv, res)
                else:
                    sums += expr.subs(rv, res)
        if sums == 0:
            return False
        return sums

    # Get full probability space of both the expression and the condition
    # 使用 pspace 函数构建完整的空间，该空间由表达式和条件决定
    fullspace = pspace(Tuple(expr, condition))
    
    # 根据条件从完整空间中获取条件化的空间
    space = fullspace.conditional_space(condition, **kwargs)
    
    # 创建一个字典，用于将表达式中的 RandomSymbols 替换为指向新条件空间的新 RandomSymbols
    swapdict = rs_swap(fullspace.values, space.values)
    
    # 在表达式中进行随机变量的替换
    expr = expr.xreplace(swapdict)
    
    # 返回替换后的表达式
    return expr
# 定义一个函数用于计算随机表达式的期望值
def expectation(expr, condition=None, numsamples=None, evaluate=True, **kwargs):
    """
    Returns the expected value of a random expression.

    Parameters
    ==========

    expr : Expr containing RandomSymbols
        The expression of which you want to compute the expectation value
    given : Expr containing RandomSymbols
        A conditional expression. E(X, X>0) is expectation of X given X > 0
    numsamples : int
        Enables sampling and approximates the expectation with this many samples
    evalf : Bool (defaults to True)
        If sampling return a number rather than a complex expression
    evaluate : Bool (defaults to True)
        In case of continuous systems return unevaluated integral

    Examples
    ========

    >>> from sympy.stats import E, Die
    >>> X = Die('X', 6)
    >>> E(X)
    7/2
    >>> E(2*X + 1)
    8

    >>> E(X, X > 3) # Expectation of X given that it is above 3
    5
    """

    # 如果表达式不是随机的，直接返回表达式本身
    if not is_random(expr):  # expr isn't random?
        return expr

    # 将 numsamples 参数传递给 kwargs 字典
    kwargs['numsamples'] = numsamples

    # 导入 Expectation 类
    from sympy.stats.symbolic_probability import Expectation

    # 根据 evaluate 参数选择是否执行期望值计算并返回结果
    if evaluate:
        return Expectation(expr, condition).doit(**kwargs)
    else:
        return Expectation(expr, condition)


# 定义一个函数用于计算条件概率
def probability(condition, given_condition=None, numsamples=None,
                evaluate=True, **kwargs):
    """
    Probability that a condition is true, optionally given a second condition.

    Parameters
    ==========

    condition : Combination of Relationals containing RandomSymbols
        The condition of which you want to compute the probability
    given_condition : Combination of Relationals containing RandomSymbols
        A conditional expression. P(X > 1, X > 0) is expectation of X > 1
        given X > 0
    numsamples : int
        Enables sampling and approximates the probability with this many samples
    evaluate : Bool (defaults to True)
        In case of continuous systems return unevaluated integral

    Examples
    ========

    >>> from sympy.stats import P, Die
    >>> from sympy import Eq
    >>> X, Y = Die('X', 6), Die('Y', 6)
    >>> P(X > 3)
    1/2
    >>> P(Eq(X, 5), X > 2) # Probability that X == 5 given that X > 2
    1/4
    >>> P(X > Y)
    5/12
    """

    # 将 numsamples 参数传递给 kwargs 字典
    kwargs['numsamples'] = numsamples

    # 导入 Probability 类
    from sympy.stats.symbolic_probability import Probability

    # 根据 evaluate 参数选择是否执行概率计算并返回结果
    if evaluate:
        return Probability(condition, given_condition).doit(**kwargs)
    else:
        return Probability(condition, given_condition)


# 定义一个表示密度函数的类
class Density(Basic):
    expr = property(lambda self: self.args[0])

    def __new__(cls, expr, condition=None):
        expr = _sympify(expr)
        if condition is None:
            obj = Basic.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Basic.__new__(cls, expr, condition)
        return obj

    @property
    def condition(self):
        if len(self.args) > 1:
            return self.args[1]
        else:
            return None
    def doit(self, evaluate=True, **kwargs):
        # 导入所需的统计模块
        from sympy.stats.random_matrix import RandomMatrixPSpace
        from sympy.stats.joint_rv import JointPSpace
        from sympy.stats.matrix_distributions import MatrixPSpace
        from sympy.stats.compound_rv import CompoundPSpace
        from sympy.stats.frv import SingleFiniteDistribution
        # 获取表达式和条件
        expr, condition = self.expr, self.condition

        # 如果表达式是 SingleFiniteDistribution 类型，则直接返回其字典表示
        if isinstance(expr, SingleFiniteDistribution):
            return expr.dict
        # 如果有条件，则根据新条件重新计算表达式
        if condition is not None:
            expr = given(expr, condition, **kwargs)
        # 如果表达式中没有随机变量符号，则返回以 expr 为参数的 DiracDelta 函数
        if not random_symbols(expr):
            return Lambda(x, DiracDelta(x - expr))
        # 如果表达式是 RandomSymbol 类型
        if isinstance(expr, RandomSymbol):
            # 如果 expr 的概率空间是 SinglePSpace、JointPSpace、MatrixPSpace，并且具有 distribution 属性，则返回其分布
            if isinstance(expr.pspace, (SinglePSpace, JointPSpace, MatrixPSpace)) and \
                hasattr(expr.pspace, 'distribution'):
                return expr.pspace.distribution
            # 如果 expr 的概率空间是 RandomMatrixPSpace，则返回其模型
            elif isinstance(expr.pspace, RandomMatrixPSpace):
                return expr.pspace.model
        # 如果表达式的概率空间是 CompoundPSpace，则设置关键字参数 'compound_evaluate' 为 evaluate 的值
        if isinstance(pspace(expr), CompoundPSpace):
            kwargs['compound_evaluate'] = evaluate
        # 计算表达式的概率空间的密度函数，使用给定的参数 kwargs
        result = pspace(expr).compute_density(expr, **kwargs)

        # 如果 evaluate 为 True，并且结果具有 doit 方法，则对结果执行 doit 方法
        if evaluate and hasattr(result, 'doit'):
            return result.doit()
        else:
            return result
# 计算随机表达式的概率密度函数，可选地给定第二个条件。
def density(expr, condition=None, evaluate=True, numsamples=None, **kwargs):
    """
    Probability density of a random expression, optionally given a second
    condition.

    Explanation
    ===========

    This density will take on different forms for different types of
    probability spaces. Discrete variables produce Dicts. Continuous
    variables produce Lambdas.

    Parameters
    ==========

    expr : Expr containing RandomSymbols
        The expression of which you want to compute the density value
    condition : Relational containing RandomSymbols
        A conditional expression. density(X > 1, X > 0) is density of X > 1
        given X > 0
    numsamples : int
        Enables sampling and approximates the density with this many samples

    Examples
    ========

    >>> from sympy.stats import density, Die, Normal
    >>> from sympy import Symbol

    >>> x = Symbol('x')
    >>> D = Die('D', 6)
    >>> X = Normal(x, 0, 1)

    >>> density(D).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
    >>> density(2*D).dict
    {2: 1/6, 4: 1/6, 6: 1/6, 8: 1/6, 10: 1/6, 12: 1/6}
    >>> density(X)(x)
    sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))
    """

    # 如果指定了 numsamples 参数，则使用采样方法来近似计算概率密度函数
    if numsamples:
        return sampling_density(expr, condition, numsamples=numsamples,
                **kwargs)

    # 否则，调用 Density 类的 doit 方法计算概率密度函数
    return Density(expr, condition).doit(evaluate=evaluate, **kwargs)


# 计算随机表达式的累积分布函数，可选地给定第二个条件。
def cdf(expr, condition=None, evaluate=True, **kwargs):
    """
    Cumulative Distribution Function of a random expression.

    optionally given a second condition.

    Explanation
    ===========

    This density will take on different forms for different types of
    probability spaces.
    Discrete variables produce Dicts.
    Continuous variables produce Lambdas.

    Examples
    ========

    >>> from sympy.stats import density, Die, Normal, cdf

    >>> D = Die('D', 6)
    >>> X = Normal('X', 0, 1)

    >>> density(D).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
    >>> cdf(D)
    {1: 1/6, 2: 1/3, 3: 1/2, 4: 2/3, 5: 5/6, 6: 1}
    >>> cdf(3*D, D > 2)
    {9: 1/4, 12: 1/2, 15: 3/4, 18: 1}

    >>> cdf(X)
    Lambda(_z, erf(sqrt(2)*_z/2)/2 + 1/2)
    """
    if condition is not None:  # 如果存在条件
        # 递归调用，重新计算新的条件表达式的累积分布函数
        return cdf(given(expr, condition, **kwargs), **kwargs)

    # 否则，将工作传递给 ProbabilitySpace 的 compute_cdf 方法
    result = pspace(expr).compute_cdf(expr, **kwargs)

    # 如果 evaluate 参数为 True 并且结果具有 doit 方法，则调用 doit 方法
    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result


# 计算随机表达式的特征函数，可选地给定第二个条件。
def characteristic_function(expr, condition=None, evaluate=True, **kwargs):
    """
    Characteristic function of a random expression, optionally given a second condition.

    Returns a Lambda.

    Examples
    ========

    >>> from sympy.stats import Normal, DiscreteUniform, Poisson, characteristic_function

    >>> X = Normal('X', 0, 1)
    >>> characteristic_function(X)
    Lambda(_t, exp(-_t**2/2))

    >>> Y = DiscreteUniform('Y', [1, 2, 7])
    """
    # 如果给定条件不为 None，则递归调用 characteristic_function 函数，
    # 并传递给定的表达式和条件参数，忽略其它 kwargs
    if condition is not None:
        return characteristic_function(given(expr, condition, **kwargs), **kwargs)

    # 否则，计算表达式的概率空间，并计算其特征函数
    result = pspace(expr).compute_characteristic_function(expr, **kwargs)

    # 如果 evaluate 为 True 并且 result 对象具有 doit 方法，
    # 则调用 doit 方法，返回计算结果
    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    # 否则，直接返回计算的结果对象
    else:
        return result
# 计算随机变量的矩生成函数
def moment_generating_function(expr, condition=None, evaluate=True, **kwargs):
    # 如果有条件，递归调用 moment_generating_function，将条件应用到表达式上
    if condition is not None:
        return moment_generating_function(given(expr, condition, **kwargs), **kwargs)

    # 使用 ProbabilitySpace 计算表达式的矩生成函数
    result = pspace(expr).compute_moment_generating_function(expr, **kwargs)

    # 如果 evaluate 为 True，并且 result 对象有 doit 方法，则调用 doit 方法进行求值
    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result


# 返回满足条件的定义域
def where(condition, given_condition=None, **kwargs):
    """
    Returns the domain where a condition is True.

    Examples
    ========

    >>> from sympy.stats import where, Die, Normal
    >>> from sympy import And

    >>> D1, D2 = Die('a', 6), Die('b', 6)
    >>> a, b = D1.symbol, D2.symbol
    >>> X = Normal('x', 0, 1)

    >>> where(X**2<1)
    Domain: (-1 < x) & (x < 1)

    >>> where(X**2<1).set
    Interval.open(-1, 1)

    >>> where(And(D1<=D2, D2<3))
    Domain: (Eq(a, 1) & Eq(b, 1)) | (Eq(a, 1) & Eq(b, 2)) | (Eq(a, 2) & Eq(b, 2))
    """
    if given_condition is not None:  # 如果存在给定条件，则将给定条件应用到主条件上，并递归调用 where 函数
        return where(given(condition, given_condition, **kwargs), **kwargs)

    # 否则，将条件传递给 ProbabilitySpace 的 where 方法进行处理
    return pspace(condition).where(condition, **kwargs)


@doctest_depends_on(modules=('scipy',))
def sample(expr, condition=None, size=(), library='scipy',
           numsamples=1, seed=None, **kwargs):
    """
    A realization of the random expression.

    Parameters
    ==========

    expr : Expression of random variables
        Expression from which sample is extracted
    condition : Expr containing RandomSymbols
        A conditional expression
    size : int, tuple
        Represents size of each sample in numsamples
    library : str
        - 'scipy' : Sample using scipy
        - 'numpy' : Sample using numpy
        - 'pymc'  : Sample using PyMC

        Choose any of the available options to sample from as string,
        by default is 'scipy'
    numsamples : int
        Number of samples, each with size as ``size``.

        .. deprecated:: 1.9

        The ``numsamples`` parameter is deprecated and is only provided for
        compatibility with v1.8. Use a list comprehension or an additional
        dimension in ``size`` instead. See
        :ref:`deprecated-sympy-stats-numsamples` for details.

    seed :
        An object to be used as seed by the given external library for sampling `expr`.
        Following is the list of possible types of object for the supported libraries,

        - 'scipy': int, numpy.random.RandomState, numpy.random.Generator
        - 'numpy': int, numpy.random.RandomState, numpy.random.Generator
        - 'pymc': int

        Optional, by default None, in which case seed settings
        related to the given library will be used.
        No modifications to environment's global seed settings
        are done by this argument.

    Returns
    =======

    """
    # sample: float/list/numpy.ndarray
    #     one sample or a collection of samples of the random expression.
    #
    #     - sample(X) returns float/numpy.float64/numpy.int64 object.
    #     - sample(X, size=int/tuple) returns numpy.ndarray object.
    #
    # Examples
    # ========
    #
    # >>> from sympy.stats import Die, sample, Normal, Geometric
    # >>> X, Y, Z = Die('X', 6), Die('Y', 6), Die('Z', 6) # Finite Random Variable
    # >>> die_roll = sample(X + Y + Z)
    # >>> die_roll # doctest: +SKIP
    # 3
    # >>> N = Normal('N', 3, 4) # Continuous Random Variable
    # >>> samp = sample(N)
    # >>> samp in N.pspace.domain.set
    # True
    # >>> samp = sample(N, N>0)
    # >>> samp > 0
    # True
    # >>> samp_list = sample(N, size=4)
    # >>> [sam in N.pspace.domain.set for sam in samp_list]
    # [True, True, True, True]
    # >>> sample(N, size = (2,3)) # doctest: +SKIP
    # array([[5.42519758, 6.40207856, 4.94991743],
    #        [1.85819627, 6.83403519, 1.9412172 ]])
    # >>> G = Geometric('G', 0.5) # Discrete Random Variable
    # >>> samp_list = sample(G, size=3)
    # >>> samp_list # doctest: +SKIP
    # [1, 3, 2]
    # >>> [sam in G.pspace.domain.set for sam in samp_list]
    # [True, True, True]
    # >>> MN = Normal("MN", [3, 4], [[2, 1], [1, 2]]) # Joint Random Variable
    # >>> samp_list = sample(MN, size=4)
    # >>> samp_list # doctest: +SKIP
    # [array([2.85768055, 3.38954165]),
    #  array([4.11163337, 4.3176591 ]),
    #  array([0.79115232, 1.63232916]),
    #  array([4.01747268, 3.96716083])]
    # >>> [tuple(sam) in MN.pspace.domain.set for sam in samp_list]
    # [True, True, True, True]
    #
    # .. versionchanged:: 1.7.0
    #     sample used to return an iterator containing the samples instead of value.
    #
    # .. versionchanged:: 1.9.0
    #     sample returns values or array of values instead of an iterator and numsamples is deprecated.

    # 使用 sample_iter 函数获取迭代器，生成随机样本
    iterator = sample_iter(expr, condition, size=size, library=library,
                                                        numsamples=numsamples, seed=seed)

    if numsamples != 1:
        # 发出警告，说明 numsamples 参数已被弃用
        sympy_deprecation_warning(
            f"""
            The numsamples parameter to sympy.stats.sample() is deprecated.
            Either use a list comprehension, like

            [sample(...) for i in range({numsamples})]

            or add a dimension to size, like

            sample(..., size={(numsamples,) + size})
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-sympy-stats-numsamples",
        )
        # 返回包含 numsamples 个样本的列表
        return [next(iterator) for i in range(numsamples)]

    # 返回迭代器中的下一个样本
    return next(iterator)
# 定义一个函数 quantile，用于计算概率分布的第 p 个顺序分位数
def quantile(expr, evaluate=True, **kwargs):
    """
    Return the :math:`p^{th}` order quantile of a probability distribution.

    Explanation
    ===========

    Quantile is defined as the value at which the probability of the random
    variable is less than or equal to the given probability.

    .. math::
        Q(p) = \inf\{x \in (-\infty, \infty) : p \le F(x)\}

    Examples
    ========

    >>> from sympy.stats import quantile, Die, Exponential
    >>> from sympy import Symbol, pprint
    >>> p = Symbol("p")

    >>> l = Symbol("lambda", positive=True)
    >>> X = Exponential("x", l)
    >>> quantile(X)(p)
    -log(1 - p)/lambda

    >>> D = Die("d", 6)
    >>> pprint(quantile(D)(p), use_unicode=False)
    /nan  for Or(p > 1, p < 0)
    |
    | 1       for p <= 1/6
    |
    | 2       for p <= 1/3
    |
    < 3       for p <= 1/2
    |
    | 4       for p <= 2/3
    |
    | 5       for p <= 5/6
    |
    \ 6        for p <= 1

    """
    # 使用 pspace 函数计算表达式的分位数
    result = pspace(expr).compute_quantile(expr, **kwargs)

    # 如果 evaluate 为 True，并且 result 有 doit 方法，则调用 doit 方法进行求值
    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result

# 定义一个函数 sample_iter，返回给定表达式的实现的迭代器
def sample_iter(expr, condition=None, size=(), library='scipy',
                    numsamples=S.Infinity, seed=None, **kwargs):

    """
    Returns an iterator of realizations from the expression given a condition.

    Parameters
    ==========

    expr: Expr
        Random expression to be realized
    condition: Expr, optional
        A conditional expression
    size : int, tuple
        Represents size of each sample in numsamples
    numsamples: integer, optional
        Length of the iterator (defaults to infinity)
    seed :
        An object to be used as seed by the given external library for sampling `expr`.
        Following is the list of possible types of object for the supported libraries,

        - 'scipy': int, numpy.random.RandomState, numpy.random.Generator
        - 'numpy': int, numpy.random.RandomState, numpy.random.Generator
        - 'pymc': int

        Optional, by default None, in which case seed settings
        related to the given library will be used.
        No modifications to environment's global seed settings
        are done by this argument.

    Examples
    ========

    >>> from sympy.stats import Normal, sample_iter
    >>> X = Normal('X', 0, 1)
    >>> expr = X*X + 3
    >>> iterator = sample_iter(expr, numsamples=3) # doctest: +SKIP
    >>> list(iterator) # doctest: +SKIP
    [12, 4, 7]

    Returns
    =======

    sample_iter: iterator object
        iterator object containing the sample/samples of given expr

    See Also
    ========

    sample
    sampling_P
    sampling_E

    """
    # 导入 JointRandomSymbol 类型以处理联合随机变量
    from sympy.stats.joint_rv import JointRandomSymbol
    # 如果导入指定的库失败，则抛出 ValueError 异常
    if not import_module(library):
        raise ValueError("Failed to import %s" % library)

    # 根据是否有条件表达式选择对应的 pspace
    if condition is not None:
        ps = pspace(Tuple(expr, condition))
    else:
        ps = pspace(expr)

    # 获取概率空间的所有随机变量列表
    rvs = list(ps.values)
    # 如果表达式是 JointRandomSymbol 类型，则替换为对应的 RandomSymbol 对象
    if isinstance(expr, JointRandomSymbol):
        expr = expr.subs({expr: RandomSymbol(expr.symbol, expr.pspace)})
    else:
        # 构建用于替换的字典 sub
        sub = {}
        # 遍历表达式中的每个参数
        for arg in expr.args:
            # 如果参数是 JointRandomSymbol 类型，则替换为对应的 RandomSymbol 对象
            if isinstance(arg, JointRandomSymbol):
                sub[arg] = RandomSymbol(arg.symbol, arg.pspace)
        # 使用 sub 字典对表达式进行替换
        expr = expr.subs(sub)

    # 定义一个函数 fn_subs，用于将参数替换到表达式中并返回结果
    def fn_subs(*args):
        return expr.subs(dict(zip(rvs, args)))

    # 定义一个函数 given_fn_subs，用于将参数替换到条件表达式中并返回结果
    def given_fn_subs(*args):
        if condition is not None:
            return condition.subs(dict(zip(rvs, args)))
        return False

    # 根据指定的库和参数 kwargs，创建一个函数 fn，用于将随机变量参数替换到表达式中并返回结果
    if library in ('pymc', 'pymc3'):
        # 当前无法在 pymc 中使用 lambdify，因此将 TODO 注释
        fn = lambdify(rvs, expr, **kwargs)
    else:
        # 使用指定的库和参数 kwargs 创建函数 fn，用于将随机变量参数替换到表达式中并返回结果
        fn = lambdify(rvs, expr, modules=library, **kwargs)

    # 如果存在条件表达式，则创建一个函数 given_fn，用于将随机变量参数替换到条件表达式中并返回结果
    if condition is not None:
        given_fn = lambdify(rvs, condition, **kwargs)

    # 定义一个无限生成器函数 return_generator_infinite
    def return_generator_infinite():
        count = 0
        # 将 size 转换为元组 _size，确保第一个维度是 1
        _size = (1,) + ((size,) if isinstance(size, int) else size)
        # 当生成的样本数量小于 numsamples 时循环
        while count < numsamples:
            # 从概率空间 ps 中采样得到一个字典 d，将随机变量映射到具体的值
            d = ps.sample(size=_size, library=library, seed=seed)
            # 根据 rvs 中的随机变量获取对应的值作为参数 args
            args = [d[rv][0] for rv in rvs]

            # 如果存在条件表达式
            if condition is not None:
                # 尝试使用 given_fn(*args) 检查条件
                try:
                    gd = given_fn(*args)
                # 如果 lambdify 无法处理 SymPy 对象的情况，使用 given_fn_subs(*args) 替代
                except (NameError, TypeError):
                    gd = given_fn_subs(*args)
                # 如果条件不是 True 或 False，则抛出 ValueError
                if gd != True and gd != False:
                    raise ValueError(
                        "Conditions must not contain free symbols")
                # 如果条件不满足，则继续生成下一个样本
                if not gd:
                    continue

            # 使用 fn(*args) 计算表达式的结果并生成
            yield fn(*args)
            count += 1
    # 定义一个返回有限数量生成器的函数
    def return_generator_finite():
        # 初始化故障标志为真，表示当前生成器处于故障状态
        faulty = True
        # 循环直到生成器不再故障
        while faulty:
            # 从概率空间中抽取样本，生成一个字典，将随机变量映射到相应的值
            d = ps.sample(size=(numsamples,) + ((size,) if isinstance(size, int) else size),
                          library=library, seed=seed)  # a dictionary that maps RVs to values

            # 将故障标志设为假，假设生成器正常工作
            faulty = False
            # 初始化计数器
            count = 0
            # 在抽样数量未达到设定数量时，并且生成器不故障时循环
            while count < numsamples and not faulty:
                # 提取参数，根据随机变量列表从字典中获取相应的值
                args = [d[rv][count] for rv in rvs]
                # 如果存在条件判断函数，检查这些值是否满足条件
                if condition is not None:
                    # TODO: 当lambdify可以处理未评估的SymPy对象时，用only given_fn(*args)替换try-except块
                    try:
                        # 计算给定条件函数的结果
                        gd = given_fn(*args)
                    except (NameError, TypeError):
                        # 如果出现错误，则使用给定的替代函数计算结果
                        gd = given_fn_subs(*args)
                    # 如果结果不是True或False，则抛出值错误异常
                    if gd != True and gd != False:
                        raise ValueError(
                            "Conditions must not contain free symbols")
                    # 如果结果为False，则将故障标志设为真，表示生成器需要重试
                    if not gd:
                        faulty = True

                count += 1

        # 初始化计数器
        count = 0
        # 在抽样数量未达到设定数量时循环
        while count < numsamples:
            # 提取参数，根据随机变量列表从字典中获取相应的值
            args = [d[rv][count] for rv in rvs]
            # TODO: 当lambdify可以处理未评估的SymPy对象时，用only fn(*args)替换try-except块
            try:
                # 生成器产生函数fn的结果
                yield fn(*args)
            except (NameError, TypeError):
                # 如果出现错误，则使用函数替代版本计算结果
                yield fn_subs(*args)
            count += 1

    # 如果抽样数量为无限，则返回无限生成器的结果
    if numsamples is S.Infinity:
        return return_generator_infinite()

    # 否则返回有限生成器的结果
    return return_generator_finite()
# 定义函数 sample_iter_lambdify，接收表达式、条件、大小、采样数、种子和额外参数
def sample_iter_lambdify(expr, condition=None, size=(),
                         numsamples=S.Infinity, seed=None, **kwargs):
    # 调用 sample_iter 函数进行迭代采样，并返回结果
    return sample_iter(expr, condition=condition, size=size,
                       numsamples=numsamples, seed=seed, **kwargs)

# 定义函数 sample_iter_subs，接收表达式、条件、大小、采样数、种子和额外参数
def sample_iter_subs(expr, condition=None, size=(),
                     numsamples=S.Infinity, seed=None, **kwargs):
    # 调用 sample_iter 函数进行迭代采样，并返回结果
    return sample_iter(expr, condition=condition, size=size,
                       numsamples=numsamples, seed=seed, **kwargs)

# 定义函数 sampling_P，接收条件、给定条件、库、采样数、是否评估、种子和额外参数
def sampling_P(condition, given_condition=None, library='scipy', numsamples=1,
               evalf=True, seed=None, **kwargs):
    """
    Sampling version of P.

    See Also
    ========

    P
    sampling_E
    sampling_density

    """
    # 初始化计数器
    count_true = 0
    count_false = 0
    # 调用 sample_iter 函数进行迭代采样，并遍历结果
    samples = sample_iter(condition, given_condition, library=library,
                          numsamples=numsamples, seed=seed, **kwargs)

    # 统计采样结果中 True 和 False 的数量
    for sample in samples:
        if sample:
            count_true += 1
        else:
            count_false += 1

    # 计算 True 的比例
    result = S(count_true) / numsamples
    # 如果需要进行数值评估，则调用 evalf() 方法
    if evalf:
        return result.evalf()
    else:
        return result

# 定义函数 sampling_E，接收表达式、给定条件、库、采样数、是否评估、种子和额外参数
def sampling_E(expr, given_condition=None, library='scipy', numsamples=1,
               evalf=True, seed=None, **kwargs):
    """
    Sampling version of E.

    See Also
    ========

    P
    sampling_P
    sampling_density
    """
    # 调用 sample_iter 函数进行迭代采样，并将结果转换为列表
    samples = list(sample_iter(expr, given_condition, library=library,
                          numsamples=numsamples, seed=seed, **kwargs))
    # 计算样本的总和并除以采样数得到期望值的估计
    result = Add(*samples) / numsamples

    # 如果需要进行数值评估，则调用 evalf() 方法
    if evalf:
        return result.evalf()
    else:
        return result

# 定义函数 sampling_density，接收表达式、给定条件、库、采样数、种子和额外参数
def sampling_density(expr, given_condition=None, library='scipy',
                    numsamples=1, seed=None, **kwargs):
    """
    Sampling version of density.

    See Also
    ========
    density
    sampling_P
    sampling_E
    """
    # 初始化结果字典
    results = {}
    # 调用 sample_iter 函数进行迭代采样，并遍历结果
    for result in sample_iter(expr, given_condition, library=library,
                              numsamples=numsamples, seed=seed, **kwargs):
        # 统计每个采样结果出现的次数
        results[result] = results.get(result, 0) + 1

    # 返回结果字典
    return results

# 定义函数 dependent，接收两个随机表达式 a 和 b
def dependent(a, b):
    """
    Dependence of two random expressions.

    Two expressions are independent if knowledge of one does not change
    computations on the other.

    Examples
    ========

    >>> from sympy.stats import Normal, dependent, given
    >>> from sympy import Tuple, Eq

    >>> X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    >>> dependent(X, Y)
    False
    >>> dependent(2*X + Y, -Y)
    True
    >>> X, Y = given(Tuple(X, Y), Eq(X + Y, 3))
    >>> dependent(X, Y)
    True

    See Also
    ========

    independent
    """
    # 如果 a 和 b 是独立的，则返回 False
    if pspace_independent(a, b):
        return False

    # 创建实数符号 z
    z = Symbol('z', real=True)
    # 如果密度函数在给定一个表达式条件时不变，则表达式 a 和 b 是依赖的
    return (density(a, Eq(b, z)) != density(a) or
            density(b, Eq(a, z)) != density(b))
    """
    Independence of two random expressions.

    Two expressions are independent if knowledge of one does not change
    computations on the other.

    Examples
    ========

    >>> from sympy.stats import Normal, independent, given
    >>> from sympy import Tuple, Eq

    >>> X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    >>> independent(X, Y)
    True
    >>> independent(2*X + Y, -Y)
    False
    >>> X, Y = given(Tuple(X, Y), Eq(X + Y, 3))
    >>> independent(X, Y)
    False

    See Also
    ========

    dependent
    """

    # 返回两个随机表达式是否独立的布尔值，通过检查它们是否依赖于彼此来判断
    return not dependent(a, b)
# 测试函数，用于检测给定的两个随机变量 a 和 b 是否独立，通过检查它们的概率空间是否具有重叠的符号来实现。
# 这是独立性的一个充分条件，但不是必要条件，并且意图是在内部使用。

def pspace_independent(a, b):
    # 获取随机变量 b 的概率空间中的符号集合
    a_symbols = set(pspace(b).symbols)
    # 获取随机变量 a 的概率空间中的符号集合
    b_symbols = set(pspace(a).symbols)

    # 如果随机选择的符号集合交集不为空，则说明随机变量之间存在相关性，返回 False
    if len(set(random_symbols(a)).intersection(random_symbols(b))) != 0:
        return False

    # 如果 a 和 b 的符号集合的交集为空，则说明它们是独立的，返回 True
    if len(a_symbols.intersection(b_symbols)) == 0:
        return True
    # 如果无法确定独立性，则返回 None
    return None


# 给定一个随机表达式，用其符号替换所有随机变量。
# 如果提供了 symbols 关键字参数，则仅将替换限制为列出的符号。
def rv_subs(expr, symbols=None):
    if symbols is None:
        symbols = random_symbols(expr)
    # 如果 symbols 为空列表，则直接返回原始表达式
    if not symbols:
        return expr
    # 创建一个字典，用于将随机变量替换为它们的符号
    swapdict = {rv: rv.symbol for rv in symbols}
    # 执行符号替换并返回结果
    return expr.subs(swapdict)


class NamedArgsMixin:
    _argnames: tuple[str, ...] = ()

    # 重写 __getattr__ 方法，以便通过属性名称获取参数值
    def __getattr__(self, attr):
        try:
            # 根据属性名称在 self.args 中获取相应的参数值
            return self.args[self._argnames.index(attr)]
        except ValueError:
            # 如果属性名称不在 _argnames 中，则抛出 AttributeError
            raise AttributeError("'%s' object has no attribute '%s'" % (
                type(self).__name__, attr))


class Distribution(Basic):
    # Distribution 类的定义从这里开始，后续内容未提供，需继续补充
    def sample(self, size=(), library='scipy', seed=None):
        """ A random realization from the distribution """

        # 根据传入的库名动态导入对应的模块
        module = import_module(library)
        # 如果库名在{'scipy', 'numpy', 'pymc3', 'pymc'}中且未成功导入模块，则抛出异常
        if library in {'scipy', 'numpy', 'pymc3', 'pymc'} and module is None:
            raise ValueError("Failed to import %s" % library)

        # 如果使用的是'scipy'库
        if library == 'scipy':
            # scipy 可以处理自定义分布，不需要使用 map 函数，但在需要的地方仍然使用了 map

            # TODO: 如果需要，在 drv.py 和 frv.py 中也执行类似的操作。
            # TODO: 如果还有更多分布，可以在这里添加。
            # 参考下面链接，查看以“A common parametrization...”开头的部分
            # 如果一切正常，将删除所有这些注释。

            # 导入 scipy 下的采样函数
            from sympy.stats.sampling.sample_scipy import do_sample_scipy
            import numpy
            # 根据种子值设置随机数生成器的状态
            if seed is None or isinstance(seed, int):
                rand_state = numpy.random.default_rng(seed=seed)
            else:
                rand_state = seed
            # 调用 scipy 下的采样函数，生成样本
            samps = do_sample_scipy(self, size, rand_state)

        # 如果使用的是'numpy'库
        elif library == 'numpy':
            # 导入 numpy 下的采样函数
            from sympy.stats.sampling.sample_numpy import do_sample_numpy
            import numpy
            # 根据种子值设置随机数生成器的状态
            if seed is None or isinstance(seed, int):
                rand_state = numpy.random.default_rng(seed=seed)
            else:
                rand_state = seed
            # 将空元组转换为 None，以便传递给采样函数
            _size = None if size == () else size
            # 调用 numpy 下的采样函数，生成样本
            samps = do_sample_numpy(self, _size, rand_state)

        # 如果使用的是'pymc'或'pymc3'库
        elif library in ('pymc', 'pymc3'):
            # 导入 pymc 或 pymc3 下的采样函数
            from sympy.stats.sampling.sample_pymc import do_sample_pymc
            import logging
            # 设置 pymc 的日志级别为 ERROR
            logging.getLogger("pymc").setLevel(logging.ERROR)
            try:
                import pymc
            except ImportError:
                import pymc3 as pymc

            # 使用 pymc 或 pymc3 的上下文管理器创建模型
            with pymc.Model():
                # 调用 pymc 或 pymc3 下的采样函数，生成样本
                if do_sample_pymc(self):
                    samps = pymc.sample(draws=prod(size), chains=1, compute_convergence_checks=False,
                            progressbar=False, random_seed=seed, return_inferencedata=False)[:]['X']
                    samps = samps.reshape(size)
                else:
                    samps = None

        # 如果使用的是其他未支持的库，则抛出未实现异常
        else:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                      % str(library))

        # 如果生成的样本不为空，则返回样本
        if samps is not None:
            return samps
        # 否则抛出未实现异常，指出当前库的采样功能未实现
        raise NotImplementedError(
            "Sampling for %s is not currently implemented from %s"
            % (self, library))
# 检查条件是否为真，若条件为假则抛出 ValueError 异常并带有指定消息；若条件为真或全部条件为真则返回 True，否则返回 False。
def _value_check(condition, message):
    """
    Raise a ValueError with message if condition is False, else
    return True if all conditions were True, else False.

    Examples
    ========

    >>> from sympy.stats.rv import _value_check
    >>> from sympy.abc import a, b, c
    >>> from sympy import And, Dummy

    >>> _value_check(2 < 3, '')
    True

    Here, the condition is not False, but it does not evaluate to True
    so False is returned (but no error is raised). So checking if the
    return value is True or False will tell you if all conditions were
    evaluated.

    >>> _value_check(a < b, '')
    False

    In this case the condition is False so an error is raised:

    >>> r = Dummy(real=True)
    >>> _value_check(r < r - 1, 'condition is not true')
    Traceback (most recent call last):
    ...
    ValueError: condition is not true

    If no condition of many conditions must be False, they can be
    checked by passing them as an iterable:

    >>> _value_check((a < 0, b < 0, c < 0), '')
    False

    The iterable can be a generator, too:

    >>> _value_check((i < 0 for i in (a, b, c)), '')
    False

    The following are equivalent to the above but do not pass
    an iterable:

    >>> all(_value_check(i < 0, '') for i in (a, b, c))
    False
    >>> _value_check(And(a < 0, b < 0, c < 0), '')
    False
    """
    # 如果条件不是可迭代的，则将其转换为列表
    if not iterable(condition):
        condition = [condition]
    # 使用 fuzzy_and 函数计算条件列表中所有条件的模糊与（fuzzy and）
    truth = fuzzy_and(condition)
    # 如果结果为 False，则抛出 ValueError 异常
    if truth == False:
        raise ValueError(message)
    # 返回条件是否全为 True 的布尔值
    return truth == True

# 将参数转换为符号（Symbol）类型，如果参数已经是字符串（str），则进行转换；否则不进行任何操作。
def _symbol_converter(sym):
    """
    Casts the parameter to Symbol if it is 'str'
    otherwise no operation is performed on it.

    Parameters
    ==========

    sym
        The parameter to be converted.

    Returns
    =======

    Symbol
        the parameter converted to Symbol.

    Raises
    ======

    TypeError
        If the parameter is not an instance of both str and
        Symbol.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.stats.rv import _symbol_converter
    >>> s = _symbol_converter('s')
    >>> isinstance(s, Symbol)
    True
    >>> _symbol_converter(1)
    Traceback (most recent call last):
    ...
    TypeError: 1 is neither a Symbol nor a string
    >>> r = Symbol('r')
    >>> isinstance(r, Symbol)
    True
    """
    # 如果参数是字符串，则将其转换为符号（Symbol）类型
    if isinstance(sym, str):
        sym = Symbol(sym)
    # 如果参数不是符号（Symbol）类型，则抛出 TypeError 异常
    if not isinstance(sym, Symbol):
        raise TypeError("%s is neither a Symbol nor a string"%(sym))
    # 返回转换后的符号（Symbol）对象
    return sym

# 从随机过程（StochasticProcess）中抽样的函数。
def sample_stochastic_process(process):
    """
    This function is used to sample from stochastic process.

    Parameters
    ==========

    process: StochasticProcess
        Process used to extract the samples. It must be an instance of
        StochasticProcess

    Examples
    ========

    >>> from sympy.stats import sample_stochastic_process, DiscreteMarkovChain
    >>> from sympy import Matrix
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    """
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    # 创建一个名为Y的离散马尔可夫链对象，状态空间为[0, 1, 2]，转移矩阵为T

    >>> next(sample_stochastic_process(Y)) in Y.state_space
    # 从马尔可夫链Y的样本中获取下一个状态，并检查其是否在Y的状态空间中
    True

    >>> next(sample_stochastic_process(Y))  # doctest: +SKIP
    # 从马尔可夫链Y的样本中获取下一个状态，但跳过这个测试，不执行

    >>> next(sample_stochastic_process(Y)) # doctest: +SKIP
    # 从马尔可夫链Y的样本中获取下一个状态，但跳过这个测试，不执行

    Returns
    =======

    sample: iterator object
        # 返回一个迭代器对象，包含给定过程的样本

    """
    from sympy.stats.stochastic_process_types import StochasticProcess
    # 从sympy.stats.stochastic_process_types模块导入StochasticProcess类

    if not isinstance(process, StochasticProcess):
        # 如果process不是StochasticProcess的实例，则抛出值错误异常
        raise ValueError("Process must be an instance of Stochastic Process")

    return process.sample()
    # 返回process对象的样本
```