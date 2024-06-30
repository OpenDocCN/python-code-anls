# `D:\src\scipysrc\sympy\sympy\stats\joint_rv.py`

```
"""
Joint Random Variables Module

See Also
========
sympy.stats.rv
sympy.stats.frv
sympy.stats.crv
sympy.stats.drv
"""
# 从 math 模块中导入 prod 函数
from math import prod

# 导入 SymPy 相关模块
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.sets.sets import ProductSet
from sympy.tensor.indexed import Indexed
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation
from sympy.core.containers import Tuple
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices import ImmutableMatrix, matrix2numpy, list2numpy
from sympy.stats.crv import SingleContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import (ProductPSpace, NamedArgsMixin, Distribution,
                            ProductDomain, RandomSymbol, random_symbols,
                            SingleDomain, _symbol_converter)
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent
from sympy.external import import_module

# __all__ = ['marginal_distribution']

# 定义 JointPSpace 类，继承自 ProductPSpace 类
class JointPSpace(ProductPSpace):
    """
    Represents a joint probability space. Represented using symbols for
    each component and a distribution.
    """
    
    # 构造函数，接受符号和分布作为参数
    def __new__(cls, sym, dist):
        # 如果分布是 SingleContinuousDistribution 类型，则返回 SingleContinuousPSpace 实例
        if isinstance(dist, SingleContinuousDistribution):
            return SingleContinuousPSpace(sym, dist)
        # 如果分布是 SingleDiscreteDistribution 类型，则返回 SingleDiscretePSpace 实例
        if isinstance(dist, SingleDiscreteDistribution):
            return SingleDiscretePSpace(sym, dist)
        # 调用 _symbol_converter 函数转换符号
        sym = _symbol_converter(sym)
        # 使用 Basic 类的 __new__ 方法创建新的 JointPSpace 实例
        return Basic.__new__(cls, sym, dist)

    # 返回当前空间的集合属性
    @property
    def set(self):
        return self.domain.set

    # 返回当前空间的符号属性
    @property
    def symbol(self):
        return self.args[0]

    # 返回当前空间的分布属性
    @property
    def distribution(self):
        return self.args[1]

    # 返回当前空间的值属性，使用 JointRandomSymbol 类
    @property
    def value(self):
        return JointRandomSymbol(self.symbol, self)

    # 返回当前空间的组件数量属性
    @property
    def component_count(self):
        # 获取分布的集合属性
        _set = self.distribution.set
        # 如果集合是 ProductSet 类型，则返回其参数个数
        if isinstance(_set, ProductSet):
            return S(len(_set.args))
        # 如果集合是 Product 类型，则返回其限制参数的最后一个值
        elif isinstance(_set, Product):
            return _set.limits[0][-1]
        # 否则返回 1
        return S.One

    # 返回当前空间的概率密度函数属性
    @property
    def pdf(self):
        # 生成当前空间组件符号的列表
        sym = [Indexed(self.symbol, i) for i in range(self.component_count)]
        # 调用分布的函数，计算概率密度函数
        return self.distribution(*sym)

    # 返回当前空间的域属性
    @property
    def domain(self):
        # 获取分布中的随机符号列表
        rvs = random_symbols(self.distribution)
        # 如果列表为空，返回 SingleDomain 实例
        if not rvs:
            return SingleDomain(self.symbol, self.distribution.set)
        # 否则返回 ProductDomain 实例，传入每个随机变量的域
        return ProductDomain(*[rv.pspace.domain for rv in rvs])

    # 返回特定组件索引的域属性
    def component_domain(self, index):
        return self.set.args[index]
    def marginal_distribution(self, *indices):
        # 获取成分数量
        count = self.component_count
        # 如果成分包含符号变量，抛出值错误异常
        if count.atoms(Symbol):
            raise ValueError("Marginal distributions cannot be computed "
                             "for symbolic dimensions. It is a work under progress.")
        # 创建原始索引列表
        orig = [Indexed(self.symbol, i) for i in range(count)]
        # 创建所有符号列表，用于替换字典
        all_syms = [Symbol(str(i)) for i in orig]
        # 创建替换字典，将所有符号与原始索引对应起来
        replace_dict = dict(zip(all_syms, orig))
        # 创建符号元组，用于限制条件
        sym = tuple(Symbol(str(Indexed(self.symbol, i))) for i in indices)
        # 初始化限制条件列表
        limits = [[i,] for i in all_syms if i not in sym]
        # 索引计数器
        index = 0
        # 遍历所有成分
        for i in range(count):
            # 如果当前成分不在索引中
            if i not in indices:
                # 添加分布设置参数到限制条件
                limits[index].append(self.distribution.set.args[i])
                # 转换为元组
                limits[index] = tuple(limits[index])
                # 更新索引
                index += 1
        # 如果分布是连续的
        if self.distribution.is_Continuous:
            # 创建 lambda 函数，用于连续积分
            f = Lambda(sym, integrate(self.distribution(*all_syms), *limits))
        # 如果分布是离散的
        elif self.distribution.is_Discrete:
            # 创建 lambda 函数，用于离散求和
            f = Lambda(sym, summation(self.distribution(*all_syms), *limits))
        # 返回函数，并替换符号变量
        return f.xreplace(replace_dict)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        # 获取所有符号元组
        syms = tuple(self.value[i] for i in range(self.component_count))
        # 如果随机变量为空，则使用所有符号
        rvs = rvs or syms
        # 如果没有任何符号在随机变量中，则返回表达式
        if not any(i in rvs for i in syms):
            return expr
        # 表达式乘以概率密度函数
        expr = expr*self.pdf
        # 替换表达式中的随机变量
        for rv in rvs:
            if isinstance(rv, Indexed):
                expr = expr.xreplace({rv: Indexed(str(rv.base), rv.args[1])})
            elif isinstance(rv, RandomSymbol):
                expr = expr.xreplace({rv: rv.symbol})
        # 如果表达式中包含随机符号，则抛出未实现错误
        if self.value in random_symbols(expr):
            raise NotImplementedError(filldedent('''
            Expectations of expression with unindexed joint random symbols
            cannot be calculated yet.'''))
        # 创建积分限制条件元组
        limits = tuple((Indexed(str(rv.base),rv.args[1]),
            self.distribution.set.args[rv.args[1]]) for rv in syms)
        # 返回积分表达式
        return Integral(expr, *limits)

    def where(self, condition):
        # 抛出未实现错误
        raise NotImplementedError()

    def compute_density(self, expr):
        # 抛出未实现错误
        raise NotImplementedError()

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        # 返回抽样结果字典，映射随机符号到实现值
        return {RandomSymbol(self.symbol, self): self.distribution.sample(size,
                    library=library, seed=seed)}

    def probability(self, condition):
        # 抛出未实现错误
        raise NotImplementedError()
class SampleJointScipy:
    """Returns the sample from scipy of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        # 调用内部方法 _sample_scipy 来获取 scipy 分布的样本数据
        return cls._sample_scipy(dist, size, seed)

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

        import numpy
        # 根据种子值生成随机状态对象
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        from scipy import stats as scipy_stats
        # 定义 SciPy 分布映射及其对应的随机变量生成函数
        scipy_rv_map = {
            'MultivariateNormalDistribution': lambda dist, size: scipy_stats.multivariate_normal.rvs(
                mean=matrix2numpy(dist.mu).flatten(),
                cov=matrix2numpy(dist.sigma), size=size, random_state=rand_state),
            'MultivariateBetaDistribution': lambda dist, size: scipy_stats.dirichlet.rvs(
                alpha=list2numpy(dist.alpha, float).flatten(), size=size, random_state=rand_state),
            'MultinomialDistribution': lambda dist, size: scipy_stats.multinomial.rvs(
                n=int(dist.n), p=list2numpy(dist.p, float).flatten(), size=size, random_state=rand_state)
        }

        # 定义不同分布的样本形状获取函数
        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        # 获取所有支持的分布名称列表
        dist_list = scipy_rv_map.keys()

        # 如果给定的分布不在支持的分布列表中，返回 None
        if dist.__class__.__name__ not in dist_list:
            return None

        # 调用相应分布的随机变量生成函数生成样本数据
        samples = scipy_rv_map[dist.__class__.__name__](dist, size)
        # 调整样本数据的形状以匹配指定的大小和分布形状
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))

class SampleJointNumpy:
    """Returns the sample from numpy of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        # 调用内部方法 _sample_numpy 来获取 numpy 分布的样本数据
        return cls._sample_numpy(dist, size, seed)
    # 定义一个静态方法，用于从 NumPy 中进行采样
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

        import numpy  # 导入 NumPy 库

        # 根据种子值创建随机数生成器对象
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed

        # 定义一个字典，包含不同分布对应的随机变量生成函数
        numpy_rv_map = {
            'MultivariateNormalDistribution': lambda dist, size: rand_state.multivariate_normal(
                mean=matrix2numpy(dist.mu, float).flatten(),  # 提取多元正态分布的均值
                cov=matrix2numpy(dist.sigma, float), size=size),  # 提取多元正态分布的协方差矩阵
            'MultivariateBetaDistribution': lambda dist, size: rand_state.dirichlet(
                alpha=list2numpy(dist.alpha, float).flatten(), size=size),  # 提取多元贝塔分布的参数 alpha
            'MultinomialDistribution': lambda dist, size: rand_state.multinomial(
                n=int(dist.n), pvals=list2numpy(dist.p, float).flatten(), size=size)  # 提取多项分布的参数 n 和概率向量 p
        }

        # 定义一个字典，存储不同分布对应的样本形状函数
        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,  # 多元正态分布的样本形状
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,  # 多元贝塔分布的样本形状
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape  # 多项分布的样本形状
        }

        # 获取支持的分布列表
        dist_list = numpy_rv_map.keys()

        # 如果给定的分布类型不在支持的列表中，则返回空值
        if dist.__class__.__name__ not in dist_list:
            return None

        # 使用对应的随机变量生成函数生成样本，并调整形状为指定大小
        samples = numpy_rv_map[dist.__class__.__name__](dist, prod(size))
        
        # 将生成的样本调整为指定大小加上对应分布的样本形状
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))
class SampleJointPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        # 使用类方法 _sample_pymc 处理参数，返回样本
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

        try:
            import pymc  # 尝试导入 pymc 库
        except ImportError:
            import pymc3 as pymc  # 如果导入失败，则导入 pymc3 库
        # 定义不同分布类型到 PyMC 随机变量的映射
        pymc_rv_map = {
            'MultivariateNormalDistribution': lambda dist:
                pymc.MvNormal('X', mu=matrix2numpy(dist.mu, float).flatten(),
                cov=matrix2numpy(dist.sigma, float), shape=(1, dist.mu.shape[0])),
            'MultivariateBetaDistribution': lambda dist:
                pymc.Dirichlet('X', a=list2numpy(dist.alpha, float).flatten()),
            'MultinomialDistribution': lambda dist:
                pymc.Multinomial('X', n=int(dist.n),
                p=list2numpy(dist.p, float).flatten(), shape=(1, len(dist.p)))
        }

        # 定义不同分布类型对应的样本形状
        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        # 获取所有支持的分布类型
        dist_list = pymc_rv_map.keys()

        # 如果给定的分布类型不在支持的列表中，返回 None
        if dist.__class__.__name__ not in dist_list:
            return None

        import logging
        logging.getLogger("pymc3").setLevel(logging.ERROR)  # 设置 pymc3 的日志级别为 ERROR

        # 使用 PyMC 模块创建上下文管理器
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)  # 创建指定分布类型的 PyMC 随机变量
            # 从 PyMC 中抽样得到样本，reshape 成指定大小
            samples = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)[:]['X']
        
        # 返回 reshape 后的样本
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))


_get_sample_class_jrv = {
    'scipy': SampleJointScipy,
    'pymc3': SampleJointPymc,  # 将 'pymc3' 映射到 SampleJointPymc 类
    'pymc': SampleJointPymc,   # 将 'pymc' 映射到 SampleJointPymc 类
    'numpy': SampleJointNumpy
}

class JointDistribution(Distribution, NamedArgsMixin):
    """
    Represented by the random variables part of the joint distribution.
    Contains methods for PDF, CDF, sampling, marginal densities, etc.
    """

    _argnames = ('pdf', )

    def __new__(cls, *args):
        # 将参数转换为 sympy 对象并创建 ImmutableMatrix 对象
        args = list(map(sympify, args))
        for i in range(len(args)):
            if isinstance(args[i], list):
                args[i] = ImmutableMatrix(args[i])
        return Basic.__new__(cls, *args)

    @property
    def domain(self):
        # 返回由符号列表构成的 ProductDomain 对象
        return ProductDomain(self.symbols)

    @property
    def pdf(self):
        # 返回密度函数的第二个参数
        return self.density.args[1]
    # 定义一个方法 cdf，用于计算累积分布函数
    def cdf(self, other):
        # 如果参数 other 不是字典类型，则抛出数值错误异常
        if not isinstance(other, dict):
            raise ValueError("%s should be of type dict, got %s"%(other, type(other)))
        
        # 获取字典 other 的键集合作为随机变量集合
        rvs = other.keys()
        
        # 获取 self 对象的 domain 属性的 set 属性的 sets 属性
        _set = self.domain.set.sets
        
        # 计算表达式 expr，调用 self 对象的 symbols 属性中各项参数的第一个参数值
        expr = self.pdf(tuple(i.args[0] for i in self.symbols))
        
        # 遍历 other 字典中的每个随机变量
        for i in range(len(other)):
            # 如果随机变量 rvs[i] 是连续型
            if rvs[i].is_Continuous:
                # 创建密度对象，积分 expr 在 (rvs[i], _set[i].inf, other[rvs[i]]) 区间上
                density = Integral(expr, (rvs[i], _set[i].inf, other[rvs[i]]))
            # 如果随机变量 rvs[i] 是离散型
            elif rvs[i].is_Discrete:
                # 创建密度对象，求和 expr 在 (rvs[i], _set[i].inf, other[rvs[i]]) 区间上
                density = Sum(expr, (rvs[i], _set[i].inf, other[rvs[i]]))
        
        # 返回计算得到的密度对象
        return density

    # 定义一个方法 sample，用于从分布中抽样
    def sample(self, size=(), library='scipy', seed=None):
        """ A random realization from the distribution """
        
        # 支持的抽样库列表
        libraries = ('scipy', 'numpy', 'pymc3', 'pymc')
        
        # 如果指定的库不在支持列表中，则抛出未实现错误异常
        if library not in libraries:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                        % str(library))
        
        # 尝试导入指定的抽样库，如果失败则抛出值错误异常
        if not import_module(library):
            raise ValueError("Failed to import %s" % library)
        
        # 根据指定的库获取抽样类的实例 samps
        samps = _get_sample_class_jrv[library](self, size, seed=seed)
        
        # 如果成功获取了抽样实例，则返回抽样结果
        if samps is not None:
            return samps
        
        # 如果未能获取抽样实例，则抛出未实现错误异常
        raise NotImplementedError(
                "Sampling for %s is not currently implemented from %s"
                % (self.__class__.__name__, library)
                )

    # 定义一个魔法方法 __call__，使对象可以被调用
    def __call__(self, *args):
        # 调用对象的 pdf 方法，传递所有参数给它，并返回其结果
        return self.pdf(*args)
class JointRandomSymbol(RandomSymbol):
    """
    Representation of random symbols with joint probability distributions
    to allow indexing."
    """
    def __getitem__(self, key):
        # 如果联合概率空间是 JointPSpace 类型
        if isinstance(self.pspace, JointPSpace):
            # 检查索引是否超出组件数量的范围
            if (self.pspace.component_count <= key) == True:
                # 抛出值错误，指明索引超出范围
                raise ValueError("Index keys for %s can only up to %s." %
                    (self.name, self.pspace.component_count - 1))
            # 返回以 key 索引的 Indexed 对象
            return Indexed(self, key)



class MarginalDistribution(Distribution):
    """
    Represents the marginal distribution of a joint probability space.

    Initialised using a probability distribution and random variables(or
    their indexed components) which should be a part of the resultant
    distribution.
    """

    def __new__(cls, dist, *rvs):
        # 如果只有一个参数且为可迭代对象，则将其解包为元组
        if len(rvs) == 1 and iterable(rvs[0]):
            rvs = tuple(rvs[0])
        # 检查所有 rv 是否为 Indexed 或 RandomSymbol 类型
        if not all(isinstance(rv, (Indexed, RandomSymbol)) for rv in rvs):
            # 如果不是，抛出值错误，说明只能用随机变量或索引随机变量初始化边缘分布
            raise ValueError(filldedent('''Marginal distribution can be
             intitialised only in terms of random variables or indexed random
             variables'''))
        # 将 rv 转换为 Tuple
        rvs = Tuple.fromiter(rv for rv in rvs)
        # 如果 dist 不是 JointDistribution 类型且其内的随机符号数量为 0
        if not isinstance(dist, JointDistribution) and len(random_symbols(dist)) == 0:
            # 直接返回 dist
            return dist
        # 否则调用基类的 __new__ 方法来创建实例
        return Basic.__new__(cls, dist, rvs)

    def check(self):
        # 留空，没有具体实现内容
        pass

    @property
    def set(self):
        # 获取所有属于 RandomSymbol 类型的 rv 的概率空间集合，并返回其乘积集
        rvs = [i for i in self.args[1] if isinstance(i, RandomSymbol)]
        return ProductSet(*[rv.pspace.set for rv in rvs])

    @property
    def symbols(self):
        # 获取所有 rv 的概率空间符号集合，并返回其集合
        rvs = self.args[1]
        return {rv.pspace.symbol for rv in rvs}

    def pdf(self, *x):
        # 获取表达式和 rv
        expr, rvs = self.args[0], self.args[1]
        # marginalise_out 包含那些在表达式中是随机符号但不在 rv 中的符号
        marginalise_out = [i for i in random_symbols(expr) if i not in rvs]
        # 如果表达式是 JointDistribution 类型
        if isinstance(expr, JointDistribution):
            # 获取域的参数数量
            count = len(expr.domain.args)
            # 创建虚拟变量 x，syms 是使用 Indexed(x, i) 构成的元组
            x = Dummy('x', real=True)
            syms = tuple(Indexed(x, i) for i in count)
            # 计算 JointDistribution 的概率密度函数
            expr = expr.pdf(syms)
        else:
            # 否则，syms 是 rv 的概率空间符号或其参数的元组
            syms = tuple(rv.pspace.symbol if isinstance(rv, RandomSymbol) else rv.args[0] for rv in rvs)
        # 返回 Lambda 表达式的结果，使用传入的 x 来计算
        return Lambda(syms, self.compute_pdf(expr, marginalise_out))(*x)

    def compute_pdf(self, expr, rvs):
        # 对于每个 rv 进行循环
        for rv in rvs:
            lpdf = 1
            # 如果 rv 是 RandomSymbol 类型，获取其概率密度函数
            if isinstance(rv, RandomSymbol):
                lpdf = rv.pspace.pdf
            # marginalise_out 表达式是对 rv 进行边缘化的结果
            expr = self.marginalise_out(expr*lpdf, rv)
        # 返回最终计算得到的表达式
        return expr
    # 定义一个方法 `marginalise_out`，用于边缘化给定的表达式 `expr` 关于随机变量 `rv`
    def marginalise_out(self, expr, rv):
        # 导入必要的模块 `Sum`，用于求和操作
        from sympy.concrete.summations import Sum
        # 如果 `rv` 是随机符号 `RandomSymbol` 类型
        if isinstance(rv, RandomSymbol):
            # 取得随机变量 `rv` 的概率空间的集合
            dom = rv.pspace.set
        # 如果 `rv` 是索引类型 `Indexed`
        elif isinstance(rv, Indexed):
            # 获取基础组件域的组件域，作为 `dom`
            dom = rv.base.component_domain(rv.pspace.component_domain(rv.args[1]))
        # 将表达式 `expr` 中的 `rv` 替换为其概率空间的符号
        expr = expr.xreplace({rv: rv.pspace.symbol})
        # 如果随机变量 `rv` 的概率空间是连续的
        if rv.pspace.is_Continuous:
            # TODO: 修改以支持对所有类型集合的积分操作
            # 对 `expr` 进行积分操作，积分变量为 `rv.pspace.symbol`，积分范围为 `dom`
            expr = Integral(expr, (rv.pspace.symbol, dom))
        # 如果随机变量 `rv` 的概率空间是离散的
        elif rv.pspace.is_Discrete:
            # 将 `expr` 转换为求和表达式 `Sum`，求和变量为 `rv.pspace.symbol`，求和范围为 `dom`
            if dom in (S.Integers, S.Naturals, S.Naturals0):
                dom = (dom.inf, dom.sup)
            expr = Sum(expr, (rv.pspace.symbol, dom))
        # 返回处理后的表达式 `expr`
        return expr

    # 定义一个方法 `__call__`，其作用是调用 `pdf` 方法，并返回其结果
    def __call__(self, *args):
        return self.pdf(*args)
```