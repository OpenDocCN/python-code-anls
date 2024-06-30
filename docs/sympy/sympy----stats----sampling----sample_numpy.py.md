# `D:\src\scipysrc\sympy\sympy\stats\sampling\sample_numpy.py`

```
# 从 functools 模块导入 singledispatch 装饰器，用于定义单分派泛型函数
from functools import singledispatch

# 从 sympy.external 模块中导入 import_module 函数
from sympy.external import import_module

# 从 sympy.stats.crv_types 模块中导入多种分布类
from sympy.stats.crv_types import BetaDistribution, ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
    LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, FDistributionDistribution, GumbelDistribution, LaplaceDistribution, \
    LogisticDistribution, RayleighDistribution, TriangularDistribution

# 从 sympy.stats.drv_types 模块中导入多种分布类
from sympy.stats.drv_types import GeometricDistribution, PoissonDistribution, ZetaDistribution

# 从 sympy.stats.frv_types 模块中导入多种分布类
from sympy.stats.frv_types import BinomialDistribution, HypergeometricDistribution

# 使用 import_module 函数导入 numpy 并赋值给 numpy 变量
numpy = import_module('numpy')

# 定义一个单分派泛型函数 do_sample_numpy，接受分布对象 dist、样本大小 size 和随机状态 rand_state
@singledispatch
def do_sample_numpy(dist, size, rand_state):
    return None

# 为 BetaDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(BetaDistribution)
def _(dist: BetaDistribution, size, rand_state):
    return rand_state.beta(a=float(dist.alpha), b=float(dist.beta), size=size)

# 为 ChiSquaredDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution, size, rand_state):
    return rand_state.chisquare(df=float(dist.k), size=size)

# 为 ExponentialDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(ExponentialDistribution)
def _(dist: ExponentialDistribution, size, rand_state):
    return rand_state.exponential(1 / float(dist.rate), size=size)

# 为 FDistributionDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(FDistributionDistribution)
def _(dist: FDistributionDistribution, size, rand_state):
    return rand_state.f(dfnum=float(dist.d1), dfden=float(dist.d2), size=size)

# 为 GammaDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(GammaDistribution)
def _(dist: GammaDistribution, size, rand_state):
    return rand_state.gamma(shape=float(dist.k), scale=float(dist.theta), size=size)

# 为 GumbelDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(GumbelDistribution)
def _(dist: GumbelDistribution, size, rand_state):
    return rand_state.gumbel(loc=float(dist.mu), scale=float(dist.beta), size=size)

# 为 LaplaceDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(LaplaceDistribution)
def _(dist: LaplaceDistribution, size, rand_state):
    return rand_state.laplace(loc=float(dist.mu), scale=float(dist.b), size=size)

# 为 LogisticDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(LogisticDistribution)
def _(dist: LogisticDistribution, size, rand_state):
    return rand_state.logistic(loc=float(dist.mu), scale=float(dist.s), size=size)

# 为 LogNormalDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(LogNormalDistribution)
def _(dist: LogNormalDistribution, size, rand_state):
    return rand_state.lognormal(mean=float(dist.mean), sigma=float(dist.std), size=size)

# 为 NormalDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(NormalDistribution)
def _(dist: NormalDistribution, size, rand_state):
    return rand_state.normal(loc=float(dist.mean), scale=float(dist.std), size=size)

# 为 RayleighDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(RayleighDistribution)
def _(dist: RayleighDistribution, size, rand_state):
    return rand_state.rayleigh(scale=float(dist.sigma), size=size)

# 为 ParetoDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(ParetoDistribution)
def _(dist: ParetoDistribution, size, rand_state):
    return (numpy.random.pareto(a=float(dist.alpha), size=size) + 1) * float(dist.xm)

# 为 TriangularDistribution 类型的分布注册 do_sample_numpy 的实现
@do_sample_numpy.register(TriangularDistribution)
# 对三角分布进行抽样，使用 numpy 的随机状态对象 rand_state
@do_sample_numpy.register(TriangularDistribution)
def _(dist: TriangularDistribution, size, rand_state):
    # 调用 rand_state.triangular 方法，指定三角分布的左端点、众数和右端点，抽取指定 size 大小的样本
    return rand_state.triangular(left=float(dist.a), mode=float(dist.b), right=float(dist.c), size=size)


# 对均匀分布进行抽样，使用 numpy 的随机状态对象 rand_state
@do_sample_numpy.register(UniformDistribution)
def _(dist: UniformDistribution, size, rand_state):
    # 调用 rand_state.uniform 方法，指定均匀分布的上下界，抽取指定 size 大小的样本
    return rand_state.uniform(low=float(dist.left), high=float(dist.right), size=size)


# 对几何分布进行抽样，使用 numpy 的随机状态对象 rand_state
@do_sample_numpy.register(GeometricDistribution)
def _(dist: GeometricDistribution, size, rand_state):
    # 调用 rand_state.geometric 方法，指定几何分布的概率 p，抽取指定 size 大小的样本
    return rand_state.geometric(p=float(dist.p), size=size)


# 对泊松分布进行抽样，使用 numpy 的随机状态对象 rand_state
@do_sample_numpy.register(PoissonDistribution)
def _(dist: PoissonDistribution, size, rand_state):
    # 调用 rand_state.poisson 方法，指定泊松分布的参数 lambda，抽取指定 size 大小的样本
    return rand_state.poisson(lam=float(dist.lamda), size=size)


# 对齐普夫分布进行抽样，使用 numpy 的随机状态对象 rand_state
@do_sample_numpy.register(ZetaDistribution)
def _(dist: ZetaDistribution, size, rand_state):
    # 调用 rand_state.zipf 方法，指定齐普夫分布的参数 a，抽取指定 size 大小的样本
    return rand_state.zipf(a=float(dist.s), size=size)


# 对二项分布进行抽样，使用 numpy 的随机状态对象 rand_state
@do_sample_numpy.register(BinomialDistribution)
def _(dist: BinomialDistribution, size, rand_state):
    # 调用 rand_state.binomial 方法，指定二项分布的参数 n 和 p，抽取指定 size 大小的样本
    return rand_state.binomial(n=int(dist.n), p=float(dist.p), size=size)


# 对超几何分布进行抽样，使用 numpy 的随机状态对象 rand_state
@do_sample_numpy.register(HypergeometricDistribution)
def _(dist: HypergeometricDistribution, size, rand_state):
    # 调用 rand_state.hypergeometric 方法，指定超几何分布的参数 ngood, nbad, nsample，抽取指定 size 大小的样本
    return rand_state.hypergeometric(ngood=int(dist.N), nbad=int(dist.m), nsample=int(dist.n), size=size)
```