# `D:\src\scipysrc\sympy\sympy\stats\sampling\sample_pymc.py`

```
# 从 functools 模块导入 singledispatch 装饰器
from functools import singledispatch
# 从 sympy.external 模块导入 import_module 函数
from sympy.external import import_module
# 从 sympy.stats.crv_types 模块导入多个分布类
from sympy.stats.crv_types import BetaDistribution, CauchyDistribution, ChiSquaredDistribution, ExponentialDistribution, \
    GammaDistribution, LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, \
    GaussianInverseDistribution
# 从 sympy.stats.drv_types 模块导入多个分布类
from sympy.stats.drv_types import PoissonDistribution, GeometricDistribution, NegativeBinomialDistribution
# 从 sympy.stats.frv_types 模块导入多个分布类
from sympy.stats.frv_types import BinomialDistribution, BernoulliDistribution

# 尝试导入 pymc 模块，如果失败则使用 import_module 函数导入 pymc3
try:
    import pymc
except ImportError:
    pymc = import_module('pymc3')

# 定义一个 singledispatch 的装饰函数 do_sample_pymc，接受一个参数 dist
@singledispatch
def do_sample_pymc(dist):
    return None

# 注册 BetaDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(BetaDistribution)
def _(dist: BetaDistribution):
    return pymc.Beta('X', alpha=float(dist.alpha), beta=float(dist.beta))

# 注册 CauchyDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(CauchyDistribution)
def _(dist: CauchyDistribution):
    return pymc.Cauchy('X', alpha=float(dist.x0), beta=float(dist.gamma))

# 注册 ChiSquaredDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution):
    return pymc.ChiSquared('X', nu=float(dist.k))

# 注册 ExponentialDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(ExponentialDistribution)
def _(dist: ExponentialDistribution):
    return pymc.Exponential('X', lam=float(dist.rate))

# 注册 GammaDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(GammaDistribution)
def _(dist: GammaDistribution):
    return pymc.Gamma('X', alpha=float(dist.k), beta=1 / float(dist.theta))

# 注册 LogNormalDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(LogNormalDistribution)
def _(dist: LogNormalDistribution):
    return pymc.Lognormal('X', mu=float(dist.mean), sigma=float(dist.std))

# 注册 NormalDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(NormalDistribution)
def _(dist: NormalDistribution):
    return pymc.Normal('X', float(dist.mean), float(dist.std))

# 注册 GaussianInverseDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(GaussianInverseDistribution)
def _(dist: GaussianInverseDistribution):
    return pymc.Wald('X', mu=float(dist.mean), lam=float(dist.shape))

# 注册 ParetoDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(ParetoDistribution)
def _(dist: ParetoDistribution):
    return pymc.Pareto('X', alpha=float(dist.alpha), m=float(dist.xm))

# 注册 UniformDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(UniformDistribution)
def _(dist: UniformDistribution):
    return pymc.Uniform('X', lower=float(dist.left), upper=float(dist.right))

# 注册 GeometricDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(GeometricDistribution)
def _(dist: GeometricDistribution):
    return pymc.Geometric('X', p=float(dist.p))

# 注册 NegativeBinomialDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(NegativeBinomialDistribution)
def _(dist: NegativeBinomialDistribution):
    return pymc.NegativeBinomial('X', mu=float((dist.p * dist.r) / (1 - dist.p)),
                                  alpha=float(dist.r))

# 注册 PoissonDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(PoissonDistribution)
def _(dist: PoissonDistribution):
    return pymc.Poisson('X', mu=float(dist.lamda))

# 注册 BernoulliDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(BernoulliDistribution)
def _(dist: BernoulliDistribution):
    return pymc.Bernoulli('X', p=float(dist.p))

# 注册 BinomialDistribution 分布类型到 do_sample_pymc 函数的处理函数
@do_sample_pymc.register(BinomialDistribution)
def _(dist: BinomialDistribution):
    return pymc.Binomial('X', n=int(dist.n), p=float(dist.p))
```