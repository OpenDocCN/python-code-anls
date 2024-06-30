# `D:\src\scipysrc\sympy\sympy\stats\sampling\sample_scipy.py`

```
# 导入 functools 库中的 singledispatch 装饰器
from functools import singledispatch

# 导入 sympy 库中所需的模块和类
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import exp
from sympy.utilities.lambdify import lambdify
from sympy.external import import_module
from sympy.stats import DiscreteDistributionHandmade
from sympy.stats.crv import SingleContinuousDistribution
from sympy.stats.crv_types import ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
    LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, BetaDistribution, \
    StudentTDistribution, CauchyDistribution
from sympy.stats.drv_types import GeometricDistribution, LogarithmicDistribution, NegativeBinomialDistribution, \
    PoissonDistribution, SkellamDistribution, YuleSimonDistribution, ZetaDistribution
from sympy.stats.frv import SingleFiniteDistribution

# 尝试导入 scipy 库中的 stats 模块
scipy = import_module("scipy", import_kwargs={'fromlist':['stats']})

# 定义 singledispatch 的函数 do_sample_scipy，初始返回 None
@singledispatch
def do_sample_scipy(dist, size, seed):
    return None


# 为 SingleContinuousDistribution 类型注册 do_sample_scipy 的特定实现
@do_sample_scipy.register(SingleContinuousDistribution)
def _(dist: SingleContinuousDistribution, size, seed):
    # 导入 scipy.stats 模块
    import scipy.stats

    # 创建一个 Dummy 符号 z
    z = Dummy('z')
    # 将 sympy 的 dist.pdf(z) 转换为 numpy 和 scipy 的函数
    handmade_pdf = lambdify(z, dist.pdf(z), ['numpy', 'scipy'])

    # 定义一个继承自 scipy.stats.rv_continuous 的类 scipy_pdf
    class scipy_pdf(scipy.stats.rv_continuous):
        def _pdf(dist, x):
            return handmade_pdf(x)

    # 创建一个 scipy_pdf 对象，设置范围为 dist.set._inf 到 dist.set._sup
    scipy_rv = scipy_pdf(a=float(dist.set._inf),
                         b=float(dist.set._sup), name='scipy_pdf')
    # 使用 random_state 和 size 参数生成随机样本并返回
    return scipy_rv.rvs(size=size, random_state=seed)


# 为 ChiSquaredDistribution 类型注册 do_sample_scipy 的特定实现
@do_sample_scipy.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution, size, seed):
    # 使用 scipy.stats.chi2.rvs 生成卡方分布的随机样本
    return scipy.stats.chi2.rvs(df=float(dist.k), size=size, random_state=seed)


# 为 ExponentialDistribution 类型注册 do_sample_scipy 的特定实现
@do_sample_scipy.register(ExponentialDistribution)
def _(dist: ExponentialDistribution, size, seed):
    # 使用 scipy.stats.expon.rvs 生成指数分布的随机样本
    return scipy.stats.expon.rvs(scale=1 / float(dist.rate), size=size, random_state=seed)


# 为 GammaDistribution 类型注册 do_sample_scipy 的特定实现
@do_sample_scipy.register(GammaDistribution)
def _(dist: GammaDistribution, size, seed):
    # 使用 scipy.stats.gamma.rvs 生成 Gamma 分布的随机样本
    return scipy.stats.gamma.rvs(a=float(dist.k), scale=float(dist.theta), size=size, random_state=seed)


# 为 LogNormalDistribution 类型注册 do_sample_scipy 的特定实现
@do_sample_scipy.register(LogNormalDistribution)
def _(dist: LogNormalDistribution, size, seed):
    # 使用 scipy.stats.lognorm.rvs 生成对数正态分布的随机样本
    return scipy.stats.lognorm.rvs(scale=float(exp(dist.mean)), s=float(dist.std), size=size, random_state=seed)


# 为 NormalDistribution 类型注册 do_sample_scipy 的特定实现
@do_sample_scipy.register(NormalDistribution)
def _(dist: NormalDistribution, size, seed):
    # 使用 scipy.stats.norm.rvs 生成正态分布的随机样本
    return scipy.stats.norm.rvs(loc=float(dist.mean), scale=float(dist.std), size=size, random_state=seed)


# 为 ParetoDistribution 类型注册 do_sample_scipy 的特定实现
@do_sample_scipy.register(ParetoDistribution)
def _(dist: ParetoDistribution, size, seed):
    # (此处为未完成的实现，应该继续添加代码)
    # 导入 Stack Overflow 上提供的链接，以获取 Pareto 分布在 Python 中的定义和使用方法的详细信息
    return scipy.stats.pareto.rvs(b=float(dist.alpha), scale=float(dist.xm), size=size, random_state=seed)
    # 使用 scipy 库中的 pareto.rvs 方法生成符合 Pareto 分布的随机变量
    # 参数 b 表示分布的形状参数 alpha，scale 表示分布的尺度参数 xm
    # size 参数指定生成随机变量的数量，random_state 参数用于确定随机数生成的种子
@do_sample_scipy.register(StudentTDistribution)
def _(dist: StudentTDistribution, size, seed):
    # 使用 scipy 中的 t 分布进行随机采样
    return scipy.stats.t.rvs(df=float(dist.nu), size=size, random_state=seed)


@do_sample_scipy.register(UniformDistribution)
def _(dist: UniformDistribution, size, seed):
    # 使用 scipy 中的均匀分布进行随机采样
    # 参考文档：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
    return scipy.stats.uniform.rvs(loc=float(dist.left), scale=float(dist.right - dist.left), size=size, random_state=seed)


@do_sample_scipy.register(BetaDistribution)
def _(dist: BetaDistribution, size, seed):
    # 使用 scipy 中的贝塔分布进行随机采样
    # 使用相同的参数化方式
    return scipy.stats.beta.rvs(a=float(dist.alpha), b=float(dist.beta), size=size, random_state=seed)


@do_sample_scipy.register(CauchyDistribution)
def _(dist: CauchyDistribution, size, seed):
    # 使用 scipy 中的柯西分布进行随机采样
    return scipy.stats.cauchy.rvs(loc=float(dist.x0), scale=float(dist.gamma), size=size, random_state=seed)


# DRV:

@do_sample_scipy.register(DiscreteDistributionHandmade)
def _(dist: DiscreteDistributionHandmade, size, seed):
    # 导入必要的模块和函数
    from scipy.stats import rv_discrete
    # 创建手工制作的离散分布的概率质量函数
    z = Dummy('z')
    handmade_pmf = lambdify(z, dist.pdf(z), ['numpy', 'scipy'])

    # 定义一个继承自 rv_discrete 的类来包装手工制作的概率质量函数
    class scipy_pmf(rv_discrete):
        def _pmf(dist, x):
            return handmade_pmf(x)

    # 创建一个 scipy 的随机变量对象
    scipy_rv = scipy_pmf(a=float(dist.set._inf), b=float(dist.set._sup),
                         name='scipy_pmf')
    # 使用该对象进行随机采样
    return scipy_rv.rvs(size=size, random_state=seed)


@do_sample_scipy.register(GeometricDistribution)
def _(dist: GeometricDistribution, size, seed):
    # 使用 scipy 中的几何分布进行随机采样
    return scipy.stats.geom.rvs(p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(LogarithmicDistribution)
def _(dist: LogarithmicDistribution, size, seed):
    # 使用 scipy 中的对数分布进行随机采样
    return scipy.stats.logser.rvs(p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(NegativeBinomialDistribution)
def _(dist: NegativeBinomialDistribution, size, seed):
    # 使用 scipy 中的负二项分布进行随机采样
    return scipy.stats.nbinom.rvs(n=float(dist.r), p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(PoissonDistribution)
def _(dist: PoissonDistribution, size, seed):
    # 使用 scipy 中的泊松分布进行随机采样
    return scipy.stats.poisson.rvs(mu=float(dist.lamda), size=size, random_state=seed)


@do_sample_scipy.register(SkellamDistribution)
def _(dist: SkellamDistribution, size, seed):
    # 使用 scipy 中的 Skellam 分布进行随机采样
    return scipy.stats.skellam.rvs(mu1=float(dist.mu1), mu2=float(dist.mu2), size=size, random_state=seed)


@do_sample_scipy.register(YuleSimonDistribution)
def _(dist: YuleSimonDistribution, size, seed):
    # 使用 scipy 中的 Yule-Simon 分布进行随机采样
    return scipy.stats.yulesimon.rvs(alpha=float(dist.rho), size=size, random_state=seed)


@do_sample_scipy.register(ZetaDistribution)
def _(dist: ZetaDistribution, size, seed):
    # 使用 scipy 中的 Zipf 分布进行随机采样
    return scipy.stats.zipf.rvs(a=float(dist.s), size=size, random_state=seed)


# FRV:

@do_sample_scipy.register(SingleFiniteDistribution)
def _(dist: SingleFiniteDistribution, size, seed):
    # scipy 可以处理自定义分布
    from scipy.stats import rv_discrete
    density_ = dist.dict
    x, y = [], []
    # 遍历 density_ 字典中的键值对，其中 k 是键，v 是值
    for k, v in density_.items():
        # 将键 k 转换为整数，并添加到 x 列表中
        x.append(int(k))
        # 将值 v 转换为浮点数，并添加到 y 列表中
        y.append(float(v))
    # 使用 x 和 y 列表创建一个离散随机变量的 scipy_rv 对象
    scipy_rv = rv_discrete(name='scipy_rv', values=(x, y))
    # 返回从 scipy_rv 生成的随机样本，样本数量为 size，随机种子为 seed
    return scipy_rv.rvs(size=size, random_state=seed)
```