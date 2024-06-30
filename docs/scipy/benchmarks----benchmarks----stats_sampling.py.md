# `D:\src\scipysrc\scipy\benchmarks\benchmarks\stats_sampling.py`

```
# 导入 NumPy 库，并将其重命名为 np
import numpy as np
# 从当前目录下的 common 模块中导入 Benchmark 和 safe_import 函数
from .common import Benchmark, safe_import

# 使用 safe_import 上下文管理器导入 scipy 库中的 stats 模块
with safe_import():
    from scipy import stats
# 使用 safe_import 上下文管理器导入 scipy 库中的 stats 模块中的 sampling 子模块
with safe_import():
    from scipy.stats import sampling
# 使用 safe_import 上下文管理器导入 scipy 库中的 special 模块
with safe_import():
    from scipy import special


# Beta 分布，参数 a=2, b=3
class contdist1:
    def __init__(self):
        self.mode = 1/3

    # 概率密度函数
    def pdf(self, x):
        return 12 * x * (1-x)**2

    # 概率密度函数的导数
    def dpdf(self, x):
        return 12 * ((1-x)**2 - 2*x*(1-x))

    # 累积分布函数
    def cdf(self, x):
        return 12 * (x**2/2 - x**3/3 + x**4/4)

    # 支持区间
    def support(self):
        return 0, 1

    # 返回类的字符串表示形式
    def __repr__(self):
        return 'beta(2, 3)'


# 标准正态分布
class contdist2:
    def __init__(self):
        self.mode = 0

    # 概率密度函数
    def pdf(self, x):
        return 1./np.sqrt(2*np.pi) * np.exp(-0.5 * x*x)

    # 概率密度函数的导数
    def dpdf(self, x):
        return 1./np.sqrt(2*np.pi) * -x * np.exp(-0.5 * x*x)

    # 累积分布函数
    def cdf(self, x):
        return special.ndtr(x)

    # 返回类的字符串表示形式
    def __repr__(self):
        return 'norm(0, 1)'


# 使用分段线性函数作为变换密度的概率密度函数，其中 T = -1/sqrt
# 取自 UNU.RAN 测试套件（来自文件 t_tdr_ps.c）
class contdist3:
    def __init__(self, shift=0.):
        self.shift = shift
        self.mode = shift

    # 概率密度函数
    def pdf(self, x):
        x -= self.shift
        y = 1. / (abs(x) + 1.)
        return y * y

    # 概率密度函数的导数
    def dpdf(self, x):
        x -= self.shift
        y = 1. / (abs(x) + 1.)
        y = 2. * y * y * y
        return y if (x < 0.) else -y

    # 累积分布函数
    def cdf(self, x):
        x -= self.shift
        if x <= 0.:
            return 0.5 / (1. - x)
        return 1. - 0.5 / (1. + x)

    # 返回支持区间
    def support(self):
        return -1, 1

    # 返回类的字符串表示形式
    def __repr__(self):
        return f'sqrtlinshft({self.shift})'


# Sin 2 分布
#          /  0.05 + 0.45*(1 +sin(2 Pi x))  if |x| <= 1
#  f(x) = <
#          \  0        otherwise
# 取自 UNU.RAN 测试套件（来自文件 t_pinv.c）
class contdist4:
    def __init__(self):
        self.mode = 0

    # 概率密度函数
    def pdf(self, x):
        return 0.05 + 0.45 * (1 + np.sin(2*np.pi*x))

    # 概率密度函数的导数
    def dpdf(self, x):
        return 0.2 * 0.45 * (2*np.pi) * np.cos(2*np.pi*x)

    # 累积分布函数
    def cdf(self, x):
        return (0.05*(x + 1) +
                0.9*(1. + 2.*np.pi*(1 + x) - np.cos(2.*np.pi*x)) /
                (4.*np.pi))

    # 返回支持区间
    def support(self):
        return -1, 1

    # 返回类的字符串表示形式
    def __repr__(self):
        return 'sin2'


# Sin 10 分布
#          /  0.05 + 0.45*(1 +sin(2 Pi x))  if |x| <= 5
#  f(x) = <
#          \  0        otherwise
# 取自 UNU.RAN 测试套件（来自文件 t_pinv.c）
class contdist5:
    def __init__(self):
        self.mode = 0

    # 概率密度函数
    def pdf(self, x):
        return 0.2 * (0.05 + 0.45 * (1 + np.sin(2*np.pi*x)))

    # 概率密度函数的导数
    def dpdf(self, x):
        return 0.2 * 0.45 * (2*np.pi) * np.cos(2*np.pi*x)

    # 累积分布函数
    def cdf(self, x):
        return x/10. + 0.5 + 0.09/(2*np.pi) * (np.cos(10*np.pi) -
                                               np.cos(2*np.pi*x))

    # 返回支持区间
    def support(self):
        return -5, 5

    # 返回类的字符串表示形式
    def __repr__(self):
        return 'sin10'
# 创建包含多个连续分布对象的列表
allcontdists = [contdist1(), contdist2(), contdist3(), contdist3(10000.),
                contdist4(), contdist5()]


# 定义一个继承自 Benchmark 类的 TransformedDensityRejection 类
class TransformedDensityRejection(Benchmark):

    # 参数名列表
    param_names = ['dist', 'c']

    # 参数组合列表
    params = [allcontdists, [0., -0.5]]

    # 初始化方法，设置随机数生成器和转换密度拒绝采样对象
    def setup(self, dist, c):
        self.urng = np.random.default_rng(0xfaad7df1c89e050200dbe258636b3265)
        # 忽略运行时警告
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                # 创建 TransformedDensityRejection 对象
                self.rng = sampling.TransformedDensityRejection(
                    dist, c=c, random_state=self.urng
                )
            except sampling.UNURANError:
                # 如果 dist 不是 c=0 时 T-凹的，抛出未实现错误
                raise NotImplementedError(f"{dist} not T-concave for c={c}")

    # 测试设置 TransformedDensityRejection 对象的时间
    def time_tdr_setup(self, dist, c):
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            # 创建 TransformedDensityRejection 对象
            sampling.TransformedDensityRejection(
                dist, c=c, random_state=self.urng
            )

    # 测试生成随机变量的时间
    def time_tdr_rvs(self, dist, c):
        # 生成随机变量
        self.rng.rvs(100000)


# 定义一个继承自 Benchmark 类的 SimpleRatioUniforms 类
class SimpleRatioUniforms(Benchmark):

    # 参数名列表
    param_names = ['dist', 'cdf_at_mode']

    # 参数组合列表
    params = [allcontdists, [0, 1]]

    # 初始化方法，设置随机数生成器和简单比率均匀分布对象
    def setup(self, dist, cdf_at_mode):
        self.urng = np.random.default_rng(0xfaad7df1c89e050200dbe258636b3265)
        try:
            # 如果 cdf_at_mode 为真，计算在 mode 处的累积分布函数值，否则为 None
            if cdf_at_mode:
                cdf_at_mode = dist.cdf(dist.mode)
            else:
                cdf_at_mode = None
            # 创建 SimpleRatioUniforms 对象
            self.rng = sampling.SimpleRatioUniforms(
                dist, mode=dist.mode,
                cdf_at_mode=cdf_at_mode,
                random_state=self.urng
            )
        except sampling.UNURANError:
            # 如果 dist 不是 T-凹的，抛出未实现错误
            raise NotImplementedError(f"{dist} not T-concave")

    # 测试设置 SimpleRatioUniforms 对象的时间
    def time_srou_setup(self, dist, cdf_at_mode):
        if cdf_at_mode:
            cdf_at_mode = dist.cdf(dist.mode)
        else:
            cdf_at_mode = None
        # 创建 SimpleRatioUniforms 对象
        sampling.SimpleRatioUniforms(
            dist, mode=dist.mode,
            cdf_at_mode=cdf_at_mode,
            random_state=self.urng
        )

    # 测试生成随机变量的时间
    def time_srou_rvs(self, dist, cdf_at_mode):
        # 生成随机变量
        self.rng.rvs(100000)


# 定义一个继承自 Benchmark 类的 NumericalInversePolynomial 类
class NumericalInversePolynomial(Benchmark):

    # 参数名列表
    param_names = ['dist']

    # 参数组合列表
    params = [allcontdists]

    # 初始化方法，设置随机数生成器和数值逆多项式对象
    def setup(self, dist):
        self.urng = np.random.default_rng(0xb235b58c1f616c59c18d8568f77d44d1)
        # 忽略运行时警告
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                # 创建 NumericalInversePolynomial 对象
                self.rng = sampling.NumericalInversePolynomial(
                    dist, random_state=self.urng
                )
            except sampling.UNURANError:
                # 如果创建失败，抛出未实现错误
                raise NotImplementedError(f"setup failed for {dist}")

    # 测试设置 NumericalInversePolynomial 对象的时间
    def time_pinv_setup(self, dist):
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            # 创建 NumericalInversePolynomial 对象
            sampling.NumericalInversePolynomial(
                dist, random_state=self.urng
            )
    # 定义一个方法 time_pinv_rvs，接收参数 self 和 dist
    def time_pinv_rvs(self, dist):
        # 使用 self.rng 调用 rvs 方法生成 100000 个随机数，但没有对结果进行任何处理或返回
        self.rng.rvs(100000)
# 定义 NumericalInverseHermite 类，继承自 Benchmark 基类，用于性能基准测试
class NumericalInverseHermite(Benchmark):

    # 参数名称列表，包括分布和阶数
    param_names = ['dist', 'order']
    # 参数值列表，分别为所有连续分布和阶数为3或5的列表
    params = [allcontdists, [3, 5]]

    # 初始化设置方法，接受 dist 和 order 参数
    def setup(self, dist, order):
        # 使用特定种子创建随机数生成器 urng
        self.urng = np.random.default_rng(0xb235b58c1f616c59c18d8568f77d44d1)
        # 屏蔽运行时警告，尝试创建 NumericalInverseHermite 对象 rng
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                self.rng = sampling.NumericalInverseHermite(
                    dist, order=order, random_state=self.urng
                )
            # 捕获 UNURANError 异常，抛出未实现的错误
            except sampling.UNURANError:
                raise NotImplementedError(f"setup failed for {dist}")

    # 测试 NumericalInverseHermite 对象的设置时间
    def time_hinv_setup(self, dist, order):
        # 屏蔽运行时警告，创建 NumericalInverseHermite 对象
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sampling.NumericalInverseHermite(
                dist, order=order, random_state=self.urng
            )

    # 测试 NumericalInverseHermite 对象的随机变量生成时间
    def time_hinv_rvs(self, dist, order):
        # 调用 NumericalInverseHermite 对象的 rvs 方法生成随机变量
        self.rng.rvs(100000)


# 定义 DiscreteAliasUrn 类，继承自 Benchmark 基类，用于性能基准测试
class DiscreteAliasUrn(Benchmark):

    # 参数名称列表，仅包括 distribution
    param_names = ['distribution']

    # 参数值列表，包括多个离散分布的名称和对应的参数
    params = [
        # 一些有限域的离散分布的子集
        [['nhypergeom', (20, 7, 1)],
         ['hypergeom', (30, 12, 6)],
         ['nchypergeom_wallenius', (140, 80, 60, 0.5)],
         ['binom', (5, 0.4)]]
    ]

    # 初始化设置方法，接受 distribution 参数
    def setup(self, distribution):
        # 解构 distribution 元组，获取分布名称和参数
        distname, params = distribution
        # 使用 getattr 根据名称获取对应的统计分布对象 dist
        dist = getattr(stats, distname)
        # 计算分布的定义域 domain
        domain = dist.support(*params)
        # 使用特定种子创建随机数生成器 urng
        self.urng = np.random.default_rng(0x2fc9eb71cd5120352fa31b7a048aa867)
        # 创建定义域范围内的值 x
        x = np.arange(domain[0], domain[1] + 1)
        # 计算概率质量函数值 pv
        self.pv = dist.pmf(x, *params)
        # 使用 pv 和 urng 创建 DiscreteAliasUrn 对象 rng
        self.rng = sampling.DiscreteAliasUrn(self.pv, random_state=self.urng)

    # 测试 DiscreteAliasUrn 对象的设置时间
    def time_dau_setup(self, distribution):
        # 创建 DiscreteAliasUrn 对象
        sampling.DiscreteAliasUrn(self.pv, random_state=self.urng)

    # 测试 DiscreteAliasUrn 对象的随机变量生成时间
    def time_dau_rvs(self, distribution):
        # 调用 DiscreteAliasUrn 对象的 rvs 方法生成随机变量
        self.rng.rvs(100000)


# 定义 DiscreteGuideTable 类，继承自 Benchmark 基类，用于性能基准测试
class DiscreteGuideTable(Benchmark):

    # 参数名称列表，仅包括 distribution
    param_names = ['distribution']

    # 参数值列表，包括多个离散分布的名称和对应的参数
    params = [
        # 一些有限域的离散分布的子集
        [['nhypergeom', (20, 7, 1)],
         ['hypergeom', (30, 12, 6)],
         ['nchypergeom_wallenius', (140, 80, 60, 0.5)],
         ['binom', (5, 0.4)]]
    ]

    # 初始化设置方法，接受 distribution 参数
    def setup(self, distribution):
        # 解构 distribution 元组，获取分布名称和参数
        distname, params = distribution
        # 使用 getattr 根据名称获取对应的统计分布对象 dist
        dist = getattr(stats, distname)
        # 计算分布的定义域 domain
        domain = dist.support(*params)
        # 使用特定种子创建随机数生成器 urng
        self.urng = np.random.default_rng(0x2fc9eb71cd5120352fa31b7a048aa867)
        # 创建定义域范围内的值 x
        x = np.arange(domain[0], domain[1] + 1)
        # 计算概率质量函数值 pv
        self.pv = dist.pmf(x, *params)
        # 使用 pv 和 urng 创建 DiscreteGuideTable 对象 rng
        self.rng = sampling.DiscreteGuideTable(self.pv, random_state=self.urng)

    # 测试 DiscreteGuideTable 对象的设置时间
    def time_dgt_setup(self, distribution):
        # 创建 DiscreteGuideTable 对象
        sampling.DiscreteGuideTable(self.pv, random_state=self.urng)

    # 测试 DiscreteGuideTable 对象的随机变量生成时间
    def time_dgt_rvs(self, distribution):
        # 调用 DiscreteGuideTable 对象的 rvs 方法生成随机变量
        self.rng.rvs(100000)
```