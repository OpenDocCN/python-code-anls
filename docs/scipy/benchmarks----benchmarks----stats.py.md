# `D:\src\scipysrc\scipy\benchmarks\benchmarks\stats.py`

```
# 导入警告模块，用于处理警告信息
import warnings

# 导入 NumPy 库，并将其命名为 np
import numpy as np

# 从当前目录中的 common 模块中导入 Benchmark 类、safe_import 函数、is_xslow 函数
from .common import Benchmark, safe_import, is_xslow

# 使用 safe_import 上下文管理器安全导入 scipy.stats 模块，并将其命名为 stats
with safe_import():
    import scipy.stats as stats

# 使用 safe_import 上下文管理器安全导入 scipy.stats._distr_params 模块中的 distcont 和 distdiscrete 对象
with safe_import():
    from scipy.stats._distr_params import distcont, distdiscrete

# 尝试导入内置的 itertools 模块中的 compress 函数
try:  
    from itertools import compress
except ImportError:
    pass


# 定义 Anderson_KSamp 类，继承自 Benchmark 类
class Anderson_KSamp(Benchmark):
    
    # 设置函数，在测试前生成随机数据
    def setup(self, *args):
        self.rand = [np.random.normal(loc=i, size=1000) for i in range(3)]

    # 测试函数，忽略 UserWarning 警告
    def time_anderson_ksamp(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            stats.anderson_ksamp(self.rand)


# 定义 CorrelationFunctions 类，继承自 Benchmark 类
class CorrelationFunctions(Benchmark):
    
    # 参数名列表
    param_names = ['alternative']
    
    # 参数列表
    params = [
        ['two-sided', 'less', 'greater']
    ]

    # 设置函数，生成随机数据
    def setup(self, mode):
        a = np.random.rand(2,2) * 10
        self.a = a

    # 测试 Fisher's Exact Test 函数
    def time_fisher_exact(self, alternative):
        stats.fisher_exact(self.a, alternative=alternative)

    # 测试 Barnard's Exact Test 函数
    def time_barnard_exact(self, alternative):
        stats.barnard_exact(self.a, alternative=alternative)

    # 测试 Boschloo's Exact Test 函数
    def time_boschloo_exact(self, alternative):
        stats.boschloo_exact(self.a, alternative=alternative)


# 定义 ANOVAFunction 类，继承自 Benchmark 类
class ANOVAFunction(Benchmark):
    
    # 设置函数，在测试前生成随机数据
    def setup(self):
        rng = np.random.default_rng(12345678)
        self.a = rng.random((6,3)) * 10
        self.b = rng.random((6,3)) * 10
        self.c = rng.random((6,3)) * 10

    # 测试单因素方差分析函数
    def time_f_oneway(self):
        stats.f_oneway(self.a, self.b, self.c)
        stats.f_oneway(self.a, self.b, self.c, axis=1)


# 定义 Kendalltau 类，继承自 Benchmark 类
class Kendalltau(Benchmark):
    
    # 参数名列表
    param_names = ['nan_policy','method','variant']
    
    # 参数列表
    params = [
        ['propagate', 'raise', 'omit'],
        ['auto', 'asymptotic', 'exact'],
        ['b', 'c']
    ]

    # 设置函数，生成随机数据
    def setup(self, nan_policy, method, variant):
        rng = np.random.default_rng(12345678)
        a = np.arange(200)
        rng.shuffle(a)
        b = np.arange(200)
        rng.shuffle(b)
        self.a = a
        self.b = b

    # 测试 Kendall Tau 相关系数函数
    def time_kendalltau(self, nan_policy, method, variant):
        stats.kendalltau(self.a, self.b, nan_policy=nan_policy,
                         method=method, variant=variant)


# 定义 KS 类，继承自 Benchmark 类
class KS(Benchmark):
    
    # 参数名列表
    param_names = ['alternative', 'mode']
    
    # 参数列表
    params = [
        ['two-sided', 'less', 'greater'],
        ['auto', 'exact', 'asymp'],
    ]

    # 设置函数，生成随机数据
    def setup(self, alternative, mode):
        rng = np.random.default_rng(0x2e7c964ff9a5cd6be22014c09f1dbba9)
        self.a = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
        self.b = stats.norm.rvs(loc=8, scale=10, size=500, random_state=rng)

    # 测试单样本 Kolmogorov-Smirnov 检验函数
    def time_ks_1samp(self, alternative, mode):
        stats.ks_1samp(self.a, stats.norm.cdf,
                       alternative=alternative, mode=mode)

    # 测试双样本 Kolmogorov-Smirnov 检验函数
    def time_ks_2samp(self, alternative, mode):
        stats.ks_2samp(self.a, self.b, alternative=alternative, mode=mode)


# 定义 RankSums 类，继承自 Benchmark 类
class RankSums(Benchmark):
    
    # 参数名列表
    param_names = ['alternative']
    
    # 参数列表
    params = [
        ['two-sided', 'less', 'greater']
    ]

# 以下部分没有实现具体的测试函数，因此没有进一步的注释
    # 定义一个方法 `setup`，用于设置对象的初始状态，接受一个参数 `alternative`
    def setup(self, alternative):
        # 使用指定种子创建一个随机数生成器 rng
        rng = np.random.default_rng(0xb6acd7192d6e5da0f68b5d8ab8ce7af2)
        # 使用 rng 生成均匀分布的随机数，存储到对象的属性 self.u1 中，范围在 [-1, 1)，共 200 个数
        self.u1 = rng.uniform(-1, 1, 200)
        # 使用 rng 生成均匀分布的随机数，存储到对象的属性 self.u2 中，范围在 [-0.5, 1.5)，共 300 个数
        self.u2 = rng.uniform(-0.5, 1.5, 300)

    # 定义一个方法 `time_ranksums`，用于计算两组数据的秩和检验，接受一个参数 `alternative`
    def time_ranksums(self, alternative):
        # 调用 scipy.stats 库中的 ranksums 函数，传入 self.u1 和 self.u2 作为参数，以及指定的 alternative 参数
        stats.ranksums(self.u1, self.u2, alternative=alternative)
# 定义一个类 BrunnerMunzel，继承自 Benchmark 类，用于执行 Brunner-Munzel 相关的基准测试
class BrunnerMunzel(Benchmark):
    # 参数名称列表
    param_names = ['alternative', 'nan_policy', 'distribution']
    # 参数组合列表
    params = [
        ['two-sided', 'less', 'greater'],  # alternative 参数的取值
        ['propagate', 'raise', 'omit'],    # nan_policy 参数的取值
        ['t', 'normal']                    # distribution 参数的取值
    ]

    # 设置方法，初始化测试所需的数据
    def setup(self, alternative, nan_policy, distribution):
        # 创建一个 RNG 对象，并设置种子
        rng = np.random.default_rng(0xb82c4db22b2818bdbc5dbe15ad7528fe)
        # 生成两组随机数，用于测试
        self.u1 = rng.uniform(-1, 1, 200)
        self.u2 = rng.uniform(-0.5, 1.5, 300)

    # 基准测试方法，执行 Brunner-Munzel 相关的统计测试
    def time_brunnermunzel(self, alternative, nan_policy, distribution):
        # 调用 scipy.stats 中的 brunnermunzel 函数进行统计测试
        stats.brunnermunzel(self.u1, self.u2, alternative=alternative,
                            distribution=distribution, nan_policy=nan_policy)


# 定义一个类 InferentialStats，用于执行多种推断统计方法的基准测试
class InferentialStats(Benchmark):
    # 设置方法，初始化测试所需的数据
    def setup(self):
        # 创建一个 RNG 对象，并设置种子
        rng = np.random.default_rng(0x13d756fadb635ae7f5a8d39bbfb0c931)
        # 生成三组符合正态分布的随机数，用于后续的 t 检验和 Friedman 检验
        self.a = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
        self.b = stats.norm.rvs(loc=8, scale=10, size=500, random_state=rng)
        self.c = stats.norm.rvs(loc=8, scale=20, size=500, random_state=rng)
        # 生成符合卡方分布的随机整数，用于卡方检验
        self.chisq = rng.integers(1, 20, 500)

    # 执行 t 检验（方差相同和方差不同情况）的基准测试方法
    def time_ttest_ind_same_var(self):
        # 使用 ttest_ind 函数进行同方差 t 检验
        stats.ttest_ind(self.a, self.b)
        # 使用 ttest_ind 函数进行异方差 t 检验
        stats.ttest_ind(self.a, self.b, equal_var=False)

    # 执行 t 检验（方差相同和方差不同情况）的基准测试方法
    def time_ttest_ind_diff_var(self):
        # 使用 ttest_ind 函数进行同方差 t 检验
        stats.ttest_ind(self.a, self.c)
        # 使用 ttest_ind 函数进行异方差 t 检验
        stats.ttest_ind(self.a, self.c, equal_var=False)

    # 执行卡方检验的基准测试方法
    def time_chisqure(self):
        # 使用 chisquare 函数进行卡方检验
        stats.chisquare(self.chisq)

    # 执行 Friedman 检验的基准测试方法
    def time_friedmanchisquare(self):
        # 使用 friedmanchisquare 函数进行 Friedman 检验
        stats.friedmanchisquare(self.a, self.b, self.c)

    # 执行 Epps-Singleton 两样本检验的基准测试方法
    def time_epps_singleton_2samp(self):
        # 使用 epps_singleton_2samp 函数进行 Epps-Singleton 两样本检验
        stats.epps_singleton_2samp(self.a, self.b)

    # 执行 Kruskal-Wallis 检验的基准测试方法
    def time_kruskal(self):
        # 使用 mstats.kruskal 函数进行 Kruskal-Wallis 检验
        stats.mstats.kruskal(self.a, self.b)


# 定义一个类 TruncnormStats，用于执行截断正态分布的基准测试
class TruncnormStats(Benchmark):
    # 参数名称列表
    param_names = ['case', 'moment']
    # 参数组合列表，case 对应于截断正态分布的不同案例，moment 对应于统计量名称
    params = [list(range(len(truncnorm_cases))), ['m', 'v', 's', 'k']]
    # 定义一个方法用于跟踪截断正态分布的统计数据误差
    def track_truncnorm_stats_error(self, case, moment):
        # 创建一个字典，将键 'm', 'v', 's', 'k' 映射到索引 2 到 5
        result_indices = dict(zip(['m', 'v', 's', 'k'], range(2, 6)))
        # 从截断正态分布案例数据中获取参考值
        ref = truncnorm_cases[case, result_indices[moment]]
        # 获取指定案例的截断正态分布的参数 a, b
        a, b = truncnorm_cases[case, 0:2]
        # 计算指定参数的截断正态分布的统计数据
        res = stats.truncnorm(a, b).stats(moments=moment)
        # 计算统计数据的绝对误差并返回
        return np.abs((res - ref)/ref)
class DistributionsAll(Benchmark):
    # 定义类 DistributionsAll，继承自 Benchmark

    # 使用集合转换来去除 distcont 和 distdiscrete 中重复出现的分布名，并排序后存入 dists 列表中
    dists = sorted(list(set([d[0] for d in distcont + distdiscrete])))

    # 参数名列表，包括分布名称和方法名称
    param_names = ['dist_name', 'method']
    # 参数值列表，分布名称来自 dists，方法名称包括 'pdf/pmf', 'logpdf/logpmf', 'cdf', 'logcdf', 'rvs', 'fit',
    # 'sf', 'logsf', 'ppf', 'isf', 'moment', 'stats_s', 'stats_v', 'stats_m', 'stats_k', 'stats_mvsk', 'entropy'
    params = [
        dists, ['pdf/pmf', 'logpdf/logpmf', 'cdf', 'logcdf', 'rvs', 'fit',
                'sf', 'logsf', 'ppf', 'isf', 'moment', 'stats_s', 'stats_v',
                'stats_m', 'stats_k', 'stats_mvsk', 'entropy']
    ]

    # 单独对 stats_mvsk 进行测试，因为它在 gh-11742 中有特别说明
    # `moment` 测试一个较高阶的矩（5阶）

    # 将 distcont 和 distdiscrete 中的分布数据合并成字典形式保存在 dist_data 中
    dist_data = dict(distcont + distdiscrete)

    # 可以为任何分布提供自定义形状值，格式为 `dist_name`: [shape1, shape2, ...]
    custom_input = {}

    # 这些是运行速度最慢的分布
    slow_dists = ['nct', 'ncx2', 'argus', 'cosine', 'foldnorm', 'gausshyper',
                  'kappa4', 'invgauss', 'wald', 'vonmises_line', 'ksone',
                  'genexpon', 'exponnorm', 'recipinvgauss', 'vonmises',
                  'foldcauchy', 'kstwo', 'levy_stable', 'skewnorm',
                  'studentized_range']
    # slow_methods 包含 'moment'，因为它需要额外的测试
    # 设置函数，用于初始化分布名称和方法
    def setup(self, dist_name, method):
        # 如果不是慢速模式且分布名称或方法在慢速列表中，抛出未实现的错误
        if not is_xslow() and (dist_name in self.slow_dists
                               or method in self.slow_methods):
            raise NotImplementedError("Skipped")

        # 根据分布名称获取相应的分布对象
        self.dist = getattr(stats, dist_name)

        # 获取特定分布的参数形状信息
        dist_shapes = self.dist_data[dist_name]

        # 如果是离散分布，则只使用 loc 参数
        if isinstance(self.dist, stats.rv_discrete):
            self.isCont = False
            kwds = {'loc': 4}
        else:
            # 如果是连续分布，则使用 loc 和 scale 参数
            self.isCont = True
            kwds = {'loc': 4, 'scale': 10}

        # 获取分布的 99% 置信区间
        bounds = self.dist.interval(.99, *dist_shapes, **kwds)
        # 在置信区间内生成等间距的样本点
        x = np.linspace(*bounds, 100)
        # 构造参数列表，包括自定义输入和生成的样本点
        args = [x, *self.custom_input.get(dist_name, dist_shapes)]
        self.args = args
        self.kwds = kwds

        # 根据方法类型进行相应的处理
        if method == 'fit':
            # 离散分布没有拟合方法，抛出未实现的错误
            if isinstance(self.dist, stats.rv_discrete):
                raise NotImplementedError("This attribute is not a member "
                                          "of the distribution")
            # 对于特定分布，拟合不可靠，抛出未实现的错误
            if self.dist.name in {'irwinhall'}:
                raise NotImplementedError("Fit is unreliable.")
            # 仅需要传入数据参数进行拟合
            self.args = [self.dist.rvs(*dist_shapes, size=100, random_state=0, **kwds)]
        elif method == 'rvs':
            # 添加 size 关键字参数用于生成数据
            kwds['size'] = 1000
            kwds['random_state'] = 0
            # 忽略线性间隔的数据，保留参数形状作为位置参数
            self.args = args[1:]
        elif method == 'pdf/pmf':
            # 根据分布类型确定使用 pmf 还是 pdf 方法
            method = ('pmf' if isinstance(self.dist, stats.rv_discrete)
                      else 'pdf')
        elif method == 'logpdf/logpmf':
            # 根据分布类型确定使用 logpmf 还是 logpdf 方法
            method = ('logpmf' if isinstance(self.dist, stats.rv_discrete)
                      else 'logpdf')
        elif method in ['ppf', 'isf']:
            # 添加 np.linspace((0, 1), 100) 作为第一个参数
            self.args = [np.linspace((0, 1), 100), *args[1:]]
        elif method == 'moment':
            # 计算第五阶矩，因为前四阶可能已经优化
            self.args = [5, *args[1:]]
        elif method.startswith('stats_'):
            # 设置 moments 关键字参数为方法名后的字符串
            kwds['moments'] = method[6:]
            method = 'stats'
            # 忽略第一个参数（等间距数据），保留后续参数
            self.args = args[1:]
        elif method == 'entropy':
            # 忽略第一个参数（等间距数据），保留后续参数
            self.args = args[1:]

        # 根据当前方法名设置对应的方法对象
        self.method = getattr(self.dist, method)

    # 对分布进行计时评估
    def time_distribution(self, dist_name, method):
        # 调用当前方法对象，传入参数和关键字参数
        self.method(*self.args, **self.kwds)
class TrackContinuousRoundtrip(Benchmark):
    # 这是一个用于跟踪连续分布的基准类
    param_names = ['dist_name']
    # 参数名称列表，指定了要测试的分布名称
    params = list(dict(distcont).keys())
    # 参数是从distcont字典中提取的所有分布名称
    dist_data = dict(distcont)
    # 分布数据字典，包含了从distcont中提取的分布数据

    def setup(self, dist_name):
        # 分布设置，类似于`DistributionsAll`基准的设置过程
        # 这里专注于ppf，所以为简化起见，处理其他函数的代码已删除
        self.dist = getattr(stats, dist_name)
        # 获取stats模块中对应分布名称的分布对象
        self.shape_args = self.dist_data[dist_name]
        # 设置分布的形状参数

    def track_distribution_ppf_roundtrip(self, dist_name):
        # 跟踪ppf -> cdf往返计算的最大相对误差
        vals = [0.001, 0.5, 0.999]

        ppf = self.dist.ppf(vals, *self.shape_args)
        # 计算给定值的分位点函数值
        round_trip = self.dist.cdf(ppf, *self.shape_args)
        # 计算分位点函数值的累积分布函数值

        err_rel = np.abs(vals - round_trip) / vals
        # 计算相对误差
        return np.max(err_rel)
        # 返回最大相对误差

    def track_distribution_ppf_roundtrip_extrema(self, dist_name):
        # 跟踪"极端"ppf -> cdf往返计算的绝对误差
        v = 1e-6
        ppf = self.dist.ppf(v, *self.shape_args)
        # 计算给定值的分位点函数值
        round_trip = self.dist.cdf(ppf, *self.shape_args)
        # 计算分位点函数值的累积分布函数值

        err_abs = np.abs(v - round_trip)
        # 计算绝对误差
        return err_abs
        # 返回绝对误差

    def track_distribution_isf_roundtrip(self, dist_name):
        # 跟踪isf -> sf往返计算的最大相对误差
        vals = [0.001, 0.5, 0.999]

        isf = self.dist.isf(vals, *self.shape_args)
        # 计算给定值的逆分位点函数值
        round_trip = self.dist.sf(isf, *self.shape_args)
        # 计算逆分位点函数值的生存函数值

        err_rel = np.abs(vals - round_trip) / vals
        # 计算相对误差
        return np.max(err_rel)
        # 返回最大相对误差

    def track_distribution_isf_roundtrip_extrema(self, dist_name):
        # 跟踪"极端"isf -> sf往返计算的绝对误差
        v = 1e-6
        ppf = self.dist.isf(v, *self.shape_args)
        # 计算给定值的逆分位点函数值
        round_trip = self.dist.sf(ppf, *self.shape_args)
        # 计算逆分位点函数值的生存函数值

        err_abs = np.abs(v - round_trip)
        # 计算绝对误差
        return err_abs
        # 返回绝对误差


class PDFPeakMemory(Benchmark):
    # 跟踪给定大数组处理时的峰值内存使用情况
    # 见gh-14095

    # 运行时间最多30分钟 - 一些分布非常慢。
    timeout = 1800.0

    x = np.arange(1e6)
    # 创建一个包含100万元素的数组

    param_names = ['dist_name']
    # 参数名称列表，指定了要测试的分布名称
    params = list(dict(distcont).keys())
    # 参数是从distcont字典中提取的所有分布名称
    dist_data = dict(distcont)
    # 分布数据字典，包含了从distcont中提取的分布数据

    # 太慢了，30分钟不足以完成。
    slow_dists = ["levy_stable"]
    # 慢分布列表，包含速度较慢的分布名称
    def setup(self, dist_name):
        # 如果环境不是 xslow，则跳过这个基准测试
        if not is_xslow():
            raise NotImplementedError("skipped - environment is not xslow. "
                                      "To enable this benchmark, set the "
                                      "environment variable SCIPY_XSLOW=1")

        # 如果分布名称在慢速分布列表中，则跳过这个基准测试
        if dist_name in self.slow_dists:
            raise NotImplementedError("skipped - dist is too slow.")

        # 获取名为 dist_name 的概率分布对象
        self.dist = getattr(stats, dist_name)
        # 设置分布参数，从预定义的数据中获取
        self.shape_args = self.dist_data[dist_name]

    def peakmem_bigarr_pdf(self, dist_name):
        # 调用分布对象的概率密度函数 pdf，并传入参数 self.x 和 self.shape_args
        self.dist.pdf(self.x, *self.shape_args)
# 定义一个继承自Benchmark的分布类，用于进行统计分布的基准测试
class Distribution(Benchmark):
    # 尽管有一个新版本的基准测试可以运行所有分布，但在撰写本文时，该基准测试在asv上表现出奇怪的行为，因此保留了这个版本。
    # 参考链接：https://pv.github.io/scipy-bench/#stats.Distribution.time_distribution

    # 参数名称列表，包括分布名称和属性
    param_names = ['distribution', 'properties']
    # 参数列表，包括三种分布和四种属性
    params = [
        ['cauchy', 'gamma', 'beta'],
        ['pdf', 'cdf', 'rvs', 'fit']
    ]

    # 设置方法，用于初始化每个基准测试实例
    def setup(self, distribution, properties):
        # 使用种子12345678创建一个随机数生成器实例
        rng = np.random.default_rng(12345678)
        # 生成一个包含100个随机数的数组，并赋值给self.x
        self.x = rng.random(100)

    # 时间分布方法，用于执行各种分布及其属性的基准测试
    def time_distribution(self, distribution, properties):
        # 根据分布类型和属性执行不同的统计分布操作
        if distribution == 'gamma':
            if properties == 'pdf':
                # 计算 Gamma 分布的概率密度函数
                stats.gamma.pdf(self.x, a=5, loc=4, scale=10)
            elif properties == 'cdf':
                # 计算 Gamma 分布的累积分布函数
                stats.gamma.cdf(self.x, a=5, loc=4, scale=10)
            elif properties == 'rvs':
                # 从 Gamma 分布中生成随机样本
                stats.gamma.rvs(size=1000, a=5, loc=4, scale=10)
            elif properties == 'fit':
                # 拟合 Gamma 分布的参数
                stats.gamma.fit(self.x, loc=4, scale=10)
        elif distribution == 'cauchy':
            if properties == 'pdf':
                # 计算 Cauchy 分布的概率密度函数
                stats.cauchy.pdf(self.x, loc=4, scale=10)
            elif properties == 'cdf':
                # 计算 Cauchy 分布的累积分布函数
                stats.cauchy.cdf(self.x, loc=4, scale=10)
            elif properties == 'rvs':
                # 从 Cauchy 分布中生成随机样本
                stats.cauchy.rvs(size=1000, loc=4, scale=10)
            elif properties == 'fit':
                # 拟合 Cauchy 分布的参数
                stats.cauchy.fit(self.x, loc=4, scale=10)
        elif distribution == 'beta':
            if properties == 'pdf':
                # 计算 Beta 分布的概率密度函数
                stats.beta.pdf(self.x, a=5, b=3, loc=4, scale=10)
            elif properties == 'cdf':
                # 计算 Beta 分布的累积分布函数
                stats.beta.cdf(self.x, a=5, b=3, loc=4, scale=10)
            elif properties == 'rvs':
                # 从 Beta 分布中生成随机样本
                stats.beta.rvs(size=1000, a=5, b=3, loc=4, scale=10)
            elif properties == 'fit':
                # 拟合 Beta 分布的参数
                stats.beta.fit(self.x, loc=4, scale=10)

    # 保留旧的基准测试结果（如果更改了基准测试，请删除此行）
    time_distribution.version = (
        "fb22ae5386501008d945783921fe44aef3f82c1dafc40cddfaccaeec38b792b0"
    )


# 描述统计类，继承自Benchmark，用于执行描述统计相关的基准测试
class DescriptiveStats(Benchmark):
    # 参数名称列表，包括n_levels
    param_names = ['n_levels']
    # 参数列表，只包括一个参数10或1000
    params = [
        [10, 1000]
    ]

    # 设置方法，用于初始化每个基准测试实例
    def setup(self, n_levels):
        # 使用种子12345678创建一个随机数生成器实例
        rng = np.random.default_rng(12345678)
        # 生成一个大小为(1000, 10)的整数随机数组，并赋值给self.levels
        self.levels = rng.integers(n_levels, size=(1000, 10))

    # 时间模式方法，用于执行模式计算的基准测试
    def time_mode(self, n_levels):
        # 计算self.levels数组在axis=0上的众数
        stats.mode(self.levels, axis=0)


# 高斯核密度估计类，继承自Benchmark，用于执行高斯核密度估计的基准测试
class GaussianKDE(Benchmark):
    # 参数名称列表，只包括points
    param_names = ['points']
    # 参数列表，只包括一个参数10或6400
    params = [10, 6400]
    # 在类中定义一个方法 `setup`，用于初始化对象的一些属性和数据
    def setup(self, points):
        # 将传入的 points 参数赋值给对象的 length 属性
        self.length = points
        # 使用默认种子 12345678 初始化一个随机数生成器 rng
        rng = np.random.default_rng(12345678)
        # 设定生成数据的数量为 n = 2000
        n = 2000
        # 从正态分布中生成 n 个随机数，存放在 m1 中
        m1 = rng.normal(size=n)
        # 从带有指定标准差的正态分布中生成 n 个随机数，存放在 m2 中
        m2 = rng.normal(scale=0.5, size=n)

        # 计算 m1 和 m2 的最小值和最大值
        xmin = m1.min()
        xmax = m1.max()
        ymin = m2.min()
        ymax = m2.max()

        # 使用 np.mgrid 创建一个二维坐标网格 X, Y，网格形状为 80x80
        X, Y = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
        # 将 X 和 Y 扁平化成一维数组，并垂直堆叠起来，形成一个 2x(80*80) 的数组，存放在 self.positions 中
        self.positions = np.vstack([X.ravel(), Y.ravel()])
        # 将 m1 和 m2 垂直堆叠起来，形成一个 2xn 的数组，并存放在 values 中
        values = np.vstack([m1, m2])
        # 使用 values 数据初始化一个高斯核密度估计对象，并赋值给 self.kernel
        self.kernel = stats.gaussian_kde(values)

    # 定义一个方法 `time_gaussian_kde_evaluate`，用于评估高斯核密度估计在给定长度上的结果
    def time_gaussian_kde_evaluate(self, length):
        # 调用 self.kernel 对象的 evaluate 方法，对 self.positions[:, :self.length] 进行评估
        self.kernel(self.positions[:, :self.length])

    # 定义一个方法 `time_gaussian_kde_logpdf`，用于计算高斯核密度估计在给定长度上的对数概率密度函数值
    def time_gaussian_kde_logpdf(self, length):
        # 调用 self.kernel 对象的 logpdf 方法，计算 self.positions[:, :self.length] 的对数概率密度函数值
        self.kernel.logpdf(self.positions[:, :self.length])
# 定义一个继承自 Benchmark 的 GroupSampling 类，用于比较不同维度的群体抽样操作的性能
class GroupSampling(Benchmark):
    # 参数名列表，仅包含维度 'dim'
    param_names = ['dim']
    # 参数设定，包含不同的维度列表
    params = [[3, 10, 50, 200]]

    # 初始化方法，设置随机数生成器 rng
    def setup(self, dim):
        self.rng = np.random.default_rng(12345678)

    # 计时方法，生成一个 dim 维的单位正交群随机变量
    def time_unitary_group(self, dim):
        stats.unitary_group.rvs(dim, random_state=self.rng)

    # 计时方法，生成一个 dim 维的正交群随机变量
    def time_ortho_group(self, dim):
        stats.ortho_group.rvs(dim, random_state=self.rng)

    # 计时方法，生成一个 dim 维的特殊正交群随机变量
    def time_special_ortho_group(self, dim):
        stats.special_ortho_group.rvs(dim, random_state=self.rng)


# 定义一个继承自 Benchmark 的 BinnedStatisticDD 类，用于不同统计量在多维数据上的分箱统计性能比较
class BinnedStatisticDD(Benchmark):
    # 参数设定，包含不同的统计量
    params = ["count", "sum", "mean", "min", "max", "median", "std", np.std]

    # 初始化方法，设置随机数生成器 rng，并生成输入数据 inp、子箱的 x 和 y 边界 subbin_x_edges、subbin_y_edges，以及调用 binned_statistic_dd 方法的结果 ret
    def setup(self, statistic):
        rng = np.random.default_rng(12345678)
        self.inp = rng.random(9999).reshape(3, 3333) * 200
        self.subbin_x_edges = np.arange(0, 200, dtype=np.float32)
        self.subbin_y_edges = np.arange(0, 200, dtype=np.float64)
        self.ret = stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            bins=[self.subbin_x_edges, self.subbin_y_edges])

    # 计时方法，调用 binned_statistic_dd 方法进行分箱统计，返回结果
    def time_binned_statistic_dd(self, statistic):
        stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            bins=[self.subbin_x_edges, self.subbin_y_edges])

    # 计时方法，重用已有的分箱统计结果进行 binned_statistic_dd 方法调用
    def time_binned_statistic_dd_reuse_bin(self, statistic):
        stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            binned_statistic_result=self.ret)


# 定义一个继承自 Benchmark 的 ContinuousFitAnalyticalMLEOverride 类，用于比较不同分布在连续拟合分析极大似然估计（MLE）覆盖操作的性能
class ContinuousFitAnalyticalMLEOverride(Benchmark):
    # 待比较的分布列表
    dists = ["pareto", "laplace", "rayleigh", "invgauss", "gumbel_r",
             "gumbel_l", "powerlaw", "lognorm"]
    # 自定义分布参数
    custom_input = {}
    # 分布的 loc、scale 和 shapes 列表
    fnames = ['floc', 'fscale', 'f0', 'f1', 'f2']
    fixed = {}

    # 参数名列表，包含分布名、案例数、loc_fixed、scale_fixed、shape1_fixed、shape2_fixed、shape3_fixed
    param_names = ["distribution", "case", "loc_fixed", "scale_fixed",
                   "shape1_fixed", "shape2_fixed", "shape3_fixed"]
    # 参数设定，包含各个分布、案例数、及各个参数的固定与非固定状态
    params = [dists, range(2), * [[True, False]] * 5]
    # 设置函数，用于初始化分布参数和固定值
    def setup(self, dist_name, case, loc_fixed, scale_fixed,
              shape1_fixed, shape2_fixed, shape3_fixed):
        # 根据传入的分布名构造对应的统计分布对象
        self.distn = eval("stats." + dist_name)

        # 默认的 loc 和 scale 分别为 0.834 和 4.342，形状参数从 `_distr_params.py` 中获取。
        # 如果 `distcont` 中有多个有效形状的情况，它们会被单独进行基准测试。
        default_shapes_n = [s[1] for s in distcont if s[0] == dist_name]
        if case >= len(default_shapes_n):
            # 如果 case 超出了默认形状数的范围，抛出未实现的错误
            raise NotImplementedError("no alternate case for this dist")
        default_shapes = default_shapes_n[case]
        # 从自定义输入中获取参数值，如果没有则使用默认值
        param_values = self.custom_input.get(dist_name, [*default_shapes,
                                                         .834, 4.342])

        # 根据参数值的数量分离相关和非相关参数
        nparam = len(param_values)
        all_parameters = [loc_fixed, scale_fixed, shape1_fixed, shape2_fixed,
                          shape3_fixed]
        relevant_parameters = all_parameters[:nparam]
        nonrelevant_parameters = all_parameters[nparam:]

        # 如果所有参数都是固定的或者非相关参数中有任何一个不为假，抛出未实现的错误
        if True in nonrelevant_parameters or False not in relevant_parameters:
            raise NotImplementedError("skip non-relevant case")

        # TODO: 修复失败的基准测试（2023年8月），暂时跳过
        if ((dist_name == "pareto" and loc_fixed and scale_fixed)
                or (dist_name == "invgauss" and loc_fixed)):
            raise NotImplementedError("skip failing benchmark")

        # 将固定的参数值添加到 self.fixed 中，键名来自 self.fnames，与 `fnames` 中相同顺序的值
        fixed_values = self.custom_input.get(dist_name, [.834, 4.342,
                                                        *default_shapes])
        self.fixed = dict(zip(compress(self.fnames, relevant_parameters),
                          compress(fixed_values, relevant_parameters)))
        self.param_values = param_values

        # 生成随机数据，使用统计分布对象的 rvs 方法
        # 形状参数需要在 loc 和 scale 之前传入
        self.data = self.distn.rvs(*param_values[2:], *param_values[:2],
                                   size=1000,
                                   random_state=np.random.default_rng(4653465))

    # 拟合函数，用于拟合数据到分布
    def time_fit(self, dist_name, case, loc_fixed, scale_fixed,
                 shape1_fixed, shape2_fixed, shape3_fixed):
        self.distn.fit(self.data, **self.fixed)
class BenchMoment(Benchmark):
    params = [
        [1, 2, 3, 8],               # 参数化测试的 order 参数，表示统计时的阶数
        [100, 1000, 10000],         # 参数化测试的 size 参数，表示生成随机数据的大小
    ]
    param_names = ["order", "size"] # 参数名称列表，对应 params 中的两个参数名

    def setup(self, order, size):
        np.random.random(1234)      # 使用 numpy 随机数生成器，但未保存结果
        self.x = np.random.random(size)  # 生成指定大小的随机数数组，并保存到实例变量中

    def time_moment(self, order, size):
        stats.moment(self.x, order) # 测试统计时序列 x 的统计矩

class BenchSkewKurtosis(Benchmark):
    params = [
        [1, 2, 3, 8],               # 参数化测试的 order 参数，表示统计时的阶数
        [100, 1000, 10000],         # 参数化测试的 size 参数，表示生成随机数据的大小
        [False, True]               # 参数化测试的 bias 参数，表示是否进行偏差修正
    ]
    param_names = ["order", "size", "bias"]  # 参数名称列表，对应 params 中的三个参数名

    def setup(self, order, size, bias):
        np.random.random(1234)      # 使用 numpy 随机数生成器，但未保存结果
        self.x = np.random.random(size)  # 生成指定大小的随机数数组，并保存到实例变量中

    def time_skew(self, order, size, bias):
        stats.skew(self.x, bias=bias)   # 测试序列 x 的偏度统计

    def time_kurtosis(self, order, size, bias):
        stats.kurtosis(self.x, bias=bias)   # 测试序列 x 的峰度统计

class BenchQMCDiscrepancy(Benchmark):
    param_names = ['method']        # 参数名称列表，对应 params 中的 method 参数
    params = [
        ["CD", "WD", "MD", "L2-star",]  # 参数化测试的 method 参数，表示 QMC 方法的名称
    ]

    def setup(self, method):
        rng = np.random.default_rng(1234)   # 使用 numpy 随机数生成器创建随机数生成器对象
        sample = rng.random((1000, 10))     # 生成指定形状的随机数数组，并保存到实例变量中
        self.sample = sample

    def time_discrepancy(self, method):
        stats.qmc.discrepancy(self.sample, method=method)  # 测试 QMC 方法的样本偏差

class BenchQMCHalton(Benchmark):
    param_names = ['d', 'scramble', 'n', 'workers']   # 参数名称列表，对应 params 中的四个参数名
    params = [
        [1, 10],                    # 参数化测试的 d 参数，表示 Halton 序列的维度
        [True, False],              # 参数化测试的 scramble 参数，表示是否打乱序列
        [10, 1_000, 100_000],       # 参数化测试的 n 参数，表示生成的随机数数量
        [1, 4]                      # 参数化测试的 workers 参数，表示并行工作线程数
    ]

    def setup(self, d, scramble, n, workers):
        self.rng = np.random.default_rng(1234)   # 使用 numpy 随机数生成器创建随机数生成器对象

    def time_halton(self, d, scramble, n, workers):
        seq = stats.qmc.Halton(d, scramble=scramble, seed=self.rng)  # 创建 Halton 序列生成器对象
        seq.random(n, workers=workers)  # 生成 Halton 序列随机数

class BenchQMCSobol(Benchmark):
    param_names = ['d', 'base2']    # 参数名称列表，对应 params 中的两个参数名
    params = [
        [1, 50, 100],               # 参数化测试的 d 参数，表示 Sobol 序列的维度
        [3, 10, 11, 12],            # 参数化测试的 base2 参数，表示生成随机数数量的底数
    ]

    def setup(self, d, base2):
        self.rng = np.random.default_rng(168525179735951991038384544)  # 使用特定种子创建随机数生成器对象
        stats.qmc.Sobol(1, bits=32)  # 初始化 Sobol 序列生成器对象，加载方向数

    def time_sobol(self, d, base2):
        seq = stats.qmc.Sobol(d, scramble=False, bits=32, seed=self.rng)  # 创建 Sobol 序列生成器对象
        seq.random_base2(base2)     # 生成 Sobol 序列随机数

class BenchPoissonDisk(Benchmark):
    param_names = ['d', 'radius', 'ncandidates', 'n']  # 参数名称列表，对应 params 中的四个参数名
    params = [
        [1, 3, 5],                  # 参数化测试的 d 参数，表示 PoissonDisk 序列的维度
        [0.2, 0.1, 0.05],           # 参数化测试的 radius 参数，表示半径大小
        [30, 60, 120],              # 参数化测试的 ncandidates 参数，表示候选点数
        [30, 100, 300]              # 参数化测试的 n 参数，表示生成的随机数数量
    ]

    def setup(self, d, radius, ncandidates, n):
        self.rng = np.random.default_rng(168525179735951991038384544)  # 使用特定种子创建随机数生成器对象

    def time_poisson_disk(self, d, radius, ncandidates, n):
        seq = stats.qmc.PoissonDisk(d, radius=radius, ncandidates=ncandidates,
                                    seed=self.rng)  # 创建 PoissonDisk 序列生成器对象
        seq.random(n)               # 生成 PoissonDisk 序列随机数

class DistanceFunctions(Benchmark):
    param_names = ['n_size']        # 参数名称列表，对应 params 中的 n_size 参数
    params = [
        [10, 4000]                  # 参数化测试的 n_size 参数，表示距离函数计算的数组大小
    ]
    # 设置对象的初始状态，生成指定大小的随机数生成器
    def setup(self, n_size):
        rng = np.random.default_rng(12345678)
        # 使用随机数生成器生成长度为 n_size 的随机数，并乘以 10
        self.u_values = rng.random(n_size) * 10
        # 使用同一个随机数生成器再次生成长度为 n_size 的随机数，并乘以 10
        self.u_weights = rng.random(n_size) * 10
        # 使用随机数生成器生成长度为 n_size//2 的随机数，并乘以 10
        self.v_values = rng.random(n_size // 2) * 10
        # 使用同一个随机数生成器再次生成长度为 n_size//2 的随机数，并乘以 10
        self.v_weights = rng.random(n_size // 2) * 10

    # 计算两个分布之间的能量距离
    def time_energy_distance(self, n_size):
        # 使用统计模块中的能量距离函数计算
        stats.energy_distance(self.u_values, self.v_values,
                              self.u_weights, self.v_weights)

    # 计算两个分布之间的Wasserstein距离
    def time_wasserstein_distance(self, n_size):
        # 使用统计模块中的Wasserstein距离函数计算
        stats.wasserstein_distance(self.u_values, self.v_values,
                                   self.u_weights, self.v_weights)
class Somersd(Benchmark):
    # 参数名为'n_size'
    param_names = ['n_size']
    # 参数为包含整数10和100的列表
    params = [
        [10, 100]
    ]

    def setup(self, n_size):
        # 使用种子12345678创建随机数生成器对象rng
        rng = np.random.default_rng(12345678)
        # 从0到n_size中选择n_size个随机整数，赋值给self.x
        self.x = rng.choice(n_size, size=n_size)
        # 从0到n_size中选择n_size个随机整数，赋值给self.y
        self.y = rng.choice(n_size, size=n_size)

    def time_somersd(self, n_size):
        # 使用self.x和self.y作为参数调用stats.somersd函数
        stats.somersd(self.x, self.y)


class KolmogorovSmirnov(Benchmark):
    # 参数名为'alternative', 'mode', 'size'
    param_names = ['alternative', 'mode', 'size']
    # 参数为包含字符串和整数的列表
    # 'two-sided'对应于exact, less对应于approx, greater对应于asymp
    # 19对应于auto, 20对应于exact, 21对应于asymp
    params = [
        ['two-sided', 'less', 'greater'],
        ['exact', 'approx', 'asymp'],
        [19, 20, 21]
    ]

    def setup(self, alternative, mode, size):
        # 使用种子12345678创建随机数生成器对象np.random
        np.random.seed(12345678)
        # 生成大小为20的正态分布随机样本，赋值给self.a
        a = stats.norm.rvs(size=20)
        self.a = a

    def time_ks(self, alternative, mode, size):
        # 使用self.a和指定的参数调用stats.kstest函数
        stats.kstest(self.a, 'norm', alternative=alternative,
                     mode=mode, N=size)


class KolmogorovSmirnovTwoSamples(Benchmark):
    # 参数名为'alternative', 'mode', 'size'
    param_names = ['alternative', 'mode', 'size']
    # 参数为包含字符串和元组的列表
    # 'two-sided'对应于exact, less对应于asymp, greater对应于无
    # (21, 20)对应于auto, (20, 20)对应于exact
    params = [
        ['two-sided', 'less', 'greater'],
        ['exact', 'asymp'],
        [(21, 20), (20, 20)]
    ]

    def setup(self, alternative, mode, size):
        # 使用种子12345678创建随机数生成器对象np.random
        np.random.seed(12345678)
        # 生成大小为size[0]的正态分布随机样本，赋值给self.a
        a = stats.norm.rvs(size=size[0])
        # 生成大小为size[1]的正态分布随机样本，赋值给self.b
        b = stats.norm.rvs(size=size[1])
        self.a = a
        self.b = b

    def time_ks2(self, alternative, mode, size):
        # 使用self.a, self.b和指定的参数调用stats.ks_2samp函数
        stats.ks_2samp(self.a, self.b, alternative=alternative, mode=mode)


class RandomTable(Benchmark):
    # 参数名为"method", "ntot", "ncell"
    param_names = ["method", "ntot", "ncell"]
    # 参数为包含字符串和整数的列表
    # "boyett"对应于patefield
    # 10, 100, 1000, 10000分别对应于不同的ntot
    # 4, 64, 256, 1024分别对应于不同的ncell
    params = [
        ["boyett", "patefield"],
        [10, 100, 1000, 10000],
        [4, 64, 256, 1024]
    ]

    def setup(self, method, ntot, ncell):
        # 使用种子12345678创建随机数生成器对象self.rng
        self.rng = np.random.default_rng(12345678)
        # 计算整数k，使得k的平方等于ncell
        k = int(ncell ** 0.5)
        assert k ** 2 == ncell
        # 创建概率相等的长度为k的数组p
        p = np.ones(k) / k
        # 生成大小为ntot的多项式分布样本，赋值给row
        row = self.rng.multinomial(ntot, p)
        # 生成大小为ntot的多项式分布样本，赋值给col
        col = self.rng.multinomial(ntot, p)
        # 使用row和col调用stats.random_table函数，生成分布表格，赋值给self.dist
        self.dist = stats.random_table(row, col)

    def time_method(self, method, ntot, ncell):
        # 使用method和self.rng作为参数调用self.dist.rvs函数
        self.dist.rvs(1000, method=method, random_state=self.rng)
```