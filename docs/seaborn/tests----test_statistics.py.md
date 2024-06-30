# `D:\src\scipysrc\seaborn\tests\test_statistics.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理

try:
    import statsmodels.distributions as smdist  # 尝试导入 statsmodels 库中的 distributions 模块
except ImportError:
    smdist = None  # 如果导入失败，则设置 smdist 为 None

import pytest  # 导入 Pytest 测试框架
from numpy.testing import assert_array_equal, assert_array_almost_equal  # 导入 NumPy 测试工具函数

from seaborn._statistics import (  # 从 seaborn 库的 _statistics 模块中导入以下类和函数
    KDE,  # 密度估计
    Histogram,  # 直方图
    ECDF,  # 经验累积分布函数
    EstimateAggregator,  # 估计聚合器
    LetterValues,  # 字母值
    WeightedAggregator,  # 加权聚合器
    _validate_errorbar_arg,  # 验证误差条参数的函数
    _no_scipy,  # 标志变量，用于检测是否导入了 scipy 库
)


class DistributionFixtures:

    @pytest.fixture
    def x(self, rng):
        return rng.normal(0, 1, 100)  # 返回一个均值为 0，标准差为 1 的正态分布随机数组成的数组，长度为 100

    @pytest.fixture
    def x2(self, rng):
        return rng.normal(0, 1, 742)  # 返回一个均值为 0，标准差为 1 的正态分布随机数组成的数组，长度为 742，用于边缘情况测试

    @pytest.fixture
    def y(self, rng):
        return rng.normal(0, 5, 100)  # 返回一个均值为 0，标准差为 5 的正态分布随机数组成的数组，长度为 100

    @pytest.fixture
    def weights(self, rng):
        return rng.uniform(0, 5, 100)  # 返回一个在 [0, 5) 区间均匀分布的随机数组成的数组，长度为 100


class TestKDE:

    def integrate(self, y, x):
        y = np.asarray(y)  # 将 y 转换为 NumPy 数组
        x = np.asarray(x)  # 将 x 转换为 NumPy 数组
        dx = np.diff(x)  # 计算 x 的差分
        return (dx * y[:-1] + dx * y[1:]).sum() / 2  # 计算积分近似值

    def test_gridsize(self, rng):

        x = rng.normal(0, 3, 1000)  # 生成一个均值为 0，标准差为 3 的正态分布随机数组成的数组，长度为 1000

        n = 200  # 设定 KDE 的网格大小参数
        kde = KDE(gridsize=n)  # 创建一个 KDE 对象，指定网格大小为 n
        density, support = kde(x)  # 计算 KDE 密度估计和支持范围
        assert density.size == n  # 断言密度数组的长度为 n
        assert support.size == n  # 断言支持数组的长度为 n

    def test_cut(self, rng):

        x = rng.normal(0, 3, 1000)  # 生成一个均值为 0，标准差为 3 的正态分布随机数组成的数组，长度为 1000

        kde = KDE(cut=0)  # 创建一个 KDE 对象，设定 cut 参数为 0
        _, support = kde(x)  # 计算 KDE 密度估计和支持范围
        assert support.min() == x.min()  # 断言支持范围的最小值等于 x 的最小值
        assert support.max() == x.max()  # 断言支持范围的最大值等于 x 的最大值

        cut = 2  # 设定 cut 参数为 2
        bw_scale = .5  # 设定带宽缩放比例为 0.5
        bw = x.std() * bw_scale  # 计算带宽
        kde = KDE(cut=cut, bw_method=bw_scale, gridsize=1000)  # 创建一个 KDE 对象，设定 cut、bw_method 和 gridsize 参数
        _, support = kde(x)  # 计算 KDE 密度估计和支持范围
        assert support.min() == pytest.approx(x.min() - bw * cut, abs=1e-2)  # 使用 pytest 的近似断言检查支持范围的最小值
        assert support.max() == pytest.approx(x.max() + bw * cut, abs=1e-2)  # 使用 pytest 的近似断言检查支持范围的最大值

    def test_clip(self, rng):

        x = rng.normal(0, 3, 100)  # 生成一个均值为 0，标准差为 3 的正态分布随机数组成的数组，长度为 100
        clip = -1, 1  # 设定 clip 参数为 (-1, 1)
        kde = KDE(clip=clip)  # 创建一个 KDE 对象，设定 clip 参数
        _, support = kde(x)  # 计算 KDE 密度估计和支持范围

        assert support.min() >= clip[0]  # 断言支持范围的最小值大于等于 clip 的第一个元素
        assert support.max() <= clip[1]  # 断言支持范围的最大值小于等于 clip 的第二个元素

    def test_density_normalization(self, rng):

        x = rng.normal(0, 3, 1000)  # 生成一个均值为 0，标准差为 3 的正态分布随机数组成的数组，长度为 1000
        kde = KDE()  # 创建一个默认参数的 KDE 对象
        density, support = kde(x)  # 计算 KDE 密度估计和支持范围
        assert self.integrate(density, support) == pytest.approx(1, abs=1e-5)  # 使用积分近似值检查密度的归一化

    @pytest.mark.skipif(_no_scipy, reason="Test requires scipy")
    def test_cumulative(self, rng):

        x = rng.normal(0, 3, 1000)  # 生成一个均值为 0，标准差为 3 的正态分布随机数组成的数组，长度为 1000
        kde = KDE(cumulative=True)  # 创建一个带累积参数的 KDE 对象
        density, _ = kde(x)  # 计算累积密度估计
        assert density[0] == pytest.approx(0, abs=1e-5)  # 使用 pytest 的近似断言检查累积密度的起始值
        assert density[-1] == pytest.approx(1, abs=1e-5)  # 使用 pytest 的近似断言检查累积密度的结束值

    def test_cached_support(self, rng):

        x = rng.normal(0, 3, 100)  # 生成一个均值为 0，标准差为 3 的正态分布随机数组成的数组，长度为 100
        kde = KDE()  # 创建一个默认参数的 KDE 对象
        kde.define_support(x)  # 定义支持范围
        _, support = kde(x[(x > -1) & (x < 1)])  # 计算限定范围内的 KDE 密度估计和支持范围
        assert_array_equal(support, kde.support)  # 使用 NumPy 测试工具函数检查支持范围是否相等

    def test_bw_method(self, rng):

        x = rng.normal(0, 3, 100)  # 生成一个均值为 0，标准差为 3 的正态分布随机数组成的数组，长度
    # 测试带有带宽调整的一维核密度估计
    def test_bw_adjust(self, rng):
        # 生成一个服从正态分布的随机数组，均值为0，标准差为3，共100个样本
        x = rng.normal(0, 3, 100)
        # 创建一个带有指定带宽调整参数的核密度估计对象
        kde1 = KDE(bw_adjust=.2)
        kde2 = KDE(bw_adjust=2)

        # 对 x 应用两种不同带宽调整参数的核密度估计
        d1, _ = kde1(x)
        d2, _ = kde2(x)

        # 断言：第一种带宽调整参数生成的密度估计的导数的平均绝对值大于第二种带宽调整参数生成的密度估计的导数的平均绝对值
        assert np.abs(np.diff(d1)).mean() > np.abs(np.diff(d2)).mean()

    # 测试二维核密度估计的网格化
    def test_bivariate_grid(self, rng):
        # 生成两个服从正态分布的随机数组，均值为0，标准差为3，每个数组包含50个样本
        x, y = rng.normal(0, 3, (2, 50))
        # 创建一个二维核密度估计对象，并指定网格大小为100
        kde = KDE(gridsize=100)
        # 对 x, y 应用二维核密度估计，返回密度估计结果和对应的网格坐标
        density, (xx, yy) = kde(x, y)

        # 断言：密度估计结果的形状应为 (100, 100)
        assert density.shape == (100, 100)
        # 断言：xx 数组的大小应为 100
        assert xx.size == 100
        # 断言：yy 数组的大小应为 100
        assert yy.size == 100

    # 测试二维核密度估计的归一化
    def test_bivariate_normalization(self, rng):
        # 生成两个服从正态分布的随机数组，均值为0，标准差为3，每个数组包含50个样本
        x, y = rng.normal(0, 3, (2, 50))
        # 创建一个二维核密度估计对象，并指定网格大小为100
        kde = KDE(gridsize=100)
        # 对 x, y 应用二维核密度估计，返回密度估计结果和对应的网格坐标
        density, (xx, yy) = kde(x, y)

        # 计算 x, y 网格上每个小矩形的宽度
        dx = xx[1] - xx[0]
        dy = yy[1] - yy[0]

        # 计算密度估计结果的总和乘以每个小矩形的面积，应当接近 1
        total = density.sum() * (dx * dy)
        # 断言：total 应当接近 1，允许误差为 1e-2
        assert total == pytest.approx(1, abs=1e-2)

    # 测试二维核密度估计的累积分布
    @pytest.mark.skipif(_no_scipy, reason="Test requires scipy")
    def test_bivariate_cumulative(self, rng):
        # 生成两个服从正态分布的随机数组，均值为0，标准差为3，每个数组包含50个样本
        x, y = rng.normal(0, 3, (2, 50))
        # 创建一个二维核密度估计对象，并指定网格大小为100，并启用累积分布
        kde = KDE(gridsize=100, cumulative=True)
        # 对 x, y 应用二维核密度估计，返回累积密度估计结果
        density, _ = kde(x, y)

        # 断言：累积密度估计结果在左上角（最小值处）接近 0，允许误差为 1e-2
        assert density[0, 0] == pytest.approx(0, abs=1e-2)
        # 断言：累积密度估计结果在右下角（最大值处）接近 1，允许误差为 1e-2
        assert density[-1, -1] == pytest.approx(1, abs=1e-2)
class TestHistogram(DistributionFixtures):

    def test_string_bins(self, x):
        # 使用字符串 "sqrt" 作为分箱方式创建直方图对象
        h = Histogram(bins="sqrt")
        # 获取定义的分箱参数
        bin_kws = h.define_bin_params(x)
        # 断言分箱范围正确
        assert bin_kws["range"] == (x.min(), x.max())
        # 断言分箱数量正确为 x 数据长度的平方根取整
        assert bin_kws["bins"] == int(np.sqrt(len(x)))

    def test_int_bins(self, x):
        # 使用整数 n = 24 作为分箱数量创建直方图对象
        n = 24
        h = Histogram(bins=n)
        # 获取定义的分箱参数
        bin_kws = h.define_bin_params(x)
        # 断言分箱范围正确
        assert bin_kws["range"] == (x.min(), x.max())
        # 断言分箱数量正确为 n
        assert bin_kws["bins"] == n

    def test_array_bins(self, x):
        # 使用列表 bins = [-3, -2, 1, 2, 3] 作为分箱边界创建直方图对象
        bins = [-3, -2, 1, 2, 3]
        h = Histogram(bins=bins)
        # 获取定义的分箱参数
        bin_kws = h.define_bin_params(x)
        # 断言分箱边界数组与输入的 bins 相等
        assert_array_equal(bin_kws["bins"], bins)

    def test_bivariate_string_bins(self, x, y):
        # 使用字符串 "sqrt" 创建直方图对象 h
        s1, s2 = "sqrt", "fd"
        h = Histogram(bins=s1)
        # 获取定义的二元分箱参数
        e1, e2 = h.define_bin_params(x, y)["bins"]
        # 断言 e1 与 np.histogram_bin_edges(x, s1) 相等
        assert_array_equal(e1, np.histogram_bin_edges(x, s1))
        # 断言 e2 与 np.histogram_bin_edges(y, s1) 相等
        assert_array_equal(e2, np.histogram_bin_edges(y, s1))

        # 使用元组 ("sqrt", "fd") 创建直方图对象 h
        h = Histogram(bins=(s1, s2))
        # 获取定义的二元分箱参数
        e1, e2 = h.define_bin_params(x, y)["bins"]
        # 断言 e1 与 np.histogram_bin_edges(x, s1) 相等
        assert_array_equal(e1, np.histogram_bin_edges(x, s1))
        # 断言 e2 与 np.histogram_bin_edges(y, s2) 相等
        assert_array_equal(e2, np.histogram_bin_edges(y, s2))

    def test_bivariate_int_bins(self, x, y):
        # 使用整数 b1 = 5 创建直方图对象 h
        b1, b2 = 5, 10
        h = Histogram(bins=b1)
        # 获取定义的二元分箱参数
        e1, e2 = h.define_bin_params(x, y)["bins"]
        # 断言 e1 长度为 b1 + 1
        assert len(e1) == b1 + 1
        # 断言 e2 长度为 b1 + 1
        assert len(e2) == b1 + 1

        # 使用元组 (b1, b2) 创建直方图对象 h
        h = Histogram(bins=(b1, b2))
        # 获取定义的二元分箱参数
        e1, e2 = h.define_bin_params(x, y)["bins"]
        # 断言 e1 长度为 b1 + 1
        assert len(e1) == b1 + 1
        # 断言 e2 长度为 b2 + 1
        assert len(e2) == b2 + 1

    def test_bivariate
    # 测试双变量直方图的范围定义
    def test_bivariate_binrange(self, x, y):
        # 定义两个不同的范围
        r1, r2 = (-4, 4), (-10, 10)

        # 创建直方图对象，使用第一个范围r1
        h = Histogram(binrange=r1)
        # 获取直方图定义的参数
        e1, e2 = h.define_bin_params(x, y)["bins"]
        # 断言直方图的边界符合预期
        assert e1.min() == r1[0]
        assert e1.max() == r1[1]
        assert e2.min() == r1[0]
        assert e2.max() == r1[1]

        # 创建直方图对象，使用第二个范围(r1, r2)
        h = Histogram(binrange=(r1, r2))
        # 获取直方图定义的参数
        e1, e2 = h.define_bin_params(x, y)["bins"]
        # 断言直方图的边界符合预期
        assert e1.min() == r1[0]
        assert e1.max() == r1[1]
        assert e2.min() == r2[0]
        assert e2.max() == r2[1]

    # 测试离散数据的直方图定义
    def test_discrete_bins(self, rng):
        # 生成二项分布数据
        x = rng.binomial(20, .5, 100)
        # 创建离散数据直方图对象
        h = Histogram(discrete=True)
        # 获取直方图定义的参数
        bin_kws = h.define_bin_params(x)
        # 断言直方图的范围和箱子数量符合预期
        assert bin_kws["range"] == (x.min() - .5, x.max() + .5)
        assert bin_kws["bins"] == (x.max() - x.min() + 1)

    # 测试单个奇数观测值的直方图创建
    def test_odd_single_observation(self):
        # GH2721测试用例
        x = np.array([0.49928])
        # 创建直方图对象，指定binwidth为0.03
        h, e = Histogram(binwidth=0.03)(x)
        # 断言直方图高度为1
        assert len(h) == 1
        # 断言直方图箱子宽度符合预期（与pytest.approx函数一致）
        assert (e[1] - e[0]) == pytest.approx(.03)

    # 测试binwidth舍入误差的直方图创建
    def test_binwidth_roundoff(self):
        # GH2785测试用例
        x = np.array([2.4, 2.5, 2.6])
        # 创建直方图对象，指定binwidth为0.01
        h, e = Histogram(binwidth=0.01)(x)
        # 断言直方图总和为输入数据的数量
        assert h.sum() == 3

    # 测试直方图与matplotlib直方图库生成的直方图对比
    def test_histogram(self, x):
        # 创建默认直方图对象
        h = Histogram()
        # 调用直方图对象，获取高度和边界
        heights, edges = h(x)
        # 使用matplotlib生成直方图，获取高度和边界
        heights_mpl, edges_mpl = np.histogram(x, bins="auto")
        # 断言两个直方图的高度和边界完全相等
        assert_array_equal(heights, heights_mpl)
        assert_array_equal(edges, edges_mpl)

    # 测试统计数量的直方图创建
    def test_count_stat(self, x):
        # 创建统计数量的直方图对象
        h = Histogram(stat="count")
        # 调用直方图对象，获取高度和边界
        heights, _ = h(x)
        # 断言直方图高度之和等于数据集长度
        assert heights.sum() == len(x)

    # 测试密度统计的直方图创建
    def test_density_stat(self, x):
        # 创建密度统计的直方图对象
        h = Histogram(stat="density")
        # 调用直方图对象，获取高度和边界
        heights, edges = h(x)
        # 断言直方图密度乘以边界间隔的总和等于1
        assert (heights * np.diff(edges)).sum() == 1

    # 测试概率统计的直方图创建
    def test_probability_stat(self, x):
        # 创建概率统计的直方图对象
        h = Histogram(stat="probability")
        # 调用直方图对象，获取高度和边界
        heights, _ = h(x)
        # 断言直方图高度之和等于1
        assert heights.sum() == 1

    # 测试频率统计的直方图创建
    def test_frequency_stat(self, x):
        # 创建频率统计的直方图对象
        h = Histogram(stat="frequency")
        # 调用直方图对象，获取高度和边界
        heights, edges = h(x)
        # 断言直方图频率乘以边界间隔的总和等于数据集长度
        assert (heights * np.diff(edges)).sum() == len(x)

    # 测试累积统计数量的直方图创建
    def test_cumulative_count(self, x):
        # 创建累积统计数量的直方图对象
        h = Histogram(stat="count", cumulative=True)
        # 调用直方图对象，获取高度和边界
        heights, _ = h(x)
        # 断言直方图最后一个高度等于数据集长度
        assert heights[-1] == len(x)

    # 测试累积密度统计的直方图创建
    def test_cumulative_density(self, x):
        # 创建累积密度统计的直方图对象
        h = Histogram(stat="density", cumulative=True)
        # 调用直方图对象，获取高度和边界
        heights, _ = h(x)
        # 断言直方图最后一个高度等于1
        assert heights[-1] == 1

    # 测试累积概率统计的直方图创建
    def test_cumulative_probability(self, x):
        # 创建累积概率统计的直方图对象
        h = Histogram(stat="probability", cumulative=True)
        # 调用直方图对象，获取高度和边界
        heights, _ = h(x)
        # 断言直方图最后一个高度等于1
        assert heights[-1] == 1

    # 测试累积频率统计的直方图创建
    def test_cumulative_frequency(self, x):
        # 创建累积频率统计的直方图对象
        h = Histogram(stat="frequency", cumulative=True)
        # 调用直方图对象，获取高度和边界
        heights, _ = h(x)
        # 断言直方图最后一个高度等于数据集长度
        assert heights[-1] == len(x)
    # 测试双变量直方图生成函数，计算并比较直方图的高度和边缘
    def test_bivariate_histogram(self, x, y):
        # 创建 Histogram 类的实例
        h = Histogram()
        # 使用 Histogram 实例计算直方图的高度和边缘
        heights, edges = h(x, y)
        # 使用 NumPy 的自动计算方式获取直方图的边缘
        bins_mpl = (
            np.histogram_bin_edges(x, "auto"),
            np.histogram_bin_edges(y, "auto"),
        )
        # 使用 matplotlib 的方法计算直方图的高度和边缘
        heights_mpl, *edges_mpl = np.histogram2d(x, y, bins_mpl)
        # 断言直方图的高度数组相等
        assert_array_equal(heights, heights_mpl)
        # 断言直方图的 x 和 y 边缘数组相等
        assert_array_equal(edges[0], edges_mpl[0])
        assert_array_equal(edges[1], edges_mpl[1])
    
    # 测试双变量直方图生成函数，使用计数统计方法
    def test_bivariate_count_stat(self, x, y):
        # 创建 Histogram 类的实例，使用计数统计方法
        h = Histogram(stat="count")
        # 使用 Histogram 实例计算直方图的高度
        heights, _ = h(x, y)
        # 断言直方图的总计数等于输入数据长度
        assert heights.sum() == len(x)
    
    # 测试双变量直方图生成函数，使用密度统计方法
    def test_bivariate_density_stat(self, x, y):
        # 创建 Histogram 类的实例，使用密度统计方法
        h = Histogram(stat="density")
        # 使用 Histogram 实例计算直方图的高度和 x、y 边缘
        heights, (edges_x, edges_y) = h(x, y)
        # 计算每个 bin 区域的面积
        areas = np.outer(np.diff(edges_x), np.diff(edges_y))
        # 断言直方图密度乘以面积之和接近 1
        assert (heights * areas).sum() == pytest.approx(1)
    
    # 测试双变量直方图生成函数，使用概率统计方法
    def test_bivariate_probability_stat(self, x, y):
        # 创建 Histogram 类的实例，使用概率统计方法
        h = Histogram(stat="probability")
        # 使用 Histogram 实例计算直方图的高度
        heights, _ = h(x, y)
        # 断言直方图的高度之和接近 1
        assert heights.sum() == 1
    
    # 测试双变量直方图生成函数，使用频率统计方法
    def test_bivariate_frequency_stat(self, x, y):
        # 创建 Histogram 类的实例，使用频率统计方法
        h = Histogram(stat="frequency")
        # 使用 Histogram 实例计算直方图的高度和 x、y 边缘
        heights, (x_edges, y_edges) = h(x, y)
        # 计算整个区域的面积
        area = np.outer(np.diff(x_edges), np.diff(y_edges))
        # 断言直方图频率乘以区域面积之和等于输入数据长度
        assert (heights * area).sum() == len(x)
    
    # 测试双变量直方图生成函数，使用累积计数统计方法
    def test_bivariate_cumulative_count(self, x, y):
        # 创建 Histogram 类的实例，使用累积计数统计方法
        h = Histogram(stat="count", cumulative=True)
        # 使用 Histogram 实例计算累积计数直方图的高度
        heights, _ = h(x, y)
        # 断言累积计数直方图的最后一个元素等于输入数据长度
        assert heights[-1, -1] == len(x)
    
    # 测试双变量直方图生成函数，使用累积密度统计方法
    def test_bivariate_cumulative_density(self, x, y):
        # 创建 Histogram 类的实例，使用累积密度统计方法
        h = Histogram(stat="density", cumulative=True)
        # 使用 Histogram 实例计算累积密度直方图的高度
        heights, _ = h(x, y)
        # 断言累积密度直方图的最后一个元素接近 1
        assert heights[-1, -1] == pytest.approx(1)
    
    # 测试双变量直方图生成函数，使用累积频率统计方法
    def test_bivariate_cumulative_frequency(self, x, y):
        # 创建 Histogram 类的实例，使用累积频率统计方法
        h = Histogram(stat="frequency", cumulative=True)
        # 使用 Histogram 实例计算累积频率直方图的高度
        heights, _ = h(x, y)
        # 断言累积频率直方图的最后一个元素等于输入数据长度
        assert heights[-1, -1] == len(x)
    
    # 测试双变量直方图生成函数，使用累积概率统计方法
    def test_bivariate_cumulative_probability(self, x, y):
        # 创建 Histogram 类的实例，使用累积概率统计方法
        h = Histogram(stat="probability", cumulative=True)
        # 使用 Histogram 实例计算累积概率直方图的高度
        heights, _ = h(x, y)
        # 断言累积概率直方图的最后一个元素接近 1
        assert heights[-1, -1] == pytest.approx(1)
    
    # 测试传入非法参数时是否会抛出 ValueError 异常
    def test_bad_stat(self):
        # 使用 pytest 检测是否会抛出 ValueError 异常
        with pytest.raises(ValueError):
            # 尝试创建 Histogram 类的实例，使用非法的统计方法参数
            Histogram(stat="invalid")
# 继承自DistributionFixtures类的TestECDF类，用于测试ECDF类的功能
class TestECDF(DistributionFixtures):

    # 测试单变量比例的ECDF计算
    def test_univariate_proportion(self, x):

        # 创建一个ECDF对象
        ecdf = ECDF()
        # 计算ECDF统计值和对应的值
        stat, vals = ecdf(x)
        # 断言：验证vals中的值按升序排列是否与x相同
        assert_array_equal(vals[1:], np.sort(x))
        # 断言：验证stat中的比例统计值是否接近于等分[0, 1]区间
        assert_array_almost_equal(stat[1:], np.linspace(0, 1, len(x) + 1)[1:])
        # 断言：验证stat中的第一个值是否为0
        assert stat[0] == 0

    # 测试单变量计数的ECDF计算
    def test_univariate_count(self, x):

        # 创建一个统计类型为"count"的ECDF对象
        ecdf = ECDF(stat="count")
        # 计算ECDF统计值和对应的值
        stat, vals = ecdf(x)

        # 断言：验证vals中的值按升序排列是否与x相同
        assert_array_equal(vals[1:], np.sort(x))
        # 断言：验证stat中的计数统计值是否与数组长度一致
        assert_array_almost_equal(stat[1:], np.arange(len(x)) + 1)
        # 断言：验证stat中的第一个值是否为0
        assert stat[0] == 0

    # 测试单变量百分比的ECDF计算
    def test_univariate_percent(self, x2):

        # 创建一个统计类型为"percent"的ECDF对象
        ecdf = ECDF(stat="percent")
        # 计算ECDF统计值和对应的值
        stat, vals = ecdf(x2)

        # 断言：验证vals中的值按升序排列是否与x2相同
        assert_array_equal(vals[1:], np.sort(x2))
        # 断言：验证stat中的百分比统计值是否与数据长度相关联的百分比值一致
        assert_array_almost_equal(stat[1:], (np.arange(len(x2)) + 1) / len(x2) * 100)
        # 断言：验证stat中的第一个值是否为0
        assert stat[0] == 0

    # 测试带权重的单变量比例的ECDF计算
    def test_univariate_proportion_weights(self, x, weights):

        # 创建一个ECDF对象
        ecdf = ECDF()
        # 使用权重计算ECDF统计值和对应的值
        stat, vals = ecdf(x, weights=weights)
        # 断言：验证vals中的值按升序排列是否与x相同
        assert_array_equal(vals[1:], np.sort(x))
        # 计算预期的统计值（带权重）
        expected_stats = weights[x.argsort()].cumsum() / weights.sum()
        # 断言：验证stat中的比例统计值是否接近于预期的带权重值
        assert_array_almost_equal(stat[1:], expected_stats)
        # 断言：验证stat中的第一个值是否为0
        assert stat[0] == 0

    # 测试带权重的单变量计数的ECDF计算
    def test_univariate_count_weights(self, x, weights):

        # 创建一个统计类型为"count"的ECDF对象
        ecdf = ECDF(stat="count")
        # 使用权重计算ECDF统计值和对应的值
        stat, vals = ecdf(x, weights=weights)
        # 断言：验证vals中的值按升序排列是否与x相同
        assert_array_equal(vals[1:], np.sort(x))
        # 断言：验证stat中的计数统计值是否接近于预期的带权重计数值
        assert_array_almost_equal(stat[1:], weights[x.argsort()].cumsum())
        # 断言：验证stat中的第一个值是否为0
        assert stat[0] == 0

    # 如果smdist不可用，则跳过测试，需要statsmodels
    @pytest.mark.skipif(smdist is None, reason="Requires statsmodels")
    def test_against_statsmodels(self, x):

        # 使用statsmodels中的ECDF对象计算参考值
        sm_ecdf = smdist.empirical_distribution.ECDF(x)

        # 创建一个ECDF对象
        ecdf = ECDF()
        # 计算ECDF统计值和对应的值
        stat, vals = ecdf(x)
        # 断言：验证vals中的值是否与statsmodels计算的值相同
        assert_array_equal(vals, sm_ecdf.x)
        # 断言：验证stat中的值是否与statsmodels计算的值相近
        assert_array_almost_equal(stat, sm_ecdf.y)

        # 创建一个使用补码形式计算的ECDF对象
        ecdf = ECDF(complementary=True)
        # 计算ECDF统计值和对应的值
        stat, vals = ecdf(x)
        # 断言：验证vals中的值是否与statsmodels计算的值相同
        assert_array_equal(vals, sm_ecdf.x)
        # 断言：验证stat中的值是否与statsmodels计算的值的逆序相近
        assert_array_almost_equal(stat, sm_ecdf.y[::-1])

    # 测试不支持的统计类型时是否引发ValueError异常
    def test_invalid_stat(self, x):

        # 使用不支持的统计类型"density"创建ECDF对象，预期引发ValueError异常
        with pytest.raises(ValueError, match="`stat` must be one of"):
            ECDF(stat="density")

    # 测试双变量数据时是否引发NotImplementedError异常
    def test_bivariate_error(self, x, y):

        # 创建一个ECDF对象
        ecdf = ECDF()
        # 尝试计算双变量数据的ECDF，预期引发NotImplementedError异常
        with pytest.raises(NotImplementedError, match="Bivariate ECDF"):
            ecdf(x, y)


# 用于测试EstimateAggregator类的功能
class TestEstimateAggregator:

    # 测试使用函数作为估算器的情况
    def test_func_estimator(self, long_df):

        # 定义一个计算均值的函数
        func = np.mean
        # 创建一个EstimateAggregator对象，使用均值函数
        agg = EstimateAggregator(func)
        # 对长数据框long_df中的"x"列进行聚合估算
        out = agg(long_df, "x")
        # 断言：验证估算的结果是否等于long_df中"x"列的均值
        assert out["x"] == func(long_df["x"])

    # 测试使用名称作为估算器的情况
    def test_name_estimator(self, long_df):

        # 创建一个EstimateAggregator对象，使用均值函数（通过名称）
        agg = EstimateAggregator("mean")
        # 对长数据框long_df中的"x"列进行聚合估算
        out = agg(long_df, "x")
        # 断言：验证估算的结果是否等于long_df中"x"列的均值
        assert out["x"] == long_df["x"].mean()

    # 测试使用自定义函数作为估算器的情况
    def test_custom_func_estimator(self, long_df):

        # 定义一个计算最小值的自定义函数
        def func(x):
            return np.asarray(x).min()

        # 创建一个EstimateAggregator对象，使用自定义最小值函数
        agg = EstimateAggregator(func)
        # 对长数据框long_df中的"x"列进行聚合估算
        out = agg(long_df, "x")
        # 断言：验证估算的结果是否等于long_df中"x"列的最小值
        assert out["x"] == func(long_df["x"])
    # 测试函数，用于验证标准误差条的计算
    def test_se_errorbars(self, long_df):
        # 创建 EstimateAggregator 对象，计算均值和标准误
        agg = EstimateAggregator("mean", "se")
        # 调用对象处理长格式数据 long_df 的列 'x'，返回估计值和误差条信息
        out = agg(long_df, "x")
        # 断言：估计值应与 long_df['x'] 的均值相等
        assert out["x"] == long_df["x"].mean()
        # 断言：xmin 应为 long_df['x'] 均值减去其标准误
        assert out["xmin"] == (long_df["x"].mean() - long_df["x"].sem())
        # 断言：xmax 应为 long_df['x'] 均值加上其标准误
        assert out["xmax"] == (long_df["x"].mean() + long_df["x"].sem())

        # 创建 EstimateAggregator 对象，计算均值和标准误，乘以系数 2
        agg = EstimateAggregator("mean", ("se", 2))
        # 调用对象处理长格式数据 long_df 的列 'x'，返回估计值和误差条信息
        out = agg(long_df, "x")
        # 断言：估计值应与 long_df['x'] 的均值相等
        assert out["x"] == long_df["x"].mean()
        # 断言：xmin 应为 long_df['x'] 均值减去其标准误的两倍
        assert out["xmin"] == (long_df["x"].mean() - 2 * long_df["x"].sem())
        # 断言：xmax 应为 long_df['x'] 均值加上其标准误的两倍
        assert out["xmax"] == (long_df["x"].mean() + 2 * long_df["x"].sem())

    # 测试函数，用于验证标准偏差条的计算
    def test_sd_errorbars(self, long_df):
        # 创建 EstimateAggregator 对象，计算均值和标准偏差
        agg = EstimateAggregator("mean", "sd")
        # 调用对象处理长格式数据 long_df 的列 'x'，返回估计值和误差条信息
        out = agg(long_df, "x")
        # 断言：估计值应与 long_df['x'] 的均值相等
        assert out["x"] == long_df["x"].mean()
        # 断言：xmin 应为 long_df['x'] 均值减去其标准偏差
        assert out["xmin"] == (long_df["x"].mean() - long_df["x"].std())
        # 断言：xmax 应为 long_df['x'] 均值加上其标准偏差
        assert out["xmax"] == (long_df["x"].mean() + long_df["x"].std())

        # 创建 EstimateAggregator 对象，计算均值和标准偏差，乘以系数 2
        agg = EstimateAggregator("mean", ("sd", 2))
        # 调用对象处理长格式数据 long_df 的列 'x'，返回估计值和误差条信息
        out = agg(long_df, "x")
        # 断言：估计值应与 long_df['x'] 的均值相等
        assert out["x"] == long_df["x"].mean()
        # 断言：xmin 应为 long_df['x'] 均值减去其标准偏差的两倍
        assert out["xmin"] == (long_df["x"].mean() - 2 * long_df["x"].std())
        # 断言：xmax 应为 long_df['x'] 均值加上其标准偏差的两倍
        assert out["xmax"] == (long_df["x"].mean() + 2 * long_df["x"].std())

    # 测试函数，用于验证百分位数误差条的计算
    def test_pi_errorbars(self, long_df):
        # 创建 EstimateAggregator 对象，计算均值和百分位数
        agg = EstimateAggregator("mean", "pi")
        # 调用对象处理长格式数据 long_df 的列 'y'，返回估计值和误差条信息
        out = agg(long_df, "y")
        # 断言：ymin 应为 long_df['y'] 的2.5百分位数
        assert out["ymin"] == np.percentile(long_df["y"], 2.5)
        # 断言：ymax 应为 long_df['y'] 的97.5百分位数
        assert out["ymax"] == np.percentile(long_df["y"], 97.5)

        # 创建 EstimateAggregator 对象，计算均值和百分位数，指定50百分位数
        agg = EstimateAggregator("mean", ("pi", 50))
        # 调用对象处理长格式数据 long_df 的列 'y'，返回估计值和误差条信息
        out = agg(long_df, "y")
        # 断言：ymin 应为 long_df['y'] 的25百分位数
        assert out["ymin"] == np.percentile(long_df["y"], 25)
        # 断言：ymax 应为 long_df['y'] 的75百分位数
        assert out["ymax"] == np.percentile(long_df["y"], 75)

    # 测试函数，用于验证置信区间误差条的计算
    def test_ci_errorbars(self, long_df):
        # 创建 EstimateAggregator 对象，计算均值和置信区间，使用大样本和指定种子
        agg = EstimateAggregator("mean", "ci", n_boot=100000, seed=0)
        # 调用对象处理长格式数据 long_df 的列 'y'，返回估计值和误差条信息
        out = agg(long_df, "y")

        # 创建参考 EstimateAggregator 对象，计算均值和标准误，乘以1.96倍，使用相同数据和种子
        agg_ref = EstimateAggregator("mean", ("se", 1.96))
        # 调用参考对象处理长格式数据 long_df 的列 'y'，返回估计值和误差条信息
        out_ref = agg_ref(long_df, "y")

        # 断言：ymin 应接近参考对象计算的 ymin，容忍度为1e-2
        assert out["ymin"] == pytest.approx(out_ref["ymin"], abs=1e-2)
        # 断言：ymax 应接近参考对象计算的 ymax，容忍度为1e-2
        assert out["ymax"] == pytest.approx(out_ref["ymax"], abs=1e-2)

        # 创建 EstimateAggregator 对象，计算均值和置信区间 68%，使用大样本和指定种子
        agg = EstimateAggregator("mean", ("ci", 68), n_boot=100000, seed=0)
        # 调用对象处理长格式数据 long_df 的列 'y'，返回估计值和误差条信息
        out = agg(long_df, "y")

        # 创建参考 EstimateAggregator 对象，计算均值和标准误，乘以1倍，使用相同数据和种子
        agg_ref = EstimateAggregator("mean", ("se", 1))
        # 调用参考对象处理长格式数据 long_df 的列 'y'，返回估计值和误差条信息
        out_ref = agg_ref(long_df, "y")

        # 断言：ymin 应接近参考对象计算的 ymin，容忍度为1e-2
        assert out["ymin"] == pytest.approx(out_ref["ymin"], abs=1e-2)
        # 断言：ymax 应接近参考对象计算的 ymax，容忍度为1e-2
        assert out["ymax"] == pytest.approx(out_ref["ymax"], abs=1e-2)

        # 创建 EstimateAggregator 对象，计算均值和置信区间，使用指定种子，原始和测试结果相等
        agg = EstimateAggregator("mean", "ci", seed=0)
        # 调用参考对象处理长格式数据 long_df 的列 'y
    # 定义测试单例误差条函数，用于验证估算聚合器在处理误差条时的行为
    def test_singleton_errorbars(self):

        # 创建一个使用"mean"和"ci"参数的估算聚合器实例
        agg = EstimateAggregator("mean", "ci")
        
        # 设置一个测试值
        val = 7
        
        # 通过将测试值构建成数据框，调用估算聚合器来计算输出
        out = agg(pd.DataFrame(dict(y=[val])), "y")
        
        # 断言输出的"y"值与预期的测试值相等
        assert out["y"] == val
        
        # 断言输出的"ymin"值为空（NaN）
        assert pd.isna(out["ymin"])
        
        # 断言输出的"ymax"值为空（NaN）
        assert pd.isna(out["ymax"])

    # 定义误差条验证函数，用于验证传入的误差条参数并返回正确的方法和水平值
    def test_errorbar_validation(self):

        # 验证 ("ci", 99) 参数，应返回方法为 "ci"，水平为 99
        method, level = _validate_errorbar_arg(("ci", 99))
        assert method == "ci"
        assert level == 99

        # 验证 "sd" 参数，应返回方法为 "sd"，默认水平为 1
        method, level = _validate_errorbar_arg("sd")
        assert method == "sd"
        assert level == 1

        # 定义一个 lambda 函数作为参数，验证其作为参数时的行为
        f = lambda x: (x.min(), x.max())  # noqa: E731
        method, level = _validate_errorbar_arg(f)
        
        # 断言返回的方法应该是定义的 lambda 函数本身
        assert method is f
        
        # 断言返回的水平值应为空（None）
        assert level is None

        # 定义一组错误参数，验证它们是否能触发相应的异常
        bad_args = [
            ("sem", ValueError),
            (("std", 2), ValueError),
            (("pi", 5, 95), ValueError),
            (95, TypeError),
            (("ci", "large"), TypeError),
        ]

        # 遍历错误参数列表，使用 pytest 的断言来检查是否抛出预期的异常
        for arg, exception in bad_args:
            with pytest.raises(exception, match="`errorbar` must be"):
                _validate_errorbar_arg(arg)
class TestWeightedAggregator:

    def test_weighted_mean(self, long_df):
        # 将长格式数据框中的 "x" 列赋给 "weight" 列
        long_df["weight"] = long_df["x"]
        # 创建 WeightedAggregator 实例，指定聚合方法为均值
        est = WeightedAggregator("mean")
        # 使用 WeightedAggregator 对象对长格式数据框进行聚合，返回聚合结果
        out = est(long_df, "y")
        # 计算加权平均值作为预期结果
        expected = np.average(long_df["y"], weights=long_df["weight"])
        # 断言聚合结果中 "y" 列与预期值相等
        assert_array_equal(out["y"], expected)
        # 断言聚合结果中 "ymin" 列的所有值均为 NaN
        assert_array_equal(out["ymin"], np.nan)
        # 断言聚合结果中 "ymax" 列的所有值均为 NaN
        assert_array_equal(out["ymax"], np.nan)

    def test_weighted_ci(self, long_df):
        # 将长格式数据框中的 "x" 列赋给 "weight" 列
        long_df["weight"] = long_df["x"]
        # 创建 WeightedAggregator 实例，指定聚合方法为均值和置信区间
        est = WeightedAggregator("mean", "ci")
        # 使用 WeightedAggregator 对象对长格式数据框进行聚合，返回聚合结果
        out = est(long_df, "y")
        # 计算加权平均值作为预期结果
        expected = np.average(long_df["y"], weights=long_df["weight"])
        # 断言聚合结果中 "y" 列与预期值相等
        assert_array_equal(out["y"], expected)
        # 断言聚合结果中 "ymin" 列的所有值均小于等于 "y" 列的所有值
        assert (out["ymin"] <= out["y"]).all()
        # 断言聚合结果中 "ymax" 列的所有值均大于等于 "y" 列的所有值
        assert (out["ymax"] >= out["y"]).all()

    def test_limited_estimator(self):
        # 使用 pytest 检查创建 WeightedAggregator 实例时是否会引发 ValueError 异常，异常信息需包含 "Weighted estimator must be 'mean'"
        with pytest.raises(ValueError, match="Weighted estimator must be 'mean'"):
            WeightedAggregator("median")

    def test_limited_ci(self):
        # 使用 pytest 检查创建 WeightedAggregator 实例时是否会引发 ValueError 异常，异常信息需包含 "Error bar method must be 'ci'"
        with pytest.raises(ValueError, match="Error bar method must be 'ci'"):
            WeightedAggregator("mean", "sd")


class TestLetterValues:

    @pytest.fixture
    def x(self, rng):
        # 返回一个包含随机 t 分布数据的 Pandas Series 对象
        return pd.Series(rng.standard_t(10, 10_000))

    def test_levels(self, x):
        # 使用 LetterValues 对象处理输入数据 x，返回处理结果
        res = LetterValues(k_depth="tukey", outlier_prop=0, trust_alpha=0)(x)
        # 获取处理结果中的 "k" 值
        k = res["k"]
        # 生成期望的 "levels" 数组，包含 tukey 方法计算得到的水平值
        expected = np.concatenate([np.arange(k), np.arange(k - 1)[::-1]])
        # 断言处理结果中的 "levels" 数组与期望的数组相等
        assert_array_equal(res["levels"], expected)

    def test_values(self, x):
        # 使用 LetterValues 对象处理输入数据 x，返回处理结果
        res = LetterValues(k_depth="tukey", outlier_prop=0, trust_alpha=0)(x)
        # 断言处理结果中的 "values" 数组与输入数据 x 的百分位数数组相等
        assert_array_equal(np.percentile(x, res["percs"]), res["values"])

    def test_fliers(self, x):
        # 使用 LetterValues 对象处理输入数据 x，返回处理结果
        res = LetterValues(k_depth="tukey", outlier_prop=0, trust_alpha=0)(x)
        # 获取处理结果中的 "fliers" 数组和 "values" 数组
        fliers = res["fliers"]
        values = res["values"]
        # 断言所有的 "fliers" 值要么小于 "values" 的最小值，要么大于 "values" 的最大值
        assert ((fliers < values.min()) | (fliers > values.max())).all()

    def test_median(self, x):
        # 使用 LetterValues 对象处理输入数据 x，返回处理结果
        res = LetterValues(k_depth="tukey", outlier_prop=0, trust_alpha=0)(x)
        # 断言处理结果中的 "median" 值与输入数据 x 的中位数相等
        assert res["median"] == np.median(x)

    def test_k_depth_int(self, x):
        # 使用 LetterValues 对象处理输入数据 x，设置 k_depth 参数为整数 k
        res = LetterValues(k_depth=(k := 12), outlier_prop=0, trust_alpha=0)(x)
        # 断言处理结果中的 "k" 值等于 k
        assert res["k"] == k
        # 断言处理结果中的 "levels" 数组长度为 2 * k - 1
        assert len(res["levels"]) == (2 * k - 1)

    def test_trust_alpha(self, x):
        # 使用 LetterValues 对象处理输入数据 x，设置 trust_alpha 参数为不同的值
        res1 = LetterValues(k_depth="trustworthy", outlier_prop=0, trust_alpha=.1)(x)
        res2 = LetterValues(k_depth="trustworthy", outlier_prop=0, trust_alpha=.001)(x)
        # 断言使用较大的 trust_alpha 值得到的 "k" 值要小于使用较小的 trust_alpha 值得到的 "k" 值
        assert res1["k"] > res2["k"]

    def test_outlier_prop(self, x):
        # 使用 LetterValues 对象处理输入数据 x，设置 outlier_prop 参数为不同的比例值
        res1 = LetterValues(k_depth="proportion", outlier_prop=.001, trust_alpha=0)(x)
        res2 = LetterValues(k_depth="proportion", outlier_prop=.1, trust_alpha=0)(x)
        # 断言使用较小的 outlier_prop 值得到的 "k" 值要小于使用较大的 outlier_prop 值得到的 "k" 值
        assert res1["k"] > res2["k"]
```