# `D:\src\scipysrc\scipy\scipy\stats\tests\test_kdeoth.py`

```
# 导入必要的库和模块
from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
                           assert_array_almost_equal,
                           assert_array_almost_equal_nulp, assert_allclose)
import pytest
from pytest import raises as assert_raises

# 定义用于测试一维核密度估计的函数
def test_kde_1d():
    # 设定随机数种子，以便结果可重现
    np.random.seed(8765678)
    # 设定基础样本数量
    n_basesample = 500
    # 从标准正态分布中抽取样本数据
    xn = np.random.randn(n_basesample)
    # 计算样本均值和样本标准差
    xnmean = xn.mean()
    xnstd = xn.std(ddof=1)

    # 计算原始样本的核密度估计
    gkde = stats.gaussian_kde(xn)

    # 对一些点评估核密度函数的密度值
    xs = np.linspace(-7, 7, 501)
    kdepdf = gkde.evaluate(xs)
    # 计算正态分布在相同点的密度值
    normpdf = stats.norm.pdf(xs, loc=xnmean, scale=xnstd)
    # 计算点之间的间隔
    intervall = xs[1] - xs[0]

    # 断言核密度估计与正态分布的差异的平方和乘以间隔小于0.01
    assert_(np.sum((kdepdf - normpdf)**2) * intervall < 0.01)
    # 计算核密度估计在均值到正无穷的箱形区域的积分概率
    prob1 = gkde.integrate_box_1d(xnmean, np.inf)
    # 计算核密度估计在负无穷到均值的箱形区域的积分概率
    prob2 = gkde.integrate_box_1d(-np.inf, xnmean)
    # 断言两个积分概率近似等于0.5
    assert_almost_equal(prob1, 0.5, decimal=1)
    assert_almost_equal(prob2, 0.5, decimal=1)
    # 断言核密度估计在均值到正无穷的区域的积分近似等于prob1
    assert_almost_equal(gkde.integrate_box(xnmean, np.inf), prob1, decimal=13)
    # 断言核密度估计在负无穷到均值的区域的积分近似等于prob2
    assert_almost_equal(gkde.integrate_box(-np.inf, xnmean), prob2, decimal=13)

    # 断言核密度估计与自身的积分近似等于密度值平方和乘以间隔
    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum() * intervall, decimal=2)
    # 断言核密度估计与正态分布的积分近似等于密度值与正态分布密度值乘积的和乘以间隔
    assert_almost_equal(gkde.integrate_gaussian(xnmean, xnstd**2),
                        (kdepdf * normpdf).sum() * intervall, decimal=2)


# 定义用于测试带权重的一维核密度估计的函数
def test_kde_1d_weighted():
    # 设定随机数种子，以便结果可重现
    np.random.seed(8765678)
    # 设定基础样本数量
    n_basesample = 500
    # 从标准正态分布中抽取样本数据
    xn = np.random.randn(n_basesample)
    # 为每个样本数据设定权重
    wn = np.random.rand(n_basesample)
    # 计算加权平均值和加权标准差
    xnmean = np.average(xn, weights=wn)
    xnstd = np.sqrt(np.average((xn - xnmean)**2, weights=wn))

    # 计算带权重的原始样本的核密度估计
    gkde = stats.gaussian_kde(xn, weights=wn)

    # 对一些点评估核密度函数的密度值
    xs = np.linspace(-7, 7, 501)
    kdepdf = gkde.evaluate(xs)
    # 计算正态分布在相同点的密度值
    normpdf = stats.norm.pdf(xs, loc=xnmean, scale=xnstd)
    # 计算点之间的间隔
    intervall = xs[1] - xs[0]

    # 断言核密度估计与正态分布的差异的平方和乘以间隔小于0.01
    assert_(np.sum((kdepdf - normpdf)**2) * intervall < 0.01)
    # 计算核密度估计在均值到正无穷的箱形区域的积分概率
    prob1 = gkde.integrate_box_1d(xnmean, np.inf)
    # 计算核密度估计在负无穷到均值的箱形区域的积分概率
    prob2 = gkde.integrate_box_1d(-np.inf, xnmean)
    # 断言两个积分概率近似等于0.5
    assert_almost_equal(prob1, 0.5, decimal=1)
    assert_almost_equal(prob2, 0.5, decimal=1)
    # 断言核密度估计在均值到正无穷的区域的积分近似等于prob1
    assert_almost_equal(gkde.integrate_box(xnmean, np.inf), prob1, decimal=13)
    # 断言核密度估计在负无穷到均值的区域的积分近似等于prob2
    assert_almost_equal(gkde.integrate_box(-np.inf, xnmean), prob2, decimal=13)

    # 断言核密度估计与自身的积分近似等于密度值平方和乘以间隔
    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum() * intervall, decimal=2)
    # 断言核密度估计与正态分布的积分近似等于密度值与正态分布密度值乘积的和乘以间隔
    assert_almost_equal(gkde.integrate_gaussian(xnmean, xnstd**2),
                        (kdepdf * normpdf).sum() * intervall, decimal=2)


# 标记测试为慢速测试，用于测试二维核密度估计
@pytest.mark.xslow
def test_kde_2d():
    # 设定随机数种子，以便结果可重现
    np.random.seed(8765678)
    # 设定基础样本数量
    n_basesample = 500

    # 设定二维正态分布的均值和协方差矩阵
    mean = np.array([1.0, 3.0])
    covariance = np.array([[1.0, 2.0], [2.0, 6.0]])
    # 生成一个多变量正态分布的样本，需要进行转置以适应 KDE 的要求（形状为 (2, 500)）
    xn = np.random.multivariate_normal(mean, covariance, size=n_basesample).T

    # 计算原始样本的核密度估计 (KDE)
    gkde = stats.gaussian_kde(xn)

    # 对一些点评估 KDE 的密度函数
    x, y = np.mgrid[-7:7:500j, -7:7:500j]
    grid_coords = np.vstack([x.ravel(), y.ravel()])
    kdepdf = gkde.evaluate(grid_coords)
    kdepdf = kdepdf.reshape(500, 500)

    # 计算多变量正态分布的概率密度函数
    normpdf = stats.multivariate_normal.pdf(np.dstack([x, y]),
                                            mean=mean, cov=covariance)

    # 计算 y 轴间隔
    intervall = y.ravel()[1] - y.ravel()[0]

    # 断言 KDE 估计的误差与正态分布的误差的平方乘以间隔的平方小于 0.01
    assert_(np.sum((kdepdf - normpdf)**2) * (intervall**2) < 0.01)

    # 设置一个极小值和一个极大值
    small = -1e100
    large = 1e100

    # 计算 KDE 在指定框内的积分
    prob1 = gkde.integrate_box([small, mean[1]], [large, large])
    prob2 = gkde.integrate_box([small, small], [large, mean[1]])

    # 断言两个积分的结果接近于 0.5（小数点精度为1）
    assert_almost_equal(prob1, 0.5, decimal=1)
    assert_almost_equal(prob2, 0.5, decimal=1)

    # 断言 KDE 的积分等于 KDE 密度函数的平方和乘以间隔的平方（小数点精度为2）
    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum()*(intervall**2), decimal=2)

    # 断言 KDE 与多变量正态分布的积分等于 KDE 密度函数与正态分布的乘积的积分（小数点精度为2）
    assert_almost_equal(gkde.integrate_gaussian(mean, covariance),
                        (kdepdf*normpdf).sum()*(intervall**2), decimal=2)
@pytest.mark.xslow
# 定义一个慢速测试标记，用于测试函数
def test_kde_2d_weighted():
    # 设定随机数种子以便复现结果
    np.random.seed(8765678)
    # 设定基础样本数
    n_basesample = 500

    # 设定均值向量和协方差矩阵
    mean = np.array([1.0, 3.0])
    covariance = np.array([[1.0, 2.0], [2.0, 6.0]])

    # 生成多元正态分布的样本，并转置以适应 KDE 的形式
    xn = np.random.multivariate_normal(mean, covariance, size=n_basesample).T
    # 生成与样本对应的权重
    wn = np.random.rand(n_basesample)

    # 计算原始样本的核密度估计
    gkde = stats.gaussian_kde(xn, weights=wn)

    # 对一些点评估核密度函数的密度值
    x, y = np.mgrid[-7:7:500j, -7:7:500j]
    grid_coords = np.vstack([x.ravel(), y.ravel()])
    kdepdf = gkde.evaluate(grid_coords)
    kdepdf = kdepdf.reshape(500, 500)

    # 计算多元正态分布的概率密度函数作为参考
    normpdf = stats.multivariate_normal.pdf(np.dstack([x, y]),
                                            mean=mean, cov=covariance)
    # 计算网格间隔
    intervall = y.ravel()[1] - y.ravel()[0]

    # 断言核密度估计与多元正态分布的差的平方的总和小于给定阈值
    assert_(np.sum((kdepdf - normpdf)**2) * (intervall**2) < 0.01)

    # 定义两个极端的值用于积分
    small = -1e100
    large = 1e100
    # 计算两个区域的积分概率
    prob1 = gkde.integrate_box([small, mean[1]], [large, large])
    prob2 = gkde.integrate_box([small, small], [large, mean[1]])

    # 断言积分结果接近期望值
    assert_almost_equal(prob1, 0.5, decimal=1)
    assert_almost_equal(prob2, 0.5, decimal=1)
    # 断言通过核密度估计计算的积分结果与核密度函数的平方和乘以网格间隔的平方接近
    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum()*(intervall**2), decimal=2)
    # 断言通过核密度估计计算的积分结果与核密度函数与多元正态分布乘积的积分接近
    assert_almost_equal(gkde.integrate_gaussian(mean, covariance),
                        (kdepdf*normpdf).sum()*(intervall**2), decimal=2)


def test_kde_bandwidth_method():
    def scotts_factor(kde_obj):
        """Same as default, just check that it works."""
        # 使用与默认相同的方式计算 Scott's 方法的带宽因子
        return np.power(kde_obj.n, -1./(kde_obj.d+4))

    # 设定随机数种子以便复现结果
    np.random.seed(8765678)
    # 设定基础样本数
    n_basesample = 50
    xn = np.random.randn(n_basesample)

    # 使用默认的带宽计算核密度估计
    gkde = stats.gaussian_kde(xn)
    # 使用可调用对象作为带宽方法
    gkde2 = stats.gaussian_kde(xn, bw_method=scotts_factor)
    # 使用标量作为带宽方法
    gkde3 = stats.gaussian_kde(xn, bw_method=gkde.factor)

    # 在一定范围内评估核密度估计的密度值
    xs = np.linspace(-7,7,51)
    kdepdf = gkde.evaluate(xs)
    kdepdf2 = gkde2.evaluate(xs)
    # 断言不同带宽方法得到的密度估计结果近似相等
    assert_almost_equal(kdepdf, kdepdf2)
    kdepdf3 = gkde3.evaluate(xs)
    # 断言使用默认带宽方法和标量带宽方法得到的密度估计结果近似相等
    assert_almost_equal(kdepdf, kdepdf3)

    # 断言当提供一个非法的带宽方法时，抛出 ValueError 异常
    assert_raises(ValueError, stats.gaussian_kde, xn, bw_method='wrongstring')


def test_kde_bandwidth_method_weighted():
    def scotts_factor(kde_obj):
        """Same as default, just check that it works."""
        # 使用与默认相同的方式计算 Scott's 方法的带宽因子
        return np.power(kde_obj.neff, -1./(kde_obj.d+4))

    # 设定随机数种子以便复现结果
    np.random.seed(8765678)
    # 设定基础样本数
    n_basesample = 50
    xn = np.random.randn(n_basesample)

    # 使用默认的带宽计算核密度估计
    gkde = stats.gaussian_kde(xn)
    # 使用可调用对象作为带宽方法
    gkde2 = stats.gaussian_kde(xn, bw_method=scotts_factor)
    # 使用标量作为带宽方法
    gkde3 = stats.gaussian_kde(xn, bw_method=gkde.factor)

    # 在一定范围内评估核密度估计的密度值
    xs = np.linspace(-7,7,51)
    kdepdf = gkde.evaluate(xs)
    kdepdf2 = gkde2.evaluate(xs)
    # 断言不同带宽方法得到的密度估计结果近似相等
    assert_almost_equal(kdepdf, kdepdf2)
    kdepdf3 = gkde3.evaluate(xs)
    # 断言使用默认带宽方法和标量带宽方法得到的密度估计结果近似相等
    assert_almost_equal(kdepdf, kdepdf3)
    # 使用 assert_raises 函数断言在调用 stats.gaussian_kde 函数时会抛出 ValueError 异常，
    # 且异常的原因是 bw_method 参数设置为 'wrongstring'
    assert_raises(ValueError, stats.gaussian_kde, xn, bw_method='wrongstring')
# Subclasses that should stay working (extracted from various sources).
# Unfortunately the earlier design of gaussian_kde made it necessary for users
# to create these kinds of subclasses, or call _compute_covariance() directly.

# 定义一个继承自 stats.gaussian_kde 的子类 _kde_subclass1
class _kde_subclass1(stats.gaussian_kde):
    def __init__(self, dataset):
        # 初始化方法，将数据集转换为至少二维数组
        self.dataset = np.atleast_2d(dataset)
        # 计算数据集的维度和样本数
        self.d, self.n = self.dataset.shape
        # 设置协方差因子为 Scott's factor
        self.covariance_factor = self.scotts_factor
        # 调用父类的 _compute_covariance 方法计算协方差
        self._compute_covariance()

# 定义另一个继承自 stats.gaussian_kde 的子类 _kde_subclass2
class _kde_subclass2(stats.gaussian_kde):
    def __init__(self, dataset):
        # 初始化方法，设置协方差因子为 Scott's factor，并调用父类的初始化方法
        self.covariance_factor = self.scotts_factor
        super().__init__(dataset)

# 定义另一个继承自 stats.gaussian_kde 的子类 _kde_subclass4
class _kde_subclass4(stats.gaussian_kde):
    def covariance_factor(self):
        # 返回协方差因子为 Silverman's factor 的一半
        return 0.5 * self.silverman_factor()

# 测试函数 test_gaussian_kde_subclassing
def test_gaussian_kde_subclassing():
    # 定义数据集 x1 和用于评估的网格点 xs
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=50)

    # 创建标准的 gaussian_kde 对象 kde
    kde = stats.gaussian_kde(x1)
    # 计算在网格点 xs 上的密度估计 ys
    ys = kde(xs)

    # 测试子类 _kde_subclass1
    kde1 = _kde_subclass1(x1)
    y1 = kde1(xs)
    # 使用 assert_array_almost_equal_nulp 断言两者的近似相等性
    assert_array_almost_equal_nulp(ys, y1, nulp=10)

    # 测试子类 _kde_subclass2
    kde2 = _kde_subclass2(x1)
    y2 = kde2(xs)
    # 使用 assert_array_almost_equal_nulp 断言两者的近似相等性
    assert_array_almost_equal_nulp(ys, y2, nulp=10)

    # 不再支持子类 3，因为不必维护对私有方法的用户调用支持

    # 测试子类 _kde_subclass4
    kde4 = _kde_subclass4(x1)
    y4 = kde4(x1)
    # 预期的密度估计值
    y_expected = [0.06292987, 0.06346938, 0.05860291, 0.08657652, 0.07904017]
    # 使用 assert_array_almost_equal 断言 y4 与预期值 y_expected 的近似相等性
    assert_array_almost_equal(y_expected, y4, decimal=6)

    # 不是子类，但检查使用 _compute_covariance() 方法
    kde5 = kde
    # 将 covariance_factor 设置为一个 lambda 函数，返回 kde 的 factor
    kde5.covariance_factor = lambda: kde.factor
    # 调用 _compute_covariance() 方法重新计算协方差
    kde5._compute_covariance()
    # 计算在网格点 xs 上的密度估计
    y5 = kde5(xs)
    # 使用 assert_array_almost_equal_nulp 断言两者的近似相等性
    assert_array_almost_equal_nulp(ys, y5, nulp=10)


# 测试函数 test_gaussian_kde_covariance_caching
def test_gaussian_kde_covariance_caching():
    # 定义数据集 x1 和用于评估的网格点 xs
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=5)
    # 预期的密度估计值，来自 scipy 0.10 的版本
    y_expected = [0.02463386, 0.04689208, 0.05395444, 0.05337754, 0.01664475]

    # 创建标准的 gaussian_kde 对象 kde
    kde = stats.gaussian_kde(x1)
    # 设置带宽为 0.5，然后重置为默认的 'scott' 方法
    kde.set_bandwidth(bw_method=0.5)
    kde.set_bandwidth(bw_method='scott')
    # 计算在网格点 xs 上的密度估计
    y2 = kde(xs)

    # 使用 assert_array_almost_equal 断言 y2 与预期值 y_expected 的近似相等性
    assert_array_almost_equal(y_expected, y2, decimal=7)


# 测试函数 test_gaussian_kde_monkeypatch
def test_gaussian_kde_monkeypatch():
    """Ugly, but people may rely on this.  See scipy pull request 123,
    specifically the linked ML thread "Width of the Gaussian in stats.kde".
    If it is necessary to break this later on, that is to be discussed on ML.
    """
    # 定义数据集 x1 和用于评估的网格点 xs
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=50)

    # 旧的 monkeypatched 版本，用于获取 Silverman's Rule
    kde = stats.gaussian_kde(x1)
    # 将 covariance_factor 设置为 silverman_factor
    kde.covariance_factor = kde.silverman_factor
    # 调用 _compute_covariance() 方法重新计算协方差
    kde._compute_covariance()
    # 计算在网格点 xs 上的密度估计
    y1 = kde(xs)

    # 新的更合理的版本。
    # 使用高斯核密度估计(stats.gaussian_kde)计算第一个数据集 x1 的密度估计函数 kde2，
    # 使用银曼带宽估计方法 ('silverman')。
    kde2 = stats.gaussian_kde(x1, bw_method='silverman')

    # 计算在给定点集 xs 处 kde2 的密度估计值，存储在 y2 中。
    y2 = kde2(xs)

    # 使用 assert_array_almost_equal_nulp 函数检查两个密度估计结果 y1 和 y2 之间的差异，
    # 确保它们的差异在 10 个单位最小浮点数单元 (nulp) 内。
    assert_array_almost_equal_nulp(y1, y2, nulp=10)
# 定义一个回归测试函数，用于检查问题 #1181 是否已修复
def test_kde_integer_input():
    # 创建一个包含整数的 NumPy 数组，范围是 [0, 1, 2, 3, 4]
    x1 = np.arange(5)
    # 使用高斯核密度估计创建 KDE 对象
    kde = stats.gaussian_kde(x1)
    # 期望的 KDE 值列表，精确到小数点后第六位
    y_expected = [0.13480721, 0.18222869, 0.19514935, 0.18222869, 0.13480721]
    # 断言计算得到的 KDE 值与期望值相等，精度为 6 位小数
    assert_array_almost_equal(kde(x1), y_expected, decimal=6)


# 支持的数据类型列表
_ftypes = ['float32', 'float64', 'float96', 'float128', 'int32', 'int64']


# 使用参数化测试框架 pytest.mark.parametrize 注册测试函数
@pytest.mark.parametrize("bw_type", _ftypes + ["scott", "silverman"])
@pytest.mark.parametrize("dtype", _ftypes)
def test_kde_output_dtype(dtype, bw_type):
    # 检查是否可以获取到指定的数据类型
    dtype = getattr(np, dtype, None)

    # 如果带宽类型是 "scott" 或 "silverman"，直接将其赋值给 bw
    if bw_type in ["scott", "silverman"]:
        bw = bw_type
    else:
        # 否则尝试获取相应的带宽类型
        bw_type = getattr(np, bw_type, None)
        # 如果获取成功，使用带宽为 3 的实例化对象；否则设置为 None
        bw = bw_type(3) if bw_type else None

    # 如果数据类型或带宽为 None，则跳过测试
    if any(dt is None for dt in [dtype, bw]):
        pytest.skip()

    # 创建带权重的数据和数据集
    weights = np.arange(5, dtype=dtype)
    dataset = np.arange(5, dtype=dtype)
    # 使用指定的数据类型和带宽方法创建高斯核密度估计对象
    k = stats.gaussian_kde(dataset, bw_method=bw, weights=weights)
    # 创建一组数据点
    points = np.arange(5, dtype=dtype)
    # 计算 KDE 对数据点的估计值
    result = k(points)
    # 断言结果的数据类型与数据集、数据点和权重的结果类型相同，应为 np.float64
    assert result.dtype == np.result_type(dataset, points, np.float64(weights),
                                          k.factor)


# 检查 KDE 对数概率密度函数的输入数据维度是否匹配
def test_pdf_logpdf_validation():
    # 创建一个随机数生成器对象
    rng = np.random.default_rng(64202298293133848336925499069837723291)
    # 生成标准正态分布的随机样本数据，维度为 (2, 10)
    xn = rng.standard_normal((2, 10))
    # 使用样本数据创建高斯核密度估计对象
    gkde = stats.gaussian_kde(xn)
    # 生成一个维度为 (3, 10) 的标准正态分布随机样本数据
    xs = rng.standard_normal((3, 10))

    # 期望引发 ValueError 异常，提示数据点的维度与数据集的维度不匹配
    msg = "points have dimension 3, dataset has dimension 2"
    with pytest.raises(ValueError, match=msg):
        # 调用 logpdf 方法计算对数概率密度值，验证是否抛出预期的异常
        gkde.logpdf(xs)


# 检查 KDE 的概率密度函数和对数概率密度函数的计算结果
def test_pdf_logpdf():
    np.random.seed(1)
    n_basesample = 50
    # 生成大小为 n_basesample 的标准正态分布样本数据
    xn = np.random.randn(n_basesample)

    # 默认情况下创建高斯核密度估计对象
    gkde = stats.gaussian_kde(xn)

    # 生成一个从 -15 到 12 的等间隔数组，共 25 个点
    xs = np.linspace(-15, 12, 25)
    # 计算默认情况下的 KDE 和 PDF
    pdf = gkde.evaluate(xs)
    pdf2 = gkde.pdf(xs)
    # 断言两种方法计算的概率密度函数值相等，精确到小数点后第12位
    assert_almost_equal(pdf, pdf2, decimal=12)

    # 计算 KDE 的对数概率密度函数
    logpdf = np.log(pdf)
    logpdf2 = gkde.logpdf(xs)
    # 断言两种方法计算的对数概率密度函数值相等，精确到小数点后第12位
    assert_almost_equal(logpdf, logpdf2, decimal=12)

    # 创建一个数据点数目比数据集更多的 KDE 对象
    gkde = stats.gaussian_kde(xs)
    # 计算 gkde 对 xn 的对数概率密度函数值
    pdf = np.log(gkde.evaluate(xn))
    pdf2 = gkde.logpdf(xn)
    # 断言两种方法计算的对数概率密度函数值相等，精确到小数点后第12位
    assert_almost_equal(pdf, pdf2, decimal=12)


# 检查带权重的 KDE 的概率密度函数和对数概率密度函数的计算结果
def test_pdf_logpdf_weighted():
    np.random.seed(1)
    n_basesample = 50
    # 生成大小为 n_basesample 的标准正态分布样本数据
    xn = np.random.randn(n_basesample)
    wn = np.random.rand(n_basesample)

    # 使用权重创建带权重的高斯核密度估计对象
    gkde = stats.gaussian_kde(xn, weights=wn)

    # 生成一个从 -15 到 12 的等间隔数组，共 25 个点
    xs = np.linspace(-15, 12, 25)
    # 计算带权重的 KDE 和 PDF
    pdf = gkde.evaluate(xs)
    pdf2 = gkde.pdf(xs)
    # 断言两种方法计算的概率密度函数值相等，精确到小数点后第12位
    assert_almost_equal(pdf, pdf2, decimal=12)

    # 计算带权重的 KDE 的对数概率密度函数
    logpdf = np.log(pdf)
    logpdf2 = gkde.logpdf(xs)
    # 断言两种方法计算的对数概率密度函数值相等，精确到小数点后第12位
    assert_almost_equal(logpdf, logpdf2, decimal=12)

    # 创建一个数据点数目比数据集更多的带权重的 KDE 对象
    gkde = stats.gaussian_kde(xs, weights=np.random.rand(len(xs)))
    # 计算 gkde 对 xn 的对数概率密度函数值
    pdf = np.log(gkde.evaluate(xn))
    pdf2 = gkde.logpdf(xn)
    # 断言两种方法计算的对数概率密度函数值相等，精确到小数点后第12位
    assert_almost_equal(pdf, pdf2, decimal=12)


# 检查一维边缘概率密度函数的计算
def test_marginal_1_axis():
    # 创建一个随机数生成器对象
    rng = np.random.default_rng(6111799263660870475)
    n_data = 50
    n_dim = 10
    # 生成服从标准正态分布的数据集，维度为 (n_dim, n_data)
    dataset = rng.normal(size=(n_dim, n_data))
    # 生成一个 n_dim 行 3 列的随机数组，每列包含从正态分布中随机抽取的值
    points = rng.normal(size=(n_dim, 3))

    # 指定要保留的维度，创建一个包含整数的 NumPy 数组
    dimensions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # dimensions to keep

    # 使用 dataset 数据集创建一个高斯核密度估计对象
    kde = stats.gaussian_kde(dataset)

    # 从 kde 对象中提取指定维度的边际密度估计对象
    marginal = kde.marginal(dimensions)

    # 计算指定维度上的边际密度函数在给定点集 points[dimensions] 处的值
    pdf = marginal.pdf(points[dimensions])

    # 定义一个函数，计算单个点的边际概率密度函数值
    def marginal_pdf_single(point):
        # 定义一个函数 f(x)，将 x 和 point[dimensions] 连接起来，计算其在 kde 中的概率密度函数值
        def f(x):
            x = np.concatenate(([x], point[dimensions]))
            return kde.pdf(x)[0]
        # 使用数值积分计算 f(x) 在整个实数范围上的积分值
        return integrate.quad(f, -np.inf, np.inf)[0]

    # 定义一个函数，计算给定点集 points 中每个点的边际概率密度函数值
    def marginal_pdf(points):
        return np.apply_along_axis(marginal_pdf_single, axis=0, arr=points)

    # 计算 points 中每个点的边际概率密度函数值
    ref = marginal_pdf(points)

    # 断言 pdf 和 ref 的每个元素在相对误差容忍度内相等，否则引发 AssertionError
    assert_allclose(pdf, ref, rtol=1e-6)
@pytest.mark.xslow
# 标记为慢速测试，用于识别需要更长时间运行的测试用例
def test_marginal_2_axis():
    rng = np.random.default_rng(6111799263660870475)
    n_data = 30
    n_dim = 4
    dataset = rng.normal(size=(n_dim, n_data))
    points = rng.normal(size=(n_dim, 3))

    dimensions = np.array([1, 3])  # 要保留的维度

    kde = stats.gaussian_kde(dataset)
    marginal = kde.marginal(dimensions)
    pdf = marginal.pdf(points[dimensions])

    def marginal_pdf(points):
        def marginal_pdf_single(point):
            def f(y, x):
                w, z = point[dimensions]
                x = np.array([x, w, y, z])
                return kde.pdf(x)[0]
            return integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)[0]

        return np.apply_along_axis(marginal_pdf_single, axis=0, arr=points)

    ref = marginal_pdf(points)

    assert_allclose(pdf, ref, rtol=1e-6)


def test_marginal_iv():
    # 输入验证测试
    rng = np.random.default_rng(6111799263660870475)
    n_data = 30
    n_dim = 4
    dataset = rng.normal(size=(n_dim, n_data))
    points = rng.normal(size=(n_dim, 3))

    kde = stats.gaussian_kde(dataset)

    # 检查正负索引是否等价
    dimensions1 = [-1, 1]
    marginal1 = kde.marginal(dimensions1)
    pdf1 = marginal1.pdf(points[dimensions1])

    dimensions2 = [3, -3]
    marginal2 = kde.marginal(dimensions2)
    pdf2 = marginal2.pdf(points[dimensions2])

    assert_equal(pdf1, pdf2)

    # 非整数维度的输入验证
    message = "Elements of `dimensions` must be integers..."
    with pytest.raises(ValueError, match=message):
        kde.marginal([1, 2.5])

    # 唯一性验证
    message = "All elements of `dimensions` must be unique."
    with pytest.raises(ValueError, match=message):
        kde.marginal([1, 2, 2])

    # 非整数维度的输入验证
    message = (r"Dimensions \[-5  6\] are invalid for a distribution in 4...")
    with pytest.raises(ValueError, match=message):
        kde.marginal([1, -5, 6])


@pytest.mark.xslow
# 标记为慢速测试，用于识别需要更长时间运行的测试用例
def test_logpdf_overflow():
    # gh-12988的回归测试；针对高维度KDE的线性代数不稳定性进行测试
    np.random.seed(1)
    n_dimensions = 2500
    n_samples = 5000
    xn = np.array([np.random.randn(n_samples) + (n) for n in range(
        0, n_dimensions)])

    # 默认情况下
    gkde = stats.gaussian_kde(xn)

    logpdf = gkde.logpdf(np.arange(0, n_dimensions))
    np.testing.assert_equal(np.isneginf(logpdf[0]), False)
    np.testing.assert_equal(np.isnan(logpdf[0]), False)


def test_weights_intact():
    # gh-9709的回归测试：权重不被修改
    np.random.seed(12345)
    vals = np.random.lognormal(size=100)
    weights = np.random.choice([1.0, 10.0, 100], size=vals.size)
    orig_weights = weights.copy()

    stats.gaussian_kde(np.log10(vals), weights=weights)
    assert_allclose(weights, orig_weights, atol=1e-14, rtol=1e-14)


def test_weights_integer():
    # 整数权重是可以的，参见gh-9709的评论
    # 设置随机种子，以确保结果可重复
    np.random.seed(12345)
    
    # 定义一个包含数值的列表
    values = [0.2, 13.5, 21.0, 75.0, 99.0]
    
    # 定义一个包含权重的列表，这些权重是整数
    weights = [1, 2, 4, 8, 16]
    
    # 使用 Gaussian Kernel Density Estimation (KDE) 创建一个概率密度函数对象，带有整数权重
    pdf_i = stats.gaussian_kde(values, weights=weights)
    
    # 使用 Gaussian KDE 创建另一个概率密度函数对象，但这次使用了 np.float64 类型的权重
    pdf_f = stats.gaussian_kde(values, weights=np.float64(weights))
    
    # 定义一个新的数值列表，用于评估两个概率密度函数对象
    xn = [0.3, 11, 88]
    
    # 断言两个概率密度函数对象在给定的数值列表上的评估结果非常接近，允许的绝对误差为 1e-14，相对误差也为 1e-14
    assert_allclose(pdf_i.evaluate(xn),
                    pdf_f.evaluate(xn), atol=1e-14, rtol=1e-14)
# 测试 resample 方法的种子选项
def test_seed():
    # 定义一个内部函数来测试 gkde_trail 对象的 resample 方法
    def test_seed_sub(gkde_trail):
        n_sample = 200
        # 如果不使用种子，每次调用 resample 结果应该不同
        samp1 = gkde_trail.resample(n_sample)
        samp2 = gkde_trail.resample(n_sample)
        # 断言 samp1 和 samp2 不相等
        assert_raises(
            AssertionError, assert_allclose, samp1, samp2, atol=1e-13
        )
        # 使用整数种子
        seed = 831
        samp1 = gkde_trail.resample(n_sample, seed=seed)
        samp2 = gkde_trail.resample(n_sample, seed=seed)
        # 断言 samp1 和 samp2 相等
        assert_allclose(samp1, samp2, atol=1e-13)
        # 使用 RandomState 对象作为种子
        rstate1 = np.random.RandomState(seed=138)
        samp1 = gkde_trail.resample(n_sample, seed=rstate1)
        rstate2 = np.random.RandomState(seed=138)
        samp2 = gkde_trail.resample(n_sample, seed=rstate2)
        # 断言 samp1 和 samp2 相等
        assert_allclose(samp1, samp2, atol=1e-13)

        # 检查是否可以使用 np.random.Generator（适用于 numpy >= 1.17）
        if hasattr(np.random, 'default_rng'):
            # 获取一个 np.random.Generator 对象
            rng = np.random.default_rng(1234)
            gkde_trail.resample(n_sample, seed=rng)

    np.random.seed(8765678)
    n_basesample = 500
    # 生成一个随机数数组
    wn = np.random.rand(n_basesample)
    # 测试 1 维情况
    xn_1d = np.random.randn(n_basesample)

    # 创建一个 1 维的高斯核密度估计对象
    gkde_1d = stats.gaussian_kde(xn_1d)
    # 调用内部测试函数
    test_seed_sub(gkde_1d)
    # 创建一个带权重的 1 维高斯核密度估计对象
    gkde_1d_weighted = stats.gaussian_kde(xn_1d, weights=wn)
    # 再次调用内部测试函数
    test_seed_sub(gkde_1d_weighted)

    # 测试 2 维情况
    mean = np.array([1.0, 3.0])
    covariance = np.array([[1.0, 2.0], [2.0, 6.0]])
    # 生成一个 2 维多元正态分布随机数数组
    xn_2d = np.random.multivariate_normal(mean, covariance, size=n_basesample).T

    # 创建一个 2 维的高斯核密度估计对象
    gkde_2d = stats.gaussian_kde(xn_2d)
    # 调用内部测试函数
    test_seed_sub(gkde_2d)
    # 创建一个带权重的 2 维高斯核密度估计对象
    gkde_2d_weighted = stats.gaussian_kde(xn_2d, weights=wn)
    # 再次调用内部测试函数
    test_seed_sub(gkde_2d_weighted)


# 测试数据维度低于维度数量时的协方差异常情况
def test_singular_data_covariance_gh10205():
    # 当数据位于低维子空间时，可能引发异常，检查错误消息是否清晰
    rng = np.random.default_rng(2321583144339784787)
    mu = np.array([1, 10, 20])
    sigma = np.array([[4, 10, 0], [10, 25, 0], [0, 0, 100]])
    # 从多元正态分布中生成数据
    data = rng.multivariate_normal(mu, sigma, 1000)
    try:  # 在某些平台上不会引发错误，这是可以接受的
        # 创建一个高斯核密度估计对象，输入数据的转置
        stats.gaussian_kde(data.T)
    except linalg.LinAlgError:
        # 期望捕获特定的线性代数错误，并验证错误消息
        msg = "The data appears to lie in a lower-dimensional subspace..."
        with assert_raises(linalg.LinAlgError, match=msg):
            # 断言引发的错误匹配预期消息
            stats.gaussian_kde(data.T)


# 测试数据点少于维度数的情况
def test_fewer_points_than_dimensions_gh17436():
    # 当数据点少于维度数时，协方差矩阵将是奇异的，并且会引发在 test_singular_data_covariance_gh10205 中测试的异常
    rng = np.random.default_rng(2046127537594925772)
    # 使用随机数生成器 rng 生成一个多元正态分布的样本集合，每个样本有3个维度
    rvs = rng.multivariate_normal(np.zeros(3), np.eye(3), size=5)
    # 设置错误消息，用于断言测试
    message = "Number of dimensions is greater than number of samples..."
    # 使用 pytest 的 raises 方法来检查是否抛出 ValueError 异常，并匹配指定的错误消息
    with pytest.raises(ValueError, match=message):
        # 调用 stats 模块中的 gaussian_kde 函数，并传入生成的多元正态分布样本 rvs
        stats.gaussian_kde(rvs)
```