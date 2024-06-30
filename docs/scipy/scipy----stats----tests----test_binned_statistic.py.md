# `D:\src\scipysrc\scipy\scipy\stats\tests\test_binned_statistic.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_allclose  # 导入 NumPy 的测试模块中的断言函数
import pytest  # 导入 pytest 测试框架
from pytest import raises as assert_raises  # 导入 pytest 的异常断言函数
from scipy.stats import (binned_statistic, binned_statistic_2d,  # 导入 SciPy 统计模块中的分箱统计函数
                         binned_statistic_dd)
from scipy._lib._util import check_random_state  # 导入 SciPy 内部的工具函数

from .common_tests import check_named_results  # 导入自定义模块中的函数

class TestBinnedStatistic:  # 定义测试类 TestBinnedStatistic

    @classmethod
    def setup_class(cls):  # 设置测试类的初始化方法
        rng = check_random_state(9865)  # 使用指定种子生成随机数生成器对象
        cls.x = rng.uniform(size=100)  # 生成长度为 100 的均匀分布随机数数组 cls.x
        cls.y = rng.uniform(size=100)  # 生成长度为 100 的均匀分布随机数数组 cls.y
        cls.v = rng.uniform(size=100)  # 生成长度为 100 的均匀分布随机数数组 cls.v
        cls.X = rng.uniform(size=(100, 3))  # 生成大小为 (100, 3) 的均匀分布随机数数组 cls.X
        cls.w = rng.uniform(size=100)  # 生成长度为 100 的均匀分布随机数数组 cls.w
        cls.u = rng.uniform(size=100) + 1e6  # 生成长度为 100 的均匀分布随机数数组并加上 1e6

    def test_1d_count(self):  # 定义测试方法，测试一维计数统计
        x = self.x  # 将类属性 cls.x 赋值给局部变量 x
        v = self.v  # 将类属性 cls.v 赋值给局部变量 v

        count1, edges1, bc = binned_statistic(x, v, 'count', bins=10)  # 使用分箱统计函数计算计数、边界和箱编号
        count2, edges2 = np.histogram(x, bins=10)  # 使用 NumPy 的直方图函数计算计数和边界

        assert_allclose(count1, count2)  # 断言两个计数结果近似相等
        assert_allclose(edges1, edges2)  # 断言两组边界结果近似相等

    def test_gh5927(self):  # 定义测试方法，测试 GH5927 功能
        # smoke test for gh5927 - binned_statistic was using `is` for string
        # comparison
        x = self.x  # 将类属性 cls.x 赋值给局部变量 x
        v = self.v  # 将类属性 cls.v 赋值给局部变量 v
        statistics = ['mean', 'median', 'count', 'sum']  # 定义统计方法列表
        for statistic in statistics:  # 遍历统计方法列表
            binned_statistic(x, v, statistic, bins=10)  # 使用分箱统计函数进行统计

    def test_big_number_std(self):  # 定义测试方法，测试大数标准差的数值稳定性
        # tests for numerical stability of std calculation
        # see issue gh-10126 for more
        x = self.x  # 将类属性 cls.x 赋值给局部变量 x
        u = self.u  # 将类属性 cls.u 赋值给局部变量 u
        stat1, edges1, bc = binned_statistic(x, u, 'std', bins=10)  # 使用分箱统计函数计算标准差、边界和箱编号
        stat2, edges2, bc = binned_statistic(x, u, np.std, bins=10)  # 使用 NumPy 的标准差函数计算标准差

        assert_allclose(stat1, stat2)  # 断言两个标准差结果近似相等

    def test_empty_bins_std(self):  # 定义测试方法，测试空箱的标准差返回值为 nan
        # tests that std returns gives nan for empty bins
        x = self.x  # 将类属性 cls.x 赋值给局部变量 x
        u = self.u  # 将类属性 cls.u 赋值给局部变量 u
        print(binned_statistic(x, u, 'count', bins=1000))  # 打印使用分箱统计函数计算计数结果
        stat1, edges1, bc = binned_statistic(x, u, 'std', bins=1000)  # 使用分箱统计函数计算标准差、边界和箱编号
        stat2, edges2, bc = binned_statistic(x, u, np.std, bins=1000)  # 使用 NumPy 的标准差函数计算标准差

        assert_allclose(stat1, stat2)  # 断言两个标准差结果近似相等

    def test_non_finite_inputs_and_int_bins(self):  # 定义测试方法，测试非有限输入和整数分箱
        # if either `values` or `sample` contain np.inf or np.nan throw
        # see issue gh-9010 for more
        x = self.x  # 将类属性 cls.x 赋值给局部变量 x
        u = self.u  # 将类属性 cls.u 赋值给局部变量 u
        orig = u[0]  # 记录原始值 u[0]
        u[0] = np.inf  # 将 u[0] 设置为 np.inf
        assert_raises(ValueError, binned_statistic, u, x, 'std', bins=10)  # 断言调用分箱统计函数时会抛出 ValueError 异常
        # need to test for non-python specific ints, e.g. np.int8, np.int64
        assert_raises(ValueError, binned_statistic, u, x, 'std',
                      bins=np.int64(10))  # 断言调用分箱统计函数时会抛出 ValueError 异常
        u[0] = np.nan  # 将 u[0] 设置为 np.nan
        assert_raises(ValueError, binned_statistic, u, x, 'count', bins=10)  # 断言调用分箱统计函数时会抛出 ValueError 异常
        u[0] = orig  # 恢复原始值给 u[0]

    def test_1d_result_attributes(self):  # 定义测试方法，测试一维结果的属性
        x = self.x  # 将类属性 cls.x 赋值给局部变量 x
        v = self.v  # 将类属性 cls.v 赋值给局部变量 v

        res = binned_statistic(x, v, 'count', bins=10)  # 使用分箱统计函数计算计数结果
        attributes = ('statistic', 'bin_edges', 'binnumber')  # 定义期望的结果属性列表
        check_named_results(res, attributes)  # 调用自定义模块中的检查结果函数，检查结果的命名属性
    # 定义测试函数，计算一维数据的总和统计量
    def test_1d_sum(self):
        # 从类属性中获取数据
        x = self.x
        v = self.v

        # 使用 binned_statistic 计算总和、边界和 bc（bin counts）
        sum1, edges1, bc = binned_statistic(x, v, 'sum', bins=10)
        # 使用 numpy 的直方图函数计算总和、边界
        sum2, edges2 = np.histogram(x, bins=10, weights=v)

        # 断言两种计算方法得到的总和和边界应该非常接近
        assert_allclose(sum1, sum2)
        assert_allclose(edges1, edges2)

    # 定义测试函数，计算一维数据的平均值统计量
    def test_1d_mean(self):
        x = self.x
        v = self.v

        # 使用 binned_statistic 计算平均值、边界和 bc（bin counts）
        stat1, edges1, bc = binned_statistic(x, v, 'mean', bins=10)
        # 使用 binned_statistic 计算平均值（使用 np.mean）、边界和 bc
        stat2, edges2, bc = binned_statistic(x, v, np.mean, bins=10)

        # 断言两种计算方法得到的平均值和边界应该非常接近
        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    # 定义测试函数，计算一维数据的标准差统计量
    def test_1d_std(self):
        x = self.x
        v = self.v

        # 使用 binned_statistic 计算标准差、边界和 bc
        stat1, edges1, bc = binned_statistic(x, v, 'std', bins=10)
        # 使用 binned_statistic 计算标准差（使用 np.std）、边界和 bc
        stat2, edges2, bc = binned_statistic(x, v, np.std, bins=10)

        # 断言两种计算方法得到的标准差和边界应该非常接近
        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    # 定义测试函数，计算一维数据的最小值统计量
    def test_1d_min(self):
        x = self.x
        v = self.v

        # 使用 binned_statistic 计算最小值、边界和 bc
        stat1, edges1, bc = binned_statistic(x, v, 'min', bins=10)
        # 使用 binned_statistic 计算最小值（使用 np.min）、边界和 bc
        stat2, edges2, bc = binned_statistic(x, v, np.min, bins=10)

        # 断言两种计算方法得到的最小值和边界应该非常接近
        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    # 定义测试函数，计算一维数据的最大值统计量
    def test_1d_max(self):
        x = self.x
        v = self.v

        # 使用 binned_statistic 计算最大值、边界和 bc
        stat1, edges1, bc = binned_statistic(x, v, 'max', bins=10)
        # 使用 binned_statistic 计算最大值（使用 np.max）、边界和 bc
        stat2, edges2, bc = binned_statistic(x, v, np.max, bins=10)

        # 断言两种计算方法得到的最大值和边界应该非常接近
        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    # 定义测试函数，计算一维数据的中位数统计量
    def test_1d_median(self):
        x = self.x
        v = self.v

        # 使用 binned_statistic 计算中位数、边界和 bc
        stat1, edges1, bc = binned_statistic(x, v, 'median', bins=10)
        # 使用 binned_statistic 计算中位数（使用 np.median）、边界和 bc
        stat2, edges2, bc = binned_statistic(x, v, np.median, bins=10)

        # 断言两种计算方法得到的中位数和边界应该非常接近
        assert_allclose(stat1, stat2)
        assert_allclose(edges1, edges2)

    # 定义测试函数，计算一维数据的计数统计量
    def test_1d_bincode(self):
        x = self.x[:20]
        v = self.v[:20]

        # 使用 binned_statistic 计算计数、边界和 bc
        count1, edges1, bc = binned_statistic(x, v, 'count', bins=3)
        # 手动创建预期的 bin counts
        bc2 = np.array([3, 2, 1, 3, 2, 3, 3, 3, 3, 1, 1, 3, 3, 1, 2, 3, 1,
                        1, 2, 1])
        # 计算实际的 bin counts
        bcount = [(bc == i).sum() for i in np.unique(bc)]

        # 断言计算得到的计数和预期的 bc 应该非常接近
        assert_allclose(bc, bc2)
        assert_allclose(bcount, count1)

    # 定义测试函数，测试 range 关键字参数
    def test_1d_range_keyword(self):
        # 设置随机种子以确保可重现性
        np.random.seed(9865)
        x = np.arange(30)
        data = np.random.random(30)

        # 使用 binned_statistic 计算不同范围下的平均值和边界
        mean, bins, _ = binned_statistic(x[:15], data[:15])
        mean_range, bins_range, _ = binned_statistic(x, data, range=[(0, 14)])
        mean_range2, bins_range2, _ = binned_statistic(x, data, range=(0, 14))

        # 断言不同范围下计算得到的平均值和边界应该非常接近
        assert_allclose(mean, mean_range)
        assert_allclose(bins, bins_range)
        assert_allclose(mean, mean_range2)
        assert_allclose(bins, bins_range2)
    # 测试1维数据的多个值的统计信息
    def test_1d_multi_values(self):
        # 从实例属性中获取数据
        x = self.x
        v = self.v
        w = self.w

        # 计算第一个变量的统计量（平均值）及其边界和箱子计数
        stat1v, edges1v, bc1v = binned_statistic(x, v, 'mean', bins=10)
        # 计算第二个变量的统计量（平均值）及其边界和箱子计数
        stat1w, edges1w, bc1w = binned_statistic(x, w, 'mean', bins=10)
        # 计算两个变量的统计量（平均值）及其边界和箱子计数
        stat2, edges2, bc2 = binned_statistic(x, [v, w], 'mean', bins=10)

        # 断言两个变量的平均值统计量应该相等
        assert_allclose(stat2[0], stat1v)
        assert_allclose(stat2[1], stat1w)
        # 断言第一个变量的边界与两个变量的边界应该相等
        assert_allclose(edges1v, edges2)
        # 断言第一个变量的箱子计数与两个变量的箱子计数应该相等
        assert_allclose(bc1v, bc2)

    # 测试2维数据的计数统计
    def test_2d_count(self):
        # 从实例属性中获取数据
        x = self.x
        y = self.y
        v = self.v

        # 计算数据的二维计数统计及其边界
        count1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'count', bins=5)
        # 使用 numpy 的直方图函数计算二维数据的计数统计及其边界
        count2, binx2, biny2 = np.histogram2d(x, y, bins=5)

        # 断言两种方法计算的二维计数统计应该非常接近
        assert_allclose(count1, count2)
        # 断言两种方法计算的 x 轴边界应该非常接近
        assert_allclose(binx1, binx2)
        # 断言两种方法计算的 y 轴边界应该非常接近
        assert_allclose(biny1, biny2)

    # 测试2维数据的统计结果属性
    def test_2d_result_attributes(self):
        # 从实例属性中获取数据
        x = self.x
        y = self.y
        v = self.v

        # 计算2维数据的统计属性，如计数，边界等
        res = binned_statistic_2d(x, y, v, 'count', bins=5)
        # 定义需要检查的结果属性
        attributes = ('statistic', 'x_edge', 'y_edge', 'binnumber')
        # 检查计算结果是否具有预期的命名属性
        check_named_results(res, attributes)

    # 测试2维数据的总和统计
    def test_2d_sum(self):
        # 从实例属性中获取数据
        x = self.x
        y = self.y
        v = self.v

        # 计算2维数据的总和统计及其边界
        sum1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'sum', bins=5)
        # 使用 numpy 的直方图函数计算2维数据的总和统计及其边界
        sum2, binx2, biny2 = np.histogram2d(x, y, bins=5, weights=v)

        # 断言两种方法计算的总和统计应该非常接近
        assert_allclose(sum1, sum2)
        # 断言两种方法计算的 x 轴边界应该非常接近
        assert_allclose(binx1, binx2)
        # 断言两种方法计算的 y 轴边界应该非常接近
        assert_allclose(biny1, biny2)

    # 测试2维数据的均值统计
    def test_2d_mean(self):
        # 从实例属性中获取数据
        x = self.x
        y = self.y
        v = self.v

        # 计算2维数据的均值统计及其边界
        stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'mean', bins=5)
        # 使用 numpy 的直方图函数计算2维数据的均值统计及其边界
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.mean, bins=5)

        # 断言两种方法计算的均值统计应该非常接近
        assert_allclose(stat1, stat2)
        # 断言两种方法计算的 x 轴边界应该非常接近
        assert_allclose(binx1, binx2)
        # 断言两种方法计算的 y 轴边界应该非常接近
        assert_allclose(biny1, biny2)

    # 测试2维数据的标准差统计
    def test_2d_std(self):
        # 从实例属性中获取数据
        x = self.x
        y = self.y
        v = self.v

        # 计算2维数据的标准差统计及其边界
        stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'std', bins=5)
        # 使用 numpy 的直方图函数计算2维数据的标准差统计及其边界
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.std, bins=5)

        # 断言两种方法计算的标准差统计应该非常接近
        assert_allclose(stat1, stat2)
        # 断言两种方法计算的 x 轴边界应该非常接近
        assert_allclose(binx1, binx2)
        # 断言两种方法计算的 y 轴边界应该非常接近
        assert_allclose(biny1, biny2)

    # 测试2维数据的最小值统计
    def test_2d_min(self):
        # 从实例属性中获取数据
        x = self.x
        y = self.y
        v = self.v

        # 计算2维数据的最小值统计及其边界
        stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'min', bins=5)
        # 使用 numpy 的直方图函数计算2维数据的最小值统计及其边界
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.min, bins=5)

        # 断言两种方法计算的最小值统计应该非常接近
        assert_allclose(stat1, stat2)
        # 断言两种方法计算的 x 轴边界应该非常接近
        assert_allclose(binx1, binx2)
        # 断言两种方法计算的 y 轴边界应该非常接近
        assert_allclose(biny1, biny2)
    # 定义测试函数，用于测试二维数据的最大值统计功能
    def test_2d_max(self):
        # 从实例变量中获取 x, y, v 数据
        x = self.x
        y = self.y
        v = self.v

        # 使用 binned_statistic_2d 函数计算二维数据的最大值统计
        stat1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'max', bins=5)
        # 使用 np.max 函数计算二维数据的最大值统计
        stat2, binx2, biny2, bc = binned_statistic_2d(x, y, v, np.max, bins=5)

        # 断言两种方法计算得到的统计结果近似相等
        assert_allclose(stat1, stat2)
        # 断言两种方法得到的 x 轴分箱结果近似相等
        assert_allclose(binx1, binx2)
        # 断言两种方法得到的 y 轴分箱结果近似相等
        assert_allclose(biny1, biny2)

    # 定义测试函数，用于测试二维数据的中位数统计功能
    def test_2d_median(self):
        # 从实例变量中获取 x, y, v 数据
        x = self.x
        y = self.y
        v = self.v

        # 使用 binned_statistic_2d 函数计算二维数据的中位数统计
        stat1, binx1, biny1, bc = binned_statistic_2d(
            x, y, v, 'median', bins=5)
        # 使用 np.median 函数计算二维数据的中位数统计
        stat2, binx2, biny2, bc = binned_statistic_2d(
            x, y, v, np.median, bins=5)

        # 断言两种方法计算得到的统计结果近似相等
        assert_allclose(stat1, stat2)
        # 断言两种方法得到的 x 轴分箱结果近似相等
        assert_allclose(binx1, binx2)
        # 断言两种方法得到的 y 轴分箱结果近似相等
        assert_allclose(biny1, biny2)

    # 定义测试函数，用于测试二维数据的计数统计功能
    def test_2d_bincode(self):
        # 从实例变量中获取部分 x, y, v 数据（前20个）
        x = self.x[:20]
        y = self.y[:20]
        v = self.v[:20]

        # 使用 binned_statistic_2d 函数计算二维数据的计数统计
        count1, binx1, biny1, bc = binned_statistic_2d(
            x, y, v, 'count', bins=3)
        
        # 手动指定的预期 bin counts
        bc2 = np.array([17, 11, 6, 16, 11, 17, 18, 17, 17, 7, 6, 18, 16,
                        6, 11, 16, 6, 6, 11, 8])

        # 计算实际的 bin counts
        bcount = [(bc == i).sum() for i in np.unique(bc)]

        # 断言实际计算得到的 bin counts 与预期的近似相等
        assert_allclose(bc, bc2)
        # 断言实际计算得到的 count1 的非零元素近似相等
        count1adj = count1[count1.nonzero()]
        assert_allclose(bcount, count1adj)

    # 定义测试函数，用于测试二维数据中多个值的平均值统计功能
    def test_2d_multi_values(self):
        # 从实例变量中获取 x, y, v, w 数据
        x = self.x
        y = self.y
        v = self.v
        w = self.w

        # 使用 binned_statistic_2d 函数计算二维数据的平均值统计
        stat1v, binx1v, biny1v, bc1v = binned_statistic_2d(
            x, y, v, 'mean', bins=8)
        stat1w, binx1w, biny1w, bc1w = binned_statistic_2d(
            x, y, w, 'mean', bins=8)
        stat2, binx2, biny2, bc2 = binned_statistic_2d(
            x, y, [v, w], 'mean', bins=8)

        # 断言两种方法计算得到的统计结果近似相等
        assert_allclose(stat2[0], stat1v)
        assert_allclose(stat2[1], stat1w)
        # 断言两种方法得到的 x 轴分箱结果近似相等
        assert_allclose(binx1v, binx2)
        # 断言两种方法得到的 y 轴分箱结果近似相等
        assert_allclose(biny1w, biny2)
        # 断言两种方法得到的 bin counts 结果近似相等
        assert_allclose(bc1v, bc2)

    # 定义测试函数，用于测试二维数据的 binnumbers 展开的功能
    def test_2d_binnumbers_unraveled(self):
        # 从实例变量中获取 x, y, v 数据
        x = self.x
        y = self.y
        v = self.v

        # 使用 binned_statistic 函数计算一维数据的平均值统计
        stat, edgesx, bcx = binned_statistic(x, v, 'mean', bins=20)
        stat, edgesy, bcy = binned_statistic(y, v, 'mean', bins=10)

        # 使用 binned_statistic_2d 函数计算二维数据的平均值统计，并展开 bin numbers
        stat2, edgesx2, edgesy2, bc2 = binned_statistic_2d(
            x, y, v, 'mean', bins=(20, 10), expand_binnumbers=True)

        # 使用 numpy.searchsorted 函数计算实际的 bin numbers
        bcx3 = np.searchsorted(edgesx, x, side='right')
        bcy3 = np.searchsorted(edgesy, y, side='right')

        # `numpy.searchsorted` 在右边界是不包含的，因此需要调整
        bcx3[x == x.max()] -= 1
        bcy3[y == y.max()] -= 1

        # 断言实际计算得到的 x 轴 bin numbers 与预期的近似相等
        assert_allclose(bcx, bc2[0])
        # 断言实际计算得到的 y 轴 bin numbers 与预期的近似相等
        assert_allclose(bcy, bc2[1])
        # 断言调整后的 x 轴 bin numbers 与预期的近似相等
        assert_allclose(bcx3, bc2[0])
        # 断言调整后的 y 轴 bin numbers 与预期的近似相等
        assert_allclose(bcy3, bc2[1])

    # 定义测试函数，用于测试多维数据的计数统计功能
    def test_dd_count(self):
        # 从实例变量中获取 X, v 数据
        X = self.X
        v = self.v

        # 使用 binned_statistic_dd 函数计算多维数据的计数统计
        count1, edges1, bc = binned_statistic_dd(X, v, 'count', bins=3)
        # 使用 np.histogramdd 函数计算多维数据的计数统计
        count2, edges2 = np.histogramdd(X, bins=3)

        # 断言两种方法计算得到的统计结果近似相等
        assert_allclose(count1, count2)
        # 断言两种方法得到的 edges 结果近似相等
        assert_allclose(edges1, edges2)
    # 测试函数，验证 binned_statistic_dd 函数返回结果的属性是否正确
    def test_dd_result_attributes(self):
        # 从测试类中获取数据 X 和 v
        X = self.X
        v = self.v

        # 调用 binned_statistic_dd 函数进行统计，并指定统计类型为 'count'，分为 3 个 bin
        res = binned_statistic_dd(X, v, 'count', bins=3)
        # 定义预期的结果属性列表
        attributes = ('statistic', 'bin_edges', 'binnumber')
        # 调用辅助函数 check_named_results 验证结果中是否包含预期的属性
        check_named_results(res, attributes)

    # 测试函数，验证 binned_statistic_dd 函数计算结果与 np.histogramdd 函数的结果是否一致
    def test_dd_sum(self):
        X = self.X
        v = self.v

        # 使用 'sum' 统计类型调用 binned_statistic_dd 函数
        sum1, edges1, bc = binned_statistic_dd(X, v, 'sum', bins=3)
        # 调用 np.histogramdd 函数计算 sum2 和 edges2
        sum2, edges2 = np.histogramdd(X, bins=3, weights=v)
        # 使用 np.sum 函数调用 binned_statistic_dd 函数计算 sum3
        sum3, edges3, bc = binned_statistic_dd(X, v, np.sum, bins=3)

        # 断言 sum1 与 sum2 的近似程度
        assert_allclose(sum1, sum2)
        # 断言 edges1 与 edges2 的近似程度
        assert_allclose(edges1, edges2)
        # 断言 sum1 与 sum3 的近似程度
        assert_allclose(sum1, sum3)
        # 断言 edges1 与 edges3 的近似程度
        assert_allclose(edges1, edges3)

    # 测试函数，验证 binned_statistic_dd 函数计算结果与 np.mean 函数的结果是否一致
    def test_dd_mean(self):
        X = self.X
        v = self.v

        # 使用 'mean' 统计类型调用 binned_statistic_dd 函数
        stat1, edges1, bc = binned_statistic_dd(X, v, 'mean', bins=3)
        # 使用 np.mean 函数调用 binned_statistic_dd 函数计算 stat2
        stat2, edges2, bc = binned_statistic_dd(X, v, np.mean, bins=3)

        # 断言 stat1 与 stat2 的近似程度
        assert_allclose(stat1, stat2)
        # 断言 edges1 与 edges2 的近似程度
        assert_allclose(edges1, edges2)

    # 测试函数，验证 binned_statistic_dd 函数计算结果与 np.std 函数的结果是否一致
    def test_dd_std(self):
        X = self.X
        v = self.v

        # 使用 'std' 统计类型调用 binned_statistic_dd 函数
        stat1, edges1, bc = binned_statistic_dd(X, v, 'std', bins=3)
        # 使用 np.std 函数调用 binned_statistic_dd 函数计算 stat2
        stat2, edges2, bc = binned_statistic_dd(X, v, np.std, bins=3)

        # 断言 stat1 与 stat2 的近似程度
        assert_allclose(stat1, stat2)
        # 断言 edges1 与 edges2 的近似程度
        assert_allclose(edges1, edges2)

    # 测试函数，验证 binned_statistic_dd 函数计算结果与 np.min 函数的结果是否一致
    def test_dd_min(self):
        X = self.X
        v = self.v

        # 使用 'min' 统计类型调用 binned_statistic_dd 函数
        stat1, edges1, bc = binned_statistic_dd(X, v, 'min', bins=3)
        # 使用 np.min 函数调用 binned_statistic_dd 函数计算 stat2
        stat2, edges2, bc = binned_statistic_dd(X, v, np.min, bins=3)

        # 断言 stat1 与 stat2 的近似程度
        assert_allclose(stat1, stat2)
        # 断言 edges1 与 edges2 的近似程度
        assert_allclose(edges1, edges2)

    # 测试函数，验证 binned_statistic_dd 函数计算结果与 np.max 函数的结果是否一致
    def test_dd_max(self):
        X = self.X
        v = self.v

        # 使用 'max' 统计类型调用 binned_statistic_dd 函数
        stat1, edges1, bc = binned_statistic_dd(X, v, 'max', bins=3)
        # 使用 np.max 函数调用 binned_statistic_dd 函数计算 stat2
        stat2, edges2, bc = binned_statistic_dd(X, v, np.max, bins=3)

        # 断言 stat1 与 stat2 的近似程度
        assert_allclose(stat1, stat2)
        # 断言 edges1 与 edges2 的近似程度
        assert_allclose(edges1, edges2)

    # 测试函数，验证 binned_statistic_dd 函数计算结果与 np.median 函数的结果是否一致
    def test_dd_median(self):
        X = self.X
        v = self.v

        # 使用 'median' 统计类型调用 binned_statistic_dd 函数
        stat1, edges1, bc = binned_statistic_dd(X, v, 'median', bins=3)
        # 使用 np.median 函数调用 binned_statistic_dd 函数计算 stat2
        stat2, edges2, bc = binned_statistic_dd(X, v, np.median, bins=3)

        # 断言 stat1 与 stat2 的近似程度
        assert_allclose(stat1, stat2)
        # 断言 edges1 与 edges2 的近似程度
        assert_allclose(edges1, edges2)

    # 测试函数，验证 binned_statistic_dd 函数计算结果中的 binnumber 是否与预期的 bc2 数组一致
    def test_dd_bincode(self):
        # 获取部分数据 X 和 v
        X = self.X[:20]
        v = self.v[:20]

        # 使用 'count' 统计类型调用 binned_statistic_dd 函数
        count1, edges1, bc = binned_statistic_dd(X, v, 'count', bins=3)
        # 预期的 binnumber 数组
        bc2 = np.array([63, 33, 86, 83, 88, 67, 57, 33, 42, 41, 82, 83, 92,
                        32, 36, 91, 43, 87, 81, 81])

        # 计算实际的每个 bin 的计数
        bcount = [(bc == i).sum() for i in np.unique(bc)]

        # 断言实际的 binnumber 数组与预期的 bc2 数组的近似程度
        assert_allclose(bc, bc2)
        # 获取非零计数的部分 count1adj
        count1adj = count1[count1.nonzero()]
        # 断言实际的每个 bin 的计数与预期的 bcount 数组的近似程度
        assert_allclose(bcount, count1adj)
    def test_dd_multi_values(self):
        X = self.X
        v = self.v
        w = self.w

        # 遍历多种统计方法
        for stat in ["count", "sum", "mean", "std", "min", "max", "median",
                     np.std]:
            # 对 X, v 进行多维统计
            stat1v, edges1v, bc1v = binned_statistic_dd(X, v, stat, bins=8)
            # 对 X, w 进行多维统计
            stat1w, edges1w, bc1w = binned_statistic_dd(X, w, stat, bins=8)
            # 对 X, [v, w] 进行多维统计
            stat2, edges2, bc2 = binned_statistic_dd(X, [v, w], stat, bins=8)
            # 检验多维统计结果的一致性
            assert_allclose(stat2[0], stat1v)
            assert_allclose(stat2[1], stat1w)
            assert_allclose(edges1v, edges2)
            assert_allclose(edges1w, edges2)
            assert_allclose(bc1v, bc2)

    def test_dd_binnumbers_unraveled(self):
        X = self.X
        v = self.v

        # 对 X 的每个维度分别进行均值统计
        stat, edgesx, bcx = binned_statistic(X[:, 0], v, 'mean', bins=15)
        stat, edgesy, bcy = binned_statistic(X[:, 1], v, 'mean', bins=20)
        stat, edgesz, bcz = binned_statistic(X[:, 2], v, 'mean', bins=10)

        # 对 X, v 进行多维统计，并展开 bin 编号
        stat2, edges2, bc2 = binned_statistic_dd(
            X, v, 'mean', bins=(15, 20, 10), expand_binnumbers=True)

        # 检验展开的 bin 编号的一致性
        assert_allclose(bcx, bc2[0])
        assert_allclose(bcy, bc2[1])
        assert_allclose(bcz, bc2[2])

    def test_dd_binned_statistic_result(self):
        # 注意：测试从先前调用中重用的 bin_edges
        x = np.random.random((10000, 3))
        v = np.random.random(10000)
        bins = np.linspace(0, 1, 10)
        bins = (bins, bins, bins)

        # 第一次调用 binned_statistic_dd
        result = binned_statistic_dd(x, v, 'mean', bins=bins)
        stat = result.statistic

        # 第二次调用 binned_statistic_dd，重用之前的结果
        result = binned_statistic_dd(x, v, 'mean',
                                     binned_statistic_result=result)
        stat2 = result.statistic

        # 检验两次调用结果的一致性
        assert_allclose(stat, stat2)

    def test_dd_zero_dedges(self):
        x = np.random.random((10000, 3))
        v = np.random.random(10000)
        bins = np.linspace(0, 1, 10)
        bins = np.append(bins, 1)
        bins = (bins, bins, bins)

        # 测试当 bin 边界差值数值上为 0 时的情况
        with assert_raises(ValueError, match='difference is numerically 0'):
            binned_statistic_dd(x, v, 'mean', bins=bins)
    def test_dd_range_errors(self):
        # 测试对于 `range` 参数错误值时是否适当引发描述性异常。（参见 gh-12996）

        # 检查 `start` 必须小于等于 `stop` 的条件，应引发 ValueError 异常
        with assert_raises(ValueError,
                           match='In range, start must be <= stop'):
            binned_statistic_dd([self.y], self.v,
                                range=[[1, 0]])

        # 检查对于第一个维度的 `range`，`start` 必须小于等于 `stop` 的条件，应引发 ValueError 异常
        with assert_raises(
                ValueError,
                match='In dimension 1 of range, start must be <= stop'):
            binned_statistic_dd([self.x, self.y], self.v,
                                range=[[1, 0], [0, 1]])

        # 检查对于第二个维度的 `range`，`start` 必须小于等于 `stop` 的条件，应引发 ValueError 异常
        with assert_raises(
                ValueError,
                match='In dimension 2 of range, start must be <= stop'):
            binned_statistic_dd([self.x, self.y], self.v,
                                range=[[0, 1], [1, 0]])

        # 检查给定的 `range` 维度为 1，但需要 2 维的条件，应引发 ValueError 异常
        with assert_raises(
                ValueError,
                match='range given for 1 dimensions; 2 required'):
            binned_statistic_dd([self.x, self.y], self.v,
                                range=[[0, 1]])

    def test_binned_statistic_float32(self):
        # 测试对于 float32 类型数据的 binned_statistic 函数

        # 创建一个 float32 类型的数组 X
        X = np.array([0, 0.42358226], dtype=np.float32)

        # 调用 binned_statistic 函数计算统计信息，使用 'count' 统计方式，分为 5 个 bin
        stat, _, _ = binned_statistic(X, None, 'count', bins=5)

        # 断言统计结果与预期的 float64 类型数组一致
        assert_allclose(stat, np.array([1, 0, 0, 0, 1], dtype=np.float64))

    def test_gh14332(self):
        # 测试当样本 `sample` 靠近 bin 边缘时的输出是否正确

        # 创建一个列表 x，包含 20 个接近于 1 的值
        x = []
        size = 20
        for i in range(size):
            x += [1-0.1**i]

        # 使用 np.linspace 创建分 bin 的边界 bins
        bins = np.linspace(0,1,11)

        # 使用 binned_statistic_dd 计算以 x 为样本，以 np.ones(len(x)) 为权重的 sum 统计量
        sum1, edges1, bc = binned_statistic_dd(x, np.ones(len(x)),
                                               bins=[bins], statistic='sum')

        # 使用 np.histogram 计算 x 的直方图，以比较结果
        sum2, edges2 = np.histogram(x, bins=bins)

        # 断言两种计算方式的结果应当非常接近
        assert_allclose(sum1, sum2)
        assert_allclose(edges1[0], edges2)

    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("statistic", [np.mean, np.median, np.sum, np.std,
                                           np.min, np.max, 'count',
                                           lambda x: (x**2).sum(),
                                           lambda x: (x**2).sum() * 1j])
    def test_dd_all(self, dtype, statistic):
        # 测试对所有 dtype 和统计方法的 binned_statistic_dd 函数

        def ref_statistic(x):
            # 根据 statistic 函数计算参考统计量
            return len(x) if statistic == 'count' else statistic(x)

        # 创建一个随机数生成器 rng
        rng = np.random.default_rng(3704743126639371)
        n = 10
        x = rng.random(size=n)
        i = x >= 0.5
        v = rng.random(size=n)

        if dtype is np.complex128:
            v = v + rng.random(size=n)*1j

        # 调用 binned_statistic_dd 函数计算统计量 stat
        stat, _, _ = binned_statistic_dd(x, v, statistic, bins=2)

        # 计算参考统计量 ref
        ref = np.array([ref_statistic(v[~i]), ref_statistic(v[i])])

        # 断言计算得到的统计量 stat 与参考统计量 ref 非常接近
        assert_allclose(stat, ref)

        # 断言 stat 的数据类型与参考 ref 的数据类型一致，且为 np.float64 类型
        assert stat.dtype == np.result_type(ref.dtype, np.float64)
```