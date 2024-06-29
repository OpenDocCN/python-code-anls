# `.\numpy\numpy\lib\tests\test_histograms.py`

```py
import numpy as np  # 导入 NumPy 库

# 从 NumPy 中导入直方图相关函数和断言方法
from numpy import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_raises, assert_allclose,
    assert_array_max_ulp, assert_raises_regex, suppress_warnings,
    )
# 从 NumPy 测试工具私有模块导入需要内存的装饰器
from numpy.testing._private.utils import requires_memory
import pytest  # 导入 pytest 库


class TestHistogram:  # 定义测试直方图的测试类

    def setup_method(self):  # 测试方法的初始化操作
        pass

    def teardown_method(self):  # 测试方法的清理操作
        pass

    def test_simple(self):  # 测试简单的直方图操作
        n = 100
        v = np.random.rand(n)  # 生成包含 n 个随机数的数组
        (a, b) = histogram(v)  # 计算 v 的直方图
        # 检查直方图的每个箱子的和是否等于样本数量 n
        assert_equal(np.sum(a, axis=0), n)
        # 当数据来自线性函数时，检查箱子计数是否均匀分布
        (a, b) = histogram(np.linspace(0, 10, 100))
        assert_array_equal(a, 10)

    def test_one_bin(self):  # 测试包含一个箱子的直方图操作
        # Ticket 632
        hist, edges = histogram([1, 2, 3, 4], [1, 2])
        assert_array_equal(hist, [2, ])  # 检查直方图的计数是否正确
        assert_array_equal(edges, [1, 2])  # 检查直方图的边界是否正确
        assert_raises(ValueError, histogram, [1, 2], bins=0)  # 检查是否会引发值错误异常
        h, e = histogram([1, 2], bins=1)  # 计算包含一个箱子的直方图
        assert_equal(h, np.array([2]))  # 检查直方图的计数是否正确
        assert_allclose(e, np.array([1., 2.]))  # 检查直方图的边界是否正确

    def test_density(self):  # 测试密度直方图操作
        # 检查密度函数的积分是否等于 1
        n = 100
        v = np.random.rand(n)
        a, b = histogram(v, density=True)  # 计算密度直方图
        area = np.sum(a * np.diff(b))  # 计算密度函数的积分面积
        assert_almost_equal(area, 1)

        # 使用非常数箱宽度进行检查
        v = np.arange(10)
        bins = [0, 1, 3, 6, 10]
        a, b = histogram(v, bins, density=True)  # 计算非常数箱宽度的密度直方图
        assert_array_equal(a, .1)  # 检查密度直方图的计数是否正确
        assert_equal(np.sum(a * np.diff(b)), 1)  # 检查密度函数的积分面积是否等于 1

        # 测试传递 False 参数也能正常工作
        a, b = histogram(v, bins, density=False)  # 计算非常数箱宽度的直方图
        assert_array_equal(a, [1, 2, 3, 4])

        # 变量箱宽度在处理无穷大时特别有用
        v = np.arange(10)
        bins = [0, 1, 3, 6, np.inf]
        a, b = histogram(v, bins, density=True)  # 计算变量箱宽度的密度直方图
        assert_array_equal(a, [.1, .1, .1, 0.])

        # 从 numpy-discussion 邮件列表的 bug 报告中获取的测试用例
        counts, dmy = np.histogram(
            [1, 2, 3, 4], [0.5, 1.5, np.inf], density=True)
        assert_equal(counts, [.25, 0])
    def test_outliers(self):
        # Check that outliers are not tallied
        # 创建一个包含10个元素的数组，从0.5到9.5
        a = np.arange(10) + .5

        # Lower outliers
        # 对数组进行直方图统计，限定范围为[0, 9]
        h, b = histogram(a, range=[0, 9])
        # 断言直方图中所有元素的和为9
        assert_equal(h.sum(), 9)

        # Upper outliers
        # 对数组进行直方图统计，限定范围为[1, 10]
        h, b = histogram(a, range=[1, 10])
        # 断言直方图中所有元素的和为9
        assert_equal(h.sum(), 9)

        # Normalization
        # 对数组进行直方图统计，限定范围为[1, 9]，进行密度归一化
        h, b = histogram(a, range=[1, 9], density=True)
        # 断言归一化后直方图乘以区间宽度的和为1，精确度到小数点后15位
        assert_almost_equal((h * np.diff(b)).sum(), 1, decimal=15)

        # Weights
        # 创建一个权重数组，从0.5到9.5
        w = np.arange(10) + .5
        # 对数组进行直方图统计，限定范围为[1, 9]，使用权重w，进行密度归一化
        h, b = histogram(a, range=[1, 9], weights=w, density=True)
        # 断言归一化后直方图乘以区间宽度的和为1
        assert_equal((h * np.diff(b)).sum(), 1)

        # 对数组进行直方图统计，使用8个bins，限定范围为[1, 9]，使用权重w
        h, b = histogram(a, bins=8, range=[1, 9], weights=w)
        # 断言直方图的值等于权重数组w的中间部分
        assert_equal(h, w[1:-1])

    def test_arr_weights_mismatch(self):
        # 测试数组和权重数组长度不匹配时的情况
        a = np.arange(10) + .5
        w = np.arange(11) + .5
        # 使用断言检查是否抛出ValueError异常，异常信息包含"same shape as"
        with assert_raises_regex(ValueError, "same shape as"):
            h, b = histogram(a, range=[1, 9], weights=w, density=True)


    def test_type(self):
        # 检查返回直方图的数据类型
        a = np.arange(10) + .5
        # 对数组a进行直方图统计
        h, b = histogram(a)
        # 断言直方图的数据类型是整数类型
        assert_(np.issubdtype(h.dtype, np.integer))

        # 对数组a进行直方图统计，进行密度归一化
        h, b = histogram(a, density=True)
        # 断言直方图的数据类型是浮点数类型
        assert_(np.issubdtype(h.dtype, np.floating))

        # 对数组a进行直方图统计，使用整数类型的权重数组
        h, b = histogram(a, weights=np.ones(10, int))
        # 断言直方图的数据类型是整数类型
        assert_(np.issubdtype(h.dtype, np.integer))

        # 对数组a进行直方图统计，使用浮点数类型的权重数组
        h, b = histogram(a, weights=np.ones(10, float))
        # 断言直方图的数据类型是浮点数类型
        assert_(np.issubdtype(h.dtype, np.floating))

    def test_f32_rounding(self):
        # gh-4799, 检查在float32类型下边界的四舍五入
        x = np.array([276.318359, -69.593948, 21.329449], dtype=np.float32)
        y = np.array([5005.689453, 4481.327637, 6010.369629], dtype=np.float32)
        # 使用二维直方图统计x和y的分布，使用100个bins
        counts_hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        # 断言直方图中所有元素的和为3
        assert_equal(counts_hist.sum(), 3.)

    def test_bool_conversion(self):
        # gh-12107
        # 参考整数类型的直方图
        a = np.array([1, 1, 0], dtype=np.uint8)
        int_hist, int_edges = np.histogram(a)

        # 应该在布尔类型的输入上发出警告
        # 确保直方图是等价的，需要抑制警告以获取实际输出
        with suppress_warnings() as sup:
            # 记录RuntimeWarning类型的警告，内容包含"Converting input from .*"
            rec = sup.record(RuntimeWarning, 'Converting input from .*')
            # 对布尔数组进行直方图统计
            hist, edges = np.histogram([True, True, False])
            # 断言记录的警告数量为1
            assert_equal(len(rec), 1)
            # 断言直方图的值与整数类型的直方图相等
            assert_array_equal(hist, int_hist)
            # 断言直方图的边界与整数类型的直方图边界相等
            assert_array_equal(edges, int_edges)
    def test_weights(self):
        # 生成一个包含100个随机数的数组
        v = np.random.rand(100)
        # 创建一个包含100个元素的全为5的数组
        w = np.ones(100) * 5
        # 计算未加权的直方图
        a, b = histogram(v)
        # 计算未加权的密度直方图
        na, nb = histogram(v, density=True)
        # 使用权重计算直方图
        wa, wb = histogram(v, weights=w)
        # 使用权重计算密度直方图
        nwa, nwb = histogram(v, weights=w, density=True)
        # 断言加权直方图的结果是否准确
        assert_array_almost_equal(a * 5, wa)
        # 断言加权密度直方图的结果是否准确
        assert_array_almost_equal(na, nwa)

        # 检查权重是否被正确应用
        v = np.linspace(0, 10, 10)
        # 创建一个包含5个0和5个1的权重数组
        w = np.concatenate((np.zeros(5), np.ones(5)))
        # 使用权重计算直方图，指定了自定义的bins
        wa, wb = histogram(v, bins=np.arange(11), weights=w)
        # 断言结果是否准确
        assert_array_almost_equal(wa, w)

        # 检查使用整数权重的情况
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1])
        # 断言结果是否准确
        assert_array_equal(wa, [4, 5, 0, 1])
        # 使用密度选项检查整数权重的情况
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1], density=True)
        # 断言结果是否准确
        assert_array_almost_equal(wa, np.array([4, 5, 0, 1]) / 10. / 3. * 4)

        # 检查非均匀bin宽度的权重情况
        a, b = histogram(np.arange(9), [0, 1, 3, 6, 10],
                         weights=[2, 1, 1, 1, 1, 1, 1, 1, 1], density=True)
        # 断言结果是否准确
        assert_almost_equal(a, [.2, .1, .1, .075])

    def test_exotic_weights(self):

        # 测试使用不是整数或浮点数的权重，如复数或对象类型。

        # 复数权重
        values = np.array([1.3, 2.5, 2.3])
        weights = np.array([1, -1, 2]) + 1j * np.array([2, 1, 2])

        # 使用自定义bins检查
        wa, wb = histogram(values, bins=[0, 2, 3], weights=weights)
        # 断言结果是否准确
        assert_array_almost_equal(wa, np.array([1, 1]) + 1j * np.array([2, 3]))

        # 使用均匀bins检查
        wa, wb = histogram(values, bins=2, range=[1, 3], weights=weights)
        # 断言结果是否准确
        assert_array_almost_equal(wa, np.array([1, 1]) + 1j * np.array([2, 3]))

        # 十进制权重
        from decimal import Decimal
        values = np.array([1.3, 2.5, 2.3])
        weights = np.array([Decimal(1), Decimal(2), Decimal(3)])

        # 使用自定义bins检查
        wa, wb = histogram(values, bins=[0, 2, 3], weights=weights)
        # 断言结果是否准确
        assert_array_almost_equal(wa, [Decimal(1), Decimal(5)])

        # 使用均匀bins检查
        wa, wb = histogram(values, bins=2, range=[1, 3], weights=weights)
        # 断言结果是否准确
        assert_array_almost_equal(wa, [Decimal(1), Decimal(5)])

    def test_no_side_effects(self):
        # 这是一个回归测试，确保传递给“histogram”的值未改变。
        values = np.array([1.3, 2.5, 2.3])
        np.histogram(values, range=[-10, 10], bins=100)
        # 断言原始数据是否未改变
        assert_array_almost_equal(values, [1.3, 2.5, 2.3])

    def test_empty(self):
        # 测试空数组的情况
        a, b = histogram([], bins=([0, 1]))
        # 断言结果是否准确
        assert_array_equal(a, np.array([0]))
        assert_array_equal(b, np.array([0, 1]))
    def test_error_binnum_type (self):
        # 测试当 bins 参数为浮点数时是否会引发正确的错误
        vals = np.linspace(0.0, 1.0, num=100)
        # 调用 histogram 函数进行直方图计算
        histogram(vals, 5)
        # 使用 assert_raises 确保 TypeError 被引发
        assert_raises(TypeError, histogram, vals, 2.4)

    def test_finite_range(self):
        # 正常的范围应该是可以的
        vals = np.linspace(0.0, 1.0, num=100)
        # 使用指定的范围调用 histogram 函数
        histogram(vals, range=[0.25,0.75])
        # 使用 assert_raises 确保 ValueError 被引发
        assert_raises(ValueError, histogram, vals, range=[np.nan,0.75])
        assert_raises(ValueError, histogram, vals, range=[0.25,np.inf])

    def test_invalid_range(self):
        # 范围的起始值必须小于结束值
        vals = np.linspace(0.0, 1.0, num=100)
        # 使用 assert_raises_regex 确保 ValueError 被引发，并且错误消息匹配
        with assert_raises_regex(ValueError, "max must be larger than"):
            np.histogram(vals, range=[0.1, 0.01])

    def test_bin_edge_cases(self):
        # 确保浮点数计算能够正确处理边缘情况
        arr = np.array([337, 404, 739, 806, 1007, 1811, 2012])
        # 调用 np.histogram 计算直方图和边缘值
        hist, edges = np.histogram(arr, bins=8296, range=(2, 2280))
        # 创建一个掩码以过滤掉零值的边缘
        mask = hist > 0
        left_edges = edges[:-1][mask]
        right_edges = edges[1:][mask]
        # 使用 assert_ 确保每个值 x 处于其对应的左右边缘之间
        for x, left, right in zip(arr, left_edges, right_edges):
            assert_(x >= left)
            assert_(x < right)

    def test_last_bin_inclusive_range(self):
        arr = np.array([0.,  0.,  0.,  1.,  2.,  3.,  3.,  4.,  5.])
        # 调用 np.histogram 计算直方图和边缘值
        hist, edges = np.histogram(arr, bins=30, range=(-0.5, 5))
        # 使用 assert_equal 确保最后一个 bin 的计数为 1
        assert_equal(hist[-1], 1)

    def test_bin_array_dims(self):
        # 优雅地处理 bins 对象大于 1 维的情况
        vals = np.linspace(0.0, 1.0, num=100)
        bins = np.array([[0, 0.5], [0.6, 1.0]])
        # 使用 assert_raises_regex 确保 ValueError 被引发，并且错误消息匹配
        with assert_raises_regex(ValueError, "must be 1d"):
            np.histogram(vals, bins=bins)

    def test_unsigned_monotonicity_check(self):
        # 确保当 bins 不单调递增时引发 ValueError
        # 当 bins 包含无符号值时 (见＃9222)
        arr = np.array([2])
        bins = np.array([1, 3, 1], dtype='uint64')
        # 使用 assert_raises 确保 ValueError 被引发
        with assert_raises(ValueError):
            hist, edges = np.histogram(arr, bins=bins)

    def test_object_array_of_0d(self):
        # 确保当数组中有 0 维对象时引发 ValueError
        assert_raises(ValueError,
            histogram, [np.array(0.4) for i in range(10)] + [-np.inf])
        assert_raises(ValueError,
            histogram, [np.array(0.4) for i in range(10)] + [np.inf])

        # 这些应该不会崩溃
        np.histogram([np.array(0.5) for i in range(10)] + [.500000000000001])
        np.histogram([np.array(0.5) for i in range(10)] + [.5])
    def test_some_nan_values(self):
        # 定义包含一个 NaN 值的数组和全是 NaN 值的数组
        one_nan = np.array([0, 1, np.nan])
        all_nan = np.array([np.nan, np.nan])

        # 使用 suppress_warnings 函数来抑制警告信息
        sup = suppress_warnings()
        # 过滤 RuntimeWarning 类型的警告
        sup.filter(RuntimeWarning)
        with sup:
            # 对含有 NaN 的数组使用 bins='auto' 参数会引发 ValueError 异常
            assert_raises(ValueError, histogram, one_nan, bins='auto')
            assert_raises(ValueError, histogram, all_nan, bins='auto')

            # 使用明确的 range 参数解决 NaN 值的问题
            h, b = histogram(one_nan, bins='auto', range=(0, 1))
            assert_equal(h.sum(), 2)  # NaN 值不会被计算
            h, b = histogram(all_nan, bins='auto', range=(0, 1))
            assert_equal(h.sum(), 0)  # NaN 值不会被计算

            # 使用明确的 bins 参数也可以解决 NaN 值的问题
            h, b = histogram(one_nan, bins=[0, 1])
            assert_equal(h.sum(), 2)  # NaN 值不会被计算
            h, b = histogram(all_nan, bins=[0, 1])
            assert_equal(h.sum(), 0)  # NaN 值不会被计算

    def test_datetime(self):
        # 定义起始日期和日期偏移量数组
        begin = np.datetime64('2000-01-01', 'D')
        offsets = np.array([0, 0, 1, 1, 2, 3, 5, 10, 20])
        bins = np.array([0, 2, 7, 20])
        dates = begin + offsets
        date_bins = begin + bins

        # 定义 timedelta64[D] 类型
        td = np.dtype('timedelta64[D]')

        # 日期和整数偏移量应该有相同的结果
        # 目前只支持显式指定的 bins，因为 linspace 不适用于日期或 timedelta
        d_count, d_edge = histogram(dates, bins=date_bins)
        t_count, t_edge = histogram(offsets.astype(td), bins=bins.astype(td))
        i_count, i_edge = histogram(offsets, bins=bins)

        assert_equal(d_count, i_count)
        assert_equal(t_count, i_count)

        assert_equal((d_edge - begin).astype(int), i_edge)
        assert_equal(t_edge.astype(int), i_edge)

        assert_equal(d_edge.dtype, dates.dtype)
        assert_equal(t_edge.dtype, td)

    def do_signed_overflow_bounds(self, dtype):
        # 计算给定 dtype 类型的最大指数
        exponent = 8 * np.dtype(dtype).itemsize - 1
        # 创建一个包含接近边界值的数组
        arr = np.array([-2**exponent + 4, 2**exponent - 4], dtype=dtype)
        # 对数组进行直方图统计，期望返回边界值数组和对应的直方图
        hist, e = histogram(arr, bins=2)
        assert_equal(e, [-2**exponent + 4, 0, 2**exponent - 4])
        assert_equal(hist, [1, 1])

    def test_signed_overflow_bounds(self):
        # 测试不同类型的有符号整数溢出边界
        self.do_signed_overflow_bounds(np.byte)
        self.do_signed_overflow_bounds(np.short)
        self.do_signed_overflow_bounds(np.intc)
        self.do_signed_overflow_bounds(np.int_)
        self.do_signed_overflow_bounds(np.longlong)
    def do_precision_lower_bound(self, float_small, float_large):
        # 获取 float_large 的机器精度
        eps = np.finfo(float_large).eps

        # 创建一个包含单个浮点数的 numpy 数组，使用 float_small 类型
        arr = np.array([1.0], float_small)
        # 创建一个包含两个浮点数的 numpy 数组，使用 float_large 类型
        range = np.array([1.0 + eps, 2.0], float_large)

        # 测试当数据类型变化时的行为
        if range.astype(float_small)[0] != 1:
            return

        # 调用 np.histogram 计算直方图，只有一个 bin，使用指定的范围
        count, x_loc = np.histogram(arr, bins=1, range=range)
        # 断言直方图中的计数为 [0]
        assert_equal(count, [0])
        # 断言 x_loc 的数据类型为 float_large

    def do_precision_upper_bound(self, float_small, float_large):
        # 获取 float_large 的机器精度
        eps = np.finfo(float_large).eps

        # 创建一个包含单个浮点数的 numpy 数组，使用 float_small 类型
        arr = np.array([1.0], float_small)
        # 创建一个包含两个浮点数的 numpy 数组，使用 float_large 类型
        range = np.array([0.0, 1.0 - eps], float_large)

        # 测试当数据类型变化时的行为
        if range.astype(float_small)[-1] != 1:
            return

        # 调用 np.histogram 计算直方图，只有一个 bin，使用指定的范围
        count, x_loc = np.histogram(arr, bins=1, range=range)
        # 断言直方图中的计数为 [0]
        assert_equal(count, [0])
        # 断言 x_loc 的数据类型为 float_large

    def do_precision(self, float_small, float_large):
        # 调用 do_precision_lower_bound 和 do_precision_upper_bound 方法
        self.do_precision_lower_bound(float_small, float_large)
        self.do_precision_upper_bound(float_small, float_large)

    def test_precision(self):
        # 依次调用 do_precision 方法，测试不同的浮点数类型组合
        self.do_precision(np.half, np.single)
        self.do_precision(np.half, np.double)
        self.do_precision(np.half, np.longdouble)
        self.do_precision(np.single, np.double)
        self.do_precision(np.single, np.longdouble)
        self.do_precision(np.double, np.longdouble)

    def test_histogram_bin_edges(self):
        # 测试 np.histogram_bin_edges 函数的行为
        hist, e = histogram([1, 2, 3, 4], [1, 2])
        edges = histogram_bin_edges([1, 2, 3, 4], [1, 2])
        assert_array_equal(edges, e)

        # 创建一个 numpy 数组 arr
        arr = np.array([0.,  0.,  0.,  1.,  2.,  3.,  3.,  4.,  5.])
        # 测试 np.histogram 函数的行为，使用指定的 bins 和 range
        hist, e = histogram(arr, bins=30, range=(-0.5, 5))
        edges = histogram_bin_edges(arr, bins=30, range=(-0.5, 5))
        assert_array_equal(edges, e)

        # 测试 np.histogram 函数的行为，使用 'auto' 作为 bins 和指定的 range
        hist, e = histogram(arr, bins='auto', range=(0, 1))
        edges = histogram_bin_edges(arr, bins='auto', range=(0, 1))
        assert_array_equal(edges, e)

    # @requires_memory(free_bytes=1e10)
    # @pytest.mark.slow
    @pytest.mark.skip(reason="Bad memory reports lead to OOM in ci testing")
    def test_big_arrays(self):
        # 创建一个非常大的 numpy 数组 sample
        sample = np.zeros([100000000, 3])
        xbins = 400
        ybins = 400
        zbins = np.arange(16000)
        # 调用 np.histogramdd 计算多维直方图，使用指定的 bins
        hist = np.histogramdd(sample=sample, bins=(xbins, ybins, zbins))
        # 断言 hist 的类型是 tuple

    def test_gh_23110(self):
        # 测试针对 GitHub 问题 23110 的情况
        hist, e = np.histogram(np.array([-0.9e-308], dtype='>f8'),
                               bins=2,
                               range=(-1e-308, -2e-313))
        expected_hist = np.array([1, 0])
        assert_array_equal(hist, expected_hist)
class TestHistogramOptimBinNums:
    """
    Provide test coverage when using provided estimators for optimal number of
    bins
    """

    def test_empty(self):
        estimator_list = ['fd', 'scott', 'rice', 'sturges',
                          'doane', 'sqrt', 'auto', 'stone']
        # check it can deal with empty data
        # 对空数据的处理
        for estimator in estimator_list:
            # 调用直方图函数，对空数据进行处理
            a, b = histogram([], bins=estimator)
            # 断言直方图结果为预期的空数组
            assert_array_equal(a, np.array([0]))
            # 断言直方图的边界为预期的 [0, 1]
            assert_array_equal(b, np.array([0, 1]))

    def test_simple(self):
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). All test values have been precomputed and the values
        shouldn't change
        """
        # Some basic sanity checking, with some fixed data.
        # Checking for the correct number of bins
        # 使用一些固定数据进行基本的检查，检验正确的 bin 数量
        basic_test = {50:   {'fd': 4,  'scott': 4,  'rice': 8,  'sturges': 7,
                             'doane': 8, 'sqrt': 8, 'auto': 7, 'stone': 2},
                      500:  {'fd': 8,  'scott': 8,  'rice': 16, 'sturges': 10,
                             'doane': 12, 'sqrt': 23, 'auto': 10, 'stone': 9},
                      5000: {'fd': 17, 'scott': 17, 'rice': 35, 'sturges': 14,
                             'doane': 17, 'sqrt': 71, 'auto': 17, 'stone': 20}}

        for testlen, expectedResults in basic_test.items():
            # Create some sort of non uniform data to test with
            # (2 peak uniform mixture)
            # 创建一些非均匀数据进行测试（包含两个峰值的混合数据）
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x = np.concatenate((x1, x2))
            for estimator, numbins in expectedResults.items():
                # 调用直方图函数，计算直方图和边界
                a, b = np.histogram(x, estimator)
                # 断言直方图的长度为预期的 bin 数量
                assert_equal(len(a), numbins, err_msg="For the {0} estimator "
                             "with datasize of {1}".format(estimator, testlen))

    def test_small(self):
        """
        Smaller datasets have the potential to cause issues with the data
        adaptive methods, especially the FD method. All bin numbers have been
        precalculated.
        """
        # 较小的数据集可能会导致数据自适应方法出现问题，特别是 FD 方法
        small_dat = {1: {'fd': 1, 'scott': 1, 'rice': 1, 'sturges': 1,
                         'doane': 1, 'sqrt': 1, 'stone': 1},
                     2: {'fd': 2, 'scott': 1, 'rice': 3, 'sturges': 2,
                         'doane': 1, 'sqrt': 2, 'stone': 1},
                     3: {'fd': 2, 'scott': 2, 'rice': 3, 'sturges': 3,
                         'doane': 3, 'sqrt': 2, 'stone': 1}}

        for testlen, expectedResults in small_dat.items():
            testdat = np.arange(testlen).astype(float)
            for estimator, expbins in expectedResults.items():
                # 调用直方图函数，计算直方图和边界
                a, b = np.histogram(testdat, estimator)
                # 断言直方图的长度为预期的 bin 数量
                assert_equal(len(a), expbins, err_msg="For the {0} estimator "
                             "with datasize of {1}".format(estimator, testlen))
    def test_incorrect_methods(self):
        """
        Check a Value Error is thrown when an unknown string is passed in
        """
        # 定义需要检查的估算器列表
        check_list = ['mad', 'freeman', 'histograms', 'IQR']
        # 对于每个估算器进行检查
        for estimator in check_list:
            # 断言调用直方图函数时，传入未知字符串会引发 ValueError 异常
            assert_raises(ValueError, histogram, [1, 2, 3], estimator)

    def test_novariance(self):
        """
        Check that methods handle no variance in data
        Primarily for Scott and FD as the SD and IQR are both 0 in this case
        """
        # 创建方差为0的数据集
        novar_dataset = np.ones(100)
        # 预期的结果字典，键为估算器名称，值为对应的预期箱数
        novar_resultdict = {'fd': 1, 'scott': 1, 'rice': 1, 'sturges': 1,
                            'doane': 1, 'sqrt': 1, 'auto': 1, 'stone': 1}

        # 对于每个估算器，执行直方图计算并进行断言
        for estimator, numbins in novar_resultdict.items():
            # 调用 numpy 的直方图函数计算直方图
            a, b = np.histogram(novar_dataset, estimator)
            # 断言实际得到的箱数与预期的箱数一致
            assert_equal(len(a), numbins, err_msg="{0} estimator, "
                         "No Variance test".format(estimator))

    def test_limited_variance(self):
        """
        Check when IQR is 0, but variance exists, we return the sturges value
        and not the fd value.
        """
        # 创建有限制方差的数据集
        lim_var_data = np.ones(1000)
        lim_var_data[:3] = 0
        lim_var_data[-4:] = 100

        # 调用直方图边界函数，期望返回自动计算的边界数组
        edges_auto = histogram_bin_edges(lim_var_data, 'auto')
        assert_equal(edges_auto, np.linspace(0, 100, 12))

        # 调用直方图边界函数，期望返回 FD 估算器计算的边界数组
        edges_fd = histogram_bin_edges(lim_var_data, 'fd')
        assert_equal(edges_fd, np.array([0, 100]))

        # 调用直方图边界函数，期望返回 Sturges 估算器计算的边界数组
        edges_sturges = histogram_bin_edges(lim_var_data, 'sturges')
        assert_equal(edges_sturges, np.linspace(0, 100, 12))

    def test_outlier(self):
        """
        Check the FD, Scott and Doane with outliers.

        The FD estimates a smaller binwidth since it's less affected by
        outliers. Since the range is so (artificially) large, this means more
        bins, most of which will be empty, but the data of interest usually is
        unaffected. The Scott estimator is more affected and returns fewer bins,
        despite most of the variance being in one area of the data. The Doane
        estimator lies somewhere between the other two.
        """
        # 创建带有异常值的数据集
        xcenter = np.linspace(-10, 10, 50)
        outlier_dataset = np.hstack((np.linspace(-110, -100, 5), xcenter))

        # 预期的结果字典，键为估算器名称，值为对应的预期箱数
        outlier_resultdict = {'fd': 21, 'scott': 5, 'doane': 11, 'stone': 6}

        # 对于每个估算器，执行直方图计算并进行断言
        for estimator, numbins in outlier_resultdict.items():
            # 调用 numpy 的直方图函数计算直方图
            a, b = np.histogram(outlier_dataset, estimator)
            # 断言实际得到的箱数与预期的箱数一致
            assert_equal(len(a), numbins)
    def test_scott_vs_stone(self):
        """Verify that Scott's rule and Stone's rule converges for normally distributed data"""

        def nbins_ratio(seed, size):
            # 创建一个指定种子和大小的随机数生成器
            rng = np.random.RandomState(seed)
            # 生成均值为0，标准差为2的正态分布数据
            x = rng.normal(loc=0, scale=2, size=size)
            # 分别计算用 Stone 和 Scott 方法计算的直方图的 bin 数量
            a, b = len(np.histogram(x, 'stone')[0]), len(np.histogram(x, 'scott')[0])
            # 返回 Stone 方法计算的直方图 bin 数量与总数之比
            return a / (a + b)

        # 生成包含不同数据集大小的 nbins_ratio 结果的列表
        ll = [[nbins_ratio(seed, size) for size in np.geomspace(start=10, stop=100, num=4).round().astype(int)]
              for seed in range(10)]

        # 计算每个数据集大小的平均差异，与期望值进行比较
        avg = abs(np.mean(ll, axis=0) - 0.5)
        # 断言平均差异接近预期值
        assert_almost_equal(avg, [0.15, 0.09, 0.08, 0.03], decimal=2)

    def test_simple_range(self):
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). Adding in a 3rd mixture that will then be
        completely ignored. All test values have been precomputed and
        the shouldn't change.
        """
        # 一些基本的合理性检查，使用预先计算好的固定数据集
        # 检查正确的 bin 数量
        basic_test = {
                      50:   {'fd': 8,  'scott': 8,  'rice': 15,
                             'sturges': 14, 'auto': 14, 'stone': 8},
                      500:  {'fd': 15, 'scott': 16, 'rice': 32,
                             'sturges': 20, 'auto': 20, 'stone': 80},
                      5000: {'fd': 33, 'scott': 33, 'rice': 69,
                             'sturges': 27, 'auto': 33, 'stone': 80}
                     }

        for testlen, expectedResults in basic_test.items():
            # 创建一些非均匀数据以供测试（3个峰值的均匀混合）
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x3 = np.linspace(-100, -50, testlen)
            x = np.hstack((x1, x2, x3))
            for estimator, numbins in expectedResults.items():
                # 使用给定的估算方法和数据范围(-20, 20)计算直方图
                a, b = np.histogram(x, estimator, range=(-20, 20))
                msg = "For the {0} estimator".format(estimator)
                msg += " with datasize of {0}".format(testlen)
                # 断言直方图的 bin 数量与期望值相等
                assert_equal(len(a), numbins, err_msg=msg)

    @pytest.mark.parametrize("bins", ['auto', 'fd', 'doane', 'scott',
                                      'stone', 'rice', 'sturges'])
    def test_signed_integer_data(self, bins):
        # 回归测试 gh-14379。
        a = np.array([-2, 0, 127], dtype=np.int8)
        # 使用指定的 binning 方法计算直方图和边界
        hist, edges = np.histogram(a, bins=bins)
        hist32, edges32 = np.histogram(a.astype(np.int32), bins=bins)
        # 断言不同数据类型计算的直方图和边界相等
        assert_array_equal(hist, hist32)
        assert_array_equal(edges, edges32)
    # 定义测试函数，用于测试整数数据的直方图分 bin 后的宽度至少为 1 的要求
    def test_integer(self, bins):
        """
        Test that bin width for integer data is at least 1.
        """
        # 使用 suppress_warnings 上下文管理器，用于捕获和忽略警告
        with suppress_warnings() as sup:
            # 如果 bins 参数为 'stone'，则过滤 RuntimeWarning 警告
            if bins == 'stone':
                sup.filter(RuntimeWarning)
            # 断言直方图的 bin 边界是否与预期的 [0, 1, 2, ..., 8] 相同
            assert_equal(
                np.histogram_bin_edges(np.tile(np.arange(9), 1000), bins),
                np.arange(9))

    # 定义测试函数，用于测试非自动 binning 情况下的直方图分 bin
    def test_integer_non_auto(self):
        """
        Test that the bin-width>=1 requirement *only* applies to auto binning.
        """
        # 断言直方图的 bin 边界是否与预期的 [0, 0.5, 1, 1.5, ..., 8.5] 相同
        assert_equal(
            np.histogram_bin_edges(np.tile(np.arange(9), 1000), 16),
            np.arange(17) / 2)
        # 断言直方图的 bin 边界是否与预期的 [0.1, 0.2] 相同
        assert_equal(
            np.histogram_bin_edges(np.tile(np.arange(9), 1000), [.1, .2]),
            [.1, .2])

    # 定义测试函数，检查加权数据是否会引发 TypeError
    def test_simple_weighted(self):
        """
        Check that weighted data raises a TypeError
        """
        # 创建估算器列表，用于迭代测试
        estimator_list = ['fd', 'scott', 'rice', 'sturges', 'auto']
        for estimator in estimator_list:
            # 断言对于给定的估算器和加权数据，调用 histogram 函数会引发 TypeError
            assert_raises(TypeError, histogram, [1, 2, 3],
                          estimator, weights=[1, 2, 3])
class TestHistogramdd:
    
    def test_simple(self):
        # 创建一个二维数组 x，用于测试直方图函数 histogramdd
        x = np.array([[-.5, .5, 1.5], [-.5, 1.5, 2.5], [-.5, 2.5, .5],
                      [.5,  .5, 1.5], [.5,  1.5, 2.5], [.5,  2.5, 2.5]])
        
        # 调用 histogramdd 函数计算直方图 H 和边界 edges
        H, edges = histogramdd(x, (2, 3, 3),
                               range=[[-1, 1], [0, 3], [0, 3]])
        
        # 预期的结果数组 answer
        answer = np.array([[[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                           [[0, 1, 0], [0, 0, 1], [0, 0, 1]]])
        
        # 断言 H 和 answer 相等
        assert_array_equal(H, answer)
        
        # 检查归一化
        ed = [[-2, 0, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
        H, edges = histogramdd(x, bins=ed, density=True)
        
        # 断言 H 和归一化后的 answer 相等
        assert_(np.all(H == answer / 12.))
        
        # 检查 H 的形状是否正确
        H, edges = histogramdd(x, (2, 3, 4),
                               range=[[-1, 1], [0, 3], [0, 4]],
                               density=True)
        answer = np.array([[[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
                           [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]])
        
        # 断言 H 和归一化后的 answer 相等，精确度为小数点后四位
        assert_array_almost_equal(H, answer / 6., 4)
        
        # 检查是否能接受数组序列并且 H 的形状是否正确
        z = [np.squeeze(y) for y in np.split(x, 3, axis=1)]
        H, edges = histogramdd(
            z, bins=(4, 3, 2), range=[[-2, 2], [0, 3], [0, 2]])
        answer = np.array([[[0, 0], [0, 0], [0, 0]],
                           [[0, 1], [0, 0], [1, 0]],
                           [[0, 1], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 0]]])
        
        # 断言 H 和 answer 相等
        assert_array_equal(H, answer)
        
        # 创建一个全零的三维数组 Z
        Z = np.zeros((5, 5, 5))
        
        # 将 Z 的对角线上的元素设置为 1
        Z[list(range(5)), list(range(5)), list(range(5))] = 1.
        
        # 使用 np.arange(5) 创建直方图 H 和边界 edges
        H, edges = histogramdd([np.arange(5), np.arange(5), np.arange(5)], 5)
        
        # 断言 H 和 Z 相等
        assert_array_equal(H, Z)

    def test_shape_3d(self):
        # 在三维中测试不同长度的 bins 的所有可能排列
        bins = ((5, 4, 6), (6, 4, 5), (5, 6, 4), (4, 6, 5), (6, 5, 4),
                (4, 5, 6))
        
        # 生成随机数据 r
        r = np.random.rand(10, 3)
        
        # 对每种 bins 进行测试
        for b in bins:
            # 调用 histogramdd 计算直方图 H 和边界 edges
            H, edges = histogramdd(r, b)
            
            # 断言 H 的形状和 bins 相等
            assert_(H.shape == b)

    def test_shape_4d(self):
        # 在四维中测试不同长度的 bins 的所有可能排列
        bins = ((7, 4, 5, 6), (4, 5, 7, 6), (5, 6, 4, 7), (7, 6, 5, 4),
                (5, 7, 6, 4), (4, 6, 7, 5), (6, 5, 7, 4), (7, 5, 4, 6),
                (7, 4, 6, 5), (6, 4, 7, 5), (6, 7, 5, 4), (4, 6, 5, 7),
                (4, 7, 5, 6), (5, 4, 6, 7), (5, 7, 4, 6), (6, 7, 4, 5),
                (6, 5, 4, 7), (4, 7, 6, 5), (4, 5, 6, 7), (7, 6, 4, 5),
                (5, 4, 7, 6), (5, 6, 7, 4), (6, 4, 5, 7), (7, 5, 6, 4))
        
        # 生成随机数据 r
        r = np.random.rand(10, 4)
        
        # 对每种 bins 进行测试
        for b in bins:
            # 调用 histogramdd 计算直方图 H 和边界 edges
            H, edges = histogramdd(r, b)
            
            # 断言 H 的形状和 bins 相等
            assert_(H.shape == b)
    # 测试权重功能
    def test_weights(self):
        # 创建一个形状为 (100, 2) 的随机数组
        v = np.random.rand(100, 2)
        # 计算不带权重的多维直方图
        hist, edges = histogramdd(v)
        # 计算带权重的多维直方图，并返回归一化的直方图和边界
        n_hist, edges = histogramdd(v, density=True)
        # 使用等权重数组计算多维直方图，并返回直方图和边界
        w_hist, edges = histogramdd(v, weights=np.ones(100))
        # 断言两个直方图数组相等
        assert_array_equal(w_hist, hist)
        # 使用带权重的数组计算多维直方图，并返回归一化的直方图和边界
        w_hist, edges = histogramdd(v, weights=np.ones(100) * 2, density=True)
        # 断言两个归一化的直方图数组相等
        assert_array_equal(w_hist, n_hist)
        # 使用整型数组的两倍权重计算多维直方图，并返回直方图和边界
        w_hist, edges = histogramdd(v, weights=np.ones(100, int) * 2)
        # 断言两个直方图数组相等
        assert_array_equal(w_hist, 2 * hist)

    # 测试相同样本
    def test_identical_samples(self):
        # 创建一个形状为 (10, 2) 的零数组
        x = np.zeros((10, 2), int)
        # 计算指定边界的多维直方图，返回直方图和边界
        hist, edges = histogramdd(x, bins=2)
        # 断言第一个边界数组与给定的数组相等
        assert_array_equal(edges[0], np.array([-0.5, 0., 0.5]))

    # 测试空数组
    def test_empty(self):
        # 计算空数组的多维直方图，指定二维边界
        a, b = histogramdd([[], []], bins=([0, 1], [0, 1]))
        # 断言返回的直方图数组与给定的数组相等
        assert_array_max_ulp(a, np.array([[0.]]))
        # 计算空三维数组的多维直方图，指定二维边界
        a, b = np.histogramdd([[], [], []], bins=2)
        # 断言返回的直方图数组与给定的数组相等
        assert_array_max_ulp(a, np.zeros((2, 2, 2)))

    # 测试边界错误
    def test_bins_errors(self):
        # 有两种指定边界的方法。检查在混合使用这些方法时是否引发正确的错误
        x = np.arange(8).reshape(2, 4)
        # 断言当混合使用指定边界方法时会引发 ValueError 错误
        assert_raises(ValueError, np.histogramdd, x, bins=[-1, 2, 4, 5])
        assert_raises(ValueError, np.histogramdd, x, bins=[1, 0.99, 1, 1])
        assert_raises(
            ValueError, np.histogramdd, x, bins=[1, 1, 1, [1, 2, 3, -3]])
        # 断言当使用合法的边界数组时返回 True
        assert_(np.histogramdd(x, bins=[1, 1, 1, [1, 2, 3, 4]]))

    # 测试使用无穷边界
    def test_inf_edges(self):
        # 测试使用 +/-inf 边界是否有效。参见 Github 问题 #1788
        with np.errstate(invalid='ignore'):
            x = np.arange(6).reshape(3, 2)
            expected = np.array([[1, 0], [0, 1], [0, 1]])
            # 计算使用指定边界的多维直方图，并返回直方图和边界
            h, e = np.histogramdd(x, bins=[3, [-np.inf, 2, 10]])
            # 断言返回的直方图与预期数组相近
            assert_allclose(h, expected)
            # 计算使用指定边界数组的多维直方图，并返回直方图和边界
            h, e = np.histogramdd(x, bins=[3, np.array([-1, 2, np.inf])])
            # 断言返回的直方图与预期数组相近
            assert_allclose(h, expected)
            # 计算使用指定边界数组的多维直方图，并返回直方图和边界
            h, e = np.histogramdd(x, bins=[3, [-np.inf, 3, np.inf]])
            # 断言返回的直方图与预期数组相近
            assert_allclose(h, expected)

    # 测试最右边界
    def test_rightmost_binedge(self):
        # 测试接近最右边界的事件。参见 Github 问题 #4266
        x = [0.9999999995]
        bins = [[0., 0.5, 1.0]]
        # 计算使用指定边界的多维直方图，并返回直方图和边界
        hist, _ = histogramdd(x, bins=bins)
        # 断言直方图第一个元素为 0.0
        assert_(hist[0] == 0.0)
        # 断言直方图第二个元素为 1.0
        assert_(hist[1] == 1.)
        x = [1.0]
        bins = [[0., 0.5, 1.0]]
        # 计算使用指定边界的多维直方图，并返回直方图和边界
        hist, _ = histogramdd(x, bins=bins)
        # 断言直方图第一个元素为 0.0
        assert_(hist[0] == 0.0)
        # 断言直方图第二个元素为 1.0
        assert_(hist[1] == 1.)
        x = [1.0000000001]
        bins = [[0., 0.5, 1.0]]
        # 计算使用指定边界的多维直方图，并返回直方图和边界
        hist, _ = histogramdd(x, bins=bins)
        # 断言直方图第一个元素为 0.0
        assert_(hist[0] == 0.0)
        # 断言直方图第二个元素为 0.0
        assert_(hist[1] == 0.0)
        x = [1.0001]
        bins = [[0., 0.5, 1.0]]
        # 计算使用指定边界的多维直方图，并返回直方图和边界
        hist, _ = histogramdd(x, bins=bins)
        # 断言直方图第一个元素为 0.0
        assert_(hist[0] == 0.0)
        # 断言直方图第二个元素为 0.0
        assert_(hist[1] == 0.0)
    # 测试有限范围的情况
    def test_finite_range(self):
        # 创建一个 100x3 的随机数组
        vals = np.random.random((100, 3))
        # 对 vals 进行多维直方图统计，限定每个维度的范围
        histogramdd(vals, range=[[0.0, 1.0], [0.25, 0.75], [0.25, 0.5]])
        # 检查是否会抛出 ValueError 异常
        assert_raises(ValueError, histogramdd, vals,
                      range=[[0.0, 1.0], [0.25, 0.75], [0.25, np.inf]])
        assert_raises(ValueError, histogramdd, vals,
                      range=[[0.0, 1.0], [np.nan, 0.75], [0.25, 0.5]])

    # 测试相邻边界相等的情况
    def test_equal_edges(self):
        # 创建两个数组和相应的边界数组，进行多维直方图统计
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        x_edges = np.array([0, 2, 2])
        y_edges = 1
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        # 期望的直方图结果
        hist_expected = np.array([
            [2.],
            [1.],  # x == 2 falls in the final bin
        ])
        assert_equal(hist, hist_expected)

    # 测试边界数组的数据类型是否被保留的情况
    def test_edge_dtype(self):
        # 创建两个数组和相应的边界数组，进行多维直方图统计
        x = np.array([0, 10, 20])
        y = x / 10
        x_edges = np.array([0, 5, 15, 20])
        y_edges = x_edges / 10
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        # 检查边界数组的数据类型是否被保留
        assert_equal(edges[0].dtype, x_edges.dtype)
        assert_equal(edges[1].dtype, y_edges.dtype)

    # 测试大整数的情况
    def test_large_integers(self):
        big = 2**60  # Too large to represent with a full precision float

        x = np.array([0], np.int64)
        x_edges = np.array([-1, +1], np.int64)
        y = big + x
        y_edges = big + x_edges

        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        assert_equal(hist[0, 0], 1)

    # 测试非均匀二维密度的情况
    def test_density_non_uniform_2d(self):
        # 定义一个二维网格
        #    0 2     8
        #   0+-+-----+
        #    + |     +
        #    + |     +
        #   6+-+-----+
        #   8+-+-----+
        x_edges = np.array([0, 2, 8])
        y_edges = np.array([0, 6, 8])
        relative_areas = np.array([
            [3, 9],
            [1, 3]])

        # 确保每个区域内的点数与其面积成比例
        x = np.array([1] + [1]*3 + [7]*3 + [7]*9)
        y = np.array([7] + [1]*3 + [7]*3 + [1]*9)

        # 检查上述操作的结果是否符合预期
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges))
        assert_equal(hist, relative_areas)

        # 由于计数和面积成比例，所以结果直方图应该是均匀的
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges), density=True)
        # 检查结果直方图的值是否为 1 / (8*8)
        assert_equal(hist, 1 / (8*8))

    # 测试非均匀一维密度的情况
    def test_density_non_uniform_1d(self):
        # 对比直方图以展示结果是否相同
        v = np.arange(10)
        bins = np.array([0, 1, 3, 6, 10])
        hist, edges = histogram(v, bins, density=True)
        hist_dd, edges_dd = histogramdd((v,), (bins,), density=True)
        assert_equal(hist, hist_dd)
        assert_equal(edges, edges_dd[0])
```