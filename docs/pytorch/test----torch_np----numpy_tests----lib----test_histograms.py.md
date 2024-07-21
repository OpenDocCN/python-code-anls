# `.\pytorch\test\torch_np\numpy_tests\lib\test_histograms.py`

```
# Owner(s): ["module: dynamo"]

# 导入 functools 模块
import functools

# 导入 unittest 模块中的 expectedFailure 和 skipIf 装饰器
from unittest import expectedFailure as xfail, skipIf

# 导入 pytest 模块中的 raises 函数，并重命名为 assert_raises
from pytest import raises as assert_raises

# 定义 skip 函数，部分应用 skipIf 装饰器
skip = functools.partial(skipIf, True)

# 导入 torch.testing._internal.common_utils 模块中的各个函数和变量
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    slowTest as slow,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 如果 TEST_WITH_TORCHDYNAMO 为真，则导入 numpy 相关函数和变量
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import histogram, histogram_bin_edges, histogramdd
    from numpy.testing import (
        assert_,  # 导入 assert_ 函数
        assert_allclose,  # 导入 assert_allclose 函数
        assert_almost_equal,  # 导入 assert_almost_equal 函数
        assert_array_almost_equal,  # 导入 assert_array_almost_equal 函数
        assert_array_equal,  # 导入 assert_array_equal 函数
        assert_equal,  # 导入 assert_equal 函数
        # assert_array_max_ulp, #assert_raises_regex, suppress_warnings,
    )
else:
    # 否则，导入 torch._numpy 中的相应函数和变量
    import torch._numpy as np
    from torch._numpy import histogram, histogramdd
    from torch._numpy.testing import (
        assert_,  # 导入 assert_ 函数
        assert_allclose,  # 导入 assert_allclose 函数
        assert_almost_equal,  # 导入 assert_almost_equal 函数
        assert_array_almost_equal,  # 导入 assert_array_almost_equal 函数
        assert_array_equal,  # 导入 assert_array_equal 函数
        assert_equal,  # 导入 assert_equal 函数
        # assert_array_max_ulp, #assert_raises_regex, suppress_warnings,
    )

# 定义一个测试类 TestHistogram，继承自 TestCase 类
class TestHistogram(TestCase):
    
    # 测试方法 test_simple
    def test_simple(self):
        # 设置样本数 n 为 100
        n = 100
        # 生成 n 个随机数并赋值给 v
        v = np.random.rand(n)
        # 调用 numpy 的 histogram 函数，得到直方图数组 a 和边界数组 b
        (a, b) = histogram(v)
        # 断言直方图数组 a 按列求和应等于样本数 n
        assert_equal(np.sum(a, axis=0), n)
        # 调用 numpy 的 histogram 函数，对线性函数生成的数据进行直方图计算
        (a, b) = histogram(np.linspace(0, 10, 100))
        # 断言直方图数组 a 应等于长度为 10 的数组
        assert_array_equal(a, 10)
    
    # 测试方法 test_one_bin
    def test_one_bin(self):
        # Ticket 632
        # 调用 histogram 函数，计算给定数据和边界的直方图
        hist, edges = histogram([1, 2, 3, 4], [1, 2])
        # 断言直方图 hist 应与预期数组 [2] 相等
        assert_array_equal(
            hist,
            [
                2,
            ],
        )
        # 断言边界数组 edges 应与预期数组 [1, 2] 相等
        assert_array_equal(edges, [1, 2])
        # 断言调用 histogram 函数时应抛出 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), histogram, [1, 2], bins=0)
        # 调用 histogram 函数，计算给定数据和 bins=1 时的直方图
        h, e = histogram([1, 2], bins=1)
        # 断言直方图 h 应与预期数组 [2] 相等
        assert_equal(h, np.array([2]))
        # 断言边界数组 e 应与预期数组 [1.0, 2.0] 相近
        assert_allclose(e, np.array([1.0, 2.0]))
    def test_density(self):
        # Check that the integral of the density equals 1.
        # 生成随机数向量v，长度为100
        n = 100
        v = np.random.rand(n)
        # 计算v的直方图a和bin的边界b，并且进行密度归一化
        a, b = histogram(v, density=True)
        # 计算归一化后的面积，应该接近1
        area = np.sum(a * np.diff(b))
        # 断言面积应该近似等于1
        assert_almost_equal(area, 1)

        # Check with non-constant bin widths
        # 使用非常量的bin宽度进行检查
        v = np.arange(10)
        bins = [0, 1, 3, 6, 10]
        a, b = histogram(v, bins, density=True)
        # 断言归一化后的直方图应该近似等于0.1
        assert_almost_equal(a, 0.1)
        # 断言归一化后的面积应该等于1
        assert_equal(np.sum(a * np.diff(b)), 1)

        # Test that passing False works too
        # 测试传递False也可以正常工作
        a, b = histogram(v, bins, density=False)
        # 断言直方图a应该等于[1, 2, 3, 4]
        assert_array_equal(a, [1, 2, 3, 4])

        # Variable bin widths are especially useful to deal with
        # infinities.
        # 变宽的bin对处理无穷大特别有用
        v = np.arange(10)
        bins = [0, 1, 3, 6, np.inf]
        a, b = histogram(v, bins, density=True)
        # 断言归一化后的直方图a应该近似等于[0.1, 0.1, 0.1, 0.0]
        assert_almost_equal(a, [0.1, 0.1, 0.1, 0.0])

        # Taken from a bug report from N. Becker on the numpy-discussion
        # mailing list Aug. 6, 2010.
        # 从N. Becker在numpy-discussion邮件列表上的bug报告中提取
        # 测试传递density=True时的直方图
        counts, dmy = np.histogram([1, 2, 3, 4], [0.5, 1.5, np.inf], density=True)
        # 断言直方图counts应该等于[0.25, 0]
        assert_equal(counts, [0.25, 0])

    def test_outliers(self):
        # Check that outliers are not tallied
        # 检查是否计数了异常值
        a = np.arange(10) + 0.5

        # Lower outliers
        # 检查低异常值
        h, b = histogram(a, range=[0, 9])
        # 断言直方图h的总和应该等于9
        assert_equal(h.sum(), 9)

        # Upper outliers
        # 检查高异常值
        h, b = histogram(a, range=[1, 10])
        # 断言直方图h的总和应该等于9
        assert_equal(h.sum(), 9)

        # Normalization
        # 归一化检查
        h, b = histogram(a, range=[1, 9], density=True)
        # 断言归一化后的直方图*h的面积应该接近等于1
        assert_almost_equal((h * np.diff(b)).sum(), 1, decimal=15)

        # Weights
        # 权重检查
        w = np.arange(10) + 0.5
        h, b = histogram(a, range=[1, 9], weights=w, density=True)
        # 断言加权后的直方图*h的面积应该等于1
        assert_equal((h * np.diff(b)).sum(), 1)

        h, b = histogram(a, bins=8, range=[1, 9], weights=w)
        # 断言直方图h应该等于w的中间部分
        assert_equal(h, w[1:-1])

    def test_arr_weights_mismatch(self):
        a = np.arange(10) + 0.5
        w = np.arange(11) + 0.5
        # 测试当权重数组和数据数组的形状不匹配时是否抛出异常
        with assert_raises((RuntimeError, ValueError)):
            h, b = histogram(a, range=[1, 9], weights=w, density=True)

    def test_type(self):
        # Check the type of the returned histogram
        # 检查返回的直方图的类型
        a = np.arange(10) + 0.5
        h, b = histogram(a)
        # 断言直方图h的dtype应该是整数类型的子类型
        assert_(np.issubdtype(h.dtype, np.integer))

        h, b = histogram(a, density=True)
        # 断言直方图h的dtype应该是浮点数类型的子类型
        assert_(np.issubdtype(h.dtype, np.floating))

        h, b = histogram(a, weights=np.ones(10, int))
        # 断言直方图h的dtype应该是整数类型的子类型
        assert_(np.issubdtype(h.dtype, np.integer))

        h, b = histogram(a, weights=np.ones(10, float))
        # 断言直方图h的dtype应该是浮点数类型的子类型
        assert_(np.issubdtype(h.dtype, np.floating))

    def test_f32_rounding(self):
        # gh-4799, check that the rounding of the edges works with float32
        # 检查float32下边界的四舍五入是否工作
        x = np.array([276.318359, -69.593948, 21.329449], dtype=np.float32)
        y = np.array([5005.689453, 4481.327637, 6010.369629], dtype=np.float32)
        counts_hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        # 断言counts_hist的总和应该等于3.0
        assert_equal(counts_hist.sum(), 3.0)
    def test_bool_conversion(self):
        # 用于测试布尔值转换的函数
        # gh-12107
        # 引用整数直方图
        a = np.array([1, 1, 0], dtype=np.uint8)
        # 计算整数直方图及其边界
        int_hist, int_edges = np.histogram(a)

        # 应该在布尔值上引发警告
        # 确保直方图等效，需要抑制警告以获取实际输出
        #      with suppress_warnings() as sup:
        #          rec = sup.record(RuntimeWarning, 'Converting input from .*')
        hist, edges = np.histogram([True, True, False])
        # 应该发出警告
        #            assert_equal(len(rec), 1)
        assert_array_equal(hist, int_hist)
        assert_array_equal(edges, int_edges)

    def test_weights(self):
        # 随机生成长度为100的数组
        v = np.random.rand(100)
        # 权重数组，所有元素为5
        w = np.ones(100) * 5
        # 计算直方图及其边界
        a, b = histogram(v)
        # 计算带密度参数的直方图
        na, nb = histogram(v, density=True)
        # 计算带权重的直方图及其边界
        wa, wb = histogram(v, weights=w)
        # 计算带权重和密度参数的直方图及其边界
        nwa, nwb = histogram(v, weights=w, density=True)
        # 检查加权直方图是否正确
        assert_array_almost_equal(a * 5, wa)
        assert_array_almost_equal(na, nwa)

        # 检查权重是否正确应用
        v = np.linspace(0, 10, 10)
        w = np.concatenate((np.zeros(5), np.ones(5)))
        # 计算带权重的直方图及其边界
        wa, wb = histogram(v, bins=np.arange(11), weights=w)
        assert_array_almost_equal(wa, w)

        # 检查使用整数权重
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1])
        assert_array_equal(wa, [4, 5, 0, 1])
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1], density=True)
        assert_array_almost_equal(wa, np.array([4, 5, 0, 1]) / 10.0 / 3.0 * 4)

        # 检查非均匀箱宽度的权重
        a, b = histogram(
            np.arange(9),
            [0, 1, 3, 6, 10],
            weights=[2, 1, 1, 1, 1, 1, 1, 1, 1],
            density=True,
        )
        assert_almost_equal(a, [0.2, 0.1, 0.1, 0.075])

    @xpassIfTorchDynamo  # (reason="histogram complex weights")
    def test_exotic_weights(self):
        # 测试使用不是整数或浮点数的权重，例如复数或对象类型。

        # 复数权重
        values = np.array([1.3, 2.5, 2.3])
        weights = np.array([1, -1, 2]) + 1j * np.array([2, 1, 2])

        # 使用自定义的箱子进行检查
        wa, wb = histogram(values, bins=[0, 2, 3], weights=weights)
        assert_array_almost_equal(wa, np.array([1, 1]) + 1j * np.array([2, 3]))

        # 使用均匀的箱子进行检查
        wa, wb = histogram(values, bins=2, range=[1, 3], weights=weights)
        assert_array_almost_equal(wa, np.array([1, 1]) + 1j * np.array([2, 3]))

        # 十进制权重
        from decimal import Decimal

        values = np.array([1.3, 2.5, 2.3])
        weights = np.array([Decimal(1), Decimal(2), Decimal(3)])

        # 使用自定义的箱子进行检查
        wa, wb = histogram(values, bins=[0, 2, 3], weights=weights)
        assert_array_almost_equal(wa, [Decimal(1), Decimal(5)])

        # 使用均匀的箱子进行检查
        wa, wb = histogram(values, bins=2, range=[1, 3], weights=weights)
        assert_array_almost_equal(wa, [Decimal(1), Decimal(5)])

    def test_no_side_effects(self):
        # 这是一个回归测试，确保传递给 ``histogram`` 的值没有被修改。
        values = np.array([1.3, 2.5, 2.3])
        np.histogram(values, range=[-10, 10], bins=100)
        assert_array_almost_equal(values, [1.3, 2.5, 2.3])

    def test_empty(self):
        # 测试对空数组的处理
        a, b = histogram([], bins=([0, 1]))
        assert_array_equal(a, np.array([0]))
        assert_array_equal(b, np.array([0, 1]))

    def test_error_binnum_type(self):
        # 测试当 bins 参数为浮点数时是否引发正确的错误
        vals = np.linspace(0.0, 1.0, num=100)
        histogram(vals, 5)
        assert_raises(TypeError, histogram, vals, 2.4)

    def test_finite_range(self):
        # 正常范围应该是可以接受的
        vals = np.linspace(0.0, 1.0, num=100)
        histogram(vals, range=[0.25, 0.75])
        assert_raises((RuntimeError, ValueError), histogram, vals, range=[np.nan, 0.75])
        assert_raises((RuntimeError, ValueError), histogram, vals, range=[0.25, np.inf])

    def test_invalid_range(self):
        # 范围的起始值必须小于结束值
        vals = np.linspace(0.0, 1.0, num=100)
        with assert_raises((RuntimeError, ValueError)):
            np.histogram(vals, range=[0.1, 0.01])

    @xfail  # (reason="edge cases")
    def test_bin_edge_cases(self):
        # 确保浮点运算正确处理边缘情况。
        arr = np.array([337, 404, 739, 806, 1007, 1811, 2012])
        hist, edges = np.histogram(arr, bins=8296, range=(2, 2280))
        mask = hist > 0
        left_edges = edges[:-1][mask]
        right_edges = edges[1:][mask]
        for x, left, right in zip(arr, left_edges, right_edges):
            assert_(x >= left)
            assert_(x < right, msg=f"{x}, {right}")
    def test_last_bin_inclusive_range(self):
        # 创建一个包含特定数据的 NumPy 数组
        arr = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        # 计算数组的直方图和边界
        hist, edges = np.histogram(arr, bins=30, range=(-0.5, 5))
        # 断言最后一个直方图 bin 的值是否为 1
        assert_equal(hist[-1], 1)

    def test_bin_array_dims(self):
        # 处理 bins 对象维度大于1的情况
        vals = np.linspace(0.0, 1.0, num=100)
        bins = np.array([[0, 0.5], [0.6, 1.0]])
        # 使用断言检测是否抛出预期异常
        with assert_raises((RuntimeError, ValueError)):
            np.histogram(vals, bins=bins)

    @xpassIfTorchDynamo  # (reason="no uint64")
    def test_unsigned_monotonicity_check(self):
        # 如果 bins 中包含无符号值并且未按单调递增顺序排列，确保抛出 ValueError
        arr = np.array([2])
        bins = np.array([1, 3, 1], dtype="uint64")
        # 使用断言检测是否抛出预期异常
        with assert_raises((RuntimeError, ValueError)):
            hist, edges = np.histogram(arr, bins=bins)

    def test_object_array_of_0d(self):
        # gh-7864
        # 检测传入包含无效值的情况是否会引发异常
        assert_raises(
            (RuntimeError, ValueError),
            histogram,
            [np.array(0.4) for i in range(10)] + [-np.inf],
        )
        assert_raises(
            (RuntimeError, ValueError),
            histogram,
            [np.array(0.4) for i in range(10)] + [np.inf],
        )

        # 这些情况不应该导致崩溃
        np.histogram([np.array(0.5) for i in range(10)] + [0.500000000000001])
        np.histogram([np.array(0.5) for i in range(10)] + [0.5])

    @xpassIfTorchDynamo  # (reason="bins='auto'")
    def test_some_nan_values(self):
        # gh-7503
        one_nan = np.array([0, 1, np.nan])
        all_nan = np.array([np.nan, np.nan])

        # 内部与 NaN 的比较会产生警告
        #      sup = suppress_warnings()
        #      sup.filter(RuntimeWarning)
        #      with sup:
        # 无法通过 nan 推断范围
        assert_raises(ValueError, histogram, one_nan, bins="auto")
        assert_raises(ValueError, histogram, all_nan, bins="auto")

        # 显式指定范围解决问题
        h, b = histogram(one_nan, bins="auto", range=(0, 1))
        assert_equal(h.sum(), 2)  # 不计算 NaN
        h, b = histogram(all_nan, bins="auto", range=(0, 1))
        assert_equal(h.sum(), 0)  # 不计算 NaN

        # 显式指定 bins 也可以解决问题
        h, b = histogram(one_nan, bins=[0, 1])
        assert_equal(h.sum(), 2)  # 不计算 NaN
        h, b = histogram(all_nan, bins=[0, 1])
        assert_equal(h.sum(), 0)  # 不计算 NaN

    def do_signed_overflow_bounds(self, dtype):
        # 计算指定数据类型的溢出边界
        exponent = 8 * np.dtype(dtype).itemsize - 1
        arr = np.array([-(2**exponent) + 4, 2**exponent - 4], dtype=dtype)
        # 计算数组的直方图和边界
        hist, e = histogram(arr, bins=2)
        # 断言计算的直方图和边界是否符合预期
        assert_equal(e, [-(2**exponent) + 4, 0, 2**exponent - 4])
        assert_equal(hist, [1, 1])
    def test_signed_overflow_bounds(self):
        # 调用 self.do_signed_overflow_bounds 方法，测试 np.byte 类型的数值范围溢出边界情况
        self.do_signed_overflow_bounds(np.byte)
        # 调用 self.do_signed_overflow_bounds 方法，测试 np.short 类型的数值范围溢出边界情况
        self.do_signed_overflow_bounds(np.short)
        # 调用 self.do_signed_overflow_bounds 方法，测试 np.intc 类型的数值范围溢出边界情况
        self.do_signed_overflow_bounds(np.intc)

    @xfail  # (reason="int->float conversin loses precision")
    def test_signed_overflow_bounds_2(self):
        # 标记为预期失败，因为从 int 转换为 float 时会丢失精度
        self.do_signed_overflow_bounds(np.int_)
        # 标记为预期失败，因为从 longlong 转换为 float 时会丢失精度
        self.do_signed_overflow_bounds(np.longlong)

    def do_precision_lower_bound(self, float_small, float_large):
        # 获取 float_large 类型的机器精度
        eps = np.finfo(float_large).eps

        # 创建一个包含单个元素 1.0 的 float_small 类型的数组 arr
        arr = np.array([1.0], float_small)
        # 创建一个包含两个元素的 float_large 类型的数组 range，第一个元素为 1.0 + eps，第二个元素为 2.0
        range = np.array([1.0 + eps, 2.0], float_large)

        # 如果将 range 转换为 float_small 类型后第一个元素不等于 1，则返回
        if range.astype(float_small)[0] != 1:
            return

        # 使用 arr 和 range 数组进行直方图统计，bins 设置为 1，range 设置为 range
        count, x_loc = np.histogram(arr, bins=1, range=range)
        # 断言 count 数组的值为 [1]
        assert_equal(count, [1])

        # 断言 x_loc 数组的数据类型为 float_small
        assert_equal(x_loc.dtype, float_small)

    def do_precision_upper_bound(self, float_small, float_large):
        # 获取 float_large 类型的机器精度
        eps = np.finfo(float_large).eps

        # 创建一个包含单个元素 1.0 的 float_small 类型的数组 arr
        arr = np.array([1.0], float_small)
        # 创建一个包含两个元素的 float_large 类型的数组 range，第一个元素为 0.0，第二个元素为 1.0 - eps
        range = np.array([0.0, 1.0 - eps], float_large)

        # 如果将 range 转换为 float_small 类型后最后一个元素不等于 1，则返回
        if range.astype(float_small)[-1] != 1:
            return

        # 使用 arr 和 range 数组进行直方图统计，bins 设置为 1，range 设置为 range
        count, x_loc = np.histogram(arr, bins=1, range=range)
        # 断言 count 数组的值为 [1]
        assert_equal(count, [1])

        # 断言 x_loc 数组的数据类型为 float_small
        assert_equal(x_loc.dtype, float_small)

    def do_precision(self, float_small, float_large):
        # 调用 do_precision_lower_bound 方法和 do_precision_upper_bound 方法，分别测试下限和上限的精度情况
        self.do_precision_lower_bound(float_small, float_large)
        self.do_precision_upper_bound(float_small, float_large)

    @xpassIfTorchDynamo  # (reason="mixed dtypes")
    def test_precision(self):
        # 不循环测试，以便在失败时获得有用的堆栈跟踪
        # 测试 np.half 和 np.single 类型的精度情况
        self.do_precision(np.half, np.single)
        # 测试 np.half 和 np.double 类型的精度情况
        self.do_precision(np.half, np.double)
        # 测试 np.single 和 np.double 类型的精度情况
        self.do_precision(np.single, np.double)

    @xpassIfTorchDynamo  # (reason="pytorch does not support bins = [int, int, array]")
    @slow
    def test_histogram_bin_edges(self):
        # 使用给定的数据和 bins 进行直方图计算，比较返回的边缘数组是否接近
        hist, e = histogram([1, 2, 3, 4], [1, 2])
        edges = histogram_bin_edges([1, 2, 3, 4], [1, 2])
        assert_allclose(edges, e, atol=2e-15)

        # 对给定数组 arr 进行直方图计算，使用 bins=30，range=(-0.5, 5)，比较返回的边缘数组是否接近
        arr = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        hist, e = histogram(arr, bins=30, range=(-0.5, 5))
        edges = histogram_bin_edges(arr, bins=30, range=(-0.5, 5))
        assert_allclose(edges, e, atol=2e-15)

        # 对给定数组 arr 进行直方图计算，使用 bins="auto"，range=(0, 1)，比较返回的边缘数组是否接近
        hist, e = histogram(arr, bins="auto", range=(0, 1))
        edges = histogram_bin_edges(arr, bins="auto", range=(0, 1))
        assert_allclose(edges, e, atol=2e-15)
    # 定义一个测试方法，用于测试处理大型数组的情况
    def test_big_arrays(self):
        # 创建一个形状为 [100000000, 3] 的全零 NumPy 数组作为样本
        sample = np.zeros([100000000, 3])
        # 定义 x 轴的直方图箱数
        xbins = 400
        # 定义 y 轴的直方图箱数
        ybins = 400
        # 创建一个包含 0 到 15999 的 NumPy 数组作为 z 轴的直方图箱数
        zbins = np.arange(16000)
        # 使用 np.histogramdd 函数计算多维直方图，返回直方图数据和箱子边界
        hist = np.histogramdd(sample=sample, bins=(xbins, ybins, zbins))
        # 断言检查 hist 的类型是否与元组相同
        assert_equal(type(hist), type((1, 2)))
# 标记为装饰器，用于跳过 Torch Dynamo 测试（原因是"TODO"）
@xpassIfTorchDynamo  # (reason="TODO")

# 标记为装饰器，用于实例化参数化测试
@instantiate_parametrized_tests

# 定义一个测试类 TestHistogramOptimBinNums，继承自 TestCase 类
class TestHistogramOptimBinNums(TestCase):
    """
    提供使用提供的估计器来优化箱数时的测试覆盖率
    """

    # 定义一个测试方法 test_empty，测试处理空数据的情况
    def test_empty(self):
        # 定义估计器列表
        estimator_list = [
            "fd",
            "scott",
            "rice",
            "sturges",
            "doane",
            "sqrt",
            "auto",
            "stone",
        ]
        # 遍历估计器列表
        for estimator in estimator_list:
            # 调用 histogram 函数，处理空数据，返回结果为 a, b
            a, b = histogram([], bins=estimator)
            # 断言数组 a 的内容等于 [0]
            assert_array_equal(a, np.array([0]))
            # 断言数组 b 的内容等于 [0, 1]
            assert_array_equal(b, np.array([0, 1]))

    # 定义一个测试方法 test_simple
    def test_simple(self):
        """
        使用线性空间数据的简单测试。所有测试值都已预先计算，不应更改其值
        """
        # 一些基本的健全性检查，使用一些固定数据
        # 检查正确的箱数
        basic_test = {
            50: {
                "fd": 4,
                "scott": 4,
                "rice": 8,
                "sturges": 7,
                "doane": 8,
                "sqrt": 8,
                "auto": 7,
                "stone": 2,
            },
            500: {
                "fd": 8,
                "scott": 8,
                "rice": 16,
                "sturges": 10,
                "doane": 12,
                "sqrt": 23,
                "auto": 10,
                "stone": 9,
            },
            5000: {
                "fd": 17,
                "scott": 17,
                "rice": 35,
                "sturges": 14,
                "doane": 17,
                "sqrt": 71,
                "auto": 17,
                "stone": 20,
            },
        }

        # 遍历 basic_test 中的每个测试长度 testlen 和其对应的期望结果 expectedResults
        for testlen, expectedResults in basic_test.items():
            # 创建某种非均匀数据以进行测试（2个峰值的均匀混合）
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x = np.concatenate((x1, x2))
            # 遍历 expectedResults 中的每个估计器 estimator 和其对应的箱数 numbins
            for estimator, numbins in expectedResults.items():
                # 调用 np.histogram 函数，处理数据 x 和估计器 estimator，返回结果为 a, b
                a, b = np.histogram(x, estimator)
                # 断言数组 a 的长度等于 numbins
                assert_equal(
                    len(a),
                    numbins,
                    err_msg=f"For the {estimator} estimator "
                    f"with datasize of {testlen}",
                )
    def test_small(self):
        """
        Smaller datasets have the potential to cause issues with the data
        adaptive methods, especially the FD method. All bin numbers have been
        precalculated.
        """
        # 定义小数据集及其对应的预期结果字典
        small_dat = {
            1: {
                "fd": 1,
                "scott": 1,
                "rice": 1,
                "sturges": 1,
                "doane": 1,
                "sqrt": 1,
                "stone": 1,
            },
            2: {
                "fd": 2,
                "scott": 1,
                "rice": 3,
                "sturges": 2,
                "doane": 1,
                "sqrt": 2,
                "stone": 1,
            },
            3: {
                "fd": 2,
                "scott": 2,
                "rice": 3,
                "sturges": 3,
                "doane": 3,
                "sqrt": 2,
                "stone": 1,
            },
        }

        # 遍历每个数据集大小及其预期结果
        for testlen, expectedResults in small_dat.items():
            # 创建测试数据集
            testdat = np.arange(testlen)
            # 遍历每种估算方法及其预期结果
            for estimator, expbins in expectedResults.items():
                # 计算直方图
                a, b = np.histogram(testdat, estimator)
                # 断言直方图的长度符合预期结果
                assert_equal(
                    len(a),
                    expbins,
                    err_msg=f"For the {estimator} estimator "
                    f"with datasize of {testlen}",
                )

    def test_incorrect_methods(self):
        """
        Check a Value Error is thrown when an unknown string is passed in
        """
        # 定义待检查的估算方法列表
        check_list = ["mad", "freeman", "histograms", "IQR"]
        # 遍历每种估算方法，预期会抛出 ValueError
        for estimator in check_list:
            # 断言调用直方图函数时，传入未知的估算方法会抛出 ValueError
            assert_raises(ValueError, histogram, [1, 2, 3], estimator)

    def test_novariance(self):
        """
        Check that methods handle no variance in data
        Primarily for Scott and FD as the SD and IQR are both 0 in this case
        """
        # 创建无方差数据集
        novar_dataset = np.ones(100)
        # 定义无方差情况下各估算方法的预期结果字典
        novar_resultdict = {
            "fd": 1,
            "scott": 1,
            "rice": 1,
            "sturges": 1,
            "doane": 1,
            "sqrt": 1,
            "auto": 1,
            "stone": 1,
        }

        # 遍历每种估算方法及其预期结果
        for estimator, numbins in novar_resultdict.items():
            # 计算无方差数据集的直方图
            a, b = np.histogram(novar_dataset, estimator)
            # 断言直方图的长度符合预期结果
            assert_equal(
                len(a),
                numbins,
                err_msg=f"{estimator} estimator, No Variance test",
            )
    def test_limited_variance(self):
        """
        Check when IQR is 0, but variance exists, we return the sturges value
        and not the fd value.
        """
        # 创建一个包含1000个元素的数组，大部分为1，前3个和最后4个元素分别设置为0和100
        lim_var_data = np.ones(1000)
        lim_var_data[:3] = 0
        lim_var_data[-4:] = 100

        # 使用 "auto" 参数调用直方图边界生成函数，期望边界与 np.linspace(0, 100, 12) 相似
        edges_auto = histogram_bin_edges(lim_var_data, "auto")
        assert_allclose(edges_auto, np.linspace(0, 100, 12), rtol=1e-15)

        # 使用 "fd" 参数调用直方图边界生成函数，期望边界为 [0, 100]
        edges_fd = histogram_bin_edges(lim_var_data, "fd")
        assert_allclose(edges_fd, np.array([0, 100]), rtol=1e-15)

        # 使用 "sturges" 参数调用直方图边界生成函数，期望边界与 np.linspace(0, 100, 12) 相似
        edges_sturges = histogram_bin_edges(lim_var_data, "sturges")
        assert_allclose(edges_sturges, np.linspace(0, 100, 12), rtol=1e-15)

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
        # 创建一个包含离群值的数据集，在给定范围内均匀分布
        xcenter = np.linspace(-10, 10, 50)
        outlier_dataset = np.hstack((np.linspace(-110, -100, 5), xcenter))

        # 预期每种估算器返回的直方图条数
        outlier_resultdict = {"fd": 21, "scott": 5, "doane": 11, "stone": 6}

        # 遍历每个估算器并检查返回的直方图条数是否与预期相符
        for estimator, numbins in outlier_resultdict.items():
            a, b = np.histogram(outlier_dataset, estimator)
            assert_equal(len(a), numbins)

    def test_scott_vs_stone(self):
        """Verify that Scott's rule and Stone's rule converges for normally distributed data"""

        def nbins_ratio(seed, size):
            # 使用指定种子和大小生成正态分布数据
            rng = np.random.RandomState(seed)
            x = rng.normal(loc=0, scale=2, size=size)
            # 分别使用 "stone" 和 "scott" 估算器计算直方图，并返回条数比例
            a, b = len(np.histogram(x, "stone")[0]), len(np.histogram(x, "scott")[0])
            return a / (a + b)

        # 为不同的种子和数据集大小计算 "stone" 和 "scott" 估算器的条数比例
        ll = [
            [
                nbins_ratio(seed, size)
                for size in np.geomspace(start=10, stop=100, num=4).round().astype(int)
            ]
            for seed in range(10)
        ]

        # 验证两种方法的平均差异随数据集大小增加而减小
        avg = abs(np.mean(ll, axis=0) - 0.5)
        assert_almost_equal(avg, [0.15, 0.09, 0.08, 0.03], decimal=2)
    def test_simple_range(self):
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). Adding in a 3rd mixture that will then be
        completely ignored. All test values have been precomputed and
        the shouldn't change.
        """
        # some basic sanity checking, with some fixed data.
        # Checking for the correct number of bins
        basic_test = {
            50: {
                "fd": 8,
                "scott": 8,
                "rice": 15,
                "sturges": 14,
                "auto": 14,
                "stone": 8,
            },
            500: {
                "fd": 15,
                "scott": 16,
                "rice": 32,
                "sturges": 20,
                "auto": 20,
                "stone": 80,
            },
            5000: {
                "fd": 33,
                "scott": 33,
                "rice": 69,
                "sturges": 27,
                "auto": 33,
                "stone": 80,
            },
        }

        # Iterate through different test lengths and expected results
        for testlen, expectedResults in basic_test.items():
            # create some sort of non uniform data to test with
            # (3 peak uniform mixture)
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x3 = np.linspace(-100, -50, testlen)
            x = np.hstack((x1, x2, x3))
            # Iterate through different estimators and expected number of bins
            for estimator, numbins in expectedResults.items():
                # Compute histogram 'a' and bins 'b' for data 'x' and given estimator
                a, b = np.histogram(x, estimator, range=(-20, 20))
                # Formulate error message based on estimator and test data size
                msg = f"For the {estimator} estimator"
                msg += f" with datasize of {testlen}"
                # Assert that the length of histogram 'a' matches expected number of bins
                assert_equal(len(a), numbins, err_msg=msg)

    @parametrize("bins", ["auto", "fd", "doane", "scott", "stone", "rice", "sturges"])
    def test_signed_integer_data(self, bins):
        # Regression test for gh-14379.
        # Define array 'a' with signed integers for testing
        a = np.array([-2, 0, 127], dtype=np.int8)
        # Compute histogram 'hist' and edges 'edges' using specified bins
        hist, edges = np.histogram(a, bins=bins)
        # Compute histogram 'hist32' and edges 'edges32' for 'a' converted to int32
        hist32, edges32 = np.histogram(a.astype(np.int32), bins=bins)
        # Assert that both histograms and edges are equal
        assert_array_equal(hist, hist32)
        assert_array_equal(edges, edges32)

    def test_simple_weighted(self):
        """
        Check that weighted data raises a TypeError
        """
        # List of estimators to test
        estimator_list = ["fd", "scott", "rice", "sturges", "auto"]
        # Iterate through each estimator
        for estimator in estimator_list:
            # Assert that calling histogram with weighted data raises a TypeError
            assert_raises(TypeError, histogram, [1, 2, 3], estimator, weights=[1, 2, 3])
class TestHistogramdd(TestCase):
    # 定义测试类 TestHistogramdd，继承自 TestCase 类

    def test_simple(self):
        # 定义测试方法 test_simple

        x = np.array(
            [
                [-0.5, 0.5, 1.5],
                [-0.5, 1.5, 2.5],
                [-0.5, 2.5, 0.5],
                [0.5, 0.5, 1.5],
                [0.5, 1.5, 2.5],
                [0.5, 2.5, 2.5],
            ]
        )
        # 创建 NumPy 数组 x，包含多个三维坐标点

        H, edges = histogramdd(x, (2, 3, 3), range=[[-1, 1], [0, 3], [0, 3]])
        # 调用 histogramdd 函数计算 x 数据的多维直方图 H 和边界 edges

        answer = np.array(
            [[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1], [0, 0, 1]]]
        )
        # 创建预期的直方图结果 answer，与 H 进行比较

        assert_array_equal(H, answer)
        # 使用 assert_array_equal 检查 H 是否与 answer 相等

        # Check normalization
        ed = [[-2, 0, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
        # 定义边界列表 ed

        H, edges = histogramdd(x, bins=ed, density=True)
        # 使用 density=True 参数计算 x 数据的归一化直方图 H 和边界 edges

        assert_(np.all(H == answer / 12.0))
        # 使用 assert_ 检查 H 是否等于 answer 除以 12.0

        # Check that H has the correct shape.
        H, edges = histogramdd(
            x, (2, 3, 4), range=[[-1, 1], [0, 3], [0, 4]], density=True
        )
        # 使用 density=True 参数计算 x 数据的归一化直方图 H 和边界 edges，并指定形状参数

        answer = np.array(
            [
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
                [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
            ]
        )
        # 更新预期的直方图结果 answer，与 H 进行比较

        assert_array_almost_equal(H, answer / 6.0, 4)
        # 使用 assert_array_almost_equal 检查 H 是否接近于 answer 除以 6.0，精确到小数点后四位

        # Check that a sequence of arrays is accepted and H has the correct
        # shape.
        z = [np.squeeze(y) for y in np.split(x, 3, axis=1)]
        # 生成 z 序列，包含 x 数据的三个切片数组，去除多余的维度

        H, edges = histogramdd(z, bins=(4, 3, 2), range=[[-2, 2], [0, 3], [0, 2]])
        # 计算 z 数据的多维直方图 H 和边界 edges，指定形状和范围参数

        answer = np.array(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 1], [0, 0], [1, 0]],
                [[0, 1], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
            ]
        )
        # 创建预期的直方图结果 answer，与 H 进行比较

        assert_array_equal(H, answer)
        # 使用 assert_array_equal 检查 H 是否与 answer 相等

        Z = np.zeros((5, 5, 5))
        Z[list(range(5)), list(range(5)), list(range(5))] = 1.0
        # 创建全零数组 Z，并将对角线元素设置为 1.0

        H, edges = histogramdd([np.arange(5), np.arange(5), np.arange(5)], 5)
        # 计算三个一维数组的多维直方图 H 和边界 edges，指定 bins=5

        assert_array_equal(H, Z)
        # 使用 assert_array_equal 检查 H 是否与 Z 相等

    def test_shape_3d(self):
        # 定义测试方法 test_shape_3d

        # All possible permutations for bins of different lengths in 3D.
        bins = ((5, 4, 6), (6, 4, 5), (5, 6, 4), (4, 6, 5), (6, 5, 4), (4, 5, 6))
        # 定义多个不同长度的 bins 元组

        r = np.random.rand(10, 3)
        # 创建一个随机的 10x3 的 NumPy 数组 r

        for b in bins:
            # 遍历 bins 中的每个元组 b

            H, edges = histogramdd(r, b)
            # 计算 r 数据的多维直方图 H 和边界 edges，指定 bins=b

            assert_(H.shape == b)
            # 使用 assert_ 检查 H 的形状是否等于 b
    def test_shape_4d(self):
        # 测试函数：test_shape_4d，用于验证在四维空间中不同长度的箱子的所有可能排列方式。

        # 定义不同长度箱子的所有排列方式
        bins = (
            (7, 4, 5, 6),
            (4, 5, 7, 6),
            (5, 6, 4, 7),
            (7, 6, 5, 4),
            (5, 7, 6, 4),
            (4, 6, 7, 5),
            (6, 5, 7, 4),
            (7, 5, 4, 6),
            (7, 4, 6, 5),
            (6, 4, 7, 5),
            (6, 7, 5, 4),
            (4, 6, 5, 7),
            (4, 7, 5, 6),
            (5, 4, 6, 7),
            (5, 7, 4, 6),
            (6, 7, 4, 5),
            (6, 5, 4, 7),
            (4, 7, 6, 5),
            (4, 5, 6, 7),
            (7, 6, 4, 5),
            (5, 4, 7, 6),
            (5, 6, 7, 4),
            (6, 4, 5, 7),
            (7, 5, 6, 4),
        )

        # 生成一个 10x4 的随机数组
        r = np.random.rand(10, 4)
        
        # 遍历每个箱子排列方式
        for b in bins:
            # 使用给定的箱子形状计算多维直方图
            H, edges = histogramdd(r, b)
            # 断言计算得到的直方图形状与箱子形状相同
            assert_(H.shape == b)

            # 重新计算直方图，使用权重为1的数组作为权重
            h1, e1 = histogramdd(r, b, weights=np.ones(10))
            # 断言两次计算得到的直方图数组完全相等
            assert_equal(H, h1)
            
            # 检查直方图的边界值是否完全一致
            for edge, e in zip(edges, e1):
                assert (edge == e).all()

    def test_weights(self):
        # 测试函数：test_weights，验证直方图计算中的权重参数功能

        # 生成一个 100x2 的随机数组
        v = np.random.rand(100, 2)
        
        # 计算未加权的直方图
        hist, edges = histogramdd(v)
        
        # 计算使用密度归一化的直方图
        n_hist, edges = histogramdd(v, density=True)
        
        # 计算使用权重为1的数组作为权重的直方图
        w_hist, edges = histogramdd(v, weights=np.ones(100))
        # 断言两个直方图数组完全相等
        assert_array_equal(w_hist, hist)
        
        # 计算使用权重为2的数组作为权重的密度归一化直方图
        w_hist, edges = histogramdd(v, weights=np.ones(100) * 2, density=True)
        # 断言两个直方图数组完全相等
        assert_array_equal(w_hist, n_hist)
        
        # 计算使用整数2的数组作为权重的直方图
        w_hist, edges = histogramdd(v, weights=np.ones(100, int) * 2)
        # 断言两个直方图数组完全相等
        assert_array_equal(w_hist, 2 * hist)

    def test_identical_samples(self):
        # 测试函数：test_identical_samples，验证对相同样本的直方图计算功能

        # 创建一个 10x2 的零数组
        x = np.zeros((10, 2), int)
        
        # 计算零数组的直方图，指定箱子数为2
        hist, edges = histogramdd(x, bins=2)
        
        # 断言直方图的第一个边界值数组与预期的值数组完全相等
        assert_array_equal(edges[0], np.array([-0.5, 0.0, 0.5]))

    def test_empty(self):
        # 测试函数：test_empty，验证对空数组的直方图计算功能

        # 计算空数组的直方图，指定每个维度的箱子数为[0, 1]和[0, 1]
        a, b = histogramdd([[], []], bins=([0, 1], [0, 1]))
        # 断言计算得到的直方图数组完全为零
        assert_allclose(a, np.array([[0.0]]), atol=1e-15)
        
        # 使用numpy的函数计算空三维数组的直方图，指定箱子数为2
        a, b = np.histogramdd([[], [], []], bins=2)
        # 断言计算得到的直方图数组完全为零
        assert_allclose(a, np.zeros((2, 2, 2)), atol=1e-15)

    def test_bins_errors(self):
        # 测试函数：test_bins_errors，验证在指定不正确的箱子参数时的错误处理

        # 创建一个 2x4 的数组
        x = np.arange(8).reshape(2, 4)
        
        # 断言调用直方图函数时会抛出 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.histogramdd, x, bins=[-1, 2, 4, 5])
        assert_raises(
            (RuntimeError, ValueError), np.histogramdd, x, bins=[1, 0.99, 1, 1]
        )
        assert_raises(
            (RuntimeError, ValueError), np.histogramdd, x, bins=[1, 1, 1, [1, 2, 3, -3]]
        )

    @xpassIfTorchDynamo  # (reason="pytorch does not support bins = [int, int, array]")
    def test_bins_error_2(self):
        # 测试函数：test_bins_error_2，验证在混合标量和显式箱子数组时的错误处理

        # 创建一个 2x4 的数组
        x = np.arange(8).reshape(2, 4)
        
        # 断言调用直方图函数时会抛出异常
        assert_(np.histogramdd(x, bins=[1, 1, 1, [1, 2, 3, 4]]))

    @xpassIfTorchDynamo  # (reason="pytorch does not support bins = [int, int, array]")
    def test_inf_edges(self):
        # 测试使用正负无穷作为边界是否有效。参见 GitHub 问题 #1788。
        x = np.arange(6).reshape(3, 2)
        expected = np.array([[1, 0], [0, 1], [0, 1]])
        # 计算多维直方图及其边界，其中第二维使用正负无穷作为边界
        h, e = np.histogramdd(x, bins=[3, [-np.inf, 2, 10]])
        assert_allclose(h, expected)
        # 重新计算多维直方图及其边界，第二维使用数组包含正无穷
        h, e = np.histogramdd(x, bins=[3, np.array([-1, 2, np.inf])])
        assert_allclose(h, expected)
        # 再次计算多维直方图及其边界，第二维使用正负无穷作为边界
        h, e = np.histogramdd(x, bins=[3, [-np.inf, 3, np.inf]])
        assert_allclose(h, expected)

    def test_rightmost_binedge(self):
        # 测试事件接近右侧最后一个边界的情况。参见 GitHub 问题 #4266
        x = [0.9999999995]
        bins = [[0.0, 0.5, 1.0]]
        # 计算多维直方图及其边界，验证接近右侧最后一个边界的情况
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 1.0)
        x = [1.0]
        bins = [[0.0, 0.5, 1.0]]
        # 再次计算多维直方图及其边界，验证右侧最后一个边界的情况
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 1.0)
        x = [1.0000000001]
        bins = [[0.0, 0.5, 1.0]]
        # 继续计算多维直方图及其边界，验证超过右侧最后一个边界的情况
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 0.0)
        x = [1.0001]
        bins = [[0.0, 0.5, 1.0]]
        # 又一次计算多维直方图及其边界，验证远超右侧最后一个边界的情况
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 0.0)

    def test_finite_range(self):
        vals = np.random.random((100, 3))
        # 计算给定范围内的多维直方图
        histogramdd(vals, range=[[0.0, 1.0], [0.25, 0.75], [0.25, 0.5]])
        # 预期在边界值设置不正确时抛出运行时错误或值错误异常
        assert_raises(
            (RuntimeError, ValueError),
            histogramdd,
            vals,
            range=[[0.0, 1.0], [0.25, 0.75], [0.25, np.inf]],
        )
        assert_raises(
            (RuntimeError, ValueError),
            histogramdd,
            vals,
            range=[[0.0, 1.0], [np.nan, 0.75], [0.25, 0.5]],
        )

    @xpassIfTorchDynamo  # (reason="pytorch does not allow equal entries")
    def test_equal_edges(self):
        """测试边界数组中相邻条目可以相等的情况"""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        x_edges = np.array([0, 2, 2])
        y_edges = 1
        # 计算多维直方图及其边界，其中边界数组中有相等的条目
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        hist_expected = np.array(
            [
                [2.0],
                [1.0],  # x == 2 落入最后一个 bin 中
            ]
        )
        assert_equal(hist, hist_expected)

    def test_edge_dtype(self):
        """测试如果输入了边界数组，则其类型是否保持不变"""
        x = np.array([0, 10, 20])
        y = x / 10
        x_edges = np.array([0, 5, 15, 20])
        y_edges = x_edges / 10
        # 计算多维直方图及其边界，验证边界数组的数据类型是否保持不变
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        assert_equal(edges[0].dtype, x_edges.dtype)
        assert_equal(edges[1].dtype, y_edges.dtype)
    def test_large_integers(self):
        # 定义一个超过完全精度浮点数表示范围的大整数
        big = 2**60  # Too large to represent with a full precision float

        # 创建一个包含单个元素的 int64 类型的 NumPy 数组
        x = np.asarray([0], dtype=np.int64)
        # 创建一个包含两个元素的 int64 类型的 NumPy 数组
        x_edges = np.array([-1, +1], np.int64)
        # 将大整数 big 加到 x 中的每个元素上
        y = big + x
        # 将大整数 big 加到 x_edges 中的每个元素上
        y_edges = big + x_edges

        # 使用 x 和 y 的数据创建多维直方图
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        # 断言直方图中特定位置的值为1
        assert_equal(hist[0, 0], 1)

    def test_density_non_uniform_2d(self):
        # 定义 x 和 y 的边界
        # 定义以下网格:
        #
        #    0 2     8
        #   0+-+-----+
        #    + |     +
        #    + |     +
        #   6+-+-----+
        #   8+-+-----+
        x_edges = np.array([0, 2, 8])
        y_edges = np.array([0, 6, 8])
        # 定义相对区域的面积
        relative_areas = np.array([[3, 9], [1, 3]])

        # 确保每个区域中的点数与其面积成比例
        x = np.array([1] + [1] * 3 + [7] * 3 + [7] * 9)
        y = np.array([7] + [1] * 3 + [7] * 3 + [1] * 9)

        # 对上述操作进行一致性检查
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges))
        assert_equal(hist, relative_areas)

        # 结果直方图应该是均匀的，因为计数和面积成比例
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges), density=True)
        assert_equal(hist, 1 / (8 * 8))

    def test_density_non_uniform_1d(self):
        # 将直方图与直方图dd进行比较以展示结果相同
        v = np.arange(10)
        bins = np.array([0, 1, 3, 6, 10])
        hist, edges = histogram(v, bins, density=True)
        hist_dd, edges_dd = histogramdd((v,), (bins,), density=True)
        assert_equal(hist, hist_dd)
        assert_equal(edges, edges_dd[0])

    def test_bins_array(self):
        x = np.array(
            [
                [-0.5, 0.5, 1.5],
                [-0.5, 1.5, 2.5],
                [-0.5, 2.5, 0.5],
                [0.5, 0.5, 1.5],
                [0.5, 1.5, 2.5],
                [0.5, 2.5, 2.5],
            ]
        )
        # 使用 x 和给定的 bin 数组创建多维直方图
        H, edges = histogramdd(x, (2, 3, 3))
        # 断言 edges 中的每个元素都是 NumPy 数组
        assert all(type(e) is np.ndarray for e in edges)
# 如果当前脚本作为主程序运行（而不是作为模块被导入执行），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```