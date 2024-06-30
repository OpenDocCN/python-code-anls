# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_fitpack2.py`

```
# 导入模块和函数，声明文件作者和创建日期
# Created by Pearu Peterson, June 2003
import itertools  # 导入itertools模块，用于迭代器操作
import numpy as np  # 导入NumPy库，并使用别名np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
        assert_array_almost_equal, assert_allclose, suppress_warnings)  # 从NumPy测试模块中导入多个断言函数
from pytest import raises as assert_raises  # 导入pytest模块的raises函数，并重命名为assert_raises

# 从NumPy库中导入一些函数和类
from numpy import array, diff, linspace, meshgrid, ones, pi, shape  
# 从SciPy插值模块中导入一些函数和类
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde  
# 从SciPy插值模块_fitpack2中导入多个类
from scipy.interpolate._fitpack2 import (UnivariateSpline, 
        LSQUnivariateSpline, InterpolatedUnivariateSpline,
        LSQBivariateSpline, SmoothBivariateSpline, RectBivariateSpline,
        LSQSphereBivariateSpline, SmoothSphereBivariateSpline,
        RectSphereBivariateSpline)

# 定义测试类TestUnivariateSpline
class TestUnivariateSpline:
    # 测试线性常数情况
    def test_linear_constant(self):
        x = [1,2,3]
        y = [3,3,3]
        # 创建UnivariateSpline对象lut，指定线性插值(k=1)
        lut = UnivariateSpline(x,y,k=1)
        # 断言节点位置数组与预期值接近
        assert_array_almost_equal(lut.get_knots(),[1,3])
        # 断言系数数组与预期值接近
        assert_array_almost_equal(lut.get_coeffs(),[3,3])
        # 断言残差接近0
        assert_almost_equal(lut.get_residual(),0.0)
        # 断言插值结果与预期值接近
        assert_array_almost_equal(lut([1,1.5,2]),[3,3,3])

    # 测试保持形状的情况
    def test_preserve_shape(self):
        x = [1, 2, 3]
        y = [0, 2, 4]
        # 创建UnivariateSpline对象lut，指定线性插值(k=1)
        lut = UnivariateSpline(x, y, k=1)
        arg = 2
        # 断言arg和lut(arg)具有相同的形状
        assert_equal(shape(arg), shape(lut(arg)))
        # 断言arg和lut(arg, nu=1)具有相同的形状
        assert_equal(shape(arg), shape(lut(arg, nu=1)))
        arg = [1.5, 2, 2.5]
        # 断言arg和lut(arg)具有相同的形状
        assert_equal(shape(arg), shape(lut(arg)))
        # 断言arg和lut(arg, nu=1)具有相同的形状
        assert_equal(shape(arg), shape(lut(arg, nu=1)))

    # 测试线性插值情况
    def test_linear_1d(self):
        x = [1,2,3]
        y = [0,2,4]
        # 创建UnivariateSpline对象lut，指定线性插值(k=1)
        lut = UnivariateSpline(x,y,k=1)
        # 断言节点位置数组与预期值接近
        assert_array_almost_equal(lut.get_knots(),[1,3])
        # 断言系数数组与预期值接近
        assert_array_almost_equal(lut.get_coeffs(),[0,4])
        # 断言残差接近0
        assert_almost_equal(lut.get_residual(),0.0)
        # 断言插值结果与预期值接近
        assert_array_almost_equal(lut([1,1.5,2]),[0,1,2])

    # 测试子类化情况
    def test_subclassing(self):
        # See #731
        # 定义一个名为ZeroSpline的UnivariateSpline子类
        class ZeroSpline(UnivariateSpline):
            def __call__(self, x):
                return 0*array(x)

        # 使用ZeroSpline类创建sp对象
        sp = ZeroSpline([1,2,3,4,5], [3,2,3,2,3], k=2)
        # 断言sp([1.5, 2.5])的结果与预期值相等
        assert_array_equal(sp([1.5, 2.5]), [0., 0.])

    # 测试空输入情况
    def test_empty_input(self):
        # Test whether empty input returns an empty output. Ticket 1014
        x = [1,3,5,7,9]
        y = [0,4,9,12,21]
        # 创建UnivariateSpline对象spl，三次样条插值(k=3)
        spl = UnivariateSpline(x, y, k=3)
        # 断言空输入的插值结果为一个空数组
        assert_array_equal(spl([]), array([]))

    # 测试根的计算情况
    def test_roots(self):
        x = [1, 3, 5, 7, 9]
        y = [0, 4, 9, 12, 21]
        # 创建UnivariateSpline对象spl，三次样条插值(k=3)
        spl = UnivariateSpline(x, y, k=3)
        # 断言根的第一个值接近于预期值
        assert_almost_equal(spl.roots()[0], 1.050290639101332)

    # 测试根的数量情况
    def test_roots_length(self): # for gh18335
        x = np.linspace(0, 50 * np.pi, 1000)
        y = np.cos(x)
        # 创建UnivariateSpline对象spl，设置平滑参数s=0
        spl = UnivariateSpline(x, y, s=0)
        # 断言根的数量等于50
        assert_equal(len(spl.roots()), 50)

    # 测试导数计算情况
    def test_derivatives(self):
        x = [1, 3, 5, 7, 9]
        y = [0, 4, 9, 12, 21]
        # 创建UnivariateSpline对象spl，三次样条插值(k=3)
        spl = UnivariateSpline(x, y, k=3)
        # 断言在点3.5处的多阶导数与预期值接近
        assert_almost_equal(spl.derivatives(3.5),
                            [5.5152902, 1.7146577, -0.1830357, 0.3125])
    def test_derivatives_2(self):
        # 创建长度为8的数组 x，包含整数0到7
        x = np.arange(8)
        # 计算 y = x^3 + 2*x^2，对数组 x 进行元素级别的计算
        y = x**3 + 2.*x**2

        # 使用样条插值法拟合 (x, y)，返回 B-spline 插值对象 tck
        tck = splrep(x, y, s=0)
        # 计算 B-spline 插值对象 tck 在点 3 处的前四个导数值
        ders = spalde(3, tck)
        # 断言前四个导数值与预期值的接近程度，设置容差为 1e-15
        assert_allclose(ders, [45.,   # 3**3 + 2*(3)**2
                               39.,   # 3*(3)**2 + 4*(3)
                               22.,   # 6*(3) + 4
                               6.],   # 6*3**0
                        atol=1e-15)

        # 使用样条插值法拟合 (x, y)，返回 k 阶的 UnivariateSpline 对象 spl
        spl = UnivariateSpline(x, y, s=0, k=3)
        # 断言 UnivariateSpline 对象 spl 在点 3 处的前四个导数值与之前计算的 ders 的接近程度
        assert_allclose(spl.derivatives(3),
                        ders,
                        atol=1e-15)

    def test_resize_regression(self):
        """Regression test for #1375."""
        # 定义输入数据 x, y, w
        x = [-1., -0.65016502, -0.58856235, -0.26903553, -0.17370892,
             -0.10011001, 0., 0.10011001, 0.17370892, 0.26903553, 0.58856235,
             0.65016502, 1.]
        y = [1.,0.62928599, 0.5797223, 0.39965815, 0.36322694, 0.3508061,
             0.35214793, 0.3508061, 0.36322694, 0.39965815, 0.5797223,
             0.62928599, 1.]
        w = [1.00000000e+12, 6.88875973e+02, 4.89314737e+02, 4.26864807e+02,
             6.07746770e+02, 4.51341444e+02, 3.17480210e+02, 4.51341444e+02,
             6.07746770e+02, 4.26864807e+02, 4.89314737e+02, 6.88875973e+02,
             1.00000000e+12]
        # 使用 UnivariateSpline 对象 spl 拟合 (x, y)，权重为 w，s=None 表示不使用平滑因子
        spl = UnivariateSpline(x=x, y=y, w=w, s=None)
        # 定义预期的输出值数组 desired
        desired = array([0.35100374, 0.51715855, 0.87789547, 0.98719344])
        # 断言 UnivariateSpline 对象 spl 在给定点数组 [0.1, 0.5, 0.9, 0.99] 处的输出值与 desired 的接近程度，设置容差为 5e-4
        assert_allclose(spl([0.1, 0.5, 0.9, 0.99]), desired, atol=5e-4)
    def test_out_of_range_regression(self):
        # 测试不同的外推模式。参见票号 3557
        x = np.arange(5, dtype=float)  # 创建一个包含5个元素的浮点数数组
        y = x**3  # 计算 x 的立方作为 y

        xp = linspace(-8, 13, 100)  # 在 -8 到 13 之间生成100个等间距的数
        xp_zeros = xp.copy()  # 复制 xp 数组到 xp_zeros
        xp_zeros[np.logical_or(xp_zeros < 0., xp_zeros > 4.)] = 0  # 将 xp_zeros 中小于0或大于4的元素置为0
        xp_clip = xp.copy()  # 复制 xp 数组到 xp_clip
        xp_clip[xp_clip < x[0]] = x[0]  # 将 xp_clip 中小于 x 数组的第一个元素的元素置为 x 的第一个元素
        xp_clip[xp_clip > x[-1]] = x[-1]  # 将 xp_clip 中大于 x 数组的最后一个元素的元素置为 x 的最后一个元素

        for cls in [UnivariateSpline, InterpolatedUnivariateSpline]:
            spl = cls(x=x, y=y)  # 使用 x 和 y 创建一个样条插值对象
            for ext in [0, 'extrapolate']:
                assert_allclose(spl(xp, ext=ext), xp**3, atol=1e-16)  # 断言样条插值在 xp 上的计算结果与 xp 的立方非常接近
                assert_allclose(cls(x, y, ext=ext)(xp), xp**3, atol=1e-16)  # 断言使用构造函数参数 ext 在 xp 上计算的结果与 xp 的立方非常接近
            for ext in [1, 'zeros']:
                assert_allclose(spl(xp, ext=ext), xp_zeros**3, atol=1e-16)  # 断言样条插值在 xp 上的计算结果与 xp_zeros 的立方非常接近
                assert_allclose(cls(x, y, ext=ext)(xp), xp_zeros**3, atol=1e-16)  # 断言使用构造函数参数 ext 在 xp 上计算的结果与 xp_zeros 的立方非常接近
            for ext in [2, 'raise']:
                assert_raises(ValueError, spl, xp, **dict(ext=ext))  # 断言在使用未知的外推模式时抛出 ValueError 异常
            for ext in [3, 'const']:
                assert_allclose(spl(xp, ext=ext), xp_clip**3, atol=1e-16)  # 断言样条插值在 xp 上的计算结果与 xp_clip 的立方非常接近
                assert_allclose(cls(x, y, ext=ext)(xp), xp_clip**3, atol=1e-16)  # 断言使用构造函数参数 ext 在 xp 上计算的结果与 xp_clip 的立方非常接近

        # 也测试 LSQUnivariateSpline（需要显式节点）
        t = spl.get_knots()[3:4]  # 选择样条插值的默认内部节点
        spl = LSQUnivariateSpline(x, y, t)
        assert_allclose(spl(xp, ext=0), xp**3, atol=1e-16)  # 断言最小二乘样条插值在 xp 上的计算结果与 xp 的立方非常接近
        assert_allclose(spl(xp, ext=1), xp_zeros**3, atol=1e-16)  # 断言最小二乘样条插值在 xp 上的计算结果与 xp_zeros 的立方非常接近
        assert_raises(ValueError, spl, xp, **dict(ext=2))  # 断言在使用未知的外推模式时抛出 ValueError 异常
        assert_allclose(spl(xp, ext=3), xp_clip**3, atol=1e-16)  # 断言最小二乘样条插值在 xp 上的计算结果与 xp_clip 的立方非常接近

        # 还要确保对 `ext` 的未知值进行早期捕获
        for ext in [-1, 'unknown']:
            spl = UnivariateSpline(x, y)
            assert_raises(ValueError, spl, xp, **dict(ext=ext))  # 断言在使用未知的外推模式时抛出 ValueError 异常
            assert_raises(ValueError, UnivariateSpline,
                          **dict(x=x, y=y, ext=ext))  # 断言在使用未知的外推模式时抛出 ValueError 异常

    def test_lsq_fpchec(self):
        xs = np.arange(100) * 1.  # 创建一个包含100个元素的浮点数数组
        ys = np.arange(100) * 1.  # 创建一个包含100个元素的浮点数数组
        knots = np.linspace(0, 99, 10)  # 在 0 到 99 之间生成10个等间距的节点
        bbox = (-1, 101)  # 指定边界框
        assert_raises(ValueError, LSQUnivariateSpline, xs, ys, knots,
                      bbox=bbox)  # 断言在创建最小二乘样条插值时，使用了超出边界框的节点会抛出 ValueError 异常

    def test_derivative_and_antiderivative(self):
        # 对 splder/splantider 的简单包装，仅进行轻量级的烟雾测试
        x = np.linspace(0, 1, 70)**3  # 创建一个包含70个元素的数组，元素为0到1之间的立方
        y = np.cos(x)  # 计算 x 的余弦作为 y

        spl = UnivariateSpline(x, y, s=0)  # 使用 x 和 y 创建一个样条插值对象
        spl2 = spl.antiderivative(2).derivative(2)  # 对样条插值进行两次反导和两次导数
        assert_allclose(spl(0.3), spl2(0.3))  # 断言样条插值在点0.3处的值与其两次反导两次导数后在点0.3处的值非常接近

        spl2 = spl.antiderivative(1)  # 对样条插值进行一次反导
        assert_allclose(spl2(0.6) - spl2(0.2),
                        spl.integral(0.2, 0.6))  # 断言样条插值在0.6和0.2处的值之差与其在0.2到0.6区间的积分值非常接近
    def test_derivative_extrapolation(self):
        # 回归测试 gh-10195：对于常数外推样条，其导数在外推时应为零
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5, 5]
        # 创建 UnivariateSpline 对象 f，使用常数外推且阶数为 3
        f = UnivariateSpline(x_values, y_values, ext='const', k=3)

        x = [-1, 0, -0.5, 9, 9.5, 10]
        # 断言：f.derivative()(x) 的结果应该在给定的绝对容差内接近于 0
        assert_allclose(f.derivative()(x), 0, atol=1e-15)

    def test_integral_out_of_bounds(self):
        # 回归测试 gh-7906：如果 a 和 b 都超出边界，.integral(a, b) 的结果错误
        x = np.linspace(0., 1., 7)
        for ext in range(4):
            # 创建 UnivariateSpline 对象 f，使用 0 平滑度和指定的外推方式
            f = UnivariateSpline(x, x, s=0, ext=ext)
            for (a, b) in [(1, 1), (1, 5), (2, 5),
                           (0, 0), (-2, 0), (-2, -1)]:
                # 断言：f.integral(a, b) 的结果应该在给定的绝对容差内接近于 0
                assert_allclose(f.integral(a, b), 0, atol=1e-15)

    def test_nan(self):
        # 如果输入数据包含 NaN，提前退出
        x = np.arange(10, dtype=float)
        y = x**3
        w = np.ones_like(x)
        # 测试 LSQUnivariateSpline [需要显式结节]
        spl = UnivariateSpline(x, y, check_finite=True)
        t = spl.get_knots()[3:4]  # 使用默认 k=3 的内部结节
        y_end = y[-1]
        for z in [np.nan, np.inf, -np.inf]:
            y[-1] = z
            # 断言：使用包含 NaN 的 y 值创建 UnivariateSpline 会引发 ValueError 异常
            assert_raises(ValueError, UnivariateSpline,
                          **dict(x=x, y=y, check_finite=True))
            # 断言：使用包含 NaN 的 y 值创建 InterpolatedUnivariateSpline 会引发 ValueError 异常
            assert_raises(ValueError, InterpolatedUnivariateSpline,
                          **dict(x=x, y=y, check_finite=True))
            # 断言：使用包含 NaN 的 y 值创建 LSQUnivariateSpline 会引发 ValueError 异常
            assert_raises(ValueError, LSQUnivariateSpline,
                          **dict(x=x, y=y, t=t, check_finite=True))
            y[-1] = y_end  # 恢复有效的 y 值但无效的 w 值检查
            w[-1] = z
            # 断言：使用包含 NaN 的 w 值创建 UnivariateSpline 会引发 ValueError 异常
            assert_raises(ValueError, UnivariateSpline,
                          **dict(x=x, y=y, w=w, check_finite=True))
            # 断言：使用包含 NaN 的 w 值创建 InterpolatedUnivariateSpline 会引发 ValueError 异常
            assert_raises(ValueError, InterpolatedUnivariateSpline,
                          **dict(x=x, y=y, w=w, check_finite=True))
            # 断言：使用包含 NaN 的 w 值创建 LSQUnivariateSpline 会引发 ValueError 异常
            assert_raises(ValueError, LSQUnivariateSpline,
                          **dict(x=x, y=y, t=t, w=w, check_finite=True))
    def test_strictly_increasing_x(self):
        # 测试对于 UnivariateSpline 当 s=0 和 InterpolatedUnivariateSpline 要求 x 必须严格递增，
        # 而对于 s>0 和 LSQUnivariateSpline 则要求 x 只需递增；参见 GitHub issue gh-8535
        xx = np.arange(10, dtype=float)  # 创建一个包含 0 到 9 的浮点数数组
        yy = xx**3  # 计算 xx 数组中每个元素的立方并赋值给 yy
        x = np.arange(10, dtype=float)  # 创建另一个与 xx 相同的浮点数数组 x
        x[1] = x[0]  # 修改 x 数组中索引为 1 的元素为索引为 0 的元素的值，使 x 不再严格递增
        y = x**3  # 计算 x 数组中每个元素的立方并赋值给 y
        w = np.ones_like(x)  # 创建一个与 x 数组形状相同的全为 1 的数组 w
        # 也测试 LSQUnivariateSpline [需要显式的节点]
        spl = UnivariateSpline(xx, yy, check_finite=True)  # 创建一个基于给定数据的样条插值对象
        t = spl.get_knots()[3:4]  # 获取样条插值对象的内部节点，默认情况下 k=3
        UnivariateSpline(x=x, y=y, w=w, s=1, check_finite=True)  # 创建一个基于给定数据的样条插值对象
        LSQUnivariateSpline(x=x, y=y, t=t, w=w, check_finite=True)  # 创建一个最小二乘样条插值对象
        assert_raises(ValueError, UnivariateSpline,
                **dict(x=x, y=y, s=0, check_finite=True))  # 断言当 s=0 时抛出 ValueError 异常
        assert_raises(ValueError, InterpolatedUnivariateSpline,
                **dict(x=x, y=y, check_finite=True))  # 断言创建 InterpolatedUnivariateSpline 对象时抛出 ValueError 异常

    def test_increasing_x(self):
        # 测试 x 必须严格递增，参见 GitHub issue gh-8535
        xx = np.arange(10, dtype=float)  # 创建一个包含 0 到 9 的浮点数数组
        yy = xx**3  # 计算 xx 数组中每个元素的立方并赋值给 yy
        x = np.arange(10, dtype=float)  # 创建另一个与 xx 相同的浮点数数组 x
        x[1] = x[0] - 1.0  # 修改 x 数组中索引为 1 的元素，使 x 不再递增
        y = x**3  # 计算 x 数组中每个元素的立方并赋值给 y
        w = np.ones_like(x)  # 创建一个与 x 数组形状相同的全为 1 的数组 w
        # 也测试 LSQUnivariateSpline [需要显式的节点]
        spl = UnivariateSpline(xx, yy, check_finite=True)  # 创建一个基于给定数据的样条插值对象
        t = spl.get_knots()[3:4]  # 获取样条插值对象的内部节点，默认情况下 k=3
        assert_raises(ValueError, UnivariateSpline,
                **dict(x=x, y=y, check_finite=True))  # 断言创建 UnivariateSpline 对象时抛出 ValueError 异常
        assert_raises(ValueError, InterpolatedUnivariateSpline,
                **dict(x=x, y=y, check_finite=True))  # 断言创建 InterpolatedUnivariateSpline 对象时抛出 ValueError 异常
        assert_raises(ValueError, LSQUnivariateSpline,
                **dict(x=x, y=y, t=t, w=w, check_finite=True))  # 断言创建 LSQUnivariateSpline 对象时抛出 ValueError 异常

    def test_invalid_input_for_univariate_spline(self):
        # 测试 UnivariateSpline 的无效输入
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5]
            UnivariateSpline(x_values, y_values)  # 创建 UnivariateSpline 对象
        assert "x and y should have a same length" in str(info.value)  # 断言异常信息中包含特定的错误信息

        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
            w_values = [-1.0, 1.0, 1.0, 1.0]
            UnivariateSpline(x_values, y_values, w=w_values)  # 创建带权重参数的 UnivariateSpline 对象
        assert "x, y, and w should have a same length" in str(info.value)  # 断言异常信息中包含特定的错误信息

        with assert_raises(ValueError) as info:
            bbox = (-1)
            UnivariateSpline(x_values, y_values, bbox=bbox)  # 创建带 bbox 参数的 UnivariateSpline 对象
        assert "bbox shape should be (2,)" in str(info.value)  # 断言异常信息中包含特定的错误信息

        with assert_raises(ValueError) as info:
            UnivariateSpline(x_values, y_values, k=6)  # 创建带 k 参数为 6 的 UnivariateSpline 对象
        assert "k should be 1 <= k <= 5" in str(info.value)  # 断言异常信息中包含特定的错误信息

        with assert_raises(ValueError) as info:
            UnivariateSpline(x_values, y_values, s=-1.0)  # 创建带 s 参数为 -1.0 的 UnivariateSpline 对象
        assert "s should be s >= 0.0" in str(info.value)  # 断言异常信息中包含特定的错误信息
    # 定义测试方法，验证 InterpolatedUnivariateSpline 对于无效输入的行为

        # 第一个测试用例：验证当 x 和 y 的长度不一致时，是否会引发 ValueError 异常
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5]
            # 创建 InterpolatedUnivariateSpline 对象，并传入不一致长度的 x 和 y
            InterpolatedUnivariateSpline(x_values, y_values)
        # 断言异常信息中包含预期的错误信息
        assert "x and y should have a same length" in str(info.value)

        # 第二个测试用例：验证当 x、y 和 w 的长度不一致时，是否会引发 ValueError 异常
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
            w_values = [-1.0, 1.0, 1.0, 1.0]
            # 创建 InterpolatedUnivariateSpline 对象，并传入不一致长度的 x、y 和 w
            InterpolatedUnivariateSpline(x_values, y_values, w=w_values)
        # 断言异常信息中包含预期的错误信息
        assert "x, y, and w should have a same length" in str(info.value)

        # 第三个测试用例：验证当 bbox 形状不符合预期 (2,) 时，是否会引发 ValueError 异常
        with assert_raises(ValueError) as info:
            bbox = (-1)
            # 创建 InterpolatedUnivariateSpline 对象，并传入不符合要求的 bbox
            InterpolatedUnivariateSpline(x_values, y_values, bbox=bbox)
        # 断言异常信息中包含预期的错误信息
        assert "bbox shape should be (2,)" in str(info.value)

        # 第四个测试用例：验证当 k 超出范围 (1 <= k <= 5) 时，是否会引发 ValueError 异常
        with assert_raises(ValueError) as info:
            # 创建 InterpolatedUnivariateSpline 对象，并传入超出范围的 k
            InterpolatedUnivariateSpline(x_values, y_values, k=6)
        # 断言异常信息中包含预期的错误信息
        assert "k should be 1 <= k <= 5" in str(info.value)

    # 定义测试方法，验证 LSQUnivariateSpline 对于无效输入的行为
    def test_invalid_input_for_lsq_univariate_spline(self):

        # 创建基本的 x 和 y 值
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
        # 创建 UnivariateSpline 对象用于获取内结点
        spl = UnivariateSpline(x_values, y_values, check_finite=True)
        # 获取内结点并截取第三个至第四个
        t_values = spl.get_knots()[3:4] 以默认的含义

        # 第一个测试用例：验证当 x 和 y 的长度不一致时，是否会引发 ValueError 异常
        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.0,  LSQUnivariateSpline(x
    # 定义一个测试函数，测试接受类似数组输入的情况
    def test_array_like_input(self):
        # 创建包含 x 值的 NumPy 数组
        x_values = np.array([1, 2, 4, 6, 8.5])
        # 创建包含 y 值的 NumPy 数组
        y_values = np.array([0.5, 0.8, 1.3, 2.5, 2.8])
        # 创建包含权重值的 NumPy 数组
        w_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        # 创建包含边界框的 NumPy 数组
        bbox = np.array([-100, 100])
        # 使用 NumPy 数组作为输入创建第一个样条插值对象
        spl1 = UnivariateSpline(x=x_values, y=y_values, w=w_values,
                                bbox=bbox)
        # 将 NumPy 数组转换为列表，并作为输入创建第二个样条插值对象
        spl2 = UnivariateSpline(x=x_values.tolist(), y=y_values.tolist(),
                                w=w_values.tolist(), bbox=bbox.tolist())
        
        # 断言两个插值对象在指定点上的值非常接近
        assert_allclose(spl1([0.1, 0.5, 0.9, 0.99]),
                        spl2([0.1, 0.5, 0.9, 0.99]))

    # 定义一个测试函数，测试在指定问题场景下的异常处理能力
    def test_fpknot_oob_crash(self):
        # 提供一个参考链接，说明此测试的背景和相关信息
        # https://github.com/scipy/scipy/issues/3691
        # 创建一个包含 x 值的范围对象
        x = range(109)
        # 创建一个长列表，包含特定模式的 y 值，用于模拟问题场景
        y = [0., 0., 0., 0., 0., 10.9, 0., 11., 0.,
             0., 0., 10.9, 0., 0., 0., 0., 0., 0.,
             10.9, 0., 0., 0., 11., 0., 0., 0., 10.9,
             0., 0., 0., 10.5, 0., 0., 0., 10.7, 0.,
             0., 0., 11., 0., 0., 0., 0., 0., 0.,
             10.9, 0., 0., 10.7, 0., 0., 0., 10.6, 0.,
             0., 0., 10.5, 0., 0., 10.7, 0., 0., 10.5,
             0., 0., 11.5, 0., 0., 0., 10.7, 0., 0.,
             10.7, 0., 0., 10.9, 0., 0., 10.8, 0., 0.,
             0., 10.7, 0., 0., 10.6, 0., 0., 0., 10.4,
             0., 0., 10.6, 0., 0., 10.5, 0., 0., 0.,
             10.7, 0., 0., 0., 10.4, 0., 0., 0., 10.8, 0.]
        # 使用警告抑制器来捕获特定类型的警告信息
        with suppress_warnings() as sup:
            # 记录并抑制用户警告，用于特定场景下的异常情况
            r = sup.record(
                UserWarning,
                r"""
# 定义一个测试类 TestLSQBivariateSpline，用于测试 LSQBivariateSpline 类的功能
class TestLSQBivariateSpline:

    # 注意: 该测试类中的系统是秩亏的
    def test_linear_constant(self):
        # 定义测试数据 x, y, z
        x = [1,1,1,2,2,2,3,3,3]
        y = [1,2,3,1,2,3,1,2,3]
        z = [3,3,3,3,3,3,3,3,3]
        s = 0.1  # 设置平滑参数 s 的值为 0.1
        tx = [1+s,3-s]  # 设置 tx 节点
        ty = [1+s,3-s]  # 设置 ty 节点
        # 使用 suppress_warnings 上下文管理器捕获 UserWarning 类型的警告信息
        with suppress_warnings() as sup:
            # 监听 "The coefficients of the spline" 警告信息，并记录到 sup 中
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            # 创建 LSQBivariateSpline 对象 lut
            lut = LSQBivariateSpline(x,y,z,tx,ty,kx=1,ky=1)
            # 断言捕获的警告信息数量为 1
            assert_equal(len(r), 1)

        # 断言 lut(2,2) 的返回值接近于 3.0
        assert_almost_equal(lut(2,2), 3.)

    def test_bilinearity(self):
        # 定义测试数据 x, y, z
        x = [1,1,1,2,2,2,3,3,3]
        y = [1,2,3,1,2,3,1,2,3]
        z = [0,7,8,3,4,7,1,3,4]
        s = 0.1  # 设置平滑参数 s 的值为 0.1
        tx = [1+s,3-s]  # 设置 tx 节点
        ty = [1+s,3-s]  # 设置 ty 节点
        # 使用 suppress_warnings 上下文管理器捕获 UserWarning 类型的警告信息
        with suppress_warnings() as sup:
            # 过滤 "The coefficients of the spline" 警告信息，不记录
            sup.filter(UserWarning, "\nThe coefficients of the spline")
            # 创建 LSQBivariateSpline 对象 lut
            lut = LSQBivariateSpline(x,y,z,tx,ty,kx=1,ky=1)

        # 获取 lut 对象的节点 tx, ty
        tx, ty = lut.get_knots()
        # 遍历节点进行插值测试
        for xa, xb in zip(tx[:-1], tx[1:]):
            for ya, yb in zip(ty[:-1], ty[1:]):
                for t in [0.1, 0.5, 0.9]:
                    for s in [0.3, 0.4, 0.7]:
                        # 计算插值点 xp, yp，以及插值 zp
                        xp = xa*(1-t) + xb*t
                        yp = ya*(1-s) + yb*s
                        zp = (+ lut(xa, ya)*(1-t)*(1-s)
                              + lut(xb, ya)*t*(1-s)
                              + lut(xa, yb)*(1-t)*s
                              + lut(xb, yb)*t*s)
                        # 断言 lut(xp,yp) 的返回值接近于 zp
                        assert_almost_equal(lut(xp,yp), zp)

    def test_integral(self):
        # 定义测试数据 x, y, z
        x = [1,1,1,2,2,2,8,8,8]
        y = [1,2,3,1,2,3,1,2,3]
        z = array([0,7,8,3,4,7,1,3,4])

        s = 0.1  # 设置平滑参数 s 的值为 0.1
        tx = [1+s,3-s]  # 设置 tx 节点
        ty = [1+s,3-s]  # 设置 ty 节点
        # 使用 suppress_warnings 上下文管理器捕获 UserWarning 类型的警告信息
        with suppress_warnings() as sup:
            # 监听 "The coefficients of the spline" 警告信息，并记录到 sup 中
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            # 创建 LSQBivariateSpline 对象 lut
            lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)
            # 断言捕获的警告信息数量为 1
            assert_equal(len(r), 1)
        # 获取 lut 对象的节点 tx, ty
        tx, ty = lut.get_knots()
        # 计算 lut 对象的积分值 trpz
        trpz = .25*(diff(tx)[:,None]*diff(ty)[None,:]
                    * (lut(tx[:-1,:-1])+lut(tx[1:,:-1])+lut(tx[:-1,1:])+lut(tx[1:,1:]))).sum()

        # 断言 lut 在指定区间的积分结果接近于 trpz
        assert_almost_equal(lut.integral(tx[0], tx[-1], ty[0], ty[-1]),
                            trpz)
    def test_empty_input(self):
        # 测试空输入是否返回空输出。Ticket 1014
        x = [1,1,1,2,2,2,3,3,3]  # 定义输入 x
        y = [1,2,3,1,2,3,1,2,3]  # 定义输入 y
        z = [3,3,3,3,3,3,3,3,3]  # 定义输入 z
        s = 0.1  # 定义平移量 s
        tx = [1+s,3-s]  # 定义平移后的 tx
        ty = [1+s,3-s]  # 定义平移后的 ty
        with suppress_warnings() as sup:  # 使用警告抑制器
            r = sup.record(UserWarning, "\nThe coefficients of the spline")  # 记录 UserWarning 类型的警告消息
            lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)  # 创建 LSQBivariateSpline 对象
            assert_equal(len(r), 1)  # 断言记录的警告数量为 1

        assert_array_equal(lut([], []), np.zeros((0,0)))  # 断言空输入时返回一个全零的二维数组
        assert_array_equal(lut([], [], grid=False), np.zeros((0,)))  # 断言空输入时返回一个全零的一维数组

    def test_invalid_input(self):
        s = 0.1  # 定义平移量 s
        tx = [1 + s, 3 - s]  # 定义平移后的 tx
        ty = [1 + s, 3 - s]  # 定义平移后的 ty

        with assert_raises(ValueError) as info:  # 捕获 ValueError 异常，并命名为 info
            x = np.linspace(1.0, 10.0)  # 生成 1.0 到 10.0 的等间距数组 x
            y = np.linspace(1.0, 10.0)  # 生成 1.0 到 10.0 的等间距数组 y
            z = np.linspace(1.0, 10.0, num=10)  # 生成长度为 10 的等间距数组 z
            LSQBivariateSpline(x, y, z, tx, ty)  # 创建 LSQBivariateSpline 对象
        assert "x, y, and z should have a same length" in str(info.value)  # 断言异常信息包含特定字符串

        with assert_raises(ValueError) as info:  # 捕获 ValueError 异常，并命名为 info
            x = np.linspace(1.0, 10.0)  # 生成 1.0 到 10.0 的等间距数组 x
            y = np.linspace(1.0, 10.0)  # 生成 1.0 到 10.0 的等间距数组 y
            z = np.linspace(1.0, 10.0)  # 生成 1.0 到 10.0 的等间距数组 z
            w = np.linspace(1.0, 10.0, num=20)  # 生成长度为 20 的等间距数组 w
            LSQBivariateSpline(x, y, z, tx, ty, w=w)  # 创建 LSQBivariateSpline 对象
        assert "x, y, z, and w should have a same length" in str(info.value)  # 断言异常信息包含特定字符串

        with assert_raises(ValueError) as info:  # 捕获 ValueError 异常，并命名为 info
            w = np.linspace(-1.0, 10.0)  # 生成 -1.0 到 10.0 的等间距数组 w
            LSQBivariateSpline(x, y, z, tx, ty, w=w)  # 创建 LSQBivariateSpline 对象
        assert "w should be positive" in str(info.value)  # 断言异常信息包含特定字符串

        with assert_raises(ValueError) as info:  # 捕获 ValueError 异常，并命名为 info
            bbox = (-100, 100, -100)  # 定义无效的 bbox
            LSQBivariateSpline(x, y, z, tx, ty, bbox=bbox)  # 创建 LSQBivariateSpline 对象
        assert "bbox shape should be (4,)" in str(info.value)  # 断言异常信息包含特定字符串

        with assert_raises(ValueError) as info:  # 捕获 ValueError 异常，并命名为 info
            LSQBivariateSpline(x, y, z, tx, ty, kx=10, ky=10)  # 创建 LSQBivariateSpline 对象
        assert "The length of x, y and z should be at least (kx+1) * (ky+1)" in \
               str(info.value)  # 断言异常信息包含特定字符串

        with assert_raises(ValueError) as exc_info:  # 捕获 ValueError 异常，并命名为 exc_info
            LSQBivariateSpline(x, y, z, tx, ty, eps=0.0)  # 创建 LSQBivariateSpline 对象
        assert "eps should be between (0, 1)" in str(exc_info.value)  # 断言异常信息包含特定字符串

        with assert_raises(ValueError) as exc_info:  # 捕获 ValueError 异常，并命名为 exc_info
            LSQBivariateSpline(x, y, z, tx, ty, eps=1.0)  # 创建 LSQBivariateSpline 对象
        assert "eps should be between (0, 1)" in str(exc_info.value)  # 断言异常信息包含特定字符串
    def test_array_like_input(self):
        s = 0.1
        # 创建包含浮点数的 NumPy 数组
        tx = np.array([1 + s, 3 - s])
        ty = np.array([1 + s, 3 - s])
        # 创建包含默认范围内均匀间隔的 NumPy 数组
        x = np.linspace(1.0, 10.0)
        y = np.linspace(1.0, 10.0)
        z = np.linspace(1.0, 10.0)
        w = np.linspace(1.0, 10.0)
        bbox = np.array([1.0, 10.0, 1.0, 10.0])

        with suppress_warnings() as sup:
            # 捕获特定类型的警告，并记录它们
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            # 使用 LSQBivariateSpline 类构造二维样条插值对象，使用 NumPy 数组作为输入
            spl1 = LSQBivariateSpline(x, y, z, tx, ty, w=w, bbox=bbox)
            # 使用 LSQBivariateSpline 类构造二维样条插值对象，将 NumPy 数组转换为列表作为输入
            spl2 = LSQBivariateSpline(x.tolist(), y.tolist(), z.tolist(),
                                      tx.tolist(), ty.tolist(), w=w.tolist(),
                                      bbox=bbox)
            # 验证两个插值对象在给定点的返回值接近
            assert_allclose(spl1(2.0, 2.0), spl2(2.0, 2.0))
            # 验证捕获的警告数量为2
            assert_equal(len(r), 2)

    def test_unequal_length_of_knots(self):
        """Test for the case when the input knot-location arrays in x and y are
        of different lengths.
        """
        # 创建网格矩阵并展平成一维数组
        x, y = np.mgrid[0:100, 0:100]
        x = x.ravel()
        y = y.ravel()
        # 创建包含常数值的 NumPy 数组
        z = 3.0 * np.ones_like(x)
        # 创建不等长度的节点数组
        tx = np.linspace(0.1, 98.0, 29)
        ty = np.linspace(0.1, 98.0, 33)
        with suppress_warnings() as sup:
            # 捕获特定类型的警告，并记录它们
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            # 使用 LSQBivariateSpline 类构造二维样条插值对象，使用 NumPy 数组作为输入
            lut = LSQBivariateSpline(x,y,z,tx,ty)
            # 验证捕获的警告数量为1
            assert_equal(len(r), 1)

        # 验证插值对象在给定点的返回值近似等于 z
        assert_almost_equal(lut(x, y, grid=False), z)
class TestSmoothBivariateSpline:
    # 定义测试类 TestSmoothBivariateSpline

    def test_linear_constant(self):
        # 定义测试方法 test_linear_constant
        x = [1,1,1,2,2,2,3,3,3]  # 定义 x 数据点
        y = [1,2,3,1,2,3,1,2,3]  # 定义 y 数据点
        z = [3,3,3,3,3,3,3,3,3]  # 定义 z 数据点
        lut = SmoothBivariateSpline(x,y,z,kx=1,ky=1)  # 创建 SmoothBivariateSpline 对象
        assert_array_almost_equal(lut.get_knots(),([1,1,3,3],[1,1,3,3]))  # 断言节点近似相等
        assert_array_almost_equal(lut.get_coeffs(),[3,3,3,3])  # 断言系数近似相等
        assert_almost_equal(lut.get_residual(),0.0)  # 断言残差近似为0.0
        assert_array_almost_equal(lut([1,1.5,2],[1,1.5]),[[3,3],[3,3],[3,3]])  # 断言插值结果近似相等

    def test_linear_1d(self):
        # 定义测试方法 test_linear_1d
        x = [1,1,1,2,2,2,3,3,3]  # 定义 x 数据点
        y = [1,2,3,1,2,3,1,2,3]  # 定义 y 数据点
        z = [0,0,0,2,2,2,4,4,4]  # 定义 z 数据点
        lut = SmoothBivariateSpline(x,y,z,kx=1,ky=1)  # 创建 SmoothBivariateSpline 对象
        assert_array_almost_equal(lut.get_knots(),([1,1,3,3],[1,1,3,3]))  # 断言节点近似相等
        assert_array_almost_equal(lut.get_coeffs(),[0,0,4,4])  # 断言系数近似相等
        assert_almost_equal(lut.get_residual(),0.0)  # 断言残差近似为0.0
        assert_array_almost_equal(lut([1,1.5,2],[1,1.5]),[[0,0],[1,1],[2,2]])  # 断言插值结果近似相等

    def test_integral(self):
        # 定义测试方法 test_integral
        x = [1,1,1,2,2,2,4,4,4]  # 定义 x 数据点
        y = [1,2,3,1,2,3,1,2,3]  # 定义 y 数据点
        z = array([0,7,8,3,4,7,1,3,4])  # 定义 z 数据点为数组

        with suppress_warnings() as sup:  # 使用 suppress_warnings 上下文管理器
            # 这里似乎会失败 (ier=1, 参见 ticket 1642).
            sup.filter(UserWarning, "\nThe required storage space")
            lut = SmoothBivariateSpline(x, y, z, kx=1, ky=1, s=0)  # 创建 SmoothBivariateSpline 对象

        tx = [1,2,4]  # 定义新的 x 数据点列表
        ty = [1,2,3]  # 定义新的 y 数据点列表

        tz = lut(tx, ty)  # 对 lut 进行插值计算
        trpz = .25*(diff(tx)[:,None]*diff(ty)[None,:]
                    * (tz[:-1,:-1]+tz[1:,:-1]+tz[:-1,1:]+tz[1:,1:])).sum()  # 计算在给定区域的积分值
        assert_almost_equal(lut.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz)  # 断言积分结果近似相等

        lut2 = SmoothBivariateSpline(x, y, z, kx=2, ky=2, s=0)  # 创建另一个 SmoothBivariateSpline 对象
        assert_almost_equal(lut2.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz,
                            decimal=0)  # 断言积分结果近似相等，精度为0

        tz = lut(tx[:-1], ty[:-1])  # 对 lut 进行插值计算，去掉最后一个点
        trpz = .25*(diff(tx[:-1])[:,None]*diff(ty[:-1])[None,:]
                    * (tz[:-1,:-1]+tz[1:,:-1]+tz[:-1,1:]+tz[1:,1:])).sum()  # 计算在给定区域的积分值
        assert_almost_equal(lut.integral(tx[0], tx[-2], ty[0], ty[-2]), trpz)  # 断言积分结果近似相等

    def test_rerun_lwrk2_too_small(self):
        # 在这种情况下，默认运行中 lwrk2 太小。这里我们检查与 bisplrep/bisplev 输出的相等性，
        # 因为在那里，如果 ier>10，会自动重新运行样条表示。
        x = np.linspace(-2, 2, 80)  # 在 [-2, 2] 区间内生成80个均匀间隔的点作为 x
        y = np.linspace(-2, 2, 80)  # 在 [-2, 2] 区间内生成80个均匀间隔的点作为 y
        z = x + y  # z 的值是 x 和 y 的和
        xi = np.linspace(-1, 1, 100)  # 在 [-1, 1] 区间内生成100个均匀间隔的点作为 xi
        yi = np.linspace(-2, 2, 100)  # 在 [-2, 2] 区间内生成100个均匀间隔的点作为 yi
        tck = bisplrep(x, y, z)  # 用 x, y, z 数据生成二维 B-spline 表示
        res1 = bisplev(xi, yi, tck)  # 使用 tck 进行插值计算
        interp_ = SmoothBivariateSpline(x, y, z)  # 创建 SmoothBivariateSpline 对象
        res2 = interp_(xi, yi)  # 对 xi, yi 进行插值计算
        assert_almost_equal(res1, res2)  # 断言插值结果近似相等
    # 定义测试无效输入的方法
    def test_invalid_input(self):

        # 检查 x, y, z 的长度不一致时是否引发 ValueError 异常
        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0, num=10)
            SmoothBivariateSpline(x, y, z)
        assert "x, y, and z should have a same length" in str(info.value)

        # 检查 x, y, z, w 的长度不一致时是否引发 ValueError 异常
        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0)
            w = np.linspace(1.0, 10.0, num=20)
            SmoothBivariateSpline(x, y, z, w=w)
        assert "x, y, z, and w should have a same length" in str(info.value)

        # 检查 w 中存在非正值时是否引发 ValueError 异常
        with assert_raises(ValueError) as info:
            w = np.linspace(-1.0, 10.0)
            SmoothBivariateSpline(x, y, z, w=w)
        assert "w should be positive" in str(info.value)

        # 检查 bbox 形状不正确时是否引发 ValueError 异常
        with assert_raises(ValueError) as info:
            bbox = (-100, 100, -100)
            SmoothBivariateSpline(x, y, z, bbox=bbox)
        assert "bbox shape should be (4,)" in str(info.value)

        # 检查 x, y, z 的长度不足以支持指定的 kx, ky 时是否引发 ValueError 异常
        with assert_raises(ValueError) as info:
            SmoothBivariateSpline(x, y, z, kx=10, ky=10)
        assert "The length of x, y and z should be at least (kx+1) * (ky+1)" in\
               str(info.value)

        # 检查 s 参数小于 0 时是否引发 ValueError 异常
        with assert_raises(ValueError) as info:
            SmoothBivariateSpline(x, y, z, s=-1.0)
        assert "s should be s >= 0.0" in str(info.value)

        # 检查 eps 参数超出 (0, 1) 范围时是否引发 ValueError 异常
        with assert_raises(ValueError) as exc_info:
            SmoothBivariateSpline(x, y, z, eps=0.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

        # 检查 eps 参数超出 (0, 1) 范围时是否引发 ValueError 异常
        with assert_raises(ValueError) as exc_info:
            SmoothBivariateSpline(x, y, z, eps=1.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

    # 定义测试数组样式输入的方法
    def test_array_like_input(self):
        # 创建示例数组
        x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        z = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        w = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        bbox = np.array([1.0, 3.0, 1.0, 3.0])
        
        # 使用 np.array 作为输入创建 SmoothBivariateSpline 对象
        spl1 = SmoothBivariateSpline(x, y, z, w=w, bbox=bbox, kx=1, ky=1)
        
        # 使用 list 作为输入创建 SmoothBivariateSpline 对象
        spl2 = SmoothBivariateSpline(x.tolist(), y.tolist(), z.tolist(),
                                     bbox=bbox.tolist(), w=w.tolist(),
                                     kx=1, ky=1)
        
        # 断言两个对象在相同输入下产生相似结果
        assert_allclose(spl1(0.1, 0.5), spl2(0.1, 0.5))
class TestLSQSphereBivariateSpline:
    def setup_method(self):
        # 定义输入数据和坐标
        ntheta, nphi = 70, 90
        theta = linspace(0.5/(ntheta - 1), 1 - 0.5/(ntheta - 1), ntheta) * pi
        phi = linspace(0.5/(nphi - 1), 1 - 0.5/(nphi - 1), nphi) * 2. * pi
        data = ones((theta.shape[0], phi.shape[0]))
        
        # 定义结节并提取结节处的数据值
        knotst = theta[::5]
        knotsp = phi[::5]
        knotdata = data[::5, ::5]
        
        # 计算样条系数
        lats, lons = meshgrid(theta, phi)
        lut_lsq = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                           data.T.ravel(), knotst, knotsp)
        self.lut_lsq = lut_lsq
        self.data = knotdata
        self.new_lons, self.new_lats = knotsp, knotst

    def test_linear_constant(self):
        # 断言残差接近零
        assert_almost_equal(self.lut_lsq.get_residual(), 0.0)
        # 断言新经纬度网格插值结果接近预定义数据
        assert_array_almost_equal(self.lut_lsq(self.new_lats, self.new_lons),
                                  self.data)

    def test_empty_input(self):
        # 断言空输入返回零数组
        assert_array_almost_equal(self.lut_lsq([], []), np.zeros((0, 0)))
        assert_array_almost_equal(self.lut_lsq([], [], grid=False), np.zeros((0,)))

    def test_array_like_input(self):
        ntheta, nphi = 70, 90
        theta = linspace(0.5 / (ntheta - 1), 1 - 0.5 / (ntheta - 1),
                         ntheta) * pi
        phi = linspace(0.5 / (nphi - 1), 1 - 0.5 / (nphi - 1),
                       nphi) * 2. * pi
        lats, lons = meshgrid(theta, phi)
        data = ones((theta.shape[0], phi.shape[0]))
        
        # 定义结节并提取结节处的数据值
        knotst = theta[::5]
        knotsp = phi[::5]
        w = ones(lats.ravel().shape[0])
        
        # 使用 np.array 输入创建样条插值对象
        spl1 = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                        data.T.ravel(), knotst, knotsp, w=w)
        # 使用列表输入创建样条插值对象
        spl2 = LSQSphereBivariateSpline(lats.ravel().tolist(),
                                        lons.ravel().tolist(),
                                        data.T.ravel().tolist(),
                                        knotst.tolist(),
                                        knotsp.tolist(), w=w.tolist())
        # 断言两种方式创建的插值对象在相同点处的值接近
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))


class TestSmoothSphereBivariateSpline:
    def setup_method(self):
        theta = array([.25*pi, .25*pi, .25*pi, .5*pi, .5*pi, .5*pi, .75*pi,
                       .75*pi, .75*pi])
        phi = array([.5 * pi, pi, 1.5 * pi, .5 * pi, pi, 1.5 * pi, .5 * pi, pi,
                     1.5 * pi])
        r = array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        # 创建 SmoothSphereBivariateSpline 对象
        self.lut = SmoothSphereBivariateSpline(theta, phi, r, s=1E10)

    def test_linear_constant(self):
        # 断言残差接近零
        assert_almost_equal(self.lut.get_residual(), 0.)
        # 断言在指定点的插值结果接近预期值
        assert_array_almost_equal(self.lut([1, 1.5, 2], [1, 1.5]),
                                  [[3, 3], [3, 3], [3, 3]])
    # 测试空输入情况
    def test_empty_input(self):
        # 断言调用 self.lut([], []) 返回一个与 np.zeros((0,0)) 几乎相等的数组
        assert_array_almost_equal(self.lut([], []), np.zeros((0,0)))
        # 断言调用 self.lut([], [], grid=False) 返回一个与 np.zeros((0,)) 几乎相等的数组
        assert_array_almost_equal(self.lut([], [], grid=False), np.zeros((0,)))

    # 测试无效输入情况
    def test_invalid_input(self):
        # 创建 theta 数组
        theta = array([.25 * pi, .25 * pi, .25 * pi, .5 * pi, .5 * pi, .5 * pi,
                       .75 * pi, .75 * pi, .75 * pi])
        # 创建 phi 数组
        phi = array([.5 * pi, pi, 1.5 * pi, .5 * pi, pi, 1.5 * pi, .5 * pi, pi,
                     1.5 * pi])
        # 创建 r 数组
        r = array([3, 3, 3, 3, 3, 3, 3, 3, 3])

        # 第一个异常断言块：验证 SmoothSphereBivariateSpline 构造函数对无效 theta 抛出 ValueError 异常
        with assert_raises(ValueError) as exc_info:
            # 创建无效的 theta 数组
            invalid_theta = array([-0.1 * pi, .25 * pi, .25 * pi, .5 * pi,
                                   .5 * pi, .5 * pi, .75 * pi, .75 * pi,
                                   .75 * pi])
            # 调用 SmoothSphereBivariateSpline 构造函数
            SmoothSphereBivariateSpline(invalid_theta, phi, r, s=1E10)
        # 断言异常信息中包含 "theta should be between [0, pi]"
        assert "theta should be between [0, pi]" in str(exc_info.value)

        # 后续的异常断言块，依次验证 SmoothSphereBivariateSpline 对其他无效输入的异常抛出，包括 theta、phi、r 和参数 s、eps 的检查
        # 每个断言块都与其对应的异常信息进行验证，确保异常信息中包含正确的错误描述信息
        # 注意，异常断言使用 assert_raises 来捕获异常并检查异常消息

        # 省略后续的异常断言代码...


这些注释详细说明了每个测试函数和其中的每个断言的作用，以及各个异常断言如何验证代码中的输入边界和参数限制。
    # 定义一个测试方法，用于测试球面双变量样条插值的类的行为
    def test_array_like_input(self):
        # 创建角度 theta 的 NumPy 数组
        theta = np.array([.25 * pi, .25 * pi, .25 * pi, .5 * pi, .5 * pi,
                          .5 * pi, .75 * pi, .75 * pi, .75 * pi])
        # 创建角度 phi 的 NumPy 数组
        phi = np.array([.5 * pi, pi, 1.5 * pi, .5 * pi, pi, 1.5 * pi, .5 * pi,
                        pi, 1.5 * pi])
        # 创建距离 r 的 NumPy 数组
        r = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        # 创建权重 w 的 NumPy 数组
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # 使用 NumPy 数组作为输入，创建 SmoothSphereBivariateSpline 对象
        spl1 = SmoothSphereBivariateSpline(theta, phi, r, w=w, s=1E10)

        # 将 NumPy 数组转换为列表，并作为输入，创建另一个 SmoothSphereBivariateSpline 对象
        spl2 = SmoothSphereBivariateSpline(theta.tolist(), phi.tolist(),
                                           r.tolist(), w=w.tolist(), s=1E10)
        
        # 断言两个插值对象在给定点 (1.0, 1.0) 处的值近似相等
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))
class TestRectBivariateSpline:
    # 定义测试类 TestRectBivariateSpline

    def test_defaults(self):
        # 定义默认参数测试方法
        x = array([1,2,3,4,5])
        # 创建数组 x 包含值 [1, 2, 3, 4, 5]
        y = array([1,2,3,4,5])
        # 创建数组 y 包含值 [1, 2, 3, 4, 5]
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        # 创建二维数组 z，表示一个平面的高度值
        lut = RectBivariateSpline(x,y,z)
        # 使用 x, y, z 创建 RectBivariateSpline 对象 lut
        assert_array_almost_equal(lut(x,y),z)
        # 断言 lut 在输入 x, y 时的输出与 z 几乎相等

    def test_evaluate(self):
        # 定义评估方法测试
        x = array([1,2,3,4,5])
        # 创建数组 x 包含值 [1, 2, 3, 4, 5]
        y = array([1,2,3,4,5])
        # 创建数组 y 包含值 [1, 2, 3, 4, 5]
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        # 创建二维数组 z，表示一个平面的高度值
        lut = RectBivariateSpline(x,y,z)
        # 使用 x, y, z 创建 RectBivariateSpline 对象 lut

        xi = [1, 2.3, 5.3, 0.5, 3.3, 1.2, 3]
        # 创建数组 xi 包含多个浮点数
        yi = [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]
        # 创建数组 yi 包含多个浮点数
        zi = lut.ev(xi, yi)
        # 使用 lut 对象的 ev 方法计算 xi, yi 所对应的插值 zi
        zi2 = array([lut(xp, yp)[0,0] for xp, yp in zip(xi, yi)])
        # 使用列表推导式计算每对 (xp, yp) 在 lut 上的插值，并转换成数组

        assert_almost_equal(zi, zi2)
        # 断言 zi 与 zi2 几乎相等

    def test_derivatives_grid(self):
        # 定义网格导数测试方法
        x = array([1,2,3,4,5])
        # 创建数组 x 包含值 [1, 2, 3, 4, 5]
        y = array([1,2,3,4,5])
        # 创建数组 y 包含值 [1, 2, 3, 4, 5]
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        # 创建二维数组 z，表示一个平面的高度值
        dx = array([[0,0,-20,0,0],[0,0,13,0,0],[0,0,4,0,0],
            [0,0,-11,0,0],[0,0,4,0,0]])/6.
        # 创建二维数组 dx，表示 x 方向的偏导数
        dy = array([[4,-1,0,1,-4],[4,-1,0,1,-4],[0,1.5,0,-1.5,0],
            [2,.25,0,-.25,-2],[4,-1,0,1,-4]])
        # 创建二维数组 dy，表示 y 方向的偏导数
        dxdy = array([[40,-25,0,25,-40],[-26,16.25,0,-16.25,26],
            [-8,5,0,-5,8],[22,-13.75,0,13.75,-22],[-8,5,0,-5,8]])/6.
        # 创建二维数组 dxdy，表示 x 和 y 方向的混合偏导数
        lut = RectBivariateSpline(x,y,z)
        # 使用 x, y, z 创建 RectBivariateSpline 对象 lut
        assert_array_almost_equal(lut(x,y,dx=1),dx)
        # 断言 lut 在输入 x, y 并计算 x 方向偏导数时的输出与 dx 几乎相等
        assert_array_almost_equal(lut(x,y,dy=1),dy)
        # 断言 lut 在输入 x, y 并计算 y 方向偏导数时的输出与 dy 几乎相等
        assert_array_almost_equal(lut(x,y,dx=1,dy=1),dxdy)
        # 断言 lut 在输入 x, y 并计算 x 和 y 方向混合偏导数时的输出与 dxdy 几乎相等

    def test_derivatives(self):
        # 定义导数测试方法
        x = array([1,2,3,4,5])
        # 创建数组 x 包含值 [1, 2, 3, 4, 5]
        y = array([1,2,3,4,5])
        # 创建数组 y 包含值 [1, 2, 3, 4, 5]
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        # 创建二维数组 z，表示一个平面的高度值
        dx = array([0,0,2./3,0,0])
        # 创建数组 dx，表示 x 方向的偏导数
        dy = array([4,-1,0,-.25,-4])
        # 创建数组 dy，表示 y 方向的偏导数
        dxdy = array([160,65,0,55,32])/24.
        # 创建数组 dxdy，表示 x 和 y 方向的混合偏导数
        lut = RectBivariateSpline(x,y,z)
        # 使用 x, y, z 创建 RectBivariateSpline 对象 lut
        assert_array_almost_equal(lut(x,y,dx=1,grid=False),dx)
        # 断言 lut 在输入 x, y 并计算 x 方向偏导数时的输出与 dx 几乎相等
        assert_array_almost_equal(lut(x,y,dy=1,grid=False),dy)
        # 断言 lut 在输入 x, y 并计算 y 方向偏导数时的输出与 dy 几乎相等
        assert_array_almost_equal(lut(x,y,dx=1,dy=1,grid=False),dxdy)
        # 断言 lut 在输入 x, y 并计算 x 和 y 方向混合偏导数时的输出与 dxdy 几乎相等
    # 定义测试函数，用于测试 RectBivariateSpline 类的偏导数方法在网格上的表现
    def test_partial_derivative_method_grid(self):
        # 定义输入变量 x, y 和二维数组 z
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1],
                   [1, 2, 1, 2, 1],
                   [1, 2, 3, 2, 1],
                   [1, 2, 2, 2, 1],
                   [1, 2, 1, 2, 1]])
        # 预期的 x 偏导数 dx，y 偏导数 dy，以及混合偏导数 dxdy
        dx = array([[0, 0, -20, 0, 0],
                    [0, 0, 13, 0, 0],
                    [0, 0, 4, 0, 0],
                    [0, 0, -11, 0, 0],
                    [0, 0, 4, 0, 0]]) / 6.
        dy = array([[4, -1, 0, 1, -4],
                    [4, -1, 0, 1, -4],
                    [0, 1.5, 0, -1.5, 0],
                    [2, .25, 0, -.25, -2],
                    [4, -1, 0, 1, -4]])
        dxdy = array([[40, -25, 0, 25, -40],
                      [-26, 16.25, 0, -16.25, 26],
                      [-8, 5, 0, -5, 8],
                      [22, -13.75, 0, 13.75, -22],
                      [-8, 5, 0, -5, 8]]) / 6.
        # 创建 RectBivariateSpline 对象 lut，基于输入的 x, y 和 z
        lut = RectBivariateSpline(x, y, z)
        # 断言在网格上计算的 x 偏导数与预期值 dx 相等
        assert_array_almost_equal(lut.partial_derivative(1, 0)(x, y), dx)
        # 断言在网格上计算的 y 偏导数与预期值 dy 相等
        assert_array_almost_equal(lut.partial_derivative(0, 1)(x, y), dy)
        # 断言在网格上计算的混合偏导数与预期值 dxdy 相等
        assert_array_almost_equal(lut.partial_derivative(1, 1)(x, y), dxdy)

    # 定义测试函数，用于测试 RectBivariateSpline 类的偏导数方法在非网格上的表现
    def test_partial_derivative_method(self):
        # 定义输入变量 x, y 和二维数组 z
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1],
                   [1, 2, 1, 2, 1],
                   [1, 2, 3, 2, 1],
                   [1, 2, 2, 2, 1],
                   [1, 2, 1, 2, 1]])
        # 预期的 x 偏导数 dx，y 偏导数 dy，以及混合偏导数 dxdy
        dx = array([0, 0, 2./3, 0, 0])
        dy = array([4, -1, 0, -.25, -4])
        dxdy = array([160, 65, 0, 55, 32]) / 24.
        # 创建 RectBivariateSpline 对象 lut，基于输入的 x, y 和 z
        lut = RectBivariateSpline(x, y, z)
        # 断言在非网格上计算的 x 偏导数与预期值 dx 相等
        assert_array_almost_equal(lut.partial_derivative(1, 0)(x, y, grid=False),
                                  dx)
        # 断言在非网格上计算的 y 偏导数与预期值 dy 相等
        assert_array_almost_equal(lut.partial_derivative(0, 1)(x, y, grid=False),
                                  dy)
        # 断言在非网格上计算的混合偏导数与预期值 dxdy 相等
        assert_array_almost_equal(lut.partial_derivative(1, 1)(x, y, grid=False),
                                  dxdy)

    # 定义测试函数，用于测试 RectBivariateSpline 类在超出阶数限制时是否引发 ValueError
    def test_partial_derivative_order_too_large(self):
        # 定义输入变量 x 和 y，以及全为 1 的二维数组 z
        x = array([0, 1, 2, 3, 4], dtype=float)
        y = x.copy()
        z = ones((x.size, y.size))
        # 创建 RectBivariateSpline 对象 lut，基于输入的 x, y 和 z
        lut = RectBivariateSpline(x, y, z)
        # 断言调用超出阶数限制的偏导数方法时，会引发 ValueError 异常
        with assert_raises(ValueError):
            lut.partial_derivative(4, 1)

    # 定义测试函数，用于测试 RectBivariateSpline 类的广播特性
    def test_broadcast(self):
        # 定义输入变量 x, y 和二维数组 z
        x = array([1,2,3,4,5])
        y = array([1,2,3,4,5])
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        # 创建 RectBivariateSpline 对象 lut，基于输入的 x, y 和 z
        lut = RectBivariateSpline(x,y,z)
        # 断言 lut 对象在不同形状的输入上的结果近似相等
        assert_allclose(lut(x, y), lut(x[:,None], y[None,:], grid=False))
    # 定义一个测试函数来测试 RectBivariateSpline 类在不合法输入情况下的行为
    def test_invalid_input(self):

        # 测试当 x 数组不是严格递增时，是否抛出 ValueError 异常，并验证异常信息
        with assert_raises(ValueError) as info:
            x = array([6, 2, 3, 4, 5])  # x 数组非严格递增
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                       [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert "x must be strictly increasing" in str(info.value)

        # 测试当 y 数组不是严格递增时，是否抛出 ValueError 异常，并验证异常信息
        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([2, 2, 3, 4, 5])  # y 数组非严格递增
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                       [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert "y must be strictly increasing" in str(info.value)

        # 测试当 z 的行数与 x 数组长度不一致时，是否抛出 ValueError 异常，并验证异常信息
        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3, 2],  # z 的行数与 x 数组长度不一致
                       [1, 2, 2, 2], [1, 2, 1, 2]])
            RectBivariateSpline(x, y, z)
        assert "x dimension of z must have same number of elements as x"\
               in str(info.value)

        # 测试当 z 的列数与 y 数组长度不一致时，是否抛出 ValueError 异常，并验证异常信息
        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                       [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert "y dimension of z must have same number of elements as y"\
               in str(info.value)

        # 测试当 bbox 的形状不是 (4,) 时，是否抛出 ValueError 异常，并验证异常信息
        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                       [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            bbox = (-100, 100, -100)  # bbox 形状不是 (4,)
            RectBivariateSpline(x, y, z, bbox=bbox)
        assert "bbox shape should be (4,)" in str(info.value)

        # 测试当 s 参数小于 0 时，是否抛出 ValueError 异常，并验证异常信息
        with assert_raises(ValueError) as info:
            RectBivariateSpline(x, y, z, s=-1.0)  # s 参数小于 0
        assert "s should be s >= 0.0" in str(info.value)

    # 定义一个测试函数来测试 RectBivariateSpline 类在数组样式输入情况下的行为
    def test_array_like_input(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                   [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        bbox = array([1, 5, 1, 5])

        # 使用数组类型的 x, y, z 和 bbox 来创建 RectBivariateSpline 实例 spl1
        spl1 = RectBivariateSpline(x, y, z, bbox=bbox)

        # 将 x, y, z 和 bbox 转换为列表形式，然后使用它们来创建 RectBivariateSpline 实例 spl2
        spl2 = RectBivariateSpline(x.tolist(), y.tolist(), z.tolist(),
                                   bbox=bbox.tolist())

        # 断言 spl1 和 spl2 在给定输入 (1.0, 1.0) 时的返回值近似相等
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))
    def test_not_increasing_input(self):
        # 定义一个测试函数，用于验证插值器的输入是否严格递增
        # 设定样本数量
        NSamp = 20
        # 在区间 [0, π) 内生成均匀分布的角度
        Theta = np.random.uniform(0, np.pi, NSamp)
        # 在区间 [0, 2π) 内生成均匀分布的角度
        Phi = np.random.uniform(0, 2 * np.pi, NSamp)
        # 创建长度为 NSamp 的全为 1 的数据数组
        Data = np.ones(NSamp)

        # 使用 SmoothSphereBivariateSpline 创建球面双变量插值器，平滑度参数 s=3.5
        Interpolator = SmoothSphereBivariateSpline(Theta, Phi, Data, s=3.5)

        # 网格经度和纬度的数量
        NLon = 6
        NLat = 3
        # 在 [0, π] 区间内生成 NLat 个纬度位置
        GridPosLats = np.arange(NLat) / NLat * np.pi
        # 在 [0, 2π] 区间内生成 NLon 个经度位置
        GridPosLons = np.arange(NLon) / NLon * 2 * np.pi

        # 用插值器对网格经纬度位置进行插值，不应引发错误
        Interpolator(GridPosLats, GridPosLons)

        # 复制纬度数组以创建非严格递增的情况
        nonGridPosLats = GridPosLats.copy()
        nonGridPosLats[2] = 0.001
        # 验证插值器对非严格递增的纬度数组是否会引发 ValueError 异常
        with assert_raises(ValueError) as exc_info:
            Interpolator(nonGridPosLats, GridPosLons)
        assert "x must be strictly increasing" in str(exc_info.value)

        # 复制经度数组以创建非严格递增的情况
        nonGridPosLons = GridPosLons.copy()
        nonGridPosLons[2] = 0.001
        # 验证插值器对非严格递增的经度数组是否会引发 ValueError 异常
        with assert_raises(ValueError) as exc_info:
            Interpolator(GridPosLats, nonGridPosLons)
        assert "y must be strictly increasing" in str(exc_info.value)
class TestRectSphereBivariateSpline:
    # 测试默认参数情况下的功能
    def test_defaults(self):
        # 在指定范围内生成等间距的7个数作为y坐标
        y = linspace(0.01, 2*pi-0.01, 7)
        # 在指定范围内生成等间距的7个数作为x坐标
        x = linspace(0.01, pi-0.01, 7)
        # 创建一个二维数组作为z值，用于插值
        z = array([[1,2,1,2,1,2,1],[1,2,1,2,1,2,1],[1,2,3,2,1,2,1],
                   [1,2,2,2,1,2,1],[1,2,1,2,1,2,1],[1,2,2,2,1,2,1],
                   [1,2,1,2,1,2,1]])
        # 创建 RectSphereBivariateSpline 对象
        lut = RectSphereBivariateSpline(x,y,z)
        # 断言插值函数在给定点上的结果与预期的z值数组相近
        assert_array_almost_equal(lut(x,y),z)

    # 测试插值函数的评估功能
    def test_evaluate(self):
        # 在指定范围内生成等间距的7个数作为y坐标
        y = linspace(0.01, 2*pi-0.01, 7)
        # 在指定范围内生成等间距的7个数作为x坐标
        x = linspace(0.01, pi-0.01, 7)
        # 创建一个二维数组作为z值，用于插值
        z = array([[1,2,1,2,1,2,1],[1,2,1,2,1,2,1],[1,2,3,2,1,2,1],
                   [1,2,2,2,1,2,1],[1,2,1,2,1,2,1],[1,2,2,2,1,2,1],
                   [1,2,1,2,1,2,1]])
        # 创建 RectSphereBivariateSpline 对象
        lut = RectSphereBivariateSpline(x,y,z)
        # 定义一组要评估的插值点
        yi = [0.2, 1, 2.3, 2.35, 3.0, 3.99, 5.25]
        xi = [1.5, 0.4, 1.1, 0.45, 0.2345, 1., 0.0001]
        # 使用插值对象评估给定点的插值结果
        zi = lut.ev(xi, yi)
        # 使用另一种方法计算每个点的插值结果
        zi2 = array([lut(xp, yp)[0,0] for xp, yp in zip(xi, yi)])
        # 断言两种方法得到的插值结果应该相近
        assert_almost_equal(zi, zi2)

    # 测试插值函数对无效输入的处理
    def test_invalid_input(self):
        # 创建一个非法输入的数据数组
        data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
                      np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

        # 检查对于超出范围的经度的处理
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(-1, 170, 9) * np.pi / 180.
            lons = np.linspace(0, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "u should be between (0, pi)" in str(exc_info.value)

        # 检查对于超出范围的纬度的处理
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 181, 9) * np.pi / 180.
            lons = np.linspace(0, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "u should be between (0, pi)" in str(exc_info.value)

        # 检查对于超出范围的第一个纬度的处理
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(-181, 10, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "v[0] should be between [-pi, pi)" in str(exc_info.value)

        # 检查对于超出范围的最后一个纬度的处理
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(-10, 360, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "v[-1] should be v[0] + 2pi or less" in str(exc_info.value)

        # 检查对于负的平滑因子的处理
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(10, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data, s=-1)
        assert "s should be positive" in str(exc_info.value)
    # 定义一个测试方法，用于测试 RectSphereBivariateSpline 类的导数计算功能
    def test_derivatives_grid(self):
        # 生成一个包含 7 个元素的等间距数组，范围从 0.01 到 2*pi-0.01
        y = linspace(0.01, 2*pi-0.01, 7)
        # 生成一个包含 7 个元素的等间距数组，范围从 0.01 到 pi-0.01
        x = linspace(0.01, pi-0.01, 7)
        # 创建一个 7x7 的二维数组，表示在球面上的函数值
        z = array([[1,2,1,2,1,2,1],[1,2,1,2,1,2,1],[1,2,3,2,1,2,1],
                   [1,2,2,2,1,2,1],[1,2,1,2,1,2,1],[1,2,2,2,1,2,1],
                   [1,2,1,2,1,2,1]])

        # 使用 RectSphereBivariateSpline 类初始化一个插值对象
        lut = RectSphereBivariateSpline(x, y, z)

        # 更新 y 数组的值，生成一个包含 7 个元素的等间距数组，范围从 0.02 到 2*pi-0.02
        y = linspace(0.02, 2*pi-0.02, 7)
        # 更新 x 数组的值，生成一个包含 7 个元素的等间距数组，范围从 0.02 到 pi-0.02

        # 使用插值对象 lut 计算在给定网格上 dtheta=1 的偏导数，并与数值偏导数结果 _numdiff_2d 进行比较
        assert_allclose(lut(x, y, dtheta=1), _numdiff_2d(lut, x, y, dx=1),
                        rtol=1e-4, atol=1e-4)
        # 使用插值对象 lut 计算在给定网格上 dphi=1 的偏导数，并与数值偏导数结果 _numdiff_2d 进行比较
        assert_allclose(lut(x, y, dphi=1), _numdiff_2d(lut, x, y, dy=1),
                        rtol=1e-4, atol=1e-4)
        # 使用插值对象 lut 计算在给定网格上 dtheta=1 和 dphi=1 的偏导数，并与数值偏导数结果 _numdiff_2d 进行比较
        assert_allclose(lut(x, y, dtheta=1, dphi=1),
                        _numdiff_2d(lut, x, y, dx=1, dy=1, eps=1e-6),
                        rtol=1e-3, atol=1e-3)

        # 使用插值对象 lut 计算在给定网格上 dtheta=1 的偏导数，与 lut.partial_derivative(1, 0)(x, y) 进行比较
        assert_array_equal(lut(x, y, dtheta=1),
                           lut.partial_derivative(1, 0)(x, y))
        # 使用插值对象 lut 计算在给定网格上 dphi=1 的偏导数，与 lut.partial_derivative(0, 1)(x, y) 进行比较
        assert_array_equal(lut(x, y, dphi=1),
                           lut.partial_derivative(0, 1)(x, y))
        # 使用插值对象 lut 计算在给定网格上 dtheta=1 和 dphi=1 的偏导数，与 lut.partial_derivative(1, 1)(x, y) 进行比较
        assert_array_equal(lut(x, y, dtheta=1, dphi=1),
                           lut.partial_derivative(1, 1)(x, y))

        # 使用插值对象 lut 计算在给定网格上 dtheta=1 的偏导数，不使用网格，与 lut.partial_derivative(1, 0)(x, y, grid=False) 进行比较
        assert_array_equal(lut(x, y, dtheta=1, grid=False),
                           lut.partial_derivative(1, 0)(x, y, grid=False))
        # 使用插值对象 lut 计算在给定网格上 dphi=1 的偏导数，不使用网格，与 lut.partial_derivative(0, 1)(x, y, grid=False) 进行比较
        assert_array_equal(lut(x, y, dphi=1, grid=False),
                           lut.partial_derivative(0, 1)(x, y, grid=False))
        # 使用插值对象 lut 计算在给定网格上 dtheta=1 和 dphi=1 的偏导数，不使用网格，与 lut.partial_derivative(1, 1)(x, y, grid=False) 进行比较
        assert_array_equal(lut(x, y, dtheta=1, dphi=1, grid=False),
                           lut.partial_derivative(1, 1)(x, y, grid=False))

    # 定义一个测试方法，用于测试 RectSphereBivariateSpline 类的偏导数功能
    def test_derivatives(self):
        # 生成一个包含 7 个元素的等间距数组，范围从 0.01 到 2*pi-0.01
        y = linspace(0.01, 2*pi-0.01, 7)
        # 生成一个包含 7 个元素的等间距数组，范围从 0.01 到 pi-0.01
        x = linspace(0.01, pi-0.01, 7)
        # 创建一个 7x7 的二维数组，表示在球面上的函数值
        z = array([[1,2,1,2,1,2,1],[1,2,1,2,1,2,1],[1,2,3,2,1,2,1],
                   [1,2,2,2,1,2,1],[1,2,1,2,1,2,1],[1,2,2,2,1,2,1],
                   [1,2,1,2,1,2,1]])

        # 使用 RectSphereBivariateSpline 类初始化一个插值对象
        lut = RectSphereBivariateSpline(x, y, z)

        # 更新 y 数组的值，生成一个包含 7 个元素的等间距数组，范围从 0.02 到 2*pi-0.02
        y = linspace(0.02, 2*pi-0.02, 7)
        # 更新 x 数组的值，生成一个包含 7 个元素的等间距数组，范围从 0.02 到 pi-0.02

        # 使用插值对象 lut 计算在给定网格上 dtheta=1 的偏导数，不使用网格，并验证其形状与 x 数组一致
        assert_equal(lut(x, y, dtheta=1, grid=False).shape, x.shape)
        # 使用插值对象 lut 计算在给定网格上 dtheta=1 的偏导数，不使用网格，并与数值偏导数结果 _numdiff_2d 进行比较
        assert_allclose(lut(x, y, dtheta=1, grid=False),
                        _numdiff_2d(lambda x,y: lut(x,y,grid=False), x, y, dx=1),
                        rtol=1e-4, atol=1e-4)
        # 使用插值对象 lut 计算在给定网格上 dphi=1 的偏导数，不使用网格，并与数值偏导数结果 _numdiff_2d 进行比较
        assert_allclose(lut(x, y, dphi=1, grid=False),
                        _numdiff_2d(lambda x,y: lut(x,y,grid=False), x, y, dy=1),
                        rtol=1e-4, atol=1e-4)
        # 使用插值对象 lut 计算在给定网格上 dtheta=1 和 dphi=1 的偏导数，不使用网格，并与数值偏导数结果 _numdiff_2d 进行比较
        assert_allclose(lut(x, y, dtheta=1, dphi=1, grid=False),
                        _
    # 测试无效输入情况，检查是否正确抛出 ValueError 异常
    def test_invalid_input_2(self):
        # 创建一个二维数组 data，计算两个向量的点乘结果
        data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
                      np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

        # 检查 RectSphereBivariateSpline 对于不同的经纬度数据抛出 ValueError 异常
        with assert_raises(ValueError) as exc_info:
            # 第一次测试，设置纬度在 (0, 170) 之间，经度在 (0, 350) 之间
            lats = np.linspace(0, 170, 9) * np.pi / 180.
            lons = np.linspace(0, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        # 断言异常信息包含 "u should be between (0, pi)"
        assert "u should be between (0, pi)" in str(exc_info.value)

        # 第二次测试，设置纬度在 (10, 180) 之间，经度在 (0, 350) 之间
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 180, 9) * np.pi / 180.
            lons = np.linspace(0, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "u should be between (0, pi)" in str(exc_info.value)

        # 第三次测试，设置纬度在 (10, 170) 之间，经度在 (-181, 10) 之间
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(-181, 10, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "v[0] should be between [-pi, pi)" in str(exc_info.value)

        # 第四次测试，设置纬度在 (10, 170) 之间，经度在 (-10, 360) 之间
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(-10, 360, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "v[-1] should be v[0] + 2pi or less" in str(exc_info.value)

        # 第五次测试，设置纬度在 (10, 170) 之间，经度在 (10, 350) 之间，同时指定 s=-1
        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(10, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data, s=-1)
        assert "s should be positive" in str(exc_info.value)

    # 测试数组样式的输入
    def test_array_like_input(self):
        # 创建均匀分布的数组 y, x, z
        y = linspace(0.01, 2 * pi - 0.01, 7)
        x = linspace(0.01, pi - 0.01, 7)
        z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1],
                   [1, 2, 3, 2, 1, 2, 1],
                   [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1],
                   [1, 2, 2, 2, 1, 2, 1],
                   [1, 2, 1, 2, 1, 2, 1]])
        # 使用 np.array 样式的输入创建 RectSphereBivariateSpline 对象
        spl1 = RectSphereBivariateSpline(x, y, z)
        # 使用列表样式的输入创建 RectSphereBivariateSpline 对象
        spl2 = RectSphereBivariateSpline(x.tolist(), y.tolist(), z.tolist())
        # 断言两个对象在给定的 (x, y) 值上的近似相等性
        assert_array_almost_equal(spl1(x, y), spl2(x, y))

    # 测试负值评估情况
    def test_negative_evaluation(self):
        # 创建纬度和经度的 np.array 输入
        lats = np.array([25, 30, 35, 40, 45])
        lons = np.array([-90, -85, -80, -75, 70])
        # 创建网格
        mesh = np.meshgrid(lats, lons)
        # 计算网格上的 lon + lat 值
        data = mesh[0] + mesh[1]  # lon + lat value
        # 将纬度和经度转换为弧度
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        # 创建 RectSphereBivariateSpline 插值器
        interpolator = RectSphereBivariateSpline(lat_r, lon_r, data)
        # 查询的纬度和经度的值，转换为弧度
        query_lat = np.radians(np.array([35, 37.5]))
        query_lon = np.radians(np.array([-80, -77.5]))
        # 对查询点进行插值
        data_interp = interpolator(query_lat, query_lon)
        # 预期的插值结果
        ans = np.array([[-45.0, -42.480862],
                        [-49.0625, -46.54315]])
        # 断言插值结果与预期结果的近似性
        assert_array_almost_equal(data_interp, ans)
    # 定义一个测试方法，用于回归测试 https://github.com/scipy/scipy/issues/14591
    def test_pole_continuity_gh_14591(self):
        # 指出这个测试方法是为了解决 GitHub 问题号 14591
        # 当 pole_continuty=(True, True) 时，内部工作数组大小过小，
        # 导致 FITPACK 数据验证错误。

        # 在 gh-14591 中的重现案例使用了一个具有 361x507 的 NetCDF4 文件，
        # 这里我们将数组大小简化到最小，仍能展示问题。
        u = np.arange(1, 10) * np.pi / 10  # 创建一个包含 9 个元素的数组 u
        v = np.arange(1, 10) * np.pi / 10  # 创建一个包含 9 个元素的数组 v
        r = np.zeros((9, 9))  # 创建一个 9x9 的零矩阵 r
        for p in [(True, True), (True, False), (False, False)]:
            # 对于三种不同的 pole_continuity 参数组合，创建 RectSphereBivariateSpline 对象
            RectSphereBivariateSpline(u, v, r, s=0, pole_continuity=p)
# 定义一个用于计算二维函数偏导数的函数
def _numdiff_2d(func, x, y, dx=0, dy=0, eps=1e-8):
    # 如果 dx 和 dy 都为 0，直接计算函数在点 (x, y) 处的值并返回
    if dx == 0 and dy == 0:
        return func(x, y)
    # 如果 dx 为 1，dy 为 0，使用中心差分法计算函数在点 (x, y) 处 x 方向的偏导数
    elif dx == 1 and dy == 0:
        return (func(x + eps, y) - func(x - eps, y)) / (2*eps)
    # 如果 dx 为 0，dy 为 1，使用中心差分法计算函数在点 (x, y) 处 y 方向的偏导数
    elif dx == 0 and dy == 1:
        return (func(x, y + eps) - func(x, y - eps)) / (2*eps)
    # 如果 dx 和 dy 都为 1，使用中心差分法计算函数在点 (x, y) 处的二阶混合偏导数
    elif dx == 1 and dy == 1:
        return (func(x + eps, y + eps) - func(x - eps, y + eps)
                - func(x + eps, y - eps) + func(x - eps, y - eps)) / (2*eps)**2
    else:
        # 如果 dx 或 dy 不在允许的范围内（只能是 0 或 1），则抛出值错误异常
        raise ValueError("invalid derivative order")


class Test_DerivedBivariateSpline:
    """Test the creation, usage, and attribute access of the (private)
    _DerivedBivariateSpline class.
    """
    # 在每个测试方法运行之前设置测试环境
    def setup_method(self):
        # 构造 x, y, z 用于不同的插值类的测试
        x = np.concatenate(list(zip(range(10), range(10))))
        y = np.concatenate(list(zip(range(10), range(1, 11))))
        z = np.concatenate((np.linspace(3, 1, 10), np.linspace(1, 3, 10)))
        # 屏蔽警告并记录特定警告类型的消息
        with suppress_warnings() as sup:
            sup.record(UserWarning, "\nThe coefficients of the spline")
            # 使用 LSQBivariateSpline 进行双变量插值，并存储为 lut_lsq 属性
            self.lut_lsq = LSQBivariateSpline(x, y, z,
                                              linspace(0.5, 19.5, 4),
                                              linspace(1.5, 20.5, 4),
                                              eps=1e-2)
        # 使用 SmoothBivariateSpline 进行双变量插值，并存储为 lut_smooth 属性
        self.lut_smooth = SmoothBivariateSpline(x, y, z)
        # 使用 RectBivariateSpline 进行双变量插值，并存储为 lut_rect 属性
        xx = linspace(0, 1, 20)
        yy = xx + 1.0
        zz = array([np.roll(z, i) for i in range(z.size)])
        self.lut_rect = RectBivariateSpline(xx, yy, zz)
        # 创建一个包含 (0,0), (0,1), (0,2), ..., (2,2) 的所有排列的列表，并存储为 orders 属性
        self.orders = list(itertools.product(range(3), range(3)))

    # 测试从 LSQ 插值对象创建导数
    def test_creation_from_LSQ(self):
        for nux, nuy in self.orders:
            # 获取 LSQ 插值对象的部分导数对象
            lut_der = self.lut_lsq.partial_derivative(nux, nuy)
            # 检查部分导数对象在特定点 (3.5, 3.5) 的值是否等于通过方法调用获取的值
            a = lut_der(3.5, 3.5, grid=False)
            b = self.lut_lsq(3.5, 3.5, dx=nux, dy=nuy, grid=False)
            assert_equal(a, b)

    # 测试从 Smooth 插值对象创建导数
    def test_creation_from_Smooth(self):
        for nux, nuy in self.orders:
            # 获取 Smooth 插值对象的部分导数对象
            lut_der = self.lut_smooth.partial_derivative(nux, nuy)
            # 检查部分导数对象在特定点 (5.5, 5.5) 的值是否等于通过方法调用获取的值
            a = lut_der(5.5, 5.5, grid=False)
            b = self.lut_smooth(5.5, 5.5, dx=nux, dy=nuy, grid=False)
            assert_equal(a, b)

    # 测试从 Rect 插值对象创建导数
    def test_creation_from_Rect(self):
        for nux, nuy in self.orders:
            # 获取 Rect 插值对象的部分导数对象
            lut_der = self.lut_rect.partial_derivative(nux, nuy)
            # 检查部分导数对象在特定点 (0.5, 1.5) 的值是否等于通过方法调用获取的值
            a = lut_der(0.5, 1.5, grid=False)
            b = self.lut_rect(0.5, 1.5, dx=nux, dy=nuy, grid=False)
            assert_equal(a, b)

    # 测试从 Smooth 插值对象获取不存在的属性时是否引发 AttributeError
    def test_invalid_attribute_fp(self):
        # 获取 Smooth 插值对象的部分导数对象
        der = self.lut_rect.partial_derivative(1, 1)
        # 检查访问不存在的属性 'fp' 是否会引发 AttributeError 异常
        with assert_raises(AttributeError):
            der.fp

    # 测试从 Rect 插值对象获取不存在的属性时是否引发 AttributeError
    def test_invalid_attribute_get_residual(self):
        # 获取 Rect 插值对象的部分导数对象
        der = self.lut_smooth.partial_derivative(1, 1)
        # 检查调用不存在的方法 'get_residual' 是否会引发 AttributeError 异常
        with assert_raises(AttributeError):
            der.get_residual()
```