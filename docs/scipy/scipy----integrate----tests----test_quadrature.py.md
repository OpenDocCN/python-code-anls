# `D:\src\scipysrc\scipy\scipy\integrate\tests\test_quadrature.py`

```
# 引入依赖库 pytest、numpy、assert_equal、assert_almost_equal、assert_allclose
# 以及 hypothesis 中的 given 和 strategies 模块
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num

# 从 scipy.integrate 中导入多个积分函数
from scipy.integrate import (romb, newton_cotes,
                             cumulative_trapezoid, trapezoid,
                             quad, simpson, fixed_quad,
                             qmc_quad, cumulative_simpson)
# 从 scipy.integrate._quadrature 中导入 _cumulative_simpson_unequal_intervals
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
# 从 scipy 中导入 stats 和 special 模块
from scipy import stats, special

# 定义一个测试类 TestFixedQuad，用于测试 fixed_quad 函数
class TestFixedQuad:
    def test_scalar(self):
        n = 4
        expected = 1/(2*n)
        # 调用 fixed_quad 对 lambda 函数进行数值积分
        got, _ = fixed_quad(lambda x: x**(2*n - 1), 0, 1, n=n)
        # 检查数值积分结果是否接近预期值
        assert_allclose(got, expected, rtol=1e-12)

    def test_vector(self):
        n = 4
        p = np.arange(1, 2*n)
        expected = 1/(p + 1)
        # 调用 fixed_quad 对 lambda 函数进行数值积分
        got, _ = fixed_quad(lambda x: x**p[:, None], 0, 1, n=n)
        # 检查数值积分结果是否接近预期值
        assert_allclose(got, expected, rtol=1e-12)

# 定义一个测试类 TestQuadrature，用于测试不同的数值积分函数
class TestQuadrature:
    # 定义一个未实现的方法 quad，用于测试重写时是否抛出 NotImplementedError
    def quad(self, x, a, b, args):
        raise NotImplementedError

    def test_romb(self):
        # 测试 romb 函数，检查是否正确计算
        assert_equal(romb(np.arange(17)), 128)

    def test_romb_gh_3731(self):
        # 检查 romb 函数是否最大程度利用了数据点
        x = np.arange(2**4+1)
        y = np.cos(0.2*x)
        val = romb(y)
        # 对比 romb 和 quad 的结果是否接近
        val2, err = quad(lambda x: np.cos(0.2*x), x.min(), x.max())
        assert_allclose(val, val2, rtol=1e-8, atol=0)

    def test_newton_cotes(self):
        """Test the first few degrees, for evenly spaced points."""
        n = 1
        # 测试 newton_cotes 函数的几个阶数，对均匀分布的点进行测试
        wts, errcoff = newton_cotes(n, 1)
        assert_equal(wts, n*np.array([0.5, 0.5]))
        assert_almost_equal(errcoff, -n**3/12.0)

        n = 2
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([1.0, 4.0, 1.0])/6.0)
        assert_almost_equal(errcoff, -n**5/2880.0)

        n = 3
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([1.0, 3.0, 3.0, 1.0])/8.0)
        assert_almost_equal(errcoff, -n**5/6480.0)

        n = 4
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([7.0, 32.0, 12.0, 32.0, 7.0])/90.0)
        assert_almost_equal(errcoff, -n**7/1935360.0)

    def test_newton_cotes2(self):
        """Test newton_cotes with points that are not evenly spaced."""
        # 测试 newton_cotes 函数，使用非均匀分布的点进行测试

        x = np.array([0.0, 1.5, 2.0])
        y = x**2
        wts, errcoff = newton_cotes(x)
        exact_integral = 8.0/3
        numeric_integral = np.dot(wts, y)
        assert_almost_equal(numeric_integral, exact_integral)

        x = np.array([0.0, 1.4, 2.1, 3.0])
        y = x**2
        wts, errcoff = newton_cotes(x)
        exact_integral = 9.0
        numeric_integral = np.dot(wts, y)
        assert_almost_equal(numeric_integral, exact_integral)
    # 定义一个名为 test_simpson 的测试方法，用于测试 simpson 函数的不同用例
    def test_simpson(self):
        # 创建一个长度为17的 numpy 数组 y，包含从0到16的整数
        y = np.arange(17)
        # 断言调用 simpson 函数后的结果应该等于 128
        assert_equal(simpson(y), 128)
        # 断言调用 simpson 函数后的结果应该等于 64，设置 dx 参数为 0.5
        assert_equal(simpson(y, dx=0.5), 64)
        # 断言调用 simpson 函数后的结果应该等于 32，设置 x 参数为从0到4均匀分布的长度为17的数组
        assert_equal(simpson(y, x=np.linspace(0, 4, 17)), 32)

        # 定义一个函数 f(x)，计算 x 的平方
        # 函数定义在 test_simpson 方法内部，作用域仅限于该方法
        def f(x):
            return x**2
        
        # 断言调用 simpson 函数后的结果应该接近于 21.0，设置 x 参数为从1到4均匀分布的长度为4的数组
        assert_allclose(simpson(f(x), x=x), 21.0)

        # 断言调用 simpson 函数后的结果应该接近于 114，设置 dx 参数为 2.0
        assert_allclose(simpson(f(x), dx=2.0), 114)

        # 测试多轴行为
        # 创建一个4x4的 numpy 数组 a 和一个4x4x4的 numpy 数组 x
        a = np.arange(16).reshape(4, 4)
        x = np.arange(64.).reshape(4, 4, 4)
        # 计算 f(x) 的值
        y = f(x)
        # 遍历轴的索引 i 为 0、1、2
        for i in range(3):
            # 调用 simpson 函数计算在特定轴 i 上的积分结果 r
            r = simpson(y, x=x, axis=i)
            # 创建一个多维迭代器 it，用于遍历数组 a 的元素
            it = np.nditer(a, flags=['multi_index'])
            for _ in it:
                # 将当前多维索引转换为列表 idx，并在索引 i 的位置插入 slice(None)
                idx = list(it.multi_index)
                idx.insert(i, slice(None))
                # 计算积分的理论值 integral
                integral = x[tuple(idx)][-1]**3 / 3 - x[tuple(idx)][0]**3 / 3
                # 断言 simpson 函数计算得到的结果 r 和理论值 integral 接近
                assert_allclose(r[it.multi_index], integral)

        # 测试积分轴只有两个点的情况
        # 创建一个8x2的 numpy 数组 x
        x = np.arange(16).reshape(8, 2)
        # 计算 f(x) 的值
        y = f(x)
        # 调用 simpson 函数计算在轴 -1 上的积分结果 r
        r = simpson(y, x=x, axis=-1)
        # 计算积分的理论值 integral
        integral = 0.5 * (y[:, 1] + y[:, 0]) * (x[:, 1] - x[:, 0])
        # 断言 simpson 函数计算得到的结果 r 和理论值 integral 接近
        assert_allclose(r, integral)

        # 测试奇数点数情况下的多轴行为
        # 创建一个5x5的 numpy 数组 a 和一个5x5x5的 numpy 数组 x
        a = np.arange(25).reshape(5, 5)
        x = np.arange(125).reshape(5, 5, 5)
        # 计算 f(x) 的值
        y = f(x)
        # 遍历轴的索引 i 为 0、1、2
        for i in range(3):
            # 调用 simpson 函数计算在特定轴 i 上的积分结果 r
            r = simpson(y, x=x, axis=i)
            # 创建一个多维迭代器 it，用于遍历数组 a 的元素
            it = np.nditer(a, flags=['multi_index'])
            for _ in it:
                # 将当前多维索引转换为列表 idx，并在索引 i 的位置插入 slice(None)
                idx = list(it.multi_index)
                idx.insert(i, slice(None))
                # 计算积分的理论值 integral
                integral = x[tuple(idx)][-1]**3 / 3 - x[tuple(idx)][0]**3 / 3
                # 断言 simpson 函数计算得到的结果 r 和理论值 integral 接近
                assert_allclose(r[it.multi_index], integral)

        # 检查基本情况的测试
        # 创建一个长度为1的 numpy 数组 x 和相应的 y 值
        x = np.array([3])
        y = np.power(x, 2)
        # 断言调用 simpson 函数后的结果应该接近于 0.0
        assert_allclose(simpson(y, x=x, axis=0), 0.0)
        assert_allclose(simpson(y, x=x, axis=-1), 0.0)

        # 创建一个长度为4的 numpy 数组 x 和相应的 y 值
        x = np.array([3, 3, 3, 3])
        y = np.power(x, 2)
        # 断言调用 simpson 函数后的结果应该接近于 0.0
        assert_allclose(simpson(y, x=x, axis=0), 0.0)
        assert_allclose(simpson(y, x=x, axis=-1), 0.0)

        # 创建一个3x4的 numpy 数组 x 和相应的 y 值
        x = np.array([[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]])
        y = np.power(x, 2)
        # 创建一个长度为4的 zero_axis 数组，每个元素为 0.0
        zero_axis = [0.0, 0.0, 0.0, 0.0]
        # 创建一个长度为3的 default_axis 数组，每个元素为 170 + 1/3
        default_axis = [170 + 1/3] * 3   # 8**3 / 3 - 1/3
        # 断言调用 simpson 函数后的结果应该接近于 zero_axis
        assert_allclose(simpson(y, x=x, axis=0), zero_axis)
        # 断言调用 simpson 函数后的结果应该精确等于 default_axis
        assert_allclose(simpson(y, x=x, axis=-1), default_axis)

        # 创建一个3x4的 numpy 数组 x 和相应的 y 值
        x = np.array([[1, 2, 4, 8], [1, 2, 4, 8], [1, 8, 16, 32]])
        y = np.power(x, 2)
        # 创建一个长度为4的 zero_axis 数组，每个元素为对应的积分结果
        zero_axis = [0.0, 136.0, 1088.0, 8704.0]
        # 创建一个长度为3的 default_axis 数组，每个元素为对应的积分结果
        default_axis = [170 + 1/3, 170 + 1/3, 32**3 / 3 - 1/3]
        # 断言调用 simpson 函数后的结果应该接近于 zero_axis
        assert_allclose(simpson(y, x=x, axis=0), zero_axis)
        # 断言调用 simpson 函数后的结果应该接近于 default_axis
        assert_allclose(simpson(y, x=x, axis=-1), default_axis)
    # 定义测试函数 test_simpson_2d_integer_no_x，用于测试 Simpson 积分计算函数对于二维整数数组的行为
    def test_simpson_2d_integer_no_x(self, droplast):
        # 输入是二维整数数组。当输入为浮点数时，期望结果应该完全相同。
        y = np.array([[2, 2, 4, 4, 8, 8, -4, 5],
                      [4, 4, 2, -4, 10, 22, -2, 10]])
        # 如果 droplast 参数为真，则删除每行的最后一个元素
        if droplast:
            y = y[:, :-1]
        # 调用 simpson 函数计算在最后一个轴（即轴=-1）上的 Simpson 积分
        result = simpson(y, axis=-1)
        # 使用浮点数类型计算期望的 Simpson 积分结果
        expected = simpson(np.array(y, dtype=np.float64), axis=-1)
        # 使用 assert_equal 断言确保计算结果与期望结果相等
        assert_equal(result, expected)
class TestCumulative_trapezoid:
    # 测试一维情况下的累积梯形积分计算
    def test_1d(self):
        # 生成等间距的五个数作为 x 值
        x = np.linspace(-2, 2, num=5)
        # y 值与 x 值相同
        y = x
        # 计算累积梯形积分，初始值为 0
        y_int = cumulative_trapezoid(y, x, initial=0)
        # 预期的累积梯形积分结果
        y_expected = [0., -1.5, -2., -1.5, 0.]
        # 断言计算结果与预期结果的接近程度
        assert_allclose(y_int, y_expected)

        # 以 None 作为初始值再次计算累积梯形积分
        y_int = cumulative_trapezoid(y, x, initial=None)
        # 断言计算结果与预期结果的接近程度，忽略首个元素
        assert_allclose(y_int, y_expected[1:])

    # 测试 y 和 x 都是多维数组的情况下的累积梯形积分计算
    def test_y_nd_x_nd(self):
        # 生成一个形状为 (3, 2, 4) 的三维数组作为 x 和 y
        x = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        y = x
        # 计算累积梯形积分，初始值为 0
        y_int = cumulative_trapezoid(y, x, initial=0)
        # 预期的累积梯形积分结果
        y_expected = np.array([[[0., 0.5, 2., 4.5],
                                [0., 4.5, 10., 16.5]],
                               [[0., 8.5, 18., 28.5],
                                [0., 12.5, 26., 40.5]],
                               [[0., 16.5, 34., 52.5],
                                [0., 20.5, 42., 64.5]]])

        # 断言计算结果与预期结果的接近程度
        assert_allclose(y_int, y_expected)

        # 使用所有轴进行进一步测试
        shapes = [(2, 2, 4), (3, 1, 4), (3, 2, 3)]
        for axis, shape in zip([0, 1, 2], shapes):
            # 根据指定的轴计算累积梯形积分，初始值为 0
            y_int = cumulative_trapezoid(y, x, initial=0, axis=axis)
            # 断言结果的形状符合预期
            assert_equal(y_int.shape, (3, 2, 4))
            # 根据指定的轴计算累积梯形积分，初始值为 None
            y_int = cumulative_trapezoid(y, x, initial=None, axis=axis)
            # 断言结果的形状符合预期
            assert_equal(y_int.shape, shape)

    # 测试 y 是多维数组且 x 是一维数组的情况下的累积梯形积分计算
    def test_y_nd_x_1d(self):
        # 生成一个形状为 (3, 2, 4) 的三维数组作为 y
        y = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        # 生成一个长度为 4 的一维数组作为 x
        x = np.arange(4)**2

        # 预期的累积梯形积分结果的集合
        ys_expected = (
            np.array([[[4., 5., 6., 7.],
                       [8., 9., 10., 11.]],
                      [[40., 44., 48., 52.],
                       [56., 60., 64., 68.]]]),
            np.array([[[2., 3., 4., 5.]],
                      [[10., 11., 12., 13.]],
                      [[18., 19., 20., 21.]]]),
            np.array([[[0.5, 5., 17.5],
                       [4.5, 21., 53.5]],
                      [[8.5, 37., 89.5],
                       [12.5, 53., 125.5]],
                      [[16.5, 69., 161.5],
                       [20.5, 85., 197.5]]]))

        for axis, y_expected in zip([0, 1, 2], ys_expected):
            # 根据指定的轴计算累积梯形积分，初始值为 None
            y_int = cumulative_trapezoid(y, x=x[:y.shape[axis]], axis=axis,
                                         initial=None)
            # 断言计算结果与预期结果的接近程度
            assert_allclose(y_int, y_expected)

    # 测试仅提供 y 值的情况下的累积梯形积分计算
    def test_x_none(self):
        # 生成等间距的五个数作为 y 值
        y = np.linspace(-2, 2, num=5)

        # 计算累积梯形积分
        y_int = cumulative_trapezoid(y)
        # 预期的累积梯形积分结果
        y_expected = [-1.5, -2., -1.5, 0.]
        # 断言计算结果与预期结果的接近程度
        assert_allclose(y_int, y_expected)

        # 计算累积梯形积分，初始值为 0
        y_int = cumulative_trapezoid(y, initial=0)
        # 预期的累积梯形积分结果
        y_expected = [0, -1.5, -2., -1.5, 0.]
        # 断言计算结果与预期结果的接近程度
        assert_allclose(y_int, y_expected)

        # 计算累积梯形积分，指定步长为 3
        y_int = cumulative_trapezoid(y, dx=3)
        # 预期的累积梯形积分结果
        y_expected = [-4.5, -6., -4.5, 0.]
        # 断言计算结果与预期结果的接近程度
        assert_allclose(y_int, y_expected)

        # 计算累积梯形积分，指定步长为 3，初始值为 0
        y_int = cumulative_trapezoid(y, dx=3, initial=0)
        # 预期的累积梯形积分结果
        y_expected = [0, -4.5, -6., -4.5, 0.]
        # 断言计算结果与预期结果的接近程度
        assert_allclose(y_int, y_expected)

    @pytest.mark.parametrize(
        "initial", [1, 0.5]
    )
    # 定义单元测试函数，测试 cumulative_trapezoid 函数在 initial 参数不为 None 或 0 时是否抛出 ValueError 异常
    def test_initial_error(self, initial):
        """If initial is not None or 0, a ValueError is raised."""
        # 创建一个包含 0 到 10 的等间距数组，共 10 个数
        y = np.linspace(0, 10, num=10)
        # 使用 pytest 模块验证 cumulative_trapezoid 函数在给定条件下是否抛出 ValueError 异常，匹配错误信息 "`initial`"
        with pytest.raises(ValueError, match="`initial`"):
            cumulative_trapezoid(y, initial=initial)
    
    # 定义单元测试函数，测试 cumulative_trapezoid 函数在 y 参数为空列表时是否抛出 ValueError 异常
    def test_zero_len_y(self):
        # 使用 pytest 模块验证 cumulative_trapezoid 函数在给定条件下是否抛出 ValueError 异常，匹配错误信息 "At least one point is required"
        with pytest.raises(ValueError, match="At least one point is required"):
            cumulative_trapezoid(y=[])
class TestTrapezoid:
    def test_simple(self):
        # 创建一个从 -10 到 10，步长为 0.1 的 NumPy 数组
        x = np.arange(-10, 10, .1)
        # 使用 trapezoid 函数计算 exp(-0.5 * x^2) / sqrt(2 * pi) 的积分，步长为 0.1
        r = trapezoid(np.exp(-.5 * x ** 2) / np.sqrt(2 * np.pi), dx=0.1)
        # 检查正态分布的积分是否等于 1
        assert_allclose(r, 1)

    def test_ndim(self):
        # 创建一维均匀间隔的三个 NumPy 数组
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 8)
        z = np.linspace(0, 3, 13)

        # 根据 x, y, z 的间隔创建权重数组 wx, wy, wz
        wx = np.ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = np.ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = np.ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2

        # 创建三维数组 q，表示 x[:, None, None] + y[None,:, None] + z[None, None,:]
        q = x[:, None, None] + y[None,:, None] + z[None, None,:]

        # 沿指定轴计算加权和 qx, qy, qz
        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)

        # 对于 n 维数组，使用 trapezoid 函数计算积分并检查结果是否接近 qx, qy, qz
        r = trapezoid(q, x=x[:, None, None], axis=0)
        assert_allclose(r, qx)
        r = trapezoid(q, x=y[None,:, None], axis=1)
        assert_allclose(r, qy)
        r = trapezoid(q, x=z[None, None,:], axis=2)
        assert_allclose(r, qz)

        # 对于一维数组，使用 trapezoid 函数计算积分并检查结果是否接近 qx, qy, qz
        r = trapezoid(q, x=x, axis=0)
        assert_allclose(r, qx)
        r = trapezoid(q, x=y, axis=1)
        assert_allclose(r, qy)
        r = trapezoid(q, x=z, axis=2)
        assert_allclose(r, qz)

    def test_masked(self):
        # 测试掩码数组的行为，掩码值对应的函数值应视为 0
        x = np.arange(5)
        y = x * x
        mask = x == 2
        ym = np.ma.array(y, mask=mask)
        r = 13.0  # sum(0.5 * (0 + 1) * 1.0 + 0.5 * (9 + 16))
        # 使用 trapezoid 函数计算掩码数组 ym 关于 x 的积分并检查结果是否接近 r
        assert_allclose(trapezoid(ym, x), r)

        xm = np.ma.array(x, mask=mask)
        # 使用 trapezoid 函数计算掩码数组 ym 关于掩码数组 xm 的积分并检查结果是否接近 r
        assert_allclose(trapezoid(ym, xm), r)

        xm = np.ma.array(x, mask=mask)
        # 使用 trapezoid 函数计算数组 y 关于掩码数组 xm 的积分并检查结果是否接近 r
        assert_allclose(trapezoid(y, xm), r)
    def basic_test(self, n_points=2**8, n_estimates=8, signs=np.ones(2)):
        # 定义测试函数的基本测试方法，接受三个参数：采样点数、估计数、符号数组
        ndim = 2
        mean = np.zeros(ndim)
        cov = np.eye(ndim)

        def func(x):
            # 定义被积函数，这里是多元正态分布的概率密度函数
            return stats.multivariate_normal.pdf(x.T, mean, cov)

        # 设置随机数生成器和QMCEngine实例
        rng = np.random.default_rng(2879434385674690281)
        qrng = stats.qmc.Sobol(ndim, seed=rng)
        
        # 设置积分的区间边界
        a = np.zeros(ndim)
        b = np.ones(ndim) * signs
        
        # 调用qmc_quad函数进行数值积分，得到积分结果res
        res = qmc_quad(func, a, b, n_points=n_points,
                       n_estimates=n_estimates, qrng=qrng)
        
        # 计算参考值ref，这里是多元正态分布的累积分布函数值
        ref = stats.multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        
        # 计算绝对误差容限atol，使用t分布的标准误差乘以置信区间的95%分位数
        atol = special.stdtrit(n_estimates-1, 0.995) * res.standard_error  # 99% CI
        
        # 断言数值积分结果res.integral接近于参考值ref，误差容限为atol
        assert_allclose(res.integral, ref, atol=atol)
        
        # 断言符号数组signs的乘积乘以积分结果res.integral大于0
        assert np.prod(signs)*res.integral > 0

        # 再次设置随机数生成器和QMCEngine实例
        rng = np.random.default_rng(2879434385674690281)
        qrng = stats.qmc.Sobol(ndim, seed=rng)
        
        # 计算对数积分结果logres，这里对函数func取对数后进行数值积分
        logres = qmc_quad(lambda *args: np.log(func(*args)), a, b,
                          n_points=n_points, n_estimates=n_estimates,
                          log=True, qrng=qrng)
        
        # 断言e的logres.integral次方接近于res.integral，相对误差为1e-14
        assert_allclose(np.exp(logres.integral), res.integral, rtol=1e-14)
        
        # 断言虚部（如果存在）与乘积signs的符号有关
        assert np.imag(logres.integral) == (np.pi if np.prod(signs) < 0 else 0)
        
        # 断言e的logres.standard_error次方接近于res.standard_error，相对误差为1e-14，绝对误差为1e-16
        assert_allclose(np.exp(logres.standard_error),
                        res.standard_error, rtol=1e-14, atol=1e-16)
    # 定义一个测试方法，用于执行基本测试，接受点数和估计数作为参数
    def test_basic(self, n_points, n_estimates):
        # 调用基本测试方法，并传入点数和估计数作为参数
        self.basic_test(n_points, n_estimates)

    # 使用 pytest 的 parametrize 装饰器定义多组参数化测试数据，对应参数是符号对的列表
    @pytest.mark.parametrize("signs", [[1, 1], [-1, -1], [-1, 1], [1, -1]])
    # 定义一个测试方法，用于执行基本测试，接受符号对作为参数
    def test_sign(self, signs):
        # 调用基本测试方法，并传入符号对作为参数
        self.basic_test(signs=signs)

    # 使用 pytest 的 parametrize 装饰器定义多组参数化测试数据，对应参数是布尔值表示是否取对数
    @pytest.mark.parametrize("log", [False, True])
    # 定义一个测试方法，用于测试当上下限相等时的情况，接受一个布尔值作为参数
    def test_zero(self, log):
        # 定义警告消息
        message = "A lower limit was equal to an upper limit, so"
        # 使用 pytest 的 warn 函数检查是否发出特定类型和匹配消息的警告
        with pytest.warns(UserWarning, match=message):
            # 调用 qmc_quad 函数进行积分计算，传入恒等于1的函数，以及区间 [0, 0] 和 [0, 1]
            res = qmc_quad(lambda x: 1, [0, 0], [0, 1], log=log)
        # 断言积分结果是否为负无穷（如果取对数）或者0（如果不取对数）
        assert res.integral == (-np.inf if log else 0)
        # 断言标准误差是否为0
        assert res.standard_error == 0

    # 定义一个测试方法，用于测试灵活的输入参数
    def test_flexible_input(self):
        # 定义一个函数，返回使用指定标准差的正态分布的概率密度函数
        def func(x):
            return stats.norm.pdf(x, scale=2)

        # 调用 qmc_quad 函数进行积分计算，传入函数 func 和区间 [0, 1]
        res = qmc_quad(func, 0, 1)
        # 计算正态分布在 [0, 1] 区间的累积概率并作为参考值
        ref = stats.norm.cdf(1, scale=2) - stats.norm.cdf(0, scale=2)
        # 使用 assert_allclose 函数断言计算得到的积分结果与参考值的误差在 1e-2 范围内
        assert_allclose(res.integral, ref, 1e-2)
def cumulative_simpson_nd_reference(y, *, x=None, dx=None, initial=None, axis=-1):
    # 如果 y 的指定轴的长度小于 3，则使用 cumulative_trapezoid
    if y.shape[axis] < 3:
        if initial is None:
            # 如果 initial 为 None，则直接调用 cumulative_trapezoid
            return cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=None)
        else:
            # 否则，加上 initial 后再调用 cumulative_trapezoid
            return initial + cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=0)

    # 确保工作轴在最后一个轴上
    y = np.moveaxis(y, axis, -1)
    x = np.moveaxis(x, axis, -1) if np.ndim(x) > 1 else x
    dx = np.moveaxis(dx, axis, -1) if np.ndim(dx) > 1 else dx
    initial = np.moveaxis(initial, axis, -1) if np.ndim(initial) > 1 else initial

    # 如果 `x` 不存在，则从 `dx` 创建它
    n = y.shape[-1]
    x = dx * np.arange(n) if dx is not None else x
    # 同样地，如果 `initial` 不存在，则将其设置为 0
    initial_was_none = initial is None
    initial = 0 if initial_was_none else initial

    # `np.apply_along_axis` 只接受一个数组，因此需要将参数拼接起来
    x = np.broadcast_to(x, y.shape)
    initial = np.broadcast_to(initial, y.shape[:-1] + (1,))
    z = np.concatenate((y, x, initial), axis=-1)

    # 使用 `np.apply_along_axis` 计算结果
    def f(z):
        return cumulative_simpson(z[:n], x=z[n:2*n], initial=z[2*n:])
    res = np.apply_along_axis(f, -1, z)

    # 如果 `initial` 不存在，则移除它，并根据需要撤消轴移动
    res = res[..., 1:] if initial_was_none else res
    res = np.moveaxis(res, -1, axis)
    return res
    @pytest.mark.parametrize('axis', np.arange(-3, 3))
    # 使用参数化测试，对 `axis` 进行多个取值进行测试，包括从-3到2的整数
    @pytest.mark.parametrize('x_ndim', (1, 3))
    # 使用参数化测试，对 `x_ndim` 进行多个取值进行测试，包括1和3
    @pytest.mark.parametrize('x_len', (1, 2, 7))
    # 使用参数化测试，对 `x_len` 进行多个取值进行测试，包括1、2和7
    @pytest.mark.parametrize('i_ndim', (None, 0, 3,))
    # 使用参数化测试，对 `i_ndim` 进行多个取值进行测试，包括None、0和3
    @pytest.mark.parametrize('dx', (None, True))
    # 使用参数化测试，对 `dx` 进行多个取值进行测试，包括None和True
    def test_nd(self, axis, x_ndim, x_len, i_ndim, dx):
        # 测试 `cumulative_simpson` 在 N 维 `y` 上的行为

        rng = np.random.default_rng(82456839535679456794)
        # 使用种子82456839535679456794初始化随机数生成器

        # 确定形状
        shape = [5, 6, x_len]
        shape[axis], shape[-1] = shape[-1], shape[axis]
        # 根据参数确定数组 `shape` 的形状

        shape_len_1 = shape.copy()
        shape_len_1[axis] = 1
        i_shape = shape_len_1 if i_ndim == 3 else ()
        # 根据 `i_ndim` 的值确定 `i_shape` 的形状

        # 初始化参数
        y = rng.random(size=shape)
        x, dx = None, None
        if dx:
            dx = rng.random(size=shape_len_1) if x_ndim > 1 else rng.random()
        else:
            x = (np.sort(rng.random(size=shape), axis=axis) if x_ndim > 1
                 else np.sort(rng.random(size=shape[axis])))
        initial = None if i_ndim is None else rng.random(size=i_shape)
        # 根据条件初始化 `x`、`dx` 和 `initial`

        # 比较结果
        res = cumulative_simpson(y, x=x, dx=dx, initial=initial, axis=axis)
        ref = cumulative_simpson_nd_reference(y, x=x, dx=dx, initial=initial, axis=axis)
        np.testing.assert_allclose(res, ref, rtol=1e-15)
        # 使用 `cumulative_simpson` 和 `cumulative_simpson_nd_reference` 比较结果，确保它们非常接近

    @pytest.mark.parametrize(('message', 'kwarg_update'), [
        ("x must be strictly increasing", dict(x=[2, 2, 3, 4])),
        ("x must be strictly increasing", dict(x=[x0, [2, 2, 4, 8]], y=[y0, y0])),
        ("x must be strictly increasing", dict(x=[x0, x0, x0], y=[y0, y0, y0], axis=0)),
        ("At least one point is required", dict(x=[], y=[])),
        ("`axis=4` is not valid for `y` with `y.ndim=1`", dict(axis=4)),
        ("shape of `x` must be the same as `y` or 1-D", dict(x=np.arange(5))),
        ("`initial` must either be a scalar or...", dict(initial=np.arange(5))),
        ("`dx` must either be a scalar or...", dict(x=None, dx=np.arange(5))),
    ])
    def test_simpson_exceptions(self, message, kwarg_update):
        # 测试 `cumulative_simpson` 的异常情况

        kwargs0 = dict(y=self.y0, x=self.x0, dx=None, initial=None, axis=-1)
        # 初始化关键字参数

        with pytest.raises(ValueError, match=message):
            cumulative_simpson(**dict(kwargs0, **kwarg_update))
        # 使用 pytest 检查是否会引发期望的 ValueError 异常，并匹配消息文本

    def test_special_cases(self):
        # 测试未在其他地方检查的特殊情况

        rng = np.random.default_rng(82456839535679456794)
        # 使用种子82456839535679456794初始化随机数生成器

        y = rng.random(size=10)
        res = cumulative_simpson(y, dx=0)
        assert_equal(res, 0)
        # 测试当 `dx=0` 时的特殊情况，期望结果为0

        # 应该添加以下测试：
        # - `x` 的所有元素相同的情况
        # 这些应该与 `simpson` 的情况相同
    def _get_theoretical_diff_between_simps_and_cum_simps(self, y, x):
        """Calculate theoretical difference between `cumulative_simpson` and `simpson`.
        
        `cumulative_simpson` and `simpson` are compared to ensure consistent results.
        `simpson` is called iteratively with increasing upper limits of integration.
        This function computes the theoretical correction needed for `simpson` at even
        intervals to match with `cumulative_simpson`.
        """
        # 计算 x 数组中相邻元素的差值
        d = np.diff(x, axis=-1)
        # 使用不等间隔的 Simpson 法计算累积积分
        sub_integrals_h1 = _cumulative_simpson_unequal_intervals(y, d)
        # 对 x 数组的逆序应用不等间隔的 Simpson 法计算累积积分
        sub_integrals_h2 = _cumulative_simpson_unequal_intervals(
            y[..., ::-1], d[..., ::-1]
        )[..., ::-1]

        # 构建差值数组
        zeros_shape = (*y.shape[:-1], 1)
        theoretical_difference = np.concatenate(
            [
                np.zeros(zeros_shape),
                (sub_integrals_h1[..., 1:] - sub_integrals_h2[..., :-1]),
                np.zeros(zeros_shape),
            ],
            axis=-1,
        )
        # 只在偶数间隔处存在差值，奇数间隔完全匹配，因此无需修正
        theoretical_difference[..., 1::2] = 0.0
        # 注意：第一个间隔不会匹配，因为 `simpson` 使用梯形法则
        return theoretical_difference

    @pytest.mark.slow
    @given(
        y=hyp_num.arrays(
            np.float64,
            hyp_num.array_shapes(max_dims=4, min_side=3, max_side=10),
            elements=st.floats(-10, 10, allow_nan=False).filter(lambda x: abs(x) > 1e-7)
        )
    )
    def test_cumulative_simpson_against_simpson_with_default_dx(
        self, y
    ):
        """Test `cumulative_simpson` against `simpson` using default integration step size.

        This test verifies if the outputs of `cumulative_simpson` and `simpson` match at
        all even indices and the last index, with the first index showing a discrepancy
        due to `simpson` using the trapezoidal rule when only two data points are present.
        Odd indices after the first are expected to match after applying a theoretical correction.
        """
        def simpson_reference(y):
            return np.stack(
                [simpson(y[..., :i], dx=1.0) for i in range(2, y.shape[-1]+1)], axis=-1,
            )

        # 计算累积 Simpson 积分的结果
        res = cumulative_simpson(y, dx=1.0)
        # 计算 `simpson` 的参考值
        ref = simpson_reference(y)
        # 计算理论上的差异
        theoretical_difference = self._get_theoretical_diff_between_simps_and_cum_simps(
            y, x=np.arange(y.shape[-1])
        )
        # 断言累积 Simpson 积分的结果与 `simpson` 的参考值加上理论差异在数值上全部接近
        np.testing.assert_allclose(
            res[..., 1:], ref[..., 1:] + theoretical_difference[..., 1:]
        )

    @pytest.mark.slow
    @given(
        y=hyp_num.arrays(
            np.float64,
            hyp_num.array_shapes(max_dims=4, min_side=3, max_side=10),
            elements=st.floats(-10, 10, allow_nan=False).filter(lambda x: abs(x) > 1e-7)
        )
    )
    def test_cumulative_simpson_against_simpson(
        self, y
    ):
        """Test `cumulative_simpson` against `simpson` for all data points.

        This test ensures that `cumulative_simpson` and `simpson` provide consistent results
        across all data points. It checks if `cumulative_simpson` matches `simpson` at even
        indices and the last index, with the first index showing a discrepancy due to `simpson`
        using the trapezoidal rule when only two data points are present. Subsequent odd indices
        are expected to match after applying a mathematical correction.
        """
        """Theoretically, the output of `cumulative_simpson` will be identical
        to `simpson` at all even indices and in the last index. The first index
        will not match as `simpson` uses the trapezoidal rule when there are only two
        data points. Odd indices after the first index are shown to match with
        a mathematically-derived correction."""
        # 计算每个数据点之间的间隔
        interval = 10/(y.shape[-1] - 1)
        # 在区间 [0, 10] 内生成均匀分布的数据点 x
        x = np.linspace(0, 10, num=y.shape[-1])
        # 对除了第一个数据点外的所有数据点施加随机扰动
        x[1:] = x[1:] + 0.2*interval*np.random.uniform(-1, 1, len(x) - 1)

        # 定义一个参考函数 `simpson_reference`，用于计算 `simpson` 方法的参考结果
        def simpson_reference(y, x):
            return np.stack(
                [simpson(y[..., :i], x=x[..., :i]) for i in range(2, y.shape[-1]+1)],
                axis=-1,
            )

        # 调用累积 Simpson 方法 `cumulative_simpson`，计算累积结果
        res = cumulative_simpson(y, x=x)
        # 调用参考函数 `simpson_reference`，计算 `simpson` 方法的参考结果
        ref = simpson_reference(y, x)
        # 计算理论上的差异，即 `simpson` 方法和累积 Simpson 方法之间的理论差异
        theoretical_difference = self._get_theoretical_diff_between_simps_and_cum_simps(
            y, x
        )
        # 使用 NumPy 测试库断言，确保累积 Simpson 方法的输出与参考结果的和理论差异在数值上全部接近
        np.testing.assert_allclose(
            res[..., 1:], ref[..., 1:] + theoretical_difference[..., 1:]
        )
```