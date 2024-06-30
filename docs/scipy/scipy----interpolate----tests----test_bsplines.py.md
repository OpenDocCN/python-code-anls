# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_bsplines.py`

```
# 导入必要的标准库和第三方库
import os               # 导入操作系统相关功能的模块
import operator         # 导入运算符模块，提供了标准的运算符函数
import itertools        # 导入迭代器模块，提供了用于操作迭代对象的函数

import numpy as np      # 导入数值计算库NumPy，并使用np作为别名
from numpy.testing import assert_equal, assert_allclose, assert_  # 从NumPy测试模块导入断言函数
from pytest import raises as assert_raises  # 导入pytest中的raises函数，并起别名assert_raises
import pytest           # 导入pytest测试框架

from scipy.interpolate import (  # 从SciPy插值模块中导入多个类和函数
        BSpline, BPoly, PPoly, make_interp_spline, make_lsq_spline, _bspl,
        splev, splrep, splprep, splder, splantider, sproot, splint, insert,
        CubicSpline, NdBSpline, make_smoothing_spline, RegularGridInterpolator,
)
import scipy.linalg as sl  # 导入SciPy线性代数模块的别名sl
import scipy.sparse.linalg as ssl  # 导入SciPy稀疏线性代数模块的别名ssl

from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
                                        _woodbury_algorithm, _periodic_knots,
                                        _make_interp_per_full_matr)
import scipy.interpolate._fitpack_impl as _impl  # 导入SciPy插值底层实现模块的别名_impl
from scipy._lib._util import AxisError  # 从SciPy工具模块中导入AxisError异常类

# XXX: move to the interpolate namespace
from scipy.interpolate._ndbspline import make_ndbspl  # 导入SciPy插值模块中的make_ndbspl函数

from scipy.interpolate import _dfitpack as dfitpack  # 从SciPy插值模块中导入_dfitpack别名dfitpack
from scipy.interpolate import _bsplines as _b  # 从SciPy插值模块中导入_bsplines别名_b

# 定义测试类TestBSpline
class TestBSpline:

    # 测试BSpline类的构造函数
    def test_ctor(self):
        # knots参数应为有序的一维数组，包含有限实数
        assert_raises((TypeError, ValueError), BSpline,
                **dict(t=[1, 1.j], c=[1.], k=0))
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, BSpline, **dict(t=[1, np.nan], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, np.inf], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, -1], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[[1], [1]], c=[1.], k=0))

        # 对于n+k+1个结点和次数为k的BSpline曲线，至少需要n个系数
        assert_raises(ValueError, BSpline, **dict(t=[0, 1, 2], c=[1], k=0))
        assert_raises(ValueError, BSpline,
                **dict(t=[0, 1, 2, 3, 4], c=[1., 1.], k=2))

        # 非整数次数
        assert_raises(TypeError, BSpline,
                **dict(t=[0., 0., 1., 2., 3., 4.], c=[1., 1., 1.], k="cubic"))
        assert_raises(TypeError, BSpline,
                **dict(t=[0., 0., 1., 2., 3., 4.], c=[1., 1., 1.], k=2.5))

        # 基本区间不能为零度量（例如[1..1]）
        assert_raises(ValueError, BSpline,
                **dict(t=[0., 0, 1, 1, 2, 3], c=[1., 1, 1], k=2))

        # 检查tck属性与self.tck是否一致
        n, k = 11, 3
        t = np.arange(n+k+1)
        c = np.random.random(n)
        b = BSpline(t, c, k)

        assert_allclose(t, b.t)
        assert_allclose(c, b.c)
        assert_equal(k, b.k)

    # 测试BSpline类的tck属性
    def test_tck(self):
        b = _make_random_spline()  # 创建一个随机的BSpline对象b
        tck = b.tck  # 获取其tck属性

        # 断言b的结点与tck的第一个元素相等
        assert_allclose(b.t, tck[0], atol=1e-15, rtol=1e-15)
        # 断言b的系数与tck的第二个元素相等
        assert_allclose(b.c, tck[1], atol=1e-15, rtol=1e-15)
        # 断言b的次数与tck的第三个元素相等
        assert_equal(b.k, tck[2])

        # 检查b.tck属性为只读
        with pytest.raises(AttributeError):
            b.tck = 'foo'
    # 定义一个测试函数，测试零阶 B 样条的情况
    def test_degree_0(self):
        # 在 [0, 1] 区间上生成等间距的 10 个点
        xx = np.linspace(0, 1, 10)

        # 创建一个零阶 B 样条对象，控制点为 [3.]，节点为 [0, 1]
        b = BSpline(t=[0, 1], c=[3.], k=0)
        # 断言 B 样条在 xx 上的计算结果近似为 3
        assert_allclose(b(xx), 3)

        # 创建另一个零阶 B 样条对象，控制点为 [3, 4]，节点为 [0, 0.35, 1]
        b = BSpline(t=[0, 0.35, 1], c=[3, 4], k=0)
        # 断言 B 样条在 xx 上的计算结果近似为 np.where(xx < 0.35, 3, 4)
        assert_allclose(b(xx), np.where(xx < 0.35, 3, 4))

    # 定义一个测试函数，测试一阶 B 样条的情况
    def test_degree_1(self):
        # 定义节点序列和控制点
        t = [0, 1, 2, 3, 4]
        c = [1, 2, 3]
        k = 1
        # 创建一阶 B 样条对象
        b = BSpline(t, c, k)

        # 在 [1, 3] 区间上生成 50 个等间距的点
        x = np.linspace(1, 3, 50)
        # 断言 B 样条在 x 上的计算结果近似于手动计算的结果
        assert_allclose(c[0]*B_012(x) + c[1]*B_012(x-1) + c[2]*B_012(x-2),
                        b(x), atol=1e-14)
        # 断言 B 样条在 x 上的计算结果近似于 spline 插值的结果
        assert_allclose(splev(x, (t, c, k)), b(x), atol=1e-14)

    # 定义一个测试函数，测试伯恩斯坦多项式的情况
    def test_bernstein(self):
        # 定义一个特殊的节点向量：伯恩斯坦多项式
        k = 3
        t = np.asarray([0]*(k+1) + [1]*(k+1))
        c = np.asarray([1., 2., 3., 4.])
        # 创建伯恩斯坦多项式对象
        bp = BPoly(c.reshape(-1, 1), [0, 1])
        bspl = BSpline(t, c, k)

        # 在 [-1., 2.] 区间上生成 10 个等间距的点
        xx = np.linspace(-1., 2., 10)
        # 断言伯恩斯坦多项式和 B 样条在 xx 上的计算结果近似
        assert_allclose(bp(xx, extrapolate=True),
                        bspl(xx, extrapolate=True), atol=1e-14)
        # 断言 spline 插值和 B 样条在 xx 上的计算结果近似
        assert_allclose(splev(xx, (t, c, k)),
                        bspl(xx), atol=1e-14)

    # 定义一个测试函数，测试随机系数样条在基础区间上的评估
    def test_rndm_naive_eval(self):
        # 创建一个随机系数的样条对象
        b = _make_random_spline()
        t, c, k = b.tck
        # 在 t[k] 到 t[-k-1] 区间上生成 50 个等间距的点
        xx = np.linspace(t[k], t[-k-1], 50)
        y_b = b(xx)

        # 使用第一种方法计算 xx 上的样条值
        y_n = [_naive_eval(x, t, c, k) for x in xx]
        # 断言样条对象和第一种方法计算结果的近似
        assert_allclose(y_b, y_n, atol=1e-14)

        # 使用第二种方法计算 xx 上的样条值
        y_n2 = [_naive_eval_2(x, t, c, k) for x in xx]
        # 断言样条对象和第二种方法计算结果的近似
        assert_allclose(y_b, y_n2, atol=1e-14)

    # 定义一个测试函数，测试随机系数样条和 spline 插值的比较
    def test_rndm_splev(self):
        # 创建一个随机系数的样条对象
        b = _make_random_spline()
        t, c, k = b.tck
        # 在 t[k] 到 t[-k-1] 区间上生成 50 个等间距的点
        xx = np.linspace(t[k], t[-k-1], 50)
        # 断言样条对象和 spline 插值在 xx 上的计算结果近似
        assert_allclose(b(xx), splev(xx, (t, c, k)), atol=1e-14)

    # 定义一个测试函数，测试随机系数样条和 spline 插值的比较（使用 splrep 函数）
    def test_rndm_splrep(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20)

        # 使用 splrep 函数拟合数据
        tck = splrep(x, y)
        b = BSpline(*tck)

        t, k = b.t, b.k
        # 在 t[k] 到 t[-k-1] 区间上生成 80 个等间距的点
        xx = np.linspace(t[k], t[-k-1], 80)
        # 断言样条对象和 spline 插值在 xx 上的计算结果近似
        assert_allclose(b(xx), splev(xx, tck), atol=1e-14)

    # 定义一个测试函数，测试将所有控制点置为 1 的情况
    def test_rndm_unity(self):
        # 创建一个随机系数的样条对象
        b = _make_random_spline()
        # 将所有控制点置为 1
        b.c = np.ones_like(b.c)
        # 在 b.t[b.k] 到 b.t[-b.k-1] 区间上生成 100 个等间距的点
        xx = np.linspace(b.t[b.k], b.t[-b.k-1], 100)
        # 断言样条对象在 xx 上的计算结果近似为 1.
        assert_allclose(b(xx), 1.)

    # 定义一个测试函数，测试向量化计算
    def test_vectorization(self):
        n, k = 22, 3
        t = np.sort(np.random.random(n))
        c = np.random.random(size=(n, 6, 7))
        # 创建一个向量化的 B 样条对象
        b = BSpline(t, c, k)
        tm, tp = t[k], t[-k-1]
        # 在 [tm, tp] 区间上生成形状为 (3, 4, 5) 的随机数组
        xx = tm + (tp - tm) * np.random.random((3, 4, 5))
        # 断言样条对象在 xx 上计算结果的形状是否为 (3, 4, 5, 6, 7)
        assert_equal(b(xx).shape, (3, 4, 5, 6, 7))
    def test_len_c(self):
        # for n+k+1 knots, only first n coefs are used.
        # and BTW this is consistent with FITPACK
        # 设置参数 n 和 k
        n, k = 33, 3
        # 生成随机排列的长度为 n+k+1 的节点数组 t
        t = np.sort(np.random.random(n+k+1))
        # 生成长度为 n 的随机系数数组 c
        c = np.random.random(n)

        # 在系数数组 c 后面添加长度为 k+1 的随机垃圾数据
        c_pad = np.r_[c, np.random.random(k+1)]

        # 创建两个 B 样条对象 b 和 b_pad
        b, b_pad = BSpline(t, c, k), BSpline(t, c_pad, k)

        # 计算节点数组 t 的全距
        dt = t[-1] - t[0]
        # 在节点数组 t 的全距范围内生成 50 个均匀分布的点 xx
        xx = np.linspace(t[0] - dt, t[-1] + dt, 50)
        # 断言 b 和 b_pad 在 xx 上的值接近，容差为 1e-14
        assert_allclose(b(xx), b_pad(xx), atol=1e-14)
        # 断言 b 在 xx 上的值与 FITPACK 求值结果接近，容差为 1e-14
        assert_allclose(b(xx), splev(xx, (t, c, k)), atol=1e-14)
        # 断言 b 在 xx 上的值与 c_pad 扩展后的 FITPACK 求值结果接近，容差为 1e-14
        assert_allclose(b(xx), splev(xx, (t, c_pad, k)), atol=1e-14)

    def test_endpoints(self):
        # base interval is closed
        # 随机生成一个 B 样条对象 b
        b = _make_random_spline()
        # 获取 b 的节点数组 t、系数数组 c 和阶数 k
        t, _, k = b.tck
        # 获取 t 的第 k 个和倒数第 k+1 个节点作为 tm 和 tp
        tm, tp = t[k], t[-k-1]
        # 对于两种 extrap（True 和 False），断言 b 在 [tm, tp] 上的值接近于
        # b 在 [tm + 1e-10, tp - 1e-10] 上的值，容差为 1e-9
        for extrap in (True, False):
            assert_allclose(b([tm, tp], extrap),
                            b([tm + 1e-10, tp - 1e-10], extrap), atol=1e-9)

    def test_continuity(self):
        # assert continuity at internal knots
        # 随机生成一个 B 样条对象 b
        b = _make_random_spline()
        # 获取 b 的节点数组 t、系数数组 c 和阶数 k
        t, _, k = b.tck
        # 断言 b 在内部节点处（t[k+1:-k-1]）的连续性，容差为 1e-9
        assert_allclose(b(t[k+1:-k-1] - 1e-10), b(t[k+1:-k-1] + 1e-10),
                atol=1e-9)

    def test_extrap(self):
        # 随机生成一个 B 样条对象 b
        b = _make_random_spline()
        # 获取 b 的节点数组 t、系数数组 c 和阶数 k
        t, c, k = b.tck
        # 计算节点数组 t 的全距
        dt = t[-1] - t[0]
        # 在节点数组 t 的全距范围内生成 50 个均匀分布的点 xx
        xx = np.linspace(t[k] - dt, t[-k-1] + dt, 50)
        # 构建一个在基本区间内的布尔掩码
        mask = (t[k] < xx) & (xx < t[-k-1])

        # 断言在基本区间内，extrapolate=True 和 extrapolate=False 时 b(xx[mask]) 的值接近
        assert_allclose(b(xx[mask], extrapolate=True),
                        b(xx[mask], extrapolate=False))

        # 断言在整个区间内，extrapolate=True 时 b(xx) 的值接近于 FITPACK 求值结果，容差为 0
        assert_allclose(b(xx, extrapolate=True),
                splev(xx, (t, c, k), ext=0))

    def test_default_extrap(self):
        # BSpline 默认 extrapolate=True
        # 随机生成一个 B 样条对象 b
        b = _make_random_spline()
        # 获取 b 的节点数组 t、系数数组 c 和阶数 k
        t, _, k = b.tck
        # 构建一个包含超出节点数组 t 两端的点的数组 xx
        xx = [t[0] - 1, t[-1] + 1]
        # 计算 b 在 xx 上的值 yy
        yy = b(xx)
        # 断言 yy 中并非所有值都是 NaN
        assert_(not np.all(np.isnan(yy)))

    def test_periodic_extrap(self):
        np.random.seed(1234)
        # 随机生成一个包含 8 个元素的随机节点数组 t
        t = np.sort(np.random.random(8))
        # 随机生成一个长度为 4 的随机系数数组 c
        c = np.random.random(4)
        k = 3
        # 创建一个周期性 B 样条对象 b
        b = BSpline(t, c, k, extrapolate='periodic')
        # 计算有效节点数量
        n = t.size - (k + 1)

        # 计算节点数组 t 的全距
        dt = t[-1] - t[0]
        # 在节点数组 t 的全距范围内生成 50 个均匀分布的点 xx
        xx = np.linspace(t[k] - dt, t[n] + dt, 50)
        # 计算相应的 xy，使用周期性处理 t[k] + (xx - t[k]) % (t[n] - t[k])
        xy = t[k] + (xx - t[k]) % (t[n] - t[k])
        # 断言 b 在 xx 上的值接近于 FITPACK 在 xy 上的值，容差为默认值
        assert_allclose(b(xx), splev(xy, (t, c, k)))

        # 直接检查
        xx = [-1, 0, 0.5, 1]
        xy = t[k] + (xx - t[k]) % (t[n] - t[k])
        # 断言周期性 extrapolate 模式下的 b(xx) 和 b(xy) 的值相等
        assert_equal(b(xx, extrapolate='periodic'), b(xy, extrapolate=True))

    def test_ppoly(self):
        # 随机生成一个 B 样条对象 b
        b = _make_random_spline()
        # 获取 b 的节点数组 t、系数数组 c 和阶数 k
        t, c, k = b.tck
        # 从 B 样条创建一个 PPoly 对象 pp
        pp = PPoly.from_spline((t, c, k))

        # 在节点数组 t 的范围内生成 100 个均匀分布的点 xx
        xx = np.linspace(t[k], t[-k], 100)
        # 断言 b 在 xx 上的值接近于 pp 在 xx 上的值，容差和相对容差均为 1e-14
        assert_allclose(b(xx), pp(xx), atol=1e-14, rtol=1e-14)
    # 测试随机生成样条的导数计算
    def test_derivative_rndm(self):
        # 生成随机样条
        b = _make_random_spline()
        # 获取样条的节点、系数、阶数
        t, c, k = b.tck
        # 在样条定义域内均匀采样点
        xx = np.linspace(t[0], t[-1], 50)
        # 将节点添加到采样点中
        xx = np.r_[xx, t]

        # 计算样条的每阶导数，并验证其与样条对象的导数计算结果的接近程度
        for der in range(1, k+1):
            yd = splev(xx, (t, c, k), der=der)
            assert_allclose(yd, b(xx, nu=der), atol=1e-14)

        # 验证更高阶导数为零
        assert_allclose(b(xx, nu=k+1), 0, atol=1e-14)

    # 测试样条在节点处的导数跳跃情况
    def test_derivative_jumps(self):
        # 使用 de Boor 的示例（书中第 IX 章，例子（24））
        # 注意：节点被扩展并且对应的系数被置零，符合约定（29）
        k = 2
        t = [-1, -1, 0, 1, 1, 3, 4, 6, 6, 6, 7, 7]
        np.random.seed(1234)
        c = np.r_[0, 0, np.random.random(5), 0, 0]
        b = BSpline(t, c, k)

        # 在 x != 6（三重节点）处，样条是连续的
        x = np.asarray([1, 3, 4, 6])
        assert_allclose(b(x[x != 6] - 1e-10),
                        b(x[x != 6] + 1e-10))
        assert_(not np.allclose(b(6.-1e-10), b(6+1e-10)))

        # 一阶导数在双重节点 1 和 6 处有跳跃
        x0 = np.asarray([3, 4])
        assert_allclose(b(x0 - 1e-10, nu=1),
                        b(x0 + 1e-10, nu=1))
        x1 = np.asarray([1, 6])
        assert_(not np.all(np.allclose(b(x1 - 1e-10, nu=1),
                                       b(x1 + 1e-10, nu=1))))

        # 二阶导数也不保证是连续的
        assert_(not np.all(np.allclose(b(x - 1e-10, nu=2),
                                       b(x + 1e-10, nu=2))))

    # 测试二次基函数元素
    def test_basis_element_quadratic(self):
        # 在指定区间内均匀采样点
        xx = np.linspace(-1, 4, 20)
        # 获取基于给定节点的二次样条基函数
        b = BSpline.basis_element(t=[0, 1, 2, 3])
        # 验证二次基函数的计算结果与样条对象在相同参数下的计算结果接近
        assert_allclose(b(xx),
                        splev(xx, (b.t, b.c, b.k)), atol=1e-14)
        # 验证二次基函数的计算结果与预定义函数 B_0123 的计算结果接近
        assert_allclose(b(xx),
                        B_0123(xx), atol=1e-14)

        # 使用另一组节点进行测试
        b = BSpline.basis_element(t=[0, 1, 1, 2])
        xx = np.linspace(0, 2, 10)
        # 验证基函数的计算结果与预期函数计算结果接近
        assert_allclose(b(xx),
                np.where(xx < 1, xx*xx, (2.-xx)**2), atol=1e-14)

    # 测试随机生成样条的基函数元素
    def test_basis_element_rndm(self):
        # 生成随机样条
        b = _make_random_spline()
        # 获取样条的节点、系数、阶数
        t, c, k = b.tck
        # 在样条定义域内均匀采样点
        xx = np.linspace(t[k], t[-k-1], 20)
        # 验证基函数元素的计算结果与自定义函数 _sum_basis_elements 的计算结果接近
        assert_allclose(b(xx), _sum_basis_elements(xx, t, c, k), atol=1e-14)

    # 测试复数系数样条的计算
    def test_cmplx(self):
        # 生成随机样条
        b = _make_random_spline()
        # 获取样条的节点、复数系数、阶数
        t, c, k = b.tck
        # 将系数扩展为复数
        cc = c * (1. + 3.j)

        # 创建具有复数系数的样条对象
        b = BSpline(t, cc, k)
        # 分别创建实部和虚部为系数的样条对象
        b_re = BSpline(t, b.c.real, k)
        b_im = BSpline(t, b.c.imag, k)

        # 在样条定义域内均匀采样点
        xx = np.linspace(t[k], t[-k-1], 20)
        # 验证实部的计算结果与实部系数样条对象的计算结果接近
        assert_allclose(b(xx).real, b_re(xx), atol=1e-14)
        # 验证虚部的计算结果与虚部系数样条对象的计算结果接近
        assert_allclose(b(xx).imag, b_im(xx), atol=1e-14)

    # 测试样条函数在输入为 NaN 时的输出
    def test_nan(self):
        # 获取基于给定节点的二次样条基函数
        b = BSpline.basis_element([0, 1, 1, 2])
        # 验证当输入为 NaN 时，输出也为 NaN
        assert_(np.isnan(b(np.nan)))
    # 测试 BSpline 对象的导数方法
    def test_derivative_method(self):
        # 生成一个随机的样条曲线对象
        b = _make_random_spline(k=5)
        # 获取样条曲线的参数
        t, c, k = b.tck
        # 根据参数创建 BSpline 对象
        b0 = BSpline(t, c, k)
        # 在参数范围内生成均匀间隔的数据点
        xx = np.linspace(t[k], t[-k-1], 20)
        # 对每个次数从 1 到 k 的导数进行测试
        for j in range(1, k):
            # 计算样条曲线的 j 阶导数
            b = b.derivative()
            # 断言计算得到的导数与预期值的接近程度
            assert_allclose(b0(xx, j), b(xx), atol=1e-12, rtol=1e-12)

    # 测试 BSpline 对象的反导数方法
    def test_antiderivative_method(self):
        # 生成一个随机的样条曲线对象
        b = _make_random_spline()
        # 获取样条曲线的参数
        t, c, k = b.tck
        # 在参数范围内生成均匀间隔的数据点
        xx = np.linspace(t[k], t[-k-1], 20)
        # 断言计算得到的反导数的导数与原始函数的接近程度
        assert_allclose(b.antiderivative().derivative()(xx),
                        b(xx), atol=1e-14, rtol=1e-14)

        # 使用 N 维数组测试 c 参数
        c = np.c_[c, c, c]
        c = np.dstack((c, c))
        b = BSpline(t, c, k)
        # 断言计算得到的反导数的导数与原始函数的接近程度
        assert_allclose(b.antiderivative().derivative()(xx),
                        b(xx), atol=1e-14, rtol=1e-14)

    # 测试 BSpline 对象的积分方法
    def test_integral(self):
        # 生成一个基础的 BSpline 对象
        b = BSpline.basis_element([0, 1, 2])  # x for x < 1 else 2 - x
        # 断言计算得到的积分值与预期值的接近程度
        assert_allclose(b.integrate(0, 1), 0.5)
        assert_allclose(b.integrate(1, 0), -1 * 0.5)
        assert_allclose(b.integrate(1, 0), -0.5)

        # 在 [0, 2] 之外进行外推或填充为零，这里使用默认的外推行为
        assert_allclose(b.integrate(-1, 1), 0)
        assert_allclose(b.integrate(-1, 1, extrapolate=True), 0)
        assert_allclose(b.integrate(-1, 1, extrapolate=False), 0.5)
        assert_allclose(b.integrate(1, -1, extrapolate=False), -1 * 0.5)

        # 测试 ``_fitpack._splint()`` 方法
        assert_allclose(b.integrate(1, -1, extrapolate=False),
                        _impl.splint(1, -1, b.tck))

        # 测试 ``extrapolate='periodic'`` 的情况
        b.extrapolate = 'periodic'
        i = b.antiderivative()
        period_int = i(2) - i(0)

        assert_allclose(b.integrate(0, 2), period_int)
        assert_allclose(b.integrate(2, 0), -1 * period_int)
        assert_allclose(b.integrate(-9, -7), period_int)
        assert_allclose(b.integrate(-8, -4), 2 * period_int)

        assert_allclose(b.integrate(0.5, 1.5), i(1.5) - i(0.5))
        assert_allclose(b.integrate(1.5, 3), i(1) - i(0) + i(2) - i(1.5))
        assert_allclose(b.integrate(1.5 + 12, 3 + 12),
                        i(1) - i(0) + i(2) - i(1.5))
        assert_allclose(b.integrate(1.5, 3 + 12),
                        i(1) - i(0) + i(2) - i(1.5) + 6 * period_int)

        assert_allclose(b.integrate(0, -1), i(0) - i(1))
        assert_allclose(b.integrate(-9, -10), i(0) - i(1))
        assert_allclose(b.integrate(0, -9), i(1) - i(2) - 4 * period_int)

    # 测试 BSpline 对象的 PPoly 积分方法
    def test_integrate_ppoly(self):
        # 测试 .integrate 方法与 PPoly.integrate 方法的一致性
        x = [0, 1, 2, 3, 4]
        # 生成一个插值样条曲线对象
        b = make_interp_spline(x, x)
        b.extrapolate = 'periodic'
        # 从样条曲线对象创建一个 PPoly 对象
        p = PPoly.from_spline(b)

        # 对不同区间进行积分计算，断言结果的接近程度
        for x0, x1 in [(-5, 0.5), (0.5, 5), (-4, 13)]:
            assert_allclose(b.integrate(x0, x1),
                            p.integrate(x0, x1))
    def test_integrate_0D_always(self):
        # 确保结果始终是零维数组（而不是 Python 标量）
        b = BSpline.basis_element([0, 1, 2])
        # 对于两种外推方式，分别进行测试
        for extrapolate in (True, False):
            # 计算积分并返回结果
            res = b.integrate(0, 1, extrapolate=extrapolate)
            assert type(res) == np.ndarray  # 确保返回结果类型为 NumPy 数组
            assert res.ndim == 0  # 确保返回结果是零维数组

    def test_subclassing(self):
        # 类方法不应退化为基类
        class B(BSpline):
            pass

        # 使用子类方法创建 BSpline 对象
        b = B.basis_element([0, 1, 2, 2])
        # 确保对象的类是子类 B
        assert_equal(b.__class__, B)
        # 确保导数的类也是子类 B
        assert_equal(b.derivative().__class__, B)
        # 确保反导数的类也是子类 B
        assert_equal(b.antiderivative().__class__, B)

    @pytest.mark.parametrize('axis', range(-4, 4))
    def test_axis(self, axis):
        n, k = 22, 3
        t = np.linspace(0, 1, n + k + 1)
        sh = [6, 7, 8]
        # 在这个测试中，我们需要正的 axis 来进行索引和切片操作。
        pos_axis = axis % 4
        sh.insert(pos_axis, n)   # 插入 n 到 sh 列表中的正轴位置
        c = np.random.random(size=sh)
        # 使用给定的轴创建 BSpline 对象
        b = BSpline(t, c, k, axis=axis)
        # 确保 b.c 的形状符合预期
        assert_equal(b.c.shape,
                     [sh[pos_axis],] + sh[:pos_axis] + sh[pos_axis+1:])

        xp = np.random.random((3, 4, 5))
        # 确保 b(xp) 的形状符合预期
        assert_equal(b(xp).shape,
                     sh[:pos_axis] + list(xp.shape) + sh[pos_axis+1:])

        # axis 的值应该在 -c.ndim 到 c.ndim 之间
        for ax in [-c.ndim - 1, c.ndim]:
            # 确保在给定的 axis 值下会引发 AxisError 异常
            assert_raises(AxisError, BSpline,
                          **dict(t=t, c=c, k=k, axis=ax))

        # 导数和反导数应该保持相同的 axis
        for b1 in [BSpline(t, c, k, axis=axis).derivative(),
                   BSpline(t, c, k, axis=axis).derivative(2),
                   BSpline(t, c, k, axis=axis).antiderivative(),
                   BSpline(t, c, k, axis=axis).antiderivative(2)]:
            assert_equal(b1.axis, b.axis)

    def test_neg_axis(self):
        k = 2
        t = [0, 1, 2, 3, 4, 5, 6]
        c = np.array([[-1, 2, 0, -1], [2, 0, -3, 1]])

        # 使用负轴创建 BSpline 对象
        spl = BSpline(t, c, k, axis=-1)
        spl0 = BSpline(t, c[0], k)
        spl1 = BSpline(t, c[1], k)
        # 确保 spl(2.5) 的结果与 spl0(2.5), spl1(2.5) 的结果相同
        assert_equal(spl(2.5), [spl0(2.5), spl1(2.5)])
    def test_design_matrix_bc_types(self):
        '''
        Splines with different boundary conditions are built on different
        types of vectors of knots. As far as design matrix depends only on
        vector of knots, `k` and `x` it is useful to make tests for different
        boundary conditions (and as following different vectors of knots).
        '''
        # 定义测试函数，用于运行不同边界条件下的设计矩阵测试
        def run_design_matrix_tests(n, k, bc_type):
            '''
            To avoid repetition of code the following function is provided.
            避免代码重复，提供以下函数。
            '''
            # 设置随机种子以确保结果可重复
            np.random.seed(1234)
            # 生成随机的 x 值并排序，将其限制在 -20 到 20 之间
            x = np.sort(np.random.random_sample(n) * 40 - 20)
            # 生成随机的 y 值，将其限制在 -20 到 20 之间
            y = np.random.random_sample(n) * 40 - 20
            # 如果边界条件为 "periodic"，则将第一个和最后一个 y 值设置为相等
            if bc_type == "periodic":
                y[0] = y[-1]

            # 创建样条插值对象 bspl，指定 x 和 y 数据，以及给定的边界条件 bc_type 和阶数 k
            bspl = make_interp_spline(x, y, k=k, bc_type=bc_type)

            # 生成单位矩阵 c，其大小为 bspl.t 的长度减去 k - 1
            c = np.eye(len(bspl.t) - k - 1)
            # 计算默认设计矩阵 des_matr_def
            des_matr_def = BSpline(bspl.t, c, k)(x)
            # 计算稀疏矩阵形式的设计矩阵 des_matr_csr
            des_matr_csr = BSpline.design_matrix(x,
                                                 bspl.t,
                                                 k).toarray()
            # 断言稀疏矩阵乘以 bspl.c 得到的结果与 y 在给定精度下相等
            assert_allclose(des_matr_csr @ bspl.c, y, atol=1e-14)
            # 断言默认设计矩阵与稀疏矩阵形式的设计矩阵在给定精度下相等
            assert_allclose(des_matr_def, des_matr_csr, atol=1e-14)

        # "clamped" 和 "natural" 仅在 k = 3 时有效
        n = 11
        k = 3
        # 分别对 "clamped" 和 "natural" 运行设计矩阵测试
        for bc in ["clamped", "natural"]:
            run_design_matrix_tests(n, k, bc)

        # "not-a-knot" 在奇数 k 值下有效
        for k in range(3, 8, 2):
            # 对奇数 k 值下的 "not-a-knot" 运行设计矩阵测试
            run_design_matrix_tests(n, k, "not-a-knot")

        # "periodic" 在任意 k 值下有效（甚至大于 n 的情况）
        n = 5  # 减小 n 以测试 k > n 的情况
        # 对不同 k 值下的 "periodic" 运行设计矩阵测试
        for k in range(2, 7):
            run_design_matrix_tests(n, k, "periodic")

    @pytest.mark.parametrize('extrapolate', [False, True, 'periodic'])
    @pytest.mark.parametrize('degree', range(5))
    def test_design_matrix_same_as_BSpline_call(self, extrapolate, degree):
        """Test that design_matrix(x) is equivalent to BSpline(..)(x)."""
        # 设置随机种子以确保结果可重复
        np.random.seed(1234)
        # 生成包含 10 * (degree + 1) 个随机值的 x 数组
        x = np.random.random_sample(10 * (degree + 1))
        # 计算 x 数组的最小值和最大值
        xmin, xmax = np.amin(x), np.amax(x)
        # 设置阶数 k 为 degree
        k = degree
        # 根据最小值和最大值生成节点数组 t
        t = np.r_[np.linspace(xmin - 2, xmin - 1, degree),
                  np.linspace(xmin, xmax, 2 * (degree + 1)),
                  np.linspace(xmax + 1, xmax + 2, degree)]
        # 生成单位矩阵 c，其大小为 t 的长度减去 k - 1
        c = np.eye(len(t) - k - 1)
        # 创建 BSpline 对象 bspline，指定节点数组 t、单位矩阵 c、阶数 k 和外推方式 extrapolate
        bspline = BSpline(t, c, k, extrapolate)
        # 断言 BSpline 对象在 x 数组上的结果与设计矩阵的稀疏矩阵形式在给定精度下相等
        assert_allclose(
            bspline(x), BSpline.design_matrix(x, t, k, extrapolate).toarray()
        )

        # 对于外推方式的处理
        # 若不允许外推，断言调用设计矩阵函数时会抛出 ValueError 异常
        x = np.array([xmin - 10, xmin - 1, xmax + 1.5, xmax + 10])
        if not extrapolate:
            with pytest.raises(ValueError):
                BSpline.design_matrix(x, t, k, extrapolate)
        else:
            # 断言 BSpline 对象在 x 数组上的结果与设计矩阵的稀疏矩阵形式在给定精度下相等
            assert_allclose(
                bspline(x),
                BSpline.design_matrix(x, t, k, extrapolate).toarray()
            )
    def test_design_matrix_x_shapes(self):
        # 测试不同 `x` 形状的情况
        np.random.seed(1234)
        n = 10
        k = 3
        x = np.sort(np.random.random_sample(n) * 40 - 20)  # 生成排序后的随机数组 `x`
        y = np.random.random_sample(n) * 40 - 20  # 生成随机数组 `y`

        bspl = make_interp_spline(x, y, k=k)  # 创建插值样条
        for i in range(1, 4):
            xc = x[:i]
            yc = y[:i]
            # 计算设计矩阵并转换为稀疏矩阵格式
            des_matr_csr = BSpline.design_matrix(xc,
                                                 bspl.t,
                                                 k).toarray()
            # 断言设计矩阵乘以插值系数等于截取的 `y` 值，允许的误差为 1e-14
            assert_allclose(des_matr_csr @ bspl.c, yc, atol=1e-14)

    def test_design_matrix_t_shapes(self):
        # 测试最小可能的 `t` 形状情况
        t = [1., 1., 1., 2., 3., 4., 4., 4.]  # 给定的 `t` 值列表
        # 计算设计矩阵并转换为稀疏矩阵格式
        des_matr = BSpline.design_matrix(2., t, 3).toarray()
        # 断言设计矩阵等于预期的矩阵值，允许的误差为 1e-14
        assert_allclose(des_matr,
                        [[0.25, 0.58333333, 0.16666667, 0.]],
                        atol=1e-14)

    def test_design_matrix_asserts(self):
        np.random.seed(1234)
        n = 10
        k = 3
        x = np.sort(np.random.random_sample(n) * 40 - 20)  # 生成排序后的随机数组 `x`
        y = np.random.random_sample(n) * 40 - 20  # 生成随机数组 `y`
        bspl = make_interp_spline(x, y, k=k)  # 创建插值样条
        # 不合法的节点向量（应为升序的一维数组）
        # 这里实际的节点向量被反转了，所以是无效的
        with assert_raises(ValueError):
            BSpline.design_matrix(x, bspl.t[::-1], k)
        k = 2
        t = [0., 1., 2., 3., 4., 5.]  # 给定的 `t` 值列表
        x = [1., 2., 3., 4.]  # 给定的 `x` 值列表
        # 越界的情况
        with assert_raises(ValueError):
            BSpline.design_matrix(x, t, k)

    @pytest.mark.parametrize('bc_type', ['natural', 'clamped',
                                         'periodic', 'not-a-knot'])
    def test_from_power_basis(self, bc_type):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))  # 生成排序后的随机数组 `x`
        y = np.random.random(20)  # 生成随机数组 `y`
        if bc_type == 'periodic':
            y[-1] = y[0]
        cb = CubicSpline(x, y, bc_type=bc_type)  # 创建立方样条
        bspl = BSpline.from_power_basis(cb, bc_type=bc_type)  # 从功率基函数创建样条
        xx = np.linspace(0, 1, 20)  # 在指定间隔内生成均匀分布的数字
        # 断言插值样条和立方样条在指定点 `xx` 的值非常接近，允许的误差为 1e-15
        assert_allclose(cb(xx), bspl(xx), atol=1e-15)
        bspl_new = make_interp_spline(x, y, bc_type=bc_type)  # 创建新的插值样条
        # 断言两个插值样条的系数非常接近，允许的误差为 1e-15
        assert_allclose(bspl.c, bspl_new.c, atol=1e-15)

    @pytest.mark.parametrize('bc_type', ['natural', 'clamped',
                                         'periodic', 'not-a-knot'])
    def test_from_power_basis_complex(self, bc_type):
        # 设置随机种子以保证可复现性
        np.random.seed(1234)
        # 生成排序后的随机数组作为 x 值
        x = np.sort(np.random.random(20))
        # 生成包含实部和虚部的随机复数数组作为 y 值
        y = np.random.random(20) + np.random.random(20) * 1j
        # 如果边界条件是周期性的，则将 y 数组的最后一个元素设置为第一个元素，形成周期性边界条件
        if bc_type == 'periodic':
            y[-1] = y[0]
        # 使用 CubicSpline 类构建样条插值 cb
        cb = CubicSpline(x, y, bc_type=bc_type)
        # 根据 power basis 构建 B-spline 插值 bspl
        bspl = BSpline.from_power_basis(cb, bc_type=bc_type)
        # 根据实部构建新的样条插值 bspl_new_real
        bspl_new_real = make_interp_spline(x, y.real, bc_type=bc_type)
        # 根据虚部构建新的样条插值 bspl_new_imag
        bspl_new_imag = make_interp_spline(x, y.imag, bc_type=bc_type)
        # 断言 B-spline 插值 bspl 的系数类型与实部和虚部构建的插值之和的类型相同
        assert_equal(bspl.c.dtype, (bspl_new_real.c
                                    + 1j * bspl_new_imag.c).dtype)
        # 断言 B-spline 插值 bspl 的系数与实部和虚部构建的插值之和在给定的精度下相等
        assert_allclose(bspl.c, bspl_new_real.c
                        + 1j * bspl_new_imag.c, atol=1e-15)

    def test_from_power_basis_exmp(self):
        '''
        For x = [0, 1, 2, 3, 4] and y = [1, 1, 1, 1, 1]
        the coefficients of Cubic Spline in the power basis:

        $[[0, 0, 0, 0, 0],\\$
        $[0, 0, 0, 0, 0],\\$
        $[0, 0, 0, 0, 0],\\$
        $[1, 1, 1, 1, 1]]$

        It could be shown explicitly that coefficients of the interpolating
        function in B-spline basis are c = [1, 1, 1, 1, 1, 1, 1]
        '''
        # 给定的示例中的 x 和 y 值
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([1, 1, 1, 1, 1])
        # 使用 CubicSpline 类构建样条插值 cs，然后根据 power basis 构建 B-spline 插值 bspl
        bspl = BSpline.from_power_basis(CubicSpline(x, y, bc_type='natural'),
                                        bc_type='natural')
        # 断言 B-spline 插值 bspl 的系数与给定的值在给定的精度下相等
        assert_allclose(bspl.c, [1, 1, 1, 1, 1, 1, 1], atol=1e-15)

    def test_read_only(self):
        # 确保 BSpline 在只读模式下处理节点和系数
        # 创建只读的节点数组 t 和系数数组 c
        t = np.array([0, 1])
        c = np.array([3.0])
        t.setflags(write=False)  # 设置 t 数组为只读
        c.setflags(write=False)  # 设置 c 数组为只读

        # 生成只读的输入数组 xx
        xx = np.linspace(0, 1, 10)
        xx.setflags(write=False)

        # 使用只读的节点 t 和系数 c 构建 BSpline 对象 b
        b = BSpline(t=t, c=c, k=0)
        # 断言在给定的输入 xx 下，BSpline 对象 b 的计算结果与预期的值 3 相等
        assert_allclose(b(xx), 3)
# 定义一个测试类 TestInsert
class TestInsert:

    # 使用 pytest 的参数化装饰器，对 test_insert 方法参数化，参数 xval 取值为 [0.0, 1.0, 2.5, 4, 6.5, 7.0]
    @pytest.mark.parametrize('xval', [0.0, 1.0, 2.5, 4, 6.5, 7.0])
    def test_insert(self, xval):
        # 创建一个包含 0 到 7 的数组 x
        x = np.arange(8)
        # 计算 sin(x) 的立方作为 y 的值
        y = np.sin(x)**3
        # 使用 make_interp_spline 函数创建一个样条插值对象 spl，阶数 k 设置为 3
        spl = make_interp_spline(x, y, k=3)

        # 调用 insert 函数插入一个节点到 spl 中，得到插值后的样条对象 spl_1f，FITPACK
        spl_1f = insert(xval, spl)     # FITPACK
        # 调用 spl 对象的 insert_knot 方法插入一个节点，得到插值后的样条对象 spl_1
        spl_1 = spl.insert_knot(xval)

        # 使用 assert_allclose 断言函数检查插入节点后的节点位置是否一致
        assert_allclose(spl_1.t, spl_1f.t, atol=1e-15)
        # 使用 assert_allclose 断言函数检查插入节点后的系数是否一致
        assert_allclose(spl_1.c, spl_1f.c[:-spl.k-1], atol=1e-15)

        # 如果插入的节点不是 x 的最后一个节点，创建新的数组 xx，插入新节点
        xx = x if xval != x[-1] else x[:-1]
        xx = np.r_[xx, 0.5*(x[1:] + x[:-1])]
        # 使用 assert_allclose 断言函数检查插入节点后的函数值是否保持不变
        assert_allclose(spl(xx), spl_1(xx), atol=1e-15)

        # 如果需要测试 ndim > 1 的情况
        # 计算 cos(x) 的立方作为 y1 的值
        y1 = np.cos(x)**3
        # 使用 make_interp_spline 函数创建新的样条插值对象 spl_y1，阶数 k 设置为 3
        spl_y1 = make_interp_spline(x, y1, k=3)
        # 将 y 和 y1 组合成二维数组，并创建新的样条插值对象 spl_yy
        spl_yy = make_interp_spline(x, np.c_[y, y1], k=3)
        # 调用 spl_yy 对象的 insert_knot 方法插入一个节点，得到插值后的样条对象 spl_yy1
        spl_yy1 = spl_yy.insert_knot(xval)

        # 使用 assert_allclose 断言函数检查插入节点后的节点位置是否一致
        assert_allclose(spl_yy1.t, spl_1.t, atol=1e-15)
        # 使用 assert_allclose 断言函数检查插入节点后的系数是否一致
        assert_allclose(spl_yy1.c, np.c_[spl.insert_knot(xval).c,
                                         spl_y1.insert_knot(xval).c], atol=1e-15)

        # 创建新的数组 xx，插入新节点
        xx = x if xval != x[-1] else x[:-1]
        xx = np.r_[xx, 0.5*(x[1:] + x[:-1])]
        # 使用 assert_allclose 断言函数检查插入节点后的函数值是否保持不变
        assert_allclose(spl_yy(xx), spl_yy1(xx), atol=1e-15)

    # 使用 pytest 的参数化装饰器，对 test_insert_multi 方法参数化，参数 xval 和 m 取不同值
    @pytest.mark.parametrize(
        'xval, m', [(0.0, 2), (1.0, 3), (1.5, 5), (4, 2), (7.0, 2)]
    )
    def test_insert_multi(self, xval, m):
        # 创建一个包含 0 到 7 的数组 x
        x = np.arange(8)
        # 计算 sin(x) 的立方作为 y 的值
        y = np.sin(x)**3
        # 使用 make_interp_spline 函数创建一个样条插值对象 spl，阶数 k 设置为 3
        spl = make_interp_spline(x, y, k=3)

        # 调用 insert 函数插入多个节点到 spl 中，得到插值后的样条对象 spl_1f
        spl_1f = insert(xval, spl, m=m)
        # 调用 spl 对象的 insert_knot 方法插入多个节点，得到插值后的样条对象 spl_1
        spl_1 = spl.insert_knot(xval, m)

        # 使用 assert_allclose 断言函数检查插入节点后的节点位置是否一致
        assert_allclose(spl_1.t, spl_1f.t, atol=1e-15)
        # 使用 assert_allclose 断言函数检查插入节点后的系数是否一致
        assert_allclose(spl_1.c, spl_1f.c[:-spl.k-1], atol=1e-15)

        # 创建新的数组 xx，插入新节点
        xx = x if xval != x[-1] else x[:-1]
        xx = np.r_[xx, 0.5*(x[1:] + x[:-1])]
        # 使用 assert_allclose 断言函数检查插入节点后的函数值是否保持不变
        assert_allclose(spl(xx), spl_1(xx), atol=1e-15)

    # 定义一个测试方法 test_insert_random
    def test_insert_random(self):
        # 创建一个随机数生成器 rng
        rng = np.random.default_rng(12345)
        # 设置 n 和 k 的值
        n, k = 11, 3

        # 生成一个排序后的随机数组 t
        t = np.sort(rng.uniform(size=n+k+1))
        # 生成一个形状为 (n, 3, 2) 的随机数组 c
        c = rng.uniform(size=(n, 3, 2))
        # 使用 BSpline 类创建一个 B 样条曲线对象 spl
        spl = BSpline(t, c, k)

        # 从 t[k+1] 到 t[-k-1] 之间生成一个随机数 xv
        xv = rng.uniform(low=t[k+1], high=t[-k-1])
        # 在 spl 中插入一个节点 xv，得到插值后的样条对象 spl_1
        spl_1 = spl.insert_knot(xv)

        # 生成一个在区间 [t[k+1], t[-k-1]] 内均匀分布的大小为 33 的随机数组 xx
        xx = rng.uniform(low=t[k+1], high=t[-k-1], size=33)
        # 使用 assert_allclose 断言函数检查插入节点后的函数值是否保持不变
        assert_allclose(spl(xx), spl_1(xx), atol=1e-15)

    # 使用 pytest 的参数化装饰器，对 test_insert_periodic 方法参数化，参数 xv 取不同值
    @pytest.mark.parametrize('xv', [0, 0.1, 2.0, 4.0, 4.5,      # l.h. edge
                                    5.5, 6.0, 6.1, 7.0]         # r.h. edge
    )
    def test_insert_periodic(self, xv):
        # 创建一个包含 0 到 7 的数组 x
        x = np.arange(8)
        # 计算 sin(x) 的立方作为 y 的值
        y = np.sin(x)**3
        # 使用 splrep 函数计算 x 和 y 的 B 样条插值的节点和系数
        tck = splrep(x, y, k=3)
        # 使用 BSpline 类创建一个周期性外推的 B 样条曲线对象 spl
        spl = BSpline(*tck, extrapolate="periodic")

        # 在 spl 中插入一个节点 xv，得到插值后的样条对象 spl_1
        spl_1 = spl.insert_k
    # 定义一个名为 test_insert_periodic_too_few_internal_knots 的测试方法，用于测试在周期性扩展时内部结点数过少的情况。
    def test_insert_periodic_too_few_internal_knots(self):
        # 在这里使用 assert_raises 来断言当内部结点数不足以进行周期性扩展时，会引发 ValueError 异常。
        # 下面是一个示例 t 数组，内部结点为 2, 3,    , 4, 5
        #                                    ^
        #                             2, 3, 3.5, 4, 5
        #   所以从新结点需要从每边取两个结点，但至少需要从左边或右边取。
        xv = 3.5
        k = 3
        t = np.array([0]*(k+1) + [2, 3, 4, 5] + [7]*(k+1))
        # 创建一个与 t 数组长度相匹配的系数数组 c，初始化为 1
        c = np.ones(len(t) - k - 1)
        # 使用 BSpline 创建样条曲线对象 spl，指定 k 为阶数，extrapolate="periodic" 表示周期性外推
        spl = BSpline(t, c, k, extrapolate="periodic")

        # 使用 assert_raises 断言插入新结点 xv 到 (t, c, k) 中时会引发 ValueError 异常，per=True 表示周期性条件下插入
        with assert_raises(ValueError):
            insert(xv, (t, c, k), per=True)

        # 使用 assert_raises 断言使用 spl.insert_knot 插入新结点 xv 时会引发 ValueError 异常
        with assert_raises(ValueError):
            spl.insert_knot(xv)

    # 定义一个名为 test_insert_no_extrap 的测试方法，用于测试在无外推条件下的结点插入。
    def test_insert_no_extrap(self):
        k = 3
        t = np.array([0]*(k+1) + [2, 3, 4, 5] + [7]*(k+1))
        # 创建一个与 t 数组长度相匹配的系数数组 c，初始化为 1
        c = np.ones(len(t) - k - 1)
        # 使用 BSpline 创建样条曲线对象 spl，指定 k 为阶数，默认不使用外推
        spl = BSpline(t, c, k)

        # 使用 assert_raises 断言在 spl 中插入超出范围的新结点 -1 时会引发 ValueError 异常
        with assert_raises(ValueError):
            spl.insert_knot(-1)

        # 使用 assert_raises 断言在 spl 中插入超出范围的新结点 8 时会引发 ValueError 异常
        with assert_raises(ValueError):
            spl.insert_knot(8)

        # 使用 assert_raises 断言在 spl 中插入新结点 3 时，由于 m=0 的条件不满足会引发 ValueError 异常
        with assert_raises(ValueError):
            spl.insert_knot(3, m=0)
def test_knots_multiplicity():
    """
    测试不同结点多重性的样条函数。

    """

    def check_splev(b, j, der=0, atol=1e-14, rtol=1e-14):
        """
        检查与 FITPACK 计算结果的样条函数评估，包括外推。

        """
        t, c, k = b.tck
        x = np.unique(t)
        x = np.r_[t[0]-0.1, 0.5*(x[1:] + x[:1]), t[-1]+0.1]
        assert_allclose(splev(x, (t, c, k), der), b(x, der),
                        atol=atol, rtol=rtol, err_msg=f'der = {der}  k = {b.k}')

    # 测试循环本身
    # [索引 `j` 用于解释失败时的回溯]
    for k in [1, 2, 3, 4, 5]:
        b = _make_random_spline(k=k)
        for j, b1 in enumerate(_make_multiples(b)):
            check_splev(b1, j)
            for der in range(1, k+1):
                check_splev(b1, j, der, 1e-12, 1e-12)


def _naive_B(x, k, i, t):
    """
    计算 B-样条基函数的简单方法。仅用于测试！

    计算 B(x; t[i],..., t[i+k+1])

    """
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * _naive_B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * _naive_B(x, k-1, i+1, t)
    return (c1 + c2)


def _naive_eval(x, t, c, k):
    """
    简单的 B-样条评估方法。仅用于测试！

    """
    if x == t[k]:
        i = k
    else:
        i = np.searchsorted(t, x) - 1
    assert t[i] <= x <= t[i+1]
    assert i >= k and i < len(t) - k
    return sum(c[i-j] * _naive_B(x, k, i-j, t) for j in range(0, k+1))


def _naive_eval_2(x, t, c, k):
    """
    另一种简单的 B-样条评估方法。仅用于测试！

    """
    n = len(t) - (k+1)
    assert n >= k+1
    assert len(c) >= n
    assert t[k] <= x <= t[n]
    return sum(c[i] * _naive_B(x, k, i, t) for i in range(n))


def _sum_basis_elements(x, t, c, k):
    """
    计算 B-样条基函数元素之和。

    """
    n = len(t) - (k+1)
    assert n >= k+1
    assert len(c) >= n
    s = 0.
    for i in range(n):
        b = BSpline.basis_element(t[i:i+k+2], extrapolate=False)(x)
        s += c[i] * np.nan_to_num(b)   # 将超出边界的元素置零
    return s


def B_012(x):
    """
    线性 B-样条函数 B(x | 0, 1, 2)。

    """
    x = np.atleast_1d(x)
    return np.piecewise(x, [(x < 0) | (x > 2),
                            (x >= 0) & (x < 1),
                            (x >= 1) & (x <= 2)],
                        [lambda x: 0., lambda x: x, lambda x: 2.-x])


def B_0123(x, der=0):
    """
    二次 B-样条函数 B(x | 0, 1, 2, 3)。

    """
    x = np.atleast_1d(x)
    conds = [x < 1, (x > 1) & (x < 2), x > 2]
    if der == 0:
        funcs = [lambda x: x*x/2.,
                 lambda x: 3./4 - (x-3./2)**2,
                 lambda x: (3.-x)**2 / 2]
    elif der == 2:
        funcs = [lambda x: 1.,
                 lambda x: -2.,
                 lambda x: 1.]
    else:
        # 如果程序执行到这里，抛出值错误异常，指示不应该到达这个分支，显示当前的 der 变量的值
        raise ValueError('never be here: der=%s' % der)
    # 使用 np.piecewise 函数根据条件数组 conds 和函数数组 funcs 对数组 x 进行分段处理
    pieces = np.piecewise(x, conds, funcs)
    # 返回分段处理后的数组 pieces
    return pieces
def _make_random_spline(n=35, k=3):
    # 设置随机数生成种子为123
    np.random.seed(123)
    # 生成n+k+1个随机数，并按升序排列，构成节点向量t
    t = np.sort(np.random.random(n+k+1))
    # 生成长度为n的随机系数向量c
    c = np.random.random(n)
    # 使用节点向量t和系数向量c构造并返回B样条曲线对象
    return BSpline.construct_fast(t, c, k)


def _make_multiples(b):
    """增加节点的重复度。"""
    c, k = b.c, b.k

    # 复制节点向量t，并将索引为17和18的元素设置为索引为17的值，构成新的节点向量t1
    t1 = b.t.copy()
    t1[17:19] = t1[17]
    # 使用节点向量t1和系数向量c构造并返回BSpline对象
    yield BSpline(t1, c, k)

    # 复制节点向量t，并将前k+1个元素设置为索引为0的值，构成新的节点向量t1
    t1 = b.t.copy()
    t1[:k+1] = t1[0]
    # 使用节点向量t1和系数向量c构造并返回BSpline对象
    yield BSpline(t1, c, k)

    # 复制节点向量t，并将最后k+1个元素设置为最后一个元素的值，构成新的节点向量t1
    t1 = b.t.copy()
    t1[-k-1:] = t1[-1]
    # 使用节点向量t1和系数向量c构造并返回BSpline对象
    yield BSpline(t1, c, k)


class TestInterop:
    #
    # 测试 FITPACK-based spline 函数能否处理 BSpline 对象
    #
    def setup_method(self):
        # 生成0到4π之间均匀分布的41个点作为自变量
        xx = np.linspace(0, 4.*np.pi, 41)
        # 计算这些点的余弦值作为因变量
        yy = np.cos(xx)
        # 生成基于(xx, yy)的插值样条对象b
        b = make_interp_spline(xx, yy)
        # 将插值样条对象b的节点向量、系数向量和次数存储在self.tck中
        self.tck = (b.t, b.c, b.k)
        self.xx, self.yy, self.b = xx, yy, b

        # 生成0到4π之间均匀分布的21个点作为新的自变量
        self.xnew = np.linspace(0, 4.*np.pi, 21)

        # 将系数向量b.c重复成为3列，并存储在self.c2中
        c2 = np.c_[b.c, b.c, b.c]
        # 使用插值样条对象b的节点向量和重复后的系数向量c2构造BSpline对象，并存储在self.b2中
        self.b2 = BSpline(b.t, self.c2, b.k)

    def test_splev(self):
        xnew, b, b2 = self.xnew, self.b, self.b2

        # 检查 splev 能否处理1维系数数组和标量x
        assert_allclose(splev(xnew, b),
                        b(xnew), atol=1e-15, rtol=1e-15)
        # 检查 splev 能否处理1维系数数组和tck元组
        assert_allclose(splev(xnew, b.tck),
                        b(xnew), atol=1e-15, rtol=1e-15)
        # 检查 splev 返回的结果是否与 b(xnew) 相近
        assert_allclose([splev(x, b) for x in xnew],
                        b(xnew), atol=1e-15, rtol=1e-15)

        # 对于N维系数，有一个问题：
        # splev(x, BSpline) 等同于 BSpline(x)，会引发 ValueError 异常
        with assert_raises(ValueError, match="Calling splev.. with BSpline"):
            splev(xnew, b2)

        # 然而，splev(x, BSpline.tck) 需要进行一些转置操作。因为
        # BSpline 沿着第一个轴插值，而传统的 FITPACK 封装器执行的是 list(map(...))，
        # 这实际上沿着最后一个轴插值。如下所示：
        sh = tuple(range(1, b2.c.ndim)) + (0,)   # sh = (1, 2, 0)
        cc = b2.c.transpose(sh)
        tck = (b2.t, cc, b2.k)
        # 检查 splev 返回的结果是否与 b2(xnew) 的转置相近
        assert_allclose(splev(xnew, tck),
                        b2(xnew).transpose(sh), atol=1e-15, rtol=1e-15)
    # 定义一个测试函数 test_splrep，用于测试 splrep 函数的功能
    def test_splrep(self):
        # 从测试实例的属性中获取 x 和 y 数据
        x, y = self.xx, self.yy
        
        # 使用 splrep 函数计算样条插值
        tck = splrep(x, y)
        
        # 调用 _impl.splrep 函数获取相同数据
        t, c, k = _impl.splrep(x, y)
        
        # 断言 tck 的输出与 _impl.splrep 的输出在给定精度下相等
        assert_allclose(tck[0], t, atol=1e-15)
        assert_allclose(tck[1], c, atol=1e-15)
        assert_equal(tck[2], k)

        # 测试 full_output=True 分支的情况
        tck_f, _, _, _ = splrep(x, y, full_output=True)
        
        # 断言 full_output=True 模式下的输出与 _impl.splrep 的输出相等
        assert_allclose(tck_f[0], t, atol=1e-15)
        assert_allclose(tck_f[1], c, atol=1e-15)
        assert_equal(tck_f[2], k)

        # 测试 splrep 的结果能否与 splev 逆向运算一致：
        # 在原始的 x 点上评估样条曲线
        yy = splev(x, tck)
        
        # 断言原始数据 y 与 splev 计算得到的 yy 在给定精度下相等
        assert_allclose(y, yy, atol=1e-15)

        # 还需要测试在 BSpline 封装下的结果是否一致
        b = BSpline(*tck)
        
        # 断言原始数据 y 与 BSpline 计算得到的结果在给定精度下相等
        assert_allclose(y, b(x), atol=1e-15)

    # 定义一个测试函数 test_splrep_errors，用于测试 splrep 函数对异常情况的处理
    def test_splrep_errors(self):
        # 测试对于多维度的 y 数组（n > 1），"old" 和 "new" splrep 都应该引发 ValueError
        x, y = self.xx, self.yy
        y2 = np.c_[y, y]
        
        # 使用 assert_raises 检查是否引发了 ValueError 异常
        with assert_raises(ValueError):
            splrep(x, y2)
        with assert_raises(ValueError):
            _impl.splrep(x, y2)

        # 测试输入数据长度小于最小要求时，是否引发 TypeError 异常
        with assert_raises(TypeError, match="m > k must hold"):
            splrep(x[:3], y[:3])
        with assert_raises(TypeError, match="m > k must hold"):
            _impl.splrep(x[:3], y[:3])

    # 定义一个测试函数 test_splprep，用于测试 splprep 函数的功能
    def test_splprep(self):
        # 创建一个多维数组 x，并使用 splprep 函数计算样条插值
        x = np.arange(15).reshape((3, 5))
        b, u = splprep(x)
        
        # 调用 _impl.splprep 函数获取相同数据
        tck, u1 = _impl.splprep(x)

        # 断言 splprep 和 _impl.splprep 的输出 u 在给定精度下相等
        assert_allclose(u, u1, atol=1e-15)
        
        # 断言 splev 在 splprep 和 _impl.splprep 输出的结果上的计算结果在给定精度下相等
        assert_allclose(splev(u, b), x, atol=1e-15)
        assert_allclose(splev(u, tck), x, atol=1e-15)

        # 测试 full_output=True 分支的情况
        (b_f, u_f), _, _, _ = splprep(x, s=0, full_output=True)
        
        # 断言 full_output=True 模式下的输出 u 与 _impl.splprep 的输出 u_f 相等
        assert_allclose(u, u_f, atol=1e-15)
        
        # 断言 splev 在 full_output=True 模式下的计算结果在给定精度下与 x 相等
        assert_allclose(splev(u_f, b_f), x, atol=1e-15)
    def test_splprep_errors(self):
        # 测试当 x.ndim > 2 时，"old" 和 "new" 代码路径都能引发异常
        x = np.arange(3*4*5).reshape((3, 4, 5))
        with assert_raises(ValueError, match="too many values to unpack"):
            splprep(x)
        with assert_raises(ValueError, match="too many values to unpack"):
            _impl.splprep(x)

        # 输入小于最小尺寸的情况
        x = np.linspace(0, 40, num=3)
        with assert_raises(TypeError, match="m > k must hold"):
            splprep([x])
        with assert_raises(TypeError, match="m > k must hold"):
            _impl.splprep([x])

        # 自动计算的参数值非递增
        # 参见 gh-7589
        x = [-50.49072266, -50.49072266, -54.49072266, -54.49072266]
        with assert_raises(ValueError, match="Invalid inputs"):
            splprep([x])
        with assert_raises(ValueError, match="Invalid inputs"):
            _impl.splprep([x])

        # 给定非递增的参数值 u
        x = [1, 3, 2, 4]
        u = [0, 0.3, 0.2, 1]
        with assert_raises(ValueError, match="Invalid inputs"):
            splprep(*[[x], None, u])

    def test_sproot(self):
        b, b2 = self.b, self.b2
        roots = np.array([0.5, 1.5, 2.5, 3.5])*np.pi
        # sproot 接受具有一维系数数组的 BSpline 对象
        assert_allclose(sproot(b), roots, atol=1e-7, rtol=1e-7)
        assert_allclose(sproot((b.t, b.c, b.k)), roots, atol=1e-7, rtol=1e-7)

        # ... 并且处理系数数组为 N 维时的尾随维度
        with assert_raises(ValueError, match="Calling sproot.. with BSpline"):
            sproot(b2, mest=50)

        # 对于带有 N 维系数的 tck 元组，保留传统行为
        c2r = b2.c.transpose(1, 2, 0)
        rr = np.asarray(sproot((b2.t, c2r, b2.k), mest=50))
        assert_equal(rr.shape, (3, 2, 4))
        assert_allclose(rr - roots, 0, atol=1e-12)

    def test_splint(self):
        # 测试 splint 接受 BSpline 对象
        b, b2 = self.b, self.b2
        assert_allclose(splint(0, 1, b),
                        splint(0, 1, b.tck), atol=1e-14)
        assert_allclose(splint(0, 1, b),
                        b.integrate(0, 1), atol=1e-14)

        # ... 并且处理系数数组为 N 维的情况
        with assert_raises(ValueError, match="Calling splint.. with BSpline"):
            splint(0, 1, b2)

        # 对于带有 N 维系数的 tck 元组，保留传统行为
        c2r = b2.c.transpose(1, 2, 0)
        integr = np.asarray(splint(0, 1, (b2.t, c2r, b2.k)))
        assert_equal(integr.shape, (3, 2))
        assert_allclose(integr,
                        splint(0, 1, b), atol=1e-14)
    def test_splder(self):
        for b in [self.b, self.b2]:
            # pad the c array (FITPACK convention)
            # 计算需要填充的长度，使得 b.c 和 b.t 长度相同
            ct = len(b.t) - len(b.c)
            if ct > 0:
                # 在 b.c 的末尾添加零以匹配 b.t 的长度
                b.c = np.r_[b.c, np.zeros((ct,) + b.c.shape[1:])]

            for n in [1, 2, 3]:
                # 计算 b 的 n 阶导数
                bd = splder(b)
                # 使用 _impl 模块计算 b 的 n 阶导数
                tck_d = _impl.splder((b.t, b.c, b.k))
                # 断言 bd 的节点 t 与 tck_d 的第一个元素相等
                assert_allclose(bd.t, tck_d[0], atol=1e-15)
                # 断言 bd 的系数 c 与 tck_d 的第二个元素相等
                assert_allclose(bd.c, tck_d[1], atol=1e-15)
                # 断言 bd 的次数 k 与 tck_d 的第三个元素相等
                assert_equal(bd.k, tck_d[2])
                # 断言 bd 是 BSpline 类的实例
                assert_(isinstance(bd, BSpline))
                # 断言 tck_d 是元组的实例，用于向后兼容性检查：输入和输出都是 tck
                assert_(isinstance(tck_d, tuple))  # back-compat: tck in and out

    def test_splantider(self):
        for b in [self.b, self.b2]:
            # pad the c array (FITPACK convention)
            # 计算需要填充的长度，使得 b.c 和 b.t 长度相同
            ct = len(b.t) - len(b.c)
            if ct > 0:
                # 在 b.c 的末尾添加零以匹配 b.t 的长度
                b.c = np.r_[b.c, np.zeros((ct,) + b.c.shape[1:])]

            for n in [1, 2, 3]:
                # 计算 b 的 n 阶导数
                bd = splantider(b)
                # 使用 _impl 模块计算 b 的 n 阶导数
                tck_d = _impl.splantider((b.t, b.c, b.k))
                # 断言 bd 的节点 t 与 tck_d 的第一个元素相等
                assert_allclose(bd.t, tck_d[0], atol=1e-15)
                # 断言 bd 的系数 c 与 tck_d 的第二个元素相等
                assert_allclose(bd.c, tck_d[1], atol=1e-15)
                # 断言 bd 的次数 k 与 tck_d 的第三个元素相等
                assert_equal(bd.k, tck_d[2])
                # 断言 bd 是 BSpline 类的实例
                assert_(isinstance(bd, BSpline))
                # 断言 tck_d 是元组的实例，用于向后兼容性检查：输入和输出都是 tck
                assert_(isinstance(tck_d, tuple))  # back-compat: tck in and out

    def test_insert(self):
        b, b2, xx = self.b, self.b2, self.xx

        j = b.t.size // 2
        tn = 0.5*(b.t[j] + b.t[j+1])

        # 插入节点 tn 到 b 中，返回新的 B 样条曲线 bn
        bn, tck_n = insert(tn, b), insert(tn, (b.t, b.c, b.k))
        # 断言使用 bn 和 tck_n 计算 xx 的值得到的结果相等
        assert_allclose(splev(xx, bn),
                        splev(xx, tck_n), atol=1e-15)
        # 断言 bn 是 BSpline 类的实例
        assert_(isinstance(bn, BSpline))
        # 断言 tck_n 是元组的实例，用于向后兼容性检查：输入和输出都是 tck
        assert_(isinstance(tck_n, tuple))   # back-compat: tck in, tck out

        # 对于 N 维系数数组，需要对 BSpline.c 进行转置
        # 转置后，结果是等效的。
        sh = tuple(range(b2.c.ndim))
        c_ = b2.c.transpose(sh[1:] + (0,))
        tck_n2 = insert(tn, (b2.t, c_, b2.k))

        bn2 = insert(tn, b2)

        # 需要对比结果进行转置，参考 test_splev
        assert_allclose(np.asarray(splev(xx, tck_n2)).transpose(2, 0, 1),
                        bn2(xx), atol=1e-15)
        # 断言 bn2 是 BSpline 类的实例
        assert_(isinstance(bn2, BSpline))
        # 断言 tck_n2 是元组的实例，用于向后兼容性检查：输入和输出都是 tck
        assert_(isinstance(tck_n2, tuple))   # back-compat: tck in, tck out
class TestInterp:
    #
    # Test basic ways of constructing interpolating splines.
    #

    # 使用 NumPy 创建一个包含 0 到 2π 之间均匀分布的数组
    xx = np.linspace(0., 2.*np.pi)
    # 对 xx 中的每个元素计算正弦值，创建对应的数组 yy
    yy = np.sin(xx)

    # 测试非整数阶数的情况
    def test_non_int_order(self):
        # 使用 assert_raises 检测是否会抛出 TypeError 异常
        with assert_raises(TypeError):
            make_interp_spline(self.xx, self.yy, k=2.5)

    # 测试阶数为 0 的情况
    def test_order_0(self):
        # 创建阶数为 0 的插值样条对象 b，并验证其在 xx 点处的值接近 yy
        b = make_interp_spline(self.xx, self.yy, k=0)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        # 使用 axis=-1 参数创建阶数为 0 的插值样条对象 b，并验证其在 xx 点处的值接近 yy
        b = make_interp_spline(self.xx, self.yy, k=0, axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    # 测试线性插值的情况
    def test_linear(self):
        # 创建线性插值样条对象 b，并验证其在 xx 点处的值接近 yy
        b = make_interp_spline(self.xx, self.yy, k=1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        # 使用 axis=-1 参数创建线性插值样条对象 b，并验证其在 xx 点处的值接近 yy
        b = make_interp_spline(self.xx, self.yy, k=1, axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    # 测试 x 和 y 数组形状不兼容的情况
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_incompatible_x_y(self, k):
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4, 5, 6, 7]
        # 使用 assert_raises 检测是否会抛出 ValueError 异常，并匹配特定的错误消息
        with assert_raises(ValueError, match="Shapes of x"):
            make_interp_spline(x, y, k=k)

    # 测试 x 数组中存在重复值或未排序的情况
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x(self, k):
        x = [0, 1, 1, 2, 3, 4]      # 存在重复值
        y = [0, 1, 2, 3, 4, 5]
        # 使用 assert_raises 检测是否会抛出 ValueError 异常，并匹配特定的错误消息
        with assert_raises(ValueError, match="x to not have duplicates"):
            make_interp_spline(x, y, k=k)

        x = [0, 2, 1, 3, 4, 5]      # 未排序
        # 使用 assert_raises 检测是否会抛出 ValueError 异常，并匹配特定的错误消息
        with assert_raises(ValueError, match="Expect x to be a 1D strictly"):
            make_interp_spline(x, y, k=k)

        x = [0, 1, 2, 3, 4, 5]
        x = np.asarray(x).reshape((1, -1))     # 将 x 转换为二维数组
        # 使用 assert_raises 检测是否会抛出 ValueError 异常，并匹配特定的错误消息
        with assert_raises(ValueError, match="Expect x to be a 1D strictly"):
            make_interp_spline(x, y, k=k)

    # 测试非均匀节点插值的情况
    def test_not_a_knot(self):
        # 遍历不同的阶数 k，创建插值样条对象 b，并验证其在 xx 点处的值接近 yy
        for k in [3, 5]:
            b = make_interp_spline(self.xx, self.yy, k)
            assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    # 测试周期性插值的情况
    def test_periodic(self):
        # 使用 bc_type='periodic' 和 k=5 创建周期性插值样条对象 b，并验证其在 xx 点处的值接近 yy
        b = make_interp_spline(self.xx, self.yy, k=5, bc_type='periodic')
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        # 在周期性插值情况下，预期边界处 k-1 阶导数相等
        for i in range(1, 5):
            assert_allclose(b(self.xx[0], nu=i), b(self.xx[-1], nu=i), atol=1e-11)
        # 使用 axis=-1 参数，创建周期性插值样条对象 b，并验证其在 xx 点处的值接近 yy
        b = make_interp_spline(self.xx, self.yy, k=5, bc_type='periodic', axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        for i in range(1, 5):
            assert_allclose(b(self.xx[0], nu=i), b(self.xx[-1], nu=i), atol=1e-11)

    # 参数化测试，测试不同阶数 k 的情况
    @pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7])
    def test_periodic_random(self, k):
        # 测试两种情况（k > n 和 k <= n）
        n = 5
        np.random.seed(1234)
        # 生成一个长度为 n 的随机数列，并排序
        x = np.sort(np.random.random_sample(n) * 10)
        # 生成一个长度为 n 的随机数列，并赋予其范围为 [0, 100)
        y = np.random.random_sample(n) * 100
        # 将第一个和最后一个元素设为相等，以确保边界条件
        y[0] = y[-1]
        # 使用指定的参数创建一个周期性的样条插值对象
        b = make_interp_spline(x, y, k=k, bc_type='periodic')
        # 断言插值结果与原始 y 值接近
        assert_allclose(b(x), y, atol=1e-14)

    def test_periodic_axis(self):
        n = self.xx.shape[0]
        np.random.seed(1234)
        # 生成一个长度为 n 的随机数列，并映射到 [0, 2π] 区间内
        x = np.random.random_sample(n) * 2 * np.pi
        x = np.sort(x)
        # 将第一个和最后一个元素分别设为 0 和 2π，以确保周期性边界条件
        x[0] = 0.
        x[-1] = 2 * np.pi
        y = np.zeros((2, n))
        # 生成一个二维数组，其中第一行是 sin(x)，第二行是 cos(x)
        y[0] = np.sin(x)
        y[1] = np.cos(x)
        # 使用指定的参数创建一个周期性的样条插值对象，沿着 axis=1 的方向插值
        b = make_interp_spline(x, y, k=5, bc_type='periodic', axis=1)
        # 断言插值结果与原始 y 值接近
        for i in range(n):
            assert_allclose(b(x[i]), y[:, i], atol=1e-14)
        # 断言首尾两点的插值结果接近
        assert_allclose(b(x[0]), b(x[-1]), atol=1e-14)

    def test_periodic_points_exception(self):
        # 当预期是周期性情况时，首尾两点应该匹配
        np.random.seed(1234)
        k = 5
        n = 8
        x = np.sort(np.random.random_sample(n))
        y = np.random.random_sample(n)
        # 确保首尾两点不相等，以引发 ValueError 异常
        y[0] = y[-1] - 1  
        with assert_raises(ValueError):
            make_interp_spline(x, y, k=k, bc_type='periodic')

    def test_periodic_knots_exception(self):
        # `periodic` 情况不适用于传递的节点向量
        np.random.seed(1234)
        k = 3
        n = 7
        x = np.sort(np.random.random_sample(n))
        y = np.random.random_sample(n)
        t = np.zeros(n + 2 * k)
        with assert_raises(ValueError):
            make_interp_spline(x, y, k, t, 'periodic')

    @pytest.mark.parametrize('k', [2, 3, 4, 5])
    def test_periodic_splev(self, k):
        # 使用 splev 函数比较周期性 b 样条与 B 样条的值
        b = make_interp_spline(self.xx, self.yy, k=k, bc_type='periodic')
        tck = splrep(self.xx, self.yy, per=True, k=k)
        spl = splev(self.xx, tck)
        # 断言插值结果与 splev 函数的结果接近
        assert_allclose(spl, b(self.xx), atol=1e-14)

        # 比较周期性 b 样条的导数与 splev 函数的结果
        for i in range(1, k):
            spl = splev(self.xx, tck, der=i)
            # 断言插值导数结果与 splev 函数的结果接近
            assert_allclose(spl, b(self.xx, nu=i), atol=1e-10)

    def test_periodic_cubic(self):
        # 使用 CubicSpline 比较周期性立方 b 样条的值
        b = make_interp_spline(self.xx, self.yy, k=3, bc_type='periodic')
        cub = CubicSpline(self.xx, self.yy, bc_type='periodic')
        # 断言插值结果与 CubicSpline 函数的结果接近
        assert_allclose(b(self.xx), cub(self.xx), atol=1e-14)

        # 边界情况：三个点上的立方插值
        n = 3
        x = np.sort(np.random.random_sample(n) * 10)
        y = np.random.random_sample(n) * 100
        # 确保首尾两点相等，以测试周期性边界条件
        y[0] = y[-1]
        b = make_interp_spline(x, y, k=3, bc_type='periodic')
        cub = CubicSpline(x, y, bc_type='periodic')
        # 断言插值结果与 CubicSpline 函数的结果接近
        assert_allclose(b(x), cub(x), atol=1e-14)
    # 定义一个测试函数，用于测试周期性完整矩阵的比较值
    def test_periodic_full_matrix(self):
        # 创建一个三次周期性 B 样条插值对象，使用指定的边界条件
        k = 3
        b = make_interp_spline(self.xx, self.yy, k=k, bc_type='periodic')
        # 计算周期性结点向量
        t = _periodic_knots(self.xx, k)
        # 使用完整矩阵方法创建周期性插值
        c = _make_interp_per_full_matr(self.xx, self.yy, t, k)
        # 使用向量化的方式定义一个函数，计算 B 样条插值在给定点的值
        b1 = np.vectorize(lambda x: _naive_eval(x, t, c, k))
        # 断言两种方法的插值结果在给定的公差范围内相等
        assert_allclose(b(self.xx), b1(self.xx), atol=1e-14)

    # 定义一个测试函数，用于测试二次导数插值
    def test_quadratic_deriv(self):
        # 定义一个包含右侧边界导数信息的列表
        der = [(1, 8.)]  # order, value: f'(x) = 8.

        # 在右边界处应用导数边界条件创建二次样条插值对象
        b = make_interp_spline(self.xx, self.yy, k=2, bc_type=(None, der))
        # 断言插值结果与原始数据在给定公差范围内相等
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        # 断言插值函数在右边界处的一阶导数值与预期值相等
        assert_allclose(b(self.xx[-1], 1), der[0][1], atol=1e-14, rtol=1e-14)

        # 在左边界处应用导数边界条件创建二次样条插值对象
        b = make_interp_spline(self.xx, self.yy, k=2, bc_type=(der, None))
        # 断言插值结果与原始数据在给定公差范围内相等
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        # 断言插值函数在左边界处的一阶导数值与预期值相等
        assert_allclose(b(self.xx[0], 1), der[0][1], atol=1e-14, rtol=1e-14)

    # 定义一个测试函数，用于测试三次导数插值
    def test_cubic_deriv(self):
        k = 3

        # 定义左右边界处的一阶导数信息列表
        der_l, der_r = [(1, 3.)], [(1, 4.)]
        # 使用给定的导数边界条件创建三次样条插值对象
        b = make_interp_spline(self.xx, self.yy, k, bc_type=(der_l, der_r))
        # 断言插值结果与原始数据在给定公差范围内相等
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        # 断言插值函数在左右边界处的一阶导数值与预期值相等
        assert_allclose([b(self.xx[0], 1), b(self.xx[-1], 1)],
                        [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)

        # 'natural' 三次样条插值，边界处的二阶导数为零
        der_l, der_r = [(2, 0)], [(2, 0)]
        # 使用给定的导数边界条件创建三次样条插值对象
        b = make_interp_spline(self.xx, self.yy, k, bc_type=(der_l, der_r))
        # 断言插值结果与原始数据在给定公差范围内相等
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    # 定义一个测试函数，用于测试五次导数插值
    def test_quintic_derivs(self):
        k, n = 5, 7
        x = np.arange(n).astype(np.float64)
        y = np.sin(x)
        # 定义左右边界处的一阶和二阶导数信息列表
        der_l = [(1, -12.), (2, 1)]
        der_r = [(1, 8.), (2, 3.)]
        # 使用给定的导数边界条件创建五次样条插值对象
        b = make_interp_spline(x, y, k=k, bc_type=(der_l, der_r))
        # 断言插值结果与原始数据在给定公差范围内相等
        assert_allclose(b(x), y, atol=1e-14, rtol=1e-14)
        # 断言插值函数在左右边界处的一阶和二阶导数值与预期值相等
        assert_allclose([b(x[0], 1), b(x[0], 2)],
                        [val for (nu, val) in der_l])
        assert_allclose([b(x[-1], 1), b(x[-1], 2)],
                        [val for (nu, val) in der_r])

    # 标记为预期失败的测试函数，用于测试三次导数插值在某些情况下的不稳定性
    @pytest.mark.xfail(reason='unstable')
    def test_cubic_deriv_unstable(self):
        # 在 x[0] 处定义一阶和二阶导数信息，而在 x[-1] 处没有导数信息
        # 问题不在于它失败了[谁会使用这个呢]，
        # 问题在于它是*静默*失败的，我不知道如何检测这种不稳定性。
        # 在这种特定情况下：对于 len(t) < 20 是 OK 的，对于更大的 `len(t)` 则会失控。
        k = 3
        # 增广结点向量
        t = _augknt(self.xx, k)

        # 定义左边界处的一阶和二阶导数信息列表
        der_l = [(1, 3.), (2, 4.)]
        # 使用给定的导数边界条件创建三次样条插值对象
        b = make_interp_spline(self.xx, self.yy, k, t, bc_type=(der_l, None))
        # 断言插值结果与原始数据在给定公差范围内相等
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
    def test_knots_not_data_sites(self):
        # Knots need not coincide with the data sites.
        # 使用二次样条插值，结点位于数据点的平均位置，
        # 边缘处的二阶导数为零的额外约束条件
        k = 2
        # 构造结点向量 t
        t = np.r_[(self.xx[0],)*(k+1),
                  (self.xx[1:] + self.xx[:-1]) / 2.,
                  (self.xx[-1],)*(k+1)]
        # 创建二次样条插值对象 b
        b = make_interp_spline(self.xx, self.yy, k, t,
                               bc_type=([(2, 0)], [(2, 0)]))

        # 断言插值结果与原始数据 self.yy 接近
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        # 断言边缘处的二阶导数为零
        assert_allclose([b(self.xx[0], 2), b(self.xx[-1], 2)], [0., 0.],
                        atol=1e-14)

    def test_minimum_points_and_deriv(self):
        # 对 f(x) = x**3 在 [0, 1] 区间进行插值。f'(x) = 3 * xx**2
        # 边界条件为 f'(0) = 0, f'(1) = 3.
        k = 3
        x = [0., 1.]
        y = [0., 1.]
        # 创建三次样条插值对象 b
        b = make_interp_spline(x, y, k, bc_type=([(1, 0.)], [(1, 3.)]))

        # 计算 f(x) = x**3 在 [0, 1] 区间的理论值
        xx = np.linspace(0., 1.)
        yy = xx**3
        # 断言插值结果与理论值 yy 接近
        assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)

    def test_deriv_spec(self):
        # 如果省略其中一个导数，则样条定义是不完整的。
        x = y = [1.0, 2, 3, 4, 5, 6]

        # 使用 assert_raises 检查各种缺少导数定义的情况
        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=([(1, 0.)], None))

        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=(1, 0.))

        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=[(1, 0.)])

        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=42)

        # CubicSpline 预期 `bc_type=(left_pair, right_pair)`，而这里预期 `bc_type=(iterable, iterable)`
        l, r = (1, 0.0), (1, 0.0)
        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=(l, r))

    def test_deriv_order_too_large(self):
        x = np.arange(7)
        y = x**2
        l, r = [(6, 0)], [(1, 0)]    # 6th derivative = 0 at x[0] for k=3
        # 边缘条件中指定的导数阶数过大的情况下，使用 assert_raises 检查异常情况
        with assert_raises(ValueError, match="Bad boundary conditions at 0."):
            make_interp_spline(x, y, bc_type=(l, r))

        l, r = [(1, 0)], [(-6, 0)]    # derivative order < 0 at x[-1]
        with assert_raises(ValueError, match="Bad boundary conditions at 6."):
            make_interp_spline(x, y, bc_type=(l, r))
    def test_complex(self):
        # 设置样条插值的阶数
        k = 3
        # 获取测试用例中的自变量 xx
        xx = self.xx
        # 获取测试用例中的因变量 yy，并转换为复数形式
        yy = self.yy + 1.j*self.yy

        # 设置左右边界处的一阶导数
        der_l, der_r = [(1, 3.j)], [(1, 4.+2.j)]
        # 使用给定的边界条件创建样条插值对象
        b = make_interp_spline(xx, yy, k, bc_type=(der_l, der_r))
        # 断言插值结果与原始因变量数据相近
        assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)
        # 断言边界处一阶导数的插值结果与预期值相近
        assert_allclose([b(xx[0], 1), b(xx[-1], 1)],
                        [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)

        # 进行零阶和一阶导数的额外测试
        for k in (0, 1):
            # 创建对应阶数的样条插值对象
            b = make_interp_spline(xx, yy, k=k)
            # 断言插值结果与原始因变量数据相近
            assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)

    def test_int_xy(self):
        # 创建整数类型的自变量和因变量数组
        x = np.arange(10).astype(int)
        y = np.arange(10).astype(int)

        # Cython 在 "buffer type mismatch"（构造时）或 "no matching signature found"（评估时）时会出错
        for k in (0, 1, 2, 3):
            # 创建对应阶数的样条插值对象
            b = make_interp_spline(x, y, k=k)
            # 对插值对象进行计算
            b(x)

    def test_sliced_input(self):
        # Cython 代码在非 C 连续数组上会出错
        xx = np.linspace(-1, 1, 100)

        # 对自变量和因变量进行切片
        x = xx[::5]
        y = xx[::5]

        for k in (0, 1, 2, 3):
            # 创建对应阶数的样条插值对象
            make_interp_spline(x, y, k=k)

    def test_check_finite(self):
        # check_finite 默认为 True；NaN 和无穷大会触发 ValueError
        x = np.arange(10).astype(float)
        y = x**2

        for z in [np.nan, np.inf, -np.inf]:
            # 将因变量数组中最后一个元素设置为 z
            y[-1] = z
            # 断言调用时会触发 ValueError
            assert_raises(ValueError, make_interp_spline, x, y)

    @pytest.mark.parametrize('k', [1, 2, 3, 5])
    def test_list_input(self, k):
        # 对 gh-8714 的回归测试：当 x, y 为列表且 k=2 时，会出现 TypeError
        x = list(range(10))
        y = [a**2 for a in x]
        # 创建对应阶数的样条插值对象
        make_interp_spline(x, y, k=k)

    def test_multiple_rhs(self):
        # 创建包含正弦和余弦函数值的因变量数组
        yy = np.c_[np.sin(self.xx), np.cos(self.xx)]
        # 设置左右边界处的一阶导数
        der_l = [(1, [1., 2.])]
        der_r = [(1, [3., 4.])]

        # 使用给定的边界条件创建多因变量的样条插值对象
        b = make_interp_spline(self.xx, yy, k=3, bc_type=(der_l, der_r))
        # 断言插值结果与原始因变量数据相近
        assert_allclose(b(self.xx), yy, atol=1e-14, rtol=1e-14)
        # 断言边界处一阶导数的插值结果与预期值相近
        assert_allclose(b(self.xx[0], 1), der_l[0][1], atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[-1], 1), der_r[0][1], atol=1e-14, rtol=1e-14)

    def test_shapes(self):
        np.random.seed(1234)
        k, n = 3, 22
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=(n, 5, 6, 7))

        # 创建对应阶数的多维数组的样条插值对象
        b = make_interp_spline(x, y, k)
        # 断言插值系数数组的形状与预期一致
        assert_equal(b.c.shape, (n, 5, 6, 7))

        # 添加一阶导数的测试
        d_l = [(1, np.random.random((5, 6, 7)))]
        d_r = [(1, np.random.random((5, 6, 7)))]
        # 使用给定的边界条件创建多维数组的样条插值对象
        b = make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        # 断言插值系数数组的形状与预期一致
        assert_equal(b.c.shape, (n + k - 1, 5, 6, 7))
    def test_string_aliases(self):
        yy = np.sin(self.xx)  # 计算 self.xx 数组中每个元素的正弦值，并赋给 yy

        # 使用单个字符串 'natural' 创建样条插值对象 b1
        b1 = make_interp_spline(self.xx, yy, k=3, bc_type='natural')
        # 使用具体的边界条件列表创建样条插值对象 b2
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=([(2, 0)], [(2, 0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)  # 断言 b1 和 b2 的系数 c 接近

        # 使用两个字符串 'natural' 和 'clamped' 创建样条插值对象 b1
        b1 = make_interp_spline(self.xx, yy, k=3,
                                bc_type=('natural', 'clamped'))
        # 使用具体的边界条件列表创建样条插值对象 b2
        b2 = make_interp_spline(self.xx, yy, k=3,
                                bc_type=([(2, 0)], [(1, 0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)  # 断言 b1 和 b2 的系数 c 接近

        # 使用单边界条件 'clamped' 创建样条插值对象 b1
        b1 = make_interp_spline(self.xx, yy, k=2, bc_type=(None, 'clamped'))
        # 使用具体的边界条件列表创建样条插值对象 b2
        b2 = make_interp_spline(self.xx, yy, k=2, bc_type=(None, [(1, 0.0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)  # 断言 b1 和 b2 的系数 c 接近

        # 使用字符串 'not-a-knot' 创建样条插值对象 b1
        b1 = make_interp_spline(self.xx, yy, k=3, bc_type='not-a-knot')
        # 使用 None 创建样条插值对象 b2，'not-a-knot' 等价于 None
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=None)
        assert_allclose(b1.c, b2.c, atol=1e-15)  # 断言 b1 和 b2 的系数 c 接近

        # 使用未知字符串 'typo' 创建样条插值对象时，应引发 ValueError 异常
        with assert_raises(ValueError):
            make_interp_spline(self.xx, yy, k=3, bc_type='typo')

        # 对于二维 yy 值，使用字符串别名处理样条插值对象 b1
        yy = np.c_[np.sin(self.xx), np.cos(self.xx)]
        der_l = [(1, [0., 0.])]
        der_r = [(2, [0., 0.])]
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=(der_l, der_r))
        # 使用字符串别名创建样条插值对象 b1
        b1 = make_interp_spline(self.xx, yy, k=3,
                                bc_type=('clamped', 'natural'))
        assert_allclose(b1.c, b2.c, atol=1e-15)  # 断言 b1 和 b2 的系数 c 接近

        # 对于 N 维 yy 值，使用字符串别名处理样条插值对象 b1
        np.random.seed(1234)
        k, n = 3, 22
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=(n, 5, 6, 7))

        # 添加一些导数值
        d_l = [(1, np.zeros((5, 6, 7)))]
        d_r = [(1, np.zeros((5, 6, 7)))]
        b1 = make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        # 使用字符串别名 'clamped' 创建样条插值对象 b2
        b2 = make_interp_spline(x, y, k, bc_type='clamped')
        assert_allclose(b1.c, b2.c, atol=1e-15)  # 断言 b1 和 b2 的系数 c 接近
    def test_woodbury(self):
        '''
        Random elements in diagonal matrix with blocks in the
        left lower and right upper corners checking the
        implementation of Woodbury algorithm.
        '''
        # 设置随机种子以保证结果可重复
        np.random.seed(1234)
        # 设置矩阵的维度
        n = 201
        # 迭代不同的 k 值
        for k in range(3, 32, 2):
            # 计算偏移量
            offset = int((k - 1) / 2)
            # 创建一个对角矩阵，其中对角线元素为随机值
            a = np.diagflat(np.random.random((1, n)))
            # 对角线矩阵的左下角和右上角添加随机块
            for i in range(1, offset + 1):
                a[:-i, i:] += np.diagflat(np.random.random((1, n - i)))
                a[i:, :-i] += np.diagflat(np.random.random((1, n - i)))
            # 创建随机的右上角矩阵 ur 和左下角矩阵 ll
            ur = np.random.random((offset, offset))
            a[:offset, -offset:] = ur
            ll = np.random.random((offset, offset))
            a[-offset:, :offset] = ll
            # 创建一个空的 k x n 的数组 d
            d = np.zeros((k, n))
            # 填充数组 d，使用 np.diagonal 从矩阵 a 中提取对角线元素
            for i, j in enumerate(range(offset, -offset - 1, -1)):
                if j < 0:
                    d[i, :j] = np.diagonal(a, offset=j)
                else:
                    d[i, j:] = np.diagonal(a, offset=j)
            # 创建随机向量 b
            b = np.random.random(n)
            # 使用 _woodbury_algorithm 函数计算 Woodbury 算法的结果，并验证与 np.linalg.solve 的结果是否接近
            assert_allclose(_woodbury_algorithm(d, ur, ll, b, k),
                            np.linalg.solve(a, b), atol=1e-14)
# 使用全矩阵方法组装阶数为 k 的样条插值，以 knots t 来插值 y(x)
# 仅支持 Not-a-knot 边界条件

def make_interp_full_matr(x, y, t, k):
    """Assemble an spline order k with knots t to interpolate
    y(x) using full matrices.
    Not-a-knot BC only.

    This routine is here for testing only (even though it's functional).
    """
    # 确保 x 和 y 的长度相同
    assert x.size == y.size
    # 确保 knots t 的长度满足条件
    assert t.size == x.size + k + 1
    n = x.size

    # 创建一个全零矩阵 A，用于存储插值所需的系数
    A = np.zeros((n, n), dtype=np.float64)

    # 遍历每个数据点
    for j in range(n):
        xval = x[j]
        # 确定 xval 属于的区间
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1

        # 使用 _bspl.evaluate_all_bspl 计算 B 样条基函数的值
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        # 将计算得到的基函数值填充到矩阵 A 的相应行中
        A[j, left-k:left+1] = bb

    # 解线性方程 A * c = y，得到系数向量 c
    c = sl.solve(A, y)
    return c


def make_lsq_full_matrix(x, y, t, k=3):
    """Make the least-square spline, full matrices."""
    # 将 x, y, t 转换为 NumPy 数组
    x, y, t = map(np.asarray, (x, y, t))
    m = x.size
    n = t.size - k - 1

    # 创建一个全零矩阵 A，用于存储最小二乘法样条插值所需的系数
    A = np.zeros((m, n), dtype=np.float64)

    # 遍历每个数据点
    for j in range(m):
        xval = x[j]
        # 确定 xval 属于的区间
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1

        # 使用 _bspl.evaluate_all_bspl 计算 B 样条基函数的值
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        # 将计算得到的基函数值填充到矩阵 A 的相应行中
        A[j, left-k:left+1] = bb

    # 构建观测矩阵 B = A^T * A 和目标向量 Y = A^T * y，用于求解最小二乘问题
    B = np.dot(A.T, A)
    Y = np.dot(A.T, y)
    # 解线性方程 B * c = Y，得到系数向量 c
    c = sl.solve(B, Y)

    return c, (A, Y)


class TestLSQ:
    #
    # Test make_lsq_spline
    #
    np.random.seed(1234)
    n, k = 13, 3
    x = np.sort(np.random.random(n))
    y = np.random.random(n)
    t = _augknt(np.linspace(x[0], x[-1], 7), k)

    def test_lstsq(self):
        # check LSQ construction vs a full matrix version
        x, y, t, k = self.x, self.y, self.t, self.k

        # 调用 make_lsq_full_matrix 函数获取最小二乘法样条插值的系数 c0 和相关矩阵 AY
        c0, AY = make_lsq_full_matrix(x, y, t, k)
        # 调用 make_lsq_spline 函数获取最小二乘法样条插值对象 b
        b = make_lsq_spline(x, y, t, k)

        # 断言两种方法计算的系数 c0 和 b.c 相等
        assert_allclose(b.c, c0)
        # 断言插值对象 b 的系数形状为 (t.size - k - 1,)
        assert_equal(b.c.shape, (t.size - k - 1,))

        # 同时使用 numpy.linalg.lstsq 验证结果
        aa, yy = AY
        c1, _, _, _ = np.linalg.lstsq(aa, y, rcond=-1)
        assert_allclose(b.c, c1)

    def test_weights(self):
        # weights = 1 is same as None
        x, y, t, k = self.x, self.y, self.t, self.k
        w = np.ones_like(x)

        # 调用 make_lsq_spline 函数分别计算权重为 1 和 None 时的插值对象 b 和 b_w
        b = make_lsq_spline(x, y, t, k)
        b_w = make_lsq_spline(x, y, t, k, w=w)

        # 断言两种插值对象 b 和 b_w 的节点 t 相等
        assert_allclose(b.t, b_w.t, atol=1e-14)
        # 断言两种插值对象 b 和 b_w 的系数 c 相等
        assert_allclose(b.c, b_w.c, atol=1e-14)
        # 断言两种插值对象 b 和 b_w 的阶数 k 相等
        assert_equal(b.k, b_w.k)

    def test_multiple_rhs(self):
        x, t, k, n = self.x, self.t, self.k, self.n
        # 创建一个多维度的随机数据 y
        y = np.random.random(size=(n, 5, 6, 7))

        # 调用 make_lsq_spline 函数计算多维度数据的插值对象 b
        b = make_lsq_spline(x, y, t, k)
        # 断言插值对象 b 的系数形状正确
        assert_equal(b.c.shape, (t.size-k-1, 5, 6, 7))

    def test_complex(self):
        # cmplx-valued `y`
        x, t, k = self.x, self.t, self.k
        # 创建复数类型的数据 yc
        yc = self.y * (1. + 2.j)

        # 调用 make_lsq_spline 函数分别计算复数数据 yc, yc.real 和 yc.imag 的插值对象
        b = make_lsq_spline(x, yc, t, k)
        b_re = make_lsq_spline(x, yc.real, t, k)
        b_im = make_lsq_spline(x, yc.imag, t, k)

        # 断言复数插值对象 b(x) 的实部和虚部与分别计算的实部和虚部插值对象的和相等
        assert_allclose(b(x), b_re(x) + 1.j*b_im(x), atol=1e-15, rtol=1e-15)
    # 定义一个测试方法，用于测试整数类型的输入数据
    def test_int_xy(self):
        # 创建一个长度为10的整数类型的NumPy数组
        x = np.arange(10).astype(int)
        # 创建另一个长度为10的整数类型的NumPy数组
        y = np.arange(10).astype(int)
        # 对输入的x进行增广节点处理，生成参数t
        t = _augknt(x, k=1)
        # 调用make_lsq_spline函数，使用x, y, t作为输入参数，执行最小二乘样条插值

    # 定义一个测试方法，用于测试切片输入数据
    def test_sliced_input(self):
        # 创建一个从-1到1均匀分布的长度为100的NumPy数组
        xx = np.linspace(-1, 1, 100)
        # 从xx数组中每隔3个元素取一个元素，生成新的数组x
        x = xx[::3]
        # 同样从xx数组中每隔3个元素取一个元素，生成新的数组y
        y = xx[::3]
        # 对输入的x进行增广节点处理，生成参数t
        t = _augknt(x, 1)
        # 调用make_lsq_spline函数，使用x, y, t作为输入参数，执行最小二乘样条插值

    # 定义一个测试方法，用于检查数据中是否包含无穷大或NaN值
    def test_checkfinite(self):
        # 创建一个长度为12的浮点数类型的NumPy数组
        x = np.arange(12).astype(float)
        # 对x数组的每个元素求平方，生成y数组
        y = x**2
        # 对输入的x进行增广节点处理，生成参数t
        t = _augknt(x, 3)

        # 遍历包含NaN和无穷大值的列表
        for z in [np.nan, np.inf, -np.inf]:
            # 将y数组的最后一个元素替换为z
            y[-1] = z
            # 断言调用make_lsq_spline函数时会触发ValueError异常
            assert_raises(ValueError, make_lsq_spline, x, y, t)

    # 定义一个测试方法，用于检查make_lsq_spline函数是否支持只读数组作为输入
    def test_read_only(self):
        # 获取已经定义好的self.x, self.y, self.t作为输入参数x, y, t
        x, y, t = self.x, self.y, self.t
        # 将x数组设置为只读模式
        x.setflags(write=False)
        # 将y数组设置为只读模式
        y.setflags(write=False)
        # 将t数组设置为只读模式
        t.setflags(write=False)
        # 调用make_lsq_spline函数，使用只读模式的x, y, t作为输入参数，执行最小二乘样条插值
class TestSmoothingSpline:
    #
    # test make_smoothing_spline
    #
    def test_invalid_input(self):
        # 设定随机种子以确保结果可重现
        np.random.seed(1234)
        # 定义样本点数量
        n = 100
        # 生成在区间[-2, 2]内均匀分布的随机数，并排序
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        # 根据指定的函数生成 y 值，同时加入正态分布噪声
        y = x**2 * np.sin(4 * x) + x**3 + np.random.normal(0., 1.5, n)

        # ``x`` 和 ``y`` 应该具有相同的形状（都是一维数组）
        with assert_raises(ValueError):
            make_smoothing_spline(x, y[1:])
        with assert_raises(ValueError):
            make_smoothing_spline(x[1:], y)
        with assert_raises(ValueError):
            make_smoothing_spline(x.reshape(1, n), y)

        # ``x`` 应该是一个升序数组
        with assert_raises(ValueError):
            make_smoothing_spline(x[::-1], y)

        # 创建 ``x`` 的副本，并修改第一个元素使其重复
        x_dupl = np.copy(x)
        x_dupl[0] = x_dupl[1]

        with assert_raises(ValueError):
            make_smoothing_spline(x_dupl, y)

        # 当 ``x`` 和 ``y`` 的长度小于 5 时应抛出异常
        x = np.arange(4)
        y = np.ones(4)
        exception_message = "``x`` and ``y`` length must be at least 5"
        with pytest.raises(ValueError, match=exception_message):
            make_smoothing_spline(x, y)
    def test_compare_with_GCVSPL(self):
        """
        Data is generated in the following way:
        >>> np.random.seed(1234)
        >>> n = 100
        >>> x = np.sort(np.random.random_sample(n) * 4 - 2)
        >>> y = np.sin(x) + np.random.normal(scale=.5, size=n)
        >>> np.savetxt('x.csv', x)
        >>> np.savetxt('y.csv', y)

        We obtain the result of performing the GCV smoothing splines
        package (by Woltring, gcvspl) on the sample data points
        using its version for Octave (https://github.com/srkuberski/gcvspl).
        In order to use this implementation, one should clone the repository
        and open the folder in Octave.
        In Octave, we load up ``x`` and ``y`` (generated from Python code
        above):

        >>> x = csvread('x.csv');
        >>> y = csvread('y.csv');

        Then, in order to access the implementation, we compile gcvspl files in
        Octave:

        >>> mex gcvsplmex.c gcvspl.c
        >>> mex spldermex.c gcvspl.c

        The first function computes the vector of unknowns from the dataset
        (x, y) while the second one evaluates the spline in certain points
        with known vector of coefficients.

        >>> c = gcvsplmex( x, y, 2 );
        >>> y0 = spldermex( x, c, 2, x, 0 );

        If we want to compare the results of the gcvspl code, we can save
        ``y0`` in csv file:

        >>> csvwrite('y0.csv', y0);

        """
        # load the data sample
        with np.load(data_file('gcvspl.npz')) as data:
            # data points
            x = data['x']  # load x coordinates from the saved data
            y = data['y']  # load corresponding y coordinates from the saved data

            y_GCVSPL = data['y_GCVSPL']  # load precomputed results from GCVSPL

        # compute the spline using custom function make_smoothing_spline
        y_compr = make_smoothing_spline(x, y)(x)

        # assertion to compare computed spline with GCVSPL results
        # using tolerance due to iterative algorithm variations in GCV criteria
        assert_allclose(y_compr, y_GCVSPL, atol=1e-4, rtol=1e-4)

    def test_non_regularized_case(self):
        """
        In case the regularization parameter is 0, the resulting spline
        is an interpolation spline with natural boundary conditions.
        """
        # create data sample
        np.random.seed(1234)
        n = 100
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        y = x**2 * np.sin(4 * x) + x**3 + np.random.normal(0., 1.5, n)

        # compute smoothing spline with lambda=0 and interpolation spline
        spline_GCV = make_smoothing_spline(x, y, lam=0.)
        spline_interp = make_interp_spline(x, y, 3, bc_type='natural')

        # create grid for comparison
        grid = np.linspace(x[0], x[-1], 2 * n)

        # assertion to compare smoothing spline with interpolation spline
        assert_allclose(spline_GCV(grid),
                        spline_interp(grid),
                        atol=1e-15)

    @pytest.mark.fail_slow(2)
    # 定义一个测试方法，用于测试加权平滑样条
    def test_weighted_smoothing_spline(self):
        # 创建数据样本
        np.random.seed(1234)
        n = 100
        # 生成在区间[-2, 2]内的随机数，并按大小排序
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        # 根据 x 计算对应的 y 值，加入随机噪声
        y = x**2 * np.sin(4 * x) + x**3 + np.random.normal(0., 1.5, n)

        # 使用默认参数创建平滑样条
        spl = make_smoothing_spline(x, y)

        # 为了避免遍历所有索引，从中随机选择10个索引
        for ind in np.random.choice(range(100), size=10):
            # 创建权重向量 w，所有权重为 1，除了选定的索引处为 30
            w = np.ones(n)
            w[ind] = 30.
            # 使用权重 w 创建加权平滑样条
            spl_w = make_smoothing_spline(x, y, w)
            # 检查加权样条在特定点是否比未加权样条更接近原始点
            orig = abs(spl(x[ind]) - y[ind])
            weighted = abs(spl_w(x[ind]) - y[ind])

            # 如果加权样条更远离原始点，则抛出异常
            if orig < weighted:
                raise ValueError(f'Spline with weights should be closer to the'
                                 f' points than the original one: {orig:.4} < '
                                 f'{weighted:.4}')
################################
# NdBSpline tests
# NdBSpline 类的测试代码

def bspline2(xy, t, c, k):
    """A naive 2D tensort product spline evaluation."""
    # 一个简单的二维张量积样条函数评估
    x, y = xy
    tx, ty = t
    nx = len(tx) - k - 1
    assert (nx >= k+1)
    ny = len(ty) - k - 1
    assert (ny >= k+1)
    return sum(c[ix, iy] * B(x, k, ix, tx) * B(y, k, iy, ty)
               for ix in range(nx) for iy in range(ny))


def B(x, k, i, t):
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2
    # 递归计算 B 样条基函数的值


def bspline(x, t, c, k):
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))
    # 计算一维 B 样条插值函数的值


class NdBSpline0:
    def __init__(self, t, c, k=3):
        """Tensor product spline object.

        c[i1, i2, ..., id] * B(x1, i1) * B(x2, i2) * ... * B(xd, id)

        Parameters
        ----------
        c : ndarray, shape (n1, n2, ..., nd, ...)
            b-spline coefficients
        t : tuple of 1D ndarrays
            knot vectors in directions 1, 2, ... d
            ``len(t[i]) == n[i] + k + 1``
        k : int or length-d tuple of integers
            spline degrees.
        """
        ndim = len(t)
        assert ndim <= len(c.shape)

        try:
            len(k)
        except TypeError:
            # make k a tuple
            k = (k,)*ndim

        self.k = tuple(operator.index(ki) for ki in k)
        self.t = tuple(np.asarray(ti, dtype=float) for ti in t)
        self.c = c
        # 初始化多维张量积样条对象

    def __call__(self, x):
        ndim = len(self.t)
        # a single evaluation point: `x` is a 1D array_like, shape (ndim,)
        assert len(x) == ndim

        # get the indices in an ndim-dimensional vector
        i = ['none', ]*ndim
        for d in range(ndim):
            td, xd = self.t[d], x[d]
            k = self.k[d]

            # find the index for x[d]
            if xd == td[k]:
                i[d] = k
            else:
                i[d] = np.searchsorted(td, xd) - 1
            assert td[i[d]] <= xd <= td[i[d]+1]
            assert i[d] >= k and i[d] < len(td) - k
        i = tuple(i)

        # iterate over the dimensions, form linear combinations of
        # products B(x_1) * B(x_2) * ... B(x_N) of (k+1)**N b-splines
        # which are non-zero at `i = (i_1, i_2, ..., i_N)`.
        result = 0
        iters = [range(i[d] - self.k[d], i[d] + 1) for d in range(ndim)]
        for idx in itertools.product(*iters):
            term = self.c[idx] * np.prod([B(x[d], self.k[d], idx[d], self.t[d])
                                          for d in range(ndim)])
            result += term
        return result
        # 计算多维张量积样条插值函数的值


class TestNdBSpline:
    # NdBSpline0 类的单元测试
    def test_1D(self):
        # test ndim=1 agrees with BSpline
        # 使用种子为12345的随机数生成器创建rng对象
        rng = np.random.default_rng(12345)
        # 定义参数n和k
        n, k = 11, 3
        # 定义训练集大小n_tr
        n_tr = 7
        # 在区间[0, 1)内生成长度为n+k+1的随机数，并排序
        t = np.sort(rng.uniform(size=n + k + 1))
        # 生成形状为(n, n_tr)的随机数矩阵c
        c = rng.uniform(size=(n, n_tr))

        # 使用BSpline类初始化对象b
        b = BSpline(t, c, k)
        # 使用NdBSpline类初始化对象nb
        nb = NdBSpline((t,), c, k)

        # 生成长度为21的随机数数组xi
        xi = rng.uniform(size=21)
        # 断言验证NdBSpline类接受的xi数组形状为(npts, ndim)
        assert_allclose(nb(xi[:, None]),
                        b(xi), atol=1e-14)
        # 断言验证nb(xi[:, None])的形状为(xi数组的长度, c矩阵的列数)
        assert nb(xi[:, None]).shape == (xi.shape[0], c.shape[1])

    def make_2d_case(self):
        # make a 2D separable spline
        # 创建一维数组x和y
        x = np.arange(6)
        y = x**3
        # 使用make_interp_spline函数生成三阶样条插值spl
        spl = make_interp_spline(x, y, k=3)

        # 创建另一个一维数组y_1
        y_1 = x**3 + 2*x
        # 使用make_interp_spline函数生成三阶样条插值spl_1
        spl_1 = make_interp_spline(x, y_1, k=3)

        # 将spl和spl_1的参数t拼接成元组t2
        t2 = (spl.t, spl_1.t)
        # 将spl和spl_1的系数矩阵c做列乘积，得到新的系数矩阵c2
        c2 = spl.c[:, None] * spl_1.c[None, :]

        return t2, c2, 3

    def make_2d_mixed(self):
        # make a 2D separable spline w/ kx=3, ky=2
        # 创建一维数组x和y
        x = np.arange(6)
        y = x**3
        # 使用make_interp_spline函数生成三阶样条插值spl
        spl = make_interp_spline(x, y, k=3)

        # 重新创建一维数组x和y_1
        x = np.arange(5) + 1.5
        y_1 = x**2 + 2*x
        # 使用make_interp_spline函数生成二阶样条插值spl_1
        spl_1 = make_interp_spline(x, y_1, k=2)

        # 将spl和spl_1的参数t拼接成元组t2
        t2 = (spl.t, spl_1.t)
        # 将spl和spl_1的系数矩阵c做列乘积，得到新的系数矩阵c2
        c2 = spl.c[:, None] * spl_1.c[None, :]

        return t2, c2, spl.k, spl_1.k

    def test_2D_separable(self):
        # 创建二维数组xi
        xi = [(1.5, 2.5), (2.5, 1), (0.5, 1.5)]
        # 调用make_2d_case方法生成t2、c2和k
        t2, c2, k = self.make_2d_case()
        # 为每个二维点(xi[0], xi[1])计算目标值并存入列表target
        target = [x**3 * (y**3 + 2*y) for (x, y) in xi]

        # 断言验证bspline2函数对每个二维点(xi[0], xi[1])给出正确的目标值
        assert_allclose([bspline2(xy, t2, c2, k) for xy in xi],
                        target,
                        atol=1e-14)

        # 断言验证NdBSpline类对xi数组的正确评估，结果形状应为(len(xi), )
        bspl2 = NdBSpline(t2, c2, k=3)
        assert bspl2(xi).shape == (len(xi), )
        # 断言验证NdBSpline类对xi数组的评估结果与target列表吻合
        assert_allclose(bspl2(xi),
                        target, atol=1e-14)

        # 再次验证多维xi的情况
        # 使用种子为12345的随机数生成器创建rng对象
        rng = np.random.default_rng(12345)
        # 生成形状为(4, 3, 2)的随机数数组xi
        xi = rng.uniform(size=(4, 3, 2)) * 5
        # 对xi进行评估，结果应为形状(4, 3)
        result = bspl2(xi)
        assert result.shape == (4, 3)

        # 再次验证评估值是否正确
        x, y = xi.reshape((-1, 2)).T
        assert_allclose(result.ravel(),
                        x**3 * (y**3 + 2*y), atol=1e-14)
    def test_2D_separable_2(self):
        # test `c` with trailing dimensions, i.e. c.ndim > ndim
        # 定义二维情况下的测试函数，测试具有超出维度的 `c`，即 c.ndim > ndim
        ndim = 2
        xi = [(1.5, 2.5), (2.5, 1), (0.5, 1.5)]
        # 定义测试点集合 xi
        target = [x**3 * (y**3 + 2*y) for (x, y) in xi]
        # 计算目标值列表，用于后续断言比较

        t2, c2, k = self.make_2d_case()
        # 获取二维情况下的 t2, c2 和 k 参数
        c2_4 = np.dstack((c2, c2, c2, c2))   # c22.shape = (6, 6, 4)
        # 将 c2 沿第三维度堆叠成 c2_4，形状为 (6, 6, 4)

        xy = (1.5, 2.5)
        # 定义测试点 xy
        bspl2_4 = NdBSpline(t2, c2_4, k=3)
        # 创建 NdBSpline 对象 bspl2_4，使用 t2, c2_4 和 k=3
        result = bspl2_4(xy)
        # 计算 bspl2_4 在点 xy 处的结果
        val_single = NdBSpline(t2, c2, k)(xy)
        # 创建 NdBSpline 对象，并计算在点 xy 处的单一值
        assert result.shape == (4,)
        # 断言结果的形状为 (4,)
        assert_allclose(result,
                        [val_single, ]*4, atol=1e-14)
        # 断言结果接近于单一值 val_single，重复四次，允许误差为 1e-14

        # now try the array xi : the output.shape is (3, 4) where 3
        # is the number of points in xi and 4 is the trailing dimension of c
        # 现在尝试使用数组 xi：输出形状为 (3, 4)，其中 3 是 xi 中点的数量，4 是 c 的尾部维度
        assert bspl2_4(xi).shape == np.shape(xi)[:-1] + bspl2_4.c.shape[ndim:]
        # 断言 bspl2_4 在数组 xi 上的形状符合预期
        assert_allclose(bspl2_4(xi) - np.asarray(target)[:, None],
                        0, atol=5e-14)
        # 断言 bspl2_4 在数组 xi 上的计算结果与目标值 target 接近，允许误差为 5e-14

        # two trailing dimensions
        # 两个尾部维度
        c2_22 = c2_4.reshape((6, 6, 2, 2))
        # 将 c2_4 重塑为形状 (6, 6, 2, 2)
        bspl2_22 = NdBSpline(t2, c2_22, k=3)
        # 创建 NdBSpline 对象 bspl2_22，使用 t2, c2_22 和 k=3

        result = bspl2_22(xy)
        # 计算 bspl2_22 在点 xy 处的结果
        assert result.shape == (2, 2)
        # 断言结果的形状为 (2, 2)
        assert_allclose(result,
                        [[val_single, val_single],
                         [val_single, val_single]], atol=1e-14)
        # 断言结果接近于包含单一值 val_single 的 2x2 矩阵，允许误差为 1e-14

        # now try the array xi : the output shape is (3, 2, 2)
        # for 3 points in xi and c trailing dimensions being (2, 2)
        # 现在尝试使用数组 xi：输出形状为 (3, 2, 2)，xi 中有 3 个点，c 的尾部维度为 (2, 2)
        assert (bspl2_22(xi).shape ==
                np.shape(xi)[:-1] + bspl2_22.c.shape[ndim:])
        # 断言 bspl2_22 在数组 xi 上的形状符合预期
        assert_allclose(bspl2_22(xi) - np.asarray(target)[:, None, None],
                        0, atol=5e-14)
        # 断言 bspl2_22 在数组 xi 上的计算结果与目标值 target 接近，允许误差为 5e-14

    def test_2D_random(self):
        rng = np.random.default_rng(12345)
        # 使用种子 12345 创建随机数生成器 rng
        k = 3
        # 定义样条阶数 k
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        # 定义结点序列 tx
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        # 定义结点序列 ty
        c = rng.uniform(size=(tx.size-k-1, ty.size-k-1))
        # 生成随机系数矩阵 c，形状为 (tx.size-k-1, ty.size-k-1)

        spl = NdBSpline((tx, ty), c, k=k)
        # 创建 NdBSpline 对象 spl，使用 (tx, ty) 和 c，阶数为 k

        xi = (1., 1.)
        # 定义测试点 xi
        assert_allclose(spl(xi),
                        bspline2(xi, (tx, ty), c, k), atol=1e-14)
        # 断言 NdBSpline 对象 spl 在点 xi 处的结果与 bspline2 函数的结果接近，允许误差为 1e-14

        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]
        # 定义多个测试点 xi
        assert_allclose(spl(xi),
                        [bspline2(xy, (tx, ty), c, k) for xy in xi],
                        atol=1e-14)
        # 断言 NdBSpline 对象 spl 在数组 xi 上的结果与对应的 bspline2 函数结果列表接近，允许误差为 1e-14

    def test_2D_mixed(self):
        t2, c2, kx, ky = self.make_2d_mixed()
        # 获取混合二维情况下的 t2, c2, kx 和 ky
        xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
        # 定义测试点集合 xi
        target = [x**3 * (y**2 + 2*y) for (x, y) in xi]
        # 计算目标值列表 target
        bspl2 = NdBSpline(t2, c2, k=(kx, ky))
        # 创建 NdBSpline 对象 bspl2，使用 t2, c2 和 (kx, ky)
        assert bspl2(xi).shape == (len(xi), )
        # 断言 bspl2 在点集 xi 上的形状为 (len(xi), )
        assert_allclose(bspl2(xi),
                        target, atol=1e-14)
        # 断言 bspl2 在点集 xi 上的结果与目标值 target 接近，允许误差为 1e-14
    def test_2D_derivative(self):
        # 准备测试数据，生成混合的二维参数
        t2, c2, kx, ky = self.make_2d_mixed()
        # 定义测试点的坐标
        xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
        # 创建 NdBSpline 对象
        bspl2 = NdBSpline(t2, c2, k=(kx, ky))

        # 测试在指定导数阶数 nu=(1, 0) 下的结果
        der = bspl2(xi, nu=(1, 0))
        # 使用断言检查计算结果的精确性
        assert_allclose(der,
                        [3*x**2 * (y**2 + 2*y) for x, y in xi], atol=1e-14)

        # 测试在指定导数阶数 nu=(1, 1) 下的结果
        der = bspl2(xi, nu=(1, 1))
        # 使用断言检查计算结果的精确性
        assert_allclose(der,
                        [3*x**2 * (2*y + 2) for x, y in xi], atol=1e-14)

        # 测试在指定导数阶数 nu=(0, 0) 下的结果
        der = bspl2(xi, nu=(0, 0))
        # 使用断言检查计算结果的精确性
        assert_allclose(der,
                        [x**3 * (y**2 + 2*y) for x, y in xi], atol=1e-14)

        # 测试 nu 参数有负数的情况，应该引发 ValueError 异常
        with assert_raises(ValueError):
            # all(nu >= 0)
            der = bspl2(xi, nu=(-1, 0))

        # 测试 nu 参数长度超出维度的情况，应该引发 ValueError 异常
        with assert_raises(ValueError):
            # len(nu) == ndim
            der = bspl2(xi, nu=(-1, 0, 1))

    def test_2D_mixed_random(self):
        # 使用随机数生成器创建 NdBSpline 对象进行测试
        rng = np.random.default_rng(12345)
        kx, ky = 2, 3
        # 生成随机的节点向量 tx 和 ty
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        # 生成随机系数矩阵 c
        c = rng.uniform(size=(tx.size - kx - 1, ty.size - ky - 1))

        # 定义测试点的坐标 xi
        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]

        # 创建 NdBSpline 对象和 NdBSpline0 对象
        bspl2 = NdBSpline((tx, ty), c, k=(kx, ky))
        bspl2_0 = NdBSpline0((tx, ty), c, k=(kx, ky))

        # 使用断言检查两种对象在相同输入下的输出是否一致
        assert_allclose(bspl2(xi),
                        [bspl2_0(xp) for xp in xi], atol=1e-14)

    def test_tx_neq_ty(self):
        # 创建一个处理长度不相等的 tx 和 ty 的 NdBSpline 对象进行测试
        x = np.arange(6)
        y = np.arange(7) + 1.5

        # 使用 make_interp_spline 创建插值样条对象 spl_x 和 spl_y
        spl_x = make_interp_spline(x, x**3, k=3)
        spl_y = make_interp_spline(y, y**2 + 2*y, k=3)
        # 计算系数矩阵 cc
        cc = spl_x.c[:, None] * spl_y.c[None, :]
        # 创建 NdBSpline 对象 bspl
        bspl = NdBSpline((spl_x.t, spl_y.t), cc, (spl_x.k, spl_y.k))

        # 创建 RegularGridInterpolator 对象 rgi
        values = (x**3)[:, None] * (y**2 + 2*y)[None, :]
        rgi = RegularGridInterpolator((x, y), values)

        # 创建测试点 xi
        xi = [(a, b) for a, b in itertools.product(x, y)]
        # 计算 NdBSpline 对象 bspl 在测试点 xi 处的值
        bxi = bspl(xi)

        # 使用断言检查计算结果的正确性
        assert not np.isnan(bxi).any()
        assert_allclose(bxi, rgi(xi), atol=1e-14)
        assert_allclose(bxi.reshape(values.shape), values, atol=1e-14)

    def make_3d_case(self):
        # 创建一个三维分离样条
        x = np.arange(6)
        y = x**3
        spl = make_interp_spline(x, y, k=3)

        y_1 = x**3 + 2*x
        spl_1 = make_interp_spline(x, y_1, k=3)

        y_2 = x**3 + 3*x + 1
        spl_2 = make_interp_spline(x, y_2, k=3)

        # 构造 NdBSpline 所需的 t2, c2, 和维数 3
        t2 = (spl.t, spl_1.t, spl_2.t)
        c2 = (spl.c[:, None, None] *
              spl_1.c[None, :, None] *
              spl_2.c[None, None, :])

        return t2, c2, 3
    def test_3D_separable(self):
        # 创建一个新的随机数生成器实例
        rng = np.random.default_rng(12345)
        # 生成三个大小为 11 的均匀分布的随机数，乘以 5
        x, y, z = rng.uniform(size=(3, 11)) * 5
        # 计算目标值
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        # 调用 make_3d_case 方法，获取 t3, c3, k
        t3, c3, k = self.make_3d_case()
        # 创建 NdBSpline 对象，使用 t3, c3 和 k=3
        bspl3 = NdBSpline(t3, c3, k=3)

        # 将 x, y, z 组合成列表 xi
        xi = [_ for _ in zip(x, y, z)]
        # 计算 bspl3 在 xi 上的结果
        result = bspl3(xi)
        # 断言结果的形状为 (11,)
        assert result.shape == (11,)
        # 断言结果与目标值在指定精度下相等
        assert_allclose(result, target, atol=1e-14)

    def test_3D_derivative(self):
        # 调用 make_3d_case 方法，获取 t3, c3, k
        t3, c3, k = self.make_3d_case()
        # 创建 NdBSpline 对象，使用 t3, c3 和 k=3
        bspl3 = NdBSpline(t3, c3, k=3)
        # 创建一个新的随机数生成器实例
        rng = np.random.default_rng(12345)
        # 生成三个大小为 11 的均匀分布的随机数，乘以 5
        x, y, z = rng.uniform(size=(3, 11)) * 5
        # 将 x, y, z 组合成列表 xi
        xi = [_ for _ in zip(x, y, z)]

        # 断言 bspl3 在 xi 上的 (1, 0, 0) 导数与给定值在指定精度下相等
        assert_allclose(bspl3(xi, nu=(1, 0, 0)),
                        3*x**2 * (y**3 + 2*y) * (z**3 + 3*z + 1), atol=1e-14)

        # 断言 bspl3 在 xi 上的 (2, 0, 0) 导数与给定值在指定精度下相等
        assert_allclose(bspl3(xi, nu=(2, 0, 0)),
                        6*x * (y**3 + 2*y) * (z**3 + 3*z + 1), atol=1e-14)

        # 断言 bspl3 在 xi 上的 (2, 1, 0) 导数与给定值在指定精度下相等
        assert_allclose(bspl3(xi, nu=(2, 1, 0)),
                        6*x * (3*y**2 + 2) * (z**3 + 3*z + 1), atol=1e-14)

        # 断言 bspl3 在 xi 上的 (2, 1, 3) 导数与给定值在指定精度下相等
        assert_allclose(bspl3(xi, nu=(2, 1, 3)),
                        6*x * (3*y**2 + 2) * (6), atol=1e-14)

        # 断言 bspl3 在 xi 上的 (2, 1, 4) 导数结果全为零
        assert_allclose(bspl3(xi, nu=(2, 1, 4)),
                        np.zeros(len(xi)), atol=1e-14)

    def test_3D_random(self):
        # 创建一个新的随机数生成器实例
        rng = np.random.default_rng(12345)
        k = 3
        # 生成 tx, ty, tz 数组
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        tz = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        # 生成随机系数数组 c
        c = rng.uniform(size=(tx.size-k-1, ty.size-k-1, tz.size-k-1))

        # 创建 NdBSpline 对象 spl
        spl = NdBSpline((tx, ty, tz), c, k=k)
        # 创建 NdBSpline0 对象 spl_0
        spl_0 = NdBSpline0((tx, ty, tz), c, k=k)

        # 测试点 xi = (1., 1., 1)
        xi = (1., 1., 1)
        # 断言 spl 在 xi 上的值与 spl_0 在 xi 上的值在指定精度下相等
        assert_allclose(spl(xi), spl_0(xi), atol=1e-14)

        # 测试多个点 xi
        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1],
                   [0.9, 1.4, 1.9]]
        # 断言 spl 在 xi 上的值与 spl_0 在每个 xi 点上的值在指定精度下相等
        assert_allclose(spl(xi), [spl_0(xp) for xp in xi], atol=1e-14)

    def test_3D_random_complex(self):
        # 创建一个新的随机数生成器实例
        rng = np.random.default_rng(12345)
        k = 3
        # 生成 tx, ty, tz 数组
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        tz = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        # 生成复数随机系数数组 c
        c = (rng.uniform(size=(tx.size-k-1, ty.size-k-1, tz.size-k-1)) +
             rng.uniform(size=(tx.size-k-1, ty.size-k-1, tz.size-k-1))*1j)

        # 创建 NdBSpline 对象 spl
        spl = NdBSpline((tx, ty, tz), c, k=k)
        # 分别创建实部和虚部为系数的 NdBSpline 对象
        spl_re = NdBSpline((tx, ty, tz), c.real, k=k)
        spl_im = NdBSpline((tx, ty, tz), c.imag, k=k)

        # 测试多个点 xi
        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1],
                   [0.9, 1.4, 1.9]]
        # 断言 spl 在 xi 上的值与 spl_re + 1j*spl_im 在 xi 上的值在指定精度下相等
        assert_allclose(spl(xi),
                        spl_re(xi) + 1j*spl_im(xi), atol=1e-14)

    @pytest.mark.parametrize('cls_extrap', [None, True])
    @pytest.mark.parametrize('call_extrap', [None, True])
    def test_extrapolate_3D_separable(self, cls_extrap, call_extrap):
        # test that extrapolate=True does extrapolate
        # 创建一个三维测试用例 t3, c3, k，并获取阶数 k
        t3, c3, k = self.make_3d_case()
        # 使用 NdBSpline 类创建一个三维 B 样条对象 bspl3，设置 extrapolate 参数
        bspl3 = NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)

        # evaluate out of bounds
        # 定义超出边界的测试点 x, y, z
        x, y, z = [-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5]
        # 将 x, y, z 转换为 NumPy 数组
        x, y, z = map(np.asarray, (x, y, z))
        # 将 x, y, z 合并成一个列表 xi
        xi = [_ for _ in zip(x, y, z)]
        # 定义目标值 target，使用 x, y, z 的公式计算
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        # 调用 bspl3 对象进行插值，使用 call_extrap 控制是否外推
        result = bspl3(xi, extrapolate=call_extrap)
        # 使用 assert_allclose 函数断言结果与目标值在给定的容差范围内相似
        assert_allclose(result, target, atol=1e-14)

    @pytest.mark.parametrize('extrap', [(False, True), (True, None)])
    def test_extrapolate_3D_separable_2(self, extrap):
        # test that call(..., extrapolate=None) defers to self.extrapolate,
        # otherwise supersedes self.extrapolate
        # 创建一个三维测试用例 t3, c3, k，并获取阶数 k
        t3, c3, k = self.make_3d_case()
        # 从参数 extrap 中获取 cls_extrap 和 call_extrap
        cls_extrap, call_extrap = extrap
        # 使用 NdBSpline 类创建一个三维 B 样条对象 bspl3，设置 extrapolate 参数
        bspl3 = NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)

        # evaluate out of bounds
        # 定义超出边界的测试点 x, y, z
        x, y, z = [-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5]
        # 将 x, y, z 转换为 NumPy 数组
        x, y, z = map(np.asarray, (x, y, z))
        # 将 x, y, z 合并成一个列表 xi
        xi = [_ for _ in zip(x, y, z)]
        # 定义目标值 target，使用 x, y, z 的公式计算
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        # 调用 bspl3 对象进行插值，使用 call_extrap 控制是否外推
        result = bspl3(xi, extrapolate=call_extrap)
        # 使用 assert_allclose 函数断言结果与目标值在给定的容差范围内相似
        assert_allclose(result, target, atol=1e-14)

    def test_extrapolate_false_3D_separable(self):
        # test that extrapolate=False produces nans for out-of-bounds values
        # 创建一个三维测试用例 t3, c3, k，并获取阶数 k
        t3, c3, k = self.make_3d_case()
        # 使用 NdBSpline 类创建一个三维 B 样条对象 bspl3，设置 extrapolate 参数为 False
        bspl3 = NdBSpline(t3, c3, k=3)

        # evaluate out of bounds and inside
        # 定义超出边界和内部的测试点 x, y, z
        x, y, z = [-2, 1, 7], [-3, 0.5, 6.5], [-1, 1.5, 7.5]
        # 将 x, y, z 转换为 NumPy 数组
        x, y, z = map(np.asarray, (x, y, z))
        # 将 x, y, z 合并成一个列表 xi
        xi = [_ for _ in zip(x, y, z)]
        # 定义目标值 target，使用 x, y, z 的公式计算
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        # 调用 bspl3 对象进行插值，设置 extrapolate 参数为 False
        result = bspl3(xi, extrapolate=False)
        # 使用 assert 函数断言结果中超出边界的部分为 NaN
        assert np.isnan(result[0])
        assert np.isnan(result[-1])
        # 使用 assert_allclose 函数断言结果与目标值在给定的容差范围内相似，但不包括超出边界的部分
        assert_allclose(result[1:-1], target[1:-1], atol=1e-14)

    def test_x_nan_3D(self):
        # test that spline(nan) is nan
        # 创建一个三维测试用例 t3, c3, k，并获取阶数 k
        t3, c3, k = self.make_3d_case()
        # 使用 NdBSpline 类创建一个三维 B 样条对象 bspl3，设置 extrapolate 参数为 False
        bspl3 = NdBSpline(t3, c3, k=3)

        # evaluate out of bounds and inside
        # 定义超出边界和内部的测试点 x, y, z，其中包括 NaN 值
        x = np.asarray([-2, 3, np.nan, 1, 2, 7, np.nan])
        y = np.asarray([-3, 3.5, 1, np.nan, 3, 6.5, 6.5])
        z = np.asarray([-1, 3.5, 2, 3, np.nan, 7.5, 7.5])
        # 将 x, y, z 合并成一个列表 xi
        xi = [_ for _ in zip(x, y, z)]
        # 定义目标值 target，使用 x, y, z 的公式计算
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)
        # 创建一个掩码 mask，用于标识 x, y, z 中的 NaN 值
        mask = np.isnan(x) | np.isnan(y) | np.isnan(z)
        # 将目标值中对应掩码位置的值设为 NaN
        target[mask] = np.nan

        # 调用 bspl3 对象进行插值
        result = bspl3(xi)
        # 使用 assert 函数断言结果中对应掩码位置的值为 NaN
        assert np.isnan(result[mask]).all()
        # 使用 assert_allclose 函数断言结果与目标值在给定的容差范围内相似
        assert_allclose(result, target, atol=1e-14)
    def test_non_c_contiguous(self):
        # 检查非C连续的输入是否可以正常工作
        rng = np.random.default_rng(12345)
        kx, ky = 3, 3
        tx = np.sort(rng.uniform(low=0, high=4, size=16))
        tx = np.r_[(tx[0],)*kx, tx, (tx[-1],)*kx]
        ty = np.sort(rng.uniform(low=0, high=4, size=16))
        ty = np.r_[(ty[0],)*ky, ty, (ty[-1],)*ky]

        assert not tx[::2].flags.c_contiguous  # 断言：检查 tx 是否为C连续
        assert not ty[::2].flags.c_contiguous  # 断言：检查 ty 是否为C连续

        c = rng.uniform(size=(tx.size//2 - kx - 1, ty.size//2 - ky - 1))
        c = c.T
        assert not c.flags.c_contiguous  # 断言：检查 c 是否为C连续

        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]

        bspl2 = NdBSpline((tx[::2], ty[::2]), c, k=(kx, ky))
        bspl2_0 = NdBSpline0((tx[::2], ty[::2]), c, k=(kx, ky))

        assert_allclose(bspl2(xi),
                        [bspl2_0(xp) for xp in xi], atol=1e-14)

    def test_readonly(self):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)

        for i in range(3):
            t3[i].flags.writeable = False
        c3.flags.writeable = False

        bspl3_ = NdBSpline(t3, c3, k=3)

        assert bspl3((1, 2, 3)) == bspl3_((1, 2, 3))

    def test_design_matrix(self):
        t3, c3, k = self.make_3d_case()

        xi = np.asarray([[1, 2, 3], [4, 5, 6]])
        dm = NdBSpline(t3, c3, k).design_matrix(xi, t3, k)
        dm1 = NdBSpline.design_matrix(xi, t3, [k, k, k])
        assert dm.shape[0] == xi.shape[0]
        assert_allclose(dm.todense(), dm1.todense(), atol=1e-16)

        with assert_raises(ValueError):
            NdBSpline.design_matrix([1, 2, 3], t3, [k]*3)

        with assert_raises(ValueError, match="Data and knots*"):
            NdBSpline.design_matrix([[1, 2]], t3, [k]*3)



# 检查只读性是否正常，比较两个不同只读输入情况下的 NdBSpline 对象输出
    def test_readonly(self):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)

        # 将 t3 的前三个维度设为不可写
        for i in range(3):
            t3[i].flags.writeable = False
        # 将 c3 设为不可写
        c3.flags.writeable = False

        # 使用只读的 t3 和 c3 创建另一个 NdBSpline 对象
        bspl3_ = NdBSpline(t3, c3, k=3)

        # 断言：比较两个对象在相同输入下的输出是否相同
        assert bspl3((1, 2, 3)) == bspl3_((1, 2, 3))

# 测试设计矩阵的生成是否正确，包括不同输入情况下的异常处理
    def test_design_matrix(self):
        t3, c3, k = self.make_3d_case()

        xi = np.asarray([[1, 2, 3], [4, 5, 6]])
        # 生成设计矩阵
        dm = NdBSpline(t3, c3, k).design_matrix(xi, t3, k)
        # 使用静态方法生成设计矩阵
        dm1 = NdBSpline.design_matrix(xi, t3, [k, k, k])
        # 断言：检查生成的设计矩阵的形状是否正确
        assert dm.shape[0] == xi.shape[0]
        # 断言：检查两个设计矩阵是否在数值上非常接近
        assert_allclose(dm.todense(), dm1.todense(), atol=1e-16)

        # 断言：检查当输入不合法时是否能抛出 ValueError 异常
        with assert_raises(ValueError):
            NdBSpline.design_matrix([1, 2, 3], t3, [k]*3)

        # 断言：检查当输入数据和结点不匹配时是否能抛出特定信息的 ValueError 异常
        with assert_raises(ValueError, match="Data and knots*"):
            NdBSpline.design_matrix([[1, 2]], t3, [k]*3)
class TestMakeND:
    # 定义测试用例：测试二维可分离的简单情况
    def test_2D_separable_simple(self):
        # 创建一维数组 x 包含 0 到 5
        x = np.arange(6)
        # 创建一维数组 y 包含 0.5 到 5.5
        y = np.arange(6) + 0.5
        # 根据 x 和 y 的外积计算二维数组 values
        values = x[:, None]**3 * (y**3 + 2*y)[None, :]
        # 生成包含 x 和 y 所有可能组合的列表 xi
        xi = [(a, b) for a, b in itertools.product(x, y)]

        # 使用 make_ndbspl 函数生成二维 B 样条对象 bspl
        bspl = make_ndbspl((x, y), values, k=1)
        # 断言 bspl 对 xi 的计算结果与 values 的展平结果非常接近
        assert_allclose(bspl(xi), values.ravel(), atol=1e-15)

        # 测试 bspl.c 与两个一维插值样条的系数外积
        spl_x = make_interp_spline(x, x**3, k=1)
        spl_y = make_interp_spline(y, y**3 + 2*y, k=1)
        cc = spl_x.c[:, None] * spl_y.c[None, :]
        # 断言 bspl.c 与 cc 的元素非常接近
        assert_allclose(cc, bspl.c, atol=1e-11, rtol=0)

        # 测试与 RegularGridInterpolator (RGI) 的结果比较
        from scipy.interpolate import RegularGridInterpolator as RGI
        rgi = RGI((x, y), values, method='linear')
        # 断言 bspl 对 xi 的计算结果与 rgi 对 xi 的计算结果非常接近
        assert_allclose(rgi(xi), bspl(xi), atol=1e-14)

    # 定义测试用例：测试带有尾部维度的情况
    def test_2D_separable_trailing_dims(self):
        # 测试带有尾部维度（c.ndim > ndim）的情况，即 c 的维度大于 ndim
        x = np.arange(6)
        y = np.arange(6)
        xi = [(a, b) for a, b in itertools.product(x, y)]

        # 创建 values4.shape = (6, 6, 4)
        values = x[:, None]**3 * (y**3 + 2*y)[None, :]
        values4 = np.dstack((values, values, values, values))
        # 使用 solver=ssl.spsolve 创建 k=3 的二维 B 样条对象 bspl
        bspl = make_ndbspl((x, y), values4, k=3, solver=ssl.spsolve)

        result = bspl(xi)
        target = np.dstack((values, values, values, values))
        # 断言 result 的形状为 (36, 4)，且其展平结果与 target 非常接近
        assert result.shape == (36, 4)
        assert_allclose(result.reshape(6, 6, 4),
                        target, atol=1e-14)

        # 现在测试两个尾部维度的情况
        values22 = values4.reshape((6, 6, 2, 2))
        # 使用 solver=ssl.spsolve 创建 k=3 的二维 B 样条对象 bspl
        bspl = make_ndbspl((x, y), values22, k=3, solver=ssl.spsolve)

        result = bspl(xi)
        # 断言 result 的形状为 (36, 2, 2)，且其展平结果与 target 的形状相同且非常接近
        assert result.shape == (36, 2, 2)
        assert_allclose(result.reshape(6, 6, 2, 2),
                        target.reshape((6, 6, 2, 2)), atol=1e-14)

    # 使用 pytest 参数化装饰器，定义多个 k 值的测试用例
    @pytest.mark.parametrize('k', [(3, 3), (1, 1), (3, 1), (1, 3), (3, 5)])
    def test_2D_mixed(self, k):
        # 创建一个二维可分离样条，其中 len(tx) != len(ty)
        x = np.arange(6)
        y = np.arange(7) + 1.5
        xi = [(a, b) for a, b in itertools.product(x, y)]

        values = (x**3)[:, None] * (y**2 + 2*y)[None, :]
        # 使用 solver=ssl.spsolve 创建 k 值为 k 的二维 B 样条对象 bspl
        bspl = make_ndbspl((x, y), values, k=k, solver=ssl.spsolve)
        # 断言 bspl 对 xi 的计算结果与 values 的展平结果非常接近
        assert_allclose(bspl(xi), values.ravel(), atol=1e-15)

    def _get_sample_2d_data(self):
        # 从 test_rgi.py::TestIntepN 获取一个二维样本数据
        x = np.array([.5, 2., 3., 4., 5.5, 6.])
        y = np.array([.5, 2., 3., 4., 5.5, 6.])
        z = np.array(
            [
                [1, 2, 1, 2, 1, 1],
                [1, 2, 1, 2, 1, 1],
                [1, 2, 3, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 1, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
            ]
        )
        return x, y, z
    `
        def test_2D_vs_RGI_linear(self):
            # 获取二维样本数据 x, y, z
            x, y, z = self._get_sample_2d_data()
            # 使用线性插值创建二维 B-spline 插值器
            bspl = make_ndbspl((x, y), z, k=1)
            # 使用线性插值创建 RegularGridInterpolator 对象
            rgi = RegularGridInterpolator((x, y), z, method='linear')
    
            # 定义插值点 xi
            xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
    
            # 断言 B-spline 和 RegularGridInterpolator 的插值结果近似相等
            assert_allclose(bspl(xi), rgi(xi), atol=1e-14)
    
        def test_2D_vs_RGI_cubic(self):
            # 获取二维样本数据 x, y, z
            x, y, z = self._get_sample_2d_data()
            # 使用三次样条插值创建二维 B-spline 插值器，使用稀疏矩阵求解器
            bspl = make_ndbspl((x, y), z, k=3, solver=ssl.spsolve)
            # 使用三次样条插值创建 RegularGridInterpolator 对象，使用传统的立方插值方法
            rgi = RegularGridInterpolator((x, y), z, method='cubic_legacy')
    
            # 定义插值点 xi
            xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
    
            # 断言 B-spline 和 RegularGridInterpolator 的插值结果近似相等
            assert_allclose(bspl(xi), rgi(xi), atol=1e-14)
    
        @pytest.mark.parametrize('solver', [ssl.gmres, ssl.gcrotmk])
        def test_2D_vs_RGI_cubic_iterative(self, solver):
            # 和 `test_2D_vs_RGI_cubic` 相同，只是使用了迭代求解器
            # 需要注意添加 rtol solver_arg 以达到目标精度 1e-14
            x, y, z = self._get_sample_2d_data()
            # 使用三次样条插值创建二维 B-spline 插值器，指定迭代求解器和相对误差容限
            bspl = make_ndbspl((x, y), z, k=3, solver=solver, rtol=1e-6)
            # 使用三次样条插值创建 RegularGridInterpolator 对象，使用传统的立方插值方法
            rgi = RegularGridInterpolator((x, y), z, method='cubic_legacy')
    
            # 定义插值点 xi
            xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
    
            # 断言 B-spline 和 RegularGridInterpolator 的插值结果近似相等
            assert_allclose(bspl(xi), rgi(xi), atol=1e-14)
    
        def test_2D_vs_RGI_quintic(self):
            # 获取二维样本数据 x, y, z
            x, y, z = self._get_sample_2d_data()
            # 使用五次样条插值创建二维 B-spline 插值器，使用稀疏矩阵求解器
            bspl = make_ndbspl((x, y), z, k=5, solver=ssl.spsolve)
            # 使用五次样条插值创建 RegularGridInterpolator 对象，使用传统的五次插值方法
            rgi = RegularGridInterpolator((x, y), z, method='quintic_legacy')
    
            # 定义插值点 xi
            xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
    
            # 断言 B-spline 和 RegularGridInterpolator 的插值结果近似相等
            assert_allclose(bspl(xi), rgi(xi), atol=1e-14)
    
        @pytest.mark.parametrize(
            'k, meth', [(1, 'linear'), (3, 'cubic_legacy'), (5, 'quintic_legacy')]
        )
        def test_3D_random_vs_RGI(self, k, meth):
            # 使用固定种子生成随机数发生器
            rndm = np.random.default_rng(123456)
            # 生成累积和数据 x, y, z
            x = np.cumsum(rndm.uniform(size=6))
            y = np.cumsum(rndm.uniform(size=7))
            z = np.cumsum(rndm.uniform(size=8))
            # 生成随机值数组 values
            values = rndm.uniform(size=(6, 7, 8))
    
            # 使用指定阶数和求解器创建多维 B-spline 插值器
            bspl = make_ndbspl((x, y, z), values, k=k, solver=ssl.spsolve)
            # 使用指定插值方法创建 RegularGridInterpolator 对象
            rgi = RegularGridInterpolator((x, y, z), values, method=meth)
    
            # 定义插值点 xi
            xi = np.random.uniform(low=0.7, high=2.1, size=(11, 3))
            # 断言 B-spline 和 RegularGridInterpolator 的插值结果近似相等
            assert_allclose(bspl(xi), rgi(xi), atol=1e-14)
    
        def test_solver_err_not_converged(self):
            # 获取二维样本数据 x, y, z
            x, y, z = self._get_sample_2d_data()
            # 设置求解器参数，限制最大迭代次数为 1
            solver_args = {'maxiter': 1}
            # 使用默认求解器尝试创建三次样条插值器，预期抛出 ValueError 异常
            with assert_raises(ValueError, match='solver'):
                make_ndbspl((x, y), z, k=3, **solver_args)
    
            # 使用堆叠数组尝试创建三次样条插值器，预期抛出 ValueError 异常
            with assert_raises(ValueError, match='solver'):
                make_ndbspl((x, y), np.dstack((z, z)), k=3, **solver_args)
class TestFpchec:
    # https://github.com/scipy/scipy/blob/main/scipy/interpolate/fitpack/fpchec.f

    # 测试条件：检查输入是否为1维序列
    def test_1D_x_t(self):
        k = 1
        t = np.arange(12).reshape(2, 6)  # 创建一个2x6的数组
        x = np.arange(12)  # 创建一个包含12个元素的数组

        # 断言引发值错误异常，异常消息包含"1D sequence"
        with pytest.raises(ValueError, match="1D sequence"):
            _b.fpcheck(x, t, k)

        # 断言引发值错误异常，异常消息包含"1D sequence"
        with pytest.raises(ValueError, match="1D sequence"):
            _b.fpcheck(t, x, k)

    # 测试条件1：检查k+1 <= n-k-1 <= m是否成立
    def test_condition_1(self):
        k = 3
        n  = 2*(k + 1) - 1    # 计算n的值，此处不满足条件
        m = n + 11            # 计算m的值，满足条件
        t = np.arange(n)
        x = np.arange(m)

        # 断言dfitpack.fpchec的返回值为10
        assert dfitpack.fpchec(x, t, k) == 10

        # 断言引发值错误异常，异常消息包含"Need k+1*"
        with pytest.raises(ValueError, match="Need k+1*"):
            _b.fpcheck(x, t, k)

        n = 2*(k+1) + 1   # 计算n的值，满足条件
        m = n - k - 2     # 计算m的值，此处不满足条件
        t = np.arange(n)
        x = np.arange(m)

        # 断言dfitpack.fpchec的返回值为10
        assert dfitpack.fpchec(x, t, k) == 10

        # 断言引发值错误异常，异常消息包含"Need k+1*"
        with pytest.raises(ValueError, match="Need k+1*"):
            _b.fpcheck(x, t, k)

    # 测试条件2：检查t(1) <= t(2) <= ... <= t(k+1)和t(n-k) <= t(n-k+1) <= ... <= t(n)是否成立
    def test_condition_2(self):
        k = 3
        t = [0]*(k+1) + [2] + [5]*(k+1)   # 创建一个符合条件的列表
        x = [1, 2, 3, 4, 4.5]

        # 断言dfitpack.fpchec的返回值为0
        assert dfitpack.fpchec(x, t, k) == 0

        # 断言_b.fpcheck不会引发异常
        assert _b.fpcheck(x, t, k) is None

        tt = t.copy()
        tt[-1] = tt[0]   # 修改使其不满足条件

        # 断言dfitpack.fpchec的返回值为20
        assert dfitpack.fpchec(x, tt, k) == 20

        # 断言引发值错误异常，异常消息包含"Last k knots*"
        with pytest.raises(ValueError, match="Last k knots*"):
            _b.fpcheck(x, tt, k)

        tt = t.copy()
        tt[0] = tt[-1]   # 修改使其不满足条件

        # 断言dfitpack.fpchec的返回值为20
        assert dfitpack.fpchec(x, tt, k) == 20

        # 断言引发值错误异常，异常消息包含"First k knots*"
        with pytest.raises(ValueError, match="First k knots*"):
            _b.fpcheck(x, tt, k)

    # 测试条件3：检查t(k+1) < t(k+2) < ... < t(n-k)是否成立
    def test_condition_3(self):
        k = 3
        t = [0]*(k+1) + [2, 3] + [5]*(k+1)   # 创建一个符合条件的列表
        x = [1, 2, 3, 3.5, 4, 4.5]

        # 断言dfitpack.fpchec的返回值为0
        assert dfitpack.fpchec(x, t, k) == 0

        # 断言_b.fpcheck不会引发异常
        assert _b.fpcheck(x, t, k) is None

        t = [0]*(k+1) + [2, 2] + [5]*(k+1)   # 创建一个不符合条件的列表

        # 断言dfitpack.fpchec的返回值为30
        assert dfitpack.fpchec(x, t, k) == 30

        # 断言引发值错误异常，异常消息包含"Internal knots*"
        with pytest.raises(ValueError, match="Internal knots*"):
            _b.fpcheck(x, t, k)
    def test_condition_4(self):
        # c      4) t(k+1) <= x(i) <= t(n-k)
        # NB: FITPACK's fpchec only checks x[0] & x[-1], so we follow.
        k = 3
        t = [0]*(k+1) + [5]*(k+1)
        x = [1, 2, 3, 3.5, 4, 4.5]      # this is OK
        assert dfitpack.fpchec(x, t, k) == 0
        assert _b.fpcheck(x, t, k) is None

        xx = x.copy()
        xx[0] = t[0]    # still OK
        assert dfitpack.fpchec(xx, t, k) == 0
        assert _b.fpcheck(x, t, k) is None

        xx = x.copy()
        xx[0] = t[0] - 1    # not OK
        assert dfitpack.fpchec(xx, t, k) == 40
        with pytest.raises(ValueError, match="Out of bounds*"):
            _b.fpcheck(xx, t, k)

        xx = x.copy()
        xx[-1] = t[-1] + 1    # not OK
        assert dfitpack.fpchec(xx, t, k) == 40
        with pytest.raises(ValueError, match="Out of bounds*"):
            _b.fpcheck(xx, t, k)

    # ### Test the S-W condition (no 5)
    # c      5) the conditions specified by schoenberg and whitney must hold
    # c         for at least one subset of data points, i.e. there must be a
    # c         subset of data points y(j) such that
    # c             t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
    def test_condition_5_x1xm(self):
        # x(1).ge.t(k2) .or. x(m).le.t(nk1)
        k = 1
        t = [0, 0, 1, 2, 2]
        x = [1.1, 1.1, 1.1]
        assert dfitpack.fpchec(x, t, k) == 50
        with pytest.raises(ValueError, match="Schoenberg-Whitney*"):
            _b.fpcheck(x, t, k)

        x = [0.5, 0.5, 0.5]
        assert dfitpack.fpchec(x, t, k) == 50
        with pytest.raises(ValueError, match="Schoenberg-Whitney*"):
            _b.fpcheck(x, t, k)

    def test_condition_5_k1(self):
        # special case nk3 (== n - k - 2) < 2
        k = 1
        t = [0, 0, 1, 1]
        x = [0.5, 0.6]
        assert dfitpack.fpchec(x, t, k) == 0
        assert _b.fpcheck(x, t, k) is None

    def test_condition_5_1(self):
        # basically, there can't be an interval of t[j]..t[j+k+1] with no x
        k = 3
        t = [0]*(k+1) + [2] + [5]*(k+1)
        x = [3]*5
        assert dfitpack.fpchec(x, t, k) == 50
        with pytest.raises(ValueError, match="Schoenberg-Whitney*"):
            _b.fpcheck(x, t, k)

        t = [0]*(k+1) + [2] + [5]*(k+1)
        x = [1]*5
        assert dfitpack.fpchec(x, t, k) == 50
        with pytest.raises(ValueError, match="Schoenberg-Whitney*"):
            _b.fpcheck(x, t, k)

    def test_condition_5_2(self):
        # same as _5_1, only the empty interval is in the middle
        k = 3
        t = [0]*(k+1) + [2, 3] + [5]*(k+1)
        x = [1.1]*5 + [4]

        assert dfitpack.fpchec(x, t, k) == 50
        with pytest.raises(ValueError, match="Schoenberg-Whitney*"):
            _b.fpcheck(x, t, k)

        # and this one is OK
        x = [1.1]*4 + [4, 4]
        assert dfitpack.fpchec(x, t, k) == 0
        assert _b.fpcheck(x, t, k) is None
    def test_condition_5_3(self):
        # 定义测试函数 test_condition_5_3，用于测试特定条件下的情况
        # 与 _5_2 类似，但覆盖了不同的失败分支

        k = 1
        # 设置整数变量 k 为 1

        t = [0, 0, 2, 3, 4, 5, 6, 7, 7]
        # 创建包含整数的列表 t

        x = [1, 1, 1, 5.2, 5.2, 5.2, 6.5]
        # 创建包含浮点数的列表 x

        # 使用 dfitpack.fpchec 函数检查 x, t, k 的值是否符合条件
        assert dfitpack.fpchec(x, t, k) == 50

        # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配异常信息 "Schoenberg-Whitney*"
        with pytest.raises(ValueError, match="Schoenberg-Whitney*"):
            _b.fpcheck(x, t, k)
```