# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_interpolate.py`

```
from numpy.testing import (assert_, assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from pytest import raises as assert_raises  # 导入 pytest 库中的 raises 函数，并将其重命名为 assert_raises
import pytest  # 导入 pytest 库

from numpy import mgrid, pi, sin, poly1d  # 导入 numpy 库中的 mgrid, pi, sin, poly1d 函数
import numpy as np  # 导入 numpy 库，并使用 np 别名

from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
        splrep, splev, splantider, splint, sproot, Akima1DInterpolator,
        NdPPoly, BSpline, PchipInterpolator)  # 导入 scipy.interpolate 中的多个插值函数和类

from scipy.special import poch, gamma  # 导入 scipy.special 库中的 poch, gamma 函数

from scipy.interpolate import _ppoly  # 导入 scipy.interpolate 库中的 _ppoly 模块

from scipy._lib._gcutils import assert_deallocated, IS_PYPY  # 导入 scipy._lib._gcutils 中的 assert_deallocated, IS_PYPY 函数和常量

from scipy.integrate import nquad  # 导入 scipy.integrate 中的 nquad 函数

from scipy.special import binom  # 导入 scipy.special 库中的 binom 函数


class TestInterp2D:
    def test_interp2d(self):
        y, x = mgrid[0:2:20j, 0:pi:21j]  # 生成二维网格 y 和 x
        z = sin(x+0.5*y)  # 计算 z 值，使用 sin 函数
        with assert_raises(NotImplementedError):  # 断言捕获 NotImplementedError 异常
            interp2d(x, y, z)  # 调用 interp2d 函数进行二维插值


class TestInterp1D:

    def setup_method(self):
        self.x5 = np.arange(5.)  # 创建长度为 5 的浮点数数组 self.x5
        self.x10 = np.arange(10.)  # 创建长度为 10 的浮点数数组 self.x10
        self.y10 = np.arange(10.)  # 创建长度为 10 的浮点数数组 self.y10
        self.x25 = self.x10.reshape((2,5))  # 将 self.x10 重塑为 2x5 的数组 self.x25
        self.x2 = np.arange(2.)  # 创建长度为 2 的浮点数数组 self.x2
        self.y2 = np.arange(2.)  # 创建长度为 2 的浮点数数组 self.y2
        self.x1 = np.array([0.])  # 创建包含单个元素 0.0 的数组 self.x1
        self.y1 = np.array([0.])  # 创建包含单个元素 0.0 的数组 self.y1

        self.y210 = np.arange(20.).reshape((2, 10))  # 创建形状为 2x10 的数组 self.y210
        self.y102 = np.arange(20.).reshape((10, 2))  # 创建形状为 10x2 的数组 self.y102
        self.y225 = np.arange(20.).reshape((2, 2, 5))  # 创建形状为 2x2x5 的数组 self.y225
        self.y25 = np.arange(10.).reshape((2, 5))  # 创建形状为 2x5 的数组 self.y25
        self.y235 = np.arange(30.).reshape((2, 3, 5))  # 创建形状为 2x3x5 的数组 self.y235
        self.y325 = np.arange(30.).reshape((3, 2, 5))  # 创建形状为 3x2x5 的数组 self.y325

        # 更新边界的测试矩阵 1
        # array([[ 30,   1,   2,   3,   4,   5,   6,   7,   8, -30],
        #        [ 30,  11,  12,  13,  14,  15,  16,  17,  18, -30]])
        self.y210_edge_updated = np.arange(20.).reshape((2, 10))  # 创建形状为 2x10 的数组 self.y210_edge_updated
        self.y210_edge_updated[:, 0] = 30  # 更新第一列为 30
        self.y210_edge_updated[:, -1] = -30  # 更新最后一列为 -30

        # 更新边界的测试矩阵 2
        # array([[ 30,  30],
        #       [  2,   3],
        #       [  4,   5],
        #       [  6,   7],
        #       [  8,   9],
        #       [ 10,  11],
        #       [ 12,  13],
        #       [ 14,  15],
        #       [ 16,  17],
        #       [-30, -30]])
        self.y102_edge_updated = np.arange(20.).reshape((10, 2))  # 创建形状为 10x2 的数组 self.y102_edge_updated
        self.y102_edge_updated[0, :] = 30  # 更新第一行为 30
        self.y102_edge_updated[-1, :] = -30  # 更新最后一行为 -30

        self.fill_value = -100.0  # 初始化填充值为 -100.0
    def test_init(self):
        # 测试初始化函数，检查构造函数是否正确初始化属性
        assert_(interp1d(self.x10, self.y10).copy)
        assert_(not interp1d(self.x10, self.y10, copy=False).copy)
        assert_(interp1d(self.x10, self.y10).bounds_error)
        assert_(not interp1d(self.x10, self.y10, bounds_error=False).bounds_error)
        assert_(np.isnan(interp1d(self.x10, self.y10).fill_value))
        assert_equal(interp1d(self.x10, self.y10, fill_value=3.0).fill_value,
                     3.0)
        assert_equal(interp1d(self.x10, self.y10, fill_value=(1.0, 2.0)).fill_value,
                     (1.0, 2.0))
        assert_equal(interp1d(self.x10, self.y10).axis, 0)
        assert_equal(interp1d(self.x10, self.y210).axis, 1)
        assert_equal(interp1d(self.x10, self.y102, axis=0).axis, 0)
        assert_array_equal(interp1d(self.x10, self.y10).x, self.x10)
        assert_array_equal(interp1d(self.x10, self.y10).y, self.y10)
        assert_array_equal(interp1d(self.x10, self.y210).y, self.y210)

    def test_assume_sorted(self):
        # 检查未排序的数组
        interp10 = interp1d(self.x10, self.y10)
        interp10_unsorted = interp1d(self.x10[::-1], self.y10[::-1])

        assert_array_almost_equal(interp10_unsorted(self.x10), self.y10)
        assert_array_almost_equal(interp10_unsorted(1.2), np.array([1.2]))
        assert_array_almost_equal(interp10_unsorted([2.4, 5.6, 6.0]),
                                  interp10([2.4, 5.6, 6.0]))

        # 检查 assume_sorted 关键字（默认为 False）
        interp10_assume_kw = interp1d(self.x10[::-1], self.y10[::-1],
                                      assume_sorted=False)
        assert_array_almost_equal(interp10_assume_kw(self.x10), self.y10)

        interp10_assume_kw2 = interp1d(self.x10[::-1], self.y10[::-1],
                                       assume_sorted=True)
        # 如果 assume_sorted=True，则对于未排序的输入应引发错误
        assert_raises(ValueError, interp10_assume_kw2, self.x10)

        # 检查如果 y 是 2-D 数组，结果是否一致
        interp10_y_2d = interp1d(self.x10, self.y210)
        interp10_y_2d_unsorted = interp1d(self.x10[::-1], self.y210[:, ::-1])
        assert_array_almost_equal(interp10_y_2d(self.x10),
                                  interp10_y_2d_unsorted(self.x10))

    def test_linear(self):
        for kind in ['linear', 'slinear']:
            self._check_linear(kind)
    # 检查线性插值的实际实现。
    interp10 = interp1d(self.x10, self.y10, kind=kind)
    # 断言线性插值在已知点上的结果与期望值相近
    assert_array_almost_equal(interp10(self.x10), self.y10)
    # 断言线性插值在新点上的结果与期望值相近
    assert_array_almost_equal(interp10(1.2), np.array([1.2]))
    assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                              np.array([2.4, 5.6, 6.0]))

    # 测试 fill_value="extrapolate"
    # 使用 extrapolate 填充方式进行插值
    extrapolator = interp1d(self.x10, self.y10, kind=kind,
                            fill_value='extrapolate')
    # 断言 extrapolate 插值在指定点上的结果与期望值相近
    assert_allclose(extrapolator([-1., 0, 9, 11]),
                    [-1, 0, 9, 11], rtol=1e-14)

    # 设置选项字典，用于测试边界错误处理
    opts = dict(kind=kind,
                fill_value='extrapolate',
                bounds_error=True)
    # 断言在边界错误设置为 True 时会引发 ValueError 异常
    assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)


    # 回归测试 gh-5898，确保对所有浮点数数据类型使用 numpy.interp 进行 1D 线性插值，
    # 并确保对于例如 np.float128 这样的数据类型也能正确处理。
    for dtyp in [np.float16,
                 np.float32,
                 np.float64,
                 np.longdouble]:
        x = np.arange(8, dtype=dtyp)
        y = x
        yp = interp1d(x, y, kind='linear')(x)
        # 断言插值结果的数据类型与原始数据类型相同
        assert_equal(yp.dtype, dtyp)
        # 断言插值结果与原始数据相近
        assert_allclose(yp, y, atol=1e-15)

    # 回归测试 gh-14531，确保对整数数据类型也能正确进行 1D 线性插值。
    x = [0, 1, 2]
    y = [np.nan, 0, 1]
    yp = interp1d(x, y)(x)
    # 断言插值结果与原始数据相近
    assert_allclose(yp, y, atol=1e-15)


    # 回归测试 gh-7273: 1D slinear 插值在 float32 输入时会失败。
    dt_r = [np.float16, np.float32, np.float64]
    dt_rc = dt_r + [np.complex64, np.complex128]
    spline_kinds = ['slinear', 'zero', 'quadratic', 'cubic']
    for dtx in dt_r:
        x = np.arange(0, 10, dtype=dtx)
        for dty in dt_rc:
            y = np.exp(-x/3.0).astype(dty)
            for dtn in dt_r:
                xnew = x.astype(dtn)
                for kind in spline_kinds:
                    f = interp1d(x, y, kind=kind, bounds_error=False)
                    # 断言插值结果与期望值相近，设置较大的容差
                    assert_allclose(f(xnew), y, atol=1e-7,
                                    err_msg=f"{dtx}, {dty} {dtn}")


    # 检查样条插值的实际实现。
    interp10 = interp1d(self.x10, self.y10, kind='cubic')
    # 断言样条插值在已知点上的结果与期望值相近
    assert_array_almost_equal(interp10(self.x10), self.y10)
    assert_array_almost_equal(interp10(1.2), np.array([1.2]))
    assert_array_almost_equal(interp10(1.5), np.array([1.5]))
    assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                              np.array([2.4, 5.6, 6.0]),)
    def test_nearest(self):
        # Check the actual implementation of nearest-neighbour interpolation.
        # Nearest asserts that half-integer case (1.5) rounds down to 1
        interp10 = interp1d(self.x10, self.y10, kind='nearest')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.))
        assert_array_almost_equal(interp10(1.5), np.array(1.))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2., 6., 6.]),)

        # test fill_value="extrapolate"
        extrapolator = interp1d(self.x10, self.y10, kind='nearest',
                                fill_value='extrapolate')
        assert_allclose(extrapolator([-1., 0, 9, 11]),
                        [0, 0, 9, 9], rtol=1e-14)

        # Define options for interpolation with nearest method
        opts = dict(kind='nearest',
                    fill_value='extrapolate',
                    bounds_error=True)
        # Assert that ValueError is raised when bounds_error=True
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_nearest_up(self):
        # Check the actual implementation of nearest-neighbour interpolation.
        # Nearest-up asserts that half-integer case (1.5) rounds up to 2
        interp10 = interp1d(self.x10, self.y10, kind='nearest-up')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.))
        assert_array_almost_equal(interp10(1.5), np.array(2.))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2., 6., 6.]),)

        # test fill_value="extrapolate"
        extrapolator = interp1d(self.x10, self.y10, kind='nearest-up',
                                fill_value='extrapolate')
        assert_allclose(extrapolator([-1., 0, 9, 11]),
                        [0, 0, 9, 9], rtol=1e-14)

        # Define options for interpolation with nearest-up method
        opts = dict(kind='nearest-up',
                    fill_value='extrapolate',
                    bounds_error=True)
        # Assert that ValueError is raised when bounds_error=True
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_zero(self):
        # Check the actual implementation of zero-order spline interpolation.
        interp10 = interp1d(self.x10, self.y10, kind='zero')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.))
        assert_array_almost_equal(interp10(1.5), np.array(1.))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2., 5., 6.]))

    def bounds_check_helper(self, interpolant, test_array, fail_value):
        # Asserts that a ValueError is raised and that the error message
        # contains the value causing this exception.
        assert_raises(ValueError, interpolant, test_array)
        try:
            interpolant(test_array)
        except ValueError as err:
            assert (f"{fail_value}" in str(err))
    # 定义一个边界检查方法，用于测试处理超出边界输入的正确性
    def _bounds_check(self, kind='linear'):
        # 使用 interp1d 函数创建 extrap10 对象，用于线性插值
        extrap10 = interp1d(self.x10, self.y10, fill_value=self.fill_value,
                            bounds_error=False, kind=kind)

        # 断言对超出边界的输入进行处理后的结果是否与 fill_value 相同
        assert_array_equal(extrap10(11.2), np.array(self.fill_value))
        assert_array_equal(extrap10(-3.4), np.array(self.fill_value))
        assert_array_equal(extrap10([[[11.2], [-3.4], [12.6], [19.3]]]),
                           np.array(self.fill_value),)
        
        # 调用 extrap10 对象的 _check_bounds 方法，检查输入是否超出边界
        assert_array_equal(extrap10._check_bounds(
                               np.array([-1.0, 0.0, 5.0, 9.0, 11.0])),
                           np.array([[True, False, False, False, False],
                                     [False, False, False, False, True]]))

        # 创建一个抛出边界错误的 interp1d 对象
        raises_bounds_error = interp1d(self.x10, self.y10, bounds_error=True,
                                       kind=kind)

        # 使用辅助函数进行边界检查，检查是否抛出预期的边界错误
        self.bounds_check_helper(raises_bounds_error, -1.0, -1.0)
        self.bounds_check_helper(raises_bounds_error, 11.0, 11.0)
        self.bounds_check_helper(raises_bounds_error, [0.0, -1.0, 0.0], -1.0)
        self.bounds_check_helper(raises_bounds_error, [0.0, 1.0, 21.0], 21.0)

        # 直接调用 raises_bounds_error 对象，检查是否抛出预期的边界错误
        raises_bounds_error([0.0, 5.0, 9.0])

    # 定义一个边界检查和 NaN 填充的整数方法
    def _bounds_check_int_nan_fill(self, kind='linear'):
        # 创建 x 和 y 数组，进行整数插值，使用 np.nan 进行填充
        x = np.arange(10).astype(int)
        y = np.arange(10).astype(int)
        c = interp1d(x, y, kind=kind, fill_value=np.nan, bounds_error=False)
        
        # 断言对于 x - 1 的插值结果是否为 NaN
        yi = c(x - 1)
        assert_(np.isnan(yi[0]))
        assert_array_almost_equal(yi, np.r_[np.nan, y[:-1]])

    # 定义一个测试边界的方法
    def test_bounds(self):
        # 遍历不同的插值方法进行边界测试
        for kind in ('linear', 'cubic', 'nearest', 'previous', 'next',
                     'slinear', 'zero', 'quadratic'):
            # 调用 _bounds_check 方法进行边界测试
            self._bounds_check(kind)
            # 调用 _bounds_check_int_nan_fill 方法进行边界和 NaN 填充测试
            self._bounds_check_int_nan_fill(kind)

    # 定义一个测试填充值的方法
    def test_fill_value(self):
        # 测试两个元素填充值是否有效
        for kind in ('linear', 'nearest', 'cubic', 'slinear', 'quadratic',
                     'zero', 'previous', 'next'):
            # 调用 _check_fill_value 方法进行填充值测试
            self._check_fill_value(kind)

    # 定义一个测试填充值可写性的方法
    def test_fill_value_writeable(self):
        # 向后兼容性测试：fill_value 是一个公共可写属性
        interp = interp1d(self.x10, self.y10, fill_value=123.0)
        assert_equal(interp.fill_value, 123.0)
        interp.fill_value = 321.0
        assert_equal(interp.fill_value, 321.0)
    def _nd_check_interp(self, kind='linear'):
        # 检查输入和输出为多维时的行为。

        # 多维输入。
        interp10 = interp1d(self.x10, self.y10, kind=kind)
        assert_array_almost_equal(interp10(np.array([[3., 5.], [2., 7.]])),
                                  np.array([[3., 5.], [2., 7.]]))

        # 标量输入 -> 0 维标量数组输出
        assert_(isinstance(interp10(1.2), np.ndarray))
        assert_equal(interp10(1.2).shape, ())

        # 多维输出。
        interp210 = interp1d(self.x10, self.y210, kind=kind)
        assert_array_almost_equal(interp210(1.), np.array([1., 11.]))
        assert_array_almost_equal(interp210(np.array([1., 2.])),
                                  np.array([[1., 2.], [11., 12.]]))

        interp102 = interp1d(self.x10, self.y102, axis=0, kind=kind)
        assert_array_almost_equal(interp102(1.), np.array([2.0, 3.0]))
        assert_array_almost_equal(interp102(np.array([1., 3.])),
                                  np.array([[2., 3.], [6., 7.]]))

        # 同时进行多维输入和输出测试！
        x_new = np.array([[3., 5.], [2., 7.]])
        assert_array_almost_equal(interp210(x_new),
                                  np.array([[[3., 5.], [2., 7.]],
                                            [[13., 15.], [12., 17.]]]))
        assert_array_almost_equal(interp102(x_new),
                                  np.array([[[6., 7.], [10., 11.]],
                                            [[4., 5.], [14., 15.]]]))

    def _nd_check_shape(self, kind='linear'):
        # 检查大型 N 维输出的形状
        a = [4, 5, 6, 7]
        y = np.arange(np.prod(a)).reshape(*a)
        for n, s in enumerate(a):
            x = np.arange(s)
            z = interp1d(x, y, axis=n, kind=kind)
            assert_array_almost_equal(z(x), y, err_msg=kind)

            x2 = np.arange(2*3*1).reshape((2,3,1)) / 12.
            b = list(a)
            b[n:n+1] = [2,3,1]
            assert_array_almost_equal(z(x2).shape, b, err_msg=kind)

    def test_nd(self):
        for kind in ('linear', 'cubic', 'slinear', 'quadratic', 'nearest',
                     'zero', 'previous', 'next'):
            self._nd_check_interp(kind)
            self._nd_check_shape(kind)

    def _check_complex(self, dtype=np.complex128, kind='linear'):
        x = np.array([1, 2.5, 3, 3.1, 4, 6.4, 7.9, 8.0, 9.5, 10])
        y = x * x ** (1 + 2j)
        y = y.astype(dtype)

        # 简单测试
        c = interp1d(x, y, kind=kind)
        assert_array_almost_equal(y[:-1], c(x)[:-1])

        # 针对分别插值实部和虚部的检查
        xi = np.linspace(1, 10, 31)
        cr = interp1d(x, y.real, kind=kind)
        ci = interp1d(x, y.imag, kind=kind)
        assert_array_almost_equal(c(xi).real, cr(xi))
        assert_array_almost_equal(c(xi).imag, ci(xi))
    def test_complex(self):
        # 对复数类型进行测试，包括不同的插值方法
        for kind in ('linear', 'nearest', 'cubic', 'slinear', 'quadratic',
                     'zero', 'previous', 'next'):
            # 调用 _check_complex 方法测试 np.complex64 类型和 np.complex128 类型
            self._check_complex(np.complex64, kind)
            self._check_complex(np.complex128, kind)

    @pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
    def test_circular_refs(self):
        # 测试 interp1d 是否能自动被垃圾回收
        x = np.linspace(0, 1)
        y = np.linspace(0, 1)
        # 确认在使用后 interp1d 能够从内存中释放
        with assert_deallocated(interp1d, x, y) as interp:
            interp([0.1, 0.2])
            del interp

    def test_overflow_nearest(self):
        # 测试当输入为整数时，x 范围不会溢出
        for kind in ('nearest', 'previous', 'next'):
            x = np.array([0, 50, 127], dtype=np.int8)
            # 创建 interp1d 实例，使用指定的插值方法 kind
            ii = interp1d(x, x, kind=kind)
            # 断言 ii(x) 的结果与 x 接近
            assert_array_almost_equal(ii(x), x)

    def test_local_nans(self):
        # 检查对于局部插值方法（slinear, zero），单个 NaN 只影响其局部邻域
        x = np.arange(10).astype(float)
        y = x.copy()
        y[6] = np.nan
        for kind in ('zero', 'slinear'):
            # 创建 interp1d 实例，使用指定的插值方法 kind
            ir = interp1d(x, y, kind=kind)
            # 对 ir([4.9, 7.0]) 的结果进行断言，确保全部为有限值
            assert_(np.isfinite(ir([4.9, 7.0])).all())

    def test_spline_nans(self):
        # 向后兼容性：单个 NaN 使整个样条插值返回正确形状的 NaN 数组
        # 不会引发异常，只会因为向后兼容性而安静地产生 NaN
        x = np.arange(8).astype(float)
        y = x.copy()
        yn = y.copy()
        yn[3] = np.nan

        for kind in ['quadratic', 'cubic']:
            # 创建 interp1d 实例，使用指定的插值方法 kind
            ir = interp1d(x, y, kind=kind)
            irn = interp1d(x, yn, kind=kind)
            for xnew in (6, [1, 6], [[1, 6], [3, 5]]):
                xnew = np.asarray(xnew)
                out, outn = ir(x), irn(x)
                # 断言 irn(x) 的结果全部为 NaN
                assert_(np.isnan(outn).all())
                # 断言 out 和 outn 的形状相同
                assert_equal(out.shape, outn.shape)

    def test_all_nans(self):
        # 对于所有输入为 NaN 的情况，检测 gh-11637: interp1d 是否会崩溃
        x = np.ones(10) * np.nan
        y = np.arange(10)
        # 使用断言检查是否会引发 ValueError
        with assert_raises(ValueError):
            interp1d(x, y, kind='cubic')

    def test_read_only(self):
        x = np.arange(0, 10)
        y = np.exp(-x / 3.0)
        xnew = np.arange(0, 9, 0.1)
        # 检查可读写和只读两种情况：
        for xnew_writeable in (True, False):
            xnew.flags.writeable = xnew_writeable
            x.flags.writeable = False
            for kind in ('linear', 'nearest', 'zero', 'slinear', 'quadratic',
                         'cubic'):
                # 创建 interp1d 实例，使用指定的插值方法 kind
                f = interp1d(x, y, kind=kind)
                # 断言 f(xnew) 的结果全部为有限值
                assert_(np.isfinite(f(xnew)).all())

    @pytest.mark.parametrize(
        "kind", ("linear", "nearest", "nearest-up", "previous", "next")
    )
    # 定义测试方法，用于单个数值插值函数的测试
    def test_single_value(self, kind):
        # 引用 GitHub 上的 issue，描述了与 interp1d 函数相关的问题
        # 创建 interp1d 对象，对给定的数据进行插值
        f = interp1d([1.5], [6], kind=kind, bounds_error=False,
                     fill_value=(2, 10))
        # 断言插值结果与预期的数组相等
        assert_array_equal(f([1, 1.5, 2]), [2, 6, 10])
        # 当 bounds_error=True 时，检查是否仍然会引发错误
        f = interp1d([1.5], [6], kind=kind, bounds_error=True)
        # 使用 assert_raises 检查是否抛出 ValueError，并匹配指定的错误信息
        with assert_raises(ValueError, match="x_new is above"):
            f(2.0)
class TestLagrange:

    def test_lagrange(self):
        # 创建一个一维多项式对象 p，系数为 [5, 2, 1, 4, 3]
        p = poly1d([5,2,1,4,3])
        # 生成一个与 p 系数长度相同的数组 xs，包含从 0 到 len(p.coeffs)-1 的整数
        xs = np.arange(len(p.coeffs))
        # 计算多项式 p 在 xs 点处的值，形成数组 ys
        ys = p(xs)
        # 使用 Lagrange 插值法计算出的插值多项式 pl
        pl = lagrange(xs, ys)
        # 断言 p 的系数与 pl 的系数几乎相等
        assert_array_almost_equal(p.coeffs, pl.coeffs)


class TestAkima1DInterpolator:
    def test_eval(self):
        # 定义输入数据 x 和 y
        x = np.arange(0., 11.)
        y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        # 使用 Akima 插值方法创建插值对象 ak
        ak = Akima1DInterpolator(x, y)
        # 定义要求值的点 xi
        xi = np.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
                       8.6, 9.9, 10.])
        # 参考值 yi，通过 MATLAB 的 makima 函数生成
        yi = np.array([0., 1.375, 2., 1.5, 1.953125, 2.484375,
                       4.1363636363636366866103344, 5.9803623910336236590978842,
                       5.5067291516462386624652936, 5.2031367459745245795943447,
                       4.1796554159017080820603951, 3.4110386597938129327189927,
                       3.])
        # 断言插值对象 ak 在点 xi 处的计算值与参考值 yi 几乎相等
        assert_allclose(ak(xi), yi)

    def test_eval_mod(self):
        # 参考值通过 MATLAB 生成，使用 makima 方法的 Akima 插值对象 ak
        x = np.arange(0., 11.)
        y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        ak = Akima1DInterpolator(x, y, method="makima")
        xi = np.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
                       8.6, 9.9, 10.])
        yi = np.array([
            0.0, 1.34471153846154, 2.0, 1.44375, 1.94375, 2.51939102564103,
            4.10366931918656, 5.98501550899192, 5.51756330960439, 5.1757231914014,
            4.12326636931311, 3.32931513157895, 3.0])
        # 断言插值对象 ak 在点 xi 处的计算值与参考值 yi 几乎相等
        assert_allclose(ak(xi), yi)

    def test_eval_2d(self):
        # 定义输入数据 x 和 y，y 是两列的矩阵
        x = np.arange(0., 11.)
        y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        y = np.column_stack((y, 2. * y))
        # 使用 Akima 插值方法创建插值对象 ak
        ak = Akima1DInterpolator(x, y)
        # 定义要求值的点 xi
        xi = np.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
                       8.6, 9.9, 10.])
        # 参考值 yi
        yi = np.array([0., 1.375, 2., 1.5, 1.953125, 2.484375,
                       4.1363636363636366866103344,
                       5.9803623910336236590978842,
                       5.5067291516462386624652936,
                       5.2031367459745245795943447,
                       4.1796554159017080820603951,
                       3.4110386597938129327189927, 3.])
        yi = np.column_stack((yi, 2. * yi))
        # 断言插值对象 ak 在点 xi 处的计算值与参考值 yi 几乎相等
        assert_allclose(ak(xi), yi)
    def test_eval_3d(self):
        # 准备输入数据 x 和 y_
        x = np.arange(0., 11.)
        y_ = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        # 创建一个空的 3 维数组 y，填充数据到不同的维度
        y = np.empty((11, 2, 2))
        y[:, 0, 0] = y_
        y[:, 1, 0] = 2. * y_
        y[:, 0, 1] = 3. * y_
        y[:, 1, 1] = 4. * y_
        # 使用 Akima1DInterpolator 类初始化 ak 对象，传入 x 和 y
        ak = Akima1DInterpolator(x, y)
        # 准备用于评估的 xi 数据
        xi = np.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
                       8.6, 9.9, 10.])
        # 准备用于比较的 yi 数据
        yi = np.empty((13, 2, 2))
        yi_ = np.array([0., 1.375, 2., 1.5, 1.953125, 2.484375,
                        4.1363636363636366866103344,
                        5.9803623910336236590978842,
                        5.5067291516462386624652936,
                        5.2031367459745245795943447,
                        4.1796554159017080820603951,
                        3.4110386597938129327189927, 3.])
        yi[:, 0, 0] = yi_
        yi[:, 1, 0] = 2. * yi_
        yi[:, 0, 1] = 3. * yi_
        yi[:, 1, 1] = 4. * yi_
        # 使用 assert_allclose 函数比较 ak 对象在 xi 上的评估结果和预期的 yi
        assert_allclose(ak(xi), yi)

    def test_degenerate_case_multidimensional(self):
        # 这个测试用例是为了检查问题 #5683
        # 准备输入数据 x 和 y
        x = np.array([0, 1, 2])
        y = np.vstack((x, x**2)).T
        # 使用 Akima1DInterpolator 类初始化 ak 对象，传入 x 和 y
        ak = Akima1DInterpolator(x, y)
        # 准备用于评估的 x_eval 数据
        x_eval = np.array([0.5, 1.5])
        # 评估 ak 对象在 x_eval 上的结果
        y_eval = ak(x_eval)
        # 使用 assert_allclose 函数比较评估结果 y_eval 和预期的 y_eval
        assert_allclose(y_eval, np.vstack((x_eval, x_eval**2)).T)

    def test_extend(self):
        # 准备输入数据 x 和 y
        x = np.arange(0., 11.)
        y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        # 使用 Akima1DInterpolator 类初始化 ak 对象，传入 x 和 y
        ak = Akima1DInterpolator(x, y)
        # 准备匹配字符串
        match = "Extending a 1-D Akima interpolator is not yet implemented"
        # 使用 pytest.raises 检查是否会抛出 NotImplementedError 异常，并匹配指定的错误信息
        with pytest.raises(NotImplementedError, match=match):
            ak.extend(None, None)

    def test_mod_invalid_method(self):
        # 准备输入数据 x 和 y
        x = np.arange(0., 11.)
        y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        # 准备匹配字符串
        match = "`method`=invalid is unsupported."
        # 使用 pytest.raises 检查是否会抛出 NotImplementedError 异常，并匹配指定的错误信息
        with pytest.raises(NotImplementedError, match=match):
            # 使用 Akima1DInterpolator 类初始化 ak 对象，传入 x, y 和一个无效的 method 参数
            Akima1DInterpolator(x, y, method="invalid")  # type: ignore

    def test_extrapolate_attr(self):
        # 准备输入数据 x 和 y
        x = np.linspace(-5, 5, 11)
        y = x**2
        # 准备用于外推的 x_ext 数据
        x_ext = np.linspace(-10, 10, 17)
        y_ext = x_ext**2
        # 使用 Akima1DInterpolator 类初始化 ak_true, ak_false 和 ak_none 对象，分别测试不同的 extrapolate 参数值
        ak_true = Akima1DInterpolator(x, y, extrapolate=True)
        ak_false = Akima1DInterpolator(x, y, extrapolate=False)
        ak_none = Akima1DInterpolator(x, y, extrapolate=None)
        # None 应默认为 False；外推的点应为 NaN
        assert_allclose(ak_false(x_ext), ak_none(x_ext), equal_nan=True, atol=1e-15)
        assert_equal(ak_false(x_ext)[0:4], np.full(4, np.nan))
        assert_equal(ak_false(x_ext)[-4:-1], np.full(3, np.nan))
        # 在调用和属性外推时应相等
        assert_allclose(ak_false(x_ext, extrapolate=True), ak_true(x_ext), atol=1e-15)
        # 测试外推到实际函数
        assert_allclose(y_ext, ak_true(x_ext), atol=1e-15)
@pytest.mark.parametrize("method", [Akima1DInterpolator, PchipInterpolator])
# 使用 pytest 的参数化功能，对 Akima1DInterpolator 和 PchipInterpolator 进行测试
def test_complex(method):
    # 复数数值数据已被弃用
    x = np.arange(0., 11.)
    y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
    # 将 y 数组中所有元素变为复数
    y = y - 2j*y
    msg = "real values"
    # 使用 pytest 检查是否抛出 ValueError 异常，且异常消息为 "real values"
    with pytest.raises(ValueError, match=msg):
        method(x, y)


class TestPPolyCommon:
    # 对 PPoly 和 BPoly 的基本功能进行测试
    def test_sort_check(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 1, 0.5])
        # 断言应该抛出 ValueError 异常，因为 c 和 x 的形状不匹配
        assert_raises(ValueError, PPoly, c, x)
        assert_raises(ValueError, BPoly, c, x)

    def test_ctor_c(self):
        # 错误的形状：`c` 必须至少是二维的
        with assert_raises(ValueError):
            PPoly([1, 2], [0, 1])

    def test_extend(self):
        # 测试向分段多项式添加新点
        np.random.seed(1234)

        order = 3
        x = np.unique(np.r_[0, 10 * np.random.rand(30), 10])
        c = 2*np.random.rand(order+1, len(x)-1, 2, 3) - 1

        for cls in (PPoly, BPoly):
            # 创建一个分段多项式对象 pp，并向其添加新的系数和节点
            pp = cls(c[:,:9], x[:10])
            pp.extend(c[:,9:], x[10:])

            pp2 = cls(c[:, 10:], x[10:])
            pp2.extend(c[:, :10], x[:10])

            pp3 = cls(c, x)

            # 断言 pp 和 pp3 的系数和节点相等
            assert_array_equal(pp.c, pp3.c)
            assert_array_equal(pp.x, pp3.x)
            assert_array_equal(pp2.c, pp3.c)
            assert_array_equal(pp2.x, pp3.x)

    def test_extend_diff_orders(self):
        # 测试以不同阶数扩展多项式
        np.random.seed(1234)

        x = np.linspace(0, 1, 6)
        c = np.random.rand(2, 5)

        x2 = np.linspace(1, 2, 6)
        c2 = np.random.rand(4, 5)

        for cls in (PPoly, BPoly):
            pp1 = cls(c, x)
            pp2 = cls(c2, x2)

            pp_comb = cls(c, x)
            pp_comb.extend(c2, x2[1:])

            # 注意：由于随机系数，pp1 与 pp2 在端点处可能不匹配
            xi1 = np.linspace(0, 1, 300, endpoint=False)
            xi2 = np.linspace(1, 2, 300)

            assert_allclose(pp1(xi1), pp_comb(xi1))
            assert_allclose(pp2(xi2), pp_comb(xi2))

    def test_extend_descending(self):
        np.random.seed(0)

        order = 3
        x = np.sort(np.random.uniform(0, 10, 20))
        c = np.random.rand(order + 1, x.shape[0] - 1, 2, 3)

        for cls in (PPoly, BPoly):
            p = cls(c, x)

            p1 = cls(c[:, :9], x[:10])
            p1.extend(c[:, 9:], x[10:])

            p2 = cls(c[:, 10:], x[10:])
            p2.extend(c[:, :10], x[:10])

            # 断言 p1 和 p2 的系数和节点与 p 的相同
            assert_array_equal(p1.c, p.c)
            assert_array_equal(p1.x, p.x)
            assert_array_equal(p2.c, p.c)
            assert_array_equal(p2.x, p.x)
    # 定义一个测试方法来验证多项式对象的形状处理
    def test_shape(self):
        # 设置随机种子以确保结果可重现
        np.random.seed(1234)
        # 创建一个随机的多维数组 c，形状为 (8, 12, 5, 6, 7)
        c = np.random.rand(8, 12, 5, 6, 7)
        # 对 x 进行排序，x 是一个包含 13 个随机数的数组
        x = np.sort(np.random.rand(13))
        # 创建一个随机的 2 维数组 xp，形状为 (3, 4)
        xp = np.random.rand(3, 4)

        # 遍历两种多项式类 PPoly 和 BPoly
        for cls in (PPoly, BPoly):
            # 使用 c 和 x 创建一个多项式对象 p
            p = cls(c, x)
            # 验证 p(xp) 的形状是否为 (3, 4, 5, 6, 7)
            assert_equal(p(xp).shape, (3, 4, 5, 6, 7))

        # 测试处理标量值的情况
        # 遍历两种多项式类 PPoly 和 BPoly
        for cls in (PPoly, BPoly):
            # 创建一个仅包含 c 的第一维数据的多项式对象 p
            p = cls(c[..., 0, 0, 0], x)

            # 验证 p(0.5) 的形状是否为 ()
            assert_equal(np.shape(p(0.5)), ())
            # 验证 p(np.array(0.5)) 的形状是否为 ()
            assert_equal(np.shape(p(np.array(0.5))), ())

            # 验证当传入一个包含不同数据类型的数组时会抛出 ValueError 异常
            assert_raises(ValueError, p, np.array([[0.1, 0.2], [0.4]], dtype=object))

    # 定义一个测试方法来验证复数系数的情况
    def test_complex_coef(self):
        # 设置随机种子以确保结果可重现
        np.random.seed(12345)
        # 对 x 进行排序，x 是一个包含 13 个随机数的数组
        x = np.sort(np.random.random(13))
        # 创建一个具有复数值的随机系数数组 c，形状为 (8, 12)
        c = np.random.random((8, 12)) * (1. + 0.3j)
        # 分别获取 c 的实部和虚部
        c_re, c_im = c.real, c.imag
        # 创建一个随机的一维数组 xp，长度为 5
        xp = np.random.random(5)

        # 遍历两种多项式类 PPoly 和 BPoly
        for cls in (PPoly, BPoly):
            # 使用 c、c_re、c_im 和 x 创建三个不同的多项式对象 p、p_re、p_im
            p, p_re, p_im = cls(c, x), cls(c_re, x), cls(c_im, x)
            # 遍历 [0, 1, 2] 三种导数阶数
            for nu in [0, 1, 2]:
                # 验证 p(xp, nu).real 和 p_re(xp, nu) 的值是否接近
                assert_allclose(p(xp, nu).real, p_re(xp, nu))
                # 验证 p(xp, nu).imag 和 p_im(xp, nu) 的值是否接近
                assert_allclose(p(xp, nu).imag, p_im(xp, nu))

    # 定义一个测试方法来验证轴向处理的情况
    def test_axis(self):
        # 设置随机种子以确保结果可重现
        np.random.seed(12345)
        # 创建一个随机的五维数组 c，形状为 (3, 4, 5, 6, 7, 8)
        c = np.random.rand(3, 4, 5, 6, 7, 8)
        # 记录 c 的形状
        c_s = c.shape
        # 创建一个随机的二维数组 xp，形状为 (1, 2)
        xp = np.random.random((1, 2))

        # 遍历轴向 [0, 1, 2, 3]
        for axis in (0, 1, 2, 3):
            # 获取指定轴向的维度大小 m
            m = c.shape[axis + 1]
            # 对 x 进行排序，x 的长度为 m+1
            x = np.sort(np.random.rand(m + 1))
            # 遍历两种多项式类 PPoly 和 BPoly
            for cls in (PPoly, BPoly):
                # 使用 c、x 和 axis 创建一个多项式对象 p
                p = cls(c, x, axis=axis)
                # 验证 p.c 的形状是否与预期相符
                assert_equal(p.c.shape,
                             c_s[axis:axis + 2] + c_s[:axis] + c_s[axis + 2:])
                # 计算 p(xp) 的结果
                res = p(xp)
                # 计算目标形状 targ_shape
                targ_shape = c_s[:axis] + xp.shape + c_s[2 + axis:]
                # 验证 p(xp) 的形状是否与 targ_shape 相符
                assert_equal(res.shape, targ_shape)

                # 对于导数和原函数，验证其不会去除轴向
                for p1 in [cls(c, x, axis=axis).derivative(),
                           cls(c, x, axis=axis).derivative(2),
                           cls(c, x, axis=axis).antiderivative(),
                           cls(c, x, axis=axis).antiderivative(2)]:
                    # 验证 p1 的轴向是否与 p 的轴向相同
                    assert_equal(p1.axis, p.axis)

        # 验证当轴向超出有效范围时是否会引发 ValueError 异常
        for axis in (-1, 4, 5, 6):
            # 遍历两种多项式类 BPoly 和 PPoly
            for cls in (BPoly, PPoly):
                # 验证当轴向参数为无效值时是否会引发 ValueError 异常
                assert_raises(ValueError, cls, **dict(c=c, x=x, axis=axis))
class TestPolySubclassing:
    # 定义一个测试类 TestPolySubclassing，用于测试多项式子类化功能

    class P(PPoly):
        # 定义内部类 P，继承自 PPoly 类
        pass

    class B(BPoly):
        # 定义内部类 B，继承自 BPoly 类
        pass

    def _make_polynomials(self):
        # 定义生成多项式的私有方法 _make_polynomials

        np.random.seed(1234)
        # 设定随机种子为 1234

        x = np.sort(np.random.random(3))
        # 生成长度为 3 的随机数，并排序，赋值给变量 x

        c = np.random.random((4, 2))
        # 生成一个 4x2 的随机数组，赋值给变量 c

        return self.P(c, x), self.B(c, x)
        # 返回通过类 P 和 B 的构造函数生成的多项式对象

    def test_derivative(self):
        # 定义测试导数的方法 test_derivative

        pp, bp = self._make_polynomials()
        # 调用 _make_polynomials 方法生成多项式对象 pp 和 bp

        for p in (pp, bp):
            # 对于每一个多项式 p 在 pp 和 bp 中

            pd = p.derivative()
            # 计算 p 的导数，赋值给变量 pd

            assert_equal(p.__class__, pd.__class__)
            # 断言 p 的类与其导数 pd 的类相同

        ppa = pp.antiderivative()
        # 计算 pp 的反导数，赋值给变量 ppa

        assert_equal(pp.__class__, ppa.__class__)
        # 断言 pp 的类与其反导数 ppa 的类相同

    def test_from_spline(self):
        # 定义从样条曲线生成多项式的测试方法 test_from_spline

        np.random.seed(1234)
        # 设定随机种子为 1234

        x = np.sort(np.r_[0, np.random.rand(11), 1])
        # 生成一个包含 0、1 之间随机数和 0、1 的数组，并排序，赋值给变量 x

        y = np.random.rand(len(x))
        # 生成长度与 x 相同的随机数组，赋值给变量 y

        spl = splrep(x, y, s=0)
        # 对 x 和 y 进行样条插值，平滑参数为 0，赋值给变量 spl

        pp = self.P.from_spline(spl)
        # 使用内部类 P 的类方法 from_spline 根据 spl 创建多项式对象 pp

        assert_equal(pp.__class__, self.P)
        # 断言 pp 的类为 P 类

    def test_conversions(self):
        # 定义多项式转换的测试方法 test_conversions

        pp, bp = self._make_polynomials()
        # 调用 _make_polynomials 方法生成多项式对象 pp 和 bp

        pp1 = self.P.from_bernstein_basis(bp)
        # 使用内部类 P 的类方法 from_bernstein_basis 将 bp 转换为 pp1

        assert_equal(pp1.__class__, self.P)
        # 断言 pp1 的类为 P 类

        bp1 = self.B.from_power_basis(pp)
        # 使用内部类 B 的类方法 from_power_basis 将 pp 转换为 bp1

        assert_equal(bp1.__class__, self.B)
        # 断言 bp1 的类为 B 类

    def test_from_derivatives(self):
        # 定义从导数生成多项式的测试方法 test_from_derivatives

        x = [0, 1, 2]
        # 定义数组 x 包含元素 0, 1, 2

        y = [[1], [2], [3]]
        # 定义二维数组 y 包含数组 [1], [2], [3]

        bp = self.B.from_derivatives(x, y)
        # 使用内部类 B 的类方法 from_derivatives 根据 x 和 y 创建多项式对象 bp

        assert_equal(bp.__class__, self.B)
        # 断言 bp 的类为 B 类


class TestPPoly:
    # 定义测试 PPoly 类的测试类 TestPPoly

    def test_simple(self):
        # 定义简单多项式测试方法 test_simple

        c = np.array([[1, 4], [2, 5], [3, 6]])
        # 定义二维数组 c 包含元素 [[1, 4], [2, 5], [3, 6]]

        x = np.array([0, 0.5, 1])
        # 定义数组 x 包含元素 0, 0.5, 1

        p = PPoly(c, x)
        # 使用 PPoly 类创建多项式对象 p

        assert_allclose(p(0.3), 1*0.3**2 + 2*0.3 + 3)
        # 断言 p(0.3) 的值接近于 1*0.3^2 + 2*0.3 + 3

        assert_allclose(p(0.7), 4*(0.7-0.5)**2 + 5*(0.7-0.5) + 6)
        # 断言 p(0.7) 的值接近于 4*(0.7-0.5)^2 + 5*(0.7-0.5) + 6

    def test_periodic(self):
        # 定义周期多项式测试方法 test_periodic

        c = np.array([[1, 4], [2, 5], [3, 6]])
        # 定义二维数组 c 包含元素 [[1, 4], [2, 5], [3, 6]]

        x = np.array([0, 0.5, 1])
        # 定义数组 x 包含元素 0, 0.5, 1

        p = PPoly(c, x, extrapolate='periodic')
        # 使用 PPoly 类创建周期多项式对象 p

        assert_allclose(p(1.3), 1 * 0.3 ** 2 + 2 * 0.3 + 3)
        # 断言 p(1.3) 的值接近于 1 * 0.3 ** 2 + 2 * 0.3 + 3

        assert_allclose(p(-0.3), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)
        # 断言 p(-0.3) 的值接近于 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6

        assert_allclose(p(1.3, 1), 2 * 0.3 + 2)
        # 断言 p(1.3, 1) 的值接近于 2 * 0.3 + 2

        assert_allclose(p(-0.3, 1), 8 * (0.7 - 0.5) + 5)
        # 断言 p(-0.3, 1) 的值接近于 8 * (0.7 - 0.5) + 5

    def test_read_only(self):
        # 定义只读测试方法 test_read_only

        c = np.array([[1, 4], [2, 5], [3, 6]])
        # 定义二维数组 c 包含元素 [[1, 4], [2, 5], [3, 6]]

        x = np.array([0, 0.5, 1])
        # 定义数组 x 包含元素 0, 0.5, 1

        xnew = np.array([0, 0.1, 0.2])
        # 定义数组 xnew 包含元素 0, 0.1, 0.2

        PPoly(c, x, extrapolate='periodic')
        # 使用 PPoly 类创建周期多项式对象

        for writeable in (True, False):
            # 对于每一个写入标志为 True 和 False 的情况

            x.flags.writeable = writeable
            # 设置 x 的可写标志为 writeable

            c.flags.writeable = writeable
            # 设置 c 的可写标志为 writeable

            f = PPoly(c, x)
            # 使用 PPoly 类创建多项式对象 f

            vals = f(xnew)
            # 计算 f 在 xnew 上的值，赋值给
    def test_descending(self):
        # 定义一个内部函数，用于生成二项式系数矩阵
        def binom_matrix(power):
            # 创建一个以0到power为元素的数组，并重塑为列向量
            n = np.arange(power + 1).reshape(-1, 1)
            # 创建一个0到power的数组
            k = np.arange(power + 1)
            # 使用二项式函数生成二项式系数矩阵
            B = binom(n, k)
            return B[::-1, ::-1]  # 返回颠倒行列顺序的矩阵

        np.random.seed(0)  # 设置随机数种子为0

        power = 3  # 设定幂次为3
        for m in [10, 20, 30]:  # 遍历不同的m值
            # 生成一个从0到10均匀分布的有序数组，并进行排序
            x = np.sort(np.random.uniform(0, 10, m + 1))
            # 生成一个(power+1) x m的随机均匀分布数组
            ca = np.random.uniform(-2, 2, size=(power + 1, m))

            # 计算x数组的差值
            h = np.diff(x)
            # 计算差值的各次幂
            h_powers = h[None, :] ** np.arange(power + 1)[::-1, None]
            # 调用binom_matrix函数，生成二项式系数矩阵
            B = binom_matrix(power)
            # 对ca和h_powers数组进行元素级别的乘法
            cap = ca * h_powers
            # 矩阵乘法，计算CDP
            cdp = np.dot(B.T, cap)
            # 对CDP进行元素级别的除法
            cd = cdp / h_powers

            # 创建PPoly对象pa，使用ca和x创建
            pa = PPoly(ca, x, extrapolate=True)
            # 创建PPoly对象pd，使用cd和x[::-1]创建
            pd = PPoly(cd[:, ::-1], x[::-1], extrapolate=True)

            # 生成一个从-10到20均匀分布的100个随机数数组
            x_test = np.random.uniform(-10, 20, 100)
            # 断言pa和pd在x_test上的值近似相等
            assert_allclose(pa(x_test), pd(x_test), rtol=1e-13)
            # 断言pa和pd的一阶导数在x_test上的值近似相等
            assert_allclose(pa(x_test, 1), pd(x_test, 1), rtol=1e-13)

            # 计算pa和pd的一阶导数
            pa_d = pa.derivative()
            pd_d = pd.derivative()
            # 断言pa_d和pd_d在x_test上的值近似相等
            assert_allclose(pa_d(x_test), pd_d(x_test), rtol=1e-13)

            # 由于修复连续性是以相反的顺序完成的，因此原函数不会相等，
            # 但差异应该相等。
            # 计算pa和pd的不定积分
            pa_i = pa.antiderivative()
            pd_i = pd.antiderivative()
            # 对于从-10到20均匀分布的5个范围的随机数对(a, b)
            for a, b in np.random.uniform(-10, 20, (5, 2)):
                # 计算pa和pd在[a, b]区间上的积分
                int_a = pa.integrate(a, b)
                int_d = pd.integrate(a, b)
                # 断言pa和pd在[a, b]区间上的积分近似相等
                assert_allclose(int_a, int_d, rtol=1e-13)
                # 断言pa_i和pd_i在b和a上的值相等
                assert_allclose(pa_i(b) - pa_i(a), pd_i(b) - pd_i(a),
                                rtol=1e-13)

            # 计算pd的根
            roots_d = pd.roots()
            # 计算pa的根
            roots_a = pa.roots()
            # 断言pa和pd的根近似相等
            assert_allclose(roots_a, np.sort(roots_d), rtol=1e-12)

    def test_multi_shape(self):
        # 创建一个随机形状为(6, 2, 1, 2, 3)的数组
        c = np.random.rand(6, 2, 1, 2, 3)
        # 创建一个数组x，包含元素[0, 0.5, 1]
        x = np.array([0, 0.5, 1])
        # 使用给定的c和x创建PPoly对象p
        p = PPoly(c, x)
        # 断言p对象的x形状与x数组相等
        assert_equal(p.x.shape, x.shape)
        # 断言p对象的系数c的形状与c数组相等
        assert_equal(p.c.shape, c.shape)
        # 断言p在0.3处的值的形状与c的形状的第三维度开始相等
        assert_equal(p(0.3).shape, c.shape[2:])

        # 生成一个形状为(5, 6) + c的形状的数组，并断言结果的形状
        assert_equal(p(np.random.rand(5, 6)).shape, (5, 6) + c.shape[2:])

        # 计算p的一阶导数dp，并断言其系数c的形状
        dp = p.derivative()
        assert_equal(dp.c.shape, (5, 2, 1, 2, 3))
        # 计算p的不定积分ip，并断言其系数c的形状
        ip = p.antiderivative()
        assert_equal(ip.c.shape, (7, 2, 1, 2, 3))

    def test_construct_fast(self):
        np.random.seed(1234)  # 设置随机数种子为1234
        # 创建一个形状为(3, 2)的浮点型数组c
        c = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
        # 创建一个数组x，包含元素[0, 0.5, 1]
        x = np.array([0, 0.5, 1])
        # 使用给定的c和x使用构造函数创建PPoly对象p
        p = PPoly.construct_fast(c, x)
        # 断言p在0.3处的值近似等于1*0.3^2 + 2*0.3 + 3
        assert_allclose(p(0.3), 1*0.3**2 + 2*0.3 + 3)
        # 断言p在0.7处的值近似等于4*(0.7-0.5)^2 + 5*(0.7-0.5) + 6
        assert_allclose(p(0.7), 4*(0.7-0.5)**2 + 5*(0.7-0.5) + 6)
    def test_vs_alternative_implementations(self):
        # 设置随机数种子，以确保测试结果可重复
        np.random.seed(1234)
        # 生成一个形状为 (3, 12, 22) 的随机数组
        c = np.random.rand(3, 12, 22)
        # 生成一个包含随机数和固定值的数组，并排序
        x = np.sort(np.r_[0, np.random.rand(11), 1])

        # 使用 PPoly 类构造一个多项式样条对象
        p = PPoly(c, x)

        # 定义要评估的插值点
        xp = np.r_[0.3, 0.5, 0.33, 0.6]
        # 调用 _ppoly_eval_1 函数计算预期值
        expected = _ppoly_eval_1(c, x, xp)
        # 断言 p(xp) 的结果与预期值的接近程度
        assert_allclose(p(xp), expected)

        # 调用 _ppoly_eval_2 函数计算预期值
        expected = _ppoly_eval_2(c[:,:,0], x, xp)
        # 断言 p(xp)[:,0] 的结果与预期值的接近程度
        assert_allclose(p(xp)[:,0], expected)

    def test_from_spline(self):
        # 设置随机数种子，以确保测试结果可重复
        np.random.seed(1234)
        # 生成一个包含随机数和固定值的数组，并排序
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        # 生成一个与 x 长度相同的随机数数组
        y = np.random.rand(len(x))

        # 使用 splrep 函数生成一个样条曲线
        spl = splrep(x, y, s=0)
        # 使用 PPoly 类从样条曲线生成分段多项式对象
        pp = PPoly.from_spline(spl)

        # 定义要评估的插值点
        xi = np.linspace(0, 1, 200)
        # 断言 pp(xi) 的结果与 splev(xi, spl) 的接近程度
        assert_allclose(pp(xi), splev(xi, spl))

        # 确保 .from_spline 方法接受 BSpline 对象
        b = BSpline(*spl)
        ppp = PPoly.from_spline(b)
        # 断言 ppp(xi) 的结果与 b(xi) 的接近程度
        assert_allclose(ppp(xi), b(xi))

        # 检查 BSpline 对象的 extrapolate 属性是否正确传播
        t, c, k = spl
        for extrap in (None, True, False):
            b = BSpline(t, c, k, extrapolate=extrap)
            p = PPoly.from_spline(b)
            # 断言 p 的 extrapolate 属性与 b 的 extrapolate 属性相等
            assert_equal(p.extrapolate, b.extrapolate)

    def test_derivative_simple(self):
        # 设置随机数种子，以确保测试结果可重复
        np.random.seed(1234)
        # 构造一个形状为 (1, 2) 的数组
        c = np.array([[4, 3, 2, 1]]).T
        # 计算其一阶导数的系数
        dc = np.array([[3*4, 2*3, 2]]).T
        # 计算其二阶导数的系数
        ddc = np.array([[2*3*4, 1*2*3]]).T
        # 定义节点数组
        x = np.array([0, 1])

        # 使用 PPoly 类构造三个不同阶数的多项式样条对象
        pp = PPoly(c, x)
        dpp = PPoly(dc, x)
        ddpp = PPoly(ddc, x)

        # 断言 pp.derivative().c 与 dpp.c 的接近程度
        assert_allclose(pp.derivative().c, dpp.c)
        # 断言 pp.derivative(2).c 与 ddpp.c 的接近程度
        assert_allclose(pp.derivative(2).c, ddpp.c)

    def test_derivative_eval(self):
        # 设置随机数种子，以确保测试结果可重复
        np.random.seed(1234)
        # 生成一个包含随机数和固定值的数组，并排序
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        # 生成一个与 x 长度相同的随机数数组
        y = np.random.rand(len(x))

        # 使用 splrep 函数生成一个样条曲线
        spl = splrep(x, y, s=0)
        # 使用 PPoly 类从样条曲线生成分段多项式对象
        pp = PPoly.from_spline(spl)

        # 定义要评估的插值点
        xi = np.linspace(0, 1, 200)
        # 循环计算 pp(xi, dx) 和 splev(xi, spl, dx) 的接近程度，dx 取值范围为 0 到 2
        for dx in range(0, 3):
            assert_allclose(pp(xi, dx), splev(xi, spl, dx))

    def test_derivative(self):
        # 设置随机数种子，以确保测试结果可重复
        np.random.seed(1234)
        # 生成一个包含随机数和固定值的数组，并排序
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        # 生成一个与 x 长度相同的随机数数组
        y = np.random.rand(len(x))

        # 使用 splrep 函数生成一个样条曲线，指定阶数为 5
        spl = splrep(x, y, s=0, k=5)
        # 使用 PPoly 类从样条曲线生成分段多项式对象
        pp = PPoly.from_spline(spl)

        # 定义要评估的插值点
        xi = np.linspace(0, 1, 200)
        # 循环计算 pp(xi, dx) 和 pp.derivative(dx)(xi) 的接近程度，dx 取值范围为 0 到 9
        for dx in range(0, 10):
            assert_allclose(pp(xi, dx), pp.derivative(dx)(xi),
                            err_msg="dx=%d" % (dx,))

    def test_antiderivative_of_constant(self):
        # 构造一个常数多项式对象 p
        p = PPoly([[1.]], [0, 1])
        # 断言 p.antiderivative().c 与 PPoly([[1], [0]], [0, 1]).c 的相等性
        assert_equal(p.antiderivative().c, PPoly([[1], [0]], [0, 1]).c)
        # 断言 p.antiderivative().x 与 PPoly([[1], [0]], [0, 1]).x 的相等性
        assert_equal(p.antiderivative().x, PPoly([[1], [0]], [0, 1]).x)

    def test_antiderivative_regression_4355(self):
        # 构造一个分段多项式对象 p
        p = PPoly([[1., 0.5]], [0, 1, 2])
        # 计算其反导数对象 q
        q = p.antiderivative()
        # 断言 q.c 的结果与预期值 [[1, 0.5], [0, 1]] 相等
        assert_equal(q.c, [[1, 0.5], [0, 1]])
        # 断言 q.x 的结果与预期值 [0, 1, 2] 相等
        assert_equal(q.x, [0, 1, 2])
        # 断言 p.integrate(0, 2) 的结果与预期值 1.5 的接近程度
        assert_allclose(p.integrate(0, 2), 1.5)
        # 断言 q(2) - q(0) 的结果与预期值 1.5 的接近程度
        assert_allclose(q(2) - q(0), 1.5)
    # 测试简单反导函数的正确性
    def test_antiderivative_simple(self):
        np.random.seed(1234)
        # 定义多项式 p1(x) = 3*x**2 + 2*x + 1 和 p2(x) = 1.6875
        c = np.array([[3, 2, 1], [0, 0, 1.6875]]).T
        # 定义它们的原函数 pp1(x) = x**3 + x**2 + x 和 pp2(x) = 1.6875*(x - 0.25) + pp1(0.25)
        ic = np.array([[1, 1, 1, 0], [0, 0, 1.6875, 0.328125]]).T
        # 定义它们的二阶原函数 ppp1(x) = (1/4)*x**4 + (1/3)*x**3 + (1/2)*x**2 和 ppp2(x) = (1.6875/2)*(x - 0.25)**2 + pp1(0.25)*x + ppp1(0.25)
        iic = np.array([[1/4, 1/3, 1/2, 0, 0],
                        [0, 0, 1.6875/2, 0.328125, 0.037434895833333336]]).T
        x = np.array([0, 0.25, 1])

        # 创建一个 PPoly 对象 pp，并计算其一阶和二阶原函数 ipp 和 iipp
        pp = PPoly(c, x)
        ipp = pp.antiderivative()
        iipp = pp.antiderivative(2)
        iipp2 = ipp.antiderivative()

        # 断言一阶和二阶原函数的正确性
        assert_allclose(ipp.x, x)
        assert_allclose(ipp.c.T, ic.T)
        assert_allclose(iipp.c.T, iic.T)
        assert_allclose(iipp2.c.T, iic.T)

    # 测试反导函数与导函数的互逆性
    def test_antiderivative_vs_derivative(self):
        np.random.seed(1234)
        x = np.linspace(0, 1, 30)**2
        y = np.random.rand(len(x))
        # 用样条插值生成 PPoly 对象 pp
        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)

        # 对每个阶数 dx 进行反导和导函数比较的测试
        for dx in range(0, 10):
            ipp = pp.antiderivative(dx)

            # 检查导函数是否为原函数的逆操作
            pp2 = ipp.derivative(dx)
            assert_allclose(pp.c, pp2.c)

            # 检查连续性
            for k in range(dx):
                pp2 = ipp.derivative(k)

                r = 1e-13
                endpoint = r*pp2.x[:-1] + (1 - r)*pp2.x[1:]

                assert_allclose(pp2(pp2.x[1:]), pp2(endpoint),
                                rtol=1e-7, err_msg="dx=%d k=%d" % (dx, k))

    # 测试反导函数与样条插值的一致性
    def test_antiderivative_vs_spline(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        # 用样条插值生成 PPoly 对象 pp
        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)

        # 对每个阶数 dx 进行反导与样条插值比较的测试
        for dx in range(0, 10):
            pp2 = pp.antiderivative(dx)
            spl2 = splantider(spl, dx)

            xi = np.linspace(0, 1, 200)
            assert_allclose(pp2(xi), splev(xi, spl2),
                            rtol=1e-7)

    # 测试反导函数的连续性和系数变化
    def test_antiderivative_continuity(self):
        c = np.array([[2, 1, 2, 2], [2, 1, 3, 3]]).T
        x = np.array([0, 0.5, 1])

        # 创建 PPoly 对象 p，并计算其反导函数 ip
        p = PPoly(c, x)
        ip = p.antiderivative()

        # 检查反导函数的连续性
        assert_allclose(ip(0.5 - 1e-9), ip(0.5 + 1e-9), rtol=1e-8)

        # 检查只有最低阶系数被改变
        p2 = ip.derivative()
        assert_allclose(p2.c, p.c)
    def test_integrate(self):
        np.random.seed(1234)  # 设置随机种子以确保可重复性
        x = np.sort(np.r_[0, np.random.rand(11), 1])  # 生成长度为 12 的随机数组并排序

        y = np.random.rand(len(x))  # 生成与 x 同样长度的随机数组

        # 使用 splrep 函数拟合 x, y 数据，生成样条曲线
        spl = splrep(x, y, s=0, k=5)
        
        # 从样条曲线对象中创建 PPoly 对象
        pp = PPoly.from_spline(spl)

        a, b = 0.3, 0.9  # 定义积分区间
        ig = pp.integrate(a, b)  # 计算 PPoly 对象在区间 [a, b] 的积分

        ipp = pp.antiderivative()  # 计算 PPoly 对象的反导函数
        assert_allclose(ig, ipp(b) - ipp(a))  # 检查积分值是否等于反导函数在区间 [a, b] 的差值
        assert_allclose(ig, splint(a, b, spl))  # 使用 splint 函数计算样条曲线在区间 [a, b] 的积分值

        a, b = -0.3, 0.9  # 更新积分区间
        ig = pp.integrate(a, b, extrapolate=True)  # 使用外推进行积分计算
        assert_allclose(ig, ipp(b) - ipp(a))  # 检查外推积分值是否等于反导函数在区间 [a, b] 的差值

        assert_(np.isnan(pp.integrate(a, b, extrapolate=False)).all())  # 检查无外推的积分值是否为 NaN

    def test_integrate_readonly(self):
        x = np.array([1, 2, 4])  # 定义数组 x
        c = np.array([[0., 0.], [-1., -1.], [2., -0.], [1., 2.]])  # 定义系数数组 c

        for writeable in (True, False):  # 循环测试写入权限
            x.flags.writeable = writeable  # 设置数组 x 的写入权限

            P = PPoly(c, x)  # 创建 PPoly 对象
            vals = P.integrate(1, 4)  # 计算 PPoly 对象在区间 [1, 4] 的积分值

            assert_(np.isfinite(vals).all())  # 检查积分值是否为有限数值

    def test_integrate_periodic(self):
        x = np.array([1, 2, 4])  # 定义数组 x
        c = np.array([[0., 0.], [-1., -1.], [2., -0.], [1., 2.]])  # 定义系数数组 c

        P = PPoly(c, x, extrapolate='periodic')  # 创建周期性的 PPoly 对象
        I = P.antiderivative()  # 计算 PPoly 对象的反导函数

        period_int = I(4) - I(1)  # 计算一个周期内的积分值

        # 检查不同区间的积分值是否等于一个周期内的积分值
        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)

        assert_allclose(P.integrate(1.5, 2.5), I(2.5) - I(1.5))
        assert_allclose(P.integrate(3.5, 5), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12),
                        I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5, 5 + 12),
                        I(2) - I(1) + I(4) - I(3.5) + 4 * period_int)

        assert_allclose(P.integrate(0, -1), I(2) - I(3))
        assert_allclose(P.integrate(-9, -10), I(2) - I(3))
        assert_allclose(P.integrate(0, -10), I(2) - I(3) - 3 * period_int)

    def test_roots(self):
        x = np.linspace(0, 1, 31)**2  # 生成 31 个平方的线性空间数组
        y = np.sin(30*x)  # 计算正弦函数在数组 x 上的值

        spl = splrep(x, y, s=0, k=3)  # 拟合 x, y 数据，生成样条曲线
        pp = PPoly.from_spline(spl)  # 从样条曲线对象中创建 PPoly 对象

        r = pp.roots()  # 计算 PPoly 对象的根
        r = r[(r >= 0 - 1e-15) & (r <= 1 + 1e-15)]  # 过滤掉在 [0, 1] 区间之外的根

        assert_allclose(r, sproot(spl), atol=1e-15)  # 检查计算得到的根是否与样条曲线的根非常接近

    def test_roots_idzero(self):
        # Roots for piecewise polynomials with identically zero
        # sections.
        c = np.array([[-1, 0.25], [0, 0], [-1, 0.25]]).T  # 定义具有相同零段的分段多项式系数
        x = np.array([0, 0.4, 0.6, 1.0])  # 定义数组 x

        pp = PPoly(c, x)  # 创建 PPoly 对象
        assert_array_equal(pp.roots(),
                           [0.25, 0.4, np.nan, 0.6 + 0.25])  # 检查计算得到的根是否与预期值一致

        const = 2.  # 定义常数值
        c1 = c.copy()
        c1[1, :] += const  # 将分段多项式的一个段加上常数值
        pp1 = PPoly(c1, x)  # 创建修改后的 PPoly 对象

        assert_array_equal(pp1.solve(const),
                           [0.25, 0.4, np.nan, 0.6 + 0.25])  # 检查求解等于常数值的根是否与预期值一致
    def test_roots_all_zero(self):
        # 测试多项式处处为零的情况
        c = [[0], [0]]  # 定义系数矩阵，表示多项式为零
        x = [0, 1]  # 定义分段点的位置
        p = PPoly(c, x)  # 创建分段多项式对象
        assert_array_equal(p.roots(), [0, np.nan])  # 检查多项式的根，期望得到[0, NaN]
        assert_array_equal(p.solve(0), [0, np.nan])  # 求解多项式在指定点的值，期望得到[0, NaN]
        assert_array_equal(p.solve(1), [])  # 求解多项式在另一指定点的值，期望得到空列表

        c = [[0, 0], [0, 0]]  # 定义另一个系数矩阵，表示多项式为零
        x = [0, 1, 2]  # 定义另一个分段点的位置
        p = PPoly(c, x)  # 创建另一个分段多项式对象
        assert_array_equal(p.roots(), [0, np.nan, 1, np.nan])  # 检查多项式的根，期望得到[0, NaN, 1, NaN]
        assert_array_equal(p.solve(0), [0, np.nan, 1, np.nan])  # 求解多项式在指定点的值，期望得到[0, NaN, 1, NaN]
        assert_array_equal(p.solve(1), [])  # 求解多项式在另一指定点的值，期望得到空列表

    def test_roots_repeated(self):
        # 检查在多个分段中重复出现的根只报告一次
        # [(x + 1)**2 - 1, -x**2] ; x == 0 是一个重复的根
        c = np.array([[1, 0, -1], [-1, 0, 0]]).T  # 定义系数矩阵
        x = np.array([-1, 0, 1])  # 定义分段点的位置

        pp = PPoly(c, x)  # 创建分段多项式对象
        assert_array_equal(pp.roots(), [-2, 0])  # 检查多项式的根，期望得到[-2, 0]
        assert_array_equal(pp.roots(extrapolate=False), [0])  # 关闭外推后，检查多项式的根，期望得到[0]

    def test_roots_discont(self):
        # 检查跨越零点的不连续性是否报告为根
        c = np.array([[1], [-1]]).T  # 定义系数矩阵
        x = np.array([0, 0.5, 1])  # 定义分段点的位置
        pp = PPoly(c, x)  # 创建分段多项式对象
        assert_array_equal(pp.roots(), [0.5])  # 检查多项式的根，期望得到[0.5]
        assert_array_equal(pp.roots(discontinuity=False), [])  # 不考虑不连续性时，检查多项式的根，期望得到空列表

        # 对于跨越 y 轴的不连续性，同样要检查是否作为根报告
        assert_array_equal(pp.solve(0.5), [0.5])  # 求解多项式在指定点的值，期望得到[0.5]
        assert_array_equal(pp.solve(0.5, discontinuity=False), [])  # 不考虑不连续性时，求解多项式在指定点的值，期望得到空列表

        assert_array_equal(pp.solve(1.5), [])  # 求解多项式在另一指定点的值，期望得到空列表
        assert_array_equal(pp.solve(1.5, discontinuity=False), [])  # 不考虑不连续性时，求解多项式在另一指定点的值，期望得到空列表

    def test_roots_random(self):
        # 检查具有随机系数的高阶多项式
        np.random.seed(1234)

        num = 0  # 初始化计数器

        for extrapolate in (True, False):
            for order in range(0, 20):
                x = np.unique(np.r_[0, 10 * np.random.rand(30), 10])  # 生成随机的分段点位置
                c = 2*np.random.rand(order+1, len(x)-1, 2, 3) - 1  # 生成随机的系数矩阵

                pp = PPoly(c, x)  # 创建分段多项式对象
                for y in [0, np.random.random()]:
                    r = pp.solve(y, discontinuity=False, extrapolate=extrapolate)  # 求解多项式在指定点的值

                    for i in range(2):
                        for j in range(3):
                            rr = r[i,j]
                            if rr.size > 0:
                                # 检查报告的根是否确实是根
                                num += rr.size
                                val = pp(rr, extrapolate=extrapolate)[:,i,j]
                                cmpval = pp(rr, nu=1, extrapolate=extrapolate)[:,i,j]
                                msg = f"({extrapolate!r}) r = {repr(rr)}"
                                assert_allclose((val-y) / cmpval, 0, atol=1e-7, err_msg=msg)

        # 检查是否检查了足够数量的根
        assert_(num > 100, repr(num))
    def test_roots_croots(self):
        # 测试复数根查找算法
        np.random.seed(1234)

        # 循环测试不同多项式次数
        for k in range(1, 15):
            # 生成随机系数矩阵
            c = np.random.rand(k, 1, 130)

            if k == 3:
                # 添加一个零判别式的情况
                c[:,0,0] = 1, 2, 1

            # 遍历测试不同的 y 值
            for y in [0, np.random.random()]:
                # 创建一个复数类型的空数组 w
                w = np.empty(c.shape, dtype=complex)
                # 调用 _croots_poly1 函数计算根
                _ppoly._croots_poly1(c, w)

                if k == 1:
                    # 对于 k 等于 1 的情况，所有值应为 NaN
                    assert_(np.isnan(w).all())
                    continue

                # 初始化结果变量
                res = 0
                cres = 0
                # 计算多项式在 w 处的值及其绝对值
                for i in range(k):
                    res += c[i,None] * w**(k-1-i)
                    cres += abs(c[i,None] * w**(k-1-i))
                # 处理可能的除零错误
                with np.errstate(invalid='ignore'):
                    res /= cres
                # 将结果展平并去除 NaN 值
                res = res.ravel()
                res = res[~np.isnan(res)]
                # 断言结果应接近于 0，允许的绝对误差为 1e-10
                assert_allclose(res, 0, atol=1e-10)

    def test_extrapolate_attr(self):
        # 测试 PPoly 对象的 extrapolate 属性

        # 定义一个多项式系数矩阵 c 和定义域 x
        c = np.array([[-1, 0, 1]]).T
        x = np.array([0, 1])

        # 循环测试不同的 extrapolate 值
        for extrapolate in [True, False, None]:
            # 创建一个 PPoly 对象 pp
            pp = PPoly(c, x, extrapolate=extrapolate)
            # 计算 pp 的导数和积分
            pp_d = pp.derivative()
            pp_i = pp.antiderivative()

            if extrapolate is False:
                # 断言在 extrapolate 为 False 时，pp 在特定点的值为 NaN
                assert_(np.isnan(pp([-0.1, 1.1])).all())
                assert_(np.isnan(pp_i([-0.1, 1.1])).all())
                assert_(np.isnan(pp_d([-0.1, 1.1])).all())
                # 断言 pp 的根为 [1]
                assert_equal(pp.roots(), [1])
            else:
                # 断言在其它情况下，pp 在特定点的值接近于预期值
                assert_allclose(pp([-0.1, 1.1]), [1-0.1**2, 1-1.1**2])
                # 断言 pp_i 和 pp_d 的值不包含 NaN
                assert_(not np.isnan(pp_i([-0.1, 1.1])).any())
                assert_(not np.isnan(pp_d([-0.1, 1.1])).any())
                # 断言 pp 的根为 [1, -1]
                assert_allclose(pp.roots(), [1, -1])
class TestBPoly:
    # 定义测试类 TestBPoly
    def test_simple(self):
        # 定义简单测试方法 test_simple
        x = [0, 1]
        # 创建变量 x，赋值为列表 [0, 1]
        c = [[3]]
        # 创建变量 c，赋值为包含列表的列表 [[3]]
        bp = BPoly(c, x)
        # 创建 BPoly 对象 bp，使用 c 和 x 初始化
        assert_allclose(bp(0.1), 3.)
        # 断言 bp 在 0.1 处的值接近于 3.0

    def test_simple2(self):
        # 定义简单测试方法 test_simple2
        x = [0, 1]
        # 创建变量 x，赋值为列表 [0, 1]
        c = [[3], [1]]
        # 创建变量 c，赋值为包含两个列表的列表 [[3], [1]]
        bp = BPoly(c, x)   # 3*(1-x) + 1*x
        # 创建 BPoly 对象 bp，使用 c 和 x 初始化，表示多项式 3*(1-x) + 1*x
        assert_allclose(bp(0.1), 3*0.9 + 1.*0.1)
        # 断言 bp 在 0.1 处的值接近于 3*0.9 + 1*0.1

    def test_simple3(self):
        # 定义简单测试方法 test_simple3
        x = [0, 1]
        # 创建变量 x，赋值为列表 [0, 1]
        c = [[3], [1], [4]]
        # 创建变量 c，赋值为包含三个列表的列表 [[3], [1], [4]]
        bp = BPoly(c, x)   # 3 * (1-x)**2 + 2 * x (1-x) + 4 * x**2
        # 创建 BPoly 对象 bp，使用 c 和 x 初始化，表示多项式 3 * (1-x)**2 + 2 * x (1-x) + 4 * x**2
        assert_allclose(bp(0.2),
                3 * 0.8*0.8 + 1 * 2*0.2*0.8 + 4 * 0.2*0.2)
        # 断言 bp 在 0.2 处的值接近于计算结果

    def test_simple4(self):
        # 定义简单测试方法 test_simple4
        x = [0, 1]
        # 创建变量 x，赋值为列表 [0, 1]
        c = [[1], [1], [1], [2]]
        # 创建变量 c，赋值为包含四个列表的列表 [[1], [1], [1], [2]]
        bp = BPoly(c, x)
        # 创建 BPoly 对象 bp，使用 c 和 x 初始化
        assert_allclose(bp(0.3), 0.7**3 +
                                 3 * 0.7**2 * 0.3 +
                                 3 * 0.7 * 0.3**2 +
                             2 * 0.3**3)
        # 断言 bp 在 0.3 处的值接近于计算结果

    def test_simple5(self):
        # 定义简单测试方法 test_simple5
        x = [0, 1]
        # 创建变量 x，赋值为列表 [0, 1]
        c = [[1], [1], [8], [2], [1]]
        # 创建变量 c，赋值为包含五个列表的列表 [[1], [1], [8], [2], [1]]
        bp = BPoly(c, x)
        # 创建 BPoly 对象 bp，使用 c 和 x 初始化
        assert_allclose(bp(0.3), 0.7**4 +
                                 4 * 0.7**3 * 0.3 +
                             8 * 6 * 0.7**2 * 0.3**2 +
                             2 * 4 * 0.7 * 0.3**3 +
                                 0.3**4)
        # 断言 bp 在 0.3 处的值接近于计算结果

    def test_periodic(self):
        # 定义周期性测试方法 test_periodic
        x = [0, 1, 3]
        # 创建变量 x，赋值为列表 [0, 1, 3]
        c = [[3, 0], [0, 0], [0, 2]]
        # 创建变量 c，赋值为包含三个列表的列表 [[3, 0], [0, 0], [0, 2]]
        # [3*(1-x)**2, 2*((x-1)/2)**2]
        # 表示多项式列表，每个子列表对应于不同区间的多项式表达式
        bp = BPoly(c, x, extrapolate='periodic')
        # 创建 BPoly 对象 bp，使用 c, x 和 extrapolate='periodic' 初始化
        assert_allclose(bp(3.4), 3 * 0.6**2)
        # 断言 bp 在 3.4 处的值接近于计算结果
        assert_allclose(bp(-1.3), 2 * (0.7/2)**2)
        # 断言 bp 在 -1.3 处的值接近于计算结果
        assert_allclose(bp(3.4, 1), -6 * 0.6)
        # 断言 bp 在 3.4 处导数为 -6 * 0.6
        assert_allclose(bp(-1.3, 1), 2 * (0.7/2))
        # 断言 bp 在 -1.3 处导数为 2 * (0.7/2)

    def test_descending(self):
        # 定义降序测试方法 test_descending
        np.random.seed(0)
        # 设定随机数种子为 0

        power = 3
        # 创建变量 power，赋值为 3
        for m in [10, 20, 30]:
            # 遍历列表 [10, 20, 30]
            x = np.sort(np.random.uniform(0, 10, m + 1))
            # 创建变量 x，赋值为 np.random.uniform(0, 10, m + 1) 的排序结果
            ca = np.random.uniform(-0.1, 0.1, size=(power + 1, m))
            # 创建变量 ca，赋值为 np.random.uniform(-0.1, 0.1, size=(power + 1, m))
            # 其中 power + 1 行 m 列的数组
            # We need only to flip coefficients to get it right!
            # 只需翻转系数即可获得正确结果！
            cd = ca[::-1].copy()

            pa = BPoly(ca, x, extrapolate=True)
            # 创建 BPoly 对象 pa，使用 ca, x 和 extrapolate=True 初始化
            pd = BPoly(cd[:, ::-1], x[::-1], extrapolate=True)
            # 创建 BPoly 对象 pd，使用 cd 翻转后的数组，x 翻转后的数组，和 extrapolate=True 初始化

            x_test = np.random.uniform(-10, 20, 100)
            # 创建变量 x_test，赋值为 np.random.uniform(-10, 20, 100)
            assert_allclose(pa(x_test), pd(x_test), rtol=1e-13)
            # 断言 pa 和 pd 在 x_test 处的值接近，相对容差为 1e-13
            assert_allclose(pa(x_test, 1), pd(x_test, 1), rtol=1e-13)
            # 断言 pa 和 pd 在 x_test 处的一阶导数值接近，相对容差为 1e-13

            pa_d = pa.derivative()
            # 创建变量 pa_d，赋值为 pa 的一阶导数对象
            pd_d = pd.derivative()
            # 创建变量 pd_d，赋值为 pd 的一阶导数对象

            assert_allclose(pa_d(x_test), pd_d(x_test), rtol=1e-13)
            # 断言 pa_d 和 pd_d 在 x_test 处的值接近，相对容差为 1e-13

            # Antiderivatives won't be equal because fixing continuity is
            # done in the reverse order, but surely the differences should be
            # equal.
            # 反导数不相等，因为修复连续性是以相反的顺序进行的，但差异应该
    # 定义一个测试方法，用于测试多维多项式对象的形状处理
    def test_multi_shape(self):
        # 创建一个形状为 (6, 2, 1, 2, 3) 的随机数组
        c = np.random.rand(6, 2, 1, 2, 3)
        # 创建一个包含 [0, 0.5, 1] 的数组 x
        x = np.array([0, 0.5, 1])
        # 使用 c 和 x 创建一个 BPoly 对象 p
        p = BPoly(c, x)
        # 断言 p.x 的形状与 x 的形状相同
        assert_equal(p.x.shape, x.shape)
        # 断言 p.c 的形状与 c 的形状相同
        assert_equal(p.c.shape, c.shape)
        # 断言 p(0.3) 的输出形状与 c 的最后三个维度相同
        assert_equal(p(0.3).shape, c.shape[2:])
        # 断言 p(np.random.rand(5,6)) 的输出形状为 (5,6) 加上 c 的最后三个维度
        assert_equal(p(np.random.rand(5,6)).shape,
                     (5,6)+c.shape[2:])

        # 计算 p 的导数 dp
        dp = p.derivative()
        # 断言 dp.c 的形状为 (5, 2, 1, 2, 3)

    # 定义一个测试方法，用于测试多项式对象在给定区间长度内的表现
    def test_interval_length(self):
        # 定义一个包含两个元素的列表 x
        x = [0, 2]
        # 定义一个包含三个子列表的列表 c
        c = [[3], [1], [4]]
        # 使用 c 和 x 创建一个 BPoly 对象 bp
        bp = BPoly(c, x)
        # 定义一个 xval = 0.1
        xval = 0.1
        # 计算 s = (x - xa) / (xb - xa)，其中 s 是一个比例值
        s = xval / 2  # s = (x - xa) / (xb - xa)
        # 断言 bp(xval) 的值与指定的多项式表达式结果相近

    # 定义一个测试方法，用于测试在两个区间上的多项式对象的行为
    def test_two_intervals(self):
        # 定义一个包含三个元素的列表 x
        x = [0, 1, 3]
        # 定义一个包含三个子列表的列表 c
        c = [[3, 0], [0, 0], [0, 2]]
        # 使用 c 和 x 创建一个 BPoly 对象 bp
        bp = BPoly(c, x)
        # 断言 bp(0.4) 的值与指定的多项式表达式结果相近
        assert_allclose(bp(0.4), 3 * 0.6*0.6)
        # 断言 bp(1.7) 的值与指定的多项式表达式结果相近

    # 定义一个测试方法，用于测试在外推属性设置下多项式对象的行为
    def test_extrapolate_attr(self):
        # 定义一个包含两个元素的列表 x
        x = [0, 2]
        # 定义一个包含三个子列表的列表 c
        c = [[3], [1], [4]]
        # 使用 c 和 x 创建一个 BPoly 对象 bp
        bp = BPoly(c, x)

        # 遍历不同的外推设置（True、False、None）
        for extrapolate in (True, False, None):
            # 根据当前的外推设置创建一个新的 BPoly 对象 bp
            bp = BPoly(c, x, extrapolate=extrapolate)
            # 计算 bp 的导数 bp_d
            bp_d = bp.derivative()
            # 如果外推设置为 False，则断言在 [-0.1, 2.1] 区间上 bp 和 bp_d 的值都为 NaN
            if extrapolate is False:
                assert_(np.isnan(bp([-0.1, 2.1])).all())
                assert_(np.isnan(bp_d([-0.1, 2.1])).all())
            # 否则，断言在 [-0.1, 2.1] 区间上 bp 和 bp_d 的值都不为 NaN
            else:
                assert_(not np.isnan(bp([-0.1, 2.1])).any())
                assert_(not np.isnan(bp_d([-0.1, 2.1])).any())
class TestBPolyCalculus:
    # 测试 BPoly 类的计算功能

    def test_derivative(self):
        # 测试导数计算功能
        x = [0, 1, 3]
        c = [[3, 0], [0, 0], [0, 2]]
        bp = BPoly(c, x)  # 使用系数 c 和节点 x 创建 BPoly 对象
        bp_der = bp.derivative()  # 计算 BPoly 对象的一阶导数
        assert_allclose(bp_der(0.4), -6*(0.6))  # 验证导数在指定点的计算结果
        assert_allclose(bp_der(1.7), 0.7)

        # derivatives in-place
        # 验证原地导数计算
        assert_allclose([bp(0.4, nu=1), bp(0.4, nu=2), bp(0.4, nu=3)],
                        [-6*(1-0.4), 6., 0.])  # 验证 BPoly 对象在不同阶导数下的计算结果
        assert_allclose([bp(1.7, nu=1), bp(1.7, nu=2), bp(1.7, nu=3)],
                        [0.7, 1., 0])

    def test_derivative_ppoly(self):
        # 确保与幂基底的一致性
        np.random.seed(1234)
        m, k = 5, 8   # 区间数，阶数
        x = np.sort(np.random.random(m))
        c = np.random.random((k, m-1))
        bp = BPoly(c, x)  # 使用系数 c 和节点 x 创建 BPoly 对象
        pp = PPoly.from_bernstein_basis(bp)  # 从伯恩斯坦基底创建 PPoly 对象

        for d in range(k):
            bp = bp.derivative()  # 计算 BPoly 对象的阶数为 d 的导数
            pp = pp.derivative()  # 计算 PPoly 对象的阶数为 d 的导数
            xp = np.linspace(x[0], x[-1], 21)
            assert_allclose(bp(xp), pp(xp))  # 验证在指定区间内的计算结果的一致性

    def test_deriv_inplace(self):
        np.random.seed(1234)
        m, k = 5, 8   # 区间数，阶数
        x = np.sort(np.random.random(m))
        c = np.random.random((k, m-1))

        # 测试实部和虚部系数
        for cc in [c.copy(), c*(1. + 2.j)]:
            bp = BPoly(cc, x)  # 使用系数 cc 和节点 x 创建 BPoly 对象
            xp = np.linspace(x[0], x[-1], 21)
            for i in range(k):
                assert_allclose(bp(xp, i), bp.derivative(i)(xp))  # 验证 BPoly 对象在指定阶数下的导数计算结果

    def test_antiderivative_simple(self):
        # f(x) = x        for x \in [0, 1),
        #        (x-1)/2  for x \in [1, 3]
        #
        # antiderivative is then
        # F(x) = x**2 / 2            for x \in [0, 1),
        #        0.5*x*(x/2 - 1) + A  for x \in [1, 3]
        # where A = 3/4 for continuity at x = 1.
        x = [0, 1, 3]
        c = [[0, 0], [1, 1]]

        bp = BPoly(c, x)  # 使用系数 c 和节点 x 创建 BPoly 对象
        bi = bp.antiderivative()  # 计算 BPoly 对象的不定积分

        xx = np.linspace(0, 3, 11)
        assert_allclose(bi(xx),
                        np.where(xx < 1, xx**2 / 2.,
                                         0.5 * xx * (xx/2. - 1) + 3./4),
                        atol=1e-12, rtol=1e-12)  # 验证不定积分的计算结果

    def test_der_antider(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10, 2, 3))
        bp = BPoly(c, x)  # 使用系数 c 和节点 x 创建 BPoly 对象

        xx = np.linspace(x[0], x[-1], 100)
        assert_allclose(bp.antiderivative().derivative()(xx),
                        bp(xx), atol=1e-12, rtol=1e-12)  # 验证 BPoly 对象的反导数和导数的一致性

    def test_antider_ppoly(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10, 2, 3))
        bp = BPoly(c, x)  # 使用系数 c 和节点 x 创建 BPoly 对象
        pp = PPoly.from_bernstein_basis(bp)  # 从伯恩斯坦基底创建 PPoly 对象

        xx = np.linspace(x[0], x[-1], 10)

        assert_allclose(bp.antiderivative(2)(xx),
                        pp.antiderivative(2)(xx), atol=1e-12, rtol=1e-12)  # 验证在指定阶数下的不定积分计算结果的一致性
    # 定义测试函数，验证连续反导函数的行为
    def test_antider_continuous(self):
        # 设置随机数种子以确保可重复性
        np.random.seed(1234)
        # 创建一个有序的随机数组，并进行排序
        x = np.sort(np.random.random(11))
        # 创建一个随机的系数矩阵
        c = np.random.random((4, 10))
        # 使用系数矩阵和节点向量创建 BPoly 对象，然后计算其反导函数
        bp = BPoly(c, x).antiderivative()

        # 从反导函数中间选取一段进行数值比较
        xx = bp.x[1:-1]
        # 断言反导函数在微小偏移后的值相等，用于验证数值稳定性
        assert_allclose(bp(xx - 1e-14),
                        bp(xx + 1e-14), atol=1e-12, rtol=1e-12)

    # 定义测试函数，验证积分函数的行为
    def test_integrate(self):
        # 设置随机数种子以确保可重复性
        np.random.seed(1234)
        # 创建一个有序的随机数组，并进行排序
        x = np.sort(np.random.random(11))
        # 创建一个随机的系数矩阵
        c = np.random.random((4, 10))
        # 使用系数矩阵和节点向量创建 BPoly 对象
        bp = BPoly(c, x)
        # 将 BPoly 对象转换为 PPoly 对象，从 Bernstein 基函数转换
        pp = PPoly.from_bernstein_basis(bp)
        # 断言 BPoly 和 PPoly 对象在区间 [0, 1] 上的积分结果相等，用于验证数值稳定性
        assert_allclose(bp.integrate(0, 1),
                        pp.integrate(0, 1), atol=1e-12, rtol=1e-12)

    # 定义测试函数，验证带外推功能的积分函数行为
    def test_integrate_extrap(self):
        # 创建一个简单的系数矩阵和节点向量
        c = [[1]]
        x = [0, 1]
        # 使用系数矩阵和节点向量创建 BPoly 对象
        b = BPoly(c, x)

        # 默认情况下 extrapolate=True，验证积分结果
        assert_allclose(b.integrate(0, 2), 2., atol=1e-14)

        # 使用 .integrate 方法的参数覆盖对象的 extrapolate 属性，验证积分结果
        b1 = BPoly(c, x, extrapolate=False)
        assert_(np.isnan(b1.integrate(0, 2)))
        assert_allclose(b1.integrate(0, 2, extrapolate=True), 2., atol=1e-14)

    # 定义测试函数，验证周期性积分函数的行为
    def test_integrate_periodic(self):
        # 创建特定的节点向量和系数矩阵
        x = np.array([1, 2, 4])
        c = np.array([[0., 0.], [-1., -1.], [2., -0.], [1., 2.]])

        # 使用节点向量和系数矩阵创建 PPoly 对象，再转换为 BPoly 对象，并设置周期性外推
        P = BPoly.from_power_basis(PPoly(c, x), extrapolate='periodic')
        # 计算 BPoly 对象的反导函数
        I = P.antiderivative()

        # 计算特定区间上的周期性积分，并与期望值进行比较，用于验证周期性积分结果
        period_int = I(4) - I(1)
        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)

        assert_allclose(P.integrate(1.5, 2.5), I(2.5) - I(1.5))
        assert_allclose(P.integrate(3.5, 5), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12),
                        I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5, 5 + 12),
                        I(2) - I(1) + I(4) - I(3.5) + 4 * period_int)

        assert_allclose(P.integrate(0, -1), I(2) - I(3))
        assert_allclose(P.integrate(-9, -10), I(2) - I(3))
        assert_allclose(P.integrate(0, -10), I(2) - I(3) - 3 * period_int)

    # 定义测试函数，验证负导数函数的行为
    def test_antider_neg(self):
        # 创建一个简单的系数矩阵和节点向量
        c = [[1]]
        x = [0, 1]
        # 使用系数矩阵和节点向量创建 BPoly 对象
        b = BPoly(c, x)

        # 在给定的区间内生成均匀分布的点
        xx = np.linspace(0, 1, 21)

        # 断言导数函数的负值与反导函数的值相等，用于验证数值稳定性
        assert_allclose(b.derivative(-1)(xx), b.antiderivative()(xx),
                        atol=1e-12, rtol=1e-12)
        assert_allclose(b.derivative(1)(xx), b.antiderivative(-1)(xx),
                        atol=1e-12, rtol=1e-12)
class TestPolyConversions:
    # 多项式转换测试类

    def test_bp_from_pp(self):
        # 测试从幂基底到多项式基底的转换
        x = [0, 1, 3]
        c = [[3, 2], [1, 8], [4, 3]]
        pp = PPoly(c, x)  # 创建 PPoly 对象
        bp = BPoly.from_power_basis(pp)  # 从幂基底创建 BPoly 对象
        pp1 = PPoly.from_bernstein_basis(bp)  # 从伯恩斯坦基底创建 PPoly 对象

        xp = [0.1, 1.4]
        assert_allclose(pp(xp), bp(xp))  # 断言两种表示在指定点的值接近
        assert_allclose(pp(xp), pp1(xp))  # 断言两种表示在指定点的值接近

    def test_bp_from_pp_random(self):
        # 随机测试从幂基底到多项式基底的转换
        np.random.seed(1234)
        m, k = 5, 8   # 区间数和阶数
        x = np.sort(np.random.random(m))  # 随机生成排序的节点
        c = np.random.random((k, m-1))  # 随机生成系数
        pp = PPoly(c, x)  # 创建 PPoly 对象
        bp = BPoly.from_power_basis(pp)  # 从幂基底创建 BPoly 对象
        pp1 = PPoly.from_bernstein_basis(bp)  # 从伯恩斯坦基底创建 PPoly 对象

        xp = np.linspace(x[0], x[-1], 21)  # 生成均匀间隔的测试点
        assert_allclose(pp(xp), bp(xp))  # 断言两种表示在指定点的值接近
        assert_allclose(pp(xp), pp1(xp))  # 断言两种表示在指定点的值接近

    def test_pp_from_bp(self):
        # 测试从多项式基底到幂基底的转换
        x = [0, 1, 3]
        c = [[3, 3], [1, 1], [4, 2]]
        bp = BPoly(c, x)  # 创建 BPoly 对象
        pp = PPoly.from_bernstein_basis(bp)  # 从伯恩斯坦基底创建 PPoly 对象
        bp1 = BPoly.from_power_basis(pp)  # 从幂基底创建 BPoly 对象

        xp = [0.1, 1.4]
        assert_allclose(bp(xp), pp(xp))  # 断言两种表示在指定点的值接近
        assert_allclose(bp(xp), bp1(xp))  # 断言两种表示在指定点的值接近

    def test_broken_conversions(self):
        # 错误的转换测试，检查是否会引发 TypeError
        x = [0, 1, 3]
        c = [[3, 3], [1, 1], [4, 2]]
        pp = PPoly(c, x)  # 创建 PPoly 对象
        with assert_raises(TypeError):
            PPoly.from_bernstein_basis(pp)  # 期望引发 TypeError

        bp = BPoly(c, x)  # 创建 BPoly 对象
        with assert_raises(TypeError):
            BPoly.from_power_basis(bp)  # 期望引发 TypeError


class TestBPolyFromDerivatives:
    # 从导数构造 BPoly 测试类

    def test_make_poly_1(self):
        # 测试构造一阶多项式
        c1 = BPoly._construct_from_derivatives(0, 1, [2], [3])
        assert_allclose(c1, [2., 3.])  # 断言系数接近预期值

    def test_make_poly_2(self):
        # 测试构造多阶多项式
        c1 = BPoly._construct_from_derivatives(0, 1, [1, 0], [1])
        assert_allclose(c1, [1., 1., 1.])  # 断言系数接近预期值

        # f'(0) = 3
        c2 = BPoly._construct_from_derivatives(0, 1, [2, 3], [1])
        assert_allclose(c2, [2., 7./2, 1.])  # 断言系数接近预期值

        # f'(1) = 3
        c3 = BPoly._construct_from_derivatives(0, 1, [2], [1, 3])
        assert_allclose(c3, [2., -0.5, 1.])  # 断言系数接近预期值

    def test_make_poly_3(self):
        # 测试构造不同阶数的多项式
        # f'(0)=2, f''(0)=3
        c1 = BPoly._construct_from_derivatives(0, 1, [1, 2, 3], [4])
        assert_allclose(c1, [1., 5./3, 17./6, 4.])  # 断言系数接近预期值

        # f'(1)=2, f''(1)=3
        c2 = BPoly._construct_from_derivatives(0, 1, [1], [4, 2, 3])
        assert_allclose(c2, [1., 19./6, 10./3, 4.])  # 断言系数接近预期值

        # f'(0)=2, f'(1)=3
        c3 = BPoly._construct_from_derivatives(0, 1, [1, 2], [4, 3])
        assert_allclose(c3, [1., 5./3, 3., 4.])  # 断言系数接近预期值

    def test_make_poly_12(self):
        # 测试通过导数构造多项式的连续性
        np.random.seed(12345)
        ya = np.r_[0, np.random.random(5)]
        yb = np.r_[0, np.random.random(5)]

        c = BPoly._construct_from_derivatives(0, 1, ya, yb)
        pp = BPoly(c[:, None], [0, 1])
        for j in range(6):
            assert_allclose([pp(0.), pp(1.)], [ya[j], yb[j]])  # 断言多项式在端点处的值符合预期
            pp = pp.derivative()  # 求多项式的导数
    # 定义一个测试方法，用于测试提升多项式的次数
    def test_raise_degree(self):
        # 设置随机数种子，确保可重复性
        np.random.seed(12345)
        # 初始化变量 x
        x = [0, 1]
        # 设定变量 k 和 d
        k, d = 8, 5
        # 创建一个随机数数组 c，表示多项式系数
        c = np.random.random((k, 1, 2, 3, 4))
        # 使用多项式系数 c 和节点 x 创建 BPoly 对象 bp
        bp = BPoly(c, x)

        # 调用 BPoly 类方法 _raise_degree，提升多项式系数 c 的次数为 d
        c1 = BPoly._raise_degree(c, d)
        # 使用提升次数后的多项式系数 c1 和节点 x 创建新的 BPoly 对象 bp1
        bp1 = BPoly(c1, x)

        # 生成一个从 0 到 1 等间距的 11 个点的数组 xp
        xp = np.linspace(0, 1, 11)
        # 断言两个多项式对象 bp 和 bp1 在 xp 上的值非常接近
        assert_allclose(bp(xp), bp1(xp))

    # 定义一个测试方法，用于测试给定参数 xi 和 yi 不合法的情况
    def test_xi_yi(self):
        # 断言调用 BPoly 类方法 from_derivatives 时会抛出 ValueError 异常
        assert_raises(ValueError, BPoly.from_derivatives, [0, 1], [0])

    # 定义一个测试方法，用于测试给定参数 xi 和 yi 的顺序不正确的情况
    def test_coords_order(self):
        # 初始化 xi 和 yi，yi 是一个二维数组
        xi = [0, 0, 1]
        yi = [[0], [0], [0]]
        # 断言调用 BPoly 类方法 from_derivatives 时会抛出 ValueError 异常
        assert_raises(ValueError, BPoly.from_derivatives, xi, yi)

    # 定义一个测试方法，用于测试多项式的一些特殊情况
    def test_zeros(self):
        # 初始化 xi 和 yi
        xi = [0, 1, 2, 3]
        yi = [[0, 0], [0], [0, 0], [0, 0]]  # 注意：需要提升多项式的次数
        # 调用 BPoly 类方法 from_derivatives 创建 pp 对象
        pp = BPoly.from_derivatives(xi, yi)
        # 断言 pp 的系数矩阵形状为 (4, 3)
        assert_(pp.c.shape == (4, 3))

        # 对 pp 求导，并在几个指定点上断言其值非常接近 0
        ppd = pp.derivative()
        for xp in [0., 0.1, 1., 1.1, 1.9, 2., 2.5]:
            assert_allclose([pp(xp), ppd(xp)], [0., 0.])

    # 定义一个辅助方法，用于生成具有随机多项式导数的 xi 和 yi
    def _make_random_mk(self, m, k):
        # 设置随机数种子
        np.random.seed(1234)
        # 生成 xi 数组，包含 m+1 个点
        xi = np.asarray([1. * j**2 for j in range(m+1)])
        # 生成 yi 数组，每个元素都是包含 k 个随机数的数组
        yi = [np.random.random(k) for j in range(m+1)]
        return xi, yi

    # 定义一个测试方法，用于测试随机生成的 xi 和 yi 的情况
    def test_random_12(self):
        # 设定 m 和 k 的值
        m, k = 5, 12
        # 生成随机的 xi 和 yi
        xi, yi = self._make_random_mk(m, k)
        # 调用 BPoly 类方法 from_derivatives 创建 pp 对象
        pp = BPoly.from_derivatives(xi, yi)

        # 对每个导数次数进行循环检查
        for order in range(k//2):
            # 断言 pp 在 xi 上的值与 yi 中对应元素的值非常接近
            assert_allclose(pp(xi), [yy[order] for yy in yi])
            # 对 pp 求导，更新 pp 为其导数
            pp = pp.derivative()

    # 定义一个测试方法，用于测试指定 orders 参数为 0 的情况
    def test_order_zero(self):
        # 设定 m 和 k 的值
        m, k = 5, 12
        # 生成随机的 xi 和 yi
        xi, yi = self._make_random_mk(m, k)
        # 断言调用 BPoly 类方法 from_derivatives 时会抛出 ValueError 异常
        assert_raises(ValueError, BPoly.from_derivatives,
                      **dict(xi=xi, yi=yi, orders=0))

    # 定义一个测试方法，用于测试指定 orders 参数过高的情况
    def test_orders_too_high(self):
        # 设定 m 和 k 的值
        m, k = 5, 12
        # 生成随机的 xi 和 yi
        xi, yi = self._make_random_mk(m, k)

        # 调用 BPoly 类方法 from_derivatives，orders 参数为 2*k-1，预期成功
        BPoly.from_derivatives(xi, yi, orders=2*k-1)
        # 断言调用 BPoly 类方法 from_derivatives 时会抛出 ValueError 异常
        assert_raises(ValueError, BPoly.from_derivatives,
                      **dict(xi=xi, yi=yi, orders=2*k))

    # 定义一个测试方法，用于测试指定 orders 参数的全局影响
    def test_orders_global(self):
        # 设定 m 和 k 的值
        m, k = 5, 12
        # 生成随机的 xi 和 yi
        xi, yi = self._make_random_mk(m, k)

        # 设定 order 变量为 5
        order = 5
        # 调用 BPoly 类方法 from_derivatives 创建 pp 对象，orders 参数为 order
        pp = BPoly.from_derivatives(xi, yi, orders=order)

        # 对每个阶数的一半加 1 进行循环检查
        for j in range(order//2+1):
            # 断言 pp 在 xi[1:-1] - 1e-12 和 xi[1:-1] + 1e-12 处的值非常接近
            assert_allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12))
            # 对 pp 求导，更新 pp 为其导数
            pp = pp.derivative()
        # 断言 pp 在 xi[1:-1] - 1e-12 和 xi[1:-1] + 1e-12 处的值不完全相等
        assert_(not np.allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)))

        # 现在将 order 设置为偶数
        order = 6
        # 调用 BPoly 类方法 from_derivatives 创建 pp 对象，orders 参数为 order
        pp = BPoly.from_derivatives(xi, yi, orders=order)
        # 对每个阶数的一半进行循环检查
        for j in range(order//2):
            # 断言 pp 在 xi[1:-1] - 1e-12 和 xi[1:-1] + 1e-12 处的值非常接近
            assert_allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12))
            # 对 pp 求导，更新 pp 为其导数
            pp = pp.derivative()
        # 断言 pp 在 xi[1:-1] - 1e-12 和 xi[1:-1] + 1e-12 处的值不完全相等
        assert_(not np.allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)))
    def test_orders_local(self):
        # 设置测试参数 m 和 k
        m, k = 7, 12
        # 使用 _make_random_mk 方法生成随机数列 xi 和 yi
        xi, yi = self._make_random_mk(m, k)

        # 创建阶数列表 orders，范围从 1 到 m+1
        orders = [o + 1 for o in range(m)]
        # 遍历 xi 中的元素，从第二个到倒数第二个
        for i, x in enumerate(xi[1:-1]):
            # 使用 BPoly 类从导数构造多项式 pp
            pp = BPoly.from_derivatives(xi, yi, orders=orders)
            # 对 pp 进行阶数 orders[i] // 2 + 1 次导数操作
            for j in range(orders[i] // 2 + 1):
                # 断言 pp 在 x-1e-12 和 x+1e-12 处的值接近
                assert_allclose(pp(x - 1e-12), pp(x + 1e-12))
                # 对 pp 求一次导数
                pp = pp.derivative()
            # 断言 pp 在 x-1e-12 和 x+1e-12 处的值不完全相等
            assert_(not np.allclose(pp(x - 1e-12), pp(x + 1e-12)))

    def test_yi_trailing_dims(self):
        # 设置测试参数 m 和 k
        m, k = 7, 5
        # 生成 m+1 个随机排列的 xi 数组
        xi = np.sort(np.random.random(m+1))
        # 生成形状为 (m+1, k, 6, 7, 8) 的随机数组 yi
        yi = np.random.random((m+1, k, 6, 7, 8))
        # 使用 BPoly 类从导数构造多项式 pp
        pp = BPoly.from_derivatives(xi, yi)
        # 断言 pp 的系数数组形状为 (2*k, m, 6, 7, 8)
        assert_equal(pp.c.shape, (2*k, m, 6, 7, 8))

    def test_gh_5430(self):
        # 至少其中一个会引发错误，除非 gh-5430 已修复。
        # 在 Python 2 中，int 是使用 C long 实现的，因此取决于系统。
        # 在 Python 3 中，只有一种任意精度整数类型，因此两者都应该失败。
        
        # 使用 np.int32 创建 orders 变量
        orders = np.int32(1)
        # 使用 BPoly 类从导数构造多项式 p
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        # 断言 p(0) 接近于 0
        assert_almost_equal(p(0), 0)
        
        # 使用 np.int64 创建 orders 变量
        orders = np.int64(1)
        # 使用 BPoly 类从导数构造多项式 p
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        # 断言 p(0) 接近于 0
        assert_almost_equal(p(0), 0)
        
        # 直接使用整数创建 orders 变量，之前是有效的，确保它仍然有效
        orders = 1
        # 使用 BPoly 类从导数构造多项式 p
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        # 断言 p(0) 接近于 0
        assert_almost_equal(p(0), 0)
        
        # 重新设置 orders 变量为 1
        orders = 1
class TestNdPPoly:
    # 定义测试类 TestNdPPoly，用于测试 NdPPoly 类的功能

    def test_simple_1d(self):
        # 测试一维情况下的简单情况

        np.random.seed(1234)
        # 设置随机种子为 1234

        c = np.random.rand(4, 5)
        # 生成一个 4x5 的随机数组，作为系数 c

        x = np.linspace(0, 1, 5+1)
        # 生成一个从 0 到 1 的等间距数组，长度为 5+1，作为变量 x

        xi = np.random.rand(200)
        # 生成一个包含 200 个随机数的数组，作为插值点 xi

        p = NdPPoly(c, (x,))
        # 使用 NdPPoly 类创建一个一维多项式 p，传入系数 c 和变量 x 的元组

        v1 = p((xi,))
        # 计算多项式 p 在插值点 xi 处的值，存储在 v1 中

        v2 = _ppoly_eval_1(c[:,:,None], x, xi).ravel()
        # 调用 _ppoly_eval_1 函数计算一维情况下的多项式在插值点 xi 处的值，返回一个扁平化的数组，存储在 v2 中
        assert_allclose(v1, v2)
        # 使用 assert_allclose 检查 v1 和 v2 是否近似相等

    def test_simple_2d(self):
        # 测试二维情况下的简单情况

        np.random.seed(1234)
        # 设置随机种子为 1234

        c = np.random.rand(4, 5, 6, 7)
        # 生成一个 4x5x6x7 的随机数组，作为系数 c

        x = np.linspace(0, 1, 6+1)
        # 生成一个从 0 到 1 的等间距数组，长度为 6+1，作为变量 x

        y = np.linspace(0, 1, 7+1)**2
        # 生成一个从 0 到 1 的等间距数组，长度为 7+1，每个元素求平方，作为变量 y

        xi = np.random.rand(200)
        # 生成一个包含 200 个随机数的数组，作为插值点 xi

        yi = np.random.rand(200)
        # 生成一个包含 200 个随机数的数组，作为插值点 yi

        v1 = np.empty([len(xi), 1], dtype=c.dtype)
        v1.fill(np.nan)
        # 创建一个形状为 (len(xi), 1) 的空数组 v1，并用 NaN 填充

        _ppoly.evaluate_nd(c.reshape(4*5, 6*7, 1),
                           (x, y),
                           np.array([4, 5], dtype=np.intc),
                           np.c_[xi, yi],
                           np.array([0, 0], dtype=np.intc),
                           1,
                           v1)
        # 调用 _ppoly.evaluate_nd 函数计算二维情况下的多项式在插值点 (xi, yi) 处的值，并存储在 v1 中

        v1 = v1.ravel()
        # 将 v1 扁平化处理

        v2 = _ppoly2d_eval(c, (x, y), xi, yi)
        # 调用 _ppoly2d_eval 函数计算二维情况下的多项式在插值点 (xi, yi) 处的值，存储在 v2 中

        assert_allclose(v1, v2)
        # 使用 assert_allclose 检查 v1 和 v2 是否近似相等

        p = NdPPoly(c, (x, y))
        # 使用 NdPPoly 类创建一个二维多项式 p，传入系数 c 和变量 x, y 的元组

        for nu in (None, (0, 0), (0, 1), (1, 0), (2, 3), (9, 2)):
            # 遍历不同的导数阶数 nu

            v1 = p(np.c_[xi, yi], nu=nu)
            # 计算多项式 p 在插值点 (xi, yi) 处的值，指定导数阶数 nu，存储在 v1 中

            v2 = _ppoly2d_eval(c, (x, y), xi, yi, nu=nu)
            # 调用 _ppoly2d_eval 函数计算二维情况下的多项式在插值点 (xi, yi) 处的值，指定导数阶数 nu，存储在 v2 中

            assert_allclose(v1, v2, err_msg=repr(nu))
            # 使用 assert_allclose 检查 v1 和 v2 是否近似相等，如果不等则输出错误信息 repr(nu)

    def test_simple_3d(self):
        # 测试三维情况下的简单情况

        np.random.seed(1234)
        # 设置随机种子为 1234

        c = np.random.rand(4, 5, 6, 7, 8, 9)
        # 生成一个 4x5x6x7x8x9 的随机数组，作为系数 c

        x = np.linspace(0, 1, 7+1)
        # 生成一个从 0 到 1 的等间距数组，长度为 7+1，作为变量 x

        y = np.linspace(0, 1, 8+1)**2
        # 生成一个从 0 到 1 的等间距数组，长度为 8+1，每个元素求平方，作为变量 y

        z = np.linspace(0, 1, 9+1)**3
        # 生成一个从 0 到 1 的等间距数组，长度为 9+1，每个元素求立方，作为变量 z

        xi = np.random.rand(40)
        # 生成一个包含 40 个随机数的数组，作为插值点 xi

        yi = np.random.rand(40)
        # 生成一个包含 40 个随机数的数组，作为插值点 yi

        zi = np.random.rand(40)
        # 生成一个包含 40 个随机数的数组，作为插值点 zi

        p = NdPPoly(c, (x, y, z))
        # 使用 NdPPoly 类创建一个三维多项式 p，传入系数 c 和变量 x, y, z 的元组

        for nu in (None, (0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 3, 0),
                   (6, 0, 2)):
            # 遍历不同的导数阶数 nu

            v1 = p((xi, yi, zi), nu=nu)
            # 计算多项式 p 在插值点 (xi, yi, zi) 处的值，指定导数阶数 nu，存储在 v1 中

            v2 = _ppoly3d_eval(c, (x, y, z), xi, yi, zi, nu=nu)
            # 调用 _ppoly3d_eval 函数计算三维情况下的多项式在插值点 (xi, yi, zi) 处的值，指定导数阶数 nu，存储在 v2 中

            assert_allclose(v1, v2, err_msg=repr(nu))
            # 使用 assert_allclose 检查 v1 和 v2 是否近似相等，如果不等则输出错误信息 repr(nu)

    def test_simple_4d(self):
        # 测试四维情况下的简单情况

        np.random.seed(1234)
        # 设置随机种子为 1234

        c = np.random.rand(4, 5, 6, 7, 8, 9, 10, 11)
        # 生成一个 4x5x6x7x8x9x10x11 的随机数组，作为系数 c

        x = np.linspace(0, 1, 8+1)
        #
    # 定义一个测试函数，用于测试三维多项式的导数计算功能
    def test_deriv_3d(self):
        np.random.seed(1234)

        # 创建一个随机数填充的数组，表示三维多项式的系数，维度为 (4, 5, 6, 7, 8, 9)
        c = np.random.rand(4, 5, 6, 7, 8, 9)
        # 在指定范围内生成均匀分布的数组，表示第一维的取值
        x = np.linspace(0, 1, 7+1)
        # 在指定范围内生成均匀分布的数组，并对每个元素进行平方，表示第二维的取值
        y = np.linspace(0, 1, 8+1)**2
        # 在指定范围内生成均匀分布的数组，并对每个元素进行立方，表示第三维的取值
        z = np.linspace(0, 1, 9+1)**3

        # 创建 NdPPoly 对象 p，传入系数 c 和对应的维度数组 (x, y, z)
        p = NdPPoly(c, (x, y, z))

        # differentiate vs x
        # 创建一个对 x 求导数的 NdPPoly 对象 p1，通过调整系数 c 的轴顺序来实现
        p1 = PPoly(c.transpose(0, 3, 1, 2, 4, 5), x)
        # 对 p 进行二阶导数计算，返回结果 dp
        dp = p.derivative(nu=[2])
        # 对 p1 进行二阶导数计算，返回结果 dp1
        dp1 = p1.derivative(2)
        # 使用 assert_allclose 函数比较 dp 和 dp1 的系数是否近似相等，通过轴顺序调整保持一致性
        assert_allclose(dp.c,
                        dp1.c.transpose(0, 2, 3, 1, 4, 5))

        # antidifferentiate vs y
        # 创建一个对 y 积分的 NdPPoly 对象 p1，通过调整系数 c 的轴顺序来实现
        p1 = PPoly(c.transpose(1, 4, 0, 2, 3, 5), y)
        # 对 p 进行指定轴方向的反导数计算，返回结果 dp
        dp = p.antiderivative(nu=[0, 1, 0])
        # 对 p1 进行一阶反导数计算，返回结果 dp1
        dp1 = p1.antiderivative(1)
        # 使用 assert_allclose 函数比较 dp 和 dp1 的系数是否近似相等，通过轴顺序调整保持一致性
        assert_allclose(dp.c,
                        dp1.c.transpose(2, 0, 3, 4, 1, 5))

        # differentiate vs z
        # 创建一个对 z 求导数的 NdPPoly 对象 p1，通过调整系数 c 的轴顺序来实现
        p1 = PPoly(c.transpose(2, 5, 0, 1, 3, 4), z)
        # 对 p 进行三阶导数计算，返回结果 dp
        dp = p.derivative(nu=[0, 0, 3])
        # 对 p1 进行三阶导数计算，返回结果 dp1
        dp1 = p1.derivative(3)
        # 使用 assert_allclose 函数比较 dp 和 dp1 的系数是否近似相等，通过轴顺序调整保持一致性
        assert_allclose(dp.c,
                        dp1.c.transpose(2, 3, 0, 4, 5, 1))

    # 定义一个测试函数，用于测试简单的三维多项式积分功能
    def test_deriv_3d_simple(self):
        # Integrate to obtain function x y**2 z**4 / (2! 4!)

        # 创建一个系数全为 1 的数组，表示简单的三维多项式，维度为 (1, 1, 1, 3, 4, 5)
        c = np.ones((1, 1, 1, 3, 4, 5))
        # 在指定范围内生成均匀分布的数组，并对每个元素进行一次方，表示第一维的取值
        x = np.linspace(0, 1, 3+1)**1
        # 在指定范围内生成均匀分布的数组，并对每个元素进行平方，表示第二维的取值
        y = np.linspace(0, 1, 4+1)**2
        # 在指定范围内生成均匀分布的数组，并对每个元素进行四次方，表示第三维的取值
        z = np.linspace(0, 1, 5+1)**3

        # 创建 NdPPoly 对象 p，传入系数 c 和对应的维度数组 (x, y, z)
        p = NdPPoly(c, (x, y, z))
        # 对 p 进行指定轴方向的反导数计算，积分指数为 (1, 0, 4)，返回结果 ip
        ip = p.antiderivative((1, 0, 4))
        # 对 ip 进行二次反导数计算，积分指数为 (0, 2, 0)，结果直接赋值给 ip
        ip = ip.antiderivative((0, 2, 0))

        # 创建随机数数组作为函数输入值的参数
        xi = np.random.rand(20)
        yi = np.random.rand(20)
        zi = np.random.rand(20)

        # 使用 assert_allclose 函数比较 ip 在给定参数上的计算结果与理论值 xi * yi**2 * zi**4 / (gamma(3)*gamma(5)) 是否近似相等
        assert_allclose(ip((xi, yi, zi)),
                        xi * yi**2 * zi**4 / (gamma(3)*gamma(5)))

    # 定义一个测试函数，用于测试二维多项式的积分功能
    def test_integrate_2d(self):
        np.random.seed(1234)
        # 创建一个随机数填充的数组，表示二维多项式的系数，维度为 (4, 5, 16, 17)
        c = np.random.rand(4, 5, 16, 17)
        # 在指定范围内生成均匀分布的数组，并对每个元素进行一次方，表示第一维的取值
        x = np.linspace(0, 1, 16+1)**1
        # 在指定范围内生成均匀分布的数组，并对每个元素进行平方，表示第二维的取值
        y = np.linspace(0, 1, 17+1)**2

        # 调整系数数组 c 的轴顺序，以使其连续可导，以便 nquad() 函数更容易处理
        c = c.transpose(0, 2, 1, 3)
        cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
        _ppoly.fix_continuity(cx, x, 2)
        c = cx.reshape(c.shape)
        c = c.transpose(0, 2, 1, 3)
        c = c.transpose(1, 3, 0, 2)
        cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
        _ppoly.fix_continuity(cx, y, 2)
        c = cx.reshape(c.shape)
        c = c.transpose(2, 0, 3, 1).copy()

        # 创建 NdPPoly 对象 p，传入系数 c 和对应的维度数组 (x, y)
        p = NdPPoly(c, (x, y))

        # 对给定区间列表 ranges 中的每个区间进行积分计算，并进行比较
        for ranges in [[(0, 1), (0, 1)],
                       [(0, 0.5), (0, 1)],
                       [(0, 1), (0, 0.5)],
                       [(0.
    # 定义一个测试方法，用于测试一维积分的功能
    def test_integrate_1d(self):
        # 设置随机种子确保结果可复现
        np.random.seed(1234)
        # 创建一个随机数组 c，形状为 (4, 5, 6, 16, 17, 18)
        c = np.random.rand(4, 5, 6, 16, 17, 18)
        # 生成一个包含 16+1 个元素的等差数列，并对每个元素取一次方
        x = np.linspace(0, 1, 16+1)**1
        # 生成一个包含 17+1 个元素的等差数列，并对每个元素取二次方
        y = np.linspace(0, 1, 17+1)**2
        # 生成一个包含 18+1 个元素的等差数列，并对每个元素取三次方
        z = np.linspace(0, 1, 18+1)**3

        # 创建 NdPPoly 对象 p，用随机数组 c 和 (x, y, z) 构造
        p = NdPPoly(c, (x, y, z))

        # 生成两个包含 200 个随机数的数组 u 和 v
        u = np.random.rand(200)
        v = np.random.rand(200)
        # 设置积分的上下限
        a, b = 0.2, 0.7

        # 对 p 进行沿 axis=0 的一维积分，得到函数 px
        px = p.integrate_1d(a, b, axis=0)
        # 计算 p 的 (1, 0, 0) 方向上的原函数，得到 pax
        pax = p.antiderivative((1, 0, 0))
        # 检查 px 在 (u, v) 处的值是否接近 pax 在 (b, u, v) 和 (a, u, v) 处的差
        assert_allclose(px((u, v)), pax((b, u, v)) - pax((a, u, v)))

        # 对 p 进行沿 axis=1 的一维积分，得到函数 py
        py = p.integrate_1d(a, b, axis=1)
        # 计算 p 的 (0, 1, 0) 方向上的原函数，得到 pay
        pay = p.antiderivative((0, 1, 0))
        # 检查 py 在 (u, v) 处的值是否接近 pay 在 (u, b, v) 和 (u, a, v) 处的差
        assert_allclose(py((u, v)), pay((u, b, v)) - pay((u, a, v)))

        # 对 p 进行沿 axis=2 的一维积分，得到函数 pz
        pz = p.integrate_1d(a, b, axis=2)
        # 计算 p 的 (0, 0, 1) 方向上的原函数，得到 paz
        paz = p.antiderivative((0, 0, 1))
        # 检查 pz 在 (u, v) 处的值是否接近 paz 在 (u, v, b) 和 (u, v, a) 处的差
        assert_allclose(pz((u, v)), paz((u, v, b)) - paz((u, v, a)))
# 评估手动分段多项式
def _ppoly_eval_1(c, x, xps):
    # 创建一个输出数组，其形状为(len(xps), c.shape[2])
    out = np.zeros((len(xps), c.shape[2]))
    # 遍历每个 xps 中的 xp
    for i, xp in enumerate(xps):
        # 如果 xp 小于 0 或大于 1，则将对应行设置为 NaN
        if xp < 0 or xp > 1:
            out[i,:] = np.nan
            continue
        # 查找 xp 在 x 中的位置索引 j
        j = np.searchsorted(x, xp) - 1
        # 断言确保 xp 在 x[j] 和 x[j+1] 之间
        assert_(x[j] <= xp < x[j+1])
        # 计算多项式的值 r，使用多项式系数 c[k,j] 和 d**(c.shape[0]-k-1)
        r = sum(c[k,j] * (xp - x[j])**(c.shape[0]-k-1) for k in range(c.shape[0]))
        # 将计算结果 r 存入输出数组的相应行
        out[i,:] = r
    # 返回计算结果数组
    return out


# 另一种方式评估手动分段多项式
def _ppoly_eval_2(coeffs, breaks, xnew, fill=np.nan):
    # 确定断点的起始和结束值
    a = breaks[0]
    b = breaks[-1]
    # 确定系数矩阵的维度
    K = coeffs.shape[0]

    # 保存输入 xnew 的形状，并将其展平为一维数组
    saveshape = np.shape(xnew)
    xnew = np.ravel(xnew)
    # 创建结果数组
    res = np.empty_like(xnew)
    # 创建布尔掩码，标记 xnew 是否在 [a, b] 范围内
    mask = (xnew >= a) & (xnew <= b)
    # 将不在范围内的位置填充为 fill
    res[~mask] = fill
    # 提取落在 [a, b] 范围内的值为 xx
    xx = xnew.compress(mask)
    # 对于每个 xx，找到它对应的断点索引
    indxs = np.searchsorted(breaks, xx) - 1
    indxs = indxs.clip(0, len(breaks))
    # 提取多项式系数
    pp = coeffs
    # 计算差值 diff
    diff = xx - breaks.take(indxs)
    # 创建 Vandermonde 矩阵 V
    V = np.vander(diff, N=K)
    # 计算多项式在每个 xx 处的值
    values = np.array([np.dot(V[k, :], pp[:, indxs[k]]) for k in range(len(xx))])
    # 将计算结果存入结果数组的对应位置
    res[mask] = values
    # 恢复结果数组的形状为原始 xnew 的形状
    res.shape = saveshape
    # 返回计算结果数组
    return res


def _dpow(x, y, n):
    """
    计算导数 d^n (x**y) / dx^n
    """
    # 如果 n 小于 0，则抛出异常
    if n < 0:
        raise ValueError("invalid derivative order")
    # 如果 n 大于 y，则返回 0
    elif n > y:
        return 0
    else:
        # 否则，计算导数并返回结果
        return poch(y - n + 1, n) * x**(y - n)


def _ppoly2d_eval(c, xs, xnew, ynew, nu=None):
    """
    直接评估二维分段多项式
    """
    # 如果 nu 未提供，则设为 (0, 0)
    if nu is None:
        nu = (0, 0)

    # 创建输出数组，其形状为 (len(xnew),)
    out = np.empty((len(xnew),), dtype=c.dtype)

    # 确定 c 的形状
    nx, ny = c.shape[:2]

    # 遍历 xnew 和 ynew 中的每一对 (x, y)
    for jout, (x, y) in enumerate(zip(xnew, ynew)):
        # 如果 x 或 y 不在对应的范围内，将输出设置为 NaN 并继续下一次循环
        if not ((xs[0][0] <= x <= xs[0][-1]) and
                (xs[1][0] <= y <= xs[1][-1])):
            out[jout] = np.nan
            continue

        # 查找 x 在 xs[0] 中的位置索引 j1，以及 y 在 xs[1] 中的位置索引 j2
        j1 = np.searchsorted(xs[0], x) - 1
        j2 = np.searchsorted(xs[1], y) - 1

        # 计算 s1 和 s2
        s1 = x - xs[0][j1]
        s2 = y - xs[1][j2]

        # 初始化值为 0
        val = 0

        # 遍历 c 中的每个 (k1, k2) 组合
        for k1 in range(c.shape[0]):
            for k2 in range(c.shape[1]):
                # 计算多项式的值，包括 c 的系数、s1 和 s2 的幂次以及 nu 的值
                val += (c[nx-k1-1, ny-k2-1, j1, j2]
                        * _dpow(s1, k1, nu[0])
                        * _dpow(s2, k2, nu[1]))

        # 将计算结果存入输出数组的相应位置
        out[jout] = val

    # 返回计算结果数组
    return out


def _ppoly3d_eval(c, xs, xnew, ynew, znew, nu=None):
    """
    直接评估三维分段多项式
    """
    # 如果 nu 未提供，则设为 (0, 0, 0)
    if nu is None:
        nu = (0, 0, 0)

    # 创建输出数组，其形状为 (len(xnew),)
    out = np.empty((len(xnew),), dtype=c.dtype)

    # 确定 c 的形状
    nx, ny, nz = c.shape[:3]
    # 遍历三个列表 xnew, ynew, znew 的元素及其对应的索引 jout
    for jout, (x, y, z) in enumerate(zip(xnew, ynew, znew)):
        # 检查 x, y, z 是否在各自维度的范围内
        if not ((xs[0][0] <= x <= xs[0][-1]) and
                (xs[1][0] <= y <= xs[1][-1]) and
                (xs[2][0] <= z <= xs[2][-1])):
            # 如果不在范围内，将 out 的当前索引 jout 设置为 NaN，并继续下一个循环
            out[jout] = np.nan
            continue

        # 在 xs 的每个维度上找到小于等于 x, y, z 的最大索引
        j1 = np.searchsorted(xs[0], x) - 1
        j2 = np.searchsorted(xs[1], y) - 1
        j3 = np.searchsorted(xs[2], z) - 1

        # 计算 x, y, z 相对于 xs 中找到的最大索引的偏移量
        s1 = x - xs[0][j1]
        s2 = y - xs[1][j2]
        s3 = z - xs[2][j3]

        # 初始化 val 为 0，用于存储插值的结果
        val = 0
        # 循环遍历 c 的所有可能组合
        for k1 in range(c.shape[0]):
            for k2 in range(c.shape[1]):
                for k3 in range(c.shape[2]):
                    # 使用三次线性插值计算最终的插值值 val
                    val += (c[nx-k1-1, ny-k2-1, nz-k3-1, j1, j2, j3]
                            * _dpow(s1, k1, nu[0])
                            * _dpow(s2, k2, nu[1])
                            * _dpow(s3, k3, nu[2]))

        # 将计算得到的插值结果 val 存储到 out 的当前索引 jout 处
        out[jout] = val

    # 返回插值结果数组 out
    return out
# 对四维分段多项式进行直接计算评估
def _ppoly4d_eval(c, xs, xnew, ynew, znew, unew, nu=None):
    # 如果 nu 未提供，默认为 (0, 0, 0, 0)
    if nu is None:
        nu = (0, 0, 0, 0)

    # 创建一个空的数组用于存储结果
    out = np.empty((len(xnew),), dtype=c.dtype)

    # 获取多维系数数组的维度信息
    mx, my, mz, mu = c.shape[:4]

    # 遍历新的坐标点集合 (xnew, ynew, znew, unew)
    for jout, (x, y, z, u) in enumerate(zip(xnew, ynew, znew, unew)):
        # 检查当前坐标点是否在各个维度的范围内
        if not ((xs[0][0] <= x <= xs[0][-1]) and
                (xs[1][0] <= y <= xs[1][-1]) and
                (xs[2][0] <= z <= xs[2][-1]) and
                (xs[3][0] <= u <= xs[3][-1])):
            # 如果坐标点超出范围，则输出结果为 NaN
            out[jout] = np.nan
            continue

        # 根据坐标点在每个维度上的位置，确定相邻的索引
        j1 = np.searchsorted(xs[0], x) - 1
        j2 = np.searchsorted(xs[1], y) - 1
        j3 = np.searchsorted(xs[2], z) - 1
        j4 = np.searchsorted(xs[3], u) - 1

        # 计算在每个维度上的插值权重
        s1 = x - xs[0][j1]
        s2 = y - xs[1][j2]
        s3 = z - xs[2][j3]
        s4 = u - xs[3][j4]

        # 初始化结果值
        val = 0

        # 使用四重循环计算多项式的值，考虑每个维度上的系数及权重
        for k1 in range(c.shape[0]):
            for k2 in range(c.shape[1]):
                for k3 in range(c.shape[2]):
                    for k4 in range(c.shape[3]):
                        val += (c[mx-k1-1, my-k2-1, mz-k3-1, mu-k4-1, j1, j2, j3, j4]
                                * _dpow(s1, k1, nu[0])
                                * _dpow(s2, k2, nu[1])
                                * _dpow(s3, k3, nu[2])
                                * _dpow(s4, k4, nu[3]))

        # 将计算结果存储在结果数组中
        out[jout] = val

    # 返回计算结果数组
    return out
```