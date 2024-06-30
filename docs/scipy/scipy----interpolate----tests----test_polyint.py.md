# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_polyint.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入 io 模块，提供了在内存中读写数据的能力
import io
# 导入 numpy 库并使用别名 np
import numpy as np

# 从 numpy.testing 模块中导入多个断言函数，用于单元测试中的数组比较
from numpy.testing import (
    assert_almost_equal, assert_array_equal, assert_array_almost_equal,
    assert_allclose, assert_equal, assert_)
# 从 pytest 模块中导入 raises 函数并起别名为 assert_raises
from pytest import raises as assert_raises
# 导入 pytest 模块
import pytest

# 从 scipy.interpolate 模块中导入多个插值器类和函数
from scipy.interpolate import (
    KroghInterpolator, krogh_interpolate,
    BarycentricInterpolator, barycentric_interpolate,
    approximate_taylor_polynomial, CubicHermiteSpline, pchip,
    PchipInterpolator, pchip_interpolate, Akima1DInterpolator, CubicSpline,
    make_interp_spline)

# 定义函数 check_shape，用于检查插值器在给定形状下的行为
def check_shape(interpolator_cls, x_shape, y_shape, deriv_shape=None, axis=0,
                extra_args={}):
    # 设置随机种子，确保结果可复现
    np.random.seed(1234)

    # 设置示例输入数据 x
    x = [-1, 0, 1, 2, 3, 4]
    # 创建一个维度序列 s，用于转置 y 的形状
    s = list(range(1, len(y_shape)+1))
    s.insert(axis % (len(y_shape)+1), 0)
    # 使用随机数据创建数组 y，形状通过 s 转置后变换
    y = np.random.rand(*((6,) + y_shape)).transpose(s)

    # 创建一个全零数组 xi，形状为 x_shape
    xi = np.zeros(x_shape)
    # 如果插值器是 CubicHermiteSpline 类，生成随机数组 dydx 作为导数
    if interpolator_cls is CubicHermiteSpline:
        dydx = np.random.rand(*((6,) + y_shape)).transpose(s)
        # 使用插值器类生成插值函数，并计算在 xi 处的插值结果 yi
        yi = interpolator_cls(x, y, dydx, axis=axis, **extra_args)(xi)
    else:
        # 使用插值器类生成插值函数，并计算在 xi 处的插值结果 yi
        yi = interpolator_cls(x, y, axis=axis, **extra_args)(xi)

    # 计算期望的输出形状 target_shape
    target_shape = ((deriv_shape or ()) + y.shape[:axis]
                    + x_shape + y.shape[axis:][1:])
    # 断言插值结果 yi 的形状与期望形状相等
    assert_equal(yi.shape, target_shape)

    # 使用列表作为输入数据，检查插值器是否能够处理列表形式的数据
    if x_shape and y.size > 0:
        if interpolator_cls is CubicHermiteSpline:
            interpolator_cls(list(x), list(y), list(dydx), axis=axis,
                             **extra_args)(list(xi))
        else:
            interpolator_cls(list(x), list(y), axis=axis,
                             **extra_args)(list(xi))

    # 当 xi 非空且 deriv_shape 为空时，检查插值结果的值是否接近期望值
    if xi.size > 0 and deriv_shape is None:
        # 计算广播后的 yv，以匹配 yi 和 y 的形状
        bs_shape = y.shape[:axis] + (1,)*len(x_shape) + y.shape[axis:][1:]
        yv = y[((slice(None,),)*(axis % y.ndim)) + (1,)]
        yv = yv.reshape(bs_shape)

        # 对比 yi 和 y，断言它们的值非常接近
        yi, y = np.broadcast_arrays(yi, yv)
        assert_allclose(yi, y)


# 定义不同的形状组合用于测试
SHAPES = [(), (0,), (1,), (6, 2, 5)]

# 定义测试函数 test_shapes
def test_shapes():

    # 定义一个内部函数 spl_interp，用于返回插值器类的实例
    def spl_interp(x, y, axis):
        return make_interp_spline(x, y, axis=axis)

    # 遍历不同的插值器类和形状组合进行测试
    for ip in [KroghInterpolator, BarycentricInterpolator, CubicHermiteSpline,
               pchip, Akima1DInterpolator, CubicSpline, spl_interp]:
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    # 根据插值器类的不同，调用 check_shape 函数进行测试
                    if ip != CubicSpline:
                        check_shape(ip, s1, s2, None, axis)
                    else:
                        # 对于 CubicSpline，遍历不同的边界条件进行测试
                        for bc in ['natural', 'clamped']:
                            extra = {'bc_type': bc}
                            check_shape(ip, s1, s2, None, axis, extra)

# 定义测试函数 test_derivs_shapes
def test_derivs_shapes():
    # 对于每个插值器类，使用 ip 依次取值为 KroghInterpolator 和 BarycentricInterpolator
    for ip in [KroghInterpolator, BarycentricInterpolator]:
        
        # 定义一个函数 interpolator_derivs，接受参数 x, y, axis，默认 axis 为 0
        def interpolator_derivs(x, y, axis=0):
            # 返回创建的插值器对象 ip 的 derivatives 属性
            return ip(x, y, axis).derivatives
        
        # 对于每个形状 s1 和 s2 组合中的每一个 s1
        for s1 in SHAPES:
            
            # 对于每个形状 s1 和 s2 组合中的每一个 s2
            for s2 in SHAPES:
                
                # 对于 axis 从负数索引到正数索引的每一个值
                for axis in range(-len(s2), len(s2)):
                    
                    # 调用 check_shape 函数，检查 interpolator_derivs 函数对于给定形状 s1, s2, (6,) 和 axis 的结果
                    check_shape(interpolator_derivs, s1, s2, (6,), axis)
# 定义一个测试函数，用于验证不同插值方法的导数计算
def test_deriv_shapes():
    # 定义一个函数，使用 Krogh 插值方法并返回导数函数
    def krogh_deriv(x, y, axis=0):
        return KroghInterpolator(x, y, axis).derivative

    # 定义一个函数，使用 Barycentric 插值方法并返回导数函数
    def bary_deriv(x, y, axis=0):
        return BarycentricInterpolator(x, y, axis).derivative

    # 定义一个函数，使用 PCHIP 插值方法并返回一阶导数函数
    def pchip_deriv(x, y, axis=0):
        return pchip(x, y, axis).derivative()

    # 定义一个函数，使用 PCHIP 插值方法并返回二阶导数函数
    def pchip_deriv2(x, y, axis=0):
        return pchip(x, y, axis).derivative(2)

    # 定义一个函数，使用 PCHIP 插值方法并返回反导数函数
    def pchip_antideriv(x, y, axis=0):
        return pchip(x, y, axis).antiderivative()

    # 定义一个函数，使用 PCHIP 插值方法并返回反二阶导数函数
    def pchip_antideriv2(x, y, axis=0):
        return pchip(x, y, axis).antiderivative(2)

    # 定义一个函数，使用 PCHIP 插值方法并返回就地一阶导数函数
    def pchip_deriv_inplace(x, y, axis=0):
        # 创建一个继承自 PchipInterpolator 的匿名类，重载 __call__ 方法以实现一阶导数计算
        class P(PchipInterpolator):
            def __call__(self, x):
                return PchipInterpolator.__call__(self, x, 1)
            pass
        return P(x, y, axis)

    # 定义一个函数，使用 Akima 插值方法并返回一阶导数函数
    def akima_deriv(x, y, axis=0):
        return Akima1DInterpolator(x, y, axis).derivative()

    # 定义一个函数，使用 Akima 插值方法并返回反导数函数
    def akima_antideriv(x, y, axis=0):
        return Akima1DInterpolator(x, y, axis).antiderivative()

    # 定义一个函数，使用 CubicSpline 插值方法并返回一阶导数函数
    def cspline_deriv(x, y, axis=0):
        return CubicSpline(x, y, axis).derivative()

    # 定义一个函数，使用 CubicSpline 插值方法并返回反导数函数
    def cspline_antideriv(x, y, axis=0):
        return CubicSpline(x, y, axis).antiderivative()

    # 定义一个函数，使用 make_interp_spline 方法并返回一阶导数函数
    def bspl_deriv(x, y, axis=0):
        return make_interp_spline(x, y, axis=axis).derivative()

    # 定义一个函数，使用 make_interp_spline 方法并返回反导数函数
    def bspl_antideriv(x, y, axis=0):
        return make_interp_spline(x, y, axis=axis).antiderivative()

    # 对于每个插值方法函数 ip，以及形状集合 SHAPES 中的每对 s1, s2，和每个可能的轴向，执行形状检查
    for ip in [krogh_deriv, bary_deriv, pchip_deriv, pchip_deriv2, pchip_deriv_inplace,
               pchip_antideriv, pchip_antideriv2, akima_deriv, akima_antideriv,
               cspline_deriv, cspline_antideriv, bspl_deriv, bspl_antideriv]:
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    check_shape(ip, s1, s2, (), axis)


# 定义一个测试函数，用于验证复数输入情况下的插值方法
def test_complex():
    # 定义 x 和 y 数组，y 包含复数
    x = [1, 2, 3, 4]
    y = [1, 2, 1j, 3]

    # 对于每个插值方法类 ip 在 [KroghInterpolator, BarycentricInterpolator, CubicSpline] 中执行以下操作
    for ip in [KroghInterpolator, BarycentricInterpolator, CubicSpline]:
        # 创建插值对象 p
        p = ip(x, y)
        # 断言插值结果与原始数据 y 的接近程度
        assert_allclose(y, p(x))

    # 定义导数数组 dydx
    dydx = [0, -1j, 2, 3j]
    # 使用 CubicHermiteSpline 插值方法创建对象 p
    p = CubicHermiteSpline(x, y, dydx)
    # 断言插值结果与原始数据 y 的接近程度
    assert_allclose(y, p(x))
    # 断言导数结果与预期 dydx 的接近程度
    assert_allclose(dydx, p(x, 1))


# 定义 Krogh 类的测试类
class TestKrogh:
    # 初始化方法
    def setup_method(self):
        # 定义 true_poly 为一个多项式对象
        self.true_poly = np.polynomial.Polynomial([-4, 5, 1, 3, -2])
        # 定义测试用的 x 数组
        self.test_xs = np.linspace(-1, 1, 100)
        # 定义用于插值的 x 数组
        self.xs = np.linspace(-1, 1, 5)
        # 计算在 self.xs 上的 true_poly 值作为 ys
        self.ys = self.true_poly(self.xs)

    # 测试 Lagrange 插值方法
    def test_lagrange(self):
        # 创建 KroghInterpolator 对象 P
        P = KroghInterpolator(self.xs, self.ys)
        # 断言测试 xs 上的插值结果接近于 true_poly 在 test_xs 上的值
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    # 测试标量输入情况
    def test_scalar(self):
        # 创建 KroghInterpolator 对象 P
        P = KroghInterpolator(self.xs, self.ys)
        # 断言对标量输入 7 的插值结果接近于 true_poly(7)
        assert_almost_equal(self.true_poly(7), P(7))
        # 断言对数组输入 [7] 的插值结果接近于 true_poly([7])
        assert_almost_equal(self.true_poly(np.array(7)), P(np.array(7)))

    # 测试导数计算
    def test_derivatives(self):
        # 创建 KroghInterpolator 对象 P
        P = KroghInterpolator(self.xs, self.ys)
        # 计算在 test_xs 上的导数数组 D
        D = P.derivatives(self.test_xs)
        # 对于 D 的每一行进行断言，验证其接近于 true_poly 的对应阶导数在 test_xs 上的值
        for i in range(D.shape[0]):
            assert_almost_equal(self.true_poly.deriv(i)(self.test_xs),
                                D[i])
    # 使用 Krogh 插值法初始化插值对象 P，并传入给定的 xs 和 ys
    P = KroghInterpolator(self.xs,self.ys)
    # 计算 P 对象在测试点 test_xs 处的前 len(self.xs)+2 阶导数，并存储在 D 中
    D = P.derivatives(self.test_xs,len(self.xs)+2)
    # 遍历计算出的导数 D 的每一行
    for i in range(D.shape[0]):
        # 断言 Krogh 插值多项式的 i 阶导数在测试点 test_xs 处与 true_poly 的 i 阶导数值几乎相等
        assert_almost_equal(self.true_poly.deriv(i)(self.test_xs),
                            D[i])

    # 使用 Krogh 插值法初始化插值对象 P，并传入给定的 xs 和 ys
    P = KroghInterpolator(self.xs,self.ys)
    # 设置要计算的导数阶数为 m
    m = 10
    # 计算 P 对象在测试点 test_xs 处的前 m 阶导数，并存储在 r 中
    r = P.derivatives(self.test_xs,m)
    # 遍历计算出的导数 r 的每一阶
    for i in range(m):
        # 断言 Krogh 插值多项式的 i 阶导数在测试点 test_xs 处与 r[i] 几乎相等
        assert_almost_equal(P.derivative(self.test_xs,i), r[i])

    # 使用 Krogh 插值法初始化插值对象 P，并传入给定的 xs 和 ys
    P = KroghInterpolator(self.xs,self.ys)
    # 遍历计算从 len(self.xs) 到 2*len(self.xs) 的每一个阶数 i
    for i in range(len(self.xs), 2*len(self.xs)):
        # 断言 Krogh 插值多项式的 i 阶导数在测试点 test_xs 处为全零向量
        assert_almost_equal(P.derivative(self.test_xs,i),
                            np.zeros(len(self.test_xs)))

    # 创建多项式 poly1, poly2, poly3，并计算它们在 xs 上的函数值，存储在 ys 中
    poly1 = self.true_poly
    poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
    poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
    ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)

    # 使用 Krogh 插值法初始化插值对象 P，并传入给定的 xs 和 ys，设置 axis=0
    P = KroghInterpolator(self.xs, ys, axis=0)
    # 计算 P 对象在测试点 test_xs 处的多阶导数 D
    D = P.derivatives(self.test_xs)
    # 遍历计算出的多阶导数 D 的每一行
    for i in range(D.shape[0]):
        # 断言 D 的第 i 行与对应多项式 poly1, poly2, poly3 的第 i 阶导数在测试点 test_xs 处几乎相等
        assert_allclose(D[i],
                        np.stack((poly1.deriv(i)(self.test_xs),
                                  poly2.deriv(i)(self.test_xs),
                                  poly3.deriv(i)(self.test_xs)),
                                 axis=-1))

    # 创建多项式 poly1, poly2, poly3，并计算它们在 xs 上的函数值，存储在 ys 中
    poly1 = self.true_poly
    poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
    poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
    ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)

    # 使用 Krogh 插值法初始化插值对象 P，并传入给定的 xs 和 ys，设置 axis=0
    P = KroghInterpolator(self.xs, ys, axis=0)
    # 遍历计算从 0 到 P.n 的每一个阶数 i
    for i in range(P.n):
        # 断言 Krogh 插值多项式的 i 阶导数在测试点 test_xs 处与 poly1, poly2, poly3 的 i 阶导数几乎相等
        assert_allclose(P.derivative(self.test_xs, i),
                        np.stack((poly1.deriv(i)(self.test_xs),
                                  poly2.deriv(i)(self.test_xs),
                                  poly3.deriv(i)(self.test_xs)),
                                 axis=-1))

    # 使用 Krogh 插值法初始化插值对象 P，并传入给定的 xs 和 ys
    P = KroghInterpolator(self.xs,self.ys)
    # 断言 Krogh 插值多项式在测试点 test_xs 处与 true_poly 的值几乎相等
    assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    # 初始化 xs 和 ys，其中 ys 是一个二维数组
    xs = [0, 1, 2]
    ys = np.array([[0,1],[1,0],[2,1]])
    # 使用 Krogh 插值法初始化插值对象 P，并传入给定的 xs 和 ys
    P = KroghInterpolator(xs,ys)
    # 对 ys 的每一列分别创建 Krogh 插值对象 Pi
    Pi = [KroghInterpolator(xs,ys[:,i]) for i in range(ys.shape[1])]
    # 生成一个测试点序列 test_xs
    test_xs = np.linspace(-1,3,100)
    # 断言 Krogh 插值对象 P 在测试点 test_xs 处的值与 Pi 中每个对象在 test_xs 处的值的转置几乎相等
    assert_almost_equal(P(test_xs),
                        np.asarray([p(test_xs) for p in Pi]).T)
    # 断言 Krogh 插值对象 P 在测试点 test_xs 处的导数与 Pi 中每个对象在 test_xs 处的多阶导数的转置几乎相等
    assert_almost_equal(P.derivatives(test_xs),
            np.transpose(np.asarray([p.derivatives(test_xs) for p in Pi]),
                (1,2,0)))

    # 使用 Krogh 插值法初始化插值对象 P，并传入给定的 xs 和 ys
    P = KroghInterpolator(self.xs,self.ys)
    # 断言当传入空数组 [] 时，Krogh 插值对象 P 返回空数组 []
    assert_array_equal(P([]), [])
    # 使用 KroghInterpolator 初始化一个插值对象 P，基于给定的 self.xs 和 self.ys 数据
    P = KroghInterpolator(self.xs,self.ys)
    # 断言返回值 P(0) 的形状是一个空元组
    assert_array_equal(np.shape(P(0)), ())
    # 断言返回值 P(np.array(0)) 的形状是一个空元组
    assert_array_equal(np.shape(P(np.array(0))), ())
    # 断言返回值 P([0]) 的形状是一个包含一个元素的元组 (1,)
    assert_array_equal(np.shape(P([0])), (1,))
    # 断言返回值 P([0,1]) 的形状是一个包含两个元素的元组 (2,)
    assert_array_equal(np.shape(P([0,1])), (2,))

    # 使用 KroghInterpolator 初始化一个插值对象 P，基于给定的 self.xs 和 self.ys 数据
    P = KroghInterpolator(self.xs,self.ys)
    # 获取插值对象 P 的阶数 n
    n = P.n
    # 断言返回值 P.derivatives(0) 的形状是一个包含 n 个元素的元组 (n,)
    assert_array_equal(np.shape(P.derivatives(0)), (n,))
    # 断言返回值 P.derivatives(np.array(0)) 的形状是一个包含 n 个元素的元组 (n,)
    assert_array_equal(np.shape(P.derivatives(np.array(0))), (n,))
    # 断言返回值 P.derivatives([0]) 的形状是一个包含 n 行 1 列的元组 (n,1)
    assert_array_equal(np.shape(P.derivatives([0])), (n,1))
    # 断言返回值 P.derivatives([0,1]) 的形状是一个包含 n 行 2 列的元组 (n,2)
    assert_array_equal(np.shape(P.derivatives([0,1])), (n,2))

    # 使用 KroghInterpolator 初始化一个插值对象 P，基于给定的 self.xs 和用 np.outer(self.ys,np.arange(3)) 生成的数据
    P = KroghInterpolator(self.xs,np.outer(self.ys,np.arange(3)))
    # 断言返回值 P(0) 的形状是一个包含 3 个元素的元组 (3,)
    assert_array_equal(np.shape(P(0)), (3,))
    # 断言返回值 P([0]) 的形状是一个包含 1 行 3 列的元组 (1,3)
    assert_array_equal(np.shape(P([0])), (1,3))
    # 断言返回值 P([0,1]) 的形状是一个包含 2 行 3 列的元组 (2,3)
    assert_array_equal(np.shape(P([0,1])), (2,3))

    # 使用 KroghInterpolator 初始化一个插值对象 P，基于给定的 self.xs 和用 np.outer(self.ys,[1]) 生成的数据
    P = KroghInterpolator(self.xs,np.outer(self.ys,[1]))
    # 断言返回值 P(0) 的形状是一个包含 1 个元素的元组 (1,)
    assert_array_equal(np.shape(P(0)), (1,))
    # 断言返回值 P([0]) 的形状是一个包含 1 行 1 列的元组 (1,1)
    assert_array_equal(np.shape(P([0])), (1,1))
    # 断言返回值 P([0,1]) 的形状是一个包含 2 行 1 列的元组 (2,1)
    assert_array_equal(np.shape(P([0,1])), (2,1))

    # 使用 KroghInterpolator 初始化一个插值对象 P，基于给定的 self.xs 和用 np.outer(self.ys,np.arange(3)) 生成的数据
    P = KroghInterpolator(self.xs,np.outer(self.ys,np.arange(3)))
    # 获取插值对象 P 的阶数 n
    n = P.n
    # 断言返回值 P.derivatives(0) 的形状是一个包含 n 行 3 列的元组 (n,3)
    assert_array_equal(np.shape(P.derivatives(0)), (n,3))
    # 断言返回值 P.derivatives([0]) 的形状是一个包含 n 行 1 列 3 深度的元组 (n,1,3)
    assert_array_equal(np.shape(P.derivatives([0])), (n,1,3))
    # 断言返回值 P.derivatives([0,1]) 的形状是一个包含 n 行 2 列 3 深度的元组 (n,2,3)
    assert_array_equal(np.shape(P.derivatives([0,1])), (n,2,3))

    # 使用 KroghInterpolator 初始化一个插值对象 P，基于给定的 self.xs 和 self.ys 数据
    P = KroghInterpolator(self.xs, self.ys)
    # 设置 ki 作为 krogh_interpolate 的别名
    ki = krogh_interpolate
    # 断言 P(self.test_xs) 的返回值与 ki(self.xs, self.ys, self.test_xs) 的近似值相等
    assert_almost_equal(P(self.test_xs), ki(self.xs, self.ys, self.test_xs))
    # 断言 P.derivative(self.test_xs, 2) 的返回值与 ki(self.xs, self.ys, self.test_xs, der=2) 的近似值相等
    assert_almost_equal(P.derivative(self.test_xs, 2),
                        ki(self.xs, self.ys, self.test_xs, der=2))
    # 断言 P.derivatives(self.test_xs, 2) 的返回值与 ki(self.xs, self.ys, self.test_xs, der=[0, 1]) 的近似值相等
    assert_almost_equal(P.derivatives(self.test_xs, 2),
                        ki(self.xs, self.ys, self.test_xs, der=[0, 1]))

    # 检查输入参数是否正确转换为浮点数，参考 gh-3669
    x = [0, 234, 468, 702, 936, 1170, 1404, 2340, 3744, 6084, 8424,
         13104, 60000]
    offset_cdf = np.array([-0.95, -0.86114777, -0.8147762, -0.64072425,
                           -0.48002351, -0.34925329, -0.26503107,
                           -0.13148093, -0.12988833, -0.12979296,
                           -0.12973574, -0.08582937, 0.05])
    # 使用 KroghInterpolator 初始化一个插值对象 f，基于给定的 x 和 offset_cdf 数据
    f = KroghInterpolator(x, offset_cdf)
    # 断言 (f(x) - offset_cdf) / f.derivative(x, 1) 的绝对值的平均值小于等于 1e-10
    assert_allclose(abs((f(x) - offset_cdf) / f.derivative(x, 1)),
                    0, atol=1e-10)
    # 定义一个测试函数，用于检验复杂情况下的导数计算
    def test_derivatives_complex(self):
        # 设置输入数据 x 和复数 y，其中包含实部和虚部
        x, y = np.array([-1, -1, 0, 1, 1]), np.array([1, 1.0j, 0, -1, 1.0j])
        # 使用 KroghInterpolator 类创建插值函数对象 func
        func = KroghInterpolator(x, y)
        # 计算在参数 0 处的复数导数
        cmplx = func.derivatives(0)

        # 将复数函数分解为实部和虚部的插值函数，并分别计算在参数 0 处的导数
        cmplx2 = (KroghInterpolator(x, y.real).derivatives(0) +
                  1j*KroghInterpolator(x, y.imag).derivatives(0))
        # 断言两种方法计算得到的复数导数在数值上的近似相等性，设置数值容差为 1e-15
        assert_allclose(cmplx, cmplx2, atol=1e-15)

    # 定义一个测试函数，用于检验高阶插值时的警告信息是否正确抛出
    def test_high_degree_warning(self):
        # 在运行时检测是否会抛出 UserWarning，匹配警告信息中是否包含 "40 degrees provided,"
        with pytest.warns(UserWarning, match="40 degrees provided,"):
            # 使用 KroghInterpolator 类创建插值函数对象，x 包含 0 到 39 的整数，y 全为 1
            KroghInterpolator(np.arange(40), np.ones(40))
class TestTaylor:
    # 定义测试类 TestTaylor
    def test_exponential(self):
        # 测试指数函数的 Taylor 多项式逼近
        degree = 5
        # 计算指数函数在0点的Taylor多项式，最高次数为degree，精度为1e-15
        p = approximate_taylor_polynomial(np.exp, 0, degree, 1, 15)
        # 对于每一个次数i，验证p的导数在0点的值是否接近1
        for i in range(degree+1):
            assert_almost_equal(p(0), 1)
            # 更新p为其导数
            p = p.deriv()
        # 最终验证p在0点的值是否接近0
        assert_almost_equal(p(0), 0)


class TestBarycentric:
    # 定义测试类 TestBarycentric
    def setup_method(self):
        # 设置测试方法的初始化
        self.true_poly = np.polynomial.Polynomial([-4, 5, 1, 3, -2])
        # 真实多项式为[-4, 5, 1, 3, -2]
        self.test_xs = np.linspace(-1, 1, 100)
        # 在[-1, 1]区间上生成100个均匀分布的点作为测试点
        self.xs = np.linspace(-1, 1, 5)
        # 在[-1, 1]区间上生成5个均匀分布的点作为插值节点
        self.ys = self.true_poly(self.xs)
        # 计算真实多项式在插值节点上的取值作为插值函数的值

    def test_lagrange(self):
        # 测试拉格朗日插值
        P = BarycentricInterpolator(self.xs, self.ys)
        # 创建BarycentricInterpolator对象P，使用插值节点和对应值初始化
        assert_allclose(P(self.test_xs), self.true_poly(self.test_xs))
        # 验证插值函数在测试点上的值是否接近真实多项式在测试点上的值

    def test_scalar(self):
        # 测试插值函数对标量输入的计算
        P = BarycentricInterpolator(self.xs, self.ys)
        # 创建BarycentricInterpolator对象P，使用插值节点和对应值初始化
        assert_allclose(P(7), self.true_poly(7))
        # 验证插值函数在标量7上的值是否接近真实多项式在7上的值
        assert_allclose(P(np.array(7)), self.true_poly(np.array(7)))
        # 验证插值函数在数组[7]上的值是否接近真实多项式在数组[7]上的值

    def test_derivatives(self):
        # 测试插值函数的导数
        P = BarycentricInterpolator(self.xs, self.ys)
        # 创建BarycentricInterpolator对象P，使用插值节点和对应值初始化
        D = P.derivatives(self.test_xs)
        # 计算插值函数在测试点上的多阶导数
        for i in range(D.shape[0]):
            assert_allclose(self.true_poly.deriv(i)(self.test_xs), D[i])
            # 验证插值函数在测试点上的第i阶导数是否接近真实多项式在测试点上的第i阶导数

    def test_low_derivatives(self):
        # 测试插值函数的低阶导数
        P = BarycentricInterpolator(self.xs, self.ys)
        # 创建BarycentricInterpolator对象P，使用插值节点和对应值初始化
        D = P.derivatives(self.test_xs, len(self.xs)+2)
        # 计算插值函数在测试点上的高阶导数
        for i in range(D.shape[0]):
            assert_allclose(self.true_poly.deriv(i)(self.test_xs),
                            D[i],
                            atol=1e-12)
            # 验证插值函数在测试点上的第i阶导数是否接近真实多项式在测试点上的第i阶导数，容忍度为1e-12

    def test_derivative(self):
        # 测试插值函数的导数计算
        P = BarycentricInterpolator(self.xs, self.ys)
        # 创建BarycentricInterpolator对象P，使用插值节点和对应值初始化
        m = 10
        # 设置导数阶数m为10
        r = P.derivatives(self.test_xs, m)
        # 计算插值函数在测试点上的前m阶导数
        for i in range(m):
            assert_allclose(P.derivative(self.test_xs, i), r[i])
            # 验证插值函数在测试点上的第i阶导数是否接近计算出的前m阶导数

    def test_high_derivative(self):
        # 测试插值函数的高阶导数
        P = BarycentricInterpolator(self.xs, self.ys)
        # 创建BarycentricInterpolator对象P，使用插值节点和对应值初始化
        for i in range(len(self.xs), 5*len(self.xs)):
            # 遍历从插值节点数量到5倍插值节点数量的阶数范围
            assert_allclose(P.derivative(self.test_xs, i),
                            np.zeros(len(self.test_xs)))
            # 验证插值函数在测试点上的第i阶导数是否全为零向量

    def test_ndim_derivatives(self):
        # 测试多维插值函数的导数
        poly1 = self.true_poly
        # 第一个真实多项式为self.true_poly
        poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
        # 第二个多项式为[-2, 5, 3, -1]
        poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
        # 第三个多项式为[12, -3, 4, -5, 6]
        ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)
        # 计算三个多项式在插值节点上的值并堆叠为多维数组

        P = BarycentricInterpolator(self.xs, ys, axis=0)
        # 创建BarycentricInterpolator对象P，使用插值节点和多维值数组初始化，沿axis=0插值
        D = P.derivatives(self.test_xs)
        # 计算多维插值函数在测试点上的多阶导数
        for i in range(D.shape[0]):
            assert_allclose(D[i],
                            np.stack((poly1.deriv(i)(self.test_xs),
                                      poly2.deriv(i)(self.test_xs),
                                      poly3.deriv(i)(self.test_xs)),
                                     axis=-1),
                            atol=1e-12)
            # 验证多维插值函数在测试点上的第i阶导数是否接近真实多项式在测试点上的第i阶导数，容忍度为1e-12
    # 测试多维导数函数
    def test_ndim_derivative(self):
        # 获取真实多项式
        poly1 = self.true_poly
        # 创建第二个多项式对象
        poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
        # 创建第三个多项式对象
        poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
        # 在自变量xs上计算poly1、poly2和poly3的值，并沿着最后一个轴堆叠起来
        ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)

        # 使用BarycentricInterpolator对象P，沿着第一个轴（axis=0）初始化插值
        P = BarycentricInterpolator(self.xs, ys, axis=0)
        # 对P的所有插值点进行迭代
        for i in range(P.n):
            # 断言P在测试点self.test_xs处的导数与poly1、poly2和poly3在相同点处的导数一致
            assert_allclose(P.derivative(self.test_xs, i),
                            np.stack((poly1.deriv(i)(self.test_xs),
                                      poly2.deriv(i)(self.test_xs),
                                      poly3.deriv(i)(self.test_xs)),
                                     axis=-1),
                            atol=1e-12)

    # 测试延迟设置插值数据的功能
    def test_delayed(self):
        # 使用BarycentricInterpolator对象P，仅使用xs初始化
        P = BarycentricInterpolator(self.xs)
        # 设置P的插值数据为self.ys
        P.set_yi(self.ys)
        # 断言真实多项式在测试点self.test_xs处的值与P在同一点处的值接近
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    # 测试追加新的自变量和因变量对
    def test_append(self):
        # 使用前三个xs和ys初始化BarycentricInterpolator对象P
        P = BarycentricInterpolator(self.xs[:3], self.ys[:3])
        # 向P中追加剩余的xs和ys数据
        P.add_xi(self.xs[3:], self.ys[3:])
        # 断言真实多项式在测试点self.test_xs处的值与P在同一点处的值接近
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    # 测试向量值插值函数
    def test_vector(self):
        # 定义自变量xs和因变量ys
        xs = [0, 1, 2]
        ys = np.array([[0, 1], [1, 0], [2, 1]])
        # 使用BarycentricInterpolator对象BI初始化P，沿着第一个轴（axis=0）插值ys
        BI = BarycentricInterpolator
        P = BI(xs, ys)
        # 分别初始化Pi，每个Pi插值ys中的一列
        Pi = [BI(xs, ys[:, i]) for i in range(ys.shape[1])]
        # 在测试点test_xs上断言P的值与Pi中每个函数在相同点处的值的转置接近
        test_xs = np.linspace(-1, 3, 100)
        assert_almost_equal(P(test_xs),
                            np.asarray([p(test_xs) for p in Pi]).T)

    # 测试标量值插值函数的形状
    def test_shapes_scalarvalue(self):
        # 使用BarycentricInterpolator对象P，使用self.xs和self.ys初始化
        P = BarycentricInterpolator(self.xs, self.ys)
        # 断言P在0处的形状为空元组
        assert_array_equal(np.shape(P(0)), ())
        # 断言P在np.array(0)处的形状为空元组
        assert_array_equal(np.shape(P(np.array(0))), ())
        # 断言P在[0]处的形状为(1,)
        assert_array_equal(np.shape(P([0])), (1,))
        # 断言P在[0, 1]处的形状为(2,)
        assert_array_equal(np.shape(P([0, 1])), (2,))

    # 测试标量值导数函数的形状
    def test_shapes_scalarvalue_derivative(self):
        # 使用BarycentricInterpolator对象P，使用self.xs和self.ys初始化
        P = BarycentricInterpolator(self.xs,self.ys)
        n = P.n
        # 断言P在0处的导数的形状为(n,)
        assert_array_equal(np.shape(P.derivatives(0)), (n,))
        # 断言P在np.array(0)处的导数的形状为(n,)
        assert_array_equal(np.shape(P.derivatives(np.array(0))), (n,))
        # 断言P在[0]处的导数的形状为(n,1)
        assert_array_equal(np.shape(P.derivatives([0])), (n,1))
        # 断言P在[0,1]处的导数的形状为(n,2)
        assert_array_equal(np.shape(P.derivatives([0,1])), (n,2))

    # 测试向量值插值函数的形状
    def test_shapes_vectorvalue(self):
        # 使用BarycentricInterpolator对象P，使用self.xs和由self.ys和np.arange(3)外积得到的数组初始化
        P = BarycentricInterpolator(self.xs, np.outer(self.ys, np.arange(3)))
        # 断言P在0处的形状为(3,)
        assert_array_equal(np.shape(P(0)), (3,))
        # 断言P在[0]处的形状为(1,3)
        assert_array_equal(np.shape(P([0])), (1, 3))
        # 断言P在[0, 1]处的形状为(2,3)
        assert_array_equal(np.shape(P([0, 1])), (2, 3))

    # 测试一维向量值插值函数的形状
    def test_shapes_1d_vectorvalue(self):
        # 使用BarycentricInterpolator对象P，使用self.xs和由self.ys和[1]外积得到的数组初始化
        P = BarycentricInterpolator(self.xs, np.outer(self.ys, [1]))
        # 断言P在0处的形状为(1,)
        assert_array_equal(np.shape(P(0)), (1,))
        # 断言P在[0]处的形状为(1,1)
        assert_array_equal(np.shape(P([0])), (1, 1))
        # 断言P在[0,1]处的形状为(2,1)
        assert_array_equal(np.shape(P([0,1])), (2, 1))
    # 测试 BarycentricInterpolator 类的 shapes_vectorvalue_derivative 方法
    def test_shapes_vectorvalue_derivative(self):
        # 使用 BarycentricInterpolator 对象 P，使用给定的 xs 和 ys 初始化
        P = BarycentricInterpolator(self.xs,np.outer(self.ys,np.arange(3)))
        # 获取 P 的节点数 n
        n = P.n
        # 断言 P.derivatives(0) 的形状与 (n,3) 相同
        assert_array_equal(np.shape(P.derivatives(0)), (n,3))
        # 断言 P.derivatives([0]) 的形状与 (n,1,3) 相同
        assert_array_equal(np.shape(P.derivatives([0])), (n,1,3))
        # 断言 P.derivatives([0,1]) 的形状与 (n,2,3) 相同
        assert_array_equal(np.shape(P.derivatives([0,1])), (n,2,3))

    # 测试 barycentric_interpolate 的 wrapper 函数
    def test_wrapper(self):
        # 使用 BarycentricInterpolator 对象 P，使用给定的 xs 和 ys 初始化
        P = BarycentricInterpolator(self.xs, self.ys)
        # 获取 barycentric_interpolate 的别名 bi
        bi = barycentric_interpolate
        # 断言 P 在 test_xs 上的插值与 bi 函数在相同参数下的结果相近
        assert_allclose(P(self.test_xs), bi(self.xs, self.ys, self.test_xs))
        # 断言 P 在 test_xs 上的二阶导数插值与 bi 函数在相同参数下的结果相近
        assert_allclose(P.derivative(self.test_xs, 2),
                            bi(self.xs, self.ys, self.test_xs, der=2))
        # 断言 P 在 test_xs 上的多阶导数插值与 bi 函数在相同参数下的结果相近
        assert_allclose(P.derivatives(self.test_xs, 2),
                            bi(self.xs, self.ys, self.test_xs, der=[0, 1]))

    # 测试整数输入的情况
    def test_int_input(self):
        # 创建一个数组 x，包含 [1000, 2000, ..., 10000]
        x = 1000 * np.arange(1, 11)  # np.prod(x[-1] - x[:-1]) overflows
        # 创建一个数组 y，包含 [1, 2, ..., 10]
        y = np.arange(1, 11)
        # 使用 barycentric_interpolate 函数在 x, y 上插值计算 1000 * 9.5 的值
        value = barycentric_interpolate(x, y, 1000 * 9.5)
        # 断言计算结果与期望值 9.5 相近
        assert_almost_equal(value, 9.5)

    # 测试大规模 Chebyshev 插值的情况
    def test_large_chebyshev(self):
        # 对于第二类 Chebyshev 点，权重可以通过解析方法求解
        # 使用简单计算的 barycentric weights 在大规模情况下会失败
        # 我们使用分析的 Chebyshev 权重测试大规模情况下的正确性

        # 设置节点数 n=1100
        n = 1100
        # 创建一个浮点数数组 j，包含 [0.0, 1.0, ..., 1100.0]
        j = np.arange(n + 1).astype(np.float64)
        # 根据 j 计算对应的 Chebyshev 点 x
        x = np.cos(j * np.pi / n)

        # 根据 Berrut 和 Trefethen 2004 年的公式计算权重 w
        w = (-1) ** j
        w[0] *= 0.5
        w[-1] *= 0.5

        # 使用 BarycentricInterpolator 初始化 P
        P = BarycentricInterpolator(x)

        # 权重中的常数因子在多项式的评估中可以抵消
        factor = P.wi[0]
        # 断言 P.wi / (2 * factor) 与预期的权重 w 相近
        assert_almost_equal(P.wi / (2 * factor), w)

    # 测试警告信息的处理情况
    def test_warning(self):
        # 测试在计算插值值等于插值点时是否正确忽略了除以零的警告
        # 使用 BarycentricInterpolator 初始化 P，使用插值点 [0, 1] 和值 [1, 2]
        P = BarycentricInterpolator([0, 1], [1, 2])
        # 在计算插值值时，检查是否忽略了除以零的警告
        with np.errstate(divide='raise'):
            yi = P(P.xi)

        # 检查插值值是否与节点处的输入值匹配
        assert_almost_equal(yi, P.yi.ravel())

    # 测试重复节点的情况
    def test_repeated_node(self):
        # 检查重复节点是否会引发 ValueError
        # （计算权重时需要除以 xi[i] - xi[j]）
        xis = np.array([0.1, 0.5, 0.9, 0.5])
        ys = np.array([1, 2, 3, 4])
        # 使用 pytest 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError,
                           match="Interpolation points xi must be distinct."):
            BarycentricInterpolator(xis, ys)
class TestPCHIP:
    # 创建一个测试 PCHIP 插值器的测试类
    def _make_random(self, npts=20):
        # 生成随机的 x 和 y 数据点
        np.random.seed(1234)
        xi = np.sort(np.random.random(npts))
        yi = np.random.random(npts)
        return pchip(xi, yi), xi, yi

    def test_overshoot(self):
        # 测试 PCHIP 插值器不应该出现过冲现象
        p, xi, yi = self._make_random()
        for i in range(len(xi)-1):
            x1, x2 = xi[i], xi[i+1]
            y1, y2 = yi[i], yi[i+1]
            if y1 > y2:
                y1, y2 = y2, y1
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            # 断言插值结果在一定误差范围内不超过数据范围
            assert_(((y1 <= yp + 1e-15) & (yp <= y2 + 1e-15)).all())

    def test_monotone(self):
        # 测试 PCHIP 插值器能够保持单调性
        p, xi, yi = self._make_random()
        for i in range(len(xi)-1):
            x1, x2 = xi[i], xi[i+1]
            y1, y2 = yi[i], yi[i+1]
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            # 断言插值结果在局部区间内保持单调性
            assert_(((y2-y1) * (yp[1:] - yp[:1]) > 0).all())

    def test_cast(self):
        # 对整数输入数据进行回归测试，参见 gh-3453
        data = np.array([[0, 4, 12, 27, 47, 60, 79, 87, 99, 100],
                         [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55]])
        xx = np.arange(100)
        curve = pchip(data[0], data[1])(xx)

        data1 = data * 1.0
        curve1 = pchip(data1[0], data1[1])(xx)

        # 断言两种数据类型的插值结果接近
        assert_allclose(curve, curve1, atol=1e-14, rtol=1e-14)

    def test_nag(self):
        # 来自 NAG C 实现的例子，用作插值器计算导数的冒烟测试，参见 gh-5326
        dataStr = '''
          7.99   0.00000E+0
          8.09   0.27643E-4
          8.19   0.43750E-1
          8.70   0.16918E+0
          9.20   0.46943E+0
         10.00   0.94374E+0
         12.00   0.99864E+0
         15.00   0.99992E+0
         20.00   0.99999E+0
        '''
        data = np.loadtxt(io.StringIO(dataStr))
        pch = pchip(data[:,0], data[:,1])

        resultStr = '''
           7.9900       0.0000
           9.1910       0.4640
          10.3920       0.9645
          11.5930       0.9965
          12.7940       0.9992
          13.9950       0.9998
          15.1960       0.9999
          16.3970       1.0000
          17.5980       1.0000
          18.7990       1.0000
          20.0000       1.0000
        '''
        result = np.loadtxt(io.StringIO(resultStr))
        # 断言插值结果与预期结果非常接近
        assert_allclose(result[:,1], pch(result[:,0]), rtol=0., atol=5e-5)

    def test_endslopes(self):
        # 这是一个针对 gh-3453 的冒烟测试：PCHIP 插值器不应该在数据不暗示边缘导数为零时将边缘斜率设为零
        x = np.array([0.0, 0.1, 0.25, 0.35])
        y1 = np.array([279.35, 0.5e3, 1.0e3, 2.5e3])
        y2 = np.array([279.35, 2.5e3, 1.50e3, 1.0e3])
        for pp in (pchip(x, y1), pchip(x, y2)):
            for t in (x[0], x[-1]):
                # 断言在边界点上，PCHIP 插值器的一阶导数不为零
                assert_(pp(t, 1) != 0)
    # 定义测试函数，测试在所有元素为零的情况下的插值器行为
    def test_all_zeros(self):
        # 创建一个包含 0 到 9 的数组
        x = np.arange(10)
        # 创建一个与 x 相同形状的零数组
        y = np.zeros_like(x)

        # 使用警告捕获机制确保不会生成任何警告
        with warnings.catch_warnings():
            # 设置警告过滤器，将警告设置为错误
            warnings.filterwarnings('error')
            # 使用 pchip 插值器处理 x, y 数据
            pch = pchip(x, y)

        # 创建一个从 0 到 9 的等间距的数组
        xx = np.linspace(0, 9, 101)
        # 断言插值器对于 xx 中的所有值返回 0
        assert_equal(pch(xx), 0.)

    # 定义测试函数，测试仅包含两个点的情况
    def test_two_points(self):
        # 回归测试，用于解决 gh-6222: 当 pchip([0, 1], [0, 1]) 失败时，
        # 因为尝试使用三点方案估算边缘导数，但实际上只有两个点可用。
        # 应该构建一个线性插值器来处理这种情况。
        # 创建一个从 0 到 1 等间距的 11 个点的数组
        x = np.linspace(0, 1, 11)
        # 创建 pchip 插值器，使用点 (0, 0) 和 (1, 2)
        p = pchip([0, 1], [0, 2])
        # 断言插值器对于 x 返回的结果与 2x 接近，容差为 1e-15
        assert_allclose(p(x), 2*x, atol=1e-15)

    # 定义测试函数，测试 pchip 插值器的插值功能
    def test_pchip_interpolate(self):
        # 断言使用 pchip_interpolate 插值器插值 [1, 2, 3] 和 [4, 5, 6]，
        # 在位置 [0.5] 处，一阶导数的插值结果为 [1.]
        assert_array_almost_equal(
            pchip_interpolate([1,2,3], [4,5,6], [0.5], der=1),
            [1.])

        # 断言使用 pchip_interpolate 插值器插值 [1, 2, 3] 和 [4, 5, 6]，
        # 在位置 [0.5] 处，零阶导数的插值结果为 [3.5]
        assert_array_almost_equal(
            pchip_interpolate([1,2,3], [4,5,6], [0.5], der=0),
            [3.5])

        # 断言使用 pchip_interpolate 插值器插值 [1, 2, 3] 和 [4, 5, 6]，
        # 在位置 [0.5] 处，同时计算零阶和一阶导数的插值结果为 [[3.5], [1]]
        assert_array_almost_equal(
            pchip_interpolate([1,2,3], [4,5,6], [0.5], der=[0, 1]),
            [[3.5], [1]])

    # 定义测试函数，测试 pchip 插值器的根功能
    def test_roots(self):
        # 回归测试，解决 gh-6357: .roots 方法应该正常工作
        # 创建 pchip 插值器，使用点 (0, -1) 和 (1, 1)
        p = pchip([0, 1], [-1, 1])
        # 计算插值器的根
        r = p.roots()
        # 断言所有根接近 0.5
        assert_allclose(r, 0.5)
# 定义一个名为 TestCubicSpline 的类，用于测试三次样条插值的正确性
class TestCubicSpline:

    # 静态方法：检查三次样条插值的正确性，包括连续性和边界条件
    @staticmethod
    def check_correctness(S, bc_start='not-a-knot', bc_end='not-a-knot',
                          tol=1e-14):
        """Check that spline coefficients satisfy the continuity and boundary
        conditions."""
        
        # 获取样条插值的节点 x 和系数 c
        x = S.x
        c = S.c
        
        # 计算节点之间的差值
        dx = np.diff(x)
        dx = dx.reshape([dx.shape[0]] + [1] * (c.ndim - 2))
        dxi = dx[:-1]

        # 检查 C2 连续性
        assert_allclose(c[3, 1:], c[0, :-1] * dxi**3 + c[1, :-1] * dxi**2 +
                        c[2, :-1] * dxi + c[3, :-1], rtol=tol, atol=tol)
        assert_allclose(c[2, 1:], 3 * c[0, :-1] * dxi**2 +
                        2 * c[1, :-1] * dxi + c[2, :-1], rtol=tol, atol=tol)
        assert_allclose(c[1, 1:], 3 * c[0, :-1] * dxi + c[1, :-1],
                        rtol=tol, atol=tol)

        # 如果节点数为 3 并且起始和结束边界条件均为 'not-a-knot'，则检查第三阶导数为 0，即为抛物线
        if x.size == 3 and bc_start == 'not-a-knot' and bc_end == 'not-a-knot':
            assert_allclose(c[0], 0, rtol=tol, atol=tol)
            return

        # 检查周期性边界条件
        if bc_start == 'periodic':
            assert_allclose(S(x[0], 0), S(x[-1], 0), rtol=tol, atol=tol)
            assert_allclose(S(x[0], 1), S(x[-1], 1), rtol=tol, atol=tol)
            assert_allclose(S(x[0], 2), S(x[-1], 2), rtol=tol, atol=tol)
            return

        # 检查其他边界条件
        if bc_start == 'not-a-knot':
            if x.size == 2:
                # 计算斜率并检查
                slope = (S(x[1]) - S(x[0])) / dx[0]
                assert_allclose(S(x[0], 1), slope, rtol=tol, atol=tol)
            else:
                assert_allclose(c[0, 0], c[0, 1], rtol=tol, atol=tol)
        elif bc_start == 'clamped':
            assert_allclose(S(x[0], 1), 0, rtol=tol, atol=tol)
        elif bc_start == 'natural':
            assert_allclose(S(x[0], 2), 0, rtol=tol, atol=tol)
        else:
            order, value = bc_start
            assert_allclose(S(x[0], order), value, rtol=tol, atol=tol)

        # 检查结束边界条件
        if bc_end == 'not-a-knot':
            if x.size == 2:
                # 计算斜率并检查
                slope = (S(x[1]) - S(x[0])) / dx[0]
                assert_allclose(S(x[1], 1), slope, rtol=tol, atol=tol)
            else:
                assert_allclose(c[0, -1], c[0, -2], rtol=tol, atol=tol)
        elif bc_end == 'clamped':
            assert_allclose(S(x[-1], 1), 0, rtol=tol, atol=tol)
        elif bc_end == 'natural':
            assert_allclose(S(x[-1], 2), 0, rtol=2*tol, atol=2*tol)
        else:
            order, value = bc_end
            assert_allclose(S(x[-1], order), value, rtol=tol, atol=tol)
    # 检查所有边界条件的函数，用于创建和测试样条插值
    def check_all_bc(self, x, y, axis):
        # 获取 y 的形状，并移除指定轴，得到导数的形状
        deriv_shape = list(y.shape)
        del deriv_shape[axis]
        # 创建一个填充数值为2的数组，作为第一导数
        first_deriv = np.empty(deriv_shape)
        first_deriv.fill(2)
        # 创建一个填充数值为-1的数组，作为第二导数
        second_deriv = np.empty(deriv_shape)
        second_deriv.fill(-1)
        # 定义所有可能的边界条件
        bc_all = [
            'not-a-knot',  # 非节点条件
            'natural',     # 自然边界条件
            'clamped',     # 夹紧边界条件
            (1, first_deriv),   # 指定第一导数的边界条件
            (2, second_deriv)   # 指定第二导数的边界条件
        ]
        # 对前三种基本边界条件进行迭代
        for bc in bc_all[:3]:
            # 创建样条插值对象 S，使用指定的边界条件
            S = CubicSpline(x, y, axis=axis, bc_type=bc)
            # 检查插值 S 的正确性
            self.check_correctness(S, bc, bc)

        # 对所有边界条件进行两两组合的迭代
        for bc_start in bc_all:
            for bc_end in bc_all:
                # 创建样条插值对象 S，使用起始和结束边界条件的组合
                S = CubicSpline(x, y, axis=axis, bc_type=(bc_start, bc_end))
                # 检查插值 S 的正确性，指定容差为 2e-14
                self.check_correctness(S, bc_start, bc_end, tol=2e-14)

    # 测试一般情况的函数
    def test_general(self):
        # 定义示例数据点 x 和对应的 y 值
        x = np.array([-1, 0, 0.5, 2, 4, 4.5, 5.5, 9])
        y = np.array([0, -0.5, 2, 3, 2.5, 1, 1, 0.5])
        # 对不同数量的数据点进行迭代测试
        for n in [2, 3, x.size]:
            # 测试在 x[:n] 上，针对轴 0 的所有边界条件
            self.check_all_bc(x[:n], y[:n], 0)

            # 创建一个形状为 (2, n, 2) 的空数组 Y，并分别填充数据
            Y = np.empty((2, n, 2))
            Y[0, :, 0] = y[:n]
            Y[0, :, 1] = y[:n] - 1
            Y[1, :, 0] = y[:n] + 2
            Y[1, :, 1] = y[:n] + 3
            # 测试在 x[:n] 上，针对轴 1 的所有边界条件
            self.check_all_bc(x[:n], Y, 1)

    # 测试周期性边界条件的函数
    def test_periodic(self):
        # 对不同数量的数据点进行迭代测试
        for n in [2, 3, 5]:
            # 在 [0, 2π] 上生成 n 个均匀分布的数据点 x，并计算对应的 y 值
            x = np.linspace(0, 2 * np.pi, n)
            y = np.cos(x)
            # 创建周期性边界条件的样条插值对象 S
            S = CubicSpline(x, y, bc_type='periodic')
            # 检查周期性样条插值 S 的正确性
            self.check_correctness(S, 'periodic', 'periodic')

            # 创建形状为 (2, n, 2) 的空数组 Y，并填充数据
            Y = np.empty((2, n, 2))
            Y[0, :, 0] = y
            Y[0, :, 1] = y + 2
            Y[1, :, 0] = y - 1
            Y[1, :, 1] = y + 5
            # 创建在轴 1 上使用周期性边界条件的样条插值对象 S
            S = CubicSpline(x, Y, axis=1, bc_type='periodic')
            # 检查周期性样条插值 S 的正确性
            self.check_correctness(S, 'periodic', 'periodic')

    # 测试周期性边界条件插值函数的评估
    def test_periodic_eval(self):
        # 在 [0, 2π] 上生成 10 个均匀分布的数据点 x，并计算对应的 y 值
        x = np.linspace(0, 2 * np.pi, 10)
        y = np.cos(x)
        # 创建周期性边界条件的样条插值对象 S
        S = CubicSpline(x, y, bc_type='periodic')
        # 断言周期性边界条件插值在 1 和 1 + 2π 处的值近似相等
        assert_almost_equal(S(1), S(1 + 2 * np.pi), decimal=15)

    # 测试二阶导数连续性的函数
    def test_second_derivative_continuity_gh_11758(self):
        # 创建包含特定数据点的数组 x 和对应的 y 值
        x = np.array([0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5.0, 6.0,
                      7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0, 13.3])
        y = np.array([1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1,
                      2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4, 1.3])
        # 创建周期性边界条件的样条插值对象 S
        S = CubicSpline(x, y, bc_type='periodic', extrapolate='periodic')
        # 检查周期性边界条件插值 S 的正确性
        self.check_correctness(S, 'periodic', 'periodic')
    def test_three_points(self):
        # gh-11758: Fails computing a_m2_m1
        # In this case, s (first derivatives) could be found manually by solving
        # system of 2 linear equations. Due to solution of this system,
        # s[i] = (h1m2 + h2m1) / (h1 + h2), where h1 = x[1] - x[0], h2 = x[2] - x[1],
        # m1 = (y[1] - y[0]) / h1, m2 = (y[2] - y[1]) / h2

        # 定义输入数据点的 x 和 y 坐标
        x = np.array([1.0, 2.75, 3.0])
        y = np.array([1.0, 15.0, 1.0])
        
        # 创建三次样条插值对象 S，并使用周期性边界条件
        S = CubicSpline(x, y, bc_type='periodic')
        
        # 调用自定义方法，检查插值结果的正确性
        self.check_correctness(S, 'periodic', 'periodic')
        
        # 断言插值函数在 x 处的一阶导数值，与预期值接近
        assert_allclose(S.derivative(1)(x), np.array([-48.0, -48.0, -48.0]))

    def test_periodic_three_points_multidim(self):
        # make sure one multidimensional interpolator does the same as multiple
        # one-dimensional interpolators

        # 定义输入数据点的 x 和 y 坐标（多维情况）
        x = np.array([0.0, 1.0, 3.0])
        y = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        
        # 创建多维度的三次样条插值对象 S，并使用周期性边界条件
        S = CubicSpline(x, y, bc_type="periodic")
        
        # 调用自定义方法，检查插值结果的正确性
        self.check_correctness(S, 'periodic', 'periodic')
        
        # 分别创建单维度的三次样条插值对象 S0 和 S1，并使用周期性边界条件
        S0 = CubicSpline(x, y[:, 0], bc_type="periodic")
        S1 = CubicSpline(x, y[:, 1], bc_type="periodic")
        
        # 在指定区间上断言多维度插值结果与单维度插值结果的一致性
        q = np.linspace(0, 2, 5)
        assert_allclose(S(q)[:, 0], S0(q))
        assert_allclose(S(q)[:, 1], S1(q))

    def test_dtypes(self):
        # 定义整数类型的输入数据点 x 和 y
        x = np.array([0, 1, 2, 3], dtype=int)
        y = np.array([-5, 2, 3, 1], dtype=int)
        
        # 创建整数类型的三次样条插值对象 S
        S = CubicSpline(x, y)
        
        # 调用自定义方法，检查插值结果的正确性
        self.check_correctness(S)

        # 定义复数类型的输入数据点 y
        y = np.array([-1+1j, 0.0, 1-1j, 0.5-1.5j])
        
        # 创建复数类型的三次样条插值对象 S
        S = CubicSpline(x, y)
        
        # 调用自定义方法，检查插值结果的正确性
        self.check_correctness(S)

        # 使用不同的边界条件类型创建三次样条插值对象 S
        S = CubicSpline(x, x ** 3, bc_type=("natural", (1, 2j)))
        
        # 调用自定义方法，检查插值结果的正确性
        self.check_correctness(S, "natural", (1, 2j))

        # 使用不同的边界条件类型列表创建三次样条插值对象 S
        S = CubicSpline(x, y, bc_type=[(1, 2 + 0.5j), (2, 0.5 - 1j)])
        
        # 调用自定义方法，检查插值结果的正确性
        self.check_correctness(S, (1, 2 + 0.5j), (2, 0.5 - 1j))

    def test_small_dx(self):
        # 使用随机数生成器创建小间距的输入数据点 x 和对应的 y 值
        rng = np.random.RandomState(0)
        x = np.sort(rng.uniform(size=100))
        y = 1e4 + rng.uniform(size=100)
        
        # 创建小间距情况下的三次样条插值对象 S
        S = CubicSpline(x, y)
        
        # 调用自定义方法，检查插值结果的正确性，设置公差为 1e-13
        self.check_correctness(S, tol=1e-13)
    # 定义测试不正确输入的测试函数
    def test_incorrect_inputs(self):
        # 创建测试用的 NumPy 数组
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3, 4])
        xc = np.array([1 + 1j, 2, 3, 4])  # 包含复数，不符合要求
        xn = np.array([np.nan, 2, 3, 4])  # 包含 NaN，不符合要求
        xo = np.array([2, 1, 3, 4])       # 数组顺序不符合要求
        yn = np.array([np.nan, 2, 3, 4])  # 包含 NaN，不符合要求
        y3 = [1, 2, 3]                    # 数组长度不符合要求
        x1 = [1]                          # 数组长度不符合要求
        y1 = [1]                          # 数组长度不符合要求

        # 断言各种不正确输入情况下会触发 ValueError 异常
        assert_raises(ValueError, CubicSpline, xc, y)
        assert_raises(ValueError, CubicSpline, xn, y)
        assert_raises(ValueError, CubicSpline, x, yn)
        assert_raises(ValueError, CubicSpline, xo, y)
        assert_raises(ValueError, CubicSpline, x, y3)
        assert_raises(ValueError, CubicSpline, x[:, np.newaxis], y)
        assert_raises(ValueError, CubicSpline, x1, y1)

        # 定义不正确的边界条件组合
        wrong_bc = [('periodic', 'clamped'),
                    ((2, 0), (3, 10)),
                    ((1, 0), ),
                    (0., 0.),
                    'not-a-typo']

        # 遍历测试不正确的边界条件组合
        for bc_type in wrong_bc:
            assert_raises(ValueError, CubicSpline, x, y, 0, bc_type, True)

        # 当给定任意导数值时，形状不匹配的情况
        Y = np.c_[y, y]
        bc1 = ('clamped', (1, 0))
        bc2 = ('clamped', (1, [0, 0, 0]))
        bc3 = ('clamped', (1, [[0, 0]]))
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc1, True)
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc2, True)
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc3, True)

        # 周期性条件要求 y[-1] 必须等于 y[0]
        assert_raises(ValueError, CubicSpline, x, y, 0, 'periodic', True)
# 定义一个测试函数，用于验证 CubicHermiteSpline 类的正确性
def test_CubicHermiteSpline_correctness():
    # 定义样本点 x, y 和导数值 dy/dx
    x = [0, 2, 7]
    y = [-1, 2, 3]
    dydx = [0, 3, 7]
    # 创建 CubicHermiteSpline 对象 s
    s = CubicHermiteSpline(x, y, dydx)
    # 断言插值函数在样本点处的值接近于给定的 y 值，允许的相对误差是 1e-15
    assert_allclose(s(x), y, rtol=1e-15)
    # 断言插值函数在样本点处的导数值接近于给定的导数值 dy/dx，允许的相对误差是 1e-15
    assert_allclose(s(x, 1), dydx, rtol=1e-15)


# 定义一个测试函数，用于验证 CubicHermiteSpline 类的错误处理
def test_CubicHermiteSpline_error_handling():
    # 定义样本点 x, y 和导数值 dy/dx
    x = [1, 2, 3]
    y = [0, 3, 5]
    dydx = [1, -1, 2, 3]
    # 断言当传入不合法的参数时，抛出 ValueError 异常
    assert_raises(ValueError, CubicHermiteSpline, x, y, dydx)

    # 定义包含 NaN 值的导数值 dy/dx
    dydx_with_nan = [1, 0, np.nan]
    # 断言当导数值包含 NaN 时，抛出 ValueError 异常
    assert_raises(ValueError, CubicHermiteSpline, x, y, dydx_with_nan)


# 定义一个测试函数，用于验证根的计算方法在特定情况下的正确性
def test_roots_extrapolate_gh_11185():
    # 定义样本点 x, y 和导数值 dy/dx
    x = np.array([0.001, 0.002])
    y = np.array([1.66066935e-06, 1.10410807e-06])
    dy = np.array([-1.60061854, -1.600619])
    # 创建 CubicHermiteSpline 对象 p
    p = CubicHermiteSpline(x, y, dy)

    # 对于具有单一区间的多项式，设置 extrapolate=True 时应返回所有三个实根
    r = p.roots(extrapolate=True)
    # 断言多项式对象 p 的系数矩阵中列的数量为 1
    assert_equal(p.c.shape[1], 1)
    # 断言计算出的根的数量为 3
    assert_equal(r.size, 3)


# 定义一个测试类 TestZeroSizeArrays，用于测试零尺寸数组的情况
class TestZeroSizeArrays:
    # 用于回归测试 gh-17241：当 y.size == 0 时，确保 CubicSpline 等类不会出现段错误
    # 下面两个方法几乎相同，但有所不同：
    # 其中一个适用于具有 `bc_type` 参数的对象（CubicSpline），
    # 另一个适用于不具有 `bc_type` 参数的对象（Pchip, Akima1D）

    @pytest.mark.parametrize('y', [np.zeros((10, 0, 5)),
                                   np.zeros((10, 5, 0))])
    @pytest.mark.parametrize('bc_type',
                             ['not-a-knot', 'periodic', 'natural', 'clamped'])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('cls', [make_interp_spline, CubicSpline])
    # 定义测试函数 test_zero_size，用于测试零尺寸数组的情况
    def test_zero_size(self, cls, y, bc_type, axis):
        # 定义样本点 x 和测试点 xval
        x = np.arange(10)
        xval = np.arange(3)

        # 创建插值对象 obj
        obj = cls(x, y, bc_type=bc_type)
        # 断言在测试点 xval 处的插值结果大小为 0
        assert obj(xval).size == 0
        # 断言在测试点 xval 处的插值结果形状与预期相符
        assert obj(xval).shape == xval.shape + y.shape[1:]

        # 创建插值对象 obj，指定非默认的 axis
        yt = np.moveaxis(y, 0, axis)  # 如果 axis=1，则从 (10, 0, 5) 移动到 (0, 10, 5)

        obj = cls(x, yt, bc_type=bc_type, axis=axis)
        sh = yt.shape[:axis] + (xval.size, ) + yt.shape[axis+1:]
        # 断言在测试点 xval 处的插值结果大小为 0
        assert obj(xval).size == 0
        # 断言在测试点 xval 处的插值结果形状与预期相符
        assert obj(xval).shape == sh

    @pytest.mark.parametrize('y', [np.zeros((10, 0, 5)),
                                   np.zeros((10, 5, 0))])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('cls', [PchipInterpolator, Akima1DInterpolator])
    # 定义测试函数 test_zero_size_2，用于测试零尺寸数组的情况
    def test_zero_size_2(self, cls, y, axis):
        # 定义样本点 x 和测试点 xval
        x = np.arange(10)
        xval = np.arange(3)

        # 创建插值对象 obj
        obj = cls(x, y)
        # 断言在测试点 xval 处的插值结果大小为 0
        assert obj(xval).size == 0
        # 断言在测试点 xval 处的插值结果形状与预期相符
        assert obj(xval).shape == xval.shape + y.shape[1:]

        # 创建插值对象 obj，指定非默认的 axis
        yt = np.moveaxis(y, 0, axis)  # 如果 axis=1，则从 (10, 0, 5) 移动到 (0, 10, 5)

        obj = cls(x, yt, axis=axis)
        sh = yt.shape[:axis] + (xval.size, ) + yt.shape[axis+1:]
        # 断言在测试点 xval 处的插值结果大小为 0
        assert obj(xval).size == 0
        # 断言在测试点 xval 处的插值结果形状与预期相符
        assert obj(xval).shape == sh
```