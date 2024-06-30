# `D:\src\scipysrc\scipy\scipy\special\tests\test_spherical_bessel.py`

```
# spherical Bessel 函数的测试。

import numpy as np  # 导入 NumPy 库，并简写为 np
from numpy.testing import (assert_almost_equal, assert_allclose,
                           assert_array_almost_equal, suppress_warnings)  # 导入测试相关的函数和上下文管理器

import pytest  # 导入 pytest 测试框架
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi  # 导入一些数学函数和常数

from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn  # 导入 scipy 中的 spherical 函数
from scipy.integrate import quad  # 导入 scipy 中的积分函数 quad


class TestSphericalJn:
    def test_spherical_jn_exact(self):
        # 精确的 spherical_jn(n, x) 测试
        # 参考文档：https://dlmf.nist.gov/10.49.E3
        # 注意：该精确表达式只对于较小的 n 或 z >> n 是数值稳定的。
        x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
        assert_allclose(spherical_jn(2, x),
                        (-1/x + 3/x**3)*sin(x) - 3/x**2*cos(x))

    def test_spherical_jn_recurrence_complex(self):
        # 复数情况下的 spherical_jn(n, x) 递推关系测试
        # 参考文档：https://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        assert_allclose(spherical_jn(n - 1, x) + spherical_jn(n + 1, x),
                        (2*n + 1)/x*spherical_jn(n, x))

    def test_spherical_jn_recurrence_real(self):
        # 实数情况下的 spherical_jn(n, x) 递推关系测试
        # 参考文档：https://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        assert_allclose(spherical_jn(n - 1, x) + spherical_jn(n + 1, x),
                        (2*n + 1)/x*spherical_jn(n, x))

    def test_spherical_jn_inf_real(self):
        # 实数情况下的 spherical_jn(n, x) 在无穷处的测试
        # 参考文档：https://dlmf.nist.gov/10.52.E3
        n = 6
        x = np.array([-inf, inf])
        assert_allclose(spherical_jn(n, x), np.array([0, 0]))

    def test_spherical_jn_inf_complex(self):
        # 复数情况下的 spherical_jn(n, x) 在无穷处的测试
        # 参考文档：https://dlmf.nist.gov/10.52.E3
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in multiply")
            assert_allclose(spherical_jn(n, x), np.array([0, 0, inf*(1+1j)]))

    def test_spherical_jn_large_arg_1(self):
        # spherical_jn(n, x) 在大参数下的测试，情况 1
        # 参考问题：https://github.com/scipy/scipy/issues/2165
        # 参考值是用 mpmath 计算的
        assert_allclose(spherical_jn(2, 3350.507), -0.00029846226538040747)

    def test_spherical_jn_large_arg_2(self):
        # spherical_jn(n, x) 在大参数下的测试，情况 2
        # 参考问题：https://github.com/scipy/scipy/issues/1641
        # 参考值是用 mpmath 计算的
        assert_allclose(spherical_jn(2, 10000), 3.0590002633029811e-05)

    def test_spherical_jn_at_zero(self):
        # spherical_jn(n, x) 在 x=0 处的测试
        # 参考文档：https://dlmf.nist.gov/10.52.E1
        # 但需要注意 n=0 是一个特例：j0 = sin(x)/x -> 1
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        assert_allclose(spherical_jn(n, x), np.array([1, 0, 0, 0, 0, 0]))


class TestSphericalYn:
    def test_spherical_yn_exact(self):
        # 球面贝塞尔函数 Y_n(x) 的精确计算测试
        # 参考链接：https://dlmf.nist.gov/10.49.E5
        # 注意：精确表达式仅在 n 较小或 z >> n 时数值稳定。
        x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
        assert_allclose(spherical_yn(2, x),
                        (1/x - 3/x**3)*cos(x) - 3/x**2*sin(x))

    def test_spherical_yn_recurrence_real(self):
        # 球面贝塞尔函数 Y_n(x) 的递推关系（实数参数）测试
        # 参考链接：https://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1,x),
                        (2*n + 1)/x*spherical_yn(n, x))

    def test_spherical_yn_recurrence_complex(self):
        # 球面贝塞尔函数 Y_n(x) 的递推关系（复数参数）测试
        # 参考链接：https://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1, x),
                        (2*n + 1)/x*spherical_yn(n, x))

    def test_spherical_yn_inf_real(self):
        # 球面贝塞尔函数 Y_n(x) 在实数 x 为无穷大时的测试
        # 参考链接：https://dlmf.nist.gov/10.52.E3
        n = 6
        x = np.array([-inf, inf])
        assert_allclose(spherical_yn(n, x), np.array([0, 0]))

    def test_spherical_yn_inf_complex(self):
        # 球面贝塞尔函数 Y_n(x) 在复数 x 为无穷大时的测试
        # 参考链接：https://dlmf.nist.gov/10.52.E3
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in multiply")
            assert_allclose(spherical_yn(n, x), np.array([0, 0, inf*(1+1j)]))

    def test_spherical_yn_at_zero(self):
        # 球面贝塞尔函数 Y_n(x) 在 x=0 处的测试
        # 参考链接：https://dlmf.nist.gov/10.52.E2
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        assert_allclose(spherical_yn(n, x), np.full(n.shape, -inf))

    def test_spherical_yn_at_zero_complex(self):
        # 球面贝塞尔函数 Y_n(x) 在复数 x=0 处的测试
        # 与 numpy 一致：
        # >>> -np.cos(0)/0
        # -inf
        # >>> -np.cos(0+0j)/(0+0j)
        # (-inf + nan*j)
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0 + 0j
        assert_allclose(spherical_yn(n, x), np.full(n.shape, nan))
class TestSphericalJnYnCrossProduct:
    def test_spherical_jn_yn_cross_product_1(self):
        # 定义要测试的n和x数组，这些是测试数据
        n = np.array([1, 5, 8])
        x = np.array([0.1, 1, 10])
        # 计算左侧表达式：spherical_jn(n+1, x) * spherical_yn(n, x) - spherical_jn(n, x) * spherical_yn(n+1, x)
        left = (spherical_jn(n + 1, x) * spherical_yn(n, x) -
                spherical_jn(n, x) * spherical_yn(n + 1, x))
        # 计算右侧参考值：1/x**2
        right = 1/x**2
        # 使用 assert_allclose 函数断言左右两侧的近似程度
        assert_allclose(left, right)

    def test_spherical_jn_yn_cross_product_2(self):
        # 定义要测试的n和x数组，这些是测试数据
        n = np.array([1, 5, 8])
        x = np.array([0.1, 1, 10])
        # 计算左侧表达式：spherical_jn(n+2, x) * spherical_yn(n, x) - spherical_jn(n, x) * spherical_yn(n+2, x)
        left = (spherical_jn(n + 2, x) * spherical_yn(n, x) -
                spherical_jn(n, x) * spherical_yn(n + 2, x))
        # 计算右侧参考值：(2*n + 3)/x**3
        right = (2*n + 3)/x**3
        # 使用 assert_allclose 函数断言左右两侧的近似程度
        assert_allclose(left, right)


class TestSphericalIn:
    def test_spherical_in_exact(self):
        # 定义要测试的x数组，这是测试数据
        x = np.array([0.12, 1.23, 12.34, 123.45])
        # 使用 assert_allclose 函数断言计算结果与参考值的近似程度
        assert_allclose(spherical_in(2, x),
                        (1/x + 3/x**3)*sinh(x) - 3/x**2*cosh(x))

    def test_spherical_in_recurrence_real(self):
        # 定义要测试的n和x，这是测试数据
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        # 使用 assert_allclose 函数断言计算结果与参考值的近似程度
        assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1, x),
                        (2*n + 1)/x*spherical_in(n, x))

    def test_spherical_in_recurrence_complex(self):
        # 定义要测试的n和x，这是测试数据
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        # 使用 assert_allclose 函数断言计算结果与参考值的近似程度
        assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1, x),
                        (2*n + 1)/x*spherical_in(n, x))

    def test_spherical_in_inf_real(self):
        # 定义要测试的n和x，这是测试数据
        n = 5
        x = np.array([-inf, inf])
        # 使用 assert_allclose 函数断言计算结果与参考值的近似程度
        assert_allclose(spherical_in(n, x), np.array([-inf, inf]))

    def test_spherical_in_inf_complex(self):
        # 定义要测试的n和x，这是测试数据
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
        # 使用 assert_allclose 函数断言计算结果与参考值的近似程度
        assert_allclose(spherical_in(n, x), np.array([-inf, inf, nan]))

    def test_spherical_in_at_zero(self):
        # 定义要测试的n和x，这是测试数据
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        # 使用 assert_allclose 函数断言计算结果与参考值的近似程度
        assert_allclose(spherical_in(n, x), np.array([1, 0, 0, 0, 0, 0]))


class TestSphericalKn:
    def test_spherical_kn_exact(self):
        # 定义要测试的x数组，这是测试数据
        x = np.array([0.12, 1.23, 12.34, 123.45])
        # 使用 assert_allclose 函数断言计算结果与参考值的近似程度
        assert_allclose(spherical_kn(2, x),
                        pi/2*exp(-x)*(1/x + 3/x**2 + 3/x**3))
    def test_spherical_kn_recurrence_real(self):
        # 测试球形 Bessel 函数的实数递推关系
        n = np.array([1, 2, 3, 7, 12])  # 定义整数数组 n
        x = 0.12  # 定义实数 x
        assert_allclose(
            (-1)**(n - 1)*spherical_kn(n - 1, x) - (-1)**(n + 1)*spherical_kn(n + 1, x),
            (-1)**n*(2*n + 1)/x*spherical_kn(n, x)  # 断言球形 Bessel 函数的递推关系
        )

    def test_spherical_kn_recurrence_complex(self):
        # 测试球形 Bessel 函数的复数递推关系
        n = np.array([1, 2, 3, 7, 12])  # 定义整数数组 n
        x = 1.1 + 1.5j  # 定义复数 x
        assert_allclose(
            (-1)**(n - 1)*spherical_kn(n - 1, x) - (-1)**(n + 1)*spherical_kn(n + 1, x),
            (-1)**n*(2*n + 1)/x*spherical_kn(n, x)  # 断言球形 Bessel 函数的递推关系
        )

    def test_spherical_kn_inf_real(self):
        # 测试球形 Bessel 函数在实数无穷远处的行为
        n = 5  # 定义整数 n
        x = np.array([-inf, inf])  # 定义实数数组 x，包括负无穷和正无穷
        assert_allclose(spherical_kn(n, x), np.array([-inf, 0]))  # 断言球形 Bessel 函数在无穷远处的值

    def test_spherical_kn_inf_complex(self):
        # 测试球形 Bessel 函数在复数无穷远处的行为
        # 复数无穷远处的行为取决于实部的符号：如果 Re(z) >= 0，则极限为 0；如果 Re(z) < 0，则为 z*inf。
        # 这种区别无法捕获，因此返回 NaN。
        n = 7  # 定义整数 n
        x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])  # 定义复数数组 x，包括负无穷、正无穷和复数无穷远处的情况
        assert_allclose(spherical_kn(n, x), np.array([-inf, 0, nan]))  # 断言球形 Bessel 函数在复数无穷远处的值

    def test_spherical_kn_at_zero(self):
        # 测试球形 Bessel 函数在零点处的行为
        n = np.array([0, 1, 2, 5, 10, 100])  # 定义整数数组 n
        x = 0  # 定义零
        assert_allclose(spherical_kn(n, x), np.full(n.shape, inf))  # 断言球形 Bessel 函数在零点处的值为正无穷

    def test_spherical_kn_at_zero_complex(self):
        # 测试球形 Bessel 函数在复数零点处的行为
        n = np.array([0, 1, 2, 5, 10, 100])  # 定义整数数组 n
        x = 0 + 0j  # 定义复数零
        assert_allclose(spherical_kn(n, x), np.full(n.shape, nan))  # 断言球形 Bessel 函数在复数零点处的值为 NaN
class SphericalDerivativesTestCase:
    # 球面导数测试用例的基类

    def fundamental_theorem(self, n, a, b):
        # 计算函数的基本定理，计算从 a 到 b 的积分并与函数差的误差容限比较
        integral, tolerance = quad(lambda z: self.df(n, z), a, b)
        assert_allclose(integral,
                        self.f(n, b) - self.f(n, a),
                        atol=tolerance)

    @pytest.mark.slow
    def test_fundamental_theorem_0(self):
        # 测试基本定理在 n=0 时的情况，区间为 [3.0, 15.0]
        self.fundamental_theorem(0, 3.0, 15.0)

    @pytest.mark.slow
    def test_fundamental_theorem_7(self):
        # 测试基本定理在 n=7 时的情况，区间为 [0.5, 1.2]
        self.fundamental_theorem(7, 0.5, 1.2)


class TestSphericalJnDerivatives(SphericalDerivativesTestCase):
    # 测试球面贝塞尔函数 J_n 的导数

    def f(self, n, z):
        # 返回球面贝塞尔函数 J_n(z)
        return spherical_jn(n, z)

    def df(self, n, z):
        # 返回球面贝塞尔函数 J_n'(z)
        return spherical_jn(n, z, derivative=True)

    def test_spherical_jn_d_zero(self):
        # 测试球面贝塞尔函数 J_n 在 z=0 处的导数
        n = np.array([0, 1, 2, 3, 7, 15])
        assert_allclose(spherical_jn(n, 0, derivative=True),
                        np.array([0, 1/3, 0, 0, 0, 0]))


class TestSphericalYnDerivatives(SphericalDerivativesTestCase):
    # 测试球面贝塞尔函数 Y_n 的导数

    def f(self, n, z):
        # 返回球面贝塞尔函数 Y_n(z)
        return spherical_yn(n, z)

    def df(self, n, z):
        # 返回球面贝塞尔函数 Y_n'(z)
        return spherical_yn(n, z, derivative=True)


class TestSphericalInDerivatives(SphericalDerivativesTestCase):
    # 测试球面修正贝塞尔函数 I_n 的导数

    def f(self, n, z):
        # 返回球面修正贝塞尔函数 I_n(z)
        return spherical_in(n, z)

    def df(self, n, z):
        # 返回球面修正贝塞尔函数 I_n'(z)
        return spherical_in(n, z, derivative=True)

    def test_spherical_in_d_zero(self):
        # 测试球面修正贝塞尔函数 I_n 在 z=0 处的导数
        n = np.array([0, 1, 2, 3, 7, 15])
        spherical_in(n, 0, derivative=False)
        assert_allclose(spherical_in(n, 0, derivative=True),
                        np.array([0, 1/3, 0, 0, 0, 0]))


class TestSphericalKnDerivatives(SphericalDerivativesTestCase):
    # 测试球面修正贝塞尔函数 K_n 的导数

    def f(self, n, z):
        # 返回球面修正贝塞尔函数 K_n(z)
        return spherical_kn(n, z)

    def df(self, n, z):
        # 返回球面修正贝塞尔函数 K_n'(z)
        return spherical_kn(n, z, derivative=True)


class TestSphericalOld:
    # 这些测试来自 test_basic.py 中的 TestSpherical 类，修改为使用 spherical_* 而不是 sph_*，其余未变。

    def test_sph_in(self):
        # 这个测试重现了 test_basic.TestSpherical.test_sph_in。
        i1n = np.empty((2,2))
        x = 0.2

        i1n[0][0] = spherical_in(0, x)
        i1n[0][1] = spherical_in(1, x)
        i1n[1][0] = spherical_in(0, x, derivative=True)
        i1n[1][1] = spherical_in(1, x, derivative=True)

        inp0 = (i1n[0][1])
        inp1 = (i1n[0][0] - 2.0/0.2 * i1n[0][1])
        assert_array_almost_equal(i1n[0], np.array([1.0066800127054699381,
                                                    0.066933714568029540839]), 12)
        assert_array_almost_equal(i1n[1], [inp0, inp1], 12)
    def test_sph_in_kn_order0(self):
        # 设置变量 x 为浮点数 1.0
        x = 1.
        # 创建一个形状为 (2,) 的空 NumPy 数组 sph_i0
        sph_i0 = np.empty((2,))
        # 计算零阶球面贝塞尔函数 spherical_in(0, x)，存入 sph_i0 数组的第一个位置
        sph_i0[0] = spherical_in(0, x)
        # 计算零阶球面贝塞尔函数的导数 spherical_in(0, x, derivative=True)，存入 sph_i0 数组的第二个位置
        sph_i0[1] = spherical_in(0, x, derivative=True)
        # 预期的 sph_i0 数组的值，使用 numpy 数组构造
        sph_i0_expected = np.array([np.sinh(x)/x,
                                    np.cosh(x)/x-np.sinh(x)/x**2])
        # 使用 assert_array_almost_equal 函数检查 sph_i0 和 sph_i0_expected 的近似相等性
        assert_array_almost_equal(r_[sph_i0], sph_i0_expected)

        # 创建一个形状为 (2,) 的空 NumPy 数组 sph_k0
        sph_k0 = np.empty((2,))
        # 计算零阶球面贝塞尔函数 spherical_kn(0, x)，存入 sph_k0 数组的第一个位置
        sph_k0[0] = spherical_kn(0, x)
        # 计算零阶球面贝塞尔函数的导数 spherical_kn(0, x, derivative=True)，存入 sph_k0 数组的第二个位置
        sph_k0[1] = spherical_kn(0, x, derivative=True)
        # 预期的 sph_k0 数组的值，使用 numpy 数组构造
        sph_k0_expected = np.array([0.5*pi*exp(-x)/x,
                                    -0.5*pi*exp(-x)*(1/x+1/x**2)])
        # 使用 assert_array_almost_equal 函数检查 sph_k0 和 sph_k0_expected 的近似相等性
        assert_array_almost_equal(r_[sph_k0], sph_k0_expected)

    def test_sph_jn(self):
        # 创建一个形状为 (2,3) 的空 NumPy 数组 s1
        s1 = np.empty((2,3))
        # 设置变量 x 为 0.2
        x = 0.2

        # 计算不同阶数的球面贝塞尔函数值，存入 s1 数组的对应位置
        s1[0][0] = spherical_jn(0, x)
        s1[0][1] = spherical_jn(1, x)
        s1[0][2] = spherical_jn(2, x)
        # 计算不同阶数的球面贝塞尔函数导数值，存入 s1 数组的对应位置
        s1[1][0] = spherical_jn(0, x, derivative=True)
        s1[1][1] = spherical_jn(1, x, derivative=True)
        s1[1][2] = spherical_jn(2, x, derivative=True)

        # 计算并存储 s1[0][1] 的负值
        s10 = -s1[0][1]
        # 计算并存储 s1[0][0] - 2.0/0.2*s1[0][1]
        s11 = s1[0][0]-2.0/0.2*s1[0][1]
        # 计算并存储 s1[0][1] - 3.0/0.2*s1[0][2]
        s12 = s1[0][1]-3.0/0.2*s1[0][2]
        # 使用 assert_array_almost_equal 函数检查 s1[0] 和预期值的近似相等性，精度为 1e-12
        assert_array_almost_equal(s1[0],[0.99334665397530607731,
                                         0.066400380670322230863,
                                         0.0026590560795273856680],12)
        # 使用 assert_array_almost_equal 函数检查 s1[1] 和预期值的近似相等性，精度为 1e-12
        assert_array_almost_equal(s1[1],[s10,s11,s12],12)

    def test_sph_kn(self):
        # 创建一个形状为 (2,3) 的空 NumPy 数组 kn
        kn = np.empty((2,3))
        # 设置变量 x 为 0.2
        x = 0.2

        # 计算不同阶数的修正球面贝塞尔函数值，存入 kn 数组的对应位置
        kn[0][0] = spherical_kn(0, x)
        kn[0][1] = spherical_kn(1, x)
        kn[0][2] = spherical_kn(2, x)
        # 计算不同阶数的修正球面贝塞尔函数导数值，存入 kn 数组的对应位置
        kn[1][0] = spherical_kn(0, x, derivative=True)
        kn[1][1] = spherical_kn(1, x, derivative=True)
        kn[1][2] = spherical_kn(2, x, derivative=True)

        # 计算并存储 kn[0][1] 的负值
        kn0 = -kn[0][1]
        # 计算并存储 -kn[0][0] - 2.0/0.2*kn[0][1]
        kn1 = -kn[0][0]-2.0/0.2*kn[0][1]
        # 计算并存储 -kn[0][1] - 3.0/0.2*kn[0][2]
        kn2 = -kn[0][1]-3.0/0.2*kn[0][2]
        # 使用 assert_array_almost_equal 函数检查 kn[0] 和预期值的近似相等性，精度为 1e-12
        assert_array_almost_equal(kn[0],[6.4302962978445670140,
                                         38.581777787067402086,
                                         585.15696310385559829],12)
        # 使用 assert_array_almost_equal 函数检查 kn[1] 和预期值的近似相等性，精度为 1e-9
        assert_array_almost_equal(kn[1],[kn0,kn1,kn2],9)

    def test_sph_yn(self):
        # 计算球面贝塞尔函数 Y_n(2, 0.2) 的值，存入 sy1
        sy1 = spherical_yn(2, 0.2)
        # 计算球面贝塞尔函数 Y_n(0, 0.2) 的值，存入 sy2
        sy2 = spherical_yn(0, 0.2)
        # 使用 assert_almost_equal 函数检查 sy1 和预期值的近似相等性，精度为 1e-5
        assert_almost_equal(sy1,-377.52483,5)  # previous values in the system
        # 使用 assert_almost_equal 函数检查 sy2 和预期值的近似相等性，精度为 1e-5
        assert_almost_equal(sy2,-4.9003329,5)
        # 计算球面贝塞尔函数 Y_n(0, 0.2) 和 2*球面贝塞尔函数 Y_n(2, 0.2) 的差值除以 3，存入 sphpy
        sphpy = (spherical_yn(0, 0.2) - 2*spherical_yn(2, 0.2))/3
        # 计算球面贝塞尔函数 Y_n(1, 0.2) 的导数值，存入 sy3
        sy3 = spherical_yn(1, 0.2, derivative=True)
        # 使用 assert_almost_equal 函数检查 sy3 和 sphpy 的近似相等性，精度为 1e-4
        assert_almost_equal(sy3,sphpy,4)
```