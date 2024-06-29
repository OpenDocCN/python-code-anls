# `.\numpy\numpy\polynomial\tests\test_laguerre.py`

```py
"""Tests for laguerre module.

"""
# 导入 reduce 函数
from functools import reduce

# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 numpy.polynomial.laguerre 模块，并使用别名 lag
import numpy.polynomial.laguerre as lag
# 导入 numpy.polynomial.polynomial 模块中的 polyval 函数
from numpy.polynomial.polynomial import polyval
# 导入 numpy.testing 中的多个断言函数
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )

# 定义 Laguerre 多项式 L0 到 L6
L0 = np.array([1])/1
L1 = np.array([1, -1])/1
L2 = np.array([2, -4, 1])/2
L3 = np.array([6, -18, 9, -1])/6
L4 = np.array([24, -96, 72, -16, 1])/24
L5 = np.array([120, -600, 600, -200, 25, -1])/120
L6 = np.array([720, -4320, 5400, -2400, 450, -36, 1])/720

# 将 Laguerre 多项式存储在列表 Llist 中
Llist = [L0, L1, L2, L3, L4, L5, L6]

# 定义 trim 函数，使用 lag.lagtrim 函数对输入进行修剪
def trim(x):
    return lag.lagtrim(x, tol=1e-6)

# 定义 TestConstants 类
class TestConstants:

    # 测试 lagdomain 是否等于 [0, 1]
    def test_lagdomain(self):
        assert_equal(lag.lagdomain, [0, 1])

    # 测试 lagzero 是否等于 [0]
    def test_lagzero(self):
        assert_equal(lag.lagzero, [0])

    # 测试 lagone 是否等于 [1]
    def test_lagone(self):
        assert_equal(lag.lagone, [1])

    # 测试 lagx 是否等于 [1, -1]
    def test_lagx(self):
        assert_equal(lag.lagx, [1, -1])

# 定义 TestArithmetic 类
class TestArithmetic:

    # 定义 x 数组，包含从 -3 到 3 的 100 个均匀分布的数值
    x = np.linspace(-3, 3, 100)

    # 测试 lagadd 函数的行为
    def test_lagadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = lag.lagadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    # 测试 lagsub 函数的行为
    def test_lagsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = lag.lagsub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    # 测试 lagmulx 函数的行为
    def test_lagmulx(self):
        assert_equal(lag.lagmulx([0]), [0])
        assert_equal(lag.lagmulx([1]), [1, -1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [-i, 2*i + 1, -(i + 1)]
            assert_almost_equal(lag.lagmulx(ser), tgt)

    # 测试 lagmul 函数的行为
    def test_lagmul(self):
        # 检查结果的数值
        for i in range(5):
            pol1 = [0]*i + [1]
            val1 = lag.lagval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0]*j + [1]
                val2 = lag.lagval(self.x, pol2)
                pol3 = lag.lagmul(pol1, pol2)
                val3 = lag.lagval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1*val2, err_msg=msg)

    # 测试 lagdiv 函数的行为
    def test_lagdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = lag.lagadd(ci, cj)
                quo, rem = lag.lagdiv(tgt, ci)
                res = lag.lagadd(lag.lagmul(quo, ci), rem)
                assert_almost_equal(trim(res), trim(tgt), err_msg=msg)
    # 定义一个测试方法 test_lagpow，用于测试 lagpow 函数的功能
    def test_lagpow(self):
        # 外层循环控制 i 的取值范围在 [0, 5)
        for i in range(5):
            # 内层循环控制 j 的取值范围在 [0, 5)
            for j in range(5):
                # 生成当前测试消息的字符串，显示当前 i 和 j 的值
                msg = f"At i={i}, j={j}"
                # 创建一个长度为 i+1 的 NumPy 数组 c，包含从 0 到 i 的整数
                c = np.arange(i + 1)
                # 使用 lagmul 函数对 c 数组进行 j 次拉格朗日多项式乘法，初始化为 [1]
                tgt = reduce(lag.lagmul, [c]*j, np.array([1]))
                # 调用 lagpow 函数计算 c 的 j 次幂的拉格朗日多项式
                res = lag.lagpow(c, j) 
                # 使用 assert_equal 函数断言 res 和 tgt 的结果在修剪（trim）后相等，如果不相等，显示错误消息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)
class TestEvaluation:
    # 定义一维多项式系数：1 + 2*x + 3*x**2
    c1d = np.array([9., -14., 6.])
    # 使用 Einstein 求和约定创建二维系数数组
    c2d = np.einsum('i,j->ij', c1d, c1d)
    # 使用 Einstein 求和约定创建三维系数数组
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # 生成在区间[-1, 1)内的随机值
    x = np.random.random((3, 5))*2 - 1
    # 计算多项式在随机值 x 上的取值
    y = polyval(x, [1., 2., 3.])

    def test_lagval(self):
        # 检查空输入的情况
        assert_equal(lag.lagval([], [1]).size, 0)

        # 检查正常输入的情况
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Llist]
        for i in range(7):
            msg = f"At i={i}"
            tgt = y[i]
            res = lag.lagval(x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # 检查结果形状是否保持不变
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(lag.lagval(x, [1]).shape, dims)
            assert_equal(lag.lagval(x, [1, 0]).shape, dims)
            assert_equal(lag.lagval(x, [1, 0, 0]).shape, dims)

    def test_lagval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises(ValueError, lag.lagval2d, x1, x2[:2], self.c2d)

        # 测试值计算
        tgt = y1*y2
        res = lag.lagval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 测试结果形状
        z = np.ones((2, 3))
        res = lag.lagval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_lagval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises(ValueError, lag.lagval3d, x1, x2, x3[:2], self.c3d)

        # 测试值计算
        tgt = y1*y2*y3
        res = lag.lagval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # 测试结果形状
        z = np.ones((2, 3))
        res = lag.lagval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_laggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试值计算
        tgt = np.einsum('i,j->ij', y1, y2)
        res = lag.laggrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 测试结果形状
        z = np.ones((2, 3))
        res = lag.laggrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3)*2)

    def test_laggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试值计算
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = lag.laggrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # 测试结果形状
        z = np.ones((2, 3))
        res = lag.laggrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3)*3)


class TestIntegral:
    # 定义一个测试函数 `test_lagint_axis`，用于测试 lagint 函数在不同轴向上的表现
    def test_lagint_axis(self):
        # 检查 axis 关键字参数的工作情况

        # 创建一个 3x4 的随机数数组
        c2d = np.random.random((3, 4))

        # 对数组的每一列应用 lag.lagint 函数，并将结果堆叠为一个新的转置后的数组
        tgt = np.vstack([lag.lagint(c) for c in c2d.T]).T
        # 在 axis=0 的情况下调用 lag.lagint 函数
        res = lag.lagint(c2d, axis=0)
        # 断言 res 和 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 对数组的每一行应用 lag.lagint 函数，并将结果堆叠为一个新的数组
        tgt = np.vstack([lag.lagint(c) for c in c2d])
        # 在 axis=1 的情况下调用 lag.lagint 函数
        res = lag.lagint(c2d, axis=1)
        # 断言 res 和 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 对数组的每一行应用 lag.lagint 函数，并指定 k=3
        tgt = np.vstack([lag.lagint(c, k=3) for c in c2d])
        # 在 axis=1 的情况下调用 lag.lagint 函数，同时指定 k=3
        res = lag.lagint(c2d, k=3, axis=1)
        # 断言 res 和 tgt 几乎相等
        assert_almost_equal(res, tgt)
class TestDerivative:

    def test_lagder(self):
        # 检查异常情况
        assert_raises(TypeError, lag.lagder, [0], .5)
        assert_raises(ValueError, lag.lagder, [0], -1)

        # 检查零阶导数不做任何操作
        for i in range(5):
            tgt = [0]*i + [1]
            res = lag.lagder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # 检查导数与积分的逆过程
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = lag.lagder(lag.lagint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # 检查带有缩放的导数计算
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = lag.lagder(lag.lagint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_lagder_axis(self):
        # 检查轴关键字的工作方式
        c2d = np.random.random((3, 4))

        tgt = np.vstack([lag.lagder(c) for c in c2d.T]).T
        res = lag.lagder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([lag.lagder(c) for c in c2d])
        res = lag.lagder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # 在区间[-1, 1)中生成随机值
    x = np.random.random((3, 5))*2 - 1

    def test_lagvander(self):
        # 检查 1 维 x 的情况
        x = np.arange(3)
        v = lag.lagvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], lag.lagval(x, coef))

        # 检查 2 维 x 的情况
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = lag.lagvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], lag.lagval(x, coef))

    def test_lagvander2d(self):
        # 同时测试 lagval2d 对于非方形系数数组的情况
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = lag.lagvander2d(x1, x2, [1, 2])
        tgt = lag.lagval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = lag.lagvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_lagvander3d(self):
        # 同时测试 lagval3d 对于非方形系数数组的情况
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = lag.lagvander3d(x1, x2, x3, [1, 2, 3])
        tgt = lag.lagval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = lag.lagvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:
    pass
    def test_lagfit(self):
        # 定义一个测试函数 f(x)，计算 x*(x - 1)*(x - 2)
        def f(x):
            return x*(x - 1)*(x - 2)

        # 测试异常情况
        assert_raises(ValueError, lag.lagfit, [1], [1], -1)
        assert_raises(TypeError, lag.lagfit, [[1]], [1], 0)
        assert_raises(TypeError, lag.lagfit, [], [1], 0)
        assert_raises(TypeError, lag.lagfit, [1], [[[1]]], 0)
        assert_raises(TypeError, lag.lagfit, [1, 2], [1], 0)
        assert_raises(TypeError, lag.lagfit, [1], [1, 2], 0)
        assert_raises(TypeError, lag.lagfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, lag.lagfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, lag.lagfit, [1], [1], [-1,])
        assert_raises(ValueError, lag.lagfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, lag.lagfit, [1], [1], [])

        # 测试拟合
        x = np.linspace(0, 2)
        y = f(x)
        # 拟合阶数为 3 的 Lagrange 插值多项式
        coef3 = lag.lagfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(lag.lagval(x, coef3), y)
        # 拟合指定节点的 Lagrange 插值多项式
        coef3 = lag.lagfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(lag.lagval(x, coef3), y)
        # 拟合阶数为 4 的 Lagrange 插值多项式
        coef4 = lag.lagfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(lag.lagval(x, coef4), y)
        # 拟合指定节点的 Lagrange 插值多项式
        coef4 = lag.lagfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(lag.lagval(x, coef4), y)
        # 对二维数据进行拟合
        coef2d = lag.lagfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = lag.lagfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # 测试加权拟合
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = lag.lagfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = lag.lagfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        # 对二维数据进行加权拟合
        wcoef2d = lag.lagfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = lag.lagfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # 测试使用复数值进行拟合
        x = [1, 1j, -1, -1j]
        assert_almost_equal(lag.lagfit(x, x, 1), [1, -1])
        assert_almost_equal(lag.lagfit(x, x, [0, 1]), [1, -1])
class TestCompanion:

    def test_raises(self):
        # 检查 lagcompanion 函数在空列表输入时是否引发 ValueError 异常
        assert_raises(ValueError, lag.lagcompanion, [])
        # 检查 lagcompanion 函数在单元素列表输入时是否引发 ValueError 异常
        assert_raises(ValueError, lag.lagcompanion, [1])

    def test_dimensions(self):
        # 测试 lagcompanion 函数生成的伴随矩阵的维度是否正确
        for i in range(1, 5):
            coef = [0]*i + [1]
            assert_(lag.lagcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        # 检查 lagcompanion 函数对 [1, 2] 输入生成的伴随矩阵的第一个元素是否为 1.5
        assert_(lag.lagcompanion([1, 2])[0, 0] == 1.5)


class TestGauss:

    def test_100(self):
        # 获取 Laguerre-Gauss 积分的节点和权重
        x, w = lag.laggauss(100)

        # 测试正交性。需要注意结果需要归一化，否则由于拉盖尔函数等快速增长函数可能导致的巨大值会非常令人困惑。
        v = lag.lagvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # 检查积分结果是否为 1
        tgt = 1.0
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_lagfromroots(self):
        # 测试 lagfromroots 函数对空根列表的处理，期望结果为 [1]
        res = lag.lagfromroots([])
        assert_almost_equal(trim(res), [1])

        # 测试 lagfromroots 函数对包含多个根的情况的处理
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = lag.lagfromroots(roots)
            res = lag.lagval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(lag.lag2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_lagroots(self):
        # 检查 lagroots 函数对不同系数输入的根的计算结果
        assert_almost_equal(lag.lagroots([1]), [])
        assert_almost_equal(lag.lagroots([0, 1]), [1])
        for i in range(2, 5):
            tgt = np.linspace(0, 3, i)
            res = lag.lagroots(lag.lagfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_lagtrim(self):
        coef = [2, -1, 1, 0]

        # 测试异常情况：lagtrim 函数对负数输入是否引发 ValueError 异常
        assert_raises(ValueError, lag.lagtrim, coef, -1)

        # 测试结果：lagtrim 函数对不同参数的截断效果
        assert_equal(lag.lagtrim(coef), coef[:-1])
        assert_equal(lag.lagtrim(coef, 1), coef[:-3])
        assert_equal(lag.lagtrim(coef, 2), [0])

    def test_lagline(self):
        # 检查 lagline 函数对给定参数的线性多项式的计算结果
        assert_equal(lag.lagline(3, 4), [7, -4])

    def test_lag2poly(self):
        # 检查 lag2poly 函数对不同阶数的拉盖尔多项式的转换结果
        for i in range(7):
            assert_almost_equal(lag.lag2poly([0]*i + [1]), Llist[i])

    def test_poly2lag(self):
        # 检查 poly2lag 函数对不同阶数的多项式的转换结果
        for i in range(7):
            assert_almost_equal(lag.poly2lag(Llist[i]), [0]*i + [1])

    def test_weight(self):
        # 检查 lagweight 函数对给定输入的权重计算结果
        x = np.linspace(0, 10, 11)
        tgt = np.exp(-x)
        res = lag.lagweight(x)
        assert_almost_equal(res, tgt)
```