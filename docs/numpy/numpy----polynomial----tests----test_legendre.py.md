# `.\numpy\numpy\polynomial\tests\test_legendre.py`

```
"""Tests for legendre module.

"""
# 导入所需模块和函数
from functools import reduce

import numpy as np  # 导入 NumPy 库并使用别名 np
import numpy.polynomial.legendre as leg  # 导入 Legendre 多项式相关函数并使用别名 leg
from numpy.polynomial.polynomial import polyval  # 导入多项式求值函数 polyval
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_  # 导入测试断言函数
    )

# 预定义 Legendre 多项式的前十个系数
L0 = np.array([1])
L1 = np.array([0, 1])
L2 = np.array([-1, 0, 3])/2
L3 = np.array([0, -3, 0, 5])/2
L4 = np.array([3, 0, -30, 0, 35])/8
L5 = np.array([0, 15, 0, -70, 0, 63])/8
L6 = np.array([-5, 0, 105, 0, -315, 0, 231])/16
L7 = np.array([0, -35, 0, 315, 0, -693, 0, 429])/16
L8 = np.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435])/128
L9 = np.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155])/128

# 将 Legendre 多项式系数存入列表
Llist = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]

# 定义修剪函数，用于裁剪 Legendre 多项式系数
def trim(x):
    return leg.legtrim(x, tol=1e-6)

# 定义测试类 TestConstants，用于测试 Legendre 多项式的常数定义
class TestConstants:

    def test_legdomain(self):
        assert_equal(leg.legdomain, [-1, 1])  # 检查 Legendre 多项式的定义域

    def test_legzero(self):
        assert_equal(leg.legzero, [0])  # 检查 Legendre 多项式的零多项式

    def test_legone(self):
        assert_equal(leg.legone, [1])  # 检查 Legendre 多项式的单位多项式

    def test_legx(self):
        assert_equal(leg.legx, [0, 1])  # 检查 Legendre 多项式的 x 多项式

# 定义测试类 TestArithmetic，用于测试 Legendre 多项式的算术操作
class TestArithmetic:
    x = np.linspace(-1, 1, 100)  # 在[-1, 1]区间上生成100个均匀分布的点作为测试使用

    def test_legadd(self):
        # 测试 Legendre 多项式加法
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = leg.legadd([0]*i + [1], [0]*j + [1])  # 执行 Legendre 多项式加法
                assert_equal(trim(res), trim(tgt), err_msg=msg)  # 断言结果与目标相等

    def test_legsub(self):
        # 测试 Legendre 多项式减法
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = leg.legsub([0]*i + [1], [0]*j + [1])  # 执行 Legendre 多项式减法
                assert_equal(trim(res), trim(tgt), err_msg=msg)  # 断言结果与目标相等

    def test_legmulx(self):
        # 测试 Legendre 多项式乘以 x
        assert_equal(leg.legmulx([0]), [0])  # 验证零多项式乘以 x 后为零多项式
        assert_equal(leg.legmulx([1]), [0, 1])  # 验证单位多项式乘以 x 后为 x 多项式
        for i in range(1, 5):
            tmp = 2*i + 1
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [i/tmp, 0, (i + 1)/tmp]
            assert_equal(leg.legmulx(ser), tgt)  # 验证乘以 x 的系数计算

    def test_legmul(self):
        # 测试 Legendre 多项式乘法
        for i in range(5):
            pol1 = [0]*i + [1]
            val1 = leg.legval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0]*j + [1]
                val2 = leg.legval(self.x, pol2)
                pol3 = leg.legmul(pol1, pol2)
                val3 = leg.legval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)  # 验证乘法结果的系数个数
                assert_almost_equal(val3, val1*val2, err_msg=msg)  # 验证乘法结果在测试点上的值与期望值接近
    # 定义一个测试方法，用于测试多项式的除法功能
    def test_legdiv(self):
        # 循环遍历 i 的取值范围 [0, 5)
        for i in range(5):
            # 循环遍历 j 的取值范围 [0, 5)
            for j in range(5):
                # 构造测试信息字符串，显示当前 i 和 j 的取值
                msg = f"At i={i}, j={j}"
                # 创建一个长度为 i+1 的列表，最后一个元素为 1，其余为 0
                ci = [0]*i + [1]
                # 创建一个长度为 j+1 的列表，最后一个元素为 1，其余为 0
                cj = [0]*j + [1]
                # 调用 leg.legadd 函数，计算多项式 ci 和 cj 的加法结果 tgt
                tgt = leg.legadd(ci, cj)
                # 调用 leg.legdiv 函数，计算 tgt 除以 ci 的商 quo 和余数 rem
                quo, rem = leg.legdiv(tgt, ci)
                # 使用 quo 和 rem 计算得到的结果 res
                res = leg.legadd(leg.legmul(quo, ci), rem)
                # 断言 trim(res) 等于 trim(tgt)，如果不等则输出错误信息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)
    
    # 定义一个测试方法，用于测试多项式的幂运算功能
    def test_legpow(self):
        # 循环遍历 i 的取值范围 [0, 5)
        for i in range(5):
            # 循环遍历 j 的取值范围 [0, 5)
            for j in range(5):
                # 构造测试信息字符串，显示当前 i 和 j 的取值
                msg = f"At i={i}, j={j}"
                # 创建一个长度为 i+1 的 numpy 数组，元素为 0, 1, ..., i
                c = np.arange(i + 1)
                # 使用 reduce 函数，计算多项式 c 的 j 次幂的目标值 tgt
                tgt = reduce(leg.legmul, [c]*j, np.array([1]))
                # 调用 leg.legpow 函数，计算 c 的 j 次幂的结果 res
                res = leg.legpow(c, j) 
                # 断言 trim(res) 等于 trim(tgt)，如果不等则输出错误信息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)
class TestEvaluation:
    # 定义一维多项式系数为 [2., 2., 2.]
    c1d = np.array([2., 2., 2.])
    # 创建二维数组，每个元素为两个一维数组的乘积
    c2d = np.einsum('i,j->ij', c1d, c1d)
    # 创建三维数组，每个元素为三个一维数组的乘积
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # 生成随机数矩阵 x，元素在 [-1, 1) 范围内
    x = np.random.random((3, 5))*2 - 1
    # 计算多项式在 x 上的值 y
    y = polyval(x, [1., 2., 3.])

    def test_legval(self):
        # 检查空输入情况，期望结果的大小为 0
        assert_equal(leg.legval([], [1]).size, 0)

        # 检查正常输入情况
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Llist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = leg.legval(x, [0]*i + [1])
            # 断言多项式值 res 与目标值 tgt 接近
            assert_almost_equal(res, tgt, err_msg=msg)

        # 检查形状保持不变
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            # 断言多项式值的形状与预期维度 dims 相同
            assert_equal(leg.legval(x, [1]).shape, dims)
            assert_equal(leg.legval(x, [1, 0]).shape, dims)
            assert_equal(leg.legval(x, [1, 0, 0]).shape, dims)

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况，期望引发 ValueError
        assert_raises(ValueError, leg.legval2d, x1, x2[:2], self.c2d)

        # 测试值
        tgt = y1 * y2
        res = leg.legval2d(x1, x2, self.c2d)
        # 断言多项式值 res 与目标值 tgt 接近
        assert_almost_equal(res, tgt)

        # 测试形状
        z = np.ones((2, 3))
        res = leg.legval2d(z, z, self.c2d)
        # 断言 res 的形状为 (2, 3)
        assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况，期望引发 ValueError
        assert_raises(ValueError, leg.legval3d, x1, x2, x3[:2], self.c3d)

        # 测试值
        tgt = y1 * y2 * y3
        res = leg.legval3d(x1, x2, x3, self.c3d)
        # 断言多项式值 res 与目标值 tgt 接近
        assert_almost_equal(res, tgt)

        # 测试形状
        z = np.ones((2, 3))
        res = leg.legval3d(z, z, z, self.c3d)
        # 断言 res 的形状为 (2, 3)
        assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试值
        tgt = np.einsum('i,j->ij', y1, y2)
        res = leg.leggrid2d(x1, x2, self.c2d)
        # 断言多项式值 res 与目标值 tgt 接近
        assert_almost_equal(res, tgt)

        # 测试形状
        z = np.ones((2, 3))
        res = leg.leggrid2d(z, z, self.c2d)
        # 断言 res 的形状为 (2, 3)*2
        assert_(res.shape == (2, 3)*2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试值
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = leg.leggrid3d(x1, x2, x3, self.c3d)
        # 断言多项式值 res 与目标值 tgt 接近
        assert_almost_equal(res, tgt)

        # 测试形状
        z = np.ones((2, 3))
        res = leg.leggrid3d(z, z, z, self.c3d)
        # 断言 res 的形状为 (2, 3)*3
        assert_(res.shape == (2, 3)*3)


class TestIntegral:
    # 这里开始下一个测试类，未完待续...
    # 定义测试方法，用于验证 legint 函数中的 axis 关键字是否正常工作
    def test_legint_axis(self):
        # 创建一个 3x4 的随机数组
        c2d = np.random.random((3, 4))

        # 对 c2d 的转置进行 Legendre 多项式积分，然后再次转置，形成目标数组
        tgt = np.vstack([leg.legint(c) for c in c2d.T]).T
        # 在 axis=0 方向上调用 legint 函数，得到结果 res
        res = leg.legint(c2d, axis=0)
        # 断言 res 与 tgt 的值近似相等
        assert_almost_equal(res, tgt)

        # 对 c2d 按行进行 Legendre 多项式积分，形成目标数组
        tgt = np.vstack([leg.legint(c) for c in c2d])
        # 在 axis=1 方向上调用 legint 函数，得到结果 res
        res = leg.legint(c2d, axis=1)
        # 断言 res 与 tgt 的值近似相等
        assert_almost_equal(res, tgt)

        # 对 c2d 按行进行 Legendre 多项式积分，设置 k=3，形成目标数组
        tgt = np.vstack([leg.legint(c, k=3) for c in c2d])
        # 在 axis=1 方向上调用 legint 函数，设置 k=3，得到结果 res
        res = leg.legint(c2d, k=3, axis=1)
        # 断言 res 与 tgt 的值近似相等
        assert_almost_equal(res, tgt)

    # 定义测试方法，用于验证 legint 函数在指定零阶导数情况下的输出是否正确
    def test_legint_zerointord(self):
        # 断言对于输入 (1, 2, 3) 和零阶导数，legint 函数的输出为 (1, 2, 3)
        assert_equal(leg.legint((1, 2, 3), 0), (1, 2, 3))
class TestDerivative:

    def test_legder(self):
        # 检查异常情况
        assert_raises(TypeError, leg.legder, [0], .5)
        assert_raises(ValueError, leg.legder, [0], -1)

        # 检查零阶导数不产生变化
        for i in range(5):
            tgt = [0]*i + [1]
            res = leg.legder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # 检查导数与积分的反函数关系
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = leg.legder(leg.legint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # 检查带有缩放的导数计算
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = leg.legder(leg.legint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_legder_axis(self):
        # 检查 axis 关键字的功能
        c2d = np.random.random((3, 4))

        tgt = np.vstack([leg.legder(c) for c in c2d.T]).T
        res = leg.legder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([leg.legder(c) for c in c2d])
        res = leg.legder(c2d, axis=1)
        assert_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        assert_equal(leg.legder(c, 4), [0])

class TestVander:
    # 在 [-1, 1) 内生成一些随机值
    x = np.random.random((3, 5))*2 - 1

    def test_legvander(self):
        # 检查 1 维 x 的情况
        x = np.arange(3)
        v = leg.legvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], leg.legval(x, coef))

        # 检查 2 维 x 的情况
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = leg.legvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], leg.legval(x, coef))

    def test_legvander2d(self):
        # 同时测试非方形系数数组的 polyval2d
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = leg.legvander2d(x1, x2, [1, 2])
        tgt = leg.legval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = leg.legvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_legvander3d(self):
        # 同时测试非方形系数数组的 polyval3d
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = leg.legvander3d(x1, x2, x3, [1, 2, 3])
        tgt = leg.legval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = leg.legvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))
    # 定义一个测试函数 `test_legvander_negdeg`，用于测试 `leg.legvander` 函数在负阶数时是否会引发 ValueError 异常
    def test_legvander_negdeg(self):
        # 使用 assert_raises 来断言调用 leg.legvander 函数时，传入参数 (1, 2, 3) 和 -1 会引发 ValueError 异常
        assert_raises(ValueError, leg.legvander, (1, 2, 3), -1)
class TestFitting:
    # 这是一个空的测试类，用于未来扩展和组织测试用例
    pass

class TestCompanion:

    def test_raises(self):
        # 断言 leg.legcompanion([]) 和 leg.legcompanion([1]) 会引发 ValueError 异常
        assert_raises(ValueError, leg.legcompanion, [])
        assert_raises(ValueError, leg.legcompanion, [1])

    def test_dimensions(self):
        # 对于 1 到 4 的范围内的系数 coef，验证 leg.legcompanion(coef) 的形状为 (i, i)
        for i in range(1, 5):
            coef = [0]*i + [1]
            assert_(leg.legcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        # 验证 leg.legcompanion([1, 2]) 的第一个元素 leg.legcompanion([1, 2])[0, 0] 等于 -.5
        assert_(leg.legcompanion([1, 2])[0, 0] == -.5)


class TestGauss:

    def test_100(self):
        # 获得 100 阶 Legendre 多项式的 Gauss 积分点和权重
        x, w = leg.leggauss(100)

        # 测试正交性。注意结果需要归一化，否则由于像 Laguerre 这样的快速增长函数可能产生的大值会非常混乱。
        v = leg.legvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        # 断言 vv 矩阵近似为单位矩阵
        assert_almost_equal(vv, np.eye(100))

        # 检查积分 1 的结果是否正确
        tgt = 2.0
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_legfromroots(self):
        # 测试 leg.legfromroots([]) 的结果近似为 [1]
        res = leg.legfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            # 对于不同阶数的多项式，生成一组根，然后使用 leg.legfromroots() 生成对应的 Legendre 多项式，
            # 然后验证计算出的多项式在这些根处的值接近于 0
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = leg.legfromroots(roots)
            res = leg.legval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(leg.leg2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_legroots(self):
        # 验证 leg.legroots([1]) 的结果近似为 []
        assert_almost_equal(leg.legroots([1]), [])
        # 验证 leg.legroots([1, 2]) 的结果近似为 [-.5]
        assert_almost_equal(leg.legroots([1, 2]), [-.5])
        for i in range(2, 5):
            # 对于不同阶数的 Legendre 多项式，生成一组根，然后验证 leg.legroots() 的结果与生成的根接近
            tgt = np.linspace(-1, 1, i)
            res = leg.legroots(leg.legfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_legtrim(self):
        coef = [2, -1, 1, 0]

        # 测试异常情况，确保 leg.legtrim(coef, -1) 会引发 ValueError 异常
        assert_raises(ValueError, leg.legtrim, coef, -1)

        # 测试结果，验证 leg.legtrim(coef) 和 leg.legtrim(coef, 1) 的正确性
        assert_equal(leg.legtrim(coef), coef[:-1])
        assert_equal(leg.legtrim(coef, 1), coef[:-3])
        assert_equal(leg.legtrim(coef, 2), [0])

    def test_legline(self):
        # 验证 leg.legline(3, 4) 返回 [3, 4]
        assert_equal(leg.legline(3, 4))

    def test_legline_zeroscl(self):
        # 验证 leg.legline(3, 0) 返回 [3]
        assert_equal(leg.legline(3, 0))

    def test_leg2poly(self):
        for i in range(10):
            # 对于 0 到 9 阶的 Legendre 多项式，验证 leg.leg2poly([0]*i + [1]) 的近似正确性
            assert_almost_equal(leg.leg2poly([0]*i + [1]), Llist[i])

    def test_poly2leg(self):
        for i in range(10):
            # 对于 0 到 9 阶的 Legendre 多项式，验证 leg.poly2leg(Llist[i]) 的近似正确性
            assert_almost_equal(leg.poly2leg(Llist[i]), [0]*i + [1])

    def test_weight(self):
        x = np.linspace(-1, 1, 11)
        tgt = 1.
        # 验证 leg.legweight(x) 的结果近似为 1.0
        res = leg.legweight(x)
        assert_almost_equal(res, tgt)
```