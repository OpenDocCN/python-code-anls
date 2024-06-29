# `.\numpy\numpy\polynomial\tests\test_hermite.py`

```py
"""Tests for hermite module.

"""
from functools import reduce  # 导入 functools 模块中的 reduce 函数

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import numpy.polynomial.hermite as herm  # 导入 NumPy 中的 Hermite 多项式模块
from numpy.polynomial.polynomial import polyval  # 导入 NumPy 中的多项式求值函数 polyval
from numpy.testing import (  # 导入 NumPy 测试模块中的多个断言函数
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )

H0 = np.array([1])  # Hermite 多项式 H0 的系数
H1 = np.array([0, 2])  # Hermite 多项式 H1 的系数
H2 = np.array([-2, 0, 4])  # Hermite 多项式 H2 的系数
H3 = np.array([0, -12, 0, 8])  # Hermite 多项式 H3 的系数
H4 = np.array([12, 0, -48, 0, 16])  # Hermite 多项式 H4 的系数
H5 = np.array([0, 120, 0, -160, 0, 32])  # Hermite 多项式 H5 的系数
H6 = np.array([-120, 0, 720, 0, -480, 0, 64])  # Hermite 多项式 H6 的系数
H7 = np.array([0, -1680, 0, 3360, 0, -1344, 0, 128])  # Hermite 多项式 H7 的系数
H8 = np.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])  # Hermite 多项式 H8 的系数
H9 = np.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])  # Hermite 多项式 H9 的系数

Hlist = [H0, H1, H2, H3, H4, H5, H6, H7, H8, H9]  # Hermite 多项式系数列表


def trim(x):
    return herm.hermtrim(x, tol=1e-6)  # 调用 hermtrim 函数来修剪 Hermite 多项式的系数


class TestConstants:

    def test_hermdomain(self):
        assert_equal(herm.hermdomain, [-1, 1])  # 断言 Hermite 多项式的定义域

    def test_hermzero(self):
        assert_equal(herm.hermzero, [0])  # 断言 Hermite 多项式零函数的系数为 [0]

    def test_hermone(self):
        assert_equal(herm.hermone, [1])  # 断言 Hermite 多项式单位函数的系数为 [1]

    def test_hermx(self):
        assert_equal(herm.hermx, [0, .5])  # 断言 Hermite 多项式的 x 函数的系数为 [0, 0.5]


class TestArithmetic:
    x = np.linspace(-3, 3, 100)  # 在 [-3, 3] 区间生成 100 个均匀间隔的点作为测试用的自变量 x

    def test_hermadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"  # 格式化消息字符串
                tgt = np.zeros(max(i, j) + 1)  # 创建一个长度足够的全零数组
                tgt[i] += 1  # 目标 Hermite 多项式的系数调整
                tgt[j] += 1  # 目标 Hermite 多项式的系数调整
                res = herm.hermadd([0]*i + [1], [0]*j + [1])  # 调用 hermadd 函数进行 Hermite 多项式的加法
                assert_equal(trim(res), trim(tgt), err_msg=msg)  # 断言结果与目标相等，并调用 trim 函数修剪 Hermite 多项式的系数

    def test_hermsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"  # 格式化消息字符串
                tgt = np.zeros(max(i, j) + 1)  # 创建一个长度足够的全零数组
                tgt[i] += 1  # 目标 Hermite 多项式的系数调整
                tgt[j] -= 1  # 目标 Hermite 多项式的系数调整
                res = herm.hermsub([0]*i + [1], [0]*j + [1])  # 调用 hermsub 函数进行 Hermite 多项式的减法
                assert_equal(trim(res), trim(tgt), err_msg=msg)  # 断言结果与目标相等，并调用 trim 函数修剪 Hermite 多项式的系数

    def test_hermmulx(self):
        assert_equal(herm.hermmulx([0]), [0])  # 断言 hermmulx 函数对 [0] 返回 [0]
        assert_equal(herm.hermmulx([1]), [0, .5])  # 断言 hermmulx 函数对 [1] 返回 [0, 0.5]
        for i in range(1, 5):
            ser = [0]*i + [1]  # 构造 Hermite 多项式的系数列表
            tgt = [0]*(i - 1) + [i, 0, .5]  # 预期的 Hermite 多项式乘以 x 后的系数列表
            assert_equal(herm.hermmulx(ser), tgt)  # 断言 hermmulx 函数的返回值与预期相等

    def test_hermmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0]*i + [1]  # 构造第一个 Hermite 多项式的系数列表
            val1 = herm.hermval(self.x, pol1)  # 计算第一个 Hermite 多项式在 x 上的值
            for j in range(5):
                msg = f"At i={i}, j={j}"  # 格式化消息字符串
                pol2 = [0]*j + [1]  # 构造第二个 Hermite 多项式的系数列表
                val2 = herm.hermval(self.x, pol2)  # 计算第二个 Hermite 多项式在 x 上的值
                pol3 = herm.hermmul(pol1, pol2)  # 计算两个 Hermite 多项式的乘积
                val3 = herm.hermval(self.x, pol3)  # 计算乘积 Hermite 多项式在 x 上的值
                assert_(len(pol3) == i + j + 1, msg)  # 断言乘积 Hermite 多项式的阶数正确
                assert_almost_equal(val3, val1*val2, err_msg=msg)  # 断言乘积 Hermite 多项式在 x 上的值与预期乘积相近
    # 定义测试函数 test_hermdiv，用于测试 herm 模块中的 hermdiv 函数
    def test_hermdiv(self):
        # 循环遍历 i 的取值范围 [0, 5)
        for i in range(5):
            # 循环遍历 j 的取值范围 [0, 5)
            for j in range(5):
                # 根据当前的 i 和 j 生成调试信息
                msg = f"At i={i}, j={j}"
                # 创建长度为 i 的列表 ci，最后一个元素为 1，其余为 0
                ci = [0]*i + [1]
                # 创建长度为 j 的列表 cj，最后一个元素为 1，其余为 0
                cj = [0]*j + [1]
                # 使用 herm 模块中的 hermadd 函数对 ci 和 cj 进行 Hermite 多项式的加法
                tgt = herm.hermadd(ci, cj)
                # 使用 herm 模块中的 hermdiv 函数计算 tgt 除以 ci 的商和余数
                quo, rem = herm.hermdiv(tgt, ci)
                # 使用 herm 模块中的 hermadd 函数计算 quo 乘以 ci 加上 rem 的结果
                res = herm.hermadd(herm.hermmul(quo, ci), rem)
                # 使用 assert_equal 函数断言 trim(res) 等于 trim(tgt)，如果不等则输出错误信息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    # 定义测试函数 test_hermpow，用于测试 herm 模块中的 hermpow 函数
    def test_hermpow(self):
        # 循环遍历 i 的取值范围 [0, 5)
        for i in range(5):
            # 循环遍历 j 的取值范围 [0, 5)
            for j in range(5):
                # 根据当前的 i 和 j 生成调试信息
                msg = f"At i={i}, j={j}"
                # 创建长度为 i+1 的 numpy 数组 c，元素为 [0, 1, ..., i]
                c = np.arange(i + 1)
                # 使用 reduce 函数和 herm 模块中的 hermmul 函数对 c 重复 j 次进行 Hermite 多项式乘法
                tgt = reduce(herm.hermmul, [c]*j, np.array([1]))
                # 使用 herm 模块中的 hermpow 函数计算 Hermite 多项式 c 的 j 次幂
                res = herm.hermpow(c, j) 
                # 使用 assert_equal 函数断言 trim(res) 等于 trim(tgt)，如果不等则输出错误信息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)
class TestEvaluation:
    # 定义一维多项式系数 [1, 2.5, 0.75]
    c1d = np.array([2.5, 1., .75])
    # 使用 einsum 函数创建二维多项式系数矩阵
    c2d = np.einsum('i,j->ij', c1d, c1d)
    # 使用 einsum 函数创建三维多项式系数张量
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # 生成随机数组，形状为 (3, 5)，数值范围在 [-1, 1)
    x = np.random.random((3, 5))*2 - 1
    # 计算多项式在随机数组 x 上的值，多项式系数为 [1, 2, 3]
    y = polyval(x, [1., 2., 3.])

    def test_hermval(self):
        # 检查空输入的情况
        assert_equal(herm.hermval([], [1]).size, 0)

        # 检查正常输入的情况
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Hlist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            # 计算 Hermite 插值，并断言结果近似于目标值
            res = herm.hermval(x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # 检查形状保持不变的情况
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            # 断言 Hermite 插值的输出形状与预期维度相同
            assert_equal(herm.hermval(x, [1]).shape, dims)
            assert_equal(herm.hermval(x, [1, 0]).shape, dims)
            assert_equal(herm.hermval(x, [1, 0, 0]).shape, dims)

    def test_hermval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises(ValueError, herm.hermval2d, x1, x2[:2], self.c2d)

        # 测试值情况
        tgt = y1 * y2
        # 计算二维 Hermite 插值，并断言结果近似于目标值
        res = herm.hermval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 测试形状情况
        z = np.ones((2, 3))
        # 断言二维 Hermite 插值的输出形状为 (2, 3)
        res = herm.hermval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_hermval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises(ValueError, herm.hermval3d, x1, x2, x3[:2], self.c3d)

        # 测试值情况
        tgt = y1 * y2 * y3
        # 计算三维 Hermite 插值，并断言结果近似于目标值
        res = herm.hermval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # 测试形状情况
        z = np.ones((2, 3))
        # 断言三维 Hermite 插值的输出形状为 (2, 3)
        res = herm.hermval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_hermgrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试值情况
        tgt = np.einsum('i,j->ij', y1, y2)
        # 计算二维 Hermite 插值网格，并断言结果近似于目标值
        res = herm.hermgrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 测试形状情况
        z = np.ones((2, 3))
        # 断言二维 Hermite 插值网格的输出形状为 (2, 3, 2, 3)
        res = herm.hermgrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3)*2)

    def test_hermgrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试值情况
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        # 计算三维 Hermite 插值网格，并断言结果近似于目标值
        res = herm.hermgrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # 测试形状情况
        z = np.ones((2, 3))
        # 断言三维 Hermite 插值网格的输出形状为 (2, 3, 2, 3, 2, 3)
        res = herm.hermgrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3)*3)
    def test_hermint_axis(self):
        # 检查轴关键字参数的工作情况

        # 创建一个 3x4 的随机数组
        c2d = np.random.random((3, 4))

        # 使用 herm.hermint 函数对 c2d 的转置进行 Hermite 处理，并堆叠成新的数组
        tgt = np.vstack([herm.hermint(c) for c in c2d.T]).T
        # 调用 herm.hermint 函数，指定 axis=0 对 c2d 进行 Hermite 处理
        res = herm.hermint(c2d, axis=0)
        # 断言 res 与 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 对 c2d 的每一行使用 herm.hermint 函数进行 Hermite 处理，并堆叠成新的数组
        tgt = np.vstack([herm.hermint(c) for c in c2d])
        # 调用 herm.hermint 函数，指定 axis=1 对 c2d 进行 Hermite 处理
        res = herm.hermint(c2d, axis=1)
        # 断言 res 与 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 对 c2d 的每一行使用 herm.hermint 函数进行 Hermite 处理，同时指定 k=3，并堆叠成新的数组
        tgt = np.vstack([herm.hermint(c, k=3) for c in c2d])
        # 调用 herm.hermint 函数，指定 k=3 和 axis=1 对 c2d 进行 Hermite 处理
        res = herm.hermint(c2d, k=3, axis=1)
        # 断言 res 与 tgt 几乎相等
        assert_almost_equal(res, tgt)
class TestDerivative:

    def test_hermder(self):
        # 检查异常情况
        assert_raises(TypeError, herm.hermder, [0], .5)
        assert_raises(ValueError, herm.hermder, [0], -1)

        # 检查零阶导数不产生变化
        for i in range(5):
            tgt = [0]*i + [1]
            res = herm.hermder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # 检查导数与积分的逆关系
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = herm.hermder(herm.hermint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # 检查带有缩放的导数
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = herm.hermder(herm.hermint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermder_axis(self):
        # 检查轴关键字的工作情况
        c2d = np.random.random((3, 4))

        tgt = np.vstack([herm.hermder(c) for c in c2d.T]).T
        res = herm.hermder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herm.hermder(c) for c in c2d])
        res = herm.hermder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # 在 [-1, 1) 范围内生成一些随机值
    x = np.random.random((3, 5))*2 - 1

    def test_hermvander(self):
        # 检查 1 维 x 的情况
        x = np.arange(3)
        v = herm.hermvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], herm.hermval(x, coef))

        # 检查 2 维 x 的情况
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = herm.hermvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], herm.hermval(x, coef))

    def test_hermvander2d(self):
        # 同时测试非方形系数数组的 hermval2d
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = herm.hermvander2d(x1, x2, [1, 2])
        tgt = herm.hermval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = herm.hermvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_hermvander3d(self):
        # 同时测试非方形系数数组的 hermval3d
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = herm.hermvander3d(x1, x2, x3, [1, 2, 3])
        tgt = herm.hermval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = herm.hermvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

class TestCompanion:
    # 测试函数，验证 hermcompanion 函数在接收空列表时是否引发 ValueError 异常
    def test_raises(self):
        assert_raises(ValueError, herm.hermcompanion, [])
    
    # 测试函数，验证 hermcompanion 函数在接收包含一个元素的列表时是否引发 ValueError 异常
    def test_dimensions(self):
        # 对于每个 i 在范围 [1, 5) 中
        for i in range(1, 5):
            # 创建一个系数列表，长度为 i+1，末尾元素为 1，其余元素为 0
            coef = [0]*i + [1]
            # 断言 hermcompanion 函数返回的矩阵形状为 (i, i)
            assert_(herm.hermcompanion(coef).shape == (i, i))
    
    # 测试函数，验证 hermcompanion 函数在接收系数列表 [1, 2] 时，返回的第一个元素的值是否为 -0.25
    def test_linear_root(self):
        assert_(herm.hermcompanion([1, 2])[0, 0] == -.25)
class TestGauss:

    def test_100(self):
        # 调用 hermgauss 函数生成 Hermite-Gauss 积分的节点和权重
        x, w = herm.hermgauss(100)

        # 测试正交性。注意需要对结果进行归一化，否则像 Laguerre 这样快速增长的函数可能产生非常大的值，令人困惑。
        v = herm.hermvander(x, 99)
        vv = np.dot(v.T * w, v)
        # 计算对角线上的标准化因子
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        # 断言结果应接近单位矩阵
        assert_almost_equal(vv, np.eye(100))

        # 检查积分 1 的正确性
        tgt = np.sqrt(np.pi)
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_hermfromroots(self):
        # 测试从根生成 Hermite 多项式
        res = herm.hermfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = herm.hermfromroots(roots)
            res = herm.hermval(roots, pol)
            tgt = 0
            # 断言多项式的阶数
            assert_(len(pol) == i + 1)
            # 断言 Hermite 多项式的最高次数系数为 1
            assert_almost_equal(herm.herm2poly(pol)[-1], 1)
            # 断言计算的结果与目标值一致
            assert_almost_equal(res, tgt)

    def test_hermroots(self):
        # 测试 Hermite 多项式的根
        assert_almost_equal(herm.hermroots([1]), [])
        assert_almost_equal(herm.hermroots([1, 1]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = herm.hermroots(herm.hermfromroots(tgt))
            # 断言修剪后的结果与目标值一致
            assert_almost_equal(trim(res), trim(tgt))

    def test_hermtrim(self):
        coef = [2, -1, 1, 0]

        # 测试异常情况
        assert_raises(ValueError, herm.hermtrim, coef, -1)

        # 测试结果
        assert_equal(herm.hermtrim(coef), coef[:-1])
        assert_equal(herm.hermtrim(coef, 1), coef[:-3])
        assert_equal(herm.hermtrim(coef, 2), [0])

    def test_hermline(self):
        # 测试生成 Hermite 多项式的系数
        assert_equal(herm.hermline(3, 4), [3, 2])

    def test_herm2poly(self):
        for i in range(10):
            # 断言 Hermite 多项式到多项式的转换结果
            assert_almost_equal(herm.herm2poly([0]*i + [1]), Hlist[i])

    def test_poly2herm(self):
        for i in range(10):
            # 断言多项式到 Hermite 多项式的转换结果
            assert_almost_equal(herm.poly2herm(Hlist[i]), [0]*i + [1])

    def test_weight(self):
        x = np.linspace(-5, 5, 11)
        tgt = np.exp(-x**2)
        res = herm.hermweight(x)
        # 断言权重函数的计算结果与目标值一致
        assert_almost_equal(res, tgt)
```