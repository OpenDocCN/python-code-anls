# `.\numpy\numpy\polynomial\tests\test_hermite_e.py`

```py
"""Tests for hermite_e module.

"""
from functools import reduce  # 导入 functools 模块中的 reduce 函数

import numpy as np  # 导入 NumPy 库并简写为 np
import numpy.polynomial.hermite_e as herme  # 导入 NumPy 中的 Hermite 多项式模块
from numpy.polynomial.polynomial import polyval  # 从 NumPy 多项式模块中导入 polyval 函数
from numpy.testing import (  # 从 NumPy 测试模块中导入多个断言函数
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )

He0 = np.array([1])  # 定义 Hermite 多项式 H0
He1 = np.array([0, 1])  # 定义 Hermite 多项式 H1
He2 = np.array([-1, 0, 1])  # 定义 Hermite 多项式 H2
He3 = np.array([0, -3, 0, 1])  # 定义 Hermite 多项式 H3
He4 = np.array([3, 0, -6, 0, 1])  # 定义 Hermite 多项式 H4
He5 = np.array([0, 15, 0, -10, 0, 1])  # 定义 Hermite 多项式 H5
He6 = np.array([-15, 0, 45, 0, -15, 0, 1])  # 定义 Hermite 多项式 H6
He7 = np.array([0, -105, 0, 105, 0, -21, 0, 1])  # 定义 Hermite 多项式 H7
He8 = np.array([105, 0, -420, 0, 210, 0, -28, 0, 1])  # 定义 Hermite 多项式 H8
He9 = np.array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])  # 定义 Hermite 多项式 H9

Helist = [He0, He1, He2, He3, He4, He5, He6, He7, He8, He9]  # 将 Hermite 多项式存入列表中


def trim(x):
    return herme.hermetrim(x, tol=1e-6)  # 调用 herme.hermetrim 函数，用于修剪 Hermite 多项式


class TestConstants:

    def test_hermedomain(self):
        assert_equal(herme.hermedomain, [-1, 1])  # 断言检查 herme.hermedomain 是否为 [-1, 1]

    def test_hermezero(self):
        assert_equal(herme.hermezero, [0])  # 断言检查 herme.hermezero 是否为 [0]

    def test_hermeone(self):
        assert_equal(herme.hermeone, [1])  # 断言检查 herme.hermeone 是否为 [1]

    def test_hermex(self):
        assert_equal(herme.hermex, [0, 1])  # 断言检查 herme.hermex 是否为 [0, 1]


class TestArithmetic:
    x = np.linspace(-3, 3, 100)  # 创建一个包含 100 个元素的等差数列，范围从 -3 到 3

    def test_hermeadd(self):
        for i in range(5):  # 循环遍历 i 从 0 到 4
            for j in range(5):  # 循环遍历 j 从 0 到 4
                msg = f"At i={i}, j={j}"  # 格式化消息字符串
                tgt = np.zeros(max(i, j) + 1)  # 创建一个长度为 max(i, j) + 1 的全零数组
                tgt[i] += 1  # 将数组 tgt 的第 i 个元素设为 1
                tgt[j] += 1  # 将数组 tgt 的第 j 个元素设为 1
                res = herme.hermeadd([0]*i + [1], [0]*j + [1])  # 调用 herme.hermeadd 函数
                assert_equal(trim(res), trim(tgt), err_msg=msg)  # 断言检查修剪后的 res 是否等于修剪后的 tgt

    def test_hermesub(self):
        for i in range(5):  # 循环遍历 i 从 0 到 4
            for j in range(5):  # 循环遍历 j 从 0 到 4
                msg = f"At i={i}, j={j}"  # 格式化消息字符串
                tgt = np.zeros(max(i, j) + 1)  # 创建一个长度为 max(i, j) + 1 的全零数组
                tgt[i] += 1  # 将数组 tgt 的第 i 个元素设为 1
                tgt[j] -= 1  # 将数组 tgt 的第 j 个元素设为 -1
                res = herme.hermesub([0]*i + [1], [0]*j + [1])  # 调用 herme.hermesub 函数
                assert_equal(trim(res), trim(tgt), err_msg=msg)  # 断言检查修剪后的 res 是否等于修剪后的 tgt

    def test_hermemulx(self):
        assert_equal(herme.hermemulx([0]), [0])  # 断言检查 herme.hermemulx([0]) 是否为 [0]
        assert_equal(herme.hermemulx([1]), [0, 1])  # 断言检查 herme.hermemulx([1]) 是否为 [0, 1]
        for i in range(1, 5):  # 循环遍历 i 从 1 到 4
            ser = [0]*i + [1]  # 创建一个 Hermite 多项式系数列表
            tgt = [0]*(i - 1) + [i, 0, 1]  # 创建目标 Hermite 多项式系数列表
            assert_equal(herme.hermemulx(ser), tgt)  # 断言检查 herme.hermemulx 函数的输出是否等于 tgt

    def test_hermemul(self):
        # check values of result
        for i in range(5):  # 循环遍历 i 从 0 到 4
            pol1 = [0]*i + [1]  # 创建第一个 Hermite 多项式系数列表
            val1 = herme.hermeval(self.x, pol1)  # 计算第一个 Hermite 多项式在 x 上的值
            for j in range(5):  # 循环遍历 j 从 0 到 4
                msg = f"At i={i}, j={j}"  # 格式化消息字符串
                pol2 = [0]*j + [1]  # 创建第二个 Hermite 多项式系数列表
                val2 = herme.hermeval(self.x, pol2)  # 计算第二个 Hermite 多项式在 x 上的值
                pol3 = herme.hermemul(pol1, pol2)  # 调用 herme.hermemul 函数得到两个 Hermite 多项式的乘积
                val3 = herme.hermeval(self.x, pol3)  # 计算乘积 Hermite 多项式在 x 上的值
                assert_(len(pol3) == i + j + 1, msg)  # 断言检查乘积 Hermite 多项式的长度是否为 i + j + 1
                assert_almost_equal(val3, val1 * val2, err_msg=msg)  # 断言检查乘积 Hermite 多项式在 x 上的值是否接近于 val1 * val2
    # 定义一个测试函数，用于测试 hermediv 方法
    def test_hermediv(self):
        # 循环遍历 i 范围内的值，从 0 到 4
        for i in range(5):
            # 循环遍历 j 范围内的值，从 0 到 4
            for j in range(5):
                # 构造消息字符串，指示当前的 i 和 j 值
                msg = f"At i={i}, j={j}"
                # 创建包含 i 个 0 和一个 1 的列表 ci
                ci = [0]*i + [1]
                # 创建包含 j 个 0 和一个 1 的列表 cj
                cj = [0]*j + [1]
                # 调用 hermeadd 方法，计算 ci 和 cj 的和，并赋给 tgt
                tgt = herme.hermeadd(ci, cj)
                # 调用 hermediv 方法，计算 tgt 除以 ci 的商和余数，分别赋给 quo 和 rem
                quo, rem = herme.hermediv(tgt, ci)
                # 调用 hermeadd 方法，计算 quo 乘以 ci 加上 rem，赋给 res
                res = herme.hermeadd(herme.hermemul(quo, ci), rem)
                # 断言 trim 后的 res 等于 trim 后的 tgt，如果不等则输出错误消息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    # 定义一个测试函数，用于测试 hermepow 方法
    def test_hermepow(self):
        # 循环遍历 i 范围内的值，从 0 到 4
        for i in range(5):
            # 循环遍历 j 范围内的值，从 0 到 4
            for j in range(5):
                # 构造消息字符串，指示当前的 i 和 j 值
                msg = f"At i={i}, j={j}"
                # 创建一个包含从 0 到 i 的连续整数的 numpy 数组 c
                c = np.arange(i + 1)
                # 使用 reduce 函数和 hermemul 方法，计算 c 中的元素的 j 次乘积，初始值为 1，赋给 tgt
                tgt = reduce(herme.hermemul, [c]*j, np.array([1]))
                # 调用 hermepow 方法，计算 c 的 j 次幂，赋给 res
                res = herme.hermepow(c, j)
                # 断言 trim 后的 res 等于 trim 后的 tgt，如果不等则输出错误消息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)
class TestEvaluation:
    # 一维多项式系数，对应于 1 + 2*x + 3*x**2
    c1d = np.array([4., 2., 3.])
    # 二维数组，使用 einsum 计算外积
    c2d = np.einsum('i,j->ij', c1d, c1d)
    # 三维数组，使用 einsum 计算三阶张量
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # 在 [-1, 1) 内生成随机值的 3x5 数组
    x = np.random.random((3, 5))*2 - 1
    # 根据多项式在 x 上的求值结果
    y = polyval(x, [1., 2., 3.])

    def test_hermeval(self):
        # 检查空输入时的行为
        assert_equal(herme.hermeval([], [1]).size, 0)

        # 检查正常输入情况下的行为
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Helist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            # 调用 hermeval 函数计算 Hermite 插值结果，并进行近似相等断言
            res = herme.hermeval(x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # 检查形状是否保持不变
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            # 验证 hermeval 函数在不同维度的输入下返回的结果形状
            assert_equal(herme.hermeval(x, [1]).shape, dims)
            assert_equal(herme.hermeval(x, [1, 0]).shape, dims)
            assert_equal(herme.hermeval(x, [1, 0, 0]).shape, dims)

    def test_hermeval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises(ValueError, herme.hermeval2d, x1, x2[:2], self.c2d)

        # 测试正常值情况
        tgt = y1 * y2
        # 调用 hermeval2d 函数计算二维 Hermite 插值结果，并进行近似相等断言
        res = herme.hermeval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 测试形状是否保持不变
        z = np.ones((2, 3))
        # 验证 hermeval2d 函数在不同维度的输入下返回的结果形状
        assert_(res.shape == (2, 3))

    def test_hermeval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises(ValueError, herme.hermeval3d, x1, x2, x3[:2], self.c3d)

        # 测试正常值情况
        tgt = y1 * y2 * y3
        # 调用 hermeval3d 函数计算三维 Hermite 插值结果，并进行近似相等断言
        res = herme.hermeval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # 测试形状是否保持不变
        z = np.ones((2, 3))
        # 验证 hermeval3d 函数在不同维度的输入下返回的结果形状
        assert_(res.shape == (2, 3))

    def test_hermegrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试值情况
        tgt = np.einsum('i,j->ij', y1, y2)
        # 调用 hermegrid2d 函数计算二维 Hermite 插值的网格结果，并进行近似相等断言
        res = herme.hermegrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 测试形状是否保持不变
        z = np.ones((2, 3))
        # 验证 hermegrid2d 函数在不同维度的输入下返回的结果形状
        assert_(res.shape == (2, 3) * 2)

    def test_hermegrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试值情况
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        # 调用 hermegrid3d 函数计算三维 Hermite 插值的网格结果，并进行近似相等断言
        res = herme.hermegrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # 测试形状是否保持不变
        z = np.ones((2, 3))
        # 验证 hermegrid3d 函数在不同维度的输入下返回的结果形状
        assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    # 定义一个测试函数，用于测试 hermeint 函数在不同轴上的行为
    def test_hermeint_axis(self):
        # 检查 axis 关键字参数是否有效
        c2d = np.random.random((3, 4))  # 创建一个 3x4 的随机数数组 c2d

        # 在轴为1的方向上对 c2d 中每列应用 hermeint 函数，然后垂直堆叠得到 tgt
        tgt = np.vstack([herme.hermeint(c) for c in c2d.T]).T
        res = herme.hermeint(c2d, axis=0)  # 调用 hermeint 函数，指定 axis=0
        assert_almost_equal(res, tgt)  # 断言结果 res 与目标 tgt 几乎相等

        # 在轴为0的方向上对 c2d 中每行应用 hermeint 函数，然后垂直堆叠得到 tgt
        tgt = np.vstack([herme.hermeint(c) for c in c2d])
        res = herme.hermeint(c2d, axis=1)  # 调用 hermeint 函数，指定 axis=1
        assert_almost_equal(res, tgt)  # 断言结果 res 与目标 tgt 几乎相等

        # 在轴为1的方向上对 c2d 中每行应用 hermeint 函数，同时指定额外的参数 k=3
        tgt = np.vstack([herme.hermeint(c, k=3) for c in c2d])
        res = herme.hermeint(c2d, k=3, axis=1)  # 调用 hermeint 函数，指定 axis=1 和 k=3
        assert_almost_equal(res, tgt)  # 断言结果 res 与目标 tgt 几乎相等
class TestDerivative:

    def test_hermeder(self):
        # 检查异常情况：验证在给定参数下是否会抛出 TypeError 异常
        assert_raises(TypeError, herme.hermeder, [0], .5)
        # 检查异常情况：验证在给定参数下是否会抛出 ValueError 异常
        assert_raises(ValueError, herme.hermeder, [0], -1)

        # 检查零阶导数是否不做任何操作
        for i in range(5):
            tgt = [0]*i + [1]
            res = herme.hermeder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # 检查导数是否是积分的逆过程
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = herme.hermeder(herme.hermeint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # 检查带有缩放的导数计算
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = herme.hermeder(
                    herme.hermeint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermeder_axis(self):
        # 检查 axis 关键字的工作情况
        c2d = np.random.random((3, 4))

        # 沿轴 0 进行导数计算的结果
        tgt = np.vstack([herme.hermeder(c) for c in c2d.T]).T
        res = herme.hermeder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        # 沿轴 1 进行导数计算的结果
        tgt = np.vstack([herme.hermeder(c) for c in c2d])
        res = herme.hermeder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # 在 [-1, 1) 范围内生成随机值
    x = np.random.random((3, 5))*2 - 1

    def test_hermevander(self):
        # 检查 1 维 x 的情况
        x = np.arange(3)
        v = herme.hermevander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], herme.hermeval(x, coef))

        # 检查 2 维 x 的情况
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = herme.hermevander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], herme.hermeval(x, coef))

    def test_hermevander2d(self):
        # 同时测试非方形系数数组的 hermeval2d 函数
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = herme.hermevander2d(x1, x2, [1, 2])
        tgt = herme.hermeval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = herme.hermevander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_hermevander3d(self):
        # 同时测试非方形系数数组的 hermeval3d 函数
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = herme.hermevander3d(x1, x2, x3, [1, 2, 3])
        tgt = herme.hermeval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = herme.hermevander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

class TestCompanion:
    # 定义测试函数，用于测试 herme.hermecompanion 函数是否能正确引发 ValueError 异常，当输入为空列表或包含一个元素时
    def test_raises(self):
        # 断言调用 herme.hermecompanion 函数时会引发 ValueError 异常，传入空列表作为参数
        assert_raises(ValueError, herme.hermecompanion, [])
        # 断言调用 herme.hermecompanion 函数时会引发 ValueError 异常，传入包含一个元素的列表作为参数
        assert_raises(ValueError, herme.hermecompanion, [1])

    # 定义测试函数，用于测试 herme.hermecompanion 函数生成的伴随矩阵的维度是否符合预期
    def test_dimensions(self):
        # 循环测试不同长度的系数列表
        for i in range(1, 5):
            # 构造系数列表，前 i-1 个元素为 0，最后一个元素为 1
            coef = [0]*i + [1]
            # 断言生成的伴随矩阵的形状是否为 (i, i)
            assert_(herme.hermecompanion(coef).shape == (i, i))

    # 定义测试函数，用于测试 herme.hermecompanion 函数生成的伴随矩阵中特定元素的值是否符合预期
    def test_linear_root(self):
        # 断言生成的伴随矩阵的第一个元素是否为 -0.5，当传入系数列表为 [1, 2] 时
        assert_(herme.hermecompanion([1, 2])[0, 0] == -.5)
class TestGauss:

    def test_100(self):
        # 调用 herme 模块的 hermegauss 函数，获取 Hermite-Gauss 积分的节点和权重
        x, w = herme.hermegauss(100)

        # 测试正交性。注意需要对结果进行归一化，否则由于 Laguerre 等快速增长函数可能产生的巨大值会很令人困惑。
        v = herme.hermevander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd

        # 断言计算出的 vv 与单位矩阵近似相等
        assert_almost_equal(vv, np.eye(100))

        # 检查积分 1 的结果是否正确
        tgt = np.sqrt(2*np.pi)
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_hermefromroots(self):
        # 调用 herme 模块的 hermefromroots 函数，验证结果近似等于 [1]
        res = herme.hermefromroots([])
        assert_almost_equal(trim(res), [1])

        # 对于范围从 1 到 4 的每个 i
        for i in range(1, 5):
            # 生成 Hermite 多项式的根
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            # 使用 hermefromroots 生成多项式
            pol = herme.hermefromroots(roots)
            # 计算多项式在给定根处的值
            res = herme.hermeval(roots, pol)
            tgt = 0

            # 断言多项式的长度为 i+1
            assert_(len(pol) == i + 1)
            # 断言 Hermite 多项式转换为多项式的最后一个系数近似为 1
            assert_almost_equal(herme.herme2poly(pol)[-1], 1)
            # 断言计算的值近似为目标值
            assert_almost_equal(res, tgt)

    def test_hermeroots(self):
        # 测试 hermeroots 函数对 [1] 的近似值
        assert_almost_equal(herme.hermeroots([1]), [])
        # 测试 hermeroots 函数对 [1, 1] 的近似值
        assert_almost_equal(herme.hermeroots([1, 1]), [-1])

        # 对于范围从 2 到 4 的每个 i
        for i in range(2, 5):
            # 生成 -1 到 1 之间的均匀分布的长度为 i 的目标值
            tgt = np.linspace(-1, 1, i)
            # 生成 Hermite 多项式的根
            res = herme.hermeroots(herme.hermefromroots(tgt))
            # 断言修剪后的结果近似等于目标值的修剪版本
            assert_almost_equal(trim(res), trim(tgt))

    def test_hermetrim(self):
        coef = [2, -1, 1, 0]

        # 测试异常情况
        assert_raises(ValueError, herme.hermetrim, coef, -1)

        # 测试结果
        assert_equal(herme.hermetrim(coef), coef[:-1])
        assert_equal(herme.hermetrim(coef, 1), coef[:-3])
        assert_equal(herme.hermetrim(coef, 2), [0])

    def test_hermeline(self):
        # 断言 hermeline 函数的返回值与输入参数相等
        assert_equal(herme.hermeline(3, 4), [3, 4])

    def test_herme2poly(self):
        # 对于范围从 0 到 9 的每个 i
        for i in range(10):
            # 断言 herme2poly 函数近似等于 Helist[i]
            assert_almost_equal(herme.herme2poly([0]*i + [1]), Helist[i])

    def test_poly2herme(self):
        # 对于范围从 0 到 9 的每个 i
        for i in range(10):
            # 断言 poly2herme 函数近似等于 [0]*i + [1]
            assert_almost_equal(herme.poly2herme(Helist[i]), [0]*i + [1])

    def test_weight(self):
        # 生成 -5 到 5 之间的均匀分布的长度为 11 的目标值
        x = np.linspace(-5, 5, 11)
        # 计算 Hermite 权重函数的值
        tgt = np.exp(-.5*x**2)
        res = herme.hermeweight(x)
        # 断言计算结果近似等于目标值
        assert_almost_equal(res, tgt)
```