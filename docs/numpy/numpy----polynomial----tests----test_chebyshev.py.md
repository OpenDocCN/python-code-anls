# `.\numpy\numpy\polynomial\tests\test_chebyshev.py`

```
"""Tests for chebyshev module.

"""
# 导入 reduce 函数
from functools import reduce

# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 numpy.polynomial.chebyshev 模块，并使用别名 cheb
import numpy.polynomial.chebyshev as cheb
# 导入 numpy.polynomial.polynomial 模块中的 polyval 函数
from numpy.polynomial.polynomial import polyval
# 导入 numpy.testing 模块中的断言函数
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )

# 定义函数 trim，用于裁剪 Chebyshev 系数
def trim(x):
    return cheb.chebtrim(x, tol=1e-6)

# 定义一系列 Chebyshev 多项式的系数列表
T0 = [1]
T1 = [0, 1]
T2 = [-1, 0, 2]
T3 = [0, -3, 0, 4]
T4 = [1, 0, -8, 0, 8]
T5 = [0, 5, 0, -20, 0, 16]
T6 = [-1, 0, 18, 0, -48, 0, 32]
T7 = [0, -7, 0, 56, 0, -112, 0, 64]
T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

# 将所有 Chebyshev 多项式的系数列表放入一个列表中
Tlist = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]

# 定义一个名为 TestPrivate 的测试类
class TestPrivate:

    # 定义测试方法 test__cseries_to_zseries，测试 _cseries_to_zseries 函数
    def test__cseries_to_zseries(self):
        # 循环测试多次
        for i in range(5):
            # 创建输入数组 inp 和目标数组 tgt
            inp = np.array([2] + [1]*i, np.double)
            tgt = np.array([.5]*i + [2] + [.5]*i, np.double)
            # 调用 _cseries_to_zseries 函数并断言结果是否与目标数组相等
            res = cheb._cseries_to_zseries(inp)
            assert_equal(res, tgt)

    # 定义测试方法 test__zseries_to_cseries，测试 _zseries_to_cseries 函数
    def test__zseries_to_cseries(self):
        # 循环测试多次
        for i in range(5):
            # 创建输入数组 inp 和目标数组 tgt
            inp = np.array([.5]*i + [2] + [.5]*i, np.double)
            tgt = np.array([2] + [1]*i, np.double)
            # 调用 _zseries_to_cseries 函数并断言结果是否与目标数组相等
            res = cheb._zseries_to_cseries(inp)
            assert_equal(res, tgt)


# 定义一个名为 TestConstants 的测试类
class TestConstants:

    # 定义测试方法 test_chebdomain，测试 chebdomain 常量
    def test_chebdomain(self):
        assert_equal(cheb.chebdomain, [-1, 1])

    # 定义测试方法 test_chebzero，测试 chebzero 常量
    def test_chebzero(self):
        assert_equal(cheb.chebzero, [0])

    # 定义测试方法 test_chebone，测试 chebone 常量
    def test_chebone(self):
        assert_equal(cheb.chebone, [1])

    # 定义测试方法 test_chebx，测试 chebx 常量
    def test_chebx(self):
        assert_equal(cheb.chebx, [0, 1])


# 定义一个名为 TestArithmetic 的测试类
class TestArithmetic:

    # 定义测试方法 test_chebadd，测试 chebadd 函数
    def test_chebadd(self):
        # 循环测试多次
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                # 创建目标数组 tgt 和调用 chebadd 函数得到的结果数组 res
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = cheb.chebadd([0]*i + [1], [0]*j + [1])
                # 断言修剪后的 res 是否与 tgt 相等
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    # 定义测试方法 test_chebsub，测试 chebsub 函数
    def test_chebsub(self):
        # 循环测试多次
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                # 创建目标数组 tgt 和调用 chebsub 函数得到的结果数组 res
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = cheb.chebsub([0]*i + [1], [0]*j + [1])
                # 断言修剪后的 res 是否与 tgt 相等
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    # 定义测试方法 test_chebmulx，测试 chebmulx 函数
    def test_chebmulx(self):
        assert_equal(cheb.chebmulx([0]), [0])
        assert_equal(cheb.chebmulx([1]), [0, 1])
        # 循环测试多次
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [.5, 0, .5]
            # 断言 chebmulx 函数的结果是否与目标数组 tgt 相等
            assert_equal(cheb.chebmulx(ser), tgt)

    # 定义测试方法 test_chebmul，测试 chebmul 函数
    def test_chebmul(self):
        # 循环测试多次
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                # 创建目标数组 tgt 和调用 chebmul 函数得到的结果数组 res
                tgt = np.zeros(i + j + 1)
                tgt[i + j] += .5
                tgt[abs(i - j)] += .5
                res = cheb.chebmul([0]*i + [1], [0]*j + [1])
                # 断言修剪后的 res 是否与 tgt 相等
                assert_equal(trim(res), trim(tgt), err_msg=msg)
    # 测试 chebdiv 函数的功能
    def test_chebdiv(self):
        # 循环遍历 i 的范围为 0 到 4
        for i in range(5):
            # 循环遍历 j 的范围为 0 到 4
            for j in range(5):
                # 创建格式化消息，指示当前循环的 i 和 j 值
                msg = f"At i={i}, j={j}"
                # 创建长度为 i+1 的列表 ci，最后一个元素为 1，其余为 0
                ci = [0]*i + [1]
                # 创建长度为 j+1 的列表 cj，最后一个元素为 1，其余为 0
                cj = [0]*j + [1]
                # 调用 chebadd 函数计算 ci 和 cj 的和，存储在 tgt 中
                tgt = cheb.chebadd(ci, cj)
                # 调用 chebdiv 函数计算 tgt 除以 ci 的商 quo 和余数 rem
                quo, rem = cheb.chebdiv(tgt, ci)
                # 调用 chebadd 函数计算 quo 乘以 ci 加上 rem 的结果 res
                res = cheb.chebadd(cheb.chebmul(quo, ci), rem)
                # 使用 assert_equal 函数验证 trim(res) 是否等于 trim(tgt)，否则输出错误消息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)
    
    # 测试 chebpow 函数的功能
    def test_chebpow(self):
        # 循环遍历 i 的范围为 0 到 4
        for i in range(5):
            # 循环遍历 j 的范围为 0 到 4
            for j in range(5):
                # 创建格式化消息，指示当前循环的 i 和 j 值
                msg = f"At i={i}, j={j}"
                # 创建长度为 i+1 的 numpy 数组 c，包含从 0 到 i 的整数
                c = np.arange(i + 1)
                # 使用 reduce 函数和 chebmul 函数计算 c 的 j 次乘方，存储在 tgt 中
                tgt = reduce(cheb.chebmul, [c]*j, np.array([1]))
                # 调用 chebpow 函数计算 c 的 j 次幂，存储在 res 中
                res = cheb.chebpow(c, j)
                # 使用 assert_equal 函数验证 trim(res) 是否等于 trim(tgt)，否则输出错误消息 msg
                assert_equal(trim(res), trim(tgt), err_msg=msg)
class TestEvaluation:
    # 定义一维多项式系数 [1, 2, 3]
    c1d = np.array([2.5, 2., 1.5])
    # 计算二维系数矩阵
    c2d = np.einsum('i,j->ij', c1d, c1d)
    # 计算三维系数张量
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # 生成范围在 [-1, 1) 内的随机数
    x = np.random.random((3, 5))*2 - 1
    # 计算多项式在 x 上的值
    y = polyval(x, [1., 2., 3.])

    def test_chebval(self):
        # 检查空输入的情况
        assert_equal(cheb.chebval([], [1]).size, 0)

        # 检查正常输入的情况
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Tlist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            # 检查结果与目标值的近似程度
            res = cheb.chebval(x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # 检查形状是否保持不变
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            # 检查输出的形状
            assert_equal(cheb.chebval(x, [1]).shape, dims)
            assert_equal(cheb.chebval(x, [1, 0]).shape, dims)
            assert_equal(cheb.chebval(x, [1, 0, 0]).shape, dims)

    def test_chebval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises(ValueError, cheb.chebval2d, x1, x2[:2], self.c2d)

        # 测试数值
        tgt = y1*y2
        res = cheb.chebval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 检查输出的形状
        z = np.ones((2, 3))
        res = cheb.chebval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_chebval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises(ValueError, cheb.chebval3d, x1, x2, x3[:2], self.c3d)

        # 测试数值
        tgt = y1*y2*y3
        res = cheb.chebval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # 检查输出的形状
        z = np.ones((2, 3))
        res = cheb.chebval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_chebgrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试数值
        tgt = np.einsum('i,j->ij', y1, y2)
        res = cheb.chebgrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 检查输出的形状
        z = np.ones((2, 3))
        res = cheb.chebgrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3)*2)

    def test_chebgrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试数值
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = cheb.chebgrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # 检查输出的形状
        z = np.ones((2, 3))
        res = cheb.chebgrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3)*3)


class TestIntegral:
    # 定义一个测试方法，用于验证 chebint 函数的 axis 参数是否正常工作
    def test_chebint_axis(self):
        # 创建一个 3x4 的随机数数组
        c2d = np.random.random((3, 4))

        # 对 c2d 的每一列应用 cheb.chebint 函数，并竖直堆叠得到目标数组 tgt
        tgt = np.vstack([cheb.chebint(c) for c in c2d.T]).T
        # 调用 cheb.chebint 函数，指定 axis=0，计算结果存入 res
        res = cheb.chebint(c2d, axis=0)
        # 断言 res 与 tgt 的值几乎相等
        assert_almost_equal(res, tgt)

        # 对 c2d 的每一行应用 cheb.chebint 函数，并竖直堆叠得到目标数组 tgt
        tgt = np.vstack([cheb.chebint(c) for c in c2d])
        # 调用 cheb.chebint 函数，指定 axis=1，计算结果存入 res
        res = cheb.chebint(c2d, axis=1)
        # 断言 res 与 tgt 的值几乎相等
        assert_almost_equal(res, tgt)

        # 对 c2d 的每一行应用 cheb.chebint 函数，并竖直堆叠得到目标数组 tgt，同时指定 k=3
        tgt = np.vstack([cheb.chebint(c, k=3) for c in c2d])
        # 调用 cheb.chebint 函数，指定 axis=1 和 k=3，计算结果存入 res
        res = cheb.chebint(c2d, k=3, axis=1)
        # 断言 res 与 tgt 的值几乎相等
        assert_almost_equal(res, tgt)
class TestDerivative:

    def test_chebder(self):
        # 检查异常情况
        assert_raises(TypeError, cheb.chebder, [0], .5)
        assert_raises(ValueError, cheb.chebder, [0], -1)

        # 检查零阶导数不产生变化
        for i in range(5):
            tgt = [0]*i + [1]
            res = cheb.chebder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # 检查导数与积分的逆过程
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = cheb.chebder(cheb.chebint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # 检查带有缩放的导数
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = cheb.chebder(cheb.chebint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_chebder_axis(self):
        # 检查轴关键字的工作情况
        c2d = np.random.random((3, 4))

        tgt = np.vstack([cheb.chebder(c) for c in c2d.T]).T
        res = cheb.chebder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([cheb.chebder(c) for c in c2d])
        res = cheb.chebder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # 在 [-1, 1) 范围内随机生成一些值
    x = np.random.random((3, 5))*2 - 1

    def test_chebvander(self):
        # 检查 1 维 x 的情况
        x = np.arange(3)
        v = cheb.chebvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], cheb.chebval(x, coef))

        # 检查 2 维 x 的情况
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = cheb.chebvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], cheb.chebval(x, coef))

    def test_chebvander2d(self):
        # 同时测试非方形系数数组的情况，也测试 chebval2d
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = cheb.chebvander2d(x1, x2, [1, 2])
        tgt = cheb.chebval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = cheb.chebvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_chebvander3d(self):
        # 同时测试非方形系数数组的情况，也测试 chebval3d
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = cheb.chebvander3d(x1, x2, x3, [1, 2, 3])
        tgt = cheb.chebval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = cheb.chebvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

class TestInterpolate:

    def f(self, x):
        return x * (x - 1) * (x - 2)
    # 定义测试函数，用于检查是否引发特定异常
    def test_raises(self):
        # 断言调用 cheb.chebinterpolate(self.f, -1) 会引发 ValueError 异常
        assert_raises(ValueError, cheb.chebinterpolate, self.f, -1)
        # 断言调用 cheb.chebinterpolate(self.f, 10.) 会引发 TypeError 异常
        assert_raises(TypeError, cheb.chebinterpolate, self.f, 10.)

    # 定义测试函数，用于检查插值结果的维度是否符合预期
    def test_dimensions(self):
        # 遍历多个插值的阶数
        for deg in range(1, 5):
            # 断言 cheb.chebinterpolate(self.f, deg) 的形状为 (deg + 1,)
            assert_(cheb.chebinterpolate(self.f, deg).shape == (deg + 1,))

    # 定义测试函数，用于检查插值的逼近精度
    def test_approximation(self):

        # 定义一个函数 powx，用于计算 x 的 p 次幂
        def powx(x, p):
            return x**p

        # 生成一个包含 10 个均匀分布点的数组 x
        x = np.linspace(-1, 1, 10)
        
        # 遍历不同的插值阶数 deg
        for deg in range(0, 10):
            # 遍历每个阶数下的指数 p
            for p in range(0, deg + 1):
                # 调用 cheb.chebinterpolate(powx, deg, (p,)) 进行 Chebyshev 插值
                c = cheb.chebinterpolate(powx, deg, (p,))
                # 断言 Chebyshev 插值结果 cheb.chebval(x, c) 与真实值 powx(x, p) 的近似精度在小数点后 12 位
                assert_almost_equal(cheb.chebval(x, c), powx(x, p), decimal=12)
class TestCompanion:

    # 检查在调用 cheb.chebcompanion 函数时是否会引发 ValueError 异常
    def test_raises(self):
        assert_raises(ValueError, cheb.chebcompanion, [])
        assert_raises(ValueError, cheb.chebcompanion, [1])

    # 检查 cheb.chebcompanion 函数返回的矩阵形状是否符合预期
    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0]*i + [1]
            assert_(cheb.chebcompanion(coef).shape == (i, i))

    # 检查对于线性多项式，cheb.chebcompanion 函数返回的首个元素是否正确
    def test_linear_root(self):
        assert_(cheb.chebcompanion([1, 2])[0, 0] == -.5)


class TestGauss:

    # 测试 cheb.chebgauss 函数计算结果的正交性
    def test_100(self):
        x, w = cheb.chebgauss(100)

        # 测试正交性。注意需要对结果进行归一化，以避免由于类似 Laguerre 函数的快速增长而导致的混淆
        v = cheb.chebvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # 检查单位函数的积分是否正确
        tgt = np.pi
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    # 测试 cheb.chebfromroots 函数对于空列表输入的情况
    def test_chebfromroots(self):
        res = cheb.chebfromroots([])
        assert_almost_equal(trim(res), [1])

        # 测试 cheb.chebfromroots 函数对于不同根数的情况
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            tgt = [0]*i + [1]
            res = cheb.chebfromroots(roots)*2**(i-1)
            assert_almost_equal(trim(res), trim(tgt))

    # 测试 cheb.chebroots 函数对于不同情况下的根的计算
    def test_chebroots(self):
        assert_almost_equal(cheb.chebroots([1]), [])
        assert_almost_equal(cheb.chebroots([1, 2]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = cheb.chebroots(cheb.chebfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    # 测试 cheb.chebtrim 函数对于多项式系数的截断
    def test_chebtrim(self):
        coef = [2, -1, 1, 0]

        # 测试异常情况
        assert_raises(ValueError, cheb.chebtrim, coef, -1)

        # 测试结果
        assert_equal(cheb.chebtrim(coef), coef[:-1])
        assert_equal(cheb.chebtrim(coef, 1), coef[:-3])
        assert_equal(cheb.chebtrim(coef, 2), [0])

    # 测试 cheb.chebline 函数的行为
    def test_chebline(self):
        assert_equal(cheb.chebline(3, 4), [3, 4])

    # 测试 cheb.cheb2poly 函数的行为
    def test_cheb2poly(self):
        for i in range(10):
            assert_almost_equal(cheb.cheb2poly([0]*i + [1]), Tlist[i])

    # 测试 cheb.poly2cheb 函数的行为
    def test_poly2cheb(self):
        for i in range(10):
            assert_almost_equal(cheb.poly2cheb(Tlist[i]), [0]*i + [1])

    # 测试 cheb.chebweight 函数的行为
    def test_weight(self):
        x = np.linspace(-1, 1, 11)[1:-1]
        tgt = 1./(np.sqrt(1 + x) * np.sqrt(1 - x))
        res = cheb.chebweight(x)
        assert_almost_equal(res, tgt)
    # 定义测试函数 test_chebpts1，用于测试 cheb.chebpts1 函数的不同输入情况
    def test_chebpts1(self):
        # 测试异常情况：当输入为浮点数 1.5 时，应抛出 ValueError 异常
        assert_raises(ValueError, cheb.chebpts1, 1.5)
        # 测试异常情况：当输入为整数 0 时，应抛出 ValueError 异常
        assert_raises(ValueError, cheb.chebpts1, 0)

        # 测试函数的输出结果是否与预期目标 tgt 相近
        tgt = [0]
        assert_almost_equal(cheb.chebpts1(1), tgt)
        tgt = [-0.70710678118654746, 0.70710678118654746]
        assert_almost_equal(cheb.chebpts1(2), tgt)
        tgt = [-0.86602540378443871, 0, 0.86602540378443871]
        assert_almost_equal(cheb.chebpts1(3), tgt)
        tgt = [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]
        assert_almost_equal(cheb.chebpts1(4), tgt)

    # 定义测试函数 test_chebpts2，用于测试 cheb.chebpts2 函数的不同输入情况
    def test_chebpts2(self):
        # 测试异常情况：当输入为浮点数 1.5 时，应抛出 ValueError 异常
        assert_raises(ValueError, cheb.chebpts2, 1.5)
        # 测试异常情况：当输入为整数 1 时，应抛出 ValueError 异常
        assert_raises(ValueError, cheb.chebpts2, 1)

        # 测试函数的输出结果是否与预期目标 tgt 相近
        tgt = [-1, 1]
        assert_almost_equal(cheb.chebpts2(2), tgt)
        tgt = [-1, 0, 1]
        assert_almost_equal(cheb.chebpts2(3), tgt)
        tgt = [-1, -0.5, .5, 1]
        assert_almost_equal(cheb.chebpts2(4), tgt)
        tgt = [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
        assert_almost_equal(cheb.chebpts2(5), tgt)
```