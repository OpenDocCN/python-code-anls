# `.\numpy\numpy\polynomial\tests\test_polynomial.py`

```py
"""Tests for polynomial module.

"""
从 functools 模块导入 reduce 函数，用于多项式操作
从 fractions 模块导入 Fraction 类，处理有理数
导入 numpy 库并将其重命名为 np，用于数值计算
从 numpy.polynomial.polynomial 模块导入 poly 对象，用于多项式操作
导入 pickle 模块，用于序列化和反序列化对象
从 copy 模块导入 deepcopy 函数，用于深拷贝对象
从 numpy.testing 模块导入多个断言函数，用于单元测试断言

定义函数 trim，接受参数 x，调用 poly.polytrim 函数并传入 tol=1e-6 参数，用于修剪多项式的小系数

定义一系列多项式 T0 到 T9，每个多项式都是一个列表，表示多项式的系数

将多项式 T0 到 T9 组成一个列表 Tlist

定义类 TestConstants，用于测试多项式模块中的常数和常量

    定义方法 test_polydomain，断言 poly.polydomain 的值为 [-1, 1]

    定义方法 test_polyzero，断言 poly.polyzero 的值为 [0]

    定义方法 test_polyone，断言 poly.polyone 的值为 [1]

    定义方法 test_polyx，断言 poly.polyx 的值为 [0, 1]

    定义方法 test_copy，创建多项式对象 x，深拷贝 x 并赋值给 y，断言 x 和 y 的值相等

    定义方法 test_pickle，创建多项式对象 x，使用 pickle 序列化和反序列化 x，断言 x 和 y 的值相等

定义类 TestArithmetic，用于测试多项式模块中的算术运算

    定义方法 test_polyadd，使用嵌套的循环遍历范围在 0 到 4 的整数 i 和 j，并生成详细的错误消息。
    创建目标数组 tgt，对于每次迭代，调用 poly.polyadd 函数计算两个多项式的和，并使用 trim 函数修剪结果，断言结果等于目标数组 tgt。

    定义方法 test_polysub，使用嵌套的循环遍历范围在 0 到 4 的整数 i 和 j，并生成详细的错误消息。
    创建目标数组 tgt，对于每次迭代，调用 poly.polysub 函数计算两个多项式的差，并使用 trim 函数修剪结果，断言结果等于目标数组 tgt。

    定义方法 test_polymulx，断言 poly.polymulx([0]) 的结果为 [0]，poly.polymulx([1]) 的结果为 [0, 1]。
    使用循环遍历范围在 1 到 4 的整数 i，创建系数数组 ser，创建目标数组 tgt，并断言 poly.polymulx 函数计算结果等于目标数组 tgt。

    定义方法 test_polymul，使用嵌套的循环遍历范围在 0 到 4 的整数 i 和 j，并生成详细的错误消息。
    创建目标数组 tgt，对于每次迭代，调用 poly.polymul 函数计算两个多项式的乘积，并使用 trim 函数修剪结果，断言结果等于目标数组 tgt。
    # 定义单元测试函数 test_polydiv
    def test_polydiv(self):
        # 检查零除错误情况
        assert_raises(ZeroDivisionError, poly.polydiv, [1], [0])

        # 检查标量除法
        quo, rem = poly.polydiv([2], [2])
        assert_equal((quo, rem), (1, 0))
        quo, rem = poly.polydiv([2, 2], [2])
        assert_equal((quo, rem), ((1, 1), 0))

        # 循环检查多种情况
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"  # 错误消息标识
                ci = [0]*i + [1, 2]  # 构造多项式 ci
                cj = [0]*j + [1, 2]  # 构造多项式 cj
                tgt = poly.polyadd(ci, cj)  # 计算多项式 ci 和 cj 的和
                quo, rem = poly.polydiv(tgt, ci)  # 对和多项式进行除法，得到商和余数
                res = poly.polyadd(poly.polymul(quo, ci), rem)  # 将商和余数重新相加得到结果
                assert_equal(res, tgt, err_msg=msg)  # 断言结果等于目标值

    # 定义单元测试函数 test_polypow
    def test_polypow(self):
        # 循环检查多种情况
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"  # 错误消息标识
                c = np.arange(i + 1)  # 创建系数数组 c
                tgt = reduce(poly.polymul, [c]*j, np.array([1]))  # 计算多项式的 j 次幂的目标值
                res = poly.polypow(c, j)   # 计算多项式的 j 次幂的结果
                assert_equal(trim(res), trim(tgt), err_msg=msg)  # 断言结果与目标值相等
class TestFraction:
    
    def test_Fraction(self):
        # 创建分数对象 f = 2/3
        f = Fraction(2, 3)
        # 创建分数对象 one = 1/1
        one = Fraction(1, 1)
        # 创建分数对象 zero = 0/1
        zero = Fraction(0, 1)
        # 使用多项式类创建对象 p，其中包含两个 f 作为系数，定义域为 [zero, one]，窗口也为 [zero, one]
        p = poly.Polynomial([f, f], domain=[zero, one], window=[zero, one])
        
        # 计算表达式 x = 2 * p + p ** 2
        x = 2 * p + p ** 2
        # 断言 x 的系数为 [16/9, 20/9, 4/9]，且数据类型为 object
        assert_equal(x.coef, np.array([Fraction(16, 9), Fraction(20, 9),
                                       Fraction(4, 9)], dtype=object))
        # 断言 p 的定义域为 [zero, one]
        assert_equal(p.domain, [zero, one])
        # 断言 p 的系数的数据类型为 np.dtypes.ObjectDType()
        assert_equal(p.coef.dtype, np.dtypes.ObjectDType())
        # 断言 p(f) 的返回值类型为 Fraction
        assert_(isinstance(p(f), Fraction))
        # 断言 p(f) 的值为 10/9
        assert_equal(p(f), Fraction(10, 9))
        
        # 创建一阶导数对象 p_deriv = 2/3，定义域为 [zero, one]，窗口为 [zero, one]
        p_deriv = poly.Polynomial([Fraction(2, 3)], domain=[zero, one],
                                  window=[zero, one])
        # 断言 p 的一阶导数为 p_deriv
        assert_equal(p.deriv(), p_deriv)


class TestEvaluation:
    # 创建系数为 [1., 2., 3.] 的 1D 数组 c1d
    c1d = np.array([1., 2., 3.])
    # 创建 c1d 与自身的外积，形成 2D 数组 c2d
    c2d = np.einsum('i,j->ij', c1d, c1d)
    # 创建 c1d 与自身的外积与自身的外积，形成 3D 数组 c3d
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # 创建大小为 (3, 5) 的随机数数组 x，元素在 [-1, 1) 范围内
    x = np.random.random((3, 5))*2 - 1
    # 使用 poly.polyval 函数计算多项式在 x 上的值，系数为 [1., 2., 3.]
    y = poly.polyval(x, [1., 2., 3.])

    def test_polyval(self):
        # 检查空输入的情况，预期输出应该是 0
        assert_equal(poly.polyval([], [1]).size, 0)

        # 检查正常输入情况
        x = np.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(5):
            tgt = y[i]
            res = poly.polyval(x, [0]*i + [1])
            # 断言 poly.polyval 的结果与预期值 tgt 几乎相等
            assert_almost_equal(res, tgt)
        tgt = x*(x**2 - 1)
        res = poly.polyval(x, [0, -1, 0, 1])
        # 断言 poly.polyval 的结果与预期值 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 检查保持形状不变
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            # 断言 poly.polyval 的输出形状与输入形状 dims 相同
            assert_equal(poly.polyval(x, [1]).shape, dims)
            assert_equal(poly.polyval(x, [1, 0]).shape, dims)
            assert_equal(poly.polyval(x, [1, 0, 0]).shape, dims)

        # 检查处理掩码数组的情况
        mask = [False, True, False]
        mx = np.ma.array([1, 2, 3], mask=mask)
        res = np.polyval([7, 5, 3], mx)
        # 断言 np.polyval 的结果掩码与输入掩码相同
        assert_array_equal(res.mask, mask)

        # 检查保持 ndarray 的子类型
        class C(np.ndarray):
            pass

        cx = np.array([1, 2, 3]).view(C)
        # 断言 np.polyval 的返回类型与输入类型相同
        assert_equal(type(np.polyval([2, 3, 4], cx)), C)
    def test_polyvalfromroots(self):
        # 检查在根数组上广播 x 值时抛出异常，因为根数组维度太少
        assert_raises(ValueError, poly.polyvalfromroots,
                      [1], [1], tensor=False)

        # 检查空输入
        assert_equal(poly.polyvalfromroots([], [1]).size, 0)
        assert_(poly.polyvalfromroots([], [1]).shape == (0,))

        # 检查空输入 + 多维根数组
        assert_equal(poly.polyvalfromroots([], [[1] * 5]).size, 0)
        assert_(poly.polyvalfromroots([], [[1] * 5]).shape == (5, 0))

        # 检查标量输入
        assert_equal(poly.polyvalfromroots(1, 1), 0)
        assert_(poly.polyvalfromroots(1, np.ones((3, 3))).shape == (3,))

        # 检查正常输入
        x = np.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(1, 5):
            tgt = y[i]
            res = poly.polyvalfromroots(x, [0]*i)
            assert_almost_equal(res, tgt)
        tgt = x*(x - 1)*(x + 1)
        res = poly.polyvalfromroots(x, [-1, 0, 1])
        assert_almost_equal(res, tgt)

        # 检查形状保持不变
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(poly.polyvalfromroots(x, [1]).shape, dims)
            assert_equal(poly.polyvalfromroots(x, [1, 0]).shape, dims)
            assert_equal(poly.polyvalfromroots(x, [1, 0, 0]).shape, dims)

        # 检查与因式分解的兼容性
        ptest = [15, 2, -16, -2, 1]
        r = poly.polyroots(ptest)
        x = np.linspace(-1, 1)
        assert_almost_equal(poly.polyval(x, ptest),
                            poly.polyvalfromroots(x, r))

        # 检查多维根数组和值数组
        # 检查 tensor=False
        rshape = (3, 5)
        x = np.arange(-3, 2)
        r = np.random.randint(-5, 5, size=rshape)
        res = poly.polyvalfromroots(x, r, tensor=False)
        tgt = np.empty(r.shape[1:])
        for ii in range(tgt.size):
            tgt[ii] = poly.polyvalfromroots(x[ii], r[:, ii])
        assert_equal(res, tgt)

        # 检查 tensor=True
        x = np.vstack([x, 2*x])
        res = poly.polyvalfromroots(x, r, tensor=True)
        tgt = np.empty(r.shape[1:] + x.shape)
        for ii in range(r.shape[1]):
            for jj in range(x.shape[0]):
                tgt[ii, jj, :] = poly.polyvalfromroots(x[jj], r[:, ii])
        assert_equal(res, tgt)

    def test_polyval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # 测试异常情况
        assert_raises_regex(ValueError, 'incompatible',
                            poly.polyval2d, x1, x2[:2], self.c2d)

        # 测试数值计算
        tgt = y1*y2
        res = poly.polyval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # 测试形状
        z = np.ones((2, 3))
        res = poly.polyval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))
    # 定义测试多项式三维求值的方法
    def test_polyval3d(self):
        # 从 self.x 中解包得到 x1, x2, x3
        x1, x2, x3 = self.x
        # 从 self.y 中解包得到 y1, y2, y3

        # 测试异常情况，断言抛出 ValueError 异常并包含 'incompatible' 字符串
        assert_raises_regex(ValueError, 'incompatible',
                      poly.polyval3d, x1, x2, x3[:2], self.c3d)

        # 计算目标值 tgt
        tgt = y1 * y2 * y3
        # 使用 poly.polyval3d 计算结果 res
        res = poly.polyval3d(x1, x2, x3, self.c3d)
        # 断言 res 与 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 测试返回结果的形状
        z = np.ones((2, 3))
        # 使用 poly.polyval3d 计算结果 res
        res = poly.polyval3d(z, z, z, self.c3d)
        # 断言 res 的形状为 (2, 3)
        assert_(res.shape == (2, 3))

    # 定义测试二维多项式网格化的方法
    def test_polygrid2d(self):
        # 从 self.x 中解包得到 x1, x2, x3
        x1, x2, x3 = self.x
        # 从 self.y 中解包得到 y1, y2, y3

        # 计算目标值 tgt
        tgt = np.einsum('i,j->ij', y1, y2)
        # 使用 poly.polygrid2d 计算结果 res
        res = poly.polygrid2d(x1, x2, self.c2d)
        # 断言 res 与 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 测试返回结果的形状
        z = np.ones((2, 3))
        # 使用 poly.polygrid2d 计算结果 res
        res = poly.polygrid2d(z, z, self.c2d)
        # 断言 res 的形状为 (2, 3, 2)
        assert_(res.shape == (2, 3)*2)

    # 定义测试三维多项式网格化的方法
    def test_polygrid3d(self):
        # 从 self.x 中解包得到 x1, x2, x3
        x1, x2, x3 = self.x
        # 从 self.y 中解包得到 y1, y2, y3

        # 计算目标值 tgt
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        # 使用 poly.polygrid3d 计算结果 res
        res = poly.polygrid3d(x1, x2, x3, self.c3d)
        # 断言 res 与 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 测试返回结果的形状
        z = np.ones((2, 3))
        # 使用 poly.polygrid3d 计算结果 res
        res = poly.polygrid3d(z, z, z, self.c3d)
        # 断言 res 的形状为 (2, 3, 3)
        assert_(res.shape == (2, 3)*3)
class TestIntegral:

    def test_polyint_axis(self):
        # 检查轴关键字的工作情况
        c2d = np.random.random((3, 4))  # 创建一个 3x4 的随机数组

        # 在轴 0 上进行多项式积分，并将结果垂直堆叠以获得目标数组
        tgt = np.vstack([poly.polyint(c) for c in c2d.T]).T
        res = poly.polyint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        # 在轴 1 上进行多项式积分，并将结果垂直堆叠以获得目标数组
        tgt = np.vstack([poly.polyint(c) for c in c2d])
        res = poly.polyint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        # 在轴 1 上进行多项式积分，同时指定多项式的次数 k=3，并将结果垂直堆叠以获得目标数组
        tgt = np.vstack([poly.polyint(c, k=3) for c in c2d])
        res = poly.polyint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_polyder(self):
        # 检查异常情况：期望抛出 TypeError 异常
        assert_raises(TypeError, poly.polyder, [0], .5)
        # 检查异常情况：期望抛出 ValueError 异常
        assert_raises(ValueError, poly.polyder, [0], -1)

        # 检查零阶导数不做任何操作
        for i in range(5):
            tgt = [0]*i + [1]
            res = poly.polyder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # 检查导数与积分的反向关系
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                # 对多次积分的多项式进行 j 阶导数运算，并断言与原始多项式相等
                res = poly.polyder(poly.polyint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # 检查带有缩放的导数运算
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                # 对多次积分的多项式进行 j 阶导数运算，并指定不同的缩放参数，断言结果与原始多项式相等
                res = poly.polyder(poly.polyint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_polyder_axis(self):
        # 检查轴关键字的工作情况
        c2d = np.random.random((3, 4))  # 创建一个 3x4 的随机数组

        # 在轴 0 上进行多项式导数，并将结果垂直堆叠以获得目标数组
        tgt = np.vstack([poly.polyder(c) for c in c2d.T]).T
        res = poly.polyder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        # 在轴 1 上进行多项式导数，并将结果垂直堆叠以获得目标数组
        tgt = np.vstack([poly.polyder(c) for c in c2d])
        res = poly.polyder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # 生成在 [-1, 1) 范围内的随机值
    x = np.random.random((3, 5))*2 - 1

    def test_polyvander(self):
        # 检查 1 维 x 的情况
        x = np.arange(3)
        v = poly.polyvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            # 断言多项式的 Vandermonde 矩阵乘以系数向量的结果与多项式的值相等
            assert_almost_equal(v[..., i], poly.polyval(x, coef))

        # 检查 2 维 x 的情况
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = poly.polyvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            # 断言多项式的 Vandermonde 矩阵乘以系数向量的结果与多项式的值相等
            assert_almost_equal(v[..., i], poly.polyval(x, coef))

    def test_polyvander2d(self):
        # 同时测试非方形系数数组的 polyval2d
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = poly.polyvander2d(x1, x2, [1, 2])
        tgt = poly.polyval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # 检查形状
        van = poly.polyvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))
    # 定义一个测试方法，用于测试 polyvander3d 函数
    def test_polyvander3d(self):
        # 同时也测试非方形系数数组的 polyval3d 函数
        x1, x2, x3 = self.x
        # 创建一个 2x3x4 的随机数组作为系数 c
        c = np.random.random((2, 3, 4))
        # 使用 polyvander3d 函数生成三维多项式的 Vandermonde 矩阵 van
        van = poly.polyvander3d(x1, x2, x3, [1, 2, 3])
        # 使用 polyval3d 函数计算目标值 tgt
        tgt = poly.polyval3d(x1, x2, x3, c)
        # 将 van 和 c 的扁平版本相乘得到结果 res
        res = np.dot(van, c.flat)
        # 使用 assert_almost_equal 断言 res 与 tgt 几乎相等
        assert_almost_equal(res, tgt)

        # 检查 van 的形状
        van = poly.polyvander3d([x1], [x2], [x3], [1, 2, 3])
        # 使用 assert_ 断言 van 的形状为 (1, 5, 24)
        assert_(van.shape == (1, 5, 24))

    # 定义一个测试方法，用于测试 polyvander 函数处理负阶数的情况
    def test_polyvandernegdeg(self):
        # 创建一个包含 0, 1, 2 的 numpy 数组 x
        x = np.arange(3)
        # 使用 assert_raises 断言 polyvander 函数对于负阶数会引发 ValueError 异常
        assert_raises(ValueError, poly.polyvander, x, -1)
# 定义名为 TestCompanion 的测试类，用于测试 polycompanion 函数的异常情况和维度
class TestCompanion:

    # 测试函数，验证当输入为空列表时是否会引发 ValueError 异常
    def test_raises(self):
        assert_raises(ValueError, poly.polycompanion, [])
        # 验证当输入只包含一个元素时是否会引发 ValueError 异常
        assert_raises(ValueError, poly.polycompanion, [1])

    # 测试函数，验证 polycompanion 函数生成的 companiom 矩阵的维度是否正确
    def test_dimensions(self):
        # 对于范围从 1 到 4 的每一个整数 i
        for i in range(1, 5):
            # 构造一个系数列表，前 i-1 个元素为 0，最后一个元素为 1
            coef = [0]*i + [1]
            # 验证生成的 companiom 矩阵的形状是否为 (i, i)
            assert_(poly.polycompanion(coef).shape == (i, i))

    # 测试函数，验证 polycompanion 函数对线性多项式 [1, 2] 的 companiom 矩阵的第一个元素是否为 -0.5
    def test_linear_root(self):
        assert_(poly.polycompanion([1, 2])[0, 0] == -.5)


# 定义名为 TestMisc 的测试类，用于测试 polyfromroots 和 polyroots 函数的正确性
class TestMisc:

    # 测试函数，验证 polyfromroots 函数对不同情况下生成的多项式系数是否正确
    def test_polyfromroots(self):
        # 对于空根列表，验证生成的多项式系数是否接近于 [1]
        res = poly.polyfromroots([])
        assert_almost_equal(trim(res), [1])
        # 对于范围从 1 到 4 的每一个整数 i
        for i in range(1, 5):
            # 生成 cos 函数值作为根的列表
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            # 获取目标多项式系数列表
            tgt = Tlist[i]
            # 根据根列表生成多项式，并乘以 2 的 (i-1) 次方
            res = poly.polyfromroots(roots)*2**(i-1)
            # 验证生成的多项式系数是否接近于目标多项式系数列表
            assert_almost_equal(trim(res), trim(tgt))

    # 测试函数，验证 polyroots 函数对不同多项式系数列表的求根结果是否正确
    def test_polyroots(self):
        # 验证对于 [1] 的多项式系数列表，求根结果是否为空列表
        assert_almost_equal(poly.polyroots([1]), [])
        # 验证对于 [1, 2] 的多项式系数列表，求根结果是否接近于 [-0.5]
        assert_almost_equal(poly.polyroots([1, 2]), [-.5])
        # 对于范围从 2 到 4 的每一个整数 i
        for i in range(2, 5):
            # 生成线性等间距分布的目标根列表
            tgt = np.linspace(-1, 1, i)
            # 根据目标根列表生成多项式系数列表，并求其根
            res = poly.polyroots(poly.polyfromroots(tgt))
            # 验证求得的多项式根是否接近于目标根列表
            assert_almost_equal(trim(res), trim(tgt))
    # 定义一个函数 f(x)，返回 x*(x - 1)*(x - 2) 的结果
    def f(x):
        return x*(x - 1)*(x - 2)

    # 定义一个函数 f2(x)，返回 x**4 + x**2 + 1 的结果
    def f2(x):
        return x**4 + x**2 + 1

    # 测试异常情况
    assert_raises(ValueError, poly.polyfit, [1], [1], -1)  # 检查是否抛出 ValueError 异常
    assert_raises(TypeError, poly.polyfit, [[1]], [1], 0)   # 检查是否抛出 TypeError 异常
    assert_raises(TypeError, poly.polyfit, [], [1], 0)      # 检查是否抛出 TypeError 异常
    assert_raises(TypeError, poly.polyfit, [1], [[[1]]], 0)  # 检查是否抛出 TypeError 异常
    assert_raises(TypeError, poly.polyfit, [1, 2], [1], 0)   # 检查是否抛出 TypeError 异常
    assert_raises(TypeError, poly.polyfit, [1], [1, 2], 0)   # 检查是否抛出 TypeError 异常
    assert_raises(TypeError, poly.polyfit, [1], [1], 0, w=[[1]])  # 检查是否抛出 TypeError 异常
    assert_raises(TypeError, poly.polyfit, [1], [1], 0, w=[1, 1])   # 检查是否抛出 TypeError 异常
    assert_raises(ValueError, poly.polyfit, [1], [1], [-1,])  # 检查是否抛出 ValueError 异常
    assert_raises(ValueError, poly.polyfit, [1], [1], [2, -1, 6])  # 检查是否抛出 ValueError 异常
    assert_raises(TypeError, poly.polyfit, [1], [1], [])  # 检查是否抛出 TypeError 异常

    # 测试多项式拟合
    x = np.linspace(0, 2)  # 生成一个从0到2的等间隔数列作为 x 值
    y = f(x)  # 计算函数 f(x) 在 x 值上的结果

    # 拟合三次多项式并验证结果
    coef3 = poly.polyfit(x, y, 3)
    assert_equal(len(coef3), 4)  # 验证多项式系数的长度是否为4
    assert_almost_equal(poly.polyval(x, coef3), y)  # 验证通过多项式计算的值是否接近真实值

    # 通过指定多项式的系数来拟合并验证结果
    coef3 = poly.polyfit(x, y, [0, 1, 2, 3])
    assert_equal(len(coef3), 4)  # 验证多项式系数的长度是否为4
    assert_almost_equal(poly.polyval(x, coef3), y)  # 验证通过多项式计算的值是否接近真实值

    # 拟合四次多项式并验证结果
    coef4 = poly.polyfit(x, y, 4)
    assert_equal(len(coef4), 5)  # 验证多项式系数的长度是否为5
    assert_almost_equal(poly.polyval(x, coef4), y)  # 验证通过多项式计算的值是否接近真实值

    # 通过指定多项式的系数来拟合并验证结果
    coef4 = poly.polyfit(x, y, [0, 1, 2, 3, 4])
    assert_equal(len(coef4), 5)  # 验证多项式系数的长度是否为5
    assert_almost_equal(poly.polyval(x, coef4), y)  # 验证通过多项式计算的值是否接近真实值

    # 拟合二维数据并验证结果
    coef2d = poly.polyfit(x, np.array([y, y]).T, 3)
    assert_almost_equal(coef2d, np.array([coef3, coef3]).T)  # 验证拟合的二维数据是否与预期的系数一致

    # 通过指定多项式的系数来拟合二维数据并验证结果
    coef2d = poly.polyfit(x, np.array([y, y]).T, [0, 1, 2, 3])
    assert_almost_equal(coef2d, np.array([coef3, coef3]).T)  # 验证拟合的二维数据是否与预期的系数一致

    # 测试加权拟合
    w = np.zeros_like(x)  # 创建与 x 形状相同的全零数组作为权重
    yw = y.copy()  # 复制 y 数组作为加权后的 y 值
    w[1::2] = 1  # 设置奇数索引位置的权重为1
    yw[0::2] = 0  # 将偶数索引位置的 y 值置为0，模拟加权效果
    wcoef3 = poly.polyfit(x, yw, 3, w=w)  # 加权拟合三次多项式并验证结果
    assert_almost_equal(wcoef3, coef3)  # 验证加权拟合的结果是否与未加权的结果一致

    # 通过指定多项式的系数来加权拟合并验证结果
    wcoef3 = poly.polyfit(x, yw, [0, 1, 2, 3], w=w)
    assert_almost_equal(wcoef3, coef3)  # 验证加权拟合的结果是否与未加权的结果一致

    # 拟合二维加权数据并验证结果
    wcoef2d = poly.polyfit(x, np.array([yw, yw]).T, 3, w=w)
    assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)  # 验证拟合的二维加权数据是否与预期的系数一致

    # 通过指定多项式的系数来拟合二维加权数据并验证结果
    wcoef2d = poly.polyfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)  # 验证拟合的二维加权数据是否与预期的系数一致

    # 测试使用复数值的情况，其中 x 值的平方和为零
    x = [1, 1j, -1, -1j]
    assert_almost_equal(poly.polyfit(x, x, 1), [0, 1])  # 验证在复数值上拟合一次多项式的结果

    # 通过指定多项式的系数来拟合复数值的情况
    assert_almost_equal(poly.polyfit(x, x, [0, 1]), [0, 1])  # 验证在复数值上拟合一次多项式的结果

    # 测试仅拟合偶数次勒让德多项式的情况
    x = np.linspace(-1, 1)  # 生成一个从-1到1的等间隔数列作为 x 值
    y = f2(x)  # 计算函数 f2(x) 在 x 值上的结果

    # 拟合四次勒让德多项式并验证结果
    coef1 = poly.polyfit(x, y, 4)
    assert_almost_equal(poly.polyval(x, coef1), y)  # 验证通过多项式计算的值是否接近真实值

    # 通过指定多项式的系数来仅拟合偶数次勒让德多项式并验证结果
    coef2 = poly.polyfit(x, y, [0, 2, 4])
    assert_almost_equal(poly.polyval(x, coef2), y)  # 验证通过多项式计算的值是否接近真实值

    assert_almost_equal(coef1, coef2)  # 验证两种方式得到的系数是否一致
    # 定义一个测试方法，用于测试 poly.polytrim 函数
    def test_polytrim(self):
        # 设置一个多项式的系数列表
        coef = [2, -1, 1, 0]

        # 测试异常情况：验证当指定的阶数为负数时，是否引发 ValueError 异常
        assert_raises(ValueError, poly.polytrim, coef, -1)

        # 测试正常情况下的结果：
        # 1. 验证默认情况下，poly.polytrim 函数能够正确地截取掉最高次项系数后的多项式系数列表
        assert_equal(poly.polytrim(coef), coef[:-1])
        # 2. 验证当指定截取的阶数为1时，poly.polytrim 函数是否正确截取掉最高次和次高次项系数
        assert_equal(poly.polytrim(coef, 1), coef[:-3])
        # 3. 验证当指定截取的阶数为2时，poly.polytrim 函数是否返回只有常数项的列表 [0]
        assert_equal(poly.polytrim(coef, 2), [0])

    # 定义一个测试方法，用于测试 poly.polyline 函数
    def test_polyline(self):
        # 验证 poly.polyline 函数是否能正确返回给定参数的列表表示的多项式
        assert_equal(poly.polyline(3, 4), [3, 4])

    # 定义一个测试方法，用于测试 poly.polyline 函数中参数为0的情况
    def test_polyline_zero(self):
        # 验证 poly.polyline 函数当第二个参数为0时，是否正确返回只含有第一个参数的列表
        assert_equal(poly.polyline(3, 0), [3])
```