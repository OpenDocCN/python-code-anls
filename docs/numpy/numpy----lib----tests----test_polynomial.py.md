# `.\numpy\numpy\lib\tests\test_polynomial.py`

```
import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_raises, assert_allclose
    )

import pytest

# 定义一个字符串，包含所有整数和浮点数类型码，但不包括布尔和时间类型
TYPE_CODES = np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "O"

# 定义测试类 TestPolynomial
class TestPolynomial:
    
    # 测试方法：测试 poly1d 对象的字符串表示和表达式
    def test_poly1d_str_and_repr(self):
        p = np.poly1d([1., 2, 3])
        # 断言 poly1d 对象的 repr 结果
        assert_equal(repr(p), 'poly1d([1., 2., 3.])')
        # 断言 poly1d 对象的 str 结果
        assert_equal(str(p),
                     '   2\n'
                     '1 x + 2 x + 3')

        q = np.poly1d([3., 2, 1])
        assert_equal(repr(q), 'poly1d([3., 2., 1.])')
        assert_equal(str(q),
                     '   2\n'
                     '3 x + 2 x + 1')

        r = np.poly1d([1.89999 + 2j, -3j, -5.12345678, 2 + 1j])
        assert_equal(str(r),
                     '            3      2\n'
                     '(1.9 + 2j) x - 3j x - 5.123 x + (2 + 1j)')

        assert_equal(str(np.poly1d([-3, -2, -1])),
                     '    2\n'
                     '-3 x - 2 x - 1')

    # 测试方法：测试 poly1d 对象的计算能力
    def test_poly1d_resolution(self):
        p = np.poly1d([1., 2, 3])
        q = np.poly1d([3., 2, 1])
        assert_equal(p(0), 3.0)
        assert_equal(p(5), 38.0)
        assert_equal(q(0), 1.0)
        assert_equal(q(5), 86.0)

    # 测试方法：测试 poly1d 对象的数学运算
    def test_poly1d_math(self):
        # 使用简单系数进行测试，以便计算更加简单
        p = np.poly1d([1., 2, 4])
        q = np.poly1d([4., 2, 1])
        assert_equal(p/q, (np.poly1d([0.25]), np.poly1d([1.5, 3.75])))
        assert_equal(p.integ(), np.poly1d([1/3, 1., 4., 0.]))
        assert_equal(p.integ(1), np.poly1d([1/3, 1., 4., 0.]))

        p = np.poly1d([1., 2, 3])
        q = np.poly1d([3., 2, 1])
        assert_equal(p * q, np.poly1d([3., 8., 14., 8., 3.]))
        assert_equal(p + q, np.poly1d([4., 4., 4.]))
        assert_equal(p - q, np.poly1d([-2., 0., 2.]))
        assert_equal(p ** 4, np.poly1d([1., 8., 36., 104., 214., 312., 324., 216., 81.]))
        assert_equal(p(q), np.poly1d([9., 12., 16., 8., 6.]))
        assert_equal(q(p), np.poly1d([3., 12., 32., 40., 34.]))
        assert_equal(p.deriv(), np.poly1d([2., 2.]))
        assert_equal(p.deriv(2), np.poly1d([2.]))
        assert_equal(np.polydiv(np.poly1d([1, 0, -1]), np.poly1d([1, 1])),
                     (np.poly1d([1., -1.]), np.poly1d([0.])))

    # 使用 pytest 的参数化装饰器，参数是 TYPE_CODES 中的类型码
    @pytest.mark.parametrize("type_code", TYPE_CODES)
    # 测试多项式对象的杂项功能，接受一个类型码参数
    def test_poly1d_misc(self, type_code: str) -> None:
        # 根据给定的类型码创建 NumPy 数据类型对象
        dtype = np.dtype(type_code)
        # 创建一个 NumPy 数组，使用指定的数据类型
        ar = np.array([1, 2, 3], dtype=dtype)
        # 使用数组创建多项式对象
        p = np.poly1d(ar)

        # 测试多项式对象的相等性 `__eq__`
        assert_equal(np.asarray(p), ar)
        # 断言多项式对象的数据类型与输入的数据类型一致
        assert_equal(np.asarray(p).dtype, dtype)
        # 断言多项式对象的阶数为 2
        assert_equal(len(p), 2)

        # 测试多项式对象的索引访问 `__getitem__`
        # 准备一个预期的索引与值的对应字典
        comparison_dct = {-1: 0, 0: 3, 1: 2, 2: 1, 3: 0}
        # 遍历字典进行断言测试
        for index, ref in comparison_dct.items():
            # 获取多项式对象在当前索引位置的值
            scalar = p[index]
            # 断言获取的值与预期值相等
            assert_equal(scalar, ref)
            # 如果数据类型是 np.object_，则额外断言获取的值是整数类型
            if dtype == np.object_:
                assert isinstance(scalar, int)
            else:
                # 否则，断言获取的值的数据类型与输入的数据类型一致
                assert_equal(scalar.dtype, dtype)

    # 测试多项式对象使用不同变量参数的行为
    def test_poly1d_variable_arg(self):
        # 使用自定义变量名 'y' 创建多项式对象并断言其字符串表示
        q = np.poly1d([1., 2, 3], variable='y')
        assert_equal(str(q),
                     '   2\n'
                     '1 y + 2 y + 3')
        # 使用自定义变量名 'lambda' 创建多项式对象并断言其字符串表示
        q = np.poly1d([1., 2, 3], variable='lambda')
        assert_equal(str(q),
                     '        2\n'
                     '1 lambda + 2 lambda + 3')

    # 测试多项式对象的其他功能
    def test_poly(self):
        # 断言计算给定系数的多项式的根，与预期的根数组相等
        assert_array_almost_equal(np.poly([3, -np.sqrt(2), np.sqrt(2)]),
                                  [1, -3, -2, 6])

        # 从 Matlab 文档中复制的测试用例
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        # 断言计算给定系数矩阵的多项式的系数，与预期的系数数组相等
        assert_array_almost_equal(np.poly(A), [1, -6, -72, -27])

        # 测试应该对于完美共轭根产生实数输出
        assert_(np.isrealobj(np.poly([+1.082j, +2.613j, -2.613j, -1.082j])))
        assert_(np.isrealobj(np.poly([0+1j, -0+-1j, 1+2j,
                                      1-2j, 1.+3.5j, 1-3.5j])))
        assert_(np.isrealobj(np.poly([1j, -1j, 1+2j, 1-2j, 1+3j, 1-3.j])))
        assert_(np.isrealobj(np.poly([1j, -1j, 2j, -2j])))
        assert_(np.isrealobj(np.poly([1j, -1j])))
        assert_(np.isrealobj(np.poly([1, -1])))

        # 断言计算给定系数的多项式的根，与预期的结果相符
        assert_(np.iscomplexobj(np.poly([1j, -1.0000001j])))

        # 随机生成一组复数系数，并断言计算多项式的根的实部结果
        np.random.seed(42)
        a = np.random.randn(100) + 1j*np.random.randn(100)
        assert_(np.isrealobj(np.poly(np.concatenate((a, np.conjugate(a))))))

    # 测试计算多项式的根功能
    def test_roots(self):
        # 断言计算给定系数的多项式的根，与预期的根数组相等
        assert_array_equal(np.roots([1, 0, 0]), [0, 0])

    # 测试多项式对象字符串表示中的前导零处理
    def test_str_leading_zeros(self):
        # 创建一个具有给定系数的多项式对象，并设置其中一个系数为零，断言其字符串表示
        p = np.poly1d([4, 3, 2, 1])
        p[3] = 0
        assert_equal(str(p),
                     "   2\n"
                     "3 x + 2 x + 1")

        # 创建一个具有给定系数的多项式对象，并将所有系数设置为零，断言其字符串表示
        p = np.poly1d([1, 2])
        p[0] = 0
        p[1] = 0
        assert_equal(str(p), " \n0")
    # 定义测试函数，验证多项式对象的操作
    def test_objects(self):
        # 导入 Decimal 类用于精确计算
        from decimal import Decimal
        # 创建多项式对象 p，系数为 [4.0, 3.0, 2.0]
        p = np.poly1d([Decimal('4.0'), Decimal('3.0'), Decimal('2.0')])
        # 将 p 乘以 Decimal('1.333333333333333') 得到 p2
        p2 = p * Decimal('1.333333333333333')
        # 断言 p2 的索引 1 的值等于 Decimal("3.9999999999999990")
        assert_(p2[1] == Decimal("3.9999999999999990"))
        # 对 p 求导数得到 p2
        p2 = p.deriv()
        # 断言 p2 的索引 1 的值等于 Decimal('8.0')
        assert_(p2[1] == Decimal('8.0'))
        # 对 p 求积分得到 p2
        p2 = p.integ()
        # 断言 p2 的索引 3 的值等于 Decimal("1.333333333333333333333333333")
        assert_(p2[3] == Decimal("1.333333333333333333333333333"))
        # 断言 p2 的索引 2 的值等于 Decimal('1.5')
        assert_(p2[2] == Decimal('1.5'))
        # 断言 p2 的系数的数据类型是 np.object_
        assert_(np.issubdtype(p2.coeffs.dtype, np.object_))
        # 创建多项式对象 p，系数为 [1, -3, 2]，并断言其与给定列表相等
        assert_equal(np.poly([Decimal(1), Decimal(2)]),
                     [1, Decimal(-3), Decimal(2)])

    # 定义测试函数，验证复数系数的多项式对象操作
    def test_complex(self):
        # 创建多项式对象 p，系数为 [3j, 2j, 1j]
        p = np.poly1d([3j, 2j, 1j])
        # 对 p 求积分得到 p2
        p2 = p.integ()
        # 断言 p2 的系数等于 [1j, 1j, 1j, 0] 的所有元素
        assert_((p2.coeffs == [1j, 1j, 1j, 0]).all())
        # 对 p 求导数得到 p2
        p2 = p.deriv()
        # 断言 p2 的系数等于 [6j, 2j] 的所有元素
        assert_((p2.coeffs == [6j, 2j]).all())

    # 定义测试函数，验证多项式积分的系数计算
    def test_integ_coeffs(self):
        # 创建多项式对象 p，系数为 [3, 2, 1]
        p = np.poly1d([3, 2, 1])
        # 对 p 求积分，指定积分常数和系数 k，得到 p2
        p2 = p.integ(3, k=[9, 7, 6])
        # 断言 p2 的系数等于计算结果的所有元素
        assert_(
            (p2.coeffs == [1/4./5., 1/3./4., 1/2./3., 9/1./2., 7, 6]).all())

    # 定义测试函数，验证处理零维多项式的异常
    def test_zero_dims(self):
        # 尝试创建零维多项式，捕获 ValueError 异常
        try:
            np.poly(np.zeros((0, 0)))
        except ValueError:
            pass

    # 定义测试函数，验证多项式整数溢出问题的回归测试
    def test_poly_int_overflow(self):
        """
        Regression test for gh-5096.
        """
        # 创建一个范围为 [1, 20] 的向量 v
        v = np.arange(1, 21)
        # 断言 np.poly(v) 与 np.poly(np.diag(v)) 的近似相等
        assert_almost_equal(np.poly(v), np.poly(np.diag(v)))

    # 定义测试函数，验证多项式零值数据类型问题的回归测试
    def test_zero_poly_dtype(self):
        """
        Regression test for gh-16354.
        """
        # 创建一个值全为零的数组 z
        z = np.array([0, 0, 0])
        # 创建整型系数的多项式对象 p，并断言其系数的数据类型为 np.int64
        p = np.poly1d(z.astype(np.int64))
        assert_equal(p.coeffs.dtype, np.int64)
        # 创建单精度浮点型系数的多项式对象 p，并断言其系数的数据类型为 np.float32
        p = np.poly1d(z.astype(np.float32))
        assert_equal(p.coeffs.dtype, np.float32)
        # 创建复数型系数的多项式对象 p，并断言其系数的数据类型为 np.complex64
        p = np.poly1d(z.astype(np.complex64))
        assert_equal(p.coeffs.dtype, np.complex64)

    # 定义测试函数，验证多项式对象的相等性操作
    def test_poly_eq(self):
        # 创建多项式对象 p，系数为 [1, 2, 3]
        p = np.poly1d([1, 2, 3])
        # 创建多项式对象 p2，系数为 [1, 2, 4]
        p2 = np.poly1d([1, 2, 4])
        # 断言 p 是否等于 None，结果为 False
        assert_equal(p == None, False)
        # 断言 p 是否不等于 None，结果为 True
        assert_equal(p != None, True)
        # 断言 p 是否等于自身，结果为 True
        assert_equal(p == p, True)
        # 断言 p 是否等于 p2，结果为 False
        assert_equal(p == p2, False)
        # 断言 p 是否不等于 p2，结果为 True
        assert_equal(p != p2, True)

    # 定义测试函数，验证多项式除法操作
    def test_polydiv(self):
        # 创建多项式对象 b 和 a
        b = np.poly1d([2, 6, 6, 1])
        a = np.poly1d([-1j, (1+2j), -(2+1j), 1])
        # 对 b 除以 a 得到商 q 和余数 r
        q, r = np.polydiv(b, a)
        # 断言商 q 和余数 r 的系数数据类型为 np.complex128
        assert_equal(q.coeffs.dtype, np.complex128)
        assert_equal(r.coeffs.dtype, np.complex128)
        # 断言 q 乘以 a 再加上 r 等于 b
        assert_equal(q*a + r, b)

        # 创建列表 c 和多项式对象 d
        c = [1, 2, 3]
        d = np.poly1d([1, 2, 3])
        # 对 c 除以 d 得到商 s 和余数 t
        s, t = np.polydiv(c, d)
        # 断言 s 和 t 的类型为 np.poly1d
        assert isinstance(s, np.poly1d)
        assert isinstance(t, np.poly1d)
        # 对 d 除以 c 得到商 u 和余数 v
        u, v = np.polydiv(d, c)
        # 断言 u 和 v 的类型为 np.poly1d
        assert isinstance(u, np.poly1d)
        assert isinstance(v, np.poly1d)
    def test_poly_coeffs_mutable(self):
        """ 测试多项式系数是否可修改 """
        # 创建一个三次多项式对象，系数为 [1, 2, 3]
        p = np.poly1d([1, 2, 3])

        # 修改多项式的系数，应该会使系数增加 1
        p.coeffs += 1
        # 断言修改后的系数应为 [2, 3, 4]
        assert_equal(p.coeffs, [2, 3, 4])

        # 修改系数的第三个元素（系数为 3），增加 10
        p.coeffs[2] += 10
        # 断言修改后的系数应为 [2, 3, 14]
        assert_equal(p.coeffs, [2, 3, 14])

        # 尝试设置系数属性为一个新的 NumPy 数组（这是不允许的，应引发 AttributeError 异常）
        assert_raises(AttributeError, setattr, p, 'coeffs', np.array(1))
```