# `.\numpy\numpy\polynomial\tests\test_printing.py`

```py
# 导入数学模块中的特定常数和函数（NaN和无穷大），用于测试
from math import nan, inf
# 导入 pytest 模块，用于编写和运行测试
import pytest
# 导入 numpy 中的数组、序列和打印选项
from numpy._core import array, arange, printoptions
# 导入 numpy 中的多项式模块
import numpy.polynomial as poly
# 导入 numpy 测试模块中的断言函数
from numpy.testing import assert_equal, assert_

# 用于测试带有对象数组的多项式打印
from fractions import Fraction  # 导入分数运算模块
from decimal import Decimal  # 导入十进制运算模块


class TestStrUnicodeSuperSubscripts:

    @pytest.fixture(scope='class', autouse=True)
    def use_unicode(self):
        # 设置默认的多项式打印风格为 Unicode
        poly.set_default_printstyle('unicode')

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·x + 3.0·x²"),  # 参数化测试用例，期望输出为多项式字符串
        ([-1, 0, 3, -1], "-1.0 + 0.0·x + 3.0·x² - 1.0·x³"),
        (arange(12), ("0.0 + 1.0·x + 2.0·x² + 3.0·x³ + 4.0·x⁴ + 5.0·x⁵ + "
                      "6.0·x⁶ + 7.0·x⁷ +\n8.0·x⁸ + 9.0·x⁹ + 10.0·x¹⁰ + "
                      "11.0·x¹¹")),
    ))
    def test_polynomial_str(self, inp, tgt):
        # 创建多项式对象
        p = poly.Polynomial(inp)
        # 获取多项式对象的字符串表示
        res = str(p)
        # 断言字符串表示与目标字符串相等
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·T₁(x) + 3.0·T₂(x)"),  # 参数化测试用例，期望输出为切比雪夫多项式字符串
        ([-1, 0, 3, -1], "-1.0 + 0.0·T₁(x) + 3.0·T₂(x) - 1.0·T₃(x)"),
        (arange(12), ("0.0 + 1.0·T₁(x) + 2.0·T₂(x) + 3.0·T₃(x) + 4.0·T₄(x) + "
                      "5.0·T₅(x) +\n6.0·T₆(x) + 7.0·T₇(x) + 8.0·T₈(x) + "
                      "9.0·T₉(x) + 10.0·T₁₀(x) + 11.0·T₁₁(x)")),
    ))
    def test_chebyshev_str(self, inp, tgt):
        # 获取切比雪夫多项式的字符串表示
        res = str(poly.Chebyshev(inp))
        # 断言字符串表示与目标字符串相等
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·P₁(x) + 3.0·P₂(x)"),  # 参数化测试用例，期望输出为勒让德多项式字符串
        ([-1, 0, 3, -1], "-1.0 + 0.0·P₁(x) + 3.0·P₂(x) - 1.0·P₃(x)"),
        (arange(12), ("0.0 + 1.0·P₁(x) + 2.0·P₂(x) + 3.0·P₃(x) + 4.0·P₄(x) + "
                      "5.0·P₅(x) +\n6.0·P₆(x) + 7.0·P₇(x) + 8.0·P₈(x) + "
                      "9.0·P₉(x) + 10.0·P₁₀(x) + 11.0·P₁₁(x)")),
    ))
    def test_legendre_str(self, inp, tgt):
        # 获取勒让德多项式的字符串表示
        res = str(poly.Legendre(inp))
        # 断言字符串表示与目标字符串相等
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·H₁(x) + 3.0·H₂(x)"),  # 参数化测试用例，期望输出为埃尔米特多项式字符串
        ([-1, 0, 3, -1], "-1.0 + 0.0·H₁(x) + 3.0·H₂(x) - 1.0·H₃(x)"),
        (arange(12), ("0.0 + 1.0·H₁(x) + 2.0·H₂(x) + 3.0·H₃(x) + 4.0·H₄(x) + "
                      "5.0·H₅(x) +\n6.0·H₆(x) + 7.0·H₇(x) + 8.0·H₈(x) + "
                      "9.0·H₉(x) + 10.0·H₁₀(x) + 11.0·H₁₁(x)")),
    ))
    def test_hermite_str(self, inp, tgt):
        # 获取埃尔米特多项式的字符串表示
        res = str(poly.Hermite(inp))
        # 断言字符串表示与目标字符串相等
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·He₁(x) + 3.0·He₂(x)"),  # 参数化测试用例，期望输出为埃尔米特多项式（第二类）字符串
        ([-1, 0, 3, -1], "-1.0 + 0.0·He₁(x) + 3.0·He₂(x) - 1.0·He₃(x)"),
        (arange(12), ("0.0 + 1.0·He₁(x) + 2.0·He₂(x) + 3.0·He₃(x) + "
                      "4.0·He₄(x) + 5.0·He₅(x) +\n6.0·He₆(x) + 7.0·He₇(x) + "
                      "8.0·He₈(x) + 9.0·He₉(x) + 10.0·He₁₀(x) +\n"
                      "11.0·He₁₁(x)")),
    ))
    def test_hermiteE_str(self, inp, tgt):
        # 获取埃尔米特多项式（第二类）的字符串表示
        res = str(poly.HermiteE(inp))
        # 断言字符串表示与目标字符串相等
        assert_equal(res, tgt)
    # 使用 pytest 模块的装饰器标记参数化测试用例
    @pytest.mark.parametrize(('inp', 'tgt'), (
        # 测试用例1：输入为 [1, 2, 3]，期望输出为 "1.0 + 2.0·L₁(x) + 3.0·L₂(x)"
        ([1, 2, 3], "1.0 + 2.0·L₁(x) + 3.0·L₂(x)"),
    
        # 测试用例2：输入为 [-1, 0, 3, -1]，期望输出为 "-1.0 + 0.0·L₁(x) + 3.0·L₂(x) - 1.0·L₃(x)"
        ([-1, 0, 3, -1], "-1.0 + 0.0·L₁(x) + 3.0·L₂(x) - 1.0·L₃(x)"),
    
        # 测试用例3：输入为 arange(12)，期望输出为较长的多项式表达式字符串
        (arange(12), ("0.0 + 1.0·L₁(x) + 2.0·L₂(x) + 3.0·L₃(x) + 4.0·L₄(x) + "
                      "5.0·L₅(x) +\n6.0·L₆(x) + 7.0·L₇(x) + 8.0·L₈(x) + "
                      "9.0·L₉(x) + 10.0·L₁₀(x) + 11.0·L₁₁(x)")),
    ))
    def test_laguerre_str(self, inp, tgt):
        # 调用 Laguerre 类的 str 方法，将输入转换为字符串
        res = str(poly.Laguerre(inp))
        # 断言测试结果等于期望输出
        assert_equal(res, tgt)
    
    # 定义测试多项式字符串表示中的域
    def test_polynomial_str_domains(self):
        # 测试用例1：多项式 [0, 1] 的字符串表示应为 '0.0 + 1.0·x'
        res = str(poly.Polynomial([0, 1]))
        tgt = '0.0 + 1.0·x'
        assert_equal(res, tgt)
    
        # 测试用例2：在指定域 [1, 2] 内的多项式 [0, 1] 的字符串表示应为 '0.0 + 1.0·(-3.0 + 2.0x)'
        res = str(poly.Polynomial([0, 1], domain=[1, 2]))
        tgt = '0.0 + 1.0·(-3.0 + 2.0x)'
        assert_equal(res, tgt)
# 定义一个测试类 TestStrAscii，用于测试多项式和多种特定函数的字符串表示
class TestStrAscii:

    # 设置一个 pytest 的 fixture，作用域为 class 级别，在整个测试类中自动使用 ASCII 打印风格
    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        # 调用 poly 模块的 set_default_printstyle 方法，设置默认打印风格为 ASCII
        poly.set_default_printstyle('ascii')

    # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例，测试 Polynomial 类的字符串表示
    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 x + 3.0 x**2"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 x + 3.0 x**2 - 1.0 x**3"),
        (arange(12), ("0.0 + 1.0 x + 2.0 x**2 + 3.0 x**3 + 4.0 x**4 + "
                      "5.0 x**5 + 6.0 x**6 +\n7.0 x**7 + 8.0 x**8 + "
                      "9.0 x**9 + 10.0 x**10 + 11.0 x**11")),
    ))
    # 定义测试方法 test_polynomial_str，参数化测试 Polynomial 类的字符串表示是否与预期相符
    def test_polynomial_str(self, inp, tgt):
        # 创建 Polynomial 对象，并将其转换为字符串表示
        res = str(poly.Polynomial(inp))
        # 使用 assert_equal 断言检查结果是否与目标字符串 tgt 相等
        assert_equal(res, tgt)

    # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例，测试 Chebyshev 类的字符串表示
    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 T_1(x) + 3.0 T_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 T_1(x) + 3.0 T_2(x) - 1.0 T_3(x)"),
        (arange(12), ("0.0 + 1.0 T_1(x) + 2.0 T_2(x) + 3.0 T_3(x) + "
                      "4.0 T_4(x) + 5.0 T_5(x) +\n6.0 T_6(x) + 7.0 T_7(x) + "
                      "8.0 T_8(x) + 9.0 T_9(x) + 10.0 T_10(x) +\n"
                      "11.0 T_11(x)")),
    ))
    # 定义测试方法 test_chebyshev_str，参数化测试 Chebyshev 类的字符串表示是否与预期相符
    def test_chebyshev_str(self, inp, tgt):
        # 创建 Chebyshev 对象，并将其转换为字符串表示
        res = str(poly.Chebyshev(inp))
        # 使用 assert_equal 断言检查结果是否与目标字符串 tgt 相等
        assert_equal(res, tgt)

    # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例，测试 Legendre 类的字符串表示
    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 P_1(x) + 3.0 P_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 P_1(x) + 3.0 P_2(x) - 1.0 P_3(x)"),
        (arange(12), ("0.0 + 1.0 P_1(x) + 2.0 P_2(x) + 3.0 P_3(x) + "
                      "4.0 P_4(x) + 5.0 P_5(x) +\n6.0 P_6(x) + 7.0 P_7(x) + "
                      "8.0 P_8(x) + 9.0 P_9(x) + 10.0 P_10(x) +\n"
                      "11.0 P_11(x)")),
    ))
    # 定义测试方法 test_legendre_str，参数化测试 Legendre 类的字符串表示是否与预期相符
    def test_legendre_str(self, inp, tgt):
        # 创建 Legendre 对象，并将其转换为字符串表示
        res = str(poly.Legendre(inp))
        # 使用 assert_equal 断言检查结果是否与目标字符串 tgt 相等
        assert_equal(res, tgt)

    # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例，测试 Hermite 类的字符串表示
    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 H_1(x) + 3.0 H_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 H_1(x) + 3.0 H_2(x) - 1.0 H_3(x)"),
        (arange(12), ("0.0 + 1.0 H_1(x) + 2.0 H_2(x) + 3.0 H_3(x) + "
                      "4.0 H_4(x) + 5.0 H_5(x) +\n6.0 H_6(x) + 7.0 H_7(x) + "
                      "8.0 H_8(x) + 9.0 H_9(x) + 10.0 H_10(x) +\n"
                      "11.0 H_11(x)")),
    ))
    # 定义测试方法 test_hermite_str，参数化测试 Hermite 类的字符串表示是否与预期相符
    def test_hermite_str(self, inp, tgt):
        # 创建 Hermite 对象，并将其转换为字符串表示
        res = str(poly.Hermite(inp))
        # 使用 assert_equal 断言检查结果是否与目标字符串 tgt 相等
        assert_equal(res, tgt)

    # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例，测试 HermiteE 类的字符串表示
    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 He_1(x) + 3.0 He_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 He_1(x) + 3.0 He_2(x) - 1.0 He_3(x)"),
        (arange(12), ("0.0 + 1.0 He_1(x) + 2.0 He_2(x) + 3.0 He_3(x) + "
                      "4.0 He_4(x) +\n5.0 He_5(x) + 6.0 He_6(x) + "
                      "7.0 He_7(x) + 8.0 He_8(x) + 9.0 He_9(x) +\n"
                      "10.0 He_10(x) + 11.0 He_11(x)")),
    ))
    # 定义测试方法 test_hermiteE_str，参数化测试 HermiteE 类的字符串表示是否与预期相符
    def test_hermiteE_str(self, inp, tgt):
        # 创建 HermiteE 对象，并将其转换为字符串表示
        res = str(poly.HermiteE(inp))
        # 使用 assert_equal 断言检查结果是否与目标字符串 tgt 相等
        assert_equal(res, tgt)
    # 使用 pytest 的参数化装饰器来定义多组输入和期望输出
    @pytest.mark.parametrize(('inp', 'tgt'), (
        # 第一组测试参数：输入为 [1, 2, 3]，期望输出为 "1.0 + 2.0 L_1(x) + 3.0 L_2(x)"
        ([1, 2, 3], "1.0 + 2.0 L_1(x) + 3.0 L_2(x)"),
        # 第二组测试参数：输入为 [-1, 0, 3, -1]，期望输出为 "-1.0 + 0.0 L_1(x) + 3.0 L_2(x) - 1.0 L_3(x)"
        ([-1, 0, 3, -1], "-1.0 + 0.0 L_1(x) + 3.0 L_2(x) - 1.0 L_3(x)"),
        # 第三组测试参数：输入为 arange(12)，期望输出为长字符串，表示多项式的展示
        (arange(12), ("0.0 + 1.0 L_1(x) + 2.0 L_2(x) + 3.0 L_3(x) + "
                      "4.0 L_4(x) + 5.0 L_5(x) +\n6.0 L_6(x) + 7.0 L_7(x) + "
                      "8.0 L_8(x) + 9.0 L_9(x) + 10.0 L_10(x) +\n"
                      "11.0 L_11(x)")),
    ))
    # 测试 Laguerre 多项式对象的字符串表示是否正确
    def test_laguerre_str(self, inp, tgt):
        # 计算多项式的字符串表示
        res = str(poly.Laguerre(inp))
        # 断言计算结果与期望输出一致
        assert_equal(res, tgt)

    # 测试 Polynomial 多项式对象的字符串表示是否正确，包括指定域的情况
    def test_polynomial_str_domains(self):
        # 测试未指定域的多项式对象
        res = str(poly.Polynomial([0, 1]))
        tgt = '0.0 + 1.0 x'
        assert_equal(res, tgt)

        # 测试指定域为 [1, 2] 的多项式对象
        res = str(poly.Polynomial([0, 1], domain=[1, 2]))
        tgt = '0.0 + 1.0 (-3.0 + 2.0x)'
        assert_equal(res, tgt)
class TestLinebreaking:

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        # 设置默认打印风格为 ASCII
        poly.set_default_printstyle('ascii')

    def test_single_line_one_less(self):
        # 使用 'ascii' 风格时，len(str(p)) 是默认行宽减1（即74）
        p = poly.Polynomial([12345678, 12345678, 12345678, 12345678, 123])
        assert_equal(len(str(p)), 74)
        assert_equal(str(p), (
            '12345678.0 + 12345678.0 x + 12345678.0 x**2 + '
            '12345678.0 x**3 + 123.0 x**4'
        ))

    def test_num_chars_is_linewidth(self):
        # len(str(p)) 等于默认行宽（75）
        p = poly.Polynomial([12345678, 12345678, 12345678, 12345678, 1234])
        assert_equal(len(str(p)), 75)
        assert_equal(str(p), (
            '12345678.0 + 12345678.0 x + 12345678.0 x**2 + '
            '12345678.0 x**3 +\n1234.0 x**4'
        ))

    def test_first_linebreak_multiline_one_less_than_linewidth(self):
        # 多行字符串，其中第一行的长度加上下一个术语的长度等于行宽减1（74）
        p = poly.Polynomial(
                [12345678, 12345678, 12345678, 12345678, 1, 12345678]
            )
        assert_equal(len(str(p).split('\n')[0]), 74)
        assert_equal(str(p), (
            '12345678.0 + 12345678.0 x + 12345678.0 x**2 + '
            '12345678.0 x**3 + 1.0 x**4 +\n12345678.0 x**5'
        ))

    def test_first_linebreak_multiline_on_linewidth(self):
        # 第一行比前一个测试长一个字符
        p = poly.Polynomial(
                [12345678, 12345678, 12345678, 12345678.12, 1, 12345678]
            )
        assert_equal(str(p), (
            '12345678.0 + 12345678.0 x + 12345678.0 x**2 + '
            '12345678.12 x**3 +\n1.0 x**4 + 12345678.0 x**5'
        ))

    @pytest.mark.parametrize(('lw', 'tgt'), (
        (75, ('0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + '
              '500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + '
              '900.0 x**9')),
        (45, ('0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 +\n40000.0 x**4 + '
              '500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 +\n'
              '900.0 x**9')),
        (132, ('0.0 + 10.0 x + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + '
               '500000.0 x**5 + 600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + '
               '900.0 x**9')),
    ))
    def test_linewidth_printoption(self, lw, tgt):
        # 创建多项式对象 p
        p = poly.Polynomial(
            [0, 10, 200, 3000, 40000, 500000, 600000, 70000, 8000, 900]
        )
        # 使用给定的行宽 lw 进行打印设置
        with printoptions(linewidth=lw):
            # 断言多项式的字符串表示符合预期目标 tgt
            assert_equal(str(p), tgt)
            # 检查每行字符串的长度是否小于行宽 lw
            for line in str(p).split('\n'):
                assert_(len(line) < lw)


def test_set_default_printoptions():
    # 创建多项式对象 p 和 Chebyshev 对象 c
    p = poly.Polynomial([1, 2, 3])
    c = poly.Chebyshev([1, 2, 3])
    # 设置默认打印风格为 ASCII，并验证多项式和 Chebyshev 的字符串表示
    poly.set_default_printstyle('ascii')
    assert_equal(str(p), "1.0 + 2.0 x + 3.0 x**2")
    assert_equal(str(c), "1.0 + 2.0 T_1(x) + 3.0 T_2(x)")
    # 恢复默认打印风格为 Unicode
    poly.set_default_printstyle('unicode')
    # 断言：验证多项式对象 p 的字符串表示是否等于指定的格式化字符串
    assert_equal(str(p), "1.0 + 2.0·x + 3.0·x²")
    
    # 断言：验证多项式对象 c 的字符串表示是否等于指定的格式化字符串
    assert_equal(str(c), "1.0 + 2.0·T₁(x) + 3.0·T₂(x)")
    
    # 使用 pytest 运行时上下文，验证设置多项式对象的默认打印样式时是否引发 ValueError 异常
    with pytest.raises(ValueError):
        poly.set_default_printstyle('invalid_input')
def test_complex_coefficients():
    """Test both numpy and built-in complex."""
    # 定义复数系数列表
    coefs = [0+1j, 1+1j, -2+2j, 3+0j]
    # 使用 numpy 创建多项式对象 p1
    p1 = poly.Polynomial(coefs)
    # 使用 Python 内置数组和对象类型创建多项式对象 p2
    p2 = poly.Polynomial(array(coefs, dtype=object))
    # 设置多项式打印格式为 unicode
    poly.set_default_printstyle('unicode')
    # 断言 p1 的字符串表示符合预期
    assert_equal(str(p1), "1j + (1+1j)·x - (2-2j)·x² + (3+0j)·x³")
    # 断言 p2 的字符串表示符合预期
    assert_equal(str(p2), "1j + (1+1j)·x + (-2+2j)·x² + (3+0j)·x³")
    # 设置多项式打印格式为 ascii
    poly.set_default_printstyle('ascii')
    # 再次断言 p1 的字符串表示符合预期
    assert_equal(str(p1), "1j + (1+1j) x - (2-2j) x**2 + (3+0j) x**3")
    # 再次断言 p2 的字符串表示符合预期
    assert_equal(str(p2), "1j + (1+1j) x + (-2+2j) x**2 + (3+0j) x**3")


@pytest.mark.parametrize(('coefs', 'tgt'), (
    # 参数化测试：测试带有分数对象系数的情况
    (array([Fraction(1, 2), Fraction(3, 4)], dtype=object), (
        "1/2 + 3/4·x"
    )),
    # 参数化测试：测试带有分数和整数对象系数的情况
    (array([1, 2, Fraction(5, 7)], dtype=object), (
        "1 + 2·x + 5/7·x²"
    )),
    # 参数化测试：测试带有 Decimal 对象系数的情况
    (array([Decimal('1.00'), Decimal('2.2'), 3], dtype=object), (
        "1.00 + 2.2·x + 3·x²"
    )),
))
def test_numeric_object_coefficients(coefs, tgt):
    # 创建多项式对象 p，并设置打印格式为 unicode
    p = poly.Polynomial(coefs)
    poly.set_default_printstyle('unicode')
    # 断言 p 的字符串表示符合预期
    assert_equal(str(p), tgt)


@pytest.mark.parametrize(('coefs', 'tgt'), (
    # 参数化测试：测试带有字符串对象系数的情况
    (array([1, 2, 'f'], dtype=object), '1 + 2·x + f·x²'),
    # 参数化测试：测试带有列表对象系数的情况
    (array([1, 2, [3, 4]], dtype=object), '1 + 2·x + [3, 4]·x²'),
))
def test_nonnumeric_object_coefficients(coefs, tgt):
    """
    Test coef fallback for object arrays of non-numeric coefficients.
    """
    # 创建多项式对象 p，并设置打印格式为 unicode
    p = poly.Polynomial(coefs)
    poly.set_default_printstyle('unicode')
    # 断言 p 的字符串表示符合预期
    assert_equal(str(p), tgt)


class TestFormat:
    def test_format_unicode(self):
        # 设置多项式打印格式为 ascii
        poly.set_default_printstyle('ascii')
        # 创建多项式对象 p
        p = poly.Polynomial([1, 2, 0, -1])
        # 断言 p 的 unicode 格式字符串表示符合预期
        assert_equal(format(p, 'unicode'), "1.0 + 2.0·x + 0.0·x² - 1.0·x³")

    def test_format_ascii(self):
        # 设置多项式打印格式为 unicode
        poly.set_default_printstyle('unicode')
        # 创建多项式对象 p
        p = poly.Polynomial([1, 2, 0, -1])
        # 断言 p 的 ascii 格式字符串表示符合预期
        assert_equal(
            format(p, 'ascii'), "1.0 + 2.0 x + 0.0 x**2 - 1.0 x**3"
        )

    def test_empty_formatstr(self):
        # 设置多项式打印格式为 ascii
        poly.set_default_printstyle('ascii')
        # 创建多项式对象 p
        p = poly.Polynomial([1, 2, 3])
        # 断言 p 的默认字符串表示符合预期
        assert_equal(format(p), "1.0 + 2.0 x + 3.0 x**2")
        # 使用 f-string 断言 p 的字符串表示符合预期
        assert_equal(f"{p}", "1.0 + 2.0 x + 3.0 x**2")

    def test_bad_formatstr(self):
        # 创建多项式对象 p
        p = poly.Polynomial([1, 2, 0, -1])
        # 断言格式化 p 时引发 ValueError 异常
        with pytest.raises(ValueError):
            format(p, '.2f')


@pytest.mark.parametrize(('poly', 'tgt'), (
    # 参数化测试：测试多项式类的 symbol 字段设置和字符串表示
    (poly.Polynomial, '1.0 + 2.0·z + 3.0·z²'),
    (poly.Chebyshev, '1.0 + 2.0·T₁(z) + 3.0·T₂(z)'),
    (poly.Hermite, '1.0 + 2.0·H₁(z) + 3.0·H₂(z)'),
    (poly.HermiteE, '1.0 + 2.0·He₁(z) + 3.0·He₂(z)'),
    (poly.Laguerre, '1.0 + 2.0·L₁(z) + 3.0·L₂(z)'),
    (poly.Legendre, '1.0 + 2.0·P₁(z) + 3.0·P₂(z)'),
))
def test_symbol(poly, tgt):
    # 使用指定的 symbol 创建多项式对象 p
    p = poly([1, 2, 3], symbol='z')
    # 断言 p 的 unicode 格式字符串表示符合预期
    assert_equal(f"{p:unicode}", tgt)


class TestRepr:
    # 测试 Polynomial 类的 repr 方法
    def test_polynomial_repr(self):
        # 创建 Polynomial 对象并调用其 repr 方法，生成字符串表示
        res = repr(poly.Polynomial([0, 1]))
        # 目标字符串表示，包含数组 [0., 1.]，领域 domain=[-1.,  1.]，窗口 window=[-1.,  1.]，符号 symbol='x'
        tgt = (
            "Polynomial([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        # 断言结果与目标一致
        assert_equal(res, tgt)
    
    # 测试 Chebyshev 类的 repr 方法
    def test_chebyshev_repr(self):
        # 创建 Chebyshev 对象并调用其 repr 方法，生成字符串表示
        res = repr(poly.Chebyshev([0, 1]))
        # 目标字符串表示，包含数组 [0., 1.]，领域 domain=[-1.,  1.]，窗口 window=[-1.,  1.]，符号 symbol='x'
        tgt = (
            "Chebyshev([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        # 断言结果与目标一致
        assert_equal(res, tgt)
    
    # 测试 Legendre 类的 repr 方法
    def test_legendre_repr(self):
        # 创建 Legendre 对象并调用其 repr 方法，生成字符串表示
        res = repr(poly.Legendre([0, 1]))
        # 目标字符串表示，包含数组 [0., 1.]，领域 domain=[-1.,  1.]，窗口 window=[-1.,  1.]，符号 symbol='x'
        tgt = (
            "Legendre([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        # 断言结果与目标一致
        assert_equal(res, tgt)
    
    # 测试 Hermite 类的 repr 方法
    def test_hermite_repr(self):
        # 创建 Hermite 对象并调用其 repr 方法，生成字符串表示
        res = repr(poly.Hermite([0, 1]))
        # 目标字符串表示，包含数组 [0., 1.]，领域 domain=[-1.,  1.]，窗口 window=[-1.,  1.]，符号 symbol='x'
        tgt = (
            "Hermite([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        # 断言结果与目标一致
        assert_equal(res, tgt)
    
    # 测试 HermiteE 类的 repr 方法
    def test_hermiteE_repr(self):
        # 创建 HermiteE 对象并调用其 repr 方法，生成字符串表示
        res = repr(poly.HermiteE([0, 1]))
        # 目标字符串表示，包含数组 [0., 1.]，领域 domain=[-1.,  1.]，窗口 window=[-1.,  1.]，符号 symbol='x'
        tgt = (
            "HermiteE([0., 1.], domain=[-1.,  1.], window=[-1.,  1.], "
            "symbol='x')"
        )
        # 断言结果与目标一致
        assert_equal(res, tgt)
    
    # 测试 Laguerre 类的 repr 方法
    def test_laguerre_repr(self):
        # 创建 Laguerre 对象并调用其 repr 方法，生成字符串表示
        res = repr(poly.Laguerre([0, 1]))
        # 目标字符串表示，包含数组 [0., 1.]，领域 domain=[0., 1.]，窗口 window=[0., 1.]，符号 symbol='x'
        tgt = (
            "Laguerre([0., 1.], domain=[0., 1.], window=[0., 1.], "
            "symbol='x')"
        )
        # 断言结果与目标一致
        assert_equal(res, tgt)
class TestLatexRepr:
    """Test the latex repr used by Jupyter"""

    @staticmethod
    def as_latex(obj):
        # 定义一个静态方法，用于获取对象的 LaTeX 表示
        # 目前忽略标量的格式化，因为它会使测试结果过于冗长。
        # 理想情况下，标量的格式化应当修复，以确保下面的测试继续通过。
        
        # 临时设置对象的标量 LaTeX 表示为 lambda 函数，将对象转换为字符串。
        obj._repr_latex_scalar = lambda x, parens=False: str(x)
        try:
            # 调用对象的 _repr_latex_ 方法，返回其 LaTeX 表示
            return obj._repr_latex_()
        finally:
            # 删除临时设置的标量 LaTeX 表示
            del obj._repr_latex_scalar

    def test_simple_polynomial(self):
        # 测试简单多项式的 LaTeX 表示
        
        # 默认输入，创建一个多项式对象 p
        p = poly.Polynomial([1, 2, 3])
        # 断言对象 p 的 LaTeX 表示符合预期
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,x + 3.0\,x^{2}$')

        # 翻译后的输入
        p = poly.Polynomial([1, 2, 3], domain=[-2, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(1.0 + x\right) + 3.0\,\left(1.0 + x\right)^{2}$')

        # 缩放后的输入
        p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(2.0x\right) + 3.0\,\left(2.0x\right)^{2}$')

        # 仿射输入
        p = poly.Polynomial([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(1.0 + 2.0x\right) + 3.0\,\left(1.0 + 2.0x\right)^{2}$')

    def test_basis_func(self):
        # 测试基函数的 LaTeX 表示
        p = poly.Chebyshev([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{T}_{0}(x) + 2.0\,{T}_{1}(x) + 3.0\,{T}_{2}(x)$')
        
        # 仿射输入 - 检查是否添加了多余的括号
        p = poly.Chebyshev([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{T}_{0}(1.0 + 2.0x) + 2.0\,{T}_{1}(1.0 + 2.0x) + 3.0\,{T}_{2}(1.0 + 2.0x)$')

    def test_multichar_basis_func(self):
        # 测试多字符基函数的 LaTeX 表示
        p = poly.HermiteE([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{He}_{0}(x) + 2.0\,{He}_{1}(x) + 3.0\,{He}_{2}(x)$')
    def test_symbol_basic(self):
        # 使用默认输入创建多项式对象，指定符号为 'z'
        p = poly.Polynomial([1, 2, 3], symbol='z')
        # 断言多项式对象转换为 LaTeX 格式的字符串是否正确
        assert_equal(self.as_latex(p),
            r'$z \mapsto 1.0 + 2.0\,z + 3.0\,z^{2}$')

        # 使用翻译后的输入创建多项式对象，指定符号为 'z'，定义域为 [-2, 0]
        p = poly.Polynomial([1, 2, 3], domain=[-2, 0], symbol='z')
        # 断言多项式对象转换为 LaTeX 格式的字符串是否正确
        assert_equal(
            self.as_latex(p),
            (
                r'$z \mapsto 1.0 + 2.0\,\left(1.0 + z\right) + 3.0\,'
                r'\left(1.0 + z\right)^{2}$'
            ),
        )

        # 使用缩放后的输入创建多项式对象，指定符号为 'z'，定义域为 [-0.5, 0.5]
        p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5], symbol='z')
        # 断言多项式对象转换为 LaTeX 格式的字符串是否正确
        assert_equal(
            self.as_latex(p),
            (
                r'$z \mapsto 1.0 + 2.0\,\left(2.0z\right) + 3.0\,'
                r'\left(2.0z\right)^{2}$'
            ),
        )

        # 使用仿射输入创建多项式对象，指定符号为 'z'，定义域为 [-1, 0]
        p = poly.Polynomial([1, 2, 3], domain=[-1, 0], symbol='z')
        # 断言多项式对象转换为 LaTeX 格式的字符串是否正确
        assert_equal(
            self.as_latex(p),
            (
                r'$z \mapsto 1.0 + 2.0\,\left(1.0 + 2.0z\right) + 3.0\,'
                r'\left(1.0 + 2.0z\right)^{2}$'
            ),
        )

    def test_numeric_object_coefficients(self):
        # 创建具有分数系数的多项式对象
        coefs = array([Fraction(1, 2), Fraction(1)])
        p = poly.Polynomial(coefs)
        # 断言多项式对象转换为 LaTeX 格式的字符串是否正确
        assert_equal(self.as_latex(p), '$x \\mapsto 1/2 + 1\\,x$')
SWITCH_TO_EXP = (
    '1.0 + (1.0e-01) x + (1.0e-02) x**2',  # 第一个多项式表达式
    '1.2 + (1.2e-01) x + (1.2e-02) x**2',  # 第二个多项式表达式
    '1.23 + 0.12 x + (1.23e-02) x**2 + (1.23e-03) x**3',  # 第三个多项式表达式
    '1.235 + 0.123 x + (1.235e-02) x**2 + (1.235e-03) x**3',  # 第四个多项式表达式
    '1.2346 + 0.1235 x + 0.0123 x**2 + (1.2346e-03) x**3 + (1.2346e-04) x**4',  # 第五个多项式表达式
    '1.23457 + 0.12346 x + 0.01235 x**2 + (1.23457e-03) x**3 + (1.23457e-04) x**4',  # 第六个多项式表达式
    '1.234568 + 0.123457 x + 0.012346 x**2 + 0.001235 x**3 + (1.234568e-04) x**4 + (1.234568e-05) x**5',  # 第七个多项式表达式
    '1.2345679 + 0.1234568 x + 0.0123457 x**2 + 0.0012346 x**3 + (1.2345679e-04) x**4 + (1.2345679e-05) x**5'  # 第八个多项式表达式
)

class TestPrintOptions:
    """
    测试通过printoptions正确配置输出选项。
    当值过小或过大时，自动启用指数表示法。
    """

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')  # 设置默认的多项式打印风格为ASCII码风格

    def test_str(self):
        p = poly.Polynomial([1/2, 1/7, 1/7*10**8, 1/7*10**9])
        assert_equal(str(p), '0.5 + 0.14285714 x + 14285714.28571429 x**2 '
                             '+ (1.42857143e+08) x**3')  # 验证多项式的字符串表示格式

        with printoptions(precision=3):
            assert_equal(str(p), '0.5 + 0.143 x + 14285714.286 x**2 '
                                 '+ (1.429e+08) x**3')  # 在指定精度下验证多项式的字符串表示格式

    def test_latex(self):
        p = poly.Polynomial([1/2, 1/7, 1/7*10**8, 1/7*10**9])
        assert_equal(p._repr_latex_(),
            r'$x \mapsto \text{0.5} + \text{0.14285714}\,x + '
            r'\text{14285714.28571429}\,x^{2} + '
            r'\text{(1.42857143e+08)}\,x^{3}$')  # 验证多项式的LaTeX表示格式
        
        with printoptions(precision=3):
            assert_equal(p._repr_latex_(),
                r'$x \mapsto \text{0.5} + \text{0.143}\,x + '
                r'\text{14285714.286}\,x^{2} + \text{(1.429e+08)}\,x^{3}$')  # 在指定精度下验证多项式的LaTeX表示格式

    def test_fixed(self):
        p = poly.Polynomial([1/2])
        assert_equal(str(p), '0.5')  # 验证固定小数位数模式下的多项式字符串表示
        
        with printoptions(floatmode='fixed'):
            assert_equal(str(p), '0.50000000')  # 在固定小数位数模式下验证多项式的字符串表示
        
        with printoptions(floatmode='fixed', precision=4):
            assert_equal(str(p), '0.5000')  # 在固定小数位数和指定精度下验证多项式的字符串表示

    def test_switch_to_exp(self):
        for i, s in enumerate(SWITCH_TO_EXP):
            with printoptions(precision=i):
                p = poly.Polynomial([1.23456789*10**-i 
                                     for i in range(i//2+3)])
                assert str(p).replace('\n', ' ') == s  # 在指定精度下验证多项式转换为指数表示的字符串格式
    
    def test_non_finite(self):
        p = poly.Polynomial([nan, inf])
        assert str(p) == 'nan + inf x'  # 验证包含非有限数的多项式的字符串表示
        assert p._repr_latex_() == r'$x \mapsto \text{nan} + \text{inf}\,x$'  # 验证包含非有限数的多项式的LaTeX表示
        
        with printoptions(nanstr='NAN', infstr='INF'):
            assert str(p) == 'NAN + INF x'  # 在指定NaN和Inf字符串的情况下验证多项式的字符串表示
            assert p._repr_latex_() == \
                r'$x \mapsto \text{NAN} + \text{INF}\,x$'  # 在指定NaN和Inf字符串的情况下验证多项式的LaTeX表示
```