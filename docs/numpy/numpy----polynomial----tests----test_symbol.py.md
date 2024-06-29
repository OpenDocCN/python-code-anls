# `.\numpy\numpy\polynomial\tests\test_symbol.py`

```py
"""
Tests related to the ``symbol`` attribute of the ABCPolyBase class.
"""

import pytest  # 导入 pytest 测试框架
import numpy.polynomial as poly  # 导入 numpy.polynomial 模块
from numpy._core import array  # 导入 numpy 数组模块
from numpy.testing import assert_equal, assert_raises, assert_  # 导入断言函数


class TestInit:
    """
    Test polynomial creation with symbol kwarg.
    """
    c = [1, 2, 3]  # 定义多项式系数列表

    def test_default_symbol(self):
        p = poly.Polynomial(self.c)  # 创建多项式对象
        assert_equal(p.symbol, 'x')  # 断言多项式的默认符号为 'x'

    @pytest.mark.parametrize(('bad_input', 'exception'), (
        ('', ValueError),  # 空字符串引发 ValueError 异常
        ('3', ValueError),  # 字符串 '3' 引发 ValueError 异常
        (None, TypeError),   # None 引发 TypeError 异常
        (1, TypeError),      # 数字 1 引发 TypeError 异常
    ))
    def test_symbol_bad_input(self, bad_input, exception):
        with pytest.raises(exception):
            p = poly.Polynomial(self.c, symbol=bad_input)  # 使用不良输入创建多项式对象

    @pytest.mark.parametrize('symbol', (
        'x',
        'x_1',
        'A',
        'xyz',
        'β',
    ))
    def test_valid_symbols(self, symbol):
        """
        Values for symbol that should pass input validation.
        """
        p = poly.Polynomial(self.c, symbol=symbol)  # 使用有效符号创建多项式对象
        assert_equal(p.symbol, symbol)  # 断言多项式的符号与预期符号相同

    def test_property(self):
        """
        'symbol' attribute is read only.
        """
        p = poly.Polynomial(self.c, symbol='x')  # 创建多项式对象
        with pytest.raises(AttributeError):
            p.symbol = 'z'  # 尝试更改多项式对象的符号，预期引发 AttributeError 异常

    def test_change_symbol(self):
        p = poly.Polynomial(self.c, symbol='y')  # 创建多项式对象
        # Create new polynomial from p with different symbol
        pt = poly.Polynomial(p.coef, symbol='t')  # 使用不同符号创建新的多项式对象
        assert_equal(pt.symbol, 't')  # 断言新多项式对象的符号为 't'


class TestUnaryOperators:
    p = poly.Polynomial([1, 2, 3], symbol='z')  # 创建带有符号 'z' 的多项式对象

    def test_neg(self):
        n = -self.p  # 对多项式取负
        assert_equal(n.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_scalarmul(self):
        out = self.p * 10  # 多项式与标量相乘
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_rscalarmul(self):
        out = 10 * self.p  # 标量与多项式相乘
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_pow(self):
        out = self.p ** 3  # 多项式的幂运算
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'


@pytest.mark.parametrize(
    'rhs',
    (
        poly.Polynomial([4, 5, 6], symbol='z'),  # 右操作数为符号为 'z' 的多项式对象
        array([4, 5, 6]),  # 右操作数为 numpy 数组
    ),
)
class TestBinaryOperatorsSameSymbol:
    """
    Ensure symbol is preserved for numeric operations on polynomials with
    the same symbol
    """
    p = poly.Polynomial([1, 2, 3], symbol='z')  # 创建带有符号 'z' 的多项式对象

    def test_add(self, rhs):
        out = self.p + rhs  # 多项式加法操作
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_sub(self, rhs):
        out = self.p - rhs  # 多项式减法操作
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_polymul(self, rhs):
        out = self.p * rhs  # 多项式乘法操作
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_divmod(self, rhs):
        for out in divmod(self.p, rhs):  # 多项式除法操作
            assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_radd(self, rhs):
        out = rhs + self.p  # 反向多项式加法操作
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_rsub(self, rhs):
        out = rhs - self.p  # 反向多项式减法操作
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'

    def test_rmul(self, rhs):
        out = rhs * self.p  # 反向多项式乘法操作
        assert_equal(out.symbol, 'z')  # 断言结果多项式的符号保持为 'z'
    # 定义一个测试方法 test_rdivmod，该方法用于测试除法和取模操作
    def test_rdivmod(self, rhs):
        # 对于每一个 divmod(rhs, self.p) 的结果，执行断言操作
        for out in divmod(rhs, self.p):
            # 断言输出对象的符号属性为 'z'
            assert_equal(out.symbol, 'z')
class TestBinaryOperatorsDifferentSymbol:
    # 创建一个多项式对象 p，使用 'x' 作为符号，系数为 [1, 2, 3]
    p = poly.Polynomial([1, 2, 3], symbol='x')
    # 创建另一个多项式对象 other，使用 'y' 作为符号，系数为 [4, 5, 6]
    other = poly.Polynomial([4, 5, 6], symbol='y')
    # 定义包含 p 不同二元操作方法的元组 ops
    ops = (p.__add__, p.__sub__, p.__mul__, p.__floordiv__, p.__mod__)

    @pytest.mark.parametrize('f', ops)
    # 测试各个二元操作方法是否会引发 ValueError 异常
    def test_binops_fails(self, f):
        assert_raises(ValueError, f, self.other)


class TestEquality:
    # 创建一个多项式对象 p，使用 'x' 作为符号，系数为 [1, 2, 3]
    p = poly.Polynomial([1, 2, 3], symbol='x')

    def test_eq(self):
        # 创建另一个与 p 相同的多项式对象 other，使用 'x' 作为符号，系数为 [1, 2, 3]
        other = poly.Polynomial([1, 2, 3], symbol='x')
        # 断言 p 与 other 相等
        assert_(self.p == other)

    def test_neq(self):
        # 创建另一个与 p 不同的多项式对象 other，使用 'y' 作为符号，系数为 [1, 2, 3]
        other = poly.Polynomial([1, 2, 3], symbol='y')
        # 断言 p 与 other 不相等
        assert_(not self.p == other)


class TestExtraMethods:
    """
    Test other methods for manipulating/creating polynomial objects.
    """
    # 创建一个多项式对象 p，使用 'z' 作为符号，系数为 [1, 2, 3, 0]
    p = poly.Polynomial([1, 2, 3, 0], symbol='z')

    def test_copy(self):
        # 复制当前多项式对象 p 得到 other
        other = self.p.copy()
        # 断言 other 的符号为 'z'
        assert_equal(other.symbol, 'z')

    def test_trim(self):
        # 对当前多项式对象 p 进行修剪操作得到 other
        other = self.p.trim()
        # 断言 other 的符号为 'z'
        assert_equal(other.symbol, 'z')

    def test_truncate(self):
        # 对当前多项式对象 p 进行截断操作得到 other
        other = self.p.truncate(2)
        # 断言 other 的符号为 'z'
        assert_equal(other.symbol, 'z')

    @pytest.mark.parametrize('kwarg', (
        {'domain': [-10, 10]},
        {'window': [-10, 10]},
        {'kind': poly.Chebyshev},
    ))
    # 测试多项式对象 p 的转换方法，接受不同的关键字参数 kwarg
    def test_convert(self, kwarg):
        other = self.p.convert(**kwarg)
        # 断言 other 的符号为 'z'
        assert_equal(other.symbol, 'z')

    def test_integ(self):
        # 对当前多项式对象 p 进行积分操作得到 other
        other = self.p.integ()
        # 断言 other 的符号为 'z'
        assert_equal(other.symbol, 'z')

    def test_deriv(self):
        # 对当前多项式对象 p 进行求导操作得到 other
        other = self.p.deriv()
        # 断言 other 的符号为 'z'
        assert_equal(other.symbol, 'z')


def test_composition():
    # 创建一个多项式对象 p，使用 "t" 作为符号，系数为 [3, 2, 1]
    p = poly.Polynomial([3, 2, 1], symbol="t")
    # 创建一个多项式对象 q，使用 "λ_1" 作为符号，系数为 [5, 1, 0, -1]
    q = poly.Polynomial([5, 1, 0, -1], symbol="λ_1")
    # 对多项式对象 p 和 q 进行组合操作得到 r
    r = p(q)
    # 断言 r 的符号为 "λ_1"
    assert r.symbol == "λ_1"


#
# Class methods that result in new polynomial class instances
#


def test_fit():
    # 创建一个多项式对象 p，使用 'z' 作为符号，通过拟合得到
    x, y = (range(10),)*2
    p = poly.Polynomial.fit(x, y, deg=1, symbol='z')
    # 断言 p 的符号为 'z'
    assert_equal(p.symbol, 'z')


def test_froomroots():
    # 从给定的根列表创建一个多项式对象 p，使用 'z' 作为符号
    roots = [-2, 2]
    p = poly.Polynomial.fromroots(roots, symbol='z')
    # 断言 p 的符号为 'z'
    assert_equal(p.symbol, 'z')


def test_identity():
    # 创建一个恒等多项式对象 p，使用 'z' 作为符号，指定定义域和窗口
    p = poly.Polynomial.identity(domain=[-1, 1], window=[5, 20], symbol='z')
    # 断言 p 的符号为 'z'
    assert_equal(p.symbol, 'z')


def test_basis():
    # 创建一个多项式对象 p，使用 'z' 作为符号，表示基函数
    p = poly.Polynomial.basis(3, symbol='z')
    # 断言 p 的符号为 'z'
    assert_equal(p.symbol, 'z')
```