# `D:\src\scipysrc\sympy\sympy\assumptions\tests\test_rel_queries.py`

```
from sympy.assumptions.lra_satask import lra_satask  # 导入 lra_satask 函数
from sympy.logic.algorithms.lra_theory import UnhandledInput  # 导入 UnhandledInput 异常类
from sympy.assumptions.ask import Q, ask  # 导入 Q 和 ask

from sympy.core import symbols, Symbol  # 导入 symbols 和 Symbol 类
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入 MatrixSymbol 类
from sympy.core.numbers import I  # 导入虚数单位 I

from sympy.testing.pytest import raises, XFAIL  # 导入 raises 和 XFAIL

x, y, z = symbols("x y z", real=True)  # 创建实数符号 x, y, z

def test_lra_satask():
    im = Symbol('im', imaginary=True)  # 创建一个虚数符号 im

    # 测试不等式预处理是否工作正常
    assert lra_satask(Q.eq(x, 1), ~Q.ne(x, 0)) is False  # 断言不等式 x == 1 和 x != 0 的情况
    assert lra_satask(Q.eq(x, 0), ~Q.ne(x, 0)) is True  # 断言不等式 x == 0 和 x != 0 的情况
    assert lra_satask(~Q.ne(x, 0), Q.eq(x, 0)) is True  # 断言不等式 x != 0 和 x == 0 的情况
    assert lra_satask(~Q.eq(x, 0), Q.eq(x, 0)) is False  # 断言不等式 x != 0 和 x == 0 的情况
    assert lra_satask(Q.ne(x, 0), Q.eq(x, 0)) is False  # 断言不等式 x != 0 和 x == 0 的情况

    # 基本测试
    assert lra_satask(Q.ne(x, x)) is False  # 断言 x != x 的情况
    assert lra_satask(Q.eq(x, x)) is True  # 断言 x == x 的情况
    assert lra_satask(Q.gt(x, 0), Q.gt(x, 1)) is True  # 断言 x > 0 和 x > 1 的情况

    # 检查 True/False 的处理情况
    assert lra_satask(Q.gt(x, 0), True) is None  # 断言 x > 0 和 True 的情况
    assert raises(ValueError, lambda: lra_satask(Q.gt(x, 0), False))  # 断言 x > 0 和 False 的情况抛出 ValueError

    # 检查虚数的处理情况
    # (im * I).is_real 返回 True，因此这是一个边缘情况
    raises(UnhandledInput, lambda: lra_satask(Q.gt(im * I, 0), Q.gt(im * I, 0)))  # 断言 im * I > 0 和 im * I > 0 的情况抛出 UnhandledInput 异常

    # 检查矩阵输入
    X = MatrixSymbol("X", 2, 2)  # 创建一个2x2矩阵符号 X
    raises(UnhandledInput, lambda: lra_satask(Q.lt(X, 2) & Q.gt(X, 3)))  # 断言矩阵 X < 2 和 X > 3 的情况抛出 UnhandledInput 异常


def test_old_assumptions():
    # 测试未处理的旧前提
    w = symbols("w")
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))  # 断言 w < 2 和 w > 3 的情况抛出 UnhandledInput 异常
    w = symbols("w", rational=False, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))  # 断言 w < 2 和 w > 3 的情况抛出 UnhandledInput 异常
    w = symbols("w", odd=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))  # 断言 w < 2 和 w > 3 的情况抛出 UnhandledInput 异常
    w = symbols("w", even=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))  # 断言 w < 2 和 w > 3 的情况抛出 UnhandledInput 异常
    w = symbols("w", prime=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))  # 断言 w < 2 和 w > 3 的情况抛出 UnhandledInput 异常
    w = symbols("w", composite=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))  # 断言 w < 2 和 w > 3 的情况抛出 UnhandledInput 异常
    w = symbols("w", integer=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))  # 断言 w < 2 和 w > 3 的情况抛出 UnhandledInput 异常
    w = symbols("w", integer=False, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))  # 断言 w < 2 和 w > 3 的情况抛出 UnhandledInput 异常

    # 测试已处理的情况
    w = symbols("w", positive=True, real=True)
    assert lra_satask(Q.le(w, 0)) is False  # 断言 w <= 0 的情况为 False
    assert lra_satask(Q.gt(w, 0)) is True  # 断言 w > 0 的情况为 True
    w = symbols("w", negative=True, real=True)
    assert lra_satask(Q.lt(w, 0)) is True  # 断言 w < 0 的情况为 True
    assert lra_satask(Q.ge(w, 0)) is False  # 断言 w >= 0 的情况为 False
    w = symbols("w", zero=True, real=True)
    assert lra_satask(Q.eq(w, 0)) is True  # 断言 w == 0 的情况为 True
    assert lra_satask(Q.ne(w, 0)) is False  # 断言 w != 0 的情况为 False
    w = symbols("w", nonzero=True, real=True)
    assert lra_satask(Q.ne(w, 0)) is True  # 断言 w != 0 的情况为 True
    assert lra_satask(Q.eq(w, 1)) is None  # 断言 w == 1 的情况为 None
    w = symbols("w", nonpositive=True, real=True)
    # 断言：确保调用 lra_satask 函数，传入 Q.le(w, 0) 条件，返回 True
    assert lra_satask(Q.le(w, 0)) is True
    # 断言：确保调用 lra_satask 函数，传入 Q.gt(w, 0) 条件，返回 False
    assert lra_satask(Q.gt(w, 0)) is False
    # 定义符号 w，要求它是非负实数
    w = symbols("w", nonnegative=True, real=True)
    # 断言：确保调用 lra_satask 函数，传入 Q.ge(w, 0) 条件，返回 True
    assert lra_satask(Q.ge(w, 0)) is True
    # 断言：确保调用 lra_satask 函数，传入 Q.lt(w, 0) 条件，返回 False
    assert lra_satask(Q.lt(w, 0)) is False
def test_rel_queries():
    # 检验是否 Q.lt(x, 2) & Q.gt(x, 3) 为假
    assert ask(Q.lt(x, 2) & Q.gt(x, 3)) is False
    # 检验是否 (x > y) & (y > z) 下 Q.positive(x - z) 为真
    assert ask(Q.positive(x - z), (x > y) & (y > z)) is True
    # 检验是否 (x < 0) & (y < 0) 下 x + y > 2 为假
    assert ask(x + y > 2, (x < 0) & (y < 0)) is False
    # 检验是否 (x > y) & (y > z) 下 x > z 为真
    assert ask(x > z, (x > y) & (y > z)) is True


def test_unhandled_queries():
    X = MatrixSymbol("X", 2, 2)
    # 检验是否 Q.lt(X, 2) & Q.gt(X, 3) 为 None
    assert ask(Q.lt(X, 2) & Q.gt(X, 3)) is None


def test_all_pred():
    # 测试可用的谓词
    assert lra_satask(Q.extended_positive(x), (x > 2)) is True
    assert lra_satask(Q.positive_infinite(x)) is False
    assert lra_satask(Q.negative_infinite(x)) is False

    # 测试不允许的谓词
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.prime(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.composite(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.odd(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.even(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.integer(x)))


def test_number_line_properties():
    # From:
    # https://en.wikipedia.org/wiki/Inequality_(mathematics)#Properties_on_the_number_line

    a, b, c = symbols("a b c", real=True)

    # 传递性
    # 如果 a <= b 且 b <= c，则 a <= c
    assert ask(a <= c, (a <= b) & (b <= c)) is True
    # 如果 a <= b 且 b < c，则 a < c
    assert ask(a < c, (a <= b) & (b < c)) is True
    # 如果 a < b 且 b <= c，则 a < c
    assert ask(a < c, (a < b) & (b <= c)) is True

    # 加法和减法
    # 如果 a <= b，则 a + c <= b + c 和 a - c <= b - c
    assert ask(a + c <= b + c, a <= b) is True
    assert ask(a - c <= b - c, a <= b) is True


@XFAIL
def test_failing_number_line_properties():
    # From:
    # https://en.wikipedia.org/wiki/Inequality_(mathematics)#Properties_on_the_number_line

    a, b, c = symbols("a b c", real=True)

    # 乘法和除法
    # 如果 a <= b 且 c > 0，则 ac <= bc 和 a/c <= b/c（对于非零的 c 成立）
    assert ask(a*c <= b*c, (a <= b) & (c > 0) & ~Q.zero(c)) is True
    assert ask(a/c <= b/c, (a <= b) & (c > 0) & ~Q.zero(c)) is True
    # 如果 a <= b 且 c < 0，则 ac >= bc 和 a/c >= b/c（对于非零的 c 成立）
    assert ask(a*c >= b*c, (a <= b) & (c < 0) & ~Q.zero(c)) is True
    assert ask(a/c >= b/c, (a <= b) & (c < 0) & ~Q.zero(c)) is True

    # 加法逆元
    # 如果 a <= b，则 -a >= -b
    assert ask(-a >= -b, a <= b) is True

    # 乘法逆元
    # 对于都是负或者都是正的 a, b：
    # 如果 a <= b，则 1/a >= 1/b
    assert ask(1/a >= 1/b, (a <= b) & Q.positive(x) & Q.positive(b)) is True
    assert ask(1/a >= 1/b, (a <= b) & Q.negative(x) & Q.negative(b)) is True


def test_equality():
    # 测试对称性和自反性
    assert ask(Q.eq(x, x)) is True
    assert ask(Q.eq(y, x), Q.eq(x, y)) is True
    assert ask(Q.eq(y, x), ~Q.eq(z, z) | Q.eq(x, y)) is True

    # 测试传递性
    assert ask(Q.eq(x,z), Q.eq(x,y) & Q.eq(y,z)) is True


@XFAIL
def test_equality_failing():
    # Note that implementing the substitution property of equality
    # most likely requires a redesign of the new assumptions.
    # See issue #25485 for why this is the case and general ideas
    # about how things could be redesigned.
    
    # 测试等价性质
    assert ask(Q.prime(x), Q.eq(x, y) & Q.prime(y)) is True
    # 断言检查 x 是否是质数，当且仅当 x 等于 y 并且 y 也是质数时应为真

    assert ask(Q.real(x), Q.eq(x, y) & Q.real(y)) is True
    # 断言检查 x 是否是实数，当且仅当 x 等于 y 并且 y 也是实数时应为真

    assert ask(Q.imaginary(x), Q.eq(x, y) & Q.imaginary(y)) is True
    # 断言检查 x 是否是虚数，当且仅当 x 等于 y 并且 y 也是虚数时应为真


这些注释解释了每个断言语句的作用，指出了它们在测试程序中的期望行为，特别是关于等价性质的测试。
```