# `D:\src\scipysrc\sympy\sympy\concrete\tests\test_guess.py`

```
# 导入从 sympy.concrete.guess 模块中导入的函数和类
# 分别是 find_simple_recurrence_vector, find_simple_recurrence, rationalize,
# guess_generating_function_rational, guess_generating_function, guess
from sympy.concrete.guess import (
            find_simple_recurrence_vector,
            find_simple_recurrence,
            rationalize,
            guess_generating_function_rational,
            guess_generating_function,
            guess
        )

# 导入从 sympy.concrete.products 模块中导入的 Product 类
from sympy.concrete.products import Product
# 导入从 sympy.core.function 模块中导入的 Function 类
from sympy.core.function import Function
# 导入从 sympy.core.numbers 模块中导入的 Rational 类
from sympy.core.numbers import Rational
# 导入从 sympy.core.singleton 模块中导入的 S 单例
from sympy.core.singleton import S
# 导入从 sympy.core.symbol 模块中导入的 Symbol 和 symbols 函数
from sympy.core.symbol import (Symbol, symbols)
# 导入从 sympy.core.sympify 模块中导入的 sympify 函数
from sympy.core.sympify import sympify
# 导入从 sympy.functions.combinatorial.factorials 模块中导入的 RisingFactorial 和 factorial 函数
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
# 导入从 sympy.functions.combinatorial.numbers 模块中导入的 fibonacci 函数
from sympy.functions.combinatorial.numbers import fibonacci
# 导入从 sympy.functions.elementary.exponential 模块中导入的 exp 函数
from sympy.functions.elementary.exponential import exp


# 定义测试函数 test_find_simple_recurrence_vector，用于测试 find_simple_recurrence_vector 函数
def test_find_simple_recurrence_vector():
    # 断言调用 find_simple_recurrence_vector 函数，并检查其返回值是否为预期的 [1, -1, -1]
    assert find_simple_recurrence_vector(
            [fibonacci(k) for k in range(12)]) == [1, -1, -1]


# 定义测试函数 test_find_simple_recurrence，用于测试 find_simple_recurrence 函数
def test_find_simple_recurrence():
    # 创建符号函数 a 和符号 n
    a = Function('a')
    n = Symbol('n')
    # 断言调用 find_simple_recurrence 函数，检查其返回的简单递推关系表达式是否正确
    assert find_simple_recurrence([fibonacci(k) for k in range(12)]) == (
        -a(n) - a(n + 1) + a(n + 2))

    # 创建符号函数 f 和符号 i
    f = Function('a')
    i = Symbol('n')
    # 创建列表 a 并填充初始值
    a = [1, 1, 1]
    for k in range(15): a.append(5*a[-1]-3*a[-2]+8*a[-3])
    # 断言调用 find_simple_recurrence 函数，检查其返回的简单递推关系表达式是否正确
    assert find_simple_recurrence(a, A=f, N=i) == (
        -8*f(i) + 3*f(i + 1) - 5*f(i + 2) + f(i + 3))
    # 断言调用 find_simple_recurrence 函数，检查对于给定列表 [0, 2, 15, 74, 12, 3, 0, 1, 2, 85, 4, 5, 63] 返回值是否为 0
    assert find_simple_recurrence([0, 2, 15, 74, 12, 3, 0,
                                    1, 2, 85, 4, 5, 63]) == 0


# 定义测试函数 test_rationalize，用于测试 rationalize 函数
def test_rationalize():
    # 从 mpmath 模块导入 cos, pi, mpf 函数
    from mpmath import cos, pi, mpf
    # 断言调用 rationalize 函数，检查对于 cos(pi/3) 的有理化结果是否为 S.Half
    assert rationalize(cos(pi/3)) == S.Half
    # 断言调用 rationalize 函数，检查对于 mpf("0.333333333333333") 的有理化结果是否为 Rational(1, 3)
    assert rationalize(mpf("0.333333333333333")) == Rational(1, 3)
    # 断言调用 rationalize 函数，检查对于 mpf("-0.333333333333333") 的有理化结果是否为 Rational(-1, 3)
    assert rationalize(mpf("-0.333333333333333")) == Rational(-1, 3)
    # 断言调用 rationalize 函数，检查对于 pi 的有理化结果是否为 Rational(355, 113)（并指定最大系数为 250）
    assert rationalize(pi, maxcoeff = 250) == Rational(355, 113)


# 定义测试函数 test_guess_generating_function_rational，用于测试 guess_generating_function_rational 函数
def test_guess_generating_function_rational():
    # 创建符号 x
    x = Symbol('x')
    # 断言调用 guess_generating_function_rational 函数，检查对于 fibonacci 序列的生成函数是否正确
    assert guess_generating_function_rational([fibonacci(k)
        for k in range(5, 15)]) == ((3*x + 5)/(-x**2 - x + 1))


# 定义测试函数 test_guess_generating_function，用于测试 guess_generating_function 函数
def test_guess_generating_function():
    # 创建符号 x
    x = Symbol('x')
    # 断言调用 guess_generating_function 函数，检查对于 fibonacci 序列的生成函数的 ogf（普通生成函数）是否正确
    assert guess_generating_function([fibonacci(k)
        for k in range(5, 15)])['ogf'] == ((3*x + 5)/(-x**2 - x + 1))
    # 断言调用 guess_generating_function 函数，检查对于指定序列的生成函数的 ogf 是否正确
    assert guess_generating_function(
        [1, 2, 5, 14, 41, 124, 383, 1200, 3799, 12122, 38919])['ogf'] == (
        (1/(x**4 + 2*x**2 - 4*x + 1))**S.Half)
    # 断言调用 guess_generating_function 函数，检查对于指定序列的生成函数的 ogf 是否正确
    assert guess_generating_function(sympify(
       "[3/2, 11/2, 0, -121/2, -363/2, 121, 4719/2, 11495/2, -8712, -178717/2]")
       )['ogf'] == (x + Rational(3, 2))/(11*x**2 - 3*x + 1)
    # 断言调用 guess_generating_function 函数，检查对于 factorial 序列的生成函数的 egf（指数生成函数）是否正确
    assert guess_generating_function([factorial(k) for k in range(12)],
       types=['egf'])['egf'] == 1/(-x + 1)
    # 断言调用 guess_generating_function 函数，检查对于给定序列的生成函数的 egf 和 lgdegf 是否正确
    assert guess_generating_function([k+1 for k in range(12)],
       types=['egf']) == {'egf': (x + 1)*exp(x), 'lgdegf': (x + 2)/(x + 1)}


# 定义测试函数 test_guess，用于测试 guess 函数
def test_guess():
    # 创建符号 i0, i1
    i0, i1 = symbols('i0 i1')
    # 断言调用 guess 函数，检查对于给定序列的猜测结果是否为 Product 类型的表达式
    assert guess([1, 2, 6, 24, 120], evaluate=False) == [Product(i1 + 1, (i1, 1, i0 - 1))]
    # 断言调用 guess 函数，检查对于给定序列的猜测结果是否为 RisingFactorial 类型的表达式
    assert guess([1, 2, 6, 24, 120]) == [RisingFactorial(2, i0 - 1)]
    # 断言，验证函数 guess 的返回值是否符合预期
    assert guess([1, 2, 7, 42, 429, 7436, 218348, 10850216], niter=4) == [
        2**(i0 - 1)*(Rational(27, 16))**(i0**2/2 - 3*i0/2 +
        1)*Product(RisingFactorial(Rational(5, 3), i1 - 1)*RisingFactorial(Rational(7, 3), i1
        - 1)/(RisingFactorial(Rational(3, 2), i1 - 1)*RisingFactorial(Rational(5, 2), i1 -
        1)), (i1, 1, i0 - 1))]
    # 断言，验证特定输入列表的函数 guess 返回空列表
    assert guess([1, 0, 2]) == []
    # 创建符号变量 x, y
    x, y = symbols('x y')
    # 断言，验证函数 guess 对特定输入列表及变量返回符合预期的结果列表
    assert guess([1, 2, 6, 24, 120], variables=[x, y]) == [RisingFactorial(2, x - 1)]
```