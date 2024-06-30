# `D:\src\scipysrc\sympy\sympy\printing\tests\test_python.py`

```
# 导入从sympy库中导入特定的模块和函数
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.series.limits import limit

# 导入sympy.printing.python模块中的python函数
from sympy.printing.python import python

# 导入sympy.testing.pytest模块中的raises和XFAIL函数
from sympy.testing.pytest import raises, XFAIL

# 定义符号变量x和y，并赋予它们Symbol对象
x, y = symbols('x,y')

# 定义符号变量th和ph，并分别赋予它们Symbol对象theta和phi
th = Symbol('theta')
ph = Symbol('phi')

# 定义测试函数test_python_basic，用于测试python函数的输出结果
def test_python_basic():
    # 测试python函数对简单数值/符号的输出
    assert python(-Rational(1)/2) == "e = Rational(-1, 2)"
    assert python(-Rational(13)/22) == "e = Rational(-13, 22)"
    assert python(oo) == "e = oo"

    # 测试python函数对指数的输出
    assert python(x**2) == "x = Symbol(\'x\')\ne = x**2"
    assert python(1/x) == "x = Symbol('x')\ne = 1/x"
    assert python(y*x**-2) == "y = Symbol('y')\nx = Symbol('x')\ne = y/x**2"
    assert python(
        x**Rational(-5, 2)) == "x = Symbol('x')\ne = x**Rational(-5, 2)"

    # 测试python函数对项的求和输出
    assert python(x**2 + x + 1) in [
        "x = Symbol('x')\ne = 1 + x + x**2",
        "x = Symbol('x')\ne = x + x**2 + 1",
        "x = Symbol('x')\ne = x**2 + x + 1", ]
    assert python(1 - x) in [
        "x = Symbol('x')\ne = 1 - x",
        "x = Symbol('x')\ne = -x + 1"]
    assert python(1 - 2*x) in [
        "x = Symbol('x')\ne = 1 - 2*x",
        "x = Symbol('x')\ne = -2*x + 1"]
    assert python(1 - Rational(3, 2)*y/x) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = 1 - 3/2*y/x",
        "y = Symbol('y')\nx = Symbol('x')\ne = -3/2*y/x + 1",
        "y = Symbol('y')\nx = Symbol('x')\ne = 1 - 3*y/(2*x)"]

    # 测试python函数对乘法的输出
    assert python(x/y) == "x = Symbol('x')\ny = Symbol('y')\ne = x/y"
    assert python(-x/y) == "x = Symbol('x')\ny = Symbol('y')\ne = -x/y"
    assert python((x + 2)/y) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = 1/y*(2 + x)",
        "y = Symbol('y')\nx = Symbol('x')\ne = 1/y*(x + 2)",
        "x = Symbol('x')\ny = Symbol('y')\ne = 1/y*(2 + x)",
        "x = Symbol('x')\ny = Symbol('y')\ne = (2 + x)/y",
        "x = Symbol('x')\ny = Symbol('y')\ne = (x + 2)/y"]
    assert python((1 + x)*y) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = y*(1 + x)",
        "y = Symbol('y')\nx = Symbol('x')\ne = y*(x + 1)", ]

    # 检查负号的正确放置
    assert python(-5*x/(x + 10)) == "x = Symbol('x')\ne = -5*x/(x + 10)"
    assert python(1 - Rational(3, 2)*(x + 1)) in [
        "x = Symbol('x')\ne = Rational(-3, 2)*x + Rational(-1, 2)",
        "x = Symbol('x')\ne = -3*x/2 + Rational(-1, 2)",
        "x = Symbol('x')\ne = -3*x/2 + Rational(-1, 2)"
    ]


def test_python_keyword_symbol_name_escaping():
    # 检查对关键字的转义处理
    # 断言：将表达式转换为字符串表示，并验证其输出结果是否符合预期
    
    assert python(
        5*Symbol("lambda")) == "lambda_ = Symbol('lambda')\ne = 5*lambda_"
    
    # 断言：将复杂表达式转换为字符串表示，并验证其输出结果是否符合预期
    assert (python(5*Symbol("lambda") + 7*Symbol("lambda_")) ==
            "lambda__ = Symbol('lambda')\nlambda_ = Symbol('lambda_')\ne = 7*lambda_ + 5*lambda__")
    
    # 断言：将包含函数调用的表达式转换为字符串表示，并验证其输出结果是否符合预期
    assert (python(5*Symbol("for") + Function("for_")(8)) ==
            "for__ = Symbol('for')\nfor_ = Function('for_')\ne = 5*for__ + for_(8)")
def test_python_keyword_function_name_escaping():
    # 断言检查将数值 5 乘以 Function 对象 'for' 的结果是否等于字符串 "for_ = Function('for')\ne = 5*for_(8)"
    assert python(
        5*Function("for")(8)) == "for_ = Function('for')\ne = 5*for_(8)"


def test_python_relational():
    # 断言检查创建并打印 x 等于 y 的等式是否等于字符串 "x = Symbol('x')\ny = Symbol('y')\ne = Eq(x, y)"
    assert python(Eq(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = Eq(x, y)"
    # 断言检查创建并打印 x 大于等于 y 的关系是否等于字符串 "x = Symbol('x')\ny = Symbol('y')\ne = x >= y"
    assert python(Ge(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x >= y"
    # 断言检查创建并打印 x 小于等于 y 的关系是否等于字符串 "x = Symbol('x')\ny = Symbol('y')\ne = x <= y"
    assert python(Le(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x <= y"
    # 断言检查创建并打印 x 大于 y 的关系是否等于字符串 "x = Symbol('x')\ny = Symbol('y')\ne = x > y"
    assert python(Gt(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x > y"
    # 断言检查创建并打印 x 小于 y 的关系是否等于字符串 "x = Symbol('x')\ny = Symbol('y')\ne = x < y"
    assert python(Lt(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x < y"
    # 断言检查创建并打印 x 除以 (y + 1) 不等于 y 平方的关系是否在给定的两种可能结果中的一种
    assert python(Ne(x/(y + 1), y**2)) in [
        "x = Symbol('x')\ny = Symbol('y')\ne = Ne(x/(1 + y), y**2)",
        "x = Symbol('x')\ny = Symbol('y')\ne = Ne(x/(y + 1), y**2)"]


def test_python_functions():
    # 简单函数测试
    assert python(2*x + exp(x)) in "x = Symbol('x')\ne = 2*x + exp(x)"
    assert python(sqrt(2)) == 'e = sqrt(2)'
    assert python(2**Rational(1, 3)) == 'e = 2**Rational(1, 3)'
    assert python(sqrt(2 + pi)) == 'e = sqrt(2 + pi)'
    assert python((2 + pi)**Rational(1, 3)) == 'e = (2 + pi)**Rational(1, 3)'
    assert python(2**Rational(1, 4)) == 'e = 2**Rational(1, 4)'
    assert python(Abs(x)) == "x = Symbol('x')\ne = Abs(x)"
    assert python(
        Abs(x/(x**2 + 1))) in ["x = Symbol('x')\ne = Abs(x/(1 + x**2))",
            "x = Symbol('x')\ne = Abs(x/(x**2 + 1))"]

    # 单变量/多变量函数测试
    f = Function('f')
    assert python(f(x)) == "x = Symbol('x')\nf = Function('f')\ne = f(x)"
    assert python(f(x, y)) == "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x, y)"
    assert python(f(x/(y + 1), y)) in [
        "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x/(1 + y), y)",
        "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x/(y + 1), y)"]

    # 嵌套平方根测试
    assert python(sqrt((sqrt(x + 1)) + 1)) in [
        "x = Symbol('x')\ne = sqrt(1 + sqrt(1 + x))",
        "x = Symbol('x')\ne = sqrt(sqrt(x + 1) + 1)"]

    # 嵌套幂测试
    assert python((((x + 1)**Rational(1, 3)) + 1)**Rational(1, 3)) in [
        "x = Symbol('x')\ne = (1 + (1 + x)**Rational(1, 3))**Rational(1, 3)",
        "x = Symbol('x')\ne = ((x + 1)**Rational(1, 3) + 1)**Rational(1, 3)"]

    # 函数的幂次方测试
    assert python(sin(x)**2) == "x = Symbol('x')\ne = sin(x)**2"


@XFAIL
def test_python_functions_conjugates():
    a, b = map(Symbol, 'ab')
    # 断言检查对复数 a + b*I 取共轭是否等于字符串 '_     _\na - I*b'
    assert python( conjugate(a + b*I) ) == '_     _\na - I*b'
    # 断言检查对复数 exp(a + b*I) 取共轭是否等于字符串 ' _     _\n a - I*b\ne       '
    assert python( conjugate(exp(a + b*I)) ) == ' _     _\n a - I*b\ne       '


def test_python_derivatives():
    # 简单导数测试
    f_1 = Derivative(log(x), x, evaluate=False)
    assert python(f_1) == "x = Symbol('x')\ne = Derivative(log(x), x)"

    f_2 = Derivative(log(x), x, evaluate=False) + x
    assert python(f_2) == "x = Symbol('x')\ne = x + Derivative(log(x), x)"

    # 多个符号的导数测试
    f_3 = Derivative(log(x) + x**2, x, y, evaluate=False)
    # 使用 assert 语句来验证 python 函数对于给定输入是否产生预期的输出
    assert python(f_3) == \
        "x = Symbol('x')\ny = Symbol('y')\ne = Derivative(x**2 + log(x), x, y)"
    
    # 创建一个 Derivative 对象 f_4，表示 2*x*y 对 y 和 x 的二阶偏导数，但禁用自动求值
    f_4 = Derivative(2*x*y, y, x, evaluate=False) + x**2
    
    # 使用 assert 语句验证 python 函数对于 f_4 的输出是否在给定的两个可能性中之一
    assert python(f_4) in [
        "x = Symbol('x')\ny = Symbol('y')\ne = x**2 + Derivative(2*x*y, y, x)",
        "x = Symbol('x')\ny = Symbol('y')\ne = Derivative(2*x*y, y, x) + x**2"]
# 定义测试函数 test_python_integrals，用于测试积分表达式转换为 Python 代码的功能
def test_python_integrals():
    # Simple case: Integral(log(x), x)
    f_1 = Integral(log(x), x)
    # 断言转换后的 Python 代码符合预期结果
    assert python(f_1) == "x = Symbol('x')\ne = Integral(log(x), x)"

    # Integral(x**2, x)
    f_2 = Integral(x**2, x)
    # 断言转换后的 Python 代码符合预期结果
    assert python(f_2) == "x = Symbol('x')\ne = Integral(x**2, x)"

    # Double nesting of pow: Integral(x**(2**x), x)
    f_3 = Integral(x**(2**x), x)
    # 断言转换后的 Python 代码符合预期结果
    assert python(f_3) == "x = Symbol('x')\ne = Integral(x**(2**x), x)"

    # Definite integral: Integral(x**2, (x, 1, 2))
    f_4 = Integral(x**2, (x, 1, 2))
    # 断言转换后的 Python 代码符合预期结果
    assert python(f_4) == "x = Symbol('x')\ne = Integral(x**2, (x, 1, 2))"

    # Definite integral with Rational bounds: Integral(x**2, (x, Rational(1, 2), 10))
    f_5 = Integral(x**2, (x, Rational(1, 2), 10))
    # 断言转换后的 Python 代码符合预期结果
    assert python(f_5) == "x = Symbol('x')\ne = Integral(x**2, (x, Rational(1, 2), 10))"

    # Nested integrals: Integral(x**2*y**2, x, y)
    f_6 = Integral(x**2*y**2, x, y)
    # 断言转换后的 Python 代码符合预期结果
    assert python(f_6) == "x = Symbol('x')\ny = Symbol('y')\ne = Integral(x**2*y**2, x, y)"


# 定义测试函数 test_python_matrix，用于测试矩阵表达式转换为 Python 代码的功能
def test_python_matrix():
    # 测试矩阵转换：Matrix([[x**2+1, 1], [y, x+y]])
    p = python(Matrix([[x**2+1, 1], [y, x+y]]))
    s = "x = Symbol('x')\ny = Symbol('y')\ne = MutableDenseMatrix([[x**2 + 1, 1], [y, x + y]])"
    # 断言转换后的 Python 代码符合预期结果
    assert p == s


# 定义测试函数 test_python_limits，用于测试极限表达式转换为 Python 代码的功能
def test_python_limits():
    # 测试极限转换：limit(x, x, oo)
    assert python(limit(x, x, oo)) == 'e = oo'
    # 测试极限转换：limit(x**2, x, 0)
    assert python(limit(x**2, x, 0)) == 'e = 0'


# 定义测试函数 test_issue_20762，用于测试符号表达式转换为 Python 代码的功能
def test_issue_20762():
    # 测试符号表达式转换：Symbol('a_{b}') * Symbol('b')
    a_b = Symbol('a_{b}')
    b = Symbol('b')
    expr = a_b*b
    # 断言转换后的 Python 代码符合预期结果
    assert python(expr) == "a_b = Symbol('a_{b}')\nb = Symbol('b')\ne = a_b*b"


# 定义测试函数 test_settings，用于测试设置处理的异常情况
def test_settings():
    # 测试设置处理：python(x, method="garbage")
    raises(TypeError, lambda: python(x, method="garbage"))
```