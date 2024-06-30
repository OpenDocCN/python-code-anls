# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_trigsimp.py`

```
# 导入模块中的 product 函数，用于生成迭代器的笛卡尔积
from itertools import product
# 从 sympy.core.function 模块中导入多个函数：Subs, count_ops, diff, expand
from sympy.core.function import (Subs, count_ops, diff, expand)
# 从 sympy.core.numbers 模块中导入常数：E, I, Rational, pi
from sympy.core.numbers import (E, I, Rational, pi)
# 从 sympy.core.singleton 模块中导入单例常数：S
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块中导入符号对象：Symbol, symbols
from sympy.core.symbol import (Symbol, symbols)
# 从 sympy.functions.elementary.exponential 模块中导入指数和对数函数：exp, log
from sympy.functions.elementary.exponential import (exp, log)
# 从 sympy.functions.elementary.hyperbolic 模块中导入双曲函数：cosh, coth, sinh, tanh
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
# 从 sympy.functions.elementary.miscellaneous 模块中导入其他函数：sqrt
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.piecewise 模块中导入分段函数：Piecewise
from sympy.functions.elementary.piecewise import Piecewise
# 从 sympy.functions.elementary.trigonometric 模块中导入三角函数：cos, cot, sin, tan
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
# 从 sympy.functions.elementary.trigonometric 模块中导入反三角函数：acos, asin, atan2
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
# 从 sympy.functions.elementary.trigonometric 模块中导入余割、正割、余切、正切反函数：asec, acsc, acot, atan
from sympy.functions.elementary.trigonometric import (asec, acsc, acot, atan)
# 从 sympy.integrals.integrals 模块中导入积分函数：integrate
from sympy.integrals.integrals import integrate
# 从 sympy.matrices.dense 模块中导入密集矩阵对象：Matrix
from sympy.matrices.dense import Matrix
# 从 sympy.simplify.simplify 模块中导入简化函数：simplify
from sympy.simplify.simplify import simplify
# 从 sympy.simplify.trigsimp 模块中导入三角函数化简函数：exptrigsimp, trigsimp
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)

# 从 sympy.testing.pytest 模块中导入 XFAIL，用于标记预期测试失败的测试用例
from sympy.testing.pytest import XFAIL

# 从 sympy.abc 模块中导入符号变量 x, y
from sympy.abc import x, y

# 定义一个测试函数 test_trigsimp1，用于测试 trigsimp 函数的各种情况
def test_trigsimp1():
    # 定义符号变量 x, y
    x, y = symbols('x,y')

    # 断言 trigsimp 函数对于不同的三角函数表达式能够进行正确的简化
    assert trigsimp(1 - sin(x)**2) == cos(x)**2
    assert trigsimp(1 - cos(x)**2) == sin(x)**2
    assert trigsimp(sin(x)**2 + cos(x)**2) == 1
    assert trigsimp(1 + tan(x)**2) == 1/cos(x)**2
    assert trigsimp(1/cos(x)**2 - 1) == tan(x)**2
    assert trigsimp(1/cos(x)**2 - tan(x)**2) == 1
    assert trigsimp(1 + cot(x)**2) == 1/sin(x)**2
    assert trigsimp(1/sin(x)**2 - 1) == 1/tan(x)**2
    assert trigsimp(1/sin(x)**2 - cot(x)**2) == 1

    # 断言 trigsimp 函数能够正确处理混合三角函数和常数的表达式
    assert trigsimp(5*cos(x)**2 + 5*sin(x)**2) == 5
    assert trigsimp(5*cos(x/2)**2 + 2*sin(x/2)**2) == 3*cos(x)/2 + Rational(7, 2)

    # 断言 trigsimp 函数能够正确简化其他三角函数的表达式
    assert trigsimp(sin(x)/cos(x)) == tan(x)
    assert trigsimp(2*tan(x)*cos(x)) == 2*sin(x)
    assert trigsimp(cot(x)**3*sin(x)**3) == cos(x)**3
    assert trigsimp(y*tan(x)**2/sin(x)**2) == y/cos(x)**2
    assert trigsimp(cot(x)/cos(x)) == 1/sin(x)

    # 断言 trigsimp 函数能够正确处理三角函数和其他三角函数组合的表达式
    assert trigsimp(sin(x + y) + sin(x - y)) == 2*sin(x)*cos(y)
    assert trigsimp(sin(x + y) - sin(x - y)) == 2*sin(y)*cos(x)
    assert trigsimp(cos(x + y) + cos(x - y)) == 2*cos(x)*cos(y)
    assert trigsimp(cos(x + y) - cos(x - y)) == -2*sin(x)*sin(y)
    assert trigsimp(tan(x + y) - tan(x)/(1 - tan(x)*tan(y))) == \
        sin(y)/(-sin(y)*tan(x) + cos(y))  # -tan(y)/(tan(x)*tan(y) - 1)

    # 断言 trigsimp 函数能够正确处理双曲函数的表达式
    assert trigsimp(sinh(x + y) + sinh(x - y)) == 2*sinh(x)*cosh(y)
    assert trigsimp(sinh(x + y) - sinh(x - y)) == 2*sinh(y)*cosh(x)
    assert trigsimp(cosh(x + y) + cosh(x - y)) == 2*cosh(x)*cosh(y)
    assert trigsimp(cosh(x + y) - cosh(x - y)) == 2*sinh(x)*sinh(y)
    assert trigsimp(tanh(x + y) - tanh(x)/(1 + tanh(x)*tanh(y))) == \
        sinh(y)/(sinh(y)*tanh(x) + cosh(y))

    # 断言 trigsimp 函数能够正确处理常数的三角函数表达式
    assert trigsimp(cos(0.12345)**2 + sin(0.12345)**2) == 1.0
    # 测试对于复杂表达式的简化是否正确
    e = 2*sin(x)**2 + 2*cos(x)**2
    assert trigsimp(log(e)) == log(2)

# 定义另一个测试函数 test_trigsimp1a，用于测试 trigsimp 函数对复杂表达式的简化
def test_trigsimp1a():
    # 断言 trigsimp 函数能够正确简化复杂的三角函数表达式
    assert trigsimp(sin(2)**2*cos(3)*exp(2)/cos(2)**2) == tan(2)**2*cos(3)*exp(2)
    # 确保表达式使用三角函数、指数函数和三角函数化简后等于给定的简化结果
    assert trigsimp(tan(2)**2*cos(3)*exp(2)*cos(2)**2) == sin(2)**2*cos(3)*exp(2)
    
    # 确保表达式使用三角函数、指数函数和三角函数化简后等于给定的简化结果
    assert trigsimp(cot(2)*cos(3)*exp(2)*sin(2)) == cos(3)*exp(2)*cos(2)
    
    # 确保表达式使用三角函数、指数函数和三角函数化简后等于给定的简化结果
    assert trigsimp(tan(2)*cos(3)*exp(2)/sin(2)) == cos(3)*exp(2)/cos(2)
    
    # 确保表达式使用三角函数、指数函数和三角函数化简后等于给定的简化结果
    assert trigsimp(cot(2)*cos(3)*exp(2)/cos(2)) == cos(3)*exp(2)/sin(2)
    
    # 确保表达式使用三角函数、指数函数和三角函数化简后等于给定的简化结果
    assert trigsimp(cot(2)*cos(3)*exp(2)*tan(2)) == cos(3)*exp(2)
    
    # 确保表达式使用双曲函数、指数函数和双曲函数化简后等于给定的简化结果
    assert trigsimp(sinh(2)*cos(3)*exp(2)/cosh(2)) == tanh(2)*cos(3)*exp(2)
    
    # 确保表达式使用双曲函数、指数函数和双曲函数化简后等于给定的简化结果
    assert trigsimp(tanh(2)*cos(3)*exp(2)*cosh(2)) == sinh(2)*cos(3)*exp(2)
    
    # 确保表达式使用双曲函数、指数函数和双曲函数化简后等于给定的简化结果
    assert trigsimp(coth(2)*cos(3)*exp(2)*sinh(2)) == cosh(2)*cos(3)*exp(2)
    
    # 确保表达式使用双曲函数、指数函数和双曲函数化简后等于给定的简化结果
    assert trigsimp(tanh(2)*cos(3)*exp(2)/sinh(2)) == cos(3)*exp(2)/cosh(2)
    
    # 确保表达式使用双曲函数、指数函数和双曲函数化简后等于给定的简化结果
    assert trigsimp(coth(2)*cos(3)*exp(2)/cosh(2)) == cos(3)*exp(2)/sinh(2)
    
    # 确保表达式使用双曲函数、指数函数和双曲函数化简后等于给定的简化结果
    assert trigsimp(coth(2)*cos(3)*exp(2)*tanh(2)) == cos(3)*exp(2)
def test_trigsimp2():
    # 定义符号变量 x, y
    x, y = symbols('x,y')
    # 断言简化三角函数表达式后等于 1
    assert trigsimp(cos(x)**2*sin(y)**2 + cos(x)**2*cos(y)**2 + sin(x)**2,
            recursive=True) == 1
    # 断言简化三角函数表达式后等于 1
    assert trigsimp(sin(x)**2*sin(y)**2 + sin(x)**2*cos(y)**2 + cos(x)**2,
            recursive=True) == 1
    # 断言简化替换表达式后的结果
    assert trigsimp(
        Subs(x, x, sin(y)**2 + cos(y)**2)) == Subs(x, x, 1)


def test_issue_4373():
    # 定义符号变量 x
    x = Symbol("x")
    # 断言简化三角函数表达式后接近 2.0
    assert abs(trigsimp(2.0*sin(x)**2 + 2.0*cos(x)**2) - 2.0) < 1e-10


def test_trigsimp3():
    # 定义符号变量 x, y
    x, y = symbols('x,y')
    # 断言简化 sin(x)/cos(x) 表达式后等于 tan(x)
    assert trigsimp(sin(x)/cos(x)) == tan(x)
    # 断言简化 sin(x)**2/cos(x)**2 表达式后等于 tan(x)**2
    assert trigsimp(sin(x)**2/cos(x)**2) == tan(x)**2
    # 断言简化 sin(x)**3/cos(x)**3 表达式后等于 tan(x)**3
    assert trigsimp(sin(x)**3/cos(x)**3) == tan(x)**3
    # 断言简化 sin(x)**10/cos(x)**10 表达式后等于 tan(x)**10
    assert trigsimp(sin(x)**10/cos(x)**10) == tan(x)**10

    # 断言简化 cos(x)/sin(x) 表达式后等于 1/tan(x)
    assert trigsimp(cos(x)/sin(x)) == 1/tan(x)
    # 断言简化 cos(x)**2/sin(x)**2 表达式后等于 1/tan(x)**2
    assert trigsimp(cos(x)**2/sin(x)**2) == 1/tan(x)**2
    # 断言简化 cos(x)**10/sin(x)**10 表达式后等于 1/tan(x)**10
    assert trigsimp(cos(x)**10/sin(x)**10) == 1/tan(x)**10

    # 断言简化 tan(x) 表达式后等于 trigsimp(sin(x)/cos(x))
    assert trigsimp(tan(x)) == trigsimp(sin(x)/cos(x))


def test_issue_4661():
    # 定义符号变量 a, x, y
    a, x, y = symbols('a x y')
    # 断言简化复杂的三角函数表达式 eq 后等于 -4
    eq = -4*sin(x)**4 + 4*cos(x)**4 - 8*cos(x)**2
    assert trigsimp(eq) == -4
    # 定义两个表达式 n 和 d
    n = sin(x)**6 + 4*sin(x)**4*cos(x)**2 + 5*sin(x)**2*cos(x)**4 + 2*cos(x)**6
    d = -sin(x)**2 - 2*cos(x)**2
    # 断言简化 n/d 表达式后等于 -1
    assert simplify(n/d) == -1
    # 断言简化复杂的三角函数表达式后等于 -1
    assert trigsimp(-2*cos(x)**2 + cos(x)**4 - sin(x)**4) == -1
    # 定义复杂的三角函数表达式 eq
    eq = (- sin(x)**3/4)*cos(x) + (cos(x)**3/4)*sin(x) - sin(2*x)*cos(2*x)/8
    # 断言简化 eq 后等于 0
    assert trigsimp(eq) == 0


def test_issue_4494():
    # 定义符号变量 a, b
    a, b = symbols('a b')
    # 断言简化复杂的三角函数表达式 eq 后等于 1
    eq = sin(a)**2*sin(b)**2 + cos(a)**2*cos(b)**2*tan(a)**2 + cos(a)**2
    assert trigsimp(eq) == 1


def test_issue_5948():
    # 定义符号变量 a, x, y
    a, x, y = symbols('a x y')
    # 断言简化复杂的积分和微分后的三角函数表达式后等于 cos(x)/sin(x)**7
    assert trigsimp(diff(integrate(cos(x)/sin(x)**7, x), x)) == \
           cos(x)/sin(x)**7


def test_issue_4775():
    # 定义符号变量 a, x, y
    a, x, y = symbols('a x y')
    # 断言简化复杂的三角函数表达式后等于 sin(x + y)
    assert trigsimp(sin(x)*cos(y)+cos(x)*sin(y)) == sin(x + y)
    # 断言简化复杂的三角函数表达式后等于 sin(x + y) + 3
    assert trigsimp(sin(x)*cos(y)+cos(x)*sin(y)+3) == sin(x + y) + 3


def test_issue_4280():
    # 定义符号变量 a, x, y
    a, x, y = symbols('a x y')
    # 断言简化复杂的三角函数表达式后等于 1
    assert trigsimp(cos(x)**2 + cos(y)**2*sin(x)**2 + sin(y)**2*sin(x)**2) == 1
    # 断言简化复杂的三角函数表达式后等于 a**2
    assert trigsimp(a**2*sin(x)**2 + a**2*cos(y)**2*cos(x)**2 + a**2*cos(x)**2*sin(y)**2) == a**2
    # 断言简化复杂的三角函数表达式后等于 a**2*sin(x)**2
    assert trigsimp(a**2*cos(y)**2*sin(x)**2 + a**2*sin(y)**2*sin(x)**2) == a**2*sin(x)**2


def test_issue_3210():
    # 定义多个复杂的三角函数表达式
    eqs = (sin(2)*cos(3) + sin(3)*cos(2),
        -sin(2)*sin(3) + cos(2)*cos(3),
        sin(2)*cos(3) - sin(3)*cos(2),
        sin(2)*sin(3) + cos(2)*cos(3),
        sin(2)*sin(3) + cos(2)*cos(3) + cos(2),
        sinh(2)*cosh(3) + sinh(3)*cosh(2),
        sinh(2)*sinh(3) + cosh(2)*cosh(3),
        )
    # 断言简化每个表达式后的结果
    assert [trigsimp(e) for e in eqs] == [
        sin(5),
        cos(5),
        -sin(1),
        cos(1),
        cos(1) + cos(2),
        sinh(5),
        cosh(5),
        ]


def test_trigsimp_issues():
    # 定义符号变量 a, x, y

    # issue 4625 - factor_terms works, too
    # 断言简化复杂的三角函数表达式后等于 sin(x)
    assert trigsimp(sin(x)**3 + cos(x)**2*sin(x)) == sin(x)

    # issue 5948
    # 断言简化复杂的积分和微分后的三角函数表达式后等于 cos(x)/sin(x)**3
    assert trigsimp(diff(integrate(cos(x)/sin(x)**3, x), x)) == \
        cos(x)/sin(x)**3
    # 断言简化三角函数的复合运算结果是否等于原始表达式
    assert trigsimp(diff(integrate(sin(x)/cos(x)**3, x), x)) == \
        sin(x)/cos(x)**3

    # 检查整数指数情况下的三角函数简化
    e = sin(x)**y/cos(x)**y
    assert trigsimp(e) == e
    assert trigsimp(e.subs(y, 2)) == tan(x)**2
    assert trigsimp(e.subs(x, 1)) == tan(1)**y

    # 检查多种模式的三角函数表达式简化
    assert (cos(x)**2/sin(x)**2*cos(y)**2/sin(y)**2).trigsimp() == \
        1/tan(x)**2/tan(y)**2
    assert trigsimp(cos(x)/sin(x)*cos(x+y)/sin(x+y)) == \
        1/(tan(x)*tan(x + y))

    # 检查包含特定因子的三角函数表达式的简化
    eq = cos(2)*(cos(3) + 1)**2/(cos(3) - 1)**2
    assert trigsimp(eq) == eq.factor()  # factor makes denom (-1 + cos(3))**2
    assert trigsimp(cos(2)*(cos(3) + 1)**2*(cos(3) - 1)**2) == \
        cos(2)*sin(3)**4

    # 检查特定情况下 trigsimp 的处理，避免 hang 的问题
    assert cot(x).equals(tan(x)) is False

    # 检查 sin 函数在不同表达式下的简化结果
    # 期望结果应不是 sin(1)，而是 nan 或未变化的表达式
    z = cos(x)**2 + sin(x)**2 - 1
    z1 = tan(x)**2 - 1/cot(x)**2
    n = (1 + z1/z)
    assert trigsimp(sin(n)) != sin(1)
    eq = x*(n - 1) - x*n
    assert trigsimp(eq) is S.NaN
    assert trigsimp(eq, recursive=True) is S.NaN
    assert trigsimp(1).is_Integer

    # 检查三角函数表达式的简化
    assert trigsimp(-sin(x)**4 - 2*sin(x)**2*cos(x)**2 - cos(x)**4) == -1
# 定义用于测试 trigsimp 函数的测试用例
def test_trigsimp_issue_2515():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言 trigsimp(x*cos(x)*tan(x)) 的简化结果等于 x*sin(x)
    assert trigsimp(x*cos(x)*tan(x)) == x*sin(x)
    # 断言 trigsimp(-sin(x) + cos(x)*tan(x)) 的简化结果等于 0
    assert trigsimp(-sin(x) + cos(x)*tan(x)) == 0


# 定义用于测试 trigsimp 函数的测试用例
def test_trigsimp_issue_3826():
    # 断言 trigsimp(tan(2*x).expand(trig=True)) 的简化结果等于 tan(2*x)
    assert trigsimp(tan(2*x).expand(trig=True)) == tan(2*x)


# 定义用于测试 trigsimp 函数的测试用例
def test_trigsimp_issue_4032():
    # 创建符号变量 n，其为正整数
    n = Symbol('n', integer=True, positive=True)
    # 断言 trigsimp(2**(n/2)*cos(pi*n/4)/2 + 2**(n - 1)/2) 的简化结果等于 2**(n/2)*cos(pi*n/4)/2 + 2**n/4
    assert trigsimp(2**(n/2)*cos(pi*n/4)/2 + 2**(n - 1)/2) == \
        2**(n/2)*cos(pi*n/4)/2 + 2**n/4


# 定义用于测试 trigsimp 函数的测试用例
def test_trigsimp_issue_7761():
    # 断言 trigsimp(cosh(pi/4)) 的简化结果等于 cosh(pi/4)
    assert trigsimp(cosh(pi/4)) == cosh(pi/4)


# 定义用于测试 trigsimp 函数的测试用例
def test_trigsimp_noncommutative():
    # 创建符号变量 x, y
    x, y = symbols('x,y')
    # 创建不可交换符号变量 A, B
    A, B = symbols('A,B', commutative=False)

    # 以下为多个 trigsimp 函数的断言，测试三角函数简化性质
    assert trigsimp(A - A*sin(x)**2) == A*cos(x)**2
    assert trigsimp(A - A*cos(x)**2) == A*sin(x)**2
    assert trigsimp(A*sin(x)**2 + A*cos(x)**2) == A
    assert trigsimp(A + A*tan(x)**2) == A/cos(x)**2
    assert trigsimp(A/cos(x)**2 - A) == A*tan(x)**2
    assert trigsimp(A/cos(x)**2 - A*tan(x)**2) == A
    assert trigsimp(A + A*cot(x)**2) == A/sin(x)**2
    assert trigsimp(A/sin(x)**2 - A) == A/tan(x)**2
    assert trigsimp(A/sin(x)**2 - A*cot(x)**2) == A

    assert trigsimp(y*A*cos(x)**2 + y*A*sin(x)**2) == y*A

    assert trigsimp(A*sin(x)/cos(x)) == A*tan(x)
    assert trigsimp(A*tan(x)*cos(x)) == A*sin(x)
    assert trigsimp(A*cot(x)**3*sin(x)**3) == A*cos(x)**3
    assert trigsimp(y*A*tan(x)**2/sin(x)**2) == y*A/cos(x)**2
    assert trigsimp(A*cot(x)/cos(x)) == A/sin(x)

    assert trigsimp(A*sin(x + y) + A*sin(x - y)) == 2*A*sin(x)*cos(y)
    assert trigsimp(A*sin(x + y) - A*sin(x - y)) == 2*A*sin(y)*cos(x)
    assert trigsimp(A*cos(x + y) + A*cos(x - y)) == 2*A*cos(x)*cos(y)
    assert trigsimp(A*cos(x + y) - A*cos(x - y)) == -2*A*sin(x)*sin(y)

    assert trigsimp(A*sinh(x + y) + A*sinh(x - y)) == 2*A*sinh(x)*cosh(y)
    assert trigsimp(A*sinh(x + y) - A*sinh(x - y)) == 2*A*sinh(y)*cosh(x)
    assert trigsimp(A*cosh(x + y) + A*cosh(x - y)) == 2*A*cosh(x)*cosh(y)
    assert trigsimp(A*cosh(x + y) - A*cosh(x - y)) == 2*A*sinh(x)*sinh(y)

    assert trigsimp(A*cos(0.12345)**2 + A*sin(0.12345)**2) == 1.0*A


# 定义用于测试 trigsimp 函数的测试用例
def test_hyperbolic_simp():
    # 创建符号变量 x, y
    x, y = symbols('x,y')

    # 以下为多个 trigsimp 函数的断言，测试双曲函数简化性质
    assert trigsimp(sinh(x)**2 + 1) == cosh(x)**2
    assert trigsimp(cosh(x)**2 - 1) == sinh(x)**2
    assert trigsimp(cosh(x)**2 - sinh(x)**2) == 1
    assert trigsimp(1 - tanh(x)**2) == 1/cosh(x)**2
    assert trigsimp(1 - 1/cosh(x)**2) == tanh(x)**2
    assert trigsimp(tanh(x)**2 + 1/cosh(x)**2) == 1
    assert trigsimp(coth(x)**2 - 1) == 1/sinh(x)**2
    assert trigsimp(1/sinh(x)**2 + 1) == 1/tanh(x)**2
    assert trigsimp(coth(x)**2 - 1/sinh(x)**2) == 1

    assert trigsimp(5*cosh(x)**2 - 5*sinh(x)**2) == 5
    assert trigsimp(5*cosh(x/2)**2 - 2*sinh(x/2)**2) == 3*cosh(x)/2 + Rational(7, 2)

    assert trigsimp(sinh(x)/cosh(x)) == tanh(x)
    assert trigsimp(tanh(x)) == trigsimp(sinh(x)/cosh(x))
    assert trigsimp(cosh(x)/sinh(x)) == 1/tanh(x)
    assert trigsimp(2*tanh(x)*cosh(x)) == 2*sinh(x)
    assert trigsimp(coth(x)**3*sinh(x)**3) == cosh(x)**3
    # 断言简化三角函数表达式：y*tanh(x)**2/sinh(x)**2 等于 y/cosh(x)**2
    assert trigsimp(y*tanh(x)**2/sinh(x)**2) == y/cosh(x)**2
    # 断言简化三角函数表达式：coth(x)/cosh(x) 等于 1/sinh(x)
    assert trigsimp(coth(x)/cosh(x)) == 1/sinh(x)

    # 对于三个不同的角度a，分别进行断言简化
    for a in (pi/6*I, pi/4*I, pi/3*I):
        # 断言简化三角函数表达式：sinh(a)*cosh(x) + cosh(a)*sinh(x) 等于 sinh(x + a)
        assert trigsimp(sinh(a)*cosh(x) + cosh(a)*sinh(x)) == sinh(x + a)
        # 断言简化三角函数表达式：-sinh(a)*cosh(x) + cosh(a)*sinh(x) 等于 sinh(x - a)
        assert trigsimp(-sinh(a)*cosh(x) + cosh(a)*sinh(x)) == sinh(x - a)

    # 定义表达式 e
    e = 2*cosh(x)**2 - 2*sinh(x)**2
    # 断言简化对数表达式：log(e) 等于 log(2)
    assert trigsimp(log(e)) == log(2)

    # issue 19535：断言简化平方根表达式：sqrt(cosh(x)**2 - 1) 等于 sqrt(sinh(x)**2)
    assert trigsimp(sqrt(cosh(x)**2 - 1)) == sqrt(sinh(x)**2)

    # 断言简化包含递归的三角函数表达式：cosh(x)**2*cosh(y)**2 - cosh(x)**2*sinh(y)**2 - sinh(x)**2 等于 1
    assert trigsimp(cosh(x)**2*cosh(y)**2 - cosh(x)**2*sinh(y)**2 - sinh(x)**2,
            recursive=True) == 1
    # 断言简化包含递归的三角函数表达式：sinh(x)**2*sinh(y)**2 - sinh(x)**2*cosh(y)**2 + cosh(x)**2 等于 1
    assert trigsimp(sinh(x)**2*sinh(y)**2 - sinh(x)**2*cosh(y)**2 + cosh(x)**2,
            recursive=True) == 1

    # 断言近似计算三角函数表达式：2.0*cosh(x)**2 - 2.0*sinh(x)**2 接近 2.0
    assert abs(trigsimp(2.0*cosh(x)**2 - 2.0*sinh(x)**2) - 2.0) < 1e-10

    # 断言简化三角函数表达式：sinh(x)**2/cosh(x)**2 等于 tanh(x)**2
    assert trigsimp(sinh(x)**2/cosh(x)**2) == tanh(x)**2
    # 断言简化三角函数表达式：sinh(x)**3/cosh(x)**3 等于 tanh(x)**3
    assert trigsimp(sinh(x)**3/cosh(x)**3) == tanh(x)**3
    # 断言简化三角函数表达式：sinh(x)**10/cosh(x)**10 等于 tanh(x)**10
    assert trigsimp(sinh(x)**10/cosh(x)**10) == tanh(x)**10
    # 断言简化三角函数表达式：cosh(x)**3/sinh(x)**3 等于 1/tanh(x)**3
    assert trigsimp(cosh(x)**3/sinh(x)**3) == 1/tanh(x)**3

    # 断言简化三角函数表达式：cosh(x)/sinh(x) 等于 1/tanh(x)
    assert trigsimp(cosh(x)/sinh(x)) == 1/tanh(x)
    # 断言简化三角函数表达式：cosh(x)**2/sinh(x)**2 等于 1/tanh(x)**2
    assert trigsimp(cosh(x)**2/sinh(x)**2) == 1/tanh(x)**2
    # 断言简化三角函数表达式：cosh(x)**10/sinh(x)**10 等于 1/tanh(x)**10
    assert trigsimp(cosh(x)**10/sinh(x)**10) == 1/tanh(x)**10

    # 断言简化三角函数表达式：x*cosh(x)*tanh(x) 等于 x*sinh(x)
    assert trigsimp(x*cosh(x)*tanh(x)) == x*sinh(x)
    # 断言简化三角函数表达式：-sinh(x) + cosh(x)*tanh(x) 等于 0
    assert trigsimp(-sinh(x) + cosh(x)*tanh(x)) == 0

    # 断言 tan(x) 不等于 1/cot(x)，因为 cot(x) 不会自动简化
    assert tan(x) != 1/cot(x)
    # 断言简化三角函数表达式：tan(x) - 1/cot(x) 等于 0
    assert trigsimp(tan(x) - 1/cot(x)) == 0
    # 断言简化三角函数表达式：3*tanh(x)**7 - 2/coth(x)**7 等于 tanh(x)**7
    assert trigsimp(3*tanh(x)**7 - 2/coth(x)**7) == tanh(x)**7
def test_trigsimp_groebner():
    # 导入 trigsimp_groebner 函数从 sympy.simplify.trigsimp 模块
    from sympy.simplify.trigsimp import trigsimp_groebner

    # 定义一些符号表达式
    c = cos(x)  # 定义 cos(x)
    s = sin(x)  # 定义 sin(x)

    # 构造一个复杂的表达式
    ex = (4*s*c + 12*s + 5*c**3 + 21*c**2 + 23*c + 15)/(
        -s*c**2 + 2*s*c + 15*s + 7*c**3 + 31*c**2 + 37*c + 21)
    
    # 定义两个结果表达式的分子和分母
    resnum = (5*s - 5*c + 1)
    resdenom = (8*s - 6*c)
    
    # 构造结果集合
    results = [resnum/resdenom, (-resnum)/(-resdenom)]
    
    # 断言 trigsimp_groebner 函数对 ex 的结果在 results 中
    assert trigsimp_groebner(ex) in results
    
    # 断言 trigsimp_groebner 函数对 s/c 使用 hints=[tan] 得到 tan(x)
    assert trigsimp_groebner(s/c, hints=[tan]) == tan(x)
    
    # 断言 trigsimp_groebner 函数对 c*s 得到 c*s
    assert trigsimp_groebner(c*s) == c*s
    
    # 断言 trigsimp 函数对 (-s + 1)/c + c/(-s + 1) 使用 method='groebner' 得到 2/c
    assert trigsimp((-s + 1)/c + c/(-s + 1),
                    method='groebner') == 2/c
    
    # 断言 trigsimp 函数对 (-s + 1)/c + c/(-s + 1) 使用 method='groebner', polynomial=True 得到 2/c
    assert trigsimp((-s + 1)/c + c/(-s + 1),
                    method='groebner', polynomial=True) == 2/c
    
    # 测试 quick=False 参数是否有效
    assert trigsimp_groebner(ex, hints=[2]) in results
    assert trigsimp_groebner(ex, hints=[int(2)]) in results
    
    # 测试复数 I 的情况
    assert trigsimp_groebner(sin(I*x)/cos(I*x), hints=[tanh]) == I*tanh(x)
    
    # 测试双曲函数和和的情况
    assert trigsimp_groebner((tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)),
                             hints=[(tanh, x, y)]) == tanh(x + y)


def test_issue_2827_trigsimp_methods():
    # 定义两个测量函数
    measure1 = lambda expr: len(str(expr))
    measure2 = lambda expr: -count_ops(expr)
    
    # 定义一个表达式
    expr = (x + 1)/(x + sin(x)**2 + cos(x)**2)
    
    # 定义一个预期的结果矩阵
    ans = Matrix([1])
    M = Matrix([expr])
    
    # 断言 trigsimp 函数使用 method='fu', measure=measure1 得到 ans
    assert trigsimp(M, method='fu', measure=measure1) == ans
    
    # 断言 trigsimp 函数使用 method='fu', measure=measure2 不等于 ans
    assert trigsimp(M, method='fu', measure=measure2) != ans
    
    # 测试所有方法对 Basic 表达式的适用性，即使它们不是 Expr 类型
    M = Matrix.eye(1)
    assert all(trigsimp(M, method=m) == M for m in
        'fu matching groebner old'.split())
    
    # 测试 exptrigsimp 函数对 E 的处理，不仅仅是 exp() 函数
    eq = 1/sqrt(E) + E
    assert exptrigsimp(eq) == eq


def test_issue_15129_trigsimp_methods():
    # 定义三个向量
    t1 = Matrix([sin(Rational(1, 50)), cos(Rational(1, 50)), 0])
    t2 = Matrix([sin(Rational(1, 25)), cos(Rational(1, 25)), 0])
    t3 = Matrix([cos(Rational(1, 25)), sin(Rational(1, 25)), 0])
    
    # 定义两个点乘结果
    r1 = t1.dot(t2)
    r2 = t1.dot(t3)
    
    # 断言 trigsimp 函数对 r1 和 r2 的结果
    assert trigsimp(r1) == cos(Rational(1, 50))
    assert trigsimp(r2) == sin(Rational(3, 50))


def test_exptrigsimp():
    # 定义一个验证函数
    def valid(a, b):
        from sympy.core.random import verify_numerically as tn
        if not (tn(a, b) and a == b):
            return False
        return True
    
    # 断言 exptrigsimp 函数的几个常见结果
    assert exptrigsimp(exp(x) + exp(-x)) == 2*cosh(x)
    assert exptrigsimp(exp(x) - exp(-x)) == 2*sinh(x)
    assert exptrigsimp((2*exp(x)-2*exp(-x))/(exp(x)+exp(-x))) == 2*tanh(x)
    assert exptrigsimp((2*exp(2*x)-2)/(exp(2*x)+1)) == 2*tanh(x)
    
    # 定义几个表达式和它们的期望结果
    e = [cos(x) + I*sin(x), cos(x) - I*sin(x),
         cosh(x) - sinh(x), cosh(x) + sinh(x)]
    ok = [exp(I*x), exp(-I*x), exp(-x), exp(x)]
    
    # 断言所有经过 exptrigsimp 处理的结果与期望结果相符合
    assert all(valid(i, j) for i, j in zip(
        [exptrigsimp(ei) for ei in e], ok))

    # 定义另外几个表达式
    ue = [cos(x) + sin(x), cos(x) - sin(x),
          cosh(x) + I*sinh(x), cosh(x) - I*sinh(x)]
    # 确保对于每个表达式 ei 在 ue 中，经过 exptrigsimp 处理后结果与原始表达式 ei 相等
    assert [exptrigsimp(ei) == ei for ei in ue]

    # 初始化结果列表
    res = []
    # 预定义期望的表达式列表 ok
    ok = [y*tanh(1), 1/(y*tanh(1)), I*y*tan(1), -I/(y*tan(1)),
        y*tanh(x), 1/(y*tanh(x)), I*y*tan(x), -I/(y*tan(x)),
        y*tanh(1 + I), 1/(y*tanh(1 + I))]
    
    # 遍历预定义表达式，并计算简化后的结果并加入到 res 中
    for a in (1, I, x, I*x, 1 + I):
        w = exp(a)
        eq = y*(w - 1/w)/(w + 1/w)
        res.append(simplify(eq))
        res.append(simplify(1/eq))
    
    # 确保所有简化后的结果与预期的结果列表 ok 相匹配
    assert all(valid(i, j) for i, j in zip(res, ok))

    # 对于范围在 1 到 2 的整数 a 进行迭代
    for a in range(1, 3):
        w = exp(a)
        # 计算 w + 1/w 并简化
        e = w + 1/w
        s = simplify(e)
        # 确保简化后的结果与 exptrigsimp 处理后的结果相等
        assert s == exptrigsimp(e)
        # 确保简化后的结果与 2*cosh(a) 相等
        assert valid(s, 2*cosh(a))
        
        # 计算 w - 1/w 并简化
        e = w - 1/w
        s = simplify(e)
        # 确保简化后的结果与 exptrigsimp 处理后的结果相等
        assert s == exptrigsimp(e)
        # 确保简化后的结果与 2*sinh(a) 相等
        assert valid(s, 2*sinh(a))
# 定义一个测试函数，用于测试非交换符号的指数简化
def test_exptrigsimp_noncommutative():
    # 创建两个非交换符号 a 和 b
    a, b = symbols('a b', commutative=False)
    # 创建一个交换符号 x
    x = Symbol('x', commutative=True)
    
    # 断言：exp(a + x) 应该等于 exptrigsimp(exp(a)*exp(x))
    assert exp(a + x) == exptrigsimp(exp(a)*exp(x))
    
    # 创建一个表达式 p
    p = exp(a)*exp(b) - exp(b)*exp(a)
    # 断言：p 应该等于 exptrigsimp(p)，且不为零
    assert p == exptrigsimp(p) != 0

# 定义一个测试函数，用于测试幂简化在数字上的应用
def test_powsimp_on_numbers():
    # 断言：2**(Rational(1, 3) - 2) 应该等于 2**Rational(1, 3)/4
    assert 2**(Rational(1, 3) - 2) == 2**Rational(1, 3)/4

# 定义一个测试函数，标记为 XFAIL，测试特定问题
@XFAIL
def test_issue_6811_fail():
    # 定义符号变量 xp, y, x, z
    xp, y, x, z = symbols('xp, y, x, z')
    # 创建复杂表达式 eq
    eq = 4*(-19*sin(x)*y + 5*sin(3*x)*y + 15*cos(2*x)*z - 21*z)*xp/(9*cos(x) - 5*cos(3*x))
    # 断言：trigsimp(eq) 应该等于简化后的表达式
    assert trigsimp(eq) == -2*(2*cos(x)*tan(x)*y + 3*z)*xp/cos(x)

# 定义一个测试函数，用于测试 Piecewise 函数
def test_Piecewise():
    # 定义三个表达式 e1, e2, e3
    e1 = x*(x + y) - y*(x + y)
    e2 = sin(x)**2 + cos(x)**2
    e3 = expand((x + y)*y/x)
    
    # 断言：trigsimp(Piecewise((e1, e3 < e2), (e3, True))) 应该等于 Piecewise((e1, e3 < simplify(e2)), (e3, True))
    assert trigsimp(Piecewise((e1, e3 < e2), (e3, True))) == \
        Piecewise((e1, e3 < simplify(e2)), (e3, True))

# 定义一个测试函数，用于测试特定问题
def test_issue_21594():
    # 断言：simplify(exp(Rational(1,2)) + exp(Rational(-1,2))) 应该等于 cosh(S.Half)*2
    assert simplify(exp(Rational(1,2)) + exp(Rational(-1,2))) == cosh(S.Half)*2

# 定义一个测试函数，用于测试 trigsimp 在旧模式下的行为
def test_trigsimp_old():
    # 定义符号变量 x, y
    x, y = symbols('x,y')
    
    # 一系列 trigsimp 的旧模式下的断言
    assert trigsimp(1 - sin(x)**2, old=True) == cos(x)**2
    assert trigsimp(1 - cos(x)**2, old=True) == sin(x)**2
    assert trigsimp(sin(x)**2 + cos(x)**2, old=True) == 1
    assert trigsimp(1 + tan(x)**2, old=True) == 1/cos(x)**2
    assert trigsimp(1/cos(x)**2 - 1, old=True) == tan(x)**2
    assert trigsimp(1/cos(x)**2 - tan(x)**2, old=True) == 1
    assert trigsimp(1 + cot(x)**2, old=True) == 1/sin(x)**2
    assert trigsimp(1/sin(x)**2 - cot(x)**2, old=True) == 1
    assert trigsimp(5*cos(x)**2 + 5*sin(x)**2, old=True) == 5
    assert trigsimp(sin(x)/cos(x), old=True) == tan(x)
    assert trigsimp(2*tan(x)*cos(x), old=True) == 2*sin(x)
    assert trigsimp(cot(x)**3*sin(x)**3, old=True) == cos(x)**3
    assert trigsimp(y*tan(x)**2/sin(x)**2, old=True) == y/cos(x)**2
    assert trigsimp(cot(x)/cos(x), old=True) == 1/sin(x)
    assert trigsimp(sin(x + y) + sin(x - y), old=True) == 2*sin(x)*cos(y)
    assert trigsimp(sin(x + y) - sin(x - y), old=True) == 2*sin(y)*cos(x)
    assert trigsimp(cos(x + y) + cos(x - y), old=True) == 2*cos(x)*cos(y)
    assert trigsimp(cos(x + y) - cos(x - y), old=True) == -2*sin(x)*sin(y)
    assert trigsimp(sinh(x + y) + sinh(x - y), old=True) == 2*sinh(x)*cosh(y)
    assert trigsimp(sinh(x + y) - sinh(x - y), old=True) == 2*sinh(y)*cosh(x)
    assert trigsimp(cosh(x + y) + cosh(x - y), old=True) == 2*cosh(x)*cosh(y)
    assert trigsimp(cosh(x + y) - cosh(x - y), old=True) == 2*sinh(x)*sinh(y)
    assert trigsimp(cos(0.12345)**2 + sin(0.12345)**2, old=True) == 1.0
    assert trigsimp(sin(x)/cos(x), old=True, method='combined') == tan(x)
    # 使用 SymPy 的 trigsimp 函数简化 sin(x)/cos(x)，默认不应用任何方法
    assert trigsimp(sin(x)/cos(x), old=True, method='groebner') == sin(x)/cos(x)
    
    # 使用 SymPy 的 trigsimp 函数简化 sin(x)/cos(x)，指定使用 'groebner' 方法，并提供提示参数 [tan]
    assert trigsimp(sin(x)/cos(x), old=True, method='groebner', hints=[tan]) == tan(x)
    
    # 使用 SymPy 的 trigsimp 函数深度简化表达式 1 - sin(sin(x)**2 + cos(x)**2)**2，应用 'old=True' 和 'deep=True' 参数
    assert trigsimp(1 - sin(sin(x)**2 + cos(x)**2)**2, old=True, deep=True) == cos(1)**2
# 定义一个测试函数，用于验证反三角函数的逆运算是否正确
def test_trigsimp_inverse():
    # 声明一个符号变量 alpha
    alpha = symbols('alpha')
    # 分别计算 sin(alpha) 和 cos(alpha)
    s, c = sin(alpha), cos(alpha)

    # 对每个反三角函数进行测试，验证其逆运算是否正确
    for finv in [asin, acos, asec, acsc, atan, acot]:
        # 获取当前反三角函数的逆运算
        f = finv.inverse(None)
        # 使用 trigsimp 函数进行逆运算的简化，并断言结果应该与 alpha 相等
        assert alpha == trigsimp(finv(f(alpha)), inverse=True)

    # 对 atan2(cos, sin), atan2(sin, cos) 等进行测试
    for a, b in [[c, s], [s, c]]:
        # 遍历 (-1, 1) 组合的所有可能
        for i, j in product([-1, 1], repeat=2):
            # 计算 atan2(i*b, j*a) 的角度
            angle = atan2(i*b, j*a)
            # 使用 trigsimp 函数进行角度的逆运算简化
            angle_inverted = trigsimp(angle, inverse=True)
            # 断言简化后的角度应不等于原始角度，以确保有简化发生
            assert angle_inverted != angle
            # 断言简化后的 sin(angle_inverted) 应与 sin(angle) 等价
            assert sin(angle_inverted) == trigsimp(sin(angle))
            # 断言简化后的 cos(angle_inverted) 应与 cos(angle) 等价
            assert cos(angle_inverted) == trigsimp(cos(angle))
```