# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_simplify.py`

```
from sympy.concrete.summations import Sum  # 导入求和符号的相关函数
from sympy.core.add import Add  # 导入加法相关函数
from sympy.core.basic import Basic  # 导入基本的符号操作函数
from sympy.core.expr import unchanged  # 导入表达式操作相关函数
from sympy.core.function import (count_ops, diff, expand, expand_multinomial, Function, Derivative)  # 导入函数操作相关函数
from sympy.core.mul import Mul, _keep_coeff  # 导入乘法相关函数
from sympy.core import GoldenRatio  # 导入黄金比例常数
from sympy.core.numbers import (E, Float, I, oo, pi, Rational, zoo)  # 导入数值常数和对象
from sympy.core.relational import (Eq, Lt, Gt, Ge, Le)  # 导入关系运算相关函数
from sympy.core.singleton import S  # 导入单例对象
from sympy.core.symbol import (Symbol, symbols)  # 导入符号相关函数
from sympy.core.sympify import sympify  # 导入符号表达式化函数
from sympy.functions.combinatorial.factorials import (binomial, factorial)  # 导入组合数学相关函数
from sympy.functions.elementary.complexes import (Abs, sign)  # 导入复数操作相关函数
from sympy.functions.elementary.exponential import (exp, exp_polar, log)  # 导入指数和对数函数
from sympy.functions.elementary.hyperbolic import (cosh, csch, sinh)  # 导入双曲函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan)  # 导入三角函数
from sympy.functions.special.error_functions import erf  # 导入误差函数
from sympy.functions.special.gamma_functions import gamma  # 导入Gamma函数
from sympy.functions.special.hyper import hyper  # 导入超几何函数
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入Kronecker delta函数
from sympy.geometry.polygon import rad  # 导入弧度函数
from sympy.integrals.integrals import (Integral, integrate)  # 导入积分相关函数
from sympy.logic.boolalg import (And, Or)  # 导入逻辑运算相关函数
from sympy.matrices.dense import (Matrix, eye)  # 导入矩阵操作相关函数
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号相关函数
from sympy.polys.polytools import (factor, Poly)  # 导入多项式操作相关函数
from sympy.simplify.simplify import (besselsimp, hypersimp, inversecombine, logcombine, nsimplify, nthroot, posify, separatevars, signsimp, simplify)  # 导入简化表达式相关函数
from sympy.solvers.solvers import solve  # 导入求解方程的函数

from sympy.testing.pytest import XFAIL, slow, _both_exp_pow  # 导入测试框架相关函数
from sympy.abc import x, y, z, t, a, b, c, d, e, f, g, h, i, n  # 导入符号变量

def test_issue_7263():
    # 测试简化表达式的准确性
    assert abs((simplify(30.8**2 - 82.5**2 * sin(rad(11.6))**2)).evalf() - \
            673.447451402970) < 1e-12

def test_factorial_simplify():
    # 测试阶乘简化的准确性
    x = Symbol('x')
    assert simplify(factorial(x)/x) == gamma(x)
    assert simplify(factorial(factorial(x))) == factorial(factorial(x))

def test_simplify_expr():
    # 测试各种表达式简化的准确性
    x, y, z, k, n, m, w, s, A = symbols('x,y,z,k,n,m,w,s,A')
    f = Function('f')

    assert all(simplify(tmp) == tmp for tmp in [I, E, oo, x, -x, -oo, -E, -I])

    e = 1/x + 1/y
    assert e != (x + y)/(x*y)
    assert simplify(e) == (x + y)/(x*y)

    e = A**2*s**4/(4*pi*k*m**3)
    assert simplify(e) == e

    e = (4 + 4*x - 2*(2 + 2*x))/(2 + 2*x)
    assert simplify(e) == 0

    e = (-4*x*y**2 - 2*y**3 - 2*x**2*y)/(x + y)**2
    assert simplify(e) == -2*y

    e = -x - y - (x + y)**(-1)*y**2 + (x + y)**(-1)*x**2
    assert simplify(e) == -2*y

    e = (x + x*y)/x
    assert simplify(e) == 1 + y

    e = (f(x) + y*f(x))/f(x)
    assert simplify(e) == 1 + y
    # 计算表达式 e 的值，其中 n 是一个符号变量
    e = (2 * (1/n - cos(n * pi)/n))/pi
    # 使用 sympy 的 simplify 函数简化表达式 e，并断言其等于 (-cos(pi*n) + 1)/(pi*n)*2
    assert simplify(e) == (-cos(pi*n) + 1)/(pi*n)*2

    # 对 1/(x**3 + 1) 进行积分，并对结果再进行一次对 x 的微分
    e = integrate(1/(x**3 + 1), x).diff(x)
    # 使用 sympy 的 simplify 函数简化表达式 e，并断言其等于 1/(x**3 + 1)
    assert simplify(e) == 1/(x**3 + 1)

    # 对 x/(x**2 + 3*x + 1) 进行积分，并对结果再进行一次对 x 的微分
    e = integrate(x/(x**2 + 3*x + 1), x).diff(x)
    # 使用 sympy 的 simplify 函数简化表达式 e，并断言其等于 x/(x**2 + 3*x + 1)
    assert simplify(e) == x/(x**2 + 3*x + 1)

    # 创建符号变量 f
    f = Symbol('f')
    # 创建一个 2x2 的 Matrix A，计算其逆矩阵，其中包含符号变量 k, m, w
    A = Matrix([[2*k - m*w**2, -k], [-k, k - m*w**2]]).inv()
    # 使用 Matrix 运算，断言一个复杂表达式等于零
    assert simplify((A*Matrix([0, f]))[1] -
            (-f*(2*k - m*w**2)/(k**2 - (k - m*w**2)*(2*k - m*w**2)))) == 0

    # 复杂的符号表达式 f，包含符号变量 x, y, z, t, a
    f = -x + y/(z + t) + z*x/(z + t) + z*a/(z + t) + t*x/(z + t)
    # 使用 sympy 的 simplify 函数简化表达式 f，并断言其等于 (y + a*z)/(z + t)

    # issue 10347 的注释可以被忽略，不需要在代码块中包含
    # 定义表达式，这是一个复杂的数学表达式，包含多项式和三角函数
    expr = -x*(y**2 - 1)*(2*y**2*(x**2 - 1)/(a*(x**2 - y**2)**2) + (x**2 - 1)
        /(a*(x**2 - y**2)))/(a*(x**2 - y**2)) + x*(-2*x**2*sqrt(-x**2*y**2 + x**2
        + y**2 - 1)*sin(z)/(a*(x**2 - y**2)**2) - x**2*sqrt(-x**2*y**2 + x**2 +
        y**2 - 1)*sin(z)/(a*(x**2 - 1)*(x**2 - y**2)) + (x**2*sqrt((-x**2 + 1)*
        (y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*sin(z)/(x**2 - 1) + sqrt(
        (-x**2 + 1)*(y**2 - 1))*(x*(-x*y**2 + x)/sqrt(-x**2*y**2 + x**2 + y**2 -
        1) + sqrt(-x**2*y**2 + x**2 + y**2 - 1))*sin(z))/(a*sqrt((-x**2 + 1)*(
        y**2 - 1))*(x**2 - y**2)))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*sin(z)/(a*
        (x**2 - y**2)) + x*(-2*x**2*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)/(a*
        (x**2 - y**2)**2) - x**2*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)/(a*
        (x**2 - 1)*(x**2 - y**2)) + (x**2*sqrt((-x**2 + 1)*(y**2 - 1))*sqrt(-x**2
        *y**2 + x**2 + y**2 - 1)*cos(z)/(x**2 - 1) + x*sqrt((-x**2 + 1)*(y**2 -
        1))*(-x*y**2 + x)*cos(z)/sqrt(-x**2*y**2 + x**2 + y**2 - 1) + sqrt((-x**2
        + 1)*(y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z))/(a*sqrt((-x**2
        + 1)*(y**2 - 1))*(x**2 - y**2)))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(
        z)/(a*(x**2 - y**2)) - y*sqrt((-x**2 + 1)*(y**2 - 1))*(-x*y*sqrt(-x**2*
        y**2 + x**2 + y**2 - 1)*sin(z)/(a*(x**2 - y**2)*(y**2 - 1)) + 2*x*y*sqrt(
        -x**2*y**2 + x**2 + y**2 - 1)*sin(z)/(a*(x**2 - y**2)**2) + (x*y*sqrt((
        -x**2 + 1)*(y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*sin(z)/(y**2 -
        1) + x*sqrt((-x**2 + 1)*(y**2 - 1))*(-x**2*y + y)*sin(z)/sqrt(-x**2*y**2
        + x**2 + y**2 - 1))/(a*sqrt((-x**2 + 1)*(y**2 - 1))*(x**2 - y**2)))*sin(
        z)/(a*(x**2 - y**2)) + y*(x**2 - 1)*(-2*x*y*(x**2 - 1)/(a*(x**2 - y**2)
        **2) + 2*x*y/(a*(x**2 - y**2)))/(a*(x**2 - y**2)) + y*(x**2 - 1)*(y**2 -
        1)*(-x*y*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)/(a*(x**2 - y**2)*(y**2
        - 1)) + 2*x*y*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)/(a*(x**2 - y**2)
        **2) + (x*y*sqrt((-x**2 + 1)*(y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 -
        1)*cos(z)/(y**2 - 1) + x*sqrt((-x**2 + 1)*(y**2 - 1))*(-x**2*y + y)*cos(
        z)/sqrt(-x**2*y**2 + x**2 + y**2 - 1))/(a*sqrt((-x**2 + 1)*(y**2 - 1)
        )*(x**2 - y**2)))*cos(z)/(a*sqrt((-x**2 + 1)*(y**2 - 1))*(x**2 - y**2)
        ) - x*sqrt((-x**2 + 1)*(y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*sin(
        z)**2/(a**2*(x**2 - 1)*(x**2 - y**2)*(y**2 - 1)) - x*sqrt((-x**2 + 1)*(
        y**2 - 1))*sqrt(-x**2*y**2 + x**2 + y**2 - 1)*cos(z)**2/(a**2*(x**2 - 1)*(
        x**2 - y**2)*(y**2 - 1))
    
    # 使用 sympy 的 simplify 函数来简化表达式，并进行断言检查
    assert simplify(expr) == 2*x/(a**2*(x**2 - y**2))

    # issue 17631 的测试断言
    assert simplify('((-1/2)*Boole(True)*Boole(False)-1)*Boole(True)') == \
            Mul(sympify('(2 + Boole(True)*Boole(False))'), sympify('-Boole(True)/2'))

    # 创建两个符号 A 和 B，并设定它们为非交换符号
    A, B = symbols('A,B', commutative=False)

    # 检查 A*B - B*A 的简化结果
    assert simplify(A*B - B*A) == A*B - B*A
    # 检查 A/(1 + y/x) 的简化结果
    assert simplify(A/(1 + y/x)) == x*A/(x + y)
    # 断言：简化表达式 A*(1/x + 1/y)，应该等于 A/x + A/y
    assert simplify(A*(1/x + 1/y)) == A/x + A/y  #(x + y)*A/(x*y)
    
    # 断言：简化 log(2) + log(3)，应该等于 log(6)
    assert simplify(log(2) + log(3)) == log(6)
    
    # 断言：简化 log(2*x) - log(2)，应该等于 log(x)
    assert simplify(log(2*x) - log(2)) == log(x)
    
    # 断言：简化超几何函数 hyper([], [], x)，应该等于 exp(x)
    assert simplify(hyper([], [], x)) == exp(x)
def test_issue_3557():
    # 定义三个线性方程
    f_1 = x*a + y*b + z*c - 1
    f_2 = x*d + y*e + z*f - 1
    f_3 = x*g + y*h + z*i - 1

    # 解方程组[solve函数用于解方程组]
    solutions = solve([f_1, f_2, f_3], x, y, z, simplify=False)

    # 断言简化后的y解为给定表达式
    assert simplify(solutions[y]) == \
        (a*i + c*d + f*g - a*f - c*g - d*i)/ \
        (a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g)


def test_simplify_other():
    # 断言简化三角恒等式sin^2(x) + cos^2(x)为1
    assert simplify(sin(x)**2 + cos(x)**2) == 1
    # 断言简化伽马函数的比值为x
    assert simplify(gamma(x + 1)/gamma(x)) == x
    # 断言简化复杂表达式
    assert simplify(sin(x)**2 + cos(x)**2 + factorial(x)/gamma(x)) == 1 + x
    # 断言简化等式sin^2(x) + cos^2(x) == factorial(x)/gamma(x)为等价形式Eq(x, 1)
    assert simplify(
        Eq(sin(x)**2 + cos(x)**2, factorial(x)/gamma(x))) == Eq(x, 1)
    # 定义非交换符号nc
    nc = symbols('nc', commutative=False)
    # 断言简化表达式x + x*nc为等价形式x*(1 + nc)
    assert simplify(x + x*nc) == x*(1 + nc)
    # issue 6123
    # f = exp(-I*(k*sqrt(t) + x/(2*sqrt(t)))**2)
    # ans = integrate(f, (k, -oo, oo), conds='none')
    # 断言简化复杂数学表达式ans
    ans = I*(-pi*x*exp(I*pi*Rational(-3, 4) + I*x**2/(4*t))*erf(x*exp(I*pi*Rational(-3, 4))/
        (2*sqrt(t)))/(2*sqrt(t)) + pi*x*exp(I*pi*Rational(-3, 4) + I*x**2/(4*t))/
        (2*sqrt(t)))*exp(-I*x**2/(4*t))/(sqrt(pi)*x) - I*sqrt(pi) * \
        (-erf(x*exp(I*pi/4)/(2*sqrt(t))) + 1)*exp(I*pi/4)/(2*sqrt(t))
    assert simplify(ans) == -(-1)**Rational(3, 4)*sqrt(pi)/sqrt(t)
    # issue 6370
    # 断言简化2**(2 + x)/4为等价形式2**x
    assert simplify(2**(2 + x)/4) == 2**x


@_both_exp_pow
def test_simplify_complex():
    # 将cos(x)和tan(x)表示为指数形式，并断言其乘积的简化结果为sin(x)
    cosAsExp = cos(x)._eval_rewrite_as_exp(x)
    tanAsExp = tan(x)._eval_rewrite_as_exp(x)
    assert simplify(cosAsExp*tanAsExp) == sin(x)  # issue 4341

    # issue 10124
    # 断言矩阵指数函数的简化结果
    assert simplify(exp(Matrix([[0, -1], [1, 0]]))) == Matrix([[cos(1),
        -sin(1)], [sin(1), cos(1)]])


def test_simplify_ratio():
    # 根据根的表达式简化
    roots = ['(1/2 - sqrt(3)*I/2)*(sqrt(21)/2 + 5/2)**(1/3) + 1/((1/2 - '
             'sqrt(3)*I/2)*(sqrt(21)/2 + 5/2)**(1/3))',
             '1/((1/2 + sqrt(3)*I/2)*(sqrt(21)/2 + 5/2)**(1/3)) + '
             '(1/2 + sqrt(3)*I/2)*(sqrt(21)/2 + 5/2)**(1/3)',
             '-(sqrt(21)/2 + 5/2)**(1/3) - 1/(sqrt(21)/2 + 5/2)**(1/3)']

    for r in roots:
        r = S(r)
        # 断言简化表达式，且其操作数不超过原表达式的操作数
        assert count_ops(simplify(r, ratio=1)) <= count_ops(r)
        # 如果ratio=oo，则始终应用simplify()
        assert simplify(r, ratio=oo) is not r


def test_simplify_measure():
    # 定义两个度量函数
    measure1 = lambda expr: len(str(expr))
    measure2 = lambda expr: -count_ops(expr)
    # 定义复杂表达式expr
    expr = (x + 1)/(x + sin(x)**2 + cos(x)**2)
    # 断言简化expr后长度的度量不超过原始表达式
    assert measure1(simplify(expr, measure=measure1)) <= measure1(expr)
    # 断言简化expr后操作数的度量不超过原始表达式
    assert measure2(simplify(expr, measure=measure2)) <= measure2(expr)

    # 定义等式表达式expr2
    expr2 = Eq(sin(x)**2 + cos(x)**2, 1)
    # 断言简化expr2后长度的度量不超过原始表达式
    assert measure1(simplify(expr2, measure=measure1)) <= measure1(expr2)
    # 断言简化expr2后操作数的度量不超过原始表达式
    assert measure2(simplify(expr2, measure=measure2)) <= measure2(expr2)


def test_simplify_rational():
    # 定义表达式expr
    expr = 2**x*2.**y
    # 断言有理化简结果为2**(x+y)
    assert simplify(expr, rational=True) == 2**(x+y)
    # 断言不指定有理化时结果为2.0**(x+y)
    assert simplify(expr, rational=None) == 2.0**(x+y)
    # 断言不进行有理化简时结果为原表达式
    assert simplify(expr, rational=False) == expr
    # 使用断言进行测试，确保调用 simplify 函数对于输入 '0.9 - 0.8 - 0.1' 并设置 rational 参数为 True 的返回值等于 0
    assert simplify('0.9 - 0.8 - 0.1', rational=True) == 0
# 定义用于测试 simplify 函数的测试用例
def test_simplify_issue_1308():
    # 断言简化 exp(-1/2) + exp(-3/2) 后的结果
    assert simplify(exp(Rational(-1, 2)) + exp(Rational(-3, 2))) == \
        (1 + E)*exp(Rational(-3, 2))


# 定义用于测试 simplify 函数的另一个测试用例
def test_issue_5652():
    # 断言简化 E + exp(-E) 后的结果
    assert simplify(E + exp(-E)) == exp(-E)
    # 定义一个非可交换符号 n
    n = symbols('n', commutative=False)
    # 断言简化 n + n**(-n) 后的结果
    assert simplify(n + n**(-n)) == n + n**(-n)


# 定义用于测试 simplify 函数失败情况的测试用例
def test_simplify_fail1():
    # 定义符号 x 和 y
    x = Symbol('x')
    y = Symbol('y')
    # 定义表达式 e
    e = (x + y)**2/(-4*x*y**2 - 2*y**3 - 2*x**2*y)
    # 断言简化 e 后的结果
    assert simplify(e) == 1 / (-2*y)


# 定义用于测试 nthroot 函数的测试用例
def test_nthroot():
    # 断言计算 nthroot(90 + 34*sqrt(7), 3) 后的结果
    assert nthroot(90 + 34*sqrt(7), 3) == sqrt(7) + 3
    # 定义符号 q
    q = 1 + sqrt(2) - 2*sqrt(3) + sqrt(6) + sqrt(7)
    # 断言计算 nthroot(expand_multinomial(q**3), 3) 后的结果
    assert nthroot(expand_multinomial(q**3), 3) == q
    # 断言计算 nthroot(41 + 29*sqrt(2), 5) 后的结果
    assert nthroot(41 + 29*sqrt(2), 5) == 1 + sqrt(2)
    # 断言计算 nthroot(-41 - 29*sqrt(2), 5) 后的结果
    assert nthroot(-41 - 29*sqrt(2), 5) == -1 - sqrt(2)
    # 定义表达式 expr
    expr = 1320*sqrt(10) + 4216 + 2576*sqrt(6) + 1640*sqrt(15)
    # 断言计算 nthroot(expr, 5) 后的结果
    assert nthroot(expr, 5) == 1 + sqrt(6) + sqrt(15)
    # 断言计算 expand_multinomial(nthroot(expand_multinomial(q**5), 5)) 后的结果
    assert expand_multinomial(nthroot(expand_multinomial(q**5), 5)) == q
    # 断言计算 nthroot(expand_multinomial(q**5), 5, 8) 后的结果
    assert nthroot(expand_multinomial(q**5), 5, 8) == q
    # 断言计算 nthroot(expand_multinomial(q**3), 3) 后的结果
    assert nthroot(expand_multinomial(q**3), 3) == q
    # 断言计算 nthroot(expand_multinomial(q**6), 6) 后的结果
    assert nthroot(expand_multinomial(q**6), 6) == q


# 定义用于测试 nthroot 函数的另一个测试用例
def test_nthroot1():
    # 定义符号 q
    q = 1 + sqrt(2) + sqrt(3) + S.One/10**20
    # 计算 p = expand_multinomial(q**5)
    p = expand_multinomial(q**5)
    # 断言计算 nthroot(p, 5) 后的结果
    assert nthroot(p, 5) == q
    # 定义符号 q
    q = 1 + sqrt(2) + sqrt(3) + S.One/10**30
    # 计算 p = expand_multinomial(q**5)
    p = expand_multinomial(q**5)
    # 断言计算 nthroot(p, 5) 后的结果
    assert nthroot(p, 5) == q


# 定义用于测试 separatevars 函数的测试用例
@_both_exp_pow
def test_separatevars():
    # 定义符号 x, y, z, n
    x, y, z, n = symbols('x,y,z,n')
    # 断言 separatevars(2*n*x*z + 2*x*y*z) 后的结果
    assert separatevars(2*n*x*z + 2*x*y*z) == 2*x*z*(n + y)
    # 断言 separatevars(x*z + x*y*z) 后的结果
    assert separatevars(x*z + x*y*z) == x*z*(1 + y)
    # 断言 separatevars(pi*x*z + pi*x*y*z) 后的结果
    assert separatevars(pi*x*z + pi*x*y*z) == pi*x*z*(1 + y)
    # 断言 separatevars(x*y**2*sin(x) + x*sin(x)*sin(y)) 后的结果
    assert separatevars(x*y**2*sin(x) + x*sin(x)*sin(y)) == \
        x*(sin(y) + y**2)*sin(x)
    # 断言 separatevars(x*exp(x + y) + x*exp(x)) 后的结果
    assert separatevars(x*exp(x + y) + x*exp(x)) == x*(1 + exp(y))*exp(x)
    # 断言 separatevars((x*(y + 1))**z).is_Pow 后的结果
    assert separatevars((x*(y + 1))**z).is_Pow  # != x**z*(1 + y)**z
    # 断言 separatevars(1 + x + y + x*y) 后的结果
    assert separatevars(1 + x + y + x*y) == (x + 1)*(y + 1)
    # 断言 separatevars(y/pi*exp(-(z - x)/cos(n))) 后的结果
    assert separatevars(y/pi*exp(-(z - x)/cos(n))) == \
        y*exp(x/cos(n))*exp(-z/cos(n))/pi
    # 断言 separatevars((x + y)*(x - y) + y**2 + 2*x + 1) 后的结果
    assert separatevars((x + y)*(x - y) + y**2 + 2*x + 1) == (x + 1)**2
    # issue 4858
    # 定义正数符号 p
    p = Symbol('p', positive=True)
    # 断言 separatevars(sqrt(p**2 + x*p**2)) 后的结果
    assert separatevars(sqrt(p**2 + x*p**2)) == p*sqrt(1 + x)
    # 断言 separatevars(sqrt(y*(p**2 + x*p**2))) 后的结果
    assert separatevars(sqrt(y*(p**2 + x*p**2))) == p*sqrt(y*(1 + x))
    # 断言 separatevars(sqrt(y*(p**2 + x*p**2)), force=True) 后的结果
    assert separatevars(sqrt(y*(p**2 + x*p**2)), force=True) == \
        p*sqrt(y)*sqrt(1 + x)
    # issue 4865
    # 断言 separatevars(sqrt(x*y)).is_Pow 后的结果
    assert separatevars(sqrt(x*y)).is_Pow
    # 断言 separatevars(sqrt(x*y), force=True) 后的结果
    assert separatevars(sqrt(x*y), force=True) == sqrt(x)*sqrt(y)
    # issue 4957
    # 任何类型的符号序列都是允许的
    # 断言 separatevars(((2*x + 2)*y), dict=True, symbols=()) 后的结果
    assert separatevars(((2*x + 2)*y), dict=True, symbols=()) == \
        {'coeff': 1, x: 2*x + 2, y: y}
    # separable
    # 断言 separatevars(((2*x + 2)*y), dict=True, symbols=[x]) 后的结果
    assert separatevars(((2*x + 2)*y), dict=True, symbols=[x]) == \
        {'coeff': y, x: 2*x + 2}
    # 断言 separatevars(((2*x + 2)*y), dict=True
    # 使用 separatevars 函数测试不同的输入和选项

    # 测试：((2*x + 2)*y)，预期返回 {'coeff': 1, x: 2*x + 2, y: y}
    assert separatevars(((2*x + 2)*y), dict=True) == \
        {'coeff': 1, x: 2*x + 2, y: y}

    # 测试：((2*x + 2)*y)，使用 dict=True 和 symbols=None，预期返回 {'coeff': y*(2*x + 2)}
    assert separatevars(((2*x + 2)*y), dict=True, symbols=None) == \
        {'coeff': y*(2*x + 2)}

    # 测试：对于不可分解的输入，预期返回 None
    # not separable
    assert separatevars(3, dict=True) is None

    # 测试：对于空 symbols 的输入，预期返回 None
    assert separatevars(2*x + y, dict=True, symbols=()) is None

    # 测试：对于默认选项的输入，预期返回 None
    assert separatevars(2*x + y, dict=True) is None

    # 测试：对于 symbols=None 的输入，预期返回 {'coeff': 2*x + y}
    assert separatevars(2*x + y, dict=True, symbols=None) == {'coeff': 2*x + y}

    # issue 4808 的测试
    n, m = symbols('n,m', commutative=False)
    assert separatevars(m + n*m) == (1 + n)*m
    assert separatevars(x + x*n) == x*(1 + n)

    # issue 4910 的测试
    f = Function('f')
    assert separatevars(f(x) + x*f(x)) == f(x) + x*f(x)

    # 测试：当存在不可交换对象时的情况
    eq = x*(1 + hyper((), (), y*z))
    assert separatevars(eq) == eq

    # abs(x*y) 的测试
    s = separatevars(abs(x*y))
    assert s == abs(x)*abs(y) and s.is_Mul

    # cos(1)**2 + sin(1)**2 - 1 的测试
    z = cos(1)**2 + sin(1)**2 - 1
    a = abs(x*z)

    # 对 abs(x*z) 的分解测试
    s = separatevars(a)
    assert not a.is_Mul and s.is_Mul and s == abs(x)*abs(z)

    # abs(x*y*z) 的测试
    s = separatevars(abs(x*y*z))
    assert s == abs(x)*abs(y)*abs(z)

    # abs((x+y)/z) 的测试，预期不会引发异常
    assert separatevars(abs((x+y)/z)) == abs((x+y)/z)
# 定义一个函数用于测试 separatevars 的高级因子化
def test_separatevars_advanced_factor():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 断言：对表达式进行 separatevars 操作后的结果与预期值相等
    assert separatevars(1 + log(x)*log(y) + log(x) + log(y)) == \
        (log(x) + 1)*(log(y) + 1)
    
    # 断言：对复杂表达式进行 separatevars 操作后的结果与预期值相等
    assert separatevars(1 + x - log(z) - x*log(z) - exp(y)*log(z) -
        x*exp(y)*log(z) + x*exp(y) + exp(y)) == \
        -((x + 1)*(log(z) - 1)*(exp(y) + 1))
    
    # 重新定义符号变量 x, y，且限定为正数
    x, y = symbols('x,y', positive=True)
    # 断言：对表达式进行 separatevars 操作后的结果与预期值相等
    assert separatevars(1 + log(x**log(y)) + log(x*y)) == \
        (log(x) + 1)*(log(y) + 1)


# 定义一个函数用于测试 hypersimp 函数
def test_hypersimp():
    # 定义符号变量 n, k，且限定 k 为整数
    n, k = symbols('n,k', integer=True)

    # 断言：对阶乘函数 factorial(k) 进行 hypersimp 操作后的结果与预期值相等
    assert hypersimp(factorial(k), k) == k + 1
    # 断言：对阶乘函数 factorial(k**2) 进行 hypersimp 操作后的结果应为 None
    assert hypersimp(factorial(k**2), k) is None

    # 断言：对 1/factorial(k) 进行 hypersimp 操作后的结果与预期值相等
    assert hypersimp(1/factorial(k), k) == 1/(k + 1)

    # 断言：对复杂表达式 2**k/factorial(k)**2 进行 hypersimp 操作后的结果与预期值相等
    assert hypersimp(2**k/factorial(k)**2, k) == 2/(k + 1)**2

    # 断言：对二项式系数 binomial(n, k) 进行 hypersimp 操作后的结果与预期值相等
    assert hypersimp(binomial(n, k), k) == (n - k)/(k + 1)
    # 断言：对 n+1 与 k 的二项式系数 binomial(n + 1, k) 进行 hypersimp 操作后的结果与预期值相等
    assert hypersimp(binomial(n + 1, k), k) == (n - k + 1)/(k + 1)

    # 定义一个复杂的数学表达式 term
    term = (4*k + 1)*factorial(k)/factorial(2*k + 1)
    # 断言：对复杂表达式 term 进行 hypersimp 操作后的结果与预期值相等
    assert hypersimp(term, k) == S.Half*((4*k + 5)/(3 + 14*k + 8*k**2))

    # 定义一个复杂的数学表达式 term
    term = 1/((2*k - 1)*factorial(2*k + 1))
    # 断言：对复杂表达式 term 进行 hypersimp 操作后的结果与预期值相等
    assert hypersimp(term, k) == (k - S.Half)/((k + 1)*(2*k + 1)*(2*k + 3))

    # 定义一个复杂的数学表达式 term
    term = binomial(n, k)*(-1)**k/factorial(k)
    # 断言：对复杂表达式 term 进行 hypersimp 操作后的结果与预期值相等
    assert hypersimp(term, k) == (k - n)/(k + 1)**2


# 定义一个函数用于测试 nsimplify 函数
def test_nsimplify():
    # 定义符号变量 x
    x = Symbol("x")
    # 断言：对整数 0 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(0) == 0
    # 断言：对整数 -1 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(-1) == -1
    # 断言：对整数 1 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(1) == 1
    # 断言：对表达式 1 + x 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(1 + x) == 1 + x
    # 断言：对浮点数 2.7 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(2.7) == Rational(27, 10)
    # 断言：对表达式 1 - GoldenRatio 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(1 - GoldenRatio) == (1 - sqrt(5))/2
    # 断言：对表达式 (1 + sqrt(5))/4 进行 nsimplify 操作后的结果与预期值相等，且以 GoldenRatio 为基准
    assert nsimplify((1 + sqrt(5))/4, [GoldenRatio]) == GoldenRatio/2
    # 断言：对表达式 2/GoldenRatio 进行 nsimplify 操作后的结果与预期值相等，且以 GoldenRatio 为基准
    assert nsimplify(2/GoldenRatio, [GoldenRatio]) == 2*GoldenRatio - 2
    # 断言：对表达式 exp(pi*I*Rational(5, 3), evaluate=False) 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(exp(pi*I*Rational(5, 3), evaluate=False)) == \
        sympify('1/2 - sqrt(3)*I/2')
    # 断言：对表达式 sin(pi*Rational(3, 5), evaluate=False) 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(sin(pi*Rational(3, 5), evaluate=False)) == \
        sympify('sqrt(sqrt(5)/8 + 5/8)')
    # 断言：对表达式 sqrt(atan('1', evaluate=False))*(2 + I) 进行 nsimplify 操作后的结果与预期值相等，且以 pi 为基准
    assert nsimplify(sqrt(atan('1', evaluate=False))*(2 + I), [pi]) == \
        sqrt(pi) + sqrt(pi)/2*I
    # 断言：对表达式 2 + exp(2*atan('1/4')*I) 进行 nsimplify 操作后的结果与预期值相等
    assert nsimplify(2 + exp(2*atan('1/4')*I)) == sympify('49/17 + 8*I/17')
    # 断言：对数值 pi 进行 nsimplify 操作后的结果与预期值相等，允许误差为 0.01
    assert nsimplify(pi, tolerance=0.01) == Rational(22, 7)
    # 断言：对数值 pi 进行 nsimplify 操作后的结果与预期值相等，允许误差为 0.001
    assert nsimplify(pi, tolerance=0.001) == Rational(355, 113)
    # 断言：对数值 0.33333 进行 nsimplify 操作后的结果与预期值相等，允许误差为 1e-4
    assert nsimplify(0.33333, tolerance=1e-4) == Rational(1, 3)
    # 断言：对数值 2.0**(1/3.) 进行 nsimplify 操作后的结果与预期值相等，允许误差为 0.001
    assert nsimplify(2.0**(1/3.), tolerance=0.001) == Rational(635, 504)
    # 断言：对数值 2.0**(1/3.) 进行
    # 使用符号运算简化表达式，确保其不包含浮点数
    assert not nsimplify(
        factor(-3.0*z**2*(z**2)**(-2.5) + 3*(z**2)**(-1.5))).atoms(Float)

    # 计算 x 的零次幂，验证结果为幂运算类型，并且 x**0.0 简化为 1
    e = x**0.0
    assert e.is_Pow and nsimplify(x**0.0) == 1

    # 使用 nsimplify 函数对浮点数进行有理化简，确保在给定的容差范围内结果为有理数
    assert nsimplify(3.333333, tolerance=0.1, rational=True) == Rational(10, 3)
    assert nsimplify(3.333333, tolerance=0.01, rational=True) == Rational(10, 3)
    assert nsimplify(3.666666, tolerance=0.1, rational=True) == Rational(11, 3)
    assert nsimplify(3.666666, tolerance=0.01, rational=True) == Rational(11, 3)
    assert nsimplify(33, tolerance=10, rational=True) == Rational(33)
    assert nsimplify(33.33, tolerance=10, rational=True) == Rational(30)
    assert nsimplify(37.76, tolerance=10, rational=True) == Rational(40)
    assert nsimplify(-203.1) == Rational(-2031, 10)
    assert nsimplify(.2, tolerance=0) == Rational(1, 5)
    assert nsimplify(-.2, tolerance=0) == Rational(-1, 5)
    assert nsimplify(.2222, tolerance=0) == Rational(1111, 5000)
    assert nsimplify(-.2222, tolerance=0) == Rational(-1111, 5000)

    # 解决问题7211和PR4112，确保对于非常小的S(2e-8)，nsimplify返回正确的有理数
    assert nsimplify(S(2e-8)) == Rational(1, 50000000)

    # 解决问题7322，确保对于非常小的数1e-42，有理化简不会将其简化为0
    assert nsimplify(1e-42, rational=True) != 0

    # 解决问题10336，对无穷大进行符号函数和符号化简操作的测试
    inf = Float('inf')
    infs = (-oo, oo, inf, -inf)
    for zi in infs:
        ans = sign(zi)*oo
        assert nsimplify(zi) == ans
        assert nsimplify(zi + x) == x + ans

    # 使用 nsimplify 对表达式进行有理化简，并确保在有理化简中使用完整精度
    assert nsimplify(0.33333333, rational=True, rational_conversion='exact') == Rational(0.33333333)

    # 确保表达式的有理化简使用完整的精度
    assert nsimplify(pi.evalf(100)*x, rational_conversion='exact').evalf(100) == pi.evalf(100)*x
def test_issue_9448():
    # 使用 sympify 函数将数学表达式转换为符号表达式
    tmp = sympify("1/(1 - (-1)**(2/3) - (-1)**(1/3)) + 1/(1 + (-1)**(2/3) + (-1)**(1/3))")
    # 断言简化后的 tmp 符号表达式等于分数 1/2
    assert nsimplify(tmp) == S.Half


def test_extract_minus_sign():
    # 定义符号变量 x, y, a, b
    x = Symbol("x")
    y = Symbol("y")
    a = Symbol("a")
    b = Symbol("b")
    # 下面的断言测试简化表达式的结果
    assert simplify(-x/-y) == x/y  # 简化 -x/-y 为 x/y
    assert simplify(-x/y) == -x/y  # 简化 -x/y 保持不变
    assert simplify(x/y) == x/y  # 简化 x/y 保持不变
    assert simplify(x/-y) == -x/y  # 简化 x/-y 为 -x/y
    assert simplify(-x/0) == zoo*x  # 简化 -x/0 为无穷大乘以 x
    assert simplify(Rational(-5, 0)) is zoo  # 简化有理数 -5/0 为无穷大
    assert simplify(-a*x/(-y - b)) == a*x/(b + y)  # 简化 -a*x/(-y - b) 为 a*x/(b + y)


def test_diff():
    # 定义符号变量 x, y，并创建函数 f(x), g(x)
    x = Symbol("x")
    y = Symbol("y")
    f = Function("f")
    g = Function("g")
    # 下面的断言测试微分表达式的简化结果
    assert simplify(g(x).diff(x)*f(x).diff(x) - f(x).diff(x)*g(x).diff(x)) == 0  # 简化 g(x)*f'(x) - f(x)*g'(x) 为 0
    assert simplify(2*f(x)*f(x).diff(x) - diff(f(x)**2, x)) == 0  # 简化 2*f(x)*f'(x) - f(x)^2 的导数 为 0
    assert simplify(diff(1/f(x), x) + f(x).diff(x)/f(x)**2) == 0  # 简化 1/f(x) 的导数 + f'(x)/f(x)^2 为 0
    assert simplify(f(x).diff(x, y) - f(x).diff(y, x)) == 0  # 简化 f(x) 在 x, y 两个变量下的混合偏导数 为 0


def test_logcombine_1():
    # 定义符号变量 x, y, a, b, z, w，并设置部分变量的属性
    x, y = symbols("x,y")
    a = Symbol("a")
    z, w = symbols("z,w", positive=True)
    b = Symbol("b", real=True)
    # 下面的断言测试对数合并函数 logcombine 的不同情况
    assert logcombine(log(x) + 2*log(y)) == log(x) + 2*log(y)  # 对 log(x) + 2*log(y) 不进行合并
    assert logcombine(log(x) + 2*log(y), force=True) == log(x*y**2)  # 强制合并 log(x) + 2*log(y) 为 log(x*y^2)
    assert logcombine(a*log(w) + log(z)) == a*log(w) + log(z)  # 对 a*log(w) + log(z) 不进行合并
    assert logcombine(b*log(z) + b*log(x)) == log(z**b) + b*log(x)  # 合并 b*log(z) + b*log(x) 为 log(z^b) + b*log(x)
    assert logcombine(b*log(z) - log(w)) == log(z**b/w)  # 合并 b*log(z) - log(w) 为 log(z^b/w)
    assert logcombine(log(x)*log(z)) == log(x)*log(z)  # 对 log(x)*log(z) 不进行合并
    assert logcombine(log(w)*log(x)) == log(w)*log(x)  # 对 log(w)*log(x) 不进行合并
    assert logcombine(cos(-2*log(z) + b*log(w))) in [cos(log(w**b/z**2)), cos(log(z**2/w**b))]  # 合并 cos(-2*log(z) + b*log(w)) 为其规范形式之一
    assert logcombine(log(log(x) - log(y)) - log(z), force=True) == log(log(x/y)/z)  # 强制合并 log(log(x/y) - log(z)) 为 log(log(x/y)/z)
    assert logcombine((2 + I)*log(x), force=True) == (2 + I)*log(x)  # 对 (2 + I)*log(x) 不进行合并
    assert logcombine((x**2 + log(x) - log(y))/(x*y), force=True) == (x**2 + log(x/y))/(x*y)  # 强制合并复杂表达式
    assert logcombine(gamma(-log(x/y))*acos(-log(x/y)), force=True) == acos(-log(x/y))*gamma(-log(x/y))  # 合并 gamma(-log(x/y))*acos(-log(x/y))

    assert logcombine(2*log(z)*log(w)*log(x) + log(z) + log(w)) == log(z**log(w**2))*log(x) + log(w*z)  # 合并复杂对数表达式
    assert logcombine(3*log(w) + 3*log(z)) == log(w**3*z**3)  # 合并 3*log(w) + 3*log(z) 为 log(w^3*z^3)
    assert logcombine(x*(y + 1) + log(2) + log(3)) == x*(y + 1) + log(6)  # 合并 x*(y + 1) + log(2) + log(3) 为 x*(y + 1) + log(6)
    assert logcombine((x + y)*log(w) + (-x - y)*log(3)) == (x + y)*log(w/3)  # 合并 (x + y)*log(w) + (-x - y)*log(3) 为 (x + y)*log(w/3)
    assert logcombine(log(x) + log(2)) == log(2*x)  # 合并 log(x) + log(2) 为 log(2*x)
    eq = log(abs(x)) + log(abs(y))
    assert logcombine(eq) == eq  # 不合并 log(abs(x)) + log(abs(y))

    reps = {x: 0, y: 0}
    assert log(abs(x)*abs(y)).subs(reps) != eq.subs(reps)  # 检查 log(abs(x)*abs(y)) 在特定替换下的结果是否与预期不同
# 定义测试函数 test_logcombine_complex_coeff
def test_logcombine_complex_coeff():
    # 创建一个积分对象 i，积分表达式为 (sin(x**2) + cos(x**3))/x
    i = Integral((sin(x**2) + cos(x**3))/x, x)
    # 断言对 i 进行对数合并操作后结果与 i 相等
    assert logcombine(i, force=True) == i
    # 断言对 i + 2*log(x) 进行对数合并操作后的结果
    assert logcombine(i + 2*log(x), force=True) == i + log(x**2)


# 定义测试函数 test_issue_5950
def test_issue_5950():
    # 声明符号 x, y，并指定它们为正数
    x, y = symbols("x,y", positive=True)
    # 断言对 log(3) - log(2) 进行对数合并后的结果
    assert logcombine(log(3) - log(2)) == log(Rational(3,2), evaluate=False)
    # 断言对 log(x) - log(y) 进行对数合并后的结果
    assert logcombine(log(x) - log(y)) == log(x/y)
    # 断言对 log(Rational(3,2), evaluate=False) - log(2) 进行对数合并后的结果
    assert logcombine(log(Rational(3,2), evaluate=False) - log(2)) == \
        log(Rational(3,4), evaluate=False)


# 定义测试函数 test_posify
def test_posify():
    # 声明符号 x
    x = symbols('x')

    # 断言将表达式 x + Symbol('p', positive=True) + Symbol('n', negative=True) 进行正数化后的结果
    assert str(posify(
        x +
        Symbol('p', positive=True) +
        Symbol('n', negative=True))) == '(_x + n + p, {_x: x})'

    # 断言对 1/x 进行正数化后的表达式对数化后的结果
    eq, rep = posify(1/x)
    assert log(eq).expand().subs(rep) == -log(x)
    # 断言对 [x, 1 + x] 进行正数化后的结果
    assert str(posify([x, 1 + x])) == '([_x, _x + 1], {_x: x})'

    # 声明符号 p, n，并指定它们的正负属性
    p = symbols('p', positive=True)
    n = symbols('n', negative=True)
    orig = [x, n, p]
    # 断言对 orig 列表进行正数化后的结果
    modified, reps = posify(orig)
    assert str(modified) == '[_x, n, p]'
    assert [w.subs(reps) for w in modified] == orig

    # 断言对 Integral(posify(1/x + y)[0], (y, 1, 3)) 进行正数化后的积分表达式展开的结果
    assert str(Integral(posify(1/x + y)[0], (y, 1, 3)).expand()) == \
        'Integral(1/_x, (y, 1, 3)) + Integral(_y, (y, 1, 3))'
    # 断言对 Sum(posify(1/x**n)[0], (n,1,3)) 进行正数化后的求和表达式展开的结果
    assert str(Sum(posify(1/x**n)[0], (n,1,3)).expand()) == \
        'Sum(_x**(-n), (n, 1, 3))'

    # issue 16438 的断言
    k = Symbol('k', finite=True)
    eq, rep = posify(k)
    assert eq.assumptions0 == {'positive': True, 'zero': False, 'imaginary': False,
     'nonpositive': False, 'commutative': True, 'hermitian': True, 'real': True, 'nonzero': True,
     'nonnegative': True, 'negative': False, 'complex': True, 'finite': True,
     'infinite': False, 'extended_real':True, 'extended_negative': False,
     'extended_nonnegative': True, 'extended_nonpositive': False,
     'extended_nonzero': True, 'extended_positive': True}


# 定义测试函数 test_issue_4194
def test_issue_4194():
    # simplify 函数应该调用 cancel 函数
    f = Function('f')
    # 断言 simplify((4*x + 6*f(y))/(2*x + 3*f(y))) 的结果为 2
    assert simplify((4*x + 6*f(y))/(2*x + 3*f(y))) == 2


# 标记为 XFAIL 的测试函数 test_simplify_float_vs_integer
@XFAIL
def test_simplify_float_vs_integer():
    # 对 issue 4473 进行测试
    assert simplify(x**2.0 - x**2) == 0
    assert simplify(x**2 - x**2.0) == 0


# 定义测试函数 test_as_content_primitive
def test_as_content_primitive():
    # 断言 (x/2 + y).as_content_primitive() 的结果
    assert (x/2 + y).as_content_primitive() == (S.Half, x + 2*y)
    # 断言 (x/2 + y).as_content_primitive(clear=False) 的结果
    assert (x/2 + y).as_content_primitive(clear=False) == (S.One, x/2 + y)
    # 断言 (y*(x/2 + y)).as_content_primitive() 的结果
    assert (y*(x/2 + y)).as_content_primitive() == (S.Half, y*(x + 2*y))
    # 断言 (y*(x/2 + y)).as_content_primitive(clear=False) 的结果
    assert (y*(x/2 + y)).as_content_primitive(clear=False) == (S.One, y*(x/2 + y))

    # 断言 (x*(2 + 2*x)*(3*x + 3)**2).as_content_primitive() 的结果
    assert (x*(2 + 2*x)*(3*x + 3)**2).as_content_primitive() == \
        (18, x*(x + 1)**3)
    # 断言 (2 + 2*x + 2*y*(3 + 3*y)).as_content_primitive() 的结果
    assert (2 + 2*x + 2*y*(3 + 3*y)).as_content_primitive() == \
        (2, x + 3*y*(y + 1) + 1)
    # 断言 ((2 + 6*x)**2).as_content_primitive() 的结果
    assert ((2 + 6*x)**2).as_content_primitive() == \
        (4, (3*x + 1)**2)
    assert ((2 + 6*x)**(2*y)).as_content_primitive() == \
        (1, (_keep_coeff(S(2), (3*x + 1)))**(2*y))
    # 断言：检查表达式 (2 + 6*x)^(2*y) 的原始内容。
    # 返回结果：应为 (1, (2*(3*x + 1))^(2*y))

    assert (5 + 10*x + 2*y*(3 + 3*y)).as_content_primitive() == \
        (1, 10*x + 6*y*(y + 1) + 5)
    # 断言：检查表达式 5 + 10*x + 2*y*(3 + 3*y) 的原始内容。
    # 返回结果：应为 (1, 10*x + 6*y*(y + 1) + 5)

    assert (5*(x*(1 + y)) + 2*x*(3 + 3*y)).as_content_primitive() == \
        (11, x*(y + 1))
    # 断言：检查表达式 5*(x*(1 + y)) + 2*x*(3 + 3*y) 的原始内容。
    # 返回结果：应为 (11, x*(y + 1))

    assert ((5*(x*(1 + y)) + 2*x*(3 + 3*y))**2).as_content_primitive() == \
        (121, x**2*(y + 1)**2)
    # 断言：检查表达式 (5*(x*(1 + y)) + 2*x*(3 + 3*y))^2 的原始内容。
    # 返回结果：应为 (121, x^2*(y + 1)^2)

    assert (y**2).as_content_primitive() == \
        (1, y**2)
    # 断言：检查表达式 y^2 的原始内容。
    # 返回结果：应为 (1, y^2)

    assert (S.Infinity).as_content_primitive() == (1, oo)
    # 断言：检查表达式 Infinity 的原始内容。
    # 返回结果：应为 (1, oo)

    eq = x**(2 + y)
    assert (eq).as_content_primitive() == (1, eq)
    # 断言：检查表达式 x^(2 + y) 的原始内容。
    # 返回结果：应为 (1, x^(2 + y))

    assert (S.Half**(2 + x)).as_content_primitive() == (Rational(1, 4), 2**(-x))
    # 断言：检查表达式 (1/2)^(2 + x) 的原始内容。
    # 返回结果：应为 (1/4, 2^(-x))

    assert (Rational(-1, 2)**(2 + x)).as_content_primitive() == \
           (Rational(1, 4), (Rational(-1, 2))**x)
    # 断言：检查表达式 (-1/2)^(2 + x) 的原始内容。
    # 返回结果：应为 (1/4, (-1/2)^x)

    assert (Rational(-1, 2)**(2 + x)).as_content_primitive() == \
           (Rational(1, 4), Rational(-1, 2)**x)
    # 断言：检查表达式 (-1/2)^(2 + x) 的原始内容（另一次调用，结果应该相同）。
    # 返回结果：应为 (1/4, (-1/2)^x)

    assert (4**((1 + y)/2)).as_content_primitive() == (2, 4**(y/2))
    # 断言：检查表达式 4^((1 + y)/2) 的原始内容。
    # 返回结果：应为 (2, 4^(y/2))

    assert (3**((1 + y)/2)).as_content_primitive() == \
           (1, 3**(Mul(S.Half, 1 + y, evaluate=False)))
    # 断言：检查表达式 3^((1 + y)/2) 的原始内容。
    # 返回结果：应为 (1, 3^(1/2 * (1 + y))，使用 Mul 表示乘法，避免求值)

    assert (5**Rational(3, 4)).as_content_primitive() == (1, 5**Rational(3, 4))
    # 断言：检查表达式 5^(3/4) 的原始内容。
    # 返回结果：应为 (1, 5^(3/4))

    assert (5**Rational(7, 4)).as_content_primitive() == (5, 5**Rational(3, 4))
    # 断言：检查表达式 5^(7/4) 的原始内容。
    # 返回结果：应为 (5, 5^(3/4))

    assert Add(z*Rational(5, 7), 0.5*x, y*Rational(3, 2), evaluate=False).as_content_primitive() == \
              (Rational(1, 14), 7.0*x + 21*y + 10*z)
    # 断言：检查复合表达式 z*(5/7) + 0.5*x + y*(3/2) 的原始内容。
    # 返回结果：应为 (1/14, 7.0*x + 21*y + 10*z)

    assert (2**Rational(3, 4) + 2**Rational(1, 4)*sqrt(3)).as_content_primitive(radical=True) == \
           (1, 2**Rational(1, 4)*(sqrt(2) + sqrt(3)))
    # 断言：检查表达式 2^(3/4) + 2^(1/4)*sqrt(3) 的原始内容。
    # 返回结果：应为 (1, 2^(1/4)*(sqrt(2) + sqrt(3)))
# 定义测试函数 test_signsimp，用于测试符号简化函数 signsimp 的功能
def test_signsimp():
    # 计算表达式 e，并断言其符号简化后为 S.true
    e = x*(-x + 1) + x*(x - 1)
    assert signsimp(Eq(e, 0)) is S.true
    # 断言绝对值表达式 Abs(x - 1) 等于 Abs(1 - x)
    assert Abs(x - 1) == Abs(1 - x)
    # 断言 signsimp(y - x) 等于 y - x
    assert signsimp(y - x) == y - x
    # 断言使用 evaluate=False 参数的 signsimp(y - x) 等于 Mul(-1, x - y, evaluate=False)
    assert signsimp(y - x, evaluate=False) == Mul(-1, x - y, evaluate=False)

# 定义测试函数 test_besselsimp，用于测试贝塞尔函数简化相关功能
def test_besselsimp():
    # 导入贝塞尔函数相关模块和函数
    from sympy.functions.special.bessel import (besseli, besselj, bessely)
    from sympy.integrals.transforms import cosine_transform
    # 断言贝塞尔函数表达式的简化结果
    assert besselsimp(exp(-I*pi*y/2)*besseli(y, z*exp_polar(I*pi/2))) == \
        besselj(y, z)
    assert besselsimp(exp(-I*pi*a/2)*besseli(a, 2*sqrt(x)*exp_polar(I*pi/2))) == \
        besselj(a, 2*sqrt(x))
    assert besselsimp(sqrt(2)*sqrt(pi)*x**Rational(1, 4)*exp(I*pi/4)*exp(-I*pi*a/2) *
                      besseli(Rational(-1, 2), sqrt(x)*exp_polar(I*pi/2)) *
                      besseli(a, sqrt(x)*exp_polar(I*pi/2))/2) == \
        besselj(a, sqrt(x)) * cos(sqrt(x))
    assert besselsimp(besseli(Rational(-1, 2), z)) == \
        sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besseli(a, z*exp_polar(-I*pi/2))) == \
        exp(-I*pi*a/2)*besselj(a, z)
    assert cosine_transform(1/t*sin(a/t), t, y) == \
        sqrt(2)*sqrt(pi)*besselj(0, 2*sqrt(a)*sqrt(y))/2

    # 复杂的贝塞尔函数表达式简化断言
    assert besselsimp(x**2*(a*(-2*besselj(5*I, x) + besselj(-2 + 5*I, x) +
    besselj(2 + 5*I, x)) + b*(-2*bessely(5*I, x) + bessely(-2 + 5*I, x) +
    bessely(2 + 5*I, x)))/4 + x*(a*(besselj(-1 + 5*I, x)/2 - besselj(1 + 5*I, x)/2)
    + b*(bessely(-1 + 5*I, x)/2 - bessely(1 + 5*I, x)/2)) + (x**2 + 25)*(a*besselj(5*I, x)
    + b*bessely(5*I, x))) == 0

    assert besselsimp(81*x**2*(a*(besselj(Rational(-5, 3), 9*x) - 2*besselj(Rational(1, 3), 9*x) + besselj(Rational(7, 3), 9*x))
    + b*(bessely(Rational(-5, 3), 9*x) - 2*bessely(Rational(1, 3), 9*x) + bessely(Rational(7, 3), 9*x)))/4 + x*(a*(9*besselj(Rational(-2, 3), 9*x)/2
    - 9*besselj(Rational(4, 3), 9*x)/2) + b*(9*bessely(Rational(-2, 3), 9*x)/2 - 9*bessely(Rational(4, 3), 9*x)/2)) +
    (81*x**2 - Rational(1, 9))*(a*besselj(Rational(1, 3), 9*x) + b*bessely(Rational(1, 3), 9*x))) == 0

    assert besselsimp(besselj(a-1,x) + besselj(a+1, x) - 2*a*besselj(a, x)/x) == 0

    assert besselsimp(besselj(a-1,x) + besselj(a+1, x) + besselj(a, x)) == (2*a + x)*besselj(a, x)/x

    assert besselsimp(x**2* besselj(a,x) + x**3*besselj(a+1, x) + besselj(a+2, x)) == \
    2*a*x*besselj(a + 1, x) + x**3*besselj(a + 1, x) - x**2*besselj(a + 2, x) + 2*x*besselj(a + 1, x) + besselj(a + 2, x)

# 定义测试函数 test_Piecewise，用于测试分段函数 Piecewise 的简化功能
def test_Piecewise():
    # 定义表达式 e1, e2, e3
    e1 = x*(x + y) - y*(x + y)
    e2 = sin(x)**2 + cos(x)**2
    e3 = expand((x + y)*y/x)
    # 简化表达式 s1, s2, s3
    s1 = simplify(e1)
    s2 = simplify(e2)
    s3 = simplify(e3)
    # 断言分段函数简化后的结果
    assert simplify(Piecewise((e1, x < e2), (e3, True))) == \
        Piecewise((s1, x < s2), (s3, True))

# 定义测试函数 test_polymorphism，用于测试多态性相关功能
def test_polymorphism():
    # 定义类 A，继承自 Basic 类
    class A(Basic):
        # 定义 _eval_simplify 方法，返回 S.One
        def _eval_simplify(x, **kwargs):
            return S.One

    # 创建 A 类的实例 a
    a = A(S(5), S(2))
    # 断言对实例 a 的简化结果为 1
    assert simplify(a) == 1

# 定义测试函数 test_issue_from_PR1599，用于测试来自 PR1599 的问题
def test_issue_from_PR1599():
    # 定义负符号的符号变量 n1, n2, n3, n4
    n1, n2, n3, n4 = symbols('n1 n2 n3 n4', negative=True)
    # 断言：简化 I*sqrt(n1) 的结果是否等于 -sqrt(-n1)
    assert simplify(I*sqrt(n1)) == -sqrt(-n1)
# 测试函数，用于验证表达式的简化功能
def test_issue_6811():
    # 定义表达式 eq
    eq = (x + 2*y)*(2*x + 2)
    # 断言简化后的 eq 应该等于 (x + 1)*(x + 2*y)*2
    assert simplify(eq) == (x + 1)*(x + 2*y)*2
    # 进行表达式扩展并断言简化后的结果
    assert simplify(eq.expand()) == \
        2*x**2 + 4*x*y + 2*x + 4*y


def test_issue_6920():
    # 定义包含四个复杂表达式的列表 e
    e = [cos(x) + I*sin(x), cos(x) - I*sin(x),
        cosh(x) - sinh(x), cosh(x) + sinh(x)]
    # 定义预期结果的列表 ok
    ok = [exp(I*x), exp(-I*x), exp(-x), exp(x)]
    # 定义函数 f
    f = Function('f')
    # 断言对 e 中每个元素 ei 应用 f 简化后的第一个参数应该等于 ok 中的对应元素
    assert [simplify(f(ei)).args[0] for ei in e] == ok


def test_issue_7001():
    # 导入变量 r 和 R
    from sympy.abc import r, R
    # 断言简化后的复杂表达式结果
    assert simplify(-(r*Piecewise((pi*Rational(4, 3), r <= R),
        (-8*pi*R**3/(3*r**3), True)) + 2*Piecewise((pi*r*Rational(4, 3), r <= R),
        (4*pi*R**3/(3*r**2), True)))/(4*pi*r)) == \
        Piecewise((-1, r <= R), (0, True))


def test_inequality_no_auto_simplify():
    # 定义左右两侧的表达式
    lhs = cos(x)**2 + sin(x)**2
    rhs = 2
    # 创建一个不自动简化的 Less Than 对象 e
    e = Lt(lhs, rhs, evaluate=False)
    # 断言 e 不等于 S.true
    assert e is not S.true
    # 断言对 e 应用简化后得到一个有效的表达式
    assert simplify(e)


def test_issue_9398():
    # 导入所需的库
    from sympy.core.numbers import Number
    from sympy.polys.polytools import cancel
    # 一系列断言，验证 cancel 和 simplify 对给定的小数和虚数的处理
    assert cancel(1e-14) != 0
    assert cancel(1e-14*I) != 0

    assert simplify(1e-14) != 0
    assert simplify(1e-14*I) != 0

    assert (I*Number(1.)*Number(10)**Number(-14)).simplify() != 0

    assert cancel(1e-20) != 0
    assert cancel(1e-20*I) != 0

    assert simplify(1e-20) != 0
    assert simplify(1e-20*I) != 0

    assert cancel(1e-100) != 0
    assert cancel(1e-100*I) != 0

    assert simplify(1e-100) != 0
    assert simplify(1e-100*I) != 0

    # 创建一个非常小的浮点数 f
    f = Float("1e-1000")
    assert cancel(f) != 0
    assert cancel(f*I) != 0

    assert simplify(f) != 0
    assert simplify(f*I) != 0


def test_issue_9324_simplify():
    # 定义一个矩阵符号 M
    M = MatrixSymbol('M', 10, 10)
    # 定义一个包含矩阵元素的复杂表达式 e
    e = M[0, 0] + M[5, 4] + 1304
    # 断言对表达式 e 进行简化后应该仍然等于 e 自身
    assert simplify(e) == e


def test_issue_9817_simplify():
    # 简化替换后的显式二次形式矩阵表达式的迹
    from sympy.matrices.expressions import Identity, trace
    # 定义符号和矩阵
    v = MatrixSymbol('v', 3, 1)
    A = MatrixSymbol('A', 3, 3)
    x = Matrix([i + 1 for i in range(3)])
    X = Identity(3)
    quadratic = v.T * A * v
    # 断言简化后的结果与预期值相等
    assert simplify((trace(quadratic.as_explicit())).xreplace({v:x, A:X})) == 14


@_both_exp_pow
def test_simplify_function_inverse():
    # "inverse" 属性不能保证 f(g(x)) 等于 x，因此这种简化不应自动发生
    # 参见 issue #12140
    x, y = symbols('x, y')
    g = Function('g')

    class f(Function):
        def inverse(self, argindex=1):
            return g

    # 断言简化 f(g(x)) 后仍等于 f(g(x))
    assert simplify(f(g(x))) == f(g(x))
    # 断言 inversecombine(f(g(x))) 应该等于 x
    assert inversecombine(f(g(x))) == x
    # 断言：简化 f(g(x)) 并求逆操作后结果应为 x
    assert simplify(f(g(x)), inverse=True) == x

    # 断言：简化 f(g(sin(x)**2 + cos(x)**2)) 并求逆操作后结果应为 1
    assert simplify(f(g(sin(x)**2 + cos(x)**2)), inverse=True) == 1

    # 断言：简化 f(g(x, y)) 并求逆操作后结果应为 f(g(x, y))
    assert simplify(f(g(x, y)), inverse=True) == f(g(x, y))

    # 断言：asin 函数应保持不变
    assert unchanged(asin, sin(x))

    # 断言：简化 asin(sin(x)) 结果应为 asin(sin(x))
    assert simplify(asin(sin(x))) == asin(sin(x))

    # 断言：简化 2*asin(sin(3*x)) 并求逆操作后结果应为 6*x
    assert simplify(2*asin(sin(3*x)), inverse=True) == 6*x

    # 断言：简化 log(exp(x)) 结果应为 log(exp(x))
    assert simplify(log(exp(x))) == log(exp(x))

    # 断言：简化 log(exp(x)) 并求逆操作后结果应为 x
    assert simplify(log(exp(x)), inverse=True) == x

    # 断言：简化 exp(log(x)) 并求逆操作后结果应为 x
    assert simplify(exp(log(x)), inverse=True) == x

    # 断言：简化 log(exp(x), 2) 并求逆操作后结果应为 x/log(2)
    assert simplify(log(exp(x), 2), inverse=True) == x/log(2)

    # 断言：简化 log(exp(x), 2, evaluate=False) 并求逆操作后结果应为 x/log(2)
    assert simplify(log(exp(x), 2, evaluate=False), inverse=True) == x/log(2)
# 定义测试函数 test_clear_coefficients，用于测试 clear_coefficients 函数的不同输入情况
def test_clear_coefficients():
    # 导入 sympy 库中的 clear_coefficients 函数
    from sympy.simplify.simplify import clear_coefficients
    # 断言：对于输入 4*y*(6*x + 3)，期望返回结果是 (y*(2*x + 1), 0)
    assert clear_coefficients(4*y*(6*x + 3)) == (y*(2*x + 1), 0)
    # 断言：对于输入 4*y*(6*x + 3) - 2，期望返回结果是 (y*(2*x + 1), Rational(1, 6))
    assert clear_coefficients(4*y*(6*x + 3) - 2) == (y*(2*x + 1), Rational(1, 6))
    # 断言：对于输入 4*y*(6*x + 3) - 2，指定变量 x，期望返回结果是 (y*(2*x + 1), x/12 + Rational(1, 6))
    assert clear_coefficients(4*y*(6*x + 3) - 2, x) == (y*(2*x + 1), x/12 + Rational(1, 6))
    # 断言：对于输入 sqrt(2) - 2，期望返回结果是 (sqrt(2), 2)
    assert clear_coefficients(sqrt(2) - 2) == (sqrt(2), 2)
    # 断言：对于输入 4*sqrt(2) - 2，期望返回结果是 (sqrt(2), S.Half)
    assert clear_coefficients(4*sqrt(2) - 2) == (sqrt(2), S.Half)
    # 断言：对于输入 S(3)，指定变量 x，期望返回结果是 (0, x - 3)
    assert clear_coefficients(S(3), x) == (0, x - 3)
    # 断言：对于输入 S.Infinity，指定变量 x，期望返回结果是 (S.Infinity, x)
    assert clear_coefficients(S.Infinity, x) == (S.Infinity, x)
    # 断言：对于输入 -S.Pi，指定变量 x，期望返回结果是 (S.Pi, -x)
    assert clear_coefficients(-S.Pi, x) == (S.Pi, -x)
    # 断言：对于输入 2 - S.Pi/3，指定变量 x，期望返回结果是 (pi, -3*x + 6)
    assert clear_coefficients(2 - S.Pi/3, x) == (pi, -3*x + 6)

# 定义测试函数 test_nc_simplify，用于测试 nc_simplify 函数的不同输入情况
def test_nc_simplify():
    # 导入 sympy 库中的 nc_simplify 函数以及相关依赖项
    from sympy.simplify.simplify import nc_simplify
    from sympy.matrices.expressions import MatPow, Identity
    from sympy.core import Pow
    from functools import reduce

    # 定义符号变量 a, b, c, d 以及一个不可交换的符号变量 x
    a, b, c, d = symbols('a b c d', commutative = False)
    x = Symbol('x')
    # 定义矩阵符号 A, B, C, D
    A = MatrixSymbol("A", x, x)
    B = MatrixSymbol("B", x, x)
    C = MatrixSymbol("C", x, x)
    D = MatrixSymbol("D", x, x)
    # 定义替换字典 subst，将符号映射到对应的矩阵符号上
    subst = {a: A, b: B, c: C, d:D}
    # 定义函数字典 funcs，包含 Add 和 Mul 函数的自定义实现
    funcs = {Add: lambda x,y: x+y, Mul: lambda x,y: x*y }

    # 定义函数 _to_matrix，将表达式转换为矩阵表达式
    def _to_matrix(expr):
        if expr in subst:
            return subst[expr]
        if isinstance(expr, Pow):
            return MatPow(_to_matrix(expr.args[0]), expr.args[1])
        elif isinstance(expr, (Add, Mul)):
            return reduce(funcs[expr.func],[_to_matrix(a) for a in expr.args])
        else:
            return expr*Identity(x)

    # 定义函数 _check，用于检查 nc_simplify 函数的输出是否符合预期
    def _check(expr, simplified, deep=True, matrix=True):
        # 断言：调用 nc_simplify 函数对 expr 进行简化后，结果应与 simplified 相等
        assert nc_simplify(expr, deep=deep) == simplified
        # 断言：对比展开后的表达式，nc_simplify 的输出应与展开后的 simplified 相等
        assert expand(expr) == expand(simplified)
        # 如果需要矩阵形式的检查
        if matrix:
            # 将 simplified 转换为矩阵表达式，并进行处理
            m_simp = _to_matrix(simplified).doit(inv_expand=False)
            # 断言：将 expr 转换为矩阵表达式后，再经过 nc_simplify 的输出应与 m_simp 相等
            assert nc_simplify(_to_matrix(expr), deep=deep) == m_simp

    # 各种情况下的测试，每个 _check 调用表示一种测试案例
    _check(a*b*a*b*a*b*c*(a*b)**3*c, ((a*b)**3*c)**2)
    _check(a*b*(a*b)**-2*a*b, 1)
    _check(a**2*b*a*b*a*b*(a*b)**-1, a*(a*b)**2, matrix=False)
    _check(b*a*b**2*a*b**2*a*b**2, b*(a*b**2)**3)
    _check(a*b*a**2*b*a**2*b*a**3, (a*b*a)**3*a**2)
    _check(a**2*b*a**4*b*a**4*b*a**2, (a**2*b*a**2)**3)
    _check(a**3*b*a**4*b*a**4*b*a, a**3*(b*a**4)**3*a**-3)
    _check(a*b*a*b + a*b*c*x*a*b*c, (a*b)**2 + x*(a*b*c)**2)
    _check(a*b*a*b*c*a*b*a*b*c, ((a*b)**2*c)**2)
    _check(b**-1*a**-1*(a*b)**2, a*b)
    _check(a**-1*b*c**-1, (c*b**-1*a)**-1)
    expr = a**3*b*a**4*b*a**4*b*a**2*b*a**2*(b*a**2)**2*b*a**2*b*a**2
    for _ in range(10):
        expr *= a*b
    _check(expr, a**3*(b*a**4)**2*(b*a**2)**6*(a*b)**10)
    _check((a*b*a*b)**2, (a*b*a*b)**2, deep=False)
    _check(a*b*(c*d)**2, a*b*(c*d)**2)
    expr = b**-1*(a**-1*b**-1 - a**-1*c*b**-1)**-1*a**-1
    # 断言：对于特定的表达式 expr，经过 nc_simplify 函数后应返回 (1-c)**-1
    assert nc_simplify(expr) == (1-c)**-1
    # 断言：对于可交换的表达式，应该返回无错误
    assert nc_simplify(2*x**2) == 2*x**2

# 定义测试函数 test_issue_15965，用于测试特定问题 15965 的情况
def test_issue_15965():
    # 定义符号变量 z, x, y，并使用它们创建求和表达式 A 和 anew
    A = Sum(z*x**y, (x, 1, a))
    anew = z*Sum(x**y, (x, 1, a))
    # 定义一个积分对象 B，表示 x*y 的积分关于 x
    B = Integral(x*y, x)
    # 计算 bdo，表示 x^2*y/2
    bdo = x**2*y/2
    # 断言：简化 A + B 的表达式应与 anew + bdo 相等
    assert simplify(A + B) == anew + bdo
    # 断言：简化 A 的表达式应与 anew 相等
    assert simplify(A) == anew
    # 断言：简化 B 的表达式应与 bdo 相等
    assert simplify(B) == bdo
    # 断言：带有 doit=False 参数的简化 B 的表达式应与 y*Integral(x, x) 相等
    assert simplify(B, doit=False) == y*Integral(x, x)
def test_issue_17137():
    # 检查对复数指数化简是否正常
    assert simplify(cos(x)**I) == cos(x)**I
    # 检查对复数指数化简是否正常
    assert simplify(cos(x)**(2 + 3*I)) == cos(x)**(2 + 3*I)


def test_issue_21869():
    # 定义两个实数符号
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # 测试逻辑表达式简化
    expr = And(Eq(x**2, 4), Le(x, y))
    assert expr.simplify() == expr

    expr = And(Eq(x**2, 4), Eq(x, 2))
    assert expr.simplify() == Eq(x, 2)

    expr = And(Eq(x**3, x**2), Eq(x, 1))
    assert expr.simplify() == Eq(x, 1)

    expr = And(Eq(sin(x), x**2), Eq(x, 0))
    assert expr.simplify() == Eq(x, 0)

    expr = And(Eq(x**3, x**2), Eq(x, 2))
    assert expr.simplify() == S.false

    expr = And(Eq(y, x**2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y,1), Eq(x, 1))

    expr = And(Eq(y**2, 1), Eq(y, x**2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y,1), Eq(x, 1))

    expr = And(Eq(y**2, 4), Eq(y, 2*x**2), Eq(x, 1))
    assert expr.simplify() == And(Eq(y,2), Eq(x, 1))

    expr = And(Eq(y**2, 4), Eq(y, x**2), Eq(x, 1))
    assert expr.simplify() == S.false


def test_issue_7971_21740():
    # 创建积分表达式
    z = Integral(x, (x, 1, 1))
    assert z != 0
    # 检查积分表达式简化结果
    assert simplify(z) is S.Zero
    # 检查常数0的简化结果
    assert simplify(S.Zero) is S.Zero
    # 检查浮点数0.0的简化结果
    z = simplify(Float(0))
    assert z is not S.Zero and z == 0.0


@slow
def test_issue_17141_slow():
    # 检查复杂表达式的递归处理是否正常
    assert simplify((2**acos(I+1)**2).rewrite('log')) == 2**((pi + 2*I*log(-1 +
                   sqrt(1 - 2*I) + I))**2/4)


def test_issue_17141():
    # 检查复杂表达式的递归处理是否正常
    assert simplify(x**(1 / acos(I))) == x**(2/(pi - 2*I*log(1 + sqrt(2))))
    # 检查复杂表达式的化简结果
    assert simplify(acos(-I)**2*acos(I)**2) == \
           log(1 + sqrt(2))**4 + pi**2*log(1 + sqrt(2))**2/2 + pi**4/16
    # 检查复杂表达式的化简结果
    assert simplify(2**acos(I)**2) == 2**((pi - 2*I*log(1 + sqrt(2)))**2/4)
    p = 2**acos(I+1)**2
    assert simplify(p) == p


def test_simplify_kroneckerdelta():
    # 定义符号 i 和 j
    i, j = symbols("i j")
    K = KroneckerDelta

    # 检查 KroneckerDelta 函数的简化
    assert simplify(K(i, j)) == K(i, j)
    assert simplify(K(0, j)) == K(0, j)
    assert simplify(K(i, 0)) == K(i, 0)

    # 检查带有 Piecewise 的 KroneckerDelta 的简化结果
    assert simplify(K(0, j).rewrite(Piecewise) * K(1, j)) == 0
    assert simplify(K(1, i) + Piecewise((1, Eq(j, 2)), (0, True))) == K(1, i) + K(2, j)

    # issue 17214 的问题检查
    assert simplify(K(0, j) * K(1, j)) == 0

    # 带有整数限制的 KroneckerDelta 的简化检查
    n = Symbol('n', integer=True)
    assert simplify(K(0, n) * K(1, n)) == 0

    # 检查矩阵 M 的简化结果
    M = Matrix(4, 4, lambda i, j: K(j - i, n) if i <= j else 0)
    assert simplify(M**2) == Matrix([[K(0, n), 0, K(1, n), 0],
                                     [0, K(0, n), 0, K(1, n)],
                                     [0, 0, K(0, n), 0],
                                     [0, 0, 0, K(0, n)]])
    # 检查矩阵乘积的简化结果
    assert simplify(eye(1) * KroneckerDelta(0, n) *
                    KroneckerDelta(1, n)) == Matrix([[0]])

    # 检查无穷大乘以 KroneckerDelta 的简化结果
    assert simplify(S.Infinity * KroneckerDelta(0, n) *
                    KroneckerDelta(1, n)) is S.NaN


def test_issue_17292():
    # 检查绝对值表达式的简化结果
    assert simplify(abs(x)/abs(x**2)) == 1/abs(x)
    # 检查深层处理是否正常工作
    # this is bigger than the issue: check that deep processing works
    # 断言语句，用于验证表达式 simplify(5*abs((x**2 - 1)/(x - 1))) == 5*Abs(x + 1) 是否为真
    assert simplify(5*abs((x**2 - 1)/(x - 1))) == 5*Abs(x + 1)
# 定义一个测试函数，用于测试 issue 19822
def test_issue_19822():
    # 创建一个布尔表达式，要求 n-2 大于 1 且 n 大于 1
    expr = And(Gt(n-2, 1), Gt(n, 1))
    # 简化表达式并断言其结果等于 n 大于 3
    assert simplify(expr) == Gt(n, 3)


# 定义一个测试函数，用于测试 issue 18645
def test_issue_18645():
    # 创建一个布尔表达式，要求 x 大于等于 3 且 x 小于等于 3
    expr = And(Ge(x, 3), Le(x, 3))
    # 简化表达式并断言其结果等于 x 等于 3
    assert simplify(expr) == Eq(x, 3)
    # 创建另一个布尔表达式，要求 x 等于 3 且 x 小于等于 3
    expr = And(Eq(x, 3), Le(x, 3))
    # 简化表达式并断言其结果等于 x 等于 3
    assert simplify(expr) == Eq(x, 3)


# 标记为预期失败的测试函数，用于测试 issue 18642
@XFAIL
def test_issue_18642():
    # 创建整数符号 i 和 n
    i = Symbol("i", integer=True)
    n = Symbol("n", integer=True)
    # 创建一个布尔表达式，要求 i 等于 2 * n 且 i 小于等于 2*n - 1
    expr = And(Eq(i, 2 * n), Le(i, 2*n - 1))
    # 简化表达式并断言其结果为假
    assert simplify(expr) == S.false


# 标记为预期失败的测试函数，用于测试 issue 18389
@XFAIL
def test_issue_18389():
    # 创建整数符号 n
    n = Symbol("n", integer=True)
    # 创建一个布尔表达式，要求 n 等于 0 或者 n 大于等于 1
    expr = Eq(n, 0) | (n >= 1)
    # 简化表达式并断言其结果为 n 大于等于 0
    assert simplify(expr) == Ge(n, 0)


# 定义一个测试函数，用于测试 issue 8373
def test_issue_8373():
    # 创建一个实数符号 x
    x = Symbol('x', real=True)
    # 断言简化表达式 Or(x 小于 1, x 大于等于 1) 的结果为真
    assert simplify(Or(x < 1, x >= 1)) == S.true


# 定义一个测试函数，用于测试 issue 7950
def test_issue_7950():
    # 创建一个布尔表达式，要求 x 等于 1 且 x 等于 2
    expr = And(Eq(x, 1), Eq(x, 2))
    # 简化表达式并断言其结果为假
    assert simplify(expr) == S.false


# 定义一个测试函数，用于测试 issue 22020
def test_issue_22020():
    # 创建一个表达式，表示 I*pi/2 -oo
    expr = I*pi/2 -oo
    # 断言简化表达式的结果等于原表达式本身
    assert simplify(expr) == expr
    # 此前可能会引发错误


# 定义一个测试函数，用于测试 issue 19484
def test_issue_19484():
    # 断言简化表达式 sign(x) * Abs(x) 的结果等于 x
    assert simplify(sign(x) * Abs(x)) == x

    # 创建多个表达式并简化它们
    e = x + sign(x + x**3)
    assert simplify(Abs(x + x**3)*e) == x**3 + x*Abs(x**3 + x) + x

    e = x**2 + sign(x**3 + 1)
    assert simplify(Abs(x**3 + 1) * e) == x**3 + x**2*Abs(x**3 + 1) + 1

    f = Function('f')
    e = x + sign(x + f(x)**3)
    assert simplify(Abs(x + f(x)**3) * e) == x*Abs(x + f(x)**3) + x + f(x)**3


# 定义一个测试函数，用于测试 issue 23543
def test_issue_23543():
    # 创建非交换符号 x, y, z
    x, y, z = symbols("x y z", commutative=False)
    # 断言表达式 (x*(y + z/2)).simplify() 的结果等于 x*(2*y + z)/2
    assert (x*(y + z/2)).simplify() == x*(2*y + z)/2


# 定义一个测试函数，用于测试 issue 11004
def test_issue_11004():
    # 定义多个嵌套函数
    def f(n):
        return sqrt(2*pi*n) * (n/E)**n

    def m(n, k):
        return  f(n) / (f(n/k)**k)

    def p(n,k):
        return m(n, k) / (k**n)

    # 创建符号 N, k
    N, k = symbols('N k')
    # 创建表达式 z，并简化并展开其结果
    z = log(p(n, k) / p(n, k + 1)).expand(force=True)
    # 简化表达式 z 在 n 等于 N 时的数值，并断言其结果为四位小数
    r = simplify(z.subs(n, N).n(4))
    assert r == (
        half*k*log(k)
        - half*k*log(k + 1)
        + half*log(N)
        - half*log(k + 1)
        + Float(0.9189224, 4)
    )


# 定义一个测试函数，用于测试 issue 19161
def test_issue_19161():
    # 创建多项式对象并简化
    polynomial = Poly('x**2').simplify()
    # 断言多项式减去 x**2 的简化结果等于 0
    assert (polynomial-x**2).simplify() == 0


# 定义一个测试函数，用于测试 issue 22210
def test_issue_22210():
    # 创建符号 d
    d = Symbol('d', integer=True)
    # 创建表达式 expr，并简化
    expr = 2*Derivative(sin(x), (x, d))
    # 断言简化后的表达式等于原表达式
    assert expr.simplify() == expr


# 定义一个测试函数，用于测试 reduce_inverses_nc_pow
def test_reduce_inverses_nc_pow():
    # 创建符号 x, y, Z
    x, y = symbols("x y", commutative=True)
    Z = symbols("Z", commutative=False)
    # 断言简化表达式 2**Z * y**Z 的结果等于 2**Z * y**Z
    assert simplify(2**Z * y**Z) == 2**Z * y**Z
    # 断言简化表达式 x**Z * y**Z 的结果等于 x**Z * y**Z
    assert simplify(x**Z * y**Z) == x**Z * y**Z
    # 创建符号 x, y（正数）
    x, y = symbols("x y", positive=True)
    # 断言展开表达式 (x*y)**Z 的结果等于 x**Z * y**Z
    assert expand((x*y)**Z) == x**Z * y**Z
    # 断言简化表达式 x**Z * y**Z 的结果等于展开表达式 (x*y)**Z 的结果
    assert simplify(x**Z * y**Z) == expand((x*y)**Z)


# 定义一个测试函数，用于测试 nc_recursion_coeff
def test_nc_recursion_coeff():
    # 创建符号 X
    X = symbols("X", commutative=False)
    # 断言简化表达式 2 * cos(pi/3) * X 的结果等于 X
    assert (2 * cos(pi/3) * X).simplify() == X
    # 断言简化表达式 2.0 * cos(pi/3) * X 的结果等于 X
    assert (2.0 * cos(pi/
```