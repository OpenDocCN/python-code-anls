# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_radsimp.py`

```
# 导入 SymPy 中的各种类和函数，用于符号计算
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.series.order import O
from sympy.simplify.radsimp import (collect, collect_const, fraction, radsimp, rcollect)

# 导入 SymPy 中的特定类和函数，用于测试
from sympy.core.expr import unchanged
from sympy.core.mul import _unevaluated_Mul as umul
from sympy.simplify.radsimp import (_unevaluated_Add,
    collect_sqrt, fraction_expand, collect_abs)
from sympy.testing.pytest import raises

# 导入 SymPy 中的预定义符号
from sympy.abc import x, y, z, a, b, c, d

# 定义测试函数 test_radsimp
def test_radsimp():
    # 定义一些常用的平方根值
    r2 = sqrt(2)
    r3 = sqrt(3)
    r5 = sqrt(5)
    r7 = sqrt(7)

    # 测试简化根式的分数部分，确保计算正确
    assert fraction(radsimp(1/r2)) == (sqrt(2), 2)
    # 测试根式简化，1/(1 + sqrt(2)) 简化为 -1 + sqrt(2)
    assert radsimp(1/(1 + r2)) == \
        -1 + sqrt(2)
    # 测试根式简化，1/(sqrt(2) + sqrt(3)) 简化为 -sqrt(2) + sqrt(3)
    assert radsimp(1/(r2 + r3)) == \
        -sqrt(2) + sqrt(3)
    # 测试根式简化，1/(1 + sqrt(2) + sqrt(3)) 的分数部分应为 (-sqrt(6) + sqrt(2) + 2, 4)
    assert fraction(radsimp(1/(1 + r2 + r3))) == \
        (-sqrt(6) + sqrt(2) + 2, 4)
    # 测试根式简化，1/(sqrt(2) + sqrt(3) + sqrt(5)) 的分数部分应为 (-sqrt(30) + 2*sqrt(3) + 3*sqrt(2), 12)
    assert fraction(radsimp(1/(r2 + r3 + r5))) == \
        (-sqrt(30) + 2*sqrt(3) + 3*sqrt(2), 12)
    # 测试根式简化，1/(1 + sqrt(2) + sqrt(3) + sqrt(5)) 的分数部分应为 (-34*sqrt(10) - 26*sqrt(15) - 55*sqrt(3) - 61*sqrt(2) + 14*sqrt(30) + 93 + 46*sqrt(6) + 53*sqrt(5), 71)
    assert fraction(radsimp(1/(1 + r2 + r3 + r5))) == (
        (-34*sqrt(10) - 26*sqrt(15) - 55*sqrt(3) - 61*sqrt(2) + 14*sqrt(30) +
        93 + 46*sqrt(6) + 53*sqrt(5), 71))
    # 测试根式简化，1/(sqrt(2) + sqrt(3) + sqrt(5) + sqrt(7)) 的分数部分应为 (-50*sqrt(42) - 133*sqrt(5) - 34*sqrt(70) - 145*sqrt(3) + 22*sqrt(105) + 185*sqrt(2) + 62*sqrt(30) + 135*sqrt(7), 215)
    assert fraction(radsimp(1/(r2 + r3 + r5 + r7))) == (
        (-50*sqrt(42) - 133*sqrt(5) - 34*sqrt(70) - 145*sqrt(3) + 22*sqrt(105)
        + 185*sqrt(2) + 62*sqrt(30) + 135*sqrt(7), 215))
    # 对复杂表达式进行根式简化，并验证其长度为 16
    z = radsimp(1/(1 + r2/3 + r3/5 + r5 + r7))
    assert len((3616791619821680643598*z).args) == 16
    # 测试根式简化的逆操作，即 radsimp(1/z) 应该等于 1/z
    assert radsimp(1/z) == 1/z
    # 测试根式简化，限制最大项数为 20，期望展开结果为 1 + sqrt(2)/3 + sqrt(3)/5 + sqrt(5) + sqrt(7)
    assert radsimp(1/z, max_terms=20).expand() == 1 + r2/3 + r3/5 + r5 + r7
    # 测试根式简化，1/(sqrt(2)*3) 应该简化为 sqrt(2)/6
    assert radsimp(1/(r2*3)) == \
        sqrt(2)/6
    # 测试根式简化，1/(sqrt(2)*a + sqrt(3) + sqrt(5) + sqrt(7)) 的分数部分为复杂表达式
    assert radsimp(1/(r2*a + r3 + r5 + r7)) == (
        (8*sqrt(2)*a**7 - 8*sqrt(7)*a**6 - 8*sqrt(5)*a**6 - 8*sqrt(3)*a**6 -
        180*sqrt(2)*a**5 + 8*sqrt(30)*a**5 + 8*sqrt(42)*a**5 + 8*sqrt(70)*a**5
        - 24*sqrt(105)*a**4 + 84*sqrt(3)*a**4 + 100*sqrt(5)*a**4 +
        116*sqrt(7)*a**4 - 72*sqrt(70)*a**3 - 40*sqrt(42)*a**3 -
        8*sqrt(30)*a**3 + 782*sqrt(2)*a**3 - 462*sqrt(3)*a**2 -
        302*sqrt(7)*a**2 - 254*sqrt(5)*a**2 + 120*sqrt(105)*a**2 -
        795*sqrt(2)*a - 62*sqrt(30)*a + 82*sqrt(42)*a + 98*sqrt(70)*a -
        118*sqrt(105) + 59*sqrt(7) + 295*sqrt(5) + 531*sqrt(3))/(16*a**8 -
        480*a**6 + 3128*a**4 - 6360*a**2 + 3481))
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/(r2*a + r2*b + r3 + r7)) == (
        (sqrt(2)*a*(a + b)**2 - 5*sqrt(2)*a + sqrt(42)*a + sqrt(2)*b*(a +
        b)**2 - 5*sqrt(2)*b + sqrt(42)*b - sqrt(7)*(a + b)**2 - sqrt(3)*(a +
        b)**2 - 2*sqrt(3) + 2*sqrt(7))/(2*a**4 + 8*a**3*b + 12*a**2*b**2 -
        20*a**2 + 8*a*b**3 - 40*a*b + 2*b**4 - 20*b**2 + 8))
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/(r2*a + r2*b + r2*c + r2*d)) == \
        sqrt(2)/(2*a + 2*b + 2*c + 2*d)
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/(1 + r2*a + r2*b + r2*c + r2*d)) == (
        (sqrt(2)*a + sqrt(2)*b + sqrt(2)*c + sqrt(2)*d - 1)/(2*a**2 + 4*a*b +
        4*a*c + 4*a*d + 2*b**2 + 4*b*c + 4*b*d + 2*c**2 + 4*c*d + 2*d**2 - 1))
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp((y**2 - x)/(y - sqrt(x))) == \
        sqrt(x) + y
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(-(y**2 - x)/(y - sqrt(x))) == \
        -(sqrt(x) + y)
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/(1 - I + a*I)) == \
        (-I*a + 1 + I)/(a**2 - 2*a + 2)
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/((-x + y)*(x - sqrt(y)))) == \
        (-x - sqrt(y))/((x - y)*(x**2 - y))
    # 定义表达式 e
    e = (3 + 3*sqrt(2))*x*(3*x - 3*sqrt(y))
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(e) == x*(3 + 3*sqrt(2))*(3*x - 3*sqrt(y))
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/e) == (
        (-9*x + 9*sqrt(2)*x - 9*sqrt(y) + 9*sqrt(2)*sqrt(y))/(9*x*(9*x**2 -
        9*y)))
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1 + 1/(1 + sqrt(3))) == \
        Mul(S.Half, -1 + sqrt(3), evaluate=False) + 1
    # 定义符号 A，非可交换的
    A = symbols("A", commutative=False)
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(x**2 + sqrt(2)*x**2 - sqrt(2)*x*A) == \
        x**2 + sqrt(2)*x**2 - sqrt(2)*x*A
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/sqrt(5 + 2 * sqrt(6))) == -sqrt(2) + sqrt(3)
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/sqrt(5 + 2 * sqrt(6))**3) == -(-sqrt(3) + sqrt(2))**3

    # issue 6532
    # 断言：对表达式进行有理化简，并检查分子分母是否正确
    assert fraction(radsimp(1/sqrt(x))) == (sqrt(x), x)
    # 断言：对表达式进行有理化简，并检查分子分母是否正确
    assert fraction(radsimp(1/sqrt(2*x + 3))) == (sqrt(2*x + 3), 2*x + 3)
    # 断言：对表达式进行有理化简，并检查分子分母是否正确
    assert fraction(radsimp(1/sqrt(2*(x + 3)))) == (sqrt(2*x + 6), 2*x + 6)

    # issue 5994
    # 定义表达式 e
    e = S('-(2 + 2*sqrt(2) + 4*2**(1/4))/'
        '(1 + 2**(3/4) + 3*2**(1/4) + 3*sqrt(2))')
    # 断言：对表达式进行有理化简并展开，验证结果是否符合预期
    assert radsimp(e).expand() == -2*2**Rational(3, 4) - 2*2**Rational(1, 4) + 2 + 2*sqrt(2)

    # issue 5986 (modifications to radimp didn't initially recognize this so
    # the test is included here)
    # 断言：对表达式进行有理化简，验证结果是否符合预期
    assert radsimp(1/(-sqrt(5)/2 - S.Half + (-sqrt(5)/2 - S.Half)**2)) == 1

    # from issue 5934
    # 定义表达式 eq
    eq = (
        (-240*sqrt(2)*sqrt(sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) -
        360*sqrt(2)*sqrt(-8*sqrt(5) + 40)*sqrt(-sqrt(5) + 5) -
        120*sqrt(10)*sqrt(-8*sqrt(5) + 40)*sqrt(-sqrt(5) + 5) +
        120*sqrt(2)*sqrt(-sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) +
        120*sqrt(2)*sqrt(-8*sqrt(5) + 40)*sqrt(sqrt(5) + 5) +
        120*sqrt(10)*sqrt(-sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) +
        120*sqrt(10)*sqrt(-8*sqrt(5) + 40)*sqrt(sqrt(5) + 5))/(-36000 -
        7200*sqrt(5) + (12*sqrt(10)*sqrt(sqrt(5) + 5) +
        24*sqrt(10)*sqrt(-sqrt(5) + 5))**2))
    # 断言：对表达式进行有理化简，验证结果是否为 NaN（0/0）
    assert radsimp(eq) is S.NaN  # it's 0/0

    # work with normal form
    # 定义表达式 e
    e = 1/sqrt(sqrt(7)/7 + 2*sqrt(2) + 3*sqrt(3) + 5*sqrt(5)) + 3
    # 使用断言来验证 radsimp(e) 的计算结果是否等于给定的表达式
    assert radsimp(e) == (
        -sqrt(sqrt(7) + 14*sqrt(2) + 21*sqrt(3) +
        35*sqrt(5))*(-11654899*sqrt(35) - 1577436*sqrt(210) - 1278438*sqrt(15)
        - 1346996*sqrt(10) + 1635060*sqrt(6) + 5709765 + 7539830*sqrt(14) +
        8291415*sqrt(21))/1300423175 + 3)

    # 对幂运算规则进行验证
    base = sqrt(3) - sqrt(2)
    assert radsimp(1/base**3) == (sqrt(3) + sqrt(2))**3
    assert radsimp(1/(-base)**3) == -(sqrt(2) + sqrt(3))**3
    assert radsimp(1/(-base)**x) == (-base)**(-x)
    assert radsimp(1/base**x) == (sqrt(2) + sqrt(3))**x
    assert radsimp(root(1/(-1 - sqrt(2)), -x)) == (-1)**(-1/x)*(1 + sqrt(2))**(1/x)

    # 递归测试
    e = cos(1/(1 + sqrt(2)))
    assert radsimp(e) == cos(-sqrt(2) + 1)
    assert radsimp(e/2) == cos(-sqrt(2) + 1)/2
    assert radsimp(1/e) == 1/cos(-sqrt(2) + 1)
    assert radsimp(2/e) == 2/cos(-sqrt(2) + 1)
    assert fraction(radsimp(e/sqrt(x))) == (sqrt(x)*cos(-sqrt(2)+1), x)

    # 测试不处理符号分母的情况
    r = 1 + sqrt(2)
    assert radsimp(x/r, symbolic=False) == -x*(-sqrt(2) + 1)
    assert radsimp(x/(y + r), symbolic=False) == x/(y + 1 + sqrt(2))
    assert radsimp(x/(y + r)/r, symbolic=False) == \
        -x*(-sqrt(2) + 1)/(y + 1 + sqrt(2))

    # 处理问题编号 7408
    eq = sqrt(x)/sqrt(y)
    assert radsimp(eq) == umul(sqrt(x), sqrt(y), 1/y)
    assert radsimp(eq, symbolic=False) == eq

    # 处理问题编号 7498
    assert radsimp(sqrt(x)/sqrt(y)**3) == umul(sqrt(x), sqrt(y**3), 1/y**3)

    # 用于覆盖率测试
    eq = sqrt(x)/y**2
    assert radsimp(eq) == eq

    # 处理非 Expr 参数
    from sympy.integrals.integrals import Integral
    eq = Integral(x/(sqrt(2) - 1), (x, 0, 1/(sqrt(2) + 1)))
    assert radsimp(eq) == Integral((sqrt(2) + 1)*x , (x, 0, sqrt(2) - 1))

    from sympy.sets import FiniteSet
    eq = FiniteSet(x/(sqrt(2) - 1))
    assert radsimp(eq) == FiniteSet((sqrt(2) + 1)*x)
def test_radsimp_issue_3214():
    # 定义符号 c 和 p，要求它们是正数
    c, p = symbols('c p', positive=True)
    # 计算表达式 s = sqrt(c**2 - p**2)，其中 sqrt 是符号库中的平方根函数
    s = sqrt(c**2 - p**2)
    # 计算表达式 b = (c + I*p - s)/(c + I*p + s)，其中 I 是虚数单位
    b = (c + I*p - s)/(c + I*p + s)
    # 断言简化 b 后的结果等于预期值
    assert radsimp(b) == -I*(c + I*p - sqrt(c**2 - p**2))**2/(2*c*p)


def test_collect_1():
    """Collect with respect to Symbol"""
    # 定义符号 x, y, z, n
    x, y, z, n = symbols('x,y,z,n')
    # 断言 collect(1, x) 的结果为 1
    assert collect(1, x) == 1
    # 断言 collect(x + y*x, x) 的结果为 x * (1 + y)
    assert collect(x + y*x, x) == x * (1 + y)
    # 断言 collect(x + x**2, x) 的结果为 x + x**2
    assert collect(x + x**2, x) == x + x**2
    # 断言 collect(x**2 + y*x**2, x) 的结果为 (x**2)*(1 + y)
    assert collect(x**2 + y*x**2, x) == (x**2)*(1 + y)
    # 断言 collect(x**2 + y*x, x) 的结果为 x*y + x**2
    assert collect(x**2 + y*x, x) == x*y + x**2
    # 断言 collect(2*x**2 + y*x**2 + 3*x*y, [x]) 的结果为 x**2*(2 + y) + 3*x*y
    assert collect(2*x**2 + y*x**2 + 3*x*y, [x]) == x**2*(2 + y) + 3*x*y
    # 断言 collect(2*x**2 + y*x**2 + 3*x*y, [y]) 的结果为 2*x**2 + y*(x**2 + 3*x)
    assert collect(2*x**2 + y*x**2 + 3*x*y, [y]) == 2*x**2 + y*(x**2 + 3*x)

    # 断言 collect(((1 + y + x)**4).expand(), x) 的结果为展开后的多项式
    assert collect(((1 + y + x)**4).expand(), x) == ((1 + y)**4).expand() + \
        x*(4*(1 + y)**3).expand() + x**2*(6*(1 + y)**2).expand() + \
        x**3*(4*(1 + y)).expand() + x**4
    # 断言 collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None) 的结果为 x*exp(x) + 3*x + (y + 2)*sin(x)
    assert collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None) == x*exp(x) + 3*x + (y + 2)*sin(x)
    # 断言 collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x + y*x + y*x*exp(x), x, exact=None) 的结果为 x*exp(x)*(y + 1) + (3 + y)*x + (y + 2)*sin(x)
    assert collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x + y*x +
        y*x*exp(x), x, exact=None) == x*exp(x)*(y + 1) + (3 + y)*x + (y + 2)*sin(x)


def test_collect_2():
    """Collect with respect to a sum"""
    # 定义符号 a, b, x
    a, b, x = symbols('a,b,x')
    # 断言 collect(a*(cos(x) + sin(x)) + b*(cos(x) + sin(x)), sin(x) + cos(x)) 的结果为 (a + b)*(cos(x) + sin(x))


def test_collect_3():
    """Collect with respect to a product"""
    # 定义符号 a, b, c, f, x, y, z, n
    a, b, c = symbols('a,b,c')
    f = Function('f')
    x, y, z, n = symbols('x,y,z,n')

    # 断言 collect(-x/8 + x*y, -x) 的结果为 x*(y - Rational(1, 8))
    assert collect(-x/8 + x*y, -x) == x*(y - Rational(1, 8))

    # 断言 collect(1 + x*(y**2), x*y) 的结果为 1 + x*(y**2)
    assert collect(1 + x*(y**2), x*y) == 1 + x*(y**2)
    # 断言 collect(x*y + a*x*y, x*y) 的结果为 x*y*(1 + a)
    assert collect(x*y + a*x*y, x*y) == x*y*(1 + a)
    # 断言 collect(1 + x*y + a*x*y, x*y) 的结果为 1 + x*y*(1 + a)
    assert collect(1 + x*y + a*x*y, x*y) == 1 + x*y*(1 + a)
    # 断言 collect(a*x*f(x) + b*(x*f(x)), x*f(x)) 的结果为 x*(a + b)*f(x)
    assert collect(a*x*f(x) + b*(x*f(x)), x*f(x)) == x*(a + b)*f(x)

    # 断言 collect(a*x*log(x) + b*(x*log(x)), x*log(x)) 的结果为 x*(a + b)*log(x)
    assert collect(a*x*log(x) + b*(x*log(x)), x*log(x)) == x*(a + b)*log(x)
    # 断言 collect(a*x**2*log(x)**2 + b*(x*log(x))**2, x*log(x)) 的结果为 x**2*log(x)**2*(a + b)
    assert collect(a*x**2*log(x)**2 + b*(x*log(x))**2, x*log(x)) == x**2*log(x)**2*(a + b)

    # 断言 collect(y*x*z + a*x*y*z, x*y*z) 的结果为 (1 + a)*x*y*z


def test_collect_4():
    """Collect with respect to a power"""
    # 定义符号 a, b, c, x
    a, b, c, x = symbols('a,b,c,x')

    # 断言 collect(a*x**c + b*x**c, x**c) 的结果为 x**c*(a + b)
    assert collect(a*x**c + b*x**c, x**c) == x**c*(a + b)
    # 断言 collect(a*x**(2*c) + b*x**(2*c), x**c) 的结果为 x**(2*c)*(a + b)


def test_collect_5():
    """Collect with respect to a tuple"""
    # 定义符号 a, x, y, z, n
    a, x, y, z, n = symbols('a,x,y,z,n')

    # 断言 collect(x**2*y**4 + z*(x*y**2)**2 + z + a*z, [x*y**2, z]) 的结果为两种可能之一
    assert collect(x**2*y**4 + z*(x*y**2)**2 + z + a*z, [x*y**2, z]) in [
        z*(1 + a + x**2*y**4) + x**2*y**4,
        z*(1 + a) + x**2*y**4*(1 + z) ]
    # 断言 collect((1 + (x + y) + (x + y)**2).expand(), [x, y]) 的结果为展开后的多项式
    assert collect((1 + (x + y) + (x + y)**2).expand(),
                   [x, y]) == 1 + y + x*(1 + 2*y) + x**2 + y**2


def test_collect_pr19431():
    """Unevaluated collect with respect to a product"""
    # 创建一个符号变量 'a'
    a = symbols('a')
    
    # 断言语句，验证 collect 函数的使用效果
    # collect 函数用于收集表达式中给定项的系数，这里的参数为 a**2*(a**2 + 1)，收集 a**2 的系数
    # evaluate=False 表示不进行表达式求值
    # 最终断言，收集后的结果应当等于 (a**2 + 1)
    assert collect(a**2*(a**2 + 1), a**2, evaluate=False)[a**2] == (a**2 + 1)
# 定义测试函数 test_collect_D，用于测试 collect 函数在求导数表达式上的功能
def test_collect_D():
    # 导数对象的引用
    D = Derivative
    # 定义一个未知函数 f(x)
    f = Function('f')
    # 符号变量 x, a, b 的声明
    x, a, b = symbols('x,a,b')
    # 计算 f(x) 对 x 的一阶导数
    fx = D(f(x), x)
    # 计算 f(x) 对 x 的二阶导数
    fxx = D(f(x), x, x)

    # 测试收集 a*fx + b*fx 中关于 fx 的项
    assert collect(a*fx + b*fx, fx) == (a + b)*fx
    # 测试收集 a*D(fx, x) + b*D(fx, x) 中关于 fx 的项
    assert collect(a*D(fx, x) + b*D(fx, x), fx) == (a + b)*D(fx, x)
    # 测试收集 a*fxx + b*fxx 中关于 fx 的项（注意此处应为 fxx，而不是 fx）
    assert collect(a*fxx + b*fxx, fx) == (a + b)*D(fx, x)
    # 测试问题编号 4784 的情况
    assert collect(5*f(x) + 3*fx, fx) == 5*f(x) + 3*fx
    # 测试复杂表达式的收集
    assert collect(f(x) + f(x)*diff(f(x), x) + x*diff(f(x), x)*f(x), f(x).diff(x)) == \
        (x*f(x) + f(x))*D(f(x), x) + f(x)
    # 测试精确收集
    assert collect(f(x) + f(x)*diff(f(x), x) + x*diff(f(x), x)*f(x), f(x).diff(x), exact=True) == \
        (x*f(x) + f(x))*D(f(x), x) + f(x)
    # 测试包含除法的收集
    assert collect(1/f(x) + 1/f(x)*diff(f(x), x) + x*diff(f(x), x)/f(x), f(x).diff(x), exact=True) == \
        (1/f(x) + x/f(x))*D(f(x), x) + 1/f(x)
    # 测试复杂表达式的展开后的收集
    e = (1 + x*fx + fx)/f(x)
    assert collect(e.expand(), fx) == fx*(x/f(x) + 1/f(x)) + 1/f(x)


# 定义测试函数 test_collect_func，用于测试 collect 函数在函数表达式上的功能
def test_collect_func():
    # 定义一个复杂表达式 f
    f = ((x + a + 1)**3).expand()

    # 测试默认收集
    assert collect(f, x) == a**3 + 3*a**2 + 3*a + x**3 + x**2*(3*a + 3) + \
        x*(3*a**2 + 6*a + 3) + 1
    # 测试使用因子进行收集
    assert collect(f, x, factor) == x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + \
        (a + 1)**3

    # 测试不进行评估的收集
    assert collect(f, x, evaluate=False) == {
        S.One: a**3 + 3*a**2 + 3*a + 1,
        x: 3*a**2 + 6*a + 3, x**2: 3*a + 3,
        x**3: 1
    }

    # 测试不进行评估的因子收集
    assert collect(f, x, factor, evaluate=False) == {
        S.One: (a + 1)**3, x: 3*(a + 1)**2,
        x**2: umul(S(3), a + 1), x**3: 1}


# 定义测试函数 test_collect_order，用于测试 collect 函数在多项式和级数上的收集功能
def test_collect_order():
    # 符号变量的声明
    a, b, x, t = symbols('a,b,x,t')

    # 测试在级数中收集 t 的项
    assert collect(t + t*x + t*x**2 + O(x**3), t) == t*(1 + x + x**2 + O(x**3))
    # 测试在级数中收集 t 的项（带更复杂的多项式）
    assert collect(t + t*x + x**2 + O(x**3), t) == \
        t*(1 + x + O(x**3)) + x**2 + O(x**3)

    # 测试在多项式中收集 x 的项
    f = a*x + b*x + c*x**2 + d*x**2 + O(x**3)
    g = x*(a + b) + x**2*(c + d) + O(x**3)

    assert collect(f, x) == g
    assert collect(f, x, distribute_order_term=False) == g

    # 测试在级数中使用三角函数的收集
    f = sin(a + b).series(b, 0, 10)

    assert collect(f, [sin(a), cos(a)]) == \
        sin(a)*cos(b).series(b, 0, 10) + cos(a)*sin(b).series(b, 0, 10)
    assert collect(f, [sin(a), cos(a)], distribute_order_term=False) == \
        sin(a)*cos(b).series(b, 0, 10).removeO() + \
        cos(a)*sin(b).series(b, 0, 10).removeO() + O(b**10)


# 定义测试函数 test_rcollect，用于测试 rcollect 函数的反向收集功能
def test_rcollect():
    # 测试对 (x**2*y + x*y + x + y)/(x + y) 进行关于 y 的反向收集
    assert rcollect((x**2*y + x*y + x + y)/(x + y), y) == \
        (x + y*(1 + x + x**2))/(x + y)
    # 测试对 sqrt(-((x + 1)*(y + 1))) 进行关于 z 的反向收集
    assert rcollect(sqrt(-((x + 1)*(y + 1))), z) == sqrt(-((x + 1)*(y + 1)))


# 定义测试函数 test_collect_D_0，用于测试 collect 函数在导数表达式上的特殊情况
def test_collect_D_0():
    # 导数对象的引用
    D = Derivative
    # 定义一个未知函数 f(x)
    f = Function('f')
    # 符号变量 x, a, b 的声明
    x, a, b = symbols('x,a,b')
    # 计算 f(x) 对 x 的二阶导数
    fxx = D(f(x), x, x)

    # 测试收集 a*fxx + b*fxx 中关于 fxx 的项
    assert collect(a*fxx + b*fxx, fxx) == (a + b)*fxx


# 定义测试函数 test_collect_Wild，用于测试 collect 函数在带有 Wild 参数的函数上的收集功能
def test_collect_Wild():
    # 符号变量的声明
    a, b, x, y = symbols('a b x y')
    # 定义一个未知函数 f(x)
    f = Function('f')
    # 通配符 Wild 的引入
    w1 = Wild('.1')
    w2 = Wild('.2')
    
    # 测试带 Wild 参数的函数 f(x) + a*f(x) 中关于 f(w1) 的项的收集
    assert collect(f(x) + a*f(x), f(w1)) == (1 + a)*f(x)
    # 测试带 Wild 参数的函数 f(x, y) + a*f(x, y) 中关于 f(w1) 的项的收集
    assert collect(f(x, y) + a*f(x, y), f(w1)) == f(x, y) + a*f(x, y)
    # 测试带 Wild 参数的函数 f(x, y) + a*f(x, y) 中关于 f(w1, w2) 的项的收集
    assert collect(f(x, y) + a*f(x, y), f(w1, w2)) == (1 + a)*f(x, y)
    # 断言语句：验证 collect 函数对给定表达式进行整理后的结果是否符合预期
    
    assert collect(f(x, y) + a*f(x, y), f(w1, w1)) == f(x, y) + a*f(x, y)
    # 验证对表达式 f(x, y) + a*f(x, y) 应用 collect 函数后，结果应为 f(x, y) + a*f(x, y)
    
    assert collect(f(x, x) + a*f(x, x), f(w1, w1)) == (1 + a)*f(x, x)
    # 验证对表达式 f(x, x) + a*f(x, x) 应用 collect 函数后，结果应为 (1 + a)*f(x, x)
    
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**y) == (1 + a)*(x + 1)**y
    # 验证对表达式 a*(x + 1)**y + (x + 1)**y 应用 collect 函数后，结果应为 (1 + a)*(x + 1)**y
    
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**b) == \
        a*(x + 1)**y + (x + 1)**y
    # 验证对表达式 a*(x + 1)**y + (x + 1)**y 应用 collect 函数后，结果应为 a*(x + 1)**y + (x + 1)**y
    
    assert collect(a*(x + 1)**y + (x + 1)**y, (x + 1)**w2) == \
        (1 + a)*(x + 1)**y
    # 验证对表达式 a*(x + 1)**y + (x + 1)**y 应用 collect 函数后，结果应为 (1 + a)*(x + 1)**y
    
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**w2) == (1 + a)*(x + 1)**y
    # 验证对表达式 a*(x + 1)**y + (x + 1)**y 应用 collect 函数后，结果应为 (1 + a)*(x + 1)**y
def test_collect_const():
    # 覆盖之前测试未提供的情况
    assert collect_const(2*sqrt(3) + 4*a*sqrt(5)) == \
        2*(2*sqrt(5)*a + sqrt(3))  # 让原始常数重新吸收
    assert collect_const(2*sqrt(3) + 4*a*sqrt(5), sqrt(3)) == \
        2*sqrt(3) + 4*a*sqrt(5)
    assert collect_const(sqrt(2)*(1 + sqrt(2)) + sqrt(3) + x*sqrt(2)) == \
        sqrt(2)*(x + 1 + sqrt(2)) + sqrt(3)

    # 问题 5290
    assert collect_const(2*x + 2*y + 1, 2) == \
        collect_const(2*x + 2*y + 1) == \
        Add(S.One, Mul(2, x + y, evaluate=False), evaluate=False)
    assert collect_const(-y - z) == Mul(-1, y + z, evaluate=False)
    assert collect_const(2*x - 2*y - 2*z, 2) == \
        Mul(2, x - y - z, evaluate=False)
    assert collect_const(2*x - 2*y - 2*z, -2) == \
        _unevaluated_Add(2*x, Mul(-2, y + z, evaluate=False))

    # 这是为什么使用 content_primitive
    eq = (sqrt(15 + 5*sqrt(2))*x + sqrt(3 + sqrt(2))*y)*2
    assert collect_sqrt(eq + 2) == \
        2*sqrt(sqrt(2) + 3)*(sqrt(5)*x + y) + 2

    # 问题 16296
    assert collect_const(a + b + x/2 + y/2) == a + b + Mul(S.Half, x + y, evaluate=False)


def test_issue_13143():
    f = Function('f')
    fx = f(x).diff(x)
    e = f(x) + fx + f(x)*fx
    # 在导数之前收集函数
    assert collect(e, Wild('w')) == f(x)*(fx + 1) + fx
    e = f(x) + f(x)*fx + x*fx*f(x)
    assert collect(e, fx) == (x*f(x) + f(x))*fx + f(x)
    assert collect(e, f(x)) == (x*fx + fx + 1)*f(x)
    e = f(x) + fx + f(x)*fx
    assert collect(e, [f(x), fx]) == f(x)*(1 + fx) + fx
    assert collect(e, [fx, f(x)]) == fx*(1 + f(x)) + f(x)


def test_issue_6097():
    assert collect(a*y**(2.0*x) + b*y**(2.0*x), y**x) == (a + b)*(y**x)**2.0
    assert collect(a*2**(2.0*x) + b*2**(2.0*x), 2**x) == (a + b)*(2**x)**2.0


def test_fraction_expand():
    eq = (x + y)*y/x
    assert eq.expand(frac=True) == fraction_expand(eq) == (x*y + y**2)/x
    assert eq.expand() == y + y**2/x


def test_fraction():
    x, y, z = map(Symbol, 'xyz')
    A = Symbol('A', commutative=False)

    assert fraction(S.Half) == (1, 2)

    assert fraction(x) == (x, 1)
    assert fraction(1/x) == (1, x)
    assert fraction(x/y) == (x, y)
    assert fraction(x/2) == (x, 2)

    assert fraction(x*y/z) == (x*y, z)
    assert fraction(x/(y*z)) == (x, y*z)

    assert fraction(1/y**2) == (1, y**2)
    assert fraction(x/y**2) == (x, y**2)

    assert fraction((x**2 + 1)/y) == (x**2 + 1, y)
    assert fraction(x*(y + 1)/y**7) == (x*(y + 1), y**7)

    assert fraction(exp(-x), exact=True) == (exp(-x), 1)
    assert fraction((1/(x + y))/2, exact=True) == (1, Mul(2,(x + y), evaluate=False))

    assert fraction(x*A/y) == (x*A, y)
    assert fraction(x*A**-1/y) == (x*A**-1, y)

    n = symbols('n', negative=True)
    assert fraction(exp(n)) == (1, exp(-n))
    assert fraction(exp(-n)) == (exp(-n), 1)

    p = symbols('p', positive=True)
    assert fraction(exp(-p)*log(p), exact=True) == (exp(-p)*log(p), 1)
    # 创建一个 Mul 对象 m，它表示多个因子的乘积，并设置不进行求值
    m = Mul(1, 1, S.Half, evaluate=False)
    
    # 断言 m 的分数表示为 (1, 2)
    assert fraction(m) == (1, 2)
    
    # 断言以精确模式计算 m 的分数，应返回 (Mul(1, 1, evaluate=False), 2)
    assert fraction(m, exact=True) == (Mul(1, 1, evaluate=False), 2)
    
    # 创建一个 Mul 对象 m，包含多个因子并设置不进行求值
    m = Mul(1, 1, S.Half, S.Half, Pow(1, -1, evaluate=False), evaluate=False)
    
    # 断言 m 的分数表示为 (1, 4)
    assert fraction(m) == (1, 4)
    
    # 断言以精确模式计算 m 的分数，应返回 (Mul(1, 1, evaluate=False), Mul(2, 2, 1, evaluate=False))
    assert fraction(m, exact=True) == \
            (Mul(1, 1, evaluate=False), Mul(2, 2, 1, evaluate=False))
# 测试函数，验证 GitHub 上的问题 #5615 的解决方案
def test_issue_5615():
    # 符号定义
    aA, Re, a, b, D = symbols('aA Re a b D')
    # 表达式计算
    e = ((D**3*a + b*aA**3)/Re).expand()
    # 断言表达式的收集结果与原表达式相等
    assert collect(e, [aA**3/Re, a]) == e


# 测试函数，验证 GitHub 上的问题 #5933 的解决方案
def test_issue_5933():
    # 导入多边形和正多边形相关的模块
    from sympy.geometry.polygon import (Polygon, RegularPolygon)
    # 导入分母函数
    from sympy.simplify.radsimp import denom
    # 计算正五边形的重心的 x 坐标
    x = Polygon(*RegularPolygon((0, 0), 1, 5).vertices).centroid.x
    # 断言重心 x 坐标的分母大于 1e-12
    assert abs(denom(x).n()) > 1e-12
    # 断言简化后的重心 x 坐标的分母大于 1e-12，以防简化未能处理它
    assert abs(denom(radsimp(x))) > 1e-12


# 测试函数，验证 GitHub 上的问题 #14608 的解决方案
def test_issue_14608():
    # 符号定义，指定为非交换
    a, b = symbols('a b', commutative=False)
    x, y = symbols('x y')
    # 使用 lambda 表达式来测试是否引发 AttributeError
    raises(AttributeError, lambda: collect(a*b + b*a, a))
    # 断言对于给定表达式 x*y + y*(x+1)，按照 a 进行收集操作结果不变
    assert collect(x*y + y*(x+1), a) == x*y + y*(x+1)
    # 断言对于给定表达式 x*y + y*(x+1) + a*b + b*a，按照 y 进行收集操作
    # 结果应为 y*(2*x + 1) + a*b + b*a
    assert collect(x*y + y*(x+1) + a*b + b*a, y) == y*(2*x + 1) + a*b + b*a


# 测试函数，验证 collect_abs 函数的行为
def test_collect_abs():
    # 绝对值表达式
    s = abs(x) + abs(y)
    # 断言对绝对值表达式应用 collect_abs 后结果不变
    assert collect_abs(s) == s
    # 断言 Mul 类型的绝对值表达式 abs(x)*abs(y) 的类型不变
    assert unchanged(Mul, abs(x), abs(y))
    # 断言 abs(x*y) 的收集结果为 Abs(x*y)
    ans = Abs(x*y)
    assert isinstance(ans, Abs)
    assert collect_abs(abs(x)*abs(y)) == ans
    # 断言对于表达式 1 + exp(abs(x)*abs(y))，应用 collect_abs 后结果为 1 + exp(Abs(x*y))
    assert collect_abs(1 + exp(abs(x)*abs(y))) == 1 + exp(ans)

    # 查看 GitHub 上的问题 #12910
    p = Symbol('p', positive=True)
    # 断言对于 p/abs(1-p) 的收集结果是可交换的
    assert collect_abs(p/abs(1-p)).is_commutative is True


# 测试函数，验证 GitHub 上的问题 #19149 的解决方案
def test_issue_19149():
    # 指数表达式
    eq = exp(3*x/4)
    # 断言对于指数表达式，按照 exp(x) 进行收集操作后结果不变
    assert collect(eq, exp(x)) == eq


# 测试函数，验证 GitHub 上的问题 #19719 的解决方案
def test_issue_19719():
    # 符号定义
    a, b = symbols('a, b')
    # 表达式定义
    expr = a**2 * (b + 1) + (7 + 1/b)/a
    # 按照 a**2 和 1/a 进行收集操作，但不进行评估
    collected = collect(expr, (a**2, 1/a), evaluate=False)
    # 断言收集操作的结果为 {a**2: b + 1, 1/a: 7 + 1/b}，使用 xreplace 不进行
    # 评估
    assert collected == {a**2: b + 1, 1/a: 7 + 1/b}


# 测试函数，验证 GitHub 上的问题 #21355 的解决方案
def test_issue_21355():
    # 断言对于给定的表达式 1/(x + sqrt(x**2))，应用 radsimp 后结果不变
    assert radsimp(1/(x + sqrt(x**2))) == 1/(x + sqrt(x**2))
    # 断言对于给定的表达式 1/(x - sqrt(x**2))，应用 radsimp 后结果不变
    assert radsimp(1/(x - sqrt(x**2))) == 1/(x - sqrt(x**2))
```