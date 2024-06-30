# `D:\src\scipysrc\sympy\sympy\core\tests\test_subs.py`

```
# 导入必要的 SymPy 模块中的具体类和函数
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.core.sympify import SympifyError
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, cot, sin, tan)
from sympy.matrices.dense import (Matrix, zeros)
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import RootOf
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import nsimplify
from sympy.core.basic import _aresame
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import a, x, y, z, t


# 定义测试函数 test_subs，用于测试 SymPy 中的符号代换功能
def test_subs():
    # 创建有理数 3
    n3 = Rational(3)
    # 创建符号 x，并将其赋给变量 e
    e = x
    # 对 e 进行符号代换，将 x 替换为 n3
    e = e.subs(x, n3)
    # 断言 e 的值为有理数 3
    assert e == Rational(3)

    # 将 e 设为 2*x
    e = 2*x
    # 断言 e 的表达式未改变
    assert e == 2*x
    # 对 e 进行符号代换，将 x 替换为 n3
    e = e.subs(x, n3)
    # 断言 e 的值为有理数 6
    assert e == Rational(6)


# 定义测试函数 test_subs_Matrix，用于测试 SymPy 中的矩阵符号代换功能
def test_subs_Matrix():
    # 创建一个 2x2 的零矩阵 z
    z = zeros(2)
    # 创建另一个 2x2 的零矩阵 z1
    z1 = ZeroMatrix(2, 2)
    # 断言 (x*y) 在将 x 替换为 z，y 替换为 0 后，结果为 z 或 z1
    assert (x*y).subs({x:z, y:0}) in [z, z1]
    # 断言 (x*y) 在将 y 替换为 z，x 替换为 0 后，结果为 0
    assert (x*y).subs({y:z, x:0}) == 0
    # 断言 (x*y) 在同时将 y 替换为 z，x 替换为 0 后，结果为 z 或 z1
    assert (x*y).subs({y:z, x:0}, simultaneous=True) in [z, z1]
    # 断言 (x + y) 在同时将 x 和 y 替换为 z 后，结果为 z 或 z1
    assert (x + y).subs({x: z, y: z}, simultaneous=True) in [z, z1]
    # 断言 (x + y) 在将 x 和 y 替换为 z 后，结果为 z 或 z1
    assert (x + y).subs({x: z, y: z}) in [z, z1]

    # Issue #15528 的测试
    # 断言 Mul(Matrix([[3]]), x) 在将 x 替换为 2.0 后，结果为 Matrix([[6.0]])
    assert Mul(Matrix([[3]]), x).subs(x, 2.0) == Matrix([[6.0]])
    # 断言 Add(Matrix([[3]]), x) 在将 x 替换为 2.0 后，结果为 Add(Matrix([[3]]), 2.0)
    # 这里不会引发 TypeError，参见 MatAdd 后处理器的注释
    assert Add(Matrix([[3]]), x).subs(x, 2.0) == Add(Matrix([[3]]), 2.0)


# 定义测试函数 test_subs_AccumBounds，用于测试 AccumBounds 类的符号代换功能
def test_subs_AccumBounds():
    # 创建符号表达式 e，并将其赋为 x
    e = x
    # 对 e 进行符号代换，将 x 替换为 AccumBounds(1, 3)
    e = e.subs(x, AccumBounds(1, 3))
    # 断言 e 的值为 AccumBounds(1, 3)
    assert e == AccumBounds(1, 3)

    # 将 e 设为 2*x
    e = 2*x
    # 对 e 进行符号代换，将 x 替换为 AccumBounds(1, 3)
    e = e.subs(x, AccumBounds(1, 3))
    # 断言 e 的值为 AccumBounds(2, 6)
    assert e == AccumBounds(2, 6)

    # 将 e 设为 x + x**2
    e = x + x**2
    # 对 e 进行符号代换，将 x 替换为 AccumBounds(-1, 1)
    e = e.subs(x, AccumBounds(-1, 1))
    # 断言 e 的值为 AccumBounds(-1, 2)
    assert e == AccumBounds(-1, 2)


# 定义测试函数 test_trigonometric，用于测试 SymPy 中三角函数的符号代换功能
def test_trigonometric():
    # 创建有理数 3
    n3 = Rational(3)
    # 对 sin(x)**2 关于 x 的导数，并将其赋给 e
    e = (sin(x)**2).diff(x)
    # 断言 e 的导数表达式为 2*sin(x)*cos(x)
    assert e == 2*sin(x)*cos(x)
    # 对 e 进行符号代换，将 x 替换为 n3
    e = e.subs(x, n3)
    # 断言 e 的值为 2*cos(3)*sin(3)
    assert e == 2*cos(n3)*sin(n3)

    # 将 e 设为 (sin(x)**2).diff(x)
    e = (sin(x)**2).diff(x)
    # 断言 e 的导数表达式为 2*sin(x)*cos(x)
    assert e == 2*sin(x)*cos(x)
    # 对 e 进行符号代换，将 sin(x) 替换为 cos(x)
    e = e.subs(sin(x), cos(x))
    # 断言 e 的值为 2*cos(x)**2
    assert e == 2*cos(x)**2

    # 断言 exp(pi) 在将 exp 替换为 sin 后，结果为 0
    assert exp(pi).subs(exp, sin) == 0
    # 断言 cos(exp(pi)) 在将 exp 替换为 sin 后，结果为 1
    assert cos(exp(pi)).subs(exp, sin) == 1

    # 创建整数符号 i
    i = Symbol('i', integer=True)
    # 创建复数无穷大 zoo
    zoo = S.ComplexInfinity
    # 断言 tan(x) 在将 x 替换为 pi/2 后，结果为 zoo
    assert tan(x).subs(x, pi/2) is zoo
    # 断言 cot(x) 在将 x 替换为 pi 后，结果为 zoo
    assert cot(x).subs(x, pi) is zoo
    # 断言 cot(i*x) 在将 x 替换为 pi 后，结果为 zoo
    assert cot(i*x).subs(x, pi) is zoo
    # 断言 tan(i*x) 在将 x 替换为 pi/2 后，结果为 tan(i*pi/2)
    assert tan(i*x).subs(x, pi/2) == tan(i*pi/2)
    # 断言 tan(i*x) 在将 x 替换为 pi/2，i 替换为 1 后，结果为 zoo
    assert tan(i*x).subs(x, pi/2).subs(i, 1) is zoo
    # 创建奇数符号 o
    o = Symbol('o', odd=True)
    # 断
    # 断言，验证 sqrt(1 - sqrt(x)).subs(x, 4) 是否等于复数单位 I
    assert sqrt(1 - sqrt(x)).subs(x, 4) == I
    
    # 断言，验证 (sqrt(1 - x**2)**3).subs(x, 2) 是否等于 -3*I*sqrt(3)
    assert (sqrt(1 - x**2)**3).subs(x, 2) == - 3*I*sqrt(3)
    
    # 断言，验证 (x**Rational(1, 3)).subs(x, 27) 是否等于 3
    assert (x**Rational(1, 3)).subs(x, 27) == 3
    
    # 断言，验证 (x**Rational(1, 3)).subs(x, -27) 是否等于 3*(-1)**Rational(1, 3)
    assert (x**Rational(1, 3)).subs(x, -27) == 3*(-1)**Rational(1, 3)
    
    # 断言，验证 ((-x)**Rational(1, 3)).subs(x, 27) 是否等于 3*(-1)**Rational(1, 3)
    assert ((-x)**Rational(1, 3)).subs(x, 27) == 3*(-1)**Rational(1, 3)
    
    # 创建一个负数符号对象 n
    n = Symbol('n', negative=True)
    
    # 断言，验证 (x**n).subs(x, 0) 是否等于 S.ComplexInfinity
    assert (x**n).subs(x, 0) is S.ComplexInfinity
    
    # 断言，验证 exp(-1).subs(S.Exp1, 0) 是否等于 S.ComplexInfinity
    assert exp(-1).subs(S.Exp1, 0) is S.ComplexInfinity
    
    # 断言，验证 (x**(4.0*y)).subs(x**(2.0*y), n) 是否等于 n**2.0
    assert (x**(4.0*y)).subs(x**(2.0*y), n) == n**2.0
    
    # 断言，验证 (2**(x + 2)).subs(2, 3) 是否等于 3**(x + 3)
    assert (2**(x + 2)).subs(2, 3) == 3**(x + 3)
def test_logexppow():   # 定义函数 test_logexppow，用于测试指数和幂运算的替换
    x = Symbol('x', real=True)  # 创建实数符号变量 x
    w = Symbol('w')  # 创建符号变量 w
    e = (3**(1 + x) + 2**(1 + x))/(3**x + 2**x)  # 定义一个复杂的表达式 e
    assert e.subs(2**x, w) != e  # 断言：用 w 替换 2**x 后的 e 不等于原始 e
    assert e.subs(exp(x*log(Rational(2))), w) != e  # 断言：用 w 替换 exp(x*log(2)) 后的 e 不等于原始 e


def test_bug():  # 定义函数 test_bug，用于测试符号替换的精度
    x1 = Symbol('x1')  # 创建符号变量 x1
    x2 = Symbol('x2')  # 创建符号变量 x2
    y = x1*x2  # 定义变量 y 为 x1 和 x2 的乘积
    assert y.subs(x1, Float(3.0)) == Float(3.0)*x2  # 断言：用 3.0 替换 x1 后的 y 等于 3.0 乘以 x2


def test_subbug1():  # 定义函数 test_subbug1，测试替换操作是否正常
    # 确保这两个表达式不会失败
    (x**x).subs(x, 1)  # 对 x**x 进行 x 替换为 1 的操作
    (x**x).subs(x, 1.0)  # 对 x**x 进行 x 替换为 1.0 的操作


def test_subbug2():  # 定义函数 test_subbug2，测试特定替换不会引起无限递归
    # 断言：用 abs(x) 替换 x 后的 Float(7.7) 与 7.7 的绝对误差小于 epsilon
    assert Float(7.7).epsilon_eq(abs(x).subs(x, -7.7))


def test_dict_set():  # 定义函数 test_dict_set，测试符号字典和集合的替换操作
    a, b, c = map(Wild, 'abc')  # 创建符号通配符 a, b, c

    f = 3*cos(4*x)  # 定义函数表达式 f
    r = f.match(a*cos(b*x))  # 匹配 f 到 a*cos(b*x) 的模式
    assert r == {a: 3, b: 4}  # 断言：匹配结果 r 应该是 {a: 3, b: 4}
    e = a/b*sin(b*x)  # 定义另一个表达式 e
    assert e.subs(r) == r[a]/r[b]*sin(r[b]*x)  # 断言：用 r 替换 e 后的结果应该等于 r[a]/r[b]*sin(r[b]*x)
    assert e.subs(r) == 3*sin(4*x) / 4  # 断言：用 r 替换 e 后的结果应该等于 3*sin(4*x) / 4
    s = set(r.items())  # 将 r 的键值对转换成集合 s
    assert e.subs(s) == r[a]/r[b]*sin(r[b]*x)  # 断言：用 s 替换 e 后的结果应该等于 r[a]/r[b]*sin(r[b]*x)
    assert e.subs(s) == 3*sin(4*x) / 4  # 断言：用 s 替换 e 后的结果应该等于 3*sin(4*x) / 4

    assert e.subs(r) == r[a]/r[b]*sin(r[b]*x)  # 断言：用 r 替换 e 后的结果应该等于 r[a]/r[b]*sin(r[b]*x)
    assert e.subs(r) == 3*sin(4*x) / 4  # 断言：用 r 替换 e 后的结果应该等于 3*sin(4*x) / 4
    assert x.subs(Dict((x, 1))) == 1  # 断言：用字典 (x, 1) 替换 x 后结果应该等于 1


def test_dict_ambigous():   # 定义函数 test_dict_ambigous，测试符号替换的歧义性
    f = x*exp(x)  # 定义表达式 f
    g = z*exp(z)  # 定义表达式 g

    df = {x: y, exp(x): y}  # 定义替换字典 df
    dg = {z: y, exp(z): y}  # 定义替换字典 dg

    assert f.subs(df) == y**2  # 断言：用 df 替换 f 后的结果应该等于 y 的平方
    assert g.subs(dg) == y**2  # 断言：用 dg 替换 g 后的结果应该等于 y 的平方

    # 下面展示替换顺序对结果的影响
    assert f.subs(x, y).subs(exp(x), y) == y*exp(y)  # 断言：先替换 x 为 y，再替换 exp(x) 为 y 的结果应该等于 y*exp(y)
    assert f.subs(exp(x), y).subs(x, y) == y**2  # 断言：先替换 exp(x) 为 y，再替换 x 为 y 的结果应该等于 y 的平方

    # 表达式长度和操作计数相同，因此默认排序键会解析顺序...
    # 如果不想要这个结果，就不应该使用无序序列。
    e = 1 + x*y
    assert e.subs({x: y, y: 2}) == 5  # 断言：用 {x: y, y: 2} 替换 e 后的结果应该等于 5
    # 这里，没有明显冲突的键或值，但结果取决于顺序
    assert exp(x/2 + y).subs({exp(y + 1): 2, x: 2}) == exp(y + 1)  # 断言：用 {exp(y + 1): 2, x: 2} 替换 exp(x/2 + y) 后的结果应该等于 exp(y + 1)


def test_deriv_sub_bug3():  # 定义函数 test_deriv_sub_bug3，测试导数替换的 bug
    f = Function('f')  # 定义函数符号 f
    pat = Derivative(f(x), x, x)  # 定义二阶导数模式 pat
    assert pat.subs(y, y**2) == Derivative(f(x), x, x)  # 断言：用 y 替换 pat 后应该等于原始 pat
    assert pat.subs(y, y**2) != Derivative(f(x), x)  # 断言：用 y 替换 pat 后不应等于一阶导数 f(x)


def test_equality_subs1():  # 定义函数 test_equality_subs1，测试等式替换
    f = Function('f')  # 定义函数符号 f
    eq = Eq(f(x)**2, x)  # 定义等式表达式 eq
    res = Eq(Integer(16), x)  # 定义结果等式 res
    assert eq.subs(f(x), 4) == res  # 断言：用 f(x) 替换 eq 后的结果应该等于 res


def test_equality_subs2():  # 定义函数 test_equality_subs2，测试等式替换的布尔值
    f = Function('f')  # 定义函数符号 f
    eq = Eq(f(x)**2, 16)  # 定义等式表达式 eq
    assert bool(eq.subs(f(x), 3)) is False  # 断言：用 f(x) 替换 eq 后应该为 False
    assert bool(eq.subs(f(x), 4)) is True  # 断言：用 f(x) 替
    # 创建符号变量 A, B, C，其中 A 和 B 不可交换
    A, B, C = symbols('A B C', commutative=0)
    
    # 断言：将 z*x 替换为 y 后，x*y*z 的结果应为 y 的平方
    assert (x*y*z).subs(z*x, y) == y**2
    
    # 断言：将 1/x 替换为 z 后，z*x 的结果应为 1
    assert (z*x).subs(1/x, z) == 1
    
    # 断言：将 1/z 替换为 a 后，x*y/z 的结果应为 a*x*y
    assert (x*y/z).subs(1/z, a) == a*x*y
    
    # 断言：将 x/z 替换为 a 后，x*y/z 的结果应为 a*y
    assert (x*y/z).subs(x/z, a) == a*y
    
    # 断言：将 y/z 替换为 a 后，x*y/z 的结果应为 a*x
    assert (x*y/z).subs(y/z, a) == a*x
    
    # 断言：将 x/z 替换为 1/a 后，x*y/z 的结果应为 y/a
    assert (x*y/z).subs(x/z, 1/a) == y/a
    
    # 断言：将 x 替换为 1/a 后，x*y/z 的结果应为 y/(z*a)
    assert (x*y/z).subs(x, 1/a) == y/(z*a)
    
    # 断言：将 5*x*y 替换为 z 后，2*x*y 的结果不应为 z*Rational(2, 5)
    assert (2*x*y).subs(5*x*y, z) != z*Rational(2, 5)
    
    # 断言：将 x*y 替换为 a 后，x*y*A 的结果应为 a*A
    assert (x*y*A).subs(x*y, a) == a*A
    
    # 断言：将 x*y**(x/2) 替换为 2 后，x**2*y**(x*Rational(3, 2)) 的结果应为 4*y**(x/2)
    assert (x**2*y**(x*Rational(3, 2))).subs(x*y**(x/2), 2) == 4*y**(x/2)
    
    # 断言：将 x*exp(x) 替换为 2 后，x*exp(x*2) 的结果应为 2*exp(x)
    assert (x*exp(x*2)).subs(x*exp(x), 2) == 2*exp(x)
    
    # 断言：将 x**y 替换为 2 后，(x**(2*y))**3 的结果应为 64
    assert ((x**(2*y))**3).subs(x**y, 2) == 64
    
    # 断言：将 x*A 替换为 y 后，x*A*B 的结果应为 y*B
    assert (x*A*B).subs(x*A, y) == y*B
    
    # 断言：将 x*y 替换为 2 后，x*y*(1 + x)*(1 + x*y) 的结果应为 6*(1 + x)
    assert (x*y*(1 + x)*(1 + x*y)).subs(x*y, 2) == 6*(1 + x)
    
    # 断言：将 A*B 替换为 x*A*B 后，(1 + A*B)*A*B 的结果应为 A*x*A*B
    assert ((1 + A*B)*A*B).subs(A*B, x*A*B)
    
    # 断言：将 x/z 替换为 A 后，x*a/z 的结果应为 a*A
    assert (x*a/z).subs(x/z, A) == a*A
    
    # 断言：将 x**2*A 替换为 a 后，x**3*A 的结果应为 a*x
    assert (x**3*A).subs(x**2*A, a) == a*x
    
    # 断言：将 x**2*A*B 替换为 a 后，x**2*A*B 的结果应为 a*A
    assert (x**2*A*B).subs(x**2*B, a) == a*A
    
    # 断言：将 x**2*A 替换为 a 后，x**2*A*B 的结果应为 a*B
    assert (x**2*A*B).subs(x**2*A, a) == a*B
    
    # 断言：将 a**4*c**3*A**3/b**4 替换为 z 后，b*A**3/(a**3*c**3) 的结果应为 b*A**3/(a**3*c**3)
    assert (b*A**3/(a**3*c**3)).subs(a**4*c**3*A**3/b**4, z) == b*A**3/(a**3*c**3)
    
    # 断言：将 2*x 替换为 y 后，6*x 的结果应为 3*y
    assert (6*x).subs(2*x, y) == 3*y
    
    # 断言：将 y*exp(x) 替换为 2 后，y*exp(x*Rational(3, 2)) 的结果应为 2*exp(x/2)
    assert (y*exp(x*Rational(3, 2))).subs(y*exp(x), 2) == 2*exp(x/2)
    
    # 断言：将 A*B*A 替换为 C 后，(1 + A*B)*A*B 的结果应为 A*C**2*A
    assert (A**2*B*A**2*B*A**2).subs(A*B*A, C) == A*C**2*A
    
    # 断言：将 x*A 替换为 y 后，x*A**3 的结果应为 y*A**2
    assert (x*A**3).subs(x*A, y) == y*A**2
    
    # 断言：将 x*A 替换为 y 后，x**2*A**3 的结果应为 y**2*A
    assert (x**2*A**3).subs(x*A, y) == y**2*A
    
    # 断言：将 x*A 替换为 B 后，x*A**3 的结果应为 B*A**2
    assert (x*A**3).subs(x*A, B) == B*A**2
    
    # 断言：将 x*A 替换为 B 后，x*A*B*A*exp(x*A*B) 的结果应为 B**2*A*exp(B*B)
    assert (x*A*B*A*exp(x*A*B)).subs(x*A, B) == B**2*A*exp(B*B)
    
    # 断言：将 x*A 替换为 B 后，x**2*A*B*A*exp(x*A*B) 的结果应为 B**3*exp(B**2)
    assert (x**2*A*B*A*exp(x*A*B)).subs(x*A, B) == B**3*exp(B**2)
    
    # 断言：将 x*A 替换为 B 后，x**3*A*exp(x*A*B)*A*exp(x*A*B) 的结果应为 x*B*exp(B**2)*B*exp(B**2)
    assert (x**3*A*exp(x*A*B)*A*exp(x*A*B)).subs(x*A, B) == x*B*exp(B**2)*B*exp(B**2)
    
    # 断言：将 x*A*B 替换为 C 后，x*A*B*C*A*B 的结果应为 C**2*A*B
    assert (x*A*B*C*A*B).subs(x*A*B, C) == C**2*A*B
    
    # 断言：将 a*b 替换为 2 后，-I*a*b 的结果应为 -2*I
    assert (-I*a*b).subs(a*b, 2) == -2*I
    
    # issue 6361
    # 断言：将 -2*a 替换为 1 后，-8*I*a 的结果应为 4*I
    assert (-8*I*a).subs(-2*a, 1) == 4*I
    
    # 断言：将 -a 替换为 1 后，-I*a 的结果应为 I
    assert (-I*a).subs(-a, 1) == I
    
    # issue 6441
    # 断言：将 2*x 替换为 y 后，4*x**2 的结果应为 y**2
    assert (4*x**2).subs(2*x, y) == y**2
    
    # 断言：将 2*x 替换为 y 后，2*4*x**2 的结果应为 2*y**2
    assert (2*4*x
def test_subs_simple():
    # 定义符号变量 a 和 x，其中 a 是可交换的，x 是不可交换的
    a = symbols('a', commutative=True)
    x = symbols('x', commutative=False)

    # 断言：对于可交换变量 a，不进行替换
    assert (2*a).subs(1, 3) == 2*a
    # 断言：对于可交换变量 a，将 2 替换为 3
    assert (2*a).subs(2, 3) == 3*a
    # 断言：对于可交换变量 a，将 a 替换为 3，结果为 6
    assert (2*a).subs(a, 3) == 6
    # 断言：对于 sin(2)，不进行替换
    assert sin(2).subs(1, 3) == sin(2)
    # 断言：对于 sin(2)，将 2 替换为 3
    assert sin(2).subs(2, 3) == sin(3)
    # 断言：对于 sin(a)，将 a 替换为 3
    assert sin(a).subs(a, 3) == sin(3)

    # 断言：对于不可交换变量 x，不进行替换
    assert (2*x).subs(1, 3) == 2*x
    # 断言：对于不可交换变量 x，将 2 替换为 3
    assert (2*x).subs(2, 3) == 3*x
    # 断言：对于不可交换变量 x，将 x 替换为 3，结果为 6
    assert (2*x).subs(x, 3) == 6
    # 断言：对于 sin(x)，将 x 替换为 3
    assert sin(x).subs(x, 3) == sin(3)


def test_subs_constants():
    # 定义符号变量 a, b, x, y，其中 a 和 b 是可交换的，x 和 y 是不可交换的
    a, b = symbols('a b', commutative=True)
    x, y = symbols('x y', commutative=False)

    # 断言：对于可交换变量 a*b，不进行替换
    assert (a*b).subs(2*a, 1) == a*b
    # 断言：对于 1.5*a*b，将 a 替换为 1，结果为 1.5*b
    assert (1.5*a*b).subs(a, 1) == 1.5*b
    # 断言：对于 2*a*b，将 2*a 替换为 1，结果为 b
    assert (2*a*b).subs(2*a, 1) == b
    # 断言：对于 2*a*b，不进行替换
    assert (2*a*b).subs(4*a, 1) == 2*a*b

    # 断言：对于不可交换变量 x*y，不进行替换
    assert (x*y).subs(2*x, 1) == x*y
    # 断言：对于 1.5*x*y，将 x 替换为 1，结果为 1.5*y
    assert (1.5*x*y).subs(x, 1) == 1.5*y
    # 断言：对于 2*x*y，将 2*x 替换为 1，结果为 y
    assert (2*x*y).subs(2*x, 1) == y
    # 断言：对于 2*x*y，不进行替换
    assert (2*x*y).subs(4*x, 1) == 2*x*y


def test_subs_commutative():
    # 定义符号变量 a, b, c, d, K，都是可交换的
    a, b, c, d, K = symbols('a b c d K', commutative=True)

    # 断言：对于 a*b，将 a*b 替换为 K，结果为 K
    assert (a*b).subs(a*b, K) == K
    # 断言：对于 a*b*a*b，将 a*b 替换为 K，结果为 K^2
    assert (a*b*a*b).subs(a*b, K) == K**2
    # 断言：对于 a*a*b*b，将 a*b 替换为 K，结果为 K^2
    assert (a*a*b*b).subs(a*b, K) == K**2
    # 断言：对于 a*b*c*d，将 a*b*c 替换为 K，结果为 d*K
    assert (a*b*c*d).subs(a*b*c, K) == d*K
    # 断言：对于 a*b**c，将 a 替换为 K，结果为 K*b**c
    assert (a*b**c).subs(a, K) == K*b**c
    # 断言：对于 a*b**c，将 b 替换为 K，结果为 a*K**c
    assert (a*b**c).subs(b, K) == a*K**c
    # 断言：对于 a*b**c，将 c 替换为 K，结果为 a*b**K
    assert (a*b**c).subs(c, K) == a*b**K
    # 断言：对于 a*b*c*b*a，将 a*b 替换为 K，结果为 c*K**2
    assert (a*b*c*b*a).subs(a*b, K) == c*K**2
    # 断言：对于 a**3*b**2*a，将 a*b 替换为 K，结果为 a**2*K**2
    assert (a**3*b**2*a).subs(a*b, K) == a**2*K**2


def test_subs_noncommutative():
    # 定义符号变量 w, x, y, z, L，其中 w, x, y, z 是不可交换的，L 是可交换的
    w, x, y, z, L = symbols('w x y z L', commutative=False)
    # 定义可交换变量 alpha 和 someint
    alpha = symbols('alpha', commutative=True)
    someint = symbols('someint', commutative=True, integer=True)

    # 断言：对于 x*y，将 x*y 替换为 L，结果为 L
    assert (x*y).subs(x*y, L) == L
    # 断言：对于 w*y*x，不进行替换
    assert (w*y*x).subs(x*y, L) == w*y*x
    # 断言：对于 w*x*y*z，将 x*y 替换为 L，结果为 w*L*z
    assert (w*x*y*z).subs(x*y, L) == w*L*z
    # 断言：对于 x*y*x*y，将 x*y 替换为 L，结果为 L**2
    assert (x*y*x*y).subs(x*y, L) == L**2
    # 断言：对于 x*x*y，将 x*y 替换为 L，结果为 x*L
    assert (x*x*y).subs(x*y, L) == x*L
    # 断言：对于 x*x*y*y，将 x*y 替换为 L，结果为 x*L*y
    assert (x*x*y*y).subs(x*y, L) == x*L*y
    # 断言：对于 w*x*y，将 x*y*z 替换为 L，不进行替换
    assert (w*x*y).subs(x*y*z, L) == w*x*y
    # 断言：对于 x*y**z，将 x 替换为 L，结果为 L*y**z
    assert (x*y**z).subs(x, L) == L*y**z
    # 断言：对于 x*y**z，将 y 替换为 L，结果为 x*L**z
    assert (x*y**z).subs(y, L) == x*L**z
    # 断言：对于 x*y**z，将 z 替换为 L，结果为 x*y**L
    assert (x*y**z).subs(z, L) == x*y**L
    # 断言：对于 w*x*y*z*x*y，将 x*y*z 替换为 L，结果为 w*L*x*y
    assert (w*x*y*z*x*y).subs(x*y*z, L) == w*L*x*y
    # 断言：对于 w*x*y*y*w*x*x*y*x*y*y*x*y，将 x*y 替换为 L，结果为 w*L*y*w*x*L**2
    # 断言：验证指数操作和替换操作的结果是否符合预期

    # 验证指数操作 x**(3*someint) 替换 x**(2*someint) 后是否等于 L * x**someint
    assert (x**(3*someint)).subs(x**(2*someint), L) == L * x**someint
    
    # 验证指数操作 x**(4*someint) 替换 x**(2*someint) 后是否等于 L**2
    assert (x**(4*someint)).subs(x**(2*someint), L) == L**2
    
    # 验证指数操作 x**(4*someint + 1) 替换 x**(2*someint) 后是否等于 L**2 * x
    assert (x**(4*someint + 1)).subs(x**(2*someint), L) == L**2 * x
    
    # 验证指数操作 x**(4*someint) 替换 x**(3*someint) 后是否等于 L * x**someint
    assert (x**(4*someint)).subs(x**(3*someint), L) == L * x**someint
    
    # 验证指数操作 x**(4*someint + 1) 替换 x**(3*someint) 后是否等于 L * x**(someint + 1)
    assert (x**(4*someint + 1)).subs(x**(3*someint), L) == L * x**(someint + 1)

    # 验证指数操作 x**(2*alpha) 替换 x**alpha 后是否等于 x**(2*alpha)
    assert (x**(2*alpha)).subs(x**alpha, L) == x**(2*alpha)
    
    # 验证指数操作 x**(2*alpha + 2) 替换 x**2 后是否等于 x**(2*alpha + 2)
    assert (x**(2*alpha + 2)).subs(x**2, L) == x**(2*alpha + 2)
    
    # 验证指数操作 (2*z)**alpha 替换 z**alpha 后是否等于 (2*z)**alpha
    assert ((2*z)**alpha).subs(z**alpha, y) == (2*z)**alpha
    
    # 验证指数操作 x**(2*someint*alpha) 替换 x**someint 后是否等于 x**(2*someint*alpha)
    assert (x**(2*someint*alpha)).subs(x**someint, L) == x**(2*someint*alpha)
    
    # 验证指数操作 x**(2*someint + alpha) 替换 x**someint 后是否等于 x**(2*someint + alpha)
    assert (x**(2*someint + alpha)).subs(x**someint, L) == x**(2*someint + alpha)

    # 对于指数操作 x**(someint**2 + 3)，说明目前不进行替换，因为需要识别 someint**2 是否可被 someint 整除
    assert (x**(someint**2 + 3)).subs(x**someint, L) == x**(someint**2 + 3)

    # 断言：验证 4**z 替换 2**z 后是否等于 y**2
    assert (4**z).subs(2**z, y) == y**2

    # 负指数操作的验证
    # 验证指数操作 x**(-1) 替换 x**3 后是否等于 x**(-1)
    assert (x**(-1)).subs(x**3, L) == x**(-1)
    
    # 验证指数操作 x**(-2) 替换 x**3 后是否等于 x**(-2)
    assert (x**(-2)).subs(x**3, L) == x**(-2)
    
    # 验证指数操作 x**(-3) 替换 x**3 后是否等于 L**(-1)
    assert (x**(-3)).subs(x**3, L) == L**(-1)
    
    # 验证指数操作 x**(-4) 替换 x**3 后是否等于 L**(-1) * x**(-1)
    assert (x**(-4)).subs(x**3, L) == L**(-1) * x**(-1)
    
    # 验证指数操作 x**(-5) 替换 x**3 后是否等于 L**(-1) * x**(-2)
    assert (x**(-5)).subs(x**3, L) == L**(-1) * x**(-2)

    # 验证指数操作 x**(-1) 替换 x**(-3) 后是否等于 x**(-1)
    assert (x**(-1)).subs(x**(-3), L) == x**(-1)
    
    # 验证指数操作 x**(-2) 替换 x**(-3) 后是否等于 x**(-2)
    assert (x**(-2)).subs(x**(-3), L) == x**(-2)
    
    # 验证指数操作 x**(-3) 替换 x**(-3) 后是否等于 L
    assert (x**(-3)).subs(x**(-3), L) == L
    
    # 验证指数操作 x**(-4) 替换 x**(-3) 后是否等于 L * x**(-1)
    assert (x**(-4)).subs(x**(-3), L) == L * x**(-1)
    
    # 验证指数操作 x**(-5) 替换 x**(-3) 后是否等于 L * x**(-2)
    assert (x**(-5)).subs(x**(-3), L) == L * x**(-2)

    # 验证指数操作 x**1 替换 x**(-3) 后是否等于 x
    assert (x**1).subs(x**(-3), L) == x
    
    # 验证指数操作 x**2 替换 x**(-3) 后是否等于 x**2
    assert (x**2).subs(x**(-3), L) == x**2
    
    # 验证指数操作 x**3 替换 x**(-3) 后是否等于 L**(-1)
    assert (x**3).subs(x**(-3), L) == L**(-1)
    
    # 验证指数操作 x**4 替换 x**(-3) 后是否等于 L**(-1) * x
    assert (x**4).subs(x**(-3), L) == L**(-1) * x
    
    # 验证指数操作 x**5 替换 x**(-3) 后是否等于 L**(-1) * x**2
    assert (x**5).subs(x**(-3), L) == L**(-1) * x**2
# 定义一个测试函数，用于测试符号代换的基本功能
def test_subs_basic_funcs():
    # 定义符号变量，并指定其中的一些为可交换的（commutative=True）
    a, b, c, d, K = symbols('a b c d K', commutative=True)
    # 定义另一组符号变量，并指定其中的一些为不可交换的（commutative=False）
    w, x, y, z, L = symbols('w x y z L', commutative=False)

    # 使用assert语句验证以下每个表达式的符号代换是否得到预期结果
    assert (x + y).subs(x + y, L) == L
    assert (x - y).subs(x - y, L) == L
    assert (x/y).subs(x, L) == L/y
    assert (x**y).subs(x, L) == L**y
    assert (x**y).subs(y, L) == x**L
    assert ((a - c)/b).subs(b, K) == (a - c)/K
    assert (exp(x*y - z)).subs(x*y, L) == exp(L - z)
    assert (a*exp(x*y - w*z) + b*exp(x*y + w*z)).subs(z, 0) == \
        a*exp(x*y) + b*exp(x*y)
    assert ((a - b)/(c*d - a*b)).subs(c*d - a*b, K) == (a - b)/K
    assert (w*exp(a*b - c)*x*y/4).subs(x*y, L) == w*exp(a*b - c)*L/4


# 定义一个测试函数，用于测试符号代换的通配符功能
def test_subs_wild():
    # 定义通配符符号变量
    R, S, T, U = symbols('R S T U', cls=Wild)

    # 使用assert语句验证以下每个表达式的符号代换是否得到预期结果
    assert (R*S).subs(R*S, T) == T
    assert (S*R).subs(R*S, T) == T
    assert (R + S).subs(R + S, T) == T
    assert (R**S).subs(R, T) == T**S
    assert (R**S).subs(S, T) == R**T
    assert (R*S**T).subs(R, U) == U*S**T
    assert (R*S**T).subs(S, U) == R*U**T
    assert (R*S**T).subs(T, U) == R*S**U


# 定义一个测试函数，用于测试符号代换的混合功能
def test_subs_mixed():
    # 定义符号变量，其中一些为可交换的（commutative=True），另一些为不可交换的（commutative=False）
    a, b, c, d, K = symbols('a b c d K', commutative=True)
    w, x, y, z, L = symbols('w x y z L', commutative=False)
    # 定义通配符符号变量
    R, S, T, U = symbols('R S T U', cls=Wild)

    # 使用assert语句验证以下每个表达式的符号代换是否得到预期结果
    assert (a*x*y).subs(x*y, L) == a*L
    assert (a*b*x*y*x).subs(x*y, L) == a*b*L*x
    assert (R*x*y*exp(x*y)).subs(x*y, L) == R*L*exp(L)
    assert (a*x*y*y*x - x*y*z*exp(a*b)).subs(x*y, L) == a*L*y*x - L*z*exp(a*b)
    e = c*y*x*y*x**(R*S - a*b) - T*(a*R*b*S)
    assert e.subs(x*y, L).subs(a*b, K).subs(R*S, U) == \
        c*y*L*x**(U - K) - T*(U*K)


# 定义一个测试函数，用于测试符号代换中的除法操作
def test_division():
    # 定义符号变量，均为可交换的（commutative=True）
    a, b, c = symbols('a b c', commutative=True)
    x, y, z = symbols('x y z', commutative=True)

    # 使用assert语句验证以下每个表达式的符号代换是否得到预期结果
    assert (1/a).subs(a, c) == 1/c
    assert (1/a**2).subs(a, c) == 1/c**2
    assert (1/a**2).subs(a, -2) == Rational(1, 4)
    assert (-(1/a**2)).subs(a, -2) == Rational(-1, 4)

    assert (1/x).subs(x, z) == 1/z
    assert (1/x**2).subs(x, z) == 1/z**2
    assert (1/x**2).subs(x, -2) == Rational(1, 4)
    assert (-(1/x**2)).subs(x, -2) == Rational(-1, 4)

    # issue 5360
    assert (1/x).subs(x, 0) == 1/S.Zero


# 定义一个测试函数，用于测试符号代换中的加法操作
def test_add():
    # 定义符号变量
    a, b, c, d, x, y, t = symbols('a b c d x y t')

    # 使用assert语句验证以下每个表达式的符号代换是否得到预期结果
    assert (a**2 - b - c).subs(a**2 - b, d) in [d - c, a**2 - b - c]
    assert (a**2 - c).subs(a**2 - c, d) == d
    assert (a**2 - b - c).subs(a**2 - c, d) in [d - b, a**2 - b - c]
    assert (a**2 - x - c).subs(a**2 - c, d) in [d - x, a**2 - x - c]
    assert (a**2 - b - sqrt(a)).subs(a**2 - sqrt(a), c) == c - b
    assert (a + b + exp(a + b)).subs(a + b, c) == c + exp(c)
    assert (c + b + exp(c + b)).subs(c + b, a) == a + exp(a)
    assert (a + b + c + d).subs(b + c, x) == a + d + x
    assert (a + b + c + d).subs(-b - c, x) == a + d - x
    assert ((x + 1)*y).subs(x + 1, t) == t*y
    assert ((-x - 1)*y).subs(x + 1, t) == -t*y
    assert ((x - 1)*y).subs(x + 1, t) == y*(t - 2)
    assert ((-x + 1)*y).subs(x + 1, t) == y*(-t + 2)
    # 计算表达式 e = a**2 - b - c
    e = a**2 - b - c
    # 断言：将 e 中的前两个参数相加后用 d 替换，应该等于 d + e 的第三个参数
    assert e.subs(Add(*e.args[:2]), d) == d + e.args[2]
    # 断言：将 e 中的 a**2 - c 部分用 d 替换，应该等于 d - b
    assert e.subs(a**2 - c, d) == d - b

    # 断言：对于表达式 (0.1 + a)，将其中的 0.1 替换为 Rational(1, 10)，结果应为 Rational(1, 10) + a
    # 0.1 和 Rational(1, 10) 不相同，因此应该做替换
    assert (0.1 + a).subs(0.1, Rational(1, 10)) == Rational(1, 10) + a

    # 计算表达式 e = (-x*(-y + 1) - y*(y - 1))
    e = (-x*(-y + 1) - y*(y - 1))
    # 计算 ans，将 e 中的表达式展开
    ans = (-x*(x) - y*(-x)).expand()
    # 断言：将 e 中的 -y + 1 替换为 x，结果应该等于 ans
    assert e.subs(-y + 1, x) == ans

    # 断言：当 x 趋向正无穷时，exp(x) + cos(x) 应该等于正无穷
    assert (exp(x) + cos(x)).subs(x, oo) == oo
    # 断言：AccumBounds(-1, 1) 和 正无穷 的加法应该等于正无穷
    assert Add(*[AccumBounds(-1, 1), oo]) == oo
    # 断言：正无穷 和 AccumBounds(-1, 1) 的加法应该等于正无穷
    assert Add(*[oo, AccumBounds(-1, 1)]) == oo
# 定义一个函数，用于测试符号代换操作是否正确
def test_subs_issue_4009():
    # 断言：对于表达式 I*Symbol('a')，将 1 替换为 2，应该得到 I*Symbol('a')
    assert (I*Symbol('a')).subs(1, 2) == I*Symbol('a')


# 定义一个函数，用于测试函数符号代换的各种情况
def test_functions_subs():
    # 创建两个函数符号 f 和 g
    f, g = symbols('f g', cls=Function)
    # 创建一个 Lambda 表达式 l，表示 sin(x) + y
    l = Lambda((x, y), sin(x) + y)
    # 断言：将 g 替换为 l，应得到 sin(y) + x + cos(x)
    assert (g(y, x) + cos(x)).subs(g, l) == sin(y) + x + cos(x)
    # 断言：将 f 替换为 sin，应得到 sin(x)**2
    assert (f(x)**2).subs(f, sin) == sin(x)**2
    # 断言：将 f 替换为 log，不应发生变化
    assert (f(x, y)).subs(f, log) == log(x, y)
    # 断言：将 f 替换为 sin，不应发生变化
    assert (f(x, y)).subs(f, sin) == f(x, y)
    # 断言：对于表达式 sin(x) + atan2(x, y)，将 atan2 替换为 f，sin 替换为 g，应得到 f(x, y) + g(x)
    assert (sin(x) + atan2(x, y)).subs([[atan2, f], [sin, g]]) == f(x, y) + g(x)
    # 断言：将 f 替换为 l，g 替换为 exp，应得到 exp(x + sin(x + y))
    assert (g(f(x + y, x))).subs([[f, l], [g, exp]]) == exp(x + sin(x + y))


# 定义一个函数，用于测试导数符号代换的各种情况
def test_derivative_subs():
    # 创建一个函数符号 f
    f = Function('f')
    # 创建一个函数符号 g
    g = Function('g')
    # 断言：对于导数 Derivative(f(x), x)，将 f(x) 替换为 y，不应等于 0
    assert Derivative(f(x), x).subs(f(x), y) != 0
    # 断言：对于导数 Derivative(f(x), x)，将 f(x) 替换为 y，然后使用 xreplace 将函数恢复，应得到 Derivative(f(x), x)
    assert Derivative(f(x), x).subs(f(x), y).xreplace({y: f(x)}) == Derivative(f(x), x)
    # 断言：对于 Derivative(f(x), x) + f(x)，应用 cse 后第二个元素的第一个元素应包含 Derivative
    assert cse(Derivative(f(x), x) + f(x))[1][0].has(Derivative)
    # 断言：对于 Derivative(f(x, y), x) + Derivative(f(x, y), y)，应用 cse 后第二个元素的第一个元素应包含 Derivative
    assert cse(Derivative(f(x, y), x) + Derivative(f(x, y), y))[1][0].has(Derivative)
    # 创建一个表达式 eq，表示 Derivative(g(x), g(x))
    eq = Derivative(g(x), g(x))
    # 断言：将 eq 中的 g 替换为 f，应得到 Derivative(f(x), f(x))
    assert eq.subs(g, f) == Derivative(f(x), f(x))
    # 断言：将 eq 中的 g(x) 替换为 f(x)，应得到 Derivative(f(x), f(x))
    assert eq.subs(g(x), f(x)) == Derivative(f(x), f(x))
    # 断言：将 eq 中的 g 替换为 cos，应得到 Subs(Derivative(y, y), y, cos(x))
    assert eq.subs(g, cos) == Subs(Derivative(y, y), y, cos(x))


# 定义一个函数，用于测试高阶导数符号代换的各种情况
def test_derivative_subs2():
    # 创建两个函数符号 f_func 和 g_func，分别表示 f(x, y, z) 和 g(x, y, z)
    f_func, g_func = symbols('f g', cls=Function)
    # 创建 f 和 g 表达式
    f, g = f_func(x, y, z), g_func(x, y, z)
    # 断言：将 Derivative(f, x, y) 替换为 g，应得到 g
    assert Derivative(f, x, y).subs(Derivative(f, x, y), g) == g
    # 断言：将 Derivative(f, y, x) 替换为 g，应得到 g
    assert Derivative(f, y, x).subs(Derivative(f, x, y), g) == g
    # 断言：将 Derivative(f, x, y) 替换为 Derivative(g, y)，应得到 Derivative(g, y)
    assert Derivative(f, x, y).subs(Derivative(f, x), g) == Derivative(g, y)
    # 断言：将 Derivative(f, x, y) 替换为 Derivative(g, x)，应得到 Derivative(g, x)
    assert Derivative(f, x, y).subs(Derivative(f, y), g) == Derivative(g, x)
    # 断言：对于 Derivative(f, x, y, z)，将 Derivative(f, x, z) 替换为 g，应得到 Derivative(g, y)
    assert (Derivative(f, x, y, z).subs(Derivative(f, x, z), g) == Derivative(g, y))
    # 断言：对于 Derivative(f, x, y, z)，将 Derivative(f, z, y) 替换为 g，应得到 Derivative(g, x)
    assert (Derivative(f, x, y, z).subs(Derivative(f, z, y), g) == Derivative(g, x))
    # 断言：对于 Derivative(f, x, y, z)，将 Derivative(f, z, y, x) 替换为 g，应得到 g
    assert (Derivative(f, x, y, z).subs(Derivative(f, z, y, x), g) == g)

    # Issue 9135
    # 断言：对于 Derivative(f, x, x, y)，将 Derivative(f, y, y) 替换为 g，应得到 Derivative(f, x, x, y)
    assert (Derivative(f, x, x, y).subs(Derivative(f, y, y), g) == Derivative(f, x, x, y))
    # 断言：对于 Derivative(f, x, y, y, z)，将 Derivative(f, x, y, y, y) 替换为 g，应得到 Derivative(f, x, y, y, z)
    assert (Derivative(f, x, y, y, z).subs(Derivative(f, x, y, y, y), g) == Derivative(f, x, y, y, z))

    # 断言：对于 Derivative(f, x, y)，将 Derivative(f_func(x), x, y) 替换为 g，应得到 Derivative(f, x, y)
    assert Derivative(f, x, y).subs(Derivative(f_func(x), x, y), g) == Derivative(f, x, y)


# 定义一个函数，用于测试导数表达式代换的特殊情况
def test_derivative_subs3():
    # 创建一个导数表达式 dex，表示 Derivative(exp(x), x)
    dex = Derivative(exp(x), x)
    # 断言：对于 Derivative(dex, x)，将 dex 替换为 exp(x)，应得到 dex
    assert Derivative(dex, x).subs(dex, exp(x)) == dex
    # 断言：对于 dex，将 exp(x) 替换为 dex，应得到 Derivative(exp(x), x, x)
    assert dex.subs(exp(x), dex) == Derivative(exp(x), x, x)


# 定义一个函数，用于测试特定代换问题的各种情况
def test_issue_5284():
    # 创建两个非交换符号 A 和 B
    # 断言：用于验证符号表达式 sin(x) 在替换字典 l 下的计算结果是否等于 2
    assert (sin(x)).subs(l) == \
           (sin(x)).subs(dict(l)) == 2
    # 断言：用于验证符号表达式 sin(x) 在反转字典 l 后的替换下的计算结果是否等于 sin(1)
    assert sin(x).subs(reversed(l)) == sin(1)

    # 创建复杂的符号表达式
    expr = sin(2*x) + sqrt(sin(2*x))*cos(2*x)*sin(exp(x)*x)
    # 定义替换字典
    reps = {sin(2*x): c,
               sqrt(sin(2*x)): a,
               cos(2*x): b,
               exp(x): e,
               x: d,}
    # 断言：验证复杂表达式在替换字典 reps 下的计算结果是否等于 c + a*b*sin(d*e)
    assert expr.subs(reps) == c + a*b*sin(d*e)

    # 定义简单的替换列表
    l = [(x, 3), (y, x**2)]
    # 断言：验证表达式 (x + y) 在替换列表 l 下的计算结果是否等于 3 + x**2
    assert (x + y).subs(l) == 3 + x**2
    # 断言：验证表达式 (x + y) 在反转替换列表 l 后的计算结果是否等于 12
    assert (x + y).subs(reversed(l)) == 12

    # 注释：如果将列表转换为字典并进行字典查找替换，这些测试可以帮助捕捉可能出现的逻辑错误
    # 定义新的替换列表
    l = [(y, z + 2), (1 + z, 5), (z, 2)]
    # 断言：验证表达式 (y - 1 + 3*x) 在替换列表 l 下的计算结果是否等于 5 + 3*x
    assert (y - 1 + 3*x).subs(l) == 5 + 3*x
    # 重新定义替换列表
    l = [(y, z + 2), (z, 3)]
    # 断言：验证表达式 (y - 2) 在替换列表 l 下的计算结果是否等于 3
    assert (y - 2).subs(l) == 3
def test_no_arith_subs_on_floats():
    # 检查在浮点数上禁止算术替换
    assert (x + 3).subs(x + 3, a) == a
    # 替换失败，因为表达式中未找到 x + 2
    assert (x + 3).subs(x + 2, a) == a + 1

    # 检查多个变量情况下的替换
    assert (x + y + 3).subs(x + 3, a) == a + y
    # 替换失败，因为表达式中未找到 x + 2
    assert (x + y + 3).subs(x + 2, a) == a + y + 1

    # 检查在浮点数上的替换
    assert (x + 3.0).subs(x + 3.0, a) == a
    # 浮点数替换失败，因为表达式中未找到 x + 2.0
    assert (x + 3.0).subs(x + 2.0, a) == x + 3.0

    # 检查多个变量情况下在浮点数上的替换
    assert (x + y + 3.0).subs(x + 3.0, a) == a + y
    # 浮点数替换失败，因为表达式中未找到 x + 2.0
    assert (x + y + 3.0).subs(x + 2.0, a) == x + y + 3.0


def test_issue_5651():
    # 定义符号变量
    a, b, c, K = symbols('a b c K', commutative=True)
    # 替换分子表达式 b*c 为 K
    assert (a/(b*c)).subs(b*c, K) == a/K
    # 替换分子表达式 b**2*c**3 为 K**2
    assert (a/(b**2*c**3)).subs(b*c, K) == a/(c*K**2)
    # 替换 x*y 为 2，结果为 S.Half
    assert (1/(x*y)).subs(x*y, 2) == S.Half
    # 替换 x*y 为 1，结果为 2
    assert ((1 + x*y)/(x*y)).subs(x*y, 1) == 2
    # 替换 x*y 为 2，结果为 2*z
    assert (x*y*z).subs(x*y, 2) == 2*z
    # 替换 x*y 为 1，结果为 2/z
    assert ((1 + x*y)/(x*y)/z).subs(x*y, 1) == 2/z


def test_issue_6075():
    # 替换 1 为 2，结果为 Tuple(2, True)
    assert Tuple(1, True).subs(1, 2) == Tuple(2, True)


def test_issue_6079():
    # 由于 x + 2.0 == x + 2，无法进行简单的等式测试
    assert _aresame((x + 2.0).subs(2, 3), x + 2.0)
    # 替换 2.0 为 3，结果为 x + 3
    assert _aresame((x + 2.0).subs(2.0, 3), x + 3)
    # 不相同，因为 x + 2 不等于 x + 2.0
    assert not _aresame(x + 2, x + 2.0)
    # 不相同，因为 Basic(cos(x), 1) 不等于 Basic(cos(x), 1.0)
    assert not _aresame(Basic(cos(x), S(1)), Basic(cos(x), S(1.)))
    # 相同，因为 cos 等于 cos
    assert _aresame(cos, cos)
    # 不相同，因为 1 不等于 S.One
    assert not _aresame(1, S.One)
    # 不相同，因为 x 不等于 positive 的符号变量 x
    assert not _aresame(x, symbols('x', positive=True))


def test_issue_4680():
    # 替换 N 为 3，结果为 3
    N = Symbol('N')
    assert N.subs({"N": 3}) == 3


def test_issue_6158():
    # 替换 1 为 y，结果为 x - y
    assert (x - 1).subs(1, y) == x - y
    # 替换 -1 为 y，结果为 x + y
    assert (x - 1).subs(-1, y) == x + y
    # 替换 -oo 为 y，结果为 x - y
    assert (x - oo).subs(oo, y) == x - y
    # 替换 -oo 为 y，结果为 x + y
    assert (x - oo).subs(-oo, y) == x + y


def test_Function_subs():
    # 定义函数符号变量
    f, g, h, i = symbols('f g h i', cls=Function)
    # 替换 Piecewise 中的 g 为 h
    p = Piecewise((g(f(x, y)), x < -1), (g(x), x <= 1))
    assert p.subs(g, h) == Piecewise((h(f(x, y)), x < -1), (h(x), x <= 1))
    # 替换 f 和 g 为 h 和 i
    assert (f(y) + g(x)).subs({f: h, g: i}) == i(x) + h(y)


def test_simultaneous_subs():
    # 定义替换字典
    reps = {x: 0, y: 0}
    # 非同时替换，结果不相等
    assert (x/y).subs(reps) != (y/x).subs(reps)
    # 同时替换，结果相等
    assert (x/y).subs(reps, simultaneous=True) == \
        (y/x).subs(reps, simultaneous=True)
    # 转换为键值对的同时替换，结果相等
    reps = reps.items()
    assert (x/y).subs(reps) != (y/x).subs(reps)
    assert (x/y).subs(reps, simultaneous=True) == \
        (y/x).subs(reps, simultaneous=True)
    # 对 Derivative 类的同时替换
    assert Derivative(x, y, z).subs(reps, simultaneous=True) == \
        Subs(Derivative(0, y, z), y, 0)


def test_issue_6419_6421():
    # 替换 x/y 为 x，结果为 1/(1 + x)
    assert (1/(1 + x/y)).subs(x/y, x) == 1/(1 + x)
    # 替换 -2*I 为 x，结果为 -x
    assert (-2*I).subs(2*I, x) == -x
    # 替换 -I*x 为 x，结果为 -x
    assert (-I*x).subs(I*x, x) == -x
    # 替换 -3*I*y**4 为 3*I*y**2，结果为 -x*y**2
    assert (-3*I*y**4).subs(3*I*y**2, x) == -x*y**2


def test_issue_6559():
    # 替换 -x 为 1，结果为 12 + y
    assert (-12*x + y).subs(-x, 1) == 12 + y
    # 虽然涉及到 cse，但生成了 Mul._eval_subs 中的失败
    x0, x1 = symbols('x0 x1')
    e = -log(-12*sqrt(2) + 17)/24 - log(-2*sqrt(2) + 3)/12 + sqrt(2)/3
    # XXX 修改 cse，使得 x1 被消除并且 x0 = -sqrt(2)？
    assert cse(e) == (
        [(x0, sqrt(2))], [x0/3 - log(-12*x0 + 17)/24 - log(-2*x0 + 3)/12])


def test_issue_5261():
    # 定义实数符号变量 x
    x = symbols('x', real=True)
    e = I*x
    # 替换 exp(x) 为 y，结果为 y**I
    assert exp(e).subs(exp(x), y) == y**I
    # 断言：验证 (2**e).subs(2**x, y) 是否等于 y**I
    assert (2**e).subs(2**x, y) == y**I
    
    # 创建一个表达式 eq，计算 (-2)**e 的值
    eq = (-2)**e
    
    # 断言：验证 eq.subs((-2)**x, y) 是否等于 eq
    assert eq.subs((-2)**x, y) == eq
def test_issue_6923():
    # 断言测试：检查是否满足条件
    assert (-2*x*sqrt(2)).subs(2*x, y) == -sqrt(2)*y


def test_2arg_hack():
    # 创建一个非交换符号 N
    N = Symbol('N', commutative=False)
    # 创建表达式 ans，不进行求值
    ans = Mul(2, y + 1, evaluate=False)
    # 断言测试：检查是否满足条件，使用 hack2 参数进行替换
    assert (2*x*(y + 1)).subs(x, 1, hack2=True) == ans
    # 断言测试：检查是否满足条件，使用 hack2 参数进行替换
    assert (2*(y + 1 + N)).subs(N, 0, hack2=True) == ans


@XFAIL
def test_mul2():
    """When this fails, remove things labelled "2-arg hack"
    1) remove special handling in the fallback of subs that
    was added in the same commit as this test
    2) remove the special handling in Mul.flatten
    """
    # 断言测试：检查是否满足条件，判断是否为 Mul 对象
    assert (2*(x + 1)).is_Mul


def test_noncommutative_subs():
    # 创建两个非交换符号 x, y
    x,y = symbols('x,y', commutative=False)
    # 断言测试：检查是否满足条件，使用 simultaneous 参数进行替换
    assert (x*y*x).subs([(x, x*y), (y, x)], simultaneous=True) == (x*y*x**2*y)


def test_issue_2877():
    # 创建一个浮点数对象 f
    f = Float(2.0)
    # 断言测试：检查是否满足条件，使用字典进行替换
    assert (x + f).subs({f: 2}) == x + 2

    # 创建函数 r，计算并返回因式分解结果
    def r(a, b, c):
        return factor(a*x**2 + b*x + c)
    # 创建表达式 e，调用函数 r 进行计算
    e = r(5.0/6, 10, 5)
    # 断言测试：检查是否满足条件，使用 nsimplify 函数进行简化
    assert nsimplify(e) == 5*x**2/6 + 10*x + 5


def test_issue_5910():
    t = Symbol('t')
    # 断言测试：检查是否满足条件，使用 zoo 进行比较
    assert (1/(1 - t)).subs(t, 1) is zoo
    # 创建变量 n, d
    n = t
    d = t - 1
    # 断言测试：检查是否满足条件，使用 zoo 进行比较
    assert (n/d).subs(t, 1) is zoo
    # 断言测试：检查是否满足条件，使用 zoo 进行比较
    assert (-n/-d).subs(t, 1) is zoo


def test_issue_5217():
    s = Symbol('s')
    z = (1 - 2*x*x)
    w = (1 + 2*x*x)
    q = 2*x*x*2*y*y
    sub = {2*x*x: s}
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert w.subs(sub) == 1 + s
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert z.subs(sub) == 1 - s
    # 断言测试：检查是否满足条件，比较 q 的值
    assert q == 4*x**2*y**2
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert q.subs(sub) == 2*y**2*s


def test_issue_10829():
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert (4**x).subs(2**x, y) == y**2
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert (9**x).subs(3**x, y) == y**2


def test_pow_eval_subs_no_cache():
    # Tests pull request 9376 is working
    from sympy.core.cache import clear_cache

    # 创建一个符号表达式 s
    s = 1/sqrt(x**2)
    # 清除缓存
    clear_cache()

    # 断言测试：检查是否满足条件，使用 subs 进行替换
    result = s.subs(sqrt(x**2), y)
    assert result == 1/y


def test_RootOf_issue_10092():
    x = Symbol('x', real=True)
    eq = x**3 - 17*x**2 + 81*x - 118
    # 创建一个根对象 r
    r = RootOf(eq, 0)
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert (x < r).subs(x, r) is S.false


def test_issue_8886():
    from sympy.physics.mechanics import ReferenceFrame as R
    # if something can't be sympified we assume that it
    # doesn't play well with SymPy and disallow the
    # substitution
    # 创建一个 ReferenceFrame 对象 v
    v = R('A').x
    # 抛出异常测试：检查是否抛出 SympifyError 异常
    raises(SympifyError, lambda: x.subs(x, v))
    # 抛出异常测试：检查是否抛出 SympifyError 异常
    raises(SympifyError, lambda: v.subs(v, x))
    # 断言测试：检查是否满足条件，比较 v 和 x 是否相等
    assert v.__eq__(x) is False


def test_issue_12657():
    # treat -oo like the atom that it is
    # 创建替换列表 reps
    reps = [(-oo, 1), (oo, 2)]
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert (x < -oo).subs(reps) == (x < 1)
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert (x < -oo).subs(list(reversed(reps))) == (x < 1)
    # 更新替换列表 reps
    reps = [(-oo, 2), (oo, 1)]
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert (x < oo).subs(reps) == (x < 1)
    # 断言测试：检查是否满足条件，使用 subs 进行替换
    assert (x < oo).subs(list(reversed(reps))) == (x < 1)


def test_recurse_Application_args():
    # 创建一个 Lambda 函数对象 F
    F = Lambda((x, y), exp(2*x + 3*y))
    # 创建一个 Function 对象 A
    f = Function('f')
    # 创建表达式 A
    A = f(x, f(x, x))
    # 创建表达式 C
    C = F(x, F(x, x))
    # 断言：确保 A 对象使用 f 替换为 F 后的结果与使用 subs 方法替换后的结果相等，并且结果与使用 replace 方法替换后的结果相等。
    assert A.subs(f, F) == A.replace(f, F) == C
# 定义一个测试函数，用于测试 Subs 类的 subs 方法
def test_Subs_subs():
    # 断言：对于 Subs(x*y, x, x).subs(x, y)，应该返回 Subs(x*y, x, y)
    assert Subs(x*y, x, x).subs(x, y) == Subs(x*y, x, y)
    # 断言：对于 Subs(x*y, x, x + 1).subs(x, y)，应该返回 Subs(x*y, x, y + 1)
    assert Subs(x*y, x, x + 1).subs(x, y) == Subs(x*y, x, y + 1)
    # 断言：对于 Subs(x*y, y, x + 1).subs(x, y)，应该返回 Subs(y**2, y, y + 1)
    assert Subs(x*y, y, x + 1).subs(x, y) == Subs(y**2, y, y + 1)
    # 创建 Subs 对象 a 和 b
    a = Subs(x*y*z, (y, x, z), (x + 1, x + z, x))
    b = Subs(x*y*z, (y, x, z), (x + 1, y + z, y))
    # 断言：a.subs(x, y) 应该等于 b
    assert a.subs(x, y) == b and \
        # 断言：a.doit().subs(x, y) 应该等于 a.subs(x, y).doit()
        a.doit().subs(x, y) == a.subs(x, y).doit()
    # 创建函数对象 f 和 g
    f = Function('f')
    g = Function('g')
    # 断言：Subs(2*f(x, y) + g(x), f(x, y), 1).subs(y, 2) 应该等于 Subs(2*f(x, y) + g(x), (f(x, y), y), (1, 2))
    assert Subs(2*f(x, y) + g(x), f(x, y), 1).subs(y, 2) == Subs(
        2*f(x, y) + g(x), (f(x, y), y), (1, 2))


# 测试特定问题 13333 的函数
def test_issue_13333():
    eq = 1/x
    # 断言：eq.subs({"x": '1/2'}) 应该等于 2
    assert eq.subs({"x": '1/2'}) == 2
    # 断言：eq.subs({"x": '(1/2)'}) 应该等于 2
    assert eq.subs({"x": '(1/2)'}) == 2


# 测试特定问题 15234 的函数
def test_issue_15234():
    # 定义符号 x 和 y 为实数
    x, y = symbols('x y', real=True)
    # 定义多项式 p 和替换后的 p_subbed
    p = 6*x**5 + x**4 - 4*x**3 + 4*x**2 - 2*x + 3
    p_subbed = 6*x**5 - 4*x**3 - 2*x + y**4 + 4*y**2 + 3
    # 断言：p.subs([(x**i, y**i) for i in [2, 4]]) 应该等于 p_subbed
    assert p.subs([(x**i, y**i) for i in [2, 4]]) == p_subbed
    # 重新定义符号 x 和 y 为复数
    x, y = symbols('x y', complex=True)
    # 重新定义多项式 p 和替换后的 p_subbed
    p = 6*x**5 + x**4 - 4*x**3 + 4*x**2 - 2*x + 3
    p_subbed = 6*x**5 - 4*x**3 - 2*x + y**4 + 4*y**2 + 3
    # 断言：p.subs([(x**i, y**i) for i in [2, 4]]) 应该等于 p_subbed
    assert p.subs([(x**i, y**i) for i in [2, 4]]) == p_subbed


# 测试特定问题 6976 的函数
def test_issue_6976():
    x, y = symbols('x y')
    # 断言：(sqrt(x)**3 + sqrt(x) + x + x**2).subs(sqrt(x), y) 应该等于 y**4 + y**3 + y**2 + y
    assert (sqrt(x)**3 + sqrt(x) + x + x**2).subs(sqrt(x), y) == \
        y**4 + y**3 + y**2 + y
    # 断言：(x**4 + x**3 + x**2 + x + sqrt(x)).subs(x**2, y) 应该等于 sqrt(x) + x**3 + x + y**2 + y
    assert (x**4 + x**3 + x**2 + x + sqrt(x)).subs(x**2, y) == \
        sqrt(x) + x**3 + x + y**2 + y
    # 断言：x.subs(x**3, y) 应该等于 x
    assert x.subs(x**3, y) == x
    # 断言：x.subs(x**Rational(1, 3), y) 应该等于 y**3
    assert x.subs(x**Rational(1, 3), y) == y**3

    # 更多使用非负符号的替换
    x, y = symbols('x y', nonnegative=True)
    # 断言：(x**4 + x**3 + x**2 + x + sqrt(x)).subs(x**2, y) 应该等于 y**Rational(1, 4) + y**Rational(3, 2) + sqrt(y) + y**2 + y
    assert (x**4 + x**3 + x**2 + x + sqrt(x)).subs(x**2, y) == \
        y**Rational(1, 4) + y**Rational(3, 2) + sqrt(y) + y**2 + y
    # 断言：x.subs(x**3, y) 应该等于 y**Rational(1, 3)
    assert x.subs(x**3, y) == y**Rational(1, 3)


# 测试特定问题 11746 的函数
def test_issue_11746():
    # 断言：(1/x).subs(x**2, 1) 应该等于 1/x
    assert (1/x).subs(x**2, 1) == 1/x
    # 断言：(1/(x**3)).subs(x**2, 1) 应该等于 x**(-3)
    assert (1/(x**3)).subs(x**2, 1) == x**(-3)
    # 断言：(1/(x**4)).subs(x**2, 1) 应该等于 1
    assert (1/(x**4)).subs(x**2, 1) == 1
    # 断言：(1/(x**3)).subs(x**4, 1) 应该等于 x**(-3)
    assert (1/(x**3)).subs(x**4, 1) == x**(-3)
    # 断言：(1/(y**5)).subs(x**5, 1) 应该等于 y**(-5)
    assert (1/(y**5)).subs(x**5, 1) == y**(-5)


# 测试特定问题 17823 的函数
def test_issue_17823():
    # 导入动力学符号
    from sympy.physics.mechanics import dynamicsymbols
    # 定义动力学符号 q1 和 q2
    q1, q2 = dynamicsymbols('q1, q2')
    # 定义表达式 expr
    expr = q1.diff().diff()**2*q1 + q1.diff()*q2.diff()
    # 定义替换字典 reps
    reps={q1: a, q1.diff(): a*x*y, q1.diff().diff(): z}
    # 断言：expr.subs(reps) 应该等于 a*x*y*Derivative(q2, t) + a*z**2
    assert expr.subs(reps) == a*x*y*Derivative(q2, t) + a*z**2


# 测试特定问题 19326 的函数
def test_issue_19326():
    # 定义符号 x 和 y，并应用函数到每个符
    # 断言语句，用于验证等式在特定变量取值下的结果是否符合预期
    assert eq.subs({x: 1, y: oo}) is S.NaN
    
    # 断言语句，使用带有元组的替换列表，同时替换变量并验证等式结果
    assert eq.subs([(x, 1), (y, oo)], simultaneous=True) is S.NaN
```