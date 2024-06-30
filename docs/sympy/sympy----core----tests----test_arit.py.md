# `D:\src\scipysrc\sympy\sympy\core\tests\test_arit.py`

```
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
    oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet

from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin

from itertools import product

# 定义多个符号变量
a, c, x, y, z = symbols('a,c,x,y,z')
# 定义一个正数符号变量
b = Symbol("b", positive=True)


def same_and_same_prec(a, b):
    # 检查两个 Float 对象是否值相同且精度相同
    # stricter matching for Floats
    return a == b and a._prec == b._prec


def test_bug1():
    # 测试复数实部函数 re 的行为
    assert re(x) != x
    # 计算 x 的一阶级数展开
    x.series(x, 0, 1)
    # 再次测试复数实部函数 re 的行为
    assert re(x) != x


def test_Symbol():
    # 测试符号变量的属性和基本运算
    e = a*b
    assert e == a*b
    assert a*b*b == a*b**2
    assert a*b*b + c == c + a*b**2
    assert a*b*b - c == -c + a*b**2

    # 测试不同参数下符号变量的属性
    x = Symbol('x', complex=True, real=False)
    assert x.is_imaginary is None  # 可能是虚数单位 I 或 1 + I
    x = Symbol('x', complex=True, imaginary=False)
    assert x.is_real is None  # 可能是实数 1 或 1 + I
    x = Symbol('x', real=True)
    assert x.is_complex
    x = Symbol('x', imaginary=True)
    assert x.is_complex
    x = Symbol('x', real=False, imaginary=False)
    assert x.is_complex is None  # 可能是非数字


def test_arit0():
    # 测试有理数 Rational 的基本运算
    p = Rational(5)
    e = a*b
    assert e == a*b
    e = a*b + b*a
    assert e == 2*a*b
    e = a*b + b*a + a*b + p*b*a
    assert e == 8*a*b
    e = a*b + b*a + a*b + p*b*a + a
    assert e == a + 8*a*b
    e = a + a
    assert e == 2*a
    e = a + b + a
    assert e == b + 2*a
    e = a + b*b + a + b*b
    assert e == 2*a + 2*b**2
    e = a + Rational(2) + b*b + a + b*b + p
    assert e == 7 + 2*a + 2*b**2
    e = (a + b*b + a + b*b)*p
    assert e == 5*(2*a + 2*b**2)
    e = (a*b*c + c*b*a + b*a*c)*p
    assert e == 15*a*b*c
    e = (a*b*c + c*b*a + b*a*c)*p - Rational(15)*a*b*c
    assert e == Rational(0)
    e = Rational(50)*(a - a)
    assert e == Rational(0)
    e = b*a - b - a*b + b
    assert e == Rational(0)
    e = a*b + c**p
    assert e == a*b + c**5
    e = a/b
    assert e == a*b**(-1)
    e = a*2*2
    assert e == 4*a
    e = 2 + a*2/2
    assert e == 2 + a
    # 计算表达式 e = 2 - a - 2
    e = 2 - a - 2
    # 断言：e 应该等于 -a
    assert e == -a
    
    # 计算表达式 e = 2*a*2
    e = 2*a*2
    # 断言：e 应该等于 4*a
    assert e == 4*a
    
    # 计算表达式 e = 2/a/2
    e = 2/a/2
    # 断言：e 应该等于 a 的倒数 (1/a)
    assert e == a**(-1)
    
    # 计算表达式 e = 2**a**2
    e = 2**a**2
    # 断言：e 应该等于 2 的 a 平方次幂 (2**(a**2))
    assert e == 2**(a**2)
    
    # 计算表达式 e = -(1 + a)
    e = -(1 + a)
    # 断言：e 应该等于 -1 - a
    assert e == -1 - a
    
    # 计算表达式 e = S.Half*(1 + a)
    e = S.Half*(1 + a)
    # 断言：e 应该等于 S.Half + a/2
    assert e == S.Half + a/2
def test_div():
    # 计算表达式 a/b
    e = a/b
    # 断言 e 等于 a 乘以 b 的倒数
    assert e == a*b**(-1)
    # 计算表达式 a/b + c/2
    e = a/b + c/2
    # 断言 e 等于 a 乘以 b 的倒数再加上 Rational(1)/2 乘以 c
    assert e == a*b**(-1) + Rational(1)/2*c
    # 计算表达式 (1 - b)/(b - 1)
    e = (1 - b)/(b - 1)
    # 断言 e 等于 (1 + -b)*((-1) + b)**(-1)
    assert e == (1 + -b)*((-1) + b)**(-1)


def test_pow_arit():
    n1 = Rational(1)
    n2 = Rational(2)
    n5 = Rational(5)
    # 计算表达式 a*a
    e = a*a
    # 断言 e 等于 a 的平方
    assert e == a**2
    # 计算表达式 a*a*a
    e = a*a*a
    # 断言 e 等于 a 的立方
    assert e == a**3
    # 计算表达式 a*a*a*a**Rational(6)
    e = a*a*a*a**Rational(6)
    # 断言 e 等于 a 的九次方
    assert e == a**9
    # 计算表达式 a*a*a*a**Rational(6) - a**Rational(9)
    e = a*a*a*a**Rational(6) - a**Rational(9)
    # 断言 e 等于 Rational(0)
    assert e == Rational(0)
    # 计算表达式 a**(b - b)
    e = a**(b - b)
    # 断言 e 等于 Rational(1)
    assert e == Rational(1)
    # 计算表达式 (a + Rational(1) - a)**b
    e = (a + Rational(1) - a)**b
    # 断言 e 等于 Rational(1)
    assert e == Rational(1)

    # 计算表达式 (a + b + c)**n2
    e = (a + b + c)**n2
    # 断言 e 等于 (a + b + c) 的平方
    assert e == (a + b + c)**2
    # 断言 e 展开后等于 2*b*c + 2*a*c + 2*a*b + a**2 + c**2 + b**2
    assert e.expand() == 2*b*c + 2*a*c + 2*a*b + a**2 + c**2 + b**2

    # 计算表达式 (a + b)**n2
    e = (a + b)**n2
    # 断言 e 等于 (a + b) 的平方
    assert e == (a + b)**2
    # 断言 e 展开后等于 2*a*b + a**2 + b**2
    assert e.expand() == 2*a*b + a**2 + b**2

    # 计算表达式 (a + b)**(n1/n2)
    e = (a + b)**(n1/n2)
    # 断言 e 等于 sqrt(a + b)
    assert e == sqrt(a + b)
    # 断言 e 展开后等于 sqrt(a + b)
    assert e.expand() == sqrt(a + b)

    n = n5**(n1/n2)
    # 断言 n 等于 sqrt(5)
    assert n == sqrt(5)
    # 计算表达式 n*a*b - n*b*a
    e = n*a*b - n*b*a
    # 断言 e 等于 Rational(0)
    assert e == Rational(0)
    # 计算表达式 n*a*b + n*b*a
    e = n*a*b + n*b*a
    # 断言 e 等于 2*a*b*sqrt(5)
    assert e == 2*a*b*sqrt(5)
    # 断言 e 对 a 的偏导数等于 2*b*sqrt(5)
    assert e.diff(a) == 2*b*sqrt(5)
    # 断言 e 对 a 的偏导数等于 2*b*sqrt(5)
    assert e.diff(a) == 2*b*sqrt(5)
    # 计算表达式 a/b**2
    e = a/b**2
    # 断言 e 等于 a 乘以 b 的负二次方
    assert e == a*b**(-2)

    # 断言 sqrt(2*(1 + sqrt(2))) 等于 (2*(1 + 2**S.Half))**S.Half
    assert sqrt(2*(1 + sqrt(2))) == (2*(1 + 2**S.Half))**S.Half

    x = Symbol('x')
    y = Symbol('y')

    # 断言 ((x*y)**3).expand() 等于 y**3 * x**3
    assert ((x*y)**3).expand() == y**3 * x**3
    # 断言 ((x*y)**-3).expand() 等于 y**-3 * x**-3
    assert ((x*y)**-3).expand() == y**-3 * x**-3

    # 断言 (x**5*(3*x)**(3)).expand() 等于 27 * x**8
    assert (x**5*(3*x)**(3)).expand() == 27 * x**8
    # 断言 (x**5*(-3*x)**(3)).expand() 等于 -27 * x**8
    assert (x**5*(-3*x)**(3)).expand() == -27 * x**8
    # 断言 (x**5*(3*x)**(-3)).expand() 等于 x**2 * Rational(1, 27)
    assert (x**5*(3*x)**(-3)).expand() == x**2 * Rational(1, 27)
    # 断言 (x**5*(-3*x)**(-3)).expand() 等于 x**2 * Rational(-1, 27)
    assert (x**5*(-3*x)**(-3)).expand() == x**2 * Rational(-1, 27)

    # expand_power_exp
    _x = Symbol('x', zero=False)
    _y = Symbol('y', zero=False)
    # 断言 _x**(y**(x + exp(x + y)) + z).expand(deep=False) 等于 _x**z*_x**(y**(x + exp(x + y)))
    assert (_x**(y**(x + exp(x + y)) + z)).expand(deep=False) == \
        _x**z*_x**(y**(x + exp(x + y)))
    # 断言 _x**(_y**(x + exp(x + y)) + z).expand() 等于 _x**z*_x**(_y**x*_y**(exp(x)*exp(y)))

    n = Symbol('n', even=False)
    k = Symbol('k', even=True)
    o = Symbol('o', odd=True)

    # 断言 unchanged(Pow, -1, x)
    assert unchanged(Pow, -1, x)
    # 断言 unchanged(Pow, -1, n)
    assert unchanged(Pow, -1, n)
    # 断言 (-2)**k 等于 2**k
    assert (-2)**k == 2**k
    # 断言 (-1)**k 等于 1
    assert (-1)**k == 1
    # 断言 (-1)**o 等于 -1
    assert (-1)**o == -1


def test_pow2():
    # 断言 (-x)**Rational(2, 3) 不等于 x**Rational(2, 3)
    assert (-x)**Rational(2, 3) != x**Rational(2, 3)
    # 断言 (-x)**Rational(5, 7) 不等于 -x**Rational(5, 7)
    assert (-x)**Rational(5, 7) != -x**Rational(5, 7)
    # 断言 ((-x)**2)**Rational(1, 3) 不等于 ((-x)**Rational(1, 3))**2
    assert ((-x)**2)**Rational(1, 3) != ((-x)**Rational(1, 3))**2
    # 断
    # 断言：验证 pow(x, y, z) 函数的返回结果等于 (x**y) % z
    assert pow(x, y, z) == x**y % z
    
    # 预期抛出 TypeError 异常，因为第二个参数 "13" 不是整数类型
    raises(TypeError, lambda: pow(S(4), "13", 497))
    
    # 预期抛出 TypeError 异常，因为第三个参数 "497" 不是整数类型
    raises(TypeError, lambda: pow(S(4), 13, "497"))
# 定义一个测试函数，用于验证幂运算的特定数学关系
def test_pow_E():
    # 断言：2^(y/log(2)) 应该等于 S.Exp1^y
    assert 2**(y/log(2)) == S.Exp1**y
    # 断言：2^(y/log(2)/3) 应该等于 S.Exp1^(y/3)
    assert 2**(y/log(2)/3) == S.Exp1**(y/3)
    # 断言：3^(1/log(-3)) 不应等于 S.Exp1
    assert 3**(1/log(-3)) != S.Exp1
    # 断言：(3 + 2*I)^(1/(log(-3 - 2*I) + I*pi)) 应等于 S.Exp1
    assert (3 + 2*I)**(1/(log(-3 - 2*I) + I*pi)) == S.Exp1
    # 断言：(4 + 2*I)^(1/(log(-4 - 2*I) + I*pi)) 应等于 S.Exp1
    assert (4 + 2*I)**(1/(log(-4 - 2*I) + I*pi)) == S.Exp1
    # 断言：(3 + 2*I)^(1/(log(-3 - 2*I, 3)/2 + I*pi/log(3)/2)) 应等于 9
    assert (3 + 2*I)**(1/(log(-3 - 2*I, 3)/2 + I*pi/log(3)/2)) == 9
    # 断言：(3 + 2*I)^(1/(log(3 + 2*I, 3)/2)) 应等于 9
    assert (3 + 2*I)**(1/(log(3 + 2*I, 3)/2)) == 9
    # 循环：每次测试时，使用不同的随机值确认这个恒等式成立
    while 1:
        b = x._random()
        r, i = b.as_real_imag()
        # 如果虚部存在，则跳出循环
        if i:
            break
    # 断言：数值验证 b^(1/(log(-b) + sign(i)*I*pi).n()) 应等于 S.Exp1
    assert verify_numerically(b**(1/(log(-b) + sign(i)*I*pi).n()), S.Exp1)


# 定义一个测试函数，验证幂运算在特定案例中的问题
def test_pow_issue_3516():
    # 断言：4^(1/4) 应该等于 sqrt(2)
    assert 4**Rational(1, 4) == sqrt(2)


# 定义一个测试函数，验证虚数幂运算的各种情况
def test_pow_im():
    # 对于一系列 m 和 d 的组合
    for m in (-2, -1, 2):
        for d in (3, 4, 5):
            b = m*I
            # 对于每个分数 i/d
            for i in range(1, 4*d + 1):
                e = Rational(i, d)
                # 断言：(b**e - b.n()**e.n()).n(2, chop=1e-10) 应该接近于 0
                assert (b**e - b.n()**e.n()).n(2, chop=1e-10) == 0

    # 特定的 e 值
    e = Rational(7, 3)
    # 断言：(2*x*I)**e 应该等于 4*2**(1/3)*(I*x)**e，与Wolfram Alpha一致
    assert (2*x*I)**e == 4*2**Rational(1, 3)*(I*x)**e
    # 定义一个虚数符号
    im = symbols('im', imaginary=True)
    # 断言：(2*im*I)**e 应该等于 4*2**(1/3)*(I*im)**e
    assert (2*im*I)**e == 4*2**Rational(1, 3)*(I*im)**e

    # 指定一组参数
    args = [I, I, I, I, 2]
    e = Rational(1, 3)
    ans = 2**e
    # 断言：Mul(*args, evaluate=False)**e 应该等于 ans
    assert Mul(*args, evaluate=False)**e == ans
    # 断言：Mul(*args)**e 应该等于 ans
    assert Mul(*args)**e == ans

    # 更新参数列表
    args = [I, I, I, 2]
    e = Rational(1, 3)
    ans = 2**e*(-I)**e
    # 断言：Mul(*args, evaluate=False)**e 应该等于 ans
    assert Mul(*args, evaluate=False)**e == ans
    # 断言：Mul(*args)**e 应该等于 ans
    assert Mul(*args)**e == ans

    # 继续更新参数列表
    args.append(-3)
    ans = (6*I)**e
    # 断言：Mul(*args, evaluate=False)**e 应该等于 ans
    assert Mul(*args, evaluate=False)**e == ans
    # 断言：Mul(*args)**e 应该等于 ans
    assert Mul(*args)**e == ans

    # 再次更新参数列表
    args.append(-1)
    ans = (-6*I)**e
    # 断言：Mul(*args, evaluate=False)**e 应该等于 ans
    assert Mul(*args, evaluate=False)**e == ans
    # 断言：Mul(*args)**e 应该等于 ans
    assert Mul(*args)**e == ans

    # 对于新的参数列表
    args = [I, I, 2]
    e = Rational(1, 3)
    ans = (-2)**e
    # 断言：Mul(*args, evaluate=False)**e 应该等于 ans
    assert Mul(*args, evaluate=False)**e == ans
    # 断言：Mul(*args)**e 应该等于 ans
    assert Mul(*args)**e == ans

    # 继续更新参数列表
    args.append(-3)
    ans = (6)**e
    # 断言：Mul(*args, evaluate=False)**e 应该等于 ans
    assert Mul(*args, evaluate=False)**e == ans
    # 断言：Mul(*args)**e 应该等于 ans
    assert Mul(*args)**e == ans

    # 再次更新参数列表
    args.append(-1)
    ans = (-6)**e
    # 断言：Mul(*args, evaluate=False)**e 应该等于 ans
    assert Mul(*args, evaluate=False)**e == ans
    # 断言：Mul(*args)**e 应该等于 ans
    assert Mul(*args)**e == ans

    # 最后一组特定的断言
    # 断言：Mul(Pow(-1, Rational(3, 2), evaluate=False), I, I) 应该等于 I
    assert Mul(Pow(-1, Rational(3, 2), evaluate=False), I, I) == I
    # 断言：Mul(I*Pow(I, S.Half, evaluate=False)) 应该等于 sqrt(I)*I
    assert Mul(I*Pow(I, S.Half, evaluate=False)) == sqrt(I)*I


# 定义一个测试函数，验证实数乘法的身份
def test_real_mul():
    # 断言：Float(0) * pi * x 应该等于 0
    assert Float(0) * pi * x == 0
    # 断言：set((Float(1) * pi * x).args) 应该包含 {Float(1), pi, x}
    assert set((Float(1) * pi * x).args) == {Float(1), pi, x}


# 定义一个测试函数，验证非交换乘法的情况
def test_ncmul():
    # 定义非交换符号 A, B, C
    A = Symbol("A", commutative=False)
    B = Symbol("B", commutative=False)
    C = Symbol("C", commutative=False)
    # 断言：A*B 不应等于 B*A
    assert A*B != B*A
    # 断言：A*B*C 不应等于 C*B*A
    assert A*B*C != C*B*A
    # 断言：A*b*B*3*C 应等于 3*b*A*B*C
    assert A*b*B*3*C == 3*b*A*B*C
    # 断言：A*b*B*
    # 使用 Mul 类创建一个有理数对象 m，其分子为 1，分母为 2，并断言 m 是 Rational 类的实例，且其分子为 2，分母为 1
    m = Mul(1, 2)
    assert isinstance(m, Rational) and m.p == 2 and m.q == 1

    # 使用 Mul 类创建一个对象 m，参数为 1 和 2，且不进行求值（evaluate=False），断言 m 是 Mul 类的实例，并且其参数为 (1, 2)
    m = Mul(1, 2, evaluate=False)
    assert isinstance(m, Mul) and m.args == (1, 2)

    # 使用 Mul 类创建一个对象 m，参数为 0 和 1，断言 m 是 S.Zero，即 SymPy 中的零值对象
    m = Mul(0, 1)
    assert m is S.Zero

    # 使用 Mul 类创建一个对象 m，参数为 0 和 1，且不进行求值（evaluate=False），断言 m 是 Mul 类的实例，并且其参数为 (0, 1)
    m = Mul(0, 1, evaluate=False)
    assert isinstance(m, Mul) and m.args == (0, 1)

    # 使用 Add 类创建一个对象 m，参数为 0 和 1，断言 m 是 S.One，即 SymPy 中的单位值对象
    m = Add(0, 1)
    assert m is S.One

    # 使用 Add 类创建一个对象 m，参数为 0 和 1，且不进行求值（evaluate=False），断言 m 是 Add 类的实例，并且其参数为 (0, 1)
    m = Add(0, 1, evaluate=False)
    assert isinstance(m, Add) and m.args == (0, 1)
# 定义一个测试函数 test_ncpow，用于测试非交换符号的幂运算性质
def test_ncpow():
    # 创建不可交换符号 x, y, z
    x = Symbol('x', commutative=False)
    y = Symbol('y', commutative=False)
    z = Symbol('z', commutative=False)
    # 创建可交换符号 a, b, c
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')

    # 测试非交换性质：(x**2)*(y**2) 不等于 (y**2)*(x**2)
    assert (x**2)*(y**2) != (y**2)*(x**2)
    # 测试幂运算与乘法顺序的影响：(x**-2)*y 不等于 y*(x**2)
    assert (x**-2)*y != y*(x**2)
    # 测试指数的加法与乘法的关系：2**x*2**y 不等于 2**(x + y)
    assert 2**x*2**y != 2**(x + y)
    # 测试多个指数相加的情况：2**x*2**y*2**z 不等于 2**(x + y + z)
    assert 2**x*2**y*2**z != 2**(x + y + z)
    # 测试指数幂的乘法结合性质：2**x*2**(2*x) 等于 2**(3*x)
    assert 2**x*2**(2*x) == 2**(3*x)
    # 测试多个指数幂的乘法结合性质：2**x*2**(2*x)*2**x 等于 2**(4*x)
    assert 2**x*2**(2*x)*2**x == 2**(4*x)
    # 测试指数幂运算与指数函数的交换性质：exp(x)*exp(y) 不等于 exp(y)*exp(x)
    assert exp(x)*exp(y) != exp(y)*exp(x)
    # 测试多个指数函数的交换性质：exp(x)*exp(y)*exp(z) 不等于 exp(y)*exp(x)*exp(z)
    assert exp(x)*exp(y)*exp(z) != exp(y)*exp(x)*exp(z)
    # 测试指数幂运算与指数函数的加法关系：exp(x)*exp(y)*exp(z) 不等于 exp(x + y + z)
    assert exp(x)*exp(y)*exp(z) != exp(x + y + z)
    # 测试指数幂的乘法规则：x**a*x**b 等于 x**(a + b)
    assert x**a*x**b == x**(a + b)
    # 测试多个指数幂的乘法规则：x**a*x**b*x**c 等于 x**(a + b + c)
    assert x**a*x**b*x**c == x**(a + b + c)
    # 测试指数幂的乘法结合性质：x**3*x**4 等于 x**7
    assert x**3*x**4 == x**7
    # 测试多个指数幂的乘法结合性质：x**3*x**4*x**2 等于 x**9
    assert x**3*x**4*x**2 == x**9
    # 测试指数幂与指数乘积的加法关系：x**a*x**(4*a) 等于 x**(5*a)
    assert x**a*x**(4*a) == x**(5*a)
    # 测试多个指数幂与指数乘积的加法关系：x**a*x**(4*a)*x**a 等于 x**(6*a)


# 定义一个测试函数 test_powerbug，用于测试幂运算中的特殊情况
def test_powerbug():
    # 创建符号 x
    x = Symbol("x")
    # 测试幂运算中正负号的影响：x**1 不等于 (-x)**1
    assert x**1 != (-x)**1
    # 测试幂运算中正负号的影响：x**2 等于 (-x)**2
    assert x**2 == (-x)**2
    # 测试幂运算中正负号的影响：x**3 不等于 (-x)**3
    assert x**3 != (-x)**3
    # 测试幂运算中正负号的影响：x**4 等于 (-x)**4
    assert x**4 == (-x)**4
    # 测试幂运算中正负号的影响：x**5 不等于 (-x)**5
    assert x**5 != (-x)**5
    # 测试幂运算中正负号的影响：x**6 等于 (-x)**6
    assert x**6 == (-x)**6
    # 测试大指数幂运算中正负号的影响：x**128 等于 (-x)**128
    assert x**128 == (-x)**128
    # 测试大指数幂运算中正负号的影响：x**129 不等于 (-x)**129
    assert x**129 != (-x)**129
    # 测试乘积中的特殊情况：(2*x)**2 等于 (-2*x)**2
    assert (2*x)**2 == (-2*x)**2


# 定义一个测试函数 test_Mul_doesnt_expand_exp，测试乘积中指数函数的特殊情况
def test_Mul_doesnt_expand_exp():
    # 创建符号 x, y
    x = Symbol('x')
    y = Symbol('y')
    # 测试 Mul 函数在指数函数 exp 中的不扩展性质
    assert unchanged(Mul, exp(x), exp(y))
    # 测试 Mul 函数在指数为常数时的不扩展性质
    assert unchanged(Mul, 2**x, 2**y)
    # 测试指数幂运算中的乘法规则：x**2*x**3 等于 x**5
    assert x**2*x**3 == x**5
    # 测试不同底数的指数幂乘法规则：2**x*3**x 等于 6**x
    assert 2**x*3**x == 6**x
    # 测试指数幂乘法中的幂运算规则：x**(y)*x**(2*y) 等于 x**(3*y)
    assert x**(y)*x**(2*y) == x**(3*y)
    # 测试平方根的乘积：sqrt(2)*sqrt(2) 等于 2
    assert sqrt(2)*sqrt(2) == 2
    # 测试指数与有理数幂乘积的指数规则：2**x*2**(2*x) 等于 2**(3*x)
    assert 2**x*2**(2*x) == 2**(3*x)
    # 测试不同底数的混合幂乘积的乘法规则：sqrt(2)*2**Rational(1, 4)*5**Rational(3, 4) 等于 10**Rational(3, 4)
    assert sqrt(2)*2**Rational(1, 4)*5**Rational(3, 4) == 10**Rational(3, 4)
    # 测试指数幂的混合运算：(x**(-log(5)/log(3))*x)/(x*x**( - log(5)/log(3))) 等于 sympify(1)
    assert (x**(-log(5)/log(3))*x)/(x*x**( - log(5)/log(3))) == sympify(1)


# 定义一个测试函数 test_Mul_is_integer，测试 Mul 函数中的整数性质
def test_Mul_is_integer():
    # 创建整数、有理数、质数、偶数、奇数、非零整数的符号
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True)
    nr = Symbol('nr', rational=False)
    ir = Symbol('ir', irrational=True)
    nz = Symbol('nz', integer=True, zero=False)
    e = Symbol
    # 定义一个表达式 e_20161，表示为 -1 * (1 * 2**(-1)), 禁止自动求值
    e_20161 = Mul(-1, Mul(1, Pow(2, -1, evaluate=False), evaluate=False), evaluate=False)
    
    # 断言 e_20161 不是整数，这是因为在不进行求值的情况下，expand(e_20161) 结果是 -1/2
    assert e_20161.is_integer is not True
# 定义一个用于测试加法和乘法的函数，验证符号的整数属性
def test_Add_Mul_is_integer():
    # 创建一个符号变量 x
    x = Symbol('x')

    # 创建整数符号变量 k, n，以及非整数符号变量 nk, nr, 以及带有特定属性 zero=False 的整数符号变量 nz
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True)
    nk = Symbol('nk', integer=False)
    nr = Symbol('nr', rational=False)
    nz = Symbol('nz', integer=True, zero=False)

    # 断言对非整数符号的取反结果为 None
    assert (-nk).is_integer is None
    # 断言对非有理数符号的取反结果为 False
    assert (-nr).is_integer is False
    # 断言对于整数符号 k 的两倍是整数
    assert (2*k).is_integer is True
    # 断言对于整数符号 k 的取反是整数
    assert (-k).is_integer is True

    # 断言整数符号 k 加上非整数符号 nk 不是整数
    assert (k + nk).is_integer is False
    # 断言整数符号 k 加上整数符号 n 是整数
    assert (k + n).is_integer is True
    # 断言整数符号 k 加上符号变量 x 结果未知
    assert (k + x).is_integer is None
    # 断言整数符号 k 加上整数符号 n 乘以符号变量 x 结果未知
    assert (k + n*x).is_integer is None
    # 断言整数符号 k 加上 n/3 结果未知
    assert (k + n/3).is_integer is None
    # 断言整数符号 k 加上 nz/3 结果未知，因为 nz 的 zero=False 特性
    assert (k + nz/3).is_integer is None
    # 断言整数符号 k 加上非有理数符号 nr/3 不是整数
    assert (k + nr/3).is_integer is False

    # 断言 (1 + sqrt(3))*(1 - sqrt(3)) 不是 False
    assert ((1 + sqrt(3))*(-sqrt(3) + 1)).is_integer is not False
    # 断言 1 + (1 + sqrt(3))*(1 - sqrt(3)) 不是 False
    assert (1 + (1 + sqrt(3))*(-sqrt(3) + 1)).is_integer is not False


# 定义一个用于测试乘法是否有限的函数，验证符号的有限属性
def test_Add_Mul_is_finite():
    # 创建一个拓展实数且无限的符号变量 x
    x = Symbol('x', extended_real=True, finite=False)

    # 断言 sin(x) 是有限的
    assert sin(x).is_finite is True
    # 断言 x*sin(x) 的有限性未知
    assert (x*sin(x)).is_finite is None
    # 断言 x*atan(x) 不是有限的
    assert (x*atan(x)).is_finite is False
    # 断言 1024*sin(x) 是有限的
    assert (1024*sin(x)).is_finite is True
    # 断言 sin(x)*exp(x) 的有限性未知
    assert (sin(x)*exp(x)).is_finite is None
    # 断言 sin(x)*cos(x) 是有限的
    assert (sin(x)*cos(x)).is_finite is True
    # 断言 x*sin(x)*exp(x) 的有限性未知
    assert (x*sin(x)*exp(x)).is_finite is None

    # 断言 sin(x) - 67 是有限的
    assert (sin(x) - 67).is_finite is True
    # 断言 sin(x) + exp(x) 不是 True
    assert (sin(x) + exp(x)).is_finite is not True
    # 断言 1 + x 不是有限的
    assert (1 + x).is_finite is False
    # 断言 1 + x**2 + (1 + x)*(1 - x) 的有限性未知
    assert (1 + x**2 + (1 + x)*(1 - x)).is_finite is None
    # 断言 sqrt(2)*(1 + x) 不是有限的
    assert (sqrt(2)*(1 + x)).is_finite is False
    # 断言 sqrt(2)*(1 + x)*(1 - x) 不是有限的
    assert (sqrt(2)*(1 + x)*(1 - x)).is_finite is False


# 定义一个用于测试乘法是否是偶数或奇数的函数，验证符号的奇偶性属性
def test_Mul_is_even_odd():
    # 创建整数符号变量 x 和 y
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)

    # 创建具有奇数属性的符号变量 k 和 n，以及具有偶数属性的符号变量 m
    k = Symbol('k', odd=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', even=True)

    # 断言 2*x 是偶数
    assert (2*x).is_even is True
    # 断言 2*x 不是奇数
    assert (2*x).is_odd is False

    # 断言 3*x 的奇偶性未知
    assert (3*x).is_even is None
    assert (3*x).is_odd is None

    # 断言 k/3 不是整数，也不是偶数或奇数
    assert (k/3).is_integer is None
    assert (k/3).is_even is None
    assert (k/3).is_odd is None

    # 断言 2*n 是偶数
    assert (2*n).is_even is True
    # 断言 2*n 不是奇数
    assert (2*n).is_odd is False

    # 断言 2*m 是偶数
    assert (2*m).is_even is True
    # 断言 2*m 不是奇数
    assert (2*m).is_odd is False

    # 断言 -n 是奇数
    assert (-n).is_even is False
    assert (-n).is_odd is True

    # 断言 k*n 是奇数
    assert (k*n).is_even is False
    assert (k*n).is_odd is True

    # 断言 k*m 是偶数
    assert (k*m).is_even is True
    assert (k*m).is_odd is False

    # 断言 k*n*m 是偶数
    assert (k*n*m).is_even is True
    assert (k*n*m).is_odd is False

    # 断言 k*m*x 是偶数
    assert (k*m*x).is_even is True
    assert (k*m*x).is_odd is False

    # issue 6791:
    # 断言 x/2 的整数性未知
    assert (x/2).is_integer is None
    # 断言 k/2 不是整数
    assert (k/2).is_integer is False
    # 断言 m/2 是整数
    assert (m/2).is_integer is True

    # 断言 x*y 的偶数性未知
    assert (x*y).is_even is None
    # 断言 x*x 的偶数性未知
    assert (x*x).is_even is None
    # 断言 x*(x + k) 是偶数
    assert (x*(x + k)).is_even is True
    # 断言 x*(x + m) 的偶数性未知
    assert (x*(x + m)).is_even is None

    # 断言 x*y 的奇数性未知
    assert (x*y).is_odd is None
    # 断言 x*x 的奇数性未知
    assert (x*x).is_odd is None
    # 断言 x*(x + k) 不是奇数
    assert (x*(x + k)).is_odd is False
    # 断言 x*(x + m) 的奇数性未知
    assert (x*(x + m)).is_odd is None

    # issue 8648
    # 断言 m**2/2 是偶数
    assert (m**2/2).is_even
    # 断言 m**
def test_evenness_in_ternary_integer_product_with_odd():
    # 测试奇数推断与项次序无关。
    # 在测试时，项的顺序取决于SymPy的符号顺序，因此尝试通过修改符号名称来强制不同的顺序。
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    k = Symbol('k', odd=True)
    # 断言 x*y*(y + k) 是否为偶数
    assert (x*y*(y + k)).is_even is True
    # 断言 y*x*(x + k) 是否为偶数
    assert (y*x*(x + k)).is_even is True


def test_evenness_in_ternary_integer_product_with_even():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    m = Symbol('m', even=True)
    # 断言 x*y*(y + m) 是否为偶数
    assert (x*y*(y + m)).is_even is None


@XFAIL
def test_oddness_in_ternary_integer_product_with_odd():
    # 测试奇数推断与项次序无关。
    # 在测试时，项的顺序取决于SymPy的符号顺序，因此尝试通过修改符号名称来强制不同的顺序。
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    k = Symbol('k', odd=True)
    # 断言 x*y*(y + k) 是否为奇数
    assert (x*y*(y + k)).is_odd is False
    # 断言 y*x*(x + k) 是否为奇数
    assert (y*x*(x + k)).is_odd is False


def test_oddness_in_ternary_integer_product_with_even():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    m = Symbol('m', even=True)
    # 断言 x*y*(y + m) 是否为奇数
    assert (x*y*(y + m)).is_odd is None


def test_Mul_is_rational():
    x = Symbol('x')
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True, nonzero=True)

    # 断言 n/m 是否为有理数
    assert (n/m).is_rational is True
    # 断言 x/pi 是否为有理数
    assert (x/pi).is_rational is None
    # 断言 x/n 是否为有理数
    assert (x/n).is_rational is None
    # 断言 m/pi 是否为有理数
    assert (m/pi).is_rational is False

    r = Symbol('r', rational=True)
    # 断言 pi*r 是否为有理数
    assert (pi*r).is_rational is None

    # issue 8008
    z = Symbol('z', zero=True)
    i = Symbol('i', imaginary=True)
    # 断言 z*i 是否为有理数
    assert (z*i).is_rational is True
    bi = Symbol('i', imaginary=True, finite=True)
    # 断言 z*bi 是否为零
    assert (z*bi).is_zero is True


def test_Add_is_rational():
    x = Symbol('x')
    n = Symbol('n', rational=True)
    m = Symbol('m', rational=True)

    # 断言 n + m 是否为有理数
    assert (n + m).is_rational is True
    # 断言 x + pi 是否为有理数
    assert (x + pi).is_rational is None
    # 断言 x + n 是否为有理数
    assert (x + n).is_rational is None
    # 断言 n + pi 是否为有理数
    assert (n + pi).is_rational is False


def test_Add_is_even_odd():
    x = Symbol('x', integer=True)

    k = Symbol('k', odd=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', even=True)

    # 断言 k + 7 是否为偶数
    assert (k + 7).is_even is True
    # 断言 k + 7 是否为奇数
    assert (k + 7).is_odd is False

    # 断言 -k + 7 是否为偶数
    assert (-k + 7).is_even is True
    # 断言 -k + 7 是否为奇数
    assert (-k + 7).is_odd is False

    # 断言 k - 12 是否为偶数
    assert (k - 12).is_even is False
    # 断言 k - 12 是否为奇数
    assert (k - 12).is_odd is True

    # 断言 -k - 12 是否为偶数
    assert (-k - 12).is_even is False
    # 断言 -k - 12 是否为奇数
    assert (-k - 12).is_odd is True

    # 断言 k + n 是否为偶数
    assert (k + n).is_even is True
    # 断言 k + n 是否为奇数
    assert (k + n).is_odd is False

    # 断言 k + m 是否为偶数
    assert (k + m).is_even is False
    # 断言 k + m 是否为奇数
    assert (k + m).is_odd is True

    # 断言 k + n + m 是否为偶数
    assert (k + n + m).is_even is True
    # 断言 k + n + m 是否为奇数
    assert (k + n + m).is_odd is False

    # 断言 k + n + x + m 是否为偶数
    assert (k + n + x + m).is_even is None
    # 断言 k + n + x + m 是否为奇数
    assert (k + n + x + m).is_odd is None


def test_Mul_is_negative_positive():
    x = Symbol('x', real=True)
    # 创建一个复数类型的符号 'y'
    y = Symbol('y', extended_real=False, complex=True)
    # 创建一个零值的符号 'z'
    z = Symbol('z', zero=True)

    # 计算 2*z 的结果，并进行断言验证
    e = 2*z
    assert e.is_Mul and e.is_positive is False and e.is_negative is False

    # 创建一个负数的符号 'neg' 和一个正数的符号 'pos'，以及非负数和非正数的符号 'nneg' 和 'npos'

    # 断言负数符号 'neg' 的属性
    assert neg.is_negative is True
    assert (-neg).is_negative is False
    assert (2*neg).is_negative is True

    # 断言正数符号 'pos' 的属性
    assert (2*pos)._eval_is_extended_negative() is False
    assert (2*pos).is_negative is False

    # 断言负数符号 'pos' 的属性
    assert pos.is_negative is False
    assert (-pos).is_negative is True
    assert (2*pos).is_negative is False

    # 断言乘积 'pos*neg' 的属性
    assert (pos*neg).is_negative is True
    assert (2*pos*neg).is_negative is True
    assert (-pos*neg).is_negative is False
    assert (pos*neg*y).is_negative is False     # y.is_real=F;  !real -> !neg

    # 断言非负数符号 'nneg' 的属性
    assert nneg.is_negative is False
    assert (-nneg).is_negative is None
    assert (2*nneg).is_negative is False

    # 断言非正数符号 'npos' 的属性
    assert npos.is_negative is None
    assert (-npos).is_negative is False
    assert (2*npos).is_negative is None

    # 断言乘积 'nneg*npos' 的属性
    assert (nneg*npos).is_negative is None

    # 断言乘积 'neg*nneg' 和 'neg*npos' 的属性
    assert (neg*nneg).is_negative is None
    assert (neg*npos).is_negative is False

    # 断言乘积 'pos*nneg' 和 'pos*npos' 的属性
    assert (pos*nneg).is_negative is False
    assert (pos*npos).is_negative is None

    # 断言乘积 'npos*neg*nneg' 和 'npos*pos*nneg' 的属性
    assert (npos*neg*nneg).is_negative is False
    assert (npos*pos*nneg).is_negative is None

    # 断言乘积 '-npos*neg*nneg' 和 '-npos*pos*nneg' 的属性
    assert (-npos*neg*nneg).is_negative is None
    assert (-npos*pos*nneg).is_negative is False

    # 断言乘积 '17*npos*neg*nneg' 和 '17*npos*pos*nneg' 的属性
    assert (17*npos*neg*nneg).is_negative is False
    assert (17*npos*pos*nneg).is_negative is None

    # 断言乘积 'neg*npos*pos*nneg' 的属性
    assert (neg*npos*pos*nneg).is_negative is False

    # 断言乘积 'x*neg' 和 'nneg*npos*pos*x*neg' 的属性
    assert (x*neg).is_negative is None
    assert (nneg*npos*pos*x*neg).is_negative is None

    # 断言负数符号 'neg' 的正数属性
    assert neg.is_positive is False
    assert (-neg).is_positive is True
    assert (2*neg).is_positive is False

    # 断言正数符号 'pos' 的正数属性
    assert pos.is_positive is True
    assert (-pos).is_positive is False
    assert (2*pos).is_positive is True

    # 断言乘积 'pos*neg' 的正数属性
    assert (pos*neg).is_positive is False
    assert (2*pos*neg).is_positive is False
    assert (-pos*neg).is_positive is True
    assert (-pos*neg*y).is_positive is False    # y.is_real=F;  !real -> !neg

    # 断言非负数符号 'nneg' 的正数属性
    assert nneg.is_positive is None
    assert (-nneg).is_positive is False
    assert (2*nneg).is_positive is None

    # 断言非正数符号 'npos' 的正数属性
    assert npos.is_positive is False
    assert (-npos).is_positive is None
    assert (2*npos).is_positive is False

    # 断言乘积 'nneg*npos' 的正数属性
    assert (nneg*npos).is_positive is False

    # 断言乘积 'neg*nneg' 和 'neg*npos' 的正数属性
    assert (neg*nneg).is_positive is False
    assert (neg*npos).is_positive is None

    # 断言乘积 'pos*nneg' 和 'pos*npos' 的正数属性
    assert (pos*nneg).is_positive is None
    assert (pos*npos).is_positive is False

    # 断言乘积 'npos*neg*nneg' 和 'npos*pos*nneg' 的正数属性
    assert (npos*neg*nneg).is_positive is None
    assert (npos*pos*nneg).is_positive is False

    # 断言乘积 '-npos*neg*nneg' 和 '-npos*pos*nneg' 的正数属性
    assert (-npos*neg*nneg).is_positive is False
    assert (-npos*pos*nneg).is_positive is None

    # 断言乘积 '17*npos*neg*nneg' 和 '17*npos*pos*nneg' 的正数属性
    assert (17*npos*neg*nneg).is_positive is None
    assert (17*npos*pos*nneg).is_positive is False
    # 断言：验证四个变量的乘积是否为正数，如果无法确定则返回 None
    assert (neg*npos*pos*nneg).is_positive is None
    
    # 断言：验证两个变量的乘积是否为正数，如果无法确定则返回 None
    assert (x*neg).is_positive is None
    
    # 断言：验证五个变量的乘积是否为正数，如果无法确定则返回 None
    assert (nneg*npos*pos*x*neg).is_positive is None
# 定义一个测试函数，用于测试乘法运算中符号属性的组合情况
def test_Mul_is_negative_positive_2():
    # 创建符号变量 a, b, c, d，限定其非负或非正属性
    a = Symbol('a', nonnegative=True)
    b = Symbol('b', nonnegative=True)
    c = Symbol('c', nonpositive=True)
    d = Symbol('d', nonpositive=True)

    # 断言乘积 a*b 的非负性为真
    assert (a*b).is_nonnegative is True
    # 断言乘积 a*b 的负性为假
    assert (a*b).is_negative is False
    # 断言乘积 a*b 的零性为不确定
    assert (a*b).is_zero is None
    # 断言乘积 a*b 的正性为不确定
    assert (a*b).is_positive is None

    # 断言乘积 c*d 的非负性为真
    assert (c*d).is_nonnegative is True
    # 断言乘积 c*d 的负性为假
    assert (c*d).is_negative is False
    # 断言乘积 c*d 的零性为不确定
    assert (c*d).is_zero is None
    # 断言乘积 c*d 的正性为不确定
    assert (c*d).is_positive is None

    # 断言乘积 a*c 的非正性为真
    assert (a*c).is_nonpositive is True
    # 断言乘积 a*c 的正性为假
    assert (a*c).is_positive is False
    # 断言乘积 a*c 的零性为不确定
    assert (a*c).is_zero is None
    # 断言乘积 a*c 的负性为不确定
    assert (a*c).is_negative is None


# 定义一个测试函数，用于测试乘法运算中非正和非负符号属性的组合情况
def test_Mul_is_nonpositive_nonnegative():
    # 创建实数域内的符号变量 x
    x = Symbol('x', real=True)

    # 创建符号变量 k, n, u, v，限定其负、正、非负和非正属性
    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)

    # 断言 k 的非正性为真
    assert k.is_nonpositive is True
    # 断言 -k 的非正性为假
    assert (-k).is_nonpositive is False
    # 断言 2*k 的非正性为真
    assert (2*k).is_nonpositive is True

    # 断言 n 的非正性为假
    assert n.is_nonpositive is False
    # 断言 -n 的非正性为真
    assert (-n).is_nonpositive is True
    # 断言 2*n 的非正性为假
    assert (2*n).is_nonpositive is False

    # 断言 n*k 的非正性为真
    assert (n*k).is_nonpositive is True
    # 断言 2*n*k 的非正性为真
    assert (2*n*k).is_nonpositive is True
    # 断言 -n*k 的非正性为假
    assert (-n*k).is_nonpositive is False

    # 断言 u 的非正性为不确定
    assert u.is_nonpositive is None
    # 断言 -u 的非正性为真
    assert (-u).is_nonpositive is True
    # 断言 2*u 的非正性为不确定
    assert (2*u).is_nonpositive is None

    # 断言 v 的非正性为真
    assert v.is_nonpositive is True
    # 断言 -v 的非正性为不确定
    assert (-v).is_nonpositive is None
    # 断言 2*v 的非正性为真
    assert (2*v).is_nonpositive is True

    # 断言 u*v 的非正性为真
    assert (u*v).is_nonpositive is True

    # 断言 k*u 的非正性为真
    assert (k*u).is_nonpositive is True
    # 断言 k*v 的非正性为不确定
    assert (k*v).is_nonpositive is None

    # 断言 n*u 的非正性为不确定
    assert (n*u).is_nonpositive is None
    # 断言 n*v 的非正性为真
    assert (n*v).is_nonpositive is True

    # 断言 v*k*u 的非正性为不确定
    assert (v*k*u).is_nonpositive is None
    # 断言 v*n*u 的非正性为真
    assert (v*n*u).is_nonpositive is True

    # 断言 -v*k*u 的非正性为真
    assert (-v*k*u).is_nonpositive is True
    # 断言 -v*n*u 的非正性为不确定
    assert (-v*n*u).is_nonpositive is None

    # 断言 17*v*k*u 的非正性为不确定
    assert (17*v*k*u).is_nonpositive is None
    # 断言 17*v*n*u 的非正性为真
    assert (17*v*n*u).is_nonpositive is True

    # 断言 v*k*n*u 的非正性为不确定
    assert (v*k*n*u).is_nonpositive is None

    # 断言 x*k 的非正性为不确定
    assert (x*k).is_nonpositive is None
    # 断言 u*v*n*x*k 的非正性为不确定
    assert (u*v*n*x*k).is_nonpositive is None

    # 断言 k 的非负性为假
    assert k.is_nonnegative is False
    # 断言 -k 的非负性为真
    assert (-k).is_nonnegative is True
    # 断言 2*k 的非负性为假
    assert (2*k).is_nonnegative is False

    # 断言 n 的非负性为真
    assert n.is_nonnegative is True
    # 断言 -n 的非负性为假
    assert (-n).is_nonnegative is False
    # 断言 2*n 的非负性为真
    assert (2*n).is_nonnegative is True

    # 断言 n*k 的非负性为假
    assert (n*k).is_nonnegative is False
    # 断言 2*n*k 的非负性为假
    assert (2*n*k).is_nonnegative is False
    # 断言 -n*k 的非负性为真
    assert (-n*k).is_nonnegative is True

    # 断言 u 的非负性为真
    assert u.is_nonnegative is True
    # 断言 -u 的非负性为不确定
    assert (-u).is_nonnegative is None
    # 断言 2*u 的非负性为真
    assert (2*u).is_nonnegative is True

    # 断言 v 的非负性为不确定
    assert v.is_nonnegative is None
    # 断言 -v 的非负性为真
    assert (-v).is_nonnegative is True
    # 断言 2*v 的非负性为不确定
    assert (2*v).is_nonnegative is None

    # 断言 u*v 的非负性为不确定
    assert (u*v).is_nonnegative is None

    # 断言 k*u 的非负性为不确定
    assert (k*u).is_nonnegative is None
    # 断言 k*v 的非负性为真
    assert (k*v).is_nonnegative is True

    # 断言 n*u 的非负性为真
    assert (n*u).is_nonnegative is True
    # 断言 n*v 的非负性为不确定
    assert (n*v).is_nonnegative is None

    # 断言 v*k*u 的非负性为真
    assert (v*k*u).is_nonnegative is True
    # 断言 v*n*u 的非负性为不确定
    assert (v*n*u).is_nonnegative is None

    # 断言 -v*k*u 的非负性为不确定
    assert (-v*k*u).is_nonnegative is None
    # 断言表达式 (-v*n*u).is_nonnegative 是否为 True
    assert (-v*n*u).is_nonnegative is True

    # 断言表达式 (17*v*k*u).is_nonnegative 是否为 True
    assert (17*v*k*u).is_nonnegative is True
    # 断言表达式 (17*v*n*u).is_nonnegative 是否为 None
    assert (17*v*n*u).is_nonnegative is None

    # 断言表达式 (k*v*n*u).is_nonnegative 是否为 True
    assert (k*v*n*u).is_nonnegative is True

    # 断言表达式 (x*k).is_nonnegative 是否为 None
    assert (x*k).is_nonnegative is None
    # 断言表达式 (u*v*n*x*k).is_nonnegative 是否为 None
    assert (u*v*n*x*k).is_nonnegative is None
# 定义一个测试函数，用于验证加法表达式的符号性质
def test_Add_is_negative_positive():
    # 创建实数符号变量 x
    x = Symbol('x', real=True)

    # 创建负数符号变量 k
    k = Symbol('k', negative=True)
    # 创建正数符号变量 n
    n = Symbol('n', positive=True)
    # 创建非负数符号变量 u
    u = Symbol('u', nonnegative=True)
    # 创建非正数符号变量 v
    v = Symbol('v', nonpositive=True)

    # 断言 k - 2 是负数
    assert (k - 2).is_negative is True
    # 断言 k + 17 的符号不确定
    assert (k + 17).is_negative is None
    # 断言 -k - 5 的符号不确定
    assert (-k - 5).is_negative is None
    # 断言 -k + 123 是非负数
    assert (-k + 123).is_negative is False

    # 断言 k - n 是负数
    assert (k - n).is_negative is True
    # 断言 k + n 的符号不确定
    assert (k + n).is_negative is None
    # 断言 -k - n 的符号不确定
    assert (-k - n).is_negative is None
    # 断言 -k + n 是非负数
    assert (-k + n).is_negative is False

    # 断言 k - n - 2 是负数
    assert (k - n - 2).is_negative is True
    # 断言 k + n + 17 的符号不确定
    assert (k + n + 17).is_negative is None
    # 断言 -k - n - 5 的符号不确定
    assert (-k - n - 5).is_negative is None
    # 断言 -k + n + 123 是非负数
    assert (-k + n + 123).is_negative is False

    # 断言 -2*k + 123*n + 17 是非负数
    assert (-2*k + 123*n + 17).is_negative is False

    # 断言 k + u 的符号不确定
    assert (k + u).is_negative is None
    # 断言 k + v 是负数
    assert (k + v).is_negative is True
    # 断言 n + u 是非负数
    assert (n + u).is_negative is False
    # 断言 n + v 的符号不确定
    assert (n + v).is_negative is None

    # 断言 u - v 是非负数
    assert (u - v).is_negative is False
    # 断言 u + v 的符号不确定
    assert (u + v).is_negative is None
    # 断言 -u - v 的符号不确定
    assert (-u - v).is_negative is None
    # 断言 -u + v 的符号不确定
    assert (-u + v).is_negative is None

    # 断言 u - v + n + 2 是非负数
    assert (u - v + n + 2).is_negative is False
    # 断言 u + v + n + 2 的符号不确定
    assert (u + v + n + 2).is_negative is None
    # 断言 -u - v + n + 2 的符号不确定
    assert (-u - v + n + 2).is_negative is None
    # 断言 -u + v + n + 2 的符号不确定
    assert (-u + v + n + 2).is_negative is None

    # 断言 k + x 的符号不确定
    assert (k + x).is_negative is None
    # 断言 k + x - n 的符号不确定
    assert (k + x - n).is_negative is None

    # 断言 k - 2 是正数
    assert (k - 2).is_positive is False
    # 断言 k + 17 的符号不确定
    assert (k + 17).is_positive is None
    # 断言 -k - 5 的符号不确定
    assert (-k - 5).is_positive is None
    # 断言 -k + 123 是正数
    assert (-k + 123).is_positive is True

    # 断言 k - n 是负数
    assert (k - n).is_positive is False
    # 断言 k + n 的符号不确定
    assert (k + n).is_positive is None
    # 断言 -k - n 的符号不确定
    assert (-k - n).is_positive is None
    # 断言 -k + n 是正数
    assert (-k + n).is_positive is True

    # 断言 k - n - 2 是负数
    assert (k - n - 2).is_positive is False
    # 断言 k + n + 17 的符号不确定
    assert (k + n + 17).is_positive is None
    # 断言 -k - n - 5 的符号不确定
    assert (-k - n - 5).is_positive is None
    # 断言 -k + n + 123 是正数
    assert (-k + n + 123).is_positive is True

    # 断言 -2*k + 123*n + 17 是正数
    assert (-2*k + 123*n + 17).is_positive is True

    # 断言 k + u 的符号不确定
    assert (k + u).is_positive is None
    # 断言 k + v 是负数
    assert (k + v).is_positive is False
    # 断言 n + u 是正数
    assert (n + u).is_positive is True
    # 断言 n + v 的符号不确定
    assert (n + v).is_positive is None

    # 断言 u - v 的符号不确定
    assert (u - v).is_positive is None
    # 断言 u + v 的符号不确定
    assert (u + v).is_positive is None
    # 断言 -u - v 的符号不确定
    assert (-u - v).is_positive is None
    # 断言 -u + v 是负数
    assert (-u + v).is_positive is False

    # 断言 u - v - n - 2 的符号不确定
    assert (u - v - n - 2).is_positive is None
    # 断言 u + v - n - 2 的符号不确定
    assert (u + v - n - 2).is_positive is None
    # 断言 -u - v - n - 2 的符号不确定
    assert (-u - v - n - 2).is_positive is None
    # 断言 -u + v - n - 2 是负数
    assert (-u + v - n - 2).is_positive is False

    # 断言 n + x 的符号不确定
    assert (n + x).is_positive is None
    # 断言 n + x - k 的符号不确定
    assert (n + x - k).is_positive is None

    # 计算并断言 z 是零
    z = (-3 - sqrt(5) + (-sqrt(10)/2 - sqrt(2)/2)**2)
    assert z.is_zero
    # 计算并断言 z 是零
    z = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    assert z.is_zero


# 定义一个测试函数，用于验证加法表达式的非正数和非负数性质
def test_Add_is_nonpositive_nonnegative():
    # 创建实数符号变量 x
    x = Symbol('x', real=True)

    # 创建负数符号变量 k
    k = Symbol('k', negative=True)
    # 创建正数符号变量 n
    n = Symbol('n', positive=True)
    # 创建非负数符号变量 u
    u = Symbol('u', nonnegative=True)
    # 创建非正数符号变量 v
    v = Symbol('v', nonpositive=True)

    # 断言 u - 2 的符号不确定
    assert (u - 2).is_nonpositive is None
    # 断言 u + 17 是非非正数
    assert (u + 17).is_nonpositive is False
    # 断言表达式 (-u - 5).is_nonpositive 是否为 True
    assert (-u - 5).is_nonpositive is True
    # 断言表达式 (-u + 123).is_nonpositive 是否为 None
    assert (-u + 123).is_nonpositive is None

    # 断言表达式 (u - v).is_nonpositive 是否为 None
    assert (u - v).is_nonpositive is None
    # 断言表达式 (u + v).is_nonpositive 是否为 None
    assert (u + v).is_nonpositive is None
    # 断言表达式 (-u - v).is_nonpositive 是否为 None
    assert (-u - v).is_nonpositive is None
    # 断言表达式 (-u + v).is_nonpositive 是否为 True
    assert (-u + v).is_nonpositive is True

    # 断言表达式 (u - v - 2).is_nonpositive 是否为 None
    assert (u - v - 2).is_nonpositive is None
    # 断言表达式 (u + v + 17).is_nonpositive 是否为 None
    assert (u + v + 17).is_nonpositive is None
    # 断言表达式 (-u - v - 5).is_nonpositive 是否为 None
    assert (-u - v - 5).is_nonpositive is None
    # 断言表达式 (-u + v - 123).is_nonpositive 是否为 True
    assert (-u + v - 123).is_nonpositive is True

    # 断言表达式 (-2*u + 123*v - 17).is_nonpositive 是否为 True
    assert (-2*u + 123*v - 17).is_nonpositive is True

    # 断言表达式 (k + u).is_nonpositive 是否为 None
    assert (k + u).is_nonpositive is None
    # 断言表达式 (k + v).is_nonpositive 是否为 True
    assert (k + v).is_nonpositive is True
    # 断言表达式 (n + u).is_nonpositive 是否为 False
    assert (n + u).is_nonpositive is False
    # 断言表达式 (n + v).is_nonpositive 是否为 None
    assert (n + v).is_nonpositive is None

    # 断言表达式 (k - n).is_nonpositive 是否为 True
    assert (k - n).is_nonpositive is True
    # 断言表达式 (k + n).is_nonpositive 是否为 None
    assert (k + n).is_nonpositive is None
    # 断言表达式 (-k - n).is_nonpositive 是否为 None
    assert (-k - n).is_nonpositive is None
    # 断言表达式 (-k + n).is_nonpositive 是否为 False
    assert (-k + n).is_nonpositive is False

    # 断言表达式 (k - n + u + 2).is_nonpositive 是否为 None
    assert (k - n + u + 2).is_nonpositive is None
    # 断言表达式 (k + n + u + 2).is_nonpositive 是否为 None
    assert (k + n + u + 2).is_nonpositive is None
    # 断言表达式 (-k - n + u + 2).is_nonpositive 是否为 None
    assert (-k - n + u + 2).is_nonpositive is None
    # 断言表达式 (-k + n + u + 2).is_nonpositive 是否为 False
    assert (-k + n + u + 2).is_nonpositive is False

    # 断言表达式 (u + x).is_nonpositive 是否为 None
    assert (u + x).is_nonpositive is None
    # 断言表达式 (v - x - n).is_nonpositive 是否为 None
    assert (v - x - n).is_nonpositive is None

    # 断言表达式 (u - 2).is_nonnegative 是否为 None
    assert (u - 2).is_nonnegative is None
    # 断言表达式 (u + 17).is_nonnegative 是否为 True
    assert (u + 17).is_nonnegative is True
    # 断言表达式 (-u - 5).is_nonnegative 是否为 False
    assert (-u - 5).is_nonnegative is False
    # 断言表达式 (-u + 123).is_nonnegative 是否为 None
    assert (-u + 123).is_nonnegative is None

    # 断言表达式 (u - v).is_nonnegative 是否为 True
    assert (u - v).is_nonnegative is True
    # 断言表达式 (u + v).is_nonnegative 是否为 None
    assert (u + v).is_nonnegative is None
    # 断言表达式 (-u - v).is_nonnegative 是否为 None
    assert (-u - v).is_nonnegative is None
    # 断言表达式 (-u + v).is_nonnegative 是否为 None
    assert (-u + v).is_nonnegative is None

    # 断言表达式 (u - v + 2).is_nonnegative 是否为 True
    assert (u - v + 2).is_nonnegative is True
    # 断言表达式 (u + v + 17).is_nonnegative 是否为 None
    assert (u + v + 17).is_nonnegative is None
    # 断言表达式 (-u - v - 5).is_nonnegative 是否为 None
    assert (-u - v - 5).is_nonnegative is None
    # 断言表达式 (-u + v - 123).is_nonnegative 是否为 False
    assert (-u + v - 123).is_nonnegative is False

    # 断言表达式 (2*u - 123*v + 17).is_nonnegative 是否为 True
    assert (2*u - 123*v + 17).is_nonnegative is True

    # 断言表达式 (k + u).is_nonnegative 是否为 None
    assert (k + u).is_nonnegative is None
    # 断言表达式 (k + v).is_nonnegative 是否为 False
    assert (k + v).is_nonnegative is False
    # 断言表达式 (n + u).is_nonnegative 是否为 True
    assert (n + u).is_nonnegative is True
    # 断言表达式 (n + v).is_nonnegative 是否为 None
    assert (n + v).is_nonnegative is None

    # 断言表达式 (k - n).is_nonnegative 是否为 False
    assert (k - n).is_nonnegative is False
    # 断言表达式 (k + n).is_nonnegative 是否为 None
    assert (k + n).is_nonnegative is None
    # 断言表达式 (-k - n).is_nonnegative 是否为 None
    assert (-k - n).is_nonnegative is None
    # 断言表达式 (-k + n).is_nonnegative 是否为 True
    assert (-k + n).is_nonnegative is True

    # 断言表达式 (k - n - u - 2).is_nonnegative 是否为 False
    assert (k - n - u - 2).is_nonnegative is False
    # 断言表达式 (k + n - u - 2).is_nonnegative 是否为 None
    assert (k + n - u - 2).is_nonnegative is None
    # 断言表达式 (-k - n - u - 2).is_nonnegative 是否为 None
    assert (-k - n - u - 2).is_nonnegative is None
    # 断言表达式 (-k + n - u - 2).is_nonnegative 是否为 None
    assert (-k + n - u - 2).is_nonnegative is None

    # 断言表达式 (u - x).is_nonnegative 是否为 None
    assert (u - x).is_nonnegative is None
    # 断言表达式 (v + x + n).is_nonnegative 是否为 None
    assert (v + x + n).is_nonnegative is None
# 定义一个测试函数，用于检查幂运算对象的整数属性
def test_Pow_is_integer():
    # 定义符号变量 x
    x = Symbol('x')

    # 定义整数类型的符号变量 k，n 和 m，并指定它们的约束条件
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True, nonnegative=True)
    m = Symbol('m', integer=True, positive=True)

    # 断言 k 的平方是否为整数
    assert (k**2).is_integer is True
    # 断言 k 的负二次方是否为整数
    assert (k**(-2)).is_integer is None
    # 断言 (m + 1) 的负二次方是否为整数
    assert ((m + 1)**(-2)).is_integer is False
    # 断言 m 的负一次方是否为整数，并标注问题编号 8580
    assert (m**(-1)).is_integer is None  # issue 8580

    # 断言 2 的 k 次方是否为整数
    assert (2**k).is_integer is None
    # 断言 2 的负 k 次方是否为整数
    assert (2**(-k)).is_integer is None

    # 断言 2 的 n 次方是否为整数
    assert (2**n).is_integer is True
    # 断言 2 的负 n 次方是否为整数
    assert (2**(-n)).is_integer is None

    # 断言 2 的 m 次方是否为整数
    assert (2**m).is_integer is True
    # 断言 2 的负 m 次方是否为整数
    assert (2**(-m)).is_integer is False

    # 断言 x 的平方是否为整数
    assert (x**2).is_integer is None
    # 断言 2 的 x 次方是否为整数
    assert (2**x).is_integer is None

    # 断言 k 的 n 次方是否为整数
    assert (k**n).is_integer is True
    # 断言 k 的负 n 次方是否为整数
    assert (k**(-n)).is_integer is None

    # 断言 k 的 x 次方是否为整数
    assert (k**x).is_integer is None
    # 断言 x 的 k 次方是否为整数
    assert (x**k).is_integer is None

    # 断言 k 的 (n*m) 次方是否为整数
    assert (k**(n*m)).is_integer is True
    # 断言 k 的负 (n*m) 次方是否为整数
    assert (k**(-n*m)).is_integer is None

    # 断言 sqrt(3) 是否为整数
    assert sqrt(3).is_integer is False
    # 断言 sqrt(.3) 是否为整数
    assert sqrt(.3).is_integer is False
    # 使用 Pow 对象指定 evaluate=False，判断 3 的 2 次方是否为整数
    assert Pow(3, 2, evaluate=False).is_integer is True
    # 使用 Pow 对象指定 evaluate=False，判断 3 的 0 次方是否为整数
    assert Pow(3, 0, evaluate=False).is_integer is True
    # 使用 Pow 对象指定 evaluate=False，判断 3 的负 2 次方是否为整数
    assert Pow(3, -2, evaluate=False).is_integer is False
    # 使用 Pow 对象指定 evaluate=False，判断 S.Half 的 3 次方是否为整数
    assert Pow(S.Half, 3, evaluate=False).is_integer is False
    # 使用 Pow 对象指定 evaluate=False，判断 S.Half 的 3 次方是否为整数
    assert Pow(3, S.Half, evaluate=False).is_integer is False
    # 使用 Pow 对象指定 evaluate=False，判断 4 的 S.Half 次方是否为整数
    assert Pow(4, S.Half, evaluate=False).is_integer is True
    # 使用 Pow 对象指定 evaluate=False，判断 S.Half 的负 2 次方是否为整数
    assert Pow(S.Half, -2, evaluate=False).is_integer is True

    # 断言 (-1)**k 是否为整数
    assert ((-1)**k).is_integer

    # 标注问题编号 8641
    # 定义实数型但非整数型的符号变量 x
    x = Symbol('x', real=True, integer=False)
    # 断言 x 的平方是否为整数
    assert (x**2).is_integer is None

    # 标注问题编号 10458
    # 定义正数型的符号变量 x
    x = Symbol('x', positive=True)
    # 断言 1/(x + 1) 是否为整数
    assert (1/(x + 1)).is_integer is False
    # 断言 1/(-x - 1) 是否为整数
    assert (1/(-x - 1)).is_integer is False
    # 断言 -1/(x + 1) 是否为整数
    assert (-1/(x + 1)).is_integer is False
    # 标注问题编号 23287
    # 断言 x 的平方除以 2 是否为整数
    assert (x**2/2).is_integer is None

    # 标注类似问题编号 8648
    # 定义偶数型的符号变量 k
    k = Symbol('k', even=True)
    # 断言 k 的 3 次方除以 2 是否为整数
    assert (k**3/2).is_integer
    # 断言 k 的 3 次方除以 8 是否为整数
    assert (k**3/8).is_integer
    # 断言 k 的 3 次方除以 16 是否为整数
    assert (k**3/16).is_integer is None
    # 断言 2/k 是否为整数
    assert (2/k).is_integer is None
    # 断言 2/k**2 是否为整数
    assert (2/k**2).is_integer is False
    # 定义奇数型的符号变量 o
    o = Symbol('o', odd=True)
    # 断言 k/o 是否为整数
    assert (k/o).is_integer is None
    # 定义同时为奇数和质数的符号变量 o
    o = Symbol('o', odd=True, prime=True)
    # 断言 k/o 是否为整数
    assert (k/o).is_integer is False
    # 断言：i 的立方是否为扩展实数（不是）
    assert (i**3).is_extended_real is False
    
    # 断言：i 的 x 次方是否为实数（可能是 (-I)**(2/3)）
    assert (i**x).is_real is None
    
    # 创建一个偶数符号 'e'
    e = Symbol('e', even=True)
    
    # 创建一个奇数符号 'o'
    o = Symbol('o', odd=True)
    
    # 创建一个整数符号 'k'
    k = Symbol('k', integer=True)
    
    # 断言：i 的偶数次方是否为扩展实数（是）
    assert (i**e).is_extended_real is True
    
    # 断言：i 的奇数次方是否为扩展实数（不是）
    assert (i**o).is_extended_real is False
    
    # 断言：i 的 k 次方是否为实数（可能是 None）
    assert (i**k).is_real is None
    
    # 断言：i 的 4k 次方是否为扩展实数（是）
    assert (i**(4*k)).is_extended_real is True
    
    # 创建一个非负数符号 'x'
    x = Symbol("x", nonnegative=True)
    
    # 创建一个非负数符号 'y'
    y = Symbol("y", nonnegative=True)
    
    # 断言：im(x**y) 展开后是否为 0
    assert im(x**y).expand(complex=True) is S.Zero
    
    # 断言：x 的 y 次方是否为实数（是）
    assert (x**y).is_real is True
    
    # 创建一个虚数符号 'i'
    i = Symbol('i', imaginary=True)
    
    # 断言：exp(i)**I 是否为扩展实数（是）
    assert (exp(i)**I).is_extended_real is True
    
    # 断言：log(exp(i)) 是否为虚数（可能是 2*pi*I）
    assert log(exp(i)).is_imaginary is None
    
    # 创建一个复数符号 'c'
    c = Symbol('c', complex=True)
    
    # 断言：log(c) 是否为实数（可能是 None，c 可能为 0 或 2 等）
    assert log(c).is_real is None
    
    # 断言：log(exp(c)) 是否为实数（可能是 None，考虑 log(0), log(E) 等）
    assert log(exp(c)).is_real is None
    
    # 创建一个非负数符号 'n'
    n = Symbol('n', negative=False)
    
    # 断言：log(n) 是否为实数（可能是 None）
    assert log(n).is_real is None
    
    # 重新创建一个非负数符号 'n'
    n = Symbol('n', nonnegative=True)
    
    # 断言：log(n) 是否为实数（可能是 None）
    assert log(n).is_real is None
    
    # 断言：sqrt(-I) 是否为实数（不是）（参考 issue 7843）
    assert sqrt(-I).is_real is False
    
    # 创建一个整数符号 'i'
    i = Symbol('i', integer=True)
    
    # 断言：1 / (i-1) 是否为实数（可能是 None）
    assert (1/(i-1)).is_real is None
    
    # 断言：1 / (i-1) 是否为扩展实数（可能是 None）
    assert (1/(i-1)).is_extended_real is None
    
    # 测试 issue 20715
    from sympy.core.parameters import evaluate
    
    # 将 x 设置为 -1
    x = S(-1)
    
    # 在禁用评估的环境中：
    with evaluate(False):
        # 断言：x 是否为负数（是）
        assert x.is_negative is True
    
    # 创建一个 Pow 对象 f，表示 x 的 -1 次方
    f = Pow(x, -1)
    
    # 在禁用评估的环境中：
    with evaluate(False):
        # 断言：f 是否为虚数（不是）
        assert f.is_imaginary is False
# 定义一个测试函数，用于检查幂运算相关的符号和性质
def test_real_Pow():
    # 创建一个整数类型、非零的符号 k
    k = Symbol('k', integer=True, nonzero=True)
    # 断言 k**(I*pi/log(k)) 是否为实数

    assert (k**(I*pi/log(k))).is_real


# 定义另一个测试函数，用于检查幂运算的有限性质
def test_Pow_is_finite():
    # 创建一个扩展实数类型的符号 xe
    xe = Symbol('xe', extended_real=True)
    # 创建一个实数类型的符号 xr
    xr = Symbol('xr', real=True)
    # 创建一个正数类型的符号 p
    p = Symbol('p', positive=True)
    # 创建一个负数类型的符号 n
    n = Symbol('n', negative=True)
    # 创建一个整数类型的符号 i
    i = Symbol('i', integer=True)

    # 断言 xe**2 是否为有限的，如果 xe 可能是无穷大，则结果为 None
    assert (xe**2).is_finite is None  # xe could be oo
    # 断言 xr**2 是否为有限的，xr 是实数，因此结果为 True
    assert (xr**2).is_finite is True

    # 断言 xe**xe 是否为有限的，由于 xe 可能是无穷大，结果为 None
    assert (xe**xe).is_finite is None
    # 断言 xr**xe 是否为有限的，由于 xe 可能是无穷大，结果为 None
    assert (xr**xe).is_finite is None
    # 断言 xe**xr 是否为有限的，由于 xe 可能是无穷大，结果为 None
    assert (xe**xr).is_finite is None
    # 断言 xr**xr 是否为有限的，xr 和 xr 都是实数，结果为 None
    assert (xr**xr).is_finite is None  # FIXME: The line below should be True rather than None

    # 断言 p**xe 是否为有限的，由于 xe 可能是无穷大，结果为 None
    assert (p**xe).is_finite is None
    # 断言 p**xr 是否为有限的，xr 是实数，结果为 True
    assert (p**xr).is_finite is True

    # 断言 n**xe 是否为有限的，由于 xe 可能是无穷大，结果为 None
    assert (n**xe).is_finite is None
    # 断言 n**xr 是否为有限的，xr 是实数，结果为 True
    assert (n**xr).is_finite is True

    # 断言 sin(xe)**2 是否为有限的，由于 sin(xe) 的平方始终是有限的，结果为 True
    assert (sin(xe)**2).is_finite is True
    # 断言 sin(xr)**2 是否为有限的，xr 是实数，sin(xr) 的平方始终是有限的，结果为 True
    assert (sin(xr)**2).is_finite is True

    # 断言 sin(xe)**xe 是否为有限的，由于 xe 和 xr 可能是 -pi，结果为 None
    assert (sin(xe)**xe).is_finite is None
    # 断言 sin(xr)**xr 是否为有限的，xr 可能是 -pi，结果为 None
    assert (sin(xr)**xr).is_finite is None

    # FIXME: Should the line below be True rather than None?
    # 断言 sin(xe)**exp(xe) 是否为有限的，由于 xe 可能是 -oo，结果为 None
    assert (sin(xe)**exp(xe)).is_finite is None
    # 断言 sin(xr)**exp(xr) 是否为有限的，xr 是实数，结果为 True
    assert (sin(xr)**exp(xr)).is_finite is True

    # 断言 1/sin(xe) 是否为有限的，如果 sin(xe) 是零，则结果为 None，否则为 True
    assert (1/sin(xe)).is_finite is None  # if zero, no, otherwise yes
    # 断言 1/sin(xr) 是否为有限的，如果 sin(xr) 是零，则结果为 None，否则为 True
    assert (1/sin(xr)).is_finite is None

    # 断言 1/exp(xe) 是否为有限的，由于 xe 可能是 -oo，结果为 None
    assert (1/exp(xe)).is_finite is None
    # 断言 1/exp(xr) 是否为有限的，xr 是实数，结果为 True
    assert (1/exp(xr)).is_finite is True

    # 断言 1/S.Pi 是否为有限的，S.Pi 是圆周率，结果为 True
    assert (1/S.Pi).is_finite is True

    # 断言 1/(i-1) 是否为有限的，由于 i 是整数，但未指定值，结果为 None
    assert (1/(i-1)).is_finite is None


# 定义一个测试函数，用于检查幂运算的奇偶性质
def test_Pow_is_even_odd():
    # 创建一个符号 x
    x = Symbol('x')

    # 创建一个偶数类型的符号 k
    k = Symbol('k', even=True)
    # 创建一个奇数类型的符号 n
    n = Symbol('n', odd=True)
    # 创建一个非负整数类型的符号 m
    m = Symbol('m', integer=True, nonnegative=True)
    # 创建一个正整数类型的符号 p
    p = Symbol('p', integer=True, positive=True)

    # 断言 (-1)**n 是否为奇数
    assert ((-1)**n).is_odd
    # 断言 (-1)**k 是否为奇数
    assert ((-1)**k).is_odd
    # 断言 (-1)**(m - p) 是否为奇数
    assert ((-1)**(m - p)).is_odd

    # 断言 k**2 是否为偶数，k 是偶数，因此结果为 True
    assert (k**2).is_even is True
    # 断言 n**2 是否为偶数，n 是奇数，因此结果为 False
    assert (n**2).is_even is False
    # 断言 2**k 是否为偶数，k 是偶数，结果为 None
    assert (2**k).is_even is None
    # 断言 x**2 是否为偶数，未指定 x 的性质，结果为 None
    assert (x**2).is_even is None

    # 断言 k**m 是否为偶数，k 是偶数，结果为 None
    assert (k**m).is_even is None
    # 断言 n**m 是否为偶数，n 是奇数，结果为 False
    assert (n**m).is_even is False

    # 断言 k**p 是否为偶数，k 是偶数，结果为 True
    assert (k**p).is_even is True
    # 断言 n**p 是否为偶数，n 是奇数，结果为 False
    assert (n**p).is_even is False

    # 断言 m**k 是否为偶数，m 是非负整数，但未指定值，结果为 None
    assert (m**k).is_even is None
    # 断言 p**k 是否为偶数，p 是正整数，但未指定值，结果为 None
    assert (p**k).is_even is None

    # 断言 m**n 是否为偶数，m 和 n 未指定值，结果为 None
    assert (m**n).is_even is None
    # 断言 p**n 是否为偶数，p 和 n 未指定值，结果为 None
    assert (p**n).is_even is None

    # 断言 k**x 是否为偶数，k 未指定值，x 未指定性质，结果为 None
    assert (k**x).is_even is None
    # 断言 n**x 是否为偶数，n 未指定值，x 未指定性质，结果为 None
    assert (n**x).is_even is None
    # 断言 (-2)**n 是否为正数，期望结果为 True
    assert ((-2)**n).is_positive is True
    # 断言 (-2)**m 是否为正数，期望结果为 False
    assert ((-2)**m).is_positive is False

    # 断言 k**2 是否为正数，期望结果为 True
    assert (k**2).is_positive is True
    # 断言 k**(-2) 是否为正数，期望结果为 True
    assert (k**(-2)).is_positive is True

    # 断言 k**r 是否为正数，期望结果为 True
    assert (k**r).is_positive is True
    # 断言 (-k)**r 是否为正数，期望结果为 None
    assert ((-k)**r).is_positive is None
    # 断言 (-k)**n 是否为正数，期望结果为 True
    assert ((-k)**n).is_positive is True
    # 断言 (-k)**m 是否为正数，期望结果为 False
    assert ((-k)**m).is_positive is False

    # 断言 2**r 是否为负数，期望结果为 False
    assert (2**r).is_negative is False
    # 断言 (-2)**r 是否为负数，期望结果为 None
    assert ((-2)**r).is_negative is None
    # 断言 (-2)**n 是否为负数，期望结果为 False
    assert ((-2)**n).is_negative is False
    # 断言 (-2)**m 是否为负数，期望结果为 True
    assert ((-2)**m).is_negative is True

    # 断言 k**2 是否为负数，期望结果为 False
    assert (k**2).is_negative is False
    # 断言 k**(-2) 是否为负数，期望结果为 False
    assert (k**(-2)).is_negative is False

    # 断言 k**r 是否为负数，期望结果为 False
    assert (k**r).is_negative is False
    # 断言 (-k)**r 是否为负数，期望结果为 None
    assert ((-k)**r).is_negative is None
    # 断言 (-k)**n 是否为负数，期望结果为 False
    assert ((-k)**n).is_negative is False
    # 断言 (-k)**m 是否为负数，期望结果为 True
    assert ((-k)**m).is_negative is True

    # 断言 2**x 是否为正数，期望结果为 None
    assert (2**x).is_positive is None
    # 断言 2**x 是否为负数，期望结果为 None
    assert (2**x).is_negative is None
# 定义一个测试函数，测试幂运算结果是否为零
def test_Pow_is_zero():
    # 创建一个符号 z，其值为零
    z = Symbol('z', zero=True)
    # 计算 z 的平方
    e = z**2
    # 断言 e 是否为零
    assert e.is_zero
    # 断言 e 是否为正数
    assert e.is_positive is False
    # 断言 e 是否为负数
    assert e.is_negative is False

    # 检查 Pow(0, 0, evaluate=False) 是否为零
    assert Pow(0, 0, evaluate=False).is_zero is False
    # 检查 Pow(0, 3, evaluate=False) 是否为零
    assert Pow(0, 3, evaluate=False).is_zero
    # 检查 Pow(0, oo, evaluate=False) 是否为零
    assert Pow(0, oo, evaluate=False).is_zero
    # 检查 Pow(0, -3, evaluate=False) 是否为零
    assert Pow(0, -3, evaluate=False).is_zero is False
    # 检查 Pow(0, -oo, evaluate=False) 是否为零
    assert Pow(0, -oo, evaluate=False).is_zero is False
    # 检查 Pow(2, 2, evaluate=False) 是否为零
    assert Pow(2, 2, evaluate=False).is_zero is False

    # 创建一个非零符号 a
    a = Symbol('a', zero=False)
    # 断言 Pow(a, 3) 是否为零，这是 issue 7965
    assert Pow(a, 3).is_zero is False

    # 检查 Pow(2, oo, evaluate=False) 是否为零
    assert Pow(2, oo, evaluate=False).is_zero is False
    # 检查 Pow(2, -oo, evaluate=False) 是否为零
    assert Pow(2, -oo, evaluate=False).is_zero
    # 检查 Pow(S.Half, oo, evaluate=False) 是否为零
    assert Pow(S.Half, oo, evaluate=False).is_zero
    # 检查 Pow(S.Half, -oo, evaluate=False) 是否为零
    assert Pow(S.Half, -oo, evaluate=False).is_zero is False

    # 检查所有实数/复数基数/指数的组合
    h = S.Half
    T = True
    F = False
    N = None

    pow_iszero = [
        ['**',  0,  h,  1,  2, -h, -1,-2,-2*I,-I/2,I/2,1+I,oo,-oo,zoo],
        [   0,  F,  T,  T,  T,  F,  F,  F,  F,  F,  F,  N,  T,  F,  N],
        [   h,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [   1,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  N],
        [   2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [  -h,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [  -1,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  N],
        [  -2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [-2*I,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [-I/2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [ I/2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [ 1+I,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [  oo,  F,  F,  F,  F,  T,  T,  T,  F,  F,  F,  F,  F,  T,  N],
        [ -oo,  F,  F,  F,  F,  T,  T,  T,  F,  F,  F,  F,  F,  T,  N],
        [ zoo,  F,  F,  F,  F,  T,  T,  T,  N,  N,  N,  N,  F,  T,  N]
    ]

    # 定义一个测试表格的函数，检查幂运算的结果是否正确
    def test_table(table):
        n = len(table[0])
        for row in range(1, n):
            base = table[row][0]
            for col in range(1, n):
                exp = table[0][col]
                is_zero = table[row][col]
                # 进行实际的测试
                assert Pow(base, exp, evaluate=False).is_zero is is_zero

    # 执行测试表格函数，传入 pow_iszero 表格
    test_table(pow_iszero)

    # 定义一个零符号
    zo, zo2 = symbols('zo, zo2', zero=True)

    # 定义所有有限符号的组合
    zf, zf2 = symbols('zf, zf2', finite=True)
    wf, wf2 = symbols('wf, wf2', nonzero=True)
    xf, xf2 = symbols('xf, xf2', real=True)
    yf, yf2 = symbols('yf, yf2', nonzero=True)
    af, af2 = symbols('af, af2', positive=True)
    bf, bf2 = symbols('bf, bf2', nonnegative=True)
    cf, cf2 = symbols('cf, cf2', negative=True)
    df, df2 = symbols('df, df2', nonpositive=True)

    # 不考虑有限性的情况下定义符号
    zi, zi2 = symbols('zi, zi2')
    wi, wi2 = symbols('wi, wi2', zero=False)
    # 使用 sympy 的 symbols 函数定义符号变量 xi, xi2，扩展为实数域
    xi, xi2 = symbols('xi, xi2', extended_real=True)
    # 使用 sympy 的 symbols 函数定义符号变量 yi, yi2，不能为零，扩展为实数域
    yi, yi2 = symbols('yi, yi2', zero=False, extended_real=True)
    # 使用 sympy 的 symbols 函数定义符号变量 ai, ai2，必须为正数，扩展为实数域
    ai, ai2 = symbols('ai, ai2', extended_positive=True)
    # 使用 sympy 的 symbols 函数定义符号变量 bi, bi2，必须为非负数，扩展为实数域
    bi, bi2 = symbols('bi, bi2', extended_nonnegative=True)
    # 使用 sympy 的 symbols 函数定义符号变量 ci, ci2，必须为负数，扩展为实数域
    ci, ci2 = symbols('ci, ci2', extended_negative=True)
    # 使用 sympy 的 symbols 函数定义符号变量 di, di2，必须为非正数，扩展为实数域
    di, di2 = symbols('di, di2', extended_nonpositive=True)

    # 定义二维列表 pow_iszero_sym，包含各种情况下的符号运算是否为零的预期结果
    pow_iszero_sym = [
        ['**',zo,wf,yf,af,cf,zf,xf,bf,df,zi,wi,xi,yi,ai,bi,ci,di],
        [ zo2, F, N, N, T, F, N, N, N, F, N, N, N, N, T, N, F, F],
        [ wf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ yf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ af2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ cf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ zf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ xf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ bf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ df2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ zi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ wi2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ xi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ yi2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ ai2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ bi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ ci2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ di2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N]
    ]

    # 调用 test_table 函数，传入 pow_iszero_sym 作为参数进行测试
    test_table(pow_iszero_sym)

    # 断言表达式，验证不同的符号幂运算是否为零的预期结果
    assert (zo ** zo).is_zero is False
    assert (wf ** wf).is_zero is False
    assert (yf ** yf).is_zero is False
    assert (af ** af).is_zero is False
    assert (cf ** cf).is_zero is False
    assert (zf ** zf).is_zero is None
    assert (xf ** xf).is_zero is None
    assert (bf ** bf).is_zero is False  # 在表格中为 None
    assert (df ** df).is_zero is None
    assert (zi ** zi).is_zero is None
    assert (wi ** wi).is_zero is None
    assert (xi ** xi).is_zero is None
    assert (yi ** yi).is_zero is None
    assert (ai ** ai).is_zero is False  # 在表格中为 None
    assert (bi ** bi).is_zero is False  # 在表格中为 None
    assert (ci ** ci).is_zero is None
    assert (di ** di).is_zero is None
# 定义一个测试函数，用于检查幂运算在数学表达式中的符号属性

def test_Pow_is_nonpositive_nonnegative():
    # 创建一个实数符号变量 x
    x = Symbol('x', real=True)

    # 创建整数符号变量 k，其非负
    k = Symbol('k', integer=True, nonnegative=True)
    # 创建整数符号变量 l，其正数
    l = Symbol('l', integer=True, positive=True)
    # 创建整数符号变量 n，其为偶数
    n = Symbol('n', even=True)
    # 创建整数符号变量 m，其为奇数
    m = Symbol('m', odd=True)

    # 断言：(x**(4*k)) 的非负属性为 True
    assert (x**(4*k)).is_nonnegative is True
    # 断言：(2**x) 的非负属性为 True
    assert (2**x).is_nonnegative is True
    # 断言：((-2)**x) 的非负属性为 None
    assert ((-2)**x).is_nonnegative is None
    # 断言：((-2)**n) 的非负属性为 True
    assert ((-2)**n).is_nonnegative is True
    # 断言：((-2)**m) 的非负属性为 False
    assert ((-2)**m).is_nonnegative is False

    # 断言：(k**2) 的非负属性为 True
    assert (k**2).is_nonnegative is True
    # 断言：(k**(-2)) 的非负属性为 None
    assert (k**(-2)).is_nonnegative is None
    # 断言：(k**k) 的非负属性为 True
    assert (k**k).is_nonnegative is True

    # 断言：(k**x) 的非负属性为 None
    assert (k**x).is_nonnegative is None    # NOTE (0**x).is_real = U
    # 断言：(l**x) 的非负属性为 True
    assert (l**x).is_nonnegative is True
    # 断言：(l**x) 的正数属性为 True
    assert (l**x).is_positive is True
    # 断言：((-k)**x) 的非负属性为 None
    assert ((-k)**x).is_nonnegative is None

    # 断言：((-k)**m) 的非负属性为 None
    assert ((-k)**m).is_nonnegative is None

    # 断言：(2**x) 的非正属性为 False
    assert (2**x).is_nonpositive is False
    # 断言：((-2)**x) 的非正属性为 None
    assert ((-2)**x).is_nonpositive is None
    # 断言：((-2)**n) 的非正属性为 False
    assert ((-2)**n).is_nonpositive is False
    # 断言：((-2)**m) 的非正属性为 True
    assert ((-2)**m).is_nonpositive is True

    # 断言：(k**2) 的非正属性为 None
    assert (k**2).is_nonpositive is None
    # 断言：(k**(-2)) 的非正属性为 None
    assert (k**(-2)).is_nonpositive is None

    # 断言：(k**x) 的非正属性为 None
    assert (k**x).is_nonpositive is None
    # 断言：((-k)**x) 的非正属性为 None
    assert ((-k)**x).is_nonpositive is None
    # 断言：((-k)**n) 的非正属性为 None
    assert ((-k)**n).is_nonpositive is None

    # 断言：(x**2) 的非负属性为 True
    assert (x**2).is_nonnegative is True
    # 创建虚数符号变量 i
    i = symbols('i', imaginary=True)
    # 断言：(i**2) 的非正属性为 True
    assert (i**2).is_nonpositive is True
    # 断言：(i**4) 的非正属性为 False
    assert (i**4).is_nonpositive is False
    # 断言：(i**3) 的非正属性为 False
    assert (i**3).is_nonpositive is False
    # 断言：(I**i) 的非负属性为 True
    assert (I**i).is_nonnegative is True
    # 断言：(exp(I)**i) 的非负属性为 True
    assert (exp(I)**i).is_nonnegative is True

    # 断言：((-l)**n) 的非负属性为 True
    assert ((-l)**n).is_nonnegative is True
    # 断言：((-l)**m) 的非正属性为 True
    assert ((-l)**m).is_nonpositive is True
    # 断言：((-k)**n) 的非负属性为 None
    assert ((-k)**n).is_nonnegative is None
    # 断言：((-k)**m) 的非正属性为 None
    assert ((-k)**m).is_nonpositive is None

# 定义一个测试函数，用于检查乘法运算在数学表达式中的虚实属性

def test_Mul_is_imaginary_real():
    # 创建一个实数符号变量 r
    r = Symbol('r', real=True)
    # 创建一个正数符号变量 p
    p = Symbol('p', positive=True)
    # 创建一个虚数符号变量 i1
    i1 = Symbol('i1', imaginary=True)
    # 创建一个虚数符号变量 i2
    i2 = Symbol('i2', imaginary=True)
    # 创建一个符号变量 x
    x = Symbol('x')

    # 断言：虚数单位 I 的虚数属性为 True
    assert I.is_imaginary is True
    # 断言：虚数单位 I 的实数属性为 False
    assert I.is_real is False
    # 断言：虚数单位 -I 的虚数属性为 True
    assert (-I).is_imaginary is True
    # 断言：虚数单位 -I 的实数属性为 False
    assert (-I).is_real is False
    # 断言：复数 3*I 的虚数属性为 True
    assert (3*I).is_imaginary is True
    # 断言：复数 3*I 的实数属性为 False
    assert (3*I).is_real is False
    # 断言：复数 I*I 的虚数属性为 False
    assert (I*I).is_imaginary is False
    # 断言：复数 I*I 的实数属性为 True
    assert (I*I).is_real is True

    # 创建一个复数表达式 e
    e = (p + p*I)
    # 创建一个整数符号变量 j，其为非零整数
    j = Symbol('j', integer=True, zero=False)
    # 断言：(e**j) 的实数属性为 None
    assert (e**j).is_real is None
    # 断言：(e**(2*j)) 的实数属性为 None
    assert (e**(2*j)).is_real is None
    # 断言：(e**j) 的虚数属性为 None
    assert (e**j).is_imaginary is None
    # 断言：(e**(2*j)) 的虚数属性为 None
    assert (e**(2*j)).is_imaginary is None

    # 断言：(e**-1) 的虚数属性为 False
    assert (e**-1).is_imaginary is False
    # 断言：(e**2) 的虚数属性为 True
    assert (e**2).is_imaginary
    # 断言：(e**3) 的虚数属性为 False
    assert (e**3).is_imaginary is False
    # 断言：(e**4) 的虚数属性为 False
    assert (e**4).is_imaginary is False
    # 断言：(e**5) 的虚数属性为 False
    assert (e**5).is_imaginary is False
    # 断言：(e**-1) 的实数属性为 False
    assert (e**-1).is_real is False
    # 断言：(e**2) 的实数属性为 False
    assert (e**2).is_real is False
    # 断言：(e**3) 的实数属性为 False
    assert (e
    # 断言 i1 乘以 i2 的结果是实数
    assert (i1*i2).is_real is True

    # 断言 r 乘以 i1 乘以 i2 的结果是虚数
    assert (r*i1*i2).is_imaginary is False
    # 断言 r 乘以 i1 乘以 i2 的结果是实数
    assert (r*i1*i2).is_real is True

    # Github 的问题 5874:
    # 创建一个名为 nr 的符号，其实部为非实数、复数形式（例如 I 或 1 + I）
    nr = Symbol('nr', real=False, complex=True)
    # 创建一个名为 a 的符号，其实部为实数且非零
    a = Symbol('a', real=True, nonzero=True)
    # 创建一个名为 b 的符号，其实部为实数
    b = Symbol('b', real=True)
    # 断言 i1 乘以 nr 的结果是不确定是否为实数
    assert (i1*nr).is_real is None
    # 断言 a 乘以 nr 的结果是非实数
    assert (a*nr).is_real is False
    # 断言 b 乘以 nr 的结果是不确定是否为实数
    assert (b*nr).is_real is None

    # 创建一个名为 ni 的符号，其虚部为非虚数、复数形式（例如 2 或 1 + I）
    ni = Symbol('ni', imaginary=False, complex=True)
    # 创建一个名为 a 的符号，其实部为实数且非零
    a = Symbol('a', real=True, nonzero=True)
    # 创建一个名为 b 的符号，其实部为实数
    b = Symbol('b', real=True)
    # 断言 i1 乘以 ni 的结果是非实数
    assert (i1*ni).is_real is False
    # 断言 a 乘以 ni 的结果是不确定是否为实数
    assert (a*ni).is_real is None
    # 断言 b 乘以 ni 的结果是不确定是否为实数
    assert (b*ni).is_real is None
def test_Mul_hermitian_antihermitian():
    # 定义符号，其中 xz 和 yz 是零向量，且是反厄米的
    xz, yz = symbols('xz, yz', zero=True, antihermitian=True)
    # xf 和 yf 是非厄米和非反厄米的有限向量
    xf, yf = symbols('xf, yf', hermitian=False, antihermitian=False, finite=True)
    # xh 是厄米且非零向量，yh 是非厄米向量
    xh, yh = symbols('xh, yh', hermitian=True, antihermitian=False, nonzero=True)
    # xa 和 ya 是非厄米和反厄米的非零有限向量
    xa, ya = symbols('xa, ya', hermitian=False, antihermitian=True, zero=False, finite=True)
    
    # 下面的 assert 语句测试乘积的性质
    assert (xz*xh).is_hermitian is True  # xz * xh 是厄米的
    assert (xz*xh).is_antihermitian is True  # xz * xh 是反厄米的
    assert (xz*xa).is_hermitian is True  # xz * xa 是厄米的
    assert (xz*xa).is_antihermitian is True  # xz * xa 是反厄米的
    assert (xf*yf).is_hermitian is None  # xf * yf 不确定其是否厄米
    assert (xf*yf).is_antihermitian is None  # xf * yf 不确定其是否反厄米
    assert (xh*yh).is_hermitian is True  # xh * yh 是厄米的
    assert (xh*yh).is_antihermitian is False  # xh * yh 不是反厄米的
    assert (xh*ya).is_hermitian is False  # xh * ya 不是厄米的
    assert (xh*ya).is_antihermitian is True  # xh * ya 是反厄米的
    assert (xa*ya).is_hermitian is True  # xa * ya 是厄米的
    assert (xa*ya).is_antihermitian is False  # xa * ya 不是反厄米的

    # 定义符号 a, b, c, d, e1, e2, e3, e4, e5, e6 来测试更复杂的乘积情况
    a = Symbol('a', hermitian=True, zero=False)
    b = Symbol('b', hermitian=True)
    c = Symbol('c', hermitian=False)
    d = Symbol('d', antihermitian=True)
    e1 = Mul(a, b, c, evaluate=False)
    e2 = Mul(b, a, c, evaluate=False)
    e3 = Mul(a, b, c, d, evaluate=False)
    e4 = Mul(b, a, c, d, evaluate=False)
    e5 = Mul(a, c, evaluate=False)
    e6 = Mul(a, c, d, evaluate=False)
    
    # 下面的 assert 语句测试复杂乘积的性质
    assert e1.is_hermitian is None  # e1 的厄米性质未知
    assert e2.is_hermitian is None  # e2 的厄米性质未知
    assert e1.is_antihermitian is None  # e1 的反厄米性质未知
    assert e2.is_antihermitian is None  # e2 的反厄米性质未知
    assert e3.is_antihermitian is None  # e3 的反厄米性质未知
    assert e4.is_antihermitian is None  # e4 的反厄米性质未知
    assert e5.is_antihermitian is None  # e5 的反厄米性质未知
    assert e6.is_antihermitian is None  # e6 的反厄米性质未知


def test_Add_is_comparable():
    # 测试加法表达式的可比较性质
    assert (x + y).is_comparable is False  # x + y 不可比较
    assert (x + 1).is_comparable is False  # x + 1 不可比较
    assert (Rational(1, 3) - sqrt(8)).is_comparable is True  # 1/3 - sqrt(8) 可比较


def test_Mul_is_comparable():
    # 测试乘法表达式的可比较性质
    assert (x*y).is_comparable is False  # x * y 不可比较
    assert (x*2).is_comparable is False  # x * 2 不可比较
    assert (sqrt(2)*Rational(1, 3)).is_comparable is True  # sqrt(2) * 1/3 可比较


def test_Pow_is_comparable():
    # 测试幂表达式的可比较性质
    assert (x**y).is_comparable is False  # x ** y 不可比较
    assert (x**2).is_comparable is False  # x ** 2 不可比较
    assert (sqrt(Rational(1, 3))).is_comparable is True  # sqrt(1/3) 可比较


def test_Add_is_positive_2():
    # 测试加法表达式的正负性质
    e = Rational(1, 3) - sqrt(8)
    assert e.is_positive is False  # e 不是正数
    assert e.is_negative is True  # e 是负数

    e = pi - 1
    assert e.is_positive is True  # e 是正数
    assert e.is_negative is False  # e 不是负数


def test_Add_is_irrational():
    # 测试加法表达式是否为无理数
    i = Symbol('i', irrational=True)

    assert i.is_irrational is True  # i 是无理数
    assert i.is_rational is False  # i 不是有理数

    assert (i + 1).is_irrational is True  # i + 1 是无理数
    assert (i + 1).is_rational is False  # i + 1 不是有理数


def test_Mul_is_irrational():
    # 测试乘法表达式是否为无理数
    expr = Mul(1, 2, 3, evaluate=False)
    assert expr.is_irrational is False  # 1 * 2 * 3 不是无理数
    expr = Mul(1, I, I, evaluate=False)
    assert expr.is_rational is None  # 1 * I * I = -1 但不允许评估
    expr = Mul(sqrt(2), I, I, evaluate=False)
    assert expr.is_irrational is None  # sqrt(2) * I * I = -sqrt(2) 是无理数，但不能确定
def test_issue_3531():
    # 测试 GitHub 问题 #3531 和相关 PR #18116 的影响
    class MightyNumeric(tuple):
        # 自定义类型 MightyNumeric 继承自 tuple，并重载了右除操作符
        def __rtruediv__(self, other):
            return "something"

    # 断言 sympify(1) 与 MightyNumeric((1, 2)) 右除后的结果为 "something"
    assert sympify(1)/MightyNumeric((1, 2)) == "something"


def test_issue_3531b():
    # 测试 GitHub 问题 #3531b
    class Foo:
        # 定义类 Foo
        def __init__(self):
            self.field = 1.0

        # 重载乘法操作符
        def __mul__(self, other):
            self.field = self.field * other

        # 重载右乘法操作符
        def __rmul__(self, other):
            self.field = other * self.field

    # 创建 Foo 类的实例 f
    f = Foo()
    # 创建符号 x
    x = Symbol("x")
    # 断言 f 乘以 x 等于 x 乘以 f
    assert f*x == x*f


def test_bug3():
    # 测试 bug3
    a = Symbol("a")
    b = Symbol("b", positive=True)
    # 创建表达式 e 和 f
    e = 2*a + b
    f = b + 2*a
    # 断言 e 等于 f
    assert e == f


def test_suppressed_evaluation():
    # 测试抑制评估
    a = Add(0, 3, 2, evaluate=False)
    b = Mul(1, 3, 2, evaluate=False)
    c = Pow(3, 2, evaluate=False)
    # 断言 a 不等于 6，且其类型为 Add
    assert a != 6
    assert a.func is Add
    assert a.args == (0, 3, 2)
    # 断言 b 不等于 6，且其类型为 Mul
    assert b != 6
    assert b.func is Mul
    assert b.args == (1, 3, 2)
    # 断言 c 不等于 9，且其类型为 Pow
    assert c != 9
    assert c.func is Pow
    assert c.args == (3, 2)


def test_AssocOp_doit():
    # 测试 AssocOp 的 doit 方法
    a = Add(x,x, evaluate=False)
    b = Mul(y,y, evaluate=False)
    c = Add(b,b, evaluate=False)
    d = Mul(a,a, evaluate=False)
    # 测试深度为 False 的 doit 方法返回 Mul 类型和相应的参数
    assert c.doit(deep=False).func == Mul
    assert c.doit(deep=False).args == (2,y,y)
    # 测试默认的 doit 方法返回 Mul 类型和相应的参数
    assert c.doit().func == Mul
    assert c.doit().args == (2, Pow(y,2))
    # 测试深度为 False 的 doit 方法返回 Pow 类型和相应的参数
    assert d.doit(deep=False).func == Pow
    assert d.doit(deep=False).args == (a, 2*S.One)
    # 测试默认的 doit 方法返回 Mul 类型和相应的参数
    assert d.doit().func == Mul
    assert d.doit().args == (4*S.One, Pow(x,2))


def test_Add_Mul_Expr_args():
    # 测试 Add 和 Mul 类的 Expr 参数
    nonexpr = [Basic(), Poly(x, x), FiniteSet(x)]
    for typ in [Add, Mul]:
        for obj in nonexpr:
            # 警告测试，忽略 SymPyDeprecationWarning 的栈级别检查
            with warns(SymPyDeprecationWarning, test_stacklevel=False):
                # 使用非 Expr 参数创建 Add 或 Mul 实例
                typ(obj, 1)


def test_Add_as_coeff_mul():
    # 测试 Add 类的 as_coeff_mul 方法
    # 这些断言应该都是 (1, self)
    assert (x + 1).as_coeff_mul() == (1, (x + 1,))
    assert (x + 2).as_coeff_mul() == (1, (x + 2,))
    assert (x + 3).as_coeff_mul() == (1, (x + 3,))
    assert (x - 1).as_coeff_mul() == (1, (x - 1,))
    assert (x - 2).as_coeff_mul() == (1, (x - 2,))
    assert (x - 3).as_coeff_mul() == (1, (x - 3,))
    n = Symbol('n', integer=True)
    assert (n + 1).as_coeff_mul() == (1, (n + 1,))
    assert (n + 2).as_coeff_mul() == (1, (n + 2,))
    assert (n + 3).as_coeff_mul() == (1, (n + 3,))
    assert (n - 1).as_coeff_mul() == (1, (n - 1,))
    assert (n - 2).as_coeff_mul() == (1, (n - 2,))
    assert (n - 3).as_coeff_mul() == (1, (n - 3,))


def test_Pow_as_coeff_mul_doesnt_expand():
    # 测试 Pow 类的 as_coeff_mul 方法不会展开
    assert exp(x + y).as_coeff_mul() == (1, (exp(x + y),))
    assert exp(x + exp(x + y)) != exp(x + exp(x)*exp(y))


def test_issue_24751():
    # 测试 GitHub 问题 #24751
    expr = Add(-2, -3, evaluate=False)
    expr1 = Add(-1, expr, evaluate=False)
    # 断言将 expr1 转为整数后与预期结果相等
    assert int(expr1) == int((-3 - 2) - 1)


def test_issue_3514_18626():
    # 测试 GitHub 问题 #3514 和 #18626
    assert sqrt(S.Half) * sqrt(6) == 2 * sqrt(3)/2
    # 使用 SymPy 模块中的 S 对象，表示符号运算
    assert S.Half*sqrt(6)*sqrt(2) == sqrt(3)
    # 使用 SymPy 的 sqrt 函数计算平方根，验证等式
    assert sqrt(6)/2*sqrt(2) == sqrt(3)
    # 使用 SymPy 的 sqrt 函数计算平方根，验证等式
    assert sqrt(6)*sqrt(2)/2 == sqrt(3)
    # 使用 SymPy 的 sqrt 函数计算平方根，验证等式
    assert sqrt(8)**Rational(2, 3) == 2
    # 使用 SymPy 的 Rational 函数创建有理数，验证等式
# 测试函数，验证 Add 类和 Mul 类的 make_args 方法
def test_make_args():
    assert Add.make_args(x) == (x,)  # 验证 Add.make_args 方法处理单个参数的情况
    assert Mul.make_args(x) == (x,)  # 验证 Mul.make_args 方法处理单个参数的情况

    assert Add.make_args(x*y*z) == (x*y*z,)  # 验证 Add.make_args 方法处理多个参数乘积的情况
    assert Mul.make_args(x*y*z) == (x*y*z).args  # 验证 Mul.make_args 方法处理多个参数乘积的情况

    assert Add.make_args(x + y + z) == (x + y + z).args  # 验证 Add.make_args 方法处理多个参数加法的情况
    assert Mul.make_args(x + y + z) == (x + y + z,)  # 验证 Mul.make_args 方法处理多个参数加法的情况

    assert Add.make_args((x + y)**z) == ((x + y)**z,)  # 验证 Add.make_args 方法处理幂运算的情况
    assert Mul.make_args((x + y)**z) == ((x + y)**z,)  # 验证 Mul.make_args 方法处理幂运算的情况


# 测试函数，验证符号问题
def test_issue_5126():
    assert (-2)**x*(-3)**x != 6**x  # 验证幂运算符号的问题
    i = Symbol('i', integer=1)
    assert (-2)**i*(-3)**i == 6**i  # 验证幂运算符号的问题


# 测试函数，验证有理数的内容原始性
def test_Rational_as_content_primitive():
    c, p = S.One, S.Zero
    assert (c*p).as_content_primitive() == (c, p)  # 验证乘积的内容原始性
    c, p = S.Half, S.One
    assert (c*p).as_content_primitive() == (c, p)  # 验证乘积的内容原始性


# 测试函数，验证 Add 类的内容原始性
def test_Add_as_content_primitive():
    assert (x + 2).as_content_primitive() == (1, x + 2)  # 验证加法的内容原始性

    assert (3*x + 2).as_content_primitive() == (1, 3*x + 2)  # 验证加法的内容原始性
    assert (3*x + 3).as_content_primitive() == (3, x + 1)  # 验证加法的内容原始性
    assert (3*x + 6).as_content_primitive() == (3, x + 2)  # 验证加法的内容原始性

    assert (3*x + 2*y).as_content_primitive() == (1, 3*x + 2*y)  # 验证加法的内容原始性
    assert (3*x + 3*y).as_content_primitive() == (3, x + y)  # 验证加法的内容原始性
    assert (3*x + 6*y).as_content_primitive() == (3, x + 2*y)  # 验证加法的内容原始性

    assert (3/x + 2*x*y*z**2).as_content_primitive() == (1, 3/x + 2*x*y*z**2)  # 验证加法的内容原始性
    assert (3/x + 3*x*y*z**2).as_content_primitive() == (3, 1/x + x*y*z**2)  # 验证加法的内容原始性
    assert (3/x + 6*x*y*z**2).as_content_primitive() == (3, 1/x + 2*x*y*z**2)  # 验证加法的内容原始性

    assert (2*x/3 + 4*y/9).as_content_primitive() == \
        (Rational(2, 9), 3*x + 2*y)  # 验证加法的内容原始性
    assert (2*x/3 + 2.5*y).as_content_primitive() == \
        (Rational(1, 3), 2*x + 7.5*y)  # 验证加法的内容原始性

    # 系数可能会排序到位置 0 之外
    p = 3 + x + y
    assert (2*p).expand().as_content_primitive() == (2, p)  # 验证乘法的内容原始性
    assert (2.0*p).expand().as_content_primitive() == (1, 2.*p)  # 验证乘法的内容原始性
    p *= -1
    assert (2*p).expand().as_content_primitive() == (2, p)  # 验证乘法的内容原始性


# 测试函数，验证 Mul 类的内容原始性
def test_Mul_as_content_primitive():
    assert (2*x).as_content_primitive() == (2, x)  # 验证乘法的内容原始性
    assert (x*(2 + 2*x)).as_content_primitive() == (2, x*(1 + x))  # 验证乘法的内容原始性
    assert (x*(2 + 2*y)*(3*x + 3)**2).as_content_primitive() == \
        (18, x*(1 + y)*(x + 1)**2)  # 验证乘法的内容原始性
    assert ((2 + 2*x)**2*(3 + 6*x) + S.Half).as_content_primitive() == \
        (S.Half, 24*(x + 1)**2*(2*x + 1) + 1)  # 验证乘法的内容原始性


# 测试函数，验证 Pow 类的内容原始性
def test_Pow_as_content_primitive():
    assert (x**y).as_content_primitive() == (1, x**y)  # 验证幂运算的内容原始性
    assert ((2*x + 2)**y).as_content_primitive() == \
        (1, (Mul(2, (x + 1), evaluate=False))**y)  # 验证幂运算的内容原始性
    assert ((2*x + 2)**3).as_content_primitive() == (8, (x + 1)**3)  # 验证幂运算的内容原始性


# 测试函数，验证问题 5460
def test_issue_5460():
    u = Mul(2, (1 + x), evaluate=False)
    assert (2 + u).args == (2, u)  # 验证问题 5460


# 测试函数，验证产品的无理数
def test_product_irrational():
    assert (I*pi).is_irrational is False  # 验证无理数乘积
    # 以下原先是从上述 bug 推导出的：
    assert (I*pi).is_positive is False  # 验证无理数乘积的正性


# 测试函数，验证问题 5919
def test_issue_5919():
    assert (x/(y*(1 + y))).expand() == x/(y**2 + y)  # 验证问题 5919


# 测试函数，验证 Mod 运算符
def test_Mod():
    assert Mod(x, 1).func is Mod  # 验证 Mod 运算符
    assert pi % pi is S.Zero  # 验证 Mod 运算符
    assert Mod(5, 3) == 2  # 验证 Mod 运算符
    assert Mod(-5, 3) == 1  # 验证 Mod 运算符
    # 断言：对于 Mod 类的实例化，计算 5 % -3 应为 -1
    assert Mod(5, -3) == -1
    # 断言：对于 Mod 类的实例化，计算 -5 % -3 应为 -2
    assert Mod(-5, -3) == -2
    # 断言：对于 Mod 类的实例化，传入浮点数和 evaluate=False 参数，返回 Mod 类型对象
    assert type(Mod(3.2, 2, evaluate=False)) == Mod
    # 断言：验证数学表达式 5 % x 是否等于 Mod(5, x)
    assert 5 % x == Mod(5, x)
    # 断言：验证数学表达式 x % 5 是否等于 Mod(x, 5)
    assert x % 5 == Mod(x, 5)
    # 断言：验证数学表达式 x % y 是否等于 Mod(x, y)
    assert x % y == Mod(x, y)
    # 断言：验证 Mod(x % y) 在给定符号代入后计算结果为 2
    assert (x % y).subs({x: 5, y: 3}) == 2
    # 断言：验证 nan 与任何数的 Mod 结果均为 nan
    assert Mod(nan, 1) is nan
    assert Mod(1, nan) is nan
    assert Mod(nan, nan) is nan

    # 断言：验证 Mod(0, x) 应为 0
    assert Mod(0, x) == 0
    # 使用 raises 检查除以 0 的情况
    with raises(ZeroDivisionError):
        Mod(x, 0)

    # 创建整数符号 k 和正整数符号 m
    k = Symbol('k', integer=True)
    m = Symbol('m', integer=True, positive=True)
    # 断言：验证幂运算中的 Mod 结果类型为 Mod
    assert (x**m % x).func is Mod
    assert (k**(-m) % k).func is Mod
    # 断言：验证 k**m % k 等于 0
    assert k**m % k == 0
    assert (-2*k)**m % k == 0

    # 浮点数处理
    point3 = Float(3.3) % 1
    # 断言：验证浮点数运算的 Mod 结果
    assert (x - 3.3) % 1 == Mod(1.*x + 1 - point3, 1)
    assert Mod(-3.3, 1) == 1 - point3
    assert Mod(0.7, 1) == Float(0.7)
    e = Mod(1.3, 1)
    assert comp(e, .3) and e.is_Float
    e = Mod(1.3, .7)
    assert comp(e, .6) and e.is_Float
    e = Mod(1.3, Rational(7, 10))
    assert comp(e, .6) and e.is_Float
    e = Mod(Rational(13, 10), 0.7)
    assert comp(e, .6) and e.is_Float
    e = Mod(Rational(13, 10), Rational(7, 10))
    assert comp(e, .6) and e.is_Rational

    # 检查数值验证
    r2 = sqrt(2)
    r3 = sqrt(3)
    for i in [-r3, -r2, r2, r3]:
        for j in [-r3, -r2, r2, r3]:
            assert verify_numerically(i % j, i.n() % j.n())
    for _x in range(4):
        for _y in range(9):
            reps = [(x, _x), (y, _y)]
            assert Mod(3*x + y, 9).subs(reps) == (3*_x + _y) % 9

    # 嵌套处理
    t = Symbol('t', real=True)
    assert Mod(Mod(x, t), t) == Mod(x, t)
    assert Mod(-Mod(x, t), t) == Mod(-x, t)
    assert Mod(Mod(x, 2*t), t) == Mod(x, t)
    assert Mod(-Mod(x, 2*t), t) == Mod(-x, t)
    assert Mod(Mod(x, t), 2*t) == Mod(x, t)
    assert Mod(-Mod(x, t), -2*t) == -Mod(x, t)
    for i in [-4, -2, 2, 4]:
        for j in [-4, -2, 2, 4]:
            for k in range(4):
                assert Mod(Mod(x, i), j).subs({x: k}) == (k % i) % j
                assert Mod(-Mod(x, i), j).subs({x: k}) == -(k % i) % j

    # 已知差异
    assert Mod(5*sqrt(2), sqrt(5)) == 5*sqrt(2) - 3*sqrt(5)
    p = symbols('p', positive=True)
    assert Mod(2, p + 3) == 2
    assert Mod(-2, p + 3) == p + 1
    assert Mod(2, -p - 3) == -p - 1
    assert Mod(-2, -p - 3) == -2
    assert Mod(p + 5, p + 3) == 2
    assert Mod(-p - 5, p + 3) == p + 1
    assert Mod(p + 5, -p - 3) == -p - 1
    assert Mod(-p - 5, -p - 3) == -2
    assert Mod(p + 1, p - 1).func is Mod

    # 处理求和
    assert (x + 3) % 1 == Mod(x, 1)
    assert (x + 3.0) % 1 == Mod(1.*x, 1)
    assert (x - S(33)/10) % 1 == Mod(x + S(7)/10, 1)

    a = Mod(.6*x + y, .3*y)
    b = Mod(0.1*y + 0.6*x, 0.3*y)
    # 断言：验证 a 和 b 在系数的精度为 1e-14 时相等
    eps = 1e-14
    assert abs((a.args[0] - b.args[0]).subs({x: 1, y: 1})) < eps
    assert abs((a.args[1] - b.args[1]).subs({x: 1, y: 1})) < eps

    # 断言：验证 (x + 1) % x 等于 1 % x
    assert (x + 1) % x == 1 % x
    # 断言：验证 (x + y) % x 等于 y % x
    assert (x + y) % x == y % x
    # 断言：验证 (x + y + 2) % x 是否等于 (y + 2) % x
    assert (x + y + 2) % x == (y + 2) % x
    # 断言：验证 (a + 3*x + 1) % (2*x) 是否等于 Mod(a + x + 1, 2*x)
    assert (a + 3*x + 1) % (2*x) == Mod(a + x + 1, 2*x)
    # 断言：验证 (12*x + 18*y) % (3*x) 是否等于 3*Mod(6*y, x)
    assert (12*x + 18*y) % (3*x) == 3*Mod(6*y, x)

    # gcd extraction
    # 断言：验证 (-3*x) % (-2*y) 是否等于 -Mod(3*x, 2*y)
    assert (-3*x) % (-2*y) == -Mod(3*x, 2*y)
    # 断言：验证 (.6*pi) % (.3*x*pi) 是否等于 0.3*pi*Mod(2, x)
    assert (.6*pi) % (.3*x*pi) == 0.3*pi*Mod(2, x)
    # 断言：验证 (.6*pi) % (.31*x*pi) 是否等于 pi*Mod(0.6, 0.31*x)
    assert (.6*pi) % (.31*x*pi) == pi*Mod(0.6, 0.31*x)
    # 断言：验证 (6*pi) % (.3*x*pi) 是否等于 0.3*pi*Mod(20, x)
    assert (6*pi) % (.3*x*pi) == 0.3*pi*Mod(20, x)
    # 断言：验证 (6*pi) % (.31*x*pi) 是否等于 pi*Mod(6, 0.31*x)
    assert (6*pi) % (.31*x*pi) == pi*Mod(6, 0.31*x)
    # 断言：验证 (6*pi) % (.42*x*pi) 是否等于 pi*Mod(6, 0.42*x)
    assert (6*pi) % (.42*x*pi) == pi*Mod(6, 0.42*x)
    # 断言：验证 (12*x) % (2*y) 是否等于 2*Mod(6*x, y)
    assert (12*x) % (2*y) == 2*Mod(6*x, y)
    # 断言：验证 (12*x) % (3*5*y) 是否等于 3*Mod(4*x, 5*y)
    assert (12*x) % (3*5*y) == 3*Mod(4*x, 5*y)
    # 断言：验证 (12*x) % (15*x*y) 是否等于 3*x*Mod(4, 5*y)
    assert (12*x) % (15*x*y) == 3*x*Mod(4, 5*y)
    # 断言：验证 (-2*pi) % (3*pi) 是否等于 pi
    assert (-2*pi) % (3*pi) == pi
    # 断言：验证 (2*x + 2) % (x + 1) 是否等于 0
    assert (2*x + 2) % (x + 1) == 0
    # 断言：验证 (x*(x + 1)) % (x + 1) 是否等于 (x + 1)*Mod(x, 1)
    assert (x*(x + 1)) % (x + 1) == (x + 1)*Mod(x, 1)
    # 断言：验证 Mod(5.0*x, 0.1*y) 是否等于 0.1*Mod(50*x, y)
    assert Mod(5.0*x, 0.1*y) == 0.1*Mod(50*x, y)
    # 符号 i 被定义为整数
    i = Symbol('i', integer=True)
    # 断言：验证 (3*i*x) % (2*i*y) 是否等于 i*Mod(3*x, 2*y)
    assert (3*i*x) % (2*i*y) == i*Mod(3*x, 2*y)
    # 断言：验证 Mod(4*i, 4) 是否等于 0
    assert Mod(4*i, 4) == 0

    # issue 8677
    # 符号 n 被定义为正整数
    n = Symbol('n', integer=True, positive=True)
    # 断言：验证 factorial(n) % n 是否等于 0
    assert factorial(n) % n == 0
    # 断言：验证 factorial(n + 2) % n 是否等于 0
    assert factorial(n + 2) % n == 0
    # 断言：验证 (factorial(n + 4) % (n + 5)).func 是否为 Mod 类型
    assert (factorial(n + 4) % (n + 5)).func is Mod

    # Wilson's theorem
    # 断言：验证 factorial(18042) % 18043 是否等于 18042
    assert factorial(18042, evaluate=False) % 18043 == 18042
    # 符号 p 被定义为质数
    p = Symbol('n', prime=True)
    # 断言：验证 factorial(p - 1) % p 是否等于 p - 1
    assert factorial(p - 1) % p == p - 1
    # 断言：验证 factorial(p - 1) % -p 是否等于 -1
    assert factorial(p - 1) % -p == -1
    # 断言：验证 (factorial(3) % 4).doit() 是否等于 2
    assert (factorial(3, evaluate=False) % 4).doit() == 2
    # 符号 n 被定义为合数和奇数
    n = Symbol('n', composite=True, odd=True)
    # 断言：验证 factorial(n - 1) % n 是否等于 0
    assert factorial(n - 1) % n == 0

    # symbolic with known parity
    # 符号 n 被定义为偶数
    n = Symbol('n', even=True)
    # 断言：验证 Mod(n, 2) 是否等于 0
    assert Mod(n, 2) == 0
    # 符号 n 被定义为奇数
    n = Symbol('n', odd=True)
    # 断言：验证 Mod(n, 2) 是否等于 1
    assert Mod(n, 2) == 1

    # issue 10963
    # 断言：验证 (x**6000%400).args[1] 是否等于 400
    assert (x**6000 % 400).args[1] == 400

    # issue 13543
    # 断言：验证 Mod(Mod(x + 1, 2) + 1, 2) 是否等于 Mod(x, 2)
    assert Mod(Mod(x + 1, 2) + 1, 2) == Mod(x, 2)

    x1 = Symbol('x1', integer=True)
    # 断言：验证 Mod(Mod(x1 + 2, 4)*(x1 + 4), 4) 是否等于 Mod(x1*(x1 + 2), 4)
    assert Mod(Mod(x1 + 2, 4)*(x1 + 4), 4) == Mod(x1*(x1 + 2), 4)
    # 断言：验证 Mod(Mod(x1 + 2, 4)*4, 4) 是否等于 0
    assert Mod(Mod(x1 + 2, 4)*4, 4) == 0

    # issue 15493
    i, j = symbols('i j', integer=True, positive=True)
    # 断言：验证 Mod(3*i, 2) 是否等于 Mod(i, 2)
    assert Mod(3*i, 2) == Mod(i, 2)
    # 断言：验证 Mod(8*i/j, 4) 是否等于 4*Mod(2*i/j, 1)
    assert Mod(8*i/j, 4) == 4*Mod(2*i/j, 1)
    # 断言：验证 Mod(8*i, 4) 是否等于 0
    assert Mod(8*i, 4) == 0

    # rewrite
    # 断言：验证 Mod(x, y).rewrite(floor) 是否等于 x - y*floor(x/y)
    assert Mod(x, y).rewrite(floor) == x - y*floor(x/y)
    # 断言：验证 ((x - Mod(x, y))/y).rewrite(floor) 是否等于 floor(x/y)
    assert ((x - Mod(x
def test_Mod_Pow():
    # modular exponentiation
    assert isinstance(Mod(Pow(2, 2, evaluate=False), 3), Integer)

    assert Mod(Pow(4, 13, evaluate=False), 497) == Mod(Pow(4, 13), 497)
    assert Mod(Pow(2, 10000000000, evaluate=False), 3) == 1
    assert Mod(Pow(32131231232, 9**10**6, evaluate=False),10**12) == \
        pow(32131231232,9**10**6,10**12)
    assert Mod(Pow(33284959323, 123**999, evaluate=False),11**13) == \
        pow(33284959323,123**999,11**13)
    assert Mod(Pow(78789849597, 333**555, evaluate=False),12**9) == \
        pow(78789849597,333**555,12**9)

    # modular nested exponentiation
    expr = Pow(2, 2, evaluate=False)
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 16
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 6487
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 32191
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 18016
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 5137

    expr = Pow(2, 2, evaluate=False)
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 16
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 256
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 6487
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 38281
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 15928

    expr = Pow(2, 2, evaluate=False)
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 256
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 9229
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 25708
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 26608
    expr = Pow(expr, expr, evaluate=False)
    # XXX This used to fail in a nondeterministic way because of overflow
    # error.
    assert Mod(expr, 3**10) == 1966


def test_Mod_is_integer():
    p = Symbol('p', integer=True)
    q1 = Symbol('q1', integer=True)
    q2 = Symbol('q2', integer=True, nonzero=True)
    assert Mod(x, y).is_integer is None
    assert Mod(p, q1).is_integer is None
    assert Mod(x, q2).is_integer is None
    assert Mod(p, q2).is_integer


def test_Mod_is_nonposneg():
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True, positive=True)
    assert (n%3).is_nonnegative
    assert Mod(n, -3).is_nonpositive
    assert Mod(n, k).is_nonnegative
    assert Mod(n, -k).is_nonpositive
    assert Mod(k, n).is_nonnegative is None


def test_issue_6001():
    A = Symbol("A", commutative=False)
    eq = A + A**2
    # it doesn't matter whether it's True or False; they should
    # just all be the same
    assert (
        eq.is_commutative ==
        (eq + 1).is_commutative ==
        (A + 1).is_commutative)

    B = Symbol("B", commutative=False)
    # Although commutative terms could cancel we return True
    # 检查表达式 sqrt(2)*A 是否是可交换的，预期结果是 False
    assert (sqrt(2)*A).is_commutative is False
    # 检查表达式 sqrt(2)*A*B 是否是可交换的，预期结果是 False
    assert (sqrt(2)*A*B).is_commutative is False
# 定义一个测试函数，用于验证极坐标相关功能
def test_polar():
    # 从sympy库中导入极坐标相关函数
    from sympy.functions.elementary.complexes import polar_lift
    # 创建一个符号变量p，标记为极坐标类型
    p = Symbol('p', polar=True)
    # 创建一个普通符号变量x
    x = Symbol('x')
    # 断言p是极坐标
    assert p.is_polar
    # 断言x不是极坐标
    assert x.is_polar is None
    # 断言S.One（即1）不是极坐标
    assert S.One.is_polar is None
    # 断言(p**x)是极坐标
    assert (p**x).is_polar is True
    # 断言(x**p)不是极坐标
    assert (x**p).is_polar is None
    # 断言((2*p)**x)是极坐标
    assert ((2*p)**x).is_polar is True
    # 断言(2*p)是极坐标
    assert (2*p).is_polar is True
    # 断言(-2*p)不是极坐标
    assert (-2*p).is_polar is not True
    # 断言(polar_lift(-2)*p)是极坐标
    assert (polar_lift(-2)*p).is_polar is True

    # 创建一个符号变量q，标记为极坐标类型
    q = Symbol('q', polar=True)
    # 断言(p*q)**2等于p**2 * q**2
    assert (p*q)**2 == p**2 * q**2
    # 断言(2*q)**2等于4 * q**2
    assert (2*q)**2 == 4 * q**2
    # 断言((p*q)**x).expand()等于p**x * q**x
    assert ((p*q)**x).expand() == p**x * q**x


def test_issue_6040():
    # 创建两个Pow对象，禁止求值
    a, b = Pow(1, 2, evaluate=False), S.One
    # 断言a不等于b
    assert a != b
    # 断言b不等于a
    assert b != a
    # 断言a不等于b
    assert not (a == b)
    # 断言b不等于a
    assert not (b == a)


def test_issue_6082():
    # 比较是对称的
    assert Basic.compare(Max(x, 1), Max(x, 2)) == \
      - Basic.compare(Max(x, 2), Max(x, 1))
    # 相等的表达式比较结果为0
    assert Basic.compare(Max(x, 1), Max(x, 1)) == 0
    # 基本子类型（如Max）与标准类型比较结果不同
    assert Basic.compare(Max(1, x), frozenset((1, x))) != 0


def test_issue_6077():
    # 一系列指数运算和乘法运算的断言
    assert x**2.0/x == x**1.0
    assert x/x**2.0 == x**-1.0
    assert x*x**2.0 == x**3.0
    assert x**1.5*x**2.5 == x**4.0

    assert 2**(2.0*x)/2**x == 2**(1.0*x)
    assert 2**x/2**(2.0*x) == 2**(-1.0*x)
    assert 2**x*2**(2.0*x) == 2**(3.0*x)
    assert 2**(1.5*x)*2**(2.5*x) == 2**(4.0*x)


def test_mul_flatten_oo():
    # 创建正数符号变量p
    p = symbols('p', positive=True)
    # 创建负数符号变量n和m
    n, m = symbols('n,m', negative=True)
    # 创建虚数符号变量x_im
    x_im = symbols('x_im', imaginary=True)
    # 断言n乘以oo等于-oo
    assert n*oo is -oo
    # 断言n乘以m乘以oo等于oo
    assert n*m*oo is oo
    # 断言p乘以oo等于oo
    assert p*oo is oo
    # 断言x_im乘以oo不等于I乘以oo，因为i可以是+/- 3*i -> +/-oo


def test_add_flatten():
    # 参见https://github.com/sympy/sympy/issues/2633#issuecomment-29545524
    a = oo + I*oo
    b = oo - I*oo
    # 断言a加b等于nan
    assert a + b is nan
    # 断言a减b等于nan
    assert a - b is nan
    # FIXME: 这将评估为:
    #   >>> 1/a
    #   0*(oo + oo*I)
    # 应该不简化为0，应在Pow.eval中修复
    #assert (1/a).simplify() == (1/b).simplify() == 0

    # 创建一个Pow对象a，禁止求值
    a = Pow(2, 3, evaluate=False)
    # 断言a加a等于16
    assert a + a == 16


def test_issue_5160_6087_6089_6090():
    # issue 6087
    # 断言((-2*x*y**y)**3.2).n(2)等于(2**3.2*(-x*y**y)**3.2).n(2)
    assert ((-2*x*y**y)**3.2).n(2) == (2**3.2*(-x*y**y)**3.2).n(2)
    # issue 6089
    # 创建非交换符号变量A, B, C
    A, B, C = symbols('A,B,C', commutative=False)
    # 断言(2.*B*C)**3等于8.0*(B*C)**3
    assert (2.*B*C)**3 == 8.0*(B*C)**3
    # 断言(-2.*B*C)**3等于-8.0*(B*C)**3
    assert (-2.*B*C)**3 == -8.0*(B*C)**3
    # 断言(-2*B*C)**2等于4*(B*C)**2
    assert (-2*B*C)**2 == 4*(B*C)**2
    # issue 5160
    # 断言sqrt(-1.0*x)等于1.0*sqrt(-x)
    assert sqrt(-1.0*x) == 1.0*sqrt(-x)
    # 断言sqrt(1.0*x)等于1.0*sqrt(x)
    assert sqrt(1.0*x) == 1.0*sqrt(x)
    # issue 6090
    # 断言(-2*x*y*A*B)**2等于4*x**2*y**2*(A*B)**2
    assert (-2*x*y*A*B)**2 == 4*x**2*y**2*(A*B)**2


def test_float_int_round():
    # 断言int(float(sqrt(10)))等于int(sqrt(10))
    assert int(float(sqrt(10))) == int(sqrt(10))
    # 断言int(pi**1000) % 10等于2
    assert int(pi**1000) % 10 == 2
    # 断言int(Float('1.123456789012345678901234567890e20', ''))等于int(112345678901234567890)
    assert int(Float('1.123456789012345678901234567890e20', '')) == \
        int(112345678901234567890)
    # 断言int(Float('1.123456789012345678901234567890e25', ''))等于int(11234567890123456789012345)
    assert int(Float('1.123456789012345678901234567890e25', '')) == \
        int(11234567890123456789012345)
    # 使用 Float 类创建一个浮点数，由于 decimal 参数为空字符串，不是以精确的以 000000 结尾的整数
    assert int(Float('1.123456789012345678901234567890e35', '')) == \
        112345678901234567890123456789000192
    # 同上，验证浮点数转换为整数后的正确性
    assert int(Float('123456789012345678901234567890e5', '')) == \
        12345678901234567890123456789000000
    # 使用 Float 类创建浮点数，然后将其转换为整数，验证转换结果的正确性
    assert Integer(Float('1.123456789012345678901234567890e20', '')) == \
        112345678901234567890
    # 同上，验证不同的浮点数转换为整数后的正确性
    assert Integer(Float('1.123456789012345678901234567890e25', '')) == \
        11234567890123456789012345
    # 使用 Float 类创建一个浮点数，由于 decimal 参数为空字符串，不是以精确的以 000000 结尾的整数
    assert Integer(Float('1.123456789012345678901234567890e35', '')) == \
        112345678901234567890123456789000192
    # 同上，验证浮点数转换为整数后的正确性
    assert Integer(Float('123456789012345678901234567890e5', '')) == \
        12345678901234567890123456789000000
    # 验证两个浮点数是否具有相同的值和精度
    assert same_and_same_prec(Float('123000e-2',''), Float('1230.00', ''))
    # 验证两个浮点数是否具有相同的值和精度
    assert same_and_same_prec(Float('123000e2',''), Float('12300000', ''))

    # 验证将有理数与整数相加后是否得到正确的整数结果
    assert int(1 + Rational('.9999999999999999999999999')) == 1
    # 验证 pi 除以 1e20 后是否得到正确的整数结果
    assert int(pi/1e20) == 0
    # 验证 1 加上 pi 除以 1e20 后是否得到正确的整数结果
    assert int(1 + pi/1e20) == 1
    # 使用 Add 类对浮点数进行加法运算，evaluate 参数为 False，验证加法的正确性
    assert int(Add(1.2, -2, evaluate=False)) == int(1.2 - 2)
    # 使用 Add 类对浮点数进行加法运算，evaluate 参数为 False，验证加法的正确性
    assert int(Add(1.2, +2, evaluate=False)) == int(1.2 + 2)
    # 使用 Add 类对浮点数进行加法运算，evaluate 参数为 False，验证加法的正确性
    assert int(Add(1 + Float('.99999999999999999', ''), evaluate=False)) == 1
    # 验证尝试对变量 x 使用 float() 函数时是否会引发 TypeError 异常
    raises(TypeError, lambda: float(x))
    # 验证尝试对负数的平方根求浮点数时是否会引发 TypeError 异常
    raises(TypeError, lambda: float(sqrt(-1)))

    # 验证将整数、余弦平方和正弦平方相加后是否得到正确的整数结果
    assert int(12345678901234567890 + cos(1)**2 + sin(1)**2) == \
        12345678901234567891
def test_issue_6611a():
    assert Mul.flatten([3**Rational(1, 3),
        Pow(-Rational(1, 9), Rational(2, 3), evaluate=False)]) == \
        ([Rational(1, 3), (-1)**Rational(2, 3)], [], None)

# 测试函数，用于验证 Mul.flatten 方法的行为
# 在这个例子中，验证了对包含有理数幂和 Pow 对象的 Mul 对象进行展平的功能


def test_denest_add_mul():
    # when working with evaluated expressions make sure they denest
    eq = x + 1
    eq = Add(eq, 2, evaluate=False)
    eq = Add(eq, 2, evaluate=False)
    assert Add(*eq.args) == x + 5
    eq = x*2
    eq = Mul(eq, 2, evaluate=False)
    eq = Mul(eq, 2, evaluate=False)
    assert Mul(*eq.args) == 8*x
    # but don't let them denest unnecessarily
    eq = Mul(-2, x - 2, evaluate=False)
    assert 2*eq == Mul(-4, x - 2, evaluate=False)
    assert -eq == Mul(2, x - 2, evaluate=False)

# 测试函数，用于验证加法和乘法表达式的去嵌套操作
# 通过添加和乘法表达式的方式，使用 evaluate=False 来防止自动化简


def test_mul_coeff():
    # It is important that all Numbers be removed from the seq;
    # This can be tricky when powers combine to produce those numbers
    p = exp(I*pi/3)
    assert p**2*x*p*y*p*x*p**2 == x**2*y

# 测试函数，用于验证乘法表达式中的系数组合和简化操作
# 在这个例子中，验证了指数函数和符号的乘法结合


def test_mul_zero_detection():
    nz = Dummy(real=True, zero=False)
    r = Dummy(extended_real=True)
    c = Dummy(real=False, complex=True)
    c2 = Dummy(real=False, complex=True)
    i = Dummy(imaginary=True)
    e = nz*r*c
    assert e.is_imaginary is None
    assert e.is_extended_real is None
    e = nz*c
    assert e.is_imaginary is None
    assert e.is_extended_real is False
    e = nz*i*c
    assert e.is_imaginary is False
    assert e.is_extended_real is None
    # check for more than one complex; it is important to use
    # uniquely named Symbols to ensure that two factors appear
    # e.g. if the symbols have the same name they just become
    # a single factor, a power.
    e = nz*i*c*c2
    assert e.is_imaginary is None
    assert e.is_extended_real is None

    # _eval_is_extended_real and _eval_is_zero both employ trapping of the
    # zero value so args should be tested in both directions and
    # TO AVOID GETTING THE CACHED RESULT, Dummy MUST BE USED

    # real is unknown
    def test(z, b, e):
        if z.is_zero and b.is_finite:
            assert e.is_extended_real and e.is_zero
        else:
            assert e.is_extended_real is None
            if b.is_finite:
                if z.is_zero:
                    assert e.is_zero
                else:
                    assert e.is_zero is None
            elif b.is_finite is False:
                if z.is_zero is None:
                    assert e.is_zero is None
                else:
                    assert e.is_zero is False

    for iz, ib in product(*[[True, False, None]]*2):
        z = Dummy('z', nonzero=iz)
        b = Dummy('f', finite=ib)
        e = Mul(z, b, evaluate=False)
        test(z, b, e)
        z = Dummy('nz', nonzero=iz)
        b = Dummy('f', finite=ib)
        e = Mul(b, z, evaluate=False)
        test(z, b, e)

    # real is True
    def test(z, b, e):
        if z.is_zero and not b.is_finite:
            assert e.is_extended_real is None
        else:
            assert e.is_extended_real is True

# 测试函数，用于验证乘法表达式中的零检测和扩展实数性质
# 在这个例子中，通过 Dummy 符号模拟不同的符号属性，以测试乘法表达式的零性质和扩展实数性质
    # 使用 itertools.product 生成两个元素的笛卡尔积，每个元素是 [True, False, None]
    for iz, ib in product(*[[True, False, None]]*2):
        # 创建一个虚拟符号 'z'，其非零属性根据 iz 的值确定，且为扩展实数
        z = Dummy('z', nonzero=iz, extended_real=True)
        # 创建一个虚拟符号 'b'，其有限属性根据 ib 的值确定，且为扩展实数
        b = Dummy('b', finite=ib, extended_real=True)
        # 创建一个乘法表达式 Mul(z, b)，但不进行求值
        e = Mul(z, b, evaluate=False)
        # 调用测试函数 test，传入 z、b 和乘法表达式 e
        test(z, b, e)
        
        # 重新创建虚拟符号 'z'，其非零属性根据 iz 的值确定，且为扩展实数
        z = Dummy('z', nonzero=iz, extended_real=True)
        # 重新创建虚拟符号 'b'，其有限属性根据 ib 的值确定，且为扩展实数
        b = Dummy('b', finite=ib, extended_real=True)
        # 创建一个乘法表达式 Mul(b, z)，但不进行求值
        e = Mul(b, z, evaluate=False)
        # 再次调用测试函数 test，传入 z、b 和乘法表达式 e
        test(z, b, e)
def test_Mul_with_zero_infinite():
    # 创建两个虚拟对象，一个表示零，一个表示无穷大
    zer = Dummy(zero=True)
    inf = Dummy(finite=False)

    # 乘法操作，不进行求值
    e = Mul(zer, inf, evaluate=False)
    # 断言结果未知
    assert e.is_extended_positive is None
    assert e.is_hermitian is None

    # 乘法操作，不进行求值
    e = Mul(inf, zer, evaluate=False)
    # 断言结果未知
    assert e.is_extended_positive is None
    assert e.is_hermitian is None


def test_Mul_does_not_cancel_infinities():
    # 定义符号变量 a 和 b
    a, b = symbols('a b')
    # 断言 (zoo + 3*a)/(3*a + zoo) 的结果是 NaN
    assert ((zoo + 3*a)/(3*a + zoo)) is nan
    # 断言 (b - oo)/(b - oo) 的结果是 NaN
    assert ((b - oo)/(b - oo)) is nan
    # issue 13904
    # 定义表达式 expr
    expr = (1/(a+b) + 1/(a-b))/(1/(a+b) - 1/(a-b))
    # 断言在 b 替换为 a 后的结果是 NaN
    assert expr.subs(b, a) is nan


def test_Mul_does_not_distribute_infinity():
    # 定义符号变量 a 和 b
    a, b = symbols('a b')
    # 断言 (1 + I)*oo 是乘法
    assert ((1 + I)*oo).is_Mul
    # 断言 (a + b)*(-oo) 是乘法
    assert ((a + b)*(-oo)).is_Mul
    # 断言 (a + 1)*zoo 是乘法
    assert ((a + 1)*zoo).is_Mul
    # 断言 (1 + I)*oo 是有限的结果为 False
    assert ((1 + I)*oo).is_finite is False
    # 定义 z = (1 + I)*oo
    z = (1 + I)*oo
    # 断言 (1 - I)*z 展开后的结果是 oo
    assert ((1 - I)*z).expand() is oo


def test_issue_8247_8354():
    from sympy.functions.elementary.trigonometric import tan
    # 定义复杂表达式 z
    z = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    # 断言 z 是否为非正数（实际上是零）
    assert z.is_positive is False  # it's 0
    # 定义复杂表达式 z
    z = S('''-2**(1/3)*(3*sqrt(93) + 29)**2 - 4*(3*sqrt(93) + 29)**(4/3) +
        12*sqrt(93)*(3*sqrt(93) + 29)**(1/3) + 116*(3*sqrt(93) + 29)**(1/3) +
        174*2**(1/3)*sqrt(93) + 1678*2**(1/3)''')
    # 断言 z 是否为非正数（实际上是零）
    assert z.is_positive is False  # it's 0
    # 定义复杂表达式 z
    z = 2*(-3*tan(19*pi/90) + sqrt(3))*cos(11*pi/90)*cos(19*pi/90) - \
        sqrt(3)*(-3 + 4*cos(19*pi/90)**2)
    # 断言 z 是否为非正数（实际上是零，不应该挂起）
    assert z.is_positive is not True  # it's zero and it shouldn't hang
    # 定义复杂表达式 z
    z = S('''9*(3*sqrt(93) + 29)**(2/3)*((3*sqrt(93) +
        29)**(1/3)*(-2**(2/3)*(3*sqrt(93) + 29)**(1/3) - 2) - 2*2**(1/3))**3 +
        72*(3*sqrt(93) + 29)**(2/3)*(81*sqrt(93) + 783) + (162*sqrt(93) +
        1566)*((3*sqrt(93) + 29)**(1/3)*(-2**(2/3)*(3*sqrt(93) + 29)**(1/3) -
        2) - 2*2**(1/3))**2''')
    # 断言 z 是否为非正数（实际上是零，单一 _mexpand 不足够）
    assert z.is_positive is False  # it's 0 (and a single _mexpand isn't enough)


def test_Add_is_zero():
    # 定义符号变量 x 和 y，都为零
    x, y = symbols('x y', zero=True)
    # 断言 x + y 是否为零
    assert (x + y).is_zero

    # Issue 15873
    # 定义复杂表达式 e
    e = -2*I + (1 + I)**2
    # 断言 e 是否为零
    assert e.is_zero is None


def test_issue_14392():
    # 断言 sin(zoo)**2 的实部和虚部都是 NaN
    assert (sin(zoo)**2).as_real_imag() == (nan, nan)


def test_divmod():
    # 断言 x 除以 y 的商和余数等于 divmod(x, y) 的结果
    assert divmod(x, y) == (x//y, x % y)
    # 断言 x 除以 3 的商和余数等于 divmod(x, 3) 的结果
    assert divmod(x, 3) == (x//3, x % 3)
    # 断言 3 除以 x 的商和余数等于 divmod(3, x) 的结果
    assert divmod(3, x) == (3//x, 3 % x)


def test__neg__():
    # 断言 -(x*y) 等于 -x*y
    assert -(x*y) == -x*y
    # 断言 -(-x*y) 等于 x*y
    assert -(-x*y) == x*y
    # 断言 -(1.*x) 等于 -1.*x
    assert -(1.*x) == -1.*x
    # 断言 -(-1.*x) 等于 1.*x
    assert -(-1.*x) == 1.*x
    # 断言 -(2.*x) 等于 -2.*x
    assert -(2.*x) == -2.*x
    # 断言 -(-2.*x) 等于 2.*x
    assert -(-2.*x) == 2.*x
    # 使用 distribute(False) 上下文
    with distribute(False):
        # 定义等式 eq = -(x + y)
        eq = -(x + y)
        # 断言 eq 是否是乘法，并且其参数为 (-1, x + y)
        assert eq.is_Mul and eq.args == (-1, x + y)
    # 使用 evaluate(False) 上下文
    with evaluate(False):
        # 定义等式 eq = -(x + y)
        eq = -(x + y)
        # 断言 eq 是否是乘法，并且其参数为 (-1, x + y)
        assert eq.is_Mul and eq.args == (-1, x + y)


def test_issue_18507():
    # 断言 Mul(zoo, zoo, 0) 的结果是 NaN
    assert Mul(zoo, zoo, 0) is nan


def test_issue_17130():
    # 定义复杂等式 e
    e = Add(b, -b, I, -I, evaluate=False)
    # 断言 e 是否为零（理想情况下应为 True）
    assert e.is_zero is None  # ideally this would be True


def test_issue_21034():
    # 定义复杂等式 e
    e = -I*log((re(asin(5)) + I*im(asin(5)))/sqrt(re(asin(5))**2 + im(asin(5))**2))/pi
    # 断言对 e 进行两位小数的四舍五入
    assert e.round(2)
``
# 测试问题编号 22021 的函数
def test_issue_22021():
    # 导入必要的模块和类
    from sympy.calculus.accumulationbounds import AccumBounds
    # Mul 中的特殊情况对象
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
    
    # 创建张量索引类型 L
    L = TensorIndexType("L")
    # 创建张量索引 i，属于类型 L
    i = tensor_indices("i", L)
    # 创建张量头 A 和 B，类型为 L
    A, B = tensor_heads("A B", [L])
    
    # 创建表达式 e = A(i) + B(i)
    e = A(i) + B(i)
    # 断言表达式 -e 等于 -1*e
    assert -e == -1*e
    
    # 创建符号表达式 zoo + x
    e = zoo + x
    # 断言表达式 -e 等于 -1*e
    assert -e == -1*e
    
    # 创建积累边界 AccumBounds(1, 2)
    a = AccumBounds(1, 2)
    # 创建表达式 e = a + x
    e = a + x
    # 断言表达式 -e 等于 -1*e
    assert -e == -1*e
    
    # 对 (zoo, a, x) 的全排列进行迭代
    for args in permutations((zoo, a, x)):
        # 创建表达式 e = Add(*args, evaluate=False)
        e = Add(*args, evaluate=False)
        # 断言表达式 -e 等于 -1*e
        assert -e == -1*e
    
    # 断言表达式 2*Add(1, x, x, evaluate=False) 等于 4*x + 2
    assert 2*Add(1, x, x, evaluate=False) == 4*x + 2


# 测试问题编号 22244 的函数
def test_issue_22244():
    # 断言表达式 -(zoo*x) 等于 zoo*x
    assert -(zoo*x) == zoo*x


# 测试问题编号 22453 的函数
def test_issue_22453():
    # 导入 cartes 函数
    from sympy.utilities.iterables import cartes
    
    # 创建符号 e，被定义为 extended_positive
    e = Symbol('e', extended_positive=True)
    
    # 对于 cartes 函数生成的每对 (a, b)
    for a, b in cartes(*[[oo, -oo, 3]]*2):
        # 如果 a 和 b 均为 3，则跳过当前循环
        if a == b == 3:
            continue
        # 创建复数 i = a + I*b
        i = a + I*b
        # 断言 i**(1 + e) 为 S.ComplexInfinity
        assert i**(1 + e) is S.ComplexInfinity
        # 断言 i**-e 为 S.Zero
        assert i**-e is S.Zero
        # 断言 unchanged(Pow, i, e)
        assert unchanged(Pow, i, e)
    
    # 断言 1/(oo + I*oo) 为 S.Zero
    assert 1/(oo + I*oo) is S.Zero
    
    # 创建虚拟数 r 和 i
    r, i = [Dummy(infinite=True, extended_real=True) for _ in range(2)]
    # 断言 1/(r + I*i) 为 S.Zero
    assert 1/(r + I*i) is S.Zero
    # 断言 1/(3 + I*i) 为 S.Zero
    assert 1/(3 + I*i) is S.Zero
    # 断言 1/(r + I*3) 为 S.Zero
    assert 1/(r + I*3) is S.Zero


# 测试问题编号 22613 的函数
def test_issue_22613():
    # 断言 (0**(x - 2)).as_content_primitive() 等于 (1, 0**(x - 2))
    assert (0**(x - 2)).as_content_primitive() == (1, 0**(x - 2))
    # 断言 (0**(x + 2)).as_content_primitive() 等于 (1, 0**(x + 2))
    assert (0**(x + 2)).as_content_primitive() == (1, 0**(x + 2))


# 测试问题编号 25176 的函数
def test_issue_25176():
    # 断言 sqrt(-4*3**(S(3)/4)*I/3) 等于 2*3**(S(7)/8)*sqrt(-I)/3
    assert sqrt(-4*3**(S(3)/4)*I/3) == 2*3**(S(7)/8)*sqrt(-I)/3
```