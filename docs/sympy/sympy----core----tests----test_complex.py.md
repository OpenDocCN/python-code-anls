# `D:\src\scipysrc\sympy\sympy\core\tests\test_complex.py`

```
# 导入 sympy 库中的一些核心函数和符号
from sympy.core.function import expand_complex
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)

# 定义测试复数计算的函数
def test_complex():
    # 定义两个实数符号 a 和 b
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    # 创建一个复数表达式
    e = (a + I*b)*(a - I*b)
    # 验证复数表达式的展开结果
    assert e.expand() == a**2 + b**2
    # 验证平方根函数对虚数单位 I 的计算
    assert sqrt(I) == Pow(I, S.Half)


# 定义测试共轭函数的函数
def test_conjugate():
    # 定义两个实数符号 a 和 b，以及两个虚数符号 c 和 d
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    c = Symbol("c", imaginary=True)
    d = Symbol("d", imaginary=True)
    # 创建一个复合表达式 z
    x = Symbol('x')
    z = a + I*b + c + I*d
    # 计算 z 的共轭
    zc = a - I*b - c + I*d
    assert conjugate(z) == zc
    # 验证 exp 函数的共轭性质
    assert conjugate(exp(z)) == exp(zc)
    # 验证 exp(I*x) 的共轭
    assert conjugate(exp(I*x)) == exp(-I*conjugate(x))
    # 验证 z 的五次幂的共轭
    assert conjugate(z**5) == zc**5
    # 验证绝对值函数的共轭性质
    assert conjugate(abs(x)) == abs(x)
    # 验证符号函数的共轭性质
    assert conjugate(sign(z)) == sign(zc)
    # 验证三角函数的共轭性质
    assert conjugate(sin(z)) == sin(zc)
    assert conjugate(cos(z)) == cos(zc)
    assert conjugate(tan(z)) == tan(zc)
    assert conjugate(cot(z)) == cot(zc)
    # 验证双曲函数的共轭性质
    assert conjugate(sinh(z)) == sinh(zc)
    assert conjugate(cosh(z)) == cosh(zc)
    assert conjugate(tanh(z)) == tanh(zc)
    assert conjugate(coth(z)) == coth(zc)


# 定义测试绝对值函数的函数
def test_abs1():
    # 定义两个实数符号 a 和 b
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    # 验证绝对值函数的性质
    assert abs(a) == Abs(a)
    assert abs(-a) == abs(a)
    assert abs(a + I*b) == sqrt(a**2 + b**2)


# 定义测试绝对值函数的函数（复数情况）
def test_abs2():
    # 定义两个非实数符号 a 和 b
    a = Symbol("a", real=False)
    b = Symbol("b", real=False)
    # 验证绝对值函数在非实数情况下的性质
    assert abs(a) != a
    assert abs(-a) != a
    assert abs(a + I*b) != sqrt(a**2 + b**2)


# 定义测试复数计算的函数
def test_evalc():
    # 定义实数符号 x 和 y，以及一个一般符号 z
    x = Symbol("x", real=True)
    y = Symbol("y", real=True)
    z = Symbol("z")
    # 验证复数计算中的展开和复数单位的性质
    assert ((x + I*y)**2).expand(complex=True) == x**2 + 2*I*x*y - y**2
    # 验证 expand_complex 函数的作用
    assert expand_complex(z**(2*I)) == (re((re(z) + I*im(z))**(2*I)) +
        I*im((re(z) + I*im(z))**(2*I)))
    assert expand_complex(z**(2*I), deep=False) == I*im(z**(2*I)) + re(z**(2*I))

    # 验证指数函数和三角函数的展开
    assert exp(I*x) != cos(x) + I*sin(x)
    assert exp(I*x).expand(complex=True) == cos(x) + I*sin(x)
    assert exp(I*x + y).expand(complex=True) == exp(y)*cos(x) + I*sin(x)*exp(y)

    # 验证正弦函数和双曲函数的展开
    assert sin(I*x).expand(complex=True) == I * sinh(x)
    assert sin(x + I*y).expand(complex=True) == sin(x)*cosh(y) + \
        I * sinh(y) * cos(x)

    # 验证余弦函数和双曲函数的展开
    assert cos(I*x).expand(complex=True) == cosh(x)
    assert cos(x + I*y).expand(complex=True) == cos(x)*cosh(y) - \
        I * sinh(y) * sin(x)

    # 验证正切函数和双曲函数的展开
    assert tan(I*x).expand(complex=True) == tanh(x) * I
    assert tan(x + I*y).expand(complex=True) == (
        sin(2*x)/(cos(2*x) + cosh(2*y)) +
        I*sinh(2*y)/(cos(2*x) + cosh(2*y)))


这样每行代码都被详细注释，说明了其作用和意图。
    # 验证双曲正弦函数的展开结果是否符合预期
    assert sinh(I*x).expand(complex=True) == I * sin(x)
    
    # 验证带复数参数的双曲正弦函数的展开结果是否符合预期
    assert sinh(x + I*y).expand(complex=True) == sinh(x)*cos(y) + \
        I * sin(y) * cosh(x)
    
    # 验证双曲余弦函数的展开结果是否符合预期
    assert cosh(I*x).expand(complex=True) == cos(x)
    
    # 验证带复数参数的双曲余弦函数的展开结果是否符合预期
    assert cosh(x + I*y).expand(complex=True) == cosh(x)*cos(y) + \
        I * sin(y) * sinh(x)
    
    # 验证双曲正切函数的展开结果是否符合预期
    assert tanh(I*x).expand(complex=True) == tan(x) * I
    
    # 验证带复数参数的双曲正切函数的展开结果是否符合预期
    assert tanh(x + I*y).expand(complex=True) == (
        (sinh(x)*cosh(x) + I*cos(y)*sin(y)) /
        (sinh(x)**2 + cos(y)**2)).expand()
def test_pythoncomplex():
    # 创建符号变量 x
    x = Symbol("x")
    # 断言不等式成立
    assert 4j*x != 4*x*I
    # 断言相等成立，注意浮点数乘法与复数乘法的区别
    assert 4j*x == 4.0*x*I
    # 断言不等式成立，同样注意浮点数乘法与复数乘法的区别
    assert 4.1j*x != 4*x*I


def test_rootcomplex():
    # 使用 Rational 别名 R
    R = Rational
    # 断言展开复数根为特定表达式
    assert ((+1 + I)**R(1, 2)).expand(complex=True) == 2**R(1, 4)*cos(pi/8) + 2**R(1, 4)*sin(pi/8)*I
    # 断言展开复数根为特定表达式
    assert ((-1 - I)**R(1, 2)).expand(complex=True) == 2**R(1, 4)*cos(3*pi/8) - 2**R(1, 4)*sin(3*pi/8)*I
    # 断言虚数的平方根展开为实部和虚部
    assert (sqrt(-10)*I).as_real_imag() == (-sqrt(10), 0)


def test_expand_inverse():
    # 断言复数的倒数展开结果
    assert (1/(1 + I)).expand(complex=True) == (1 - I)/2
    # 断言复数的负幂展开结果
    assert ((1 + 2*I)**(-2)).expand(complex=True) == (-3 - 4*I)/25
    # 断言复数的负八次幂展开结果
    assert ((1 + I)**(-8)).expand(complex=True) == Rational(1, 16)


def test_expand_complex():
    # 断言复数的十次幂展开结果
    assert ((2 + 3*I)**10).expand(complex=True) == -341525 - 145668*I
    # 以下两个测试确保 SymPy 使用高效算法计算复数的幂次方
    # 它们应该在大约0.01秒内执行完成。
    assert ((2 + 3*I)**1000).expand(complex=True) == \
        -81079464736246615951519029367296227340216902563389546989376269312984127074385455204551402940331021387412262494620336565547972162814110386834027871072723273110439771695255662375718498785908345629702081336606863762777939617745464755635193139022811989314881997210583159045854968310911252660312523907616129080027594310008539817935736331124833163907518549408018652090650537035647520296539436440394920287688149200763245475036722326561143851304795139005599209239350981457301460233967137708519975586996623552182807311159141501424576682074392689622074945519232029999 + \
        46938745946789557590804551905243206242164799136976022474337918748798900569942573265747576032611189047943842446167719177749107138603040963603119861476016947257034472364028585381714774667326478071264878108114128915685688115488744955550920239128462489496563930809677159214598114273887061533057125164518549173898349061972857446844052995037423459472376202251620778517659247970283904820245958198842631651569984310559418135975795868314764489884749573052997832686979294085577689571149679540256349988338406458116270429842222666345146926395233040564229555893248370000*I
    # 断言复数的指数展开结果
    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    assert exp(a*(2 + I*b)).expand(complex=True) == \
        I*exp(2*a)*sin(a*b) + exp(2*a)*cos(a*b)


def test_expand():
    # 展开表达式并进行断言
    f = (16 - 2*sqrt(29))**2
    assert f.expand() == 372 - 64*sqrt(29)
    # 展开表达式并进行断言
    f = (Integer(1)/2 + I/2)**10
    assert f.expand() == I/32
    # 展开表达式并进行断言
    f = (Integer(1)/2 + I)**10
    assert f.expand() == Integer(237)/1024 - 779*I/256


def test_re_im1652():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言实部函数的性质
    assert re(x) == re(conjugate(x))
    # 断言虚部函数的性质
    assert im(x) == - im(conjugate(x))
    # 断言实部和虚部的乘积为零
    assert im(x)*re(conjugate(x)) + im(conjugate(x)) * re(x) == 0


def test_issue_5084():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言复数除以复数的实部和虚部
    assert ((x + x*I)/(1 + I)).as_real_imag() == (re((x + I*x)/(1 + I)
            ), im((x + I*x)/(1 + I)))


def test_issue_5236():
    # 留待后续添加测试内容
    pass
    # 断言，验证一个条件是否为真；此处验证复数计算结果的实部和虚部是否等于指定的值
    assert (cos(1 + I)**3).as_real_imag() == (
        -3*sin(1)**2*sinh(1)**2*cos(1)*cosh(1) + cos(1)**3*cosh(1)**3, 
        -3*cos(1)**2*cosh(1)**2*sin(1)*sinh(1) + sin(1)**3*sinh(1)**3
    )
# 定义测试函数 test_real_imag，用于测试复数和实部/虚部相关的功能
def test_real_imag():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x, y, z')
    # 定义非交换符号变量 X, Y, Z
    X, Y, Z = symbols('X, Y, Z', commutative=False)
    # 定义一个实数符号变量 a
    a = Symbol('a', real=True)
    
    # 断言测试复数乘法的实部和虚部分解
    assert (2*a*x).as_real_imag() == (2*a*re(x), 2*a*im(x))

    # issue 5395:
    # 断言复数的模的平方和虚部为零
    assert (x*x.conjugate()).as_real_imag() == (Abs(x)**2, 0)
    # 断言复数的虚部为零
    assert im(x*x.conjugate()) == 0
    # 断言复数乘积的虚部等于各部分虚部的乘积
    assert im(x*y.conjugate()*z*y) == im(x*z)*Abs(y)**2
    # 断言复数乘积的虚部等于各部分虚部的乘积
    assert im(x*y.conjugate()*x*y) == im(x**2)*Abs(y)**2
    # 断言复数乘积的虚部等于各部分虚部的乘积
    assert im(Z*y.conjugate()*X*y) == im(Z*X)*Abs(y)**2
    # 断言复数的乘积的虚部等于其共轭转置的虚部，禁止求值
    assert im(X*X.conjugate()) == im(X*X.conjugate(), evaluate=False)
    # 断言正弦函数乘积的实部和虚部
    assert (sin(x)*sin(x).conjugate()).as_real_imag() == \
        (Abs(sin(x))**2, 0)

    # issue 6573:
    # 断言平方的实部和虚部分解
    assert (x**2).as_real_imag() == (re(x)**2 - im(x)**2, 2*re(x)*im(x))

    # issue 6428:
    # 定义一个实数符号变量 r 和一个虚数符号变量 i
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)
    # 断言虚数乘积的实部和虚部分解
    assert (i*r*x).as_real_imag() == (I*i*r*im(x), -I*i*r*re(x))
    # 断言虚数乘积的实部和虚部分解
    assert (i*r*x*(y + 2)).as_real_imag() == (
        I*i*r*(re(y) + 2)*im(x) + I*i*r*re(x)*im(y),
        -I*i*r*(re(y) + 2)*re(x) + I*i*r*im(x)*im(y))

    # issue 7106:
    # 断言复数的除法的实部和虚部分解
    assert ((1 + I)/(1 - I)).as_real_imag() == (0, 1)
    # 断言复数乘积的实部和虚部分解
    assert ((1 + 2*I)*(1 + 3*I)).as_real_imag() == (-5, 5)


# 定义测试函数 test_pow_issue_1724，用于测试复数的幂问题
def test_pow_issue_1724():
    # 定义一个复数表达式 e
    e = ((S.NegativeOne)**(S.One/3))
    # 断言复数的共轭和数值计算后的结果相等
    assert e.conjugate().n() == e.n().conjugate()
    # 定义一个复数表达式 e
    e = S('-2/3 - (-29/54 + sqrt(93)/18)**(1/3) - 1/(9*(-29/54 + sqrt(93)/18)**(1/3))')
    # 断言复数的共轭和数值计算后的结果相等
    assert e.conjugate().n() == e.n().conjugate()
    # 定义一个复数表达式 e
    e = 2**I
    # 断言复数的共轭和数值计算后的结果相等
    assert e.conjugate().n() == e.n().conjugate()


# 定义测试函数 test_issue_5429，用于测试平方根的共轭问题
def test_issue_5429():
    # 断言复数平方根的共轭不等于其本身
    assert sqrt(I).conjugate() != sqrt(I)


# 定义测试函数 test_issue_4124，用于测试无穷大与虚数乘法的展开
def test_issue_4124():
    # 导入 oo（无穷大）常数
    from sympy.core.numbers import oo
    # 断言无穷大乘虚数等于无穷大乘虚数
    assert expand_complex(I*oo) == oo*I


# 定义测试函数 test_issue_11518，用于测试共轭和绝对值的关系
def test_issue_11518():
    # 定义两个实数符号变量 x 和 y
    x = Symbol("x", real=True)
    y = Symbol("y", real=True)
    # 定义半径 r 为 x 和 y 的平方和的平方根
    r = sqrt(x**2 + y**2)
    # 断言半径的共轭等于半径本身
    assert conjugate(r) == r
    # 定义复数 s 为 x + i*y 的绝对值
    s = abs(x + I * y)
    # 断言复数 s 的共轭等于半径 r
    assert conjugate(s) == r
```