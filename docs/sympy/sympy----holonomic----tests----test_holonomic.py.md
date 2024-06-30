# `D:\src\scipysrc\sympy\sympy\holonomic\tests\test_holonomic.py`

```
# 导入 SymPy 中的不同模块和函数

from sympy.holonomic import (DifferentialOperator, HolonomicFunction,
                             DifferentialOperators, from_hyper,
                             from_meijerg, expr_to_holonomic)
from sympy.holonomic.recurrence import RecurrenceOperators, HolonomicSequence
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import besselj
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (Ci, Si, erf, erfc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.simplify.hyperexpand import hyperexpand
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR

# 定义一个测试函数，用于测试 DifferentialOperator 类的功能
def test_DifferentialOperator():
    # 定义符号变量 x
    x = symbols('x')
    # 使用有理数域 QQ 上的多项式环和 'Dx' 作为导数运算符创建 DifferentialOperators 对象
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 断言导数运算符 Dx 等于 R 的导数运算符
    assert Dx == R.derivative_operator
    # 断言 Dx 等于 DifferentialOperator([R.base.zero, R.base.one], R)
    assert Dx == DifferentialOperator([R.base.zero, R.base.one], R)
    # 断言 x * Dx + x**2 * Dx**2 等于 DifferentialOperator([0, x, x**2], R)
    assert x * Dx + x**2 * Dx**2 == DifferentialOperator([0, x, x**2], R)
    # 断言 (x**2 + 1) + Dx + x * Dx**5 等于 DifferentialOperator([x**2 + 1, 1, 0, 0, 0, x], R)
    assert (x**2 + 1) + Dx + x * \
        Dx**5 == DifferentialOperator([x**2 + 1, 1, 0, 0, 0, x], R)
    # 断言 (x * Dx + x**2 + 1 - Dx * (x**3 + x))**3 等于 (-48 * x**6) + (-57 * x**7) * Dx + (-15 * x**8) * Dx**2 + (-x**9) * Dx**3
    assert (x * Dx + x**2 + 1 - Dx * (x**3 + x))**3 == (-48 * x**6) + \
        (-57 * x**7) * Dx + (-15 * x**8) * Dx**2 + (-x**9) * Dx**3
    # 定义 p 和 q 作为 HolonomicFunction 对象
    p = (x * Dx**2 + (x**2 + 3) * Dx**5) * (Dx + x**2)
    q = (2 * x) + (4 * x**2) * Dx + (x**3) * Dx**2 + \
        (20 * x**2 + x + 60) * Dx**3 + (10 * x**3 + 30 * x) * Dx**4 + \
        (x**4 + 3 * x**2) * Dx**5 + (x**2 + 3) * Dx**6
    # 断言 p 等于 q
    assert p == q

# 定义一个测试函数，用于测试 HolonomicFunction 类中的加法功能
def test_HolonomicFunction_addition():
    # 定义符号变量 x
    x = symbols('x')
    # 使用整数环 ZZ 上的多项式环和 'Dx' 作为导数运算符创建 DifferentialOperators 对象
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    # 创建 HolonomicFunction 对象 p 和 q
    p = HolonomicFunction(Dx**2 * x, x)
    q = HolonomicFunction((2) * Dx + (x) * Dx**2, x)
    # 断言 p 等于 q
    assert p == q
    # 创建 HolonomicFunction 对象 p 和 q，并进行加法操作
    p = HolonomicFunction(x * Dx + 1, x)
    q = HolonomicFunction(Dx + 1, x)
    r = HolonomicFunction((x - 2) + (x**2 - 2) * Dx + (x**2 - x) * Dx**2, x)
    # 断言 p + q 等于 r
    assert p + q == r
    # 创建 HolonomicFunction 对象 p 和 q，并进行加法操作
    p = HolonomicFunction(x * Dx + Dx**2 * (x**2 + 2), x)
    q = HolonomicFunction(Dx - 3, x)
    r = HolonomicFunction((-54 * x**2 - 126 * x - 150) + (-135 * x**3 - 252 * x**2 - 270 * x + 140) * Dx +\
                 (-27 * x**4 - 24 * x**2 + 14 * x - 150) * Dx**2 + \
                 (9 * x**4 + 15 * x**3 + 38 * x**2 + 30 * x +40) * Dx**3, x)
    # 断言 p + q 等于 r
    assert p + q == r
    # 创建 HolonomicFunction 对象 p 和 q，并进行加法操作
    p = HolonomicFunction(Dx**5 - 1, x)
    q = HolonomicFunction(x**3 + Dx, x)
    # 创建一个关于 x 的全纯函数 r，使用 HolonomicFunction 类进行定义
    r = HolonomicFunction((-x**18 + 45*x**14 - 525*x**10 + 1575*x**6 - x**3 - 630*x**2) + \
        (-x**15 + 30*x**11 - 195*x**7 + 210*x**3 - 1)*Dx + (x**18 - 45*x**14 + 525*x**10 - \
        1575*x**6 + x**3 + 630*x**2)*Dx**5 + (x**15 - 30*x**11 + 195*x**7 - 210*x**3 + \
        1)*Dx**6, x)
    # 断言表达式 p+q 应与 r 相等
    assert p+q == r
    
    # 定义多项式 p 和 q
    p = x**2 + 3*x + 8
    q = x**3 - 7*x + 5
    # 计算 p 和 q 的导数，并生成对应的全纯函数
    p = p*Dx - p.diff()
    q = q*Dx - q.diff()
    # 使用 HolonomicFunction 类创建函数 r 和 s，并将它们相加
    r = HolonomicFunction(p, x) + HolonomicFunction(q, x)
    s = HolonomicFunction((6*x**2 + 18*x + 14) + (-4*x**3 - 18*x**2 - 62*x + 10)*Dx +\
        (x**4 + 6*x**3 + 31*x**2 - 10*x - 71)*Dx**2, x)
    # 断言 r 应与 s 相等
    assert r == s
# 定义测试函数，用于测试HolonomicFunction类的乘法功能
def test_HolonomicFunction_multiplication():
    # 符号变量x
    x = symbols('x')
    # 创建整数环和微分操作符
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    # 创建HolonomicFunction对象p，给定微分方程和自变量x
    p = HolonomicFunction(Dx+x+x*Dx**2, x)
    # 创建HolonomicFunction对象q，给定微分方程和自变量x
    q = HolonomicFunction(x*Dx+Dx*x+Dx**2, x)
    # 创建HolonomicFunction对象r，给定复杂的微分方程和自变量x
    r = HolonomicFunction((8*x**6 + 4*x**4 + 6*x**2 + 3) + (24*x**5 - 4*x**3 + 24*x)*Dx + \
        (8*x**6 + 20*x**4 + 12*x**2 + 2)*Dx**2 + (8*x**5 + 4*x**3 + 4*x)*Dx**3 + \
        (2*x**4 + x**2)*Dx**4, x)
    # 断言p乘以q等于r
    assert p*q == r
    
    # 更换p和q，继续进行测试
    p = HolonomicFunction(Dx**2+1, x)
    q = HolonomicFunction(Dx-1, x)
    r = HolonomicFunction((2) + (-2)*Dx + (1)*Dx**2, x)
    assert p*q == r
    
    # 更换p和q，继续进行测试
    p = HolonomicFunction(Dx**2+1+x+Dx, x)
    q = HolonomicFunction((Dx*x-1)**2, x)
    r = HolonomicFunction((4*x**7 + 11*x**6 + 16*x**5 + 4*x**4 - 6*x**3 - 7*x**2 - 8*x - 2) + \
        (8*x**6 + 26*x**5 + 24*x**4 - 3*x**3 - 11*x**2 - 6*x - 2)*Dx + \
        (8*x**6 + 18*x**5 + 15*x**4 - 3*x**3 - 6*x**2 - 6*x - 2)*Dx**2 + (8*x**5 + \
            10*x**4 + 6*x**3 - 2*x**2 - 4*x)*Dx**3 + (4*x**5 + 3*x**4 - x**2)*Dx**4, x)
    assert p*q == r
    
    # 更换p和q，继续进行测试
    p = HolonomicFunction(x*Dx**2-1, x)
    q = HolonomicFunction(Dx*x-x, x)
    r = HolonomicFunction((x - 3) + (-2*x + 2)*Dx + (x)*Dx**2, x)
    assert p*q == r


# 定义测试函数，用于测试HolonomicFunction类的乘方功能
def test_HolonomicFunction_power():
    # 符号变量x
    x = symbols('x')
    # 创建整数环和微分操作符
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    # 创建HolonomicFunction对象p，给定微分方程和自变量x
    p = HolonomicFunction(Dx+x+x*Dx**2, x)
    # 创建HolonomicFunction对象a，给定微分方程和自变量x
    a = HolonomicFunction(Dx, x)
    # 循环测试p的乘方是否等于a
    for n in range(10):
        assert a == p**n
        a *= p


# 定义测试函数，用于测试HolonomicFunction类的加法功能及初始条件
def test_addition_initial_condition():
    # 符号变量x
    x = symbols('x')
    # 创建有理数环和微分操作符
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 创建HolonomicFunction对象p，给定微分方程、自变量x、初始条件和系数列表
    p = HolonomicFunction(Dx-1, x, 0, [3])
    # 创建HolonomicFunction对象q，给定微分方程、自变量x、初始条件和系数列表
    q = HolonomicFunction(Dx**2+1, x, 0, [1, 0])
    # 创建HolonomicFunction对象r，给定复杂的微分方程、自变量x、初始条件和系数列表
    r = HolonomicFunction(-1 + Dx - Dx**2 + Dx**3, x, 0, [4, 3, 2])
    # 断言p加q等于r
    assert p + q == r
    
    # 更换p和q，继续进行测试
    p = HolonomicFunction(Dx - x + Dx**2, x, 0, [1, 2])
    q = HolonomicFunction(Dx**2 + x, x, 0, [1, 0])
    r = HolonomicFunction((-x**4 - x**3/4 - x**2 + Rational(1, 4)) + (x**3 + x**2/4 + x*Rational(3, 4) + 1)*Dx + \
        (x*Rational(-3, 2) + Rational(7, 4))*Dx**2 + (x**2 - x*Rational(7, 4) + Rational(1, 4))*Dx**3 + (x**2 + x/4 + S.Half)*Dx**4, x, 0, [2, 2, -2, 2])
    assert p + q == r
    
    # 更换p和q，继续进行测试
    p = HolonomicFunction(Dx**2 + 4*x*Dx + x**2, x, 0, [3, 4])
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 1])
    r = HolonomicFunction((x**6 + 2*x**4 - 5*x**2 - 6) + (4*x**5 + 36*x**3 - 32*x)*Dx + \
         (x**6 + 3*x**4 + 5*x**2 - 9)*Dx**2 + (4*x**5 + 36*x**3 - 32*x)*Dx**3 + (x**4 + \
            10*x**2 - 3)*Dx**4, x, 0, [4, 5, -1, -17])
    assert p + q == r
    
    # 更换p和q，继续进行测试
    q = HolonomicFunction(Dx**3 + x, x, 2, [3, 0, 1])
    p = HolonomicFunction(Dx - 1, x, 2, [1])
    r = HolonomicFunction((-x**2 - x + 1) + (x**2 + x)*Dx + (-x - 2)*Dx**3 + \
        (x + 1)*Dx**4, x, 2, [4, 1, 2, -5 ])
    assert p + q == r
    
    # 创建HolonomicFunction对象p和q，给定sin(x)和1/x的表达式
    p = expr_to_holonomic(sin(x))
    q = expr_to_holonomic(1/x, x0=1)
    # 使用给定的表达式构造一个关于 x 的高阶全纯函数（HolonomicFunction）
    r = HolonomicFunction((x**2 + 6) + (x**3 + 2*x)*Dx + (x**2 + 6)*Dx**2 + (x**3 + 2*x)*Dx**3, \
        x, 1, [sin(1) + 1, -1 + cos(1), -sin(1) + 2])

    # 断言 p + q 的结果等于 r
    assert p + q == r

    # 定义符号 C_1
    C_1 = symbols('C_1')

    # 将 sqrt(x) 转换为关于 x 的全纯函数 p
    p = expr_to_holonomic(sqrt(x))

    # 将 sqrt(x**2 - x) 转换为关于 x 的全纯函数 q
    q = expr_to_holonomic(sqrt(x**2-x))

    # 计算 p + q 的结果，并将符号 C_1 替换为 -I/2，然后展开表达式
    r = (p + q).to_expr().subs(C_1, -I/2).expand()

    # 断言 r 的结果等于给定的表达式
    assert r == I*sqrt(x)*sqrt(-x + 1) + sqrt(x)
# 定义一个测试函数，用于测试乘法初值条件
def test_multiplication_initial_condition():
    # 定义符号变量 x
    x = symbols('x')
    # 创建有理函数环 QQ 的多项式环
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 创建 HolonomicFunction 对象 p，表示微分方程 Dx**2 + x*Dx - 1，初始条件为 x=0 时 [3, 1]
    p = HolonomicFunction(Dx**2 + x*Dx - 1, x, 0, [3, 1])
    # 创建 HolonomicFunction 对象 q，表示微分方程 Dx**2 + 1，初始条件为 x=0 时 [1, 1]
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 1])
    # 创建 HolonomicFunction 对象 r，表示微分方程 (x**4 + 14*x**2 + 60) + 4*x*Dx + (x**4 + 9*x**2 + 20)*Dx**2 + \
    #     (2*x**3 + 18*x)*Dx**3 + (x**2 + 10)*Dx**4，初始条件为 x=0 时 [3, 4, 2, 3]
    r = HolonomicFunction((x**4 + 14*x**2 + 60) + 4*x*Dx + (x**4 + 9*x**2 + 20)*Dx**2 + \
        (2*x**3 + 18*x)*Dx**3 + (x**2 + 10)*Dx**4, x, 0, [3, 4, 2, 3])
    # 断言 p 乘以 q 等于 r
    assert p * q == r
    
    # 更新 p，表示微分方程 Dx**2 + x，初始条件为 x=0 时 [1, 0]
    p = HolonomicFunction(Dx**2 + x, x, 0, [1, 0])
    # 更新 q，表示微分方程 Dx**3 - x**2，初始条件为 x=0 时 [3, 3, 3]
    q = HolonomicFunction(Dx**3 - x**2, x, 0, [3, 3, 3])
    # 更新 r，表示复杂的微分方程，初始条件为 x=0 时 [3, 3, 3, -3, -12, -24]
    r = HolonomicFunction((x**8 - 37*x**7/27 - 10*x**6/27 - 164*x**5/9 - 184*x**4/9 + \
        160*x**3/27 + 404*x**2/9 + 8*x + Rational(40, 3)) + (6*x**7 - 128*x**6/9 - 98*x**5/9 - 28*x**4/9 + \
        8*x**3/9 + 28*x**2 + x*Rational(40, 9) - 40)*Dx + (3*x**6 - 82*x**5/9 + 76*x**4/9 + 4*x**3/3 + \
        220*x**2/9 - x*Rational(80, 3))*Dx**2 + (-2*x**6 + 128*x**5/27 - 2*x**4/3 -80*x**2/9 + Rational(200, 9))*Dx**3 + \
        (3*x**5 - 64*x**4/9 - 28*x**3/9 + 6*x**2 - x*Rational(20, 9) - Rational(20, 3))*Dx**4 + (-4*x**3 + 64*x**2/9 + \
            x*Rational(8, 3))*Dx**5 + (x**4 - 64*x**3/27 - 4*x**2/3 + Rational(20, 9))*Dx**6, x, 0, [3, 3, 3, -3, -12, -24])
    # 断言 p 乘以 q 等于 r
    assert p * q == r
    
    # 更新 p，表示微分方程 Dx - 1，初始条件为 x=0 时 [2]
    p = HolonomicFunction(Dx - 1, x, 0, [2])
    # 更新 q，表示微分方程 Dx**2 + 1，初始条件为 x=0 时 [0, 1]
    q = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])
    # 更新 r，表示微分方程 2 - 2*Dx + Dx**2，初始条件为 x=0 时 [0, 2]
    r = HolonomicFunction(2 - 2*Dx + Dx**2, x, 0, [0, 2])
    # 断言 p 乘以 q 等于 r
    assert p * q == r
    
    # 更新 q，表示微分方程 x*Dx**2 + 1 + 2*Dx，初始条件为 x=0 时 [0, 1]
    q = HolonomicFunction(x*Dx**2 + 1 + 2*Dx, x, 0,[0, 1])
    # 更新 r，表示微分方程 (x - 1) + (-2*x + 2)*Dx + x*Dx**2，初始条件为 x=0 时 [0, 2]
    r = HolonomicFunction((x - 1) + (-2*x + 2)*Dx + x*Dx**2, x, 0, [0, 2])
    # 断言 p 乘以 q 等于 r
    assert p * q == r
    
    # 更新 p，表示微分方程 Dx**2 - 1，初始条件为 x=0 时 [1, 3]
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 3])
    # 更新 q，表示微分方程 Dx**3 + 1，初始条件为 x=0 时 [1, 2, 1]
    q = HolonomicFunction(Dx**3 + 1, x, 0, [1, 2, 1])
    # 更新 r，表示微分方程 6*Dx + 3*Dx**2 + 2*Dx**3 - 3*Dx**4 + Dx**6，初始条件为 x=0 时 [1, 5, 14, 17, 17, 2]
    r = HolonomicFunction(6*Dx + 3*Dx**2 + 2*Dx**3 - 3*Dx**4 + Dx**6, x, 0, [1, 5, 14, 17, 17, 2])
    # 断言 p 乘以 q 等于 r
    assert p * q == r
    
    # 创建 HolonomicFunction 对象 p，表示 sin(x) 的霍洛莫尼函数
    p = expr_to_holonomic(sin(x))
    # 创建 HolonomicFunction 对象 q，表示 1/x 的霍洛莫尼函数，x0=1
    q = expr_to_holonomic(1/x, x0=1)
    # 创建 HolonomicFunction 对象 r，表示 (x + 2*Dx + x*Dx**2)，初始条件为 x=1 时 [sin(1), -sin(1) + cos(1)]
    r = HolonomicFunction(x + 2*Dx + x*Dx**2, x, 1, [sin(1), -sin(1) + cos(1)])
    # 断言 p 乘以 q 等于 r
    assert p * q == r
    
    # 创建 HolonomicFunction 对象 p，表示 sqrt(x) 的霍洛莫尼函数
    p = expr_to_holonomic(sqrt(x))
    # 创建 HolonomicFunction 对象 q，表示 sqrt(x**2-x) 的霍洛莫尼函数
    q = expr_to_holonomic(sqrt(x**2-x))
    # 计算 p 乘以 q 并转换为表达式，赋值给 r
    r = (p * q).to_expr()
    # 断言 r 等于 I*x*sqrt(-x + 1)
    assert r == I*x*sqrt(-x + 1)


# 定义一个测试函数，用于测试霍洛莫尼函数的复合
def test_HolonomicFunction_composition():
    # 定义符号变量 x
    x = symbols('x')
    # 创建整数环 ZZ 的
    # 使用给定的参数创建一个关于 x 的HolonomicFunction对象 p，该对象表示为 (Dx**2+1) 组合于 (x - 2/(x**2 + 1))。
    p = HolonomicFunction(Dx**2+1, x).composition(x - 2/(x**2 + 1))
    
    # 使用给定的参数创建另一个关于 x 的HolonomicFunction对象 r，该对象表示为：
    # (x**12 + 6*x**10 + 12*x**9 + 15*x**8 + 48*x**7 + 68*x**6 + 72*x**5 + 111*x**4 + 112*x**3 + 
    # 54*x**2 + 12*x + 1) + (12*x**8 + 32*x**6 + 24*x**4 - 4)*Dx + 
    # (x**12 + 6*x**10 + 4*x**9 + 15*x**8 + 16*x**7 + 20*x**6 + 24*x**5 + 15*x**4 + 16*x**3 + 
    # 6*x**2 + 4*x + 1)*Dx**2。
    r = HolonomicFunction((x**12 + 6*x**10 + 12*x**9 + 15*x**8 + 48*x**7 + 68*x**6 + \
        72*x**5 + 111*x**4 + 112*x**3 + 54*x**2 + 12*x + 1) + (12*x**8 + 32*x**6 + \
        24*x**4 - 4)*Dx + (x**12 + 6*x**10 + 4*x**9 + 15*x**8 + 16*x**7 + 20*x**6 + 24*x**5+ \
        15*x**4 + 16*x**3 + 6*x**2 + 4*x + 1)*Dx**2, x)
    
    # 断言 p 与 r 相等，用于验证它们是否代表相同的HolonomicFunction。
    assert p == r
def test_from_hyper():
    # 符号 'x' 的定义
    x = symbols('x')
    # 创建有理函数环 R 和微分算子 Dx
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 定义超几何函数 p
    p = hyper([1, 1], [Rational(3, 2)], x**2/4)
    # 创建 HolonomicFunction 对象 q
    q = HolonomicFunction((4*x) + (5*x**2 - 8)*Dx + (x**3 - 4*x)*Dx**2, x, 1, [2*sqrt(3)*pi/9, -4*sqrt(3)*pi/27 + Rational(4, 3)])
    # 使用 from_hyper 函数从超几何函数 p 创建 HolonomicFunction 对象 r
    r = from_hyper(p)
    # 断言 r 和 q 相等
    assert r == q
    # 使用 from_hyper 函数从超几何函数创建 HolonomicFunction 对象 p
    p = from_hyper(hyper([1], [Rational(3, 2)], x**2/4))
    # 创建 HolonomicFunction 对象 q
    q = HolonomicFunction(-x + (-x**2/2 + 2)*Dx + x*Dx**2, x)
    # 断言 p.y0 的字符串表示与预期值 y0 相等
    # x0 被注释掉了，y0 是预期的字符串
    y0 = '[sqrt(pi)*exp(1/4)*erf(1/2), -sqrt(pi)*exp(1/4)*erf(1/2)/2 + 1]'
    assert sstr(p.y0) == y0
    # 断言 q 的消解子等于 p 的消解子
    assert q.annihilator == p.annihilator


def test_from_meijerg():
    # 符号 'x' 的定义
    x = symbols('x')
    # 创建有理函数环 R 和微分算子 Dx
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 使用 from_meijerg 函数从 MeijerG 函数创建 HolonomicFunction 对象 p
    p = from_meijerg(meijerg(([], [Rational(3, 2)]), ([S.Half], [S.Half, 1]), x))
    # 创建 HolonomicFunction 对象 q
    q = HolonomicFunction(x/2 - Rational(1, 4) + (-x**2 + x/4)*Dx + x**2*Dx**2 + x**3*Dx**3, x, 1, \
        [1/sqrt(pi), 1/(2*sqrt(pi)), -1/(4*sqrt(pi))])
    # 断言 p 和 q 相等
    assert p == q
    # 使用 from_meijerg 函数从 MeijerG 函数创建 HolonomicFunction 对象 p
    p = from_meijerg(meijerg(([], []), ([0], []), x))
    # 创建 HolonomicFunction 对象 q
    q = HolonomicFunction(1 + Dx, x, 0, [1])
    # 断言 p 和 q 相等
    assert p == q
    # 使用 from_meijerg 函数从 MeijerG 函数创建 HolonomicFunction 对象 p
    p = from_meijerg(meijerg(([1], []), ([S.Half], [0]), x))
    # 创建 HolonomicFunction 对象 q
    q = HolonomicFunction((x + S.Half)*Dx + x*Dx**2, x, 1, [sqrt(pi)*erf(1), exp(-1)])
    # 断言 p 和 q 相等
    assert p == q
    # 使用 from_meijerg 函数从 MeijerG 函数创建 HolonomicFunction 对象 p
    p = from_meijerg(meijerg(([0], [1]), ([0], []), 2*x**2))
    # 创建 HolonomicFunction 对象 q
    q = HolonomicFunction((3*x**2 - 1)*Dx + x**3*Dx**2, x, 1, [-exp(Rational(-1, 2)) + 1, -exp(Rational(-1, 2))])
    # 断言 p 和 q 相等
    assert p == q


def test_to_Sequence():
    # 符号 'x' 和整数 'n' 的定义
    x = symbols('x')
    n = symbols('n', integer=True)
    # 创建整数环 ZZ 和微分算子 Dx
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    # 创建整数环 ZZ 和递归算子 Sn
    _, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    # 使用 HolonomicFunction 对象的方法 to_sequence 创建 HolonomicSequence 对象 p
    p = HolonomicFunction(x**2*Dx**4 + x + Dx, x).to_sequence()
    # 创建 HolonomicSequence 对象 q
    q = [(HolonomicSequence(1 + (n + 2)*Sn**2 + (n**4 + 6*n**3 + 11*n**2 + 6*n)*Sn**3), 0, 1)]
    # 断言 p 和 q 相等
    assert p == q
    # 使用 HolonomicFunction 对象的方法 to_sequence 创建 HolonomicSequence 对象 p
    p = HolonomicFunction(x**2*Dx**4 + x**3 + Dx**2, x).to_sequence()
    # 创建 HolonomicSequence 对象 q
    q = [(HolonomicSequence(1 + (n**4 + 14*n**3 + 72*n**2 + 163*n + 140)*Sn**5), 0, 0)]
    # 断言 p 和 q 相等
    assert p == q
    # 使用 HolonomicFunction 对象的方法 to_sequence 创建 HolonomicSequence 对象 p
    p = HolonomicFunction(x**3*Dx**4 + 1 + Dx**2, x).to_sequence()
    # 创建 HolonomicSequence 对象 q
    q = [(HolonomicSequence(1 + (n**4 - 2*n**3 - n**2 + 2*n)*Sn + (n**2 + 3*n + 2)*Sn**2), 0, 0)]
    # 断言 p 和 q 相等
    assert p == q
    # 使用 HolonomicFunction 对象的方法 to_sequence 创建 HolonomicSequence 对象 p
    p = HolonomicFunction(3*x**3*Dx**4 + 2*x*Dx + x*Dx**3, x).to_sequence()
    # 创建 HolonomicSequence 对象 q
    q = [(HolonomicSequence(2*n + (3*n**4 - 6*n**3 - 3*n**2 + 6*n)*Sn + (n**3 + 3*n**2 + 2*n)*Sn**2), 0, 1)]
    # 断言 p 和 q 相等
    assert p == q


def test_to_Sequence_Initial_Coniditons():
    # 符号 'x' 和整数 'n' 的定义
    x = symbols('x')
    n = symbols('n', integer=True)
    # 创建有理函数环 QQ 和微分算子 Dx
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 创建有理函数环 QQ 和递归算子 Sn
    _, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    # 使用 HolonomicFunction 对象的方法 to_sequence 创建 HolonomicSequence 对象 p
    p = HolonomicFunction(Dx - 1, x, 0, [1]).to_sequence()
    # 创建 HolonomicSequence 对象 q
    q = [(HolonomicSequence(-1 + (n + 1)*Sn, 1), 0)]
    # 断言 p 和 q 相等
    assert p == q
    # 使用 HolonomicFunction 对象的方法 to_sequence 创建 HolonomicSequence 对象 p
    p = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).to_sequence()
    # 创建 HolonomicSequence 对象 q
    q = [(HolonomicSequence(1 + (n**2 + 3*n + 2)*Sn**2, [0, 1]), 0)]
    # 断言 p 和 q 相等
    assert p == q
    # 使用 HolonomicFunction 对象的方法 to_sequence 创建 HolonomicSequence 对象 p
    p = HolonomicFunction(Dx**2 + 1 + x**3*Dx, x, 0, [2, 3]).to_sequence()
    # 没有完整的断言，因为代码片段未完成
    # 创建一个包含 HolonomicSequence 对象和整数 1 的元组列表 q
    q = [(HolonomicSequence(n + Sn**2 + (n**2 + 7*n + 12)*Sn**4, [2, 3, -1, Rational(-1, 2), Rational(1, 12)]), 1)]
    # 断言语句，验证 p 是否等于 q
    assert p == q
    
    # 将 x**3*Dx**5 + 1 + Dx 表达式转换为 HolonomicFunction 对象，并获取其序列形式赋值给 p
    p = HolonomicFunction(x**3*Dx**5 + 1 + Dx, x).to_sequence()
    # 创建一个包含 HolonomicSequence 对象和整数 3 的元组列表 q
    q = [(HolonomicSequence(1 + (n + 1)*Sn + (n**5 - 5*n**3 + 4*n)*Sn**2), 0, 3)]
    # 断言语句，验证 p 是否等于 q
    assert p == q
    
    # 创建符号 C_0, C_1, C_2, C_3
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    # 将 log(1+x**2) 表达式转换为 HolonomicSequence 对象的序列形式赋值给 p
    p = expr_to_holonomic(log(1+x**2))
    # 创建一个包含 HolonomicSequence 对象和整数 1 的元组列表 q
    q = [(HolonomicSequence(n**2 + (n**2 + 2*n)*Sn**2, [0, 0, C_2]), 0, 1)]
    # 断言语句，验证 p 的序列形式是否等于 q
    assert p.to_sequence() == q
    
    # 对 p 序列形式进行微分操作
    p = p.diff()
    # 创建一个包含 HolonomicSequence 对象和整数 0 的元组列表 q
    q = [(HolonomicSequence((n + 2) + (n + 2)*Sn**2, [C_0, 0]), 1, 0)]
    # 断言语句，验证 p 的序列形式是否等于 q
    assert p.to_sequence() == q
    
    # 将 erf(x) + x 表达式转换为 HolonomicSequence 对象的序列形式赋值给 p
    p = expr_to_holonomic(erf(x) + x).to_sequence()
    # 创建一个包含 HolonomicSequence 对象和整数 2 的元组列表 q
    q = [(HolonomicSequence((2*n**2 - 2*n) + (n**3 + 2*n**2 - n - 2)*Sn**2, [0, 1 + 2/sqrt(pi), 0, C_3]), 0, 2)]
    # 断言语句，验证 p 是否等于 q
    assert p == q
# 定义一个测试函数，用于测试符号计算相关的功能
def test_series():
    # 符号 x 的定义
    x = symbols('x')
    # 定义一个多项式环 R 和微分算子 Dx
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')

    # 创建一个 HolonomicFunction 对象 p，表示一个特定微分方程的级数展开
    p = HolonomicFunction(Dx**2 + 2*x*Dx, x, 0, [0, 1]).series(n=10)
    # 创建一个参考值 q，用于验证 p 的结果
    q = x - x**3/3 + x**5/10 - x**7/42 + x**9/216 + O(x**10)
    # 断言 p 等于 q
    assert p == q

    # 创建另一个 HolonomicFunction 对象 p，表示 e^(x**2) 的级数展开
    p = HolonomicFunction(Dx - 1, x).composition(x**2, 0, [1])
    # 创建一个 HolonomicFunction 对象 q，表示 cos(x) 的级数展开
    q = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])
    # 创建一个新的 HolonomicFunction 对象 r，表示 p * q 的级数展开
    r = (p * q).series(n=10)
    # 创建一个参考值 s，用于验证 r 的结果
    s = 1 + x**2/2 + x**4/24 - 31*x**6/720 - 179*x**8/8064 + O(x**10)
    # 断言 r 等于 s
    assert r == s

    # 创建一个 HolonomicFunction 对象 t，表示 log(1 + x) 的级数展开
    t = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1])
    # 创建一个新的 HolonomicFunction 对象 r，表示 (p * t + q) 的级数展开
    r = (p * t + q).series(n=10)
    # 创建一个参考值 s，用于验证 r 的结果
    s = 1 + x - x**2 + 4*x**3/3 - 17*x**4/24 + 31*x**5/30 - 481*x**6/720 + \
        71*x**7/105 - 20159*x**8/40320 + 379*x**9/840 + O(x**10)
    # 断言 r 等于 s
    assert r == s

    # 创建一个 HolonomicFunction 对象 p，表示 (6+6*x-3*x**2) - (10*x-3*x**2-3*x**3)*Dx + (4-6*x**3+2*x**4)*Dx**2 的级数展开
    p = HolonomicFunction((6+6*x-3*x**2) - (10*x-3*x**2-3*x**3)*Dx + \
        (4-6*x**3+2*x**4)*Dx**2, x, 0, [0, 1]).series(n=7)
    # 创建一个参考值 q，用于验证 p 的结果
    q = x + x**3/6 - 3*x**4/16 + x**5/20 - 23*x**6/960 + O(x**7)
    # 断言 p 等于 q
    assert p == q

    # 创建一个 HolonomicFunction 对象 p，表示 (6+6*x-3*x**2) - (10*x-3*x**2-3*x**3)*Dx + (4-6*x**3+2*x**4)*Dx**2 的级数展开
    p = HolonomicFunction((6+6*x-3*x**2) - (10*x-3*x**2-3*x**3)*Dx + \
        (4-6*x**3+2*x**4)*Dx**2, x, 0, [1, 0]).series(n=7)
    # 创建一个参考值 q，用于验证 p 的结果
    q = 1 - 3*x**2/4 - x**3/4 - 5*x**4/32 - 3*x**5/40 - 17*x**6/384 + O(x**7)
    # 断言 p 等于 q
    assert p == q

    # 创建一个 HolonomicFunction 对象 p，表示 erf(x) + x 的级数展开
    p = expr_to_holonomic(erf(x) + x).series(n=10)
    # 创建一个新的符号 C_3
    C_3 = symbols('C_3')
    # 创建一个 HolonomicFunction 对象 q，表示 erf(x) + x 的级数展开
    q = (erf(x) + x).series(n=10)
    # 断言将 C_3 替换为 -2/(3*sqrt(pi)) 后的 p 等于 q
    assert p.subs(C_3, -2/(3*sqrt(pi))) == q

    # 断言将 sqrt(x**3 + x) 转换为 HolonomicFunction 后的级数展开结果等于原始表达式的级数展开结果
    assert expr_to_holonomic(sqrt(x**3 + x)).series(n=10) == sqrt(x**3 + x).series(n=10)

    # 断言将 (2*x - 3*x**2)**Rational(1, 3) 转换为 HolonomicFunction 后的级数展开结果等于原始表达式的级数展开结果
    assert expr_to_holonomic((2*x - 3*x**2)**Rational(1, 3)).series() == ((2*x - 3*x**2)**Rational(1, 3)).series()

    # 断言将 cos(x)**2/x**2 转换为 HolonomicFunction 后的级数展开结果等于原始表达式的级数展开结果
    assert expr_to_holonomic(cos(x)**2/x**2, y0={-2: [1, 0, -1]}).series(n=10) == (cos(x)**2/x**2).series(n=10)

    # 断言将 cos(x)**2/x**2 转换为 HolonomicFunction 后的级数展开结果的合并形式等于原始表达式的级数展开结果的合并形式
    assert expr_to_holonomic(cos(x)**2/x**2, x0=1).series(n=10).together() == (cos(x)**2/x**2).series(n=10, x0=1).together()

    # 断言将 cos(x-1)**2/(x-1)**2 转换为 HolonomicFunction 后的级数展开结果等于原始表达式的级数展开结果
    assert expr_to_holonomic(cos(x-1)**2/(x-1)**2, x0=1, y0={-2: [1, 0, -1]}).series(n=10) \
        == (cos(x-1)**2/(x-1)**2).series(x0=1, n=10)
    # 使用指定方法计算 p 在给定路径 r 上的值，取最后一个值，并将其转换为字符串形式
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # 使用线性路径从 0 到 pi/2 计算 sin(pi/2) 的值
    r = [0.1]
    for i in range(14):
        r.append(r[-1] + 0.1)
    r.append(pi/2)
    s = '1.08016557252834' # 接近于 1.0 的精确解
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # 使用矩形路径计算 sin(pi/2) 的值，从 0 到 i，再从 i 到 pi/2
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(15):
        r.append(r[-1]+0.1)
    r.append(pi/2+I)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    # 接近于 1.0 的值
    s = '0.976882381836257 - 1.65557671738537e-16*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # 创建一个关于 cos(x) 的 HolonomicFunction 对象，从 0 到 pi 计算 cos(pi) 的值
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])
    r = [0.05]
    for i in range(61):
        r.append(r[-1]+0.05)
    r.append(pi)
    # 精确解为 -1
    s = '-1.08140824719196'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s

    # 使用矩形路径计算值，从 0 到 i，再从 2+i 到 2
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(20):
        r.append(r[-1]+0.1)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    # 对 HolonomicFunction 对象 p 在给定路径 r 上的最后一个值进行 Euler 方法计算，并转换为字符串形式
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1,1]).evalf(r, method='Euler')
    s = '0.501421652861245 - 3.88578058618805e-16*I'
    assert sstr(p[-1]) == s
def test_evalf_rk4():
    # 定义符号变量 x
    x = symbols('x')
    # 创建有理数域 QQ 上的旧多项式环 R 和微分操作符 Dx
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # 定义 HolonomicFunction 对象 p，表示 log(1+x)
    p = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1])

    # 定义路径 r 为从 0 到 1 的直线段，沿实轴
    r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # 预期的值 s，接近 log(2)
    s = '0.693146363174626'
    assert sstr(p.evalf(r)[-1]) == s

    # 定义路径 r 为从 0.1+0.1i 开始的三角形路径，最终到达 1+0.1i
    r = [0.1 + 0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1+0.1*I)
    for i in range(10):
        r.append(r[-1]+0.1-0.1*I)

    # 预期的值 s，接近 1.09861228866811，虚部接近零
    s = '1.098616 + 1.36083e-7*I'
    assert sstr(p.evalf(r)[-1].n(7)) == s

    # 定义 HolonomicFunction 对象 p，表示 sin(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])
    # 预期的值 s，接近 0.90929463522785，虚部接近零
    s = '0.90929463522785 + 1.52655665885959e-16*I'
    assert sstr(p.evalf(r)[-1]) == s

    # 定义路径 r 为从 0 到 pi/2 的线性路径，计算 sin(pi/2)
    r = [0.1]
    for i in range(14):
        r.append(r[-1] + 0.1)
    r.append(pi/2)
    # 预期的值 s，接近 0.999999895088917
    s = '0.999999895088917'
    assert sstr(p.evalf(r)[-1]) == s

    # 定义路径 r 为从 0.1i 开始的矩形路径，计算 sin(pi/2)
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(15):
        r.append(r[-1]+0.1)
    r.append(pi/2+I)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    # 预期的值 s，接近 1.00000003415141，虚部接近零
    s = '1.00000003415141 + 6.11940487991086e-16*I'
    assert sstr(p.evalf(r)[-1]) == s

    # 定义 HolonomicFunction 对象 p，表示 cos(x)
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0])
    # 定义路径 r 为从 0 到 pi 的线性路径，计算 cos(pi)
    r = [0.05]
    for i in range(61):
        r.append(r[-1]+0.05)
    r.append(pi)
    # 预期的值 s，接近 -1
    s = '-0.999999993238714'
    assert sstr(p.evalf(r)[-1]) == s

    # 定义路径 r 为从 0.1i 开始的矩形路径，计算 cos(pi)
    r = [0.1*I]
    for i in range(9):
        r.append(r[-1]+0.1*I)
    for i in range(20):
        r.append(r[-1]+0.1)
    for i in range(10):
        r.append(r[-1]-0.1*I)

    # 计算 HolonomicFunction 对象 p 在路径 r 上的值，预期的值 s
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1,1]).evalf(r)
    s = '0.493152791638442 - 1.41553435639707e-15*I'
    assert sstr(p[-1]) == s
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction((2*x**3 + 10*x**2 + 20*x + 18) + (-2*x**4 - 10*x**3 - 20*x**2 \
        - 18*x)*Dx + (2*x**5 + 6*x**4 + 7*x**3 + 8*x**2 + 10*x - 4)*Dx**2 + \
        (-2*x**5 - 5*x**4 - 2*x**3 + 2*x**2 - x + 4)*Dx**3 + (x**5 + 2*x**4 - x**3 - \
        7*x**2/2 + x + Rational(5, 2))*Dx**4, x, 0, [0, 1, 4, -1])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(x*exp(x)+cos(x)+1)
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction((-x - 3)*Dx + (x + 2)*Dx**2 + (-x - 3)*Dx**3 + (x + 2)*Dx**4, x, \
        0, [2, 1, 1, 3])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 断言语句，用于验证表达式的 Taylor 级数展开与 p 的 Taylor 级数展开是否相等
    assert (x*exp(x)+cos(x)+1).series(n=10) == p.series(n=10)
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(log(1 + x)**2 + 1)
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction(Dx + (3*x + 3)*Dx**2 + (x**2 + 2*x + 1)*Dx**3, x, 0, [1, 0, 2])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(erf(x)**2 + x)
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction((8*x**4 - 2*x**2 + 2)*Dx**2 + (6*x**3 - x/2)*Dx**3 + \
        (x**2+ Rational(1, 4))*Dx**4, x, 0, [0, 1, 8/pi, 0])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(cosh(x)*x)
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction((-x**2 + 2) -2*x*Dx + x**2*Dx**2, x, 0, [0, 1])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(besselj(2, x))
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction((x**2 - 4) + x*Dx + x**2*Dx**2, x, 0, [0, 0])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(besselj(0, x) + exp(x))
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction((-x**2 - x/2 + S.Half) + (x**2 - x/2 - Rational(3, 2))*Dx + (-x**2 + x/2 + 1)*Dx**2 +\
        (x**2 + x/2)*Dx**3, x, 0, [2, 1, S.Half])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(sin(x)**2/x)
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction(4 + 4*x*Dx + 3*Dx**2 + x*Dx**3, x, 0, [0, 1, 0])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(sin(x)**2/x, x0=2)
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction((4) + (4*x)*Dx + (3)*Dx**2 + (x)*Dx**3, x, 2, [sin(2)**2/2,
        sin(2)*cos(2) - sin(2)**2/4, -3*sin(2)**2/4 + cos(2)**2 - sin(2)*cos(2)])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(log(x)/2 - Ci(2*x)/2 + Ci(2)/2)
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction(4*Dx + 4*x*Dx**2 + 3*Dx**3 + x*Dx**4, x, 0, \
        [-log(2)/2 - EulerGamma/2 + Ci(2)/2, 0, 1, 0])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 将 p 转换为表达式对象，并赋值给 p
    p = p.to_expr()
    # 将一个表达式赋值给 q
    q = log(x)/2 - Ci(2*x)/2 + Ci(2)/2
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(x**S.Half, x0=1)
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = HolonomicFunction(x*Dx - S.Half, x, 1, [1])
    # 断言语句，用于验证 p 和 q 对象是否相等
    assert p == q
    
    # 使用 expr_to_holonomic 函数将表达式转换为 HolonomicFunction 对象，并赋值给 p
    p = expr_to_holonomic(sqrt(1 + x**2))
    # 使用 HolonomicFunction 类创建一个名为 q 的对象，表示一个分析函数的线性组合
    q = Holonomic
# 定义测试函数 test_to_hyper
def test_to_hyper():
    # 创建符号变量 x
    x = symbols('x')
    # 使用 QQ.old_poly_ring(x) 创建有理数域和 x 的微分操作符 Dx
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 使用 HolonomicFunction 类创建一个超几何函数 p，其微分操作为 Dx - 2，初始条件为 x=0，系数为 [3]
    p = HolonomicFunction(Dx - 2, x, 0, [3]).to_hyper()
    # 创建一个超几何函数 q，其形式为 3 * hyper([], [], 2*x)
    q = 3 * hyper([], [], 2*x)
    # 断言 p 等于 q
    assert p == q
    # 对 HolonomicFunction((1 + x) * Dx - 3, x, 0, [2]) 创建的超几何函数应用 hyperexpand 并展开
    p = hyperexpand(HolonomicFunction((1 + x) * Dx - 3, x, 0, [2]).to_hyper()).expand()
    # 创建一个期望值 q，其形式为 2*x**3 + 6*x**2 + 6*x + 2
    q = 2*x**3 + 6*x**2 + 6*x + 2
    # 断言 p 等于 q
    assert p == q
    # 使用 HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1]) 创建超几何函数 p
    p = HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1]).to_hyper()
    # 创建期望值 q，其形式为 -x**2*hyper((2, 2, 1), (3, 2), -x)/2 + x
    q = -x**2*hyper((2, 2, 1), (3, 2), -x)/2 + x
    # 断言 p 等于 q
    assert p == q
    # 使用 HolonomicFunction(2*x*Dx + Dx**2, x, 0, [0, 2/sqrt(pi)]) 创建超几何函数 p
    p = HolonomicFunction(2*x*Dx + Dx**2, x, 0, [0, 2/sqrt(pi)]).to_hyper()
    # 创建期望值 q，其形式为 2*x*hyper((S.Half,), (Rational(3, 2),), -x**2)/sqrt(pi)
    q = 2*x*hyper((S.Half,), (Rational(3, 2),), -x**2)/sqrt(pi)
    # 断言 p 等于 q
    assert p == q
    # 对 HolonomicFunction(2*x*Dx + Dx**2, x, 0, [1, -2/sqrt(pi)]) 创建的超几何函数应用 hyperexpand
    p = hyperexpand(HolonomicFunction(2*x*Dx + Dx**2, x, 0, [1, -2/sqrt(pi)]).to_hyper())
    # 创建期望值 q，其形式为 erfc(x)
    q = erfc(x)
    # 断言 p 重写为 erfc 后等于 q
    assert p.rewrite(erfc) == q
    # 对 HolonomicFunction((x**2 - 1) + x*Dx + x**2*Dx**2, x, 0, [0, S.Half]) 创建超几何函数并应用 hyperexpand
    p = hyperexpand(HolonomicFunction((x**2 - 1) + x*Dx + x**2*Dx**2,
        x, 0, [0, S.Half]).to_hyper())
    # 创建期望值 q，其形式为 besselj(1, x)
    q = besselj(1, x)
    # 断言 p 等于 q
    assert p == q
    # 对 HolonomicFunction(x*Dx**2 + Dx + x, x, 0, [1, 0]) 创建超几何函数并应用 hyperexpand
    p = hyperexpand(HolonomicFunction(x*Dx**2 + Dx + x, x, 0, [1, 0]).to_hyper())
    # 创建期望值 q，其形式为 besselj(0, x)
    q = besselj(0, x)
    # 断言 p 等于 q
    assert p == q

# 定义测试函数 test_to_expr
def test_to_expr():
    # 创建符号变量 x
    x = symbols('x')
    # 使用 ZZ.old_poly_ring(x) 创建整数域和 x 的微分操作符 Dx
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    # 使用 HolonomicFunction 类创建一个表达式 p，其微分操作为 Dx - 1，初始条件为 x=0，系数为 [1]
    p = HolonomicFunction(Dx - 1, x, 0, [1]).to_expr()
    # 创建期望值 q，其形式为 exp(x)
    q = exp(x)
    # 断言 p 等于 q
    assert p == q
    # 使用 HolonomicFunction(Dx**2 + 1, x, 0, [1, 0]) 创建表达式 p
    p = HolonomicFunction(Dx**2 + 1, x, 0, [1, 0]).to_expr()
    # 创建期望值 q，其形式为 cos(x)
    q = cos(x)
    # 断言 p 等于 q
    assert p == q
    # 使用 HolonomicFunction(Dx**2 - 1, x, 0, [1, 0]) 创建表达式 p
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 0]).to_expr()
    # 创建期望值 q，其形式为 cosh(x)
    q = cosh(x)
    # 断言 p 等于 q
    assert p == q
    # 使用 HolonomicFunction(2 + (4*x - 1)*Dx + (x**2 - x)*Dx**2, x, 0, [1, 2]) 创建表达式 p 并展开
    p = HolonomicFunction(2 + (4*x - 1)*Dx + \
        (x**2 - x)*Dx**2, x, 0, [1, 2]).to_expr().expand()
    # 创建期望值 q，其形式为 1/(x**2 - 2*x + 1)
    q = 1/(x**2 - 2*x + 1)
    # 断言 p 等于 q
    assert p == q
    # 对 sin(x)**2/x 应用 expr_to_holonomic 并对其积分，再转换为表达式 p
    p = expr_to_holonomic(sin(x)**2/x).integrate((x, 0, x)).to_expr()
    # 创建期望值 q，其形式为 (sin(x)**2/x).integrate((x, 0, x))
    q = (sin(x)**2/x).integrate((x, 0, x))
    # 断言 p 等于 q
    assert p == q
    # 创建符号变量 C_0, C_1, C_2, C_3
    C_0, C_1, C_2, C_3 = symbols('C_0, C_1, C_2, C_3')
    # 对 log(1+x**2) 应用 expr_to_holonomic 并转换为表达式 p
    p = expr_to_holonomic(log(1+x**2)).to_expr()
    # 创建期望值 q，其形式为 C_2*log(x**2 + 1)
    q = C_2*log(x**2 + 1)
    # 断言 p 等于 q
    assert p == q
    # 对 log(1+x**2) 应用 expr_to_holonomic 并对其求导后转换为表达式 p
    p = expr_to_holonomic(log(1+x**2)).diff().to_expr()
    # 创建期望值 q，其形式为 C_0*x/(x**2 + 1)
    q = C_0*x/(x**2 + 1)
    # 断言 p 等于 q
    assert p == q
    # 对 erf(x) + x 应用 expr_to_holonomic 并转换为表达式 p
    p = expr_to_holonomic(erf(x) + x).to_expr()
    # 创建期
    # 断言：将一个多项式转换为相应的霍洛莫尼克函数再转换回表达式后，应该与原多项式相同
    assert expr_to_holonomic(2*x**3 - 3*x**2).to_expr().expand() == \
        2*x**3 - 3*x**2
    
    # 创建一个符号变量 'a'
    a = symbols("a")
    
    # 将两个霍洛莫尼克函数的乘积转换为表达式
    p = (expr_to_holonomic(1.4*x) * expr_to_holonomic(a*x, x)).to_expr()
    
    # 计算期望的乘积表达式
    q = 1.4*a*x**2
    
    # 断言：两个表达式应该相等
    assert p == q
    
    # 将两个霍洛莫尼克函数的和转换为表达式
    p = (expr_to_holonomic(1.4*x) + expr_to_holonomic(a*x, x)).to_expr()
    
    # 计算期望的和表达式
    q = x*(a + 1.4)
    
    # 断言：两个表达式应该相等
    assert p == q
    
    # 将两个霍洛莫尼克函数的和转换为表达式
    p = (expr_to_holonomic(1.4*x) + expr_to_holonomic(x)).to_expr()
    
    # 断言：表达式应该等于期望的表达式
    assert p == 2.4*x
# 定义测试函数 test_integrate()
def test_integrate():
    # 符号变量 x
    x = symbols('x')
    # 创建整数环和微分算子
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    # 将表达式 sin(x)**2/x 转换为霍洛莫尼克函数，并对 x 从 2 到 3 积分，得到 p
    p = expr_to_holonomic(sin(x)**2/x, x0=1).integrate((x, 2, 3))
    # 预期的结果字符串形式 '0.166270406994788'，与 p 相等
    q = '0.166270406994788'
    assert sstr(p) == q
    # 将表达式 sin(x) 转换为霍洛莫尼克函数，并对 x 从 0 积分到 x，再转换为表达式，得到 p
    p = expr_to_holonomic(sin(x)).integrate((x, 0, x)).to_expr()
    # 预期的结果表达式 1 - cos(x)，与 p 相等
    q = 1 - cos(x)
    assert p == q
    # 将表达式 sin(x) 转换为霍洛莫尼克函数，并对 x 从 0 积分到 3，得到 p
    p = expr_to_holonomic(sin(x)).integrate((x, 0, 3))
    # 预期的结果表达式 1 - cos(3)，与 p 相等
    q = 1 - cos(3)
    assert p == q
    # 将表达式 sin(x)/x 转换为霍洛莫尼克函数，并对 x 从 1 积分到 2，得到 p
    p = expr_to_holonomic(sin(x)/x, x0=1).integrate((x, 1, 2))
    # 预期的结果字符串形式 '0.659329913368450'，与 p 相等
    q = '0.659329913368450'
    assert sstr(p) == q
    # 将表达式 sin(x)**2/x 转换为霍洛莫尼克函数，并对 x 从 1 积分到 0，得到 p
    p = expr_to_holonomic(sin(x)**2/x, x0=1).integrate((x, 1, 0))
    # 预期的结果字符串形式 '-0.423690480850035'，与 p 相等
    q = '-0.423690480850035'
    assert sstr(p) == q
    # 将表达式 sin(x)/x 转换为霍洛莫尼克函数，并获取其积分后的霍洛莫尼克函数，与 Si(x) 相等
    p = expr_to_holonomic(sin(x)/x)
    assert p.integrate(x).to_expr() == Si(x)
    # 对表达式 sin(x)/x 进行从 0 积分到 2，结果应为 Si(2)
    assert p.integrate((x, 0, 2)) == Si(2)
    # 将表达式 sin(x)**2/x 转换为霍洛莫尼克函数，并获取其霍洛莫尼克函数，与其表达式形式相等
    p = expr_to_holonomic(sin(x)**2/x)
    q = p.to_expr()
    # 对表达式 sin(x)**2/x 进行积分后的霍洛莫尼克函数，与从 0 积分到 x 后的 q 相等
    assert p.integrate(x).to_expr() == q.integrate((x, 0, x))
    # 对表达式 sin(x)**2/x 进行从 0 积分到 1，结果应与从 0 积分到 1 的 q 相等
    assert p.integrate((x, 0, 1)) == q.integrate((x, 0, 1))
    # 将表达式 1/x 转换为霍洛莫尼克函数，x0=1，对 x 积分后转换为表达式，结果应为 log(x)
    assert expr_to_holonomic(1/x, x0=1).integrate(x).to_expr() == log(x)
    # 将表达式 (x + 1)**3*exp(-x) 转换为霍洛莫尼克函数，x0=-1，对 x 积分后转换为表达式，结果应为 q
    p = expr_to_holonomic((x + 1)**3*exp(-x), x0=-1).integrate(x).to_expr()
    q = (-x**3 - 6*x**2 - 15*x + 6*exp(x + 1) - 16)*exp(-x)
    assert p == q
    # 将表达式 cos(x)**2/x**2 转换为霍洛莫尼克函数，y0 设为 {-2: [1, 0, -1]}，对 x 积分后转换为表达式，结果应为 q
    p = expr_to_holonomic(cos(x)**2/x**2, y0={-2: [1, 0, -1]}).integrate(x).to_expr()
    q = -Si(2*x) - cos(x)**2/x
    assert p == q
    # 将表达式 sqrt(x**2+x) 转换为霍洛莫尼克函数，对 x 积分后转换为表达式，结果应为 q
    p = expr_to_holonomic(sqrt(x**2+x)).integrate(x).to_expr()
    q = (x**Rational(3, 2)*(2*x**2 + 3*x + 1) - x*sqrt(x + 1)*asinh(sqrt(x)))/(4*x*sqrt(x + 1))
    assert p == q
    # 将表达式 sqrt(x**2+1) 转换为霍洛莫尼克函数，对 x 积分后转换为表达式，结果应与 q 的积分形式相等
    p = expr_to_holonomic(sqrt(x**2+1)).integrate(x).to_expr()
    q = (sqrt(x**2+1)).integrate(x)
    assert (p-q).simplify() == 0
    # 将表达式 1/x**2 转换为霍洛莫尼克函数，y0 设为 {-2:[1, 0, 0]}，结果应与 r 相等
    p = expr_to_holonomic(1/x**2, y0={-2:[1, 0, 0]})
    r = expr_to_holonomic(1/x**2, lenics=3)
    assert p == r
    # 将表达式 cos(x)**2 转换为霍洛莫尼克函数，与 r 的乘积积分后转换为表达式，结果应为 -Si(2*x) - cos(x)**2/x
    q = expr_to_holonomic(cos(x)**2)
    assert (r*q).integrate(x).to_expr() == -Si(2*x) - cos(x)**2/x


# 定义测试函数 test_diff()
def test_diff():
    # 符号变量 x, y
    x, y = symbols('x, y')
    # 创建整数环和微分算子
    R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    # 创建 HolonomicFunction 对象 p，使用 x*Dx**2 + 1 初始化，x0=0，初始条件 [0, 1]
    p = HolonomicFunction(x*Dx**2 + 1, x, 0, [0, 1])
    # 对 p 求导后的表达式，应与 p 的表达式求导并简化后相等
    assert p.diff().to_expr() == p.to_expr().diff().simplify()
    # 创建 HolonomicFunction 对象 p，使用 Dx**2 - 1 初始化，x0=0，初始条件 [1, 0]
    p = HolonomicFunction(Dx**2 - 1, x, 0, [1, 0])
    # 对 p 进行二阶 x 方向的求导后的表达式，应
    # 断言检查 p 和 q 是否相等
    assert p == q
    # 断言检查 p 转换为表达式后是否等于 1.1329138213*x
    assert p.to_expr() == 1.1329138213*x
    # 断言检查 p 在区间 (1, 2) 上的积分是否等于 (1.1329138213*x) 在同一区间上的积分
    assert sstr(p.integrate((x, 1, 2))) == sstr((1.1329138213*x).integrate((x, 1, 2)))
    # 定义符号变量 y 和 z
    y, z = symbols('y, z')
    # 将 sin(x*y*z) 转换为霍洛莫尼克对象，并赋给 p
    p = expr_to_holonomic(sin(x*y*z), x=x)
    # 断言检查 p 转换为表达式后是否等于 sin(x*y*z)
    assert p.to_expr() == sin(x*y*z)
    # 断言检查 p 对 x 的积分转换为表达式后是否等于 (-cos(x*y*z) + 1)/(y*z)
    assert p.integrate(x).to_expr() == (-cos(x*y*z) + 1)/(y*z)
    # 将 sin(x*y + z) 转换为霍洛莫尼克对象，并对 x 积分后再转换为表达式赋给 p
    p = expr_to_holonomic(sin(x*y + z), x=x).integrate(x).to_expr()
    # 定义符号表达式 q
    q = (cos(z) - cos(x*y + z))/y
    # 断言检查 p 是否等于 q
    assert p == q
    # 定义符号变量 a
    a = symbols('a')
    # 将 a*x 转换为霍洛莫尼克对象赋给 p
    p = expr_to_holonomic(a*x, x)
    # 断言检查 p 转换为表达式后是否等于 a*x
    assert p.to_expr() == a*x
    # 断言检查 p 对 x 的积分转换为表达式后是否等于 a*x**2/2
    assert p.integrate(x).to_expr() == a*x**2/2
    # 定义符号变量 D_2 和 C_1
    D_2, C_1 = symbols("D_2, C_1")
    # 将 x 和 1.2*cos(x) 转换为霍洛莫尼克对象并相加，赋给 p
    p = expr_to_holonomic(x) + expr_to_holonomic(1.2*cos(x))
    # 将 p 转换为表达式后，替换掉所有的 D_2，并重新赋给 p
    p = p.to_expr().subs(D_2, 0)
    # 断言检查 p 减去 x 减去 1.2*cos(1.0*x) 是否等于 0
    assert p - x - 1.2*cos(1.0*x) == 0
    # 将 x 和 1.2*cos(x) 转换为霍洛莫尼克对象并相乘，赋给 p
    p = expr_to_holonomic(x) * expr_to_holonomic(1.2*cos(x))
    # 将 p 转换为表达式后，替换掉所有的 C_1，并重新赋给 p
    p = p.to_expr().subs(C_1, 0)
    # 断言检查 p 减去 1.2*x*cos(1.0*x) 是否等于 0
    assert p - 1.2*x*cos(1.0*x) == 0
def test_to_meijerg():
    # 定义符号变量 x
    x = symbols('x')
    # 测试 hyperexpand(expr_to_holonomic(sin(x)).to_meijerg()) 是否等于 sin(x)
    assert hyperexpand(expr_to_holonomic(sin(x)).to_meijerg()) == sin(x)
    # 测试 hyperexpand(expr_to_holonomic(cos(x)).to_meijerg()) 是否等于 cos(x)
    assert hyperexpand(expr_to_holonomic(cos(x)).to_meijerg()) == cos(x)
    # 测试 hyperexpand(expr_to_holonomic(exp(x)).to_meijerg()) 是否等于 exp(x)
    assert hyperexpand(expr_to_holonomic(exp(x)).to_meijerg()) == exp(x)
    # 测试 hyperexpand(expr_to_holonomic(log(x)).to_meijerg()).simplify() 是否等于 log(x)
    assert hyperexpand(expr_to_holonomic(log(x)).to_meijerg()).simplify() == log(x)
    # 测试 expr_to_holonomic(4*x**2/3 + 7).to_meijerg() 是否等于 4*x**2/3 + 7
    assert expr_to_holonomic(4*x**2/3 + 7).to_meijerg() == 4*x**2/3 + 7
    # 测试 hyperexpand(expr_to_holonomic(besselj(2, x), lenics=3).to_meijerg()) 是否等于 besselj(2, x)
    assert hyperexpand(expr_to_holonomic(besselj(2, x), lenics=3).to_meijerg()) == besselj(2, x)
    # 定义超函 hyper((Rational(-1, 2), -3), (), x)
    p = hyper((Rational(-1, 2), -3), (), x)
    # 测试 from_hyper(p).to_meijerg() 是否等于 hyperexpand(p)
    assert from_hyper(p).to_meijerg() == hyperexpand(p)
    # 定义超函 hyper((S.One, S(3)), (S(2), ), x)
    p = hyper((S.One, S(3)), (S(2), ), x)
    # 测试 (hyperexpand(from_hyper(p).to_meijerg()) - hyperexpand(p)).expand() 是否为 0
    assert (hyperexpand(from_hyper(p).to_meijerg()) - hyperexpand(p)).expand() == 0
    # 定义 p 为 from_hyper(hyper((-2, -3), (S.Half, ), x))
    p = from_hyper(hyper((-2, -3), (S.Half, ), x))
    # 定义 s 为 hyperexpand(hyper((-2, -3), (S.Half, ), x))
    s = hyperexpand(hyper((-2, -3), (S.Half, ), x))
    # 定义符号 C_0, C_1, D_0
    C_0 = Symbol('C_0')
    C_1 = Symbol('C_1')
    D_0 = Symbol('D_0')
    # 测试 (hyperexpand(p.to_meijerg()).subs({C_0:1, D_0:0}) - s).simplify() 是否为 0
    assert (hyperexpand(p.to_meijerg()).subs({C_0:1, D_0:0}) - s).simplify() == 0
    # 设置 p.y0 的值为 {0: [1], S.Half: [0]}
    p.y0 = {0: [1], S.Half: [0]}
    # 测试 (hyperexpand(p.to_meijerg()) - s).simplify() 是否为 0
    assert (hyperexpand(p.to_meijerg()) - s).simplify() == 0
    # 定义 p 为 expr_to_holonomic(besselj(S.Half, x), initcond=False)
    p = expr_to_holonomic(besselj(S.Half, x), initcond=False)
    # 测试 (p.to_expr() - (D_0*sin(x) + C_0*cos(x) + C_1*sin(x))/sqrt(x)).simplify() 是否为 0
    assert (p.to_expr() - (D_0*sin(x) + C_0*cos(x) + C_1*sin(x))/sqrt(x)).simplify() == 0
    # 定义 p 为 expr_to_holonomic(besselj(S.Half, x), y0={Rational(-1, 2): [sqrt(2)/sqrt(pi), sqrt(2)/sqrt(pi)]})
    p = expr_to_holonomic(besselj(S.Half, x), y0={Rational(-1, 2): [sqrt(2)/sqrt(pi), sqrt(2)/sqrt(pi)]})
    # 测试 (p.to_expr() - besselj(S.Half, x) - besselj(Rational(-1, 2), x)).simplify() 是否为 0
    assert (p.to_expr() - besselj(S.Half, x) - besselj(Rational(-1, 2), x)).simplify() == 0


def test_gaussian():
    # 定义符号变量 mu, x
    mu, x = symbols("mu x")
    # 定义正数符号变量 sd
    sd = symbols("sd", positive=True)
    # 创建有理数域 QQ[mu, sd]
    Q = QQ[mu, sd].get_field()
    # 定义 e
    e = sqrt(2)*exp(-(-mu + x)**2/(2*sd**2))/(2*sqrt(pi)*sd)
    # 将 e 转换为霍洛米形式 h1
    h1 = expr_to_holonomic(e, x, domain=Q)

    # 创建微分算子 Dx
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    # 创建霍洛米函数 h2
    h2 = HolonomicFunction((-mu/sd**2 + x/sd**2) + (1)*Dx, x)

    # 断言 h1 与 h2 相等
    assert h1 == h2


def test_beta():
    # 定义正数符号变量 a, b, x
    a, b, x = symbols("a b x", positive=True)
    # 定义 e
    e = x**(a - 1)*(-x + 1)**(b - 1)/beta(a, b)
    # 创建有理数域 QQ[a, b]
    Q = QQ[a, b].get_field()
    # 将 e 转换为霍洛米形式 h1
    h1 = expr_to_holonomic(e, x, domain=Q)

    # 创建微分算子 Dx
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    # 创建霍洛米函数 h2
    h2 = HolonomicFunction((a + x*(-a - b + 2) - 1) + (x**2 - x)*Dx, x)

    # 断言 h1 与 h2 相等
    assert h1 == h2


def test_gamma():
    # 定义正数符号变量 a, b, x
    a, b, x = symbols("a b x", positive=True)
    # 定义 e
    e = b**(-a)*x**(a - 1)*exp(-x/b)/gamma(a)
    # 创建有理数域 QQ[a, b]
    Q = QQ[a, b].get_field()
    # 将 e 转换为霍洛米形式 h1
    h1 = expr_to_holonomic(e, x, domain=Q)

    # 创建微分算子 Dx
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    # 创建霍洛米函数 h2
    h2 = HolonomicFunction((-a + 1 + x/b) + (x)*Dx, x)

    # 断言 h1 与 h2 相等
    assert h1 == h2


def test_symbolic_power():
    # 定义符号变量 x, n
    x, n = symbols("x n")
    # 创建有理数域 QQ[n]
    Q = QQ[n].get_field()
    # 创建微分算子 Dx
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    # 创建霍洛米函数 h1
    h1 = HolonomicFunction((-1) + (x)*Dx, x) ** -n
    # 创建霍洛米函数 h2
    h2 = HolonomicFunction((n) + (x)*Dx, x)

    # 断言 h1 与 h2 相等
    assert h1 == h2


def test_negative_power():
    # 定义符号变量 x
    x = symbols("x")
    # 创建微分算子 Dx
    _, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 创建霍
    # 使用 SymPy 的 symbols 函数创建符号变量 x 和 n
    x, n = symbols("x n")
    
    # 获取有理函数域 QQ[n] 的域对象
    Q = QQ[n].get_field()
    
    # 使用 SymPy 的 DifferentialOperators 函数创建一个微分算子 Dx，
    # 并在多项式环 Q.old_poly_ring(x) 上应用这个微分算子，得到微分算子 Dx
    _, Dx = DifferentialOperators(Q.old_poly_ring(x), 'Dx')
    
    # 创建一个 HolonomicFunction 对象 h1，其定义为 (-1 + x*Dx)^(n - 3)
    h1 = HolonomicFunction((-1) + (x)*Dx, x) ** (n - 3)
    
    # 创建一个 HolonomicFunction 对象 h2，其定义为 (-n + 3 + x*Dx)
    h2 = HolonomicFunction((-n + 3) + (x)*Dx, x)
    
    # 断言 h1 和 h2 相等，用于测试目的
    assert h1 == h2
# 定义一个测试函数，用于测试微分算子的多项式等式
def test_DifferentialOperatorEqPoly():
    # 创建一个整数符号变量 x
    x = symbols('x', integer=True)
    # 使用整数符号 x 创建有理数多项式环 R 和微分算子 Dx
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 创建一个微分算子 do，其多项式为 [x**2, 0, 0]，所属环为 R
    do = DifferentialOperator([x**2, R.base.zero, R.base.zero], R)
    # 创建另一个微分算子 do2，其多项式为 [x**2, 1, x]，所属环为 R
    do2 = DifferentialOperator([x**2, 1, x], R)
    # 断言 do 不等于 do2
    assert not do == do2

    # 下面的代码被注释掉，因为存在多项式比较问题，详见 https://github.com/sympy/sympy/pull/15799
    # 这部分代码在问题解决后应该能够正常工作
    # p = do.listofpoly[0]
    # assert do == p

    # 取出 do2 的第一个多项式 p2
    p2 = do2.listofpoly[0]
    # 断言 do2 不等于 p2
    assert not do2 == p2


# 定义一个测试函数，用于测试微分算子的乘幂运算
def test_DifferentialOperatorPow():
    # 创建一个整数符号变量 x
    x = symbols('x', integer=True)
    # 使用整数符号 x 创建有理数多项式环 R 和微分算子 Dx，只接收第一个返回值 R
    R, _ = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    # 创建一个微分算子 do，其多项式为 [x**2, 0, 0]，所属环为 R
    do = DifferentialOperator([x**2, R.base.zero, R.base.zero], R)
    # 创建一个微分算子 a，其多项式为 [1]，所属环为 R
    a = DifferentialOperator([R.base.one], R)
    # 循环计算 do 的乘幂直到 n = 9
    for n in range(10):
        # 断言 a 等于 do 的 n 次幂
        assert a == do**n
        # 更新 a 为 a 乘以 do
        a *= do
```