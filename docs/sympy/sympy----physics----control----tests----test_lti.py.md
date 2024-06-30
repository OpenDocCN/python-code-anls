# `D:\src\scipysrc\sympy\sympy\physics\control\tests\test_lti.py`

```
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, pi, Rational, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan
from sympy.matrices.dense import eye
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.simplify.simplify import simplify
from sympy.core.containers import Tuple
from sympy.matrices import ImmutableMatrix, Matrix, ShapeError
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.physics.control import (TransferFunction, Series, Parallel,
    Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback,
    StateSpace, gbt, bilinear, forward_diff, backward_diff, phase_margin, gain_margin)
from sympy.testing.pytest import raises

# 定义符号变量
a, x, b, c, s, g, d, p, k, tau, zeta, wn, T = symbols('a, x, b, c, s, g, d, p, k,\
    tau, zeta, wn, T')
# 定义多个符号变量
a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = symbols('a0:4,\
    b0:4, c0:4, d0:4')

# 创建第一个传递函数对象 TF1
TF1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
# 创建第二个传递函数对象 TF2
TF2 = TransferFunction(k, 1, s)
# 创建第三个传递函数对象 TF3
TF3 = TransferFunction(a2*p - s, a2*s + p, s)

# 定义测试函数 test_TransferFunction_construction
def test_TransferFunction_construction():
    # 测试传递函数对象 tf
    tf = TransferFunction(s + 1, s**2 + s + 1, s)
    assert tf.num == (s + 1)
    assert tf.den == (s**2 + s + 1)
    assert tf.args == (s + 1, s**2 + s + 1, s)

    # 测试传递函数对象 tf1
    tf1 = TransferFunction(s + 4, s - 5, s)
    assert tf1.num == (s + 4)
    assert tf1.den == (s - 5)
    assert tf1.args == (s + 4, s - 5, s)

    # 使用不同的多项式变量进行测试 tf2
    tf2 = TransferFunction(p + 3, p**2 - 9, p)
    assert tf2.num == (p + 3)
    assert tf2.den == (p**2 - 9)
    assert tf2.args == (p + 3, p**2 - 9, p)

    # 测试 tf3
    tf3 = TransferFunction(p**3 + 5*p**2 + 4, p**4 + 3*p + 1, p)
    assert tf3.args == (p**3 + 5*p**2 + 4, p**4 + 3*p + 1, p)

    # 没有自己的极点零点取消。
    tf4 = TransferFunction((s + 3)*(s - 1), (s - 1)*(s + 5), s)
    assert tf4.den == (s - 1)*(s + 5)
    assert tf4.args == ((s + 3)*(s - 1), (s - 1)*(s + 5), s)

    # 测试 tf4_
    tf4_ = TransferFunction(p + 2, p + 2, p)
    assert tf4_.args == (p + 2, p + 2, p)

    # 测试 tf5
    tf5 = TransferFunction(s - 1, 4 - p, s)
    assert tf5.args == (s - 1, 4 - p, s)

    # 测试 tf5_
    tf5_ = TransferFunction(s - 1, s - 1, s)
    assert tf5_.args == (s - 1, s - 1, s)

    # 测试 tf6
    tf6 = TransferFunction(5, 6, s)
    assert tf6.num == 5
    assert tf6.den == 6
    assert tf6.args == (5, 6, s)

    # 测试 tf6_
    tf6_ = TransferFunction(1/2, 4, s)
    assert tf6_.num == 0.5
    assert tf6_.den == 4
    assert tf6_.args == (0.500000000000000, 4, s)

    # 测试 tf7 和 tf8
    tf7 = TransferFunction(3*s**2 + 2*p + 4*s, 8*p**2 + 7*s, s)
    tf8 = TransferFunction(3*s**2 + 2*p + 4*s, 8*p**2 + 7*s, p)
    # 断言 tf7 和 tf8 不相等
    assert not tf7 == tf8

    # 创建带有指定参数的传递函数对象 tf7_
    tf7_ = TransferFunction(a0*s + a1*s**2 + a2*s**3, b0*p - b1*s, s)
    # 创建带有相同参数的另一个传递函数对象 tf8_
    tf8_ = TransferFunction(a0*s + a1*s**2 + a2*s**3, b0*p - b1*s, s)
    # 断言 tf7_ 和 tf8_ 相等
    assert tf7_ == tf8_
    # 断言 -(-tf7_) 等于 tf7_，并检查链式取反操作
    assert -(-tf7_) == tf7_ == -(-(-(-tf7_)))

    # 创建带有指定参数的传递函数对象 tf9
    tf9 = TransferFunction(a*s**3 + b*s**2 + g*s + d, d*p + g*p**2 + g*s, s)
    # 断言 tf9 的参数与预期的值相等
    assert tf9.args == (a*s**3 + b*s**2 + d + g*s, d*p + g*p**2 + g*s, s)

    # 创建带有指定参数的传递函数对象 tf10 和 tf10_
    tf10 = TransferFunction(p**3 + d, g*s**2 + d*s + a, p)
    tf10_ = TransferFunction(p**3 + d, g*s**2 + d*s + a, p)
    # 断言 tf10 的参数与预期的值相等
    assert tf10.args == (d + p**3, a + d*s + g*s**2, p)
    # 断言 tf10 和 tf10_ 相等
    assert tf10_ == tf10

    # 创建带有指定参数的传递函数对象 tf11
    tf11 = TransferFunction(a1*s + a0, b2*s**2 + b1*s + b0, s)
    # 断言 tf11 的分子部分与预期的值相等
    assert tf11.num == (a0 + a1*s)
    # 断言 tf11 的分母部分与预期的值相等
    assert tf11.den == (b0 + b1*s + b2*s**2)
    # 断言 tf11 的参数与预期的值相等
    assert tf11.args == (a0 + a1*s, b0 + b1*s + b2*s**2, s)

    # 当分子为 0 时，保持分母不变
    tf12 = TransferFunction(0, p**2 - p + 1, p)
    # 断言 tf12 的参数与预期的值相等
    assert tf12.args == (0, p**2 - p + 1, p)

    # 创建带有指定参数的传递函数对象 tf13
    tf13 = TransferFunction(0, 1, s)
    # 断言 tf13 的参数与预期的值相等
    assert tf13.args == (0, 1, s)

    # 创建带有浮点数指数的传递函数对象 tf14
    tf14 = TransferFunction(a0*s**0.5 + a2*s**0.6 - a1, a1*p**(-8.7), s)
    # 断言 tf14 的参数与预期的值相等
    assert tf14.args == (a0*s**0.5 - a1 + a2*s**0.6, a1*p**(-8.7), s)

    # 创建带有浮点数指数的传递函数对象 tf15
    tf15 = TransferFunction(a2**2*p**(1/4) + a1*s**(-4/5), a0*s - p, p)
    # 断言 tf15 的参数与预期的值相等
    assert tf15.args == (a1*s**(-0.8) + a2**2*p**0.25, a0*s - p, p)

    # 创建符号变量 omega_o, k_p, k_o, k_i，并使用它们创建传递函数对象 tf18
    omega_o, k_p, k_o, k_i = symbols('omega_o, k_p, k_o, k_i')
    tf18 = TransferFunction((k_p + k_o*s + k_i/s), s**2 + 2*omega_o*s + omega_o**2, s)
    # 断言 tf18 的分子部分与预期的值相等
    assert tf18.num == k_i/s + k_o*s + k_p
    # 断言 tf18 的参数与预期的值相等
    assert tf18.args == (k_i/s + k_o*s + k_p, omega_o**2 + 2*omega_o*s + s**2, s)

    # 当分母为零时，引发 ValueError 异常
    raises(ValueError, lambda: TransferFunction(4, 0, s))
    raises(ValueError, lambda: TransferFunction(s, 0, s))
    raises(ValueError, lambda: TransferFunction(0, 0, s))

    # 当传递函数对象参数类型不符合预期时，引发 TypeError 异常
    raises(TypeError, lambda: TransferFunction(Matrix([1, 2, 3]), s, s))
    raises(TypeError, lambda: TransferFunction(s**2 + 2*s - 1, s + 3, 3))
    raises(TypeError, lambda: TransferFunction(p + 1, 5 - p, 4))
    raises(TypeError, lambda: TransferFunction(3, 4, 8))
# 定义一个名为 test_TransferFunction_functions 的函数，用于测试 TransferFunction 类的各种方法和功能
def test_TransferFunction_functions():
    # 创建一个表示 0 的表达式，乘以 s 的倒数，并禁用评估
    expr_1 = Mul(0, Pow(s, -1, evaluate=False), evaluate=False)
    # 创建一个除以 0 的表达式，预期会引发异常
    expr_2 = s/0
    # 创建一个包含符号 p 和 s 的复杂有理表达式
    expr_3 = (p*s**2 + 5*s)/(s + 1)**3
    # 创建一个简单的常数表达式
    expr_4 = 6
    # 创建一个包含符号 s 的复杂有理表达式
    expr_5 = ((2 + 3*s)*(5 + 2*s))/((9 + 3*s)*(5 + 2*s**2))
    # 创建一个包含符号 s 的复杂有理表达式
    expr_6 = (9*s**4 + 4*s**2 + 8)/((s + 1)*(s + 9))
    # 创建一个 TransferFunction 对象 tf，使用 s + 1 作为分子，s**2 + 2 作为分母，s 作为变量
    tf = TransferFunction(s + 1, s**2 + 2, s)
    # 创建一个具有 exp(-s/tau) 延迟的表达式 delay
    delay = exp(-s/tau)
    # 创建一个表达式，由 delay 乘以 tf.to_expr() 组成
    expr_7 = delay*tf.to_expr()
    # 使用 from_rational_expression 方法创建一个 TransferFunction 对象 H1，参数为 expr_7 和 s
    H1 = TransferFunction.from_rational_expression(expr_7, s)
    # 创建一个 TransferFunction 对象 H2，分子为 s + 1，分母为 (s**2 + 2)*exp(s/tau)，s 作为变量
    H2 = TransferFunction(s + 1, (s**2 + 2)*exp(s/tau), s)
    # 创建一个复杂的有理表达式，包含符号 s
    expr_8 = Add(2,  3*s/(s**2 + 1), evaluate=False)

    # 使用 assert 断言，验证 from_rational_expression 方法的输出是否符合预期
    assert TransferFunction.from_rational_expression(expr_1) == TransferFunction(0, s, s)
    # 使用 raises 函数断言，验证 from_rational_expression 方法在遇到除以零时是否会抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: TransferFunction.from_rational_expression(expr_2))
    # 使用 raises 函数断言，验证 from_rational_expression 方法在遇到无法解析的表达式时是否会抛出 ValueError 异常
    raises(ValueError, lambda: TransferFunction.from_rational_expression(expr_3))
    # 使用 assert 断言，验证 from_rational_expression 方法的输出是否符合预期，可指定变量 s
    assert TransferFunction.from_rational_expression(expr_3, s) == TransferFunction((p*s**2 + 5*s), (s + 1)**3, s)
    # 使用 assert 断言，验证 from_rational_expression 方法的输出是否符合预期，可指定变量 p
    assert TransferFunction.from_rational_expression(expr_3, p) == TransferFunction((p*s**2 + 5*s), (s + 1)**3, p)
    # 使用 raises 函数断言，验证 from_rational_expression 方法在遇到无法解析的表达式时是否会抛出 ValueError 异常
    raises(ValueError, lambda: TransferFunction.from_rational_expression(expr_4))
    # 使用 assert 断言，验证 from_rational_expression 方法的输出是否符合预期，可指定变量 s
    assert TransferFunction.from_rational_expression(expr_4, s) == TransferFunction(6, 1, s)
    # 使用 assert 断言，验证 from_rational_expression 方法的输出是否符合预期，可指定变量 s
    assert TransferFunction.from_rational_expression(expr_5, s) == \
        TransferFunction((2 + 3*s)*(5 + 2*s), (9 + 3*s)*(5 + 2*s**2), s)
    # 使用 assert 断言，验证 from_rational_expression 方法的输出是否符合预期，可指定变量 s
    assert TransferFunction.from_rational_expression(expr_6, s) == \
        TransferFunction((9*s**4 + 4*s**2 + 8), (s + 1)*(s + 9), s)
    # 使用 assert 断言，验证 H1 和 H2 对象是否相等
    assert H1 == H2
    # 使用 assert 断言，验证 from_rational_expression 方法的输出是否符合预期，可指定变量 s
    assert TransferFunction.from_rational_expression(expr_8, s) == \
        TransferFunction(2*s**2 + 3*s + 2, s**2 + 1, s)

    # 使用 from_coeff_lists 方法创建 TransferFunction 对象 tf1，分子系数为 [1, 2]，分母系数为 [3, 4, 5]，s 作为变量
    tf1 = TransferFunction.from_coeff_lists([1, 2], [3, 4, 5], s)
    # 创建一个分子系数列表 num2，分母系数列表 den2
    num2 = [p**2, 2*p]
    den2 = [p**3, p + 1, 4]
    # 使用 from_coeff_lists 方法创建 TransferFunction 对象 tf2，分子系数为 num2，分母系数为 den2，s 作为变量
    tf2 = TransferFunction.from_coeff_lists(num2, den2, s)
    # 创建一个分子系数列表 num3，分母系数列表 den3
    num3 = [1, 2, 3]
    den3 = [0, 0]

    # 使用 assert 断言，验证 tf1 对象的输出是否符合预期
    assert tf1 == TransferFunction(s + 2, 3*s**2 + 4*s + 5, s)
    # 使用 assert 断言，验证 tf2 对象的输出是否符合预期
    assert tf2 == TransferFunction(p**2*s + 2*p, p**3*s**2 + s*(p + 1) + 4, s)
    # 使用 raises 函数断言，验证 from_coeff_lists 方法在遇到分母系数为零时是否会抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: TransferFunction.from_coeff_lists(num3, den3, s))

    # 使用 from_zpk 方法创建 TransferFunction 对象 tf1，给定零点列表 zeros、极点列表 poles、增益 gain，s 作为变量
    zeros = [4]
    poles = [-1+2j, -1-2j]
    gain = 3
    tf1 = TransferFunction.from_zpk(zeros, poles, gain, s)

    # 使用 assert 断言，验证 tf1 对象的输出是否符合预期
    assert tf1 == TransferFunction(3*s - 12, (s + 1.0 - 2.0*I)*(s + 1.0 + 2.0*I), s)

    # 显式取消极点和零点。
    # 创建一个 TransferFunction 对象 tf0，分子为 s**5 + s**3 + s，分母为 s - s**2，s 作为变量
    tf0 = TransferFunction(s**5 + s**3 + s, s - s**2, s)
    # 创建一个 TransferFunction 对象 a，分子为 -(s**4 + s**2 + 1)，分母为 s - 1，s 作为变量
    a = TransferFunction(-(s**4 + s**2 + 1), s - 1, s)
    # 使用 assert 断言，验证 tf0 的简化结果是否等于 a
    assert tf0.simplify() == simplify(tf0) == a

    # 创建一个 TransferFunction 对象 tf1，分子为 (p + 3)*(p - 1)，分母为 (p - 1)*(p + 5)，p 作为变量
    tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
    # 创建一个 TransferFunction 对象 b，分子为 p + 3，分母为 p + 5，p 作为变量
    b = TransferFunction(p +
    d = (b0*s**s + b1*p**s)*(b2*s*p + p**p)
    e = a0*p**p*p**s + a0*p**p*s**p + a1*p**s*s**s + a1*s**p*s**s + a2*p**s*s**p + a2*s**(2*p)
    f = b0*b2*p*s*s**s + b0*p**p*s**s + b1*b2*p*p**s*s + b1*p**p*p**s
    g = a1*a2*s*s**p + a1*p*s + a2*b1*p*s*s**p + b1*p**2*s

    # 定义多个变量d, e, f, g，使用给定的数学表达式计算它们的值

    G3 = TransferFunction(c, d, s)
    G4 = TransferFunction(a0*s**s - b0*p**p, (a1*s + b1*s*p)*(a2*s**p + p), p)

    # 创建两个传递函数对象G3和G4，使用变量c, d, a0, b0, a1, b1, a2, p, s来初始化这些对象

    assert G1.expand() == TransferFunction(s**2 - 2*s + 1, s**4 + 2*s**2 + 1, s)
    assert tf1.expand() == TransferFunction(p**2 + 2*p - 3, p**2 + 4*p - 5, p)
    assert G2.expand() == G2
    assert G3.expand() == TransferFunction(e, f, s)
    assert G4.expand() == TransferFunction(a0*s**s - b0*p**p, g, p)

    # 执行多个断言，验证传递函数对象的展开结果与预期的传递函数对象是否相等

    # purely symbolic polynomials.
    p1 = a1*s + a0
    p2 = b2*s**2 + b1*s + b0
    SP1 = TransferFunction(p1, p2, s)
    expect1 = TransferFunction(2.0*s + 1.0, 5.0*s**2 + 4.0*s + 3.0, s)
    expect1_ = TransferFunction(2*s + 1, 5*s**2 + 4*s + 3, s)

    # 创建两个多项式p1和p2，将它们作为参数创建一个传递函数对象SP1，并定义两个预期的传递函数对象expect1和expect1_

    assert SP1.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}) == expect1_
    assert SP1.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}).evalf() == expect1
    assert expect1_.evalf() == expect1

    # 执行多个断言，验证符号多项式的替换、数值化及其与预期结果的比较

    c1, d0, d1, d2 = symbols('c1, d0:3')
    p3, p4 = c1*p, d2*p**3 + d1*p**2 - d0
    SP2 = TransferFunction(p3, p4, p)
    expect2 = TransferFunction(2.0*p, 5.0*p**3 + 2.0*p**2 - 3.0, p)
    expect2_ = TransferFunction(2*p, 5*p**3 + 2*p**2 - 3, p)

    # 定义多个符号变量c1, d0, d1, d2，并使用它们创建两个多项式p3和p4，将它们作为参数创建传递函数对象SP2，并定义两个预期的传递函数对象expect2和expect2_

    assert SP2.subs({c1: 2, d0: 3, d1: 2, d2: 5}) == expect2_
    assert SP2.subs({c1: 2, d0: 3, d1: 2, d2: 5}).evalf() == expect2
    assert expect2_.evalf() == expect2

    # 执行多个断言，验证符号多项式的替换、数值化及其与预期结果的比较

    SP3 = TransferFunction(a0*p**3 + a1*s**2 - b0*s + b1, a1*s + p, s)
    expect3 = TransferFunction(2.0*p**3 + 4.0*s**2 - s + 5.0, p + 4.0*s, s)
    expect3_ = TransferFunction(2*p**3 + 4*s**2 - s + 5, p + 4*s, s)

    # 创建一个复杂的传递函数对象SP3，定义一个预期的传递函数对象expect3和expect3_

    assert SP3.subs({a0: 2, a1: 4, b0: 1, b1: 5}) == expect3_
    assert SP3.subs({a0: 2, a1: 4, b0: 1, b1: 5}).evalf() == expect3
    assert expect3_.evalf() == expect3

    # 执行多个断言，验证符号多项式的替换、数值化及其与预期结果的比较

    SP4 = TransferFunction(s - a1*p**3, a0*s + p, p)
    expect4 = TransferFunction(7.0*p**3 + s, p - s, p)
    expect4_ = TransferFunction(7*p**3 + s, p - s, p)

    # 创建另一个复杂的传递函数对象SP4，定义一个预期的传递函数对象expect4和expect4_

    assert SP4.subs({a0: -1, a1: -7}) == expect4_
    assert SP4.subs({a0: -1, a1: -7}).evalf() == expect4
    assert expect4_.evalf() == expect4

    # 执行多个断言，验证符号多项式的替换、数值化及其与预期结果的比较

    # evaluate the transfer function at particular frequencies.
    assert tf1.eval_frequency(wn) == wn**2/(wn**2 + 4*wn - 5) + 2*wn/(wn**2 + 4*wn - 5) - 3/(wn**2 + 4*wn - 5)
    assert G1.eval_frequency(1 + I) == S(3)/25 + S(4)*I/25
    assert G4.eval_frequency(S(5)/3) == \
        a0*s**s/(a1*a2*s**(S(8)/3) + S(5)*a1*s/3 + 5*a2*b1*s**(S(8)/3)/3 + S(25)*b1*s/9) - 5*3**(S(1)/3)*5**(S(2)/3)*b0/(9*a1*a2*s**(S(8)/3) + 15*a1*s + 15*a2*b1*s**(S(8)/3) + 25*b1*s)

    # 执行多个断言，验证传递函数对象在特定频率下的评估结果

    # Low-frequency (or DC) gain.
    assert tf0.dc_gain() == 1
    assert tf1.dc_gain() == Rational(3, 5)
    assert SP2.dc_gain() == 0
    assert expect4.dc_gain() == -1
    assert expect2_.dc_gain() == 0
    assert TransferFunction(1, s, s).dc_gain() == oo

    # 执行多个断言，验证传递函数对象在低频（或直流）增益的计算结果
    # 定义传递函数的极点。
    tf_ = TransferFunction(x**3 - k, k, x)
    _tf = TransferFunction(k, x**4 - k, x)
    TF_ = TransferFunction(x**2, x**10 + x + x**2, x)
    _TF = TransferFunction(x**10 + x + x**2, x**2, x)
    assert G1.poles() == [I, I, -I, -I]  # 检查传递函数 G1 的极点是否符合预期
    assert G2.poles() == []  # 检查传递函数 G2 是否没有极点
    assert tf1.poles() == [-5, 1]  # 检查传递函数 tf1 的极点是否为 -5 和 1
    assert expect4_.poles() == [s]  # 检查传递函数 expect4_ 的极点是否为 [s]
    assert SP4.poles() == [-a0*s]  # 检查传递函数 SP4 的极点是否为 [-a0*s]
    assert expect3.poles() == [-0.25*p]  # 检查传递函数 expect3 的极点是否为 [-0.25*p]
    assert str(expect2.poles()) == str([0.729001428685125, -0.564500714342563 - 0.710198984796332*I, -0.564500714342563 + 0.710198984796332*I])  # 检查传递函数 expect2 的极点是否符合预期
    assert str(expect1.poles()) == str([-0.4 - 0.66332495807108*I, -0.4 + 0.66332495807108*I])  # 检查传递函数 expect1 的极点是否符合预期
    assert _tf.poles() == [k**(Rational(1, 4)), -k**(Rational(1, 4)), I*k**(Rational(1, 4)), -I*k**(Rational(1, 4))]  # 检查传递函数 _tf 的极点是否符合预期
    assert TF_.poles() == [CRootOf(x**9 + x + 1, 0), 0, CRootOf(x**9 + x + 1, 1), CRootOf(x**9 + x + 1, 2),
        CRootOf(x**9 + x + 1, 3), CRootOf(x**9 + x + 1, 4), CRootOf(x**9 + x + 1, 5), CRootOf(x**9 + x + 1, 6),
        CRootOf(x**9 + x + 1, 7), CRootOf(x**9 + x + 1, 8)]  # 检查传递函数 TF_ 的极点是否符合预期
    raises(NotImplementedError, lambda: TransferFunction(x**2, a0*x**10 + x + x**2, x).poles())  # 检查抛出 NotImplementedError 异常是否符合预期

    # 检查传递函数的稳定性。
    q, r = symbols('q, r', negative=True)
    t = symbols('t', positive=True)
    TF_ = TransferFunction(s**2 + a0 - a1*p, q*s - r, s)
    stable_tf = TransferFunction(s**2 + a0 - a1*p, q*s - 1, s)
    stable_tf_ = TransferFunction(s**2 + a0 - a1*p, q*s - t, s)

    assert G1.is_stable() is False  # 检查传递函数 G1 是否不稳定
    assert G2.is_stable() is True  # 检查传递函数 G2 是否稳定
    assert tf1.is_stable() is False   # 检查传递函数 tf1 是否不稳定，因为一个极点是正数，另一个是负数。
    assert expect2.is_stable() is False  # 检查传递函数 expect2 是否不稳定
    assert expect1.is_stable() is True  # 检查传递函数 expect1 是否稳定
    assert stable_tf.is_stable() is True  # 检查传递函数 stable_tf 是否稳定
    assert stable_tf_.is_stable() is True  # 检查传递函数 stable_tf_ 是否稳定
    assert TF_.is_stable() is False  # 检查传递函数 TF_ 是否不稳定
    assert expect4_.is_stable() is None   # 检查传递函数 expect4_ 的稳定性，因为对单个极点 's' 没有提供假设。
    assert SP4.is_stable() is None  # 检查传递函数 SP4 的稳定性，因为没有提供假设。

    # 定义传递函数的零点。
    assert G1.zeros() == [1, 1]  # 检查传递函数 G1 的零点是否为 [1, 1]
    assert G2.zeros() == []  # 检查传递函数 G2 是否没有零点
    assert tf1.zeros() == [-3, 1]  # 检查传递函数 tf1 的零点是否为 [-3, 1]
    assert expect4_.zeros() == [7**(Rational(2, 3))*(-s)**(Rational(1, 3))/7, -7**(Rational(2, 3))*(-s)**(Rational(1, 3))/14 -
        sqrt(3)*7**(Rational(2, 3))*I*(-s)**(Rational(1, 3))/14, -7**(Rational(2, 3))*(-s)**(Rational(1, 3))/14 + sqrt(3)*7**(Rational(2, 3))*I*(-s)**(Rational(1, 3))/14]  # 检查传递函数 expect4_ 的零点是否符合预期
    assert SP4.zeros() == [(s/a1)**(Rational(1, 3)), -(s/a1)**(Rational(1, 3))/2 - sqrt(3)*I*(s/a1)**(Rational(1, 3))/2,
        -(s/a1)**(Rational(1, 3))/2 + sqrt(3)*I*(s/a1)**(Rational(1, 3))/2]  # 检查传递函数 SP4 的零点是否符合预期
    assert str(expect3.zeros()) == str([0.125 - 1.11102430216445*sqrt(-0.405063291139241*p**3 - 1.0),
        1.11102430216445*sqrt(-0.405063291139241*p**3 - 1.0) + 0.125])  # 检查传递函数 expect3 的零点是否符合预期
    assert tf_.zeros() == [k**(Rational(1, 3)), -k**(Rational(1, 3))/2 - sqrt(3)*I*k**(Rational(1, 3))/2,
        -k**(Rational(1, 3))/2 + sqrt(3)*I*k**(Rational(1, 3))/2]  # 检查传递函数 tf_ 的零点是否符合预期
    # 断言 _TF.zeros() 的返回值是否符合预期列表
    assert _TF.zeros() == [CRootOf(x**9 + x + 1, 0), 0, CRootOf(x**9 + x + 1, 1), CRootOf(x**9 + x + 1, 2),
        CRootOf(x**9 + x + 1, 3), CRootOf(x**9 + x + 1, 4), CRootOf(x**9 + x + 1, 5), CRootOf(x**9 + x + 1, 6),
        CRootOf(x**9 + x + 1, 7), CRootOf(x**9 + x + 1, 8)]
    
    # 使用 lambda 表达式调用 TransferFunction 对象的 zeros 方法，断言抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: TransferFunction(a0*x**10 + x + x**2, x**2, x).zeros())

    # 对 TransferFunction 对象 tf2 进行负运算
    tf2 = TransferFunction(s + 3, s**2 - s**3 + 9, s)
    tf3 = TransferFunction(-3*p + 3, 1 - p, p)
    assert -tf2 == TransferFunction(-s - 3, s**2 - s**3 + 9, s)
    assert -tf3 == TransferFunction(3*p - 3, 1 - p, p)

    # 对 TransferFunction 对象进行幂运算
    tf4 = TransferFunction(p + 4, p - 3, p)
    tf5 = TransferFunction(s**2 + 1, 1 - s, s)
    expect2 = TransferFunction((s**2 + 1)**3, (1 - s)**3, s)
    expect1 = TransferFunction((p + 4)**2, (p - 3)**2, p)
    assert (tf4*tf4).doit() == tf4**2 == pow(tf4, 2) == expect1
    assert (tf5*tf5*tf5).doit() == tf5**3 == pow(tf5, 3) == expect2
    assert tf5**0 == pow(tf5, 0) == TransferFunction(1, 1, s)
    assert Series(tf4).doit()**-1 == tf4**-1 == pow(tf4, -1) == TransferFunction(p - 3, p + 4, p)
    assert (tf5*tf5).doit()**-1 == tf5**-2 == pow(tf5, -2) == TransferFunction((1 - s)**2, (s**2 + 1)**2, s)

    # 断言传递给 TransferFunction 对象的非法幂运算将抛出 ValueError 异常
    raises(ValueError, lambda: tf4**(s**2 + s - 1))
    raises(ValueError, lambda: tf5**s)
    raises(ValueError, lambda: tf4**tf5)

    # SymPy 内置函数的应用
    tf = TransferFunction(s - 1, s**2 - 2*s + 1, s)
    tf6 = TransferFunction(s + p, p**2 - 5, s)
    assert factor(tf) == TransferFunction(s - 1, (s - 1)**2, s)
    assert tf.num.subs(s, 2) == tf.den.subs(s, 2) == 1

    # subs 和 xreplace 方法的使用
    assert tf.subs(s, 2) == TransferFunction(s - 1, s**2 - 2*s + 1, s)
    assert tf6.subs(p, 3) == TransferFunction(s + 3, 4, s)
    assert tf3.xreplace({p: s}) == TransferFunction(-3*s + 3, 1 - s, s)
    raises(TypeError, lambda: tf3.xreplace({p: exp(2)}))
    assert tf3.subs(p, exp(2)) == tf3

    # 使用 xreplace 方法和 subs 方法替换 TransferFunction 对象中的符号变量 s 和 p
    tf7 = TransferFunction(a0*s**p + a1*p**s, a2*p - s, s)
    assert tf7.xreplace({s: k}) == TransferFunction(a0*k**p + a1*p**k, a2*p - k, k)
    assert tf7.subs(s, k) == TransferFunction(a0*s**p + a1*p**s, a2*p - s, s)

    # 转换为 Expr 类型使用 to_expr() 方法
    tf8 = TransferFunction(a0*s**5 + 5*s**2 + 3, s**6 - 3, s)
    tf9 = TransferFunction((5 + s), (5 + s)*(6 + s), s)
    tf10 = TransferFunction(0, 1, s)
    tf11 = TransferFunction(1, 1, s)
    assert tf8.to_expr() == Mul((a0*s**5 + 5*s**2 + 3), Pow((s**6 - 3), -1, evaluate=False), evaluate=False)
    assert tf9.to_expr() == Mul((s + 5), Pow((5 + s)*(6 + s), -1, evaluate=False), evaluate=False)
    assert tf10.to_expr() == Mul(S(0), Pow(1, -1, evaluate=False), evaluate=False)
    assert tf11.to_expr() == Pow(1, -1, evaluate=False)
# 定义测试函数，用于测试 TransferFunction 类的加法和减法功能
def test_TransferFunction_addition_and_subtraction():
    # 创建四个 TransferFunction 对象，分别用不同的参数初始化
    tf1 = TransferFunction(s + 6, s - 5, s)
    tf2 = TransferFunction(s + 3, s + 1, s)
    tf3 = TransferFunction(s + 1, s**2 + s + 1, s)
    tf4 = TransferFunction(p, 2 - p, p)

    # 加法操作的断言测试
    assert tf1 + tf2 == Parallel(tf1, tf2)
    assert tf3 + tf1 == Parallel(tf3, tf1)
    assert -tf1 + tf2 + tf3 == Parallel(-tf1, tf2, tf3)
    assert tf1 + (tf2 + tf3) == Parallel(tf1, tf2, tf3)

    # 创建一个非交换符号 c
    c = symbols("c", commutative=False)
    # 预期抛出异常的加法操作
    raises(ValueError, lambda: tf1 + Matrix([1, 2, 3]))
    raises(ValueError, lambda: tf2 + c)
    raises(ValueError, lambda: tf3 + tf4)
    raises(ValueError, lambda: tf1 + (s - 1))
    raises(ValueError, lambda: tf1 + 8)
    raises(ValueError, lambda: (1 - p**3) + tf1)

    # 减法操作的断言测试
    assert tf1 - tf2 == Parallel(tf1, -tf2)
    assert tf3 - tf2 == Parallel(tf3, -tf2)
    assert -tf1 - tf3 == Parallel(-tf1, -tf3)
    assert tf1 - tf2 + tf3 == Parallel(tf1, -tf2, tf3)

    # 预期抛出异常的减法操作
    raises(ValueError, lambda: tf1 - Matrix([1, 2, 3]))
    raises(ValueError, lambda: tf3 - tf4)
    raises(ValueError, lambda: tf1 - (s - 1))
    raises(ValueError, lambda: tf1 - 8)
    raises(ValueError, lambda: (s + 5) - tf2)
    raises(ValueError, lambda: (1 + p**4) - tf1)


# 定义测试函数，用于测试 TransferFunction 类的乘法和除法功能
def test_TransferFunction_multiplication_and_division():
    # 创建多个 TransferFunction 对象，分别用不同的参数初始化
    G1 = TransferFunction(s + 3, -s**3 + 9, s)
    G2 = TransferFunction(s + 1, s - 5, s)
    G3 = TransferFunction(p, p**4 - 6, p)
    G4 = TransferFunction(p + 4, p - 5, p)
    G5 = TransferFunction(s + 6, s - 5, s)
    G6 = TransferFunction(s + 3, s + 1, s)
    G7 = TransferFunction(1, 1, s)

    # 乘法操作的断言测试
    assert G1 * G2 == Series(G1, G2)
    assert -G1 * G5 == Series(-G1, G5)
    assert -G2 * G5 * -G6 == Series(-G2, G5, -G6)
    assert -G1 * -G2 * -G5 * -G6 == Series(-G1, -G2, -G5, -G6)
    assert G3 * G4 == Series(G3, G4)
    assert (G1 * G2) * -(G5 * G6) == Series(G1, G2, TransferFunction(-1, 1, s), Series(G5, G6))
    assert G1 * G2 * (G5 + G6) == Series(G1, G2, Parallel(G5, G6))

    # 除法操作的断言测试 - 有关除以 Parallel 对象的测试请参阅 ``test_Feedback_functions()``
    assert G5 / G6 == Series(G5, pow(G6, -1))
    assert -G3 / G4 == Series(-G3, pow(G4, -1))
    assert (G5 * G6) / G7 == Series(G5, G6, pow(G7, -1))

    # 创建一个非交换符号 c
    c = symbols("c", commutative=False)
    # 预期抛出异常的乘法操作
    raises(ValueError, lambda: G3 * Matrix([1, 2, 3]))
    raises(ValueError, lambda: G1 * c)
    raises(ValueError, lambda: G3 * G5)
    raises(ValueError, lambda: G5 * (s - 1))
    raises(ValueError, lambda: 9 * G5)

    # 预期抛出异常的除法操作
    raises(ValueError, lambda: G3 / Matrix([1, 2, 3]))
    raises(ValueError, lambda: G6 / 0)
    raises(ValueError, lambda: G3 / G5)
    raises(ValueError, lambda: G5 / 2)
    raises(ValueError, lambda: G5 / s**2)
    raises(ValueError, lambda: (s - 4 * s**2) / G2)
    raises(ValueError, lambda: 0 / G4)
    raises(ValueError, lambda: G7 / (1 + G6))
    raises(ValueError, lambda: G7 / (G5 * G6))
    raises(ValueError, lambda: G7 / (G7 + (G5 + G6)))
    # 使用 sympy 库中的 symbols 函数创建符号变量 omega_o, zeta, tau
    omega_o, zeta, tau = symbols('omega_o, zeta, tau')
    
    # 创建传递函数 G1，使用 TransferFunction 类，参数分别为：
    # - 分子为 omega_o**2
    # - 分母为 s**2 + p*omega_o*zeta*s + omega_o**2，其中 s 是符号变量，omega_o 是给定参数
    # - 参数 omega_o 用于指定传递函数的特定频率
    G1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    
    # 创建传递函数 G2，使用 TransferFunction 类，参数分别为：
    # - 分子为 tau - s**3
    # - 分母为 tau + p**4，其中 tau 是符号变量，p 是给定参数
    G2 = TransferFunction(tau - s**3, tau + p**4, tau)
    
    # 创建传递函数 G3，使用 TransferFunction 类，参数分别为：
    # - 分子为 a*b*s**3 + s**2 - a*p + s
    # - 分母为 b - s*p**2，其中 a, b, p 是给定参数，s 是符号变量
    G3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    
    # 创建传递函数 G4，使用 TransferFunction 类，参数分别为：
    # - 分子为 b*s**2 + p**2 - a*p + s
    # - 分母为 b - p**2，其中 a, b, p 是给定参数，s 是符号变量
    G4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    
    # 断言 G1 是严格的传递函数（proper transfer function）
    assert G1.is_proper
    
    # 断言 G2 是严格的传递函数（proper transfer function）
    assert G2.is_proper
    
    # 断言 G3 是严格的传递函数（proper transfer function）
    assert G3.is_proper
    
    # 断言 G4 不是严格的传递函数（proper transfer function）
    assert not G4.is_proper
def test_TransferFunction_is_strictly_proper():
    # 定义符号变量 omega_o, zeta, tau
    omega_o, zeta, tau = symbols('omega_o, zeta, tau')
    # 创建 TransferFunction 对象 tf1，表示一个传递函数
    tf1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    # 创建 TransferFunction 对象 tf2，表示另一个传递函数
    tf2 = TransferFunction(tau - s**3, tau + p**4, tau)
    # 创建 TransferFunction 对象 tf3，表示另一个传递函数
    tf3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    # 创建 TransferFunction 对象 tf4，表示另一个传递函数
    tf4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    # 断言 tf1 不是严格适当的传递函数
    assert not tf1.is_strictly_proper
    # 断言 tf2 不是严格适当的传递函数
    assert not tf2.is_strictly_proper
    # 断言 tf3 是严格适当的传递函数
    assert tf3.is_strictly_proper
    # 断言 tf4 不是严格适当的传递函数


def test_TransferFunction_is_biproper():
    # 定义符号变量 tau, omega_o, zeta
    tau, omega_o, zeta = symbols('tau, omega_o, zeta')
    # 创建 TransferFunction 对象 tf1，表示一个传递函数
    tf1 = TransferFunction(omega_o**2, s**2 + p*omega_o*zeta*s + omega_o**2, omega_o)
    # 创建 TransferFunction 对象 tf2，表示另一个传递函数
    tf2 = TransferFunction(tau - s**3, tau + p**4, tau)
    # 创建 TransferFunction 对象 tf3，表示另一个传递函数
    tf3 = TransferFunction(a*b*s**3 + s**2 - a*p + s, b - s*p**2, p)
    # 创建 TransferFunction 对象 tf4，表示另一个传递函数
    tf4 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
    # 断言 tf1 是双适当的传递函数
    assert tf1.is_biproper
    # 断言 tf2 是双适当的传递函数
    assert tf2.is_biproper
    # 断言 tf3 不是双适当的传递函数
    assert not tf3.is_biproper
    # 断言 tf4 不是双适当的传递函数


def test_Series_construction():
    # 创建 TransferFunction 对象 tf
    tf = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    # 创建 TransferFunction 对象 tf2
    tf2 = TransferFunction(a2*p - s, a2*s + p, s)
    # 创建 TransferFunction 对象 tf3
    tf3 = TransferFunction(a0*p + p**a1 - s, p, p)
    # 创建 TransferFunction 对象 tf4
    tf4 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    # 创建输入函数符号对象 inp
    inp = Function('X_d')(s)
    # 创建输出函数符号对象 out
    out = Function('X')(s)

    # 创建 Series 对象 s0，由 tf 和 tf2 组成
    s0 = Series(tf, tf2)
    assert s0.args == (tf, tf2)
    assert s0.var == s

    # 创建 Series 对象 s1，由 Parallel(tf, -tf2) 和 tf2 组成
    s1 = Series(Parallel(tf, -tf2), tf2)
    assert s1.args == (Parallel(tf, -tf2), tf2)
    assert s1.var == s

    # 创建 TransferFunction 对象 tf3_
    tf3_ = TransferFunction(inp, 1, s)
    # 创建 TransferFunction 对象 tf4_
    tf4_ = TransferFunction(-out, 1, s)
    # 创建 Series 对象 s2，由 tf, Parallel(tf3_, tf4_), tf2 组成
    s2 = Series(tf, Parallel(tf3_, tf4_), tf2)
    assert s2.args == (tf, Parallel(tf3_, tf4_), tf2)

    # 创建 Series 对象 s3，由 tf, tf2, tf4 组成
    s3 = Series(tf, tf2, tf4)
    assert s3.args == (tf, tf2, tf4)

    # 创建 Series 对象 s4，由 tf3_, tf4_ 组成
    s4 = Series(tf3_, tf4_)
    assert s4.args == (tf3_, tf4_)
    assert s4.var == s

    # 创建 Series 对象 s6，由 tf2, tf4, Parallel(tf2, -tf), tf4 组成
    s6 = Series(tf2, tf4, Parallel(tf2, -tf), tf4)
    assert s6.args == (tf2, tf4, Parallel(tf2, -tf), tf4)

    # 断言 s0 等于 s7
    s7 = Series(tf, tf2)
    assert s0 == s7
    # 断言 s0 不等于 s2
    assert not s0 == s2

    # 断言以下操作引发特定异常
    raises(ValueError, lambda: Series(tf, tf3))
    raises(ValueError, lambda: Series(tf, tf2, tf3, tf4))
    raises(ValueError, lambda: Series(-tf3, tf2))
    raises(TypeError, lambda: Series(2, tf, tf4))
    raises(TypeError, lambda: Series(s**2 + p*s, tf3, tf2))
    raises(TypeError, lambda: Series(tf3, Matrix([1, 2, 3, 4])))


def test_MIMOSeries_construction():
    # 创建 TransferFunction 对象 tf_1
    tf_1 = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    # 创建 TransferFunction 对象 tf_2
    tf_2 = TransferFunction(a2*p - s, a2*s + p, s)
    # 创建 TransferFunction 对象 tf_3
    tf_3 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)

    # 创建 TransferFunctionMatrix 对象 tfm_1，包含多个传递函数对象
    tfm_1 = TransferFunctionMatrix([[tf_1, tf_2, tf_3], [-tf_3, -tf_2, tf_1]])
    # 创建 TransferFunctionMatrix 对象 tfm_2，包含多个传递函数对象
    tfm_2 = TransferFunctionMatrix([[-tf_2], [-tf_2], [-tf_3]])
    # 创建 TransferFunctionMatrix 对象 tfm_3，包含多个传递函数对象
    tfm_3 = TransferFunctionMatrix([[-tf_3]])
    # 创建 TransferFunctionMatrix 对象 tfm_4，包含多个传递函数对象
    tfm_4 = TransferFunctionMatrix([[TF3], [TF2], [-TF1]])
    # 创建 TransferFunctionMatrix 对象 tfm_5，从 Matrix([1/p]) 创建
    tfm_5 = TransferFunctionMatrix.from_Matrix(Matrix([1/p]), p)

    # 创建 MIMOSeries 对象 s8，由 tfm_2, tfm_1 组成
    s8 = MIMOSeries(tfm_2, tfm_1)
    # 断言确保 s8 的参数为 (tfm_2, tfm_1)
    assert s8.args == (tfm_2, tfm_1)
    # 断言确保 s8 的变量为 s
    assert s8.var == s
    # 断言确保 s8 的形状为 (s8.num_outputs, s8.num_inputs)，并且确保其值为 (2, 1)
    assert s8.shape == (s8.num_outputs, s8.num_inputs) == (2, 1)

    # 使用 tfm_3, tfm_2, tfm_1 创建一个 MIMOSeries 对象 s9
    s9 = MIMOSeries(tfm_3, tfm_2, tfm_1)
    # 断言确保 s9 的参数为 (tfm_3, tfm_2, tfm_1)
    assert s9.args == (tfm_3, tfm_2, tfm_1)
    # 断言确保 s9 的变量为 s
    assert s9.var == s
    # 断言确保 s9 的形状为 (s9.num_outputs, s9.num_inputs)，并且确保其值为 (2, 1)
    assert s9.shape == (s9.num_outputs, s9.num_inputs) == (2, 1)

    # 使用 tfm_3, MIMOParallel(-tfm_2, -tfm_4), tfm_1 创建一个 MIMOSeries 对象 s11
    s11 = MIMOSeries(tfm_3, MIMOParallel(-tfm_2, -tfm_4), tfm_1)
    # 断言确保 s11 的参数为 (tfm_3, MIMOParallel(-tfm_2, -tfm_4), tfm_1)
    assert s11.args == (tfm_3, MIMOParallel(-tfm_2, -tfm_4), tfm_1)
    # 断言确保 s11 的形状为 (s11.num_outputs, s11.num_inputs)，并且确保其值为 (2, 1)
    assert s11.shape == (s11.num_outputs, s11.num_inputs) == (2, 1)

    # 引发 ValueError 异常，因为 MIMOSeries 不接受空元组作为参数
    raises(ValueError, lambda: MIMOSeries())

    # 引发 TypeError 异常，因为 MIMOSeries 的参数不能同时包含 SISO 和 MIMO 系统
    raises(TypeError, lambda: MIMOSeries(tfm_1, tf_1))

    # 引发 ValueError 异常，因为相邻的传递函数矩阵的输入输出不匹配
    raises(ValueError, lambda: MIMOSeries(tfm_1, tfm_2, -tfm_1))

    # 引发 ValueError 异常，因为所有传递函数矩阵必须使用相同的复杂变量
    raises(ValueError, lambda: MIMOSeries(tfm_3, tfm_5))

    # 引发 TypeError 异常，因为参数中不允许出现数字或表达式
    raises(TypeError, lambda: MIMOSeries(2, tfm_2, tfm_3))
    raises(TypeError, lambda: MIMOSeries(s**2 + p*s, -tfm_2, tfm_3))
    raises(TypeError, lambda: MIMOSeries(Matrix([1/p]), tfm_3))
`
def test_Series_functions():
    # 创建传输函数 tf1
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    # 创建传输函数 tf2
    tf2 = TransferFunction(k, 1, s)
    # 创建传输函数 tf3
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    # 创建传输函数 tf4
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    # 创建传输函数 tf5
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    # 测试传输函数的乘法，验证其结合律
    assert tf1*tf2*tf3 == Series(tf1, tf2, tf3) == Series(Series(tf1, tf2), tf3) \
        == Series(tf1, Series(tf2, tf3))
    # 测试传输函数的乘法与加法的结合律
    assert tf1*(tf2 + tf3) == Series(tf1, Parallel(tf2, tf3))
    # 测试传输函数的加法与乘法的组合
    assert tf1*tf2 + tf5 == Parallel(Series(tf1, tf2), tf5)
    # 测试传输函数的减法与加法
    assert tf1*tf2 - tf5 == Parallel(Series(tf1, tf2), -tf5)
    # 测试多个传输函数的并联
    assert tf1*tf2 + tf3 + tf5 == Parallel(Series(tf1, tf2), tf3, tf5)
    # 测试多个传输函数的并联与减法
    assert tf1*tf2 - tf3 - tf5 == Parallel(Series(tf1, tf2), -tf3, -tf5)
    # 测试传输函数的加减组合
    assert tf1*tf2 - tf3 + tf5 == Parallel(Series(tf1, tf2), -tf3, tf5)
    # 测试传输函数的加法与乘法组合
    assert tf1*tf2 + tf3*tf5 == Parallel(Series(tf1, tf2), Series(tf3, tf5))
    # 测试传输函数的减法与乘法组合
    assert tf1*tf2 - tf3*tf5 == Parallel(Series(tf1, tf2), Series(TransferFunction(-1, 1, s), Series(tf3, tf5))))
    # 测试多个传输函数的组合，验证其正确性
    assert tf2*tf3*(tf2 - tf1)*tf3 == Series(tf2, tf3, Parallel(tf2, -tf1), tf3)
    # 测试传输函数的负号运算
    assert -tf1*tf2 == Series(-tf1, tf2)
    # 测试传输函数的负号运算，验证其正确性
    assert -(tf1*tf2) == Series(TransferFunction(-1, 1, s), Series(tf1, tf2))
    # 验证乘法运算中抛出 ValueError 的情况
    raises(ValueError, lambda: tf1*tf2*tf4)
    raises(ValueError, lambda: tf1*(tf2 - tf4))
    raises(ValueError, lambda: tf3*Matrix([1, 2, 3]))

    # evaluate=True -> doit()
    # 测试将传输函数进行级联合并后的结果
    assert Series(tf1, tf2, evaluate=True) == Series(tf1, tf2).doit() == \
        TransferFunction(k, s**2 + 2*s*wn*zeta + wn**2, s)
    # 测试传输函数的复杂级联与并联运算
    assert Series(tf1, tf2, Parallel(tf1, -tf3), evaluate=True) == Series(tf1, tf2, Parallel(tf1, -tf3)).doit() == \
        TransferFunction(k*(a2*s + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)**2, s)
    # 测试传输函数的级联与负号运算
    assert Series(tf2, tf1, -tf3, evaluate=True) == Series(tf2, tf1, -tf3).doit() == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 验证级联与负号运算的结果是否相等
    assert not Series(tf1, -tf2, evaluate=False) == Series(tf1, -tf2).doit()

    # 测试传输函数的并联运算与级联运算
    assert Series(Parallel(tf1, tf2), Parallel(tf2, -tf3)).doit() == \
        TransferFunction((k*(s**2 + 2*s*wn*zeta + wn**2) + 1)*(-a2*p + k*(a2*s + p) + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 测试传输函数的负号运算与级联运算
    assert Series(-tf1, -tf2, -tf3).doit() == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 验证负号传输函数的级联运算
    assert -Series(tf1, tf2, tf3).doit() == \
        TransferFunction(-k*(a2*p - s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 验证传输函数的并联与级联运算
    assert Series(tf2, tf3, Parallel(tf2, -tf1), tf3).doit() == \
        TransferFunction(k*(a2*p - s)**2*(k*(s**2 + 2*s*wn*zeta + wn**2) - 1), (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2), s)

    # 测试将传输函数重写为 TransferFunction 的形式
    assert Series(tf1, tf2).rewrite(TransferFunction) == TransferFunction(k, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Series(tf2, tf1, -tf3).rewrite(TransferFunction) == \
        TransferFunction(k*(-a2*p + s), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    # 创建 Series 对象
    S1 = Series(Parallel(tf1, tf2), Parallel(tf2, -tf3))
    # 断言 S1 是一个 proper 系统
    assert S1.is_proper
    
    # 断言 S1 不是 strictly proper 系统
    assert not S1.is_strictly_proper
    
    # 断言 S1 是一个 biproper 系统
    assert S1.is_biproper
    
    # 创建 Series 对象 S2，包含 tf1, tf2, tf3 作为其参数
    S2 = Series(tf1, tf2, tf3)
    
    # 断言 S2 是一个 proper 系统
    assert S2.is_proper
    
    # 断言 S2 是一个 strictly proper 系统
    assert S2.is_strictly_proper
    
    # 断言 S2 不是一个 biproper 系统
    assert not S2.is_biproper
    
    # 创建 Series 对象 S3，包含 tf1, -tf2, Parallel(tf1, -tf3) 作为其参数
    S3 = Series(tf1, -tf2, Parallel(tf1, -tf3))
    
    # 断言 S3 是一个 proper 系统
    assert S3.is_proper
    
    # 断言 S3 是一个 strictly proper 系统
    assert S3.is_strictly_proper
    
    # 断言 S3 不是一个 biproper 系统
    assert not S3.is_biproper
def test_MIMOSeries_functions():
    # 创建 TransferFunctionMatrix 对象 tfm1，包含两行三列的传递函数
    tfm1 = TransferFunctionMatrix([[TF1, TF2, TF3], [-TF3, -TF2, TF1]])
    # 创建 TransferFunctionMatrix 对象 tfm2，包含三行一列的传递函数
    tfm2 = TransferFunctionMatrix([[-TF1], [-TF2], [-TF3]])
    # 创建 TransferFunctionMatrix 对象 tfm3，包含一行一列的传递函数
    tfm3 = TransferFunctionMatrix([[-TF1]])
    # 创建 TransferFunctionMatrix 对象 tfm4，包含两行两列的传递函数
    tfm4 = TransferFunctionMatrix([[-TF2, -TF3], [-TF1, TF2]])
    # 创建 TransferFunctionMatrix 对象 tfm5，包含两行两列的传递函数
    tfm5 = TransferFunctionMatrix([[TF2, -TF2], [-TF3, -TF2]])
    # 创建 TransferFunctionMatrix 对象 tfm6，包含两行一列的传递函数
    tfm6 = TransferFunctionMatrix([[-TF3], [TF1]])
    # 创建 TransferFunctionMatrix 对象 tfm7，包含两行一列的传递函数
    tfm7 = TransferFunctionMatrix([[TF1], [-TF2]])

    # 断言验证 MIMOParallel 对象的运算结果
    assert tfm1*tfm2 + tfm6 == MIMOParallel(MIMOSeries(tfm2, tfm1), tfm6)
    assert tfm1*tfm2 + tfm7 + tfm6 == MIMOParallel(MIMOSeries(tfm2, tfm1), tfm7, tfm6)
    assert tfm1*tfm2 - tfm6 - tfm7 == MIMOParallel(MIMOSeries(tfm2, tfm1), -tfm6, -tfm7)
    assert tfm4*tfm5 + (tfm4 - tfm5) == MIMOParallel(MIMOSeries(tfm5, tfm4), tfm4, -tfm5)
    assert tfm4*-tfm6 + (-tfm4*tfm6) == MIMOParallel(MIMOSeries(-tfm6, tfm4), MIMOSeries(tfm6, -tfm4))

    # 验证特定异常被正确抛出
    raises(ValueError, lambda: tfm1*tfm2 + TF1)
    raises(TypeError, lambda: tfm1*tfm2 + a0)
    raises(TypeError, lambda: tfm4*tfm6 - (s - 1))
    raises(TypeError, lambda: tfm4*-tfm6 - 8)
    raises(TypeError, lambda: (-1 + p**5) + tfm1*tfm2)

    # 形状标准验证

    raises(TypeError, lambda: -tfm1*tfm2 + tfm4)
    raises(TypeError, lambda: tfm1*tfm2 - tfm4 + tfm5)
    raises(TypeError, lambda: tfm1*tfm2 - tfm4*tfm5)

    # 断言验证 MIMOSeries 对象的运算结果
    assert tfm1*tfm2*-tfm3 == MIMOSeries(-tfm3, tfm2, tfm1)
    assert (tfm1*-tfm2)*tfm3 == MIMOSeries(tfm3, -tfm2, tfm1)

    # 不允许将 Series 对象与 SISO TF 相乘

    raises(ValueError, lambda: tfm4*tfm5*TF1)
    raises(TypeError, lambda: tfm4*tfm5*a1)
    raises(TypeError, lambda: tfm4*-tfm5*(s - 2))
    raises(TypeError, lambda: tfm5*tfm4*9)
    raises(TypeError, lambda: (-p**3 + 1)*tfm5*tfm4)

    # 在参数中使用 Transfer function matrix

    assert (MIMOSeries(tfm2, tfm1, evaluate=True) == MIMOSeries(tfm2, tfm1).doit()
        == TransferFunctionMatrix(((TransferFunction(-k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2)**2 - (a2*s + p)**2,
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2, s),),
        (TransferFunction(k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*p - s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2),
        (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2, s),))))

    # doit() 不应取消极点和零点
    mat_1 = Matrix([[1/(1+s), (1+s)/(1+s**2+2*s)**3]])
    mat_2 = Matrix([[(1+s)], [(1+s**2+2*s)**3/(1+s)]])
    tm_1, tm_2 = TransferFunctionMatrix.from_Matrix(mat_1, s), TransferFunctionMatrix.from_Matrix(mat_2, s)
    assert (MIMOSeries(tm_2, tm_1).doit()
        == TransferFunctionMatrix(((TransferFunction(2*(s + 1)**2*(s**2 + 2*s + 1)**3, (s + 1)**2*(s**2 + 2*s + 1)**3, s),),)))
    assert MIMOSeries(tm_2, tm_1).doit().simplify() == TransferFunctionMatrix(((TransferFunction(2, 1, s),),))

    # 调用 doit() 将扩展内部的 Series 和 Parallel 对象
    # 断言：验证两个 MIMOSeries 对象相等
    assert (MIMOSeries(-tfm3, -tfm2, tfm1, evaluate=True)
        # 调用 doit() 方法，计算 MIMOSeries 对象的值
        == MIMOSeries(-tfm3, -tfm2, tfm1).doit()
        # 验证 TransferFunctionMatrix 对象相等
        == TransferFunctionMatrix((
            # 第一个 TransferFunction 对象
            (TransferFunction(k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (a2*p - s)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (a2*s + p)**2,
                               (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**3, s),),
            # 第二个 TransferFunction 对象
            (TransferFunction(-k**2*(a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**2 + (-a2*p + s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*p - s)*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2),
                               (a2*s + p)**2*(s**2 + 2*s*wn*zeta + wn**2)**3, s),))))
    
    # 断言：验证两个 MIMOSeries 对象相等
    assert (MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5, evaluate=True)
        # 调用 doit() 方法，计算 MIMOSeries 对象的值
        == MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5).doit()
        # 验证 TransferFunctionMatrix 对象相等
        == TransferFunctionMatrix((
            # 第一个元组中的两个 TransferFunction 对象
            (TransferFunction(-k*(-a2*s - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s),
             TransferFunction(k*(-a2*p - k*(a2*s + p) + s), a2*s + p, s)),
            # 第二个元组中的两个 TransferFunction 对象
            (TransferFunction(-k*(-a2*s - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2)), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s),
             TransferFunction((-a2*p + s)*(-a2*p - k*(a2*s + p) + s), (a2*s + p)**2, s)))))
        # 调用 rewrite 方法，重新表示为 TransferFunctionMatrix 形式
        == MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5).rewrite(TransferFunctionMatrix))
# 定义一个测试函数，用于测试并验证 Parallel 类的构建
def test_Parallel_construction():
    # 创建 TransferFunction 对象 tf，表示传递函数
    tf = TransferFunction(a0*s**3 + a1*s**2 - a2*s, b0*p**4 + b1*p**3 - b2*s*p, s)
    # 创建另一个 TransferFunction 对象 tf2
    tf2 = TransferFunction(a2*p - s, a2*s + p, s)
    # 创建第三个 TransferFunction 对象 tf3
    tf3 = TransferFunction(a0*p + p**a1 - s, p, p)
    # 创建第四个 TransferFunction 对象 tf4
    tf4 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    # 创建一个 Function 对象 inp，表示输入函数
    inp = Function('X_d')(s)
    # 创建一个 Function 对象 out，表示输出函数
    out = Function('X')(s)

    # 创建并实例化 Parallel 类对象 p0，包含 tf 和 tf2 两个传递函数对象
    p0 = Parallel(tf, tf2)
    # 断言 p0 的参数是 (tf, tf2)
    assert p0.args == (tf, tf2)
    # 断言 p0 的变量是 s
    assert p0.var == s

    # 创建并实例化 Parallel 类对象 p1，包含 Series 类和 tf2 传递函数对象
    p1 = Parallel(Series(tf, -tf2), tf2)
    # 断言 p1 的参数是 (Series(tf, -tf2), tf2)
    assert p1.args == (Series(tf, -tf2), tf2)
    # 断言 p1 的变量是 s
    assert p1.var == s

    # 创建 tf3_ 和 tf4_ 两个新的 TransferFunction 对象
    tf3_ = TransferFunction(inp, 1, s)
    tf4_ = TransferFunction(-out, 1, s)
    # 创建包含 tf, Series(tf3_, -tf4_), tf2 三个传递函数对象的 Parallel 类对象 p2
    p2 = Parallel(tf, Series(tf3_, -tf4_), tf2)
    # 断言 p2 的参数是 (tf, Series(tf3_, -tf4_), tf2)
    assert p2.args == (tf, Series(tf3_, -tf4_), tf2)

    # 创建并实例化 Parallel 类对象 p3，包含 tf, tf2, tf4 三个传递函数对象
    p3 = Parallel(tf, tf2, tf4)
    # 断言 p3 的参数是 (tf, tf2, tf4)
    assert p3.args == (tf, tf2, tf4)

    # 创建并实例化 Parallel 类对象 p4，包含 tf3_, tf4_ 两个传递函数对象
    p4 = Parallel(tf3_, tf4_)
    # 断言 p4 的参数是 (tf3_, tf4_)
    assert p4.args == (tf3_, tf4_)
    # 断言 p4 的变量是 s
    assert p4.var == s

    # 创建另一个 Parallel 类对象 p5，与 p0 相同
    p5 = Parallel(tf, tf2)
    # 断言 p0 等于 p5
    assert p0 == p5
    # 断言 p0 不等于 p1
    assert not p0 == p1

    # 创建并实例化 Parallel 类对象 p6，包含 tf2, tf4, Series(tf2, -tf4) 三个传递函数对象
    p6 = Parallel(tf2, tf4, Series(tf2, -tf4))
    # 断言 p6 的参数是 (tf2, tf4, Series(tf2, -tf4))
    assert p6.args == (tf2, tf4, Series(tf2, -tf4))

    # 创建并实例化 Parallel 类对象 p7，包含 tf2, tf4, Series(tf2, -tf), tf4 四个传递函数对象
    p7 = Parallel(tf2, tf4, Series(tf2, -tf), tf4)
    # 断言 p7 的参数是 (tf2, tf4, Series(tf2, -tf), tf4)
    assert p7.args == (tf2, tf4, Series(tf2, -tf), tf4)

    # 使用 lambda 函数测试引发 ValueError 异常
    raises(ValueError, lambda: Parallel(tf, tf3))
    raises(ValueError, lambda: Parallel(tf, tf2, tf3, tf4))
    raises(ValueError, lambda: Parallel(-tf3, tf4))
    # 使用 lambda 函数测试引发 TypeError 异常
    raises(TypeError, lambda: Parallel(2, tf, tf4))
    raises(TypeError, lambda: Parallel(s**2 + p*s, tf3, tf2))
    raises(TypeError, lambda: Parallel(tf3, Matrix([1, 2, 3, 4])))


# 定义一个测试函数，用于测试并验证 MIMOParallel 类的构建
def test_MIMOParallel_construction():
    # 创建 TransferFunctionMatrix 对象 tfm1
    tfm1 = TransferFunctionMatrix([[TF1], [TF2], [TF3]])
    # 创建 TransferFunctionMatrix 对象 tfm2
    tfm2 = TransferFunctionMatrix([[-TF3], [TF2], [TF1]])
    # 创建 TransferFunctionMatrix 对象 tfm3
    tfm3 = TransferFunctionMatrix([[TF1]])
    # 创建 TransferFunctionMatrix 对象 tfm4
    tfm4 = TransferFunctionMatrix([[TF2], [TF1], [TF3]])
    # 创建 TransferFunctionMatrix 对象 tfm5
    tfm5 = TransferFunctionMatrix([[TF1, TF2], [TF2, TF1]])
    # 创建 TransferFunctionMatrix 对象 tfm6
    tfm6 = TransferFunctionMatrix([[TF2, TF1], [TF1, TF2]])
    # 创建 TransferFunctionMatrix 对象 tfm7
    tfm7 = TransferFunctionMatrix.from_Matrix(Matrix([[1/p]]), p)

    # 创建并实例化 MIMOParallel 类对象 p8，包含 tfm1, tfm2 两个传递函数矩阵对象
    p8 = MIMOParallel(tfm1, tfm2)
    # 断言 p8 的参数是 (tfm1, tfm2)
    assert p8.args == (tfm1, tfm2)
    # 断言 p8 的变量是 s
    assert p8.var == s
    # 断言 p8 的形状是 (3, 1)，即输出和输入数量都是 3 和 1
    assert p8.shape == (p8.num_outputs, p8.num_inputs) == (3, 1)

    # 创建并实例化 MIMOParallel 类对象 p9，包含 MIMOSeries(tfm3, tfm1), tfm2 两个传递函数矩阵对象
    p9 = MIMOParallel(MIMOSeries(tfm3, tfm1), tfm2)
    # 断言 p9 的参数是 (MIMOSeries(tfm3, tfm1), tfm2)
    assert p9.args == (MIMOSeries(tfm3, tfm1), tfm2)
    # 断言 p9 的变量是 s
    assert p9.var == s
    # 断言 p9 的形状是 (3, 1)，即输出和输入数量都是 3 和 1
    assert p9.shape == (p9.num_outputs, p9.num_inputs) == (3, 1)

    # 创建并实例化 MIMOParallel 类对象 p10，包含 tfm1, MIMOSeries(tfm3, tfm4), tfm2 三个传递函数矩阵对象
    p10 = MIMOParallel(tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    # 断言 p10 的参数是 (tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    assert p10.args == (tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    # 断言 p10 的变量是 s
    assert p10.var == s
    # 断言 p10 的形状是 (3, 1)，即输出和输入数量都是 3 和 1
    assert p10.shape == (p10.num_outputs, p10.num_inputs) == (3, 1)

    # 创建并实例化 MIMOParallel 类对象 p11，包含 tfm2, tfm1, tfm4 三个传递函数矩阵对象
    p11 = MIMOParallel(tfm2, tf
    # 如果参数是空元组，则会引发 TypeError 异常
    raises(TypeError, lambda: MIMOParallel(()))

    # 参数不能同时包含 SISO 和 MIMO 系统，否则会引发 TypeError 异常
    raises(TypeError, lambda: MIMOParallel(tfm1, tfm2, TF1))

    # 所有传递的传递函数矩阵（TFMs）必须具有相同的形状，否则会引发 TypeError 异常
    raises(TypeError, lambda: MIMOParallel(tfm1, tfm3, tfm4))

    # 所有传递的传递函数矩阵（TFMs）必须使用相同的复变量，否则会引发 ValueError 异常
    raises(ValueError, lambda: MIMOParallel(tfm3, tfm7))

    # 参数中不允许出现数字或表达式，否则会引发 TypeError 异常
    raises(TypeError, lambda: MIMOParallel(2, tfm1, tfm4))
    raises(TypeError, lambda: MIMOParallel(s**2 + p*s, -tfm4, tfm2))
# 定义一个测试函数，用于测试并验证并联（Parallel）函数的功能
def test_Parallel_functions():
    # 创建五个传递函数对象，分别表示不同的传递函数
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    # 断言语句，验证并联函数（Parallel）的运算结果与预期值是否相等
    assert tf1 + tf2 + tf3 == Parallel(tf1, tf2, tf3)
    assert tf1 + tf2 + tf3 + tf5 == Parallel(tf1, tf2, tf3, tf5)
    assert tf1 + tf2 - tf3 - tf5 == Parallel(tf1, tf2, -tf3, -tf5)
    assert tf1 + tf2*tf3 == Parallel(tf1, Series(tf2, tf3))
    assert tf1 - tf2*tf3 == Parallel(tf1, -Series(tf2,tf3))
    assert -tf1 - tf2 == Parallel(-tf1, -tf2)
    assert -(tf1 + tf2) == Series(TransferFunction(-1, 1, s), Parallel(tf1, tf2))
    assert (tf2 + tf3)*tf1 == Series(Parallel(tf2, tf3), tf1)
    assert (tf1 + tf2)*(tf3*tf5) == Series(Parallel(tf1, tf2), tf3, tf5)
    assert -(tf2 + tf3)*-tf5 == Series(TransferFunction(-1, 1, s), Parallel(tf2, tf3), -tf5)
    assert tf2 + tf3 + tf2*tf1 + tf5 == Parallel(tf2, tf3, Series(tf2, tf1), tf5)
    assert tf2 + tf3 + tf2*tf1 - tf3 == Parallel(tf2, tf3, Series(tf2, tf1), -tf3)
    assert (tf1 + tf2 + tf5)*(tf3 + tf5) == Series(Parallel(tf1, tf2, tf5), Parallel(tf3, tf5))
    
    # 使用 lambda 函数检查是否引发了预期的异常
    raises(ValueError, lambda: tf1 + tf2 + tf4)
    raises(ValueError, lambda: tf1 - tf2*tf4)
    raises(ValueError, lambda: tf3 + Matrix([1, 2, 3]))

    # 断言语句，验证带有 evaluate=True 参数的并联函数（Parallel）与其 doit() 方法的结果是否相等
    assert Parallel(tf1, tf2, evaluate=True) == Parallel(tf1, tf2).doit() == \
        TransferFunction(k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s)
    assert Parallel(tf1, tf2, Series(-tf1, tf3), evaluate=True) == \
        Parallel(tf1, tf2, Series(-tf1, tf3)).doit() == TransferFunction(k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)**2 + \
            (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2) + (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), (a2*s + p)*(s**2 + \
                2*s*wn*zeta + wn**2)**2, s)
    assert Parallel(tf2, tf1, -tf3, evaluate=True) == Parallel(tf2, tf1, -tf3).doit() == \
        TransferFunction(a2*s + k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2) \
            , (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert not Parallel(tf1, -tf2, evaluate=False) == Parallel(tf1, -tf2).doit()

    # 断言语句，验证带有 Series 函数的并联函数（Parallel）与其 doit() 方法的结果是否相等
    assert Parallel(Series(tf1, tf2), Series(tf2, tf3)).doit() == \
        TransferFunction(k*(a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2) + k*(a2*s + p), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert Parallel(-tf1, -tf2, -tf3).doit() == \
        TransferFunction(-a2*s - k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + wn**2), \
            (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    assert -Parallel(tf1, tf2, tf3).doit() == \
        TransferFunction(-a2*s - k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - p - (a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2), \
            (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 断言：使用 Parallel 进行组合并计算后，验证结果是否等于指定的 TransferFunction 对象
    assert Parallel(tf2, tf3, Series(tf2, -tf1), tf3).doit() == \
        TransferFunction(k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) - k*(a2*s + p) + (2*a2*p - 2*s)*(s**2 + 2*s*wn*zeta \
            + wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    # 断言：使用 Parallel 进行组合并重写为 TransferFunction 后，验证结果是否等于指定的 TransferFunction 对象
    assert Parallel(tf1, tf2).rewrite(TransferFunction) == \
        TransferFunction(k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s)
    
    # 断言：使用 Parallel 进行组合，并将其重写为 TransferFunction 后，验证结果是否等于指定的 TransferFunction 对象
    assert Parallel(tf2, tf1, -tf3).rewrite(TransferFunction) == \
        TransferFunction(a2*s + k*(a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2) + p + (-a2*p + s)*(s**2 + 2*s*wn*zeta + \
             wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)

    # 断言：验证 Parallel 的结合性质，即多次组合结果是否等价
    assert Parallel(tf1, Parallel(tf2, tf3)) == Parallel(tf1, tf2, tf3) == Parallel(Parallel(tf1, tf2), tf3)

    # 创建并验证并联系统 P1 是否为适当系统、非严格适当系统和双适当系统
    P1 = Parallel(Series(tf1, tf2), Series(tf2, tf3))
    assert P1.is_proper
    assert not P1.is_strictly_proper
    assert P1.is_biproper

    # 创建并验证并联系统 P2 是否为适当系统、非严格适当系统和双适当系统
    P2 = Parallel(tf1, -tf2, -tf3)
    assert P2.is_proper
    assert not P2.is_strictly_proper
    assert P2.is_biproper

    # 创建并验证并联系统 P3 是否为适当系统、非严格适当系统和双适当系统
    P3 = Parallel(tf1, -tf2, Series(tf1, tf3))
    assert P3.is_proper
    assert not P3.is_strictly_proper
    assert P3.is_biproper
# 定义测试函数 test_MIMOParallel_functions，用于测试多输入多输出并行系统的功能
def test_MIMOParallel_functions():
    # 创建传递函数 tf4，其中 a0*p + p**a1 - s 是传递函数的分子，p 是传递函数的变量
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    # 创建传递函数 tf5，其中 a1*s**2 + a2*s - a0 是传递函数的分子，s + a0 是传递函数的变量
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    # 创建传递函数矩阵 tfm1，包含单一传递函数 TF1
    tfm1 = TransferFunctionMatrix([[TF1]])
    # 创建传递函数矩阵 tfm2，包含单一传递函数 -TF2
    tfm2 = TransferFunctionMatrix([[-TF2]])
    # 创建传递函数矩阵 tfm3，包含单一传递函数 tf5
    tfm3 = TransferFunctionMatrix([[tf5]])
    # 创建传递函数矩阵 tfm4，包含两个传递函数 TF2 和 -tf5
    tfm4 = TransferFunctionMatrix([[TF2, -tf5]])
    # 创建传递函数矩阵 tfm5，包含四个传递函数 TF1、TF2、TF3 和 -tf5
    tfm5 = TransferFunctionMatrix([[TF1, TF2], [TF3, -tf5]])
    # 创建传递函数矩阵 tfm6，包含单一传递函数 -TF2
    tfm6 = TransferFunctionMatrix([[-TF2]])
    # 创建传递函数矩阵 tfm7，包含三个传递函数 tf4、-tf4 和 tf4
    tfm7 = TransferFunctionMatrix([[tf4], [-tf4], [tf4]])

    # 使用 assert 语句验证多输入多输出并行系统的加法组合结果
    assert tfm1 + tfm2 + tfm3 == MIMOParallel(tfm1, tfm2, tfm3) == MIMOParallel(MIMOParallel(tfm1, tfm2), tfm3)
    # 使用 assert 语句验证多输入多输出并行系统的减法组合结果
    assert tfm2 - tfm1 - tfm3 == MIMOParallel(tfm2, -tfm1, -tfm3)
    # 使用 assert 语句验证多输入多输出并行系统的复合操作结果
    assert tfm2 - tfm3 + (-tfm1*tfm6*-tfm6) == MIMOParallel(tfm2, -tfm3, MIMOSeries(-tfm6, tfm6, -tfm1))
    assert tfm1 + tfm1 - (-tfm1*tfm6) == MIMOParallel(tfm1, tfm1, -MIMOSeries(tfm6, -tfm1))
    assert tfm2 - tfm3 - tfm1 + tfm2 == MIMOParallel(tfm2, -tfm3, -tfm1, tfm2)
    assert tfm1 + tfm2 - tfm3 - tfm1 == MIMOParallel(tfm1, tfm2, -tfm3, -tfm1)
    
    # 使用 raises 函数和 lambda 表达式验证异常情况
    raises(ValueError, lambda: tfm1 + tfm2 + TF2)
    raises(TypeError, lambda: tfm1 - tfm2 - a1)
    raises(TypeError, lambda: tfm2 - tfm3 - (s - 1))
    raises(TypeError, lambda: -tfm3 - tfm2 - 9)
    raises(TypeError, lambda: (1 - p**3) - tfm3 - tfm2)
    # 所有传递函数矩阵必须使用相同的复变量 'p'，但 tfm7 使用了 'p'
    raises(ValueError, lambda: tfm3 - tfm2 - tfm7)
    raises(ValueError, lambda: tfm2 - tfm1 + tfm7)
    # (tfm1 +/- tfm2) 的形状为 (3, 1)，而 tfm4 的形状为 (2, 2)
    raises(TypeError, lambda: tfm1 + tfm2 + tfm4)
    raises(TypeError, lambda: (tfm1 - tfm2) - tfm4)

    # 使用 assert 语句验证多输入多输出并行系统与 MIMOSeries 的组合操作结果
    assert (tfm1 + tfm2)*tfm6 == MIMOSeries(tfm6, MIMOParallel(tfm1, tfm2))
    assert (tfm2 - tfm3)*tfm6*-tfm6 == MIMOSeries(-tfm6, tfm6, MIMOParallel(tfm2, -tfm3))
    assert (tfm2 - tfm1 - tfm3)*(tfm6 + tfm6) == MIMOSeries(MIMOParallel(tfm6, tfm6), MIMOParallel(tfm2, -tfm1, -tfm3))
    # 验证异常情况
    raises(ValueError, lambda: (tfm4 + tfm5)*TF1)
    raises(TypeError, lambda: (tfm2 - tfm3)*a2)
    raises(TypeError, lambda: (tfm3 + tfm2)*(s - 6))
    raises(TypeError, lambda: (tfm1 + tfm2 + tfm3)*0)
    raises(TypeError, lambda: (1 - p**3)*(tfm1 + tfm3))

    # (tfm3 - tfm2) 的形状为 (3, 1)，而 tfm4*tfm5 的形状为 (2, 2)
    raises(ValueError, lambda: (tfm3 - tfm2)*tfm4*tfm5)
    # (tfm1 - tfm2) 的形状为 (3, 1)，而 tfm5 的形状为 (2, 2)
    raises(ValueError, lambda: (tfm1 - tfm2)*tfm5)

    # 使用 assert 语句验证 MIMOParallel 函数的多种等价形式
    assert (MIMOParallel(tfm1, tfm2, evaluate=True) == MIMOParallel(tfm1, tfm2).doit()
            == MIMOParallel(tfm1, tfm2).rewrite(TransferFunctionMatrix)
            == TransferFunctionMatrix(((TransferFunction(-k*(s**2 + 2*s*wn*zeta + wn**2) + 1, s**2 + 2*s*wn*zeta + wn**2, s),),
                                       (TransferFunction(-a0 + a1*s**2 + a2*s + k*(a0 + s), a0 + s, s),),
                                       (TransferFunction(-a2*s - p + (a2*p - s)*(s**2 + 2*s*wn*zeta + wn**2), (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s),))))
# 定义一个名为 test_Feedback_construction 的测试函数
def test_Feedback_construction():
    # 创建多个传递函数对象并赋值给不同的变量
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    tf6 = TransferFunction(s - p, p + s, p)

    # 创建一个反馈系统对象 f1，使用 TransferFunction 构造的三个传递函数对象相乘
    f1 = Feedback(TransferFunction(1, 1, s), tf1*tf2*tf3)
    assert f1.args == (TransferFunction(1, 1, s), Series(tf1, tf2, tf3), -1)
    assert f1.sys1 == TransferFunction(1, 1, s)
    assert f1.sys2 == Series(tf1, tf2, tf3)
    assert f1.var == s

    # 创建一个反馈系统对象 f2，使用 tf1 和 tf2*tf3 构造
    f2 = Feedback(tf1, tf2*tf3)
    assert f2.args == (tf1, Series(tf2, tf3), -1)
    assert f2.sys1 == tf1
    assert f2.sys2 == Series(tf2, tf3)
    assert f2.var == s

    # 创建一个反馈系统对象 f3，使用 tf1*tf2 和 tf5 构造
    f3 = Feedback(tf1*tf2, tf5)
    assert f3.args == (Series(tf1, tf2), tf5, -1)
    assert f3.sys1 == Series(tf1, tf2)

    # 创建一个反馈系统对象 f4，使用 tf4 和 tf6 构造
    f4 = Feedback(tf4, tf6)
    assert f4.args == (tf4, tf6, -1)
    assert f4.sys1 == tf4
    assert f4.var == p

    # 创建一个反馈系统对象 f5，使用 tf5 和 TransferFunction(1, 1, s) 构造
    f5 = Feedback(tf5, TransferFunction(1, 1, s))
    assert f5.args == (tf5, TransferFunction(1, 1, s), -1)
    assert f5.var == s
    assert f5 == Feedback(tf5)  # 当未显式传递 sys2 参数时，假定为单位传递函数 tf.

    # 创建一个反馈系统对象 f6，使用 TransferFunction(1, 1, p) 和 tf4 构造
    f6 = Feedback(TransferFunction(1, 1, p), tf4)
    assert f6.args == (TransferFunction(1, 1, p), tf4, -1)
    assert f6.var == p

    # 创建一个反馈系统对象 f7，使用 tf4*tf6 和 TransferFunction(1, 1, p) 构造，并取其相反数
    f7 = -Feedback(tf4*tf6, TransferFunction(1, 1, p))
    assert f7.args == (Series(TransferFunction(-1, 1, p), Series(tf4, tf6)), -TransferFunction(1, 1, p), -1)
    assert f7.sys1 == Series(TransferFunction(-1, 1, p), Series(tf4, tf6))

    # 检查以下情况会引发异常
    raises(TypeError, lambda: Feedback(tf1, tf2 + tf3))  # 分母不能是并联传递函数
    raises(TypeError, lambda: Feedback(tf1, Matrix([1, 2, 3])))  # 分母不能是矩阵
    raises(TypeError, lambda: Feedback(TransferFunction(1, 1, s), s - 1))  # 分母不能为零
    raises(TypeError, lambda: Feedback(1, 1))  # 参数必须是传递函数对象
    raises(ValueError, lambda: Feedback(tf2, tf4*tf5))  # 分子和分母的变量必须一致
    raises(ValueError, lambda: Feedback(tf2, tf1, 1.5))  # sign 参数只能是 -1 或 1
    raises(ValueError, lambda: Feedback(tf1, -tf1**-1))  # 分母不能为零
    raises(ValueError, lambda: Feedback(tf4, tf5))  # 分子和分母的变量必须一致


# 定义一个名为 test_Feedback_functions 的测试函数
def test_Feedback_functions():
    # 创建多个传递函数对象并赋值给不同的变量
    tf = TransferFunction(1, 1, s)
    tf1 = TransferFunction(1, s**2 + 2*zeta*wn*s + wn**2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2*p - s, a2*s + p, s)
    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    tf6 = TransferFunction(s - p, p + s, p)

    # 断言一系列等式，验证传递函数的运算
    assert (tf1*tf2*tf3 / tf3*tf5) == Series(tf1, tf2, tf3, pow(tf3, -1), tf5)
    assert (tf1*tf2*tf3) / (tf3*tf5) == Series((tf1*tf2*tf3).doit(), pow((tf3*tf5).doit(),-1))
    assert tf / (tf + tf1) == Feedback(tf, tf1)
    assert tf / (tf + tf1*tf2*tf3) == Feedback(tf, tf1*tf2*tf3)
    # 确定等式是否成立，计算左侧表达式和右侧的反馈
    assert tf1 / (tf + tf1*tf2*tf3) == Feedback(tf1, tf2*tf3)
    # 确定等式是否成立，计算左侧表达式和右侧的反馈
    assert (tf1*tf2) / (tf + tf1*tf2) == Feedback(tf1*tf2, tf)
    # 确定等式是否成立，计算左侧表达式和右侧的反馈
    assert (tf1*tf2) / (tf + tf1*tf2*tf5) == Feedback(tf1*tf2, tf5)
    # 确定等式是否成立，计算左侧表达式和右侧的反馈，右侧有两种可能结果
    assert (tf1*tf2) / (tf + tf1*tf2*tf5*tf3) in (Feedback(tf1*tf2, tf5*tf3), Feedback(tf1*tf2, tf3*tf5))
    # 确定等式是否成立，计算左侧表达式和右侧的反馈
    assert tf4 / (TransferFunction(1, 1, p) + tf4*tf6) == Feedback(tf4, tf6)
    # 确定等式是否成立，计算左侧表达式和右侧的反馈
    assert tf5 / (tf + tf5) == Feedback(tf5, tf)

    # 预期引发 TypeError 异常，计算左侧表达式
    raises(TypeError, lambda: tf1*tf2*tf3 / (1 + tf1*tf2*tf3))
    # 预期引发 ValueError 异常，计算左侧表达式
    raises(ValueError, lambda: tf2*tf3 / (tf + tf2*tf3*tf4))

    # 确定等式是否成立，计算左侧表达式的结果
    assert Feedback(tf, tf1*tf2*tf3).doit() == \
        TransferFunction((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), k*(a2*p - s) + \
        (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 确定等式是否成立，计算左侧表达式的灵敏度
    assert Feedback(tf, tf1*tf2*tf3).sensitivity == \
        1/(k*(a2*p - s)/((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    # 确定等式是否成立，计算左侧表达式的结果
    assert Feedback(tf1, tf2*tf3).doit() == \
        TransferFunction((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2), (k*(a2*p - s) + \
        (a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 确定等式是否成立，计算左侧表达式的灵敏度
    assert Feedback(tf1, tf2*tf3).sensitivity == \
        1/(k*(a2*p - s)/((a2*s + p)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    # 确定等式是否成立，计算左侧表达式的结果
    assert Feedback(tf1*tf2, tf5).doit() == \
        TransferFunction(k*(a0 + s)*(s**2 + 2*s*wn*zeta + wn**2), (k*(-a0 + a1*s**2 + a2*s) + \
        (a0 + s)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 确定等式是否成立，计算左侧表达式的灵敏度
    assert Feedback(tf1*tf2, tf5, 1).sensitivity == \
        1/(-k*(-a0 + a1*s**2 + a2*s)/((a0 + s)*(s**2 + 2*s*wn*zeta + wn**2)) + 1)
    # 确定等式是否成立，计算左侧表达式的结果
    assert Feedback(tf4, tf6).doit() == \
        TransferFunction(p*(p + s)*(a0*p + p**a1 - s), p*(p*(p + s) + (-p + s)*(a0*p + p**a1 - s)), p)
    # 确定等式是否成立，计算左侧表达式的结果
    assert -Feedback(tf4*tf6, TransferFunction(1, 1, p)).doit() == \
        TransferFunction(-p*(-p + s)*(p + s)*(a0*p + p**a1 - s), p*(p + s)*(p*(p + s) + (-p + s)*(a0*p + p**a1 - s)), p)
    # 确定等式是否成立，计算左侧表达式的结果
    assert Feedback(tf, tf).doit() == TransferFunction(1, 2, s)

    # 确定等式是否成立，计算左侧表达式的结果并将结果转换为传递函数
    assert Feedback(tf1, tf2*tf5).rewrite(TransferFunction) == \
        TransferFunction((a0 + s)*(s**2 + 2*s*wn*zeta + wn**2), (k*(-a0 + a1*s**2 + a2*s) + \
        (a0 + s)*(s**2 + 2*s*wn*zeta + wn**2))*(s**2 + 2*s*wn*zeta + wn**2), s)
    # 确定等式是否成立，计算左侧表达式的结果并将结果转换为传递函数
    assert Feedback(TransferFunction(1, 1, p), tf4).rewrite(TransferFunction) == \
        TransferFunction(p, a0*p + p + p**a1 - s, p)
def test_Feedback_as_TransferFunction():
    # 解决问题 https://github.com/sympy/sympy/issues/26161
    # 创建 TransferFunction 对象 tf1 和 tf2
    tf1 = TransferFunction(s+1, 1, s)
    tf2 = TransferFunction(s+2, 1, s)
    
    # 创建负反馈系统和正反馈系统的 Feedback 对象
    fd1 = Feedback(tf1, tf2, -1) # 负反馈系统
    fd2 = Feedback(tf1, tf2, 1)  # 正反馈系统
    
    # 创建单位 TransferFunction 对象
    unit = TransferFunction(1, 1, s)

    # 检查对象类型
    assert isinstance(fd1, TransferFunction)
    assert isinstance(fd1, Feedback)

    # 测试分子和分母
    assert fd1.num == tf1
    assert fd2.num == tf1
    assert fd1.den == Parallel(unit, Series(tf2, tf1))
    assert fd2.den == Parallel(unit, -Series(tf2, tf1))

    # 测试使用反馈和 TransferFunction 进行串联和并联组合
    s1 = Series(tf1, fd1)
    p1 = Parallel(tf1, fd1)
    assert tf1 * fd1 == s1
    assert tf1 + fd1 == p1
    assert s1.doit() == TransferFunction((s + 1)**2, (s + 1)*(s + 2) + 1, s)
    assert p1.doit() == TransferFunction(s + (s + 1)*((s + 1)*(s + 2) + 1) + 1, (s + 1)*(s + 2) + 1, s)

    # 测试使用带有反馈的 TransferFunction 进行串联
    fd3 = Feedback(tf1*fd1, tf2, -1)
    assert fd3 == Feedback(Series(tf1, fd1), tf2)
    assert fd3.num == tf1 * fd1
    assert fd3.den == Parallel(unit, Series(tf2, Series(tf1, fd1)))

    # 测试使用带有 TransferFunction 的反馈系统
    tf3 = TransferFunction(tf1*fd1, tf2, s)
    assert tf3 == TransferFunction(Series(tf1, fd1), tf2, s)
    assert tf3.num == tf1*fd1

def test_issue_26161():
    # 解决问题 https://github.com/sympy/sympy/issues/26161
    # 定义符号变量
    Ib, Is, m, h, l2, l1 = symbols('I_b, I_s, m, h, l2, l1',
                                    real=True, nonnegative=True)
    KD, KP, v = symbols('K_D, K_P, v', real=True)

    # 计算参数
    tau1_sq = (Ib + m * h ** 2) / m / g / h
    tau2 = l2 / v
    tau3 = v / (l1 + l2)
    K = v ** 2 / g / (l1 + l2)

    # 创建 TransferFunction 对象
    Gtheta = TransferFunction(-K * (tau2 * s + 1), tau1_sq * s ** 2 - 1, s)
    Gdelta = TransferFunction(1, Is * s ** 2 + c * s, s)
    Gpsi = TransferFunction(1, tau3 * s, s)
    Dcont = TransferFunction(KD * s, 1, s)
    PIcont = TransferFunction(KP, s, s)
    Gunity = TransferFunction(1, 1, s)

    # 创建反馈系统
    Ginner = Feedback(Dcont * Gdelta, Gtheta)
    Gouter = Feedback(PIcont * Ginner * Gpsi, Gunity)

    # 断言期望结果
    assert Gouter == Feedback(Series(PIcont, Series(Ginner, Gpsi)), Gunity)
    assert Gouter.num == Series(PIcont, Series(Ginner, Gpsi))
    assert Gouter.den == Parallel(Gunity, Series(Gunity, Series(PIcont, Series(Ginner, Gpsi))))
    # 计算表达式 `expr`，该表达式是一个复杂的数学公式，涉及多个变量和常数
    expr = (KD*KP*g*s**3*v**2*(l1 + l2)*(Is*s**2 + c*s)**2*(-g*h*m + s**2*(Ib + h**2*m))*(-KD*g*h*m*s*v**2*(l2*s + v) + \
            g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m))))/((s**2*v*(Is*s**2 + c*s)*(-KD*g*h*m*s*v**2* \
            (l2*s + v) + g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m)))*(KD*KP*g*s*v*(l1 + l2)**2* \
            (Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m)) + s**2*v*(Is*s**2 + c*s)*(-KD*g*h*m*s*v**2*(l2*s + v) + \
            g*v*(l1 + l2)*(Is*s**2 + c*s)*(-g*h*m + s**2*(Ib + h**2*m))))/(l1 + l2)))
    
    # 断言表达式 `Gouter.to_expr() - expr` 简化后等于 0
    assert (Gouter.to_expr() - expr).simplify() == 0
# 定义测试函数，用于测试 MIMOFeedback 类的构造功能
def test_MIMOFeedback_construction():
    # 创建四个传递函数对象，分别对应不同的分子和分母
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**3 - 1, s)
    tf3 = TransferFunction(s, s + 1, s)
    tf4 = TransferFunction(s, s**2 + 1, s)

    # 创建传递函数矩阵对象 tfm_1, tfm_2, tfm_3，每个矩阵包含两个传递函数对象
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf3], [tf4, tf1]])
    tfm_3 = TransferFunctionMatrix([[tf3, tf4], [tf1, tf2]])

    # 创建 MIMOFeedback 对象 f1，其参数为 tfm_1 和 tfm_2，反馈符号为 -1
    f1 = MIMOFeedback(tfm_1, tfm_2)
    # 断言 f1 对象的属性符合预期值
    assert f1.args == (tfm_1, tfm_2, -1)
    assert f1.sys1 == tfm_1
    assert f1.sys2 == tfm_2
    assert f1.var == s
    assert f1.sign == -1
    # 断言反复反馈 -f1 后与 f1 对象相等
    assert -(-f1) == f1

    # 创建另一个 MIMOFeedback 对象 f2，其参数为 tfm_2 和 tfm_1，反馈符号为 1
    f2 = MIMOFeedback(tfm_2, tfm_1, 1)
    # 断言 f2 对象的属性符合预期值
    assert f2.args == (tfm_2, tfm_1, 1)
    assert f2.sys1 == tfm_2
    assert f2.sys2 == tfm_1
    assert f2.var == s
    assert f2.sign == 1

    # 创建第三个 MIMOFeedback 对象 f3，其参数为 tfm_1 和 MIMOSeries(tfm_3, tfm_2)，反馈符号为 -1
    f3 = MIMOFeedback(tfm_1, MIMOSeries(tfm_3, tfm_2))
    # 断言 f3 对象的属性符合预期值
    assert f3.args == (tfm_1, MIMOSeries(tfm_3, tfm_2), -1)
    assert f3.sys1 == tfm_1
    assert f3.sys2 == MIMOSeries(tfm_3, tfm_2)
    assert f3.var == s
    assert f3.sign == -1

    # 创建矩阵 mat，用于创建 sys1 和 controller 对象
    mat = Matrix([[1, 1/s], [0, 1]])
    sys1 = controller = TransferFunctionMatrix.from_Matrix(mat, s)
    # 创建第四个 MIMOFeedback 对象 f4，其参数为 sys1 和 controller，反馈符号为 -1
    f4 = MIMOFeedback(sys1, controller)
    # 断言 f4 对象的属性符合预期值
    assert f4.args == (sys1, controller, -1)
    assert f4.sys1 == f4.sys2 == sys1


# 定义测试函数，用于测试 MIMOFeedback 类的错误情况
def test_MIMOFeedback_errors():
    # 创建多个传递函数对象，用于测试不同的错误情况
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**3 - 1, s)
    tf3 = TransferFunction(s, s - 1, s)
    tf4 = TransferFunction(s, s**2 + 1, s)
    tf5 = TransferFunction(1, 1, s)
    tf6 = TransferFunction(-1, s - 1, s)

    # 创建多个传递函数矩阵对象，用于测试不同的错误情况
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf3], [tf4, tf1]])
    tfm_3 = TransferFunctionMatrix.from_Matrix(eye(2), var=s)
    tfm_4 = TransferFunctionMatrix([[tf1, tf5], [tf5, tf5]])
    tfm_5 = TransferFunctionMatrix([[-tf3, tf3], [tf3, tf6]])
    tfm_6 = TransferFunctionMatrix([[-tf3]])
    tfm_7 = TransferFunctionMatrix([[tf3, tf4]])

    # 断言以下情况会引发指定的错误类型
    # 1. 传递函数对象类型不支持
    raises(TypeError, lambda: MIMOFeedback(tf1, tf2))
    raises(TypeError, lambda: MIMOFeedback(MIMOParallel(tfm_1, tfm_2), tfm_3))
    # 2. 形状错误
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_6, 1))
    raises(ValueError, lambda: MIMOFeedback(tfm_7, tfm_7))
    # 3. 反馈符号不为 1 或 -1
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_2, -2))
    # 4. 系统不可逆
    raises(ValueError, lambda: MIMOFeedback(tfm_5, tfm_4, 1))
    raises(ValueError, lambda: MIMOFeedback(tfm_4, -tfm_5))
    raises(ValueError, lambda: MIMOFeedback(tfm_3, tfm_3, 1))
    # 5. 系统中变量不同
    tfm_8 = TransferFunctionMatrix.from_Matrix(eye(2), var=p)
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_8, 1))


# 定义测试函数，用于测试 MIMOFeedback 类的功能函数
def test_MIMOFeedback_functions():
    # 创建多个传递函数对象，用于测试功能函数
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s - 1, s)
    tf3 = TransferFunction(1, 1, s)
    tf4 = TransferFunction(-1, s - 1, s)

    # 创建传递函数矩阵对象 tfm_1，其变量为 s
    tfm_1 = TransferFunctionMatrix.from_Matrix(eye(2), var=s)
    # 创建三个传递函数矩阵对象，分别用给定的传递函数填充
    tfm_2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf3]])
    tfm_3 = TransferFunctionMatrix([[-tf2, tf2], [tf2, tf4]])
    tfm_4 = TransferFunctionMatrix([[tf1, tf2], [-tf2, tf1]])

    # 创建两个多输入多输出系统的反馈对象，使用给定的传递函数矩阵作为参数
    F_1 = MIMOFeedback(tfm_2, tfm_3)
    F_2 = MIMOFeedback(tfm_2, MIMOSeries(tfm_4, -tfm_1), 1)

    # 断言语句，验证 F_1 的灵敏度矩阵是否符合预期
    assert F_1.sensitivity == Matrix([[S.Half, 0], [0, S.Half]])
    
    # 断言语句，验证 F_2 的灵敏度矩阵是否符合预期
    assert F_2.sensitivity == Matrix([[(-2*s**4 + s**2)/(s**2 - s + 1),
        (2*s**3 - s**2)/(s**2 - s + 1)], [-s**2, s]])

    # 断言语句，验证 F_1 的 doit() 方法是否正确计算并与预期的传递函数矩阵相等
    assert F_1.doit() == \
        TransferFunctionMatrix(((TransferFunction(1, 2*s, s),
        TransferFunction(1, 2, s)), (TransferFunction(1, 2, s),
        TransferFunction(1, 2, s))))
    
    # 断言语句，验证 F_2 的 doit() 方法是否正确计算并与预期的传递函数矩阵相等，同时考虑取消因式分解和展开
    assert F_2.doit(cancel=False, expand=True) == \
        TransferFunctionMatrix(((TransferFunction(-s**5 + 2*s**4 - 2*s**3 + s**2, s**5 - 2*s**4 + 3*s**3 - 2*s**2 + s, s),
        TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))
    
    # 断言语句，验证 F_2 的 doit() 方法是否正确计算并与预期的传递函数矩阵相等，取消因式分解
    assert F_2.doit(cancel=False) == \
        TransferFunctionMatrix(((TransferFunction(s*(2*s**3 - s**2)*(s**2 - s + 1) + \
        (-2*s**4 + s**2)*(s**2 - s + 1), s*(s**2 - s + 1)**2, s), TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)),
        (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))
    
    # 断言语句，验证 F_2 的 doit() 方法是否正确计算并与预期的传递函数矩阵相等
    assert F_2.doit() == \
        TransferFunctionMatrix(((TransferFunction(s*(-2*s**2 + s*(2*s - 1) + 1), s**2 - s + 1, s),
        TransferFunction(-2*s**3*(s - 1), s**2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(s*(1 - s), 1, s))))
    
    # 断言语句，验证 F_2 的 doit() 方法是否正确计算并与预期的传递函数矩阵相等，展开表达式
    assert F_2.doit(expand=True) == \
        TransferFunctionMatrix(((TransferFunction(-s**2 + s, s**2 - s + 1, s), TransferFunction(-2*s**4 + 2*s**3, s**2 - s + 1, s)),
        (TransferFunction(0, 1, s), TransferFunction(-s**2 + s, 1, s))))
    
    # 断言语句，验证负号应用在 F_1 的 doit() 方法结果上是否等同于先计算结果再应用负号
    assert -(F_1.doit()) == (-F_1).doit()  # First negating then calculating vs calculating then negating.
def test_TransferFunctionMatrix_construction():
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)
    # 创建一个传递函数 tf5，其中包含表达式 a1*s**2 + a2*s - a0，并指定变量 s 和分母 s + a0

    tf4 = TransferFunction(a0*p + p**a1 - s, p, p)
    # 创建另一个传递函数 tf4，其包含表达式 a0*p + p**a1 - s，并指定变量 p 和分母 p

    tfm3_ = TransferFunctionMatrix([[-TF3]])
    # 创建一个传递函数矩阵 tfm3_，包含单个元素 -TF3

    assert tfm3_.shape == (tfm3_.num_outputs, tfm3_.num_inputs) == (1, 1)
    # 断言确认 tfm3_ 的形状为 (1, 1)，即一个输出和一个输入
    assert tfm3_.args == Tuple(Tuple(Tuple(-TF3)))
    # 断言确认 tfm3_ 的参数 args 是一个嵌套的元组结构 (-TF3)

    assert tfm3_.var == s
    # 断言确认 tfm3_ 使用的变量是 s

    tfm5 = TransferFunctionMatrix([[TF1, -TF2], [TF3, tf5]])
    # 创建一个传递函数矩阵 tfm5，包含 TF1, -TF2 和 TF3, tf5 作为其元素

    assert tfm5.shape == (tfm5.num_outputs, tfm5.num_inputs) == (2, 2)
    # 断言确认 tfm5 的形状为 (2, 2)，即两个输出和两个输入
    assert tfm5.args == Tuple(Tuple(Tuple(TF1, -TF2), Tuple(TF3, tf5)))
    # 断言确认 tfm5 的参数 args 是一个嵌套的元组结构 ((TF1, -TF2), (TF3, tf5))

    assert tfm5.var == s
    # 断言确认 tfm5 使用的变量是 s

    tfm7 = TransferFunctionMatrix([[TF1, TF2], [TF3, -tf5], [-tf5, TF2]])
    # 创建一个传递函数矩阵 tfm7，包含 TF1, TF2, TF3, -tf5 和 -tf5, TF2 作为其元素

    assert tfm7.shape == (tfm7.num_outputs, tfm7.num_inputs) == (3, 2)
    # 断言确认 tfm7 的形状为 (3, 2)，即三个输出和两个输入
    assert tfm7.args == Tuple(Tuple(Tuple(TF1, TF2), Tuple(TF3, -tf5), Tuple(-tf5, TF2)))
    # 断言确认 tfm7 的参数 args 是一个嵌套的元组结构 ((TF1, TF2), (TF3, -tf5), (-tf5, TF2))

    assert tfm7.var == s
    # 断言确认 tfm7 使用的变量是 s

    # 所有的传递函数矩阵都应该使用相同的复杂变量。tf4 使用 'p'。
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1], [TF2], [tf4]]))
    # 预期会引发 ValueError 异常，因为传递函数矩阵中包含不同的变量 'p'

    raises(ValueError, lambda: TransferFunctionMatrix([[TF1, tf4], [TF3, tf5]]))
    # 预期会引发 ValueError 异常，因为传递函数矩阵中包含不同的变量 'p'

    # 所有 TFM 中列表的长度应该相等。
    raises(ValueError, lambda: TransferFunctionMatrix([[TF1], [TF3, tf5]]))
    # 预期会引发 ValueError 异常，因为传递函数矩阵中某些行的长度不相等

    raises(ValueError, lambda: TransferFunctionMatrix([[TF1, TF3], [tf5]]))
    # 预期会引发 ValueError 异常，因为传递函数矩阵中某些行的长度不相等

    # 列表中只支持传递函数。
    raises(TypeError, lambda: TransferFunctionMatrix([[TF1, TF2], [TF3, Matrix([1, 2])]]))
    # 预期会引发 TypeError 异常，因为传递函数矩阵中包含了非传递函数对象 Matrix([1, 2])

    raises(TypeError, lambda: TransferFunctionMatrix([[TF1, Matrix([1, 2])], [TF3, TF2]]))
    # 预期会引发 TypeError 异常，因为传递函数矩阵中包含了非传递函数对象 Matrix([1, 2])

    # `arg` 必须严格是传递函数的嵌套列表。
    raises(ValueError, lambda: TransferFunctionMatrix([TF1, TF2, tf5]))
    # 预期会引发 ValueError 异常，因为传递函数矩阵的参数不是嵌套列表

    raises(ValueError, lambda: TransferFunctionMatrix([TF1]))
    # 预期会引发 ValueError 异常，因为传递函数矩阵的参数不是嵌套列表

def test_TransferFunctionMatrix_functions():
    tf5 = TransferFunction(a1*s**2 + a2*s - a0, s + a0, s)

    # Classmethod (from_matrix)

    mat_1 = ImmutableMatrix([
        [s*(s + 1)*(s - 3)/(s**4 + 1), 2],
        [p, p*(s + 1)/(s*(s**1 + 1))]
        ])
    mat_2 = ImmutableMatrix([[(2*s + 1)/(s**2 - 9)]])
    mat_3 = ImmutableMatrix([[1, 2], [3, 4]])

    assert TransferFunctionMatrix.from_Matrix(mat_1, s) == \
        TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)],
        [TransferFunction(p, 1, s), TransferFunction(p, s, s)]])
    # 断言确认从给定的矩阵 mat_1 中创建的传递函数矩阵与期望的结果匹配

    assert TransferFunctionMatrix.from_Matrix(mat_2, s) == \
        TransferFunctionMatrix([[TransferFunction(2*s + 1, s**2 - 9, s)]])
    # 断言确认从给定的矩阵 mat_2 中创建的传递函数矩阵与期望的结果匹配

    assert TransferFunctionMatrix.from_Matrix(mat_3, p) == \
        TransferFunctionMatrix([[TransferFunction(1, 1, p), TransferFunction(2, 1, p)],
        [TransferFunction(3, 1, p), TransferFunction(4, 1, p)]])
    # 断言确认从给定的矩阵 mat_3 中创建的传递函数矩阵与期望的结果匹配

    # Negating a TFM

    tfm1 = TransferFunctionMatrix([[TF1], [TF2]])
    assert -tfm1 == TransferFunctionMatrix([[-TF1], [-TF2]])
    # 断言确认传递函数矩阵 tfm1 取负后的结果与期望的结果匹配

    tfm2 = TransferFunctionMatrix([[TF1, TF2, TF3], [tf5, -TF1, -TF3]])
    assert -tfm2 == TransferFunctionMatrix([[-TF1, -TF2, -TF3], [-tf5, TF1, TF3]])
    # 断言确认传递函数矩阵 tfm2 取负后的结果与期望的结果匹配

    # subs()
    # 使用给定的矩阵 mat_1 和复变量 s 创建传递函数矩阵 H_1
    H_1 = TransferFunctionMatrix.from_Matrix(mat_1, s)

    # 创建一个传递函数矩阵 H_2，包含两个传递函数
    H_2 = TransferFunctionMatrix([[TransferFunction(a*p*s, k*s**2, s), TransferFunction(p*s, k*(s**2 - a), s)]])

    # 断言 H_1 在替换 p 为 1 后等于指定的传递函数矩阵
    assert H_1.subs(p, 1) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])

    # 断言 H_1 在将 p 替换为 1 后等于指定的传递函数矩阵
    assert H_1.subs({p: 1}) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])

    # 断言 H_1 在将 p 和 s 替换为 1 后等于指定的传递函数矩阵，s 被标记为 `var`，应被忽略
    assert H_1.subs({p: 1, s: 1}) == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]]) # This should ignore `s` as it is `var`

    # 断言 H_2 在替换 p 为 2 后等于指定的传递函数矩阵
    assert H_2.subs(p, 2) == TransferFunctionMatrix([[TransferFunction(2*a*s, k*s**2, s), TransferFunction(2*s, k*(-a + s**2), s)]])

    # 断言 H_2 在替换 k 为 1 后等于指定的传递函数矩阵
    assert H_2.subs(k, 1) == TransferFunctionMatrix([[TransferFunction(a*p*s, s**2, s), TransferFunction(p*s, -a + s**2, s)]])

    # 断言 H_2 在替换 a 为 0 后等于指定的传递函数矩阵
    assert H_2.subs(a, 0) == TransferFunctionMatrix([[TransferFunction(0, k*s**2, s), TransferFunction(p*s, k*s**2, s)]])

    # 断言 H_2 在将 p、k 和 a 替换为给定值后等于指定的传递函数矩阵
    assert H_2.subs({p: 1, k: 1, a: a0}) == TransferFunctionMatrix([[TransferFunction(a0*s, s**2, s), TransferFunction(s, -a0 + s**2, s)]])

    # 调用 eval_frequency() 方法，断言其返回值等于指定的矩阵
    assert H_2.eval_frequency(S(1)/2 + I) == Matrix([[2*a*p/(5*k) - 4*I*a*p/(5*k), I*p/(-a*k - 3*k/4 + I*k) + p/(-2*a*k - 3*k/2 + 2*I*k)]])

    # 调用 transpose() 方法，断言其返回值等于指定的传递函数矩阵
    assert H_1.transpose() == TransferFunctionMatrix([[TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(p, 1, s)], [TransferFunction(2, 1, s), TransferFunction(p, s, s)]])
    assert H_2.transpose() == TransferFunctionMatrix([[TransferFunction(a*p*s, k*s**2, s)], [TransferFunction(p*s, k*(-a + s**2), s)]])
    assert H_1.transpose().transpose() == H_1
    assert H_2.transpose().transpose() == H_2

    # 调用 elem_poles() 方法，断言其返回值等于指定的元素极点列表
    assert H_1.elem_poles() == [[[-sqrt(2)/2 - sqrt(2)*I/2, -sqrt(2)/2 + sqrt(2)*I/2, sqrt(2)/2 - sqrt(2)*I/2, sqrt(2)/2 + sqrt(2)*I/2], []],
        [[], [0]]]
    assert H_2.elem_poles() == [[[0, 0], [sqrt(a), -sqrt(a)]]]
    assert tfm2.elem_poles() == [[[wn*(-zeta + sqrt((zeta - 1)*(zeta + 1))), wn*(-zeta - sqrt((zeta - 1)*(zeta + 1)))], [], [-p/a2]],
        [[-a0], [wn*(-zeta + sqrt((zeta - 1)*(zeta + 1))), wn*(-zeta - sqrt((zeta - 1)*(zeta + 1)))], [-p/a2]]]

    # 调用 elem_zeros() 方法，断言其返回值等于指定的元素零点列表
    assert H_1.elem_zeros() == [[[-1, 0, 3], []], [[], []]]
    assert H_2.elem_zeros() == [[[0], [0]]]
    assert tfm2.elem_zeros() == [[[], [], [a2*p]],
        [[-a2/(2*a1) - sqrt(4*a0*a1 + a2**2)/(2*a1), -a2/(2*a1) + sqrt(4*a0*a1 + a2**2)/(2*a1)], [], [a2*p]]]

    # 调用 doit() 方法，创建传递函数矩阵 H_3
    H_3 = TransferFunctionMatrix([[Series(TransferFunction(1, s**3 - 3, s), TransferFunction(s**2 - 2*s + 5, 1, s), TransferFunction(1, s, s))]])
    H_4 = TransferFunctionMatrix([[Parallel(TransferFunction(s**3 - 3, 4*s**4 - s**2 - 2*s + 5, s), TransferFunction(4 - s**3, 4*s**4 - s**2 - 2*s + 5, s))]])
    # 创建一个传递函数矩阵 H_4，包含一个并联结构，其中包括两个传递函数：
    #   - 第一个传递函数：分子为 s**3 - 3，分母为 4*s**4 - s**2 - 2*s + 5
    #   - 第二个传递函数：分子为 4 - s**3，分母同样为 4*s**4 - s**2 - 2*s + 5

    assert H_3.doit() == TransferFunctionMatrix([[TransferFunction(s**2 - 2*s + 5, s*(s**3 - 3), s)]])
    # 断言：H_3.doit() 的结果应该等于包含一个传递函数的传递函数矩阵：
    #   - 传递函数的分子为 s**2 - 2*s + 5
    #   - 传递函数的分母为 s*(s**3 - 3)

    assert H_4.doit() == TransferFunctionMatrix([[TransferFunction(1, 4*s**4 - s**2 - 2*s + 5, s)]])
    # 断言：H_4.doit() 的结果应该等于包含一个传递函数的传递函数矩阵：
    #   - 传递函数的分子为 1
    #   - 传递函数的分母为 4*s**4 - s**2 - 2*s + 5

    # _flat()

    assert H_1._flat() == [TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s), TransferFunction(2, 1, s), TransferFunction(p, 1, s), TransferFunction(p, s, s)]
    # 断言：H_1._flat() 应该返回一个列表，包含以下四个传递函数：
    #   - TransferFunction(s*(s - 3)*(s + 1), s**4 + 1, s)
    #   - TransferFunction(2, 1, s)
    #   - TransferFunction(p, 1, s)
    #   - TransferFunction(p, s, s)

    assert H_2._flat() == [TransferFunction(a*p*s, k*s**2, s), TransferFunction(p*s, k*(-a + s**2), s)]
    # 断言：H_2._flat() 应该返回一个列表，包含以下两个传递函数：
    #   - TransferFunction(a*p*s, k*s**2, s)
    #   - TransferFunction(p*s, k*(-a + s**2), s)

    assert H_3._flat() == [Series(TransferFunction(1, s**3 - 3, s), TransferFunction(s**2 - 2*s + 5, 1, s), TransferFunction(1, s, s))]
    # 断言：H_3._flat() 应该返回一个列表，包含一个 Series 结构的传递函数：
    #   - 传递函数结构为 Series(TransferFunction(1, s**3 - 3, s), TransferFunction(s**2 - 2*s + 5, 1, s), TransferFunction(1, s, s))

    assert H_4._flat() == [Parallel(TransferFunction(s**3 - 3, 4*s**4 - s**2 - 2*s + 5, s), TransferFunction(4 - s**3, 4*s**4 - s**2 - 2*s + 5, s))]
    # 断言：H_4._flat() 应该返回一个列表，包含一个 Parallel 结构的传递函数：
    #   - 传递函数结构为 Parallel(TransferFunction(s**3 - 3, 4*s**4 - s**2 - 2*s + 5, s), TransferFunction(4 - s**3, 4*s**4 - s**2 - 2*s + 5, s))

    # evalf()

    assert H_1.evalf() == \
        TransferFunctionMatrix(((TransferFunction(s*(s - 3.0)*(s + 1.0), s**4 + 1.0, s), TransferFunction(2.0, 1, s)),
                                (TransferFunction(1.0*p, 1, s), TransferFunction(p, s, s))))
    # 断言：H_1.evalf() 的结果应该等于以下传递函数矩阵：
    #   - 第一行包含两个传递函数：
    #     - TransferFunction(s*(s - 3.0)*(s + 1.0), s**4 + 1.0, s)
    #     - TransferFunction(2.0, 1, s)
    #   - 第二行包含两个传递函数：
    #     - TransferFunction(1.0*p, 1, s)
    #     - TransferFunction(p, s, s)

    assert H_2.subs({a:3.141, p:2.88, k:2}).evalf() == \
        TransferFunctionMatrix(((TransferFunction(4.5230399999999999494093572138808667659759521484375, s, s),
                                TransferFunction(2.87999999999999989341858963598497211933135986328125*s, 2.0*s**2 - 6.282000000000000028421709430404007434844970703125, s)),))
    # 断言：将 H_2 中的符号 a 替换为 3.141，p 替换为 2.88，k 替换为 2，并对其结果进行 evalf() 后应等于以下传递函数矩阵：
    #   - 包含一个传递函数矩阵，其中包含一个传递函数：
    #     - TransferFunction(4.5230399999999999494093572138808667659759521484375, s, s)
    #     - TransferFunction(2.87999999999999989341858963598497211933135986328125*s, 2.0*s**2 - 6.282000000000000028421709430404007434844970703125, s)

    # simplify()

    H_5 = TransferFunctionMatrix([[TransferFunction(s**5 + s**3 + s, s - s**2, s),
                                  TransferFunction((s + 3)*(s - 1), (s - 1)*(s + 5), s)]])
    # 创建一个传递函数矩阵 H_5，包含两个传递函数：
    #   - 第一个传递函数：分子为 s**5 + s**3 + s，分母为 s - s**2
    #   - 第二个传递函数：分子为 (s + 3)*(s - 1)，分母为 (s - 1)*(s + 5)

    assert H_5.simplify() == simplify(H_5) == \
        TransferFunctionMatrix(((TransferFunction(-s**4 - s**2 - 1, s - 1, s), TransferFunction(s + 3, s + 5, s)),))
    # 断言：对 H_5 进行 simplify() 后应等于简化后的传递函数矩阵：
    #   - 包含一个传递函数矩阵，其中包含一个传递函数：
    #     - TransferFunction(-s**4 - s**2 - 1, s - 1, s)
    #     - TransferFunction(s + 3, s + 5, s)

    # expand()

    assert (H_1.expand()
            == TransferFunctionMatrix(((TransferFunction(s**3 - 2*s**2 - 3*s, s**4 + 1, s), TransferFunction(2, 1, s)),
                                      (TransferFunction(p, 1, s), TransferFunction(p, s, s)))))
    # 断言：对 H_1 进行 expand() 后应等于以下传递函数矩阵：
    #   - 第一行包含两个传递函数：
    #     - TransferFunction(s**3 - 2*s**2 - 3*s, s**4 + 1, s)
    #     - TransferFunction(2, 1, s)
    #   - 第二行包含两个传递函数：
    #     - TransferFunction(p, 1, s)
    #     - TransferFunction
def test_TransferFunction_gbt():
    # 定义一个简单的传递函数，例如欧姆定律
    tf = TransferFunction(1, a*s+b, s)
    # 使用 gbt 方法计算离散化传递函数的分子和分母系数
    numZ, denZ = gbt(tf, T, 0.5)
    # 使用来自 tf.gbt() 的系数创建离散化传递函数
    tf_test_bilinear = TransferFunction(s * numZ[0] + numZ[1], s * denZ[0] + denZ[1], s)
    # 使用手动计算的系数创建相应的传递函数
    tf_test_manual = TransferFunction(s * T/(2*(a + b*T/2)) + T/(2*(a + b*T/2)), s + (-a + b*T/2)/(a + b*T/2), s)

    assert S.Zero == (tf_test_bilinear.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    # 使用 gbt 方法计算离散化传递函数的分子和分母系数（另一种情况）
    numZ, denZ = gbt(tf, T, 0)
    # 使用来自 tf.gbt() 的系数创建离散化传递函数（另一种情况）
    tf_test_forward = TransferFunction(numZ[0], s*denZ[0]+denZ[1], s)
    # 使用手动计算的系数创建相应的传递函数（另一种情况）
    tf_test_manual = TransferFunction(T/a, s + (-a + b*T)/a, s)

    assert S.Zero == (tf_test_forward.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    # 使用 gbt 方法计算离散化传递函数的分子和分母系数（另一种情况）
    numZ, denZ = gbt(tf, T, 1)
    # 使用来自 tf.gbt() 的系数创建离散化传递函数（另一种情况）
    tf_test_backward = TransferFunction(s*numZ[0], s*denZ[0]+denZ[1], s)
    # 使用手动计算的系数创建相应的传递函数（另一种情况）
    tf_test_manual = TransferFunction(s * T/(a + b*T), s - a/(a + b*T), s)

    assert S.Zero == (tf_test_backward.simplify()-tf_test_manual.simplify()).simplify().num

    tf = TransferFunction(1, a*s+b, s)
    # 使用 gbt 方法计算离散化传递函数的分子和分母系数（另一种情况）
    numZ, denZ = gbt(tf, T, 0.3)
    # 使用来自 tf.gbt() 的系数创建离散化传递函数
    tf_test_gbt = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # 使用手动计算的系数创建相应的传递函数
    tf_test_manual = TransferFunction(s*3*T/(10*(a + 3*b*T/10)) + 7*T/(10*(a + 3*b*T/10)), s + (-a + 7*b*T/10)/(a + 3*b*T/10), s)

    assert S.Zero == (tf_test_gbt.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_bilinear():
    # 定义一个简单的传递函数，例如欧姆定律
    tf = TransferFunction(1, a*s+b, s)
    # 使用 bilinear 方法计算离散化传递函数的分子和分母系数
    numZ, denZ = bilinear(tf, T)
    # 使用来自 tf.bilinear() 的系数创建离散化传递函数
    tf_test_bilinear = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # 使用手动计算的系数创建相应的传递函数
    tf_test_manual = TransferFunction(s * T/(2*(a + b*T/2)) + T/(2*(a + b*T/2)), s + (-a + b*T/2)/(a + b*T/2), s)

    assert S.Zero == (tf_test_bilinear.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_forward_diff():
    # 定义一个简单的传递函数，例如欧姆定律
    tf = TransferFunction(1, a*s+b, s)
    # 使用 forward_diff 方法计算离散化传递函数的分子和分母系数
    numZ, denZ = forward_diff(tf, T)
    # 使用来自 tf.forward_diff() 的系数创建离散化传递函数
    tf_test_forward = TransferFunction(numZ[0], s*denZ[0]+denZ[1], s)
    # 使用手动计算的系数创建相应的传递函数
    tf_test_manual = TransferFunction(T/a, s + (-a + b*T)/a, s)

    assert S.Zero == (tf_test_forward.simplify()-tf_test_manual.simplify()).simplify().num
def test_TransferFunction_backward_diff():
    # 定义一个简单的传递函数，例如欧姆定律
    tf = TransferFunction(1, a*s+b, s)
    # 使用 backward_diff 函数计算离散化后的传递函数系数
    numZ, denZ = backward_diff(tf, T)
    # 使用从 tf.backward_diff() 获得的系数创建离散化的传递函数
    tf_test_backward = TransferFunction(s*numZ[0]+numZ[1], s*denZ[0]+denZ[1], s)
    # 使用手动计算的系数创建相应的传递函数
    tf_test_manual = TransferFunction(s * T/(a + b*T), s - a/(a + b*T), s)

    # 断言两个简化后的传递函数的差为零
    assert S.Zero == (tf_test_backward.simplify()-tf_test_manual.simplify()).simplify().num

def test_TransferFunction_phase_margin():
    # 测试相位裕度
    tf1 = TransferFunction(10, p**3 + 1, p)
    tf2 = TransferFunction(s**2, 10, s)
    tf3 = TransferFunction(1, a*s+b, s)
    tf4 = TransferFunction((s + 1)*exp(s/tau), s**2 + 2, s)
    tf_m = TransferFunctionMatrix([[tf2],[tf3]])

    # 断言相位裕度的计算结果
    assert phase_margin(tf1) == -180 + 180*atan(3*sqrt(11))/pi
    assert phase_margin(tf2) == 0

    # 检查未实现相位裕度计算的传递函数抛出异常
    raises(NotImplementedError, lambda: phase_margin(tf4))
    raises(ValueError, lambda: phase_margin(tf3))
    raises(ValueError, lambda: phase_margin(MIMOSeries(tf_m)))

def test_TransferFunction_gain_margin():
    # 测试增益裕度
    tf1 = TransferFunction(s**2, 5*(s+1)*(s-5)*(s-10), s)
    tf2 = TransferFunction(s**2 + 2*s + 1, 1, s)
    tf3 = TransferFunction(1, a*s+b, s)
    tf4 = TransferFunction((s + 1)*exp(s/tau), s**2 + 2, s)
    tf_m = TransferFunctionMatrix([[tf2],[tf3]])

    # 断言增益裕度的计算结果
    assert gain_margin(tf1) == -20*log(S(7)/540)/log(10)
    assert gain_margin(tf2) == oo

    # 检查未实现增益裕度计算的传递函数抛出异常
    raises(NotImplementedError, lambda: gain_margin(tf4))
    raises(ValueError, lambda: gain_margin(tf3))
    raises(ValueError, lambda: gain_margin(MIMOSeries(tf_m)))

def test_StateSpace_construction():
    # 使用不同的数字构建一个 SISO 系统
    A1 = Matrix([[0, 1], [1, 0]])
    B1 = Matrix([1, 0])
    C1 = Matrix([[0, 1]])
    D1 = Matrix([0])
    ss1 = StateSpace(A1, B1, C1, D1)

    # 断言状态空间的矩阵属性
    assert ss1.state_matrix == Matrix([[0, 1], [1, 0]])
    assert ss1.input_matrix == Matrix([1, 0])
    assert ss1.output_matrix == Matrix([[0, 1]])
    assert ss1.feedforward_matrix == Matrix([0])
    assert ss1.args == (Matrix([[0, 1], [1, 0]]), Matrix([[1], [0]]), Matrix([[0, 1]]), Matrix([[0]]))

    # 使用不同的符号构建一个 SISO 系统
    ss2 = StateSpace(Matrix([a0]), Matrix([a1]),
                    Matrix([a2]), Matrix([a3]))

    # 断言状态空间的矩阵属性
    assert ss2.state_matrix == Matrix([[a0]])
    assert ss2.input_matrix == Matrix([[a1]])
    assert ss2.output_matrix == Matrix([[a2]])
    assert ss2.feedforward_matrix == Matrix([[a3]])
    assert ss2.args == (Matrix([[a0]]), Matrix([[a1]]), Matrix([[a2]]), Matrix([[a3]]))

    # 使用不同的数字构建一个 MIMO 系统
    ss3 = StateSpace(Matrix([[-1.5, -2], [1, 0]]),
                    Matrix([[0.5, 0], [0, 1]]),
                    Matrix([[0, 1], [0, 2]]),
                    Matrix([[2, 2], [1, 1]]))

    # 断言状态空间的状态矩阵属性
    assert ss3.state_matrix == Matrix([[-1.5, -2], [1,  0]])
    # 断言确保输入矩阵的正确性
    assert ss3.input_matrix == Matrix([[0.5, 0], [0, 1]])
    # 断言确保输出矩阵的正确性
    assert ss3.output_matrix == Matrix([[0, 1], [0, 2]])
    # 断言确保前馈矩阵的正确性
    assert ss3.feedforward_matrix == Matrix([[2, 2], [1, 1]])
    # 断言确保参数元组的正确性
    assert ss3.args == (Matrix([[-1.5, -2],
                                [1,  0]]),
                        Matrix([[0.5, 0],
                                [0, 1]]),
                        Matrix([[0, 1],
                                [0, 2]]),
                        Matrix([[2, 2],
                                [1, 1]]))

    # 使用不同的符号定义一个MIMO系统
    A4 = Matrix([[a0, a1], [a2, a3]])
    B4 = Matrix([[b0, b1], [b2, b3]])
    C4 = Matrix([[c0, c1], [c2, c3]])
    D4 = Matrix([[d0, d1], [d2, d3]])
    ss4 = StateSpace(A4, B4, C4, D4)

    # 断言确保状态矩阵的正确性
    assert ss4.state_matrix == Matrix([[a0, a1], [a2, a3]])
    # 断言确保输入矩阵的正确性
    assert ss4.input_matrix == Matrix([[b0, b1], [b2, b3]])
    # 断言确保输出矩阵的正确性
    assert ss4.output_matrix == Matrix([[c0, c1], [c2, c3]])
    # 断言确保前馈矩阵的正确性
    assert ss4.feedforward_matrix == Matrix([[d0, d1], [d2, d3]])
    # 断言确保参数元组的正确性
    assert ss4.args == (Matrix([[a0, a1],
                                [a2, a3]]),
                        Matrix([[b0, b1],
                                [b2, b3]]),
                        Matrix([[c0, c1],
                                [c2, c3]]),
                        Matrix([[d0, d1],
                                [d2, d3]]))

    # 使用较少的矩阵。其余部分将用最少量的零填充。
    ss5 = StateSpace()
    # 断言确保参数元组的正确性
    assert ss5.args == (Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[0]]))

    A6 = Matrix([[0, 1], [1, 0]])
    B6 = Matrix([1, 1])
    ss6 = StateSpace(A6, B6)

    # 断言确保状态矩阵的正确性
    assert ss6.state_matrix == Matrix([[0, 1], [1, 0]])
    # 断言确保输入矩阵的正确性
    assert ss6.input_matrix ==  Matrix([1, 1])
    # 断言确保输出矩阵的正确性
    assert ss6.output_matrix == Matrix([[0, 0]])
    # 断言确保前馈矩阵的正确性
    assert ss6.feedforward_matrix == Matrix([[0]])
    # 断言确保参数元组的正确性
    assert ss6.args == (Matrix([[0, 1],
                                [1, 0]]),
                        Matrix([[1],
                                [1]]),
                        Matrix([[0, 0]]),
                        Matrix([[0]]))

    # 检查系统是SISO还是MIMO。
    # 如果系统不是SISO，则肯定是MIMO。

    assert ss1.is_SISO == True
    assert ss2.is_SISO == True
    assert ss3.is_SISO == False
    assert ss4.is_SISO == False
    assert ss5.is_SISO == True
    assert ss6.is_SISO == True

    # 如果矩阵尺寸不匹配，则会引发ShapeError异常
    raises(ShapeError, lambda: StateSpace(Matrix([s, (s+1)**2]), Matrix([s+1]),
                                          Matrix([s**2 - 1]), Matrix([2*s])))
    raises(ShapeError, lambda: StateSpace(Matrix([s]), Matrix([s+1, s**3 + 1]),
                                          Matrix([s**2 - 1]), Matrix([2*s])))
    raises(ShapeError, lambda: StateSpace(Matrix([s]), Matrix([s+1]),
                                          Matrix([[s**2 - 1], [s**2 + 2*s + 1]]), Matrix([2*s])))
    # 如果传入的状态空间参数不符合预期的形状，会抛出 ShapeError 异常
    raises(ShapeError, lambda: StateSpace(Matrix([[-s, -s], [s, 0]]),
                                          Matrix([[s/2, 0], [0, s]]),
                                          Matrix([[0, s]]),
                                          Matrix([[2*s, 2*s], [s, s]])))
    
    # 如果传入的参数不是 sympy 矩阵对象，则会抛出 TypeError 异常
    raises(TypeError, lambda: StateSpace(s**2, s+1, 2*s, 1))
    
    # 如果传入的参数不是 sympy 矩阵对象，则会抛出 TypeError 异常
    raises(TypeError, lambda: StateSpace(Matrix([2, 0.5]), Matrix([-1]),
                                         Matrix([1]), 0))
def test_StateSpace_add():
    # 创建状态空间系统 ss1
    A1 = Matrix([[4, 1],[2, -3]])
    B1 = Matrix([[5, 2],[-3, -3]])
    C1 = Matrix([[2, -4],[0, 1]])
    D1 = Matrix([[3, 2],[1, -1]])
    ss1 = StateSpace(A1, B1, C1, D1)

    # 创建状态空间系统 ss2
    A2 = Matrix([[-3, 4, 2],[-1, -3, 0],[2, 5, 3]])
    B2 = Matrix([[1, 4],[-3, -3],[-2, 1]])
    C2 = Matrix([[4, 2, -3],[1, 4, 3]])
    D2 = Matrix([[-2, 4],[0, 1]])
    ss2 = StateSpace(A2, B2, C2, D2)

    # 创建空状态空间系统 ss3
    ss3 = StateSpace()

    # 创建状态空间系统 ss4，包含单元素矩阵
    ss4 = StateSpace(Matrix([1]), Matrix([2]), Matrix([3]), Matrix([4]))

    # 预期的状态空间系统相加结果 expected_add
    expected_add = \
        StateSpace(
        Matrix([
        [4,  1,  0,  0, 0],
        [2, -3,  0,  0, 0],
        [0,  0, -3,  4, 2],
        [0,  0, -1, -3, 0],
        [0,  0,  2,  5, 3]]),
        Matrix([
        [ 5,  2],
        [-3, -3],
        [ 1,  4],
        [-3, -3],
        [-2,  1]]),
        Matrix([
        [2, -4, 4, 2, -3],
        [0,  1, 1, 4,  3]]),
        Matrix([
        [1, 6],
        [1, 0]]))

    # 预期的状态空间系统相乘结果 expected_mul
    expected_mul = \
        StateSpace(
        Matrix([
        [ -3,   4,  2, 0,  0],
        [ -1,  -3,  0, 0,  0],
        [  2,   5,  3, 0,  0],
        [ 22,  18, -9, 4,  1],
        [-15, -18,  0, 2, -3]]),
        Matrix([
        [  1,   4],
        [ -3,  -3],
        [ -2,   1],
        [-10,  22],
        [  6, -15]]),
        Matrix([
        [14, 14, -3, 2, -4],
        [ 3, -2, -6, 0,  1]]),
        Matrix([
        [-6, 14],
        [-2,  3]]))

    # 断言相加和相乘操作的结果与预期相符
    assert ss1 + ss2 == expected_add
    assert ss1*ss2 == expected_mul
    # 断言空状态空间系统加上标量的结果
    assert ss3 + 1/2 == StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[0.5]]))
    # 断言状态空间系统乘以标量的结果
    assert ss4*1.5 == StateSpace(Matrix([[1]]), Matrix([[2]]), Matrix([[4.5]]), Matrix([[6.0]]))
    # 断言标量乘以状态空间系统的结果
    assert 1.5*ss4 == StateSpace(Matrix([[1]]), Matrix([[3.0]]), Matrix([[3]]), Matrix([[6.0]]))
    # 断言形状错误引发异常
    raises(ShapeError, lambda: ss1 + ss3)
    raises(ShapeError, lambda: ss2*ss4)

def test_StateSpace_negation():
    # 创建包含符号参数的状态空间系统 SS
    A = Matrix([[a0, a1], [a2, a3]])
    B = Matrix([[b0, b1], [b2, b3]])
    C = Matrix([[c0, c1], [c1, c2], [c2, c3]])
    D = Matrix([[d0, d1], [d1, d2], [d2, d3]])
    SS = StateSpace(A, B, C, D)
    # 计算 SS 的负
    SS_neg = -SS

    # 创建另一个状态空间系统 system
    state_mat = Matrix([[-1, 1], [1, -1]])
    input_mat = Matrix([1, -1])
    output_mat = Matrix([[-1, 1]])
    feedforward_mat = Matrix([1])
    system = StateSpace(state_mat, input_mat, output_mat, feedforward_mat)

    # 断言 SS 的负与预期结果相符
    assert SS_neg == \
        StateSpace(Matrix([[a0, a1],
                           [a2, a3]]),
                   Matrix([[b0, b1],
                           [b2, b3]]),
                   Matrix([[-c0, -c1],
                           [-c1, -c2],
                           [-c2, -c3]]),
                   Matrix([[-d0, -d1],
                           [-d1, -d2],
                           [-d2, -d3]]))
    # 断言 system 的负与预期结果相符
    assert -system == \
        StateSpace(Matrix([[-1,  1],
                           [ 1, -1]]),
                   Matrix([[ 1],[-1]]),
                   Matrix([[1, -1]]),
                   Matrix([[-1]]))
    # 断言 SS_neg 的负等于 SS 自身
    assert -SS_neg == SS
    # 对系统变量进行多次负负取正操作，实际效果是取系统变量的值本身
    assert -(-(-(-system))) == system
def test_SymPy_substitution_functions():
    # 创建两个 StateSpace 对象 ss1 和 ss2
    ss1 = StateSpace(Matrix([s]), Matrix([(s + 1)**2]), Matrix([s**2 - 1]), Matrix([2*s]))
    ss2 = StateSpace(Matrix([s + p]), Matrix([(s + 1)*(p - 1)]), Matrix([p**3 - s**3]), Matrix([s - p]))

    # 断言 ss1 在将 s 替换为 5 后的结果
    assert ss1.subs({s:5}) == StateSpace(Matrix([[5]]), Matrix([[36]]), Matrix([[24]]), Matrix([[10]]))
    # 断言 ss2 在将 p 替换为 1 后的结果
    assert ss2.subs({p:1}) == StateSpace(Matrix([[s + 1]]), Matrix([[0]]), Matrix([[1 - s**3]]), Matrix([[s - 1]]))

    # 断言 ss1 在将 s 替换为 p 后的结果
    assert ss1.xreplace({s:p}) == \
        StateSpace(Matrix([[p]]), Matrix([[(p + 1)**2]]), Matrix([[p**2 - 1]]), Matrix([[2*p]]))
    # 断言 ss2 在将 s 替换为 a, p 替换为 b 后的结果
    assert ss2.xreplace({s:a, p:b}) == \
        StateSpace(Matrix([[a + b]]), Matrix([[(a + 1)*(b - 1)]]), Matrix([[-a**3 + b**3]]), Matrix([[a - b]]))

    # 创建 StateSpace 对象 G，并定义两个多项式 p1 和 p2
    p1 = a1*s + a0
    p2 = b2*s**2 + b1*s + b0
    G = StateSpace(Matrix([p1]), Matrix([p2]))
    # 创建预期结果 expect 和 expect_
    expect = StateSpace(Matrix([[2*s + 1]]), Matrix([[5*s**2 + 4*s + 3]]), Matrix([[0]]), Matrix([[0]]))
    expect_ = StateSpace(Matrix([[2.0*s + 1.0]]), Matrix([[5.0*s**2 + 4.0*s + 3.0]]), Matrix([[0]]), Matrix([[0]]))
    # 断言 G 在给定系数后的结果
    assert G.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}) == expect
    # 断言 G 在给定系数后进行 evalf() 后的结果
    assert G.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}).evalf() == expect_
    # 断言 expect 进行 evalf() 后的结果
    assert expect.evalf() == expect_

def test_conversion():
    # 创建 StateSpace 对象 H1
    A1 = Matrix([[-5, -1], [3, -1]])
    B1 = Matrix([2, 5])
    C1 = Matrix([[1, 2]])
    D1 = Matrix([0])
    H1 = StateSpace(A1, B1, C1, D1)
    # 将 H1 转换为 TransferFunction 对象 tm1 和 tm2
    tm1 = H1.rewrite(TransferFunction)
    tm2 = (-H1).rewrite(TransferFunction)

    # 提取 tm1 和 tm2 的传递函数 tf1 和 tf2
    tf1 = tm1[0][0]
    tf2 = tm2[0][0]

    # 断言 tf1 的值符合预期
    assert tf1 == TransferFunction(12*s + 59, s**2 + 6*s + 8, s)
    # 断言 tf2 的分子为 tf1 的相反数
    assert tf2.num == -tf1.num
    # 断言 tf2 的分母与 tf1 相同
    assert tf2.den == tf1.den

    # 创建 StateSpace 对象 H2
    A2 = Matrix([[-1.5, -2, 3], [1, 0, 1], [2, 1, 1]])
    B2 = Matrix([[0.5, 0, 1], [0, 1, 2], [2, 2, 3]])
    C2 = Matrix([[0, 1, 0], [0, 2, 1], [1, 0, 2]])
    D2 = Matrix([[2, 2, 0], [1, 1, 1], [3, 2, 1]])
    H2 = StateSpace(A2, B2, C2, D2)
    # 将 H2 转换为 TransferFunction 对象 tm3
    tm3 = H2.rewrite(TransferFunction)

    # 断言 tm3 的一些元素转换为 TransferFunction 后的预期值
    assert tm3[0][0] == TransferFunction(2.0*s**3 + 1.0*s**2 - 10.5*s + 4.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)
    assert tm3[0][1] == TransferFunction(2.0*s**3 + 2.0*s**2 - 10.5*s - 3.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)
    assert tm3[0][2] == TransferFunction(2.0*s**2 + 5.0*s - 0.5, 1.0*s**3 + 0.5*s**2 - 6.5*s - 2.5, s)

    # 断言 StateSpace 对象 SS 的重写结果
    SS = TF1.rewrite(StateSpace)
    assert SS == \
        StateSpace(Matrix([[     0,          1],
                           [-wn**2, -2*wn*zeta]]),
                   Matrix([[0],
                           [1]]),
                   Matrix([[1, 0]]),
                   Matrix([[0]]))
    # 断言 SS 重新转换为 TransferFunction 后的结果
    assert SS.rewrite(TransferFunction)[0][0] == TF1

    # Transfer function has to be proper
    # 使用 raises 函数测试是否会抛出 ValueError 异常，lambda 函数内调用 TransferFunction 对象的 rewrite 方法，并将其传入 StateSpace 参数
    raises(ValueError, lambda: TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s).rewrite(StateSpace))
# 定义测试函数 test_StateSpace_dsolve，用于测试 StateSpace 类的 dsolve 方法
def test_StateSpace_dsolve():
    # 设置链接参考文档：https://web.mit.edu/2.14/www/Handouts/StateSpaceResponse.pdf
    # 设置链接参考文档：https://lpsa.swarthmore.edu/Transient/TransMethSS.html
    
    # 定义状态空间系统的矩阵 A1
    A1 = Matrix([[0, 1], [-2, -3]])
    # 定义输入矩阵 B1
    B1 = Matrix([[0], [1]])
    # 定义输出矩阵 C1
    C1 = Matrix([[1, -1]])
    # 定义直通矩阵 D1
    D1 = Matrix([0])
    # 定义初始条件矩阵 I1
    I1 = Matrix([[1], [2]])
    # 定义符号变量 t
    t = symbols('t')
    # 创建 StateSpace 对象 ss1，传入 A1, B1, C1, D1 参数
    ss1 = StateSpace(A1, B1, C1, D1)
    
    # 测试零输入和零初始条件的 dsolve 方法，预期返回一个零矩阵
    assert ss1.dsolve() == Matrix([[0]])
    # 测试带有初始条件 I1 的 dsolve 方法，预期返回一个包含表达式的矩阵
    assert ss1.dsolve(initial_conditions=I1) == Matrix([[8*exp(-t) - 9*exp(-2*t)]])

    # 定义状态空间系统的矩阵 A2
    A2 = Matrix([[-2, 0], [1, -1]])
    # 定义输出矩阵 C2
    C2 = eye(2,2)
    # 定义初始条件矩阵 I2
    I2 = Matrix([2, 3])
    # 创建 StateSpace 对象 ss2，传入 A2, None, C2, None 参数
    ss2 = StateSpace(A=A2, C=C2)
    # 测试带有初始条件 I2 的 dsolve 方法，预期返回一个包含表达式的矩阵
    assert ss2.dsolve(initial_conditions=I2) == Matrix([[2*exp(-2*t)], [5*exp(-t) - 2*exp(-2*t)]])

    # 定义状态空间系统的矩阵 A3
    A3 = Matrix([[-1, 1], [-4, -4]])
    # 定义输入矩阵 B3
    B3 = Matrix([[0], [4]])
    # 定义输出矩阵 C3
    C3 = Matrix([[0, 1]])
    # 定义直通矩阵 D3
    D3 = Matrix([0])
    # 定义输入向量 U3
    U3 = Matrix([10])
    # 创建 StateSpace 对象 ss3，传入 A3, B3, C3, D3 参数
    ss3 = StateSpace(A3, B3, C3, D3)
    # 使用 input_vector=U3 和 var=t 参数调用 dsolve 方法，存储结果到 op 变量
    op = ss3.dsolve(input_vector=U3, var=t)
    # 断言 op 的简化、展开、数值化的第一个元素满足预期表达式
    assert str(op.simplify().expand().evalf()[0]) == str(5.0 + 20.7880460155075*exp(-5*t/2)*sin(sqrt(7)*t/2)
                                            - 5.0*exp(-5*t/2)*cos(sqrt(7)*t/2))
    
    # 定义状态空间系统的矩阵 A4
    A4 = Matrix([[-1, 1], [-4, -4]])
    # 定义输入矩阵 B4
    B4 = Matrix([[0], [4]])
    # 定义输出矩阵 C4
    C4 = Matrix([[0, 1]])
    # 定义输入向量 U4，包含 Heaviside(t) 函数
    U4 = Matrix([[10*Heaviside(t)]])
    # 创建 StateSpace 对象 ss4，传入 A4, B4, C4, None 参数
    ss4 = StateSpace(A4, B4, C4)
    # 使用 var=t 和 input_vector=U4 调用 dsolve 方法，并将结果字符串化后存储到 op4 变量
    op4 = str(ss4.dsolve(var=t, input_vector=U4)[0].simplify().expand().evalf())
    # 断言 op4 符合预期的表达式
    assert op4 == str(5.0*Heaviside(t) + 20.7880460155075*exp(-5*t/2)*sin(sqrt(7)*t/2)*Heaviside(t)
                                            - 5.0*exp(-5*t/2)*cos(sqrt(7)*t/2)*Heaviside(t))
    
    # 定义符号变量 m, a, x0
    m, a, x0 = symbols('m a x_0')
    # 定义状态空间系统的矩阵 A5
    A5 = Matrix([[0, 1], [0, 0]])
    # 定义输入矩阵 B5
    B5 = Matrix([[0], [1 / m]])
    # 定义输出矩阵 C5
    C5 = Matrix([[1, 0]])
    # 定义初始条件矩阵 I5
    I5 = Matrix([[x0], [0]])
    # 定义输入向量 U5
    U5 = Matrix([[exp(-a * t)]])
    # 创建 StateSpace 对象 ss5，传入 A5, B5, C5, None 参数
    ss5 = StateSpace(A5, B5, C5)
    # 使用 initial_conditions=I5, input_vector=U5, var=t 调用 dsolve 方法并简化结果，存储到 op5 变量
    op5 = ss5.dsolve(initial_conditions=I5, input_vector=U5, var=t).simplify()
    # 断言 op5 的第一个元素满足预期表达式
    assert op5[0].args[0][0] == x0 + t/(a*m) - 1/(a**2*m) + exp(-a*t)/(a**2*m)
    
    # 定义符号变量 a11, a12, a21, a22, b1, b2, c1, c2, i1, i2
    a11, a12, a21, a22, b1, b2, c1, c2, i1, i2 = symbols('a_11 a_12 a_21 a_22 b_1 b_2 c_1 c_2 i_1 i_2')
    # 定义状态空间系统的矩阵 A6
    A6 = Matrix([[a11, a12], [a21, a22]])
    # 定义输入矩阵 B6
    B6 = Matrix([b1, b2])
    # 定义输出矩阵 C6
    C6 = Matrix([[c1, c2]])
    # 定义初始条件矩阵 I6
    I6 = Matrix([i1, i2])
    # 创建 StateSpace 对象 ss6，传入 A6, B6, C6, None 参数
    ss6 = StateSpace(A6, B6, C6)
    # 调用 dsolve 方法并简化结果，存储到 expr6 变量
    expr6 = ss6.dsolve(initial_conditions=I6)[0]
    # 使用 subs 方法替换表达式中的符号变量，并将结果存储回 expr6 变量
    expr6 = expr6.subs([(a11, 0), (a12, 1), (a21, -2), (a22, -3), (b1, 0), (b2, 1), (c1, 1), (c2, -1), (i1, 1), (i2, 2)])
    # 断言 expr6 符合预期的表达式
    assert expr6 == 8*exp(-t) - 9*exp(-2*t)


# 定义测试函数 test_StateSpace_functions，用于测试 StateSpace 类的其他功能
def test_StateSpace_functions():
    # 设置链接参考文档：https://in.mathworks.com
    # 确保 SS1 可观测
    assert SS1.is_observable() == True
    # 确保 SS2 不可观测
    assert SS2.is_observable() == False
    # 获取 SS1 的可观测性矩阵
    assert SS1.observability_matrix() == Matrix([[0, 1], [1, 0]])
    # 获取 SS2 的可观测性矩阵
    assert SS2.observability_matrix() == Matrix([[-1,  1], [ 1, -1], [ 3, -3], [-3,  3]])
    # 获取 SS1 的可观测子空间
    assert SS1.observable_subspace() == [Matrix([[0], [1]]), Matrix([[1], [0]])]
    # 获取 SS2 的可观测子空间
    assert SS2.observable_subspace() == [Matrix([[-1], [ 1], [ 3], [-3]])]

    # 可控性检查
    # 确保 SS1 可控
    assert SS1.is_controllable() == True
    # 确保 SS3 不可控
    assert SS3.is_controllable() == False
    # 获取 SS1 的可控性矩阵
    assert SS1.controllability_matrix() ==  Matrix([[0.5, -0.75], [  0,   0.5]])
    # 获取 SS3 的可控性矩阵
    assert SS3.controllability_matrix() == Matrix([[1, -1, 2, -2], [1, -1, 2, -2]])
    # 获取 SS1 的可控子空间
    assert SS1.controllable_subspace() == [Matrix([[0.5], [  0]]), Matrix([[-0.75], [  0.5]])]
    # 获取 SS3 的可控子空间
    assert SS3.controllable_subspace() == [Matrix([[1], [1]])]

    # 追加操作
    # 定义状态空间模型 A1
    A1 = Matrix([[0, 1], [1, 0]])
    # 定义输入矩阵 B1
    B1 = Matrix([[0], [1]])
    # 定义输出矩阵 C1
    C1 = Matrix([[0, 1]])
    # 定义前馈矩阵 D1
    D1 = Matrix([[0]])
    # 创建状态空间对象 ss1
    ss1 = StateSpace(A1, B1, C1, D1)
    # 创建状态空间对象 ss2
    ss2 = StateSpace(Matrix([[1, 0], [0, 1]]), Matrix([[1], [0]]), Matrix([[1, 0]]), Matrix([[1]]))
    # 将 ss2 追加到 ss1，形成新的状态空间对象 ss3
    ss3 = ss1.append(ss2)

    # 确保 ss3 的状态数等于 ss1 和 ss2 的状态数之和
    assert ss3.num_states == ss1.num_states + ss2.num_states
    # 确保 ss3 的输入数等于 ss1 和 ss2 的输入数之和
    assert ss3.num_inputs == ss1.num_inputs + ss2.num_inputs
    # 确保 ss3 的输出数等于 ss1 和 ss2 的输出数之和
    assert ss3.num_outputs == ss1.num_outputs + ss2.num_outputs
    # 确保 ss3 的状态矩阵正确
    assert ss3.state_matrix == Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # 确保 ss3 的输入矩阵正确
    assert ss3.input_matrix == Matrix([[0, 0], [1, 0], [0, 1], [0, 0]])
    # 确保 ss3 的输出矩阵正确
    assert ss3.output_matrix == Matrix([[0, 1, 0, 0], [0, 0, 1, 0]])
    # 确保 ss3 的前馈矩阵正确
    assert ss3.feedforward_matrix == Matrix([[0, 0], [0, 1]])
def test_StateSpace_series():
    # For SISO Systems
    # 定义第一个 SISO 系统的状态空间模型
    a1 = Matrix([[0, 1], [1, 0]])
    b1 = Matrix([[0], [1]])
    c1 = Matrix([[0, 1]])
    d1 = Matrix([[0]])
    # 定义第二个 SISO 系统的状态空间模型
    a2 = Matrix([[1, 0], [0, 1]])
    b2 = Matrix([[1], [0]])
    c2 = Matrix([[1, 0]])
    d2 = Matrix([[1]])

    # 创建第一个状态空间对象 ss1
    ss1 = StateSpace(a1, b1, c1, d1)
    # 创建第二个状态空间对象 ss2
    ss2 = StateSpace(a2, b2, c2, d2)
    
    # 创建传递函数对象 tf1
    tf1 = TransferFunction(s, s + 1, s)
    
    # 创建 ss1 和 ss2 的串联对象 ser1，并进行断言检查
    ser1 = Series(ss1, ss2)
    assert ser1 == Series(StateSpace(Matrix([
                            [0, 1],
                            [1, 0]]), Matrix([
                            [0],
                            [1]]), Matrix([[0, 1]]), Matrix([[0]])), StateSpace(Matrix([
                            [1, 0],
                            [0, 1]]), Matrix([
                            [1],
                            [0]]), Matrix([[1, 0]]), Matrix([[1]])))
    
    # 对 ser1 进行 doit() 操作，并进行断言检查
    assert ser1.doit() == StateSpace(
                            Matrix([
                            [0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 1]]),
                            Matrix([
                            [0],
                            [1],
                            [0],
                            [0]]),
                            Matrix([[0, 1, 1, 0]]),
                            Matrix([[0]]))

    # 检查 ser1 的输入数量是否为 1
    assert ser1.num_inputs == 1
    # 检查 ser1 的输出数量是否为 1
    assert ser1.num_outputs == 1
    # 对 ser1 以传递函数形式重写，并进行断言检查
    assert ser1.rewrite(TransferFunction) == TransferFunction(s**2, s**3 - s**2 - s + 1, s)
    
    # 创建 ser2，其中包含 ss1 的串联对象
    ser2 = Series(ss1)
    # 创建 ser3，由 ser2 和 ss2 构成的串联对象，并进行断言检查
    ser3 = Series(ser2, ss2)
    assert ser3.doit() == ser1.doit()
    
    # 创建 ser_tf，其中包含传递函数对象 tf1 和 ss1 的串联对象，并进行断言检查
    ser_tf = Series(tf1, ss1)
    assert ser_tf == Series(TransferFunction(s, s + 1, s), StateSpace(Matrix([
                            [0, 1],
                            [1, 0]]), Matrix([
                            [0],
                            [1]]), Matrix([[0, 1]]), Matrix([[0]])))
    
    # 对 ser_tf 进行 doit() 操作，并进行断言检查
    assert ser_tf.doit() == StateSpace(
                            Matrix([
                            [-1, 0,  0],
                            [0, 0,  1],
                            [-1, 1, 0]]),
                            Matrix([
                            [1],
                            [0],
                            [1]]),
                            Matrix([[0, 0, 1]]),
                            Matrix([[0]]))
    
    # 对 ser_tf 以传递函数形式重写，并进行断言检查
    assert ser_tf.rewrite(TransferFunction) == TransferFunction(s**2, s**3 + s**2 - s - 1, s)
    
    # For MIMO Systems
    # 定义第一个 MIMO 系统的状态空间模型
    a3 = Matrix([[4, 1], [2, -3]])
    b3 = Matrix([[5, 2], [-3, -3]])
    c3 = Matrix([[2, -4], [0, 1]])
    d3 = Matrix([[3, 2], [1, -1]])
    # 定义第二个 MIMO 系统的状态空间模型
    a4 = Matrix([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]])
    b4 = Matrix([[1, 4], [-3, -3], [-2, 1]])
    c4 = Matrix([[4, 2, -3], [1, 4, 3]])
    d4 = Matrix([[-2, 4], [0, 1]])
    
    # 创建第三个和第四个状态空间对象 ss3 和 ss4
    ss3 = StateSpace(a3, b3, c3, d3)
    ss4 = StateSpace(a4, b4, c4, d4)
    
    # 创建 ss3 和 ss4 的 MIMO 系统的串联对象 ser4
    ser4 = MIMOSeries(ss3, ss4)
    # 确定 ser4 对象与给定的 StateSpace 对象是否相等
    assert ser4 == MIMOSeries(StateSpace(Matrix([
                    [4,  1],
                    [2, -3]]), Matrix([
                    [ 5,  2],
                    [-3, -3]]), Matrix([
                    [2, -4],
                    [0,  1]]), Matrix([
                    [3,  2],
                    [1, -1]])), StateSpace(Matrix([
                    [-3,  4, 2],
                    [-1, -3, 0],
                    [ 2,  5, 3]]), Matrix([
                    [ 1,  4],
                    [-3, -3],
                    [-2,  1]]), Matrix([
                    [4, 2, -3],
                    [1, 4,  3]]), Matrix([
                    [-2, 4],
                    [ 0, 1]])))
    
    # 对 ser4 对象应用 doit() 方法，验证其状态空间的转换结果是否符合预期
    assert ser4.doit() == StateSpace(
                        Matrix([
                        [4,   1,  0, 0,  0],
                        [2,  -3,  0, 0,  0],
                        [2,   0,  -3, 4,  2],
                        [-6,  9, -1, -3,  0],
                        [-4, 9,  2, 5, 3]]),
                        Matrix([
                        [5,   2],
                        [-3,  -3],
                        [7,   -2],
                        [-12,  -3],
                        [-5, -5]]),
                        Matrix([
                        [-4, 12, 4, 2, -3],
                        [0, 1, 1, 4, 3]]),
                        Matrix([
                        [-2, -8],
                        [1, -1]]))
    
    # 检查 ser4 和 ss3 对象的输入端口数是否相等
    assert ser4.num_inputs == ss3.num_inputs
    
    # 检查 ser4 和 ss4 对象的输出端口数是否相等
    assert ser4.num_outputs == ss4.num_outputs
    
    # 创建 ser5 对象，作为 ss3 的串联
    ser5 = MIMOSeries(ss3)
    
    # 创建 ser6 对象，作为 ser5 和 ss4 的串联
    ser6 = MIMOSeries(ser5, ss4)
    
    # 验证 ser6 的 doit() 方法结果与 ser4 的结果是否相等
    assert ser6.doit() == ser4.doit()
    
    # 验证 ser6 的 rewrite 方法应用 TransferFunctionMatrix 后的结果与 ser4 的结果是否相等
    assert ser6.rewrite(TransferFunctionMatrix) == ser4.rewrite(TransferFunctionMatrix)
    
    # 创建三个 TransferFunction 对象 tf2, tf3, tf4
    tf2 = TransferFunction(1, s, s)
    tf3 = TransferFunction(1, s+1, s)
    tf4 = TransferFunction(s, s+2, s)
    
    # 创建 TransferFunctionMatrix 对象 tfm，包含 tf1, tf2, tf3, tf4
    tfm = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    
    # 创建 ser6 对象，作为 ss3 和 tfm 的串联
    ser6 = MIMOSeries(ss3, tfm)
    
    # 验证 ser6 对象是否与给定的 MIMOSeries 对象相等
    assert ser6 == MIMOSeries(StateSpace(Matrix([
                        [4,  1],
                        [2, -3]]), Matrix([
                        [ 5,  2],
                        [-3, -3]]), Matrix([
                        [2, -4],
                        [0,  1]]), Matrix([
                        [3,  2],
                        [1, -1]])), TransferFunctionMatrix((
                        (TransferFunction(s, s + 1, s), TransferFunction(1, s, s)),
                        (TransferFunction(1, s + 1, s), TransferFunction(s, s + 2, s)))))
def test_StateSpace_parallel():
    # For SISO system

    # 定义第一个单输入单输出系统的状态空间模型
    a1 = Matrix([[0, 1], [1, 0]])
    b1 = Matrix([[0], [1]])
    c1 = Matrix([[0, 1]])
    d1 = Matrix([[0]])

    # 定义第二个单输入单输出系统的状态空间模型
    a2 = Matrix([[1, 0], [0, 1]])
    b2 = Matrix([[1], [0]])
    c2 = Matrix([[1, 0]])
    d2 = Matrix([[1]])

    # 创建第一个状态空间对象
    ss1 = StateSpace(a1, b1, c1, d1)
    # 创建第二个状态空间对象
    ss2 = StateSpace(a2, b2, c2, d2)

    # 将两个状态空间对象并联
    p1 = Parallel(ss1, ss2)

    # 验证并联后的状态空间对象是否与预期相同
    assert p1 == Parallel(StateSpace(Matrix([[0, 1], [1, 0]]), Matrix([[0], [1]]), Matrix([[0, 1]]), Matrix([[0]])),
                          StateSpace(Matrix([[1, 0], [0, 1]]), Matrix([[1], [0]]), Matrix([[1, 0]]), Matrix([[1]])))

    # 对并联后的系统执行求解操作，验证结果是否与预期相同
    assert p1.doit() == StateSpace(Matrix([
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]),
                        Matrix([
                        [0],
                        [1],
                        [1],
                        [0]]),
                        Matrix([[0, 1, 1, 0]]),
                        Matrix([[1]]))

    # 将并联后的系统重写为传递函数形式，验证结果是否与预期相同
    assert p1.rewrite(TransferFunction) == TransferFunction(s*(s + 2), s**2 - 1, s)

    # Connecting StateSpace with TransferFunction
    # 将状态空间系统与传递函数进行连接

    # 创建一个传递函数对象
    tf1 = TransferFunction(s, s+1, s)
    # 将状态空间系统ss1与传递函数tf1并联
    p2 = Parallel(ss1, tf1)

    # 验证并联后的系统是否与预期相同
    assert p2 == Parallel(StateSpace(Matrix([
                        [0, 1],
                        [1, 0]]), Matrix([
                        [0],
                        [1]]), Matrix([[0, 1]]), Matrix([[0]])), TransferFunction(s, s + 1, s))

    # 对并联后的系统执行求解操作，验证结果是否与预期相同
    assert p2.doit() == StateSpace(
                        Matrix([
                        [0, 1,  0],
                        [1, 0,  0],
                        [0, 0, -1]]),
                        Matrix([
                        [0],
                        [1],
                        [1]]),
                        Matrix([[0, 1, -1]]),
                        Matrix([[1]]))

    # 将并联后的系统重写为传递函数形式，验证结果是否与预期相同
    assert p2.rewrite(TransferFunction) == TransferFunction(s**2, s**2 - 1, s)

    # For MIMO
    # 多输入多输出系统

    # 定义第一个多输入多输出系统的状态空间模型
    a3 = Matrix([[4, 1], [2, -3]])
    b3 = Matrix([[5, 2], [-3, -3]])
    c3 = Matrix([[2, -4], [0, 1]])
    d3 = Matrix([[3, 2], [1, -1]])

    # 定义第二个多输入多输出系统的状态空间模型
    a4 = Matrix([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]])
    b4 = Matrix([[1, 4], [-3, -3], [-2, 1]])
    c4 = Matrix([[4, 2, -3], [1, 4, 3]])
    d4 = Matrix([[-2, 4], [0, 1]])

    # 创建第一个多输入多输出系统的状态空间对象
    ss3 = StateSpace(a3, b3, c3, d3)
    # 创建第二个多输入多输出系统的状态空间对象
    ss4 = StateSpace(a4, b4, c4, d4)

    # 将两个多输入多输出系统进行并联
    p3 = MIMOParallel(ss3, ss4)
    # 断言语句，验证 p3 是否等于使用 MIMOParallel 构建的 StateSpace 对象
    assert p3 == MIMOParallel(
        StateSpace(Matrix([
            [4,  1],
            [2, -3]]),
        Matrix([
            [ 5,  2],
            [-3, -3]]),
        Matrix([
            [2, -4],
            [0,  1]]),
        Matrix([
            [3,  2],
            [1, -1]])),
        StateSpace(Matrix([
            [-3,  4,  2],
            [-1, -3,  0],
            [ 2,  5,  3]]),
        Matrix([
            [ 1,  4],
            [-3, -3],
            [-2,  1]]),
        Matrix([
            [ 4,  2, -3],
            [ 1,  4,  3]]),
        Matrix([
            [-2,  4],
            [ 0,  1]])
    )

    # 断言语句，验证 p3.doit() 返回的 StateSpace 对象是否等于指定的 StateSpace 对象
    assert p3.doit() == StateSpace(
        Matrix([
            [4, 1, 0, 0, 0],
            [2, -3, 0, 0, 0],
            [0, 0, -3, 4, 2],
            [0, 0, -1, -3, 0],
            [0, 0, 2, 5, 3]]),
        Matrix([
            [5, 2],
            [-3, -3],
            [1, 4],
            [-3, -3],
            [-2, 1]]),
        Matrix([
            [2, -4, 4, 2, -3],
            [0, 1, 1, 4, 3]]),
        Matrix([
            [1, 6],
            [1, 0]])
    )

    # 使用 StateSpace 和 MIMOParallel 创建对象 p4
    tf2 = TransferFunction(1, s, s)
    tf3 = TransferFunction(1, s + 1, s)
    tf4 = TransferFunction(s, s + 2, s)
    tfm = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    p4 = MIMOParallel(tfm, ss3)

    # 断言语句，验证 p4 是否等于使用 MIMOParallel 构建的 StateSpace 对象
    assert p4 == MIMOParallel(
        TransferFunctionMatrix((
            (TransferFunction(s, s + 1, s), TransferFunction(1, s, s)),
            (TransferFunction(1, s + 1, s), TransferFunction(s, s + 2, s)))),
        StateSpace(Matrix([
            [4, 1],
            [2, -3]]),
        Matrix([
            [5, 2],
            [-3, -3]]),
        Matrix([
            [2, -4],
            [0, 1]]),
        Matrix([
            [3, 2],
            [1, -1]])
    )
```