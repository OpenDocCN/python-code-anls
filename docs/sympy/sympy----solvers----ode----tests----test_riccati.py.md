# `D:\src\scipysrc\sympy\sympy\solvers\ode\tests\test_riccati.py`

```
from sympy.core.random import randint  # 导入随机整数生成函数
from sympy.core.function import Function  # 导入函数相关的基础类
from sympy.core.mul import Mul  # 导入乘法运算相关类
from sympy.core.numbers import (I, Rational, oo)  # 导入虚数单位、有理数和无穷大的表示
from sympy.core.relational import Eq  # 导入等式类
from sympy.core.singleton import S  # 导入单例类
from sympy.core.symbol import (Dummy, symbols)  # 导入虚拟符号和符号类
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.hyperbolic import tanh  # 导入双曲正切函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import sin  # 导入正弦函数
from sympy.polys.polytools import Poly  # 导入多项式操作工具类
from sympy.simplify.ratsimp import ratsimp  # 导入有理化简函数
from sympy.solvers.ode.subscheck import checkodesol  # 导入常微分方程解检查函数
from sympy.testing.pytest import slow  # 导入测试框架中的慢速标记
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
    riccati_reduced, match_riccati, inverse_transform_poly, limit_at_inf,
    check_necessary_conds, val_at_inf, construct_c_case_1,
    construct_c_case_2, construct_c_case_3, construct_d_case_4,
    construct_d_case_5, construct_d_case_6, rational_laurent_series,
    solve_riccati)  # 导入Riccati方程相关函数

f = Function('f')  # 创建一个名为f的函数对象
x = symbols('x')  # 创建一个名为x的符号对象

# 这些函数用于生成测试，不应直接在测试中使用

def rand_rational(maxint):
    """
    生成一个随机有理数

    Parameters
    ----------
    maxint : int
        随机数的最大绝对值

    Returns
    -------
    Rational
        生成的随机有理数
    """
    return Rational(randint(-maxint, maxint), randint(1, maxint))


def rand_poly(x, degree, maxint):
    """
    生成一个随机多项式

    Parameters
    ----------
    x : Symbol
        多项式的符号
    degree : int
        多项式的最高次数
    maxint : int
        随机数的最大绝对值

    Returns
    -------
    Poly
        生成的随机多项式对象
    """
    return Poly([rand_rational(maxint) for _ in range(degree+1)], x)


def rand_rational_function(x, degree, maxint):
    """
    生成一个随机有理函数

    Parameters
    ----------
    x : Symbol
        函数的符号
    degree : int
        函数的最高次数
    maxint : int
        随机数的最大绝对值

    Returns
    -------
    Mul
        生成的随机有理函数对象
    """
    degnum = randint(1, degree)
    degden = randint(1, degree)
    num = rand_poly(x, degnum, maxint)
    den = rand_poly(x, degden, maxint)
    while den == Poly(0, x):
        den = rand_poly(x, degden, maxint)
    return num / den


def find_riccati_ode(ratfunc, x, yf):
    """
    寻找Riccati型ODE的表达式

    Parameters
    ----------
    ratfunc : Mul
        有理函数
    x : Symbol
        自变量符号
    yf : Function
        依赖于自变量的函数

    Returns
    -------
    Eq
        Riccati型ODE的等式表达式
    """
    y = ratfunc
    yp = y.diff(x)
    q1 = rand_rational_function(x, 1, 3)
    q2 = rand_rational_function(x, 1, 3)
    while q2 == 0:
        q2 = rand_rational_function(x, 1, 3)
    q0 = ratsimp(yp - q1*y - q2*y**2)
    eq = Eq(yf.diff(), q0 + q1*yf + q2*yf**2)
    sol = Eq(yf, y)
    assert checkodesol(eq, sol) == (True, 0)
    return eq, q0, q1, q2


# 测试函数开始

def test_riccati_transformation():
    """
    此函数测试Riccati ODE的解到其对应正常Riccati ODE的转换。

    每个测试案例包含4个值 -

    1. w - 要转换的解
    2. b1 - ODE中f(x)的系数
    3. b2 - ODE中f(x)**2的系数
    4. y - 正常Riccati ODE的解
    """
    tests = [
    (
        x/(x - 1),
        (x**2 + 7)/3*x,
        x,
        -x**2/(x - 1) - x*(x**2/3 + S(7)/3)/2 - 1/(2*x)
    ),
    (
        (2*x + 3)/(2*x + 2),
        (3 - 3*x)/(x + 1),
        5*x,
        -5*x*(2*x + 3)/(2*x + 2) - (3 - 3*x)/(Mul(2, x + 1, evaluate=False)) - 1/(2*x)
    ),
    # 定义测试用例列表 `tests`，每个元素是一个包含多个表达式的元组
    tests = [
        (
            # 第一个测试元组，包含四个表达式
            -1/(2*x**2 - 1),                                     # 第一个表达式：分数形式的数学表达式
            0,                                                   # 第二个表达式：常数0
            (2 - x)/(4*x - 2),                                   # 第三个表达式：分数形式的数学表达式
            (2 - x)/((4*x - 2)*(2*x**2 - 1)) - (4*x - 2)*(Mul(-4, 2 - x, evaluate=False)/(4*x - 2)**2 - 1/(4*x - 2))/(Mul(2, 2 - x, evaluate=False))  # 第四个表达式：复杂的数学表达式
        ),
        (
            # 第二个测试元组，包含四个表达式
            x,                                                   # 第一个表达式：变量x
            (8*x - 12)/(12*x + 9),                               # 第二个表达式：分数形式的数学表达式
            x**3/(6*x - 9),                                      # 第三个表达式：分数形式的数学表达式
            -x**4/(6*x - 9) - (8*x - 12)/(Mul(2, 12*x + 9, evaluate=False)) - (6*x - 9)*(-6*x**3/(6*x - 9)**2 + 3*x**2/(6*x - 9))/(2*x**3)  # 第四个表达式：复杂的数学表达式
        )]
        
    # 对于每个测试元组中的参数 w, b1, b2, y，执行以下断言
    for w, b1, b2, y in tests:
        assert y == riccati_normal(w, x, b1, b2)                # 断言：使用 riccati_normal 函数计算的结果与预期的 y 值相等
        assert w == riccati_inverse_normal(y, x, b1, b2).cancel()  # 断言：使用 riccati_inverse_normal 函数计算的结果取消化后与预期的 w 值相等
    
    # 测试 riccati_inverse_normal 函数的 bp 参数
    tests = [
        (
            # 第一个测试元组，包含五个表达式
            (-2*x - 1)/(2*x**2 + 2*x - 2),                        # 第一个表达式：分数形式的数学表达式
            -2/x,                                                 # 第二个表达式：分数形式的数学表达式
            (-x - 1)/(4*x),                                       # 第三个表达式：分数形式的数学表达式
            8*x**2*(1/(4*x) + (-x - 1)/(4*x**2))/(-x - 1)**2 + 4/(-x - 1),  # 第四个表达式：复杂的数学表达式
            -2*x*(-1/(4*x) - (-x - 1)/(4*x**2))/(-x - 1) - (-2*x - 1)*(-x - 1)/(4*x*(2*x**2 + 2*x - 2)) + 1/x  # 第五个表达式：复杂的数学表达式
        ),
        (
            # 第二个测试元组，包含五个表达式
            3/(2*x**2),                                           # 第一个表达式：分数形式的数学表达式
            -2/x,                                                 # 第二个表达式：分数形式的数学表达式
            (-x - 1)/(4*x),                                       # 第三个表达式：分数形式的数学表达式
            8*x**2*(1/(4*x) + (-x - 1)/(4*x**2))/(-x - 1)**2 + 4/(-x - 1),  # 第四个表达式：复杂的数学表达式
            -2*x*(-1/(4*x) - (-x - 1)/(4*x**2))/(-x - 1) + 1/x - Mul(3, -x - 1, evaluate=False)/(8*x**3)  # 第五个表达式：复杂的数学表达式
        )]
        
    # 对于每个测试元组中的参数 w, b1, b2, bp, y，执行以下断言
    for w, b1, b2, bp, y in tests:
        assert y == riccati_normal(w, x, b1, b2)                # 断言：使用 riccati_normal 函数计算的结果与预期的 y 值相等
        assert w == riccati_inverse_normal(y, x, b1, b2, bp).cancel()  # 断言：使用 riccati_inverse_normal 函数计算的结果取消化后与预期的 w 值相等
# 定义一个函数，用于测试将 Riccati 微分方程转换为其标准形式的正确性
def test_riccati_reduced():
    """
    This function tests the transformation of a
    Riccati ODE to its normal Riccati ODE.

    Each test case 2 values -

    1. eq - A Riccati ODE.
    2. normal_eq - The normal Riccati ODE of eq.
    """
    # 定义测试用例列表，每个测试用例包含两个元素：原始 Riccati 微分方程和其标准形式的 Riccati 微分方程
    tests = [
    (
        f(x).diff(x) - x**2 - x*f(x) - x*f(x)**2,

        f(x).diff(x) + f(x)**2 + x**3 - x**2/4 - 3/(4*x**2)
    ),
    (
        6*x/(2*x + 9) + f(x).diff(x) - (x + 1)*f(x)**2/x,

        -3*x**2*(1/x + (-x - 1)/x**2)**2/(4*(-x - 1)**2) + Mul(6, \
        -x - 1, evaluate=False)/(2*x + 9) + f(x)**2 + f(x).diff(x) \
        - (-1 + (x + 1)/x)/(x*(-x - 1))
    ),
    (
        f(x)**2 + f(x).diff(x) - (x - 1)*f(x)/(-x - S(1)/2),

        -(2*x - 2)**2/(4*(2*x + 1)**2) + (2*x - 2)/(2*x + 1)**2 + \
        f(x)**2 + f(x).diff(x) - 1/(2*x + 1)
    ),
    (
        f(x).diff(x) - f(x)**2/x,

        f(x)**2 + f(x).diff(x) + 1/(4*x**2)
    ),
    (
        -3*(-x**2 - x + 1)/(x**2 + 6*x + 1) + f(x).diff(x) + f(x)**2/x,

        f(x)**2 + f(x).diff(x) + (3*x**2/(x**2 + 6*x + 1) + 3*x/(x**2 \
        + 6*x + 1) - 3/(x**2 + 6*x + 1))/x + 1/(4*x**2)
    ),
    (
        6*x/(2*x + 9) + f(x).diff(x) - (x + 1)*f(x)/x,

        False  # 不匹配的测试用例，标记为 False
    ),
    (
        f(x)*f(x).diff(x) - 1/x + f(x)/3 + f(x)**2/(x**2 - 2),

        False  # 不匹配的测试用例，标记为 False
    )]
    # 遍历所有测试用例
    for eq, normal_eq in tests:
        # 断言标准形式的 Riccati 微分方程是否等于转换函数 riccati_reduced 处理后的结果
        assert normal_eq == riccati_reduced(eq, f, x)


# 定义一个函数，用于测试一个微分方程是否为 Riccati 方程
def test_match_riccati():
    """
    This function tests if an ODE is Riccati or not.

    Each test case has 5 values -

    1. eq - The Riccati ODE.
    2. match - Boolean indicating if eq is a Riccati ODE.
    3. b0 -
    4. b1 - Coefficient of f(x) in eq.
    5. b2 - Coefficient of f(x)**2 in eq.
    """
    # 定义测试用例列表，每个测试用例包含五个元素：Riccati 微分方程、是否匹配 Riccati 方程、系数 b0、系数 b1、系数 b2
    tests = [
    # Test Rational Riccati ODEs
    (
        f(x).diff(x) - (405*x**3 - 882*x**2 - 78*x + 92)/(243*x**4 \
        - 945*x**3 + 846*x**2 + 180*x - 72) - 2 - f(x)**2/(3*x + 1) \
        - (S(1)/3 - x)*f(x)/(S(1)/3 - 3*x/2),

        True,  # 是一个 Riccati 方程

        45*x**3/(27*x**4 - 105*x**3 + 94*x**2 + 20*x - 8) - 98*x**2/ \
        (27*x**4 - 105*x**3 + 94*x**2 + 20*x - 8) - 26*x/(81*x**4 - \
        315*x**3 + 282*x**2 + 60*x - 24) + 2 + 92/(243*x**4 - 945*x**3 \
        + 846*x**2 + 180*x - 72),

        Mul(-1, 2 - 6*x, evaluate=False)/(9*x - 2),

        1/(3*x + 1)
    ),
    (
        f(x).diff(x) + 4*x/27 - (x/3 - 1)*f(x)**2 - (2*x/3 + \
        1)*f(x)/(3*x + 2) - S(10)/27 - (265*x**2 + 423*x + 162) \
        /(324*x**3 + 216*x**2),

        True,  # 是一个 Riccati 方程

        -4*x/27 + S(10)/27 + 3/(6*x**3 + 4*x**2) + 47/(36*x**2 \
        + 24*x) + 265/(324*x + 216),

        Mul(-1, -2*x - 3, evaluate=False)/(9*x + 6),

        x/3 - 1
    ),
    (
        f(x).diff(x) - (304*x**5 - 745*x**4 + 631*x**3 - 876*x**2 \
        + 198*x - 108)/(36*x**6 - 216*x**5 + 477*x**4 - 567*x**3 + \
        360*x**2 - 108) - S(17)/9 - (x - S(3)/2)*f(x)/(x/2 - \
        S(3)/2) - (x/3 - 3)*f(x)**2/(3*x),
        # 第一个微分方程的定义，包括多个项的数学表达式
        True,
        # 是否为 Riccati ODE 的测试结果
        304*x**4/(36*x**5 - 216*x**4 + 477*x**3 - 567*x**2 + 360*x - \
        108) - 745*x**3/(36*x**5 - 216*x**4 + 477*x**3 - 567*x**2 + \
        360*x - 108) + 631*x**2/(36*x**5 - 216*x**4 + 477*x**3 - 567* \
        x**2 + 360*x - 108) - 292*x/(12*x**5 - 72*x**4 + 159*x**3 - \
        189*x**2 + 120*x - 36) + S(17)/9 - 12/(4*x**6 - 24*x**5 + \
        53*x**4 - 63*x**3 + 40*x**2 - 12*x) + 22/(4*x**5 - 24*x**4 \
        + 53*x**3 - 63*x**2 + 40*x - 12),
        # 第一个微分方程中的一部分表达式，涉及数学计算和常数
        Mul(-1, 3 - 2*x, evaluate=False)/(x - 3),
        # 第一个微分方程中的一部分表达式，使用了 Mul 类表示负数分数
        Mul(-1, 9 - x, evaluate=False)/(9*x)
        # 第一个微分方程中的一部分表达式，使用了 Mul 类表示负数分数
    ),
    # 测试非有理 Riccati ODE
    (
        f(x).diff(x) - x**(S(3)/2)/(x**(S(1)/2) - 2) + x**2*f(x) + \
        x*f(x)**2/(x**(S(3)/4)),
        False, 0, 0, 0
        # 第二个微分方程的定义，包括多个项的数学表达式，以及是否为 Riccati ODE 的测试结果
    ),
    (
        f(x).diff(x) - sin(x**2) + exp(x)*f(x) + log(x)*f(x)**2,
        False, 0, 0, 0
        # 第三个微分方程的定义，包括多个项的数学表达式，以及是否为 Riccati ODE 的测试结果
    ),
    (
        f(x).diff(x) - tanh(x + sqrt(x)) + f(x) + x**4*f(x)**2,
        False, 0, 0, 0
        # 第四个微分方程的定义，包括多个项的数学表达式，以及是否为 Riccati ODE 的测试结果
    ),
    # 测试非 Riccati ODE
    (
        (1 - x**2)*f(x).diff(x, 2) - 2*x*f(x).diff(x) + 20*f(x),
        False, 0, 0, 0
        # 第五个微分方程的定义，包括多个项的数学表达式，以及是否为 Riccati ODE 的测试结果
    ),
    (
        f(x).diff(x) - x**2 + x**3*f(x) + (x**2/(x + 1))*f(x)**3,
        False, 0, 0, 0
        # 第六个微分方程的定义，包括多个项的数学表达式，以及是否为 Riccati ODE 的测试结果
    ),
    (
        f(x).diff(x)*f(x)**2 + (x**2 - 1)/(x**3 + 1)*f(x) + 1/(2*x \
        + 3) + f(x)**2,
        False, 0, 0, 0
        # 第七个微分方程的定义，包括多个项的数学表达式，以及是否为 Riccati ODE 的测试结果
    )]
    for eq, res, b0, b1, b2 in tests:
        # 对于每个测试案例，调用 match_riccati 函数来匹配 Riccati ODE
        match, funcs = match_riccati(eq, f, x)
        # 断言匹配结果是否与预期相符
        assert match == res
        if res:
            # 如果是 Riccati ODE，断言附加条件是否满足
            assert [b0, b1, b2] == funcs
    # 定义测试函数，测试有理函数在无穷远处的估值
    def test_val_at_inf():
        """
        This function tests the valuation of rational
        function at oo.

        Each test case has 3 values -

        1. num - Numerator of rational function.
        2. den - Denominator of rational function.
        3. val_inf - Valuation of rational function at oo
        """
        # 准备测试用例列表
        tests = [
            # 分母的次数 > 分子的次数
            (
                Poly(10*x**3 + 8*x**2 - 13*x + 6, x),
                Poly(-13*x**10 - x**9 + 5*x**8 + 7*x**7 + 10*x**6 + 6*x**5 - 7*x**4 + 11*x**3 - 8*x**2 + 5*x + 13, x),
                7
            ),
            (
                Poly(1, x),
                Poly(-9*x**4 + 3*x**3 + 15*x**2 - 6*x - 14, x),
                4
            ),
            # 分母的次数 == 分子的次数
            (
                Poly(-6*x**3 - 8*x**2 + 8*x - 6, x),
                Poly(-5*x**3 + 12*x**2 - 6*x - 9, x),
                0
            ),
            # 分母的次数 < 分子的次数
            (
                Poly(12*x**8 - 12*x**7 - 11*x**6 + 8*x**5 + 3*x**4 - x**3 + x**2 - 11*x, x),
                Poly(-14*x**2 + x, x),
                -6
            ),
            (
                Poly(5*x**6 + 9*x**5 - 11*x**4 - 9*x**3 + x**2 - 4*x + 4, x),
                Poly(15*x**4 + 3*x**3 - 8*x**2 + 15*x + 12, x),
                -2
            )
        ]
        # 遍历测试用例，断言有理函数在无穷远处的估值等于预期值
        for num, den, val in tests:
            assert val_at_inf(num, den, x) == val


    # 测试有理Riccati ODE有理特解的必要条件
    def test_necessary_conds():
        """
        This function tests the necessary conditions for
        a Riccati ODE to have a rational particular solution.
        """
        # 无穷远处的估值是奇数负整数
        assert check_necessary_conds(-3, [1, 2, 4]) == False
        # 无穷远处的估值是小于2的正整数
        assert check_necessary_conds(1, [1, 2, 4]) == False
        # 极点的重数是大于1的奇数整数
        assert check_necessary_conds(2, [3, 1, 6]) == False
        # 所有条件均正确
        assert check_necessary_conds(-10, [1, 2, 8, 12]) == True


    # 测试有理函数在换元 x -> 1/x 后的逆变换
    def test_inverse_transform_poly():
        """
        This function tests the substitution x -> 1/x
        in rational functions represented using Poly.
        """
        # 准备有理函数列表
        fns = [
            (15*x**3 - 8*x**2 - 2*x - 6)/(18*x + 6),

            (180*x**5 + 40*x**4 + 80*x**3 + 30*x**2 - 60*x - 80)/(180*x**3 - 150*x**2 + 75*x + 12),

            (-15*x**5 - 36*x**4 + 75*x**3 - 60*x**2 - 80*x - 60)/(80*x**4 + 60*x**3 + 60*x**2 + 60*x - 80),

            (60*x**7 + 24*x**6 - 15*x**5 - 20*x**4 + 30*x**2 + 100*x - 60)/(240*x**2 - 20*x - 30),

            (30*x**6 - 12*x**5 + 15*x**4 - 15*x**2 + 10*x + 60)/(3*x**10 - 45*x**9 + 15*x**5 + 15*x**4 - 5*x**3 \
            + 15*x**2 + 45*x - 15)
        ]
        # 遍历函数列表，对每个函数进行测试
        for f in fns:
            # 将有理函数表示为分子和分母的Poly对象
            num, den = [Poly(e, x) for e in f.as_numer_denom()]
            # 进行 x -> 1/x 的逆变换
            num, den = inverse_transform_poly(num, den, x)
            # 断言变换后的函数与原函数进行 x -> 1/x 变换后的化简结果相等
            assert f.subs(x, 1/x).cancel() == num/den


    # 测试有理函数在无穷远处的极限值
    def test_limit_at_inf():
        """
        This function tests the limit at oo of a
        rational function.

        Each test case has 3 values -

        1. num - Numerator of rational function.
        2. den - Denominator of rational function.
        3. limit_at_inf - Limit of rational function at oo
        """
        # 准备测试用例列表
        tests = [
            # 分母的次数 > 分子的次数时的测试用例
    (
        Poly(-12*x**2 + 20*x + 32, x),
        Poly(32*x**3 + 72*x**2 + 3*x - 32, x),
        0
    ),
    # 当分母的次数小于分子的次数时
    (
        Poly(1260*x**4 - 1260*x**3 - 700*x**2 - 1260*x + 1400, x),
        Poly(6300*x**3 - 1575*x**2 + 756*x - 540, x),
        oo
    ),
    # 当分母的次数小于分子的次数，并且其中一个主导系数为负数时
    (
        Poly(-735*x**8 - 1400*x**7 + 1680*x**6 - 315*x**5 - 600*x**4 + 840*x**3 - 525*x**2 \
        + 630*x + 3780, x),
        Poly(1008*x**7 - 2940*x**6 - 84*x**5 + 2940*x**4 - 420*x**3 + 1512*x**2 + 105*x + 168, x),
        -oo
    ),
    # 当分母的次数等于分子的次数时
    (
        Poly(105*x**7 - 960*x**6 + 60*x**5 + 60*x**4 - 80*x**3 + 45*x**2 + 120*x + 15, x),
        Poly(735*x**7 + 525*x**6 + 720*x**5 + 720*x**4 - 8400*x**3 - 2520*x**2 + 2800*x + 280, x),
        S(1)/7
    ),
    # 用于测试的多项式对及其极限值
    (
        Poly(288*x**4 - 450*x**3 + 280*x**2 - 900*x - 90, x),
        Poly(607*x**4 + 840*x**3 - 1050*x**2 + 420*x + 420, x),
        S(288)/607
    )]
    for num, den, lim in tests:
        # 断言：计算给定多项式对的极限值是否等于预期的极限值
        assert limit_at_inf(num, den, x) == lim
# 定义测试函数，用于测试计算 c 向量系数的第一种情况
def test_construct_c_case_1():
    """
    This function tests the Case 1 in the step
    to calculate coefficients of c-vectors.

    Each test case has 4 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. pole - Pole of a(x) for which c-vector is being
       calculated.
    4. c - The c-vector for the pole.
    """
    # 定义测试用例列表
    tests = [
        (
            Poly(-3*x**3 + 3*x**2 + 4*x - 5, x, extension=True),
            Poly(4*x**8 + 16*x**7 + 9*x**5 + 12*x**4 + 6*x**3 + 12*x**2, x, extension=True),
            S(0),
            [[S(1)/2 + sqrt(6)*I/6], [S(1)/2 - sqrt(6)*I/6]]
        ),
        (
            Poly(1200*x**3 + 1440*x**2 + 816*x + 560, x, extension=True),
            Poly(128*x**5 - 656*x**4 + 1264*x**3 - 1125*x**2 + 385*x + 49, x, extension=True),
            S(7)/4,
            [[S(1)/2 + sqrt(16367978)/634], [S(1)/2 - sqrt(16367978)/634]]
        ),
        (
            Poly(4*x + 2, x, extension=True),
            Poly(18*x**4 + (2 - 18*sqrt(3))*x**3 + (14 - 11*sqrt(3))*x**2 + (4 - 6*sqrt(3))*x \
                + 8*sqrt(3) + 16, x, domain='QQ<sqrt(3)>'),
            (S(1) + sqrt(3))/2,
            [[S(1)/2 + sqrt(Mul(4, 2*sqrt(3) + 4, evaluate=False)/(19*sqrt(3) + 44) + 1)/2], \
                [S(1)/2 - sqrt(Mul(4, 2*sqrt(3) + 4, evaluate=False)/(19*sqrt(3) + 44) + 1)/2]]
        )
    ]
    
    # 遍历测试用例，进行断言验证
    for num, den, pole, c in tests:
        assert construct_c_case_1(num, den, x, pole) == c


# 定义测试函数，用于测试计算 c 向量系数的第二种情况
def test_construct_c_case_2():
    """
    This function tests the Case 2 in the step
    to calculate coefficients of c-vectors.

    Each test case has 5 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. pole - Pole of a(x) for which c-vector is being
       calculated.
    4. mul - The multiplicity of the pole.
    5. c - The c-vector for the pole.
    """
    # 定义测试用例列表
    tests = [
        # Testing poles with multiplicity 2
        (
            Poly(1, x, extension=True),
            Poly((x - 1)**2*(x - 2), x, extension=True),
            1, 2,
            [[-I*(-1 - I)/2], [I*(-1 + I)/2]]
        ),
        (
            Poly(3*x**5 - 12*x**4 - 7*x**3 + 1, x, extension=True),
            Poly((3*x - 1)**2*(x + 2)**2, x, extension=True),
            S(1)/3, 2,
            [[-S(89)/98], [-S(9)/98]]
        ),
        # Testing poles with multiplicity 4
        (
            Poly(x**3 - x**2 + 4*x, x, extension=True),
            Poly((x - 2)**4*(x + 5)**2, x, extension=True),
            2, 4,
            [[7*sqrt(3)*(S(60)/343 - 4*sqrt(3)/7)/12, 2*sqrt(3)/7], \
            [-7*sqrt(3)*(S(60)/343 + 4*sqrt(3)/7)/12, -2*sqrt(3)/7]]
        ),
        (
            Poly(3*x**5 + x**4 + 3, x, extension=True),
            Poly((4*x + 1)**4*(x + 2), x, extension=True),
            -S(1)/4, 4,
            [[128*sqrt(439)*(-sqrt(439)/128 - S(55)/14336)/439, sqrt(439)/256], \
            [-128*sqrt(439)*(sqrt(439)/128 - S(55)/14336)/439, -sqrt(439)/256]]
        ),
        # Testing poles with multiplicity 6
    ]
    
    # 遍历测试用例，进行断言验证
    for num, den, pole, mul, c in tests:
        assert construct_c_case_2(num, den, x, pole, mul) == c
    # 定义一个元组列表 `tests`，包含多个元组，每个元组都是测试用例
    (
        # 第一个元组的内容
        Poly(x**3 + 2, x, extension=True),  # 多项式对象，表示 x^3 + 2，使用扩展模式
        Poly((3*x - 1)**6*(x**2 + 1), x, extension=True),  # 多项式对象，表示 (3*x - 1)^6 * (x^2 + 1)，使用扩展模式
        S(1)/3,  # 符号表达式，表示有理数 1/3
        6,  # 整数 6
        # 二维列表，包含两个子列表，每个子列表有三个表达式
        [[27*sqrt(66)*(-sqrt(66)/54 - S(131)/267300)/22, -2*sqrt(66)/1485, sqrt(66)/162], \
        [-27*sqrt(66)*(sqrt(66)/54 - S(131)/267300)/22, 2*sqrt(66)/1485, -sqrt(66)/162]]
    ),
    (
        # 第二个元组的内容
        Poly(x**2 + 12, x, extension=True),  # 多项式对象，表示 x^2 + 12，使用扩展模式
        Poly((x - sqrt(2))**6, x, extension=True),  # 多项式对象，表示 (x - sqrt(2))^6，使用扩展模式
        sqrt(2),  # 符号表达式，表示根号 2
        6,  # 整数 6
        # 二维列表，包含两个子列表，每个子列表有三个表达式
        [[sqrt(14)*(S(6)/7 - 3*sqrt(14))/28, sqrt(7)/7, sqrt(14)], \
        [-sqrt(14)*(S(6)/7 + 3*sqrt(14))/28, -sqrt(7)/7, -sqrt(14)]]
    )]
    # 对于每个元组 (num, den, pole, mul, c) 在 tests 列表中，验证 construct_c_case_2 函数的返回值是否等于 c
    for num, den, pole, mul, c in tests:
        assert construct_c_case_2(num, den, x, pole, mul) == c
def test_construct_c_case_3():
    """
    This function tests the Case 3 in the step
    to calculate coefficients of c-vectors.
    """
    # 断言检查构造的 c-vectors 是否正确
    assert construct_c_case_3() == [[1]]


def test_construct_d_case_4():
    """
    This function tests the Case 4 in the step
    to calculate coefficients of the d-vector.

    Each test case has 4 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. mul - Multiplicity of oo as a pole.
    4. d - The d-vector.
    """
    # 不同的测试案例
    tests = [
        # Tests with multiplicity at oo = 2
        (
            Poly(-x**5 - 2*x**4 + 4*x**3 + 2*x + 5, x, extension=True),
            Poly(9*x**3 - 2*x**2 + 10*x - 2, x, extension=True),
            2,
            [[10*I/27, I/3, -3*I*(S(158)/243 - I/3)/2], \
            [-10*I/27, -I/3, 3*I*(S(158)/243 + I/3)/2]]
        ),
        (
            Poly(-x**6 + 9*x**5 + 5*x**4 + 6*x**3 + 5*x**2 + 6*x + 7, x, extension=True),
            Poly(x**4 + 3*x**3 + 12*x**2 - x + 7, x, extension=True),
            2,
            [[-6*I, I, -I*(17 - I)/2], [6*I, -I, I*(17 + I)/2]]
        ),
        # Tests with multiplicity at oo = 4
        (
            Poly(-2*x**6 - x**5 - x**4 - 2*x**3 - x**2 - 3*x - 3, x, extension=True),
            Poly(3*x**2 + 10*x + 7, x, extension=True),
            4,
            [[269*sqrt(6)*I/288, -17*sqrt(6)*I/36, sqrt(6)*I/3, -sqrt(6)*I*(S(16969)/2592 \
            - 2*sqrt(6)*I/3)/4], [-269*sqrt(6)*I/288, 17*sqrt(6)*I/36, -sqrt(6)*I/3, \
            sqrt(6)*I*(S(16969)/2592 + 2*sqrt(6)*I/3)/4]]
        ),
        (
            Poly(-3*x**5 - 3*x**4 - 3*x**3 - x**2 - 1, x, extension=True),
            Poly(12*x - 2, x, extension=True),
            4,
            [[41*I/192, 7*I/24, I/2, -I*(-S(59)/6912 - I)], \
            [-41*I/192, -7*I/24, -I/2, I*(-S(59)/6912 + I)]]
        ),
        # Tests with multiplicity at oo = 4
        (
            Poly(-x**7 - x**5 - x**4 - x**2 - x, x, extension=True),
            Poly(x + 2, x, extension=True),
            6,
            [[-5*I/2, 2*I, -I, I, -I*(-9 - 3*I)/2], [5*I/2, -2*I, I, -I, I*(-9 + 3*I)/2]]
        ),
        (
            Poly(-x**7 - x**6 - 2*x**5 - 2*x**4 - x**3 - x**2 + 2*x - 2, x, extension=True),
            Poly(2*x - 2, x, extension=True),
            6,
            [[3*sqrt(2)*I/4, 3*sqrt(2)*I/4, sqrt(2)*I/2, sqrt(2)*I/2, -sqrt(2)*I*(-S(7)/8 - \
            3*sqrt(2)*I/2)/2], [-3*sqrt(2)*I/4, -3*sqrt(2)*I/4, -sqrt(2)*I/2, -sqrt(2)*I/2, \
            sqrt(2)*I*(-S(7)/8 + 3*sqrt(2)*I/2)/2]]
        )
    ]
    # 遍历测试案例并断言检查构造的 d-vectors 是否正确
    for num, den, mul, d in tests:
        ser = rational_laurent_series(num, den, x, oo, mul, 1)
        assert construct_d_case_4(ser, mul//2) == d


def test_construct_d_case_5():
    """
    This function tests the Case 5 in the step
    to calculate coefficients of the d-vector.

    Each test case has 3 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. d - The d-vector.
    """
    # 不同的测试案例
    tests = [
    (
        Poly(2*x**3 + x**2 + x - 2, x, extension=True),  # 创建多项式对象，指定变量 x 和扩展域为 True
        Poly(9*x**3 + 5*x**2 + 2*x - 1, x, extension=True),  # 创建多项式对象，指定变量 x 和扩展域为 True
        [[sqrt(2)/3, -sqrt(2)/108], [-sqrt(2)/3, sqrt(2)/108]]  # 二维列表，包含有理数表达式
    ),
    (
        Poly(3*x**5 + x**4 - x**3 + x**2 - 2*x - 2, x, domain='ZZ'),  # 创建多项式对象，指定变量 x 和整数环域为 'ZZ'
        Poly(9*x**5 + 7*x**4 + 3*x**3 + 2*x**2 + 5*x + 7, x, domain='ZZ'),  # 创建多项式对象，指定变量 x 和整数环域为 'ZZ'
        [[sqrt(3)/3, -2*sqrt(3)/27], [-sqrt(3)/3, 2*sqrt(3)/27]]  # 二维列表，包含有理数表达式
    ),
    (
        Poly(x**2 - x + 1, x, domain='ZZ'),  # 创建多项式对象，指定变量 x 和整数环域为 'ZZ'
        Poly(3*x**2 + 7*x + 3, x, domain='ZZ'),  # 创建多项式对象，指定变量 x 和整数环域为 'ZZ'
        [[sqrt(3)/3, -5*sqrt(3)/9], [-sqrt(3)/3, 5*sqrt(3)/9]]  # 二维列表，包含有理数表达式
    )]
    for num, den, d in tests:
        # 对于每个元组 (num, den, d)，执行以下操作：
        # Multiplicity of oo is 0
        ser = rational_laurent_series(num, den, x, oo, 0, 1)  # 调用 rational_laurent_series 函数，生成 Laurent 级数 ser
        assert construct_d_case_5(ser) == d  # 断言确保 construct_d_case_5 函数应用于 ser 后得到结果等于 d
def test_construct_d_case_6():
    """
    This function tests the Case 6 in the step
    to calculate coefficients of the d-vector.

    Each test case has 3 values -

    1. num - Numerator of the rational function a(x).
    2. den - Denominator of the rational function a(x).
    3. d - The d-vector.
    """
    # 定义测试用例的列表
    tests = [
    (
        Poly(-2*x**2 - 5, x, domain='ZZ'),  # 分子的多项式
        Poly(4*x**4 + 2*x**2 + 10*x + 2, x, domain='ZZ'),  # 分母的多项式
        [[S(1)/2 + I/2], [S(1)/2 - I/2]]  # 期望的 d-vector
    ),
    (
        Poly(-2*x**3 - 4*x**2 - 2*x - 5, x, domain='ZZ'),  # 分子的多项式
        Poly(x**6 - x**5 + 2*x**4 - 4*x**3 - 5*x**2 - 5*x + 9, x, domain='ZZ'),  # 分母的多项式
        [[1], [0]]  # 期望的 d-vector
    ),
    (
        Poly(-5*x**3 + x**2 + 11*x + 12, x, domain='ZZ'),  # 分子的多项式
        Poly(6*x**8 - 26*x**7 - 27*x**6 - 10*x**5 - 44*x**4 - 46*x**3 - 34*x**2 \
        - 27*x - 42, x, domain='ZZ'),  # 分母的多项式
        [[1], [0]]  # 期望的 d-vector
    )]
    # 对于每个测试案例，验证 construct_d_case_6 函数的输出是否符合期望的 d-vector
    for num, den, d in tests:
        assert construct_d_case_6(num, den, x) == d


def test_rational_laurent_series():
    """
    This function tests the computation of coefficients
    of Laurent series of a rational function.

    Each test case has 5 values -

    1. num - Numerator of the rational function.
    2. den - Denominator of the rational function.
    3. x0 - Point about which Laurent series is to
       be calculated.
    4. mul - Multiplicity of x0 if x0 is a pole of
       the rational function (0 otherwise).
    5. n - Number of terms upto which the series
       is to be calculated.
    """
    # 定义测试用例的列表
    tests = [
    # Laurent series about simple pole (Multiplicity = 1)
    (
        Poly(x**2 - 3*x + 9, x, extension=True),  # 分子的多项式
        Poly(x**2 - x, x, extension=True),  # 分母的多项式
        S(1), 1, 6,  # x0, multiplicity, n
        {1: 7, 0: -8, -1: 9, -2: -9, -3: 9, -4: -9}  # 期望的 Laurent 级数系数
    ),
    # Laurent series about multiple pole (Multiplicity > 1)
    (
        Poly(64*x**3 - 1728*x + 1216, x, extension=True),  # 分子的多项式
        Poly(64*x**4 - 80*x**3 - 831*x**2 + 1809*x - 972, x, extension=True),  # 分母的多项式
        S(9)/8, 2, 3,  # x0, multiplicity, n
        {0: S(32177152)/46521675, 2: S(1019)/984, -1: S(11947565056)/28610830125, \
        1: S(209149)/75645}  # 期望的 Laurent 级数系数
    ),
    (
        Poly(1, x, extension=True),  # 分子的多项式
        Poly(x**5 + (-4*sqrt(2) - 1)*x**4 + (4*sqrt(2) + 12)*x**3 + (-12 - 8*sqrt(2))*x**2 \
        + (4 + 8*sqrt(2))*x - 4, x, extension=True),  # 分母的多项式
        sqrt(2), 4, 6,  # x0, multiplicity, n
        {4: 1 + sqrt(2), 3: -3 - 2*sqrt(2), 2: Mul(-1, -3 - 2*sqrt(2), evaluate=False)/(-1 \
        + sqrt(2)), 1: (-3 - 2*sqrt(2))/(-1 + sqrt(2))**2, 0: Mul(-1, -3 - 2*sqrt(2), evaluate=False \
        )/(-1 + sqrt(2))**3, -1: (-3 - 2*sqrt(2))/(-1 + sqrt(2))**4}  # 期望的 Laurent 级数系数
    ),
    # Laurent series about oo
    (
        Poly(x**5 - 4*x**3 + 6*x**2 + 10*x - 13, x, extension=True),  # 分子的多项式
        Poly(x**2 - 5, x, extension=True),  # 分母的多项式
        oo, 3, 6,  # x0, multiplicity, n
        {3: 1, 2: 0, 1: 1, 0: 6, -1: 15, -2: 17}  # 期望的 Laurent 级数系数
    ),
    # Laurent series at x0 where x0 is not a pole of the function
    # Using multiplicity as 0 (as x0 will not be a pole)
    ]

    # 对于每个测试案例，验证 rational_laurent_series 函数的输出是否符合期望的 Laurent 级数系数
    for num, den, x0, mul, n, expected_coeffs in tests:
        assert rational_laurent_series(num, den, x0, mul, n) == expected_coeffs
    (
        Poly(3*x**3 + 6*x**2 - 2*x + 5, x, extension=True),  # 第一个多项式的定义：3x³ + 6x² - 2x + 5，使用 x 作为变量
        Poly(9*x**4 - x**3 - 3*x**2 + 4*x + 4, x, extension=True),  # 第二个多项式的定义：9x⁴ - x³ - 3x² + 4x + 4，使用 x 作为变量
        S(2)/5,  # 分数表达式：2/5
        0,  # 整数 0
        1,  # 整数 1
        {0: S(3345)/3304, -1: S(399325)/2729104, -2: S(3926413375)/4508479808, \
        -3: S(-5000852751875)/1862002160704, -4: S(-6683640101653125)/6152055138966016}  # 字典定义，包含数值键值对
    ),
    (
        Poly(-7*x**2 + 2*x - 4, x, extension=True),  # 第一个多项式的定义：-7x² + 2x - 4，使用 x 作为变量
        Poly(7*x**5 + 9*x**4 + 8*x**3 + 3*x**2 + 6*x + 9, x, extension=True),  # 第二个多项式的定义：7x⁵ + 9x⁴ + 8x³ + 3x² + 6x + 9，使用 x 作为变量
        oo,  # 无穷大表示
        0,  # 整数 0
        6,  # 整数 6
        {0: 0, -2: 0, -5: -S(71)/49, -1: 0, -3: -1, -4: S(11)/7}  # 字典定义，包含数值键值对
    )]
    for num, den, x0, mul, n, ser in tests:
        assert ser == rational_laurent_series(num, den, x, x0, mul, n)


注释：

- 第一个元组：
  - 第一个多项式：3x³ + 6x² - 2x + 5，使用 x 作为变量。
  - 第二个多项式：9x⁴ - x³ - 3x² + 4x + 4，使用 x 作为变量。
  - 分数：2/5。
  - 整数：0。
  - 整数：1。
  - 字典：包含数值键值对 {0: 3345/3304, -1: 399325/2729104, -2: 3926413375/4508479808, -3: -5000852751875/1862002160704, -4: -6683640101653125/6152055138966016}。
- 第二个元组：
  - 第一个多项式：-7x² + 2x - 4，使用 x 作为变量。
  - 第二个多项式：7x⁵ + 9x⁴ + 8x³ + 3x² + 6x + 9，使用 x 作为变量。
  - 无穷大：表示正无穷大。
  - 整数：0。
  - 整数：6。
  - 字典：包含数值键值对 {0: 0, -2: 0, -5: -71/49, -1: 0, -3: -1, -4: 11/7}。
- 循环语句：对于 tests 中的每个元组，执行断言检查，确保 ser 等于 rational_laurent_series 函数的返回值。
def check_dummy_sol(eq, solse, dummy_sym):
    """
    Helper function to check if actual solution
    matches expected solution if actual solution
    contains dummy symbols.
    """
    # 如果输入的方程是一个方程对象，将其转换为等式形式
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    # 调用 match_riccati 函数匹配 Riccati 方程，并获取相应的函数列表
    _, funcs = match_riccati(eq, f, x)

    # 解 Riccati 方程并返回一组解
    sols = solve_riccati(f(x), x, *funcs)
    
    # 创建一个名为 C1 的虚拟符号
    C1 = Dummy('C1')
    # 使用 dummy_sym 替换解集中的虚拟符号 C1
    sols = [sol.subs(C1, dummy_sym) for sol in sols]

    # 断言：检查所有解是否满足 ODE 的解的条件
    assert all(x[0] for x in checkodesol(eq, sols))
    # 断言：检查所有解是否满足给定的实际解 solse 的条件，使用 dummy_sym 进行比较
    assert all(s1.dummy_eq(s2, dummy_sym) for s1, s2 in zip(sols, solse))


def test_solve_riccati():
    """
    This function tests the computation of rational
    particular solutions for a Riccati ODE.

    Each test case has 2 values -

    1. eq - Riccati ODE to be solved.
    2. sol - Expected solution to the equation.

    Some examples have been taken from the paper - "Statistical Investigation of
    First-Order Algebraic ODEs and their Rational General Solutions" by
    Georg Grasegger, N. Thieu Vo, Franz Winkler

    https://www3.risc.jku.at/publications/download/risc_5197/RISCReport15-19.pdf
    """
    # 创建一个名为 C0 的虚拟符号
    C0 = Dummy('C0')
    # 定义测试用例列表，每个测试用例是一个元组
    tests = [
    # a(x) is a constant
    (
        Eq(f(x).diff(x) + f(x)**2 - 2, 0),  # 待解的 Riccati ODE 方程
        [Eq(f(x), sqrt(2)), Eq(f(x), -sqrt(2))]  # 期望的解
    ),
    # a(x) is a constant
    (
        f(x)**2 + f(x).diff(x) + 4*f(x)/x + 2/x**2,  # 待解的 Riccati ODE 方程
        [Eq(f(x), (-2*C0 - x)/(C0*x + x**2))]  # 期望的解
    ),
    # a(x) is a constant
    (
        2*x**2*f(x).diff(x) - x*(4*f(x) + f(x).diff(x) - 4) + (f(x) - 1)*f(x),  # 待解的 Riccati ODE 方程
        [Eq(f(x), (C0 + 2*x**2)/(C0 + x))]  # 期望的解
    ),
    # Pole with multiplicity 1
    (
        Eq(f(x).diff(x), -f(x)**2 - 2/(x**3 - x**2)),  # 待解的 Riccati ODE 方程
        [Eq(f(x), 1/(x**2 - x))]  # 期望的解
    ),
    # One pole of multiplicity 2
    (
        x**2 - (2*x + 1/x)*f(x) + f(x)**2 + f(x).diff(x),  # 待解的 Riccati ODE 方程
        [Eq(f(x), (C0*x + x**3 + 2*x)/(C0 + x**2)), Eq(f(x), x)]  # 期望的解
    ),
    (
        x**4*f(x).diff(x) + x**2 - x*(2*f(x)**2 + f(x).diff(x)) + f(x),  # 待解的 Riccati ODE 方程
        [Eq(f(x), (C0*x**2 + x)/(C0 + x**2)), Eq(f(x), x**2)]  # 期望的解
    ),
    # Multiple poles of multiplicity 2
    (
        -f(x)**2 + f(x).diff(x) + (15*x**2 - 20*x + 7)/((x - 1)**2*(2*x \
            - 1)**2),  # 待解的 Riccati ODE 方程
        [Eq(f(x), (9*C0*x - 6*C0 - 15*x**5 + 60*x**4 - 94*x**3 + 72*x**2 \
        - 30*x + 6)/(6*C0*x**2 - 9*C0*x + 3*C0 + 6*x**6 - 29*x**5 + \
        57*x**4 - 58*x**3 + 30*x**2 - 6*x)), Eq(f(x), (3*x - 2)/(2*x**2 \
        - 3*x + 1))]  # 期望的解
    ),
    # Regression: Poles with even multiplicity > 2 fixed
    (
        f(x)**2 + f(x).diff(x) - (4*x**6 - 8*x**5 + 12*x**4 + 4*x**3 + \
            7*x**2 - 20*x + 4)/(4*x**4),  # 待解的 Riccati ODE 方程
        [Eq(f(x), (2*x**5 - 2*x**4 - x**3 + 4*x**2 + 3*x - 2)/(2*x**4 \
            - 2*x**2))]  # 期望的解
    ),
    # Regression: Poles with even multiplicity > 2 fixed
    (
        Eq(f(x).diff(x), (-x**6 + 15*x**4 - 40*x**3 + 45*x**2 - 24*x + 4)/\
            (x**12 - 12*x**11 + 66*x**10 - 220*x**9 + 495*x**8 - 792*x**7 + 924*x**6 - \
            792*x**5 + 495*x**4 - 220*x**3 + 66*x**2 - 12*x + 1) + f(x)**2 + f(x)),
        [Eq(f(x), 1/(x**6 - 6*x**5 + 15*x**4 - 20*x**3 + 15*x**2 - 6*x + 1))]
    ),
    # 求解微分方程，分子为多项式，分母为多项式，右侧还包括 f(x) 的平方和 f(x) 自身
    # 返回 f(x) 满足的方程和其解的列表

    (
        Eq(f(x).diff(x), x*f(x) + 2*x + (3*x - 2)*f(x)**2/(4*x + 2) + \
            (8*x**2 - 7*x + 26)/(16*x**3 - 24*x**2 + 8) - S(3)/2),
        [Eq(f(x), (1 - 4*x)/(2*x - 2))]
    ),
    # 求解微分方程，右侧包括 f(x) 和其平方的项，以及一些复杂的多项式分数形式
    # 返回 f(x) 满足的方程和其解的列表

    (
        Eq(f(x).diff(x), (-12*x**2 - 48*x - 15)/(24*x**3 - 40*x**2 + 8*x + 8) \
            + 3*f(x)**2/(6*x + 2)),
        [Eq(f(x), (2*x + 1)/(2*x - 2))]
    ),
    # 求解微分方程，右侧包括 f(x) 的平方和复杂的多项式分数形式
    # 返回 f(x) 满足的方程和其解的列表

    (
        f(x).diff(x) + (3*x**2 + 1)*f(x)**2/x + (6*x**2 - x + 3)*f(x)/(x*(x \
            - 1)) + (3*x**2 - 2*x + 2)/(x*(x - 1)**2),
        [Eq(f(x), (-C0 - x**3 + x**2 - 2*x)/(C0*x - C0 + x**4 - x**3 + x**2 \
            - x)), Eq(f(x), -1/(x - 1))],
    ),
    # 求解微分方程，右侧包括 f(x) 的平方、f(x) 和多项式乘积的项
    # 返回 f(x) 满足的方程和其解的列表

    (
        f(x).diff(x) - 2*I*(f(x)**2 + 1)/x,
        [Eq(f(x), (-I*C0 + I*x**4)/(C0 + x**4)), Eq(f(x), -I)]
    ),
    # 求解微分方程，右侧包括虚数项和 f(x) 的平方
    # 返回 f(x) 满足的方程和其解的列表

    (
        Eq(f(x).diff(x), x*f(x)/(S(3)/2 - 2*x) + (x/2 - S(1)/3)*f(x)**2/\
            (2*x/3 - S(1)/2) - S(5)/4 + (281*x**2 - 1260*x + 756)/(16*x**3 - 12*x**2)),
        [Eq(f(x), (9 - x)/x), Eq(f(x), (40*x**14 + 28*x**13 + 420*x**12 + 2940*x**11 + \
            18480*x**10 + 103950*x**9 + 519750*x**8 + 2286900*x**7 + 8731800*x**6 + 28378350*\
            x**5 + 76403250*x**4 + 163721250*x**3 + 261954000*x**2 + 278326125*x + 147349125)/\
            ((24*x**14 + 140*x**13 + 840*x**12 + 4620*x**11 + 23100*x**10 + 103950*x**9 + \
            415800*x**8 + 1455300*x**7 + 4365900*x**6 + 10914750*x**5 + 21829500*x**4 + 32744250\
            *x**3 + 32744250*x**2 + 16372125*x)))]
    ),
    # 求解微分方程，右侧包括 f(x) 和其平方的项，以及多项式分数形式
    # 返回 f(x) 满足的方程和其解的列表

    (
        Eq(f(x).diff(x), 18*x**3 + 18*x**2 + (-x/2 - S(1)/2)*f(x)**2 + 6),
        [Eq(f(x), 6*x)]
    ),
    # 求解微分方程，右侧包括 f(x) 的平方和一些常数项
    # 返回 f(x) 满足的方程和其解的列表

    (
        Eq(f(x).diff(x), -3*x**3/4 + 15*x/2 + (x/3 - S(4)/3)*f(x)**2 \
            + 9 + (1 - x)*f(x)/x + 3/x),
        [Eq(f(x), -3*x/2 - 3)]
    )]
    # 求解微分方程，右侧包括 f(x) 的平方和多项式分数形式
    # 返回 f(x) 满足的方程和其解的列表

    for eq, sol in tests:
        check_dummy_sol(eq, sol, C0)
@slow
# 声明一个装饰器 @slow，用于标记该测试函数较慢执行

def test_solve_riccati_slow():
    """
    This function tests the computation of rational
    particular solutions for a Riccati ODE.

    Each test case has 2 values -

    1. eq - Riccati ODE to be solved.
    2. sol - Expected solution to the equation.
    """
    # 声明一个符号变量 C0
    C0 = Dummy('C0')

    # 定义测试用例列表
    tests = [
        # Very large values of m (989 and 991)
        (
            # 设置 Riccati ODE 方程
            Eq(f(x).diff(x), (1 - x)*f(x)/(x - 3) + (2 - 12*x)*f(x)**2/(2*x - 9) + \
                (54924*x**3 - 405264*x**2 + 1084347*x - 1087533)/(8*x**4 - 132*x**3 + 810*x**2 - \
                2187*x + 2187) + 495),
            # 期望的方程解
            [Eq(f(x), (18*x + 6)/(2*x - 9))]
        )
    ]

    # 对每个测试用例进行迭代
    for eq, sol in tests:
        # 调用函数 check_dummy_sol，检查使用 Dummy 变量 C0 的解是否符合预期解
        check_dummy_sol(eq, sol, C0)
```