# `D:\src\scipysrc\sympy\sympy\core\tests\test_exprtools.py`

```
"""Tests for tools for manipulating of large commutative expressions. """

# 导入所需的模块和类
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import (decompose_power, Factors, Term, _gcd_terms,
                                  gcd_terms, factor_terms, factor_nc, _mask_nc,
                                  _monotonic_sign)
from sympy.core.mul import _keep_coeff as _keep_coeff
from sympy.simplify.cse_opts import sub_pre
from sympy.testing.pytest import raises

# 从 sympy.abc 模块中导入符号变量
from sympy.abc import a, b, t, x, y, z


# 定义测试函数 test_decompose_power，测试 decompose_power 函数
def test_decompose_power():
    assert decompose_power(x) == (x, 1)
    assert decompose_power(x**2) == (x, 2)
    assert decompose_power(x**(2*y)) == (x**y, 2)
    assert decompose_power(x**(2*y/3)) == (x**(y/3), 2)
    assert decompose_power(x**(y*Rational(2, 3))) == (x**(y/3), 2)


# 定义测试函数 test_Factors，测试 Factors 类的各种方法
def test_Factors():
    assert Factors() == Factors({}) == Factors(S.One)
    assert Factors().as_expr() is S.One
    assert Factors({x: 2, y: 3, sin(x): 4}).as_expr() == x**2*y**3*sin(x)**4
    assert Factors(S.Infinity) == Factors({oo: 1})
    assert Factors(S.NegativeInfinity) == Factors({oo: 1, -1: 1})
    # issue #18059:
    assert Factors((x**2)**S.Half).as_expr() == (x**2)**S.Half

    a = Factors({x: 5, y: 3, z: 7})
    b = Factors({      y: 4, z: 3, t: 10})

    assert a.mul(b) == a*b == Factors({x: 5, y: 7, z: 10, t: 10})

    assert a.div(b) == divmod(a, b) == \
        (Factors({x: 5, z: 4}), Factors({y: 1, t: 10}))
    assert a.quo(b) == a/b == Factors({x: 5, z: 4})
    assert a.rem(b) == a % b == Factors({y: 1, t: 10})

    assert a.pow(3) == a**3 == Factors({x: 15, y: 9, z: 21})
    assert b.pow(3) == b**3 == Factors({y: 12, z: 9, t: 30})

    assert a.gcd(b) == Factors({y: 3, z: 3})
    assert a.lcm(b) == Factors({x: 5, y: 4, z: 7, t: 10})

    a = Factors({x: 4, y: 7, t: 7})
    b = Factors({z: 1, t: 3})

    assert a.normal(b) == (Factors({x: 4, y: 7, t: 4}), Factors({z: 1}))

    assert Factors(sqrt(2)*x).as_expr() == sqrt(2)*x

    assert Factors(-I)*I == Factors()
    assert Factors({S.NegativeOne: S(3)})*Factors({S.NegativeOne: S.One, I: S(5)}) == \
        Factors(I)
    assert Factors(sqrt(I)*I) == Factors(I**(S(3)/2)) == Factors({I: S(3)/2})
    assert Factors({I: S(3)/2}).as_expr() == I**(S(3)/2)
    # 断言：计算 S(2)**x 的因子化结果是否能整除 S(3)**x 的因子化结果
    assert Factors(S(2)**x).div(S(3)**x) == \
        (Factors({S(2): x}), Factors({S(3): x}))

    # 断言：计算 2**(2*x + 2) 的因子化结果是否能整除 S(8) 的因子化结果
    assert Factors(2**(2*x + 2)).div(S(8)) == \
        (Factors({S(2): 2*x + 2}), Factors({S(8): S.One}))

    # 覆盖测试
    # /!\ 如果不为 True，会导致其他问题
    assert Factors({S.NegativeOne: Rational(3, 2)}) == Factors({I: S.One, S.NegativeOne: S.One})

    # 断言：将 {I: S.One, S.NegativeOne: Rational(1, 3)} 的因子化结果转换为表达式是否正确
    assert Factors({I: S.One, S.NegativeOne: Rational(1, 3)}).as_expr() == I*(-1)**Rational(1, 3)

    # 断言：计算 -1.0 的因子化结果是否正确
    assert Factors(-1.) == Factors({S.NegativeOne: S.One, S(1.): 1})

    # 断言：计算 -2.0 的因子化结果是否正确
    assert Factors(-2.) == Factors({S.NegativeOne: S.One, S(2.): 1})

    # 断言：计算 (-2.0)**x 的因子化结果是否正确
    assert Factors((-2.)**x) == Factors({S(-2.): x})

    # 断言：计算 S(-2) 的因子化结果是否正确
    assert Factors(S(-2)) == Factors({S.NegativeOne: S.One, S(2): 1})

    # 断言：计算 S.Half 的因子化结果是否正确
    assert Factors(S.Half) == Factors({S(2): -S.One})

    # 断言：计算 Rational(3, 2) 的因子化结果是否正确
    assert Factors(Rational(3, 2)) == Factors({S(3): S.One, S(2): S.NegativeOne})

    # 断言：计算 {I: S.One} 的因子化结果是否正确
    assert Factors({I: S.One}) == Factors(I)

    # 断言：计算 {-1.0: 2, I: 1} 的因子化结果是否正确
    assert Factors({-1.0: 2, I: 1}) == Factors({S(1.0): 1, I: 1})

    # 断言：将 {S.NegativeOne: Rational(-3, 2)} 的因子化结果转换为表达式是否正确
    assert Factors({S.NegativeOne: Rational(-3, 2)}).as_expr() == I

    # 定义符号 A，并断言：计算 2*A**2 的因子化结果是否正确
    A = symbols('A', commutative=False)
    assert Factors(2*A**2) == Factors({S(2): 1, A**2: 1})

    # 断言：计算 I 的因子化结果是否正确
    assert Factors(I) == Factors({I: S.One})

    # 断言：对 x 进行以 S(2) 为基数的正常化操作是否正确
    assert Factors(x).normal(S(2)) == (Factors(x), Factors(S(2)))

    # 断言：对 x 进行以 S.Zero 为基数的正常化操作是否正确
    assert Factors(x).normal(S.Zero) == (Factors(), Factors(S.Zero))

    # 断言：对 x 进行以 S.Zero 为除数的操作是否会引发 ZeroDivisionError
    raises(ZeroDivisionError, lambda: Factors(x).div(S.Zero))

    # 断言：对 x 进行以 S(2) 为因子的乘法操作是否正确
    assert Factors(x).mul(S(2)) == Factors(2*x)

    # 断言：对 x 进行以 S.Zero 为因子的乘法操作结果是否为零
    assert Factors(x).mul(S.Zero).is_zero

    # 断言：对 x 进行以 1/x 为因子的乘法操作结果是否为一
    assert Factors(x).mul(1/x).is_one

    # 断言：对 x**sqrt(2)**3 进行转换为表达式操作结果是否正确
    assert Factors(x**sqrt(2)**3).as_expr() == x**(2*sqrt(2))

    # 断言：对 Factors(x)**Factors(S(2)) 进行操作结果是否正确
    assert Factors(x)**Factors(S(2)) == Factors(x**2)

    # 断言：对 x 进行以 S.Zero 为因子的 gcd 操作结果是否正确
    assert Factors(x).gcd(S.Zero) == Factors(x)

    # 断言：对 x 进行以 S.Zero 为因子的 lcm 操作结果是否为零
    assert Factors(x).lcm(S.Zero).is_zero

    # 断言：对 Factors(S.Zero) 进行以 x 为除数的操作结果是否正确
    assert Factors(S.Zero).div(x) == (Factors(S.Zero), Factors())

    # 断言：对 x 进行以 x 为除数的操作结果是否正确
    assert Factors(x).div(x) == (Factors(), Factors())

    # 断言：对 {x: .2} 和 {x: .2} 进行除法操作结果是否为空 Factors 对象
    assert Factors({x: .2})/Factors({x: .2}) == Factors()

    # 断言：对 x 进行 Factors 操作结果是否不等于空 Factors 对象
    assert Factors(x) != Factors()

    # 断言：对 Factors(S.Zero) 进行以 x 为基数的正常化操作是否正确
    assert Factors(S.Zero).normal(x) == (Factors(S.Zero), Factors())

    # 定义 n 和 d，分别计算 x**(2 + y) 的因子化结果
    n, d = x**(2 + y), x**2
    f = Factors(n)

    # 断言：对 Factors(n) 进行以 d 为除数的 div 和 normal 操作结果是否正确
    assert f.div(d) == f.normal(d) == (Factors(x**y), Factors())

    # 断言：对 Factors(n) 进行以 d 为因子的 gcd 操作结果是否正确
    assert f.gcd(d) == Factors()

    # 重新定义 d，计算 x**y 的因子化结果
    d = x**y

    # 断言：对 Factors(n) 进行以 d 为除数的 div 和 normal 操作结果是否正确
    assert f.div(d) == f.normal(d) == (Factors(x**2), Factors())

    # 断言：对 Factors(n) 进行以 d 为因子的 gcd 操作结果是否正确
    assert f.gcd(d) == Factors(d)

    # 重新定义 n 和 d，分别计算 2**x 的因子化结果
    n, d = 2**x, 2**y
    f = Factors(n)

    # 断言：对 Factors(n) 进行以 d 为除数的 div 和 normal 操作结果是否正确
    assert f.div(d) == f.normal(d) == (Factors({S(2): x}), Factors({S(2): y}))

    # 断言：对 Factors(n) 进行以 d 为因子的 gcd 操作结果是否正确
    assert f.gcd(d) == Factors()

    # 提取常数部分的测试
    n = x**(x + 3)

    # 断言：对 Factors(n) 进行以 x**-3 为基数的正常化操作是否正确
    assert Factors(n).normal(x**-3) == (Factors({x: x + 6}), Factors({}))

    # 断言：对 Factors(n) 进行以 x**3 为基数的正常化操作是否正确
    assert Factors(n).normal(x**3) == (Factors({x: x}), Factors({}))

    # 断言：对 Factors(n) 进行以 x**4 为基数的正常化操作是否正确
    assert Factors(n).normal(x**4) == (Factors({x: x}), Factors({x: 1}))

    # 断言：对 Factors(n) 进行以 x**(y - 3) 为基数的正常化操作是否正确
    assert Factors(n).normal(x**(
    # 断言语句，验证 Factors 对象调用 normal 方法后的返回结果是否符合预期
    assert Factors(n).normal(x**(y + 4)) == \
        (Factors({x: x}), Factors({x: y + 1}))

    # 断言语句，验证 Factors 对象调用 div 方法后的返回结果是否符合预期
    assert Factors(n).div(x**-3) == (Factors({x: x + 6}), Factors({}))
    # 断言语句，验证 Factors 对象调用 div 方法后的返回结果是否符合预期
    assert Factors(n).div(x**3) == (Factors({x: x}), Factors({}))
    # 断言语句，验证 Factors 对象调用 div 方法后的返回结果是否符合预期
    assert Factors(n).div(x**4) == (Factors({x: x}), Factors({x: 1}))
    # 断言语句，验证 Factors 对象调用 div 方法后的返回结果是否符合预期
    assert Factors(n).div(x**(y - 3)) == \
        (Factors({x: x + 6}), Factors({x: y}))
    # 断言语句，验证 Factors 对象调用 div 方法后的返回结果是否符合预期
    assert Factors(n).div(x**(y + 3)) == (Factors({x: x}), Factors({x: y}))
    # 断言语句，验证 Factors 对象调用 div 方法后的返回结果是否符合预期
    assert Factors(n).div(x**(y + 4)) == \
        (Factors({x: x}), Factors({x: y + 1}))

    # 断言语句，验证 Factors 对象初始化以及因数分解是否正确
    assert Factors(3 * x / 2) == Factors({3: 1, 2: -1, x: 1})
    # 断言语句，验证 Factors 对象初始化以及因数分解是否正确
    assert Factors(x * x / y) == Factors({x: 2, y: -1})
    # 断言语句，验证 Factors 对象初始化以及因数分解是否正确
    assert Factors(27 * x / y**9) == Factors({27: 1, x: 1, y: -9})
# 定义一个测试函数 test_Term，用于测试 Term 类的各种方法和操作
def test_Term():
    # 创建一个 Term 对象 a，表示数学表达式 4*x*y**2/z/t**3
    a = Term(4*x*y**2/z/t**3)
    # 创建一个 Term 对象 b，表示数学表达式 2*x**3*y**5/t**3
    b = Term(2*x**3*y**5/t**3)

    # 断言：验证对象 a 的值与指定的 Term 对象相等
    assert a == Term(4, Factors({x: 1, y: 2}), Factors({z: 1, t: 3}))
    # 断言：验证对象 b 的值与指定的 Term 对象相等
    assert b == Term(2, Factors({x: 3, y: 5}), Factors({t: 3}))

    # 断言：验证对象 a 转换为数学表达式的结果是否正确
    assert a.as_expr() == 4*x*y**2/z/t**3
    # 断言：验证对象 b 转换为数学表达式的结果是否正确
    assert b.as_expr() == 2*x**3*y**5/t**3

    # 断言：验证对象 a 的倒数方法是否正确计算
    assert a.inv() == \
        Term(S.One/4, Factors({z: 1, t: 3}), Factors({x: 1, y: 2}))
    # 断言：验证对象 b 的倒数方法是否正确计算
    assert b.inv() == Term(S.Half, Factors({t: 3}), Factors({x: 3, y: 5}))

    # 断言：验证对象 a 与对象 b 的乘法操作是否正确
    assert a.mul(b) == a*b == \
        Term(8, Factors({x: 4, y: 7}), Factors({z: 1, t: 6}))
    # 断言：验证对象 a 与对象 b 的除法操作是否正确
    assert a.quo(b) == a/b == Term(2, Factors({}), Factors({x: 2, y: 3, z: 1}))

    # 断言：验证对象 a 的幂运算方法是否正确计算
    assert a.pow(3) == a**3 == \
        Term(64, Factors({x: 3, y: 6}), Factors({z: 3, t: 9}))
    # 断言：验证对象 b 的幂运算方法是否正确计算
    assert b.pow(3) == b**3 == Term(8, Factors({x: 9, y: 15}), Factors({t: 9}))

    # 断言：验证对象 a 的负幂运算方法是否正确计算
    assert a.pow(-3) == a**(-3) == \
        Term(S.One/64, Factors({z: 3, t: 9}), Factors({x: 3, y: 6}))
    # 断言：验证对象 b 的负幂运算方法是否正确计算
    assert b.pow(-3) == b**(-3) == \
        Term(S.One/8, Factors({t: 9}), Factors({x: 9, y: 15}))

    # 断言：验证对象 a 与对象 b 的最大公约数方法是否正确计算
    assert a.gcd(b) == Term(2, Factors({x: 1, y: 2}), Factors({t: 3}))
    # 断言：验证对象 a 与对象 b 的最小公倍数方法是否正确计算
    assert a.lcm(b) == Term(4, Factors({x: 3, y: 5}), Factors({z: 1, t: 3}))

    # 重新赋值对象 a 和对象 b，进行新的乘法操作验证
    a = Term(4*x*y**2/z/t**3)
    b = Term(2*x**3*y**5*t**7)

    # 断言：验证新的对象 a 与对象 b 的乘法操作是否正确
    assert a.mul(b) == Term(8, Factors({x: 4, y: 7, t: 4}), Factors({z: 1}))

    # 断言：验证 Term 对象可以正确处理复杂表达式，例如 (2*x + 2)**3
    assert Term((2*x + 2)**3) == Term(8, Factors({x + 1: 3}), Factors({}))

    # 断言：验证 Term 对象可以正确处理更复杂的乘法表达式
    assert Term((2*x + 2)*(3*x + 6)**2) == \
        Term(18, Factors({x + 1: 1, x + 2: 2}), Factors({}))


# 定义一个测试函数 test_gcd_terms，用于测试 gcd_terms 函数的各种情况
def test_gcd_terms():
    # 创建一个复杂的数学表达式 f
    f = 2*(x + 1)*(x + 4)/(5*x**2 + 5) + (2*x + 2)*(x + 5)/(x**2 + 1)/5 + \
        (2*x + 2)*(x + 6)/(5*x**2 + 5)

    # 断言：验证 _gcd_terms 函数对 f 的处理结果是否正确
    assert _gcd_terms(f) == ((Rational(6, 5))*((1 + x)/(1 + x**2)), 5 + x, 1)
    # 断言：验证 _gcd_terms 函数对 f 转换为 Add 类型后的处理结果是否正确
    assert _gcd_terms(Add.make_args(f)) == \
        ((Rational(6, 5))*((1 + x)/(1 + x**2)), 5 + x, 1)

    # 创建一个新的表达式 newf，用于比较 gcd_terms 函数处理结果
    newf = (Rational(6, 5))*((1 + x)*(5 + x)/(1 + x**2))
    # 断言：验证 gcd_terms 函数对 f 的处理结果是否正确
    assert gcd_terms(f) == newf
    # 将 f 转换为各种容器形式，验证 gcd_terms 函数的处理结果是否一致
    args = Add.make_args(f)
    assert gcd_terms(list(args)) == newf
    assert gcd_terms(tuple(args)) == newf
    assert gcd_terms(set(args)) == newf
    # 使用 Basic 类型的数据验证 gcd_terms 函数的处理结果是否与预期一致
    assert gcd_terms(Tuple(*args)) != newf
    assert gcd_terms(Basic(Tuple(S(1), 3*y + 3*x*y), Tuple(S(1), S(3)))) == \
        Basic(Tuple(S(1), 3*y*(x + 1)), Tuple(S(1), S(3)))
    # 对于字典类型的数据，验证 gcd_terms 函数的处理结果是否保留所有键
    assert gcd_terms(Dict((x*(1 + y), S(2)), (x + x*y, y + x*y))) == \
        Dict({x*(y + 1): S(2), x + x*y: y*(1 + x)})

    # 断言：验证 gcd_terms 函数对复杂表达式的处理结果是否正确
    assert gcd_terms((2*x + 2)**3 + (2*x + 2)**2) == 4*(x + 1)**2*(2*x + 3)

    # 断言：验证 gcd_terms 函数对特殊输入值的处理结果
    assert gcd_terms(0) == 0
    assert gcd_terms(1) == 1
    assert gcd_terms(x) == x
    assert gcd_terms(2 + 2*x) == Mul(2, 1 + x, evaluate=False)
    arg = x*(2*x + 4*y)
    garg = 2*x*(x + 2*y)
    assert gcd_terms(arg) == garg
    assert gcd_terms(sin(arg)) == sin(garg)

    # 针对类似 issue 6139 的问题，验证 gcd_terms 函数的处理结果
    alpha, alpha1, alpha2, alpha3 = symbols('alpha:4')
    # 计算表达式 a 的值，其中 alpha 是一个变量
    a = alpha**2 - alpha*x**2 + alpha + x**3 - x*(alpha + 1)
    
    # 定义替换元组 rep，用于替换 alpha
    rep = (alpha, (1 + sqrt(5))/2 + alpha1*x + alpha2*x**2 + alpha3*x**3)
    
    # 对表达式 a 进行替换并在 x = 0 处展开成级数
    s = (a/(x - alpha)).subs(*rep).series(x, 0, 1)
    
    # 断言简化后的级数形式与预期结果相等
    assert simplify(collect(s, x)) == -sqrt(5)/2 - Rational(3, 2) + O(x)
    
    # 检查 _gcd_terms 函数在特定输入上的行为
    assert _gcd_terms([S.Zero, S.Zero]) == (0, 0, 1)
    assert _gcd_terms([2*x + 4]) == (2, x + 2, 1)
    
    # 测试 gcd_terms 函数在不同选项下的输出结果
    eq = x/(x + 1/x)
    assert gcd_terms(eq, fraction=False) == eq
    
    eq = x/2/y + 1/x/y
    # 使用 fraction=True 和 clear=True 参数调用 gcd_terms 函数
    assert gcd_terms(eq, fraction=True, clear=True) == (x**2 + 2)/(2*x*y)
    
    # 使用 fraction=True 和 clear=False 参数调用 gcd_terms 函数
    assert gcd_terms(eq, fraction=True, clear=False) == (x**2/2 + 1)/(x*y)
    
    # 使用 fraction=False 和 clear=True 参数调用 gcd_terms 函数
    assert gcd_terms(eq, fraction=False, clear=True) == (x + 2/x)/(2*y)
    
    # 使用 fraction=False 和 clear=False 参数调用 gcd_terms 函数
    assert gcd_terms(eq, fraction=False, clear=False) == (x/2 + 1/x)/y
# 定义一个测试函数 `test_factor_terms`
def test_factor_terms():
    # 创建一个非可交换符号 `A`
    A = Symbol('A', commutative=False)
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms(9*(x + x*y + 1) + (3*x + 3)**(2 + 2*x)) == \
        9*x*y + 9*x + _keep_coeff(S(3), x + 1)**_keep_coeff(S(2), x + 1) + 9
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms(9*(x + x*y + 1) + (3)**(2 + 2*x)) == \
        _keep_coeff(S(9), 3**(2*x) + x*y + x + 1
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms(3**(2 + 2*x) + a*3**(2 + 2*x)) == \
        9*3**(2*x)*(a + 1)
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms(x + x*A) == \
        x*(1 + A)
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms(sin(x + x*A)) == \
        sin(x*(1 + A))
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms((3*x + 3)**((2 + 2*x)/3)) == \
        _keep_coeff(S(3), x + 1)**_keep_coeff(Rational(2, 3), x + 1)
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms(x + (x*y + x)**(3*x + 3)) == \
        x + (x*(y + 1))**_keep_coeff(S(3), x + 1)
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms(a*(x + x*y) + b*(x*2 + y*x*2)) == \
        x*(a + 2*b)*(y + 1)
    
    # 创建一个积分对象 `i`
    i = Integral(x, (x, 0, oo))
    # 断言验证 `factor_terms` 函数对积分对象的处理结果
    assert factor_terms(i) == i
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果，使用 `fraction=False` 选项
    assert factor_terms(x/2 + y) == x/2 + y
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果，使用 `fraction=True` 选项
    # 整数分母不适用于分式处理
    assert factor_terms(x/2 + y, fraction=True) == x/2 + y
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果，使用 `clear=True` 选项
    # 清除整数分母的影响
    assert factor_terms(x/2 + y, clear=True) == Mul(S.Half, x + 2*y, evaluate=False)
    
    # 检查根式提取的处理
    eq = sqrt(2) + sqrt(10)
    assert factor_terms(eq) == eq
    assert factor_terms(eq, radical=True) == sqrt(2)*(1 + sqrt(5))
    
    eq = root(-6, 3) + root(6, 3)
    assert factor_terms(eq, radical=True) == 6**(S.One/3)*(1 + (-1)**(S.One/3))
    
    eq = [x + x*y]
    ans = [x*(y + 1)]
    
    # 针对列表、元组、集合的表达式，验证 `factor_terms` 函数的处理结果
    for c in [list, tuple, set]:
        assert factor_terms(c(eq)) == c(ans)
    
    # 断言验证 `factor_terms` 函数对元组的处理结果
    assert factor_terms(Tuple(x + x*y)) == Tuple(x*(y + 1))
    
    # 断言验证 `factor_terms` 函数对区间对象的处理结果
    assert factor_terms(Interval(0, 1)) == Interval(0, 1)
    
    # 创建一个表达式 `e`
    e = 1/sqrt(a/2 + 1)
    # 断言验证 `factor_terms` 函数对表达式 `e` 的处理结果，使用 `clear=False` 选项
    assert factor_terms(e, clear=False) == 1/sqrt(a/2 + 1)
    # 断言验证 `factor_terms` 函数对表达式 `e` 的处理结果，使用 `clear=True` 选项
    assert factor_terms(e, clear=True) == sqrt(2)/sqrt(a + 2)
    
    # 创建一个表达式 `eq`
    eq = x/(x + 1/x) + 1/(x**2 + 1)
    # 断言验证 `factor_terms` 函数对表达式 `eq` 的处理结果，使用 `fraction=False` 选项
    assert factor_terms(eq, fraction=False) == eq
    # 断言验证 `factor_terms` 函数对表达式 `eq` 的处理结果，使用 `fraction=True` 选项
    assert factor_terms(eq, fraction=True) == 1
    
    # 断言验证 `factor_terms` 函数对表达式的处理结果
    assert factor_terms((1/(x**3 + x**2) + 2/x**2)*y) == \
        y*(2 + 1/(x + 1))/x**2
    
    # 如果条件不成立，则 `factor_terms` 不需要处理
    assert gcd_terms(-x - y) == -x - y
    # 断言验证 `factor_terms` 函数对表达式的处理结果，确保符号 `-` 被保留
    assert factor_terms(-x - y) == Mul(-1, x + y, evaluate=False)
    
    # 如果条件不成立，则 `factor_terms` 中的特殊处理不需要
    assert gcd_terms(exp(Mul(-1, x + 1))) == exp(-x - 1)
    e = exp(-x - 2) + x
    # 断言验证 `factor_terms` 函数对表达式 `e` 的处理结果，保留 `-1` 和 `2`
    assert factor_terms(e) == exp(Mul(-1, x + 2, evaluate=False)) + x
    # 断言验证 `factor_terms` 函数对表达式 `e` 的处理结果，不进行符号处理
    assert factor_terms(e, sign=False) == e
    # 断言验证 `factor_terms` 函数对表达式的处理结果，保留 `-4` 和 `2`
    assert factor_terms(exp(-4*x - 2) - x) == -x + exp(Mul(-2, 2*x + 1, evaluate=False))
    
    # 对和与积分的测试
    for F in (Sum, Integral):
        # 断言验证 `factor_terms` 函数对和与积分对象的处理结果
        assert factor_terms(F(x, (y, 1, 10))) == x * F(1, (y, 1, 10))
        # 断言验证 `factor_terms` 函数对和与积分对象的处理结果，与常数 `x` 的乘积
        assert factor_terms(F(x, (y, 1, 10)) + x) == x * (1 + F(1, (y, 1, 10)))
        # 断言验证 `factor_terms` 函数对和与积分对象的处理结果，与变量 `x` 的乘积
        assert factor_terms(F(x*y + x*y**2, (y, 1, 10))) == x*F(y*(y + 1), (y, 1, 10))
    # 断言：验证调用 factor_terms 函数并传入参数 0**(x - 2) - 1 的结果是否等于 0**(x - 2) - 1
    assert factor_terms(0**(x - 2) - 1) == 0**(x - 2) - 1
    
    # 断言：验证调用 factor_terms 函数并传入参数 0**(x + 2) - 1 的结果是否等于 0**(x + 2) - 1
    assert factor_terms(0**(x + 2) - 1) == 0**(x + 2) - 1
    
    # 断言：验证调用 factor_terms 函数并传入参数 (0**(x + 2) - 1).subs(x,-2) 的结果是否等于 0
    assert factor_terms((0**(x + 2) - 1).subs(x,-2)) == 0
def test_xreplace():
    # 创建一个表达式 e = 2 * (1 + x)，但不进行求值
    e = Mul(2, 1 + x, evaluate=False)
    # 使用空字典替换表达式中的变量，预期结果与原始表达式相同
    assert e.xreplace({}) == e
    # 使用 {y: x} 替换表达式中的变量 y，预期结果与原始表达式相同
    assert e.xreplace({y: x}) == e


def test_factor_nc():
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 定义整数符号变量 k
    k = symbols('k', integer=True)
    # 定义非交换符号变量 n, m, o
    n, m, o = symbols('n,m,o', commutative=False)

    # 导入 _mexpand 函数用于乘积和多项式扩展
    from sympy.core.function import _mexpand
    # 创建表达式 e = x * (1 + y)**2
    e = x*(1 + y)**2
    # 断言 _mexpand(e) 的结果应该是 x + 2*x*y + x*y**2
    assert _mexpand(e) == x + 2*x*y + x*y**2

    # 定义一个用于测试 factor_nc 函数的嵌套函数
    def factor_nc_test(e):
        # 对表达式 e 进行乘积和多项式扩展
        ex = _mexpand(e)
        # 断言 ex 是一个加法表达式
        assert ex.is_Add
        # 对 ex 进行非交换因式分解
        f = factor_nc(ex)
        # 断言 f 不是加法表达式，并且 _mexpand(f) 等于 ex
        assert not f.is_Add and _mexpand(f) == ex

    # 对多种表达式进行 factor_nc_test 测试
    factor_nc_test(x*(1 + y))
    factor_nc_test(n*(x + 1))
    factor_nc_test(n*(x + m))
    factor_nc_test((x + m)*n)
    factor_nc_test(n*m*(x*o + n*o*m)*n)
    s = Sum(x, (x, 1, 2))
    factor_nc_test(x*(1 + s))
    factor_nc_test(x*(1 + s)*s)
    factor_nc_test(x*(1 + sin(s)))
    factor_nc_test((1 + n)**2)

    factor_nc_test((x + n)*(x + m)*(x + y))
    factor_nc_test(x*(n*m + 1))
    factor_nc_test(x*(n*m + x))
    factor_nc_test(x*(x*n*m + 1))
    factor_nc_test(n*(m/x + o))
    factor_nc_test(m*(n + o/2))
    factor_nc_test(x*n*(x*m + 1))
    factor_nc_test(x*(m*n + x*n*m))
    factor_nc_test(n*(1 - m)*n**2)

    factor_nc_test((n + m)**2)
    factor_nc_test((n - m)*(n + m)**2)
    factor_nc_test((n + m)**2*(n - m))
    factor_nc_test((m - n)*(n + m)**2*(n - m))

    # 断言 factor_nc(n*(n + n*m)) 的结果是 n**2*(1 + m)
    assert factor_nc(n*(n + n*m)) == n**2*(1 + m)
    # 断言 factor_nc(m*(m*n + n*m*n**2)) 的结果是 m*(m + n*m*n)*n
    assert factor_nc(m*(m*n + n*m*n**2)) == m*(m + n*m*n)*n
    # 创建等式 eq = m*sin(n) - sin(n)*m
    eq = m*sin(n) - sin(n)*m
    # 断言 factor_nc(eq) 的结果等于 eq
    assert factor_nc(eq) == eq

    # 对于覆盖测试:
    # 导入 Commutator 和 factor 函数
    from sympy.physics.secondquant import Commutator
    from sympy.polys.polytools import factor
    # 创建等式 eq = 1 + x*Commutator(m, n)
    eq = 1 + x*Commutator(m, n)
    # 断言 factor_nc(eq) 的结果等于 eq
    assert factor_nc(eq) == eq
    # 创建等式 eq = x*Commutator(m, n) + x*Commutator(m, o)*Commutator(m, n)
    eq = x*Commutator(m, n) + x*Commutator(m, o)*Commutator(m, n)
    # 断言 factor(eq) 的结果等于 x*(1 + Commutator(m, o))*Commutator(m, n)

    # issue 6534
    # 断言 (2*n + 2*m).factor() 的结果是 2*(n + m)

    # issue 6701
    # 定义非零且非交换的符号变量 _n
    _n = symbols('nz', zero=False, commutative=False)
    # 断言 factor_nc(_n**k + _n**(k + 1)) 的结果是 _n**k*(1 + _n)
    assert factor_nc(_n**k + _n**(k + 1)) == _n**k*(1 + _n)
    # 断言 factor_nc((m*n)**k + (m*n)**(k + 1)) 的结果是 (1 + m*n)*(m*n)**k

    # issue 6918
    # 断言 factor_nc(-n*(2*x**2 + 2*x)) 的结果是 -2*n*x*(x + 1)


def test_issue_6360():
    # 定义符号变量 a 和 b
    a, b = symbols("a b")
    # 定义 apb = a + b
    apb = a + b
    # 创建等式 eq = apb + apb**2*(-2*a - 2*b)
    eq = apb + apb**2*(-2*a - 2*b)
    # 断言 factor_terms(sub_pre(eq)) 的结果等于 a + b - 2*(a + b)**3


def test_issue_7903():
    # 定义符号变量 a，并指定其为实数
    a = symbols(r'a', real=True)
    # 创建等式 t = exp(I*cos(a)) + exp(-I*sin(a))
    t = exp(I*cos(a)) + exp(-I*sin(a))
    # 断言 t.simplify() 的结果为真


def test_issue_8263():
    # 定义非交换的符号函数 F 和 G
    F, G = symbols('F, G', commutative=False, cls=Function)
    # 定义符号变量 x 和 y
    x, y = symbols('x, y')
    # 对 _mask_nc(F(x)*G(y) - G(y)*F(x)) 进行解析，返回表达式 expr, dummies, _
    expr, dummies, _ = _mask_nc(F(x)*G(y) - G(y)*F(x))
    # 断言 dummies.values() 中的所有值都不是可交换的符号
    for v in dummies.values():
        assert not v.is_commutative
    # 断言 expr 不为零


def test_monotonic_sign():
    # 定义函数 F 为 _monotonic_sign
    F = _monotonic_sign
    # 定义符号变量 x
    x = symbols('x')
    # 断言 F(x) 的结果为 None
    assert F(x) is None
    # 断言 F(-x) 的结果为 None
    assert F(-x) is None
    # 断言 F(Dummy(prime=True)) 的结果为 2
    assert F(Dummy(prime=True)) == 2
    # 断言 F(Dummy(prime=True, odd=True)) 的结果为 3
    assert F(Dummy(prime=True, odd=True)) == 3
    # 断言 F(Dummy(composite=True)) 的结果为 4
    assert F(Dummy(composite=True)) == 4
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 9
    assert F(Dummy(composite=True, odd=True)) == 9
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 1
    assert F(Dummy(positive=True, integer=True)) == 1
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 2
    assert F(Dummy(positive=True, even=True)) == 2
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 4
    assert F(Dummy(positive=True, even=True, prime=False)) == 4
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 -1
    assert F(Dummy(negative=True, integer=True)) == -1
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 -2
    assert F(Dummy(negative=True, even=True)) == -2
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 0
    assert F(Dummy(zero=True)) == 0
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 0
    assert F(Dummy(nonnegative=True)) == 0
    # 断言语句，验证函数 F 对于 Dummy 对象的返回结果是否等于预期值 0
    assert F(Dummy(nonpositive=True)) == 0

    # 断言语句，验证函数 F 对于 Dummy 对象与整数 1 的和的结果是否为正数
    assert F(Dummy(positive=True) + 1).is_positive
    # 断言语句，验证函数 F 对于 Dummy 对象与整数 1 相减的结果是否为非负数
    assert F(Dummy(positive=True, integer=True) - 1).is_nonnegative
    # 断言语句，验证函数 F 对于 Dummy 对象与整数 1 相减的结果是否为 None
    assert F(Dummy(positive=True) - 1) is None
    # 断言语句，验证函数 F 对于 Dummy 对象与整数 1 的和的结果是否为 None
    assert F(Dummy(negative=True) + 1) is None
    # 断言语句，验证函数 F 对于 Dummy 对象与整数 1 相减的结果是否为非正数
    assert F(Dummy(negative=True, integer=True) - 1).is_nonpositive
    # 断言语句，验证函数 F 对于 Dummy 对象与整数 1 相减的结果是否为负数
    assert F(Dummy(negative=True) - 1).is_negative
    # 断言语句，验证函数 F 对于负数 Dummy 对象与整数 1 的和的结果是否为 None
    assert F(-Dummy(positive=True) + 1) is None
    # 断言语句，验证函数 F 对于负数 Dummy 对象与整数 1 相减的结果是否为负数
    assert F(-Dummy(positive=True, integer=True) - 1).is_negative
    # 断言语句，验证函数 F 对于负数 Dummy 对象与整数 1 相减的结果是否为负数
    assert F(-Dummy(positive=True) - 1).is_negative
    # 断言语句，验证函数 F 对于负数 Dummy 对象与整数 1 的和的结果是否为正数
    assert F(-Dummy(negative=True) + 1).is_positive
    # 断言语句，验证函数 F 对于负数 Dummy 对象与整数 1 相减的结果是否为非负数
    assert F(-Dummy(negative=True, integer=True) - 1).is_nonnegative
    # 断言语句，验证函数 F 对于负数 Dummy 对象与整数 1 相减的结果是否为 None
    assert F(-Dummy(negative=True) - 1) is None
    # 变量赋值语句，将 Dummy(negative=True) 的结果赋给变量 x
    x = Dummy(negative=True)
    # 断言语句，验证函数 F 对于 x 的立方的结果是否为非正数
    assert F(x**3).is_nonpositive
    # 断言语句，验证函数 F 对于 x 的立方加上 log(2)*x 减去 1 的结果是否为负数
    assert F(x**3 + log(2)*x - 1).is_negative
    # 变量赋值语句，将 Dummy(positive=True) 的结果赋给变量 x
    x = Dummy(positive=True)
    # 断言语句，验证函数 F 对于负数 x 的立方的结果是否为非正数
    assert F(-x**3).is_nonpositive

    # 变量赋值语句，将 Dummy(positive=True) 的结果赋给变量 p
    p = Dummy(positive=True)
    # 断言语句，验证函数 F 对于 1/p 的结果是否为正数
    assert F(1/p).is_positive
    # 断言语句，验证函数 F 对于 p/(p + 1) 的结果是否为正数
    assert F(p/(p + 1)).is_positive
    # 变量赋值语句，将 Dummy(nonnegative=True) 的结果赋给变量 p
    p = Dummy(nonnegative=True)
    # 断言语句，验证函数 F 对于 p/(p + 1) 的结果是否为非负数
    assert F(p/(p + 1)).is_nonnegative
    # 变量赋值语句，将 Dummy(positive=True) 的结果赋给变量 p
    p = Dummy(positive=True)
    # 断言语句，验证函数 F 对于 -1/p 的结果是否为负数
    assert F(-1/p).is_negative
    # 变量赋值语句，将 Dummy(nonpositive=True) 的结果赋给变量 p
    p = Dummy(nonpositive=True)
    # 断言语句，验证函数 F 对于 p/(-p + 1) 的结果是否为非正数
    assert F(p/(-p + 1)).is_nonpositive

    # 变量赋值语句，将 Dummy(positive=True, integer=True) 的结果赋给变量 p
    p = Dummy(positive=True, integer=True)
    # 变量赋值语句，将 Dummy(positive=True, integer=True) 的结果赋给变量 q
    q = Dummy(positive=True, integer=True)
    # 断言语句，验证函数 F 对于 -2/p/q 的结果是否为负数
    assert F(-2/p/q).is_negative
    # 断言语句，验证函数 F 对于 -2/(p - 1)/q 的结果是否为 None
    assert F(-2/(p - 1)/q) is None

    # 断言语句，验证函数 F 对于 (p - 1)*q + 1 的结果是否为正数
    assert F((p - 1)*q + 1).is_positive
    # 断言语句，验证函数 F 对于 -(p - 1)*q - 1 的结果是否为负数
    assert F(-(p - 1)*q - 1).is_negative
# 定义一个测试函数，用于测试问题编号为 17256 的问题
def test_issue_17256():
    # 从 sympy.sets.fancysets 模块中导入 Range 类
    from sympy.sets.fancysets import Range
    # 创建一个符号变量 x
    x = Symbol('x')
    # 创建一个求和表达式 s1，求和范围为 x 从 1 到 9
    s1 = Sum(x + 1, (x, 1, 9))
    # 创建一个求和表达式 s2，求和范围为 x 属于 Range(1, 10)
    s2 = Sum(x + 1, (x, Range(1, 10)))
    # 创建一个符号变量 a
    a = Symbol('a')
    # 使用 xreplace 方法替换 s1 中的 x 为 a，得到替换后的结果 r1
    r1 = s1.xreplace({x:a})
    # 使用 xreplace 方法替换 s2 中的 x 为 a，得到替换后的结果 r2
    r2 = s2.xreplace({x:a})

    # 断言两个表达式 r1 和 r2 经过求值（doit()）后的结果应该相等
    assert r1.doit() == r2.doit()

    # 重新定义求和表达式 s1，求和范围为 x 从 0 到 9
    s1 = Sum(x + 1, (x, 0, 9))
    # 重新定义求和表达式 s2，求和范围为 x 属于 Range(10)
    s2 = Sum(x + 1, (x, Range(10)))
    # 重新创建一个符号变量 a
    a = Symbol('a')
    # 使用 xreplace 方法替换 s1 中的 x 为 a，得到替换后的结果 r1
    r1 = s1.xreplace({x:a})
    # 使用 xreplace 方法替换 s2 中的 x 为 a，得到替换后的结果 r2
    r2 = s2.xreplace({x:a})
    # 断言两个表达式 r1 和 r2 相等
    assert r1 == r2

# 定义一个测试函数，用于测试问题编号为 21623 的问题
def test_issue_21623():
    # 从 sympy.matrices.expressions.matexpr 模块中导入 MatrixSymbol 类
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    # 创建一个名为 M 的 2x2 矩阵符号
    M = MatrixSymbol('X', 2, 2)
    # 断言 gcd_terms 函数应用于 M[0,0] 和 1 的结果等于 M[0,0] 自身
    assert gcd_terms(M[0,0], 1) == M[0,0]
```