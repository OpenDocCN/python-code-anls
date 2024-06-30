# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\test_hyperbolic.py`

```
from sympy.calculus.accumulationbounds import AccumBounds  # 导入AccumBounds类，用于计算区间累积边界
from sympy.core.function import (expand_mul, expand_trig)  # 导入函数expand_mul和expand_trig，用于展开乘法和三角函数
from sympy.core.numbers import (E, I, Integer, Rational, nan, oo, pi, zoo)  # 导入常数E、I、Integer、Rational等
from sympy.core.singleton import S  # 导入S单例，表示符号表达式中的一个特殊符号
from sympy.core.symbol import (Symbol, symbols)  # 导入符号类Symbol和symbols函数，用于定义符号变量
from sympy.functions.elementary.complexes import (im, re)  # 导入虚部和实部函数im和re
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数和对数函数exp和log
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh, tanh)  # 导入双曲函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, cot, sec, sin, tan)  # 导入三角函数
from sympy.series.order import O  # 导入O类，表示大O符号

from sympy.core.expr import unchanged  # 导入unchanged函数，用于测试函数是否未改变
from sympy.core.function import ArgumentIndexError, PoleError  # 导入异常类ArgumentIndexError和PoleError
from sympy.testing.pytest import raises  # 导入raises函数，用于测试函数是否引发异常


def test_sinh():
    x, y = symbols('x,y')  # 定义符号变量x和y

    k = Symbol('k', integer=True)  # 定义整数符号变量k

    assert sinh(nan) is nan  # 断言sinh函数对nan返回nan
    assert sinh(zoo) is nan  # 断言sinh函数对zoo（无穷大）返回nan

    assert sinh(oo) is oo  # 断言sinh函数对无穷大返回无穷大
    assert sinh(-oo) is -oo  # 断言sinh函数对负无穷大返回负无穷大

    assert sinh(0) == 0  # 断言sinh函数对0返回0

    assert unchanged(sinh, 1)  # 断言sinh函数对1未改变
    assert sinh(-1) == -sinh(1)  # 断言sinh函数对-1等于-sinh(1)

    assert unchanged(sinh, x)  # 断言sinh函数对x未改变
    assert sinh(-x) == -sinh(x)  # 断言sinh函数对-x等于-sinh(x)

    assert unchanged(sinh, pi)  # 断言sinh函数对π未改变
    assert sinh(-pi) == -sinh(pi)  # 断言sinh函数对-π等于-sinh(π)

    assert unchanged(sinh, 2**1024 * E)  # 断言sinh函数对2^1024 * E未改变
    assert sinh(-2**1024 * E) == -sinh(2**1024 * E)  # 断言sinh函数对-2^1024 * E等于-sinh(2^1024 * E)

    assert sinh(pi*I) == 0  # 断言sinh函数对π*i等于0
    assert sinh(-pi*I) == 0  # 断言sinh函数对-π*i等于0
    assert sinh(2*pi*I) == 0  # 断言sinh函数对2π*i等于0
    assert sinh(-2*pi*I) == 0  # 断言sinh函数对-2π*i等于0
    assert sinh(-3*10**73*pi*I) == 0  # 断言sinh函数对-3*10^73π*i等于0
    assert sinh(7*10**103*pi*I) == 0  # 断言sinh函数对7*10^103π*i等于0

    assert sinh(pi*I/2) == I  # 断言sinh函数对π*i/2等于i
    assert sinh(-pi*I/2) == -I  # 断言sinh函数对-π*i/2等于-i
    assert sinh(pi*I*Rational(5, 2)) == I  # 断言sinh函数对π*i*5/2等于i
    assert sinh(pi*I*Rational(7, 2)) == -I  # 断言sinh函数对π*i*7/2等于-i

    assert sinh(pi*I/3) == S.Half*sqrt(3)*I  # 断言sinh函数对π*i/3等于1/2*sqrt(3)*i
    assert sinh(pi*I*Rational(-2, 3)) == Rational(-1, 2)*sqrt(3)*I  # 断言sinh函数对π*i*(-2/3)等于-1/2*sqrt(3)*i

    assert sinh(pi*I/4) == S.Half*sqrt(2)*I  # 断言sinh函数对π*i/4等于1/2*sqrt(2)*i
    assert sinh(-pi*I/4) == Rational(-1, 2)*sqrt(2)*I  # 断言sinh函数对-π*i/4等于-1/2*sqrt(2)*i
    assert sinh(pi*I*Rational(17, 4)) == S.Half*sqrt(2)*I  # 断言sinh函数对π*i*17/4等于1/2*sqrt(2)*i
    assert sinh(pi*I*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)*I  # 断言sinh函数对π*i*(-3/4)等于-1/2*sqrt(2)*i

    assert sinh(pi*I/6) == S.Half*I  # 断言sinh函数对π*i/6等于1/2*i
    assert sinh(-pi*I/6) == Rational(-1, 2)*I  # 断言sinh函数对-π*i/6等于-1/2*i
    assert sinh(pi*I*Rational(7, 6)) == Rational(-1, 2)*I  # 断言sinh函数对π*i*7/6等于-1/2*i
    assert sinh(pi*I*Rational(-5, 6)) == Rational(-1, 2)*I  # 断言sinh函数对π*i*(-5/6)等于-1/2*i

    assert sinh(pi*I/105) == sin(pi/105)*I  # 断言sinh函数对π*i/105等于sin(π/105)*i
    assert sinh(-pi*I/105) == -sin(pi/105)*I  # 断言sinh函数对-π*i/105等于-sin(π/105)*i

    assert unchanged(sinh, 2 + 3*I)  # 断言sinh函数对2 + 3*i未改变

    assert sinh(x*I) == sin(x)*I  # 断言sinh函数对x*i等于sin(x)*i

    assert sinh(k*pi*I) == 0  # 断言sinh函数对k*π*i等于0
    assert sinh(17*k*pi*I) == 0  # 断言sinh函数对17*k*π*i等于0

    assert sinh(k*pi*I/2) == sin(k*pi/2)*I  # 断言sinh函数对k*π*i/2等于sin(k*π/2)*i

    assert sinh(x).as_real_imag(deep=False) == (cos(im(x))*sinh(re(x)),  # 断言sinh函数对x的实部和虚部（深度为False）等于(cos(im(x))*sinh(re(x)),
                sin(im(x))*cosh(re(x)))  # sin(im(x))*cosh(re(x)))

    x = Symbol('x', extended_real=True)
    assert sinh(x).as_real_imag(deep=False) == (sinh(x), 0)  # 断言sinh函数对扩展实数域的x的实部和虚部（深度为False）等于(sinh(x), 0)

    x = Symbol('x', real=True)
    assert sinh(I*x).
    # 导入的数学函数 sinh、pi、I 来自 sympy 模块，通过禁止求值参数 evaluate=False 使用高级数学计算
    assert sinh(2*pi*I, evaluate=False).is_zero is True
# 定义测试函数 test_sinh_series，测试 sinh 函数的级数展开是否正确
def test_sinh_series():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言 sinh(x) 在 x=0 处展开到 10 阶的级数表达式
    assert sinh(x).series(x, 0, 10) == \
        x + x**3/6 + x**5/120 + x**7/5040 + x**9/362880 + O(x**10)


# 定义测试函数 test_sinh_fdiff，测试 sinh 函数的二阶导数是否引发 ArgumentIndexError 异常
def test_sinh_fdiff():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言对 sinh(x) 求二阶导数引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: sinh(x).fdiff(2))


# 定义测试函数 test_cosh，测试 cosh 函数的多种输入和性质
def test_cosh():
    # 创建符号变量 x 和 y
    x, y = symbols('x,y')

    # 创建整数符号变量 k
    k = Symbol('k', integer=True)

    # 断言 cosh(nan) 是 nan
    assert cosh(nan) is nan
    # 断言 cosh(zoo) 是 nan
    assert cosh(zoo) is nan

    # 断言 cosh(oo) 是 oo
    assert cosh(oo) is oo
    # 断言 cosh(-oo) 是 oo
    assert cosh(-oo) is oo

    # 断言 cosh(0) 等于 1
    assert cosh(0) == 1

    # 断言 cosh 函数不变性测试，cosh(1) 等于 cosh(-1)
    assert unchanged(cosh, 1)
    assert cosh(-1) == cosh(1)

    # 断言 cosh 函数对 x 的不变性，cosh(x) 等于 cosh(-x)
    assert unchanged(cosh, x)
    assert cosh(-x) == cosh(x)

    # 断言 cosh(pi*I) 等于 cos(pi)
    assert cosh(pi*I) == cos(pi)
    assert cosh(-pi*I) == cos(pi)

    # 断言 cosh 函数不变性测试，cosh(2**1024 * E) 等于 cosh(-2**1024 * E)
    assert unchanged(cosh, 2**1024 * E)
    assert cosh(-2**1024 * E) == cosh(2**1024 * E)

    # 断言 cosh(pi*I/2) 等于 0
    assert cosh(pi*I/2) == 0
    assert cosh(-pi*I/2) == 0
    assert cosh((-3*10**73 + 1)*pi*I/2) == 0
    assert cosh((7*10**103 + 1)*pi*I/2) == 0

    # 断言 cosh(pi*I) 等于 -1
    assert cosh(pi*I) == -1
    assert cosh(-pi*I) == -1
    assert cosh(5*pi*I) == -1
    assert cosh(8*pi*I) == 1

    # 断言 cosh(pi*I/3) 等于 S.Half
    assert cosh(pi*I/3) == S.Half
    assert cosh(pi*I*Rational(-2, 3)) == Rational(-1, 2)

    # 断言 cosh(pi*I/4) 等于 S.Half*sqrt(2)
    assert cosh(pi*I/4) == S.Half*sqrt(2)
    assert cosh(-pi*I/4) == S.Half*sqrt(2)
    assert cosh(pi*I*Rational(11, 4)) == Rational(-1, 2)*sqrt(2)
    assert cosh(pi*I*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    # 断言 cosh(pi*I/6) 等于 S.Half*sqrt(3)
    assert cosh(pi*I/6) == S.Half*sqrt(3)
    assert cosh(-pi*I/6) == S.Half*sqrt(3)
    assert cosh(pi*I*Rational(7, 6)) == Rational(-1, 2)*sqrt(3)
    assert cosh(pi*I*Rational(-5, 6)) == Rational(-1, 2)*sqrt(3)

    # 断言 cosh(pi*I/105) 等于 cos(pi/105)
    assert cosh(pi*I/105) == cos(pi/105)
    assert cosh(-pi*I/105) == cos(pi/105)

    # 断言 cosh 函数不变性测试，cosh(2 + 3*I) 不变
    assert unchanged(cosh, 2 + 3*I)

    # 断言 cosh(x*I) 等于 cos(x)
    assert cosh(x*I) == cos(x)

    # 断言 cosh(k*pi*I) 等于 cos(k*pi)
    assert cosh(k*pi*I) == cos(k*pi)
    assert cosh(17*k*pi*I) == cos(17*k*pi)

    # 断言 cosh 函数不变性测试，cosh(k*pi) 不变
    assert unchanged(cosh, k*pi)

    # 断言 cosh(x).as_real_imag(deep=False) 的实部和虚部的结果
    assert cosh(x).as_real_imag(deep=False) == (cos(im(x))*cosh(re(x)),
                sin(im(x))*sinh(re(x)))

    # 当 x 为 extended_real 类型时，断言 cosh(x).as_real_imag(deep=False) 的结果
    x = Symbol('x', extended_real=True)
    assert cosh(x).as_real_imag(deep=False) == (cosh(x), 0)

    # 当 x 为 real 类型时，对 cosh(I*x) 的性质进行验证
    x = Symbol('x', real=True)
    assert cosh(I*x).is_finite is True
    assert cosh(I*x).is_real is True
    assert cosh(I*2 + 1).is_real is False
    assert cosh(5*I*S.Pi/2, evaluate=False).is_zero is True
    assert cosh(x).is_zero is False


# 定义测试函数 test_cosh_series，测试 cosh 函数的级数展开是否正确
def test_cosh_series():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言 cosh(x) 在 x=0 处展开到 10 阶的级数表达式
    assert cosh(x).series(x, 0, 10) == \
        1 + x**2/2 + x**4/24 + x**6/720 + x**8/40320 + O(x**10)


# 定义测试函数 test_cosh_fdiff，测试 cosh 函数的二阶导数是否引发 ArgumentIndexError 异常
def test_cosh_fdiff():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言对 cosh(x) 求二阶导数引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: cosh(x).fdiff(2))


# 定义测试函数 test_tanh，测试 tanh 函数的多种输入和性质
def test_tanh():
    # 创建符号变量 x 和 y
    x, y = symbols('x,y')

    # 创建整数符号变量 k
    k = Symbol('k', integer=True)

    # 断言 tanh(nan) 是 nan
    assert tanh(nan) is nan
    # 断言 tanh(zoo) 是 nan
    assert tanh(zoo) is nan

    # 断言 tanh(oo) 等于 1
    assert tanh(oo) == 1
    # 断言 tanh(-oo) 等于 -1
    assert tanh(-oo) == -1

    # 断言 tanh(0) 等于 0
    assert tanh(0) == 0

    # 断言 tanh 函数不变性测试，tanh(1) 等于 -tanh(-1)
    assert unchanged(tanh, 1)
    assert tanh(-1) == -tanh(1)

    # 断言 tanh 函数对 x 的不变性，tanh(x) 等于 -tanh(-x)
    assert unchanged(tanh, x)
    assert tanh(-x) == -tanh(x)

    # 断言 tanh 函数对 pi 的不变性，tanh(pi
    # 断言：双曲正切函数在复数域上的性质
    assert tanh(-2**1024 * E) == -tanh(2**1024 * E)

    # 断言：双曲正切函数在虚数域上的性质
    assert tanh(pi*I) == 0
    assert tanh(-pi*I) == 0
    assert tanh(2*pi*I) == 0
    assert tanh(-2*pi*I) == 0
    assert tanh(-3*10**73*pi*I) == 0
    assert tanh(7*10**103*pi*I) == 0

    # 断言：双曲正切函数在复数域上的性质，极限情况处理
    assert tanh(pi*I/2) is zoo
    assert tanh(-pi*I/2) is zoo
    assert tanh(pi*I*Rational(5, 2)) is zoo
    assert tanh(pi*I*Rational(7, 2)) is zoo

    # 断言：双曲正切函数在复数域上的性质，特定角度的计算结果
    assert tanh(pi*I/3) == sqrt(3)*I
    assert tanh(pi*I*Rational(-2, 3)) == sqrt(3)*I

    # 断言：双曲正切函数在复数域上的性质，特定角度的计算结果
    assert tanh(pi*I/4) == I
    assert tanh(-pi*I/4) == -I
    assert tanh(pi*I*Rational(17, 4)) == I
    assert tanh(pi*I*Rational(-3, 4)) == I

    # 断言：双曲正切函数在复数域上的性质，特定角度的计算结果
    assert tanh(pi*I/6) == I/sqrt(3)
    assert tanh(-pi*I/6) == -I/sqrt(3)
    assert tanh(pi*I*Rational(7, 6)) == I/sqrt(3)
    assert tanh(pi*I*Rational(-5, 6)) == I/sqrt(3)

    # 断言：双曲正切函数在复数域上的性质，与正切函数的关系
    assert tanh(pi*I/105) == tan(pi/105)*I
    assert tanh(-pi*I/105) == -tan(pi/105)*I

    # 断言：双曲正切函数不受实部影响的性质
    assert unchanged(tanh, 2 + 3*I)

    # 断言：双曲正切函数在复数域上的性质，与正切函数的关系
    assert tanh(x*I) == tan(x)*I

    # 断言：双曲正切函数在复数域上的性质，特定倍数角的计算结果
    assert tanh(k*pi*I) == 0
    assert tanh(17*k*pi*I) == 0

    # 断言：双曲正切函数在复数域上的性质，特定倍数角的计算结果
    assert tanh(k*pi*I/2) == tan(k*pi/2)*I

    # 断言：双曲正切函数的实部和虚部表示
    assert tanh(x).as_real_imag(deep=False) == (
        sinh(re(x))*cosh(re(x))/(cos(im(x))**2 + sinh(re(x))**2),
        sin(im(x))*cos(im(x))/(cos(im(x))**2 + sinh(re(x))**2)
    )

    # 符号变量定义
    x = Symbol('x', extended_real=True)

    # 断言：双曲正切函数的实部和虚部表示
    assert tanh(x).as_real_imag(deep=False) == (tanh(x), 0)

    # 断言：双曲正切函数的实部和虚部表示，以及实数性质
    assert tanh(I*pi/3 + 1).is_real is False
    assert tanh(x).is_real is True
    assert tanh(I*pi*x/2).is_real is None
# 定义测试函数，用于验证双曲正切函数的级数展开是否正确
def test_tanh_series():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言双曲正切函数在 x=0 处展开的前10项级数是否符合预期
    assert tanh(x).series(x, 0, 10) == \
        x - x**3/3 + 2*x**5/15 - 17*x**7/315 + 62*x**9/2835 + O(x**10)


# 定义测试函数，用于验证双曲正切函数的二阶导数是否能够引发参数索引错误
def test_tanh_fdiff():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言尝试对双曲正切函数的二阶导数操作是否会引发参数索引错误异常
    raises(ArgumentIndexError, lambda: tanh(x).fdiff(2))


# 定义测试函数，用于验证余切双曲函数的多个数学性质和特定值
def test_coth():
    # 创建符号变量 x, y
    x, y = symbols('x,y')

    # 创建整数符号变量 k
    k = Symbol('k', integer=True)

    # 断言若输入非数值时余切双曲函数应返回 nan
    assert coth(nan) is nan
    assert coth(zoo) is nan

    # 断言余切双曲函数在正无穷大和负无穷大时的极限值
    assert coth(oo) == 1
    assert coth(-oo) == -1

    # 断言余切双曲函数在零点处的极限值和部分数学性质
    assert coth(0) is zoo
    assert unchanged(coth, 1)
    assert coth(-1) == -coth(1)

    # 断言余切双曲函数的奇偶性质
    assert unchanged(coth, x)
    assert coth(-x) == -coth(x)

    # 断言余切双曲函数与余切函数之间的关系
    assert coth(pi*I) == -I*cot(pi)
    assert coth(-pi*I) == cot(pi)*I

    # 断言余切双曲函数在特定复数域的表现
    assert unchanged(coth, 2**1024 * E)
    assert coth(-2**1024 * E) == -coth(2**1024 * E)

    # 断言余切双曲函数在不同复数情境下的性质
    assert coth(pi*I/2) == 0
    assert coth(-pi*I/2) == 0
    assert coth(pi*I*Rational(5, 2)) == 0
    assert coth(pi*I*Rational(7, 2)) == 0

    # 断言余切双曲函数在复平面的三角特定角度值的表现
    assert coth(pi*I/3) == -I/sqrt(3)
    assert coth(pi*I*Rational(-2, 3)) == -I/sqrt(3)

    # 断言余切双曲函数在复平面的特定角度值的表现
    assert coth(pi*I/4) == -I
    assert coth(-pi*I/4) == I
    assert coth(pi*I*Rational(17, 4)) == -I
    assert coth(pi*I*Rational(-3, 4)) == -I

    # 断言余切双曲函数在复平面的特定角度值的表现
    assert coth(pi*I/6) == -sqrt(3)*I
    assert coth(-pi*I/6) == sqrt(3)*I
    assert coth(pi*I*Rational(7, 6)) == -sqrt(3)*I
    assert coth(pi*I*Rational(-5, 6)) == -sqrt(3)*I

    # 断言余切双曲函数在复平面的特定角度值的表现
    assert coth(pi*I/105) == -cot(pi/105)*I
    assert coth(-pi*I/105) == cot(pi/105)*I

    # 断言余切双曲函数在复平面的数值性质
    assert unchanged(coth, 2 + 3*I)

    # 断言余切双曲函数与余切函数的关系
    assert coth(x*I) == -cot(x)*I

    # 断言余切双曲函数在复平面的倍数关系
    assert coth(k*pi*I) == -cot(k*pi)*I
    assert coth(17*k*pi*I) == -cot(17*k*pi)*I

    # 断言余切双曲函数与余切函数的关系
    assert coth(k*pi*I) == -cot(k*pi)*I

    # 断言余切双曲函数与余切函数的对数关系
    assert coth(log(tan(2))) == coth(log(-tan(2)))
    assert coth(1 + I*pi/2) == tanh(1)

    # 断言余切双曲函数的实部和虚部的展开
    assert coth(x).as_real_imag(deep=False) == (sinh(re(x))*cosh(re(x))/(sin(im(x))**2
                                + sinh(re(x))**2),
                                -sin(im(x))*cos(im(x))/(sin(im(x))**2 + sinh(re(x))**2))
    # 创建扩展实数域的符号变量 x
    x = Symbol('x', extended_real=True)
    # 断言余切双曲函数的实部和虚部的展开
    assert coth(x).as_real_imag(deep=False) == (coth(x), 0)

    # 断言余切双曲函数的三角展开
    assert expand_trig(coth(2*x)) == (coth(x)**2 + 1)/(2*coth(x))
    assert expand_trig(coth(3*x)) == (coth(x)**3 + 3*coth(x))/(1 + 3*coth(x)**2)

    # 断言余切双曲函数的三角展开
    assert expand_trig(coth(x + y)) == (1 + coth(x)*coth(y))/(coth(x) + coth(y))


# 定义测试函数，用于验证余双曲函数的级数展开是否正确
def test_coth_series():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言余切双曲函数在 x=0 处展开的前8项级数是否符合预期
    assert coth(x).series(x, 0, 8) == \
        1/x + x/3 - x**3/45 + 2*x**5/945 - x**7/4725 + O(x**8)


# 定义测试函数，用于验证余切双曲函数的二阶导数是否能够引发参数索引错误
def test_coth_fdiff():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言尝试对余切双曲函数的二阶导数操作是否会引发参数索引错误异常
    raises(ArgumentIndexError, lambda: coth(x).fdiff(2))


# 定义测试函数，用于验证余弦双曲函数的多个数学性质和特定值
def test_csch():
    # 创建符号变量 x, y
    x, y = symbols('x,y')

    # 创建整数符号变量 k
    k = Symbol('k', integer=True)

    # 创建正数符号变量 n
    n = Symbol('n', positive=True)

    # 断言若输入非数值时余弦双曲函数应返回 nan
    assert csch(nan) is nan
    assert csch(zoo) is nan

    # 断言余弦双曲函数在正无穷大时的极限值
    assert csch(oo) == 0
    # 断言 csch(-oo) 等于 0
    assert csch(-oo) == 0

    # 断言 csch(0) 是无穷大 (zoo)
    assert csch(0) is zoo

    # 断言 csch(-1) 等于 -csch(1)
    assert csch(-1) == -csch(1)

    # 断言 csch(-x) 等于 -csch(x)
    assert csch(-x) == -csch(x)
    # 断言 csch(-pi) 等于 -csch(pi)
    assert csch(-pi) == -csch(pi)
    # 断言 csch(-2**1024 * E) 等于 -csch(2**1024 * E)
    assert csch(-2**1024 * E) == -csch(2**1024 * E)

    # 断言 csch(pi*I) 是无穷大 (zoo)
    assert csch(pi*I) is zoo
    # 断言 csch(-pi*I) 是无穷大 (zoo)
    assert csch(-pi*I) is zoo
    # 断言 csch(2*pi*I) 是无穷大 (zoo)
    assert csch(2*pi*I) is zoo
    # 断言 csch(-2*pi*I) 是无穷大 (zoo)
    assert csch(-2*pi*I) is zoo
    # 断言 csch(-3*10**73*pi*I) 是无穷大 (zoo)
    assert csch(-3*10**73*pi*I) is zoo
    # 断言 csch(7*10**103*pi*I) 是无穷大 (zoo)
    assert csch(7*10**103*pi*I) is zoo

    # 断言 csch(pi*I/2) 等于 -I
    assert csch(pi*I/2) == -I
    # 断言 csch(-pi*I/2) 等于 I
    assert csch(-pi*I/2) == I
    # 断言 csch(pi*I*Rational(5, 2)) 等于 -I
    assert csch(pi*I*Rational(5, 2)) == -I
    # 断言 csch(pi*I*Rational(7, 2)) 等于 I
    assert csch(pi*I*Rational(7, 2)) == I

    # 断言 csch(pi*I/3) 等于 -2/sqrt(3)*I
    assert csch(pi*I/3) == -2/sqrt(3)*I
    # 断言 csch(pi*I*Rational(-2, 3)) 等于 2/sqrt(3)*I
    assert csch(pi*I*Rational(-2, 3)) == 2/sqrt(3)*I

    # 断言 csch(pi*I/4) 等于 -sqrt(2)*I
    assert csch(pi*I/4) == -sqrt(2)*I
    # 断言 csch(-pi*I/4) 等于 sqrt(2)*I
    assert csch(-pi*I/4) == sqrt(2)*I
    # 断言 csch(pi*I*Rational(7, 4)) 等于 sqrt(2)*I
    assert csch(pi*I*Rational(7, 4)) == sqrt(2)*I
    # 断言 csch(pi*I*Rational(-3, 4)) 等于 sqrt(2)*I
    assert csch(pi*I*Rational(-3, 4)) == sqrt(2)*I

    # 断言 csch(pi*I/6) 等于 -2*I
    assert csch(pi*I/6) == -2*I
    # 断言 csch(-pi*I/6) 等于 2*I
    assert csch(-pi*I/6) == 2*I
    # 断言 csch(pi*I*Rational(7, 6)) 等于 2*I
    assert csch(pi*I*Rational(7, 6)) == 2*I
    # 断言 csch(pi*I*Rational(-7, 6)) 等于 -2*I
    assert csch(pi*I*Rational(-7, 6)) == -2*I
    # 断言 csch(pi*I*Rational(-5, 6)) 等于 2*I
    assert csch(pi*I*Rational(-5, 6)) == 2*I

    # 断言 csch(pi*I/105) 等于 -1/sin(pi/105)*I
    assert csch(pi*I/105) == -1/sin(pi/105)*I
    # 断言 csch(-pi*I/105) 等于 1/sin(pi/105)*I
    assert csch(-pi*I/105) == 1/sin(pi/105)*I

    # 断言 csch(x*I) 等于 -1/sin(x)*I
    assert csch(x*I) == -1/sin(x)*I

    # 断言 csch(k*pi*I) 是无穷大 (zoo)
    assert csch(k*pi*I) is zoo
    # 断言 csch(17*k*pi*I) 是无穷大 (zoo)
    assert csch(17*k*pi*I) is zoo

    # 断言 csch(k*pi*I/2) 等于 -1/sin(k*pi/2)*I
    assert csch(k*pi*I/2) == -1/sin(k*pi/2)*I

    # 断言 csch(n) 的虚部为真 (is_real is True)
    assert csch(n).is_real is True

    # 断言展开三角函数 csch(x + y) 等于 1/(sinh(x)*cosh(y) + cosh(x)*sinh(y))
    assert expand_trig(csch(x + y)) == 1/(sinh(x)*cosh(y) + cosh(x)*sinh(y))
# 定义一个测试函数，用于测试 csch 函数的级数展开是否正确
def test_csch_series():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言 csch(x) 在 x=0 处展开到 x**10 阶的级数，与给定的级数表达式相等
    assert csch(x).series(x, 0, 10) == \
       1/ x - x/6 + 7*x**3/360 - 31*x**5/15120 + 127*x**7/604800 \
          - 73*x**9/3421440


# 定义一个测试函数，用于测试 csch 函数的二阶导数是否会抛出 ArgumentIndexError 异常
def test_csch_fdiff():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言在对 csch(x) 进行二阶导数操作时会抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: csch(x).fdiff(2))


# 定义一个测试函数，用于测试 sech 函数的多个性质和边界条件
def test_sech():
    # 定义符号变量 x, y
    x, y = symbols('x, y')

    # 定义 k 和 n 为整数和正数的符号变量
    k = Symbol('k', integer=True)
    n = Symbol('n', positive=True)

    # 一系列断言，测试 sech 函数在特定输入下的输出值是否符合预期
    assert sech(nan) is nan
    assert sech(zoo) is nan

    assert sech(oo) == 0
    assert sech(-oo) == 0

    assert sech(0) == 1

    assert sech(-1) == sech(1)
    assert sech(-x) == sech(x)

    assert sech(pi*I) == sec(pi)

    assert sech(-pi*I) == sec(pi)
    assert sech(-2**1024 * E) == sech(2**1024 * E)

    assert sech(pi*I/2) is zoo
    assert sech(-pi*I/2) is zoo
    assert sech((-3*10**73 + 1)*pi*I/2) is zoo
    assert sech((7*10**103 + 1)*pi*I/2) is zoo

    assert sech(pi*I) == -1
    assert sech(-pi*I) == -1
    assert sech(5*pi*I) == -1
    assert sech(8*pi*I) == 1

    assert sech(pi*I/3) == 2
    assert sech(pi*I*Rational(-2, 3)) == -2

    assert sech(pi*I/4) == sqrt(2)
    assert sech(-pi*I/4) == sqrt(2)
    assert sech(pi*I*Rational(5, 4)) == -sqrt(2)
    assert sech(pi*I*Rational(-5, 4)) == -sqrt(2)

    assert sech(pi*I/6) == 2/sqrt(3)
    assert sech(-pi*I/6) == 2/sqrt(3)
    assert sech(pi*I*Rational(7, 6)) == -2/sqrt(3)
    assert sech(pi*I*Rational(-5, 6)) == -2/sqrt(3)

    assert sech(pi*I/105) == 1/cos(pi/105)
    assert sech(-pi*I/105) == 1/cos(pi/105)

    assert sech(x*I) == 1/cos(x)

    assert sech(k*pi*I) == 1/cos(k*pi)
    assert sech(17*k*pi*I) == 1/cos(17*k*pi)

    assert sech(n).is_real is True

    assert expand_trig(sech(x + y)) == 1/(cosh(x)*cosh(y) + sinh(x)*sinh(y))


# 定义一个测试函数，用于测试 sech 函数的级数展开是否正确
def test_sech_series():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言 sech(x) 在 x=0 处展开到 x**10 阶的级数，与给定的级数表达式相等
    assert sech(x).series(x, 0, 10) == \
        1 - x**2/2 + 5*x**4/24 - 61*x**6/720 + 277*x**8/8064


# 定义一个测试函数，用于测试 sech 函数的二阶导数是否会抛出 ArgumentIndexError 异常
def test_sech_fdiff():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言在对 sech(x) 进行二阶导数操作时会抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: sech(x).fdiff(2))


# 定义一个测试函数，用于测试 asinh 函数的多个性质和边界条件
def test_asinh():
    # 定义符号变量 x, y
    x, y = symbols('x,y')
    # 一系列断言，测试 asinh 函数在特定输入下的输出值是否符合预期
    assert unchanged(asinh, x)
    assert asinh(-x) == -asinh(x)

    assert asinh(nan) is nan
    assert asinh(0) == 0
    assert asinh(+1) == log(sqrt(2) + 1)

    assert asinh(-1) == log(sqrt(2) - 1)
    assert asinh(I) == pi*I/2
    assert asinh(-I) == -pi*I/2
    assert asinh(I/2) == pi*I/6
    assert asinh(-I/2) == -pi*I/6

    assert asinh(oo) is oo
    assert asinh(-oo) is -oo

    assert asinh(I*oo) is oo
    assert asinh(-I *oo) is -oo

    assert asinh(zoo) is zoo

    assert asinh(I *(sqrt(3) - 1)/(2**Rational(3, 2))) == pi*I/12
    assert asinh(-I *(sqrt(3) - 1)/(2**Rational(3, 2))) == -pi*I/12

    assert asinh(I*(sqrt(5) - 1)/4) == pi*I/10
    assert asinh(-I*(sqrt(5) - 1)/4) == -pi*I/10

    assert asinh(I*(sqrt(5) + 1)/4) == pi*I*Rational(3, 10)
    assert asinh(-I*(sqrt(5) + 1)/4) == pi*I*Rational(-3, 10)
    # 断言反双曲正弦函数应为实数
    assert asinh(S(2)).is_real is True
    # 断言反双曲正弦函数应为有限数
    assert asinh(S(2)).is_finite is True
    # 断言反双曲正弦函数应为实数
    assert asinh(S(-2)).is_real is True
    # 断言反双曲正弦函数应为扩展实数
    assert asinh(S(oo)).is_extended_real is True
    # 断言反双曲正弦函数不应为实数
    assert asinh(-S(oo)).is_real is False
    # 断言反双曲正弦函数与负无穷的差应为负无穷
    assert (asinh(2) - oo) == -oo
    # 断言反双曲正弦函数对实数符号应为实数
    assert asinh(symbols('y', real=True)).is_real is True

    # 对称性检查
    assert asinh(Rational(-1, 2)) == -asinh(S.Half)

    # 反函数组合检查
    assert unchanged(asinh, sinh(Symbol('v1')))

    # 断言反双曲正弦函数应用于双曲正弦函数结果为零
    assert asinh(sinh(0, evaluate=False)) == 0
    assert asinh(sinh(-3, evaluate=False)) == -3
    assert asinh(sinh(2, evaluate=False)) == 2
    assert asinh(sinh(I, evaluate=False)) == I
    assert asinh(sinh(-I, evaluate=False)) == -I
    assert asinh(sinh(5*I, evaluate=False)) == -2*I*pi + 5*I
    assert asinh(sinh(15 + 11*I)) == 15 - 4*I*pi + 11*I
    assert asinh(sinh(-73 + 97*I)) == 73 - 97*I + 31*I*pi
    assert asinh(sinh(-7 - 23*I)) == 7 - 7*I*pi + 23*I
    assert asinh(sinh(13 - 3*I)) == -13 - I*pi + 3*I

    # 符号为正的符号应用于反双曲正弦函数不应为零
    p = Symbol('p', positive=True)
    assert asinh(p).is_zero is False
    # 断言反双曲正弦函数应用于双曲正弦函数结果为零（不进行评估）
    assert asinh(sinh(0, evaluate=False), evaluate=False).is_zero is True
def test_asinh_rewrite():
    x = Symbol('x')
    # 测试 asinh 函数使用 log 重写的情况
    assert asinh(x).rewrite(log) == log(x + sqrt(x**2 + 1))
    # 测试 asinh 函数使用 atanh 重写的情况
    assert asinh(x).rewrite(atanh) == atanh(x/sqrt(1 + x**2))
    # 测试 asinh 函数使用 asin 重写的情况
    assert asinh(x).rewrite(asin) == -I*asin(I*x, evaluate=False)
    # 测试带复数因子的 asinh 函数使用 asin 重写的情况
    assert asinh(x*(1 + I)).rewrite(asin) == -I*asin(I*x*(1+I))
    # 测试 asinh 函数使用 acos 重写的情况
    assert asinh(x).rewrite(acos) == I*acos(I*x, evaluate=False) - I*pi/2


def test_asinh_leading_term():
    x = Symbol('x')
    # 测试 asinh 函数的主导项，涉及分支点的情况
    assert asinh(x).as_leading_term(x, cdir=1) == x
    assert asinh(x + I).as_leading_term(x, cdir=1) == I*pi/2
    assert asinh(x - I).as_leading_term(x, cdir=1) == -I*pi/2
    assert asinh(1/x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert asinh(1/x).as_leading_term(x, cdir=-1) == log(x) - log(2) - I*pi
    # 测试 asinh 函数的主导项，涉及分支切割线的情况
    assert asinh(x + 2*I).as_leading_term(x, cdir=1) == I*asin(2)
    assert asinh(x + 2*I).as_leading_term(x, cdir=-1) == -I*asin(2) + I*pi
    assert asinh(x - 2*I).as_leading_term(x, cdir=1) == -I*pi + I*asin(2)
    assert asinh(x - 2*I).as_leading_term(x, cdir=-1) == -I*asin(2)
    # 测试 asinh 函数的主导项，涉及 re(ndir) == 0 的情况
    assert asinh(2*I + I*x - x**2).as_leading_term(x, cdir=1) == log(2 - sqrt(3)) + I*pi/2
    assert asinh(2*I + I*x - x**2).as_leading_term(x, cdir=-1) == log(2 - sqrt(3)) + I*pi/2


def test_asinh_series():
    x = Symbol('x')
    # 测试 asinh 函数的级数展开
    assert asinh(x).series(x, 0, 8) == \
        x - x**3/6 + 3*x**5/40 - 5*x**7/112 + O(x**8)
    t5 = asinh(x).taylor_term(5, x)
    assert t5 == 3*x**5/40
    assert asinh(x).taylor_term(7, x, t5, 0) == -5*x**7/112


def test_asinh_nseries():
    x = Symbol('x')
    # 测试 asinh 函数的数值级数展开，涉及分支点的情况
    assert asinh(x + I)._eval_nseries(x, 4, None) == I*pi/2 - \
    sqrt(2)*sqrt(I)*I*sqrt(x) + sqrt(2)*sqrt(I)*x**(S(3)/2)/12 + 3*sqrt(2)*sqrt(I)*I*x**(S(5)/2)/160 - \
    5*sqrt(2)*sqrt(I)*x**(S(7)/2)/896 + O(x**4)
    assert asinh(x - I)._eval_nseries(x, 4, None) == -I*pi/2 + \
    sqrt(2)*I*sqrt(x)*sqrt(-I) + sqrt(2)*x**(S(3)/2)*sqrt(-I)/12 - \
    3*sqrt(2)*I*x**(S(5)/2)*sqrt(-I)/160 - 5*sqrt(2)*x**(S(7)/2)*sqrt(-I)/896 + O(x**4)
    # 测试 asinh 函数的数值级数展开，涉及分支切割线的情况
    assert asinh(x + 2*I)._eval_nseries(x, 4, None, cdir=1) == I*asin(2) - \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x + 2*I)._eval_nseries(x, 4, None, cdir=-1) == I*pi - I*asin(2) + \
    sqrt(3)*I*x/3 - sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x - 2*I)._eval_nseries(x, 4, None, cdir=1) == I*asin(2) - I*pi + \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x - 2*I)._eval_nseries(x, 4, None, cdir=-1) == -I*asin(2) - \
    sqrt(3)*I*x/3 - sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    # 测试 asinh 函数的数值级数展开，涉及 re(ndir) == 0 的情况
    assert asinh(2*I + I*x - x**2)._eval_nseries(x, 4, None) == I*pi/2 + log(2 - sqrt(3)) + \
    x*(-3 + 2*sqrt(3))/(-6 + 3*sqrt(3)) + x**2*(12 - 36*I + sqrt(3)*(-7 + 21*I))/(-63 + \
    36*sqrt(3)) + x**3*(-168 + sqrt(3)*(97 - 388*I) + 672*I)/(-1746 + 1008*sqrt(3)) + O(x**4)



    # 这是一个数学表达式，可能是某种数学模型或算法的一部分
    # 在这里，x 是一个变量，sqrt(3) 表示 3 的平方根，I 是虚数单位
    # 表达式结合了常数、变量 x 的各次幂和复数运算，计算其值可能涉及复数域和数学分析
    # O(x**4) 表示高阶无穷小项，表明在此表达式中 x 的 4 次及更高次幂的项被忽略
# 定义一个测试函数，用于测试 asinh 函数的二阶导数
def test_asinh_fdiff():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言调用 asinh(x) 的二阶导数会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: asinh(x).fdiff(2))


# 定义一个测试函数，用于测试 acosh 函数
def test_acosh():
    # 创建符号变量 x
    x = Symbol('x')

    # 断言 acosh(-x) 的结果不变
    assert unchanged(acosh, -x)

    # 在特定点的断言
    assert acosh(1) == 0
    assert acosh(-1) == pi*I
    assert acosh(0) == I*pi/2
    assert acosh(S.Half) == I*pi/3
    assert acosh(Rational(-1, 2)) == pi*I*Rational(2, 3)
    assert acosh(nan) is nan

    # 在无穷点的断言
    assert acosh(oo) is oo
    assert acosh(-oo) is oo

    assert acosh(I*oo) == oo + I*pi/2
    assert acosh(-I*oo) == oo - I*pi/2

    assert acosh(zoo) is zoo

    assert acosh(I) == log(I*(1 + sqrt(2)))
    assert acosh(-I) == log(-I*(1 + sqrt(2)))
    assert acosh((sqrt(3) - 1)/(2*sqrt(2))) == pi*I*Rational(5, 12)
    assert acosh(-(sqrt(3) - 1)/(2*sqrt(2))) == pi*I*Rational(7, 12)
    assert acosh(sqrt(2)/2) == I*pi/4
    assert acosh(-sqrt(2)/2) == I*pi*Rational(3, 4)
    assert acosh(sqrt(3)/2) == I*pi/6
    assert acosh(-sqrt(3)/2) == I*pi*Rational(5, 6)
    assert acosh(sqrt(2 + sqrt(2))/2) == I*pi/8
    assert acosh(-sqrt(2 + sqrt(2))/2) == I*pi*Rational(7, 8)
    assert acosh(sqrt(2 - sqrt(2))/2) == I*pi*Rational(3, 8)
    assert acosh(-sqrt(2 - sqrt(2))/2) == I*pi*Rational(5, 8)
    assert acosh((1 + sqrt(3))/(2*sqrt(2))) == I*pi/12
    assert acosh(-(1 + sqrt(3))/(2*sqrt(2))) == I*pi*Rational(11, 12)
    assert acosh((sqrt(5) + 1)/4) == I*pi/5
    assert acosh(-(sqrt(5) + 1)/4) == I*pi*Rational(4, 5)

    assert str(acosh(5*I).n(6)) == '2.31244 + 1.5708*I'
    assert str(acosh(-5*I).n(6)) == '2.31244 - 1.5708*I'

    # 对 acosh 函数的逆组合的断言
    assert unchanged(acosh, Symbol('v1'))

    assert acosh(cosh(-3, evaluate=False)) == 3
    assert acosh(cosh(3, evaluate=False)) == 3
    assert acosh(cosh(0, evaluate=False)) == 0
    assert acosh(cosh(I, evaluate=False)) == I
    assert acosh(cosh(-I, evaluate=False)) == I
    assert acosh(cosh(7*I, evaluate=False)) == -2*I*pi + 7*I
    assert acosh(cosh(1 + I)) == 1 + I
    assert acosh(cosh(3 - 3*I)) == 3 - 3*I
    assert acosh(cosh(-3 + 2*I)) == 3 - 2*I
    assert acosh(cosh(-5 - 17*I)) == 5 - 6*I*pi + 17*I
    assert acosh(cosh(-21 + 11*I)) == 21 - 11*I + 4*I*pi
    assert acosh(cosh(cosh(1) + I)) == cosh(1) + I
    assert acosh(1, evaluate=False).is_zero is True

    # 实数性质的断言
    assert acosh(S(2)).is_real is True
    assert acosh(S(2)).is_extended_real is True
    assert acosh(oo).is_extended_real is True
    assert acosh(S(2)).is_finite is True
    assert acosh(S(1) / 5).is_real is False
    assert (acosh(2) - oo) == -oo
    assert acosh(symbols('y', real=True)).is_real is None


# 定义一个测试函数，用于测试 acosh 函数的重写
def test_acosh_rewrite():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言 acosh(x) 使用对数重写的结果
    assert acosh(x).rewrite(log) == log(x + sqrt(x - 1)*sqrt(x + 1))
    # 断言 acosh(x) 使用 arcsin 重写的结果
    assert acosh(x).rewrite(asin) == sqrt(x - 1)*(-asin(x) + pi/2)/sqrt(1 - x)
    # 断言 acosh(x) 使用 asinh 重写的结果
    assert acosh(x).rewrite(asinh) == sqrt(x - 1)*(I*asinh(I*x, evaluate=False) + pi/2)/sqrt(1 - x)
    # 断言：使用反双曲余弦函数的 atan 替换表达式验证
    assert acosh(x).rewrite(atanh) == \
        (sqrt(x - 1)*sqrt(x + 1)*atanh(sqrt(x**2 - 1)/x)/sqrt(x**2 - 1) +
         pi*sqrt(x - 1)*(-x*sqrt(x**(-2)) + 1)/(2*sqrt(1 - x)))
    
    # 创建一个正数符号变量 x
    x = Symbol('x', positive=True)
    
    # 再次进行断言：使用反双曲余弦函数的 atan 替换表达式验证
    assert acosh(x).rewrite(atanh) == \
        sqrt(x - 1)*sqrt(x + 1)*atanh(sqrt(x**2 - 1)/x)/sqrt(x**2 - 1)
def test_acosh_leading_term():
    x = Symbol('x')
    # Tests concerning branch points
    assert acosh(x).as_leading_term(x) == I*pi/2
    assert acosh(x + 1).as_leading_term(x) == sqrt(2)*sqrt(x)
    assert acosh(x - 1).as_leading_term(x) == I*pi
    assert acosh(1/x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert acosh(1/x).as_leading_term(x, cdir=-1) == -log(x) + log(2) + 2*I*pi
    # Tests concerning points lying on branch cuts
    assert acosh(I*x - 2).as_leading_term(x, cdir=1) == acosh(-2)
    assert acosh(-I*x - 2).as_leading_term(x, cdir=1) == -2*I*pi + acosh(-2)
    assert acosh(x**2 - I*x + S(1)/3).as_leading_term(x, cdir=1) == -acosh(S(1)/3)
    assert acosh(x**2 - I*x + S(1)/3).as_leading_term(x, cdir=-1) == acosh(S(1)/3)
    assert acosh(1/(I*x - 3)).as_leading_term(x, cdir=1) == -acosh(-S(1)/3)
    assert acosh(1/(I*x - 3)).as_leading_term(x, cdir=-1) == acosh(-S(1)/3)
    # Tests concerning im(ndir) == 0
    assert acosh(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == log(sqrt(3) + 2) - I*pi
    assert acosh(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == log(sqrt(3) + 2) - I*pi


def test_acosh_series():
    x = Symbol('x')
    assert acosh(x).series(x, 0, 8) == \
        -I*x + pi*I/2 - I*x**3/6 - 3*I*x**5/40 - 5*I*x**7/112 + O(x**8)
    t5 = acosh(x).taylor_term(5, x)
    assert t5 == - 3*I*x**5/40
    assert acosh(x).taylor_term(7, x, t5, 0) == - 5*I*x**7/112


def test_acosh_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert acosh(x + 1)._eval_nseries(x, 4, None) == sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/12 + 3*sqrt(2)*x**(S(5)/2)/160 - 5*sqrt(2)*x**(S(7)/2)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert acosh(x - 1)._eval_nseries(x, 4, None) == I*pi - \
    sqrt(2)*I*sqrt(x) - sqrt(2)*I*x**(S(3)/2)/12 - 3*sqrt(2)*I*x**(S(5)/2)/160 - \
    5*sqrt(2)*I*x**(S(7)/2)/896 + O(x**4)
    assert acosh(I*x - 2)._eval_nseries(x, 4, None, cdir=1) == acosh(-2) - \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acosh(-I*x - 2)._eval_nseries(x, 4, None, cdir=1) == acosh(-2) - \
    2*I*pi + sqrt(3)*I*x/3 + sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert acosh(1/(I*x - 3))._eval_nseries(x, 4, None, cdir=1) == -acosh(-S(1)/3) + \
    sqrt(2)*x/12 + 17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert acosh(1/(I*x - 3))._eval_nseries(x, 4, None, cdir=-1) == acosh(-S(1)/3) - \
    sqrt(2)*x/12 - 17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert acosh(-I*x**2 + x - 2)._eval_nseries(x, 4, None) == -I*pi + log(sqrt(3) + 2) + \
    x*(-2*sqrt(3) - 3)/(3*sqrt(3) + 6) + x**2*(-12 + 36*I + sqrt(3)*(-7 + 21*I))/(36*sqrt(3) + \
    63) + x**3*(-168 + 672*I + sqrt(3)*(-97 + 388*I))/(1008*sqrt(3) + 1746) + O(x**4)


def test_acosh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: acosh(x).fdiff(2))


def test_asech():
    x = Symbol('x')
    # 使用 assert 语句来测试 unchanged 函数对 asech 函数在 -x 处的返回值是否不变
    assert unchanged(asech, -x)

    # 测试 asech 函数在固定点上的返回值是否正确
    assert asech(1) == 0
    assert asech(-1) == pi*I
    assert asech(0) is oo
    assert asech(2) == I*pi/3
    assert asech(-2) == 2*I*pi / 3
    assert asech(nan) is nan

    # 测试 asech 函数在无穷远点处的返回值
    assert asech(oo) == I*pi/2
    assert asech(-oo) == I*pi/2
    assert asech(zoo) == I*AccumBounds(-pi/2, pi/2)

    # 测试 asech 函数在复数域的返回值
    assert asech(I) == log(1 + sqrt(2)) - I*pi/2
    assert asech(-I) == log(1 + sqrt(2)) + I*pi/2
    assert asech(sqrt(2) - sqrt(6)) == 11*I*pi / 12
    assert asech(sqrt(2 - 2/sqrt(5))) == I*pi / 10
    assert asech(-sqrt(2 - 2/sqrt(5))) == 9*I*pi / 10
    assert asech(2 / sqrt(2 + sqrt(2))) == I*pi / 8
    assert asech(-2 / sqrt(2 + sqrt(2))) == 7*I*pi / 8
    assert asech(sqrt(5) - 1) == I*pi / 5
    assert asech(1 - sqrt(5)) == 4*I*pi / 5
    assert asech(-sqrt(2*(2 + sqrt(2)))) == 5*I*pi / 8

    # 测试 asech 函数的性质
    assert asech(sqrt(2)) == acosh(1/sqrt(2))
    assert asech(2/sqrt(3)) == acosh(sqrt(3)/2)
    assert asech(2/sqrt(2 + sqrt(2))) == acosh(sqrt(2 + sqrt(2))/2)
    assert asech(2) == acosh(S.Half)

    # 测试 asech 函数的实部性质
    assert asech(S(2)).is_real is False
    assert asech(-S(1) / 3).is_real is False
    assert asech(S(2) / 3).is_finite is True
    assert asech(S(0)).is_real is False
    assert asech(S(0)).is_extended_real is True
    assert asech(symbols('y', real=True)).is_real is None

    # 测试 asech 函数与 acos 函数的关系
    # asech(x) == I*acos(1/x)
    # （注意：确切的公式是 asech(x) == +/- I*acos(1/x)）
    assert asech(-sqrt(2)) == I*acos(-1/sqrt(2))
    assert asech(-2/sqrt(3)) == I*acos(-sqrt(3)/2)
    assert asech(-S(2)) == I*acos(Rational(-1, 2))
    assert asech(-2/sqrt(2)) == I*acos(-sqrt(2)/2)

    # 测试 sech(asech(x)) / x == 1 的数值验证
    assert expand_mul(sech(asech(sqrt(6) - sqrt(2))) / (sqrt(6) - sqrt(2))) == 1
    assert expand_mul(sech(asech(sqrt(6) + sqrt(2))) / (sqrt(6) + sqrt(2))) == 1
    assert (sech(asech(sqrt(2 + 2/sqrt(5)))) / (sqrt(2 + 2/sqrt(5)))).simplify() == 1
    assert (sech(asech(-sqrt(2 + 2/sqrt(5)))) / (-sqrt(2 + 2/sqrt(5)))).simplify() == 1
    assert (sech(asech(sqrt(2*(2 + sqrt(2))))) / (sqrt(2*(2 + sqrt(2))))).simplify() == 1
    assert expand_mul(sech(asech(1 + sqrt(5))) / (1 + sqrt(5))) == 1
    assert expand_mul(sech(asech(-1 - sqrt(5))) / (-1 - sqrt(5))) == 1
    assert expand_mul(sech(asech(-sqrt(6) - sqrt(2))) / (-sqrt(6) - sqrt(2))) == 1

    # 测试 asech 函数的数值评估
    assert str(asech(5*I).n(6)) == '0.19869 - 1.5708*I'
    assert str(asech(-5*I).n(6)) == '0.19869 + 1.5708*I'
def test_asech_leading_term():
    x = Symbol('x')
    # Tests concerning branch points
    assert asech(x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert asech(x).as_leading_term(x, cdir=-1) == -log(x) + log(2) + 2*I*pi
    assert asech(x + 1).as_leading_term(x, cdir=1) == sqrt(2)*I*sqrt(x)
    assert asech(1/x).as_leading_term(x, cdir=1) == I*pi/2
    # Tests concerning points lying on branch cuts
    assert asech(x - 1).as_leading_term(x, cdir=1) == I*pi
    assert asech(I*x + 3).as_leading_term(x, cdir=1) == -asech(3)
    assert asech(-I*x + 3).as_leading_term(x, cdir=1) == asech(3)
    assert asech(I*x - 3).as_leading_term(x, cdir=1) == -asech(-3)
    assert asech(-I*x - 3).as_leading_term(x, cdir=1) == asech(-3)
    assert asech(I*x - S(1)/3).as_leading_term(x, cdir=1) == -2*I*pi + asech(-S(1)/3)
    assert asech(I*x - S(1)/3).as_leading_term(x, cdir=-1) == asech(-S(1)/3)
    # Tests concerning im(ndir) == 0
    assert asech(-I*x**2 + x - 3).as_leading_term(x, cdir=1) == log(-S(1)/3 + 2*sqrt(2)*I/3)
    assert asech(-I*x**2 + x - 3).as_leading_term(x, cdir=-1) == log(-S(1)/3 + 2*sqrt(2)*I/3)


def test_asech_series():
    x = Symbol('x')
    assert asech(x).series(x, 0, 9, cdir=1) == log(2) - log(x) - x**2/4 - 3*x**4/32 \
    - 5*x**6/96 - 35*x**8/1024 + O(x**9)
    assert asech(x).series(x, 0, 9, cdir=-1) == I*pi + log(2) - log(-x) - x**2/4 - \
    3*x**4/32 - 5*x**6/96 - 35*x**8/1024 + O(x**9)
    t6 = asech(x).taylor_term(6, x)
    assert t6 == -5*x**6/96
    assert asech(x).taylor_term(8, x, t6, 0) == -35*x**8/1024


def test_asech_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert asech(x + 1)._eval_nseries(x, 4, None) == sqrt(2)*sqrt(-x) + 5*sqrt(2)*(-x)**(S(3)/2)/12 + \
    43*sqrt(2)*(-x)**(S(5)/2)/160 + 177*sqrt(2)*(-x)**(S(7)/2)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert asech(x - 1)._eval_nseries(x, 4, None) == I*pi + sqrt(2)*sqrt(x) + \
    5*sqrt(2)*x**(S(3)/2)/12 + 43*sqrt(2)*x**(S(5)/2)/160 + 177*sqrt(2)*x**(S(7)/2)/896 + O(x**4)
    assert asech(I*x + 3)._eval_nseries(x, 4, None) == -asech(3) + sqrt(2)*x/12 - \
    17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(-I*x + 3)._eval_nseries(x, 4, None) == asech(3) + sqrt(2)*x/12 + \
    17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(I*x - 3)._eval_nseries(x, 4, None) == -asech(-3) - sqrt(2)*x/12 - \
    17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(-I*x - 3)._eval_nseries(x, 4, None) == asech(-3) - sqrt(2)*x/12 + \
    17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert asech(-I*x**2 + x - 2)._eval_nseries(x, 3, None) == 2*I*pi/3 + \
    x*(-sqrt(3) + 3*I)/(6*sqrt(3) + 6*I) + x**2*(36 + sqrt(3)*(7 - 12*I) + 21*I)/(72*sqrt(3) - \
    72*I)


def test_asech_rewrite():
    x = Symbol('x')
    assert asech(x).rewrite(log) == log(1/x + sqrt(1/x - 1) * sqrt(1/x + 1))
    # 断言：使用反双曲余割函数 asech(x) 重写为双曲反余弦函数 acosh(1/x)
    assert asech(x).rewrite(acosh) == acosh(1/x)
    
    # 断言：使用反双曲余割函数 asech(x) 重写为双曲反正弦函数 asinh
    # 注意：这里使用了复数和虚数单位 I
    assert asech(x).rewrite(asinh) == sqrt(-1 + 1/x)*(I*asinh(I/x, evaluate=False) + pi/2)/sqrt(1 - 1/x)
    
    # 断言：使用反双曲余割函数 asech(x) 重写为双曲反正切函数 atanh
    assert asech(x).rewrite(atanh) == \
        sqrt(x + 1)*sqrt(1/(x + 1))*atanh(sqrt(1 - x**2)) + I*pi*(-sqrt(x)*sqrt(1/x) + 1 - I*sqrt(x**2)/(2*sqrt(-x**2)) - I*sqrt(-x)/(2*sqrt(x)))
def test_asech_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: asech(x).fdiff(2))


# 测试 asech(x) 的二阶导数是否会引发 ArgumentIndexError 异常
def test_asech_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: asech(x).fdiff(2))



def test_acsch():
    x = Symbol('x')

    assert unchanged(acsch, x)
    assert acsch(-x) == -acsch(x)

    # values at fixed points
    assert acsch(1) == log(1 + sqrt(2))
    assert acsch(-1) == - log(1 + sqrt(2))
    assert acsch(0) is zoo
    assert acsch(2) == log((1+sqrt(5))/2)
    assert acsch(-2) == - log((1+sqrt(5))/2)

    assert acsch(I) == - I*pi/2
    assert acsch(-I) == I*pi/2
    assert acsch(-I*(sqrt(6) + sqrt(2))) == I*pi / 12
    assert acsch(I*(sqrt(2) + sqrt(6))) == -I*pi / 12
    assert acsch(-I*(1 + sqrt(5))) == I*pi / 10
    assert acsch(I*(1 + sqrt(5))) == -I*pi / 10
    assert acsch(-I*2 / sqrt(2 - sqrt(2))) == I*pi / 8
    assert acsch(I*2 / sqrt(2 - sqrt(2))) == -I*pi / 8
    assert acsch(-I*2) == I*pi / 6
    assert acsch(I*2) == -I*pi / 6
    assert acsch(-I*sqrt(2 + 2/sqrt(5))) == I*pi / 5
    assert acsch(I*sqrt(2 + 2/sqrt(5))) == -I*pi / 5
    assert acsch(-I*sqrt(2)) == I*pi / 4
    assert acsch(I*sqrt(2)) == -I*pi / 4
    assert acsch(-I*(sqrt(5)-1)) == 3*I*pi / 10
    assert acsch(I*(sqrt(5)-1)) == -3*I*pi / 10
    assert acsch(-I*2 / sqrt(3)) == I*pi / 3
    assert acsch(I*2 / sqrt(3)) == -I*pi / 3
    assert acsch(-I*2 / sqrt(2 + sqrt(2))) == 3*I*pi / 8
    assert acsch(I*2 / sqrt(2 + sqrt(2))) == -3*I*pi / 8
    assert acsch(-I*sqrt(2 - 2/sqrt(5))) == 2*I*pi / 5
    assert acsch(I*sqrt(2 - 2/sqrt(5))) == -2*I*pi / 5
    assert acsch(-I*(sqrt(6) - sqrt(2))) == 5*I*pi / 12
    assert acsch(I*(sqrt(6) - sqrt(2))) == -5*I*pi / 12
    assert acsch(nan) is nan

    # properties
    # acsch(x) == asinh(1/x)
    assert acsch(-I*sqrt(2)) == asinh(I/sqrt(2))
    assert acsch(-I*2 / sqrt(3)) == asinh(I*sqrt(3) / 2)

    # reality
    assert acsch(S(2)).is_real is True
    assert acsch(S(2)).is_finite is True
    assert acsch(S(-2)).is_real is True
    assert acsch(S(oo)).is_extended_real is True
    assert acsch(-S(oo)).is_real is True
    assert (acsch(2) - oo) == -oo
    assert acsch(symbols('y', extended_real=True)).is_extended_real is True

    # acsch(x) == -I*asin(I/x)
    assert acsch(-I*sqrt(2)) == -I*asin(-1/sqrt(2))
    assert acsch(-I*2 / sqrt(3)) == -I*asin(-sqrt(3)/2)

    # csch(acsch(x)) / x == 1
    assert expand_mul(csch(acsch(-I*(sqrt(6) + sqrt(2)))) / (-I*(sqrt(6) + sqrt(2)))) == 1
    assert expand_mul(csch(acsch(I*(1 + sqrt(5)))) / (I*(1 + sqrt(5)))) == 1
    assert (csch(acsch(I*sqrt(2 - 2/sqrt(5)))) / (I*sqrt(2 - 2/sqrt(5)))).simplify() == 1
    assert (csch(acsch(-I*sqrt(2 - 2/sqrt(5)))) / (-I*sqrt(2 - 2/sqrt(5)))).simplify() == 1

    # numerical evaluation
    assert str(acsch(5*I+1).n(6)) == '0.0391819 - 0.193363*I'
    assert str(acsch(-5*I+1).n(6)) == '0.0391819 + 0.193363*I'


# 测试 acsch 函数对无穷大和复数 zoo 的处理
def test_acsch_infinities():
    assert acsch(oo) == 0
    assert acsch(-oo) == 0
    assert acsch(zoo) == 0


# 测试 acsch(1/x) 的主导项是否等于 x
def test_acsch_leading_term():
    x = Symbol('x')
    assert acsch(1/x).as_leading_term(x) == x
``` 
    # Tests concerning branch points
    # 断言测试反双曲正弦函数在分支点的行为
    assert acsch(x + I).as_leading_term(x) == -I*pi/2
    # 断言测试反双曲正弦函数在分支点的行为
    assert acsch(x - I).as_leading_term(x) == I*pi/2
    
    # Tests concerning points lying on branch cuts
    # 断言测试反双曲正弦函数在分支切割线上点的行为
    assert acsch(x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    # 断言测试反双曲正弦函数在分支切割线上点的行为
    assert acsch(x).as_leading_term(x, cdir=-1) == log(x) - log(2) - I*pi
    # 断言测试反双曲正弦函数在分支切割线上点的行为
    assert acsch(x + I/2).as_leading_term(x, cdir=1) == -I*pi - acsch(I/2)
    # 断言测试反双曲正弦函数在分支切割线上点的行为
    assert acsch(x + I/2).as_leading_term(x, cdir=-1) == acsch(I/2)
    # 断言测试反双曲正弦函数在分支切割线上点的行为
    assert acsch(x - I/2).as_leading_term(x, cdir=1) == -acsch(I/2)
    # 断言测试反双曲正弦函数在分支切割线上点的行为
    assert acsch(x - I/2).as_leading_term(x, cdir=-1) == acsch(I/2) + I*pi
    
    # Tests concerning re(ndir) == 0
    # 断言测试反双曲正弦函数在实部(ndir)等于0的情况下的行为
    assert acsch(I/2 + I*x - x**2).as_leading_term(x, cdir=1) == log(2 - sqrt(3)) - I*pi/2
    # 断言测试反双曲正弦函数在实部(ndir)等于0的情况下的行为
    assert acsch(I/2 + I*x - x**2).as_leading_term(x, cdir=-1) == log(2 - sqrt(3)) - I*pi/2
# 定义测试函数 test_acsch_series，用于测试 acsch 函数的级数展开
def test_acsch_series():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言：对 acsch(x) 在 x=0 处展开到第 9 阶的级数，与预期值比较
    assert acsch(x).series(x, 0, 9) == log(2) - log(x) + x**2/4 - 3*x**4/32 \
    + 5*x**6/96 - 35*x**8/1024 + O(x**9)
    # 计算 acsch(x) 的 taylor_term 的第 4 阶展开项，并进行断言
    t4 = acsch(x).taylor_term(4, x)
    assert t4 == -3*x**4/32
    # 断言：对 acsch(x) 在 x=0 处展开到第 6 阶的级数，与预期值比较
    assert acsch(x).taylor_term(6, x, t4, 0) == 5*x**6/96


# 定义测试函数 test_acsch_nseries，用于测试 acsch 函数的数值级数展开
def test_acsch_nseries():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言：对 acsch(x + I) 在 x=0 处展开到第 4 阶的数值级数，与预期值比较
    assert acsch(x + I)._eval_nseries(x, 4, None) == -I*pi/2 + \
    sqrt(2)*I*sqrt(x)*sqrt(-I) - 5*x**(S(3)/2)*(1 - I)/12 - \
    43*sqrt(2)*I*x**(S(5)/2)*sqrt(-I)/160 + 177*x**(S(7)/2)*(1 - I)/896 + O(x**4)
    # 断言：对 acsch(x - I) 在 x=0 处展开到第 4 阶的数值级数，与预期值比较
    assert acsch(x - I)._eval_nseries(x, 4, None) == I*pi/2 - \
    sqrt(2)*sqrt(I)*I*sqrt(x) - 5*x**(S(3)/2)*(1 + I)/12 + \
    43*sqrt(2)*sqrt(I)*I*x**(S(5)/2)/160 + 177*x**(S(7)/2)*(1 + I)/896 + O(x**4)
    # 断言：对 acsch(x + I/2) 在 x=0 处展开到第 4 阶的数值级数，与预期值比较（考虑分支切割点）
    assert acsch(x + I/2)._eval_nseries(x, 4, None, cdir=1) == -acsch(I/2) - \
    I*pi + 4*sqrt(3)*I*x/3 - 8*sqrt(3)*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    # 断言：对 acsch(x + I/2) 在 x=0 处展开到第 4 阶的数值级数，与预期值比较（考虑分支切割点）
    assert acsch(x + I/2)._eval_nseries(x, 4, None, cdir=-1) == acsch(I/2) - \
    4*sqrt(3)*I*x/3 + 8*sqrt(3)*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    # 断言：对 acsch(x - I/2) 在 x=0 处展开到第 4 阶的数值级数，与预期值比较（考虑分支切割点）
    assert acsch(x - I/2)._eval_nseries(x, 4, None, cdir=1) == -acsch(I/2) - \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    # 断言：对 acsch(x - I/2) 在 x=0 处展开到第 4 阶的数值级数，与预期值比较（考虑分支切割点）
    assert acsch(x - I/2)._eval_nseries(x, 4, None, cdir=-1) == I*pi + \
    acsch(I/2) + 4*sqrt(3)*I*x/3 + 8*sqrt(3)*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    # 断言：对 acsch(I/2 + I*x - x**2) 在 x=0 处展开到第 4 阶的数值级数，与预期值比较
    assert acsch(I/2 + I*x - x**2)._eval_nseries(x, 4, None) == -I*pi/2 + \
    log(2 - sqrt(3)) + x*(12 - 8*sqrt(3))/(-6 + 3*sqrt(3)) + x**2*(-96 + \
    sqrt(3)*(56 - 84*I) + 144*I)/(-63 + 36*sqrt(3)) + x**3*(2688 - 2688*I + \
    sqrt(3)*(-1552 + 1552*I))/(-873 + 504*sqrt(3)) + O(x**4)


# 定义测试函数 test_acsch_rewrite，用于测试 acsch 函数的重写方法
def test_acsch_rewrite():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言：对 acsch(x) 使用 log 函数重写，与预期值比较
    assert acsch(x).rewrite(log) == log(1/x + sqrt(1/x**2 + 1))
    # 断言：对 acsch(x) 使用 asinh 函数重写，与预期值比较
    assert acsch(x).rewrite(asinh) == asinh(1/x)
    # 断言：对 acsch(x) 使用 atanh 函数重写，与预期值比较
    assert acsch(x).rewrite(atanh) == (sqrt(-x**2)*(-sqrt(-(x**2 + 1)**2)
                                                    *atanh(sqrt(x**2 + 1))/(x**2 + 1)
                                                    + pi/2)/x)


# 定义测试函数 test_acsch_fdiff，用于测试 acsch 函数的偏导数方法
def test_acsch_fdiff():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言：尝试对 acsch(x) 的第二阶导数，预期抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: acsch(x).fdiff(2))


# 定义测试函数 test_atanh，用于测试 atanh 函数的各种情况
def test_atanh():
    # 定义符号变量 x
    x = Symbol('x')

    # 断言：在特定点处计算 atanh 函数值，与预期值比较
    assert atanh(0) == 0
    assert atanh(I) == I*pi/4
    assert atanh(-I) == -I*pi/4
    assert atanh(1) is oo
    assert atanh(-1) is -oo
    assert atanh(nan) is nan

    # 断言：在无穷点处计算 atanh 函数值，与预期值比较
    assert atanh(oo) == -I*pi/2
    assert atanh(-oo) == I*pi/2

    assert atanh(I*oo) == I*pi/2
    assert atanh(-I*oo) == -I*pi/2

    assert atanh(zoo) == I*AccumBounds(-pi/2, pi/2)

    # 断言：验证 atanh 函数的性质
    assert atanh(-x) == -atanh(x)

    # 断言：验证 atanh 函数的实部性质
    assert atanh(S(2)).is_real is False
    assert atanh(S(-1)/5).is_real is True
    assert atanh(symbols('y', extended_real=True)).is_real is None
    assert atanh(S(1)).is_real is False
    # 断言：atanh(S(1)) 的结果是扩展实数
    assert atanh(S(1)).is_extended_real is True
    # 断言：atanh(S(-1)) 的结果不是实数
    assert atanh(S(-1)).is_real is False

    # 特殊值测试
    # 断言：atanh(I/sqrt(3)) 的结果是 I*pi/6
    assert atanh(I/sqrt(3)) == I*pi/6
    # 断言：atanh(-I/sqrt(3)) 的结果是 -I*pi/6
    assert atanh(-I/sqrt(3)) == -I*pi/6
    # 断言：atanh(I*sqrt(3)) 的结果是 I*pi/3
    assert atanh(I*sqrt(3)) == I*pi/3
    # 断言：atanh(-I*sqrt(3)) 的结果是 -I*pi/3
    assert atanh(-I*sqrt(3)) == -I*pi/3
    # 断言：atanh(I*(1 + sqrt(2))) 的结果是 pi*I*Rational(3, 8)
    assert atanh(I*(1 + sqrt(2))) == pi*I*Rational(3, 8)
    # 断言：atanh(I*(sqrt(2) - 1)) 的结果是 pi*I/8
    assert atanh(I*(sqrt(2) - 1)) == pi*I/8
    # 断言：atanh(I*(1 - sqrt(2))) 的结果是 -pi*I/8
    assert atanh(I*(1 - sqrt(2))) == -pi*I/8
    # 断言：atanh(-I*(1 + sqrt(2))) 的结果是 pi*I*Rational(-3, 8)
    assert atanh(-I*(1 + sqrt(2))) == pi*I*Rational(-3, 8)
    # 断言：atanh(I*sqrt(5 + 2*sqrt(5))) 的结果是 I*pi*Rational(2, 5)
    assert atanh(I*sqrt(5 + 2*sqrt(5))) == I*pi*Rational(2, 5)
    # 断言：atanh(-I*sqrt(5 + 2*sqrt(5))) 的结果是 I*pi*Rational(-2, 5)
    assert atanh(-I*sqrt(5 + 2*sqrt(5))) == I*pi*Rational(-2, 5)
    # 断言：atanh(I*(2 - sqrt(3))) 的结果是 pi*I/12
    assert atanh(I*(2 - sqrt(3))) == pi*I/12
    # 断言：atanh(I*(sqrt(3) - 2)) 的结果是 -pi*I/12
    assert atanh(I*(sqrt(3) - 2)) == -pi*I/12
    # 断言：atanh(oo) 的结果是 -I*pi/2
    assert atanh(oo) == -I*pi/2

    # 对称性测试
    # 断言：atanh(Rational(-1, 2)) 的结果是 -atanh(S.Half)
    assert atanh(Rational(-1, 2)) == -atanh(S.Half)

    # 反向组合测试
    # 断言：unchanged(atanh, tanh(Symbol('v1'))) 返回 True
    assert unchanged(atanh, tanh(Symbol('v1')))

    # 边界条件测试
    # 断言：atanh(tanh(-5, evaluate=False)) 的结果是 -5
    assert atanh(tanh(-5, evaluate=False)) == -5
    # 断言：atanh(tanh(0, evaluate=False)) 的结果是 0
    assert atanh(tanh(0, evaluate=False)) == 0
    # 断言：atanh(tanh(7, evaluate=False)) 的结果是 7
    assert atanh(tanh(7, evaluate=False)) == 7
    # 断言：atanh(tanh(I, evaluate=False)) 的结果是 I
    assert atanh(tanh(I, evaluate=False)) == I
    # 断言：atanh(tanh(-I, evaluate=False)) 的结果是 -I
    assert atanh(tanh(-I, evaluate=False)) == -I
    # 断言：atanh(tanh(-11*I, evaluate=False)) 的结果是 -11*I + 4*I*pi
    assert atanh(tanh(-11*I, evaluate=False)) == -11*I + 4*I*pi
    # 断言：atanh(tanh(3 + I)) 的结果是 3 + I
    assert atanh(tanh(3 + I)) == 3 + I
    # 断言：atanh(tanh(4 + 5*I)) 的结果是 4 - 2*I*pi + 5*I
    assert atanh(tanh(4 + 5*I)) == 4 - 2*I*pi + 5*I
    # 断言：atanh(tanh(pi/2)) 的结果是 pi/2
    assert atanh(tanh(pi/2)) == pi/2
    # 断言：atanh(tanh(pi)) 的结果是 pi
    assert atanh(tanh(pi)) == pi
    # 断言：atanh(tanh(-3 + 7*I)) 的结果是 -3 - 2*I*pi + 7*I
    assert atanh(tanh(-3 + 7*I)) == -3 - 2*I*pi + 7*I
    # 断言：atanh(tanh(9 - I*2/3)) 的结果是 9 - I*2/3
    assert atanh(tanh(9 - I*2/3)) == 9 - I*2/3
    # 断言：atanh(tanh(-32 - 123*I)) 的结果是 -32 - 123*I + 39*I*pi
    assert atanh(tanh(-32 - 123*I)) == -32 - 123*I + 39*I*pi
def test_atanh_rewrite():
    x = Symbol('x')
    # 断言，验证 atanh(x) 的对数重写形式
    assert atanh(x).rewrite(log) == (log(1 + x) - log(1 - x)) / 2
    # 断言，验证 atanh(x) 的反双曲正弦重写形式
    assert atanh(x).rewrite(asinh) == \
        pi*x/(2*sqrt(-x**2)) - sqrt(-x)*sqrt(1 - x**2)*sqrt(1/(x**2 - 1))*asinh(sqrt(1/(x**2 - 1)))/sqrt(x)


def test_atanh_leading_term():
    x = Symbol('x')
    # 断言，验证 atanh(x) 的主导项
    assert atanh(x).as_leading_term(x) == x
    # 关于分支点的测试
    assert atanh(x + 1).as_leading_term(x, cdir=1) == -log(x)/2 + log(2)/2 - I*pi/2
    assert atanh(x + 1).as_leading_term(x, cdir=-1) == -log(x)/2 + log(2)/2 + I*pi/2
    assert atanh(x - 1).as_leading_term(x, cdir=1) == log(x)/2 - log(2)/2
    assert atanh(x - 1).as_leading_term(x, cdir=-1) == log(x)/2 - log(2)/2
    assert atanh(1/x).as_leading_term(x, cdir=1) == -I*pi/2
    assert atanh(1/x).as_leading_term(x, cdir=-1) == I*pi/2
    # 关于位于分支切割线上的点的测试
    assert atanh(I*x + 2).as_leading_term(x, cdir=1) == atanh(2) + I*pi
    assert atanh(-I*x + 2).as_leading_term(x, cdir=1) == atanh(2)
    assert atanh(I*x - 2).as_leading_term(x, cdir=1) == -atanh(2)
    assert atanh(-I*x - 2).as_leading_term(x, cdir=1) == -I*pi - atanh(2)
    # 关于 im(ndir) == 0 的测试
    assert atanh(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == -log(3)/2 - I*pi/2
    assert atanh(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == -log(3)/2 - I*pi/2


def test_atanh_series():
    x = Symbol('x')
    # 断言，验证 atanh(x) 的级数展开
    assert atanh(x).series(x, 0, 10) == \
        x + x**3/3 + x**5/5 + x**7/7 + x**9/9 + O(x**10)


def test_atanh_nseries():
    x = Symbol('x')
    # 关于分支点的测试
    assert atanh(x + 1)._eval_nseries(x, 4, None, cdir=1) == -I*pi/2 + log(2)/2 - \
    log(x)/2 + x/4 - x**2/16 + x**3/48 + O(x**4)
    assert atanh(x + 1)._eval_nseries(x, 4, None, cdir=-1) == I*pi/2 + log(2)/2 - \
    log(x)/2 + x/4 - x**2/16 + x**3/48 + O(x**4)
    assert atanh(x - 1)._eval_nseries(x, 4, None, cdir=1) == -log(2)/2 + log(x)/2 + \
    x/4 + x**2/16 + x**3/48 + O(x**4)
    assert atanh(x - 1)._eval_nseries(x, 4, None, cdir=-1) == -log(2)/2 + log(x)/2 + \
    x/4 + x**2/16 + x**3/48 + O(x**4)
    # 关于位于分支切割线上的点的测试
    assert atanh(I*x + 2)._eval_nseries(x, 4, None, cdir=1) == I*pi + atanh(2) - \
    I*x/3 - 2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x + 2)._eval_nseries(x, 4, None, cdir=-1) == atanh(2) - I*x/3 - \
    2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x - 2)._eval_nseries(x, 4, None, cdir=1) == -atanh(2) - I*x/3 + \
    2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x - 2)._eval_nseries(x, 4, None, cdir=-1) == -atanh(2) - I*pi - \
    I*x/3 + 2*x**2/9 + 13*I*x**3/81 + O(x**4)
    # 关于 im(ndir) == 0 的测试
    assert atanh(-I*x**2 + x - 2)._eval_nseries(x, 4, None) == -I*pi/2 - log(3)/2 - x/3 + \
    x**2*(-S(1)/4 + I/2) + x**2*(S(1)/36 - I/6) + x**3*(-S(1)/6 + I/2) + x**3*(S(1)/162 - I/18) + O(x**4)


def test_atanh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: atanh(x).fdiff(2))



# 调用 raises 函数来测试异常情况
raises(
    # 抛出 ArgumentIndexError 异常，如果以下 lambda 表达式执行时抛出异常
    ArgumentIndexError, 
    # lambda 表达式计算 atanh(x) 的二阶导数
    lambda: atanh(x).fdiff(2)
)
def test_acoth():
    x = Symbol('x')

    # 在特定点上进行断言
    assert acoth(0) == I*pi/2
    assert acoth(I) == -I*pi/4
    assert acoth(-I) == I*pi/4
    assert acoth(1) is oo
    assert acoth(-1) is -oo
    assert acoth(nan) is nan

    # 在无穷点上进行断言
    assert acoth(oo) == 0
    assert acoth(-oo) == 0
    assert acoth(I*oo) == 0
    assert acoth(-I*oo) == 0
    assert acoth(zoo) == 0

    # 属性测试
    assert acoth(-x) == -acoth(x)

    assert acoth(I/sqrt(3)) == -I*pi/3
    assert acoth(-I/sqrt(3)) == I*pi/3
    assert acoth(I*sqrt(3)) == -I*pi/6
    assert acoth(-I*sqrt(3)) == I*pi/6
    assert acoth(I*(1 + sqrt(2))) == -pi*I/8
    assert acoth(-I*(sqrt(2) + 1)) == pi*I/8
    assert acoth(I*(1 - sqrt(2))) == pi*I*Rational(3, 8)
    assert acoth(I*(sqrt(2) - 1)) == pi*I*Rational(-3, 8)
    assert acoth(I*sqrt(5 + 2*sqrt(5))) == -I*pi/10
    assert acoth(-I*sqrt(5 + 2*sqrt(5))) == I*pi/10
    assert acoth(I*(2 + sqrt(3))) == -pi*I/12
    assert acoth(-I*(2 + sqrt(3))) == pi*I/12
    assert acoth(I*(2 - sqrt(3))) == pi*I*Rational(-5, 12)
    assert acoth(I*(sqrt(3) - 2)) == pi*I*Rational(5, 12)

    # 真实性测试
    assert acoth(S(2)).is_real is True
    assert acoth(S(2)).is_finite is True
    assert acoth(S(2)).is_extended_real is True
    assert acoth(S(-2)).is_real is True
    assert acoth(S(1)).is_real is False
    assert acoth(S(1)).is_extended_real is True
    assert acoth(S(-1)).is_real is False
    assert acoth(symbols('y', real=True)).is_real is None

    # 对称性测试
    assert acoth(Rational(-1, 2)) == -acoth(S.Half)


def test_acoth_rewrite():
    x = Symbol('x')
    assert acoth(x).rewrite(log) == (log(1 + 1/x) - log(1 - 1/x)) / 2
    assert acoth(x).rewrite(atanh) == atanh(1/x)
    assert acoth(x).rewrite(asinh) == \
        x*sqrt(x**(-2))*asinh(sqrt(1/(x**2 - 1))) + I*pi*(sqrt((x - 1)/x)*sqrt(x/(x - 1)) - sqrt(x/(x + 1))*sqrt(1 + 1/x))/2


def test_acoth_leading_term():
    x = Symbol('x')
    # 关于分支点的测试
    assert acoth(x + 1).as_leading_term(x, cdir=1) == -log(x)/2 + log(2)/2
    assert acoth(x + 1).as_leading_term(x, cdir=-1) == -log(x)/2 + log(2)/2
    assert acoth(x - 1).as_leading_term(x, cdir=1) == log(x)/2 - log(2)/2 + I*pi/2
    assert acoth(x - 1).as_leading_term(x, cdir=-1) == log(x)/2 - log(2)/2 - I*pi/2
    # 关于位于分支切线上的点的测试
    assert acoth(x).as_leading_term(x, cdir=-1) == I*pi/2
    assert acoth(x).as_leading_term(x, cdir=1) == -I*pi/2
    assert acoth(I*x + 1/2).as_leading_term(x, cdir=1) == acoth(1/2)
    assert acoth(-I*x + 1/2).as_leading_term(x, cdir=1) == acoth(1/2) + I*pi
    assert acoth(I*x - 1/2).as_leading_term(x, cdir=1) == -I*pi - acoth(1/2)
    assert acoth(-I*x - 1/2).as_leading_term(x, cdir=1) == -acoth(1/2)
    # 关于 im(ndir) == 0 的测试
    assert acoth(-I*x**2 - x - S(1)/2).as_leading_term(x, cdir=1) == -log(3)/2 + I*pi/2
    assert acoth(-I*x**2 - x - S(1)/2).as_leading_term(x, cdir=-1) == -log(3)/2 + I*pi/2
def test_acoth_series():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言 acoth(x) 的级数展开结果是否等于给定表达式
    assert acoth(x).series(x, 0, 10) == \
        -I*pi/2 + x + x**3/3 + x**5/5 + x**7/7 + x**9/9 + O(x**10)


def test_acoth_nseries():
    # 定义符号变量 x
    x = Symbol('x')
    # 测试涉及分支点
    assert acoth(x + 1)._eval_nseries(x, 4, None) == log(2)/2 - log(x)/2 + x/4 - \
    x**2/16 + x**3/48 + O(x**4)
    assert acoth(x - 1)._eval_nseries(x, 4, None, cdir=1) == I*pi/2 - log(2)/2 + \
    log(x)/2 + x/4 + x**2/16 + x**3/48 + O(x**4)
    assert acoth(x - 1)._eval_nseries(x, 4, None, cdir=-1) == -I*pi/2 - log(2)/2 + \
    log(x)/2 + x/4 + x**2/16 + x**3/48 + O(x**4)
    # 测试涉及分支切割线上的点
    assert acoth(I*x + S(1)/2)._eval_nseries(x, 4, None, cdir=1) == acoth(S(1)/2) + \
    4*I*x/3 - 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x + S(1)/2)._eval_nseries(x, 4, None, cdir=-1) == I*pi + \
    acoth(S(1)/2) + 4*I*x/3 - 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x - S(1)/2)._eval_nseries(x, 4, None, cdir=1) == -acoth(S(1)/2) - \
    I*pi + 4*I*x/3 + 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x - S(1)/2)._eval_nseries(x, 4, None, cdir=-1) == -acoth(S(1)/2) + \
    4*I*x/3 + 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    # 测试 im(ndir) == 0 的情况
    assert acoth(-I*x**2 - x - S(1)/2)._eval_nseries(x, 4, None) == I*pi/2 - log(3)/2 - \
    4*x/3 + x**2*(-S(8)/9 + 2*I/3) - 2*I*x**2 + x**3*(S(104)/81 - 16*I/9) - 8*x**3/3 + O(x**4)


def test_acoth_fdiff():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言对 acoth(x) 调用 fdiff(2) 时会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: acoth(x).fdiff(2))


def test_inverses():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言各反函数的逆函数是否符合预期
    assert sinh(x).inverse() == asinh
    raises(AttributeError, lambda: cosh(x).inverse())
    assert tanh(x).inverse() == atanh
    assert coth(x).inverse() == acoth
    assert asinh(x).inverse() == sinh
    assert acosh(x).inverse() == cosh
    assert atanh(x).inverse() == tanh
    assert acoth(x).inverse() == coth
    assert asech(x).inverse() == sech
    assert acsch(x).inverse() == csch


def test_leading_term():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言 cosh(x) 和 coth(x) 的主导项
    assert cosh(x).as_leading_term(x) == 1
    assert coth(x).as_leading_term(x) == 1/x
    # 对于 sinh(x), tanh(x) 类型的函数，验证其主导项
    for func in [sinh, tanh]:
        assert func(x).as_leading_term(x) == x
    # 对于 sinh, cosh, tanh, coth 类型的函数，和 ar = 1/x, S.Half 的组合，验证其主导项
    for func in [sinh, cosh, tanh, coth]:
        for ar in (1/x, S.Half):
            eq = func(ar)
            assert eq.as_leading_term(x) == eq
    # 对于 csch, sech 类型的函数，验证其主导项
    for func in [csch, sech]:
        eq = func(S.Half)
        assert eq.as_leading_term(x) == eq


def test_complex():
    # 定义实数符号变量 a, b
    a, b = symbols('a,b', real=True)
    # 定义复数变量 z
    z = a + b*I
    # 对 sinh, cosh, tanh, coth, sech, csch 函数进行复共轭的验证
    for func in [sinh, cosh, tanh, coth, sech, csch]:
        assert func(z).conjugate() == func(a - b*I)
    # 对于每个布尔值深度参数，分别进行以下断言：
    for deep in [True, False]:
        # 断言 hyperbolic sine 函数的展开结果
        assert sinh(z).expand(complex=True, deep=deep) == sinh(a)*cos(b) + I*cosh(a)*sin(b)
        # 断言 hyperbolic cosine 函数的展开结果
        assert cosh(z).expand(complex=True, deep=deep) == cosh(a)*cos(b) + I*sinh(a)*sin(b)
        # 断言 hyperbolic tangent 函数的展开结果
        assert tanh(z).expand(complex=True, deep=deep) == sinh(a)*cosh(a)/(cos(b)**2 + sinh(a)**2) + I*sin(b)*cos(b)/(cos(b)**2 + sinh(a)**2)
        # 断言 hyperbolic cotangent 函数的展开结果
        assert coth(z).expand(complex=True, deep=deep) == sinh(a)*cosh(a)/(sin(b)**2 + sinh(a)**2) - I*sin(b)*cos(b)/(sin(b)**2 + sinh(a)**2)
        # 断言 hyperbolic cosecant 函数的展开结果
        assert csch(z).expand(complex=True, deep=deep) == cos(b) * sinh(a) / (sin(b)**2*cosh(a)**2 + cos(b)**2*sinh(a)**2) - I*sin(b) * cosh(a) / (sin(b)**2*cosh(a)**2 + cos(b)**2*sinh(a)**2)
        # 断言 hyperbolic secant 函数的展开结果
        assert sech(z).expand(complex=True, deep=deep) == cos(b) * cosh(a) / (sin(b)**2*sinh(a)**2 + cos(b)**2*cosh(a)**2) - I*sin(b) * sinh(a) / (sin(b)**2*sinh(a)**2 + cos(b)**2*cosh(a)**2)
# 定义一个名为 test_complex_2899 的测试函数
def test_complex_2899():
    # 使用 sympy 的 symbols 函数定义两个实数符号 a 和 b
    a, b = symbols('a,b', real=True)
    # 循环迭代 deep 和 func，其中 deep 取值为 True 和 False
    for deep in [True, False]:
        # 循环迭代 func，包括 sinh, cosh, tanh, coth 四个函数
        for func in [sinh, cosh, tanh, coth]:
            # 断言特定函数对符号 a 的展开结果等于该函数在复数域上对符号 a 的值
            assert func(a).expand(complex=True, deep=deep) == func(a)


# 定义一个名为 test_simplifications 的测试函数
def test_simplifications():
    # 使用 sympy 的 Symbol 函数定义一个符号 x
    x = Symbol('x')
    # 断言 sinh 函数的反函数 asinh(x) 的结果等于 x
    assert sinh(asinh(x)) == x
    # 断言 sinh 函数的反函数 acosh(x) 的结果等于 sqrt(x - 1) * sqrt(x + 1)
    assert sinh(acosh(x)) == sqrt(x - 1) * sqrt(x + 1)
    # 断言 sinh 函数的反函数 atanh(x) 的结果等于 x / sqrt(1 - x**2)
    assert sinh(atanh(x)) == x / sqrt(1 - x**2)
    # 断言 sinh 函数的反函数 acoth(x) 的结果等于 1 / (sqrt(x - 1) * sqrt(x + 1))
    assert sinh(acoth(x)) == 1 / (sqrt(x - 1) * sqrt(x + 1))

    # 同理断言 cosh, tanh, coth, csch, sech 函数及其反函数的各种情况


# 定义一个名为 test_issue_4136 的测试函数
def test_issue_4136():
    # 断言 cosh 函数的反函数 asinh(Integer(3)/2) 的结果等于 sqrt(Integer(13)/4)
    assert cosh(asinh(Integer(3)/2)) == sqrt(Integer(13)/4)


# 定义一个名为 test_sinh_rewrite 的测试函数
def test_sinh_rewrite():
    # 使用 sympy 的 Symbol 函数定义一个符号 x
    x = Symbol('x')
    # 断言 sinh(x) 函数的 rewrite(exp) 结果等于 (exp(x) - exp(-x))/2
    assert sinh(x).rewrite(exp) == (exp(x) - exp(-x))/2 \
        == sinh(x).rewrite('tractable')
    # 断言 sinh(x) 函数的 rewrite(cosh) 结果等于 -I*cosh(x + I*pi/2)
    assert sinh(x).rewrite(cosh) == -I*cosh(x + I*pi/2)
    # 同理断言 sinh(x) 函数的 rewrite(tanh) 和 rewrite(coth) 的结果


# 定义一个名为 test_cosh_rewrite 的测试函数
def test_cosh_rewrite():
    # 使用 sympy 的 Symbol 函数定义一个符号 x
    x = Symbol('x')
    # 断言 cosh(x) 函数的 rewrite(exp) 结果等于 (exp(x) + exp(-x))/2
    assert cosh(x).rewrite(exp) == (exp(x) + exp(-x))/2 \
        == cosh(x).rewrite('tractable')
    # 断言 cosh(x) 函数的 rewrite(sinh) 结果等于 -I*sinh(x + I*pi/2, evaluate=False)
    assert cosh(x).rewrite(sinh) == -I*sinh(x + I*pi/2, evaluate=False)
    # 同理断言 cosh(x) 函数的 rewrite(tanh) 和 rewrite(coth) 的结果


# 定义一个名为 test_tanh_rewrite 的测试函数
def test_tanh_rewrite():
    # 使用 sympy 的 Symbol 函数定义一个符号 x
    x = Symbol('x')
    # 断言 tanh(x) 函数的 rewrite(exp) 结果等于 (exp(x) - exp(-x))/(exp(x) + exp(-x))
    assert tanh(x).rewrite(exp) == (exp(x) - exp(-x))/(exp(x) + exp(-x)) \
        == tanh(x).rewrite('tractable')
    # 断言 tanh(x) 函数的 rewrite(sinh) 结果等于 I*sinh(x)/sinh(I*pi/2 - x, evaluate=False)
    assert tanh(x).rewrite(sinh) == I*sinh(x)/sinh(I*pi/2 - x, evaluate=False)
    # 同理断言 tanh(x) 函数的 rewrite(cosh) 和 rewrite(coth) 的结果


# 定义一个名为 test_coth_rewrite 的测试函数
def test_coth_rewrite():
    # 使用 sympy 的 Symbol 函数定义一个符号 x
    x = Symbol('x')
    # 断言 coth(x) 函数的 rewrite(exp) 结果等于 (exp(x) + exp(-x))/(exp(x) - exp(-x))
    assert coth(x).rewrite(exp) == (exp(x) + exp(-x))/(exp(x) - exp(-x)) \
        == coth(x).rewrite('tractable')
    # 断言 coth(x) 函数的 rewrite(sinh) 结果等于 -I*sinh(I*pi/2 - x, evaluate=False)/sinh(x)
    assert coth(x).rewrite(sinh) == -I*sinh(I*pi/2 - x, evaluate=False)/sinh(x)
    # 同理断言 coth(x) 函数的 rewrite(cosh) 和 rewrite(tanh) 的结果
def test_csch_rewrite():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言 csch(x) 通过指数函数重写为 1 / (exp(x)/2 - exp(-x)/2)
    assert csch(x).rewrite(exp) == 1 / (exp(x)/2 - exp(-x)/2) \
        # 另一种写法，也应等于 'tractable'
        == csch(x).rewrite('tractable')
    # 断言 csch(x) 通过双曲余弦重写为 I/cosh(x + I*pi/2, evaluate=False)
    assert csch(x).rewrite(cosh) == I/cosh(x + I*pi/2, evaluate=False)
    # 计算 tanh(S.Half*x)
    tanh_half = tanh(S.Half*x)
    # 断言 csch(x) 通过双曲正切重写为 (1 - tanh_half**2)/(2*tanh_half)
    assert csch(x).rewrite(tanh) == (1 - tanh_half**2)/(2*tanh_half)
    # 计算 coth(S.Half*x)
    coth_half = coth(S.Half*x)
    # 断言 csch(x) 通过双曲余切重写为 (coth_half**2 - 1)/(2*coth_half)
    assert csch(x).rewrite(coth) == (coth_half**2 - 1)/(2*coth_half)


def test_sech_rewrite():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言 sech(x) 通过指数函数重写为 1 / (exp(x)/2 + exp(-x)/2)
    assert sech(x).rewrite(exp) == 1 / (exp(x)/2 + exp(-x)/2) \
        # 另一种写法，也应等于 'tractable'
        == sech(x).rewrite('tractable')
    # 断言 sech(x) 通过双曲正弦重写为 I/sinh(x + I*pi/2, evaluate=False)
    assert sech(x).rewrite(sinh) == I/sinh(x + I*pi/2, evaluate=False)
    # 计算 tanh(S.Half*x)**2
    tanh_half = tanh(S.Half*x)**2
    # 断言 sech(x) 通过双曲正切重写为 (1 - tanh_half)/(1 + tanh_half)
    assert sech(x).rewrite(tanh) == (1 - tanh_half)/(1 + tanh_half)
    # 计算 coth(S.Half*x)**2
    coth_half = coth(S.Half*x)**2
    # 断言 sech(x) 通过双曲余切重写为 (coth_half - 1)/(coth_half + 1)
    assert sech(x).rewrite(coth) == (coth_half - 1)/(coth_half + 1)


def test_derivs():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言 coth(x) 的导数为 -sinh(x)**(-2)
    assert coth(x).diff(x) == -sinh(x)**(-2)
    # 断言 sinh(x) 的导数为 cosh(x)
    assert sinh(x).diff(x) == cosh(x)
    # 断言 cosh(x) 的导数为 sinh(x)
    assert cosh(x).diff(x) == sinh(x)
    # 断言 tanh(x) 的导数为 -tanh(x)**2 + 1
    assert tanh(x).diff(x) == -tanh(x)**2 + 1
    # 断言 csch(x) 的导数为 -coth(x)*csch(x)
    assert csch(x).diff(x) == -coth(x)*csch(x)
    # 断言 sech(x) 的导数为 -tanh(x)*sech(x)
    assert sech(x).diff(x) == -tanh(x)*sech(x)
    # 断言 acoth(x) 的导数为 1/(-x**2 + 1)
    assert acoth(x).diff(x) == 1/(-x**2 + 1)
    # 断言 asinh(x) 的导数为 1/sqrt(x**2 + 1)
    assert asinh(x).diff(x) == 1/sqrt(x**2 + 1)
    # 断言 acosh(x) 的导数为 1/(sqrt(x - 1)*sqrt(x + 1))
    assert acosh(x).diff(x) == 1/(sqrt(x - 1)*sqrt(x + 1))
    # 断言 acosh(x) 通过对数重写后的导数等于原始形式的导数
    assert acosh(x).diff(x) == acosh(x).rewrite(log).diff(x).together()
    # 断言 atanh(x) 的导数为 1/(-x**2 + 1)
    assert atanh(x).diff(x) == 1/(-x**2 + 1)
    # 断言 asech(x) 的导数为 -1/(x*sqrt(1 - x**2))
    assert asech(x).diff(x) == -1/(x*sqrt(1 - x**2))
    # 断言 acsch(x) 的导数为 -1/(x**2*sqrt(1 + x**(-2)))


def test_sinh_expansion():
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 断言 sinh(x+y) 在展开三角函数时等于 sinh(x)*cosh(y) + cosh(x)*sinh(y)
    assert sinh(x+y).expand(trig=True) == sinh(x)*cosh(y) + cosh(x)*sinh(y)
    # 断言 sinh(2*x) 在展开三角函数时等于 2*sinh(x)*cosh(x)
    assert sinh(2*x).expand(trig=True) == 2*sinh(x)*cosh(x)
    # 断言 sinh(3*x) 在展开三角函数后展开等于 sinh(x)**3 + 3*sinh(x)*cosh(x)**2
    assert sinh(3*x).expand(trig=True).expand() == \
        sinh(x)**3 + 3*sinh(x)*cosh(x)**2


def test_cosh_expansion():
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 断言 cosh(x+y) 在展开三角函数时等于 cosh(x)*cosh(y) + sinh(x)*sinh(y)
    assert cosh(x+y).expand(trig=True) == cosh(x)*cosh(y) + sinh(x)*sinh(y)
    # 断言 cosh(2*x) 在展开三角函数时等于 cosh(x)**2 + sinh(x)**2
    assert cosh(2*x).expand(trig=True) == cosh(x)**2 + sinh(x)**2
    # 断言 cosh(3*x) 在展开三角函数后展开等于 3*sinh(x)**2*cosh(x) + cosh(x)**3
    assert cosh(3*x).expand(trig=True).expand() == \
        3*sinh(x)**2*cosh(x) + cosh(x)**3


def test_cosh_positive():
    # 测试 cosh(x) 对于实数 x 是正数
    k = symbols('k', real=True)
    n = symbols('n', integer=True)

    assert cosh(k, evaluate=False).is_positive is True
    assert cosh(k + 2*n*pi*I, evaluate=False).is_positive is True
    assert cosh(I*pi/4, evaluate=False).is_positive is True
    assert cosh(3*I*pi/4, evaluate=False).is_positive is False


def test_cosh_nonnegative():
    # 测试 cosh(x) 对于实数 x 是非负数
    k = symbols('k', real=True)
    n = symbols('n', integer=True)

    assert cosh(k, evaluate=False).is_nonnegative is True
    assert cosh(k + 2*n*pi*I, evaluate=False).is_nonnegative is True
    assert cosh(I*pi/4, evaluate=False).is_nonnegative is True
    assert cosh(3*I*pi/4, evaluate=False).is_nonnegative is False
    assert cosh(S.Zero, evaluate=False).is_nonnegative is True


def test_real_assumptions():
    # 定义一个虚数符号变量 z
    z = Symbol('z', real=False)
    # 断言 sinh(z) 的返回值的实部是 None
    assert sinh(z).is_real is None
    # 断言 cosh(z) 的返回值的实部是 None
    assert cosh(z).is_real is None
    # 断言 tanh(z) 的返回值的实部是 None
    assert tanh(z).is_real is None
    # 断言 sech(z) 的返回值的实部是 None
    assert sech(z).is_real is None
    # 断言 csch(z) 的返回值的实部是 None
    assert csch(z).is_real is None
    # 断言 coth(z) 的返回值的实部是 None
    assert coth(z).is_real is None
# 定义测试函数，验证一些符号数学函数的假设
def test_sign_assumptions():
    # 定义符号 p，具有正属性
    p = Symbol('p', positive=True)
    # 定义符号 n，具有负属性
    n = Symbol('n', negative=True)
    # 断言 sinh(n) 的结果为负
    assert sinh(n).is_negative is True
    # 断言 sinh(p) 的结果为正
    assert sinh(p).is_positive is True
    # 断言 cosh(n) 的结果为正
    assert cosh(n).is_positive is True
    # 断言 cosh(p) 的结果为正
    assert cosh(p).is_positive is True
    # 断言 tanh(n) 的结果为负
    assert tanh(n).is_negative is True
    # 断言 tanh(p) 的结果为正
    assert tanh(p).is_positive is True
    # 断言 csch(n) 的结果为负
    assert csch(n).is_negative is True
    # 断言 csch(p) 的结果为正
    assert csch(p).is_positive is True
    # 断言 sech(n) 的结果为正
    assert sech(n).is_positive is True
    # 断言 sech(p) 的结果为正
    assert sech(p).is_positive is True
    # 断言 coth(n) 的结果为负
    assert coth(n).is_negative is True
    # 断言 coth(p) 的结果为正
    assert coth(p).is_positive is True


def test_issue_25847():
    x = Symbol('x')

    # 测试 atanh 函数
    assert atanh(sin(x)/x).as_leading_term(x) == atanh(sin(x)/x)
    # 断言 atanh(exp(1/x)) 引发 PoleError 异常
    raises(PoleError, lambda: atanh(exp(1/x)).as_leading_term(x))

    # 测试 asinh 函数
    assert asinh(sin(x)/x).as_leading_term(x) == log(1 + sqrt(2))
    # 断言 asinh(exp(1/x)) 引发 PoleError 异常
    raises(PoleError, lambda: asinh(exp(1/x)).as_leading_term(x))

    # 测试 acosh 函数
    assert acosh(sin(x)/x).as_leading_term(x) == 0
    # 断言 acosh(exp(1/x)) 引发 PoleError 异常
    raises(PoleError, lambda: acosh(exp(1/x)).as_leading_term(x))

    # 测试 acoth 函数
    assert acoth(sin(x)/x).as_leading_term(x) == acoth(sin(x)/x)
    # 断言 acoth(exp(1/x)) 引发 PoleError 异常
    raises(PoleError, lambda: acoth(exp(1/x)).as_leading_term(x))

    # 测试 asech 函数
    assert asech(sinh(x)/x).as_leading_term(x) == 0
    # 断言 asech(exp(1/x)) 引发 PoleError 异常
    raises(PoleError, lambda: asech(exp(1/x)).as_leading_term(x))

    # 测试 acsch 函数
    assert acsch(sin(x)/x).as_leading_term(x) == log(1 + sqrt(2))
    # 断言 acsch(exp(1/x)) 引发 PoleError 异常
    raises(PoleError, lambda: acsch(exp(1/x)).as_leading_term(x))


def test_issue_25175():
    x = Symbol('x')
    # 定义数学表达式 g1
    g1 = 2*acosh(1 + 2*x/3) - acosh(S(5)/3 - S(8)/3/(x + 4))
    # 定义数学表达式 g2
    g2 = 2*log(sqrt((x + 4)/3)*(sqrt(x + 3)+sqrt(x))**2/(2*sqrt(x + 3) + sqrt(x)))
    # 断言 g1 - g2 在 x 的级数展开中的结果为 O(x**6)
    assert (g1 - g2).series(x) == O(x**6)
```