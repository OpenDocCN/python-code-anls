# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\test_integers.py`

```
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, tan
from sympy.polys.rootoftools import RootOf, CRootOf
from sympy import Integers
from sympy.sets.sets import Interval
from sympy.sets.fancysets import ImageSet
from sympy.core.function import Lambda

# 定义符号变量
x = Symbol('x')
i = Symbol('i', imaginary=True)
y = Symbol('y', real=True)
k, n = symbols('k,n', integer=True)

# 定义测试函数 test_floor
def test_floor():

    # 测试 NaN 的情况，应该返回 NaN
    assert floor(nan) is nan

    # 测试正无穷的情况，应该返回正无穷
    assert floor(oo) is oo
    # 测试负无穷的情况，应该返回负无穷
    assert floor(-oo) is -oo
    # 测试复无穷的情况，应该返回复无穷
    assert floor(zoo) is zoo

    # 测试整数 0，应该返回 0
    assert floor(0) == 0

    # 测试正整数 1，应该返回 1
    assert floor(1) == 1
    # 测试负整数 -1，应该返回 -1
    assert floor(-1) == -1

    # 测试常数 E，应该返回大于 E 的最小整数，即 2
    assert floor(E) == 2
    # 测试常数 -E，应该返回小于 -E 的最大整数，即 -3
    assert floor(-E) == -3

    # 测试常数 2*E，应该返回大于 2*E 的最小整数，即 5
    assert floor(2*E) == 5
    # 测试常数 -2*E，应该返回小于 -2*E 的最大整数，即 -6
    assert floor(-2*E) == -6

    # 测试常数 pi，应该返回大于 pi 的最小整数，即 3
    assert floor(pi) == 3
    # 测试常数 -pi，应该返回小于 -pi 的最大整数，即 -4
    assert floor(-pi) == -4

    # 测试常数 S.Half，应该返回 0
    assert floor(S.Half) == 0
    # 测试有理数 -1/2，应该返回 -1
    assert floor(Rational(-1, 2)) == -1

    # 测试有理数 7/3，应该返回 2
    assert floor(Rational(7, 3)) == 2
    # 测试有理数 -7/3，应该返回 -3
    assert floor(Rational(-7, 3)) == -3
    # 测试有理数 -7/3，应该返回 -3
    assert floor(-Rational(7, 3)) == -3

    # 测试浮点数 17.0，应该返回 17
    assert floor(Float(17.0)) == 17
    # 测试浮点数 -17.0，应该返回 -17
    assert floor(-Float(17.0)) == -17

    # 测试浮点数 7.69，应该返回 7
    assert floor(Float(7.69)) == 7
    # 测试浮点数 -7.69，应该返回 -8
    assert floor(-Float(7.69)) == -8

    # 测试虚数单位 I，应该返回 I
    assert floor(I) == I
    # 测试虚数单位 -I，应该返回 -I
    assert floor(-I) == -I
    # 测试虚数单位 i，应该返回 floor(i)
    e = floor(i)
    assert e.func is floor and e.args[0] == i

    # 测试复数 oo*I，应该返回 oo*I
    assert floor(oo*I) == oo*I
    # 测试复数 -oo*I，应该返回 -oo*I
    assert floor(-oo*I) == -oo*I
    # 测试复数 exp(I*pi/4)*oo，应该返回 exp(I*pi/4)*oo
    assert floor(exp(I*pi/4)*oo) == exp(I*pi/4)*oo

    # 测试复数 2*I，应该返回 2*I
    assert floor(2*I) == 2*I
    # 测试复数 -2*I，应该返回 -2*I
    assert floor(-2*I) == -2*I

    # 测试复数 I/2，应该返回 0
    assert floor(I/2) == 0
    # 测试复数 -I/2，应该返回 -I
    assert floor(-I/2) == -I

    # 测试表达式 E + 17，应该返回 19
    assert floor(E + 17) == 19
    # 测试表达式 pi + 2，应该返回 5
    assert floor(pi + 2) == 5

    # 测试表达式 E + pi，应该返回 5
    assert floor(E + pi) == 5
    # 测试表达式 I + pi，应该返回 3 + I
    assert floor(I + pi) == 3 + I

    # 测试两次 floor 函数嵌套，应该返回 3
    assert floor(floor(pi)) == 3
    # 测试 floor(y)，应该返回 floor(y)
    assert floor(floor(y)) == floor(y)
    # 测试 floor(x)，应该返回 floor(x)
    assert floor(floor(x)) == floor(x)

    # 测试 unchanged 函数对 x 的作用，应该返回 True
    assert unchanged(floor, x)
    # 测试 unchanged 函数对 2*x 的作用，应该返回 True
    assert unchanged(floor, 2*x)
    # 测试 unchanged 函数对 k*x 的作用，应该返回 True
    assert unchanged(floor, k*x)

    # 测试 floor 函数对 k 的作用，应该返回 k
    assert floor(k) == k
    # 测试 floor 函数对 2*k 的作用，应该返回 2*k
    assert floor(2*k) == 2*k
    # 测试 floor 函数对 k*n 的作用，应该返回 k*n
    assert floor(k*n) == k*n

    # 测试 unchanged 函数对 k/2 的作用，应该返回 True
    assert unchanged(floor, k/2)

    # 测试 unchanged 函数对 x + y 的作用，应该返回 True
    assert unchanged(floor, x + y)

    # 测试 floor 函数对 x + 3 的作用，应该返回 floor(x) + 3
    assert floor(x + 3) == floor(x) + 3
    # 测试 floor 函数对 x + k 的作用，应该返回 floor(x) + k
    assert floor(x + k) == floor(x) + k

    # 测试 floor 函数对 y + 3 的作用，应该返回 floor(y) + 3
    assert floor(y + 3) == floor(y) + 3
    # 测试 floor 函数对 y + k 的作用，应该返回 floor(y) + k
    assert floor(y + k) == floor(y) + k

    # 测试 floor 函数对 3 + I*y + pi 的作用，应该返回 6 + floor(y)*I
    assert floor(3 + I*y + pi) == 6 + floor(y)*I

    # 测试 floor 函数对 k + n 的作用，应该返回 k + n
    assert floor(k + n) == k + n

    # 测试 unchanged 函数对 x*I 的作用，应该返回 True
    assert unchanged(floor, x*I)
    # 测试 floor 函数对 k*I 的作用，应该返回 k*I
    assert floor(k*I) == k*I

    # 测试 floor 函数对 Rational(23, 10) - E*I 的作用，应该返回 2 - 3*I
    assert floor(Rational(23, 10) - E*I) == 2
    # 断言：e^2 的下取整应该等于 7
    assert floor(exp(2)) == 7

    # 断言：log2(8) 的下取整不应该等于 2
    assert floor(log(8)/log(2)) != 2
    # 断言：log2(8) 的下取整经过精确求值应该等于 3
    assert int(floor(log(8)/log(2)).evalf(chop=True)) == 3

    # 断言：50 的阶乘除以 e 的下取整应该等于指定的大整数值
    assert (floor(factorial(50)/exp(1)) ==
            11188719610782480504630258070757734324011354208865721592720336800)

    # 一系列关于下取整函数 floor 的比较断言
    assert (floor(y) < y) == False
    assert (floor(y) <= y) == True
    assert (floor(y) > y) == False
    assert (floor(y) >= y) == False
    assert (floor(x) <= x).is_Relational  # x 可能是非实数
    assert (floor(x) > x).is_Relational
    assert (floor(x) <= y).is_Relational  # 参数不同于右侧
    assert (floor(x) > y).is_Relational
    assert (floor(y) <= oo) == True
    assert (floor(y) < oo) == True
    assert (floor(y) >= -oo) == True
    assert (floor(y) > -oo) == True

    # 断言：使用 frac 重写 floor(y) 的结果
    assert floor(y).rewrite(frac) == y - frac(y)
    assert floor(y).rewrite(ceiling) == -ceiling(-y)
    assert floor(y).rewrite(frac).subs(y, -pi) == floor(-pi)
    assert floor(y).rewrite(frac).subs(y, E) == floor(E)
    assert floor(y).rewrite(ceiling).subs(y, E) == -ceiling(-E)
    assert floor(y).rewrite(ceiling).subs(y, -pi) == -ceiling(pi)

    # 断言：floor(y) 等于 y - frac(y) 和 -ceiling(-y)
    assert Eq(floor(y), y - frac(y))
    assert Eq(floor(y), -ceiling(-y))

    # 符号声明
    neg = Symbol('neg', negative=True)
    nn = Symbol('nn', nonnegative=True)
    pos = Symbol('pos', positive=True)
    np = Symbol('np', nonpositive=True)

    # 一系列关于不同符号的 floor 断言
    assert (floor(neg) < 0) == True
    assert (floor(neg) <= 0) == True
    assert (floor(neg) > 0) == False
    assert (floor(neg) >= 0) == False
    assert (floor(neg) <= -1) == True
    assert (floor(neg) >= -3) == (neg >= -3)
    assert (floor(neg) < 5) == (neg < 5)

    assert (floor(nn) < 0) == False
    assert (floor(nn) >= 0) == True

    assert (floor(pos) < 0) == False
    assert (floor(pos) <= 0) == (pos < 1)
    assert (floor(pos) > 0) == (pos >= 1)
    assert (floor(pos) >= 0) == True
    assert (floor(pos) >= 3) == (pos >= 3)

    assert (floor(np) <= 0) == True
    assert (floor(np) > 0) == False

    # floor 函数的属性断言
    assert floor(neg).is_negative == True
    assert floor(neg).is_nonnegative == False
    assert floor(nn).is_negative == False
    assert floor(nn).is_nonnegative == True
    assert floor(pos).is_negative == False
    assert floor(pos).is_nonnegative == True
    assert floor(np).is_negative is None
    assert floor(np).is_nonnegative is None

    # 使用 evaluate=False 的 floor 函数的比较断言
    assert (floor(7, evaluate=False) >= 7) == True
    assert (floor(7, evaluate=False) > 7) == False
    assert (floor(7, evaluate=False) <= 7) == True
    assert (floor(7, evaluate=False) < 7) == False

    assert (floor(7, evaluate=False) >= 6) == True
    assert (floor(7, evaluate=False) > 6) == True
    assert (floor(7, evaluate=False) <= 6) == False
    assert (floor(7, evaluate=False) < 6) == False

    assert (floor(7, evaluate=False) >= 8) == False
    assert (floor(7, evaluate=False) > 8) == False
    assert (floor(7, evaluate=False) <= 8) == True
    assert (floor(7, evaluate=False) < 8) == True

    # 断言：floor(x) 小于等于 5.5
    assert (floor(x) <= 5.5) == Le(floor(x), 5.5, evaluate=False)
    # 断言：向下取整(floor)的结果与给定的值比较
    assert (floor(x) >= -3.2) == Ge(floor(x), -3.2, evaluate=False)
    # 断言：向下取整(floor)的结果与给定的值比较
    assert (floor(x) < 2.9) == Lt(floor(x), 2.9, evaluate=False)
    # 断言：向下取整(floor)的结果与给定的值比较
    assert (floor(x) > -1.7) == Gt(floor(x), -1.7, evaluate=False)

    # 断言：向下取整(floor)的结果与给定的值比较，转换为与y的比较
    assert (floor(y) <= 5.5) == (y < 6)
    # 断言：向下取整(floor)的结果与给定的值比较，转换为与y的比较
    assert (floor(y) >= -3.2) == (y >= -3)
    # 断言：向下取整(floor)的结果与给定的值比较，转换为与y的比较
    assert (floor(y) < 2.9) == (y < 3)
    # 断言：向下取整(floor)的结果与给定的值比较，转换为与y的比较
    assert (floor(y) > -1.7) == (y >= -1)

    # 断言：向下取整(floor)的结果与给定的值比较，转换为与n的比较
    assert (floor(y) <= n) == (y < n + 1)
    # 断言：向下取整(floor)的结果与给定的值比较，转换为与n的比较
    assert (floor(y) >= n) == (y >= n)
    # 断言：向下取整(floor)的结果与给定的值比较，转换为与n的比较
    assert (floor(y) < n) == (y < n)
    # 断言：向下取整(floor)的结果与给定的值比较，转换为与n的比较
    assert (floor(y) > n) == (y >= n + 1)

    # 断言：计算方程 x**3 - 27*x 的第二个根的向下取整结果是否为5
    assert floor(RootOf(x**3 - 27*x, 2)) == 5
def test_ceiling():

    # 确保对于 NaN，返回仍然是 NaN
    assert ceiling(nan) is nan

    # 对于正无穷大，返回仍然是正无穷大
    assert ceiling(oo) is oo
    # 对于负无穷大，返回仍然是负无穷大
    assert ceiling(-oo) is -oo
    # 对于复数无穷大，返回仍然是复数无穷大
    assert ceiling(zoo) is zoo

    # 对于整数 0，天花板函数返回仍然是 0
    assert ceiling(0) == 0

    # 对于正整数 1，天花板函数返回仍然是 1
    assert ceiling(1) == 1
    # 对于负整数 -1，天花板函数返回仍然是 -1
    assert ceiling(-1) == -1

    # 对于自然常数 e，天花板函数返回上限整数 3
    assert ceiling(E) == 3
    # 对于负的自然常数 e，天花板函数返回上限整数 -2
    assert ceiling(-E) == -2

    # 对于 2*E，天花板函数返回上限整数 6
    assert ceiling(2*E) == 6
    # 对于 -2*E，天花板函数返回上限整数 -5
    assert ceiling(-2*E) == -5

    # 对于圆周率 pi，天花板函数返回上限整数 4
    assert ceiling(pi) == 4
    # 对于负的圆周率 pi，天花板函数返回上限整数 -3
    assert ceiling(-pi) == -3

    # 对于有理数 S.Half (1/2)，天花板函数返回上限整数 1
    assert ceiling(S.Half) == 1
    # 对于负的有理数 -1/2，天花板函数返回上限整数 0
    assert ceiling(Rational(-1, 2)) == 0

    # 对于有理数 7/3，天花板函数返回上限整数 3
    assert ceiling(Rational(7, 3)) == 3
    # 对于负的有理数 -7/3，天花板函数返回上限整数 -2
    assert ceiling(-Rational(7, 3)) == -2

    # 对于浮点数 Float(17.0)，天花板函数返回上限整数 17
    assert ceiling(Float(17.0)) == 17
    # 对于负的浮点数 -Float(17.0)，天花板函数返回上限整数 -17
    assert ceiling(-Float(17.0)) == -17

    # 对于浮点数 Float(7.69)，天花板函数返回上限整数 8
    assert ceiling(Float(7.69)) == 8
    # 对于负的浮点数 -Float(7.69)，天花板函数返回上限整数 -7
    assert ceiling(-Float(7.69)) == -7

    # 对于虚数单位 I，天花板函数返回仍然是虚数单位 I
    assert ceiling(I) == I
    # 对于负的虚数单位 -I，天花板函数返回仍然是负的虚数单位 -I
    assert ceiling(-I) == -I
    # 对于复数 i，天花板函数返回一个与 i 相同的天花板函数对象 e
    e = ceiling(i)
    assert e.func is ceiling and e.args[0] == i

    # 对于正无穷大乘虚数单位 oo*I，天花板函数返回仍然是 oo*I
    assert ceiling(oo*I) == oo*I
    # 对于负无穷大乘虚数单位 -oo*I，天花板函数返回仍然是 -oo*I
    assert ceiling(-oo*I) == -oo*I
    # 对于 exp(I*pi/4)*oo，天花板函数返回仍然是 exp(I*pi/4)*oo
    assert ceiling(exp(I*pi/4)*oo) == exp(I*pi/4)*oo

    # 对于虚数单位乘以 2，天花板函数返回仍然是 2*I
    assert ceiling(2*I) == 2*I
    # 对于负的虚数单位乘以 -2，天花板函数返回仍然是 -2*I
    assert ceiling(-2*I) == -2*I

    # 对于虚数单位除以 2，天花板函数返回仍然是 I
    assert ceiling(I/2) == I
    # 对于负的虚数单位除以 -2，天花板函数返回上限整数 0
    assert ceiling(-I/2) == 0

    # 对于 E + 17，天花板函数返回上限整数 20
    assert ceiling(E + 17) == 20
    # 对于 pi + 2，天花板函数返回上限整数 6
    assert ceiling(pi + 2) == 6

    # 对于 E + pi，天花板函数返回上限整数 6
    assert ceiling(E + pi) == 6
    # 对于 I + pi，天花板函数返回 I + 4
    assert ceiling(I + pi) == I + 4

    # 对于天花板函数的嵌套 ceiling(pi)，天花板函数返回上限整数 4
    assert ceiling(ceiling(pi)) == 4
    # 对于天花板函数的嵌套 ceiling(y)，天花板函数返回与 y 的天花板函数结果相同
    assert ceiling(ceiling(y)) == ceiling(y)
    # 对于天花板函数的嵌套 ceiling(x)，天花板函数返回与 x 的天花板函数结果相同
    assert ceiling(ceiling(x)) == ceiling(x)

    # 确保 ceiling 函数对 x 的应用不改变结果
    assert unchanged(ceiling, x)
    assert unchanged(ceiling, 2*x)
    assert unchanged(ceiling, k*x)

    # 对于数值 k，天花板函数返回 k 本身
    assert ceiling(k) == k
    # 对于数值 2*k，天花板函数返回 2*k
    assert ceiling(2*k) == 2*k
    # 对于数值 k*n，天花板函数返回 k*n
    assert ceiling(k*n) == k*n

    # 确保 ceiling 函数对 k/2 的应用不改变结果
    assert unchanged(ceiling, k/2)

    # 确保 ceiling 函数对 x + y 的应用不改变结果
    assert unchanged(ceiling, x + y)

    # 对于表达式 x + 3，天花板函数返回 ceiling(x) + 3
    assert ceiling(x + 3) == ceiling(x) + 3
    # 对于表达式 x + 3.0，天花板函数返回 ceiling(x) + 3
    assert ceiling(x + 3.0) == ceiling(x) + 3
    # 对于表达式 x + 3.0*I，天花板函数返回 ceiling(x) + 3*I
    assert ceiling(x + 3.0*I) == ceiling(x) + 3*I
    # 对于表达式 x + k，天花板函数返回 ceiling(x) + k
    assert ceiling(x + k) == ceiling(x) + k

    # 对于表达式 y + 3，天花板函数返回 ceiling(y) + 3
    assert ceiling(y + 3) == ceiling(y) + 3
    # 对于表达式 y + k，天花板函数返回 ceiling(y) + k
    assert ceiling(y + k) == ceiling(y) + k

    # 对于复杂表达式 3 + pi + y*I，天花板函数返回上限整数 7 + ceiling(y)*I
    assert ceiling(3 + pi + y*I) == 7 + ceiling(y)*I

    # 对于表达式 k + n，天花板函数返回 k + n
    assert ceiling(k + n) == k + n

    # 确保 ceiling 函数对 x*I 的应用不改变结果
    assert unchanged(ceiling, x*I)
    # 对于 k*I，天花板函数返回 k*I
    # 断言：对于给定的 y，计算其 ceiling 值后，再用 floor 函数重写，应等于 -floor(-y)
    assert ceiling(y).rewrite(floor) == -floor(-y)
    
    # 断言：对于给定的 y，计算其 ceiling 值后，再用 frac 函数重写，应等于 y + frac(-y)
    assert ceiling(y).rewrite(frac) == y + frac(-y)
    
    # 断言：对于给定的 y，计算其 ceiling 值后再用 floor 函数重写，然后替换 y 为 -pi，应等于 -floor(pi)
    assert ceiling(y).rewrite(floor).subs(y, -pi) == -floor(pi)
    
    # 断言：对于给定的 y，计算其 ceiling 值后再用 floor 函数重写，然后替换 y 为 E，应等于 -floor(-E)
    assert ceiling(y).rewrite(floor).subs(y, E) == -floor(-E)
    
    # 断言：对于给定的 y，计算其 ceiling 值后再用 frac 函数重写，然后替换 y 为 pi，应等于 ceiling(pi)
    assert ceiling(y).rewrite(frac).subs(y, pi) == ceiling(pi)
    
    # 断言：对于给定的 y，计算其 ceiling 值后再用 frac 函数重写，然后替换 y 为 -E，应等于 ceiling(-E)
    assert ceiling(y).rewrite(frac).subs(y, -E) == ceiling(-E)
    
    # 断言：确保 ceiling(neg) 小于等于 0
    assert (ceiling(neg) <= 0) == True
    
    # 断言：确保 ceiling(neg) 小于 0 等价于 neg 小于等于 -1
    assert (ceiling(neg) < 0) == (neg <= -1)
    
    # 断言：确保 ceiling(neg) 大于 0
    assert (ceiling(neg) > 0) == False
    
    # 断言：确保 ceiling(neg) 大于等于 0 等价于 neg 大于 -1
    assert (ceiling(neg) >= 0) == (neg > -1)
    
    # 断言：确保 ceiling(neg) 大于 -3
    assert (ceiling(neg) > -3) == (neg > -3)
    
    # 断言：确保 ceiling(neg) 小于等于 10 等价于 neg 小于等于 10
    assert (ceiling(neg) <= 10) == (neg <= 10)
    
    # 确保 ceiling(nn) 小于 0 为 False
    assert (ceiling(nn) < 0) == False
    
    # 确保 ceiling(nn) 大于等于 0 为 True
    assert (ceiling(nn) >= 0) == True
    
    # 确保 ceiling(pos) 小于 0 为 False
    assert (ceiling(pos) < 0) == False
    
    # 确保 ceiling(pos) 小于等于 0 为 False
    assert (ceiling(pos) <= 0) == False
    
    # 确保 ceiling(pos) 大于 0 为 True
    assert (ceiling(pos) > 0) == True
    
    # 确保 ceiling(pos) 大于等于 0 为 True
    assert (ceiling(pos) >= 0) == True
    
    # 确保 ceiling(pos) 大于等于 1 为 True
    assert (ceiling(pos) >= 1) == True
    
    # 确保 ceiling(pos) 大于 5 等价于 pos 大于 5
    assert (ceiling(pos) > 5) == (pos > 5)
    
    # 确保 ceiling(np) 小于等于 0 为 True
    assert (ceiling(np) <= 0) == True
    
    # 确保 ceiling(np) 大于 0 为 False
    assert (ceiling(np) > 0) == False
    
    # 确保 ceiling(neg) 的 is_positive 属性为 False
    assert ceiling(neg).is_positive == False
    
    # 确保 ceiling(neg) 的 is_nonpositive 属性为 True
    assert ceiling(neg).is_nonpositive == True
    
    # 确保 ceiling(nn) 的 is_positive 属性为 None
    assert ceiling(nn).is_positive is None
    
    # 确保 ceiling(nn) 的 is_nonpositive 属性为 None
    assert ceiling(nn).is_nonpositive is None
    
    # 确保 ceiling(pos) 的 is_positive 属性为 True
    assert ceiling(pos).is_positive == True
    
    # 确保 ceiling(pos) 的 is_nonpositive 属性为 False
    assert ceiling(pos).is_nonpositive == False
    
    # 确保 ceiling(np) 的 is_positive 属性为 False
    assert ceiling(np).is_positive == False
    
    # 确保 ceiling(np) 的 is_nonpositive 属性为 True
    assert ceiling(np).is_nonpositive == True
    
    # 断言：确保 ceiling(7, evaluate=False) 大于等于 7 为 True
    assert (ceiling(7, evaluate=False) >= 7) == True
    
    # 断言：确保 ceiling(7, evaluate=False) 大于 7 为 False
    assert (ceiling(7, evaluate=False) > 7) == False
    
    # 断言：确保 ceiling(7, evaluate=False) 小于等于 7 为 True
    assert (ceiling(7, evaluate=False) <= 7) == True
    
    # 断言：确保 ceiling(7, evaluate=False) 小于 7 为 False
    assert (ceiling(7, evaluate=False) < 7) == False
    
    # 断言：确保 ceiling(7, evaluate=False) 大于等于 6 为 True
    assert (ceiling(7, evaluate=False) >= 6) == True
    
    # 断言：确保 ceiling(7, evaluate=False) 大于 6 为 True
    assert (ceiling(7, evaluate=False) > 6) == True
    
    # 断言：确保 ceiling(7, evaluate=False) 小于等于 6 为 False
    assert (ceiling(7, evaluate=False) <= 6) == False
    
    # 断言：确保 ceiling(7, evaluate=False) 小于 6 为 False
    assert (ceiling(7, evaluate=False) < 6) == False
    
    # 断言：确保 ceiling(7, evaluate=False) 大于等于 8 为 False
    assert (ceiling(7, evaluate=False) >= 8) == False
    
    # 断言：确保 ceiling(7, evaluate=False) 大于 8 为 False
    assert (ceiling(7, evaluate=False) > 8) == False
    
    # 断言：确保 ceiling(7, evaluate=False) 小于等于 8 为 True
    assert (ceiling(7, evaluate=False) <= 8) == True
    
    # 断言：确保 ceiling(7, evaluate=False) 小于 8 为 True
    assert (ceiling(7, evaluate=False) < 8) == True
    
    # 断言：确保 ceiling(x) 小于等于 5.5 等价于 Le(ceiling(x), 5.5, evaluate=False)
    assert (ceiling(x) <= 5.5) == Le(ceiling(x), 5.5, evaluate=False)
    
    # 断言：确保 ceiling(x) 大于等于 -3.2 等价于 Ge(ceiling(x), -3.2, evaluate=False)
    assert (ceiling(x) >= -3.2) == Ge(ceiling(x), -3.2, evaluate=False)
    
    # 断言：确保 ceiling(x) 小于 2.9 等价于 Lt(ceiling(x), 2.9, evaluate=False)
    assert (ceiling(x) < 2.9) == Lt(ceiling(x), 2.9, evaluate=False)
    
    # 断言：确保 ceiling(x) 大于 -1.7 等价于 Gt(ceiling(x), -1.7, evaluate=False)
    assert (ceiling(x) > -1.7) == Gt(ceiling(x), -1.7, evaluate=False)
    
    # 断言：确保 ceiling(y) 小于等于 5.5 等价于 y 小于等于 5
    assert (ceiling(y) <= 5.5) == (y <=
    # 创建一个 CRootOf 对象，表示方程 x**5 - x**2 + 1 的根之一，这里选择第一个根
    f = CRootOf(x**5 - x**2 + 1, 0)
    
    # 使用 Lambda 表达式定义一个映射，将整数 n 映射为 n + f 的结果
    s = ImageSet(Lambda(n, n + f), Integers)
    
    # 断言：s 和区间 [-10, 10] 的交集应该是集合 {i + f | i 属于 [-9, 11]}
    assert s.intersect(Interval(-10, 10)) == {i + f for i in range(-9, 11)}
# 定义一个测试函数 test_frac
def test_frac():
    # 断言 frac(x) 返回的类型是 frac 类型
    assert isinstance(frac(x), frac)
    # 断言 frac(oo) 返回 AccumBounds(0, 1)
    assert frac(oo) == AccumBounds(0, 1)
    # 断言 frac(-oo) 返回 AccumBounds(0, 1)
    assert frac(-oo) == AccumBounds(0, 1)
    # 断言 frac(zoo) 返回 NaN
    assert frac(zoo) is nan

    # 断言 frac(n) 返回 0
    assert frac(n) == 0
    # 断言 frac(nan) 返回 NaN
    assert frac(nan) is nan
    # 断言 frac(Rational(4, 3)) 返回 Rational(1, 3)
    assert frac(Rational(4, 3)) == Rational(1, 3)
    # 断言 frac(-Rational(4, 3)) 返回 Rational(2, 3)
    assert frac(-Rational(4, 3)) == Rational(2, 3)
    # 断言 frac(Rational(-4, 3)) 返回 Rational(2, 3)
    assert frac(Rational(-4, 3)) == Rational(2, 3)

    # 创建一个实数符号 r
    r = Symbol('r', real=True)
    # 断言 frac(I*r) 返回 I*frac(r)
    assert frac(I*r) == I*frac(r)
    # 断言 frac(1 + I*r) 返回 I*frac(r)
    assert frac(1 + I*r) == I*frac(r)
    # 断言 frac(0.5 + I*r) 返回 0.5 + I*frac(r)
    assert frac(0.5 + I*r) == 0.5 + I*frac(r)
    # 断言 frac(n + I*r) 返回 I*frac(r)
    assert frac(n + I*r) == I*frac(r)
    # 断言 frac(n + I*k) 返回 0
    assert frac(n + I*k) == 0
    # 断言 unchanged(frac, x + I*x)
    assert unchanged(frac, x + I*x)
    # 断言 frac(x + I*n) 返回 frac(x)
    assert frac(x + I*n) == frac(x)

    # 断言 frac(x).rewrite(floor) 返回 x - floor(x)
    assert frac(x).rewrite(floor) == x - floor(x)
    # 断言 frac(x).rewrite(ceiling) 返回 x + ceiling(-x)
    assert frac(x).rewrite(ceiling) == x + ceiling(-x)
    # 断言 frac(y).rewrite(floor).subs(y, pi) 返回 frac(pi)
    assert frac(y).rewrite(floor).subs(y, pi) == frac(pi)
    # 断言 frac(y).rewrite(floor).subs(y, -E) 返回 frac(-E)
    assert frac(y).rewrite(floor).subs(y, -E) == frac(-E)
    # 断言 frac(y).rewrite(ceiling).subs(y, -pi) 返回 frac(-pi)
    assert frac(y).rewrite(ceiling).subs(y, -pi) == frac(-pi)
    # 断言 frac(y).rewrite(ceiling).subs(y, E) 返回 frac(E)
    assert frac(y).rewrite(ceiling).subs(y, E) == frac(E)

    # 断言 frac(y) 等于 y - floor(y)
    assert Eq(frac(y), y - floor(y))
    # 断言 frac(y) 等于 y + ceiling(-y)
    assert Eq(frac(y), y + ceiling(-y))

    # 创建实数符号 r，整数符号 p_i, n_i, np_i, nn_i, p_r, n_r, np_r, nn_r
    r = Symbol('r', real=True)
    p_i = Symbol('p_i', integer=True, positive=True)
    n_i = Symbol('p_i', integer=True, negative=True)  # 这里应该是 'n_i'
    np_i = Symbol('np_i', integer=True, nonpositive=True)
    nn_i = Symbol('nn_i', integer=True, nonnegative=True)
    p_r = Symbol('p_r', positive=True)
    n_r = Symbol('n_r', negative=True)
    np_r = Symbol('np_r', real=True, nonpositive=True)
    nn_r = Symbol('nn_r', real=True, nonnegative=True)

    # Real frac argument, integer rhs
    # 断言 frac(r) <= p_i
    assert (frac(r) <= p_i).has(Le)
    # 断言 frac(r) 不小于 n_i
    assert not frac(r) <= n_i
    # 断言 frac(r) <= np_i 中含有 Le（小于等于）关系
    assert (frac(r) <= np_i).has(Le)
    # 断言 frac(r) <= nn_i 中含有 Le（小于等于）关系
    assert (frac(r) <= nn_i).has(Le)
    # 断言 frac(r) < p_i
    assert (frac(r) < p_i).has(Lt)
    # 断言 frac(r) 不小于 n_i
    assert not frac(r) < n_i
    # 断言 frac(r) 不小于 np_i
    assert not frac(r) < np_i
    # 断言 frac(r) < nn_i 中含有 Lt（小于）关系
    assert (frac(r) < nn_i).has(Lt)
    # 断言 frac(r) 不大于 p_i
    assert not frac(r) >= p_i
    # 断言 frac(r) 大于等于 n_i
    assert frac(r) >= n_i
    # 断言 frac(r) 大于等于 np_i
    assert frac(r) >= np_i
    # 断言 frac(r) >= nn_i 中含有 Ge（大于等于）关系
    assert (frac(r) >= nn_i).has(Ge)
    # 断言 frac(r) 不大于 p_i
    assert not frac(r) > p_i
    # 断言 frac(r) 大于 n_i
    assert frac(r) > n_i
    # 断言 frac(r) > np_i 中含有 Gt（大于）关系
    assert (frac(r) > np_i).has(Gt)
    # 断言 frac(r) > nn_i 中含有 Gt（大于）关系
    assert (frac(r) > nn_i).has(Gt)

    # 断言 frac(r) 不等于 p_i
    assert not Eq(frac(r), p_i)
    # 断言 frac(r) 不等于 n_i
    assert not Eq(frac(r), n_i)
    # 断言 frac(r) 等于 np_i 中含有 Eq（等于）关系
    assert Eq(frac(r), np_i).has(Eq)
    # 断言 frac(r) 等于 nn_i 中含有 Eq（等于）关系
    assert Eq(frac(r), nn_i).has(Eq)

    # 断言 frac(r) 不等于 p_i
    assert Ne(frac(r), p_i)
    # 断言 frac(r) 不等于 n_i
    assert Ne(frac(r), n_i)
    # 断言 frac(r) 不等于 np_i 中含有 Ne（不等于）关系
    assert Ne(frac(r), np_i).has(Ne)
    # 断言 frac(r) 不等于 nn_i 中含有 Ne（不等于）关系
    assert Ne(frac(r), nn_i).has(Ne)

    # Real frac argument, real rhs
    # 断言 frac(r) <= p_r 中含有 Le（小于等于）关系
    assert (frac(r) <= p_r).has(Le)
    # 断言 frac(r) 不小于 n_r
    assert not frac(r) <= n_r
    # 断言 frac(r) <= np_r 中含有 Le（小于等于）关系
    assert (frac(r) <= np_r).has(Le)
    # 断言 frac(r) <= nn_r 中含有 Le（小于等于）关系
    assert (frac(r) <= nn_r).has(Le)
    # 断言 frac(r) < p_r 中含有 Lt（小于）关系
    assert (frac(r) < p_r).has(Lt)
    # 断言 frac(r) 不小于 n_r
    assert not frac(r) < n_r
    # 断言 frac(r) 不小于 np_r
    assert not frac(r) < np_r
    # 断言 frac(r) < nn_r 中含有 Lt（小于）关系
    assert (frac(r) < nn_r).has(Lt)
    # 断言 frac(r) >= p_r 中含有 Ge（大于等于）关系
    assert (frac(r) >= p_r).has(Ge)
    # 断言 frac(r)
    # 断言分数 r 的分子与 nn_r 相等
    assert Eq(frac(r), nn_r).has(Eq)

    # 断言分数 r 的分子与 p_r 不相等
    assert Ne(frac(r), p_r).has(Ne)
    # 断言分数 r 的分子与 n_r 不相等
    assert Ne(frac(r), n_r)
    # 断言分数 r 的分子与 np_r 不相等
    assert Ne(frac(r), np_r).has(Ne)
    # 断言分数 r 的分子与 nn_r 不相等
    assert Ne(frac(r), nn_r).has(Ne)

    # 断言分数 r 的值小于正无穷
    assert frac(r) < oo
    # 断言分数 r 的值小于等于正无穷
    assert frac(r) <= oo
    # 断言分数 r 的值不大于正无穷
    assert not frac(r) > oo
    # 断言分数 r 的值不大于等于正无穷
    assert not frac(r) >= oo

    # 断言分数 r 的值不小于负无穷
    assert not frac(r) < -oo
    # 断言分数 r 的值不小于等于负无穷
    assert not frac(r) <= -oo
    # 断言分数 r 的值大于负无穷
    assert frac(r) > -oo
    # 断言分数 r 的值大于等于负无穷
    assert frac(r) >= -oo

    # 断言分数 r 的值小于 1
    assert frac(r) < 1
    # 断言分数 r 的值小于等于 1
    assert frac(r) <= 1
    # 断言分数 r 的值不大于 1
    assert not frac(r) > 1
    # 断言分数 r 的值不大于等于 1
    assert not frac(r) >= 1

    # 断言分数 r 的值不小于 0
    assert not frac(r) < 0
    # 断言分数 r 的值小于等于 0
    assert (frac(r) <= 0).has(Le)
    # 断言分数 r 的值大于 0
    assert (frac(r) > 0).has(Gt)
    # 断言分数 r 的值大于等于 0
    assert frac(r) >= 0

    # 一些关于数值的测试
    # 断言分数 r 的值小于等于 sqrt(2)
    assert frac(r) <= sqrt(2)
    # 断言分数 r 的值小于等于 sqrt(3) - sqrt(2)
    assert (frac(r) <= sqrt(3) - sqrt(2)).has(Le)
    # 断言分数 r 的值不小于等于 sqrt(2) - sqrt(3)
    assert not frac(r) <= sqrt(2) - sqrt(3)
    # 断言分数 r 的值不大于等于 sqrt(2)
    assert not frac(r) >= sqrt(2)
    # 断言分数 r 的值大于等于 sqrt(3) - sqrt(2)
    assert (frac(r) >= sqrt(3) - sqrt(2)).has(Ge)
    # 断言分数 r 的值大于等于 sqrt(2) - sqrt(3)
    assert frac(r) >= sqrt(2) - sqrt(3)

    # 断言分数 r 的值不等于 sqrt(2)
    assert not Eq(frac(r), sqrt(2))
    # 断言分数 r 的值等于 sqrt(3) - sqrt(2)
    assert Eq(frac(r), sqrt(3) - sqrt(2)).has(Eq)
    # 断言分数 r 的值不等于 sqrt(2) - sqrt(3)
    assert not Eq(frac(r), sqrt(2) - sqrt(3))
    # 断言分数 r 的值不等于 sqrt(2)
    assert Ne(frac(r), sqrt(2))
    # 断言分数 r 的值不等于 sqrt(3) - sqrt(2)
    assert Ne(frac(r), sqrt(3) - sqrt(2)).has(Ne)
    # 断言分数 r 的值不等于 sqrt(2) - sqrt(3)
    assert Ne(frac(r), sqrt(2) - sqrt(3))

    # 断言分数 p_i 的值为零（evaluate=False 表示不进行数值计算）
    assert frac(p_i, evaluate=False).is_zero
    # 断言分数 p_i 的值有限（evaluate=False 表示不进行数值计算）
    assert frac(p_i, evaluate=False).is_finite
    # 断言分数 p_i 的值为整数（evaluate=False 表示不进行数值计算）
    assert frac(p_i, evaluate=False).is_integer
    # 断言分数 p_i 的值为实数（evaluate=False 表示不进行数值计算）
    assert frac(p_i, evaluate=False).is_real
    # 断言分数 r 的值为有限数
    assert frac(r).is_finite
    # 断言分数 r 的值为实数
    assert frac(r).is_real
    # 断言分数 r 的值不为零
    assert frac(r).is_zero is None
    # 断言分数 r 的值不为整数
    assert frac(r).is_integer is None

    # 断言分数 oo（正无穷）的值为有限数
    assert frac(oo).is_finite
    # 断言分数 oo（正无穷）的值为实数
    assert frac(oo).is_real
def test_series():
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 断言：floor(x) 的数值级数展开结果，应该等于 floor(y)
    assert floor(x).nseries(x, y, 100) == floor(y)
    # 断言：ceiling(x) 的数值级数展开结果，应该等于 ceiling(y)
    assert ceiling(x).nseries(x, y, 100) == ceiling(y)
    # 断言：floor(x) 的数值级数展开结果，应该等于 3
    assert floor(x).nseries(x, pi, 100) == 3
    # 断言：ceiling(x) 的数值级数展开结果，应该等于 4
    assert ceiling(x).nseries(x, pi, 100) == 4
    # 断言：floor(x) 的数值级数展开结果，应该等于 0
    assert floor(x).nseries(x, 0, 100) == 0
    # 断言：ceiling(x) 的数值级数展开结果，应该等于 1
    assert ceiling(x).nseries(x, 0, 100) == 1
    # 断言：floor(-x) 的数值级数展开结果，应该等于 -1
    assert floor(-x).nseries(x, 0, 100) == -1
    # 断言：ceiling(-x) 的数值级数展开结果，应该等于 0
    assert ceiling(-x).nseries(x, 0, 100) == 0


def test_issue_14355():
    # 这些测试检查当 arg0 为 S.NaN 时，floor 和 ceil 函数的主导项和级数。
    assert floor((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir=1) == -2
    assert floor((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir=-1) == -1
    assert floor((cos(x) - 1)/x).as_leading_term(x, cdir=1) == -1
    assert floor((cos(x) - 1)/x).as_leading_term(x, cdir=-1) == 0
    assert floor(sin(x)/x).as_leading_term(x, cdir=1) == 0
    assert floor(sin(x)/x).as_leading_term(x, cdir=-1) == 0
    assert floor(-tan(x)/x).as_leading_term(x, cdir=1) == -2
    assert floor(-tan(x)/x).as_leading_term(x, cdir=-1) == -2
    assert floor(sin(x)/x/3).as_leading_term(x, cdir=1) == 0
    assert floor(sin(x)/x/3).as_leading_term(x, cdir=-1) == 0
    assert ceiling((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir=1) == -1
    assert ceiling((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir=-1) == 0
    assert ceiling((cos(x) - 1)/x).as_leading_term(x, cdir=1) == 0
    assert ceiling((cos(x) - 1)/x).as_leading_term(x, cdir=-1) == 1
    assert ceiling(sin(x)/x).as_leading_term(x, cdir=1) == 1
    assert ceiling(sin(x)/x).as_leading_term(x, cdir=-1) == 1
    assert ceiling(-tan(x)/x).as_leading_term(x, cdir=1) == -1
    assert ceiling(-tan(x)/x).as_leading_term(x, cdir=-1) == -1
    assert ceiling(sin(x)/x/3).as_leading_term(x, cdir=1) == 1
    assert ceiling(sin(x)/x/3).as_leading_term(x, cdir=-1) == 1
    # 测试级数展开结果
    assert floor(sin(x)/x).series(x, 0, 100, cdir=1) == 0
    assert floor(sin(x)/x).series(x, 0, 100, cdir=-1) == 0
    assert floor((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir=1) == -2
    assert floor((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir=-1) == -1
    assert ceiling(sin(x)/x).series(x, 0, 100, cdir=1) == 1
    assert ceiling(sin(x)/x).series(x, 0, 100, cdir=-1) == 1
    assert ceiling((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir=1) == -1
    assert ceiling((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir=-1) == 0


def test_frac_leading_term():
    # 断言：frac(x) 的主导项应该是 x
    assert frac(x).as_leading_term(x) == x
    # 断言：frac(x) 的主导项应该是 x，从正方向 (cdir = 1)
    assert frac(x).as_leading_term(x, cdir=1) == x
    # 断言：frac(x) 的主导项应该是 1，从负方向 (cdir = -1)
    assert frac(x).as_leading_term(x, cdir=-1) == 1
    # 断言：frac(x + S.Half) 的主导项应该是 S.Half，从正方向 (cdir = 1)
    assert frac(x + S.Half).as_leading_term(x, cdir=1) == S.Half
    # 断言：frac(x + S.Half) 的主导项应该是 S.Half，从负方向 (cdir = -1)
    assert frac(x + S.Half).as_leading_term(x, cdir=-1) == S.Half
    # 断言：frac(-2*x + 1) 的主导项应该是 1，从正方向 (cdir = 1)
    assert frac(-2*x + 1).as_leading_term(x, cdir=1) == S.One
    # 断言：frac(-2*x + 1) 的主导项应该是 -2*x，从负方向 (cdir = -1)
    assert frac(-2*x + 1).as_leading_term(x, cdir=-1) == -2*x
    # 断言：frac(sin(x) + 5) 的主导项应该是 x，从正方向 (cdir = 1)
    assert frac(sin(x) + 5).as_leading_term(x, cdir=1) == x
    # 断言：对表达式 frac(sin(x) + 5) 的主导项进行检查，期望结果为常数 1
    assert frac(sin(x) + 5).as_leading_term(x, cdir=-1) == S.One
    
    # 断言：对表达式 frac(sin(x**2) + 5) 的主导项进行检查，期望结果为 x**2，在 x 正向的情况下
    assert frac(sin(x**2) + 5).as_leading_term(x, cdir=1) == x**2
    
    # 断言：对表达式 frac(sin(x**2) + 5) 的主导项进行检查，期望结果为 x**2，在 x 负向的情况下
    assert frac(sin(x**2) + 5).as_leading_term(x, cdir=-1) == x**2
`
@XFAIL
# 标记此测试函数为“预期失败”，即测试在当前条件下应该失败
def test_issue_4149():
    # 断言：对表达式进行向下取整操作，验证结果是否符合预期
    assert floor(3 + pi*I + y*I) == 3 + floor(pi + y)*I
    # 断言：对表达式进行向下取整操作，验证结果是否符合预期
    assert floor(3*I + pi*I + y*I) == floor(3 + pi + y)*I
    # 断言：对表达式进行向下取整操作，验证结果是否符合预期
    assert floor(3 + E + pi*I + y*I) == 5 + floor(pi + y)*I


def test_issue_21651():
    # 创建一个正整数符号 k
    k = Symbol('k', positive=True, integer=True)
    # 计算指数表达式
    exp = 2*2**(-k)
    # 断言：对表达式进行向下取整操作，验证结果是否为向下取整对象
    assert isinstance(floor(exp), floor)


def test_issue_11207():
    # 断言：对 x 进行两次向下取整操作，验证结果是否相同
    assert floor(floor(x)) == floor(x)
    # 断言：对 x 进行向上取整后再向下取整操作，验证结果是否相同
    assert floor(ceiling(x)) == ceiling(x)
    # 断言：对 x 进行向下取整后再向上取整操作，验证结果是否相同
    assert ceiling(floor(x)) == floor(x)
    # 断言：对 x 进行两次向上取整操作，验证结果是否相同
    assert ceiling(ceiling(x)) == ceiling(x)


def test_nested_floor_ceiling():
    # 断言：对表达式进行嵌套的向下取整和向上取整操作，验证结果是否符合预期
    assert floor(-floor(ceiling(x**3)/y)) == -floor(ceiling(x**3)/y)
    # 断言：对表达式进行嵌套的向上取整和向下取整操作，验证结果是否符合预期
    assert ceiling(-floor(ceiling(x**3)/y)) == -floor(ceiling(x**3)/y)
    # 断言：对表达式进行向上取整和向下取整的嵌套操作，验证结果是否符合预期
    assert floor(ceiling(-floor(x**Rational(7, 2)/y))) == -floor(x**Rational(7, 2)/y)
    # 断言：对表达式进行向下取整后取相反数再向上取整操作，验证结果是否符合预期
    assert -ceiling(-ceiling(floor(x)/y)) == ceiling(floor(x)/y)


def test_issue_18689():
    # 断言：对 x 进行三次向下取整操作后加 3，验证结果是否相同
    assert floor(floor(floor(x)) + 3) == floor(x) + 3
    # 断言：对 x 进行三次向上取整操作后加 1，验证结果是否相同
    assert ceiling(ceiling(ceiling(x)) + 1) == ceiling(x) + 1
    # 断言：对 x 进行向上取整后再向上取整操作后加 3，验证结果是否相同
    assert ceiling(ceiling(floor(x)) + 3) == floor(x) + 3


def test_issue_18421():
    # 断言：对浮点数 0 进行向下取整操作，验证结果是否为零
    assert floor(float(0)) is S.Zero
    # 断言：对浮点数 0 进行向上取整操作，验证结果是否为零
    assert ceiling(float(0)) is S.Zero


def test_issue_25230():
    # 创建一个实数符号 a
    a = Symbol('a', real=True)
    # 创建一个正数符号 b
    b = Symbol('b', positive=True)
    # 创建一个负数符号 c
    c = Symbol('c', negative=True)
    # 抛出未实现错误，测试对 x/a 进cdir=1) == 1
    assert ceiling(x/b).as_leading_term(x, cdir=-1) == 0
    assert ceiling(x/c).as_leading_term(x, cdir=1) == 0
    assert ceiling(x/c).as_leading_term(x, cdir=-1) == 1
```