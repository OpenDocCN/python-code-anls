# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\test_miscellaneous.py`

```
import itertools as it  # 导入 itertools 库，并用别名 it 指代

from sympy.core.expr import unchanged  # 从 sympy.core.expr 模块导入 unchanged 符号
from sympy.core.function import Function  # 从 sympy.core.function 模块导入 Function 类
from sympy.core.numbers import I, oo, Rational  # 从 sympy.core.numbers 模块导入 I, oo, Rational 符号
from sympy.core.power import Pow  # 从 sympy.core.power 模块导入 Pow 类
from sympy.core.singleton import S  # 从 sympy.core.singleton 模块导入 S 符号
from sympy.core.symbol import Symbol  # 从 sympy.core.symbol 模块导入 Symbol 类
from sympy.external import import_module  # 从 sympy.external 模块导入 import_module 函数
from sympy.functions.elementary.exponential import log  # 从 sympy.functions.elementary.exponential 模块导入 log 函数
from sympy.functions.elementary.integers import floor, ceiling  # 从 sympy.functions.elementary.integers 模块导入 floor, ceiling 函数
from sympy.functions.elementary.miscellaneous import (sqrt, cbrt, root, Min,  # 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt, cbrt, root, Min,
                                                      Max, real_root, Rem)  # Max, real_root, Rem 函数
from sympy.functions.elementary.trigonometric import cos, sin  # 从 sympy.functions.elementary.trigonometric 模块导入 cos, sin 函数
from sympy.functions.special.delta_functions import Heaviside  # 从 sympy.functions.special.delta_functions 模块导入 Heaviside 函数

from sympy.utilities.lambdify import lambdify  # 从 sympy.utilities.lambdify 模块导入 lambdify 函数
from sympy.testing.pytest import raises, skip, ignore_warnings  # 从 sympy.testing.pytest 模块导入 raises, skip, ignore_warnings 函数

def test_Min():
    from sympy.abc import x, y, z  # 从 sympy.abc 模块导入 x, y, z 符号
    n = Symbol('n', negative=True)  # 创建一个符号 n，指定为负数
    n_ = Symbol('n_', negative=True)  # 创建一个符号 n_，指定为负数
    nn = Symbol('nn', nonnegative=True)  # 创建一个符号 nn，指定为非负数
    nn_ = Symbol('nn_', nonnegative=True)  # 创建一个符号 nn_，指定为非负数
    p = Symbol('p', positive=True)  # 创建一个符号 p，指定为正数
    p_ = Symbol('p_', positive=True)  # 创建一个符号 p_，指定为正数
    np = Symbol('np', nonpositive=True)  # 创建一个符号 np，指定为非正数
    np_ = Symbol('np_', nonpositive=True)  # 创建一个符号 np_，指定为非正数
    r = Symbol('r', real=True)  # 创建一个符号 r，指定为实数

    assert Min(5, 4) == 4  # 断言：5 和 4 中的最小值为 4
    assert Min(-oo, -oo) is -oo  # 断言：负无穷和负无穷的最小值为负无穷
    assert Min(-oo, n) is -oo  # 断言：负无穷和符号 n 的最小值为负无穷
    assert Min(n, -oo) is -oo  # 断言：符号 n 和负无穷的最小值为负无穷
    assert Min(-oo, np) is -oo  # 断言：负无穷和符号 np 的最小值为负无穷
    assert Min(np, -oo) is -oo  # 断言：符号 np 和负无穷的最小值为负无穷
    assert Min(-oo, 0) is -oo  # 断言：负无穷和0的最小值为负无穷
    assert Min(0, -oo) is -oo  # 断言：0和负无穷的最小值为负无穷
    assert Min(-oo, nn) is -oo  # 断言：负无穷和符号 nn 的最小值为负无穷
    assert Min(nn, -oo) is -oo  # 断言：符号 nn 和负无穷的最小值为负无穷
    assert Min(-oo, p) is -oo  # 断言：负无穷和符号 p 的最小值为负无穷
    assert Min(p, -oo) is -oo  # 断言：符号 p 和负无穷的最小值为负无穷
    assert Min(-oo, oo) is -oo  # 断言：负无穷和正无穷的最小值为负无穷
    assert Min(oo, -oo) is -oo  # 断言：正无穷和负无穷的最小值为负无穷
    assert Min(n, n) == n  # 断言：符号 n 和符号 n 的最小值为符号 n 本身
    assert unchanged(Min, n, np)  # 断言：Min 函数应该保持符号 n 和符号 np 不变
    assert Min(np, n) == Min(n, np)  # 断言：符号 np 和符号 n 的最小值等于符号 n 和符号 np 的最小值
    assert Min(n, 0) == n  # 断言：符号 n 和0的最小值为符号 n
    assert Min(0, n) == n  # 断言：0和符号 n 的最小值为符号 n
    assert Min(n, nn) == n  # 断言：符号 n 和符号 nn 的最小值为符号 n
    assert Min(nn, n) == n  # 断言：符号 nn 和符号 n 的最小值为符号 n
    assert Min(n, p) == n  # 断言：符号 n 和符号 p 的最小值为符号 n
    assert Min(p, n) == n  # 断言：符号 p 和符号 n 的最小值为符号 n
    assert Min(n, oo) == n  # 断言：符号 n 和正无穷的最小值为符号 n
    assert Min(oo, n) == n  # 断言：正无穷和符号 n 的最小值为符号 n
    assert Min(np, np) == np  # 断言：符号 np 和符号 np 的最小值为符号 np
    assert Min(np, 0) == np  # 断言：符号 np 和0的最小值为符号 np
    assert Min(0, np) == np  # 断言：0和符号 np 的最小值为符号 np
    assert Min(np, nn) == np  # 断言：符号 np 和符号 nn 的最小值为符号 np
    assert Min(nn, np) == np  # 断言：符号 nn 和符号 np 的最小值为符号 np
    assert Min(np, p) == np  # 断言：符号 np 和符号 p 的最小值为符号 np
    assert Min(p, np) == np  # 断言：符号 p 和符号 np 的最小值为符号 np
    assert Min(np, oo) == np  # 断言：符号 np 和正无穷的最小值为符号 np
    assert Min(oo, np) == np  # 断言：正无穷和符号 np 的最小值为符号 np
    assert Min(0, 0) == 0  # 断言：0和0的最小值为0
    assert Min(0, nn) == 0  # 断言：0和符号 nn 的最小值为0
    assert Min(nn, 0) == 0  # 断言：符号 nn 和0的最小值为0
    assert Min(0, p) == 0  # 断言：0和符号 p 的最小值为0
    assert Min(p, 0) == 0  # 断言：符
    # 断言语句：确保 Min 函数的行为符合预期
    assert Min(x, Max(y, -oo)) == Min(x, y)
    # 断言语句：确保 Min 函数的行为符合预期
    assert Min(p, oo, n, p, p, p_) == n
    # 断言语句：确保 Min 函数的行为符合预期
    assert Min(p_, n_, p) == n_
    # 断言语句：确保 Min 函数的行为符合预期
    assert Min(n, oo, -7, p, p, 2) == Min(n, -7)
    # 断言语句：确保 Min 函数的行为符合预期
    assert Min(2, x, p, n, oo, n_, p, 2, -2, -2) == Min(-2, x, n, n_)
    # 断言语句：确保 Min 函数的行为符合预期
    assert Min(0, x, 1, y) == Min(0, x, y)
    # 断言语句：确保 Min 函数的行为符合预期
    assert Min(1000, 100, -100, x, p, n) == Min(n, x, -100)
    # 断言语句：确保 unchanged 函数对 Min 函数的调用不改变其行为
    assert unchanged(Min, sin(x), cos(x))
    # 断言语句：确保 Min 函数的对称性质成立
    assert Min(sin(x), cos(x)) == Min(cos(x), sin(x))
    # 断言语句：确保 Min 函数在特定点上的求值结果符合预期
    assert Min(cos(x), sin(x)).subs(x, 1) == cos(1)
    # 断言语句：确保 Min 函数在特定点上的求值结果符合预期
    assert Min(cos(x), sin(x)).subs(x, S.Half) == sin(S.Half)
    # 断言语句：验证 Min 函数在复数域上的行为，预期会引发 ValueError
    raises(ValueError, lambda: Min(cos(x), sin(x)).subs(x, I))
    # 断言语句：验证 Min 函数在给定参数下会引发 ValueError
    raises(ValueError, lambda: Min(I))
    # 断言语句：验证 Min 函数在给定参数下会引发 ValueError
    raises(ValueError, lambda: Min(I, x))
    # 断言语句：验证 Min 函数在给定参数下会引发 ValueError
    raises(ValueError, lambda: Min(S.ComplexInfinity, x))

    # 断言语句：验证 Min 函数对 x 的导数计算结果符合预期
    assert Min(1, x).diff(x) == Heaviside(1 - x)
    # 断言语句：验证 Min 函数对 x 的导数计算结果符合预期
    assert Min(x, 1).diff(x) == Heaviside(1 - x)
    # 断言语句：验证 Min 函数对 x 的导数计算结果符合预期
    assert Min(0, -x, 1 - 2*x).diff(x) == -Heaviside(x + Min(0, -2*x + 1)) \
        - 2*Heaviside(2*x + Min(0, -x) - 1)

    # 问题 7619 的测试
    f = Function('f')
    assert Min(1, 2*Min(f(1), 2))  # doesn't fail

    # 问题 7233 的测试
    e = Min(0, x)
    assert e.n().args == (0, x)

    # 问题 8643 的测试
    m = Min(n, p_, n_, r)
    assert m.is_positive is False
    assert m.is_nonnegative is False
    assert m.is_negative is True

    m = Min(p, p_)
    assert m.is_positive is True
    assert m.is_nonnegative is True
    assert m.is_negative is False

    m = Min(p, nn_, p_)
    assert m.is_positive is None
    assert m.is_nonnegative is True
    assert m.is_negative is False

    m = Min(nn, p, r)
    assert m.is_positive is None
    assert m.is_nonnegative is None
    assert m.is_negative is None
# 定义一个测试函数 test_Max，用于测试 Max 函数的各种用例
def test_Max():
    # 从 sympy.abc 模块导入符号 x, y, z
    from sympy.abc import x, y, z
    # 创建一个符号 n，其取值为负数
    n = Symbol('n', negative=True)
    # 创建一个符号 n_，其取值为负数
    n_ = Symbol('n_', negative=True)
    # 创建一个符号 nn，其取值为非负数
    nn = Symbol('nn', nonnegative=True)
    # 创建一个符号 p，其取值为正数
    p = Symbol('p', positive=True)
    # 创建一个符号 p_，其取值为正数
    p_ = Symbol('p_', positive=True)
    # 创建一个符号 r，其取值为实数
    r = Symbol('r', real=True)

    # 断言：比较 5 和 4 的最大值，应该返回 5
    assert Max(5, 4) == 5

    # lists

    # 断言：没有给定参数时，Max 函数应返回负无穷大
    assert Max() is S.NegativeInfinity
    # 断言：当只有一个参数 x 时，Max(x) 应返回 x
    assert Max(x) == x
    # 断言：Max(x, y) 应等同于 Max(y, x)
    assert Max(x, y) == Max(y, x)
    # 断言：Max(x, y, z) 应等同于 Max(z, y, x)
    assert Max(x, y, z) == Max(z, y, x)
    # 断言：Max(x, Max(y, z)) 应等同于 Max(z, y, x)
    assert Max(x, Max(y, z)) == Max(z, y, x)
    # 断言：Max(x, Min(y, oo)) 应等同于 Max(x, y)
    assert Max(x, Min(y, oo)) == Max(x, y)
    # 断言：Max(n, -oo, n_, p, 2) 应等同于 Max(p, 2)
    assert Max(n, -oo, n_, p, 2) == Max(p, 2)
    # 断言：Max(n, -oo, n_, p) 应等同于 p
    assert Max(n, -oo, n_, p) == p
    # 断言：Max(2, x, p, n, -oo, S.NegativeInfinity, n_, p, 2) 应等同于 Max(2, x, p)
    assert Max(2, x, p, n, -oo, S.NegativeInfinity, n_, p, 2) == Max(2, x, p)
    # 断言：Max(0, x, 1, y) 应等同于 Max(1, x, y)
    assert Max(0, x, 1, y) == Max(1, x, y)
    # 断言：Max(r, r + 1, r - 1) 应等同于 1 + r
    assert Max(r, r + 1, r - 1) == 1 + r
    # 断言：Max(1000, 100, -100, x, p, n) 应等同于 Max(p, x, 1000)
    assert Max(1000, 100, -100, x, p, n) == Max(p, x, 1000)
    # 断言：Max(cos(x), sin(x)) 应等同于 Max(sin(x), cos(x))
    assert Max(cos(x), sin(x)) == Max(sin(x), cos(x))
    # 断言：对 Max(cos(x), sin(x)) 在 x = 1 处进行符号替换应返回 sin(1)
    assert Max(cos(x), sin(x)).subs(x, 1) == sin(1)
    # 断言：对 Max(cos(x), sin(x)) 在 x = S.Half 处进行符号替换应返回 cos(S.Half)
    assert Max(cos(x), sin(x)).subs(x, S.Half) == cos(S.Half)
    # 断言：对于包含虚数的 Max(cos(x), sin(x))，应引发 ValueError 异常
    raises(ValueError, lambda: Max(cos(x), sin(x)).subs(x, I))
    # 断言：对于包含虚数的 Max(I)，应引发 ValueError 异常
    raises(ValueError, lambda: Max(I))
    # 断言：对于包含虚数的 Max(I, x)，应引发 ValueError 异常
    raises(ValueError, lambda: Max(I, x))
    # 断言：对于包含复无穷大的 Max(S.ComplexInfinity, 1)，应引发 ValueError 异常
    raises(ValueError, lambda: Max(S.ComplexInfinity, 1))
    # 断言：Max(n, -oo, n_,  p, 2) 应等同于 Max(p, 2)
    assert Max(n, -oo, n_,  p, 2) == Max(p, 2)
    # 断言：Max(n, -oo, n_,  p, 1000) 应等同于 Max(p, 1000)
    assert Max(n, -oo, n_,  p, 1000) == Max(p, 1000)

    # 断言：对 Max(1, x) 关于 x 求导数应返回 Heaviside(x - 1)
    assert Max(1, x).diff(x) == Heaviside(x - 1)
    # 断言：对 Max(x, 1) 关于 x 求导数应返回 Heaviside(x - 1)
    assert Max(x, 1).diff(x) == Heaviside(x - 1)
    # 断言：对 Max(x**2, 1 + x, 1) 关于 x 求导数应返回复杂的表达式
    assert Max(x**2, 1 + x, 1).diff(x) == \
        2*x*Heaviside(x**2 - Max(1, x + 1)) \
        + Heaviside(x - Max(1, x**2) + 1)

    # e = Max(0, x) 的数值化后应返回 (0, x)
    e = Max(0, x)
    assert e.n().args == (0, x)

    # issue 8643
    # 对于符号 m = Max(p, p_, n, r)，应该满足其数学属性
    m = Max(p, p_, n, r)
    assert m.is_positive is True
    assert m.is_nonnegative is True
    assert m.is_negative is False

    # 对于符号 m = Max(n, n_)，应该满足其数学属性
    m = Max(n, n_)
    assert m.is_positive is False
    assert m.is_nonnegative is False
    assert m.is_negative is True

    # 对于符号 m = Max(n, n_, r)，其数学属性应为不确定
    m = Max(n, n_, r)
    assert m.is_positive is None
    assert m.is_nonnegative is None
    assert m.is_negative is None

    # 对于符号 m = Max(n, nn, r)，其数学属性应为不确定
    m = Max(n, nn, r)
    assert m.is_positive is None
    assert m.is_nonnegative is True
    assert m.is_negative is False


# 定义一个测试函数 test_minmax_assumptions，测试 Min 和 Max 函数的数学假设
def test_minmax_assumptions():
    # 定义一些带有数学属性的符号
    r = Symbol('r', real=True)
    a = Symbol('a', real=True, algebraic=True)
    t = Symbol('t', real=True, transcendental=True)
    q = Symbol('q', rational=True)
    p = Symbol('p', irrational=True)
    n = Symbol('n', rational=True, integer=False)
    i = Symbol('i', integer=True)
    o = Symbol('o', odd=True)
    e = Symbol('e', even=True)
    k = Symbol('k', prime=True)
    # 将这些符号放入列表 reals 中
    reals = [r, a, t, q, p, n, i, o, e, k]
    # 对于每个扩展函数 (Max 或 Min)，遍历实数集合的笛卡尔积
    for ext in (Max, Min):
        for x, y in it.product(reals, repeat=2):

            # 断言结果必须是实数
            assert ext(x, y).is_real

            # 检查是否都是代数数？
            if x.is_algebraic and y.is_algebraic:
                # 断言结果也应该是代数数
                assert ext(x, y).is_algebraic
            elif x.is_transcendental and y.is_transcendental:
                # 断言结果应该是超越数
                assert ext(x, y).is_transcendental
            else:
                # 否则结果应该是 None
                assert ext(x, y).is_algebraic is None

            # 检查是否都是有理数？
            if x.is_rational and y.is_rational:
                # 断言结果应该是有理数
                assert ext(x, y).is_rational
            elif x.is_irrational and y.is_irrational:
                # 断言结果应该是无理数
                assert ext(x, y).is_irrational
            else:
                # 否则结果应该是 None
                assert ext(x, y).is_rational is None

            # 检查是否都是整数？
            if x.is_integer and y.is_integer:
                # 断言结果应该是整数
                assert ext(x, y).is_integer
            elif x.is_noninteger and y.is_noninteger:
                # 断言结果应该是非整数
                assert ext(x, y).is_noninteger
            else:
                # 否则结果应该是 None
                assert ext(x, y).is_integer is None

            # 检查是否都是奇数？
            if x.is_odd and y.is_odd:
                # 断言结果应该是奇数
                assert ext(x, y).is_odd
            elif x.is_odd is False and y.is_odd is False:
                # 断言结果应该是偶数
                assert ext(x, y).is_odd is False
            else:
                # 否则结果应该是 None
                assert ext(x, y).is_odd is None

            # 检查是否都是偶数？
            if x.is_even and y.is_even:
                # 断言结果应该是偶数
                assert ext(x, y).is_even
            elif x.is_even is False and y.is_even is False:
                # 断言结果应该是奇数
                assert ext(x, y).is_even is False
            else:
                # 否则结果应该是 None
                assert ext(x, y).is_even is None

            # 检查是否都是素数？
            if x.is_prime and y.is_prime:
                # 断言结果应该是素数
                assert ext(x, y).is_prime
            elif x.is_prime is False and y.is_prime is False:
                # 断言结果应该不是素数
                assert ext(x, y).is_prime is False
            else:
                # 否则结果应该是 None
                assert ext(x, y).is_prime is None
def test_issue_8413():
    # 创建一个实数符号 x
    x = Symbol('x', real=True)
    
    # 检查 Min 函数在特定条件下的行为
    assert Min(floor(x), x) == floor(x)
    
    # 检查 Min 函数在特定条件下的行为
    assert Min(ceiling(x), x) == x
    
    # 检查 Max 函数在特定条件下的行为
    assert Max(floor(x), x) == x
    
    # 检查 Max 函数在特定条件下的行为
    assert Max(ceiling(x), x) == ceiling(x)


def test_root():
    # 从 sympy.abc 中导入符号 x
    from sympy.abc import x
    
    # 创建一个整数符号 n 和 k
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True)
    
    # 根据给定参数计算平方根
    assert root(2, 2) == sqrt(2)
    
    # 根据给定参数计算立方根
    assert root(2, 1) == 2
    
    # 根据给定参数计算立方根
    assert root(2, 3) == 2**Rational(1, 3)
    
    # 根据给定参数计算立方根
    assert root(2, 3) == cbrt(2)
    
    # 根据给定参数计算负指数根
    assert root(2, -5) == 2**Rational(4, 5)/2
    
    # 根据给定参数计算平方根
    assert root(-2, 1) == -2
    
    # 根据给定参数计算平方根
    assert root(-2, 2) == sqrt(2)*I
    
    # 根据给定参数计算平方根
    assert root(-2, 1) == -2
    
    # 根据给定参数计算 x 的平方根
    assert root(x, 2) == sqrt(x)
    
    # 根据给定参数计算 x 的一次根
    assert root(x, 1) == x
    
    # 根据给定参数计算 x 的立方根
    assert root(x, 3) == x**Rational(1, 3)
    
    # 根据给定参数计算 x 的立方根
    assert root(x, 3) == cbrt(x)
    
    # 根据给定参数计算 x 的负指数根
    assert root(x, -5) == x**Rational(-1, 5)
    
    # 根据给定参数计算 x 的 n 次根
    assert root(x, n) == x**(1/n)
    
    # 根据给定参数计算 x 的负 n 次根
    assert root(x, -n) == x**(-1/n)
    
    # 根据给定参数计算 x 的 n 次根，带 k 偏移量
    assert root(x, n, k) == (-1)**(2*k/n)*x**(1/n)


def test_real_root():
    # 检查对于 -8 的立方根的实数解
    assert real_root(-8, 3) == -2
    
    # 检查对于 -16 的四次根的实数解
    assert real_root(-16, 4) == root(-16, 4)
    
    # 计算 -7 的四次根，并检查其实数解
    r = root(-7, 4)
    assert real_root(r) == r
    
    # 计算 -1 的立方根，并使用其平方值和四次根计算其实数解
    r1 = root(-1, 3)
    r2 = r1**2
    r3 = root(-1, 4)
    assert real_root(r1 + r2 + r3) == -1 + r2 + r3
    
    # 检查 -2 的立方根的实数解
    assert real_root(root(-2, 3)) == -root(2, 3)
    
    # 检查对于 -8. 的立方根的实数解
    assert real_root(-8., 3) == -2.0
    
    # 创建符号 x 和 n，并计算 real_root 函数结果 g
    x = Symbol('x')
    n = Symbol('n')
    g = real_root(x, n)
    
    # 检查符号替换后的结果
    assert g.subs({"x": -8, "n": 3}) == -2
    assert g.subs({"x": 8, "n": 3}) == 2
    
    # 如果没有实数根则返回主根，这里忽略 RuntimeWarning 警告
    assert g.subs({"x": I, "n": 3}) == cbrt(I)
    assert g.subs({"x": -8, "n": 2}) == sqrt(-8)
    assert g.subs({"x": I, "n": 2}) == sqrt(I)


def test_issue_11463():
    # 导入 numpy 模块
    numpy = import_module('numpy')
    
    # 如果 numpy 未安装则跳过测试
    if not numpy:
        skip("numpy not installed.")
    
    # 创建符号 x，并使用 lambdify 函数将表达式转换为 numpy 函数 f
    x = Symbol('x')
    f = lambdify(x, real_root((log(x/(x-2))), 3), 'numpy')
    
    # 使用 ignore_warnings 上下文管理器忽略 RuntimeWarning 警告
    with ignore_warnings(RuntimeWarning):
        assert f(numpy.array(-1)) < -1


def test_rewrite_MaxMin_as_Heaviside():
    # 从 sympy.abc 中导入符号 x
    from sympy.abc import x
    
    # 检查 Max 函数重写为 Heaviside 函数的结果
    assert Max(0, x).rewrite(Heaviside) == x*Heaviside(x)
    
    # 检查 Max 函数重写为 Heaviside 函数的结果
    assert Max(3, x).rewrite(Heaviside) == x*Heaviside(x - 3) + \
        3*Heaviside(-x + 3)
    
    # 检查 Max 函数重写为 Heaviside 函数的结果
    assert Max(0, x+2, 2*x).rewrite(Heaviside) == \
        2*x*Heaviside(2*x)*Heaviside(x - 2) + \
        (x + 2)*Heaviside(-x + 2)*Heaviside(x + 2)
    
    # 检查 Min 函数重写为 Heaviside 函数的结果
    assert Min(0, x).rewrite(Heaviside) == x*Heaviside(-x)
    
    # 检查 Min 函数重写为 Heaviside 函数的结果
    assert Min(3, x).rewrite(Heaviside) == x*Heaviside(-x + 3)
    # 断言语句，用于验证 Min 函数在应用 Heaviside 重写后的结果是否与预期相等
    assert Min(x, -x, -2).rewrite(Heaviside) == \
        # 计算 Min(x, -x, -2) 在应用 Heaviside 函数重写后的表达式
        x * Heaviside(-2*x) * Heaviside(-x - 2) - \
        x * Heaviside(2*x) * Heaviside(x - 2) \
        - 2 * Heaviside(-x + 2) * Heaviside(x + 2)
def test_rewrite_MaxMin_as_Piecewise():
    # 导入符号变量和条件分段函数
    from sympy.core.symbol import symbols
    from sympy.functions.elementary.piecewise import Piecewise
    
    # 定义符号变量
    x, y, z, a, b = symbols('x y z a b', real=True)
    # 定义额外的符号变量
    vx, vy, va = symbols('vx vy va')
    
    # 测试最大值函数 Max 重写为 Piecewise 形式
    assert Max(a, b).rewrite(Piecewise) == Piecewise((a, a >= b), (b, True))
    assert Max(x, y, z).rewrite(Piecewise) == Piecewise((x, (x >= y) & (x >= z)), (y, y >= z), (z, True))
    assert Max(x, y, a, b).rewrite(Piecewise) == Piecewise((a, (a >= b) & (a >= x) & (a >= y)),
        (b, (b >= x) & (b >= y)), (x, x >= y), (y, True))
    
    # 测试最小值函数 Min 重写为 Piecewise 形式
    assert Min(a, b).rewrite(Piecewise) == Piecewise((a, a <= b), (b, True))
    assert Min(x, y, z).rewrite(Piecewise) == Piecewise((x, (x <= y) & (x <= z)), (y, y <= z), (z, True))
    assert Min(x,  y, a, b).rewrite(Piecewise) ==  Piecewise((a, (a <= b) & (a <= x) & (a <= y)),
        (b, (b <= x) & (b <= y)), (x, x <= y), (y, True))

    # 测试对非显式实数参数的 Max/Min 函数进行 Piecewise 重写
    assert Max(vx, vy).rewrite(Piecewise) == Piecewise((vx, vx >= vy), (vy, True))
    assert Min(va, vx, vy).rewrite(Piecewise) == Piecewise((va, (va <= vx) & (va <= vy)), (vx, vx <= vy), (vy, True))


def test_issue_11099():
    # 导入符号变量
    from sympy.abc import x, y
    
    # 一些固定值的测试数据
    fixed_test_data = {x: -2, y: 3}
    assert Min(x, y).evalf(subs=fixed_test_data) == \
        Min(x, y).subs(fixed_test_data).evalf()
    assert Max(x, y).evalf(subs=fixed_test_data) == \
        Max(x, y).subs(fixed_test_data).evalf()
    
    # 随机生成一些测试数据
    from sympy.core.random import randint
    for i in range(20):
        random_test_data = {x: randint(-100, 100), y: randint(-100, 100)}
        assert Min(x, y).evalf(subs=random_test_data) == \
            Min(x, y).subs(random_test_data).evalf()
        assert Max(x, y).evalf(subs=random_test_data) == \
            Max(x, y).subs(random_test_data).evalf()


def test_issue_12638():
    # 导入符号变量
    from sympy.abc import a, b, c
    
    # 测试 Min/Max 函数与 Max(a, b) 之间的关系
    assert Min(a, b, c, Max(a, b)) == Min(a, b, c)
    assert Min(a, b, Max(a, b, c)) == Min(a, b)
    assert Min(a, b, Max(a, c)) == Min(a, b)


def test_issue_21399():
    # 导入符号变量
    from sympy.abc import a, b, c
    
    # 测试 Max(Min(a, b), Min(a, b, c)) 的计算
    assert Max(Min(a, b), Min(a, b, c)) == Min(a, b)


def test_instantiation_evaluation():
    # 导入符号变量
    from sympy.abc import v, w, x, y, z
    
    # 测试实例化和评估
    assert Min(1, Max(2, x)) == 1
    assert Max(3, Min(2, x)) == 3
    assert Min(Max(x, y), Max(x, z)) == Max(x, Min(y, z))
    assert set(Min(Max(w, x), Max(y, z)).args) == {
        Max(w, x), Max(y, z)}
    assert Min(Max(x, y), Max(x, z), w) == Min(
        w, Max(x, Min(y, z)))
    A, B = Min, Max
    for i in range(2):
        assert A(x, B(x, y)) == x
        assert A(x, B(y, A(x, w, z))) == A(x, B(y, A(w, z)))
        A, B = B, A
    assert Min(w, Max(x, y), Max(v, x, z)) == Min(
        w, Max(x, Min(y, Max(v, z))))


def test_rewrite_as_Abs():
    # 导入排列组合工具
    from itertools import permutations
    from sympy.functions.elementary.complexes import Abs
    # 从 sympy.abc 模块中导入符号 x, y, z, w
    from sympy.abc import x, y, z, w

    # 定义一个测试函数 test，接受一个表达式 e 作为参数
    def test(e):
        # 获取表达式 e 中的自由符号
        free = e.free_symbols
        # 将表达式 e 重写为绝对值形式
        a = e.rewrite(Abs)
        # 断言绝对值形式的表达式 a 不包含 Min 和 Max 函数
        assert not a.has(Min, Max)
        # 对自由符号的排列进行循环测试
        for i in permutations(range(len(free))):
            # 将自由符号与排列的索引对应起来，形成替换字典
            reps = dict(zip(free, i))
            # 断言用排列替换后的表达式 a 等于用排列替换后的原始表达式 e
            assert a.xreplace(reps) == e.xreplace(reps)

    # 调用 test 函数，测试 Min 函数应用于符号 x 和 y 的情况
    test(Min(x, y))

    # 调用 test 函数，测试 Max 函数应用于符号 x 和 y 的情况
    test(Max(x, y))

    # 调用 test 函数，测试 Min 函数应用于符号 x, y, z 的情况
    test(Min(x, y, z))

    # 调用 test 函数，测试 Min 函数应用于 Max(w, x) 和 Max(y, z) 的情况
    test(Min(Max(w, x), Max(y, z)))
# 定义测试函数 test_issue_14000
def test_issue_14000():
    # 断言 sqrt(4, evaluate=False) 返回类型为 Pow 类型
    assert isinstance(sqrt(4, evaluate=False), Pow) == True
    # 断言 cbrt(3.5, evaluate=False) 返回类型为 Pow 类型
    assert isinstance(cbrt(3.5, evaluate=False), Pow) == True
    # 断言 root(16, 4, evaluate=False) 返回类型为 Pow 类型
    assert isinstance(root(16, 4, evaluate=False), Pow) == True

    # 断言 sqrt(4, evaluate=False) 的计算结果为 Pow(4, S.Half, evaluate=False)
    assert sqrt(4, evaluate=False) == Pow(4, S.Half, evaluate=False)
    # 断言 cbrt(3.5, evaluate=False) 的计算结果为 Pow(3.5, Rational(1, 3), evaluate=False)
    assert cbrt(3.5, evaluate=False) == Pow(3.5, Rational(1, 3), evaluate=False)
    # 断言 root(4, 2, evaluate=False) 的计算结果为 Pow(4, S.Half, evaluate=False)
    assert root(4, 2, evaluate=False) == Pow(4, S.Half, evaluate=False)

    # 断言 root(16, 4, 2, evaluate=False) 中存在 Pow 类型
    assert root(16, 4, 2, evaluate=False).has(Pow) == True
    # 断言 real_root(-8, 3, evaluate=False) 中存在 Pow 类型
    assert real_root(-8, 3, evaluate=False).has(Pow) == True

# 定义测试函数 test_issue_6899
def test_issue_6899():
    # 导入 Lambda 类
    from sympy.core.function import Lambda
    # 创建符号变量 x
    x = Symbol('x')
    # 创建 Lambda 表达式 eqn = Lambda(x, x)
    eqn = Lambda(x, x)
    # 断言 eqn.func(*eqn.args) == eqn，即 Lambda 表达式的属性符合预期
    assert eqn.func(*eqn.args) == eqn

# 定义测试函数 test_Rem
def test_Rem():
    # 导入符号变量 x, y
    from sympy.abc import x, y
    # 断言 Rem(5, 3) 的计算结果为 2
    assert Rem(5, 3) == 2
    # 断言 Rem(-5, 3) 的计算结果为 -2
    assert Rem(-5, 3) == -2
    # 断言 Rem(5, -3) 的计算结果为 2
    assert Rem(5, -3) == 2
    # 断言 Rem(-5, -3) 的计算结果为 -2
    assert Rem(-5, -3) == -2
    # 断言 Rem(x**3, y) 等于 Rem(x**3, y)，即符号表达式保持一致
    assert Rem(x**3, y) == Rem(x**3, y)
    # 断言 Rem(Rem(-5, 3) + 3, 3) 的计算结果为 1
    assert Rem(Rem(-5, 3) + 3, 3) == 1

# 定义测试函数 test_minmax_no_evaluate
def test_minmax_no_evaluate():
    # 导入 evaluate 符号
    from sympy import evaluate
    # 创建正数符号变量 p
    p = Symbol('p', positive=True)

    # 断言 Max(1, 3) 的计算结果为 3
    assert Max(1, 3) == 3
    # 断言 Max(1, 3) 的参数为空元组
    assert Max(1, 3).args == ()
    # 断言 Max(0, p) 的计算结果为 p
    assert Max(0, p) == p
    # 断言 Max(0, p) 的参数为空元组
    assert Max(0, p).args == ()
    # 断言 Min(0, p) 的计算结果为 0
    assert Min(0, p) == 0
    # 断言 Min(0, p) 的参数为空元组
    assert Min(0, p).args == ()

    # 断言 Max(1, 3, evaluate=False) 的计算结果不等于 3
    assert Max(1, 3, evaluate=False) != 3
    # 断言 Max(1, 3, evaluate=False) 的参数为 (1, 3)
    assert Max(1, 3, evaluate=False).args == (1, 3)
    # 断言 Max(0, p, evaluate=False) 的计算结果不等于 p
    assert Max(0, p, evaluate=False) != p
    # 断言 Max(0, p, evaluate=False) 的参数为 (0, p)
    assert Max(0, p, evaluate=False).args == (0, p)
    # 断言 Min(0, p, evaluate=False) 的计算结果不等于 0
    assert Min(0, p, evaluate=False) != 0
    # 断言 Min(0, p, evaluate=False) 的参数为 (0, p)

    # 使用 evaluate(False) 上下文管理器
    with evaluate(False):
        # 断言 Max(1, 3) 的计算结果不等于 3
        assert Max(1, 3) != 3
        # 断言 Max(1, 3) 的参数为 (1, 3)
        assert Max(1, 3).args == (1, 3)
        # 断言 Max(0, p) 的计算结果不等于 p
        assert Max(0, p) != p
        # 断言 Max(0, p) 的参数为 (0, p)
        assert Max(0, p).args == (0, p)
        # 断言 Min(0, p) 的计算结果不等于 0
        assert Min(0, p) != 0
        # 断言 Min(0, p) 的参数为 (0, p)
```