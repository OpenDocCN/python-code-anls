# `D:\src\scipysrc\sympy\sympy\calculus\tests\test_accumulationbounds.py`

```
# 从 sympy.core.numbers 模块导入 E, Rational, oo, pi, zoo
# 从 sympy.core.singleton 模块导入 S
# 从 sympy.core.symbol 模块导入 Symbol
# 从 sympy.functions.elementary.exponential 模块导入 exp, log
# 从 sympy.functions.elementary.miscellaneous 模块导入 Max, Min, sqrt
# 从 sympy.functions.elementary.trigonometric 模块导入 cos, sin, tan
# 从 sympy.calculus.accumulationbounds 模块导入 AccumBounds
# 从 sympy.core 模块导入 Add, Mul, Pow
# 从 sympy.core.expr 模块导入 unchanged
# 从 sympy.testing.pytest 模块导入 raises, XFAIL
# 从 sympy.abc 模块导入 x
from sympy.core.numbers import (E, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import Add, Mul, Pow
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x

# 创建一个实数符号对象 a
a = Symbol('a', real=True)
# B 被赋值为 AccumBounds 类型，用于表示区间累积边界
B = AccumBounds


# 定义一个测试函数 test_AccumBounds
def test_AccumBounds():
    # 断言验证累积边界对象 B(1, 2) 的构造参数为 (1, 2)
    assert B(1, 2).args == (1, 2)
    # 断言验证累积边界对象 B(1, 2) 的 delta 属性为 S.One
    assert B(1, 2).delta is S.One
    # 断言验证累积边界对象 B(1, 2) 的中点 mid 为有理数 3/2
    assert B(1, 2).mid == Rational(3, 2)
    # 断言验证累积边界对象 B(1, 3) 是否为实数，期望为 True
    assert B(1, 3).is_real == True

    # 断言验证累积边界对象 B(1, 1) 是否为 S.One
    assert B(1, 1) is S.One

    # 累积边界对象的加法操作
    assert B(1, 2) + 1 == B(2, 3)
    assert 1 + B(1, 2) == B(2, 3)
    assert B(1, 2) + B(2, 3) == B(3, 5)

    # 累积边界对象的负数操作
    assert -B(1, 2) == B(-2, -1)

    # 累积边界对象的减法操作
    assert B(1, 2) - 1 == B(0, 1)
    assert 1 - B(1, 2) == B(-1, 0)
    assert B(2, 3) - B(1, 2) == B(0, 2)

    # 累积边界对象与符号对象 x 的加法操作
    assert x + B(1, 2) == Add(B(1, 2), x)
    # 累积边界对象与实数符号对象 a 的加法操作
    assert a + B(1, 2) == B(1 + a, 2 + a)
    # 累积边界对象与符号对象 x 的减法操作
    assert B(1, 2) - x == Add(B(1, 2), -x)

    # 累积边界对象与无穷大的加减法操作
    assert B(-oo, 1) + oo == B(-oo, oo)
    assert B(1, oo) + oo is oo
    assert B(1, oo) - oo == B(-oo, oo)
    assert (-oo - B(-1, oo)) is -oo
    assert B(-oo, 1) - oo is -oo

    # 累积边界对象与无穷大的加减法操作
    assert B(1, oo) - oo == B(-oo, oo)
    assert B(-oo, 1) - (-oo) == B(-oo, oo)
    assert (oo - B(1, oo)) == B(-oo, oo)
    assert (-oo - B(1, oo)) is -oo

    # 累积边界对象与实数的除法操作
    assert B(1, 2)/2 == B(S.Half, 1)
    assert 2/B(2, 3) == B(Rational(2, 3), 1)
    assert 1/B(-1, 1) == B(-oo, oo)

    # 累积边界对象的绝对值操作
    assert abs(B(1, 2)) == B(1, 2)
    assert abs(B(-2, -1)) == B(1, 2)
    assert abs(B(-2, 1)) == B(0, 2)
    assert abs(B(-1, 2)) == B(0, 2)

    # 引发 ValueError 异常的情况
    c = Symbol('c')
    raises(ValueError, lambda: B(0, c))
    raises(ValueError, lambda: B(1, -1))
    r = Symbol('r', real=True)
    raises(ValueError, lambda: B(r, r - 1))


# 定义一个测试函数 test_AccumBounds_mul
def test_AccumBounds_mul():
    # 累积边界对象的乘法操作
    assert B(1, 2)*2 == B(2, 4)
    assert 2*B(1, 2) == B(2, 4)
    assert B(1, 2)*B(2, 3) == B(2, 6)
    assert B(0, 2)*B(2, oo) == B(0, oo)
    l, r = B(-oo, oo), B(-a, a)
    assert l*r == B(-oo, oo)
    assert r*l == B(-oo, oo)
    l, r = B(1, oo), B(-3, -2)
    assert l*r == B(-oo, -2)
    assert r*l == B(-oo, -2)
    assert B(1, 2)*0 == 0
    assert B(1, oo)*0 == B(0, oo)
    assert B(-oo, 1)*0 == B(-oo, 0)
    assert B(-oo, oo)*0 == B(-oo, oo)

    # 累积边界对象与符号对象 x 的乘法操作，禁止自动求值
    assert B(1, 2)*x == Mul(B(1, 2), x, evaluate=False)

    # 累积边界对象与无穷大的乘法操作
    assert B(0, 2)*oo == B(0, oo)
    assert B(-2, 0)*oo == B(-oo, 0)
    assert B(0, 2)*(-oo) == B(-oo, 0)
    assert B(-2, 0)*(-oo) == B(0, oo)
    assert B(-1, 1)*oo == B(-oo, oo)
    assert B(-1, 1)*(-oo) == B(-oo, oo)
    assert B(-oo, oo)*oo == B(-oo, oo)


# 定义一个测试函数 test_AccumBounds_div
def test_AccumBounds_div():
    # 累积边界对象的除法操作
    assert B(-1, 3)/B(3, 4) == B(Rational(-1, 3), 1)
    assert B(-2, 4)/B(-3, 4) == B(-oo, oo)
    assert B(-3, -2)/B(-4, 0) == B(S.Half, oo)

    # 这两个测试用例在 B 的并集改进后应该有更好的结果
    # after Union of B is improved


这段代码包含了对 SymPy 中累积边界 `Acc
    # 断言：区间 B(-3, -2) 除以区间 B(-2, 1)，结果应为整个实数轴 B(-oo, oo)
    assert B(-3, -2)/B(-2, 1) == B(-oo, oo)

    # 断言：区间 B(2, 3) 除以区间 B(-2, 2)，结果应为整个实数轴 B(-oo, oo)
    assert B(2, 3)/B(-2, 2) == B(-oo, oo)

    # 断言：区间 B(-3, -2) 除以区间 B(0, 4)，结果应为 B(-oo, -1/2)
    assert B(-3, -2)/B(0, 4) == B(-oo, Rational(-1, 2))

    # 断言：区间 B(2, 4) 除以区间 B(-3, 0)，结果应为 B(-oo, -2/3)
    assert B(2, 4)/B(-3, 0) == B(-oo, Rational(-2, 3))

    # 断言：区间 B(2, 4) 除以区间 B(0, 3)，结果应为 B(2/3, oo)
    assert B(2, 4)/B(0, 3) == B(Rational(2, 3), oo)

    # 断言：区间 B(0, 1) 除以区间 B(0, 1)，结果应为 B(0, oo)
    assert B(0, 1)/B(0, 1) == B(0, oo)

    # 断言：区间 B(-1, 0) 除以区间 B(0, 1)，结果应为 B(-oo, 0)
    assert B(-1, 0)/B(0, 1) == B(-oo, 0)

    # 断言：区间 B(-1, 2) 除以区间 B(-2, 2)，结果应为整个实数轴 B(-oo, oo)
    assert B(-1, 2)/B(-2, 2) == B(-oo, oo)

    # 断言：区间 1 除以区间 B(-1, 2)，结果应为整个实数轴 B(-oo, oo)
    assert 1/B(-1, 2) == B(-oo, oo)

    # 断言：区间 1 除以区间 B(0, 2)，结果应为 B(1/2, oo)
    assert 1/B(0, 2) == B(S.Half, oo)

    # 断言：区间 -1 除以区间 B(0, 2)，结果应为 B(-oo, -1/2)
    assert (-1)/B(0, 2) == B(-oo, Rational(-1, 2))

    # 断言：区间 1 除以区间 B(-oo, 0)，结果应为 B(-oo, 0)
    assert 1/B(-oo, 0) == B(-oo, 0)

    # 断言：区间 1 除以区间 B(-1, 0)，结果应为 B(-oo, -1)
    assert 1/B(-1, 0) == B(-oo, -1)

    # 断言：区间 -2 除以区间 B(-oo, 0)，结果应为 B(0, oo)
    assert (-2)/B(-oo, 0) == B(0, oo)

    # 断言：区间 1 除以区间 B(-oo, -1)，结果应为 B(-1, 0)
    assert 1/B(-oo, -1) == B(-1, 0)

    # 断言：区间 B(1, 2) 除以变量 a，结果应为 B(1/2, oo) * (1/a)，不进行评估
    assert B(1, 2)/a == Mul(B(1, 2), 1/a, evaluate=False)

    # 断言：区间 B(1, 2) 除以 0，结果应为 B(1, 2) * zoo
    assert B(1, 2)/0 == B(1, 2)*zoo

    # 断言：区间 B(1, oo) 除以 oo，结果应为 B(0, oo)
    assert B(1, oo)/oo == B(0, oo)

    # 断言：区间 B(1, oo) 除以 -oo，结果应为 B(-oo, 0)
    assert B(1, oo)/(-oo) == B(-oo, 0)

    # 断言：区间 B(-oo, -1) 除以 oo，结果应为 B(-oo, 0)
    assert B(-oo, -1)/oo == B(-oo, 0)

    # 断言：区间 B(-oo, -1) 除以 -oo，结果应为 B(0, oo)
    assert B(-oo, -1)/(-oo) == B(0, oo)

    # 断言：区间 B(-oo, oo) 除以 oo，结果应为 B(-oo, oo)
    assert B(-oo, oo)/oo == B(-oo, oo)

    # 断言：区间 B(-oo, oo) 除以 -oo，结果应为 B(-oo, oo)
    assert B(-oo, oo)/(-oo) == B(-oo, oo)

    # 断言：区间 B(-1, oo) 除以 oo，结果应为 B(0, oo)
    assert B(-1, oo)/oo == B(0, oo)

    # 断言：区间 B(-1, oo) 除以 -oo，结果应为 B(-oo, 0)
    assert B(-1, oo)/(-oo) == B(-oo, 0)

    # 断言：区间 B(-oo, 1) 除以 oo，结果应为 B(-oo, 0)
    assert B(-oo, 1)/oo == B(-oo, 0)

    # 断言：区间 B(-oo, 1) 除以 -oo，结果应为 B(0, oo)
    assert B(-oo, 1)/(-oo) == B(0, oo)
# 定义名为 test_issue_18795 的测试函数
def test_issue_18795():
    # 创建一个符号 r，并指定其为实数
    r = Symbol('r', real=True)
    # 创建 AccumulationBounds 对象 a，表示区间 [-1, 1]
    a = B(-1, 1)
    # 创建 AccumulationBounds 对象 c，表示区间 [7, ∞)
    c = B(7, oo)
    # 创建 AccumulationBounds 对象 b，表示整个实数轴范围
    b = B(-oo, oo)
    # 断言表达式 c - tan(r) 的结果应该是区间 [7 - tan(r), ∞)
    assert c - tan(r) == B(7 - tan(r), oo)
    # 断言表达式 b + tan(r) 的结果应该是整个实数轴范围
    assert b + tan(r) == B(-oo, oo)
    # 断言表达式 (a + r)/a 的结果应该是区间 [-∞, ∞] * [r - 1, r + 1]
    assert (a + r)/a == B(-oo, oo)*B(r - 1, r + 1)
    # 断言表达式 (b + a)/a 的结果应该是区间 [-∞, ∞]
    assert (b + a)/a == B(-oo, oo)


# 定义名为 test_AccumBounds_func 的测试函数
def test_AccumBounds_func():
    # 断言表达式 (x**2 + 2*x + 1).subs(x, B(-1, 1)) 的结果应该是区间 [-1, 4]
    assert (x**2 + 2*x + 1).subs(x, B(-1, 1)) == B(-1, 4)
    # 断言表达式 exp(B(0, 1)) 的结果应该是区间 [1, e]
    assert exp(B(0, 1)) == B(1, E)
    # 断言表达式 exp(B(-oo, oo)) 的结果应该是区间 [0, ∞)
    assert exp(B(-oo, oo)) == B(0, oo)
    # 断言表达式 log(B(3, 6)) 的结果应该是区间 [log(3), log(6)]
    assert log(B(3, 6)) == B(log(3), log(6))


# 标记为 XFAIL 的测试函数 test_AccumBounds_powf，表示预期它会失败
@XFAIL
def test_AccumBounds_powf():
    # 创建一个非负整数符号 nn
    nn = Symbol('nn', nonnegative=True)
    # 断言表达式 B(1 + nn, 2 + nn)**B(1, 2) 的结果应该是区间 [1 + nn, (2 + nn)^2]
    assert B(1 + nn, 2 + nn)**B(1, 2) == B(1 + nn, (2 + nn)**2)
    # 创建一个整数符号 i，且为负数
    i = Symbol('i', integer=True, negative=True)
    # 断言表达式 B(1, 2)**i 的结果应该是区间 [2^i, 1]
    assert B(1, 2)**i == B(2**i, 1)


# 定义名为 test_AccumBounds_pow 的测试函数
def test_AccumBounds_pow():
    # 断言表达式 B(0, 2)**2 的结果应该是区间 [0, 4]
    assert B(0, 2)**2 == B(0, 4)
    # 断言表达式 B(-1, 1)**2 的结果应该是区间 [0, 1]
    assert B(-1, 1)**2 == B(0, 1)
    # 断言表达式 B(1, 2)**2 的结果应该是区间 [1, 4]
    assert B(1, 2)**2 == B(1, 4)
    # 断言表达式 B(-1, 2)**3 的结果应该是区间 [-1, 8]
    assert B(-1, 2)**3 == B(-1, 8)
    # 断言表达式 B(-1, 1)**0 的结果应该是单一值 1
    assert B(-1, 1)**0 == 1

    # 断言表达式 B(1, 2)**Rational(5, 2) 的结果应该是区间 [1, 4*sqrt(2)]
    assert B(1, 2)**Rational(5, 2) == B(1, 4*sqrt(2))
    # 断言表达式 B(0, 2)**S.Half 的结果应该是区间 [0, sqrt(2)]
    assert B(0, 2)**S.Half == B(0, sqrt(2))

    # 创建一个负数符号 neg
    neg = Symbol('neg', negative=True)
    # 断言表达式 unchanged(Pow, B(neg, 1), S.Half) 应保持不变
    assert unchanged(Pow, B(neg, 1), S.Half)
    # 创建一个非负整数符号 nn
    nn = Symbol('nn', nonnegative=True)
    # 断言表达式 B(nn, nn + 1)**S.Half 的结果应该是区间 [sqrt(nn), sqrt(nn + 1)]
    assert B(nn, nn + 1)**S.Half == B(sqrt(nn), sqrt(nn + 1))
    # 断言表达式 B(nn, nn + 1)**nn 的结果应该是区间 [nn^nn, (nn + 1)^nn]
    assert B(nn, nn + 1)**nn == B(nn**nn, (nn + 1)**nn)
    # 断言表达式 unchanged(Pow, B(nn, nn + 1), x) 应保持不变
    assert unchanged(Pow, B(nn, nn + 1), x)
    # 创建一个整数符号 i
    i = Symbol('i', integer=True)
    # 断言表达式 B(1, 2)**i 的结果应该是区间 [Min(1, 2^i), Max(1, 2^i)]
    assert B(1, 2)**i == B(Min(1, 2**i), Max(1, 2**i))
    # 创建一个非负整数符号 i
    i = Symbol('i', integer=True, nonnegative=True)
    # 断言表达式 B(1, 2)**i 的结果应该是区间 [1, 2^i]
    assert B(1, 2)**i == B(1, 2**i)
    # 断言表达式 B(0, 1)**i 的结果应该是区间 [0^i, 1]
    assert B(0, 1)**i == B(0**i, 1)

    # 断言表达式 B(1, 5)**(-2) 的结果应该是区间 [1/25, 1]
    assert B(1, 5)**(-2) == B(Rational(1, 25), 1)
    # 断言表达式 B(-1, 3)**(-2) 的结果应该是区间 [0, ∞)
    assert B(-1, 3)**(-2) == B(0, oo)
    # 断言表达式 B(0, 2)**(-3) 的结果应该是区间 [1/8, ∞)
    assert B(0, 2)**(-3) == B(Rational(1, 8), oo)
    # 断言表达式 B(-2, 0)**(-3) 的结果应该是区间 [-∞, -1/8]
    assert B(-2, 0)**(-3) == B(-oo, -Rational(1, 8))
    # 断言表达式 B(0, 2)**(-2) 的结果应该是区间 [1/4, ∞)
    assert B(0, 2)**(-2) == B(Rational(1, 4), oo)
    # 断言表达式 B(-1, 2)**(-3) 的结果应该是区间 [-∞, ∞)
    assert B(-1, 2)**(-3) == B(-oo, oo)
    # 断言表达式 B(-3, -2)**(-3) 的结果应该是区间 [-1/8, -1/27]
    assert B(-3, -2)**(-3) == B(Rational(-1, 8), Rational(-1
    # 断言：验证负无穷到正无穷的负一次幂等于负无穷到正无穷的区间
    assert B(-1, 2)**(-oo) == B(-oo, oo)
    
    # 断言：验证正切函数的正弦函数的双倍角的幂函数在区间[0, pi/2]内的代换结果
    assert (tan(x)**sin(2*x)).subs(x, B(0, pi/2)) == \
        Pow(B(-oo, oo), B(0, 1))
# 定义一个测试函数，用于测试 AccumBounds 类的指数运算
def test_AccumBounds_exponent():
    # 当基数为 0 时，计算 0 的指数范围 a 到 a + S.Half
    z = 0**B(a, a + S.Half)
    # 断言当 a = 0 时，结果应该是 B(0, 1)
    assert z.subs(a, 0) == B(0, 1)
    # 断言当 a = 1 时，结果应该是 0
    assert z.subs(a, 1) == 0
    # 检查当 a = -1 时，结果是否为 Pow 类型，且参数为 (0, B(-1, -S.Half))
    p = z.subs(a, -1)
    assert p.is_Pow and p.args == (0, B(-1, -S.Half))

    # 当基数大于 0 时的测试情况
    # 当基数为 1 时，无论边界类型如何，结果应为 1
    assert 1**B(a, a + 1) == 1
    # 当基数为 S.Half 时，边界为 -2 到 2，结果应为 B(S(1)/4, 4)
    assert S.Half**B(-2, 2) == B(S(1)/4, 4)
    # 当基数为 2 时，边界为 -2 到 2，结果应为 B(S(1)/4, 4)
    assert 2**B(-2, 2) == B(S(1)/4, 4)

    # 对于 +eps 可能会引入 +oo 的情况，特别是当指数为负整数时
    assert B(0, 1)**B(S(1)/2, 1) == B(0, 1)
    assert B(0, 1)**B(0, 1) == B(0, 1)

    # 正数基数具有正数边界
    assert B(2, 3)**B(-3, -2) == B(S(1)/27, S(1)/4)
    assert B(2, 3)**B(-3, 2) == B(S(1)/27, 9)

    # 边界生成虚部未评估的情况
    assert unchanged(Pow, B(-1, 1), B(1, 2))
    assert B(0, S(1)/2)**B(1, oo) == B(0, S(1)/2)
    assert B(0, 1)**B(1, oo) == B(0, oo)
    assert B(0, 2)**B(1, oo) == B(0, oo)
    assert B(0, oo)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 1)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 1)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, 1)**B(-oo, oo) == B(0, oo)
    assert B(S(1)/2, 2)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 2)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, 2)**B(-oo, oo) == B(0, oo)
    assert B(S(1)/2, oo)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, oo)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, oo)**B(-oo, oo) == B(0, oo)
    assert B(1, 2)**B(1, oo) == B(0, oo)
    assert B(1, 2)**B(-oo, -1) == B(0, oo)
    assert B(1, 2)**B(-oo, oo) == B(0, oo)
    assert B(1, oo)**B(1, oo) == B(0, oo)
    assert B(1, oo)**B(-oo, -1) == B(0, oo)
    assert B(1, oo)**B(-oo, oo) == B(0, oo)
    assert B(2, oo)**B(1, oo) == B(2, oo)
    assert B(2, oo)**B(-oo, -1) == B(0, S(1)/2)
    assert B(2, oo)**B(-oo, oo) == B(0, oo)


# 测试 AccumBounds 类的比较操作
def test_comparison_AccumBounds():
    assert (B(1, 3) < 4) == S.true
    assert (B(1, 3) < -1) == S.false
    assert (B(1, 3) < 2).rel_op == '<'
    assert (B(1, 3) <= 2).rel_op == '<='

    assert (B(1, 3) > 4) == S.false
    assert (B(1, 3) > -1) == S.true
    assert (B(1, 3) > 2).rel_op == '>'
    assert (B(1, 3) >= 2).rel_op == '>='

    assert (B(1, 3) < B(4, 6)) == S.true
    assert (B(1, 3) < B(2, 4)).rel_op == '<'
    assert (B(1, 3) < B(-2, 0)) == S.false

    assert (B(1, 3) <= B(4, 6)) == S.true
    assert (B(1, 3) <= B(-2, 0)) == S.false

    assert (B(1, 3) > B(4, 6)) == S.false
    assert (B(1, 3) > B(-2, 0)) == S.true

    assert (B(1, 3) >= B(4, 6)) == S.false
    assert (B(1, 3) >= B(-2, 0)) == S.true

    # issue 13499
    assert (cos(x) > 0).subs(x, oo) == (B(-1, 1) > 0)

    c = Symbol('c')
    # 检查当比较符号左侧为 AccumBounds 对象，右侧为符号时是否会引发 TypeError
    raises(TypeError, lambda: (B(0, 1) < c))
    raises(TypeError, lambda: (B(0, 1) <= c))
    raises(TypeError, lambda: (B(0, 1) > c))
    raises(TypeError, lambda: (B(0, 1) >= c))


# 测试 AccumBounds 类的包含操作
def test_contains_AccumBounds():
    assert (1 in B(1, 2)) == S.true
    raises(TypeError, lambda: a in B(1, 2))
    assert 0 in B(-1, 0)
    # 使用 raises 函数来验证是否会引发 TypeError 异常
    raises(TypeError, lambda:
        (cos(1)**2 + sin(1)**2 - 1) in B(-1, 0))
    # 使用 assert 断言来验证 -oo 是否在 B(1, oo) 范围内，预期结果为真
    assert (-oo in B(1, oo)) == S.true
    # 使用 assert 断言来验证 oo 是否在 B(-oo, 0) 范围内，预期结果为真
    assert (oo in B(-oo, 0)) == S.true

    # issue 13159 的问题验证
    # 使用 assert 断言来验证 Mul(0, B(-1, 1)) 的结果是否为 0
    assert Mul(0, B(-1, 1)) == Mul(B(-1, 1), 0) == 0
    # 导入 itertools 库，使用 itertools.permutations 生成包含 [0, B(-1, 1), x] 的所有排列
    import itertools
    # 遍历每个排列 perm
    for perm in itertools.permutations([0, B(-1, 1), x]):
        # 使用 assert 断言来验证 Mul(*perm) 的结果是否为 0
        assert Mul(*perm) == 0
# 测试函数，用于验证 AccumBounds 类的 intersection 方法
def test_intersection_AccumBounds():
    # 断言：B(0, 3) 和 B(1, 2) 的交集应该是 B(1, 2)
    assert B(0, 3).intersection(B(1, 2)) == B(1, 2)
    # 断言：B(0, 3) 和 B(1, 4) 的交集应该是 B(1, 3)
    assert B(0, 3).intersection(B(1, 4)) == B(1, 3)
    # 断言：B(0, 3) 和 B(-1, 2) 的交集应该是 B(0, 2)
    assert B(0, 3).intersection(B(-1, 2)) == B(0, 2)
    # 断言：B(0, 3) 和 B(-1, 4) 的交集应该是 B(0, 3)
    assert B(0, 3).intersection(B(-1, 4)) == B(0, 3)
    # 断言：B(0, 1) 和 B(2, 3) 的交集应该是空集 S.EmptySet
    assert B(0, 1).intersection(B(2, 3)) == S.EmptySet
    # 断言：调用 intersection 方法时传入类型错误的参数，应该抛出 TypeError 异常
    raises(TypeError, lambda: B(0, 3).intersection(1))

# 测试函数，用于验证 AccumBounds 类的 union 方法
def test_union_AccumBounds():
    # 断言：B(0, 3) 和 B(1, 2) 的并集应该是 B(0, 3)
    assert B(0, 3).union(B(1, 2)) == B(0, 3)
    # 断言：B(0, 3) 和 B(1, 4) 的并集应该是 B(0, 4)
    assert B(0, 3).union(B(1, 4)) == B(0, 4)
    # 断言：B(0, 3) 和 B(-1, 2) 的并集应该是 B(-1, 3)
    assert B(0, 3).union(B(-1, 2)) == B(-1, 3)
    # 断言：B(0, 3) 和 B(-1, 4) 的并集应该是 B(-1, 4)
    assert B(0, 3).union(B(-1, 4)) == B(-1, 4)
    # 断言：调用 union 方法时传入类型错误的参数，应该抛出 TypeError 异常
    raises(TypeError, lambda: B(0, 3).union(1))
```