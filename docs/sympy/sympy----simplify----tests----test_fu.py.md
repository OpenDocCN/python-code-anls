# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_fu.py`

```
# 从 sympy.core.add 模块中导入 Add 类
from sympy.core.add import Add
# 从 sympy.core.mul 模块中导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.numbers 模块中导入 I, Rational, pi 等数学常数和符号
from sympy.core.numbers import (I, Rational, pi)
# 从 sympy.core.parameters 模块中导入 evaluate 函数
from sympy.core.parameters import evaluate
# 从 sympy.core.singleton 模块中导入 S 符号
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块中导入 Dummy, Symbol, symbols 等符号相关函数和类
from sympy.core.symbol import (Dummy, Symbol, symbols)
# 从 sympy.functions.elementary.hyperbolic 模块中导入双曲函数 cosh, coth, csch, sech, sinh, tanh
from sympy.functions.elementary.hyperbolic import (cosh, coth, csch, sech, sinh, tanh)
# 从 sympy.functions.elementary.miscellaneous 模块中导入根函数 root, 平方根函数 sqrt
from sympy.functions.elementary.miscellaneous import (root, sqrt)
# 从 sympy.functions.elementary.trigonometric 模块中导入三角函数 cos, cot, csc, sec, sin, tan
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
# 从 sympy.simplify.powsimp 模块中导入 powsimp 函数
from sympy.simplify.powsimp import powsimp
# 从 sympy.simplify.fu 模块中导入多个函数和变量
from sympy.simplify.fu import (
    L, TR1, TR10, TR10i, TR11, _TR11, TR12, TR12i, TR13, TR14, TR15, TR16,
    TR111, TR2, TR2i, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TRmorrie, _TR56 as T,
    TRpower, hyper_as_trig, fu, process_common_addends, trig_split,
    as_f_sign_1)
# 从 sympy.core.random 模块中导入 verify_numerically 函数
from sympy.core.random import verify_numerically
# 从 sympy.abc 模块中导入符号 a, b, c, x, y, z
from sympy.abc import a, b, c, x, y, z

# 定义一个测试函数 test_TR1，验证 TR1 函数的行为
def test_TR1():
    # 断言 TR1(2*csc(x) + sec(x)) 的结果等于 1/cos(x) + 2/sin(x)
    assert TR1(2*csc(x) + sec(x)) == 1/cos(x) + 2/sin(x)

# 定义一个测试函数 test_TR2，验证 TR2 函数的行为
def test_TR2():
    # 断言 TR2(tan(x)) 的结果等于 sin(x)/cos(x)
    assert TR2(tan(x)) == sin(x)/cos(x)
    # 断言 TR2(cot(x)) 的结果等于 cos(x)/sin(x)
    assert TR2(cot(x)) == cos(x)/sin(x)
    # 断言 TR2(tan(tan(x) - sin(x)/cos(x))) 的结果等于 0
    assert TR2(tan(tan(x) - sin(x)/cos(x))) == 0

# 定义一个测试函数 test_TR2i，验证 TR2i 函数的行为
def test_TR2i():
    # 测试对于指数的比率只有在分子和分母都是正数基数或整数指数时才能简化的提醒
    # 例如，在 y=-1，x=1/2 时，给出 sqrt(2)*I != -sqrt(2)*I 的结果
    assert powsimp(2**x/y**x) != (2/y)**x

    # 断言 TR2i(sin(x)/cos(x)) 的结果等于 tan(x)
    assert TR2i(sin(x)/cos(x)) == tan(x)
    # 断言 TR2i(sin(x)*sin(y)/cos(x)) 的结果等于 tan(x)*sin(y)
    assert TR2i(sin(x)*sin(y)/cos(x)) == tan(x)*sin(y)
    # 断言 TR2i(1/(sin(x)/cos(x))) 的结果等于 1/tan(x)
    assert TR2i(1/(sin(x)/cos(x))) == 1/tan(x)
    # 断言 TR2i(1/(sin(x)*sin(y)/cos(x))) 的结果等于 1/tan(x)/sin(y)
    assert TR2i(1/(sin(x)*sin(y)/cos(x))) == 1/tan(x)/sin(y)
    # 断言 TR2i(sin(x)/2/(cos(x) + 1)) 的结果等于 sin(x)/(cos(x) + 1)/2
    assert TR2i(sin(x)/2/(cos(x) + 1)) == sin(x)/(cos(x) + 1)/2

    # 断言 TR2i(sin(x)/2/(cos(x) + 1), half=True) 的结果等于 tan(x/2)/2
    assert TR2i(sin(x)/2/(cos(x) + 1), half=True) == tan(x/2)/2
    # 断言 TR2i(sin(1)/(cos(1) + 1), half=True) 的结果等于 tan(S.Half)
    assert TR2i(sin(1)/(cos(1) + 1), half=True) == tan(S.Half)
    # 断言 TR2i(sin(2)/(cos(2) + 1), half=True) 的结果等于 tan(1)
    assert TR2i(sin(2)/(cos(2) + 1), half=True) == tan(1)
    # 断言 TR2i(sin(4)/(cos(4) + 1), half=True) 的结果等于 tan(2)
    assert TR2i(sin(4)/(cos(4) + 1), half=True) == tan(2)
    # 断言 TR2i(sin(5)/(cos(5) + 1), half=True) 的结果等于 tan(5*S.Half)
    assert TR2i(sin(5)/(cos(5) + 1), half=True) == tan(5*S.Half)
    # 断言 TR2i((cos(1) + 1)/sin(1), half=True) 的结果等于 1/tan(S.Half)
    assert TR2i((cos(1) + 1)/sin(1), half=True) == 1/tan(S.Half)
    # 断言 TR2i((cos(2) + 1)/sin(2), half=True) 的结果等于 1/tan(1)
    assert TR2i((cos(2) + 1)/sin(2), half=True) == 1/tan(1)
    # 断言 TR2i((cos(4) + 1)/sin(4), half=True) 的结果等于 1/tan(2)
    assert TR2i((cos(4) + 1)/sin(4), half=True) == 1/tan(2)
    # 断言 TR2i((cos(5) + 1)/sin(5), half=True) 的结果等于 1/tan(5*S.Half)
    assert TR2i((cos(5) + 1)/sin(5), half=True) == 1/tan(5*S.Half)
    # 断言 TR2i((cos(1) + 1)**(-a)*sin(1)**a, half=True) 的结果等于 tan(S.Half)**a
    assert TR2i((cos(1) + 1)**(-a)*sin(1)**a, half=True) == tan(S.Half)**a
    # 断言 TR2i((cos(2) + 1)**(-a)*sin(2)**a, half=True) 的结果等于 tan(1)**a
    assert TR2i((cos(2) + 1)**(-a)*sin(2)**a, half=True) == tan(1)**a
    # 断言 TR2i((cos(4) + 1)**(-a)*sin(4)**a, half=True) 的结果等于 (cos(4) + 1)**(-a)*sin(4)**a
    assert TR2i((cos(4) + 1)**(-a)*sin(4)**a, half=True) == (cos(4) + 1)**(-a)*sin(4)**a
    # 断言 TR2i((cos(5) + 1)**
    # 断言：验证 TR2i 函数对给定表达式的处理结果是否与预期相符
    assert TR2i(((cos(5) + 1)**i * sin(5)**(-i)), half=True) == tan(5*S.Half)**(-i)
    
    # 断言：验证 TR2i 函数对给定表达式的处理结果是否与预期相符
    assert TR2i(1 / ((cos(5) + 1)**i * sin(5)**(-i)), half=True) == tan(5*S.Half)**i
# 定义测试函数 test_TR3，用于测试 TR3 函数的功能
def test_TR3():
    # 断言 TR3(cos(y - x*(y - x))) 的返回值应为 cos(x*(x - y) + y)
    assert TR3(cos(y - x*(y - x))) == cos(x*(x - y) + y)
    # 断言 cos(pi/2 + x) 的返回值应为 -sin(x)
    assert cos(pi/2 + x) == -sin(x)
    # 断言 cos(30*pi/2 + x) 的返回值应为 -cos(x)
    assert cos(30*pi/2 + x) == -cos(x)

    # 对于函数列表中的每个函数 f，执行以下操作
    for f in (cos, sin, tan, cot, csc, sec):
        # 计算 f(pi*3/7)，并将结果赋给 i
        i = f(pi*Rational(3, 7))
        # 对 i 应用 TR3 函数，将结果赋给 j
        j = TR3(i)
        # 断言数值验证函数 verify_numerically(i, j) 返回 True，并且 i 的类型不等于 j 的类型
        assert verify_numerically(i, j) and i.func != j.func

    # 关闭求值功能，执行以下操作
    with evaluate(False):
        # 计算 cos(9*pi/22)，将结果赋给 eq
        eq = cos(9*pi/22)
    # 断言 eq 包含 9*pi，并且 TR3(eq) 的返回值应为 sin(pi/11)
    assert eq.has(9*pi) and TR3(eq) == sin(pi/11)


# 定义测试函数 test_TR4，用于测试 TR4 函数的功能
def test_TR4():
    # 对于列表中的每个角度值 i，执行以下操作
    for i in [0, pi/6, pi/4, pi/3, pi/2]:
        # 关闭求值功能，执行以下操作
        with evaluate(False):
            # 计算 cos(i)，将结果赋给 eq
            eq = cos(i)
        # 断言 eq 的类型为 cos，并且 TR4(eq) 的返回值应为 cos(i)
        assert isinstance(eq, cos) and TR4(eq) == cos(i)


# 定义测试函数 test__TR56，用于测试 T 函数的功能
def test__TR56():
    # 定义匿名函数 h(x)，计算 1 - x
    h = lambda x: 1 - x
    # 断言 T(sin(x)**3, sin, cos, h, 4, False) 的返回值应为 sin(x)*(-cos(x)**2 + 1)
    assert T(sin(x)**3, sin, cos, h, 4, False) == sin(x)*(-cos(x)**2 + 1)
    # 断言 T(sin(x)**10, sin, cos, h, 4, False) 的返回值应为 sin(x)**10
    assert T(sin(x)**10, sin, cos, h, 4, False) == sin(x)**10
    # 断言 T(sin(x)**6, sin, cos, h, 6, False) 的返回值应为 (-cos(x)**2 + 1)**3
    assert T(sin(x)**6, sin, cos, h, 6, False) == (-cos(x)**2 + 1)**3
    # 断言 T(sin(x)**6, sin, cos, h, 6, True) 的返回值应为 sin(x)**6
    assert T(sin(x)**6, sin, cos, h, 6, True) == sin(x)**6
    # 断言 T(sin(x)**8, sin, cos, h, 10, True) 的返回值应为 (-cos(x)**2 + 1)**4
    assert T(sin(x)**8, sin, cos, h, 10, True) == (-cos(x)**2 + 1)**4

    # issue 17137
    # 断言 T(sin(x)**I, sin, cos, h, 4, True) 的返回值应为 sin(x)**I
    assert T(sin(x)**I, sin, cos, h, 4, True) == sin(x)**I
    # 断言 T(sin(x)**(2*I + 1), sin, cos, h, 4, True) 的返回值应为 sin(x)**(2*I + 1)
    assert T(sin(x)**(2*I + 1), sin, cos, h, 4, True) == sin(x)**(2*I + 1)


# 定义测试函数 test_TR5，用于测试 TR5 函数的功能
def test_TR5():
    # 断言 TR5(sin(x)**2) 的返回值应为 -cos(x)**2 + 1
    assert TR5(sin(x)**2) == -cos(x)**2 + 1
    # 断言 TR5(sin(x)**-2) 的返回值应为 sin(x)**(-2)
    assert TR5(sin(x)**-2) == sin(x)**(-2)
    # 断言 TR5(sin(x)**4) 的返回值应为 (-cos(x)**2 + 1)**2
    assert TR5(sin(x)**4) == (-cos(x)**2 + 1)**2


# 定义测试函数 test_TR6，用于测试 TR6 函数的功能
def test_TR6():
    # 断言 TR6(cos(x)**2) 的返回值应为 -sin(x)**2 + 1
    assert TR6(cos(x)**2) == -sin(x)**2 + 1
    # 断言 TR6(cos(x)**-2) 的返回值应为 cos(x)**(-2)
    assert TR6(cos(x)**-2) == cos(x)**(-2)
    # 断言 TR6(cos(x)**4) 的返回值应为 (-sin(x)**2 + 1)**2
    assert TR6(cos(x)**4) == (-sin(x)**2 + 1)**2


# 定义测试函数 test_TR7，用于测试 TR7 函数的功能
def test_TR7():
    # 断言 TR7(cos(x)**2) 的返回值应为 cos(2*x)/2 + S.Half
    assert TR7(cos(x)**2) == cos(2*x)/2 + S.Half
    # 断言 TR7(cos(x)**2 + 1) 的返回值应为 cos(2*x)/2 + Rational(3, 2)
    assert TR7(cos(x)**2 + 1) == cos(2*x)/2 + Rational(3, 2)


# 定义测试函数 test_TR8，用于测试 TR8 函数的功能
def test_TR8():
    # 断言 TR8(cos(2)*cos(3)) 的返回值应为 cos(5)/2 + cos(1)/2
    assert TR8(cos(2)*cos(3)) == cos(5)/2 + cos(1)/2
    # 断言 TR8(cos(2)*sin(3)) 的返回值应为 sin(5)/2 + sin(1)/2
    assert TR8(cos(2)*sin(3)) == sin(5)/2 + sin(1)/2
    # 断言 TR8(sin(2)*sin(3)) 的返回值应为 -cos(5)/2 + cos(1)/2
    assert TR8(sin(2)*sin(3)) == -cos(5)/2 + cos(1)/2
    # 断言 TR8(sin(1)*sin(2)*sin(3)) 的返回值应为 sin(4)/4 - sin(6)/4 + sin(2)/4
    assert TR8(sin(1)*sin(2)*sin(3)) == sin(4)/4 - sin(6)/4 + sin(2)/4
    # 断言 TR8(cos(2)*cos(3)*cos(4)*cos(5)) 的返回值应为 cos(4)/4 + cos(10)/8 + cos(2)/8 + cos(8)/8 + cos(14)/8 + cos(6)/8 + Rational(1, 8)
    assert TR8(cos(2)*cos(3)*cos(4)*cos(5)) == \
        cos(4)/4 + cos(10)/8 + cos(2)/8 + cos(8)/8 + cos(14)/8 + \
        cos(6)/8 + Rational(1, 8)
    # 断言 TR8(cos(2)*cos(3)*cos(4)*cos(5)*cos(6)) 的返回值应为 cos(10)/8 + cos(4)/8 + 3*cos(2)/16 + cos(16)/16 + cos(8)/8 + cos(14)/16 + cos(20)/16 + cos(12)/16 + Rational(1, 16) + cos(6)/8
    assert TR8(cos(2)*cos(3)*cos(4)*cos(5)*cos(6)) == \
    # 断言一个特定的等式成立
    assert TR9(-sin(y) + sin(x*y)) == 2*sin(x*y/2 - y/2)*cos(x*y/2 + y/2)
    
    # 计算余弦和正弦函数的值，并赋给变量 c 和 s
    c = cos(x)
    s = sin(x)
    
    # 遍历四组不同的符号和函数对
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for a in ((c, s), (s, c), (cos(x), cos(x*y)), (sin(x), sin(x*y))):
            # 将当前的符号对和函数对组合成一个可迭代对象 args
            args = zip(si, a)
            # 构建一个表达式 ex，通过乘法运算符合并每个符号和函数的乘积
            ex = Add(*[Mul(*ai) for ai in args])
            # 对 ex 应用 TR9 函数，返回结果赋给变量 t
            t = TR9(ex)
            
            # 断言以下条件不成立，如果成立则抛出异常
            assert not (
                # 如果函数的类型相同且数值验证不通过，或者 t 是一个加法表达式
                a[0].func == a[1].func and (
                    not verify_numerically(ex, t.expand(trig=True)) or t.is_Add
                ) or
                # 如果函数的类型不同且 ex 不等于 t
                a[1].func != a[0].func and ex != t
            )
# 定义一个名为 test_TR10 的函数，用于测试 TR10 函数的功能
def test_TR10():
    # 断言：TR10 函数应用于 cos(a + b)，应该返回 -sin(a)*sin(b) + cos(a)*cos(b)
    assert TR10(cos(a + b)) == -sin(a)*sin(b) + cos(a)*cos(b)
    # 断言：TR10 函数应用于 sin(a + b)，应该返回 sin(a)*cos(b) + sin(b)*cos(a)
    assert TR10(sin(a + b)) == sin(a)*cos(b) + sin(b)*cos(a)
    # 断言：TR10 函数应用于 sin(a + b + c)，应该返回复杂的三角函数表达式
    assert TR10(sin(a + b + c)) == \
        (-sin(a)*sin(b) + cos(a)*cos(b))*sin(c) + \
        (sin(a)*cos(b) + sin(b)*cos(a))*cos(c)
    # 断言：TR10 函数应用于 cos(a + b + c)，应该返回复杂的三角函数表达式
    assert TR10(cos(a + b + c)) == \
        (-sin(a)*sin(b) + cos(a)*cos(b))*cos(c) - \
        (sin(a)*cos(b) + sin(b)*cos(a))*sin(c)


# 定义一个名为 test_TR10i 的函数，用于测试 TR10i 函数的功能
def test_TR10i():
    # 断言：TR10i 函数应用于 cos(1)*cos(3) + sin(1)*sin(3)，应该返回 cos(2)
    assert TR10i(cos(1)*cos(3) + sin(1)*sin(3)) == cos(2)
    # 断言：TR10i 函数应用于 cos(1)*cos(3) - sin(1)*sin(3)，应该返回 cos(4)
    assert TR10i(cos(1)*cos(3) - sin(1)*sin(3)) == cos(4)
    # 断言：TR10i 函数应用于 cos(1)*sin(3) - sin(1)*cos(3)，应该返回 sin(2)
    assert TR10i(cos(1)*sin(3) - sin(1)*cos(3)) == sin(2)
    # 断言：TR10i 函数应用于 cos(1)*sin(3) + sin(1)*cos(3)，应该返回 sin(4)
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3)) == sin(4)
    # 断言：TR10i 函数应用于 cos(1)*sin(3) + sin(1)*cos(3) + 7，应该返回 sin(4) + 7
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3) + 7) == sin(4) + 7
    # 断言：TR10i 函数应用于 cos(1)*sin(3) + sin(1)*cos(3) + cos(3)，应该返回 cos(3) + sin(4)
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3) + cos(3)) == cos(3) + sin(4)
    # 断言：TR10i 函数应用于 2*cos(1)*sin(3) + 2*sin(1)*cos(3) + cos(3)，应该返回 2*sin(4) + cos(3)
    assert TR10i(2*cos(1)*sin(3) + 2*sin(1)*cos(3) + cos(3)) == \
        2*sin(4) + cos(3)
    # 断言：TR10i 函数应用于复杂的三角函数表达式，应该返回 cos(1)
    assert TR10i(cos(2)*cos(3) + sin(2)*(cos(1)*sin(2) + cos(2)*sin(1))) == \
        cos(1)
    # 创建一个复杂的等式表达式
    eq = (cos(2)*cos(3) + sin(2)*(
        cos(1)*sin(2) + cos(2)*sin(1)))*cos(5) + sin(1)*sin(5)
    # 断言：TR10i 函数应用于复杂等式及其展开形式，应该返回 cos(4)
    assert TR10i(eq) == TR10i(eq.expand()) == cos(4)
    # 断言：TR10i 函数应用于 sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x，应该返回 2*sqrt(2)*x*sin(x + pi/6)
    assert TR10i(sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x) == \
        2*sqrt(2)*x*sin(x + pi/6)
    # 断言：TR10i 函数应用于复杂的三角函数表达式，应该返回 4*sqrt(6)*sin(x + pi/6)/9
    assert TR10i(cos(x)/sqrt(6) + sin(x)/sqrt(2) +
            cos(x)/sqrt(6)/3 + sin(x)/sqrt(2)/3) == 4*sqrt(6)*sin(x + pi/6)/9
    # 断言：TR10i 函数应用于复杂的三角函数表达式，应该返回带有不同变量的结果
    assert TR10i(cos(x)/sqrt(6) + sin(x)/sqrt(2) +
            cos(y)/sqrt(6)/3 + sin(y)/sqrt(2)/3) == \
        sqrt(6)*sin(x + pi/6)/3 + sqrt(6)*sin(y + pi/6)/9
    # 断言：TR10i 函数应用于 cos(x) + sqrt(3)*sin(x) + 2*sqrt(3)*cos(x + pi/6)，应该返回 4*cos(x)
    assert TR10i(cos(x) + sqrt(3)*sin(x) + 2*sqrt(3)*cos(x + pi/6)) == 4*cos(x)
    # 断言：TR10i 函数应用于 cos(x) + sqrt(3)*sin(x) + 2*sqrt(3)*cos(x + pi/6) + 4*sin(x)，应该返回 4*sqrt(2)*sin(x + pi/4)
    assert TR10i(cos(x) + sqrt(3)*sin(x) +
            2*sqrt(3)*cos(x + pi/6) + 4*sin(x)) == 4*sqrt(2)*sin(x + pi/4)
    # 断言：TR10i 函数应用于 cos(2)*sin(3) + sin(2)*cos(4)，应该返回 sin(2)*cos(4) + sin(3)*cos(2)
    assert TR10i(cos(2)*sin(3) + sin(2)*cos(4)) == \
        sin(2)*cos(4) + sin(3)*cos(2)

    # 创建一个非交换符号 A
    A = Symbol('A', commutative=False)
    # 断言：TR10i 函数应用于 sqrt(2)*cos(x)*A + sqrt(6)*sin(x)*A，应该返回 2*sqrt(2)*sin(x + pi/6)*A
    assert TR10i(sqrt(2)*cos(x)*A + sqrt(6)*sin(x)*A) == \
        2*sqrt(2)*sin(x + pi/6)*A

    # 初始化一些三角函数表达式的变量
    c = cos(x)
    s = sin(x)
    h = sin(y)
    r = cos(y)
    # 遍历 si 和 argsi 的组合
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for argsi in ((c*r, s*h), (c*h, s*r)): # 显式的两个参数
            args = zip(si, argsi)
            # 创建一个表达式 ex
            ex = Add(*[Mul(*ai) for ai in args])
            # 计算 TR10i 函数应用于 ex 的结果
            t = TR10i(ex)
            # 断言：ex 与其展开形式的差值为零，或者 t 不是一个加法表达式
            assert not (ex - t.expand(trig=True) or t.is_Add)

    # 再次初始化三角函数表达式的变量，这次使用固定角度值
    c = cos(x)
    s = sin(x)
    h = sin(pi/6)
    r = cos(pi/6)
    #
    # 断言语句，验证 TR11 函数对给定表达式的计算结果是否符合预期
    assert TR11(cos(4*x)) == \
        (-sin(x)**2 + cos(x)**2)**2 - 4*sin(x)**2*cos(x)**2

    # 断言语句，验证 TR11 函数对给定值的计算结果是否符合预期
    assert TR11(cos(2)) == cos(2)

    # 断言语句，验证 TR11 函数对给定表达式和参数的计算结果是否符合预期
    assert TR11(cos(pi*Rational(3, 7)), pi*Rational(2, 7)) == -cos(pi*Rational(2, 7))**2 + sin(pi*Rational(2, 7))**2
    
    # 断言语句，验证 TR11 函数对给定值和参数的计算结果是否符合预期
    assert TR11(cos(4), 2) == -sin(2)**2 + cos(2)**2
    
    # 断言语句，验证 TR11 函数对给定值和参数的计算结果是否符合预期
    assert TR11(cos(6), 2) == cos(6)
    
    # 断言语句，验证 TR11 函数对给定表达式和参数的计算结果是否符合预期
    assert TR11(sin(x)/cos(x/2), x/2) == 2*sin(x/2)
def test__TR11():

    # 断言：_TR11 函数对表达式 sin(x/3)*sin(2*x)*sin(x/4)/(cos(x/6)*cos(x/8)) 进行转换
    assert _TR11(sin(x/3)*sin(2*x)*sin(x/4)/(cos(x/6)*cos(x/8))) == \
        4*sin(x/8)*sin(x/6)*sin(2*x),_TR11(sin(x/3)*sin(2*x)*sin(x/4)/(cos(x/6)*cos(x/8)))
    
    # 断言：_TR11 函数对表达式 sin(x/3)/cos(x/6) 进行转换
    assert _TR11(sin(x/3)/cos(x/6)) == 2*sin(x/6)

    # 断言：_TR11 函数对表达式 cos(x/6)/sin(x/3) 进行转换
    assert _TR11(cos(x/6)/sin(x/3)) == 1/(2*sin(x/6))
    
    # 断言：_TR11 函数对表达式 sin(2*x)*cos(x/8)/sin(x/4) 进行转换
    assert _TR11(sin(2*x)*cos(x/8)/sin(x/4)) == sin(2*x)/(2*sin(x/8)), _TR11(sin(2*x)*cos(x/8)/sin(x/4))
    
    # 断言：_TR11 函数对表达式 sin(x)/sin(x/2) 进行转换
    assert _TR11(sin(x)/sin(x/2)) == 2*cos(x/2)


def test_TR12():
    
    # 断言：TR12 函数对 tan(x + y) 进行转换
    assert TR12(tan(x + y)) == (tan(x) + tan(y))/(-tan(x)*tan(y) + 1)
    
    # 断言：TR12 函数对 tan(x + y + z) 进行转换
    assert TR12(tan(x + y + z)) ==\
        (tan(z) + (tan(x) + tan(y))/(-tan(x)*tan(y) + 1))/(
        1 - (tan(x) + tan(y))*tan(z)/(-tan(x)*tan(y) + 1))
    
    # 断言：TR12 函数对 tan(x*y) 进行转换
    assert TR12(tan(x*y)) == tan(x*y)


def test_TR13():
    
    # 断言：TR13 函数对 tan(3)*tan(2) 进行转换
    assert TR13(tan(3)*tan(2)) == -tan(2)/tan(5) - tan(3)/tan(5) + 1
    
    # 断言：TR13 函数对 cot(3)*cot(2) 进行转换
    assert TR13(cot(3)*cot(2)) == 1 + cot(3)*cot(5) + cot(2)*cot(5)
    
    # 断言：TR13 函数对 tan(1)*tan(2)*tan(3) 进行转换
    assert TR13(tan(1)*tan(2)*tan(3)) == \
        (-tan(2)/tan(5) - tan(3)/tan(5) + 1)*tan(1)
    
    # 断言：TR13 函数对 tan(1)*tan(2)*cot(3) 进行转换
    assert TR13(tan(1)*tan(2)*cot(3)) == \
        (-tan(2)/tan(3) + 1 - tan(1)/tan(3))*cot(3)


def test_L():
    
    # 断言：L 函数对 cos(x) + sin(x) 进行转换
    assert L(cos(x) + sin(x)) == 2


def test_fu():

    # 断言：fu 函数对 sin(50)**2 + cos(50)**2 + sin(pi/6) 进行转换
    assert fu(sin(50)**2 + cos(50)**2 + sin(pi/6)) == Rational(3, 2)
    
    # 断言：fu 函数对 sqrt(6)*cos(x) + sqrt(2)*sin(x) 进行转换
    assert fu(sqrt(6)*cos(x) + sqrt(2)*sin(x)) == 2*sqrt(2)*sin(x + pi/3)

    # 创建一个表达式
    eq = sin(x)**4 - cos(y)**2 + sin(y)**2 + 2*cos(x)**2
    # 断言：fu 函数对创建的表达式 eq 进行转换
    assert fu(eq) == cos(x)**4 - 2*cos(y)**2 + 2

    # 断言：fu 函数对 S.Half - cos(2*x)/2 进行转换
    assert fu(S.Half - cos(2*x)/2) == sin(x)**2

    # 断言：fu 函数对 sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b)) 进行转换
    assert fu(sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b))) == \
        sqrt(2)*sin(a + b + pi/4)

    # 断言：fu 函数对 sqrt(3)*cos(x)/2 + sin(x)/2 进行转换
    assert fu(sqrt(3)*cos(x)/2 + sin(x)/2) == sin(x + pi/3)

    # 断言：fu 函数对 1 - sin(2*x)**2/4 - sin(y)**2 - cos(x)**4 进行转换
    assert fu(1 - sin(2*x)**2/4 - sin(y)**2 - cos(x)**4) == \
        -cos(x)**2 + cos(y)**2

    # 断言：fu 函数对 cos(pi*Rational(4, 9)) 进行转换
    assert fu(cos(pi*Rational(4, 9))) == sin(pi/18)
    
    # 断言：fu 函数对 cos(pi/9)*cos(pi*Rational(2, 9))*cos(pi*Rational(3, 9))*cos(pi*Rational(4, 9)) 进行转换
    assert fu(cos(pi/9)*cos(pi*Rational(2, 9))*cos(pi*Rational(3, 9))*cos(pi*Rational(4, 9))) == Rational(1, 16)

    # 断言：fu 函数对 tan(pi*Rational(7, 18)) + tan(pi*Rational(5, 18)) - sqrt(3)*tan(pi*Rational(5, 18))*tan(pi*Rational(7, 18)) 进行转换
    assert fu(
        tan(pi*Rational(7, 18)) + tan(pi*Rational(5, 18)) - sqrt(3)*tan(pi*Rational(5, 18))*tan(pi*Rational(7, 18))) == \
        -sqrt(3)

    # 断言：fu 函数对 tan(1)*tan(2) 进行转换
    assert fu(tan(1)*tan(2)) == tan(1)*tan(2)

    # 创建一个表达式
    expr = Mul(*[cos(2**i) for i in range(10)])
    # 断言：fu 函数对创建的表达式 expr 进行转换
    assert fu(expr) == sin(1024)/(1024*sin(1))

    # issue #18059:
    # 断言：fu 函数对 cos(x) + sqrt(sin(x)**2) 进行转换
    assert fu(cos(x) + sqrt(sin(x)**2)) == cos(x) + sqrt(sin(x)**2)

    # 断言：fu 函数对 (-14*sin(x)**3 + 35*sin(x) + 6*sqrt(3)*cos(x)**3 + 9*sqrt(3)*cos(x))/((cos(2*x) + 4)) 进行转换
    assert fu((-14*sin(x)**3 + 35*sin(x) + 6*sqrt(3)*cos(x)**3 + 9*sqrt(3)*cos(x))/((cos(2*x) + 4))) == \
        7*sin(x) + 3*sqrt(3)*cos(x)


def test_objective():
    
    # 断言：fu 函数对 sin(x)/cos(x) 进行转换，并使用 count_ops() 函数作为度量函数
    assert fu(sin(x)/cos(x), measure=lambda x: x.count_ops()) == \
            tan(x)
    
    # 断言：fu 函数对 sin(x)/cos(x) 进行转换，并使用 -x.count_ops() 函数作为度量函数
    assert fu(sin(x)/cos(x), measure=lambda x: -x.count_ops()) == \
            sin(x)/cos(x)


def test_process_common_addends():
    # this tests that the args are
    # 检查 trig_split 函数对 cos(x), cos(y) 的返回结果是否等于 (1, 1, 1, x, y, True)
    assert trig_split(cos(x), cos(y)) == (1, 1, 1, x, y, True)
    
    # 检查 trig_split 函数对 2*cos(x), -2*cos(y) 的返回结果是否等于 (2, 1, -1, x, y, True)
    assert trig_split(2*cos(x), -2*cos(y)) == (2, 1, -1, x, y, True)
    
    # 检查 trig_split 函数对 cos(x)*sin(y), cos(y)*sin(y) 的返回结果是否等于 (sin(y), 1, 1, x, y, True)
    assert trig_split(cos(x)*sin(y), cos(y)*sin(y)) == (sin(y), 1, 1, x, y, True)
    
    # 检查 trig_split 函数对 cos(x), -sqrt(3)*sin(x) 的返回结果是否等于 (2, 1, -1, x, pi/6, False)，并且传入参数 two=True
    assert trig_split(cos(x), -sqrt(3)*sin(x), two=True) == (2, 1, -1, x, pi/6, False)
    
    # 检查 trig_split 函数对 cos(x), sin(x) 的返回结果是否等于 (sqrt(2), 1, 1, x, pi/4, False)，并且传入参数 two=True
    assert trig_split(cos(x), sin(x), two=True) == (sqrt(2), 1, 1, x, pi/4, False)
    
    # 检查 trig_split 函数对 cos(x), -sin(x) 的返回结果是否等于 (sqrt(2), 1, -1, x, pi/4, False)，并且传入参数 two=True
    assert trig_split(cos(x), -sin(x), two=True) == (sqrt(2), 1, -1, x, pi/4, False)
    
    # 检查 trig_split 函数对 sqrt(2)*cos(x), -sqrt(6)*sin(x) 的返回结果是否等于 (2*sqrt(2), 1, -1, x, pi/6, False)，并且传入参数 two=True
    assert trig_split(sqrt(2)*cos(x), -sqrt(6)*sin(x), two=True) == (2*sqrt(2), 1, -1, x, pi/6, False)
    
    # 检查 trig_split 函数对 -sqrt(6)*cos(x), -sqrt(2)*sin(x) 的返回结果是否等于 (-2*sqrt(2), 1, 1, x, pi/3, False)，并且传入参数 two=True
    assert trig_split(-sqrt(6)*cos(x), -sqrt(2)*sin(x), two=True) == (-2*sqrt(2), 1, 1, x, pi/3, False)
    
    # 检查 trig_split 函数对 cos(x)/sqrt(6), sin(x)/sqrt(2) 的返回结果是否等于 (sqrt(6)/3, 1, 1, x, pi/6, False)，并且传入参数 two=True
    assert trig_split(cos(x)/sqrt(6), sin(x)/sqrt(2), two=True) == (sqrt(6)/3, 1, 1, x, pi/6, False)
    
    # 检查 trig_split 函数对 -sqrt(6)*cos(x)*sin(y), -sqrt(2)*sin(x)*sin(y) 的返回结果是否等于 (-2*sqrt(2)*sin(y), 1, 1, x, pi/3, False)，并且传入参数 two=True
    assert trig_split(-sqrt(6)*cos(x)*sin(y), -sqrt(2)*sin(x)*sin(y), two=True) == (-2*sqrt(2)*sin(y), 1, 1, x, pi/3, False)
    
    # 检查 trig_split 函数对 cos(x), sin(x) 的返回结果是否为 None
    assert trig_split(cos(x), sin(x)) is None
    
    # 检查 trig_split 函数对 cos(x), sin(z) 的返回结果是否为 None
    assert trig_split(cos(x), sin(z)) is None
    
    # 检查 trig_split 函数对 2*cos(x), -sin(x) 的返回结果是否为 None
    assert trig_split(2*cos(x), -sin(x)) is None
    
    # 检查 trig_split 函数对 cos(x), -sqrt(3)*sin(x) 的返回结果是否为 None
    assert trig_split(cos(x), -sqrt(3)*sin(x)) is None
    
    # 检查 trig_split 函数对 cos(x)*cos(y), sin(x)*sin(z) 的返回结果是否为 None
    assert trig_split(cos(x)*cos(y), sin(x)*sin(z)) is None
    
    # 检查 trig_split 函数对 cos(x)*cos(y), sin(x)*sin(y) 的返回结果是否为 None
    assert trig_split(cos(x)*cos(y), sin(x)*sin(y)) is None
    
    # 检查 trig_split 函数对 -sqrt(6)*cos(x), sqrt(2)*sin(x)*sin(y) 的返回结果是否为 None，并且传入参数 two=True
    assert trig_split(-sqrt(6)*cos(x), sqrt(2)*sin(x)*sin(y), two=True) is None
    
    # 检查 trig_split 函数对 sqrt(3)*sqrt(x), cos(3) 的返回结果是否为 None，并且传入参数 two=True
    assert trig_split(sqrt(3)*sqrt(x), cos(3), two=True) is None
    
    # 检查 trig_split 函数对 sqrt(3)*root(x, 3), sin(3)*cos(2) 的返回结果是否为 None，并且传入参数 two=True
    assert trig_split(sqrt(3)*root(x, 3), sin(3)*cos(2), two=True) is None
    
    # 检查 trig_split 函数对 cos(5)*cos(6), cos(7)*sin(5) 的返回结果是否为 None，并且传入参数 two=True
    assert trig_split(cos(5)*cos(6), cos(7)*sin(5), two=True) is None
# 定义名为 test_TRmorrie 的测试函数
def test_TRmorrie():
    # 断言调用 TRmorrie 函数返回的结果与预期相等
    assert TRmorrie(7*Mul(*[cos(i) for i in range(10)])) == \
        7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))
    # 断言调用 TRmorrie 函数返回的结果与参数 x 相等
    assert TRmorrie(x) == x
    # 断言调用 TRmorrie 函数返回的结果与参数 2*x 相等
    assert TRmorrie(2*x) == 2*x
    # 计算表达式 e 的值
    e = cos(pi/7)*cos(pi*Rational(2, 7))*cos(pi*Rational(4, 7))
    # 断言调用 TR8 函数对 TRmorrie 函数返回结果的计算结果与预期结果相等
    assert TR8(TRmorrie(e)) == Rational(-1, 8)
    # 计算表达式 e 的值
    e = Mul(*[cos(2**i*pi/17) for i in range(1, 17)])
    # 断言调用 TR8 和 TR3 函数对 TRmorrie 函数返回结果的计算结果与预期结果相等
    assert TR8(TR3(TRmorrie(e))) == Rational(1, 65536)
    # issue 17063
    # 计算表达式 eq 的值
    eq = cos(x)/cos(x/2)
    # 断言调用 TRmorrie 函数返回结果与参数 eq 相等
    assert TRmorrie(eq) == eq
    # issue #20430
    # 计算表达式 eq 的值
    eq = cos(x/2)*sin(x/2)*cos(x)**3
    # 断言调用 TRmorrie 函数返回结果的计算结果与预期结果相等
    assert TRmorrie(eq) == sin(2*x)*cos(x)**2/4


# 定义名为 test_TRpower 的测试函数
def test_TRpower():
    # 断言调用 TRpower 函数返回的结果与预期相等
    assert TRpower(1/sin(x)**2) == 1/sin(x)**2
    # 断言调用 TRpower 函数返回的结果与预期相等
    assert TRpower(cos(x)**3*sin(x/2)**4) == \
        (3*cos(x)/4 + cos(3*x)/4)*(-cos(x)/2 + cos(2*x)/8 + Rational(3, 8))
    # 遍历 k 的范围
    for k in range(2, 8):
        # 断言数值验证，sin(x)**k 与 TRpower(sin(x)**k) 的结果相等
        assert verify_numerically(sin(x)**k, TRpower(sin(x)**k))
        # 断言数值验证，cos(x)**k 与 TRpower(cos(x)**k) 的结果相等
        assert verify_numerically(cos(x)**k, TRpower(cos(x)**k))


# 定义名为 test_hyper_as_trig 的测试函数
def test_hyper_as_trig():
    # 导入 _osborne 和 _osbornei 函数
    from sympy.simplify.fu import _osborne, _osbornei

    # 计算表达式 eq 的值
    eq = sinh(x)**2 + cosh(x)**2
    # 调用 hyper_as_trig 函数
    t, f = hyper_as_trig(eq)
    # 断言调用 f(fu(t)) 返回结果与预期相等
    assert f(fu(t)) == cosh(2*x)
    # 计算表达式 e 的值
    e, f = hyper_as_trig(tanh(x + y))
    # 断言调用 f(TR12(e)) 返回结果与预期相等
    assert f(TR12(e)) == (tanh(x) + tanh(y))/(tanh(x)*tanh(y) + 1)

    # 创建一个虚拟变量 d
    d = Dummy()
    # 断言调用 _osborne 函数的结果与预期相等
    assert _osborne(sinh(x), d) == I*sin(x*d)
    assert _osborne(tanh(x), d) == I*tan(x*d)
    assert _osborne(coth(x), d) == cot(x*d)/I
    assert _osborne(cosh(x), d) == cos(x*d)
    assert _osborne(sech(x), d) == sec(x*d)
    assert _osborne(csch(x), d) == csc(x*d)/I
    # 遍历函数列表，断言 _osbornei 函数应用于 _osborne 函数的结果与预期相等
    for func in (sinh, cosh, tanh, coth, sech, csch):
        h = func(pi)
        assert _osbornei(_osborne(h, d), d) == h
    # /!\ _osborne 函数不用于 o(i(trig, d), d) 方向，仅检查其功能正确性
    assert _osbornei(cos(x*y + z), y) == cosh(x + z*I)
    assert _osbornei(sin(x*y + z), y) == sinh(x + z*I)/I
    assert _osbornei(tan(x*y + z), y) == tanh(x + z*I)/I
    assert _osbornei(cot(x*y + z), y) == coth(x + z*I)*I
    assert _osbornei(sec(x*y + z), y) == sech(x + z*I)
    assert _osbornei(csc(x*y + z), y) == csch(x + z*I)*I


# 定义名为 test_TR12i 的测试函数
def test_TR12i():
    # 定义 ta, tb, tc 作为 tan(a), tan(b), tan(c) 的别名
    ta, tb, tc = [tan(i) for i in (a, b, c)]
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i((ta + tb)/(-ta*tb + 1)) == tan(a + b)
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i((ta + tb)/(ta*tb - 1)) == -tan(a + b)
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i((-ta - tb)/(ta*tb - 1)) == tan(a + b)
    # 计算表达式 eq 的值
    eq = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i(eq.expand()) == \
        -3*tan(a + b)*tan(a + c)/(tan(a) + tan(b) - 1)/2
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i(tan(x)/sin(x)) == tan(x)/sin(x)
    # 计算表达式 eq 的值
    eq = (ta + cos(2))/(-ta*tb + 1)
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i(eq) == eq
    # 计算表达式 eq 的值
    eq = (ta + tb + 2)**2/(-ta*tb + 1)
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i(eq) == eq
    # 计算表达式 eq 的值
    eq = ta/(-ta*tb + 1)
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i(eq) == eq
    # 计算表达式 eq 的值
    eq = (((ta + tb)*(a + 1)).expand())**2/(ta*tb - 1)
    # 断言调用 TR12i 函数返回的结果与预期相等
    assert TR12i(eq) == -(a + 1)**2*tan(a + b)


# 定义名为 test_TR14 的测试函数
def test_TR14():
    # 计算表达式 eq 的值
    eq = (cos(x) - 1)*(cos(x) + 1)
    # 计算表达式 ans 的值
    ans = -sin(x)**2
    # 断言调用 TR14 函数返回的结果与预期相等
    assert TR14(eq)
    assert TR14(1/eq) == 1/ans
    # 断言：使用 TR14 函数对 1/eq 进行转换后应该等于 1/ans

    assert TR14((cos(x) - 1)**2*(cos(x) + 1)**2) == ans**2
    # 断言：使用 TR14 函数对 ((cos(x) - 1)**2*(cos(x) + 1)**2) 进行转换后应该等于 ans 的平方

    assert TR14((cos(x) - 1)**2*(cos(x) + 1)**3) == ans**2*(cos(x) + 1)
    # 断言：使用 TR14 函数对 ((cos(x) - 1)**2*(cos(x) + 1)**3) 进行转换后应该等于 ans 的平方乘以 (cos(x) + 1)

    assert TR14((cos(x) - 1)**3*(cos(x) + 1)**2) == ans**2*(cos(x) - 1)
    # 断言：使用 TR14 函数对 ((cos(x) - 1)**3*(cos(x) + 1)**2) 进行转换后应该等于 ans 的平方乘以 (cos(x) - 1)

    eq = (cos(x) - 1)**y*(cos(x) + 1)**y
    # 定义变量 eq 为 (cos(x) - 1)**y*(cos(x) + 1)**y

    assert TR14(eq) == eq
    # 断言：使用 TR14 函数对 eq 进行转换后应该等于 eq 本身

    eq = (cos(x) - 2)**y*(cos(x) + 1)
    # 重新定义变量 eq 为 (cos(x) - 2)**y*(cos(x) + 1)

    assert TR14(eq) == eq
    # 断言：使用 TR14 函数对 eq 进行转换后应该等于 eq 本身

    eq = (tan(x) - 2)**2*(cos(x) + 1)
    # 重新定义变量 eq 为 (tan(x) - 2)**2*(cos(x) + 1)

    assert TR14(eq) == eq
    # 断言：使用 TR14 函数对 eq 进行转换后应该等于 eq 本身

    i = symbols('i', integer=True)
    # 定义符号 i，限定为整数

    assert TR14((cos(x) - 1)**i*(cos(x) + 1)**i) == ans**i
    # 断言：使用 TR14 函数对 ((cos(x) - 1)**i*(cos(x) + 1)**i) 进行转换后应该等于 ans 的 i 次方

    assert TR14((sin(x) - 1)**i*(sin(x) + 1)**i) == (-cos(x)**2)**i
    # 断言：使用 TR14 函数对 ((sin(x) - 1)**i*(sin(x) + 1)**i) 进行转换后应该等于 (-cos(x)**2) 的 i 次方

    # 可以在这种情况下使用提取
    eq = (cos(x) - 1)**(i + 1)*(cos(x) + 1)**i
    # 定义变量 eq 为 (cos(x) - 1)**(i + 1)*(cos(x) + 1)**i

    assert TR14(eq) in [(cos(x) - 1)*ans**i, eq]
    # 断言：使用 TR14 函数对 eq 进行转换后应该在 [(cos(x) - 1)*ans**i, eq] 中找到结果

    assert TR14((sin(x) - 1)*(sin(x) + 1)) == -cos(x)**2
    # 断言：使用 TR14 函数对 ((sin(x) - 1)*(sin(x) + 1)) 进行转换后应该等于 -cos(x)**2

    p1 = (cos(x) + 1)*(cos(x) - 1)
    # 定义变量 p1 为 (cos(x) + 1)*(cos(x) - 1)

    p2 = (cos(y) - 1)*2*(cos(y) + 1)
    # 定义变量 p2 为 (cos(y) - 1)*2*(cos(y) + 1)

    p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))
    # 定义变量 p3 为 (3*(cos(y) - 1))*(3*(cos(y) + 1))

    assert TR14(p1*p2*p3*(x - 1)) == -18*((x - 1)*sin(x)**2*sin(y)**4)
    # 断言：使用 TR14 函数对 p1*p2*p3*(x - 1) 进行转换后应该等于 -18*((x - 1)*sin(x)**2*sin(y)**4)
# 定义用于测试三角函数变换的函数
def test_TR15_16_17():
    # 断言：TR15 变换测试
    assert TR15(1 - 1/sin(x)**2) == -cot(x)**2
    # 断言：TR16 变换测试
    assert TR16(1 - 1/cos(x)**2) == -tan(x)**2
    # 断言：TR111 变换测试
    assert TR111(1 - 1/tan(x)**2) == 1 - cot(x)**2


# 定义用于测试符号分析函数的函数
def test_as_f_sign_1():
    # 断言：as_f_sign_1 函数测试
    assert as_f_sign_1(x + 1) == (1, x, 1)
    assert as_f_sign_1(x - 1) == (1, x, -1)
    assert as_f_sign_1(-x + 1) == (-1, x, -1)
    assert as_f_sign_1(-x - 1) == (-1, x, 1)
    assert as_f_sign_1(2*x + 2) == (2, x, 1)
    assert as_f_sign_1(x*y - y) == (y, x, -1)
    assert as_f_sign_1(-x*y + y) == (-y, x, -1)


# 定义用于测试已知问题 #25590 的函数
def test_issue_25590():
    # 创建非交换符号 A 和 B
    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)

    # 断言：TR8 变换测试
    assert TR8(2*cos(x)*sin(x)*B*A) == sin(2*x)*B*A
    # 断言：TR13 变换测试
    assert TR13(tan(2)*tan(3)*B*A) == (-tan(2)/tan(5) - tan(3)/tan(5) + 1)*B*A

    # 注意：结果可能不如 sin(2*x)*B*A + cos(x)**2 最优，未来可能会发生变化
    assert (2*cos(x)*sin(x)*B*A + cos(x)**2).simplify() == sin(2*x)*B*A + cos(2*x)/2 + S.One/2
```