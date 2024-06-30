# `D:\src\scipysrc\sympy\sympy\calculus\tests\test_finite_diff.py`

```
# 导入 itertools 模块中的 product 函数，用于计算笛卡尔积
from itertools import product

# 导入 sympy 库中的各个子模块和函数
from sympy.core.function import (Function, diff)
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.calculus.finite_diff import (
    apply_finite_diff, differentiate_finite, finite_diff_weights,
    _as_finite_diff
)
# 导入 sympy.testing.pytest 中的 raises 和 warns_deprecated_sympy 函数
from sympy.testing.pytest import raises, warns_deprecated_sympy


# 定义测试函数 test_apply_finite_diff
def test_apply_finite_diff():
    # 定义符号变量 x 和 h
    x, h = symbols('x h')
    # 定义函数 f
    f = Function('f')
    # 断言应用有限差分公式计算得到的结果为零
    assert (apply_finite_diff(1, [x-h, x+h], [f(x-h), f(x+h)], x) -
            (f(x+h)-f(x-h))/(2*h)).simplify() == 0

    # 断言应用有限差分公式计算得到的结果为零
    assert (apply_finite_diff(1, [5, 6, 7], [f(5), f(6), f(7)], 5) -
            (Rational(-3, 2)*f(5) + 2*f(6) - S.Half*f(7))).simplify() == 0
    # 断言调用 apply_finite_diff 函数时抛出 ValueError 异常
    raises(ValueError, lambda: apply_finite_diff(1, [x, h], [f(x)]))


# 定义测试函数 test_finite_diff_weights
def test_finite_diff_weights():

    # 计算一阶有限差分权重
    d = finite_diff_weights(1, [5, 6, 7], 5)
    assert d[1][2] == [Rational(-3, 2), 2, Rational(-1, 2)]

    # 使用表格中的数据验证高阶有限差分权重
    xl = [0, 1, -1, 2, -2, 3, -3, 4, -4]

    # 计算多阶有限差分权重
    d = finite_diff_weights(4, xl, S.Zero)

    # 验证零阶导数的权重
    for i in range(5):
        assert d[0][i] == [S.One] + [S.Zero]*8

    # 验证一阶导数的权重
    assert d[1][0] == [S.Zero]*9
    assert d[1][2] == [S.Zero, S.Half, Rational(-1, 2)] + [S.Zero]*6
    assert d[1][4] == [S.Zero, Rational(2, 3), Rational(-2, 3), Rational(-1, 12), Rational(1, 12)] + [S.Zero]*4
    assert d[1][6] == [S.Zero, Rational(3, 4), Rational(-3, 4), Rational(-3, 20), Rational(3, 20),
                       Rational(1, 60), Rational(-1, 60)] + [S.Zero]*2
    assert d[1][8] == [S.Zero, Rational(4, 5), Rational(-4, 5), Rational(-1, 5), Rational(1, 5),
                       Rational(4, 105), Rational(-4, 105), Rational(-1, 280), Rational(1, 280)]

    # 验证二阶导数的权重
    for i in range(2):
        assert d[2][i] == [S.Zero]*9
    assert d[2][2] == [-S(2), S.One, S.One] + [S.Zero]*6
    assert d[2][4] == [Rational(-5, 2), Rational(4, 3), Rational(4, 3), Rational(-1, 12), Rational(-1, 12)] + [S.Zero]*4
    assert d[2][6] == [Rational(-49, 18), Rational(3, 2), Rational(3, 2), Rational(-3, 20), Rational(-3, 20),
                       Rational(1, 90), Rational(1, 90)] + [S.Zero]*2
    assert d[2][8] == [Rational(-205, 72), Rational(8, 5), Rational(8, 5), Rational(-1, 5), Rational(-1, 5),
                       Rational(8, 315), Rational(8, 315), Rational(-1, 560), Rational(-1, 560)]

    # 验证三阶导数的权重
    for i in range(3):
        assert d[3][i] == [S.Zero]*9
    assert d[3][4] == [S.Zero, -S.One, S.One, S.Half, Rational(-1, 2)] + [S.Zero]*4
    assert d[3][6] == [S.Zero, Rational(-13, 8), Rational(13, 8), S.One, -S.One,
                       Rational(-1, 8), Rational(1, 8)] + [S.Zero]*2
    # 断言语句，验证 d[3][8] 是否等于指定的列表
    assert d[3][8] == [S.Zero, Rational(-61, 30), Rational(61, 30), Rational(169, 120), Rational(-169, 120),
                       Rational(-3, 10), Rational(3, 10), Rational(7, 240), Rational(-7, 240)]

    # 计算第四阶导数
    for i in range(4):
        # 断言语句，验证 d[4][i] 是否全为零
        assert d[4][i] == [S.Zero]*9
    # 断言语句，验证 d[4][4] 的特定值
    assert d[4][4] == [S(6), -S(4), -S(4), S.One, S.One] + [S.Zero]*4
    # 断言语句，验证 d[4][6] 的特定值
    assert d[4][6] == [Rational(28, 3), Rational(-13, 2), Rational(-13, 2), S(2), S(2),
                       Rational(-1, 6), Rational(-1, 6)] + [S.Zero]*2
    # 断言语句，验证 d[4][8] 的特定值
    assert d[4][8] == [Rational(91, 8), Rational(-122, 15), Rational(-122, 15), Rational(169, 60), Rational(169, 60),
                       Rational(-2, 5), Rational(-2, 5), Rational(7, 240), Rational(7, 240)]

    # 表格来源于文献 doi:10.1090/S0025-5718-1988-0935077-0 的第 703 页 Table 2
    # --------------------------------------------------------
    # 生成列表 xl，包含嵌套的列表，每个内部列表根据公式生成
    xl = [[j/S(2) for j in list(range(-i*2+1, 0, 2))+list(range(1, i*2+1, 2))]
          for i in range(1, 5)]

    # d 列表包含所有的系数
    d = [finite_diff_weights({0: 1, 1: 2, 2: 4, 3: 4}[i], xl[i], 0) for
         i in range(4)]

    # 零阶导数
    assert d[0][0][1] == [S.Half, S.Half]
    # 一阶导数
    assert d[1][0][3] == [Rational(-1, 16), Rational(9, 16), Rational(9, 16), Rational(-1, 16)]
    # 二阶导数
    assert d[2][0][5] == [Rational(3, 256), Rational(-25, 256), Rational(75, 128), Rational(75, 128),
                          Rational(-25, 256), Rational(3, 256)]
    # 三阶导数
    assert d[3][0][7] == [Rational(-5, 2048), Rational(49, 2048), Rational(-245, 2048), Rational(1225, 2048),
                          Rational(1225, 2048), Rational(-245, 2048), Rational(49, 2048), Rational(-5, 2048)]

    # 一阶导数
    assert d[0][1][1] == [-S.One, S.One]
    # 二阶导数
    assert d[1][1][3] == [Rational(1, 24), Rational(-9, 8), Rational(9, 8), Rational(-1, 24)]
    # 三阶导数
    assert d[2][1][5] == [Rational(-3, 640), Rational(25, 384), Rational(-75, 64),
                          Rational(75, 64), Rational(-25, 384), Rational(3, 640)]
    # 四阶导数
    assert d[3][1][7] == [Rational(5, 7168), Rational(-49, 5120),
                          Rational(245, 3072), Rational(-1225, 1024),
                          Rational(1225, 1024), Rational(-245, 3072),
                          Rational(49, 5120), Rational(-5, 7168)]

    # 目前测试剩余部分表格是否正确的检验被认为是过多的
    # 在此处引发 ValueError 异常，测试对无效输入的处理
    raises(ValueError, lambda: finite_diff_weights(-1, [1, 2]))
    raises(ValueError, lambda: finite_diff_weights(1.2, [1, 2]))
    # 声明符号变量 x
    x = symbols('x')
    raises(ValueError, lambda: finite_diff_weights(x, [1, 2]))
# 定义一个测试函数，用于测试有限差分方法的实现
def test_as_finite_diff():
    # 定义符号变量 x
    x = symbols('x')
    # 定义一个函数 f(x)
    f = Function('f')
    # 定义一个函数 dx(x)
    dx = Function('dx')

    # 调用 _as_finite_diff 函数，对 f(x) 关于 x 的导数进行有限差分计算
    _as_finite_diff(f(x).diff(x), [x-2, x-1, x, x+1, x+2])

    # 定义真实的导数表达式 df_true
    # 这里使用了未定义的函数 dx(x) 在 points 参数中
    df_true = -f(x+dx(x)/2-dx(x+dx(x)/2)/2) / dx(x+dx(x)/2) \
              + f(x+dx(x)/2+dx(x+dx(x)/2)/2) / dx(x+dx(x)/2)
    # 使用 sympy 的 diff 函数计算 f(x) 关于 x 的导数，并应用有限差分点 dx(x)
    df_test = diff(f(x), x).as_finite_difference(points=dx(x), x0=x+dx(x)/2)
    # 断言 df_test 和 df_true 的简化结果为 0
    assert (df_test - df_true).simplify() == 0


# 定义测试函数，用于测试不同变量的有限差分方法
def test_differentiate_finite():
    # 定义符号变量 x, y, h
    x, y, h = symbols('x y h')
    # 定义一个函数 f(x, y)
    f = Function('f')
    
    # 使用 warns_deprecated_sympy() 上下文管理器警告 sympy 的过时用法
    with warns_deprecated_sympy():
        # 调用 differentiate_finite 函数，对 f(x, y) + exp(42) 关于 x, y 进行有限差分计算
        res0 = differentiate_finite(f(x, y) + exp(42), x, y, evaluate=True)
    # 定义参考值 ref0
    xm, xp, ym, yp = [v + sign*S.Half for v, sign in product([x, y], [-1, 1])]
    ref0 = f(xm, ym) + f(xp, yp) - f(xm, yp) - f(xp, ym)
    # 断言 res0 和 ref0 的简化结果为 0
    assert (res0 - ref0).simplify() == 0

    # 定义一个函数 g(x)
    g = Function('g')
    # 使用 warns_deprecated_sympy() 上下文管理器警告 sympy 的过时用法
    with warns_deprecated_sympy():
        # 调用 differentiate_finite 函数，对 f(x)*g(x) + 42 关于 x 进行有限差分计算
        res1 = differentiate_finite(f(x)*g(x) + 42, x, evaluate=True)
    # 定义参考值 ref1
    ref1 = (-f(x - S.Half) + f(x + S.Half))*g(x) + \
           (-g(x - S.Half) + g(x + S.Half))*f(x)
    # 断言 res1 和 ref1 的简化结果为 0
    assert (res1 - ref1).simplify() == 0

    # 调用 differentiate_finite 函数，对 f(x) + x**3 + 42 关于 x 在指定点进行有限差分计算
    res2 = differentiate_finite(f(x) + x**3 + 42, x, points=[x-1, x+1])
    # 定义参考值 ref2
    ref2 = (f(x + 1) + (x + 1)**3 - f(x - 1) - (x - 1)**3)/2
    # 断言 res2 和 ref2 的简化结果为 0
    assert (res2 - ref2).simplify() == 0
    # 使用 raises 检查是否引发 TypeError 异常
    raises(TypeError, lambda: differentiate_finite(f(x)*g(x), x,
                                                   pints=[x-1, x+1]))

    # 调用 differentiate_finite 函数，对 f(x)*g(x).diff(x) 关于 x 进行有限差分计算
    res3 = differentiate_finite(f(x)*g(x).diff(x), x)
    # 定义参考值 ref3
    ref3 = (-g(x) + g(x + 1))*f(x + S.Half) - (g(x) - g(x - 1))*f(x - S.Half)
    # 断言 res3 等于 ref3
    assert res3 == ref3

    # 调用 differentiate_finite 函数，对 f(x)*g(x).diff(x).diff(x) 关于 x 进行有限差分计算
    res4 = differentiate_finite(f(x)*g(x).diff(x).diff(x), x)
    # 定义参考值 ref4
    ref4 = -((g(x - Rational(3, 2)) - 2*g(x - S.Half) + g(x + S.Half))*f(x - S.Half)) \
           + (g(x - S.Half) - 2*g(x + S.Half) + g(x + Rational(3, 2)))*f(x + S.Half)
    # 断言 res4 等于 ref4
    assert res4 == ref4

    # 定义表达式 res5_expr
    res5_expr = f(x).diff(x)*g(x).diff(x)
    # 调用 differentiate_finite 函数，对 res5_expr 在指定点进行有限差分计算
    res5 = differentiate_finite(res5_expr, points=[x-h, x, x+h])
    # 定义参考值 ref5
    ref5 = (-2*f(x)/h + f(-h + x)/(2*h) + 3*f(h + x)/(2*h))*(-2*g(x)/h + g(-h + x)/(2*h) \
           + 3*g(h + x)/(2*h))/(2*h) - (2*f(x)/h - 3*f(-h + x)/(2*h) - \
           f(h + x)/(2*h))*(2*g(x)/h - 3*g(-h + x)/(2*h) - g(h + x)/(2*h))/(2*h)
    # 断言 res5 等于 ref5
    assert res5 == ref5

    # 对 res5 取极限 h -> 0，并求值
    res6 = res5.limit(h, 0).doit()
    # 定义参考值 ref6
    ref6 = diff(res5_expr, x)
    # 断言 res6 等于 ref6
    assert res6 == ref6
```