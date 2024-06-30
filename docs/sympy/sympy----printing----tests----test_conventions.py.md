# `D:\src\scipysrc\sympy\sympy\printing\tests\test_conventions.py`

```
# -*- coding: utf-8 -*-

# 导入必要的函数和模块
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.functions.special.bessel import besselj
from sympy.functions.special.polynomials import legendre
from sympy.functions.combinatorial.numbers import bell
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.testing.pytest import XFAIL

# 定义测试函数test_super_sub，用于测试split_super_sub函数的功能
def test_super_sub():
    # 断言split_super_sub("beta_13_2")的返回值为("beta", [], ["13", "2"])
    assert split_super_sub("beta_13_2") == ("beta", [], ["13", "2"])
    # 断言split_super_sub("beta_132_20")的返回值为("beta", [], ["132", "20"])
    assert split_super_sub("beta_132_20") == ("beta", [], ["132", "20"])
    # 断言split_super_sub("beta_13")的返回值为("beta", [], ["13"])
    assert split_super_sub("beta_13") == ("beta", [], ["13"])
    # 断言split_super_sub("x_a_b")的返回值为("x", [], ["a", "b"])
    assert split_super_sub("x_a_b") == ("x", [], ["a", "b"])
    # 断言split_super_sub("x_1_2_3")的返回值为("x", [], ["1", "2", "3"])
    assert split_super_sub("x_1_2_3") == ("x", [], ["1", "2", "3"])
    # 断言split_super_sub("x_a_b1")的返回值为("x", [], ["a", "b1"])
    assert split_super_sub("x_a_b1") == ("x", [], ["a", "b1"])
    # 断言split_super_sub("x_a_1")的返回值为("x", [], ["a", "1"])
    assert split_super_sub("x_a_1") == ("x", [], ["a", "1"])
    # 断言split_super_sub("x_1_a")的返回值为("x", [], ["1", "a"])
    assert split_super_sub("x_1_a") == ("x", [], ["1", "a"])
    # 断言split_super_sub("x_1^aa")的返回值为("x", ["aa"], ["1"])
    assert split_super_sub("x_1^aa") == ("x", ["aa"], ["1"])
    # 断言split_super_sub("x_1__aa")的返回值为("x", ["aa"], ["1"])
    assert split_super_sub("x_1__aa") == ("x", ["aa"], ["1"])
    # 断言split_super_sub("x_11^a")的返回值为("x", ["a"], ["11"])
    assert split_super_sub("x_11^a") == ("x", ["a"], ["11"])
    # 断言split_super_sub("x_11__a")的返回值为("x", ["a"], ["11"])
    assert split_super_sub("x_11__a") == ("x", ["a"], ["11"])
    # 断言split_super_sub("x_a_b_c_d")的返回值为("x", [], ["a", "b", "c", "d"])
    assert split_super_sub("x_a_b_c_d") == ("x", [], ["a", "b", "c", "d"])
    # 断言split_super_sub("x_a_b^c^d")的返回值为("x", ["c", "d"], ["a", "b"])
    assert split_super_sub("x_a_b^c^d") == ("x", ["c", "d"], ["a", "b"])
    # 断言split_super_sub("x_a_b__c__d")的返回值为("x", ["c", "d"], ["a", "b"])
    assert split_super_sub("x_a_b__c__d") == ("x", ["c", "d"], ["a", "b"])
    # 断言split_super_sub("x_a^b_c^d")的返回值为("x", ["b", "d"], ["a", "c"])
    assert split_super_sub("x_a^b_c^d") == ("x", ["b", "d"], ["a", "c"])
    # 断言split_super_sub("x_a__b_c__d")的返回值为("x", ["b", "d"], ["a", "c"])
    assert split_super_sub("x_a__b_c__d") == ("x", ["b", "d"], ["a", "c"])
    # 断言split_super_sub("x^a^b_c_d")的返回值为("x", ["a", "b"], ["c", "d"])
    assert split_super_sub("x^a^b_c_d") == ("x", ["a", "b"], ["c", "d"])
    # 断言split_super_sub("x__a__b_c_d")的返回值为("x", ["a", "b"], ["c", "d"])
    assert split_super_sub("x__a__b_c_d") == ("x", ["a", "b"], ["c", "d"])
    # 断言split_super_sub("x^a^b^c^d")的返回值为("x", ["a", "b", "c", "d"], [])
    assert split_super_sub("x^a^b^c^d") == ("x", ["a", "b", "c", "d"], [])
    # 断言split_super_sub("x__a__b__c__d")的返回值为("x", ["a", "b", "c", "d"], [])
    assert split_super_sub("x__a__b__c__d") == ("x", ["a", "b", "c", "d"], [])
    # 断言split_super_sub("alpha_11")的返回值为("alpha", [], ["11"])
    assert split_super_sub("alpha_11") == ("alpha", [], ["11"])
    # 断言split_super_sub("alpha_11_11")的返回值为("alpha", [], ["11", "11"])
    assert split_super_sub("alpha_11_11") == ("alpha", [], ["11", "11"])
    # 断言split_super_sub("w1")的返回值为("w", [], ["1"])
    assert split_super_sub("w1") == ("w", [], ["1"])
    # 断言split_super_sub("w𝟙")的返回值为("w", [], ["𝟙"])
    assert split_super_sub("w𝟙") == ("w", [], ["𝟙"])
    # 断言split_super_sub("w11")的返回值为("w", [], ["11"])
    assert split_super_sub("w11") == ("w", [], ["11"])
    # 断言split_super_sub("w𝟙𝟙")的返回值为("w", [], ["𝟙𝟙"])
    assert split_super_sub("w𝟙𝟙") == ("w", [], ["𝟙𝟙"])
    # 断言split_super_sub("w𝟙2𝟙")的返回值为("w", [], ["𝟙2𝟙"])
    assert split_super_sub("w𝟙2𝟙") == ("w", [], ["𝟙2𝟙"])
    # 断言split_super_sub("w1^a")的返回值为("w", ["a"], ["1"])
    assert split_super_sub("w1^a") == ("w", ["a"], ["1"])
    # 断言split_super_sub("ω1")的返回值为("ω", [], ["1"])
    assert split_super_sub("ω1") == ("ω", [], ["1"])
    # 断言split_super_sub("ω11")的返回值为("ω", [], ["11"])
    assert split_super_sub("ω11") == ("ω", [], ["11"])
    # 断言split_super_sub("ω1^a")的返回值为("ω", ["a"], ["1"])
    assert split_super_sub("ω1^a") == ("ω", ["a"], ["1"])
    # 断言split_super_sub("ω𝟙^α")的返回值为("ω", ["α"], ["𝟙"])
    assert split_super_sub("ω𝟙^α") == ("ω", ["α"], ["𝟙"])
    # 断言split_super_sub("ω𝟙2^3α")的返回值为("ω", ["3α"], ["𝟙2"])
    assert split_super_sub("ω𝟙2^3α") == ("ω", ["3α"], ["𝟙2"])
    # 断言split_super_sub("")的返回值为("", [], [])
    assert split_super
    # 断言对于 f 对 x 的偏导数要求为 True
    assert requires_partial(Derivative(f, x)) is True
    # 断言对于 f 对 y 的偏导数要求为 True
    assert requires_partial(Derivative(f, y)) is True

    ## 对其中一个变量进行积分
    # 断言对于积分 exp(-x * y) 关于 y 的偏导数求值为 False
    assert requires_partial(Derivative(Integral(exp(-x * y), (x, 0, oo)), y, evaluate=False)) is False

    ## 贝塞尔函数与平滑参数
    # 计算贝塞尔函数 besselj(nu, x)
    f = besselj(nu, x)
    # 断言对于 f 对 x 的偏导数要求为 True
    assert requires_partial(Derivative(f, x)) is True
    # 断言对于 f 对 nu 的偏导数要求为 True
    assert requires_partial(Derivative(f, nu)) is True

    ## 贝塞尔函数与整数参数
    # 计算贝塞尔函数 besselj(n, x)
    f = besselj(n, x)
    # 断言对于 f 对 x 的偏导数要求为 False
    assert requires_partial(Derivative(f, x)) is False
    # 对于整数参数，偏导数的符号不适用，但这里应保证不抛出异常
    assert requires_partial(Derivative(f, n)) is False

    ## 贝尔多项式
    # 计算贝尔多项式 bell(n, x)
    f = bell(n, x)
    # 断言对于 f 对 x 的偏导数要求为 False
    assert requires_partial(Derivative(f, x)) is False
    # 对于整数参数，偏导数的符号不适用
    assert requires_partial(Derivative(f, n)) is False

    ## 勒让德多项式
    # 计算勒让德多项式 legendre(0, x)
    f = legendre(0, x)
    # 断言对于 f 对 x 的偏导数要求为 False
    assert requires_partial(Derivative(f, x)) is False

    # 计算勒让德多项式 legendre(n, x)
    f = legendre(n, x)
    # 断言对于 f 对 x 的偏导数要求为 False
    assert requires_partial(Derivative(f, x)) is False
    # 对于整数参数，偏导数的符号不适用
    assert requires_partial(Derivative(f, n)) is False

    ## 幂函数
    f = x ** n
    # 断言对于 f 对 x 的偏导数要求为 False
    assert requires_partial(Derivative(f, x)) is False

    # 断言对于积分 (x*y) ** n * exp(-x * y) 关于 y 的偏导数求值为 False
    assert requires_partial(Derivative(Integral((x*y) ** n * exp(-x * y), (x, 0, oo)), y, evaluate=False)) is False

    # 参数方程
    f = (exp(t), cos(t))
    g = sum(f)
    # 断言对于 g 关于 t 的偏导数要求为 False
    assert requires_partial(Derivative(g, t)) is False

    # 符号函数
    f = symbols('f', cls=Function)
    # 断言对于 f(x) 关于 x 的偏导数要求为 False
    assert requires_partial(Derivative(f(x), x)) is False
    # 断言对于 f(x) 关于 y 的偏导数要求为 False
    assert requires_partial(Derivative(f(x), y)) is False
    # 断言对于 f(x, y) 关于 x 的偏导数要求为 True
    assert requires_partial(Derivative(f(x, y), x)) is True
    # 断言对于 f(x, y) 关于 y 的偏导数要求为 True
    assert requires_partial(Derivative(f(x, y), y)) is True
    # 断言对于 f(x, y) 关于 z 的偏导数要求为 True
    assert requires_partial(Derivative(f(x, y), z)) is True
    # 断言对于 f(x, y) 关于 x, y 的混合偏导数要求为 True
    assert requires_partial(Derivative(f(x, y), x, y)) is True
@XFAIL
# 标记为 XFAIL 的测试函数，表示这个测试预期会失败
def test_requires_partial_unspecified_variables():
    # 创建符号变量 x 和 y
    x, y = symbols('x y')
    # 创建一个未指定变量的函数符号 f
    f = symbols('f', cls=Function)
    # 断言对于 f 关于 x 的导数不需要部分求导
    assert requires_partial(Derivative(f, x)) is False
    # 断言对于 f 关于 x 和 y 的导数需要部分求导
    assert requires_partial(Derivative(f, x, y)) is True
```