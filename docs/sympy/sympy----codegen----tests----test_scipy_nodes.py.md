# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_scipy_nodes.py`

```
# 导入需要的模块和函数
from itertools import product
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos
from sympy.core.numbers import pi
from sympy.codegen.scipy_nodes import cosm1, powm1

# 定义符号变量
x, y, z = symbols('x y z')

# 定义测试函数 test_cosm1，用于测试 cosm1 函数的正确性
def test_cosm1():
    # 计算 cosm1(x*y)，表示 cos(x*y) - 1
    cm1_xy = cosm1(x*y)
    # 参考值为 cos(x*y) - 1
    ref_xy = cos(x*y) - 1
    
    # 对 x, y, z 的所有可能性进行三阶导数的比较
    for wrt, deriv_order in product([x, y, z], range(3)):
        # 断言 cosm1(x*y) 的导数减去参考值的导数为零，使用 cos 函数重写并简化
        assert (
            cm1_xy.diff(wrt, deriv_order) -
            ref_xy.diff(wrt, deriv_order)
        ).rewrite(cos).simplify() == 0

    # 计算 cosm1(pi)，并断言其用 cos 函数重写后为 -2
    expr_minus2 = cosm1(pi)
    assert expr_minus2.rewrite(cos) == -2
    
    # 断言 cosm1(3.14) 简化后与其自身相等，无法简化常数 3.14
    assert cosm1(3.14).simplify() == cosm1(3.14)
    
    # 断言 cosm1(pi/2) 简化后为 -1
    assert cosm1(pi/2).simplify() == -1
    
    # 断言 (1/cos(x) - 1 + cosm1(x)/cos(x)) 简化后为 0
    assert (1/cos(x) - 1 + cosm1(x)/cos(x)).simplify() == 0


# 定义测试函数 test_powm1，用于测试 powm1 函数的正确性
def test_powm1():
    # 定义不同的表达式和其对应的参考值
    cases = {
            powm1(x, y): x**y - 1,
            powm1(x*y, z): (x*y)**z - 1,
            powm1(x, y*z): x**(y*z)-1,
            powm1(x*y*z, x*y*z): (x*y*z)**(x*y*z)-1
    }
    # 对每个表达式和参考值进行验证
    for pm1_e, ref_e in cases.items():
        for wrt, deriv_order in product([x, y, z], range(3)):
            # 计算 powm1 函数的导数
            der = pm1_e.diff(wrt, deriv_order)
            # 参考值的导数
            ref = ref_e.diff(wrt, deriv_order)
            # 计算差异并重写为 Pow 形式，然后简化
            delta = (der - ref).rewrite(Pow)
            assert delta.simplify() == 0

    # 计算 powm1(x, 1/log(x))，并断言其用 Pow 函数重写后为 exp(1) - 1
    eulers_constant_m1 = powm1(x, 1/log(x))
    assert eulers_constant_m1.rewrite(Pow) == exp(1) - 1
    # 断言简化后的结果与 exp(1) - 1 相等
    assert eulers_constant_m1.simplify() == exp(1) - 1
```