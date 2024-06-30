# `D:\src\scipysrc\sympy\sympy\calculus\tests\test_euler.py`

```
from sympy.core.function import (Derivative as D, Function)  # 导入 Derivative 别名为 D 和 Function 类
from sympy.core.relational import Eq  # 导入 Eq 类
from sympy.core.symbol import (Symbol, symbols)  # 导入 Symbol 类和 symbols 函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入 cos 和 sin 函数
from sympy.testing.pytest import raises  # 导入 raises 函数用于测试
from sympy.calculus.euler import euler_equations as euler  # 导入 euler_equations 函数并别名为 euler


def test_euler_interface():
    x = Function('x')  # 创建符号函数 x(t)
    y = Symbol('y')  # 创建符号 y
    t = Symbol('t')  # 创建符号 t
    raises(TypeError, lambda: euler())  # 确保调用 euler() 抛出 TypeError 异常
    raises(TypeError, lambda: euler(D(x(t), t)*y(t), [x(t), y]))  # 确保调用 euler() 抛出 TypeError 异常
    raises(ValueError, lambda: euler(D(x(t), t)*x(y), [x(t), x(y)]))  # 确保调用 euler() 抛出 ValueError 异常
    raises(TypeError, lambda: euler(D(x(t), t)**2, x(0)))  # 确保调用 euler() 抛出 TypeError 异常
    raises(TypeError, lambda: euler(D(x(t), t)*y(t), [t]))  # 确保调用 euler() 抛出 TypeError 异常
    assert euler(D(x(t), t)**2/2, {x(t)}) == [Eq(-D(x(t), t, t), 0)]  # 验证 euler 方程为 D(x(t), t)**2/2，返回 [-D(x(t), t, t) = 0]
    assert euler(D(x(t), t)**2/2, x(t), {t}) == [Eq(-D(x(t), t, t), 0)]  # 验证 euler 方程为 D(x(t), t)**2/2，返回 [-D(x(t), t, t) = 0]


def test_euler_pendulum():
    x = Function('x')  # 创建符号函数 x(t)
    t = Symbol('t')  # 创建符号 t
    L = D(x(t), t)**2/2 + cos(x(t))  # 定义拉格朗日量 L
    assert euler(L, x(t), t) == [Eq(-sin(x(t)) - D(x(t), t, t), 0)]  # 验证 euler 方程为 L，返回 [-sin(x(t)) - D(x(t), t, t) = 0]


def test_euler_henonheiles():
    x = Function('x')  # 创建符号函数 x(t)
    y = Function('y')  # 创建符号函数 y(t)
    t = Symbol('t')  # 创建符号 t
    L = sum(D(z(t), t)**2/2 - z(t)**2/2 for z in [x, y])  # 定义拉格朗日量 L
    L += -x(t)**2*y(t) + y(t)**3/3  # 添加额外的拉格朗日量成分
    assert euler(L, [x(t), y(t)], t) == [Eq(-2*x(t)*y(t) - x(t) -
                                            D(x(t), t, t), 0),
                                         Eq(-x(t)**2 + y(t)**2 -
                                            y(t) - D(y(t), t, t), 0)]  # 验证 euler 方程为 L，返回 [-2*x(t)*y(t) - x(t) - D(x(t), t, t) = 0, -x(t)**2 + y(t)**2 - y(t) - D(y(t), t, t) = 0]


def test_euler_sineg():
    psi = Function('psi')  # 创建符号函数 psi(t, x)
    t = Symbol('t')  # 创建符号 t
    x = Symbol('x')  # 创建符号 x
    L = D(psi(t, x), t)**2/2 - D(psi(t, x), x)**2/2 + cos(psi(t, x))  # 定义拉格朗日量 L
    assert euler(L, psi(t, x), [t, x]) == [Eq(-sin(psi(t, x)) -
                                              D(psi(t, x), t, t) +
                                              D(psi(t, x), x, x), 0)]  # 验证 euler 方程为 L，返回 [-sin(psi(t, x)) - D(psi(t, x), t, t) + D(psi(t, x), x, x) = 0]


def test_euler_high_order():
    # an example from hep-th/0309038
    m = Symbol('m')  # 创建符号 m
    k = Symbol('k')  # 创建符号 k
    x = Function('x')  # 创建符号函数 x(t)
    y = Function('y')  # 创建符号函数 y(t)
    t = Symbol('t')  # 创建符号 t
    L = (m*D(x(t), t)**2/2 + m*D(y(t), t)**2/2 -
         k*D(x(t), t)*D(y(t), t, t) + k*D(y(t), t)*D(x(t), t, t))  # 定义拉格朗日量 L
    assert euler(L, [x(t), y(t)]) == [Eq(2*k*D(y(t), t, t, t) -
                                         m*D(x(t), t, t), 0),
                                      Eq(-2*k*D(x(t), t, t, t) -
                                         m*D(y(t), t, t), 0)]  # 验证 euler 方程为 L，返回 [2*k*D(y(t), t, t, t) - m*D(x(t), t, t) = 0, -2*k*D(x(t), t, t, t) - m*D(y(t), t, t) = 0]

    w = Symbol('w')  # 创建符号 w
    L = D(x(t, w), t, w)**2/2  # 定义拉格朗日量 L
    assert euler(L) == [Eq(D(x(t, w), t, t, w, w), 0)]  # 验证 euler 方程为 L，返回 [D(x(t, w), t, t, w, w) = 0]


def test_issue_18653():
    x, y, z = symbols("x y z")  # 创建符号 x, y, z
    f, g, h = symbols("f g h", cls=Function, args=(x, y))  # 创建符号函数 f(x, y), g(x, y), h(x, y)
    f, g, h = f(), g(), h()  # 调用函数 f, g, h
    expr2 = f.diff(x)*h.diff(z)  # 定义表达式 expr2
    assert euler(expr2, (f,), (x, y)) == []  # 验证 euler 方程为 expr2，返回空列表
```