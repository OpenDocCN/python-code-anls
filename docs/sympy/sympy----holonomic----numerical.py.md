# `D:\src\scipysrc\sympy\sympy\holonomic\numerical.py`

```
"""Numerical Methods for Holonomic Functions"""

# 导入必要的模块和函数
from sympy.core.sympify import sympify  # 导入 sympify 函数，用于将输入转换为 SymPy 表达式
from sympy.holonomic.holonomic import DMFsubs  # 从 sympy.holonomic.holonomic 模块导入 DMFsubs 函数

# 导入 mpmath 中的 mp 对象
from mpmath import mp


# 定义一个函数，用于数值方法的数值积分，沿着复平面上的一组给定点
def _evalf(func, points, derivatives=False, method='RK4'):
    """
    Numerical methods for numerical integration along a given set of
    points in the complex plane.
    """
    
    # 获取给定函数的湮没子 ann
    ann = func.annihilator
    a = ann.order  # 获取 ann 的阶数
    R = ann.parent.base  # 获取 ann 的环
    K = R.get_field()  # 获取 ann 环的域

    # 根据指定的数值积分方法选择相应的函数
    if method == 'Euler':
        meth = _euler
    else:
        meth = _rk4

    # 创建一个包含 ann.listofpoly 中元素的 K 对象的列表
    dmf = [K.new(j.to_list()) for j in ann.listofpoly]
    # 计算 red 列表，每个元素是对应 dmf 元素的相反数除以 ann 的最高次幂 dmf[a]
    red = [-dmf[i] / dmf[a] for i in range(a)]

    y0 = func.y0  # 获取函数的初始向量 y0
    if len(y0) < a:
        raise TypeError("Not Enough Initial Conditions")  # 如果初始条件不足，抛出 TypeError 异常
    x0 = func.x0  # 获取函数的初始点 x0
    sol = [meth(red, x0, points[0], y0, a)]  # 使用选择的数值积分方法计算第一个点的解

    # 对于给定点中的每一个点，使用数值积分方法计算解
    for i, j in enumerate(points[1:]):
        sol.append(meth(red, points[i], j, sol[-1], a))

    # 如果 derivatives 参数为 False，则返回解列表中每个元素的 SymPy 表达式形式
    if not derivatives:
        return [sympify(i[0]) for i in sol]
    else:
        return sympify(sol)


# 定义 Euler 方法进行数值积分
def _euler(red, x0, x1, y0, a):
    """
    Euler's method for numerical integration.
    From x0 to x1 with initial values given at x0 as vector y0.
    """

    A = sympify(x0)._to_mpmath(mp.prec)  # 将 x0 转换为 mpmath 中的精度
    B = sympify(x1)._to_mpmath(mp.prec)  # 将 x1 转换为 mpmath 中的精度
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]  # 将 y0 中的每个元素转换为 mpmath 中的精度
    h = B - A  # 计算步长 h
    f_0 = y_0[1:]  # 初始化 f_0 为 y_0 的第二个元素开始的列表
    f_0_n = 0  # 初始化 f_0_n 为 0

    # 计算 Euler 方法中的每一步
    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)

    return [y_0[i] + h * f_0[i] for i in range(a)]


# 定义 Runge-Kutta 4 阶方法进行数值积分
def _rk4(red, x0, x1, y0, a):
    """
    Runge-Kutta 4th order numerical method.
    """

    A = sympify(x0)._to_mpmath(mp.prec)  # 将 x0 转换为 mpmath 中的精度
    B = sympify(x1)._to_mpmath(mp.prec)  # 将 x1 转换为 mpmath 中的精度
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]  # 将 y0 中的每个元素转换为 mpmath 中的精度
    h = B - A  # 计算步长 h

    f_0_n = 0  # 初始化 f_0_n 为 0
    f_1_n = 0  # 初始化 f_1_n 为 0
    f_2_n = 0  # 初始化 f_2_n 为 0
    f_3_n = 0  # 初始化 f_3_n 为 0

    f_0 = y_0[1:]  # 初始化 f_0 为 y_0 的第二个元素开始的列表
    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)

    f_1 = [y_0[i] + f_0[i]*h/2 for i in range(1, a)]  # 计算 f_1 列表
    for i in range(a):
        f_1_n += sympify(DMFsubs(red[i], A + h/2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_0[i]*h/2)
    f_1.append(f_1_n)

    f_2 = [y_0[i] + f_1[i]*h/2 for i in range(1, a)]  # 计算 f_2 列表
    for i in range(a):
        f_2_n += sympify(DMFsubs(red[i], A + h/2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_1[i]*h/2)
    f_2.append(f_2_n)

    f_3 = [y_0[i] + f_2[i]*h for i in range(1, a)]  # 计算 f_3 列表
    for i in range(a):
        f_3_n += sympify(DMFsubs(red[i], A + h, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_2[i]*h)
    f_3.append(f_3_n)

    return [y_0[i] + h*(f_0[i]+2*f_1[i]+2*f_2[i]+f_3[i])/6 for i in range(a)]
```