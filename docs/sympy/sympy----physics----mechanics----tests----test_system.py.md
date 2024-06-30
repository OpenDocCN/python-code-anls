# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_system.py`

```
# 导入所需的符号计算模块和函数
from sympy import symbols, Matrix, atan, zeros
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, Particle, Point,
                                     ReferenceFrame, SymbolicSystem)
from sympy.testing.pytest import raises

# 定义动力学变量和常量符号
x, y, u, v, lam = dynamicsymbols('x y u v lambda')  # 定义动力学变量
m, l, g = symbols('m l g')  # 定义常量符号

# 设置不同形式的方程式表示形式
# [1] 显式形式，其中动力学和运动学结合
# x' = F(x, t, r, p)
#
# [2] 隐式形式，其中动力学和运动学结合
# M(x, p) x' = F(x, t, r, p)
#
# [3] 隐式形式，其中动力学和运动学分离
# M(q, p) u' = F(q, u, t, r, p)
# q' = G(q, u, t, r, p)
dyn_implicit_mat = Matrix([[1, 0, -x/m],
                           [0, 1, -y/m],
                           [0, 0, l**2/m]])  # 隐式动力学方程的系数矩阵

dyn_implicit_rhs = Matrix([0, 0, u**2 + v**2 - g*y])  # 隐式动力学方程的右侧项

comb_implicit_mat = Matrix([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, -x/m],
                            [0, 0, 0, 1, -y/m],
                            [0, 0, 0, 0, l**2/m]])  # 结合形式的隐式动力学方程的系数矩阵

comb_implicit_rhs = Matrix([u, v, 0, 0, u**2 + v**2 - g*y])  # 结合形式的隐式动力学方程的右侧项

kin_explicit_rhs = Matrix([u, v])  # 显式运动学方程的右侧项

comb_explicit_rhs = comb_implicit_mat.LUsolve(comb_implicit_rhs)  # 结合形式的显式动力学方程的右侧项

# 设置一个质点和加载以传递到系统中
theta = atan(x/y)  # 计算角度
N = ReferenceFrame('N')  # 定义一个参考系
A = N.orientnew('A', 'Axis', [theta, N.z])  # 在参考系中建立一个新的方向
O = Point('O')  # 定义一个点
P = O.locatenew('P', l * A.x)  # 在点O处相对位置定义一个新的点P

Pa = Particle('Pa', P, m)  # 创建一个质点

bodies = [Pa]  # 质点列表
loads = [(P, g * m * N.x)]  # 负载列表

# 设置一些输出方程，以便传递给SymbolicSystem
PE = symbols("PE")
out_eqns = {PE: m*g*(l+y)}  # 输出方程字典

# 设置传递给SymbolicSystem的剩余参数
alg_con = [2]  # 算法约束
alg_con_full = [4]  # 完整的算法约束
coordinates = (x, y, lam)  # 坐标变量
speeds = (u, v)  # 速度变量
states = (x, y, u, v, lam)  # 状态变量
coord_idxs = (0, 1)  # 坐标索引
speed_idxs = (2, 3)  # 速度索引

def test_form_1():
    # 创建一个SymbolicSystem对象，用于测试
    symsystem1 = SymbolicSystem(states, comb_explicit_rhs,
                                alg_con=alg_con_full, output_eqns=out_eqns,
                                coord_idxs=coord_idxs, speed_idxs=speed_idxs,
                                bodies=bodies, loads=loads)

    assert symsystem1.coordinates == Matrix([x, y])
    assert symsystem1.speeds == Matrix([u, v])
    assert symsystem1.states == Matrix([x, y, u, v, lam])

    assert symsystem1.alg_con == [4]

    inter = comb_explicit_rhs
    assert simplify(symsystem1.comb_explicit_rhs - inter) == zeros(5, 1)

    assert set(symsystem1.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem1.dynamic_symbols()) == tuple
    assert set(symsystem1.constant_symbols()) == {l, g, m}
    assert type(symsystem1.constant_symbols()) == tuple

    assert symsystem1.output_eqns == out_eqns

    assert symsystem1.bodies == (Pa,)
    # 断言语句，用于验证 symsystem1.loads 是否等于 ((P, g * m * N.x),)
    assert symsystem1.loads == ((P, g * m * N.x),)
def test_form_2():
    # 创建一个SymbolicSystem对象，用于符号计算系统
    symsystem2 = SymbolicSystem(coordinates, comb_implicit_rhs, speeds=speeds,
                                mass_matrix=comb_implicit_mat,
                                alg_con=alg_con_full, output_eqns=out_eqns,
                                bodies=bodies, loads=loads)

    # 断言符号系统的坐标变量
    assert symsystem2.coordinates == Matrix([x, y, lam])
    # 断言符号系统的速度变量
    assert symsystem2.speeds == Matrix([u, v])
    # 断言符号系统的状态变量
    assert symsystem2.states == Matrix([x, y, lam, u, v])

    # 断言符号系统的代数约束
    assert symsystem2.alg_con == [4]

    # 暂存comb_implicit_rhs的值
    inter = comb_implicit_rhs
    # 断言符号系统的隐式组合右手边与暂存值的差为零
    assert simplify(symsystem2.comb_implicit_rhs - inter) == zeros(5, 1)
    # 断言符号系统的隐式组合质量矩阵与给定的质量矩阵的差为零
    assert simplify(symsystem2.comb_implicit_mat - comb_implicit_mat) == zeros(5)

    # 断言符号系统的动态符号集合
    assert set(symsystem2.dynamic_symbols()) == {y, v, lam, u, x}
    # 断言符号系统的动态符号集合类型为元组
    assert type(symsystem2.dynamic_symbols()) == tuple
    # 断言符号系统的常数符号集合
    assert set(symsystem2.constant_symbols()) == {l, g, m}
    # 断言符号系统的常数符号集合类型为元组
    assert type(symsystem2.constant_symbols()) == tuple

    # 暂存comb_explicit_rhs的值
    inter = comb_explicit_rhs
    # 计算符号系统的显式形式
    symsystem2.compute_explicit_form()
    # 断言符号系统的显式组合右手边与暂存值的差为零
    assert simplify(symsystem2.comb_explicit_rhs - inter) == zeros(5, 1)

    # 断言符号系统的输出方程集合
    assert symsystem2.output_eqns == out_eqns

    # 断言符号系统的物体集合
    assert symsystem2.bodies == (Pa,)
    # 断言符号系统的负载集合
    assert symsystem2.loads == ((P, g * m * N.x),)


def test_form_3():
    # 创建一个SymbolicSystem对象，用于符号计算系统
    symsystem3 = SymbolicSystem(states, dyn_implicit_rhs,
                                mass_matrix=dyn_implicit_mat,
                                coordinate_derivatives=kin_explicit_rhs,
                                alg_con=alg_con, coord_idxs=coord_idxs,
                                speed_idxs=speed_idxs, bodies=bodies,
                                loads=loads)

    # 断言符号系统的坐标变量
    assert symsystem3.coordinates == Matrix([x, y])
    # 断言符号系统的速度变量
    assert symsystem3.speeds == Matrix([u, v])
    # 断言符号系统的状态变量
    assert symsystem3.states == Matrix([x, y, u, v, lam])

    # 断言符号系统的代数约束
    assert symsystem3.alg_con == [4]

    # 暂存kin_explicit_rhs的值
    inter1 = kin_explicit_rhs
    # 暂存dyn_implicit_rhs的值
    inter2 = dyn_implicit_rhs
    # 断言符号系统的显式动力学右手边与暂存值的差为零
    assert simplify(symsystem3.kin_explicit_rhs - inter1) == zeros(2, 1)
    # 断言符号系统的隐式动力学质量矩阵与给定的质量矩阵的差为零
    assert simplify(symsystem3.dyn_implicit_mat - dyn_implicit_mat) == zeros(3)
    # 断言符号系统的隐式动力学右手边与暂存值的差为零
    assert simplify(symsystem3.dyn_implicit_rhs - inter2) == zeros(3, 1)

    # 暂存comb_implicit_rhs的值
    inter = comb_implicit_rhs
    # 断言符号系统的隐式组合右手边与暂存值的差为零
    assert simplify(symsystem3.comb_implicit_rhs - inter) == zeros(5, 1)
    # 断言符号系统的隐式组合质量矩阵与给定的质量矩阵的差为零
    assert simplify(symsystem3.comb_implicit_mat - comb_implicit_mat) == zeros(5)

    # 暂存comb_explicit_rhs的值
    inter = comb_explicit_rhs
    # 计算符号系统的显式形式
    symsystem3.compute_explicit_form()
    # 断言符号系统的显式组合右手边与暂存值的差为零
    assert simplify(symsystem3.comb_explicit_rhs - inter) == zeros(5, 1)

    # 断言符号系统的动态符号集合
    assert set(symsystem3.dynamic_symbols()) == {y, v, lam, u, x}
    # 断言符号系统的动态符号集合类型为元组
    assert type(symsystem3.dynamic_symbols()) == tuple
    # 断言符号系统的常数符号集合
    assert set(symsystem3.constant_symbols()) == {l, g, m}
    # 断言符号系统的常数符号集合类型为元组
    assert type(symsystem3.constant_symbols()) == tuple

    # 断言符号系统的输出方程集合为空字典
    assert symsystem3.output_eqns == {}

    # 断言符号系统的物体集合
    assert symsystem3.bodies == (Pa,)
    # 断言符号系统的负载集合
    assert symsystem3.loads == ((P, g * m * N.x),)
    # 使用给定的参数初始化 SymbolicSystem 对象
    symsystem = SymbolicSystem(states, comb_explicit_rhs,
                               alg_con=alg_con_full, output_eqns=out_eqns,
                               coord_idxs=coord_idxs, speed_idxs=speed_idxs,
                               bodies=bodies, loads=loads)

    # 测试设置对象属性时是否会引发 AttributeError 异常
    with raises(AttributeError):
        symsystem.bodies = 42  # 测试设置 bodies 属性
    with raises(AttributeError):
        symsystem.coordinates = 42  # 测试设置 coordinates 属性
    with raises(AttributeError):
        symsystem.dyn_implicit_rhs = 42  # 测试设置 dyn_implicit_rhs 属性
    with raises(AttributeError):
        symsystem.comb_implicit_rhs = 42  # 测试设置 comb_implicit_rhs 属性
    with raises(AttributeError):
        symsystem.loads = 42  # 测试设置 loads 属性
    with raises(AttributeError):
        symsystem.dyn_implicit_mat = 42  # 测试设置 dyn_implicit_mat 属性
    with raises(AttributeError):
        symsystem.comb_implicit_mat = 42  # 测试设置 comb_implicit_mat 属性
    with raises(AttributeError):
        symsystem.kin_explicit_rhs = 42  # 测试设置 kin_explicit_rhs 属性
    with raises(AttributeError):
        symsystem.comb_explicit_rhs = 42  # 测试设置 comb_explicit_rhs 属性
    with raises(AttributeError):
        symsystem.speeds = 42  # 测试设置 speeds 属性
    with raises(AttributeError):
        symsystem.states = 42  # 测试设置 states 属性
    with raises(AttributeError):
        symsystem.alg_con = 42  # 测试设置 alg_con 属性


这段代码的作用是初始化一个 `SymbolicSystem` 对象，并逐个测试设置对象的不同属性时是否会抛出 `AttributeError` 异常。
# 定义一个测试函数，用于测试当尝试访问对象创建时未指定或在创建时指定但用户尝试重新计算的属性时产生的错误。
def test_not_specified_errors():
    """This test will cover errors that arise from trying to access attributes
    that were not specified upon object creation or were specified on creation
    and the user tries to recalculate them."""
    
    # 创建一个符号系统对象symsystem1，使用给定的states和comb_explicit_rhs参数
    symsystem1 = SymbolicSystem(states, comb_explicit_rhs)

    # 断言尝试访问comb_implicit_mat属性时会引发AttributeError异常
    with raises(AttributeError):
        symsystem1.comb_implicit_mat
    # 断言尝试访问comb_implicit_rhs属性时会引发AttributeError异常
    with raises(AttributeError):
        symsystem1.comb_implicit_rhs
    # 断言尝试访问dyn_implicit_mat属性时会引发AttributeError异常
    with raises(AttributeError):
        symsystem1.dyn_implicit_mat
    # 断言尝试访问dyn_implicit_rhs属性时会引发AttributeError异常
    with raises(AttributeError):
        symsystem1.dyn_implicit_rhs
    # 断言尝试访问kin_explicit_rhs属性时会引发AttributeError异常
    with raises(AttributeError):
        symsystem1.kin_explicit_rhs
    # 断言尝试调用compute_explicit_form方法时会引发AttributeError异常
    with raises(AttributeError):
        symsystem1.compute_explicit_form()

    # 创建另一个符号系统对象symsystem2，使用给定的coordinates, comb_implicit_rhs,
    # speeds和mass_matrix参数
    symsystem2 = SymbolicSystem(coordinates, comb_implicit_rhs, speeds=speeds,
                                mass_matrix=comb_implicit_mat)

    # 断言尝试访问dyn_implicit_mat属性时会引发AttributeError异常
    with raises(AttributeError):
        symsystem2.dyn_implicit_mat
    # 断言尝试访问dyn_implicit_rhs属性时会引发AttributeError异常
    with raises(AttributeError):
        symsystem2.dyn_implicit_rhs
    # 断言尝试访问kin_explicit_rhs属性时会引发AttributeError异常
    with raises(AttributeError):
        symsystem2.kin_explicit_rhs

    # 断言尝试访问coordinates和speeds属性时会引发AttributeError异常，因为仅指定了states
    with raises(AttributeError):
        symsystem1.coordinates
    with raises(AttributeError):
        symsystem1.speeds

    # 断言尝试访问bodies和loads属性时会引发AttributeError异常，因为它们未被指定
    with raises(AttributeError):
        symsystem1.bodies
    with raises(AttributeError):
        symsystem1.loads

    # 断言尝试访问comb_explicit_rhs属性时会引发AttributeError异常，因为它在计算之前未被指定
    with raises(AttributeError):
        symsystem2.comb_explicit_rhs
```