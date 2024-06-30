# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_models.py`

```
# 导入 sympy.physics.mechanics.models 模块中的 models
# 从 sympy 模块导入 cos, sin, Matrix, symbols, zeros
# 从 sympy.simplify.simplify 模块导入 simplify 函数
# 从 sympy.physics.mechanics 模块中导入 dynamicsymbols
import sympy.physics.mechanics.models as models
from sympy import (cos, sin, Matrix, symbols, zeros)
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols)

# 定义测试函数 test_multi_mass_spring_damper_inputs
def test_multi_mass_spring_damper_inputs():

    # 定义符号变量 c0, k0, m0
    c0, k0, m0 = symbols("c0 k0 m0")
    # 定义符号变量 g
    g = symbols("g")
    # 定义动态符号变量 v0, x0, f0
    v0, x0, f0 = dynamicsymbols("v0 x0 f0")

    # 使用 models 模块中的 multi_mass_spring_damper 函数创建 kane1 对象
    kane1 = models.multi_mass_spring_damper(1)
    # 创建质量矩阵 massmatrix1
    massmatrix1 = Matrix([[m0]])
    # 创建强制矩阵 forcing1
    forcing1 = Matrix([[-c0*v0 - k0*x0]])
    # 断言验证质量矩阵的简化结果与 kane1 的质量矩阵相等
    assert simplify(massmatrix1 - kane1.mass_matrix) == Matrix([0])
    # 断言验证强制矩阵的简化结果与 kane1 的强制矩阵相等
    assert simplify(forcing1 - kane1.forcing) == Matrix([0])

    # 使用 models 模块中的 multi_mass_spring_damper 函数创建 kane2 对象，启用重力
    kane2 = models.multi_mass_spring_damper(1, True)
    # 创建质量矩阵 massmatrix2
    massmatrix2 = Matrix([[m0]])
    # 创建强制矩阵 forcing2
    forcing2 = Matrix([[-c0*v0 + g*m0 - k0*x0]])
    # 断言验证质量矩阵的简化结果与 kane2 的质量矩阵相等
    assert simplify(massmatrix2 - kane2.mass_matrix) == Matrix([0])
    # 断言验证强制矩阵的简化结果与 kane2 的强制矩阵相等
    assert simplify(forcing2 - kane2.forcing) == Matrix([0])

    # 使用 models 模块中的 multi_mass_spring_damper 函数创建 kane3 对象，启用重力和外力
    kane3 = models.multi_mass_spring_damper(1, True, True)
    # 创建质量矩阵 massmatrix3
    massmatrix3 = Matrix([[m0]])
    # 创建强制矩阵 forcing3
    forcing3 = Matrix([[-c0*v0 + g*m0 - k0*x0 + f0]])
    # 断言验证质量矩阵的简化结果与 kane3 的质量矩阵相等
    assert simplify(massmatrix3 - kane3.mass_matrix) == Matrix([0])
    # 断言验证强制矩阵的简化结果与 kane3 的强制矩阵相等
    assert simplify(forcing3 - kane3.forcing) == Matrix([0])

    # 使用 models 模块中的 multi_mass_spring_damper 函数创建 kane4 对象，仅启用外力
    kane4 = models.multi_mass_spring_damper(1, False, True)
    # 创建质量矩阵 massmatrix4
    massmatrix4 = Matrix([[m0]])
    # 创建强制矩阵 forcing4
    forcing4 = Matrix([[-c0*v0 - k0*x0 + f0]])
    # 断言验证质量矩阵的简化结果与 kane4 的质量矩阵相等
    assert simplify(massmatrix4 - kane4.mass_matrix) == Matrix([0])
    # 断言验证强制矩阵的简化结果与 kane4 的强制矩阵相等
    assert simplify(forcing4 - kane4.forcing) == Matrix([0])


# 定义测试函数 test_multi_mass_spring_damper_higher_order
def test_multi_mass_spring_damper_higher_order():
    # 定义符号变量 c0, k0, m0
    c0, k0, m0 = symbols("c0 k0 m0")
    # 定义符号变量 c1, k1, m1
    c1, k1, m1 = symbols("c1 k1 m1")
    # 定义符号变量 c2, k2, m2
    c2, k2, m2 = symbols("c2 k2 m2")
    # 定义动态符号变量 v0, x0
    v0, x0 = dynamicsymbols("v0 x0")
    # 定义动态符号变量 v1, x1
    v1, x1 = dynamicsymbols("v1 x1")
    # 定义动态符号变量 v2, x2
    v2, x2 = dynamicsymbols("v2 x2")

    # 使用 models 模块中的 multi_mass_spring_damper 函数创建 kane1 对象
    kane1 = models.multi_mass_spring_damper(3)
    # 创建质量矩阵 massmatrix1
    massmatrix1 = Matrix([[m0 + m1 + m2, m1 + m2, m2],
                          [m1 + m2, m1 + m2, m2],
                          [m2, m2, m2]])
    # 创建强制矩阵 forcing1
    forcing1 = Matrix([[-c0*v0 - k0*x0],
                       [-c1*v1 - k1*x1],
                       [-c2*v2 - k2*x2]])
    # 断言验证质量矩阵的简化结果与 kane1 的质量矩阵相等
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(3)
    # 断言验证强制矩阵的简化结果与 kane1 的强制矩阵相等
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0, 0])


# 定义测试函数 test_n_link_pendulum_on_cart_inputs
def test_n_link_pendulum_on_cart_inputs():
    # 定义符号变量 l0, m0
    l0, m0 = symbols("l0 m0")
    # 定义符号变量 m1
    m1 = symbols("m1")
    # 定义符号变量 g
    g = symbols("g")
    # 定义动态符号变量 q0, q1, F, T1
    q0, q1, F, T1 = dynamicsymbols("q0 q1 F T1")
    # 定义动态符号变量 u0, u1
    u0, u1 = dynamicsymbols("u0 u1")

    # 使用 models 模块中的 n_link_pendulum_on_cart 函数创建 kane1 对象
    kane1 = models.n_link_pendulum_on_cart(1)
    # 创建质量矩阵 massmatrix1
    massmatrix1 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    # 创建强制矩阵 forcing1
    forcing1 = Matrix([[-l0*m1*u1**2*sin(q1) + F], [g*l0*m1*sin(q1)]])
    # 断言验证质量矩阵的简化结果与 kane1 的质量矩阵相等
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(2)
    # 断言验证强制矩阵的简化结果与 kane1 的强制矩阵相等
    assert simplify(forcing1 - kane1.forcing) == Matrix([0,
    # 确保计算得到的简化结果与零矩阵相等，表明两个表达式在数值上完全相等
    assert simplify(forcing2 - kane2.forcing) == Matrix([0, 0])

    # 创建一个包含非线性摆动的模型对象，不考虑外力，考虑惯性力
    kane3 = models.n_link_pendulum_on_cart(1, False, True)
    # 定义与该模型相关的质量矩阵
    massmatrix3 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    # 定义施加于模型上的外力向量
    forcing3 = Matrix([[-l0*m1*u1**2*sin(q1)], [g*l0*m1*sin(q1) + T1]])
    # 确保计算得到的简化质量矩阵与模型中的质量矩阵相等
    assert simplify(massmatrix3 - kane3.mass_matrix) == zeros(2)
    # 确保计算得到的简化外力向量与模型中的外力向量相等
    assert simplify(forcing3 - kane3.forcing) == Matrix([0, 0])

    # 创建一个包含非线性摆动的模型对象，考虑外力，不考虑惯性力
    kane4 = models.n_link_pendulum_on_cart(1, True, False)
    # 定义与该模型相关的质量矩阵
    massmatrix4 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    # 定义施加于模型上的外力向量，包括一个额外的力 F
    forcing4 = Matrix([[-l0*m1*u1**2*sin(q1) + F], [g*l0*m1*sin(q1)]])
    # 确保计算得到的简化质量矩阵与模型中的质量矩阵相等
    assert simplify(massmatrix4 - kane4.mass_matrix) == zeros(2)
    # 确保计算得到的简化外力向量与模型中的外力向量相等
    assert simplify(forcing4 - kane4.forcing) == Matrix([0, 0])
# 定义一个测试函数，用于测试 n-link pendulum on cart 模型的高阶动力学表达式
def test_n_link_pendulum_on_cart_higher_order():
    # 定义符号变量 l0, m0 表示长度和质量
    l0, m0 = symbols("l0 m0")
    # 定义符号变量 l1, m1 表示长度和质量
    l1, m1 = symbols("l1 m1")
    # 定义符号变量 m2 表示质量
    m2 = symbols("m2")
    # 定义符号变量 g 表示重力加速度
    g = symbols("g")
    # 定义动力学符号变量 q0, q1, q2 表示广义坐标
    q0, q1, q2 = dynamicsymbols("q0 q1 q2")
    # 定义动力学符号变量 u0, u1, u2 表示广义速度
    u0, u1, u2 = dynamicsymbols("u0 u1 u2")
    # 定义动力学符号变量 F, T1 表示外力和转矩

    # 使用 models 模块中的 n_link_pendulum_on_cart 函数创建 kane1 对象
    kane1 = models.n_link_pendulum_on_cart(2)
    # 创建质量矩阵 massmatrix1，包含质量和惯性项
    massmatrix1 = Matrix([[m0 + m1 + m2, -l0*m1*cos(q1) - l0*m2*cos(q1),
                           -l1*m2*cos(q2)],
                          [-l0*m1*cos(q1) - l0*m2*cos(q1), l0**2*m1 + l0**2*m2,
                           l0*l1*m2*(sin(q1)*sin(q2) + cos(q1)*cos(q2))],
                          [-l1*m2*cos(q2),
                           l0*l1*m2*(sin(q1)*sin(q2) + cos(q1)*cos(q2)),
                           l1**2*m2]])
    # 创建 forcing1，表示外力和约束反力项
    forcing1 = Matrix([[-l0*m1*u1**2*sin(q1) - l0*m2*u1**2*sin(q1) -
                        l1*m2*u2**2*sin(q2) + F],
                       [g*l0*m1*sin(q1) + g*l0*m2*sin(q1) -
                        l0*l1*m2*(sin(q1)*cos(q2) - sin(q2)*cos(q1))*u2**2],
                       [g*l1*m2*sin(q2) - l0*l1*m2*(-sin(q1)*cos(q2) +
                                                    sin(q2)*cos(q1))*u1**2]])
    # 断言质量矩阵的简化结果与 kane1 中的质量矩阵相等
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(3)
    # 断言外力项的简化结果与 kane1 中的外力项相等
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0, 0])
```