# `D:\src\scipysrc\sympy\sympy\solvers\ode\tests\test_systems.py`

```
# 导入所需的符号和函数模块
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
                                       ODEOrderError, ODENonlinearError, _simpsol,
                                       _is_commutative_anti_derivative, linodesolve,
                                       canonical_odes, dsolve_system, _component_division,
                                       _eqs2dict, _dict2graph)
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import raises, slow, tooslow, XFAIL

# 定义一些符号变量
C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10 = symbols('C0:11')
x = symbols('x')
f = Function('f')
g = Function('g')
h = Function('h')

# 定义测试函数 test_linear_ode_to_matrix
def test_linear_ode_to_matrix():
    # 定义符号变量 f, g, h 作为函数
    f, g, h = symbols("f, g, h", cls=Function)
    # 定义符号变量 t
    t = Symbol("t")
    # 函数列表，包含 f(t), g(t), h(t)
    funcs = [f(t), g(t), h(t)]
    # 定义一阶导数
    f1 = f(t).diff(t)
    g1 = g(t).diff(t)
    h1 = h(t).diff(t)
    # 定义二阶导数
    f2 = f(t).diff(t, 2)
    g2 = g(t).diff(t, 2)
    h2 = h(t).diff(t, 2)

    # 第一个测试用例
    eqs_1 = [Eq(f1, g(t)), Eq(g1, f(t))]
    sol_1 = ([Matrix([[1, 0], [0, 1]]), Matrix([[ 0, 1], [1,  0]])], Matrix([[0],[0]]))
    assert linear_ode_to_matrix(eqs_1, funcs[:-1], t, 1) == sol_1

    # 第二个测试用例
    eqs_2 = [Eq(f1, f(t) + 2*g(t)), Eq(g1, h(t)), Eq(h1, g(t) + h(t) + f(t))]
    sol_2 = ([Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), Matrix([[1, 2,  0], [ 0,  0, 1], [1, 1, 1]])],
             Matrix([[0], [0], [0]]))
    assert linear_ode_to_matrix(eqs_2, funcs, t, 1) == sol_2

    # 第三个测试用例
    eqs_3 = [Eq(2*f1 + 3*h1, f(t) + g(t)), Eq(4*h1 + 5*g1, f(t) + h(t)), Eq(5*f1 + 4*g1, g(t) + h(t))]
    sol_3 = ([Matrix([[2, 0, 3], [0, 5, 4], [5, 4, 0]]), Matrix([[1, 1,  0], [1,  0, 1], [0, 1, 1]])],
             Matrix([[0], [0], [0]]))
    assert linear_ode_to_matrix(eqs_3, funcs, t, 1) == sol_3

    # 第四个测试用例
    eqs_4 = [Eq(f2 + h(t), f1 + g(t)), Eq(2*h2 + g2 + g1 + g(t), 0), Eq(3*h1, 4)]
    sol_4 = ([Matrix([[1, 0, 0], [0, 1, 2], [0, 0, 0]]), Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -3]]),
              Matrix([[0, 1, -1], [0,  -1, 0], [0, 0, 0]])], Matrix([[0], [0], [4]]))
    assert linear_ode_to_matrix(eqs_4, funcs, t, 2) == sol_4

    # 第五个测试用例，测试非线性方程组抛出的异常
    eqs_5 = [Eq(f2, g(t)), Eq(f1 + g1, f(t))]
    raises(ODEOrderError, lambda: linear_ode_to_matrix(eqs_5, funcs[:-1], t, 1))
    # 创建包含两个方程的列表，每个方程包含一个函数与一个表达式的相等关系
    eqs_6 = [Eq(f1, f(t)**2), Eq(g1, f(t) + g(t))]
    # 使用 lambda 函数和 raises 函数测试 linear_ode_to_matrix 是否会引发 ODENonlinearError 异常
    raises(ODENonlinearError, lambda: linear_ode_to_matrix(eqs_6, funcs[:-1], t, 1))
def test__classify_linear_system():
    # 定义符号变量 x, y, z, w，并将它们作为函数对象
    x, y, z, w = symbols('x, y, z, w', cls=Function)
    # 定义额外的符号变量 t, k, l
    t, k, l = symbols('t k l')
    
    # 计算各个函数的一阶导数
    x1 = diff(x(t), t)
    y1 = diff(y(t), t)
    z1 = diff(z(t), t)
    w1 = diff(w(t), t)
    
    # 计算各个函数的二阶导数
    x2 = diff(x(t), t, t)
    y2 = diff(y(t), t, t)
    
    # 创建包含函数 x(t), y(t) 的列表
    funcs = [x(t), y(t)]
    # 创建包含函数 x(t), y(t), z(t), w(t) 的列表
    funcs_2 = funcs + [z(t), w(t)]
    
    # 定义线性系统方程组 eqs_1，并验证其结果为 None
    eqs_1 = (5 * x1 + 12 * x(t) - 6 * (y(t)), (2 * y1 - 11 * t * x(t) + 3 * y(t) + t))
    assert _classify_linear_system(eqs_1, funcs, t) is None
    
    # 定义线性系统方程组 eqs_2，并验证其结果与预期的 sol2 相同
    eqs_2 = (5 * (x1**2) + 12 * x(t) - 6 * (y(t)), (2 * y1 - 11 * t * x(t) + 3 * y(t) + t))
    sol2 = {'is_implicit': True,
            'canon_eqs': [[Eq(Derivative(x(t), t), -sqrt(-12*x(t)/5 + 6*y(t)/5)),
                           Eq(Derivative(y(t), t), 11*t*x(t)/2 - t/2 - 3*y(t)/2)],
                          [Eq(Derivative(x(t), t), sqrt(-12*x(t)/5 + 6*y(t)/5)),
                           Eq(Derivative(y(t), t), 11*t*x(t)/2 - t/2 - 3*y(t)/2)]]}
    assert _classify_linear_system(eqs_2, funcs, t) == sol2
    
    # 定义非齐次线性系统方程组 eqs_2_1，并验证其结果为 None
    eqs_2_1 = [Eq(Derivative(x(t), t), -sqrt(-12*x(t)/5 + 6*y(t)/5)),
               Eq(Derivative(y(t), t), 11*t*x(t)/2 - t/2 - 3*y(t)/2)]
    assert _classify_linear_system(eqs_2_1, funcs, t) is None
    
    # 定义非齐次线性系统方程组 eqs_2_2，并验证其结果为 None
    eqs_2_2 = [Eq(Derivative(x(t), t), sqrt(-12*x(t)/5 + 6*y(t)/5)),
               Eq(Derivative(y(t), t), 11*t*x(t)/2 - t/2 - 3*y(t)/2)]
    assert _classify_linear_system(eqs_2_2, funcs, t) is None
    
    # 定义包含多个方程的线性系统方程组 eqs_3，并验证其结果与预期的 answer_3 相同
    eqs_3 = (5 * x1 + 12 * x(t) - 6 * (y(t)), (2 * y1 - 11 * x(t) + 3 * y(t)), (5 * w1 + z(t)), (z1 + w(t)))
    answer_3 = {'no_of_equation': 4,
                'eq': (12*x(t) - 6*y(t) + 5*Derivative(x(t), t),
                       -11*x(t) + 3*y(t) + 2*Derivative(y(t), t),
                       z(t) + 5*Derivative(w(t), t),
                       w(t) + Derivative(z(t), t)),
                'func': [x(t), y(t), z(t), w(t)],
                'order': {x(t): 1, y(t): 1, z(t): 1, w(t): 1},
                'is_linear': True,
                'is_constant': True,
                'is_homogeneous': True,
                'func_coeff': -Matrix([
                    [Rational(12, 5), Rational(-6, 5), 0, 0],
                    [Rational(-11, 2), Rational(3, 2), 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, Rational(1, 5), 0]]),
                'type_of_equation': 'type1',
                'is_general': True}
    assert _classify_linear_system(eqs_3, funcs_2, t) == answer_3
    
    # 定义包含多个方程的线性系统方程组 eqs_4，并验证其结果与预期的 answer_4 相同
    eqs_4 = (5 * x1 + 12 * x(t) - 6 * (y(t)), (2 * y1 - 11 * x(t) + 3 * y(t)), (z1 - w(t)), (w1 - z(t)))
    answer_4 = {'no_of_equation': 4,
                'eq': (12 * x(t) - 6 * y(t) + 5 * Derivative(x(t), t),
                       -11 * x(t) + 3 * y(t) + 2 * Derivative(y(t), t),
                       -w(t) + Derivative(z(t), t),
                       -z(t) + Derivative(w(t), t)),
                'func': [x(t), y(t), z(t), w(t)],
                'order': {x(t): 1, y(t): 1, z(t): 1, w(t): 1},
                'is_linear': True,
                'is_constant': True,
                'is_homogeneous': True,
                'func_coeff': -Matrix([
                    [Rational(12, 5), Rational(-6, 5), 0, 0],
                    [Rational(-11, 2), Rational(3, 2), 0, 0],
                    [0, 0, 0, -1],
                    [0, 0, -1, 0]]),
                'type_of_equation': 'type1',
                'is_general': True}
    assert _classify_linear_system(eqs_4, funcs_2, t) == answer_4
    # 定义线性系统的方程组 eqs_5，包括多个变量和导数
    eqs_5 = (5*x1 + 12*x(t) - 6*(y(t)) + x2, (2*y1 - 11*x(t) + 3*y(t)), (z1 - w(t)), (w1 - z(t)))
    # 定义预期的答案 answer_5，包含方程数量、方程表达式、函数列表、阶数、线性性质等信息
    answer_5 = {'no_of_equation': 4, 'eq': (12*x(t) - 6*y(t) + 5*Derivative(x(t), t) + Derivative(x(t), (t, 2)),
                -11*x(t) + 3*y(t) + 2*Derivative(y(t), t), -w(t) + Derivative(z(t), t), -z(t) + Derivative(w(t),
                t)), 'func': [x(t), y(t), z(t), w(t)], 'order': {x(t): 2, y(t): 1, z(t): 1, w(t): 1}, 'is_linear':
                True, 'is_homogeneous': True, 'is_general': True, 'type_of_equation': 'type0', 'is_higher_order': True}
    # 使用断言验证函数 _classify_linear_system 对 eqs_5 的分类结果与 answer_5 是否一致
    assert _classify_linear_system(eqs_5, funcs_2, t) == answer_5
    
    # 定义线性系统的方程组 eqs_6，使用 SymPy 的 Eq 函数表示方程
    eqs_6 = (Eq(x1, 3*y(t) - 11*z(t)), Eq(y1, 7*z(t) - 3*x(t)), Eq(z1, 11*x(t) - 7*y(t)))
    # 定义预期的答案 answer_6，包含方程数量、方程表达式、函数列表、阶数、线性性质、系数矩阵等信息
    answer_6 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)), Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),
            Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1},
            'is_linear': True, 'is_constant': True, 'is_homogeneous': True,
            'func_coeff': -Matrix([
                         [  0, -3, 11],
                         [  3,  0, -7],
                         [-11,  7,  0]]),
            'type_of_equation': 'type1', 'is_general': True}
    # 使用断言验证函数 _classify_linear_system 对 eqs_6 的分类结果与 answer_6 是否一致
    assert _classify_linear_system(eqs_6, funcs_2[:-1], t) == answer_6
    
    # 定义线性系统的方程组 eqs_7，使用 SymPy 的 Eq 函数表示方程
    eqs_7 = (Eq(x1, y(t)), Eq(y1, x(t)))
    # 定义预期的答案 answer_7，包含方程数量、方程表达式、函数列表、阶数、线性性质、系数矩阵等信息
    answer_7 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), y(t)), Eq(Derivative(y(t), t), x(t))),
                'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': True,
                'is_homogeneous': True, 'func_coeff': -Matrix([
                                                        [ 0, -1],
                                                        [-1,  0]]),
                'type_of_equation': 'type1', 'is_general': True}
    # 使用断言验证函数 _classify_linear_system 对 eqs_7 的分类结果与 answer_7 是否一致
    assert _classify_linear_system(eqs_7, funcs, t) == answer_7
    
    # 定义线性系统的方程组 eqs_8，使用 SymPy 的 Eq 函数表示方程
    eqs_8 = (Eq(x1, 21*x(t)), Eq(y1, 17*x(t) + 3*y(t)), Eq(z1, 5*x(t) + 7*y(t) + 9*z(t)))
    # 定义预期的答案 answer_8，包含方程数量、方程表达式、函数列表、阶数、线性性质、系数矩阵等信息
    answer_8 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 21*x(t)), Eq(Derivative(y(t), t), 17*x(t) + 3*y(t)),
            Eq(Derivative(z(t), t), 5*x(t) + 7*y(t) + 9*z(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1},
            'is_linear': True, 'is_constant': True, 'is_homogeneous': True,
            'func_coeff': -Matrix([
                            [-21,  0,  0],
                            [-17, -3,  0],
                            [ -5, -7, -9]]),
            'type_of_equation': 'type1', 'is_general': True}
    # 使用断言验证函数 _classify_linear_system 对 eqs_8 的分类结果与 answer_8 是否一致
    assert _classify_linear_system(eqs_8, funcs_2[:-1], t) == answer_8
    
    # 定义线性系统的方程组 eqs_9，使用 SymPy 的 Eq 函数表示方程
    eqs_9 = (Eq(x1, 4*x(t) + 5*y(t) + 2*z(t)), Eq(y1, x(t) + 13*y(t) + 9*z(t)), Eq(z1, 32*x(t) + 41*y(t) + 11*z(t)))
    #python
    # 定义一个包含线性方程组相关信息的字典 answer_9
    answer_9 = {'no_of_equation': 3,  # 方程数量为 3
                'eq': (Eq(Derivative(x(t), t), 4*x(t) + 5*y(t) + 2*z(t)),  # 第一个方程
                       Eq(Derivative(y(t), t), x(t) + 13*y(t) + 9*z(t)),   # 第二个方程
                       Eq(Derivative(z(t), t), 32*x(t) + 41*y(t) + 11*z(t))),  # 第三个方程
                'func': [x(t), y(t), z(t)],  # 方程中涉及的函数列表
                'order': {x(t): 1, y(t): 1, z(t): 1},  # 函数的阶数字典
                'is_linear': True,  # 方程组是否为线性的
                'is_constant': True,  # 方程组中的系数是否为常数
                'is_homogeneous': True,  # 方程组是否是齐次的
                'func_coeff': -Matrix([
                            [ -4,  -5,  -2],  # 函数系数的矩阵表示
                            [ -1, -13,  -9],
                            [-32, -41, -11]]),  # 函数系数的具体数值
                'type_of_equation': 'type1',  # 方程组的类型
                'is_general': True}  # 方程组是否为一般类型

    # 断言 _classify_linear_system 函数对 eqs_9 的输出与 answer_9 相等
    assert _classify_linear_system(eqs_9, funcs_2[:-1], t) == answer_9

    # 定义包含线性方程组信息的字典 answer_10
    eqs_10 = (Eq(3*x1, 4*5*(y(t) - z(t))),  # 第一个方程
              Eq(4*y1, 3*5*(z(t) - x(t))),  # 第二个方程
              Eq(5*z1, 3*4*(x(t) - y(t))))  # 第三个方程
    answer_10 = {'no_of_equation': 3,  # 方程数量为 3
                 'eq': (Eq(3*Derivative(x(t), t), 20*y(t) - 20*z(t)),  # 第一个方程
                        Eq(4*Derivative(y(t), t), -15*x(t) + 15*z(t)),  # 第二个方程
                        Eq(5*Derivative(z(t), t), 12*x(t) - 12*y(t))),  # 第三个方程
                 'func': [x(t), y(t), z(t)],  # 方程中涉及的函数列表
                 'order': {x(t): 1, y(t): 1, z(t): 1},  # 函数的阶数字典
                 'is_linear': True,  # 方程组是否为线性的
                 'is_constant': True,  # 方程组中的系数是否为常数
                 'is_homogeneous': True,  # 方程组是否是齐次的
                 'func_coeff': -Matrix([
                                [  0, Rational(-20, 3),  Rational(20, 3)],  # 函数系数的矩阵表示
                                [Rational(15, 4),     0, Rational(-15, 4)],
                                [Rational(-12, 5), Rational(12, 5),  0]]),  # 函数系数的具体数值
                 'type_of_equation': 'type1',  # 方程组的类型
                 'is_general': True}  # 方程组是否为一般类型

    # 断言 _classify_linear_system 函数对 eqs_10 的输出与 answer_10 相等
    assert _classify_linear_system(eqs_10, funcs_2[:-1], t) == answer_10

    # 定义包含线性方程组信息的字典 sol11
    eq11 = (Eq(x1, 3*y(t) - 11*z(t)),  # 第一个方程
            Eq(y1, 7*z(t) - 3*x(t)),  # 第二个方程
            Eq(z1, 11*x(t) - 7*y(t)))  # 第三个方程
    sol11 = {'no_of_equation': 3,  # 方程数量为 3
             'eq': (Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)),  # 第一个方程
                    Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),  # 第二个方程
                    Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))),  # 第三个方程
             'func': [x(t), y(t), z(t)],  # 方程中涉及的函数列表
             'order': {x(t): 1, y(t): 1, z(t): 1},  # 函数的阶数字典
             'is_linear': True,  # 方程组是否为线性的
             'is_constant': True,  # 方程组中的系数是否为常数
             'is_homogeneous': True,  # 方程组是否是齐次的
             'func_coeff': -Matrix([
                            [  0, -3, 11],  # 函数系数的矩阵表示
                            [  3,  0, -7],
                            [-11,  7,  0]]),  # 函数系数的具体数值
             'type_of_equation': 'type1',  # 方程组的类型
             'is_general': True}  # 方程组是否为一般类型

    # 断言 _classify_linear_system 函数对 eq11 的输出与 sol11 相等
    assert _classify_linear_system(eq11, funcs_2[:-1], t) == sol11

    # 定义包含线性方程组信息的字典 sol12
    eq12 = (Eq(Derivative(x(t), t), y(t)),  # 第一个方程
            Eq(Derivative(y(t), t), x(t)))  # 第二个方程
    sol12 = {'no_of_equation': 2,  # 方程数量为 2
             'eq': (Eq(Derivative(x(t), t), y(t)),  # 第一个方程
                    Eq(Derivative(y(t), t), x(t))),  # 第二个方程
             'func': [x(t), y(t)],  # 方程中涉及的函数列表
             'order': {x(t): 1, y(t): 1},  # 函数的阶数字典
             'is_linear': True,  # 方程组是否为线性的
             'is_constant': True,  # 方程组中的系数是否为常数
             'is_homogeneous': True,  # 方程组是否是齐次的
             'func_coeff': -Matrix([
                            [0, -1],  # 函数系数的矩阵表示
                            [-1, 0]]),  # 函数系数的具体数值
             'type_of_equation': 'type1',  # 方程组的类型
             'is_general': True}  # 方程组是否为一般类型

    # 断言 _classify_linear_system 函数对 eq12 的输出与 sol12 相等
    assert _classify_linear_system(eq12, [x(t), y(t)], t) == sol12

    # 定义包含线性方程组信息的字典 sol13
    eq13 = (Eq(Derivative(x(t), t), 21*x(t)),  # 第一个方程
            Eq(Derivative(y(t), t), 17*x(t) + 3*y(t)),  # 第二个方程
            Eq(Derivative(z(t), t), 5*x(t) + 7*y(t) + 9*z(t)))  # 第三个方程
    sol13 = {'no_of_equation': 3,  # 方程数量为 3
             'eq': (Eq(Derivative(x(t), t), 21 * x(t)),  # 第一个方程
                    Eq(Derivative(y(t), t), 17 * x(t) + 3 * y(t)),  # 第二个方程
                    Eq(Derivative(z(t), t), 5 * x(t) + 7 * y(t) + 9 * z(t))),  #```
    # 定义一个包含线性方程组信息的字典，描述了方程的性质、系数和类型
    answer_9 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
                'eq': (Eq(Derivative(x(t), t), 4*x(t) + 5*y(t) + 2*z(t)),  # 第一个方程
                       Eq(Derivative(y(t), t), x(t) + 13*y(t) + 9*z(t)),   # 第二个方程
                       Eq(Derivative(z(t), t), 32*x(t) + 41*y(t) + 11*z(t))),  # 第三个方程
                'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
                'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
                'is_linear': True,  # 方程组是否为线性的
                'is_constant': True,  # 方程组是否为常数系数的
                'is_homogeneous': True,  # 方程组是否为齐次的
                'func_coeff': -Matrix([
                            [ -4,  -5,  -2],  # 函数的系数矩阵的负值
                            [ -1, -13,  -9],
                            [-32, -41, -11]]),  # 各项系数的负值矩阵
                'type_of_equation': 'type1',  # 方程组的类型
                'is_general': True}  # 方程组是否为一般形式的
    # 验证分类线性系统的函数对给定的方程组是否返回了预期的结果字典
    assert _classify_linear_system(eqs_9, funcs_2[:-1], t) == answer_9

    # 定义另一个线性方程组的元组
    eqs_10 = (Eq(3*x1, 4*5*(y(t) - z(t))),  # 第一个方程
              Eq(4*y1, 3*5*(z(t) - x(t))),  # 第二个方程
              Eq(5*z1, 3*4*(x(t) - y(t))))  # 第三个方程
    # 对应的预期结果字典
    answer_10 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
                 'eq': (Eq(3*Derivative(x(t), t), 20*y(t) - 20*z(t)),  # 第一个方程
                        Eq(4*Derivative(y(t), t), -15*x(t) + 15*z(t)),  # 第二个方程
                        Eq(5*Derivative(z(t), t), 12*x(t) - 12*y(t))),  # 第三个方程
                 'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
                 'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
                 'is_linear': True,  # 方程组是否为线性的
                 'is_constant': True,  # 方程组是否为常数系数的
                 'is_homogeneous': True,  # 方程组是否为齐次的
                 'func_coeff': -Matrix([
                                [  0, Rational(-20, 3),  Rational(20, 3)],  # 函数的系数矩阵的负值
                                [Rational(15, 4),     0, Rational(-15, 4)],
                                [Rational(-12, 5), Rational(12, 5),  0]]),  # 各项系数的负值矩阵
                 'type_of_equation': 'type1',  # 方程组的类型
                 'is_general': True}  # 方程组是否为一般形式的
    # 验证分类线性系统的函数对给定的方程组是否返回了预期的结果字典
    assert _classify_linear_system(eqs_10, funcs_2[:-1], t) == answer_10

    # 第三个方程组的定义
    eq11 = (Eq(x1, 3*y(t) - 11*z(t)),  # 第一个方程
            Eq(y1, 7*z(t) - 3*x(t)),  # 第二个方程
            Eq(z1, 11*x(t) - 7*y(t)))  # 第三个方程
    # 对应的预期结果字典
    sol11 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
             'eq': (Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)),  # 第一个方程
                    Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),  # 第二个方程
                    Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))),  # 第三个方程
             'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
             'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
             'is_linear': True,  # 方程组是否为线性的
             'is_constant': True,  # 方程组是否为```
    # 定义一个包含线性方程组信息的字典，描述了方程的性质、系数和类型
    answer_9 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
                'eq': (Eq(Derivative(x(t), t), 4*x(t) + 5*y(t) + 2*z(t)),  # 第一个方程
                       Eq(Derivative(y(t), t), x(t) + 13*y(t) + 9*z(t)),   # 第二个方程
                       Eq(Derivative(z(t), t), 32*x(t) + 41*y(t) + 11*z(t))),  # 第三个方程
                'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
                'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
                'is_linear': True,  # 方程组是否为线性的
                'is_constant': True,  # 方程组是否为常数系数的
                'is_homogeneous': True,  # 方程组是否为齐次的
                'func_coeff': -Matrix([
                            [ -4,  -5,  -2],  # 函数的系数矩阵的负值
                            [ -1, -13,  -9],
                            [-32, -41, -11]]),  # 各项系数的负值矩阵
                'type_of_equation': 'type1',  # 方程组的类型
                'is_general': True}  # 方程组是否为一般形式的
    # 验证分类线性系统的函数对给定的方程组是否返回了预期的结果字典
    assert _classify_linear_system(eqs_9, funcs_2[:-1], t) == answer_9

    # 定义另一个线性方程组的元组
    eqs_10 = (Eq(3*x1, 4*5*(y(t) - z(t))),  # 第一个方程
              Eq(4*y1, 3*5*(z(t) - x(t))),  # 第二个方程
              Eq(5*z1, 3*4*(x(t) - y(t))))  # 第三个方程
    # 对应的预期结果字典
    answer_10 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
                 'eq': (Eq(3*Derivative(x(t), t), 20*y(t) - 20*z(t)),  # 第一个方程
                        Eq(4*Derivative(y(t), t), -15*x(t) + 15*z(t)),  # 第二个方程
                        Eq(5*Derivative(z(t), t), 12*x(t) - 12*y(t))),  # 第三个方程
                 'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
                 'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
                 'is_linear': True,  # 方程组是否为线性的
                 'is_constant': True,  # 方程组是否为常数系数的
                 'is_homogeneous': True,  # 方程组是否为齐次的
                 'func_coeff': -Matrix([
                                [  0, Rational(-20, 3),  Rational(20, 3)],  # 函数的系数矩阵的负值
                                [Rational(15, 4),     0, Rational(-15, 4)],
                                [Rational(-12, 5), Rational(12, 5),  0]]),  # 各项系数的负值矩阵
                 'type_of_equation': 'type1',  # 方程组的类型
                 'is_general': True}  # 方程组是否为一般形式的
    # 验证分类线性系统的函数对给定的方程组是否返回了预期的结果字典
    assert _classify_linear_system(eqs_10, funcs_2[:-1], t) == answer_10

    # 第三个方程组的定义
    eq11 = (Eq(x1, 3*y(t) - 11*z(t)),  # 第一个方程
            Eq(y1, 7*z(t) - 3*x(t)),  # 第二个方程
            Eq(z1, 11*x(t) - 7*y(t)))  # 第三个方程
    # 对应的预期结果字典
    sol11 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
             'eq': (Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)),  # 第一个方程
                    Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),  # 第二个方程
                    Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))),  # 第三个方程
             'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
             'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
             'is_linear': True,  # 方程组是否为线性的
             'is_constant': True,  # 方程组是否为常```
    # 定义一个包含线性方程组信息的字典，描述了方程的性质、系数和类型
    answer_9 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
                'eq': (Eq(Derivative(x(t), t), 4*x(t) + 5*y(t) + 2*z(t)),  # 第一个方程
                       Eq(Derivative(y(t), t), x(t) + 13*y(t) + 9*z(t)),   # 第二个方程
                       Eq(Derivative(z(t), t), 32*x(t) + 41*y(t) + 11*z(t))),  # 第三个方程
                'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
                'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
                'is_linear': True,  # 方程组是否为线性的
                'is_constant': True,  # 方程组是否为常数系数的
                'is_homogeneous': True,  # 方程组是否为齐次的
                'func_coeff': -Matrix([
                            [ -4,  -5,  -2],  # 函数的系数矩阵的负值
                            [ -1, -13,  -9],
                            [-32, -41, -11]]),  # 各项系数的负值矩阵
                'type_of_equation': 'type1',  # 方程组的类型
                'is_general': True}  # 方程组是否为一般形式的
    # 验证分类线性系统的函数对给定的方程组是否返回了预期的结果字典
    assert _classify_linear_system(eqs_9, funcs_2[:-1], t) == answer_9

    # 定义另一个线性方程组的元组
    eqs_10 = (Eq(3*x1, 4*5*(y(t) - z(t))),  # 第一个方程
              Eq(4*y1, 3*5*(z(t) - x(t))),  # 第二个方程
              Eq(5*z1, 3*4*(x(t) - y(t))))  # 第三个方程
    # 对应的预期结果字典
    answer_10 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
                 'eq': (Eq(3*Derivative(x(t), t), 20*y(t) - 20*z(t)),  # 第一个方程
                        Eq(4*Derivative(y(t), t), -15*x(t) + 15*z(t)),  # 第二个方程
                        Eq(5*Derivative(z(t), t), 12*x(t) - 12*y(t))),  # 第三个方程
                 'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
                 'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
                 'is_linear': True,  # 方程组是否为线性的
                 'is_constant': True,  # 方程组是否为常数系数的
                 'is_homogeneous': True,  # 方程组是否为齐次的
                 'func_coeff': -Matrix([
                                [  0, Rational(-20, 3),  Rational(20, 3)],  # 函数的系数矩阵的负值
                                [Rational(15, 4),     0, Rational(-15, 4)],
                                [Rational(-12, 5), Rational(12, 5),  0]]),  # 各项系数的负值矩阵
                 'type_of_equation': 'type1',  # 方程组的类型
                 'is_general': True}  # 方程组是否为一般形式的
    # 验证分类线性系统的函数对给定的方程组是否返回了预期的结果字典
    assert _classify_linear_system(eqs_10, funcs_2[:-1], t) == answer_10

    # 第三个方程组的定义
    eq11 = (Eq(x1, 3*y(t) - 11*z(t)),  # 第一个方程
            Eq(y1, 7*z(t) - 3*x(t)),  # 第二个方程
            Eq(z1, 11*x(t) - 7*y(t)))  # 第三个方程
    # 对应的预期结果字典
    sol11 = {'no_of_equation': 3,  # 方程组中方程的数量为3个
             'eq': (Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)),  # 第一个方程
                    Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),  # 第二个方程
                    Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))),  # 第三个方程
             'func': [x(t), y(t), z(t)],  # 方程组中的函数列表
             'order': {x(t): 1, y(t): 1, z(t): 1},  # 每个函数的阶数
             'is_linear': True,  # 方程组是否为线性的
             'is_constant': True,  # 方程组是否为常数系数的
             'is_homogeneous': True,  # 方程组是否为齐次的
             'func_coeff': -Matrix([
                            [  0, -3, 11],  # 函数的系数矩阵的负值
                            [  3,  0, -7],
                            [-11,  7,  0]]),  # 各项系数的负值矩阵
             'type_of_equation': 'type1',  # 方程组的类型
             'is_general': True}  # 方程组是否为一般形式的
    # 验证分类线性系统的函数对给定的方程组是否返回了预期的结果字典
    assert _classify_linear_system(eq11, funcs_2[:-1], t) == sol11

    # 第四个方程组的定义
    eq12 = (Eq(Derivative(x(t), t), y(t)),  # 第一个方程
            Eq(Derivative(y(t), t), x(t)))  # 第二个方程
    # 对应的预期结果字典
    sol12 = {'no_of_equation': 2,  # 方程组中方程的数量为2个
             'eq': (Eq(Derivative(x(t), t), y(t)),  # 第一个方程
                    Eq(Derivative(y(t), t), x(t))),  # 第二个方程
             'func': [x(t), y(t)],  # 方程组中的函数列表
             'order': {x(t): 1, y(t): 1},  # 每个函数的阶数
             'is_linear': True,  # 方程组是否为线性的
             'is_constant': True,  # 方程组是否为常数系数的
             'is_homogeneous': True,  # 方程组是否为齐次的
             'func_coeff': -Matrix([
                            [0, -1],  # 函数的系数矩阵的负值
                            [-1, 0]]),  # 各项系数的负值矩阵
             'type_of_equation': 'type1',  # 方程组的类型
             'is_general': True}  # 方程组是否为一般形式的
    # 验证分类线性系统的函数对给定的方程组是否返回了预期的结果字典
    assert _classify_linear_system(eq12, [x(t), y(t)], t) == sol12

    # 第五个方程组的定义
    eq13 = (Eq(Derivative(x(t), t), 21*x(t)),  # 第一个方程
            Eq(Derivative(y(t), t), 17*x(t) + 3*y(t)),  # 第二个方程
            Eq(Derivative(z(t), t), 5*x(t) + 7*y(t) + 9*z(t)))  # 第三个方程
    # 对应的预期结果字典
    sol13 = {'no_of_equation': 3,  # 方程组中方
    # 定义一个包含两个方程的元组，表示一个线性常系数齐次ODE系统
    eq1 = (Eq(diff(x(t), t), x(t) + y(t) + 9), Eq(diff(y(t), t), 2*x(t) + 5*y(t) + 23))
    
    # 定义预期的解字典，描述此ODE系统的特征和性质
    sol1 = {'no_of_equation': 2,  # 方程的数量
            'eq': (Eq(Derivative(x(t), t), x(t) + y(t) + 9),  # 方程组
                   Eq(Derivative(y(t), t), 2*x(t) + 5*y(t) + 23)),
            'func': [x(t), y(t)],  # 函数列表
            'order': {x(t): 1, y(t): 1},  # 每个函数的阶数
            'is_linear': True,  # 是否为线性ODE
            'is_constant': True,  # 系数是否为常数
            'is_homogeneous': False,  # 是否为齐次ODE
            'is_general': True,  # 是否为通解形式
            'func_coeff': -Matrix([[-1, -1], [-2, -5]]),  # 系数矩阵的负值
            'rhs': Matrix([[ 9], [23]]),  # 方程的右侧项
            'type_of_equation': 'type2'}  # 方程类型
    
    # 断言调用 _classify_linear_system 函数返回的结果与预期的解 sol1 相等
    assert _classify_linear_system(eq1, funcs, t) == sol1

    # 非常数系数的齐次ODEs
    eq1 = (Eq(diff(x(t), t), 5*t*x(t) + 2*y(t)), Eq(diff(y(t), t), 2*x(t) + 5*t*y(t)))
    # 定义线性系统的解决方案字典sol1，描述了一个包含两个方程的线性系统
    sol1 = {
        'no_of_equation': 2,  # 方程的数量为2
        'eq': (Eq(Derivative(x(t), t), 5*t*x(t) + 2*y(t)),  # 第一个方程是关于x(t)的微分方程
               Eq(Derivative(y(t), t), 5*t*y(t) + 2*x(t))),  # 第二个方程是关于y(t)的微分方程
        'func': [x(t), y(t)],  # 系统中的函数列表包括x(t)和y(t)
        'order': {x(t): 1, y(t): 1},  # 函数的阶数字典，x(t)和y(t)的阶数均为1
        'is_linear': True,  # 系统是线性的
        'is_constant': False,  # 系统的系数不是常数
        'is_homogeneous': True,  # 系统是齐次的
        'func_coeff': -Matrix([[-5*t, -2], [-2, -5*t]]),  # 系统的函数系数矩阵
        'commutative_antiderivative': Matrix([[5*t**2/2, 2*t], [2*t, 5*t**2/2]]),  # 可交换的反导数矩阵
        'type_of_equation': 'type3',  # 方程类型为type3
        'is_general': True  # 系统是一般的
    }
    # 使用_assert_语句验证_classify_linear_system函数计算的结果与sol1是否匹配
    assert _classify_linear_system(eq1, funcs, t) == sol1

    # 非常数系数非齐次ODEs（常微分方程）
    eq1 = [
        Eq(x1, x(t) + t*y(t) + t),  # 第一个方程
        Eq(y1, t*x(t) + y(t))  # 第二个方程
    ]
    # 定义非常数系数非齐次ODEs的解决方案字典sol1
    sol1 = {
        'no_of_equation': 2,  # 方程的数量为2
        'eq': [
            Eq(Derivative(x(t), t), t*y(t) + t + x(t)),  # 第一个方程的微分方程形式
            Eq(Derivative(y(t), t), t*x(t) + y(t))  # 第二个方程的微分方程形式
        ],
        'func': [x(t), y(t)],  # 系统中的函数列表包括x(t)和y(t)
        'order': {x(t): 1, y(t): 1},  # 函数的阶数字典，x(t)和y(t)的阶数均为1
        'is_linear': True,  # 系统是线性的
        'is_constant': False,  # 系统的系数不是常数
        'is_homogeneous': False,  # 系统是非齐次的
        'is_general': True,  # 系统是一般的
        'func_coeff': -Matrix([[-1, -t], [-t, -1]]),  # 系统的函数系数矩阵
        'commutative_antiderivative': Matrix([[t, t**2/2], [t**2/2, t]]),  # 可交换的反导数矩阵
        'rhs': Matrix([[t], [0]]),  # 右手边的向量
        'type_of_equation': 'type4'  # 方程类型为type4
    }
    # 使用_assert_语句验证_classify_linear_system函数计算的结果与sol1是否匹配
    assert _classify_linear_system(eq1, funcs, t) == sol1

    # 定义非常数系数非齐次ODEs的解决方案字典sol2
    eq2 = [
        Eq(x1, t*x(t) + t*y(t) + t),  # 第一个方程
        Eq(y1, t*x(t) + t*y(t) + cos(t))  # 第二个方程
    ]
    sol2 = {
        'no_of_equation': 2,  # 方程的数量为2
        'eq': [
            Eq(Derivative(x(t), t), t*x(t) + t*y(t) + t),  # 第一个方程的微分方程形式
            Eq(Derivative(y(t), t), t*x(t) + t*y(t) + cos(t))  # 第二个方程的微分方程形式
        ],
        'func': [x(t), y(t)],  # 系统中的函数列表包括x(t)和y(t)
        'order': {x(t): 1, y(t): 1},  # 函数的阶数字典，x(t)和y(t)的阶数均为1
        'is_linear': True,  # 系统是线性的
        'is_homogeneous': False,  # 系统是非齐次的
        'is_general': True,  # 系统是一般的
        'rhs': Matrix([[t], [cos(t)]]),  # 右手边的向量
        'func_coeff': Matrix([[t, t], [t, t]]),  # 系统的函数系数矩阵
        'is_constant': False,  # 系统的系数不是常数
        'type_of_equation': 'type4',  # 方程类型为type4
        'commutative_antiderivative': Matrix([[t**2/2, t**2/2], [t**2/2, t**2/2]])  # 可交换的反导数矩阵
    }
    # 使用_assert_语句验证_classify_linear_system函数计算的结果与sol2是否匹配
    assert _classify_linear_system(eq2, funcs, t) == sol2

    # 定义非常数系数非齐次ODEs的解决方案字典sol3
    eq3 = [
        Eq(x1, t*(x(t) + y(t) + z(t) + 1)),  # 第一个方程
        Eq(y1, t*(x(t) + y(t) + z(t))),  # 第二个方程
        Eq(z1, t*(x(t) + y(t) + z(t)))  # 第三个方程
    ]
    sol3 = {
        'no_of_equation': 3,  # 方程的数量为3
        'eq': [
            Eq(Derivative(x(t), t), t*(x(t) + y(t) + z(t) + 1)),  # 第一个方程的微分方程形式
            Eq(Derivative(y(t), t), t*(x(t) + y(t) + z(t))),  # 第二个方程的微分方程形式
            Eq(Derivative(z(t), t), t*(x(t) + y(t) + z(t)))  # 第三个方程的微分方程形式
        ],
        'func': [x(t), y(t), z(t)],  # 系统中的函数列表包括x(t)，y(t)和z(t)
        'order': {x(t): 1, y(t): 1, z(t): 1},  # 函数的阶数字典，x(t)，y(t)和z(t)的阶数均为1
        'is_linear': True,  # 系统是线性的
        'is_constant': False,  # 系统的系数不是常数
        'is_homogeneous': False,  # 系统是非齐次的
        'is_general': True,  # 系统是一般的
        'func_coeff': -Matrix([[-t, -t, -t], [-t, -t, -t], [-t, -t, -t]]),  # 系统的函数系数矩阵
        'commutative_antiderivative': Matrix([[t**2/2, t**2/2, t**2/2], [t**2/2, t**2/2, t**2/2], [t**2/2, t**2/2, t**2/2]]),  # 可交换的反导数
    # 定义一个包含两个方程的元组，表示二阶常系数线性微分方程组
    eq2 = (
        # 第一个方程，左边是二阶导数关于 t 的表达式，右边是 x(t) 和 y(t) 的线性组合
        Eq((4*t**2 + 7*t + 1)**2 * Derivative(x(t), (t, 2)), 5*x(t) + 35*y(t)),
        # 第二个方程，左边是二阶导数关于 t 的表达式，右边是 x(t) 和 y(t) 的线性组合
        Eq((4*t**2 + 7*t + 1)**2 * Derivative(y(t), (t, 2)), x(t) + 9*y(t))
    )
    
    # 定义预期的解 sol2，包含了方程组的各种属性和特征
    sol2 = {
        'no_of_equation': 2,  # 方程的数量为 2
        'eq': eq2,  # 方程组
        'func': [x(t), y(t)],  # 函数列表
        'order': {x(t): 2, y(t): 2},  # 每个函数的阶数
        'is_linear': True,  # 方程组是否为线性的
        'is_homogeneous': False,  # 方程组是否是齐次的
        'is_general': True,  # 方程组是否为一般形式
        'rhs': Matrix([[0], [0]]),  # 右侧的常数项
        'type_of_equation': 'type4'  # 方程组的类型
    }
    # 使用 assert 语句检查 _classify_linear_system 函数对 eq2 的分类结果是否与 sol2 一致
    assert _classify_linear_system(eq2, funcs, t) == sol2
    # 定义函数sol2，包含一个线性方程组的解和相关属性
    sol2 = {
        'no_of_equation': 2,  # 方程组的数量为2
        'eq': (Eq((4*t**2 + 7*t + 1)**2*Derivative(x(t), (t, 2)), 5*x(t) + 35*y(t)),  # 第一个方程
               Eq((4*t**2 + 7*t + 1)**2*Derivative(y(t), (t, 2)), x(t) + 9*y(t))),  # 第二个方程
        'func': [x(t), y(t)],  # 方程组涉及的函数列表
        'order': {x(t): 2, y(t): 2},  # 每个函数的阶数
        'is_linear': True,  # 方程组是否为线性的
        'is_homogeneous': True,  # 方程组是否是齐次的
        'is_general': True,  # 方程组是否是一般形式的
        'type_of_equation': 'type2',  # 方程组的类型
        'A0': Matrix([ [Rational(53, 4),   35], [   1, Rational(69, 4)]]),  # 系数矩阵 A0
        'g(t)': sqrt(4*t**2 + 7*t + 1),  # 函数 g(t)
        'tau': sqrt(33)*log(t - sqrt(33)/8 + Rational(7, 8))/33 - sqrt(33)*log(t + sqrt(33)/8 + Rational(7, 8))/33,  # 参数 tau
        'is_transformed': True,  # 方程组是否已经进行了变换
        't_': t_,  # 时间变量 t_
        'is_second_order': True,  # 方程是否为二阶的
        'is_higher_order': True  # 方程是否为高阶的
    }
    # 断言对应的函数调用返回期望的结果 sol2
    assert _classify_linear_system(eq2, funcs, t) == sol2

    # 定义函数eq3，包含一个线性方程组
    eq3 = (
        (t*Derivative(x(t), t) - x(t))*log(t) + (t*Derivative(y(t), t) - y(t))*exp(t) + Derivative(x(t), (t, 2)),
        t**2*(t*Derivative(x(t), t) - x(t)) + t*(t*Derivative(y(t), t) - y(t)) + Derivative(y(t), (t, 2))
    )
    # 定义函数sol3，包含eq3方程组的解和相关属性
    sol3 = {
        'no_of_equation': 2,  # 方程组的数量为2
        'eq': eq3,  # 方程组的表达式
        'func': [x(t), y(t)],  # 方程组涉及的函数列表
        'order': {x(t): 2, y(t): 2},  # 每个函数的阶数
        'is_linear': True,  # 方程组是否为线性的
        'is_homogeneous': True,  # 方程组是否是齐次的
        'is_general': True,  # 方程组是否是一般形式的
        'type_of_equation': 'type1',  # 方程组的类型
        'A1': Matrix([ [-t*log(t), -t*exp(t)], [    -t**3,     -t**2]]),  # 系数矩阵 A1
        'is_second_order': True,  # 方程是否为二阶的
        'is_higher_order': True  # 方程是否为高阶的
    }
    # 断言对应的函数调用返回期望的结果 sol3
    assert _classify_linear_system(eq3, funcs, t) == sol3

    # 定义函数eq4，包含一个线性方程组
    eq4 = (
        Eq(x2, k*x(t) - l*y1),
        Eq(y2, l*x1 + k*y(t))
    )
    # 定义函数sol4，包含eq4方程组的解和相关属性
    sol4 = {
        'no_of_equation': 2,  # 方程组的数量为2
        'eq': (
            Eq(Derivative(x(t), (t, 2)), k*x(t) - l*Derivative(y(t), t)),
            Eq(Derivative(y(t), (t, 2)), k*y(t) + l*Derivative(x(t), t))
        ),
        'func': [x(t), y(t)],  # 方程组涉及的函数列表
        'order': {x(t): 2, y(t): 2},  # 每个函数的阶数
        'is_linear': True,  # 方程组是否为线性的
        'is_homogeneous': True,  # 方程组是否是齐次的
        'is_general': True,  # 方程组是否是一般形式的
        'type_of_equation': 'type0',  # 方程组的类型
        'is_second_order': True,  # 方程是否为二阶的
        'is_higher_order': True  # 方程是否为高阶的
    }
    # 断言对应的函数调用返回期望的结果 sol4
    assert _classify_linear_system(eq4, funcs, t) == sol4

    # 多重匹配情况

    # 创建符号函数 f 和 g
    f, g = symbols("f g", cls=Function)
    y, t_ = symbols("y t_")
    funcs = [f(t), g(t)]

    # 定义方程组eq1，包含两个方程
    eq1 = [
        Eq(Derivative(f(t), t)**2 - 2*Derivative(f(t), t) + 1, 4),
        Eq(-y*f(t) + Derivative(g(t), t), 0)
    ]
    # 定义函数sol1，包含eq1方程组的解和相关属性
    sol1 = {
        'is_implicit': True,  # 方程组是否是隐式的
        'canon_eqs': [
            [Eq(Derivative(f(t), t), -1), Eq(Derivative(g(t), t), y*f(t))],
            [Eq(Derivative(f(t), t), 3), Eq(Derivative(g(t), t), y*f(t))]
        ]  # 方程组的标准化形式列表
    }
    # 断言对应的函数调用返回期望的结果 sol1
    assert _classify_linear_system(eq1, funcs, t) == sol1

    # 断言对一个不合理的函数调用，引发 ValueError 异常
    raises(ValueError, lambda: _classify_linear_system(eq1, funcs[:1], t))

    # 定义方程组eq2，包含两个方程
    eq2 = [
        Eq(Derivative(f(t), t), (2*f(t) + g(t) + 1)/t),
        Eq(Derivative(g(t), t), (f(t) + 2*g(t))/t)
    ]
    # 定义包含两个微分方程的线性系统的解析解sol2字典，包括方程、函数、系数等信息
    sol2 = {'no_of_equation': 2, 'eq': [Eq(Derivative(f(t), t), (2*f(t) + g(t) + 1)/t), Eq(Derivative(g(t), t),
            (f(t) + 2*g(t))/t)], 'func': [f(t), g(t)], 'order': {f(t): 1, g(t): 1}, 'is_linear': True,
            'is_homogeneous': False, 'is_general': True, 'rhs': Matrix([ [1], [0]]), 'func_coeff': Matrix([ [2,
            1], [1, 2]]), 'is_constant': False, 'type_of_equation': 'type6', 't_': t_, 'tau': log(t),
            'commutative_antiderivative': Matrix([ [2*log(t),   log(t)], [  log(t), 2*log(t)]])}
    # 使用_assert_语句验证函数_classify_linear_system是否返回了预期的解析解sol2
    assert _classify_linear_system(eq2, funcs, t) == sol2

    # 定义包含两个微分方程的线性系统的解析解sol3字典，这是一个齐次方程组，包括方程、函数、系数等信息
    eq3 = [Eq(Derivative(f(t), t), (2*f(t) + g(t))/t), Eq(Derivative(g(t), t), (f(t) + 2*g(t))/t)]
    sol3 = {'no_of_equation': 2, 'eq': [Eq(Derivative(f(t), t), (2*f(t) + g(t))/t), Eq(Derivative(g(t), t),
            (f(t) + 2*g(t))/t)], 'func': [f(t), g(t)], 'order': {f(t): 1, g(t): 1}, 'is_linear': True,
            'is_homogeneous': True, 'is_general': True, 'func_coeff': Matrix([ [2, 1], [1, 2]]), 'is_constant':
            False, 'type_of_equation': 'type5', 't_': t_, 'rhs': Matrix([ [0], [0]]), 'tau': log(t),
            'commutative_antiderivative': Matrix([ [2*log(t),   log(t)], [  log(t), 2*log(t)]])}
    # 使用_assert_语句验证函数_classify_linear_system是否返回了预期的解析解sol3
    assert _classify_linear_system(eq3, funcs, t) == sol3
def test_matrix_exp():
    # 导入所需的矩阵运算库和符号表示库
    from sympy.matrices.dense import Matrix, eye, zeros
    from sympy.solvers.ode.systems import matrix_exp
    # 定义符号变量 t
    t = Symbol('t')

    # 第一个测试：对于零矩阵，指数函数为单位矩阵
    for n in range(1, 6+1):
        assert matrix_exp(zeros(n), t) == eye(n)

    # 第二个测试：对于单位矩阵，指数函数为 exp(t) * 单位矩阵
    for n in range(1, 6+1):
        A = eye(n)
        expAt = exp(t) * eye(n)
        assert matrix_exp(A, t) == expAt

    # 第三个测试：对于特定对角线上元素递增的矩阵 A，计算其指数函数 exp(At)
    for n in range(1, 6+1):
        A = Matrix(n, n, lambda i,j: i+1 if i==j else 0)
        expAt = Matrix(n, n, lambda i,j: exp((i+1)*t) if i==j else 0)
        assert matrix_exp(A, t) == expAt

    # 特定矩阵 A = [[0, 1], [-1, 0]] 的指数函数
    A = Matrix([[0, 1], [-1, 0]])
    expAt = Matrix([[cos(t), sin(t)], [-sin(t), cos(t)]])
    assert matrix_exp(A, t) == expAt

    # 特定矩阵 A = [[2, -5], [2, -4]] 的指数函数
    A = Matrix([[2, -5], [2, -4]])
    expAt = Matrix([
            [3*exp(-t)*sin(t) + exp(-t)*cos(t), -5*exp(-t)*sin(t)],
            [2*exp(-t)*sin(t), -3*exp(-t)*sin(t) + exp(-t)*cos(t)]
            ])
    assert matrix_exp(A, t) == expAt

    # 特定矩阵 A = [[21, 17, 6], [-5, -1, -6], [4, 4, 16]] 的指数函数
    A = Matrix([[21, 17, 6], [-5, -1, -6], [4, 4, 16]])
    expAt = Matrix([
        [2*t*exp(16*t) + 5*exp(16*t)/4 - exp(4*t)/4, 2*t*exp(16*t) + 5*exp(16*t)/4 - 5*exp(4*t)/4,  exp(16*t)/2 - exp(4*t)/2],
        [ -2*t*exp(16*t) - exp(16*t)/4 + exp(4*t)/4,  -2*t*exp(16*t) - exp(16*t)/4 + 5*exp(4*t)/4, -exp(16*t)/2 + exp(4*t)/2],
        [                             4*t*exp(16*t),                                4*t*exp(16*t),                 exp(16*t)]
        ])
    assert matrix_exp(A, t) == expAt

    # 特定矩阵 A = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, -S(1)/8], [0, 0, S(1)/2, S(1)/2]] 的指数函数
    A = Matrix([[1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, -S(1)/8],
                [0, 0, S(1)/2, S(1)/2]])
    expAt = Matrix([
        [exp(t), t*exp(t), 4*t*exp(3*t/4) + 8*t*exp(t) + 48*exp(3*t/4) - 48*exp(t),
                            -2*t*exp(3*t/4) - 2*t*exp(t) - 16*exp(3*t/4) + 16*exp(t)],
        [0, exp(t), -t*exp(3*t/4) - 8*exp(3*t/4) + 8*exp(t), t*exp(3*t/4)/2 + 2*exp(3*t/4) - 2*exp(t)],
        [0, 0, t*exp(3*t/4)/4 + exp(3*t/4), -t*exp(3*t/4)/8],
        [0, 0, t*exp(3*t/4)/2, -t*exp(3*t/4)/4 + exp(3*t/4)]
        ])
    assert matrix_exp(A, t) == expAt

    # 特定矩阵 A = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]] 的指数函数
    A = Matrix([
    [ 0, 1,  0, 0],
    [-1, 0,  0, 0],
    [ 0, 0,  0, 1],
    [ 0, 0, -1, 0]])

    expAt = Matrix([
    [ cos(t), sin(t),         0,        0],
    [-sin(t), cos(t),         0,        0],
    [      0,      0,    cos(t),   sin(t)],
    [      0,      0,   -sin(t),   cos(t)]])
    assert matrix_exp(A, t) == expAt

    # 最后一个测试：特定矩阵 A = [[0, 1, 1, 0], [-1, 0, 0, 1], [0, 0, 0, 1], [0, 0, -1, 0]] 的指数函数
    A = Matrix([
    [ 0, 1,  1, 0],
    [-1, 0,  0, 1],
    [ 0, 0,  0, 1],
    [ 0, 0, -1, 0]])

    expAt = Matrix([
    [ cos(t), sin(t),  t*cos(t), t*sin(t)],
    [-sin(t), cos(t), -t*sin(t), t*cos(t)],
    A = Matrix([[0, I], [I, 0]])
    expAt = Matrix([
    [exp(I*t)/2 + exp(-I*t)/2, exp(I*t)/2 - exp(-I*t)/2],
    [exp(I*t)/2 - exp(-I*t)/2, exp(I*t)/2 + exp(-I*t)/2]])

# 定义一个2x2的复数矩阵 A
# 计算矩阵 A 的指数 e^(At)，其中 t 是一个符号变量
# 断言计算结果与预期的指数矩阵 expAt 相等


    # Testing Errors
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 7, 7]])
    M1 = Matrix([[t, 1], [1, 1]])

    raises(ValueError, lambda: matrix_exp(M[:, :2], t))
    raises(ValueError, lambda: matrix_exp(M[:2, :], t))
    raises(ValueError, lambda: matrix_exp(M1, t))
    raises(ValueError, lambda: matrix_exp(M1[:1, :1], t))

# 测试矩阵指数函数的异常情况
# 创建两个测试用的矩阵 M 和 M1
# 断言对于给定的切片或不合法的矩阵输入，matrix_exp 函数会引发 ValueError 异常
def test_canonical_odes():
    # 定义符号函数 f, g, h
    f, g, h = symbols('f g h', cls=Function)
    # 定义符号变量 x
    x = symbols('x')
    # 定义函数列表
    funcs = [f(x), g(x), h(x)]

    # 第一组常规微分方程
    eqs1 = [Eq(f(x).diff(x, x), f(x) + 2*g(x)), Eq(g(x) + 1, g(x).diff(x) + f(x))]
    # 对应的解
    sol1 = [[Eq(Derivative(f(x), (x, 2)), f(x) + 2*g(x)), Eq(Derivative(g(x), x), -f(x) + g(x) + 1)]]
    # 断言函数返回的结果与预期解相等
    assert canonical_odes(eqs1, funcs[:2], x) == sol1

    # 第二组常规微分方程
    eqs2 = [Eq(f(x).diff(x), h(x).diff(x) + f(x)), Eq(g(x).diff(x)**2, f(x) + h(x)), Eq(h(x).diff(x), f(x))]
    # 对应的解
    sol2 = [[Eq(Derivative(f(x), x), 2*f(x)), Eq(Derivative(g(x), x), -sqrt(f(x) + h(x))), Eq(Derivative(h(x), x), f(x))],
            [Eq(Derivative(f(x), x), 2*f(x)), Eq(Derivative(g(x), x), sqrt(f(x) + h(x))), Eq(Derivative(h(x), x), f(x))]]
    # 断言函数返回的结果与预期解相等
    assert canonical_odes(eqs2, funcs, x) == sol2


def test_sysode_linear_neq_order1_type1():
    # 定义符号函数 f, g, x, y, h
    f, g, x, y, h = symbols('f g x y h', cls=Function)
    # 定义符号变量 a, b, c, t
    a, b, c, t = symbols('a b c t')

    # 第一组线性非齐次一阶系统微分方程
    eqs1 = [Eq(Derivative(x(t), t), x(t)),
            Eq(Derivative(y(t), t), y(t))]
    # 对应的解
    sol1 = [Eq(x(t), C1*exp(t)),
            Eq(y(t), C2*exp(t))]
    # 断言解的正确性及其符合微分方程
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    # 第二组线性非齐次一阶系统微分方程
    eqs2 = [Eq(Derivative(x(t), t), 2*x(t)),
            Eq(Derivative(y(t), t), 3*y(t))]
    # 对应的解
    sol2 = [Eq(x(t), C1*exp(2*t)),
            Eq(y(t), C2*exp(3*t))]
    # 断言解的正确性及其符合微分方程
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    # 第三组线性非齐次一阶系统微分方程
    eqs3 = [Eq(Derivative(x(t), t), a*x(t)),
            Eq(Derivative(y(t), t), a*y(t))]
    # 对应的解
    sol3 = [Eq(x(t), C1*exp(a*t)),
            Eq(y(t), C2*exp(a*t))]
    # 断言解的正确性及其符合微分方程
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])

    # 回归测试用例，针对问题 #15474
    # https://github.com/sympy/sympy/issues/15474
    # 第四组线性非齐次一阶系统微分方程
    eqs4 = [Eq(Derivative(x(t), t), a*x(t)),
            Eq(Derivative(y(t), t), b*y(t))]
    # 对应的解
    sol4 = [Eq(x(t), C1*exp(a*t)),
            Eq(y(t), C2*exp(b*t))]
    # 断言解的正确性及其符合微分方程
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])

    # 第五组线性非齐次一阶系统微分方程
    eqs5 = [Eq(Derivative(x(t), t), -y(t)),
            Eq(Derivative(y(t), t), x(t))]
    # 对应的解
    sol5 = [Eq(x(t), -C1*sin(t) - C2*cos(t)),
            Eq(y(t), C1*cos(t) - C2*sin(t))]
    # 断言解的正确性及其符合微分方程
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])

    # 第六组线性非齐次一阶系统微分方程
    eqs6 = [Eq(Derivative(x(t), t), -2*y(t)),
            Eq(Derivative(y(t), t), 2*x(t))]
    # 对应的解
    sol6 = [Eq(x(t), -C1*sin(2*t) - C2*cos(2*t)),
            Eq(y(t), C1*cos(2*t) - C2*sin(2*t))]
    # 断言解的正确性及其符合微分方程
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0])

    # 第七组线性非齐次一阶系统微分方程
    eqs7 = [Eq(Derivative(x(t), t), I*y(t)),
            Eq(Derivative(y(t), t), I*x(t))]
    # 对应的解
    sol7 = [Eq(x(t), -C1*exp(-I*t) + C2*exp(I*t)),
            Eq(y(t), C1*exp(-I*t) + C2*exp(I*t))]
    # 断言解的正确性及其符合微分方程
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])

    # 第八组线性非齐次一阶系统微分方程
    eqs8 = [Eq(Derivative(x(t), t), -a*y(t)),
            Eq(Derivative(y(t), t), a*x(t))]
    # 定义微分方程组 eqs9
    eqs9 = [Eq(Derivative(x(t), t), x(t) + y(t)),
            Eq(Derivative(y(t), t), x(t) - y(t))]
    # 定义微分方程组 eqs9 的解析解 sol9
    sol9 = [Eq(x(t), C1*(1 - sqrt(2))*exp(-sqrt(2)*t) + C2*(1 + sqrt(2))*exp(sqrt(2)*t)),
            Eq(y(t), C1*exp(-sqrt(2)*t) + C2*exp(sqrt(2)*t))]
    # 断言 dsolve 函数对 eqs9 的求解结果与预期解 sol9 相等
    assert dsolve(eqs9) == sol9
    # 断言 checksysodesol 函数确认 eqs9 的解 sol9 是其解的正确性和常数的结果
    assert checksysodesol(eqs9, sol9) == (True, [0, 0])

    # 定义微分方程组 eqs10
    eqs10 = [Eq(Derivative(x(t), t), x(t) + y(t)),
             Eq(Derivative(y(t), t), x(t) + y(t))]
    # 定义微分方程组 eqs10 的解析解 sol10
    sol10 = [Eq(x(t), -C1 + C2*exp(2*t)),
             Eq(y(t), C1 + C2*exp(2*t))]
    # 断言 dsolve 函数对 eqs10 的求解结果与预期解 sol10 相等
    assert dsolve(eqs10) == sol10
    # 断言 checksysodesol 函数确认 eqs10 的解 sol10 是其解的正确性和常数的结果
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])

    # 定义微分方程组 eqs11
    eqs11 = [Eq(Derivative(x(t), t), 2*x(t) + y(t)),
             Eq(Derivative(y(t), t), -x(t) + 2*y(t))]
    # 定义微分方程组 eqs11 的解析解 sol11
    sol11 = [Eq(x(t), C1*exp(2*t)*sin(t) + C2*exp(2*t)*cos(t)),
             Eq(y(t), C1*exp(2*t)*cos(t) - C2*exp(2*t)*sin(t))]
    # 断言 dsolve 函数对 eqs11 的求解结果与预期解 sol11 相等
    assert dsolve(eqs11) == sol11
    # 断言 checksysodesol 函数确认 eqs11 的解 sol11 是其解的正确性和常数的结果
    assert checksysodesol(eqs11, sol11) == (True, [0, 0])

    # 定义微分方程组 eqs12
    eqs12 = [Eq(Derivative(x(t), t), x(t) + 2*y(t)),
             Eq(Derivative(y(t), t), 2*x(t) + y(t))]
    # 定义微分方程组 eqs12 的解析解 sol12
    sol12 = [Eq(x(t), -C1*exp(-t) + C2*exp(3*t)),
             Eq(y(t), C1*exp(-t) + C2*exp(3*t))]
    # 断言 dsolve 函数对 eqs12 的求解结果与预期解 sol12 相等
    assert dsolve(eqs12) == sol12
    # 断言 checksysodesol 函数确认 eqs12 的解 sol12 是其解的正确性和常数的结果
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])

    # 定义微分方程组 eqs13
    eqs13 = [Eq(Derivative(x(t), t), 4*x(t) + y(t)),
             Eq(Derivative(y(t), t), -x(t) + 2*y(t))]
    # 定义微分方程组 eqs13 的解析解 sol13
    sol13 = [Eq(x(t), C2*t*exp(3*t) + (C1 + C2)*exp(3*t)),
             Eq(y(t), -C1*exp(3*t) - C2*t*exp(3*t))]
    # 断言 dsolve 函数对 eqs13 的求解结果与预期解 sol13 相等
    assert dsolve(eqs13) == sol13
    # 断言 checksysodesol 函数确认 eqs13 的解 sol13 是其解的正确性和常数的结果
    assert checksysodesol(eqs13, sol13) == (True, [0, 0])

    # 定义微分方程组 eqs14
    eqs14 = [Eq(Derivative(x(t), t), a*y(t)),
             Eq(Derivative(y(t), t), a*x(t))]
    # 定义微分方程组 eqs14 的解析解 sol14
    sol14 = [Eq(x(t), -C1*exp(-a*t) + C2*exp(a*t)),
             Eq(y(t), C1*exp(-a*t) + C2*exp(a*t))]
    # 断言 dsolve 函数对 eqs14 的求解结果与预期解 sol14 相等
    assert dsolve(eqs14) == sol14
    # 断言 checksysodesol 函数确认 eqs14 的解 sol14 是其解的正确性和常数的结果
    assert checksysodesol(eqs14, sol14) == (True, [0, 0])

    # 定义微分方程组 eqs15
    eqs15 = [Eq(Derivative(x(t), t), a*y(t)),
             Eq(Derivative(y(t), t), b*x(t))]
    # 定义微分方程组 eqs15 的解析解 sol15
    sol15 = [Eq(x(t), -C1*a*exp(-t*sqrt(a*b))/sqrt(a*b) + C2*a*exp(t*sqrt(a*b))/sqrt(a*b)),
             Eq(y(t), C1*exp(-t*sqrt(a*b)) + C2*exp(t*sqrt(a*b)))]
    # 断言 dsolve 函数对 eqs15 的求解结果与预期解 sol15 相等
    assert dsolve(eqs15) == sol15
    # 断言 checksysodesol 函数确认 eqs15 的解 sol15 是其解的正确性和常数的结果
    assert checksysodesol(eqs15, sol15) == (True, [0, 0])

    # 定义微分方程组 eqs16
    eqs16 = [Eq(Derivative(x(t), t), a*x(t) + b*y(t)),
             Eq(Derivative(y(t), t), c*x(t))]
    # 定义微分方程组 eqs16 的解析解 sol16
    sol16 = [Eq(x(t), -2*C1*b*exp(t*(a + sqrt(a**2 + 4*b*c))/2)/(a - sqrt(a**2 + 4*b*c)) - 2*C2*b*exp(t*(a -
              sqrt(a**2 + 4*b*c))/2)/(a + sqrt(a**2 + 4*b*c))),
             Eq(y(t), C1*exp(t*(a + sqrt(a**2 + 4*b*c))/2) + C2*exp(t*(a - sqrt(a**2 + 4*b*c))/2))]
    # 断言 dsolve 函数对 eqs16 的求解结果与预期解 sol16 相等
    assert dsolve(eqs16) == sol16
    # 断言 checksysodesol 函数确认 eqs16 的解 sol16 是其解的正确性和常数的结果
    assert checksysodesol(eqs16, sol16) == (True, [0, 0])

    # 用于问题 #18562 的回归测试案例
    # https://github.com/sympy/sympy/issues/18562
    # 定义一个包含两个微分方程的列表，描述了动态系统的演变
    eqs17 = [Eq(Derivative(x(t), t), a*y(t) + x(t)),
             Eq(Derivative(y(t), t), a*x(t) - y(t))]
    # 解微分方程系统eqs17，并将解存储在sol17列表中
    sol17 = [Eq(x(t), C1*a*exp(t*sqrt(a**2 + 1))/(sqrt(a**2 + 1) - 1) - C2*a*exp(-t*sqrt(a**2 + 1))/(sqrt(a**2 + 1) + 1)),
             Eq(y(t), C1*exp(t*sqrt(a**2 + 1)) + C2*exp(-t*sqrt(a**2 + 1)))]
    # 断言求解的微分方程系统eqs17与预期解sol17相等
    assert dsolve(eqs17) == sol17
    # 断言解sol17是否满足微分方程系统eqs17的条件
    assert checksysodesol(eqs17, sol17) == (True, [0, 0])
    
    # 定义一个包含两个微分方程的列表，这两个方程都是零阶导数
    eqs18 = [Eq(Derivative(x(t), t), 0),
             Eq(Derivative(y(t), t), 0)]
    # 解微分方程系统eqs18，并将解存储在sol18列表中
    sol18 = [Eq(x(t), C1),
             Eq(y(t), C2)]
    # 断言求解的微分方程系统eqs18与预期解sol18相等
    assert dsolve(eqs18) == sol18
    # 断言解sol18是否满足微分方程系统eqs18的条件
    assert checksysodesol(eqs18, sol18) == (True, [0, 0])
    
    # 定义一个包含两个微分方程的列表，描述线性常系数微分方程组
    eqs19 = [Eq(Derivative(x(t), t), 2*x(t) - y(t)),
             Eq(Derivative(y(t), t), x(t))]
    # 解微分方程系统eqs19，并将解存储在sol19列表中
    sol19 = [Eq(x(t), C2*t*exp(t) + (C1 + C2)*exp(t)),
             Eq(y(t), C1*exp(t) + C2*t*exp(t))]
    # 断言求解的微分方程系统eqs19与预期解sol19相等
    assert dsolve(eqs19) == sol19
    # 断言解sol19是否满足微分方程系统eqs19的条件
    assert checksysodesol(eqs19, sol19) == (True, [0, 0])
    
    # 定义一个包含两个微分方程的列表，描述了非常数系数的线性微分方程组
    eqs20 = [Eq(Derivative(x(t), t), x(t)),
             Eq(Derivative(y(t), t), x(t) + y(t))]
    # 解微分方程系统eqs20，并将解存储在sol20列表中
    sol20 = [Eq(x(t), C1*exp(t)),
             Eq(y(t), C1*t*exp(t) + C2*exp(t))]
    # 断言求解的微分方程系统eqs20与预期解sol20相等
    assert dsolve(eqs20) == sol20
    # 断言解sol20是否满足微分方程系统eqs20的条件
    assert checksysodesol(eqs20, sol20) == (True, [0, 0])
    
    # 定义一个包含两个微分方程的列表，描述了指数增长和指数增长与线性组合的情况
    eqs21 = [Eq(Derivative(x(t), t), 3*x(t)),
             Eq(Derivative(y(t), t), x(t) + y(t))]
    # 解微分方程系统eqs21，并将解存储在sol21列表中
    sol21 = [Eq(x(t), 2*C1*exp(3*t)),
             Eq(y(t), C1*exp(3*t) + C2*exp(t))]
    # 断言求解的微分方程系统eqs21与预期解sol21相等
    assert dsolve(eqs21) == sol21
    # 断言解sol21是否满足微分方程系统eqs21的条件
    assert checksysodesol(eqs21, sol21) == (True, [0, 0])
    
    # 定义一个包含两个微分方程的列表，描述了指数增长和指数增长的情况
    eqs22 = [Eq(Derivative(x(t), t), 3*x(t)),
             Eq(Derivative(y(t), t), y(t))]
    # 解微分方程系统eqs22，并将解存储在sol22列表中
    sol22 = [Eq(x(t), C1*exp(3*t)),
             Eq(y(t), C2*exp(t))]
    # 断言求解的微分方程系统eqs22与预期解sol22相等
    assert dsolve(eqs22) == sol22
    # 断言解sol22是否满足微分方程系统eqs22的条件
    assert checksysodesol(eqs22, sol22) == (True, [0, 0])
# 标记该测试函数为一个慢速测试（假设在测试套件中可能需要更长时间执行）
@slow
def test_sysode_linear_neq_order1_type1_slow():

    # 定义符号变量和函数
    t = Symbol('t')
    Z0 = Function('Z0')
    Z1 = Function('Z1')
    Z2 = Function('Z2')
    Z3 = Function('Z3')

    # 定义微分方程中的参数
    k01, k10, k20, k21, k23, k30 = symbols('k01 k10 k20 k21 k23 k30')

    # 第一组线性非齐次一阶微分方程
    eqs1 = [Eq(Derivative(Z0(t), t), -k01*Z0(t) + k10*Z1(t) + k20*Z2(t) + k30*Z3(t)),
            Eq(Derivative(Z1(t), t), k01*Z0(t) - k10*Z1(t) + k21*Z2(t)),
            Eq(Derivative(Z2(t), t), (-k20 - k21 - k23)*Z2(t)),
            Eq(Derivative(Z3(t), t), k23*Z2(t) - k30*Z3(t))]
    
    # 求解微分方程
    sol1 = [Eq(Z0(t), C1*k10/k01 - C2*(k10 - k30)*exp(-k30*t)/(k01 + k10 - k30) - C3*(k10*(k20 + k21 - k30) -
             k20**2 - k20*(k21 + k23 - k30) + k23*k30)*exp(-t*(k20 + k21 + k23))/(k23*(-k01 - k10 + k20 + k21 +
             k23)) - C4*exp(-t*(k01 + k10))),
            Eq(Z1(t), C1 - C2*k01*exp(-k30*t)/(k01 + k10 - k30) + C3*(-k01*(k20 + k21 - k30) + k20*k21 + k21**2
             + k21*(k23 - k30))*exp(-t*(k20 + k21 + k23))/(k23*(-k01 - k10 + k20 + k21 + k23)) + C4*exp(-t*(k01 +
             k10))),
            Eq(Z2(t), -C3*(k20 + k21 + k23 - k30)*exp(-t*(k20 + k21 + k23))/k23),
            Eq(Z3(t), C2*exp(-k30*t) + C3*exp(-t*(k20 + k21 + k23)))]

    # 验证微分方程的解
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0, 0, 0])

    # 定义更多的符号变量和函数
    x, y, z, u, v, w = symbols('x y z u v w', cls=Function)
    k2, k3 = symbols('k2 k3')
    a_b, a_c = symbols('a_b a_c', real=True)

    # 第二组线性非齐次一阶微分方程
    eqs2 = [Eq(Derivative(z(t), t), k2*y(t)),
            Eq(Derivative(x(t), t), k3*y(t)),
            Eq(Derivative(y(t), t), (-k2 - k3)*y(t))]
    
    # 求解微分方程
    sol2 = [Eq(z(t), C1 - C2*k2*exp(-t*(k2 + k3))/(k2 + k3)),
            Eq(x(t), -C2*k3*exp(-t*(k2 + k3))/(k2 + k3) + C3),
            Eq(y(t), C2*exp(-t*(k2 + k3)))]

    # 验证微分方程的解
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0, 0])

    # 第三组线性非齐次一阶微分方程
    eqs3 = [4*u(t) - v(t) - 2*w(t) + Derivative(u(t), t),
            2*u(t) + v(t) - 2*w(t) + Derivative(v(t), t),
            5*u(t) + v(t) - 3*w(t) + Derivative(w(t), t)]
    
    # 求解微分方程
    sol3 = [Eq(u(t), C3*exp(-2*t) + (C1/2 + sqrt(3)*C2/6)*cos(sqrt(3)*t) + sin(sqrt(3)*t)*(sqrt(3)*C1/6 +
             C2*Rational(-1, 2))),
            Eq(v(t), (C1/2 + sqrt(3)*C2/6)*cos(sqrt(3)*t) + sin(sqrt(3)*t)*(sqrt(3)*C1/6 + C2*Rational(-1, 2))),
            Eq(w(t), C1*cos(sqrt(3)*t) - C2*sin(sqrt(3)*t) + C3*exp(-2*t))]

    # 验证微分方程的解
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0, 0])

    # 第四组线性非齐次一阶微分方程
    eqs4 = [Eq(Derivative(x(t), t), w(t)*Rational(-2, 9) + 2*x(t) + y(t) + z(t)*Rational(-8, 9)),
            Eq(Derivative(y(t), t), w(t)*Rational(4, 9) + 2*y(t) + z(t)*Rational(16, 9)),
            Eq(Derivative(z(t), t), w(t)*Rational(-2, 9) + z(t)*Rational(37, 9)),
            Eq(Derivative(w(t), t), w(t)*Rational(44, 9) + z(t)*Rational(-4, 9))]
    
    # 求解微分方程
    sol4 = [Eq(x(t), C1*exp(2*t) + C2*t*exp(2*t)),
            Eq(y(t), C2*exp(2*t) + 2*C3*exp(4*t)),
            Eq(z(t), 2*C3*exp(4*t) + C4*exp(5*t)*Rational(-1, 4)),
            Eq(w(t), C3*exp(4*t) + C4*exp(5*t))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0, 0])

    # 回归测试用例，针对问题 #15574
    # https://github.com/sympy/sympy/issues/15574
    eq5 = [Eq(x(t).diff(t), x(t)), Eq(y(t).diff(t), y(t)), Eq(z(t).diff(t), z(t)), Eq(w(t).diff(t), w(t))]
    sol5 = [Eq(x(t), C1*exp(t)), Eq(y(t), C2*exp(t)), Eq(z(t), C3*exp(t)), Eq(w(t), C4*exp(t))]
    assert dsolve(eq5) == sol5
    assert checksysodesol(eq5, sol5) == (True, [0, 0, 0, 0])

    eqs6 = [Eq(Derivative(x(t), t), x(t) + y(t)),
            Eq(Derivative(y(t), t), y(t) + z(t)),
            Eq(Derivative(z(t), t), w(t)*Rational(-1, 8) + z(t)),
            Eq(Derivative(w(t), t), w(t)/2 + z(t)/2)]
    sol6 = [Eq(x(t), C1*exp(t) + C2*t*exp(t) + 4*C4*t*exp(t*Rational(3, 4)) + (4*C3 + 48*C4)*exp(t*Rational(3,
             4))),
            Eq(y(t), C2*exp(t) - C4*t*exp(t*Rational(3, 4)) - (C3 + 8*C4)*exp(t*Rational(3, 4))),
            Eq(z(t), C4*t*exp(t*Rational(3, 4))/4 + (C3/4 + C4)*exp(t*Rational(3, 4))),
            Eq(w(t), C3*exp(t*Rational(3, 4))/2 + C4*t*exp(t*Rational(3, 4))/2)]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0, 0])

    # 回归测试用例，针对问题 #15574
    # https://github.com/sympy/sympy/issues/15574
    eq7 = [Eq(Derivative(x(t), t), x(t)), Eq(Derivative(y(t), t), y(t)), Eq(Derivative(z(t), t), z(t)),
           Eq(Derivative(w(t), t), w(t)), Eq(Derivative(u(t), t), u(t))]
    sol7 = [Eq(x(t), C1*exp(t)), Eq(y(t), C2*exp(t)), Eq(z(t), C3*exp(t)), Eq(w(t), C4*exp(t)),
            Eq(u(t), C5*exp(t))]
    assert dsolve(eq7) == sol7
    assert checksysodesol(eq7, sol7) == (True, [0, 0, 0, 0, 0])

    eqs8 = [Eq(Derivative(x(t), t), 2*x(t) + y(t)),
            Eq(Derivative(y(t), t), 2*y(t)),
            Eq(Derivative(z(t), t), 4*z(t)),
            Eq(Derivative(w(t), t), u(t) + 5*w(t)),
            Eq(Derivative(u(t), t), 5*u(t))]
    sol8 = [Eq(x(t), C1*exp(2*t) + C2*t*exp(2*t)),
            Eq(y(t), C2*exp(2*t)),
            Eq(z(t), C3*exp(4*t)),
            Eq(w(t), C4*exp(5*t) + C5*t*exp(5*t)),
            Eq(u(t), C5*exp(5*t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0, 0, 0, 0])

    # 回归测试用例，针对问题 #15574
    # https://github.com/sympy/sympy/issues/15574
    eq9 = [Eq(Derivative(x(t), t), x(t)), Eq(Derivative(y(t), t), y(t)), Eq(Derivative(z(t), t), z(t))]
    sol9 = [Eq(x(t), C1*exp(t)), Eq(y(t), C2*exp(t)), Eq(z(t), C3*exp(t))]
    assert dsolve(eq9) == sol9
    assert checksysodesol(eq9, sol9) == (True, [0, 0, 0])

    # 回归测试用例，针对问题 #15407
    # https://github.com/sympy/sympy/issues/15407
    eqs10 = [Eq(Derivative(x(t), t), (-a_b - a_c)*x(t)),
             Eq(Derivative(y(t), t), a_b*y(t)),
             Eq(Derivative(z(t), t), a_c*x(t))]
    sol10 = [Eq(x(t), -C1*(a_b + a_c)*exp(-t*(a_b + a_c))/a_c),
             Eq(y(t), C2*exp(a_b*t)),
             Eq(z(t), C1*exp(-t*(a_b + a_c)) + C3)]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0, 0])

    # 用于检验方程组 eqs10 的解是否等于预期的解 sol10
    assert dsolve(eqs10) == sol10
    # 检查方程组 eqs10 的解是否满足其微分方程的定义
    assert checksysodesol(eqs10, sol10) == (True, [0, 0, 0])

    # Regression test case for issue #14312
    # https://github.com/sympy/sympy/issues/14312
    eqs11 = [Eq(Derivative(x(t), t), k3*y(t)),
             Eq(Derivative(y(t), t), (-k2 - k3)*y(t)),
             Eq(Derivative(z(t), t), k2*y(t))]
    sol11 = [Eq(x(t), C1 + C2*k3*exp(-t*(k2 + k3))/k2),
             Eq(y(t), -C2*(k2 + k3)*exp(-t*(k2 + k3))/k2),
             Eq(z(t), C2*exp(-t*(k2 + k3)) + C3)]
    # 用于检验方程组 eqs11 的解是否等于预期的解 sol11
    assert dsolve(eqs11) == sol11
    # 检查方程组 eqs11 的解是否满足其微分方程的定义
    assert checksysodesol(eqs11, sol11) == (True, [0, 0, 0])

    # Regression test case for issue #14312
    # https://github.com/sympy/sympy/issues/14312
    eqs12 = [Eq(Derivative(z(t), t), k2*y(t)),
             Eq(Derivative(x(t), t), k3*y(t)),
             Eq(Derivative(y(t), t), (-k2 - k3)*y(t))]
    sol12 = [Eq(z(t), C1 - C2*k2*exp(-t*(k2 + k3))/(k2 + k3)),
             Eq(x(t), -C2*k3*exp(-t*(k2 + k3))/(k2 + k3) + C3),
             Eq(y(t), C2*exp(-t*(k2 + k3)))]
    # 用于检验方程组 eqs12 的解是否等于预期的解 sol12
    assert dsolve(eqs12) == sol12
    # 检查方程组 eqs12 的解是否满足其微分方程的定义
    assert checksysodesol(eqs12, sol12) == (True, [0, 0, 0])

    f, g, h = symbols('f, g, h', cls=Function)
    a, b, c = symbols('a, b, c')

    # Regression test case for issue #15474
    # https://github.com/sympy/sympy/issues/15474
    eqs13 = [Eq(Derivative(f(t), t), 2*f(t) + g(t)),
             Eq(Derivative(g(t), t), a*f(t))]
    sol13 = [Eq(f(t), C1*exp(t*(sqrt(a + 1) + 1))/(sqrt(a + 1) - 1) - C2*exp(-t*(sqrt(a + 1) - 1))/(sqrt(a + 1) + 1)),
             Eq(g(t), C1*exp(t*(sqrt(a + 1) + 1)) + C2*exp(-t*(sqrt(a + 1) - 1)))]
    # 用于检验方程组 eqs13 的解是否等于预期的解 sol13
    assert dsolve(eqs13) == sol13
    # 检查方程组 eqs13 的解是否满足其微分方程的定义
    assert checksysodesol(eqs13, sol13) == (True, [0, 0])

    eqs14 = [Eq(Derivative(f(t), t), 2*g(t) - 3*h(t)),
             Eq(Derivative(g(t), t), -2*f(t) + 4*h(t)),
             Eq(Derivative(h(t), t), 3*f(t) - 4*g(t))]
    sol14 = [Eq(f(t), 2*C1 - sin(sqrt(29)*t)*(sqrt(29)*C2*Rational(3, 25) + C3*Rational(-8, 25)) -
              cos(sqrt(29)*t)*(C2*Rational(8, 25) + sqrt(29)*C3*Rational(3, 25))),
             Eq(g(t), C1*Rational(3, 2) + sin(sqrt(29)*t)*(sqrt(29)*C2*Rational(4, 25) + C3*Rational(6, 25)) -
              cos(sqrt(29)*t)*(C2*Rational(6, 25) + sqrt(29)*C3*Rational(-4, 25))),
             Eq(h(t), C1 + C2*cos(sqrt(29)*t) - C3*sin(sqrt(29)*t))]
    # 用于检验方程组 eqs14 的解是否等于预期的解 sol14
    assert dsolve(eqs14) == sol14
    # 检查方程组 eqs14 的解是否满足其微分方程的定义
    assert checksysodesol(eqs14, sol14) == (True, [0, 0, 0])

    eqs15 = [Eq(2*Derivative(f(t), t), 12*g(t) - 12*h(t)),
             Eq(3*Derivative(g(t), t), -8*f(t) + 8*h(t)),
             Eq(4*Derivative(h(t), t), 6*f(t) - 6*g(t))]
    # 定义微分方程组的解集合 sol15
    sol15 = [Eq(f(t), C1 - sin(sqrt(29)*t)*(sqrt(29)*C2*Rational(6, 13) + C3*Rational(-16, 13)) -
              cos(sqrt(29)*t)*(C2*Rational(16, 13) + sqrt(29)*C3*Rational(6, 13))),
             Eq(g(t), C1 + sin(sqrt(29)*t)*(sqrt(29)*C2*Rational(8, 39) + C3*Rational(16, 13)) -
              cos(sqrt(29)*t)*(C2*Rational(16, 13) + sqrt(29)*C3*Rational(-8, 39))),
             Eq(h(t), C1 + C2*cos(sqrt(29)*t) - C3*sin(sqrt(29)*t))]
    # 断言求解微分方程组 eqs15 返回的解等于预期的解 sol15
    assert dsolve(eqs15) == sol15
    # 断言检查方程组 eqs15 和解 sol15 是否是一个有效的解
    assert checksysodesol(eqs15, sol15) == (True, [0, 0, 0])

    # 定义微分方程组 eq16
    eq16 = (Eq(diff(x(t), t), 21*x(t)), Eq(diff(y(t), t), 17*x(t) + 3*y(t)),
            Eq(diff(z(t), t), 5*x(t) + 7*y(t) + 9*z(t)))
    # 定义微分方程组 eq16 的解集合 sol16
    sol16 = [Eq(x(t), 216*C1*exp(21*t)/209),
             Eq(y(t), 204*C1*exp(21*t)/209 - 6*C2*exp(3*t)/7),
             Eq(z(t), C1*exp(21*t) + C2*exp(3*t) + C3*exp(9*t))]
    # 断言求解微分方程组 eq16 返回的解等于预期的解 sol16
    assert dsolve(eq16) == sol16
    # 断言检查方程组 eq16 和解 sol16 是否是一个有效的解
    assert checksysodesol(eq16, sol16) == (True, [0, 0, 0])

    # 定义微分方程组 eqs17
    eqs17 = [Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)),
             Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),
             Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))]
    # 定义微分方程组 eqs17 的解集合 sol17
    sol17 = [Eq(x(t), C1*Rational(7, 3) - sin(sqrt(179)*t)*(sqrt(179)*C2*Rational(11, 170) + C3*Rational(-21,
              170)) - cos(sqrt(179)*t)*(C2*Rational(21, 170) + sqrt(179)*C3*Rational(11, 170))),
             Eq(y(t), C1*Rational(11, 3) + sin(sqrt(179)*t)*(sqrt(179)*C2*Rational(7, 170) + C3*Rational(33,
              170)) - cos(sqrt(179)*t)*(C2*Rational(33, 170) + sqrt(179)*C3*Rational(-7, 170))),
             Eq(z(t), C1 + C2*cos(sqrt(179)*t) - C3*sin(sqrt(179)*t))]
    # 断言求解微分方程组 eqs17 返回的解等于预期的解 sol17
    assert dsolve(eqs17) == sol17
    # 断言检查方程组 eqs17 和解 sol17 是否是一个有效的解
    assert checksysodesol(eqs17, sol17) == (True, [0, 0, 0])

    # 定义微分方程组 eqs18
    eqs18 = [Eq(3*Derivative(x(t), t), 20*y(t) - 20*z(t)),
             Eq(4*Derivative(y(t), t), -15*x(t) + 15*z(t)),
             Eq(5*Derivative(z(t), t), 12*x(t) - 12*y(t))]
    # 定义微分方程组 eqs18 的解集合 sol18
    sol18 = [Eq(x(t), C1 - sin(5*sqrt(2)*t)*(sqrt(2)*C2*Rational(4, 3) - C3) - cos(5*sqrt(2)*t)*(C2 +
              sqrt(2)*C3*Rational(4, 3))),
             Eq(y(t), C1 + sin(5*sqrt(2)*t)*(sqrt(2)*C2*Rational(3, 4) + C3) - cos(5*sqrt(2)*t)*(C2 +
              sqrt(2)*C3*Rational(-3, 4))),
             Eq(z(t), C1 + C2*cos(5*sqrt(2)*t) - C3*sin(5*sqrt(2)*t))]
    # 断言求解微分方程组 eqs18 返回的解等于预期的解 sol18
    assert dsolve(eqs18) == sol18
    # 断言检查方程组 eqs18 和解 sol18 是否是一个有效的解
    assert checksysodesol(eqs18, sol18) == (True, [0, 0, 0])

    # 定义微分方程组 eqs19
    eqs19 = [Eq(Derivative(x(t), t), 4*x(t) - z(t)),
             Eq(Derivative(y(t), t), 2*x(t) + 2*y(t) - z(t)),
             Eq(Derivative(z(t), t), 3*x(t) + y(t))]
    # 定义微分方程组 eqs19 的解集合 sol19
    sol19 = [Eq(x(t), C2*t**2*exp(2*t)/2 + t*(2*C2 + C3)*exp(2*t) + (C1 + C2 + 2*C3)*exp(2*t)),
             Eq(y(t), C2*t**2*exp(2*t)/2 + t*(2*C2 + C3)*exp(2*t) + (C1 + 2*C3)*exp(2*t)),
             Eq(z(t), C2*t**2*exp(2*t) + t*(3*C2 + 2*C3)*exp(2*t) + (2*C1 + 3*C3)*exp(2*t))]
    # 断言求解微分方程组 eqs19 返回的解等于预期的解 sol19
    assert dsolve(eqs19) == sol19
    # 断言检查方程组 eqs19 和解 sol19 是否是一个有效的解
    assert checksysodesol(eqs19, sol19) == (True, [0, 0, 0])
    # 定义一个包含三个微分方程的列表
    eqs20 = [Eq(Derivative(x(t), t), 4*x(t) - y(t) - 2*z(t)),
             Eq(Derivative(y(t), t), 2*x(t) + y(t) - 2*z(t)),
             Eq(Derivative(z(t), t), 5*x(t) - 3*z(t))]
    # 解微分方程组，得到的解存储在列表 sol20 中
    sol20 = [Eq(x(t), C1*exp(2*t) - sin(t)*(C2*Rational(3, 5) + C3/5) - cos(t)*(C2/5 + C3*Rational(-3, 5))),
             Eq(y(t), -sin(t)*(C2*Rational(3, 5) + C3/5) - cos(t)*(C2/5 + C3*Rational(-3, 5))),
             Eq(z(t), C1*exp(2*t) - C2*sin(t) + C3*cos(t))]
    # 验证解是否正确
    assert dsolve(eqs20) == sol20
    # 检查所得解是否满足原微分方程组
    assert checksysodesol(eqs20, sol20) == (True, [0, 0, 0])

    # 定义一个包含两个微分方程的列表
    eq21 = (Eq(diff(x(t), t), 9*y(t)), Eq(diff(y(t), t), 12*x(t)))
    # 解微分方程组，得到的解存储在列表 sol21 中
    sol21 = [Eq(x(t), -sqrt(3)*C1*exp(-6*sqrt(3)*t)/2 + sqrt(3)*C2*exp(6*sqrt(3)*t)/2),
             Eq(y(t), C1*exp(-6*sqrt(3)*t) + C2*exp(6*sqrt(3)*t))]
    # 验证解是否正确
    assert dsolve(eq21) == sol21
    # 检查所得解是否满足原微分方程组
    assert checksysodesol(eq21, sol21) == (True, [0, 0])

    # 定义一个包含两个微分方程的列表
    eqs22 = [Eq(Derivative(x(t), t), 2*x(t) + 4*y(t)),
             Eq(Derivative(y(t), t), 12*x(t) + 41*y(t))]
    # 解微分方程组，得到的解存储在列表 sol22 中
    sol22 = [Eq(x(t), C1*(39 - sqrt(1713))*exp(t*(sqrt(1713) + 43)/2)*Rational(-1, 24) + C2*(39 +
              sqrt(1713))*exp(t*(43 - sqrt(1713))/2)*Rational(-1, 24)),
             Eq(y(t), C1*exp(t*(sqrt(1713) + 43)/2) + C2*exp(t*(43 - sqrt(1713))/2))]
    # 验证解是否正确
    assert dsolve(eqs22) == sol22
    # 检查所得解是否满足原微分方程组
    assert checksysodesol(eqs22, sol22) == (True, [0, 0])

    # 定义一个包含两个微分方程的列表
    eqs23 = [Eq(Derivative(x(t), t), x(t) + y(t)),
             Eq(Derivative(y(t), t), -2*x(t) + 2*y(t))]
    # 解微分方程组，得到的解存储在列表 sol23 中
    sol23 = [Eq(x(t), (C1/4 + sqrt(7)*C2/4)*cos(sqrt(7)*t/2)*exp(t*Rational(3, 2)) +
              sin(sqrt(7)*t/2)*(sqrt(7)*C1/4 + C2*Rational(-1, 4))*exp(t*Rational(3, 2))),
             Eq(y(t), C1*cos(sqrt(7)*t/2)*exp(t*Rational(3, 2)) - C2*sin(sqrt(7)*t/2)*exp(t*Rational(3, 2)))]
    # 验证解是否正确
    assert dsolve(eqs23) == sol23
    # 检查所得解是否满足原微分方程组
    assert checksysodesol(eqs23, sol23) == (True, [0, 0])

    # 回归测试案例，用于解决问题 #15474
    # https://github.com/sympy/sympy/issues/15474
    # 定义包含两个微分方程的列表，其中一个包含符号参数 a
    a = Symbol("a", real=True)
    eq24 = [x(t).diff(t) - a*y(t), y(t).diff(t) + a*x(t)]
    # 解微分方程组，得到的解存储在列表 sol24 中
    sol24 = [Eq(x(t), C1*sin(a*t) + C2*cos(a*t)), Eq(y(t), C1*cos(a*t) - C2*sin(a*t))]
    # 验证解是否正确
    assert dsolve(eq24) == sol24
    # 检查所得解是否满足原微分方程组
    assert checksysodesol(eq24, sol24) == (True, [0, 0])

    # 回归测试案例，用于解决问题 #19150
    # https://github.com/sympy/sympy/issues/19150
    # 定义包含五个微分方程的列表，其中每个方程包含一个符号参数 b 和 c
    eqs25 = [Eq(Derivative(f(t), t), 0),
             Eq(Derivative(g(t), t), (f(t) - 2*g(t) + x(t))/(b*c)),
             Eq(Derivative(x(t), t), (g(t) - 2*x(t) + y(t))/(b*c)),
             Eq(Derivative(y(t), t), (h(t) + x(t) - 2*y(t))/(b*c)),
             Eq(Derivative(h(t), t), 0)]
    # 解微分方程组，得到的解存储在列表 sol25 中
    sol25 = [Eq(f(t), -3*C1 + 4*C2),
             Eq(g(t), -2*C1 + 3*C2 - C3*exp(-2*t/(b*c)) + C4*exp(-t*(sqrt(2) + 2)/(b*c)) + C5*exp(-t*(2 -
              sqrt(2))/(b*c))),
             Eq(x(t), -C1 + 2*C2 - sqrt(2)*C4*exp(-t*(sqrt(2) + 2)/(b*c)) + sqrt(2)*C5*exp(-t*(2 -
              sqrt(2))/(b*c))),
             Eq(y(t), C2 + C3*exp(-2*t/(b*c)) + C4*exp(-t*(sqrt(2) + 2)/(b*c)) + C5*exp(-t*(2 - sqrt(2))/(b*c))),
             Eq(h(t), C1)]
    # 验证解是否正确
    assert dsolve(eqs25) == sol25
    # 检查所得解是否满足原微分方程组
    assert checksysodesol(eqs25, sol25) == (True, [0, 0, 0, 0, 0])
    # 断言解微分方程组 eqs25 的解是否等于 sol25
    assert dsolve(eqs25) == sol25
    # 断言验证解 sol25 是否满足微分方程组 eqs25
    assert checksysodesol(eqs25, sol25) == (True, [0, 0, 0, 0, 0])

    # 定义微分方程组 eq26 和其解 sol26
    eq26 = [Eq(Derivative(f(t), t), 2*f(t)), Eq(Derivative(g(t), t), 3*f(t) + 7*g(t))]
    sol26 = [Eq(f(t), -5*C1*exp(2*t)/3), Eq(g(t), C1*exp(2*t) + C2*exp(7*t))]
    # 断言解微分方程组 eq26 的解是否等于 sol26
    assert dsolve(eq26) == sol26
    # 断言验证解 sol26 是否满足微分方程组 eq26
    assert checksysodesol(eq26, sol26) == (True, [0, 0])

    # 定义微分方程组 eq27 和其解 sol27
    eq27 = [Eq(Derivative(f(t), t), -9*I*f(t) - 4*g(t)), Eq(Derivative(g(t), t), -4*I*g(t))]
    sol27 = [Eq(f(t), 4*I*C1*exp(-4*I*t)/5 + C2*exp(-9*I*t)), Eq(g(t), C1*exp(-4*I*t))]
    # 断言解微分方程组 eq27 的解是否等于 sol27
    assert dsolve(eq27) == sol27
    # 断言验证解 sol27 是否满足微分方程组 eq27
    assert checksysodesol(eq27, sol27) == (True, [0, 0])

    # 定义微分方程组 eq28 和其解 sol28
    eq28 = [Eq(Derivative(f(t), t), -9*I*f(t)), Eq(Derivative(g(t), t), -4*I*g(t))]
    sol28 = [Eq(f(t), C1*exp(-9*I*t)), Eq(g(t), C2*exp(-4*I*t))]
    # 断言解微分方程组 eq28 的解是否等于 sol28
    assert dsolve(eq28) == sol28
    # 断言验证解 sol28 是否满足微分方程组 eq28
    assert checksysodesol(eq28, sol28) == (True, [0, 0])

    # 定义微分方程组 eq29 和其解 sol29
    eq29 = [Eq(Derivative(f(t), t), 0), Eq(Derivative(g(t), t), 0)]
    sol29 = [Eq(f(t), C1), Eq(g(t), C2)]
    # 断言解微分方程组 eq29 的解是否等于 sol29
    assert dsolve(eq29) == sol29
    # 断言验证解 sol29 是否满足微分方程组 eq29
    assert checksysodesol(eq29, sol29) == (True, [0, 0])

    # 定义微分方程组 eq30 和其解 sol30
    eq30 = [Eq(Derivative(f(t), t), f(t)), Eq(Derivative(g(t), t), 0)]
    sol30 = [Eq(f(t), C1*exp(t)), Eq(g(t), C2)]
    # 断言解微分方程组 eq30 的解是否等于 sol30
    assert dsolve(eq30) == sol30
    # 断言验证解 sol30 是否满足微分方程组 eq30
    assert checksysodesol(eq30, sol30) == (True, [0, 0])

    # 定义微分方程组 eq31 和其解 sol31
    eq31 = [Eq(Derivative(f(t), t), g(t)), Eq(Derivative(g(t), t), 0)]
    sol31 = [Eq(f(t), C1 + C2*t), Eq(g(t), C2)]
    # 断言解微分方程组 eq31 的解是否等于 sol31
    assert dsolve(eq31) == sol31
    # 断言验证解 sol31 是否满足微分方程组 eq31
    assert checksysodesol(eq31, sol31) == (True, [0, 0])

    # 定义微分方程组 eq32 和其解 sol32
    eq32 = [Eq(Derivative(f(t), t), 0), Eq(Derivative(g(t), t), f(t))]
    sol32 = [Eq(f(t), C1), Eq(g(t), C1*t + C2)]
    # 断言解微分方程组 eq32 的解是否等于 sol32
    assert dsolve(eq32) == sol32
    # 断言验证解 sol32 是否满足微分方程组 eq32
    assert checksysodesol(eq32, sol32) == (True, [0, 0])

    # 定义微分方程组 eq33 和其解 sol33
    eq33 = [Eq(Derivative(f(t), t), 0), Eq(Derivative(g(t), t), g(t))]
    sol33 = [Eq(f(t), C1), Eq(g(t), C2*exp(t))]
    # 断言解微分方程组 eq33 的解是否等于 sol33
    assert dsolve(eq33) == sol33
    # 断言验证解 sol33 是否满足微分方程组 eq33
    assert checksysodesol(eq33, sol33) == (True, [0, 0])

    # 定义微分方程组 eq34 和其解 sol34
    eq34 = [Eq(Derivative(f(t), t), f(t)), Eq(Derivative(g(t), t), I*g(t))]
    sol34 = [Eq(f(t), C1*exp(t)), Eq(g(t), C2*exp(I*t))]
    # 断言解微分方程组 eq34 的解是否等于 sol34
    assert dsolve(eq34) == sol34
    # 断言验证解 sol34 是否满足微分方程组 eq34
    assert checksysodesol(eq34, sol34) == (True, [0, 0])

    # 定义微分方程组 eq35 和其解 sol35
    eq35 = [Eq(Derivative(f(t), t), I*f(t)), Eq(Derivative(g(t), t), -I*g(t))]
    sol35 = [Eq(f(t), C1*exp(I*t)), Eq(g(t), C2*exp(-I*t))]
    # 断言解微分方程组 eq35 的解是否等于 sol35
    assert dsolve(eq35) == sol35
    # 断言验证解 sol35 是否满足微分方程组 eq35
    assert checksysodesol(eq35, sol35) == (True, [0, 0])

    # 定义微分方程组 eq36 和其解 sol36
    eq36 = [Eq(Derivative(f(t), t), I*g(t)), Eq(Derivative(g(t), t), 0)]
    sol36 = [Eq(f(t), I*C1 + I*C2*t), Eq(g(t), C2)]
    # 断言解微分方程组 eq36 的解是否等于 sol36
    assert dsolve(eq36) == sol36
    # 断言验证解 sol36 是否满足微分方程组 eq36
    assert checksysodesol(eq36, sol36) == (True, [0, 0])

    # 定义微分方程组 eq37 和其解 sol37
    eq37 = [Eq(Derivative(f(t), t), I*g(t)), Eq(Derivative(g(t), t), I*f(t))]
    sol37 = [Eq(f(t), -C1*exp(-I*t) + C2*exp(I*t)), Eq(g(t), C1*exp(-I*t) + C2*exp(I*t))]
    # 断言解微分方程组 eq37 的解是否等于 sol37
    assert dsolve(eq37) == sol37
    # 断言验证解 sol37 是否满足微分方程组 eq37
    assert checksysodesol(eq37, sol37) == (True, [0, 0])

    # 多个方程组的情况
    # 定义微分方程组 eq1
    eq1 = [Eq(Derivative(f(t), t)**2, g(t)**2), Eq(-f(t) + Derivative(g(t), t), 0)]
    # 定义两个不同解的列表 sol1，每个解包含两个方程
    sol1 = [[Eq(f(t), -C1*sin(t) - C2*cos(t)),
             Eq(g(t), C1*cos(t) - C2*sin(t))],
            [Eq(f(t), -C1*exp(-t) + C2*exp(t)),
             Eq(g(t), C1*exp(-t) + C2*exp(t))]]
    
    # 使用 dsolve 函数对方程 eq1 进行求解，预期结果与 sol1 相等
    assert dsolve(eq1) == sol1
    
    # 遍历 sol1 中的每个解 sol，并检查它们是否满足方程组 eq1 的解的条件
    for sol in sol1:
        # 调用 checksysodesol 函数检查 sol 是否是方程组 eq1 的解
        # 返回结果是一个元组，第一个元素是布尔值表示是否满足解的条件，
        # 第二个元素是一个列表，包含额外的信息，这里期望是 [0, 0] 表示零额外信息
        assert checksysodesol(eq1, sol) == (True, [0, 0])
def test_sysode_linear_neq_order1_type2():

    # 定义函数和符号变量
    f, g, h, k = symbols('f g h k', cls=Function)
    x, t, a, b, c, d, y = symbols('x t a b c d y')
    k1, k2 = symbols('k1 k2')

    # 第一个方程组及其解
    eqs1 = [Eq(Derivative(f(x), x), f(x) + g(x) + 5),
            Eq(Derivative(g(x), x), -f(x) - g(x) + 7)]
    sol1 = [Eq(f(x), C1 + C2 + 6*x**2 + x*(C2 + 5)),
            Eq(g(x), -C1 - 6*x**2 - x*(C2 - 7))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    # 第二个方程组及其解
    eqs2 = [Eq(Derivative(f(x), x), f(x) + g(x) + 5),
            Eq(Derivative(g(x), x), f(x) + g(x) + 7)]
    sol2 = [Eq(f(x), -C1 + C2*exp(2*x) - x - 3),
            Eq(g(x), C1 + C2*exp(2*x) + x - 3)]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    # 第三个方程组及其解
    eqs3 = [Eq(Derivative(f(x), x), f(x) + 5),
            Eq(Derivative(g(x), x), f(x) + 7)]
    sol3 = [Eq(f(x), C1*exp(x) - 5),
            Eq(g(x), C1*exp(x) + C2 + 2*x - 5)]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])

    # 第四个方程组及其解
    eqs4 = [Eq(Derivative(f(x), x), f(x) + exp(x)),
            Eq(Derivative(g(x), x), x*exp(x) + f(x) + g(x))]
    sol4 = [Eq(f(x), C1*exp(x) + x*exp(x)),
            Eq(g(x), C1*x*exp(x) + C2*exp(x) + x**2*exp(x))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])

    # 第五个方程组及其解
    eqs5 = [Eq(Derivative(f(x), x), 5*x + f(x) + g(x)),
            Eq(Derivative(g(x), x), f(x) - g(x))]
    sol5 = [Eq(f(x), C1*(1 + sqrt(2))*exp(sqrt(2)*x) + C2*(1 - sqrt(2))*exp(-sqrt(2)*x) + x*Rational(-5, 2) +
             Rational(-5, 2)),
            Eq(g(x), C1*exp(sqrt(2)*x) + C2*exp(-sqrt(2)*x) + x*Rational(-5, 2))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])

    # 第六个方程组及其解
    eqs6 = [Eq(Derivative(f(x), x), -9*f(x) - 4*g(x)),
            Eq(Derivative(g(x), x), -4*g(x)),
            Eq(Derivative(h(x), x), h(x) + exp(x))]
    sol6 = [Eq(f(x), C2*exp(-4*x)*Rational(-4, 5) + C1*exp(-9*x)),
            Eq(g(x), C2*exp(-4*x)),
            Eq(h(x), C3*exp(x) + x*exp(x))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0])

    # Regression test case for issue #8859
    # https://github.com/sympy/sympy/issues/8859
    eqs7 = [Eq(Derivative(f(t), t), 3*t + f(t)),
            Eq(Derivative(g(t), t), g(t))]
    sol7 = [Eq(f(t), C1*exp(t) - 3*t - 3),
            Eq(g(t), C2*exp(t))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])

    # Regression test case for issue #8567
    # https://github.com/sympy/sympy/issues/8567
    eqs8 = [Eq(Derivative(f(t), t), f(t) + 2*g(t)),
            Eq(Derivative(g(t), t), -2*f(t) + g(t) + 2*exp(t))]
    sol8 = [Eq(f(t), C1*exp(t)*sin(2*t) + C2*exp(t)*cos(2*t)
                + exp(t)*sin(2*t)**2 + exp(t)*cos(2*t)**2),
            Eq(g(t), C1*exp(t)*cos(2*t) - C2*exp(t)*sin(2*t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])
    # 对于问题 #19150 的回归测试案例
    # https://github.com/sympy/sympy/issues/19150
    eqs9 = [Eq(Derivative(f(t), t), (c - 2*f(t) + g(t))/(a*b)),
            Eq(Derivative(g(t), t), (f(t) - 2*g(t) + h(t))/(a*b)),
            Eq(Derivative(h(t), t), (d + g(t) - 2*h(t))/(a*b))]
    sol9 = [Eq(f(t), -C1*exp(-2*t/(a*b)) + C2*exp(-t*(sqrt(2) + 2)/(a*b)) + C3*exp(-t*(2 - sqrt(2))/(a*b)) +
             Mul(Rational(1, 4), 3*c + d, evaluate=False)),
            Eq(g(t), -sqrt(2)*C2*exp(-t*(sqrt(2) + 2)/(a*b)) + sqrt(2)*C3*exp(-t*(2 - sqrt(2))/(a*b)) +
             Mul(Rational(1, 2), c + d, evaluate=False)),
            Eq(h(t), C1*exp(-2*t/(a*b)) + C2*exp(-t*(sqrt(2) + 2)/(a*b)) + C3*exp(-t*(2 - sqrt(2))/(a*b)) +
             Mul(Rational(1, 4), c + 3*d, evaluate=False))]
    # 断言求解微分方程组的结果等于预期解
    assert dsolve(eqs9) == sol9
    # 断言检查系统的解是否满足微分方程组
    assert checksysodesol(eqs9, sol9) == (True, [0, 0, 0])

    # 对于问题 #16635 的回归测试案例
    # https://github.com/sympy/sympy/issues/16635
    eqs10 = [Eq(Derivative(f(t), t), 15*t + f(t) - g(t) - 10),
             Eq(Derivative(g(t), t), -15*t + f(t) - g(t) - 5)]
    sol10 = [Eq(f(t), C1 + C2 + 5*t**3 + 5*t**2 + t*(C2 - 10)),
             Eq(g(t), C1 + 5*t**3 - 10*t**2 + t*(C2 - 5))]
    # 断言求解微分方程组的结果等于预期解
    assert dsolve(eqs10) == sol10
    # 断言检查系统的解是否满足微分方程组
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])

    # 多解情况的测试案例
    eqs11 = [Eq(Derivative(f(t), t)**2 - 2*Derivative(f(t), t) + 1, 4),
             Eq(-y*f(t) + Derivative(g(t), t), 0)]
    sol11 = [[Eq(f(t), C1 - t), Eq(g(t), C1*t*y + C2*y + t**2*y*Rational(-1, 2))],
             [Eq(f(t), C1 + 3*t), Eq(g(t), C1*t*y + C2*y + t**2*y*Rational(3, 2))]]
    # 断言求解微分方程组的结果等于预期解集
    assert dsolve(eqs11) == sol11
    # 遍历每个解并断言检查其是否满足微分方程组
    for s11 in sol11:
        assert checksysodesol(eqs11, s11) == (True, [0, 0])

    # 对于问题 #19831 的回归测试案例
    # https://github.com/sympy/sympy/issues/19831
    n = symbols('n', positive=True)
    x0 = symbols('x_0')
    t0 = symbols('t_0')
    x_0 = symbols('x_0')
    t_0 = symbols('t_0')
    t = symbols('t')
    x = Function('x')
    y = Function('y')
    T = symbols('T')

    eqs12 = [Eq(Derivative(y(t), t), x(t)),
             Eq(Derivative(x(t), t), n*(y(t) + 1))]
    sol12 = [Eq(y(t), C1*exp(sqrt(n)*t)*n**Rational(-1, 2) - C2*exp(-sqrt(n)*t)*n**Rational(-1, 2) - 1),
             Eq(x(t), C1*exp(sqrt(n)*t) + C2*exp(-sqrt(n)*t))]
    # 断言求解微分方程组的结果等于预期解
    assert dsolve(eqs12) == sol12
    # 断言检查系统的解是否满足微分方程组
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])

    # 针对问题 #19831 的附加测试案例
    sol12b = [
        Eq(y(t), (T*exp(-sqrt(n)*t_0)/2 + exp(-sqrt(n)*t_0)/2 +
            x_0*exp(-sqrt(n)*t_0)/(2*sqrt(n)))*exp(sqrt(n)*t) +
            (T*exp(sqrt(n)*t_0)/2 + exp(sqrt(n)*t_0)/2 -
                x_0*exp(sqrt(n)*t_0)/(2*sqrt(n)))*exp(-sqrt(n)*t) - 1),
        Eq(x(t), (T*sqrt(n)*exp(-sqrt(n)*t_0)/2 + sqrt(n)*exp(-sqrt(n)*t_0)/2
            + x_0*exp(-sqrt(n)*t_0)/2)*exp(sqrt(n)*t)
                - (T*sqrt(n)*exp(sqrt(n)*t_0)/2 + sqrt(n)*exp(sqrt(n)*t_0)/2 -
                    x_0*exp(sqrt(n)*t_0)/2)*exp(-sqrt(n)*t))
    ]
    assert dsolve(eqs12, ics={y(t0): T, x(t0): x0}) == sol12b
    # 使用 SymPy 的 dsolve 函数求解微分方程组 eqs12，使用初始条件 ics，确保在 t=t0 时 y(t)=T 和 x(t)=x0，检验结果是否等于 sol12b。

    assert checksysodesol(eqs12, sol12b) == (True, [0, 0])
    # 使用 SymPy 的 checksysodesol 函数检查 dsolve 求解的微分方程组 eqs12 和解 sol12b 是否满足微分方程的定义。期望返回结果为 True 和 [0, 0]。

    # Test cases added for the issue 19763
    # https://github.com/sympy/sympy/issues/19763

    eq13 = [Eq(Derivative(f(t), t), f(t) + g(t) + 9),
            # 定义微分方程组 eq13
            Eq(Derivative(g(t), t), 2*f(t) + 5*g(t) + 23)]
            # 第二个微分方程

    sol13 = [Eq(f(t), -C1*(2 + sqrt(6))*exp(t*(3 - sqrt(6)))/2 - C2*(2 - sqrt(6))*exp(t*(sqrt(6) + 3))/2 - Rational(22,3)),
             # eq13 的第一个方程的解
             Eq(g(t), C1*exp(t*(3 - sqrt(6))) + C2*exp(t*(sqrt(6) + 3)) - Rational(5,3))]
             # eq13 的第二个方程的解

    assert dsolve(eq13) == sol13
    # 使用 dsolve 求解微分方程组 eq13，并验证其解是否等于 sol13。

    assert checksysodesol(eq13, sol13) == (True, [0, 0])
    # 使用 checksysodesol 检查 dsolve 求解的微分方程组 eq13 和解 sol13 是否满足微分方程的定义。期望返回结果为 True 和 [0, 0]。

    eq14 = [Eq(Derivative(f(t), t), f(t) + g(t) + 81),
            # 定义微分方程组 eq14
            Eq(Derivative(g(t), t), -2*f(t) + g(t) + 23)]
            # 第二个微分方程

    sol14 = [Eq(f(t), sqrt(2)*C1*exp(t)*sin(sqrt(2)*t)/2
                    + sqrt(2)*C2*exp(t)*cos(sqrt(2)*t)/2
                    - 58*sin(sqrt(2)*t)**2/3 - 58*cos(sqrt(2)*t)**2/3),
             # eq14 的第一个方程的解
             Eq(g(t), C1*exp(t)*cos(sqrt(2)*t) - C2*exp(t)*sin(sqrt(2)*t)
                    - 185*sin(sqrt(2)*t)**2/3 - 185*cos(sqrt(2)*t)**2/3)]
             # eq14 的第二个方程的解

    assert dsolve(eq14) == sol14
    # 使用 dsolve 求解微分方程组 eq14，并验证其解是否等于 sol14。

    assert checksysodesol(eq14, sol14) == (True, [0,0])
    # 使用 checksysodesol 检查 dsolve 求解的微分方程组 eq14 和解 sol14 是否满足微分方程的定义。期望返回结果为 True 和 [0, 0]。

    eq15 = [Eq(Derivative(f(t), t), f(t) + 2*g(t) + k1),
            # 定义微分方程组 eq15
            Eq(Derivative(g(t), t), 3*f(t) + 4*g(t) + k2)]
            # 第二个微分方程

    sol15 = [Eq(f(t), -C1*(3 - sqrt(33))*exp(t*(5 + sqrt(33))/2)/6 -
              C2*(3 + sqrt(33))*exp(t*(5 - sqrt(33))/2)/6 + 2*k1 - k2),
             # eq15 的第一个方程的解
             Eq(g(t), C1*exp(t*(5 + sqrt(33))/2) + C2*exp(t*(5 - sqrt(33))/2) -
              Mul(Rational(1,2), 3*k1 - k2, evaluate = False))]
             # eq15 的第二个方程的解

    assert dsolve(eq15) == sol15
    # 使用 dsolve 求解微分方程组 eq15，并验证其解是否等于 sol15。

    assert checksysodesol(eq15, sol15) == (True, [0,0])
    # 使用 checksysodesol 检查 dsolve 求解的微分方程组 eq15 和解 sol15 是否满足微分方程的定义。期望返回结果为 True 和 [0, 0]。

    eq16 = [Eq(Derivative(f(t), t), k1),
            # 定义微分方程组 eq16
            Eq(Derivative(g(t), t), k2)]
            # 第二个微分方程

    sol16 = [Eq(f(t), C1 + k1*t),
             # eq16 的第一个方程的解
             Eq(g(t), C2 + k2*t)]
             # eq16 的第二个方程的解

    assert dsolve(eq16) == sol16
    # 使用 dsolve 求解微分方程组 eq16，并验证其解是否等于 sol16。

    assert checksysodesol(eq16, sol16) == (True, [0,0])
    # 使用 checksysodesol 检查 dsolve 求解的微分方程组 eq16 和解 sol16 是否满足微分方程的定义。期望返回结果为 True 和 [0, 0]。

    eq17 = [Eq(Derivative(f(t), t), 0),
            # 定义微分方程组 eq17
            Eq(Derivative(g(t), t), c*f(t) + k2)]
            # 第二个微分方程

    sol17 = [Eq(f(t), C1),
             # eq17 的第一个方程的解
             Eq(g(t), C2*c + t*(C1*c + k2))]
             # eq17 的第二个方程的解

    assert dsolve(eq17) == sol17
    # 使用 dsolve 求解微分方程组 eq17，并验证其解是否等于 sol17。

    assert checksysodesol(eq17 , sol17) == (True , [0,0])
    # 使用 checksysodesol 检查 dsolve 求解的微分方程组 eq17 和解 sol17 是否满足微分方程的定义。期望返回结果为 True 和 [0, 0]。

    eq18 = [Eq(Derivative(f(t), t), k1),
            # 定义微分方程组 eq18
            Eq(Derivative(g(t), t), f(t) + k2)]
            # 第二个微分方程

    sol18 = [Eq(f(t), C1 + k1*t),
             # eq18 的第一个方程的解
             Eq(g(t), C2 + k1*t**2/2 + t*(C1 + k2))]
             # eq18 的第二个方程的解

    assert dsolve(eq18) == sol18
    # 使用 dsolve 求解微分方程组 eq18，并验证其解是否等于 sol18。

    assert checksysodesol(eq18 , sol18) == (True , [0,0])
    # 使用 checksysodesol 检查 dsolve 求解的微分方程组 eq18 和解 sol18 是否满足微分方程的定义。期望返回结果为 True 和 [0, 0]。

    eq19 = [Eq(Derivative(f(t), t), k1),
            # 定义微分方程组 eq19
            Eq(Derivative(g(t), t), f(t) + 2*g(t) + k2)]
            # 第二个微分方程

    sol19 = [Eq(f(t), -2*C1 + k1*t),
            # eq19 的第一个方程的解
            Eq(g(t), C1 + C2*exp(2*t) - k1*t/2 - Mul(Rational(1,4), k1 + 2*k2 , evaluate = False))]
            # eq19 的第二个方程的解

    assert dsolve(eq19) == sol19
    # 使用 dsolve 求解
   `
# 检查函数 checksysodesol 对于给定的方程组 eq20 和其解 sol20 是否返回预期的结果
assert checksysodesol(eq20 , sol20) == (True , [0,0])

# 定义一个包含两个微分方程的列表 eq21
eq21 = [Eq(diff(f(t), t), g(t) + k1),
        Eq(diff(g(t), t), 0)]
# 定义预期的解 sol21，其中包含 f(t) 和 g(t) 的表达式
sol21 = [Eq(f(t), C1 + t*(C2 + k1)),
         Eq(g(t), C2)]
# 检查函数 dsolve 是否正确解决了微分方程组 eq21，并验证解是否等于 sol21
assert dsolve(eq21) == sol21
# 检查函数 checksysodesol 对于方程组 eq21 和其解 sol21 是否返回预期的结果
assert checksysodesol(eq21 , sol21) == (True , [0,0])

# 定义一个包含两个带有常数 k1 和 k2 的微分方程的列表 eq22
eq22 = [Eq(Derivative(f(t), t), f(t) + 2*g(t) + k1),
        Eq(Derivative(g(t), t), k2)]
# 定义预期的解 sol22，其中包含 f(t) 和 g(t) 的表达式
sol22 = [Eq(f(t), -2*C1 + C2*exp(t) - k1 - 2*k2*t - 2*k2),
         Eq(g(t), C1 + k2*t)]
# 检查函数 dsolve 是否正确解决了微分方程组 eq22，并验证解是否等于 sol22
assert dsolve(eq22) == sol22
# 检查函数 checksysodesol 对于方程组 eq22 和其解 sol22 是否返回预期的结果
assert checksysodesol(eq22 , sol22) == (True , [0,0])

# 定义一个包含带有常数 k1 和 k2 的两个微分方程的列表 eq23
eq23 = [Eq(Derivative(f(t), t), g(t) + k1),
        Eq(Derivative(g(t), t), 2*g(t) + k2)]
# 定义预期的解 sol23，其中包含 f(t) 和 g(t) 的表达式
sol23 = [Eq(f(t), C1 + C2*exp(2*t)/2 - k2/4 + t*(2*k1 - k2)/2),
         Eq(g(t), C2*exp(2*t) - k2/2)]
# 检查函数 dsolve 是否正确解决了微分方程组 eq23，并验证解是否等于 sol23
assert dsolve(eq23) == sol23
# 检查函数 checksysodesol 对于方程组 eq23 和其解 sol23 是否返回预期的结果
assert checksysodesol(eq23 , sol23) == (True , [0,0])

# 定义一个包含带有常数 k1 和 k2 的两个微分方程的列表 eq24
eq24 = [Eq(Derivative(f(t), t), f(t) + k1),
        Eq(Derivative(g(t), t), 2*f(t) + k2)]
# 定义预期的解 sol24，其中包含 f(t) 和 g(t) 的表达式
sol24 = [Eq(f(t), C1*exp(t)/2 - k1),
         Eq(g(t), C1*exp(t) + C2 - 2*k1 - t*(2*k1 - k2))]
# 检查函数 dsolve 是否正确解决了微分方程组 eq24，并验证解是否等于 sol24
assert dsolve(eq24) == sol24
# 检查函数 checksysodesol 对于方程组 eq24 和其解 sol24 是否返回预期的结果
assert checksysodesol(eq24 , sol24) == (True , [0,0])

# 定义一个包含带有常数 k1 和 k2 的两个微分方程的列表 eq25
eq25 = [Eq(Derivative(f(t), t), f(t) + 2*g(t) + k1),
        Eq(Derivative(g(t), t), 3*f(t) + 6*g(t) + k2)]
# 定义预期的解 sol25，其中包含 f(t) 和 g(t) 的表达式
sol25 = [Eq(f(t), -2*C1 + C2*exp(7*t)/3 + 2*t*(3*k1 - k2)/7 -
          Mul(Rational(1,49), k1 + 2*k2 , evaluate = False)),
         Eq(g(t), C1 + C2*exp(7*t) - t*(3*k1 - k2)/7 -
          Mul(Rational(3,49), k1 + 2*k2 , evaluate = False))]
# 检查函数 dsolve 是否正确解决了微分方程组 eq25，并验证解是否等于 sol25
assert dsolve(eq25) == sol25
# 检查函数 checksysodesol 对于方程组 eq25 和其解 sol25 是否返回预期的结果
assert checksysodesol(eq25 , sol25) == (True , [0,0])

# 定义一个包含带有常数 k1 和 k2 的两个微分方程的列表 eq26
eq26 = [Eq(Derivative(f(t), t), 2*f(t) - g(t) + k1),
        Eq(Derivative(g(t), t), 4*f(t) - 2*g(t) + 2*k1)]
# 定义预期的解 sol26，其中包含 f(t) 和 g(t) 的表达式
sol26 = [Eq(f(t), C1 + 2*C2 + t*(2*C1 + k1)),
         Eq(g(t), 4*C2 + t*(4*C1 + 2*k1))]
# 检查函数 dsolve 是否正确解决了微分方程组 eq26，并验证解是否等于 sol26
assert dsolve(eq26) == sol26
# 检查函数 checksysodesol 对于方程组 eq26 和其解 sol26 是否返回预期的结果
assert checksysodesol(eq26 , sol26) == (True , [0,0])

# 测试案例用于解决问题＃22715
# https://github.com/sympy/sympy/issues/22715
# 定义一个包含带有常数的两个微分方程的列表 eq27
eq27 = [Eq(diff(x(t),t),-1*y(t)+10), Eq(diff(y(t),t),5*x(t)-2*y(t)+3)]
# 定义预期的解 sol27，其中包含 x(t) 和 y(t) 的表达式
sol27 = [Eq(x(t), (C1/5 - 2*C2/5)*exp(-t)*cos(2*t)
                - (2*C1/5 + C2/5)*exp(-t)*sin(2*t)
                + 17*sin(2*t)**2/5 + 17*cos(2*t)**2),
        Eq(y(t), C1*exp(-t)*cos(2*t) - C2*exp(-t)*sin(2*t)
                + 10*sin(2*t)**2 + 10*cos(2*t)**2)]
# 检查函数 dsolve 是否正确解决了微分方程组 eq27，并验证解是否等于 sol27
assert dsolve(eq27) == sol27
# 检查函数 checksysodesol 对于方程组 eq27 和其解 sol27 是否返回预期的结果
assert checksysodesol(eq27 , sol27) == (True , [0,0])
# 定义一个测试函数，测试解线性非齐次常系数微分方程组类型3的情况
def test_sysode_linear_neq_order1_type3():

    # 定义符号变量和函数
    f, g, h, k, x0 , y0 = symbols('f g h k x0 y0', cls=Function)
    x, t, a = symbols('x t a')
    r = symbols('r', real=True)

    # 第一组微分方程
    eqs1 = [Eq(Derivative(f(r), r), r*g(r) + f(r)),
            Eq(Derivative(g(r), r), -r*f(r) + g(r))]
    # 解第一组微分方程得到的通解
    sol1 = [Eq(f(r), C1*exp(r)*sin(r**2/2) + C2*exp(r)*cos(r**2/2)),
            Eq(g(r), C1*exp(r)*cos(r**2/2) - C2*exp(r)*sin(r**2/2))]
    # 断言解的正确性
    assert dsolve(eqs1) == sol1
    # 检查解是否满足微分方程组
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    # 第二组微分方程
    eqs2 = [Eq(Derivative(f(x), x), x**2*g(x) + x*f(x)),
            Eq(Derivative(g(x), x), 2*x**2*f(x) + (3*x**2 + x)*g(x))]
    # 解第二组微分方程得到的通解
    sol2 = [Eq(f(x), (sqrt(17)*C1/17 + C2*(17 - 3*sqrt(17))/34)*exp(x**3*(3 + sqrt(17))/6 + x**2/2) -
             exp(x**3*(3 - sqrt(17))/6 + x**2/2)*(sqrt(17)*C1/17 + C2*(3*sqrt(17) + 17)*Rational(-1, 34))),
            Eq(g(x), exp(x**3*(3 - sqrt(17))/6 + x**2/2)*(C1*(17 - 3*sqrt(17))/34 + sqrt(17)*C2*Rational(-2,
             17)) + exp(x**3*(3 + sqrt(17))/6 + x**2/2)*(C1*(3*sqrt(17) + 17)/34 + sqrt(17)*C2*Rational(2, 17)))]
    # 断言解的正确性
    assert dsolve(eqs2) == sol2
    # 检查解是否满足微分方程组
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    # 第三组微分方程
    eqs3 = [Eq(f(x).diff(x), x*f(x) + g(x)),
            Eq(g(x).diff(x), -f(x) + x*g(x))]
    # 解第三组微分方程得到的通解
    sol3 = [Eq(f(x), (C1/2 + I*C2/2)*exp(x**2/2 - I*x) + exp(x**2/2 + I*x)*(C1/2 + I*C2*Rational(-1, 2))),
            Eq(g(x), (I*C1/2 + C2/2)*exp(x**2/2 + I*x) - exp(x**2/2 - I*x)*(I*C1/2 + C2*Rational(-1, 2)))]
    # 断言解的正确性
    assert dsolve(eqs3) == sol3
    # 检查解是否满足微分方程组
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])

    # 第四组微分方程
    eqs4 = [Eq(f(x).diff(x), x*(f(x) + g(x) + h(x))), Eq(g(x).diff(x), x*(f(x) + g(x) + h(x))),
            Eq(h(x).diff(x), x*(f(x) + g(x) + h(x)))]
    # 解第四组微分方程得到的通解
    sol4 = [Eq(f(x), -C1/3 - C2/3 + 2*C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2)),
            Eq(g(x), 2*C1/3 - C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2)),
            Eq(h(x), -C1/3 + 2*C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2))]
    # 断言解的正确性
    assert dsolve(eqs4) == sol4
    # 检查解是否满足微分方程组
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0])

    # 第五组微分方程
    eqs5 = [Eq(f(x).diff(x), x**2*(f(x) + g(x) + h(x))), Eq(g(x).diff(x), x**2*(f(x) + g(x) + h(x))),
            Eq(h(x).diff(x), x**2*(f(x) + g(x) + h(x)))]
    # 解第五组微分方程得到的通解
    sol5 = [Eq(f(x), -C1/3 - C2/3 + 2*C3/3 + (C1/3 + C2/3 + C3/3)*exp(x**3)),
            Eq(g(x), 2*C1/3 - C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(x**3)),
            Eq(h(x), -C1/3 + 2*C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(x**3))]
    # 断言解的正确性
    assert dsolve(eqs5) == sol5
    # 检查解是否满足微分方程组
    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0])

    # 第六组微分方程
    eqs6 = [Eq(Derivative(f(x), x), x*(f(x) + g(x) + h(x) + k(x))),
            Eq(Derivative(g(x), x), x*(f(x) + g(x) + h(x) + k(x))),
            Eq(Derivative(h(x), x), x*(f(x) + g(x) + h(x) + k(x))),
            Eq(Derivative(k(x), x), x*(f(x) + g(x) + h(x) + k(x)))]
    # 定义包含四个微分方程的列表，每个方程使用SymPy的Eq对象表示
    sol6 = [Eq(f(x), -C1/4 - C2/4 - C3/4 + 3*C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(2*x**2)),
            Eq(g(x), 3*C1/4 - C2/4 - C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(2*x**2)),
            Eq(h(x), -C1/4 + 3*C2/4 - C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(2*x**2)),
            Eq(k(x), -C1/4 - C2/4 + 3*C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(2*x**2))]
    # 断言解dsolve(eqs6)等于sol6
    assert dsolve(eqs6) == sol6
    # 断言checksysodesol(eqs6, sol6)返回True和[0, 0, 0, 0]
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0, 0])

    # 定义符号y，声明其为实数
    y = symbols("y", real=True)

    # 定义包含两个微分方程的列表，每个方程使用SymPy的Eq对象表示
    eqs7 = [Eq(Derivative(f(y), y), y*f(y) + g(y)),
            Eq(Derivative(g(y), y), y*g(y) - f(y))]
    # 定义期望的解sol7，每个解使用SymPy的Eq对象表示
    sol7 = [Eq(f(y), C1*exp(y**2/2)*sin(y) + C2*exp(y**2/2)*cos(y)),
            Eq(g(y), C1*exp(y**2/2)*cos(y) - C2*exp(y**2/2)*sin(y))]
    # 断言解dsolve(eqs7)等于sol7
    assert dsolve(eqs7) == sol7
    # 断言checksysodesol(eqs7, sol7)返回True和[0, 0]
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])

    # 为问题19763添加测试用例
    # https://github.com/sympy/sympy/issues/19763

    # 定义包含两个微分方程的列表，每个方程使用SymPy的Eq对象表示
    eqs8 = [Eq(Derivative(f(t), t), 5*t*f(t) + 2*h(t)),
            Eq(Derivative(h(t), t), 2*f(t) + 5*t*h(t))]
    # 定义期望的解sol8，每个解使用SymPy的Eq对象表示
    sol8 = [Eq(f(t), Mul(-1, (C1/2 - C2/2), evaluate = False)*exp(5*t**2/2 - 2*t) + (C1/2 + C2/2)*exp(5*t**2/2 + 2*t)),
            Eq(h(t), (C1/2 - C2/2)*exp(5*t**2/2 - 2*t) + (C1/2 + C2/2)*exp(5*t**2/2 + 2*t))]
    # 断言解dsolve(eqs8)等于sol8
    assert dsolve(eqs8) == sol8
    # 断言checksysodesol(eqs8, sol8)返回True和[0, 0]
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])

    # 定义包含两个微分方程的列表，每个方程使用SymPy的Eq对象表示
    eqs9 = [Eq(diff(f(t), t), 5*t*f(t) + t**2*g(t)),
            Eq(diff(g(t), t), -t**2*f(t) + 5*t*g(t))]
    # 定义期望的解sol9，每个解使用SymPy的Eq对象表示
    sol9 = [Eq(f(t), (C1/2 - I*C2/2)*exp(I*t**3/3 + 5*t**2/2) + (C1/2 + I*C2/2)*exp(-I*t**3/3 + 5*t**2/2)),
            Eq(g(t), Mul(-1, (I*C1/2 - C2/2) , evaluate = False)*exp(-I*t**3/3 + 5*t**2/2) + (I*C1/2 + C2/2)*exp(I*t**3/3 + 5*t**2/2))]
    # 断言解dsolve(eqs9)等于sol9
    assert dsolve(eqs9) == sol9
    # 断言checksysodesol(eqs9, sol9)返回True和[0, 0]
    assert checksysodesol(eqs9 , sol9) == (True , [0,0])

    # 定义包含两个微分方程的列表，每个方程使用SymPy的Eq对象表示
    eqs10 = [Eq(diff(f(t), t), t**2*g(t) + 5*t*f(t)),
             Eq(diff(g(t), t), -t**2*f(t) + (9*t**2 + 5*t)*g(t))]
    # 定义期望的解sol10，每个解使用SymPy的Eq对象表示
    sol10 = [Eq(f(t), (C1*(77 - 9*sqrt(77))/154 + sqrt(77)*C2/77)*exp(t**3*(sqrt(77) + 9)/6 + 5*t**2/2) + (C1*(77 + 9*sqrt(77))/154 - sqrt(77)*C2/77)*exp(t**3*(9 - sqrt(77))/6 + 5*t**2/2)),
             Eq(g(t), (sqrt(77)*C1/77 + C2*(77 - 9*sqrt(77))/154)*exp(t**3*(9 - sqrt(77))/6 + 5*t**2/2) - (sqrt(77)*C1/77 - C2*(77 + 9*sqrt(77))/154)*exp(t**3*(sqrt(77) + 9)/6 + 5*t**2/2))]
    # 断言解dsolve(eqs10)等于sol10
    assert dsolve(eqs10 , sol10) == (True , [0,0])
    
    # 定义包含两个微分方程的列表，每个方程使用SymPy的Eq对象表示
    eqs11 = [Eq(diff(f(t), t), 5*t*f(t) + t**2*g(t)),
             Eq(diff(g(t), t), (1-t**2)*f(t) + (5*t + 9*t**2)*g(t))]
    # 定义期望的解sol11，每个解使用SymPy的Eq对象表示
    sol11 = [Eq(f(t), C1*x0(t) + C2*x0(t)*Integral(t**2*exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)**2, t)),
             Eq(g(t), C1*y0(t) + C2*(y0(t)*Integral(t**2*exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)**2, t) + exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)))]
    # 断言解dsolve(eqs11)等于sol11
    assert dsolve(eqs11) == sol11
# 定义一个测试函数，用于测试一阶非线性常微分方程组中类型为4的系统
@slow
def test_sysode_linear_neq_order1_type4():
    # 声明符号变量和函数
    f, g, h, k = symbols('f g h k', cls=Function)
    x, t, a = symbols('x t a')
    r = symbols('r', real=True)

    # 第一组微分方程
    eqs1 = [Eq(diff(f(r), r), f(r) + r*g(r) + r**2), Eq(diff(g(r), r), -r*f(r) + g(r) + r)]
    # 解第一组微分方程
    sol1 = [Eq(f(r), C1*exp(r)*sin(r**2/2) + C2*exp(r)*cos(r**2/2) + exp(r)*sin(r**2/2)*Integral(r**2*exp(-r)*sin(r**2/2) +
                r*exp(-r)*cos(r**2/2), r) + exp(r)*cos(r**2/2)*Integral(r**2*exp(-r)*cos(r**2/2) - r*exp(-r)*sin(r**2/2), r)),
            Eq(g(r), C1*exp(r)*cos(r**2/2) - C2*exp(r)*sin(r**2/2) - exp(r)*sin(r**2/2)*Integral(r**2*exp(-r)*cos(r**2/2) -
                r*exp(-r)*sin(r**2/2), r) + exp(r)*cos(r**2/2)*Integral(r**2*exp(-r)*sin(r**2/2) + r*exp(-r)*cos(r**2/2), r))]
    # 断言解是正确的，并且检查解是否满足微分方程组
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    # 第二组微分方程
    eqs2 = [Eq(diff(f(r), r), f(r) + r*g(r) + r), Eq(diff(g(r), r), -r*f(r) + g(r) + log(r))]
    # 解第二组微分方程，这里的 dsolve_system 有可能导致集成过程 hang，所以加了 XXX 注释
    sol2 = [Eq(f(r), C1*exp(r)*sin(r**2/2) + C2*exp(r)*cos(r**2/2) + exp(r)*sin(r**2/2)*Integral(r*exp(-r)*sin(r**2/2) +
                exp(-r)*log(r)*cos(r**2/2), r) + exp(r)*cos(r**2/2)*Integral(r*exp(-r)*cos(r**2/2) - exp(-r)*log(r)*sin(
                r**2/2), r)),
            Eq(g(r), C1*exp(r)*cos(r**2/2) - C2*exp(r)*sin(r**2/2) - exp(r)*sin(r**2/2)*Integral(r*exp(-r)*cos(r**2/2) -
                exp(-r)*log(r)*sin(r**2/2), r) + exp(r)*cos(r**2/2)*Integral(r*exp(-r)*sin(r**2/2) + exp(-r)*log(r)*cos(
                r**2/2), r))]
    # 断言解是正确的，并且检查解是否满足微分方程组
    assert dsolve_system(eqs2, simplify=False, doit=False) == [sol2]
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    # 第三组微分方程
    eqs3 = [Eq(Derivative(f(x), x), x*(f(x) + g(x) + h(x)) + x),
            Eq(Derivative(g(x), x), x*(f(x) + g(x) + h(x)) + x),
            Eq(Derivative(h(x), x), x*(f(x) + g(x) + h(x)) + 1)]
    # 解第三组微分方程
    sol3 = [Eq(f(x), C1*Rational(-1, 3) + C2*Rational(-1, 3) + C3*Rational(2, 3) + x**2/6 + x*Rational(-1, 3) +
             (C1/3 + C2/3 + C3/3)*exp(x**2*Rational(3, 2)) +
             sqrt(6)*sqrt(pi)*erf(sqrt(6)*x/2)*exp(x**2*Rational(3, 2))/18 + Rational(-2, 9)),
            Eq(g(x), C1*Rational(2, 3) + C2*Rational(-1, 3) + C3*Rational(-1, 3) + x**2/6 + x*Rational(-1, 3) +
             (C1/3 + C2/3 + C3/3)*exp(x**2*Rational(3, 2)) +
             sqrt(6)*sqrt(pi)*erf(sqrt(6)*x/2)*exp(x**2*Rational(3, 2))/18 + Rational(-2, 9)),
            Eq(h(x), C1*Rational(-1, 3) + C2*Rational(2, 3) + C3*Rational(-1, 3) + x**2*Rational(-1, 3) +
             x*Rational(2, 3) + (C1/3 + C2/3 + C3/3)*exp(x**2*Rational(3, 2)) +
             sqrt(6)*sqrt(pi)*erf(sqrt(6)*x/2)*exp(x**2*Rational(3, 2))/18 + Rational(-2, 9))]
    # 断言解是正确的，并且检查解是否满足微分方程组
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0, 0])

    # 第四组微分方程，未完整解析
    eqs4 = [Eq(Derivative(f(x), x), x*(f(x) + g(x) + h(x)) + sin(x)),
            Eq(Derivative(g(x), x), x*(f(x) + g(x) + h(x)) + sin(x)),
            Eq(Derivative(h(x), x), x*(f(x) + g(x) + h(x)) + sin(x))]
    sol4 = [Eq(f(x), C1*Rational(-1, 3) + C2*Rational(-1, 3) + C3*Rational(2, 3) + (C1/3 + C2/3 +
             C3/3)*exp(x**2*Rational(3, 2)) + Integral(sin(x)*exp(x**2*Rational(-3, 2)), x)*exp(x**2*Rational(3,
             2))),
            Eq(g(x), C1*Rational(2, 3) + C2*Rational(-1, 3) + C3*Rational(-1, 3) + (C1/3 + C2/3 +
             C3/3)*exp(x**2*Rational(3, 2)) + Integral(sin(x)*exp(x**2*Rational(-3, 2)), x)*exp(x**2*Rational(3,
             2))),
            Eq(h(x), C1*Rational(-1, 3) + C2*Rational(2, 3) + C3*Rational(-1, 3) + (C1/3 + C2/3 +
             C3/3)*exp(x**2*Rational(3, 2)) + Integral(sin(x)*exp(x**2*Rational(-3, 2)), x)*exp(x**2*Rational(3,
             2)))]
    # 定义ODE系统的解sol4，包含多个方程，每个方程表示为关于f(x), g(x), h(x)的表达式
    assert dsolve(eqs4) == sol4
    # 断言dsolve函数求解eqs4得到的结果与预期的sol4相等

    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0])
    # 断言checksysodesol函数检查eqs4和sol4是一个正确的ODE系统解，返回值为True，误差列表为[0, 0, 0]

    eqs5 = [Eq(Derivative(f(x), x), x*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(g(x), x), x*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(h(x), x), x*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(k(x), x), x*(f(x) + g(x) + h(x) + k(x) + 1))]
    # 定义ODE系统eqs5，包含多个方程，每个方程表示为关于f(x), g(x), h(x), k(x)的导数的表达式

    sol5 = [Eq(f(x), C1*Rational(-1, 4) + C2*Rational(-1, 4) + C3*Rational(-1, 4) + C4*Rational(3, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(2*x**2) + Rational(-1, 4)),
            Eq(g(x), C1*Rational(3, 4) + C2*Rational(-1, 4) + C3*Rational(-1, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(2*x**2) + Rational(-1, 4)),
            Eq(h(x), C1*Rational(-1, 4) + C2*Rational(3, 4) + C3*Rational(-1, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(2*x**2) + Rational(-1, 4)),
            Eq(k(x), C1*Rational(-1, 4) + C2*Rational(-1, 4) + C3*Rational(3, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(2*x**2) + Rational(-1, 4))]
    # 定义ODE系统的解sol5，包含多个方程，每个方程表示为关于f(x), g(x), h(x), k(x)的表达式

    assert dsolve(eqs5) == sol5
    # 断言dsolve函数求解eqs5得到的结果与预期的sol5相等

    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0, 0])
    # 断言checksysodesol函数检查eqs5和sol5是一个正确的ODE系统解，返回值为True，误差列表为[0, 0, 0, 0]

    eqs6 = [Eq(Derivative(f(x), x), x**2*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(g(x), x), x**2*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(h(x), x), x**2*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(k(x), x), x**2*(f(x) + g(x) + h(x) + k(x) + 1))]
    # 定义ODE系统eqs6，包含多个方程，每个方程表示为关于f(x), g(x), h(x), k(x)的二阶导数的表达式

    sol6 = [Eq(f(x), C1*Rational(-1, 4) + C2*Rational(-1, 4) + C3*Rational(-1, 4) + C4*Rational(3, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(x**3*Rational(4, 3)) + Rational(-1, 4)),
            Eq(g(x), C1*Rational(3, 4) + C2*Rational(-1, 4) + C3*Rational(-1, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(x**3*Rational(4, 3)) + Rational(-1, 4)),
            Eq(h(x), C1*Rational(-1, 4) + C2*Rational(3, 4) + C3*Rational(-1, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(x**3*Rational(4, 3)) + Rational(-1, 4)),
            Eq(k(x), C1*Rational(-1, 4) + C2*Rational(-1, 4) + C3*Rational(3, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(x**3*Rational(4, 3)) + Rational(-1, 4))]
    # 定义ODE系统的解sol6，包含多个方程，每个方程表示为关于f(x), g(x), h(x), k(x)的表达式

    assert dsolve(eqs6) == sol6
    # 断言dsolve函数求解eqs6得到的结果与预期的sol6相等
    # 检查函数 `checksysodesol` 对于给定的方程组 `eqs6` 和解 `sol6` 是否返回预期的结果
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0, 0])
    
    # 定义方程组 `eqs7`，包括三个微分方程，每个方程的右侧为复合函数的导数
    eqs7 = [Eq(Derivative(f(x), x), (f(x) + g(x) + h(x))*log(x) + sin(x)),
            Eq(Derivative(g(x), x), (f(x) + g(x) + h(x))*log(x) + sin(x)),
            Eq(Derivative(h(x), x), (f(x) + g(x) + h(x))*log(x) + sin(x))]
    
    # 定义解 `sol7`，包含三个函数 `f(x)`, `g(x)`, `h(x)` 的表达式
    sol7 = [Eq(f(x), -C1/3 - C2/3 + 2*C3/3 + (C1/3 + C2/3 + C3/3)*exp(x*(3*log(x) - 3)) + exp(x*(3*log(x) - 3)) * Integral(exp(3*x)*exp(-3*x*log(x))*sin(x), x)),
            Eq(g(x), 2*C1/3 - C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(x*(3*log(x) - 3)) + exp(x*(3*log(x) - 3)) * Integral(exp(3*x)*exp(-3*x*log(x))*sin(x), x)),
            Eq(h(x), -C1/3 + 2*C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(x*(3*log(x) - 3)) + exp(x*(3*log(x) - 3)) * Integral(exp(3*x)*exp(-3*x*log(x))*sin(x), x))]
    
    # 应用 dotprodsimp 来简化微分方程组的求解过程
    with dotprodsimp(True):
        # 检查 `dsolve` 函数是否能正确求解方程组 `eqs7`，期望结果为 `sol7`
        assert dsolve(eqs7, simplify=False, doit=False) == sol7
    
    # 再次检查函数 `checksysodesol` 对于方程组 `eqs7` 和解 `sol7` 是否返回预期的结果
    assert checksysodesol(eqs7, sol7) == (True, [0, 0, 0])
    
    # 定义方程组 `eqs8`，包括四个微分方程，每个方程的右侧为复合函数的导数
    eqs8 = [Eq(Derivative(f(x), x), (f(x) + g(x) + h(x) + k(x))*log(x) + sin(x)),
            Eq(Derivative(g(x), x), (f(x) + g(x) + h(x) + k(x))*log(x) + sin(x)),
            Eq(Derivative(h(x), x), (f(x) + g(x) + h(x) + k(x))*log(x) + sin(x)),
            Eq(Derivative(k(x), x), (f(x) + g(x) + h(x) + k(x))*log(x) + sin(x))]
    
    # 定义解 `sol8`，包含四个函数 `f(x)`, `g(x)`, `h(x)`, `k(x)` 的表达式
    sol8 = [Eq(f(x), -C1/4 - C2/4 - C3/4 + 3*C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(x*(4*log(x) - 4)) + exp(x*(4*log(x) - 4)) * Integral(exp(4*x)*exp(-4*x*log(x))*sin(x), x)),
            Eq(g(x), 3*C1/4 - C2/4 - C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(x*(4*log(x) - 4)) + exp(x*(4*log(x) - 4)) * Integral(exp(4*x)*exp(-4*x*log(x))*sin(x), x)),
            Eq(h(x), -C1/4 + 3*C2/4 - C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(x*(4*log(x) - 4)) + exp(x*(4*log(x) - 4)) * Integral(exp(4*x)*exp(-4*x*log(x))*sin(x), x)),
            Eq(k(x), -C1/4 - C2/4 + 3*C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(x*(4*log(x) - 4)) + exp(x*(4*log(x) - 4)) * Integral(exp(4*x)*exp(-4*x*log(x))*sin(x), x))]
    
    # 应用 dotprodsimp 来简化微分方程组的求解过程
    with dotprodsimp(True):
        # 检查 `dsolve` 函数是否能正确求解方程组 `eqs8`，期望结果为 `sol8`
        assert dsolve(eqs8) == sol8
    
    # 再次检查函数 `checksysodesol` 对于方程组 `eqs8` 和解 `sol8` 是否返回预期的结果
    assert checksysodesol(eqs8, sol8) == (True, [0, 0, 0, 0])
def test_sysode_linear_neq_order1_type5_type6():
    f, g = symbols("f g", cls=Function)  # 定义函数 f, g
    x, x_ = symbols("x x_")  # 定义符号 x, x_

    # Type 5
    eqs1 = [Eq(Derivative(f(x), x), (2*f(x) + g(x))/x), Eq(Derivative(g(x), x), (f(x) + 2*g(x))/x)]
    sol1 = [Eq(f(x), -C1*x + C2*x**3), Eq(g(x), C1*x + C2*x**3)]
    assert dsolve(eqs1) == sol1  # 使用 dsolve 求解微分方程组 eqs1，并验证解是否为 sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])  # 验证 sol1 是否满足微分方程组 eqs1 的解

    # Type 6
    eqs2 = [Eq(Derivative(f(x), x), (2*f(x) + g(x) + 1)/x),
            Eq(Derivative(g(x), x), (x + f(x) + 2*g(x))/x)]
    sol2 = [Eq(f(x), C2*x**3 - x*(C1 + Rational(1, 4)) + x*log(x)*Rational(-1, 2) + Rational(-2, 3)),
            Eq(g(x), C2*x**3 + x*log(x)/2 + x*(C1 + Rational(-1, 4)) + Rational(1, 3))]
    assert dsolve(eqs2) == sol2  # 使用 dsolve 求解微分方程组 eqs2，并验证解是否为 sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])  # 验证 sol2 是否满足微分方程组 eqs2 的解


def test_higher_order_to_first_order():
    f, g = symbols('f g', cls=Function)  # 定义函数 f, g
    x = symbols('x')  # 定义符号 x

    eqs1 = [Eq(Derivative(f(x), (x, 2)), 2*f(x) + g(x)),
            Eq(Derivative(g(x), (x, 2)), -f(x))]
    sol1 = [Eq(f(x), -C2*x*exp(-x) + C3*x*exp(x) - (C1 - C2)*exp(-x) + (C3 + C4)*exp(x)),
            Eq(g(x), C2*x*exp(-x) - C3*x*exp(x) + (C1 + C2)*exp(-x) + (C3 - C4)*exp(x))]
    assert dsolve(eqs1) == sol1  # 使用 dsolve 求解微分方程组 eqs1，并验证解是否为 sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])  # 验证 sol1 是否满足微分方程组 eqs1 的解

    eqs2 = [Eq(f(x).diff(x, 2), 0), Eq(g(x).diff(x, 2), f(x))]
    sol2 = [Eq(f(x), C1 + C2*x), Eq(g(x), C1*x**2/2 + C2*x**3/6 + C3 + C4*x)]
    assert dsolve(eqs2) == sol2  # 使用 dsolve 求解微分方程组 eqs2，并验证解是否为 sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])  # 验证 sol2 是否满足微分方程组 eqs2 的解

    eqs3 = [Eq(Derivative(f(x), (x, 2)), 2*f(x)),
            Eq(Derivative(g(x), (x, 2)), -f(x) + 2*g(x))]
    sol3 = [Eq(f(x), 4*C1*exp(-sqrt(2)*x) + 4*C2*exp(sqrt(2)*x)),
            Eq(g(x), sqrt(2)*C1*x*exp(-sqrt(2)*x) - sqrt(2)*C2*x*exp(sqrt(2)*x) + (C1 +
             sqrt(2)*C4)*exp(-sqrt(2)*x) + (C2 - sqrt(2)*C3)*exp(sqrt(2)*x))]
    assert dsolve(eqs3) == sol3  # 使用 dsolve 求解微分方程组 eqs3，并验证解是否为 sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])  # 验证 sol3 是否满足微分方程组 eqs3 的解

    eqs4 = [Eq(Derivative(f(x), (x, 2)), 2*f(x) + g(x)),
            Eq(Derivative(g(x), (x, 2)), 2*g(x))]
    sol4 = [Eq(f(x), C1*x*exp(sqrt(2)*x)/4 + C3*x*exp(-sqrt(2)*x)/4 + (C2/4 + sqrt(2)*C3/8)*exp(-sqrt(2)*x) -
             exp(sqrt(2)*x)*(sqrt(2)*C1/8 + C4*Rational(-1, 4))),
            Eq(g(x), sqrt(2)*C1*exp(sqrt(2)*x)/2 + sqrt(2)*C3*exp(-sqrt(2)*x)*Rational(-1, 2))]
    assert dsolve(eqs4) == sol4  # 使用 dsolve 求解微分方程组 eqs4，并验证解是否为 sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])  # 验证 sol4 是否满足微分方程组 eqs4 的解

    eqs5 = [Eq(f(x).diff(x, 2), f(x)), Eq(g(x).diff(x, 2), f(x))]
    sol5 = [Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), -C1*exp(-x) + C2*exp(x) + C3 + C4*x)]
    assert dsolve(eqs5) == sol5  # 使用 dsolve 求解微分方程组 eqs5，并验证解是否为 sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])  # 验证 sol5 是否满足微分方程组 eqs5 的解

    eqs6 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x)),
            Eq(Derivative(g(x), (x, 2)), -f(x) - g(x))]
    sol6 = [Eq(f(x), C1 + C2*x**2/2 + C2 + C4*x**3/6 + x*(C3 + C4)),
            Eq(g(x), -C1 + C2*x**2*Rational(-1, 2) - C3*x + C4*x**3*Rational(-1, 6))]
    assert dsolve(eqs6) == sol6  # 使用 dsolve 求解微分方程组 eqs6，并验证解是否为 sol6
    # 断言检查方程组eqs6的解是否正确
    assert checksysodesol(eqs6, sol6) == (True, [0, 0])

    # 定义方程组eqs7和其解sol7
    eqs7 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x) + 1),
            Eq(Derivative(g(x), (x, 2)), f(x) + g(x) + 1)]
    sol7 = [Eq(f(x), -C1 - C2*x + sqrt(2)*C3*exp(sqrt(2)*x)/2 + sqrt(2)*C4*exp(-sqrt(2)*x)*Rational(-1, 2) +
             Rational(-1, 2)),
            Eq(g(x), C1 + C2*x + sqrt(2)*C3*exp(sqrt(2)*x)/2 + sqrt(2)*C4*exp(-sqrt(2)*x)*Rational(-1, 2) +
             Rational(-1, 2))]
    # 断言使用dsolve求解eqs7得到的结果是否与sol7相等
    assert dsolve(eqs7) == sol7
    # 断言检查方程组eqs7的解是否正确
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])

    # 定义方程组eqs8和其解sol8
    eqs8 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x) + 1),
            Eq(Derivative(g(x), (x, 2)), -f(x) - g(x) + 1)]
    sol8 = [Eq(f(x), C1 + C2 + C4*x**3/6 + x**4/12 + x**2*(C2/2 + Rational(1, 2)) + x*(C3 + C4)),
            Eq(g(x), -C1 - C3*x + C4*x**3*Rational(-1, 6) + x**4*Rational(-1, 12) - x**2*(C2/2 + Rational(-1,
             2)))]
    # 断言使用dsolve求解eqs8得到的结果是否与sol8相等
    assert dsolve(eqs8) == sol8
    # 断言检查方程组eqs8的解是否正确
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])

    # 定义函数x(t)和y(t)作为符号函数
    x, y = symbols('x, y', cls=Function)
    t, l = symbols('t, l')

    # 定义方程组eqs10和其解sol10
    eqs10 = [Eq(Derivative(x(t), (t, 2)), 5*x(t) + 43*y(t)),
             Eq(Derivative(y(t), (t, 2)), x(t) + 9*y(t))]
    sol10 = [Eq(x(t), C1*(61 - 9*sqrt(47))*sqrt(sqrt(47) + 7)*exp(-t*sqrt(sqrt(47) + 7))/2 + C2*sqrt(7 -
              sqrt(47))*(61 + 9*sqrt(47))*exp(-t*sqrt(7 - sqrt(47)))/2 + C3*(61 - 9*sqrt(47))*sqrt(sqrt(47) +
              7)*exp(t*sqrt(sqrt(47) + 7))*Rational(-1, 2) + C4*sqrt(7 - sqrt(47))*(61 + 9*sqrt(47))*exp(t*sqrt(7
              - sqrt(47)))*Rational(-1, 2)),
             Eq(y(t), C1*(7 - sqrt(47))*sqrt(sqrt(47) + 7)*exp(-t*sqrt(sqrt(47) + 7))*Rational(-1, 2) + C2*sqrt(7
              - sqrt(47))*(sqrt(47) + 7)*exp(-t*sqrt(7 - sqrt(47)))*Rational(-1, 2) + C3*(7 -
              sqrt(47))*sqrt(sqrt(47) + 7)*exp(t*sqrt(sqrt(47) + 7))/2 + C4*sqrt(7 - sqrt(47))*(sqrt(47) +
              7)*exp(t*sqrt(7 - sqrt(47)))/2)]
    # 断言使用dsolve求解eqs10得到的结果是否与sol10相等
    assert dsolve(eqs10) == sol10
    # 断言检查方程组eqs10的解是否正确
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])

    # 定义方程组eqs11和其解sol11
    eqs11 = [Eq(7*x(t) + Derivative(x(t), (t, 2)) - 9*Derivative(y(t), t), 0),
             Eq(7*y(t) + 9*Derivative(x(t), t) + Derivative(y(t), (t, 2)), 0)]
    sol11 = [Eq(y(t), C1*(9 - sqrt(109))*sin(sqrt(2)*t*sqrt(9*sqrt(109) + 95)/2)/14 + C2*(9 -
              sqrt(109))*cos(sqrt(2)*t*sqrt(9*sqrt(109) + 95)/2)*Rational(-1, 14) + C3*(9 +
              sqrt(109))*sin(sqrt(2)*t*sqrt(95 - 9*sqrt(109))/2)/14 + C4*(9 + sqrt(109))*cos(sqrt(2)*t*sqrt(95 -
              9*sqrt(109))/2)*Rational(-1, 14)),
             Eq(x(t), C1*(9 - sqrt(109))*cos(sqrt(2)*t*sqrt(9*sqrt(109) + 95)/2)*Rational(-1, 14) + C2*(9 -
              sqrt(109))*sin(sqrt(2)*t*sqrt(9*sqrt(109) + 95)/2)*Rational(-1, 14) + C3*(9 +
              sqrt(109))*cos(sqrt(2)*t*sqrt(95 - 9*sqrt(109))/2)/14 + C4*(9 + sqrt(109))*sin(sqrt(2)*t*sqrt(95 -
              9*sqrt(109))/2)/14)]
    # 断言使用dsolve求解eqs11得到的结果是否与sol11相等
    assert dsolve(eqs11) == sol11
    # 断言检查方程组eqs11的解是否正确
    assert checksysodesol(eqs11, sol11) == (True, [0, 0])

    # Euler Systems
    # 定义一个包含 Euler 系统求解器的示例，带有非齐次项。
    eqs13 = [
        Eq(Derivative(f(t), (t, 2)), Derivative(f(t), t)/t + f(t)/t**2 + g(t)/t**2),
        Eq(Derivative(g(t), (t, 2)), g(t)/t**2)
    ]
    # 计算 Euler 系统的解析解
    sol13 = [
        Eq(f(t), C1*(sqrt(5) + 3)*Rational(-1, 2)*t**(Rational(1, 2) + sqrt(5)*Rational(-1, 2))
                 + C2*t**(Rational(1, 2) + sqrt(5)/2)*(3 - sqrt(5))*Rational(-1, 2)
                 - C3*t**(1 - sqrt(2))*(1 + sqrt(2))
                 - C4*t**(1 + sqrt(2))*(1 - sqrt(2))),
        Eq(g(t), C1*(1 + sqrt(5))*Rational(-1, 2)*t**(Rational(1, 2) + sqrt(5)*Rational(-1, 2))
                 + C2*t**(Rational(1, 2) + sqrt(5)/2)*(1 - sqrt(5))*Rational(-1, 2)
                 )
    ]
    # 断言解析解是否符合求解结果
    assert dsolve(eqs13) == sol13
    # 检查解析解是否满足 Euler 系统的解方程
    assert checksysodesol(eqs13, sol13) == (True, [0, 0])
    
    # 分别使用 dsolve 解决系统方程组
    eqs14 = [
        Eq(Derivative(f(t), (t, 2)), t*f(t)),
        Eq(Derivative(g(t), (t, 2)), t*g(t))
    ]
    # 计算 Euler 系统的解析解
    sol14 = [
        Eq(f(t), C1*airyai(t) + C2*airybi(t)),
        Eq(g(t), C3*airyai(t) + C4*airybi(t))
    ]
    # 断言解析解是否符合求解结果
    assert dsolve(eqs14) == sol14
    # 检查解析解是否满足 Euler 系统的解方程
    assert checksysodesol(eqs14, sol14) == (True, [0, 0])
    
    # 定义另一个 Euler 系统的示例方程组
    eqs15 = [
        Eq(Derivative(x(t), (t, 2)), t*(4*Derivative(x(t), t) + 8*Derivative(y(t), t))),
        Eq(Derivative(y(t), (t, 2)), t*(12*Derivative(x(t), t) - 6*Derivative(y(t), t)))
    ]
    # 计算 Euler 系统的解析解
    sol15 = [
        Eq(x(t), C1 - erf(sqrt(6)*t)*(sqrt(6)*sqrt(pi)*C2/33 + sqrt(6)*sqrt(pi)*C3*Rational(-1, 44))
                 + erfi(sqrt(5)*t)*(sqrt(5)*sqrt(pi)*C2*Rational(2, 55) + sqrt(5)*sqrt(pi)*C3*Rational(4, 55))),
        Eq(y(t), C4 + erf(sqrt(6)*t)*(sqrt(6)*sqrt(pi)*C2*Rational(2, 33) + sqrt(6)*sqrt(pi)*C3*Rational(-1, 22))
                 + erfi(sqrt(5)*t)*(sqrt(5)*sqrt(pi)*C2*Rational(3, 110) + sqrt(5)*sqrt(pi)*C3*Rational(3, 55)))
    ]
    # 断言解析解是否符合求解结果
    assert dsolve(eqs15) == sol15
    # 检查解析解是否满足 Euler 系统的解方程
    assert checksysodesol(eqs15, sol15) == (True, [0, 0])
@slow
# 定义一个测试函数，测试将高阶微分方程组转换为一阶微分方程组的功能
def test_higher_order_to_first_order_9():
    # 定义符号变量 f(x) 和 g(x) 为函数
    f, g = symbols('f g', cls=Function)
    # 定义符号变量 x
    x = symbols('x')

    # 定义第一个微分方程组
    eqs9 = [f(x) + g(x) - 2*exp(I*x) + 2*Derivative(f(x), x) + Derivative(f(x), (x, 2)),
            f(x) + g(x) - 2*exp(I*x) + 2*Derivative(g(x), x) + Derivative(g(x), (x, 2))]
    # 求解微分方程组得到的解
    sol9 =  [Eq(f(x), -C1 + C4*exp(-2*x)/2 - (C2/2 - C3/2)*exp(-x)*cos(x)
                    + (C2/2 + C3/2)*exp(-x)*sin(x) + 2*((1 - 2*I)*exp(I*x)*sin(x)**2/5)
                    + 2*((1 - 2*I)*exp(I*x)*cos(x)**2/5)),
            Eq(g(x), C1 - C4*exp(-2*x)/2 - (C2/2 - C3/2)*exp(-x)*cos(x)
                    + (C2/2 + C3/2)*exp(-x)*sin(x) + 2*((1 - 2*I)*exp(I*x)*sin(x)**2/5)
                    + 2*((1 - 2*I)*exp(I*x)*cos(x)**2/5))]
    # 断言求解结果与预期解相等
    assert dsolve(eqs9) == sol9
    # 断言检查微分方程组的解是否满足方程组
    assert checksysodesol(eqs9, sol9) == (True, [0, 0])


# 定义另一个测试函数，测试将高阶微分方程组转换为一阶微分方程组的功能
def test_higher_order_to_first_order_12():
    # 定义符号变量 f(x) 和 g(x) 为函数
    f, g = symbols('f g', cls=Function)
    # 定义符号变量 x
    x = symbols('x')

    # 定义第二个微分方程组
    eqs12 = [Eq(4*x(t) + Derivative(x(t), (t, 2)) + 8*Derivative(y(t), t), 0),
             Eq(4*y(t) - 8*Derivative(x(t), t) + Derivative(y(t), (t, 2)), 0)]
    # 求解微分方程组得到的解
    sol12 = [Eq(y(t), C1*(2 - sqrt(5))*sin(2*t*sqrt(4*sqrt(5) + 9))*Rational(-1, 2) + C2*(2 -
              sqrt(5))*cos(2*t*sqrt(4*sqrt(5) + 9))/2 + C3*(2 + sqrt(5))*sin(2*t*sqrt(9 - 4*sqrt(5)))*Rational(-1,
              2) + C4*(2 + sqrt(5))*cos(2*t*sqrt(9 - 4*sqrt(5)))/2),
             Eq(x(t), C1*(2 - sqrt(5))*cos(2*t*sqrt(4*sqrt(5) + 9))*Rational(-1, 2) + C2*(2 -
              sqrt(5))*sin(2*t*sqrt(4*sqrt(5) + 9))*Rational(-1, 2) + C3*(2 + sqrt(5))*cos(2*t*sqrt(9 -
              4*sqrt(5)))/2 + C4*(2 + sqrt(5))*sin(2*t*sqrt(9 - 4*sqrt(5)))/2)]
    # 断言求解结果与预期解相等
    assert dsolve(eqs12) == sol12
    # 断言检查微分方程组的解是否满足方程组
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])


# 定义另一个测试函数，测试将二阶微分方程组转换为一阶微分方程组的功能
def test_second_order_to_first_order_2():
    # 定义符号变量 f(x) 和 g(x) 为函数
    f, g = symbols("f g", cls=Function)
    # 定义符号变量 x, t, x_, t_, d, a, m
    x, t, x_, t_, d, a, m = symbols("x t x_ t_ d a m")

    # 定义第三个微分方程组
    eqs2 = [Eq(f(x).diff(x, 2), 2*(x*g(x).diff(x) - g(x))),
            Eq(g(x).diff(x, 2),-2*(x*f(x).diff(x) - f(x)))]
    # 求解微分方程组得到的解
    sol2 = [Eq(f(x), C1*x + x*Integral(C2*exp(-x_)*exp(I*exp(2*x_))/2 + C2*exp(-x_)*exp(-I*exp(2*x_))/2 -
                I*C3*exp(-x_)*exp(I*exp(2*x_))/2 + I*C3*exp(-x_)*exp(-I*exp(2*x_))/2, (x_, log(x)))),
            Eq(g(x), C4*x + x*Integral(I*C2*exp(-x_)*exp(I*exp(2*x_))/2 - I*C2*exp(-x_)*exp(-I*exp(2*x_))/2 +
                C3*exp(-x_)*exp(I*exp(2*x_))/2 + C3*exp(-x_)*exp(-I*exp(2*x_))/2, (x_, log(x))))]
    # XXX: dsolve 在这个积分过程中出现挂起现象
    assert dsolve_system(eqs2, simplify=False, doit=False) == [sol2]
    # 断言检查微分方程组的解是否满足方程组
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    # 定义另一个微分方程组
    eqs3 = (Eq(diff(f(t),t,t), 9*t*diff(g(t),t)-9*g(t)), Eq(diff(g(t),t,t),7*t*diff(f(t),t)-7*f(t)))
    sol3 = [Eq(f(t), C1*t + t*Integral(C2*exp(-t_)*exp(3*sqrt(7)*exp(2*t_)/2)/2 + C2*exp(-t_)*
                exp(-3*sqrt(7)*exp(2*t_)/2)/2 + 3*sqrt(7)*C3*exp(-t_)*exp(3*sqrt(7)*exp(2*t_)/2)/14 -
                3*sqrt(7)*C3*exp(-t_)*exp(-3*sqrt(7)*exp(2*t_)/2)/14, (t_, log(t)))),
            Eq(g(t), C4*t + t*Integral(sqrt(7)*C2*exp(-t_)*exp(3*sqrt(7)*exp(2*t_)/2)/6 - sqrt(7)*C2*exp(-t_)*
                exp(-3*sqrt(7)*exp(2*t_)/2)/6 + C3*exp(-t_)*exp(3*sqrt(7)*exp(2*t_)/2)/2 + C3*exp(-t_)*exp(-3*sqrt(7)*
                exp(2*t_)/2)/2, (t_, log(t))))]
    # XXX: dsolve hangs for this in integration
    # 注释：在积分计算时，dsolve 在这里陷入无法终止的状态

    assert dsolve_system(eqs3, simplify=False, doit=False) == [sol3]
    # 注释：验证使用指定方程组求解的结果是否等于预期的解 sol3

    assert checksysodesol(eqs3, sol3) == (True, [0, 0])
    # 注释：验证解 sol3 是否满足方程组 eqs3 的所有微分方程

    # Regression Test case for sympy#19238
    # https://github.com/sympy/sympy/issues/19238
    # Note: When the doit method is removed, these particular types of systems
    # can be divided first so that we have lesser number of big matrices.
    # 注释：这是对 sympy#19238 的回归测试案例，讨论在去除 doit 方法时，特定类型的系统可以首先分解，以减少大矩阵的数量

    eqs5 = [Eq(Derivative(g(t), (t, 2)), a*m),
            Eq(Derivative(f(t), (t, 2)), 0)]
    sol5 = [Eq(g(t), C1 + C2*t + a*m*t**2/2),
            Eq(f(t), C3 + C4*t)]
    assert dsolve(eqs5) == sol5
    # 注释：验证使用指定方程组求解的结果是否等于预期的解 sol5

    assert checksysodesol(eqs5, sol5) == (True, [0, 0])
    # 注释：验证解 sol5 是否满足方程组 eqs5 的所有微分方程

    # Type 2
    eqs6 = [Eq(Derivative(f(t), (t, 2)), f(t)/t**4),
            Eq(Derivative(g(t), (t, 2)), d*g(t)/t**4)]
    sol6 = [Eq(f(t), C1*sqrt(t**2)*exp(-1/t) - C2*sqrt(t**2)*exp(1/t)),
            Eq(g(t), C3*sqrt(t**2)*exp(-sqrt(d)/t)*d**Rational(-1, 2) -
             C4*sqrt(t**2)*exp(sqrt(d)/t)*d**Rational(-1, 2))]
    assert dsolve(eqs6) == sol6
    # 注释：验证使用指定方程组求解的结果是否等于预期的解 sol6

    assert checksysodesol(eqs6, sol6) == (True, [0, 0])
    # 注释：验证解 sol6 是否满足方程组 eqs6 的所有微分方程
@slow
def test_second_order_to_first_order_slow1():
    f, g = symbols("f g", cls=Function)  # 定义函数符号 f, g
    x, t, x_, t_, d, a, m = symbols("x t x_ t_ d a m")  # 定义符号变量 x, t, x_, t_, d, a, m

    # Type 1
    # 定义方程组 eqs1 和解 sol1
    eqs1 = [Eq(f(x).diff(x, 2), 2/x *(x*g(x).diff(x) - g(x))),
           Eq(g(x).diff(x, 2),-2/x *(x*f(x).diff(x) - f(x)))]
    sol1 = [Eq(f(x), C1*x + 2*C2*x*Ci(2*x) - C2*sin(2*x) - 2*C3*x*Si(2*x) - C3*cos(2*x)),
            Eq(g(x), -2*C2*x*Si(2*x) - C2*cos(2*x) - 2*C3*x*Ci(2*x) + C3*sin(2*x) + C4*x)]
    assert dsolve(eqs1) == sol1  # 断言解 eqs1 的微分方程组为 sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])  # 断言检查 eqs1 和 sol1 是否为解

def test_second_order_to_first_order_slow4():
    f, g = symbols("f g", cls=Function)  # 定义函数符号 f, g
    x, t, x_, t_, d, a, m = symbols("x t x_ t_ d a m")  # 定义符号变量 x, t, x_, t_, d, a, m

    # 定义方程组 eqs4 和解 sol4
    eqs4 = [Eq(Derivative(f(t), (t, 2)), t*sin(t)*Derivative(g(t), t) - g(t)*sin(t)),
            Eq(Derivative(g(t), (t, 2)), t*sin(t)*Derivative(f(t), t) - f(t)*sin(t))]
    sol4 = [Eq(f(t), C1*t + t*Integral(C2*exp(-t_)*exp(exp(t_)*cos(exp(t_)))*exp(-sin(exp(t_)))/2 +
                C2*exp(-t_)*exp(-exp(t_)*cos(exp(t_)))*exp(sin(exp(t_)))/2 - C3*exp(-t_)*exp(exp(t_)*cos(exp(t_)))*
                exp(-sin(exp(t_)))/2 +
                C3*exp(-t_)*exp(-exp(t_)*cos(exp(t_)))*exp(sin(exp(t_)))/2, (t_, log(t)))),
            Eq(g(t), C4*t + t*Integral(-C2*exp(-t_)*exp(exp(t_)*cos(exp(t_)))*exp(-sin(exp(t_)))/2 +
                C2*exp(-t_)*exp(-exp(t_)*cos(exp(t_)))*exp(sin(exp(t_)))/2 + C3*exp(-t_)*exp(exp(t_)*cos(exp(t_)))*
                exp(-sin(exp(t_)))/2 + C3*exp(-t_)*exp(-exp(t_)*cos(exp(t_)))*exp(sin(exp(t_)))/2, (t_, log(t))))]
    # XXX: dsolve hangs for this in integration
    assert dsolve_system(eqs4, simplify=False, doit=False) == [sol4]  # 断言解 eqs4 的系统微分方程为 sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])  # 断言检查 eqs4 和 sol4 是否为解

def test_component_division():
    f, g, h, k = symbols('f g h k', cls=Function)  # 定义函数符号 f, g, h, k
    x = symbols("x")  # 定义符号变量 x
    funcs = [f(x), g(x), h(x), k(x)]  # 函数列表

    # 定义方程组 eqs1 和解 sol1
    eqs1 = [Eq(Derivative(f(x), x), 2*f(x)),
            Eq(Derivative(g(x), x), f(x)),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), h(x)**4 + k(x))]
    sol1 = [Eq(f(x), 2*C1*exp(2*x)),
            Eq(g(x), C1*exp(2*x) + C2),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C3**4*exp(4*x)/3 + C4*exp(x))]
    assert dsolve(eqs1) == sol1  # 断言解 eqs1 的微分方程组为 sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0, 0, 0])  # 断言检查 eqs1 和 sol1 是否为解

    # 定义组件分割 components1, eqsdict1, graph1
    components1 = {((Eq(Derivative(f(x), x), 2*f(x)),), (Eq(Derivative(g(x), x), f(x)),)),
                   ((Eq(Derivative(h(x), x), h(x)),), (Eq(Derivative(k(x), x), h(x)**4 + k(x)),))}
    eqsdict1 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {h(x)}},
                {f(x): Eq(Derivative(f(x), x), 2*f(x)),
                g(x): Eq(Derivative(g(x), x), f(x)),
                h(x): Eq(Derivative(h(x), x), h(x)),
                k(x): Eq(Derivative(k(x), x), h(x)**4 + k(x))})
    graph1 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), h(x))}]
    assert {tuple(tuple(scc) for scc in wcc) for wcc in _component_division(eqs1, funcs, x)} == components1
    # 断言：检查 _eqs2dict 函数对 eqs1 和 funcs 的计算结果是否等于 eqsdict1
    assert _eqs2dict(eqs1, funcs) == eqsdict1
    
    # 断言：检查 _dict2graph 函数对 eqsdict1[0] 的结果是否与 graph1 相等
    assert [set(element) for element in _dict2graph(eqsdict1[0])] == graph1

    # 定义包含多个常微分方程的列表 eqs2
    eqs2 = [Eq(Derivative(f(x), x), 2*f(x)),
            Eq(Derivative(g(x), x), f(x)),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), f(x)**4 + k(x))]
    
    # 求解常微分方程组 eqs2 的解 sol2
    sol2 = [Eq(f(x), C1*exp(2*x)),
            Eq(g(x), C1*exp(2*x)/2 + C2),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C1**4*exp(8*x)/7 + C4*exp(x))]
    
    # 断言：检查 dsolve 函数对 eqs2 的结果是否等于 sol2
    assert dsolve(eqs2) == sol2
    
    # 断言：检查 checksysodesol 函数对 eqs2 和 sol2 的结果是否符合预期
    assert checksysodesol(eqs2, sol2) == (True, [0, 0, 0, 0])

    # 定义具有多个连通分量的集合 components2
    components2 = {frozenset([(Eq(Derivative(f(x), x), 2*f(x)),),
                    (Eq(Derivative(g(x), x), f(x)),),
                    (Eq(Derivative(k(x), x), f(x)**4 + k(x)),)]),
                   frozenset([(Eq(Derivative(h(x), x), h(x)),)])}
    
    # 构建常微分方程到其依赖关系集合的字典 eqsdict2
    eqsdict2 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {f(x)}},
                 {f(x): Eq(Derivative(f(x), x), 2*f(x)),
                  g(x): Eq(Derivative(g(x), x), f(x)),
                  h(x): Eq(Derivative(h(x), x), h(x)),
                  k(x): Eq(Derivative(k(x), x), f(x)**4 + k(x))})
    
    # 构建包含所有变量的图形结构列表 graph2
    graph2 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), f(x))}]
    
    # 断言：检查 _component_division 函数对 eqs2、funcs 和 x 的结果是否等于 components2
    assert {frozenset(tuple(scc) for scc in wcc) for wcc in _component_division(eqs2, funcs, x)} == components2
    
    # 断言：检查 _eqs2dict 函数对 eqs2 和 funcs 的计算结果是否等于 eqsdict2
    assert _eqs2dict(eqs2, funcs) == eqsdict2
    
    # 断言：检查 _dict2graph 函数对 eqsdict2[0] 的结果是否与 graph2 相等
    assert [set(element) for element in _dict2graph(eqsdict2[0])] == graph2

    # 定义具有不同形式常微分方程的列表 eqs3
    eqs3 = [Eq(Derivative(f(x), x), 2*f(x)),
            Eq(Derivative(g(x), x), x + f(x)),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), f(x)**4 + k(x))]
    
    # 求解常微分方程组 eqs3 的解 sol3
    sol3 = [Eq(f(x), C1*exp(2*x)),
            Eq(g(x), C1*exp(2*x)/2 + C2 + x**2/2),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C1**4*exp(8*x)/7 + C4*exp(x))]
    
    # 断言：检查 dsolve 函数对 eqs3 的结果是否等于 sol3
    assert dsolve(eqs3) == sol3
    
    # 断言：检查 checksysodesol 函数对 eqs3 和 sol3 的结果是否符合预期
    assert checksysodesol(eqs3, sol3) == (True, [0, 0, 0, 0])

    # 定义具有多个连通分量的集合 components3
    components3 = {frozenset([(Eq(Derivative(f(x), x), 2*f(x)),),
                    (Eq(Derivative(g(x), x), x + f(x)),),
                    (Eq(Derivative(k(x), x), f(x)**4 + k(x)),)]),
                    frozenset([(Eq(Derivative(h(x), x), h(x)),),])}
    
    # 构建常微分方程到其依赖关系集合的字典 eqsdict3
    eqsdict3 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {f(x)}},
                {f(x): Eq(Derivative(f(x), x), 2*f(x)),
                g(x): Eq(Derivative(g(x), x), x + f(x)),
                h(x): Eq(Derivative(h(x), x), h(x)),
                k(x): Eq(Derivative(k(x), x), f(x)**4 + k(x))})
    
    # 构建包含所有变量的图形结构列表 graph3
    graph3 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), f(x))}]
    
    # 断言：检查 _component_division 函数对 eqs3、funcs 和 x 的结果是否等于 components3
    assert {frozenset(tuple(scc) for scc in wcc) for wcc in _component_division(eqs3, funcs, x)} == components3

    # 注意：在默认选项修改后，取消注释，以便首先调用 dsolve 对单个ODE系统进行重排。这可以在 dsolve 的 doit 选项默认设置为 False 后完成。
    # 定义包含四个方程的列表，每个方程都是由 Derivative 对象的等式组成
    eqs4 = [Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),  # 第一个方程
            Eq(Derivative(g(x), x), f(x) + x*g(x) + x),  # 第二个方程
            Eq(Derivative(h(x), x), h(x)),  # 第三个方程
            Eq(Derivative(k(x), x), f(x)**4 + k(x))]  # 第四个方程

    # 定义包含四个解的列表，每个解都是由对应的函数表达式组成
    sol4 = [Eq(f(x), (C1/2 - sqrt(2)*C2/2 - sqrt(2)*Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 + x*exp(-x**2/2 +\
                sqrt(2)*x)/2, x)/2 + Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 - sqrt(2)*x*exp(-x**2/2 +\
                sqrt(2)*x)/2, x)/2)*exp(x**2/2 - sqrt(2)*x) + (C1/2 + sqrt(2)*C2/2 + sqrt(2)*Integral(x*exp(-x**2/2
                - sqrt(2)*x)/2 + x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2 + Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2
                - sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2)*exp(x**2/2 + sqrt(2)*x)),  # 第一个函数的解
            Eq(g(x), (-sqrt(2)*C1/4 + C2/2 + Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 + x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2 -\
                sqrt(2)*Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 - sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2,
                x)/4)*exp(x**2/2 - sqrt(2)*x) + (sqrt(2)*C1/4 + C2/2 + Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 +
                x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2 + sqrt(2)*Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 -
                sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2, x)/4)*exp(x**2/2 + sqrt(2)*x)),  # 第二个函数的解
            Eq(h(x), C3*exp(x)),  # 第三个函数的解
            Eq(k(x), C4*exp(x) + exp(x)*Integral((C1*exp(x**2/2 - sqrt(2)*x)/2 + C1*exp(x**2/2 + sqrt(2)*x)/2 -
                sqrt(2)*C2*exp(x**2/2 - sqrt(2)*x)/2 + sqrt(2)*C2*exp(x**2/2 + sqrt(2)*x)/2 - sqrt(2)*exp(x**2/2 -
                sqrt(2)*x)*Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 + x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2 + exp(x**2/2 -
                sqrt(2)*x)*Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 - sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2,
                x)/2 + sqrt(2)*exp(x**2/2 + sqrt(2)*x)*Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 + x*exp(-x**2/2 +
                sqrt(2)*x)/2, x)/2 + exp(x**2/2 + sqrt(2)*x)*Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 -
                sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2)**4*exp(-x), x))]  # 第四个函数的解

    # 定义包含各个函数之间依赖关系的集合
    components4 = {(frozenset([Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),  # 第一个组件
                    Eq(Derivative(g(x), x), x*g(x) + x + f(x))]),  # 第二个组件
                    frozenset([Eq(Derivative(k(x), x), f(x)**4 + k(x)),])),  # 第三个组件
                    (frozenset([Eq(Derivative(h(x), x), h(x)),]),)}  # 第四个组件

    # 定义将方程转换为字典形式的结果
    eqsdict4 = ({f(x): {g(x)}, g(x): {f(x)}, h(x): set(), k(x): {f(x)}},  # 第一个字典
                {f(x): Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),  # 第二个字典
                g(x): Eq(Derivative(g(x), x), x*g(x) + x + f(x)),  # 第三个字典
                h(x): Eq(Derivative(h(x), x), h(x)),  # 第四个字典
                k(x): Eq(Derivative(k(x), x), f(x)**4 + k(x))})  # 第五个字典

    # 定义由函数集合和函数之间依赖关系组成的图
    graph4 = [{f(x), g(x), h(x), k(x)}, {(f(x), g(x)), (g(x), f(x)), (k(x), f(x))}]

    # 断言语句，验证计算得到的组件与预期的组件集合相等
    assert {tuple(frozenset(scc) for scc in wcc) for wcc in _component_division(eqs4, funcs, x)} == components4

    # 断言语句，验证将方程转换为字典形式的函数与预期的字典相等
    assert _eqs2dict(eqs4, funcs) == eqsdict4

    # 断言语句，验证将字典形式的函数转换为图的节点集合与预期的图节点集合相等
    assert [set(element) for element in _dict2graph(eqsdict4[0])] == graph4
    # XXX: dsolve hangs in integration here:
    # 在这里，dsolve 函数在积分过程中出现挂起现象
    assert dsolve_system(eqs4, simplify=False, doit=False) == [sol4]
    # 断言检查解方程组的结果是否符合预期
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0, 0])

    # 定义一个包含四个微分方程的列表 eqs5
    eqs5 = [Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
            Eq(Derivative(g(x), x), x*g(x) + f(x)),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), f(x)**4 + k(x))]
    # 给定方程组的解 sol5
    sol5 = [Eq(f(x), (C1/2 - sqrt(2)*C2/2)*exp(x**2/2 - sqrt(2)*x) + (C1/2 + sqrt(2)*C2/2)*exp(x**2/2 + sqrt(2)*x)),
            Eq(g(x), (-sqrt(2)*C1/4 + C2/2)*exp(x**2/2 - sqrt(2)*x) + (sqrt(2)*C1/4 + C2/2)*exp(x**2/2 + sqrt(2)*x)),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C4*exp(x) + exp(x)*Integral((C1*exp(x**2/2 - sqrt(2)*x)/2 + C1*exp(x**2/2 + sqrt(2)*x)/2 -
                sqrt(2)*C2*exp(x**2/2 - sqrt(2)*x)/2 + sqrt(2)*C2*exp(x**2/2 + sqrt(2)*x)/2)**4*exp(-x), x))]
    
    # 根据微分方程的组成部分划分成不同的组件 components5
    components5 = {(frozenset([Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
                    Eq(Derivative(g(x), x), x*g(x) + f(x))]),
                    frozenset([Eq(Derivative(k(x), x), f(x)**4 + k(x)),])),
                    (frozenset([Eq(Derivative(h(x), x), h(x)),]),)}
    
    # 将微分方程映射为字典形式 eqsdict5
    eqsdict5 = ({f(x): {g(x)}, g(x): {f(x)}, h(x): set(), k(x): {f(x)}},
                {f(x): Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
                g(x): Eq(Derivative(g(x), x), x*g(x) + f(x)),
                h(x): Eq(Derivative(h(x), x), h(x)),
                k(x): Eq(Derivative(k(x), x), f(x)**4 + k(x))})
    
    # 将微分方程转换为图形式，表示其关系 graph5
    graph5 = [{f(x), g(x), h(x), k(x)}, {(f(x), g(x)), (g(x), f(x)), (k(x), f(x))}]
    
    # 断言检查方程组的弱联通分量是否与预期的组件 components5 匹配
    assert {tuple(frozenset(scc) for scc in wcc) for wcc in _component_division(eqs5, funcs, x)} == components5
    # 断言检查将方程组转换为字典形式的结果是否与预期的 eqsdict5 匹配
    assert _eqs2dict(eqs5, funcs) == eqsdict5
    # 断言检查将字典形式的方程组转换为图形式的结果是否与预期的 graph5 匹配
    assert [set(element) for element in _dict2graph(eqsdict5[0])] == graph5
    # XXX: dsolve hangs in integration here:
    # 在这里，dsolve 函数在积分过程中出现挂起现象
    assert dsolve_system(eqs5, simplify=False, doit=False) == [sol5]
    # 断言检查解方程组的结果是否符合预期
    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0, 0])
def test_linodesolve():
    t, x, a = symbols("t x a")
    f, g, h = symbols("f g h", cls=Function)

    # Testing the Errors
    raises(ValueError, lambda: linodesolve(1, t))
    raises(ValueError, lambda: linodesolve(a, t))

    # 创建一个非方阵矩阵 A1，并测试 linodesolve 函数是否会引发 NonSquareMatrixError 异常
    A1 = Matrix([[1, 2], [2, 4], [4, 6]])
    raises(NonSquareMatrixError, lambda: linodesolve(A1, t))

    # 创建一个非方阵矩阵 A2，并测试 linodesolve 函数是否会引发 NonSquareMatrixError 异常
    A2 = Matrix([[1, 2, 1], [3, 1, 2]])
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t))

    # Testing auto functionality
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t) + g(t).diff(t), g(t)), Eq(g(t).diff(t), f(t))]
    # 转换 ODE 到标准形式
    ceq = canonical_odes(eq, func, t)
    # 将线性方程组转换为矩阵形式，并获取 A0 矩阵及 b 向量
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0
    # 解线性常系数 ODE 并验证结果是否正确
    sol = [C1*(-Rational(1, 2) + sqrt(5)/2)*exp(t*(-Rational(1, 2) + sqrt(5)/2)) + C2*(-sqrt(5)/2 - Rational(1, 2))*
           exp(t*(-sqrt(5)/2 - Rational(1, 2))),
           C1*exp(t*(-Rational(1, 2) + sqrt(5)/2)) + C2*exp(t*(-sqrt(5)/2 - Rational(1, 2)))]
    assert constant_renumber(linodesolve(A, t), variables=Tuple(*eq).free_symbols) == sol

    # Testing the Errors
    raises(ValueError, lambda: linodesolve(1, t, b=Matrix([t+1])))
    raises(ValueError, lambda: linodesolve(a, t, b=Matrix([log(t) + sin(t)])))

    # 测试不支持的 b 参数类型引发的 ValueError 异常
    raises(ValueError, lambda: linodesolve(Matrix([7]), t, b=t**2))
    raises(ValueError, lambda: linodesolve(Matrix([a+10]), t, b=log(t)*cos(t)))

    # 测试不支持的 A 参数类型引发的 ValueError 异常
    raises(ValueError, lambda: linodesolve(7, t, b=t**2))
    raises(ValueError, lambda: linodesolve(a, t, b=log(t) + sin(t)))

    # 创建一个非方阵矩阵 A1 和 b1 向量，并测试 linodesolve 函数是否会引发 NonSquareMatrixError 异常
    A1 = Matrix([[1, 2], [2, 4], [4, 6]])
    b1 = Matrix([t, 1, t**2])
    raises(NonSquareMatrixError, lambda: linodesolve(A1, t, b=b1))

    # 创建一个非方阵矩阵 A2 和 b2 向量，并测试 linodesolve 函数是否会引发 NonSquareMatrixError 异常
    A2 = Matrix([[1, 2, 1], [3, 1, 2]])
    b2 = Matrix([t, t**2])
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t, b=b2))

    # 测试部分矩阵 A1[:2, :] 和 b1 引发的 ValueError 异常
    raises(ValueError, lambda: linodesolve(A1[:2, :], t, b=b1))
    raises(ValueError, lambda: linodesolve(A1[:2, :], t, b=b1[:1]))

    # DOIT check
    A1 = Matrix([[1, -1], [1, -1]])
    b1 = Matrix([15*t - 10, -15*t - 5])
    sol1 = [C1 + C2*t + C2 - 10*t**3 + 10*t**2 + t*(15*t**2 - 5*t) - 10*t,
            C1 + C2*t - 10*t**3 - 5*t**2 + t*(15*t**2 - 5*t) - 5*t]
    # 解线性常系数 ODE，应用 DOIT，验证结果是否正确
    assert constant_renumber(linodesolve(A1, t, b=b1, type="type2", doit=True),
                             variables=[t]) == sol1

    # Testing auto functionality
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t) + g(t).diff(t), g(t) + t), Eq(g(t).diff(t), f(t))]
    # 转换 ODE 到标准形式
    ceq = canonical_odes(eq, func, t)
    # 将线性方程组转换为矩阵形式，并获取 A0 矩阵及 b 向量
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0
    # 解常微分方程组的第一个测试示例
    sol = [-C1*exp(-t/2 + sqrt(5)*t/2)/2 + sqrt(5)*C1*exp(-t/2 + sqrt(5)*t/2)/2 - sqrt(5)*C2*exp(-sqrt(5)*t/2 -
          t/2)/2 - C2*exp(-sqrt(5)*t/2 - t/2)/2 - exp(-t/2 + sqrt(5)*t/2)*Integral(t*exp(-sqrt(5)*t/2 +
          t/2)/(-5 + sqrt(5)) - sqrt(5)*t*exp(-sqrt(5)*t/2 + t/2)/(-5 + sqrt(5)), t)/2 + sqrt(5)*exp(-t/2 +
          sqrt(5)*t/2)*Integral(t*exp(-sqrt(5)*t/2 + t/2)/(-5 + sqrt(5)) - sqrt(5)*t*exp(-sqrt(5)*t/2 +
          t/2)/(-5 + sqrt(5)), t)/2 - sqrt(5)*exp(-sqrt(5)*t/2 - t/2)*Integral(-sqrt(5)*t*exp(t/2 +
          sqrt(5)*t/2)/5, t)/2 - exp(-sqrt(5)*t/2 - t/2)*Integral(-sqrt(5)*t*exp(t/2 + sqrt(5)*t/2)/5, t)/2,
         C1*exp(-t/2 + sqrt(5)*t/2) + C2*exp(-sqrt(5)*t/2 - t/2) + exp(-t/2 +
          sqrt(5)*t/2)*Integral(t*exp(-sqrt(5)*t/2 + t/2)/(-5 + sqrt(5)) - sqrt(5)*t*exp(-sqrt(5)*t/2 +
          t/2)/(-5 + sqrt(5)), t) + exp(-sqrt(5)*t/2 -
              t/2)*Integral(-sqrt(5)*t*exp(t/2 + sqrt(5)*t/2)/5, t)]

    # 使用 assert 断言验证线性常微分方程组的解与预期解 sol 相符
    assert constant_renumber(linodesolve(A, t, b=b), variables=[t]) == sol

    # 解常微分方程组的第二个测试示例
    sol1 = [-C1*exp(-t/2 + sqrt(5)*t/2)/2 + sqrt(5)*C1*exp(-t/2 + sqrt(5)*t/2)/2 - sqrt(5)*C2*exp(-sqrt(5)*t/2
                - t/2)/2 - C2*exp(-sqrt(5)*t/2 - t/2)/2,
            C1*exp(-t/2 + sqrt(5)*t/2) + C2*exp(-sqrt(5)*t/2 - t/2)]

    # 使用 assert 断言验证线性常微分方程组的解与预期解 sol1 相符
    assert constant_renumber(linodesolve(A, t, type="type2"), variables=[t]) == sol1

    # 测试错误情况：假设非齐次项为 0
    raises(ValueError, lambda: linodesolve(t+10, t))
    raises(ValueError, lambda: linodesolve(a*t, t))

    # 创建矩阵 A1，并尝试获取其反导数 B1
    A1 = Matrix([[1, t], [-t, 1]])
    B1, _ = _is_commutative_anti_derivative(A1, t)

    # 测试错误情况：A1 不是方阵
    raises(NonSquareMatrixError, lambda: linodesolve(A1[:, :1], t, B=B1))
    raises(ValueError, lambda: linodesolve(A1, t, B=1))

    # 创建矩阵 A2，并尝试获取其反导数 B2
    A2 = Matrix([[t, t, t], [t, t, t], [t, t, t]])
    B2, _ = _is_commutative_anti_derivative(A2, t)

    # 测试错误情况：B2 的行数小于 A2 的行数
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t, B=B2[:2, :]))
    raises(ValueError, lambda: linodesolve(A2, t, B=2))

    # 测试错误情况：使用了不支持的解类型 "type31"
    raises(ValueError, lambda: linodesolve(A2, t, B=B2, type="type31"))

    # 测试错误情况：A1 和 A2 的 B 矩阵不匹配
    raises(ValueError, lambda: linodesolve(A1, t, B=B2))
    raises(ValueError, lambda: linodesolve(A2, t, B=B1))

    # 测试自动功能
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t), f(t) + t*g(t)), Eq(g(t).diff(t), -t*f(t) + g(t))]

    # 将常微分方程组转换为矩阵形式，并获取线性常微分方程组的系数矩阵 A
    ceq = canonical_odes(eq, func, t)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0

    # 预期解 sol2
    sol2 = [(C1/2 - I*C2/2)*exp(I*t**2/2 + t) + (C1/2 + I*C2/2)*exp(-I*t**2/2 + t),
           (-I*C1/2 + C2/2)*exp(-I*t**2/2 + t) + (I*C1/2 + C2/2)*exp(I*t**2/2 + t)]

    # 使用 assert 断言验证线性常微分方程组的解与预期解 sol2 相符
    assert constant_renumber(linodesolve(A, t), variables=Tuple(*eq).free_symbols) == sol2

    # 使用 assert 断言验证使用 "type3" 解类型时的结果与预期解 sol2 相符
    assert constant_renumber(linodesolve(A, t, type="type3"), variables=Tuple(*eq).free_symbols) == sol2

    # 创建矩阵 A1，并测试其不支持的情况
    A1 = Matrix([[t, 1], [t, -1]])
    raises(NotImplementedError, lambda: linodesolve(A1, t))

    # 测试错误情况：非矩阵 B 的形式
    raises(ValueError, lambda: linodesolve(t+10, t, b=Matrix([t+1])))
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(a*t, t, b=Matrix([log(t) + sin(t)])))
    
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(Matrix([7*t]), t, b=t**2))
    
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(Matrix([a + 10*log(t)]), t, b=log(t)*cos(t)))
    
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(7*t, t, b=t**2))
    
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(a*t**2, t, b=log(t) + sin(t)))
    
    # 创建矩阵 A1，包含特定的元素
    A1 = Matrix([[1, t], [-t, 1]])
    
    # 创建向量 b1，包含特定的元素
    b1 = Matrix([t, t ** 2])
    
    # 调用 _is_commutative_anti_derivative 函数计算 A1 的特定形式 B1
    B1, _ = _is_commutative_anti_derivative(A1, t)
    
    # 调用 linodesolve 函数测试函数抛出 NonSquareMatrixError 异常，使用 lambda 匿名函数包装调用
    raises(NonSquareMatrixError, lambda: linodesolve(A1[:, :1], t, b=b1))
    
    # 创建矩阵 A2，包含特定的元素
    A2 = Matrix([[t, t, t], [t, t, t], [t, t, t]])
    
    # 创建向量 b2，包含特定的元素
    b2 = Matrix([t, 1, t**2])
    
    # 调用 _is_commutative_anti_derivative 函数计算 A2 的特定形式 B2
    B2, _ = _is_commutative_anti_derivative(A2, t)
    
    # 调用 linodesolve 函数测试函数抛出 NonSquareMatrixError 异常，使用 lambda 匿名函数包装调用
    raises(NonSquareMatrixError, lambda: linodesolve(A2[:2, :], t, b=b2))
    
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(A1, t, b=b2))
    
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(A2, t, b=b1))
    
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(A1, t, b=b1, B=B2))
    
    # 调用 linodesolve 函数测试函数抛出 ValueError 异常，使用 lambda 匿名函数包装调用
    raises(ValueError, lambda: linodesolve(A2, t, b=b2, B=B1))
    
    # 进行自动功能测试
    func = [f(x), g(x), h(x)]
    eq = [Eq(f(x).diff(x), x*(f(x) + g(x) + h(x)) + x),
          Eq(g(x).diff(x), x*(f(x) + g(x) + h(x)) + x),
          Eq(h(x).diff(x), x*(f(x) + g(x) + h(x)) + 1)]
    
    # 将微分方程组转化为标准形式
    ceq = canonical_odes(eq, func, x)
    
    # 将线性微分方程转换为矩阵形式
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, x, 1)
    
    # 定义一些临时变量
    _x1 = exp(-3*x**2/2)
    _x2 = exp(3*x**2/2)
    _x3 = Integral(2*_x1*x/3 + _x1/3 + x/3 - Rational(1, 3), x)
    _x4 = 2*_x2*_x3/3
    _x5 = Integral(2*_x1*x/3 + _x1/3 - 2*x/3 + Rational(2, 3), x)
    
    # 计算解的表达式列表
    sol = [
        C1*_x2/3 - C1/3 + C2*_x2/3 - C2/3 + C3*_x2/3 + 2*C3/3 + _x2*_x5/3 + _x3/3 + _x4 - _x5/3,
        C1*_x2/3 + 2*C1/3 + C2*_x2/3 - C2/3 + C3*_x2/3 - C3/3 + _x2*_x5/3 + _x3/3 + _x4 - _x5/3,
        C1*_x2/3 - C1/3 + C2*_x2/3 + 2*C2/3 + C3*_x2/3 - C3/3 + _x2*_x5/3 - 2*_x3/3 + _x4 + 2*_x5/3,
    ]
    
    # 使用 constant_renumber 函数验证 linodesolve 的返回结果与 sol 是否一致
    assert constant_renumber(linodesolve(A, x, b=b), variables=Tuple(*eq).free_symbols) == sol
    
    # 使用 constant_renumber 函数验证 linodesolve 的返回结果与 sol 是否一致，采用 type="type4" 参数
    assert constant_renumber(linodesolve(A, x, b=b, type="type4"), variables=Tuple(*eq).free_symbols) == sol
    
    # 创建矩阵 A1，包含特定的元素
    A1 = Matrix([[t, 1], [t, -1]])
    
    # 调用 linodesolve 函数测试函数抛出 NotImplementedError 异常，使用 lambda 匿名函数包装调用
    raises(NotImplementedError, lambda: linodesolve(A1, t, b=b1))
    
    # 验证在没有传递非齐次项的情况下 linodesolve 返回正确的解
    sol1 = [-C1/3 - C2/3 + 2*C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2),
            2*C1/3 - C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2),
            -C1/3 + 2*C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2)]
    
    # 使用 constant_renumber 函数验证 linodesolve 的返回结果与 sol1 是否一致
    assert constant_renumber(linodesolve(A, x, type="type4", doit=True), variables=Tuple(*eq).free_symbols) == sol1
# 标记为慢速测试函数，测试一阶线性非齐次方程组
@slow
def test_linear_3eq_order1_type4_slow():
    # 定义符号变量 x, y, z 和 t，均为函数类
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')

    # 定义函数 f 和 g
    f = t ** 3 + log(t)
    g = t ** 2 + sin(t)

    # 构造包含三个方程的方程组 eq1
    eq1 = (Eq(diff(x(t), t), (4 * f + g) * x(t) - f * y(t) - 2 * f * z(t)),
           Eq(diff(y(t), t), 2 * f * x(t) + (f + g) * y(t) - 2 * f * z(t)),
           Eq(diff(z(t), t), 5 * f * x(t) + f * y(t) + (-3 * f + g) * z(t)))

    # 使用 dotprodsimp 函数对方程组进行简化求解
    with dotprodsimp(True):
        dsolve(eq1)


# 标记为慢速测试函数，测试一阶线性非齐次方程组
@slow
def test_linear_neq_order1_type2_slow1():
    # 定义符号变量 i, r1, c1, r2, c2, t
    i, r1, c1, r2, c2, t = symbols('i, r1, c1, r2, c2, t')
    # 定义函数 x1 和 x2
    x1 = Function('x1')
    x2 = Function('x2')

    # 构造两个方程 eq1 和 eq2
    eq1 = r1 * c1 * Derivative(x1(t), t) + x1(t) - x2(t) - r1 * i
    eq2 = r2 * c1 * Derivative(x1(t), t) + r2 * c2 * Derivative(x2(t), t) + x2(t) - r2 * i
    eq = [eq1, eq2]

    # 使用 dsolve_system 函数求解方程组，并禁用简化和执行
    [sol] = dsolve_system(eq, simplify=False, doit=False)
    # 断言检查所得解是否满足原方程组
    assert checksysodesol(eq, sol) == (True, [0, 0])


# 标记为过慢测试函数，检查洛伦兹方程组的线性一阶非齐次类型
@tooslow
def test_linear_new_order1_type2_de_lorentz_slow_check():
    # 定义符号变量 m, q, t
    m = Symbol("m", real=True)
    q = Symbol("q", real=True)
    t = Symbol("t", real=True)

    # 定义实数符号变量 e1, e2, e3, b1, b2, b3 和 v1, v2, v3
    e1, e2, e3 = symbols("e1:4", real=True)
    b1, b2, b3 = symbols("b1:4", real=True)
    v1, v2, v3 = symbols("v1:4", cls=Function, real=True)

    # 构造洛伦兹方程组的三个方程 eqs
    eqs = [
        -e1 * q + m * Derivative(v1(t), t) - q * (-b2 * v3(t) + b3 * v2(t)),
        -e2 * q + m * Derivative(v2(t), t) - q * (b1 * v3(t) - b3 * v1(t)),
        -e3 * q + m * Derivative(v3(t), t) - q * (-b1 * v2(t) + b2 * v1(t))
    ]
    # 使用 dsolve 函数求解方程组
    sol = dsolve(eqs)
    # 断言检查所得解是否满足原方程组
    assert checksysodesol(eqs, sol) == (True, [0, 0, 0])


# 标记为慢速测试函数，检查一阶线性非齐次类型方程组的解
@slow
def test_linear_neq_order1_type2_slow_check():
    # 定义符号变量 RC, t, C, Vs, L, R1, V0, I0
    RC, t, C, Vs, L, R1, V0, I0 = symbols("RC t C Vs L R1 V0 I0")
    # 定义函数 V 和 I
    V = Function("V")
    I = Function("I")
    # 构造包含两个方程的方程组 system
    system = [
        Eq(V(t).diff(t), -1/RC * V(t) + I(t)/C),
        Eq(I(t).diff(t), -R1/L * I(t) - 1/L * V(t) + Vs/L)
    ]
    # 使用 dsolve_system 函数求解方程组，并禁用简化和执行
    [sol] = dsolve_system(system, simplify=False, doit=False)

    # 断言检查所得解是否满足原方程组
    assert checksysodesol(system, sol) == (True, [0, 0])


# 内部函数，计算长格式的一阶线性非齐次方程组的解
def _linear_3eq_order1_type4_long():
    # 定义符号变量 x, y, z 和 t，均为函数类
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')

    # 定义函数 f 和 g
    f = t ** 3 + log(t)
    g = t ** 2 + sin(t)

    # 构造包含三个方程的方程组 eq1
    eq1 = (Eq(diff(x(t), t), (4*f + g)*x(t) - f*y(t) - 2*f*z(t)),
           Eq(diff(y(t), t), 2*f*x(t) + (f + g)*y(t) - 2*f*z(t)),
           Eq(diff(z(t), t), 5*f*x(t) + f*y(t) + (-3*f + g)*z(t)))

    # 使用 dsolve 函数求解方程组
    dsolve_sol = dsolve(eq1)

    # 对每个解进行简化处理
    dsolve_sol1 = [_simpsol(sol) for sol in dsolve_sol]

    # 定义多个表达式 x_1 到 x_8，并进行计算
    x_1 = sqrt(-t**6 - 8*t**3*log(t) + 8*t**3 - 16*log(t)**2 + 32*log(t) - 16)
    x_2 = sqrt(3)
    x_3 = 8324372644*C1*x_1*x_2 + 4162186322*C2*x_1*x_2 - 8324372644*C3*x_1*x_2
    x_4 = 1 / (1903457163*t**3 + 3825881643*x_1*x_2 + 7613828652*log(t) - 7613828652)
    x_5 = exp(t**3/3 + t*x_1*x_2/4 - cos(t))
    x_6 = exp(t**3/3 - t*x_1*x_2/4 - cos(t))
    x_7 = exp(t**4/2 + t**3/3 + 2*t*log(t) - 2*t - cos(t))
    x_8 = 91238*C1*x_1*x_2 + 91238*C2*x_1*x_2 - 91238*C3*x_1*x_2
    # 计算 x_9 的值，使用给定的表达式
    x_9 = 1 / (66049*t**3 - 50629*x_1*x_2 + 264196*log(t) - 264196)
    
    # 计算 x_10 的值，根据给定的表达式计算
    x_10 = 50629 * C1 / 25189 + 37909*C2/25189 - 50629*C3/25189 - x_3*x_4
    
    # 计算 x_11 的值，根据给定的表达式计算
    x_11 = -50629*C1/25189 - 12720*C2/25189 + 50629*C3/25189 + x_3*x_4
    
    # 构造微分方程组的解 sol，使用给定的表达式组合成方程
    sol = [Eq(x(t), x_10*x_5 + x_11*x_6 + x_7*(C1 - C2)), Eq(y(t), x_10*x_5 + x_11*x_6), Eq(z(t), x_5*(
            -424*C1/257 - 167*C2/257 + 424*C3/257 - x_8*x_9) + x_6*(167*C1/257 + 424*C2/257 -
            167*C3/257 + x_8*x_9) + x_7*(C1 - C2))]
    
    # 断言 dsolve_sol1 等于预期的解 sol
    assert dsolve_sol1 == sol
    
    # 断言所得的微分方程组解符合原始方程 eq1，并返回检查结果和误差列表
    assert checksysodesol(eq1, dsolve_sol1) == (True, [0, 0, 0])
@slow
def test_neq_order1_type4_slow_check1():
    # 定义符号函数 f 和 g
    f, g = symbols("f g", cls=Function)
    # 定义符号变量 x
    x = symbols("x")

    # 定义微分方程组
    eqs = [Eq(diff(f(x), x), x*f(x) + x**2*g(x) + x),
           Eq(diff(g(x), x), 2*x**2*f(x) + (x + 3*x**2)*g(x) + 1)]
    # 求解微分方程组
    sol = dsolve(eqs)
    # 验证解是否符合原微分方程组
    assert checksysodesol(eqs, sol) == (True, [0, 0])


@slow
def test_neq_order1_type4_slow_check2():
    # 定义符号函数 f, g, h
    f, g, h = symbols("f, g, h", cls=Function)
    # 定义符号变量 x
    x = Symbol("x")

    # 定义微分方程组
    eqs = [
        Eq(Derivative(f(x), x), x*h(x) + f(x) + g(x) + 1),
        Eq(Derivative(g(x), x), x*g(x) + f(x) + h(x) + 10),
        Eq(Derivative(h(x), x), x*f(x) + x + g(x) + h(x))
    ]
    # 在 dotprodsimp 环境中求解微分方程组
    with dotprodsimp(True):
        sol = dsolve(eqs)
    # 验证解是否符合原微分方程组
    assert checksysodesol(eqs, sol) == (True, [0, 0, 0])


def _neq_order1_type4_slow3():
    # 定义符号函数 f 和 g
    f, g = symbols("f g", cls=Function)
    # 定义符号变量 x
    x = symbols("x")

    # 定义微分方程组
    eqs = [
        Eq(Derivative(f(x), x), x*f(x) + g(x) + sin(x)),
        Eq(Derivative(g(x), x), x**2 + x*g(x) - f(x))
    ]
    # 定义微分方程组的解
    sol = [
        Eq(f(x), (C1/2 - I*C2/2 - I*Integral(x**2*exp(-x**2/2 - I*x)/2 +
            x**2*exp(-x**2/2 + I*x)/2 + I*exp(-x**2/2 - I*x)*sin(x)/2 -
            I*exp(-x**2/2 + I*x)*sin(x)/2, x)/2 + Integral(-I*x**2*exp(-x**2/2
            - I*x)/2 + I*x**2*exp(-x**2/2 + I*x)/2 + exp(-x**2/2 -
            I*x)*sin(x)/2 + exp(-x**2/2 + I*x)*sin(x)/2, x)/2)*exp(x**2/2 +
            I*x) + (C1/2 + I*C2/2 + I*Integral(x**2*exp(-x**2/2 - I*x)/2 +
            x**2*exp(-x**2/2 + I*x)/2 + I*exp(-x**2/2 - I*x)*sin(x)/2 -
            I*exp(-x**2/2 + I*x)*sin(x)/2, x)/2 + Integral(-I*x**2*exp(-x**2/2
            - I*x)/2 + I*x**2*exp(-x**2/2 + I*x)/2 + exp(-x**2/2 -
            I*x)*sin(x)/2 + exp(-x**2/2 + I*x)*sin(x)/2, x)/2)*exp(x**2/2 -
            I*x)),
        Eq(g(x), (-I*C1/2 + C2/2 + Integral(x**2*exp(-x**2/2 - I*x)/2 +
            x**2*exp(-x**2/2 + I*x)/2 + I*exp(-x**2/2 - I*x)*sin(x)/2 -
            I*exp(-x**2/2 + I*x)*sin(x)/2, x)/2 -
            I*Integral(-I*x**2*exp(-x**2/2 - I*x)/2 + I*x**2*exp(-x**2/2 +
            I*x)/2 + exp(-x**2/2 - I*x)*sin(x)/2 + exp(-x**2/2 +
            I*x)*sin(x)/2, x)/2)*exp(x**2/2 - I*x) + (I*C1/2 + C2/2 +
            Integral(x**2*exp(-x**2/2 - I*x)/2 + x**2*exp(-x**2/2 + I*x)/2 +
            I*exp(-x**2/2 - I*x)*sin(x)/2 - I*exp(-x**2/2 + I*x)*sin(x)/2,
            x)/2 + I*Integral(-I*x**2*exp(-x**2/2 - I*x)/2 +
            I*x**2*exp(-x**2/2 + I*x)/2 + exp(-x**2/2 - I*x)*sin(x)/2 +
            exp(-x**2/2 + I*x)*sin(x)/2, x)/2)*exp(x**2/2 + I*x))
    ]

    return eqs, sol


def test_neq_order1_type4_slow3():
    # 获取微分方程组和解
    eqs, sol = _neq_order1_type4_slow3()
    # 求解微分方程组，不简化，不执行
    assert dsolve_system(eqs, simplify=False, doit=False) == [sol]
    # XXX: dsolve 在积分时出现错误:
    #     assert dsolve(eqs) == sol
    # https://github.com/sympy/sympy/issues/20155


@slow
def test_neq_order1_type4_slow_check3():
    # 获取微分方程组和解
    eqs, sol = _neq_order1_type4_slow3()
    # 验证解是否符合原微分方程组
    assert checksysodesol(eqs, sol) == (True, [0, 0])
def test_linear_3eq_order1_type4_long_dsolve_slow_xfail():
    eq, sol = _linear_3eq_order1_type4_long()
    # 获取线性方程组及其解
    dsolve_sol = dsolve(eq)
    # 求解微分方程组
    dsolve_sol1 = [_simpsol(sol) for sol in dsolve_sol]
    # 对每个解应用简化函数
    assert dsolve_sol1 == sol
    # 断言求解结果与预期解相等


@tooslow
def test_linear_3eq_order1_type4_long_dsolve_dotprodsimp():
    eq, sol = _linear_3eq_order1_type4_long()
    # 获取线性方程组及其解
    # XXX: 只有在使用 dotprodsimp 时才有效，参见 test_linear_3eq_order1_type4_long_dsolve_slow_xfail，这个测试太慢
    with dotprodsimp(True):
        dsolve_sol = dsolve(eq)
    # 求解微分方程组
    dsolve_sol1 = [_simpsol(sol) for sol in dsolve_sol]
    assert dsolve_sol1 == sol
    # 断言求解结果与预期解相等


@tooslow
def test_linear_3eq_order1_type4_long_check():
    eq, sol = _linear_3eq_order1_type4_long()
    # 获取线性方程组及其解
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])
    # 断言求解结果满足系统常微分方程的解


def test_dsolve_system():
    f, g = symbols("f g", cls=Function)
    x = symbols("x")
    # 定义方程和函数符号
    eqs = [Eq(f(x).diff(x), f(x) + g(x)), Eq(g(x).diff(x), f(x) + g(x))]
    funcs = [f(x), g(x)]
    # 设置方程组和函数列表

    sol = [[Eq(f(x), -C1 + C2*exp(2*x)), Eq(g(x), C1 + C2*exp(2*x))]]
    assert dsolve_system(eqs, funcs=funcs, t=x, doit=True) == sol
    # 断言求解多元常微分方程组的结果与预期解相等

    raises(ValueError, lambda: dsolve_system(1))
    raises(ValueError, lambda: dsolve_system(eqs, 1))
    raises(ValueError, lambda: dsolve_system(eqs, funcs, 1))
    raises(ValueError, lambda: dsolve_system(eqs, funcs[:1], x))
    # 断言对于错误的输入，会引发 ValueError

    eq = (Eq(f(x).diff(x), 12 * f(x) - 6 * g(x)), Eq(g(x).diff(x) ** 2, 11 * f(x) + 3 * g(x)))
    raises(NotImplementedError, lambda: dsolve_system(eq) == ([], []))
    # 断言对于未实现的方程类型，会引发 NotImplementedError

    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)]) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], t=x) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], t=x, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, t=x, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], ics={f(0): 1, g(0): 1}) == ([], []))
    # 断言对于带有不同参数的 dsolve_system 调用，会引发 NotImplementedError


def test_dsolve():

    f, g = symbols('f g', cls=Function)
    x, y = symbols('x y')

    eqs = [f(x).diff(x) - x, f(x).diff(x) + x]
    with raises(ValueError):
        dsolve(eqs)
    # 断言对于无效的微分方程组，会引发 ValueError

    eqs = [f(x, y).diff(x)]
    with raises(ValueError):
        dsolve(eqs)
    # 断言对于无效的微分方程组，会引发 ValueError

    eqs = [f(x, y).diff(x)+g(x).diff(x), g(x).diff(x)]
    with raises(ValueError):
        dsolve(eqs)
    # 断言对于无效的微分方程组，会引发 ValueError


@slow
def test_higher_order1_slow1():
    x, y = symbols("x y", cls=Function)
    t = symbols("t")

    eq = [
        Eq(diff(x(t),t,t), (log(t)+t**2)*diff(x(t),t)+(log(t)+t**2)*3*diff(y(t),t)),
        Eq(diff(y(t),t,t), (log(t)+t**2)*2*diff(x(t),t)+(log(t)+t**2)*9*diff(y(t),t))
    ]
    sol, = dsolve_system(eq, simplify=False, doit=False)
    # 定义高阶微分方程组

    # The solution is too long to write out explicitly and checkodesol is too
    # slow so we test for particular values of t:
    # 解的具体形式过长，而 checkodesol 过慢，因此我们对 t 的特定值进行测试
    # 对于方程集合中的每个方程 e 进行以下操作
    for e in eq:
        # 使用给定的解 sol 对 e 的左右两侧进行代入求值，并计算结果
        res = (e.lhs - e.rhs).subs({sol[0].lhs:sol[0].rhs, sol[1].lhs:sol[1].rhs})
        # 对于结果中包含的所有导数项 d，计算其不进行深度运算的值
        res = res.subs({d: d.doit(deep=False) for d in res.atoms(Derivative)})
        # 断言经过化简后，结果在 t=1 处的值为 0
        assert ratsimp(res.subs(t, 1)) == 0
# 定义一个测试函数，用于测试二阶非线性微分方程组的第二种类型的解法
def test_second_order_type2_slow1():
    # 定义符号变量 x(t), y(t), z(t) 为函数
    x, y, z = symbols('x, y, z', cls=Function)
    # 定义符号变量 t, l
    t, l = symbols('t, l')

    # 定义包含两个二阶微分方程的列表
    eqs1 = [
        Eq(Derivative(x(t), (t, 2)), t*(2*x(t) + y(t))),  # 第一个微分方程
        Eq(Derivative(y(t), (t, 2)), t*(-x(t) + 2*y(t)))  # 第二个微分方程
    ]
    
    # 定义这两个微分方程的解
    sol1 = [
        Eq(x(t), I*C1*airyai(t*(2 - I)**(S(1)/3)) + I*C2*airybi(t*(2 - I)**(S(1)/3)) - I*C3*airyai(t*(2 + I)**(S(1)/3)) - I*C4*airybi(t*(2 + I)**(S(1)/3))),  # x(t) 的解
        Eq(y(t), C1*airyai(t*(2 - I)**(S(1)/3)) + C2*airybi(t*(2 - I)**(S(1)/3)) + C3*airyai(t*(2 + I)**(S(1)/3)) + C4*airybi(t*(2 + I)**(S(1)/3)))  # y(t) 的解
    ]
    
    # 断言解微分方程组的结果
    assert dsolve(eqs1) == sol1
    # 断言验证解是否满足原微分方程组
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])


# 标记为太慢的测试用例
@tooslow
# 标记为预期失败的测试用例
@XFAIL
def test_nonlinear_3eq_order1_type1():
    # 定义符号变量 a, b, c
    a, b, c = symbols('a b c')

    # 定义包含三个一阶非线性微分方程的列表
    eqs = [
        a * f(x).diff(x) - (b - c) * g(x) * h(x),  # 第一个微分方程
        b * g(x).diff(x) - (c - a) * h(x) * f(x),  # 第二个微分方程
        c * h(x).diff(x) - (a - b) * f(x) * g(x)   # 第三个微分方程
    ]

    # 断言求解微分方程组引发 NotImplementedError 异常
    assert dsolve(eqs)  # NotImplementedError


# 标记为预期失败的测试用例
@XFAIL
def test_nonlinear_3eq_order1_type4():
    # 定义包含三个一阶非线性微分方程的列表
    eqs = [
        Eq(f(x).diff(x), (2*h(x)*g(x) - 3*g(x)*h(x))),  # 第一个微分方程
        Eq(g(x).diff(x), (4*f(x)*h(x) - 2*h(x)*f(x))),  # 第二个微分方程
        Eq(h(x).diff(x), (3*g(x)*f(x) - 4*f(x)*g(x)))   # 第三个微分方程
    ]
    
    # 尝试求解微分方程组引发 KeyError 异常
    dsolve(eqs)  # KeyError when matching
    # sol = ?
    # assert dsolve_sol == sol
    # assert checksysodesol(eqs, dsolve_sol) == (True, [0, 0, 0])


# 标记为太慢的测试用例
@tooslow
# 标记为预期失败的测试用例
@XFAIL
def test_nonlinear_3eq_order1_type3():
    # 定义包含三个一阶非线性微分方程的列表
    eqs = [
        Eq(f(x).diff(x), (2*f(x)**2 - 3)),               # 第一个微分方程
        Eq(g(x).diff(x), (4 - 2*h(x))),                  # 第二个微分方程
        Eq(h(x).diff(x), (3*h(x) - 4*f(x)**2))           # 第三个微分方程
    ]
    
    # 尝试求解微分方程组，不确定是否能完成
    dsolve(eqs)  # Not sure if this finishes...
    # sol = ?
    # assert dsolve_sol == sol
    # assert checksysodesol(eqs, dsolve_sol) == (True, [0, 0, 0])


# 标记为预期失败的测试用例
@XFAIL
def test_nonlinear_3eq_order1_type5():
    # 定义包含三个一阶非线性微分方程的列表
    eqs = [
        Eq(f(x).diff(x), f(x)*(2*f(x) - 3*g(x))),        # 第一个微分方程
        Eq(g(x).diff(x), g(x)*(4*g(x) - 2*h(x))),        # 第二个微分方程
        Eq(h(x).diff(x), h(x)*(3*h(x) - 4*f(x)))         # 第三个微分方程
    ]
    
    # 尝试求解微分方程组引发 KeyError 异常
    dsolve(eqs)  # KeyError
    # sol = ?
    # assert dsolve_sol == sol
    # assert checksysodesol(eqs, dsolve_sol) == (True, [0, 0, 0])


# 定义一个测试函数，用于测试二阶线性微分方程组的解法
def test_linear_2eq_order1():
    # 定义符号变量 x(t), y(t), z(t) 为函数
    x, y, z = symbols('x, y, z', cls=Function)
    # 定义整数符号变量 k, l, m, n
    k, l, m, n = symbols('k, l, m, n', Integer=True)
    # 定义符号变量 t
    t = Symbol('t')
    # 定义符号函数 x0(t), y0(t)
    x0, y0 = symbols('x0, y0', cls=Function)

    # 定义包含两个一阶线性微分方程的列表
    eq1 = (Eq(diff(x(t),t), x(t) + y(t) + 9), Eq(diff(y(t),t), 2*x(t) + 5*y(t) + 23))  # 第一组微分方程
    sol1 = [Eq(x(t), C1*exp(t*(sqrt(6) + 3)) + C2*exp(t*(-sqrt(6) + 3)) - Rational(22, 3)), \
    Eq(y(t), C1*(2 + sqrt(6))*exp(t*(sqrt(6) + 3)) + C2*(-sqrt(6) + 2)*exp(t*(-sqrt(6) + 3)) - Rational(5, 3))]  # 第一组微分方程的解
    # 断言验证解是否满足原微分方程组
    assert checksysodesol(eq1, sol1) == (True, [0, 0])

    # 定义第二组包含两个一阶线性微分方程的列表
    eq2 = (Eq(diff(x(t),t), x(t) + y(t) + 81), Eq(diff(y(t),t), -2*x(t) + y(t) + 23))  # 第二组微分方程
    sol2 = [Eq(x(t), (C1*cos(sqrt(2)*t) + C2*sin(sqrt(2)*t))*exp(t) - Rational(58, 3)), \
    Eq(y(t), (-sqrt(2)*C1*sin(sqrt(2)*t) + sqrt(2)*C2*cos(sqrt(2)*t))*exp(t) - Rational(185, 3))]  # 第二组微分方程的解
    # 断言验证解是否满足原微分方程组
    assert checksysodesol(eq2, sol2) == (True, [0, 0])
    # 定义常微分方程组 eq3
    eq3 = (Eq(diff(x(t),t), 5*t*x(t) + 2*y(t)), Eq(diff(y(t),t), 2*x(t) + 5*t*y(t)))
    # 给出方程组 eq3 的解 sol3
    sol3 = [Eq(x(t), (C1*exp(2*t) + C2*exp(-2*t))*exp(Rational(5, 2)*t**2)), \
    Eq(y(t), (C1*exp(2*t) - C2*exp(-2*t))*exp(Rational(5, 2)*t**2))]
    # 检查给定的常微分方程组的解是否满足条件，期望返回 (True, [0, 0])
    assert checksysodesol(eq3, sol3) == (True, [0, 0])

    # 定义常微分方程组 eq4
    eq4 = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + 5*t*y(t)))
    # 给出方程组 eq4 的解 sol4
    sol4 = [Eq(x(t), (C1*cos((t**3)/3) + C2*sin((t**3)/3))*exp(Rational(5, 2)*t**2)), \
    Eq(y(t), (-C1*sin((t**3)/3) + C2*cos((t**3)/3))*exp(Rational(5, 2)*t**2))]
    # 检查给定的常微分方程组的解是否满足条件，期望返回 (True, [0, 0])
    assert checksysodesol(eq4, sol4) == (True, [0, 0])

    # 定义常微分方程组 eq5
    eq5 = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + (5*t+9*t**2)*y(t)))
    # 给出方程组 eq5 的解 sol5
    sol5 = [Eq(x(t), (C1*exp((sqrt(77)/2 + Rational(9, 2))*(t**3)/3) + \
    C2*exp((-sqrt(77)/2 + Rational(9, 2))*(t**3)/3))*exp(Rational(5, 2)*t**2)), \
    Eq(y(t), (C1*(sqrt(77)/2 + Rational(9, 2))*exp((sqrt(77)/2 + Rational(9, 2))*(t**3)/3) + \
    C2*(-sqrt(77)/2 + Rational(9, 2))*exp((-sqrt(77)/2 + Rational(9, 2))*(t**3)/3))*exp(Rational(5, 2)*t**2))]
    # 检查给定的常微分方程组的解是否满足条件，期望返回 (True, [0, 0])
    assert checksysodesol(eq5, sol5) == (True, [0, 0])

    # 定义常微分方程组 eq6
    eq6 = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), (1-t**2)*x(t) + (5*t+9*t**2)*y(t)))
    # 给出方程组 eq6 的解 sol6
    sol6 = [Eq(x(t), C1*x0(t) + C2*x0(t)*Integral(t**2*exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)**2, t)), \
    Eq(y(t), C1*y0(t) + C2*(y0(t)*Integral(t**2*exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)**2, t) + \
    exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)))]
    # 对方程组 eq6 求解
    s = dsolve(eq6)
    # 检查求解得到的结果 s 是否与预期解 sol6 相符（此处注释掉的 assert 语句说明测试过于复杂，不执行）
    # assert s == sol6   # too complicated to test with subs and simplify
    # assert checksysodesol(eq10, sol10) == (True, [0, 0])  # this one fails
# 定义一个测试函数，用于测试包含非线性方程组的一阶微分方程组
def test_nonlinear_2eq_order1():
    # 定义符号变量 x, y, z，它们是 Function 类的实例
    x, y, z = symbols('x, y, z', cls=Function)
    # 定义符号变量 t
    t = Symbol('t')
    
    # 定义第一个方程组 eq1
    eq1 = (Eq(diff(x(t),t),x(t)*y(t)**3), Eq(diff(y(t),t),y(t)**5))
    
    # 找到第一个方程组的解 sol1
    sol1 = [
        Eq(x(t), C1*exp((-1/(4*C2 + 4*t))**(Rational(-1, 4)))),
        Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), C1*exp(-1/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), C1*exp(-I/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), C1*exp(I/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    
    # 断言求解方程组 eq1 得到的解与预期的 sol1 相等
    assert dsolve(eq1) == sol1
    # 断言检查方程组 eq1 的解 sol1 是否满足方程组
    assert checksysodesol(eq1, sol1) == (True, [0, 0])

    # 定义第二个方程组 eq2
    eq2 = (Eq(diff(x(t),t), exp(3*x(t))*y(t)**3),Eq(diff(y(t),t), y(t)**5))
    
    # 找到第二个方程组的解 sol2
    sol2 = [
        Eq(x(t), -log(C1 - 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3),
        Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), -log(C1 + 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3),
        Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), -log(C1 + 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3),
        Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), -log(C1 - 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3),
        Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    
    # 断言求解方程组 eq2 得到的解与预期的 sol2 相等
    assert dsolve(eq2) == sol2
    # 断言检查方程组 eq2 的解 sol2 是否满足方程组
    assert checksysodesol(eq2, sol2) == (True, [0, 0])

    # 定义第三个方程组 eq3
    eq3 = (Eq(diff(x(t),t), y(t)*x(t)), Eq(diff(y(t),t), x(t)**3))
    # 定义有理数 tt
    tt = Rational(2, 3)
    
    # 找到第三个方程组的解 sol3
    sol3 = [
        Eq(x(t), 6**tt/(6*(-sinh(sqrt(C1)*(C2 + t)/2)/sqrt(C1))**tt)),
        Eq(y(t), sqrt(C1 + C1/sinh(sqrt(C1)*(C2 + t)/2)**2)/3)]
    
    # 断言求解方程组 eq3 得到的解与预期的 sol3 相等
    assert dsolve(eq3) == sol3
    # FIXME: 断言检查方程组 eq3 的解 sol3 是否满足方程组

    # 定义第四个方程组 eq4
    eq4 = (Eq(diff(x(t),t),x(t)*y(t)*sin(t)**2), Eq(diff(y(t),t),y(t)**2*sin(t)**2))
    
    # 找到第四个方程组的解 sol4
    sol4 = {Eq(x(t), -2*exp(C1)/(C2*exp(C1) + t - sin(2*t)/2)), Eq(y(t), -2/(C1 + t - sin(2*t)/2))}
    
    # 断言求解方程组 eq4 得到的解与预期的 sol4 相等
    assert dsolve(eq4) == sol4
    # FIXME: 断言检查方程组 eq4 的解 sol4 是否满足方程组

    # 定义第五个方程组 eq5
    eq5 = (Eq(x(t),t*diff(x(t),t)+diff(x(t),t)*diff(y(t),t)), Eq(y(t),t*diff(y(t),t)+diff(y(t),t)**2))
    
    # 找到第五个方程组的解 sol5
    sol5 = {Eq(x(t), C1*C2 + C1*t), Eq(y(t), C2**2 + C2*t)}
    
    # 断言求解方程组 eq5 得到的解与预期的 sol5 相等
    assert dsolve(eq5) == sol5
    # 断言检查方程组 eq5 的解 sol5 是否满足方程组
    assert checksysodesol(eq5, sol5) == (True, [0, 0])

    # 定义第六个方程组 eq6
    eq6 = (Eq(diff(x(t),t),x(t)**2*y(t)**3), Eq(diff(y(t),t),y(t)**5))
    
    # 找到第六个方程组的解 sol6
    sol6 = [
        Eq(x(t), 1/(C1 - 1/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), 1/(C1 + (-1/(4*C2 + 4*t))**(Rational(-1, 4)))),
        Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), 1/(C1 + I/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), 1/(C1 - I/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    
    # 断言求解方程组 eq6 得到的解与预期的 sol6 相等
    assert dsolve(eq6) == sol6
    # 断言检查方程组 eq6 的解 sol6 是否满足方程组
    assert checksysodesol(eq6, sol6) == (True, [0, 0])


@slow
# 定义一个装饰器 @slow，表示这个测试函数执行比较慢
def test_nonlinear_3eq_order1():
    # 使用 sympy 库的 symbols 函数定义符号变量 x, y, z，它们是时间 t 的函数
    x, y, z = symbols('x, y, z', cls=Function)
    # 使用 sympy 库的 symbols 函数定义符号变量 t, u
    t, u = symbols('t u')
    # 定义微分方程组 eq1，包括三个微分方程
    eq1 = (4*diff(x(t),t) + 2*y(t)*z(t), 3*diff(y(t),t) - z(t)*x(t), 5*diff(z(t),t) - x(t)*y(t))
    # 求解微分方程组 eq1，得到的解为 sol1
    sol1 = [Eq(4*Integral(1/(sqrt(-4*u**2 - 3*C1 + C2)*sqrt(-4*u**2 + 5*C1 - C2)), (u, x(t))),
        C3 - sqrt(15)*t/15), Eq(3*Integral(1/(sqrt(-6*u**2 - C1 + 5*C2)*sqrt(3*u**2 + C1 - 4*C2)),
        (u, y(t))), C3 + sqrt(5)*t/10), Eq(5*Integral(1/(sqrt(-10*u**2 - 3*C1 + C2)*
        sqrt(5*u**2 + 4*C1 - C2)), (u, z(t))), C3 + sqrt(3)*t/6)]
    # 使用 assert 语句检查 dsolve(eq1) 返回的解是否与 sol1 中的每个方程相等
    assert [i.dummy_eq(j) for i, j in zip(dsolve(eq1), sol1)]
    # FIXME: assert checksysodesol(eq1, sol1) == (True, [0, 0, 0])

    # 定义微分方程组 eq2，包括三个微分方程，其中包含 sin(t) 因子
    eq2 = (4*diff(x(t),t) + 2*y(t)*z(t)*sin(t), 3*diff(y(t),t) - z(t)*x(t)*sin(t), 5*diff(z(t),t) - x(t)*y(t)*sin(t))
    # 求解微分方程组 eq2，得到的解为 sol2
    sol2 = [Eq(3*Integral(1/(sqrt(-6*u**2 - C1 + 5*C2)*sqrt(3*u**2 + C1 - 4*C2)), (u, x(t))), C3 +
        sqrt(5)*cos(t)/10), Eq(4*Integral(1/(sqrt(-4*u**2 - 3*C1 + C2)*sqrt(-4*u**2 + 5*C1 - C2)),
        (u, y(t))), C3 - sqrt(15)*cos(t)/15), Eq(5*Integral(1/(sqrt(-10*u**2 - 3*C1 + C2)*
        sqrt(5*u**2 + 4*C1 - C2)), (u, z(t))), C3 + sqrt(3)*cos(t)/6)]
    # 使用 assert 语句检查 dsolve(eq2) 返回的解是否与 sol2 中的每个方程相等
    assert [i.dummy_eq(j) for i, j in zip(dsolve(eq2), sol2)]
    # FIXME: assert checksysodesol(eq2, sol2) == (True, [0, 0, 0])
# 定义测试函数 test_C1_function_9239，用于测试某个特定的函数或功能
def test_C1_function_9239():
    # 定义符号变量 t
    t = Symbol('t')
    # 定义函数符号 C1 和 C2
    C1 = Function('C1')
    C2 = Function('C2')
    # 定义符号常量 C3 和 C4
    C3 = Symbol('C3')
    C4 = Symbol('C4')
    # 定义微分方程组 eq，包含两个方程
    eq = (Eq(diff(C1(t), t), 9*C2(t)), Eq(diff(C2(t), t), 12*C1(t)))
    # 定义解 sol，为微分方程组的解
    sol = [Eq(C1(t), 9*C3*exp(6*sqrt(3)*t) + 9*C4*exp(-6*sqrt(3)*t)),
           Eq(C2(t), 6*sqrt(3)*C3*exp(6*sqrt(3)*t) - 6*sqrt(3)*C4*exp(-6*sqrt(3)*t))]
    # 断言检查解是否符合微分方程组 eq 的要求，期望返回 (True, [0, 0])
    assert checksysodesol(eq, sol) == (True, [0, 0])


# 定义测试函数 test_dsolve_linsystem_symbol，用于测试另一个特定的函数或功能
def test_dsolve_linsystem_symbol():
    # 定义正数符号变量 epsilon，并命名为 eps
    eps = Symbol('epsilon', positive=True)
    # 定义微分方程组 eq1，包含两个方程
    eq1 = (Eq(diff(f(x), x), -eps*g(x)), Eq(diff(g(x), x), eps*f(x)))
    # 定义解 sol1，为微分方程组 eq1 的解
    sol1 = [Eq(f(x), -C1*eps*cos(eps*x) - C2*eps*sin(eps*x)),
            Eq(g(x), -C1*eps*sin(eps*x) + C2*eps*cos(eps*x))]
    # 断言检查解是否符合微分方程组 eq1 的要求，期望返回 (True, [0, 0])
    assert checksysodesol(eq1, sol1) == (True, [0, 0])
```