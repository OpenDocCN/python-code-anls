# `D:\src\scipysrc\sympy\sympy\solvers\diophantine\tests\test_diophantine.py`

```
# 从 sympy.core.add 模块中导入 Add 类
from sympy.core.add import Add
# 从 sympy.core.mul 模块中导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.numbers 模块中导入 Rational, oo, pi 等类和常量
from sympy.core.numbers import (Rational, oo, pi)
# 从 sympy.core.relational 模块中导入 Eq 类
from sympy.core.relational import Eq
# 从 sympy.core.singleton 模块中导入 S 类
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块中导入 symbols 函数
from sympy.core.symbol import symbols
# 从 sympy.matrices.dense 模块中导入 Matrix 类
from sympy.matrices.dense import Matrix
# 从 sympy.ntheory.factor_ 模块中导入 factorint 函数
from sympy.ntheory.factor_ import factorint
# 从 sympy.simplify.powsimp 模块中导入 powsimp 函数
from sympy.simplify.powsimp import powsimp
# 从 sympy.core.function 模块中导入 _mexpand 函数
from sympy.core.function import _mexpand
# 从 sympy.core.sorting 模块中导入 default_sort_key, ordered 函数
from sympy.core.sorting import default_sort_key, ordered
# 从 sympy.functions.elementary.trigonometric 模块中导入 sin 函数
from sympy.functions.elementary.trigonometric import sin
# 从 sympy.solvers.diophantine 模块中导入 diophantine 函数和相关类和函数
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
    diop_solve, diop_ternary_quadratic_normal,
    diop_general_pythagorean, diop_ternary_quadratic, diop_linear,
    diop_quadratic, diop_general_sum_of_squares, diop_general_sum_of_even_powers,
    descent, diop_bf_DN, divisible, equivalent, find_DN, ldescent, length,
    reconstruct, partition, power_representation,
    prime_as_sum_of_two_squares, square_factor, sum_of_four_squares,
    sum_of_three_squares, transformation_to_DN, transformation_to_normal,
    classify_diop, base_solution_linear, cornacchia, sqf_normal, gaussian_reduce, holzer,
    check_param, parametrize_ternary_quadratic, sum_of_powers, sum_of_squares,
    _diop_ternary_quadratic_normal, _nint_or_floor,
    _odd, _even, _remove_gcd, _can_do_sum_of_squares, DiophantineSolutionSet, GeneralPythagorean,
    BinaryQuadratic)
# 从 sympy.testing.pytest 模块中导入 slow, raises, XFAIL 函数
from sympy.testing.pytest import slow, raises, XFAIL
# 从 sympy.utilities.iterables 模块中导入 signed_permutations 函数
from sympy.utilities.iterables import (
        signed_permutations)

# 定义整数类型的符号变量
a, b, c, d, p, q, x, y, z, w, t, u, v, X, Y, Z = symbols(
    "a, b, c, d, p, q, x, y, z, w, t, u, v, X, Y, Z", integer=True)
# 定义整数类型的符号变量 t_0 到 t_6
t_0, t_1, t_2, t_3, t_4, t_5, t_6 = symbols("t_:7", integer=True)
# 定义整数类型的符号变量 m1, m2, m3
m1, m2, m3 = symbols('m1:4', integer=True)
# 定义整数类型的符号变量 n1
n1 = symbols('n1', integer=True)


# 定义 diop_simplify 函数，简化丢番图方程的表达式
def diop_simplify(eq):
    return _mexpand(powsimp(_mexpand(eq)))


# 定义 test_input_format 测试函数，验证 diophantine 函数输入格式的异常情况
def test_input_format():
    raises(TypeError, lambda: diophantine(sin(x)))
    raises(TypeError, lambda: diophantine(x/pi - 3))


# 定义 test_nosols 测试函数，验证 diophantine 函数对于无解的情况的处理
def test_nosols():
    # diophantine 应该将输入的表达式转化为 sympy 对象，因此以下断言应该成立
    assert diophantine(3) == set()
    assert diophantine(S(3)) == set()


# 定义 test_univariate 测试函数，验证 diop_solve 函数对一元丢番图方程的求解
def test_univariate():
    assert diop_solve((x - 1)*(x - 2)**2) == {(1,), (2,)}
    assert diop_solve((x - 1)*(x - 2)) == {(1,), (2,)}


# 定义 test_classify_diop 测试函数，验证 classify_diop 函数对丢番图方程分类的准确性和异常情况的处理
def test_classify_diop():
    raises(TypeError, lambda: classify_diop(x**2/3 - 1))
    raises(ValueError, lambda: classify_diop(1))
    raises(NotImplementedError, lambda: classify_diop(w*x*y*z - 1))
    raises(NotImplementedError, lambda: classify_diop(x**3 + y**3 + z**4 - 90))
    assert classify_diop(14*x**2 + 15*x - 42) == (
        [x], {1: -42, x: 15, x**2: 14}, 'univariate')
    assert classify_diop(x*y + z) == (
        [x, y, z], {x*y: 1, z: 1}, 'inhomogeneous_ternary_quadratic')
    assert classify_diop(x*y + z + w + x**2) == (
        [w, x, y, z], {x*y: 1, w: 1, x**2: 1, z: 1}, 'inhomogeneous_general_quadratic')
    # 断言：验证 classify_diop 函数对于特定输入的返回值是否符合预期
    assert classify_diop(x*y + x*z + x**2 + 1) == (
        [x, y, z], {x*y: 1, x*z: 1, x**2: 1, 1: 1}, 'inhomogeneous_general_quadratic')
    
    # 断言：验证 classify_diop 函数对于特定输入的返回值是否符合预期
    assert classify_diop(x*y + z + w + 42) == (
        [w, x, y, z], {x*y: 1, w: 1, 1: 42, z: 1}, 'inhomogeneous_general_quadratic')
    
    # 断言：验证 classify_diop 函数对于特定输入的返回值是否符合预期
    assert classify_diop(x*y + z*w) == (
        [w, x, y, z], {x*y: 1, w*z: 1}, 'homogeneous_general_quadratic')
    
    # 断言：验证 classify_diop 函数对于特定输入的返回值是否符合预期
    assert classify_diop(x*y**2 + 1) == (
        [x, y], {x*y**2: 1, 1: 1}, 'cubic_thue')
    
    # 断言：验证 classify_diop 函数对于特定输入的返回值是否符合预期
    assert classify_diop(x**4 + y**4 + z**4 - (1 + 16 + 81)) == (
        [x, y, z], {1: -98, x**4: 1, z**4: 1, y**4: 1}, 'general_sum_of_even_powers')
    
    # 断言：验证 classify_diop 函数对于特定输入的返回值是否符合预期
    assert classify_diop(x**2 + y**2 + z**2) == (
        [x, y, z], {x**2: 1, y**2: 1, z**2: 1}, 'homogeneous_ternary_quadratic_normal')
# 定义一个测试函数，用于测试线性二次方程的解法函数 diop_solve
def test_linear():
    # 断言解 x 的线性二次方程为 (0,)
    assert diop_solve(x) == (0,)
    # 断言解 1*x 的线性二次方程为 (0,)
    assert diop_solve(1*x) == (0,)
    # 断言解 3*x 的线性二次方程为 (0,)
    assert diop_solve(3*x) == (0,)
    # 断言解 x + 1 的线性二次方程为 (-1,)
    assert diop_solve(x + 1) == (-1,)
    # 断言解 2*x + 1 的线性二次方程为 (None,)
    assert diop_solve(2*x + 1) == (None,)
    # 断言解 2*x + 4 的线性二次方程为 (-2,)
    assert diop_solve(2*x + 4) == (-2,)
    # 断言解 y + x 的线性二次方程为 (t_0, -t_0)
    assert diop_solve(y + x) == (t_0, -t_0)
    # 断言解 y + x + 0 的线性二次方程为 (t_0, -t_0)
    assert diop_solve(y + x + 0) == (t_0, -t_0)
    # 断言解 y + x - 0 的线性二次方程为 (t_0, -t_0)
    assert diop_solve(y + x - 0) == (t_0, -t_0)
    # 断言解 0*x - y - 5 的线性二次方程为 (-5,)
    assert diop_solve(0*x - y - 5) == (-5,)
    # 断言解 3*y + 2*x - 5 的线性二次方程为 (3*t_0 - 5, -2*t_0 + 5)
    assert diop_solve(3*y + 2*x - 5) == (3*t_0 - 5, -2*t_0 + 5)
    # 断言解 2*x - 3*y - 5 的线性二次方程为 (3*t_0 - 5, 2*t_0 - 5)
    assert diop_solve(2*x - 3*y - 5) == (3*t_0 - 5, 2*t_0 - 5)
    # 断言解 -2*x - 3*y - 5 的线性二次方程为 (3*t_0 + 5, -2*t_0 - 5)
    assert diop_solve(-2*x - 3*y - 5) == (3*t_0 + 5, -2*t_0 - 5)
    # 断言解 7*x + 5*y 的线性二次方程为 (5*t_0, -7*t_0)
    assert diop_solve(7*x + 5*y) == (5*t_0, -7*t_0)
    # 断言解 2*x + 4*y 的线性二次方程为 (-2*t_0, t_0)
    assert diop_solve(2*x + 4*y) == (-2*t_0, t_0)
    # 断言解 4*x + 6*y - 4 的线性二次方程为 (3*t_0 - 2, -2*t_0 + 2)
    assert diop_solve(4*x + 6*y - 4) == (3*t_0 - 2, -2*t_0 + 2)
    # 断言解 4*x + 6*y - 3 的线性二次方程为 (None, None)
    assert diop_solve(4*x + 6*y - 3) == (None, None)
    # 断言解 0*x + 3*y - 4*z + 5 的线性二次方程为 (4*t_0 + 5, 3*t_0 + 5)
    assert diop_solve(0*x + 3*y - 4*z + 5) == (4*t_0 + 5, 3*t_0 + 5)
    # 断言解 4*x + 3*y - 4*z + 5 的线性二次方程为 (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)
    assert diop_solve(4*x + 3*y - 4*z + 5) == (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)
    # 断言解 4*x + 3*y - 4*z + 5, None 的线性二次方程为 (0, 5, 5)
    assert diop_solve(4*x + 3*y - 4*z + 5, None) == (0, 5, 5)
    # 断言解 4*x + 2*y + 8*z - 5 的线性二次方程为 (None, None, None)
    assert diop_solve(4*x + 2*y + 8*z - 5) == (None, None, None)
    # 断言解 5*x + 7*y - 2*z - 6 的线性二次方程为 (t_0, -3*t_0 + 2*t_1 + 6, -8*t_0 + 7*t_1 + 18)
    assert diop_solve(5*x + 7*y - 2*z - 6) == (t_0, -3*t_0 + 2*t_1 + 6, -8*t_0 + 7*t_1 + 18)
    # 断言解 3*x - 6*y + 12*z - 9 的线性二次方程为 (2*t_0 + 3, t_0 + 2*t_1, t_1)
    assert diop_solve(3*x - 6*y + 12*z - 9) == (2*t_0 + 3, t_0 + 2*t_1, t_1)
    # 断言解 6*w + 9*x + 20*y - z 的线性二次方程为 (t_0, t_1, t_1 + t_2, 6*t_0 + 29*t_1 + 20*t_2)

    # 忽略常数因子，使用 diophantine 函数
    raises(TypeError, lambda: diop_solve(x/2))


# 定义一个测试函数，用于测试二次方程的解法函数 diop_solve，简单的双曲线情况
def test_quadratic_simple_hyperbolic_case():
    # 简单的双曲线情况：A = C = 0，B ≠ 0
    assert diop_solve(3*x*y + 34*x - 12*y + 1) == {(-133, -11), (5, -57)}
    assert diop_solve(6*x*y + 2*x + 3*y + 1) == set()
    assert diop_solve(-13*x*y + 2*x - 4*y - 54) == {(27, 0)}
    assert diop_solve(-27*x*y - 30*x - 12*y - 54) == {(-14, -1)}
    assert diop_solve(2*x*y + 5*x + 56*y + 7) == {(-161, -3), (-47, -6), (-35, -12),
                                                  (-29, -69), (-27, 64), (-21, 7),
                                                  (-9, 1), (105, -2)}
    assert diop_solve(6*x*y + 9*x + 2*y + 3) == set()
    assert diop_solve(x*y + x + y + 1) == {(-1, t), (t, -1)}
    assert diophantine(48*x*y)


# 定义一个测试函数，用于测试二次方程的解法函数 diop_solve，椭圆情况
def test_quadratic_elliptical_case():
    # 椭圆情况：B**2 - 4AC < 0
    assert diop_solve(42*x**2 + 8*x*y + 15*y**2 + 23*x + 17*y - 4915) == {(-11, -1)}
    assert diop_solve(4*x**2 + 3*y**2 + 5*x - 11*y + 12) == set()
    assert diop_solve(x**2 + y**2 + 2*x + 2*y + 2) == {(-1, -1)}
    # 断言，验证方程的解是否满足特定条件
    assert check_solutions(8*x**2 + 24*x*y + 18*y**2 + 4*x + 6*y - 7)

    # 断言，验证方程的解是否满足特定条件
    assert check_solutions(-4*x**2 + 4*x*y - y**2 + 2*x - 3)

    # 断言，验证方程的解是否满足特定条件
    assert check_solutions(x**2 + 2*x*y + y**2 + 2*x + 2*y + 1)

    # 断言，验证方程的解是否满足特定条件
    assert check_solutions(x**2 - 2*x*y + y**2 + 2*x + 2*y + 1)

    # 断言，验证方程的解是否满足特定条件
    assert check_solutions(y**2 - 41*x + 40)
def test_quadratic_perfect_square():
    # 检查方程 B**2 - 4*A*C > 0 是否成立，确保方程有实数解
    assert check_solutions(48*x*y)
    # 检查方程 B**2 - 4*A*C 是否为完全平方数
    assert check_solutions(4*x**2 - 5*x*y + y**2 + 2)
    assert check_solutions(-2*x**2 - 3*x*y + 2*y**2 -2*x - 17*y + 25)
    assert check_solutions(12*x**2 + 13*x*y + 3*y**2 - 2*x + 3*y - 12)
    assert check_solutions(8*x**2 + 10*x*y + 2*y**2 - 32*x - 13*y - 23)
    assert check_solutions(4*x**2 - 4*x*y - 3*y- 8*x - 3)
    assert check_solutions(- 4*x*y - 4*y**2 - 3*y- 5*x - 10)
    assert check_solutions(x**2 - y**2 - 2*x - 2*y)
    assert check_solutions(x**2 - 9*y**2 - 2*x - 6*y)
    assert check_solutions(4*x**2 - 9*y**2 - 4*x - 12*y - 3)


def test_quadratic_non_perfect_square():
    # 检查方程 B**2 - 4*A*C 不是完全平方数，需要用 check_solutions() 检查复杂表达式的解
    assert check_solutions(x**2 - 2*x - 5*y**2)
    assert check_solutions(3*x**2 - 2*y**2 - 2*x - 2*y)
    assert check_solutions(x**2 - x*y - y**2 - 3*y)
    assert check_solutions(x**2 - 9*y**2 - 2*x - 6*y)
    assert BinaryQuadratic(x**2 + y**2 + 2*x + 2*y + 2).solve() == {(-1, -1)}


def test_issue_9106():
    eq = -48 - 2*x*(3*x - 1) + y*(3*y - 1)
    v = (x, y)
    for sol in diophantine(eq):
        # 确保 diophantine 方程的简化结果不为真
        assert not diop_simplify(eq.xreplace(dict(zip(v, sol))))


def test_issue_18138():
    eq = x**2 - x - y**2
    v = (x, y)
    for sol in diophantine(eq):
        # 确保 diophantine 方程的简化结果不为真
        assert not diop_simplify(eq.xreplace(dict(zip(v, sol))))


@slow
def test_quadratic_non_perfect_slow():
    assert check_solutions(8*x**2 + 10*x*y - 2*y**2 - 32*x - 13*y - 23)
    # 这会导致非常大的数字。
    # assert check_solutions(5*x**2 - 13*x*y + y**2 - 4*x - 4*y - 15)
    assert check_solutions(-3*x**2 - 2*x*y + 7*y**2 - 5*x - 7)
    assert check_solutions(-4 - x + 4*x**2 - y - 3*x*y - 4*y**2)
    assert check_solutions(1 + 2*x + 2*x**2 + 2*y + x*y - 2*y**2)


def test_DN():
    # 大多数测试用例改编自
    # Solving the generalized Pell equation x**2 - D*y**2 = N, John P. Robertson, July 31, 2004.
    # https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    # 其他测试通过 Wolfram Alpha 验证。

    # 涵盖 D <= 0 或 D > 0 且 D 是完全平方数，或 N = 0 的情况
    # 这些情况下的解是直接的。
    assert diop_DN(3, 0) == [(0, 0)]
    assert diop_DN(-17, -5) == []
    assert diop_DN(-19, 23) == [(2, 1)]
    assert diop_DN(-13, 17) == [(2, 1)]
    assert diop_DN(-15, 13) == []
    assert diop_DN(0, 5) == []
    assert diop_DN(0, 9) == [(3, t)]
    assert diop_DN(9, 0) == [(3*t, t)]
    assert diop_DN(16, 24) == []
    assert diop_DN(9, 180) == [(18, 4)]
    assert diop_DN(9, -180) == [(12, 6)]
    assert diop_DN(7, 0) == [(0, 0)]

    # 当方程为 x**2 + y**2 = N 时
    # 解是可以互换的
    assert diop_DN(-1, 5) == [(2, 1), (1, 2)]
    assert diop_DN(-1, 169) == [(12, 5), (5, 12), (13, 0), (0, 13)]
    # D > 0 and D is not a square

    # N = 1
    # 断言检查对应不同 D 和 N = 1 的情况下，diop_DN 函数的返回结果是否符合预期
    assert diop_DN(13, 1) == [(649, 180)]
    assert diop_DN(980, 1) == [(51841, 1656)]
    assert diop_DN(981, 1) == [(158070671986249, 5046808151700)]
    assert diop_DN(986, 1) == [(49299, 1570)]
    assert diop_DN(991, 1) == [(379516400906811930638014896080, 12055735790331359447442538767)]
    assert diop_DN(17, 1) == [(33, 8)]
    assert diop_DN(19, 1) == [(170, 39)]

    # N = -1
    # 断言检查对应不同 D 和 N = -1 的情况下，diop_DN 函数的返回结果是否符合预期
    assert diop_DN(13, -1) == [(18, 5)]
    assert diop_DN(991, -1) == []
    assert diop_DN(41, -1) == [(32, 5)]
    assert diop_DN(290, -1) == [(17, 1)]
    assert diop_DN(21257, -1) == [(13913102721304, 95427381109)]
    assert diop_DN(32, -1) == []

    # |N| > 1
    # 一些测试是在以下网站上创建的：http://www.numbertheory.org/php/patz.html
    # 断言检查对应不同 D 和 |N| > 1 的情况下，diop_DN 函数的返回结果是否符合预期
    assert diop_DN(13, -4) == [(3, 1), (393, 109), (36, 10)]
    # 我参考的来源返回了 (3, 1), (393, 109) 和 (-3, 1) 作为基本解
    # 因此 (-3, 1) 和 (393, 109) 应属于同一等价类
    assert equivalent(-3, 1, 393, 109, 13, -4) == True

    assert diop_DN(13, 27) == [(220, 61), (40, 11), (768, 213), (12, 3)]
    assert set(diop_DN(157, 12)) == {(13, 1), (10663, 851), (579160, 46222),
                                     (483790960, 38610722), (26277068347, 2097138361),
                                     (21950079635497, 1751807067011)}
    assert diop_DN(13, 25) == [(3245, 900)]
    assert diop_DN(192, 18) == []
    assert diop_DN(23, 13) == [(-6, 1), (6, 1)]
    assert diop_DN(167, 2) == [(13, 1)]
    assert diop_DN(167, -2) == []

    assert diop_DN(123, -2) == [(11, 1)]
    # 一个计算器返回了 [(11, 1), (-11, 1)] 但这两个解在同一等价类中
    assert equivalent(11, 1, -11, 1, 123, -2)

    assert diop_DN(123, -23) == [(-10, 1), (10, 1)]

    assert diop_DN(0, 0, t) == [(0, t)]
    assert diop_DN(0, -1, t) == []
def test_bf_pell():
    # 测试 diop_bf_DN 函数是否返回预期的结果列表
    assert diop_bf_DN(13, -4) == [(3, 1), (-3, 1), (36, 10)]
    assert diop_bf_DN(13, 27) == [(12, 3), (-12, 3), (40, 11), (-40, 11)]
    assert diop_bf_DN(167, -2) == []
    assert diop_bf_DN(1729, 1) == [(44611924489705, 1072885712316)]
    assert diop_bf_DN(89, -8) == [(9, 1), (-9, 1)]
    assert diop_bf_DN(21257, -1) == [(13913102721304, 95427381109)]
    assert diop_bf_DN(340, -4) == [(756, 41)]
    assert diop_bf_DN(-1, 0, t) == [(0, 0)]
    assert diop_bf_DN(0, 0, t) == [(0, t)]
    assert diop_bf_DN(4, 0, t) == [(2*t, t), (-2*t, t)]
    assert diop_bf_DN(3, 0, t) == [(0, 0)]
    assert diop_bf_DN(1, -2, t) == []


def test_length():
    # 测试 length 函数是否返回预期的结果
    assert length(2, 1, 0) == 1
    assert length(-2, 4, 5) == 3
    assert length(-5, 4, 17) == 4
    assert length(0, 4, 13) == 6
    assert length(7, 13, 11) == 23
    assert length(1, 6, 4) == 2


def is_pell_transformation_ok(eq):
    """
    根据 transformation_to_pell() 转换方程，并检查变换后的方程中是否包含 X*Y、X 或 Y 项。
    如果这些项不存在，且 X**2 的系数是 Y**2 和常数项的约数，则返回 True。
    否则返回 False。
    """
    A, B = transformation_to_DN(eq)
    u = (A*Matrix([X, Y]) + B)[0]
    v = (A*Matrix([X, Y]) + B)[1]
    simplified = diop_simplify(eq.subs(zip((x, y), (u, v))))

    coeff = dict([reversed(t.as_independent(*[X, Y])) for t in simplified.args])

    for term in [X*Y, X, Y]:
        if term in coeff.keys():
            return False

    for term in [X**2, Y**2, 1]:
        if term not in coeff.keys():
            coeff[term] = 0

    if coeff[X**2] != 0:
        return divisible(coeff[Y**2], coeff[X**2]) and \
        divisible(coeff[1], coeff[X**2])

    return True


def test_transformation_to_pell():
    # 测试 is_pell_transformation_ok 函数是否正确处理给定的 Pell 方程
    assert is_pell_transformation_ok(-13*x**2 - 7*x*y + y**2 + 2*x - 2*y - 14)
    assert is_pell_transformation_ok(-17*x**2 + 19*x*y - 7*y**2 - 5*x - 13*y - 23)
    assert is_pell_transformation_ok(x**2 - y**2 + 17)
    assert is_pell_transformation_ok(-x**2 + 7*y**2 - 23)
    assert is_pell_transformation_ok(25*x**2 - 45*x*y + 5*y**2 - 5*x - 10*y + 5)
    assert is_pell_transformation_ok(190*x**2 + 30*x*y + y**2 - 3*y - 170*x - 130)
    assert is_pell_transformation_ok(x**2 - 2*x*y - 190*y**2 - 7*y - 23*x - 89)
    assert is_pell_transformation_ok(15*x**2 - 9*x*y + 14*y**2 - 23*x - 14*y - 4950)


def test_find_DN():
    # 测试 find_DN 函数是否返回预期的 D 和 N 值对
    assert find_DN(x**2 - 2*x - y**2) == (1, 1)
    assert find_DN(x**2 - 3*y**2 - 5) == (3, 5)
    assert find_DN(x**2 - 2*x*y - 4*y**2 - 7) == (5, 7)
    assert find_DN(4*x**2 - 8*x*y - y**2 - 9) == (20, 36)
    assert find_DN(7*x**2 - 2*x*y - y**2 - 12) == (8, 84)
    assert find_DN(-3*x**2 + 4*x*y - y**2) == (1, 0)
    assert find_DN(-13*x**2 - 7*x*y + y**2 + 2*x - 2*y - 14) == (101, -7825480)


def test_ldescent():
    # 这个测试函数是空的，因为没有提供任何等式来测试
    # 等式需要有解
    # 给定的参数列表
    u = ([(13, 23), (3, -11), (41, -113), (4, -7), (-7, 4), (91, -3), (1, 1), (1, -1),
        (4, 32), (17, 13), (123689, 1), (19, -570)])
    
    # 遍历参数列表中的每对参数
    for a, b in u:
        # 调用函数 ldescent，传入参数 a 和 b，返回结果给变量 w, x, y
        w, x, y = ldescent(a, b)
        # 断言条件，检查满足整数解条件
        assert a*x**2 + b*y**2 == w**2
    
    # 断言调用 ldescent 函数返回 None 的情况
    assert ldescent(-1, -1) is None
    assert ldescent(2, 6) is None
# 测试二元二次多项式的解是否满足特定的条件
def test_diop_ternary_quadratic_normal():
    # 检查解是否满足给定的二元二次多项式方程，返回布尔值
    assert check_solutions(234*x**2 - 65601*y**2 - z**2)
    assert check_solutions(23*x**2 + 616*y**2 - z**2)
    assert check_solutions(5*x**2 + 4*y**2 - z**2)
    assert check_solutions(3*x**2 + 6*y**2 - 3*z**2)
    assert check_solutions(x**2 + 3*y**2 - z**2)
    assert check_solutions(4*x**2 + 5*y**2 - z**2)
    assert check_solutions(x**2 + y**2 - z**2)
    assert check_solutions(16*x**2 + y**2 - 25*z**2)
    assert check_solutions(6*x**2 - y**2 + 10*z**2)
    assert check_solutions(213*x**2 + 12*y**2 - 9*z**2)
    assert check_solutions(34*x**2 - 3*y**2 - 301*z**2)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)


# 检查是否进行了正常的转换
def is_normal_transformation_ok(eq):
    # 将方程转换为正常形式，并返回转换后的矩阵
    A = transformation_to_normal(eq)
    # 将变量 X, Y, Z 分别乘以转换后的矩阵 A
    X, Y, Z = A*Matrix([x, y, z])
    # 简化转换后的方程，将变量 x, y, z 替换为 X, Y, Z，并返回简化结果
    simplified = diop_simplify(eq.subs(zip((x, y, z), (X, Y, Z))))

    # 将简化后方程的每一项与变量 X, Y, Z 分离，并形成系数的字典
    coeff = dict([reversed(t.as_independent(*[X, Y, Z])) for t in simplified.args])
    # 检查是否存在 X*Y, Y*Z, X*Z 这些项，如果存在则返回 False
    for term in [X*Y, Y*Z, X*Z]:
        if term in coeff.keys():
            return False

    # 如果上述条件都不满足，则返回 True
    return True


# 测试转换为正常形式的函数
def test_transformation_to_normal():
    # 检查转换后的方程是否满足正常转换的条件
    assert is_normal_transformation_ok(x**2 + 3*y**2 + z**2 - 13*x*y - 16*y*z + 12*x*z)
    assert is_normal_transformation_ok(x**2 + 3*y**2 - 100*z**2)
    assert is_normal_transformation_ok(x**2 + 23*y*z)
    assert is_normal_transformation_ok(3*y**2 - 100*z**2 - 12*x*y)
    assert is_normal_transformation_ok(x**2 + 23*x*y - 34*y*z + 12*x*z)
    assert is_normal_transformation_ok(z**2 + 34*x*y - 23*y*z + x*z)
    assert is_normal_transformation_ok(x**2 + y**2 + z**2 - x*y - y*z - x*z)
    assert is_normal_transformation_ok(x**2 + 2*y*z + 3*z**2)
    assert is_normal_transformation_ok(x*y + 2*x*z + 3*y*z)
    assert is_normal_transformation_ok(2*x*z + 3*y*z)


# 测试三元二次多项式的解是否满足特定的条件
def test_diop_ternary_quadratic():
    assert check_solutions(2*x**2 + z**2 + y**2 - 4*x*y)
    assert check_solutions(x**2 - y**2 - z**2 - x*y - y*z)
    assert check_solutions(3*x**2 - x*y - y*z - x*z)
    assert check_solutions(x**2 - y*z - x*z)
    assert check_solutions(5*x**2 - 3*x*y - x*z)
    assert check_solutions(4*x**2 - 5*y**2 - x*z)
    assert check_solutions(3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z)
    assert check_solutions(8*x**2 - 12*y*z)
    assert check_solutions(45*x**2 - 7*y**2 - 8*x*y - z**2)
    assert check_solutions(x**2 - 49*y**2 - z**2 + 13*z*y - 8*x*y)
    assert check_solutions(90*x**2 + 3*y**2 + 5*x*y + 2*z*y + 5*x*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - x*y - 17*y*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - x*y - 16*y*z + 12*x*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - 13*x*y - 16*y*z + 12*x*z)
    assert check_solutions(x*y - 7*y*z + 13*x*z)

    # 检查三元二次多项式 x**2 + y**2 + z**2 的解是否为 None
    assert diop_ternary_quadratic_normal(x**2 + y**2 + z**2) == (None, None, None)
    # 检查二元二次多项式 x**2 + y**2 的解是否为 None
    assert diop_ternary_quadratic_normal(x**2 + y**2) is None
    # 检查未知的 ValueError 异常是否被正确地引发
    raises(ValueError, lambda: _diop_ternary_quadratic_normal((x, y, z), {x*y: 1, x**2: 2, y**2: 3, z**2: 0}))
    # 定义一个二次三元整数方程
    eq = -2*x*y - 6*x*z + 7*y**2 - 3*y*z + 4*z**2
    # 断言使用 diop_ternary_quadratic 函数解方程，期望结果为 (7, 2, 0)
    assert diop_ternary_quadratic(eq) == (7, 2, 0)
    # 断言使用 diop_ternary_quadratic_normal 函数解标准形式的三元二次方程，期望结果为 (1, 0, 2)
    assert diop_ternary_quadratic_normal(4*x**2 + 5*y**2 - z**2) == (1, 0, 2)
    # 断言使用 diop_ternary_quadratic 函数解方程，期望结果为 (-2, 0, n1)，其中 n1 是变量
    assert diop_ternary_quadratic(x*y + 2*y*z) == (-2, 0, n1)
    # 定义一个二次三元整数方程
    eq = -5*x*y - 8*x*z - 3*y*z + 8*z**2
    # 断言使用 parametrize_ternary_quadratic 函数解参数化后的三元二次方程，期望结果为 (8*p**2 - 3*p*q, -8*p*q + 8*q**2, 5*p*q)
    assert parametrize_ternary_quadratic(eq) == (8*p**2 - 3*p*q, -8*p*q + 8*q**2, 5*p*q)
    # 断言使用 diop_solve 函数解方程，期望结果为 (-2*p*q, -n1*p**2 + p**2, p*q)
    # 由于该方程可能会因式分解成乘积形式，因此无法使用 diophantine 进行测试
    assert diop_solve(x*y + 2*y*z) == (-2*p*q, -n1*p**2 + p**2, p*q)
# 定义一个测试函数，用于测试 `square_factor` 函数的行为是否符合预期
def test_square_factor():
    # 检查 square_factor(1) 和 square_factor(-1) 的返回值是否都是 1
    assert square_factor(1) == square_factor(-1) == 1
    # 检查 square_factor(0) 的返回值是否为 1
    assert square_factor(0) == 1
    # 检查 square_factor(5) 和 square_factor(-5) 的返回值是否都是 1
    assert square_factor(5) == square_factor(-5) == 1
    # 检查 square_factor(4) 和 square_factor(-4) 的返回值是否都是 2
    assert square_factor(4) == square_factor(-4) == 2
    # 检查 square_factor(12) 和 square_factor(-12) 的返回值是否都是 2
    assert square_factor(12) == square_factor(-12) == 2
    # 检查 square_factor(6) 的返回值是否为 1
    assert square_factor(6) == 1
    # 检查 square_factor(18) 的返回值是否为 3
    assert square_factor(18) == 3
    # 检查 square_factor(52) 的返回值是否为 2
    assert square_factor(52) == 2
    # 检查 square_factor(49) 的返回值是否为 7
    assert square_factor(49) == 7
    # 检查 square_factor(392) 的返回值是否为 14
    assert square_factor(392) == 14
    # 检查 square_factor(factorint(-12)) 的返回值是否为 2

# 定义一个测试函数，用于测试 `parametrize_ternary_quadratic` 函数的行为是否符合预期
def test_parametrize_ternary_quadratic():
    # 检查各个方程式在 `check_solutions` 下的解是否为真
    assert check_solutions(x**2 + y**2 - z**2)
    assert check_solutions(x**2 + 2*x*y + z**2)
    assert check_solutions(234*x**2 - 65601*y**2 - z**2)
    assert check_solutions(3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z)
    assert check_solutions(x**2 - y**2 - z**2)
    assert check_solutions(x**2 - 49*y**2 - z**2 + 13*z*y - 8*x*y)
    assert check_solutions(8*x*y + z**2)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)
    assert check_solutions(236*x**2 - 225*y**2 - 11*x*y - 13*y*z - 17*x*z)
    assert check_solutions(90*x**2 + 3*y**2 + 5*x*y + 2*z*y + 5*x*z)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)

# 定义一个测试函数，用于测试 `no_square_ternary_quadratic` 函数的行为是否符合预期
def test_no_square_ternary_quadratic():
    # 检查各个方程式在 `check_solutions` 下的解是否为真
    assert check_solutions(2*x*y + y*z - 3*x*z)
    assert check_solutions(189*x*y - 345*y*z - 12*x*z)
    assert check_solutions(23*x*y + 34*y*z)
    assert check_solutions(x*y + y*z + z*x)
    assert check_solutions(23*x*y + 23*y*z + 23*x*z)

# 定义一个测试函数，用于测试 `descent` 函数的行为是否符合预期
def test_descent():
    # 遍历给定的测试数据，检查 `descent` 函数返回的值是否符合预期
    u = ([(13, 23), (3, -11), (41, -113), (91, -3), (1, 1), (1, -1), (17, 13), (123689, 1), (19, -570)])
    for a, b in u:
        w, x, y = descent(a, b)
        assert a*x**2 + b*y**2 == w**2
    # 检查对于不良输入的异常处理，这些异常被期望抛出
    raises(TypeError, lambda: descent(-1, -3))
    raises(ZeroDivisionError, lambda: descent(0, 3))
    raises(TypeError, lambda: descent(4, 3))

# 定义一个测试函数，用于测试 `diophantine` 函数的行为是否符合预期
def test_diophantine():
    # 检查各个方程式在 `check_solutions` 下的解是否为真
    assert check_solutions((x - y)*(y - z)*(z - x))
    assert check_solutions((x - y)*(x**2 + y**2 - z**2))
    assert check_solutions((x - 3*y + 7*z)*(x**2 + y**2 - z**2))
    assert check_solutions(x**2 - 3*y**2 - 1)
    assert check_solutions(y**2 + 7*x*y)
    assert check_solutions(x**2 - 3*x*y + y**2)
    assert check_solutions(z*(x**2 - y**2 - 15))
    assert check_solutions(x*(2*y - 2*z + 5))
    assert check_solutions((x**2 - 3*y**2 - 1)*(x**2 - y**2 - 15))
    assert check_solutions((x**2 - 3*y**2 - 1)*(y - 7*z))
    assert check_solutions((x**2 + y**2 - z**2)*(x - 7*y - 3*z + 4*w))
    # 这个测试案例导致了参数化表示法的问题
    # 但可以通过将 y 因子化来解决，无需使用三元二次方程式方法
    assert check_solutions(y**2 - 7*x*y + 4*y*z)
    assert check_solutions(x**2 - 2*x + 1)
    # 检查等式转换是否与 `diophantine` 的行为一致
    assert diophantine(x - y) == diophantine(Eq(x, y))
    # 18196
    eq = x**4 + y**4 - 97
    # 断言：检查对等式 eq 的迪菲兰方程解是否与其负相等
    assert diophantine(eq, permute=True) == diophantine(-eq, permute=True)
    
    # 断言：检查对于表达式 3*x*pi - 2*y*pi 的迪菲兰方程解是否为 {(2*t_0, 3*t_0)}
    assert diophantine(3*x*pi - 2*y*pi) == {(2*t_0, 3*t_0)}
    
    # 定义方程 eq = x**2 + y**2 + z**2 - 14
    eq = x**2 + y**2 + z**2 - 14
    
    # 定义基本解 base_sol = {(1, 2, 3)}
    base_sol = {(1, 2, 3)}
    
    # 断言：检查方程 eq 的迪菲兰方程解是否等于 base_sol
    assert diophantine(eq) == base_sol
    
    # 计算完整解集合，使用 signed_permutations 函数处理 base_sol 中的一个元组
    complete_soln = set(signed_permutations(base_sol.pop()))
    
    # 断言：检查方程 eq 的迪菲兰方程解是否等于 complete_soln，考虑排列的影响
    assert diophantine(eq, permute=True) == complete_soln
    
    # 断言：检查对于方程 x**2 + x*Rational(15, 14) - 3 的迪菲兰方程解是否为空集
    assert diophantine(x**2 + x*Rational(15, 14) - 3) == set()
    
    # 测试问题 11049
    eq = 92*x**2 - 99*y**2 - z**2
    coeff = eq.as_coefficients_dict()
    
    # 断言：检查通过系数字典 coeff 求解方程 (x, y, z) 的特定三元二次型正常解是否为 {(9, 7, 51)}
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(9, 7, 51)}
    
    # 断言：检查方程 eq 的迪菲兰方程解是否等于给定的集合
    assert diophantine(eq) == {
        (891*p**2 + 9*q**2, -693*p**2 - 102*p*q + 7*q**2,
         5049*p**2 - 1386*p*q - 51*q**2)}
    
    # 定义方程 eq = 2*x**2 + 2*y**2 - z**2
    eq = 2*x**2 + 2*y**2 - z**2
    coeff = eq.as_coefficients_dict()
    
    # 断言：检查通过系数字典 coeff 求解方程 (x, y, z) 的特定三元二次型正常解是否为 {(1, 1, 2)}
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(1, 1, 2)}
    
    # 断言：检查方程 eq 的迪菲兰方程解是否等于给定的集合
    assert diophantine(eq) == {
        (2*p**2 - q**2, -2*p**2 + 4*p*q - q**2,
         4*p**2 - 4*p*q + 2*q**2)}
    
    # 定义方程 eq = 411*x**2+57*y**2-221*z**2
    eq = 411*x**2 + 57*y**2 - 221*z**2
    coeff = eq.as_coefficients_dict()
    
    # 断言：检查通过系数字典 coeff 求解方程 (x, y, z) 的特定三元二次型正常解是否为 {(2021, 2645, 3066)}
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(2021, 2645, 3066)}
    
    # 断言：检查方程 eq 的迪菲兰方程解是否等于给定的集合
    assert diophantine(eq) == {
        (115197*p**2 - 446641*q**2, -150765*p**2 + 1355172*p*q -
         584545*q**2, 174762*p**2 - 301530*p*q + 677586*q**2)}
    
    # 定义方程 eq = 573*x**2+267*y**2-984*z**2
    eq = 573*x**2 + 267*y**2 - 984*z**2
    coeff = eq.as_coefficients_dict()
    
    # 断言：检查通过系数字典 coeff 求解方程 (x, y, z) 的特定三元二次型正常解是否为 {(49, 233, 127)}
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(49, 233, 127)}
    
    # 断言：检查方程 eq 的迪菲兰方程解是否等于给定的集合
    assert diophantine(eq) == {
        (4361*p**2 - 16072*q**2, -20737*p**2 + 83312*p*q - 76424*q**2,
         11303*p**2 - 41474*p*q + 41656*q**2)}
    
    # 定义方程 eq = x**2 + 3*y**2 - 12*z**2
    eq = x**2 + 3*y**2 - 12*z**2
    coeff = eq.as_coefficients_dict()
    
    # 断言：检查通过系数字典 coeff 求解方程 (x, y, z) 的特定三元二次型正常解是否为 {(0, 2, 1)}
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == {(0, 2, 1)}
    
    # 断言：检查方程 eq 的迪菲兰方程解是否等于给定的集合
    assert diophantine(eq) == {
        (24*p*q, 2*p**2 - 24*q**2, p**2 + 12*q**2)}
    
    # 断言：检查求解 x*y**2 + 1 是否引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: diophantine(x*y**2 + 1))
    
    # 断言：检查对于方程 1/x 的迪菲兰方程解是否为空集
    assert diophantine(1/x) == set()
    
    # 断言：检查对于方程 1/x + 1/y - S.Half 的迪菲兰方程解是否为给定集合
    assert diophantine(1/x + 1/y - S.Half) == {(6, 3), (-2, 1), (4, 4), (1, -2), (3, 6)}
    
    # 断言：检查对于方程 x**2 + y**2 +3*x- 5 的迪菲兰方程解是否为给定集合，考虑排列的影响
    assert diophantine(x**2 + y**2 + 3*x - 5, permute=True) == {
        (-1, 1), (-4, -1), (1, -1), (1, 1), (-4, 1), (-1, -1), (4, 1), (4, -1)}
    
    # 测试问题 18186
    assert diophantine(y**4 + x**4 - 2**4 - 3**4, syms=(x, y), permute=True) == {
        (-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}
    
    # 断言：检查对于问题 18122 中的方程 x**2 - y 的解是否为真
    assert check_solutions(x**2 - y)
    
    # 断言：检查对于问题 18122 中的方程 y**2 - x 的解是否为真
    assert check_solutions(y**2 - x)
    
    # 断言：检查方程 (x**2 - y) 的迪菲兰方程解是否为 {(t, t**2)}
    assert diophantine((x**2 - y), t) == {(t,
# 定义一个测试函数，用于验证通用的勾股数问题
def test_general_pythagorean():
    # 从 sympy.abc 导入符号 a, b, c, d, e
    from sympy.abc import a, b, c, d, e

    # 断言检查 a**2 + b**2 + c**2 - d**2 的解
    assert check_solutions(a**2 + b**2 + c**2 - d**2)
    # 断言检查 a**2 + 4*b**2 + 4*c**2 - d**2 的解
    assert check_solutions(a**2 + 4*b**2 + 4*c**2 - d**2)
    # 断言检查 9*a**2 + 4*b**2 + 4*c**2 - d**2 的解
    assert check_solutions(9*a**2 + 4*b**2 + 4*c**2 - d**2)
    # 断言检查 9*a**2 + 4*b**2 - 25*d**2 + 4*c**2 的解
    assert check_solutions(9*a**2 + 4*b**2 - 25*d**2 + 4*c**2)
    # 断言检查 9*a**2 - 16*d**2 + 4*b**2 + 4*c**2 的解
    assert check_solutions(9*a**2 - 16*d**2 + 4*b**2 + 4*c**2)
    # 断言检查 -e**2 + 9*a**2 + 4*b**2 + 4*c**2 + 25*d**2 的解
    assert check_solutions(-e**2 + 9*a**2 + 4*b**2 + 4*c**2 + 25*d**2)
    # 断言检查 16*a**2 - b**2 + 9*c**2 + d**2 + 25*e**2 的解
    assert check_solutions(16*a**2 - b**2 + 9*c**2 + d**2 + 25*e**2)

    # 断言验证 GeneralPythagorean(a**2 + b**2 + c**2 - d**2) 的解
    assert GeneralPythagorean(a**2 + b**2 + c**2 - d**2).solve(parameters=[x, y, z]) == \
           {(x**2 + y**2 - z**2, 2*x*z, 2*y*z, x**2 + y**2 + z**2)}


# 定义一个测试函数，用于快速检查通用的平方和问题
def test_diop_general_sum_of_squares_quick():
    # 对于范围在 [3, 10) 内的每个整数 i
    for i in range(3, 10):
        # 断言检查 sum(i**2 for i in symbols(':%i' % i)) - i 的解
        assert check_solutions(sum(i**2 for i in symbols(':%i' % i)) - i)

    # 断言检查 diop_general_sum_of_squares(x**2 + y**2 - 2) 的结果为 None
    assert diop_general_sum_of_squares(x**2 + y**2 - 2) is None
    # 断言检查 diop_general_sum_of_squares(x**2 + y**2 + z**2 + 2) 的解为空集
    assert diop_general_sum_of_squares(x**2 + y**2 + z**2 + 2) == set()
    # 构建方程 eq = x**2 + y**2 + z**2 - (1 + 4 + 9)
    eq = x**2 + y**2 + z**2 - (1 + 4 + 9)
    # 断言检查 diop_general_sum_of_squares(eq) 的解为 {(1, 2, 3)}
    assert diop_general_sum_of_squares(eq) == \
           {(1, 2, 3)}
    # 构建方程 eq = u**2 + v**2 + x**2 + y**2 + z**2 - 1313
    eq = u**2 + v**2 + x**2 + y**2 + z**2 - 1313
    # 断言检查 diophantine(eq) 的解数目为 3
    assert len(diophantine(eq, 3)) == 3
    # 定义变量 var，包含正负符号的符号列表
    var = symbols(':5') + (symbols('6', negative=True),)
    # 构建方程 eq = Add(*[i**2 for i in var]) - 112
    eq = Add(*[i**2 for i in var]) - 112

    # 断言检查 diophantine(eq) 的解与 base_soln 匹配
    base_soln = {(0, 1, 1, 5, 6, -7), (1, 1, 1, 3, 6, -8), (2, 3, 3, 4, 5, -7), (0, 1, 1, 1, 3, -10),
                 (0, 0, 4, 4, 4, -8), (1, 2, 3, 3, 5, -8), (0, 1, 2, 3, 7, -7), (2, 2, 4, 4, 6, -6),
                 (1, 1, 3, 4, 6, -7), (0, 2, 3, 3, 3, -9), (0, 0, 2, 2, 2, -10), (1, 1, 2, 3, 4, -9),
                 (0, 1, 1, 2, 5, -9), (0, 0, 2, 6, 6, -6), (1, 3, 4, 5, 5, -6), (0, 2, 2, 2, 6, -8),
                 (0, 3, 3, 3, 6, -7), (0, 2, 3, 5, 5, -7), (0, 1, 5, 5, 5, -6)}
    assert diophantine(eq) == base_soln
    # 断言检查 diophantine(eq, permute=True) 的解数目为 196800
    assert len(diophantine(eq, permute=True)) == 196800

    # 使用 signsimp 处理带有负号平方项的方程
    assert diophantine(12 - x**2 - y**2 - z**2) == {(2, 2, 2)}
    # 断言检查 classify_diop(-eq) 引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: classify_diop(-eq))


# 定义一个测试函数，用于解决 issue 23807
def test_issue_23807():
    # 构建方程 eq = x**2 + y**2 + z**2 - 1000000
    eq = x**2 + y**2 + z**2 - 1000000
    # base_soln 是方程 eq 的已知解集合
    base_soln = {(0, 0, 1000), (0, 352, 936), (480, 600, 640), (24, 640, 768), (192, 640, 744),
                 (192, 480, 856), (168, 224, 960), (0, 600, 800), (280, 576, 768), (152, 480, 864),
                 (0, 280, 960), (352, 360, 864), (424, 480, 768), (360, 480, 800), (224, 600, 768),
                 (96, 360, 928), (168, 576, 800), (96, 480, 872)}

    # 断言检查 diophantine(eq) 的解与 base_soln 匹配
    assert diophantine(eq) == base_soln


# 定义一个测试函数，用于测试分区函数 partition
def test_diop_partition():
    # 对于 n 在 [8, 10] 范围内的每个整数
    for n in [8, 10]:
        # 对于 k 在 [1, 8) 范围内的每个整数
        for k in range(1, 8):
            # 对于 partition(n, k) 返回的每个分区 p
            for p in partition(n, k):
                # 断言检查每个分区 p 的长度为 k
                assert len(p
    # 断言：验证 partition 函数对于给定参数的返回结果是否符合预期
    assert [list(p) for p in partition(3, 5, 1)] == [
        [0, 0, 0, 0, 3], [0, 0, 0, 1, 2], [0, 0, 1, 1, 1]]
    # 断言：验证 partition 函数对于空参数的返回结果是否为一个空元组
    assert list(partition(0)) == [()]
    # 断言：验证 partition 函数对于参数 (1, 0) 的返回结果是否为一个空元组
    assert list(partition(1, 0)) == [()]
    # 断言：验证 partition 函数对于单个参数 3 的返回结果是否符合预期
    assert [list(i) for i in partition(3)] == [[1, 1, 1], [1, 2], [3]]
def test_prime_as_sum_of_two_squares():
    # 对于每个指定的整数进行测试
    for i in [5, 13, 17, 29, 37, 41, 2341, 3557, 34841, 64601]:
        # 调用函数 prime_as_sum_of_two_squares，返回两个整数 a 和 b，使得 a^2 + b^2 == i
        a, b = prime_as_sum_of_two_squares(i)
        # 断言 a^2 + b^2 == i，即验证返回的两个整数确实满足条件
        assert a**2 + b**2 == i
    # 断言 prime_as_sum_of_two_squares(7) 返回 None
    assert prime_as_sum_of_two_squares(7) is None
    # 调用函数 prime_as_sum_of_two_squares(800029)，并断言返回的结果是 (450, 773)，并且返回的类型是元组
    ans = prime_as_sum_of_two_squares(800029)
    assert ans == (450, 773) and type(ans[0]) is int


def test_sum_of_three_squares():
    # 对于每个指定的整数进行测试
    for i in [0, 1, 2, 34, 123, 34304595905, 34304595905394941, 343045959052344,
              800, 801, 802, 803, 804, 805, 806]:
        # 调用函数 sum_of_three_squares，返回三个整数 a, b, c，使得 a^2 + b^2 + c^2 == i
        a, b, c = sum_of_three_squares(i)
        # 断言 a^2 + b^2 + c^2 == i，即验证返回的三个整数确实满足条件
        assert a**2 + b**2 + c**2 == i
        # 断言 a 大于等于 0
        assert a >= 0

    # 抛出 ValueError 错误，因为参数是负数
    raises(ValueError, lambda: sum_of_three_squares(-1))

    # 断言 sum_of_three_squares(7) 返回 None
    assert sum_of_three_squares(7) is None
    # 断言 sum_of_three_squares((4**5)*15) 返回 None
    assert sum_of_three_squares((4**5)*15) is None
    # 断言 sum_of_three_squares(25) 返回 (0, 0, 5)，即 25 的三个平方数和为 25
    assert sum_of_three_squares(25) == (0, 0, 5)
    # 断言 sum_of_three_squares(4) 返回 (0, 0, 2)，即 4 的三个平方数和为 4
    assert sum_of_three_squares(4) == (0, 0, 2)


def test_sum_of_four_squares():
    from sympy.core.random import randint

    # 断言对于随机生成的整数 n，sum(i**2 for i in sum_of_four_squares(n)) 的和等于 n
    n = randint(1, 100000000000000)
    assert sum(i**2 for i in sum_of_four_squares(n)) == n

    # 抛出 ValueError 错误，因为参数是负数
    raises(ValueError, lambda: sum_of_four_squares(-1))

    # 对于范围内的每个整数 n 进行测试
    for n in range(1000):
        # 调用函数 sum_of_four_squares(n)，返回四个整数组成的列表 result
        result = sum_of_four_squares(n)
        # 断言 result 的长度为 4
        assert len(result) == 4
        # 断言 result 中的每个整数都大于等于 0
        assert all(r >= 0 for r in result)
        # 断言 result 中四个整数的平方和等于 n
        assert sum(r**2 for r in result) == n
        # 断言 result 是按升序排列的
        assert list(result) == sorted(result)


def test_power_representation():
    # 定义多个测试用例，每个测试用例包括整数 n、p、k 和一个期望的生成器 f
    tests = [(1729, 3, 2), (234, 2, 4), (2, 1, 2), (3, 1, 3), (5, 2, 2), (12352, 2, 4),
             (32760, 2, 3)]

    # 对于每个测试用例进行测试
    for test in tests:
        n, p, k = test
        # 调用函数 power_representation(n, p, k)，返回生成器 f
        f = power_representation(n, p, k)

        # 循环直到生成器 f 抛出 StopIteration 异常
        while True:
            try:
                l = next(f)
                # 断言生成器返回的列表 l 的长度等于 k
                assert len(l) == k

                # 计算 l 中每个元素的 p 次幂之和，断言其等于 n
                chk_sum = 0
                for l_i in l:
                    chk_sum = chk_sum + l_i**p
                assert chk_sum == n

            except StopIteration:
                break

    # 断言调用 power_representation(20, 2, 4, True) 返回的列表与期望的结果匹配
    assert list(power_representation(20, 2, 4, True)) == [(1, 1, 3, 3), (0, 0, 2, 4)]
    # 抛出 ValueError 错误，因为参数 n 不是整数
    raises(ValueError, lambda: list(power_representation(1.2, 2, 2)))
    # 抛出 ValueError 错误，因为参数 p 为 0
    raises(ValueError, lambda: list(power_representation(2, 0, 2)))
    # 抛出 ValueError 错误，因为参数 k 为 0
    raises(ValueError, lambda: list(power_representation(2, 2, 0)))
    # 断言调用 power_representation(-1, 2, 2) 返回空列表，因为参数 n 是负数
    assert list(power_representation(-1, 2, 2)) == []
    # 断言调用 power_representation(1, 1, 1) 返回 [(1,)]，因为参数 n 等于 1
    assert list(power_representation(1, 1, 1)) == [(1,)]
    # 断言调用 power_representation(3, 2, 1) 返回空列表，因为无法表示为一个整数的两个平方数之和
    assert list(power_representation(3, 2, 1)) == []
    # 断言调用 power_representation(4, 2, 1) 返回 [(2,)]，因为参数 n 等于 4
    assert list(power_representation(4, 2, 1)) == [(2,)]
    # 断言调用 power_representation(3**4, 4, 6, zeros=True) 返回 [(1, 2, 2, 2, 2, 2), (0, 0, 0, 0, 0, 3)]
    assert list(power_representation(3**4, 4, 6, zeros=True)) == [(1, 2, 2, 2, 2, 2), (0, 0, 0, 0, 0, 3)]
    # 断言调用 power_representation(3**4, 4, 5, zeros=False) 返回空列表，因为无法找到满足条件的组合
    assert list(power_representation(3**4, 4, 5, zeros=False)) == []
    # 断言调用 power_representation(-2, 3, 2) 返回 [(-1, -1)]，因为参数 n 是负数
    assert list(power_representation(-2, 3, 2)) == [(-1, -1)]
    # 断言调用 power_representation(-2, 4, 2) 返回空列表，因为无法找到满足条件的组合
    assert list(power_representation(-2, 4, 2)) == []
    # 断言调用 power_representation(0, 3, 2, True) 返回 [(0, 0)]，因为参数 n 等于 0
    assert list(power_representation(0, 3, 2, True)) == [(0, 0)]
    # 断言：验证调用 power_representation 函数并期望返回空列表，表示没有符合条件的结果
    assert list(power_representation(0, 3, 2, False)) == []
    
    # 断言：验证调用 power_representation 函数并期望返回的列表长度为0，用于检查是否有符合条件的结果
    # 当处理平方数时，进行可行性检查
    assert len(list(power_representation(4**10*(8*10 + 7), 2, 3))) == 0
    
    # 设置一个大整数常量
    big = 2**30
    
    # 遍历一系列整数并进行断言：验证调用 sum_of_powers 函数并期望返回空列表，以确保没有符合条件的结果
    # 如果这些条件不被识别，可能会导致递归错误
    for i in [13, 10, 7, 5, 4, 2, 1]:
        assert list(sum_of_powers(big, 2, big - i)) == []
def test_assumptions():
    """
    Test whether diophantine respects the assumptions.
    """
    # 从以下 Stack Overflow 问题中取得测试用例，探讨 diophantine 模块中的假设条件
    # https://stackoverflow.com/questions/23301941/how-can-i-declare-natural-symbols-with-sympy
    m, n = symbols('m n', integer=True, positive=True)
    # 调用 diophantine 函数计算方程 n**2 + m*n - 500 的解集
    diof = diophantine(n**2 + m*n - 500)
    # 断言解集是否等于预期的集合
    assert diof == {(5, 20), (40, 10), (95, 5), (121, 4), (248, 2), (499, 1)}

    a, b = symbols('a b', integer=True, positive=False)
    # 再次调用 diophantine 函数计算方程 a*b + 2*a + 3*b - 6 的解集
    diof = diophantine(a*b + 2*a + 3*b - 6)
    # 断言解集是否等于预期的集合
    assert diof == {(-15, -3), (-9, -4), (-7, -5), (-6, -6), (-5, -8), (-4, -14)}


def check_solutions(eq):
    """
    Determines whether solutions returned by diophantine() satisfy the original
    equation. Hope to generalize this so we can remove functions like check_ternay_quadratic,
    check_solutions_normal, check_solutions()
    """
    s = diophantine(eq)

    factors = Mul.make_args(eq)

    var = list(eq.free_symbols)
    var.sort(key=default_sort_key)

    while s:
        solution = s.pop()
        for f in factors:
            # 使用 diop_simplify 函数简化并检查方程的解是否满足原始方程
            if diop_simplify(f.subs(zip(var, solution))) == 0:
                break
        else:
            return False
    return True


def test_diopcoverage():
    eq = (2*x + y + 1)**2
    # 断言 diop_solve 函数能够正确解决方程 (2*x + y + 1)**2
    assert diop_solve(eq) == {(t_0, -2*t_0 - 1)}
    eq = 2*x**2 + 6*x*y + 12*x + 4*y**2 + 18*y + 18
    # 断言 diop_solve 函数能够正确解决方程 2*x**2 + 6*x*y + 12*x + 4*y**2 + 18*y + 18
    assert diop_solve(eq) == {(t, -t - 3), (-2*t - 3, t)}
    # 断言 diop_quadratic 函数能够正确解决方程 x + y**2 - 3
    assert diop_quadratic(x + y**2 - 3) == {(-t**2 + 3, t)}

    # 断言 diop_linear 函数能够正确解决方程 x + y - 3
    assert diop_linear(x + y - 3) == (t_0, 3 - t_0)

    # 断言 base_solution_linear 函数的正确性
    assert base_solution_linear(0, 1, 2, t=None) == (0, 0)
    ans = (3*t - 1, -2*t + 1)
    assert base_solution_linear(4, 8, 12, t) == ans
    assert base_solution_linear(4, 8, 12, t=None) == tuple(_.subs(t, 0) for _ in ans)

    # 断言 cornacchia 函数的正确性
    assert cornacchia(1, 1, 20) == set()
    assert cornacchia(1, 1, 5) == {(2, 1)}
    assert cornacchia(1, 2, 17) == {(3, 2)}

    # 断言 reconstruct 函数抛出 ValueError 异常
    raises(ValueError, lambda: reconstruct(4, 20, 1))

    # 断言 gaussian_reduce 函数的正确性
    assert gaussian_reduce(4, 1, 3) == (1, 1)
    eq = -w**2 - x**2 - y**2 + z**2

    # 断言 diop_general_pythagorean 函数的正确性
    assert diop_general_pythagorean(eq) == \
        diop_general_pythagorean(-eq) == \
            (m1**2 + m2**2 - m3**2, 2*m1*m3,
            2*m2*m3, m1**2 + m2**2 + m3**2)

    # 断言 check_param 函数的正确性
    assert len(check_param(S(3) + x/3, S(4) + x/2, S(2), [x])) == 0
    assert len(check_param(Rational(3, 2), S(4) + x, S(2), [x])) == 0
    assert len(check_param(S(4) + x, Rational(3, 2), S(2), [x])) == 0

    # 断言 _nint_or_floor 函数的正确性
    assert _nint_or_floor(16, 10) == 2
    assert _odd(1) == (not _even(1)) == True
    assert _odd(0) == (not _even(0)) == False
    assert _remove_gcd(2, 4, 6) == (1, 2, 3)
    raises(TypeError, lambda: _remove_gcd((2, 4, 6)))
    # 断言 sqf_normal 函数的正确性
    assert sqf_normal(2*3**2*5, 2*5*11, 2*7**2*11)  == \
        (11, 1, 5)

    # 断言以下方程目前尚未实现求解器，应抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: diophantine(x**2 + y**2 + x*y + 2*y*z - 12))
    raises(NotImplementedError, lambda: diophantine(x**3 + y**2))
    # 调用 diop_quadratic 函数来解方程 x**2 + y**2 - 1**2 - 3**4 = 0，并断言其返回的结果是否等于集合 {(-9, -1), (-9, 1), (-1, -9), (-1, 9), (1, -9), (1, 9), (9, -1), (9, 1)}
    assert diop_quadratic(x**2 + y**2 - 1**2 - 3**4) == \
           {(-9, -1), (-9, 1), (-1, -9), (-1, 9), (1, -9), (1, 9), (9, -1), (9, 1)}
def test_sum_of_squares_powers():
    tru = {(0, 0, 1, 1, 11), (0, 0, 5, 7, 7), (0, 1, 3, 7, 8), (0, 1, 4, 5, 9), (0, 3, 4, 7, 7), (0, 3, 5, 5, 8),
           (1, 1, 2, 6, 9), (1, 1, 6, 6, 7), (1, 2, 3, 3, 10), (1, 3, 4, 4, 9), (1, 5, 5, 6, 6), (2, 2, 3, 5, 9),
           (2, 3, 5, 6, 7), (3, 3, 4, 5, 8)}
    # 定义方程 u^2 + v^2 + x^2 + y^2 + z^2 - 123
    eq = u**2 + v**2 + x**2 + y**2 + z**2 - 123
    # 使用 diop_general_sum_of_squares 函数求解方程，允许使用无限大 oo
    ans = diop_general_sum_of_squares(eq, oo)
    # 断言结果集合的长度为 14
    assert len(ans) == 14
    # 断言求解得到的结果集合与预期结果 tru 相等
    assert ans == tru

    # 使用 lambda 函数调用 sum_of_squares 函数并断言引发 ValueError 异常
    raises(ValueError, lambda: list(sum_of_squares(10, -1)))
    # 断言 sum_of_squares(1, 1) 返回 [(1,)]
    assert list(sum_of_squares(1, 1)) == [(1,)]
    # 断言 sum_of_squares(1, 2) 返回空列表
    assert list(sum_of_squares(1, 2)) == []
    # 断言 sum_of_squares(1, 2, True) 返回 [(0, 1)]
    assert list(sum_of_squares(1, 2, True)) == [(0, 1)]
    # 断言 sum_of_squares(-10, 2) 返回空列表
    assert list(sum_of_squares(-10, 2)) == []
    # 断言 sum_of_squares(2, 3) 返回空列表
    assert list(sum_of_squares(2, 3)) == []
    # 断言 sum_of_squares(0, 3, True) 返回 [(0, 0, 0)]
    assert list(sum_of_squares(0, 3, True)) == [(0, 0, 0)]
    # 断言 sum_of_squares(0, 3) 返回空列表
    assert list(sum_of_squares(0, 3)) == []
    # 断言 sum_of_squares(4, 1) 返回 [(2,)]
    assert list(sum_of_squares(4, 1)) == [(2,)]
    # 断言 sum_of_squares(5, 1) 返回空列表
    assert list(sum_of_squares(5, 1)) == []
    # 断言 sum_of_squares(50, 2) 返回 [(5, 5), (1, 7)]
    assert list(sum_of_squares(50, 2)) == [(5, 5), (1, 7)]
    # 断言 sum_of_squares(11, 5, True) 返回 [(1, 1, 1, 2, 2), (0, 0, 1, 1, 3)]
    assert list(sum_of_squares(11, 5, True)) == [(1, 1, 1, 2, 2), (0, 0, 1, 1, 3)]
    # 断言 sum_of_squares(8, 8) 返回 [(1, 1, 1, 1, 1, 1, 1, 1)]
    assert list(sum_of_squares(8, 8)) == [(1, 1, 1, 1, 1, 1, 1, 1)]
    # 第一个断言：验证 sum_of_squares 函数对于每个 i 的输出长度与预期列表中的长度相等
    assert [len(list(sum_of_squares(i, 5, True))) for i in range(30)] == [
        1, 1, 1, 1, 2,    # i = 0~4
        2, 1, 1, 2, 2,    # i = 5~9
        2, 2, 2, 3, 2,    # i = 10~14
        1, 3, 3, 3, 3,    # i = 15~19
        4, 3, 3, 2, 2,    # i = 20~24
        4, 4, 4, 4, 5]    # i = 25~29
    
    # 第二个断言：验证 sum_of_squares 函数对于每个 i 的输出长度与预期列表中的长度相等
    assert [len(list(sum_of_squares(i, 5))) for i in range(30)] == [
        0, 0, 0, 0, 0,    # i = 0~4
        1, 0, 0, 1, 0,    # i = 5~9
        0, 1, 0, 1, 1,    # i = 10~14
        0, 1, 1, 0, 1,    # i = 15~19
        2, 1, 1, 1, 1,    # i = 20~24
        1, 1, 1, 1, 3]    # i = 25~29
    
    # 对于每个 i，分别进行以下断言：
    for i in range(30):
        # 使用 sum_of_squares 函数生成集合 s1，并验证所有 t 中的 j 的平方和等于 i
        s1 = set(sum_of_squares(i, 5, True))
        assert not s1 or all(sum(j**2 for j in t) == i for t in s1)
        
        # 使用 sum_of_squares 函数生成集合 s2，并验证所有 t 中的 j 的平方和等于 i
        s2 = set(sum_of_squares(i, 5))
        assert all(sum(j**2 for j in t) == i for t in s2)
    
    # 确保 sum_of_powers 函数能够正确地抛出 ValueError 异常
    raises(ValueError, lambda: list(sum_of_powers(2, -1, 1)))
    raises(ValueError, lambda: list(sum_of_powers(2, 1, -1)))
    
    # 验证 sum_of_powers 函数的各种输入与预期输出是否匹配
    assert list(sum_of_powers(-2, 3, 2)) == [(-1, -1)]
    assert list(sum_of_powers(-2, 4, 2)) == []
    assert list(sum_of_powers(2, 1, 1)) == [(2,)]
    assert list(sum_of_powers(2, 1, 3, True)) == [(0, 0, 2), (0, 1, 1)]
    assert list(sum_of_powers(5, 1, 2, True)) == [(0, 5), (1, 4), (2, 3)]
    assert list(sum_of_powers(6, 2, 2)) == []
    assert list(sum_of_powers(3**5, 3, 1)) == []
    assert list(sum_of_powers(3**6, 3, 1)) == [(9,)] and (9**3 == 3**6)
    assert list(sum_of_powers(2**1000, 5, 2)) == []
# 定义测试函数 test__can_do_sum_of_squares
def test__can_do_sum_of_squares():
    # 断言 _can_do_sum_of_squares(3, -1) 返回 False
    assert _can_do_sum_of_squares(3, -1) is False
    # 断言 _can_do_sum_of_squares(-3, 1) 返回 False
    assert _can_do_sum_of_squares(-3, 1) is False
    # 断言 _can_do_sum_of_squares(0, 1) 返回 True
    assert _can_do_sum_of_squares(0, 1)
    # 断言 _can_do_sum_of_squares(4, 1) 返回 True
    assert _can_do_sum_of_squares(4, 1)
    # 断言 _can_do_sum_of_squares(1, 2) 返回 True
    assert _can_do_sum_of_squares(1, 2)
    # 断言 _can_do_sum_of_squares(2, 2) 返回 True
    assert _can_do_sum_of_squares(2, 2)
    # 断言 _can_do_sum_of_squares(3, 2) 返回 False
    assert _can_do_sum_of_squares(3, 2) is False


# 定义测试函数 test_diophantine_permute_sign
def test_diophantine_permute_sign():
    # 导入 sympy.abc 中的符号 a, b, c, d, e
    from sympy.abc import a, b, c, d, e
    # 设置方程 eq = a**4 + b**4 - (2**4 + 3**4)
    eq = a**4 + b**4 - (2**4 + 3**4)
    # 定义基本解 base_sol 为 {(2, 3)}
    base_sol = {(2, 3)}
    # 断言 diophantine(eq) 返回 base_sol
    assert diophantine(eq) == base_sol
    # 对基本解 base_sol 中的元素进行符号排列组合，形成 complete_soln 集合
    complete_soln = set(signed_permutations(base_sol.pop()))
    # 断言带有符号排列的 diophantine(eq, permute=True) 返回 complete_soln
    assert diophantine(eq, permute=True) == complete_soln

    # 设置方程 eq = a**2 + b**2 + c**2 + d**2 + e**2 - 234
    eq = a**2 + b**2 + c**2 + d**2 + e**2 - 234
    # 断言 diophantine(eq) 的结果集长度为 35
    assert len(diophantine(eq)) == 35
    # 断言带有符号排列的 diophantine(eq, permute=True) 的结果集长度为 62000
    assert len(diophantine(eq, permute=True)) == 62000
    # 设置解集 soln 为 {(-1, -1), (-1, 2), (1, -2), (1, 1)}
    soln = {(-1, -1), (-1, 2), (1, -2), (1, 1)}
    # 断言带有符号排列的 diophantine(10*x**2 + 12*x*y + 12*y**2 - 34, permute=True) 返回 soln
    assert diophantine(10*x**2 + 12*x*y + 12*y**2 - 34, permute=True) == soln


# 标记为预期失败的测试函数 test_not_implemented
@XFAIL
def test_not_implemented():
    # 设置方程 eq = x**2 + y**4 - 1**2 - 3**4
    eq = x**2 + y**4 - 1**2 - 3**4
    # 断言 diophantine(eq, syms=[x, y]) 返回 {(9, 1), (1, 3)}
    assert diophantine(eq, syms=[x, y]) == {(9, 1), (1, 3)}


# 定义测试函数 test_issue_9538
def test_issue_9538():
    # 设置方程 eq = x - 3*y + 2
    eq = x - 3*y + 2
    # 断言 diophantine(eq, syms=[y,x]) 返回 {(t_0, 3*t_0 - 2)}
    assert diophantine(eq, syms=[y,x]) == {(t_0, 3*t_0 - 2)}
    # 断言调用 diophantine(eq, syms={y, x}) 时抛出 TypeError 异常
    raises(TypeError, lambda: diophantine(eq, syms={y, x}))


# 定义测试函数 test_ternary_quadratic
def test_ternary_quadratic():
    # 解方程 2*x**2 + y**2 - 2*z**2 的解集 s
    s = diophantine(2*x**2 + y**2 - 2*z**2)
    # 按照顺序获取 s 中的自由符号 p, q, r
    p, q, r = ordered(S(s).free_symbols)
    # 断言 s 等于给定的解集
    assert s == {
        p**2 - 2*q**2,
        -2*p**2 + 4*p*q - 4*p*r - 4*q**2,
        p**2 - 4*p*q + 2*q**2 - 4*q*r
    }

    # 解方程 x**2 + 2*y**2 - 2*z**2 的解集 s
    s = diophantine(x**2 + 2*y**2 - 2*z**2)
    # 断言 s 等于给定的解集
    assert s == {
        (4*p*q, p**2 - 2*q**2, p**2 + 2*q**2)
    }

    # 解方程 2*x**2 + 2*y**2 - z**2 的解集 s
    s = diophantine(2*x**2 + 2*y**2 - z**2)
    # 断言 s 等于给定的解集
    assert s == {
        (2*p**2 - q**2, -2*p**2 + 4*p*q - q**2, 4*p**2 - 4*p*q + 2*q**2)
    }

    # 解方程 3*x**2 + 72*y**2 - 27*z**2 的解集 s
    s = diophantine(3*x**2 + 72*y**2 - 27*z**2)
    # 断言 s 等于给定的解集
    assert s == {
        (24*p**2 - 9*q**2, 6*p*q, 8*p**2 + 3*q**2)
    }

    # 断言 parametrize_ternary_quadratic(3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z) 的返回值
    # 符合预期的形式
    assert parametrize_ternary_quadratic(
        3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z) == (
        2*p**2 - 2*p*q - q**2, 2*p**2 + 2*p*q - q**2, 2*p**2 -
        2*p*q + 3*q**2)

    # 断言 parametrize_ternary_quadratic(124*x**2 - 30*y**2 - 7729*z**2) 的返回值
    # 符合预期的形式
    assert parametrize_ternary_quadratic(
        124*x**2 - 30*y**2 - 7729*z**2) == (
        -1410*p**2 - 363263*q**2, 2700*p**2 + 30916*p*q -
        695610*q**2, -60*p**2 + 5400*p*q + 15458*q**2)


# 定义测试函数 test_diophantine_solution_set
def test_diophantine_solution_set():
    # 创建一个空的 DiophantineSolutionSet 实例 s1
    s1 = DiophantineSolutionSet([], [])
    # 断言 s1 转换为集合后为空集
    assert set(s1) == set()
    # 断言 s1 的符号集合为空元组
    assert s1.symbols == ()
    # 断言 s1 的参数集合为空元组
    assert s1.parameters == ()
    # 尝试向 s1 添加 (x,) 引发 ValueError 异常
    raises(ValueError, lambda: s1.add((x,)))
    # 断言 s1 的迭代器 dict_iterator 返回空
    # 创建 DiophantineSolutionSet 对象 s3，使用变量 [x, y, z] 和参数 [t, u]
    s3 = DiophantineSolutionSet([x, y, z], [t, u])
    # 断言参数的数量为 2
    assert len(s3.parameters) == 2
    # 向 s3 中添加一个三元组作为解
    s3.add((t**2 + u, t - u, 1))
    # 断言 s3 的解集合是否包含指定的三元组
    assert set(s3) == {(t**2 + u, t - u, 1)}
    # 使用 subs 方法替换 t，并断言返回的解集
    assert s3.subs(t, 2) == {(u + 4, 2 - u, 1)}
    # 使用函数调用形式替换 t，并断言返回的解集
    assert s3(2) == {(u + 4, 2 - u, 1)}
    # 使用 subs 方法同时替换 t 和 u，并断言返回的解集
    assert s3.subs({t: 7, u: 8}) == {(57, -1, 1)}
    # 使用函数调用形式同时替换 t 和 u，并断言返回的解集
    assert s3(7, 8) == {(57, -1, 1)}
    # 使用 subs 方法替换 t，并断言返回的解集
    assert s3.subs({t: 5}) == {(u + 25, 5 - u, 1)}
    # 使用函数调用形式替换 t，并断言返回的解集
    assert s3(5) == {(u + 25, 5 - u, 1)}
    # 使用 subs 方法替换 u，并断言返回的解集
    assert s3.subs(u, -3) == {(t**2 - 3, t + 3, 1)}
    # 使用函数调用形式替换 u，并断言返回的解集
    assert s3(None, -3) == {(t**2 - 3, t + 3, 1)}
    # 使用 subs 方法同时替换 t 和 u，并断言返回的解集
    assert s3.subs({t: 2, u: 8}) == {(12, -6, 1)}
    # 使用函数调用形式同时替换 t 和 u，并断言返回的解集
    assert s3(2, 8) == {(12, -6, 1)}
    # 使用 subs 方法同时替换 t 和 u，并断言返回的解集
    assert s3.subs({t: 5, u: -3}) == {(22, 8, 1)}
    # 使用函数调用形式同时替换 t 和 u，并断言返回的解集
    assert s3(5, -3) == {(22, 8, 1)}
    # 使用 lambda 表达式测试 subs 方法抛出 ValueError 异常
    raises(ValueError, lambda: s3.subs(x=1))
    # 使用 lambda 表达式测试 subs 方法抛出 ValueError 异常
    raises(ValueError, lambda: s3.subs(1, 2, 3))
    # 使用 lambda 表达式测试 add 方法抛出 ValueError 异常
    raises(ValueError, lambda: s3.add(()))
    # 使用 lambda 表达式测试 add 方法抛出 ValueError 异常
    raises(ValueError, lambda: s3.add((1, 2, 3, 4)))
    # 使用 lambda 表达式测试 add 方法抛出 ValueError 异常
    raises(ValueError, lambda: s3.add((1, 2)))
    # 使用 lambda 表达式测试函数调用形式抛出 ValueError 异常
    raises(ValueError, lambda: s3(1, 2, 3))
    # 使用 lambda 表达式测试函数调用形式抛出 TypeError 异常
    raises(TypeError, lambda: s3(t=1))

    # 创建 DiophantineSolutionSet 对象 s4，使用变量 [x, y] 和参数 [t, u]
    s4 = DiophantineSolutionSet([x, y], [t, u])
    # 向 s4 中添加两个二元组作为解
    s4.add((t, 11*t))
    s4.add((-t, 22*t))
    # 断言 s4 在参数 (0, 0) 处的解集
    assert s4(0, 0) == {(0, 0)}
# 定义一个测试函数，用于测试二次二进制方程的参数传递功能
def test_quadratic_parameter_passing():
    # 定义一个二次二进制方程
    eq = -33*x*y + 3*y**2
    # 创建二次二进制方程的求解器，并传入参数列表[t, u]
    solution = BinaryQuadratic(eq).solve(parameters=[t, u])
    # 断言，验证参数是否正确传递至最终的解
    assert solution == {(t, 11*t), (t, -22*t)}
    # 断言，验证在参数为(0, 0)时，解是否为{(0, 0)}
    assert solution(0, 0) == {(0, 0)}
```