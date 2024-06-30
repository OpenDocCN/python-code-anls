# `D:\src\scipysrc\sympy\sympy\polys\tests\test_galoistools.py`

```
# 导入 SymPy 中的有限域多项式工具模块
from sympy.polys.galoistools import (
    gf_crt, gf_crt1, gf_crt2, gf_int,  # 导入有限域的 CRT 相关函数
    gf_degree, gf_strip, gf_trunc, gf_normal,  # 导入多项式的度、去除多余项、截断、标准化函数
    gf_from_dict, gf_to_dict,  # 导入多项式与字典之间的转换函数
    gf_from_int_poly, gf_to_int_poly,  # 导入从整数多项式到有限域多项式的转换函数
    gf_neg, gf_add_ground, gf_sub_ground, gf_mul_ground,  # 导入多项式的负、加、减、乘以常数函数
    gf_add, gf_sub, gf_add_mul, gf_sub_mul, gf_mul, gf_sqr,  # 导入多项式的加、减、混合加、混合减、乘、平方函数
    gf_div, gf_rem, gf_quo, gf_exquo,  # 导入多项式的除、取余、商、扩展商函数
    gf_lshift, gf_rshift, gf_expand,  # 导入多项式的左移、右移、展开函数
    gf_pow, gf_pow_mod,  # 导入多项式的幂、模幂函数
    gf_gcdex, gf_gcd, gf_lcm, gf_cofactors,  # 导入多项式的扩展最大公约数、最大公约数、最小公倍数、互素因子函数
    gf_LC, gf_TC, gf_monic,  # 导入多项式的主系数、首项系数、首一化函数
    gf_eval, gf_multi_eval,  # 导入多项式的求值、多重求值函数
    gf_compose, gf_compose_mod,  # 导入多项式的复合、模复合函数
    gf_trace_map,  # 导入多项式的迹映射函数
    gf_diff,  # 导入多项式的微分函数
    gf_irreducible, gf_irreducible_p,  # 导入多项式的不可约性检测函数
    gf_irred_p_ben_or, gf_irred_p_rabin,  # 导入多项式的 Ben-Or、Rabin 算法检测函数
    gf_sqf_list, gf_sqf_part, gf_sqf_p,  # 导入多项式的平方自由分解列表、部分、判断函数
    gf_Qmatrix, gf_Qbasis,  # 导入多项式的 Q 矩阵、Q 基函数
    gf_ddf_zassenhaus, gf_ddf_shoup,  # 导入多项式的 Zassenhaus、Shoup 因式分解函数
    gf_edf_zassenhaus, gf_edf_shoup,  # 导入多项式的扩展 Zassenhaus、Shoup 因式分解函数
    gf_berlekamp,  # 导入多项式的 Berlekamp 算法因式分解函数
    gf_factor_sqf, gf_factor,  # 导入多项式的平方自由因式分解、因式分解函数
    gf_value, linear_congruence, _csolve_prime_las_vegas,  # 导入多项式的值、线性同余方程、特定拉斯维加斯算法函数
    csolve_prime, gf_csolve,  # 导入多项式的素数解、解函数
    gf_frobenius_map, gf_frobenius_monomial_base  # 导入多项式的 Frobenius 映射、单项式基函数
)

# 导入 SymPy 多项式错误模块中的精确商失败异常类
from sympy.polys.polyerrors import (
    ExactQuotientFailed,
)

# 导入 SymPy 中的多项式配置模块
from sympy.polys import polyconfig as config

# 导入 SymPy 中的整数域 ZZ
from sympy.polys.domains import ZZ

# 导入 SymPy 核心模块中的 pi 常数
from sympy.core.numbers import pi

# 导入 SymPy 数论模块中的下一个素数生成函数
from sympy.ntheory.generate import nextprime

# 导入 SymPy 测试模块中的异常检测函数 raises
from sympy.testing.pytest import raises


# 定义测试函数 test_gf_crt，用于测试有限域中的 CRT 相关函数
def test_gf_crt():
    # 设置测试用例中的 U 和 M
    U = [49, 76, 65]
    M = [99, 97, 95]

    # 设置测试用例中的 p 和 u
    p = 912285
    u = 639985

    # 断言：使用 gf_crt 函数计算 U 和 M 的 CRT 结果应该等于 u
    assert gf_crt(U, M, ZZ) == u

    # 设置测试用例中的 E 和 S
    E = [9215, 9405, 9603]
    S = [62, 24, 12]

    # 断言：使用 gf_crt1 函数计算 M 的 CRT 结果应该是 (p, E, S)
    assert gf_crt1(M, ZZ) == (p, E, S)

    # 断言：使用 gf_crt2 函数计算 U、M、p、E、S 的 CRT 结果应该等于 u
    assert gf_crt2(U, M, p, E, S, ZZ) == u


# 定义测试函数 test_gf_int，用于测试有限域中的整数转换函数 gf_int
def test_gf_int():
    # 断言：gf_int(0, 5) 应该等于 0
    assert gf_int(0, 5) == 0
    # 断言：gf_int(1, 5) 应该等于 1
    assert gf_int(1, 5) == 1
    # 断言：gf_int(2, 5) 应该等于 2
    assert gf_int(2, 5) == 2
    # 断言：gf_int(3, 5) 应该等于 -2
    assert gf_int(3, 5) == -2
    # 断言：gf_int(4, 5) 应该等于 -1
    assert gf_int(4, 5) == -1
    # 断言：gf_int(5, 5) 应该等于 0
    assert gf_int(5, 5) == 0


# 定义测试函数 test_gf_degree，用于测试有限域中的多项式的次数函数 gf_degree
def test_gf_degree():
    # 断言：gf_degree([]) 应该等于 -1
    assert gf_degree([]) == -1
    # 断言：gf_degree([1]) 应该等于 0
    assert gf_degree([1]) == 0
    # 断言：gf_degree([1, 0]) 应该等于 1
    assert gf_degree([1, 0]) == 1
    # 断言：gf_degree([1, 0, 0, 0, 1]) 应该等于 4
    assert gf_degree([1, 0, 0, 0, 1]) == 4


# 定义测试函数 test_gf_strip，用于测试有限域中的多项式的去除零系数函数 gf_strip
def test_gf_strip():
    # 断言：gf_strip([]) 应该等于 []
    assert gf_strip([]) == []
    # 断言：gf_strip([0]) 应该等于 []
    assert gf_strip([0]) == []
    # 断言：gf_strip([0, 0, 0]) 应该等于 []
    assert gf_strip([0, 0, 0]) == []

    # 断言：gf_strip([1]) 应该等于 [1]
    assert gf_strip([1]) == [1]
    # 断言：gf_strip([0, 1]) 应该等于 [1]
    assert gf_strip([0, 1]) == [1]
    # 断言：gf_strip([0, 0, 0, 1]) 应该等
    # 使用 gf_to_dict 函数测试对于输入 [10]，GF(11) 下对称和非对称情况的结果是否符合预期
    assert gf_to_dict([10], 11, symmetric=True) == {0: -1}
    # 使用 gf_to_dict 函数测试对于输入 [10]，GF(11) 下非对称情况的结果是否符合预期
    assert gf_to_dict([10], 11, symmetric=False) == {0: 10}
# 测试从整数多项式转换为有限域多项式
def test_gf_from_to_int_poly():
    # 测试将整数多项式 [1, 0, 7, 2, 20] 转换为有限域多项式，基数为 5，预期结果为 [1, 0, 2, 2, 0]
    assert gf_from_int_poly([1, 0, 7, 2, 20], 5) == [1, 0, 2, 2, 0]
    # 测试将有限域多项式 [1, 0, 4, 2, 3] 转换为整数多项式，基数为 5，预期结果为 [1, 0, -1, 2, -2]
    assert gf_to_int_poly([1, 0, 4, 2, 3], 5) == [1, 0, -1, 2, -2]

    # 测试带有 symmetric=True 参数的有限域多项式 [10] 转换为整数多项式，基数为 11，预期结果为 [-1]
    assert gf_to_int_poly([10], 11, symmetric=True) == [-1]
    # 测试带有 symmetric=False 参数的有限域多项式 [10] 转换为整数多项式，基数为 11，预期结果为 [10]
    assert gf_to_int_poly([10], 11, symmetric=False) == [10]


# 测试计算有限域多项式的首项系数（leading coefficient）
def test_gf_LC():
    # 测试空列表作为多项式的首项系数，预期结果为 0
    assert gf_LC([], ZZ) == 0
    # 测试 [1] 的首项系数，预期结果为 1
    assert gf_LC([1], ZZ) == 1
    # 测试 [1, 2] 的首项系数，预期结果为 1
    assert gf_LC([1, 2], ZZ) == 1


# 测试计算有限域多项式的尾项系数（trailing coefficient）
def test_gf_TC():
    # 测试空列表作为多项式的尾项系数，预期结果为 0
    assert gf_TC([], ZZ) == 0
    # 测试 [1] 的尾项系数，预期结果为 1
    assert gf_TC([1], ZZ) == 1
    # 测试 [1, 2] 的尾项系数，预期结果为 2
    assert gf_TC([1, 2], ZZ) == 2


# 测试将有限域多项式转换为首一多项式（monic polynomial）
def test_gf_monic():
    # 测试空列表作为多项式的首一化结果，基数为 11，预期结果为 (0, [])
    assert gf_monic(ZZ.map([]), 11, ZZ) == (0, [])

    # 测试 [1] 的首一化结果，基数为 11，预期结果为 (1, [1])
    assert gf_monic(ZZ.map([1]), 11, ZZ) == (1, [1])
    # 测试 [2] 的首一化结果，基数为 11，预期结果为 (2, [1])
    assert gf_monic(ZZ.map([2]), 11, ZZ) == (2, [1])

    # 测试 [1, 2, 3, 4] 的首一化结果，基数为 11，预期结果为 (1, [1, 2, 3, 4])
    assert gf_monic(ZZ.map([1, 2, 3, 4]), 11, ZZ) == (1, [1, 2, 3, 4])
    # 测试 [2, 3, 4, 5] 的首一化结果，基数为 11，预期结果为 (2, [1, 7, 2, 8])
    assert gf_monic(ZZ.map([2, 3, 4, 5]), 11, ZZ) == (2, [1, 7, 2, 8])


# 测试有限域多项式的算术运算：取负、加常数、减常数、乘常数、加法、减法、加法乘以另一多项式、减法乘以另一多项式、乘法
def test_gf_arith():
    # 测试空列表的取负，基数为 11，预期结果为 []
    assert gf_neg([], 11, ZZ) == []
    # 测试 [1] 的取负，基数为 11，预期结果为 [10]
    assert gf_neg([1], 11, ZZ) == [10]
    # 测试 [1, 2, 3] 的取负，基数为 11，预期结果为 [10, 9, 8]

    # 测试空列表加 0，基数为 11，预期结果为 []
    assert gf_add_ground([], 0, 11, ZZ) == []
    # 测试空列表减 0，基数为 11，预期结果为 []
    assert gf_sub_ground([], 0, 11, ZZ) == []

    # 测试空列表加 3，基数为 11，预期结果为 [3]
    assert gf_add_ground([], 3, 11, ZZ) == [3]
    # 测试空列表减 3，基数为 11，预期结果为 [8]
    assert gf_sub_ground([], 3, 11, ZZ) == [8]

    # 测试 [1] 加 3，基数为 11，预期结果为 [4]
    assert gf_add_ground([1], 3, 11, ZZ) == [4]
    # 测试 [1] 减 3，基数为 11，预期结果为 [9]
    assert gf_sub_ground([1], 3, 11, ZZ) == [9]

    # 测试 [8] 加 3，基数为 11，预期结果为 []
    assert gf_add_ground([8], 3, 11, ZZ) == []
    # 测试 [3] 减 3，基数为 11，预期结果为 []
    assert gf_sub_ground([3], 3, 11, ZZ) == []

    # 测试 [1, 2, 3] 加 3，基数为 11，预期结果为 [1, 2, 6]
    assert gf_add_ground([1, 2, 3], 3, 11, ZZ) == [1, 2, 6]
    # 测试 [1, 2, 3] 减 3，基数为 11，预期结果为 [1, 2, 0]

    # 测试空列表乘 0，基数为 11，预期结果为 []
    assert gf_mul_ground([], 0, 11, ZZ) == []
    # 测试空列表乘 1，基数为 11，预期结果为 []

    # 测试 [1] 乘 0，基数为 11，预期结果为 []
    assert gf_mul_ground([1], 0, 11, ZZ) == []
    # 测试 [1] 乘 1，基数为 11，预期结果为 [1]

    # 测试 [1, 2, 3] 乘 0，基数为 11，预期结果为 []
    assert gf_mul_ground([1, 2, 3], 0, 11, ZZ) == []
    # 测试 [1, 2, 3] 乘 1，基数为 11，预期结果为 [1, 2, 3]
    # 测试 [1, 2, 3] 乘 7，基数为 11，预期结果为 [7, 3, 10]

    # 测试空列表加空列表，基数为 11，预期结果为 []
    assert gf_add([], [], 11, ZZ) == []
    # 测试 [1] 加空列表，基数为 11，预期结果为 [1]
    assert gf_add([1], [], 11, ZZ) == [1]
    # 测试空列表加 [1]，基数为 11，预期结果为 [1]
    assert gf_add([], [1], 11, ZZ) == [1]
    # 测试 [1] 加 [1]，基数为 11，预期结果为 [2]
    assert gf_add([1], [1], 11, ZZ) == [2]
    #
    # 断言：验证在有限域上的多项式乘法结果是否符合预期
    assert gf_mul([5], [7], 11, ZZ) == [2]
    
    # 断言：验证在有限域上的多项式乘法结果是否符合预期
    assert gf_mul([3, 0, 0, 6, 1, 2], [4, 0, 1, 0], 11, ZZ) == [1, 0,
                  3, 2, 4, 3, 1, 2, 0]
    # 断言：验证在有限域上的多项式乘法结果是否符合预期（乘法顺序相反）
    assert gf_mul([4, 0, 1, 0], [3, 0, 0, 6, 1, 2], 11, ZZ) == [1, 0,
                  3, 2, 4, 3, 1, 2, 0]
    
    # 断言：验证在有限域上的多项式乘法结果是否符合预期（同一个多项式自身的乘法）
    assert gf_mul([2, 0, 0, 1, 7], [2, 0, 0, 1, 7], 11, ZZ) == [4, 0,
                  0, 4, 6, 0, 1, 3, 5]
    
    # 断言：验证在有限域上的多项式平方操作结果是否符合预期（空列表）
    assert gf_sqr([], 11, ZZ) == []
    # 断言：验证在有限域上的多项式平方操作结果是否符合预期
    assert gf_sqr([2], 11, ZZ) == [4]
    # 断言：验证在有限域上的多项式平方操作结果是否符合预期
    assert gf_sqr([1, 2], 11, ZZ) == [1, 4, 4]
    
    # 断言：验证在有限域上的多项式平方操作结果是否符合预期
    assert gf_sqr([2, 0, 0, 1, 7], 11, ZZ) == [4, 0, 0, 4, 6, 0, 1, 3, 5]
def test_gf_division():
    # 测试除法函数，期望引发 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: gf_div([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_rem([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_quo([1, 2, 3], [], 11, ZZ))
    raises(ZeroDivisionError, lambda: gf_quo([1, 2, 3], [], 11, ZZ))

    # 测试在有限域上的除法、取余和商函数
    assert gf_div([1], [1, 2, 3], 7, ZZ) == ([], [1])
    assert gf_rem([1], [1, 2, 3], 7, ZZ) == [1]
    assert gf_quo([1], [1, 2, 3], 7, ZZ) == []

    # 定义多项式 f 和 g，并期望的商和余数
    f = ZZ.map([5, 4, 3, 2, 1, 0])
    g = ZZ.map([1, 2, 3])
    q = [5, 1, 0, 6]
    r = [3, 3]

    # 测试多项式除法的结果是否符合预期
    assert gf_div(f, g, 7, ZZ) == (q, r)
    assert gf_rem(f, g, 7, ZZ) == r
    assert gf_quo(f, g, 7, ZZ) == q

    # 在除法失败时引发 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: gf_exquo(f, g, 7, ZZ))

    # 更新多项式 g，再次测试除法的结果
    f = ZZ.map([5, 4, 3, 2, 1, 0])
    g = ZZ.map([1, 2, 3, 0])
    q = [5, 1, 0]
    r = [6, 1, 0]

    # 测试更新后的多项式除法的结果
    assert gf_div(f, g, 7, ZZ) == (q, r)
    assert gf_rem(f, g, 7, ZZ) == r
    assert gf_quo(f, g, 7, ZZ) == q

    # 在除法失败时再次引发 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: gf_exquo(f, g, 7, ZZ))

    # 测试另一组多项式在有限域上的除法
    assert gf_quo(ZZ.map([1, 2, 1]), ZZ.map([1, 1]), 11, ZZ) == [1, 1]


def test_gf_shift():
    # 定义一个多项式 f
    f = [1, 2, 3, 4, 5]

    # 测试左移函数，期望结果为空列表
    assert gf_lshift([], 5, ZZ) == []
    # 测试右移函数，期望结果为元组 ([], [])
    assert gf_rshift([], 5, ZZ) == ([], [])

    # 测试左移函数，移动一个位置
    assert gf_lshift(f, 1, ZZ) == [1, 2, 3, 4, 5, 0]
    # 测试左移函数，移动两个位置
    assert gf_lshift(f, 2, ZZ) == [1, 2, 3, 4, 5, 0, 0]

    # 测试右移函数，移动零个位置，期望结果为原多项式 f 和空列表
    assert gf_rshift(f, 0, ZZ) == (f, [])
    # 测试右移函数，移动一个位置，期望结果为 ([1, 2, 3, 4], [5])
    assert gf_rshift(f, 1, ZZ) == ([1, 2, 3, 4], [5])
    # 测试右移函数，移动三个位置，期望结果为 ([1, 2], [3, 4, 5])
    assert gf_rshift(f, 3, ZZ) == ([1, 2], [3, 4, 5])
    # 测试右移函数，移动五个位置，期望结果为 ([], f)
    assert gf_rshift(f, 5, ZZ) == ([], f)


def test_gf_expand():
    # 定义一个多项式列表 F
    F = [([1, 1], 2), ([1, 2], 3)]

    # 测试多项式扩展函数，期望结果为 [1, 8, 3, 5, 6, 8]
    assert gf_expand(F, 11, ZZ) == [1, 8, 3, 5, 6, 8]
    # 测试多项式扩展函数，带有首项系数为 4，期望结果为 [4, 10, 1, 9, 2, 10]
    assert gf_expand((4, F), 11, ZZ) == [4, 10, 1, 9, 2, 10]


def test_gf_powering():
    # 测试多项式幂函数，幂为 0 时，期望结果为 [1]
    assert gf_pow([1, 0, 0, 1, 8], 0, 11, ZZ) == [1]
    # 测试多项式幂函数，幂为 1 时，期望结果为 [1, 0, 0, 1, 8]
    assert gf_pow([1, 0, 0, 1, 8], 1, 11, ZZ) == [1, 0, 0, 1, 8]
    # 测试多项式幂函数，幂为 2 时，期望结果为 [1, 0, 0, 2, 5, 0, 1, 5, 9]
    assert gf_pow([1, 0, 0, 1, 8], 2, 11, ZZ) == [1, 0, 0, 2, 5, 0, 1, 5, 9]

    # 测试多项式幂函数，幂为 5 时的较长结果
    assert gf_pow([1, 0, 0, 1, 8], 5, 11, ZZ) == [
        1, 0, 0, 5, 7, 0, 10, 6, 2, 10, 9, 6, 10, 6, 6, 0, 5, 2, 5, 9, 10]

    # 测试多项式幂函数，幂为 8 时的更长结果
    assert gf_pow([1, 0, 0, 1, 8], 8, 11, ZZ) == [
        1, 0, 0, 8, 9, 0, 6, 8, 10, 1, 2, 5, 10, 7, 7, 9, 1, 2, 0, 0, 6, 2,
        5, 2, 5, 7, 7, 9, 10, 10, 7, 5, 5]

    # 测试多项式幂函数，幂为 45 时的极长结果
    assert gf_pow([1, 0, 0, 1, 8], 45, 11, ZZ) == [
        1, 0, 0,  1,  8, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,  4, 0, 0,  4, 10, 0, 0, 0, 0, 0, 0,
        10, 0, 0, 10,  3, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,
        6
    # 断言：验证 gf_pow_mod 函数的返回结果是否符合预期
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 0, ZZ.map([2, 0, 7]), 11, ZZ) == [1]
    # 断言：验证 gf_pow_mod 函数的返回结果是否符合预期
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 1, ZZ.map([2, 0, 7]), 11, ZZ) == [1, 1]
    # 断言：验证 gf_pow_mod 函数的返回结果是否符合预期
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 2, ZZ.map([2, 0, 7]), 11, ZZ) == [2, 3]
    # 断言：验证 gf_pow_mod 函数的返回结果是否符合预期
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 5, ZZ.map([2, 0, 7]), 11, ZZ) == [7, 8]
    # 断言：验证 gf_pow_mod 函数的返回结果是否符合预期
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 8, ZZ.map([2, 0, 7]), 11, ZZ) == [1, 5]
    # 断言：验证 gf_pow_mod 函数的返回结果是否符合预期
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 45, ZZ.map([2, 0, 7]), 11, ZZ) == [5, 4]
# 定义测试函数 test_gf_gcdex，用于测试 gf_gcdex 函数的不同输入组合
def test_gf_gcdex():
    # 断言：调用 gf_gcdex 函数，传入空的 ZZ 映射，期望返回 ([1], [], [])
    assert gf_gcdex(ZZ.map([]), ZZ.map([]), 11, ZZ) == ([1], [], [])
    # 断言：调用 gf_gcdex 函数，传入 [2] 和空的 ZZ 映射，期望返回 ([6], [], [1])
    assert gf_gcdex(ZZ.map([2]), ZZ.map([]), 11, ZZ) == ([6], [], [1])
    # 断言：调用 gf_gcdex 函数，传入空的 ZZ 映射和 [2]，期望返回 ([], [6], [1])
    assert gf_gcdex(ZZ.map([]), ZZ.map([2]), 11, ZZ) == ([], [6], [1])
    # 断言：调用 gf_gcdex 函数，传入 [2] 和 [2]，期望返回 ([], [6], [1])
    assert gf_gcdex(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == ([], [6], [1])

    # 断言：调用 gf_gcdex 函数，传入 [3, 0] 和空的 ZZ 映射，期望返回 ([], [4], [1, 0])
    assert gf_gcdex(ZZ.map([]), ZZ.map([3, 0]), 11, ZZ) == ([], [4], [1, 0])
    # 断言：调用 gf_gcdex 函数，传入 [3, 0] 和空的 ZZ 映射，期望返回 ([4], [], [1, 0])
    assert gf_gcdex(ZZ.map([3, 0]), ZZ.map([]), 11, ZZ) == ([4], [], [1, 0])

    # 断言：调用 gf_gcdex 函数，传入 [3, 0] 和 [3, 0]，期望返回 ([], [4], [1, 0])
    assert gf_gcdex(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == ([], [4], [1, 0])

    # 断言：调用 gf_gcdex 函数，传入 [1, 8, 7] 和 [1, 7, 1, 7]，期望返回 ([5, 6], [6], [1, 7])
    assert gf_gcdex(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == ([5, 6], [6], [1, 7])


# 定义测试函数 test_gf_gcd，用于测试 gf_gcd 函数的不同输入组合
def test_gf_gcd():
    # 断言：调用 gf_gcd 函数，传入空的 ZZ 映射，期望返回空列表 []
    assert gf_gcd(ZZ.map([]), ZZ.map([]), 11, ZZ) == []
    # 断言：调用 gf_gcd 函数，传入 [2] 和空的 ZZ 映射，期望返回 [1]
    assert gf_gcd(ZZ.map([2]), ZZ.map([]), 11, ZZ) == [1]
    # 断言：调用 gf_gcd 函数，传入空的 ZZ 映射和 [2]，期望返回 [1]
    assert gf_gcd(ZZ.map([]), ZZ.map([2]), 11, ZZ) == [1]
    # 断言：调用 gf_gcd 函数，传入 [2] 和 [2]，期望返回 [1]
    assert gf_gcd(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == [1]

    # 断言：调用 gf_gcd 函数，传入空的 ZZ 映射和 [1, 0]，期望返回 [1, 0]
    assert gf_gcd(ZZ.map([]), ZZ.map([1, 0]), 11, ZZ) == [1, 0]
    # 断言：调用 gf_gcd 函数，传入 [1, 0] 和空的 ZZ 映射，期望返回 [1, 0]
    assert gf_gcd(ZZ.map([1, 0]), ZZ.map([]), 11, ZZ) == [1, 0]

    # 断言：调用 gf_gcd 函数，传入 [3, 0] 和 [3, 0]，期望返回 [1, 0]
    assert gf_gcd(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == [1, 0]
    # 断言：调用 gf_gcd 函数，传入 [1, 8, 7] 和 [1, 7, 1, 7]，期望返回 [1, 7]
    assert gf_gcd(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == [1, 7]


# 定义测试函数 test_gf_lcm，用于测试 gf_lcm 函数的不同输入组合
def test_gf_lcm():
    # 断言：调用 gf_lcm 函数，传入空的 ZZ 映射，期望返回空列表 []
    assert gf_lcm(ZZ.map([]), ZZ.map([]), 11, ZZ) == []
    # 断言：调用 gf_lcm 函数，传入 [2] 和空的 ZZ 映射，期望返回空列表 []
    assert gf_lcm(ZZ.map([2]), ZZ.map([]), 11, ZZ) == []
    # 断言：调用 gf_lcm 函数，传入空的 ZZ 映射和 [2]，期望返回空列表 []
    assert gf_lcm(ZZ.map([]), ZZ.map([2]), 11, ZZ) == []
    # 断言：调用 gf_lcm 函数，传入 [2] 和 [2]，期望返回 [1]
    assert gf_lcm(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == [1]

    # 断言：调用 gf_lcm 函数，传入空的 ZZ 映射和 [1, 0]，期望返回空列表 []
    assert gf_lcm(ZZ.map([]), ZZ.map([1, 0]), 11, ZZ) == []
    # 断言：调用 gf_lcm 函数，传入 [1, 0] 和空的 ZZ 映射，期望返回空列表 []
    assert gf_lcm(ZZ.map([1, 0]), ZZ.map([]), 11, ZZ) == []

    # 断言：调用 gf_lcm 函数，传入 [3, 0] 和 [3, 0]，期望返回 [1, 0]
    assert gf_lcm(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == [1, 0]
    # 断言：调用 gf_lcm 函数，传入 [1, 8, 7] 和 [1, 7, 1, 7]，期望返回 [1, 8, 8, 8, 7]
    assert gf_lcm(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == [1, 8, 8, 8, 7]


# 定义测试函数 test_gf_cofactors，用于测试 gf_cofactors 函数的不同输入组合
def test_gf_cofactors():
    # 断言：调用 gf_cofactors 函数，传入空的 ZZ 映射，期望返回空的三元组 ([], [], [])
    assert gf_cofactors(ZZ.map([]), ZZ.map([]), 11, ZZ) == ([], [], [])
    # 断言：调用 gf_cofactors 函数，传入 [2] 和空的 ZZ 映射，期望返回 ([1], [2], [])
    assert gf_cofactors(ZZ.map([2]), ZZ.map([]), 11, ZZ) == ([1], [2], [])
    # 断言：调用 gf_cofactors 函数，传入空的 ZZ 映射和 [2]，期望返回 ([1], [], [2])
    assert gf_cofactors(ZZ.map([]), ZZ.map([2]), 11, ZZ) == ([1], [], [2])
    # 断言
    # 使用 gf_eval 函数验证给定参数的 Galois Field 上的多项式求值是否正确
    assert gf_eval([1, 0, 3, 2, 4, 3, 1, 2, 0], 27, 11, ZZ) == 5
    
    # 使用 gf_eval 函数验证给定参数的 Galois Field 上的多项式求值是否正确
    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 0, 11, ZZ) == 5
    # 使用 gf_eval 函数验证给定参数的 Galois Field 上的多项式求值是否正确
    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 4, 11, ZZ) == 3
    # 使用 gf_eval 函数验证给定参数的 Galois Field 上的多项式求值是否正确
    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 27, 11, ZZ) == 9
    
    # 使用 gf_multi_eval 函数验证给定参数的 Galois Field 上的多项式组求值是否正确
    assert gf_multi_eval([3, 2, 1], [0, 1, 2, 3], 11, ZZ) == [1, 6, 6, 1]
# 定义测试函数 test_gf_compose，用于测试 gf_compose 和 gf_compose_mod 函数
def test_gf_compose():
    # 空列表组合结果应为空列表
    assert gf_compose([], [1, 0], 11, ZZ) == []
    # 空列表模合结果应为空列表
    assert gf_compose_mod([], [1, 0], [1, 0], 11, ZZ) == []

    # 单项式 1 组合结果应为 [1]
    assert gf_compose([1], [], 11, ZZ) == [1]
    # 单项式 [1, 0] 组合结果应为空列表
    assert gf_compose([1, 0], [], 11, ZZ) == []
    # 单项式 [1, 0] 组合单项式 [1, 0] 结果应为 [1, 0]
    assert gf_compose([1, 0], [1, 0], 11, ZZ) == [1, 0]

    # 定义多项式 f, g, h
    f = ZZ.map([1, 1, 4, 9, 1])
    g = ZZ.map([1, 1, 1])
    h = ZZ.map([1, 0, 0, 2])

    # 测试 gf_compose 函数组合 g 和 h 的结果应为 [1, 0, 0, 5, 0, 0, 7]
    assert gf_compose(g, h, 11, ZZ) == [1, 0, 0, 5, 0, 0, 7]
    # 测试 gf_compose_mod 函数组合 g 和 h 模 f 的结果应为 [3, 9, 6, 10]
    assert gf_compose_mod(g, h, f, 11, ZZ) == [3, 9, 6, 10]


# 定义测试函数 test_gf_trace_map，用于测试 gf_trace_map 函数
def test_gf_trace_map():
    # 定义多项式 f, a, c，并计算 b
    f = ZZ.map([1, 1, 4, 9, 1])
    a = [1, 1, 1]
    c = ZZ.map([1, 0])
    b = gf_pow_mod(c, 11, f, 11, ZZ)

    # 测试 gf_trace_map 函数 trace 为 0 时的返回结果
    assert gf_trace_map(a, b, c, 0, f, 11, ZZ) == ([1, 1, 1], [1, 1, 1])
    # 测试 gf_trace_map 函数 trace 为 1 时的返回结果
    assert gf_trace_map(a, b, c, 1, f, 11, ZZ) == ([5, 2, 10, 3], [5, 3, 0, 4])
    # 测试 gf_trace_map 函数 trace 为 2 时的返回结果
    assert gf_trace_map(a, b, c, 2, f, 11, ZZ) == ([5, 9, 5, 3], [10, 1, 5, 7])
    # 测试 gf_trace_map 函数 trace 为 3 时的返回结果
    assert gf_trace_map(a, b, c, 3, f, 11, ZZ) == ([1, 10, 6, 0], [7])
    # 测试 gf_trace_map 函数 trace 为 4 时的返回结果
    assert gf_trace_map(a, b, c, 4, f, 11, ZZ) == ([1, 1, 1], [1, 1, 8])
    # 测试 gf_trace_map 函数 trace 为 5 时的返回结果
    assert gf_trace_map(a, b, c, 5, f, 11, ZZ) == ([5, 2, 10, 3], [5, 3, 0, 0])
    # 测试 gf_trace_map 函数 trace 为 11 时的返回结果
    assert gf_trace_map(a, b, c, 11, f, 11, ZZ) == ([1, 10, 6, 0], [10])


# 定义测试函数 test_gf_irreducible，用于测试 gf_irreducible 函数
def test_gf_irreducible():
    # 测试 gf_irreducible_p 函数对多个次数的多项式是否判断为不可约
    assert gf_irreducible_p(gf_irreducible(1, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(2, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(3, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(4, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(5, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(6, 11, ZZ), 11, ZZ) is True
    assert gf_irreducible_p(gf_irreducible(7, 11, ZZ), 11, ZZ) is True


# 定义测试函数 test_gf_irreducible_p，用于测试 gf_irreducible_p 函数及其相关函数
def test_gf_irreducible_p():
    # 测试不同方法对多项式是否不可约的判断结果
    assert gf_irred_p_ben_or(ZZ.map([7]), 11, ZZ) is True
    assert gf_irred_p_ben_or(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irred_p_ben_or(ZZ.map([7, 3, 1]), 11, ZZ) is False

    assert gf_irred_p_rabin(ZZ.map([7]), 11, ZZ) is True
    assert gf_irred_p_rabin(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irred_p_rabin(ZZ.map([7, 3, 1]), 11, ZZ) is False

    # 设置不同的不可约性判断方法并测试结果
    config.setup('GF_IRRED_METHOD', 'ben-or')
    assert gf_irreducible_p(ZZ.map([7]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3, 1]), 11, ZZ) is False

    config.setup('GF_IRRED_METHOD', 'rabin')
    assert gf_irreducible_p(ZZ.map([7]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3, 1]), 11, ZZ) is False

    # 测试设置不存在的方法时是否抛出 KeyError 异常
    config.setup('GF_IRRED_METHOD', 'other')
    raises(KeyError, lambda: gf_irreducible_p([7], 11, ZZ))
    config.setup('GF_IRRED_METHOD')

    # 定义多项式 f, g，并计算 h
    f = ZZ.map([1, 9, 9, 13, 16, 15, 6, 7, 7, 7, 10])
    g = ZZ.map([1, 7, 16, 7, 15, 13, 13, 11, 16, 10, 9])
    h = gf_mul(f, g, 17, ZZ)

    # 测试 ben-or 方法对 f, g 是否判断为不可约
    assert gf_irred_p_ben_or(f, 17, ZZ) is True
    assert gf_irred_p_ben_or(g, 17, ZZ) is True
    # 断言验证给定多项式 h 在域 ZZ 上，是否为不可约多项式（Ben-Or 算法）
    assert gf_irred_p_ben_or(h, 17, ZZ) is False
    
    # 断言验证给定多项式 f 在域 ZZ 上，是否为不可约多项式（Rabin 算法）
    assert gf_irred_p_rabin(f, 17, ZZ) is True
    # 断言验证给定多项式 g 在域 ZZ 上，是否为不可约多项式（Rabin 算法）
    assert gf_irred_p_rabin(g, 17, ZZ) is True
    
    # 断言验证给定多项式 h 在域 ZZ 上，是否为不可约多项式（Rabin 算法）
    assert gf_irred_p_rabin(h, 17, ZZ) is False
def test_gf_squarefree():
    # 测试空多项式的平方自由列表
    assert gf_sqf_list([], 11, ZZ) == (0, [])
    # 测试常数多项式的平方自由列表
    assert gf_sqf_list([1], 11, ZZ) == (1, [])
    # 测试非平方自由多项式的平方自由列表
    assert gf_sqf_list([1, 1], 11, ZZ) == (1, [([1, 1], 1)])

    # 测试空多项式的平方自由性判断
    assert gf_sqf_p([], 11, ZZ) is True
    # 测试常数多项式的平方自由性判断
    assert gf_sqf_p([1], 11, ZZ) is True
    # 测试非平方自由多项式的平方自由性判断
    assert gf_sqf_p([1, 1], 11, ZZ) is True

    # 创建一个GF多项式 f = x^11 + 1
    f = gf_from_dict({11: 1, 0: 1}, 11, ZZ)
    # 测试非平方自由多项式的平方自由性判断
    assert gf_sqf_p(f, 11, ZZ) is False

    # 测试非平方自由多项式的平方自由列表
    assert gf_sqf_list(f, 11, ZZ) == \
        (1, [([1, 1], 11)])

    # 测试普通多项式 f = [1, 5, 8, 4] 的平方自由性判断
    f = [1, 5, 8, 4]
    assert gf_sqf_p(f, 11, ZZ) is False

    # 测试普通多项式 f = [1, 5, 8, 4] 的平方自由列表
    assert gf_sqf_list(f, 11, ZZ) == \
        (1, [([1, 1], 1),
             ([1, 2], 2)])

    # 测试普通多项式 f = [1, 5, 8, 4] 的平方自由部分
    assert gf_sqf_part(f, 11, ZZ) == [1, 3, 2]

    # 测试特定多项式 f = [1, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0] 在模数 3 下的平方自由列表
    f = [1, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0]
    assert gf_sqf_list(f, 3, ZZ) == \
        (1, [([1, 0], 1),
             ([1, 1], 3),
             ([1, 2], 6)])

def test_gf_frobenius_map():
    # 创建两个GF多项式 f 和 g
    f = ZZ.map([2, 0, 1, 0, 2, 2, 0, 2, 2, 2])
    g = ZZ.map([1,1,0,2,0,1,0,2,0,1])
    p = 3
    # 计算 g 的 Frobenius 单项式基
    b = gf_frobenius_monomial_base(g, p, ZZ)
    # 计算 f 和 g 的 Frobenius 映射
    h = gf_frobenius_map(f, g, b, p, ZZ)
    # 计算 f 的 p 次幂模 g
    h1 = gf_pow_mod(f, p, g, p, ZZ)
    # 验证两种方法得到的结果相等
    assert h == h1

def test_gf_berlekamp():
    # 创建 GF 多项式 f = x^6 - 3x^5 + x^4 - 3x^3 - x^2 - 3x + 1
    f = gf_from_int_poly([1, -3, 1, -3, -1, -3, 1], 11)

    # 预期的 Q 矩阵
    Q = [[1, 0, 0, 0, 0, 0],
         [3, 5, 8, 8, 6, 5],
         [3, 6, 6, 1, 10, 0],
         [9, 4, 10, 3, 7, 9],
         [7, 8, 10, 0, 0, 8],
         [8, 10, 7, 8, 10, 8]]

    # 预期的 Q 基
    V = [[1, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 0],
         [0, 0, 7, 9, 0, 1]]

    # 验证计算的 Q 矩阵与预期值是否相等
    assert gf_Qmatrix(f, 11, ZZ) == Q
    # 验证计算的 Q 基与预期值是否相等
    assert gf_Qbasis(Q, 11, ZZ) == V

    # 验证 Berlekamp 算法得到的因子分解是否正确
    assert gf_berlekamp(f, 11, ZZ) == \
        [[1, 1], [1, 5, 3], [1, 2, 3, 4]]

    # 创建另一个 GF 多项式 f = [1, 0, 1, 0, 10, 10, 8, 2, 8]
    f = ZZ.map([1, 0, 1, 0, 10, 10, 8, 2, 8])

    # 预期的 Q 矩阵
    Q = ZZ.map([[1, 0, 0, 0, 0, 0, 0, 0],
         [2, 1, 7, 11, 10, 12, 5, 11],
         [3, 6, 4, 3, 0, 4, 7, 2],
         [4, 3, 6, 5, 1, 6, 2, 3],
         [2, 11, 8, 8, 3, 1, 3, 11],
         [6, 11, 8, 6, 2, 7, 10, 9],
         [5, 11, 7, 10, 0, 11, 7, 12],
         [3, 3, 12, 5, 0, 11, 9, 12]])

    # 预期的 Q 基
    V = [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 5, 5, 0, 9, 5, 1, 0],
         [0, 9, 11, 9, 10, 12, 0, 1]]

    # 验证计算的 Q 矩阵与预期值是否相等
    assert gf_Qmatrix(f, 13, ZZ) == Q
    # 验证计算的 Q 基与预期值是否相等
    assert gf_Qbasis(Q, 13, ZZ) == V

    # 验证 Berlekamp 算法得到的因子分解是否正确
    assert gf_berlekamp(f, 13, ZZ) == \
        [[1, 3], [1, 8, 4, 12]]

def test_gf_ddf():
    # 创建 GF 多项式 f = x^15 - 1
    f = gf_from_dict({15: ZZ(1), 0: ZZ(-1)}, 11, ZZ)
    # 预期的 Zassenhaus 和 Shoup 算法的因子分解结果
    g = [([1, 0, 0, 0, 0, 10], 1),
         ([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 2)]

    # 验证 Zassenhaus 算法得到的因子分解是否正确
    assert gf_ddf_zassenhaus(f, 11, ZZ) == g
    # 验证 Shoup 算法得到的因子分解是否正确
    assert gf_ddf_shoup(f, 11, ZZ) == g

    # 创建
    # 定义多项式 g，每个元素是一个元组，包含多项式系数列表和其分解的因子个数
    g = [([1, 1, 0], 1),
         ([1, 1, 0, 1, 2], 2)]

    # 使用 Zassenhaus 方法计算多项式 f 在有限域 ZZ/3ZZ 上的因式分解，并断言结果与 g 相等
    assert gf_ddf_zassenhaus(f, 3, ZZ) == g
    # 使用 Shoup 方法计算多项式 f 在有限域 ZZ/3ZZ 上的因式分解，并断言结果与 g 相等
    assert gf_ddf_shoup(f, 3, ZZ) == g

    # 将列表 [1, 2, 5, 26, 677, 436, 791, 325, 456, 24, 577] 映射为有限域 ZZ 上的多项式 f
    f = ZZ.map([1, 2, 5, 26, 677, 436, 791, 325, 456, 24, 577])
    # 定义多项式 g，每个元素是一个元组，包含多项式系数列表和其分解的因子个数
    g = [([1, 701], 1),
         ([1, 110, 559, 532, 694, 151, 110, 70, 735, 122], 9)]

    # 使用 Zassenhaus 方法计算多项式 f 在有限域 ZZ/809ZZ 上的因式分解，并断言结果与 g 相等
    assert gf_ddf_zassenhaus(f, 809, ZZ) == g
    # 使用 Shoup 方法计算多项式 f 在有限域 ZZ/809ZZ 上的因式分解，并断言结果与 g 相等
    assert gf_ddf_shoup(f, 809, ZZ) == g

    # 计算 p = nextprime(int((2**15 * pi).evalf()))，并将其转换为有限域 ZZ 上的整数
    p = ZZ(nextprime(int((2**15 * pi).evalf())))
    # 使用字典形式创建多项式 f = x^15 + x + 1 在有限域 ZZ/pZZ 上的表示
    f = gf_from_dict({15: 1, 1: 1, 0: 1}, p, ZZ)
    # 定义多项式 g，每个元素是一个元组，包含多项式系数列表和其分解的因子个数
    g = [([1, 22730, 68144], 2),
         ([1, 64876, 83977, 10787, 12561, 68608, 52650, 88001, 84356], 4),
         ([1, 15347, 95022, 84569, 94508, 92335], 5)]

    # 使用 Zassenhaus 方法计算多项式 f 在有限域 ZZ/pZZ 上的因式分解，并断言结果与 g 相等
    assert gf_ddf_zassenhaus(f, p, ZZ) == g
    # 使用 Shoup 方法计算多项式 f 在有限域 ZZ/pZZ 上的因式分解，并断言结果与 g 相等
    assert gf_ddf_shoup(f, p, ZZ) == g
# 定义测试函数 test_gf_edf，用于测试 gf_edf_zassenhaus 和 gf_edf_shoup 函数
def test_gf_edf():
    # 创建多项式 f = [1, 1, 0, 1, 2]，使用 ZZ.map 将其转换为整数多项式
    f = ZZ.map([1, 1, 0, 1, 2])
    # 创建矩阵 g = [[1, 0, 1], [1, 1, 2]]，使用 ZZ.map 将其转换为整数矩阵
    g = ZZ.map([[1, 0, 1], [1, 1, 2]])

    # 断言使用 Zassenhaus 方法对 f 进行有限域因式分解得到的结果等于 g
    assert gf_edf_zassenhaus(f, 2, 3, ZZ) == g
    # 断言使用 Shoup 方法对 f 进行有限域因式分解得到的结果等于 g
    assert gf_edf_shoup(f, 2, 3, ZZ) == g


# 定义测试函数 test_issue_23174，用于测试特定问题的解决方案
def test_issue_23174():
    # 创建多项式 f = [1, 1, ..., 1]，长度为 17，使用 ZZ.map 将其转换为整数多项式
    f = ZZ.map([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # 创建矩阵 g = [[1, 0, 0, 1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0, 1, 1, 1]]，使用 ZZ.map 将其转换为整数矩阵
    g = ZZ.map([[1, 0, 0, 1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0, 1, 1, 1]])

    # 断言使用 Zassenhaus 方法对 f 进行有限域因式分解得到的结果等于 g
    assert gf_edf_zassenhaus(f, 8, 2, ZZ) == g


# 定义测试函数 test_gf_factor，用于测试 gf_factor 和 gf_factor_sqf 函数
def test_gf_factor():
    # 断言对空多项式使用 GF(11) 下的因式分解返回结果为 (0, [])
    assert gf_factor([], 11, ZZ) == (0, [])
    # 断言对 [1] 多项式使用 GF(11) 下的因式分解返回结果为 (1, [])
    assert gf_factor([1], 11, ZZ) == (1, [])
    # 断言对 [1, 1] 多项式使用 GF(11) 下的因式分解返回结果为 (1, [([1, 1], 1)])
    assert gf_factor([1, 1], 11, ZZ) == (1, [([1, 1], 1)])

    # 断言对空多项式使用 GF(11) 下的平方自由因式分解返回结果为 (0, [])
    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    # 断言对 [1] 多项式使用 GF(11) 下的平方自由因式分解返回结果为 (1, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    # 断言对 [1, 1] 多项式使用 GF(11) 下的平方自由因式分解返回结果为 (1, [[1, 1]])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])

    # 设置 GF_FACTOR_METHOD 环境为 'berlekamp'
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    # 断言对空多项式使用 GF(11) 下的平方自由因式分解返回结果为 (0, [])
    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    # 断言对 [1] 多项式使用 GF(11) 下的平方自由因式分解返回结果为 (1, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    # 断言对 [1, 1] 多项式使用 GF(11) 下的平方自由因式分解返回结果为 (1, [[1, 1]])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])

    # 设置 GF_FACTOR_METHOD 环境为 'zassenhaus'
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    # 断言对空多项式使用 GF(11) 下的平方自由因式分解返回结果为 (0, [])
    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    # 断言对 [1] 多项式使用 GF(11) 下的平方自由因式分解返回结果为 (1, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    # 断言对 [1, 1] 多项式使用 GF(11) 下的平方自由因式分解返回结果为 (1, [[1, 1]])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])

    # 设置 GF_FACTOR_METHOD 环境为 'shoup'
    config.setup('GF_FACTOR_METHOD', 'shoup')
    # 断言对空多项式使用 GF(11) 下的平方自由因式分解返回结果为 (0, [])
    assert gf_factor_sqf(ZZ.map([]), 11, ZZ) == (0, [])
    # 断言对 [1] 多项式使用 GF(11) 下的平方自由因式分解返回结果为 (1, [])
    assert gf_factor_sqf(ZZ.map([1]), 11, ZZ) == (1, [])
    # 断言对 [1, 1] 多项式使用 GF(11) 下的平方自由因式分解返回结果为 (1, [[1, 1]])
    assert gf_factor_sqf(ZZ.map([1, 1]), 11, ZZ) == (1, [[1, 1]])

    # 创建多项式 f = [1, 0, 0, 1, 0]，有限域为 GF(2)，指数 p = 2
    f, p = ZZ.map([1, 0, 0, 1, 0]), 2
    # 创建期望的因式分解结果 g
    g = (1, [([1, 0], 1),
             ([1, 1], 1),
             ([1, 1, 1], 1)])

    # 设置 GF_FACTOR_METHOD 环境为 'berlekamp'
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    # 断言对 f 使用 GF(2) 下的因式分解返回结果等于 g
    assert gf_factor(f, p, ZZ) == g

    # 设置 GF_FACTOR_METHOD 环境为 'zassenhaus'
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    # 断言对 f 使用 GF(2) 下的因式分解返回结果等于 g
    assert gf_factor(f, p, ZZ) == g

    # 设置 GF_FACTOR_METHOD 环境为 'shoup'
    config.setup('GF_FACTOR_METHOD', 'shoup')
    # 断言对 f 使用 GF(2) 下的因式分解返回结果等于 g
    assert gf_factor(f, p, ZZ) == g

    # 更新 f 和 p 为新的值
    f, p = gf_from_int_poly([1, -3, 1, -3, -1, -3, 1], 11), 11
    # 创建期望的因式分解结果 g
    g = (1, [([1, 1], 1),
             ([1, 5, 3], 1),
             ([1, 2, 3, 4], 1)])

    # 设置 GF_FACTOR_METHOD 环境为 'berlekamp'
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    # 断言对 f 使用 GF(11) 下的因式分解返回结果
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'zassenhaus'
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'shoup'
    config.setup('GF_FACTOR_METHOD', 'shoup')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 使用 gf_from_dict 函数创建有限域元素 f 和素数 p
    f, p = gf_from_dict({32: 1, 0: 1}, 11, ZZ), 11

    # 预期的有限域因式分解结果 g
    g = (1, [([1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 10], 1),
             ([1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 10], 1)])

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'berlekamp'
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'zassenhaus'
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'shoup'
    config.setup('GF_FACTOR_METHOD', 'shoup')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 使用 gf_from_dict 函数创建有限域元素 f 和素数 p
    f, p = gf_from_dict({32: ZZ(8), 0: ZZ(5)}, 11, ZZ), 11

    # 预期的有限域因式分解结果 g
    g = (8, [([1, 3], 1),
             ([1, 8], 1),
             ([1, 0, 9], 1),
             ([1, 2, 2], 1),
             ([1, 9, 2], 1),
             ([1, 0, 5, 0, 7], 1),
             ([1, 0, 6, 0, 7], 1),
             ([1, 0, 0, 0, 1, 0, 0, 0, 6], 1),
             ([1, 0, 0, 0, 10, 0, 0, 0, 6], 1)])

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'berlekamp'
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'zassenhaus'
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'shoup'
    config.setup('GF_FACTOR_METHOD', 'shoup')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 使用 gf_from_dict 函数创建有限域元素 f 和素数 p
    f, p = gf_from_dict({63: ZZ(8), 0: ZZ(5)}, 11, ZZ), 11

    # 预期的有限域因式分解结果 g
    g = (8, [([1, 7], 1),
             ([1, 4, 5], 1),
             ([1, 6, 8, 2], 1),
             ([1, 9, 9, 2], 1),
             ([1, 0, 0, 9, 0, 0, 4], 1),
             ([1, 2, 0, 8, 4, 6, 4], 1),
             ([1, 2, 3, 8, 0, 6, 4], 1),
             ([1, 2, 6, 0, 8, 4, 4], 1),
             ([1, 3, 3, 1, 6, 8, 4], 1),
             ([1, 5, 6, 0, 8, 6, 4], 1),
             ([1, 6, 2, 7, 9, 8, 4], 1),
             ([1, 10, 4, 7, 10, 7, 4], 1),
             ([1, 10, 10, 1, 4, 9, 4], 1)])

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'berlekamp'
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'zassenhaus'
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'shoup'
    config.setup('GF_FACTOR_METHOD', 'shoup')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置 p 为适当的素数，以便创建 Gathen 多项式
    p = ZZ(nextprime(int((2**15 * pi).evalf())))
    # 使用 gf_from_dict 函数创建有限域元素 f 和素数 p
    f = gf_from_dict({15: 1, 1: 1, 0: 1}, p, ZZ)

    # 断言，验证调用 gf_sqf_p 函数判断 f 是否为平方因子化多项式
    assert gf_sqf_p(f, p, ZZ) is True

    # 预期的平方因子分解结果 g
    g = (1, [([1, 22730, 68144], 1),
             ([1, 81553, 77449, 86810, 4724], 1),
             ([1, 86276, 56779, 14859, 31575], 1),
             ([1, 15347, 95022, 84569, 94508, 92335], 1)])

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'zassenhaus'
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 设置全局配置，指定 GF_FACTOR_METHOD 为 'shoup'
    config.setup('GF_FACTOR_METHOD', 'shoup')
    # 断言，验证调用 gf_factor 函数计算得到的结果是否与预期值 g 相等
    assert gf_factor(f, p, ZZ) == g

    # 预期的平方自由因子分解结果 g
    g = (1, [[1, 22730, 68144
    # 断言检查 gf_factor_sqf 函数的返回结果是否与 g 相等
    assert gf_factor_sqf(f, p, ZZ) == g

    # 设置 Shoup 多项式的参数：f = a_0 x**n + a_1 x**(n-1) + ... + a_n
    # (mod p > 2**(n-2) * pi)，其中 a_n = a_{n-1}**2 + 1, a_0 = 1
    p = ZZ(nextprime(int((2**4 * pi).evalf())))
    f = ZZ.map([1, 2, 5, 26, 41, 39, 38])

    # 断言检查 gf_sqf_p 函数是否返回 True，表示给定多项式 f 在有限域 p 上是平方因子的
    assert gf_sqf_p(f, p, ZZ) is True

    # 预期的分解结果 g
    g = (1, [([1, 44, 26], 1),
             ([1, 11, 25, 18, 30], 1)])

    # 设置 GF_FACTOR_METHOD 为 'zassenhaus'，并断言 gf_factor 函数的返回结果与预期的 g 相等
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g

    # 设置 GF_FACTOR_METHOD 为 'shoup'，并断言 gf_factor 函数的返回结果与预期的 g 相等
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g

    # 预期的分解结果 g，改写为嵌套列表形式
    g = (1, [[1, 44, 26],
             [1, 11, 25, 18, 30]])

    # 设置 GF_FACTOR_METHOD 为 'zassenhaus'，并断言 gf_factor_sqf 函数的返回结果与预期的 g 相等
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor_sqf(f, p, ZZ) == g

    # 设置 GF_FACTOR_METHOD 为 'shoup'，并断言 gf_factor_sqf 函数的返回结果与预期的 g 相等
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor_sqf(f, p, ZZ) == g

    # 设置 GF_FACTOR_METHOD 为 'other'，预期会引发 KeyError 异常，因为 'other' 方法未实现
    raises(KeyError, lambda: gf_factor([1, 1], 11, ZZ))
    config.setup('GF_FACTOR_METHOD')
# 定义一个测试函数 `test_gf_csolve`，用于测试以下几个函数的正确性
def test_gf_csolve():
    # 断言检查 `gf_value([1, 7, 2, 4], 11)` 的返回值是否为 2204
    assert gf_value([1, 7, 2, 4], 11) == 2204

    # 断言检查 `linear_congruence(4, 3, 5)` 的返回值是否为 [2]
    assert linear_congruence(4, 3, 5) == [2]
    # 断言检查 `linear_congruence(0, 3, 5)` 的返回值是否为空列表 []
    assert linear_congruence(0, 3, 5) == []
    # 断言检查 `linear_congruence(6, 1, 4)` 的返回值是否为空列表 []
    assert linear_congruence(6, 1, 4) == []
    # 断言检查 `linear_congruence(0, 5, 5)` 的返回值是否为 [0, 1, 2, 3, 4]
    assert linear_congruence(0, 5, 5) == [0, 1, 2, 3, 4]
    # 断言检查 `linear_congruence(3, 12, 15)` 的返回值是否为 [4, 9, 14]
    assert linear_congruence(3, 12, 15) == [4, 9, 14]
    # 断言检查 `linear_congruence(6, 0, 18)` 的返回值是否为 [0, 3, 6, 9, 12, 15]
    assert linear_congruence(6, 0, 18) == [0, 3, 6, 9, 12, 15]

    # 使用 `_csolve_prime_las_vegas([2, 3, 1], 5)` 进行断言，检查返回值是否为 [2, 4]
    assert _csolve_prime_las_vegas([2, 3, 1], 5) == [2, 4]
    # 使用 `_csolve_prime_las_vegas([2, 0, 1], 5)` 进行断言，检查返回值是否为空列表 []
    assert _csolve_prime_las_vegas([2, 0, 1], 5) == []

    # 导入 `primerange` 函数，用于生成素数范围内的素数
    from sympy.ntheory import primerange
    # 对于在素数范围 [2, 100) 内的每个素数 p 进行以下操作
    for p in primerange(2, 100):
        # 计算 f = x**(p-1) - 1，并使用 `_csolve_prime_las_vegas` 函数进行断言，检查返回值是否为 [1, 2, ..., p-1]
        f = gf_sub_ground(gf_pow([1, 0], p - 1, p, ZZ), 1, p, ZZ)
        assert _csolve_prime_las_vegas(f, p) == list(range(1, p))

    # 断言检查 `csolve_prime([1, 3, 2, 17], 7)` 的返回值是否为 [3]
    assert csolve_prime([1, 3, 2, 17], 7) == [3]
    # 断言检查 `csolve_prime([1, 3, 1, 5], 5)` 的返回值是否为 [0, 1]
    assert csolve_prime([1, 3, 1, 5], 5) == [0, 1]
    # 断言检查 `csolve_prime([3, 6, 9, 3], 3)` 的返回值是否为 [0, 1, 2]
    assert csolve_prime([3, 6, 9, 3], 3) == [0, 1, 2]

    # 断言检查 `csolve_prime([1, 1, 223], 3, 4)` 的返回值是否为 [4, 13, 22, 31, 40, 49, 58, 67, 76]
    assert csolve_prime([1, 1, 223], 3, 4) == [4, 13, 22, 31, 40, 49, 58, 67, 76]
    # 断言检查 `csolve_prime([3, 5, 2, 25], 5, 3)` 的返回值是否为 [16, 50, 99]
    assert csolve_prime([3, 5, 2, 25], 5, 3) == [16, 50, 99]
    # 断言检查 `csolve_prime([3, 2, 2, 49], 7, 3)` 的返回值是否为 [147, 190, 234]
    assert csolve_prime([3, 2, 2, 49], 7, 3) == [147, 190, 234]

    # 断言检查 `gf_csolve([1, 1, 7], 189)` 的返回值是否为 [13, 49, 76, 112, 139, 175]
    assert gf_csolve([1, 1, 7], 189) == [13, 49, 76, 112, 139, 175]
    # 断言检查 `gf_csolve([1, 3, 4, 1, 30], 60)` 的返回值是否为 [10, 30]
    assert gf_csolve([1, 3, 4, 1, 30], 60) == [10, 30]
    # 断言检查 `gf_csolve([1, 1, 7], 15)` 的返回值是否为空列表 []
    assert gf_csolve([1, 1, 7], 15) == []
```