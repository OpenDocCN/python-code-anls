# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_qs.py`

```
# 导入 __future__ 模块的 annotations 功能，使得函数的类型提示支持在函数内使用其自身类型
from __future__ import annotations

# 导入 sympy.ntheory 中的 qs 函数及相关模块
from sympy.ntheory import qs
from sympy.ntheory.qs import SievePolynomial, _generate_factor_base, \
    _initialize_first_polynomial, _initialize_ith_poly, \
    _gen_sieve_array, _check_smoothness, _trial_division_stage, _gauss_mod_2, \
    _build_matrix, _find_factor

# 导入 sympy.testing.pytest 中的 slow 装饰器，用于标记测试函数为较慢的测试用例
from sympy.testing.pytest import slow

# 使用 @slow 装饰器标记的测试函数，用于测试 qs 函数
@slow
def test_qs_1():
    # 断言调用 qs 函数并验证其返回结果是否符合预期
    assert qs(10009202107, 100, 10000) == {100043, 100049}
    assert qs(211107295182713951054568361, 1000, 10000) == \
        {13791315212531, 15307263442931}
    assert qs(980835832582657*990377764891511, 3000, 50000) == \
        {980835832582657, 990377764891511}
    assert qs(18640889198609*20991129234731, 1000, 50000) == \
        {18640889198609, 20991129234731}

# 用于测试 SievePolynomial 类及相关函数的测试函数
def test_qs_2() -> None:
    n = 10009202107
    M = 50
    # 创建一个 SievePolynomial 对象，设置参数和系数，并测试其 eval 方法的结果
    sieve_poly = SievePolynomial([100,  1600, -10009195707], 10, 80)
    assert sieve_poly.eval(10) == -10009169707
    assert sieve_poly.eval(5) == -10009185207

    # 调用 _generate_factor_base 函数生成因子基和相关索引，进行断言验证
    idx_1000, idx_5000, factor_base = _generate_factor_base(2000, n)
    assert idx_1000 == 82
    assert [factor_base[i].prime for i in range(15)] == \
        [2, 3, 7, 11, 17, 19, 29, 31, 43, 59, 61, 67, 71, 73, 79]
    assert [factor_base[i].tmem_p for i in range(15)] == \
        [1, 1, 3, 5, 3, 6, 6, 14, 1, 16, 24, 22, 18, 22, 15]
    assert [factor_base[i].log_p for i in range(5)] == \
        [710, 1125, 1993, 2455, 2901]

    # 调用 _initialize_first_polynomial 函数初始化第一个多项式和参数 B
    g, B = _initialize_first_polynomial(
        n, M, factor_base, idx_1000, idx_5000, seed=0)
    assert g.a == 1133107
    assert g.b == 682543
    assert B == [272889, 409654]
    assert [factor_base[i].soln1 for i in range(15)] == \
        [0, 0, 3, 7, 13, 0, 8, 19, 9, 43, 27, 25, 63, 29, 19]
    assert [factor_base[i].soln2 for i in range(15)] == \
        [0, 1, 1, 3, 12, 16, 15, 6, 15, 1, 56, 55, 61, 58, 16]
    assert [factor_base[i].a_inv for i in range(15)] == \
        [1, 1, 5, 7, 3, 5, 26, 6, 40, 5, 21, 45, 4, 1, 8]
    assert [factor_base[i].b_ainv for i in range(5)] == \
        [[0, 0], [0, 2], [3, 0], [3, 9], [13, 13]]

    # 调用 _initialize_ith_poly 函数初始化第 i 个多项式，并验证其结果
    g_1 = _initialize_ith_poly(n, factor_base, 1, g, B)
    assert g_1.a == 1133107
    assert g_1.b == 136765

    # 调用 _gen_sieve_array 函数生成筛选数组 sieve_array，并进行断言验证
    sieve_array = _gen_sieve_array(M, factor_base)
    assert sieve_array[0:5] == [8424, 13603, 1835, 5335, 710]

    # 调用 _check_smoothness 函数检查数值的平滑性，并进行断言验证
    assert _check_smoothness(9645, factor_base) == (5, False)
    assert _check_smoothness(210313, factor_base)[0][0:15] == \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    assert _check_smoothness(210313, factor_base)[1]

    # 初始化 partial_relations 字典，调用 _trial_division_stage 函数执行试除法阶段
    partial_relations: dict[int, tuple[int, int]] = {}
    smooth_relation, partial_relation = _trial_division_stage(
        n, M, factor_base, sieve_array, sieve_poly, partial_relations,
        ERROR_TERM=25*2**10)

    # 断言 partial_relations 字典中的部分关系是否符合预期
    assert partial_relations == {
        8699: (440, -10009008507),
        166741: (490, -10008962007),
        131449: (530, -10008921207),
        6653: (550, -10008899607)
    }
    # 断言检查 smooth_relation 列表中每个元素的第一个值是否与预期列表相等
    assert [smooth_relation[i][0] for i in range(5)] == [
        -250, -670615476700, -45211565844500, -231723037747200, -1811665537200]
    
    # 断言检查 smooth_relation 列表中每个元素的第二个值是否与预期列表相等
    assert [smooth_relation[i][1] for i in range(5)] == [
        -10009139607, 1133094251961, 5302606761, 53804049849, 1950723889]
    
    # 断言检查 smooth_relation 列表中第一个元素的第三个子列表的前15个元素是否与预期列表相等
    assert smooth_relation[0][2][0:15] == [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # 断言检查 _gauss_mod_2 函数的返回值是否与预期值相等
    assert _gauss_mod_2(
        [[0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]]
    ) == (
        # 第一个断言结果：列表包含两个元素，每个元素是包含列表和整数的元组
        [[[0, 1, 1], 3], [[0, 1, 1], 4]],
        # 第二个断言结果：列表，表示每个输入行向量是否为非零向量
        [True, True, True, False, False],
        # 第三个断言结果：与输入矩阵相同的矩阵，但以模 2 的方式进行高斯消元后的结果
        [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1]]
    )
# 定义一个测试函数 test_qs_3
def test_qs_3():
    # 设定常量 N 为 1817
    N = 1817
    # 初始化 smooth_relations 列表，包含多个元组，每个元组包括三个值
    smooth_relations = [
        (2455024, 637, [0, 0, 0, 1]),
        (-27993000, 81536, [0, 1, 0, 1]),
        (11461840, 12544, [0, 0, 0, 0]),
        (149, 20384, [0, 1, 0, 1]),
        (-31138074, 19208, [0, 1, 0, 0])
    ]

    # 调用 _build_matrix 函数，根据 smooth_relations 构建矩阵
    matrix = _build_matrix(smooth_relations)
    # 断言 matrix 是否等于预期的值
    assert matrix == [
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0]
    ]

    # 调用 _gauss_mod_2 函数，对 matrix 执行高斯消元，得到 dependent_row, mark, gauss_matrix
    dependent_row, mark, gauss_matrix = _gauss_mod_2(matrix)
    # 断言 dependent_row 是否等于预期的值
    assert dependent_row == [[[0, 0, 0, 0], 2], [[0, 1, 0, 0], 3]]
    # 断言 mark 是否等于预期的值
    assert mark == [True, True, False, False, True]
    # 断言 gauss_matrix 是否等于预期的值
    assert gauss_matrix == [
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1]
    ]

    # 调用 _find_factor 函数，寻找因子，参数包括 dependent_row, mark, gauss_matrix, 0, smooth_relations, N
    factor = _find_factor(
        dependent_row, mark, gauss_matrix, 0, smooth_relations, N)
    # 断言 factor 是否等于预期的值
    assert factor == 23
```