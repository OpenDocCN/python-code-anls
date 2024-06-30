# `D:\src\scipysrc\sympy\sympy\polys\tests\test_densetools.py`

```
"""Tests for dense recursive polynomials' tools. """

# 导入密集多项式基础模块中的函数和类
from sympy.polys.densebasic import (
    dup_normal, dmp_normal,
    dup_from_raw_dict,
    dmp_convert, dmp_swap,
)

# 导入密集多项式算术操作模块中的函数
from sympy.polys.densearith import dmp_mul_ground

# 导入密集多项式工具模块中的函数
from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,
    dup_integrate, dmp_integrate, dmp_integrate_in,
    dup_diff, dmp_diff, dmp_diff_in,
    dup_eval, dmp_eval, dmp_eval_in,
    dmp_eval_tail, dmp_diff_eval_in,
    dup_trunc, dmp_trunc, dmp_ground_trunc,
    dup_monic, dmp_ground_monic,
    dup_content, dmp_ground_content,
    dup_primitive, dmp_ground_primitive,
    dup_extract, dmp_ground_extract,
    dup_real_imag,
    dup_mirror, dup_scale, dup_shift, dmp_shift,
    dup_transform,
    dup_compose, dmp_compose,
    dup_decompose,
    dmp_lift,
    dup_sign_variations,
    dup_revert, dmp_revert,
)

# 导入多项式类 ANP
from sympy.polys.polyclasses import ANP

# 导入多项式相关的异常类
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    ExactQuotientFailed,
    NotReversible,
    DomainError,
)

# 导入特殊多项式 f_polys
from sympy.polys.specialpolys import f_polys

# 导入多项式域和环的定义
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, EX, RR
from sympy.polys.rings import ring

# 导入核心数学类和函数
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.trigonometric import sin

# 导入符号变量 x
from sympy.abc import x

# 导入测试框架中的 raises 函数
from sympy.testing.pytest import raises

# 生成多项式列表 f_0 到 f_6，转换为密集表示
f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]

# 定义测试函数 test_dup_integrate
def test_dup_integrate():
    # 测试 dup_integrate 函数的基本用法
    assert dup_integrate([], 1, QQ) == []
    assert dup_integrate([], 2, QQ) == []

    # 测试在多项式 [1] 上进行一次和两次积分
    assert dup_integrate([QQ(1)], 1, QQ) == [QQ(1), QQ(0)]
    assert dup_integrate([QQ(1)], 2, QQ) == [QQ(1, 2), QQ(0), QQ(0)]

    # 测试在多项式 [1, 2, 3] 上进行零次、一次、二次和三次积分
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 0, QQ) == \
        [QQ(1), QQ(2), QQ(3)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 1, QQ) == \
        [QQ(1, 3), QQ(1), QQ(3), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 2, QQ) == \
        [QQ(1, 12), QQ(1, 3), QQ(3, 2), QQ(0), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 3, QQ) == \
        [QQ(1, 60), QQ(1, 12), QQ(1, 2), QQ(0), QQ(0), QQ(0)]

    # 测试在由原始字典 {29: 17} 构成的多项式上进行三次积分
    assert dup_integrate(dup_from_raw_dict({29: QQ(17)}, QQ), 3, QQ) == \
        dup_from_raw_dict({32: QQ(17, 29760)}, QQ)

    # 测试在由原始字典 {29: 17, 5: 1/2} 构成的多项式上进行三次积分
    assert dup_integrate(dup_from_raw_dict({29: QQ(17), 5: QQ(1, 2)}, QQ), 3, QQ) == \
        dup_from_raw_dict({32: QQ(17, 29760), 8: QQ(1, 672)}, QQ)

# 定义测试函数 test_dmp_integrate
def test_dmp_integrate():
    # 测试 dmp_integrate 函数在 [1] 上进行两次积分
    assert dmp_integrate([QQ(1)], 2, 0, QQ) == [QQ(1, 2), QQ(0), QQ(0)]

    # 测试 dmp_integrate 函数在 [[[1]]] 上进行一次和两次积分
    assert dmp_integrate([[[]]], 1, 2, QQ) == [[[]]]
    assert dmp_integrate([[[]]], 2, 2, QQ) == [[[]]]

    # 测试 dmp_integrate 函数在 [[1], [2], [3]] 上进行零次和一次积分
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 0, 1, QQ) == \
        [[QQ(1)], [QQ(2)], [QQ(3)]]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 1, 1, QQ) == \
        [[QQ(1, 3)], [QQ(1)], [QQ(3)], []]
    # 断言语句，验证 dmp_integrate 函数的返回结果是否符合预期
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 2, 1, QQ) == \
        [[QQ(1, 12)], [QQ(1, 3)], [QQ(3, 2)], [], []]
    # 断言语句，验证 dmp_integrate 函数的返回结果是否符合预期
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 3, 1, QQ) == \
        [[QQ(1, 60)], [QQ(1, 12)], [QQ(1, 2)], [], [], []]
# 定义一个测试函数，用于测试 dmp_integrate_in 函数的行为
def test_dmp_integrate_in():
    # 将 f_6 转换为有理数多项式，采用 3 次系数的整数环 QQ
    f = dmp_convert(f_6, 3, ZZ, QQ)

    # 断言使用 dmp_integrate_in 函数对 f 进行积分，然后与交换操作的结果相等
    assert dmp_integrate_in(f, 2, 1, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 1, 3, QQ), 2, 3, QQ), 0, 1, 3, QQ)
    
    # 断言使用 dmp_integrate_in 函数对 f 进行积分，然后与交换操作的结果相等
    assert dmp_integrate_in(f, 3, 1, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 1, 3, QQ), 3, 3, QQ), 0, 1, 3, QQ)
    
    # 断言使用 dmp_integrate_in 函数对 f 进行积分，然后与交换操作的结果相等
    assert dmp_integrate_in(f, 2, 2, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 2, 3, QQ), 2, 3, QQ), 0, 2, 3, QQ)
    
    # 断言使用 dmp_integrate_in 函数对 f 进行积分，然后与交换操作的结果相等
    assert dmp_integrate_in(f, 3, 2, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 2, 3, QQ), 3, 3, QQ), 0, 2, 3, QQ)

    # 断言调用 dmp_integrate_in 函数时，对于无效索引抛出 IndexError 异常
    raises(IndexError, lambda: dmp_integrate_in(f, 1, -1, 3, QQ))
    
    # 断言调用 dmp_integrate_in 函数时，对于无效索引抛出 IndexError 异常
    raises(IndexError, lambda: dmp_integrate_in(f, 1, 4, 3, QQ))


# 定义一个测试函数，用于测试 dup_diff 函数的行为
def test_dup_diff():
    # 断言对空列表调用 dup_diff 函数返回空列表
    assert dup_diff([], 1, ZZ) == []
    
    # 断言对包含单个元素的列表调用 dup_diff 函数返回空列表
    assert dup_diff([7], 1, ZZ) == []
    
    # 断言对包含两个元素的列表调用 dup_diff 函数返回第一个元素
    assert dup_diff([2, 7], 1, ZZ) == [2]
    
    # 断言对包含三个元素的列表调用 dup_diff 函数返回相邻元素之差组成的列表
    assert dup_diff([1, 2, 1], 1, ZZ) == [2, 2]
    
    # 断言对包含四个元素的列表调用 dup_diff 函数返回相邻元素之差组成的列表
    assert dup_diff([1, 2, 3, 4], 1, ZZ) == [3, 4, 3]
    
    # 断言对包含多个元素的列表调用 dup_diff 函数返回相邻元素之差组成的列表
    assert dup_diff([1, -1, 0, 0, 2], 1, ZZ) == [4, -3, 0, 0]

    # 创建一个长列表 f
    f = dup_normal([17, 34, 56, -345, 23, 76, 0, 0, 12, 3, 7], ZZ)

    # 断言对 f 调用 dup_diff 函数时，阶数为 0 时返回 f 本身
    assert dup_diff(f, 0, ZZ) == f
    
    # 断言对 f 调用 dup_diff 函数时，阶数为 1 时返回相邻元素之差组成的列表
    assert dup_diff(f, 1, ZZ) == [170, 306, 448, -2415, 138, 380, 0, 0, 24, 3]
    
    # 断言对 f 调用 dup_diff 函数时，阶数为 2 时与连续两次阶数为 1 的调用结果相同
    assert dup_diff(f, 2, ZZ) == dup_diff(dup_diff(f, 1, ZZ), 1, ZZ)
    
    # 断言对 f 调用 dup_diff 函数时，阶数为 3 时与连续三次阶数为 1 的调用结果相同
    assert dup_diff(f, 3, ZZ) == dup_diff(dup_diff(dup_diff(f, 1, ZZ), 1, ZZ), 1, ZZ)

    # 创建有限域 FF(3)
    K = FF(3)
    
    # 将 f 转换为 K 上的多项式
    f = dup_normal([17, 34, 56, -345, 23, 76, 0, 0, 12, 3, 7], K)

    # 断言对 K 上的 f 调用 dup_diff 函数时，阶数为 1 时返回相邻元素之差组成的列表
    assert dup_diff(f, 1, K) == dup_normal([2, 0, 1, 0, 0, 2, 0, 0, 0, 0], K)
    
    # 断言对 K 上的 f 调用 dup_diff 函数时，阶数为 2 时与连续两次阶数为 1 的调用结果相同
    assert dup_diff(f, 2, K) == dup_diff(dup_diff(f, 1, K), 1, K)
    
    # 断言对 K 上的 f 调用 dup_diff 函数时，阶数为 3 时与连续三次阶数为 1 的调用结果相同
    assert dup_diff(f, 3, K) == dup_diff(dup_diff(dup_diff(f, 1, K), 1, K), 1, K)


# 定义一个测试函数，用于测试 dmp_diff 函数的行为
def test_dmp_diff():
    # 断言对空列表调用 dmp_diff 函数返回空列表
    assert dmp_diff([], 1, 0, ZZ) == []
    
    # 断言对包含单个空列表的列表调用 dmp_diff 函数返回单个空列表的列表
    assert dmp_diff([[]], 1, 1, ZZ) == [[]]
    
    # 断言对包含单个空列表的列表调用 dmp_diff 函数返回单个空列表的列表的列表
    assert dmp_diff([[[]]], 1, 2, ZZ) == [[[]]]

    # 断言对包含复杂结构的列表调用 dmp_diff 函数返回特定结构的列表
    assert dmp_diff([[[1], [2]]], 1, 2, ZZ) == [[[]]]

    # 断言对包含特定结构的列表调用 dmp_diff 函数返回特定结构的列表
    assert dmp_diff([[[1]], [[]]], 1, 2, ZZ) == [[[1]]]
    
    # 断言对包含复杂结构的列表调用 dmp_diff 函数返回特定结构的列表
    assert dmp_diff([[[3]], [[1]], [[]]], 1, 2, ZZ) == [[[6]], [[1]]]

    # 断言对普通多项式调用 dmp_diff 函数与调用 dup_diff 函数结果相同
    assert dmp_diff([1, -1, 0, 0, 2], 1, 0, ZZ) == \
        dup_diff([1, -1, 0, 0, 2], 1, ZZ)

    # 断言对 f_6 调用 dmp_diff 函数时，阶数为 0 时返回 f_6 本身
    assert dmp_diff
    # 断言：验证对于给定的多项式 F_6 和整数 2，3，域 K，进行一阶偏导数后的结果与先进行两次一阶偏导数的结果是否相同
    assert dmp_diff(F_6, 2, 3, K) == dmp_diff(dmp_diff(F_6, 1, 3, K), 1, 3, K)
    
    # 断言：验证对于给定的多项式 F_6 和整数 3，3，域 K，进行一阶偏导数后的结果与先进行三次一阶偏导数的结果是否相同
    assert dmp_diff(F_6, 3, 3, K) == dmp_diff(dmp_diff(dmp_diff(F_6, 1, 3, K), 1, 3, K), 1, 3, K)
def test_dmp_diff_in():
    # 测试 dmp_diff_in 函数，验证其结果是否符合预期
    assert dmp_diff_in(f_6, 2, 1, 3, ZZ) == \
        # 调用 dmp_diff_in 的结果应该等于 dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 2, 3, ZZ), 0, 1, 3, ZZ)
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 2, 3, ZZ), 0, 1, 3, ZZ)
    assert dmp_diff_in(f_6, 3, 1, 3, ZZ) == \
        # 调用 dmp_diff_in 的结果应该等于 dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 3, 3, ZZ), 0, 1, 3, ZZ)
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 3, 3, ZZ), 0, 1, 3, ZZ)
    assert dmp_diff_in(f_6, 2, 2, 3, ZZ) == \
        # 调用 dmp_diff_in 的结果应该等于 dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 2, 3, ZZ), 0, 2, 3, ZZ)
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 2, 3, ZZ), 0, 2, 3, ZZ)
    assert dmp_diff_in(f_6, 3, 2, 3, ZZ) == \
        # 调用 dmp_diff_in 的结果应该等于 dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 3, 3, ZZ), 0, 2, 3, ZZ)
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 3, 3, ZZ), 0, 2, 3, ZZ)

    # 测试是否能正确抛出 IndexError 异常
    raises(IndexError, lambda: dmp_diff_in(f_6, 1, -1, 3, ZZ))
    raises(IndexError, lambda: dmp_diff_in(f_6, 1, 4, 3, ZZ))


def test_dup_eval():
    # 测试 dup_eval 函数的不同输入情况下的返回值是否符合预期
    assert dup_eval([], 7, ZZ) == 0
    assert dup_eval([1, 2], 0, ZZ) == 2
    assert dup_eval([1, 2, 3], 7, ZZ) == 66


def test_dmp_eval():
    # 测试 dmp_eval 函数的不同输入情况下的返回值是否符合预期

    assert dmp_eval([], 3, 0, ZZ) == 0

    assert dmp_eval([[]], 3, 1, ZZ) == []
    assert dmp_eval([[[]]], 3, 2, ZZ) == [[]]

    assert dmp_eval([[1, 2]], 0, 1, ZZ) == [1, 2]

    assert dmp_eval([[[1]]], 3, 2, ZZ) == [[1]]
    assert dmp_eval([[[1, 2]]], 3, 2, ZZ) == [[1, 2]]

    assert dmp_eval([[3, 2], [1, 2]], 3, 1, ZZ) == [10, 8]
    assert dmp_eval([[[3, 2]], [[1, 2]]], 3, 2, ZZ) == [[10, 8]]


def test_dmp_eval_in():
    # 测试 dmp_eval_in 函数，验证其结果是否符合预期

    assert dmp_eval_in(
        f_6, -2, 1, 3, ZZ) == dmp_eval(dmp_swap(f_6, 0, 1, 3, ZZ), -2, 3, ZZ)
    assert dmp_eval_in(
        f_6, 7, 1, 3, ZZ) == dmp_eval(dmp_swap(f_6, 0, 1, 3, ZZ), 7, 3, ZZ)
    assert dmp_eval_in(f_6, -2, 2, 3, ZZ) == dmp_swap(
        dmp_eval(dmp_swap(f_6, 0, 2, 3, ZZ), -2, 3, ZZ), 0, 1, 2, ZZ)
    assert dmp_eval_in(f_6, 7, 2, 3, ZZ) == dmp_swap(
        dmp_eval(dmp_swap(f_6, 0, 2, 3, ZZ), 7, 3, ZZ), 0, 1, 2, ZZ)

    f = [[[int(45)]], [[]], [[]], [[int(-9)], [-1], [], [int(3), int(0), int(10), int(0)]]]

    assert dmp_eval_in(f, -2, 2, 2, ZZ) == \
        [[45], [], [], [-9, -1, 0, -44]]

    # 测试是否能正确抛出 IndexError 异常
    raises(IndexError, lambda: dmp_eval_in(f_6, ZZ(1), -1, 3, ZZ))
    raises(IndexError, lambda: dmp_eval_in(f_6, ZZ(1), 4, 3, ZZ))


def test_dmp_eval_tail():
    # 测试 dmp_eval_tail 函数的不同输入情况下的返回值是否符合预期

    assert dmp_eval_tail([[]], [1], 1, ZZ) == []
    assert dmp_eval_tail([[[]]], [1], 2, ZZ) == [[]]
    assert dmp_eval_tail([[[]]], [1, 2], 2, ZZ) == []

    assert dmp_eval_tail(f_0, [], 2, ZZ) == f_0

    assert dmp_eval_tail(f_0, [1, -17, 8], 2, ZZ) == 84496
    assert dmp_eval_tail(f_0, [-17, 8], 2, ZZ) == [-1409, 3, 85902]
    assert dmp_eval_tail(f_0, [8], 2, ZZ) == [[83, 2], [3], [302, 81, 1]]

    assert dmp_eval_tail(f_1, [-17, 8], 2, ZZ) == [-136, 15699, 9166, -27144]

    assert dmp_eval_tail(
        f_2, [-12, 3], 2, ZZ) == [-1377, 0, -702, -1224, 0, -624]
    assert dmp_eval_tail(
        f_3, [-12, 3], 2, ZZ) == [144, 82, -5181, -28872, -14868, -540]

    assert dmp_eval_tail(
        f_4, [25, -1], 2, ZZ) == [152587890625, 9765625, -59605407714843750,
        -3839159765625, -1562475, 9536712644531250, 610349546750, -4, 24414375000, 1562520]
    assert dmp_eval_tail(f_5, [25, -1], 2, ZZ) == [-1, -78, -2028, -17576]
    # 使用断言来验证 dmp_eval_tail 函数的返回结果是否符合预期
    assert dmp_eval_tail(f_6, [0, 2, 4], 3, ZZ) == [5040, 0, 0, 4480]
# 测试函数，用于验证 dmp_diff_eval_in 函数的正确性
def test_dmp_diff_eval_in():
    # 断言 dmp_diff_eval_in 函数的返回值与 dmp_eval 函数的返回值相等
    assert dmp_diff_eval_in(f_6, 2, 7, 1, 3, ZZ) == \
        dmp_eval(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 2, 3, ZZ), 7, 3, ZZ)

    # 断言 dmp_diff_eval_in 函数的返回值与 dmp_eval 函数的返回值相等
    assert dmp_diff_eval_in(f_6, 2, 7, 0, 3, ZZ) == \
        dmp_eval(dmp_diff(f_6, 2, 3, ZZ), 7, 3, ZZ)

    # 验证 dmp_diff_eval_in 函数在索引错误时会抛出 IndexError 异常
    raises(IndexError, lambda: dmp_diff_eval_in(f_6, 1, ZZ(1), 4, 3, ZZ))


# 测试函数，用于验证 dup_revert 函数的正确性
def test_dup_revert():
    # 定义原始列表 f 和期望的结果列表 g
    f = [-QQ(1, 720), QQ(0), QQ(1, 24), QQ(0), -QQ(1, 2), QQ(0), QQ(1)]
    g = [QQ(61, 720), QQ(0), QQ(5, 24), QQ(0), QQ(1, 2), QQ(0), QQ(1)]

    # 断言 dup_revert 函数的返回值与预期的结果列表 g 相等
    assert dup_revert(f, 8, QQ) == g

    # 验证 dup_revert 函数在不可逆的情况下会抛出 NotReversible 异常
    raises(NotReversible, lambda: dup_revert([QQ(1), QQ(0)], 3, QQ))


# 测试函数，用于验证 dmp_revert 函数的正确性
def test_dmp_revert():
    # 定义原始列表 f 和期望的结果列表 g
    f = [-QQ(1, 720), QQ(0), QQ(1, 24), QQ(0), -QQ(1, 2), QQ(0), QQ(1)]
    g = [QQ(61, 720), QQ(0), QQ(5, 24), QQ(0), QQ(1, 2), QQ(0), QQ(1)]

    # 断言 dmp_revert 函数的返回值与预期的结果列表 g 相等
    assert dmp_revert(f, 8, 0, QQ) == g

    # 验证 dmp_revert 函数在多元多项式错误时会抛出 MultivariatePolynomialError 异常
    raises(MultivariatePolynomialError, lambda: dmp_revert([[1]], 2, 1, QQ))


# 测试函数，用于验证 dup_trunc 函数的正确性
def test_dup_trunc():
    # 断言 dup_trunc 函数对给定的列表进行截断后的结果与预期的结果相等
    assert dup_trunc([1, 2, 3, 4, 5, 6], ZZ(3), ZZ) == [1, -1, 0, 1, -1, 0]
    assert dup_trunc([6, 5, 4, 3, 2, 1], ZZ(3), ZZ) == [-1, 1, 0, -1, 1]

    # 定义不同的环 R、K，并验证 dup_trunc 函数对其处理的结果
    R = ZZ_I
    assert dup_trunc([R(3), R(4), R(5)], R(3), R) == [R(1), R(-1)]

    K = FF(5)
    assert dup_trunc([K(3), K(4), K(5)], K(3), K) == [K(1), K(0)]


# 测试函数，用于验证 dmp_trunc 函数的正确性
def test_dmp_trunc():
    # 断言 dmp_trunc 函数对给定的多项式列表进行截断后的结果与预期的结果相等
    assert dmp_trunc([[]], [1, 2], 2, ZZ) == [[]]
    assert dmp_trunc([[1, 2], [1, 4, 1], [1]], [1, 2], 1, ZZ) == [[-3], [1]]


# 测试函数，用于验证 dmp_ground_trunc 函数的正确性
def test_dmp_ground_trunc():
    # 断言 dmp_ground_trunc 函数对给定的多项式进行截断和规范化后的结果与预期的结果相等
    assert dmp_ground_trunc(f_0, ZZ(3), 2, ZZ) == \
        dmp_normal(
            [[[1, -1, 0], [-1]], [[]], [[1, -1, 0], [1, -1, 1], [1]]], 2, ZZ)


# 测试函数，用于验证 dup_monic 函数的正确性
def test_dup_monic():
    # 断言 dup_monic 函数对给定的整数列表进行首一化后的结果与预期的结果相等
    assert dup_monic([3, 6, 9], ZZ) == [1, 2, 3]

    # 验证 dup_monic 函数在无法进行整除时会抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: dup_monic([3, 4, 5], ZZ))

    # 验证 dup_monic 函数在空列表时返回空列表
    assert dup_monic([], QQ) == []
    assert dup_monic([QQ(1)], QQ) == [QQ(1)]
    assert dup_monic([QQ(7), QQ(1), QQ(21)], QQ) == [QQ(1), QQ(1, 7), QQ(3)]


# 测试函数，用于验证 dmp_ground_monic 函数的正确性
def test_dmp_ground_monic():
    # 断言 dmp_ground_monic 函数对给定的多项式列表进行首一化后的结果与预期的结果相等
    assert dmp_ground_monic([3, 6, 9], 0, ZZ) == [1, 2, 3]

    # 断言 dmp_ground_monic 函数对给定的多元多项式列表进行首一化后的结果与预期的结果相等
    assert dmp_ground_monic([[3], [6], [9]], 1, ZZ) == [[1], [2], [3]]

    # 验证 dmp_ground_monic 函数在无法进行整除时会抛出 ExactQuotientFailed 异常
    raises(
        ExactQuotientFailed, lambda: dmp_ground_monic([[3], [4], [5]], 1, ZZ))

    # 验证 dmp_ground_monic 函数在空列表时返回空列表
    assert dmp_ground_monic([[]], 1, QQ) == [[]]
    assert dmp_ground_monic([[QQ(1)]], 1, QQ) == [[QQ(1)]]
    assert dmp_ground_monic(
        [[QQ(7)], [QQ(1)], [QQ(21)]], 1, QQ) == [[QQ(1)], [QQ(1, 7)], [QQ(3)]]


# 测试函数，用于验证 dup_content 函数的正确性
def test_dup_content():
    # 断言 dup_content 函数对给定的整数列表计算内容后的结果与预期的结果相等
    assert dup_content([], ZZ) == ZZ(0)
    assert dup_content([1], ZZ) == ZZ(1)
    assert dup_content([-1], ZZ) == ZZ(1)
    assert dup_content([1, 1], ZZ) == ZZ(1)
    assert dup_content([2, 2], ZZ) == ZZ(2)
    assert dup_content([1, 2, 1], ZZ) == ZZ(1)
    assert dup_content([2, 4, 2], ZZ) == ZZ(2)

    # 验证 dup_content 函数对给定的有理数列表计算内容后的结果与预期的结果相等
    assert dup_content([QQ(2, 3), QQ(4, 9)], QQ) == QQ(2, 9)
    assert dup_content([QQ(2, 3), QQ(4, 5)], QQ) == QQ(2, 15)


# 测试函数，用于验证 dmp
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content([[-1]], 1, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content([[1], [1]], 1, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content([[2], [2]], 1, ZZ) == ZZ(2)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content([[1], [2], [1]], 1, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content([[2], [4], [2]], 1, ZZ) == ZZ(2)
    
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content([[QQ(2, 3)], [QQ(4, 9)]], 1, QQ) == QQ(2, 9)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content([[QQ(2, 3)], [QQ(4, 5)]], 1, QQ) == QQ(2, 15)
    
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(f_0, 2, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(dmp_mul_ground(f_0, ZZ(2), 2, ZZ), 2, ZZ) == ZZ(2)
    
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(f_1, 2, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(dmp_mul_ground(f_1, ZZ(3), 2, ZZ), 2, ZZ) == ZZ(3)
    
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(f_2, 2, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(dmp_mul_ground(f_2, ZZ(4), 2, ZZ), 2, ZZ) == ZZ(4)
    
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(f_3, 2, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(dmp_mul_ground(f_3, ZZ(5), 2, ZZ), 2, ZZ) == ZZ(5)
    
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(f_4, 2, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(dmp_mul_ground(f_4, ZZ(6), 2, ZZ), 2, ZZ) == ZZ(6)
    
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(f_5, 2, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(dmp_mul_ground(f_5, ZZ(7), 2, ZZ), 2, ZZ) == ZZ(7)
    
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(f_6, 3, ZZ) == ZZ(1)
    # 断言，验证 dmp_ground_content 函数对给定参数的返回值是否符合预期
    assert dmp_ground_content(dmp_mul_ground(f_6, ZZ(8), 3, ZZ), 3, ZZ) == ZZ(8)
# 定义测试函数 test_dup_primitive
def test_dup_primitive():
    # 断言空列表经过 dup_primitive 处理后，返回 ZZ 类型的 0 和空列表
    assert dup_primitive([], ZZ) == (ZZ(0), [])
    # 断言包含单个 ZZ(1) 元素的列表经过 dup_primitive 处理后，返回 ZZ 类型的 1 和包含 ZZ(1) 元素的列表
    assert dup_primitive([ZZ(1)], ZZ) == (ZZ(1), [ZZ(1)])
    # 断言包含两个 ZZ(1) 元素的列表经过 dup_primitive 处理后，返回 ZZ 类型的 1 和包含两个 ZZ(1) 元素的列表
    assert dup_primitive([ZZ(1), ZZ(1)], ZZ) == (ZZ(1), [ZZ(1), ZZ(1)])
    # 断言包含两个 ZZ(2) 元素的列表经过 dup_primitive 处理后，返回 ZZ 类型的 2 和包含两个 ZZ(1) 元素的列表
    assert dup_primitive([ZZ(2), ZZ(2)], ZZ) == (ZZ(2), [ZZ(1), ZZ(1)])
    # 断言包含 ZZ(1), ZZ(2), ZZ(1) 元素的列表经过 dup_primitive 处理后，返回 ZZ 类型的 1 和包含 ZZ(1), ZZ(2), ZZ(1) 元素的列表
    assert dup_primitive([ZZ(1), ZZ(2), ZZ(1)], ZZ) == (ZZ(1), [ZZ(1), ZZ(2), ZZ(1)])
    # 断言包含 ZZ(2), ZZ(4), ZZ(2) 元素的列表经过 dup_primitive 处理后，返回 ZZ 类型的 2 和包含 ZZ(1), ZZ(2), ZZ(1) 元素的列表
    assert dup_primitive([ZZ(2), ZZ(4), ZZ(2)], ZZ) == (ZZ(2), [ZZ(1), ZZ(2), ZZ(1)])

    # 断言空列表经过 dup_primitive 处理后，返回 QQ 类型的 0 和空列表
    assert dup_primitive([], QQ) == (QQ(0), [])
    # 断言包含单个 QQ(1) 元素的列表经过 dup_primitive 处理后，返回 QQ 类型的 1 和包含 QQ(1) 元素的列表
    assert dup_primitive([QQ(1)], QQ) == (QQ(1), [QQ(1)])
    # 断言包含两个 QQ(1) 元素的列表经过 dup_primitive 处理后，返回 QQ 类型的 1 和包含两个 QQ(1) 元素的列表
    assert dup_primitive([QQ(1), QQ(1)], QQ) == (QQ(1), [QQ(1), QQ(1)])
    # 断言包含两个 QQ(2) 元素的列表经过 dup_primitive 处理后，返回 QQ 类型的 2 和包含两个 QQ(1) 元素的列表
    assert dup_primitive([QQ(2), QQ(2)], QQ) == (QQ(2), [QQ(1), QQ(1)])
    # 断言包含 QQ(1), QQ(2), QQ(1) 元素的列表经过 dup_primitive 处理后，返回 QQ 类型的 1 和包含 QQ(1), QQ(2), QQ(1) 元素的列表
    assert dup_primitive([QQ(1), QQ(2), QQ(1)], QQ) == (QQ(1), [QQ(1), QQ(2), QQ(1)])
    # 断言包含 QQ(2), QQ(4), QQ(2) 元素的列表经过 dup_primitive 处理后，返回 QQ 类型的 2 和包含 QQ(1), QQ(2), QQ(1) 元素的列表
    assert dup_primitive([QQ(2), QQ(4), QQ(2)], QQ) == (QQ(2), [QQ(1), QQ(2), QQ(1)])

    # 断言包含 QQ(2, 3), QQ(4, 9) 元素的列表经过 dup_primitive 处理后，返回 QQ 类型的 2, 9 和包含 QQ(3), QQ(2) 元素的列表
    assert dup_primitive([QQ(2, 3), QQ(4, 9)], QQ) == (QQ(2, 9), [QQ(3), QQ(2)])
    # 断言包含 QQ(2, 3), QQ(4, 5) 元素的列表经过 dup_primitive 处理后，返回 QQ 类型的 2, 15 和包含 QQ(5), QQ(6) 元素的列表
    assert dup_primitive([QQ(2, 3), QQ(4, 5)], QQ) == (QQ(2, 15), [QQ(5), QQ(6)])


# 定义测试函数 test_dmp_ground_primitive
def test_dmp_ground_primitive():
    # 断言将包含 ZZ(1) 元素的列表当作一次项，使用 0 阶处理后，返回 ZZ 类型的 1 和包含 ZZ(1) 元素的列表
    assert dmp_ground_primitive([ZZ(1)], 0, ZZ) == (ZZ(1), [ZZ(1)])

    # 断言将包含空列表的列表当作二次项，使用 1 阶处理后，返回 ZZ 类型的 0 和包含空列表的列表
    assert dmp_ground_primitive([[]], 1, ZZ) == (ZZ(0), [[]])

    # 断言使用 f_0 列表作为多项式，使用 2 阶处理后，返回 ZZ 类型的 1 和 f_0 原始列表
    assert dmp_ground_primitive(f_0, 2, ZZ) == (ZZ(1), f_0)
    # 断言将 f_0 列表作为多项式，乘以 ZZ(2) 作为一次项，使用 2 阶处理后，返回 ZZ 类型的 2 和 f_0 原始列表
    assert dmp_ground_primitive(dmp_mul_ground(f_0, ZZ(2), 2, ZZ), 2, ZZ) == (ZZ(2), f_0)

    # 断言使用 f_1 列表作为多项式，使用 2 阶处理后，返回 ZZ 类型的 1 和 f_1 原始列表
    assert dmp_ground_primitive(f_1, 2, ZZ) == (ZZ(1), f_1)
    # 断言将 f_1 列表作为多项式，乘以 ZZ(3) 作为一次项，使用 2 阶处理后，返回 ZZ 类型的 3 和 f_1 原始列表
    assert dmp_ground_primitive(dmp_mul_ground(f_1, ZZ(3), 2, ZZ), 2, ZZ) == (ZZ(3), f_1)

    # 断言使用 f_2 列表作为多项式，使用 2 阶处理后，返回 ZZ 类型的 1 和 f_2 原始列表
    assert dmp_ground_primitive(f_2, 2, ZZ) == (ZZ(1), f_2)
    # 断言将 f_2 列表作为多项式，乘以 ZZ(4) 作为一次项，使用 2 阶处理后，返回 ZZ 类型的 4 和 f_2 原始列表
    assert dmp_ground_primitive(dmp_mul_ground(f_2, ZZ(4), 2, ZZ), 2, ZZ) == (ZZ(4), f_2)

    # 断言使用 f_3 列表作为多项式，使用 2 阶处理后，返回 ZZ 类型的 1 和 f_3 原始列表
    assert dmp_ground_primitive(f_3, 2, ZZ) == (ZZ(1), f_3)
    # 断言将 f_3 列表作为多项式，乘以 ZZ(5) 作为一次项，使用 2 阶处理后，返回 ZZ 类型的 5 和 f_3 原始列表
    assert dmp_ground_primitive(dmp_mul_ground(f_3, ZZ(5), 2, ZZ), 2, ZZ) == (ZZ(5), f_3)

    # 断言使用 f_4 列表作为多项式，使用 2 阶处理后，返回 ZZ 类型的 1 和 f_4 原始列表
    assert dmp_ground_primitive(f_4, 2, ZZ) == (ZZ(1), f_4)
    # 断
    # 断言检查函数 dup_extract 返回的结果是否等于 (45796, F, G)
    assert dup_extract(f, g, ZZ) == (45796, F, G)
# 定义函数 test_dmp_ground_extract，用于测试 dmp_ground_extract 函数
def test_dmp_ground_extract():
    # 定义变量 f，调用 dmp_normal 函数生成的结果
    f = dmp_normal([[2930944], [], [2198208], [], [549552], [], [45796]], 1, ZZ)
    # 定义变量 g，调用 dmp_normal 函数生成的结果
    g = dmp_normal([[17585664], [], [8792832], [], [1099104], []], 1, ZZ)

    # 定义变量 F，调用 dmp_normal 函数生成的结果
    F = dmp_normal([[64], [], [48], [], [12], [], [1]], 1, ZZ)
    # 定义变量 G，调用 dmp_normal 函数生成的结果
    G = dmp_normal([[384], [], [192], [], [24], []], 1, ZZ)

    # 断言语句，验证 dmp_ground_extract 函数的返回值
    assert dmp_ground_extract(f, g, 1, ZZ) == (45796, F, G)


# 定义函数 test_dup_real_imag，用于测试 dup_real_imag 函数
def test_dup_real_imag():
    # 断言语句，验证 dup_real_imag 函数的返回值
    assert dup_real_imag([], ZZ) == ([[]], [[]])
    assert dup_real_imag([1], ZZ) == ([[1]], [[]])

    assert dup_real_imag([1, 1], ZZ) == ([[1], [1]], [[1, 0]])
    assert dup_real_imag([1, 2], ZZ) == ([[1], [2]], [[1, 0]])

    assert dup_real_imag([1, 2, 3], ZZ) == ([[1], [2], [-1, 0, 3]], [[2, 0], [2, 0]])

    assert dup_real_imag([ZZ(1), ZZ(0), ZZ(1), ZZ(3)], ZZ) == (
        [[ZZ(1)], [], [ZZ(-3), ZZ(0), ZZ(1)], [ZZ(3)]],
        [[ZZ(3), ZZ(0)], [], [ZZ(-1), ZZ(0), ZZ(1), ZZ(0)]]
    )

    # 引发异常，测试 dup_real_imag 函数对非整数环的处理
    raises(DomainError, lambda: dup_real_imag([EX(1), EX(2)], EX))


# 定义函数 test_dup_mirror，用于测试 dup_mirror 函数
def test_dup_mirror():
    # 断言语句，验证 dup_mirror 函数的返回值
    assert dup_mirror([], ZZ) == []
    assert dup_mirror([1], ZZ) == [1]

    assert dup_mirror([1, 2, 3, 4, 5], ZZ) == [1, -2, 3, -4, 5]
    assert dup_mirror([1, 2, 3, 4, 5, 6], ZZ) == [-1, 2, -3, 4, -5, 6]


# 定义函数 test_dup_scale，用于测试 dup_scale 函数
def test_dup_scale():
    # 断言语句，验证 dup_scale 函数的返回值
    assert dup_scale([], -1, ZZ) == []
    assert dup_scale([1], -1, ZZ) == [1]

    assert dup_scale([1, 2, 3, 4, 5], -1, ZZ) == [1, -2, 3, -4, 5]
    assert dup_scale([1, 2, 3, 4, 5], -7, ZZ) == [2401, -686, 147, -28, 5]


# 定义函数 test_dup_shift，用于测试 dup_shift 函数
def test_dup_shift():
    # 断言语句，验证 dup_shift 函数的返回值
    assert dup_shift([], 1, ZZ) == []
    assert dup_shift([1], 1, ZZ) == [1]

    assert dup_shift([1, 2, 3, 4, 5], 1, ZZ) == [1, 6, 15, 20, 15]
    assert dup_shift([1, 2, 3, 4, 5], 7, ZZ) == [1, 30, 339, 1712, 3267]


# 定义函数 test_dmp_shift，用于测试 dmp_shift 函数
def test_dmp_shift():
    # 断言语句，验证 dmp_shift 函数的返回值
    assert dmp_shift([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == [ZZ(1), ZZ(3)]

    assert dmp_shift([[]], [ZZ(1), ZZ(2)], 1, ZZ) == [[]]

    xy = [[ZZ(1), ZZ(0)], []]               # x*y
    x1y2 = [[ZZ(1), ZZ(2)], [ZZ(1), ZZ(2)]] # (x+1)*(y+2)
    assert dmp_shift(xy, [ZZ(1), ZZ(2)], 1, ZZ) == x1y2


# 定义函数 test_dup_transform，用于测试 dup_transform 函数
def test_dup_transform():
    # 断言语句，验证 dup_transform 函数的返回值
    assert dup_transform([], [], [1, 1], ZZ) == []
    assert dup_transform([], [1], [1, 1], ZZ) == []
    assert dup_transform([], [1, 2], [1, 1], ZZ) == []

    assert dup_transform([6, -5, 4, -3, 17], [1, -3, 4], [2, -3], ZZ) == \
        [6, -82, 541, -2205, 6277, -12723, 17191, -13603, 4773]


# 定义函数 test_dup_compose，用于测试 dup_compose 函数
def test_dup_compose():
    # 断言语句，验证 dup_compose 函数的返回值
    assert dup_compose([], [], ZZ) == []
    assert dup_compose([], [1], ZZ) == []
    assert dup_compose([], [1, 2], ZZ) == []

    assert dup_compose([1], [], ZZ) == [1]

    assert dup_compose([1, 2, 0], [], ZZ) == []
    assert dup_compose([1, 2, 1], [], ZZ) == [1]

    assert dup_compose([1, 2, 1], [1], ZZ) == [4]
    assert dup_compose([1, 2, 1], [7], ZZ) == [64]

    assert dup_compose([1, 2, 1], [1, -1], ZZ) == [1, 0, 0]
    assert dup_compose([1, 2, 1], [1, 1], ZZ) == [1, 4, 4]
    assert dup_compose([1, 2, 1], [1, 2, 1], ZZ) == [1, 4, 8, 8, 4]


# 定义函数 test_dmp_compose，用于测试 dmp_compose 函数
def test_dmp_compose():
    # 这里留空，因为函数体未完整，无法添加注释
    # 调用 dmp_compose 函数，检查返回的结果是否与预期的结果相等
    assert dmp_compose([1, 2, 1], [1, 2, 1], 0, ZZ) == [1, 4, 8, 8, 4]
    
    # 再次调用 dmp_compose 函数，验证对空列表的处理是否正确
    assert dmp_compose([[[]]], [[[]]], 2, ZZ) == [[[]]]
    # 继续验证 dmp_compose 对包含一个非空子列表的列表的处理是否正确
    assert dmp_compose([[[]]], [[[1]]], 2, ZZ) == [[[]]]
    # 验证 dmp_compose 对包含多个非空子列表的列表的处理是否正确
    assert dmp_compose([[[]]], [[[1]], [[2]]], 2, ZZ) == [[[]]]
    
    # 验证 dmp_compose 对空列表的处理是否正确
    assert dmp_compose([[[1]]], [], 2, ZZ) == [[[1]]]
    
    # 验证 dmp_compose 对包含空子列表和非空子列表的列表的处理是否正确
    assert dmp_compose([[1], [2], [ ]], [[]], 1, ZZ) == [[]]
    # 继续验证 dmp_compose 对包含多个非空子列表的列表的处理是否正确
    assert dmp_compose([[1], [2], [1]], [[]], 1, ZZ) == [[1]]
    
    # 验证 dmp_compose 对两个列表中包含相同整数的处理是否正确
    assert dmp_compose([[1], [2], [1]], [[1]], 1, ZZ) == [[4]]
    # 继续验证 dmp_compose 对两个列表中包含不同整数的处理是否正确
    assert dmp_compose([[1], [2], [1]], [[7]], 1, ZZ) == [[64]]
    
    # 验证 dmp_compose 对两个列表中包含不同整数的处理是否正确
    assert dmp_compose([[1], [2], [1]], [[1], [-1]], 1, ZZ) == [[1], [ ], [ ]]
    # 继续验证 dmp_compose 对两个列表中包含相同整数的处理是否正确
    assert dmp_compose([[1], [2], [1]], [[1], [ 1]], 1, ZZ) == [[1], [4], [4]]
    
    # 最后验证 dmp_compose 对两个较大的列表的处理是否正确
    assert dmp_compose(
        [[1], [2], [1]], [[1], [2], [1]], 1, ZZ) == [[1], [4], [8], [8], [4]]
def test_dup_decompose():
    # 测试单项式 [1] 被分解成 [[1]]
    assert dup_decompose([1], ZZ) == [[1]]

    # 测试单项式 [1, 0] 被分解成 [[1, 0]]
    assert dup_decompose([1, 0], ZZ) == [[1, 0]]
    # 测试单项式 [1, 0, 0, 0] 被分解成 [[1, 0, 0, 0]]
    assert dup_decompose([1, 0, 0, 0], ZZ) == [[1, 0, 0, 0]]

    # 测试单项式 [1, 0, 0, 0, 0] 被分解成 [[1, 0, 0], [1, 0, 0]]
    assert dup_decompose([1, 0, 0, 0, 0], ZZ) == [[1, 0, 0], [1, 0, 0]]
    # 测试单项式 [1, 0, 0, 0, 0, 0, 0] 被分解成 [[1, 0, 0, 0], [1, 0, 0]]
    assert dup_decompose([1, 0, 0, 0, 0, 0, 0], ZZ) == [[1, 0, 0, 0], [1, 0, 0]]

    # 测试单项式 [7, 0, 0, 0, 1] 被分解成 [[7, 0, 1], [1, 0, 0]]
    assert dup_decompose([7, 0, 0, 0, 1], ZZ) == [[7, 0, 1], [1, 0, 0]]
    # 测试单项式 [4, 0, 3, 0, 2] 被分解成 [[4, 3, 2], [1, 0, 0]]
    assert dup_decompose([4, 0, 3, 0, 2], ZZ) == [[4, 3, 2], [1, 0, 0]]

    # 测试多项式 f 被分解成 [[1, 0, 0, -2, 9], [1, 0, 5, 0]]
    f = [1, 0, 20, 0, 150, 0, 500, 0, 625, -2, 0, -10, 9]
    assert dup_decompose(f, ZZ) == [[1, 0, 0, -2, 9], [1, 0, 5, 0]]

    # 测试多项式 f 被分解成 [[2, 0, 0, -4, 18], [1, 0, 5, 0]]
    f = [2, 0, 40, 0, 300, 0, 1000, 0, 1250, -4, 0, -20, 18]
    assert dup_decompose(f, ZZ) == [[2, 0, 0, -4, 18], [1, 0, 5, 0]]

    # 测试多项式 f 被分解成 [[1, -8, 24, -34, 29], [1, 0, 5, 0]]
    f = [1, 0, 20, -8, 150, -120, 524, -600, 865, -1034, 600, -170, 29]
    assert dup_decompose(f, ZZ) == [[1, -8, 24, -34, 29], [1, 0, 5, 0]]

    # 测试多项式 f 在多项式环 R 中的分解结果为原多项式 f
    R, t = ring("t", ZZ)
    f = [6*t**2 - 42,
         48*t**2 + 96,
         144*t**2 + 648*t + 288,
         624*t**2 + 864*t + 384,
         108*t**3 + 312*t**2 + 432*t + 192]
    assert dup_decompose(f, R.to_domain()) == [f]


def test_dmp_lift():
    # 测试在有理数域上的多项式提升
    q = [QQ(1, 1), QQ(0, 1), QQ(1, 1)]

    f_a = [ANP([QQ(1, 1)], q, QQ), ANP([], q, QQ), ANP([], q, QQ),
         ANP([QQ(1, 1), QQ(0, 1)], q, QQ), ANP([QQ(17, 1), QQ(0, 1)], q, QQ)]

    f_lift = [QQ(1), QQ(0), QQ(0), QQ(0), QQ(0), QQ(0), QQ(2), QQ(0), QQ(578),
              QQ(0), QQ(0), QQ(0), QQ(1), QQ(0), QQ(-578), QQ(0), QQ(83521)]

    assert dmp_lift(f_a, 0, QQ.algebraic_field(I)) == f_lift

    # 测试在有理数域上的多项式提升（带虚数单位）
    f_g = [QQ_I(1), QQ_I(0), QQ_I(0), QQ_I(0, 1), QQ_I(0, 17)]
    assert dmp_lift(f_g, 0, QQ_I) == f_lift

    # 测试对非多项式环的提升，预期抛出 DomainError 异常
    raises(DomainError, lambda: dmp_lift([EX(1), EX(2)], 0, EX))


def test_dup_sign_variations():
    # 测试空列表的符号变化数为 0
    assert dup_sign_variations([], ZZ) == 0
    # 测试单项式 [1, 0] 的符号变化数为 0
    assert dup_sign_variations([1, 0], ZZ) == 0
    # 测试单项式 [1, 0, 2] 的符号变化数为 0
    assert dup_sign_variations([1, 0, 2], ZZ) == 0
    # 测试单项式 [1, 0, 3, 0] 的符号变化数为 0
    assert dup_sign_variations([1, 0, 3, 0], ZZ) == 0
    # 测试单项式 [1, 0, 4, 0, 5] 的符号变化数为 0
    assert dup_sign_variations([1, 0, 4, 0, 5], ZZ) == 0

    # 测试单项式 [-1, 0, 2] 的符号变化数为 1
    assert dup_sign_variations([-1, 0, 2], ZZ) == 1
    # 测试单项式 [-1, 0, 3, 0] 的符号变化数为 1
    assert dup_sign_variations([-1, 0, 3, 0], ZZ) == 1
    # 测试单项式 [-1, 0, 4, 0, 5] 的符号变化数为 1
    assert dup_sign_variations([-1, 0, 4, 0, 5], ZZ) == 1

    # 测试单项式 [-1, -4, -5] 的符号变化数为 0
    assert dup_sign_variations([-1, -4, -5], ZZ) == 0
    # 测试单项式 [1, -4, -5] 的符号变化数为 1
    assert dup_sign_variations([1, -4, -5], ZZ) == 1
    # 测试单项式 [1, 4, -5] 的符号变化数为 1
    assert dup_sign_variations([1, 4, -5], ZZ) == 1
    # 测试单项式 [1, -4, 5] 的符号变化数为 2
    assert dup_sign_variations([1, -4, 5], ZZ) == 2
    # 测试单项式 [-1, 4, -5] 的符号变化数为 2
    assert dup_sign_variations([-1, 4, -5], ZZ) == 2
    # 测试单项式 [-1, 4, 5] 的符号变化数为 1
    assert dup_sign_variations([-1, 4, 5], ZZ) == 1
    # 测试单项式 [-1,
    # 断言函数调用，验证给定列表中具有重复的正负变化的次数
    assert dup_sign_variations([-1, 0, -4, 0, 5], ZZ) == 1
    # 断言函数调用，验证给定列表中具有重复的正负变化的次数
    assert dup_sign_variations([ 1, 0, 4, 0, 5], ZZ) == 0
def test_dup_clear_denoms():
    # 确保当输入空列表时，返回的结果是整数1和空列表
    assert dup_clear_denoms([], QQ, ZZ) == (ZZ(1), [])

    # 确保当输入包含一个有理数1时，返回结果是整数1和包含有理数1的列表
    assert dup_clear_denoms([QQ(1)], QQ, ZZ) == (ZZ(1), [QQ(1)])

    # 确保当输入包含一个有理数7时，返回结果是整数1和包含有理数7的列表
    assert dup_clear_denoms([QQ(7)], QQ, ZZ) == (ZZ(1), [QQ(7)])

    # 确保当输入包含一个有理数7/3时，对于有理数域 QQ，返回结果是整数3和包含有理数7的列表
    assert dup_clear_denoms([QQ(7, 3)], QQ) == (ZZ(3), [QQ(7)])

    # 确保当输入包含一个有理数7/3时，对于整数环 ZZ，返回结果是整数3和包含有理数7的列表
    assert dup_clear_denoms([QQ(7, 3)], QQ, ZZ) == (ZZ(3), [QQ(7)])

    # 确保当输入包含有理数3, 1, 0时，返回结果是整数1和包含这些有理数的列表
    assert dup_clear_denoms([QQ(3), QQ(1), QQ(0)], QQ, ZZ) == (ZZ(1), [QQ(3), QQ(1), QQ(0)])

    # 确保当输入包含有理数1, 1/2, 0时，返回结果是整数2和包含有理数2, 1, 0的列表
    assert dup_clear_denoms([QQ(1), QQ(1, 2), QQ(0)], QQ, ZZ) == (ZZ(2), [QQ(2), QQ(1), QQ(0)])

    # 确保当输入包含有理数3, 1, 0时，并进行转换为整数的操作，返回结果是整数1和包含整数3, 1, 0的列表
    assert dup_clear_denoms([QQ(3), QQ(1), QQ(0)], QQ, ZZ, convert=True) == (ZZ(1), [ZZ(3), ZZ(1), ZZ(0)])

    # 确保当输入包含有理数1, 1/2, 0时，并进行转换为整数的操作，返回结果是整数2和包含整数2, 1, 0的列表
    assert dup_clear_denoms([QQ(1), QQ(1, 2), QQ(0)], QQ, ZZ, convert=True) == (ZZ(2), [ZZ(2), ZZ(1), ZZ(0)])

    # 确保当输入包含表达式 S(3)/2 和 S(9)/4 时，返回结果是表达式 4 和包含表达式 6, 9 的列表
    assert dup_clear_denoms([EX(S(3)/2), EX(S(9)/4)], EX) == (EX(4), [EX(6), EX(9)])

    # 确保当输入包含一个表达式 7 时，返回结果是表达式 1 和包含表达式 7 的列表
    assert dup_clear_denoms([EX(7)], EX) == (EX(1), [EX(7)])

    # 确保当输入包含表达式 sin(x)/x 和 0 时，返回结果是表达式 x 和包含表达式 sin(x), 0 的列表
    assert dup_clear_denoms([EX(sin(x)/x), EX(0)], EX) == (EX(x), [EX(sin(x)), EX(0)])

def test_dmp_clear_denoms():
    # 确保当输入包含空列表时，返回结果是整数1和包含空列表的列表
    assert dmp_clear_denoms([[]], 1, QQ, ZZ) == (ZZ(1), [[]])

    # 确保当输入包含包含有理数1的列表时，返回结果是整数1和包含包含有理数1的列表的列表
    assert dmp_clear_denoms([[QQ(1)]], 1, QQ, ZZ) == (ZZ(1), [[QQ(1)]])

    # 确保当输入包含包含有理数7的列表时，返回结果是整数1和包含包含有理数7的列表的列表
    assert dmp_clear_denoms([[QQ(7)]], 1, QQ, ZZ) == (ZZ(1), [[QQ(7)]])

    # 确保当输入包含包含有理数7/3的列表时，对于有理数域 QQ，返回结果是整数3和包含包含有理数7的列表的列表
    assert dmp_clear_denoms([[QQ(7, 3)]], 1, QQ) == (ZZ(3), [[QQ(7)]])

    # 确保当输入包含包含有理数7/3的列表时，对于整数环 ZZ，返回结果是整数3和包含包含有理数7的列表的列表
    assert dmp_clear_denoms([[QQ(7, 3)]], 1, QQ, ZZ) == (ZZ(3), [[QQ(7)]])

    # 确保当输入包含有理数3, 1, 0的列表时，返回结果是整数1和包含这些列表的列表
    assert dmp_clear_denoms([[QQ(3)], [QQ(1)], []], 1, QQ, ZZ) == (ZZ(1), [[QQ(3)], [QQ(1)], []])

    # 确保当输入包含有理数1, 1/2, 0的列表时，返回结果是整数2和包含这些列表的列表
    assert dmp_clear_denoms([[QQ(1)], [QQ(1, 2)], []], 1, QQ, ZZ) == (ZZ(2), [[QQ(2)], [QQ(1)], []])

    # 确保当输入包含有理数3, 1, 0的列表时，并进行转换为整数的操作，返回结果是整数1和包含这些整数的列表
    assert dmp_clear_denoms([QQ(3), QQ(1), QQ(0)], 0, QQ, ZZ, convert=True) == (ZZ(1), [ZZ(3), ZZ(1), ZZ(0)])

    # 确保当输入包含有理数1, 1/2, 0的列表时，并进行转换为整数的操作，返回结果是整数2和包含这些整数的列表
    assert dmp_clear_denoms([QQ(1), QQ(1, 2), QQ(0)], 0, QQ, ZZ, convert=True) == (ZZ(2), [ZZ(2), ZZ(1), ZZ(0)])

    # 确保当输入包含包含表达式 S(3)/2 和 S(9)/4的列表时，返回结果是表达式 4 和包含表达式 6, 9 的列表的列表
    assert dmp_clear_denoms([[EX(S(3)/2)], [EX(S(9)/4)]], 1, EX) == (EX(4), [[EX(6)], [EX(9)]])

    # 确保当输入包含包含表达式 7的列表时，返回结果是表达式 1 和包含表达式 7 的列表的列表
    assert dmp_clear_denoms([[EX(7)]], 1, EX) == (EX(1), [[EX(7)]])

    # 确保当输入包含包含表达式 sin(x)/x 和 0的列表时，返回结果是表达式 x 和包含表达式 sin(x), 0 的列表的列表
    assert dmp_clear_denoms([[EX(sin(x)/x), EX(0)]], 1, EX) == (EX(x), [[EX(sin(x)), EX(0)]])
```