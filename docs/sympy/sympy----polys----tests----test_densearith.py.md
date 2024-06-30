# `D:\src\scipysrc\sympy\sympy\polys\tests\test_densearith.py`

```
# 导入所需的模块和函数

"""Tests for dense recursive polynomials' arithmetics. """
# 导入对稠密递归多项式算术的测试

from sympy.external.gmpy import GROUND_TYPES

from sympy.polys.densebasic import (
    dup_normal, dmp_normal,
)
# 导入稠密多项式的基本操作函数：dup_normal用于一元多项式，dmp_normal用于多变量多项式

from sympy.polys.densearith import (
    dup_add_term, dmp_add_term,
    dup_sub_term, dmp_sub_term,
    dup_mul_term, dmp_mul_term,
    dup_add_ground, dmp_add_ground,
    dup_sub_ground, dmp_sub_ground,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground,
    dup_exquo_ground, dmp_exquo_ground,
    dup_lshift, dup_rshift,
    dup_abs, dmp_abs,
    dup_neg, dmp_neg,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_sqr, dmp_sqr,
    dup_pow, dmp_pow,
    dup_add_mul, dmp_add_mul,
    dup_sub_mul, dmp_sub_mul,
    dup_pdiv, dup_prem, dup_pquo, dup_pexquo,
    dmp_pdiv, dmp_prem, dmp_pquo, dmp_pexquo,
    dup_rr_div, dmp_rr_div,
    dup_ff_div, dmp_ff_div,
    dup_div, dup_rem, dup_quo, dup_exquo,
    dmp_div, dmp_rem, dmp_quo, dmp_exquo,
    dup_max_norm, dmp_max_norm,
    dup_l1_norm, dmp_l1_norm,
    dup_l2_norm_squared, dmp_l2_norm_squared,
    dup_expand, dmp_expand,
)
# 导入稠密多项式的算术运算函数，包括加减乘除、范数计算、特殊运算等

from sympy.polys.polyerrors import (
    ExactQuotientFailed,
)
# 导入多项式运算可能引发的错误类型

from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
# 导入特殊多项式（f_polys）、有限域（FF）、整数环（ZZ）、有理数域（QQ）

from sympy.testing.pytest import raises
# 导入测试框架中的 raises 函数，用于测试异常情况

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]
# 将 f_polys 返回的多项式转换为稠密表示，分别赋值给 f_0 到 f_6

F_0 = dmp_mul_ground(dmp_normal(f_0, 2, QQ), QQ(1, 7), 2, QQ)
# 对 f_0 进行多变量多项式正常化（dmp_normal），然后乘以有理数 QQ(1, 7)，最后乘以变量数量为 2 的因子

def test_dup_add_term():
    # 测试 dup_add_term 函数

    f = dup_normal([], ZZ)
    # 创建一个空的整数环多项式 f

    assert dup_add_term(f, ZZ(0), 0, ZZ) == dup_normal([], ZZ)
    # 断言向空多项式添加常数项 0 得到空多项式

    assert dup_add_term(f, ZZ(1), 0, ZZ) == dup_normal([1], ZZ)
    # 断言向空多项式添加常数项 1 得到 [1]

    assert dup_add_term(f, ZZ(1), 1, ZZ) == dup_normal([1, 0], ZZ)
    # 断言向空多项式的 x^1 项添加常数项 1 得到 [1, 0]

    assert dup_add_term(f, ZZ(1), 2, ZZ) == dup_normal([1, 0, 0], ZZ)
    # 断言向空多项式的 x^2 项添加常数项 1 得到 [1, 0, 0]

    f = dup_normal([1, 1, 1], ZZ)
    # 创建一个有系数 [1, 1, 1] 的整数环多项式 f

    assert dup_add_term(f, ZZ(1), 0, ZZ) == dup_normal([1, 1, 2], ZZ)
    # 断言向多项式 f 添加常数项 1 得到 [1, 1, 2]

    assert dup_add_term(f, ZZ(1), 1, ZZ) == dup_normal([1, 2, 1], ZZ)
    # 断言向多项式 f 的 x^1 项添加常数项 1 得到 [1, 2, 1]

    assert dup_add_term(f, ZZ(1), 2, ZZ) == dup_normal([2, 1, 1], ZZ)
    # 断言向多项式 f 的 x^2 项添加常数项 1 得到 [2, 1, 1]

    assert dup_add_term(f, ZZ(1), 3, ZZ) == dup_normal([1, 1, 1, 1], ZZ)
    # 断言向多项式 f 的 x^3 项添加常数项 1 得到 [1, 1, 1, 1]

    assert dup_add_term(f, ZZ(1), 4, ZZ) == dup_normal([1, 0, 1, 1, 1], ZZ)
    # 断言向多项式 f 的 x^4 项添加常数项 1 得到 [1, 0, 1, 1, 1]

    assert dup_add_term(f, ZZ(1), 5, ZZ) == dup_normal([1, 0, 0, 1, 1, 1], ZZ)
    # 断言向多项式 f 的 x^5 项添加常数项 1 得到 [1, 0, 0, 1, 1, 1]

    assert dup_add_term(
        f, ZZ(1), 6, ZZ) == dup_normal([1, 0, 0, 0, 1, 1, 1], ZZ)
    # 断言向多项式 f 的 x^6 项添加常数项 1 得到 [1, 0, 0, 0, 1, 1, 1]

    assert dup_add_term(f, ZZ(-1), 2, ZZ) == dup_normal([1, 1], ZZ)
    # 断言向多项式 f 的 x^2 项添加常数项 -1 得到 [1, 1]


def test_dmp_add_term():
    # 测试 dmp_add_term 函数

    assert dmp_add_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, 0, ZZ) == \
        dup_add_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, ZZ)
    # 断言向一元多项式添加常数项得到的结果与等效的多变量多项式添加常数项结果一致

    assert dmp_add_term(f_0, [[]], 3, 2, ZZ) == f_0
    # 断言向多变量多项式 f_0 添加常数项 [][x^2] 得到 f_0 本身

    assert dmp_add_term(F_0, [[]], 3, 2, QQ) == F_0
    # 断言向多变量多项式 F_0 添加常数项 [][x^2] 得到 F_0 本身


def test_dup_sub_term():
    # 测试 dup_sub_term 函数

    f = dup_normal([], ZZ)
    # 创建一个空的整数环多项式 f

    assert dup_sub_term(f, ZZ(0), 0, ZZ) == dup_normal([], ZZ)
    # 断言从空多项式减去常数项 0 得到空多项式

    assert dup_sub_term(f, ZZ(1), 0, ZZ) == dup_normal([-1], ZZ)
    # 断言从空多项式减去常数项 1 得到 [-1]

    assert dup_sub_term(f, ZZ(1), 1, ZZ) == dup_normal([-1, 0], ZZ)
    # 断言从空多项式的 x^1 项减去常数项 1 得到 [-1,
    # 调用 dup_normal 函数生成多项式 f，初始为 [1, 1, 1]
    f = dup_normal([1, 1, 1], ZZ)
    
    # 断言，验证 dup_sub_term 函数在给定的多项式 f 上进行子项替换的结果
    # 使用 dup_sub_term 替换 f 中指定位置的系数，得到新的多项式，期望结果是 [ 1, 1, -1]
    assert dup_sub_term(f, ZZ(2), 0, ZZ) == dup_normal([ 1, 1, -1], ZZ)
    
    # 类似地，替换第二个位置的系数，期望结果是 [ 1, -1, 1]
    assert dup_sub_term(f, ZZ(2), 1, ZZ) == dup_normal([ 1, -1, 1], ZZ)
    
    # 替换第三个位置的系数，期望结果是 [-1, 1, 1]
    assert dup_sub_term(f, ZZ(2), 2, ZZ) == dup_normal([-1, 1, 1], ZZ)
    
    # 继续对 f 进行子项替换，这次替换位置为 3，期望结果是 [-1, 1, 1, 1]
    assert dup_sub_term(f, ZZ(1), 3, ZZ) == dup_normal([-1, 1, 1, 1], ZZ)
    
    # 替换位置为 4，期望结果是 [-1, 0, 1, 1, 1]
    assert dup_sub_term(f, ZZ(1), 4, ZZ) == dup_normal([-1, 0, 1, 1, 1], ZZ)
    
    # 替换位置为 5，期望结果是 [-1, 0, 0, 1, 1, 1]
    assert dup_sub_term(f, ZZ(1), 5, ZZ) == dup_normal([-1, 0, 0, 1, 1, 1], ZZ)
    
    # 替换位置为 6，期望结果是 [-1, 0, 0, 0, 1, 1, 1]
    assert dup_sub_term(f, ZZ(1), 6, ZZ) == dup_normal([-1, 0, 0, 0, 1, 1, 1], ZZ)
    
    # 最后一次替换位置为 2，期望结果是 [1, 1]
    assert dup_sub_term(f, ZZ(1), 2, ZZ) == dup_normal([1, 1], ZZ)
# 定义用于测试 dmp_sub_term 函数的函数
def test_dmp_sub_term():
    # 断言调用 dmp_sub_term 函数后的返回值符合预期
    assert dmp_sub_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, 0, ZZ) == \
        dup_sub_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, ZZ)
    # 断言调用 dmp_sub_term 函数后的返回值符合预期
    assert dmp_sub_term(f_0, [[]], 3, 2, ZZ) == f_0
    # 断言调用 dmp_sub_term 函数后的返回值符合预期
    assert dmp_sub_term(F_0, [[]], 3, 2, QQ) == F_0


# 定义用于测试 dup_mul_term 函数的函数
def test_dup_mul_term():
    # 创建一个空的多项式 f
    f = dup_normal([], ZZ)

    # 断言调用 dup_mul_term 函数后的返回值符合预期
    assert dup_mul_term(f, ZZ(2), 3, ZZ) == dup_normal([], ZZ)

    # 将 f 初始化为 [1, 1]
    f = dup_normal([1, 1], ZZ)

    # 断言调用 dup_mul_term 函数后的返回值符合预期
    assert dup_mul_term(f, ZZ(0), 3, ZZ) == dup_normal([], ZZ)

    # 将 f 初始化为 [1, 2, 3]
    f = dup_normal([1, 2, 3], ZZ)

    # 断言调用 dup_mul_term 函数后的返回值符合预期
    assert dup_mul_term(f, ZZ(2), 0, ZZ) == dup_normal([2, 4, 6], ZZ)
    # 断言调用 dup_mul_term 函数后的返回值符合预期
    assert dup_mul_term(f, ZZ(2), 1, ZZ) == dup_normal([2, 4, 6, 0], ZZ)
    # 断言调用 dup_mul_term 函数后的返回值符合预期
    assert dup_mul_term(f, ZZ(2), 2, ZZ) == dup_normal([2, 4, 6, 0, 0], ZZ)
    # 断言调用 dup_mul_term 函数后的返回值符合预期
    assert dup_mul_term(f, ZZ(2), 3, ZZ) == dup_normal([2, 4, 6, 0, 0, 0], ZZ)


# 定义用于测试 dmp_mul_term 函数的函数
def test_dmp_mul_term():
    # 断言调用 dmp_mul_term 函数后的返回值符合预期
    assert dmp_mul_term([ZZ(1), ZZ(2), ZZ(3)], ZZ(2), 1, 0, ZZ) == \
        dup_mul_term([ZZ(1), ZZ(2), ZZ(3)], ZZ(2), 1, ZZ)

    # 断言调用 dmp_mul_term 函数后的返回值符合预期
    assert dmp_mul_term([[]], [ZZ(2)], 3, 1, ZZ) == [[]]
    # 断言调用 dmp_mul_term 函数后的返回值符合预期
    assert dmp_mul_term([[ZZ(1)]], [], 3, 1, ZZ) == [[]]

    # 断言调用 dmp_mul_term 函数后的返回值符合预期
    assert dmp_mul_term([[ZZ(1), ZZ(2)], [ZZ(3)]], [ZZ(2)], 2, 1, ZZ) == \
        [[ZZ(2), ZZ(4)], [ZZ(6)], [], []]

    # 断言调用 dmp_mul_term 函数后的返回值符合预期
    assert dmp_mul_term([[]], [QQ(2, 3)], 3, 1, QQ) == [[]]
    # 断言调用 dmp_mul_term 函数后的返回值符合预期
    assert dmp_mul_term([[QQ(1, 2)]], [], 3, 1, QQ) == [[]]

    # 断言调用 dmp_mul_term 函数后的返回值符合预期
    assert dmp_mul_term([[QQ(1, 5), QQ(2, 5)], [QQ(3, 5)]], [QQ(2, 3)], 2, 1, QQ) == \
        [[QQ(2, 15), QQ(4, 15)], [QQ(6, 15)], [], []]


# 定义用于测试 dup_add_ground 函数的函数
def test_dup_add_ground():
    # 将 f 初始化为 [1, 2, 3, 4]
    f = ZZ.map([1, 2, 3, 4])
    # 将 g 初始化为 [1, 2, 3, 8]
    g = ZZ.map([1, 2, 3, 8])

    # 断言调用 dup_add_ground 函数后的返回值符合预期
    assert dup_add_ground(f, ZZ(4), ZZ) == g


# 定义用于测试 dmp_add_ground 函数的函数
def test_dmp_add_ground():
    # 将 f 初始化为 [[1], [2], [3], [4]]
    f = ZZ.map([[1], [2], [3], [4]])
    # 将 g 初始化为 [[1], [2], [3], [8]]
    g = ZZ.map([[1], [2], [3], [8]])

    # 断言调用 dmp_add_ground 函数后的返回值符合预期
    assert dmp_add_ground(f, ZZ(4), 1, ZZ) == g


# 定义用于测试 dup_sub_ground 函数的函数
def test_dup_sub_ground():
    # 将 f 初始化为 [1, 2, 3, 4]
    f = ZZ.map([1, 2, 3, 4])
    # 将 g 初始化为 [1, 2, 3, 0]
    g = ZZ.map([1, 2, 3, 0])

    # 断言调用 dup_sub_ground 函数后的返回值符合预期
    assert dup_sub_ground(f, ZZ(4), ZZ) == g


# 定义用于测试 dmp_sub_ground 函数的函数
def test_dmp_sub_ground():
    # 将 f 初始化为 [[1], [2], [3], [4]]
    f = ZZ.map([[1], [2], [3], [4]])
    # 将 g 初始化为 [[1], [2], [3], []]
    g = ZZ.map([[1], [2], [3], []])

    # 断言调用 dmp_sub_ground 函数后的返回值符合预期
    assert dmp_sub_ground(f, ZZ(4), 1, ZZ) == g


# 定义用于测试 dup_mul_ground 函数的函数
def test_dup_mul_ground():
    # 创建一个空的多项式 f
    f = dup_normal([], ZZ)

    # 断言调用 dup_mul_ground 函数后的返回值符合预期
    assert dup_mul_ground(f, ZZ(2), ZZ) == dup_normal([], ZZ)

    # 将 f 初始化为 [1, 2, 3]
    f = dup_normal([1, 2, 3], ZZ)

    # 断言调用 dup_mul_ground 函数后的返回值符合预期
    assert dup_mul_ground(f, ZZ(0), ZZ) == dup_normal([], ZZ)
    # 断言调用 dup_mul_ground 函数后的返回值符合预期
    assert dup_mul_ground(f, ZZ(2), ZZ) == dup_normal([2, 4, 6], ZZ)


# 定义用于测试 dmp_mul_ground 函数的函数
def test_dmp_mul_ground():
    # 断言调用 dmp_mul_ground 函数后的返回值符合预期
    assert dmp_mul_ground(f_0, ZZ(2), 2, ZZ) == [
        [[ZZ(2), ZZ(4), ZZ(6)], [ZZ(4)]],
        [[ZZ(6)]],
        [[ZZ(8), ZZ(10), ZZ(12)], [ZZ(2), ZZ(4), ZZ(2)], [ZZ(2)]]
    ]

    # 断言调用 dmp_mul_ground 函数后的返回值符合预期
    assert dmp_mul_ground(F_0, QQ(1, 2), 2, QQ) == [
        [[QQ(1, 14), QQ(2, 14), QQ(3, 14)], [QQ(2, 14)]],
        [[QQ(3, 14)]],
        [[QQ(4, 14), QQ(5, 14), QQ(6, 14)], [
    # 对输入的多项式 f 进行 Duplication 算法的正规化，使用整数环 ZZ 作为系数环
    f = dup_normal([6, 2, 8], ZZ)
    
    # 断言：使用整数环 ZZ 对 f 进行以 1 为除数的 Duplication 算法，结果应该等于 f 本身
    assert dup_quo_ground(f, ZZ(1), ZZ) == f
    
    # 断言：使用整数环 ZZ 对 f 进行以 2 为除数的 Duplication 算法，结果应该等于给定的多项式 [3, 1, 4]
    assert dup_quo_ground(f, ZZ(2), ZZ) == dup_normal([3, 1, 4], ZZ)
    
    # 断言：使用整数环 ZZ 对 f 进行以 3 为除数的 Duplication 算法，结果应该等于给定的多项式 [2, 0, 2]
    assert dup_quo_ground(f, ZZ(3), ZZ) == dup_normal([2, 0, 2], ZZ)
    
    # 对输入的多项式 f 进行 Duplication 算法的正规化，使用有理数环 QQ 作为系数环
    f = dup_normal([6, 2, 8], QQ)
    
    # 断言：使用有理数环 QQ 对 f 进行以 1 为除数的 Duplication 算法，结果应该等于 f 本身
    assert dup_quo_ground(f, QQ(1), QQ) == f
    
    # 断言：使用有理数环 QQ 对 f 进行以 2 为除数的 Duplication 算法，结果应该等于给定的多项式 [QQ(3), QQ(1), QQ(4)]
    assert dup_quo_ground(f, QQ(2), QQ) == [QQ(3), QQ(1), QQ(4)]
    
    # 断言：使用有理数环 QQ 对 f 进行以 7 为除数的 Duplication 算法，结果应该等于给定的多项式 [QQ(6, 7), QQ(2, 7), QQ(8, 7)]
    assert dup_quo_ground(f, QQ(7), QQ) == [QQ(6, 7), QQ(2, 7), QQ(8, 7)]
# 测试函数，验证 dup_exquo_ground 函数对于 ZeroDivisionError 是否抛出异常
def test_dup_exquo_ground():
    raises(ZeroDivisionError, lambda: dup_exquo_ground(dup_normal([1,
           2, 3], ZZ), ZZ(0), ZZ))
    
    # 测试函数，验证 dup_exquo_ground 函数对于 ExactQuotientFailed 是否抛出异常
    raises(ExactQuotientFailed, lambda: dup_exquo_ground(dup_normal([1,
           2, 3], ZZ), ZZ(3), ZZ))

    # 创建一个空的多项式 f
    f = dup_normal([], ZZ)

    # 断言 dup_exquo_ground 函数对于 f 和 3 的除法结果为一个空的多项式
    assert dup_exquo_ground(f, ZZ(3), ZZ) == dup_normal([], ZZ)

    # 创建一个多项式 f
    f = dup_normal([6, 2, 8], ZZ)

    # 断言 dup_exquo_ground 函数对于 f 和 1 的除法结果为 f 自身
    assert dup_exquo_ground(f, ZZ(1), ZZ) == f

    # 断言 dup_exquo_ground 函数对于 f 和 2 的除法结果为 [3, 1, 4]
    assert dup_exquo_ground(f, ZZ(2), ZZ) == dup_normal([3, 1, 4], ZZ)

    # 创建一个多项式 f
    f = dup_normal([6, 2, 8], QQ)

    # 断言 dup_exquo_ground 函数对于 f 和 1 的除法结果为 f 自身
    assert dup_exquo_ground(f, QQ(1), QQ) == f

    # 断言 dup_exquo_ground 函数对于 f 和 2 的除法结果为 [QQ(3), QQ(1), QQ(4)]
    assert dup_exquo_ground(f, QQ(2), QQ) == [QQ(3), QQ(1), QQ(4)]

    # 断言 dup_exquo_ground 函数对于 f 和 7 的除法结果为 [QQ(6, 7), QQ(2, 7), QQ(8, 7)]
    assert dup_exquo_ground(f, QQ(7), QQ) == [QQ(6, 7), QQ(2, 7), QQ(8, 7)]


# 测试函数，验证 dmp_quo_ground 函数
def test_dmp_quo_ground():
    # 创建一个多项式 f
    f = dmp_normal([[6], [2], [8]], 1, ZZ)

    # 断言 dmp_quo_ground 函数对于 f 和 1 的除法结果为 f 自身
    assert dmp_quo_ground(f, ZZ(1), 1, ZZ) == f

    # 断言 dmp_quo_ground 函数对于 f 和 2 的除法结果为 [[3], [1], [4]]
    assert dmp_quo_ground(f, ZZ(2), 1, ZZ) == dmp_normal([[3], [1], [4]], 1, ZZ)

    # 断言 dmp_normal 函数对于 dmp_quo_ground 函数的结果进行标准化的结果与 f 标准化的结果相同
    assert dmp_normal(dmp_quo_ground(
        f, ZZ(3), 1, ZZ), 1, ZZ) == dmp_normal([[2], [], [2]], 1, ZZ)


# 测试函数，验证 dmp_exquo_ground 函数
def test_dmp_exquo_ground():
    # 创建一个多项式 f
    f = dmp_normal([[6], [2], [8]], 1, ZZ)

    # 断言 dmp_exquo_ground 函数对于 f 和 1 的除法结果为 f 自身
    assert dmp_exquo_ground(f, ZZ(1), 1, ZZ) == f

    # 断言 dmp_exquo_ground 函数对于 f 和 2 的除法结果为 [[3], [1], [4]]
    assert dmp_exquo_ground(f, ZZ(2), 1, ZZ) == dmp_normal([[3], [1], [4]], 1, ZZ)


# 测试函数，验证 dup_lshift 函数
def test_dup_lshift():
    # 断言 dup_lshift 函数对于空列表和位移 3 的结果为一个空列表
    assert dup_lshift([], 3, ZZ) == []

    # 断言 dup_lshift 函数对于 [1] 和位移 3 的结果为 [1, 0, 0, 0]
    assert dup_lshift([1], 3, ZZ) == [1, 0, 0, 0]


# 测试函数，验证 dup_rshift 函数
def test_dup_rshift():
    # 断言 dup_rshift 函数对于空列表和位移 3 的结果为一个空列表
    assert dup_rshift([], 3, ZZ) == []

    # 断言 dup_rshift 函数对于 [1, 0, 0, 0] 和位移 3 的结果为 [1]
    assert dup_rshift([1, 0, 0, 0], 3, ZZ) == [1]


# 测试函数，验证 dup_abs 函数
def test_dup_abs():
    # 断言 dup_abs 函数对于空列表和环 ZZ 的结果为一个空列表
    assert dup_abs([], ZZ) == []

    # 断言 dup_abs 函数对于 [ZZ(1)] 和环 ZZ 的结果为 [ZZ(1)]
    assert dup_abs([ZZ(1)], ZZ) == [ZZ(1)]

    # 断言 dup_abs 函数对于 [ZZ(-7)] 和环 ZZ 的结果为 [ZZ(7)]
    assert dup_abs([ZZ(-7)], ZZ) == [ZZ(7)]

    # 断言 dup_abs 函数对于 [ZZ(-1), ZZ(2), ZZ(3)] 和环 ZZ 的结果为 [ZZ(1), ZZ(2), ZZ(3)]
    assert dup_abs([ZZ(-1), ZZ(2), ZZ(3)], ZZ) == [ZZ(1), ZZ(2), ZZ(3)]

    # 断言 dup_abs 函数对于空列表和域 QQ 的结果为一个空列表
    assert dup_abs([], QQ) == []

    # 断言 dup_abs 函数对于 [QQ(1, 2)] 和域 QQ 的结果为 [QQ(1, 2)]
    assert dup_abs([QQ(1, 2)], QQ) == [QQ(1, 2)]

    # 断言 dup_abs 函数对于 [QQ(-7, 3)] 和域 QQ 的结果为 [QQ(7, 3)]
    assert dup_abs([QQ(-7, 3)], QQ) == [QQ(7, 3)]

    # 断言 dup_abs 函数对于 [QQ(-1, 7), QQ(2, 7), QQ(3, 7)] 和域 QQ 的结果为 [QQ(1, 7), QQ(2, 7), QQ(3, 7)]
    assert dup_abs(
        [QQ(-1, 7), QQ(2, 7), QQ(3, 7)], QQ) == [QQ(1, 7), QQ(2, 7), QQ(3, 7)]


# 测试函数，验证 dmp_abs 函数
def test_dmp_abs():
    # 断言 dmp_abs 函数对于 [ZZ(-1)]、级别 0 和环 ZZ 的结果为 [ZZ(1)]
    assert dmp_abs([ZZ(-1)], 0, ZZ) == [ZZ(1)]

    # 断言 dmp_abs 函数对于 [QQ(-1, 2)]、级别 0 和域 QQ 的结果为 [QQ(1, 2)]
    assert dmp_abs([QQ(-1, 2)], 0, QQ) == [QQ(1, 2)]

    # 断言 dmp_abs 函数对于 [[[[]]]]、级别 2 和环 ZZ 的结果为 [[[[]]]]
    assert dmp_abs([[[]]], 2, ZZ) == [[[]]]

    # 断言 dmp_abs 函数对于 [[[ZZ(1)]]]、级别 2 和环 ZZ 的结果为 [[[ZZ(1)]]]
    assert dmp_abs([[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]

    # 断言 dmp_abs 函数对于 [[[ZZ(-7)]]]、级别 2 和环 ZZ 的结果为 [[[ZZ(7)]]]
    assert dmp_abs([[[ZZ(-7)]]], 2, ZZ) == [[[ZZ(7)]]]

    # 断言 dmp_abs 函数对于 [[[[]]]]、级别 2 和域 QQ 的结果为 [[[[]]]]
    assert dmp_abs([[[]]], 2, QQ) == [[[]]]

    # 断言 dmp_abs 函数对于 [[[QQ(1, 2
    # 对三维多项式数组中的每个元素进行负号操作，使用指定的环来表示元素类型
    assert dmp_neg([[[ZZ(1)]]], 2, ZZ) == [[[ZZ(-1)]]]
    # 验证负号操作是否正确，将 [[[1]]] 转换为负数 [[[ZZ(-1)]]]
    
    assert dmp_neg([[[ZZ(-7)]]], 2, ZZ) == [[[ZZ(7)]]]
    # 验证负号操作是否正确，将 [[[ZZ(-7)]]] 转换为负数 [[[ZZ(7)]]]
    
    assert dmp_neg([[[]]], 2, QQ) == [[[]]]
    # 验证负号操作是否正确，空数组不改变 [[[[]]]]
    
    assert dmp_neg([[[QQ(1, 9)]]], 2, QQ) == [[[QQ(-1, 9)]]]
    # 验证负号操作是否正确，将 [[[1/9]]] 转换为负数 [[[QQ(-1, 9)]]]
    
    assert dmp_neg([[[QQ(-7, 9)]]], 2, QQ) == [[[QQ(7, 9)]]]
    # 验证负号操作是否正确，将 [[[QQ(-7, 9)]]] 转换为负数 [[[QQ(7, 9)]]]
# 定义一个测试函数 test_dup_add，用于测试 dup_add 函数的各种情况
def test_dup_add():
    # 断言空列表和空列表进行 dup_add 操作结果为空列表
    assert dup_add([], [], ZZ) == []
    # 断言包含一个 ZZ(1) 的列表与空列表进行 dup_add 操作结果为 [ZZ(1)]
    assert dup_add([ZZ(1)], [], ZZ) == [ZZ(1)]
    # 断言空列表与包含一个 ZZ(1) 的列表进行 dup_add 操作结果为 [ZZ(1)]
    assert dup_add([], [ZZ(1)], ZZ) == [ZZ(1)]
    # 断言两个包含 ZZ(1) 的列表进行 dup_add 操作结果为 [ZZ(2)]
    assert dup_add([ZZ(1)], [ZZ(1)], ZZ) == [ZZ(2)]
    # 断言一个包含 ZZ(1) 的列表和一个包含 ZZ(2) 的列表进行 dup_add 操作结果为 [ZZ(3)]
    assert dup_add([ZZ(1)], [ZZ(2)], ZZ) == [ZZ(3)]

    # 断言包含 ZZ(1) 和 ZZ(2) 的列表与包含 ZZ(1) 的列表进行 dup_add 操作结果为 [ZZ(1), ZZ(3)]
    assert dup_add([ZZ(1), ZZ(2)], [ZZ(1)], ZZ) == [ZZ(1), ZZ(3)]
    # 断言包含 ZZ(1) 的列表与包含 ZZ(1) 和 ZZ(2) 的列表进行 dup_add 操作结果为 [ZZ(1), ZZ(3)]
    assert dup_add([ZZ(1)], [ZZ(1), ZZ(2)], ZZ) == [ZZ(1), ZZ(3)]

    # 断言包含 ZZ(1), ZZ(2), ZZ(3) 的列表与包含 ZZ(8), ZZ(9), ZZ(10) 的列表进行 dup_add 操作结果为 [ZZ(9), ZZ(11), ZZ(13)]
    assert dup_add([ZZ(1), ZZ(2), ZZ(3)], [ZZ(8), ZZ(9), ZZ(10)], ZZ) == [ZZ(9), ZZ(11), ZZ(13)]

    # 断言空列表和空列表进行 dup_add 操作结果为空列表
    assert dup_add([], [], QQ) == []
    # 断言包含一个 QQ(1, 2) 的列表与空列表进行 dup_add 操作结果为 [QQ(1, 2)]
    assert dup_add([QQ(1, 2)], [], QQ) == [QQ(1, 2)]
    # 断言空列表与包含一个 QQ(1, 2) 的列表进行 dup_add 操作结果为 [QQ(1, 2)]
    assert dup_add([], [QQ(1, 2)], QQ) == [QQ(1, 2)]
    # 断言两个包含 QQ(1, 4) 的列表进行 dup_add 操作结果为 [QQ(1, 2)]
    assert dup_add([QQ(1, 4)], [QQ(1, 4)], QQ) == [QQ(1, 2)]
    # 断言一个包含 QQ(1, 4) 的列表和一个包含 QQ(1, 2) 的列表进行 dup_add 操作结果为 [QQ(3, 4)]
    assert dup_add([QQ(1, 4)], [QQ(1, 2)], QQ) == [QQ(3, 4)]

    # 断言包含 QQ(1, 2) 和 QQ(2, 3) 的列表与包含 QQ(1) 的列表进行 dup_add 操作结果为 [QQ(1, 2), QQ(5, 3)]
    assert dup_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ) == [QQ(1, 2), QQ(5, 3)]
    # 断言包含 QQ(1) 的列表与包含 QQ(1, 2) 和 QQ(2, 3) 的列表进行 dup_add 操作结果为 [QQ(1, 2), QQ(5, 3)]
    assert dup_add([QQ(1)], [QQ(1, 2), QQ(2, 3)], QQ) == [QQ(1, 2), QQ(5, 3)]

    # 断言包含 QQ(1, 7), QQ(2, 7), QQ(3, 7) 的列表与包含 QQ(8, 7), QQ(9, 7), QQ(10, 7) 的列表进行 dup_add 操作结果为 [QQ(9, 7), QQ(11, 7), QQ(13, 7)]
    assert dup_add([QQ(1, 7), QQ(2, 7), QQ(3, 7)], [QQ(8, 7), QQ(9, 7), QQ(10, 7)], QQ) == [QQ(9, 7), QQ(11, 7), QQ(13, 7)]


# 定义一个测试函数 test_dmp_add，用于测试 dmp_add 函数的各种情况
def test_dmp_add():
    # 断言包含 ZZ(1) 和 ZZ(2) 的列表与包含 ZZ(1) 的列表进行 dmp_add 操作结果与对应的 dup_add 操作结果相同
    assert dmp_add([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == \
        dup_add([ZZ(1), ZZ(2)], [ZZ(1)], ZZ)
    # 断言包含 QQ(1, 2) 和 QQ(2, 3) 的列表与包含 QQ(1) 的列表进行 dmp_add 操作结果与对应的 dup_add 操作结果相同
    assert dmp_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], 0, QQ) == \
        dup_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ)

    # 断言包含空列表的三重嵌套列表与另一个相同的三重嵌套列表进行 dmp_add 操作结果为相同的三重嵌套列表
    assert dmp_add([[[]]], [[[]]], 2, ZZ) == [[[]]]
    # 断言包含一个 ZZ(1) 的三重嵌套列表与包含空列表的三重嵌套列表进行 dmp_add 操作结果为包含一个 ZZ(1) 的三重嵌套列表
    assert dmp_add([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[ZZ(1)]]]
    # 断言包含空列表的三重嵌套列表与包含一个 ZZ(1) 的三重嵌套列表进行 dmp_add 操作结果为包含一个 ZZ(1) 的三重嵌套列表
    assert dmp_add([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    # 断言包含一个 ZZ(2) 的三重嵌套列表与包含一个 ZZ(1) 的三重嵌套列表进行 dmp_add 操作结果为包含一个 ZZ(3) 的三重嵌套列表
    assert dmp_add([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(3)]]]
    # 断言包含一个 ZZ(1) 的三重嵌套列表与包含一个 ZZ(2) 的三重嵌套列表进行 dmp_add 操作结果为包含一个 ZZ(3) 的三重嵌套列表
    assert dmp_add([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(3)]]]

    # 断言包含空列表的三重嵌套列表与另一个相同的三重嵌套列表进行 dmp_add 操作结果为相同的三重嵌套列表
    assert dmp_add([[[]]], [[[]]], 2, QQ) == [[[]]]
    # 断言包含一个 QQ(1, 2) 的三重嵌套列表与包含空列表的三重嵌套列表进行 dmp_add 操作结果为包含一个 QQ(1, 2) 的三重嵌套列表
    assert dmp_add([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[QQ(1, 2)]]]
    # 断言包含空列表的三重嵌套列表与包含一个 QQ(1, 2) 的三重嵌套列表进行 dmp_add 操作
# 定义测试函数 `test_dmp_sub`
def test_dmp_sub():
    # 断言两个多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == \
        dup_sub([ZZ(1), ZZ(2)], [ZZ(1)], ZZ)
    # 断言两个多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([QQ(1, 2), QQ(2, 3)], [QQ(1)], 0, QQ) == \
        dup_sub([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ)

    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[]]], [[[]]], 2, ZZ) == [[[]]]
    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[ZZ(1)]]]
    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(-1)]]]
    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(-1)]]]

    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[]]], [[[]]], 2, QQ) == [[[]]]
    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[QQ(1, 2)]]]
    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[QQ(-1, 2)]]]
    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[QQ(2, 7)]]], [[[QQ(1, 7)]]], 2, QQ) == [[[QQ(1, 7)]]]
    # 断言两个三维多项式列表的差的计算，使用了特定的对象和运算函数
    assert dmp_sub([[[QQ(1, 7)]]], [[[QQ(2, 7)]]], 2, QQ) == [[[QQ(-1, 7)]]]


# 定义测试函数 `test_dup_add_mul`
def test_dup_add_mul():
    # 断言两个多项式列表的加权乘积的计算，使用了特定的对象和运算函数
    assert dup_add_mul([ZZ(1), ZZ(2), ZZ(3)], [ZZ(3), ZZ(2), ZZ(1)],
               [ZZ(1), ZZ(2)], ZZ) == [ZZ(3), ZZ(9), ZZ(7), ZZ(5)]
    # 断言两个多维多项式列表的加权乘积的计算，使用了特定的对象和运算函数
    assert dmp_add_mul([[ZZ(1), ZZ(2)], [ZZ(3)]], [[ZZ(3)], [ZZ(2), ZZ(1)]],
               [[ZZ(1)], [ZZ(2)]], 1, ZZ) == [[ZZ(3)], [ZZ(3), ZZ(9)], [ZZ(4), ZZ(5)]]


# 定义测试函数 `test_dup_sub_mul`
def test_dup_sub_mul():
    # 断言两个多项式列表的加权差的乘积的计算，使用了特定的对象和运算函数
    assert dup_sub_mul([ZZ(1), ZZ(2), ZZ(3)], [ZZ(3), ZZ(2), ZZ(1)],
               [ZZ(1), ZZ(2)], ZZ) == [ZZ(-3), ZZ(-7), ZZ(-3), ZZ(1)]
    # 断言两个多维多项式列表的加权差的乘积的计算，使用了特定的对象和运算函数
    assert dmp_sub_mul([[ZZ(1), ZZ(2)], [ZZ(3)]], [[ZZ(3)], [ZZ(2), ZZ(1)]],
               [[ZZ(1)], [ZZ(2)]], 1, ZZ) == [[ZZ(-3)], [ZZ(-1), ZZ(-5)], [ZZ(-4), ZZ(1)]]


# 定义测试函数 `test_dup_mul`
def test_dup_mul():
    # 断言两个多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([], [], ZZ) == []
    # 断言两个多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([], [ZZ(1)], ZZ) == []
    # 断言两个多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([ZZ(1)], [], ZZ) == []
    # 断言两个多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([ZZ(1)], [ZZ(1)], ZZ) == [ZZ(1)]
    # 断言两个多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([ZZ(5)], [ZZ(7)], ZZ) == [ZZ(35)]

    # 断言两个有理数多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([], [], QQ) == []
    # 断言两个有理数多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([], [QQ(1, 2)], QQ) == []
    # 断言两个有理数多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([QQ(1, 2)], [], QQ) == []
    # 断言两个有理数多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([QQ(1, 2)], [QQ(4, 7)], QQ) == [QQ(2, 7)]
    # 断言两个有理数多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul([QQ(5, 7)], [QQ(3, 7)], QQ) == [QQ(15, 49)]

    # 计算整数多项式列表的乘积，使用了特定的对象和运算函数
    f = dup_normal([3, 0, 0, 6, 1, 2], ZZ)
    g = dup_normal([4, 0, 1, 0], ZZ)
    h = dup_normal([12, 0, 3, 24, 4, 14, 1, 2, 0], ZZ)

    # 断言两个多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul(f, g, ZZ) == h
    # 断言两个多项式列表的乘积的计算，使用了特定的对象和运算函数
    assert dup_mul(g, f, ZZ) == h

    # 计算整数多项式列表的乘积，使用了特定的对象和运算函数
    f = dup_normal([2, 0, 0, 1, 7],
    p1 = dup_normal([79, -1, 78, -94, -10, 11, 32, -19, 78, 2, -89, 30, 73, 42,
        85, 77, 83, -30, -34, -2, 95, -81, 37, -49, -46, -58, -16, 37, 35, -11,
        -57, -15, -31, 67, -20, 27, 76, 2, 70, 67, -65, 65, -26, -93, -44, -12,
        -92, 57, -90, -57, -11, -67, -98, -69, 97, -41, 89, 33, 89, -50, 81,
        -31, 60, -27, 43, 29, -77, 44, 21, -91, 32, -57, 33, 3, 53, -51, -38,
        -99, -84, 23, -50, 66, -100, 1, -75, -25, 27, -60, 98, -51, -87, 6, 8,
        78, -28, -95, -88, 12, -35, 26, -9, 16, -92, 55, -7, -86, 68, -39, -46,
        84, 94, 45, 60, 92, 68, -75, -74, -19, 8, 75, 78, 91, 57, 34, 14, -3,
        -49, 65, 78, -18, 6, -29, -80, -98, 17, 13, 58, 21, 20, 9, 37, 7, -30,
        -53, -20, 34, 67, -42, 89, -22, 73, 43, -6, 5, 51, -8, -15, -52, -22,
        -58, -72, -3, 43, -92, 82, 83, -2, -13, -23, -60, 16, -94, -8, -28,
        -95, -72, 63, -90, 76, 6, -43, -100, -59, 76, 3, 3, 46, -85, 75, 62,
        -71, -76, 88, 97, -72, -1, 30, -64, 72, -48, 14, -78, 58, 63, -91, 24,
        -87, -27, -80, -100, -44, 98, 70, 100, -29, -38, 11, 77, 100, 52, 86,
        65, -5, -42, -81, -38, -42, 43, -2, -70, -63, -52], ZZ)
    # 创建第一个多项式 p1，使用给定的系数列表和环 ZZ

    p2 = dup_normal([65, -19, -47, 1, 90, 81, -15, -34, 25, -75, 9, -83, 50, -5,
        -44, 31, 1, 70, -7, 78, 74, 80, 85, 65, 21, 41, 66, 19, -40, 63, -21,
        -27, 32, 69, 83, 34, -35, 14, 81, 57, -75, 32, -67, -89, -100, -61, 46,
        84, -78, -29, -50, -94, -24, -32, -68, -16, 100, -7, -72, -89, 35, 82,
        58, 81, -92, 62, 5, -47, -39, -58, -72, -13, 84, 44, 55, -25, 48, -54,
        -31, -56, -11, -50, -84, 10, 67, 17, 13, -14, 61, 76, -64, -44, -40,
        -96, 11, -11, -94, 2, 6, 27, -6, 68, -54, 66, -74, -14, -1, -24, -73,
        96, 89, -11, -89, 56, -53, 72, -43, 96, 25, 63, -31, 29, 68, 83, 91,
        -93, -19, -38, -40, 40, -12, -19, -79, 44, 100, -66, -29, -77, 62, 39,
        -8, 11, -97, 14, 87, 64, 21, -18, 13, 15, -59, -75, -99, -88, 57, 54,
        56, -67, 6, -63, -59, -14, 28, 87, -20, -39, 84, -91, -2, 49, -75, 11,
        -24, -95, 36, 66, 5, 25, -72, -40, 86, 90, 37, -33, 57, -35, 29, -18,
        4, -79, 64, -17, -27, 21, 29, -5, -44, -87, -24, 52, 78, 11, -23, -53,
        36, 42, 21, -68, 94, -91, -51, -21, 51, -76, 72, 31, 24, -48, -80, -9,
        37, -47, -6, -8, -63, -91, 79, -79, -100, 38, -20, 38, 100, 83, -90,
        87, 63, -36, 82, -19, 18, -98, -38, 26, 98, -70, 79, 92, 12, 12, 70,
        74, 36, 48, -13, 31, 31, -47, -71, -12, -64, 36, -42, 32, -86, 60, 83,
        70, 55, 0, 1, 29, -35, 8, -82, 8, -73, -46, -50, 43, 48, -5, -86, -72,
        44, -90, 19, 19, 5, -20, 97, -13, -66, -5, 5, -69, 64, -30, 41, 51, 36,
        13, -99, -61, 94, -12, 74, 98, 68, 24, 46, -97, -87, -6, -27, 82, 62,
        -11, -77, 86, 66, -47, -49, -50, 13, 18, 89, -89, 46, -80, 13, 98, -35,
        -36, -25, 12, 20, 26, -52, 79, 27, 79, 100, 8, 62, -58, -28, 37], ZZ)
    # 创建第二个多项式 p2，使用给定的系数列表和环 ZZ

    assert dup_mul(p1, p2, ZZ) == res
    # 断言：调用 dup_mul 函数计算 p1 和 p2 的乘积，并与预期结果 res 比较
    # 创建变量 p1，调用 dup_normal 函数生成一个长列表，包含大量整数
    p1 = dup_normal([83, -61, -86, -24, 12, 43, -88, -9, 42, 55, -66, 74, 95,
        -25, -12, 68, -99, 4, 45, 6, -15, -19, 78, 65, -55, 47, -13, 17, 86,
        81, -58, -27, 50, -40, -24, 39, -41, -92, 75, 90, -1, 40, -15, -27,
        -35, 68, 70, -64, -40, 78, -88, -58, -39, 69, 46, 12, 28, -94, -37,
        -50, -80, -96, -61, 25, 1, 71, 4, 12, 48, 4, 34, -47, -75, 5, 48, 82,
        88, 23, 98, 35, 17, -10, 48, -61, -95, 47, 65, -19, -66, -57, -6, -51,
        -42, -89, 66, -13, 18, 37, 90, -23, 72, 96, -53, 0, 40, -73, -52, -68,
        32, -25, -53, 79, -52, 18, 44, 73, -81, 31, -90, 70, 3, 36, 48, 76,
        -24, -44, 23, 98, -4, 73, 69, 88, -70, 14, -68, 94, -78, -15, -64, -97,
        -70, -35, 65, 88, 49, -53, -7, 12, -45, -7, 59, -94, 99, -2, 67, -60,
        -71, 29, -62, -77, 1, 51, 17, 80, -20, -47, -19, 24, -9, 39, -23, 21,
        -84, 10, 84, 56, -17, -21, -66, 85, 70, 46, -51, -22, -95, 78, -60,
        -96, -97, -45, 72, 35, 30, -61, -92, -93, -60, -61, 4, -4, -81, -73,
        46, 53, -11, 26, 94, 45, 14, -78, 55, 84, -68, 98, 60, 23, 100, -63,
        68, 96, -16, 3, 56, 21, -58, 62, -67, 66, 85, 41, -79, -22, 97, -67,
        82, 82, -96, -20, -7, 48, -67, 48, -9, -39, 78], ZZ)
    # 创建变量 p2，调用 dup_normal 函数生成一个长列表，包含大量整数
    p2 = dup_normal([52, 88, 76, 66, 9, -64, 46, -20, -28, 69, 60, 96, -36,
        -92, -30, -11, -35, 35, 55, 63, -92, -7, 25, -58, 74, 55, -6, 4, 47,
        -92, -65, 67, -45, 74, -76, 59, -6, 69, 39, 24, -71, -7, 39, -45, 60,
        -68, 98, 97, -79, 17, 4, 94, -64, 68, -100, -96, -2, 3, 22, 96, 54,
        -77, -86, 67, 6, 57, 37, 40, 89, -78, 64, -94, -45, -92, 57, 87, -26,
        36, 19, 97, 25, 77, -87, 24, 43, -5, 35, 57, 83, 71, 35, 63, 61, 96,
        -22, 8, -1, 96, 43, 45, 94, -93, 36, 71, -41, -99, 85, -48, 59, 52,
        -17, 5, 87, -16, -68, -54, 76, -18, 100, 91, -42, -70, -66, -88, -12,
        1, 95, -82, 52, 43, -29, 3, 12, 72, -99, -43, -32, -93, -51, 16, -20,
        -12, -11, 5, 33, -38, 93, -5, -74, 25, 74, -58, 93, 59, -63, -86, 63,
        -20, -4, -74, -73, -95, 29, -28, 93, -91, -2, -38, -62, 77, -58, -85,
        -28, 95, 38, 19, -69, 86, 94, 25, -2, -4, 47, 34, -59, 35, -48, 29,
        -63, -53, 34, 29, 66, 73, 6, 92, -84, 89, 15, 81, 93, 97, 51, -72, -78,
        25, 60, 90, -45, 39, 67, -84, -62, 57, 26, -32, -56, -14, -83, 76, 5,
        -2, 99, -100, 28, 46, 94, -7, 53, -25, 16, -23, -36, 89, -78, -63, 31,
        1, 84, -99, -52, 76, 48, 90, -76, 44, -19, 54, -36, -9, -73, -100, -69,
        31, 42, 25, -39, 76, -26, -8, -14, 51, 3, 37, 45, 2, -54, 13, -34, -92,
        17, -25, -65, 53, -63, 30, 4, -70, -67, 90, 52, 51, 18, -3, 31, -45,
        -9, 59, 63, -87, 22, -32, 29, -38, 21, 36, -82, 27, -11], ZZ)
    # 断言：调用 dup_mul 函数对 p1 和 p2 进行乘法运算，期望结果为变量 res 的值
    assert dup_mul(p1, p2, ZZ) == res
def test_dmp_mul():
    # 测试 dmp_mul 函数，对整数多项式进行乘法运算
    assert dmp_mul([ZZ(5)], [ZZ(7)], 0, ZZ) == \
        dup_mul([ZZ(5)], [ZZ(7)], ZZ)
    # 测试 dmp_mul 函数，对有理数多项式进行乘法运算
    assert dmp_mul([QQ(5, 7)], [QQ(3, 7)], 0, QQ) == \
        dup_mul([QQ(5, 7)], [QQ(3, 7)], QQ)

    # 测试 dmp_mul 函数，对多维整数多项式进行乘法运算
    assert dmp_mul([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(2)]]]
    assert dmp_mul([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(2)]]]

    # 测试 dmp_mul 函数，对多维有理数多项式进行乘法运算
    assert dmp_mul([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[QQ(2, 7)]]], [[[QQ(1, 3)]]], 2, QQ) == [[[QQ(2, 21)]]]
    assert dmp_mul([[[QQ(1, 7)]]], [[[QQ(2, 3)]]], 2, QQ) == [[[QQ(2, 21)]]]

    K = FF(6)

    # 测试 dmp_mul 函数，对有限域中的多项式进行乘法运算
    assert dmp_mul(
        [[K(2)], [K(1)]], [[K(3)], [K(4)]], 1, K) == [[K(5)], [K(4)]]


def test_dup_sqr():
    # 测试 dup_sqr 函数，对整数系数的多项式进行平方运算
    assert dup_sqr([], ZZ) == []
    assert dup_sqr([ZZ(2)], ZZ) == [ZZ(4)]
    assert dup_sqr([ZZ(1), ZZ(2)], ZZ) == [ZZ(1), ZZ(4), ZZ(4)]

    # 测试 dup_sqr 函数，对有理数系数的多项式进行平方运算
    assert dup_sqr([], QQ) == []
    assert dup_sqr([QQ(2, 3)], QQ) == [QQ(4, 9)]
    assert dup_sqr([QQ(1, 3), QQ(2, 3)], QQ) == [QQ(1, 9), QQ(4, 9), QQ(4, 9)]

    # 使用 dup_normal 函数创建多项式 f
    f = dup_normal([2, 0, 0, 1, 7], ZZ)

    # 测试 dup_sqr 函数，对 f 进行平方运算
    assert dup_sqr(f, ZZ) == dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)

    K = FF(9)

    # 测试 dup_sqr 函数，对有限域中的多项式进行平方运算
    assert dup_sqr([K(3), K(4)], K) == [K(6), K(7)]


def test_dmp_sqr():
    # 测试 dmp_sqr 函数，对多项式进行平方运算（零维）
    assert dmp_sqr([ZZ(1), ZZ(2)], 0, ZZ) == \
        dup_sqr([ZZ(1), ZZ(2)], ZZ)

    # 测试 dmp_sqr 函数，对多维整数系数的多项式进行平方运算
    assert dmp_sqr([[[]]], 2, ZZ) == [[[]]]
    assert dmp_sqr([[[ZZ(2)]]], 2, ZZ) == [[[ZZ(4)]]]

    # 测试 dmp_sqr 函数，对多维有理数系数的多项式进行平方运算
    assert dmp_sqr([[[]]], 2, QQ) == [[[]]]
    assert dmp_sqr([[[QQ(2, 3)]]], 2, QQ) == [[[QQ(4, 9)]]]

    K = FF(9)

    # 测试 dmp_sqr 函数，对有限域中的多项式进行平方运算
    assert dmp_sqr([[K(3)], [K(4)]], 1, K) == [[K(6)], [K(7)]]


def test_dup_pow():
    # 测试 dup_pow 函数，对整数系数的多项式进行幂运算
    assert dup_pow([], 0, ZZ) == [ZZ(1)]
    assert dup_pow([], 0, QQ) == [QQ(1)]

    assert dup_pow([], 1, ZZ) == []
    assert dup_pow([], 7, ZZ) == []

    assert dup_pow([ZZ(1)], 0, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(1)], 1, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(1)], 7, ZZ) == [ZZ(1)]

    assert dup_pow([ZZ(3)], 0, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(3)], 1, ZZ) == [ZZ(3)]
    assert dup_pow([ZZ(3)], 7, ZZ) == [ZZ(2187)]

    # 测试 dup_pow 函数，对有理数系数的多项式进行幂运算
    assert dup_pow([QQ(1, 1)], 0, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(1, 1)], 1, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(1, 1)], 7, QQ) == [QQ(1, 1)]

    assert dup_pow([QQ(3, 7)], 0, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(3, 7)], 1, QQ) == [QQ(3, 7)]
    assert dup_pow([QQ(3, 7)], 7, QQ) == [QQ(2187, 823543)]

    # 使用 dup_normal 函数创建多项式 f
    f = dup_normal([2, 0, 0, 1, 7], ZZ)

    # 测试 dup_pow 函数，对 f 进行幂运算
    assert dup_pow(f, 0, ZZ) == dup_normal([1], ZZ)
    assert dup_pow(f, 1, ZZ) == dup_normal([2, 0, 0, 1, 7], ZZ)
    assert dup_pow(f, 2, ZZ) == dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)
    assert dup_pow(f, 3, ZZ) == dup_normal(
        [8, 0, 0, 12, 84, 0, 6, 84, 294, 1, 21, 147, 343], ZZ)


def test_dmp_pow():
    # 待补充的测试函数，暂无代码
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[]], 0, 1, ZZ) == [[ZZ(1)]]
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[]], 0, 1, QQ) == [[QQ(1)]]
    
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[]], 1, 1, ZZ) == [[]]
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[]], 7, 1, ZZ) == [[]]
    
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[ZZ(1)]], 0, 1, ZZ) == [[ZZ(1)]]
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[ZZ(1)]], 1, 1, ZZ) == [[ZZ(1)]]
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[ZZ(1)]], 7, 1, ZZ) == [[ZZ(1)]]
    
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[QQ(3, 7)]], 0, 1, QQ) == [[QQ(1, 1)]]
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[QQ(3, 7)]], 1, 1, QQ) == [[QQ(3, 7)]]
    # 断言：验证 dmp_pow 函数对于给定的参数返回预期的结果
    assert dmp_pow([[QQ(3, 7)]], 7, 1, QQ) == [[QQ(2187, 823543)]]
    
    # 创建一个多项式 f = [2, 0, 0, 1, 7]，其系数在整数环 ZZ 中
    f = dup_normal([2, 0, 0, 1, 7], ZZ)
    
    # 断言：验证 dmp_pow 函数与 dup_pow 函数在给定参数下返回相同的结果
    assert dmp_pow(f, 2, 0, ZZ) == dup_pow(f, 2, ZZ)
# 测试 dup_pdiv 函数：用于多项式整除的测试
def test_dup_pdiv():
    # 创建多项式 f 和 g，使用整数环 ZZ
    f = dup_normal([3, 1, 1, 5], ZZ)
    g = dup_normal([5, -3, 1], ZZ)

    # 手动计算的商和余数
    q = dup_normal([15, 14], ZZ)
    r = dup_normal([52, 111], ZZ)

    # 验证 dup_pdiv 函数的返回值是否等于手动计算的商和余数
    assert dup_pdiv(f, g, ZZ) == (q, r)

    # 验证 dup_pquo 函数的返回值是否等于手动计算的商
    assert dup_pquo(f, g, ZZ) == q

    # 验证 dup_prem 函数的返回值是否等于手动计算的余数
    assert dup_prem(f, g, ZZ) == r

    # 验证当无法精确除法时是否引发 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: dup_pexquo(f, g, ZZ))

    # 使用有理数环 QQ 测试相同的函数
    f = dup_normal([3, 1, 1, 5], QQ)
    g = dup_normal([5, -3, 1], QQ)

    q = dup_normal([15, 14], QQ)
    r = dup_normal([52, 111], QQ)

    assert dup_pdiv(f, g, QQ) == (q, r)
    assert dup_pquo(f, g, QQ) == q
    assert dup_prem(f, g, QQ) == r

    raises(ExactQuotientFailed, lambda: dup_pexquo(f, g, QQ))


# 测试 dmp_pdiv 函数：用于多元多项式整除的测试
def test_dmp_pdiv():
    # 创建多元多项式 f 和 g，使用整数环 ZZ
    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[1], [-1, 0]], 1, ZZ)

    q = dmp_normal([[1], [1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    # 验证 dmp_pdiv 函数的返回值是否等于手动计算的商和余数
    assert dmp_pdiv(f, g, 1, ZZ) == (q, r)
    assert dmp_pquo(f, g, 1, ZZ) == q
    assert dmp_prem(f, g, 1, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_pexquo(f, g, 1, ZZ))

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[2], [-2, 0]], 1, ZZ)

    q = dmp_normal([[2], [2, 0]], 1, ZZ)
    r = dmp_normal([[8, 0, 0]], 1, ZZ)

    assert dmp_pdiv(f, g, 1, ZZ) == (q, r)
    assert dmp_pquo(f, g, 1, ZZ) == q
    assert dmp_prem(f, g, 1, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_pexquo(f, g, 1, ZZ))


# 测试 dup_rr_div 函数：测试重根多项式的除法
def test_dup_rr_div():
    # 验证当除数是空列表时是否引发 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: dup_rr_div([1, 2, 3], [], ZZ))

    # 创建多项式 f 和 g，使用整数环 ZZ
    f = dup_normal([3, 1, 1, 5], ZZ)
    g = dup_normal([5, -3, 1], ZZ)

    # 期望的商和余数
    q, r = [], f

    # 验证 dup_rr_div 函数的返回值是否等于期望的商和余数
    assert dup_rr_div(f, g, ZZ) == (q, r)


# 测试 dmp_rr_div 函数：测试重根多元多项式的除法
def test_dmp_rr_div():
    # 验证当除数包含空列表时是否引发 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: dmp_rr_div([[1, 2], [3]], [[]], 1, ZZ))

    # 创建多元多项式 f 和 g，使用整数环 ZZ
    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[1], [-1, 0]], 1, ZZ)

    q = dmp_normal([[1], [1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    # 验证 dmp_rr_div 函数的返回值是否等于期望的商和余数
    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[-1], [1, 0]], 1, ZZ)

    q = dmp_normal([[-1], [-1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[2], [-2, 0]], 1, ZZ)

    q, r = [[]], f

    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)


# 测试 dup_ff_div 函数：测试有理数系数的多项式除法
def test_dup_ff_div():
    # 验证当除数是空列表时是否引发 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: dup_ff_div([1, 2, 3], [], QQ))

    # 创建有理数系数的多项式 f 和 g
    f = dup_normal([3, 1, 1, 5], QQ)
    g = dup_normal([5, -3, 1], QQ)

    # 期望的商和余数，使用 QQ 类型的有理数
    q = [QQ(3, 5), QQ(14, 25)]
    r = [QQ(52, 25), QQ(111, 25)]

    # 验证 dup_ff_div 函数的返回值是否等于期望的商和余数
    assert dup_ff_div(f, g, QQ) == (q, r)


# 测试 dup_ff_div_gmpy2 函数：测试使用 gmpy2 库的有理数系数的多项式除法
def test_dup_ff_div_gmpy2():
    # 如果 GROUND_TYPES 不是 'gmpy2'，则跳过测试
    if GROUND_TYPES != 'gmpy2':
        return

    # 导入必要的库和模块
    from gmpy2 import mpq
    from sympy.polys.domains import GMPYRationalField
    K = GMPYRationalField()

    # 创建有理数系数的多项式 f 和 g，使用 GMPYRationalField
    f = [mpq(1,3), mpq(3,2)]
    g = [mpq(2,1)]

    # 期望的商和余数
    assert dmp_ff_div(f, g, 0, K) == ([mpq(1,6), mpq(3,4)], [])

    f = [mpq(1,2), mpq(1,3), mpq(1,4), mpq(1,5)]
    g = [mpq(-1,1), mpq(1,1), mpq(-1,1)]

    # 验证 dmp_ff_div 函数的返回值是否等于期望的商和余数
    assert dmp_ff_div(f, g, 0, K) == ([mpq(1,2), mpq(0), mpq(0), mpq(0)], [])
    # 使用断言来验证函数调用的结果是否符合预期
    assert dmp_ff_div(f, g, 0, K) == ([mpq(-1,2), mpq(-5,6)], [mpq(7,12), mpq(-19,30)])
# 定义用于测试 dmp_ff_div 函数的测试函数
def test_dmp_ff_div():
    # 确保在调用 dmp_ff_div 时抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: dmp_ff_div([[1, 2], [3]], [[]], 1, QQ))

    # 准备测试所需的多项式 f 和 g
    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[1], [-1, 0]], 1, QQ)

    # 预期的商和余数
    q = [[QQ(1, 1)], [QQ(1, 1), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    # 断言 dmp_ff_div 的返回值与预期的商和余数相等
    assert dmp_ff_div(f, g, 1, QQ) == (q, r)

    # 重复上述步骤，测试不同的输入多项式 f 和 g
    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[-1], [1, 0]], 1, QQ)

    q = [[QQ(-1, 1)], [QQ(-1, 1), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    assert dmp_ff_div(f, g, 1, QQ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[2], [-2, 0]], 1, QQ)

    q = [[QQ(1, 2)], [QQ(1, 2), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    assert dmp_ff_div(f, g, 1, QQ) == (q, r)


# 定义用于测试 dup_div 相关函数的测试函数
def test_dup_div():
    # 设置多项式 f, g, 商 q 和余数 r
    f, g, q, r = [5, 4, 3, 2, 1], [1, 2, 3], [5, -6, 0], [20, 1]

    # 断言 dup_div 返回正确的商和余数
    assert dup_div(f, g, ZZ) == (q, r)
    assert dup_quo(f, g, ZZ) == q
    assert dup_rem(f, g, ZZ) == r

    # 确保 dup_exquo 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: dup_exquo(f, g, ZZ))

    # 重复上述步骤，测试不同的输入多项式 f, g, q 和 r
    f, g, q, r = [5, 4, 3, 2, 1, 0], [1, 2, 0, 0, 9], [5, -6], [15, 2, -44, 54]

    assert dup_div(f, g, ZZ) == (q, r)
    assert dup_quo(f, g, ZZ) == q
    assert dup_rem(f, g, ZZ) == r

    raises(ExactQuotientFailed, lambda: dup_exquo(f, g, ZZ))


# 定义用于测试 dmp_div 相关函数的测试函数
def test_dmp_div():
    # 设置多项式 f, g, 商 q 和余数 r
    f, g, q, r = [5, 4, 3, 2, 1], [1, 2, 3], [5, -6, 0], [20, 1]

    # 断言 dmp_div 返回正确的商和余数
    assert dmp_div(f, g, 0, ZZ) == (q, r)
    assert dmp_quo(f, g, 0, ZZ) == q
    assert dmp_rem(f, g, 0, ZZ) == r

    # 确保 dmp_exquo 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: dmp_exquo(f, g, 0, ZZ))

    # 重复上述步骤，测试不同的输入多项式 f, g, q 和 r
    f, g, q, r = [[[1]]], [[[2]], [1]], [[[]]], [[[1]]]

    assert dmp_div(f, g, 2, ZZ) == (q, r)
    assert dmp_quo(f, g, 2, ZZ) == q
    assert dmp_rem(f, g, 2, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_exquo(f, g, 2, ZZ))


# 定义用于测试 dup_max_norm 函数的测试函数
def test_dup_max_norm():
    # 断言 dup_max_norm 返回正确的结果
    assert dup_max_norm([], ZZ) == 0
    assert dup_max_norm([1], ZZ) == 1

    assert dup_max_norm([1, 4, 2, 3], ZZ) == 4


# 定义用于测试 dmp_max_norm 函数的测试函数
def test_dmp_max_norm():
    # 断言 dmp_max_norm 返回正确的结果
    assert dmp_max_norm([[[]]], 2, ZZ) == 0
    assert dmp_max_norm([[[1]]], 2, ZZ) == 1

    # 断言 dmp_max_norm 使用实际测试数据 f_0 返回正确的结果
    assert dmp_max_norm(f_0, 2, ZZ) == 6


# 定义用于测试 dup_l1_norm 函数的测试函数
def test_dup_l1_norm():
    # 断言 dup_l1_norm 返回正确的结果
    assert dup_l1_norm([], ZZ) == 0
    assert dup_l1_norm([1], ZZ) == 1
    assert dup_l1_norm([1, 4, 2, 3], ZZ) == 10


# 定义用于测试 dmp_l1_norm 函数的测试函数
def test_dmp_l1_norm():
    # 断言 dmp_l1_norm 返回正确的结果
    assert dmp_l1_norm([[[]]], 2, ZZ) == 0
    assert dmp_l1_norm([[[1]]], 2, ZZ) == 1

    # 断言 dmp_l1_norm 使用实际测试数据 f_0 返回正确的结果
    assert dmp_l1_norm(f_0, 2, ZZ) == 31


# 定义用于测试 dup_l2_norm_squared 函数的测试函数
def test_dup_l2_norm_squared():
    # 断言 dup_l2_norm_squared 返回正确的结果
    assert dup_l2_norm_squared([], ZZ) == 0
    assert dup_l2_norm_squared([1], ZZ) == 1
    assert dup_l2_norm_squared([1, 4, 2, 3], ZZ) == 30


# 定义用于测试 dmp_l2_norm_squared 函数的测试函数
def test_dmp_l2_norm_squared():
    # 断言 dmp_l2_norm_squared 返回正确的结果
    assert dmp_l2_norm_squared([[[]]], 2, ZZ) == 0
    assert dmp_l2_norm_squared([[[1]]], 2, ZZ) == 1
    assert dmp_l2_norm_squared(f_0, 2, ZZ) == 111


# 定义用于测试 dup_expand 函数的测试函数
def test_dup_expand():
    # 断言 dup_expand 返回正确的结果
    assert dup_expand((), ZZ) == [1]
    assert dup_expand(([1, 2, 3], [1, 2], [7, 5, 4, 3]), ZZ) == \
        dup_mul([1, 2, 3], dup_mul([1, 2], [7, 5, 4, 3], ZZ), ZZ)
    # 断言：使用 dmp_expand 函数对空元组进行扩展，期望返回 [[1]]
    assert dmp_expand((), 1, ZZ) == [[1]]

    # 断言：使用 dmp_expand 函数对三个元组进行扩展，并与 dmp_mul 的结果进行比较
    assert dmp_expand(([[1], [2], [3]], [[1], [2]], [[7], [5], [4], [3]]), 1, ZZ) == \
        dmp_mul(
            # 对第一个元组 [[1], [2], [3]] 和 dmp_mul 的结果进行乘法运算
            [[1], [2], [3]],
            # 使用 dmp_mul 函数对 [[1], [2]] 和 [[7], [5], [4], [3]] 进行乘法运算
            dmp_mul([[1], [2]], [[7], [5], [4], [3]], 1, ZZ),
            1, ZZ
        )
```