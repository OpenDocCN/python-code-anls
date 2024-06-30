# `D:\src\scipysrc\sympy\sympy\polys\tests\test_densebasic.py`

```
"""Tests for dense recursive polynomials' basic tools. """

# 导入从 densebasic 模块中的具体函数
from sympy.polys.densebasic import (
    ninf,                       # 导入 ninf 常量，表示负无穷
    dup_LC, dmp_LC,             # 导入 dup_LC 和 dmp_LC 函数，用于获取多项式的首项系数
    dup_TC, dmp_TC,             # 导入 dup_TC 和 dmp_TC 函数，用于获取多项式的尾项系数
    dmp_ground_LC, dmp_ground_TC,   # 导入 dmp_ground_LC 和 dmp_ground_TC 函数，获取多项式在给定变量下的首尾系数
    dmp_true_LT,                # 导入 dmp_true_LT 函数，返回多项式的真正首项
    dup_degree, dmp_degree,     # 导入 dup_degree 和 dmp_degree 函数，返回多项式的次数
    dmp_degree_in, dmp_degree_list,  # 导入 dmp_degree_in 和 dmp_degree_list 函数，获取多项式在给定变量下的次数
    dup_strip, dmp_strip,       # 导入 dup_strip 和 dmp_strip 函数，去除多项式的零系数
    dmp_validate,               # 导入 dmp_validate 函数，验证多项式的有效性
    dup_reverse,                # 导入 dup_reverse 函数，颠倒多项式的系数顺序
    dup_copy, dmp_copy,         # 导入 dup_copy 和 dmp_copy 函数，复制多项式
    dup_normal, dmp_normal,     # 导入 dup_normal 和 dmp_normal 函数，规范化多项式的系数
    dup_convert, dmp_convert,   # 导入 dup_convert 和 dmp_convert 函数，将多项式转换为不同表示
    dup_from_sympy, dmp_from_sympy, # 导入 dup_from_sympy 和 dmp_from_sympy 函数，从 SymPy 对象构建多项式
    dup_nth, dmp_nth, dmp_ground_nth,   # 导入 dup_nth、dmp_nth 和 dmp_ground_nth 函数，获取多项式的特定项系数
    dmp_zero_p, dmp_zero,       # 导入 dmp_zero_p 和 dmp_zero 函数，检查和创建零多项式
    dmp_one_p, dmp_one,         # 导入 dmp_one_p 和 dmp_one 函数，检查和创建单位多项式
    dmp_ground_p, dmp_ground,   # 导入 dmp_ground_p 和 dmp_ground 函数，检查和创建常数多项式
    dmp_negative_p, dmp_positive_p,   # 导入 dmp_negative_p 和 dmp_positive_p 函数，检查多项式的符号
    dmp_zeros, dmp_grounds,     # 导入 dmp_zeros 和 dmp_grounds 函数，生成多个零多项式或常数多项式
    dup_from_dict, dup_from_raw_dict,   # 导入 dup_from_dict 和 dup_from_raw_dict 函数，从字典构建多项式
    dup_to_dict, dup_to_raw_dict,       # 导入 dup_to_dict 和 dup_to_raw_dict 函数，将多项式转换为字典
    dmp_from_dict, dmp_to_dict,         # 导入 dmp_from_dict 和 dmp_to_dict 函数，将多项式转换为字典
    dmp_swap, dmp_permute,       # 导入 dmp_swap 和 dmp_permute 函数，交换和排列多项式的变量
    dmp_nest, dmp_raise,         # 导入 dmp_nest 和 dmp_raise 函数，提升和降低多项式的变量次数
    dup_deflate, dmp_deflate,     # 导入 dup_deflate 和 dmp_deflate 函数，收缩多项式的重复根
    dup_multi_deflate, dmp_multi_deflate,   # 导入 dup_multi_deflate 和 dmp_multi_deflate 函数，多次收缩多项式的重复根
    dup_inflate, dmp_inflate,     # 导入 dup_inflate 和 dmp_inflate 函数，膨胀多项式的重复根
    dmp_exclude, dmp_include,     # 导入 dmp_exclude 和 dmp_include 函数，排除和包含特定的多项式项
    dmp_inject, dmp_eject,       # 导入 dmp_inject 和 dmp_eject 函数，插入和弹出多项式的项
    dup_terms_gcd, dmp_terms_gcd,    # 导入 dup_terms_gcd 和 dmp_terms_gcd 函数，计算多项式项的最大公约数
    dmp_list_terms, dmp_apply_pairs,    # 导入 dmp_list_terms 和 dmp_apply_pairs 函数，列出多项式的项和应用函数对
    dup_slice,                  # 导入 dup_slice 函数，切片多项式的系数
    dup_random,                 # 导入 dup_random 函数，生成随机多项式
)

# 导入特定多项式和环
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring

# 导入 SymPy 的单例 S 和测试模块 raises
from sympy.core.singleton import S
from sympy.testing.pytest import raises

# 导入 SymPy 的 oo（无穷大）常量
from sympy.core.numbers import oo

# 从 f_polys 生成多项式 f_0 到 f_6 的稠密表示
f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]

# 定义测试函数 test_dup_LC
def test_dup_LC():
    assert dup_LC([], ZZ) == 0     # 空列表的首项系数应为 0
    assert dup_LC([2, 3, 4, 5], ZZ) == 2   # 列表 [2, 3, 4, 5] 的首项系数应为 2

# 定义测试函数 test_dup_TC
def test_dup_TC():
    assert dup_TC([], ZZ) == 0     # 空列表的尾项系数应为 0
    assert dup_TC([2, 3, 4, 5], ZZ) == 5   # 列表 [2, 3, 4, 5] 的尾项系数应为 5

# 定义测试函数 test_dmp_LC
def test_dmp_LC():
    assert dmp_LC([[]], ZZ) == []      # 二重列表 [[[]]] 在 ZZ 上的首项系数应为空列表
    assert dmp_LC([[2, 3, 4], [5]], ZZ) == [2, 3, 4]   # 二重列表 [[2, 3, 4], [5]] 在 ZZ 上的首项系数应为 [2, 3, 4]
    assert dmp_LC([[[]]], ZZ) == [[]]  # 二重列表 [[[[]]]] 在 ZZ 上的首项系数应为 [[]]
    assert dmp_LC([[[2], [3, 4]], [[5]]], ZZ) == [[2], [3, 4]]   # 三重列表 [[[2], [3, 4]], [[5]]] 在 ZZ 上的首项系数应为 [[2], [3, 4]]

# 定义测试函数 test_dmp_TC
def test_dmp_TC():
    assert dmp_TC([[]], ZZ) == []      # 二重列表 [[[]]] 在 ZZ 上的尾项系数应为空列表
    assert dmp_TC([[2, 3, 4], [5]], ZZ) == [5]   # 二重列表 [[2, 3, 4], [5]] 在 ZZ 上的尾项系数应为 [5]
    assert dmp_TC([[[]]], ZZ) == [[]]  # 二重列表 [[[[]]]] 在 ZZ 上的尾项系数应为 [[]]
    assert dmp_TC([[[2], [3, 4]], [[5]]], ZZ) == [[5]]   # 三重列表 [[[2], [3, 4]], [[5]]] 在 ZZ 上的尾项系数应为 [[5]]

# 定义测试
    # 断言：检查调用 dmp_degree 函数对于给定参数 [[[]]], 2 的返回值是否是负无穷
    assert dmp_degree([[[]]], 2) is ninf
    
    # 断言：检查调用 dmp_degree 函数对于给定参数 [[1]], 1 的返回值是否等于 0
    assert dmp_degree([[1]], 1) == 0
    
    # 断言：检查调用 dmp_degree 函数对于给定参数 [[2], [1]], 1 的返回值是否等于 1
    assert dmp_degree([[2], [1]], 1) == 1
# 测试 dmp_degree_in 函数
def test_dmp_degree_in():
    # 检查 dmp_degree_in([[[]]], 0, 2) 是否返回负无穷
    assert dmp_degree_in([[[]]], 0, 2) is ninf
    # 检查 dmp_degree_in([[[]]], 1, 2) 是否返回负无穷
    assert dmp_degree_in([[[]]], 1, 2) is ninf
    # 检查 dmp_degree_in([[[]]], 2, 2) 是否返回负无穷
    assert dmp_degree_in([[[]]], 2, 2) is ninf

    # 检查 dmp_degree_in([[[1]]], 0, 2) 是否返回 0
    assert dmp_degree_in([[[1]]], 0, 2) == 0
    # 检查 dmp_degree_in([[[1]]], 1, 2) 是否返回 0
    assert dmp_degree_in([[[1]]], 1, 2) == 0
    # 检查 dmp_degree_in([[[1]]], 2, 2) 是否返回 0
    assert dmp_degree_in([[[1]]], 2, 2) == 0

    # 检查 dmp_degree_in(f_4, 0, 2) 是否返回 9
    assert dmp_degree_in(f_4, 0, 2) == 9
    # 检查 dmp_degree_in(f_4, 1, 2) 是否返回 12
    assert dmp_degree_in(f_4, 1, 2) == 12
    # 检查 dmp_degree_in(f_4, 2, 2) 是否返回 8
    assert dmp_degree_in(f_4, 2, 2) == 8

    # 检查 dmp_degree_in(f_6, 0, 2) 是否返回 4
    assert dmp_degree_in(f_6, 0, 2) == 4
    # 检查 dmp_degree_in(f_6, 1, 2) 是否返回 4
    assert dmp_degree_in(f_6, 1, 2) == 4
    # 检查 dmp_degree_in(f_6, 2, 2) 是否返回 6
    assert dmp_degree_in(f_6, 2, 2) == 6
    # 检查 dmp_degree_in(f_6, 3, 3) 是否返回 3
    assert dmp_degree_in(f_6, 3, 3) == 3

    # 检查 lambda 函数是否引发 IndexError 异常
    raises(IndexError, lambda: dmp_degree_in([[1]], -5, 1))


# 测试 dmp_degree_list 函数
def test_dmp_degree_list():
    # 检查 dmp_degree_list([[[[ ]]]], 3) 是否返回四个负无穷
    assert dmp_degree_list([[[[ ]]]], 3) == (-oo, -oo, -oo, -oo)
    # 检查 dmp_degree_list([[[[1]]]], 3) 是否返回四个 0
    assert dmp_degree_list([[[[1]]]], 3) == (0, 0, 0, 0)

    # 检查 dmp_degree_list(f_0, 2) 是否返回 (2, 2, 2)
    assert dmp_degree_list(f_0, 2) == (2, 2, 2)
    # 检查 dmp_degree_list(f_1, 2) 是否返回 (3, 3, 3)
    assert dmp_degree_list(f_1, 2) == (3, 3, 3)
    # 检查 dmp_degree_list(f_2, 2) 是否返回 (5, 3, 3)
    assert dmp_degree_list(f_2, 2) == (5, 3, 3)
    # 检查 dmp_degree_list(f_3, 2) 是否返回 (5, 4, 7)
    assert dmp_degree_list(f_3, 2) == (5, 4, 7)
    # 检查 dmp_degree_list(f_4, 2) 是否返回 (9, 12, 8)
    assert dmp_degree_list(f_4, 2) == (9, 12, 8)
    # 检查 dmp_degree_list(f_5, 2) 是否返回 (3, 3, 3)
    assert dmp_degree_list(f_5, 2) == (3, 3, 3)
    # 检查 dmp_degree_list(f_6, 3) 是否返回 (4, 4, 6, 3)
    assert dmp_degree_list(f_6, 3) == (4, 4, 6, 3)


# 测试 dup_strip 函数
def test_dup_strip():
    # 检查 dup_strip([]) 是否返回空列表
    assert dup_strip([]) == []
    # 检查 dup_strip([0]) 是否返回空列表
    assert dup_strip([0]) == []
    # 检查 dup_strip([0, 0, 0]) 是否返回空列表
    assert dup_strip([0, 0, 0]) == []

    # 检查 dup_strip([1]) 是否返回 [1]
    assert dup_strip([1]) == [1]
    # 检查 dup_strip([0, 1]) 是否返回 [1]
    assert dup_strip([0, 1]) == [1]
    # 检查 dup_strip([0, 0, 0, 1]) 是否返回 [1]
    assert dup_strip([0, 0, 0, 1]) == [1]

    # 检查 dup_strip([1, 2, 0]) 是否返回 [1, 2, 0]
    assert dup_strip([1, 2, 0]) == [1, 2, 0]
    # 检查 dup_strip([0, 1, 2, 0]) 是否返回 [1, 2, 0]
    assert dup_strip([0, 1, 2, 0]) == [1, 2, 0]
    # 检查 dup_strip([0, 0, 0, 1, 2, 0]) 是否返回 [1, 2, 0]
    assert dup_strip([0, 0, 0, 1, 2, 0]) == [1, 2, 0]


# 测试 dmp_strip 函数
def test_dmp_strip():
    # 检查 dmp_strip([0, 1, 0], 0) 是否返回 [1, 0]
    assert dmp_strip([0, 1, 0], 0) == [1, 0]

    # 检查 dmp_strip([[]], 1) 是否返回 [[]]
    assert dmp_strip([[]], 1) == [[]]
    # 检查 dmp_strip([[], []], 1) 是否返回 [[]]
    assert dmp_strip([[], []], 1) == [[]]
    # 检查 dmp_strip([[], [], []], 1) 是否返回 [[]]
    assert dmp_strip([[], [], []], 1) == [[]]

    # 检查 dmp_strip([[[]]], 2) 是否返回 [[[]]]
    assert dmp_strip([[[]]], 2) == [[[]]]
    # 检查 dmp_strip([[[]], [[]]], 2) 是否返回 [[[]]]
    assert dmp_strip([[[]], [[]]], 2) == [[[]]]
    # 检查 dmp_strip([[[]], [[]], [[]]], 2) 是否返回 [[[]]]
    assert dmp_strip([[[]], [[]], [[]]], 2) == [[[]]]

    # 检查 dmp_strip([[[1]]], 2) 是否返回 [[[1]]]
    assert dmp_strip([[[1]]], 2) == [[[1]]]
    # 检查 dmp_strip([[[]], [[1]]], 2) 是否返回 [[[1]]]
    assert dmp_strip([[[]], [[1]]], 2) == [[[1]]]
    # 检查 dmp_strip([[[]], [[1]], [[]]], 2) 是否返回 [[[1]], [[]]]
    assert dmp_strip([[[]], [[1]], [[]]], 2) == [[[1]], [[]]]


# 测试 dmp_validate 函数
def test_dmp_validate():
    # 检查 dmp_validate([]) 是否返回空列表和整数 0
    assert dmp_validate([]) == ([], 0)
    # 检查 dmp_validate([0, 0, 0, 1, 0]) 是否返回 ([1, 0], 0)
    assert dmp_validate([0, 0, 0, 1
def test_dup_convert():
    # 设置环域 K0 和 K1，分别为 ZZ['x'] 和 ZZ
    K0, K1 = ZZ['x'], ZZ

    # 创建多项式列表 f
    f = [K0(1), K0(2), K0(0), K0(3)]

    # 断言调用 dup_convert 函数将 f 从 K0 转换为 K1 后的结果
    assert dup_convert(f, K0, K1) == \
        [ZZ(1), ZZ(2), ZZ(0), ZZ(3)]


def test_dmp_convert():
    # 设置环域 K0 和 K1，分别为 ZZ['x'] 和 ZZ
    K0, K1 = ZZ['x'], ZZ

    # 创建多项式列表 f
    f = [[K0(1)], [K0(2)], [], [K0(3)]]

    # 断言调用 dmp_convert 函数将 f 从 K0 转换为 K1 后的结果
    assert dmp_convert(f, 1, K0, K1) == \
        [[ZZ(1)], [ZZ(2)], [], [ZZ(3)]]


def test_dup_from_sympy():
    # 断言调用 dup_from_sympy 函数将 SymPy 的数值列表转换为 ZZ 类型的列表
    assert dup_from_sympy([S.One, S(2)], ZZ) == \
        [ZZ(1), ZZ(2)]
    assert dup_from_sympy([S.Half, S(3)], QQ) == \
        [QQ(1, 2), QQ(3, 1)]


def test_dmp_from_sympy():
    # 断言调用 dmp_from_sympy 函数将 SymPy 的多项式转换为 ZZ 类型的列表
    assert dmp_from_sympy([[S.One, S(2)], [S.Zero]], 1, ZZ) == \
        [[ZZ(1), ZZ(2)], []]
    assert dmp_from_sympy([[S.Half, S(2)]], 1, QQ) == \
        [[QQ(1, 2), QQ(2, 1)]]


def test_dup_nth():
    # 断言调用 dup_nth 函数从列表中提取第 n 个元素
    assert dup_nth([1, 2, 3], 0, ZZ) == 3
    assert dup_nth([1, 2, 3], 1, ZZ) == 2
    assert dup_nth([1, 2, 3], 2, ZZ) == 1

    # 断言调用 dup_nth 函数当 n 超出列表范围时返回默认值 0
    assert dup_nth([1, 2, 3], 9, ZZ) == 0

    # 断言调用 dup_nth 函数处理负索引时抛出 IndexError 异常
    raises(IndexError, lambda: dup_nth([3, 4, 5], -1, ZZ))


def test_dmp_nth():
    # 断言调用 dmp_nth 函数从多维列表中提取第 n 个元素
    assert dmp_nth([[1], [2], [3]], 0, 1, ZZ) == [3]
    assert dmp_nth([[1], [2], [3]], 1, 1, ZZ) == [2]
    assert dmp_nth([[1], [2], [3]], 2, 1, ZZ) == [1]

    # 断言调用 dmp_nth 函数当 n 超出列表范围时返回空列表
    assert dmp_nth([[1], [2], [3]], 9, 1, ZZ) == []

    # 断言调用 dmp_nth 函数处理负索引时抛出 IndexError 异常
    raises(IndexError, lambda: dmp_nth([[3], [4], [5]], -1, 1, ZZ))


def test_dmp_ground_nth():
    # 断言调用 dmp_ground_nth 函数从多维列表中提取指定位置的元素
    assert dmp_ground_nth([[]], (0, 0), 1, ZZ) == 0
    assert dmp_ground_nth([[1], [2], [3]], (0, 0), 1, ZZ) == 3
    assert dmp_ground_nth([[1], [2], [3]], (1, 0), 1, ZZ) == 2
    assert dmp_ground_nth([[1], [2], [3]], (2, 0), 1, ZZ) == 1

    # 断言调用 dmp_ground_nth 函数当索引超出范围时返回默认值 0
    assert dmp_ground_nth([[1], [2], [3]], (2, 1), 1, ZZ) == 0
    assert dmp_ground_nth([[1], [2], [3]], (3, 0), 1, ZZ) == 0

    # 断言调用 dmp_ground_nth 函数处理负索引时抛出 IndexError 异常
    raises(IndexError, lambda: dmp_ground_nth([[3], [4], [5]], (2, -1), 1, ZZ))


def test_dmp_zero_p():
    # 断言调用 dmp_zero_p 函数判断多项式列表是否为零多项式
    assert dmp_zero_p([], 0) is True
    assert dmp_zero_p([[]], 1) is True

    # 断言调用 dmp_zero_p 函数处理更高维度的情况
    assert dmp_zero_p([[[]]], 2) is True
    assert dmp_zero_p([[[1]]], 2) is False


def test_dmp_zero():
    # 断言调用 dmp_zero 函数生成给定维度的零多项式
    assert dmp_zero(0) == []
    assert dmp_zero(2) == [[[]]]


def test_dmp_one_p():
    # 断言调用 dmp_one_p 函数判断多项式列表是否为单位多项式
    assert dmp_one_p([1], 0, ZZ) is True
    assert dmp_one_p([[1]], 1, ZZ) is True
    assert dmp_one_p([[[1]]], 2, ZZ) is True
    assert dmp_one_p([[[12]]], 2, ZZ) is False


def test_dmp_one():
    # 断言调用 dmp_one 函数生成给定维度的单位多项式
    assert dmp_one(0, ZZ) == [ZZ(1)]
    assert dmp_one(2, ZZ) == [[[ZZ(1)]]]


def test_dmp_ground_p():
    # 断言调用 dmp_ground_p 函数判断多项式列表是否为常数多项式
    assert dmp_ground_p([], 0, 0) is True
    assert dmp_ground_p([[]], 0, 1) is True
    assert dmp_ground_p([[]], 1, 1) is False

    # 断言调用 dmp_ground_p 函数处理更高维度的情况
    assert dmp_ground_p([[ZZ(1)]], 1, 1) is True
    assert dmp_ground_p([[[ZZ(2)]]], 2, 2) is True
    assert dmp_ground_p([[[ZZ(2)]]], 3, 2) is False
    assert dmp_ground_p([[[ZZ(3)], []]], 3, 2) is False

    # 断言调用 dmp_ground_p 函数处理未指定维度的情况
    assert dmp_ground_p([], None, 0) is True
    assert dmp_ground_p([[]], None, 1) is True
    assert dmp_ground_p([ZZ(1)], None, 0) is True
    assert dmp_ground_p([[[ZZ(1)]]], None, 2) is True
    assert dmp_ground_p([[[ZZ(3)], []]], None, 2) is False
    # 确保使用 dmp_ground 函数处理整数 0，生成一个三维列表
    assert dmp_ground(ZZ(0), 2) == [[[]]]
    
    # 确保使用 dmp_ground 函数处理整数 7，当指数为 -1 时返回整数 7
    assert dmp_ground(ZZ(7), -1) == ZZ(7)
    # 确保使用 dmp_ground 函数处理整数 7，当指数为 0 时返回一个包含整数 7 的列表
    assert dmp_ground(ZZ(7), 0) == [ZZ(7)]
    # 确保使用 dmp_ground 函数处理整数 7，当指数为 2 时返回一个三维列表，其中包含整数 7
    assert dmp_ground(ZZ(7), 2) == [[[ZZ(7)]]]
def test_dmp_zeros():
    # 断言：调用 dmp_zeros 函数，期望返回一个包含四个空列表的列表
    assert dmp_zeros(4, 0, ZZ) == [[], [], [], []]

    # 断言：调用 dmp_zeros 函数，期望返回一个空列表
    assert dmp_zeros(0, 2, ZZ) == []
    # 断言：调用 dmp_zeros 函数，期望返回一个包含一个包含一个空列表的列表
    assert dmp_zeros(1, 2, ZZ) == [[[[]]]]
    # 断言：调用 dmp_zeros 函数，期望返回一个包含两个包含一个空列表的列表
    assert dmp_zeros(2, 2, ZZ) == [[[[]]], [[[]]]]
    # 断言：调用 dmp_zeros 函数，期望返回一个包含三个包含一个空列表的列表
    assert dmp_zeros(3, 2, ZZ) == [[[[]]], [[[]]], [[[]]]]

    # 断言：调用 dmp_zeros 函数，期望返回一个包含三个零的列表
    assert dmp_zeros(3, -1, ZZ) == [0, 0, 0]


def test_dmp_grounds():
    # 断言：调用 dmp_grounds 函数，将整数 7 包装为 ZZ，期望返回一个空列表
    assert dmp_grounds(ZZ(7), 0, 2) == []

    # 断言：调用 dmp_grounds 函数，将整数 7 包装为 ZZ，期望返回一个包含一个包含一个包含整数 7 的列表的列表
    assert dmp_grounds(ZZ(7), 1, 2) == [[[[7]]]]
    # 断言：调用 dmp_grounds 函数，将整数 7 包装为 ZZ，期望返回一个包含两个包含一个包含整数 7 的列表的列表
    assert dmp_grounds(ZZ(7), 2, 2) == [[[[7]]], [[[7]]]]
    # 断言：调用 dmp_grounds 函数，将整数 7 包装为 ZZ，期望返回一个包含三个包含一个包含整数 7 的列表的列表
    assert dmp_grounds(ZZ(7), 3, 2) == [[[[7]]], [[[7]]], [[[7]]]]

    # 断言：调用 dmp_grounds 函数，将整数 7 包装为 ZZ，期望返回一个包含三个整数 7 的列表
    assert dmp_grounds(ZZ(7), 3, -1) == [7, 7, 7]


def test_dmp_negative_p():
    # 断言：调用 dmp_negative_p 函数，期望返回 False
    assert dmp_negative_p([[[]]], 2, ZZ) is False
    # 断言：调用 dmp_negative_p 函数，期望返回 False
    assert dmp_negative_p([[[1], [2]]], 2, ZZ) is False
    # 断言：调用 dmp_negative_p 函数，期望返回 True
    assert dmp_negative_p([[[-1], [2]]], 2, ZZ) is True


def test_dmp_positive_p():
    # 断言：调用 dmp_positive_p 函数，期望返回 False
    assert dmp_positive_p([[[]]], 2, ZZ) is False
    # 断言：调用 dmp_positive_p 函数，期望返回 True
    assert dmp_positive_p([[[1], [2]]], 2, ZZ) is True
    # 断言：调用 dmp_positive_p 函数，期望返回 False
    assert dmp_positive_p([[[-1], [2]]], 2, ZZ) is False


def test_dup_from_to_dict():
    # 断言：调用 dup_from_raw_dict 函数，传入一个空字典和 ZZ，期望返回一个空列表
    assert dup_from_raw_dict({}, ZZ) == []
    # 断言：调用 dup_from_dict 函数，传入一个空字典和 ZZ，期望返回一个空列表
    assert dup_from_dict({}, ZZ) == []

    # 断言：调用 dup_to_raw_dict 函数，传入一个空列表，期望返回一个空字典
    assert dup_to_raw_dict([]) == {}
    # 断言：调用 dup_to_dict 函数，传入一个空列表，期望返回一个空字典
    assert dup_to_dict([]) == {}

    # 断言：调用 dup_to_raw_dict 函数，传入一个空列表、ZZ 和 zero=True，期望返回一个包含键值为 0，值为 ZZ(0) 的字典
    assert dup_to_raw_dict([], ZZ, zero=True) == {0: ZZ(0)}
    # 断言：调用 dup_to_dict 函数，传入一个空列表、ZZ 和 zero=True，期望返回一个包含键值为 (0,)，值为 ZZ(0) 的字典
    assert dup_to_dict([], ZZ, zero=True) == {(0,): ZZ(0)}

    # 设置变量 f 和 g 用于后续断言
    f = [3, 0, 0, 2, 0, 0, 0, 0, 8]
    g = {8: 3, 5: 2, 0: 8}
    h = {(8,): 3, (5,): 2, (0,): 8}

    # 断言：调用 dup_from_raw_dict 函数，传入字典 g 和 ZZ，期望返回列表 f
    assert dup_from_raw_dict(g, ZZ) == f
    # 断言：调用 dup_from_dict 函数，传入字典 h 和 ZZ，期望返回列表 f
    assert dup_from_dict(h, ZZ) == f

    # 断言：调用 dup_to_raw_dict 函数，传入列表 f，期望返回字典 g
    assert dup_to_raw_dict(f) == g
    # 断言：调用 dup_to_dict 函数，传入列表 f，期望返回字典 h
    assert dup_to_dict(f) == h

    # 定义环 R 和变量 x, y
    R, x, y = ring("x,y", ZZ)
    K = R.to_domain()

    # 设置变量 f、g 和 h 用于后续断言
    f = [R(3), R(0), R(2), R(0), R(0), R(8)]
    g = {5: R(3), 3: R(2), 0: R(8)}
    h = {(5,): R(3), (3,): R(2), (0,): R(8)}

    # 断言：调用 dup_from_raw_dict 函数，传入字典 g 和 K，期望返回列表 f
    assert dup_from_raw_dict(g, K) == f
    # 断言：调用 dup_from_dict 函数，传入字典 h 和 K，期望返回列表 f
    assert dup_from_dict(h, K) == f

    # 断言：调用 dup_to_raw_dict 函数，传入列表 f，期望返回字典 g
    assert dup_to_raw_dict(f) == g
    # 断言：调用 dup_to_dict 函数，传入列表 f，期望返回字典 h
    assert dup_to_dict(f) == h


def test_dmp_from_to_dict():
    # 断言：调用 dmp_from_dict 函数，传入一个空字典、1 和 ZZ，期望返回一个包含一个空列表的列表
    assert dmp_from_dict({}, 1, ZZ) == [[]]
    # 断言：调用 dmp_to_dict 函数，传入一个包含一个空列表的列表和 1，期望返回一个空字典
    assert dmp_to_dict([[]], 1) == {}

    # 断言：调用 dmp_to_dict 函数，传入一个空列表、0、ZZ 和 zero=True，期望返回一个包含键值为 (0,)，值为 ZZ(0) 的字典
    assert dmp_to_dict([], 0, ZZ, zero=True) == {(0,): ZZ(0)}
    # 断言：调用 dmp_to_dict 函数，传入一个包含一个空列表的列表、1、ZZ 和 zero=True，期望返回一个包含键值为 (0, 0)，值为 ZZ(0) 的字典
    assert dmp_to_dict([[]], 1
    # 对给定的多项式列表进行多次嵌套，返回嵌套后的结果
    assert dmp_nest([[1]], 0, ZZ) == [[1]]
    
    # 对给定的多项式列表进行一次嵌套，返回嵌套后的结果
    assert dmp_nest([[1]], 1, ZZ) == [[[1]]]
    
    # 对给定的多项式列表进行两次嵌套，返回嵌套后的结果
    assert dmp_nest([[1]], 2, ZZ) == [[[[1]]]]
# 定义用于测试 dmp_raise 函数的测试函数
def test_dmp_raise():
    # 断言调用 dmp_raise 函数，传入空列表、2、0 和 ZZ 参数，预期返回 [[[[]]]]
    assert dmp_raise([], 2, 0, ZZ) == [[[]]]
    # 断言调用 dmp_raise 函数，传入包含一个列表 [[1]]、0、1 和 ZZ 参数，预期返回 [[1]]
    assert dmp_raise([[1]], 0, 1, ZZ) == [[1]]

    # 断言调用 dmp_raise 函数，传入包含三个列表 [[1, 2, 3], [], [2, 3]]、2、1 和 ZZ 参数，预期返回如下嵌套列表：
    # [
    #     [[[1]], [[2]], [[3]]],
    #     [[[]]],
    #     [[[2]], [[3]]]
    # ]
    assert dmp_raise([[1, 2, 3], [], [2, 3]], 2, 1, ZZ) == \
        [[[[1]], [[2]], [[3]]], [[[]]], [[[2]], [[3]]]]


# 定义用于测试 dup_deflate 函数的测试函数
def test_dup_deflate():
    # 断言调用 dup_deflate 函数，传入空列表和 ZZ 参数，预期返回元组 (1, [])
    assert dup_deflate([], ZZ) == (1, [])
    # 断言调用 dup_deflate 函数，传入包含一个元素的列表 [2] 和 ZZ 参数，预期返回元组 (1, [2])
    assert dup_deflate([2], ZZ) == (1, [2])
    # 断言调用 dup_deflate 函数，传入包含三个元素的列表 [1, 2, 3] 和 ZZ 参数，预期返回元组 (1, [1, 2, 3])
    assert dup_deflate([1, 2, 3], ZZ) == (1, [1, 2, 3])
    # 断言调用 dup_deflate 函数，传入包含五个元素的列表 [1, 0, 2, 0, 3] 和 ZZ 参数，预期返回元组 (2, [1, 2, 3])
    assert dup_deflate([1, 0, 2, 0, 3], ZZ) == (2, [1, 2, 3])

    # 断言调用 dup_deflate 函数，传入调用 dup_from_raw_dict 生成的元组 (1, [1, 0, 0, 0, 0, 0, 1, 0]) 和 ZZ 参数，预期返回元组 (1, [1, 0, 0, 0, 0, 0, 1, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1, 1: 1}, ZZ), ZZ) == \
        (1, [1, 0, 0, 0, 0, 0, 1, 0])
    # 断言调用 dup_deflate 函数，传入调用 dup_from_raw_dict 生成的元组 (1, [1, 1]) 和 ZZ 参数，预期返回元组 (7, [1, 1])
    assert dup_deflate(dup_from_raw_dict({7: 1, 0: 1}, ZZ), ZZ) == \
        (7, [1, 1])
    # 断言调用 dup_deflate 函数，传入调用 dup_from_raw_dict 生成的元组 (1, [1, 0, 0, 0, 1, 0, 0, 0]) 和 ZZ 参数，预期返回元组 (1, [1, 0, 0, 0, 1, 0, 0, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1, 3: 1}, ZZ), ZZ) == \
        (1, [1, 0, 0, 0, 1, 0, 0, 0])

    # 断言调用 dup_deflate 函数，传入调用 dup_from_raw_dict 生成的元组 (1, [1, 0, 0, 1, 0, 0, 0, 0]) 和 ZZ 参数，预期返回元组 (1, [1, 0, 0, 1, 0, 0, 0, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1, 4: 1}, ZZ), ZZ) == \
        (1, [1, 0, 0, 1, 0, 0, 0, 0])
    # 断言调用 dup_deflate 函数，传入调用 dup_from_raw_dict 生成的元组 (4, [1, 1, 0]) 和 ZZ 参数，预期返回元组 (4, [1, 1, 0])
    assert dup_deflate(dup_from_raw_dict({8: 1, 4: 1}, ZZ), ZZ) == \
        (4, [1, 1, 0])

    # 断言调用 dup_deflate 函数，传入调用 dup_from_raw_dict 生成的元组 (8, [1, 0]) 和 ZZ 参数，预期返回元组 (8, [1, 0])
    assert dup_deflate(dup_from_raw_dict({8: 1}, ZZ), ZZ) == \
        (8, [1, 0])
    # 断言调用 dup_deflate 函数，传入调用 dup_from_raw_dict 生成的元组 (7, [1, 0]) 和 ZZ 参数，预期返回元组 (7, [1, 0])
    assert dup_deflate(dup_from_raw_dict({7: 1}, ZZ), ZZ) == \
        (7, [1, 0])
    # 断言调用 dup_deflate 函数，传入调用 dup_from_raw_dict 生成的元组 (1, [1, 0]) 和 ZZ 参数，预期返回元组 (1, [1, 0])
    assert dup_deflate(dup_from_raw_dict({1: 1}, ZZ), ZZ) == \
        (1, [1, 0])


# 定义用于测试 dmp_deflate 函数的测试函数
def test_dmp_deflate():
    # 断言调用 dmp_deflate 函数，传入包含一个空列表 [[]]、1 和 ZZ 参数，预期返回元组 ((1, 1), [[]])
    assert dmp_deflate([[]], 1, ZZ) == ((1, 1), [[]])
    # 断言调用 dmp_deflate 函数，传入包含一个包含一个元素的列表 [[2]]、1 和 ZZ 参数，预期返回元组 ((1, 1), [[2]])
    assert dmp_deflate([[2]], 1, ZZ) == ((1, 1), [[2]])

    # 定义包含复杂结构的列表 f
    f = [[1, 0, 0], [], [1, 0], [], [1]]

    # 断言调用 dmp_deflate 函数，传入列表 f、1 和 ZZ 参数，预期返回元组 ((2, 1), [[1, 0, 0], [1, 0], [1]])
    assert dmp_deflate(f, 1, ZZ) == ((2, 1), [[1, 0, 0], [1, 0], [1]])


# 定义用于测试 dup_multi_deflate 函数的测试函数
def test_dup_multi_deflate():
    # 断言调用 dup_multi_deflate 函数，传入一个包含一个元组 ([2],) 和 ZZ 参数，预期返回元组 (1, ([2],))
    assert dup_multi_deflate(([2],), ZZ) == (1, ([2],))
    # 断言调用 dup_multi_deflate 函数，传入一个包含两个空列表的元组 ([], []) 和 ZZ 参数，预期返回元组 (1, ([], []))
    assert dup_multi_deflate(([], []), ZZ) == (1, ([], []))

    # 断言调用 dup_multi_deflate 函数，传入一个包含一个列表 [1, 2, 3] 的元组 ([1, 2, 3],) 和 ZZ 参数，预期返回元组 (1, ([1, 2, 3],))
    assert dup_multi_deflate(([1, 2, 3],), ZZ) == (1, ([1, 2, 3],))
    # 断言调用 dup_multi_deflate 函数，传入一个包含五个元素的列表 [1, 0, 2, 0, 3] 的元组 ([1, 0, 2,
    # 断言：使用 dmp_multi_deflate 函数对元组 (f,) 进行操作，期望返回结果为 ((2, 1), ([[1, 0, 0], [1, 0], [1]],))
    assert dmp_multi_deflate((f,), 1, ZZ) == \
        ((2, 1), ([[1, 0, 0], [1, 0], [1]],))
    
    # 断言：使用 dmp_multi_deflate 函数对元组 (f, g) 进行操作，期望返回结果为 ((2, 1), ([[1, 0, 0], [1, 0], [1]], [[1, 0, 1, 0], [1]]))
    assert dmp_multi_deflate((f, g), 1, ZZ) == \
        ((2, 1), ([[1, 0, 0], [1, 0], [1]],
                  [[1, 0, 1, 0], [1]]))
# 定义函数 `test_dup_inflate()`，用于测试 `dup_inflate` 函数
def test_dup_inflate():
    # 断言空列表经 `dup_inflate` 处理后返回空列表
    assert dup_inflate([], 17, ZZ) == []

    # 断言对包含 `[1, 2, 3]` 的列表调用 `dup_inflate`，不复制元素，返回原列表
    assert dup_inflate([1, 2, 3], 1, ZZ) == [1, 2, 3]
    # 断言对包含 `[1, 2, 3]` 的列表调用 `dup_inflate`，每个元素复制一次，返回 `[1, 0, 2, 0, 3]`
    assert dup_inflate([1, 2, 3], 2, ZZ) == [1, 0, 2, 0, 3]
    # 断言对包含 `[1, 2, 3]` 的列表调用 `dup_inflate`，每个元素复制两次，返回 `[1, 0, 0, 2, 0, 0, 3]`
    assert dup_inflate([1, 2, 3], 3, ZZ) == [1, 0, 0, 2, 0, 0, 3]
    # 断言对包含 `[1, 2, 3]` 的列表调用 `dup_inflate`，每个元素复制三次，返回 `[1, 0, 0, 0, 2, 0, 0, 0, 3]`
    assert dup_inflate([1, 2, 3], 4, ZZ) == [1, 0, 0, 0, 2, 0, 0, 0, 3]

    # 断言调用 `dup_inflate` 时，复制次数为零会抛出 IndexError 异常
    raises(IndexError, lambda: dup_inflate([1, 2, 3], 0, ZZ))


# 定义函数 `test_dmp_inflate()`，用于测试 `dmp_inflate` 函数
def test_dmp_inflate():
    # 断言对包含单个元素 `[1]` 的列表调用 `dmp_inflate`，不复制元素，返回原列表
    assert dmp_inflate([1], (3,), 0, ZZ) == [1]

    # 断言对包含空列表 `[[[]]]` 的列表调用 `dmp_inflate`，不复制元素，返回原列表
    assert dmp_inflate([[]], (3, 7), 1, ZZ) == [[]]
    # 断言对包含 `[[2]]` 的列表调用 `dmp_inflate`，每个子列表复制一次，返回 `[[2]]`
    assert dmp_inflate([[2]], (1, 2), 1, ZZ) == [[2]]

    # 断言对包含 `[[2, 0]]` 的列表调用 `dmp_inflate`，每个子列表第一个元素复制一次，返回 `[[2, 0]]`
    assert dmp_inflate([[2, 0]], (1, 1), 1, ZZ) == [[2, 0]]
    # 断言对包含 `[[2, 0]]` 的列表调用 `dmp_inflate`，每个子列表第一个元素复制两次，返回 `[[2, 0, 0]]`
    assert dmp_inflate([[2, 0]], (1, 2), 1, ZZ) == [[2, 0, 0]]
    # 断言对包含 `[[2, 0]]` 的列表调用 `dmp_inflate`，每个子列表第一个元素复制三次，返回 `[[2, 0, 0, 0]]`
    assert dmp_inflate([[2, 0]], (1, 3), 1, ZZ) == [[2, 0, 0, 0]]

    # 断言对包含 `[[1, 0, 0], [1], [1, 0]]` 的列表调用 `dmp_inflate`，每个子列表第一个元素复制两次，返回 `[[1, 0, 0], [], [1], [], [1, 0]]`
    assert dmp_inflate([[1, 0, 0], [1], [1, 0]], (2, 1), 1, ZZ) == [[1, 0, 0], [], [1], [], [1, 0]]

    # 断言调用 `dmp_inflate` 时，复制次数为负数会抛出 IndexError 异常
    raises(IndexError, lambda: dmp_inflate([[]], (-3, 7), 1, ZZ))


# 定义函数 `test_dmp_exclude()`，用于测试 `dmp_exclude` 函数
def test_dmp_exclude():
    # 断言对包含 `[[[]]]` 的列表调用 `dmp_exclude`，排除深度为 2 的元素，返回空列表和原始列表及排除深度
    assert dmp_exclude([[[]]], 2, ZZ) == ([], [[[]]], 2)
    # 断言对包含 `[[[7]]]` 的列表调用 `dmp_exclude`，排除深度为 2 的元素，返回空列表和原始列表及排除深度
    assert dmp_exclude([[[7]]], 2, ZZ) == ([], [[[7]]], 2)

    # 断言对包含 `[1, 2, 3]` 的列表调用 `dmp_exclude`，排除深度为 0 的元素，返回空列表和原始列表及排除深度
    assert dmp_exclude([1, 2, 3], 0, ZZ) == ([], [1, 2, 3], 0)
    # 断言对包含 `[[1], [2, 3]]` 的列表调用 `dmp_exclude`，排除深度为 1 的元素，返回空列表和原始列表及排除深度
    assert dmp_exclude([[1], [2, 3]], 1, ZZ) == ([], [[1], [2, 3]], 1)

    # 断言对包含 `[[1, 2, 3]]` 的列表调用 `dmp_exclude`，排除深度为 1 的元素，返回 `[0]`、原始列表和排除深度
    assert dmp_exclude([[1, 2, 3]], 1, ZZ) == ([0], [1, 2, 3], 0)
    # 断言对包含 `[[1], [2], [3]]` 的列表调用 `dmp_exclude`，排除深度为 1 的元素，返回 `[1]`、原始列表和排除深度
    assert dmp_exclude([[1], [2], [3]], 1, ZZ) == ([1], [1, 2, 3], 0)

    # 断言对包含 `[[[1, 2, 3]]]` 的列表调用 `dmp_exclude`，排除深度为 2 的元素，返回 `[0, 1]`、原始列表和排除深度
    assert dmp_exclude([[[1, 2, 3]]], 2, ZZ) == ([0, 1], [1, 2, 3], 0)
    # 断言对包含 `[[[1]], [[2]], [[3]]]` 的列表调用 `dmp_exclude`，排除深度为 2 的元素，返回 `[1, 2]`、原始列表和排除深度
    assert dmp_exclude([[[1]], [[2]], [[3]]], 2, ZZ) == ([1, 2], [1, 2, 3], 0)


# 定义函数 `test_dmp_include()`，用于测试 `dmp_include` 函数
def test_dmp_include():
    # 断言对包含 `[1, 2, 3]` 的列表调用 `dmp_include`，包含空索引列表，返回原列表
    assert dmp_include([1, 2, 3], [], 0, ZZ) == [1, 2, 3]

    # 断言对包含 `[1, 2, 3]` 的列表调用 `dmp_include`，包含 `[0]` 的索引列表，返回 `[[1, 2, 3]]`
    assert dmp_include([1, 2,
    # 使用 assert 语句来测试 dup_terms_gcd 函数的返回值是否符合预期
    assert dup_terms_gcd([1, 0, 1], ZZ) == (0, [1, 0, 1])
    # 再次使用 assert 语句测试 dup_terms_gcd 函数的返回值是否符合预期
    assert dup_terms_gcd([1, 0, 1, 0], ZZ) == (1, [1, 0, 1])
# 测试 dmp_terms_gcd 函数
def test_dmp_terms_gcd():
    # 检查空列表情况下的最大公因式计算，预期返回 ((0, 0), [[]])
    assert dmp_terms_gcd([[]], 1, ZZ) == ((0, 0), [[]])

    # 检查多项式 [1, 0, 1, 0] 的最大公因式计算，预期返回 ((1,), [1, 0, 1])
    assert dmp_terms_gcd([1, 0, 1, 0], 0, ZZ) == ((1,), [1, 0, 1])
    
    # 检查二维多项式 [[1], [], [1], []] 的最大公因式计算，预期返回 ((1, 0), [[1], [], [1]])
    assert dmp_terms_gcd([[1], [], [1], []], 1, ZZ) == ((1, 0), [[1], [], [1]])

    # 检查二维多项式 [[1, 0], [], [1]] 的最大公因式计算，预期返回 ((0, 0), [[1, 0], [], [1]])
    assert dmp_terms_gcd([[1, 0], [], [1]], 1, ZZ) == ((0, 0), [[1, 0], [], [1]])

    # 检查二维多项式 [[1, 0], [1, 0, 0], [], []] 的最大公因式计算，预期返回 ((2, 1), [[1], [1, 0]])
    assert dmp_terms_gcd([[1, 0], [1, 0, 0], [], []], 1, ZZ) == ((2, 1), [[1], [1, 0]])


# 测试 dmp_list_terms 函数
def test_dmp_list_terms():
    # 检查三维多项式 [[[[]]]] 的项列表生成，预期返回 [((0, 0, 0), 0)]
    assert dmp_list_terms([[[]]], 2, ZZ) == [((0, 0, 0), 0)]
    
    # 检查三维多项式 [[[1]]] 的项列表生成，预期返回 [((0, 0, 0), 1)]
    assert dmp_list_terms([[[1]]], 2, ZZ) == [((0, 0, 0), 1)]

    # 检查多项式 [1, 2, 4, 3, 5] 的项列表生成（一维），预期返回 [((4,), 1), ((3,), 2), ((2,), 4), ((1,), 3), ((0,), 5)]
    assert dmp_list_terms([1, 2, 4, 3, 5], 0, ZZ) == [((4,), 1), ((3,), 2), ((2,), 4), ((1,), 3), ((0,), 5)]

    # 检查二维多项式 [[1], [2, 4], [3, 5, 0]] 的项列表生成，预期返回 [((2, 0), 1), ((1, 1), 2), ((1, 0), 4), ((0, 2), 3), ((0, 1), 5)]
    assert dmp_list_terms([[1], [2, 4], [3, 5, 0]], 1, ZZ) == [((2, 0), 1), ((1, 1), 2), ((1, 0), 4), ((0, 2), 3), ((0, 1), 5)]

    # 创建多项式 f
    f = [[2, 0, 0, 0], [1, 0, 0], []]

    # 检查多项式 f 在字典序下的项列表生成，预期返回 [((2, 3), 2), ((1, 2), 1)]
    assert dmp_list_terms(f, 1, ZZ, order='lex') == [((2, 3), 2), ((1, 2), 1)]

    # 检查多项式 f 在逆字典序下的项列表生成，预期返回 [((2, 3), 2), ((1, 2), 1)]
    assert dmp_list_terms(f, 1, ZZ, order='grlex') == [((2, 3), 2), ((1, 2), 1)]

    # 更新多项式 f
    f = [[2, 0, 0, 0], [1, 0, 0, 0, 0, 0], []]

    # 检查多项式 f 在字典序下的项列表生成，预期返回 [((2, 3), 2), ((1, 5), 1)]
    assert dmp_list_terms(f, 1, ZZ, order='lex') == [((2, 3), 2), ((1, 5), 1)]

    # 检查多项式 f 在逆字典序下的项列表生成，预期返回 [((1, 5), 1), ((2, 3), 2)]
    assert dmp_list_terms(f, 1, ZZ, order='grlex') == [((1, 5), 1), ((2, 3), 2)]


# 测试 dmp_apply_pairs 函数
def test_dmp_apply_pairs():
    # 定义二元操作函数 h
    h = lambda a, b: a*b

    # 检查多项式 [1, 2, 3] 和 [4, 5, 6] 的对应元素操作，预期返回 [4, 10, 18]
    assert dmp_apply_pairs([1, 2, 3], [4, 5, 6], h, [], 0, ZZ) == [4, 10, 18]

    # 检查多项式 [2, 3] 和 [4, 5, 6] 的对应元素操作，预期返回 [10, 18]
    assert dmp_apply_pairs([2, 3], [4, 5, 6], h, [], 0, ZZ) == [10, 18]

    # 检查二维多项式 [[1, 2], [3]] 和 [[4, 5], [6]] 的对应元素操作，预期返回 [[4, 10], [18]]
    assert dmp_apply_pairs([[1, 2], [3]], [[4, 5], [6]], h, [], 1, ZZ) == [[4, 10], [18]]

    # 检查二维多项式 [[1, 2], [3]] 和 [[4], [5, 6]] 的对应元素操作，预期返回 [[8], [18]]
    assert dmp_apply_pairs([[1, 2], [3]], [[4], [5, 6]], h, [], 1, ZZ) == [[8], [18]]

    # 检查二维多项式 [[1], [2, 3]] 和 [[4, 5], [6]] 的对应元素操作，预期返回 [[5], [18]]
    assert dmp_apply_pairs([[1], [2, 3]], [[4, 5], [6]], h, [], 1, ZZ) == [[5], [18]]


# 测试 dup_slice 函数
def test_dup_slice():
    # 定义多项式 f
    f = [1, 2, 3, 4]

    # 检查多项式 f 在索引范围 [0, 0) 的切片操作，预期返回空列表 []
    assert dup_slice(f, 0, 0, ZZ) == []

    # 检查多项式 f 在索引范围 [0, 1) 的切片操作，预期返回 [4]
    assert dup_slice(f, 0, 1, ZZ) == [4]

    # 检查多项式 f 在索引范围 [0, 2) 的切片操作，预期返回 [3, 4]
    assert dup_slice(f, 0, 2, ZZ) == [3, 4]

    # 检查多项式 f 在索引范围 [0, 3) 的切片操作，预期返回 [2, 3, 4]
    assert dup_slice(f, 0, 3, ZZ) == [2, 3, 4]

    # 检查多项式 f 在索引范围 [0, 4) 的切片操作，预期返回 [1, 2
```