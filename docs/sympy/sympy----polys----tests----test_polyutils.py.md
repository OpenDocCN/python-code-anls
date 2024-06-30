# `D:\src\scipysrc\sympy\sympy\polys\tests\test_polyutils.py`

```
"""Tests for useful utilities for higher level polynomial classes. """

# 导入必要的模块和函数
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.testing.pytest import raises

# 导入多项式相关的工具函数和类
from sympy.polys.polyutils import (
    _nsort,
    _sort_gens,
    _unify_gens,
    _analyze_gens,
    _sort_factors,
    parallel_dict_from_expr,
    dict_from_expr,
)

# 导入多项式相关的异常类
from sympy.polys.polyerrors import PolynomialError

# 导入整数环 ZZ
from sympy.polys.domains import ZZ

# 定义多个符号变量
x, y, z, p, q, r, s, t, u, v, w = symbols('x,y,z,p,q,r,s,t,u,v,w')
A, B = symbols('A,B', commutative=False)


def test__nsort():
    # issue 6137
    # 创建一个复杂的表达式 r
    r = S('''[3/2 + sqrt(-14/3 - 2*(-415/216 + 13*I/12)**(1/3) - 4/sqrt(-7/3 +
    61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 + 13*I/12)**(1/3)) -
    61/(18*(-415/216 + 13*I/12)**(1/3)))/2 - sqrt(-7/3 + 61/(18*(-415/216
    + 13*I/12)**(1/3)) + 2*(-415/216 + 13*I/12)**(1/3))/2, 3/2 - sqrt(-7/3
    + 61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 +
    13*I/12)**(1/3))/2 - sqrt(-14/3 - 2*(-415/216 + 13*I/12)**(1/3) -
    4/sqrt(-7/3 + 61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 +
    13*I/12)**(1/3)) - 61/(18*(-415/216 + 13*I/12)**(1/3)))/2, 3/2 +
    sqrt(-14/3 - 2*(-415/216 + 13*I/12)**(1/3) + 4/sqrt(-7/3 +
    61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 + 13*I/12)**(1/3)) -
    61/(18*(-415/216 + 13*I/12)**(1/3)))/2 + sqrt(-7/3 + 61/(18*(-415/216
    + 13*I/12)**(1/3)) + 2*(-415/216 + 13*I/12)**(1/3))/2, 3/2 + sqrt(-7/3
    + 61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 +
    13*I/12)**(1/3))/2 - sqrt(-14/3 - 2*(-415/216 + 13*I/12)**(1/3) +
    4/sqrt(-7/3 + 61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 +
    13*I/12)**(1/3)) - 61/(18*(-415/216 + 13*I/12)**(1/3)))/2]''')
    # 创建预期的排序后的结果 ans
    ans = [r[1], r[0], r[-1], r[-2]]
    # 验证 _nsort 函数对 r 的排序结果与 ans 相等
    assert _nsort(r) == ans
    # 验证 _nsort 函数在使用 separated=True 参数时返回空列表
    assert len(_nsort(r, separated=True)[0]) == 0
    # issue 12560
    # 创建一个余弦平方加正弦平方等于一的表达式 a
    a = cos(1)**2 + sin(1)**2 - 1
    # 验证 _nsort 函数对包含 a 的列表排序后结果与 [a] 相等
    assert _nsort([a]) == [a]


def test__sort_gens():
    # 验证空列表的排序结果为空元组
    assert _sort_gens([]) == ()

    # 验证单个变量 x 的排序结果为 (x,)
    assert _sort_gens([x]) == (x,)
    # 验证单个变量 p 的排序结果为 (p,)
    assert _sort_gens([p]) == (p,)
    # 验证单个变量 q 的排序结果为 (q,)
    assert _sort_gens([q]) == (q,)

    # 验证包含两个变量 x 和 p 的排序结果为 (x, p) 和 (p, x)
    assert _sort_gens([x, p]) == (x, p)
    assert _sort_gens([p, x]) == (x, p)
    # 验证包含两个变量 q 和 p 的排序结果为 (p, q)
    assert _sort_gens([q, p]) == (p, q)

    # 验证包含三个变量 x, p 和 q 的排序结果为 (x, p, q)
    assert _sort_gens([q, p, x]) == (x, p, q)

    # 验证使用 wrt 参数进行排序，以及参数类型为变量名字符串时的情况
    assert _sort_gens([x, p, q], wrt=x) == (x, p, q)
    assert _sort_gens([x, p, q], wrt=p) == (p, x, q)
    assert _sort_gens([x, p, q], wrt=q) == (q, x, p)

    assert _sort_gens([x, p, q], wrt='x') == (x, p, q)
    assert _sort_gens([x, p, q], wrt='p') == (p, x, q)


这段代码中，每行都按照要求添加了详细的注释，解释了代码的功能和作用。
    # 验证函数 _sort_gens 的行为是否符合预期：对于给定的生成器列表和选项，返回按照指定顺序排序后的结果是否正确
    assert _sort_gens([x, p, q], wrt='q') == (q, x, p)
    
    # 验证 _sort_gens 函数在多个变量按不同顺序排序时的行为是否正确
    assert _sort_gens([x, p, q], wrt='x,q') == (x, q, p)
    assert _sort_gens([x, p, q], wrt='q,x') == (q, x, p)
    assert _sort_gens([x, p, q], wrt='p,q') == (p, q, x)
    assert _sort_gens([x, p, q], wrt='q,p') == (q, p, x)
    
    # 验证 _sort_gens 函数对带有空格的排序选项的处理是否正确
    assert _sort_gens([x, p, q], wrt='x, q') == (x, q, p)
    assert _sort_gens([x, p, q], wrt='q, x') == (q, x, p)
    assert _sort_gens([x, p, q], wrt='p, q') == (p, q, x)
    assert _sort_gens([x, p, q], wrt='q, p') == (q, p, x)
    
    # 验证 _sort_gens 函数对带有列表形式的排序选项的处理是否正确
    assert _sort_gens([x, p, q], wrt=[x, 'q']) == (x, q, p)
    assert _sort_gens([x, p, q], wrt=[q, 'x']) == (q, x, p)
    assert _sort_gens([x, p, q], wrt=[p, 'q']) == (p, q, x)
    assert _sort_gens([x, p, q], wrt=[q, 'p']) == (q, p, x)
    
    # 验证 _sort_gens 函数对带有字符串形式的排序选项的处理是否正确
    assert _sort_gens([x, p, q], wrt=['x', 'q']) == (x, q, p)
    assert _sort_gens([x, p, q], wrt=['q', 'x']) == (q, x, p)
    assert _sort_gens([x, p, q], wrt=['p', 'q']) == (p, q, x)
    assert _sort_gens([x, p, q], wrt=['q', 'p']) == (q, p, x)
    
    # 验证 _sort_gens 函数对使用字符串表达式进行排序的处理是否正确
    assert _sort_gens([x, p, q], sort='x > p > q') == (x, p, q)
    assert _sort_gens([x, p, q], sort='p > x > q') == (p, x, q)
    assert _sort_gens([x, p, q], sort='p > q > x') == (p, q, x)
    
    # 验证 _sort_gens 函数在指定变量排序的同时使用排序字符串表达式的处理是否正确
    assert _sort_gens([x, p, q], wrt='x', sort='q > p') == (x, q, p)
    assert _sort_gens([x, p, q], wrt='p', sort='q > x') == (p, q, x)
    assert _sort_gens([x, p, q], wrt='q', sort='p > x') == (q, p, x)
    
    # 验证 _sort_gens 函数在处理特殊符号时的行为是否正确
    # https://github.com/sympy/sympy/issues/19353
    n1 = Symbol('\n1')
    assert _sort_gens([n1]) == (n1,)
    assert _sort_gens([x, n1]) == (x, n1)
    
    # 验证 _sort_gens 函数对符号列表的排序是否正确
    X = symbols('x0,x1,x2,x10,x11,x12,x20,x21,x22')
    assert _sort_gens(X) == X
# 定义一个测试函数，用于测试 _unify_gens 函数的不同情况
def test__unify_gens():
    # 空列表的情况下，期望返回空元组
    assert _unify_gens([], []) == ()

    # 单个元素列表相等的情况下，返回包含该元素的元组
    assert _unify_gens([x], [x]) == (x,)
    assert _unify_gens([y], [y]) == (y,)

    # 元素不同的列表，其中一个包含另一个的情况下，返回合并后的元组
    assert _unify_gens([x, y], [x]) == (x, y)
    assert _unify_gens([x], [x, y]) == (x, y)

    # 两个列表完全相同的情况下，返回相同的元组
    assert _unify_gens([x, y], [x, y]) == (x, y)
    assert _unify_gens([y, x], [y, x]) == (y, x)

    # 单个元素不同的列表，返回合并后的元组
    assert _unify_gens([x], [y]) == (x, y)
    assert _unify_gens([y], [x]) == (y, x)

    # 一个包含两个元素，另一个包含三个元素的列表，返回合并后的元组
    assert _unify_gens([x], [y, x]) == (y, x)
    assert _unify_gens([y, x], [x]) == (y, x)

    # 元素顺序不同的情况下，返回按顺序合并后的元组
    assert _unify_gens([x, y, z], [x, y, z]) == (x, y, z)
    assert _unify_gens([z, y, x], [x, y, z]) == (z, y, x)
    assert _unify_gens([x, y, z], [z, y, x]) == (x, y, z)
    assert _unify_gens([z, y, x], [z, y, x]) == (z, y, x)

    # 多个元素不同的列表，返回按顺序合并后的元组
    assert _unify_gens([x, y, z], [t, x, p, q, z]) == (t, x, y, p, q, z)


# 定义测试函数，用于测试 _analyze_gens 函数
def test__analyze_gens():
    # 测试传入元组和列表的情况，期望返回相同的元组
    assert _analyze_gens((x, y, z)) == (x, y, z)
    assert _analyze_gens([x, y, z]) == (x, y, z)

    # 测试嵌套的元组的情况，期望返回相同的元组
    assert _analyze_gens(([x, y, z],)) == (x, y, z)
    assert _analyze_gens(((x, y, z),)) == (x, y, z)


# 定义测试函数，用于测试 _sort_factors 函数
def test__sort_factors():
    # 空列表的情况下，无论是否多个，都期望返回空列表
    assert _sort_factors([], multiple=True) == []
    assert _sort_factors([], multiple=False) == []

    # 测试按列表中子列表长度排序的情况
    F = [[1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [1, 2, 3]]
    assert _sort_factors(F, multiple=False) == G

    F = [[1, 2], [1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [1, 2], [1, 2, 3]]
    assert _sort_factors(F, multiple=False) == G

    F = [[2, 2], [1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [2, 2], [1, 2, 3]]
    assert _sort_factors(F, multiple=False) == G

    # 测试带有元组的列表的情况，期望按照元组第一个元素排序
    F = [([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([1, 2, 3], 1)]
    assert _sort_factors(F, multiple=True) == G

    F = [([1, 2], 1), ([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([1, 2], 1), ([1, 2, 3], 1)]
    assert _sort_factors(F, multiple=True) == G

    F = [([2, 2], 1), ([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([2, 2], 1), ([1, 2, 3], 1)]
    assert _sort_factors(F, multiple=True) == G

    F = [([2, 2], 1), ([1, 2, 3], 1), ([1, 2], 2), ([1], 1)]
    G = [([1], 1), ([2, 2], 1), ([1, 2], 2), ([1, 2, 3], 1)]
    assert _sort_factors(F, multiple=True) == G


# 定义测试函数，用于测试 dict_from_expr_if_gens 函数
def test__dict_from_expr_if_gens():
    # 测试传入整数表达式和变量列表的情况，期望返回对应的字典和变量元组
    assert dict_from_expr(
        Integer(17), gens=(x,)) == ({(0,): Integer(17)}, (x,))
    assert dict_from_expr(
        Integer(17), gens=(x, y)) == ({(0, 0): Integer(17)}, (x, y))
    assert dict_from_expr(
        Integer(17), gens=(x, y, z)) == ({(0, 0, 0): Integer(17)}, (x, y, z))

    # 测试传入负整数表达式和变量列表的情况，期望返回对应的字典和变量元组
    assert dict_from_expr(
        Integer(-17), gens=(x,)) == ({(0,): Integer(-17)}, (x,))
    assert dict_from_expr(
        Integer(-17), gens=(x, y)) == ({(0, 0): Integer(-17)}, (x, y))
    assert dict_from_expr(
        Integer(-17), gens=(x, y, z)) == ({(0, 0, 0): Integer(-17)}, (x, y, z))

    # 测试传入带有变量的整数表达式和变量列表的情况，期望返回对应的字典和变量元组
    assert dict_from_expr(
        Integer(17)*x, gens=(x,)) == ({(1,): Integer(17)}, (x,))


这段代码包含了多个测试函数，用于验证不同函数在各种输入情况下的行为。每个函数的测试包含了多个断言来覆盖不同的输入和预期输出情况。
    # 断言：根据表达式生成字典，指定变量集合为 (x, y)，结果应为 {(1, 0): Integer(17)}，变量集合为 (x, y)
    assert dict_from_expr(Integer(17)*x, gens=(x, y)) == ({(1, 0): Integer(17)}, (x, y))
    # 断言：根据表达式生成字典，指定变量集合为 (x, y, z)，结果应为 {(1, 0, 0): Integer(17)}，变量集合为 (x, y, z)
    assert dict_from_expr(Integer(17)*x, gens=(x, y, z)) == ({(1, 0, 0): Integer(17)}, (x, y, z))

    # 断言：根据表达式生成字典，指定变量集合为 (x,)，结果应为 {(7,): Integer(17)}，变量集合为 (x,)
    assert dict_from_expr(Integer(17)*x**7, gens=(x,)) == ({(7,): Integer(17)}, (x,))
    # 断言：根据表达式生成字典，指定变量集合为 (x, y)，结果应为 {(7, 1): Integer(17)}，变量集合为 (x, y)
    assert dict_from_expr(Integer(17)*x**7*y, gens=(x, y)) == ({(7, 1): Integer(17)}, (x, y))
    # 断言：根据表达式生成字典，指定变量集合为 (x, y, z)，结果应为 {(7, 1, 12): Integer(17)}，变量集合为 (x, y, z)
    assert dict_from_expr(Integer(17)*x**7*y*z**12, gens=(x, y, z)) == ({(7, 1, 12): Integer(17)}, (x, y, z))

    # 断言：根据表达式生成字典，指定变量集合为 (x,)，结果应为 {(1,): Integer(1), (0,): 2*y + 3*z}，变量集合为 (x,)
    assert dict_from_expr(x + 2*y + 3*z, gens=(x,)) == ({(1,): Integer(1), (0,): 2*y + 3*z}, (x,))
    # 断言：根据表达式生成字典，指定变量集合为 (x, y)，结果应为 {(1, 0): Integer(1), (0, 1): Integer(2), (0, 0): 3*z}，变量集合为 (x, y)
    assert dict_from_expr(x + 2*y + 3*z, gens=(x, y)) == ({(1, 0): Integer(1), (0, 1): Integer(2), (0, 0): 3*z}, (x, y))
    # 断言：根据表达式生成字典，指定变量集合为 (x, y, z)，结果应为 {(1, 0, 0): Integer(1), (0, 1, 0): Integer(2), (0, 0, 1): Integer(3)}，变量集合为 (x, y, z)
    assert dict_from_expr(x + 2*y + 3*z, gens=(x, y, z)) == ({(1, 0, 0): Integer(1), (0, 1, 0): Integer(2), (0, 0, 1): Integer(3)}, (x, y, z))

    # 断言：根据表达式生成字典，指定变量集合为 (x,)，结果应为 {(1,): y + 2*z, (0,): 3*y*z}，变量集合为 (x,)
    assert dict_from_expr(x*y + 2*x*z + 3*y*z, gens=(x,)) == ({(1,): y + 2*z, (0,): 3*y*z}, (x,))
    # 断言：根据表达式生成字典，指定变量集合为 (x, y)，结果应为 {(1, 1): Integer(1), (1, 0): 2*z, (0, 1): 3*z}，变量集合为 (x, y)
    assert dict_from_expr(x*y + 2*x*z + 3*y*z, gens=(x, y)) == ({(1, 1): Integer(1), (1, 0): 2*z, (0, 1): 3*z}, (x, y))
    # 断言：根据表达式生成字典，指定变量集合为 (x, y, z)，结果应为 {(1, 1, 0): Integer(1), (1, 0, 1): Integer(2), (0, 1, 1): Integer(3)}，变量集合为 (x, y, z)
    assert dict_from_expr(x*y + 2*x*z + 3*y*z, gens=(x, y, z)) == ({(1, 1, 0): Integer(1), (1, 0, 1): Integer(2), (0, 1, 1): Integer(3)}, (x, y, z))

    # 断言：根据表达式生成字典，指定变量集合为 (x,)，结果应为 {(1,): 2**y}，变量集合为 (x,)
    assert dict_from_expr(2**y*x, gens=(x,)) == ({(1,): 2**y}, (x,))
    # 断言：根据表达式生成字典，未指定变量集合，应引发 PolynomialError 异常
    raises(PolynomialError, lambda: dict_from_expr(2**y*x, gens=(x, y)))
# 定义测试函数 test__dict_from_expr_no_gens，用于测试 dict_from_expr 函数在没有生成器的情况下的行为
def test__dict_from_expr_no_gens():
    # 断言：对整数 17 调用 dict_from_expr 应返回空字典中包含一个元组键值的字典，以及空元组作为剩余项
    assert dict_from_expr(Integer(17)) == ({(): Integer(17)}, ())

    # 断言：对变量 x 调用 dict_from_expr 应返回字典，包含以元组 (1,) 为键的整数 1，剩余项为 (x,)
    assert dict_from_expr(x) == ({(1,): Integer(1)}, (x,))
    # 断言：对变量 y 调用 dict_from_expr 应返回字典，包含以元组 (1,) 为键的整数 1，剩余项为 (y,)
    assert dict_from_expr(y) == ({(1,): Integer(1)}, (y,))

    # 断言：对表达式 x*y 调用 dict_from_expr 应返回字典，包含以元组 (1, 1) 为键的整数 1，剩余项为 (x, y)
    assert dict_from_expr(x*y) == ({(1, 1): Integer(1)}, (x, y))
    # 断言：对表达式 x+y 调用 dict_from_expr 应返回字典，包含以元组 (1, 0) 和 (0, 1) 为键的整数 1，剩余项为 (x, y)
    assert dict_from_expr(x + y) == ({(1, 0): Integer(1), (0, 1): Integer(1)}, (x, y))

    # 断言：对 sqrt(2) 调用 dict_from_expr 应返回字典，包含以元组 (1,) 为键的整数 1，剩余项为 (sqrt(2),)
    assert dict_from_expr(sqrt(2)) == ({(1,): Integer(1)}, (sqrt(2),))
    # 断言：对 sqrt(2) 调用 dict_from_expr，greedy 参数为 False，应返回空字典作为结果，剩余项为空元组
    assert dict_from_expr(sqrt(2), greedy=False) == ({(): sqrt(2)}, ())

    # 断言：对表达式 x*y 调用 dict_from_expr，domain 参数为 ZZ[x]，应返回字典，包含以元组 (1,) 为键的变量 x，剩余项为 (y,)
    assert dict_from_expr(x*y, domain=ZZ[x]) == ({(1,): x}, (y,))
    # 断言：对表达式 x*y 调用 dict_from_expr，domain 参数为 ZZ[y]，应返回字典，包含以元组 (1,) 为键的变量 y，剩余项为 (x,)
    assert dict_from_expr(x*y, domain=ZZ[y]) == ({(1,): y}, (x,))

    # 断言：对表达式 3*sqrt(2)*pi*x*y 调用 dict_from_expr，extension 参数为 None，应返回字典，包含以元组 (1, 1, 1, 1) 为键的整数 3，剩余项为 (x, y, pi, sqrt(2))
    assert dict_from_expr(3*sqrt(2)*pi*x*y, extension=None) == ({(1, 1, 1, 1): 3}, (x, y, pi, sqrt(2)))
    # 断言：对表达式 3*sqrt(2)*pi*x*y 调用 dict_from_expr，extension 参数为 True，应返回字典，包含以元组 (1, 1, 1) 为键的 3*sqrt(2)，剩余项为 (x, y, pi)
    assert dict_from_expr(3*sqrt(2)*pi*x*y, extension=True) == ({(1, 1, 1): 3*sqrt(2)}, (x, y, pi))

    # 断言：对表达式 cos(x)*sin(x) + cos(x)*sin(y) + cos(y)*sin(x) + cos(y)*sin(y) 调用 dict_from_expr，应返回字典，包含多个键值对，剩余项为 (cos(x), cos(y), sin(x), sin(y))
    assert dict_from_expr(f) == ({(0, 1, 0, 1): 1, (0, 1, 1, 0): 1, (1, 0, 0, 1): 1, (1, 0, 1, 0): 1}, (cos(x), cos(y), sin(x), sin(y)))


# 定义测试函数 test__parallel_dict_from_expr_if_gens，用于测试 parallel_dict_from_expr 函数在有生成器的情况下的行为
def test__parallel_dict_from_expr_if_gens():
    # 断言：对 [x + 2*y + 3*z, Integer(7)] 调用 parallel_dict_from_expr，gens 参数为 (x,)，应返回包含两个字典的列表，剩余项为 (x,)
    assert parallel_dict_from_expr([x + 2*y + 3*z, Integer(7)], gens=(x,)) == ([{(1,): Integer(1), (0,): 2*y + 3*z}, {(0,): Integer(7)}], (x,))


# 定义测试函数 test__parallel_dict_from_expr_no_gens，用于测试 parallel_dict_from_expr 函数在没有生成器的情况下的行为
def test__parallel_dict_from_expr_no_gens():
    # 断言：对 [x*y, Integer(3)] 调用 parallel_dict_from_expr，应返回包含两个字典的列表，剩余项为 (x, y)
    assert parallel_dict_from_expr([x*y, Integer(3)]) == ([{(1, 1): Integer(1)}, {(0, 0): Integer(3)}], (x, y))
    # 断言：对 [x*y, 2*z, Integer(3)] 调用 parallel_dict_from_expr，应返回包含三个字典的列表，剩余项为 (x, y, z)
    assert parallel_dict_from_expr([x*y, 2*z, Integer(3)]) == ([{(1, 1, 0): Integer(1)}, {(0, 0, 1): Integer(2)}, {(0, 0, 0): Integer(3)}], (x, y, z))
    # 断言：对 (Mul(x, x**2, evaluate=False),) 调用 parallel_dict_from_expr，应返回包含一个字典的列表，剩余项为 (x,)
    assert parallel_dict_from_expr((Mul(x, x**2, evaluate=False),)) == ([{(3,): 1}], (x,))


# 定义测试函数 test_parallel_dict_from_expr，用于测试 parallel_dict_from_expr 函数的行为
def test_parallel_dict_from_expr():
    # 断言：对 [Eq(x, 1), Eq(x**2, 2)] 调用 parallel_dict_from_expr，应返回包含两个字典的列表，剩余项为 (x,)
    assert parallel_dict_from_expr([Eq(x, 1), Eq(x**2, 2)]) == ([{(0,): -Integer(1), (1,): Integer(1)}, {(0,): -Integer(2), (2,): Integer(1)}], (x,))
    # 断言：对 [A*B - B*A] 调用 parallel_dict_from_expr，应引发 PolynomialError 异常
    raises(PolynomialError, lambda: parallel_dict_from_expr([A*B - B*A]))


# 定义测试函数 test_dict_from_expr，用于测试 dict_from_expr 函数的行为
def test_dict_from_expr():
    # 断言：对 Eq(x, 1) 调用 dict_from_expr，应返回包含两个键值对的字典，剩余项为 (x,)
    assert dict_from_expr(Eq(x, 1)) == ({(0,): -Integer(1), (1,): Integer(1)}, (x,))
    # 断言：对 A*B - B*A 调用 dict_from_expr，应引发 PolynomialError 异常
    raises(PolynomialError, lambda: dict_from_expr(A*B - B*A))
    # 断言：对 S.true 调用 dict_from_expr，应引发 PolynomialError 异常
    raises(PolynomialError, lambda: dict_from_expr(S.true))
```