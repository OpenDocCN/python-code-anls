# `D:\src\scipysrc\sympy\sympy\tensor\tests\test_index_methods.py`

```
from sympy.core import symbols, S, Pow, Function  # 导入符号、常数S、幂次操作和函数模块
from sympy.functions import exp  # 导入指数函数exp
from sympy.testing.pytest import raises  # 导入测试工具raises
from sympy.tensor.indexed import Idx, IndexedBase  # 导入索引符号Idx和索引基类IndexedBase
from sympy.tensor.index_methods import IndexConformanceException  # 导入索引方法异常类IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)  # 导入获取收缩结构和索引的方法


def test_trivial_indices():
    x, y = symbols('x y')  # 创建符号变量x和y
    assert get_indices(x) == (set(), {})  # 检查单个符号x的索引为空集合
    assert get_indices(x*y) == (set(), {})  # 检查表达式x*y的索引为空集合
    assert get_indices(x + y) == (set(), {})  # 检查表达式x + y的索引为空集合
    assert get_indices(x**y) == (set(), {})  # 检查表达式x**y的索引为空集合


def test_get_indices_Indexed():
    x = IndexedBase('x')  # 创建IndexedBase对象x
    i, j = Idx('i'), Idx('j')  # 创建Idx对象i和j
    assert get_indices(x[i, j]) == ({i, j}, {})  # 检查Indexed对象x[i, j]的索引为{i, j}
    assert get_indices(x[j, i]) == ({j, i}, {})  # 检查Indexed对象x[j, i]的索引为{j, i}


def test_get_indices_Idx():
    f = Function('f')  # 创建函数对象f
    i, j = Idx('i'), Idx('j')  # 创建Idx对象i和j
    assert get_indices(f(i)*j) == ({i, j}, {})  # 检查表达式f(i)*j的索引为{i, j}
    assert get_indices(f(j, i)) == ({j, i}, {})  # 检查表达式f(j, i)的索引为{j, i}
    assert get_indices(f(i)*i) == (set(), {})  # 检查表达式f(i)*i的索引为空集合


def test_get_indices_mul():
    x = IndexedBase('x')  # 创建IndexedBase对象x
    y = IndexedBase('y')  # 创建IndexedBase对象y
    i, j = Idx('i'), Idx('j')  # 创建Idx对象i和j
    assert get_indices(x[j]*y[i]) == ({i, j}, {})  # 检查表达式x[j]*y[i]的索引为{i, j}
    assert get_indices(x[i]*y[j]) == ({i, j}, {})  # 检查表达式x[i]*y[j]的索引为{i, j}


def test_get_indices_exceptions():
    x = IndexedBase('x')  # 创建IndexedBase对象x
    y = IndexedBase('y')  # 创建IndexedBase对象y
    i, j = Idx('i'), Idx('j')  # 创建Idx对象i和j
    raises(IndexConformanceException, lambda: get_indices(x[i] + y[j]))  # 检查表达式x[i] + y[j]是否引发IndexConformanceException异常


def test_scalar_broadcast():
    x = IndexedBase('x')  # 创建IndexedBase对象x
    y = IndexedBase('y')  # 创建IndexedBase对象y
    i, j = Idx('i'), Idx('j')  # 创建Idx对象i和j
    assert get_indices(x[i] + y[i, i]) == ({i}, {})  # 检查表达式x[i] + y[i, i]的索引为{i}
    assert get_indices(x[i] + y[j, j]) == ({i}, {})  # 检查表达式x[i] + y[j, j]的索引为{i}


def test_get_indices_add():
    x = IndexedBase('x')  # 创建IndexedBase对象x
    y = IndexedBase('y')  # 创建IndexedBase对象y
    A = IndexedBase('A')  # 创建IndexedBase对象A
    i, j, k = Idx('i'), Idx('j'), Idx('k')  # 创建Idx对象i, j, k
    assert get_indices(x[i] + 2*y[i]) == ({i}, {})  # 检查表达式x[i] + 2*y[i]的索引为{i}
    assert get_indices(y[i] + 2*A[i, j]*x[j]) == ({i}, {})  # 检查表达式y[i] + 2*A[i, j]*x[j]的索引为{i}
    assert get_indices(y[i] + 2*(x[i] + A[i, j]*x[j])) == ({i}, {})  # 检查表达式y[i] + 2*(x[i] + A[i, j]*x[j])的索引为{i}
    assert get_indices(y[i] + x[i]*(A[j, j] + 1)) == ({i}, {})  # 检查表达式y[i] + x[i]*(A[j, j] + 1)的索引为{i}
    assert get_indices(
        y[i] + x[i]*x[j]*(y[j] + A[j, k]*x[k])) == ({i}, {})  # 检查表达式y[i] + x[i]*x[j]*(y[j] + A[j, k]*x[k])的索引为{i}


def test_get_indices_Pow():
    x = IndexedBase('x')  # 创建IndexedBase对象x
    y = IndexedBase('y')  # 创建IndexedBase对象y
    A = IndexedBase('A')  # 创建IndexedBase对象A
    i, j, k = Idx('i'), Idx('j'), Idx('k')  # 创建Idx对象i, j, k
    assert get_indices(Pow(x[i], y[j])) == ({i, j}, {})  # 检查表达式Pow(x[i], y[j])的索引为{i, j}
    assert get_indices(Pow(x[i, k], y[j, k])) == ({i, j, k}, {})  # 检查表达式Pow(x[i, k], y[j, k])的索引为{i, j, k}
    assert get_indices(Pow(A[i, k], y[k] + A[k, j]*x[j])) == ({i, k}, {})  # 检查表达式Pow(A[i, k], y[k] + A[k, j]*x[j])的索引为{i, k}
    assert get_indices(Pow(2, x[i])) == get_indices(exp(x[i]))  # 检查表达式Pow(2, x[i])的索引与exp(x[i])相同

    # test of a design decision, this may change:
    assert get_indices(Pow(x[i], 2)) == ({i}, {})  # 检查表达式Pow(x[i], 2)的索引为{i}


def test_get_contraction_structure_basic():
    x = IndexedBase('x')  # 创建IndexedBase对象x
    y = IndexedBase('y')  # 创建IndexedBase对象y
    i, j = Idx('i'), Idx('j')  # 创建Idx对象i和j
    assert get_contraction_structure(x[i]*y[j]) == {None: {x[i]*y[j]}}  # 检查表达式x[i]*y[j]的收缩结构
    assert get_contraction_structure(x[i] + y[j]) == {None: {x[i], y[j]}}  # 检查表达式x[i] + y[j]的收缩结构
    assert get_contraction_structure(x[i]*y[i]) == {(i,): {x[i]*y[i]}}  # 检查表达式x[i]*y[i]的收缩结构
    assert get_contraction_structure(
        1 + x[i]*y[i]) == {None: {S.One}, (i,): {x[i]*y[i]}}  # 检查表达式1 + x[i]*y[i]的收缩结构
    # 使用断言检查函数 get_contraction_structure 的返回结果是否与预期相符
    assert get_contraction_structure(x[i]**y[i]) == {None: {x[i]**y[i]}}
def test_get_contraction_structure_complex():
    # 创建 IndexedBase 对象 x, y, A，分别用于索引 x, y 和 A
    x = IndexedBase('x')
    y = IndexedBase('y')
    A = IndexedBase('A')
    # 创建 Idx 对象 i, j, k，分别用于索引 i, j, k
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    # 创建表达式 expr1，表示 y[i] + A[i, j]*x[j]
    expr1 = y[i] + A[i, j]*x[j]
    # 创建预期的结构字典 d1，用于存储 expr1 的收缩结构
    d1 = {None: {y[i]}, (j,): {A[i, j]*x[j]}}
    # 断言调用 get_contraction_structure 函数后返回的结构字典与预期的 d1 相等
    assert get_contraction_structure(expr1) == d1
    # 创建表达式 expr2，表示 expr1*A[k, i] + x[k]
    expr2 = expr1*A[k, i] + x[k]
    # 创建预期的结构字典 d2，用于存储 expr2 的收缩结构
    d2 = {None: {x[k]}, (i,): {expr1*A[k, i]}, expr1*A[k, i]: [d1]}
    # 断言调用 get_contraction_structure 函数后返回的结构字典与预期的 d2 相等
    assert get_contraction_structure(expr2) == d2


def test_contraction_structure_simple_Pow():
    # 创建 IndexedBase 对象 x, y，分别用于索引 x 和 y
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建 Idx 对象 i, j, k，分别用于索引 i, j, k
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    # 创建表达式 ii_jj，表示 x[i, i]**y[j, j]
    ii_jj = x[i, i]**y[j, j]
    # 创建预期的结构字典，用于存储 ii_jj 的收缩结构
    assert get_contraction_structure(ii_jj) == {
        None: {ii_jj},
        ii_jj: [
            {(i,): {x[i, i]}},
            {(j,): {y[j, j]}}
        ]
    }

    # 创建表达式 ii_jk，表示 x[i, i]**y[j, k]
    ii_jk = x[i, i]**y[j, k]
    # 创建预期的结构字典，用于存储 ii_jk 的收缩结构
    assert get_contraction_structure(ii_jk) == {
        None: {x[i, i]**y[j, k]},
        x[i, i]**y[j, k]: [
            {(i,): {x[i, i]}}
        ]
    }


def test_contraction_structure_Mul_and_Pow():
    # 创建 IndexedBase 对象 x, y，分别用于索引 x 和 y
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建 Idx 对象 i, j, k，分别用于索引 i, j, k
    i, j, k = Idx('i'), Idx('j'), Idx('k')

    # 创建表达式 i_ji，表示 x[i]**(y[j]*x[i])
    i_ji = x[i]**(y[j]*x[i])
    # 创建预期的结构字典，用于存储 i_ji 的收缩结构
    assert get_contraction_structure(i_ji) == {None: {i_ji}}

    # 创建表达式 ij_i，表示 (x[i]*y[j])**(y[i])
    ij_i = (x[i]*y[j])**(y[i])
    # 创建预期的结构字典，用于存储 ij_i 的收缩结构
    assert get_contraction_structure(ij_i) == {None: {ij_i}}

    # 创建表达式 j_ij_i，表示 x[j]*(x[i]*y[j])**(y[i])
    j_ij_i = x[j]*(x[i]*y[j])**(y[i])
    # 创建预期的结构字典，用于存储 j_ij_i 的收缩结构
    assert get_contraction_structure(j_ij_i) == {(j,): {j_ij_i}}

    # 创建表达式 j_i_ji，表示 x[j]*x[i]**(y[j]*x[i])
    j_i_ji = x[j]*x[i]**(y[j]*x[i])
    # 创建预期的结构字典，用于存储 j_i_ji 的收缩结构
    assert get_contraction_structure(j_i_ji) == {(j,): {j_i_ji}}

    # 创建表达式 ij_exp_kki，表示 x[i]*y[j]*exp(y[i]*y[k, k])
    ij_exp_kki = x[i]*y[j]*exp(y[i]*y[k, k])
    # 调用 get_contraction_structure 函数得到结果
    result = get_contraction_structure(ij_exp_kki)
    # 创建预期的结构字典 expected，用于存储 ij_exp_kki 的收缩结构
    expected = {
        (i,): {ij_exp_kki},
        ij_exp_kki: [{
            None: {exp(y[i]*y[k, k])},
            exp(y[i]*y[k, k]): [{
                None: {y[i]*y[k, k]},
                y[i]*y[k, k]: [{(k,): {y[k, k]}}]
            }]
        }]
    }
    # 断言调用 get_contraction_structure 函数后返回的结果与预期的 expected 相等
    assert result == expected


def test_contraction_structure_Add_in_Pow():
    # 创建 IndexedBase 对象 x, y，分别用于索引 x 和 y
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建 Idx 对象 i, j, k，分别用于索引 i, j, k
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    # 创建表达式 s_ii_jj_s，表示 (1 + x[i, i])**(1 + y[j, j])
    s_ii_jj_s = (1 + x[i, i])**(1 + y[j, j])
    # 创建预期的结构字典 expected，用于存储 s_ii_jj_s 的收缩结构
    expected = {
        None: {s_ii_jj_s},
        s_ii_jj_s: [
            {None: {S.One}, (i,): {x[i, i]}},
            {None: {S.One}, (j,): {y[j, j]}}
        ]
    }
    # 调用 get_contraction_structure 函数得到结果
    result = get_contraction_structure(s_ii_jj_s)
    # 断言调用 get_contraction_structure 函数后返回的结果与预期的 expected 相等
    assert result == expected

    # 创建表达式 s_ii_jk_s，表示 (1 + x[i, i]) ** (1 + y[j, k])
    s_ii_jk_s = (1 + x[i, i]) ** (1 + y[j, k])
    # 创建预期的结构字典 expected_2，用于存储 s_ii_jk_s 的收缩结构
    expected_2 = {
        None: {(x[i, i] + 1)**(y[j, k] + 1)},
        s_ii_jk_s: [
            {None: {S.One}, (i,): {x[i, i]}}
        ]
    }
    # 调用 get_contraction_structure 函数得到结果
    result_2 = get_contraction_structure(s_ii_jk_s)
    # 断言调用 get_contraction_structure 函数后返回的结果与预期的 expected_2 相等
    assert result_2 == expected_2


def test_contraction_structure_Pow_in_Pow():
    # 创建 IndexedBase 对象 x, y, z，分别用于索引 x, y, z
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    # 创建 Idx 对象 i, j, k，分别用于索引 i, j, k
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    # 创建表达式 ii_jj_kk，表示 x[i, i]**y[j, j]**z[k, k]
    expected = {
        None: {ii_jj_kk},  # 创建一个字典，键为None，值为集合{ii_jj_kk}
        ii_jj_kk: [  # 以 ii_jj_kk 为键，创建一个包含两个字典的列表
            {(i,): {x[i, i]}},  # 第一个字典，键为元组(i,)，值为{x[i, i]}
            {  # 第二个字典，键为None，值为集合{y[j, j]**z[k, k]}
                None: {y[j, j]**z[k, k]},  # 第二个字典中的键为None，值为集合{y[j, j]**z[k, k]}
                y[j, j]**z[k, k]: [  # 创建一个列表，以 y[j, j]**z[k, k] 为键
                    {(j,): {y[j, j]}},  # 列表中的第一个字典，键为元组(j,)，值为{y[j, j]}
                    {(k,): {z[k, k]}}  # 列表中的第二个字典，键为元组(k,)，值为{z[k, k]}
                ]
            }
        ]
    }
    # 断言调用函数 get_contraction_structure(ii_jj_kk) 返回的结果与 expected 相等
    assert get_contraction_structure(ii_jj_kk) == expected
# 定义一个测试函数，用于测试一些特定函数和符号的索引操作
def test_ufunc_support():
    # 创建函数对象 f 和 g
    f = Function('f')
    g = Function('g')
    # 创建 IndexedBase 对象 x 和 y，用于表示带索引的变量
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建索引对象 i 和 j，用于表示索引变量
    i, j = Idx('i'), Idx('j')
    # 创建符号对象 a，用于表示常数或标量变量

    # 断言语句，测试 get_indices 函数对不同形式 f 函数参数的索引提取
    assert get_indices(f(x[i])) == ({i}, {})
    assert get_indices(f(x[i], y[j])) == ({i, j}, {})
    assert get_indices(f(y[i])*g(x[i])) == (set(), {})
    assert get_indices(f(a, x[i])) == ({i}, {})
    assert get_indices(f(a, y[i], x[j])*g(x[i])) == ({j}, {})
    assert get_indices(g(f(x[i]))) == ({i}, {})

    # 断言语句，测试 get_contraction_structure 函数对不同形式 f 函数参数的收缩结构分析
    assert get_contraction_structure(f(x[i])) == {None: {f(x[i])}}
    assert get_contraction_structure(
        f(y[i])*g(x[i])) == {(i,): {f(y[i])*g(x[i])}}
    assert get_contraction_structure(
        f(y[i])*g(f(x[i]))) == {(i,): {f(y[i])*g(f(x[i]))}}
    assert get_contraction_structure(
        f(x[j], y[i])*g(x[i])) == {(i,): {f(x[j], y[i])*g(x[i])}}
```