# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_normalforms.py`

```
# 导入用于测试的警告函数
from sympy.testing.pytest import warns_deprecated_sympy

# 导入符号相关的类和函数
from sympy.core.symbol import Symbol
# 导入多项式相关的类和函数
from sympy.polys.polytools import Poly
# 导入矩阵类
from sympy.matrices import Matrix
# 导入正规形相关的函数
from sympy.matrices.normalforms import (
    invariant_factors,  # 导入不变因子计算函数
    smith_normal_form,   # 导入史密斯正规形计算函数
    hermite_normal_form   # 导入赫尔米特正规形计算函数
)
# 导入多项式的环和域
from sympy.polys.domains import ZZ, QQ
# 导入整数类
from sympy.core.numbers import Integer


# 定义测试函数 test_smith_normal
def test_smith_normal():
    # 创建一个整数类型的矩阵
    m = Matrix([[12, 6, 4, 8], [3, 9, 6, 12], [2, 16, 14, 28], [20, 10, 10, 20]])
    # 预期的史密斯正规形矩阵
    smf = Matrix([[1, 0, 0, 0], [0, 10, 0, 0], [0, 0, -30, 0], [0, 0, 0, 0]])
    # 断言计算得到的史密斯正规形与预期的相等
    assert smith_normal_form(m) == smf

    # 创建符号变量 x
    x = Symbol('x')
    # 使用警告函数监控即将弃用的功能
    with warns_deprecated_sympy():
        # 创建一个多项式类型的矩阵
        m = Matrix([[Poly(x - 1), Poly(1, x), Poly(-1, x)],
                    [0, Poly(x), Poly(-1, x)],
                    [Poly(0, x), Poly(-1, x), Poly(x)]])
    # 预期的不变因子
    invs = 1, x - 1, x**2 - 1
    # 断言计算得到的不变因子与预期的相等
    assert invariant_factors(m, domain=QQ[x]) == invs

    # 创建一个整数类型的矩阵
    m = Matrix([[2, 4]])
    # 预期的史密斯正规形矩阵
    smf = Matrix([[2, 0]])
    # 断言计算得到的史密斯正规形与预期的相等
    assert smith_normal_form(m) == smf


# 定义测试函数 test_smith_normal_deprecated
def test_smith_normal_deprecated():
    # 导入原始矩阵类别名为 Matrix
    from sympy.polys.solvers import RawMatrix as Matrix

    # 使用警告函数监控即将弃用的功能
    with warns_deprecated_sympy():
        # 创建一个整数类型的矩阵
        m = Matrix([[12, 6, 4, 8], [3, 9, 6, 12], [2, 16, 14, 28], [20, 10, 10, 20]])
    # 设置矩阵的环为整数环 ZZ
    setattr(m, 'ring', ZZ)
    # 使用警告函数监控即将弃用的功能
    with warns_deprecated_sympy():
        # 预期的史密斯正规形矩阵
        smf = Matrix([[1, 0, 0, 0], [0, 10, 0, 0], [0, 0, -30, 0], [0, 0, 0, 0]])
    # 断言计算得到的史密斯正规形与预期的相等
    assert smith_normal_form(m) == smf

    # 创建符号变量 x
    x = Symbol('x')
    # 使用警告函数监控即将弃用的功能
    with warns_deprecated_sympy():
        # 创建一个多项式类型的矩阵
        m = Matrix([[Poly(x - 1), Poly(1, x), Poly(-1, x)],
                    [0, Poly(x), Poly(-1, x)],
                    [Poly(0, x), Poly(-1, x), Poly(x)]])
    # 设置矩阵的域为多项式环 QQ[x]
    setattr(m, 'ring', QQ[x])
    # 预期的不变因子
    invs = (Poly(1, x, domain='QQ'), Poly(x - 1, domain='QQ'), Poly(x**2 - 1, domain='QQ'))
    # 断言计算得到的不变因子与预期的相等
    assert invariant_factors(m) == invs

    # 使用警告函数监控即将弃用的功能
    with warns_deprecated_sympy():
        # 创建一个整数类型的矩阵
        m = Matrix([[2, 4]])
    # 设置矩阵的环为整数环 ZZ
    setattr(m, 'ring', ZZ)
    # 使用警告函数监控即将弃用的功能
    with warns_deprecated_sympy():
        # 预期的史密斯正规形矩阵
        smf = Matrix([[2, 0]])
    # 断言计算得到的史密斯正规形与预期的相等
    assert smith_normal_form(m) == smf


# 定义测试函数 test_hermite_normal
def test_hermite_normal():
    # 创建一个整数类型的矩阵
    m = Matrix([[2, 7, 17, 29, 41], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]])
    # 预期的赫尔米特正规形矩阵
    hnf = Matrix([[1, 0, 0], [0, 2, 1], [0, 0, 1]])
    # 断言计算得到的赫尔米特正规形与预期的相等
    assert hermite_normal_form(m) == hnf

    # 转置后的预期赫尔米特正规形矩阵
    tr_hnf = Matrix([[37, 0, 19], [222, -6, 113], [48, 0, 25], [0, 2, 1], [0, 0, 1]])
    # 断言转置后计算得到的赫尔米特正规形与预期的相等
    assert hermite_normal_form(m.transpose()) == tr_hnf

    # 创建一个整数类型的矩阵
    m = Matrix([[8, 28, 68, 116, 164], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]])
    # 预期的赫尔米特正规形矩阵
    hnf = Matrix([[4, 0, 0], [0, 2, 1], [0, 0, 1]])
    # 断言计算得到的赫尔米特正规形与预期的相等
    assert hermite_normal_form(m) == hnf
    # 使用参数 D=8，预期的赫尔米特正规形矩阵
    assert hermite_normal_form(m, D=8) == hnf
    # 使用参数 D=ZZ(8)，预期的赫尔米特正规形矩阵
    assert hermite_normal_form(m, D=ZZ(8)) == hnf
    # 使用参数 D=Integer(8)，预期的赫尔米特正规形矩阵
    assert hermite_normal_form(m, D=Integer(8)) == hnf

    # 创建一个整数类型的矩阵
    m = Matrix([[10, 8, 6, 30, 2], [45, 36, 27, 18, 9], [5, 4, 3, 2, 1]])
    # 预期的赫尔米特正规形矩阵
    hnf = Matrix([[26, 2], [0
    # 断言语句，用于检查 hermite_normal_form(A) 的返回值是否等于 H
    assert hermite_normal_form(A) == H
```