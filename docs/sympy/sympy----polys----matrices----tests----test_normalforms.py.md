# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_normalforms.py`

```
from sympy.testing.pytest import raises  # 导入测试框架中的 raises 函数

from sympy.core.symbol import Symbol  # 导入符号运算中的 Symbol 类
from sympy.polys.matrices.normalforms import (  # 导入多项式矩阵正常形式相关函数
    invariant_factors, smith_normal_form,
    hermite_normal_form, _hermite_normal_form, _hermite_normal_form_modulo_D)
from sympy.polys.domains import ZZ, QQ  # 导入整数和有理数域
from sympy.polys.matrices import DomainMatrix, DM  # 导入域矩阵和矩阵类
from sympy.polys.matrices.exceptions import DMDomainError, DMShapeError  # 导入矩阵异常类


def test_smith_normal():  # 定义测试 Smith 正常形式的函数

    m = DM([[12, 6, 4, 8], [3, 9, 6, 12], [2, 16, 14, 28], [20, 10, 10, 20]], ZZ)  # 创建整数域矩阵 m
    smf = DM([[1, 0, 0, 0], [0, 10, 0, 0], [0, 0, -30, 0], [0, 0, 0, 0]], ZZ)  # 创建整数域矩阵 smf
    assert smith_normal_form(m).to_dense() == smf  # 断言 m 的 Smith 正常形式与 smf 相同

    x = Symbol('x')  # 创建符号 x
    m = DM([[x-1,  1, -1],
            [  0,  x, -1],
            [  0, -1,  x]], QQ[x])  # 创建有理数域矩阵 m，元素是符号 x
    dx = m.domain.gens[0]  # 获取 m 的域的生成元
    assert invariant_factors(m) == (1, dx-1, dx**2-1)  # 断言 m 的不变因子

    zr = DomainMatrix([], (0, 2), ZZ)  # 创建空域矩阵 zr，形状为 (0, 2)
    zc = DomainMatrix([[], []], (2, 0), ZZ)  # 创建空域矩阵 zc，形状为 (2, 0)
    assert smith_normal_form(zr).to_dense() == zr  # 断言 zr 的 Smith 正常形式与 zr 相同
    assert smith_normal_form(zc).to_dense() == zc  # 断言 zc 的 Smith 正常形式与 zc 相同

    assert smith_normal_form(DM([[2, 4]], ZZ)).to_dense() == DM([[2, 0]], ZZ)  # 断言指定整数域矩阵的 Smith 正常形式
    assert smith_normal_form(DM([[0, -2]], ZZ)).to_dense() == DM([[-2, 0]], ZZ)  # 断言指定整数域矩阵的 Smith 正常形式
    assert smith_normal_form(DM([[0], [-2]], ZZ)).to_dense() == DM([[-2], [0]], ZZ)  # 断言指定整数域矩阵的 Smith 正常形式

    m =   DM([[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]], ZZ)  # 创建整数域矩阵 m
    snf = DM([[1, 0, 0, 0], [0, 6, 0, 0], [0, 0, 0, 0]], ZZ)  # 创建整数域矩阵 snf
    assert smith_normal_form(m).to_dense() == snf  # 断言 m 的 Smith 正常形式与 snf 相同

    raises(ValueError, lambda: smith_normal_form(DM([[1]], ZZ[x])))  # 断言在特定情况下会引发 ValueError 异常


def test_hermite_normal():  # 定义测试 Hermite 正常形式的函数

    m = DM([[2, 7, 17, 29, 41], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]], ZZ)  # 创建整数域矩阵 m
    hnf = DM([[1, 0, 0], [0, 2, 1], [0, 0, 1]], ZZ)  # 创建整数域矩阵 hnf
    assert hermite_normal_form(m) == hnf  # 断言 m 的 Hermite 正常形式与 hnf 相同
    assert hermite_normal_form(m, D=ZZ(2)) == hnf  # 断言指定 D 值的 Hermite 正常形式
    assert hermite_normal_form(m, D=ZZ(2), check_rank=True) == hnf  # 断言指定 D 值和检查秩的 Hermite 正常形式

    m = m.transpose()  # 转置矩阵 m
    hnf = DM([[37, 0, 19], [222, -6, 113], [48, 0, 25], [0, 2, 1], [0, 0, 1]], ZZ)  # 创建整数域矩阵 hnf
    assert hermite_normal_form(m) == hnf  # 断言 m 的 Hermite 正常形式与 hnf 相同
    raises(DMShapeError, lambda: _hermite_normal_form_modulo_D(m, ZZ(96)))  # 断言在特定情况下会引发 DMShapeError 异常
    raises(DMDomainError, lambda: _hermite_normal_form_modulo_D(m, QQ(96)))  # 断言在特定情况下会引发 DMDomainError 异常

    m = DM([[8, 28, 68, 116, 164], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]], ZZ)  # 创建整数域矩阵 m
    hnf = DM([[4, 0, 0], [0, 2, 1], [0, 0, 1]], ZZ)  # 创建整数域矩阵 hnf
    assert hermite_normal_form(m) == hnf  # 断言 m 的 Hermite 正常形式与 hnf 相同
    assert hermite_normal_form(m, D=ZZ(8)) == hnf  # 断言指定 D 值的 Hermite 正常形式
    assert hermite_normal_form(m, D=ZZ(8), check_rank=True) == hnf  # 断言指定 D 值和检查秩的 Hermite 正常形式

    m = DM([[10, 8, 6, 30, 2], [45, 36, 27, 18, 9], [5, 4, 3, 2, 1]], ZZ)  # 创建整数域矩阵 m
    hnf = DM([[26, 2], [0, 9], [0, 1]], ZZ)  # 创建整数域矩阵 hnf
    assert hermite_normal_form(m) == hnf  # 断言 m 的 Hermite 正常形式与 hnf 相同

    m = DM([[2, 7], [0, 0], [0, 0]], ZZ)  # 创建整数域矩阵 m
    hnf = DM([[1], [0], [0]], ZZ)  # 创建整数域矩阵 hnf
    assert hermite_normal_form(m) == hnf  # 断言 m 的 Hermite 正常形式与 hnf 相同

    m = DM([[-2, 1], [0, 1]], ZZ)  # 创建整数域矩阵 m
    hnf = DM([[2, 1], [0, 1]], ZZ)  # 创建整数域矩阵 hnf
    assert hermite_normal_form(m) == hnf  # 断言 m 的 Hermite 正常形式与 hnf 相同

    m = DomainMatrix([[QQ(
    # 调用 raises 函数，期望它引发 DMDomainError 异常，
    # 并执行 lambda 函数以捕获 _hermite_normal_form_modulo_D(m, ZZ(1)) 的执行结果
    raises(DMDomainError, lambda: _hermite_normal_form_modulo_D(m, ZZ(1)))
```