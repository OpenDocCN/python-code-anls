# `D:\src\scipysrc\sympy\sympy\polys\matrices\eigen.py`

```
"""

Routines for computing eigenvectors with DomainMatrix.

"""
# 导入所需的模块和类
from sympy.core.symbol import Dummy
# 导入扩展模块
from ..agca.extensions import FiniteExtension
# 导入因子分解工具函数
from ..factortools import dup_factor_list
# 导入多项式根计算函数
from ..polyroots import roots
# 导入多项式类
from ..polytools import Poly
# 导入复根对象
from ..rootoftools import CRootOf
# 导入自定义的DomainMatrix类
from .domainmatrix import DomainMatrix


# 定义函数：计算给定矩阵A的有理和代数特征向量
def dom_eigenvects(A, l=Dummy('lambda')):
    # 计算矩阵A的特征多项式
    charpoly = A.charpoly()
    # 获取矩阵A的行数和列数
    rows, cols = A.shape
    # 获取矩阵A的定义域
    domain = A.domain
    # 对特征多项式进行因子分解
    _, factors = dup_factor_list(charpoly, domain)

    # 初始化有理特征向量列表和代数特征向量列表
    rational_eigenvects = []
    algebraic_eigenvects = []

    # 遍历特征多项式的因子
    for base, exp in factors:
        # 如果因子长度为2，说明是有理特征值
        if len(base) == 2:
            # 使用矩阵A的定义域计算特征值
            field = domain
            eigenval = -base[1] / base[0]

            # 构造特征值矩阵EE
            EE_items = [
                [eigenval if i == j else field.zero for j in range(cols)]
                for i in range(rows)]
            EE = DomainMatrix(EE_items, (rows, cols), field)

            # 计算零空间基础
            basis = (A - EE).nullspace(divide_last=True)
            rational_eigenvects.append((field, eigenval, exp, basis))
        # 否则为代数特征值
        else:
            # 构造最小多项式对象
            minpoly = Poly.from_list(base, l, domain=domain)
            # 使用有限扩展域计算特征值
            field = FiniteExtension(minpoly)
            eigenval = field(l)

            # 构造矩阵AA
            AA_items = [
                [Poly.from_list([item], l, domain=domain).rep for item in row]
                for row in A.rep.to_ddm()]
            AA_items = [[field(item) for item in row] for row in AA_items]
            AA = DomainMatrix(AA_items, (rows, cols), field)

            # 构造特征值矩阵EE
            EE_items = [
                [eigenval if i == j else field.zero for j in range(cols)]
                for i in range(rows)]
            EE = DomainMatrix(EE_items, (rows, cols), field)

            # 计算零空间基础
            basis = (AA - EE).nullspace(divide_last=True)
            algebraic_eigenvects.append((field, minpoly, exp, basis))

    # 返回有理和代数特征向量列表
    return rational_eigenvects, algebraic_eigenvects


# 定义函数：将DomainMatrix中的特征向量转换为Sympy中的矩阵表示
def dom_eigenvects_to_sympy(
    rational_eigenvects, algebraic_eigenvects,
    Matrix, **kwargs
):
    # 初始化结果列表
    result = []

    # 处理有理特征向量
    for field, eigenvalue, multiplicity, eigenvects in rational_eigenvects:
        # 将特征向量转换为Sympy矩阵表示
        eigenvects = eigenvects.rep.to_ddm()
        eigenvalue = field.to_sympy(eigenvalue)
        new_eigenvects = [
            Matrix([field.to_sympy(x) for x in vect])
            for vect in eigenvects]
        result.append((eigenvalue, multiplicity, new_eigenvects))

    # 处理代数特征向量
    for field, minpoly, multiplicity, eigenvects in algebraic_eigenvects:
        # 将特征向量转换为Sympy矩阵表示
        eigenvects = eigenvects.rep.to_ddm()
        l = minpoly.gens[0]

        eigenvects = [[field.to_sympy(x) for x in vect] for vect in eigenvects]

        # 计算最小多项式的次数和表达式
        degree = minpoly.degree()
        minpoly = minpoly.as_expr()
        # 计算多项式的根
        eigenvals = roots(minpoly, l, **kwargs)
        if len(eigenvals) != degree:
            eigenvals = [CRootOf(minpoly, l, idx) for idx in range(degree)]

        # 对每个特征值计算新的特征向量矩阵
        for eigenvalue in eigenvals:
            new_eigenvects = [
                Matrix([x.subs(l, eigenvalue) for x in vect])
                for vect in eigenvects]
            result.append((eigenvalue, multiplicity, new_eigenvects))
    # 返回变量 result 中存储的结果
    return result
```