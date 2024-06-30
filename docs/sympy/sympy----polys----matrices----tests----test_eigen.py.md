# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_eigen.py`

```
"""
Tests for the sympy.polys.matrices.eigen module
"""

# 导入必要的模块和函数
from sympy.core.singleton import S                            # 导入S符号
from sympy.functions.elementary.miscellaneous import sqrt    # 导入平方根函数
from sympy.matrices.dense import Matrix                       # 导入矩阵类

from sympy.polys.agca.extensions import FiniteExtension      # 导入有限扩展类
from sympy.polys.domains import QQ                            # 导入有理数域
from sympy.polys.polytools import Poly                        # 导入多项式类
from sympy.polys.rootoftools import CRootOf                   # 导入根式工具类
from sympy.polys.matrices.domainmatrix import DomainMatrix    # 导入域矩阵类

from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy  # 导入特征向量相关函数


def test_dom_eigenvects_rational():
    # Rational eigenvalues
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(1), QQ(2)]], (2, 2), QQ)  # 创建有理数域矩阵A
    rational_eigenvects = [
        (QQ, QQ(3), 1, DomainMatrix([[QQ(1), QQ(1)]], (1, 2), QQ)),   # 有理特征向量组成的列表
        (QQ, QQ(0), 1, DomainMatrix([[QQ(-2), QQ(1)]], (1, 2), QQ)),  # 另一个有理特征向量组成的列表
    ]
    assert dom_eigenvects(A) == (rational_eigenvects, [])  # 断言验证特征向量计算结果

    # Test converting to Expr:
    sympy_eigenvects = [
        (S(3), 1, [Matrix([1, 1])]),    # 转换为SymPy表达式后的特征向量列表
        (S(0), 1, [Matrix([-2, 1])]),   # 另一个转换为SymPy表达式后的特征向量列表
    ]
    assert dom_eigenvects_to_sympy(rational_eigenvects, [], Matrix) == sympy_eigenvects  # 断言验证转换结果


def test_dom_eigenvects_algebraic():
    # Algebraic eigenvalues
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)  # 创建有理数域矩阵A
    Avects = dom_eigenvects(A)  # 计算特征向量

    # Extract the dummy to build the expected result:
    lamda = Avects[1][0][1].gens[0]  # 提取特征值的占位符
    irreducible = Poly(lamda**2 - 5*lamda - 2, lamda, domain=QQ)  # 创建不可约多项式
    K = FiniteExtension(irreducible)  # 使用不可约多项式创建有限扩展
    KK = K.from_sympy
    algebraic_eigenvects = [
        (K, irreducible, 1, DomainMatrix([[KK((lamda-4)/3), KK(1)]], (1, 2), K)),  # 代数特征向量组成的列表
    ]
    assert Avects == ([], algebraic_eigenvects)  # 断言验证特征向量计算结果

    # Test converting to Expr:
    sympy_eigenvects = [
        (S(5)/2 - sqrt(33)/2, 1, [Matrix([[-sqrt(33)/6 - S(1)/2], [1]])]),   # 转换为SymPy表达式后的特征向量列表
        (S(5)/2 + sqrt(33)/2, 1, [Matrix([[-S(1)/2 + sqrt(33)/6], [1]])]),    # 另一个转换为SymPy表达式后的特征向量列表
    ]
    assert dom_eigenvects_to_sympy([], algebraic_eigenvects, Matrix) == sympy_eigenvects  # 断言验证转换结果


def test_dom_eigenvects_rootof():
    # Algebraic eigenvalues
    A = DomainMatrix([
        [0, 0, 0, 0, -1],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]], (5, 5), QQ)  # 创建有理数域矩阵A
    Avects = dom_eigenvects(A)  # 计算特征向量

    # Extract the dummy to build the expected result:
    lamda = Avects[1][0][1].gens[0]  # 提取特征值的占位符
    irreducible = Poly(lamda**5 - lamda + 1, lamda, domain=QQ)  # 创建不可约多项式
    K = FiniteExtension(irreducible)  # 使用不可约多项式创建有限扩展
    KK = K.from_sympy
    algebraic_eigenvects = [
        (K, irreducible, 1,
            DomainMatrix([
                [KK(lamda**4-1), KK(lamda**3), KK(lamda**2), KK(lamda), KK(1)]   # 代数特征向量组成的列表
            ], (1, 5), K)),
    ]
    assert Avects == ([], algebraic_eigenvects)  # 断言验证特征向量计算结果

    # Test converting to Expr (slow):
    l0, l1, l2, l3, l4 = [CRootOf(lamda**5 - lamda + 1, i) for i in range(5)]  # 计算根式的列表
    # 定义包含多个元组的列表，每个元组包含以下内容：
    # l0, l1, l2, l3, l4：代表五个未知数
    # 1：一个整数
    # [Matrix([...])]：包含一个元素的列表，元素是一个 Matrix 对象，其内容为一个向量表达式
    sympy_eigenvects = [
        (l0, 1, [Matrix([-1 + l0**4, l0**3, l0**2, l0, 1])]),
        (l1, 1, [Matrix([-1 + l1**4, l1**3, l1**2, l1, 1])]),
        (l2, 1, [Matrix([-1 + l2**4, l2**3, l2**2, l2, 1])]),
        (l3, 1, [Matrix([-1 + l3**4, l3**3, l3**2, l3, 1])]),
        (l4, 1, [Matrix([-1 + l4**4, l4**3, l4**2, l4, 1])]),
    ]
    
    # 使用 dom_eigenvects_to_sympy 函数对空列表、代数特征向量、Matrix 类进行调用，并将结果与 sympy_eigenvects 进行比较
    assert dom_eigenvects_to_sympy([], algebraic_eigenvects, Matrix) == sympy_eigenvects
```