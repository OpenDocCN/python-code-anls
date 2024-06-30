# `D:\src\scipysrc\sympy\sympy\liealgebras\cartan_matrix.py`

```
# 导入CartanType类，用于处理李代数的Cartan类型
from .cartan_type import CartanType

# 定义函数CartanMatrix，用于获取特定李代数的Cartan矩阵
def CartanMatrix(ct):
    """Access the Cartan matrix of a specific Lie algebra

    Examples
    ========

    >>> from sympy.liealgebras.cartan_matrix import CartanMatrix
    >>> CartanMatrix("A2")
    Matrix([
    [ 2, -1],
    [-1,  2]])

    >>> CartanMatrix(['C', 3])
    Matrix([
    [ 2, -1,  0],
    [-1,  2, -1],
    [ 0, -2,  2]])

    This method works by returning the Cartan matrix
    which corresponds to Cartan type t.
    """
    
    # 创建CartanType对象，使用给定的Cartan类型（ct），并获取其Cartan矩阵
    return CartanType(ct).cartan_matrix()
```