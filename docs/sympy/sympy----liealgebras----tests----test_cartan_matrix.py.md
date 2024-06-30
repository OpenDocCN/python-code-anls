# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\test_cartan_matrix.py`

```
# 导入 SymPy 库中的 CartanMatrix 类，用于处理李代数的 Cartan 矩阵
from sympy.liealgebras.cartan_matrix import CartanMatrix
# 导入 SymPy 库中的 Matrix 类，用于创建和操作矩阵
from sympy.matrices import Matrix

# 定义测试函数 test_CartanMatrix
def test_CartanMatrix():
    # 创建一个 CartanMatrix 对象，参数为字符串 "A3"
    c = CartanMatrix("A3")
    # 创建一个 3x3 的 Matrix 对象，表示一个特定的矩阵
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -1, 0, -1, 2])
    # 断言两个对象相等
    assert c == m

    # 创建一个 CartanMatrix 对象，参数为列表 ["G",2]
    a = CartanMatrix(["G",2])
    # 创建一个 2x2 的 Matrix 对象，表示另一个特定的矩阵
    mt = Matrix(2, 2, [2, -1, -3, 2])
    # 断言两个对象相等
    assert a == mt
```