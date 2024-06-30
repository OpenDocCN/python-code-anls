# `D:\src\scipysrc\sympy\sympy\tensor\tests\test_functions.py`

```
from sympy.tensor.functions import TensorProduct   # 导入张量积函数
from sympy.matrices.dense import Matrix            # 导入矩阵类
from sympy.matrices.expressions.matexpr import MatrixSymbol   # 导入矩阵符号类
from sympy.tensor.array import Array               # 导入数组类
from sympy.abc import x, y, z                     # 导入符号 x, y, z
from sympy.abc import i, j, k, l                   # 导入符号 i, j, k, l

A = MatrixSymbol("A", 3, 3)                        # 定义一个 3x3 的矩阵符号 A
B = MatrixSymbol("B", 3, 3)                        # 定义一个 3x3 的矩阵符号 B
C = MatrixSymbol("C", 3, 3)                        # 定义一个 3x3 的矩阵符号 C

def test_TensorProduct_construction():
    assert TensorProduct(3, 4) == 12                # 测试标量的张量积结果
    assert isinstance(TensorProduct(A, A), TensorProduct)   # 测试矩阵符号的张量积结果是否为张量积对象

    expr = TensorProduct(TensorProduct(x, y), z)    # 创建符号表达式：(x ⊗ y) ⊗ z
    assert expr == x*y*z                            # 验证符号表达式的正确性

    expr = TensorProduct(TensorProduct(A, B), C)    # 创建矩阵符号的张量积：(A ⊗ B) ⊗ C
    assert expr == TensorProduct(A, B, C)           # 验证矩阵符号张量积的正确性

    expr = TensorProduct(Matrix.eye(2), Array([[0, -1], [1, 0]]))   # 创建矩阵和数组的张量积
    assert expr == Array([                          # 验证张量积结果是否正确
        [
            [[0, -1], [1, 0]],
            [[0, 0], [0, 0]]
        ],
        [
            [[0, 0], [0, 0]],
            [[0, -1], [1, 0]]
        ]
    ])

def test_TensorProduct_shape():
    expr = TensorProduct(3, 4, evaluate=False)      # 创建不进行评估的张量积
    assert expr.shape == ()                         # 验证形状是否为空元组
    assert expr.rank() == 0                         # 验证张量积的秩是否为0

    expr = TensorProduct(Array([1, 2]), Array([x, y]), evaluate=False)   # 创建不进行评估的数组的张量积
    assert expr.shape == (2, 2)                     # 验证张量积的形状为 (2, 2)
    assert expr.rank() == 2                         # 验证张量积的秩为2
    expr = TensorProduct(expr, expr, evaluate=False) # 创建不进行评估的张量积的张量积
    assert expr.shape == (2, 2, 2, 2)               # 验证张量积的形状为 (2, 2, 2, 2)
    assert expr.rank() == 4                         # 验证张量积的秩为4

    expr = TensorProduct(Matrix.eye(2), Array([[0, -1], [1, 0]]), evaluate=False)   # 创建不进行评估的矩阵和数组的张量积
    assert expr.shape == (2, 2, 2, 2)               # 验证张量积的形状为 (2, 2, 2, 2)
    assert expr.rank() == 4                         # 验证张量积的秩为4

def test_TensorProduct_getitem():
    expr = TensorProduct(A, B)                      # 创建矩阵符号 A 和 B 的张量积
    assert expr[i, j, k, l] == A[i, j]*B[k, l]      # 验证张量积的索引操作结果
```