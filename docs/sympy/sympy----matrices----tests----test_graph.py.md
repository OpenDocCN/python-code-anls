# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_graph.py`

```
# 导入Permutation类从sympy.combinatorics模块，用于处理置换
# 导入symbols函数从sympy.core.symbol模块，用于创建符号变量
# 导入Matrix类从sympy.matrices模块，用于处理矩阵运算
# 导入PermutationMatrix、BlockDiagMatrix、BlockMatrix类从sympy.matrices.expressions模块，用于特定矩阵表达式的处理
from sympy.combinatorics import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.matrices.expressions import (
    PermutationMatrix, BlockDiagMatrix, BlockMatrix)


def test_connected_components():
    # 创建符号变量a到m
    a, b, c, d, e, f, g, h, i, j, k, l, m = symbols('a:m')

    # 创建一个13x13的符号矩阵M
    M = Matrix([
        [a, 0, 0, 0, b, 0, 0, 0, 0, 0, c, 0, 0],
        [0, d, 0, 0, 0, e, 0, 0, 0, 0, 0, f, 0],
        [0, 0, g, 0, 0, 0, h, 0, 0, 0, 0, 0, i],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [m, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, m, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, m, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [j, 0, 0, 0, k, 0, 0, 1, 0, 0, l, 0, 0],
        [0, j, 0, 0, 0, k, 0, 0, 1, 0, 0, l, 0],
        [0, 0, j, 0, 0, 0, k, 0, 0, 1, 0, 0, l],
        [0, 0, 0, 0, d, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 1]])

    # 找到矩阵M的连通分量
    cc = M.connected_components()
    # 断言连通分量的结果是否符合预期
    assert cc == [[0, 4, 7, 10], [1, 5, 8, 11], [2, 6, 9, 12], [3]]

    # 获取矩阵M的连通分量分解P和B
    P, B = M.connected_components_decomposition()
    # 创建一个置换p，表示连通分量的排列顺序
    p = Permutation([0, 4, 7, 10, 1, 5, 8, 11, 2, 6, 9, 12, 3])
    # 断言P是否等于由置换p创建的置换矩阵
    assert P == PermutationMatrix(p)

    # 创建四个子矩阵B0、B1、B2和B3，用于表示矩阵M的分块对角形式
    B0 = Matrix([
        [a, b, 0, c],
        [m, 1, 0, 0],
        [j, k, 1, l],
        [0, d, 0, 1]])
    B1 = Matrix([
        [d, e, 0, f],
        [m, 1, 0, 0],
        [j, k, 1, l],
        [0, d, 0, 1]])
    B2 = Matrix([
        [g, h, 0, i],
        [m, 1, 0, 0],
        [j, k, 1, l],
        [0, d, 0, 1]])
    B3 = Matrix([[1]])
    # 断言分块对角矩阵B是否等于由B0、B1、B2和B3组成的BlockDiagMatrix
    assert B == BlockDiagMatrix(B0, B1, B2, B3)


def test_strongly_connected_components():
    # 创建一个6x6的整数矩阵M
    M = Matrix([
        [11, 14, 10, 0, 15, 0],
        [0, 44, 0, 0, 45, 0],
        [1, 4, 0, 0, 5, 0],
        [0, 0, 0, 22, 0, 23],
        [0, 54, 0, 0, 55, 0],
        [0, 0, 0, 32, 0, 33]])

    # 找到矩阵M的强连通分量
    scc = M.strongly_connected_components()
    # 断言强连通分量的结果是否符合预期
    assert scc == [[1, 4], [0, 2], [3, 5]]

    # 获取矩阵M的强连通分量分解P和B
    P, B = M.strongly_connected_components_decomposition()
    # 创建一个置换p，表示强连通分量的排列顺序
    p = Permutation([1, 4, 0, 2, 3, 5])
    # 断言P是否等于由置换p创建的置换矩阵
    assert P == PermutationMatrix(p)
    # 断言B是否等于由多个子矩阵组成的BlockMatrix
    assert B == BlockMatrix([
        [
            Matrix([[44, 45], [54, 55]]),
            Matrix.zeros(2, 2),
            Matrix.zeros(2, 2)
        ],
        [
            Matrix([[14, 15], [4, 5]]),
            Matrix([[11, 10], [1, 0]]),
            Matrix.zeros(2, 2)
        ],
        [
            Matrix.zeros(2, 2),
            Matrix.zeros(2, 2),
            Matrix([[22, 23], [32, 33]])
        ]
    ])
    # 将P和B转换为显式矩阵，并断言它们与原始矩阵M的乘积相等
    P = P.as_explicit()
    B = B.as_explicit()
    assert P.T * B * P == M

    # 获取M的强连通分量分解，不使用较低索引
    P, B = M.strongly_connected_components_decomposition(lower=False)
    # 创建一个置换p，表示强连通分量的排列顺序
    p = Permutation([3, 5, 0, 2, 1, 4])
    # 断言P是否等于由置换p创建的置换矩阵
    assert P == PermutationMatrix(p)
    # 断言语句，验证矩阵 B 与给定的 BlockMatrix 对象是否相等
    assert B == BlockMatrix([
        [
            Matrix([[22, 23], [32, 33]]),  # 第一行的第一个子矩阵
            Matrix.zeros(2, 2),            # 第一行的第二个子矩阵，一个2x2的零矩阵
            Matrix.zeros(2, 2)             # 第一行的第三个子矩阵，一个2x2的零矩阵
        ],
        [
            Matrix.zeros(2, 2),            # 第二行的第一个子矩阵，一个2x2的零矩阵
            Matrix([[11, 10], [1, 0]]),    # 第二行的第二个子矩阵
            Matrix([[14, 15], [4, 5]])     # 第二行的第三个子矩阵
        ],
        [
            Matrix.zeros(2, 2),            # 第三行的第一个子矩阵，一个2x2的零矩阵
            Matrix.zeros(2, 2),            # 第三行的第二个子矩阵，一个2x2的零矩阵
            Matrix([[44, 45], [54, 55]])   # 第三行的第三个子矩阵
        ]
    ])
    
    # 将矩阵 P 转换为显式矩阵（明确表示所有元素）
    P = P.as_explicit()
    
    # 将矩阵 B 转换为显式矩阵（明确表示所有元素）
    B = B.as_explicit()
    
    # 断言语句，验证等式 P^T * B * P == M 是否成立
    assert P.T * B * P == M
```