# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_reordering.py`

```
# 导入必要的库：numpy 和相关的测试工具
import numpy as np
from numpy.testing import assert_equal
# 导入用于稀疏图操作的函数
from scipy.sparse.csgraph import reverse_cuthill_mckee, structural_rank
# 导入稀疏矩阵的三种格式
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix


# 定义测试函数：测试 reverse_cuthill_mckee 函数
def test_graph_reverse_cuthill_mckee():
    # 创建一个稀疏矩阵 A
    A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 1, 0, 1],
                  [0, 1, 1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 1, 0, 1]], dtype=int)
    
    # 将矩阵 A 转换为 CSR 格式的稀疏矩阵
    graph = csr_matrix(A)
    # 使用 reverse_cuthill_mckee 算法得到置换结果
    perm = reverse_cuthill_mckee(graph)
    # 正确的置换顺序
    correct_perm = np.array([6, 3, 7, 5, 1, 2, 4, 0])
    # 断言置换结果与正确结果相等
    assert_equal(perm, correct_perm)
    
    # 测试使用 int64 类型的索引
    graph.indices = graph.indices.astype('int64')
    graph.indptr = graph.indptr.astype('int64')
    perm = reverse_cuthill_mckee(graph, True)
    # 再次断言置换结果与正确结果相等
    assert_equal(perm, correct_perm)


# 定义测试函数：测试 reverse_cuthill_mckee 函数（顺序自定义）
def test_graph_reverse_cuthill_mckee_ordering():
    # 定义 COO 格式的稀疏矩阵
    data = np.ones(63, dtype=int)
    rows = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 
                     2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                     6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9,
                     9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 
                     12, 12, 12, 13, 13, 13, 13, 14, 14, 14,
                     14, 15, 15, 15, 15, 15])
    cols = np.array([0, 2, 5, 8, 10, 1, 3, 9, 11, 0, 2,
                     7, 10, 1, 3, 11, 4, 6, 12, 14, 0, 7, 13, 
                     15, 4, 6, 14, 2, 5, 7, 15, 0, 8, 10, 13,
                     1, 9, 11, 0, 2, 8, 10, 15, 1, 3, 9, 11,
                     4, 12, 14, 5, 8, 13, 15, 4, 6, 12, 14,
                     5, 7, 10, 13, 15])
    graph = coo_matrix((data, (rows, cols))).tocsr()
    # 使用 reverse_cuthill_mckee 算法得到置换结果
    perm = reverse_cuthill_mckee(graph)
    # 正确的置换顺序
    correct_perm = np.array([12, 14, 4, 6, 10, 8, 2, 15,
                             0, 13, 7, 5, 9, 11, 1, 3])
    # 断言置换结果与正确结果相等
    assert_equal(perm, correct_perm)


# 定义测试函数：测试 structural_rank 函数
def test_graph_structural_rank():
    # 测试方阵 #1
    A = csc_matrix([[1, 1, 0], 
                    [1, 0, 1],
                    [0, 1, 0]])
    # 断言结构秩的计算结果为 3
    assert_equal(structural_rank(A), 3)
    
    # 测试方阵 #2
    rows = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    cols = np.array([0, 1, 2, 3, 4, 2, 5, 2, 6, 0, 1, 3, 5, 6, 7, 4, 5, 5, 6, 2, 6, 2, 4])
    data = np.ones_like(rows)
    B = coo_matrix((data, (rows, cols)), shape=(8, 8))
    # 断言结构秩的计算结果为 6
    assert_equal(structural_rank(B), 6)
    
    # 测试非方阵
    C = csc_matrix([[1, 0, 2, 0], 
                    [2, 0, 4, 0]])
    # 断言结构秩的计算结果为 2
    assert_equal(structural_rank(C), 2)
    
    # 测试长方阵
    # 断言结构秩的计算结果为 2
    assert_equal(structural_rank(C.T), 2)
```