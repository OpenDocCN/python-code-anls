# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_matching.py`

```
from itertools import product  # 导入 itertools 模块中的 product 函数

import numpy as np  # 导入 numpy 库并简写为 np
from numpy.testing import assert_array_equal, assert_equal  # 导入 numpy.testing 模块中的两个函数
import pytest  # 导入 pytest 库

from scipy.sparse import csr_matrix, coo_matrix, diags  # 从 scipy.sparse 库中导入稀疏矩阵相关类
from scipy.sparse.csgraph import (
    maximum_bipartite_matching, min_weight_full_bipartite_matching
)  # 从 scipy.sparse.csgraph 模块中导入最大二分图匹配和最小权重全二分图匹配函数


def test_maximum_bipartite_matching_raises_on_dense_input():
    with pytest.raises(TypeError):  # 使用 pytest 来检测是否会抛出 TypeError 异常
        graph = np.array([[0, 1], [0, 0]])  # 创建一个 numpy 数组作为二分图的邻接矩阵
        maximum_bipartite_matching(graph)  # 调用最大二分图匹配函数对该图进行匹配


def test_maximum_bipartite_matching_empty_graph():
    graph = csr_matrix((0, 0))  # 创建一个空的 CSR 稀疏矩阵作为一个空图
    x = maximum_bipartite_matching(graph, perm_type='row')  # 对空图进行最大二分图匹配，以行排列顺序进行排列
    y = maximum_bipartite_matching(graph, perm_type='column')  # 对空图进行最大二分图匹配，以列排列顺序进行排列
    expected_matching = np.array([])  # 期望的匹配结果是一个空的 numpy 数组
    assert_array_equal(expected_matching, x)  # 断言行排列的匹配结果与期望一致
    assert_array_equal(expected_matching, y)  # 断言列排列的匹配结果与期望一致


def test_maximum_bipartite_matching_empty_left_partition():
    graph = csr_matrix((2, 0))  # 创建一个左分区为空的 CSR 稀疏矩阵
    x = maximum_bipartite_matching(graph, perm_type='row')  # 对左分区为空的图进行最大二分图匹配，以行排列顺序进行排列
    y = maximum_bipartite_matching(graph, perm_type='column')  # 对左分区为空的图进行最大二分图匹配，以列排列顺序进行排列
    assert_array_equal(np.array([]), x)  # 断言行排列的匹配结果是一个空的 numpy 数组
    assert_array_equal(np.array([-1, -1]), y)  # 断言列排列的匹配结果是 [-1, -1] 的 numpy 数组


def test_maximum_bipartite_matching_empty_right_partition():
    graph = csr_matrix((0, 3))  # 创建一个右分区为空的 CSR 稀疏矩阵
    x = maximum_bipartite_matching(graph, perm_type='row')  # 对右分区为空的图进行最大二分图匹配，以行排列顺序进行排列
    y = maximum_bipartite_matching(graph, perm_type='column')  # 对右分区为空的图进行最大二分图匹配，以列排列顺序进行排列
    assert_array_equal(np.array([-1, -1, -1]), x)  # 断言行排列的匹配结果是 [-1, -1, -1] 的 numpy 数组
    assert_array_equal(np.array([]), y)  # 断言列排列的匹配结果是一个空的 numpy 数组


def test_maximum_bipartite_matching_graph_with_no_edges():
    graph = csr_matrix((2, 2))  # 创建一个没有边的小型 CSR 稀疏矩阵
    x = maximum_bipartite_matching(graph, perm_type='row')  # 对没有边的图进行最大二分图匹配，以行排列顺序进行排列
    y = maximum_bipartite_matching(graph, perm_type='column')  # 对没有边的图进行最大二分图匹配，以列排列顺序进行排列
    assert_array_equal(np.array([-1, -1]), x)  # 断言行排列的匹配结果是 [-1, -1] 的 numpy 数组
    assert_array_equal(np.array([-1, -1]), y)  # 断言列排列的匹配结果是 [-1, -1] 的 numpy 数组


def test_maximum_bipartite_matching_graph_that_causes_augmentation():
    # 在这个图中，列 1 最初被分配给行 1，但应重新分配以为行 2 腾出空间。
    graph = csr_matrix([[1, 1], [1, 0]])  # 创建一个特定的 CSR 稀疏矩阵
    x = maximum_bipartite_matching(graph, perm_type='column')  # 对该图进行最大二分图匹配，以列排列顺序进行排列
    y = maximum_bipartite_matching(graph, perm_type='row')  # 对该图进行最大二分图匹配，以行排列顺序进行排列
    expected_matching = np.array([1, 0])  # 期望的匹配结果是 [1, 0] 的 numpy 数组
    assert_array_equal(expected_matching, x)  # 断言列排列的匹配结果与期望一致
    assert_array_equal(expected_matching, y)  # 断言行排列的匹配结果与期望一致


def test_maximum_bipartite_matching_graph_with_more_rows_than_columns():
    graph = csr_matrix([[1, 1], [1, 0], [0, 1]])  # 创建一个具有更多行数的 CSR 稀疏矩阵
    x = maximum_bipartite_matching(graph, perm_type='column')  # 对该图进行最大二分图匹配，以列排列顺序进行排列
    y = maximum_bipartite_matching(graph, perm_type='row')  # 对该图进行最大二分图匹配，以行排列顺序进行排列
    assert_array_equal(np.array([0, -1, 1]), x)  # 断言列排列的匹配结果是 [0, -1, 1] 的 numpy 数组
    assert_array_equal(np.array([0, 2]), y)  # 断言行排列的匹配结果是 [0, 2] 的 numpy 数组


def test_maximum_bipartite_matching_graph_with_more_columns_than_rows():
    graph = csr_matrix([[1, 1, 0], [0, 0, 1]])  # 创建一个具有更多列数的 CSR 稀疏矩阵
    x = maximum_bipartite_matching(graph, perm_type='column')  # 对该图进行最大二分图匹配，以列排列顺序进行排列
    y = maximum_bipartite_matching(graph, perm_type='row')  # 对该图进行最大二分图匹配，以行排列顺序进行排列
    assert_array_equal(np.array([0, 2]), x)  # 断言列排列的匹配结果是 [0, 2] 的 numpy 数组
    assert_array_equal(np.array([0, -1, 1]), y)  # 断言行排列的匹配结果是 [0, -1, 1] 的 numpy 数组


def test_maximum_bipartite_matching_explicit_zeros_count_as_edges():
    data = [0, 0]  # 数据数组
    indices = [1, 0]  # 索引数组
    indptr = [0, 1, 2]  # 行指针数组
    # 使用提供的数据创建一个稀疏矩阵，采用 CSR (Compressed Sparse Row) 存储格式
    graph = csr_matrix((data, indices, indptr), shape=(2, 2))
    # 对于给定的稀疏图，执行最大二分图匹配算法，以行为基准进行排列
    x = maximum_bipartite_matching(graph, perm_type='row')
    # 对于给定的稀疏图，执行最大二分图匹配算法，以列为基准进行排列
    y = maximum_bipartite_matching(graph, perm_type='column')
    # 预期的匹配结果，用于断言检查行匹配结果是否符合预期
    expected_matching = np.array([1, 0])
    # 检查行匹配结果是否与预期匹配数组相等，如果不相等则会引发 AssertionError
    assert_array_equal(expected_matching, x)
    # 检查列匹配结果是否与预期匹配数组相等，如果不相等则会引发 AssertionError
    assert_array_equal(expected_matching, y)
def test_maximum_bipartite_matching_feasibility_of_result():
    # 这是用于 GitHub 问题 #11458 的回归测试
    data = np.ones(50, dtype=int)  # 创建一个包含50个整数1的NumPy数组
    indices = [11, 12, 19, 22, 23, 5, 22, 3, 8, 10, 5, 6, 11, 12, 13, 5, 13,
               14, 20, 22, 3, 15, 3, 13, 14, 11, 12, 19, 22, 23, 5, 22, 3, 8,
               10, 5, 6, 11, 12, 13, 5, 13, 14, 20, 22, 3, 15, 3, 13, 14]  # 索引列表
    indptr = [0, 5, 7, 10, 10, 15, 20, 22, 22, 23, 25, 30, 32, 35, 35, 40, 45,
              47, 47, 48, 50]  # 指针列表
    graph = csr_matrix((data, indices, indptr), shape=(20, 25))  # 创建一个稀疏矩阵 graph
    x = maximum_bipartite_matching(graph, perm_type='row')  # 调用函数进行最大二分匹配，按行排列
    y = maximum_bipartite_matching(graph, perm_type='column')  # 调用函数进行最大二分匹配，按列排列
    assert (x != -1).sum() == 13  # 断言：行匹配结果中不等于-1的数量为13
    assert (y != -1).sum() == 13  # 断言：列匹配结果中不等于-1的数量为13
    # 确保匹配中的每个元素实际上是图中的一条边
    for u, v in zip(range(graph.shape[0]), y):
        if v != -1:
            assert graph[u, v]
    for u, v in zip(x, range(graph.shape[1])):
        if u != -1:
            assert graph[u, v]


def test_matching_large_random_graph_with_one_edge_incident_to_each_vertex():
    np.random.seed(42)  # 设置随机种子
    A = diags(np.ones(25), offsets=0, format='csr')  # 创建对角矩阵 A
    rand_perm = np.random.permutation(25)  # 生成随机排列
    rand_perm2 = np.random.permutation(25)  # 生成另一个随机排列

    Rrow = np.arange(25)  # 创建从0到24的行索引
    Rcol = rand_perm  # 使用随机排列作为列索引
    Rdata = np.ones(25, dtype=int)  # 创建包含25个整数1的数据数组
    Rmat = coo_matrix((Rdata, (Rrow, Rcol))).tocsr()  # 创建稀疏矩阵 Rmat

    Crow = rand_perm2  # 使用另一个随机排列作为行索引
    Ccol = np.arange(25)  # 创建从0到24的列索引
    Cdata = np.ones(25, dtype=int)  # 创建包含25个整数1的数据数组
    Cmat = coo_matrix((Cdata, (Crow, Ccol))).tocsr()  # 创建稀疏矩阵 Cmat
    # 随机置换单位矩阵
    B = Rmat * A * Cmat  # 计算 B = Rmat * A * Cmat

    # 按行置换
    perm = maximum_bipartite_matching(B, perm_type='row')  # 调用函数进行最大二分匹配，按行排列
    Rrow = np.arange(25)  # 创建从0到24的行索引
    Rcol = perm  # 使用匹配结果作为列索引
    Rdata = np.ones(25, dtype=int)  # 创建包含25个整数1的数据数组
    Rmat = coo_matrix((Rdata, (Rrow, Rcol))).tocsr()  # 创建稀疏矩阵 Rmat
    C1 = Rmat * B  # 计算 C1 = Rmat * B

    # 按列置换
    perm2 = maximum_bipartite_matching(B, perm_type='column')  # 调用函数进行最大二分匹配，按列排列
    Crow = perm2  # 使用匹配结果作为行索引
    Ccol = np.arange(25)  # 创建从0到24的列索引
    Cdata = np.ones(25, dtype=int)  # 创建包含25个整数1的数据数组
    Cmat = coo_matrix((Cdata, (Crow, Ccol))).tocsr()  # 创建稀疏矩阵 Cmat
    C2 = B * Cmat  # 计算 C2 = B * Cmat

    # 应该得到单位矩阵
    assert_equal(any(C1.diagonal() == 0), False)  # 断言：C1的对角线上没有零元素
    assert_equal(any(C2.diagonal() == 0), False)  # 断言：C2的对角线上没有零元素


@pytest.mark.parametrize('num_rows,num_cols', [(0, 0), (2, 0), (0, 3)])
def test_min_weight_full_matching_trivial_graph(num_rows, num_cols):
    biadjacency_matrix = csr_matrix((num_cols, num_rows))  # 创建一个稀疏的二分邻接矩阵
    row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency_matrix)  # 调用函数进行最小权重全匹配
    assert len(row_ind) == 0  # 断言：行索引列表为空
    assert len(col_ind) == 0  # 断言：列索引列表为空


@pytest.mark.parametrize('biadjacency_matrix',
                         [
                            [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
                            [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
                            [[1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]],
                            [[1, 0, 0], [2, 0, 0]],
                            [[0, 1, 0], [0, 2, 0]],
                            [[1, 0], [2, 0], [5, 0]]
                         ])
# 当传入的双分配问题不可解时，使用 pytest 的上下文管理器检查是否引发 ValueError 异常
def test_min_weight_full_matching_infeasible_problems(biadjacency_matrix):
    with pytest.raises(ValueError):
        min_weight_full_bipartite_matching(csr_matrix(biadjacency_matrix))


# GitHub 问题 #17269 的回归测试，检查是否引发 ValueError 并包含特定匹配错误消息
def test_min_weight_full_matching_large_infeasible():
    with pytest.raises(ValueError, match='no full matching exists'):
        min_weight_full_bipartite_matching(csr_matrix(a))


# 显式零值引发警告的测试，使用 pytest 的上下文管理器检查是否引发 UserWarning
def test_explicit_zero_causes_warning():
    with pytest.warns(UserWarning):
        # 创建一个稀疏双分配矩阵 biadjacency_matrix
        biadjacency_matrix = csr_matrix(((2, 0, 3), (0, 1, 1), (0, 2, 3)))
        # 调用最小权重完全二分图匹配函数，期望触发警告
        min_weight_full_bipartite_matching(biadjacency_matrix)


# 线性和分配求解器的通用测试，确保在 scipy.optimize.linear_sum_assignment 上可以依赖相同的测试
def linear_sum_assignment_assertions(
    solver, array_type, sign, test_case
):
    # 解包测试用例
    cost_matrix, expected_cost = test_case
    maximize = sign == -1
    # 调整成本矩阵和预期成本为符号乘以相应数组类型的值
    cost_matrix = sign * array_type(cost_matrix)
    expected_cost = sign * np.array(expected_cost)

    # 第一次调用求解器，解决最小权重完全二分图匹配问题
    row_ind, col_ind = solver(cost_matrix, maximize=maximize)
    # 断言行索引已排序
    assert_array_equal(row_ind, np.sort(row_ind))
    # 断言预期成本等于排序后的成本矩阵的扁平化数组
    assert_array_equal(expected_cost,
                       np.array(cost_matrix[row_ind, col_ind]).flatten())

    # 转置成本矩阵，再次调用求解器
    cost_matrix = cost_matrix.T
    row_ind, col_ind = solver(cost_matrix, maximize=maximize)
    # 断言行索引已排序
    assert_array_equal(row_ind, np.sort(row_ind))
    # 断言排序后的预期成本等于排序后的成本矩阵的扁平化数组
    assert_array_equal(np.sort(expected_cost),
                       np.sort(np.array(
                           cost_matrix[row_ind, col_ind])).flatten())


# 使用 product 生成线性和分配测试的参数化测试用例
linear_sum_assignment_test_cases = product(
    [-1, 1],
    [
        # 方阵
        ([[400, 150, 400],
          [400, 450, 600],
          [300, 225, 300]],
         [150, 400, 300]),

        # 长方形变体
        ([[400, 150, 400, 1],
          [400, 450, 600, 2],
          [300, 225, 300, 3]],
         [150, 2, 300]),

        ([[10, 10, 8],
          [9, 8, 1],
          [9, 7, 4]],
         [10, 1, 7]),

        # 方阵
        ([[10, 10, 8, 11],
          [9, 8, 1, 1],
          [9, 7, 4, 10]],
         [10, 1, 4]),

        # 长方形变体
        ([[10, float("inf"), float("inf")],
          [float("inf"), float("inf"), 1],
          [float("inf"), 7, float("inf")]],
         [10, 1, 7])
    ])


# 参数化测试，测试最小权重完全匹配小输入
@pytest.mark.parametrize('sign,test_case', linear_sum_assignment_test_cases)
def test_min_weight_full_matching_small_inputs(sign, test_case):
    linear_sum_assignment_assertions(
        min_weight_full_bipartite_matching, csr_matrix, sign, test_case)
```