# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_connected_components.py`

```
`
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_equal, assert_array_almost_equal  # 导入测试函数，用于断言测试结果
from scipy.sparse import csgraph, csr_array  # 导入 scipy.sparse 库中的 csgraph 和 csr_array 模块

# 定义测试弱连接的函数
def test_weak_connections():
    Xde = np.array([[0, 1, 0],  # 创建一个二维 NumPy 数组 Xde，表示一个有向图的邻接矩阵
                    [0, 0, 0],
                    [0, 0, 0]])

    Xsp = csgraph.csgraph_from_dense(Xde, null_value=0)  # 将密集矩阵转换为稀疏图

    for X in Xsp, Xde:  # 遍历 Xsp 和 Xde
        n_components, labels =\
            csgraph.connected_components(X, directed=True,  # 计算连接组件和标签
                                         connection='weak')

        assert_equal(n_components, 2)  # 断言连接组件数量为 2
        assert_array_almost_equal(labels, [0, 0, 1])  # 断言标签数组的值接近 [0, 0, 1]

# 定义测试强连接的函数
def test_strong_connections():
    X1de = np.array([[0, 1, 0],  # 创建两个二维 NumPy 数组 X1de 和 X2de，表示有向图的邻接矩阵
                     [0, 0, 0],
                     [0, 0, 0]])
    X2de = X1de + X1de.T  # X2de 是 X1de 的转置矩阵与自身相加的结果

    X1sp = csgraph.csgraph_from_dense(X1de, null_value=0)  # 将 X1de 转换为稀疏图
    X2sp = csgraph.csgraph_from_dense(X2de, null_value=0)  # 将 X2de 转换为稀疏图

    for X in X1sp, X1de:  # 遍历 X1sp 和 X1de
        n_components, labels =\
            csgraph.connected_components(X, directed=True,  # 计算连接组件和标签
                                         connection='strong')

        assert_equal(n_components, 3)  # 断言连接组件数量为 3
        labels.sort()  # 对标签数组进行排序
        assert_array_almost_equal(labels, [0, 1, 2])  # 断言排序后的标签数组接近 [0, 1, 2]

    for X in X2sp, X2de:  # 遍历 X2sp 和 X2de
        n_components, labels =\
            csgraph.connected_components(X, directed=True,  # 计算连接组件和标签
                                         connection='strong')

        assert_equal(n_components, 2)  # 断言连接组件数量为 2
        labels.sort()  # 对标签数组进行排序
        assert_array_almost_equal(labels, [0, 0, 1])  # 断言排序后的标签数组接近 [0, 0, 1]

# 定义第二个测试强连接的函数
def test_strong_connections2():
    X = np.array([[0, 0, 0, 0, 0, 0],  # 创建一个二维 NumPy 数组 X，表示一个有向图的邻接矩阵
                  [1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])
    n_components, labels =\
        csgraph.connected_components(X, directed=True,  # 计算连接组件和标签
                                     connection='strong')
    assert_equal(n_components, 5)  # 断言连接组件数量为 5
    labels.sort()  # 对标签数组进行排序
    assert_array_almost_equal(labels, [0, 1, 2, 2, 3, 4])  # 断言排序后的标签数组接近 [0, 1, 2, 2, 3, 4]

# 定义测试弱连接的函数
def test_weak_connections2():
    X = np.array([[0, 0, 0, 0, 0, 0],  # 创建一个二维 NumPy 数组 X，表示一个有向图的邻接矩阵
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])
    n_components, labels =\
        csgraph.connected_components(X, directed=True,  # 计算连接组件和标签
                                     connection='weak')
    assert_equal(n_components, 2)  # 断言连接组件数量为 2
    labels.sort()  # 对标签数组进行排序
    assert_array_almost_equal(labels, [0, 0, 1, 1, 1, 1])  # 断言排序后的标签数组接近 [0, 0, 1, 1, 1, 1]

# 定义 Regression 测试函数
def test_ticket1876():
    # 回归测试：这在原始实现中失败了
    # 应该有两个强连通分量；之前只给出了一个
    g = np.array([[0, 1, 1, 0],  # 创建一个二维 NumPy 数组 g，表示一个有向图的邻接矩阵
                  [1, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])
    n_components, labels = csgraph.connected_components(g, connection='strong')  # 计算连接组件和标签

    assert_equal(n_components, 2)  # 断言连接组件数量为 2
    assert_equal(labels[0], labels[1])  # 断言标签数组的第一个和第二个元素相等
    assert_equal(labels[2], labels[3])  # 断言标签数组的第三个和第四个元素相等
    # 创建一个4x4的全1矩阵
    g = np.ones((4, 4))
    # 使用csgraph.connected_components函数计算连接组件数和标签
    n_components, labels = csgraph.connected_components(g)
    # 断言：确保连接组件数为1，否则会触发异常
    assert_equal(n_components, 1)
# 测试函数，用于验证无向图的情况下的整数索引是否正常工作
def test_int64_indices_undirected():
    # 参考 https://github.com/scipy/scipy/issues/18716
    # 创建一个压缩稀疏行矩阵 (Compressed Sparse Row Matrix, CSR) 的数组 g，
    # 包含一个非零元素 [1]，以及一个索引数组 [[0], [1]]，索引数组的数据类型为 np.int64
    g = csr_array(([1], np.array([[0], [1]], dtype=np.int64)), shape=(2, 2))
    # 断言 g 的索引数组的数据类型为 np.int64
    assert g.indices.dtype == np.int64
    # 使用 csgraph.connected_components 函数计算 g 的连通分量，设置为无向图
    n, labels = csgraph.connected_components(g, directed=False)
    # 断言只有一个连通分量
    assert n == 1
    # 断言 labels 的值接近于 [0, 0]
    assert_array_almost_equal(labels, [0, 0])


# 测试函数，用于验证有向图的情况下的整数索引是否正常工作
def test_int64_indices_directed():
    # 参考 https://github.com/scipy/scipy/issues/18716
    # 创建一个压缩稀疏行矩阵 (Compressed Sparse Row Matrix, CSR) 的数组 g，
    # 包含一个非零元素 [1]，以及一个索引数组 [[0], [1]]，索引数组的数据类型为 np.int64
    g = csr_array(([1], np.array([[0], [1]], dtype=np.int64)), shape=(2, 2))
    # 断言 g 的索引数组的数据类型为 np.int64
    assert g.indices.dtype == np.int64
    # 使用 csgraph.connected_components 函数计算 g 的连通分量，设置为有向图，连接类型为强连接
    n, labels = csgraph.connected_components(g, directed=True, connection='strong')
    # 断言有两个连通分量
    assert n == 2
    # 断言 labels 的值接近于 [1, 0]
    assert_array_almost_equal(labels, [1, 0])
```