# `D:\src\scipysrc\scipy\scipy\sparse\csgraph\tests\test_conversions.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_array_almost_equal  # 导入 NumPy 测试模块中的数组近似相等断言函数
from scipy.sparse import csr_matrix  # 导入 SciPy 稀疏矩阵模块中的 CSR 矩阵类
from scipy.sparse.csgraph import csgraph_from_dense, csgraph_to_dense  # 导入 SciPy 稀疏矩阵图算法模块中的函数


def test_csgraph_from_dense():
    np.random.seed(1234)  # 设置随机数种子以保证结果可重复
    G = np.random.random((10, 10))  # 创建一个 10x10 的随机数矩阵
    some_nulls = (G < 0.4)  # 找出小于 0.4 的元素位置
    all_nulls = (G < 0.8)  # 找出小于 0.8 的元素位置

    for null_value in [0, np.nan, np.inf]:
        G[all_nulls] = null_value  # 将所有小于 0.8 的元素替换为指定的 null_value
        with np.errstate(invalid="ignore"):
            G_csr = csgraph_from_dense(G, null_value=0)  # 调用 csgraph_from_dense 将稠密矩阵转换为 CSR 格式的稀疏图

        G[all_nulls] = 0  # 将所有小于 0.8 的元素重置为 0
        assert_array_almost_equal(G, G_csr.toarray())  # 使用断言验证转换后的稀疏图与原始矩阵近似相等

    for null_value in [np.nan, np.inf]:
        G[all_nulls] = 0  # 将所有小于 0.8 的元素重置为 0
        G[some_nulls] = null_value  # 将一部分小于 0.4 的元素替换为指定的 null_value
        with np.errstate(invalid="ignore"):
            G_csr = csgraph_from_dense(G, null_value=0)  # 再次调用 csgraph_from_dense 进行转换

        G[all_nulls] = 0  # 将所有小于 0.8 的元素重置为 0
        assert_array_almost_equal(G, G_csr.toarray())  # 使用断言验证转换后的稀疏图与原始矩阵近似相等


def test_csgraph_to_dense():
    np.random.seed(1234)  # 设置随机数种子以保证结果可重复
    G = np.random.random((10, 10))  # 创建一个 10x10 的随机数矩阵
    nulls = (G < 0.8)  # 找出小于 0.8 的元素位置
    G[nulls] = np.inf  # 将所有小于 0.8 的元素替换为无穷大

    G_csr = csgraph_from_dense(G)  # 使用 csgraph_from_dense 将稠密矩阵转换为 CSR 格式的稀疏图

    for null_value in [0, 10, -np.inf, np.inf]:
        G[nulls] = null_value  # 将所有小于 0.8 的元素替换为指定的 null_value
        assert_array_almost_equal(G, csgraph_to_dense(G_csr, null_value))  # 使用断言验证转换后的稀疏图与原始矩阵近似相等


def test_multiple_edges():
    # 创建一个带有偶数个元素的随机方阵
    np.random.seed(1234)  # 设置随机数种子以保证结果可重复
    X = np.random.random((10, 10))  # 创建一个 10x10 的随机数矩阵
    Xcsr = csr_matrix(X)  # 将稠密矩阵转换为 CSR 格式的稀疏矩阵

    # 将每隔一个列的索引复制一次，创建多重边
    Xcsr.indices[::2] = Xcsr.indices[1::2]

    # 常规的稀疏矩阵转稠密矩阵操作会将重复边的权重求和
    Xdense = Xcsr.toarray()  # 将 CSR 格式的稀疏矩阵转换为稠密矩阵
    assert_array_almost_equal(Xdense[:, 1::2],
                              X[:, ::2] + X[:, 1::2])  # 使用断言验证求和结果与原始矩阵的期望值近似相等

    # csgraph_to_dense 选择每个重复边的最小权重
    Xdense = csgraph_to_dense(Xcsr)  # 使用 csgraph_to_dense 将 CSR 格式的稀疏矩阵转换为稠密矩阵
    assert_array_almost_equal(Xdense[:, 1::2],
                              np.minimum(X[:, ::2], X[:, 1::2]))  # 使用断言验证最小化结果与原始矩阵的期望值近似相等
```