# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hdbscan\tests\test_reachibility.py`

```
import numpy as np
import pytest

from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
)


# 定义测试函数，验证当稀疏格式不是 CSR 时是否引发错误
def test_mutual_reachability_graph_error_sparse_format():
    """Check that we raise an error if the sparse format is not CSR."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = X.T @ X  # 计算 X 的转置乘积
    np.fill_diagonal(X, 0.0)  # 将 X 对角线元素填充为 0
    X = _convert_container(X, "sparse_csc")  # 转换 X 为稀疏 CSC 格式

    err_msg = "Only sparse CSR matrices are supported"
    with pytest.raises(ValueError, match=err_msg):
        mutual_reachability_graph(X)


# 定义参数化测试函数，验证操作是否原地进行
@pytest.mark.parametrize("array_type", ["array", "sparse_csr"])
def test_mutual_reachability_graph_inplace(array_type):
    """Check that the operation is happening inplace."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = X.T @ X  # 计算 X 的转置乘积
    np.fill_diagonal(X, 0.0)  # 将 X 对角线元素填充为 0
    X = _convert_container(X, array_type)  # 根据 array_type 转换 X 的类型

    mr_graph = mutual_reachability_graph(X)  # 计算互达图

    assert id(mr_graph) == id(X)  # 断言互达图的身份与 X 相同


# 定义测试函数，验证稠密和稀疏实现得到相同结果
def test_mutual_reachability_graph_equivalence_dense_sparse():
    """Check that we get the same results for dense and sparse implementation."""
    rng = np.random.RandomState(0)
    X = rng.randn(5, 5)
    X_dense = X.T @ X  # 计算 X 的转置乘积作为稠密矩阵
    X_sparse = _convert_container(X_dense, "sparse_csr")  # 转换为稀疏 CSR 格式

    mr_graph_dense = mutual_reachability_graph(X_dense, min_samples=3)  # 计算稠密矩阵的互达图
    mr_graph_sparse = mutual_reachability_graph(X_sparse, min_samples=3)  # 计算稀疏矩阵的互达图

    assert_allclose(mr_graph_dense, mr_graph_sparse.toarray())  # 断言稠密和稀疏互达图结果相近


# 定义参数化测试函数，验证计算过程中是否保留数据类型
@pytest.mark.parametrize("array_type", ["array", "sparse_csr"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mutual_reachability_graph_preserve_dtype(array_type, dtype):
    """Check that the computation preserve dtype thanks to fused types."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = (X.T @ X).astype(dtype)  # 计算 X 的转置乘积并转换为指定 dtype
    np.fill_diagonal(X, 0.0)  # 将 X 对角线元素填充为 0
    X = _convert_container(X, array_type)  # 根据 array_type 转换 X 的类型

    assert X.dtype == dtype  # 断言 X 的数据类型为指定 dtype
    mr_graph = mutual_reachability_graph(X)  # 计算互达图
    assert mr_graph.dtype == dtype  # 断言互达图的数据类型与 X 的数据类型相同
```