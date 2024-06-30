# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_graph.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库
from scipy.sparse.csgraph import connected_components  # 从 scipy.sparse.csgraph 模块导入 connected_components 函数

from sklearn.metrics.pairwise import pairwise_distances  # 导入 pairwise_distances 函数
from sklearn.neighbors import kneighbors_graph  # 导入 kneighbors_graph 函数
from sklearn.utils.graph import _fix_connected_components  # 导入 _fix_connected_components 函数


def test_fix_connected_components():
    # 测试 _fix_connected_components 函数是否将组件数目减少到 1
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")

    # 计算原始图的连通组件数和标签
    n_connected_components, labels = connected_components(graph)
    assert n_connected_components > 1  # 断言原始图的连通组件数大于 1

    # 调用 _fix_connected_components 函数修复图
    graph = _fix_connected_components(X, graph, n_connected_components, labels)

    # 再次计算修复后图的连通组件数和标签
    n_connected_components, labels = connected_components(graph)
    assert n_connected_components == 1  # 断言修复后图的连通组件数为 1


def test_fix_connected_components_precomputed():
    # 测试 _fix_connected_components 函数是否接受预计算的距离矩阵作为输入
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")

    # 计算原始图的连通组件数和标签
    n_connected_components, labels = connected_components(graph)
    assert n_connected_components > 1  # 断言原始图的连通组件数大于 1

    # 计算点对距离矩阵
    distances = pairwise_distances(X)
    # 调用 _fix_connected_components 函数修复图，使用预计算的距离矩阵
    graph = _fix_connected_components(
        distances, graph, n_connected_components, labels, metric="precomputed"
    )

    # 再次计算修复后图的连通组件数和标签
    n_connected_components, labels = connected_components(graph)
    assert n_connected_components == 1  # 断言修复后图的连通组件数为 1

    # 但是 _fix_connected_components 函数不能处理预计算的邻居图
    with pytest.raises(RuntimeError, match="does not work with a sparse"):
        _fix_connected_components(
            graph, graph, n_connected_components, labels, metric="precomputed"
        )


def test_fix_connected_components_wrong_mode():
    # 测试当模式字符串不正确时是否引发错误
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")
    n_connected_components, labels = connected_components(graph)

    # 断言调用 _fix_connected_components 函数时会引发 ValueError，并且错误消息包含 "Unknown mode"
    with pytest.raises(ValueError, match="Unknown mode"):
        graph = _fix_connected_components(
            X, graph, n_connected_components, labels, mode="foo"
        )


def test_fix_connected_components_connectivity_mode():
    # 测试连接模式是否会用 1 填充新连接
    X = np.array([0, 1, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=1, mode="connectivity")
    n_connected_components, labels = connected_components(graph)

    # 调用 _fix_connected_components 函数修复图，连接模式下应当用 1 填充新连接
    graph = _fix_connected_components(
        X, graph, n_connected_components, labels, mode="connectivity"
    )

    # 断言修复后图的数据是否全部为 1
    assert np.all(graph.data == 1)


def test_fix_connected_components_distance_mode():
    # 测试距离模式是否不会用 1 填充新连接
    X = np.array([0, 1, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=1, mode="distance")

    # 断言原始图的数据是否全部为 1
    assert np.all(graph.data == 1)

    # 计算原始图的连通组件数
    n_connected_components, labels = connected_components(graph)
    # 调用函数 _fix_connected_components 对图数据进行处理，以修复连接组件
    graph = _fix_connected_components(
        X, graph, n_connected_components, labels, mode="distance"
    )
    # 断言检查：确保图数据中并非所有的数据都等于1
    assert not np.all(graph.data == 1)
```