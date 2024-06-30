# `D:\src\scipysrc\scikit-learn\sklearn\manifold\tests\test_locally_linear.py`

```
# 导入所需的模块和函数
from itertools import product  # 从itertools模块导入product函数

import numpy as np  # 导入NumPy库并使用np作为别名
import pytest  # 导入pytest库
from scipy import linalg  # 从SciPy库导入linalg子模块

from sklearn import manifold, neighbors  # 从sklearn库导入manifold和neighbors模块
from sklearn.datasets import make_blobs  # 从sklearn库导入make_blobs函数
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph  # 从sklearn库导入barycenter_kneighbors_graph函数
from sklearn.utils._testing import (  # 从sklearn库导入_testing模块中的若干函数
    assert_allclose,
    assert_array_equal,
    ignore_warnings,
)

eigen_solvers = ["dense", "arpack"]  # 定义一个包含两种解法的列表

# ----------------------------------------------------------------------
# 测试实用函数

def test_barycenter_kneighbors_graph(global_dtype):
    # 创建一个包含三个点的二维NumPy数组X
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]], dtype=global_dtype)

    # 调用barycenter_kneighbors_graph函数生成一个稀疏图graph
    graph = barycenter_kneighbors_graph(X, 1)
    # 预期的稀疏图expected_graph
    expected_graph = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=global_dtype
    )

    # 断言graph的数据类型与global_dtype相同
    assert graph.dtype == global_dtype

    # 断言graph的稀疏数组形式与expected_graph相近
    assert_allclose(graph.toarray(), expected_graph)

    # 再次调用barycenter_kneighbors_graph函数生成另一个稀疏图graph
    graph = barycenter_kneighbors_graph(X, 2)
    # 断言每列的和为一
    assert_allclose(np.sum(graph.toarray(), axis=1), np.ones(3))
    # 计算预测值pred，并断言其与X的差异较小
    pred = np.dot(graph.toarray(), X)
    assert linalg.norm(pred - X) / X.shape[0] < 1


# ----------------------------------------------------------------------
# 通过在一些流形上计算重建误差来测试LLE

def test_lle_simple_grid(global_dtype):
    # 注意：由于ARPACK在数值上不稳定，因此这个测试会对一些随机种子失败。
    #       我们选择42因为这些测试通过了。
    #       对于arm64平台，2会导致测试失败。
    # TODO: 重写这个测试，使其对随机种子的敏感性更低，不考虑平台。

    rng = np.random.RandomState(42)  # 使用种子42创建一个随机数生成器rng

    # 创建一个二维中等距点的网格X，并加上小量随机扰动
    X = np.array(list(product(range(5), repeat=2)))
    X = X + 1e-10 * rng.uniform(size=X.shape)
    X = X.astype(global_dtype, copy=False)

    n_components = 2  # 定义流形的维数为2
    clf = manifold.LocallyLinearEmbedding(
        n_neighbors=5, n_components=n_components, random_state=rng
    )
    tol = 0.1  # 定义容差为0.1

    N = barycenter_kneighbors_graph(X, clf.n_neighbors).toarray()
    reconstruction_error = linalg.norm(np.dot(N, X) - X, "fro")
    # 断言重建误差小于容差tol
    assert reconstruction_error < tol

    # 遍历eigen_solvers中的每个求解器solver
    for solver in eigen_solvers:
        clf.set_params(eigen_solver=solver)
        clf.fit(X)
        # 断言嵌入矩阵的列数等于n_components
        assert clf.embedding_.shape[1] == n_components
        # 计算重建误差并断言其小于容差tol
        reconstruction_error = (
            linalg.norm(np.dot(N, clf.embedding_) - clf.embedding_, "fro") ** 2
        )
        assert reconstruction_error < tol
        # 断言clf.reconstruction_error_与reconstruction_error相近
        assert_allclose(clf.reconstruction_error_, reconstruction_error, atol=1e-1)

    # 使用transform方法重新嵌入带有噪声的版本X_reembedded
    noise = rng.randn(*X.shape).astype(global_dtype, copy=False) / 100
    X_reembedded = clf.transform(X + noise)
    # 断言X_reembedded与clf.embedding_的差异较小
    assert linalg.norm(X_reembedded - clf.embedding_) < tol


@pytest.mark.parametrize("method", ["standard", "hessian", "modified", "ltsa"])
@pytest.mark.parametrize("solver", eigen_solvers)
def test_lle_manifold(global_dtype, method, solver):
    rng = np.random.RandomState(0)
    # 使用种子0初始化随机数生成器rng，以便结果可重复

    # 创建一个二维数组X，包含了0到17的所有整数对的笛卡尔积，并在每个元素后添加平方项除以18
    X = np.array(list(product(np.arange(18), repeat=2)))
    X = np.c_[X, X[:, 0] ** 2 / 18]

    # 对数组X中的所有元素加上微小的扰动，以防止出现除以零或其他数值问题
    X = X + 1e-10 * rng.uniform(size=X.shape)

    # 将数组X的数据类型转换为全局变量global_dtype指定的数据类型，无需复制数据
    X = X.astype(global_dtype, copy=False)

    # 设定局部线性嵌入的目标维度数
    n_components = 2

    # 使用局部线性嵌入方法创建分类器clf，设定相关参数
    clf = manifold.LocallyLinearEmbedding(
        n_neighbors=6, n_components=n_components, method=method, random_state=0
    )

    # 根据不同的方法设定误差容限tol
    tol = 1.5 if method == "standard" else 3

    # 使用barycenter_kneighbors_graph函数计算数据X的重心k近邻图的邻接矩阵N
    N = barycenter_kneighbors_graph(X, clf.n_neighbors).toarray()

    # 计算重构误差，即原始数据X经过N的变换后与X之间的范数差
    reconstruction_error = linalg.norm(np.dot(N, X) - X)

    # 断言重构误差小于设定的容限tol
    assert reconstruction_error < tol

    # 设置分类器clf的特定参数eigen_solver为solver，重新拟合数据X
    clf.set_params(eigen_solver=solver)
    clf.fit(X)

    # 断言嵌入数据的维度与设定的目标维度n_components相等
    assert clf.embedding_.shape[1] == n_components

    # 计算重构误差的Frobenius范数平方，衡量嵌入数据经过N的变换后与嵌入数据之间的误差
    reconstruction_error = (
        linalg.norm(np.dot(N, clf.embedding_) - clf.embedding_, "fro") ** 2
    )

    # 构建包含求解器solver和方法method信息的详细说明字符串
    details = "solver: %s, method: %s" % (solver, method)

    # 断言重构误差小于设定的容限tol，并输出详细说明字符串作为断言失败时的额外信息
    assert reconstruction_error < tol, details

    # 断言分类器clf的重构误差与计算得到的重构误差之间的差异在一定容限内，并输出详细说明字符串
    assert (
        np.abs(clf.reconstruction_error_ - reconstruction_error)
        < tol * reconstruction_error
    ), details
# 检查 LocallyLinearEmbedding 在 Pipeline 中的正常工作
# 仅检查是否没有引发错误。
# TODO 确认它实际上执行了有用的操作
def test_pipeline():
    # 导入必要的模块
    from sklearn import datasets, pipeline
    
    # 创建随机数据集
    X, y = datasets.make_blobs(random_state=0)
    
    # 定义 Pipeline 包括两个步骤：数据过滤和 K 近邻分类器
    clf = pipeline.Pipeline(
        [
            ("filter", manifold.LocallyLinearEmbedding(random_state=0)),
            ("clf", neighbors.KNeighborsClassifier()),
        ]
    )
    
    # 在数据上拟合 Pipeline
    clf.fit(X, y)
    
    # 断言分类器的得分高于 0.9
    assert 0.9 < clf.score(X, y)


# 测试当权重矩阵是奇异时是否引发错误
def test_singular_matrix():
    # 创建一个全为 1 的矩阵
    M = np.ones((200, 3))
    f = ignore_warnings
    
    # 使用 pytest 断言捕获特定的 ValueError 错误，匹配特定的错误信息
    with pytest.raises(ValueError, match="Error in determining null-space with ARPACK"):
        # 调用 locally_linear_embedding 函数进行局部线性嵌入
        f(
            manifold.locally_linear_embedding(
                M,
                n_neighbors=2,
                n_components=1,
                method="standard",
                eigen_solver="arpack",
            )
        )


# 回归测试 #6033
def test_integer_input():
    # 创建一个随机状态对象
    rand = np.random.RandomState(0)
    
    # 生成一个随机整数矩阵
    X = rand.randint(0, 100, size=(20, 3))

    # 使用不同的方法进行局部线性嵌入，测试是否会引发 TypeError
    for method in ["standard", "hessian", "modified", "ltsa"]:
        clf = manifold.LocallyLinearEmbedding(method=method, n_neighbors=10)
        clf.fit(X)


def test_get_feature_names_out():
    """检查 LocallyLinearEmbedding 的 get_feature_names_out 方法。"""
    # 生成一个随机数据集
    X, y = make_blobs(random_state=0, n_features=4)
    n_components = 2
    
    # 创建 LocallyLinearEmbedding 对象
    iso = manifold.LocallyLinearEmbedding(n_components=n_components)
    iso.fit(X)
    
    # 获取输出的特征名称
    names = iso.get_feature_names_out()
    
    # 断言特征名称与预期的格式匹配
    assert_array_equal(
        [f"locallylinearembedding{i}" for i in range(n_components)], names
    )
```