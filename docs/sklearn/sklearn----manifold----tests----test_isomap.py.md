# `D:\src\scipysrc\scikit-learn\sklearn\manifold\tests\test_isomap.py`

```
# 导入必要的库
import math  # 导入数学函数库
from itertools import product  # 导入用于迭代器的函数

import numpy as np  # 导入数值计算库 numpy
import pytest  # 导入测试框架 pytest
from scipy.sparse import rand as sparse_rand  # 导入稀疏矩阵生成函数

# 导入 sklearn 中的各种模块和函数
from sklearn import clone, datasets, manifold, neighbors, pipeline, preprocessing
from sklearn.datasets import make_blobs  # 从 sklearn 数据集中导入生成聚类数据的函数
from sklearn.metrics.pairwise import pairwise_distances  # 导入用于计算成对距离的函数
from sklearn.utils._testing import (
    assert_allclose,  # 导入用于测试所有元素是否接近的函数
    assert_allclose_dense_sparse,  # 导入用于测试密集和稀疏矩阵是否接近的函数
    assert_array_equal,  # 导入用于测试两个数组是否相等的函数
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入用于兼容 CSR 稀疏矩阵的修复函数

# 定义两个常量列表
eigen_solvers = ["auto", "dense", "arpack"]
path_methods = ["auto", "FW", "D"]


def create_sample_data(dtype, n_pts=25, add_noise=False):
    # 创建一个样本数据集，返回一个 numpy 数组
    # 如果需要添加噪声，则在第三个维度添加随机噪声
    n_per_side = int(math.sqrt(n_pts))
    X = np.array(list(product(range(n_per_side), repeat=2))).astype(dtype, copy=False)
    if add_noise:
        # 添加第三维度的噪声
        rng = np.random.RandomState(0)
        noise = 0.1 * rng.randn(n_pts, 1).astype(dtype, copy=False)
        X = np.concatenate((X, noise), 1)
    return X


# 测试函数，使用 pytest.mark.parametrize 进行参数化测试
@pytest.mark.parametrize("n_neighbors, radius", [(24, None), (None, np.inf)])
@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
def test_isomap_simple_grid(
    global_dtype, n_neighbors, radius, eigen_solver, path_method
):
    # 测试 Isomap 在简单网格上的表现
    n_pts = 25
    X = create_sample_data(global_dtype, n_pts=n_pts, add_noise=False)

    # 计算每个点到其他所有点的距离矩阵
    if n_neighbors is not None:
        G = neighbors.kneighbors_graph(X, n_neighbors, mode="distance")
    else:
        G = neighbors.radius_neighbors_graph(X, radius, mode="distance")

    # 创建 Isomap 对象并拟合数据
    clf = manifold.Isomap(
        n_neighbors=n_neighbors,
        radius=radius,
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
    )
    clf.fit(X)

    # 计算转换后的点之间的距离矩阵
    if n_neighbors is not None:
        G_iso = neighbors.kneighbors_graph(clf.embedding_, n_neighbors, mode="distance")
    else:
        G_iso = neighbors.radius_neighbors_graph(
            clf.embedding_, radius, mode="distance"
        )
    atol = 1e-5 if global_dtype == np.float32 else 0

    # 使用 assert_allclose 函数检查转换后的距离矩阵和原始距离矩阵的接近程度
    assert_allclose_dense_sparse(G, G_iso, atol=atol)


# 另一个参数化测试函数，测试 Isomap 的重建误差
@pytest.mark.parametrize("n_neighbors, radius", [(24, None), (None, np.inf)])
@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
def test_isomap_reconstruction_error(
    global_dtype, n_neighbors, radius, eigen_solver, path_method
):
    if global_dtype is np.float32:
        # 如果数据类型为 float32，则跳过测试，因为数值不稳定
        pytest.skip(
            "Skipping test due to numerical instabilities on float32 data"
            "from KernelCenterer used in the reconstruction_error method"
        )

    # 和 test_isomap_simple_grid 函数相同的设置，添加了一个额外的维度进行噪声测试
    n_pts = 25
    X = create_sample_data(global_dtype, n_pts=n_pts, add_noise=True)

    # 计算输入核
    # 如果指定了邻居数量，则使用 k 近邻方法计算距离图 G
    if n_neighbors is not None:
        G = neighbors.kneighbors_graph(X, n_neighbors, mode="distance").toarray()
    else:
        # 否则，使用半径邻居方法计算距离图 G
        G = neighbors.radius_neighbors_graph(X, radius, mode="distance").toarray()

    # 使用核心化中心化器对 G 进行处理，计算核 K
    centerer = preprocessing.KernelCenterer()
    K = centerer.fit_transform(-0.5 * G**2)

    # 创建 Isomap 对象 clf，设置参数并拟合数据 X
    clf = manifold.Isomap(
        n_neighbors=n_neighbors,
        radius=radius,
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
    )
    clf.fit(X)

    # 计算输出核 G_iso
    if n_neighbors is not None:
        # 如果指定了邻居数量，则使用 k 近邻方法计算新的距离图 G_iso
        G_iso = neighbors.kneighbors_graph(clf.embedding_, n_neighbors, mode="distance")
    else:
        # 否则，使用半径邻居方法计算新的距离图 G_iso
        G_iso = neighbors.radius_neighbors_graph(clf.embedding_, radius, mode="distance")
    G_iso = G_iso.toarray()

    # 使用核心化中心化器对 G_iso 进行处理，计算核 K_iso
    K_iso = centerer.fit_transform(-0.5 * G_iso**2)

    # 确保重构误差一致性
    reconstruction_error = np.linalg.norm(K - K_iso) / n_pts
    # 根据全局数据类型设置容差值
    atol = 1e-5 if global_dtype == np.float32 else 0
    # 断言重构误差满足预期
    assert_allclose(reconstruction_error, clf.reconstruction_error(), atol=atol)
# 使用 pytest.mark.parametrize 装饰器为 test_transform 函数定义多组参数化测试用例
@pytest.mark.parametrize("n_neighbors, radius", [(2, None), (None, 0.5)])
def test_transform(global_dtype, n_neighbors, radius):
    # 设置样本数量、组件数量和噪声比例
    n_samples = 200
    n_components = 10
    noise_scale = 0.01

    # 创建 S-curve 数据集
    X, y = datasets.make_s_curve(n_samples, random_state=0)

    # 将数据类型转换为全局数据类型
    X = X.astype(global_dtype, copy=False)

    # 计算 Isomap 的嵌入
    iso = manifold.Isomap(
        n_components=n_components, n_neighbors=n_neighbors, radius=radius
    )
    X_iso = iso.fit_transform(X)

    # 对数据添加噪声并重新嵌入
    rng = np.random.RandomState(0)
    noise = noise_scale * rng.randn(*X.shape)
    X_iso2 = iso.transform(X + noise)

    # 确保重新嵌入后的均方根误差与噪声比例相当小
    assert np.sqrt(np.mean((X_iso - X_iso2) ** 2)) < 2 * noise_scale


# 使用 pytest.mark.parametrize 装饰器为 test_pipeline 函数定义多组参数化测试用例
@pytest.mark.parametrize("n_neighbors, radius", [(2, None), (None, 10.0)])
def test_pipeline(n_neighbors, radius, global_dtype):
    # 检查 Isomap 在 Pipeline 中作为转换器的运行情况，确保不会引发错误
    # 只检查是否能正常运行，未验证是否确实有用
    X, y = datasets.make_blobs(random_state=0)

    # 将数据类型转换为全局数据类型
    X = X.astype(global_dtype, copy=False)

    # 创建 Pipeline 包含 Isomap 转换器和 KNeighborsClassifier 分类器
    clf = pipeline.Pipeline(
        [
            ("isomap", manifold.Isomap(n_neighbors=n_neighbors, radius=radius)),
            ("clf", neighbors.KNeighborsClassifier()),
        ]
    )
    clf.fit(X, y)

    # 断言分类器的得分高于指定阈值
    assert 0.9 < clf.score(X, y)


# test_pipeline_with_nearest_neighbors_transformer 函数不需要参数化测试
def test_pipeline_with_nearest_neighbors_transformer(global_dtype):
    # 测试 NearestNeighborsTransformer 和 Isomap 的串联使用，
    # 并设置 neighbors_algorithm='precomputed'
    algorithm = "auto"
    n_neighbors = 10

    # 创建两组随机数据集
    X, _ = datasets.make_blobs(random_state=0)
    X2, _ = datasets.make_blobs(random_state=1)

    # 将数据类型转换为全局数据类型
    X = X.astype(global_dtype, copy=False)
    X2 = X2.astype(global_dtype, copy=False)

    # 比较串联版本和紧凑版本的 Isomap 嵌入结果
    est_chain = pipeline.make_pipeline(
        neighbors.KNeighborsTransformer(
            n_neighbors=n_neighbors, algorithm=algorithm, mode="distance"
        ),
        manifold.Isomap(n_neighbors=n_neighbors, metric="precomputed"),
    )
    est_compact = manifold.Isomap(
        n_neighbors=n_neighbors, neighbors_algorithm=algorithm
    )

    Xt_chain = est_chain.fit_transform(X)
    Xt_compact = est_compact.fit_transform(X)

    # 断言两种嵌入结果在数值上接近
    assert_allclose(Xt_chain, Xt_compact)

    Xt_chain = est_chain.transform(X2)
    Xt_compact = est_compact.transform(X2)

    # 断言两种嵌入结果在数值上接近
    assert_allclose(Xt_chain, Xt_compact)


# 使用 pytest.mark.parametrize 装饰器为 test_different_metric 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "metric, p, is_euclidean",
    [
        ("euclidean", 2, True),
        ("manhattan", 1, False),
        ("minkowski", 1, False),
        ("minkowski", 2, True),
        (lambda x1, x2: np.sqrt(np.sum(x1**2 + x2**2)), 2, False),
    ],
)
def test_different_metric(global_dtype, metric, p, is_euclidean):
    # 测试 Isomap 在不同度量参数下的正确工作，应默认使用欧氏距离
    X, _ = datasets.make_blobs(random_state=0)
    # 将变量 X 转换为指定的全局数据类型，不复制数据
    X = X.astype(global_dtype, copy=False)
    
    # 使用默认参数创建 Isomap 对象，并对输入数据 X 进行拟合和转换，得到降维后的参考结果 reference
    reference = manifold.Isomap().fit_transform(X)
    
    # 使用指定的距离度量 metric 和参数 p 创建 Isomap 对象，并对输入数据 X 进行拟合和转换，得到降维后的嵌入结果 embedding
    embedding = manifold.Isomap(metric=metric, p=p).fit_transform(X)
    
    # 如果使用的是欧几里得距离，验证 embedding 是否与 reference 接近
    if is_euclidean:
        assert_allclose(embedding, reference)
    else:
        # 如果不是欧几里得距离，则验证 embedding 是否与 reference 不接近，并抛出断言错误，错误消息为 "Not equal to tolerance"
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            assert_allclose(embedding, reference)
def test_isomap_clone_bug():
    # 回归测试，针对问题 #6062 报告的 bug
    model = manifold.Isomap()
    for n_neighbors in [10, 15, 20]:
        model.set_params(n_neighbors=n_neighbors)
        model.fit(np.random.rand(50, 2))
        assert model.nbrs_.n_neighbors == n_neighbors


@pytest.mark.parametrize("eigen_solver", eigen_solvers)
@pytest.mark.parametrize("path_method", path_methods)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input(
    global_dtype, eigen_solver, path_method, global_random_seed, csr_container
):
    # TODO: 按照 https://github.com/scikit-learn/scikit-learn/pull/23585#discussion_r968388186 提议，比较稠密数据和稀疏数据的结果
    X = csr_container(
        sparse_rand(
            100,
            3,
            density=0.1,
            format="csr",
            dtype=global_dtype,
            random_state=global_random_seed,
        )
    )

    iso_dense = manifold.Isomap(
        n_components=2,
        eigen_solver=eigen_solver,
        path_method=path_method,
        n_neighbors=8,
    )
    iso_sparse = clone(iso_dense)

    X_trans_dense = iso_dense.fit_transform(X.toarray())
    X_trans_sparse = iso_sparse.fit_transform(X)

    assert_allclose(X_trans_sparse, X_trans_dense, rtol=1e-4, atol=1e-4)


def test_isomap_fit_precomputed_radius_graph(global_dtype):
    # Isomap.fit_transform 在使用预计算距离矩阵时必须产生类似的结果

    X, y = datasets.make_s_curve(200, random_state=0)
    X = X.astype(global_dtype, copy=False)
    radius = 10

    g = neighbors.radius_neighbors_graph(X, radius=radius, mode="distance")
    isomap = manifold.Isomap(n_neighbors=None, radius=radius, metric="precomputed")
    isomap.fit(g)
    precomputed_result = isomap.embedding_

    isomap = manifold.Isomap(n_neighbors=None, radius=radius, metric="minkowski")
    result = isomap.fit_transform(X)
    atol = 1e-5 if global_dtype == np.float32 else 0
    assert_allclose(precomputed_result, result, atol=atol)


def test_isomap_fitted_attributes_dtype(global_dtype):
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    iso = manifold.Isomap(n_neighbors=2)

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)

    iso.fit(X)

    assert iso.dist_matrix_.dtype == global_dtype
    assert iso.embedding_.dtype == global_dtype


def test_isomap_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    iso_32 = manifold.Isomap(n_neighbors=2)
    X_32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    iso_32.fit(X_32)

    iso_64 = manifold.Isomap(n_neighbors=2)
    X_64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    iso_64.fit(X_64)

    assert_allclose(iso_32.dist_matrix_, iso_64.dist_matrix_)


def test_isomap_raise_error_when_neighbor_and_radius_both_set():
    # Isomap.fit_transform 必须在同时设置邻居数和半径时引发 ValueError
    # 加载手写数字数据集，返回特征矩阵 X 和标签向量（这里的 _ 表示我们不关心标签）
    X, _ = datasets.load_digits(return_X_y=True)
    
    # 使用流形学习中的等距映射（Isomap）算法，设定参数 n_neighbors=3 和 radius=5.5
    isomap = manifold.Isomap(n_neighbors=3, radius=5.5)
    
    # 准备用于匹配的错误消息，说明 n_neighbors 和 radius 都已经提供
    msg = "Both n_neighbors and radius are provided"
    
    # 使用 pytest 库的 raises 方法来检查是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        # 对手写数字数据集 X 进行等距映射的拟合和转换，期望这里会抛出异常
        isomap.fit_transform(X)
# 测试多个连接组件的情况
def test_multiple_connected_components():
    # 测试当图形具有多个组件时是否会触发警告
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    # 使用 pytest 来检测是否会引发 UserWarning，并匹配警告信息中是否包含 "number of connected components"
    with pytest.warns(UserWarning, match="number of connected components"):
        # 使用 Isomap 算法拟合数据 X
        manifold.Isomap(n_neighbors=2).fit(X)


# 测试多个连接组件的情况，使用预计算的图形
def test_multiple_connected_components_metric_precomputed(global_dtype):
    # 测试当图形有多个组件时以及 X 是预计算的邻居图时是否会引发错误
    X = np.array([0, 1, 2, 5, 6, 7])[:, None].astype(global_dtype, copy=False)

    # 使用预计算的距离矩阵（密集形式）
    X_distances = pairwise_distances(X)
    with pytest.warns(UserWarning, match="number of connected components"):
        # 使用 Isomap 算法拟合预计算的距离矩阵 X_distances
        manifold.Isomap(n_neighbors=1, metric="precomputed").fit(X_distances)

    # 使用预计算的邻居图（稀疏形式）
    X_graph = neighbors.kneighbors_graph(X, n_neighbors=2, mode="distance")
    with pytest.raises(RuntimeError, match="number of connected components"):
        # 使用 Isomap 算法拟合预计算的邻居图 X_graph
        manifold.Isomap(n_neighbors=1, metric="precomputed").fit(X_graph)


# 测试 Isomap 的 get_feature_names_out 方法
def test_get_feature_names_out():
    """Check get_feature_names_out for Isomap."""
    # 创建一个具有 4 个特征的随机数据集
    X, y = make_blobs(random_state=0, n_features=4)
    n_components = 2

    # 使用 Isomap 进行降维到 n_components 维度
    iso = manifold.Isomap(n_components=n_components)
    iso.fit_transform(X)
    # 调用 Isomap 的 get_feature_names_out 方法获取输出特征名
    names = iso.get_feature_names_out()
    # 检查输出特征名是否符合预期
    assert_array_equal([f"isomap{i}" for i in range(n_components)], names)
```