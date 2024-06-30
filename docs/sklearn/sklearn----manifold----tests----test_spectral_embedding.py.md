# `D:\src\scipysrc\scikit-learn\sklearn\manifold\tests\test_spectral_embedding.py`

```
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding, _spectral_embedding, spectral_embedding
from sklearn.manifold._spectral_embedding import (
    _graph_connected_component,
    _graph_is_connected,
)
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import (
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    parse_version,
    sp_version,
)
from sklearn.utils.fixes import laplacian as csgraph_laplacian

try:
    from pyamg import smoothed_aggregation_solver  # noqa

    pyamg_available = True
except ImportError:
    pyamg_available = False
skip_if_no_pyamg = pytest.mark.skipif(
    not pyamg_available, reason="PyAMG is required for the tests in this function."
)

# non centered, sparse centers to check the
centers = np.array(
    [
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 5.0, 1.0],
    ]
)
n_samples = 1000
n_clusters, n_features = centers.shape
S, true_labels = make_blobs(
    n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
)


def _assert_equal_with_sign_flipping(A, B, tol=0.0):
    """Check array A and B are equal with possible sign flipping on
    each columns"""
    tol_squared = tol**2
    for A_col, B_col in zip(A.T, B.T):
        assert (
            np.max((A_col - B_col) ** 2) <= tol_squared
            or np.max((A_col + B_col) ** 2) <= tol_squared
        )


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_sparse_graph_connected_component(coo_container):
    """Test function for checking connected components in sparse graphs."""
    rng = np.random.RandomState(42)
    n_samples = 300
    boundaries = [0, 42, 121, 200, n_samples]
    p = rng.permutation(n_samples)
    connections = []

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group = p[start:stop]
        # Connect all elements within the group at least once via an
        # arbitrary path that spans the group.
        for i in range(len(group) - 1):
            connections.append((group[i], group[i + 1]))

        # Add some more random connections within the group
        min_idx, max_idx = 0, len(group) - 1
        n_random_connections = 1000
        source = rng.randint(min_idx, max_idx, size=n_random_connections)
        target = rng.randint(min_idx, max_idx, size=n_random_connections)
        connections.extend(zip(group[source], group[target]))

    # Build a symmetric affinity matrix
    row_idx, column_idx = tuple(np.array(connections).T)
    data = rng.uniform(0.1, 42, size=len(connections))
    # 使用输入数据和行列索引创建稀疏矩阵
    affinity = coo_container((data, (row_idx, column_idx)))
    
    # 将矩阵的转置与自身的平均值加权，得到亲和性矩阵
    affinity = 0.5 * (affinity + affinity.T)

    # 遍历边界列表，每次处理一个组件的范围
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        # 根据起始点 p[start] 获取连接的图组件
        component_1 = _graph_connected_component(affinity, p[start])
        
        # 计算组件的大小
        component_size = stop - start
        # 断言第一个组件的元素总和等于组件的大小
        assert component_1.sum() == component_size

        # 通过终点 p[stop - 1] 获取另一个端点开始的相同组件掩码
        component_2 = _graph_connected_component(affinity, p[stop - 1])
        # 断言第二个组件的元素总和也等于组件的大小
        assert component_2.sum() == component_size
        # 断言两个组件的掩码数组相等
        assert_array_equal(component_1, component_2)
# 定义测试函数，用于测试 spectral embedding 的两个组件情况
@pytest.mark.parametrize(
    "eigen_solver",
    [
        "arpack",  # 使用 arpack 求解特征值问题
        "lobpcg",  # 使用 lobpcg 求解特征值问题
        pytest.param("amg", marks=skip_if_no_pyamg),  # 如果有 pyamg 库，则使用 amg 求解特征值问题
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])  # 参数化 dtype，支持 float32 和 float64 类型
def test_spectral_embedding_two_components(eigen_solver, dtype, seed=0):
    # Test spectral embedding with two components
    random_state = np.random.RandomState(seed)  # 使用指定种子创建随机数生成器对象
    n_sample = 100  # 样本数量
    affinity = np.zeros(shape=[n_sample * 2, n_sample * 2])  # 创建全零矩阵作为亲和度矩阵

    # 第一个组件
    affinity[0:n_sample, 0:n_sample] = (
        np.abs(random_state.randn(n_sample, n_sample)) + 2  # 填充亲和度矩阵的左上角子矩阵
    )

    # 第二个组件
    affinity[n_sample::, n_sample::] = (
        np.abs(random_state.randn(n_sample, n_sample)) + 2  # 填充亲和度矩阵的右下角子矩阵
    )

    # 测试 _graph_connected_component 函数在连接之前的表现
    component = _graph_connected_component(affinity, 0)  # 获取从节点 0 开始的连通组件
    assert component[:n_sample].all()  # 断言前半部分节点都在同一连通组件中
    assert not component[n_sample:].any()  # 断言后半部分节点不在同一连通组件中

    component = _graph_connected_component(affinity, -1)  # 获取从最后一个节点开始的连通组件
    assert not component[:n_sample].any()  # 断言前半部分节点不在同一连通组件中
    assert component[n_sample:].all()  # 断言后半部分节点都在同一连通组件中

    # 连接两个组件
    affinity[0, n_sample + 1] = 1
    affinity[n_sample + 1, 0] = 1
    affinity.flat[:: 2 * n_sample + 1] = 0
    affinity = 0.5 * (affinity + affinity.T)  # 将亲和度矩阵设为对称

    true_label = np.zeros(shape=2 * n_sample)  # 创建真实标签数组
    true_label[0:n_sample] = 1  # 标记前半部分节点属于第一组件

    # 创建并拟合 spectral embedding 对象，使用预计算的亲和度矩阵
    se_precomp = SpectralEmbedding(
        n_components=1,
        affinity="precomputed",  # 指定使用预计算的亲和度矩阵
        random_state=np.random.RandomState(seed),  # 使用指定种子创建随机数生成器对象
        eigen_solver=eigen_solver,  # 指定特征值求解器
    )

    embedded_coordinate = se_precomp.fit_transform(affinity.astype(dtype))  # 拟合并转换数据
    label_ = np.array(embedded_coordinate.ravel() < 0, dtype=np.int64)  # 基于第一维度的阈值标签化
    assert normalized_mutual_info_score(true_label, label_) == pytest.approx(1.0)  # 断言标签的规范化互信息分数为接近1.0


@pytest.mark.parametrize("sparse_container", [None, *CSR_CONTAINERS])  # 参数化稀疏矩阵容器
@pytest.mark.parametrize(
    "eigen_solver",
    [
        "arpack",  # 使用 arpack 求解特征值问题
        "lobpcg",  # 使用 lobpcg 求解特征值问题
        pytest.param("amg", marks=skip_if_no_pyamg),  # 如果有 pyamg 库，则使用 amg 求解特征值问题
    ],
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))  # 参数化 dtype，支持 float32 和 float64 类型
def test_spectral_embedding_precomputed_affinity(
    sparse_container, eigen_solver, dtype, seed=36
):
    # Test spectral embedding with precomputed kernel
    gamma = 1.0  # RBF 内核的 gamma 参数
    X = S if sparse_container is None else sparse_container(S)  # 根据稀疏容器创建输入数据 X

    # 创建并拟合 spectral embedding 对象，使用预计算的亲和度矩阵
    se_precomp = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",  # 指定使用预计算的亲和度矩阵
        random_state=np.random.RandomState(seed),  # 使用指定种子创建随机数生成器对象
        eigen_solver=eigen_solver,  # 指定特征值求解器
    )

    # 创建并拟合 spectral embedding 对象，使用 RBF 内核的亲和度矩阵
    se_rbf = SpectralEmbedding(
        n_components=2,
        affinity="rbf",  # 指定使用 RBF 内核的亲和度矩阵
        gamma=gamma,  # 指定 RBF 内核的 gamma 参数
        random_state=np.random.RandomState(seed),  # 使用指定种子创建随机数生成器对象
        eigen_solver=eigen_solver,  # 指定特征值求解器
    )
    # 使用预计算的核函数（RBF）转换器对数据集 X 进行转换，并生成嵌入向量
    embed_precomp = se_precomp.fit_transform(rbf_kernel(X.astype(dtype), gamma=gamma))
    # 使用常规的核函数（RBF）转换器对数据集 X 进行转换，并生成嵌入向量
    embed_rbf = se_rbf.fit_transform(X.astype(dtype))
    # 断言预计算核函数和常规核函数生成的亲和矩阵近似相等，精度为 0.05
    assert_array_almost_equal(se_precomp.affinity_matrix_, se_rbf.affinity_matrix_)
    # 使用带符号翻转的相等性断言函数检查 embed_precomp 和 embed_rbf 的近似相等性，精度为 0.05
    _assert_equal_with_sign_flipping(embed_precomp, embed_rbf, 0.05)
# 测试预先计算的最近邻过滤器
def test_precomputed_nearest_neighbors_filtering():
    # 定义最近邻数量
    n_neighbors = 2
    # 存储结果的列表
    results = []
    # 对于不同的附加邻居数进行迭代
    for additional_neighbors in [0, 10]:
        # 使用 NearestNeighbors 拟合数据集 S，并设置邻居数量
        nn = NearestNeighbors(n_neighbors=n_neighbors + additional_neighbors).fit(S)
        # 生成数据集 S 的邻接图
        graph = nn.kneighbors_graph(S, mode="connectivity")
        # 使用预先计算的最近邻作为亲和力的谱嵌入，设置随机种子和组件数
        embedding = (
            SpectralEmbedding(
                random_state=0,
                n_components=2,
                affinity="precomputed_nearest_neighbors",
                n_neighbors=n_neighbors,
            )
            .fit(graph)  # 对图进行拟合
            .embedding_  # 提取嵌入结果
        )
        # 将嵌入结果添加到列表中
        results.append(embedding)

    # 断言两个结果相等
    assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("sparse_container", [None, *CSR_CONTAINERS])
def test_spectral_embedding_callable_affinity(sparse_container, seed=36):
    # 测试具有可调用亲和力的谱嵌入
    gamma = 0.9
    # 计算 RBF 核矩阵
    kern = rbf_kernel(S, gamma=gamma)
    # 根据稀疏容器类型选择数据类型
    X = S if sparse_container is None else sparse_container(S)

    # 使用可调用的亲和力函数定义谱嵌入对象
    se_callable = SpectralEmbedding(
        n_components=2,
        affinity=(lambda x: rbf_kernel(x, gamma=gamma)),  # 定义可调用的亲和力函数
        gamma=gamma,  # 设置 RBF 核的 gamma 参数
        random_state=np.random.RandomState(seed),  # 设置随机种子
    )
    # 使用 RBF 亲和力定义谱嵌入对象
    se_rbf = SpectralEmbedding(
        n_components=2,
        affinity="rbf",  # 使用 RBF 亲和力
        gamma=gamma,  # 设置 RBF 核的 gamma 参数
        random_state=np.random.RandomState(seed),  # 设置随机种子
    )
    # 对数据进行谱嵌入转换并计算结果
    embed_rbf = se_rbf.fit_transform(X)
    embed_callable = se_callable.fit_transform(X)
    # 断言两个谱嵌入对象的亲和力矩阵近似相等
    assert_array_almost_equal(se_callable.affinity_matrix_, se_rbf.affinity_matrix_)
    # 断言 RBF 核矩阵与谱嵌入对象的亲和力矩阵近似相等
    assert_array_almost_equal(kern, se_rbf.affinity_matrix_)
    # 使用带有符号翻转的相等性检查函数
    _assert_equal_with_sign_flipping(embed_rbf, embed_callable, 0.05)


# TODO: 当 pyamg 替换 sp.rand 调用为 np.random.rand 时移除
# https://github.com/scikit-learn/scikit-learn/issues/15913
# TODO: 当 pyamg 移除对 np.float 的使用时移除
# TODO: 当 pyamg 移除对 pinv2 的使用时移除
# TODO: 当 pyamg 移除对 np.find_common_type 的使用时移除
@pytest.mark.filterwarnings(
    "ignore:scipy.rand is deprecated:DeprecationWarning:pyamg.*"
)
@pytest.mark.filterwarnings(
    "ignore:`np.float` is a deprecated alias:DeprecationWarning:pyamg.*"
)
@pytest.mark.filterwarnings(
    "ignore:scipy.linalg.pinv2 is deprecated:DeprecationWarning:pyamg.*"
)
@pytest.mark.filterwarnings(
    "ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*"
)
# 如果 PyAMG 不可用，则跳过这个测试函数
@pytest.mark.skipif(
    not pyamg_available, reason="PyAMG is required for the tests in this function."
)
# 参数化测试函数，用于不同的数据类型和 COO 容器
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_spectral_embedding_amg_solver(dtype, coo_container, seed=36):
    # 使用 AMG 求解器进行谱嵌入
    se_amg = SpectralEmbedding(
        n_components=2,
        affinity="nearest_neighbors",
        eigen_solver="amg",
        n_neighbors=5,
        random_state=np.random.RandomState(seed),  # 设置随机种子
    )
    # 使用 SpectralEmbedding 创建对象 se_arpack，设置参数如下：
    # n_components=2 表示嵌入空间的维度为2
    # affinity="nearest_neighbors" 使用最近邻方法计算相似度
    # eigen_solver="arpack" 使用 arpack 解决特征值问题
    # n_neighbors=5 指定最近邻的数量为5
    # random_state=np.random.RandomState(seed) 设置随机数种子为给定的 seed
    se_arpack = SpectralEmbedding(
        n_components=2,
        affinity="nearest_neighbors",
        eigen_solver="arpack",
        n_neighbors=5,
        random_state=np.random.RandomState(seed),
    )

    # 使用 se_amg 对象拟合并转换输入数据 S，结果嵌入到 embed_amg 中
    embed_amg = se_amg.fit_transform(S.astype(dtype))

    # 使用 se_arpack 对象拟合并转换输入数据 S，结果嵌入到 embed_arpack 中
    embed_arpack = se_arpack.fit_transform(S.astype(dtype))

    # 调用 _assert_equal_with_sign_flipping 函数验证 embed_amg 和 embed_arpack 是否相等，精度为 1e-5
    _assert_equal_with_sign_flipping(embed_amg, embed_arpack, 1e-5)

    # 处理特殊情况：当 amg 实际上未被使用时
    # 这是 #10715 号问题的回归测试
    # 计算节点之间的相似度
    row = np.array([0, 0, 1, 2, 3, 3, 4], dtype=np.int32)
    col = np.array([1, 2, 2, 3, 4, 5, 5], dtype=np.int32)
    val = np.array([100, 100, 100, 1, 100, 100, 100], dtype=np.int64)

    # 创建一个 coo_container 对象 affinity，包含值、行和列信息，形状为 (6, 6)
    affinity = coo_container(
        (np.hstack([val, val]), (np.hstack([row, col]), np.hstack([col, row]))),
        shape=(6, 6),
    )

    # 设置 se_amg 和 se_arpack 对象的 affinity 属性为 "precomputed"
    se_amg.affinity = "precomputed"
    se_arpack.affinity = "precomputed"

    # 使用 se_amg 对象拟合并转换输入的 precomputed 相似度矩阵 affinity，结果嵌入到 embed_amg 中
    embed_amg = se_amg.fit_transform(affinity.astype(dtype))

    # 使用 se_arpack 对象拟合并转换输入的 precomputed 相似度矩阵 affinity，结果嵌入到 embed_arpack 中
    embed_arpack = se_arpack.fit_transform(affinity.astype(dtype))

    # 再次调用 _assert_equal_with_sign_flipping 函数验证 embed_amg 和 embed_arpack 是否相等，精度为 1e-5
    _assert_equal_with_sign_flipping(embed_amg, embed_arpack, 1e-5)

    # 检查是否会因为稀疏矩阵的索引类型为 np.int64 而引发错误
    # 或者根据安装的 SciPy 版本是否成功
    # 使用 CSR 矩阵来确保在验证过程中不进行任何转换
    affinity = affinity.tocsr()
    affinity.indptr = affinity.indptr.astype(np.int64)
    affinity.indices = affinity.indices.astype(np.int64)

    # PR：https://github.com/scipy/scipy/pull/18913
    # 第一次整合于 1.11.3 版本：https://github.com/scipy/scipy/pull/19279
    # 检查当前 SciPy 版本是否支持 np.int64 索引类型
    scipy_graph_traversal_supports_int64_index = sp_version >= parse_version("1.11.3")
    if scipy_graph_traversal_supports_int64_index:
        # 如果支持，使用 se_amg 对象拟合输入的稀疏矩阵 affinity
        se_amg.fit_transform(affinity)
    else:
        # 如果不支持，预期抛出 ValueError 异常，匹配错误信息 err_msg
        err_msg = "Only sparse matrices with 32-bit integer indices are accepted"
        with pytest.raises(ValueError, match=err_msg):
            se_amg.fit_transform(affinity)
# 设置 pytest 标记，忽略由于 scipy.rand 被弃用导致的警告
@pytest.mark.filterwarnings(
    "ignore:scipy.rand is deprecated:DeprecationWarning:pyamg.*"
)
# 设置 pytest 标记，忽略由于 np.float 被弃用导致的警告
@pytest.mark.filterwarnings(
    "ignore:`np.float` is a deprecated alias:DeprecationWarning:pyamg.*"
)
# 设置 pytest 标记，忽略由于 scipy.linalg.pinv2 被弃用导致的警告
@pytest.mark.filterwarnings(
    "ignore:scipy.linalg.pinv2 is deprecated:DeprecationWarning:pyamg.*"
)
# 设置 pytest 标记，如果 pyamg 不可用，则跳过此测试函数
@pytest.mark.skipif(
    not pyamg_available, reason="PyAMG is required for the tests in this function."
)
# 设置 pytest 标记，忽略由于 np.find_common_type 被弃用导致的警告
@pytest.mark.filterwarnings(
    "ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*"
)
# 参数化测试函数，传入不同的 dtype 参数进行测试
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_spectral_embedding_amg_solver_failure(dtype, seed=36):
    # 非回归测试，用于测试 amg 求解器的失败情况（github 上的 issue #13393）
    num_nodes = 100
    # 生成稀疏随机矩阵 X，设置密度为 0.1，使用给定的随机种子
    X = sparse.rand(num_nodes, num_nodes, density=0.1, random_state=seed)
    # 将 X 转换为指定的 dtype 类型
    X = X.astype(dtype)
    # 计算上三角矩阵
    upper = sparse.triu(X) - sparse.diags(X.diagonal())
    # 构造对称矩阵
    sym_matrix = upper + upper.T
    # 使用 spectral_embedding 进行嵌入，选择 amg 作为特征值求解器
    embedding = spectral_embedding(
        sym_matrix, n_components=10, eigen_solver="amg", random_state=0
    )

    # 检查学习到的嵌入结果是否对随机求解器初始化稳定
    for i in range(3):
        new_embedding = spectral_embedding(
            sym_matrix, n_components=10, eigen_solver="amg", random_state=i + 1
        )
        _assert_equal_with_sign_flipping(embedding, new_embedding, tol=0.05)


# 设置 pytest 标记，忽略在版本 0.22 中 nmi 行为的变更警告
@pytest.mark.filterwarnings("ignore:the behavior of nmi will change in version 0.22")
def test_pipeline_spectral_clustering(seed=36):
    # 使用管道进行谱聚类的测试
    random_state = np.random.RandomState(seed)
    # 使用 RBF 核进行谱嵌入
    se_rbf = SpectralEmbedding(
        n_components=n_clusters, affinity="rbf", random_state=random_state
    )
    # 使用最近邻方法进行谱嵌入
    se_knn = SpectralEmbedding(
        n_components=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=5,
        random_state=random_state,
    )
    # 遍历不同的谱嵌入方法
    for se in [se_rbf, se_knn]:
        # 使用 KMeans 进行聚类，验证标签的归一化互信息分数接近 1.0
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        km.fit(se.fit_transform(S))
        assert_array_almost_equal(
            normalized_mutual_info_score(km.labels_, true_labels), 1.0, 2
        )


def test_connectivity(seed=36):
    # 测试图的连通性检测是否正常工作
    graph = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    # 断言图不是连通的
    assert not _graph_is_connected(graph)
    # 遍历 CSR 格式的容器，断言图不是连通的
    for csr_container in CSR_CONTAINERS:
        assert not _graph_is_connected(csr_container(graph))
    # 遍历 CSC 格式的容器，断言图不是连通的
    for csc_container in CSC_CONTAINERS:
        assert not _graph_is_connected(csc_container(graph))
    # 创建一个表示图的邻接矩阵，描述了图中的连接关系
    graph = np.array(
        [
            [1, 1, 0, 0, 0],   # 第一个节点与第二个节点相连，第三个节点与第二个节点相连
            [1, 1, 1, 0, 0],   # 第二个节点与第三个节点相连，第四个节点与第三个节点相连
            [0, 1, 1, 1, 0],   # 第三个节点与第四个节点相连，第五个节点与第四个节点相连
            [0, 0, 1, 1, 1],   # 第四个节点与第五个节点相连
            [0, 0, 0, 1, 1],   # 第五个节点与第四个节点相连
        ]
    )
    
    # 检查用于连接性的辅助函数是否返回 True
    assert _graph_is_connected(graph)
    
    # 对于每个CSR容器类型，确保传入的图是连通的
    for csr_container in CSR_CONTAINERS:
        assert _graph_is_connected(csr_container(graph))
    
    # 对于每个CSC容器类型，确保传入的图是连通的
    for csc_container in CSC_CONTAINERS:
        assert _graph_is_connected(csc_container(graph))
def test_spectral_embedding_deterministic():
    # 测试谱嵌入是否确定性的
    random_state = np.random.RandomState(36)
    # 创建随机数生成器，指定种子为36
    data = random_state.randn(10, 30)
    # 生成一个10x30的随机数据矩阵
    sims = rbf_kernel(data)
    # 计算径向基函数核矩阵
    embedding_1 = spectral_embedding(sims)
    # 使用谱嵌入计算嵌入结果1
    embedding_2 = spectral_embedding(sims)
    # 再次使用谱嵌入计算嵌入结果2
    assert_array_almost_equal(embedding_1, embedding_2)
    # 断言两次嵌入结果应该几乎相等


def test_spectral_embedding_unnormalized():
    # 测试 spectral_embedding 是否正确处理未归一化的拉普拉斯矩阵
    random_state = np.random.RandomState(36)
    # 创建随机数生成器，指定种子为36
    data = random_state.randn(10, 30)
    # 生成一个10x30的随机数据矩阵
    sims = rbf_kernel(data)
    # 计算径向基函数核矩阵
    n_components = 8
    embedding_1 = spectral_embedding(
        sims, norm_laplacian=False, n_components=n_components, drop_first=False
    )
    # 使用 spectral_embedding 计算嵌入结果1，关闭拉普拉斯归一化，指定嵌入维度和保留第一个特征向量

    # 使用密集特征值求解计算拉普拉斯矩阵和扩散映射
    laplacian, dd = csgraph_laplacian(sims, normed=False, return_diag=True)
    _, diffusion_map = eigh(laplacian)
    embedding_2 = diffusion_map.T[:n_components]
    embedding_2 = _deterministic_vector_sign_flip(embedding_2).T

    assert_array_almost_equal(embedding_1, embedding_2)
    # 断言嵌入结果1和手动计算的嵌入结果2应该几乎相等


def test_spectral_embedding_first_eigen_vector():
    # 测试 spectral_embedding 的第一个特征向量是否恒定，第二个是否不恒定（对于连通图）
    random_state = np.random.RandomState(36)
    # 创建随机数生成器，指定种子为36
    data = random_state.randn(10, 30)
    # 生成一个10x30的随机数据矩阵
    sims = rbf_kernel(data)
    # 计算径向基函数核矩阵
    n_components = 2

    for seed in range(10):
        # 遍历10个不同的种子
        embedding = spectral_embedding(
            sims,
            norm_laplacian=False,
            n_components=n_components,
            drop_first=False,
            random_state=seed,
        )

        assert np.std(embedding[:, 0]) == pytest.approx(0)
        # 断言第一个特征向量的标准差应接近于0
        assert np.std(embedding[:, 1]) > 1e-3
        # 断言第二个特征向量的标准差应大于0.001


@pytest.mark.parametrize(
    "eigen_solver",
    [
        "arpack",
        "lobpcg",
        pytest.param("amg", marks=skip_if_no_pyamg),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spectral_embedding_preserves_dtype(eigen_solver, dtype):
    """检查 SpectralEmbedding 是否保持拟合属性和转换数据的数据类型不变。

    理想情况下，这个测试应该由公共测试 `check_transformer_preserve_dtypes` 覆盖。
    但是，这个测试只运行在实现 `transform` 的转换器上，而 `SpectralEmbedding`
    只实现了 `fit_transform`。
    """
    X = S.astype(dtype)
    # 将输入数据类型转换为指定的 dtype
    se = SpectralEmbedding(
        n_components=2, affinity="rbf", eigen_solver=eigen_solver, random_state=0
    )
    # 创建 SpectralEmbedding 对象
    X_trans = se.fit_transform(X)
    # 使用 SpectralEmbedding 对象拟合并转换数据

    assert X_trans.dtype == dtype
    # 断言转换后的数据类型应为指定的 dtype
    assert se.embedding_.dtype == dtype
    # 断言拟合后的嵌入结果数据类型应为指定的 dtype
    assert se.affinity_matrix_.dtype == dtype
    # 断言亲和力矩阵数据类型应为指定的 dtype


@pytest.mark.skipif(
    pyamg_available,
    reason="PyAMG is installed and we should not test for an error.",
)
def test_error_pyamg_not_available():
    se_precomp = SpectralEmbedding(
        n_components=2,
        affinity="rbf",
        eigen_solver="amg",
    )
    # 创建 SpectralEmbedding 对象，使用不可用的 eigen_solver
    # 定义错误消息字符串，用于指示特定情况下的错误信息
    err_msg = "The eigen_solver was set to 'amg', but pyamg is not available."
    
    # 使用 pytest 的上下文管理器 `pytest.raises` 来捕获预期的 ValueError 异常，
    # 并检查异常消息是否与预设的 `err_msg` 匹配
    with pytest.raises(ValueError, match=err_msg):
        # 在 `se_precomp` 对象上调用 `fit_transform` 方法，预期会抛出 ValueError 异常
        se_precomp.fit_transform(S)
# TODO: Remove when pyamg removes the use of np.find_common_type
# 使用 pytest.mark.filterwarnings 忽略特定警告信息
@pytest.mark.filterwarnings(
    "ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*"
)
# 使用 pytest.mark.parametrize 来参数化测试函数，指定 solver 参数为 "arpack", "amg", "lobpcg" 中的一个
@pytest.mark.parametrize("solver", ["arpack", "amg", "lobpcg"])
# 使用 pytest.mark.parametrize 来参数化测试函数，参数为 csr_container 为预定义的 CSR_CONTAINERS 之一
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_spectral_eigen_tol_auto(monkeypatch, solver, csr_container):
    """Test that `eigen_tol="auto"` is resolved correctly"""
    # 如果 solver 是 "amg" 且 pyamg 不可用，则跳过测试
    if solver == "amg" and not pyamg_available:
        pytest.skip("PyAMG is not available.")

    # 生成两个簇的样本数据 X
    X, _ = make_blobs(
        n_samples=200, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )
    # 计算样本间的距离矩阵 D
    D = pairwise_distances(X)  # Distance matrix
    # 计算相似性矩阵 S，基于距离矩阵的最大值减去距离矩阵的每个元素
    S = np.max(D) - D  # Similarity matrix

    # 根据 solver 类型选择相应的求解函数
    solver_func = eigsh if solver == "arpack" else lobpcg
    # 根据 solver 类型设置默认的 tol 值
    default_value = 0 if solver == "arpack" else None
    # 如果 solver 是 "amg"，将 S 转换为 csr_container 类型
    if solver == "amg":
        S = csr_container(S)

    # 创建 solver_func 的 Mock 对象
    mocked_solver = Mock(side_effect=solver_func)

    # 使用 monkeypatch 修改 _spectral_embedding 模块中 solver_func 方法的行为为 mocked_solver
    monkeypatch.setattr(_spectral_embedding, solver_func.__qualname__, mocked_solver)

    # 调用 spectral_embedding 函数，使用 solver_func 求解特征向量，设置 eigen_tol 为 "auto"
    spectral_embedding(S, random_state=42, eigen_solver=solver, eigen_tol="auto")
    # 断言 mocked_solver 被调用过
    mocked_solver.assert_called()

    # 获取 mocked_solver 调用时传递的参数
    _, kwargs = mocked_solver.call_args
    # 断言 kwargs 中的 tol 参数值为默认值 default_value
    assert kwargs["tol"] == default_value
```