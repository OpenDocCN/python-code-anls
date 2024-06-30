# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_spectral.py`

```
"""Testing for Spectral Clustering methods"""

# 导入必要的库和模块
import pickle  # 用于对象的序列化和反序列化
import re  # 正则表达式模块

import numpy as np  # 数组操作库
import pytest  # 测试框架
from scipy.linalg import LinAlgError  # 线性代数错误处理

# 导入 Spectral Clustering 相关的类和函数
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.cluster._spectral import cluster_qr, discretize
from sklearn.datasets import make_blobs  # 生成聚类用的数据
from sklearn.feature_extraction import img_to_graph  # 图像转换为图的工具
from sklearn.metrics import adjusted_rand_score  # 调整兰德指数，用于评估聚类效果
from sklearn.metrics.pairwise import kernel_metrics, rbf_kernel  # 核函数相关
from sklearn.neighbors import NearestNeighbors  # 最近邻模型
from sklearn.utils import check_random_state  # 随机状态检查
from sklearn.utils._testing import assert_array_equal  # 断言数组相等
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS  # 稀疏矩阵容器的修复

try:
    from pyamg import smoothed_aggregation_solver  # 尝试导入 PyAMG 的平滑聚集求解器
    amg_loaded = True  # 如果成功导入，则设置为 True
except ImportError:
    amg_loaded = False  # 导入失败则设置为 False

# 创建模拟数据的中心点和样本数据
centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
X, _ = make_blobs(
    n_samples=60,
    n_features=2,
    centers=centers,
    cluster_std=0.4,
    shuffle=True,
    random_state=0,
)

# 使用参数化测试对 Spectral Clustering 进行多组测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)  # 参数化 CSR 格式的稀疏矩阵容器
@pytest.mark.parametrize("eigen_solver", ("arpack", "lobpcg"))  # 参数化特征值求解器
@pytest.mark.parametrize("assign_labels", ("kmeans", "discretize", "cluster_qr"))  # 参数化标签分配方法
def test_spectral_clustering(eigen_solver, assign_labels, csr_container):
    S = np.array(  # 创建一个预计算的相似度矩阵 S
        [
            [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    for mat in (S, csr_container(S)):  # 遍历使用不同稀疏矩阵容器包装的 S
        model = SpectralClustering(  # 创建 SpectralClustering 模型
            random_state=0,
            n_clusters=2,
            affinity="precomputed",  # 使用预计算的相似度矩阵
            eigen_solver=eigen_solver,  # 特征值求解器
            assign_labels=assign_labels,  # 标签分配方法
        ).fit(mat)  # 在给定的相似度矩阵上训练模型
        labels = model.labels_  # 获取模型预测的类别标签

        # 调整兰德指数，确保与预期标签的一致性
        if labels[0] == 0:
            labels = 1 - labels
        assert adjusted_rand_score(labels, [1, 1, 1, 0, 0, 0, 0]) == 1  # 断言调整兰德指数为 1

        model_copy = pickle.loads(pickle.dumps(model))  # 深拷贝模型并反序列化
        assert model_copy.n_clusters == model.n_clusters  # 断言拷贝后的模型参数一致
        assert model_copy.eigen_solver == model.eigen_solver  # 断言拷贝后的模型参数一致
        assert_array_equal(model_copy.labels_, model.labels_)  # 断言拷贝后的标签与原模型一致


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)  # 参数化 COO 格式的稀疏矩阵容器
@pytest.mark.parametrize("assign_labels", ("kmeans", "discretize", "cluster_qr"))  # 参数化标签分配方法
def test_spectral_clustering_sparse(assign_labels, coo_container):
    X, y = make_blobs(  # 生成聚类用的数据
        n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )

    S = rbf_kernel(X, gamma=1)  # 计算 RBF 核函数生成相似度矩阵
    S = np.maximum(S - 1e-4, 0)  # 对相似度矩阵进行处理，确保非负
    S = coo_container(S)  # 使用指定的稀疏矩阵容器包装相似度矩阵

    labels = (  # 使用稀疏矩阵进行 Spectral Clustering
        SpectralClustering(
            random_state=0,
            n_clusters=2,
            affinity="precomputed",  # 使用预计算的相似度矩阵
            assign_labels=assign_labels,  # 标签分配方法
        )
        .fit(S)  # 在给定的相似度矩阵上训练模型
        .labels_  # 获取模型预测的类别标签
    )
    # 使用 adjusted_rand_score 函数比较 y 和 labels 两个参数的相似度是否为1，断言此表达式为真
    assert adjusted_rand_score(y, labels) == 1
def test_precomputed_nearest_neighbors_filtering():
    # 测试在包含过多邻居时的预计算图过滤
    # 创建包含两个中心的样本集合
    X, y = make_blobs(
        n_samples=200, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )

    n_neighbors = 2
    results = []
    for additional_neighbors in [0, 10]:
        # 训练最近邻模型，设置邻居数目为 n_neighbors + additional_neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors + additional_neighbors).fit(X)
        # 构建最近邻连接图
        graph = nn.kneighbors_graph(X, mode="connectivity")
        # 使用预计算的最近邻亲和性进行谱聚类
        labels = (
            SpectralClustering(
                random_state=0,
                n_clusters=2,
                affinity="precomputed_nearest_neighbors",
                n_neighbors=n_neighbors,
            )
            .fit(graph)
            .labels_
        )
        results.append(labels)

    # 断言两次结果相等
    assert_array_equal(results[0], results[1])


def test_affinities():
    # 注意：在以下代码中，random_state 已被选择，以确保在 OSX 和 Linux 上构建时产生稳定的特征值分解
    X, y = make_blobs(
        n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )
    # 最近邻亲和性
    sp = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=0)
    # 断言警告信息中包含 "not fully connected"
    with pytest.warns(UserWarning, match="not fully connected"):
        sp.fit(X)
    # 断言调整后的兰德指数为 1
    assert adjusted_rand_score(y, sp.labels_) == 1

    # 使用 gamma 参数的谱聚类
    sp = SpectralClustering(n_clusters=2, gamma=2, random_state=0)
    labels = sp.fit(X).labels_
    # 断言调整后的兰德指数为 1
    assert adjusted_rand_score(y, labels) == 1

    X = check_random_state(10).rand(10, 5) * 10

    kernels_available = kernel_metrics()
    for kern in kernels_available:
        # 加性卡方核会产生负的相似性矩阵，对谱聚类没有意义
        if kern != "additive_chi2":
            sp = SpectralClustering(n_clusters=2, affinity=kern, random_state=0)
            labels = sp.fit(X).labels_
            assert (X.shape[0],) == labels.shape

    # 使用 lambda 函数作为亲和性度量
    sp = SpectralClustering(n_clusters=2, affinity=lambda x, y: 1, random_state=0)
    labels = sp.fit(X).labels_
    assert (X.shape[0],) == labels.shape

    def histogram(x, y, **kwargs):
        # 作为可调用函数实现的直方图核
        assert kwargs == {}  # 没有未请求的 kernel_params 参数
        return np.minimum(x, y).sum()

    sp = SpectralClustering(n_clusters=2, affinity=histogram, random_state=0)
    labels = sp.fit(X).labels_
    assert (X.shape[0],) == labels.shape


def test_cluster_qr():
    # cluster_qr 本身不应用于聚类通用数据，只适用于谱聚类中特征向量的行
    # 但是，cluster_qr 必须仍然对通用固定输入的不同 dtype 保留标签，即使标签可能无意义。
    random_state = np.random.RandomState(seed=8)
    n_samples, n_components = 10, 5
    data = random_state.randn(n_samples, n_components)
    labels_float64 = cluster_qr(data.astype(np.float64))
    # 每个样本被分配一个聚类标识符
    assert labels_float64.shape == (n_samples,)
    # 所有的组件都应该被分配到聚类中
    assert np.array_equal(np.unique(labels_float64), np.arange(n_components))
    # 单精度数据应该得到相同的聚类分配结果
    labels_float32 = cluster_qr(data.astype(np.float32))
    # 检查双精度和单精度数据的聚类结果是否一致
    assert np.array_equal(labels_float64, labels_float32)
def test_cluster_qr_permutation_invariance():
    # cluster_qr 必须对样本排列的置换具有不变性。
    # 设置随机种子为8
    random_state = np.random.RandomState(seed=8)
    # 设定样本数量和成分数量
    n_samples, n_components = 100, 5
    # 生成随机数据矩阵
    data = random_state.randn(n_samples, n_components)
    # 生成样本的置换索引
    perm = random_state.permutation(n_samples)
    # 断言两个经过排列的数据集相等
    assert np.array_equal(
        cluster_qr(data)[perm],
        cluster_qr(data[perm]),
    )


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
@pytest.mark.parametrize("n_samples", [50, 100, 150, 500])
def test_discretize(n_samples, coo_container):
    # 使用噪声分配矩阵测试离散化
    # 设置随机种子为8
    random_state = np.random.RandomState(seed=8)
    # 循环不同的类别数量范围
    for n_class in range(2, 10):
        # 随机生成类别标签
        y_true = random_state.randint(0, n_class + 1, n_samples)
        y_true = np.array(y_true, float)
        # 生成噪声类别分配矩阵
        y_indicator = coo_container(
            (np.ones(n_samples), (np.arange(n_samples), y_true)),
            shape=(n_samples, n_class + 1),
        )
        # 添加噪声到真实标签
        y_true_noisy = y_indicator.toarray() + 0.1 * random_state.randn(
            n_samples, n_class + 1
        )
        # 进行离散化操作
        y_pred = discretize(y_true_noisy, random_state=random_state)
        # 断言调整后的兰德指数大于0.8
        assert adjusted_rand_score(y_true, y_pred) > 0.8


# TODO: Remove when pyamg does replaces sp.rand call with np.random.rand
# https://github.com/scikit-learn/scikit-learn/issues/15913
@pytest.mark.filterwarnings(
    "ignore:scipy.rand is deprecated:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of np.float
@pytest.mark.filterwarnings(
    "ignore:`np.float` is a deprecated alias:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of pinv2
@pytest.mark.filterwarnings(
    "ignore:scipy.linalg.pinv2 is deprecated:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of np.find_common_type
@pytest.mark.filterwarnings(
    "ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*"
)
def test_spectral_clustering_with_arpack_amg_solvers():
    # 测试使用 arpack 和 amg 求解器的谱聚类结果是否一致
    # 基于 plot_segmentation_toy.py 中的玩具示例

    # 创建一个小的二元图像
    x, y = np.indices((40, 40))

    center1, center2 = (14, 12), (20, 25)
    radius1, radius2 = 8, 7

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1**2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2**2

    circles = circle1 | circle2
    mask = circles.copy()
    img = circles.astype(float)

    graph = img_to_graph(img, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())

    labels_arpack = spectral_clustering(
        graph, n_clusters=2, eigen_solver="arpack", random_state=0
    )

    # 断言聚类后的标签数量为2
    assert len(np.unique(labels_arpack)) == 2
    # 如果已经加载了AMG求解器，则执行以下代码块
    if amg_loaded:
        # 使用谱聚类算法对图形进行聚类，要求生成2个簇，使用AMG作为特征值求解器，随机种子为0
        labels_amg = spectral_clustering(
            graph, n_clusters=2, eigen_solver="amg", random_state=0
        )
        # 断言使用ARPACK和AMG求解器得到的标签完全匹配
        assert adjusted_rand_score(labels_arpack, labels_amg) == 1
    # 如果未加载AMG求解器，则应该引发值错误异常
    else:
        with pytest.raises(ValueError):
            # 使用谱聚类算法对图形进行聚类，要求生成2个簇，使用AMG作为特征值求解器，随机种子为0
            spectral_clustering(graph, n_clusters=2, eigen_solver="amg", random_state=0)
def test_n_components():
    # 添加 n_components 后，验证结果不同，且默认 n_components = n_clusters
    X, y = make_blobs(
        n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )
    # 创建 SpectralClustering 实例，设置 n_clusters=2
    sp = SpectralClustering(n_clusters=2, random_state=0)
    # 对数据 X 进行聚类，并获取标签
    labels = sp.fit(X).labels_
    
    # 设置 n_components = n_clusters，验证结果是否相同
    labels_same_ncomp = (
        SpectralClustering(n_clusters=2, n_components=2, random_state=0).fit(X).labels_
    )
    # 验证 n_components 默认等于 n_clusters
    assert_array_equal(labels, labels_same_ncomp)

    # 验证 n_components 对结果的影响
    # 默认 n_clusters=8，设置 n_components=2
    labels_diff_ncomp = (
        SpectralClustering(n_components=2, random_state=0).fit(X).labels_
    )
    assert not np.array_equal(labels, labels_diff_ncomp)


@pytest.mark.parametrize("assign_labels", ("kmeans", "discretize", "cluster_qr"))
def test_verbose(assign_labels, capsys):
    # 检查 KMeans 的详细模式，以增强覆盖率
    X, y = make_blobs(
        n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )

    # 创建 SpectralClustering 实例，设置 n_clusters=2，启用 verbose 模式
    SpectralClustering(n_clusters=2, random_state=42, verbose=1).fit(X)

    captured = capsys.readouterr()

    assert re.search(r"Computing label assignment using", captured.out)

    if assign_labels == "kmeans":
        assert re.search(r"Initialization complete", captured.out)
        assert re.search(r"Iteration [0-9]+, inertia", captured.out)


def test_spectral_clustering_np_matrix_raises():
    """验证 spectral_clustering 当传入 np.matrix 时抛出详细的错误信息"""
    X = np.matrix([[0.0, 2.0], [2.0, 0.0]])

    msg = r"np\.matrix is not supported. Please convert to a numpy array"
    with pytest.raises(TypeError, match=msg):
        spectral_clustering(X)


def test_spectral_clustering_not_infinite_loop(capsys, monkeypatch):
    """验证 discretize 当 SVD 无法收敛时抛出 LinAlgError 错误"""
    
    # 设置新的 svd 函数，使其在调用时抛出 LinAlgError
    def new_svd(*args, **kwargs):
        raise LinAlgError()

    # 使用 monkeypatch 替换 np.linalg.svd 函数
    monkeypatch.setattr(np.linalg, "svd", new_svd)
    vectors = np.ones((10, 4))

    # 验证是否抛出 LinAlgError，并检查错误信息匹配
    with pytest.raises(LinAlgError, match="SVD did not converge"):
        discretize(vectors)
```