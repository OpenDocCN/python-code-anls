# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_neighbors_pipeline.py`

```
"""
This is testing the equivalence between some estimators with internal nearest
neighbors computations, and the corresponding pipeline versions with
KNeighborsTransformer or RadiusNeighborsTransformer to precompute the
neighbors.
"""

import numpy as np

from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsTransformer,
    LocalOutlierFactor,
    RadiusNeighborsRegressor,
    RadiusNeighborsTransformer,
)
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal


def test_spectral_clustering():
    # Test chaining KNeighborsTransformer and SpectralClustering
    n_neighbors = 5
    X, _ = make_blobs(random_state=0)

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="connectivity"),
        SpectralClustering(
            n_neighbors=n_neighbors, affinity="precomputed", random_state=42
        ),
    )
    est_compact = SpectralClustering(
        n_neighbors=n_neighbors, affinity="nearest_neighbors", random_state=42
    )
    labels_compact = est_compact.fit_predict(X)
    labels_chain = est_chain.fit_predict(X)
    assert_array_almost_equal(labels_chain, labels_compact)


def test_spectral_embedding():
    # Test chaining KNeighborsTransformer and SpectralEmbedding
    n_neighbors = 5

    n_samples = 1000
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    S, true_labels = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="connectivity"),
        SpectralEmbedding(
            n_neighbors=n_neighbors, affinity="precomputed", random_state=42
        ),
    )
    est_compact = SpectralEmbedding(
        n_neighbors=n_neighbors, affinity="nearest_neighbors", random_state=42
    )
    St_compact = est_compact.fit_transform(S)
    St_chain = est_chain.fit_transform(S)
    assert_array_almost_equal(St_chain, St_compact)


def test_dbscan():
    # Test chaining RadiusNeighborsTransformer and DBSCAN
    radius = 0.3
    n_clusters = 3
    X = generate_clustered_data(n_clusters=n_clusters)

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        RadiusNeighborsTransformer(radius=radius, mode="distance"),
        DBSCAN(metric="precomputed", eps=radius),
    )
    est_compact = DBSCAN(eps=radius)

    labels_chain = est_chain.fit_predict(X)
    labels_compact = est_compact.fit_predict(X)


注释：


# 测试使用 KNeighborsTransformer 和 SpectralClustering 进行链式调用
n_neighbors = 5
X, _ = make_blobs(random_state=0)

# 比较链式调用版本和紧凑版本
est_chain = make_pipeline(
    KNeighborsTransformer(n_neighbors=n_neighbors, mode="connectivity"),
    SpectralClustering(
        n_neighbors=n_neighbors, affinity="precomputed", random_state=42
    ),
)
est_compact = SpectralClustering(
    n_neighbors=n_neighbors, affinity="nearest_neighbors", random_state=42
)
labels_compact = est_compact.fit_predict(X)
labels_chain = est_chain.fit_predict(X)
assert_array_almost_equal(labels_chain, labels_compact)


# 测试使用 KNeighborsTransformer 和 SpectralEmbedding 进行链式调用
n_neighbors = 5

n_samples = 1000
centers = np.array(
    [
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 5.0, 1.0],
    ]
)
S, true_labels = make_blobs(
    n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
)

# 比较链式调用版本和紧凑版本
est_chain = make_pipeline(
    KNeighborsTransformer(n_neighbors=n_neighbors, mode="connectivity"),
    SpectralEmbedding(
        n_neighbors=n_neighbors, affinity="precomputed", random_state=42
    ),
)
est_compact = SpectralEmbedding(
    n_neighbors=n_neighbors, affinity="nearest_neighbors", random_state=42
)
St_compact = est_compact.fit_transform(S)
St_chain = est_chain.fit_transform(S)
assert_array_almost_equal(St_chain, St_compact)


# 测试使用 RadiusNeighborsTransformer 和 DBSCAN 进行链式调用
radius = 0.3
n_clusters = 3
X = generate_clustered_data(n_clusters=n_clusters)

# 比较链式调用版本和紧凑版本
est_chain = make_pipeline(
    RadiusNeighborsTransformer(radius=radius, mode="distance"),
    DBSCAN(metric="precomputed", eps=radius),
)
est_compact = DBSCAN(eps=radius)

labels_chain = est_chain.fit_predict(X)
labels_compact = est_compact.fit_predict(X)
    # 使用 NumPy 的 assert_array_almost_equal 函数比较 labels_chain 和 labels_compact 两个数组的近似相等性
    assert_array_almost_equal(labels_chain, labels_compact)
# 测试使用 KNeighborsTransformer 和 Isomap 进行链式操作，其中 neighbors_algorithm='precomputed'
def test_isomap():
    # 设定算法和邻居数
    algorithm = "auto"
    n_neighbors = 10

    # 生成随机数据集 X 和 X2
    X, _ = make_blobs(random_state=0)
    X2, _ = make_blobs(random_state=1)

    # 比较链式版本和紧凑版本
    est_chain = make_pipeline(
        KNeighborsTransformer(
            n_neighbors=n_neighbors, algorithm=algorithm, mode="distance"
        ),
        Isomap(n_neighbors=n_neighbors, metric="precomputed"),
    )
    est_compact = Isomap(n_neighbors=n_neighbors, neighbors_algorithm=algorithm)

    # 对数据 X 进行拟合和转换
    Xt_chain = est_chain.fit_transform(X)
    Xt_compact = est_compact.fit_transform(X)
    assert_array_almost_equal(Xt_chain, Xt_compact)

    # 对数据 X2 进行转换
    Xt_chain = est_chain.transform(X2)
    Xt_compact = est_compact.transform(X2)
    assert_array_almost_equal(Xt_chain, Xt_compact)


# 测试使用 KNeighborsTransformer 和 TSNE 进行链式操作
def test_tsne():
    # 设定最大迭代次数和困惑度
    max_iter = 250
    perplexity = 5
    n_neighbors = int(3.0 * perplexity + 1)

    # 随机生成数据集 X
    rng = np.random.RandomState(0)
    X = rng.randn(20, 2)

    # 遍历指定的距离度量方法
    for metric in ["minkowski", "sqeuclidean"]:
        # 比较链式版本和紧凑版本
        est_chain = make_pipeline(
            KNeighborsTransformer(
                n_neighbors=n_neighbors, mode="distance", metric=metric
            ),
            TSNE(
                init="random",
                metric="precomputed",
                perplexity=perplexity,
                method="barnes_hut",
                random_state=42,
                max_iter=max_iter,
            ),
        )
        est_compact = TSNE(
            init="random",
            metric=metric,
            perplexity=perplexity,
            max_iter=max_iter,
            method="barnes_hut",
            random_state=42,
        )

        # 对数据 X 进行拟合和转换
        Xt_chain = est_chain.fit_transform(X)
        Xt_compact = est_compact.fit_transform(X)
        assert_array_almost_equal(Xt_chain, Xt_compact)


# 测试使用 KNeighborsTransformer 和 LocalOutlierFactor 进行链式操作，novelty=False
def test_lof_novelty_false():
    # 设定邻居数
    n_neighbors = 4

    # 随机生成数据集 X
    rng = np.random.RandomState(0)
    X = rng.randn(40, 2)

    # 比较链式版本和紧凑版本
    est_chain = make_pipeline(
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance"),
        LocalOutlierFactor(
            metric="precomputed",
            n_neighbors=n_neighbors,
            novelty=False,
            contamination="auto",
        ),
    )
    est_compact = LocalOutlierFactor(
        n_neighbors=n_neighbors, novelty=False, contamination="auto"
    )

    # 对数据 X 进行拟合和预测
    pred_chain = est_chain.fit_predict(X)
    pred_compact = est_compact.fit_predict(X)
    assert_array_almost_equal(pred_chain, pred_compact)


# 测试使用 KNeighborsTransformer 和 LocalOutlierFactor 进行链式操作，novelty=True
def test_lof_novelty_true():
    # 设定邻居数
    n_neighbors = 4

    # 随机生成数据集 X1 和 X2
    rng = np.random.RandomState(0)
    X1 = rng.randn(40, 2)
    X2 = rng.randn(40, 2)
    # 创建一个使用管道方法的异常检测估算器（estimator）
    est_chain = make_pipeline(
        # 使用 KNeighborsTransformer 进行特征转换，使用距离模式，设定邻居数
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance"),
        # 使用 LocalOutlierFactor 进行异常检测，使用预先计算的距离矩阵作为度量
        LocalOutlierFactor(
            metric="precomputed",
            n_neighbors=n_neighbors,
            novelty=True,
            contamination="auto",
        ),
    )
    # 创建一个紧凑版本的异常检测估算器（estimator）
    est_compact = LocalOutlierFactor(
        # 设定邻居数和异常检测模式为 novelty
        n_neighbors=n_neighbors, novelty=True, contamination="auto"
    )

    # 对 X1 进行拟合并预测异常值，使用管道方法的估算器
    pred_chain = est_chain.fit(X1).predict(X2)
    # 对 X1 进行拟合并预测异常值，使用紧凑版本的估算器
    pred_compact = est_compact.fit(X1).predict(X2)
    # 检查两种方法得到的预测结果是否几乎相等
    assert_array_almost_equal(pred_chain, pred_compact)


这段代码用于比较使用管道方法和紧凑版本两种方式进行异常检测估算的结果。
# 定义一个测试函数，用于测试 KNeighborsRegressor 的功能
def test_kneighbors_regressor():
    # 初始化一个随机数生成器，用于生成一些测试数据
    rng = np.random.RandomState(0)
    # 创建一个 40x5 的随机矩阵 X，数值范围在 [-1, 1) 之间
    X = 2 * rng.rand(40, 5) - 1
    # 创建另一个 40x5 的随机矩阵 X2，数值范围也在 [-1, 1) 之间
    X2 = 2 * rng.rand(40, 5) - 1
    # 创建一个 40x1 的随机数组 y，数值范围在 [0, 1) 之间
    y = rng.rand(40, 1)

    # 设置 K 近邻算法的参数
    n_neighbors = 12
    radius = 1.5
    # 为了保证在 k 近邻转换器和半径近邻转换器之间的等效性，我们预先计算比必要的邻居更多的数量
    factor = 2

    # 创建 K 近邻转换器对象，使用距离模式
    k_trans = KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance")
    # 创建另一个 K 近邻转换器对象，邻居数量是原始值的 factor 倍，使用距离模式
    k_trans_factor = KNeighborsTransformer(
        n_neighbors=int(n_neighbors * factor), mode="distance"
    )

    # 创建半径近邻转换器对象，使用距离模式
    r_trans = RadiusNeighborsTransformer(radius=radius, mode="distance")
    # 创建另一个半径近邻转换器对象，半径是原始值的 factor 倍，使用距离模式
    r_trans_factor = RadiusNeighborsTransformer(
        radius=int(radius * factor), mode="distance"
    )

    # 创建 K 近邻回归器对象，设定邻居数量
    k_reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    # 创建半径近邻回归器对象，设定半径
    r_reg = RadiusNeighborsRegressor(radius=radius)

    # 将转换器和回归器对象组合成元组列表，用于后续的测试
    test_list = [
        (k_trans, k_reg),
        (k_trans_factor, r_reg),
        (r_trans, r_reg),
        (r_trans_factor, k_reg),
    ]

    # 遍历测试列表中的每个元组，执行测试
    for trans, reg in test_list:
        # 创建回归器的克隆版本，用于紧凑版本的比较
        reg_compact = clone(reg)
        # 创建回归器的克隆版本，设置参数以使用预先计算的距离矩阵
        reg_precomp = clone(reg)
        reg_precomp.set_params(metric="precomputed")

        # 使用 make_pipeline 创建一个包含转换器和预计算回归器的管道
        reg_chain = make_pipeline(clone(trans), reg_precomp)

        # 使用管道进行拟合和预测，并获取预测结果
        y_pred_chain = reg_chain.fit(X, y).predict(X2)
        # 使用紧凑版本的回归器进行拟合和预测，并获取预测结果
        y_pred_compact = reg_compact.fit(X, y).predict(X2)
        
        # 断言预测结果的近似相等性
        assert_array_almost_equal(y_pred_chain, y_pred_compact)
```