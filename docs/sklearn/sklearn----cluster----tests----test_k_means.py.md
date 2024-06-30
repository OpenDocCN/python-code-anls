# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_k_means.py`

```
# 导入所需的模块和库
import re
import sys
from io import StringIO

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest，用于编写和运行测试
from scipy import sparse as sp  # 导入sparse子模块，用于稀疏矩阵操作

# 导入scikit-learn相关模块和函数
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from sklearn.cluster._k_means_common import (
    _euclidean_dense_dense_wrapper,
    _euclidean_sparse_dense_wrapper,
    _inertia_dense,
    _inertia_sparse,
    _is_same_clustering,
    _relocate_empty_clusters_dense,
    _relocate_empty_clusters_sparse,
)
from sklearn.cluster._kmeans import _labels_inertia, _mini_batch_step
from sklearn.datasets import make_blobs  # 生成聚类测试数据的函数
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
    create_memmap_backed_data,
)
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS  # 修复旧版本兼容性的工具函数和类
from sklearn.utils.parallel import _get_threadpool_controller  # 获取线程池控制器

# 非居中的稀疏中心点，用于检查聚类结果
centers = np.array(
    [
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 4.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 5.0, 1.0],
    ]
)

n_samples = 100  # 样本数
n_clusters, n_features = centers.shape  # 簇数和特征数
X, true_labels = make_blobs(
    n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
)  # 生成用于聚类的测试数据
X_as_any_csr = [container(X) for container in CSR_CONTAINERS]  # 将数据转换为稀疏格式的容器
data_containers = [np.array] + CSR_CONTAINERS  # 数据容器列表
data_containers_ids = (
    ["dense", "sparse_matrix", "sparse_array"]
    if len(X_as_any_csr) == 2
    else ["dense", "sparse_matrix"]
)  # 数据容器的ID列表

@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
@pytest.mark.parametrize("algo", ["lloyd", "elkan"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_results(array_constr, algo, dtype):
    """
    检查KMeans在小型数据集上的工作是否与手动计算的预期结果一致。
    """
    # 准备测试数据
    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
    sample_weight = [3, 1, 1, 3]  # 样本权重
    init_centers = np.array([[0, 0], [1, 1]], dtype=dtype)  # 初始聚类中心

    # 预期的结果
    expected_labels = [0, 0, 1, 1]  # 预期的类别标签
    expected_inertia = 0.375  # 预期的惯性
    expected_centers = np.array([[0.125, 0], [0.875, 1]], dtype=dtype)  # 预期的聚类中心
    expected_n_iter = 2  # 预期的迭代次数

    # 创建并拟合KMeans模型
    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X, sample_weight=sample_weight)

    # 断言检查
    assert_array_equal(kmeans.labels_, expected_labels)  # 检查类别标签
    assert_allclose(kmeans.inertia_, expected_inertia)  # 检查惯性
    assert_allclose(kmeans.cluster_centers_, expected_centers)  # 检查聚类中心
    assert kmeans.n_iter_ == expected_n_iter  # 检查迭代次数


@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
@pytest.mark.parametrize("algo", ["lloyd", "elkan"])
def test_kmeans_relocated_clusters(array_constr, algo):
    """
    检查空聚类是否按预期重新定位。
    """
    # 创建一个包含四个点的二维数组 X
    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])

    # 初始化聚类中心，其中第二个中心点距离其他点太远，在第一次迭代时可能为空
    init_centers = np.array([[0.5, 0.5], [3, 3]])

    # 使用 KMeans 算法进行聚类，设置为 2 个聚类中心，只进行一次初始化，使用给定的初始中心点数组
    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X)

    # 预期迭代次数为 3 次
    expected_n_iter = 3
    # 预期的惯性值为 0.25
    expected_inertia = 0.25
    # 断言实际的惯性值与预期的惯性值相近
    assert_allclose(kmeans.inertia_, expected_inertia)
    # 断言实际的迭代次数与预期的迭代次数相等
    assert kmeans.n_iter_ == expected_n_iter

    # 在此示例中，有两种可接受的方法来重新定位聚类中心，输出取决于如何处理 argpartition 策略中的平局情况。
    # 我们接受两种输出结果。
    try:
        # 预期的标签顺序
        expected_labels = [0, 0, 1, 1]
        # 预期的聚类中心坐标
        expected_centers = [[0.25, 0], [0.75, 1]]
        # 断言实际的标签与预期的标签相等
        assert_array_equal(kmeans.labels_, expected_labels)
        # 断言实际的聚类中心与预期的聚类中心在数值上相近
        assert_allclose(kmeans.cluster_centers_, expected_centers)
    except AssertionError:
        # 如果第一组断言失败，则采用另一组预期的标签顺序和聚类中心坐标
        expected_labels = [1, 1, 0, 0]
        expected_centers = [[0.75, 1.0], [0.25, 0.0]]
        assert_array_equal(kmeans.labels_, expected_labels)
        assert_allclose(kmeans.cluster_centers_, expected_centers)
@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
def test_relocate_empty_clusters(array_constr):
    # 对 _relocate_empty_clusters_(dense/sparse) 辅助函数进行测试

    # 创建一个包含3个不同大小明显簇的合成数据集
    X = np.array([-10.0, -9.5, -9, -8.5, -8, -1, 1, 9, 9.5, 10]).reshape(-1, 1)
    X = array_constr(X)
    sample_weight = np.ones(10)

    # 将所有中心点初始化为X的第一个点
    centers_old = np.array([-10.0, -10, -10]).reshape(-1, 1)

    # 使用这种初始化方式，所有点都将被分配到第一个中心点
    # 此时centers_new中的中心点是其包含点的加权和（如果不为空），否则与之前相同。
    centers_new = np.array([-16.5, -10, -10]).reshape(-1, 1)
    weight_in_clusters = np.array([10.0, 0, 0])
    labels = np.zeros(10, dtype=np.int32)

    if array_constr is np.array:
        _relocate_empty_clusters_dense(
            X, sample_weight, centers_old, centers_new, weight_in_clusters, labels
        )
    else:
        _relocate_empty_clusters_sparse(
            X.data,
            X.indices,
            X.indptr,
            sample_weight,
            centers_old,
            centers_new,
            weight_in_clusters,
            labels,
        )

    # 重新分配方案会将距离中心最远的2个点分配给2个空的簇，即点10和点9.9。
    # 第一个中心点将更新以包含其余8个点。
    assert_array_equal(weight_in_clusters, [8, 1, 1])
    assert_allclose(centers_new, [[-36], [10], [9.5]])


@pytest.mark.parametrize("distribution", ["normal", "blobs"])
@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
@pytest.mark.parametrize("tol", [1e-2, 1e-8, 1e-100, 0])
def test_kmeans_elkan_results(distribution, array_constr, tol, global_random_seed):
    # 检查在Lloyd和Elkan算法之间的结果是否一致

    rnd = np.random.RandomState(global_random_seed)
    if distribution == "normal":
        X = rnd.normal(size=(5000, 10))
    else:
        X, _ = make_blobs(random_state=rnd)
    X[X < 0] = 0
    X = array_constr(X)

    km_lloyd = KMeans(n_clusters=5, random_state=global_random_seed, n_init=1, tol=tol)
    km_elkan = KMeans(
        algorithm="elkan",
        n_clusters=5,
        random_state=global_random_seed,
        n_init=1,
        tol=tol,
    )

    km_lloyd.fit(X)
    km_elkan.fit(X)
    assert_allclose(km_elkan.cluster_centers_, km_lloyd.cluster_centers_)
    assert_array_equal(km_elkan.labels_, km_lloyd.labels_)
    assert km_elkan.n_iter_ == km_lloyd.n_iter_
    assert km_elkan.inertia_ == pytest.approx(km_lloyd.inertia_, rel=1e-6)


@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
def test_kmeans_convergence(algorithm, global_random_seed):
    # 检查当tol=0时，KMeans是否在收敛时停止（#16075）
    # 使用给定的全局随机种子创建一个随机数生成器对象
    rnd = np.random.RandomState(global_random_seed)
    # 生成一个形状为 (5000, 10) 的正态分布随机数组
    X = rnd.normal(size=(5000, 10))
    # 设置最大迭代次数为 300
    max_iter = 300
    
    # 创建 KMeans 聚类对象并进行聚类操作
    km = KMeans(
        algorithm=algorithm,  # 指定聚类算法
        n_clusters=5,  # 指定聚类簇的数量
        random_state=global_random_seed,  # 设置随机种子
        n_init=1,  # 指定每个初始质心集的重新初始化次数
        tol=0,  # 指定收敛阈值，这里为 0 表示要求完全收敛
        max_iter=max_iter,  # 设置最大迭代次数
    ).fit(X)  # 对数据 X 进行聚类并训练模型
    
    # 断言条件：确保实际的迭代次数 km.n_iter_ 小于设定的最大迭代次数 max_iter
    assert km.n_iter_ < max_iter
@pytest.mark.parametrize("X_csr", X_as_any_csr)
def test_minibatch_update_consistency(X_csr, global_random_seed):
    # 检查稠密和稀疏小批量更新是否产生相同结果

    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)

    # 复制当前聚类中心并添加随机扰动
    centers_old = centers + rng.normal(size=centers.shape)
    centers_old_csr = centers_old.copy()

    # 初始化新的聚类中心
    centers_new = np.zeros_like(centers_old)
    centers_new_csr = np.zeros_like(centers_old_csr)

    # 初始化权重和
    weight_sums = np.zeros(centers_old.shape[0], dtype=X.dtype)
    weight_sums_csr = np.zeros(centers_old.shape[0], dtype=X.dtype)

    # 初始化样本权重
    sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # 提取一个小的样本批次
    X_mb = X[:10]
    X_mb_csr = X_csr[:10]
    sample_weight_mb = sample_weight[:10]

    # 步骤1：计算稠密小批量更新
    old_inertia = _mini_batch_step(
        X_mb,
        sample_weight_mb,
        centers_old,
        centers_new,
        weight_sums,
        np.random.RandomState(global_random_seed),
        random_reassign=False,
    )
    assert old_inertia > 0.0

    # 在同一批次上计算新的惯性以检查是否减少
    labels, new_inertia = _labels_inertia(X_mb, sample_weight_mb, centers_new)
    assert new_inertia > 0.0
    assert new_inertia < old_inertia

    # 步骤2：计算稀疏小批量更新
    old_inertia_csr = _mini_batch_step(
        X_mb_csr,
        sample_weight_mb,
        centers_old_csr,
        centers_new_csr,
        weight_sums_csr,
        np.random.RandomState(global_random_seed),
        random_reassign=False,
    )
    assert old_inertia_csr > 0.0

    # 在同一批次上计算新的惯性以检查是否减少
    labels_csr, new_inertia_csr = _labels_inertia(
        X_mb_csr, sample_weight_mb, centers_new_csr
    )
    assert new_inertia_csr > 0.0
    assert new_inertia_csr < old_inertia_csr

    # 步骤3：检查稀疏和稠密更新是否产生相同结果
    assert_array_equal(labels, labels_csr)
    assert_allclose(centers_new, centers_new_csr)
    assert_allclose(old_inertia, old_inertia_csr)
    assert_allclose(new_inertia, new_inertia_csr)


def _check_fitted_model(km):
    # 检查聚类模型是否适配好

    # 检查聚类中心的形状是否符合预期
    centers = km.cluster_centers_
    assert centers.shape == (n_clusters, n_features)

    # 检查聚类标签的唯一值数量是否等于聚类中心数量
    labels = km.labels_
    assert np.unique(labels).shape[0] == n_clusters

    # 检查标签分配是否完美（允许排列）
    assert_allclose(v_measure_score(true_labels, labels), 1.0)
    assert km.inertia_ > 0.0


@pytest.mark.parametrize(
    "input_data",
    [X] + X_as_any_csr,
    ids=data_containers_ids,
)
@pytest.mark.parametrize(
    "init",
    ["random", "k-means++", centers, lambda X, k, random_state: centers],
    ids=["random", "k-means++", "ndarray", "callable"],
)
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_all_init(Estimator, input_data, init):
    # 测试不同初始化方法的聚类模型的初始化行为
    # 检查使用所有可能的初始化方法对 KMeans 和 MiniBatchKMeans 进行评估。
    # 如果 init 是字符串类型，则设置 n_init 为 10，否则为 1。
    n_init = 10 if isinstance(init, str) else 1
    
    # 创建一个 Estimator 对象，使用指定的参数进行初始化，并对输入数据 input_data 进行拟合。
    # 这里使用了 random_state=42 来确保结果的可重复性。
    km = Estimator(
        init=init, n_clusters=n_clusters, random_state=42, n_init=n_init
    ).fit(input_data)
    
    # 检查拟合后的模型 km，确保模型已经合理拟合并可以进行后续操作。
    _check_fitted_model(km)
@pytest.mark.parametrize(
    "init",
    ["random", "k-means++", centers, lambda X, k, random_state: centers],
    ids=["random", "k-means++", "ndarray", "callable"],
)
# 定义参数化测试函数，用于测试不同初始化方式的 MiniBatchKMeans 模型
def test_minibatch_kmeans_partial_fit_init(init):
    # Check MiniBatchKMeans init with partial_fit
    # 根据初始化类型确定初始聚类中心的个数
    n_init = 10 if isinstance(init, str) else 1
    # 创建 MiniBatchKMeans 模型对象
    km = MiniBatchKMeans(
        init=init, n_clusters=n_clusters, random_state=0, n_init=n_init
    )
    for i in range(100):
        # 使用 partial_fit 方法进行模型拟合
        km.partial_fit(X)
    # 检查模型是否拟合完成
    _check_fitted_model(km)


@pytest.mark.parametrize(
    "init, expected_n_init",
    [
        ("k-means++", 1),
        ("random", "default"),
        (
            lambda X, n_clusters, random_state: random_state.uniform(
                size=(n_clusters, X.shape[1])
            ),
            "default",
        ),
        ("array-like", 1),
    ],
)
# 参数化测试函数，用于检查 KMeans 和 MiniBatchKMeans 的初始化方法及其预期的初始化次数
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_kmeans_init_auto_with_initial_centroids(Estimator, init, expected_n_init):
    """Check that `n_init="auto"` chooses the right number of initializations.
    Non-regression test for #26657:
    https://github.com/scikit-learn/scikit-learn/pull/26657
    """
    n_sample, n_features, n_clusters = 100, 10, 5
    X = np.random.randn(n_sample, n_features)
    if init == "array-like":
        init = np.random.randn(n_clusters, n_features)
    if expected_n_init == "default":
        expected_n_init = 3 if Estimator is MiniBatchKMeans else 10

    # 创建 Estimator 模型对象，使用 `n_init="auto"` 进行拟合
    kmeans = Estimator(n_clusters=n_clusters, init=init, n_init="auto").fit(X)
    # 断言确保选择的初始化次数与预期一致
    assert kmeans._n_init == expected_n_init


@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_fortran_aligned_data(Estimator, global_random_seed):
    # Check that KMeans works with fortran-aligned data.
    # 将数据转换为 Fortran 对齐格式
    X_fortran = np.asfortranarray(X)
    centers_fortran = np.asfortranarray(centers)

    # 使用 Fortran 对齐数据初始化 Estimator 模型对象进行拟合
    km_c = Estimator(
        n_clusters=n_clusters, init=centers, n_init=1, random_state=global_random_seed
    ).fit(X)
    km_f = Estimator(
        n_clusters=n_clusters,
        init=centers_fortran,
        n_init=1,
        random_state=global_random_seed,
    ).fit(X_fortran)
    # 断言确保两种数据格式下的聚类中心和标签一致
    assert_allclose(km_c.cluster_centers_, km_f.cluster_centers_)
    assert_array_equal(km_c.labels_, km_f.labels_)


def test_minibatch_kmeans_verbose():
    # Check verbose mode of MiniBatchKMeans for better coverage.
    # 检查 MiniBatchKMeans 的详细模式以增强覆盖率
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, verbose=1)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        km.fit(X)
    finally:
        sys.stdout = old_stdout


@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
@pytest.mark.parametrize("tol", [1e-2, 0])
def test_kmeans_verbose(algorithm, tol, capsys):
    # Check verbose mode of KMeans for better coverage.
    # 检查 KMeans 的详细模式以增强覆盖率
    X = np.random.RandomState(0).normal(size=(5000, 10))
    # 使用 KMeans 算法进行聚类
    KMeans(
        algorithm=algorithm,    # 指定聚类算法的类型（如"k-means++"）
        n_clusters=n_clusters,  # 指定聚类的簇数目
        random_state=42,        # 设置随机种子以保证结果的可重复性
        init="random",          # 使用随机初始化中心点的方法
        n_init=1,               # 指定运行 K 均值算法的次数，选择最优初始化的结果
        tol=tol,                # 设置收敛阈值
        verbose=1,              # 设置详细程度为1，输出每次迭代的信息
    ).fit(X)                    # 对数据 X 进行聚类，并拟合 KMeans 模型

    # 读取并捕获在测试过程中的输出信息
    captured = capsys.readouterr()

    # 断言在捕获的输出中是否包含初始化完成的信息
    assert re.search(r"Initialization complete", captured.out)
    # 断言在捕获的输出中是否包含迭代次数和惯性的信息
    assert re.search(r"Iteration [0-9]+, inertia", captured.out)

    # 如果收敛阈值为0，则断言捕获的输出中包含严格收敛的信息
    if tol == 0:
        assert re.search(r"strict convergence", captured.out)
    # 否则，断言捕获的输出中包含中心点移动在容忍范围内的信息
    else:
        assert re.search(r"center shift .* within tolerance", captured.out)
# 测试函数：测试当 init_size 小于 n_clusters 时是否会触发警告
def test_minibatch_kmeans_warning_init_size():
    # 使用 pytest 来检查是否会引发 RuntimeWarning，匹配警告信息中包含 "init_size.* should be larger than n_clusters"
    with pytest.warns(
        RuntimeWarning, match=r"init_size.* should be larger than n_clusters"
    ):
        # 创建 MiniBatchKMeans 对象，设定 init_size=10, n_clusters=20，并对输入数据 X 进行拟合
        MiniBatchKMeans(init_size=10, n_clusters=20).fit(X)


# 参数化测试函数：测试当 n_init > 1 且 init 参数为数组时是否会触发警告
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_warning_n_init_precomputed_centers(Estimator):
    # 使用 pytest 来检查是否会引发 RuntimeWarning，匹配警告信息为 "Explicit initial center position passed: performing only one init"
    with pytest.warns(
        RuntimeWarning,
        match="Explicit initial center position passed: performing only one init",
    ):
        # 创建指定类型 Estimator 对象，设定 init=centers, n_clusters=n_clusters, n_init=10，并对输入数据 X 进行拟合
        Estimator(init=centers, n_clusters=n_clusters, n_init=10).fit(X)


# 测试函数：检查 MiniBatchKMeans 在重新分配聚类中心时的行为
def test_minibatch_sensible_reassign(global_random_seed):
    # 生成具有 100 个样本、5 个中心的数据集，并指定部分样本的特征值为零
    zeroed_X, true_labels = make_blobs(
        n_samples=100, centers=5, random_state=global_random_seed
    )
    zeroed_X[::2, :] = 0

    # 创建 MiniBatchKMeans 对象，设定 n_clusters=20, batch_size=10，并使用 random 初始化，对 zeroed_X 进行拟合
    km = MiniBatchKMeans(
        n_clusters=20, batch_size=10, random_state=global_random_seed, init="random"
    ).fit(zeroed_X)
    # 计算非零聚类中心的数量
    num_non_zero_clusters = km.cluster_centers_.any(axis=1).sum()
    # 断言非零聚类中心数量大于 9
    assert num_non_zero_clusters > 9, f"{num_non_zero_clusters=} is too small"

    # 使用 batch_size > X.shape[0] 的设置再次创建 MiniBatchKMeans 对象，并对 zeroed_X 进行拟合
    km = MiniBatchKMeans(
        n_clusters=20, batch_size=200, random_state=global_random_seed, init="random"
    ).fit(zeroed_X)
    # 计算非零聚类中心的数量
    num_non_zero_clusters = km.cluster_centers_.any(axis=1).sum()
    # 断言非零聚类中心数量大于 9
    assert num_non_zero_clusters > 9, f"{num_non_zero_clusters=} is too small"

    # 使用 partial_fit API 进行拟合
    km = MiniBatchKMeans(n_clusters=20, random_state=global_random_seed, init="random")
    for i in range(100):
        km.partial_fit(zeroed_X)
    # 计算非零聚类中心的数量
    num_non_zero_clusters = km.cluster_centers_.any(axis=1).sum()
    # 断言非零聚类中心数量大于 9
    assert num_non_zero_clusters > 9, f"{num_non_zero_clusters=} is too small"


# 参数化测试函数：测试 MiniBatchKMeans 在重新分配聚类中心时的行为，使用不同的输入数据
@pytest.mark.parametrize(
    "input_data",
    [X] + X_as_any_csr,
    ids=data_containers_ids,
)
def test_minibatch_reassign(input_data, global_random_seed):
    # 根据真实标签计算完美聚类中心
    perfect_centers = np.empty((n_clusters, n_features))
    for i in range(n_clusters):
        perfect_centers[i] = X[true_labels == i].mean(axis=0)

    sample_weight = np.ones(n_samples)
    centers_new = np.empty_like(perfect_centers)

    # 给定完美初始化聚类中心，但设置较大的重新分配比率，预计许多中心将被重新分配，模型不再优秀
    score_before = -_labels_inertia(input_data, sample_weight, perfect_centers, 1)[1]
    # 调用一个函数来执行迭代的Mini-batch K-means聚类步骤，使用给定的参数
    _mini_batch_step(
        input_data,  # 输入数据集
        sample_weight,  # 样本权重
        perfect_centers,  # 完美的聚类中心
        centers_new,  # 更新后的聚类中心
        np.zeros(n_clusters),  # 初始聚类中心的累积权重
        np.random.RandomState(global_random_seed),  # 使用全局随机种子创建的随机数生成器
        random_reassign=True,  # 是否允许随机重新分配
        reassignment_ratio=1,  # 重新分配比率，此时不会重新分配任何中心
    )
    
    # 计算执行Mini-batch K-means聚类步骤后的得分，评估聚类质量
    score_after = -_labels_inertia(input_data, sample_weight, centers_new, 1)[1]
    
    # 断言：在完美初始化的情况下，通过设置极小的重新分配比率，
    # 不应该有任何中心被重新分配。
    _mini_batch_step(
        input_data,  # 输入数据集
        sample_weight,  # 样本权重
        perfect_centers,  # 完美的聚类中心
        centers_new,  # 更新后的聚类中心
        np.zeros(n_clusters),  # 初始聚类中心的累积权重
        np.random.RandomState(global_random_seed),  # 使用全局随机种子创建的随机数生成器
        random_reassign=True,  # 是否允许随机重新分配
        reassignment_ratio=1e-15,  # 极小的重新分配比率
    )
    
    # 断言：验证更新后的聚类中心与完美的聚类中心非常接近，即它们几乎相等。
    assert_allclose(centers_new, perfect_centers)
def test_minibatch_with_many_reassignments():
    """
    # Test for the case that the number of clusters to reassign is bigger
    # than the batch_size. Run the test with 100 clusters and a batch_size of
    # 10 because it turned out that these values ensure that the number of
    # clusters to reassign is always bigger than the batch_size.
    """
    MiniBatchKMeans(
        n_clusters=100,
        batch_size=10,
        init_size=n_samples,
        random_state=42,
        verbose=True,
    ).fit(X)


def test_minibatch_kmeans_init_size():
    """
    # Check the internal _init_size attribute of MiniBatchKMeans

    # default init size should be 3 * batch_size
    """
    km = MiniBatchKMeans(n_clusters=10, batch_size=5, n_init=1).fit(X)
    assert km._init_size == 15

    # if 3 * batch size < n_clusters, it should then be 3 * n_clusters
    km = MiniBatchKMeans(n_clusters=10, batch_size=1, n_init=1).fit(X)
    assert km._init_size == 30

    # it should not be larger than n_samples
    km = MiniBatchKMeans(
        n_clusters=10, batch_size=5, n_init=1, init_size=n_samples + 1
    ).fit(X)
    assert km._init_size == n_samples


@pytest.mark.parametrize("tol, max_no_improvement", [(1e-4, None), (0, 10)])
def test_minibatch_declared_convergence(capsys, tol, max_no_improvement):
    """
    # Check convergence detection based on ewa batch inertia or on
    # small center change.
    """
    X, _, centers = make_blobs(centers=3, random_state=0, return_centers=True)

    km = MiniBatchKMeans(
        n_clusters=3,
        init=centers,
        batch_size=20,
        tol=tol,
        random_state=0,
        max_iter=10,
        n_init=1,
        verbose=1,
        max_no_improvement=max_no_improvement,
    )

    km.fit(X)
    assert 1 < km.n_iter_ < 10

    captured = capsys.readouterr()
    if max_no_improvement is None:
        assert "Converged (small centers change)" in captured.out
    if tol == 0:
        assert "Converged (lack of improvement in inertia)" in captured.out


def test_minibatch_iter_steps():
    """
    # Check consistency of n_iter_ and n_steps_ attributes.
    """
    batch_size = 30
    n_samples = X.shape[0]
    km = MiniBatchKMeans(n_clusters=3, batch_size=batch_size, random_state=0).fit(X)

    # n_iter_ is the number of started epochs
    assert km.n_iter_ == np.ceil((km.n_steps_ * batch_size) / n_samples)
    assert isinstance(km.n_iter_, int)

    # without stopping condition, max_iter should be reached
    km = MiniBatchKMeans(
        n_clusters=3,
        batch_size=batch_size,
        random_state=0,
        tol=0,
        max_no_improvement=None,
        max_iter=10,
    ).fit(X)

    assert km.n_iter_ == 10
    assert km.n_steps_ == (10 * n_samples) // batch_size
    assert isinstance(km.n_steps_, int)


def test_kmeans_copyx():
    """
    # Check that copy_x=False returns nearly equal X after de-centering.
    """
    my_X = X.copy()
    km = KMeans(copy_x=False, n_clusters=n_clusters, random_state=42)
    km.fit(my_X)
    _check_fitted_model(km)
    # 确保 my_X 是去中心化的
    assert_allclose(my_X, X)
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_score_max_iter(Estimator, global_random_seed):
    # 检查使用更多迭代次数来拟合 KMeans 或 MiniBatchKMeans 是否会得到更好的评分

    # 生成随机数据集 X
    X = np.random.RandomState(global_random_seed).randn(100, 10)

    # 创建 Estimator 对象，设置最大迭代次数为 1 的 KMeans 模型
    km1 = Estimator(n_init=1, random_state=global_random_seed, max_iter=1)
    # 使用 X 拟合模型并计算评分 s1
    s1 = km1.fit(X).score(X)

    # 创建 Estimator 对象，设置最大迭代次数为 10 的 KMeans 模型
    km2 = Estimator(n_init=1, random_state=global_random_seed, max_iter=10)
    # 使用 X 拟合模型并计算评分 s2
    s2 = km2.fit(X).score(X)

    # 断言使用更多迭代次数得到的评分 s2 比 s1 更高
    assert s2 > s1


@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
@pytest.mark.parametrize(
    "Estimator, algorithm",
    [(KMeans, "lloyd"), (KMeans, "elkan"), (MiniBatchKMeans, None)],
)
@pytest.mark.parametrize("max_iter", [2, 100])
def test_kmeans_predict(
    Estimator, algorithm, array_constr, max_iter, global_dtype, global_random_seed
):
    # 检查 predict 方法以及 fit.predict 和 fit_predict 之间的等价性

    # 生成随机数据集 X
    X, _ = make_blobs(
        n_samples=200, n_features=10, centers=10, random_state=global_random_seed
    )
    X = array_constr(X, dtype=global_dtype)

    # 创建 Estimator 对象
    km = Estimator(
        n_clusters=10,
        init="random",
        n_init=10,
        max_iter=max_iter,
        random_state=global_random_seed,
    )
    # 根据算法设置参数
    if algorithm is not None:
        km.set_params(algorithm=algorithm)
    # 使用 X 拟合模型
    km.fit(X)
    # 获取模型预测的标签
    labels = km.labels_

    # 使用 predict 方法重新预测训练集的标签
    pred = km.predict(X)
    assert_array_equal(pred, labels)

    # 使用 fit_predict 方法重新预测训练集的标签
    pred = km.fit_predict(X)
    assert_array_equal(pred, labels)

    # 预测质心标签
    pred = km.predict(km.cluster_centers_)
    assert_array_equal(pred, np.arange(10))


@pytest.mark.parametrize("X_csr", X_as_any_csr)
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_dense_sparse(Estimator, X_csr, global_random_seed):
    # 检查稠密和稀疏输入的结果是否相同

    # 生成随机样本权重
    sample_weight = np.random.RandomState(global_random_seed).random_sample(
        (n_samples,)
    )

    # 创建稠密输入的 Estimator 对象
    km_dense = Estimator(
        n_clusters=n_clusters, random_state=global_random_seed, n_init=1
    )
    # 使用稠密输入 X 拟合模型
    km_dense.fit(X, sample_weight=sample_weight)

    # 创建稀疏输入的 Estimator 对象
    km_sparse = Estimator(
        n_clusters=n_clusters, random_state=global_random_seed, n_init=1
    )
    # 使用稀疏输入 X_csr 拟合模型
    km_sparse.fit(X_csr, sample_weight=sample_weight)

    # 断言稠密和稀疏输入的标签结果相等
    assert_array_equal(km_dense.labels_, km_sparse.labels_)
    # 断言稠密和稀疏输入的聚类中心结果相近
    assert_allclose(km_dense.cluster_centers_, km_sparse.cluster_centers_)


@pytest.mark.parametrize("X_csr", X_as_any_csr)
@pytest.mark.parametrize(
    "init", ["random", "k-means++", centers], ids=["random", "k-means++", "ndarray"]
)
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_predict_dense_sparse(Estimator, init, X_csr):
    # 检查在稠密和稀疏输入情况下，模型在预测时的表现是否一致
    # 根据初始化类型（字符串或其他）确定初始值的数量
    n_init = 10 if isinstance(init, str) else 1
    # 创建一个聚类估计器对象，指定簇的数量、初始化方法、初始值数量和随机状态
    km = Estimator(n_clusters=n_clusters, init=init, n_init=n_init, random_state=0)
    
    # 使用稀疏矩阵 X_csr 进行聚类模型的训练
    km.fit(X_csr)
    # 断言预测的结果与标签相等，用于验证模型的准确性
    assert_array_equal(km.predict(X), km.labels_)
    
    # 使用完整数据集 X 进行聚类模型的训练
    km.fit(X)
    # 断言使用稀疏矩阵 X_csr 进行预测得到的结果与模型标签相等，用于验证模型的准确性
    assert_array_equal(km.predict(X_csr), km.labels_)
@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
# 使用 pytest.mark.parametrize 装饰器，为 array_constr 参数提供多组输入数据，使用 data_containers 提供的数据和对应的标识符 data_containers_ids
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
# 使用 pytest.mark.parametrize 装饰器，为 dtype 参数提供两种数据类型选项：np.int32 和 np.int64
@pytest.mark.parametrize("init", ["k-means++", "ndarray"])
# 使用 pytest.mark.parametrize 装饰器，为 init 参数提供两种初始化方式选项："k-means++" 和 "ndarray"
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
# 使用 pytest.mark.parametrize 装饰器，为 Estimator 参数提供两种聚类器选项：KMeans 和 MiniBatchKMeans
def test_integer_input(Estimator, array_constr, dtype, init, global_random_seed):
    # 检查 KMeans 和 MiniBatchKMeans 是否能处理整数输入
    X_dense = np.array([[0, 0], [10, 10], [12, 9], [-1, 1], [2, 0], [8, 10]])
    # 创建稠密的整数数组 X_dense

    X = array_constr(X_dense, dtype=dtype)
    # 使用 array_constr 将 X_dense 转换为指定的数据类型 dtype，并赋值给 X

    n_init = 1 if init == "ndarray" else 10
    # 如果 init 是 "ndarray"，则 n_init 设为 1，否则设为 10

    init = X_dense[:2] if init == "ndarray" else init
    # 如果 init 是 "ndarray"，则将 X_dense 的前两行作为初始化值，否则保持不变

    km = Estimator(
        n_clusters=2, init=init, n_init=n_init, random_state=global_random_seed
    )
    # 根据给定的 Estimator 类创建聚类器 km，设置聚类数为 2，初始化方式为 init，n_init 设为 n_init，随机种子为 global_random_seed

    if Estimator is MiniBatchKMeans:
        km.set_params(batch_size=2)
        # 如果 Estimator 是 MiniBatchKMeans，则设置 batch_size 参数为 2

    km.fit(X)
    # 对数据 X 进行聚类

    assert km.cluster_centers_.dtype == np.float64
    # 断言聚类中心的数据类型为 np.float64

    expected_labels = [0, 1, 1, 0, 0, 1]
    # 预期的标签

    assert_allclose(v_measure_score(km.labels_, expected_labels), 1.0)
    # 使用 v_measure_score 检查 km.labels_ 和 expected_labels 的相似度是否接近 1.0

    if Estimator is MiniBatchKMeans:
        km = clone(km).partial_fit(X)
        # 如果 Estimator 是 MiniBatchKMeans，则使用 clone(km).partial_fit(X) 进行部分拟合

        assert km.cluster_centers_.dtype == np.float64
        # 再次断言聚类中心的数据类型为 np.float64


@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
# 使用 pytest.mark.parametrize 装饰器，为 Estimator 参数提供两种聚类器选项：KMeans 和 MiniBatchKMeans
def test_transform(Estimator, global_random_seed):
    # 检查 transform 方法
    km = Estimator(n_clusters=n_clusters, random_state=global_random_seed).fit(X)
    # 创建并拟合聚类器 km，设置聚类数为 n_clusters，随机种子为 global_random_seed

    # 转换 cluster_centers_ 应该返回中心点之间的成对距离
    Xt = km.transform(km.cluster_centers_)
    # 使用 transform 方法将聚类器的 cluster_centers_ 转换成成对距离，赋值给 Xt

    assert_allclose(Xt, pairwise_distances(km.cluster_centers_))
    # 使用 assert_allclose 检查 Xt 和 km.cluster_centers_ 之间的成对距离是否接近

    # 特别地，对角线应该为 0
    assert_array_equal(Xt.diagonal(), np.zeros(n_clusters))
    # 使用 assert_array_equal 检查 Xt 的对角线是否全为 0

    # 转换 X 应该返回数据 X 和中心点之间的成对距离
    Xt = km.transform(X)
    # 使用 transform 方法将数据 X 转换成成对距离，赋值给 Xt

    assert_allclose(Xt, pairwise_distances(X, km.cluster_centers_))
    # 使用 assert_allclose 检查 Xt 和数据 X 与中心点之间的成对距离是否接近


@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
# 使用 pytest.mark.parametrize 装饰器，为 Estimator 参数提供两种聚类器选项：KMeans 和 MiniBatchKMeans
def test_fit_transform(Estimator, global_random_seed):
    # 检查 fit.transform 和 fit_transform 之间的等价性
    X1 = Estimator(random_state=global_random_seed, n_init=1).fit(X).transform(X)
    # 使用 fit 方法拟合数据 X，然后使用 transform 方法转换，结果赋值给 X1

    X2 = Estimator(random_state=global_random_seed, n_init=1).fit_transform(X)
    # 使用 fit_transform 方法对数据 X 进行拟合和转换，结果赋值给 X2

    assert_allclose(X1, X2)
    # 使用 assert_allclose 检查 X1 和 X2 是否接近


def test_n_init(global_random_seed):
    # 检查增加初始化次数是否提高质量
    previous_inertia = np.inf
    # 设置前一个惯性为无穷大

    for n_init in [1, 5, 10]:
        # 遍历不同的初始化次数：1、5、10
        # 设置 max_iter=1 避免找到全局最小值，并每次获得相同的惯性
        km = KMeans(
            n_clusters=n_clusters,
            init="random",
            n_init=n_init,
            random_state=global_random_seed,
            max_iter=1,
        ).fit(X)
        # 创建 KMeans 聚类器 km，设置聚类数为 n_clusters，初始化方式为 "random"，初始化次数为 n_init，随机种子为 global_random_seed，最大迭代次数为 1，拟合数据 X

        assert km.inertia_ <= previous_inertia
        # 使用 assert 检查当前的惯性是否小于等于前一个惯性
        previous_inertia = km.inertia_
        # 更新前一个惯性为当前的惯性值


def test_k_means_function(global_random_seed):
    # 测试直接调用 k_means 函数
    # 使用 K-Means 算法对数据 X 进行聚类，得到聚类中心、类别标签和总内聚度
    cluster_centers, labels, inertia = k_means(
        X, n_clusters=n_clusters, sample_weight=None, random_state=global_random_seed
    )

    # 断言聚类中心的形状应为 (聚类数, 特征数)，确保聚类中心计算正确
    assert cluster_centers.shape == (n_clusters, n_features)
    
    # 断言标签的唯一值数量应为聚类数，确保每个聚类都有对应的标签
    assert np.unique(labels).shape[0] == n_clusters

    # 检查标签的分配是否完美（允许排列的变化），应接近 1.0 表示非常相似
    assert_allclose(v_measure_score(true_labels, labels), 1.0)
    
    # 断言总内聚度大于 0.0，确保聚类效果合理
    assert inertia > 0.0
# 使用 pytest 的 mark.parametrize 装饰器，为 test_float_precision 函数参数化测试数据。
# input_data 参数包含 X 和 X_as_any_csr 中的数据，使用 data_containers_ids 进行标识。
@pytest.mark.parametrize(
    "input_data",
    [X] + X_as_any_csr,
    ids=data_containers_ids,
)
# 使用 pytest 的 mark.parametrize 装饰器，为 Estimator 参数参数化测试数据，包括 KMeans 和 MiniBatchKMeans。
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_float_precision(Estimator, input_data, global_random_seed):
    # 创建 Estimator 对象 km，设置 n_init 和 random_state 参数
    km = Estimator(n_init=1, random_state=global_random_seed)

    # 初始化空字典和变量用于存储结果
    inertia = {}
    Xt = {}
    centers = {}
    labels = {}

    # 循环测试不同的数据类型 np.float64 和 np.float32
    for dtype in [np.float64, np.float32]:
        # 将 input_data 转换为当前循环的数据类型 X
        X = input_data.astype(dtype, copy=False)
        # 对当前数据类型的 X 运行 KMeans 算法
        km.fit(X)

        # 将结果存储在对应的字典中
        inertia[dtype] = km.inertia_
        Xt[dtype] = km.transform(X)
        centers[dtype] = km.cluster_centers_
        labels[dtype] = km.labels_

        # 断言聚类中心的数据类型与输入数据类型一致
        assert km.cluster_centers_.dtype == dtype

        # 对于 MiniBatchKMeans，同样使用 partial_fit 方法进行测试
        if Estimator is MiniBatchKMeans:
            km.partial_fit(X[0:3])
            assert km.cluster_centers_.dtype == dtype

    # 比较低精度下的数组，由于 32 位和 64 位之间的差异来自舍入误差的累积
    assert_allclose(inertia[np.float32], inertia[np.float64], rtol=1e-4)
    assert_allclose(Xt[np.float32], Xt[np.float64], atol=Xt[np.float64].max() * 1e-4)
    assert_allclose(
        centers[np.float32], centers[np.float64], atol=centers[np.float64].max() * 1e-4
    )
    assert_array_equal(labels[np.float32], labels[np.float64])


# 使用 pytest 的 mark.parametrize 装饰器，为 dtype 和 Estimator 参数化测试数据。
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_centers_not_mutated(Estimator, dtype):
    # 检查 KMeans 和 MiniBatchKMeans 是否在静默情况下修改了用户提供的初始化中心点
    X_new_type = X.astype(dtype, copy=False)
    centers_new_type = centers.astype(dtype, copy=False)

    # 创建 Estimator 对象 km，使用新类型的数据和中心点进行初始化
    km = Estimator(init=centers_new_type, n_clusters=n_clusters, n_init=1)
    km.fit(X_new_type)

    # 断言 km.cluster_centers_ 和 centers_new_type 不共享内存
    assert not np.may_share_memory(km.cluster_centers_, centers_new_type)


# 使用 pytest 的 mark.parametrize 装饰器，为 input_data 参数参数化测试数据。
@pytest.mark.parametrize(
    "input_data",
    [X] + X_as_any_csr,
    ids=data_containers_ids,
)
def test_kmeans_init_fitted_centers(input_data):
    # 检查从局部最优解开始拟合不应该改变解决方案
    km1 = KMeans(n_clusters=n_clusters).fit(input_data)
    km2 = KMeans(n_clusters=n_clusters, init=km1.cluster_centers_, n_init=1).fit(
        input_data
    )

    # 断言 km1 和 km2 的聚类中心相似
    assert_allclose(km1.cluster_centers_, km2.cluster_centers_)


def test_kmeans_warns_less_centers_than_unique_points(global_random_seed):
    # 检查当找到的聚类数少于预期时的 KMeans 行为
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 0]])  # 最后一个点是重复的
    km = KMeans(n_clusters=4, random_state=global_random_seed)

    # KMeans 应该警告使用的标签比聚类中心少
    msg = (
        r"Number of distinct clusters \(3\) found smaller than "
        r"n_clusters \(4\). Possibly due to duplicate points in X."
    )
    # 定义警告消息，用于检测收敛警告，匹配特定的消息格式
    with pytest.warns(ConvergenceWarning, match=msg):
        # 使用 KMeans 对象 km 对数据 X 进行拟合
        km.fit(X)
        # 断言：聚类标签应该是 {0, 1, 2} 这三个值，因为只有三个独特的聚类点
        assert set(km.labels_) == set(range(3))
# 对输入的中心点数组按照每列（即每个维度）进行排序
def _sort_centers(centers):
    return np.sort(centers, axis=0)


# 测试带权重和重复样本的 KMeans 的等效性
def test_weighted_vs_repeated(global_random_seed):
    # 使用全局随机种子创建一个随机状态对象，生成一个长度为 n_samples 的随机整数数组作为样本权重
    sample_weight = np.random.RandomState(global_random_seed).randint(
        1, 5, size=n_samples
    )
    # 对输入数据 X 进行重复采样，按照样本权重进行重复，生成扩展后的数据集 X_repeat
    X_repeat = np.repeat(X, sample_weight, axis=0)

    # 使用 KMeans 进行聚类，指定初始中心点、聚类数目等参数
    km = KMeans(
        init=centers, n_init=1, n_clusters=n_clusters, random_state=global_random_seed
    )

    # 克隆 KMeans 对象并使用带有样本权重的数据进行训练
    km_weighted = clone(km).fit(X, sample_weight=sample_weight)
    # 将带有权重的标签重复 n_samples 次，以匹配 X_repeat 的标签
    repeated_labels = np.repeat(km_weighted.labels_, sample_weight)
    # 使用重复数据集 X_repeat 训练另一个 KMeans 对象
    km_repeated = clone(km).fit(X_repeat)

    # 断言两种方法的聚类结果相同
    assert_array_equal(km_repeated.labels_, repeated_labels)
    # 断言两种方法的惯性（inertia）近似相等
    assert_allclose(km_weighted.inertia_, km_repeated.inertia_)
    # 断言两种方法的聚类中心近似相等，经过排序后比较
    assert_allclose(
        _sort_centers(km_weighted.cluster_centers_),
        _sort_centers(km_repeated.cluster_centers_),
    )


# 使用参数化测试来比较不带权重和带权重的 KMeans 方法
@pytest.mark.parametrize(
    "input_data",
    [X] + X_as_any_csr,
    ids=data_containers_ids,
)
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_unit_weights_vs_no_weights(Estimator, input_data, global_random_seed):
    # 检查不传递样本权重和传递全部样本权重为1的等效性
    sample_weight = np.ones(n_samples)

    # 使用不同的估算器（Estimator）初始化 KMeans 对象，传递全局随机种子和初始聚类数目
    km = Estimator(n_clusters=n_clusters, random_state=global_random_seed, n_init=1)
    # 分别使用不带权重和带有全1权重的数据进行训练
    km_none = clone(km).fit(input_data, sample_weight=None)
    km_ones = clone(km).fit(input_data, sample_weight=sample_weight)

    # 断言两种方法的聚类结果相同
    assert_array_equal(km_none.labels_, km_ones.labels_)
    # 断言两种方法的聚类中心近似相等
    assert_allclose(km_none.cluster_centers_, km_ones.cluster_centers_)


# 使用参数化测试来比较缩放后的样本权重对结果的影响
@pytest.mark.parametrize(
    "input_data",
    [X] + X_as_any_csr,
    ids=data_containers_ids,
)
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_scaled_weights(Estimator, input_data, global_random_seed):
    # 检查通过缩放样本权重不影响最终聚类结果
    sample_weight = np.random.RandomState(global_random_seed).uniform(size=n_samples)

    # 使用不同的估算器（Estimator）初始化 KMeans 对象，传递全局随机种子和初始聚类数目
    km = Estimator(n_clusters=n_clusters, random_state=global_random_seed, n_init=1)
    # 使用原始样本权重进行训练
    km_orig = clone(km).fit(input_data, sample_weight=sample_weight)
    # 使用缩放后的样本权重（0.5倍）进行训练
    km_scaled = clone(km).fit(input_data, sample_weight=0.5 * sample_weight)

    # 断言两种方法的聚类结果相同
    assert_array_equal(km_orig.labels_, km_scaled.labels_)
    # 断言两种方法的聚类中心近似相等
    assert_allclose(km_orig.cluster_centers_, km_scaled.cluster_centers_)


# 测试 KMeans 使用 Elkan 算法时的迭代次数属性
def test_kmeans_elkan_iter_attribute():
    # 回归测试，检查修复之前的 bug，确保 n_iter_ 的值正确
    km = KMeans(algorithm="elkan", max_iter=1).fit(X)
    assert km.n_iter_ == 1


# 使用参数化测试来检查 KMeans 空簇重新定位的行为
@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
def test_kmeans_empty_cluster_relocated(array_constr):
    # 检查在使用样本权重时，空的聚类中心是否被正确重新定位（见 GitHub 问题 #13486）
    
    # 创建一个包含两个样本的二维数组 X，每个样本是一个一维数组
    X = array_constr([[-1], [1]])
    
    # 设置样本的权重
    sample_weight = [1.9, 0.1]
    
    # 设置初始的聚类中心
    init = np.array([[-1], [10]])
    
    # 使用 KMeans 算法，设置聚类数为 2，初始聚类中心为 init，仅运行一次初始化
    km = KMeans(n_clusters=2, init=init, n_init=1)
    
    # 对数据 X 进行聚类，同时传入样本权重 sample_weight
    km.fit(X, sample_weight=sample_weight)
    
    # 断言聚类标签的唯一值数量为 2
    assert len(set(km.labels_)) == 2
    
    # 断言聚类中心的值接近于[[-1], [1]]
    assert_allclose(km.cluster_centers_, [[-1], [1]])
# 使用 pytest.mark.parametrize 装饰器，定义了一个参数化测试函数，测试不同的 KMeans/MiniBatchKMeans 类型
@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_result_equal_in_diff_n_threads(Estimator, global_random_seed):
    # 检查在并行模式下 KMeans/MiniBatchKMeans 的结果与顺序模式下是否一致
    rnd = np.random.RandomState(global_random_seed)
    X = rnd.normal(size=(50, 10))

    # 在限制并行度为1的条件下，执行 Estimator.fit(X).labels_，并记录结果为 result_1
    with _get_threadpool_controller().limit(limits=1, user_api="openmp"):
        result_1 = (
            Estimator(n_clusters=n_clusters, random_state=global_random_seed)
            .fit(X)
            .labels_
        )
    
    # 在限制并行度为2的条件下，执行 Estimator.fit(X).labels_，并记录结果为 result_2
    with _get_threadpool_controller().limit(limits=2, user_api="openmp"):
        result_2 = (
            Estimator(n_clusters=n_clusters, random_state=global_random_seed)
            .fit(X)
            .labels_
        )
    
    # 断言两次执行结果是否相等
    assert_array_equal(result_1, result_2)


def test_warning_elkan_1_cluster():
    # 检查与 KMeans 特定的警告信息相关联
    with pytest.warns(
        RuntimeWarning,
        match="algorithm='elkan' doesn't make sense for a single cluster",
    ):
        # 当 n_clusters=1 且 algorithm="elkan" 时，触发 RuntimeWarning 警告
        KMeans(n_clusters=1, algorithm="elkan").fit(X)


# 使用 pytest.mark.parametrize 装饰器，定义了一个参数化测试函数，测试 KMeans 的单次迭代结果
@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
@pytest.mark.parametrize("algo", ["lloyd", "elkan"])
def test_k_means_1_iteration(array_constr, algo, global_random_seed):
    # 检查经过单次迭代（E步骤 M步骤 E步骤）后的结果，与纯Python实现进行比较
    X = np.random.RandomState(global_random_seed).uniform(size=(100, 5))
    init_centers = X[:5]
    X = array_constr(X)

    # 定义纯Python版本的 KMeans 算法实现
    def py_kmeans(X, init):
        new_centers = init.copy()
        labels = pairwise_distances_argmin(X, init)
        for label in range(init.shape[0]):
            new_centers[label] = X[labels == label].mean(axis=0)
        labels = pairwise_distances_argmin(X, new_centers)
        return labels, new_centers

    # 使用纯Python实现计算的标签和中心点
    py_labels, py_centers = py_kmeans(X, init_centers)

    # 使用 Cython 加速的 KMeans 进行单次迭代计算
    cy_kmeans = KMeans(
        n_clusters=5, n_init=1, init=init_centers, algorithm=algo, max_iter=1
    ).fit(X)
    cy_labels = cy_kmeans.labels_
    cy_centers = cy_kmeans.cluster_centers_

    # 断言 Cython 实现与纯Python实现的结果是否一致
    assert_array_equal(py_labels, cy_labels)
    assert_allclose(py_centers, cy_centers)


# 使用 pytest.mark.parametrize 装饰器，定义了一个参数化测试函数，测试欧氏距离计算的准确性
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("squared", [True, False])
def test_euclidean_distance(dtype, squared, global_random_seed):
    # 检查 _euclidean_dense_dense_wrapper 函数在不同参数条件下的输出是否正确
    rng = np.random.RandomState(global_random_seed)
    a_sparse = sp.random(
        1, 100, density=0.5, format="csr", random_state=rng, dtype=dtype
    )
    a_dense = a_sparse.toarray().reshape(-1)
    b = rng.randn(100).astype(dtype, copy=False)
    b_squared_norm = (b**2).sum()

    # 计算期望的欧氏距离的平方或平方根
    expected = ((a_dense - b) ** 2).sum()
    expected = expected if squared else np.sqrt(expected)

    # 使用 _euclidean_dense_dense_wrapper 计算实际的欧氏距离的平方或平方根
    distance_dense_dense = _euclidean_dense_dense_wrapper(a_dense, b, squared)
    # 计算稀疏向量和密集向量之间的欧氏距离，通过调用特定的包装函数
    distance_sparse_dense = _euclidean_sparse_dense_wrapper(
        a_sparse.data, a_sparse.indices, b, b_squared_norm, squared
    )
    
    # 根据数据类型确定相对误差容限
    rtol = 1e-4 if dtype == np.float32 else 1e-7
    # 断言稠密向量之间的距离与稀疏-密集向量之间的距离在容限范围内相等
    assert_allclose(distance_dense_dense, distance_sparse_dense, rtol=rtol)
    # 断言稠密向量之间的距离与预期值之间的距离在容限范围内相等
    assert_allclose(distance_dense_dense, expected, rtol=rtol)
    # 断言稀疏-密集向量之间的距离与预期值之间的距离在容限范围内相等
    assert_allclose(distance_sparse_dense, expected, rtol=rtol)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inertia(dtype, global_random_seed):
    # 检查 _inertia_(dense/sparse) 辅助函数是否产生正确的结果。
    rng = np.random.RandomState(global_random_seed)
    # 创建稀疏矩阵 X_sparse 和对应的稠密矩阵 X_dense
    X_sparse = sp.random(
        100, 10, density=0.5, format="csr", random_state=rng, dtype=dtype
    )
    X_dense = X_sparse.toarray()
    # 创建样本权重数组 sample_weight 和聚类中心数组 centers
    sample_weight = rng.randn(100).astype(dtype, copy=False)
    centers = rng.randn(5, 10).astype(dtype, copy=False)
    # 创建随机标签数组 labels
    labels = rng.randint(5, size=100, dtype=np.int32)

    # 计算每个样本到其所属聚类中心的距离平方和
    distances = ((X_dense - centers[labels]) ** 2).sum(axis=1)
    expected = np.sum(distances * sample_weight)

    # 调用 _inertia_dense 和 _inertia_sparse 计算惯性
    inertia_dense = _inertia_dense(X_dense, sample_weight, centers, labels, n_threads=1)
    inertia_sparse = _inertia_sparse(
        X_sparse, sample_weight, centers, labels, n_threads=1
    )

    rtol = 1e-4 if dtype == np.float32 else 1e-6
    # 检查 dense 和 sparse 计算的惯性值是否接近
    assert_allclose(inertia_dense, inertia_sparse, rtol=rtol)
    # 检查 dense 计算的惯性值与预期值是否接近
    assert_allclose(inertia_dense, expected, rtol=rtol)
    # 检查 sparse 计算的惯性值与预期值是否接近
    assert_allclose(inertia_sparse, expected, rtol=rtol)

    # 检查单个标签的参数
    label = 1
    mask = labels == label
    # 计算特定标签的样本到其所属聚类中心的距离平方和
    distances = ((X_dense[mask] - centers[label]) ** 2).sum(axis=1)
    expected = np.sum(distances * sample_weight[mask])

    # 使用单个标签调用 _inertia_dense 和 _inertia_sparse 计算惯性
    inertia_dense = _inertia_dense(
        X_dense, sample_weight, centers, labels, n_threads=1, single_label=label
    )
    inertia_sparse = _inertia_sparse(
        X_sparse, sample_weight, centers, labels, n_threads=1, single_label=label
    )

    # 检查 dense 和 sparse 计算的惯性值是否接近
    assert_allclose(inertia_dense, inertia_sparse, rtol=rtol)
    # 检查 dense 计算的惯性值与预期值是否接近
    assert_allclose(inertia_dense, expected, rtol=rtol)
    # 检查 sparse 计算的惯性值与预期值是否接近
    assert_allclose(inertia_sparse, expected, rtol=rtol)


@pytest.mark.parametrize("Klass, default_n_init", [(KMeans, 10), (MiniBatchKMeans, 3)])
def test_n_init_auto(Klass, default_n_init):
    # 测试 n_init="auto" 时 KMeans 和 MiniBatchKMeans 初始化的行为
    est = Klass(n_init="auto", init="k-means++")
    est.fit(X)
    # 检查是否正确设置了 _n_init 属性
    assert est._n_init == 1

    est = Klass(n_init="auto", init="random")
    est.fit(X)
    # 根据 Klass 类型检查 _n_init 属性是否被正确设置
    assert est._n_init == 10 if Klass.__name__ == "KMeans" else 3


@pytest.mark.parametrize("Estimator", [KMeans, MiniBatchKMeans])
def test_sample_weight_unchanged(Estimator):
    # 检查 KMeans 和 MiniBatchKMeans 是否不会修改原始的 sample_weight 数组 (#17204)
    X = np.array([[1], [2], [4]])
    sample_weight = np.array([0.5, 0.2, 0.3])
    Estimator(n_clusters=2, random_state=0).fit(X, sample_weight=sample_weight)

    # 检查 sample_weight 数组是否保持不变
    assert_array_equal(sample_weight, np.array([0.5, 0.2, 0.3]))
    [
        # 参数示例：n_clusters 参数设置为样本数加一
        ({"n_clusters": n_samples + 1}, r"n_samples.* should be >= n_clusters"),
        # 参数示例：init 参数设置为 X 的前两行作为初始中心点
        (
            {"init": X[:2]},
            r"The shape of the initial centers .* does not match "
            r"the number of clusters",
        ),
        # 参数示例：init 参数为一个 lambda 函数，使用 X 的前两行作为初始中心点
        (
            {"init": lambda X_, k, random_state: X_[:2]},
            r"The shape of the initial centers .* does not match "
            r"the number of clusters",
        ),
        # 参数示例：init 参数设置为 X 的前8行的前两列作为初始中心点
        (
            {"init": X[:8, :2]},
            r"The shape of the initial centers .* does not match "
            r"the number of features of the data",
        ),
        # 参数示例：init 参数为一个 lambda 函数，使用 X 的前8行的前两列作为初始中心点
        (
            {"init": lambda X_, k, random_state: X_[:8, :2]},
            r"The shape of the initial centers .* does not match "
            r"the number of features of the data",
        ),
    ],
# 检查当参数错误时是否会引发清晰的错误消息
# 设置 n_init=1 以避免在预计算初始化时出现警告
km = Estimator(n_init=1)
with pytest.raises(ValueError, match=match):
    km.set_params(**param).fit(X)



# 参数化测试：检查 KMeans++ 算法中参数错误时是否会引发特定的 ValueError
with pytest.raises(ValueError, match=match):
    kmeans_plusplus(X, n_clusters, **param)



# 参数化测试：检查 KMeans++ 算法在不同输入数据下的输出是否正确
# 将输入数据转换为指定的 dtype
data = input_data.astype(dtype)
# 调用 KMeans++ 算法获取聚类中心和样本索引
centers, indices = kmeans_plusplus(
    data, n_clusters, random_state=global_random_seed
)

# 检查索引的数量是否正确，并且所有索引均为非负且在样本数量范围内
assert indices.shape[0] == n_clusters
assert (indices >= 0).all()
assert (indices <= data.shape[0]).all()

# 检查聚类中心的数量是否正确，并且每个维度的值都在数据的范围内
assert centers.shape[0] == n_clusters
assert (centers.max(axis=0) <= data.max(axis=0)).all()
assert (centers.min(axis=0) >= data.min(axis=0)).all()

# 检查索引是否对应于报告的聚类中心
# 使用 X 进行比较，而不是 data，以便对稀疏数据计算得到的中心进行测试
assert_allclose(X[indices].astype(dtype), centers)



# 参数化测试：检查 KMeans++ 算法在指定 x_squared_norms 下的输出是否正确
centers, indices = kmeans_plusplus(X, n_clusters, x_squared_norms=x_squared_norms)

assert_allclose(X[indices], centers)



# 测试：检查内存布局对结果的影响
# 检查 C 风格的数据布局是否与 Fortran 风格的数据布局得到相同的聚类中心
centers_c, _ = kmeans_plusplus(X, n_clusters, random_state=global_random_seed)

X_fortran = np.asfortranarray(X)
centers_fortran, _ = kmeans_plusplus(
    X_fortran, n_clusters, random_state=global_random_seed
)

assert_allclose(centers_c, centers_fortran)



# 测试：_is_same_clustering 实用函数的基本检查
labels1 = np.array([1, 0, 0, 1, 2, 0, 2, 1], dtype=np.int32)
# 断言两组标签是否表示相同的聚类
assert _is_same_clustering(labels1, labels1, 3)

# 另一组标签表示相同的聚类，因为我们可以通过简单地重命名标签来检索第一组标签
# 0 -> 1, 1 -> 2, 2 -> 0
    # 创建一个包含整数的 NumPy 数组，表示第一组聚类的标签
    labels2 = np.array([0, 2, 2, 0, 1, 2, 1, 0], dtype=np.int32)
    # 使用自定义函数 _is_same_clustering 检查 labels1 和 labels2 是否表示相同的聚类结果，期望聚类数为 3
    assert _is_same_clustering(labels1, labels2, 3)

    # 创建另一个包含整数的 NumPy 数组，表示第二组聚类的标签
    labels3 = np.array([1, 0, 0, 2, 2, 0, 2, 1], dtype=np.int32)
    # 使用自定义函数 _is_same_clustering 检查 labels1 和 labels3 是否表示相同的聚类结果，期望聚类数为 3
    # 由于并非所有的 1 都映射到相同的值，因此这些标签不表示相同的聚类
    assert not _is_same_clustering(labels1, labels3, 3)
@pytest.mark.parametrize(
    "kwargs", ({"init": np.str_("k-means++")}, {"init": [[0, 0], [1, 1]], "n_init": 1})
)
def test_kmeans_with_array_like_or_np_scalar_init(kwargs):
    """检查 init 参数是否能够接受 numpy 标量字符串。

    针对 #21964 的非回归测试。
    """
    X = np.asarray([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=np.float64)

    # 创建 KMeans 对象并进行聚类
    clustering = KMeans(n_clusters=2, **kwargs)
    # 不应该引发异常
    clustering.fit(X)


@pytest.mark.parametrize(
    "Klass, method",
    [(KMeans, "fit"), (MiniBatchKMeans, "fit"), (MiniBatchKMeans, "partial_fit")],
)
def test_feature_names_out(Klass, method):
    """检查 `feature_names_out` 在 `KMeans` 和 `MiniBatchKMeans` 中的表现。"""
    class_name = Klass.__name__.lower()
    kmeans = Klass()
    # 调用指定的方法（fit 或 partial_fit）
    getattr(kmeans, method)(X)
    n_clusters = kmeans.cluster_centers_.shape[0]

    # 获取特征名数组
    names_out = kmeans.get_feature_names_out()
    assert_array_equal([f"{class_name}{i}" for i in range(n_clusters)], names_out)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS + [None])
def test_predict_does_not_change_cluster_centers(csr_container):
    """检查 predict 方法是否不会改变聚类中心。

    针对 gh-24253 的非回归测试。
    """
    X, _ = make_blobs(n_samples=200, n_features=10, centers=10, random_state=0)
    if csr_container is not None:
        X = csr_container(X)

    kmeans = KMeans()
    y_pred1 = kmeans.fit_predict(X)
    # 将 cluster_centers_ 设置为只读
    kmeans.cluster_centers_ = create_memmap_backed_data(kmeans.cluster_centers_)
    kmeans.labels_ = create_memmap_backed_data(kmeans.labels_)

    y_pred2 = kmeans.predict(X)
    assert_array_equal(y_pred1, y_pred2)


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_sample_weight_init(init, global_random_seed):
    """检查初始化过程中是否使用了样本权重。

    `_init_centroids` 在所有继承自 _BaseKMeans 的类中是共享的，因此仅需检查 KMeans。
    """
    rng = np.random.RandomState(global_random_seed)
    X, _ = make_blobs(
        n_samples=200, n_features=10, centers=10, random_state=global_random_seed
    )
    x_squared_norms = row_norms(X, squared=True)

    kmeans = KMeans()
    # 使用加权样本进行初始化聚类中心
    clusters_weighted = kmeans._init_centroids(
        X=X,
        x_squared_norms=x_squared_norms,
        init=init,
        sample_weight=rng.uniform(size=X.shape[0]),
        n_centroids=5,
        random_state=np.random.RandomState(global_random_seed),
    )
    # 使用均匀样本权重进行初始化聚类中心
    clusters = kmeans._init_centroids(
        X=X,
        x_squared_norms=x_squared_norms,
        init=init,
        sample_weight=np.ones(X.shape[0]),
        n_centroids=5,
        random_state=np.random.RandomState(global_random_seed),
    )
    # 断言加权和均匀初始化的聚类中心不相等
    with pytest.raises(AssertionError):
        assert_allclose(clusters_weighted, clusters)


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_sample_weight_zero(init, global_random_seed):
    """检查当样本权重为 0 时，该样本是否不会被选中。"""
    # 使用全局随机种子创建一个随机数生成器对象
    rng = np.random.RandomState(global_random_seed)
    # 生成一个包含5个中心的、每个中心包含100个样本的、5个特征的数据集，并指定随机种子
    X, _ = make_blobs(
        n_samples=100, n_features=5, centers=5, random_state=global_random_seed
    )
    # 为每个样本生成一个随机的样本权重，其中一半的样本权重设置为0
    sample_weight = rng.uniform(size=X.shape[0])
    sample_weight[::2] = 0
    # 计算数据集中每个样本的平方范数
    x_squared_norms = row_norms(X, squared=True)
    
    # 创建一个 KMeans 对象
    kmeans = KMeans()
    # 使用 KMeans 对象的 _init_centroids 方法初始化质心
    clusters_weighted = kmeans._init_centroids(
        X=X,
        x_squared_norms=x_squared_norms,
        init=init,
        sample_weight=sample_weight,
        n_centroids=10,
        random_state=np.random.RandomState(global_random_seed),
    )
    # 断言：没有任何一个质心与权重为0的样本点的距离应该接近0
    d = euclidean_distances(X[::2], clusters_weighted)
    assert not np.any(np.isclose(d, 0))
# 使用 pytest 的 parametrize 装饰器为测试函数提供多组参数化输入，array_constr 从 data_containers 中获取，ids 使用 data_containers_ids
@pytest.mark.parametrize("array_constr", data_containers, ids=data_containers_ids)
# 使用 pytest 的 parametrize 装饰器为测试函数提供多组参数化输入，algorithm 可以是 "lloyd" 或 "elkan"
@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
# 定义测试函数 test_relocating_with_duplicates，用于验证当中心点数量大于非重复样本数量时，KMeans 是否停止
def test_relocating_with_duplicates(algorithm, array_constr):
    """Check that kmeans stops when there are more centers than non-duplicate samples

    Non-regression test for issue:
    https://github.com/scikit-learn/scikit-learn/issues/28055
    """
    # 定义样本数据 X
    X = np.array([[0, 0], [1, 1], [1, 1], [1, 0], [0, 1]])
    # 初始化 KMeans 模型，设定簇数为 5，初始中心点为 X，使用指定算法
    km = KMeans(n_clusters=5, init=X, algorithm=algorithm)

    # 预期的警告信息，用于检测是否发出 ConvergenceWarning 警告并匹配特定的消息
    msg = r"Number of distinct clusters \(4\) found smaller than n_clusters \(5\)"
    # 使用 pytest 的 warn 装饰器，检查是否发出 ConvergenceWarning 警告并匹配指定消息
    with pytest.warns(ConvergenceWarning, match=msg):
        # 使用参数化的 array_constr 函数处理 X，然后拟合 KMeans 模型
        km.fit(array_constr(X))

    # 验证 KMeans 模型的迭代次数是否为 1
    assert km.n_iter_ == 1
```