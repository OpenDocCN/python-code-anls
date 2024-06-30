# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_affinity_propagation.py`

```
"""
Testing for Clustering methods

"""

# 导入警告模块
import warnings

# 导入必要的库
import numpy as np
import pytest

# 导入 AffinityPropagation 相关的类和函数
from sklearn.cluster import AffinityPropagation, affinity_propagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS

# 定义簇的数量和中心点
n_clusters = 3
centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
X, _ = make_blobs(
    n_samples=60,
    n_features=2,
    centers=centers,
    cluster_std=0.4,
    shuffle=True,
    random_state=0,
)

# TODO: AffinityPropagation 必须保留其拟合属性的数据类型，相应的测试也需要相应更新。
# 更多详情请参考：https://github.com/scikit-learn/scikit-learn/issues/11000


def test_affinity_propagation(global_random_seed, global_dtype):
    """Test consistency of the affinity propagations."""
    # 计算欧氏距离的平方并取负数，作为相似性矩阵 S
    S = -euclidean_distances(X.astype(global_dtype, copy=False), squared=True)
    preference = np.median(S) * 10
    # 进行亲和传播聚类
    cluster_centers_indices, labels = affinity_propagation(
        S, preference=preference, random_state=global_random_seed
    )

    n_clusters_ = len(cluster_centers_indices)

    # 断言簇的数量与预期的相同
    assert n_clusters == n_clusters_


def test_affinity_propagation_precomputed():
    """Check equality of precomputed affinity matrix to internally computed affinity
    matrix.
    """
    # 计算预先计算的亲和性矩阵和内部计算的亲和性矩阵的一致性
    S = -euclidean_distances(X, squared=True)
    preference = np.median(S) * 10
    # 使用 AffinityPropagation 进行聚类，affinity 设置为 "precomputed"
    af = AffinityPropagation(
        preference=preference, affinity="precomputed", random_state=28
    )
    labels_precomputed = af.fit(S).labels_

    # 再次使用 AffinityPropagation 进行聚类，verbose 设置为 True
    af = AffinityPropagation(preference=preference, verbose=True, random_state=37)
    labels = af.fit(X).labels_

    # 断言两种方法得到的标签数组相等
    assert_array_equal(labels, labels_precomputed)

    cluster_centers_indices = af.cluster_centers_indices_

    n_clusters_ = len(cluster_centers_indices)
    # 断言簇的数量与预期的相同
    assert np.unique(labels).size == n_clusters_
    assert n_clusters == n_clusters_


def test_affinity_propagation_no_copy():
    """Check behaviour of not copying the input data."""
    # 计算预先计算的亲和性矩阵 S
    S = -euclidean_distances(X, squared=True)
    S_original = S.copy()
    preference = np.median(S) * 10
    assert not np.allclose(S.diagonal(), preference)

    # 使用 copy=True 时，S 不应该被修改
    affinity_propagation(S, preference=preference, copy=True, random_state=0)
    assert_allclose(S, S_original)
    assert not np.allclose(S.diagonal(), preference)
    assert_allclose(S.diagonal(), np.zeros(S.shape[0]))

    # 使用 copy=False 时，S 将会被原地修改
    affinity_propagation(S, preference=preference, copy=False, random_state=0)
    assert_allclose(S.diagonal(), preference)

    # 测试 copy=True 和 copy=False 得到相同的结果
    S = S_original.copy()
    # 使用 Affinity Propagation 算法创建对象 af，设置偏好值为 preference，启用详细输出，随机种子为 0
    af = AffinityPropagation(preference=preference, verbose=True, random_state=0)
    
    # 使用对象 af 对数据 X 进行拟合，得到聚类标签 labels
    labels = af.fit(X).labels_
    
    # 调用 affinity_propagation 函数，对相似度矩阵 S 进行聚类，设置偏好值为 preference，不复制输入数据，随机种子为 74
    _, labels_no_copy = affinity_propagation(
        S, preference=preference, copy=False, random_state=74
    )
    
    # 断言：确保两种方法得到的聚类标签完全一致
    assert_array_equal(labels, labels_no_copy)
def test_affinity_propagation_affinity_shape():
    """检查使用 `affinity_propagation` 时亲和矩阵的形状。"""
    # 计算欧氏距离的平方作为相似性矩阵
    S = -euclidean_distances(X, squared=True)
    # 错误信息：相似性矩阵必须是一个方阵
    err_msg = "The matrix of similarities must be a square array"
    # 使用 pytest 检测是否抛出 ValueError，并匹配特定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 调用 affinity_propagation 函数，传入非方阵的 S 切片
        affinity_propagation(S[:, :-1])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_affinity_propagation_precomputed_with_sparse_input(csr_container):
    """测试使用稀疏输入预计算的 AffinityPropagation。"""
    # 错误信息：传入了稀疏数据 X，但需要密集数据
    err_msg = "Sparse data was passed for X, but dense data is required"
    # 使用 pytest 检测是否抛出 TypeError，并匹配特定的错误消息
    with pytest.raises(TypeError, match=err_msg):
        # 创建 AffinityPropagation 实例，指定 affinity 为 "precomputed"，使用 csr_container 创建 (3, 3) 大小的输入
        AffinityPropagation(affinity="precomputed").fit(csr_container((3, 3)))


def test_affinity_propagation_predict(global_random_seed, global_dtype):
    """测试 AffinityPropagation.predict 方法。"""
    # 创建 AffinityPropagation 实例，指定 affinity 为 "euclidean"，并设置随机种子
    af = AffinityPropagation(affinity="euclidean", random_state=global_random_seed)
    # 将 X 转换为指定的数据类型 global_dtype，并在原地修改
    X_ = X.astype(global_dtype, copy=False)
    # 训练模型并预测标签
    labels = af.fit_predict(X_)
    # 直接预测标签，用于测试预测方法的一致性
    labels2 = af.predict(X_)
    # 检查两次预测结果是否一致
    assert_array_equal(labels, labels2)


def test_affinity_propagation_predict_error():
    """测试 AffinityPropagation.predict 中的异常情况。"""
    # 创建 AffinityPropagation 实例，指定 affinity 为 "euclidean"
    af = AffinityPropagation(affinity="euclidean")
    # 使用 pytest 检测是否抛出 NotFittedError，因为模型尚未拟合
    with pytest.raises(NotFittedError):
        af.predict(X)

    # 创建预先计算的相似性矩阵 S
    S = np.dot(X, X.T)
    # 创建 AffinityPropagation 实例，指定 affinity 为 "precomputed"，并设置随机种子
    af = AffinityPropagation(affinity="precomputed", random_state=57)
    # 对预先计算的相似性矩阵 S 进行拟合
    af.fit(S)
    # 使用 pytest 检测是否抛出 ValueError，并匹配特定的错误消息
    with pytest.raises(ValueError, match="expecting 60 features as input"):
        # 预测时传入 X，这在 affinity="precomputed" 模式下不支持
        af.predict(X)


def test_affinity_propagation_fit_non_convergence(global_dtype):
    """测试 affinity_propagation() 不收敛时的情况。"""
    # 创建包含三个样本的数组 X
    X = np.array([[0, 0], [1, 1], [-2, -2]], dtype=global_dtype)
    # 强制只允许单次迭代，以引发不收敛的情况
    af = AffinityPropagation(preference=-10, max_iter=1, random_state=82)

    with pytest.warns(ConvergenceWarning):
        # 对数据 X 进行拟合
        af.fit(X)
    # 检查聚类中心是否为空数组
    assert_allclose(np.empty((0, 2)), af.cluster_centers_)
    # 检查标签是否都被标记为噪音 (-1)
    assert_array_equal(np.array([-1, -1, -1]), af.labels_)


def test_affinity_propagation_equal_mutual_similarities(global_dtype):
    """测试互相相等的相似性的情况。"""
    # 创建包含两个样本的数组 X
    X = np.array([[-1, 1], [1, -1]], dtype=global_dtype)
    # 计算欧氏距离的平方作为相似性矩阵
    S = -euclidean_distances(X, squared=True)

    # 设置 preference > similarity，期望所有样本成为范例
    with pytest.warns(UserWarning, match="mutually equal"):
        # 调用 affinity_propagation 函数，设置 preference 为 0
        cluster_center_indices, labels = affinity_propagation(S, preference=0)
    # 检查每个样本是否成为范例
    assert_array_equal([0, 1], cluster_center_indices)
    assert_array_equal([0, 1], labels)

    # 设置 preference < similarity，期望只有一个聚类，并且第一个样本成为范例
    with pytest.warns(UserWarning, match="mutually equal"):
        # 调用 affinity_propagation 函数，设置 preference 为 -10
        cluster_center_indices, labels = affinity_propagation(S, preference=-10)
    # 检查只有一个聚类，并且第一个样本成为范例
    assert_array_equal([0], cluster_center_indices)
    assert_array_equal([0, 0], labels)
    # 设置不同的警告处理偏好
    with warnings.catch_warnings():
        # 设置警告过滤器，将特定类型的警告转换为异常抛出
        warnings.simplefilter("error", UserWarning)
        # 使用亲和传播算法计算聚类中心和标签，设置首选项为[-20, -10]，随机种子为37
        cluster_center_indices, labels = affinity_propagation(
            S, preference=[-20, -10], random_state=37
        )
    
    # 断言：期望仅有一个聚类中心，且最高优先级的样本作为代表
    assert_array_equal([1], cluster_center_indices)
    # 断言：期望标签为[0, 0]
    assert_array_equal([0, 0], labels)
def test_affinity_propagation_predict_non_convergence(global_dtype):
    # 在亲和传播算法不收敛的情况下，聚类中心应为空数组
    X = np.array([[0, 0], [1, 1], [-2, -2]], dtype=global_dtype)

    # 强制只允许单次迭代以引发不收敛
    with pytest.warns(ConvergenceWarning):
        af = AffinityPropagation(preference=-10, max_iter=1, random_state=75).fit(X)

    # 在预测时，考虑新样本为噪声，因为没有聚类
    to_predict = np.array([[2, 2], [3, 3], [4, 4]])
    with pytest.warns(ConvergenceWarning):
        y = af.predict(to_predict)
    assert_array_equal(np.array([-1, -1, -1]), y)


def test_affinity_propagation_non_convergence_regressiontest(global_dtype):
    X = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1]], dtype=global_dtype
    )
    # 创建亲和传播对象，指定参数并执行拟合
    af = AffinityPropagation(affinity="euclidean", max_iter=2, random_state=34)
    msg = (
        "Affinity propagation did not converge, this model may return degenerate"
        " cluster centers and labels."
    )
    # 断言拟合时会触发收敛警告，并匹配特定的警告信息
    with pytest.warns(ConvergenceWarning, match=msg):
        af.fit(X)

    # 断言聚类标签与预期的数组相等
    assert_array_equal(np.array([0, 0, 0]), af.labels_)


def test_equal_similarities_and_preferences(global_dtype):
    # 不相等的距离
    X = np.array([[0, 0], [1, 1], [-2, -2]], dtype=global_dtype)
    S = -euclidean_distances(X, squared=True)

    # 断言函数对于不同的输入不会返回相等结果
    assert not _equal_similarities_and_preferences(S, np.array(0))
    assert not _equal_similarities_and_preferences(S, np.array([0, 0]))
    assert not _equal_similarities_and_preferences(S, np.array([0, 1]))

    # 相等的距离
    X = np.array([[0, 0], [1, 1]], dtype=global_dtype)
    S = -euclidean_distances(X, squared=True)

    # 不同的偏好
    assert not _equal_similarities_and_preferences(S, np.array([0, 1]))

    # 相同的偏好
    assert _equal_similarities_and_preferences(S, np.array([0, 0]))
    assert _equal_similarities_and_preferences(S, np.array(0))


def test_affinity_propagation_random_state():
    """检查不同的随机状态是否导致不同的初始化，
    通过观察两次迭代后的聚类中心位置来验证。
    """
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=0
    )

    # random_state = 0
    ap = AffinityPropagation(convergence_iter=1, max_iter=2, random_state=0)
    ap.fit(X)
    centers0 = ap.cluster_centers_

    # random_state = 76
    ap = AffinityPropagation(convergence_iter=1, max_iter=2, random_state=76)
    ap.fit(X)
    centers76 = ap.cluster_centers_

    # 断言两个随机状态下的聚类中心尚未收敛到相同的解决方案
    assert np.mean((centers0 - centers76) ** 2) > 1


@pytest.mark.parametrize("container", CSR_CONTAINERS + [np.array])
def test_affinity_propagation_convergence_warning_dense_sparse(container, global_dtype):
    """
    此处包含稠密和稀疏容器的测试参数化。
    """
    # 创建一个包含一个零矩阵的容器 `centers`
    centers = container(np.zeros((1, 10)))
    # 使用种子为42的随机数生成器创建随机数据矩阵 `X`，并设置其数据类型为 `global_dtype`
    rng = np.random.RandomState(42)
    X = rng.rand(40, 10).astype(global_dtype, copy=False)
    # 创建随机标签 `y`，其数据类型为整数
    y = (4 * rng.rand(40)).astype(int)
    # 创建 AffinityPropagation 对象 `ap`，并设置随机种子为46
    ap = AffinityPropagation(random_state=46)
    # 使用数据 `X` 和标签 `y` 来拟合 AffinityPropagation 模型
    ap.fit(X, y)
    # 将预设的簇中心设置为先前创建的 `centers`
    ap.cluster_centers_ = centers
    # 使用警告捕获上下文，捕获收敛警告，确保预测结果与预期相符
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        # 断言预测的簇标签与全零数组的预期结果相等
        assert_array_equal(ap.predict(X), np.zeros(X.shape[0], dtype=int))
# FIXME; this test is broken with different random states, needs to be revisited
# 测试函数，用于验证由于 dtype 改变导致集群错误的问题修复情况
# （非回归测试，针对问题＃10832）
def test_correct_clusters(global_dtype):
    # 创建一个包含四个样本的二维数组，数据类型为全局指定的类型
    X = np.array(
        [[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=global_dtype
    )
    # 使用指定参数创建 AffinityPropagation 对象，并对数据 X 进行拟合
    afp = AffinityPropagation(preference=1, affinity="precomputed", random_state=0).fit(
        X
    )
    # 预期的样本标签结果
    expected = np.array([0, 1, 1, 2])
    # 验证模型输出的标签是否与预期相符
    assert_array_equal(afp.labels_, expected)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input_for_predict(csr_container):
    # 确保稀疏输入能够被 predict 方法接受
    # （非回归测试，针对问题＃20049）
    af = AffinityPropagation(affinity="euclidean", random_state=42)
    # 对模型进行拟合
    af.fit(X)
    # 使用 csr_container 创建一个稀疏矩阵并预测其标签
    labels = af.predict(csr_container((2, 2)))
    # 验证预测的标签是否与期望相符
    assert_array_equal(labels, (2, 2))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_input_for_fit_predict(csr_container):
    # 确保稀疏输入能够被 fit_predict 方法接受
    # （非回归测试，针对问题＃20049）
    af = AffinityPropagation(affinity="euclidean", random_state=42)
    # 创建一个随机状态对象，并使用 csr_container 创建一个稀疏矩阵 X
    rng = np.random.RandomState(42)
    X = csr_container(rng.randint(0, 2, size=(5, 5)))
    # 对模型进行拟合和预测，并获取其标签
    labels = af.fit_predict(X)
    # 验证预测的标签是否与期望相符
    assert_array_equal(labels, (0, 1, 1, 2, 3))


def test_affinity_propagation_equal_points():
    """确保我们不会为相等的点分配多个集群。

    非回归测试，针对：
    https://github.com/scikit-learn/scikit-learn/pull/20043
    """
    # 创建一个所有元素为零的二维数组 X
    X = np.zeros((8, 1))
    # 使用指定参数创建 AffinityPropagation 对象并拟合 X
    af = AffinityPropagation(affinity="euclidean", damping=0.5, random_state=42).fit(X)
    # 验证所有样本的标签是否都为0
    assert np.all(af.labels_ == 0)
```