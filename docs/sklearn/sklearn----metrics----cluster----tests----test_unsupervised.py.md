# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\tests\test_unsupervised.py`

```
# 导入警告模块
import warnings

# 导入必要的库
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse

# 导入 sklearn 中的数据集和评估指标
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
# 导入 sklearn 内部的辅助函数和修复工具
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)

# 使用 pytest 的参数化装饰器定义测试函数 test_silhouette
@pytest.mark.parametrize(
    "sparse_container",
    [None] + CSR_CONTAINERS + CSC_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS,
)
@pytest.mark.parametrize("sample_size", [None, "half"])
def test_silhouette(sparse_container, sample_size):
    # 测试 Silhouette Coefficient
    # 加载鸢尾花数据集
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    # 如果 sparse_container 不为 None，则将数据转换为稀疏格式
    if sparse_container is not None:
        X = sparse_container(X)
    # 根据 sample_size 参数设置样本大小
    sample_size = int(X.shape[0] / 2) if sample_size == "half" else sample_size

    # 计算样本间的欧几里得距离矩阵
    D = pairwise_distances(X, metric="euclidean")
    # 使用预计算的距离矩阵 D 和真实标签 y 计算 Silhouette Coefficient
    score_precomputed = silhouette_score(
        D, y, metric="precomputed", sample_size=sample_size, random_state=0
    )
    # 使用原始数据 X 和真实标签 y 计算 Silhouette Coefficient
    score_euclidean = silhouette_score(
        X, y, metric="euclidean", sample_size=sample_size, random_state=0
    )
    # 断言预计算的 Silhouette Coefficient 大于 0
    assert score_precomputed > 0
    # 断言基于欧几里得距离的 Silhouette Coefficient 大于 0
    assert score_euclidean > 0
    # 断言两种方法计算的 Silhouette Coefficient 相等（近似相等）
    assert score_precomputed == pytest.approx(score_euclidean)


def test_cluster_size_1():
    # 当一个簇中只有一个样本时，断言 Silhouette Coefficient 等于 0（簇 0）。
    # 我们还测试了只有相同样本作为簇的成员时（簇 2），根据我们的知识，这种情况
    # 没有在参考资料中讨论，我们选择一个样本得分为 1 的情况。
    X = [[0.0], [1.0], [1.0], [2.0], [3.0], [3.0]]
    labels = np.array([0, 1, 1, 1, 2, 2])

    # 簇 0: 只有一个样本 -> 根据 Rousseeuw 的惯例，得分为 0
    # 簇 1: intra-cluster = [.5, .5, 1]
    #      inter-cluster = [1, 1, 1]
    #      silhouette    = [.5, .5, 0]
    # 簇 2: intra-cluster = [0, 0]
    #      inter-cluster = [任意值, 任意值]
    #      silhouette    = [1., 1.]

    # 计算整体数据集的 Silhouette Coefficient
    silhouette = silhouette_score(X, labels)
    # 断言 Silhouette Coefficient 不是 NaN
    assert not np.isnan(silhouette)
    # 计算每个样本的 Silhouette Coefficient
    ss = silhouette_samples(X, labels)
    # 断言每个样本的 Silhouette Coefficient 和预期的数组相等
    assert_array_equal(ss, [0, 0.5, 0.5, 0, 1, 1])


def test_silhouette_paper_example():
    # 明确检查每个样本的结果与 Rousseeuw (1987) 的表格 1 对比
    # 表格 1 中的数据

    # （此处未完全展示代码，因为示例中只有注释部分的展示要求）
    lower = [
        5.58,   # 定义一个包含浮点数的列表，用于构建对称矩阵的下三角部分
        7.00,
        6.50,
        7.08,
        7.00,
        3.83,
        4.83,
        5.08,
        8.17,
        5.83,
        2.17,
        5.75,
        6.67,
        6.92,
        4.92,
        6.42,
        5.00,
        5.58,
        6.00,
        4.67,
        6.42,
        3.42,
        5.50,
        6.42,
        6.42,
        5.00,
        3.92,
        6.17,
        2.50,
        4.92,
        6.25,
        7.33,
        4.50,
        2.25,
        6.33,
        2.75,
        6.08,
        6.67,
        4.25,
        2.67,
        6.00,
        6.17,
        6.17,
        6.92,
        6.17,
        5.25,
        6.83,
        4.50,
        3.75,
        5.75,
        5.42,
        6.08,
        5.83,
        6.67,
        3.67,
        4.75,
        3.00,
        6.08,
        6.67,
        5.00,
        5.58,
        4.83,
        6.17,
        5.67,
        6.50,
        6.92,
    ]
    D = np.zeros((12, 12))  # 创建一个12x12的零矩阵，准备存放对称矩阵
    D[np.tril_indices(12, -1)] = lower  # 使用lower列表填充对称矩阵的下三角部分
    D += D.T  # 将对称矩阵的下三角部分复制到上三角，得到完整的对称矩阵

    names = [
        "BEL",  # 定义一个国家名称列表
        "BRA",
        "CHI",
        "CUB",
        "EGY",
        "FRA",
        "IND",
        "ISR",
        "USA",
        "USS",
        "YUG",
        "ZAI",
    ]

    # Data from Figure 2
    labels1 = [1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1]  # 第一个数据集的标签列表
    expected1 = {
        "USA": 0.43,   # 第一个数据集的预期轮廓系数
        "BEL": 0.39,
        "FRA": 0.35,
        "ISR": 0.30,
        "BRA": 0.22,
        "EGY": 0.20,
        "ZAI": 0.19,
        "CUB": 0.40,
        "USS": 0.34,
        "CHI": 0.33,
        "YUG": 0.26,
        "IND": -0.04,
    }
    score1 = 0.28  # 第一个数据集的总体轮廓系数

    # Data from Figure 3
    labels2 = [1, 2, 3, 3, 1, 1, 2, 1, 1, 3, 3, 2]  # 第二个数据集的标签列表
    expected2 = {
        "USA": 0.47,   # 第二个数据集的预期轮廓系数
        "FRA": 0.44,
        "BEL": 0.42,
        "ISR": 0.37,
        "EGY": 0.02,
        "ZAI": 0.28,
        "BRA": 0.25,
        "IND": 0.17,
        "CUB": 0.48,
        "USS": 0.44,
        "YUG": 0.31,
        "CHI": 0.31,
    }
    score2 = 0.33  # 第二个数据集的总体轮廓系数

    for labels, expected, score in [
        (labels1, expected1, score1),  # 遍历两个数据集的标签、预期轮廓系数和总体轮廓系数
        (labels2, expected2, score2),
    ]:
        expected = [expected[name] for name in names]  # 提取每个国家的预期轮廓系数
        # 使用预计的轮廓系数和标签调用silhouette_samples函数，检查结果精确到小数点后两位
        pytest.approx(
            expected,
            silhouette_samples(D, np.array(labels), metric="precomputed"),
            abs=1e-2,
        )
        # 使用总体轮廓系数和标签调用silhouette_score函数，检查结果精确到小数点后两位
        pytest.approx(
            score, silhouette_score(D, np.array(labels), metric="precomputed"), abs=1e-2
        )
def test_correct_labelsize():
    # Assert 1 < n_labels < n_samples
    # 加载鸢尾花数据集
    dataset = datasets.load_iris()
    # 获取数据集的特征数据
    X = dataset.data

    # n_labels = n_samples
    # 创建与样本数相同长度的标签数组
    y = np.arange(X.shape[0])
    # 设置错误消息模板，用于匹配异常信息
    err_msg = (
        r"Number of labels is %d\. Valid values are 2 "
        r"to n_samples - 1 \(inclusive\)" % len(np.unique(y))
    )
    # 断言调用 silhouette_score 函数时引发 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        silhouette_score(X, y)

    # n_labels = 1
    # 创建长度与样本数相同的标签数组，全部为零
    y = np.zeros(X.shape[0])
    # 更新错误消息模板
    err_msg = (
        r"Number of labels is %d\. Valid values are 2 "
        r"to n_samples - 1 \(inclusive\)" % len(np.unique(y))
    )
    # 断言调用 silhouette_score 函数时引发 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        silhouette_score(X, y)


def test_non_encoded_labels():
    # 加载鸢尾花数据集
    dataset = datasets.load_iris()
    # 获取数据集的特征数据
    X = dataset.data
    # 获取数据集的标签数据
    labels = dataset.target
    # 断言调用 silhouette_score 函数时返回值相等
    assert silhouette_score(X, labels * 2 + 10) == silhouette_score(X, labels)
    # 断言调用 silhouette_samples 函数时返回值数组相等
    assert_array_equal(
        silhouette_samples(X, labels * 2 + 10), silhouette_samples(X, labels)
    )


def test_non_numpy_labels():
    # 加载鸢尾花数据集
    dataset = datasets.load_iris()
    # 获取数据集的特征数据
    X = dataset.data
    # 获取数据集的标签数据
    y = dataset.target
    # 断言调用 silhouette_score 函数时返回值相等
    assert silhouette_score(list(X), list(y)) == silhouette_score(X, y)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_silhouette_nonzero_diag(dtype):
    # Make sure silhouette_samples requires diagonal to be zero.
    # Non-regression test for #12178

    # 构造一个对角线为零的矩阵
    dists = pairwise_distances(
        np.array([[0.2, 0.1, 0.12, 1.34, 1.11, 1.6]], dtype=dtype).T
    )
    # 创建标签列表
    labels = [0, 0, 0, 1, 1, 1]

    # 对角线上的小值是允许的
    dists[2][2] = np.finfo(dists.dtype).eps * 10
    # 调用 silhouette_samples 函数，验证预计算的度量值
    silhouette_samples(dists, labels, metric="precomputed")

    # 对角线上大于 eps * 100 的值不被允许
    dists[2][2] = np.finfo(dists.dtype).eps * 1000
    # 断言调用 silhouette_samples 函数时引发 ValueError 异常
    with pytest.raises(ValueError, match="contains non-zero"):
        silhouette_samples(dists, labels, metric="precomputed")


@pytest.mark.parametrize(
    "sparse_container",
    CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS,
)
def test_silhouette_samples_precomputed_sparse(sparse_container):
    """Check that silhouette_samples works for sparse matrices correctly."""
    # 创建一个特征矩阵 X
    X = np.array([[0.2, 0.1, 0.1, 0.2, 0.1, 1.6, 0.2, 0.1]], dtype=np.float32).T
    # 创建一个标签数组 y
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    # 计算特征矩阵 X 的距离矩阵，以稀疏矩阵的方式存储
    pdist_dense = pairwise_distances(X)
    pdist_sparse = sparse_container(pdist_dense)
    # 断言 pdist_sparse 是稀疏矩阵
    assert issparse(pdist_sparse)
    # 调用 silhouette_samples 函数，验证预计算的度量值（稀疏输入）
    output_with_sparse_input = silhouette_samples(pdist_sparse, y, metric="precomputed")
    # 调用 silhouette_samples 函数，验证预计算的度量值（密集输入）
    output_with_dense_input = silhouette_samples(pdist_dense, y, metric="precomputed")
    # 断言两种输入方式得到的输出结果近似相等
    assert_allclose(output_with_sparse_input, output_with_dense_input)


@pytest.mark.parametrize(
    "sparse_container",
    CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS,
)
def test_silhouette_samples_euclidean_sparse(sparse_container):
    """Check that silhouette_samples works for sparse matrices correctly."""
    # 创建一个包含单列数据的 NumPy 数组，数据类型为单精度浮点数
    X = np.array([[0.2, 0.1, 0.1, 0.2, 0.1, 1.6, 0.2, 0.1]], dtype=np.float32).T
    
    # 标签数组 y，表示与 X 中每个数据点相关联的类别
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    
    # 计算 X 中数据点之间的成对距离，返回一个密集矩阵
    pdist_dense = pairwise_distances(X)
    
    # 将密集矩阵 pdist_dense 转换为稀疏格式的容器
    pdist_sparse = sparse_container(pdist_dense)
    
    # 使用 assert 断言 pdist_sparse 是一个稀疏矩阵
    assert issparse(pdist_sparse)
    
    # 使用稀疏输入计算轮廓系数，输出结果保存在 output_with_sparse_input 中
    output_with_sparse_input = silhouette_samples(pdist_sparse, y)
    
    # 使用密集输入计算轮廓系数，输出结果保存在 output_with_dense_input 中
    output_with_dense_input = silhouette_samples(pdist_dense, y)
    
    # 使用 assert_allclose 断言稀疏输入和密集输入计算出的轮廓系数结果非常接近
    assert_allclose(output_with_sparse_input, output_with_dense_input)
@pytest.mark.parametrize(
    "sparse_container", CSC_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS
)
def test_silhouette_reduce(sparse_container):
    """Check for non-CSR input to private method `_silhouette_reduce`."""
    # 创建一个包含样本数据的二维数组 X
    X = np.array([[0.2, 0.1, 0.1, 0.2, 0.1, 1.6, 0.2, 0.1]], dtype=np.float32).T
    # 计算 X 中样本之间的距离矩阵
    pdist_dense = pairwise_distances(X)
    # 将密集距离矩阵转换为稀疏矩阵
    pdist_sparse = sparse_container(pdist_dense)
    # 创建样本的标签
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    # 统计每个标签出现的次数
    label_freqs = np.bincount(y)
    # 断言在输入为非 CSR 稀疏矩阵时会引发 TypeError 异常
    with pytest.raises(
        TypeError,
        match="Expected CSR matrix. Please pass sparse matrix in CSR format.",
    ):
        _silhouette_reduce(pdist_sparse, start=0, labels=y, label_freqs=label_freqs)


def assert_raises_on_only_one_label(func):
    """Assert message when there is only one label"""
    # 创建一个随机数生成器
    rng = np.random.RandomState(seed=0)
    # 断言当只有一个标签时会引发 ValueError 异常
    with pytest.raises(ValueError, match="Number of labels is"):
        func(rng.rand(10, 2), np.zeros(10))


def assert_raises_on_all_points_same_cluster(func):
    """Assert message when all point are in different clusters"""
    # 创建一个随机数生成器
    rng = np.random.RandomState(seed=0)
    # 断言当所有点都在不同的簇中时会引发 ValueError 异常
    with pytest.raises(ValueError, match="Number of labels is"):
        func(rng.rand(10, 2), np.arange(10)


def test_calinski_harabasz_score():
    assert_raises_on_only_one_label(calinski_harabasz_score)

    assert_raises_on_all_points_same_cluster(calinski_harabasz_score)

    # 断言当所有样本相同时值为 1.0
    assert 1.0 == calinski_harabasz_score(np.ones((10, 2)), [0] * 5 + [1] * 5)

    # 断言当所有簇的均值相同时值为 0.0
    assert 0.0 == calinski_harabasz_score([[-1, -1], [1, 1]] * 10, [0] * 10 + [1] * 10)

    # 一般情况下的断言测试
    X = (
        [[0, 0], [1, 1]] * 5
        + [[3, 3], [4, 4]] * 5
        + [[0, 4], [1, 3]] * 5
        + [[3, 1], [4, 0]] * 5
    )
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    pytest.approx(calinski_harabasz_score(X, labels), 45 * (40 - 4) / (5 * (4 - 1)))


def test_davies_bouldin_score():
    assert_raises_on_only_one_label(davies_bouldin_score)
    assert_raises_on_all_points_same_cluster(davies_bouldin_score)

    # 断言当所有样本相同时值为 0.0
    assert davies_bouldin_score(np.ones((10, 2)), [0] * 5 + [1] * 5) == pytest.approx(
        0.0
    )

    # 断言当所有簇的均值相同时值为 0.0
    assert davies_bouldin_score(
        [[-1, -1], [1, 1]] * 10, [0] * 10 + [1] * 10
    ) == pytest.approx(0.0)

    # 一般情况下的断言测试
    X = (
        [[0, 0], [1, 1]] * 5
        + [[3, 3], [4, 4]] * 5
        + [[0, 4], [1, 3]] * 5
        + [[3, 1], [4, 0]] * 5
    )
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    pytest.approx(davies_bouldin_score(X, labels), 2 * np.sqrt(0.5) / 3)

    # 确保在一般情况下不会引发除以零的警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        davies_bouldin_score(X, labels)
    # 定义一个用于测试的数据集 X，包含四个样本，每个样本是一个二维向量
    X = [[0, 0], [2, 2], [3, 3], [5, 5]]
    
    # 定义每个样本的类标签
    labels = [0, 0, 1, 2]
    
    # 调用 davies_bouldin_score 函数计算指定标签的 Davies-Bouldin 分数，并使用 pytest 的 approx 函数进行近似比较
    # Davies-Bouldin 分数是一种聚类效果评估指标
    pytest.approx(davies_bouldin_score(X, labels), (5.0 / 4) / 3)
# 定义一个测试函数，用于检查 silhouette_score 在预先计算的整数度量上的工作情况
def test_silhouette_score_integer_precomputed():
    """Check that silhouette_score works for precomputed metrics that are integers.

    Non-regression test for #22107.
    """
    # 调用 silhouette_score 函数，传入预先计算的整数度量和对应的标签，指定度量方式为预先计算
    result = silhouette_score(
        [[0, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 0, 1], metric="precomputed"
    )
    # 断言结果接近 1/6
    assert result == pytest.approx(1 / 6)

    # 当整数度量中对角线上存在非零值时，应该引发 ValueError 异常
    with pytest.raises(ValueError, match="contains non-zero"):
        # 调用 silhouette_score 函数，传入包含非零值的预先计算整数度量和对应的标签，指定度量方式为预先计算
        silhouette_score(
            [[1, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 0, 1], metric="precomputed"
        )
```