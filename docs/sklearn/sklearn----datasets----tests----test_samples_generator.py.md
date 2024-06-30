# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_samples_generator.py`

```
# 导入必要的库和模块
import re  # 导入正则表达式模块
from collections import defaultdict  # 导入默认字典模块
from functools import partial  # 导入偏函数模块

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest模块，用于编写和运行测试
import scipy.sparse as sp  # 导入SciPy稀疏矩阵模块

from sklearn.datasets import (  # 导入sklearn中的数据生成模块
    make_biclusters,
    make_blobs,
    make_checkerboard,
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_hastie_10_2,
    make_low_rank_matrix,
    make_moons,
    make_multilabel_classification,
    make_regression,
    make_s_curve,
    make_sparse_coded_signal,
    make_sparse_spd_matrix,
    make_sparse_uncorrelated,
    make_spd_matrix,
    make_swiss_roll,
)
from sklearn.utils._testing import (  # 导入sklearn中的测试工具模块
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.validation import assert_all_finite  # 导入sklearn中的数据验证模块


def test_make_classification():
    # 定义权重列表
    weights = [0.1, 0.25]
    # 生成分类数据集
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=1,
        n_repeated=1,
        n_classes=3,
        n_clusters_per_class=1,
        hypercube=False,
        shift=None,
        scale=None,
        weights=weights,
        random_state=0,
    )

    # 断言权重与预期相符
    assert weights == [0.1, 0.25]
    # 断言生成数据集的形状
    assert X.shape == (100, 20), "X shape mismatch"
    assert y.shape == (100,), "y shape mismatch"
    # 断言类别数目符合预期
    assert np.unique(y).shape == (3,), "Unexpected number of classes"
    # 断言类别0的样本数目符合预期
    assert sum(y == 0) == 10, "Unexpected number of samples in class #0"
    # 断言类别1的样本数目符合预期
    assert sum(y == 1) == 25, "Unexpected number of samples in class #1"
    # 断言类别2的样本数目符合预期
    assert sum(y == 2) == 65, "Unexpected number of samples in class #2"

    # 测试当n_features > 30时的数据集生成
    X, y = make_classification(
        n_samples=2000,
        n_features=31,
        n_informative=31,
        n_redundant=0,
        n_repeated=0,
        hypercube=True,
        scale=0.5,
        random_state=0,
    )

    # 断言生成数据集的形状
    assert X.shape == (2000, 31), "X shape mismatch"
    assert y.shape == (2000,), "y shape mismatch"
    # 断言生成的数据集中有预期数目的唯一行
    assert (
        np.unique(X.view([("", X.dtype)] * X.shape[1]))
        .view(X.dtype)
        .reshape(-1, X.shape[1])
        .shape[0]
        == 2000
    ), "Unexpected number of unique rows"


def test_make_classification_informative_features():
    """Test the construction of informative features in make_classification

    Also tests `n_clusters_per_class`, `n_classes`, `hypercube` and
    fully-specified `weights`.
    """
    # 设置非常分离的类别；检查顶点是否唯一且对应于类别
    class_sep = 1e6
    # 部分应用make_classification函数，设置参数
    make = partial(
        make_classification,
        class_sep=class_sep,
        n_redundant=0,
        n_repeated=0,
        flip_y=0,
        shift=0,
        scale=1,
        shuffle=False,
    )
    for n_informative, weights, n_clusters_per_class in [
        (2, [1], 1),  # 设置数据集特征数为2，权重为[1]，每个类的簇数为1
        (2, [1 / 3] * 3, 1),  # 设置数据集特征数为2，权重为[1/3, 1/3, 1/3]，每个类的簇数为1
        (2, [1 / 4] * 4, 1),  # 设置数据集特征数为2，权重为[1/4, 1/4, 1/4, 1/4]，每个类的簇数为1
        (2, [1 / 2] * 2, 2),  # 设置数据集特征数为2，权重为[1/2, 1/2]，每个类的簇数为2
        (2, [3 / 4, 1 / 4], 2),  # 设置数据集特征数为2，权重为[3/4, 1/4]，每个类的簇数为2
        (10, [1 / 3] * 3, 10),  # 设置数据集特征数为10，权重为[1/3, 1/3, 1/3]，每个类的簇数为10
        (int(64), [1], 1),  # 设置数据集特征数为64，权重为[1]，每个类的簇数为1
    ]:
        n_classes = len(weights)  # 计算类的数量，即权重列表的长度
        n_clusters = n_classes * n_clusters_per_class  # 计算总的簇数
        n_samples = n_clusters * 50  # 计算总的样本数，每个簇生成50个样本

        for hypercube in (False, True):  # 遍历是否在超立方体上的选项
            X, y = make(
                n_samples=n_samples,  # 指定生成的样本数
                n_classes=n_classes,  # 指定类的数量
                weights=weights,  # 指定每个类的权重
                n_features=n_informative,  # 指定特征数
                n_informative=n_informative,  # 指定有信息特征数
                n_clusters_per_class=n_clusters_per_class,  # 指定每个类的簇数
                hypercube=hypercube,  # 指定是否在超立方体上
                random_state=0,  # 指定随机种子
            )

            assert X.shape == (n_samples, n_informative)  # 断言生成的样本数据形状正确
            assert y.shape == (n_samples,)  # 断言生成的标签数据形状正确

            # 根据符号进行聚类，将其视为字符串以允许唯一化处理
            signs = np.sign(X)  # 计算数据的符号
            signs = signs.view(dtype="|S{0}".format(signs.strides[0])).ravel()  # 将符号视图转换为字符串形式
            unique_signs, cluster_index = np.unique(signs, return_inverse=True)  # 获取唯一符号和其在原数组中的索引

            assert (
                len(unique_signs) == n_clusters
            ), "Wrong number of clusters, or not in distinct quadrants"  # 断言簇的数量是否正确

            clusters_by_class = defaultdict(set)  # 创建一个默认字典，用于按类别存储簇的集合
            for cluster, cls in zip(cluster_index, y):  # 遍历每个簇和其对应的类别
                clusters_by_class[cls].add(cluster)  # 将簇添加到相应类别的集合中
            for clusters in clusters_by_class.values():  # 遍历所有类别的簇集合
                assert (
                    len(clusters) == n_clusters_per_class
                ), "Wrong number of clusters per class"  # 断言每个类别的簇数是否正确
            assert len(clusters_by_class) == n_classes, "Wrong number of classes"  # 断言类的数量是否正确

            assert_array_almost_equal(
                np.bincount(y) / len(y) // weights,
                [1] * n_classes,
                err_msg="Wrong number of samples per class",
            )  # 断言每个类的样本数是否正确

            # 确保在超立方体的顶点上
            for cluster in range(len(unique_signs)):  # 遍历每个簇
                centroid = X[cluster_index == cluster].mean(axis=0)  # 计算簇的质心
                if hypercube:  # 如果在超立方体上
                    assert_array_almost_equal(
                        np.abs(centroid) / class_sep,
                        np.ones(n_informative),
                        decimal=5,
                        err_msg="Clusters are not centered on hypercube vertices",
                    )  # 断言簇是否位于超立方体顶点上
                else:  # 如果不在超立方体上
                    with pytest.raises(AssertionError):
                        assert_array_almost_equal(
                            np.abs(centroid) / class_sep,
                            np.ones(n_informative),
                            decimal=5,
                            err_msg=(
                                "Clusters should not be centered on hypercube vertices"
                            ),
                        )  # 断言簇不应位于超立方体顶点上时会引发断言错误

    with pytest.raises(ValueError):
        make(n_features=2, n_informative=2, n_classes=5, n_clusters_per_class=1)
    # 使用 pytest 框架来测试异常情况，确保 make 函数在给定参数下会引发 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 make 函数，传入参数：特征数为 2，信息特征数为 2，类别数为 3，每类簇数为 2
        make(n_features=2, n_informative=2, n_classes=3, n_clusters_per_class=2)
# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试不同的输入组合
@pytest.mark.parametrize(
    "weights, err_type, err_msg",
    [
        # 空列表作为 weights，预期引发 ValueError 异常，且异常消息为指定文本
        ([], ValueError, "Weights specified but incompatible with number of classes."),
        (
            # 非空列表 weights，预期引发 ValueError 异常，且异常消息为指定文本
            [0.25, 0.75, 0.1],
            ValueError,
            "Weights specified but incompatible with number of classes.",
        ),
        (
            # 空的 NumPy 数组 weights，预期引发 ValueError 异常，且异常消息为指定文本
            np.array([]),
            ValueError,
            "Weights specified but incompatible with number of classes.",
        ),
        (
            # 非空的 NumPy 数组 weights，预期引发 ValueError 异常，且异常消息为指定文本
            np.array([0.25, 0.75, 0.1]),
            ValueError,
            "Weights specified but incompatible with number of classes.",
        ),
        (
            # 使用随机生成的权重数组，预期引发 ValueError 异常，且异常消息为指定文本
            np.random.random(3),
            ValueError,
            "Weights specified but incompatible with number of classes.",
        ),
    ],
)
def test_make_classification_weights_type(weights, err_type, err_msg):
    # 使用 pytest.raises 检查是否引发了指定类型的异常，并且异常消息匹配预期的文本
    with pytest.raises(err_type, match=err_msg):
        make_classification(weights=weights)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试不同的输入组合
@pytest.mark.parametrize("kwargs", [{}, {"n_classes": 3, "n_informative": 3}])
def test_make_classification_weights_array_or_list_ok(kwargs):
    # 使用列表形式的 weights 调用 make_classification 函数，并验证结果近似相等
    X1, y1 = make_classification(weights=[0.1, 0.9], random_state=0, **kwargs)
    # 使用 NumPy 数组形式的 weights 调用 make_classification 函数，并验证结果近似相等
    X2, y2 = make_classification(weights=np.array([0.1, 0.9]), random_state=0, **kwargs)
    # 断言两种调用方式得到的 X 和 y 结果近似相等
    assert_almost_equal(X1, X2)
    assert_almost_equal(y1, y2)


# 测试 make_multilabel_classification 函数的返回结果是否符合预期
def test_make_multilabel_classification_return_sequences():
    # 对于 allow_unlabeled 和 min_length 参数进行迭代测试
    for allow_unlabeled, min_length in zip((True, False), (0, 1)):
        # 调用 make_multilabel_classification 函数生成数据集 X, Y
        X, Y = make_multilabel_classification(
            n_samples=100,
            n_features=20,
            n_classes=3,
            random_state=0,
            return_indicator=False,
            allow_unlabeled=allow_unlabeled,
        )
        # 断言生成的 X 的形状符合预期
        assert X.shape == (100, 20), "X shape mismatch"
        # 如果不允许无标签，断言 Y 中的最大值不超过 2
        if not allow_unlabeled:
            assert max([max(y) for y in Y]) == 2
        # 断言 Y 中每个样本的标签长度不小于 min_length
        assert min([len(y) for y in Y]) == min_length
        # 断言 Y 中每个样本的标签长度不超过 3
        assert max([len(y) for y in Y]) <= 3


# 测试 make_multilabel_classification 函数返回的指示器形式的 Y 是否符合预期
def test_make_multilabel_classification_return_indicator():
    # 对于 allow_unlabeled 和 min_length 参数进行迭代测试
    for allow_unlabeled, min_length in zip((True, False), (0, 1)):
        # 调用 make_multilabel_classification 函数生成数据集 X, Y
        X, Y = make_multilabel_classification(
            n_samples=25,
            n_features=20,
            n_classes=3,
            random_state=0,
            allow_unlabeled=allow_unlabeled,
        )
        # 断言生成的 X 的形状符合预期
        assert X.shape == (25, 20), "X shape mismatch"
        # 断言生成的 Y 的形状为 (25, 3)
        assert Y.shape == (25, 3), "Y shape mismatch"
        # 断言 Y 中每个类别的总和大于 min_length
        assert np.all(np.sum(Y, axis=0) > min_length)

    # 此外，测试返回分布和指示器的情况为 True 的情况
    X2, Y2, p_c, p_w_c = make_multilabel_classification(
        n_samples=25,
        n_features=20,
        n_classes=3,
        random_state=0,
        allow_unlabeled=allow_unlabeled,
        return_distributions=True,
    )

    # 断言 X, Y, p_c, p_w_c 四个返回结果近似相等
    assert_array_almost_equal(X, X2)
    assert_array_equal(Y, Y2)
    # 断言 p_c 的形状为 (3,)，并且其元素之和近似为 1
    assert p_c.shape == (3,)
    assert_almost_equal(p_c.sum(), 1)
    # 断言 p_w_c 的形状为 (20, 3)，并且每列的和近似为 1
    assert p_w_c.shape == (20, 3)
    assert_almost_equal(p_w_c.sum(axis=0), [1] * 3)
# 测试生成多标签分类数据集，返回稀疏表示的指示器矩阵
def test_make_multilabel_classification_return_indicator_sparse():
    # 遍历两种参数组合：允许未标记的样本和最小长度
    for allow_unlabeled, min_length in zip((True, False), (0, 1)):
        # 使用 make_multilabel_classification 生成数据集
        X, Y = make_multilabel_classification(
            n_samples=25,                   # 样本数
            n_features=20,                  # 特征数
            n_classes=3,                    # 类别数
            random_state=0,                 # 随机种子
            return_indicator="sparse",      # 返回稀疏表示的指示器矩阵
            allow_unlabeled=allow_unlabeled, # 是否允许未标记的样本
        )
        assert X.shape == (25, 20), "X shape mismatch"  # 断言确认 X 的形状
        assert Y.shape == (25, 3), "Y shape mismatch"  # 断言确认 Y 的形状
        assert sp.issparse(Y)                         # 断言确认 Y 是否为稀疏矩阵


# 测试生成 Hastie 数据集（10维，2类）
def test_make_hastie_10_2():
    X, y = make_hastie_10_2(n_samples=100, random_state=0)
    assert X.shape == (100, 10), "X shape mismatch"    # 断言确认 X 的形状
    assert y.shape == (100,), "y shape mismatch"       # 断言确认 y 的形状
    assert np.unique(y).shape == (2,), "Unexpected number of classes"  # 断言确认 y 中类别数为2


# 测试生成回归数据集
def test_make_regression():
    X, y, c = make_regression(
        n_samples=100,                   # 样本数
        n_features=10,                   # 特征数
        n_informative=3,                 # 有信息特征数
        effective_rank=5,                # 有效秩
        coef=True,                       # 是否返回系数
        bias=0.0,                        # 偏置
        noise=1.0,                       # 噪声标准差
        random_state=0,                  # 随机种子
    )

    assert X.shape == (100, 10), "X shape mismatch"    # 断言确认 X 的形状
    assert y.shape == (100,), "y shape mismatch"       # 断言确认 y 的形状
    assert c.shape == (10,), "coef shape mismatch"     # 断言确认系数 c 的形状
    assert sum(c != 0.0) == 3, "Unexpected number of informative features"  # 断言确认有信息特征数为3

    # 测试 y 是否接近于 np.dot(X, c) + bias + N(0, 1.0)
    assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)

    # 测试少量特征的情况
    X, y = make_regression(n_samples=100, n_features=1)  # n_informative=3
    assert X.shape == (100, 1)


# 测试生成多目标回归数据集
def test_make_regression_multitarget():
    X, y, c = make_regression(
        n_samples=100,                   # 样本数
        n_features=10,                   # 特征数
        n_informative=3,                 # 有信息特征数
        n_targets=3,                     # 目标数
        coef=True,                       # 是否返回系数
        noise=1.0,                       # 噪声标准差
        random_state=0,                  # 随机种子
    )

    assert X.shape == (100, 10), "X shape mismatch"    # 断言确认 X 的形状
    assert y.shape == (100, 3), "y shape mismatch"     # 断言确认 y 的形状
    assert c.shape == (10, 3), "coef shape mismatch"   # 断言确认系数 c 的形状
    assert_array_equal(sum(c != 0.0), 3, "Unexpected number of informative features")  # 断言确认有信息特征数为3

    # 测试 y 是否接近于 np.dot(X, c) + bias + N(0, 1.0)
    assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)


# 测试生成聚类数据集
def test_make_blobs():
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    X, y = make_blobs(
        random_state=0,                  # 随机种子
        n_samples=50,                    # 样本数
        n_features=2,                    # 特征数
        centers=cluster_centers,         # 聚类中心
        cluster_std=cluster_stds,        # 聚类标准差
    )

    assert X.shape == (50, 2), "X shape mismatch"    # 断言确认 X 的形状
    assert y.shape == (50,), "y shape mismatch"       # 断言确认 y 的形状
    assert np.unique(y).shape == (3,), "Unexpected number of blobs"  # 断言确认 y 中类别数为3
    for i, (ctr, std) in enumerate(zip(cluster_centers, cluster_stds)):
        assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")


# 测试生成带有指定样本数列表的聚类数据集
def test_make_blobs_n_samples_list():
    n_samples = [50, 30, 20]
    X, y = make_blobs(n_samples=n_samples, n_features=2, random_state=0)

    assert X.shape == (sum(n_samples), 2), "X shape mismatch"  # 断言确认 X 的形状
    # 断言语句，验证所有类别标签 y 中每个类别的样本数是否与预期的 n_samples 数量一致
    assert all(
        np.bincount(y, minlength=len(n_samples)) == n_samples
    ), "Incorrect number of samples per blob"
# 定义一个测试函数，用于测试 make_blobs 函数生成样本数据时指定 n_samples 和 centers 参数的情况
def test_make_blobs_n_samples_list_with_centers():
    # 指定每个簇中的样本数目列表
    n_samples = [20, 20, 20]
    # 指定每个簇的中心点坐标
    centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    # 指定每个簇的标准差
    cluster_stds = np.array([0.05, 0.2, 0.4])
    # 调用 make_blobs 函数生成样本数据 X 和对应的标签 y
    X, y = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=cluster_stds, random_state=0
    )

    # 断言生成的样本数据 X 的形状是否正确
    assert X.shape == (sum(n_samples), 2), "X shape mismatch"
    # 断言生成的标签 y 中每个簇的样本数是否与 n_samples 中指定的一致
    assert all(
        np.bincount(y, minlength=len(n_samples)) == n_samples
    ), "Incorrect number of samples per blob"
    # 对每个簇的数据进行额外的标准差检查
    for i, (ctr, std) in enumerate(zip(centers, cluster_stds)):
        assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")


# 使用 pytest 的参数化装饰器，测试 make_blobs 函数在 n_samples 参数为列表、NumPy 数组和元组时的行为
@pytest.mark.parametrize(
    "n_samples", [[5, 3, 0], np.array([5, 3, 0]), tuple([5, 3, 0])]
)
def test_make_blobs_n_samples_centers_none(n_samples):
    # 将 centers 参数设为 None，测试 make_blobs 函数生成样本数据 X 和对应的标签 y
    centers = None
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=0)

    # 断言生成的样本数据 X 的形状是否正确
    assert X.shape == (sum(n_samples),
    # 使用 assert_array_almost_equal 函数验证 y 数组与表达式的计算结果的几乎相等性
    assert_array_almost_equal(
        y, (X[:, 0] ** 2 + (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5
    )
# 定义测试函数，生成 Friedman 3 数据集的测试用例
def test_make_friedman3():
    # 使用 make_friedman3 函数生成数据集 X 和目标变量 y
    X, y = make_friedman3(n_samples=5, noise=0.0, random_state=0)

    # 断言 X 的形状为 (5, 4)，验证数据集 X 的形状是否符合预期
    assert X.shape == (5, 4), "X shape mismatch"
    # 断言 y 的形状为 (5,)，验证目标变量 y 的形状是否符合预期
    assert y.shape == (5,), "y shape mismatch"

    # 断言生成的 y 与给定公式 np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) 的结果近似相等
    assert_array_almost_equal(
        y, np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0])
    )


# 定义测试函数，生成低秩矩阵的测试用例
def test_make_low_rank_matrix():
    # 使用 make_low_rank_matrix 函数生成低秩矩阵 X
    X = make_low_rank_matrix(
        n_samples=50,
        n_features=25,
        effective_rank=5,
        tail_strength=0.01,
        random_state=0,
    )

    # 断言 X 的形状为 (50, 25)，验证生成的矩阵 X 的形状是否符合预期
    assert X.shape == (50, 25), "X shape mismatch"

    # 导入 svd 函数，对矩阵 X 进行奇异值分解
    from numpy.linalg import svd

    u, s, v = svd(X)
    # 断言奇异值 s 的总和与指定的有效秩数 5 之间的差距小于 0.1，验证矩阵 X 的秩是否近似为 5
    assert sum(s) - 5 < 0.1, "X rank is not approximately 5"


# 定义测试函数，生成稀疏编码信号的测试用例
def test_make_sparse_coded_signal():
    # 使用 make_sparse_coded_signal 函数生成稀疏编码信号的数据集 Y, D, X
    Y, D, X = make_sparse_coded_signal(
        n_samples=5,
        n_components=8,
        n_features=10,
        n_nonzero_coefs=3,
        random_state=0,
    )

    # 断言 Y 的形状为 (5, 10)，验证数据集 Y 的形状是否符合预期
    assert Y.shape == (5, 10), "Y shape mismatch"
    # 断言 D 的形状为 (8, 10)，验证字典 D 的形状是否符合预期
    assert D.shape == (8, 10), "D shape mismatch"
    # 断言 X 的形状为 (5, 8)，验证编码矩阵 X 的形状是否符合预期
    assert X.shape == (5, 8), "X shape mismatch"
    # 对于每一行的编码向量 X[row]，断言其非零元素的数量是否符合预期值 3
    for row in X:
        assert len(np.flatnonzero(row)) == 3, "Non-zero coefs mismatch"
    # 断言 Y 与 X @ D 的乘积结果近似相等
    assert_allclose(Y, X @ D)
    # 断言 D 的每行向量的二范数是否近似为 1
    assert_allclose(np.sqrt((D**2).sum(axis=1)), np.ones(D.shape[0]))


# 定义测试函数，生成稀疏不相关数据集的测试用例
def test_make_sparse_uncorrelated():
    # 使用 make_sparse_uncorrelated 函数生成稀疏不相关数据集 X, y
    X, y = make_sparse_uncorrelated(n_samples=5, n_features=10, random_state=0)

    # 断言 X 的形状为 (5, 10)，验证数据集 X 的形状是否符合预期
    assert X.shape == (5, 10), "X shape mismatch"
    # 断言 y 的形状为 (5,)，验证目标变量 y 的形状是否符合预期
    assert y.shape == (5,), "y shape mismatch"


# 定义测试函数，生成正定对称矩阵的测试用例
def test_make_spd_matrix():
    # 使用 make_spd_matrix 函数生成正定对称矩阵 X
    X = make_spd_matrix(n_dim=5, random_state=0)

    # 断言 X 的形状为 (5, 5)，验证生成的矩阵 X 的形状是否符合预期
    assert X.shape == (5, 5), "X shape mismatch"
    # 断言矩阵 X 是否近似等于其转置矩阵
    assert_array_almost_equal(X, X.T)

    # 导入 eig 函数，计算矩阵 X 的特征值和特征向量
    from numpy.linalg import eig

    eigenvalues, _ = eig(X)
    # 断言矩阵 X 的所有特征值是否大于 0，验证矩阵 X 是否为正定矩阵
    assert np.all(eigenvalues > 0), "X is not positive-definite"


# 导入 pytest 的参数化装饰器，定义参数化测试函数，生成稀疏正定对称矩阵的测试用例
@pytest.mark.parametrize("norm_diag", [True, False])
@pytest.mark.parametrize(
    "sparse_format", [None, "bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
)
def test_make_sparse_spd_matrix(norm_diag, sparse_format, global_random_seed):
    n_dim = 5
    # 使用 make_sparse_spd_matrix 函数生成稀疏正定对称矩阵 X
    X = make_sparse_spd_matrix(
        n_dim=n_dim,
        norm_diag=norm_diag,
        sparse_format=sparse_format,
        random_state=global_random_seed,
    )

    # 断言 X 的形状为 (n_dim, n_dim)，验证生成的矩阵 X 的形状是否符合预期
    assert X.shape == (n_dim, n_dim), "X shape mismatch"
    if sparse_format is None:
        # 如果 sparse_format 为 None，则断言 X 不是稀疏矩阵，并且 X 近似等于其转置矩阵
        assert not sp.issparse(X)
        assert_allclose(X, X.T)
        Xarr = X
    else:
        # 如果 sparse_format 不为 None，则断言 X 是稀疏矩阵，并且稀疏格式为 sparse_format，同时 X 与其转置矩阵近似相等
        assert sp.issparse(X) and X.format == sparse_format
        assert_allclose_dense_sparse(X, X.T)
        Xarr = X.toarray()

    # 导入 eig 函数，计算矩阵 Xarr 的特征值和特征向量
    from numpy.linalg import eig

    # 不使用 scipy.sparse.linalg.eigs 因为它无法找到所有特征值
    eigenvalues, _ = eig(Xarr)
    # 断言矩阵 Xarr 的所有特征值是否大于 0，验证矩阵 X 是否为正定矩阵
    assert np.all(eigenvalues > 0), "X is not positive-definite"

    if norm_diag:
        # 如果 norm_diag 为 True，则检查矩阵 Xarr 的主对角线元素是否近似为 1
        assert_array_almost_equal(Xarr.diagonal(), np.ones(n_dim))


# 标记待办事项，测试函数将在版本 1.6 移除
def test_make_sparse_spd_matrix_deprecation_warning():
    """Check the message for future deprecation."""
    # 断言在测试函数运行时会发出版本 1.4 中弃用的警告消息
    warn_msg = "dim was deprecated in version 1.4"
    # 使用 pytest 检查是否会发出 FutureWarning 并匹配特定的警告消息
    with pytest.warns(FutureWarning, match=warn_msg):
        # 调用函数 make_sparse_spd_matrix，设置参数 dim=1
        make_sparse_spd_matrix(
            dim=1,
        )
    
    # 定义错误消息，指出 `dim` 和 `n_dim` 不能同时指定
    error_msg = "`dim` and `n_dim` cannot be both specified"
    # 使用 pytest 检查是否会引发 ValueError 并匹配特定的错误消息
    with pytest.raises(ValueError, match=error_msg):
        # 调用函数 make_sparse_spd_matrix，设置参数 dim=1 和 n_dim=1
        make_sparse_spd_matrix(
            dim=1,
            n_dim=1,
        )
    
    # 调用函数 make_sparse_spd_matrix()，返回值赋给变量 X
    X = make_sparse_spd_matrix()
    # 断言 X 的列数为 1
    assert X.shape[1] == 1
# 使用 pytest.mark.parametrize 装饰器为 test_make_swiss_roll 函数设置参数化测试，hole 参数分别为 False 和 True
@pytest.mark.parametrize("hole", [False, True])
def test_make_swiss_roll(hole):
    # 调用 make_swiss_roll 函数生成数据集 X 和目标向量 t
    X, t = make_swiss_roll(n_samples=5, noise=0.0, random_state=0, hole=hole)

    # 断言数据集 X 的形状为 (5, 3)
    assert X.shape == (5, 3)
    # 断言目标向量 t 的形状为 (5,)
    assert t.shape == (5,)
    # 断言 X 的第一列近似等于 t 乘以其余两列的余弦值
    assert_array_almost_equal(X[:, 0], t * np.cos(t))
    # 断言 X 的第三列近似等于 t 乘以其余两列的正弦值
    assert_array_almost_equal(X[:, 2], t * np.sin(t))


# 测试 make_s_curve 函数
def test_make_s_curve():
    # 生成数据集 X 和目标向量 t
    X, t = make_s_curve(n_samples=5, noise=0.0, random_state=0)

    # 断言数据集 X 的形状为 (5, 3)，如果不匹配，输出错误信息 "X shape mismatch"
    assert X.shape == (5, 3), "X shape mismatch"
    # 断言目标向量 t 的形状为 (5)，如果不匹配，输出错误信息 "t shape mismatch"
    assert t.shape == (5,), "t shape mismatch"
    # 断言 X 的第一列近似等于 t 的正弦值
    assert_array_almost_equal(X[:, 0], np.sin(t))
    # 断言 X 的第三列近似等于 t 的符号函数乘以 t 的余弦值减一
    assert_array_almost_equal(X[:, 2], np.sign(t) * (np.cos(t) - 1))


# 测试 make_biclusters 函数
def test_make_biclusters():
    # 生成数据集 X、行索引 rows 和列索引 cols
    X, rows, cols = make_biclusters(
        shape=(100, 100), n_clusters=4, shuffle=True, random_state=0
    )
    # 断言数据集 X 的形状为 (100, 100)，如果不匹配，输出错误信息 "X shape mismatch"
    assert X.shape == (100, 100), "X shape mismatch"
    # 断言行索引 rows 的形状为 (4, 100)，如果不匹配，输出错误信息 "rows shape mismatch"
    assert rows.shape == (4, 100), "rows shape mismatch"
    # 断言列索引 cols 的形状为 (4, 100)，如果不匹配，输出错误信息 "columns shape mismatch"
    assert cols.shape == (
        4,
        100,
    ), "columns shape mismatch"
    # 断言 X、rows 和 cols 的所有元素均为有限数值
    assert_all_finite(X)
    assert_all_finite(rows)
    assert_all_finite(cols)

    # 重新生成数据集 X2，与之前的 X 进行近似相等的断言
    X2, _, _ = make_biclusters(
        shape=(100, 100), n_clusters=4, shuffle=True, random_state=0
    )
    assert_array_almost_equal(X, X2)


# 测试 make_checkerboard 函数
def test_make_checkerboard():
    # 生成数据集 X、行索引 rows 和列索引 cols
    X, rows, cols = make_checkerboard(
        shape=(100, 100), n_clusters=(20, 5), shuffle=True, random_state=0
    )
    # 断言数据集 X 的形状为 (100, 100)，如果不匹配，输出错误信息 "X shape mismatch"
    assert X.shape == (100, 100), "X shape mismatch"
    # 断言行索引 rows 的形状为 (100, 100)，如果不匹配，输出错误信息 "rows shape mismatch"
    assert rows.shape == (100, 100), "rows shape mismatch"
    # 断言列索引 cols 的形状为 (100, 100)，如果不匹配，输出错误信息 "columns shape mismatch"
    assert cols.shape == (
        100,
        100,
    ), "columns shape mismatch"

    # 生成数据集 X、行索引 rows 和列索引 cols
    X, rows, cols = make_checkerboard(
        shape=(100, 100), n_clusters=2, shuffle=True, random_state=0
    )
    # 断言 X、rows 和 cols 的所有元素均为有限数值
    assert_all_finite(X)
    assert_all_finite(rows)
    assert_all_finite(cols)

    # 重新生成数据集 X1 和 X2，断言它们近似相等
    X1, _, _ = make_checkerboard(
        shape=(100, 100), n_clusters=2, shuffle=True, random_state=0
    )
    X2, _, _ = make_checkerboard(
        shape=(100, 100), n_clusters=2, shuffle=True, random_state=0
    )
    assert_array_almost_equal(X1, X2)


# 测试 make_moons 函数
def test_make_moons():
    # 生成数据集 X 和标签 y
    X, y = make_moons(3, shuffle=False)
    # 对每个样本进行遍历，验证其是否位于预期的单位圆上
    for x, label in zip(X, y):
        center = [0.0, 0.0] if label == 0 else [1.0, 0.5]
        dist_sqr = ((x - center) ** 2).sum()
        assert_almost_equal(
            dist_sqr, 1.0, err_msg="Point is not on expected unit circle"
        )


# 测试 make_moons 函数中不平衡样本情况
def test_make_moons_unbalanced():
    # 生成数据集 X 和标签 y
    X, y = make_moons(n_samples=(7, 5))
    # 断言标签 y 中标签 0 的数量为 7，标签 1 的数量为 5，否则输出错误信息 "Number of samples in a moon is wrong"
    assert (
        np.sum(y == 0) == 7 and np.sum(y == 1) == 5
    ), "Number of samples in a moon is wrong"
    # 断言数据集 X 的形状为 (12, 2)，如果不匹配，输出错误信息 "X shape mismatch"
    assert X.shape == (12, 2), "X shape mismatch"
    # 断言标签 y 的形状为 (12)，如果不匹配，输出错误信息 "y shape mismatch"
    assert y.shape == (12,), "y shape mismatch"

    # 使用 pytest.raises 捕获 ValueError 异常，断言异常信息是否匹配指定正则表达式
    with pytest.raises(
        ValueError,
        match=r"`n_samples` can be either an int " r"or a two-element tuple.",
    ):
        make_moons(n_samples=(10,))


# 测试 make_circles 函数，该函数未完成，待续...
def test_make_circles():
    factor = 0.3
    # 对于每组样本数量 (n_samples, n_outer, n_inner)，分别为 (7, 3, 4) 和 (8, 4, 4)
    for n_samples, n_outer, n_inner in [(7, 3, 4), (8, 4, 4)]:
        # 进行奇数和偶数情况的测试，因为以前 make_circles 函数总是创建偶数个样本。
        X, y = make_circles(n_samples, shuffle=False, noise=None, factor=factor)
        # 断言验证 X 的形状是否为 (n_samples, 2)，确保 X 的形状匹配
        assert X.shape == (n_samples, 2), "X shape mismatch"
        # 断言验证 y 的形状是否为 (n_samples,)，确保 y 的形状匹配
        assert y.shape == (n_samples,), "y shape mismatch"
        # 设置圆心坐标为 [0.0, 0.0]
        center = [0.0, 0.0]
        # 遍历 X 和对应的标签 y
        for x, label in zip(X, y):
            # 计算点到圆心的距离的平方
            dist_sqr = ((x - center) ** 2).sum()
            # 如果标签为 0，则预期距离为 1.0；否则为 factor 的平方
            dist_exp = 1.0 if label == 0 else factor**2
            # 断言验证计算的距离平方与预期值是否相符
            assert_almost_equal(
                dist_sqr, dist_exp, err_msg="Point is not on expected circle"
            )

        # 断言验证标签为 0 的样本在 X 中的形状是否为 (n_outer, 2)，确保样本分布在圆周外圈
        assert X[y == 0].shape == (
            n_outer,
            2,
        ), "Samples not correctly distributed across circles."
        # 断言验证标签为 1 的样本在 X 中的形状是否为 (n_inner, 2)，确保样本分布在圆周内圈
        assert X[y == 1].shape == (
            n_inner,
            2,
        ), "Samples not correctly distributed across circles."
# 定义一个测试函数，用于测试 make_circles 函数生成不平衡数据集的情况
def test_make_circles_unbalanced():
    # 调用 make_circles 函数生成数据集 X 和标签 y，其中内圈样本数为2，外圈样本数为8
    X, y = make_circles(n_samples=(2, 8))

    # 断言确保内圈的样本数为2
    assert np.sum(y == 0) == 2, "Number of samples in inner circle is wrong"
    
    # 断言确保外圈的样本数为8
    assert np.sum(y == 1) == 8, "Number of samples in outer circle is wrong"
    
    # 断言确保数据集 X 的形状为 (10, 2)
    assert X.shape == (10, 2), "X shape mismatch"
    
    # 断言确保标签 y 的形状为 (10,)
    assert y.shape == (10,), "y shape mismatch"

    # 使用 pytest 模块的 raises 函数，断言 make_circles 函数在传入 n_samples 参数为 (10,) 时抛出 ValueError 异常，并且异常信息包含特定字符串
    with pytest.raises(
        ValueError,
        match="When a tuple, n_samples must have exactly two elements.",
    ):
        make_circles(n_samples=(10,))
```