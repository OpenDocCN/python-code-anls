# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_mean_shift.py`

```
"""
Testing for mean shift clustering methods

"""

import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift  # 从sklearn.cluster导入MeanShift聚类算法相关函数
from sklearn.datasets import make_blobs  # 从sklearn.datasets导入make_blobs用于生成数据集
from sklearn.metrics import v_measure_score  # 导入v_measure_score评估指标
from sklearn.utils._testing import assert_allclose, assert_array_equal  # 导入测试工具函数

n_clusters = 3  # 设定聚类数目为3
centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10  # 定义聚类中心并偏移
X, _ = make_blobs(
    n_samples=300,  # 生成300个样本
    n_features=2,   # 每个样本有2个特征
    centers=centers,  # 使用定义的中心进行聚类
    cluster_std=0.4,  # 聚类标准差为0.4
    shuffle=True,    # 打乱样本顺序
    random_state=11,  # 随机数种子
)


def test_convergence_of_1d_constant_data():
    # Test convergence using 1D constant data
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/28926
    model = MeanShift()  # 创建MeanShift聚类模型对象
    n_iter = model.fit(np.ones(10).reshape(-1, 1)).n_iter_  # 使用全为1的数据拟合模型并获取迭代次数
    assert n_iter < model.max_iter  # 断言迭代次数小于最大迭代次数


def test_estimate_bandwidth():
    # Test estimate_bandwidth
    bandwidth = estimate_bandwidth(X, n_samples=200)  # 使用数据集X进行带宽估计
    assert 0.9 <= bandwidth <= 1.5  # 断言带宽在0.9到1.5之间


def test_estimate_bandwidth_1sample(global_dtype):
    # Test estimate_bandwidth when n_samples=1 and quantile<1, so that
    # n_neighbors is set to 1.
    bandwidth = estimate_bandwidth(
        X.astype(global_dtype, copy=False), n_samples=1, quantile=0.3
    )  # 使用指定的数据类型和参数进行带宽估计

    assert bandwidth.dtype == X.dtype  # 断言带宽数据类型与X数据类型相同
    assert bandwidth == pytest.approx(0.0, abs=1e-5)  # 断言带宽接近于0.0，精度为1e-5


@pytest.mark.parametrize(
    "bandwidth, cluster_all, expected, first_cluster_label",
    [(1.2, True, 3, 0), (1.2, False, 4, -1)],
)
def test_mean_shift(
    global_dtype, bandwidth, cluster_all, expected, first_cluster_label
):
    # Test MeanShift algorithm
    X_with_global_dtype = X.astype(global_dtype, copy=False)  # 使用全局数据类型对X进行类型转换
    ms = MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)  # 创建MeanShift聚类模型对象
    labels = ms.fit(X_with_global_dtype).labels_  # 对数据进行拟合并获取标签
    labels_unique = np.unique(labels)  # 获取唯一标签
    n_clusters_ = len(labels_unique)  # 计算聚类数目
    assert n_clusters_ == expected  # 断言聚类数目符合预期
    assert labels_unique[0] == first_cluster_label  # 断言第一个聚类标签符合预期
    assert ms.cluster_centers_.dtype == global_dtype  # 断言聚类中心数据类型与全局数据类型一致

    cluster_centers, labels_mean_shift = mean_shift(
        X_with_global_dtype, cluster_all=cluster_all
    )  # 使用mean_shift函数进行聚类
    labels_mean_shift_unique = np.unique(labels_mean_shift)  # 获取唯一mean_shift标签
    n_clusters_mean_shift = len(labels_mean_shift_unique)  # 计算mean_shift聚类数目
    assert n_clusters_mean_shift == expected  # 断言mean_shift聚类数目符合预期
    assert labels_mean_shift_unique[0] == first_cluster_label  # 断言mean_shift第一个聚类标签符合预期
    assert cluster_centers.dtype == global_dtype  # 断言mean_shift聚类中心数据类型与全局数据类型一致


def test_parallel(global_dtype):
    centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10  # 定义并偏移聚类中心
    X, _ = make_blobs(
        n_samples=50,  # 生成50个样本
        n_features=2,  # 每个样本有2个特征
        centers=centers,  # 使用定义的中心进行聚类
        cluster_std=0.4,  # 聚类标准差为0.4
        shuffle=True,    # 打乱样本顺序
        random_state=11,  # 随机数种子
    )

    X = X.astype(global_dtype, copy=False)  # 使用全局数据类型对X进行类型转换

    ms1 = MeanShift(n_jobs=2)  # 创建允许并行计算的MeanShift对象
    ms1.fit(X)  # 对数据进行拟合

    ms2 = MeanShift()  # 创建默认MeanShift对象
    ms2.fit(X)  # 对数据进行拟合

    assert_allclose(ms1.cluster_centers_, ms2.cluster_centers_)  # 断言两个模型的聚类中心相近
    assert ms1.cluster_centers_.dtype == ms2.cluster_centers_.dtype  # 断言两个模型的聚类中心数据类型相同
    assert_array_equal(ms1.labels_, ms2.labels_)  # 断言两个模型的标签数组相等
# 测试 MeanShift 类中的 predict 方法
def test_meanshift_predict(global_dtype):
    # 创建 MeanShift 对象，设定带宽为 1.2
    ms = MeanShift(bandwidth=1.2)
    # 将输入数据 X 转换为指定的全局数据类型
    X_with_global_dtype = X.astype(global_dtype, copy=False)
    # 使用转换后的数据进行拟合并预测类标签
    labels = ms.fit_predict(X_with_global_dtype)
    # 使用相同数据进行预测类标签
    labels2 = ms.predict(X_with_global_dtype)
    # 断言两种方式预测的类标签数组相等
    assert_array_equal(labels, labels2)


# 测试 MeanShift 类中处理所有孤立点的情况
def test_meanshift_all_orphans():
    # 创建 MeanShift 对象，设定带宽为 0.1，并初始化种子点远离数据点
    ms = MeanShift(bandwidth=0.1, seeds=[[-9, -9], [-10, -10]])
    # 预期的错误信息
    msg = "No point was within bandwidth=0.1"
    # 使用 pytest 检测是否会抛出预期的 ValueError 异常，并包含特定的错误信息
    with pytest.raises(ValueError, match=msg):
        ms.fit(
            X,
        )


# 测试未拟合的情况
def test_unfitted():
    # 创建 MeanShift 对象
    ms = MeanShift()
    # 断言该对象没有属性 "cluster_centers_"
    assert not hasattr(ms, "cluster_centers_")
    # 断言该对象没有属性 "labels_"
    assert not hasattr(ms, "labels_")


# 测试簇强度相等的情况
def test_cluster_intensity_tie(global_dtype):
    # 创建包含 6 个数据点的数组 X，并指定全局数据类型
    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]], dtype=global_dtype)
    # 使用带宽为 2 的 MeanShift 对象进行拟合
    c1 = MeanShift(bandwidth=2).fit(X)

    # 重新设置 X，并使用带宽为 2 的 MeanShift 对象进行拟合
    X = np.array([[4, 7], [3, 5], [3, 6], [1, 1], [2, 1], [1, 0]], dtype=global_dtype)
    c2 = MeanShift(bandwidth=2).fit(X)

    # 断言两次拟合的类标签数组相等
    assert_array_equal(c1.labels_, [1, 1, 1, 0, 0, 0])
    assert_array_equal(c2.labels_, [0, 0, 0, 1, 1, 1])


# 测试使用 bin seeding 技术的情况
def test_bin_seeds(global_dtype):
    # 测试数据 X 包含 6 个二维平面点
    X = np.array(
        [[1.0, 1.0], [1.4, 1.4], [1.8, 1.2], [2.0, 1.0], [2.1, 1.1], [0.0, 0.0]],
        dtype=global_dtype,
    )

    # 预期的地面真值是三个 bin 的种子点
    ground_truth = {(1.0, 1.0), (2.0, 1.0), (0.0, 0.0)}
    # 调用 get_bin_seeds 函数进行测试，bin coarseness 为 1.0，min_bin_freq 为 1
    test_bins = get_bin_seeds(X, 1, 1)
    test_result = set(tuple(p) for p in test_bins)
    # 断言测试结果与预期结果完全一致
    assert len(ground_truth.symmetric_difference(test_result)) == 0

    # 重新设置 ground_truth 和测试条件，bin coarseness 为 1.0，min_bin_freq 为 2
    ground_truth = {(1.0, 1.0), (2.0, 1.0)}
    test_bins = get_bin_seeds(X, 1, 2)
    test_result = set(tuple(p) for p in test_bins)
    # 断言测试结果与预期结果完全一致
    assert len(ground_truth.symmetric_difference(test_result)) == 0

    # 重新设置测试条件，bin 大小为 0.01，min_bin_freq 为 1
    # 这种情况下会产生警告，并使用全部数据
    with warnings.catch_warnings(record=True):
        test_bins = get_bin_seeds(X, 0.01, 1)
    # 断言测试结果与原始数据 X 完全一致
    assert_allclose(test_bins, X)

    # 创建密集聚类围绕 [0, 0] 和 [1, 1] 的情况，预期只能得到两个 bin
    X, _ = make_blobs(
        n_samples=100,
        n_features=2,
        centers=[[0, 0], [1, 1]],
        cluster_std=0.1,
        random_state=0,
    )
    X = X.astype(global_dtype, copy=False)
    test_bins = get_bin_seeds(X, 1)
    # 断言测试结果与预期的聚类中心完全一致
    assert_array_equal(test_bins, [[0, 0], [1, 1]])


# 使用参数化测试检查 max_iter 参数的情况
@pytest.mark.parametrize("max_iter", [1, 100])
def test_max_iter(max_iter):
    # 调用 mean_shift 函数得到 clusters1 和 _，并使用 max_iter 参数
    clusters1, _ = mean_shift(X, max_iter=max_iter)
    # 创建 MeanShift 对象，使用 max_iter 参数进行拟合
    ms = MeanShift(max_iter=max_iter).fit(X)
    # 获取 MeanShift 对象的聚类中心
    clusters2 = ms.cluster_centers_

    # 断言 MeanShift 对象的实际迭代次数不超过设定的最大迭代次数
    assert ms.n_iter_ <= ms.max_iter
    # 断言 clusters1 和 clusters2 的长度相等
    assert len(clusters1) == len(clusters2)
    # 使用 zip 函数同时迭代 clusters1 和 clusters2 中的元素 c1 和 c2
    for c1, c2 in zip(clusters1, clusters2):
        # 断言：检查 c1 和 c2 是否在数值上全部接近（近似相等）
        assert np.allclose(c1, c2)
def test_mean_shift_zero_bandwidth(global_dtype):
    # 检查当估计带宽为0时，均值漂移算法是否能正常工作。
    X = np.array([1, 1, 1, 2, 2, 2, 3, 3], dtype=global_dtype).reshape(-1, 1)
    
    # 使用默认参数的 estimate_bandwidth 在这个数据集上返回0
    bandwidth = estimate_bandwidth(X)
    assert bandwidth == 0
    
    # 当 bin_size 为0时，get_bin_seeds 应返回数据集本身
    assert get_bin_seeds(X, bin_size=bandwidth) is X
    
    # 使用带有 binning 和估计带宽为0的 MeanShift 应该等效于没有 binning 的情况。
    ms_binning = MeanShift(bin_seeding=True, bandwidth=None).fit(X)
    ms_nobinning = MeanShift(bin_seeding=False).fit(X)
    expected_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    
    # 验证使用 binning 的 MeanShift 的标签与预期标签的相似度
    assert v_measure_score(ms_binning.labels_, expected_labels) == pytest.approx(1)
    # 验证不使用 binning 的 MeanShift 的标签与预期标签的相似度
    assert v_measure_score(ms_nobinning.labels_, expected_labels) == pytest.approx(1)
    # 验证使用 binning 和不使用 binning 的 MeanShift 的聚类中心是否接近
    assert_allclose(ms_binning.cluster_centers_, ms_nobinning.cluster_centers_)
```