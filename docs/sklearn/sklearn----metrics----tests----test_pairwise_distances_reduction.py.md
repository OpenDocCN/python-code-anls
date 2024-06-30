# `D:\src\scipysrc\scikit-learn\sklearn\metrics\tests\test_pairwise_distances_reduction.py`

```
import itertools
import re
import warnings
from functools import partial

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
    ArgKmin,
    ArgKminClassMode,
    BaseDistancesReductionDispatcher,
    RadiusNeighbors,
    RadiusNeighborsClassMode,
    sqeuclidean_row_norms,
)
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
    create_memmap_backed_data,
)
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.parallel import _get_threadpool_controller

# Common supported metric between scipy.spatial.distance.cdist
# and BaseDistanceReductionDispatcher.
# This allows constructing tests to check consistency of results
# of concrete BaseDistanceReductionDispatcher on some metrics using APIs
# from scipy and numpy.
CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "euclidean",
    "minkowski",
    "seuclidean",
]

def _get_metric_params_list(metric: str, n_features: int, seed: int = 1):
    """Return list of dummy DistanceMetric kwargs for tests."""
    
    # Distinguishing on cases not to compute unneeded datastructures.
    rng = np.random.RandomState(seed)

    if metric == "minkowski":
        # Define specific kwargs for different cases of the Minkowski metric.
        minkowski_kwargs = [
            dict(p=1.5),
            dict(p=2),
            dict(p=3),
            dict(p=np.inf),
            dict(p=3, w=rng.rand(n_features)),
        ]
        
        return minkowski_kwargs

    if metric == "seuclidean":
        # Return kwargs for the standardized Euclidean metric.
        return [dict(V=rng.rand(n_features))]

    # For other metrics like "euclidean", "manhattan", "chebyshev", "haversine", or any other metric,
    # no specific kwargs are needed.
    return [{}]

def assert_same_distances_for_common_neighbors(
    query_idx,
    dist_row_a,
    dist_row_b,
    indices_row_a,
    indices_row_b,
    rtol,
    atol,
):
    """Check that the distances of common neighbors are equal up to tolerance.

    This does not check if there are missing neighbors in either result set.
    Missingness is handled by assert_no_missing_neighbors.
    """
    # Compute a mapping from indices to distances for each result set and
    # check that the computed neighbors with matching indices are within
    # the expected distance tolerance.
    indices_to_dist_a = dict(zip(indices_row_a, dist_row_a))
    indices_to_dist_b = dict(zip(indices_row_b, dist_row_b))

    # Find common indices between indices_row_a and indices_row_b.
    common_indices = set(indices_row_a).intersection(set(indices_row_b))
    # 对于每个在common_indices中的索引idx，执行以下操作
    for idx in common_indices:
        # 从indices_to_dist_a和indices_to_dist_b中获取距离值dist_a和dist_b
        dist_a = indices_to_dist_a[idx]
        dist_b = indices_to_dist_b[idx]
        try:
            # 使用assert_allclose函数断言dist_a和dist_b的接近程度，给定rtol和atol作为容差
            assert_allclose(dist_a, dist_b, rtol=rtol, atol=atol)
        except AssertionError as e:
            # 捕获AssertionError异常，并包装以提供更多上下文信息，同时包括原始异常和计算出的绝对和相对差异
            raise AssertionError(
                f"Query vector with index {query_idx} lead to different distances"
                f" for common neighbor with index {idx}:"
                f" dist_a={dist_a} vs dist_b={dist_b} (with atol={atol} and"
                f" rtol={rtol})"
            ) from e
# 比较两个结果集中邻居索引的匹配情况，确保距离低于精度阈值的邻居索引在另一个结果集中有对应匹配。
def assert_no_missing_neighbors(
    query_idx,
    dist_row_a,
    dist_row_b,
    indices_row_a,
    indices_row_b,
    threshold,
):
    """Compare the indices of neighbors in two results sets.

    Any neighbor index with a distance below the precision threshold should
    match one in the other result set. We ignore the last few neighbors beyond
    the threshold as those can typically be missing due to rounding errors.

    For radius queries, the threshold is just the radius minus the expected
    precision level.

    For k-NN queries, it is the maximum distance to the k-th neighbor minus the
    expected precision level.
    """
    # 创建布尔掩码，标识距离低于阈值的邻居
    mask_a = dist_row_a < threshold
    mask_b = dist_row_b < threshold
    # 找出在indices_row_a[mask_a]中而不在indices_row_b中的邻居索引
    missing_from_b = np.setdiff1d(indices_row_a[mask_a], indices_row_b)
    # 找出在indices_row_b[mask_b]中而不在indices_row_a中的邻居索引
    missing_from_a = np.setdiff1d(indices_row_b[mask_b], indices_row_a)
    # 如果存在不匹配的邻居索引，抛出断言错误
    if len(missing_from_a) > 0 or len(missing_from_b) > 0:
        raise AssertionError(
            f"Query vector with index {query_idx} lead to mismatched result indices:\n"
            f"neighbors in b missing from a: {missing_from_a}\n"
            f"neighbors in a missing from b: {missing_from_b}\n"
            f"dist_row_a={dist_row_a}\n"
            f"dist_row_b={dist_row_b}\n"
            f"indices_row_a={indices_row_a}\n"
            f"indices_row_b={indices_row_b}\n"
        )


# 确保argkmin结果在舍入误差范围内有效
def assert_compatible_argkmin_results(
    neighbors_dists_a,
    neighbors_dists_b,
    neighbors_indices_a,
    neighbors_indices_b,
    rtol=1e-5,
    atol=1e-6,
):
    """Assert that argkmin results are valid up to rounding errors.

    This function asserts that the results of argkmin queries are valid up to:
    - rounding error tolerance on distance values;
    - permutations of indices for distances values that differ up to the
      expected precision level.

    Furthermore, the distances must be sorted.

    To be used for testing neighbors queries on float32 datasets: we accept
    neighbors rank swaps only if they are caused by small rounding errors on
    the distance computations.
    """
    # Lambda函数检查数组是否已排序
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    # 断言所有输入数组的形状一致
    assert (
        neighbors_dists_a.shape
        == neighbors_dists_b.shape
        == neighbors_indices_a.shape
        == neighbors_indices_b.shape
    ), "Arrays of results have incompatible shapes."

    # 获取查询数目和邻居数量
    n_queries, _ = neighbors_dists_a.shape

    # 逐行比较结果的相等性
    # 对每个查询索引循环，逐个处理邻居距离和索引
    for query_idx in range(n_queries):
        # 从neighbors_dists_a中获取当前查询索引的距离行
        dist_row_a = neighbors_dists_a[query_idx]
        # 从neighbors_dists_b中获取当前查询索引的距离行
        dist_row_b = neighbors_dists_b[query_idx]
        # 从neighbors_indices_a中获取当前查询索引的索引行
        indices_row_a = neighbors_indices_a[query_idx]
        # 从neighbors_indices_b中获取当前查询索引的索引行
        indices_row_b = neighbors_indices_b[query_idx]

        # 断言当前查询索引的距离行是有序的，否则抛出异常
        assert is_sorted(dist_row_a), f"Distances aren't sorted on row {query_idx}"
        assert is_sorted(dist_row_b), f"Distances aren't sorted on row {query_idx}"

        # 断言对于共同邻居，它们的距离应该一致，否则抛出异常
        assert_same_distances_for_common_neighbors(
            query_idx,
            dist_row_a,
            dist_row_b,
            indices_row_a,
            indices_row_b,
            rtol,
            atol,
        )

        # 检查距离低于舍入误差阈值的邻居是否具有匹配的索引
        # 阈值由以下公式确定：
        # (1 - rtol) * dist_k - atol
        # 其中dist_k定义为两个结果集中第k个邻居的最大距离。
        # 这种定义阈值的方式比简单取两者最小值更严格。
        threshold = (1 - rtol) * np.maximum(
            np.max(dist_row_a), np.max(dist_row_b)
        ) - atol
        # 断言在距离低于阈值的情况下，没有缺失的邻居索引
        assert_no_missing_neighbors(
            query_idx,
            dist_row_a,
            dist_row_b,
            indices_row_a,
            indices_row_b,
            threshold,
        )
# 定义一个函数用于计算非平凡半径，
# 只接受关键字参数，并采用以下参数：
# X: 第一个数据集
# Y: 第二个数据集
# metric: 距离度量方法
# precomputed_dists: 预先计算的距离矩阵
# expected_n_neighbors: 期望的邻居数量平均值
# n_subsampled_queries: 子采样查询的数量
# **metric_kwargs: 其他传递给距离度量函数的参数
def _non_trivial_radius(
    *,
    X=None,
    Y=None,
    metric=None,
    precomputed_dists=None,
    expected_n_neighbors=10,
    n_subsampled_queries=10,
    **metric_kwargs,
):
    # 使用 X 和 Y 的部分配对距离的子样本来找到一个非平凡的半径：
    # 我们希望平均返回大约 expected_n_neighbors 个邻居。
    # 如果返回的结果太多会使测试变慢（因为检查结果集合对于大的结果集合来说很昂贵），
    # 如果大部分时间返回 0 会使得测试无效。
    assert (
        precomputed_dists is not None or metric is not None
    ), "Either metric or precomputed_dists must be provided."

    if precomputed_dists is None:
        assert X is not None
        assert Y is not None
        # 计算 X 和 Y 之间的配对距离
        sampled_dists = pairwise_distances(X, Y, metric=metric, **metric_kwargs)
    else:
        # 使用预先计算的距离矩阵的子样本
        sampled_dists = precomputed_dists[:n_subsampled_queries].copy()
    # 对采样距离进行排序
    sampled_dists.sort(axis=1)
    # 返回期望邻居数量的平均距离
    return sampled_dists[:, expected_n_neighbors].mean()


def assert_compatible_radius_results(
    neighbors_dists_a,
    neighbors_dists_b,
    neighbors_indices_a,
    neighbors_indices_b,
    radius,
    check_sorted=True,
    rtol=1e-5,
    atol=1e-6,
):
    """Assert that radius neighborhood results are valid up to:

      - relative and absolute tolerance on computed distance values
      - permutations of indices for distances values that differ up to
        a precision level
      - missing or extra last elements if their distance is
        close to the radius

    To be used for testing neighbors queries on float32 datasets: we
    accept neighbors rank swaps only if they are caused by small
    rounding errors on the distance computations.

    Input arrays must be sorted w.r.t distances.
    """
    # 检查输入数组的长度是否一致
    assert (
        len(neighbors_dists_a)
        == len(neighbors_dists_b)
        == len(neighbors_indices_a)
        == len(neighbors_indices_b)
    )

    n_queries = len(neighbors_dists_a)

    # 定义一个函数检查数组是否已经排序
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    # 断言结果的一致性，逐个向量进行比较
    `
        # 遍历查询索引范围内的所有查询
        for query_idx in range(n_queries):
            # 获取查询索引对应的邻居距离数据
            dist_row_a = neighbors_dists_a[query_idx]
            # 获取查询索引对应的邻居距离数据
            dist_row_b = neighbors_dists_b[query_idx]
            # 获取查询索引对应的邻居索引数据
            indices_row_a = neighbors_indices_a[query_idx]
            # 获取查询索引对应的邻居索引数据
            indices_row_b = neighbors_indices_b[query_idx]
    
            # 如果需要检查距离是否已排序，则进行断言检查
            if check_sorted:
                assert is_sorted(dist_row_a), f"Distances aren't sorted on row {query_idx}"
                assert is_sorted(dist_row_b), f"Distances aren't sorted on row {query_idx}"
    
            # 断言检查，确保距离数据长度等于索引数据长度
            assert len(dist_row_a) == len(indices_row_a)
            assert len(dist_row_b) == len(indices_row_b)
    
            # 检查所有距离是否在请求的半径内
            if len(dist_row_a) > 0:
                max_dist_a = np.max(dist_row_a)
                # 断言检查，确保最大距离在请求的半径内
                assert max_dist_a <= radius, (
                    f"Largest returned distance {max_dist_a} not within requested"
                    f" radius {radius} on row {query_idx}"
                )
            if len(dist_row_b) > 0:
                max_dist_b = np.max(dist_row_b)
                # 断言检查，确保最大距离在请求的半径内
                assert max_dist_b <= radius, (
                    f"Largest returned distance {max_dist_b} not within requested"
                    f" radius {radius} on row {query_idx}"
                )
    
            # 验证对于公共邻居的距离是否一致
            assert_same_distances_for_common_neighbors(
                query_idx,
                dist_row_a,
                dist_row_b,
                indices_row_a,
                indices_row_b,
                rtol,
                atol,
            )
    
            # 计算阈值，依据相对容忍度和绝对容忍度调整
            threshold = (1 - rtol) * radius - atol
            # 断言检查，确保没有缺失邻居
            assert_no_missing_neighbors(
                query_idx,
                dist_row_a,
                dist_row_b,
                indices_row_a,
                indices_row_b,
                threshold,
            )
# 定义浮点数比较的容差字典，包括绝对误差容限和相对误差容限
FLOAT32_TOLS = {
    "atol": 1e-7,
    "rtol": 1e-5,
}

# 定义另一个浮点数比较的容差字典，具有更严格的绝对误差容限和相对误差容限
FLOAT64_TOLS = {
    "atol": 1e-9,
    "rtol": 1e-7,
}

# 定义断言结果的字典，包含不同函数和数据类型的部分应用函数
ASSERT_RESULT = {
    # 对于 (ArgKmin, np.float64) 类型，使用特定的容差字典进行部分应用
    (ArgKmin, np.float64): partial(assert_compatible_argkmin_results, **FLOAT64_TOLS),
    # 对于 (ArgKmin, np.float32) 类型，使用特定的容差字典进行部分应用
    (ArgKmin, np.float32): partial(assert_compatible_argkmin_results, **FLOAT32_TOLS),
    # 对于 (RadiusNeighbors, np.float64) 类型，使用特定的容差字典进行部分应用
    (RadiusNeighbors, np.float64): partial(assert_compatible_radius_results, **FLOAT64_TOLS),
    # 对于 (RadiusNeighbors, np.float32) 类型，使用特定的容差字典进行部分应用
    (RadiusNeighbors, np.float32): partial(assert_compatible_radius_results, **FLOAT32_TOLS),
}


def test_assert_compatible_argkmin_results():
    # 设置绝对误差容限和相对误差容限的字典
    atol = 1e-7
    rtol = 0.0
    tols = dict(atol=atol, rtol=rtol)

    # 计算用于比较的误差值
    eps = atol / 3
    _1m = 1.0 - eps
    _1p = 1.0 + eps

    _6_1m = 6.1 - eps
    _6_1p = 6.1 + eps

    # 定义参考距离数组
    ref_dist = np.array([
        [1.2, 2.5, _6_1m, 6.1, _6_1p],
        [_1m, _1m, 1, _1p, _1p],
    ])
    
    # 定义参考索引数组
    ref_indices = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ])

    # 断言：比较参考结果与自身的兼容性
    assert_compatible_argkmin_results(ref_dist, ref_dist, ref_indices, ref_indices, rtol)

    # 应用有效的索引排列：最后三个点非常接近，因此可以接受它们在排名上的任何排列
    assert_compatible_argkmin_results(
        np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
        np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
        np.array([[1, 2, 3, 4, 5]]),
        np.array([[1, 2, 5, 4, 3]]),
        **tols,
    )

    # 由于距离的舍入误差，最后几个索引不一定要匹配：可能在边界上存在并列的结果
    assert_compatible_argkmin_results(
        np.array([[1.2, 2.5, 3.0, 6.1, _6_1p]]),
        np.array([[1.2, 2.5, 3.0, _6_1m, 6.1]]),
        np.array([[1, 2, 3, 4, 5]]),
        np.array([[1, 2, 3, 6, 7]]),
        **tols,
    )

    # 所有点的距离非常接近，因此查询结果的任何排列都是有效的
    assert_compatible_argkmin_results(
        np.array([[_1m, 1, _1p, _1p, _1p]]),
        np.array([[1, 1, 1, 1, _1p]]),
        np.array([[7, 6, 8, 10, 9]]),
        np.array([[6, 9, 7, 8, 10]]),
        **tols,
    )

    # 也可能是非常大且几乎相同结果集的近似截断，因此在这种情况下所有索引也可以是不同的：
    assert_compatible_argkmin_results(
        np.array([[_1m, 1, _1p, _1p, _1p]]),
        np.array([[_1m, 1, 1, 1, _1p]]),
        np.array([[34, 30, 8, 12, 24]]),
        np.array([[42, 1, 21, 13, 3]]),
        **tols,
    )

    # 应用无效的索引排列：对最近的两个邻居的排列无效，因为它们的距离值差异太大
    msg = re.escape(
        "Query vector with index 0 lead to different distances for common neighbor with"
        " index 1: dist_a=1.2 vs dist_b=2.5"
    )
    # 使用 pytest 检测是否会引发 AssertionError，并检查其匹配给定的错误消息
    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_compatible_argkmin_results 函数，验证是否符合预期结果
        assert_compatible_argkmin_results(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),  # 提供输入数组 a
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),  # 提供输入数组 b
            np.array([[1, 2, 3, 4, 5]]),  # 提供预期输出的最小索引数组 expected_kmin_a
            np.array([[2, 1, 3, 4, 5]]),  # 提供预期输出的最小索引数组 expected_kmin_b
            **tols,  # 传递额外的参数
        )

    # 检测预期精度级别内的缺失索引，即使距离完全匹配。
    msg = re.escape(
        "neighbors in b missing from a: [12]\nneighbors in a missing from b: [1]"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),  # 提供输入数组 a
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),  # 提供输入数组 b
            np.array([[1, 2, 3, 4, 5]]),  # 提供预期输出的最小索引数组 expected_kmin_a
            np.array([[12, 2, 4, 11, 3]]),  # 提供预期输出的最小索引数组 expected_kmin_b
            **tols,  # 传递额外的参数
        )

    # 检测预期精度级别外的缺失索引。
    msg = re.escape(
        "neighbors in b missing from a: []\nneighbors in a missing from b: [3]"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[_1m, 1.0, _6_1m, 6.1, _6_1p]]),  # 提供输入数组 a
            np.array([[1.0, 1.0, _6_1m, 6.1, 7]]),  # 提供输入数组 b
            np.array([[1, 2, 3, 4, 5]]),  # 提供预期输出的最小索引数组 expected_kmin_a
            np.array([[2, 1, 4, 5, 12]]),  # 提供预期输出的最小索引数组 expected_kmin_b
            **tols,  # 传递额外的参数
        )

    # 检测预期精度级别外的缺失索引，在另一个方向上。
    msg = re.escape(
        "neighbors in b missing from a: [5]\nneighbors in a missing from b: []"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[_1m, 1.0, _6_1m, 6.1, 7]]),  # 提供输入数组 a
            np.array([[1.0, 1.0, _6_1m, 6.1, _6_1p]]),  # 提供输入数组 b
            np.array([[1, 2, 3, 4, 12]]),  # 提供预期输出的最小索引数组 expected_kmin_a
            np.array([[2, 1, 5, 3, 4]]),  # 提供预期输出的最小索引数组 expected_kmin_b
            **tols,  # 传递额外的参数
        )

    # 检测距离在第一行上没有正确排序
    msg = "Distances aren't sorted on row 0"
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),  # 提供输入数组 a
            np.array([[2.5, 1.2, _6_1m, 6.1, _6_1p]]),  # 提供输入数组 b
            np.array([[1, 2, 3, 4, 5]]),  # 提供预期输出的最小索引数组 expected_kmin_a
            np.array([[2, 1, 4, 5, 3]]),  # 提供预期输出的最小索引数组 expected_kmin_b
            **tols,  # 传递额外的参数
        )
@pytest.mark.parametrize("check_sorted", [True, False])
def test_assert_compatible_radius_results(check_sorted):
    atol = 1e-7  # 定义绝对误差容限
    rtol = 0.0   # 定义相对误差容限
    tols = dict(atol=atol, rtol=rtol)  # 将误差容限组织为字典

    eps = atol / 3  # 计算误差容限的三分之一
    _1m = 1.0 - eps  # 计算 1.0 减去误差容限
    _1p = 1.0 + eps  # 计算 1.0 加上误差容限
    _6_1m = 6.1 - eps  # 计算 6.1 减去误差容限
    _6_1p = 6.1 + eps  # 计算 6.1 加上误差容限

    ref_dist = [  # 定义参考距离数组
        np.array([1.2, 2.5, _6_1m, 6.1, _6_1p]),  # 第一个参考距离向量
        np.array([_1m, 1, _1p, _1p]),            # 第二个参考距离向量
    ]

    ref_indices = [  # 定义参考索引数组
        np.array([1, 2, 3, 4, 5]),    # 第一个参考索引向量
        np.array([6, 7, 8, 9]),       # 第二个参考索引向量
    ]

    # Sanity check: compare the reference results to themselves.
    assert_compatible_radius_results(
        ref_dist,            # 参考距离数组
        ref_dist,            # 参考距离数组（再次使用）
        ref_indices,         # 参考索引数组
        ref_indices,         # 参考索引数组（再次使用）
        radius=7.0,          # 半径参数
        check_sorted=check_sorted,  # 检查是否已排序参数
        **tols,              # 传递误差容限参数
    )

    # Apply valid permutation on indices
    assert_compatible_radius_results(
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),   # 第一个查询向量
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),   # 第二个查询向量
        np.array([np.array([1, 2, 3, 4, 5])]),                 # 第一个查询索引
        np.array([np.array([1, 2, 4, 5, 3])]),                 # 第二个查询索引（有效排列）
        radius=7.0,          # 半径参数
        check_sorted=check_sorted,  # 检查是否已排序参数
        **tols,              # 传递误差容限参数
    )
    assert_compatible_radius_results(
        np.array([np.array([_1m, _1m, 1, _1p, _1p])]),         # 第一个查询向量
        np.array([np.array([_1m, _1m, 1, _1p, _1p])]),         # 第二个查询向量
        np.array([np.array([6, 7, 8, 9, 10])]),                # 第一个查询索引
        np.array([np.array([6, 9, 7, 8, 10])]),                # 第二个查询索引（有效排列）
        radius=7.0,          # 半径参数
        check_sorted=check_sorted,  # 检查是否已排序参数
        **tols,              # 传递误差容限参数
    )

    # Apply invalid permutation on indices
    msg = re.escape(
        "Query vector with index 0 lead to different distances for common neighbor with"
        " index 1: dist_a=1.2 vs dist_b=2.5"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),   # 第一个查询向量
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),   # 第二个查询向量
            np.array([np.array([1, 2, 3, 4, 5])]),                 # 第一个查询索引
            np.array([np.array([2, 1, 3, 4, 5])]),                 # 第二个查询索引（无效排列）
            radius=7.0,          # 半径参数
            check_sorted=check_sorted,  # 检查是否已排序参数
            **tols,              # 传递误差容限参数
        )

    # Having extra last or missing elements is valid if they are in the
    # tolerated rounding error range: [(1 - rtol) * radius - atol, radius]
    assert_compatible_radius_results(
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p, _6_1p])]),   # 第一个查询向量（多一个元素）
        np.array([np.array([1.2, 2.5, _6_1m, 6.1])]),                 # 第二个查询向量
        np.array([np.array([1, 2, 3, 4, 5, 7])]),                     # 第一个查询索引
        np.array([np.array([1, 2, 3, 6])]),                           # 第二个查询索引
        radius=_6_1p,        # 半径参数
        check_sorted=check_sorted,  # 检查是否已排序参数
        **tols,              # 传递误差容限参数
    )

    # Any discrepancy outside the tolerated rounding error range is invalid and
    # indicates a missing neighbor in one of the result sets.
    msg = re.escape(
        "Query vector with index 0 lead to mismatched result indices:\nneighbors in b"
        " missing from a: []\nneighbors in a missing from b: [3]"
    )
    # 使用 pytest 的上下文管理器来验证特定的异常（AssertionError），并且匹配特定的错误消息（msg）
    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_compatible_radius_results 函数，验证其行为
        assert_compatible_radius_results(
            # 提供第一个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1.2, 2.5, 6])]),
            # 提供第二个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1.2, 2.5])]),
            # 提供第三个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1, 2, 3])]),
            # 提供第四个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1, 2])]),
            # 提供关键字参数 radius，设置为 6.1
            radius=6.1,
            # 提供关键字参数 check_sorted，值为 check_sorted 变量的值
            check_sorted=check_sorted,
            # 传递额外的关键字参数 **tols
            **tols,
        )
    
    # 设置消息变量 msg，用于后续的异常匹配
    msg = re.escape(
        "Query vector with index 0 lead to mismatched result indices:\nneighbors in b"
        " missing from a: [4]\nneighbors in a missing from b: [2]"
    )
    
    # 使用 pytest 的上下文管理器来验证特定的异常（AssertionError），并且匹配特定的错误消息（msg）
    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_compatible_radius_results 函数，验证其行为
        assert_compatible_radius_results(
            # 提供第一个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1.2, 2.1, 2.5])]),
            # 提供第二个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1.2, 2, 2.5])]),
            # 提供第三个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1, 2, 3])]),
            # 提供第四个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1, 4, 3])]),
            # 提供关键字参数 radius，设置为 6.1
            radius=6.1,
            # 提供关键字参数 check_sorted，值为 check_sorted 变量的值
            check_sorted=check_sorted,
            # 传递额外的关键字参数 **tols
            **tols,
        )

    # 设置消息变量 msg，用于后续的异常匹配
    msg = re.escape(
        "Largest returned distance 6.100000033333333 not within requested radius 6.1 on"
        " row 0"
    )
    
    # 使用 pytest 的上下文管理器来验证特定的异常（AssertionError），并且匹配特定的错误消息（msg）
    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_compatible_radius_results 函数，验证其行为
        assert_compatible_radius_results(
            # 提供第一个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            # 提供第二个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, 6.1])]),
            # 提供第三个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1, 2, 3, 4, 5])]),
            # 提供第四个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([2, 1, 4, 5, 3])]),
            # 提供关键字参数 radius，设置为 6.1
            radius=6.1,
            # 提供关键字参数 check_sorted，值为 check_sorted 变量的值
            check_sorted=check_sorted,
            # 传递额外的关键字参数 **tols
            **tols,
        )
    
    # 使用 pytest 的上下文管理器来验证特定的异常（AssertionError），并且匹配特定的错误消息（msg）
    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_compatible_radius_results 函数，验证其行为
        assert_compatible_radius_results(
            # 提供第一个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, 6.1])]),
            # 提供第二个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            # 提供第三个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([1, 2, 3, 4, 5])]),
            # 提供第四个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
            np.array([np.array([2, 1, 4, 5, 3])]),
            # 提供关键字参数 radius，设置为 6.1
            radius=6.1,
            # 提供关键字参数 check_sorted，值为 check_sorted 变量的值
            check_sorted=check_sorted,
            # 传递额外的关键字参数 **tols
            **tols,
        )

    if check_sorted:
        # 如果 check_sorted 为真，则进入条件分支
        # 设置消息变量 msg，用于后续的异常匹配
        msg = "Distances aren't sorted on row 0"
        # 使用 pytest 的上下文管理器来验证特定的异常（AssertionError），并且匹配特定的错误消息（msg）
        with pytest.raises(AssertionError, match=msg):
            # 调用 assert_compatible_radius_results 函数，验证其行为
            assert_compatible_radius_results(
                # 提供第一个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
                np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
                # 提供第二个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
                np.array([np.array([2.5, 1.2, _6_1m, 6.1, _6_1p])]),
                # 提供第三个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
                np.array([np.array([1, 2, 3, 4, 5])]),
                # 提供第四个参数，一个包含单个元素的 NumPy 数组，该元素本身也是一个 NumPy 数组
                np.array([np.array([2, 1, 4, 5, 3])]),
                # 提供关键字参数 radius，设置为 _6_1p 变量的值
                radius=_
# 使用 pytest 模块的 parametrize 装饰器，参数化测试函数，对每个 csr_container 进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_distances_reduction_is_usable_for(csr_container):
    # 创建一个伪随机数生成器，种子为 0
    rng = np.random.RandomState(0)
    # 生成一个 100x10 的随机数组 X
    X = rng.rand(100, 10)
    # 生成一个 100x10 的随机数组 Y
    Y = rng.rand(100, 10)
    # 将 X 转换为 CSR 格式
    X_csr = csr_container(X)
    # 将 Y 转换为 CSR 格式
    Y_csr = csr_container(Y)
    # 设置距离度量方式为 "manhattan"

    # 必须适用于所有可能的 {dense, sparse} 数据集对的基础距离缩减调度器
    assert BaseDistancesReductionDispatcher.is_usable_for(X, Y, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y_csr, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X, Y_csr, metric)

    # 使用 np.float64 类型的 X 和 Y 进行距离计算
    assert BaseDistancesReductionDispatcher.is_usable_for(
        X.astype(np.float64), Y.astype(np.float64), metric
    )

    # 使用 np.float32 类型的 X 和 Y 进行距离计算
    assert BaseDistancesReductionDispatcher.is_usable_for(
        X.astype(np.float32), Y.astype(np.float32), metric
    )

    # 使用 np.int64 类型的 X 和 Y，不支持此类型的距离计算
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        X.astype(np.int64), Y.astype(np.int64), metric
    )

    # 不支持 metric="pyfunc" 的距离计算
    assert not BaseDistancesReductionDispatcher.is_usable_for(X, Y, metric="pyfunc")

    # 不支持 X.astype(np.float32) 的距离计算
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        X.astype(np.float32), Y, metric
    )

    # 不支持 Y.astype(np.int32) 的距离计算
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        X, Y.astype(np.int32), metric
    )

    # 不支持 F-order 数组的距离计算
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        np.asfortranarray(X), Y, metric
    )

    # 对于 metric="euclidean"，可以使用 CSR 格式的 X 和 Y 进行距离计算
    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y, metric="euclidean")

    # 对于 metric="sqeuclidean"，可以使用 CSR 格式的 X 和 Y 进行距离计算
    assert BaseDistancesReductionDispatcher.is_usable_for(
        X, Y_csr, metric="sqeuclidean"
    )

    # FIXME: 当前的 Cython 实现对于大量特征的数据集速度太慢，暂时禁用，使用 SciPy 实现
    # 参考：https://github.com/scikit-learn/scikit-learn/issues/28191
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        X_csr, Y_csr, metric="sqeuclidean"
    )
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        X_csr, Y_csr, metric="euclidean"
    )

    # 当前不支持没有非零元素的 CSR 矩阵
    # TODO: 支持没有非零元素的 CSR 矩阵
    X_csr_0_nnz = csr_container(X * 0)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr_0_nnz, Y, metric)

    # 目前不支持具有 int64 索引和 indptr 的 CSR 矩阵（例如大 nnz 或大 n_features）
    # 参考：https://github.com/scikit-learn/scikit-learn/issues/23653
    # TODO: 支持具有 int64 索引和 indptr 的 CSR 矩阵
    X_csr_int64 = csr_container(X)
    X_csr_int64.indices = X_csr_int64.indices.astype(np.int64)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr_int64, Y, metric)


# 测试 argkmin 工厂方法的错误用法
def test_argkmin_factory_method_wrong_usages():
    # 创建一个伪随机数生成器，种子为 1
    rng = np.random.RandomState(1)
    # 生成一个 100x10 的随机数组 X
    X = rng.rand(100, 10)
    # 生成一个 100x10 的随机数组 Y
    Y = rng.rand(100, 10)
    # 设置 k 值为 5
    k = 5
    # 设置默认的距离度量为欧氏距离
    metric = "euclidean"

    # 准备错误消息，用于在特定情况下引发值错误
    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float32 and Y.dtype=float64"
    )
    # 使用 pytest 来检查是否引发特定的值错误，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 ArgKmin 的 compute 方法，传入转换后的 X，指定的 Y，以及其他参数
        ArgKmin.compute(X=X.astype(np.float32), Y=Y, k=k, metric=metric)

    # 准备另一个错误消息，检查另一种情况的数据类型错误
    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float64 and Y.dtype=int32"
    )
    # 再次使用 pytest 来检查是否引发特定的值错误，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 ArgKmin 的 compute 方法，传入原始的 X 和转换后的 Y，以及其他参数
        ArgKmin.compute(X=X, Y=Y.astype(np.int32), k=k, metric=metric)

    # 使用 pytest 来检查是否引发 k 值小于 -1 的值错误
    with pytest.raises(ValueError, match="k == -1, must be >= 1."):
        # 调用 ArgKmin 的 compute 方法，传入指定的参数，其中 k 为 -1
        ArgKmin.compute(X=X, Y=Y, k=-1, metric=metric)

    # 使用 pytest 来检查是否引发 k 值等于 0 的值错误
    with pytest.raises(ValueError, match="k == 0, must be >= 1."):
        # 调用 ArgKmin 的 compute 方法，传入指定的参数，其中 k 为 0
        ArgKmin.compute(X=X, Y=Y, k=0, metric=metric)

    # 使用 pytest 来检查是否引发未识别距离度量的值错误
    with pytest.raises(ValueError, match="Unrecognized metric"):
        # 调用 ArgKmin 的 compute 方法，传入指定的参数，其中指定了错误的距离度量
        ArgKmin.compute(X=X, Y=Y, k=k, metric="wrong metric")

    # 使用 pytest 来检查是否引发维度不匹配的值错误
    with pytest.raises(
        ValueError, match=r"Buffer has wrong number of dimensions \(expected 2, got 1\)"
    ):
        # 调用 ArgKmin 的 compute 方法，传入包含错误维度数据的 X 和其他参数
        ArgKmin.compute(X=np.array([1.0, 2.0]), Y=Y, k=k, metric=metric)

    # 使用 pytest 来检查是否引发 ndarray 不是 C 连续的值错误
    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        # 调用 ArgKmin 的 compute 方法，传入转换后的 X 和其他参数
        ArgKmin.compute(X=np.asfortranarray(X), Y=Y, k=k, metric=metric)

    # 准备一个不使用的 metric_kwargs 字典，用于检查是否引发 UserWarning
    unused_metric_kwargs = {"p": 3}

    # 准备用于匹配的消息，检查是否引发特定的 UserWarning
    message = r"Some metric_kwargs have been passed \({'p': 3}\) but"

    # 使用 pytest 来检查是否引发特定的 UserWarning，并匹配消息
    with pytest.warns(UserWarning, match=message):
        # 调用 ArgKmin 的 compute 方法，传入指定的参数和 metric_kwargs
        ArgKmin.compute(
            X=X, Y=Y, k=k, metric=metric, metric_kwargs=unused_metric_kwargs
        )

    # 准备一个包含未使用的 metric_kwargs 的字典，用于检查是否引发 UserWarning
    metric_kwargs = {
        "p": 3,  # unused
        "Y_norm_squared": sqeuclidean_row_norms(Y, num_threads=2),
    }

    # 准备用于匹配的消息，检查是否引发特定的 UserWarning
    message = r"Some metric_kwargs have been passed \({'p': 3, 'Y_norm_squared'"

    # 使用 pytest 来检查是否引发特定的 UserWarning，并匹配消息
    with pytest.warns(UserWarning, match=message):
        # 调用 ArgKmin 的 compute 方法，传入指定的参数和 metric_kwargs
        ArgKmin.compute(X=X, Y=Y, k=k, metric=metric, metric_kwargs=metric_kwargs)

    # 在这种情况下，不应引发 UserWarning
    metric_kwargs = {
        "X_norm_squared": sqeuclidean_row_norms(X, num_threads=2),
    }
    # 使用 warnings 模块来捕获并断言不应引发任何 UserWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        # 调用 ArgKmin 的 compute 方法，传入指定的参数和 metric_kwargs
        ArgKmin.compute(X=X, Y=Y, k=k, metric=metric, metric_kwargs=metric_kwargs)

    # 在这种情况下，不应引发 UserWarning
    metric_kwargs = {
        "X_norm_squared": sqeuclidean_row_norms(X, num_threads=2),
        "Y_norm_squared": sqeuclidean_row_norms(Y, num_threads=2),
    }
    # 使用 warnings 模块来捕获并断言不应引发任何 UserWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        # 调用 ArgKmin 的 compute 方法，传入指定的参数和 metric_kwargs
        ArgKmin.compute(X=X, Y=Y, k=k, metric=metric, metric_kwargs=metric_kwargs)
def test_argkmin_classmode_factory_method_wrong_usages():
    # 创建一个随机数生成器对象，种子为1
    rng = np.random.RandomState(1)
    # 生成一个 100x10 的随机浮点数数组 X
    X = rng.rand(100, 10)
    # 生成一个 100x10 的随机浮点数数组 Y
    Y = rng.rand(100, 10)
    # 设置 k 的值为 5
    k = 5
    # 设置距离度量为 "manhattan"
    metric = "manhattan"

    # 设置权重为 "uniform"
    weights = "uniform"
    # 生成一个包含 100 个元素的随机整数数组 Y_labels，范围是从 0 到 10
    Y_labels = rng.randint(low=0, high=10, size=100)
    # 找出 Y_labels 数组中的唯一值
    unique_Y_labels = np.unique(Y_labels)

    # 错误消息内容，指出 X.dtype=float32 而 Y.dtype=float64 时抛出 ValueError
    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float32 and Y.dtype=float64"
    )
    # 验证抛出 ValueError 并匹配指定的错误消息
    with pytest.raises(ValueError, match=msg):
        ArgKminClassMode.compute(
            X=X.astype(np.float32),
            Y=Y,
            k=k,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    # 错误消息内容，指出 X.dtype=float64 而 Y.dtype=int32 时抛出 ValueError
    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float64 and Y.dtype=int32"
    )
    # 验证抛出 ValueError 并匹配指定的错误消息
    with pytest.raises(ValueError, match=msg):
        ArgKminClassMode.compute(
            X=X,
            Y=Y.astype(np.int32),
            k=k,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    # 验证抛出 ValueError 并匹配 "k == -1, must be >= 1."
    with pytest.raises(ValueError, match="k == -1, must be >= 1."):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=-1,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    # 验证抛出 ValueError 并匹配 "k == 0, must be >= 1."
    with pytest.raises(ValueError, match="k == 0, must be >= 1."):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=0,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    # 验证抛出 ValueError 并匹配 "Unrecognized metric"
    with pytest.raises(ValueError, match="Unrecognized metric"):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=k,
            metric="wrong metric",
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    # 验证抛出 ValueError 并匹配 "Buffer has wrong number of dimensions (expected 2, got 1)"
    with pytest.raises(
        ValueError, match=r"Buffer has wrong number of dimensions \(expected 2, got 1\)"
    ):
        ArgKminClassMode.compute(
            X=np.array([1.0, 2.0]),
            Y=Y,
            k=k,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    # 验证抛出 ValueError 并匹配 "ndarray is not C-contiguous"
    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        ArgKminClassMode.compute(
            X=np.asfortranarray(X),
            Y=Y,
            k=k,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    # 非存在的权重策略
    non_existent_weights_strategy = "non_existent_weights_strategy"
    # 生成相应的错误消息内容，指出只支持 'uniform' 或 'distance' 权重选项
    message = (
        "Only the 'uniform' or 'distance' weights options are supported at this time. "
        f"Got: weights='{non_existent_weights_strategy}'."
    )
    # 使用 pytest 的上下文管理器，验证是否会抛出 ValueError 异常，并且异常消息匹配特定的消息字符串。
    with pytest.raises(ValueError, match=message):
        # 调用 ArgKminClassMode 类的 compute 方法，传入以下参数进行计算：
        # - X: 输入数据 X
        # - Y: 输入数据 Y
        # - k: 参数 k
        # - metric: 度量标准
        # - weights: 不存在的权重策略
        # - Y_labels: Y 的标签
        # - unique_Y_labels: Y 的唯一标签
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=k,
            metric=metric,
            weights=non_existent_weights_strategy,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )
    
    # TODO: 一旦支持 ArgKminClassMode 的欧氏特化版本，引入对 UserWarnings 的断言。
    # 目前代码中的 TODO 标记，表示将来要添加对用户警告（UserWarnings）的断言，特别是针对 ArgKminClassMode 的欧氏特化版本。
def test_radius_neighbors_factory_method_wrong_usages():
    # 创建一个随机数生成器，种子为1
    rng = np.random.RandomState(1)
    # 生成一个大小为100x10的随机数组，数据类型为float64
    X = rng.rand(100, 10)
    # 生成一个大小为100x10的随机数组，数据类型为float64
    Y = rng.rand(100, 10)
    # 设置半径为5
    radius = 5
    # 设置距离度量为"euclidean"
    metric = "euclidean"

    # 准备错误消息字符串
    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float32 and Y.dtype=float64"
    )
    # 检查是否抛出ValueError，并匹配错误消息
    with pytest.raises(
        ValueError,
        match=msg,
    ):
        RadiusNeighbors.compute(
            X=X.astype(np.float32), Y=Y, radius=radius, metric=metric
        )

    # 准备错误消息字符串
    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float64 and Y.dtype=int32"
    )
    # 检查是否抛出ValueError，并匹配错误消息
    with pytest.raises(
        ValueError,
        match=msg,
    ):
        RadiusNeighbors.compute(X=X, Y=Y.astype(np.int32), radius=radius, metric=metric)

    # 检查是否抛出ValueError，并匹配错误消息
    with pytest.raises(ValueError, match="radius == -1.0, must be >= 0."):
        RadiusNeighbors.compute(X=X, Y=Y, radius=-1, metric=metric)

    # 检查是否抛出ValueError，并匹配错误消息
    with pytest.raises(ValueError, match="Unrecognized metric"):
        RadiusNeighbors.compute(X=X, Y=Y, radius=radius, metric="wrong metric")

    # 检查是否抛出ValueError，并匹配错误消息
    with pytest.raises(
        ValueError, match=r"Buffer has wrong number of dimensions \(expected 2, got 1\)"
    ):
        RadiusNeighbors.compute(
            X=np.array([1.0, 2.0]), Y=Y, radius=radius, metric=metric
        )

    # 检查是否抛出ValueError，并匹配错误消息
    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        RadiusNeighbors.compute(
            X=np.asfortranarray(X), Y=Y, radius=radius, metric=metric
        )

    # 准备未使用的度量参数字典
    unused_metric_kwargs = {"p": 3}

    # 检查是否产生UserWarning，并匹配警告消息
    message = r"Some metric_kwargs have been passed \({'p': 3}\) but"
    with pytest.warns(UserWarning, match=message):
        RadiusNeighbors.compute(
            X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=unused_metric_kwargs
        )

    # 准备度量参数字典，包含未使用的参数
    metric_kwargs = {
        "p": 3,  # unused
        "Y_norm_squared": sqeuclidean_row_norms(Y, num_threads=2),
    }

    # 检查是否产生UserWarning，并匹配警告消息
    message = r"Some metric_kwargs have been passed \({'p': 3, 'Y_norm_squared'"
    with pytest.warns(UserWarning, match=message):
        RadiusNeighbors.compute(
            X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs
        )

    # 准备度量参数字典，包含所有正确的参数
    metric_kwargs = {
        "X_norm_squared": sqeuclidean_row_norms(X, num_threads=2),
        "Y_norm_squared": sqeuclidean_row_norms(Y, num_threads=2),
    }
    # 检查是否不产生UserWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        RadiusNeighbors.compute(
            X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs
        )

    # 准备度量参数字典，包含部分正确的参数
    metric_kwargs = {
        "X_norm_squared": sqeuclidean_row_norms(X, num_threads=2),
    }
    # 使用 `warnings` 模块捕获警告信息
    with warnings.catch_warnings():
        # 设置警告过滤器，将特定类型的警告转换为异常抛出
        warnings.simplefilter("error", category=UserWarning)
        # 调用 RadiusNeighbors 类的 compute 方法，进行计算
        # 参数说明：
        #   - X: 输入数据集 X
        #   - Y: 输入数据集 Y
        #   - radius: 半径参数，用于定义邻域的范围
        #   - metric: 距离度量方法
        #   - metric_kwargs: 距离度量方法的额外参数
        RadiusNeighbors.compute(
            X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs
        )
# 定义测试函数，用于测试 RadiusNeighborsClassMode 的工厂方法的错误使用情况
def test_radius_neighbors_classmode_factory_method_wrong_usages():
    # 创建随机数生成器对象，种子为1
    rng = np.random.RandomState(1)
    # 生成一个形状为 (100, 10) 的随机浮点数数组 X
    X = rng.rand(100, 10)
    # 生成一个形状为 (100, 10) 的随机浮点数数组 Y
    Y = rng.rand(100, 10)
    # 设置半径为 5
    radius = 5
    # 设置距离度量方式为 "manhattan"
    metric = "manhattan"
    # 设置权重策略为 "uniform"
    weights = "uniform"
    # 生成一个形状为 (100,) 的随机整数数组 Y_labels，取值范围为 [0, 10)
    Y_labels = rng.randint(low=0, high=10, size=100)
    # 获取 Y_labels 数组中唯一值的数组
    unique_Y_labels = np.unique(Y_labels)

    # 准备错误消息字符串，用于匹配 pytest 抛出的 ValueError 异常
    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float32 and Y.dtype=float64"
    )
    # 测试传入 X.dtype=float32 和 Y 的情况是否抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        RadiusNeighborsClassMode.compute(
            X=X.astype(np.float32),
            Y=Y,
            radius=radius,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    # 准备错误消息字符串，用于匹配 pytest 抛出的 ValueError 异常
    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float64 and Y.dtype=int32"
    )
    # 测试传入 X 和 Y.dtype=int32 的情况是否抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        RadiusNeighborsClassMode.compute(
            X=X,
            Y=Y.astype(np.int32),
            radius=radius,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    # 测试半径为负数时是否抛出 ValueError 异常
    with pytest.raises(ValueError, match="radius == -1.0, must be >= 0."):
        RadiusNeighborsClassMode.compute(
            X=X,
            Y=Y,
            radius=-1,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    # 测试不支持的距离度量方式时是否抛出 ValueError 异常
    with pytest.raises(ValueError, match="Unrecognized metric"):
        RadiusNeighborsClassMode.compute(
            X=X,
            Y=Y,
            radius=-1,
            metric="wrong_metric",
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    # 测试输入数组维度不符合预期时是否抛出 ValueError 异常
    with pytest.raises(
        ValueError, match=r"Buffer has wrong number of dimensions \(expected 2, got 1\)"
    ):
        RadiusNeighborsClassMode.compute(
            X=np.array([1.0, 2.0]),
            Y=Y,
            radius=radius,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    # 测试输入数组不是 C 连续时是否抛出 ValueError 异常
    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        RadiusNeighborsClassMode.compute(
            X=np.asfortranarray(X),
            Y=Y,
            radius=radius,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    # 定义一个不存在的权重策略字符串
    non_existent_weights_strategy = "non_existent_weights_strategy"
    # 创建错误消息字符串，指示仅支持 'uniform' 或 'distance' 权重选项
    msg = (
        "Only the 'uniform' or 'distance' weights options are supported at this time. "
        f"Got: weights='{non_existent_weights_strategy}'."
    )
    # 使用 pytest 模块验证是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 RadiusNeighborsClassMode 类的 compute 方法，传入以下参数：
        # X: 特征数据集
        # Y: 目标标签数据集
        # radius: 半径值
        # metric: 错误的距离度量方法（这里用于引发异常）
        # weights: 非存在的权重策略（用于引发异常）
        # Y_labels: Y 的标签数据
        # unique_Y_labels: Y 的唯一标签
        # outlier_label: 异常值标签（这里为 None）
        RadiusNeighborsClassMode.compute(
            X=X,
            Y=Y,
            radius=radius,
            metric="wrong_metric",
            weights=non_existent_weights_strategy,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )
@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
# 参数化测试：Dispatcher 可以是 ArgKmin 或 RadiusNeighbors
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
# 参数化测试：dtype 可以是 np.float64 或 np.float32
def test_chunk_size_agnosticism(
    global_random_seed,
    Dispatcher,
    dtype,
    n_features=100,
):
    """Check that results do not depend on the chunk size."""
    # 检查结果不依赖于分块大小

    rng = np.random.RandomState(global_random_seed)
    # 使用全局随机种子初始化随机数生成器

    spread = 100
    # spread 设置为 100

    n_samples_X, n_samples_Y = rng.choice([97, 100, 101, 500], size=2, replace=False)
    # 随机选择 n_samples_X 和 n_samples_Y 的值，从 [97, 100, 101, 500] 中选择，不放回

    X = rng.rand(n_samples_X, n_features).astype(dtype) * spread
    # 生成 n_samples_X 行 n_features 列的随机数组 X，数据类型转换为 dtype，并乘以 spread

    Y = rng.rand(n_samples_Y, n_features).astype(dtype) * spread
    # 生成 n_samples_Y 行 n_features 列的随机数组 Y，数据类型转换为 dtype，并乘以 spread

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric="euclidean")
        # 计算 X 和 Y 的非平凡半径，使用欧几里得距离
        parameter = radius
        check_parameters = {"radius": radius}
        compute_parameters = {"sort_results": True}

    ref_dist, ref_indices = Dispatcher.compute(
        X,
        Y,
        parameter,
        chunk_size=256,  # 默认值
        metric="manhattan",
        return_distance=True,
        **compute_parameters,
    )
    # 使用 Dispatcher 计算 X 和 Y 的参考距离和索引，使用给定的参数，分块大小为 256，曼哈顿距离，返回距离

    dist, indices = Dispatcher.compute(
        X,
        Y,
        parameter,
        chunk_size=41,
        metric="manhattan",
        return_distance=True,
        **compute_parameters,
    )
    # 使用 Dispatcher 计算 X 和 Y 的距离和索引，使用给定的参数，分块大小为 41，曼哈顿距离，返回距离

    ASSERT_RESULT[(Dispatcher, dtype)](
        ref_dist, dist, ref_indices, indices, **check_parameters
    )
    # 断言结果与预期结果匹配


@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
# 参数化测试：Dispatcher 可以是 ArgKmin 或 RadiusNeighbors
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
# 参数化测试：dtype 可以是 np.float64 或 np.float32
def test_n_threads_agnosticism(
    global_random_seed,
    Dispatcher,
    dtype,
    n_features=100,
):
    """Check that results do not depend on the number of threads."""
    # 检查结果不依赖于线程数

    rng = np.random.RandomState(global_random_seed)
    # 使用全局随机种子初始化随机数生成器

    n_samples_X, n_samples_Y = rng.choice([97, 100, 101, 500], size=2, replace=False)
    # 随机选择 n_samples_X 和 n_samples_Y 的值，从 [97, 100, 101, 500] 中选择，不放回

    spread = 100
    # spread 设置为 100

    X = rng.rand(n_samples_X, n_features).astype(dtype) * spread
    # 生成 n_samples_X 行 n_features 列的随机数组 X，数据类型转换为 dtype，并乘以 spread

    Y = rng.rand(n_samples_Y, n_features).astype(dtype) * spread
    # 生成 n_samples_Y 行 n_features 列的随机数组 Y，数据类型转换为 dtype，并乘以 spread

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric="euclidean")
        # 计算 X 和 Y 的非平凡半径，使用欧几里得距离
        parameter = radius
        check_parameters = {"radius": radius}
        compute_parameters = {"sort_results": True}

    ref_dist, ref_indices = Dispatcher.compute(
        X,
        Y,
        parameter,
        chunk_size=25,  # 确保使用多线程
        return_distance=True,
        **compute_parameters,
    )
    # 使用 Dispatcher 计算 X 和 Y 的参考距离和索引，使用给定的参数，分块大小为 25，返回距离

    with _get_threadpool_controller().limit(limits=1, user_api="openmp"):
        # 使用 _get_threadpool_controller() 控制线程池，限制并发数为 1，用户 API 为 openmp
        dist, indices = Dispatcher.compute(
            X,
            Y,
            parameter,
            chunk_size=25,
            return_distance=True,
            **compute_parameters,
        )
        # 使用 Dispatcher 计算 X 和 Y 的距离和索引，使用给定的参数，分块大小为 25，返回距离

    ASSERT_RESULT[(Dispatcher, dtype)](
        ref_dist, dist, ref_indices, indices, **check_parameters
    )
    # 断言结果与预期结果匹配
    [
        # 创建一个包含四个元组的列表，每个元组包含两个元素：一个是 ArgKmin，另一个是 np.float64 类型
        (ArgKmin, np.float64),
        # 同上，但第二个元素是 np.float32 类型
        (RadiusNeighbors, np.float32),
        # 同上，但第一个元素是 ArgKmin，第二个元素是 np.float32 类型
        (ArgKmin, np.float32),
        # 同上，但第一个元素是 RadiusNeighbors，第二个元素是 np.float64 类型
        (RadiusNeighbors, np.float64),
    ],
)
# 使用 pytest 的 mark.parametrize 装饰器，为测试函数 test_format_agnosticism 参数化，依次传入 CSR_CONTAINERS 中的每个参数进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_format_agnosticism(
    global_random_seed,  # 全局随机种子
    Dispatcher,  # 分发器对象，根据不同的情况分别是 ArgKmin 或者 RadiusNeighbors
    dtype,  # 数据类型
    csr_container,  # CSR 容器，用于将输入转换为稀疏表示
):
    """Check that results do not depend on the format (dense, sparse) of the input."""
    rng = np.random.RandomState(global_random_seed)  # 使用全局随机种子创建随机数生成器对象
    spread = 100  # 随机数范围
    n_samples, n_features = 100, 100  # 样本数和特征数

    X = rng.rand(n_samples, n_features).astype(dtype) * spread  # 生成随机的输入数据 X
    Y = rng.rand(n_samples, n_features).astype(dtype) * spread  # 生成随机的输入数据 Y

    X_csr = csr_container(X)  # 将 X 转换为 CSR 格式
    Y_csr = csr_container(Y)  # 将 Y 转换为 CSR 格式

    if Dispatcher is ArgKmin:
        parameter = 10  # 参数设置为 10
        check_parameters = {}  # 检查参数为空字典
        compute_parameters = {}  # 计算参数为空字典
    else:
        # 调整半径以确保期望结果既不是空的也不是太大
        radius = _non_trivial_radius(X=X, Y=Y, metric="euclidean")  # 根据欧几里得距离计算非平凡半径
        parameter = radius  # 参数设置为计算得到的半径
        check_parameters = {"radius": radius}  # 检查参数包含半径信息
        compute_parameters = {"sort_results": True}  # 计算参数包含排序结果的信息

    # 调用分发器对象的 compute 方法计算距离和索引
    dist_dense, indices_dense = Dispatcher.compute(
        X,
        Y,
        parameter,
        chunk_size=50,
        return_distance=True,
        **compute_parameters,
    )

    # 使用 itertools.product 对 (X, X_csr) 和 (Y, Y_csr) 进行排列组合
    for _X, _Y in itertools.product((X, X_csr), (Y, Y_csr)):
        if _X is X and _Y is Y:
            continue
        # 调用分发器对象的 compute 方法计算距离和索引
        dist, indices = Dispatcher.compute(
            _X,
            _Y,
            parameter,
            chunk_size=50,
            return_distance=True,
            **compute_parameters,
        )
        # 调用 ASSERT_RESULT[(Dispatcher, dtype)] 对比计算结果
        ASSERT_RESULT[(Dispatcher, dtype)](
            dist_dense,
            dist,
            indices_dense,
            indices,
            **check_parameters,
        )


# 使用 pytest 的 mark.parametrize 装饰器，为测试函数 test_strategies_consistency 参数化，依次传入 ArgKmin 和 RadiusNeighbors 进行测试
@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
def test_strategies_consistency(
    global_random_seed,  # 全局随机种子
    global_dtype,  # 全局数据类型
    Dispatcher,  # 分发器对象，分别为 ArgKmin 或 RadiusNeighbors
    n_features=10,  # 特征数，默认为 10
):
    """Check that the results do not depend on the strategy used."""
    rng = np.random.RandomState(global_random_seed)  # 使用全局随机种子创建随机数生成器对象
    metric = rng.choice(  # 随机选择一个距离度量方法
        np.array(
            [
                "euclidean",
                "minkowski",
                "manhattan",
                "haversine",
            ],
            dtype=object,
        )
    )
    # 随机选择样本数目，X 和 Y 的行数分别在 [97, 100, 101, 500] 中选择，不允许重复
    n_samples_X, n_samples_Y = rng.choice([97, 100, 101, 500], size=2, replace=False)
    spread = 100  # 随机数范围
    X = rng.rand(n_samples_X, n_features).astype(global_dtype) * spread  # 生成随机的输入数据 X
    Y = rng.rand(n_samples_Y, n_features).astype(global_dtype) * spread  # 生成随机的输入数据 Y

    # 如果距离度量为 haversine，则只接受 2D 数据
    if metric == "haversine":
        X = np.ascontiguousarray(X[:, :2])
        Y = np.ascontiguousarray(Y[:, :2])

    if Dispatcher is ArgKmin:
        parameter = 10  # 参数设置为 10
        check_parameters = {}  # 检查参数为空字典
        compute_parameters = {}  # 计算参数为空字典
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric=metric)  # 根据指定的距离度量计算非平凡半径
        parameter = radius  # 参数设置为计算得到的半径
        check_parameters = {"radius": radius}  # 检查参数包含半径信息
        compute_parameters = {"sort_results": True}  # 计算参数包含排序结果的信息
    # 调用 Dispatcher 类的 compute 方法，计算两个数据集之间的距离和索引
    dist_par_X, indices_par_X = Dispatcher.compute(
        X,
        Y,
        parameter,
        metric=metric,
        # 获取指定度量参数列表中的第一个参数作为 metric_kwargs
        metric_kwargs=_get_metric_params_list(
            metric, n_features, seed=global_random_seed
        )[0],
        # 设置并行计算的块大小为 X 数据集样本数的四分之一
        chunk_size=n_samples_X // 4,
        # 设置策略为在 X 数据集上并行计算
        strategy="parallel_on_X",
        # 返回距离信息
        return_distance=True,
        **compute_parameters,
    )

    # 调用 Dispatcher 类的 compute 方法，计算两个数据集之间的距离和索引
    dist_par_Y, indices_par_Y = Dispatcher.compute(
        X,
        Y,
        parameter,
        metric=metric,
        # 获取指定度量参数列表中的第一个参数作为 metric_kwargs
        metric_kwargs=_get_metric_params_list(
            metric, n_features, seed=global_random_seed
        )[0],
        # 设置并行计算的块大小为 Y 数据集样本数的四分之一
        chunk_size=n_samples_Y // 4,
        # 设置策略为在 Y 数据集上并行计算
        strategy="parallel_on_Y",
        # 返回距离信息
        return_distance=True,
        **compute_parameters,
    )

    # 使用 ASSERT_RESULT[(Dispatcher, global_dtype)] 进行断言检查，验证计算结果
    ASSERT_RESULT[(Dispatcher, global_dtype)](
        dist_par_X, dist_par_Y, indices_par_X, indices_par_Y, **check_parameters
    )
# 使用pytest的装饰器标记此函数为参数化测试，对CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS中的每个metric参数化
# 对strategy参数化为("parallel_on_X", "parallel_on_Y")，dtype参数化为[np.float64, np.float32]
@pytest.mark.parametrize("metric", CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS)
@pytest.mark.parametrize("strategy", ("parallel_on_X", "parallel_on_Y"))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
# 定义测试函数test_pairwise_distances_argkmin，接受多个参数和默认参数
def test_pairwise_distances_argkmin(
    global_random_seed,  # 全局随机种子
    metric,  # 距离度量方法，从参数化的metric中选择
    strategy,  # 策略，从参数化的strategy中选择
    dtype,  # 数据类型，从参数化的dtype中选择
    csr_container,  # CSR容器类型，用于装载稀疏数据
    n_queries=5,  # 查询点数目，默认为5
    n_samples=100,  # 样本数目，默认为100
    k=10,  # argkmin返回的最小值数目，默认为10
):
    rng = np.random.RandomState(global_random_seed)  # 使用全局随机种子创建随机数生成器实例rng
    n_features = rng.choice([50, 500])  # 随机选择特征数目为50或500
    translation = rng.choice([0, 1e6])  # 随机选择平移值为0或1e6
    spread = 1000  # 扩展值为1000
    X = translation + rng.rand(n_queries, n_features).astype(dtype) * spread  # 创建查询数据X
    Y = translation + rng.rand(n_samples, n_features).astype(dtype) * spread  # 创建样本数据Y

    X_csr = csr_container(X)  # 使用CSR容器装载查询数据X
    Y_csr = csr_container(Y)  # 使用CSR容器装载样本数据Y

    # 仅当metric为"haversine"时，对X和Y进行切片以保证数据为2维
    if metric == "haversine":
        X = np.ascontiguousarray(X[:, :2])
        Y = np.ascontiguousarray(Y[:, :2])

    # 获取metric参数的度量参数列表的第一个值
    metric_kwargs = _get_metric_params_list(metric, n_features)[0]

    # 如果metric为"euclidean"，使用euclidean_distances计算距离矩阵dist_matrix
    if metric == "euclidean":
        dist_matrix = euclidean_distances(X, Y)
    else:
        # 否则使用cdist计算距离矩阵dist_matrix
        dist_matrix = cdist(X, Y, metric=metric, **metric_kwargs)

    # 取距离矩阵dist_matrix每行最小的k个索引，得到argkmin_indices_ref
    argkmin_indices_ref = np.argsort(dist_matrix, axis=1)[:, :k]

    # 初始化argkmin_distances_ref为相同形状的零矩阵
    argkmin_distances_ref = np.zeros(argkmin_indices_ref.shape, dtype=np.float64)

    # 遍历argkmin_indices_ref的每一行，填充argkmin_distances_ref为对应的距离值
    for row_idx in range(argkmin_indices_ref.shape[0]):
        argkmin_distances_ref[row_idx] = dist_matrix[
            row_idx, argkmin_indices_ref[row_idx]
        ]

    # 使用itertools.product生成_X和_Y的组合，对每个组合调用ArgKmin.compute方法
    for _X, _Y in itertools.product((X, X_csr), (Y, Y_csr)):
        # 调用ArgKmin.compute方法计算argkmin_distances和argkmin_indices
        argkmin_distances, argkmin_indices = ArgKmin.compute(
            _X,
            _Y,
            k,
            metric=metric,
            metric_kwargs=metric_kwargs,
            return_distance=True,
            # 设置chunk_size以提升并行性能，至少使用四分之一的样本数
            chunk_size=n_samples // 4,
            strategy=strategy,
        )

        # 使用ASSERT_RESULT[(ArgKmin, dtype)]对计算结果进行断言
        ASSERT_RESULT[(ArgKmin, dtype)](
            argkmin_distances,
            argkmin_distances_ref,
            argkmin_indices,
            argkmin_indices_ref,
        )


# 对CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS中的每个metric参数化
# 对strategy参数化为("parallel_on_X", "parallel_on_Y")，dtype参数化为[np.float64, np.float32]
@pytest.mark.parametrize("metric", CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS)
@pytest.mark.parametrize("strategy", ("parallel_on_X", "parallel_on_Y"))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
# 定义测试函数test_pairwise_distances_radius_neighbors，接受多个参数和默认参数
def test_pairwise_distances_radius_neighbors(
    global_random_seed,  # 全局随机种子
    metric,  # 距离度量方法，从参数化的metric中选择
    strategy,  # 策略，从参数化的strategy中选择
    dtype,  # 数据类型，从参数化的dtype中选择
    n_queries=5,  # 查询点数目，默认为5
    n_samples=100,  # 样本数目，默认为100
):
    rng = np.random.RandomState(global_random_seed)  # 使用全局随机种子创建随机数生成器实例rng
    n_features = rng.choice([50, 500])  # 随机选择特征数目为50或500
    translation = rng.choice([0, 1e6])  # 随机选择平移值为0或1e6
    spread = 1000  # 扩展值为1000
    X = translation + rng.rand(n_queries, n_features).astype(dtype) * spread  # 创建查询数据X
    # 构建翻译后的点集 Y，加上随机数，用于扩展数据分布
    Y = translation + rng.rand(n_samples, n_features).astype(dtype) * spread

    # 获取用于度量的参数列表
    metric_kwargs = _get_metric_params_list(
        metric, n_features, seed=global_random_seed
    )[0]

    # 根据选择的度量计算距离矩阵
    # 如果度量是欧几里德距离，使用 scikit-learn 的优化实现
    if metric == "euclidean":
        dist_matrix = euclidean_distances(X, Y)
    else:
        dist_matrix = cdist(X, Y, metric=metric, **metric_kwargs)

    # 计算非平凡半径
    radius = _non_trivial_radius(precomputed_dists=dist_matrix)

    # 根据给定半径获取邻居
    neigh_indices_ref = []
    neigh_distances_ref = []

    # 遍历距离矩阵的每一行，找到小于等于半径的邻居
    for row in dist_matrix:
        ind = np.arange(row.shape[0])[row <= radius]
        dist = row[ind]

        sort = np.argsort(dist)
        ind, dist = ind[sort], dist[sort]

        neigh_indices_ref.append(ind)
        neigh_distances_ref.append(dist)

    # 使用 RadiusNeighbors 类计算邻居
    neigh_distances, neigh_indices = RadiusNeighbors.compute(
        X,
        Y,
        radius,
        metric=metric,
        metric_kwargs=metric_kwargs,
        return_distance=True,
        # 设置 chunk_size 以增加并行性
        chunk_size=n_samples // 4,
        strategy=strategy,
        sort_results=True,
    )

    # 对比计算结果与参考结果
    ASSERT_RESULT[(RadiusNeighbors, dtype)](
        neigh_distances, neigh_distances_ref, neigh_indices, neigh_indices_ref, radius
    )
# 标记为测试参数化，分别使用 ArgKmin 和 RadiusNeighbors 作为 Dispatcher
# 参数化 metric 使用 "manhattan" 和 "euclidean"
# 参数化 dtype 使用 np.float64 和 np.float32
@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
@pytest.mark.parametrize("metric", ["manhattan", "euclidean"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_memmap_backed_data(
    metric,
    Dispatcher,
    dtype,
):
    """Check that the results do not depend on the datasets writability."""
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    spread = 100
    n_samples, n_features = 128, 10
    # 生成随机数据矩阵 X 和 Y，数据类型由 dtype 决定
    X = rng.rand(n_samples, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples, n_features).astype(dtype) * spread

    # 创建只读的内存映射数据集 X_mm 和 Y_mm
    X_mm, Y_mm = create_memmap_backed_data([X, Y])

    if Dispatcher is ArgKmin:
        # 对于 ArgKmin Dispatcher，设置参数为 10
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        # 对于 RadiusNeighbors Dispatcher，稍微按维度数缩放半径
        radius = 10 ** np.log(n_features)
        parameter = radius
        check_parameters = {"radius": radius}
        compute_parameters = {"sort_results": True}

    # 计算原始数据集的距离和索引
    ref_dist, ref_indices = Dispatcher.compute(
        X,
        Y,
        parameter,
        metric=metric,
        return_distance=True,
        **compute_parameters,
    )

    # 计算内存映射数据集的距离和索引
    dist_mm, indices_mm = Dispatcher.compute(
        X_mm,
        Y_mm,
        parameter,
        metric=metric,
        return_distance=True,
        **compute_parameters,
    )

    # 断言结果一致性
    ASSERT_RESULT[(Dispatcher, dtype)](
        ref_dist, dist_mm, ref_indices, indices_mm, **check_parameters
    )


# 参数化 dtype 使用 np.float64 和 np.float32
# 参数化 csr_container 使用 CSR_CONTAINERS 中的不同容器
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sqeuclidean_row_norms(
    global_random_seed,
    dtype,
    csr_container,
):
    # 创建全局随机数生成器
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    # 随机选择样本数和特征数
    n_samples = rng.choice([97, 100, 101, 1000])
    n_features = rng.choice([5, 10, 100])
    num_threads = rng.choice([1, 2, 8])
    # 生成随机数据矩阵 X，数据类型由 dtype 决定
    X = rng.rand(n_samples, n_features).astype(dtype) * spread

    # 使用 csr_container 创建稀疏矩阵 X_csr
    X_csr = csr_container(X)

    # 计算原始数据集的平方欧氏范数参考值
    sq_row_norm_reference = np.linalg.norm(X, axis=1) ** 2
    # 计算原始数据集的平方欧氏范数
    sq_row_norm = sqeuclidean_row_norms(X, num_threads=num_threads)

    # 计算稀疏数据集的平方欧氏范数
    sq_row_norm_csr = sqeuclidean_row_norms(X_csr, num_threads=num_threads)

    # 断言平方欧氏范数的计算结果一致
    assert_allclose(sq_row_norm_reference, sq_row_norm)
    assert_allclose(sq_row_norm_reference, sq_row_norm_csr)

    # 测试对 Fortran 数组的异常处理
    with pytest.raises(ValueError):
        X = np.asfortranarray(X)
        sqeuclidean_row_norms(X, num_threads=num_threads)


# 测试 ArgKminClassMode 类模式下的策略一致性
def test_argkmin_classmode_strategy_consistent():
    # 创建随机数生成器
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    k = 5
    metric = "manhattan"

    weights = "uniform"
    Y_labels = rng.randint(low=0, high=10, size=100)
    unique_Y_labels = np.unique(Y_labels)
    # 调用 ArgKminClassMode 的 compute 方法，验证策略为 "parallel_on_X"
    results_X = ArgKminClassMode.compute(
        X=X,
        Y=Y,
        k=k,
        metric=metric,
        weights=weights,
        Y_labels=Y_labels,
        unique_Y_labels=unique_Y_labels,
        strategy="parallel_on_X",
    )
    # 使用 ArgKminClassMode 类的 compute 方法计算结果
    results_Y = ArgKminClassMode.compute(
        X=X,  # 输入参数 X
        Y=Y,  # 输入参数 Y
        k=k,  # 输入参数 k，指定类数量
        metric=metric,  # 输入参数 metric，指定度量方式
        weights=weights,  # 输入参数 weights，权重
        Y_labels=Y_labels,  # 输入参数 Y_labels，Y 的标签
        unique_Y_labels=unique_Y_labels,  # 输入参数 unique_Y_labels，唯一的 Y 标签
        strategy="parallel_on_Y",  # 输入参数 strategy，指定计算策略为在 Y 上并行
    )
    
    # 使用 assert_array_equal 函数比较 results_X 和 results_Y 是否相等
    assert_array_equal(results_X, results_Y)
# 使用 pytest 的 parametrize 标记定义一个参数化测试函数，参数为 outlier_label，可以依次取 None, 0, 3, 6, 9
@pytest.mark.parametrize("outlier_label", [None, 0, 3, 6, 9])
# 定义测试函数 test_radius_neighbors_classmode_strategy_consistent，用于测试半径邻居分类模式策略的一致性
def test_radius_neighbors_classmode_strategy_consistent(outlier_label):
    # 使用种子为 1 的随机数生成器创建 RandomState 对象
    rng = np.random.RandomState(1)
    # 创建一个形状为 (100, 10) 的随机数组 X
    X = rng.rand(100, 10)
    # 创建一个形状为 (100, 10) 的随机数组 Y
    Y = rng.rand(100, 10)
    # 设置半径为 5
    radius = 5
    # 设置距离度量为 "manhattan"
    metric = "manhattan"

    # 设置权重为 "uniform"
    weights = "uniform"
    # 从 0 到 10 之间随机生成 100 个整数，作为 Y_labels
    Y_labels = rng.randint(low=0, high=10, size=100)
    # 获取 Y_labels 中的唯一值
    unique_Y_labels = np.unique(Y_labels)
    
    # 调用 RadiusNeighborsClassMode 类的 compute 方法计算结果 results_X
    results_X = RadiusNeighborsClassMode.compute(
        X=X,
        Y=Y,
        radius=radius,
        metric=metric,
        weights=weights,
        Y_labels=Y_labels,
        unique_Y_labels=unique_Y_labels,
        outlier_label=outlier_label,
        strategy="parallel_on_X",
    )
    # 调用 RadiusNeighborsClassMode 类的 compute 方法计算结果 results_Y
    results_Y = RadiusNeighborsClassMode.compute(
        X=X,
        Y=Y,
        radius=radius,
        metric=metric,
        weights=weights,
        Y_labels=Y_labels,
        unique_Y_labels=unique_Y_labels,
        outlier_label=outlier_label,
        strategy="parallel_on_Y",
    )
    # 使用 assert_allclose 函数断言 results_X 和 results_Y 的近似相等性
    assert_allclose(results_X, results_Y)
```