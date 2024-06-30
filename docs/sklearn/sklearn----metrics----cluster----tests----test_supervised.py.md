# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\tests\test_supervised.py`

```
# 导入警告模块，用于处理可能的警告信息
import warnings

# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 导入 pytest 模块，用于测试
import pytest

# 从 numpy.testing 模块中导入断言函数
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

# 从 sklearn.base 模块中导入 config_context 函数
from sklearn.base import config_context

# 从 sklearn.metrics.cluster 模块中导入多个聚类评估指标函数
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    contingency_matrix,
    entropy,
    expected_mutual_information,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    rand_score,
    v_measure_score,
)

# 从 sklearn.metrics.cluster._supervised 模块中导入 _generalized_average 和 check_clusterings 函数
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings

# 从 sklearn.utils 模块中导入 assert_all_finite 函数
from sklearn.utils import assert_all_finite

# 从 sklearn.utils._array_api 模块中导入 yield_namespace_device_dtype_combinations 函数
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations

# 从 sklearn.utils._testing 模块中导入 _array_api_for_tests 和 assert_almost_equal 函数
from sklearn.utils._testing import _array_api_for_tests, assert_almost_equal

# 定义一个列表 score_funcs，包含了多个聚类评估指标函数
score_funcs = [
    adjusted_rand_score,
    rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
]


# 定义一个测试函数 test_error_messages_on_wrong_input
def test_error_messages_on_wrong_input():
    # 遍历 score_funcs 列表中的每个函数 score_func
    for score_func in score_funcs:
        # 定义预期的错误信息字符串
        expected = (
            r"Found input variables with inconsistent numbers " r"of samples: \[2, 3\]"
        )
        # 使用 pytest 的 pytest.raises 函数断言是否抛出 ValueError 异常，并匹配预期的错误信息
        with pytest.raises(ValueError, match=expected):
            score_func([0, 1], [1, 1, 1])

        expected = r"labels_true must be 1D: shape is \(2"
        with pytest.raises(ValueError, match=expected):
            score_func([[0, 1], [1, 0]], [1, 1, 1])

        expected = r"labels_pred must be 1D: shape is \(2"
        with pytest.raises(ValueError, match=expected):
            score_func([0, 1, 0], [[1, 1], [0, 0]])


# 定义一个测试函数 test_generalized_average
def test_generalized_average():
    # 定义变量 a 和 b 的值
    a, b = 1, 2
    # 定义方法列表 methods
    methods = ["min", "geometric", "arithmetic", "max"]
    # 对每种方法，调用 _generalized_average 函数计算平均值，并存储结果到列表 means 中
    means = [_generalized_average(a, b, method) for method in methods]
    # 断言 means 列表中的值是否按照方法顺序递增
    assert means[0] <= means[1] <= means[2] <= means[3]
    # 重新定义变量 c 和 d 的值
    c, d = 12, 12
    # 再次对每种方法，调用 _generalized_average 函数计算平均值，并存储结果到列表 means 中
    means = [_generalized_average(c, d, method) for method in methods]
    # 断言 means 列表中的所有值是否相等
    assert means[0] == means[1] == means[2] == means[3]


# 定义一个测试函数 test_perfect_matches
def test_perfect_matches():
    # 遍历 score_funcs 列表中的每个函数 score_func
    for score_func in score_funcs:
        # 断言调用 score_func 函数时，空列表作为参数得到的分数近似为 1.0
        assert score_func([], []) == pytest.approx(1.0)
        # 断言调用 score_func 函数时，不同标签的单一样本得到的分数近似为 1.0
        assert score_func([0], [1]) == pytest.approx(1.0)
        # 断言调用 score_func 函数时，完全匹配的标签得到的分数近似为 1.0
        assert score_func([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)
        # 断言调用 score_func 函数时，部分匹配的标签得到的分数近似为 1.0
        assert score_func([0, 1, 0], [42, 7, 42]) == pytest.approx(1.0)
        # 断言调用 score_func 函数时，使用浮点数标签得到的分数近似为 1.0
        assert score_func([0.0, 1.0, 0.0], [42.0, 7.0, 42.0]) == pytest.approx(1.0)
        # 断言调用 score_func 函数时，使用浮点数标签得到的分数近似为 1.0
        assert score_func([0.0, 1.0, 2.0], [42.0, 7.0, 2.0]) == pytest.approx(1.0)
        # 断言调用 score_func 函数时，使用整数标签得到的分数近似为 1.0
        assert score_func([0, 1, 2], [42, 7, 2]) == pytest.approx(1.0)
    # 定义一个特定的评估指标函数列表 score_funcs_with_changing_means
    score_funcs_with_changing_means = [
        normalized_mutual_info_score,
        adjusted_mutual_info_score,
    ]
    # 定义一个集合 means，包含多种方法名称
    means = {"min", "geometric", "arithmetic", "max"}
    # 对于每个定义了变化均值的评分函数进行迭代
    for score_func in score_funcs_with_changing_means:
        # 对于每个均值进行迭代
        for mean in means:
            # 使用空列表计算得分，预期结果接近 1.0
            assert score_func([], [], average_method=mean) == pytest.approx(1.0)
            # 使用单个元素列表计算得分，预期结果接近 1.0
            assert score_func([0], [1], average_method=mean) == pytest.approx(1.0)
            # 使用相同值的列表计算得分，预期结果接近 1.0
            assert score_func(
                [0, 0, 0], [0, 0, 0], average_method=mean
            ) == pytest.approx(1.0)
            # 使用不同值的列表计算得分，预期结果接近 1.0
            assert score_func(
                [0, 1, 0], [42, 7, 42], average_method=mean
            ) == pytest.approx(1.0)
            # 使用浮点数值的列表计算得分，预期结果接近 1.0
            assert score_func(
                [0.0, 1.0, 0.0], [42.0, 7.0, 42.0], average_method=mean
            ) == pytest.approx(1.0)
            # 使用不同浮点数值的列表计算得分，预期结果接近 1.0
            assert score_func(
                [0.0, 1.0, 2.0], [42.0, 7.0, 2.0], average_method=mean
            ) == pytest.approx(1.0)
            # 使用整数和浮点数混合的列表计算得分，预期结果接近 1.0
            assert score_func(
                [0, 1, 2], [42, 7, 2], average_method=mean
            ) == pytest.approx(1.0)
# 定义测试函数，测试同质但不完全标记的情况
def test_homogeneous_but_not_complete_labeling():
    # 调用 homogeneity_completeness_v_measure 函数计算同质性、完整性和V度量
    h, c, v = homogeneity_completeness_v_measure([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 2, 2])
    # 断言同质性接近1.00，精确到小数点后两位
    assert_almost_equal(h, 1.00, 2)
    # 断言完整性接近0.69，精确到小数点后两位
    assert_almost_equal(c, 0.69, 2)
    # 断言V度量接近0.81，精确到小数点后两位
    assert_almost_equal(v, 0.81, 2)


# 定义测试函数，测试完整但不同质标记的情况
def test_complete_but_not_homogeneous_labeling():
    # 调用 homogeneity_completeness_v_measure 函数计算同质性、完整性和V度量
    h, c, v = homogeneity_completeness_v_measure([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1])
    # 断言同质性接近0.58，精确到小数点后两位
    assert_almost_equal(h, 0.58, 2)
    # 断言完整性接近1.00，精确到小数点后两位
    assert_almost_equal(c, 1.00, 2)
    # 断言V度量接近0.73，精确到小数点后两位
    assert_almost_equal(v, 0.73, 2)


# 定义测试函数，测试既不完整也不同质但情况不算太糟的标记
def test_not_complete_and_not_homogeneous_labeling():
    # 调用 homogeneity_completeness_v_measure 函数计算同质性、完整性和V度量
    h, c, v = homogeneity_completeness_v_measure([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2])
    # 断言同质性接近0.67，精确到小数点后两位
    assert_almost_equal(h, 0.67, 2)
    # 断言完整性接近0.42，精确到小数点后两位
    assert_almost_equal(c, 0.42, 2)
    # 断言V度量接近0.52，精确到小数点后两位
    assert_almost_equal(v, 0.52, 2)


# 定义测试函数，测试 beta 参数传递给 homogeneity_completeness_v_measure 和 v_measure_score 函数的情况
def test_beta_parameter():
    # 设定测试用的 beta 值
    beta_test = 0.2
    # 设定预期的同质性、完整性和V度量的测试值
    h_test = 0.67
    c_test = 0.42
    v_test = (1 + beta_test) * h_test * c_test / (beta_test * h_test + c_test)

    # 调用 homogeneity_completeness_v_measure 函数计算同质性、完整性和V度量，传入 beta 参数
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2], beta=beta_test
    )
    # 断言同质性等于预期测试值，精确到小数点后两位
    assert_almost_equal(h, h_test, 2)
    # 断言完整性等于预期测试值，精确到小数点后两位
    assert_almost_equal(c, c_test, 2)
    # 断言V度量等于预期测试值，精确到小数点后两位
    assert_almost_equal(v, v_test, 2)

    # 调用 v_measure_score 函数计算V度量，传入 beta 参数
    v = v_measure_score([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2], beta=beta_test)
    # 断言V度量等于预期测试值，精确到小数点后两位
    assert_almost_equal(v, v_test, 2)


# 定义测试函数，测试标签中存在间隔的情况
def test_non_consecutive_labels():
    # 调用 homogeneity_completeness_v_measure 函数计算同质性、完整性和V度量
    h, c, v = homogeneity_completeness_v_measure([0, 0, 0, 2, 2, 2], [0, 1, 0, 1, 2, 2])
    # 断言同质性接近0.67，精确到小数点后两位
    assert_almost_equal(h, 0.67, 2)
    # 断言完整性接近0.42，精确到小数点后两位
    assert_almost_equal(c, 0.42, 2)
    # 断言V度量接近0.52，精确到小数点后两位
    assert_almost_equal(v, 0.52, 2)

    # 调用 homogeneity_completeness_v_measure 函数计算同质性、完整性和V度量
    h, c, v = homogeneity_completeness_v_measure([0, 0, 0, 1, 1, 1], [0, 4, 0, 4, 2, 2])
    # 断言同质性接近0.67，精确到小数点后两位
    assert_almost_equal(h, 0.67, 2)
    # 断言完整性接近0.42，精确到小数点后两位
    assert_almost_equal(c, 0.42, 2)
    # 断言V度量接近0.52，精确到小数点后两位
    assert_almost_equal(v, 0.52, 2)

    # 计算调整后的兰德指数分数
    ari_1 = adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2])
    ari_2 = adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 4, 0, 4, 2, 2])
    # 断言调整后的兰德指数分数等于0.24，精确到小数点后两位
    assert_almost_equal(ari_1, 0.24, 2)
    assert_almost_equal(ari_2, 0.24, 2)

    # 计算兰德指数分数
    ri_1 = rand_score([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2])
    ri_2 = rand_score([0, 0, 0, 1, 1, 1], [0, 4, 0, 4, 2, 2])
    # 断言兰德指数分数等于0.66，精确到小数点后两位
    assert_almost_equal(ri_1, 0.66, 2)
    assert_almost_equal(ri_2, 0.66, 2)


# 定义函数，计算随机均匀簇标记的分数
def uniform_labelings_scores(score_func, n_samples, k_range, n_runs=10, seed=42):
    # 使用给定的种子生成随机数，获取随机整数生成器
    random_labels = np.random.RandomState(seed).randint
    # 初始化分数数组
    scores = np.zeros((len(k_range), n_runs))
    # 遍历每个簇数
    for i, k in enumerate(k_range):
        # 执行多次运行以获取均
    # 定义用于评估调整后兰德指数在随机标签下的近似为零的函数
    # 设定聚类数的范围
    n_clusters_range = [2, 10, 50, 90]
    # 每个样本点的数量
    n_samples = 100
    # 运行次数
    n_runs = 10

    # 调用 uniform_labelings_scores 函数计算 adjusted_rand_score 的评分
    scores = uniform_labelings_scores(
        adjusted_rand_score, n_samples, n_clusters_range, n_runs
    )

    # 计算每行中的最大绝对值得分
    max_abs_scores = np.abs(scores).max(axis=1)
    # 断言最大绝对值得分接近给定值 [0.02, 0.03, 0.03, 0.02]，精确到小数点后两位
    assert_array_almost_equal(max_abs_scores, [0.02, 0.03, 0.03, 0.02], 2)
def test_adjusted_mutual_info_score():
    # 计算调整后的互信息并与已知值进行测试

    # 定义两个标签数组
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])

    # 计算互信息
    mi = mutual_info_score(labels_a, labels_b)
    assert_almost_equal(mi, 0.41022, 5)

    # 使用稀疏的列联表进行互信息计算
    C = contingency_matrix(labels_a, labels_b, sparse=True)
    mi = mutual_info_score(labels_a, labels_b, contingency=C)
    assert_almost_equal(mi, 0.41022, 5)

    # 使用密集的列联表进行互信息计算
    C = contingency_matrix(labels_a, labels_b)
    mi = mutual_info_score(labels_a, labels_b, contingency=C)
    assert_almost_equal(mi, 0.41022, 5)

    # 计算期望互信息
    n_samples = C.sum()
    emi = expected_mutual_information(C, n_samples)
    assert_almost_equal(emi, 0.15042, 5)

    # 计算调整后的互信息
    ami = adjusted_mutual_info_score(labels_a, labels_b)
    assert_almost_equal(ami, 0.27821, 5)

    # 使用简单的示例进行调整后的互信息测试
    ami = adjusted_mutual_info_score([1, 1, 2, 2], [2, 2, 3, 3])
    assert ami == pytest.approx(1.0)

    # 使用非常大的数组进行调整后的互信息测试
    a110 = np.array([list(labels_a) * 110]).flatten()
    b110 = np.array([list(labels_b) * 110]).flatten()
    ami = adjusted_mutual_info_score(a110, b110)
    assert_almost_equal(ami, 0.38, 2)


def test_expected_mutual_info_overflow():
    # 测试当列联表单元超过 2**16 时的回归情况
    # 这会导致 np.outer 中溢出，从而导致期望互信息大于 1
    assert expected_mutual_information(np.array([[70000]]), 70000) <= 1


def test_int_overflow_mutual_info_fowlkes_mallows_score():
    # 测试 mutual_info_classif 和 fowlkes_mallows_score 中的整数溢出情况
    x = np.array(
        [1] * (52632 + 2529)
        + [2] * (14660 + 793)
        + [3] * (3271 + 204)
        + [4] * (814 + 39)
        + [5] * (316 + 20)
    )
    y = np.array(
        [0] * 52632
        + [1] * 2529
        + [0] * 14660
        + [1] * 793
        + [0] * 3271
        + [1] * 204
        + [0] * 814
        + [1] * 39
        + [0] * 316
        + [1] * 20
    )

    # 确保互信息分数和 fowlkes_mallows_score 函数输出都是有限的
    assert_all_finite(mutual_info_score(x, y))
    assert_all_finite(fowlkes_mallows_score(x, y))


def test_entropy():
    # 测试熵函数的准确性
    assert_almost_equal(entropy([0, 0, 42.0]), 0.6365141, 5)
    assert_almost_equal(entropy([]), 1)
    assert entropy([1, 1, 1, 1]) == 0


@pytest.mark.parametrize(
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
def test_entropy_array_api(array_namespace, device, dtype_name):
    # 使用参数化测试对数组 API 中的熵函数进行测试
    xp = _array_api_for_tests(array_namespace, device)
    float_labels = xp.asarray(np.asarray([0, 0, 42.0], dtype=dtype_name), device=device)
    empty_int32_labels = xp.asarray([], dtype=xp.int32, device=device)
    int_labels = xp.asarray([1, 1, 1, 1], device=device)
    # 使用指定的配置上下文，设置数组 API 调度为 True
    with config_context(array_api_dispatch=True):
        # 断言计算浮点标签的熵是否接近于 0.6365141，允许误差为 1e-5
        assert entropy(float_labels) == pytest.approx(0.6365141, abs=1e-5)
        # 断言计算空的 int32 标签的熵是否为 1
        assert entropy(empty_int32_labels) == 1
        # 断言计算整数标签的熵是否为 0
        assert entropy(int_labels) == 0
# 定义一个测试函数，用于测试 contingency_matrix 函数的行为
def test_contingency_matrix():
    # 创建标签数组 labels_a 和 labels_b，分别表示两组分类结果
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])
    # 调用 contingency_matrix 函数计算混淆矩阵 C
    C = contingency_matrix(labels_a, labels_b)
    # 使用 np.histogram2d 函数计算二维直方图 C2，并取其第一个元素作为结果
    C2 = np.histogram2d(labels_a, labels_b, bins=(np.arange(1, 5), np.arange(1, 5)))[0]
    # 断言 C 和 C2 几乎相等
    assert_array_almost_equal(C, C2)
    # 再次调用 contingency_matrix 函数，使用 eps=0.1 参数计算混淆矩阵 C
    C = contingency_matrix(labels_a, labels_b, eps=0.1)
    # 断言 C 和 C2 加上 0.1 后几乎相等
    assert_array_almost_equal(C, C2 + 0.1)


# 定义一个测试函数，测试 contingency_matrix 函数在 sparse=True 时的行为
def test_contingency_matrix_sparse():
    # 创建标签数组 labels_a 和 labels_b，分别表示两组分类结果
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])
    # 调用 contingency_matrix 函数计算混淆矩阵 C
    C = contingency_matrix(labels_a, labels_b)
    # 调用 contingency_matrix 函数，使用 sparse=True 参数计算稀疏矩阵 C_sparse，并转换为数组
    C_sparse = contingency_matrix(labels_a, labels_b, sparse=True).toarray()
    # 断言 C 和 C_sparse 几乎相等
    assert_array_almost_equal(C, C_sparse)
    # 使用 pytest 检查当 sparse=True 时设置 eps=1e-10 会引发 ValueError 异常
    with pytest.raises(ValueError, match="Cannot set 'eps' when sparse=True"):
        contingency_matrix(labels_a, labels_b, eps=1e-10, sparse=True)


# 定义一个测试函数，测试当信息得分恰好为零时的数值稳定性
def test_exactly_zero_info_score():
    # 循环遍历四个整数范围，从 10 到 10000
    for i in np.logspace(1, 4, 4).astype(int):
        # 创建 labels_a 和 labels_b，labels_a 全为 1，labels_b 从 0 到 i-1
        labels_a, labels_b = (np.ones(i, dtype=int), np.arange(i, dtype=int))
        # 断言 normalized_mutual_info_score 函数返回值接近于 0.0
        assert normalized_mutual_info_score(labels_a, labels_b) == pytest.approx(0.0)
        # 断言 v_measure_score 函数返回值接近于 0.0
        assert v_measure_score(labels_a, labels_b) == pytest.approx(0.0)
        # 断言 adjusted_mutual_info_score 函数返回值接近于 0.0
        assert adjusted_mutual_info_score(labels_a, labels_b) == pytest.approx(0.0)
        # 再次断言 normalized_mutual_info_score 函数返回值接近于 0.0
        assert normalized_mutual_info_score(labels_a, labels_b) == pytest.approx(0.0)
        # 遍历方法列表，检查 adjusted_mutual_info_score 和 normalized_mutual_info_score 函数的返回值接近于 0.0
        for method in ["min", "geometric", "arithmetic", "max"]:
            assert adjusted_mutual_info_score(
                labels_a, labels_b, average_method=method
            ) == pytest.approx(0.0)
            assert normalized_mutual_info_score(
                labels_a, labels_b, average_method=method
            ) == pytest.approx(0.0)


# 定义一个测试函数，测试 v_measure_score 和 mutual_information 函数的关系
def test_v_measure_and_mutual_information(seed=36):
    # 循环遍历四个整数范围，从 10 到 10000
    for i in np.logspace(1, 4, 4).astype(int):
        # 使用给定种子创建随机数生成器
        random_state = np.random.RandomState(seed)
        # 创建 labels_a 和 labels_b，随机整数范围在 0 到 9 之间，共 i 个
        labels_a, labels_b = (
            random_state.randint(0, 10, i),
            random_state.randint(0, 10, i),
        )
        # 断言 v_measure_score 函数返回值接近于 mutual_info_score 的两倍除以 entropy(labels_a) 和 entropy(labels_b) 的和
        assert_almost_equal(
            v_measure_score(labels_a, labels_b),
            2.0
            * mutual_info_score(labels_a, labels_b)
            / (entropy(labels_a) + entropy(labels_b)),
            0,
        )
        # 使用 "arithmetic" 方法断言 v_measure_score 函数返回值接近于 normalized_mutual_info_score 函数的返回值
        avg = "arithmetic"
        assert_almost_equal(
            v_measure_score(labels_a, labels_b),
            normalized_mutual_info_score(labels_a, labels_b, average_method=avg),
        )


# 定义一个测试函数，测试 fowlkes_mallows_score 函数的行为
def test_fowlkes_mallows_score():
    # 测试一般情况下的 fowlkes_mallows_score 函数的行为
    score = fowlkes_mallows_score([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2])
    # 断言 score 的值接近于 4.0 除以 np.sqrt(12.0 * 6.0)
    assert_almost_equal(score, 4.0 / np.sqrt(12.0 * 6.0))

    # 测试完全匹配但标签名称变化的情况
    perfect_score = fowlkes_mallows_score([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0])
    # 断言几乎相等，验证 perfect_score 是否接近 1.0
    assert_almost_equal(perfect_score, 1.0)

    # 计算最差情况下的 Fowlkes-Mallows 分数
    worst_score = fowlkes_mallows_score([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5])
    # 断言几乎相等，验证 worst_score 是否接近 0.0
    assert_almost_equal(worst_score, 0.0)
def test_fowlkes_mallows_score_properties():
    # 手工创建的示例
    labels_a = np.array([0, 0, 0, 1, 1, 2])  # 第一组标签数组
    labels_b = np.array([1, 1, 2, 2, 0, 0])  # 第二组标签数组
    expected = 1.0 / np.sqrt((1.0 + 3.0) * (1.0 + 2.0))  # 预期的 FMI 值计算结果

    score_original = fowlkes_mallows_score(labels_a, labels_b)  # 计算原始标签下的 FMI 分数
    assert_almost_equal(score_original, expected)  # 断言：计算的 FMI 分数与预期值接近

    # 对称性质测试
    score_symmetric = fowlkes_mallows_score(labels_b, labels_a)  # 计算对称标签下的 FMI 分数
    assert_almost_equal(score_symmetric, expected)  # 断言：计算的 FMI 分数与预期值接近

    # 排列性质测试
    score_permuted = fowlkes_mallows_score((labels_a + 1) % 3, labels_b)  # 计算排列后的 FMI 分数
    assert_almost_equal(score_permuted, expected)  # 断言：计算的 FMI 分数与预期值接近

    # 对称和排列性质综合测试
    score_both = fowlkes_mallows_score(labels_b, (labels_a + 2) % 3)  # 计算对称且排列后的 FMI 分数
    assert_almost_equal(score_both, expected)  # 断言：计算的 FMI 分数与预期值接近


@pytest.mark.parametrize(
    "labels_true, labels_pred",
    [
        (["a"] * 6, [1, 1, 0, 0, 1, 1]),  # 标签预测为连续字符数组
        ([1] * 6, [1, 1, 0, 0, 1, 1]),  # 真实标签和预测标签都为整数数组
        ([1, 1, 0, 0, 1, 1], ["a"] * 6),  # 真实标签为整数数组，预测标签为连续字符数组
        ([1, 1, 0, 0, 1, 1], [1] * 6),  # 真实标签和预测标签都为整数数组
        (["a"] * 6, ["a"] * 6),  # 真实标签和预测标签都为连续字符数组
    ],
)
def test_mutual_info_score_positive_constant_label(labels_true, labels_pred):
    # 检查当一个或两个标签都是常量时，互信息为 0
    # 针对 #16355 的非回归测试
    assert mutual_info_score(labels_true, labels_pred) == 0


def test_check_clustering_error():
    # 测试连续值警告消息
    rng = np.random.RandomState(42)
    noise = rng.rand(500)
    wavelength = np.linspace(0.01, 1, 500) * 1e-6
    msg = (
        "Clustering metrics expects discrete values but received "
        "continuous values for label, and continuous values for "
        "target"
    )

    with pytest.warns(UserWarning, match=msg):
        check_clusterings(wavelength, noise)


def test_pair_confusion_matrix_fully_dispersed():
    # 边缘情况：每个元素都是自己的一个独立聚类
    N = 100
    clustering1 = list(range(N))
    clustering2 = clustering1
    expected = np.array([[N * (N - 1), 0], [0, 0]])
    assert_array_equal(pair_confusion_matrix(clustering1, clustering2), expected)


def test_pair_confusion_matrix_single_cluster():
    # 边缘情况：只有一个聚类
    N = 100
    clustering1 = np.zeros((N,))
    clustering2 = clustering1
    expected = np.array([[0, 0], [0, N * (N - 1)]])
    assert_array_equal(pair_confusion_matrix(clustering1, clustering2), expected)


def test_pair_confusion_matrix():
    # 常规情况：不同的非平凡聚类
    n = 10
    N = n**2
    clustering1 = np.hstack([[i + 1] * n for i in range(n)])
    clustering2 = np.hstack([[i + 1] * (n + 1) for i in range(n)])[:N]
    # 基本的二次实现
    expected = np.zeros(shape=(2, 2), dtype=np.int64)
    # 对于 clustering1 中的每个元素，依次执行以下操作
    for i in range(len(clustering1)):
        # 对于 clustering2 中的每个元素，依次执行以下操作
        for j in range(len(clustering2)):
            # 如果 i 不等于 j，则执行以下操作
            if i != j:
                # 检查 clustering1 中的第 i 个元素是否与第 j 个元素相等，结果转换为整数
                same_cluster_1 = int(clustering1[i] == clustering1[j])
                # 检查 clustering2 中的第 i 个元素是否与第 j 个元素相等，结果转换为整数
                same_cluster_2 = int(clustering2[i] == clustering2[j])
                # 在 expected 数组中增加对应位置的计数
                expected[same_cluster_1, same_cluster_2] += 1
    
    # 断言函数 pair_confusion_matrix(clustering1, clustering2) 的返回值与 expected 数组相等
    assert_array_equal(pair_confusion_matrix(clustering1, clustering2), expected)
@pytest.mark.parametrize(
    "clustering1, clustering2",
    [(list(range(100)), list(range(100))), (np.zeros((100,)), np.zeros((100,)))],
)
# 定义参数化测试，包含两个测试用例：
# 1. 所有元素分别为其自身的聚类
# 2. 只有一个聚类
def test_rand_score_edge_cases(clustering1, clustering2):
    # 边界情况1：每个元素都是其自身的聚类
    # 边界情况2：只有一个聚类
    assert_allclose(rand_score(clustering1, clustering2), 1.0)


def test_rand_score():
    # 常规情况：不同的非平凡聚类
    clustering1 = [0, 0, 0, 1, 1, 1]
    clustering2 = [0, 1, 0, 1, 2, 2]
    # 组对混淆矩阵
    D11 = 2 * 2  # 有序对 (1, 3), (5, 6)
    D10 = 2 * 4  # 有序对 (1, 2), (2, 3), (4, 5), (4, 6)
    D01 = 2 * 1  # 有序对 (2, 4)
    D00 = 5 * 6 - D11 - D01 - D10  # 剩余的所有对
    # 兰德指数
    expected_numerator = D00 + D11
    expected_denominator = D00 + D01 + D10 + D11
    expected = expected_numerator / expected_denominator
    assert_allclose(rand_score(clustering1, clustering2), expected)


def test_adjusted_rand_score_overflow():
    """检查大量数据不会导致`adjusted_rand_score`溢出。
    这是对以下问题的非回归测试：
    https://github.com/scikit-learn/scikit-learn/issues/20305
    """
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 100_000, dtype=np.int8)
    y_pred = rng.randint(0, 2, 100_000, dtype=np.int8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("average_method", ["min", "arithmetic", "geometric", "max"])
def test_normalized_mutual_info_score_bounded(average_method):
    """检查标准化互信息得分在0（包含）到1（不包含）之间。
    这是对问题＃13836的非回归测试。
    """
    labels1 = [0] * 469
    labels2 = [1] + labels1[1:]
    labels3 = [0, 1] + labels1[2:]

    # labels1是常数。labels1与任何其他标签的互信息为0。
    nmi = normalized_mutual_info_score(labels1, labels2, average_method=average_method)
    assert nmi == 0

    # 非常数，非完美匹配的标签
    nmi = normalized_mutual_info_score(labels2, labels3, average_method=average_method)
    assert 0 <= nmi < 1
```