# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\tests\test_common.py`

```
# 导入必要的模块和函数
from functools import partial  # 导入 partial 函数，用于创建部分函数应用
from itertools import chain  # 导入 chain 函数，用于迭代多个可迭代对象的元素串联

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 库

from sklearn.metrics.cluster import (  # 从 sklearn.metrics.cluster 模块导入多个聚类评估指标函数
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.utils._testing import assert_allclose  # 导入 assert_allclose 函数，用于比较数值数组的近似相等性

# Dictionaries of metrics
# ------------------------
# 以下两个字典用于存储聚类评估指标函数和它们的名称，以便后续系统测试
# SUPERVISED_METRICS: 所有有监督聚类评估指标函数（需要真实值）
# UNSUPERVISED_METRICS: 所有无监督聚类评估指标函数
SUPERVISED_METRICS = {
    "adjusted_mutual_info_score": adjusted_mutual_info_score,
    "adjusted_rand_score": adjusted_rand_score,
    "rand_score": rand_score,
    "completeness_score": completeness_score,
    "homogeneity_score": homogeneity_score,
    "mutual_info_score": mutual_info_score,
    "normalized_mutual_info_score": normalized_mutual_info_score,
    "v_measure_score": v_measure_score,
    "fowlkes_mallows_score": fowlkes_mallows_score,
}

UNSUPERVISED_METRICS = {
    "silhouette_score": silhouette_score,
    "silhouette_manhattan": partial(silhouette_score, metric="manhattan"),
    "calinski_harabasz_score": calinski_harabasz_score,
    "davies_bouldin_score": davies_bouldin_score,
}

# Lists of metrics with common properties
# ---------------------------------------
# 下面的列表用于存储具有共同特性的聚类评估指标名称
# SYMMETRIC_METRICS: 对称的聚类评估指标，对 y_true 和 y_pred 对称，仅适用于有监督聚类
# NON_SYMMETRIC_METRICS: 非对称的聚类评估指标，适用于有监督聚类
# NORMALIZED_METRICS: 上界为1的聚类评估指标，适用于有监督和无监督聚类
SYMMETRIC_METRICS = [
    "adjusted_rand_score",
    "rand_score",
    "v_measure_score",
    "mutual_info_score",
    "adjusted_mutual_info_score",
    "normalized_mutual_info_score",
    "fowlkes_mallows_score",
]

NON_SYMMETRIC_METRICS = ["homogeneity_score", "completeness_score"]

NORMALIZED_METRICS = [
    "adjusted_rand_score",
    "rand_score",
    "homogeneity_score",
    "completeness_score",
    "v_measure_score",
    "adjusted_mutual_info_score",
    "fowlkes_mallows_score",
    "normalized_mutual_info_score",
]

rng = np.random.RandomState(0)  # 创建随机数生成器 rng，种子为 0
y1 = rng.randint(3, size=30)  # 生成长度为 30 的随机整数数组 y1，取值范围为 [0, 3)
y2 = rng.randint(3, size=30)  # 生成长度为 30 的随机整数数组 y2，取值范围为 [0, 3)

# Function to test symmetric and non-symmetric union
# ---------------------------------------------------
# test_symmetric_non_symmetric_union 函数用于测试 SYMMETRIC_METRICS 和 NON_SYMMETRIC_METRICS 的并集是否等于 SUPERVISED_METRICS 的键
def test_symmetric_non_symmetric_union():
    assert sorted(SYMMETRIC_METRICS + NON_SYMMETRIC_METRICS) == sorted(
        SUPERVISED_METRICS
    )
# 0.22 AMI and NMI changes
# 忽略 FutureWarning 警告
@pytest.mark.filterwarnings("ignore::FutureWarning")
# 参数化测试，对称度量函数的测试
@pytest.mark.parametrize(
    "metric_name, y1, y2", [(name, y1, y2) for name in SYMMETRIC_METRICS]
)
def test_symmetry(metric_name, y1, y2):
    # 获取指定名称的度量函数
    metric = SUPERVISED_METRICS[metric_name]
    # 断言对称性度量结果应该近似相等
    assert metric(y1, y2) == pytest.approx(metric(y2, y1))


# 参数化测试，非对称度量函数的测试
@pytest.mark.parametrize(
    "metric_name, y1, y2", [(name, y1, y2) for name in NON_SYMMETRIC_METRICS]
)
def test_non_symmetry(metric_name, y1, y2):
    # 获取指定名称的度量函数
    metric = SUPERVISED_METRICS[metric_name]
    # 断言非对称性度量结果应该不近似相等
    assert metric(y1, y2) != pytest.approx(metric(y2, y1))


# 参数化测试，归一化度量函数的输出测试
@pytest.mark.parametrize("metric_name", NORMALIZED_METRICS)
def test_normalized_output(metric_name):
    # 定义上界数据
    upper_bound_1 = [0, 0, 0, 1, 1, 1]
    upper_bound_2 = [0, 0, 0, 1, 1, 1]
    # 获取指定名称的度量函数
    metric = SUPERVISED_METRICS[metric_name]
    # 断言归一化输出应该大于0.0
    assert metric([0, 0, 0, 1, 1], [0, 0, 0, 1, 2]) > 0.0
    assert metric([0, 0, 1, 1, 2], [0, 0, 1, 1, 1]) > 0.0
    assert metric([0, 0, 0, 1, 2], [0, 1, 1, 1, 1]) < 1.0
    assert metric([0, 0, 0, 1, 2], [0, 1, 1, 1, 1]) < 1.0
    assert metric(upper_bound_1, upper_bound_2) == pytest.approx(1.0)

    # 定义下界数据
    lower_bound_1 = [0, 0, 0, 0, 0, 0]
    lower_bound_2 = [0, 1, 2, 3, 4, 5]
    # 计算度量函数对下界数据的得分
    score = np.array(
        [metric(lower_bound_1, lower_bound_2), metric(lower_bound_2, lower_bound_1)]
    )
    # 断言得分中没有小于0的值
    assert not (score < 0).any()


# 参数化测试，标签置换下度量函数的测试
@pytest.mark.parametrize("metric_name", chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
def test_permute_labels(metric_name):
    # 所有聚类度量不受标签置换影响，即当0和1交换时
    y_label = np.array([0, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0])
    if metric_name in SUPERVISED_METRICS:
        # 获取指定名称的监督度量函数
        metric = SUPERVISED_METRICS[metric_name]
        # 计算预测标签置换后的分数
        score_1 = metric(y_pred, y_label)
        assert_allclose(score_1, metric(1 - y_pred, y_label))
        assert_allclose(score_1, metric(1 - y_pred, 1 - y_label))
        assert_allclose(score_1, metric(y_pred, 1 - y_label))
    else:
        # 获取指定名称的非监督度量函数
        metric = UNSUPERVISED_METRICS[metric_name]
        X = np.random.randint(10, size=(7, 10))
        # 计算数据集和预测标签置换后的分数
        score_1 = metric(X, y_pred)
        assert_allclose(score_1, metric(X, 1 - y_pred))


# 参数化测试，格式不变性度量函数的测试
@pytest.mark.parametrize("metric_name", chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
# 所有聚类度量函数的输入参数可以是数组、列表、正数、负数或字符串形式
def test_format_invariance(metric_name):
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7]
    # 定义一个生成器函数，用于生成多种格式的输入数据
    def generate_formats(y):
        # 将输入 y 转换为 NumPy 数组
        y = np.array(y)
        # 返回原始数组 y 和描述信息 "array of ints"
        yield y, "array of ints"
        # 返回将 y 转换为列表的结果和描述信息 "list of ints"
        yield y.tolist(), "list of ints"
        # 返回将 y 转换为由每个元素加上后缀 '-a' 的字符串组成的列表，以及描述信息 "list of strs"
        yield [str(x) + "-a" for x in y.tolist()], "list of strs"
        # 返回将上述字符串列表转换为 NumPy 数组的结果，以及描述信息 "array of strs"
        yield (
            np.array([str(x) + "-a" for x in y.tolist()], dtype=object),
            "array of strs",
        )
        # 返回将 y 中每个元素减去 1 后的结果，以及描述信息 "including negative ints"
        yield y - 1, "including negative ints"
        # 返回将 y 中每个元素加上 1 后的结果，以及描述信息 "strictly positive ints"
        yield y + 1, "strictly positive ints"

    # 如果 metric_name 是 SUPERVISED_METRICS 中的一个指标名
    if metric_name in SUPERVISED_METRICS:
        # 从 SUPERVISED_METRICS 中获取相应的评估函数
        metric = SUPERVISED_METRICS[metric_name]
        # 计算真实值 y_true 和预测值 y_pred 的评分
        score_1 = metric(y_true, y_pred)
        # 生成真实值 y_true 的多种格式
        y_true_gen = generate_formats(y_true)
        # 生成预测值 y_pred 的多种格式
        y_pred_gen = generate_formats(y_pred)
        # 遍历真实值和预测值的多种格式，并确保它们使用相同的评估函数计算的得分一致
        for (y_true_fmt, fmt_name), (y_pred_fmt, _) in zip(y_true_gen, y_pred_gen):
            assert score_1 == metric(y_true_fmt, y_pred_fmt)
    else:
        # 如果 metric_name 不在 SUPERVISED_METRICS 中，则属于无监督评估指标
        # 从 UNSUPERVISED_METRICS 中获取相应的评估函数
        metric = UNSUPERVISED_METRICS[metric_name]
        # 创建一个随机生成的 8x10 的整数数组 X
        X = np.random.randint(10, size=(8, 10))
        # 计算 X 和真实值 y_true 的评分
        score_1 = metric(X, y_true)
        # 确保将 X 转换为浮点数后与 y_true 计算得到的评分一致
        assert score_1 == metric(X.astype(float), y_true)
        # 生成真实值 y_true 的多种格式
        y_true_gen = generate_formats(y_true)
        # 遍历真实值的多种格式，并确保它们与 X 计算得到的评分一致
        for y_true_fmt, fmt_name in y_true_gen:
            assert score_1 == metric(X, y_true_fmt)
# 使用 Pytest 的 parametrize 装饰器为 test_single_sample 函数参数化，参数是 SUPERVISED_METRICS 字典中的值
@pytest.mark.parametrize("metric", SUPERVISED_METRICS.values())
def test_single_sample(metric):
    # 只有受监督指标支持单样本测试
    # 对于每对 (i, j)，分别是 (0, 0), (0, 1), (1, 0), (1, 1)
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        # 调用指标函数 metric，传入单个样本 [i] 和 [j]
        metric([i], [j])


# 使用 Pytest 的 parametrize 装饰器为 test_inf_nan_input 函数参数化，参数是 SUPERVISED_METRICS 和 UNSUPERVISED_METRICS 合并后的字典项
@pytest.mark.parametrize(
    "metric_name, metric_func", dict(SUPERVISED_METRICS, **UNSUPERVISED_METRICS).items()
)
def test_inf_nan_input(metric_name, metric_func):
    # 如果 metric_name 在 SUPERVISED_METRICS 中
    if metric_name in SUPERVISED_METRICS:
        # 创建包含无效输入的列表
        invalids = [
            ([0, 1], [np.inf, np.inf]),   # 第一个样本包含无穷大值
            ([0, 1], [np.nan, np.nan]),   # 第一个样本包含 NaN 值
            ([0, 1], [np.nan, np.inf]),   # 第一个样本同时包含 NaN 和无穷大值
        ]
    else:
        # 否则，随机生成一个 2x10 的整数矩阵 X
        X = np.random.randint(10, size=(2, 10))
        # 创建包含无效输入的列表
        invalids = [(X, [np.inf, np.inf]), (X, [np.nan, np.nan]), (X, [np.nan, np.inf])]
    # 使用 pytest.raises 捕获 ValueError 异常，并匹配正则表达式 "contains (NaN|infinity)"
    with pytest.raises(ValueError, match=r"contains (NaN|infinity)"):
        # 对于每组无效参数 args，在 metric_func 上调用，捕获期望的异常
        for args in invalids:
            metric_func(*args)
```