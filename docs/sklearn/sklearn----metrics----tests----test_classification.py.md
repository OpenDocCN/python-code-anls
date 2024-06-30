# `D:\src\scipysrc\scikit-learn\sklearn\metrics\tests\test_classification.py`

```
# 导入正则表达式模块
import re
# 导入警告模块
import warnings
# 导入偏函数模块
from functools import partial
# 导入链式迭代工具模块、排列组合工具模块、笛卡尔积工具模块
from itertools import chain, permutations, product

# 导入科学计算库NumPy
import numpy as np
# 导入Pytest测试框架
import pytest
# 导入科学计算库SciPy中的线性代数模块
from scipy import linalg
# 导入SciPy中的Hamming距离函数并重命名为sp_hamming
from scipy.spatial.distance import hamming as sp_hamming
# 导入SciPy中的伯努利分布模块
from scipy.stats import bernoulli

# 导入Scikit-Learn机器学习库中的数据集模块和支持向量机模块
from sklearn import datasets, svm
# 导入Scikit-Learn中的多标签分类数据生成函数
from sklearn.datasets import make_multilabel_classification
# 导入Scikit-Learn中的未定义度量警告
from sklearn.exceptions import UndefinedMetricWarning
# 导入Scikit-Learn中的各种评估指标模块
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    zero_one_loss,
)
# 导入Scikit-Learn中的分类度量校验函数和对数损失函数评分函数
from sklearn.metrics._classification import _check_targets, d2_log_loss_score
# 导入Scikit-Learn中的交叉验证函数
from sklearn.model_selection import cross_val_score
# 导入Scikit-Learn中的标签二值化函数和标签二值化
from sklearn.preprocessing import LabelBinarizer, label_binarize
# 导入Scikit-Learn中的决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-Learn中的模拟DataFrame工具
from sklearn.utils._mocking import MockDataFrame
# 导入Scikit-Learn中的测试辅助函数
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_no_warnings,
    ignore_warnings,
)
# 导入Scikit-Learn中的额外数学函数工具
from sklearn.utils.extmath import _nanaverage
# 导入Scikit-Learn中的修复模块
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
# 导入Scikit-Learn中的验证模块的随机状态检查函数
from sklearn.utils.validation import check_random_state

###############################################################################
# 测试工具函数

def make_prediction(dataset=None, binary=False):
    """使用SVC在一个玩具数据集上进行一些分类预测

    如果binary为True，则限制为二元分类问题，而不是多类分类问题
    """

    if dataset is None:
        # 导入一些用于演示的数据
        dataset = datasets.load_iris()

    X = dataset.data
    y = dataset.target

    if binary:
        # 限制为二元分类任务
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # 添加噪声特征以使问题更难，并避免完美结果
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # 运行分类器，获取类别概率和标签预测
    clf = svm.SVC(kernel="linear", probability=True, random_state=0)
    y_pred_proba = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # 只关注正例的概率
        # XXX: 我们真的想要为二元情况特别设计API吗？
        y_pred_proba = y_pred_proba[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return y_true, y_pred, y_pred_proba
###############################################################################
# Tests

# 定义测试函数，验证分类报告生成的字典输出的准确性
def test_classification_report_dictionary_output():
    # 使用内置的鸢尾花数据集加载数据
    iris = datasets.load_iris()
    # 生成模拟预测结果，返回真实标签、预测标签及额外信息
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # 预期的分类报告字典
    expected_report = {
        "setosa": {
            "precision": 0.82608695652173914,
            "recall": 0.79166666666666663,
            "f1-score": 0.8085106382978724,
            "support": 24,
        },
        "versicolor": {
            "precision": 0.33333333333333331,
            "recall": 0.096774193548387094,
            "f1-score": 0.15000000000000002,
            "support": 31,
        },
        "virginica": {
            "precision": 0.41860465116279072,
            "recall": 0.90000000000000002,
            "f1-score": 0.57142857142857151,
            "support": 20,
        },
        "macro avg": {
            "f1-score": 0.5099797365754813,
            "precision": 0.5260083136726211,
            "recall": 0.596146953405018,
            "support": 75,
        },
        "accuracy": 0.5333333333333333,
        "weighted avg": {
            "f1-score": 0.47310435663627154,
            "precision": 0.5137535108414785,
            "recall": 0.5333333333333333,
            "support": 75,
        },
    }

    # 生成实际的分类报告字典，使用输出字典选项
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
        output_dict=True,
    )

    # 断言预期报告字典和实际报告字典的键相同
    assert report.keys() == expected_report.keys()
    # 遍历比较每个指标的值是否准确
    for key in expected_report:
        if key == "accuracy":
            # 断言准确率为浮点数且与预期值相等
            assert isinstance(report[key], float)
            assert report[key] == expected_report[key]
        else:
            # 断言每个类别的指标值字典键相同，并且值准确到小数点后几位
            assert report[key].keys() == expected_report[key].keys()
            for metric in expected_report[key]:
                assert_almost_equal(expected_report[key][metric], report[key][metric])

    # 断言预期报告字典中特定指标的数据类型
    assert isinstance(expected_report["setosa"]["precision"], float)
    assert isinstance(expected_report["macro avg"]["precision"], float)
    assert isinstance(expected_report["setosa"]["support"], int)
    assert isinstance(expected_report["macro avg"]["support"], int)


# 测试空输入的分类报告生成情况
def test_classification_report_output_dict_empty_input():
    # 生成空输入条件下的分类报告字典
    report = classification_report(y_true=[], y_pred=[], output_dict=True)
    # 预期的空输入条件下的分类报告字典
    expected_report = {
        "accuracy": 0.0,
        "macro avg": {
            "f1-score": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "support": 0,
        },
        "weighted avg": {
            "f1-score": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "support": 0,
        },
    }
    # 断言生成的报告是字典类型
    assert isinstance(report, dict)
    # 断言预期报告字典和生成的报告字典的键相同
    assert report.keys() == expected_report.keys()
    # 遍历期望的报告数据的每一个关键字
    for key in expected_report:
        # 如果关键字是 "accuracy"
        if key == "accuracy":
            # 断言报告中该关键字对应的值是一个浮点数
            assert isinstance(report[key], float)
            # 断言报告中的准确率与期望的准确率相等
            assert report[key] == expected_report[key]
        else:
            # 断言报告中该关键字对应的字典的键与期望的字典键相同
            assert report[key].keys() == expected_report[key].keys()
            # 对于期望的每个度量指标，断言报告中对应的度量指标值与期望值几乎相等
            for metric in expected_report[key]:
                assert_almost_equal(expected_report[key][metric], report[key][metric])
# 使用 pytest 的参数化装饰器，定义了一个测试函数，用于测试 classification_report 函数在不同 zero_division 参数下的行为
@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_classification_report_zero_division_warning(zero_division):
    # 定义测试用的真实标签和预测标签
    y_true, y_pred = ["a", "b", "c"], ["a", "b", "d"]
    # 使用 warnings 模块捕获警告信息
    with warnings.catch_warnings(record=True) as record:
        # 调用 classification_report 函数并记录警告信息
        classification_report(
            y_true, y_pred, zero_division=zero_division, output_dict=True
        )
        # 如果 zero_division 参数为 "warn"，则断言捕获的警告记录大于1
        if zero_division == "warn":
            assert len(record) > 1
            # 遍历每条记录，检查是否包含特定的警告消息
            for item in record:
                msg = "Use `zero_division` parameter to control this behavior."
                assert msg in str(item.message)
        else:
            # 否则，断言没有捕获到警告记录
            assert not record


# 使用 pytest 的参数化装饰器，定义了另一个测试函数，用于测试 classification_report 函数在不同 labels 和 show_micro_avg 参数下的行为
@pytest.mark.parametrize(
    "labels, show_micro_avg", [([0], True), ([0, 1], False), ([0, 1, 2], False)]
)
def test_classification_report_labels_subset_superset(labels, show_micro_avg):
    """Check the behaviour of passing `labels` as a superset or subset of the labels.
    When a superset, we expect to show the "accuracy" in the report while it should be
    the micro-averaging if this is a subset.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27927
    """
    # 定义测试用的真实标签和预测标签
    y_true, y_pred = [0, 1], [0, 1]
    # 调用 classification_report 函数并返回结果字典
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    # 根据 show_micro_avg 参数断言不同的结果
    if show_micro_avg:
        assert "micro avg" in report
        assert "accuracy" not in report
    else:  # accuracy should be shown
        assert "accuracy" in report
        assert "micro avg" not in report


# 定义一个测试函数，测试多标签分类准确率分数
def test_multilabel_accuracy_score_subset_accuracy():
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    # 断言不同条件下的准确率分数
    assert accuracy_score(y1, y2) == 0.5
    assert accuracy_score(y1, y1) == 1
    assert accuracy_score(y2, y2) == 1
    assert accuracy_score(y2, np.logical_not(y2)) == 0
    assert accuracy_score(y1, np.logical_not(y1)) == 0
    assert accuracy_score(y1, np.zeros(y1.shape)) == 0
    assert accuracy_score(y2, np.zeros(y1.shape)) == 0


# 定义一个测试函数，测试二分类任务的精确率、召回率和 F1 分数
def test_precision_recall_f1_score_binary():
    # Test Precision Recall and F1 Score for binary classification task
    y_true, y_pred, _ = make_prediction(binary=True)

    # detailed measures for each class
    # 计算每个类别的精确率、召回率、F1 分数和支持数
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.73, 0.85], 2)
    assert_array_almost_equal(r, [0.88, 0.68], 2)
    assert_array_almost_equal(f, [0.80, 0.76], 2)
    assert_array_equal(s, [25, 25])

    # individual scoring function that can be used for grid search: in the
    # binary class case the score is the value of the measure for the positive
    # class (e.g. label == 1). This is deprecated for average != 'binary'.
    # 对于二分类情况下的评分函数，评分值为正类别（例如标签 == 1）的度量值。对于 average != 'binary'，此用法已不推荐。
    for kwargs, my_assert in [
        ({}, assert_no_warnings),
        ({"average": "binary"}, assert_no_warnings),
        # 计算精确度分数，并使用自定义断言函数验证其准确性
        ps = my_assert(precision_score, y_true, y_pred, **kwargs)
        # 断言精确度分数与预期值0.85相等，精确到小数点后两位
        assert_array_almost_equal(ps, 0.85, 2)

        # 计算召回率分数，并使用自定义断言函数验证其准确性
        rs = my_assert(recall_score, y_true, y_pred, **kwargs)
        # 断言召回率分数与预期值0.68相等，精确到小数点后两位
        assert_array_almost_equal(rs, 0.68, 2)

        # 计算F1分数，并使用自定义断言函数验证其准确性
        fs = my_assert(f1_score, y_true, y_pred, **kwargs)
        # 断言F1分数与预期值0.76相等，精确到小数点后两位
        assert_array_almost_equal(fs, 0.76, 2)

        # 计算F-beta分数（beta=2），并使用自定义断言函数验证其准确性
        assert_almost_equal(
            my_assert(fbeta_score, y_true, y_pred, beta=2, **kwargs),
            # 计算F-beta分数的预期值，使用公式：(1 + beta^2) * ps * rs / (beta^2 * ps + rs)
            (1 + 2**2) * ps * rs / (2**2 * ps + rs),
            # 断言F-beta分数与预期值相等，精确到小数点后两位
            2,
        )
# 带有 @ignore_warnings 装饰器的测试函数，用于测试二分类情况下的精确度、召回率和 F 分数行为
@ignore_warnings
def test_precision_recall_f_binary_single_class():
    # 测试在只有单一正类或负类情况下，精确度、召回率和 F 分数的行为
    # 这种情况可能出现在非分层交叉验证中
    assert 1.0 == precision_score([1, 1], [1, 1])
    assert 1.0 == recall_score([1, 1], [1, 1])
    assert 1.0 == f1_score([1, 1], [1, 1])
    assert 1.0 == fbeta_score([1, 1], [1, 1], beta=0)

    assert 0.0 == precision_score([-1, -1], [-1, -1])
    assert 0.0 == recall_score([-1, -1], [-1, -1])
    assert 0.0 == f1_score([-1, -1], [-1, -1])
    assert 0.0 == fbeta_score([-1, -1], [-1, -1], beta=float("inf"))
    assert fbeta_score([-1, -1], [-1, -1], beta=float("inf")) == pytest.approx(
        fbeta_score([-1, -1], [-1, -1], beta=1e5)
    )


# 带有 @ignore_warnings 装饰器的测试函数，用于测试额外标签的精确度、召回率和 F 分数处理
@ignore_warnings
def test_precision_recall_f_extra_labels():
    # 测试 PRF 处理明确额外（不在输入中）标签的情况
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]

    for i, (y_true, y_pred) in enumerate(data):
        # 没有平均值：数组中的零
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None)
        assert_array_almost_equal([0.0, 1.0, 1.0, 0.5, 0.0], actual)

        # 宏平均值已更改
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro")
        assert_array_almost_equal(np.mean([0.0, 1.0, 1.0, 0.5, 0.0]), actual)

        # 否则不产生影响
        for average in ["micro", "weighted", "samples"]:
            if average == "samples" and i == 0:
                continue
            assert_almost_equal(
                recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=average),
                recall_score(y_true, y_pred, labels=None, average=average),
            )

    # 在多标签情况下引入无效标签会出错
    # （尽管只会影响 average='macro'/None 的性能）
    for average in [None, "macro", "micro", "samples"]:
        with pytest.raises(ValueError):
            recall_score(y_true_bin, y_pred_bin, labels=np.arange(6), average=average)
        with pytest.raises(ValueError):
            recall_score(
                y_true_bin, y_pred_bin, labels=np.arange(-1, 4), average=average
            )

    # 测试问题 #10307 上的非回归
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", labels=[0, 1]
    )
    assert_almost_equal(np.array([p, r, f]), np.array([3 / 4, 1, 5 / 6]))


# 带有 @ignore_warnings 装饰器的测试函数，用于测试可以请求 PRF 子集标签的情况
@ignore_warnings
def test_precision_recall_f_ignored_labels():
    # 测试可以请求 PRF 子集标签的情况
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    # 将 y_pred 标签向量化成二进制形式，用于后续的评估
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    # 将真实标签和预测标签及其二进制形式数据组成列表
    data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]

    # 遍历数据列表，获取索引 i 和对应的真实标签 y_true、预测标签 y_pred
    for i, (y_true, y_pred) in enumerate(data):
        # 定义偏函数 recall_13，计算仅针对标签 [1, 3] 的召回率
        recall_13 = partial(recall_score, y_true, y_pred, labels=[1, 3])
        # 定义偏函数 recall_all，计算所有标签的召回率
        recall_all = partial(recall_score, y_true, y_pred, labels=None)

        # 断言特定条件下的召回率值，用于验证 recall_13 的计算结果
        assert_array_almost_equal([0.5, 1.0], recall_13(average=None))
        # 断言宏平均召回率的计算结果是否接近期望值
        assert_almost_equal((0.5 + 1.0) / 2, recall_13(average="macro"))
        # 断言加权平均召回率的计算结果是否接近期望值
        assert_almost_equal((0.5 * 2 + 1.0 * 1) / 3, recall_13(average="weighted"))
        # 断言微平均召回率的计算结果是否接近期望值
        assert_almost_equal(2.0 / 3, recall_13(average="micro"))

        # 确保上述测试具有实际意义：
        # 针对各种平均方式（宏平均、加权平均、微平均），确保 recall_13 的结果与 recall_all 不同
        for average in ["macro", "weighted", "micro"]:
            assert recall_13(average=average) != recall_all(average=average)
# 定义一个测试函数，用于测试 `average_precision_score` 处理非二进制多类输出的情况
def test_average_precision_score_non_binary_class():
    """Test multiclass-multiouptut for `average_precision_score`."""
    # 定义真实标签 y_true，包含多个样本的多类标签
    y_true = np.array(
        [
            [2, 2, 1],
            [1, 2, 0],
            [0, 1, 2],
            [1, 2, 1],
            [2, 0, 1],
            [1, 2, 1],
        ]
    )
    # 定义预测得分 y_score，每个样本对应多个类别的预测概率
    y_score = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.3, 0.3],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    # 错误信息，指示多类输出格式不被支持
    err_msg = "multiclass-multioutput format is not supported"
    # 使用 pytest 检查是否抛出 ValueError 异常，并匹配错误信息
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_score, pos_label=2)


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        (
            [0, 0, 1, 2],
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.3, 0.5],
                ]
            ),
        ),
        (
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0.1, 0.1, 0.4, 0.5, 0.6, 0.6, 0.9, 0.9, 1, 1],
        ),
    ],
)
def test_average_precision_score_duplicate_values(y_true, y_score):
    """
    Duplicate values with precision-recall require a different
    processing than when computing the AUC of a ROC, because the
    precision-recall curve is a decreasing curve
    The following situation corresponds to a perfect
    test statistic, the average_precision_score should be 1.
    """
    # 断言 `average_precision_score` 对给定的 y_true 和 y_score 应该为 1
    assert average_precision_score(y_true, y_score) == 1


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        (
            [2, 2, 1, 1, 0],
            np.array(
                [
                    [0.2, 0.3, 0.5],
                    [0.2, 0.3, 0.5],
                    [0.4, 0.5, 0.3],
                    [0.4, 0.5, 0.3],
                    [0.8, 0.5, 0.3],
                ]
            ),
        ),
        (
            [0, 1, 1],
            [0.5, 0.5, 0.6],
        ),
    ],
)
def test_average_precision_score_tied_values(y_true, y_score):
    # 如果从左到右观察 y_true，0 值与 1 值是分开的，因此看起来我们正确地排序了分类。
    # 但实际上，前两个值具有相同的分数 (0.5)，因此可以交换位置，导致排序不完美。
    # 这种不完美应该反映在最终分数中，使其小于 1。
    assert average_precision_score(y_true, y_score) != 1.0


def test_precision_recall_f_unused_pos_label():
    # 检查警告，当 pos_label 设置为非默认值，但 average 不等于 'binary' 时会被忽略
    # 虽然数据是二进制的，可以使用 labels=[pos_label] 来指定单个正类。
    
    msg = (
        r"Note that pos_label \(set to 2\) is "
        r"ignored when average != 'binary' \(got 'macro'\). You "
        r"may use labels=\[pos_label\] to specify a single "
        "positive class."
    )
    # 使用 pytest 提供的上下文管理器，在测试中检测是否发出特定类型的警告信息
    with pytest.warns(UserWarning, match=msg):
        # 调用 precision_recall_fscore_support 函数计算指定数据集的指标
        # 第一个参数是真实标签列表 [1, 2, 1]
        # 第二个参数是预测标签列表 [1, 2, 2]
        # pos_label 指定正例标签为 2
        # average 参数指定使用 "macro" 模式计算平均值
        precision_recall_fscore_support(
            [1, 2, 1], [1, 2, 2], pos_label=2, average="macro"
        )
def test_confusion_matrix_binary():
    # Test confusion matrix - binary classification case
    # 调用 make_prediction 函数生成二分类的真实值 y_true 和预测值 y_pred，忽略第三个返回值
    y_true, y_pred, _ = make_prediction(binary=True)

    def test(y_true, y_pred):
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        # 断言混淆矩阵与预期值相等
        assert_array_equal(cm, [[22, 3], [8, 17]])

        # 展开混淆矩阵，计算 true positive, false positive, false negative, true negative
        tp, fp, fn, tn = cm.flatten()
        # 计算 Matthews 相关系数的分子
        num = tp * tn - fp * fn
        # 计算 Matthews 相关系数的分母
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # 计算真实的 Matthews 相关系数（如果分母为零，则设为零）
        true_mcc = 0 if den == 0 else num / den
        # 使用 matthews_corrcoef 计算 Matthews 相关系数
        mcc = matthews_corrcoef(y_true, y_pred)
        # 断言 Matthews 相关系数与预期值近似相等
        assert_array_almost_equal(mcc, true_mcc, decimal=2)
        assert_array_almost_equal(mcc, 0.57, decimal=2)

    # 调用 test 函数进行测试
    test(y_true, y_pred)
    # 以字符串形式再次调用 test 函数进行测试
    test([str(y) for y in y_true], [str(y) for y in y_pred])


def test_multilabel_confusion_matrix_binary():
    # Test multilabel confusion matrix - binary classification case
    # 调用 make_prediction 函数生成二分类的真实值 y_true 和预测值 y_pred，忽略第三个返回值
    y_true, y_pred, _ = make_prediction(binary=True)

    def test(y_true, y_pred):
        # 计算多标签混淆矩阵
        cm = multilabel_confusion_matrix(y_true, y_pred)
        # 断言多标签混淆矩阵与预期值相等
        assert_array_equal(cm, [[[17, 8], [3, 22]], [[22, 3], [8, 17]]])

    # 调用 test 函数进行测试
    test(y_true, y_pred)
    # 以字符串形式再次调用 test 函数进行测试
    test([str(y) for y in y_true], [str(y) for y in y_pred])


def test_multilabel_confusion_matrix_multiclass():
    # Test multilabel confusion matrix - multi-class case
    # 调用 make_prediction 函数生成多类别的真实值 y_true 和预测值 y_pred，忽略第三个返回值
    y_true, y_pred, _ = make_prediction(binary=False)

    def test(y_true, y_pred, string_type=False):
        # 计算多标签混淆矩阵（默认使用标签自省）
        cm = multilabel_confusion_matrix(y_true, y_pred)
        # 断言多标签混淆矩阵与预期值相等
        assert_array_equal(
            cm, [[[47, 4], [5, 19]], [[38, 6], [28, 3]], [[30, 25], [2, 18]]]
        )

        # 计算多标签混淆矩阵（使用显式标签排序）
        labels = ["0", "2", "1"] if string_type else [0, 2, 1]
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        # 断言多标签混淆矩阵与预期值相等
        assert_array_equal(
            cm, [[[47, 4], [5, 19]], [[30, 25], [2, 18]], [[38, 6], [28, 3]]]
        )

        # 计算多标签混淆矩阵（使用超集的标签）
        labels = ["0", "2", "1", "3"] if string_type else [0, 2, 1, 3]
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        # 断言多标签混淆矩阵与预期值相等
        assert_array_equal(
            cm,
            [
                [[47, 4], [5, 19]],
                [[30, 25], [2, 18]],
                [[38, 6], [28, 3]],
                [[75, 0], [0, 0]],
            ],
        )

    # 调用 test 函数进行测试
    test(y_true, y_pred)
    # 以字符串形式再次调用 test 函数进行测试
    test([str(y) for y in y_true], [str(y) for y in y_pred], string_type=True)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_multilabel_confusion_matrix_multilabel(csc_container, csr_container):
    # Test multilabel confusion matrix - multilabel-indicator case
    # 创建示例的多标签真实值和预测值
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    # 将数组转换为压缩稀疏列（CSR）和压缩稀疏行（CSC）格式
    y_true_csr = csr_container(y_true)
    y_pred_csr = csr_container(y_pred)
    y_true_csc = csc_container(y_true)
    y_pred_csc = csc_container(y_pred)
    # 定义样本权重数组
    sample_weight = np.array([2, 1, 3])
    # 定义真实混淆矩阵
    real_cm = [[[1, 0], [1, 1]], [[1, 0], [1, 1]], [[0, 2], [1, 0]]]
    # 定义真实标签数组
    trues = [y_true, y_true_csr, y_true_csc]
    # 定义预测数组
    preds = [y_pred, y_pred_csr, y_pred_csc]

    # 对每个真实标签和预测数组进行交叉测试
    for y_true_tmp in trues:
        for y_pred_tmp in preds:
            # 计算多标签混淆矩阵
            cm = multilabel_confusion_matrix(y_true_tmp, y_pred_tmp)
            # 断言多标签混淆矩阵与真实混淆矩阵相等
            assert_array_equal(cm, real_cm)

    # 测试支持按样本计算混淆矩阵
    cm = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
    # 断言按样本计算的混淆矩阵与指定的真实混淆矩阵相等
    assert_array_equal(cm, [[[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [2, 0]]])

    # 测试支持指定标签计算混淆矩阵
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0])
    # 断言按指定标签计算的混淆矩阵与指定的真实混淆矩阵相等
    assert_array_equal(cm, [[[0, 2], [1, 0]], [[1, 0], [1, 1]]])

    # 测试支持指定标签并按样本计算混淆矩阵
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0], samplewise=True)
    # 断言按指定标签并按样本计算的混淆矩阵与指定的真实混淆矩阵相等
    assert_array_equal(cm, [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])

    # 测试支持样本权重并按样本计算混淆矩阵
    cm = multilabel_confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, samplewise=True
    )
    # 断言按样本权重并按样本计算的混淆矩阵与指定的真实混淆矩阵相等
    assert_array_equal(cm, [[[2, 0], [2, 2]], [[1, 1], [0, 1]], [[0, 3], [6, 0]]])
def test_multilabel_confusion_matrix_errors():
    # 定义真实标签和预测标签的多标签混淆矩阵测试用例
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    # 测试不良的样本权重
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        multilabel_confusion_matrix(y_true, y_pred, sample_weight=[1, 2])
    with pytest.raises(ValueError, match="should be a 1d array"):
        multilabel_confusion_matrix(
            y_true, y_pred, sample_weight=[[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        )

    # 测试不良的标签
    err_msg = r"All labels must be in \[0, n labels\)"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[-1])
    err_msg = r"All labels must be in \[0, n labels\)"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[3])

    # 在多标签情况下使用单样本指标
    with pytest.raises(ValueError, match="Samplewise metrics"):
        multilabel_confusion_matrix([0, 1, 2], [1, 2, 0], samplewise=True)

    # 测试不良的 y_type
    err_msg = "multiclass-multioutput is not supported"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix([[0, 1, 2], [2, 1, 0]], [[1, 2, 0], [1, 0, 2]])


@pytest.mark.parametrize(
    "normalize, cm_dtype, expected_results",
    [
        ("true", "f", 0.333333333),
        ("pred", "f", 0.333333333),
        ("all", "f", 0.1111111111),
        (None, "i", 2),
    ],
)
def test_confusion_matrix_normalize(normalize, cm_dtype, expected_results):
    # 定义混淆矩阵标准化测试用例，包括标准化类型、数据类型和预期结果
    y_test = [0, 1, 2] * 6
    y_pred = list(chain(*permutations([0, 1, 2])))
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    assert_allclose(cm, expected_results)
    assert cm.dtype.kind == cm_dtype


def test_confusion_matrix_normalize_single_class():
    # 测试仅包含单个类别的混淆矩阵标准化行为
    y_test = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0]

    cm_true = confusion_matrix(y_test, y_pred, normalize="true")
    assert cm_true.sum() == pytest.approx(2.0)

    # 另外检查确保由于零除法而没有引发警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cm_pred = confusion_matrix(y_test, y_pred, normalize="pred")

    assert cm_pred.sum() == pytest.approx(1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        confusion_matrix(y_pred, y_test, normalize="true")


def test_confusion_matrix_single_label():
    """测试 `confusion_matrix` 在仅发现一个标签时是否会发出警告。"""
    y_test = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]

    with pytest.warns(UserWarning, match="A single label was found in"):
        confusion_matrix(y_pred, y_test)


@pytest.mark.parametrize(
    "params, warn_msg",
    [
        # 当 y_test 只包含一类且 y_test==y_pred 时，LR+ 未定义
        (
            {
                "y_true": np.array([0, 0, 0, 0, 0, 0]),
                "y_pred": np.array([0, 0, 0, 0, 0, 0]),
            },
            "samples of only one class were seen during testing",
        ),
        # 当 `fp == 0` 且 `tp != 0` 时，LR+ 未定义
        (
            {
                "y_true": np.array([1, 1, 1, 0, 0, 0]),
                "y_pred": np.array([1, 1, 1, 0, 0, 0]),
            },
            "positive_likelihood_ratio ill-defined and being set to nan",
        ),
        # 当 `fp == 0` 且 `tp == 0` 时，LR+ 未定义
        (
            {
                "y_true": np.array([1, 1, 1, 0, 0, 0]),
                "y_pred": np.array([0, 0, 0, 0, 0, 0]),
            },
            "no samples predicted for the positive class",
        ),
        # 当 `tn == 0` 时，LR- 未定义
        (
            {
                "y_true": np.array([1, 1, 1, 0, 0, 0]),
                "y_pred": np.array([0, 0, 0, 1, 1, 1]),
            },
            "negative_likelihood_ratio ill-defined and being set to nan",
        ),
        # 当 `tp + fn == 0` 时，两种比率均未定义
        (
            {
                "y_true": np.array([0, 0, 0, 0, 0, 0]),
                "y_pred": np.array([1, 1, 1, 0, 0, 0]),
            },
            "no samples of the positive class were present in the testing set",
        ),
    ],
)

# 定义测试函数 test_likelihood_ratios_warnings，用于检查 likelihood_ratios 是否在至少一个比率不确定时触发警告
def test_likelihood_ratios_warnings(params, warn_msg):
    # likelihood_ratios must raise warnings when at
    # least one of the ratios is ill-defined.
    
    # 使用 pytest 的 warn 模块检查是否有 UserWarning，并匹配给定的警告消息
    with pytest.warns(UserWarning, match=warn_msg):
        class_likelihood_ratios(**params)


# 使用 pytest 的 parametrize 装饰器定义参数化测试函数 test_likelihood_ratios_errors
@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {
                "y_true": np.array([0, 1, 0, 1, 0]),
                "y_pred": np.array([1, 1, 0, 0, 2]),
            },
            (
                "class_likelihood_ratios only supports binary classification "
                "problems, got targets of type: multiclass"
            ),
        ),
    ],
)
# 定义测试函数 test_likelihood_ratios_errors，检查 likelihood_ratios 在非二元类别时是否引发错误，以避免辛普森悖论
def test_likelihood_ratios_errors(params, err_msg):
    # likelihood_ratios must raise error when attempting
    # non-binary classes to avoid Simpson's paradox
    with pytest.raises(ValueError, match=err_msg):
        class_likelihood_ratios(**params)


# 定义测试函数 test_likelihood_ratios，用于测试 class_likelihood_ratios 函数的正常情况
def test_likelihood_ratios():
    # Build confusion matrix with tn=9, fp=8, fn=1, tp=2,
    # sensitivity=2/3, specificity=9/17, prevalence=3/20,
    # LR+=34/24, LR-=17/27
    # 构建混淆矩阵和相关指标
    y_true = np.array([1] * 3 + [0] * 17)
    y_pred = np.array([1] * 2 + [0] * 10 + [1] * 8)

    # 调用 class_likelihood_ratios 计算正负类别的 likelihood ratios，并断言其值的接近性
    pos, neg = class_likelihood_ratios(y_true, y_pred)
    assert_allclose(pos, 34 / 24)
    assert_allclose(neg, 17 / 27)

    # Build limit case with y_pred = y_true
    # 构建 y_pred 与 y_true 完全相等的情况
    pos, neg = class_likelihood_ratios(y_true, y_true)
    assert_array_equal(pos, np.nan * 2)  # 断言正类别结果为 NaN
    assert_allclose(neg, np.zeros(2), rtol=1e-12)  # 断言负类别结果接近零，给定相对误差

    # Ignore last 5 samples to get tn=9, fp=3, fn=1, tp=2,
    # sensitivity=2/3, specificity=9/12, prevalence=3/20,
    # LR+=24/9, LR-=12/27
    # 忽略最后 5 个样本，计算混淆矩阵和相关指标
    sample_weight = np.array([1.0] * 15 + [0.0] * 5)
    pos, neg = class_likelihood_ratios(y_true, y_pred, sample_weight=sample_weight)
    assert_allclose(pos, 24 / 9)  # 断言正类别结果的接近性
    assert_allclose(neg, 12 / 27)  # 断言负类别结果的接近性


# 定义测试函数 test_cohen_kappa，用于测试 cohen_kappa_score 函数的不同用例
def test_cohen_kappa():
    # These label vectors reproduce the contingency matrix from Artstein and
    # Poesio (2008), Table 1: np.array([[20, 20], [10, 50]]).
    # 定义与 Artstein 和 Poesio (2008) 表1 中相同的标签向量
    y1 = np.array([0] * 40 + [1] * 60)
    y2 = np.array([0] * 20 + [1] * 20 + [0] * 10 + [1] * 50)

    # 计算 Cohen's kappa 系数，并断言其值的接近性
    kappa = cohen_kappa_score(y1, y2)
    assert_almost_equal(kappa, 0.348, decimal=3)
    assert kappa == cohen_kappa_score(y2, y1)

    # Add spurious labels and ignore them.
    # 添加不相关的标签并忽略它们
    y1 = np.append(y1, [2] * 4)
    y2 = np.append(y2, [2] * 4)
    assert cohen_kappa_score(y1, y2, labels=[0, 1]) == kappa  # 断言忽略不相关标签后的 kappa 值相同

    assert_almost_equal(cohen_kappa_score(y1, y1), 1.0)  # 断言完全相同标签的 kappa 值接近 1.0

    # Multiclass example: Artstein and Poesio, Table 4.
    # 多类别示例，与 Artstein 和 Poesio, Table 4 相同
    y1 = np.array([0] * 46 + [1] * 44 + [2] * 10)
    y2 = np.array([0] * 52 + [1] * 32 + [2] * 16)
    assert_almost_equal(cohen_kappa_score(y1, y2), 0.8013, decimal=4)

    # Weighting example: none, linear, quadratic.
    # 权重示例：无权重、线性权重、二次权重
    y1 = np.array([0] * 46 + [1] * 44 + [2] * 10)
    y2 = np.array([0] * 50 + [1] * 40 + [2] * 10)
    assert_almost_equal(cohen_kappa_score(y1, y2), 0.9315, decimal=4)
    assert_almost_equal(cohen_kappa_score(y1, y2, weights="linear"), 0.9412, decimal=4)
    # 使用 assert_almost_equal 函数检查 y1 和 y2 之间的 Cohen's kappa 分数是否接近于 0.9541，
    # 使用 quadratic 权重进行加权计算，精确到小数点后四位。
    assert_almost_equal(
        cohen_kappa_score(y1, y2, weights="quadratic"), 0.9541, decimal=4
    )
# 定义一个测试函数，测试 `matthews_corrcoef` 函数处理 NaN 值时的行为
def test_matthews_corrcoef_nan():
    # 断言当输入 `[0]` 和 `[1]` 时，`matthews_corrcoef` 应返回 0.0
    assert matthews_corrcoef([0], [1]) == 0.0
    # 断言当输入 `[0, 0]` 和 `[0, 1]` 时，`matthews_corrcoef` 应返回 0.0
    assert matthews_corrcoef([0, 0], [0, 1]) == 0.0


# 使用 `pytest.mark.parametrize` 标记，定义多组参数用于参数化测试
@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
@pytest.mark.parametrize("y_true, y_pred", [([0], [0]), ([], [])])
@pytest.mark.parametrize(
    "metric",
    [
        f1_score,  # F1 分数
        partial(fbeta_score, beta=1),  # F-beta 分数，beta 值设为 1
        precision_score,  # 精确率
        recall_score,  # 召回率
        partial(cohen_kappa_score, labels=[0, 1]),  # Cohen's Kappa 系数，指定标签为 [0, 1]
    ],
)
# 定义测试函数，测试 `zero_division` 为 NaN 时的行为
def test_zero_division_nan_no_warning(metric, y_true, y_pred, zero_division):
    """Check the behaviour of `zero_division` when setting to 0, 1 or np.nan.
    No warnings should be raised.
    """
    # 使用 `warnings.catch_warnings()` 捕获警告
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 设定简单过滤器，将警告转换为错误
        result = metric(y_true, y_pred, zero_division=zero_division)  # 调用度量函数

    # 如果 `zero_division` 是 NaN，则断言结果也应该是 NaN
    if np.isnan(zero_division):
        assert np.isnan(result)
    else:
        assert result == zero_division  # 否则断言结果与 `zero_division` 相等


# 使用 `pytest.mark.parametrize` 标记，定义多组参数用于参数化测试
@pytest.mark.parametrize("y_true, y_pred", [([0], [0]), ([], [])])
@pytest.mark.parametrize(
    "metric",
    [
        f1_score,  # F1 分数
        partial(fbeta_score, beta=1),  # F-beta 分数，beta 值设为 1
        precision_score,  # 精确率
        recall_score,  # 召回率
        cohen_kappa_score,  # Cohen's Kappa 系数
    ],
)
# 定义测试函数，测试 `zero_division` 为 "warn" 时的行为
def test_zero_division_nan_warning(metric, y_true, y_pred):
    """Check the behaviour of `zero_division` when setting to "warn".
    A `UndefinedMetricWarning` should be raised.
    """
    # 使用 `pytest.warns` 断言会引发 `UndefinedMetricWarning` 警告
    with pytest.warns(UndefinedMetricWarning):
        result = metric(y_true, y_pred, zero_division="warn")
    assert result == 0.0  # 断言结果为 0.0


# 定义测试函数，测试 `matthews_corrcoef` 函数与 `np.corrcoef` 函数的结果是否接近
def test_matthews_corrcoef_against_numpy_corrcoef():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=20)  # 生成随机的 y_true
    y_pred = rng.randint(0, 2, size=20)  # 生成随机的 y_pred

    # 断言 `matthews_corrcoef` 函数与 `np.corrcoef` 函数计算的结果接近
    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred), np.corrcoef(y_true, y_pred)[0, 1], 10
    )


# 定义测试函数，测试 `matthews_corrcoef` 函数与 Jurman 等人定义的方法的一致性
def test_matthews_corrcoef_against_jurman():
    # 检查多分类情况下 `matthews_corrcoef` 与 Jurman 等人的定义是否一致
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=20)  # 生成随机的 y_true
    y_pred = rng.randint(0, 2, size=20)  # 生成随机的 y_pred
    sample_weight = rng.rand(20)  # 生成随机的样本权重

    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)  # 计算混淆矩阵 C
    N = len(C)
    # 计算 Jurman 等人定义的 MCC
    cov_ytyp = sum(
        [
            C[k, k] * C[m, l] - C[l, k] * C[k, m]
            for k in range(N)
            for m in range(N)
            for l in range(N)
        ]
    )
    cov_ytyt = sum(
        [
            C[:, k].sum()
            * np.sum([C[g, f] for f in range(N) for g in range(N) if f != k])
            for k in range(N)
        ]
    )
    cov_ypyp = np.sum(
        [
            C[k, :].sum()
            * np.sum([C[f, g] for f in range(N) for g in range(N) if f != k])
            for k in range(N)
        ]
    )
    mcc_jurman = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)  # 计算 Jurman 等人定义的 MCC
    mcc_ours = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)  # 计算我们定义的 MCC
    # 使用断言检查两个变量 mcc_ours 和 mcc_jurman 的值是否在小数点后10位上几乎相等
    assert_almost_equal(mcc_ours, mcc_jurman, 10)
# 定义一个测试函数，用于测试 matthews_corrcoef 函数的各种情况
def test_matthews_corrcoef():
    # 设置随机数种子为0，以确保结果可复现
    rng = np.random.RandomState(0)
    # 根据随机数生成器生成长度为20的随机二分类真实标签列表
    y_true = ["a" if i == 0 else "b" for i in rng.randint(0, 2, size=20)]

    # 测试当两个向量完全相同时，matthews_corrcoef 应该为1
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)

    # 测试当两个向量完全相反时，matthews_corrcoef 应该为-1
    y_true_inv = ["b" if i == "a" else "a" for i in y_true]
    assert_almost_equal(matthews_corrcoef(y_true, y_true_inv), -1)

    # 测试当一个向量是二分类形式时，matthews_corrcoef 应该为-1
    y_true_inv2 = label_binarize(y_true, classes=["a", "b"])
    y_true_inv2 = np.where(y_true_inv2, "a", "b")
    assert_almost_equal(matthews_corrcoef(y_true, y_true_inv2), -1)

    # 测试当两个向量都是全0向量时，matthews_corrcoef 应该为0
    assert_almost_equal(matthews_corrcoef([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    # 测试当一个向量是全一向量，另一个向量是全a向量时，matthews_corrcoef 应该为0
    assert_almost_equal(matthews_corrcoef(y_true, ["a"] * len(y_true)), 0.0)

    # 测试两个向量之间没有相关性时，matthews_corrcoef 应该为0
    y_1 = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    y_2 = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert_almost_equal(matthews_corrcoef(y_1, y_2), 0.0)

    # 测试样本权重能够选择性地排除一部分样本时，matthews_corrcoef 应该触发 AssertionError
    mask = [1] * 10 + [0] * 10
    with pytest.raises(AssertionError):
        assert_almost_equal(matthews_corrcoef(y_1, y_2, sample_weight=mask), 0.0)


# 定义一个测试函数，用于测试 matthews_corrcoef 函数在多类别情况下的表现
def test_matthews_corrcoef_multiclass():
    # 设置随机数种子为0，以确保结果可复现
    rng = np.random.RandomState(0)
    # 将字符'a'的ASCII码作为起点，生成长度为20的随机多类别真实标签列表
    ord_a = ord("a")
    n_classes = 4
    y_true = [chr(ord_a + i) for i in rng.randint(0, n_classes, size=20)]

    # 测试当两个向量完全相同时，matthews_corrcoef 应该为1
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)

    # 测试多类别情况下，完全相反的向量，matthews_corrcoef 应该为-0.5
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred_bad = [2, 2, 0, 0, 1, 1]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred_bad), -0.5)

    # 测试最小化假阳性和假阴性时，matthews_corrcoef 的极小值
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred_min = [1, 1, 0, 0, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred_min), -12 / np.sqrt(24 * 16))

    # 测试当两个向量的方差为0时，matthews_corrcoef 应该为0
    y_true = [0, 1, 2]
    y_pred = [3, 3, 3]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), 0.0)

    # 同样测试在全真实标签方差为0时，matthews_corrcoef 应该为0
    y_true = [3, 3, 3]
    y_pred = [0, 1, 2]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), 0.0)

    # 测试两个向量之间没有相关性时，matthews_corrcoef 应该为0
    y_1 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_2 = [1, 1, 1, 2, 2, 2, 0, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_1, y_2), 0.0)

    # 可以使用多类别计算来测试二元假设的有效性
    # 定义真实标签和预测标签，以及样本权重，用于测试 Matthews 相关系数的计算

    # 当屏蔽掉样本权重为0的最后一个标签时，预期 Matthews 相关系数为 -1
    y_true = [0, 0, 1, 1, 2]
    y_pred = [1, 1, 0, 0, 2]
    sample_weight = [1, 1, 1, 1, 0]
    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight), -1
    )

    # 对于全零向量的情况，无法计算相关系数，预期输出为 0
    y_true = [0, 0, 1, 2]
    y_pred = [0, 0, 1, 2]
    sample_weight = [1, 1, 0, 0]
    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight), 0.0
    )
@pytest.mark.parametrize("n_points", [100, 10000])
def test_matthews_corrcoef_overflow(n_points):
    # 根据 GitHub 上的 issue 报告，解决了 matthews_corrcoef 函数在特定情况下的溢出问题
    rng = np.random.RandomState(20170906)

    def mcc_safe(y_true, y_pred):
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_true, y_pred)
        true_pos = conf_matrix[1, 1]  # 真正例数
        false_pos = conf_matrix[1, 0]  # 假正例数
        false_neg = conf_matrix[0, 1]  # 假负例数
        n_points = len(y_true)  # 样本总数
        pos_rate = (true_pos + false_neg) / n_points  # 正类率
        activity = (true_pos + false_pos) / n_points  # 活跃度
        # 计算 MCC 的分子部分
        mcc_numerator = true_pos / n_points - pos_rate * activity
        # 计算 MCC 的分母部分
        mcc_denominator = activity * pos_rate * (1 - activity) * (1 - pos_rate)
        return mcc_numerator / np.sqrt(mcc_denominator)

    def random_ys(n_points):  # 生成随机的二分类标签
        x_true = rng.random_sample(n_points)
        x_pred = x_true + 0.2 * (rng.random_sample(n_points) - 0.5)
        y_true = x_true > 0.5  # 生成真实标签
        y_pred = x_pred > 0.5  # 生成预测标签
        return y_true, y_pred

    arr = np.repeat([0.0, 1.0], n_points)  # 生成二分类数据
    assert_almost_equal(matthews_corrcoef(arr, arr), 1.0)
    arr = np.repeat([0.0, 1.0, 2.0], n_points)  # 生成多分类数据
    assert_almost_equal(matthews_corrcoef(arr, arr), 1.0)

    y_true, y_pred = random_ys(n_points)
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), mcc_safe(y_true, y_pred))


def test_precision_recall_f1_score_multiclass():
    # 测试多分类任务的 Precision、Recall 和 F1 Score
    y_true, y_pred, _ = make_prediction(binary=False)

    # 使用默认的标签内省计算分数
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
    assert_array_almost_equal(r, [0.79, 0.09, 0.90], 2)
    assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
    assert_array_equal(s, [24, 31, 20])

    # 进行平均测试
    ps = precision_score(y_true, y_pred, pos_label=1, average="micro")
    assert_array_almost_equal(ps, 0.53, 2)

    rs = recall_score(y_true, y_pred, average="micro")
    assert_array_almost_equal(rs, 0.53, 2)

    fs = f1_score(y_true, y_pred, average="micro")
    assert_array_almost_equal(fs, 0.53, 2)

    ps = precision_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(ps, 0.53, 2)

    rs = recall_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(rs, 0.60, 2)

    fs = f1_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(fs, 0.51, 2)

    ps = precision_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(ps, 0.51, 2)

    rs = recall_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(rs, 0.53, 2)

    fs = f1_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(fs, 0.47, 2)

    with pytest.raises(ValueError):
        precision_score(y_true, y_pred, average="samples")
    # 使用 pytest 检查是否会抛出 ValueError 异常，测试 recall_score 函数
    with pytest.raises(ValueError):
        recall_score(y_true, y_pred, average="samples")
    
    # 使用 pytest 检查是否会抛出 ValueError 异常，测试 f1_score 函数
    with pytest.raises(ValueError):
        f1_score(y_true, y_pred, average="samples")
    
    # 使用 pytest 检查是否会抛出 ValueError 异常，测试 fbeta_score 函数，设置 beta 值为 0.5
    with pytest.raises(ValueError):
        fbeta_score(y_true, y_pred, average="samples", beta=0.5)

    # 计算精确率、召回率、F1 分数以及支持数，使用指定的标签顺序进行计算
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 2, 1], average=None
    )
    
    # 检查计算得到的精确率 p 是否与期望值接近，精确度保留两位小数
    assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
    
    # 检查计算得到的召回率 r 是否与期望值接近，精确度保留两位小数
    assert_array_almost_equal(r, [0.79, 0.90, 0.10], 2)
    
    # 检查计算得到的 F1 分数 f 是否与期望值接近，精确度保留两位小数
    assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
    
    # 检查支持数 s 是否与期望值完全相等
    assert_array_equal(s, [24, 20, 31])
@pytest.mark.parametrize("average", ["samples", "micro", "macro", "weighted", None])
# 使用 pytest 的 parametrize 装饰器，为 average 参数提供多组测试参数
def test_precision_refcall_f1_score_multilabel_unordered_labels(average):
    # 测试多标签情况下，标签无需排序的情况
    y_true = np.array([[1, 1, 0, 0]])
    y_pred = np.array([[0, 0, 1, 1]])
    # 计算精确度、召回率、F1 分数和支持度
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[3, 0, 1, 2], warn_for=[], average=average
    )
    assert_array_equal(p, 0)
    assert_array_equal(r, 0)
    assert_array_equal(f, 0)
    if average is None:
        assert_array_equal(s, [0, 1, 1, 0])


def test_precision_recall_f1_score_binary_averaged():
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1])

    # 使用默认标签推断计算分数
    ps, rs, fs, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert p == np.mean(ps)
    assert r == np.mean(rs)
    assert f == np.mean(fs)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    support = np.bincount(y_true)
    assert p == np.average(ps, weights=support)
    assert r == np.average(rs, weights=support)
    assert f == np.average(fs, weights=support)


def test_zero_precision_recall():
    # 检查特殊情况下不会产生 NaN
    old_error_settings = np.seterr(all="raise")

    try:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([2, 0, 1, 1, 2, 0])

        assert_almost_equal(precision_score(y_true, y_pred, average="macro"), 0.0, 2)
        assert_almost_equal(recall_score(y_true, y_pred, average="macro"), 0.0, 2)
        assert_almost_equal(f1_score(y_true, y_pred, average="macro"), 0.0, 2)

    finally:
        np.seterr(**old_error_settings)


def test_confusion_matrix_multiclass_subset_labels():
    # 测试混淆矩阵 - 多类情况下使用部分标签
    y_true, y_pred, _ = make_prediction(binary=False)

    # 计算只考虑前两个标签的混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert_array_equal(cm, [[19, 4], [4, 3]])

    # 计算仅对子集标签使用明确的标签顺序的混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[2, 1])
    assert_array_equal(cm, [[18, 2], [24, 3]])

    # y_true 中不存在的标签应该导致相应行/列为零
    extra_label = np.max(y_true) + 1
    cm = confusion_matrix(y_true, y_pred, labels=[2, extra_label])
    assert_array_equal(cm, [[18, 0], [0, 0]])


@pytest.mark.parametrize(
    "labels, err_msg",
    [
        ([], "'labels' should contains at least one label."),
        ([3, 4], "At least one label specified must be in y_true"),
    ],
    ids=["empty list", "unknown labels"],
)
# 使用 pytest 的 parametrize 装饰器，为 labels 和 err_msg 参数提供多组测试参数，并定义相应的标识符
def test_confusion_matrix_error(labels, err_msg):
    # 使用 make_prediction 函数生成真实值（y_true）、预测值（y_pred）、以及一个无关的占位变量（_）
    y_true, y_pred, _ = make_prediction(binary=False)
    
    # 使用 pytest 模块中的 pytest.raises 上下文管理器来捕获 ValueError 异常，并检查其错误消息是否匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        # 调用 confusion_matrix 函数计算混淆矩阵，传入真实值 y_true、预测值 y_pred，以及自定义标签列表 labels
        confusion_matrix(y_true, y_pred, labels=labels)
@pytest.mark.parametrize(
    "labels", (None, [0, 1], [0, 1, 2]), ids=["None", "binary", "multiclass"]
)
# 使用 pytest 的 parametrize 标记来定义多个参数化测试用例，测试不同的 labels 值
def test_confusion_matrix_on_zero_length_input(labels):
    # 计算期望的类别数量，如果 labels 存在则为其长度，否则为 0
    expected_n_classes = len(labels) if labels else 0
    # 创建一个全零矩阵作为期望的混淆矩阵，数据类型为整数
    expected = np.zeros((expected_n_classes, expected_n_classes), dtype=int)
    # 调用 confusion_matrix 函数，传入空列表和空列表，并指定 labels
    cm = confusion_matrix([], [], labels=labels)
    # 断言混淆矩阵与期望矩阵相等
    assert_array_equal(cm, expected)


def test_confusion_matrix_dtype():
    y = [0, 1, 1]
    weight = np.ones(len(y))
    # confusion_matrix 默认返回 int64 类型的矩阵
    cm = confusion_matrix(y, y)
    # 断言混淆矩阵的数据类型为 np.int64
    assert cm.dtype == np.int64
    # 遍历多种数据类型，检查 confusion_matrix 的数据类型始终为 np.int64
    for dtype in [np.bool_, np.int32, np.uint64]:
        cm = confusion_matrix(y, y, sample_weight=weight.astype(dtype, copy=False))
        assert cm.dtype == np.int64
    # 遍历多种数据类型，检查 confusion_matrix 的数据类型是否为 np.float64
    for dtype in [np.float32, np.float64, None, object]:
        cm = confusion_matrix(y, y, sample_weight=weight.astype(dtype, copy=False))
        assert cm.dtype == np.float64

    # 使用 np.uint32 的最大值作为权重，检查累加是否正确
    weight = np.full(len(y), 4294967295, dtype=np.uint32)
    cm = confusion_matrix(y, y, sample_weight=weight)
    # 断言混淆矩阵中的特定元素值为 4294967295
    assert cm[0, 0] == 4294967295
    assert cm[1, 1] == 8589934590

    # 使用 np.int64 的最大值作为权重，检查是否发生溢出
    weight = np.full(len(y), 9223372036854775807, dtype=np.int64)
    cm = confusion_matrix(y, y, sample_weight=weight)
    # 断言混淆矩阵中的特定元素值为 9223372036854775807 和 -2
    assert cm[0, 0] == 9223372036854775807
    assert cm[1, 1] == -2


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
# 使用 pytest 的 parametrize 标记来定义不同的 dtype 参数化测试用例
def test_confusion_matrix_pandas_nullable(dtype):
    """Checks that confusion_matrix works with pandas nullable dtypes.

    Non-regression test for gh-25635.
    """
    pd = pytest.importorskip("pandas")

    y_ndarray = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1])
    # 创建一个 pandas 的 Series 对象，使用指定的 dtype
    y_true = pd.Series(y_ndarray, dtype=dtype)
    y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype="int64")

    # 调用 confusion_matrix 计算输出
    output = confusion_matrix(y_true, y_predicted)
    # 使用原始的 ndarray 和预测结果计算期望输出
    expected_output = confusion_matrix(y_ndarray, y_predicted)

    # 断言输出与期望输出相等
    assert_array_equal(output, expected_output)


def test_classification_report_multiclass():
    # 测试多类别分类报告
    iris = datasets.load_iris()
    # 进行预测，并获取真实标签、预测标签和额外信息
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # 打印带有类名的分类报告
    expected_report = """\
              precision    recall  f1-score   support

      setosa       0.83      0.79      0.81        24
  versicolor       0.33      0.10      0.15        31
   virginica       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    # 调用 classification_report 函数生成报告
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
    )
    # 断言生成的报告与预期报告相等
    assert report == expected_report


def test_classification_report_multiclass_balanced():
    # 这是一个未完整的测试函数，用于测试多类别平衡的分类报告
    # 定义真实标签和预测标签的列表
    y_true, y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]
    
    # 定义期望的分类报告，包含精确率、召回率、F1 分数和支持度等信息
    expected_report = """\
              precision    recall  f1-score   support
    
           0       0.33      0.33      0.33         3
           1       0.33      0.33      0.33         3
           2       0.33      0.33      0.33         3
    
    accuracy                           0.33         9
    macro avg       0.33      0.33      0.33         9
    """
# 定义测试函数，验证分类报告与预期报告是否一致
def test_classification_report_multiclass_with_label_detection():
    # 载入鸢尾花数据集
    iris = datasets.load_iris()
    # 生成预测结果与真实标签，不是二元分类
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # 期望的分类报告字符串，包含各类别的精确度、召回率、F1 分数和支持数
    expected_report = """\
              precision    recall  f1-score   support

           0       0.83      0.79      0.81        24
           1       0.33      0.10      0.15        31
           2       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    # 生成实际的分类报告
    report = classification_report(y_true, y_pred)
    # 断言实际的分类报告与期望的分类报告一致
    assert report == expected_report
    # 生成分类报告，基于真实标签和预测标签，指定目标类别名称为 ["a", "b", "c"]
    report = classification_report(y_true, y_pred, target_names=["a", "b", "c"])
    # 使用断言检查生成的分类报告是否与期望的报告一致
    assert report == expected_report
# 定义一个测试函数，用于测试多类别分类报告中包含 Unicode 标签的情况
def test_classification_report_multiclass_with_unicode_label():
    # 调用 make_prediction 函数获取真实标签、预测标签和其他信息
    y_true, y_pred, _ = make_prediction(binary=False)

    # 创建包含 Unicode 标签的数组
    labels = np.array(["blue\xa2", "green\xa2", "red\xa2"])
    # 使用真实标签替换 y_true 中的索引，使 y_true 包含 Unicode 标签
    y_true = labels[y_true]
    # 使用预测标签替换 y_pred 中的索引，使 y_pred 包含 Unicode 标签
    y_pred = labels[y_pred]

    # 预期的分类报告，包含了 Unicode 标签的精度、召回率、F1 分数和支持度
    expected_report = """\
              precision    recall  f1-score   support

       blue\xa2       0.83      0.79      0.81        24
      green\xa2       0.33      0.10      0.15        31
        red\xa2       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    # 调用 classification_report 函数生成实际的分类报告
    report = classification_report(y_true, y_pred)
    # 使用断言验证实际的分类报告与预期的分类报告是否一致
    assert report == expected_report


# 定义一个测试函数，用于测试多类别分类报告中包含长字符串标签的情况
def test_classification_report_multiclass_with_long_string_label():
    # 调用 make_prediction 函数获取真实标签、预测标签和其他信息
    y_true, y_pred, _ = make_prediction(binary=False)

    # 创建包含长字符串标签的数组
    labels = np.array(["blue", "green" * 5, "red"])
    # 使用真实标签替换 y_true 中的索引，使 y_true 包含长字符串标签
    y_true = labels[y_true]
    # 使用预测标签替换 y_pred 中的索引，使 y_pred 包含长字符串标签
    y_pred = labels[y_pred]

    # 预期的分类报告，包含了长字符串标签的精度、召回率、F1 分数和支持度
    expected_report = """\
                           precision    recall  f1-score   support

                     blue       0.83      0.79      0.81        24
greengreengreengreengreen       0.33      0.10      0.15        31
                      red       0.42      0.90      0.57        20

                 accuracy                           0.53        75
                macro avg       0.53      0.60      0.51        75
             weighted avg       0.51      0.53      0.47        75
"""

    # 调用 classification_report 函数生成实际的分类报告
    report = classification_report(y_true, y_pred)
    # 使用断言验证实际的分类报告与预期的分类报告是否一致
    assert report == expected_report


# 定义一个测试函数，用于测试分类报告中标签和目标名称长度不一致的情况
def test_classification_report_labels_target_names_unequal_length():
    # 创建真实标签和预测标签列表
    y_true = [0, 0, 2, 0, 0]
    y_pred = [0, 2, 2, 0, 0]
    # 创建目标名称列表，包含三个元素
    target_names = ["class 0", "class 1", "class 2"]

    # 预期的警告信息，指示标签的长度与目标名称的长度不匹配
    msg = "labels size, 2, does not match size of target_names, 3"
    # 使用 pytest.warns 检查是否会发出 UserWarning，并匹配预期的警告信息
    with pytest.warns(UserWarning, match=msg):
        # 调用 classification_report 函数，指定标签和目标名称进行生成分类报告
        classification_report(y_true, y_pred, labels=[0, 2], target_names=target_names)


# 定义一个测试函数，用于测试分类报告中没有指定标签和目标名称长度不一致的情况
def test_classification_report_no_labels_target_names_unequal_length():
    # 创建真实标签和预测标签列表
    y_true = [0, 0, 2, 0, 0]
    y_pred = [0, 2, 2, 0, 0]
    # 创建目标名称列表，包含三个元素
    target_names = ["class 0", "class 1", "class 2"]

    # 预期的错误信息，指示类别的数量与目标名称的数量不匹配，建议指定标签参数
    err_msg = (
        "Number of classes, 2, does not "
        "match size of target_names, 3. "
        "Try specifying the labels parameter"
    )
    # 使用 pytest.raises 检查是否会抛出 ValueError，并匹配预期的错误信息
    with pytest.raises(ValueError, match=err_msg):
        # 调用 classification_report 函数，未指定标签，但指定了目标名称，生成分类报告
        classification_report(y_true, y_pred, target_names=target_names)


# 定义一个装饰器函数，用于忽略警告的测试函数
@ignore_warnings
def test_multilabel_classification_report():
    # 设置类别数量和样本数量
    n_classes = 4
    n_samples = 50

    # 调用 make_multilabel_classification 函数生成多标签分类数据集的真实标签
    _, y_true = make_multilabel_classification(
        n_features=1, n_samples=n_samples, n_classes=n_classes, random_state=0
    )

    # 调用 make_multilabel_classification 函数生成多标签分类数据集的预测标签
    _, y_pred = make_multilabel_classification(
        n_features=1, n_samples=n_samples, n_classes=n_classes, random_state=1
    )
    # 定义一个预期的分类报告，包含精度、召回率、F1 分数和支持度等指标
    expected_report = """\
              precision    recall  f1-score   support

           0       0.50      0.67      0.57        24
           1       0.51      0.74      0.61        27
           2       0.29      0.08      0.12        26
           3       0.52      0.56      0.54        27

   micro avg       0.50      0.51      0.50       104
   macro avg       0.45      0.51      0.46       104

"""
    report = classification_report(y_true, y_pred)
    # 生成分类报告，包括精确度、召回率、F1 值等
    assert report == expected_report


def test_multilabel_zero_one_loss_subset():
    # Dense label indicator matrix format
    # 创建稠密标签指示矩阵格式的示例数据
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    # 验证 zero-one loss 函数的预期行为
    assert zero_one_loss(y1, y2) == 0.5
    assert zero_one_loss(y1, y1) == 0
    assert zero_one_loss(y2, y2) == 0
    assert zero_one_loss(y2, np.logical_not(y2)) == 1
    assert zero_one_loss(y1, np.logical_not(y1)) == 1
    assert zero_one_loss(y1, np.zeros(y1.shape)) == 1
    assert zero_one_loss(y2, np.zeros(y1.shape)) == 1


def test_multilabel_hamming_loss():
    # Dense label indicator matrix format
    # 创建稠密标签指示矩阵格式的示例数据
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])
    w = np.array([1, 3])

    # 验证 Hamming loss 函数的预期行为，包括加权的情况
    assert hamming_loss(y1, y2) == 1 / 6
    assert hamming_loss(y1, y1) == 0
    assert hamming_loss(y2, y2) == 0
    assert hamming_loss(y2, 1 - y2) == 1
    assert hamming_loss(y1, 1 - y1) == 1
    assert hamming_loss(y1, np.zeros(y1.shape)) == 4 / 6
    assert hamming_loss(y2, np.zeros(y1.shape)) == 0.5
    assert hamming_loss(y1, y2, sample_weight=w) == 1.0 / 12
    assert hamming_loss(y1, 1 - y2, sample_weight=w) == 11.0 / 12
    assert hamming_loss(y1, np.zeros_like(y1), sample_weight=w) == 2.0 / 3
    # sp_hamming only works with 1-D arrays
    assert hamming_loss(y1[0], y2[0]) == sp_hamming(y1[0], y2[0])


def test_jaccard_score_validation():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])

    # 验证在二元分类设置下，Jaccard score 函数的异常处理
    err_msg = r"pos_label=2 is not a valid label. It should be one of \[0, 1\]"
    with pytest.raises(ValueError, match=err_msg):
        jaccard_score(y_true, y_pred, average="binary", pos_label=2)

    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])

    # 验证在二元分类设置下，Jaccard score 函数的异常处理
    msg1 = (
        r"Target is multilabel-indicator but average='binary'. "
        r"Please choose another average setting, one of \[None, "
        r"'micro', 'macro', 'weighted', 'samples'\]."
    )
    with pytest.raises(ValueError, match=msg1):
        jaccard_score(y_true, y_pred, average="binary", pos_label=-1)

    y_true = np.array([0, 1, 1, 0, 2])
    y_pred = np.array([1, 1, 1, 1, 0])

    # 验证在二元分类设置下，Jaccard score 函数的异常处理
    msg2 = (
        r"Target is multiclass but average='binary'. Please choose "
        r"another average setting, one of \[None, 'micro', 'macro', "
        r"'weighted'\]."
    )
    with pytest.raises(ValueError, match=msg2):
        jaccard_score(y_true, y_pred, average="binary")

    # 验证在样本度量不适用于非多标签分类情况下的异常处理
    msg3 = "Samplewise metrics are not available outside of multilabel classification."
    with pytest.raises(ValueError, match=msg3):
        jaccard_score(y_true, y_pred, average="samples")
    # 创建包含多行警告消息的字符串，用于在测试中匹配警告
    msg = (
        r"Note that pos_label \(set to 3\) is ignored when "
        r"average != 'binary' \(got 'micro'\). You may use "
        r"labels=\[pos_label\] to specify a single positive "
        "class."
    )
    
    # 使用 pytest 模块来捕获特定类型的警告，并验证是否触发指定的警告消息
    with pytest.warns(UserWarning, match=msg):
        # 调用 jaccard_score 函数，使用 "micro" 平均方式计算 Jaccard 分数，
        # 并指定 pos_label 为 3，期望触发警告消息
        jaccard_score(y_true, y_pred, average="micro", pos_label=3)
def test_multilabel_jaccard_score(recwarn):
    # Dense label indicator matrix format
    # 创建两个示例的稠密标签指示矩阵
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    # size(y1 \inter y2) = [1, 2]
    # size(y1 \union y2) = [2, 2]

    # 断言不同参数下的 Jaccard 分数
    assert jaccard_score(y1, y2, average="samples") == 0.75
    assert jaccard_score(y1, y1, average="samples") == 1
    assert jaccard_score(y2, y2, average="samples") == 1
    assert jaccard_score(y2, np.logical_not(y2), average="samples") == 0
    assert jaccard_score(y1, np.logical_not(y1), average="samples") == 0
    assert jaccard_score(y1, np.zeros(y1.shape), average="samples") == 0
    assert jaccard_score(y2, np.zeros(y1.shape), average="samples") == 0

    # 创建另外两个示例的真实与预测标签矩阵
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])

    # average='macro'
    # 断言不同平均方式下的 Jaccard 分数
    assert_almost_equal(jaccard_score(y_true, y_pred, average="macro"), 2.0 / 3)
    # average='micro'
    assert_almost_equal(jaccard_score(y_true, y_pred, average="micro"), 3.0 / 5)
    # average='samples'
    assert_almost_equal(jaccard_score(y_true, y_pred, average="samples"), 7.0 / 12)
    assert_almost_equal(
        jaccard_score(y_true, y_pred, average="samples", labels=[0, 2]), 1.0 / 2
    )
    assert_almost_equal(
        jaccard_score(y_true, y_pred, average="samples", labels=[1, 2]), 1.0 / 2
    )
    # average=None
    assert_array_equal(
        jaccard_score(y_true, y_pred, average=None), np.array([1.0 / 2, 1.0, 1.0 / 2])
    )

    # 再次使用相同的真实与预测标签矩阵
    y_true = np.array([[0, 1, 1], [1, 0, 1]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    # average='macro'
    assert_almost_equal(jaccard_score(y_true, y_pred, average="macro"), 5.0 / 6)
    # average='weighted'
    assert_almost_equal(jaccard_score(y_true, y_pred, average="weighted"), 7.0 / 8)

    # 测试异常情况下的错误提示信息
    msg2 = "Got 4 > 2"
    with pytest.raises(ValueError, match=msg2):
        jaccard_score(y_true, y_pred, labels=[4], average="macro")
    msg3 = "Got -1 < 0"
    with pytest.raises(ValueError, match=msg3):
        jaccard_score(y_true, y_pred, labels=[-1], average="macro")

    # 测试警告提示信息
    msg = (
        "Jaccard is ill-defined and being set to 0.0 in labels "
        "with no true or predicted samples."
    )
    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert (
            jaccard_score(np.array([[0, 1]]), np.array([[0, 1]]), average="macro")
            == 0.5
        )

    msg = (
        "Jaccard is ill-defined and being set to 0.0 in samples "
        "with no true or predicted labels."
    )
    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert (
            jaccard_score(
                np.array([[0, 0], [1, 1]]),
                np.array([[0, 0], [1, 1]]),
                average="samples",
            )
            == 0.5
        )

    # 确认没有警告被触发
    assert not list(recwarn)


def test_multiclass_jaccard_score(recwarn):
    # 创建示例的多类别真实与预测标签列表
    y_true = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "bird"]
    y_pred = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "cat"]
    # 定义标签列表
    labels = ["ant", "bird", "cat"]
    # 创建标签二值化器对象
    lb = LabelBinarizer()
    # 用标签列表来训练标签二值化器
    lb.fit(labels)
    # 将真实标签转换为二值化格式
    y_true_bin = lb.transform(y_true)
    # 将预测标签转换为二值化格式
    y_pred_bin = lb.transform(y_pred)
    # 创建多标签 Jaccard 分数计算的偏函数，固定了 y_true 和 y_pred 参数
    multi_jaccard_score = partial(jaccard_score, y_true, y_pred)
    # 创建二值化 Jaccard 分数计算的偏函数，固定了 y_true_bin 和 y_pred_bin 参数
    bin_jaccard_score = partial(jaccard_score, y_true_bin, y_pred_bin)
    # 多标签组合列表，包含各种标签组合的列表
    multi_labels_list = [
        ["ant", "bird"],
        ["ant", "cat"],
        ["cat", "bird"],
        ["ant"],
        ["bird"],
        ["cat"],
        None,
    ]
    # 二值化标签组合列表，包含各种标签组合的列表
    bin_labels_list = [[0, 1], [0, 2], [2, 1], [0], [1], [2], None]
    
    # 对于每个平均值类型进行迭代，测试除了 'samples'/'none-samples' 之外的所有情况
    for average in ("macro", "weighted", "micro", None):
        # 对多标签列表和二值化标签列表进行并行迭代
        for m_label, b_label in zip(multi_labels_list, bin_labels_list):
            # 使用多标签 Jaccard 分数偏函数计算分数，使用给定的平均值类型和标签
            assert_almost_equal(
                multi_jaccard_score(average=average, labels=m_label),
                # 使用二值化 Jaccard 分数偏函数计算分数，使用给定的平均值类型和标签
                bin_jaccard_score(average=average, labels=b_label),
            )
    
    # 设置新的真实和预测标签数组
    y_true = np.array([[0, 0], [0, 0], [0, 0]])
    y_pred = np.array([[0, 0], [0, 0], [0, 0]])
    # 忽略警告后，使用加权平均计算 Jaccard 分数，确保结果为零
    with ignore_warnings():
        assert jaccard_score(y_true, y_pred, average="weighted") == 0
    
    # 断言没有警告被触发
    assert not list(recwarn)
def test_average_binary_jaccard_score(recwarn):
    # tp=0, fp=0, fn=1, tn=0
    # 检验平均二进制Jaccard分数，预期为0.0
    assert jaccard_score([1], [0], average="binary") == 0.0

    # tp=0, fp=0, fn=0, tn=1
    # 检验平均二进制Jaccard分数，由于没有真实或预测样本，Jaccard不确定，被设为0.0
    msg = (
        "Jaccard is ill-defined and being set to 0.0 due to "
        "no true or predicted samples"
    )
    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert jaccard_score([0, 0], [0, 0], average="binary") == 0.0

    # tp=1, fp=0, fn=0, tn=0 (pos_label=0)
    # 使用pos_label=0检验平均二进制Jaccard分数，预期为1.0
    assert jaccard_score([0], [0], pos_label=0, average="binary") == 1.0

    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 1])
    # 检验平均二进制Jaccard分数，预期为3/4
    assert_almost_equal(jaccard_score(y_true, y_pred, average="binary"), 3.0 / 4)
    # 使用pos_label=0检验平均二进制Jaccard分数，预期为1/2
    assert_almost_equal(
        jaccard_score(y_true, y_pred, average="binary", pos_label=0), 1.0 / 2
    )

    assert not list(recwarn)


def test_jaccard_score_zero_division_warning():
    # 检查默认行为下，当发生零除法时是否会触发警告
    y_true = np.array([[1, 0, 1], [0, 0, 0]])
    y_pred = np.array([[0, 0, 0], [0, 0, 0]])
    msg = (
        "Jaccard is ill-defined and being set to 0.0 in "
        "samples with no true or predicted labels."
        " Use `zero_division` parameter to control this behavior."
    )
    with pytest.warns(UndefinedMetricWarning, match=msg):
        score = jaccard_score(y_true, y_pred, average="samples", zero_division="warn")
        assert score == pytest.approx(0.0)


@pytest.mark.parametrize("zero_division, expected_score", [(0, 0), (1, 0.5)])
def test_jaccard_score_zero_division_set_value(zero_division, expected_score):
    # 检查通过传递zero_division参数，确保不会发出警告
    y_true = np.array([[1, 0, 1], [0, 0, 0]])
    y_pred = np.array([[0, 0, 0], [0, 0, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndefinedMetricWarning)
        score = jaccard_score(
            y_true, y_pred, average="samples", zero_division=zero_division
        )
    assert score == pytest.approx(expected_score)


@ignore_warnings
def test_precision_recall_f1_score_multilabel_1():
    # 在一个人工构建的多标签示例上测试precision_recall_f1_score
    # 第一个人工构建的例子

    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    y_pred = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]])

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)

    # tp = [0, 1, 1, 0]
    # fn = [1, 0, 0, 1]
    # fp = [1, 1, 0, 0]
    # 检查每个类别的指标

    assert_array_almost_equal(p, [0.0, 0.5, 1.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 1.0, 1.0, 0.0], 2)
    assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
    assert_array_almost_equal(s, [1, 1, 1, 1], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    support = s
    assert_array_almost_equal(f2, [0, 0.83, 1, 0], 2)

    # 检查宏平均
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro")
    # 检查准确率、召回率和 F-score 是否接近预期值，对于宏平均
    assert_almost_equal(p, 1.5 / 4)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2.5 / 1.5 * 0.25)
    # 确认样本集 s 为 None
    assert s is None
    # 检查 F-beta score 是否接近预期值，对于宏平均
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
    )

    # 检查准确率、召回率和 F-score 是否接近预期值，对于微平均
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
    assert_almost_equal(p, 0.5)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 0.5)
    # 确认样本集 s 为 None
    assert s is None
    # 检查 F-beta score 是否接近预期值，对于微平均
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="micro"),
        (1 + 4) * p * r / (4 * p + r),
    )

    # 检查准确率、召回率和 F-score 是否接近预期值，对于加权平均
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    assert_almost_equal(p, 1.5 / 4)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2.5 / 1.5 * 0.25)
    # 确认样本集 s 为 None
    assert s is None
    # 检查 F-beta score 是否接近预期值，对于加权平均
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        np.average(f2, weights=support),
    )

    # 检查准确率、召回率和 F-score 是否接近预期值，对于样本平均
    # |h(x_i) inter y_i | = [0, 1, 1]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [1, 1, 2]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    assert_almost_equal(p, 0.5)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 0.5)
    # 确认样本集 s 为 None
    assert s is None
    # 检查 F-beta score 是否接近预期值，对于样本平均
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average="samples"), 0.5)
@ignore_warnings
# 定义一个测试函数，用于测试 precision_recall_f1_score 在一个构造的多标签示例上的表现（第二个例子）
def test_precision_recall_f1_score_multilabel_2():
    # 创建真实标签和预测标签的 NumPy 数组
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]])

    # 计算 precision、recall、f-score 和 support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    # 断言计算得到的 precision 数组与期望值接近
    assert_array_almost_equal(p, [0.0, 1.0, 0.0, 0.0], 2)
    # 断言计算得到的 recall 数组与期望值接近
    assert_array_almost_equal(r, [0.0, 0.5, 0.0, 0.0], 2)
    # 断言计算得到的 f-score 数组与期望值接近
    assert_array_almost_equal(f, [0.0, 0.66, 0.0, 0.0], 2)
    # 断言计算得到的 support 数组与期望值接近
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)

    # 计算 beta=2 的 f-score
    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    # 将 support 赋值给变量
    support = s
    # 断言计算得到的 f2 数组与期望值接近
    assert_array_almost_equal(f2, [0, 0.55, 0, 0], 2)

    # 计算 micro 平均的 precision、recall、f-score 和 support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
    # 断言计算得到的 micro 平均 precision 与期望值接近
    assert_almost_equal(p, 0.25)
    # 断言计算得到的 micro 平均 recall 与期望值接近
    assert_almost_equal(r, 0.25)
    # 断言计算得到的 micro 平均 f-score 与期望值接近
    assert_almost_equal(f, 2 * 0.25 * 0.25 / 0.5)
    # 断言 support 为 None
    assert s is None
    # 断言计算得到的 micro 平均 f-beta score 与期望值接近
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="micro"),
        (1 + 4) * p * r / (4 * p + r),
    )

    # 计算 macro 平均的 precision、recall、f-score 和 support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro")
    # 断言计算得到的 macro 平均 precision 与期望值接近
    assert_almost_equal(p, 0.25)
    # 断言计算得到的 macro 平均 recall 与期望值接近
    assert_almost_equal(r, 0.125)
    # 断言计算得到的 macro 平均 f-score 与期望值接近
    assert_almost_equal(f, 2 / 12)
    # 断言 support 为 None
    assert s is None
    # 断言计算得到的 macro 平均 f-beta score 与期望值接近
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
    )

    # 计算 weighted 平均的 precision、recall、f-score 和 support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    # 断言计算得到的 weighted 平均 precision 与期望值接近
    assert_almost_equal(p, 2 / 4)
    # 断言计算得到的 weighted 平均 recall 与期望值接近
    assert_almost_equal(r, 1 / 4)
    # 断言计算得到的 weighted 平均 f-score 与期望值接近
    assert_almost_equal(f, 2 / 3 * 2 / 4)
    # 断言 support 为 None
    assert s is None
    # 断言计算得到的 weighted 平均 f-beta score 与期望值接近
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        np.average(f2, weights=support),
    )

    # 计算 samples 平均的 precision、recall、f-score 和 support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    # 断言计算得到的 samples 平均 precision 与期望值接近
    assert_almost_equal(p, 1 / 6)
    # 断言计算得到的 samples 平均 recall 与期望值接近
    assert_almost_equal(r, 1 / 6)
    # 断言计算得到的 samples 平均 f-score 与期望值接近
    assert_almost_equal(f, 2 / 4 * 1 / 3)
    # 断言 support 为 None
    assert s is None
    # 断言计算得到的 samples 平均 f-beta score 与期望值接近
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="samples"), 0.1666, 2
    )


@ignore_warnings
@pytest.mark.parametrize(
    "zero_division, zero_division_expected",
    [("warn", 0), (0, 0), (1, 1), (np.nan, np.nan)],
)
# 使用参数化测试标记，测试 precision_recall_f1_score 在空预测情况下的表现
def test_precision_recall_f1_score_with_an_empty_prediction(
    zero_division, zero_division_expected
):
    # 创建真实标签和预测标签的 NumPy 数组
    y_true = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])

    # 计算 precision、recall、f-score 和 support，可以设置 zero_division 的行为
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=zero_division
    )

    # 断言计算得到的 precision 数组与期望值接近
    assert_array_almost_equal(p, [zero_division_expected, 1.0, 1.0, 0.0], 2)
    # 对 r 数组进行准确度断言，期望结果包括 0.0, 0.5, 1.0 以及 zero_division_expected
    assert_array_almost_equal(r, [0.0, 0.5, 1.0, zero_division_expected], 2)
    # 初始化预期的 f 值为 0
    expected_f = 0
    # 对 f 数组进行准确度断言，期望结果包括 expected_f, 1 / 1.5, 1, 以及 expected_f
    assert_array_almost_equal(f, [expected_f, 1 / 1.5, 1, expected_f], 2)
    # 对 s 数组进行准确度断言，期望结果为 [1, 2, 1, 0]
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)

    # 计算 F2 分数，使用 F-beta 分数评估函数，参数设置 beta=2, average=None, zero_division=zero_division
    f2 = fbeta_score(y_true, y_pred, beta=2, average=None, zero_division=zero_division)
    # 将 s 赋值给 support
    support = s
    # 对 f2 数组进行准确度断言，期望结果包括 expected_f, 0.55, 1, 以及 expected_f
    assert_array_almost_equal(f2, [expected_f, 0.55, 1, expected_f], 2)

    # 计算宏平均的精确率 (p), 召回率 (r), F 值 (f), 支持度 (s)，使用 precision_recall_fscore_support 函数
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=zero_division
    )
    # 计算用于求和的值，如果 zero_division_expected 是 NaN 则为 0，否则为 zero_division_expected
    value_to_sum = 0 if np.isnan(zero_division_expected) else zero_division_expected
    # 计算用于求平均的值，初始值为 3，如果 zero_division_expected 不是 NaN，则再加 1
    values_to_average = 3 + (not np.isnan(zero_division_expected))
    # 对精确率 p 进行准确度断言，期望结果是 (2 + value_to_sum) / values_to_average
    assert_almost_equal(p, (2 + value_to_sum) / values_to_average)
    # 对召回率 r 进行准确度断言，期望结果是 (1.5 + value_to_sum) / values_to_average
    assert_almost_equal(r, (1.5 + value_to_sum) / values_to_average)
    # 计算预期的 F 值
    expected_f = (2 / 3 + 1) / 4
    # 对 F 值 f 进行准确度断言，期望结果是 expected_f
    assert_almost_equal(f, expected_f)
    # 断言 s 为 None
    assert s is None
    # 对宏平均 F2 分数进行准确度断言，使用 fbeta_score 函数计算，参数设置 beta=2, average="macro", zero_division=zero_division
    assert_almost_equal(
        fbeta_score(
            y_true,
            y_pred,
            beta=2,
            average="macro",
            zero_division=zero_division,
        ),
        _nanaverage(f2, weights=None),
    )

    # 计算微平均的精确率 (p), 召回率 (r), F 值 (f), 支持度 (s)，使用 precision_recall_fscore_support 函数
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=zero_division
    )
    # 对精确率 p 进行准确度断言，期望结果是 2 / 3
    assert_almost_equal(p, 2 / 3)
    # 对召回率 r 进行准确度断言，期望结果是 0.5
    assert_almost_equal(r, 0.5)
    # 对 F 值 f 进行准确度断言，期望结果是 2 / 3 / (2 / 3 + 0.5)
    assert_almost_equal(f, 2 / 3 / (2 / 3 + 0.5))
    # 断言 s 为 None
    assert s is None
    # 对微平均 F2 分数进行准确度断言，使用 fbeta_score 函数计算，参数设置 beta=2, average="micro", zero_division=zero_division
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="micro", zero_division=zero_division
        ),
        (1 + 4) * p * r / (4 * p + r),
    )

    # 计算加权平均的精确率 (p), 召回率 (r), F 值 (f), 支持度 (s)，使用 precision_recall_fscore_support 函数
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=zero_division
    )
    # 对精确率 p 进行准确度断言，期望结果是 3 / 4 如果 zero_division_expected 等于 0，否则为 1.0
    assert_almost_equal(p, 3 / 4 if zero_division_expected == 0 else 1.0)
    # 对召回率 r 进行准确度断言，期望结果是 0.5
    assert_almost_equal(r, 0.5)
    # 计算 values_to_average 的值为 4
    values_to_average = 4
    # 对 F 值 f 进行准确度断言，期望结果是 (2 * 2 / 3 + 1) / values_to_average
    assert_almost_equal(f, (2 * 2 / 3 + 1) / values_to_average)
    # 断言 s 为 None
    assert s is None
    # 对加权平均 F2 分数进行准确度断言，使用 fbeta_score 函数计算，参数设置 beta=2, average="weighted", zero_division=zero_division
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="weighted", zero_division=zero_division
        ),
        _nanaverage(f2, weights=support),
    )

    # 计算样本平均的精确率 (p), 召回率 (r), F 值 (f), 支持度 (s)，使用 precision_recall_fscore_support 函数
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    # 对精确率 p 进行准确度断言，期望结果是 1 / 3
    assert_almost_equal(p, 1 / 3)
    # 对召回率 r 进行准确度断言，期望结果是 1 / 3
    assert_almost_equal(r, 1 / 3)
    # 对 F 值 f 进行准确度断言，期望结果是 1 / 3
    assert_almost_equal(f, 1 / 3)
    # 断言 s 为 None
    assert s is None
    # 计算预期结果值为 0.333
    expected_result = 0.333
    # 对样本平均 F2 分数进行准确度断言，使用 fbeta_score 函数计算，参数设置 beta=2, average="samples", zero_division=zero_division
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="samples", zero_division=zero_division
        ),
        expected_result,
        2,
    )
@pytest.mark.parametrize("beta", [1])
@pytest.mark.parametrize("average", ["macro", "micro", "weighted", "samples"])
@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
# 定义测试函数，参数化三个参数：beta, average, zero_division
def test_precision_recall_f1_no_labels(beta, average, zero_division):
    # 创建一个全零的二维数组作为真实标签和预测标签
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # 调用 assert_no_warnings 函数计算精确率、召回率、F1 分数和支持数
    p, r, f, s = assert_no_warnings(
        precision_recall_fscore_support,
        y_true,
        y_pred,
        average=average,
        beta=beta,
        zero_division=zero_division,
    )
    # 调用 assert_no_warnings 函数计算 F-beta 分数
    fbeta = assert_no_warnings(
        fbeta_score,
        y_true,
        y_pred,
        beta=beta,
        average=average,
        zero_division=zero_division,
    )
    # 断言支持数 s 应为 None
    assert s is None

    # 如果 zero_division 是 NaN，则检查所有的指标是否均为 NaN，然后退出
    if np.isnan(zero_division):
        for metric in [p, r, f, fbeta]:
            assert np.isnan(metric)
        return

    # 将 zero_division 转换为 float 类型
    zero_division = float(zero_division)
    # 断言精确率、召回率、F1 分数与 zero_division 几乎相等
    assert_almost_equal(p, zero_division)
    assert_almost_equal(r, zero_division)
    assert_almost_equal(f, zero_division)

    assert_almost_equal(fbeta, float(zero_division))


@pytest.mark.parametrize("average", ["macro", "micro", "weighted", "samples"])
# 参数化 average 参数的测试函数
def test_precision_recall_f1_no_labels_check_warnings(average):
    # 创建一个全零的二维数组作为真实标签和预测标签
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # 使用 precision_recall_fscore_support 函数计算精确率、召回率、F1 分数和支持数，并捕获 UndefinedMetricWarning
    func = precision_recall_fscore_support
    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = func(y_true, y_pred, average=average, beta=1.0)

    # 断言精确率、召回率、F1 分数几乎为 0，支持数为 None
    assert_almost_equal(p, 0)
    assert_almost_equal(r, 0)
    assert_almost_equal(f, 0)
    assert s is None

    # 使用 fbeta_score 函数计算 F-beta 分数，并捕获 UndefinedMetricWarning
    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, average=average, beta=1.0)

    # 断言 F-beta 分数几乎为 0
    assert_almost_equal(fbeta, 0)


@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
# 参数化 zero_division 参数的测试函数
def test_precision_recall_f1_no_labels_average_none(zero_division):
    # 创建一个全零的二维数组作为真实标签和预测标签
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # 使用 assert_no_warnings 函数计算精确率、召回率、F1 分数和支持数，average 参数为 None
    p, r, f, s = assert_no_warnings(
        precision_recall_fscore_support,
        y_true,
        y_pred,
        average=None,
        beta=1.0,
        zero_division=zero_division,
    )
    # 使用 assert_no_warnings 函数计算 F-beta 分数，average 参数为 None
    fbeta = assert_no_warnings(
        fbeta_score, y_true, y_pred, beta=1.0, average=None, zero_division=zero_division
    )
    # 将 zero_division 转换为 np.float64 类型
    zero_division = np.float64(zero_division)
    # 断言精确率、召回率、F1 分数数组与 zero_division 数组几乎相等，保留两位小数
    assert_array_almost_equal(p, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(r, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(f, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(s, [0, 0, 0], 2)

    assert_array_almost_equal(fbeta, [zero_division, zero_division, zero_division], 2)


# 定义 average 参数为 None 的测试函数
def test_precision_recall_f1_no_labels_average_none_warn():
    # 创建一个全零的二维数组作为真实标签
    y_true = np.zeros((20, 3))
    # 初始化一个与 y_true 相同形状的全零数组作为预测结果 y_pred
    y_pred = np.zeros_like(y_true)

    # 初始化各种评估指标的零数组，每个数组都有三个元素，对应三个类别
    # tp 表示 True Positive 数组
    # fn 表示 False Negative 数组
    # fp 表示 False Positive 数组
    # support 表示每个类别的支持数数组
    # |y_hat_i inter y_i | 表示每个类别预测与真实标签交集的大小数组
    # |y_i| 表示每个类别真实标签的大小数组
    # |y_hat_i| 表示每个类别预测标签的大小数组
    tp = [0, 0, 0]
    fn = [0, 0, 0]
    fp = [0, 0, 0]
    support = [0, 0, 0]
    |y_hat_i inter y_i | = [0, 0, 0]
    |y_i| = [0, 0, 0]
    |y_hat_i| = [0, 0, 0]

    # 使用 pytest 捕获 UndefinedMetricWarning 异常
    with pytest.warns(UndefinedMetricWarning):
        # 计算 precision、recall、f1-score 和 support 的值，针对每个类别分别计算
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, average=None, beta=1
        )

    # 断言计算得到的 precision、recall、f1-score 和 support 的数组与全零数组非常接近（精度为小数点后两位）
    assert_array_almost_equal(p, [0, 0, 0], 2)
    assert_array_almost_equal(r, [0, 0, 0], 2)
    assert_array_almost_equal(f, [0, 0, 0], 2)
    assert_array_almost_equal(s, [0, 0, 0], 2)

    # 再次使用 pytest 捕获 UndefinedMetricWarning 异常
    with pytest.warns(UndefinedMetricWarning):
        # 计算 F-beta 分数，这里 beta 设为 1，针对每个类别分别计算
        fbeta = fbeta_score(y_true, y_pred, beta=1, average=None)

    # 断言计算得到的 F-beta 分数数组与全零数组非常接近（精度为小数点后两位）
    assert_array_almost_equal(fbeta, [0, 0, 0], 2)
# 定义一个测试函数，用于测试 precision_recall_fscore_support 函数在不同情况下的警告行为
def test_prf_warnings():
    # 使用 precision_recall_fscore_support 函数和 UndefinedMetricWarning 警告类型
    f, w = precision_recall_fscore_support, UndefinedMetricWarning
    
    # 遍历三种不同的 average 参数值：None, "weighted", "macro"
    for average in [None, "weighted", "macro"]:
        # 设置 Precision 不确定警告的消息内容
        msg = (
            "Precision is ill-defined and "
            "being set to 0.0 in labels with no predicted samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        # 使用 pytest 的 warns 方法捕获 UndefinedMetricWarning 并匹配消息内容
        with pytest.warns(w, match=msg):
            # 调用 precision_recall_fscore_support 函数进行计算
            f([0, 1, 2], [1, 1, 2], average=average)

        # 设置 Recall 不确定警告的消息内容
        msg = (
            "Recall is ill-defined and "
            "being set to 0.0 in labels with no true samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        # 使用 pytest 的 warns 方法捕获 UndefinedMetricWarning 并匹配消息内容
        with pytest.warns(w, match=msg):
            # 调用 precision_recall_fscore_support 函数进行计算
            f([1, 1, 2], [0, 1, 2], average=average)

    # 设置 Precision 不确定警告的消息内容，针对 "samples" average
    msg = (
        "Precision is ill-defined and "
        "being set to 0.0 in samples with no predicted labels."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    # 使用 pytest 的 warns 方法捕获 UndefinedMetricWarning 并匹配消息内容
    with pytest.warns(w, match=msg):
        # 调用 precision_recall_fscore_support 函数进行计算
        f(np.array([[1, 0], [1, 0]]), np.array([[1, 0], [0, 0]]), average="samples")

    # 设置 Recall 不确定警告的消息内容，针对 "samples" average
    msg = (
        "Recall is ill-defined and "
        "being set to 0.0 in samples with no true labels."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    # 使用 pytest 的 warns 方法捕获 UndefinedMetricWarning 并匹配消息内容
    with pytest.warns(w, match=msg):
        # 调用 precision_recall_fscore_support 函数进行计算
        f(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [1, 0]]), average="samples")

    # 设置 Precision 不确定警告的消息内容，针对 "micro" average
    msg = (
        "Precision is ill-defined and "
        "being set to 0.0 due to no predicted samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    # 使用 pytest 的 warns 方法捕获 UndefinedMetricWarning 并匹配消息内容
    with pytest.warns(w, match=msg):
        # 调用 precision_recall_fscore_support 函数进行计算
        f(np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), average="micro")

    # 设置 Recall 不确定警告的消息内容，针对 "micro" average
    msg = (
        "Recall is ill-defined and "
        "being set to 0.0 due to no true samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    # 使用 pytest 的 warns 方法捕获 UndefinedMetricWarning 并匹配消息内容
    with pytest.warns(w, match=msg):
        # 调用 precision_recall_fscore_support 函数进行计算
        f(np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]), average="micro")

    # 设置 Precision 不确定警告的消息内容，针对 "binary" average
    msg = (
        "Precision is ill-defined and "
        "being set to 0.0 due to no predicted samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    # 使用 pytest 的 warns 方法捕获 UndefinedMetricWarning 并匹配消息内容
    with pytest.warns(w, match=msg):
        # 调用 precision_recall_fscore_support 函数进行计算
        f([1, 1], [-1, -1], average="binary")

    # 设置 Recall 不确定警告的消息内容，针对 "binary" average
    msg = (
        "Recall is ill-defined and "
        "being set to 0.0 due to no true samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    # 使用 pytest 的 warns 方法捕获 UndefinedMetricWarning 并匹配消息内容
    with pytest.warns(w, match=msg):
        # 调用 precision_recall_fscore_support 函数进行计算
        f([-1, -1], [1, 1], average="binary")
    # 使用 `warnings` 模块捕获警告信息
    with warnings.catch_warnings(record=True) as record:
        # 设置警告过滤器，始终显示警告
        warnings.simplefilter("always")
        # 调用 precision_recall_fscore_support 函数计算二分类的精确度、召回率和 F-score
        precision_recall_fscore_support([0, 0], [0, 0], average="binary")
        # 设置错误消息，指出由于没有真实样本或预测样本，F-score 被设置为 0.0
        msg = (
            "F-score is ill-defined and being set to 0.0 due to no true nor "
            "predicted samples. Use `zero_division` parameter to control this"
            " behavior."
        )
        # 断言捕获的警告消息与预期消息相符
        assert str(record.pop().message) == msg
        # 设置错误消息，指出由于没有真实样本，召回率被设置为 0.0
        msg = (
            "Recall is ill-defined and "
            "being set to 0.0 due to no true samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        # 断言捕获的警告消息与预期消息相符
        assert str(record.pop().message) == msg
        # 设置错误消息，指出由于没有预测样本，精确度被设置为 0.0
        msg = (
            "Precision is ill-defined and "
            "being set to 0.0 due to no predicted samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        # 断言捕获的警告消息与预期消息相符
        assert str(record.pop().message) == msg
# 使用 pytest 的 @pytest.mark.parametrize 装饰器，参数化测试函数 test_prf_no_warnings_if_zero_division_set 和 test_recall_warnings 的 zero_division 参数
@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
def test_prf_no_warnings_if_zero_division_set(zero_division):
    # 将 precision_recall_fscore_support 函数赋给变量 f，该函数计算精确率、召回率、F1 分数和支持度
    f = precision_recall_fscore_support
    # 遍历不同的 average 参数值进行测试
    for average in [None, "weighted", "macro"]:
        # 断言调用 precision_recall_fscore_support 函数时不产生警告，比较预测标签和实际标签的结果
        assert_no_warnings(
            f, [0, 1, 2], [1, 1, 2], average=average, zero_division=zero_division
        )

        assert_no_warnings(
            f, [1, 1, 2], [0, 1, 2], average=average, zero_division=zero_division
        )

    # 测试 average 参数为 "samples" 时的单样本得分
    assert_no_warnings(
        f,
        np.array([[1, 0], [1, 0]]),
        np.array([[1, 0], [0, 0]]),
        average="samples",
        zero_division=zero_division,
    )

    assert_no_warnings(
        f,
        np.array([[1, 0], [0, 0]]),
        np.array([[1, 0], [1, 0]]),
        average="samples",
        zero_division=zero_division,
    )

    # 测试 average 参数为 "micro" 时的单一得分
    assert_no_warnings(
        f,
        np.array([[1, 1], [1, 1]]),
        np.array([[0, 0], [0, 0]]),
        average="micro",
        zero_division=zero_division,
    )

    assert_no_warnings(
        f,
        np.array([[0, 0], [0, 0]]),
        np.array([[1, 1], [1, 1]]),
        average="micro",
        zero_division=zero_division,
    )

    # 测试 average 参数为 "binary" 时单一正标签的情况
    assert_no_warnings(
        f, [1, 1], [-1, -1], average="binary", zero_division=zero_division
    )

    assert_no_warnings(
        f, [-1, -1], [1, 1], average="binary", zero_division=zero_division
    )

    # 使用 warnings.catch_warnings 捕获警告，并检查是否有警告记录
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        # 调用 precision_recall_fscore_support 函数，期望 "binary" 平均计算时不产生警告
        precision_recall_fscore_support(
            [0, 0], [0, 0], average="binary", zero_division=zero_division
        )
        # 断言没有任何警告被记录
        assert len(record) == 0


# 使用 pytest 的 @pytest.mark.parametrize 装饰器，参数化测试函数 test_recall_warnings 的 zero_division 参数
@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_recall_warnings(zero_division):
    # 断言调用 recall_score 函数时不产生警告，比较预测标签和实际标签的结果
    assert_no_warnings(
        recall_score,
        np.array([[1, 1], [1, 1]]),
        np.array([[0, 0], [0, 0]]),
        average="micro",
        zero_division=zero_division,
    )
    # 使用 `warnings` 模块捕获警告信息，记录在 `record` 列表中
    with warnings.catch_warnings(record=True) as record:
        # 设置警告过滤器，将所有警告转换为常规警告
        warnings.simplefilter("always")
        # 计算召回率，传入两个二维数组作为参数，设置 `average` 为 "micro"，并根据 `zero_division` 参数处理零除错误
        recall_score(
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            average="micro",
            zero_division=zero_division,
        )
        # 如果 `zero_division` 设置为 "warn"
        if zero_division == "warn":
            # 断言最后一个警告消息与指定的字符串相等，用于检查是否发出了特定警告
            assert (
                str(record.pop().message) == "Recall is ill-defined and "
                "being set to 0.0 due to no true samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )
        else:
            # 如果没有警告被记录，则断言 `record` 列表长度为 0，即没有警告被触发
            assert len(record) == 0

        # 再次计算召回率，传入两个一维数组作为参数，用于测试没有警告时的情况
        recall_score([0, 0], [0, 0])
        # 如果 `zero_division` 设置为 "warn"
        if zero_division == "warn":
            # 断言最后一个警告消息与指定的字符串相等，用于检查是否发出了特定警告
            assert (
                str(record.pop().message) == "Recall is ill-defined and "
                "being set to 0.0 due to no true samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )
# 使用 pytest 的 parametrize 装饰器，定义了一个名为 test_precision_warnings 的测试函数，参数为 zero_division。
@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_precision_warnings(zero_division):
    # 使用 warnings.catch_warnings 上下文管理器捕获警告信息
    with warnings.catch_warnings(record=True) as record:
        # 设置警告过滤器，使得所有警告都被记录
        warnings.simplefilter("always")
        
        # 调用 precision_score 函数，计算精度得分
        precision_score(
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 0], [0, 0]]),
            average="micro",
            zero_division=zero_division,
        )
        
        # 如果 zero_division 参数为 "warn"，则验证是否捕获到了预期的警告消息
        if zero_division == "warn":
            assert (
                str(record.pop().message) == "Precision is ill-defined and "
                "being set to 0.0 due to no predicted samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )
        else:
            # 如果 zero_division 参数不为 "warn"，则验证没有捕获到任何警告消息
            assert len(record) == 0
        
        # 再次调用 precision_score 函数，计算另一组数据的精度得分
        precision_score([0, 0], [0, 0])
        
        # 如果 zero_division 参数为 "warn"，则验证是否捕获到了预期的警告消息
        if zero_division == "warn":
            assert (
                str(record.pop().message) == "Precision is ill-defined and "
                "being set to 0.0 due to no predicted samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )
    
    # 断言在测试过程中没有捕获到任何警告
    assert_no_warnings(
        precision_score,
        np.array([[0, 0], [0, 0]]),
        np.array([[1, 1], [1, 1]]),
        average="micro",
        zero_division=zero_division,
    )


# 使用 pytest 的 parametrize 装饰器，定义了一个名为 test_fscore_warnings 的测试函数，参数为 zero_division。
@pytest.mark.parametrize("zero_division", ["warn", 0, 1, np.nan])
def test_fscore_warnings(zero_division):
    # 使用 warnings.catch_warnings 上下文管理器捕获警告信息
    with warnings.catch_warnings(record=True) as record:
        # 设置警告过滤器，使得所有警告都被记录
        warnings.simplefilter("always")

        # 遍历评分函数列表，分别对每个评分函数执行以下操作
        for score in [f1_score, partial(fbeta_score, beta=2)]:
            # 调用当前评分函数，计算评分
            score(
                np.array([[1, 1], [1, 1]]),
                np.array([[0, 0], [0, 0]]),
                average="micro",
                zero_division=zero_division,
            )
            # 验证没有捕获到任何警告消息
            assert len(record) == 0

            # 再次调用当前评分函数，计算另一组数据的评分
            score(
                np.array([[0, 0], [0, 0]]),
                np.array([[1, 1], [1, 1]]),
                average="micro",
                zero_division=zero_division,
            )
            # 验证没有捕获到任何警告消息
            assert len(record) == 0

            # 再次调用当前评分函数，计算另一组数据的评分
            score(
                np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]]),
                average="micro",
                zero_division=zero_division,
            )
            # 如果 zero_division 参数为 "warn"，则验证是否捕获到了预期的警告消息
            if zero_division == "warn":
                assert (
                    str(record.pop().message) == "F-score is ill-defined and "
                    "being set to 0.0 due to no true nor predicted "
                    "samples. Use `zero_division` parameter to "
                    "control this behavior."
                )
            else:
                # 如果 zero_division 参数不为 "warn"，则验证没有捕获到任何警告消息
                assert len(record) == 0


# 定义一个名为 test_prf_average_binary_data_non_binary 的测试函数，用于测试非二进制平均数据的情况
def test_prf_average_binary_data_non_binary():
    # 定义多类别数据的真实标签和预测标签
    y_true_mc = [1, 2, 3, 3]
    y_pred_mc = [1, 2, 3, 1]
    # 定义预期的错误消息正则表达式
    msg_mc = (
        r"Target is multiclass but average='binary'. Please "
        r"choose another average setting, one of \["
        r"None, 'micro', 'macro', 'weighted'\]."
    )
    # 创建一个多标签分类的真实值数组，每行表示一个样本的标签情况
    y_true_ind = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])

    # 创建一个多标签分类的预测值数组，每行表示一个样本的预测标签情况
    y_pred_ind = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    # 设置一条警告消息，说明多标签指示器适用于二进制平均，建议选择其他平均设置
    msg_ind = (
        r"Target is multilabel-indicator but average='binary'. Please "
        r"choose another average setting, one of \["
        r"None, 'micro', 'macro', 'weighted', 'samples'\]."
    )

    # 遍历两组真实值、预测值和消息
    for y_true, y_pred, msg in [
        (y_true_mc, y_pred_mc, msg_mc),  # 这里未定义 y_true_mc, y_pred_mc, msg_mc，可能是其它代码中定义的变量
        (y_true_ind, y_pred_ind, msg_ind),
    ]:
        # 遍历四种评估指标函数：precision_score、recall_score、f1_score、fbeta_score（beta=2）
        for metric in [
            precision_score,
            recall_score,
            f1_score,
            partial(fbeta_score, beta=2),
        ]:
            # 使用 pytest 框架来断言当前评估指标函数会抛出 ValueError 异常，并且异常信息匹配设定的消息
            with pytest.raises(ValueError, match=msg):
                metric(y_true, y_pred)
# 定义函数 test__check_targets，用于测试 _check_targets 函数的行为
def test__check_targets():
    # 检查 _check_targets 函数是否正确合并目标类型、压缩输出并在输入长度不同时失败
    IND = "multilabel-indicator"  # 多标签指示器类型
    MC = "multiclass"  # 多类分类类型
    BIN = "binary"  # 二元分类类型
    CNT = "continuous"  # 连续型类型
    MMC = "multiclass-multioutput"  # 多类多输出类型
    MCN = "continuous-multioutput"  # 连续型多输出类型
    
    # 所有类型长度为3的示例
    EXAMPLES = [
        (IND, np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])),  # 不应被视为二元分类
        (IND, np.array([[0, 1], [1, 0], [1, 1]])),  # 不应被视为二元分类
        (MC, [2, 3, 1]),  # 多类分类
        (BIN, [0, 1, 1]),  # 二元分类
        (CNT, [0.0, 1.5, 1.0]),  # 连续型数据
        (MC, np.array([[2], [3], [1]])),  # 多类分类
        (BIN, np.array([[0], [1], [1]])),  # 二元分类
        (CNT, np.array([[0.0], [1.5], [1.0]])),  # 连续型数据
        (MMC, np.array([[0, 2], [1, 3], [2, 3]])),  # 多类多输出
        (MCN, np.array([[0.5, 2.0], [1.1, 3.0], [2.0, 3.0]])),  # 连续型多输出
    ]
    
    # 期望的输出类型，根据输入类型组合给出的预期输出类型，或错误时为 None
    EXPECTED = {
        (IND, IND): IND,
        (MC, MC): MC,
        (BIN, BIN): BIN,
        (MC, IND): None,
        (BIN, IND): None,
        (BIN, MC): MC,
        # 不允许的类型组合
        (CNT, CNT): None,
        (MMC, MMC): None,
        (MCN, MCN): None,
        (IND, CNT): None,
        (MC, CNT): None,
        (BIN, CNT): None,
        (MMC, CNT): None,
        (MCN, CNT): None,
        (IND, MMC): None,
        (MC, MMC): None,
        (BIN, MMC): None,
        (MCN, MMC): None,
        (IND, MCN): None,
        (MC, MCN): None,
        (BIN, MCN): None,
    }

    # 对 EXAMPLES 中的每一对 (type1, y1) 和 (type2, y2) 进行迭代
    for (type1, y1), (type2, y2) in product(EXAMPLES, repeat=2):
        try:
            expected = EXPECTED[type1, type2]
        except KeyError:
            expected = EXPECTED[type2, type1]
        
        # 如果预期结果为 None，则断言调用 _check_targets(y1, y2) 会引发 ValueError
        if expected is None:
            with pytest.raises(ValueError):
                _check_targets(y1, y2)
            
            # 如果 type1 和 type2 不相同，则断言 ValueError 异常消息符合预期
            if type1 != type2:
                err_msg = (
                    "Classification metrics can't handle a mix "
                    "of {0} and {1} targets".format(type1, type2)
                )
                with pytest.raises(ValueError, match=err_msg):
                    _check_targets(y1, y2)
            
            # 否则，如果 type1 不在 (BIN, MC, IND) 中，则断言 ValueError 异常消息符合预期
            else:
                if type1 not in (BIN, MC, IND):
                    err_msg = "{0} is not supported".format(type1)
                    with pytest.raises(ValueError, match=err_msg):
                        _check_targets(y1, y2)
        
        # 否则，预期的结果不为 None
        else:
            # 调用 _check_targets(y1, y2)，并断言返回的 merged_type 与预期的 expected 相同
            merged_type, y1out, y2out = _check_targets(y1, y2)
            assert merged_type == expected
            
            # 如果 merged_type 以 "multilabel" 开头，则断言 y1out 和 y2out 的格式为 "csr"
            if merged_type.startswith("multilabel"):
                assert y1out.format == "csr"
                assert y2out.format == "csr"
            else:
                # 否则，断言 y1out 和 y2out 与 np.squeeze(y1) 和 np.squeeze(y2) 相等
                assert_array_equal(y1out, np.squeeze(y1))
                assert_array_equal(y2out, np.squeeze(y2))
            
            # 断言调用 _check_targets(y1[:-1], y2) 会引发 ValueError
            with pytest.raises(ValueError):
                _check_targets(y1[:-1], y2)
    
    # 确保不支持序列的序列
    y1 = [(1, 2), (0, 2, 3)]
    y2 = [(2,), (0, 2)]
    # 定义一个错误消息，用于警告使用过时的多标签数据表示方式
    msg = (
        "You appear to be using a legacy multi-label data representation. "
        "Sequence of sequences are no longer supported; use a binary array"
        " or sparse matrix instead - the MultiLabelBinarizer"
        " transformer can convert to this format."
    )
    
    # 使用 pytest 的上下文管理器，检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用函数 _check_targets，传入参数 y1 和 y2，期望引发指定错误
        _check_targets(y1, y2)
# 测试函数：检查在同时使用二进制的 y_true 和 y_pred 下的多类分类情况
def test__check_targets_multiclass_with_both_y_true_and_y_pred_binary():
    # 设置测试用例的真实标签和预测标签
    y_true = [0, 1]
    y_pred = [0, -1]
    # 断言调用 _check_targets 函数返回的第一个值应为 "multiclass"
    assert _check_targets(y_true, y_pred)[0] == "multiclass"


# 测试函数：针对二元分类的 Hinge Loss 计算
def test_hinge_loss_binary():
    # 设置二元分类的真实标签和决策分数预测
    y_true = np.array([-1, 1, 1, -1])
    pred_decision = np.array([-8.5, 0.5, 1.5, -0.3])
    # 断言计算的 Hinge Loss 应为 1.2 / 4
    assert hinge_loss(y_true, pred_decision) == 1.2 / 4

    # 设置不同的二元分类的真实标签和相同的决策分数预测
    y_true = np.array([0, 2, 2, 0])
    # 断言计算的 Hinge Loss 应为 1.2 / 4
    assert hinge_loss(y_true, pred_decision) == 1.2 / 4


# 测试函数：针对多类分类的 Hinge Loss 计算
def test_hinge_loss_multiclass():
    # 设置多类分类的决策分数预测和真实标签
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58, -0.99],
            [-0.54, -0.37, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
            [-0.54, -0.38, -0.48, -0.58],
            [-2.36, -0.79, -0.27, +0.24],
            [-1.45, -0.58, -0.38, -0.17],
        ]
    )
    y_true = np.array([0, 1, 2, 1, 3, 2])
    # 创建虚拟的损失数组，并通过 np.clip 进行修剪
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][1] + pred_decision[1][2],
            1 - pred_decision[2][2] + pred_decision[2][3],
            1 - pred_decision[3][1] + pred_decision[3][2],
            1 - pred_decision[4][3] + pred_decision[4][2],
            1 - pred_decision[5][2] + pred_decision[5][3],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    # 计算虚拟损失的平均值作为 dummy_hinge_loss
    dummy_hinge_loss = np.mean(dummy_losses)
    # 断言计算的 Hinge Loss 应与 dummy_hinge_loss 相等
    assert hinge_loss(y_true, pred_decision) == dummy_hinge_loss


# 测试函数：处理多类分类时缺失标签的情况（没有包括所有标签）
def test_hinge_loss_multiclass_missing_labels_with_labels_none():
    # 设置多类分类的真实标签和决策分数预测
    y_true = np.array([0, 1, 2, 2])
    pred_decision = np.array(
        [
            [+1.27, 0.034, -0.68, -1.40],
            [-1.45, -0.58, -0.38, -0.17],
            [-2.36, -0.79, -0.27, +0.24],
            [-2.36, -0.79, -0.27, +0.24],
        ]
    )
    # 定义错误信息
    error_message = (
        "Please include all labels in y_true or pass labels as third argument"
    )
    # 使用 pytest 检查是否抛出 ValueError，并匹配特定错误信息
    with pytest.raises(ValueError, match=error_message):
        hinge_loss(y_true, pred_decision)


# 测试函数：检查多类分类中预测决策形状不一致的情况
def test_hinge_loss_multiclass_no_consistent_pred_decision_shape():
    # 设置多类分类的真实标签和不一致的决策分数预测
    y_true = np.array([2, 1, 0, 1, 0, 1, 1])
    pred_decision = np.array([0, 1, 2, 1, 0, 2, 1])
    # 定义错误信息
    error_message = (
        "The shape of pred_decision cannot be 1d array"
        "with a multiclass target. pred_decision shape "
        "must be (n_samples, n_classes), that is "
        "(7, 3). Got: (7,)"
    )
    # 使用 pytest 检查是否抛出 ValueError，并匹配特定错误信息
    with pytest.raises(ValueError, match=re.escape(error_message)):
        hinge_loss(y_true=y_true, pred_decision=pred_decision)

    # 设置不一致的决策分数预测形状和标签数量的情况
    pred_decision = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [2, 0], [0, 1], [1, 0]])
    labels = [0, 1, 2]
    # 定义错误信息字符串，指出预测决策数据的形状与类别数不一致的问题
    error_message = (
        "The shape of pred_decision is not "
        "consistent with the number of classes. "
        "With a multiclass target, pred_decision "
        "shape must be (n_samples, n_classes), that is "
        "(7, 3). Got: (7, 2)"
    )
    # 使用 pytest 框架的上下文管理器 pytest.raises 来捕获 ValueError 异常，并验证其异常消息
    with pytest.raises(ValueError, match=re.escape(error_message)):
        # 调用 hinge_loss 函数，传入真实标签 y_true、预测决策 pred_decision 和标签列表 labels
        hinge_loss(y_true=y_true, pred_decision=pred_decision, labels=labels)
def test_hinge_loss_multiclass_with_missing_labels():
    # 定义预测决策值矩阵，每行代表一个样本的决策值
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58, -0.99],
            [-0.55, -0.38, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
            [-0.55, -0.38, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
        ]
    )
    # 真实标签
    y_true = np.array([0, 1, 2, 1, 2])
    # 全部可能的标签值
    labels = np.array([0, 1, 2, 3])
    # 计算虚拟损失值
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][1] + pred_decision[1][2],
            1 - pred_decision[2][2] + pred_decision[2][3],
            1 - pred_decision[3][1] + pred_decision[3][2],
            1 - pred_decision[4][2] + pred_decision[4][3],
        ]
    )
    # 将损失值限制在最小值为0
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    # 计算虚拟的 Hinge 损失值的均值
    dummy_hinge_loss = np.mean(dummy_losses)
    # 断言计算出的 Hinge 损失值等于虚拟的 Hinge 损失值
    assert hinge_loss(y_true, pred_decision, labels=labels) == dummy_hinge_loss


def test_hinge_loss_multiclass_missing_labels_only_two_unq_in_y_true():
    # 非回归测试：
    # https://github.com/scikit-learn/scikit-learn/issues/17630
    # 验证在提供允许不在 y_true 中包含所有标签的数组时是否能计算 Hinge 损失
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58],
            [-0.15, -0.58, -0.48],
            [-1.45, -0.58, -0.38],
            [-0.55, -0.78, -0.42],
            [-1.45, -0.58, -0.38],
        ]
    )
    # 真实标签
    y_true = np.array([0, 2, 2, 0, 2])
    # 全部可能的标签值
    labels = np.array([0, 1, 2])
    # 计算虚拟损失值
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][2] + pred_decision[1][0],
            1 - pred_decision[2][2] + pred_decision[2][1],
            1 - pred_decision[3][0] + pred_decision[3][2],
            1 - pred_decision[4][2] + pred_decision[4][1],
        ]
    )
    # 将损失值限制在最小值为0
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    # 计算虚拟的 Hinge 损失值的均值
    dummy_hinge_loss = np.mean(dummy_losses)
    # 断言计算出的 Hinge 损失值与虚拟的 Hinge 损失值几乎相等
    assert_almost_equal(
        hinge_loss(y_true, pred_decision, labels=labels), dummy_hinge_loss
    )


def test_hinge_loss_multiclass_invariance_lists():
    # 目前，无法在常见的不变性测试中测试字符串和整数标签的不变性，
    # 因为多类决策函数的不变性测试尚未实现。
    # 真实标签
    y_true = ["blue", "green", "red", "green", "white", "red"]
    # 预测决策值矩阵，每行代表一个样本的决策值
    pred_decision = [
        [+0.36, -0.17, -0.58, -0.99],
        [-0.55, -0.38, -0.48, -0.58],
        [-1.45, -0.58, -0.38, -0.17],
        [-0.55, -0.38, -0.48, -0.58],
        [-2.36, -0.79, -0.27, +0.24],
        [-1.45, -0.58, -0.38, -0.17],
    ]
    # 创建一个包含预测决策相关损失的 numpy 数组，每个元素通过特定计算得出
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],  # 第一个元素的计算
            1 - pred_decision[1][1] + pred_decision[1][2],  # 第二个元素的计算
            1 - pred_decision[2][2] + pred_decision[2][3],  # 第三个元素的计算
            1 - pred_decision[3][1] + pred_decision[3][2],  # 第四个元素的计算
            1 - pred_decision[4][3] + pred_decision[4][2],  # 第五个元素的计算
            1 - pred_decision[5][2] + pred_decision[5][3],  # 第六个元素的计算
        ]
    )
    # 将 dummy_losses 数组中的元素限制在最小值 0，最大值不限，存回 dummy_losses 中
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    # 计算 dummy_losses 数组中元素的平均值，作为 dummy_hinge_loss 的值
    dummy_hinge_loss = np.mean(dummy_losses)
    # 使用断言验证 hinge_loss 函数对于给定的 y_true 和 pred_decision 的输出等于 dummy_hinge_loss
    assert hinge_loss(y_true, pred_decision) == dummy_hinge_loss
# 定义测试函数，用于验证 log_loss 函数的正确性
def test_log_loss():
    # binary case with symbolic labels ("no" < "yes")
    y_true = ["no", "no", "no", "yes", "yes", "yes"]
    y_pred = np.array(
        [[0.5, 0.5], [0.1, 0.9], [0.01, 0.99], [0.9, 0.1], [0.75, 0.25], [0.001, 0.999]]
    )
    # 计算二分类情况下的 log_loss
    loss = log_loss(y_true, y_pred)
    # 计算真实的 loss 值，使用贝努力分布的 log 概率负均值
    loss_true = -np.mean(bernoulli.logpmf(np.array(y_true) == "yes", y_pred[:, 1]))
    # 断言计算得到的 loss 与真实的 loss 值相近
    assert_allclose(loss, loss_true)

    # multiclass case; adapted from http://bit.ly/RJJHWA
    y_true = [1, 0, 2]
    y_pred = [[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]
    # 计算多分类情况下的 log_loss，并进行归一化处理
    loss = log_loss(y_true, y_pred, normalize=True)
    # 断言计算得到的 loss 值与参考值相近
    assert_allclose(loss, 0.6904911)

    # check that we got all the shapes and axes right
    # by doubling the length of y_true and y_pred
    # 将 y_true 和 y_pred 的长度加倍，检查形状和轴是否正确
    y_true *= 2
    y_pred *= 2
    # 计算未归一化的 log_loss
    loss = log_loss(y_true, y_pred, normalize=False)
    # 断言计算得到的 loss 值与参考值相近
    assert_allclose(loss, 0.6904911 * 6)

    # raise error if number of classes are not equal.
    # 如果类别数不相等，则引发 ValueError 异常
    y_true = [1, 0, 2]
    y_pred = [[0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]
    with pytest.raises(ValueError):
        log_loss(y_true, y_pred)

    # case when y_true is a string array object
    # 当 y_true 是字符串数组对象时的情况
    y_true = ["ham", "spam", "spam", "ham"]
    y_pred = [[0.3, 0.7], [0.6, 0.4], [0.4, 0.6], [0.7, 0.3]]
    # 计算 log_loss
    loss = log_loss(y_true, y_pred)
    # 断言计算得到的 loss 值与参考值相近
    assert_allclose(loss, 0.7469410)

    # test labels option

    y_true = [2, 2]
    y_pred = [[0.2, 0.8], [0.6, 0.4]]
    y_score = np.array([[0.1, 0.9], [0.1, 0.9]])
    error_str = (
        r"y_true contains only one label \(2\). Please provide "
        r"the true labels explicitly through the labels argument."
    )
    # 使用 labels 参数时，验证是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=error_str):
        log_loss(y_true, y_pred)

    y_pred = [[0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]
    error_str = r"Found input variables with inconsistent numbers of samples: \[3, 2\]"
    # 验证当样本数不一致时是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=error_str):
        log_loss(y_true, y_pred)

    # works when the labels argument is used
    # 使用 labels 参数时的正常工作情况验证

    true_log_loss = -np.mean(np.log(y_score[:, 1]))
    calculated_log_loss = log_loss(y_true, y_score, labels=[1, 2])
    # 断言计算得到的 log_loss 与真实值相近
    assert_allclose(calculated_log_loss, true_log_loss)

    # ensure labels work when len(np.unique(y_true)) != y_pred.shape[1]
    # 确保在 len(np.unique(y_true)) != y_pred.shape[1] 时 labels 参数正常工作
    y_true = [1, 2, 2]
    y_score2 = [[0.7, 0.1, 0.2], [0.2, 0.7, 0.1], [0.1, 0.7, 0.2]]
    # 计算 log_loss
    loss = log_loss(y_true, y_score2, labels=[1, 2, 3])
    # 断言计算得到的 loss 值与参考值相近
    assert_allclose(loss, -np.log(0.7))
    """Check that log_loss raises a warning when y_pred values don't sum to 1."""
    # 定义真实标签数组
    y_true = np.array([0, 1, 1, 0])
    # 定义预测标签数组，包含概率预测值
    y_pred = np.array([[0.2, 0.7], [0.6, 0.3], [0.4, 0.7], [0.8, 0.3]], dtype=dtype)

    # 使用 pytest 来捕获 UserWarning，并检查警告消息中是否包含特定文本
    with pytest.warns(UserWarning, match="The y_pred values do not sum to one."):
        # 调用 log_loss 函数，期望它引发警告
        log_loss(y_true, y_pred)
@pytest.mark.parametrize(
    "y_true, y_pred",
    [  # 使用 pytest 的参数化装饰器定义多组测试参数
        ([0, 1, 0], [0, 1, 0]),  # 测试参数1：完美预测，期望结果为0
        ([0, 1, 0], [[1, 0], [0, 1], [1, 0]]),  # 测试参数2：预测为二维数组
        ([0, 1, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # 测试参数3：多类别情况
    ],
)
def test_log_loss_perfect_predictions(y_true, y_pred):
    """Check that log_loss returns 0 for perfect predictions."""
    # 因为使用了近似匹配，所以结果不会精确为0
    assert log_loss(y_true, y_pred) == pytest.approx(0)


def test_log_loss_pandas_input():
    # 测试输入为 pandas Series 和 DataFrame 的情况
    y_tr = np.array(["ham", "spam", "spam", "ham"])
    y_pr = np.array([[0.3, 0.7], [0.6, 0.4], [0.4, 0.6], [0.7, 0.3]])
    types = [(MockDataFrame, MockDataFrame)]  # 定义模拟数据框类型的元组
    try:
        from pandas import DataFrame, Series

        types.append((Series, DataFrame))  # 如果可以导入 pandas，则添加真实的数据框类型
    except ImportError:
        pass
    for TrueInputType, PredInputType in types:
        # 使用不同类型的输入数据框来测试，其中 y_true 是 Series，y_pred 是 DataFrame
        y_true, y_pred = TrueInputType(y_tr), PredInputType(y_pr)
        loss = log_loss(y_true, y_pred)
        assert_allclose(loss, 0.7469410)


def test_brier_score_loss():
    # 检查 brier_score_loss 函数
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.9, 0.3, 1.0, 0.95])
    true_score = linalg.norm(y_true - y_pred) ** 2 / len(y_true)

    assert_almost_equal(brier_score_loss(y_true, y_true), 0.0)  # 确保对于完美预测，损失为0
    assert_almost_equal(brier_score_loss(y_true, y_pred), true_score)  # 检查真实分数计算是否正确
    assert_almost_equal(brier_score_loss(1.0 + y_true, y_pred), true_score)  # 对 y_true 进行缩放
    assert_almost_equal(brier_score_loss(2 * y_true - 1, y_pred), true_score)  # 对 y_true 进行缩放
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred[1:])  # 预测值缺少一个元素，应引发 ValueError
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred + 1.0)  # 预测值加1，应引发 ValueError
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred - 1.0)  # 预测值减1，应引发 ValueError

    # 确保对多类别的 y_true 引发错误
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0.8, 0.6, 0.4, 0.2])
    error_message = (
        "Only binary classification is supported. The type of the target is multiclass"
    )

    with pytest.raises(ValueError, match=error_message):
        brier_score_loss(y_true, y_pred)

    # 当 y_true 中只有一个类别时，确保计算是正确的
    assert_almost_equal(brier_score_loss([-1], [0.4]), 0.16)
    assert_almost_equal(brier_score_loss([0], [0.4]), 0.16)
    assert_almost_equal(brier_score_loss([1], [0.4]), 0.36)
    assert_almost_equal(brier_score_loss(["foo"], [0.4], pos_label="bar"), 0.16)
    assert_almost_equal(brier_score_loss(["foo"], [0.4], pos_label="foo"), 0.36)


def test_balanced_accuracy_score_unseen():
    msg = "y_pred contains classes not in y_true"
    with pytest.warns(UserWarning, match=msg):
        balanced_accuracy_score([0, 0, 0], [0, 0, 1])
    # 定义一个包含三个元组的列表，每个元组包含两个列表作为元素
    [
        (["a", "b", "a", "b"], ["a", "a", "a", "b"]),  # 第一个元组，包含两个输入列表
        (["a", "b", "c", "b"], ["a", "a", "a", "b"]),  # 第二个元组，包含两个输入列表
        (["a", "a", "a", "b"], ["a", "b", "c", "b"]),  # 第三个元组，包含两个输入列表
    ],
# 定义一个测试函数，用于计算平衡准确率分数
def test_balanced_accuracy_score(y_true, y_pred):
    # 计算宏平均召回率
    macro_recall = recall_score(
        y_true, y_pred, average="macro", labels=np.unique(y_true)
    )
    # 忽略警告上下文
    with ignore_warnings():
        # 使用平衡准确率计算分类器预测结果的平衡性
        balanced = balanced_accuracy_score(y_true, y_pred)
    # 断言平衡准确率与宏平均召回率近似相等
    assert balanced == pytest.approx(macro_recall)
    # 计算调整后的平衡准确率
    adjusted = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    # 计算基线的平衡准确率
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[0]))
    # 断言调整后的平衡准确率与公式计算结果近似相等
    assert adjusted == (balanced - chance) / (1 - chance)


# 参数化测试装饰器，用于测试不同分类度量标准函数
@pytest.mark.parametrize(
    "metric",
    [
        jaccard_score,
        f1_score,
        partial(fbeta_score, beta=0.5),
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        brier_score_loss,
    ],
)
# 参数化测试装饰器，测试不同类型的正类标签
@pytest.mark.parametrize(
    "classes", [(False, True), (0, 1), (0.0, 1.0), ("zero", "one")]
)
def test_classification_metric_pos_label_types(metric, classes):
    """Check that the metric works with different types of `pos_label`.

    We can expect `pos_label` to be a bool, an integer, a float, a string.
    No error should be raised for those types.
    """
    # 创建随机数生成器
    rng = np.random.RandomState(42)
    # 设定样本数和正类标签
    n_samples, pos_label = 10, classes[-1]
    # 从指定类别中随机选择真实标签
    y_true = rng.choice(classes, size=n_samples, replace=True)
    # 如果度量标准是 brier_score_loss，则预测值需要是概率值
    if metric is brier_score_loss:
        y_pred = rng.uniform(size=n_samples)
    else:
        y_pred = y_true.copy()
    # 计算度量标准的结果
    result = metric(y_true, y_pred, pos_label=pos_label)
    # 断言结果中不存在 NaN 值
    assert not np.any(np.isnan(result))


# 参数化测试装饰器，测试小规模二分类输入下 f1-score 的行为
@pytest.mark.parametrize(
    "y_true, y_pred, expected_score",
    [
        (np.array([0, 1]), np.array([1, 0]), 0.0),
        (np.array([0, 1]), np.array([0, 1]), 1.0),
        (np.array([0, 1]), np.array([0, 0]), 0.0),
        (np.array([0, 0]), np.array([0, 0]), 1.0),
    ],
)
def test_f1_for_small_binary_inputs_with_zero_division(y_true, y_pred, expected_score):
    """Check the behaviour of `zero_division` for f1-score.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/26965
    """
    # 断言使用指定的 zero_division 参数计算 f1-score 后与预期得分近似相等
    assert f1_score(y_true, y_pred, zero_division=1.0) == pytest.approx(expected_score)


# 参数化测试装饰器，测试分类度量标准中的 np.nan 除法处理
@pytest.mark.parametrize(
    "scoring",
    [
        make_scorer(f1_score, zero_division=np.nan),
        make_scorer(fbeta_score, beta=2, zero_division=np.nan),
        make_scorer(precision_score, zero_division=np.nan),
        make_scorer(recall_score, zero_division=np.nan),
    ],
)
def test_classification_metric_division_by_zero_nan_validaton(scoring):
    """Check that we validate `np.nan` properly for classification metrics.

    With `n_jobs=2` in cross-validation, the `np.nan` used for the singleton will be
    different in the sub-process and we should not use the `is` operator but
    `math.isnan`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27563
    """
    # 生成分类数据集
    X, y = datasets.make_classification(random_state=0)
    # 使用决策树分类器进行训练，设定最大深度为3，随机种子为0
    classifier = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    # 使用交叉验证评估分类器的性能，设定评分方式为scoring，同时使用2个CPU核心进行计算
    cross_val_score(classifier, X, y, scoring=scoring, n_jobs=2, error_score="raise")
# TODO(1.7): remove
def test_brier_score_loss_deprecation_warning():
    """Check the message for future deprecation."""
    # 定义测试数据
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.9, 0.3, 1.0, 0.95])

    # 设置警告消息并验证是否触发 FutureWarning
    warn_msg = "y_prob was deprecated in version 1.5"
    with pytest.warns(FutureWarning, match=warn_msg):
        brier_score_loss(
            y_true,
            y_prob=y_pred,
        )

    # 设置错误消息并验证是否触发 ValueError
    error_msg = "`y_prob` and `y_proba` cannot be both specified"
    with pytest.raises(ValueError, match=error_msg):
        brier_score_loss(
            y_true,
            y_prob=y_pred,
            y_proba=y_pred,
        )


def test_d2_log_loss_score():
    # 准备测试数据
    y_true = [0, 0, 0, 1, 1, 1]
    y_true_string = ["no", "no", "no", "yes", "yes", "yes"]
    y_pred = np.array(
        [
            [0.5, 0.5],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.35, 0.65],
            [0.01, 0.99],
        ]
    )
    y_pred_null = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    # 计算 D2 损失分数并验证
    d2_score = d2_log_loss_score(y_true=y_true, y_pred=y_pred)
    log_likelihood = log_loss(y_true=y_true, y_pred=y_pred, normalize=False)
    log_likelihood_null = log_loss(y_true=y_true, y_pred=y_pred_null, normalize=False)
    d2_score_true = 1 - log_likelihood / log_likelihood_null
    assert d2_score == pytest.approx(d2_score_true)

    # 使用样本权重验证 D2 损失分数
    sample_weight = np.array([2, 1, 3, 4, 3, 1])
    y_pred_null[:, 0] = sample_weight[:3].sum() / sample_weight.sum()
    y_pred_null[:, 1] = sample_weight[3:].sum() / sample_weight.sum()
    d2_score = d2_log_loss_score(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
    )
    log_likelihood = log_loss(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        normalize=False,
    )
    log_likelihood_null = log_loss(
        y_true=y_true,
        y_pred=y_pred_null,
        sample_weight=sample_weight,
        normalize=False,
    )
    d2_score_true = 1 - log_likelihood / log_likelihood_null
    assert d2_score == pytest.approx(d2_score_true)

    # 检查优秀预测是否给出较高的 D2 分数
    y_pred = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_pred)
    assert 0.5 < d2_score < 1.0
    # 检查字符串标签的相似值
    d2_score_string = d2_log_loss_score(y_true_string, y_pred)
    assert d2_score_string == pytest.approx(d2_score)

    # 检查不良预测是否给出相对较低的 D2 分数
    # 创建一个包含预测概率的 NumPy 数组 y_pred
    y_pred = np.array(
        [
            [0.5, 0.5],  # 第一个样本的预测概率
            [0.1, 0.9],  # 第二个样本的预测概率
            [0.1, 0.9],  # 第三个样本的预测概率
            [0.9, 0.1],  # 第四个样本的预测概率
            [0.75, 0.25],  # 第五个样本的预测概率
            [0.1, 0.9],  # 第六个样本的预测概率
        ]
    )
    # 计算使用 D2 损失函数的得分，评估预测和真实标签的匹配程度
    d2_score = d2_log_loss_score(y_true, y_pred)
    # 确保 D2 分数小于 0
    assert d2_score < 0

    # 使用字符串标签的情况下，检查是否得到了类似的分数
    d2_score_string = d2_log_loss_score(y_true_string, y_pred)
    # 确保字符串标签的 D2 分数近似等于先前计算的数值分数
    assert d2_score_string == pytest.approx(d2_score)

    # 当使用每个类别的平均值作为预测时，检查是否得到 D2 分数为 0
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = np.array(
        [
            [0.5, 0.5],  # 所有样本的预测概率都相同
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_pred)
    # 确保 D2 分数等于 0
    assert d2_score == 0
    # 对字符串标签的情况也进行相同的检查
    d2_score_string = d2_log_loss_score(y_true_string, y_pred)
    assert d2_score_string == 0

    # 当正类占比较高时，检查是否得到 D2 分数为 0
    y_true = [0, 1, 1, 1]
    y_true_string = ["no", "yes", "yes", "yes"]
    y_pred = np.array([[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]])
    d2_score = d2_log_loss_score(y_true, y_pred)
    # 确保 D2 分数等于 0
    assert d2_score == 0
    # 对字符串标签的情况也进行相同的检查
    d2_score_string = d2_log_loss_score(y_true_string, y_pred)
    assert d2_score_string == 0
    # 使用样本权重时的检查
    sample_weight = [2, 2, 2, 2]
    d2_score_with_sample_weight = d2_log_loss_score(
        y_true, y_pred, sample_weight=sample_weight
    )
    # 确保带有样本权重的 D2 分数等于 0
    assert d2_score_with_sample_weight == 0

    # 当标签包含多于两个类别时，确保 D2 分数在 0.5 到 1.0 之间
    y_true = ["high", "high", "low", "neutral"]
    sample_weight = [1.4, 0.6, 0.8, 0.2]

    y_pred = np.array(
        [
            [0.8, 0.1, 0.1],  # 第一个样本的预测概率分布
            [0.8, 0.1, 0.1],  # 第二个样本的预测概率分布
            [0.1, 0.8, 0.1],  # 第三个样本的预测概率分布
            [0.1, 0.1, 0.8],  # 第四个样本的预测概率分布
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_pred)
    # 确保 D2 分数在 0.5 到 1.0 之间
    assert 0.5 < d2_score < 1.0
    # 使用样本权重时的检查
    d2_score = d2_log_loss_score(y_true, y_pred, sample_weight=sample_weight)
    # 确保带有样本权重的 D2 分数在 0.5 到 1.0 之间
    assert 0.5 < d2_score < 1.0

    y_pred = np.array(
        [
            [0.2, 0.5, 0.3],  # 第一个样本的预测概率分布
            [0.1, 0.7, 0.2],  # 第二个样本的预测概率分布
            [0.1, 0.1, 0.8],  # 第三个样本的预测概率分布
            [0.2, 0.7, 0.1],  # 第四个样本的预测概率分布
        ]
    )
    d2_score = d2_log_loss_score(y_true, y_pred)
    # 确保 D2 分数小于 0
    assert d2_score < 0
    # 使用样本权重时的检查
    d2_score = d2_log_loss_score(y_true, y_pred, sample_weight=sample_weight)
    # 确保带有样本权重的 D2 分数小于 0
    assert d2_score < 0
# 定义测试函数，验证 d2_log_loss_score 在无效输入时是否会引发适当的错误。
def test_d2_log_loss_score_raises():
    # 定义真实标签和预测标签
    y_true = [0, 1, 2]
    y_pred = [[0.2, 0.8], [0.5, 0.5], [0.4, 0.6]]
    err = "contain different number of classes"
    # 使用 pytest 的 raises 函数检查是否引发 ValueError 异常，并匹配特定错误信息
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_pred)

    # 检查当标签中的类别数与 y_pred 中的类别数不匹配时是否引发错误
    y_true = ["a", "b", "c"]
    y_pred = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    labels = [0, 1, 2]
    err = "number of classes in labels is different"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_pred, labels=labels)

    # 检查当 y_true 和 y_pred 的长度不相等时是否引发错误
    y_true = [0, 1, 2]
    y_pred = [[0.5, 0.5, 0.5], [0.6, 0.3, 0.1]]
    err = "inconsistent numbers of samples"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_pred)

    # 检查样本数少于 2 时是否会发出警告
    y_true = [1]
    y_pred = [[0.5, 0.5]]
    err = "score is not well-defined"
    with pytest.warns(UndefinedMetricWarning, match=err):
        d2_log_loss_score(y_true, y_pred)

    # 检查当 y_true 中仅包含一个标签时是否引发错误
    y_true = [1, 1, 1]
    y_pred = [[0.5, 0.5], [0.5, 0.5], [0.5, 5]]
    err = "y_true contains only one label"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_pred)

    # 检查当 y_true 中仅包含一个标签且 labels 也只有一个标签时是否引发错误
    y_true = [1, 1, 1]
    labels = [1]
    y_pred = [[0.5, 0.5], [0.5, 0.5], [0.5, 5]]
    err = "The labels array needs to contain at least two"
    with pytest.raises(ValueError, match=err):
        d2_log_loss_score(y_true, y_pred, labels=labels)
```