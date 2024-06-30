# `D:\src\scipysrc\scikit-learn\sklearn\metrics\tests\test_ranking.py`

```
import re  # 导入正则表达式模块
import warnings  # 导入警告处理模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架
from scipy import stats  # 从SciPy库中导入统计模块

from sklearn import datasets, svm  # 导入sklearn中的数据集和支持向量机模块
from sklearn.datasets import make_multilabel_classification  # 导入多标签数据生成函数
from sklearn.exceptions import UndefinedMetricWarning  # 导入未定义度量警告类
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import (  # 导入多个评估指标函数
    accuracy_score,
    auc,
    average_precision_score,
    coverage_error,
    dcg_score,
    det_curve,
    label_ranking_average_precision_score,
    label_ranking_loss,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores  # 导入评分函数
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.preprocessing import label_binarize  # 导入标签二值化函数
from sklearn.random_projection import _sparse_random_matrix  # 导入稀疏随机矩阵函数
from sklearn.utils._testing import (  # 导入测试辅助函数
    _convert_container,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.extmath import softmax  # 导入softmax函数
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入CSR容器修复函数
from sklearn.utils.validation import (  # 导入数据验证函数
    check_array,
    check_consistent_length,
    check_random_state,
)

###############################################################################
# Utilities for testing

CURVE_FUNCS = [
    det_curve,  # 检测曲线函数
    precision_recall_curve,  # 精确率-召回率曲线函数
    roc_curve,  # ROC曲线函数
]


def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC

    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """

    if dataset is None:
        # import some data to play with
        dataset = datasets.load_iris()  # 载入鸢尾花数据集

    X = dataset.data  # 提取数据特征
    y = dataset.target  # 提取目标标签

    if binary:
        # restrict to a binary classification task
        X, y = X[y < 2], y[y < 2]  # 若为二分类任务，则只保留两类数据

    n_samples, n_features = X.shape  # 计算样本数和特征数
    p = np.arange(n_samples)  # 创建样本索引

    rng = check_random_state(37)  # 设定随机数生成器种子
    rng.shuffle(p)  # 打乱样本索引顺序
    X, y = X[p], y[p]  # 根据打乱后的索引重新排列数据集

    half = int(n_samples / 2)  # 将样本分成两半

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)  # 设定新的随机数生成器种子
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]  # 添加噪声特征以增加问题难度

    # run classifier, get class probabilities and label predictions
    clf = svm.SVC(kernel="linear", probability=True, random_state=0)  # 创建线性核SVM分类器对象
    y_score = clf.fit(X[:half], y[:half]).predict_proba(X[half:])  # 在一半的数据上训练分类器并预测概率

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        y_score = y_score[:, 1]  # 若为二分类任务，则只保留正类的概率

    y_pred = clf.predict(X[half:])  # 预测标签
    y_true = y[half:]  # 真实标签
    return y_true, y_pred, y_score  # 返回真实标签、预测标签和预测概率数组


###############################################################################
# Tests


def _auc(y_true, y_score):
    """Alternative implementation to check for correctness of
    `roc_auc_score`."""
    pos_label = np.unique(y_true)[1]  # 找到真实标签中的正类标签
    # 从 y_true 和 y_score 中选择正例和负例的分数
    pos = y_score[y_true == pos_label]
    neg = y_score[y_true != pos_label]
    
    # 创建一个差值矩阵，计算每个正例分数与每个负例分数之间的差值
    diff_matrix = pos.reshape(1, -1) - neg.reshape(-1, 1)
    
    # 计算差值矩阵中大于零的元素个数，即正例分数大于负例分数的次数
    n_correct = np.sum(diff_matrix > 0)

    # 返回正例分数大于负例分数的次数与正负例样本数乘积的比率作为结果
    return n_correct / float(len(pos) * len(neg))
# 计算平均精度 (Average Precision) 的另一种实现，用于验证 `average_precision_score` 的正确性。
def _average_precision(y_true, y_score):
    # 获取正例标签值
    pos_label = np.unique(y_true)[1]
    # 统计正例的数量
    n_pos = np.sum(y_true == pos_label)
    # 按照 y_score 的值对索引进行排序，降序排列
    order = np.argsort(y_score)[::-1]
    y_score = y_score[order]
    y_true = y_true[order]

    score = 0
    # 遍历排序后的 y_score
    for i in range(len(y_score)):
        if y_true[i] == pos_label:
            # 计算到第 i 个文档的精度，即截至到第 i 个文档的相关文档的百分比
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= i + 1.0
            score += prec

    # 返回平均精度
    return score / n_pos


# 第二种平均精度的实现，与维基百科中的定义接近，用于验证 `average_precision_score` 的正确性。
def _average_precision_slow(y_true, y_score):
    # 计算精确率和召回率曲线
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    # 将精确率和召回率反转
    precision = list(reversed(precision))
    recall = list(reversed(recall))
    average_precision = 0
    # 计算平均精度
    for i in range(1, len(precision)):
        average_precision += precision[i] * (recall[i] - recall[i - 1])
    return average_precision


# 用于验证带有设定 `max_fpr` 的 `roc_auc_score` 的正确性的另一种实现。
def _partial_roc_auc_score(y_true, y_predict, max_fpr):
    # 内部函数，计算部分 ROC 曲线
    def _partial_roc(y_true, y_predict, max_fpr):
        # 计算 ROC 曲线的假阳性率（fpr）、真阳性率（tpr）
        fpr, tpr, _ = roc_curve(y_true, y_predict)
        # 筛选出小于等于设定 `max_fpr` 的新假阳性率
        new_fpr = fpr[fpr <= max_fpr]
        new_fpr = np.append(new_fpr, max_fpr)
        new_tpr = tpr[fpr <= max_fpr]
        # 找到大于 `max_fpr` 的位置索引，进行插值
        idx_out = np.argmax(fpr > max_fpr)
        idx_in = idx_out - 1
        x_interp = [fpr[idx_in], fpr[idx_out]]
        y_interp = [tpr[idx_in], tpr[idx_out]]
        new_tpr = np.append(new_tpr, np.interp(max_fpr, x_interp, y_interp))
        return (new_fpr, new_tpr)

    # 调用内部函数，计算部分 ROC 曲线
    new_fpr, new_tpr = _partial_roc(y_true, y_predict, max_fpr)
    # 计算部分 AUC
    partial_auc = auc(new_fpr, new_tpr)

    # 根据 McClish 1989 论文中的公式 (5) 计算部分 AUC
    fpr1 = 0
    fpr2 = max_fpr
    min_area = 0.5 * (fpr2 - fpr1) * (fpr2 + fpr1)
    max_area = fpr2 - fpr1
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


# 参数化测试函数，测试 ROC 曲线下面积（AUC）
@pytest.mark.parametrize("drop", [True, False])
def test_roc_curve(drop):
    # 获取预测结果的真实值、未使用的占位符和预测分数
    y_true, _, y_score = make_prediction(binary=True)
    # 预期的 AUC 值，使用 _auc 函数计算
    expected_auc = _auc(y_true, y_score)
    # 计算 ROC 曲线的 FPR（假正率）、TPR（真正率）和阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=drop)
    
    # 计算 ROC 曲线下的面积（AUC 值）
    roc_auc = auc(fpr, tpr)
    
    # 断言计算得到的 AUC 值与预期的 AUC 值几乎相等，精确到小数点后两位
    assert_array_almost_equal(roc_auc, expected_auc, decimal=2)
    
    # 断言计算得到的 AUC 值与使用 roc_auc_score 函数计算的 AUC 值几乎相等
    assert_almost_equal(roc_auc, roc_auc_score(y_true, y_score))
    
    # 断言 FPR、TPR 和阈值的形状必须一致
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape
def test_roc_curve_end_points():
    # 确保 roc_curve 返回的曲线从 0 开始到 1 结束，即使在极端情况下也是如此
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = rng.randint(3, size=100)
    fpr, tpr, thr = roc_curve(y_true, y_pred, drop_intermediate=True)
    assert fpr[0] == 0
    assert fpr[-1] == 1
    assert fpr.shape == tpr.shape
    assert fpr.shape == thr.shape


def test_roc_returns_consistency():
    # 测试返回的阈值与 tpr 是否匹配
    # 创建一个小型的测试数据集
    y_true, _, y_score = make_prediction(binary=True)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # 使用给定的阈值计算 tpr
    tpr_correct = []
    for t in thresholds:
        tp = np.sum((y_score >= t) & y_true)
        p = np.sum(y_true)
        tpr_correct.append(1.0 * tp / p)

    # 比较 tpr 和 tpr_correct，以确定阈值的顺序是否正确
    assert_array_almost_equal(tpr, tpr_correct, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_multi():
    # 多类问题不适用 roc_curve
    y_true, _, y_score = make_prediction(binary=False)

    with pytest.raises(ValueError):
        roc_curve(y_true, y_score)


def test_roc_curve_confidence():
    # 用于置信度分数的 roc_curve
    y_true, _, y_score = make_prediction(binary=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_score - 0.5)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.90, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_hard():
    # 用于硬决策的 roc_curve
    y_true, pred, y_score = make_prediction(binary=True)

    # 总是预测为正类
    trivial_pred = np.ones(y_true.shape)
    fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.50, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # 总是预测为负类
    trivial_pred = np.zeros(y_true.shape)
    fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.50, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # 硬决策
    fpr, tpr, thresholds = roc_curve(y_true, pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.78, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_one_label():
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # 断言应该有警告
    expected_message = (
        "No negative samples in y_true, false positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # 检查断言，确保所有的假阳性率（fpr）都应该是 NaN
    assert_array_equal(fpr, np.full(len(thresholds), np.nan))
    # 断言假阳性率（fpr）、真阳性率（tpr）和阈值（thresholds）的形状相同
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # 断言会触发警告
    expected_message = (
        "No positive samples in y_true, true positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        # 计算 ROC 曲线，针对所有真实标签取反后的预测值
        fpr, tpr, thresholds = roc_curve([1 - x for x in y_true], y_pred)
    # 检查断言，确保所有的真阳性率（tpr）都应该是 NaN
    assert_array_equal(tpr, np.full(len(thresholds), np.nan))
    # 再次确认假阳性率（fpr）、真阳性率（tpr）和阈值（thresholds）的形状相同
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape
def test_roc_curve_toydata():
    # Binary classification
    # 定义二分类真实标签和预测得分
    y_true = [0, 1]
    y_score = [0, 1]
    # 计算ROC曲线的真正率（TPR）、假正率（FPR）和阈值
    tpr, fpr, _ = roc_curve(y_true, y_score)
    # 计算ROC曲线下面积（AUC）
    roc_auc = roc_auc_score(y_true, y_score)
    # 断言TPR和FPR的数组近似等于指定值
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    # 断言ROC曲线下面积近似等于1.0
    assert_almost_equal(roc_auc, 1.0)

    y_true = [0, 1]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1, 1])
    assert_array_almost_equal(fpr, [0, 0, 1])
    assert_almost_equal(roc_auc, 0.0)

    y_true = [1, 0]
    y_score = [1, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [1, 0]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [1, 0]
    y_score = [0.5, 0.5]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [0, 0]
    y_score = [0.25, 0.75]
    # 断言由于y_true中没有正样本而引发的UndefinedMetricWarning
    expected_message = (
        "No positive samples in y_true, true positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)

    # 断言由于y_true中没有正样本而引发的ValueError
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)
    # 断言TPR和FPR的数组近似等于指定值，且包含NaN
    assert_array_almost_equal(tpr, [0.0, 0.5, 1.0])
    assert_array_almost_equal(fpr, [np.nan, np.nan, np.nan])

    y_true = [1, 1]
    y_score = [0.25, 0.75]
    # 断言由于y_true中没有负样本而引发的UndefinedMetricWarning
    expected_message = (
        "No negative samples in y_true, false positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)

    # 断言由于y_true中没有负样本而引发的ValueError
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)
    # 断言TPR和FPR的数组近似等于指定值，且包含NaN
    assert_array_almost_equal(tpr, [np.nan, np.nan, np.nan])
    assert_array_almost_equal(fpr, [0.0, 0.5, 1.0])

    # Multi-label classification task
    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [0, 1]])
    # 断言由于多标签分类不支持宏平均而引发的ValueError
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="macro")
    # 断言由于多标签分类不支持加权平均而引发的ValueError
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="weighted")
    # 断言对样本平均得到的ROC曲线下面积近似等于1.0
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 1.0)
    # 断言对微观平均得到的ROC曲线下面积近似等于1.0
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 1.0)
    # 定义一个测试用的预测分数矩阵，其中包含两个类别的分数
    y_score = np.array([[0, 1], [1, 0]])
    # 使用 pytest 检查 roc_auc_score 函数在传入不合法参数时是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="macro")
    # 同样使用 pytest 检查 roc_auc_score 函数在传入不合法参数时是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="weighted")
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 samples 平均值是否接近 0.5
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0.5)
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 micro 平均值是否接近 0.5
    
    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 macro 平均值是否接近 0
    assert_almost_equal(roc_auc_score(y_true, y_score, average="macro"), 0)
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 weighted 平均值是否接近 0
    assert_almost_equal(roc_auc_score(y_true, y_score, average="weighted"), 0)
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 samples 平均值是否接近 0
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0)
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 micro 平均值是否接近 0
    
    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0.5, 0.5], [0.5, 0.5]])
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 macro 平均值是否接近 0.5
    assert_almost_equal(roc_auc_score(y_true, y_score, average="macro"), 0.5)
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 weighted 平均值是否接近 0.5
    assert_almost_equal(roc_auc_score(y_true, y_score, average="weighted"), 0.5)
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 samples 平均值是否接近 0.5
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0.5)
    # 使用 assert_almost_equal 函数检查 roc_auc_score 计算的 micro 平均值是否接近 0.5
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0.5)
def test_roc_curve_drop_intermediate():
    # 测试 drop_intermediate 参数是否正确地删除了阈值
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    assert_array_almost_equal(thresholds, [np.inf, 1.0, 0.7, 0.0])

    # 测试重复得分时是否正确删除了阈值
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.6, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    assert_array_almost_equal(thresholds, [np.inf, 1.0, 0.9, 0.7, 0.6, 0.0])


def test_roc_curve_fpr_tpr_increasing():
    # 确保 roc_curve 返回的 fpr 和 tpr 是递增的。
    # 构建一个边界情况，其中有浮点型的 y_score 和 sample_weight，
    # 当一些相邻的 fpr 和 tpr 值实际上相同时。
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.7, 0.3, 0.4, 0.5]
    sample_weight = np.repeat(0.2, 5)
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    assert (np.diff(fpr) < 0).sum() == 0
    assert (np.diff(tpr) < 0).sum() == 0


def test_auc():
    # 测试计算曲线下面积（AUC）
    x = [0, 1]
    y = [0, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [1, 0]
    y = [0, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [1, 0, 0]
    y = [0, 1, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [0, 1]
    y = [1, 1]
    assert_array_almost_equal(auc(x, y), 1)
    x = [0, 0.5, 1]
    y = [0, 0.5, 1]
    assert_array_almost_equal(auc(x, y), 0.5)


def test_auc_errors():
    # 测试错误情况下的 AUC 计算
    # 不兼容的形状
    with pytest.raises(ValueError):
        auc([0.0, 0.5, 1.0], [0.1, 0.2])

    # x 值过少
    with pytest.raises(ValueError):
        auc([0.0], [0.1])

    # x 值未按顺序排列
    x = [2, 1, 3, 4]
    y = [5, 6, 7, 8]
    error_message = "x is neither increasing nor decreasing : {}".format(np.array(x))
    with pytest.raises(ValueError, match=re.escape(error_message)):
        auc(x, y)


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 1, 0, 2]), [0, 1, 2]),
        (np.array([0, 1, 0, 2]), None),
        (["a", "b", "a", "c"], ["a", "b", "c"]),
        (["a", "b", "a", "c"], None),
    ],
)
def test_multiclass_ovo_roc_auc_toydata(y_true, labels):
    # 测试一对一的多类别 ROC AUC 算法
    # 在一个小例子上，代表了一个预期的使用情况。
    y_scores = np.array(
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]]
    )

    # 用于计算预期输出。
    # 考虑标签 0 和 1：
    # 正标签是 0，负标签是 1
    score_01 = roc_auc_score([1, 0, 1], [0.1, 0.3, 0.35])
    # 正标签是 1，负标签是 0
    score_10 = roc_auc_score([0, 1, 0], [0.8, 0.4, 0.5])
    average_score_01 = (score_01 + score_10) / 2

    # 考虑标签 0 和 2：
    # 计算两个二分类问题的 ROC AUC 分数
    score_02 = roc_auc_score([1, 1, 0], [0.1, 0.35, 0])
    score_20 = roc_auc_score([0, 0, 1], [0.1, 0.15, 0.8])
    # 计算这两个二分类问题的平均 ROC AUC 分数
    average_score_02 = (score_02 + score_20) / 2

    # 考虑标签 1 和 2：
    score_12 = roc_auc_score([1, 0], [0.4, 0.2])
    score_21 = roc_auc_score([0, 1], [0.3, 0.8])
    # 计算这两个二分类问题的平均 ROC AUC 分数
    average_score_12 = (score_12 + score_21) / 2

    # 使用未加权的一对一多类别 ROC AUC 算法
    ovo_unweighted_score = (average_score_01 + average_score_02 + average_score_12) / 3
    assert_almost_equal(
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo"),
        ovo_unweighted_score,
    )

    # 使用加权的一对一多类别 ROC AUC 算法
    # 每个项根据正标签的流行度进行加权
    pair_scores = [average_score_01, average_score_02, average_score_12]
    prevalence = [0.75, 0.75, 0.50]
    ovo_weighted_score = np.average(pair_scores, weights=prevalence)
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, labels=labels, multi_class="ovo", average="weighted"
        ),
        ovo_weighted_score,
    )

    # 检查当 average=None 时，是否会引发 NotImplemented 错误
    error_message = "average=None is not implemented for multi_class='ovo'."
    with pytest.raises(NotImplementedError, match=error_message):
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo", average=None)
@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 2, 0, 2]), [0, 1, 2]),
        (np.array(["a", "d", "a", "d"]), ["a", "b", "d"]),
    ],
)
def test_multiclass_ovo_roc_auc_toydata_binary(y_true, labels):
    # Tests the one-vs-one multiclass ROC AUC algorithm for binary y_true
    #
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[0.2, 0.0, 0.8], [0.6, 0.0, 0.4], [0.55, 0.0, 0.45], [0.4, 0.0, 0.6]]
    )

    # Used to compute the expected output.
    # Consider labels 0 and 1:
    # positive label is 0, negative label is 1
    score_01 = roc_auc_score([1, 0, 1, 0], [0.2, 0.6, 0.55, 0.4])
    # positive label is 1, negative label is 0
    score_10 = roc_auc_score([0, 1, 0, 1], [0.8, 0.4, 0.45, 0.6])
    ovo_score = (score_01 + score_10) / 2

    assert_almost_equal(
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo"), ovo_score
    )

    # Weighted, one-vs-one multiclass ROC AUC algorithm
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, labels=labels, multi_class="ovo", average="weighted"
        ),
        ovo_score,
    )


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 1, 2, 2]), None),
        (["a", "b", "c", "c"], None),
        ([0, 1, 2, 2], [0, 1, 2]),
        (["a", "b", "c", "c"], ["a", "b", "c"]),
    ],
)
def test_multiclass_ovr_roc_auc_toydata(y_true, labels):
    # Tests the unweighted, one-vs-rest multiclass ROC AUC algorithm
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[1.0, 0.0, 0.0], [0.1, 0.5, 0.4], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]
    )
    # Compute the expected result by individually computing the 'one-vs-rest'
    # ROC AUC scores for classes 0, 1, and 2.
    out_0 = roc_auc_score([1, 0, 0, 0], y_scores[:, 0])
    out_1 = roc_auc_score([0, 1, 0, 0], y_scores[:, 1])
    out_2 = roc_auc_score([0, 0, 1, 1], y_scores[:, 2])
    assert_almost_equal(
        roc_auc_score(y_true, y_scores, multi_class="ovr", labels=labels, average=None),
        [out_0, out_1, out_2],
    )

    # Compute unweighted results (default behaviour is average="macro")
    result_unweighted = (out_0 + out_1 + out_2) / 3.0
    assert_almost_equal(
        roc_auc_score(y_true, y_scores, multi_class="ovr", labels=labels),
        result_unweighted,
    )

    # Tests the weighted, one-vs-rest multiclass ROC AUC algorithm
    # on the same input (Provost & Domingos, 2000)
    result_weighted = out_0 * 0.25 + out_1 * 0.25 + out_2 * 0.5
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, multi_class="ovr", labels=labels, average="weighted"
        ),
        result_weighted,
    )


@pytest.mark.parametrize(
    "multi_class, average",
    [
        ("ovr", "macro"),
        ("ovr", "micro"),
        ("ovo", "macro"),
    ],
)
def test_perfect_imperfect_chance_multiclass_roc_auc(multi_class, average):
    # Tests the multiclass ROC AUC algorithm for perfect and imperfect
    #
    # chance on different multi_class and average strategies.
    # 创建一个包含真实标签的 NumPy 数组
    y_true = np.array([3, 1, 2, 0])

    # 完美分类器（从排名角度）具有 roc_auc_score = 1.0
    y_perfect = [
        [0.0, 0.0, 0.0, 1.0],  # 第一类完全正确
        [0.0, 1.0, 0.0, 0.0],  # 第二类完全正确
        [0.0, 0.0, 1.0, 0.0],  # 第三类完全正确
        [0.75, 0.05, 0.05, 0.15],  # 第四类部分正确
    ]
    assert_almost_equal(
        # 检查完美分类器的 roc_auc_score
        roc_auc_score(y_true, y_perfect, multi_class=multi_class, average=average),
        1.0,
    )

    # 不完美分类器具有 roc_auc_score < 1.0
    y_imperfect = [
        [0.0, 0.0, 0.0, 1.0],  # 第一类完全正确
        [0.0, 1.0, 0.0, 0.0],  # 第二类完全正确
        [0.0, 0.0, 1.0, 0.0],  # 第三类完全正确
        [0.0, 0.0, 0.0, 1.0],  # 第四类完全错误
    ]
    assert (
        # 检查不完美分类器的 roc_auc_score 小于 1.0
        roc_auc_score(y_true, y_imperfect, multi_class=multi_class, average=average)
        < 1.0
    )

    # 机会水平分类器具有 roc_auc_score = 5.0
    y_chance = 0.25 * np.ones((4, 4))  # 每个类别的预测概率均为 0.25
    assert roc_auc_score(
        # 检查机会水平分类器的 roc_auc_score
        y_true, y_chance, multi_class=multi_class, average=average
    ) == pytest.approx(0.5)  # 期望的平均 AUC 值为 0.5
def test_micro_averaged_ovr_roc_auc(global_random_seed):
    # 设置全局随机种子
    seed = global_random_seed
    # 生成一组随机预测和对应的真实标签，确保预测不完美
    # 使用狄利克雷先验（多项分布的共轭先验）创建一个不均衡的类分布
    y_pred = stats.dirichlet.rvs([2.0, 1.0, 0.5], size=1000, random_state=seed)
    # 根据预测结果生成真实标签，以确保每次预测只有一个类别被选中
    y_true = np.asarray(
        [
            stats.multinomial.rvs(n=1, p=y_pred_i, random_state=seed).argmax()
            for y_pred_i in y_pred
        ]
    )
    # 将真实标签转换为独热编码形式
    y_onehot = label_binarize(y_true, classes=[0, 1, 2])
    # 计算 ROC 曲线的假阳率和真阳率
    fpr, tpr, _ = roc_curve(y_onehot.ravel(), y_pred.ravel())
    # 计算手动计算的 ROC AUC
    roc_auc_by_hand = auc(fpr, tpr)
    # 使用库函数计算的 ROC AUC，采用“ovr”策略和“micro”平均
    roc_auc_auto = roc_auc_score(y_true, y_pred, multi_class="ovr", average="micro")
    # 断言手动计算的 ROC AUC 与自动计算的 ROC AUC 相近
    assert roc_auc_by_hand == pytest.approx(roc_auc_auto)


@pytest.mark.parametrize(
    "msg, y_true, labels",
    [
        ("Parameter 'labels' must be unique", np.array([0, 1, 2, 2]), [0, 2, 0]),
        # y_true 中的标签必须唯一
        (
            "Parameter 'labels' must be unique",
            np.array(["a", "b", "c", "c"]),
            ["a", "a", "b"],
        ),
        # y_true 中的标签必须唯一
        (
            (
                "Number of classes in y_true not equal to the number of columns "
                "in 'y_score'"
            ),
            np.array([0, 2, 0, 2]),
            None,
        ),
        # y_true 中的类别数量必须等于 y_score 中的列数
        (
            "Parameter 'labels' must be ordered",
            np.array(["a", "b", "c", "c"]),
            ["a", "c", "b"],
        ),
        # labels 参数必须按顺序排列
        (
            (
                "Number of given labels, 2, not equal to the number of columns in "
                "'y_score', 3"
            ),
            np.array([0, 1, 2, 2]),
            [0, 1],
        ),
        # 给定的 labels 数量与 y_score 中的列数不匹配
        (
            (
                "Number of given labels, 2, not equal to the number of columns in "
                "'y_score', 3"
            ),
            np.array(["a", "b", "c", "c"]),
            ["a", "b"],
        ),
        # 给定的 labels 数量与 y_score 中的列数不匹配
        (
            (
                "Number of given labels, 4, not equal to the number of columns in "
                "'y_score', 3"
            ),
            np.array([0, 1, 2, 2]),
            [0, 1, 2, 3],
        ),
        # 给定的 labels 数量与 y_score 中的列数不匹配
        (
            (
                "Number of given labels, 4, not equal to the number of columns in "
                "'y_score', 3"
            ),
            np.array(["a", "b", "c", "c"]),
            ["a", "b", "c", "d"],
        ),
        # 给定的 labels 数量与 y_score 中的列数不匹配
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array(["a", "b", "c", "e"]),
            ["a", "b", "c"],
        ),
        # y_true 中含有不在 labels 参数中的标签
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array(["a", "b", "c", "d"]),
            ["a", "b", "c"],
        ),
        # y_true 中含有不在 labels 参数中的标签
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array([0, 1, 2, 3]),
            [0, 1, 2],
        ),
        # y_true 中含有不在 labels 参数中的标签
    ],
)
# 使用 pytest.mark.parametrize 来提供多组参数化测试数据
# 使用 pytest 的装饰器标记测试函数，参数化 multi_class 参数为 "ovo" 和 "ovr"
@pytest.mark.parametrize("multi_class", ["ovo", "ovr"])
def test_roc_auc_score_multiclass_labels_error(msg, y_true, labels, multi_class):
    # 创建一个包含预测得分的二维 NumPy 数组
    y_scores = np.array(
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]]
    )

    # 使用 pytest 的上下文管理器，测试 roc_auc_score 函数是否会抛出 ValueError 异常，异常信息为 msg
    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_scores, labels=labels, multi_class=multi_class)


# 参数化 msg 和 kwargs，用于测试不同参数下的 roc_auc_score 函数的异常情况
@pytest.mark.parametrize(
    "msg, kwargs",
    [
        (
            (
                r"average must be one of \('macro', 'weighted', None\) for "
                r"multiclass problems"
            ),
            {"average": "samples", "multi_class": "ovo"},
        ),
        (
            (
                r"average must be one of \('micro', 'macro', 'weighted', None\) for "
                r"multiclass problems"
            ),
            {"average": "samples", "multi_class": "ovr"},
        ),
        (
            (
                r"sample_weight is not supported for multiclass one-vs-one "
                r"ROC AUC, 'sample_weight' must be None in this case"
            ),
            {"multi_class": "ovo", "sample_weight": []},
        ),
        (
            (
                r"Partial AUC computation not available in multiclass setting, "
                r"'max_fpr' must be set to `None`, received `max_fpr=0.5` "
                r"instead"
            ),
            {"multi_class": "ovo", "max_fpr": 0.5},
        ),
        (r"multi_class must be in \('ovo', 'ovr'\)", {}),
    ],
)
def test_roc_auc_score_multiclass_error(msg, kwargs):
    # 测试 roc_auc_score 函数在不同参数下是否会抛出 ValueError 异常
    # 使用固定的随机数生成器创建 y_score 数组
    rng = check_random_state(404)
    y_score = rng.rand(20, 3)
    y_prob = softmax(y_score)
    # 生成随机的 y_true 数组，其中元素的取值范围为 [0, 3)
    y_true = rng.randint(0, 3, size=20)
    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_prob, **kwargs)


def test_auc_score_non_binary_class():
    # 测试 roc_auc_score 函数在非二进制分类值情况下是否会抛出 ValueError 异常
    rng = check_random_state(404)
    y_pred = rng.rand(10)
    # y_true 只包含一个类别值，期望会抛出异常，异常信息为 "ROC AUC score is not defined"
    y_true = np.zeros(10, dtype="int")
    err_msg = "ROC AUC score is not defined"
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    y_true = np.ones(10, dtype="int")
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    y_true = np.full(10, -1, dtype="int")
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    # 使用 warnings 模块捕获所有警告信息，这样它们不会被打印出来或中断程序流程
    with warnings.catch_warnings(record=True):
        # 使用给定的种子值（404）生成随机数生成器
        rng = check_random_state(404)
        # 生成一个包含 10 个随机预测值的数组
        y_pred = rng.rand(10)
        # y_true 只包含一个类别值
        y_true = np.zeros(10, dtype="int")
        # 断言 roc_auc_score 函数在给定条件下会抛出 ValueError 异常，并且异常信息与 err_msg 匹配
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
        # y_true 全部为 1 的数组
        y_true = np.ones(10, dtype="int")
        # 断言 roc_auc_score 函数在给定条件下会抛出 ValueError 异常，并且异常信息与 err_msg 匹配
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
        # y_true 全部为 -1 的数组
        y_true = np.full(10, -1, dtype="int")
        # 断言 roc_auc_score 函数在给定条件下会抛出 ValueError 异常，并且异常信息与 err_msg 匹配
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
# 使用 pytest.mark.parametrize 装饰器，对每个测试函数参数化，参数为 CURVE_FUNCS 中的函数
@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_binary_clf_curve_multiclass_error(curve_func):
    # 创建一个随机数生成器对象，种子为 404
    rng = check_random_state(404)
    # 生成一个长度为 10 的随机整数数组作为 y_true
    y_true = rng.randint(0, 3, size=10)
    # 生成一个长度为 10 的随机浮点数数组作为 y_pred
    y_pred = rng.rand(10)
    # 指定错误消息
    msg = "multiclass format is not supported"
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 curve_func 函数，传入 y_true 和 y_pred 作为参数
        curve_func(y_true, y_pred)


@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_binary_clf_curve_implicit_pos_label(curve_func):
    # 检查使用字符串类标签时是否会引发错误，并提供详细信息：
    msg = (
        "y_true takes value in {'a', 'b'} and pos_label is "
        "not specified: either make y_true take "
        "value in {0, 1} or {-1, 1} or pass pos_label "
        "explicitly."
    )
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        # 使用 np.array 创建一个包含字符串数组的 ndarray，指定数据类型为 '<U1'
        curve_func(np.array(["a", "b"], dtype="<U1"), [0.0, 1.0])

    # 再次使用 pytest.raises 检查是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        # 使用 np.array 创建一个包含对象数组的 ndarray，数据类型为 object
        curve_func(np.array(["a", "b"], dtype=object), [0.0, 1.0])

    # 检查可以使用类似整数类标签的浮点类标签：
    y_pred = [0.0, 1.0, 0.2, 0.42]
    # 分别调用 curve_func 函数，传入整数类标签数组和浮点类标签数组，并获取结果
    int_curve = curve_func([0, 1, 1, 0], y_pred)
    float_curve = curve_func([0.0, 1.0, 1.0, 0.0], y_pred)
    # 使用 np.testing.assert_allclose 检查两组曲线是否接近
    for int_curve_part, float_curve_part in zip(int_curve, float_curve):
        np.testing.assert_allclose(int_curve_part, float_curve_part)


# TODO(1.7): Update test to check for error when bytes support is removed.
@ignore_warnings(category=FutureWarning)
@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
@pytest.mark.parametrize("labels_type", ["list", "array"])
def test_binary_clf_curve_implicit_bytes_pos_label(curve_func, labels_type):
    # 检查使用字节类标签时是否会引发错误，并提供详细信息：
    labels = _convert_container([b"a", b"b"], labels_type)
    msg = (
        "y_true takes value in {b'a', b'b'} and pos_label is not "
        "specified: either make y_true take value in {0, 1} or "
        "{-1, 1} or pass pos_label explicitly."
    )
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 curve_func 函数，传入字节类标签数组和浮点数数组作为参数
        curve_func(labels, [0.0, 1.0])


@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_binary_clf_curve_zero_sample_weight(curve_func):
    # 定义 y_true、y_score 和 sample_weight 三个数组
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5]
    sample_weight = [1, 1, 1, 0.5, 0]

    # 分别调用 curve_func 函数，计算两组不同参数的结果
    result_1 = curve_func(y_true, y_score, sample_weight=sample_weight)
    result_2 = curve_func(y_true[:-1], y_score[:-1], sample_weight=sample_weight[:-1])

    # 使用 assert_allclose 检查两组结果的各个数组是否接近
    for arr_1, arr_2 in zip(result_1, result_2):
        np.testing.assert_allclose(arr_1, arr_2)


@pytest.mark.parametrize("drop", [True, False])
def test_precision_recall_curve(drop):
    # 调用 make_prediction 函数生成 y_true、_ 和 y_score
    y_true, _, y_score = make_prediction(binary=True)
    # 调用 _test_precision_recall_curve 函数，测试 Precision-Recall 曲线
    _test_precision_recall_curve(y_true, y_score, drop)

    # 确保 Precision-Recall 曲线上的第一个点是：
    # (p=1.0, r=class balance) 在非平衡数据集上的 [1:] 处
    # 计算精确率和召回率曲线
    p, r, t = precision_recall_curve(y_true[1:], y_score[1:], drop_intermediate=drop)
    # 断言：确保召回率的第一个值为1.0
    assert r[0] == 1.0
    # 断言：确保精确率的第一个值等于去除第一个真实标签后的平均值
    assert p[0] == y_true[1:].mean()

    # 使用{-1, 1}作为标签；确保不修改原始标签
    y_true[np.where(y_true == 0)] = -1
    # 复制一份原始的真实标签
    y_true_copy = y_true.copy()
    # 测试精确率和召回率曲线函数，检验drop参数是否影响结果
    _test_precision_recall_curve(y_true, y_score, drop)
    # 断言：确保复制的标签与原始标签一致
    assert_array_equal(y_true_copy, y_true)

    # 定义示例用的标签和预测概率
    labels = [1, 0, 0, 1]
    predict_probas = [1, 2, 3, 4]
    # 计算示例数据的精确率和召回率曲线
    p, r, t = precision_recall_curve(labels, predict_probas, drop_intermediate=drop)
    if drop:
        # 如果drop参数为True，进行断言检查精确率、召回率和阈值的近似相等性
        assert_allclose(p, [0.5, 0.33333333, 1.0, 1.0])
        assert_allclose(r, [1.0, 0.5, 0.5, 0.0])
        assert_allclose(t, [1, 2, 4])
    else:
        # 如果drop参数为False，进行断言检查精确率、召回率和阈值的近似相等性
        assert_allclose(p, [0.5, 0.33333333, 0.5, 1.0, 1.0])
        assert_allclose(r, [1.0, 0.5, 0.5, 0.5, 0.0])
        assert_allclose(t, [1, 2, 3, 4])
    # 断言：确保精确率的大小等于召回率的大小，且等于阈值的大小加一
    assert p.size == r.size
    assert p.size == t.size + 1
# 定义一个函数用于测试 Precision-Recall 曲线及其下面积
def _test_precision_recall_curve(y_true, y_score, drop):
    # 计算 Precision-Recall 曲线的精度、召回率和阈值
    p, r, thresholds = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
    # 计算慢速方法得到的平均精度和指定的值进行比较
    precision_recall_auc = _average_precision_slow(y_true, y_score)
    assert_array_almost_equal(precision_recall_auc, 0.859, 3)
    # 使用 sklearn 提供的函数计算平均精度分数并进行比较
    assert_array_almost_equal(
        precision_recall_auc, average_precision_score(y_true, y_score)
    )
    # 在处理 0.5 平分的情况下 `_average_precision` 不够精确：采用更大的容忍度
    assert_almost_equal(
        _average_precision(y_true, y_score), precision_recall_auc, decimal=2
    )
    # 确保 Precision 和 Recall 数组大小一致
    assert p.size == r.size
    # 确保阈值数组比 Precision 和 Recall 数组大一
    assert p.size == thresholds.size + 1
    # 在概率只有一个值的情况下进行烟雾测试
    p, r, thresholds = precision_recall_curve(
        y_true, np.zeros_like(y_score), drop_intermediate=drop
    )
    # 确保 Precision 和 Recall 数组大小一致
    assert p.size == r.size
    # 确保阈值数组比 Precision 和 Recall 数组大一


@pytest.mark.parametrize("drop", [True, False])
def test_precision_recall_curve_toydata(drop):
    with np.errstate(all="ignore"):
        # 如果一个类别从未出现，权重应该不是 NaN
        y_true = np.array([[0, 0], [0, 1]])
        y_score = np.array([[0, 0], [0, 1]])
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            # 检查 average_precision_score 对于权重的表现
            assert_allclose(
                average_precision_score(y_true, y_score, average="weighted"), 1
            )


def test_precision_recall_curve_drop_intermediate():
    """检查 `drop_intermediate` 参数的行为."""
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
    # 测试在 `drop_intermediate=True` 时 Precision-Recall 曲线的表现
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, drop_intermediate=True
    )
    # 确保阈值数组的值与预期的一致
    assert_allclose(thresholds, [0.0, 0.7, 1.0])

    # 测试在具有重复分数的情况下删除阈值
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.6, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, drop_intermediate=True
    )
    # 确保阈值数组的值与预期的一致
    assert_allclose(thresholds, [0.0, 0.6, 0.7, 0.8, 0.9, 1.0])

    # 测试 `drop_intermediate=False` 时只保留端点阈值
    y_true = [0, 0, 0, 0]
    y_score = [0.0, 0.1, 0.2, 0.3]
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, drop_intermediate=True
    )
    # 确保阈值数组的值与预期的一致
    assert_allclose(thresholds, [0.0, 0.3])

    # 测试 `drop_intermediate=True` 时保留所有阈值
    y_true = [1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.2, 0.3]
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, drop_intermediate=True
    )
    # 确保阈值数组的值与预期的一致
    assert_allclose(thresholds, [0.0, 0.1, 0.2, 0.3])


def test_average_precision_constant_values():
    # 检查常数预测器的 average_precision_score 是否等于 TPR
    # 生成一个包含 25% 正例的数据集
    y_true = np.zeros(100, dtype=int)
    y_true[::4] = 1
    # 所有样本的预测分数都为 1
    y_score = np.ones(100)
    # 断言：平均精度得分等于0.25，这是因为只有一个阈值，精度是正数占所有召回的比例。
    assert average_precision_score(y_true, y_score) == 0.25
def test_average_precision_score_binary_pos_label_errors():
    # 当 pos_label 不在二进制 y_true 中时引发错误
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    err_msg = r"pos_label=2 is not a valid label. It should be one of \[0, 1\]"
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred, pos_label=2)


def test_average_precision_score_multilabel_pos_label_errors():
    # 对于多标签指示器 y_true，如果 pos_label 不是 1，则引发错误
    y_true = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    err_msg = (
        "Parameter pos_label is fixed to 1 for multilabel-indicator y_true. "
        "Do not set pos_label or set pos_label to 1."
    )
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred, pos_label=0)


def test_average_precision_score_multiclass_pos_label_errors():
    # 对于多类别 y_true，如果 pos_label 不是 1，则引发错误
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array(
        [
            [0.5, 0.2, 0.1],
            [0.4, 0.5, 0.3],
            [0.1, 0.2, 0.6],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
        ]
    )
    err_msg = (
        "Parameter pos_label is fixed to 1 for multiclass y_true. "
        "Do not set pos_label or set pos_label to 1."
    )
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred, pos_label=3)


def test_score_scale_invariance():
    # 测试 average_precision_score 和 roc_auc_score 对概率值的缩放或偏移的不变性
    # 该测试在响应 github 问题 #3864 等问题时扩展（添加了 scaled_down），其中过度的四舍五入对具有非常小 y_score 值的用户造成问题
    y_true, _, y_score = make_prediction(binary=True)

    roc_auc = roc_auc_score(y_true, y_score)
    roc_auc_scaled_up = roc_auc_score(y_true, 100 * y_score)
    roc_auc_scaled_down = roc_auc_score(y_true, 1e-6 * y_score)
    roc_auc_shifted = roc_auc_score(y_true, y_score - 10)
    assert roc_auc == roc_auc_scaled_up
    assert roc_auc == roc_auc_scaled_down
    assert roc_auc == roc_auc_shifted

    pr_auc = average_precision_score(y_true, y_score)
    pr_auc_scaled_up = average_precision_score(y_true, 100 * y_score)
    pr_auc_scaled_down = average_precision_score(y_true, 1e-6 * y_score)
    pr_auc_shifted = average_precision_score(y_true, y_score - 10)
    assert pr_auc == pr_auc_scaled_up
    assert pr_auc == pr_auc_scaled_down
    assert pr_auc == pr_auc_shifted


@pytest.mark.parametrize(
    "y_true,y_score,expected_fpr,expected_fnr",
    # 定义一个包含多个元组的列表，每个元组代表一个模式
    [
        # 第一个元组：模式为 [0, 0, 1], [0, 0.5, 1], [0], [0]
        ([0, 0, 1], [0, 0.5, 1], [0], [0]),
        # 第二个元组：模式为 [0, 0, 1], [0, 0.25, 0.5], [0], [0]
        ([0, 0, 1], [0, 0.25, 0.5], [0], [0]),
        # 第三个元组：模式为 [0, 0, 1], [0.5, 0.75, 1], [0], [0]
        ([0, 0, 1], [0.5, 0.75, 1], [0], [0]),
        # 第四个元组：模式为 [0, 0, 1], [0.25, 0.5, 0.75], [0], [0]
        ([0, 0, 1], [0.25, 0.5, 0.75], [0], [0]),
        # 第五个元组：模式为 [0, 1, 0], [0, 0.5, 1], [0.5], [0]
        ([0, 1, 0], [0, 0.5, 1], [0.5], [0]),
        # 第六个元组：模式为 [0, 1, 0], [0, 0.25, 0.5], [0.5], [0]
        ([0, 1, 0], [0, 0.25, 0.5], [0.5], [0]),
        # 第七个元组：模式为 [0, 1, 0], [0.5, 0.75, 1], [0.5], [0]
        ([0, 1, 0], [0.5, 0.75, 1], [0.5], [0]),
        # 第八个元组：模式为 [0, 1, 0], [0.25, 0.5, 0.75], [0.5], [0]
        ([0, 1, 0], [0.25, 0.5, 0.75], [0.5], [0]),
        # 第九个元组：模式为 [0, 1, 1], [0, 0.5, 1], [0.0], [0]
        ([0, 1, 1], [0, 0.5, 1], [0.0], [0]),
        # 第十个元组：模式为 [0, 1, 1], [0, 0.25, 0.5], [0], [0]
        ([0, 1, 1], [0, 0.25, 0.5], [0], [0]),
        # 第十一个元组：模式为 [0, 1, 1], [0.5, 0.75, 1], [0], [0]
        ([0, 1, 1], [0.5, 0.75, 1], [0], [0]),
        # 第十二个元组：模式为 [0, 1, 1], [0.25, 0.5, 0.75], [0], [0]
        ([0, 1, 1], [0.25, 0.5, 0.75], [0], [0]),
        # 第十三个元组：模式为 [1, 0, 0], [0, 0.5, 1], [1, 1, 0.5], [0, 1, 1]
        ([1, 0, 0], [0, 0.5, 1], [1, 1, 0.5], [0, 1, 1]),
        # 第十四个元组：模式为 [1, 0, 0], [0, 0.25, 0.5], [1, 1, 0.5], [0, 1, 1]
        ([1, 0, 0], [0, 0.25, 0.5], [1, 1, 0.5], [0, 1, 1]),
        # 第十五个元组：模式为 [1, 0, 0], [0.5, 0.75, 1], [1, 1, 0.5], [0, 1, 1]
        ([1, 0, 0], [0.5, 0.75, 1], [1, 1, 0.5], [0, 1, 1]),
        # 第十六个元组：模式为 [1, 0, 0], [0.25, 0.5, 0.75], [1, 1, 0.5], [0, 1, 1]
        ([1, 0, 0], [0.25, 0.5, 0.75], [1, 1, 0.5], [0, 1, 1]),
        # 第十七个元组：模式为 [1, 0, 1], [0, 0.5, 1], [1, 1, 0], [0, 0.5, 0.5]
        ([1, 0, 1], [0, 0.5, 1], [1, 1, 0], [0, 0.5, 0.5]),
        # 第十八个元组：模式为 [1, 0, 1], [0, 0.25, 0.5], [1, 1, 0], [0, 0.5, 0.5]
        ([1, 0, 1], [0, 0.25, 0.5], [1, 1, 0], [0, 0.5, 0.5]),
        # 第十九个元组：模式为 [1, 0, 1], [0.5, 0.75, 1], [1, 1, 0], [0, 0.5, 0.5]
        ([1, 0, 1], [0.5, 0.75, 1], [1, 1, 0], [0, 0.5, 0.5]),
        # 第二十个元组：模式为 [1, 0, 1], [0.25, 0.5, 0.75], [1, 1, 0], [0, 0.5, 0.5]
        ([1, 0, 1], [0.25, 0.5, 0.75], [1, 1, 0], [0, 0.5, 0.5]),
    ],
# 测试DET曲线生成函数对小例子的效果
def test_det_curve_toydata(y_true, y_score, expected_fpr, expected_fnr):
    # 调用det_curve函数计算真阳性率（FPR）、假阴性率（FNR）以及阈值（threshold）
    fpr, fnr, _ = det_curve(y_true, y_score)

    # 使用assert_allclose函数检查计算出的FPR和预期FPR的近似程度
    assert_allclose(fpr, expected_fpr)
    # 使用assert_allclose函数检查计算出的FNR和预期FNR的近似程度
    assert_allclose(fnr, expected_fnr)


# 参数化测试，测试DET曲线生成函数在处理并列情况时的表现
@pytest.mark.parametrize(
    "y_true,y_score,expected_fpr,expected_fnr",
    [
        ([1, 0], [0.5, 0.5], [1], [0]),
        ([0, 1], [0.5, 0.5], [1], [0]),
        ([0, 0, 1], [0.25, 0.5, 0.5], [0.5], [0]),
        ([0, 1, 0], [0.25, 0.5, 0.5], [0.5], [0]),
        ([0, 1, 1], [0.25, 0.5, 0.5], [0], [0]),
        ([1, 0, 0], [0.25, 0.5, 0.5], [1], [0]),
        ([1, 0, 1], [0.25, 0.5, 0.5], [1], [0]),
        ([1, 1, 0], [0.25, 0.5, 0.5], [1], [0]),
    ],
)
def test_det_curve_tie_handling(y_true, y_score, expected_fpr, expected_fnr):
    # 调用det_curve函数计算真阳性率（FPR）、假阴性率（FNR）以及阈值（threshold）
    fpr, fnr, _ = det_curve(y_true, y_score)

    # 使用assert_allclose函数检查计算出的FPR和预期FPR的近似程度
    assert_allclose(fpr, expected_fpr)
    # 使用assert_allclose函数检查计算出的FNR和预期FNR的近似程度
    assert_allclose(fnr, expected_fnr)


# 测试DET曲线生成函数的基本正确性
def test_det_curve_sanity_check():
    # 使用assert_allclose函数检查两组完全重复输入的DET曲线结果是否一致
    assert_allclose(
        det_curve([0, 0, 1], [0, 0.5, 1]),
        det_curve([0, 0, 0, 0, 1, 1], [0, 0, 0.5, 0.5, 1, 1]),
    )


# 参数化测试，测试DET曲线生成函数在处理常数预测分数时的表现
@pytest.mark.parametrize("y_score", [(0), (0.25), (0.5), (0.75), (1)])
def test_det_curve_constant_scores(y_score):
    # 调用det_curve函数计算真阳性率（FPR）、假阴性率（FNR）以及阈值（threshold）
    fpr, fnr, threshold = det_curve(
        y_true=[0, 1, 0, 1, 0, 1], y_score=np.full(6, y_score)
    )

    # 使用assert_allclose函数检查计算出的FPR和预期FPR的近似程度
    assert_allclose(fpr, [1])
    # 使用assert_allclose函数检查计算出的FNR和预期FNR的近似程度
    assert_allclose(fnr, [0])
    # 使用assert_allclose函数检查计算出的阈值和预期阈值的近似程度
    assert_allclose(threshold, [y_score])


# 参数化测试，测试DET曲线生成函数在处理完美预测分数时的表现
@pytest.mark.parametrize(
    "y_true",
    [
        ([0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 1, 1]),
        ([0, 0, 0, 1, 1, 1]),
        ([0, 0, 1, 1, 1, 1]),
        ([0, 1, 1, 1, 1, 1]),
    ],
)
def test_det_curve_perfect_scores(y_true):
    # 调用det_curve函数计算真阳性率（FPR）、假阴性率（FNR）以及阈值（threshold）
    fpr, fnr, _ = det_curve(y_true=y_true, y_score=y_true)

    # 使用assert_allclose函数检查计算出的FPR和预期FPR的近似程度
    assert_allclose(fpr, [0])
    # 使用assert_allclose函数检查计算出的FNR和预期FNR的近似程度
    assert_allclose(fnr, [0])


# 参数化测试，测试DET曲线生成函数在处理错误输入时的异常抛出
@pytest.mark.parametrize(
    "y_true, y_pred, err_msg",
    [
        ([0, 1], [0, 0.5, 1], "inconsistent numbers of samples"),
        ([0, 1, 1], [0, 0.5], "inconsistent numbers of samples"),
        ([0, 0, 0], [0, 0.5, 1], "Only one class present in y_true"),
        ([1, 1, 1], [0, 0.5, 1], "Only one class present in y_true"),
        (
            ["cancer", "cancer", "not cancer"],
            [0.2, 0.3, 0.8],
            "pos_label is not specified",
        ),
    ],
)
def test_det_curve_bad_input(y_true, y_pred, err_msg):
    # 使用pytest.raises检查是否抛出预期的ValueError异常，异常信息需要匹配给定的err_msg
    with pytest.raises(ValueError, match=err_msg):
        det_curve(y_true, y_pred)


# 测试DET曲线生成函数在指定正例标签时的表现
def test_det_curve_pos_label():
    y_true = ["cancer"] * 3 + ["not cancer"] * 7
    y_pred_pos_not_cancer = np.array([0.1, 0.4, 0.6, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
    y_pred_pos_cancer = 1 - y_pred_pos_not_cancer

    # 调用det_curve函数计算真阳性率（FPR）、假阴性率（FNR）以及阈值（threshold）
    fpr_pos_cancer, fnr_pos_cancer, th_pos_cancer = det_curve(
        y_true,
        y_pred_pos_cancer,
        pos_label="cancer",
    )
    # 调用函数det_curve计算正例为“not cancer”时的假正率（fpr）、假阴率（fnr）和阈值（th）
    fpr_pos_not_cancer, fnr_pos_not_cancer, th_pos_not_cancer = det_curve(
        y_true,
        y_pred_pos_not_cancer,
        pos_label="not cancer",
    )

    # 检查第一个阈值是否根据将哪个标签视为正例而变化
    assert th_pos_cancer[0] == pytest.approx(0.4)
    assert th_pos_not_cancer[0] == pytest.approx(0.2)

    # 检查假正率（fpr）和假阴率（fnr）的对称性
    assert_allclose(fpr_pos_cancer, fnr_pos_not_cancer[::-1])
    assert_allclose(fnr_pos_cancer, fpr_pos_not_cancer[::-1])
# 定义一个函数用于测试 LRAP 分数的计算
def check_lrap_toy(lrap_score):
    # 检查在几个小例子上的计算是否正确
    assert_almost_equal(lrap_score([[0, 1]], [[0.25, 0.75]]), 1)
    assert_almost_equal(lrap_score([[0, 1]], [[0.75, 0.25]]), 1 / 2)
    assert_almost_equal(lrap_score([[1, 1]], [[0.75, 0.25]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 1 / 3)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.25, 0.5, 0.75]]), (2 / 3 + 1 / 1) / 2
    )
    assert_almost_equal(
        lrap_score([[1, 1, 0]], [[0.25, 0.5, 0.75]]), (2 / 3 + 1 / 2) / 2
    )

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.75, 0.5, 0.25]]), 1 / 3)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.75, 0.5, 0.25]]), 1 / 2)
    assert_almost_equal(
        lrap_score([[0, 1, 1]], [[0.75, 0.5, 0.25]]), (1 / 2 + 2 / 3) / 2
    )
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 1]], [[0.75, 0.5, 0.25]]), (1 + 2 / 3) / 2)
    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.75, 0.5, 0.25]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.5, 0.75, 0.25]]), 1 / 3)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.5, 0.75, 0.25]]), (1 + 2 / 3) / 2)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.5, 0.75, 0.25]]), 1 / 2)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.5, 0.75, 0.25]]), (1 / 2 + 2 / 3) / 2
    )
    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 1)

    # 处理并列情况
    assert_almost_equal(lrap_score([[1, 0]], [[0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1]], [[0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[1, 1]], [[0.5, 0.5]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 1 / 3)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.25, 0.5, 0.5]]), (2 / 3 + 1 / 2) / 2
    )
    assert_almost_equal(
        lrap_score([[1, 1, 0]], [[0.25, 0.5, 0.5]]), (2 / 3 + 1 / 2) / 2
    )
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 1)

    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.5, 0.5, 0.5]]), 2 / 3)

    assert_almost_equal(lrap_score([[1, 1, 1, 0]], [[0.5, 0.5, 0.5, 0.5]]), 3 / 4)


# 定义一个函数用于测试随机状态下的零或全相关标签
def check_zero_or_all_relevant_labels(lrap_score):
    random_state = check_random_state(0)
    # 对于不同的标签数目，进行多次测试
    for n_labels in range(2, 5):
        # 生成随机得分，形状为(1, n_labels)
        y_score = random_state.uniform(size=(1, n_labels))
        # 创建一个与y_score相同形状的全零数组，用于测试无关标签的情况
        y_score_ties = np.zeros_like(y_score)
    
        # 情况1：没有相关标签
        y_true = np.zeros((1, n_labels))
        # 断言：计算 lrap 分数应为 1.0
        assert lrap_score(y_true, y_score) == 1.0
        # 断言：计算 lrap 分数应为 1.0（即使有平局的情况）
        assert lrap_score(y_true, y_score_ties) == 1.0
    
        # 情况2：只有相关标签
        y_true = np.ones((1, n_labels))
        # 断言：计算 lrap 分数应为 1.0
        assert lrap_score(y_true, y_score) == 1.0
        # 断言：计算 lrap 分数应为 1.0（即使有平局的情况）
        assert lrap_score(y_true, y_score_ties) == 1.0
    
    # 特殊情况：只有一个标签的情况
    assert_almost_equal(
        lrap_score([[1], [0], [1], [0]], [[0.5], [0.5], [0.5], [0.5]]), 1.0
    )
# 检查 LRAP 分数是否引发错误
def check_lrap_error_raised(lrap_score):
    # 如果不是适当的格式，引发值错误
    with pytest.raises(ValueError):
        lrap_score([0, 1, 0], [0.25, 0.3, 0.2])
    with pytest.raises(ValueError):
        lrap_score([0, 1, 2], [[0.25, 0.75, 0.0], [0.7, 0.3, 0.0], [0.8, 0.2, 0.0]])
    with pytest.raises(ValueError):
        lrap_score(
            [(0), (1), (2)], [[0.25, 0.75, 0.0], [0.7, 0.3, 0.0], [0.8, 0.2, 0.0]]
        )

    # 检查 y_true.shape != y_score.shape 是否引发适当的异常
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [0, 1])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0], [1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0], [1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0], [1]])


# 检查 LRAP 只处理并列情况
def check_lrap_only_ties(lrap_score):
    # 检查分数中的并列处理
    # 基本检查，仅有并列并逐渐增加标签空间
    for n_labels in range(2, 10):
        y_score = np.ones((1, n_labels))

        # 检查不断增加的并列相关标签数量
        for n_relevant in range(1, n_labels):
            # 检查一系列位置
            for pos in range(n_labels - n_relevant):
                y_true = np.zeros((1, n_labels))
                y_true[0, pos : pos + n_relevant] = 1
                assert_almost_equal(lrap_score(y_true, y_score), n_relevant / n_labels)


# 检查 LRAP 不处理并列且逐渐增加分数
def check_lrap_without_tie_and_increasing_score(lrap_score):
    # 检查标签排序平均精度是否适用于不同情况
    # 基本检查，增加标签空间大小并降低分数
    for n_labels in range(2, 10):
        y_score = n_labels - (np.arange(n_labels).reshape((1, n_labels)) + 1)

        # 第一个和最后一个
        y_true = np.zeros((1, n_labels))
        y_true[0, 0] = 1
        y_true[0, -1] = 1
        assert_almost_equal(lrap_score(y_true, y_score), (2 / n_labels + 1) / 2)

        # 检查不断增加的并列相关标签数量
        for n_relevant in range(1, n_labels):
            # 检查一系列位置
            for pos in range(n_labels - n_relevant):
                y_true = np.zeros((1, n_labels))
                y_true[0, pos : pos + n_relevant] = 1
                assert_almost_equal(
                    lrap_score(y_true, y_score),
                    sum(
                        (r + 1) / ((pos + r + 1) * n_relevant)
                        for r in range(n_relevant)
                    ),
                )


def _my_lrap(y_true, y_score):
    """简单实现的标签排序平均精度（LRAP）"""
    # 检查输入的长度是否一致
    check_consistent_length(y_true, y_score)
    # 将 y_true 和 y_score 转换为数组
    y_true = check_array(y_true)
    y_score = check_array(y_score)
    n_samples, n_labels = y_true.shape
    # 创建一个空的 numpy 数组用于存储分数，大小为 n_samples
    score = np.empty((n_samples,))
    # 遍历每个样本
    for i in range(n_samples):
        # 最佳排名对应于 1，排名比 1 更高的更差。
        # 最佳逆序排名对应于 n_labels。
        unique_rank, inv_rank = np.unique(y_score[i], return_inverse=True)
        n_ranks = unique_rank.size
        # 计算真实排名，考虑到可能存在的并列情况
        rank = n_ranks - inv_rank

        # 需要校正排名以考虑并列情况
        # 例如：排名 1 并列意味着这两个标签的排名都是 2。
        corr_rank = np.bincount(rank, minlength=n_ranks + 1).cumsum()
        rank = corr_rank[rank]

        # 找出真实标签中的非零索引，即相关的标签
        relevant = y_true[i].nonzero()[0]
        
        # 如果没有相关的标签或者所有标签都相关，则将分数设置为 1
        if relevant.size == 0 or relevant.size == n_labels:
            score[i] = 1
            continue

        # 初始化分数为 0.0
        score[i] = 0.0
        # 遍历每个相关的标签
        for label in relevant:
            # 统计排名比当前标签好（排名更低）的相关标签数量
            n_ranked_above = sum(rank[r] <= rank[label] for r in relevant)

            # 根据当前标签的排名权重来加权
            score[i] += n_ranked_above / rank[label]

        # 对分数进行平均，以相关标签的数量为分母
        score[i] /= relevant.size

    # 返回分数的平均值作为最终结果
    return score.mean()
# 定义一个函数用于检查备选的 LRAP 实现是否正确
def check_alternative_lrap_implementation(
    lrap_score, n_classes=5, n_samples=20, random_state=0
):
    # 生成一个多标签分类的模拟数据集，其中包含特征数为 1 的特征向量和相关联的真实标签
    _, y_true = make_multilabel_classification(
        n_features=1,
        allow_unlabeled=False,
        random_state=random_state,
        n_classes=n_classes,
        n_samples=n_samples,
    )

    # 使用 _sparse_random_matrix 函数生成一个随机的 y_score 矩阵
    y_score = _sparse_random_matrix(
        n_components=y_true.shape[0],
        n_features=y_true.shape[1],
        random_state=random_state,
    )

    # 如果 y_score 对象具有 toarray 方法，则将其转换为稠密数组
    if hasattr(y_score, "toarray"):
        y_score = y_score.toarray()

    # 计算使用标签排序平均精度的得分
    score_lrap = label_ranking_average_precision_score(y_true, y_score)
    # 计算自定义的 LRAP 得分
    score_my_lrap = _my_lrap(y_true, y_score)
    # 断言两种 LRAP 计算方法的得分几乎相等
    assert_almost_equal(score_lrap, score_my_lrap)

    # 均匀分布的 y_score
    random_state = check_random_state(random_state)
    y_score = random_state.uniform(size=(n_samples, n_classes))
    # 计算均匀分布下的 LRAP 得分
    score_lrap = label_ranking_average_precision_score(y_true, y_score)
    # 计算自定义的 LRAP 得分
    score_my_lrap = _my_lrap(y_true, y_score)
    # 断言两种 LRAP 计算方法的得分几乎相等
    assert_almost_equal(score_lrap, score_my_lrap)


# 使用 pytest 的 parametrize 装饰器指定多个参数化测试用例，测试不同的 LRAP 计算函数
@pytest.mark.parametrize(
    "check",
    (
        check_lrap_toy,
        check_lrap_without_tie_and_increasing_score,
        check_lrap_only_ties,
        check_zero_or_all_relevant_labels,
    ),
)
@pytest.mark.parametrize("func", (label_ranking_average_precision_score, _my_lrap))
def test_label_ranking_avp(check, func):
    # 调用给定的检查函数，用指定的 LRAP 计算函数计算结果并进行检查
    check(func)


# 测试当使用 label_ranking_average_precision_score 函数时是否会引发错误
def test_lrap_error_raised():
    # 检查在调用 label_ranking_average_precision_score 函数时是否会引发错误
    check_lrap_error_raised(label_ranking_average_precision_score)


# 使用 pytest 的 parametrize 装饰器指定多个参数化测试用例，测试不同的 n_samples、n_classes 和 random_state 组合下的备选 LRAP 实现
@pytest.mark.parametrize("n_samples", (1, 2, 8, 20))
@pytest.mark.parametrize("n_classes", (2, 5, 10))
@pytest.mark.parametrize("random_state", range(1))
def test_alternative_lrap_implementation(n_samples, n_classes, random_state):
    # 调用检查备选 LRAP 实现函数，使用指定的参数进行测试
    check_alternative_lrap_implementation(
        label_ranking_average_precision_score, n_classes, n_samples, random_state
    )


# 测试 LRAP 在样本权重为零标签的情况下的计算结果
def test_lrap_sample_weighting_zero_labels():
    # 定义一个包含特定标签和得分的真实标签 y_true 和预测得分 y_score
    y_true = np.array([[1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]], dtype=bool)
    y_score = np.array(
        [[0.3, 0.4, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
    )
    # 样本权重和预期的样本 LRAP
    samplewise_lraps = np.array([0.5, 0.75, 1.0])
    sample_weight = np.array([1.0, 1.0, 0.0])

    # 使用样本权重计算 LRAP，并断言其与预期结果的准确性
    assert_almost_equal(
        label_ranking_average_precision_score(
            y_true, y_score, sample_weight=sample_weight
        ),
        np.sum(sample_weight * samplewise_lraps) / np.sum(sample_weight),
    )


# 测试覆盖误差函数的计算
def test_coverage_error():
    # 简单的测试用例，使用 coverage_error 函数计算覆盖误差并进行断言
    assert_almost_equal(coverage_error([[0, 1]], [[0.25, 0.75]]), 1)
    assert_almost_equal(coverage_error([[0, 1]], [[0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1]], [[0.75, 0.25]]), 2)
    # 对于单一点的情况，计算其覆盖误差，预期结果是 0
    assert_almost_equal(coverage_error([[0, 0]], [[0.75, 0.25]]), 0)
    
    # 对于三个点的情况，计算其覆盖误差，预期结果是 0
    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.25, 0.5, 0.75]]), 0)
    # 一个点与三个点的第一个维度相差较大，计算其覆盖误差，预期结果是 1
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 1)
    # 一个点与三个点的第二个维度相差较大，计算其覆盖误差，预期结果是 2
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 2)
    # 一个点与三个点的前两个维度都相差较大，计算其覆盖误差，预期结果是 2
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 2)
    # 一个点与三个点的第三个维度相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 3)
    # 一个点与三个点的第一、三个维度相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.25, 0.5, 0.75]]), 3)
    # 一个点与三个点的第二、三个维度相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.25, 0.5, 0.75]]), 3)
    # 一个点与三个点的所有维度都相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.25, 0.5, 0.75]]), 3)
    
    # 对于三个点的情况，计算其覆盖误差，预期结果是 0
    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.75, 0.5, 0.25]]), 0)
    # 一个点与三个点的第一个维度相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.75, 0.5, 0.25]]), 3)
    # 一个点与三个点的第二个维度相差较大，计算其覆盖误差，预期结果是 2
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.75, 0.5, 0.25]]), 2)
    # 一个点与三个点的前两个维度都相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.75, 0.5, 0.25]]), 3)
    # 一个点与三个点的第三个维度相差较大，计算其覆盖误差，预期结果是 1
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.75, 0.5, 0.25]]), 1)
    # 一个点与三个点的第一、三个维度相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.75, 0.5, 0.25]]), 3)
    # 一个点与三个点的第二、三个维度相差较大，计算其覆盖误差，预期结果是 2
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.75, 0.5, 0.25]]), 2)
    # 一个点与三个点的所有维度都相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.75, 0.5, 0.25]]), 3)
    
    # 对于三个点的情况，计算其覆盖误差，预期结果是 0
    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.5, 0.75, 0.25]]), 0)
    # 一个点与三个点的第一个维度相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.5, 0.75, 0.25]]), 3)
    # 一个点与三个点的第二个维度相差较大，计算其覆盖误差，预期结果是 1
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    # 一个点与三个点的前两个维度都相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.5, 0.75, 0.25]]), 3)
    # 一个点与三个点的第三个维度相差较大，计算其覆盖误差，预期结果是 2
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.5, 0.75, 0.25]]), 2)
    # 一个点与三个点的第一、三个维度相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.5, 0.75, 0.25]]), 3)
    # 一个点与三个点的第二、三个维度相差较大，计算其覆盖误差，预期结果是 2
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.5, 0.75, 0.25]]), 2)
    # 一个点与三个点的所有维度都相差较大，计算其覆盖误差，预期结果是 3
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 3)
    
    # 对于非平凡情况，计算其覆盖误差，预期结果是 (1 + 3) / 2.0
    assert_almost_equal(
        coverage_error([[0, 1, 0], [1, 1, 0]], [[0.1, 10.0, -3], [0, 1, 3]]),
        (1 + 3) / 2.0,
    )
    
    # 对于三个点的情况，计算其覆盖误差，预期结果是 (1 + 3 + 3) / 3.0
    assert_almost_equal(
        coverage_error(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [0, 1, 3], [0, 2, 0]]
        ),
        (1 + 3 + 3) / 3.0,
    )
    
    # 对于三个点的情况，计
def test_coverage_tie_handling():
    # 检查 coverage_error 函数对于特定输入的输出是否接近预期值
    assert_almost_equal(coverage_error([[0, 0]], [[0.5, 0.5]]), 0)
    assert_almost_equal(coverage_error([[1, 0]], [[0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1]], [[0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[1, 1]], [[0.5, 0.5]]), 2)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 3)


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        ([1, 0, 1], [0.25, 0.5, 0.5]),  # 测试参数化情况：y_true 是一维列表，y_score 是一维列表
        ([1, 0, 1], [[0.25, 0.5, 0.5]]),  # 测试参数化情况：y_true 是一维列表，y_score 是二维列表
        ([[1, 0, 1]], [0.25, 0.5, 0.5]),  # 测试参数化情况：y_true 是二维列表，y_score 是一维列表
    ],
)
def test_coverage_1d_error_message(y_true, y_score):
    # 非回归测试，验证 https://github.com/scikit-learn/scikit-learn/issues/23368 是否修复
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
        coverage_error(y_true, y_score)


def test_label_ranking_loss():
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.25, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.75, 0.25]]), 1)

    assert_almost_equal(label_ranking_loss([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 2 / 2)
    assert_almost_equal(label_ranking_loss([[1, 0, 1]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[1, 1, 0]], [[0.25, 0.5, 0.75]]), 2 / 2)

    # 未定义的度量 - 排名无关紧要
    assert_almost_equal(label_ranking_loss([[0, 0]], [[0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1]], [[0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 0]], [[0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1]], [[0.5, 0.5]]), 0)

    assert_almost_equal(label_ranking_loss([[0, 0, 0]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 0, 0]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 0)

    # 非平凡情况
    assert_almost_equal(
        label_ranking_loss([[0, 1, 0], [1, 1, 0]], [[0.1, 10.0, -3], [0, 1, 3]]),
        (0 + 2 / 2) / 2.0,
    )
    # 对于标签排名损失函数的断言测试
    assert_almost_equal(
        # 调用 label_ranking_loss 函数计算标签排名损失
        label_ranking_loss(
            # 第一个参数为真实标签的列表
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
            # 第二个参数为预测标签的列表
            [[0.1, 10, -3], [0, 1, 3], [0, 2, 0]]
        ),
        # 期望的损失值，计算方式为 (0 + 2 / 2 + 1 / 2) / 3.0
        (0 + 2 / 2 + 1 / 2) / 3.0,
    )
    
    # 对于标签排名损失函数的断言测试
    assert_almost_equal(
        # 调用 label_ranking_loss 函数计算标签排名损失
        label_ranking_loss(
            # 第一个参数为真实标签的列表
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
            # 第二个参数为预测标签的列表
            [[0.1, 10, -3], [3, 1, 3], [0, 2, 0]]
        ),
        # 期望的损失值，计算方式为 (0 + 2 / 2 + 1 / 2) / 3.0
        (0 + 2 / 2 + 1 / 2) / 3.0,
    )
# 使用 pytest.mark.parametrize 注解，为测试函数 test_label_ranking_loss_sparse 参数化测试数据
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_label_ranking_loss_sparse(csr_container):
    # 断言 label_ranking_loss 函数对给定的稀疏矩阵 csr_container 和预测值 [[0.1, 10, -3], [3, 1, 3]] 的结果接近于 (0 + 2 / 2) / 2.0
    assert_almost_equal(
        label_ranking_loss(
            csr_container(np.array([[0, 1, 0], [1, 1, 0]])), [[0.1, 10, -3], [3, 1, 3]]
        ),
        (0 + 2 / 2) / 2.0,
    )


# 测试函数，验证 label_ranking_loss 函数对不合适的输入形状是否会引发 ValueError 异常
def test_ranking_appropriate_input_shape():
    # 使用 pytest.raises 检查是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [0, 1])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0], [1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0], [1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0], [1]])


# 测试函数，验证 label_ranking_loss 函数对于 ties（并列情况）的处理是否正确
def test_ranking_loss_ties_handling():
    # 断言 label_ranking_loss 函数对不同 ties 情况的处理结果是否符合预期
    assert_almost_equal(label_ranking_loss([[1, 0]], [[0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[1, 0, 1]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[1, 1, 0]], [[0.25, 0.5, 0.5]]), 1)


# 测试函数，验证 DCG（折损累计增益）评分的计算
def test_dcg_score():
    # 使用 make_multilabel_classification 创建多标签分类问题，生成 y_true 和相应的 y_score
    _, y_true = make_multilabel_classification(random_state=0, n_classes=10)
    y_score = -y_true + 1
    # 调用 _test_dcg_score_for 函数进行 DCG 评分测试
    _test_dcg_score_for(y_true, y_score)
    # 使用随机数生成器生成不同形状的 y_true 和 y_score 进行 DCG 评分测试
    y_true, y_score = np.random.RandomState(0).random_sample((2, 100, 10))
    _test_dcg_score_for(y_true, y_score)


# 辅助函数，用于测试 DCG（折损累计增益）评分的计算
def _test_dcg_score_for(y_true, y_score):
    # 计算折扣系数
    discount = np.log2(np.arange(y_true.shape[1]) + 2)
    # 计算理想排序的 DCG 分数
    ideal = _dcg_sample_scores(y_true, y_true)
    # 计算给定 y_score 的 DCG 分数
    score = _dcg_sample_scores(y_true, y_score)
    # 断言计算出的分数不超过理想分数
    assert (score <= ideal).all()
    # 断言限定 k 值后的 DCG 分数不超过理想分数
    assert (_dcg_sample_scores(y_true, y_true, k=5) <= ideal).all()
    # 断言理想分数和 score 的形状正确
    assert ideal.shape == (y_true.shape[0],)
    assert score.shape == (y_true.shape[0],)
    # 使用 pytest.approx 断言理想分数的计算结果
    assert ideal == pytest.approx((np.sort(y_true)[:, ::-1] / discount).sum(axis=1))


# 测试函数，验证 DCG（折损累计增益）评分在 ties（并列情况）下的计算
def test_dcg_ties():
    # 创建一个 y_true 数组，包含从 0 到 4 的序列
    y_true = np.asarray([np.arange(5)])
    y_score = np.zeros(y_true.shape)
    # 计算未处理 ties 的 DCG 分数
    dcg = _dcg_sample_scores(y_true, y_score)
    # 计算忽略 ties 的 DCG 分数
    dcg_ignore_ties = _dcg_sample_scores(y_true, y_score, ignore_ties=True)
    # 计算折扣系数
    discounts = 1 / np.log2(np.arange(2, 7))
    # 使用 pytest.approx 断言未处理 ties 的 DCG 分数
    assert dcg == pytest.approx([discounts.sum() * y_true.mean()])
    # 使用 pytest.approx 断言忽略 ties 的 DCG 分数
    assert dcg_ignore_ties == pytest.approx([(discounts * y_true[:, ::-1]).sum()])
    # 修改 y_score 以引入 ties，并重新计算 DCG 分数
    y_score[0, 3:] = 1
    dcg = _dcg_sample_scores(y_true, y_score)
    dcg_ignore_ties = _dcg_sample_scores(y_true, y_score, ignore_ties=True)
    # 断言语句，验证 dcg_ignore_ties 是否与 [(discounts * y_true[:, ::-1]).sum()] 近似相等
    assert dcg_ignore_ties == pytest.approx([(discounts * y_true[:, ::-1]).sum()])
    
    # 断言语句，验证 dcg 是否与给定表达式的计算结果近似相等
    assert dcg == pytest.approx(
        [
            # 计算折扣在前两个位置的和乘以对应 y_true 第一行的后三个元素的均值
            discounts[:2].sum() * y_true[0, 3:].mean()
            # 加上折扣在第三个位置之后的和乘以对应 y_true 第一行的前三个元素的均值
            + discounts[2:].sum() * y_true[0, :3].mean()
        ]
    )
# 测试忽略平局时的 NDCG 计算结果，设置一个 2x6 的 NumPy 数组 a
def test_ndcg_ignore_ties_with_k():
    a = np.arange(12).reshape((2, 6))
    # 断言忽略平局时，使用 k=3 计算的 ndcg_score 结果应该等于相同设置下的另一个 ndcg_score 结果
    assert ndcg_score(a, a, k=3, ignore_ties=True) == pytest.approx(
        ndcg_score(a, a, k=3, ignore_ties=True)
    )


# 检查当 `y_true` 包含负值时，`ndcg_score` 应该引发异常
def test_ndcg_negative_ndarray_error():
    """Check `ndcg_score` exception when `y_true` contains negative values."""
    y_true = np.array([[-0.89, -0.53, -0.47, 0.39, 0.56]])
    y_score = np.array([[0.07, 0.31, 0.75, 0.33, 0.27]])
    expected_message = "ndcg_score should not be used on negative y_true values"
    # 使用 pytest 的 raises 断言来检测是否引发 ValueError 异常，并检查异常消息
    with pytest.raises(ValueError, match=expected_message):
        ndcg_score(y_true, y_score)


# 检查 NDCG 在特定情况下的不变性
def test_ndcg_invariant():
    y_true = np.arange(70).reshape(7, 10)
    # 添加随机噪声到 y_true，计算 ndcg_score
    y_score = y_true + np.random.RandomState(0).uniform(-0.2, 0.2, size=y_true.shape)
    ndcg = ndcg_score(y_true, y_score)
    # 在忽略平局的情况下再次计算 ndcg_score，并与未忽略平局的结果进行比较
    ndcg_no_ties = ndcg_score(y_true, y_score, ignore_ties=True)
    assert ndcg == pytest.approx(ndcg_no_ties)
    # 断言 ndcg 的结果应该接近 1.0
    assert ndcg == pytest.approx(1.0)
    # 将 y_score 所有值增加 1000，再次计算 ndcg_score，应该仍然接近 1.0
    y_score += 1000
    assert ndcg_score(y_true, y_score) == pytest.approx(1.0)


# 使用参数化测试来测试在不同 ignore_ties 值下的 NDCG 计算
@pytest.mark.parametrize("ignore_ties", [True, False])
def test_ndcg_toy_examples(ignore_ties):
    y_true = 3 * np.eye(7)[:5]
    y_score = np.tile(np.arange(6, -1, -1), (5, 1))
    y_score_noisy = y_score + np.random.RandomState(0).uniform(
        -0.2, 0.2, size=y_score.shape
    )
    # 测试 _dcg_sample_scores 函数在不同设置下的计算结果是否与预期接近
    assert _dcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log2(np.arange(2, 7)))
    assert _dcg_sample_scores(
        y_true, y_score_noisy, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log2(np.arange(2, 7)))
    # 测试 _ndcg_sample_scores 函数在不同设置下的计算结果是否与预期接近
    assert _ndcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(1 / np.log2(np.arange(2, 7)))
    assert _dcg_sample_scores(
        y_true, y_score, log_base=10, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log10(np.arange(2, 7)))
    # 断言 ndcg_score 函数在不同设置下的计算结果是否与预期接近
    assert ndcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        (1 / np.log2(np.arange(2, 7))).mean()
    )
    assert dcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        (3 / np.log2(np.arange(2, 7))).mean()
    )
    y_true = 3 * np.ones((5, 7))
    expected_dcg_score = (3 / np.log2(np.arange(2, 9))).sum()
    # 测试 _dcg_sample_scores 函数在不同设置下的计算结果是否与预期接近
    assert _dcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(expected_dcg_score * np.ones(5))
    # 测试 _ndcg_sample_scores 函数在不同设置下的计算结果是否与预期接近
    assert _ndcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(np.ones(5))
    # 断言 dcg_score 和 ndcg_score 函数在不同设置下的计算结果是否与预期接近
    assert dcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        expected_dcg_score
    )
    assert ndcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(1.0)


# 检查当只有一个文档时，计算 NDCG 是否会引发错误
def test_ndcg_error_single_document():
    """Check that we raise an informative error message when trying to
    compute NDCG with a single document."""
    err_msg = (
        "Computing NDCG is only meaningful when there is more than 1 document. "
        "Got 1 instead."
    )
    # 使用 pytest 框架中的 `raises` 上下文管理器来测试是否会抛出指定类型的异常，并匹配异常消息字符串
    with pytest.raises(ValueError, match=err_msg):
        # 调用 ndcg_score 函数，传入两个参数列表 [[1]] 和 [[1]] 进行测试
        ndcg_score([[1]], [[1]])
# 定义一个测试函数，用于测试 ndcg_score 函数的功能
def test_ndcg_score():
    # 使用 make_multilabel_classification 生成随机的多标签分类数据集，获取其中的真实标签 y_true
    _, y_true = make_multilabel_classification(random_state=0, n_classes=10)
    # 构造一个预测分数 y_score，满足 -y_true + 1 的条件
    y_score = -y_true + 1
    # 调用 _test_ndcg_score_for 函数，对 y_true 和 y_score 进行 NDCG 分数的测试
    _test_ndcg_score_for(y_true, y_score)
    # 生成随机的 y_true 和 y_score，调用 _test_ndcg_score_for 函数进行 NDCG 分数的测试
    y_true, y_score = np.random.RandomState(0).random_sample((2, 100, 10))
    _test_ndcg_score_for(y_true, y_score)


# 定义一个辅助函数，用于测试给定 y_true 和 y_score 的 NDCG 分数
def _test_ndcg_score_for(y_true, y_score):
    # 计算理想情况下的 NDCG 分数
    ideal = _ndcg_sample_scores(y_true, y_true)
    # 计算给定 y_true 和 y_score 的 NDCG 分数
    score = _ndcg_sample_scores(y_true, y_score)
    # 断言计算得到的分数不超过理想情况下的分数
    assert (score <= ideal).all()
    # 检查所有样本的真实标签是否全为零
    all_zero = (y_true == 0).all(axis=1)
    # 断言在非全零标签的情况下，理想分数等于全为1的数组（使用 pytest.approx 近似判断）
    assert ideal[~all_zero] == pytest.approx(np.ones((~all_zero).sum()))
    # 断言在全零标签的情况下，理想分数等于全为零的数组
    assert ideal[all_zero] == pytest.approx(np.zeros(all_zero.sum()))
    # 断言计算得到的分数在非全零标签的情况下，等于相应的 DCG 分数除以理想情况下的 DCG 分数
    assert score[~all_zero] == pytest.approx(
        _dcg_sample_scores(y_true, y_score)[~all_zero]
        / _dcg_sample_scores(y_true, y_true)[~all_zero]
    )
    # 断言在全零标签的情况下，计算得到的分数为全为零的数组
    assert score[all_zero] == pytest.approx(np.zeros(all_zero.sum()))
    # 断言理想分数的形状为 (样本数,)
    assert ideal.shape == (y_true.shape[0],)
    # 断言计算得到的分数的形状为 (样本数,)
    assert score.shape == (y_true.shape[0],)


# 定义一个测试函数，用于测试 partial_roc_auc_score 函数的功能
def test_partial_roc_auc_score():
    # 检查当 max_fpr != None 时的 roc_auc_score
    y_true = np.array([0, 0, 1, 1])
    assert roc_auc_score(y_true, y_true, max_fpr=1) == 1
    assert roc_auc_score(y_true, y_true, max_fpr=0.001) == 1
    # 检查当 max_fpr 超出范围时是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=-0.1)
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=1.1)
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=0)

    # 创建一个预测分数数组 y_scores
    y_scores = np.array([0.1, 0, 0.1, 0.01])
    # 计算 max_fpr=1 时的 roc_auc_score 和无约束条件下的 roc_auc_score，断言它们相等
    roc_auc_with_max_fpr_one = roc_auc_score(y_true, y_scores, max_fpr=1)
    unconstrained_roc_auc = roc_auc_score(y_true, y_scores)
    assert roc_auc_with_max_fpr_one == unconstrained_roc_auc
    # 断言当 max_fpr=0.3 时的 roc_auc_score 为 0.5
    assert roc_auc_score(y_true, y_scores, max_fpr=0.3) == 0.5

    # 生成一个二分类预测的 y_true 和 y_pred，并针对不同的 max_fpr 参数进行部分 roc_auc_score 的测试
    y_true, y_pred, _ = make_prediction(binary=True)
    for max_fpr in np.linspace(1e-4, 1, 5):
        # 使用 assert_almost_equal 检查计算得到的部分 roc_auc_score 与预期值的近似性
        assert_almost_equal(
            roc_auc_score(y_true, y_pred, max_fpr=max_fpr),
            _partial_roc_auc_score(y_true, y_pred, max_fpr),
        )


# 使用 pytest.mark.parametrize 进行参数化测试，测试 top_k_accuracy_score 函数的功能
@pytest.mark.parametrize(
    "y_true, k, true_score",
    [
        ([0, 1, 2, 3], 1, 0.25),
        ([0, 1, 2, 3], 2, 0.5),
        ([0, 1, 2, 3], 3, 0.75),
    ],
)
def test_top_k_accuracy_score(y_true, k, true_score):
    # 创建一个预测分数矩阵 y_score
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.4, 0.1, 0.2, 0.3],
            [0.3, 0.2, 0.4, 0.1],
        ]
    )
    # 调用 top_k_accuracy_score 函数计算得分，并断言与预期得分相符
    score = top_k_accuracy_score(y_true, y_score, k=k)
    assert score == pytest.approx(true_score)


# 使用 pytest.mark.parametrize 进行参数化测试，测试二分类情况下的 top_k_accuracy_score 函数的功能
@pytest.mark.parametrize(
    "y_score, k, true_score",
    [
        (np.array([-1, -1, 1, 1]), 1, 1),
        (np.array([-1, 1, -1, 1]), 1, 0.5),
        (np.array([-1, 1, -1, 1]), 2, 1),
        (np.array([0.2, 0.2, 0.7, 0.7]), 1, 1),
        (np.array([0.2, 0.7, 0.2, 0.7]), 1, 0.5),
        (np.array([0.2, 0.7, 0.2, 0.7]), 2, 1),
    ],
)
def test_top_k_accuracy_score_binary(y_score, k, true_score):
    # 创建二分类的真实标签 y_true
    y_true = [0, 0, 1, 1]
    # 如果预测分数的最小值大于等于 0，且最大值小于等于 1，则设定阈值为 0.5；否则设定阈值为 0
    threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0

    # 如果 k 等于 1，则使用阈值判断 y_score 是否大于阈值，并转换为 np.int64 类型，否则使用 y_true 作为预测值
    y_pred = (y_score > threshold).astype(np.int64) if k == 1 else y_true

    # 计算 top-k 准确率，返回的分数保存在 score 中
    score = top_k_accuracy_score(y_true, y_score, k=k)
    
    # 计算预测值 y_pred 和真实值 y_true 的准确率，并保存在 score_acc 中
    score_acc = accuracy_score(y_true, y_pred)

    # 使用 pytest 的 approx 函数断言 top-k 准确率 score、普通准确率 score_acc 与真实的 true_score 相等
    assert score == score_acc == pytest.approx(true_score)
@pytest.mark.parametrize(
    "y_true, true_score, labels",
    [
        (np.array([0, 1, 1, 2]), 0.75, [0, 1, 2, 3]),  # 参数化测试用例1：定义y_true、true_score和labels
        (np.array([0, 1, 1, 1]), 0.5, [0, 1, 2, 3]),   # 参数化测试用例2：定义y_true、true_score和labels
        (np.array([1, 1, 1, 1]), 0.5, [0, 1, 2, 3]),   # 参数化测试用例3：定义y_true、true_score和labels
        (np.array(["a", "e", "e", "a"]), 0.75, ["a", "b", "d", "e"]),   # 参数化测试用例4：定义y_true、true_score和labels
    ],
)
@pytest.mark.parametrize("labels_as_ndarray", [True, False])
def test_top_k_accuracy_score_multiclass_with_labels(
    y_true, true_score, labels, labels_as_ndarray
):
    """Test when labels and y_score are multiclass."""
    if labels_as_ndarray:
        labels = np.asarray(labels)  # 如果labels_as_ndarray为True，则将labels转换为NumPy数组
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],   # 测试用的预测分数矩阵的一部分
            [0.1, 0.3, 0.4, 0.2],   # 测试用的预测分数矩阵的一部分
            [0.4, 0.1, 0.2, 0.3],   # 测试用的预测分数矩阵的一部分
            [0.3, 0.2, 0.4, 0.1],   # 测试用的预测分数矩阵的一部分
        ]
    )

    score = top_k_accuracy_score(y_true, y_score, k=2, labels=labels)  # 调用函数计算top-k准确度分数
    assert score == pytest.approx(true_score)  # 断言计算出的分数与预期分数近似相等


def test_top_k_accuracy_score_increasing():
    # Make sure increasing k leads to a higher score
    X, y = datasets.make_classification(
        n_classes=10, n_samples=1000, n_informative=10, random_state=0
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    for X, y in zip((X_train, X_test), (y_train, y_test)):
        scores = [
            top_k_accuracy_score(y, clf.predict_proba(X), k=k) for k in range(2, 10)
        ]  # 计算不同k值下的top-k准确度分数

        assert np.all(np.diff(scores) > 0)  # 断言分数随k的增加而增加


@pytest.mark.parametrize(
    "y_true, k, true_score",
    [
        ([0, 1, 2, 3], 1, 0.25),   # 参数化测试用例：定义y_true、k和true_score
        ([0, 1, 2, 3], 2, 0.5),    # 参数化测试用例：定义y_true、k和true_score
        ([0, 1, 2, 3], 3, 1),      # 参数化测试用例：定义y_true、k和true_score
    ],
)
def test_top_k_accuracy_score_ties(y_true, k, true_score):
    # Make sure highest indices labels are chosen first in case of ties
    y_score = np.array(
        [
            [5, 5, 7, 0],   # 测试用的预测分数矩阵的一部分
            [1, 5, 5, 5],   # 测试用的预测分数矩阵的一部分
            [0, 0, 3, 3],   # 测试用的预测分数矩阵的一部分
            [1, 1, 1, 1],   # 测试用的预测分数矩阵的一部分
        ]
    )
    assert top_k_accuracy_score(y_true, y_score, k=k) == pytest.approx(true_score)  # 断言计算出的分数与预期分数近似相等


@pytest.mark.parametrize(
    "y_true, k",
    [
        ([0, 1, 2, 3], 4),   # 参数化测试用例：定义y_true和k
        ([0, 1, 2, 3], 5),   # 参数化测试用例：定义y_true和k
    ],
)
def test_top_k_accuracy_score_warning(y_true, k):
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],   # 测试用的预测分数矩阵的一部分
            [0.1, 0.4, 0.3, 0.2],   # 测试用的预测分数矩阵的一部分
            [0.2, 0.1, 0.4, 0.3],   # 测试用的预测分数矩阵的一部分
            [0.3, 0.2, 0.1, 0.4],   # 测试用的预测分数矩阵的一部分
        ]
    )
    expected_message = (
        r"'k' \(\d+\) greater than or equal to 'n_classes' \(\d+\) will result in a "
        "perfect score and is therefore meaningless."
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        score = top_k_accuracy_score(y_true, y_score, k=k)  # 调用函数计算top-k准确度分数
    assert score == 1   # 断言返回的分数为1
    [
        (
            [0, 0.57, 1, 2],  # y_true: 样本标签数组，可以是浮点数的列表
            [
                [0.2, 0.1, 0.7],  # y_score: 预测分数的二维数组，每个子数组代表一个类的概率分布
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            None,  # labels: 标签数组，可以是字符串列表或None
            "y type must be 'binary' or 'multiclass', got 'continuous'",  # 错误信息，指示 y 的类型应为二元或多类别，而不是连续的
        ),
        (
            [0, 1, 2, 3],  # y_true: 样本标签数组，表示四个类别
            [
                [0.2, 0.1, 0.7],  # y_score: 预测分数的二维数组，每个子数组代表一个类的概率分布
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            None,  # labels: 标签数组，可以是字符串列表或None
            r"Number of classes in 'y_true' \(4\) not equal to the number of "
            r"classes in 'y_score' \(3\).",  # 错误信息，指示 y_true 的类别数（4）与 y_score 的类别数（3）不相等
        ),
        (
            ["c", "c", "a", "b"],  # y_true: 样本标签数组，包含字符串标签
            [
                [0.2, 0.1, 0.7],  # y_score: 预测分数的二维数组，每个子数组代表一个类的概率分布
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            ["a", "b", "c", "c"],  # labels: 标签数组，包含所有可能的类别，应为唯一值
            "Parameter 'labels' must be unique.",  # 错误信息，指示标签数组必须是唯一的
        ),
        (
            ["c", "c", "a", "b"],  # y_true: 样本标签数组，包含字符串标签
            [
                [0.2, 0.1, 0.7],  # y_score: 预测分数的二维数组，每个子数组代表一个类的概率分布
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            ["a", "c", "b"],  # labels: 标签数组，包含所有可能的类别，应按顺序排列
            "Parameter 'labels' must be ordered.",  # 错误信息，指示标签数组必须按顺序排列
        ),
        (
            [0, 0, 1, 2],  # y_true: 样本标签数组，表示四个类别
            [
                [0.2, 0.1, 0.7],  # y_score: 预测分数的二维数组，每个子数组代表一个类的概率分布
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            [0, 1, 2, 3],  # labels: 标签数组，包含所有可能的类别
            r"Number of given labels \(4\) not equal to the number of classes in "
            r"'y_score' \(3\).",  # 错误信息，指示给定标签数（4）与 y_score 的类别数（3）不相等
        ),
        (
            [0, 0, 1, 2],  # y_true: 样本标签数组，表示四个类别
            [
                [0.2, 0.1, 0.7],  # y_score: 预测分数的二维数组，每个子数组代表一个类的概率分布
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.5, 0.1],
            ],
            [0, 1, 3],  # labels: 标签数组，包含所有可能的类别，但不包含某些 y_true 中的标签
            "'y_true' contains labels not in parameter 'labels'.",  # 错误信息，指示 y_true 包含未在标签数组中提供的标签
        ),
        (
            [0, 1],  # y_true: 二元样本标签数组
            [[0.5, 0.2, 0.2], [0.3, 0.4, 0.2]],  # y_score: 预测分数的二维数组，每个子数组代表一个类的概率分布
            None,  # labels: 标签数组，可以是字符串列表或None
            (
                "`y_true` is binary while y_score is 2d with 3 classes. If"
                " `y_true` does not contain all the labels, `labels` must be provided"
            ),  # 错误信息，指示 y_true 是二元的，但 y_score 是二维数组且包含 3 个类别，如果 y_true 不包含所有标签，则必须提供 labels 参数
        ),
    ],
# 测试函数，用于验证 top_k_accuracy_score 函数在特定条件下抛出 ValueError 异常
def test_top_k_accuracy_score_error(y_true, y_score, labels, msg):
    with pytest.raises(ValueError, match=msg):
        top_k_accuracy_score(y_true, y_score, k=2, labels=labels)


# 使用参数化测试装饰器，对 label_ranking_avg_precision_score 函数进行测试
# 验证该函数能够接受稀疏矩阵 y_true 输入
# 非回归测试，针对 issue #22575
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_label_ranking_avg_precision_score_should_allow_csr_matrix_for_y_true_input(
    csr_container,
):
    # 测试 label_ranking_avg_precision_score 函数接受稀疏矩阵 y_true
    y_true = csr_container([[1, 0, 0], [0, 0, 1]])
    y_score = np.array([[0.5, 0.9, 0.6], [0, 0, 1]])
    result = label_ranking_average_precision_score(y_true, y_score)
    assert result == pytest.approx(2 / 3)


# 使用参数化测试装饰器，测试多个评估指标函数能够处理不同类型的 pos_label
@pytest.mark.parametrize(
    "metric", [average_precision_score, det_curve, precision_recall_curve, roc_curve]
)
@pytest.mark.parametrize(
    "classes", [(False, True), (0, 1), (0.0, 1.0), ("zero", "one")]
)
def test_ranking_metric_pos_label_types(metric, classes):
    """验证评估指标函数能够处理不同类型的 `pos_label`。

    我们可以期望 `pos_label` 是布尔值、整数、浮点数或字符串，对于这些类型不应该抛出错误。
    """
    rng = np.random.RandomState(42)
    n_samples, pos_label = 10, classes[-1]
    y_true = rng.choice(classes, size=n_samples, replace=True)
    y_proba = rng.rand(n_samples)
    result = metric(y_true, y_proba, pos_label=pos_label)
    if isinstance(result, float):
        assert not np.isnan(result)
    else:
        metric_1, metric_2, thresholds = result
        assert not np.isnan(metric_1).any()
        assert not np.isnan(metric_2).any()
        assert not np.isnan(thresholds).any()


# 测试函数，验证当 `y_score` 是概率估计时，`roc_curve` 函数不会生成大于 1.0 的阈值
def test_roc_curve_with_probablity_estimates(global_random_seed):
    """验证当 `y_score` 是概率估计时，`roc_curve` 函数不会生成大于 1.0 的阈值。

    非回归测试，针对:
    https://github.com/scikit-learn/scikit-learn/issues/26193
    """
    rng = np.random.RandomState(global_random_seed)
    y_true = rng.randint(0, 2, size=10)
    y_score = rng.rand(10)
    _, _, thresholds = roc_curve(y_true, y_score)
    assert np.isinf(thresholds[0])


# TODO(1.7): 移除
# 测试函数，检查关于未来弃用的警告消息
def test_precision_recall_curve_deprecation_warning():
    """检查未来弃用的警告消息内容。"""
    # 使用 precision_recall_curve 函数进行检查
    y_true, _, y_score = make_prediction(binary=True)

    warn_msg = "probas_pred was deprecated in version 1.5"
    with pytest.warns(FutureWarning, match=warn_msg):
        precision_recall_curve(
            y_true,
            probas_pred=y_score,
        )

    error_msg = "`probas_pred` and `y_score` cannot be both specified"
    with pytest.raises(ValueError, match=error_msg):
        precision_recall_curve(
            y_true,
            probas_pred=y_score,
            y_score=y_score,
        )
```