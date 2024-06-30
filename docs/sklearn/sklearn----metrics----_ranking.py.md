# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_ranking.py`

```
"""Metrics to assess performance on classification task given scores.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import warnings  # 导入警告模块，用于处理警告信息
from functools import partial  # 导入 partial 函数，用于创建偏函数
from numbers import Integral, Real  # 导入整数和实数类型判断的模块

import numpy as np  # 导入 NumPy 库
from scipy.sparse import csr_matrix, issparse  # 导入稀疏矩阵相关函数
from scipy.stats import rankdata  # 导入排名数据相关函数

from ..exceptions import UndefinedMetricWarning  # 导入自定义异常模块，处理未定义指标警告
from ..preprocessing import label_binarize  # 导入标签二值化函数
from ..utils import (  # 导入各种工具函数
    assert_all_finite,  # 检查数组是否包含有限数字
    check_array,  # 检查和转换数组
    check_consistent_length,  # 检查数组长度是否一致
    column_or_1d,  # 将输入转换为一维数组
)
from ..utils._encode import _encode, _unique  # 导入编码和唯一值处理函数
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params  # 导入参数验证相关函数和类
from ..utils.extmath import stable_cumsum  # 导入稳定累积和函数
from ..utils.fixes import trapezoid  # 导入梯形积分函数
from ..utils.multiclass import type_of_target  # 导入多类别分类相关函数
from ..utils.sparsefuncs import count_nonzero  # 导入稀疏矩阵非零元素计数函数
from ..utils.validation import (  # 导入验证函数
    _check_pos_label_consistency,  # 检查正类标签一致性
    _check_sample_weight,  # 检查样本权重
)
from ._base import (  # 导入基础模块的函数
    _average_binary_score,  # 计算二分类平均分数
    _average_multiclass_ovo_score,  # 计算多类别一对一平均分数
)


@validate_params(
    {"x": ["array-like"], "y": ["array-like"]},  # 参数验证装饰器，验证 x 和 y 是否为数组样式
    prefer_skip_nested_validation=True,
)
def auc(x, y):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.

    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.

    Parameters
    ----------
    x : array-like of shape (n,)
        X coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : array-like of shape (n,)
        Y coordinates.

    Returns
    -------
    auc : float
        Area Under the Curve.

    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    average_precision_score : Compute average precision from prediction scores.
    precision_recall_curve : Compute precision-recall pairs for different
        probability thresholds.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75
    """
    check_consistent_length(x, y)  # 检查 x 和 y 的长度是否一致
    x = column_or_1d(x)  # 将 x 转换为一维数组
    y = column_or_1d(y)  # 将 y 转换为一维数组

    if x.shape[0] < 2:  # 如果 x 的长度小于2，则无法计算曲线下面积，抛出异常
        raise ValueError(
            "At least 2 points are needed to compute area under curve, but x.shape = %s"
            % x.shape
        )

    direction = 1  # 初始化方向为正
    dx = np.diff(x)  # 计算 x 的差分
    if np.any(dx < 0):  # 如果有任何一个 dx 小于0
        if np.all(dx <= 0):  # 如果所有的 dx 都小于等于0
            direction = -1  # 方向设置为负
        else:
            raise ValueError("x is neither increasing nor decreasing : {}.".format(x))  # 抛出异常，x 既不是递增也不是递减

    area = direction * trapezoid(y, x)  # 计算曲线下面积，使用梯形积分法
    # 检查变量 area 是否是 np.memmap 类型的实例
    if isinstance(area, np.memmap):
        # 对于 numpy.memmap 实例，内部使用的归约操作（如 .sum）不像
        # 普通的 numpy.ndarray 实例那样默认返回标量。
        # 因此，需要使用 area.dtype.type(area) 来确保返回一个标量值。
        area = area.dtype.type(area)
    # 返回处理后的 area 变量，可能是归约操作后的标量值或原始输入
    return area
# 使用装饰器 validate_params 对 average_precision_score 函数进行参数验证
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应为类似数组的数据类型
        "y_score": ["array-like"],  # y_score 参数应为类似数组的数据类型
        "average": [StrOptions({"micro", "samples", "weighted", "macro"}), None],  # average 参数可以是 'micro', 'samples', 'weighted', 'macro' 中的一个，或者为 None
        "pos_label": [Real, str, "boolean"],  # pos_label 参数可以是实数、字符串或布尔值
        "sample_weight": ["array-like", None],  # sample_weight 参数应为类似数组的数据类型，或者为 None
    },
    prefer_skip_nested_validation=True,  # 更喜欢跳过嵌套验证
)
# 定义 average_precision_score 函数，计算预测分数的平均精度 (AP)
def average_precision_score(
    y_true, y_score, *, average="macro", pos_label=1, sample_weight=None
):
    """Compute average precision (AP) from prediction scores.

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    average : {'micro', 'samples', 'weighted', 'macro'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    pos_label : int, float, bool or str, default=1
        The label of the positive class. Only applied to binary ``y_true``.
        For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    average_precision : float
        Average precision score.

    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    def _binary_uninterpolated_average_precision(
        y_true, y_score, pos_label=1, sample_weight=None
    ):
        # 计算精确率-召回率曲线
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
        )
        # 返回阶梯函数的积分
        # 这里可以工作是因为精确率的最后一个条目保证是1，由 precision_recall_curve 返回
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
    
    # 确定 y_true 的类型
    y_type = type_of_target(y_true, input_name="y_true")
    
    # 将其转换为 Python 基本类型，避免 NumPy 类型与 Python 字符串的比较
    # 参考 https://github.com/numpy/numpy/issues/6784
    present_labels = np.unique(y_true).tolist()
    
    # 对二元分类情况进行检查
    if y_type == "binary":
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    
    # 对多标签指示器的情况进行检查
    elif y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError(
            "Parameter pos_label is fixed to 1 for multilabel-indicator y_true. "
            "Do not set pos_label or set pos_label to 1."
        )
    
    # 对多类别情况进行检查
    elif y_type == "multiclass":
        if pos_label != 1:
            raise ValueError(
                "Parameter pos_label is fixed to 1 for multiclass y_true. "
                "Do not set pos_label or set pos_label to 1."
            )
        # 对 y_true 进行标签二值化处理
        y_true = label_binarize(y_true, classes=present_labels)
    
    # 创建平均精度的部分函数
    average_precision = partial(
        _binary_uninterpolated_average_precision, pos_label=pos_label
    )
    # 返回平均二元分类分数
    return _average_binary_score(
        average_precision, y_true, y_score, average, sample_weight=sample_weight
    )
@validate_params(
    {
        "y_true": ["array-like"],  # 参数验证装饰器，验证 y_true 应为类数组
        "y_score": ["array-like"],  # 参数验证装饰器，验证 y_score 应为类数组
        "pos_label": [Real, str, "boolean", None],  # 参数验证装饰器，验证 pos_label 可接受 Real、str、boolean 或 None 类型
        "sample_weight": ["array-like", None],  # 参数验证装饰器，验证 sample_weight 应为类数组或 None
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器的首选选项，跳过嵌套验证
)
def det_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Compute error rates for different probability thresholds.

    .. note::
       This metric is used for evaluation of ranking and error tradeoffs of
       a binary classification task.

    Read more in the :ref:`User Guide <det_curve>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : ndarray of shape of (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int, float, bool or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fpr : ndarray of shape (n_thresholds,)
        False positive rate (FPR) such that element i is the false positive
        rate of predictions with score >= thresholds[i]. This is occasionally
        referred to as false acceptance probability or fall-out.

    fnr : ndarray of shape (n_thresholds,)
        False negative rate (FNR) such that element i is the false negative
        rate of predictions with score >= thresholds[i]. This is occasionally
        referred to as false rejection or miss rate.

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.

    See Also
    --------
    DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
        some data.
    DetCurveDisplay.from_predictions : Plot DET curve given the true and
        predicted labels.
    DetCurveDisplay : DET curve visualization.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    precision_recall_curve : Compute precision-recall curve.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import det_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, fnr, thresholds = det_curve(y_true, y_scores)
    >>> fpr
    array([0.5, 0.5, 0. ])
    >>> fnr
    array([0. , 0.5, 0.5])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])
    """
    # 调用内部函数 _binary_clf_curve 计算二元分类器曲线的指标
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )
    # 如果 y_true 中的唯一值数量不为 2，则抛出 ValueError 异常
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Only one class present in y_true. Detection error "
            "tradeoff curve is not defined in that case."
        )

    # 计算 false negatives 数组
    fns = tps[-1] - tps
    # 计算正例数量
    p_count = tps[-1]
    # 计算负例数量
    n_count = fps[-1]

    # 确定切片的起始位置，使得 false positives 从 0 开始
    first_ind = (
        fps.searchsorted(fps[0], side="right") - 1
        if fps.searchsorted(fps[0], side="right") > 0
        else None
    )
    # 确定切片的结束位置，使得 true positives 达到最大值时结束
    last_ind = tps.searchsorted(tps[-1]) + 1
    sl = slice(first_ind, last_ind)

    # 将 false positives、false negatives 和阈值按照 false positives 递减的顺序进行输出
    return (fps[sl][::-1] / n_count, fns[sl][::-1] / p_count, thresholds[sl][::-1])
# 定义一个函数，用于计算二分类情况下的 ROC AUC 分数
def _binary_roc_auc_score(y_true, y_score, sample_weight=None, max_fpr=None):
    """Binary roc auc score."""
    # 检查 y_true 中是否只有一个类别，如果是，则抛出 ValueError 异常
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Only one class present in y_true. ROC AUC score "
            "is not defined in that case."
        )

    # 计算 ROC 曲线的假正率（fpr）、真正率（tpr）以及阈值（thresholds）
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    
    # 如果 max_fpr 为 None 或者等于 1，直接计算 AUC
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    
    # 如果 max_fpr 不在 (0, 1] 范围内，抛出 ValueError 异常
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected max_fpr in range (0, 1], got: %r" % max_fpr)

    # 在 fpr 曲线上进行线性插值，添加一个点到 max_fpr
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    
    # 计算部分 AUC
    partial_auc = auc(fpr, tpr)

    # McClish 校正：如果非区分性为 0.5，最大为 1
    min_area = 0.5 * max_fpr**2
    max_area = max_fpr
    
    # 根据 McClish 校正公式，标准化结果
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


# 使用装饰器 validate_params 对 roc_auc_score 函数的参数进行验证
@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
        "average": [StrOptions({"micro", "macro", "samples", "weighted"}), None],
        "sample_weight": ["array-like", None],
        "max_fpr": [Interval(Real, 0.0, 1, closed="right"), None],
        "multi_class": [StrOptions({"raise", "ovr", "ovo"})],
        "labels": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
# 定义 roc_auc_score 函数，计算 ROC AUC 曲线下面积
def roc_auc_score(
    y_true,
    y_score,
    *,
    average="macro",
    sample_weight=None,
    max_fpr=None,
    multi_class="raise",
    labels=None,
):
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) \
    from prediction scores.

    Note: this implementation can be used with binary, multiclass and
    multilabel classification, but some restrictions apply (see Parameters).

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases
        expect labels with shape (n_samples,) while the multilabel case expects
        binary label indicators with shape (n_samples, n_classes).
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

        * In the binary case, it corresponds to an array of shape
          `(n_samples,)`. Both probability estimates and non-thresholded
          decision values can be provided. The probability estimates correspond
          to the **probability of the class with the greater label**,
          i.e. `estimator.classes_[1]` and thus
          `estimator.predict_proba(X, y)[:, 1]`. The decision values
          corresponds to the output of `estimator.decision_function(X, y)`.
          See more information in the :ref:`User guide <roc_auc_binary>`;
        * In the multiclass case, it corresponds to an array of shape
          `(n_samples, n_classes)` of probability estimates provided by the
          `predict_proba` method. The probability estimates **must**
          sum to 1 across the possible classes. In addition, the order of the
          class scores must correspond to the order of ``labels``,
          if provided, or else to the numerical or lexicographical order of
          the labels in ``y_true``. See more information in the
          :ref:`User guide <roc_auc_multiclass>`;
        * In the multilabel case, it corresponds to an array of shape
          `(n_samples, n_classes)`. Probability estimates are provided by the
          `predict_proba` method and the non-thresholded decision values by
          the `decision_function` method. The probability estimates correspond
          to the **probability of the class with the greater label for each
          output** of the classifier. See more information in the
          :ref:`User guide <roc_auc_multilabel>`.

    average : {'micro', 'macro', 'samples', 'weighted'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data.
        Note: multiclass ROC AUC currently only handles the 'macro' and
        'weighted' averages. For multiclass targets, `average=None` is only
        implemented for `multi_class='ovr'` and `average='micro'` is only
        implemented for `multi_class='ovr'`.

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    max_fpr : float > 0 and <= 1, default=None
        如果不为 ``None``，则返回范围 [0, max_fpr] 内的标准化部分AUC [2]_。
        对于多类别情况，``max_fpr`` 应为 ``None`` 或 ``1.0``，因为目前不支持多类别的AUC ROC部分计算。

    multi_class : {'raise', 'ovr', 'ovo'}, default='raise'
        仅用于多类别目标。确定要使用的配置类型。默认值会引发错误，因此必须显式传递 ``'ovr'`` 或 ``'ovo'``。

        ``'ovr'``:
            即一对多（One-vs-rest）。计算每个类别与其余所有类别的AUC [3]_ [4]_。
            这种方式将多类别情况视为多标签情况。即使 ``average == 'macro'``，也会对类别不平衡敏感，
            因为类别不平衡会影响每个“其余”组的构成。
        ``'ovo'``:
            即一对一（One-vs-one）。计算所有可能的类别配对组合的平均AUC [5]_。
            当 ``average == 'macro'`` 时，对类别不平衡不敏感。

    labels : array-like of shape (n_classes,), default=None
        仅用于多类别目标。索引 ``y_score`` 中类别的标签列表。如果为 ``None``，则使用 ``y_true`` 中标签的数值或词典顺序。

    Returns
    -------
    auc : float
        曲线下面积（AUC）得分。

    See Also
    --------
    average_precision_score : 精度-召回曲线下面积。
    roc_curve : 计算接收者操作特征（ROC）曲线。
    RocCurveDisplay.from_estimator : 给定估计器和数据，绘制接收者操作特征（ROC）曲线。
    RocCurveDisplay.from_predictions : 给定真实值和预测值，绘制接收者操作特征（ROC）曲线。

    Notes
    -----
    Gini系数是二元分类器排名能力的总结性测量指标。它使用ROC曲线下面积表示如下：

    G = 2 * AUC - 1

    其中，G是Gini系数，AUC是ROC-AUC得分。此归一化确保随机猜测的期望得分为0，并且上限为1。

    References
    ----------
    .. [1] `接收者操作特征的维基百科条目
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] `分析ROC曲线的一部分。McClish, 1989
            <https://www.ncbi.nlm.nih.gov/pubmed/2668680>`_

    .. [3] Provost, F., Domingos, P. (2000). Well-trained PETs: Improving
           probability estimation trees (Section 6.2), CeDER Working Paper
           #IS-00-04, Stern School of Business, New York University.
    # 确定目标变量 y_true 的类型，可能是多分类、二分类或多标签分类
    y_type = type_of_target(y_true, input_name="y_true")
    
    # 将 y_true 转换为数组，确保是二维数组
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    
    # 将 y_score 转换为数组，确保是二维数组
    y_score = check_array(y_score, ensure_2d=False)
    
    # 如果 y_type 是多分类，或者是二分类且 y_score 是二维且列数大于2时执行以下代码块
    if y_type == "multiclass" or (
        y_type == "binary" and y_score.ndim == 2 and y_score.shape[1] > 2
    ):
        # 如果要计算部分 AUC 并且是多类别问题，则不支持
        if max_fpr is not None and max_fpr != 1.0:
            # 抛出值错误异常，说明部分 AUC 在多类别设置下不可用
            raise ValueError(
                "Partial AUC computation not available in "
                "multiclass setting, 'max_fpr' must be"
                " set to `None`, received `max_fpr={0}` "
                "instead".format(max_fpr)
            )
        if multi_class == "raise":
            # 如果 multi_class 设置为 'raise'，则抛出值错误异常
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        # 返回多类别 ROC AUC 分数
        return _multiclass_roc_auc_score(
            y_true, y_score, labels, multi_class, average, sample_weight
        )
    elif y_type == "binary":
        # 获取唯一的标签值
        labels = np.unique(y_true)
        # 将 y_true 转换为二进制矩阵，并仅保留第一列
        y_true = label_binarize(y_true, classes=labels)[:, 0]
        # 返回平均的二元分类 ROC AUC 分数
        return _average_binary_score(
            partial(_binary_roc_auc_score, max_fpr=max_fpr),
            y_true,
            y_score,
            average,
            sample_weight=sample_weight,
        )
    else:  # multilabel-indicator
        # 返回平均的二元分类 ROC AUC 分数，适用于多标签指示符情况
        return _average_binary_score(
            partial(_binary_roc_auc_score, max_fpr=max_fpr),
            y_true,
            y_score,
            average,
            sample_weight=sample_weight,
        )
# 定义一个计算多类别 ROC AUC 分数的函数
def _multiclass_roc_auc_score(
    y_true, y_score, labels, multi_class, average, sample_weight
):
    """Multiclass roc auc score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        真实的多类别标签。

    y_score : array-like of shape (n_samples, n_classes)
        目标分数，对应于样本属于特定类的概率估计。

    labels : array-like of shape (n_classes,) or None
        用于多类别索引 y_score 的标签列表。如果为 None，则使用 y_true 的词法顺序索引 y_score。

    multi_class : {'ovr', 'ovo'}
        确定要使用的多类别配置类型。
        ``'ovr'``:
            使用一对剩余法（one-vs-rest approach）计算多类别情况下的指标。
        ``'ovo'``:
            使用一对一法（one-vs-one approach）计算多类别情况下的指标。

    average : {'micro', 'macro', 'weighted'}
        确定在成对二进制度量分数上执行的平均类型。
        ``'micro'``:
            计算二进制化并展平类的度量。仅支持 `multi_class='ovr'`。

        .. versionadded:: 1.2

        ``'macro'``:
            为每个标签计算度量，并找到它们的非加权平均值。这不考虑标签不平衡。假设类是均匀分布的。
        ``'weighted'``:
            为每个标签计算度量，考虑类的普遍性。

    sample_weight : array-like of shape (n_samples,) or None
        样本权重。
    """
    
    # 验证输入 y_score 的有效性
    if not np.allclose(1, y_score.sum(axis=1)):
        raise ValueError(
            "Target scores need to be probabilities for multiclass "
            "roc_auc, i.e. they should sum up to 1.0 over classes"
        )

    # 验证 multiclass 参数的规范
    average_options = ("macro", "weighted", None)
    if multi_class == "ovr":
        average_options = ("micro",) + average_options
    if average not in average_options:
        raise ValueError(
            "average must be one of {0} for multiclass problems".format(average_options)
        )

    multiclass_options = ("ovo", "ovr")
    if multi_class not in multiclass_options:
        raise ValueError(
            "multi_class='{0}' is not supported "
            "for multiclass ROC AUC, multi_class must be "
            "in {1}".format(multi_class, multiclass_options)
        )

    # 如果 average 为 None 而 multi_class 为 'ovo'，则抛出未实现错误
    if average is None and multi_class == "ovo":
        raise NotImplementedError(
            "average=None is not implemented for multi_class='ovo'."
        )
    # 如果 labels 参数不为 None，则进行以下处理
    if labels is not None:
        # 将 labels 转换为一维数组，确保数据结构一致性
        labels = column_or_1d(labels)
        # 获取 labels 中的唯一值
        classes = _unique(labels)
        # 检查 labels 是否唯一
        if len(classes) != len(labels):
            raise ValueError("Parameter 'labels' must be unique")
        # 检查 labels 是否有序
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered")
        # 检查 labels 的数量与 y_score 的列数是否相等
        if len(classes) != y_score.shape[1]:
            raise ValueError(
                "Number of given labels, {0}, not equal to the number "
                "of columns in 'y_score', {1}".format(len(classes), y_score.shape[1])
            )
        # 检查 y_true 是否包含在 labels 中的标签
        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'")
    else:
        # 如果 labels 参数为 None，则根据 y_true 获取类别
        classes = _unique(y_true)
        # 检查 y_true 的类别数量与 y_score 的列数是否相等
        if len(classes) != y_score.shape[1]:
            raise ValueError(
                "Number of classes in y_true not equal to the number of "
                "columns in 'y_score'"
            )

    # 如果 multi_class 参数为 "ovo"，执行以下处理
    if multi_class == "ovo":
        # 如果 sample_weight 不为 None，则抛出异常
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is not supported "
                "for multiclass one-vs-one ROC AUC, "
                "'sample_weight' must be None in this case."
            )
        # 将 y_true 编码为整数
        y_true_encoded = _encode(y_true, uniques=classes)
        # 返回多类别一对一 ROC AUC 的平均分数
        # 使用 Hand & Till (2001) 的实现方式 (ovo)
        return _average_multiclass_ovo_score(
            _binary_roc_auc_score, y_true_encoded, y_score, average=average
        )
    else:
        # 如果 multi_class 参数不为 "ovo"，即为 "ovr" 或 "multilabel"
        # 将 y_true 转换为多标签形式
        y_true_multilabel = label_binarize(y_true, classes=classes)
        # 返回二元分类 ROC AUC 的平均分数
        return _average_binary_score(
            _binary_roc_auc_score,
            y_true_multilabel,
            y_score,
            average,
            sample_weight=sample_weight,
        )
    # 检查 y_true 的类型，确保是有效的目标类型
    y_type = type_of_target(y_true, input_name="y_true")
    if not (y_type == "
    # 计算真实正例（y_true * weight）的加权累积和，并根据 threshold_idxs 进行切片
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    
    if sample_weight is not None:
        # 如果存在样本权重，则计算伪正例（(1 - y_true) * weight）的加权累积和，
        # 并确保即使在浮点数误差存在的情况下，fps（假正例）也能保持增加
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        # 如果不存在样本权重，则计算固定的假正例（1 + threshold_idxs - tps）
        fps = 1 + threshold_idxs - tps
    
    # 返回计算出的假正例（fps）、真实正例（tps）、以及在 threshold_idxs 索引处的 y_score
    return fps, tps, y_score[threshold_idxs]
# 使用装饰器 @validate_params 对 precision_recall_curve 函数进行参数验证
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应该是 array-like 类型
        "y_score": ["array-like", Hidden(None)],  # y_score 参数可以是 array-like 类型，但 Hidden(None) 表示可以省略
        "pos_label": [Real, str, "boolean", None],  # pos_label 参数可以是实数、字符串、布尔值或者 None
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是 array-like 类型或者 None
        "drop_intermediate": ["boolean"],  # drop_intermediate 参数应该是布尔值
        "probas_pred": [  # probas_pred 参数可以是 array-like 类型，但 Hidden(StrOptions({"deprecated"})) 表示可以省略，或者是 "deprecated" 字符串
            "array-like",
            Hidden(StrOptions({"deprecated"})),
        ],
    },
    prefer_skip_nested_validation=True,  # 设置 prefer_skip_nested_validation 参数为 True，表示偏好跳过嵌套验证
)
def precision_recall_curve(
    y_true,
    y_score=None,
    *,
    pos_label=None,  # pos_label 参数是一个关键字参数，默认值为 None
    sample_weight=None,  # sample_weight 参数是一个关键字参数，默认值为 None
    drop_intermediate=False,  # drop_intermediate 参数是一个关键字参数，默认值为 False
    probas_pred="deprecated",  # probas_pred 参数是一个关键字参数，默认值为 "deprecated"
):
    """Compute precision-recall pairs for different probability thresholds.

    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold. This ensures that the graph starts on the
    y axis.

    The first precision and recall values are precision=class balance and recall=1.0
    which corresponds to a classifier that always predicts the positive class.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, or non-thresholded measure of decisions (as returned by
        `decision_function` on some classifiers).

    pos_label : int, float, bool or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    drop_intermediate : bool, default=False
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted precision-recall curve. This is useful in order to create
        lighter precision-recall curves.

        .. versionadded:: 1.3
    # `probas_pred`是一个数组，形状为(n_samples,)，存储了目标分数，可以是正类的概率估计，或者一些分类器返回的未经阈值处理的决策度量。
    # 在1.7版本中，`probas_pred`已经被弃用并将移除，建议使用`y_score`代替。
    # 返回三个数组：precision、recall和thresholds。
    # precision的形状为(n_thresholds + 1,)，其中第i个元素是在得分>=thresholds[i]时的精确度，最后一个元素为1。
    # recall的形状为(n_thresholds + 1,)，递减的召回率值，其中第i个元素是在得分>=thresholds[i]时的召回率，最后一个元素为0。
    # thresholds的形状为(n_thresholds,)，是用于计算精确度和召回率的决策函数的递增阈值。
    # 其中n_thresholds = len(np.unique(probas_pred))，表示probas_pred中唯一值的数量。
    # 如果`probas_pred`未指定，会发出FutureWarning，提醒使用`y_score`替代`probas_pred`。
    # 其中的示例演示了如何使用precision_recall_curve函数计算precision、recall和thresholds。
    # `probas_pred`在1.5版本中已弃用，并将在1.7版本中移除。
    # 如果y_score不是None且probas_pred不是字符串，则会引发ValueError异常。
    if y_score is not None and not isinstance(probas_pred, str):
        raise ValueError(
            "`probas_pred` and `y_score` cannot be both specified. Please use `y_score`"
            " only as `probas_pred` is deprecated in v1.5 and will be removed in v1.7."
        )
    # 如果y_score为None，则会发出警告，提醒在1.7版本中移除`probas_pred`，建议使用`y_score`。
    if y_score is None:
        warnings.warn(
            (
                "probas_pred was deprecated in version 1.5 and will be removed in 1.7."
                "Please use ``y_score`` instead."
            ),
            FutureWarning,
        )
        # 将y_score设置为probas_pred，以保持向后兼容性。
        y_score = probas_pred

    # 调用内部函数_binary_clf_curve计算二元分类器的假阳性率(fps)、真阳性率(tps)和阈值(thresholds)。
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )
    # 如果 drop_intermediate 为真，并且 fps 数组长度大于2：
    if drop_intermediate and len(fps) > 2:
        # 找出真正例 (tps) 不发生变化的阈值点，保留每个 tps 值的第一个和最后一个点，
        # 这些具有相同 tps 值的点具有相同的 recall 和 x 坐标，因此在图中呈现为垂直线。
        optimal_idxs = np.where(
            np.concatenate(
                [[True], np.logical_or(np.diff(tps[:-1]), np.diff(tps[1:])), [True]]
            )
        )[0]
        # 根据 optimal_idxs 筛选出符合条件的 fps、tps 和 thresholds 数组
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # 计算每个阈值下的正例数 ps
    ps = tps + fps
    # 初始化精确度数组为零，以确保 precision[ps == 0] 不包含未初始化的值
    precision = np.zeros_like(tps)
    # 计算精确度，避免除以零的情况
    np.divide(tps, ps, out=precision, where=(ps != 0))

    # 当 y_true 中没有正例标签时，设置 recall 为1
    # tps[-1] == 0 <=> y_true 中全部是负例标签
    if tps[-1] == 0:
        # 发出警告，提示 y_true 中未找到正类，recall 被设置为所有阈值下的1
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        # 将 recall 设置为所有阈值下的1
        recall = np.ones_like(tps)
    else:
        # 计算每个阈值下的 recall
        recall = tps / tps[-1]

    # 将输出反转以确保 recall 是递减的
    sl = slice(None, None, -1)
    # 返回调整后的 precision、recall 和 thresholds
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]
@validate_params(
    {
        "y_true": ["array-like"],  # 参数验证装饰器，验证参数 y_true 应为数组形式
        "y_score": ["array-like"],  # 参数验证装饰器，验证参数 y_score 应为数组形式
        "pos_label": [Real, str, "boolean", None],  # 参数验证装饰器，验证参数 pos_label 可以是实数、字符串、布尔值或者 None
        "sample_weight": ["array-like", None],  # 参数验证装饰器，验证参数 sample_weight 应为数组形式或者 None
        "drop_intermediate": ["boolean"],  # 参数验证装饰器，验证参数 drop_intermediate 应为布尔值
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器的选项，优先跳过嵌套验证
)
def roc_curve(
    y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True
):
    """Compute Receiver operating characteristic (ROC).

    Note: this implementation is restricted to the binary classification task.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int, float, bool or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    drop_intermediate : bool, default=True
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

        .. versionadded:: 0.17
           parameter *drop_intermediate*.

    Returns
    -------
    fpr : ndarray of shape (>2,)
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= `thresholds[i]`.

    tpr : ndarray of shape (>2,)
        Increasing true positive rates such that element `i` is the true
        positive rate of predictions with score >= `thresholds[i]`.

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `np.inf`.

    See Also
    --------
    RocCurveDisplay.from_estimator : Plot Receiver Operating Characteristic
        (ROC) curve given an estimator and some data.
    RocCurveDisplay.from_predictions : Plot Receiver Operating Characteristic
        (ROC) curve given the true and predicted values.
    det_curve: Compute error rates for different probability thresholds.
    roc_auc_score : Compute the area under the ROC curve.

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.
    """
    An arbitrary threshold is added for the case `tpr=0` and `fpr=0` to
    ensure that the curve starts at `(0, 0)`. This threshold corresponds to the
    `np.inf`.

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
           Letters, 2006, 27(8):861-874.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> thresholds
    array([ inf, 0.8 , 0.4 , 0.35, 0.1 ])
    """
    # Compute True Positive Rate (TPR), False Positive Rate (FPR), and thresholds
    # using the provided true labels (`y_true`) and predicted scores (`y_score`).
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    # Attempt to drop thresholds that do not significantly impact the ROC curve
    # to reduce unnecessary complexity. This step uses second derivatives to
    # identify corners in the ROC curve that do not affect the Area Under the Curve (AUC).
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # Ensure that the ROC curve starts at (0, 0) by adding an extra threshold position
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, thresholds]  # Add infinity as a threshold to start at (0, 0)

    # Calculate False Positive Rate (FPR) and handle cases where it may be undefined
    if fps[-1] <= 0:
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless",
            UndefinedMetricWarning,
        )
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    # Calculate True Positive Rate (TPR) and handle cases where it may be undefined
    if tps[-1] <= 0:
        warnings.warn(
            "No positive samples in y_true, true positive value should be meaningless",
            UndefinedMetricWarning,
        )
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    # Return False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    return fpr, tpr, thresholds
# 使用装饰器 validate_params 对 label_ranking_average_precision_score 函数的参数进行验证，确保满足指定的类型要求
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数可以是数组或稀疏矩阵
        "y_score": ["array-like"],  # y_score 参数是数组
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是数组或者 None
    },
    prefer_skip_nested_validation=True,  # 偏好跳过嵌套验证
)
def label_ranking_average_precision_score(y_true, y_score, *, sample_weight=None):
    """Compute ranking-based average precision.

    Label ranking average precision (LRAP) is the average over each ground
    truth label assigned to each sample, of the ratio of true vs. total
    labels with lower score.

    This metric is used in multilabel ranking problem, where the goal
    is to give better rank to the labels associated to each sample.

    The obtained score is always strictly greater than 0 and
    the best value is 1.

    Read more in the :ref:`User Guide <label_ranking_average_precision>`.

    Parameters
    ----------
    y_true : {array-like, sparse matrix} of shape (n_samples, n_labels)
        True binary labels in binary indicator format.

    y_score : array-like of shape (n_samples, n_labels)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

        .. versionadded:: 0.20

    Returns
    -------
    score : float
        Ranking-based average precision score.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import label_ranking_average_precision_score
    >>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
    >>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    >>> label_ranking_average_precision_score(y_true, y_score)
    0.416...
    """
    # 检查 y_true, y_score 和 sample_weight 的长度是否一致
    check_consistent_length(y_true, y_score, sample_weight)
    # 将 y_true 转换为 2 维数组格式，确保不是稀疏矩阵
    y_true = check_array(y_true, ensure_2d=False, accept_sparse="csr")
    # 将 y_score 转换为 2 维数组格式
    y_score = check_array(y_score, ensure_2d=False)

    # 如果 y_true 和 y_score 的形状不一致，抛出异常
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    # 确定 y_true 的类型，必须是 multilabel-indicator 类型或者是二分类的 2 维数组
    y_type = type_of_target(y_true, input_name="y_true")
    if y_type != "multilabel-indicator" and not (
        y_type == "binary" and y_true.ndim == 2
    ):
        raise ValueError("{0} format is not supported".format(y_type))

    # 如果 y_true 不是稀疏矩阵，则转换为 csr_matrix 格式
    if not issparse(y_true):
        y_true = csr_matrix(y_true)

    # 对 y_score 取负数，用于后续排序
    y_score = -y_score

    # 获取样本数和标签数
    n_samples, n_labels = y_true.shape

    # 初始化输出变量
    out = 0.0
    # 使用enumerate遍历y_true.indptr数组，i为索引，(start, stop)为每个索引对应的起始和结束指针
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        # 根据起始和结束指针获取y_true.indices中相关的索引
        relevant = y_true.indices[start:stop]

        # 如果relevant数组为空或者大小等于n_labels，则得分为1.0，因为标签排序无意义
        if relevant.size == 0 or relevant.size == n_labels:
            # 如果所有标签都相关或者都不相关，得分也为1.0。标签排序没有意义。
            aux = 1.0
        else:
            # 获取y_score中第i行的分数
            scores_i = y_score[i]
            # 使用"max"方法对scores_i进行排名，并且只选择relevant中的索引进行排名
            rank = rankdata(scores_i, "max")[relevant]
            # 对scores_i中relevant部分的分数再次进行排名
            L = rankdata(scores_i[relevant], "max")
            # 计算aux作为平均值(L / rank)，即平均相关性等级
            aux = (L / rank).mean()

        # 如果sample_weight不为None，则乘以对应的sample_weight[i]
        if sample_weight is not None:
            aux = aux * sample_weight[i]
        
        # 将aux添加到out中
        out += aux

    # 如果sample_weight为None，则计算平均值out除以n_samples
    if sample_weight is None:
        out /= n_samples
    else:
        # 否则，计算out除以sample_weight的总和
        out /= np.sum(sample_weight)

    # 返回计算结果out作为最终的输出值
    return out
@validate_params(
    {
        "y_true": ["array-like"],  # 参数验证装饰器，验证y_true必须是类数组类型
        "y_score": ["array-like"],  # 参数验证装饰器，验证y_score必须是类数组类型
        "sample_weight": ["array-like", None],  # 参数验证装饰器，验证sample_weight可以是类数组类型或者None
    },
    prefer_skip_nested_validation=True,  # 设置参数验证装饰器选项，优先跳过嵌套验证
)
def coverage_error(y_true, y_score, *, sample_weight=None):
    """Coverage error measure.

    Compute how far we need to go through the ranked scores to cover all
    true labels. The best value is equal to the average number
    of labels in ``y_true`` per sample.

    Ties in ``y_scores`` are broken by giving maximal rank that would have
    been assigned to all tied values.

    Note: Our implementation's score is 1 greater than the one given in
    Tsoumakas et al., 2010. This extends it to handle the degenerate case
    in which an instance has 0 true labels.

    Read more in the :ref:`User Guide <coverage_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True binary labels in binary indicator format.

    y_score : array-like of shape (n_samples, n_labels)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    coverage_error : float
        The coverage error.

    References
    ----------
    .. [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010).
           Mining multi-label data. In Data mining and knowledge discovery
           handbook (pp. 667-685). Springer US.

    Examples
    --------
    >>> from sklearn.metrics import coverage_error
    >>> y_true = [[1, 0, 0], [0, 1, 1]]
    >>> y_score = [[1, 0, 0], [0, 1, 1]]
    >>> coverage_error(y_true, y_score)
    1.5
    """
    y_true = check_array(y_true, ensure_2d=True)  # 检查并确保y_true是二维数组
    y_score = check_array(y_score, ensure_2d=True)  # 检查并确保y_score是二维数组
    check_consistent_length(y_true, y_score, sample_weight)  # 检查y_true, y_score和sample_weight的长度一致性

    y_type = type_of_target(y_true, input_name="y_true")  # 获取y_true的类型
    if y_type != "multilabel-indicator":  # 如果y_true不是多标签指示器类型
        raise ValueError("{0} format is not supported".format(y_type))  # 抛出值错误异常

    if y_true.shape != y_score.shape:  # 如果y_true和y_score的形状不同
        raise ValueError("y_true and y_score have different shape")  # 抛出值错误异常

    y_score_mask = np.ma.masked_array(y_score, mask=np.logical_not(y_true))  # 创建掩码数组，将非y_true标签的y_score项掩盖
    y_min_relevant = y_score_mask.min(axis=1).reshape((-1, 1))  # 计算每行中y_true标签的最小y_score值
    coverage = (y_score >= y_min_relevant).sum(axis=1)  # 计算每行中高于或等于最小y_score的数量
    coverage = coverage.filled(0)  # 将缺失值填充为0

    return np.average(coverage, weights=sample_weight)  # 返回加权平均覆盖错误度量


@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # 参数验证装饰器，验证y_true可以是类数组类型或稀疏矩阵
        "y_score": ["array-like"],  # 参数验证装饰器，验证y_score必须是类数组类型
        "sample_weight": ["array-like", None],  # 参数验证装饰器，验证sample_weight可以是类数组类型或None
    },
    prefer_skip_nested_validation=True,  # 设置参数验证装饰器选项，优先跳过嵌套验证
)
def label_ranking_loss(y_true, y_score, *, sample_weight=None):
    """Compute Ranking loss measure.

    Compute the average number of label pairs that are incorrectly ordered
    """
        given y_score weighted by the size of the label set and the number of
        labels not in the label set.
    
        This is similar to the error set size, but weighted by the number of
        relevant and irrelevant labels. The best performance is achieved with
        a ranking loss of zero.
    
        Read more in the :ref:`User Guide <label_ranking_loss>`.
    
        .. versionadded:: 0.17
           A function *label_ranking_loss*
    
        Parameters
        ----------
        y_true : {array-like, sparse matrix} of shape (n_samples, n_labels)
            True binary labels in binary indicator format.
    
        y_score : array-like of shape (n_samples, n_labels)
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by "decision_function" on some classifiers).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
    
        Returns
        -------
        loss : float
            Average number of label pairs that are incorrectly ordered given
            y_score weighted by the size of the label set and the number of labels not
            in the label set.
    
        References
        ----------
        .. [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010).
               Mining multi-label data. In Data mining and knowledge discovery
               handbook (pp. 667-685). Springer US.
    
        Examples
        --------
        >>> from sklearn.metrics import label_ranking_loss
        >>> y_true = [[1, 0, 0], [0, 0, 1]]
        >>> y_score = [[0.75, 0.5, 1], [1, 0.2, 0.1]]
        >>> label_ranking_loss(y_true, y_score)
        0.75...
        """
        # 将 y_true 转换为数组或稀疏矩阵格式，确保是二维的
        y_true = check_array(y_true, ensure_2d=False, accept_sparse="csr")
        # 将 y_score 转换为数组格式，确保是二维的
        y_score = check_array(y_score, ensure_2d=False)
        # 检查 y_true, y_score 和 sample_weight 的长度是否一致
        check_consistent_length(y_true, y_score, sample_weight)
    
        # 确定 y_true 的类型
        y_type = type_of_target(y_true, input_name="y_true")
        # 如果 y_true 不是多标签指示器格式，则引发 ValueError
        if y_type not in ("multilabel-indicator",):
            raise ValueError("{0} format is not supported".format(y_type))
    
        # 如果 y_true 和 y_score 的形状不一致，则引发 ValueError
        if y_true.shape != y_score.shape:
            raise ValueError("y_true and y_score have different shape")
    
        # 获取样本数和标签数
        n_samples, n_labels = y_true.shape
    
        # 将 y_true 转换为 CSR 格式的稀疏矩阵
        y_true = csr_matrix(y_true)
    
        # 初始化 loss 为全零数组，长度为样本数
        loss = np.zeros(n_samples)
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        # 对标签分数进行排序和分组
        unique_scores, unique_inverse = np.unique(y_score[i], return_inverse=True)
        # 计算真实标签在逆序排名下的计数
        true_at_reversed_rank = np.bincount(
            unique_inverse[y_true.indices[start:stop]], minlength=len(unique_scores)
        )
        # 计算所有标签在逆序排名下的计数
        all_at_reversed_rank = np.bincount(unique_inverse, minlength=len(unique_scores))
        # 计算假标签在逆序排名下的计数
        false_at_reversed_rank = all_at_reversed_rank - true_at_reversed_rank

        # 如果分数已经排序，可以通过线性时间来计算错误排序对的数量，
        # 通过累计计算具有较高分数的假标签数量，这些分数比具有较低分数的累计真标签数量高。
        loss[i] = np.dot(true_at_reversed_rank.cumsum(), false_at_reversed_rank)

    n_positives = count_nonzero(y_true, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # 根据正样本数和总标签数计算每个样本的损失值
        loss /= (n_labels - n_positives) * n_positives

    # 当没有正样本或没有负样本标签时，这些值应被视为正确的，即排名无关紧要。
    loss[np.logical_or(n_positives == 0, n_positives == n_labels)] = 0.0

    # 返回加权平均损失值
    return np.average(loss, weights=sample_weight)
# 定义一个函数来计算 Discounted Cumulative Gain（DCG）。
def _dcg_sample_scores(y_true, y_score, k=None, log_base=2, ignore_ties=False):
    """Compute Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : ndarray of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, default=None
        Only consider the highest k scores in the ranking. If `None`, use all
        outputs.

    log_base : float, default=2
        Base of the logarithm used for the discount. A low value means a
        sharper discount (top results are more important).

    ignore_ties : bool, default=False
        Assume that there are no ties in y_score (which is likely to be the
        case if y_score is continuous) for efficiency gains.

    Returns
    -------
    discounted_cumulative_gain : ndarray of shape (n_samples,)
        The DCG score for each sample.

    See Also
    --------
    ndcg_score : The Discounted Cumulative Gain divided by the Ideal Discounted
        Cumulative Gain (the DCG obtained for a perfect ranking), in order to
        have a score between 0 and 1.
    """
    # 计算每个位置的折扣值，以便在计算DCG时使用
    discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_base))
    
    # 如果指定了k，将所有高于k位置的折扣值设为0
    if k is not None:
        discount[k:] = 0
    
    # 如果忽略并列的情况，按照预测分数的顺序排列y_score并计算DCG
    if ignore_ties:
        # 按照y_score的逆序对索引进行排序
        ranking = np.argsort(y_score)[:, ::-1]
        # 按照排序后的索引重新排列y_true，并计算加权累积收益
        ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
        cumulative_gains = discount.dot(ranked.T)
    else:
        # 计算折扣的累计和
        discount_cumsum = np.cumsum(discount)
        # 对于每个样本，计算平均并列DCG
        cumulative_gains = [
            _tie_averaged_dcg(y_t, y_s, discount_cumsum)
            for y_t, y_s in zip(y_true, y_score)
        ]
        cumulative_gains = np.asarray(cumulative_gains)
    
    # 返回计算得到的累积收益
    return cumulative_gains


# 定义一个函数来计算平均并列DCG
def _tie_averaged_dcg(y_true, y_score, discount_cumsum):
    """
    Compute DCG by averaging over possible permutations of ties.

    The gain (`y_true`) of an index falling inside a tied group (in the order
    induced by `y_score`) is replaced by the average gain within this group.
    The discounted gain for a tied group is then the average `y_true` within
    this group times the sum of discounts of the corresponding ranks.

    This amounts to averaging scores for all possible orderings of the tied
    groups.

    (note in the case of dcg@k the discount is 0 after index k)

    Parameters
    ----------
    y_true : ndarray
        The true relevance scores.

    y_score : ndarray
        Predicted scores.
    """
    discount_cumsum : ndarray
        预先计算的折扣累积和数组。

    Returns
    -------
    discounted_cumulative_gain : float
        折扣累积增益。

    References
    ----------
    McSherry, F., & Najork, M. (2008, March). Computing information retrieval
    performance measures efficiently in the presence of tied scores. In
    European conference on information retrieval (pp. 414-421). Springer,
    Berlin, Heidelberg.
    """
    # 使用负值的预测分数对数组进行唯一化，同时返回反向索引和每个唯一值的出现次数
    _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
    # 创建一个全零数组，其长度为唯一值的数量
    ranked = np.zeros(len(counts))
    # 将y_true的值按照inv索引加到ranked数组中对应位置
    np.add.at(ranked, inv, y_true)
    # 将ranked数组中的每个元素除以其对应的出现次数
    ranked /= counts
    # 计算累积组的末尾索引数组
    groups = np.cumsum(counts) - 1
    # 创建一个长度为唯一值数量的空数组
    discount_sums = np.empty(len(counts))
    # 将折扣累积和数组中第一个元素设置为discount_cumsum中groups[0]位置的值
    discount_sums[0] = discount_cumsum[groups[0]]
    # 计算并填充discount_cumsum数组中groups位置之间的差异值
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    # 返回ranked数组和discount_sums数组的点积之和作为最终结果
    return (ranked * discount_sums).sum()
# 定义一个函数来检查目标类型是否符合要求
def _check_dcg_target_type(y_true):
    # 获取目标 y_true 的类型，使用 type_of_target 函数进行判断
    y_type = type_of_target(y_true, input_name="y_true")
    
    # 支持的目标类型格式
    supported_fmt = (
        "multilabel-indicator",   # 多标签指示器
        "continuous-multioutput",  # 连续型多输出
        "multiclass-multioutput",  # 多类别多输出
    )
    
    # 如果 y_true 的类型不在支持的格式列表中，则抛出 ValueError 异常
    if y_type not in supported_fmt:
        raise ValueError(
            "Only {} formats are supported. Got {} instead".format(
                supported_fmt, y_type
            )
        )

# 使用装饰器 validate_params 对 dcg_score 函数的参数进行验证
@validate_params(
    {
        "y_true": ["array-like"],        # y_true 参数必须为类数组
        "y_score": ["array-like"],       # y_score 参数必须为类数组
        "k": [Interval(Integral, 1, None, closed="left"), None],   # k 参数为大于等于1的整数或者 None
        "log_base": [Interval(Real, 0.0, None, closed="neither")], # log_base 参数为大于0的实数
        "sample_weight": ["array-like", None],    # sample_weight 参数可以为类数组或者 None
        "ignore_ties": ["boolean"],     # ignore_ties 参数必须为布尔值
    },
    prefer_skip_nested_validation=True,  # 更喜欢跳过嵌套验证
)
# 定义一个函数来计算 Discounted Cumulative Gain（DCG）得分
def dcg_score(
    y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False
):
    """Compute Discounted Cumulative Gain.

    计算折扣累积增益（DCG）。

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount.

    在预测分数导致的顺序中对真实分数进行加和，应用对数折扣后的结果。

    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.

    如果真实标签在 ``y_score`` 中排名较高，则此排名指标将产生较高的值。

    Usually the Normalized Discounted Cumulative Gain (NDCG, computed by
    ndcg_score) is preferred.

    通常更喜欢标准化折扣累积增益（NDCG，由 ndcg_score 计算）。

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : array-like of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.

    log_base : float, default=2
        Base of the logarithm used for the discount. A low value means a
        sharper discount (top results are more important).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.

    ignore_ties : bool, default=False
        Assume that there are no ties in y_score (which is likely to be the
        case if y_score is continuous) for efficiency gains.

    Returns
    -------
    discounted_cumulative_gain : float
        The averaged sample DCG scores.

    返回
    -------
    折扣累积增益的平均样本得分。

    See Also
    --------
    ndcg_score : The Discounted Cumulative Gain divided by the Ideal Discounted
        Cumulative Gain (the DCG obtained for a perfect ranking), in order to
        have a score between 0 and 1.

    参见
    --------
    ndcg_score: 折扣累积增益除以理想折扣累积增益（对于完美排名获得的DCG），以便得到0到1之间的分数。

    References
    ----------
    `Wikipedia entry for Discounted Cumulative Gain
    <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_.

    维基百科关于折扣累积增益的条目

    Jarvelin, K., & Kekalainen, J. (2002).
    Cumulated gain-based evaluation of IR techniques. ACM Transactions on
    Information Systems (TOIS), 20(4), 422-446.

    Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T. Y. (2013, May).
    ```
    """
    A theoretical analysis of NDCG ranking measures. In Proceedings of the 26th
    Annual Conference on Learning Theory (COLT 2013).

    McSherry, F., & Najork, M. (2008, March). Computing information retrieval
    performance measures efficiently in the presence of tied scores. In
    European conference on information retrieval (pp. 414-421). Springer,
    Berlin, Heidelberg.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import dcg_score
    >>> # we have ground-truth relevance of some answers to a query:
    >>> true_relevance = np.asarray([[10, 0, 0, 1, 5]])
    >>> # we predict scores for the answers
    >>> scores = np.asarray([[.1, .2, .3, 4, 70]])
    >>> dcg_score(true_relevance, scores)
    9.49...
    >>> # we can set k to truncate the sum; only top k answers contribute
    >>> dcg_score(true_relevance, scores, k=2)
    5.63...
    >>> # now we have some ties in our prediction
    >>> scores = np.asarray([[1, 0, 0, 0, 1]])
    >>> # by default ties are averaged, so here we get the average true
    >>> # relevance of our top predictions: (10 + 5) / 2 = 7.5
    >>> dcg_score(true_relevance, scores, k=1)
    7.5
    >>> # we can choose to ignore ties for faster results, but only
    >>> # if we know there aren't ties in our scores, otherwise we get
    >>> # wrong results:
    >>> dcg_score(true_relevance,
    ...           scores, k=1, ignore_ties=True)
    5.0
    """
    # 将 y_true 转换为 NumPy 数组，确保是一维或二维数组
    y_true = check_array(y_true, ensure_2d=False)
    # 将 y_score 转换为 NumPy 数组，确保是一维或二维数组
    y_score = check_array(y_score, ensure_2d=False)
    # 检查 y_true 和 y_score 的长度是否一致，以及是否与 sample_weight 长度一致
    check_consistent_length(y_true, y_score, sample_weight)
    # 检查 y_true 的类型是否符合 DCG 目标的要求
    _check_dcg_target_type(y_true)
    # 返回加权平均后的样本 DCG 分数
    return np.average(
        _dcg_sample_scores(
            y_true, y_score, k=k, log_base=log_base, ignore_ties=ignore_ties
        ),
        weights=sample_weight,
    )
def _ndcg_sample_scores(y_true, y_score, k=None, ignore_ties=False):
    """Compute Normalized Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount. Then divide by the best possible
    score (Ideal DCG, obtained for a perfect ranking) to obtain a score between
    0 and 1.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : ndarray of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.

    ignore_ties : bool, default=False
        Assume that there are no ties in y_score (which is likely to be the
        case if y_score is continuous) for efficiency gains.

    Returns
    -------
    normalized_discounted_cumulative_gain : ndarray of shape (n_samples,)
        The NDCG score for each sample (float in [0., 1.]).

    See Also
    --------
    dcg_score : Discounted Cumulative Gain (not normalized).

    """
    # Compute the Discounted Cumulative Gain (DCG) using the provided y_true and y_score
    gain = _dcg_sample_scores(y_true, y_score, k, ignore_ties=ignore_ties)
    
    # Compute the Ideal DCG as if y_true were perfectly ordered (to normalize)
    normalizing_gain = _dcg_sample_scores(y_true, y_true, k, ignore_ties=True)
    
    # Identify samples where the ideal DCG is zero (all irrelevant items)
    all_irrelevant = normalizing_gain == 0
    
    # Set gain to 0 for samples where the ideal DCG is zero
    gain[all_irrelevant] = 0
    
    # Normalize gain for samples where the ideal DCG is not zero
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
    
    return gain


@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
        "k": [Interval(Integral, 1, None, closed="left"), None],
        "sample_weight": ["array-like", None],
        "ignore_ties": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False):
    """Compute Normalized Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount. Then divide by the best possible
    score (Ideal DCG, obtained for a perfect ranking) to obtain a score between
    0 and 1.

    This ranking metric returns a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : ndarray of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, all samples are weighted equally.

    ignore_ties : bool, default=False
        Assume that there are no ties in y_score (which is likely to be the
        case if y_score is continuous) for efficiency gains.

    """
    y_true : array-like of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked. Negative values in `y_true` may result in an output
        that is not between 0 and 1.

    y_score : array-like of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, default=None
        Only consider the highest k scores in the ranking. If `None`, use all
        outputs.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.

    ignore_ties : bool, default=False
        Assume that there are no ties in y_score (which is likely to be the
        case if y_score is continuous) for efficiency gains.
    # 计算给定评分在特定排名下的归一化折损累积增益（NDCG）评分
    >>> ndcg_score(true_relevance, scores, k=1)
    0.75...
    # 如果我们知道评分中不存在并列的情况，可以选择忽略并列以加快计算速度，但如果存在并列，则会导致错误的结果：
    >>> # we can choose to ignore ties for faster results, but only
    >>> # if we know there aren't ties in our scores, otherwise we get
    >>> # wrong results:
    # 使用给定的真实标签和预测评分计算 NDCG 分数，限制在 k=1 个项目下，忽略并列情况
    >>> ndcg_score(true_relevance,
    ...           scores, k=1, ignore_ties=True)
    0.5...
    """
    # 将真实标签和预测评分转换为数组，并确保它们是一维的
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    # 检查真实标签、预测评分和样本权重的长度一致性
    check_consistent_length(y_true, y_score, sample_weight)

    # 如果真实标签中存在负值，抛出异常
    if y_true.min() < 0:
        raise ValueError("ndcg_score should not be used on negative y_true values.")
    # 如果真实标签是多维的，并且每个样本只有一个文档，则抛出异常
    if y_true.ndim > 1 and y_true.shape[1] <= 1:
        raise ValueError(
            "Computing NDCG is only meaningful when there is more than 1 document. "
            f"Got {y_true.shape[1]} instead."
        )
    # 检查真实标签的目标类型
    _check_dcg_target_type(y_true)
    # 计算每个样本的 NDCG 分数
    gain = _ndcg_sample_scores(y_true, y_score, k=k, ignore_ties=ignore_ties)
    # 根据样本权重计算平均增益
    return np.average(gain, weights=sample_weight)
# 使用装饰器 @validate_params 对函数进行参数验证，确保参数类型和取值范围符合要求
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应为类数组类型
        "y_score": ["array-like"],  # y_score 参数应为类数组类型
        "k": [Interval(Integral, 1, None, closed="left")],  # k 参数应为大于等于1的整数
        "normalize": ["boolean"],  # normalize 参数应为布尔值
        "sample_weight": ["array-like", None],  # sample_weight 参数可以为类数组或者 None
        "labels": ["array-like", None],  # labels 参数可以为类数组或者 None
    },
    prefer_skip_nested_validation=True,  # 设置为 True，优先跳过嵌套验证
)
# 定义函数 top_k_accuracy_score，计算 Top-k 准确率的分类分数
def top_k_accuracy_score(
    y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None
):
    """Top-k Accuracy classification score.

    This metric computes the number of times where the correct label is among
    the top `k` labels predicted (ranked by predicted scores). Note that the
    multilabel case isn't covered here.

    Read more in the :ref:`User Guide <top_k_accuracy_score>`

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores. These can be either probability estimates or
        non-thresholded decision values (as returned by
        :term:`decision_function` on some classifiers).
        The binary case expects scores with shape (n_samples,) while the
        multiclass case expects scores with shape (n_samples, n_classes).
        In the multiclass case, the order of the class scores must
        correspond to the order of ``labels``, if provided, or else to
        the numerical or lexicographical order of the labels in ``y_true``.
        If ``y_true`` does not contain all the labels, ``labels`` must be
        provided.

    k : int, default=2
        Number of most likely outcomes considered to find the correct label.

    normalize : bool, default=True
        If `True`, return the fraction of correctly classified samples.
        Otherwise, return the number of correctly classified samples.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.

    labels : array-like of shape (n_classes,), default=None
        Multiclass only. List of labels that index the classes in ``y_score``.
        If ``None``, the numerical or lexicographical order of the labels in
        ``y_true`` is used. If ``y_true`` does not contain all the labels,
        ``labels`` must be provided.

    Returns
    -------
    score : float
        The top-k accuracy score. The best performance is 1 with
        `normalize == True` and the number of samples with
        `normalize == False`.

    See Also
    --------
    accuracy_score : Compute the accuracy score. By default, the function will
        return the fraction of correct predictions divided by the total number
        of predictions.

    Notes
    -----
    In cases where two or more labels are assigned equal predicted scores,
    the labels with the highest indices will be chosen first. This might
    impact the result if the correct label falls after the threshold because
    of that.

    Examples
    --------
    """
    >>> import numpy as np
    >>> from sklearn.metrics import top_k_accuracy_score
    >>> y_true = np.array([0, 1, 2, 2])
    >>> y_score = np.array([[0.5, 0.2, 0.2],  # 预测结果第一个样本，分类0在前两个最高概率中
    ...                     [0.3, 0.4, 0.2],  # 预测结果第二个样本，分类1在前两个最高概率中
    ...                     [0.2, 0.4, 0.3],  # 预测结果第三个样本，分类2在前两个最高概率中
    ...                     [0.7, 0.2, 0.1]]) # 预测结果第四个样本，分类2不在前两个最高概率中
    >>> top_k_accuracy_score(y_true, y_score, k=2)
    0.75
    >>> # Not normalizing gives the number of "correctly" classified samples
    >>> top_k_accuracy_score(y_true, y_score, k=2, normalize=False)
    3
    
    
    
    """
    确保'y_true'是数组，若不是二维的则将其转换
    """
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    """
    确保'y_true'是一维的数组
    """
    y_true = column_or_1d(y_true)
    """
    确定'y_true'的类型，可能为二元或多类别
    """
    y_type = type_of_target(y_true, input_name="y_true")
    """
    如果'y_type'是"binary"且标签不为空并且标签数大于2，则将'y_type'设为"multiclass"
    """
    if y_type == "binary" and labels is not None and len(labels) > 2:
        y_type = "multiclass"
    """
    如果'y_type'不是"binary"或"multiclass"中的任何一个，则引发值错误
    """
    if y_type not in {"binary", "multiclass"}:
        raise ValueError(
            f"y type must be 'binary' or 'multiclass', got '{y_type}' instead."
        )
    """
    确保'y_score'是数组，且不是二维的
    """
    y_score = check_array(y_score, ensure_2d=False)
    """
    如果'y_type'是"binary"
    """
    if y_type == "binary":
        """
        如果'y_score'是二维的且第二个维度不等于1，则引发值错误
        """
        if y_score.ndim == 2 and y_score.shape[1] != 1:
            raise ValueError(
                "`y_true` is binary while y_score is 2d with"
                f" {y_score.shape[1]} classes. If `y_true` does not contain all the"
                " labels, `labels` must be provided."
            )
        """
        将'y_score'转换为一维数组
        """
        y_score = column_or_1d(y_score)
    
    """
    检查'y_true'、'y_score'和'sample_weight'的长度是否一致
    """
    check_consistent_length(y_true, y_score, sample_weight)
    """
    如果'y_score'的维度是二维的，则'y_score_n_classes'为其第二个维度的大小；否则为2
    """
    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2
    
    """
    如果标签'labels'为空
    """
    if labels is None:
        """
        获取'y_true'中的所有类别并赋值给'classes'
        """
        classes = _unique(y_true)
        """
        计算类别数并赋值给'n_classes'
        """
        n_classes = len(classes)
    
        """
        如果'n_classes'不等于'y_score_n_classes'，则引发值错误
        """
        if n_classes != y_score_n_classes:
            raise ValueError(
                f"Number of classes in 'y_true' ({n_classes}) not equal "
                f"to the number of classes in 'y_score' ({y_score_n_classes})."
                "You can provide a list of all known classes by assigning it "
                "to the `labels` parameter."
            )
    """
    否则，如果标签'labels'不为空
    """
    else:
        """
        将'labels'转换为一维数组
        """
        labels = column_or_1d(labels)
        """
        获取'labels'中的所有类别并赋值给'classes'
        """
        classes = _unique(labels)
        """
        计算标签数和类别数并分别赋值给'n_labels'和'n_classes'
        """
        n_labels = len(labels)
        n_classes = len(classes)
    
        """
        如果'n_classes'不等于'n_labels'，则引发值错误
        """
        if n_classes != n_labels:
            raise ValueError("Parameter 'labels' must be unique.")
    
        """
        如果'classes'和'labels'不相等，则引发值错误
        """
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered.")
    
        """
        如果'n_classes'不等于'y_score_n_classes'，则引发值错误
        """
        if n_classes != y_score_n_classes:
            raise ValueError(
                f"Number of given labels ({n_classes}) not equal to the "
                f"number of classes in 'y_score' ({y_score_n_classes})."
            )
    
        """
        如果'y_true'中存在不在参数'labels'中的标签，则引发值错误
        """
        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'.")
    # 如果 k 大于等于 n_classes，发出警告，因为这会导致得分完美且无意义
    if k >= n_classes:
        warnings.warn(
            (
                f"'k' ({k}) greater than or equal to 'n_classes' ({n_classes}) "
                "will result in a perfect score and is therefore meaningless."
            ),
            UndefinedMetricWarning,
        )

    # 对真实标签进行编码，使用指定的类别列表
    y_true_encoded = _encode(y_true, uniques=classes)

    # 如果目标类型为二元分类
    if y_type == "binary":
        # 如果 k 等于 1，则根据预测分数确定阈值，并将预测结果转换为整数类型
        if k == 1:
            threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            # 检查预测结果与编码后的真实标签是否相等
            hits = y_pred == y_true_encoded
        else:
            # 如果 k 不等于 1，则所有预测都视为命中
            hits = np.ones_like(y_score, dtype=np.bool_)
    # 如果目标类型为多类分类
    elif y_type == "multiclass":
        # 对预测分数进行排序，获取前 k 个最高分数的索引
        sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]
        # 检查编码后的真实标签是否在前 k 个预测中
        hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)

    # 如果需要归一化，则计算加权平均的命中率
    if normalize:
        return np.average(hits, weights=sample_weight)
    # 如果不需要归一化且未提供样本权重，则返回命中次数的总和
    elif sample_weight is None:
        return np.sum(hits)
    # 如果不需要归一化且提供了样本权重，则返回命中次数与样本权重的加权和
    else:
        return np.dot(hits, sample_weight)
```