# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_classification.py`

```
# Metrics to assess performance on classification task given class prediction.
# Functions named as ``*_score`` return a scalar value to maximize: the higher
# the better.
# Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
# the lower the better.
"""
Metrics to assess performance on classification task given class prediction.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import warnings  # 引入警告模块
from numbers import Integral, Real  # 从 numbers 模块中引入 Integral 和 Real 类型

import numpy as np  # 引入 NumPy 库，并使用 np 别名
from scipy.sparse import coo_matrix, csr_matrix  # 从 SciPy 稀疏矩阵模块引入 coo_matrix 和 csr_matrix
from scipy.special import xlogy  # 从 SciPy 特殊函数模块引入 xlogy 函数

from ..exceptions import UndefinedMetricWarning  # 从当前包中引入 UndefinedMetricWarning 异常
from ..preprocessing import LabelBinarizer, LabelEncoder  # 从当前包中引入 LabelBinarizer 和 LabelEncoder 类
from ..utils import (  # 从当前包中引入多个工具函数
    assert_all_finite,
    check_array,
    check_consistent_length,
    column_or_1d,
)
from ..utils._array_api import (  # 从当前包中引入 _array_api 模块下的函数和类
    _average,
    _union1d,
    get_namespace,
    get_namespace_and_device,
)
from ..utils._param_validation import (  # 从当前包中引入 _param_validation 模块下的类
    Hidden,
    Interval,
    Options,
    StrOptions,
    validate_params,
)
from ..utils.extmath import _nanaverage  # 从当前包中引入 _nanaverage 函数
from ..utils.multiclass import type_of_target, unique_labels  # 从当前包中引入 type_of_target 和 unique_labels 函数
from ..utils.sparsefuncs import count_nonzero  # 从当前包中引入 count_nonzero 函数
from ..utils.validation import (  # 从当前包中引入 _check_pos_label_consistency、_check_sample_weight 和 _num_samples 函数
    _check_pos_label_consistency,
    _check_sample_weight,
    _num_samples,
)


def _check_zero_division(zero_division):
    # 检查 zero_division 参数，如果是字符串且为 "warn"，返回浮点数 0.0
    if isinstance(zero_division, str) and zero_division == "warn":
        return np.float64(0.0)
    # 如果是整数或浮点数且在 [0, 1] 范围内，返回对应的浮点数
    elif isinstance(zero_division, (int, float)) and zero_division in [0, 1]:
        return np.float64(zero_division)
    else:  # 如果是 NaN，则返回 NaN
        return np.nan


def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task.

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.

    y_pred : array-like
        Estimated targets as returned by a classifier.

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``.

    y_true : array or indicator matrix
        Ground truth (correct) target values. Returns as array or CSR matrix.

    y_pred : array or indicator matrix
        Estimated targets as returned by a classifier. Returns as array or CSR matrix.
    """
    check_consistent_length(y_true, y_pred)  # 检查 y_true 和 y_pred 的长度一致性
    type_true = type_of_target(y_true, input_name="y_true")  # 获取 y_true 的类型
    type_pred = type_of_target(y_pred, input_name="y_pred")  # 获取 y_pred 的类型

    y_type = {type_true, type_pred}  # 将类型存入集合中
    if y_type == {"binary", "multiclass"}:  # 如果包含二分类和多分类类型，将其视为多分类
        y_type = {"multiclass"}

    if len(y_type) > 1:  # 如果集合中的元素大于 1，说明类型不一致，抛出 ValueError 异常
        raise ValueError(
            "Classification metrics can't handle a mix of {0} and {1} targets".format(
                type_true, type_pred
            )
        )

    # 确定 y_type 的唯一值 => 集合不再需要
    y_type = y_type.pop()
    # 检查 y_type 是否为支持的格式之一，若不是则抛出 ValueError 异常
    if y_type not in ["binary", "multiclass", "multilabel-indicator"]:
        raise ValueError("{0} is not supported".format(y_type))

    # 如果 y_type 是 "binary" 或 "multiclass" 中的一种
    if y_type in ["binary", "multiclass"]:
        # 调用 get_namespace 函数获取 xp 和 _
        xp, _ = get_namespace(y_true, y_pred)
        # 将 y_true 和 y_pred 转换为一维数组或列向量
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        
        # 如果 y_type 是 "binary"
        if y_type == "binary":
            try:
                # 使用 _union1d 函数获取 y_true 和 y_pred 中的唯一值
                unique_values = _union1d(y_true, y_pred, xp)
            except TypeError as e:
                # 如果 y_true 和 y_pred 的数据类型不同，抛出详细的 TypeError 异常
                raise TypeError(
                    "Labels in y_true and y_pred should be of the same type. "
                    f"Got y_true={xp.unique(y_true)} and "
                    f"y_pred={xp.unique(y_pred)}. Make sure that the "
                    "predictions provided by the classifier coincides with "
                    "the true labels."
                ) from e
            # 如果唯一值的数量大于 2，则说明 y_type 应为 "multiclass"
            if unique_values.shape[0] > 2:
                y_type = "multiclass"

    # 如果 y_type 以 "multilabel" 开头
    if y_type.startswith("multilabel"):
        # 将 y_true 和 y_pred 转换为 CSR 矩阵
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        # 设置 y_type 为 "multilabel-indicator"
        y_type = "multilabel-indicator"

    # 返回确定的 y_type、转换后的 y_true 和 y_pred
    return y_type, y_true, y_pred
@validate_params(
    # 使用装饰器 @validate_params 来验证函数的参数
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数应为 array-like 或稀疏矩阵
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数应为 array-like 或稀疏矩阵
        "normalize": ["boolean"],  # normalize 参数应为布尔类型
        "sample_weight": ["array-like", None],  # sample_weight 参数应为 array-like 或者为 None
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
    {
        # y_true: 真实标签数据，类型为类数组
        "y_true": ["array-like"],
        # y_pred: 预测标签数据，类型为类数组
        "y_pred": ["array-like"],
        # labels: 标签集，类型为类数组或空值
        "labels": ["array-like", None],
        # sample_weight: 样本权重，类型为类数组或空值
        "sample_weight": ["array-like", None],
        # normalize: 标准化选项，字符串类型，可选值为{"true", "pred", "all"}或空值
        "normalize": [StrOptions({"true", "pred", "all"}), None],
    },
    # prefer_skip_nested_validation: 是否优先跳过嵌套验证，默认为真
    prefer_skip_nested_validation=True,
# 定义混淆矩阵函数，用于评估分类器的准确性
def confusion_matrix(
    y_true, y_pred, *, labels=None, sample_weight=None, normalize=None
):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Read more in the :ref:`User Guide <confusion_matrix>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

        .. versionadded:: 0.18

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and predicted label being j-th class.

    See Also
    --------
    ConfusionMatrixDisplay.from_estimator : Plot the confusion matrix
        given an estimator, the data, and the label.
    ConfusionMatrixDisplay.from_predictions : Plot the confusion matrix
        given the true and predicted labels.
    ConfusionMatrixDisplay : Confusion Matrix visualization.

    References
    ----------
    .. [1] `Wikipedia entry for the Confusion matrix
           <https://en.wikipedia.org/wiki/Confusion_matrix>`_
           (Wikipedia and other references may use a different
           convention for axes).

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    In the binary case, we can extract true positives, etc. as follows:
    """
    # 从混淆矩阵中获取真负例、假正例、假负例、真正例的数量
    >>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    # 输出真负例、假正例、假负例、真正例的数量
    >>> (tn, fp, fn, tp)
    (0, 2, 1, 1)
    """
    # 检查目标值的类型，并返回目标值的真实值和预测值
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    # 如果目标值类型不是二元或多类分类，则抛出异常
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    # 如果标签为空，则获取真实值和预测值的唯一标签
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)
        n_labels = labels.size
        # 如果标签数量为0，则抛出异常
        if n_labels == 0:
            raise ValueError("'labels' should contains at least one label.")
        # 如果真实值数量为0，则返回全零矩阵
        elif y_true.size == 0:
            return np.zeros((n_labels, n_labels), dtype=int)
        # 如果真实值和标签没有交集，则抛出异常
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in y_true")

    # 如果样本权重为空，则设置样本权重为全1
    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    # 检查真实值、预测值和样本权重的长度是否一致
    check_consistent_length(y_true, y_pred, sample_weight)

    n_labels = labels.size
    # 如果标签不是从零开始的连续整数，则需要将真实值和预测值转换为索引形式
    need_index_conversion = not (
        labels.dtype.kind in {"i", "u", "b"}
        and np.all(labels == np.arange(n_labels))
        and y_true.min() >= 0
        and y_pred.min() >= 0
    )
    if need_index_conversion:
        label_to_ind = {y: x for x, y in enumerate(labels)}
        y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
        y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # 交集真实值和预测值与标签，消除不在标签中的项
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    if not np.all(ind):
        y_pred = y_pred[ind]
        y_true = y_true[ind]
        # 同时消除被消除项的权重
        sample_weight = sample_weight[ind]

    # 选择累加器数据类型以保持高精度
    if sample_weight.dtype.kind in {"i", "u", "b"}:
        dtype = np.int64
    else:
        dtype = np.float64

    # 创建稀疏矩阵并转换为数组形式
    cm = coo_matrix(
        (sample_weight, (y_true, y_pred)),
        shape=(n_labels, n_labels),
        dtype=dtype,
    ).toarray()

    # 忽略警告
    with np.errstate(all="ignore"):
        # 根据normalize参数对混淆矩阵进行归一化
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
        # 将NaN替换为0
        cm = np.nan_to_num(cm)

    # 如果混淆矩阵的形状为(1, 1)，则发出警告
    if cm.shape == (1, 1):
        warnings.warn(
            (
                "A single label was found in 'y_true' and 'y_pred'. For the confusion "
                "matrix to have the correct shape, use the 'labels' parameter to pass "
                "all known labels."
            ),
            UserWarning,
        )

    # 返回混淆矩阵
    return cm
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # 参数验证装饰器，验证函数参数类型和可选性
        "y_pred": ["array-like", "sparse matrix"],  # y_true 和 y_pred 分别是真实值和预测值，可以是数组或稀疏矩阵
        "sample_weight": ["array-like", None],  # 可选参数，样本权重，默认为 None
        "labels": ["array-like", None],  # 可选参数，类别标签，默认为 None
        "samplewise": ["boolean"],  # 可选参数，布尔值，表示是否对每个样本计算混淆矩阵
    },
    prefer_skip_nested_validation=True,  # 参数验证时，优先跳过嵌套验证
)
def multilabel_confusion_matrix(
    y_true, y_pred, *, sample_weight=None, labels=None, samplewise=False
):
    """Compute a confusion matrix for each class or sample.

    .. versionadded:: 0.21  # 标明在版本0.21中添加了此函数

    Compute class-wise (default) or sample-wise (samplewise=True) multilabel
    confusion matrix to evaluate the accuracy of a classification, and output
    confusion matrices for each class or sample.

    In multilabel confusion matrix :math:`MCM`, the count of true negatives
    is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
    true positives is :math:`MCM_{:,1,1}` and false positives is
    :math:`MCM_{:,0,1}`.

    Multiclass data will be treated as if binarized under a one-vs-rest
    transformation. Returned confusion matrices will be in the order of
    sorted unique labels in the union of (y_true, y_pred).

    Read more in the :ref:`User Guide <multilabel_confusion_matrix>`.

    Parameters
    ----------
    y_true : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Ground truth (correct) target values.  # 真实的目标值或正确标签

    y_pred : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Estimated targets as returned by a classifier.  # 分类器返回的预测目标值

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.  # 样本权重，默认为 None

    labels : array-like of shape (n_classes,), default=None
        A list of classes or column indices to select some (or to force
        inclusion of classes absent from the data).  # 类别标签列表或列索引，用于选择类别或强制包含数据中缺失的类别

    samplewise : bool, default=False
        In the multilabel case, this calculates a confusion matrix per sample.  # 多标签情况下，是否计算每个样本的混淆矩阵

    Returns
    -------
    multi_confusion : ndarray of shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default.

    See Also
    --------
    confusion_matrix : Compute confusion matrix to evaluate the accuracy of a
        classifier.

    Notes
    -----
    The `multilabel_confusion_matrix` calculates class-wise or sample-wise
    multilabel confusion matrices, and in multiclass tasks, labels are
    binarized under a one-vs-rest way; while
    :func:`~sklearn.metrics.confusion_matrix` calculates one confusion matrix
    for confusion between every two classes.

    Examples
    --------
    Multilabel-indicator case:
    """
    Perform confusion matrix calculation for multilabel or multiclass classification.

    Parameters:
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.

    labels : array-like of shape (n_classes,), optional
        List of labels to index the confusion matrix. If None, derive from y_true and y_pred.

    samplewise : bool, default=False
        Whether to evaluate metrics sample-wise.

    Returns:
    -------
    tp_sum : array, shape (n_labels,)
        True positives for each label.

    true_sum : array, shape (n_labels,)
        Number of true instances for each label.

    pred_sum : array, shape (n_labels,)
        Number of predicted instances for each label.
    """
    # Validate and preprocess input labels and predictions
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    
    # Handle sample weights if provided
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    
    # Ensure consistent length of inputs
    check_consistent_length(y_true, y_pred, sample_weight)

    # Check if y_type is one of the supported types
    if y_type not in ("binary", "multiclass", "multilabel-indicator"):
        raise ValueError("%s is not supported" % y_type)

    # Determine unique labels present in y_true and y_pred
    present_labels = unique_labels(y_true, y_pred)
    
    # If labels parameter is not provided, use present_labels
    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        # Ensure labels are in present_labels and in the correct order
        labels = np.hstack(
            [labels, np.setdiff1d(present_labels, labels, assume_unique=True)]
        )

    # Handle the case when y_true is 1-dimensional
    if y_true.ndim == 1:
        # Raise error if samplewise metrics are requested (not supported for 1D y_true)
        if samplewise:
            raise ValueError(
                "Samplewise metrics are not available outside of "
                "multilabel classification."
            )

        # Encode labels to numerical values
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # Calculate true positives (tp)
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        
        # Apply sample weights to tp_bins if provided
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None
        
        # Calculate tp_sum using bincount, considering minimum length of labels
        if len(tp_bins):
            tp_sum = np.bincount(
                tp_bins, weights=tp_bins_weights, minlength=len(labels)
            )
        else:
            # Handle the case when there are no true positives
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        
        # Calculate pred_sum and true_sum using bincount, considering minimum length of labels
        if len(y_pred):
            pred_sum = np.bincount(y_pred, weights=sample_weight, minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight, minlength=len(labels))

        # Retain only selected labels as per sorted_labels and labels[:n_labels]
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]
    else:
        sum_axis = 1 if samplewise else 0

        # All labels are index integers for multilabel.
        # Select labels:
        # 检查是否需要更新标签集合
        if not np.array_equal(labels, present_labels):
            # 如果标签中存在超出当前标签集合范围的最大值，抛出数值错误
            if np.max(labels) > np.max(present_labels):
                raise ValueError(
                    "All labels must be in [0, n labels) for "
                    "multilabel targets. "
                    "Got %d > %d" % (np.max(labels), np.max(present_labels))
                )
            # 如果标签中存在小于零的值，抛出数值错误
            if np.min(labels) < 0:
                raise ValueError(
                    "All labels must be in [0, n labels) for "
                    "multilabel targets. "
                    "Got %d < 0" % np.min(labels)
                )

        # 如果有限制标签数量，则截取前 n_labels 个标签
        if n_labels is not None:
            y_true = y_true[:, labels[:n_labels]]
            y_pred = y_pred[:, labels[:n_labels]]

        # 计算加权计数
        true_and_pred = y_true.multiply(y_pred)
        # 计算真实正类的总数
        tp_sum = count_nonzero(
            true_and_pred, axis=sum_axis, sample_weight=sample_weight
        )
        # 计算预测正类的总数
        pred_sum = count_nonzero(y_pred, axis=sum_axis, sample_weight=sample_weight)
        # 计算真实类的总数
        true_sum = count_nonzero(y_true, axis=sum_axis, sample_weight=sample_weight)

    # 计算假正类的总数
    fp = pred_sum - tp_sum
    # 计算假负类的总数
    fn = true_sum - tp_sum
    # 计算真正类的总数
    tp = tp_sum

    # 处理样本权重和 samplewise 参数
    if sample_weight is not None and samplewise:
        sample_weight = np.array(sample_weight)
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        # 计算真负类的总数
        tn = sample_weight * y_true.shape[1] - tp - fp - fn
    elif sample_weight is not None:
        # 计算真负类的总数
        tn = sum(sample_weight) - tp - fp - fn
    elif samplewise:
        # 计算真负类的总数
        tn = y_true.shape[1] - tp - fp - fn
    else:
        # 计算真负类的总数
        tn = y_true.shape[0] - tp - fp - fn

    # 返回结果数组，包含真负类、假正类、假负类、真正类的统计信息
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)
# 定义一个函数用于处理分母为零的情况，主要用于测量度量的函数
def _metric_handle_division(*, numerator, denominator, metric, zero_division):
    """Helper to handle zero-division.

    Parameters
    ----------
    numerator : numbers.Real
        The numerator of the division.
    denominator : numbers.Real
        The denominator of the division.
    metric : str
        Name of the caller metric function.
    zero_division : {0.0, 1.0, "warn"}
        The strategy to use when encountering 0-denominator.

    Returns
    -------
    result : numbers.Real
        The resulting of the division
    is_zero_division : bool
        Whether or not we encountered a zero division. This value could be
        required to early return `result` in the "caller" function.
    """
    # 如果分母接近于零
    if np.isclose(denominator, 0):
        # 如果零除策略是警告
        if zero_division == "warn":
            # 构造警告信息，提示用户使用 `zero_division` 参数控制行为
            msg = f"{metric} is ill-defined and set to 0.0. Use the `zero_division` " \
                  "param to control this behavior."
            # 发出警告
            warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)
        # 返回零除策略的结果和 True 表示发生了零除
        return _check_zero_division(zero_division), True
    # 返回正常的除法结果和 False 表示没有零除
    return numerator / denominator, False


# 使用装饰器验证参数，并定义计算 Cohen's kappa 的函数
@validate_params(
    {
        "y1": ["array-like"],
        "y2": ["array-like"],
        "labels": ["array-like", None],
        "weights": [StrOptions({"linear", "quadratic"}), None],
        "sample_weight": ["array-like", None],
        "zero_division": [
            StrOptions({"warn"}),
            Options(Real, {0.0, 1.0, np.nan}),
        ],
    },
    prefer_skip_nested_validation=True,
)
def cohen_kappa_score(
    y1, y2, *, labels=None, weights=None, sample_weight=None, zero_division="warn"
):
    r"""Compute Cohen's kappa: a statistic that measures inter-annotator agreement.

    This function computes Cohen's kappa [1]_, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement on the label
    assigned to any sample (the observed agreement ratio), and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly.
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels [2]_.

    Read more in the :ref:`User Guide <cohen_kappa>`.

    Parameters
    ----------
    y1 : array-like of shape (n_samples,)
        Labels assigned by the first annotator.

    y2 : array-like of shape (n_samples,)
        Labels assigned by the second annotator. The kappa statistic is
        symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.

    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to select a
        subset of labels. If `None`, all labels that appear at least once in
        ``y1`` or ``y2`` are used.
    # 权重类型，用于计算分数。默认为 `None` 表示不使用权重；"linear" 表示线性加权；"quadratic" 表示二次加权。
    weights : {'linear', 'quadratic'}, default=None
        Weighting type to calculate the score. `None` means not weighted;
        "linear" means linear weighting; "quadratic" means quadratic weighting.

    # 样本权重，形状为 (n_samples,) 的数组，默认为 `None`。
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    # 当存在零除法时设置返回值。例如当 `y1` 和 `y2` 都只包含0类别（如 `[0, 0, 0, 0]`）或者两者都为空时。
    # 如果设置为 "warn"，返回 `0.0`，并且会发出警告。
    # 
    # .. versionadded:: 1.6
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Sets the return value when there is a zero division. This is the case when both
        labelings `y1` and `y2` both exclusively contain the 0 class (e. g.
        `[0, 0, 0, 0]`) (or if both are empty). If set to "warn", returns `0.0`, but a
        warning is also raised.

    # 返回值
    # -------
    # kappa : float
    #     Cohen's kappa 统计量，取值范围为 -1 到 1。最大值表示完全一致；0 或更低表示偶然一致。
    # 
    # 参考文献
    # ----------
    # .. [1] :doi:`J. Cohen (1960). "A coefficient of agreement for nominal scales".
    #        Educational and Psychological Measurement 20(1):37-46.
    #        <10.1177/001316446002000104>`
    # .. [2] `R. Artstein and M. Poesio (2008). "Inter-coder agreement for
    #        computational linguistics". Computational Linguistics 34(4):555-596
    #        <https://www.mitpressjournals.org/doi/pdf/10.1162/coli.07-034-R2>`_.
    # .. [3] `Wikipedia entry for the Cohen's kappa
    #         <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_.

    # 示例
    # --------
    # >>> from sklearn.metrics import cohen_kappa_score
    # >>> y1 = ["negative", "positive", "negative", "neutral", "positive"]
    # >>> y2 = ["negative", "positive", "negative", "neutral", "negative"]
    # >>> cohen_kappa_score(y1, y2)
    # 0.6875
    confusion = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)

    # 计算分子
    numerator = np.outer(sum0, sum1)
    denominator = np.sum(sum0)
    expected, is_zero_division = _metric_handle_division(
        numerator=numerator,
        denominator=denominator,
        metric="cohen_kappa_score()",
        zero_division=zero_division,
    )

    # 处理零除法情况
    if is_zero_division:
        return expected

    # 根据权重类型构建权重矩阵
    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=int)
        w_mat.flat[:: n_classes + 1] = 0
    else:  # "linear" or "quadratic"
        w_mat = np.zeros([n_classes, n_classes], dtype=int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2

    # 计算新的分子和分母
    numerator = np.sum(w_mat * confusion)
    denominator = np.sum(w_mat * expected)

    # 再次处理零除法情况
    score, is_zero_division = _metric_handle_division(
        numerator=numerator,
        denominator=denominator,
        metric="cohen_kappa_score()",
        zero_division=zero_division,
    )

    # 返回最终得分
    if is_zero_division:
        return score
    return 1 - score
@validate_params(
    {  # 参数验证装饰器，验证函数参数的类型和取值范围
        "y_true": ["array-like", "sparse matrix"],  # y_true参数可以是类数组或稀疏矩阵
        "y_pred": ["array-like", "sparse matrix"],  # y_pred参数可以是类数组或稀疏矩阵
        "labels": ["array-like", None],  # labels参数可以是类数组，或者可以为None
        "pos_label": [Real, str, "boolean", None],  # pos_label参数可以是实数、字符串、布尔值或者None
        "average": [
            StrOptions({"micro", "macro", "samples", "weighted", "binary"}),
            None,
        ],  # average参数可以是预定义字符串集合中的一种，或者可以为None
        "sample_weight": ["array-like", None],  # sample_weight参数可以是类数组，或者可以为None
        "zero_division": [
            Options(Real, {0, 1}),
            StrOptions({"warn"}),
        ],  # zero_division参数可以是实数0或1的集合，或者字符串"warn"
    },
    prefer_skip_nested_validation=True,  # 设置跳过嵌套验证为True
)
def jaccard_score(
    y_true,
    y_pred,
    *,
    labels=None,  # labels参数，默认为None
    pos_label=1,  # pos_label参数，默认为1
    average="binary",  # average参数，默认为"binary"
    sample_weight=None,  # sample_weight参数，默认为None
    zero_division="warn",  # zero_division参数，默认为"warn"
):
    """Jaccard similarity coefficient score.

    The Jaccard index [1], or Jaccard similarity coefficient, defined as
    the size of the intersection divided by the size of the union of two label
    sets, is used to compare set of predicted labels for a sample to the
    corresponding set of labels in ``y_true``.

    Support beyond term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return the
    Jaccard similarity coefficient for `pos_label`. If `average` is not `'binary'`,
    `pos_label` is ignored and scores for both classes are computed, then averaged or
    both returned (when `average=None`). Similarly, for :term:`multiclass` and
    :term:`multilabel` targets, scores for all `labels` are either returned or
    averaged depending on the `average` parameter. Use `labels` specify the set of
    labels to calculate the score for.

    Read more in the :ref:`User Guide <jaccard_similarity_score>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    labels : array-like of shape (n_classes,), default=None
        The set of labels to include when `average != 'binary'`, and their
        order if `average is None`. Labels present in the data can be
        excluded, for example in multiclass classification to exclude a "negative
        class". Labels not present in the data can be included and will be
        "assigned" 0 samples. For multilabel targets, labels are column indices.
        By default, all labels in `y_true` and `y_pred` are used in sorted order.

    pos_label : int, float, bool or str, default=1
        The class to report if `average='binary'` and the data is binary,
        otherwise this parameter is ignored.
        For multiclass or multilabel targets, set `labels=[pos_label]` and
        `average != 'binary'` to report metrics for one label only.
    average : {'micro', 'macro', 'samples', 'weighted', \
            'binary'} or None, default='binary'
        如果为 ``None``，则返回每个类别的得分。否则，确定对数据执行的平均类型：

        ``'binary'``:
            仅报告由 ``pos_label`` 指定的类别的结果。
            仅当目标（``y_{true,pred}``）为二元时适用。
        ``'micro'``:
            通过计算总的真正例、假反例和假正例来全局计算指标。
        ``'macro'``:
            为每个标签计算指标，并找到它们的未加权平均值。这不考虑标签不平衡。
        ``'weighted'``:
            为每个标签计算指标，并找到它们的加权平均值，权重为支持度（每个标签的真实实例数）。这改变了 'macro' 以考虑标签不平衡。
        ``'samples'``:
            为每个实例计算指标，并找到它们的平均值（对多标签分类有意义）。

    sample_weight : array-like of shape (n_samples,), default=None
        样本权重。

    zero_division : "warn", {0.0, 1.0}, default="warn"
        在存在零除法（即预测和标签中没有负值）时设置返回值。如果设置为 "warn"，则行为类似于 0，但还会引发警告。

    Returns
    -------
    score : float or ndarray of shape (n_unique_labels,), dtype=np.float64
        Jaccard 分数。当 `average` 不是 `None` 时，返回单个标量。

    See Also
    --------
    accuracy_score : 用于计算准确率的函数。
    f1_score : 用于计算 F1 分数的函数。
    multilabel_confusion_matrix : 用于为每个类别或样本计算混淆矩阵的函数。

    Notes
    -----
    :func:`jaccard_score` 如果某些样本或类别没有正例，可能是一个较差的度量。如果某些样本或类别没有真实或预测的标签，则 Jaccard 未定义，我们的实现将返回带有警告的分数 0。

    References
    ----------
    .. [1] `Jaccard index 的维基百科条目
           <https://en.wikipedia.org/wiki/Jaccard_index>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import jaccard_score
    >>> y_true = np.array([[0, 1, 1],
    ...                    [1, 1, 0]])
    >>> y_pred = np.array([[1, 1, 1],
    ...                    [1, 0, 0]])

    在二元情况下：

    >>> jaccard_score(y_true[0], y_pred[0])
    0.6666...

    在二维比较情况下（例如图像相似性）：

    >>> jaccard_score(y_true, y_pred, average="micro")
    0.6

    在多标签情况下：

    >>> jaccard_score(y_true, y_pred, average='samples')
    0.5833...
    >>> jaccard_score(y_true, y_pred, average='macro')
    # 计算 Jaccard 相似度分数（Intersection over Union）的多类别情况下的评估值

    # 对真实标签和预测标签进行检查和准备，根据指定的平均方式和标签进行处理
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

    # 判断是否需要基于样本进行计算
    samplewise = average == "samples"

    # 计算多标签混淆矩阵，包括样本权重和指定的标签
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )

    # 计算 Jaccard 相似度的分子部分，即正例与正例的交集
    numerator = MCM[:, 1, 1]

    # 计算 Jaccard 相似度的分母部分，包括正例与负例的交集和反例与正例的交集
    denominator = MCM[:, 1, 1] + MCM[:, 0, 1] + MCM[:, 1, 0]

    # 如果指定计算微观平均，则将分子和分母的计算结果转为数组并求和
    if average == "micro":
        numerator = np.array([numerator.sum()])
        denominator = np.array([denominator.sum()])

    # 使用内部函数计算 Jaccard 相似度，包括处理零除情况
    jaccard = _prf_divide(
        numerator,
        denominator,
        "jaccard",
        "true or predicted",
        average,
        ("jaccard",),
        zero_division=zero_division,
    )

    # 如果未指定平均方式，则直接返回 Jaccard 相似度
    if average is None:
        return jaccard

    # 如果指定按权重平均，计算权重并进行加权平均
    if average == "weighted":
        weights = MCM[:, 1, 0] + MCM[:, 1, 1]
        if not np.any(weights):
            # 如果权重为空，即分子为0，通常会发出警告
            weights = None
    # 如果指定按样本平均，并且提供了样本权重，则使用样本权重
    elif average == "samples" and sample_weight is not None:
        weights = sample_weight
    else:
        weights = None

    # 返回加权平均后的 Jaccard 相似度
    return np.average(jaccard, weights=weights)
# 使用装饰器 validate_params 对函数进行参数验证，确保 y_true、y_pred 和 sample_weight 符合预期类型
@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
# 定义 Matthews 相关系数函数，用于评估二分类和多分类模型预测的质量
def matthews_corrcoef(y_true, y_pred, *, sample_weight=None):
    """Compute the Matthews correlation coefficient (MCC).

    The Matthews correlation coefficient is used in machine learning as a
    measure of the quality of binary and multiclass classifications. It takes
    into account true and false positives and negatives and is generally
    regarded as a balanced measure which can be used even if the classes are of
    very different sizes. The MCC is in essence a correlation coefficient value
    between -1 and +1. A coefficient of +1 represents a perfect prediction, 0
    an average random prediction and -1 an inverse prediction.  The statistic
    is also known as the phi coefficient. [source: Wikipedia]

    Binary and multiclass labels are supported.  Only in the binary case does
    this relate to information about true and false positives and negatives.
    See references below.

    Read more in the :ref:`User Guide <matthews_corrcoef>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

        .. versionadded:: 0.18

    Returns
    -------
    mcc : float
        The Matthews correlation coefficient (+1 represents a perfect
        prediction, 0 an average random prediction and -1 and inverse
        prediction).

    References
    ----------
    .. [1] :doi:`Baldi, Brunak, Chauvin, Andersen and Nielsen, (2000). Assessing the
       accuracy of prediction algorithms for classification: an overview.
       <10.1093/bioinformatics/16.5.412>`

    .. [2] `Wikipedia entry for the Matthews Correlation Coefficient (phi coefficient)
       <https://en.wikipedia.org/wiki/Phi_coefficient>`_.

    .. [3] `Gorodkin, (2004). Comparing two K-category assignments by a
        K-category correlation coefficient
        <https://www.sciencedirect.com/science/article/pii/S1476927104000799>`_.

    .. [4] `Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC and CEN
        Error Measures in MultiClass Prediction
        <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041882>`_.

    Examples
    --------
    >>> from sklearn.metrics import matthews_corrcoef
    >>> y_true = [+1, +1, +1, -1]
    >>> y_pred = [+1, -1, +1, +1]
    >>> matthews_corrcoef(y_true, y_pred)
    -0.33...
    """
    # 检查 y_true 和 y_pred 的类型，并将它们转换为内部标签类型
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    # 检查 y_true、y_pred 和 sample_weight 是否具有一致的长度
    check_consistent_length(y_true, y_pred, sample_weight)
    # 如果 y_type 不是 "binary" 或 "multiclass"，则抛出错误
    if y_type not in {"binary", "multiclass"}:
        raise ValueError("%s is not supported" % y_type)

    # 初始化标签编码器并拟合 y_true 和 y_pred 的组合
    lb = LabelEncoder()
    lb.fit(np.hstack([y_true, y_pred]))
    # 使用标签编码器转换真实标签和预测标签为二进制矩阵形式
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)

    # 计算混淆矩阵 C，可以指定样本权重
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    
    # 计算每个类别的真实样本数和预测样本数的总和
    t_sum = C.sum(axis=1, dtype=np.float64)  # 真实样本数的总和
    p_sum = C.sum(axis=0, dtype=np.float64)  # 预测样本数的总和
    
    # 计算正确预测的样本数（对角线元素之和）
    n_correct = np.trace(C, dtype=np.float64)
    
    # 计算总样本数（预测样本数的总和）
    n_samples = p_sum.sum()
    
    # 计算协方差 cov_ytyp，用于后续计算
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    
    # 计算协方差 cov_ypyp，用于后续计算
    cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
    
    # 计算协方差 cov_ytyt，用于后续计算
    cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)
    
    # 如果 cov_ypyp * cov_ytyt 的乘积为 0，则返回 0.0，否则计算并返回相关系数
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    else:
        return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
# 使用装饰器 @validate_params 对 zero_one_loss 函数进行参数验证，确保输入参数的类型和取值符合预期
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数可以是类数组或稀疏矩阵
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数可以是类数组或稀疏矩阵
        "normalize": ["boolean"],  # normalize 参数必须是布尔值
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是类数组或者 None
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
    {
        # 真实标签（ground truth），可以是类似数组的数据结构或稀疏矩阵
        "y_true": ["array-like", "sparse matrix"],
        # 预测标签，与真实标签结构相同
        "y_pred": ["array-like", "sparse matrix"],
        # 标签列表，可以是类似数组的数据结构或为空
        "labels": ["array-like", None],
        # 正类标签的定义，可以是实数、字符串、布尔值或为空
        "pos_label": [Real, str, "boolean", None],
        # 平均方法，可以是字符串选项集合{"micro", "macro", "samples", "weighted", "binary"}或为空
        "average": [
            StrOptions({"micro", "macro", "samples", "weighted", "binary"}),
            None,
        ],
        # 样本权重，可以是类似数组的数据结构或为空
        "sample_weight": ["array-like", None],
        # 零除警告时的行为定义，可以是实数选项集合{0.0, 1.0}、"nan"或字符串选项集合{"warn"}
        "zero_division": [
            Options(Real, {0.0, 1.0}),
            "nan",
            StrOptions({"warn"}),
        ],
    },
    # 设置为True，表示首选跳过嵌套验证
    prefer_skip_nested_validation=True,
# 定义函数 f1_score，用于计算 F1 分数，即平衡 F 分数或 F 测量值。

# F1 分数可以解释为精确度和召回率的调和平均值，在 F1 分数中，最佳值为 1，最差值为 0。
# 精确度和召回率对 F1 分数的相对贡献是相等的。F1 分数的计算公式如下：
# F1 = 2 * TP / (2 * TP + FP + FN)
# 其中 TP 是真阳性的数量，FN 是假阴性的数量，FP 是假阳性的数量。
# 当没有真阳性、假阴性或假阳性时，默认情况下 F1 被计算为 0.0。

# 支持超出二元目标，通过将多类和多标签数据视为每个标签的二进制问题的集合来实现。
# 对于二元情况，设置 `average='binary'` 将返回 `pos_label` 的 F1 分数。
# 如果 `average` 不是 `'binary'`，则忽略 `pos_label` 并计算两个类别的 F1 分数，
# 然后对其进行平均或同时返回（当 `average=None` 时）。
# 类似地，对于多类和多标签目标，根据 `average` 参数，返回或平均计算所有 `labels` 的 F1 分数。
# 使用 `labels` 参数指定要计算 F1 分数的标签集合。

# 更多详细信息请参阅用户指南中的 precision_recall_f_measure_metrics 部分。

# 参数说明：
def f1_score(
    y_true,  # 1 维数组或标签指示器数组/稀疏矩阵，真实的目标值（正确值）。
    y_pred,  # 1 维数组或标签指示器数组/稀疏矩阵，分类器返回的预测目标值。
    *,  # 之后的参数为关键字参数。
    labels=None,  # 数组型，默认为 None。在 `average != 'binary'` 时要包含的标签集合及其顺序。在多类分类中，可以排除数据中的某些标签，如排除“负类”。不在数据中的标签可以包含，并将被“分配”为 0 个样本。对于多标签目标，标签是列索引。默认情况下，按顺序使用 `y_true` 和 `y_pred` 中的所有标签。
    pos_label=1,  # 整数、浮点数、布尔值或字符串，默认为 1。当 `average='binary'` 且数据为二元时报告的类别。对于多类或多标签目标，设置 `labels=[pos_label]` 和 `average != 'binary'` 来仅报告一个标签的指标。
    average="binary",  # 字符串，默认为 "binary"。指定在多类和多标签目标中如何计算指标。如果为 "binary"，则返回 `pos_label` 的 F1 分数。如果不是 "binary"，则忽略 `pos_label` 并计算两个类别的 F1 分数，然后平均或同时返回（当 `average=None` 时）。
    sample_weight=None,  # 数组型，默认为 None。样本权重。
    zero_division="warn",  # 字符串，默认为 "warn"。控制当分母为零时警告的行为。
):
    """
    计算 F1 分数，也称为平衡 F 分数或 F 测量值。

    F1 分数可以解释为精确度和召回率的调和平均值，在 F1 分数中，最佳值为 1，最差值为 0。
    精确度和召回率对 F1 分数的相对贡献是相等的。F1 分数的计算公式如下：

    .. math::
        \\text{F1} = \\frac{2 * \\text{TP}}{2 * \\text{TP} + \\text{FP} + \\text{FN}}

    其中 :math:`\\text{TP}` 是真阳性的数量，:math:`\\text{FN}` 是假阴性的数量，:math:`\\text{FP}` 是假阳性的数量。
    当没有真阳性、假阴性或假阳性时，默认情况下 F1 被计算为 0.0。

    支持超出二元目标，通过将多类和多标签数据视为每个标签的二进制问题的集合来实现。
    对于二元情况，设置 `average='binary'` 将返回 `pos_label` 的 F1 分数。
    如果 `average` 不是 `'binary'`，则忽略 `pos_label` 并计算两个类别的 F1 分数，
    然后对其进行平均或同时返回（当 `average=None` 时）。
    类似地，对于多类和多标签目标，根据 `average` 参数，返回或平均计算所有 `labels` 的 F1 分数。
    使用 `labels` 参数指定要计算 F1 分数的标签集合。

    详细信息请参阅用户指南中的 precision_recall_f_measure_metrics 部分。

    Parameters
    ----------
    y_true : 1 维数组型，或标签指示器数组/稀疏矩阵
        真实的目标值（正确值）。

    y_pred : 1 维数组型，或标签指示器数组/稀疏矩阵
        分类器返回的预测目标值。

    labels : 数组型，默认为 None
        在 `average != 'binary'` 时要包含的标签集合及其顺序。在多类分类中，可以排除数据中的某些标签，如排除“负类”。不在数据中的标签可以包含，并将被“分配”为 0 个样本。对于多标签目标，标签是列索引。默认情况下，按顺序使用 `y_true` 和 `y_pred` 中的所有标签。

        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.

    pos_label : 整数、浮点数、布尔值或字符串，默认为 1
        当 `average='binary'` 且数据为二元时报告的类别。对于多类或多标签目标，设置 `labels=[pos_label]` 和 `average != 'binary'` 来仅报告一个标签的指标。
    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, \
            default='binary'
        这个参数对于多类别/多标签目标是必需的。
        如果为 ``None``，则返回每个类别的分数。否则，确定在数据上执行的平均类型：

        ``'binary'``:
            仅报告由 ``pos_label`` 指定的类别的结果。
            仅当目标（``y_{true,pred}``）是二进制时适用。
        ``'micro'``:
            通过计算总体的真正例、假反例和假正例来全局计算度量标准。
        ``'macro'``:
            为每个标签计算度量标准，并找到它们的未加权平均值。这不考虑标签不平衡。
        ``'weighted'``:
            为每个标签计算度量标准，并按支持度加权平均。这会改变 'macro' 来考虑标签不平衡；可能导致 F-score 不在精确率和召回率之间。
        ``'samples'``:
            为每个实例计算度量标准，并找到它们的平均值（仅在多标签分类中有意义，在这种情况下与 :func:`accuracy_score` 不同）。

    sample_weight : array-like of shape (n_samples,), default=None
        样本权重。

    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        设置当存在零除法（即所有预测和标签均为负）时要返回的值。

        注意：
        - 如果设置为 "warn"，则行为类似于 0，但也会引发警告。
        - 如果设置为 `np.nan`，这样的值将从平均值中排除。

        .. versionadded:: 1.3
           添加了 `np.nan` 选项。

    Returns
    -------
    f1_score : float or array of float, shape = [n_unique_labels]
        二分类中正类别的 F1 分数或多类任务中每个类别的加权平均 F1 分数。

    See Also
    --------
    fbeta_score : 计算 F-beta 分数。
    precision_recall_fscore_support : 计算精确率、召回率、F-score 和支持度。
    jaccard_score : 计算杰卡德相似系数分数。
    multilabel_confusion_matrix : 计算每个类别或样本的混淆矩阵。

    Notes
    -----
    当 ``真正例 + 假正例 + 假反例 == 0``（即一个类在 ``y_true`` 或 ``y_pred`` 中完全不存在）时，F-score 未定义。在这种情况下，默认情况下，F-score 将设置为 0.0，并引发 ``UndefinedMetricWarning`` 警告。可以通过设置 ``zero_division`` 参数修改此行为。

    References
    ----------
    .. [1] `F1-score 的维基百科条目
           <https://en.wikipedia.org/wiki/F1_score>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import f1_score  # 导入 f1_score 函数，用于计算 F1 分数
    >>> y_true = [0, 1, 2, 0, 1, 2]  # 真实标签列表
    >>> y_pred = [0, 2, 1, 0, 0, 1]  # 预测标签列表
    >>> f1_score(y_true, y_pred, average='macro')  # 计算宏平均 F1 分数
    0.26...
    >>> f1_score(y_true, y_pred, average='micro')  # 计算微平均 F1 分数
    0.33...
    >>> f1_score(y_true, y_pred, average='weighted')  # 计算加权平均 F1 分数
    0.26...
    >>> f1_score(y_true, y_pred, average=None)  # 计算每个类别的 F1 分数
    array([0.8, 0. , 0. ])

    >>> # 二分类情况
    >>> y_true_empty = [0, 0, 0, 0, 0, 0]  # 空的真实标签列表
    >>> y_pred_empty = [0, 0, 0, 0, 0, 0]  # 空的预测标签列表
    >>> f1_score(y_true_empty, y_pred_empty)  # 计算 F1 分数，处理零除错误
    0.0...
    >>> f1_score(y_true_empty, y_pred_empty, zero_division=1.0)  # 使用指定的零除值
    1.0...
    >>> f1_score(y_true_empty, y_pred_empty, zero_division=np.nan)  # 使用 NaN 作为零除值
    nan...

    >>> # 多标签分类情况
    >>> y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]  # 多标签的真实标签列表
    >>> y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]  # 多标签的预测标签列表
    >>> f1_score(y_true, y_pred, average=None)  # 计算每个类别的 F1 分数
    array([0.66666667, 1.        , 0.66666667])
    """
    return fbeta_score(  # 返回 F-beta 分数，这是一个扩展的 F1 分数
        y_true,
        y_pred,
        beta=1,  # 设置 beta 值为 1，即 F1 分数
        labels=labels,  # 指定要评估的标签
        pos_label=pos_label,  # 指定正类标签
        average=average,  # 指定计算平均值的方式
        sample_weight=sample_weight,  # 指定样本权重
        zero_division=zero_division,  # 指定零除情况下的返回值
    )
# 使用装饰器 @validate_params 对函数 fbeta_score 进行参数验证
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数应为 array-like 或 sparse matrix 类型
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数应为 array-like 或 sparse matrix 类型
        "beta": [Interval(Real, 0.0, None, closed="both")],  # beta 参数应为大于等于 0 的实数类型
        "labels": ["array-like", None],  # labels 参数应为 array-like 类型或 None
        "pos_label": [Real, str, "boolean", None],  # pos_label 参数应为实数、字符串、布尔值或 None
        "average": [
            StrOptions({"micro", "macro", "samples", "weighted", "binary"}),  # average 参数应为预定义字符串集合中的一种
            None,  # average 参数也可以为 None
        ],
        "sample_weight": ["array-like", None],  # sample_weight 参数应为 array-like 类型或 None
        "zero_division": [
            Options(Real, {0.0, 1.0}),  # zero_division 参数应为实数，可以是 0.0 或 1.0
            "nan",  # zero_division 参数也可以为字符串 "nan"
            StrOptions({"warn"}),  # 或者为预定义字符串集合中的 "warn"
        ],
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def fbeta_score(
    y_true,
    y_pred,
    *,
    beta,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    """Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter represents the ratio of recall importance to
    precision importance. `beta > 1` gives more weight to recall, while
    `beta < 1` favors precision. For example, `beta = 2` makes recall twice
    as important as precision, while `beta = 0.5` does the opposite.
    Asymptotically, `beta -> +inf` considers only recall, and `beta -> 0`
    only precision.

    The formula for F-beta score is:

    .. math::

       F_\\beta = \\frac{(1 + \\beta^2) \\text{tp}}
                        {(1 + \\beta^2) \\text{tp} + \\text{fp} + \\beta^2 \\text{fn}}

    Where :math:`\\text{tp}` is the number of true positives, :math:`\\text{fp}` is the
    number of false positives, and :math:`\\text{fn}` is the number of false negatives.

    Support beyond term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return
    F-beta score for `pos_label`. If `average` is not `'binary'`, `pos_label` is
    ignored and F-beta score for both classes are computed, then averaged or both
    returned (when `average=None`). Similarly, for :term:`multiclass` and
    :term:`multilabel` targets, F-beta score for all `labels` are either returned or
    averaged depending on the `average` parameter. Use `labels` specify the set of
    labels to calculate F-beta score for.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float
        Determines the weight of recall in the combined score.
    # labels参数：用于指定要包括在计算平均值时的类别标签集合。
    # 当average不等于'binary'时，labels参数指定要包含的标签，并确定它们的顺序。
    # 在多类分类中，可以排除数据中存在的标签，例如排除一个"负类"。
    # 对于多标签目标，labels是列索引。
    labels : array-like, default=None
        The set of labels to include when `average != 'binary'`, and their
        order if `average is None`. Labels present in the data can be
        excluded, for example in multiclass classification to exclude a "negative
        class". Labels not present in the data can be included and will be
        "assigned" 0 samples. For multilabel targets, labels are column indices.
        By default, all labels in `y_true` and `y_pred` are used in sorted order.

    # pos_label参数：当average='binary'且数据为二进制时，指定要报告的类别。
    # 对于多类或多标签目标，设置`labels=[pos_label]`并且`average != 'binary'`，
    # 可以报告仅一个标签的度量指标。
    pos_label : int, float, bool or str, default=1
        The class to report if `average='binary'` and the data is binary,
        otherwise this parameter is ignored.
        For multiclass or multilabel targets, set `labels=[pos_label]` and
        `average != 'binary'` to report metrics for one label only.

    # average参数：用于控制如何计算多类/多标签目标的平均值。
    # 如果为`None`，则返回每个类别的分数；否则，确定在数据上执行的平均类型。
    # 可选值包括：'binary', 'micro', 'macro', 'weighted', 'samples'。
    # 每个选项对应不同的计算方式，例如'binary'只报告由pos_label指定的类别的结果。
    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, \
            default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    # sample_weight参数：用于指定每个样本的权重。
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    # zero_division参数：设置当出现零除法（即所有预测和标签都为负时）时返回的值。
    # 可选值包括："warn", 0.0, 1.0, np.nan。
    # 如果设置为"warn"，则行为类似于0，但还会引发警告。
    # 如果设置为`np.nan`，这些值将从平均值中排除。
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative.

        Notes:
        - If set to "warn", this acts like 0, but a warning is also raised.
        - If set to `np.nan`, such values will be excluded from the average.

        .. versionadded:: 1.3
           `np.nan` option was added.

    # 返回：根据所计算的度量指标返回相应的结果。
    Returns
    -------
    # 计算 F-beta 分数并返回
    
    _, _, f, _ = precision_recall_fscore_support(
        y_true,                  # 真实标签
        y_pred,                  # 预测标签
        beta=beta,               # F-beta 中的 beta 参数
        labels=labels,           # 要评估的类标签
        pos_label=pos_label,     # 正类的标签值
        average=average,         # 平均模式（'binary', 'micro', 'macro', 'weighted' 或 None）
        warn_for=("f-score",),   # 当出现警告的情况下指定要警告的指标类型
        sample_weight=sample_weight,   # 样本权重
        zero_division=zero_division,   # 当分母为零时的处理方式
    )
    # 返回 F-score 的值
    return f
# 定义一个函数 `_prf_divide`，用于执行除法并处理除零情况
def _prf_divide(
    numerator, denominator, metric, modifier, average, warn_for, zero_division="warn"
):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0, 1 or np.nan (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    # 创建一个布尔掩码，标记分母为0的位置
    mask = denominator == 0.0
    # 复制分母数组，避免直接修改原数组
    denominator = denominator.copy()
    # 将分母数组中标记为0的位置设为1，避免产生无穷大或NaN
    denominator[mask] = 1  # avoid infs/nans
    # 执行除法操作，计算结果
    result = numerator / denominator

    # 如果没有任何分母为0的情况，则直接返回结果
    if not np.any(mask):
        return result

    # 将分母为0的位置的结果设置为 `zero_division_value`
    zero_division_value = _check_zero_division(zero_division)
    result[mask] = zero_division_value

    # 如果 `zero_division` 不是 "warn"，或者当前指标不在警告列表中，则直接返回结果
    if zero_division != "warn" or metric not in warn_for:
        return result

    # 构建相应的警告信息
    if metric in warn_for:
        _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))

    return result


# 定义函数 `_warn_prf`，用于生成性能评估相关的警告信息
def _warn_prf(average, modifier, msg_start, result_size):
    axis0, axis1 = "sample", "label"
    # 根据 `average` 参数调整警告信息中的轴标签顺序
    if average == "samples":
        axis0, axis1 = axis1, axis0
    # 构建警告消息的格式字符串
    msg = (
        "{0} ill-defined and being set to 0.0 {{0}} "
        "no {1} {2}s. Use `zero_division` parameter to control"
        " this behavior.".format(msg_start, modifier, axis0)
    )
    # 根据 `result_size` 的值进行进一步格式化警告消息
    if result_size == 1:
        msg = msg.format("due to")
    else:
        msg = msg.format("in {0}s with".format(axis1))
    # 发出警告
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)


# 定义函数 `_check_set_wise_labels`，用于验证与集合指标相关的标签
def _check_set_wise_labels(y_true, y_pred, average, labels, pos_label):
    """Validation associated with set-wise metrics.

    Returns identified labels.
    """
    # 支持的平均选项列表
    average_options = (None, "micro", "macro", "weighted", "samples")
    # 如果 `average` 参数不在支持的选项中且不是 "binary"，则引发 ValueError 异常
    if average not in average_options and average != "binary":
        raise ValueError("average has to be one of " + str(average_options))

    # 检查目标类型，并确保 `y_true` 和 `y_pred` 都符合规范
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    # 将标签转换为 Python 原始类型，以避免 NumPy 类型和 Python 字符串的比较问题
    # 参见 https://github.com/numpy/numpy/issues/6784
    # 获取唯一的标签列表
    present_labels = unique_labels(y_true, y_pred).tolist()
    # 如果平均值设定为 "binary"
    if average == "binary":
        # 如果目标值类型为 "binary"
        if y_type == "binary":
            # 如果指定的正类标签不在当前存在的标签列表中
            if pos_label not in present_labels:
                # 如果当前存在的标签数量大于等于2，则引发值错误异常
                if len(present_labels) >= 2:
                    raise ValueError(
                        f"pos_label={pos_label} is not a valid label. It "
                        f"should be one of {present_labels}"
                    )
            # 将标签列表设定为只包含指定的正类标签
            labels = [pos_label]
        else:
            # 如果目标值类型为 "multiclass"
            # 转换平均选项为列表，然后移除 "samples"
            average_options = list(average_options)
            average_options.remove("samples")
            # 引发值错误异常，指示目标是多类别，但平均值设定为 "binary"
            raise ValueError(
                "Target is %s but average='binary'. Please "
                "choose another average setting, one of %r." % (y_type, average_options)
            )
    # 否则，如果指定的正类标签不是 None 或 1
    elif pos_label not in (None, 1):
        # 发出警告，指示在平均值不为 "binary" 时，忽略了正类标签设定
        warnings.warn(
            "Note that pos_label (set to %r) is ignored when "
            "average != 'binary' (got %r). You may use "
            "labels=[pos_label] to specify a single positive class."
            % (pos_label, average),
            UserWarning,
        )
    # 返回最终的标签列表
    return labels
# 使用 @validate_params 装饰器对 precision_recall_fscore_support 函数进行参数验证和类型检查
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数可以是 array-like 或者 sparse matrix 类型
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数可以是 array-like 或者 sparse matrix 类型
        "beta": [Interval(Real, 0.0, None, closed="both")],  # beta 参数必须是大于等于 0.0 的实数
        "labels": ["array-like", None],  # labels 参数可以是 array-like 类型或者 None
        "pos_label": [Real, str, "boolean", None],  # pos_label 参数可以是实数、字符串、布尔值或者 None
        "average": [
            StrOptions({"micro", "macro", "samples", "weighted", "binary"}),  # average 参数必须是指定集合中的字符串
            None,
        ],  # 或者可以为 None
        "warn_for": [list, tuple, set],  # warn_for 参数可以是 list、tuple 或者 set
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是 array-like 类型或者 None
        "zero_division": [
            Options(Real, {0.0, 1.0}),  # zero_division 参数可以是 0.0 或者 1.0
            "nan",  # 或者可以是字符串 "nan"
            StrOptions({"warn"}),  # 或者可以是字符串 "warn"
        ],
    },
    prefer_skip_nested_validation=True,  # 设置 prefer_skip_nested_validation 参数为 True
)
# 定义 precision_recall_fscore_support 函数，计算每个类别的精确率、召回率、F-score 和支持度
def precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    beta=1.0,  # 默认 beta 参数为 1.0
    labels=None,  # labels 参数默认为 None
    pos_label=1,  # 默认 pos_label 参数为 1
    average=None,  # 默认 average 参数为 None
    warn_for=("precision", "recall", "f-score"),  # warn_for 参数默认包含 "precision", "recall", "f-score"
    sample_weight=None,  # sample_weight 参数默认为 None
    zero_division="warn",  # zero_division 参数默认为 "warn"
):
    """Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label a negative sample as
    positive.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    Support beyond term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return
    metrics for `pos_label`. If `average` is not `'binary'`, `pos_label` is ignored
    and metrics for both classes are computed, then averaged or both returned (when
    `average=None`). Similarly, for :term:`multiclass` and :term:`multilabel` targets,
    metrics for all `labels` are either returned or averaged depending on the `average`
    parameter. Use `labels` specify the set of labels to calculate metrics for.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float, default=1.0
        The strength of recall versus precision in the F-score.
    # labels: array-like, default=None
    #     要包含的标签集，当 `average != 'binary'` 时使用，以及它们的顺序如果 `average is None`。
    #     可以排除数据中存在的标签，例如在多类分类中排除“负类”。
    #     不在数据中的标签可以包括，并将被“分配”0个样本。对于多标签目标，标签是列索引。
    #     默认情况下，`y_true` 和 `y_pred` 中的所有标签都按排序顺序使用。

    # pos_label: int, float, bool or str, default=1
    #     当 `average='binary'` 并且数据是二进制时报告的类别，否则此参数将被忽略。
    #     对于多类别或多标签目标，设置 `labels=[pos_label]` 并且 `average != 'binary'`，以报告仅一个标签的指标。

    # average: {'binary', 'micro', 'macro', 'samples', 'weighted'}, \
    #         default=None
    #     如果 ``None``，则返回每个类别的指标。否则，确定在数据上执行的平均类型：
    #     
    #     ``'binary'``:
    #         仅报告由 ``pos_label`` 指定的类别的结果。
    #         仅当目标 (`y_{true,pred}`) 是二进制时适用。
    #     ``'micro'``:
    #         通过计算总的真正例、假负例和假正例来全局计算指标。
    #     ``'macro'``:
    #         为每个标签计算指标，并找到它们的未加权平均值。
    #         这不考虑标签不平衡。
    #     ``'weighted'``:
    #         为每个标签计算指标，并找到它们的加权平均值，权重由支持度（每个标签的真实实例数）决定。
    #         这改变了 'macro' 来考虑标签不平衡；它可能导致不在精确率和召回率之间的 F-分数。
    #     ``'samples'``:
    #         为每个实例计算指标，并找到它们的平均值（仅在多标签分类中有意义，这与 :func:`accuracy_score` 不同）。

    # warn_for: list, tuple or set, for internal use
    #     确定在此函数用于返回其指标之一时将发出哪些警告。

    # sample_weight: array-like of shape (n_samples,), default=None
    #     样本权重。

    # zero_division: {"warn", 0.0, 1.0, np.nan}, default="warn"
    #     设置在发生零除法时返回的值：
    #        - 召回率：当没有正标签时
    #        - 精确率：当没有正预测时
    #        - F-分数：两者都是
    #     
    #     注意：
    #     - 如果设置为 "warn"，则其行为类似于 0，但还会引发警告。
    #     - 如果设置为 `np.nan`，这些值将从平均中排除。
    #     
    #     .. versionadded:: 1.3
    #        添加了 `np.nan` 选项。
    """
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        精确率得分。

    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        召回率得分。

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-beta 分数。

    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        ``y_true`` 中每个标签的出现次数。

    Notes
    -----
    当 ``true positive + false positive == 0`` 时，精确率未定义。
    当 ``true positive + false negative == 0`` 时，召回率未定义。
    当 ``true positive + false negative + false positive == 0`` 时，F 分数未定义。
    在这些情况下，默认情况下度量值将被设置为 0，并引发 ``UndefinedMetricWarning``。
    可以使用 ``zero_division`` 修改此行为。

    References
    ----------
    .. [1] `Precision and recall的维基百科条目
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

    .. [2] `F1-score的维基百科条目
           <https://en.wikipedia.org/wiki/F1_score>`_.

    .. [3] `多标签分类中的判别方法 Advances in Knowledge Discovery and Data Mining (2004),
           pp. 22-30 由 Shantanu Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    (0.22..., 0.33..., 0.26..., None)

    可以计算每个标签的精确率、召回率、F1 分数和支持度，而不是进行平均计算：

    >>> precision_recall_fscore_support(y_true, y_pred, average=None,
    ... labels=['pig', 'dog', 'cat'])
    (array([0.        , 0.        , 0.66...]),
     array([0., 0., 1.]), array([0. , 0. , 0.8]),
     array([2, 2, 2]))
    """
    _check_zero_division(zero_division)
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

    # 计算 tp_sum, pred_sum, true_sum ###
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )
    tp_sum = MCM[:, 1, 1]  # 计算每个标签的真正例数量
    pred_sum = tp_sum + MCM[:, 0, 1]  # 计算每个标签的预测总数（真正例 + 假正例）
    true_sum = tp_sum + MCM[:, 1, 0]  # 计算每个标签的真实总数（真正例 + 假反例）
    # 如果计算的平均方式是"micro"，则将各统计量转换为单元素的数组
    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # 计算 beta 的平方
    beta2 = beta**2

    # 调用 _prf_divide 函数计算精确率（precision）
    precision = _prf_divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
    )

    # 调用 _prf_divide 函数计算召回率（recall）
    recall = _prf_divide(
        tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
    )

    # 根据 beta 的值计算 F-score
    if np.isposinf(beta):
        f_score = recall
    elif beta == 0:
        f_score = precision
    else:
        # 根据混淆矩阵的条目计算 F-score
        denom = beta2 * true_sum + pred_sum
        f_score = _prf_divide(
            (1 + beta2) * tp_sum,
            denom,
            "f-score",
            "true nor predicted",
            average,
            warn_for,
            zero_division,
        )

    # 根据计算平均方式选择权重
    if average == "weighted":
        weights = true_sum
    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    # 如果指定了平均方式，对精确率、召回率和 F-score 进行加权平均
    if average is not None:
        # 对于平均方式为"binary"，确保精确率只有一个元素
        assert average != "binary" or len(precision) == 1
        precision = _nanaverage(precision, weights=weights)
        recall = _nanaverage(recall, weights=weights)
        f_score = _nanaverage(f_score, weights=weights)
        true_sum = None  # 不返回支持量（support）

    # 返回精确率、召回率、F-score 和支持量
    return precision, recall, f_score, true_sum
# 使用 @validate_params 装饰器验证函数输入参数的类型和条件
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数可以是类数组或稀疏矩阵
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数可以是类数组或稀疏矩阵
        "labels": ["array-like", None],             # labels 参数可以是类数组或空值
        "sample_weight": ["array-like", None],      # sample_weight 参数可以是类数组或空值
        "raise_warning": ["boolean"],               # raise_warning 参数必须是布尔值
    },
    prefer_skip_nested_validation=True,             # prefer_skip_nested_validation 设置为 True，优先跳过嵌套验证
)
# 定义函数 class_likelihood_ratios，计算二元分类的正类和负类似然比
def class_likelihood_ratios(
    y_true,                 # 真实目标值，可以是一维类数组或稀疏矩阵
    y_pred,                 # 预测目标值，可以是一维类数组或稀疏矩阵
    *,
    labels=None,            # 标签值，可以是类数组或空值，默认为 None
    sample_weight=None,     # 样本权重，可以是类数组或空值，默认为 None
    raise_warning=True,     # 是否提出警告，默认为 True
):
    """Compute binary classification positive and negative likelihood ratios.

    The positive likelihood ratio is `LR+ = sensitivity / (1 - specificity)`
    where the sensitivity or recall is the ratio `tp / (tp + fn)` and the
    specificity is `tn / (tn + fp)`. The negative likelihood ratio is `LR- = (1
    - sensitivity) / specificity`. Here `tp` is the number of true positives,
    `fp` the number of false positives, `tn` is the number of true negatives and
    `fn` the number of false negatives. Both class likelihood ratios can be used
    to obtain post-test probabilities given a pre-test probability.

    `LR+` ranges from 1 to infinity. A `LR+` of 1 indicates that the probability
    of predicting the positive class is the same for samples belonging to either
    class; therefore, the test is useless. The greater `LR+` is, the more a
    positive prediction is likely to be a true positive when compared with the
    pre-test probability. A value of `LR+` lower than 1 is invalid as it would
    indicate that the odds of a sample being a true positive decrease with
    respect to the pre-test odds.

    `LR-` ranges from 0 to 1. The closer it is to 0, the lower the probability
    of a given sample to be a false negative. A `LR-` of 1 means the test is
    useless because the odds of having the condition did not change after the
    test. A value of `LR-` greater than 1 invalidates the classifier as it
    indicates an increase in the odds of a sample belonging to the positive
    class after being classified as negative. This is the case when the
    classifier systematically predicts the opposite of the true label.

    A typical application in medicine is to identify the positive/negative class
    to the presence/absence of a disease, respectively; the classifier being a
    diagnostic test; the pre-test probability of an individual having the
    disease can be the prevalence of such disease (proportion of a particular
    population found to be affected by a medical condition); and the post-test
    probabilities would be the probability that the condition is truly present
    given a positive test result.

    Read more in the :ref:`User Guide <class_likelihood_ratios>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    # 检查目标向量的类型，并确保是二元分类问题
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type != "binary":
        # 如果不是二元分类问题，抛出数值错误
        raise ValueError(
            "class_likelihood_ratios only supports binary classification "
            f"problems, got targets of type: {y_type}"
        )

    # 计算混淆矩阵
    cm = confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
    )

    # 处理当 `y_true` 只包含单一类别且 `y_true == y_pred` 的情况
    # 这可能发生在交叉验证不平衡数据时，不应被解释为完美的得分。
    # 检查混淆矩阵是否为 (1, 1)，即只包含一个类别的样本
    if cm.shape == (1, 1):
        # 提示信息，说明只在测试期间看到了一个类别的样本
        msg = "samples of only one class were seen during testing "
        # 如果设置了警告标志，发出用户警告
        if raise_warning:
            warnings.warn(msg, UserWarning, stacklevel=2)
        # 正向似然比和负向似然比设置为 NaN
        positive_likelihood_ratio = np.nan
        negative_likelihood_ratio = np.nan
    else:
        # 展开混淆矩阵的值
        tn, fp, fn, tp = cm.ravel()
        # 计算正类和负类的支持数
        support_pos = tp + fn
        support_neg = tn + fp
        # 计算正向似然比的分子和分母
        pos_num = tp * support_neg
        pos_denom = fp * support_pos
        # 计算负向似然比的分子和分母
        neg_num = fn * support_neg
        neg_denom = tn * support_pos

        # 如果正类支持数为 0，则发出警告并将得分设置为 NaN
        if support_pos == 0:
            msg = "no samples of the positive class were present in the testing set "
            if raise_warning:
                warnings.warn(msg, UserWarning, stacklevel=2)
            positive_likelihood_ratio = np.nan
            negative_likelihood_ratio = np.nan
        # 如果假阳性为 0，则发出警告并将正向似然比设置为 NaN
        if fp == 0:
            if tp == 0:
                msg = "no samples predicted for the positive class"
            else:
                msg = "positive_likelihood_ratio ill-defined and being set to nan "
            if raise_warning:
                warnings.warn(msg, UserWarning, stacklevel=2)
            positive_likelihood_ratio = np.nan
        else:
            positive_likelihood_ratio = pos_num / pos_denom
        # 如果真阴性为 0，则发出警告并将负向似然比设置为 NaN
        if tn == 0:
            msg = "negative_likelihood_ratio ill-defined and being set to nan "
            if raise_warning:
                warnings.warn(msg, UserWarning, stacklevel=2)
            negative_likelihood_ratio = np.nan
        else:
            negative_likelihood_ratio = neg_num / neg_denom

    # 返回正向似然比和负向似然比
    return positive_likelihood_ratio, negative_likelihood_ratio
# 使用 @validate_params 装饰器对 precision_score 函数的参数进行验证和规范化
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数可以是类数组或稀疏矩阵
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数可以是类数组或稀疏矩阵
        "labels": ["array-like", None],  # labels 参数可以是类数组，也可以为 None
        "pos_label": [Real, str, "boolean", None],  # pos_label 参数可以是实数、字符串、布尔值或 None
        "average": [  # average 参数可以是字符串选项集合 {"micro", "macro", "samples", "weighted", "binary"} 或 None
            StrOptions({"micro", "macro", "samples", "weighted", "binary"}),
            None,
        ],
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是类数组或 None
        "zero_division": [  # zero_division 参数可以是实数选项集合 {0.0, 1.0}、"nan" 或字符串选项集合 {"warn"}
            Options(Real, {0.0, 1.0}),
            "nan",
            StrOptions({"warn"}),
        ],
    },
    prefer_skip_nested_validation=True,  # 设置 prefer_skip_nested_validation 参数为 True，优先跳过嵌套验证
)
def precision_score(
    y_true,
    y_pred,
    *,
    labels=None,  # labels 参数默认为 None
    pos_label=1,  # pos_label 参数默认为 1
    average="binary",  # average 参数默认为 "binary"
    sample_weight=None,  # sample_weight 参数默认为 None
    zero_division="warn",  # zero_division 参数默认为 "warn"
):
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Support beyond term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return
    precision for `pos_label`. If `average` is not `'binary'`, `pos_label` is ignored
    and precision for both classes are computed, then averaged or both returned (when
    `average=None`). Similarly, for :term:`multiclass` and :term:`multilabel` targets,
    precision for all `labels` are either returned or averaged depending on the
    `average` parameter. Use `labels` specify the set of labels to calculate precision
    for.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        The set of labels to include when `average != 'binary'`, and their
        order if `average is None`. Labels present in the data can be
        excluded, for example in multiclass classification to exclude a "negative
        class". Labels not present in the data can be included and will be
        "assigned" 0 samples. For multilabel targets, labels are column indices.
        By default, all labels in `y_true` and `y_pred` are used in sorted order.

        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.
    pos_label : int, float, bool or str, default=1
        # 正类标签：int、float、bool或str，默认为1
        如果 `average='binary'` 并且数据是二分类的，则报告此类。
        否则，此参数将被忽略。
        对于多类别或多标签目标，请设置 `labels=[pos_label]` 和 `average != 'binary'`，
        以报告仅一个标签的指标。

    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, \
            default='binary'
        # 平均类型：{'micro', 'macro', 'samples', 'weighted', 'binary'}或None，默认='binary'
        此参数对多类别/多标签目标是必需的。
        如果 ``None``，则返回每个类别的得分。否则，此参数确定在数据上执行的平均类型：

        ``'binary'``:
            # 二分类：
            仅报告由 ``pos_label`` 指定的类别的结果。
            仅当目标 (``y_{true,pred}``) 是二分类时适用。
        ``'micro'``:
            # 微平均：
            通过计算总的真正例、假反例和假正例来计算全局指标。
        ``'macro'``:
            # 宏平均：
            为每个标签计算指标，并找到它们的未加权平均值。不考虑标签不平衡。
        ``'weighted'``:
            # 加权平均：
            为每个标签计算指标，并找到其支持加权的平均值
            （每个标签的真实实例数）。这会修改 'macro' 以考虑标签不平衡；
            可能导致 F-score 不处于精确率和召回率之间。
        ``'samples'``:
            # 样本平均：
            计算每个实例的指标，并找到它们的平均值
            （仅在多标签分类中有意义，在这种情况下与 :func:`accuracy_score` 不同）。

    sample_weight : array-like of shape (n_samples,), default=None
        # 样本权重：形状为 (n_samples,) 的数组，默认为 None
        样本权重。

    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        # 零除法处理：{"warn", 0.0, 1.0, np.nan}，默认为 "warn"
        设置零除法时返回的值。

        Notes:
        - 如果设置为 "warn"，则表现为0，但还会引发警告。
        - 如果设置为 `np.nan`，这样的值将从平均值中排除。

        .. versionadded:: 1.3
           添加了 `np.nan` 选项。

    Returns
    -------
    precision : float (if average is not None) or array of float of shape \
                (n_unique_labels,)
        # 精确率：如果 average 不是 None，则为 float 或形状为 (n_unique_labels,) 的浮点数数组
        二分类中正类的精确率或多类任务中每个类别精确率的加权平均值。

    See Also
    --------
    precision_recall_fscore_support : 计算每个类别的精确率、召回率、F1值和支持度。
    recall_score : 计算 ``tp / (tp + fn)`` 的比率，其中 ``tp`` 是真正例，``fn`` 是假反例。
    PrecisionRecallDisplay.from_estimator : 给定估算器和一些数据，绘制精确率-召回率曲线。
    PrecisionRecallDisplay.from_predictions : 给定二分类预测，绘制精确率-召回率曲线。
    # 计算多标签分类的精确度（Precision）。
    multilabel_confusion_matrix : Compute a confusion matrix for each class or
        sample.
    
    Notes
    -----
    当 ``true positive + false positive == 0`` 时，精确度为 0，并且会引发 ``UndefinedMetricWarning``。可以通过 ``zero_division`` 参数修改这种行为。
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> precision_score(y_true, y_pred, average='macro')
    0.22...
    >>> precision_score(y_true, y_pred, average='micro')
    0.33...
    >>> precision_score(y_true, y_pred, average='weighted')
    0.22...
    >>> precision_score(y_true, y_pred, average=None)
    array([0.66..., 0.        , 0.        ])
    >>> y_pred = [0, 0, 0, 0, 0, 0]
    >>> precision_score(y_true, y_pred, average=None)
    array([0.33..., 0.        , 0.        ])
    >>> precision_score(y_true, y_pred, average=None, zero_division=1)
    array([0.33..., 1.        , 1.        ])
    >>> precision_score(y_true, y_pred, average=None, zero_division=np.nan)
    array([0.33...,        nan,        nan])
    
    >>> # multilabel classification
    >>> y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
    >>> y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
    >>> precision_score(y_true, y_pred, average=None)
    array([0.5, 1. , 1. ])
    """
    # 使用 precision_recall_fscore_support 函数计算精确度（Precision）
    p, _, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("precision",),
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    # 返回计算得到的精确度数组
    return p
# 使用 @validate_params 装饰器验证参数，确保参数满足指定的类型要求
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数应为数组或稀疏矩阵
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数应为数组或稀疏矩阵
        "labels": ["array-like", None],             # labels 参数应为数组或为空
        "pos_label": [Real, str, "boolean", None],  # pos_label 参数可以是实数、字符串、布尔值或为空
        "average": [                                # average 参数可以是指定字符串集合中的一个或为空
            StrOptions({"micro", "macro", "samples", "weighted", "binary"}),
            None,
        ],
        "sample_weight": ["array-like", None],      # sample_weight 参数应为数组或为空
        "zero_division": [                          # zero_division 参数可以是实数、指定字符串集合中的一个或 "nan"
            Options(Real, {0.0, 1.0}),
            "nan",
            StrOptions({"warn"}),
        ],
    },
    prefer_skip_nested_validation=True,             # 设置 prefer_skip_nested_validation 参数为 True
)
def recall_score(
    y_true,
    y_pred,
    *,
    labels=None,          # labels 参数默认为 None
    pos_label=1,          # pos_label 参数默认为 1
    average="binary",     # average 参数默认为 "binary"
    sample_weight=None,   # sample_weight 参数默认为 None
    zero_division="warn", # zero_division 参数默认为 "warn"
):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Support beyond term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return
    recall for `pos_label`. If `average` is not `'binary'`, `pos_label` is ignored
    and recall for both classes are computed then averaged or both returned (when
    `average=None`). Similarly, for :term:`multiclass` and :term:`multilabel` targets,
    recall for all `labels` are either returned or averaged depending on the `average`
    parameter. Use `labels` specify the set of labels to calculate recall for.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        The set of labels to include when `average != 'binary'`, and their
        order if `average is None`. Labels present in the data can be
        excluded, for example in multiclass classification to exclude a "negative
        class". Labels not present in the data can be included and will be
        "assigned" 0 samples. For multilabel targets, labels are column indices.
        By default, all labels in `y_true` and `y_pred` are used in sorted order.

        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.

    pos_label : int, float, bool or str, default=1
        The class to report if `average='binary'` and the data is binary,
        otherwise this parameter is ignored.
        For multiclass or multilabel targets, set `labels=[pos_label]` and
        `average != 'binary'` to report metrics for one label only.
    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, \
            default='binary'
        # 平均类型参数，可选值为 'micro', 'macro', 'samples', 'weighted', 'binary' 或 None，默认为 'binary'
        This parameter is required for multiclass/multilabel targets.
        # 这个参数在多类/多标签目标任务中是必需的
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        # 如果为 ``'binary'``，只报告由 ``pos_label`` 指定的类别的结果。仅当目标（``y_{true,pred}``）是二进制时适用。
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        # ``'micro'``：通过统计总的真正例、假反例和假正例来全局计算度量指标。
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        # ``'macro'``：为每个标签计算度量指标，并找到它们的未加权平均值。这不考虑标签不平衡问题。
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall. Weighted recall
            is equal to accuracy.
        # ``'weighted'``：为每个标签计算度量指标，并通过支持度（每个标签的真实实例数）加权平均。这种方式修改了 'macro'，以解决标签不平衡问题；可能导致 F 分数不在精确率和召回率之间。加权召回率等于准确率。
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
        # ``'samples'``：为每个实例计算度量指标，并找到它们的平均值（仅在多标签分类中有意义，与 :func:`accuracy_score` 不同）。

    sample_weight : array-like of shape (n_samples,), default=None
        # 样本权重：形状为 (n_samples,) 的类数组，默认为 None

    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        # 零除问题处理参数：{"warn", 0.0, 1.0, np.nan}，默认为 "warn"
        Sets the value to return when there is a zero division.
        # 在发生零除时设置返回的值

        Notes:
        # 注意事项：
        - If set to "warn", this acts like 0, but a warning is also raised.
        # 如果设置为 "warn"，它的行为类似于 0，但还会发出警告。
        - If set to `np.nan`, such values will be excluded from the average.
        # 如果设置为 `np.nan`，这样的值将被排除在平均值之外。

        .. versionadded:: 1.3
           `np.nan` option was added.
        # 版本说明：新增了 `np.nan` 选项。

    Returns
    -------
    recall : float (if average is not None) or array of float of shape \
             (n_unique_labels,)
        # 返回：召回率，如果平均类型不为 None，则为浮点数，否则为形状为 (n_unique_labels,) 的浮点数数组。
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.
        # 二元分类中正类的召回率或多类任务中每个类别召回率的加权平均。

    See Also
    --------
    precision_recall_fscore_support : Compute precision, recall, F-measure and
        support for each class.
    precision_score : Compute the ratio ``tp / (tp + fp)`` where ``tp`` is the
        number of true positives and ``fp`` the number of false positives.
    balanced_accuracy_score : Compute balanced accuracy to deal with imbalanced
        datasets.
    multilabel_confusion_matrix : Compute a confusion matrix for each class or
        sample.
    PrecisionRecallDisplay.from_estimator : Plot precision-recall curve given
        an estimator and some data.
    PrecisionRecallDisplay.from_predictions : Plot precision-recall curve given
        binary class predictions.

    Notes
    -----
    When ``true positive + false negative == 0``, recall returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be modified with
    # 当真正例 + 假反例 == 0 时，召回率返回 0 并引发 ``UndefinedMetricWarning``。可以通过以下方式修改这种行为：
    # Calculate recall scores for each class or label in multiclass/multilabel classification.
    # This function computes the recall score for each class individually, returning an array of scores.
    
    _, r, _, _ = precision_recall_fscore_support(
        y_true,               # True labels for the data
        y_pred,               # Predicted labels
        labels=labels,        # Specific labels to include in the computation (default is all)
        pos_label=pos_label,  # Label considered as positive in binary/multiclass (default is 1)
        average=average,      # Strategy for averaging recall scores ('binary', 'micro', 'macro', 'weighted', 'samples', 'multiclass', or None)
        warn_for=("recall",), # Types of warnings to generate (here, only for recall)
        sample_weight=sample_weight,  # Array of weights to apply to individual samples (optional)
        zero_division=zero_division,  # Value to be used for recall scores when a denominator is zero
    )
    # Return the recall scores as computed by the precision_recall_fscore_support function
    return r
# 使用装饰器 @validate_params 对函数参数进行验证，确保参数满足以下要求
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应为类数组
        "y_pred": ["array-like"],  # y_pred 参数应为类数组
        "sample_weight": ["array-like", None],  # sample_weight 参数可以为类数组或者为 None
        "adjusted": ["boolean"],  # adjusted 参数应为布尔类型
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False):
    """Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.

    The best value is 1 and the worst value is 0 when ``adjusted=False``.

    Read more in the :ref:`User Guide <balanced_accuracy_score>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, while keeping perfect performance at a score
        of 1.

    Returns
    -------
    balanced_accuracy : float
        Balanced accuracy score.

    See Also
    --------
    average_precision_score : Compute average precision (AP) from prediction
        scores.
    precision_score : Compute the precision score.
    recall_score : Compute the recall score.
    roc_auc_score : Compute Area Under the Receiver Operating Characteristic
        Curve (ROC AUC) from prediction scores.

    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy. Our
    definition is equivalent to :func:`accuracy_score` with class-balanced
    sample weights, and shares desirable properties with the binary case.
    See the :ref:`User Guide <balanced_accuracy_score>`.

    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.

    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    # 计算混淆矩阵
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    # 忽略除以零的警告，并设置为无效值
    with np.errstate(divide="ignore", invalid="ignore"):
        # 计算每个类别的召回率
        per_class = np.diag(C) / C.sum(axis=1)
    # 如果 per_class 中存在 NaN 值，则发出警告信息
    if np.any(np.isnan(per_class)):
        warnings.warn("y_pred contains classes not in y_true")
        # 过滤掉 per_class 中的 NaN 值
        per_class = per_class[~np.isnan(per_class)]
    
    # 计算 per_class 数组的平均值作为 score
    score = np.mean(per_class)
    
    # 如果需要进行调整
    if adjusted:
        # 计算类别数量
        n_classes = len(per_class)
        # 计算随机猜测的概率
        chance = 1 / n_classes
        # 对 score 进行调整，以减去随机猜测的影响并进行标准化
        score -= chance
        score /= 1 - chance
    
    # 返回最终的得分值
    return score
# 应用装饰器，验证和规范化 classification_report 函数的参数
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数应为数组或稀疏矩阵
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数应为数组或稀疏矩阵
        "labels": ["array-like", None],  # labels 参数可选，为数组
        "target_names": ["array-like", None],  # target_names 参数可选，为数组
        "sample_weight": ["array-like", None],  # sample_weight 参数可选，为数组
        "digits": [Interval(Integral, 0, None, closed="left")],  # digits 参数为非负整数
        "output_dict": ["boolean"],  # output_dict 参数为布尔值
        "zero_division": [
            Options(Real, {0.0, 1.0}),  # zero_division 参数为实数，可选值为 0.0 或 1.0
            "nan",  # zero_division 参数可选值为 "nan"
            StrOptions({"warn"}),  # zero_division 参数可选值为 "warn"
        ],
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
# 定义分类报告函数
def classification_report(
    y_true,
    y_pred,
    *,
    labels=None,  # 标签列表，默认为 None
    target_names=None,  # 标签显示名称列表，默认为 None
    sample_weight=None,  # 样本权重，默认为 None
    digits=2,  # 输出浮点值的小数位数，默认为 2
    output_dict=False,  # 是否返回字典形式的输出，默认为 False
    zero_division="warn",  # 零除警告行为，默认为 "warn"
):
    """Build a text report showing the main classification metrics.

    Read more in the :ref:`User Guide <classification_report>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_labels,), default=None
        Optional list of label indices to include in the report.

    target_names : array-like of shape (n_labels,), default=None
        Optional display names matching the labels (same order).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    digits : int, default=2
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.

    output_dict : bool, default=False
        If True, return output as dict.

        .. versionadded:: 0.20

    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

        .. versionadded:: 1.3
           `np.nan` option was added.

    Returns
    -------
    ```
    # 检查目标变量的类型，并确保它们符合预期的格式
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if labels is None:
        # 如果未提供 labels 参数，则通过 y_true 和 y_pred 计算唯一的标签集合
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        # 如果提供了 labels 参数，则将其转换为 NumPy 数组，并标记 labels_given 为 True
        labels = np.asarray(labels)
        labels_given = True

    # 计算微观平均值的准确性
    micro_is_accuracy = (y_type == "multiclass" or y_type == "binary") and (
        not labels_given or (set(labels) >= set(unique_labels(y_true, y_pred)))
    )

    if target_names is not None and len(labels) != len(target_names):
        # 如果 target_names 参数不为空且其长度与 labels 不相等
        if labels_given:
            # 如果已提供 labels，则发出警告
            warnings.warn(
                "labels size, {0}, does not match size of target_names, {1}".format(
                    len(labels), len(target_names)
                )
            )
        else:
            # 如果未提供 labels，则引发 ValueError
            raise ValueError(
                "Number of classes, {0}, does not match size of "
                "target_names, {1}. Try specifying the labels "
                "parameter".format(len(labels), len(target_names))
            )

    if target_names is None:
        # 如果 target_names 为空，则使用 labels 的字符串表示作为 target_names
        target_names = ["%s" % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    # 计算每个类别的结果，不进行平均
    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        # 如果 y_type 是多标签类型，则选择的平均选项为 ("micro", "macro", "weighted", "samples")
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        # 否则，选择的平均选项为 ("micro", "macro", "weighted")
        average_options = ("micro", "macro", "weighted")

    if output_dict:
        # 如果输出为字典形式
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            # 将得分组成字典，每个标签对应一个字典，包含各项指标的分数
            report_dict[label] = dict(zip(headers, [float(i) for i in scores]))
    else:
        # 否则，输出为字符串形式
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        for row in rows:
            # 格式化每一行的输出，包括类别名称和各项指标的数值
            report += row_fmt.format(*row, width=width, digits=digits)
        report += "\n"

    # 计算所有适用的平均值
    for average in average_options:
        # 检查平均选项，确定行标题
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        # 使用指定的平均方法计算精确度、召回率、F1 值
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]

        # 如果输出为字典，则将结果以键值对形式存入字典
        if output_dict:
            report_dict[line_heading] = dict(zip(headers, [float(i) for i in avg]))
        else:
            # 如果行标题为 "accuracy"，使用特定格式输出
            if line_heading == "accuracy":
                row_fmt_accuracy = (
                    "{:>{width}s} "
                    + " {:>9.{digits}}" * 2
                    + " {:>9.{digits}f}"
                    + " {:>9}\n"
                )
                report += row_fmt_accuracy.format(
                    line_heading, "", "", *avg[2:], width=width, digits=digits
                )
            else:
                # 否则使用通用格式输出
                report += row_fmt.format(line_heading, *avg, width=width, digits=digits)

    # 如果输出为字典，处理精度数据
    if output_dict:
        if "accuracy" in report_dict.keys():
            report_dict["accuracy"] = report_dict["accuracy"]["precision"]
        return report_dict
    else:
        # 否则返回格式化后的报告字符串
        return report
# 使用装饰器验证参数，确保函数接收到正确类型和格式的输入数据
@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],  # y_true 参数应为类数组或稀疏矩阵
        "y_pred": ["array-like", "sparse matrix"],  # y_pred 参数应为类数组或稀疏矩阵
        "sample_weight": ["array-like", None],      # sample_weight 参数应为类数组或为空
    },
    prefer_skip_nested_validation=True,  # 设置跳过嵌套验证以提高效率
)
def hamming_loss(y_true, y_pred, *, sample_weight=None):
    """Compute the average Hamming loss.

    The Hamming loss is the fraction of labels that are incorrectly predicted.

    Read more in the :ref:`User Guide <hamming_loss>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

        .. versionadded:: 0.18

    Returns
    -------
    loss : float or int
        Return the average Hamming loss between element of ``y_true`` and
        ``y_pred``.

    See Also
    --------
    accuracy_score : Compute the accuracy score. By default, the function will
        return the fraction of correct predictions divided by the total number
        of predictions.
    jaccard_score : Compute the Jaccard similarity coefficient score.
    zero_one_loss : Compute the Zero-one classification loss. By default, the
        function will return the percentage of imperfectly predicted subsets.

    Notes
    -----
    In multiclass classification, the Hamming loss corresponds to the Hamming
    distance between ``y_true`` and ``y_pred`` which is equivalent to the
    subset ``zero_one_loss`` function, when `normalize` parameter is set to
    True.

    In multilabel classification, the Hamming loss is different from the
    subset zero-one loss. The zero-one loss considers the entire set of labels
    for a given sample incorrect if it does not entirely match the true set of
    labels. Hamming loss is more forgiving in that it penalizes only the
    individual labels.

    The Hamming loss is upperbounded by the subset zero-one loss, when
    `normalize` parameter is set to True. It is always between 0 and 1,
    lower being better.

    References
    ----------
    .. [1] Grigorios Tsoumakas, Ioannis Katakis. Multi-Label Classification:
           An Overview. International Journal of Data Warehousing & Mining,
           3(3), 1-13, July-September 2007.

    .. [2] `Wikipedia entry on the Hamming distance
           <https://en.wikipedia.org/wiki/Hamming_distance>`_.

    Examples
    --------
    >>> from sklearn.metrics import hamming_loss
    >>> y_pred = [1, 2, 3, 4]
    >>> y_true = [2, 2, 3, 4]
    >>> hamming_loss(y_true, y_pred)
    0.25

    In the multilabel case with binary label indicators:

    >>> import numpy as np
    >>> hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))
    0.75
    """

    # 检查目标值的类型，并确保 y_true 和 y_pred 都符合相同的类型
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    
    # 检查 y_true 和 y_pred 是否具有一致的长度，如果有 sample_weight，则也进行检查
    check_consistent_length(y_true, y_pred, sample_weight)
    # 如果样本权重为空，则默认权重平均值为1.0
    if sample_weight is None:
        weight_average = 1.0
    else:
        # 计算样本权重的平均值
        weight_average = np.mean(sample_weight)

    # 如果目标变量类型是多标签类型
    if y_type.startswith("multilabel"):
        # 计算预测与真实标签不同的数量
        n_differences = count_nonzero(y_true - y_pred, sample_weight=sample_weight)
        # 返回不同标签的比率，考虑样本数量、标签数量和权重平均值
        return n_differences / (y_true.shape[0] * y_true.shape[1] * weight_average)

    # 如果目标变量类型是二分类或多分类之一
    elif y_type in ["binary", "multiclass"]:
        # 计算分类错误的平均值，考虑样本权重
        return float(_average(y_true != y_pred, weights=sample_weight, normalize=True))
    else:
        # 抛出异常，指出不支持的目标变量类型
        raise ValueError("{0} is not supported".format(y_type))
@validate_params(
    {
        "y_true": ["array-like"],  # 参数验证装饰器，验证函数输入参数，确保 y_true 是类数组类型
        "y_pred": ["array-like"],  # 参数验证装饰器，验证函数输入参数，确保 y_pred 是类数组类型
        "normalize": ["boolean"],  # 参数验证装饰器，验证函数输入参数，确保 normalize 是布尔类型
        "sample_weight": ["array-like", None],  # 参数验证装饰器，验证函数输入参数，确保 sample_weight 是类数组类型或者 None
        "labels": ["array-like", None],  # 参数验证装饰器，验证函数输入参数，确保 labels 是类数组类型或者 None
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器选项，优先跳过嵌套验证
)
def log_loss(y_true, y_pred, *, normalize=True, sample_weight=None, labels=None):
    r"""Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``.
    The log loss is only defined for two or more labels.
    For a single sample with true label :math:`y \in \{0,1\}` and
    a probability estimate :math:`p = \operatorname{Pr}(y = 1)`, the log
    loss is:

    .. math::
        L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))

    Read more in the :ref:`User Guide <log_loss>`.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.

    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`~sklearn.preprocessing.LabelBinarizer`.

        `y_pred` values are clipped to `[eps, 1-eps]` where `eps` is the machine
        precision for `y_pred`'s dtype.

    normalize : bool, default=True
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like, default=None
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.

        .. versionadded:: 0.18

    Returns
    -------
    loss : float
        Log loss, aka logistic loss or cross-entropy loss.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.

    Examples
    --------
    >>> from sklearn.metrics import log_loss
    >>> log_loss(["spam", "ham", "ham", "spam"],
    ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
    0.21616...
    """
    y_pred = check_array(
        y_pred, ensure_2d=False, dtype=[np.float64, np.float32, np.float16]  # 使用 check_array 函数验证和转换 y_pred，确保其为指定的浮点数类型
    )

    check_consistent_length(y_pred, y_true, sample_weight)  # 使用 check_consistent_length 函数验证 y_pred, y_true 和 sample_weight 的长度一致性
    lb = LabelBinarizer()  # 创建 LabelBinarizer 对象，用于处理标签的二进制化

    if labels is not None:
        lb.fit(labels)  # 如果提供了 labels 参数，则用 labels 来拟合 LabelBinarizer
    # 如果y_true中只有一个标签，且labels参数未提供，则抛出异常
    else:
        lb.fit(y_true)

    # 如果lb对象中的类别数量为1
    if len(lb.classes_) == 1:
        if labels is None:
            # 抛出数值错误异常，说明y_true只包含一个标签，需要通过labels参数提供真实标签
            raise ValueError(
                "y_true contains only one label ({0}). Please "
                "provide the true labels explicitly through the "
                "labels argument.".format(lb.classes_[0])
            )
        else:
            # 抛出数值错误异常，说明y_true至少需要包含两个标签以计算log_loss，但当前只有一个标签
            raise ValueError(
                "The labels array needs to contain at least two "
                "labels for log_loss, "
                "got {0}.".format(lb.classes_)
            )

    # 将y_true转换为二进制形式
    transformed_labels = lb.transform(y_true)

    # 如果转换后的标签形状只有一列，即二进制形式
    if transformed_labels.shape[1] == 1:
        # 在第二列插入转换后标签的补集
        transformed_labels = np.append(
            1 - transformed_labels, transformed_labels, axis=1
        )

    # 如果y_pred的维度为1，假设y_true是二进制的
    if y_pred.ndim == 1:
        # 将y_pred转换为二维数组
        y_pred = y_pred[:, np.newaxis]
    # 如果y_pred的形状只有一列，即二进制形式
    if y_pred.shape[1] == 1:
        # 在第二列插入y_pred的补集
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # 计算y_pred数据类型的机器极小值
    eps = np.finfo(y_pred.dtype).eps

    # 确保y_pred归一化
    y_pred_sum = y_pred.sum(axis=1)
    # 如果y_pred的和不接近1，发出警告，建议传递概率值
    if not np.allclose(y_pred_sum, 1, rtol=np.sqrt(eps)):
        warnings.warn(
            "The y_pred values do not sum to one. Make sure to pass probabilities.",
            UserWarning,
        )

    # 将y_pred值限制在[eps, 1-eps]之间
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 检查维度是否一致
    transformed_labels = check_array(transformed_labels)
    # 如果lb对象中的类别数量与y_pred的列数不同
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            # 抛出数值错误异常，说明y_true和y_pred包含不同数量的类别
            raise ValueError(
                "y_true and y_pred contain different number of "
                "classes {0}, {1}. Please provide the true "
                "labels explicitly through the labels argument. "
                "Classes found in "
                "y_true: {2}".format(
                    transformed_labels.shape[1], y_pred.shape[1], lb.classes_
                )
            )
        else:
            # 抛出数值错误异常，说明labels中的类别数量与y_pred不同
            raise ValueError(
                "The number of classes in labels is different "
                "from that in y_pred. Classes found in "
                "labels: {0}".format(lb.classes_)
            )

    # 计算损失值，使用负对数似然函数（negative log-likelihood）
    loss = -xlogy(transformed_labels, y_pred).sum(axis=1)

    # 返回损失值的平均值
    return float(_average(loss, weights=sample_weight, normalize=normalize))
# 使用装饰器 validate_params 进行参数验证，确保函数接收到正确的参数类型和结构
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应该是类数组类型
        "pred_decision": ["array-like"],  # pred_decision 参数应该是类数组类型
        "labels": ["array-like", None],  # labels 参数可以是类数组类型或者 None
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是类数组类型或者 None
    },
    prefer_skip_nested_validation=True,  # 设置跳过嵌套验证以提高效率
)
def hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None):
    """Average hinge loss (non-regularized).

    In binary class case, assuming labels in y_true are encoded with +1 and -1,
    when a prediction mistake is made, ``margin = y_true * pred_decision`` is
    always negative (since the signs disagree), implying ``1 - margin`` is
    always greater than 1.  The cumulated hinge loss is therefore an upper
    bound of the number of mistakes made by the classifier.

    In multiclass case, the function expects that either all the labels are
    included in y_true or an optional labels argument is provided which
    contains all the labels. The multilabel margin is calculated according
    to Crammer-Singer's method. As in the binary case, the cumulated hinge loss
    is an upper bound of the number of mistakes made by the classifier.

    Read more in the :ref:`User Guide <hinge_loss>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target, consisting of integers of two values. The positive label
        must be greater than the negative label.

    pred_decision : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted decisions, as output by decision_function (floats).

    labels : array-like, default=None
        Contains all the labels for the problem. Used in multiclass hinge loss.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        Average hinge loss.

    References
    ----------
    .. [1] `Wikipedia entry on the Hinge loss
           <https://en.wikipedia.org/wiki/Hinge_loss>`_.

    .. [2] Koby Crammer, Yoram Singer. On the Algorithmic
           Implementation of Multiclass Kernel-based Vector
           Machines. Journal of Machine Learning Research 2,
           (2001), 265-292.

    .. [3] `L1 AND L2 Regularization for Multiclass Hinge Loss Models
           by Robert C. Moore, John DeNero
           <https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37362.pdf>`_.

    Examples
    --------
    >>> from sklearn import svm
    >>> from sklearn.metrics import hinge_loss
    >>> X = [[0], [1]]
    >>> y = [-1, 1]
    >>> est = svm.LinearSVC(random_state=0)
    >>> est.fit(X, y)
    LinearSVC(random_state=0)
    >>> pred_decision = est.decision_function([[-2], [3], [0.5]])
    >>> pred_decision
    array([-2.18...,  2.36...,  0.09...])
    >>> hinge_loss([-1, 1, 1], pred_decision)
    0.30...

    In the multiclass case:

    >>> import numpy as np
    >>> X = np.array([[0], [1], [2], [3]])
    >>> Y = np.array([0, 1, 2, 3])
    >>> labels = np.array([0, 1, 2, 3])
    >>> est = svm.LinearSVC()
    >>> est.fit(X, Y)
    LinearSVC()
    # 初始化一个线性支持向量机分类器，但未分配给任何变量，这行代码可能有误

    >>> pred_decision = est.decision_function([[-1], [2], [3]])
    # 使用已经训练好的模型 est 计算给定样本的决策函数值

    >>> y_true = [0, 2, 3]
    # 真实的目标标签，用于计算损失函数

    >>> hinge_loss(y_true, pred_decision, labels=labels)
    # 使用 Hinge 损失函数计算预测值与真实值之间的损失

    0.56...
    """
    check_consistent_length(y_true, pred_decision, sample_weight)
    # 检查输入数据的长度是否一致，以确保后续操作的有效性

    pred_decision = check_array(pred_decision, ensure_2d=False)
    # 将预测决策函数的输入数据转换为数组格式，并确保不要求二维数据

    y_true = column_or_1d(y_true)
    # 将真实标签转换为一维数组或一列数组

    y_true_unique = np.unique(labels if labels is not None else y_true)
    # 确定真实标签的唯一值集合，如果提供了标签，则使用标签；否则使用真实标签

    if y_true_unique.size > 2:
        if pred_decision.ndim <= 1:
            raise ValueError(
                "The shape of pred_decision cannot be 1d array"
                "with a multiclass target. pred_decision shape "
                "must be (n_samples, n_classes), that is "
                f"({y_true.shape[0]}, {y_true_unique.size})."
                f" Got: {pred_decision.shape}"
            )
        
        # pred_decision.ndim > 1 is true
        if y_true_unique.size != pred_decision.shape[1]:
            if labels is None:
                raise ValueError(
                    "Please include all labels in y_true "
                    "or pass labels as third argument"
                )
            else:
                raise ValueError(
                    "The shape of pred_decision is not "
                    "consistent with the number of classes. "
                    "With a multiclass target, pred_decision "
                    "shape must be "
                    "(n_samples, n_classes), that is "
                    f"({y_true.shape[0]}, {y_true_unique.size}). "
                    f"Got: {pred_decision.shape}"
                )
        if labels is None:
            labels = y_true_unique
        
        # 使用标签编码器将真实标签编码为数值形式
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        
        # 生成一个布尔掩码，用于指示每个预测决策的最大值
        mask = np.ones_like(pred_decision, dtype=bool)
        mask[np.arange(y_true.shape[0]), y_true] = False
        
        # 计算预测决策与最大预测决策之间的差值
        margin = pred_decision[~mask]
        margin -= np.max(pred_decision[mask].reshape(y_true.shape[0], -1), axis=1)

    else:
        # 处理二元分类情况
        # 假设正负标签分别编码为 +1 和 -1
        pred_decision = column_or_1d(pred_decision)
        pred_decision = np.ravel(pred_decision)
        
        # 标签二值化，只关注第一列
        lbin = LabelBinarizer(neg_label=-1)
        y_true = lbin.fit_transform(y_true)[:, 0]
        
        # 计算边缘损失
        try:
            margin = y_true * pred_decision
        except TypeError:
            raise TypeError("pred_decision should be an array of floats.")

    losses = 1 - margin
    # Hinge 损失函数不惩罚足够好的预测
    np.clip(losses, 0, None, out=losses)
    # 计算加权平均损失
    return np.average(losses, weights=sample_weight)
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应为类似数组的对象
        "y_proba": ["array-like", Hidden(None)],  # y_proba 参数可为类似数组的对象，隐藏类型为 None
        "sample_weight": ["array-like", None],  # sample_weight 参数可为类似数组的对象或者为 None
        "pos_label": [Real, str, "boolean", None],  # pos_label 参数可为实数、字符串、布尔值或者为 None
        "y_prob": ["array-like", Hidden(StrOptions({"deprecated"}))],  # y_prob 参数可为类似数组的对象，隐藏类型为包含字符串"deprecated"的选项
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证为 True
)
def brier_score_loss(
    y_true, y_proba=None, *, sample_weight=None, pos_label=None, y_prob="deprecated"
):
    """Compute the Brier score loss.

    The smaller the Brier score loss, the better, hence the naming with "loss".
    The Brier score measures the mean squared difference between the predicted
    probability and the actual outcome. The Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). It can be decomposed as the sum of refinement loss and
    calibration loss.

    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another). Which label is considered to be the positive
    label is controlled via the parameter `pos_label`, which defaults to
    the greater label unless `y_true` is all 0 or all -1, in which case
    `pos_label` defaults to 1.

    Read more in the :ref:`User Guide <brier_score_loss>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_proba : array-like of shape (n_samples,)
        Probabilities of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    pos_label : int, float, bool or str, default=None
        Label of the positive class. `pos_label` will be inferred in the
        following manner:

        * if `y_true` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `y_true` contains string, an error will be raised and
          `pos_label` should be explicitly specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(y_true)[-1]`.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

        .. deprecated:: 1.5
            `y_prob` is deprecated and will be removed in 1.7. Use
            `y_proba` instead.

    Returns
    -------
    score : float
        Brier score loss.

    References
    ----------
    .. [1] `Wikipedia entry for the Brier score
            <https://en.wikipedia.org/wiki/Brier_score>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
    # TODO(1.7): remove in 1.7 and reset y_proba to be required
    # Note: validate params will raise an error if y_prob is not array-like,
    # or "deprecated"
    如果 y_proba 不为空且 y_prob 不是字符串，则抛出 ValueError 错误，提示在 v1.5 中 y_prob 已经弃用，并将在 v1.7 中移除，建议仅使用 y_proba 参数。
    
    if y_proba is not None and not isinstance(y_prob, str):
        raise ValueError(
            "`y_prob` and `y_proba` cannot be both specified. Please use `y_proba` only"
            " as `y_prob` is deprecated in v1.5 and will be removed in v1.7."
        )
    
    # 如果 y_proba 为空，则发出警告，提示在 v1.5 中 y_prob 已经弃用，并将在 v1.7 中移除，建议使用 y_proba 代替。
    if y_proba is None:
        warnings.warn(
            (
                "y_prob was deprecated in version 1.5 and will be removed in 1.7."
                "Please use ``y_proba`` instead."
            ),
            FutureWarning,
        )
        y_proba = y_prob
    
    # 确保 y_true 和 y_proba 都是一维数组或者列向量
    y_true = column_or_1d(y_true)
    y_proba = column_or_1d(y_proba)
    
    # 确保 y_true 和 y_proba 中没有无穷大或者 NaN 值
    assert_all_finite(y_true)
    assert_all_finite(y_proba)
    
    # 检查 y_true, y_proba 和 sample_weight 的长度一致性
    check_consistent_length(y_true, y_proba, sample_weight)
    
    # 确定 y_true 的类型
    y_type = type_of_target(y_true, input_name="y_true")
    
    # 如果 y_true 不是二元分类，则抛出 ValueError 错误
    if y_type != "binary":
        raise ValueError(
            "Only binary classification is supported. The type of the target "
            f"is {y_type}."
        )
    
    # 如果 y_proba 中的最大值大于 1，则抛出 ValueError 错误
    if y_proba.max() > 1:
        raise ValueError("y_proba contains values greater than 1.")
    
    # 如果 y_proba 中的最小值小于 0，则抛出 ValueError 错误
    if y_proba.min() < 0:
        raise ValueError("y_proba contains values less than 0.")
    
    # 检查正类标签的一致性，如果不一致则重新确定 pos_label
    try:
        pos_label = _check_pos_label_consistency(pos_label, y_true)
    except ValueError:
        classes = np.unique(y_true)
        # 如果类别不是字符串类型，则将 pos_label 设置为类别中的最大值
        if classes.dtype.kind not in ("O", "U", "S"):
            pos_label = classes[-1]
        else:
            raise
    
    # 将 y_true 转换为以 pos_label 为正类标签的二元数组
    y_true = np.array(y_true == pos_label, int)
    
    # 计算 Brier 评分
    return np.average((y_true - y_proba) ** 2, weights=sample_weight)
@validate_params(
    {
        "y_true": ["array-like"],  # 参数验证装饰器，确保输入的 y_true 是类数组型数据
        "y_pred": ["array-like"],  # 参数验证装饰器，确保输入的 y_pred 是类数组型数据
        "sample_weight": ["array-like", None],  # 参数验证装饰器，可选参数，表示样本权重的类数组型数据或者 None
        "labels": ["array-like", None],  # 参数验证装饰器，可选参数，表示标签的类数组型数据或者 None
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器选项，优先跳过嵌套验证
)
def d2_log_loss_score(y_true, y_pred, *, sample_weight=None, labels=None):
    """
    :math:`D^2` score function, fraction of log loss explained.

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always predicts the per-class proportions
    of `y_true`, disregarding the input features, gets a D^2 score of 0.0.

    Read more in the :ref:`User Guide <d2_score_classification>`.

    .. versionadded:: 1.5

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        The actuals labels for the n_samples samples.

    y_pred : array-like of shape (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`~sklearn.preprocessing.LabelBinarizer`.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like, default=None
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.

    Returns
    -------
    d2 : float or ndarray of floats
        The D^2 score.

    Notes
    -----
    This is not a symmetric function.

    Like R^2, D^2 score may be negative (it need not actually be the square of
    a quantity D).

    This metric is not well-defined for a single sample and will return a NaN
    value if n_samples is less than two.
    """
    y_pred = check_array(y_pred, ensure_2d=False, dtype="numeric")  # 检查并确保 y_pred 是数值类型的类数组型数据
    check_consistent_length(y_pred, y_true, sample_weight)  # 检查 y_pred, y_true, sample_weight 的长度一致性
    if _num_samples(y_pred) < 2:  # 如果样本数小于两个，D^2 分数未定义，返回 NaN
        msg = "D^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")

    # log loss of the fitted model
    numerator = log_loss(
        y_true=y_true,
        y_pred=y_pred,
        normalize=False,
        sample_weight=sample_weight,
        labels=labels,
    )

    # Proportion of labels in the dataset
    weights = _check_sample_weight(sample_weight, y_true)  # 检查并获取样本权重

    _, y_value_indices = np.unique(y_true, return_inverse=True)
    counts = np.bincount(y_value_indices, weights=weights)
    y_prob = counts / weights.sum()
    y_pred_null = np.tile(y_prob, (len(y_true), 1))

    # log loss of the null model
    # 计算对数损失函数的分母部分，不进行归一化，根据指定的真实标签、空模型预测值、样本权重和标签集合计算
    denominator = log_loss(
        y_true=y_true,           # 真实的目标变量标签
        y_pred=y_pred_null,      # 空模型预测的概率值
        normalize=False,         # 不对损失进行归一化处理
        sample_weight=sample_weight,  # 可选的样本权重数组
        labels=labels,           # 所有可能的标签值列表
    )
    
    # 计算并返回归一化对数损失函数的结果，即 1 减去分子除以分母的比值
    return 1 - (numerator / denominator)
```