# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_base.py`

```
"""
Common code for all metrics.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import combinations  # 导入 combinations 函数，用于生成组合

import numpy as np  # 导入 NumPy 库，并使用 np 别名

from ..utils import check_array, check_consistent_length  # 导入自定义工具函数 check_array 和 check_consistent_length
from ..utils.multiclass import type_of_target  # 从多类别工具模块中导入 type_of_target 函数


def _average_binary_score(binary_metric, y_true, y_score, average, sample_weight=None):
    """Average a binary metric for multilabel classification.

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels in binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    average : {None, 'micro', 'macro', 'samples', 'weighted'}, default='macro'
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

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    binary_metric : callable, returns shape [n_classes]
        The binary metric function to use.

    Returns
    -------
    score : float or array of shape [n_classes]
        If not ``None``, average the score, else return the score for each
        classes.

    """
    average_options = (None, "micro", "macro", "weighted", "samples")  # 定义平均选项的元组

    # 如果 average 不在合法选项中，则抛出 ValueError 异常
    if average not in average_options:
        raise ValueError("average has to be one of {0}".format(average_options))

    y_type = type_of_target(y_true)  # 获取 y_true 的类型
    # 如果 y_type 不是 "binary" 或 "multilabel-indicator"，抛出异常
    if y_type not in ("binary", "multilabel-indicator"):
        raise ValueError("{0} format is not supported".format(y_type))

    if y_type == "binary":
        return binary_metric(y_true, y_score, sample_weight=sample_weight)  # 如果 y_type 是 "binary"，直接返回二元度量值

    check_consistent_length(y_true, y_score, sample_weight)  # 检查 y_true、y_score 和 sample_weight 的长度是否一致
    y_true = check_array(y_true)  # 使用 check_array 函数检查并转换 y_true
    y_score = check_array(y_score)  # 使用 check_array 函数检查并转换 y_score

    not_average_axis = 1  # 设置不进行平均的轴为 1
    score_weight = sample_weight  # 将 sample_weight 赋值给 score_weight
    average_weight = None  # 初始化 average_weight 为 None

    if average == "micro":
        if score_weight is not None:
            score_weight = np.repeat(score_weight, y_true.shape[1])  # 如果 score_weight 不为 None，则重复扩展为适应 y_true 的形状
        y_true = y_true.ravel()  # 将 y_true 展平为一维数组
        y_score = y_score.ravel()  # 将 y_score 展平为一维数组
    # 如果平均方式为加权平均
    elif average == "weighted":
        # 如果有指定分数权重
        if score_weight is not None:
            # 计算加权平均权重
            average_weight = np.sum(
                np.multiply(y_true, np.reshape(score_weight, (-1, 1))), axis=0
            )
        else:
            # 否则计算未加权平均权重
            average_weight = np.sum(y_true, axis=0)
        
        # 如果加权平均权重总和接近于零，则返回 0
        if np.isclose(average_weight.sum(), 0.0):
            return 0

    # 如果平均方式为样本平均
    elif average == "samples":
        # 交换平均权重和分数权重的引用
        average_weight = score_weight
        score_weight = None
        # 设定非平均轴索引为 0
        not_average_axis = 0

    # 如果 y_true 的维度为 1，则重新调整形状为 (-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    # 如果 y_score 的维度为 1，则重新调整形状为 (-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))

    # 获取类别数目
    n_classes = y_score.shape[not_average_axis]

    # 初始化得分数组
    score = np.zeros((n_classes,))

    # 遍历每个类别
    for c in range(n_classes):
        # 提取当前类别的真实标签并展平
        y_true_c = y_true.take([c], axis=not_average_axis).ravel()
        # 提取当前类别的预测分数并展平
        y_score_c = y_score.take([c], axis=not_average_axis).ravel()
        # 计算二元度量得分
        score[c] = binary_metric(y_true_c, y_score_c, sample_weight=score_weight)

    # 如果指定了平均方式
    if average is not None:
        # 如果存在加权平均权重
        if average_weight is not None:
            # 强制将权重为 0 的得分置为 0，以防止平均得分受到 0 权重的 NaN 元素影响
            average_weight = np.asarray(average_weight)
            score[average_weight == 0] = 0
        # 返回加权平均得分
        return np.average(score, weights=average_weight)
    else:
        # 否则返回原始得分数组
        return score
def _average_multiclass_ovo_score(binary_metric, y_true, y_score, average="macro"):
    """Average one-versus-one scores for multiclass classification.

    Uses the binary metric for one-vs-one multiclass classification,
    where the score is computed according to the Hand & Till (2001) algorithm.

    Parameters
    ----------
    binary_metric : callable
        The binary metric function to use that accepts the following as input:
            y_true_target : array, shape = [n_samples_target]
                Some sub-array of y_true for a pair of classes designated
                positive and negative in the one-vs-one scheme.
            y_score_target : array, shape = [n_samples_target]
                Scores corresponding to the probability estimates
                of a sample belonging to the designated positive class label

    y_true : array-like of shape (n_samples,)
        True multiclass labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.

    average : {'macro', 'weighted'}, default='macro'
        Determines the type of averaging performed on the pairwise binary
        metric scores:
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account. Classes
            are assumed to be uniformly distributed.
        ``'weighted'``:
            Calculate metrics for each label, taking into account the
            prevalence of the classes.

    Returns
    -------
    score : float
        Average of the pairwise binary metric scores.
    """
    # Ensure the length of y_true and y_score are consistent
    check_consistent_length(y_true, y_score)

    # Get unique classes from y_true
    y_true_unique = np.unique(y_true)
    n_classes = y_true_unique.shape[0]
    
    # Calculate the number of pairwise combinations
    n_pairs = n_classes * (n_classes - 1) // 2
    
    # Array to store pairwise scores
    pair_scores = np.empty(n_pairs)
    
    # Determine if 'weighted' averaging is required
    is_weighted = average == "weighted"
    
    # Array to store class prevalences if using 'weighted' averaging
    prevalence = np.empty(n_pairs) if is_weighted else None
    
    # Iterate over pairwise combinations of classes
    for ix, (a, b) in enumerate(combinations(y_true_unique, 2)):
        # Mask for samples belonging to class a or class b
        a_mask = y_true == a
        b_mask = y_true == b
        ab_mask = np.logical_or(a_mask, b_mask)
        
        # If 'weighted' averaging, compute class prevalence
        if is_weighted:
            prevalence[ix] = np.average(ab_mask)
        
        # Subset of y_true and corresponding y_score for classes a and b
        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]
        
        # Compute binary metric scores for classes a and b
        a_true_score = binary_metric(a_true, y_score[ab_mask, a])
        b_true_score = binary_metric(b_true, y_score[ab_mask, b])
        
        # Compute pairwise score as average of scores for classes a and b
        pair_scores[ix] = (a_true_score + b_true_score) / 2
    
    # Calculate the average of pairwise scores, optionally weighted by class prevalence
    return np.average(pair_scores, weights=prevalence)
```