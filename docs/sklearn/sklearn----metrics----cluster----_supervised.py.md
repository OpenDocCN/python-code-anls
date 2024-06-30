# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\_supervised.py`

```
"""
Utilities to evaluate the clustering performance of models.

Functions named as *_score return a scalar value to maximize: the higher the
better.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import warnings
from math import log
from numbers import Real

import numpy as np
from scipy import sparse as sp

from ...utils._array_api import get_namespace
from ...utils._param_validation import Interval, StrOptions, validate_params
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length
from ._expected_mutual_info_fast import expected_mutual_information


def check_clusterings(labels_true, labels_pred):
    """Check that the labels arrays are 1D and of same dimension.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        The true labels.

    labels_pred : array-like of shape (n_samples,)
        The predicted labels.
    """
    # Ensure labels_true is a 1D array, handle edge cases
    labels_true = check_array(
        labels_true,
        ensure_2d=False,
        ensure_min_samples=0,
        dtype=None,
    )

    # Ensure labels_pred is a 1D array, handle edge cases
    labels_pred = check_array(
        labels_pred,
        ensure_2d=False,
        ensure_min_samples=0,
        dtype=None,
    )

    # Determine the type of labels for both true and predicted labels
    type_label = type_of_target(labels_true)
    type_pred = type_of_target(labels_pred)

    # Warn if labels are continuous (should be discrete for clustering metrics)
    if "continuous" in (type_pred, type_label):
        msg = (
            "Clustering metrics expects discrete values but received"
            f" {type_label} values for label, and {type_pred} values "
            "for target"
        )
        warnings.warn(msg, UserWarning)

    # Check dimensions of labels arrays
    if labels_true.ndim != 1:
        raise ValueError("labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError("labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    
    # Check that labels_true and labels_pred have the same length
    check_consistent_length(labels_true, labels_pred)

    return labels_true, labels_pred


def _generalized_average(U, V, average_method):
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError(
            "'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'"
        )


@validate_params(
    {
        "labels_true": ["array-like", None],
        "labels_pred": ["array-like", None],
        "eps": [Interval(Real, 0, None, closed="left"), None],
        "sparse": ["boolean"],
        "dtype": "no_validation",  # delegate the validation to SciPy
    },
    prefer_skip_nested_validation=True,
)
def contingency_matrix(
    labels_true, labels_pred, *, eps=None, sparse=False, dtype=np.int64
):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        True labels.

    labels_pred : array-like of shape (n_samples,)
        Predicted labels.

    eps : float, default=None
        If a float, ensures entries in contingency matrix are non-negative.
        If None, use scipy.sparse.coo_matrix to construct the contingency matrix.

    sparse : boolean, default=False
        If True, return a sparse coo_matrix instead of a dense array.

    dtype : np.dtype, default=np.int64
        The data type of the resulting matrix.
    # labels_true : array-like of shape (n_samples,)
    #     Ground truth class labels to be used as a reference.
    # labels_pred : array-like of shape (n_samples,)
    #     Cluster labels to evaluate.
    # eps : float, default=None
    #     If a float, that value is added to all values in the contingency
    #     matrix. This helps to stop NaN propagation.
    #     If ``None``, nothing is adjusted.
    # sparse : bool, default=False
    #     If `True`, return a sparse CSR contingency matrix. If `eps` is not
    #     `None` and `sparse` is `True` will raise ValueError.
    #     .. versionadded:: 0.18
    # dtype : numeric type, default=np.int64
    #     Output dtype. Ignored if `eps` is not `None`.
    #     .. versionadded:: 0.24
    # Returns
    # -------
    # contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
    #     Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
    #     true class :math:`i` and in predicted class :math:`j`. If
    #     ``eps is None``, the dtype of this array will be integer unless set
    #     otherwise with the ``dtype`` argument. If ``eps`` is given, the dtype
    #     will be float.
    #     Will be a ``sklearn.sparse.csr_matrix`` if ``sparse=True``.
    # Examples
    # --------
    # >>> from sklearn.metrics.cluster import contingency_matrix
    # >>> labels_true = [0, 0, 1, 1, 2, 2]
    # >>> labels_pred = [1, 0, 2, 1, 0, 2]
    # >>> contingency_matrix(labels_true, labels_pred)
    # array([[1, 1, 0],
    #        [0, 1, 1],
    #        [1, 0, 1]])
    """

    # Check if sparse and eps are compatible
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    # Find unique classes and clusters, and map labels to indices
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]

    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix(
        (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=dtype,
    )

    # Convert to CSR format if sparse=True
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        # Convert to dense array if sparse=False
        contingency = contingency.toarray()
        # Add eps to contingency matrix if eps is not None
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps

    return contingency
# clustering measures

# 定义函数 pair_confusion_matrix，用于计算两个聚类结果的配对混淆矩阵
@validate_params(
    {
        "labels_true": ["array-like"],  # 真实标签数组，用作参考
        "labels_pred": ["array-like"],  # 预测的聚类标签数组，用于评估
    },
    prefer_skip_nested_validation=True,
)
def pair_confusion_matrix(labels_true, labels_pred):
    """Pair confusion matrix arising from two clusterings.

    The pair confusion matrix :math:`C` computes a 2 by 2 similarity matrix
    between two clusterings by considering all pairs of samples and counting
    pairs that are assigned into the same or into different clusters under
    the true and predicted clusterings [1]_.

    Considering a pair of samples that is clustered together a positive pair,
    then as in binary classification the count of true negatives is
    :math:`C_{00}`, false negatives is :math:`C_{10}`, true positives is
    :math:`C_{11}` and false positives is :math:`C_{01}`.

    Read more in the :ref:`User Guide <pair_confusion_matrix>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.

    Returns
    -------
    C : ndarray of shape (2, 2), dtype=np.int64
        The contingency matrix.

    See Also
    --------
    sklearn.metrics.rand_score : Rand Score.
    sklearn.metrics.adjusted_rand_score : Adjusted Rand Score.
    sklearn.metrics.adjusted_mutual_info_score : Adjusted Mutual Information.

    References
    ----------
    .. [1] :doi:`Hubert, L., Arabie, P. "Comparing partitions."
           Journal of Classification 2, 193–218 (1985).
           <10.1007/BF01908075>`

    Examples
    --------
    Perfectly matching labelings have all non-zero entries on the
    diagonal regardless of actual label values:

      >>> from sklearn.metrics.cluster import pair_confusion_matrix
      >>> pair_confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
      array([[8, 0],
             [0, 4]]...

    Labelings that assign all classes members to the same clusters
    are complete but may be not always pure, hence penalized, and
    have some off-diagonal non-zero entries:

      >>> pair_confusion_matrix([0, 0, 1, 2], [0, 0, 1, 1])
      array([[8, 2],
             [0, 2]]...

    Note that the matrix is not symmetric.
    """
    # 确保输入的聚类结果有效性，并返回标签数组
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    # 计算样本数
    n_samples = np.int64(labels_true.shape[0])

    # 计算配对混淆矩阵的关键数据
    contingency = contingency_matrix(
        labels_true, labels_pred, sparse=True, dtype=np.int64
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data**2).sum()

    # 填充混淆矩阵 C
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares

    return C
    # 返回计算结果 C
    return C
# 使用装饰器 @validate_params 进行参数验证，确保输入的 labels_true 和 labels_pred 符合预期的格式和类型
@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
# 定义函数 adjusted_rand_score，计算调整后的 Rand 指数
def adjusted_rand_score(labels_true, labels_pred):
    """Rand index adjusted for chance.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.

    The raw RI score is then "adjusted for chance" into the ARI score
    """
    # 调用 pair_confusion_matrix 函数，计算混淆矩阵
    contingency = pair_confusion_matrix(labels_true, labels_pred)
    # 计算混淆矩阵对角线元素之和，作为分子
    numerator = contingency.diagonal().sum()
    # 计算混淆矩阵所有元素之和，作为分母
    denominator = contingency.sum()

    # 处理特殊情况：如果分子等于分母或者分母为0，则返回完美匹配的相似性分数1.0
    if numerator == denominator or denominator == 0:
        # 特殊情况：数据未分割成簇或每个文档被分配到唯一的簇，返回1.0表示完美匹配
        return 1.0

    # 计算并返回调整后的 Rand 指数
    return numerator / denominator
    """
    Compute the adjusted Rand index (ARI) to evaluate the similarity between
    two clusterings.

    ARI measures the similarity of the two clusterings, considering them as
    partitions of the same set of samples. It is adjusted for chance and
    accounts for the fact that random labelings have a value close to 0.0.

    ARI is based on the Rand index (RI), which is the ratio of the number of
    pairs of samples that are either in the same or different clusters in both
    the predicted and true labelings.

    ARI is computed using the following scheme::

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation). The adjusted Rand index is bounded below by -0.5 for
    especially discordant clusterings.

    ARI is a symmetric measure::

        adjusted_rand_score(a, b) == adjusted_rand_score(b, a)

    Read more in the :ref:`User Guide <adjusted_rand_score>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=int
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=int
        Cluster labels to evaluate.

    Returns
    -------
    ARI : float
       Similarity score between -0.5 and 1.0. Random labelings have an ARI
       close to 0.0. 1.0 stands for perfect match.

    See Also
    --------
    adjusted_mutual_info_score : Adjusted Mutual Information.

    References
    ----------
    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    .. [Steinley2004] D. Steinley, Properties of the Hubert-Arabie
      adjusted Rand index, Psychological Methods 2004

    .. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

    .. [Chacon] :doi:`Minimum adjusted Rand index for two clusterings of a given size,
      2022, J. E. Chacón and A. I. Rastrojo <10.1007/s11634-022-00491-w>`

    Examples
    --------
    Perfectly matching labelings have a score of 1 even

      >>> from sklearn.metrics.cluster import adjusted_rand_score
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete but may not always be pure, hence penalized::

      >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])
      0.57...

    ARI is symmetric, so labelings that have pure clusters with members
    coming from the same classes but unnecessary splits are penalized::

      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])
      0.57...

    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::

      >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    ARI may take a negative value for especially discordant labelings that
    are a worse choice than the expected value of random labels::

      >>> adjusted_rand_score([0, 0, 1, 1], [0, 1, 0, 1])
      -0.5
    """

    # Compute the pair confusion matrix (true negatives, false positives,
    # false negatives, true positives) based on the given labels
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

    # Convert values to Python integer types to prevent overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    # 处理特殊情况：当假阴性数和假阳性数均为零时，返回1.0
    if fn == 0 and fp == 0:
        返回1.0，表示模型预测与实际情况完全一致的特殊情况处理
    
    # 计算并返回二倍的 Matthews 相关系数
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
# 使用装饰器 @validate_params 进行参数验证，确保函数输入的正确性
@validate_params(
    {
        "labels_true": ["array-like"],  # labels_true 参数应为类数组
        "labels_pred": ["array-like"],  # labels_pred 参数应为类数组
        "beta": [Interval(Real, 0, None, closed="left")],  # beta 参数为大于等于0的实数
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
# 定义函数 homogeneity_completeness_v_measure，计算同质性、完整性和 V-Measure 分数
def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0):
    """Compute the homogeneity and completeness and V-Measure scores at once.

    Those metrics are based on normalized conditional entropy measures of
    the clustering labeling to evaluate given the knowledge of a Ground
    Truth class labels of the same samples.

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    Both scores have positive values between 0.0 and 1.0, larger values
    being desirable.

    Those 3 metrics are independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score values in any way.

    V-Measure is furthermore symmetric: swapping ``labels_true`` and
    ``label_pred`` will give the same score. This does not hold for
    homogeneity and completeness. V-Measure is identical to
    :func:`normalized_mutual_info_score` with the arithmetic averaging
    method.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    beta : float, default=1.0
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.

    Returns
    -------
    homogeneity : float
        Score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling.

    completeness : float
        Score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling.

    v_measure : float
        Harmonic mean of the first two.

    See Also
    --------
    homogeneity_score : Homogeneity metric of cluster labeling.
    completeness_score : Completeness metric of cluster labeling.
    v_measure_score : V-Measure (NMI with arithmetic mean option).

    Examples
    --------
    >>> from sklearn.metrics import homogeneity_completeness_v_measure
    >>> y_true, y_pred = [0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2]
    >>> homogeneity_completeness_v_measure(y_true, y_pred)
    (0.71..., 0.77..., 0.73...)
    """
    
    # 使用 check_clusterings 函数验证 labels_true 和 labels_pred 的合法性
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    
    # 如果 labels_true 的长度为 0，返回最大值的同质性、完整性和 V-Measure 分数
    if len(labels_true) == 0:
        return 1.0, 1.0, 1.0
    
    # 计算 labels_true 和 labels_pred 的熵值
    entropy_C = entropy(labels_true)
    entropy_K = entropy(labels_pred)
    # 计算真实标签和预测标签的列联表，以稀疏矩阵形式返回
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    
    # 计算互信息，传入的参数为 None，因为互信息的计算已经使用了列联表
    MI = mutual_info_score(None, None, contingency=contingency)

    # 计算同质性，如果条件熵 entropy_C 存在且不为零，则使用互信息 MI 进行计算，否则返回 1.0
    homogeneity = MI / (entropy_C) if entropy_C else 1.0
    
    # 计算完整性，如果条件熵 entropy_K 存在且不为零，则使用互信息 MI 进行计算，否则返回 1.0
    completeness = MI / (entropy_K) if entropy_K else 1.0

    # 计算 V-measure 分数，根据同质性和完整性的加权平均值进行计算
    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = (
            (1 + beta)
            * homogeneity
            * completeness
            / (beta * homogeneity + completeness)
        )

    # 返回同质性、完整性和 V-measure 分数
    return homogeneity, completeness, v_measure_score
# 使用装饰器 @validate_params 对 completeness_score 函数进行参数验证
@validate_params(
    {
        "labels_true": ["array-like"],  # 参数 labels_true 应为类数组类型
        "labels_pred": ["array-like"],  # 参数 labels_pred 应为类数组类型
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
# 计算给定聚类标签的完整度指标
def completeness_score(labels_true, labels_pred):
    """Compute completeness metric of a cluster labeling given a ground truth.

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    """
    # 调用完整度-一致性-互信息 V-Measure 方法，返回完整度评分
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[1]
    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`homogeneity_score` which will be different in
    general.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    completeness : float
       Score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling.

    See Also
    --------
    homogeneity_score : Homogeneity metric of cluster labeling.
    v_measure_score : V-Measure (NMI with arithmetic mean option).

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    Examples
    --------

    Perfect labelings are complete::

      >>> from sklearn.metrics.cluster import completeness_score
      >>> completeness_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-perfect labelings that assign all classes members to the same clusters
    are still complete::

      >>> print(completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))
      1.0
      >>> print(completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))
      0.999...

    If classes members are split across different clusters, the
    assignment cannot be complete::

      >>> print(completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))
      0.0
      >>> print(completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
      0.0
    """
    # 返回使用 labels_true 和 labels_pred 计算的完整性度量值
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[1]
@validate_params(
    {
        "labels_true": ["array-like"],  # 参数验证装饰器，验证labels_true参数为数组样式
        "labels_pred": ["array-like"],  # 参数验证装饰器，验证labels_pred参数为数组样式
        "beta": [Interval(Real, 0, None, closed="left")],  # 参数验证装饰器，验证beta参数为大于等于0的实数
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器设置，优先跳过嵌套验证
)
def v_measure_score(labels_true, labels_pred, *, beta=1.0):
    """
    给定一个基准的V-measure簇标签。

    这个得分与normalized_mutual_info_score函数相同，使用'arithmetic'选项进行平均。

    V-measure是均匀性和完整性之间的调和平均值::

        v = (1 + beta) * homogeneity * completeness
             / (beta * homogeneity + completeness)

    这个度量与标签的绝对值无关：
    类或簇标签值的任意置换都不会以任何方式改变得分值。

    这个度量进一步是对称的：
    将'labels_true'与'labels_pred'互换将返回相同的得分值。当真实基准不知道时，
    这对于衡量两个独立标签分配策略在相同数据集上的一致性是有用的。

    在:ref:`User Guide <homogeneity_completeness>`中阅读更多。

    Parameters
    ----------
    labels_true : 形状为(n_samples,)的数组样式
        用作参考的真实类标签。

    labels_pred : 形状为(n_samples,)的数组样式
        要评估的簇标签。

    beta : 浮点数，默认为1.0
        比率，用于衡量'均匀性'与'完整性'的权重。
        如果beta大于1，则在计算中更加强调'完整性'。
        如果beta小于1，则在计算中更加强调'均匀性'。

    Returns
    -------
    v_measure : 浮点数
       在0.0到1.0之间的得分。1.0代表完全完整的标签。

    See Also
    --------
    homogeneity_score : 簇标签的均匀性度量。
    completeness_score : 簇标签的完整性度量。
    normalized_mutual_info_score : 正规化互信息。

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    Examples
    --------
    完美的标签既是均匀的又是完整的，因此得分为1.0::

      >>> from sklearn.metrics.cluster import v_measure_score
      >>> v_measure_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> v_measure_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    将所有类成员分配到相同簇的标签是完整的但不是均匀的，因此受到惩罚::

      >>> print("%.6f" % v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))
      0.8...
      >>> print("%.6f" % v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
      0.66...
    """
    # 这段代码片段用于展示如何使用 V-measure 评分函数来评估聚类结果的准确性。
    # V-measure 评分考虑了聚类结果的完整性和同质性，通过计算它们的加权平均来获得一个综合的度量。
    # 下面的示例演示了不同聚类结果的 V-measure 分数输出：
    # - 第一个示例中，聚类结果的 V-measure 得分为 0.8...，表示较高的完整性和同质性。
    # - 第二个示例中，聚类结果的 V-measure 得分为 0.66...，表示中等程度的完整性和同质性。
    # - 第三个示例中，由于类成员完全分布在不同的聚类中，聚类结果的 V-measure 得分为 0.0...，表示完全不完整。
    # - 最后一个示例中，由于样本来自于完全不同的类别被分到同一个聚类中，导致聚类结果的 V-measure 得分为 0.0...，表示完全不同质性。
    # 返回的值是 homogeneity, completeness, V-measure 中的 V-measure 部分，用于评估聚类结果的综合性能。
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred, beta=beta)[2]
@validate_params(
    {
        "labels_true": ["array-like", None],  # 参数验证装饰器，验证labels_true参数为array-like类型或None
        "labels_pred": ["array-like", None],  # 参数验证装饰器，验证labels_pred参数为array-like类型或None
        "contingency": ["array-like", "sparse matrix", None],  # 参数验证装饰器，验证contingency参数为array-like、sparse matrix类型或None
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器的设置，优先跳过嵌套验证
)
def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    """Mutual Information between two clusterings.

    The Mutual Information is a measure of the similarity between two labels
    of the same data. Where :math:`|U_i|` is the number of the samples
    in cluster :math:`U_i` and :math:`|V_j|` is the number of the
    samples in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:

    .. math::

        MI(U,V)=\\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}
        \\log\\frac{N|U_i \\cap V_j|}{|U_i||V_j|}

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (i.e
    ``label_true``) with :math:`V` (i.e. ``label_pred``) will return the
    same score value. This can be useful to measure the agreement of two
    independent label assignments strategies on the same dataset when the
    real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    contingency : {array-like, sparse matrix} of shape \
            (n_classes_true, n_classes_pred), default=None
        A contingency matrix given by the
        :func:`~sklearn.metrics.cluster.contingency_matrix` function. If value
        is ``None``, it will be computed, otherwise the given value is used,
        with ``labels_true`` and ``labels_pred`` ignored.

    Returns
    -------
    mi : float
       Mutual information, a non-negative value, measured in nats using the
       natural logarithm.

    See Also
    --------
    adjusted_mutual_info_score : Adjusted against chance Mutual Information.
    normalized_mutual_info_score : Normalized Mutual Information.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    Examples
    --------
    >>> from sklearn.metrics import mutual_info_score
    >>> labels_true = [0, 1, 1, 0, 1, 0]
    >>> labels_pred = [0, 1, 0, 0, 1, 1]
    >>> mutual_info_score(labels_true, labels_pred)
    0.056...
    """
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)  # 检查并返回labels_true和labels_pred
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)  # 使用labels_true和labels_pred计算并返回contingency matrix
    # 如果传入的 contingency 是一个 ndarray 对象
    if isinstance(contingency, np.ndarray):
        # 对于数组，找出非零元素的位置
        nzx, nzy = np.nonzero(contingency)
        # 提取非零元素的值
        nz_val = contingency[nzx, nzy]
    else:
        # 对于稀疏矩阵，使用 sp.find 函数找出非零元素的位置和对应的值
        nzx, nzy, nz_val = sp.find(contingency)

    # 计算 contingency 的总和
    contingency_sum = contingency.sum()
    # 计算 contingency 按行求和（pi）和按列求和（pj），并展平为一维数组
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # 根据互信息的定义，如果 pi 或 pj 的大小为 1，则互信息为 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    # 计算非零元素的对数值
    log_contingency_nm = np.log(nz_val)
    # 计算 contingency_nm，即非零元素值与 contingency 总和的比值
    contingency_nm = nz_val / contingency_sum
    # 计算外积的对数值，但只需计算非零元素位置处的乘积
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    # 计算外积的对数值，其中 log_outer = -log(outer) + log(pi.sum()) + log(pj.sum())
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
    # 计算互信息（MI）的每一项
    mi = (
        contingency_nm * (log_contingency_nm - np.log(contingency_sum))
        + contingency_nm * log_outer
    )
    # 将非常接近于零的互信息值设置为零，以避免数值计算中的误差
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    # 对互信息的所有项求和，并限制结果的下界为 0.0
    return np.clip(mi.sum(), 0.0, None)
# 使用装饰器 `@validate_params` 对 `adjusted_mutual_info_score` 函数进行参数验证
@validate_params(
    {
        "labels_true": ["array-like"],  # 参数 labels_true 应为类数组对象
        "labels_pred": ["array-like"],  # 参数 labels_pred 应为类数组对象
        "average_method": [StrOptions({"arithmetic", "max", "min", "geometric"})],  # 参数 average_method 应为字符串，取值为集合 {"arithmetic", "max", "min", "geometric"} 中的一个
    },
    prefer_skip_nested_validation=True,  # 设置忽略嵌套验证
)
# 定义 adjusted_mutual_info_score 函数，计算两个聚类结果之间的调整互信息分数
def adjusted_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic"  # 标签真实值，预测值，以及计算正则化因子的方法，默认为 'arithmetic'
):
    """Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. It accounts for the fact that
    the MI is generally higher for two clusterings with a larger number of
    clusters, regardless of whether there is actually more information shared.
    For two clusterings :math:`U` and :math:`V`, the AMI is given as::

        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (``label_true``)
    with :math:`V` (``labels_pred``) will return the same score value. This can
    be useful to measure the agreement of two independent label assignments
    strategies on the same dataset when the real ground truth is not known.

    Be mindful that this function is an order of magnitude slower than other
    metrics, such as the Adjusted Rand Index.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer in the denominator.

        .. versionadded:: 0.20

        .. versionchanged:: 0.22
           The default value of ``average_method`` changed from 'max' to
           'arithmetic'.

    Returns
    -------
    ami: float (upperlimited by 1.0)
       The AMI returns a value of 1 when the two partitions are identical
       (ie perfectly matched). Random partitions (independent labellings) have
       an expected AMI around 0 on average hence can be negative. The value is
       in adjusted nats (based on the natural logarithm).

    See Also
    --------
    adjusted_rand_score : Adjusted Rand Index.
    mutual_info_score : Mutual Information (not adjusted for chance).

    References
    ----------
    .. [1] `Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
       Clusterings Comparison: Variants, Properties, Normalization and
       Correction for Chance, JMLR
       <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf>`_

    """
    # 实现计算调整互信息分数的函数体，不进行具体注释
    # 导入必要的库和函数
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    # 定义函数，计算调整后的互信息分数，用于比较两个聚类结果的相似度
    adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
    # 示例：完美的标签分配，既是同质的又是完整的，因此分数为1.0
    adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
    # 示例：如果类的成员完全分布在不同的聚类中，则分配是完全不完整的，因此AMI为空，分数为0.0

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    # 检查并确保标签数组的一致性

    n_samples = labels_true.shape[0]
    # 获取样本数量

    classes = np.unique(labels_true)
    # 获取真实标签中的唯一类别

    clusters = np.unique(labels_pred)
    # 获取预测标签中的唯一聚类

    # 处理特殊情况：如果类别数目和聚类数目均为1或均为0，则表示没有聚类，返回AMI为1.0
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # 计算标签的列联表，用于互信息计算

    # 计算实际互信息
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)

    # 计算期望互信息
    emi = expected_mutual_information(contingency, n_samples)

    # 计算真实标签和预测标签的熵
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)

    # 根据平均方法计算归一化因子
    normalizer = _generalized_average(h_true, h_pred, average_method)

    # 计算分母，避免出现分母为零的情况，确保AMI的计算正确性
    denominator = normalizer - emi
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)

    # 计算AMI并返回
    ami = (mi - emi) / denominator
    return ami
# 使用装饰器 @validate_params 对函数进行参数验证，确保输入参数的类型和值符合要求
@validate_params(
    {
        "labels_true": ["array-like"],  # labels_true 参数应为类数组类型
        "labels_pred": ["array-like"],  # labels_pred 参数应为类数组类型
        "average_method": [StrOptions({"arithmetic", "max", "min", "geometric"})],  # average_method 参数应为字符串，且取值在指定集合内
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证以提高效率
)
# 定义函数 normalized_mutual_info_score，计算两个聚类结果之间的标准化互信息
def normalized_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic"
):
    """Normalized Mutual Information between two clusterings.

    Normalized Mutual Information (NMI) is a normalization of the Mutual
    Information (MI) score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation). In this function, mutual
    information is normalized by some generalized mean of ``H(labels_true)``
    and ``H(labels_pred))``, defined by the `average_method`.

    This measure is not adjusted for chance. Therefore
    :func:`adjusted_mutual_info_score` might be preferred.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets.

    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer in the denominator.

        .. versionadded:: 0.20

        .. versionchanged:: 0.22
           The default value of ``average_method`` changed from 'geometric' to
           'arithmetic'.

    Returns
    -------
    nmi : float
       Score between 0.0 and 1.0 in normalized nats (based on the natural
       logarithm). 1.0 stands for perfectly complete labeling.

    See Also
    --------
    v_measure_score : V-Measure (NMI with arithmetic mean option).
    adjusted_rand_score : Adjusted Rand Index.
    adjusted_mutual_info_score : Adjusted Mutual Information (adjusted
        against chance).

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the NMI is null::

      >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0
    """
    # 检查并确保输入的聚类结果标签格式正确，返回格式化后的标签列表
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    # 获取真实标签中的所有类别
    classes = np.unique(labels_true)
    # 获取预测标签中的所有类别
    clusters = np.unique(labels_pred)

    # 处理特殊情况：当真实标签和预测标签都只有一个类别或者都为空时，直接返回 NMI 为 1.0
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    # 计算真实标签和预测标签之间的列联表，使用稀疏矩阵存储
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # 将列联表转换为 float64 类型，避免后续计算时出现数据类型错误
    contingency = contingency.astype(np.float64, copy=False)
    # 计算真实标签和预测标签之间的互信息
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)

    # 如果互信息 mi 等于 0，则无论归一化方法如何，NMI 都为 0
    if mi == 0:
        return 0.0

    # 计算真实标签和预测标签的熵
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)

    # 根据指定的平均方法计算归一化因子
    normalizer = _generalized_average(h_true, h_pred, average_method)
    # 返回归一化互信息
    return mi / normalizer
# 使用装饰器对函数参数进行验证，并指定参数的数据类型和其他约束条件
@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
        "sparse": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
# 计算 Fowlkes-Mallows 分数，衡量两个数据集的聚类相似性
def fowlkes_mallows_score(labels_true, labels_pred, *, sparse=False):
    """Measure the similarity of two clusterings of a set of points.

    .. versionadded:: 0.18

    The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of
    the precision and recall::

        FMI = TP / sqrt((TP + FP) * (TP + FN))

    Where ``TP`` is the number of **True Positive** (i.e. the number of pair of
    points that belongs in the same clusters in both ``labels_true`` and
    ``labels_pred``), ``FP`` is the number of **False Positive** (i.e. the
    number of pair of points that belongs in the same clusters in
    ``labels_true`` and not in ``labels_pred``) and ``FN`` is the number of
    **False Negative** (i.e. the number of pair of points that belongs in the
    same clusters in ``labels_pred`` and not in ``labels_True``).

    The score ranges from 0 to 1. A high value indicates a good similarity
    between two clusters.

    Read more in the :ref:`User Guide <fowlkes_mallows_scores>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=int
        A clustering of the data into disjoint subsets.

    labels_pred : array-like of shape (n_samples,), dtype=int
        A clustering of the data into disjoint subsets.

    sparse : bool, default=False
        Compute contingency matrix internally with sparse matrix.

    Returns
    -------
    score : float
       The resulting Fowlkes-Mallows score.

    References
    ----------
    .. [1] `E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
       hierarchical clusterings". Journal of the American Statistical
       Association
       <https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478008>`_

    .. [2] `Wikipedia entry for the Fowlkes-Mallows Index
           <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import fowlkes_mallows_score
      >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally random, hence the FMI is null::

      >>> fowlkes_mallows_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    """
    # 调用函数检查并规范化传入的聚类标签
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    # 获取样本数
    (n_samples,) = labels_true.shape

    # 计算包含稀疏矩阵选项的列联表
    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    # 将矩阵类型转换为 int64 类型，提高计算精度
    c = c.astype(np.int64, copy=False)
    # 计算 TP^2 的和减去样本数
    tk = np.dot(c.data, c.data) - n_samples
    # 计算列和的平方和减去样本数
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    # 计算行和的平方和减去样本数
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    # 如果 tk 不等于 0.0，则计算 tk / pk 和 tk / qk 的平方根乘积，并返回结果
    # 如果 tk 等于 0.0，则返回 0.0
    return np.sqrt(tk / pk) * np.sqrt(tk / qk) if tk != 0.0 else 0.0
@validate_params(
    {
        "labels": ["array-like"],  # 参数验证装饰器，用于验证函数参数
    },
    prefer_skip_nested_validation=True,
)
def entropy(labels):
    """Calculate the entropy for a labeling.

    Parameters
    ----------
    labels : array-like of shape (n_samples,), dtype=int
        The labels.

    Returns
    -------
    entropy : float
       The entropy for a labeling.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    xp, is_array_api_compliant = get_namespace(labels)  # 获取命名空间和数组API兼容性
    labels_len = labels.shape[0] if is_array_api_compliant else len(labels)  # 计算标签的长度

    if labels_len == 0:  # 如果标签长度为0，返回熵为1.0
        return 1.0

    pi = xp.astype(xp.unique_counts(labels)[1], xp.float64)  # 计算标签的频率并转换为浮点数类型

    # 单一簇 => 熵为零
    if pi.size == 1:  # 如果只有一个簇，返回熵为0.0
        return 0.0

    pi_sum = xp.sum(pi)  # 计算频率的总和
    # 计算熵的公式为 -Σ(p_i / Σ(p_i) * (log(p_i) - log(Σ(p_i))))
    # 为了避免精度损失，log(a / b) 应计算为 log(a) - log(b)
    # 始终将结果转换为Python标量（在CPU上），而不是设备特定的标量数组。
    return float(-xp.sum((pi / pi_sum) * (xp.log(pi) - log(pi_sum))))
```