# `D:\src\scipysrc\scikit-learn\sklearn\metrics\cluster\_unsupervised.py`

```
# 导入 functools 库，用于函数式编程支持
import functools
# 导入 Integral 类型，用于验证参数类型
from numbers import Integral

# 导入 numpy 库，并指定为 np 别名
import numpy as np
# 导入 issparse 函数，用于检查是否稀疏矩阵
from scipy.sparse import issparse

# 从 scikit-learn 的 preprocessing 模块中导入 LabelEncoder 类
from ...preprocessing import LabelEncoder
# 从 scikit-learn 的 utils 模块中导入 _safe_indexing, check_random_state, check_X_y 函数
from ...utils import _safe_indexing, check_random_state, check_X_y
# 从 scikit-learn 的 utils._array_api 模块中导入 _atol_for_type 函数
from ...utils._array_api import _atol_for_type
# 从 scikit-learn 的 utils._param_validation 模块中导入 Interval, StrOptions, validate_params 函数
from ...utils._param_validation import (
    Interval,
    StrOptions,
    validate_params,
)
# 从 scikit-learn 的 metrics.pairwise 模块中导入 _VALID_METRICS, pairwise_distances, pairwise_distances_chunked 函数
from ..pairwise import _VALID_METRICS, pairwise_distances, pairwise_distances_chunked


def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    n_samples : int
        Number of samples.
    """
    # 检查标签数量是否有效
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "labels": ["array-like"],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "sample_size": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def silhouette_score(
    X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds
):
    """Compute the mean Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels
    is ``2 <= n_labels <= n_samples - 1``.

    This function returns the mean Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`silhouette_samples`.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Read more in the :ref:`User Guide <silhouette_coefficient>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    """
    # 如果指定了 sample_size 参数，则进行样本抽样
    if sample_size is not None:
        # 检查并确保 X 和 labels 是兼容的输入格式
        X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])
        # 检查并设置随机数生成器的种子，以保证结果可复现性
        random_state = check_random_state(random_state)
        # 从样本中随机选择指定数量的索引
        indices = random_state.permutation(X.shape[0])[:sample_size]
        # 如果使用预先计算的距离矩阵作为输入
        if metric == "precomputed":
            # 从距离矩阵中选择对应的样本子集
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            # 从原始数据中选择对应的样本子集
            X, labels = X[indices], labels[indices]
    # 计算所有样本的 Silhouette Coefficient 的平均值，并返回
    return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))
# 计算指定数据块的轮廓统计信息

def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X.

    Parameters
    ----------
    D_chunk : {array-like, sparse matrix} of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk. If a sparse matrix is provided,
        only CSR format is accepted.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    # 计算数据块中样本的数量
    n_chunk_samples = D_chunk.shape[0]
    
    # 初始化一个数组用于累积每个样本到每个簇的距离
    cluster_distances = np.zeros(
        (n_chunk_samples, len(label_freqs)), dtype=D_chunk.dtype
    )

    # 如果数据块是稀疏矩阵，则按稀疏矩阵的方式处理
    if issparse(D_chunk):
        # 确保稀疏矩阵为CSR格式
        if D_chunk.format != "csr":
            raise TypeError(
                "Expected CSR matrix. Please pass sparse matrix in CSR format."
            )
        # 遍历每个样本，累积其与对应簇的距离
        for i in range(n_chunk_samples):
            indptr = D_chunk.indptr
            indices = D_chunk.indices[indptr[i] : indptr[i + 1]]
            sample_weights = D_chunk.data[indptr[i] : indptr[i + 1]]
            sample_labels = np.take(labels, indices)
            cluster_distances[i] += np.bincount(
                sample_labels, weights=sample_weights, minlength=len(label_freqs)
            )
    else:
        # 处理稠密矩阵情况，累积每个样本与对应簇的距离
        for i in range(n_chunk_samples):
            sample_weights = D_chunk[i]
            sample_labels = labels
            cluster_distances[i] += np.bincount(
                sample_labels, weights=sample_weights, minlength=len(label_freqs)
            )

    # 选取数据块内部簇距离
    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    intra_cluster_distances = cluster_distances[intra_index]

    # 将选取的簇距离置为无穷大，进行归一化处理并提取最小值作为簇间距离
    cluster_distances[intra_index] = np.inf
    cluster_distances /= label_freqs
    inter_cluster_distances = cluster_distances.min(axis=1)

    # 返回内部簇距离和簇间距离
    return intra_cluster_distances, inter_cluster_distances
    """
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 ``<= n_labels <= n_samples - 1``.

    This function returns the Silhouette Coefficient for each sample.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.

    Read more in the :ref:`User Guide <silhouette_coefficient>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array. If
        a sparse matrix is provided, CSR format should be favoured avoiding
        an additional copy.

    labels : array-like of shape (n_samples,)
        Label values for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`~sklearn.metrics.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    silhouette : array-like of shape (n_samples,)
        Silhouette Coefficients for each sample.

    References
    ----------

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Examples
    --------
    >>> from sklearn.metrics import silhouette_samples
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>> X, y = make_blobs(n_samples=50, random_state=42)
    >>> kmeans = KMeans(n_clusters=3, random_state=42)
    >>> labels = kmeans.fit_predict(X)
    >>> silhouette_samples(X, labels)
    array([...])
    """

    # Validate and convert input data X and labels y to ensure compatibility
    X, labels = check_X_y(X, labels, accept_sparse=["csr"])

    # Check for non-zero diagonal entries in precomputed distance matrix
    # 如果距离度量(metric)为"precomputed"，则进行以下检查
    if metric == "precomputed":
        # 创建一个错误消息，指出预计算的距离矩阵包含非零对角线元素的问题
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero "
            "elements on the diagonal. Use np.fill_diagonal(X, 0)."
        )
        # 如果输入矩阵 X 的数据类型为浮点型
        if X.dtype.kind == "f":
            # 根据 X 的数据类型获取容差阈值
            atol = _atol_for_type(X.dtype)

            # 如果发现 X 的对角线元素的绝对值大于容差阈值
            if np.any(np.abs(X.diagonal()) > atol):
                # 抛出先前创建的错误消息
                raise error_msg
        # 如果输入矩阵 X 的对角线元素不全为零（整数类型数据）
        elif np.any(X.diagonal() != 0):
            # 抛出先前创建的错误消息
            raise error_msg

    # 创建一个标签编码器对象
    le = LabelEncoder()
    # 使用标签编码器对象对输入的标签进行编码转换
    labels = le.fit_transform(labels)
    # 获取标签的样本数量
    n_samples = len(labels)
    # 统计每个标签出现的频率
    label_freqs = np.bincount(labels)
    # 检查标签类别数和样本数的匹配性
    check_number_of_labels(len(le.classes_), n_samples)

    # 将距离度量参数(metric)添加到关键字参数中
    kwds["metric"] = metric
    # 创建一个偏函数，用于减少轮廓系数计算的函数
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs
    )
    # 使用分块计算距离函数计算两两样本之间的距离
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    # 将分块计算结果解压缩为两个列表，分别表示簇内距离和簇间距离
    intra_clust_dists, inter_clust_dists = results
    # 将簇内距离和簇间距离的列表连接成一个数组
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    # 计算每个样本点的簇内距离的分母部分
    denom = (label_freqs - 1).take(labels, mode="clip")
    # 忽略除以零产生的警告，计算归一化的簇内距离
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    # 计算每个样本点的轮廓系数
    sil_samples = inter_clust_dists - intra_clust_dists
    # 忽略除以零产生的警告，计算标准化的轮廓系数
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # 将结果中的NaN值（由于簇大小为1而产生）转换为0
    return np.nan_to_num(sil_samples)
# 使用装饰器 @validate_params 进行参数验证，确保输入参数 X 和 labels 符合指定的数据类型要求
@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
# 定义计算 Calinski-Harabasz 分数的函数
def calinski_harabasz_score(X, labels):
    """Compute the Calinski and Harabasz score.

    It is also known as the Variance Ratio Criterion.

    The score is defined as ratio of the sum of between-cluster dispersion and
    of within-cluster dispersion.

    Read more in the :ref:`User Guide <calinski_harabasz_index>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.

    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.metrics import calinski_harabasz_score
    >>> X, _ = make_blobs(random_state=0)
    >>> kmeans = KMeans(n_clusters=3, random_state=0,).fit(X)
    >>> calinski_harabasz_score(X, kmeans.labels_)
    114.8...
    """
    # 使用 sklearn 提供的函数检查输入参数 X 和 labels，确保它们符合预期的格式
    X, labels = check_X_y(X, labels)
    # 使用 LabelEncoder 对 labels 进行编码
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # 获取样本数和类别数
    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    # 检查标签数和样本数是否匹配
    check_number_of_labels(n_labels, n_samples)

    # 初始化额外离散度和内部离散度为 0
    extra_disp, intra_disp = 0.0, 0.0
    # 计算样本均值
    mean = np.mean(X, axis=0)
    # 遍历每个类别 k
    for k in range(n_labels):
        # 获取属于当前类别 k 的样本集合
        cluster_k = X[labels == k]
        # 计算当前类别 k 的样本均值
        mean_k = np.mean(cluster_k, axis=0)
        # 计算额外离散度
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        # 计算内部离散度
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    # 计算 Calinski-Harabasz 分数并返回
    return (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )


# 使用装饰器 @validate_params 进行参数验证，确保输入参数 X 和 labels 符合指定的数据类型要求
@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
# 定义计算 Davies-Bouldin 分数的函数
def davies_bouldin_score(X, labels):
    """Compute the Davies-Bouldin score.

    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.

    The minimum score is zero, with lower values indicating better clustering.

    Read more in the :ref:`User Guide <davies-bouldin_index>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    """
    # 检查输入的数据 X 和对应的标签 labels，确保它们符合要求并返回处理后的数据
    X, labels = check_X_y(X, labels)

    # 创建一个标签编码器对象
    le = LabelEncoder()
    # 对标签进行编码转换为数值
    labels = le.fit_transform(labels)

    # 获取样本数和标签类别数
    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    # 检查标签数是否与样本数匹配，确保聚类合理性
    check_number_of_labels(n_labels, n_samples)

    # 初始化每个类别内部距离的数组
    intra_dists = np.zeros(n_labels)
    # 初始化每个类别的质心数组
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)

    # 遍历每个类别
    for k in range(n_labels):
        # 提取属于当前类别 k 的数据点集合
        cluster_k = _safe_indexing(X, labels == k)
        # 计算当前类别 k 的质心坐标
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        # 计算当前类别 k 内部所有数据点与其质心的平均距离
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))

    # 计算所有类别之间质心距离的矩阵
    centroid_distances = pairwise_distances(centroids)

    # 如果所有类别内部距离或质心距离矩阵中的元素都非常接近零，则返回 0.0 分数
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    # 将质心距离矩阵中的零元素替换为无穷大，以避免除零错误
    centroid_distances[centroid_distances == 0] = np.inf

    # 计算修正后的类别内部距离之和与质心距离的比值的最大值，作为每个类别的分数
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)

    # 返回所有类别分数的平均值作为最终的 Davies-Bouldin 分数
    return np.mean(scores)
```