# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_affinity_propagation.py`

```
# 导入警告模块，用于显示警告信息
import warnings
# 导入整数和实数类型，用于参数类型验证
from numbers import Integral, Real

# 导入NumPy库，并命名为np，用于科学计算
import numpy as np

# 导入配置上下文管理器，用于处理配置参数
from .._config import config_context
# 导入基础估计器、集群混合器和拟合上下文管理器
from ..base import BaseEstimator, ClusterMixin, _fit_context
# 导入收敛警告异常类
from ..exceptions import ConvergenceWarning
# 导入欧几里得距离计算函数和寻找最小距离的函数
from ..metrics import euclidean_distances, pairwise_distances_argmin
# 导入随机状态检查函数
from ..utils import check_random_state
# 导入参数验证相关工具：区间、字符串选项和参数验证函数
from ..utils._param_validation import Interval, StrOptions, validate_params
# 导入检查是否拟合函数，用于验证模型是否已经训练
from ..utils.validation import check_is_fitted


def _equal_similarities_and_preferences(S, preference):
    def all_equal_preferences():
        # 检查偏好是否全部相等
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # 创建掩码以忽略S的对角线
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        # 检查相似性是否全部相等
        return np.all(S[mask].flat == S[mask].flat[0])

    return all_equal_preferences() and all_equal_similarities()


def _affinity_propagation(
    S,
    *,
    preference,
    convergence_iter,
    max_iter,
    damping,
    verbose,
    return_n_iter,
    random_state,
):
    """Main affinity propagation algorithm."""
    # 获取样本数量
    n_samples = S.shape[0]
    # 如果样本数为1或者偏好和相似性全相等，则直接返回结果
    if n_samples == 1 or _equal_similarities_and_preferences(S, preference):
        warnings.warn(
            "All samples have mutually equal similarities. "
            "Returning arbitrary cluster center(s)."
        )
        # 根据偏好的值来决定返回的结果
        if preference.flat[0] > S.flat[n_samples - 1]:
            return (
                (np.arange(n_samples), np.arange(n_samples), 0)
                if return_n_iter
                else (np.arange(n_samples), np.arange(n_samples))
            )
        else:
            return (
                (np.array([0]), np.array([0] * n_samples), 0)
                if return_n_iter
                else (np.array([0]), np.array([0] * n_samples))
            )

    # 将偏好值放在S的对角线上
    S.flat[:: (n_samples + 1)] = preference

    # 初始化消息传递矩阵A和R
    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # 中间结果存储数组
    tmp = np.zeros((n_samples, n_samples))

    # 移除退化情况
    # 添加随机噪声，以防止退化情况
    S += (
        np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100
    ) * random_state.standard_normal(size=(n_samples, n_samples))

    # 执行并行的亲和传播更新
    e = np.zeros((n_samples, convergence_iter))

    # 初始化索引数组
    ind = np.arange(n_samples)
    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)  # 计算 A + S，将结果存入 tmp
        I = np.argmax(tmp, axis=1)  # 沿着每行找到最大值的索引，存入 I
        Y = tmp[ind, I]  # 获取 tmp 中指定索引位置的值，存入 Y
        tmp[ind, I] = -np.inf  # 将 tmp 中指定位置的值设为负无穷

        Y2 = np.max(tmp, axis=1)  # 沿着每行找到 tmp 的最大值，存入 Y2

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)  # 计算 S - Y[:, None]，结果存入 tmp
        tmp[ind, I] = S[ind, I] - Y2  # 更新 tmp 中指定位置的值

        # Damping
        tmp *= 1 - damping  # 对 tmp 应用阻尼系数
        R *= damping  # 对 R 应用阻尼系数
        R += tmp  # 更新 R

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)  # 计算 R 和 0 的元素最大值，结果存入 tmp
        tmp.flat[:: n_samples + 1] = R.flat[:: n_samples + 1]  # 将 tmp 中每隔 n_samples + 1 个元素的值设为 R 中对应位置的值

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)  # 对 tmp 按列求和，并将结果减去 tmp 自身
        dA = np.diag(tmp).copy()  # 获取 tmp 对角线上的值，并复制到 dA
        tmp.clip(0, np.inf, tmp)  # 将 tmp 中的值裁剪到区间 [0, 正无穷)
        tmp.flat[:: n_samples + 1] = dA  # 将 tmp 中每隔 n_samples + 1 个元素的值设为 dA 中对应位置的值

        # Damping
        tmp *= 1 - damping  # 对 tmp 应用阻尼系数
        A *= damping  # 对 A 应用阻尼系数
        A -= tmp  # 更新 A

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0  # 检查是否收敛的条件，结果存入 E
        e[:, it % convergence_iter] = E  # 将 E 存入 e 的第 it % convergence_iter 列
        K = np.sum(E, axis=0)  # 沿着列求和，得到每列中 True 的个数

        if it >= convergence_iter:
            se = np.sum(e, axis=1)  # 沿着行求和，得到每行中 True 的个数
            unconverged = np.sum((se == convergence_iter) + (se == 0)) != n_samples  # 检查是否有未收敛的样本
            if (not unconverged and (K > 0)) or (it == max_iter):
                never_converged = False
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        never_converged = True
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)  # 获取 E 中非零元素的索引，存入 I
    K = I.size  # 获取 I 的大小，即确定的中心数量

    if K > 0:
        if never_converged:
            warnings.warn(
                (
                    "Affinity propagation did not converge, this model "
                    "may return degenerate cluster centers and labels."
                ),
                ConvergenceWarning,
            )
        c = np.argmax(S[:, I], axis=1)  # 在 S 的指定列中找到最大值的索引，存入 c
        c[I] = np.arange(K)  # 将 c 中的特定索引位置赋值为 0 到 K-1 的值，确定聚类
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]  # 获取 c 中等于 k 的索引位置
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))  # 在 S 的子集中找到总和最大的索引
            I[k] = ii[j]  # 更新 I 中的值

        c = np.argmax(S[:, I], axis=1)  # 重新计算 c
        c[I] = np.arange(K)  # 再次确定聚类
        labels = I[c]  # 根据 c 确定标签
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)  # 获取唯一的聚类中心索引
        labels = np.searchsorted(cluster_centers_indices, labels)  # 将 labels 缩减到排序的、无间隔的列表
    else:
        warnings.warn(
            (
                "Affinity propagation did not converge and this model "
                "will not have any cluster centers."
            ),
            ConvergenceWarning,
        )
        labels = np.array([-1] * n_samples)  # 如果 K <= 0，则返回未分类的标签数组
        cluster_centers_indices = []  # 如果 K <= 0，则返回空的聚类中心索引数组

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1  # 如果需要返回迭代次数，则返回聚类中心索引、标签和迭代次数
    else:
        return cluster_centers_indices, labels  # 否则，只返回聚类中心索引和标签
###############################################################################
# Public API

# 定义一个装饰器，用于验证参数类型
@validate_params(
    {
        "S": ["array-like"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
def affinity_propagation(
    S,
    *,
    preference=None,  # 每个点的偏好值或者偏好值数组，影响簇的数量
    convergence_iter=15,  # 收敛时停止迭代的无变化迭代次数
    max_iter=200,  # 最大迭代次数
    damping=0.5,  # 阻尼系数，控制收敛速度
    copy=True,  # 是否复制亲和力矩阵以便算法修改，以节省内存
    verbose=False,  # 冗余度级别
    return_n_iter=False,  # 是否返回迭代次数
    random_state=None,  # 随机数种子
):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    S : array-like of shape (n_samples, n_samples)
        Matrix of similarities between points.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, default=200
        Maximum number of iterations.

    damping : float, default=0.5
        Damping factor between 0.5 and 1.

    copy : bool, default=True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency.

    verbose : bool, default=False
        The verbosity level.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Returns
    -------
    cluster_centers_indices : ndarray of shape (n_clusters,)
        Index of clusters centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    When the algorithm does not converge, it will still return a arrays of
    ``cluster_center_indices`` and labels if there are any exemplars/clusters,
    however they may be degenerate and should be used with caution.
    """
    """
    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import affinity_propagation
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> S = -euclidean_distances(X, squared=True)
    >>> cluster_centers_indices, labels = affinity_propagation(S, random_state=0)
    >>> cluster_centers_indices
    array([0, 3])
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    """
    # 使用 Affinity Propagation 算法进行聚类
    estimator = AffinityPropagation(
        damping=damping,  # 阻尼因子
        max_iter=max_iter,  # 最大迭代次数
        convergence_iter=convergence_iter,  # 收敛迭代次数
        copy=copy,  # 是否复制输入数据
        preference=preference,  # 首选项参数，影响簇的数量和大小
        affinity="precomputed",  # 使用预先计算的相似性矩阵
        verbose=verbose,  # 是否输出详细信息
        random_state=random_state,  # 随机数种子
    ).fit(S)  # 对相似性矩阵 S 进行聚类

    if return_n_iter:
        # 如果需要返回迭代次数，则返回簇中心索引、标签和迭代次数
        return estimator.cluster_centers_indices_, estimator.labels_, estimator.n_iter_
    # 否则，只返回簇中心索引和标签
    return estimator.cluster_centers_indices_, estimator.labels_
# 定义 Affinity Propagation 聚类算法类，继承自 ClusterMixin 和 BaseEstimator
class AffinityPropagation(ClusterMixin, BaseEstimator):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, default=0.5
        阻尼因子，范围在 `[0.5, 1.0)`，用于控制当前值相对于传入值的维持程度
        （按权重 1 - damping）。这是为了在更新这些值（消息）时避免数值振荡。

    max_iter : int, default=200
        最大迭代次数。

    convergence_iter : int, default=15
        在没有估计集群数变化的迭代次数，用于停止收敛。

    copy : bool, default=True
        是否复制输入数据。

    preference : array-like of shape (n_samples,) or float, default=None
        每个点的偏好值 - 具有较大偏好值的点更有可能被选择为样本点。
        偏好值的大小影响输入偏好值确定的样本点数量。如果没有传入偏好值作为参数，
        则将其设置为输入相似性的中位数。

    affinity : {'euclidean', 'precomputed'}, default='euclidean'
        使用的相似度度量。目前支持 'precomputed' 和 'euclidean'。
        'euclidean' 使用点之间的负平方欧氏距离。

    verbose : bool, default=False
        是否输出详细信息。

    random_state : int, RandomState instance or None, default=None
        伪随机数生成器，用于控制起始状态。使用整数可实现函数调用间的可重复结果。
        参见 :term:`Glossary <random_state>`。

        .. versionadded:: 0.23
            此参数先前硬编码为 0。

    Attributes
    ----------
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        聚类中心的索引。

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        聚类中心（如果 affinity != 'precomputed'）。

    labels_ : ndarray of shape (n_samples,)
        每个点的标签。

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        存储在 ``fit`` 中使用的相似度矩阵。

    n_iter_ : int
        收敛所用的迭代次数。

    n_features_in_ : int
        在 `fit` 过程中观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 `X` 具有所有特征名均为字符串时定义的特征名称。

        .. versionadded:: 1.0

    See Also
    --------
    AgglomerativeClustering : 递归地合并最小增加给定链接距离的一对集群。
    FeatureAgglomeration : 类似于 AgglomerativeClustering，但递归地合并特征而不是样本。

    """
    # 定义 AffinityPropagation 类，用于亲和传播聚类算法
    KMeans : K-Means clustering.
    # KMeans：K均值聚类算法
    MiniBatchKMeans : Mini-Batch K-Means clustering.
    # MiniBatchKMeans：小批量K均值聚类算法
    MeanShift : Mean shift clustering using a flat kernel.
    # MeanShift：使用平坦核函数的均值漂移聚类算法
    SpectralClustering : Apply clustering to a projection
        of the normalized Laplacian.
    # SpectralClustering：对归一化拉普拉斯的投影应用聚类算法

    Notes
    -----
    # 注意事项部分开始
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.
    # 有关示例，请参阅示例代码 examples/cluster/plot_affinity_propagation.py。

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.
    # 亲和传播算法的复杂度随数据点数目的增加而呈二次方增长。

    When the algorithm does not converge, it will still return a arrays of
    ``cluster_center_indices`` and labels if there are any exemplars/clusters,
    however they may be degenerate and should be used with caution.
    # 当算法不收敛时，如果有实例/聚类，仍会返回“cluster_center_indices”和标签数组，
    # 但它们可能是退化的，应谨慎使用。

    When ``fit`` does not converge, ``cluster_centers_`` is still populated
    however it may be degenerate. In such a case, proceed with caution.
    # 当“fit”方法不收敛时，“cluster_centers_”仍然被填充，但它可能是退化的。在这种情况下，请谨慎处理。

    If ``fit`` does not converge and fails to produce any ``cluster_centers_``
    then ``predict`` will label every sample as ``-1``.
    # 如果“fit”方法不收敛且未生成任何“cluster_centers_”，则“predict”方法将所有样本标记为“-1”。

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    # 当所有训练样本具有相等的相似性和偏好时，聚类中心和标签的分配取决于偏好值。

    References
    ----------
    # 参考文献部分开始
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    # Brendan J. Frey 和 Delbert Dueck 的文章《通过数据点之间的消息传递进行聚类》，2007年2月。

    Examples
    --------
    # 示例部分开始
    >>> from sklearn.cluster import AffinityPropagation
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AffinityPropagation(random_state=5).fit(X)
    >>> clustering
    AffinityPropagation(random_state=5)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.predict([[0, 0], [4, 4]])
    array([0, 1])
    >>> clustering.cluster_centers_
    array([[1, 2],
           [4, 2]])
    """

    # 参数约束字典定义
    _parameter_constraints: dict = {
        "damping": [Interval(Real, 0.5, 1.0, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "convergence_iter": [Interval(Integral, 1, None, closed="left")],
        "copy": ["boolean"],
        "preference": [
            "array-like",
            Interval(Real, None, None, closed="neither"),
            None,
        ],
        "affinity": [StrOptions({"euclidean", "precomputed"})],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        damping=0.5,
        max_iter=200,
        convergence_iter=15,
        copy=True,
        preference=None,
        affinity="euclidean",
        verbose=False,
        random_state=None,
        # 初始化方法，设置亲和传播聚类算法的参数
    ):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state



        # 将传入的参数分配给对应的对象属性
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state



    def _more_tags(self):
        # 返回一个字典，指明是否是成对的（pairwise），根据 affinity 是否为 "precomputed" 判断
        return {"pairwise": self.affinity == "precomputed"}



    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the clustering from features, or affinity matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Returns the instance itself.
        """
        if self.affinity == "precomputed":
            # 验证输入数据 X，确保其可用性并且可写（force_writeable=True）
            X = self._validate_data(X, copy=self.copy, force_writeable=True)
            # 将验证后的 X 赋给 affinity_matrix_
            self.affinity_matrix_ = X
        else:  # self.affinity == "euclidean"
            # 验证输入数据 X，接受稀疏矩阵并转换为 csr_matrix
            X = self._validate_data(X, accept_sparse="csr")
            # 计算欧氏距离的平方，并赋给 affinity_matrix_
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)

        if self.affinity_matrix_.shape[0] != self.affinity_matrix_.shape[1]:
            # 如果相似度矩阵不是方阵，则抛出 ValueError
            raise ValueError(
                "The matrix of similarities must be a square array. "
                f"Got {self.affinity_matrix_.shape} instead."
            )

        if self.preference is None:
            # 如果未指定 preference，则使用 affinity_matrix_ 的中位数作为 preference
            preference = np.median(self.affinity_matrix_)
        else:
            preference = self.preference
        preference = np.asarray(preference)

        # 检查并设置随机种子
        random_state = check_random_state(self.random_state)

        (
            self.cluster_centers_indices_,
            self.labels_,
            self.n_iter_,
        ) = _affinity_propagation(
            self.affinity_matrix_,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            preference=preference,
            damping=self.damping,
            verbose=self.verbose,
            return_n_iter=True,
            random_state=random_state,
        )

        if self.affinity != "precomputed":
            # 如果不是预计算的相似度，根据 cluster_centers_indices_ 获取 cluster_centers_
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self



        # 注释结束
    # 预测每个样本属于哪个最近的簇。
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 验证数据格式并可能转换为稀疏矩阵格式
        X = self._validate_data(X, reset=False, accept_sparse="csr")
        # 如果模型的属性中没有簇中心，则抛出错误
        if not hasattr(self, "cluster_centers_"):
            raise ValueError(
                "Predict method is not supported when affinity='precomputed'."
            )

        # 如果簇中心存在，则进行预测
        if self.cluster_centers_.shape[0] > 0:
            # 使用 assume_finite=True 的上下文环境计算 X 与簇中心之间的距离，并返回最近簇的索引
            with config_context(assume_finite=True):
                return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            # 如果模型没有簇中心，则发出警告，并将所有样本标记为 '-1'
            warnings.warn(
                (
                    "This model does not have any cluster centers "
                    "because affinity propagation did not converge. "
                    "Labeling every sample as '-1'."
                ),
                ConvergenceWarning,
            )
            return np.array([-1] * X.shape[0])

    # 拟合聚类并返回聚类标签
    def fit_predict(self, X, y=None):
        """Fit clustering from features/affinity matrix; return cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        # 调用父类的 fit_predict 方法进行聚类拟合和预测
        return super().fit_predict(X, y)
```