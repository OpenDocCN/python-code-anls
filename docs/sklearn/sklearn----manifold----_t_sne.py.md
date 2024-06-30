# `D:\src\scipysrc\scikit-learn\sklearn\manifold\_t_sne.py`

```
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    # This is the exact and Barnes-Hut t-SNE implementation. There are other
    # modifications of the algorithm:
    # * Fast Optimization for t-SNE:
    #   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf

    # 导入警告模块，用于管理警告信息
    import warnings
    # 从 numbers 模块导入 Integral 和 Real 类，用于数值类型检查
    from numbers import Integral, Real
    # 导入 time 模块的 time 函数，用于计时
    from time import time

    # 导入 numpy 库，并指定别名 np
    import numpy as np
    # 导入 scipy 库中的 linalg 模块
    from scipy import linalg
    # 导入 scipy 库中的 sparse 模块的 csr_matrix 类和 issparse 函数
    from scipy.sparse import csr_matrix, issparse
    # 导入 scipy.spatial.distance 模块的 pdist 和 squareform 函数
    from scipy.spatial.distance import pdist, squareform

    # 从 scikit-learn 库的 base 模块导入以下类和函数
    from ..base import (
        BaseEstimator,
        ClassNamePrefixFeaturesOutMixin,
        TransformerMixin,
        _fit_context,
    )
    # 从 scikit-learn 库的 decomposition 模块导入 PCA 类
    from ..decomposition import PCA
    # 从 scikit-learn 库的 metrics.pairwise 模块导入 _VALID_METRICS 和 pairwise_distances 函数
    from ..metrics.pairwise import _VALID_METRICS, pairwise_distances
    # 从 scikit-learn 库的 neighbors 模块导入 NearestNeighbors 类
    from ..neighbors import NearestNeighbors
    # 从 scikit-learn 库的 utils 模块导入 check_random_state 函数
    from ..utils import check_random_state
    # 从 scikit-learn 库的 utils._openmp_helpers 模块导入 _openmp_effective_n_threads 函数
    from ..utils._openmp_helpers import _openmp_effective_n_threads
    # 从 scikit-learn 库的 utils._param_validation 模块导入 Hidden、Interval、StrOptions 和 validate_params 函数
    from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
    # 从 scikit-learn 库的 utils.validation 模块导入 _num_samples 和 check_non_negative 函数

    # 引入模块时的 mypy 错误信息忽略声明
    # mypy error: Module 'sklearn.manifold' has no attribute '_utils'
    # mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
    from . import _barnes_hut_tsne, _utils  # type: ignore

    # 定义一个常量，表示机器 epsilon，即浮点数的最小精度
    MACHINE_EPSILON = np.finfo(np.double).eps


    def _joint_probabilities(distances, desired_perplexity, verbose):
        """Compute joint probabilities p_ij from distances.

        Parameters
        ----------
        distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
            Distances of samples are stored as condensed matrices, i.e.
            we omit the diagonal and duplicate entries and store everything
            in a one-dimensional array.

        desired_perplexity : float
            Desired perplexity of the joint probability distributions.

        verbose : int
            Verbosity level.

        Returns
        -------
        P : ndarray of shape (n_samples * (n_samples-1) / 2,)
            Condensed joint probability matrix.
        """
        # 将距离数组转换为单精度浮点数类型，以减少内存占用
        distances = distances.astype(np.float32, copy=False)
        # 使用二分搜索算法计算条件概率，以近似达到指定的困惑度
        conditional_P = _utils._binary_search_perplexity(
            distances, desired_perplexity, verbose
        )
        # 构建对称的联合概率分布矩阵
        P = conditional_P + conditional_P.T
        # 计算概率矩阵的和，确保非负且非零
        sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
        # 归一化联合概率矩阵，确保所有概率非负且和为 1
        P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
        return P


    def _joint_probabilities_nn(distances, desired_perplexity, verbose):
        """Compute joint probabilities p_ij from distances using just nearest
        neighbors.

        This method is approximately equal to _joint_probabilities. The latter
        is O(N), but limiting the joint probability to nearest neighbors improves
        this substantially to O(uN).

        Parameters
        ----------
        distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
            Distances of samples are stored as condensed matrices, i.e.
            we omit the diagonal and duplicate entries and store everything
            in a one-dimensional array.

        desired_perplexity : float
            Desired perplexity of the joint probability distributions.

        verbose : int
            Verbosity level.
        """
    t0 = time()
    # 记录当前时间，用于计算函数执行时间

    # 对距离矩阵的非零元素进行排序，按照索引排序
    distances.sort_indices()

    # 获取样本数量
    n_samples = distances.shape[0]

    # 将距离数据重塑为二维数组，每行代表一个样本的距离信息
    distances_data = distances.data.reshape(n_samples, -1)

    # 将距离数据类型转换为 np.float32，加快计算速度
    distances_data = distances_data.astype(np.float32, copy=False)

    # 使用二分搜索计算条件概率，以匹配给定的困惑度
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose
    )

    # 检查所有的概率值是否为有限值
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # 使用稀疏矩阵操作将条件概率矩阵对称化，生成联合概率分布矩阵 P
    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T  # 对称化操作

    # 对联合概率分布进行归一化处理
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    # 断言确保所有的数据在 [-1, 1] 的范围内
    assert np.all(np.abs(P.data) <= 1.0)

    # 如果 verbosity 大于等于 2，打印计算条件概率分布所花费的时间
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))

    # 返回最终计算得到的联合概率分布矩阵 P
    return P
# t-SNE 目标函数：KL 散度的梯度，以及绝对误差。

def _kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    skip_num_points=0,
    compute_error=True,
):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    
    # 将 params 重塑为二维数组 X_embedded
    X_embedded = params.reshape(n_samples, n_components)

    # Q 是一个重尾分布：学生 t 分布
    # 计算欧氏距离的平方
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    # 归一化
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # 优化技巧：使用 np.dot(x, y) 比 np.sum(x * y) 更快，因为它调用 BLAS

    # 目标函数：C（P 和 Q 的 KL 散度）
    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # 梯度：dC/dY
    # pdist 总是返回双精度距离，因此我们需要取平方形式
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        # 计算梯度
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    # 乘以常数因子
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _kl_divergence_bh(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    angle=0.5,
    skip_num_points=0,
    verbose=False,
    compute_error=True,
    num_threads=1,
):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2).

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    angle : float, default=0.5
        Control the trade-off between speed and accuracy for Barnes-Hut.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    verbose : bool, default=False
        Verbosity level.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    num_threads : int, default=1
        Number of threads to use for the computation.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray

    Notes
    -----
    The bh-tSNE implementation includes a correction for the total mass.
    """
    # 这里省略了具体的实现部分，因为在这个示例中我们只需要注释函数的参数和文档字符串部分
    # 将参数数组转换为单精度浮点型，不复制数据
    params = params.astype(np.float32, copy=False)
    
    # 将参数数组重新形状为二维数组，表示嵌入空间中的样本数和组件数
    X_embedded = params.reshape(n_samples, n_components)

    # 将P矩阵的数据部分转换为单精度浮点型，不复制数据
    val_P = P.data.astype(np.float32, copy=False)
    
    # 将P矩阵的索引部分转换为64位整型，不复制数据
    neighbors = P.indices.astype(np.int64, copy=False)
    
    # 将P矩阵的行指针部分转换为64位整型，不复制数据
    indptr = P.indptr.astype(np.int64, copy=False)

    # 创建一个与X_embedded形状相同的全零数组，用于存储梯度
    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    
    # 调用_barnes_hut_tsne模块中的gradient函数计算梯度和误差
    error = _barnes_hut_tsne.gradient(
        val_P,
        X_embedded,
        neighbors,
        indptr,
        grad,
        angle,
        n_components,
        verbose,
        dof=degrees_of_freedom,
        compute_error=compute_error,
        num_threads=num_threads,
    )
    
    # 计算常数c，用于调整梯度的大小
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    
    # 将梯度数组展平为一维，并乘以常数c
    grad = grad.ravel()
    grad *= c

    # 返回计算得到的误差和调整后的梯度
    return error, grad
    # 如果参数 args 为 None，则设置为空列表
    if args is None:
        args = []
    # 如果参数 kwargs 为 None，则设置为空字典
    if kwargs is None:
        kwargs = {}

    # 复制初始参数向量 p0，并将其展平为一维数组
    p = p0.copy().ravel()
    # 创建一个与 p 形状相同的零数组，用于存储更新量
    update = np.zeros_like(p)
    # 创建一个与 p 形状相同的全为1的数组，用于存储每个参数的增益
    gains = np.ones_like(p)
    # 设置初始误差为浮点数最大值
    error = np.finfo(float).max
    # 设置最佳误差为浮点数最大值
    best_error = np.finfo(float).max
    # 将当前迭代次数 it 赋给 best_iter 和 i
    best_iter = i = it

    # 记录开始时间
    tic = time()
    # 迭代优化过程，从当前迭代次数到最大迭代次数之间
    for i in range(it, max_iter):
        # 检查是否需要计算误差
        check_convergence = (i + 1) % n_iter_check == 0
        # 仅在需要时计算误差
        kwargs["compute_error"] = check_convergence or i == max_iter - 1

        # 计算目标函数的误差和梯度
        error, grad = objective(p, *args, **kwargs)

        # 判断梯度的变化趋势
        inc = update * grad < 0.0
        dec = np.invert(inc)
        # 增加梯度缩放因子
        gains[inc] += 0.2
        # 减少梯度缩放因子
        gains[dec] *= 0.8
        # 限制梯度缩放因子的范围
        np.clip(gains, min_gain, np.inf, out=gains)
        # 应用梯度缩放到梯度上
        grad *= gains
        # 计算更新步长
        update = momentum * update - learning_rate * grad
        # 更新参数
        p += update

        # 如果达到检查收敛的条件
        if check_convergence:
            # 记录当前时间
            toc = time()
            # 计算迭代的持续时间
            duration = toc - tic
            # 更新起始时间
            tic = toc
            # 计算当前梯度的范数
            grad_norm = linalg.norm(grad)

            # 如果设置了详细输出
            if verbose >= 2:
                # 打印当前迭代的信息：误差、梯度范数等
                print(
                    "[t-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            # 如果当前误差优于历史最佳误差
            if error < best_error:
                # 更新最佳误差和迭代次数
                best_error = error
                best_iter = i
            # 如果已经连续多次迭代没有进展
            elif i - best_iter > n_iter_without_progress:
                # 如果设置了详细输出
                if verbose >= 2:
                    # 打印没有进展的信息，并结束迭代
                    print(
                        "[t-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                # 跳出循环，结束优化过程
                break
            # 如果梯度范数小于等于最小梯度范数
            if grad_norm <= min_grad_norm:
                # 如果设置了详细输出
                if verbose >= 2:
                    # 打印梯度范数低于阈值的信息，并结束迭代
                    print(
                        "[t-SNE] Iteration %d: gradient norm %f. Finished."
                        % (i + 1, grad_norm)
                    )
                # 跳出循环，结束优化过程
                break

    # 返回优化后的参数 p、最终的误差 error、以及最终迭代次数 i
    return p, error, i
# 使用装饰器 @validate_params 对 trustworthiness 函数进行参数验证和类型检查
@validate_params(
    # X 参数可以是 array-like 或 sparse matrix
    {
        "X": ["array-like", "sparse matrix"],
        # X_embedded 参数可以是 array-like 或 sparse matrix
        "X_embedded": ["array-like", "sparse matrix"],
        # n_neighbors 参数必须是大于等于 1 的整数
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        # metric 参数可以是预定义的度量方法集合 _VALID_METRICS 或者可调用对象
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
    },
    # 设置为 True，表示优先跳过嵌套验证
    prefer_skip_nested_validation=True,
)
# trustworthiness 函数计算嵌入空间中局部结构保留的可信度
def trustworthiness(X, X_embedded, *, n_neighbors=5, metric="euclidean"):
    r"""Indicate to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
        (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : {array-like, sparse matrix} of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, default=5
        The number of neighbors that will be considered. Should be fewer than
        `n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
        mentioned in [1]_. An error will be raised otherwise.

    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, for a list
        of available metrics, see the documentation of argument metric in
        `sklearn.pairwise.pairwise_distances` and metrics listed in
        `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
        "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.

        .. versionadded:: 0.20

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.

    References
    ----------
    .. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
           Preservation in Nonlinear Projection Methods: An Experimental Study.
           In Proceedings of the International Conference on Artificial Neural Networks
           (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.

    .. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
           Local Structure. Proceedings of the Twelfth International Conference on
           Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.

    Examples
    --------
    n_samples = _num_samples(X)
    # 计算样本数目
    if n_neighbors >= n_samples / 2:
        # 如果邻居数大于等于样本数的一半，抛出数值错误
        raise ValueError(
            f"n_neighbors ({n_neighbors}) should be less than n_samples / 2"
            f" ({n_samples / 2})"
        )
    dist_X = pairwise_distances(X, metric=metric)
    # 计算样本间的距离矩阵
    if metric == "precomputed":
        # 如果使用预先计算的距离矩阵，则复制该距离矩阵
        dist_X = dist_X.copy()
    # 将对角线设置为 np.inf，排除样本本身与自己的距离
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # 对距离矩阵的每行进行排序，ind_X[i] 是样本 i 与其他样本的距离排序后的索引

    # 在嵌入空间中找到每个样本的最近邻索引
    ind_X_embedded = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(X_embedded)
        .kneighbors(return_distance=False)
    )

    # 在输入空间中构建邻居的倒排索引：对于样本 i，定义 `inverted_index[i]` 为距离排序的倒排索引
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]

    # 计算 ranks，ranks 表示每个样本在嵌入空间中的排序位置减去邻居数
    ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
    )

    # 计算 Trustworthiness 指标 t
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )
    return t
class TSNE(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """T-distributed Stochastic Neighbor Embedding.

    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].

    Read more in the :ref:`User Guide <t_sne>`.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results. The perplexity must be less than the number
        of samples.

    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    learning_rate : float or "auto", default="auto"
        学习率用于 t-SNE 算法，通常在 [10.0, 1000.0] 的范围内。如果学习率过高，数据可能呈现球状，使得任意点几乎等距离其最近邻。如果学习率过低，大多数点可能被压缩在一个密集的云中，且少数离群点。如果成本函数陷入不良局部最小值，增加学习率可能有所帮助。
        需要注意的是，其他许多 t-SNE 实现（如 bhtsne、FIt-SNE、openTSNE 等）的学习率定义比我们的小四倍。因此，我们的 learning_rate=200 对应于这些实现中的 learning_rate=800。
        "auto" 选项将学习率设置为 `max(N / early_exaggeration / 4, 50)`，其中 N 是样本大小，参考文献 [4] 和 [5]。

        .. versionchanged:: 1.2
           默认值更改为 `"auto"`.

    max_iter : int, default=1000
        最大迭代次数，用于优化过程。应至少为 250。

        .. versionchanged:: 1.5
            参数名称从 `n_iter` 更改为 `max_iter`。

    n_iter_without_progress : int, default=300
        在优化过程中，若经过了 250 次初始迭代（带有早期夸张），则最多允许多少次迭代没有进展便终止优化。注意，进展仅每 50 次迭代检查一次，因此该值会向上取整至最接近的 50 的倍数。

        .. versionadded:: 0.17
           参数 *n_iter_without_progress* 用于控制停止标准。

    min_grad_norm : float, default=1e-7
        如果梯度的范数低于此阈值，则停止优化过程。

    metric : str or callable, default='euclidean'
        在特征数组中计算实例之间距离时使用的度量方式。如果 metric 是一个字符串，它必须是 scipy.spatial.distance.pdist 中允许的选项之一，或者是 pairwise.PAIRWISE_DISTANCE_FUNCTIONS 中列出的度量方式之一。
        如果 metric 是 "precomputed"，则假定 X 是一个距离矩阵。
        或者，如果 metric 是一个可调用函数，则它将对每对实例（行）调用，并记录返回的距离值。可调用函数应接受两个来自 X 的数组作为输入，并返回表示它们之间距离的值。默认值是 "euclidean"，即被解释为平方欧几里得距离。

    metric_params : dict, default=None
        度量函数的额外关键字参数。

        .. versionadded:: 1.1
    init : {"random", "pca"} or ndarray of shape (n_samples, n_components), \
            default="pca"
        # 初始化嵌入的方法。可以选择使用 "random" 或 "pca"，或者直接提供一个形状为 (n_samples, n_components) 的数组作为初始化。
        # 默认为 "pca"。使用 PCA 初始化不能与预先计算的距离一起使用，并且通常比随机初始化更加全局稳定。

        .. versionchanged:: 1.2
           # 默认值已更改为 `"pca"`。

    verbose : int, default=0
        # 冗长级别。控制输出详细程度。

    random_state : int, RandomState instance or None, default=None
        # 确定随机数生成器。传递一个整数以确保多次函数调用时得到可复现的结果。
        # 不同的初始化可能导致成本函数的不同局部最小值。参见“术语表”。

    method : {'barnes_hut', 'exact'}, default='barnes_hut'
        # 梯度计算算法的方法选择。默认使用 Barnes-Hut 近似算法，运行时间复杂度为 O(NlogN)。
        # method='exact' 使用精确但更慢的 O(N^2) 算法。当需要比 3% 更好的最近邻误差时应使用精确方法。
        # 然而，精确方法无法扩展到数百万个示例。

        .. versionadded:: 0.17
           # 通过 Barnes-Hut 方法进行近似优化。

    angle : float, default=0.5
        # 仅当 method='barnes_hut' 时有效。这是 Barnes-Hut T-SNE 在速度和准确性之间的权衡。
        # 'angle' 是远程节点的角度大小（在文献中称为 theta）。如果节点的大小小于 'angle'，则将其用作其中包含的所有点的摘要节点。
        # 在 0.2 - 0.8 范围内，该方法对该参数的变化不太敏感。小于 0.2 的角度会快速增加计算时间，而大于 0.8 的角度会快速增加误差。

    n_jobs : int, default=None
        # 用于邻居搜索的并行作业数。当 `metric="precomputed"` 或 (`metric="euclidean"` 和 `method="exact"`) 时，此参数无影响。
        # `None` 表示使用 1 个处理器，除非在 `joblib.parallel_backend` 上下文中。
        # `-1` 表示使用所有处理器。参见“术语表”以获取更多详细信息。

        .. versionadded:: 0.22

    n_iter : int
        # 优化的最大迭代次数。应至少为 250。

        .. deprecated:: 1.5
            # `n_iter` 已在版本 1.5 中弃用，并将在 1.7 中移除。
            # 请改用 `max_iter`。

    Attributes
    ----------
    embedding_ : array-like of shape (n_samples, n_components)
        # 存储嵌入向量。

    kl_divergence_ : float
        # 优化后的 Kullback-Leibler 散度。

    n_features_in_ : int
        # 在拟合过程中看到的特征数。

        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0
        特征名称数组，形状为 (`n_features_in_`,)，记录了在 `fit` 过程中观察到的特征名称。仅在 `X` 中所有特征名称均为字符串时定义。

    learning_rate_ : float
        Effective learning rate.

        .. versionadded:: 1.2
        有效的学习率，用于控制优化算法在每次迭代中更新权重时的步长。

    n_iter_ : int
        Number of iterations run.
        运行的迭代次数，即模型训练过程中完成的总迭代次数。

    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis that is a linear
        dimensionality reduction method.
    sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
        kernels and PCA.
    MDS : Manifold learning using multidimensional scaling.
    Isomap : Manifold learning based on Isometric Mapping.
    LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
    SpectralEmbedding : Spectral embedding for non-linear dimensionality.

    Notes
    -----
    For an example of using :class:`~sklearn.manifold.TSNE` in combination with
    :class:`~sklearn.neighbors.KNeighborsTransformer` see
    :ref:`sphx_glr_auto_examples_neighbors_approximate_nearest_neighbors.py`.

    References
    ----------

    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/

    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf

    [4] Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J.,
        & Snyder-Cappione, J. E. (2019). Automated optimized parameters for
        T-distributed stochastic neighbor embedding improve visualization
        and analysis of large datasets. Nature Communications, 10(1), 1-12.

    [5] Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell
        transcriptomics. Nature Communications, 10(1), 1-14.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2, learning_rate='auto',
    ...                   init='random', perplexity=3).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    """
    # 参数约束字典，指定了各个参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "perplexity": [Interval(Real, 0, None, closed="neither")],
        "early_exaggeration": [Interval(Real, 1, None, closed="left")],
        "learning_rate": [
            StrOptions({"auto"}),
            Interval(Real, 0, None, closed="neither"),
        ],
        "max_iter": [Interval(Integral, 250, None, closed="left"), None],
        "n_iter_without_progress": [Interval(Integral, -1, None, closed="left")],
        "min_grad_norm": [Interval(Real, 0, None, closed="left")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "init": [
            StrOptions({"pca", "random"}),
            np.ndarray,
        ],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "method": [StrOptions({"barnes_hut", "exact"})],
        "angle": [Interval(Real, 0, 1, closed="both")],
        "n_jobs": [None, Integral],
        "n_iter": [
            Interval(Integral, 250, None, closed="left"),
            Hidden(StrOptions({"deprecated"})),
        ],
    }

    # 控制使用 early_exaggeration 参数时的最大迭代次数
    _EXPLORATION_MAX_ITER = 250

    # 控制在每次进度检查之间的迭代次数
    _N_ITER_CHECK = 50

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=None,  # TODO(1.7): set to 1000
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        init="pca",
        verbose=0,
        random_state=None,
        method="barnes_hut",
        angle=0.5,
        n_jobs=None,
        n_iter="deprecated",
    ):
        # 初始化 t-SNE 参数
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.n_iter = n_iter

    def _check_params_vs_input(self, X):
        # 检查 perplexity 参数是否小于样本数，如果不是则抛出 ValueError
        if self.perplexity >= X.shape[0]:
            raise ValueError("perplexity must be less than n_samples")

    def _tsne(
        self,
        P,
        degrees_of_freedom,
        n_samples,
        X_embedded,
        neighbors=None,
        skip_num_points=0,
    ):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        
        # 将输入数据 X_embedded 摊平成一维数组，作为优化参数的初始值
        params = X_embedded.ravel()

        # 设置优化参数的初始值
        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate_,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_MAX_ITER,
            "max_iter": self._EXPLORATION_MAX_ITER,
            "momentum": 0.5,
        }
        
        # 如果使用 Barnes-Hut 方法，则使用对应的目标函数和设置角度参数
        if self.method == "barnes_hut":
            obj_func = _kl_divergence_bh
            opt_args["kwargs"]["angle"] = self.angle
            opt_args["kwargs"]["verbose"] = self.verbose
            # 获取用于梯度计算的线程数，以避免在每次迭代时重新计算
            opt_args["kwargs"]["num_threads"] = _openmp_effective_n_threads()
        else:
            obj_func = _kl_divergence

        # 学习时间表（第一部分）：使用较低的动量和通过早期夸大参数控制的更高学习率进行 250 次迭代
        P *= self.early_exaggeration
        
        # 使用梯度下降方法进行优化，返回更新后的参数、KL 散度和迭代次数
        params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
        
        # 如果 verbose 为 True，则打印迭代后的 KL 散度
        if self.verbose:
            print(
                "[t-SNE] KL divergence after %d iterations with early exaggeration: %f"
                % (it + 1, kl_divergence)
            )

        # 学习时间表（第二部分）：取消早期夸大，使用动量 0.8 完成优化
        P /= self.early_exaggeration
        remaining = self._max_iter - self._EXPLORATION_MAX_ITER
        
        # 如果迭代次数小于探索最大迭代次数或仍有剩余迭代次数，则继续优化
        if it < self._EXPLORATION_MAX_ITER or remaining > 0:
            opt_args["max_iter"] = self._max_iter
            opt_args["it"] = it + 1
            opt_args["momentum"] = 0.8
            opt_args["n_iter_without_progress"] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)

        # 保存最终的迭代次数
        self.n_iter_ = it
        
        # 如果 verbose 为 True，则打印最终迭代后的 KL 散度
        if self.verbose:
            print(
                "[t-SNE] KL divergence after %d iterations: %f"
                % (it + 1, kl_divergence)
            )

        # 将更新后的参数重新整形为原始形状的二维数组 X_embedded
        X_embedded = params.reshape(n_samples, self.n_components)
        # 保存计算得到的 KL 散度值
        self.kl_divergence_ = kl_divergence

        # 返回降维后的嵌入空间
        return X_embedded

    @_fit_context(
        # TSNE.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        # TODO(1.7): remove
        # 设置 `_max_iter` 属性以便在迭代次数非废弃时使用
        if self.n_iter != "deprecated":
            if self.max_iter is not None:
                raise ValueError(
                    "Both 'n_iter' and 'max_iter' attributes were set. Attribute"
                    " 'n_iter' was deprecated in version 1.5 and will be removed in"
                    " 1.7. To avoid this error, only set the 'max_iter' attribute."
                )
            # 发出警告，提示用户 'n_iter' 已废弃并将在未来版本移除
            warnings.warn(
                (
                    "'n_iter' was renamed to 'max_iter' in version 1.5 and "
                    "will be removed in 1.7."
                ),
                FutureWarning,
            )
            self._max_iter = self.n_iter
        elif self.max_iter is None:
            # 如果 `max_iter` 为 None，则设置默认值为 1000
            self._max_iter = 1000
        else:
            # 否则使用给定的 `max_iter` 值
            self._max_iter = self.max_iter

        # 检查参数与输入数据的一致性
        self._check_params_vs_input(X)
        # 执行数据拟合操作，并获取嵌入结果
        embedding = self._fit(X)
        # 将嵌入结果存储在 `embedding_` 属性中
        self.embedding_ = embedding
        # 返回嵌入结果
        return self.embedding_

    @_fit_context(
        # TSNE.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 调用 `fit_transform` 方法来执行拟合与转换操作
        self.fit_transform(X)
        # 返回自身实例
        return self

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # 返回嵌入结果的特征数
        return self.embedding_.shape[1]

    def _more_tags(self):
        # 返回一个字典，指示是否为成对数据的度量值
        return {"pairwise": self.metric == "precomputed"}
```