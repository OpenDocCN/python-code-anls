# `D:\src\scipysrc\scikit-learn\sklearn\manifold\_mds.py`

```
"""
Multi-dimensional Scaling (MDS).
"""

# author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

# 引入警告模块，用于可能的警告处理
import warnings
# 引入数值处理模块中的整数和实数类型
from numbers import Integral, Real

# 引入NumPy库，用于数值计算
import numpy as np
# 从joblib库中导入并行处理工作的函数
from joblib import effective_n_jobs

# 从scikit-learn基础模块中导入基本估计器和拟合上下文
from ..base import BaseEstimator, _fit_context
# 从scikit-learn中引入保序回归模块
from ..isotonic import IsotonicRegression
# 从scikit-learn中引入欧氏距离计算函数
from ..metrics import euclidean_distances
# 从scikit-learn工具模块中引入数据检查、随机数检查和对称性检查函数
from ..utils import check_array, check_random_state, check_symmetric
# 从scikit-learn工具参数验证模块中引入区间、字符串选项和参数验证函数
from ..utils._param_validation import Interval, StrOptions, validate_params
# 从scikit-learn并行处理模块中导入并行处理、延迟执行函数
from ..utils.parallel import Parallel, delayed


def _smacof_single(
    dissimilarities,
    metric=True,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    normalized_stress=False,
):
    """Computes multidimensional scaling using SMACOF algorithm.

    Parameters
    ----------
    dissimilarities : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    normalized_stress : bool, default=False
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS. The
        caller must ensure that if `normalized_stress=True` then `metric=False`

        .. versionadded:: 1.2

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.
    """
    # 检查并确保输入的距离矩阵 `dissimilarities` 是对称的，如果不是则会抛出异常
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)

    # 获取样本数量
    n_samples = dissimilarities.shape[0]

    # 检查并设置随机数生成器
    random_state = check_random_state(random_state)

    # 将距离矩阵转换为扁平化的相似度数组，排除对角线上的元素
    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()

    # 过滤掉相似度数组中为零的元素
    sim_flat_w = sim_flat[sim_flat != 0]

    if init is None:
        # 如果未提供初始配置，则随机选择初始点配置
        X = random_state.uniform(size=n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # 如果提供了初始配置，则使用提供的初始化矩阵
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)" % (n_samples, n_components)
            )
        X = init

    # 初始化旧的应力值为 None
    old_stress = None

    # 创建保序回归对象
    ir = IsotonicRegression()
    for it in range(max_iter):
        # 对于每次迭代，执行以下步骤：

        # 计算样本点之间的欧氏距离
        dis = euclidean_distances(X)

        # 如果使用自定义的度量方式，则使用预先计算的不相似度
        if metric:
            disparities = dissimilarities
        else:
            # 将二维距离矩阵展平为一维数组
            dis_flat = dis.ravel()
            # 从展平的距离数组中排除与0相似的值，将其视为缺失值
            dis_flat_w = dis_flat[sim_flat != 0]

            # 使用单调回归计算不相似度
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            # 将计算得到的不相似度复制回原始数组中
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            # 将一维数组转换回二维距离矩阵的形状
            disparities = disparities.reshape((n_samples, n_samples))
            # 根据公式缩放不相似度矩阵
            disparities *= np.sqrt(
                (n_samples * (n_samples - 1) / 2) / (disparities**2).sum()
            )

        # 计算应力值
        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
        # 如果需要归一化应力值，则进行归一化处理
        if normalized_stress:
            stress = np.sqrt(stress / ((disparities.ravel() ** 2).sum() / 2))

        # 使用古特曼变换更新样本点的位置矩阵 X
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = -ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1.0 / n_samples * np.dot(B, X)

        # 计算更新后的样本点间的新的欧氏距离
        dis = np.sqrt((X**2).sum(axis=1)).sum()

        # 如果需要输出详细信息，则打印当前迭代的应力值
        if verbose >= 2:
            print("it: %d, stress %s" % (it, stress))

        # 如果存在旧的应力值，并且当前迭代的应力减小值小于给定阈值 eps，则终止迭代
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                if verbose:
                    print("breaking at iteration %d with stress %s" % (it, stress))
                break

        # 更新旧的应力值
        old_stress = stress / dis

    # 返回更新后的样本点位置矩阵 X，最终的应力值 stress，以及迭代次数 it + 1
    return X, stress, it + 1
# 使用装饰器 validate_params 进行参数验证，确保输入参数符合指定的类型和条件
@validate_params(
    {
        "dissimilarities": ["array-like"],  # 参数 dissimilarities 应为数组或类数组类型
        "metric": ["boolean"],  # 参数 metric 应为布尔类型
        "n_components": [Interval(Integral, 1, None, closed="left")],  # 参数 n_components 应为大于等于1的整数
        "init": ["array-like", None],  # 参数 init 应为数组或类数组类型，或者为 None
        "n_init": [Interval(Integral, 1, None, closed="left")],  # 参数 n_init 应为大于等于1的整数
        "n_jobs": [Integral, None],  # 参数 n_jobs 应为整数或 None
        "max_iter": [Interval(Integral, 1, None, closed="left")],  # 参数 max_iter 应为大于等于1的整数
        "verbose": ["verbose"],  # 参数 verbose 应为 verbose 类型
        "eps": [Interval(Real, 0, None, closed="left")],  # 参数 eps 应为大于等于0的实数
        "random_state": ["random_state"],  # 参数 random_state 应为 random_state 类型
        "return_n_iter": ["boolean"],  # 参数 return_n_iter 应为布尔类型
        "normalized_stress": ["boolean", StrOptions({"auto"})],  # 参数 normalized_stress 应为布尔类型或者字符串 'auto'
    },
    prefer_skip_nested_validation=True,  # 设置为 True，表示优先跳过嵌套验证
)
def smacof(
    dissimilarities,
    *,
    metric=True,  # 默认为 True，表示使用度量型 SMACOF 算法
    n_components=2,  # 默认为 2，表示嵌入空间的维数
    init=None,  # 默认为 None，表示不使用指定的初始化配置
    n_init=8,  # 默认为 8，表示运行算法的初始化次数
    n_jobs=None,  # 默认为 None，表示不使用并行工作
    max_iter=300,  # 默认为 300，表示算法的最大迭代次数
    verbose=0,  # 默认为 0，表示不输出冗长信息
    eps=1e-3,  # 默认为 1e-3，表示算法收敛的容忍度
    random_state=None,  # 默认为 None，表示不使用随机状态
    return_n_iter=False,  # 默认为 False，表示不返回迭代次数
    normalized_stress="auto",  # 默认为 'auto'，表示自动选择是否归一化 stress
):
    """Compute multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can be summarized by the following
    steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    dissimilarities : array-like of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : array-like of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, default=8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.
    n_jobs : int, default=None
        # 定义并行计算的作业数量。如果使用多个初始化（`n_init`），则每次算法运行都会并行计算。

        ``None`` 表示使用一个作业，除非在 :obj:`joblib.parallel_backend` 上下文中。
        ``-1`` 表示使用所有处理器。详细信息请参见 :term:`Glossary <n_jobs>`。

    max_iter : int, default=300
        # 单次运行SMACOF算法的最大迭代次数。

    verbose : int, default=0
        # 冗长输出的级别。

    eps : float, default=1e-3
        # 与应该在其收敛时相对容差有关的应力。`eps` 的值应根据是否使用 `normalized_stress` 进行调整。

    random_state : int, RandomState instance or None, default=None
        # 确定用于初始化中心的随机数生成器。传递一个整数以在多次函数调用之间获得可重复的结果。
        # 请参见 :term:`Glossary <random_state>`。

    return_n_iter : bool, default=False
        # 是否返回迭代次数。

    normalized_stress : bool or "auto" default="auto"
        # 是否使用和返回归一化应力值（Stress-1），而不是默认计算的原始应力。仅在非度量MDS中支持。

        .. versionadded:: 1.2

        .. versionchanged:: 1.4
           默认值从版本1.4中的 `False` 更改为 `"auto"`。

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        # 在 `n_components` 空间中点的坐标。

    stress : float
        # 应力的最终值（所有约束点的差异和距离的平方和）。
        # 如果 `normalized_stress=True`，且 `metric=False`，则返回 Stress-1。
        # 值为0表示“完美”拟合，0.025表示优秀，0.05表示良好，0.1表示一般，0.2表示差 [1]_。

    n_iter : int
        # 对应最佳应力的迭代次数。仅当 `return_n_iter` 设置为 `True` 时返回。

    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
           Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
           hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
           Groenen P. Springer Series in Statistics (1997)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import smacof
    >>> from sklearn.metrics import euclidean_distances
    >>> X = np.array([[0, 1, 2], [1, 0, 3],[2, 3, 0]])
    >>> dissimilarities = euclidean_distances(X)
    >>> mds_result, stress = smacof(dissimilarities, n_components=2, random_state=42)
    >>> mds_result
    array([[ 0.05... -1.07... ],
           [ 1.74..., -0.75...],
           [-1.79...,  1.83...]])
    >>> stress
    0.0012...
    """


    # 定义了一个示例的数组和相应的stress值，这里的示例作为注释和代码块的分隔符

    dissimilarities = check_array(dissimilarities)
    # 使用check_array函数验证并确保dissimilarities是一个合法的数组格式

    random_state = check_random_state(random_state)
    # 使用check_random_state函数验证并确保random_state是一个合法的随机状态对象

    if normalized_stress == "auto":
        normalized_stress = not metric
        # 如果normalized_stress设为"auto"，则根据metric的值自动设定normalized_stress的布尔值

    if normalized_stress and metric:
        raise ValueError(
            "Normalized stress is not supported for metric MDS. Either set"
            " `normalized_stress=False` or use `metric=False`."
        )
        # 如果同时设置了normalized_stress和metric为True，则抛出异常，因为metric MDS不支持normalized_stress

    if hasattr(init, "__array__"):
        init = np.asarray(init).copy()
        # 如果init具有__array__属性，则将其转换为NumPy数组，并复制一份副本

        if not n_init == 1:
            warnings.warn(
                "Explicit initial positions passed: "
                "performing only one init of the MDS instead of %d" % n_init
            )
            n_init = 1
            # 如果n_init不等于1，则发出警告，并将n_init设为1，只执行一次MDS初始化过程

    best_pos, best_stress = None, None
    # 初始化best_pos和best_stress为None

    if effective_n_jobs(n_jobs) == 1:
        # 如果指定的并行工作数为1

        for it in range(n_init):
            # 迭代执行n_init次

            pos, stress, n_iter_ = _smacof_single(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=random_state,
                normalized_stress=normalized_stress,
            )
            # 调用_smacof_single函数进行单次MDS优化，返回位置(pos)、应力值(stress)和迭代次数(n_iter_)

            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
                # 如果当前迭代的stress更优，则更新最佳的stress值、位置和迭代次数

    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        # 生成n_init个随机种子数组，用于并行计算

        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=seed,
                normalized_stress=normalized_stress,
            )
            for seed in seeds
        )
        # 并行计算多个_smacof_single函数的结果

        positions, stress, n_iters = zip(*results)
        # 解压缩并获取所有结果的位置、stress值和迭代次数

        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]
        # 找到最优的stress值对应的位置、stress值和迭代次数

    if return_n_iter:
        return best_pos, best_stress, best_iter
        # 如果设置了return_n_iter为True，则返回最佳位置、最佳stress值和对应的迭代次数

    else:
        return best_pos, best_stress
        # 否则，只返回最佳位置和最佳stress值
# 定义 MDS 类，用于多维缩放分析。
class MDS(BaseEstimator):
    """Multidimensional scaling.

    Read more in the :ref:`User Guide <multidimensional_scaling>`.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.

    metric : bool, default=True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Dissimilarity measure to use:

        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.

        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.

    normalized_stress : bool or "auto" default="auto"
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS.

        .. versionadded:: 1.2

        .. versionchanged:: 1.4
           The default value changed from `False` to `"auto"` in version 1.4.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.

    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.
    """
    # 保存点间差异度矩阵，形状为 (样本数, 样本数)
    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Symmetric matrix that:

        - either uses a custom dissimilarity matrix by setting `dissimilarity`
          to 'precomputed';
        - or constructs a dissimilarity matrix from data using
          Euclidean distances.

    # 记录在拟合过程中观察到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在特征名为字符串的情况下，记录在拟合过程中观察到的特征名
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 最佳 stress 对应的迭代次数
    n_iter_ : int
        The number of iterations corresponding to the best stress.

    # 相关的模型和算法参考
    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis that is a linear
        dimensionality reduction method.
    sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
        kernels and PCA.
    TSNE : T-distributed Stochastic Neighbor Embedding.
    Isomap : Manifold learning based on Isometric Mapping.
    LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
    SpectralEmbedding : Spectral embedding for non-linear dimensionality.

    # 相关的文献引用
    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
       Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
       hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
       Groenen P. Springer Series in Statistics (1997)

    # 示例用法
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import MDS
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = MDS(n_components=2, normalized_stress='auto')
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)

    For a more detailed example of usage, see:
    :ref:`sphx_glr_auto_examples_manifold_plot_mds.py`
    """

    # 参数约束字典，定义了 MDS 模型可接受的参数类型和取值范围
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "metric": ["boolean"],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["verbose"],
        "eps": [Interval(Real, 0.0, None, closed="left")],
        "n_jobs": [None, Integral],
        "random_state": ["random_state"],
        "dissimilarity": [StrOptions({"euclidean", "precomputed"})],
        "normalized_stress": ["boolean", StrOptions({"auto"})],
    }
    # 初始化方法，用于设置和初始化SMACOF算法的参数
    def __init__(
        self,
        n_components=2,                       # 设置嵌入空间的维度，默认为2维
        *,
        metric=True,                          # 是否使用距离度量，默认为True
        n_init=4,                             # 算法运行时的初始化次数，默认为4次
        max_iter=300,                         # 算法的最大迭代次数，默认为300次
        verbose=0,                            # 控制算法的输出信息详细程度，默认为0（不输出详细信息）
        eps=1e-3,                              # 算法迭代的停止阈值，默认为1e-3
        n_jobs=None,                          # 并行运算时使用的CPU核数，默认为None（不并行）
        random_state=None,                    # 控制随机数生成的种子值，默认为None
        dissimilarity="euclidean",            # 用于计算点之间差异的度量方法，默认为欧氏距离
        normalized_stress="auto",             # 是否自动规范化应力，默认为"auto"
    ):
        self.n_components = n_components      # 将参数设置为对象的属性，维度
        self.dissimilarity = dissimilarity    # 差异度量方法
        self.metric = metric                  # 是否使用度量
        self.n_init = n_init                  # 初始化次数
        self.max_iter = max_iter              # 最大迭代次数
        self.eps = eps                        # 迭代停止阈值
        self.verbose = verbose                # 输出详细信息的程度
        self.n_jobs = n_jobs                  # 并行计算的CPU核数
        self.random_state = random_state      # 随机数种子
        self.normalized_stress = normalized_stress  # 规范化应力的设置

    # 返回一个字典，用于指示是否采用预计算的距离矩阵
    def _more_tags(self):
        return {"pairwise": self.dissimilarity == "precomputed"}

    # 训练方法，计算数据点在嵌入空间中的位置
    def fit(self, X, y=None, init=None):
        """
        计算数据点在嵌入空间中的位置。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            输入数据。如果`dissimilarity=='precomputed'`，输入应为距离矩阵。

        y : 被忽略
            不被使用，按照惯例用于API的一致性。

        init : ndarray of shape (n_samples, n_components), default=None
            嵌入的初始配置，用于初始化SMACOF算法。默认情况下，使用随机选择的数组初始化算法。

        Returns
        -------
        self : object
            拟合后的估计器。
        """
        self.fit_transform(X, init=init)     # 调用fit_transform方法进行拟合和转换
        return self

    # 使用装饰器，进行适配上下文的处理
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, init=None):
        """
        Fit the data from `X`, and returns the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            X transformed in the new space.
        """
        # Validate input data X
        X = self._validate_data(X)

        # Check if X is a square matrix and dissimilarity type is not precomputed
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            # Issue a warning about API change in MDS
            warnings.warn(
                "The MDS API has changed. ``fit`` now constructs an"
                " dissimilarity matrix from data. To use a custom "
                "dissimilarity matrix, set "
                "``dissimilarity='precomputed'``."
            )

        # Determine how to compute dissimilarity matrix based on dissimilarity type
        if self.dissimilarity == "precomputed":
            # Use the provided dissimilarity matrix directly
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            # Compute Euclidean distances between points in X
            self.dissimilarity_matrix_ = euclidean_distances(X)

        # Run SMACOF algorithm to perform multidimensional scaling
        self.embedding_, self.stress_, self.n_iter_ = smacof(
            self.dissimilarity_matrix_,
            metric=self.metric,
            n_components=self.n_components,
            init=init,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
            random_state=self.random_state,
            return_n_iter=True,
            normalized_stress=self.normalized_stress,
        )

        # Return the embedded coordinates in the new space
        return self.embedding_
```