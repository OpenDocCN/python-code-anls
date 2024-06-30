# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_optics.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 从 numbers 模块导入 Integral（整数）和 Real（实数）类
from numbers import Integral, Real

# 导入 numpy 库，并重命名为 np
import numpy as np
# 导入稀疏效率警告和 issparse 函数，用于稀疏矩阵处理
from scipy.sparse import SparseEfficiencyWarning, issparse

# 从当前包的 base 模块中导入 BaseEstimator、ClusterMixin 和 _fit_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
# 导入数据转换警告异常类
from ..exceptions import DataConversionWarning
# 从 metrics 模块中导入 pairwise_distances 函数
from ..metrics import pairwise_distances
# 从 metrics.pairwise 模块中导入 _VALID_METRICS 和 PAIRWISE_BOOLEAN_FUNCTIONS
from ..metrics.pairwise import _VALID_METRICS, PAIRWISE_BOOLEAN_FUNCTIONS
# 从 neighbors 模块中导入 NearestNeighbors 类
from ..neighbors import NearestNeighbors
# 从 utils 模块中导入 gen_batches 函数
from ..utils import gen_batches
# 从 utils._chunking 模块中导入 get_chunk_n_rows 函数
from ..utils._chunking import get_chunk_n_rows
# 从 utils._param_validation 模块中导入 HasMethods、Interval、RealNotInt、StrOptions 和 validate_params 函数
from ..utils._param_validation import (
    HasMethods,
    Interval,
    RealNotInt,
    StrOptions,
    validate_params,
)
# 从 utils.validation 模块中导入 check_memory 函数
from ..utils.validation import check_memory

# 定义 OPTICS 类，继承自 ClusterMixin 和 BaseEstimator
class OPTICS(ClusterMixin, BaseEstimator):
    """Estimate clustering structure from vector array.

    OPTICS (Ordering Points To Identify the Clustering Structure), closely
    related to DBSCAN, finds core sample of high density and expands clusters
    from them [1]_. Unlike DBSCAN, keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large datasets than the
    current sklearn implementation of DBSCAN.

    Clusters are then extracted using a DBSCAN-like method
    (cluster_method = 'dbscan') or an automatic
    technique proposed in [1]_ (cluster_method = 'xi').

    This implementation deviates from the original OPTICS by first performing
    k-nearest-neighborhood searches on all points to identify core sizes, then
    computing only the distances to unprocessed points when constructing the
    cluster order. Note that we do not employ a heap to manage the expansion
    candidates, so the time complexity will be O(n^2).

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    min_samples : int > 1 or float between 0 and 1, default=5
        The number of samples in a neighborhood for a point to be considered as
        a core point. Also, up and down steep regions can't have more than
        ``min_samples`` consecutive non-steep points. Expressed as an absolute
        number or a fraction of the number of samples (rounded to be at least
        2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.
    """
    # 类级别的文档字符串，描述 OPTICS 算法及其特点

    # 初始化方法，定义 OPTICS 类的实例对象
    def __init__(
        self,
        min_samples=5,
        max_eps=np.inf,
    ):
        # 调用父类 ClusterMixin 和 BaseEstimator 的初始化方法
        super().__init__()
        # 设置 min_samples 参数，指定邻域内核心点的最小样本数
        self.min_samples = min_samples
        # 设置 max_eps 参数，指定两个样本之间的最大距离
        self.max_eps = max_eps

    # 更多的方法和逻辑可以在此处添加，用于执行 OPTICS 算法的各个步骤
    # 距离计算所使用的度量方式，可以是字符串或可调用的函数，默认为'minkowski'
    metric : str or callable, default='minkowski'
    
        # 如果metric是一个可调用的函数，将其应用于每一对实例（行），并记录结果值。
        # 这个可调用函数应该接受两个数组作为输入，并返回一个值，表示它们之间的距离。
        # 对于Scipy的度量函数，这种方式有效，但比将度量名称作为字符串传递要低效。
        # 如果metric是"precomputed"，则假定X是一个距离矩阵，并且必须是方阵。
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", `X` is assumed to be a distance matrix and must be
        square.
    
        # 可以使用的度量标准包括：
        Valid values for metric are:
    
        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']
    
        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']
    
    # Minkowski度量的参数p，默认为2
    p : float, default=2
    
        # 用于Minkowski度量的参数，当p = 1时，等同于曼哈顿距离（l1），当p = 2时，等同于欧几里得距离（l2）。
        # 对于任意的p，使用Minkowski距离（l_p）。
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    
    # 度量函数的额外关键字参数，作为字典传递，默认为None
    metric_params : dict, default=None
    
    # 用于提取聚类的方法，可能的取值为"xi"和"dbscan"，默认为'xi'
    cluster_method : str, default='xi'
    
    # 提取聚类时使用的最大邻域距离，仅当cluster_method='dbscan'时使用，默认为None
    eps : float, default=None
    
    # 确定在可达性图上构成聚类边界的最小陡度，取值范围为0到1，默认为0.05
    xi : float between 0 and 1, default=0.05
    
        # 确定在可达性图上构成聚类边界的最小陡度。例如，可达性图中的向上点由其一个点到其后继点的比率不超过1-xi定义。
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.
    
    # 是否根据OPTICS计算的前趋节点对聚类进行校正，默认为True
    predecessor_correction : bool, default=True
    
        # 根据OPTICS计算的前趋节点对聚类进行校正。对大多数数据集影响较小。
        Correct clusters according to the predecessors calculated by OPTICS
        [2]_.
    # 定义最小的 OPTICS 簇中样本数，可以是绝对数或样本数的比例（至少为 2）
    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.
        Used only when ``cluster_method='xi'``.

    # 用于计算最近邻的算法选择
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`~sklearn.neighbors.BallTree`.
        - 'kd_tree' will use :class:`~sklearn.neighbors.KDTree`.
        - 'brute' will use a brute-force search.
        - 'auto' (default) will attempt to decide the most appropriate
          algorithm based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    # 叶子大小传递给 BallTree 或 KDTree 的参数
    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.

    # 用于缓存树计算结果的内存配置
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    # 并行搜索近邻时使用的作业数
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    # 定义属性

    # 每个样本的簇标签，对应于 fit() 方法中的数据集
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples and points which are not included in a leaf cluster
        of ``cluster_hierarchy_`` are labeled as -1.

    # 每个样本的可达距离，按对象顺序索引。使用 ``clust.reachability_[clust.ordering_]`` 以按簇顺序访问。
    reachability_ : ndarray of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    # 样本的簇顺序列表的索引
    ordering_ : ndarray of shape (n_samples,)
        The cluster ordered list of sample indices.

    # 每个样本成为核心点的距离，按对象顺序索引。永远不会成为核心的点的距离为 inf。
    core_distances_ : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    # 每个样本到达的前驱点，按对象顺序索引。种子点的前驱点为 -1。
    predecessor_ : ndarray of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.
    cluster_hierarchy_ : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to
        ``(end, -start)`` (ascending) so that larger clusters encompassing
        smaller clusters come after those smaller ones. Since ``labels_`` does
        not reflect the hierarchy, usually
        ``len(cluster_hierarchy_) > np.unique(optics.labels_)``. Please also
        note that these indices are of the ``ordering_``, i.e.
        ``X[ordering_][start:end + 1]`` form a cluster.
        Only available when ``cluster_method='xi'``.

        描述变量 `cluster_hierarchy_` 的含义及其结构，包含每个聚类的起始和结束索引，以及它们的顺序和层次关系。在 `cluster_method='xi'` 时可用。

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

        记录在 `fit` 过程中观察到的特征数量。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

        在 `X` 中的特征名称列表，仅当所有特征名称均为字符串时定义。

    See Also
    --------
    DBSCAN : A similar clustering for a specified neighborhood radius (eps).
        Our implementation is optimized for runtime.

        相关的聚类方法 `DBSCAN`，适用于指定邻域半径（eps）的类似聚类问题，我们的实现优化了运行时间。

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

       OPTICS 算法的文献参考，用于识别聚类结构的点的排序方法。

    .. [2] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.

       改进从 OPTICS 图中提取的聚类结构的方法的参考文献。

    Examples
    --------
    >>> from sklearn.cluster import OPTICS
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> clustering = OPTICS(min_samples=2).fit(X)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])

    For a more detailed example see
    :ref:`sphx_glr_auto_examples_cluster_plot_optics.py`.

        示例代码，展示了如何使用 OPTICS 算法进行聚类分析。
    _parameter_constraints: dict = {
        "min_samples": [
            Interval(Integral, 2, None, closed="left"),  # 约束 min_samples 必须是大于等于 2 的整数
            Interval(RealNotInt, 0, 1, closed="both"),  # 约束 min_samples 必须是 [0, 1] 范围内的非整数实数
        ],
        "max_eps": [Interval(Real, 0, None, closed="both")],  # 约束 max_eps 必须是大于等于 0 的实数
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],  # 约束 metric 必须是预定义的度量选项或可调用对象
        "p": [Interval(Real, 1, None, closed="left")],  # 约束 p 必须是大于等于 1 的实数
        "metric_params": [dict, None],  # 约束 metric_params 可以是字典类型或者 None
        "cluster_method": [StrOptions({"dbscan", "xi"})],  # 约束 cluster_method 必须是 {"dbscan", "xi"} 中的一个字符串选项
        "eps": [Interval(Real, 0, None, closed="both"), None],  # 约束 eps 必须是大于等于 0 的实数或者 None
        "xi": [Interval(Real, 0, 1, closed="both")],  # 约束 xi 必须是 [0, 1] 范围内的实数
        "predecessor_correction": ["boolean"],  # 约束 predecessor_correction 必须是布尔值
        "min_cluster_size": [
            Interval(Integral, 2, None, closed="left"),  # 约束 min_cluster_size 必须是大于等于 2 的整数
            Interval(RealNotInt, 0, 1, closed="right"),  # 约束 min_cluster_size 必须是 (0, 1] 范围内的非整数实数
            None,  # 允许 min_cluster_size 是 None
        ],
        "algorithm": [StrOptions({"auto", "brute", "ball_tree", "kd_tree"})],  # 约束 algorithm 必须是 {"auto", "brute", "ball_tree", "kd_tree"} 中的一个字符串选项
        "leaf_size": [Interval(Integral, 1, None, closed="left")],  # 约束 leaf_size 必须是大于等于 1 的整数
        "memory": [str, HasMethods("cache"), None],  # 约束 memory 可以是字符串类型、有 "cache" 方法的对象，或者 None
        "n_jobs": [Integral, None],  # 约束 n_jobs 必须是整数或者 None
    }
    
    def __init__(
        self,
        *,
        min_samples=5,  # 初始化对象时的参数，设置 min_samples 默认值为 5
        max_eps=np.inf,  # 初始化对象时的参数，设置 max_eps 默认值为正无穷大
        metric="minkowski",  # 初始化对象时的参数，设置 metric 默认值为 "minkowski"
        p=2,  # 初始化对象时的参数，设置 p 默认值为 2
        metric_params=None,  # 初始化对象时的参数，设置 metric_params 默认值为 None
        cluster_method="xi",  # 初始化对象时的参数，设置 cluster_method 默认值为 "xi"
        eps=None,  # 初始化对象时的参数，设置 eps 默认值为 None
        xi=0.05,  # 初始化对象时的参数，设置 xi 默认值为 0.05
        predecessor_correction=True,  # 初始化对象时的参数，设置 predecessor_correction 默认值为 True
        min_cluster_size=None,  # 初始化对象时的参数，设置 min_cluster_size 默认值为 None
        algorithm="auto",  # 初始化对象时的参数，设置 algorithm 默认值为 "auto"
        leaf_size=30,  # 初始化对象时的参数，设置 leaf_size 默认值为 30
        memory=None,  # 初始化对象时的参数，设置 memory 默认值为 None
        n_jobs=None,  # 初始化对象时的参数，设置 n_jobs 默认值为 None
    ):
        self.max_eps = max_eps  # 将传入的 max_eps 参数赋值给对象的 max_eps 属性
        self.min_samples = min_samples  # 将传入的 min_samples 参数赋值给对象的 min_samples 属性
        self.min_cluster_size = min_cluster_size  # 将传入的 min_cluster_size 参数赋值给对象的 min_cluster_size 属性
        self.algorithm = algorithm  # 将传入的 algorithm 参数赋值给对象的 algorithm 属性
        self.metric = metric  # 将传入的 metric 参数赋值给对象的 metric 属性
        self.metric_params = metric_params  # 将传入的 metric_params 参数赋值给对象的 metric_params 属性
        self.p = p  # 将传入的 p 参数赋值给对象的 p 属性
        self.leaf_size = leaf_size  # 将传入的 leaf_size 参数赋值给对象的 leaf_size 属性
        self.cluster_method = cluster_method  # 将传入的 cluster_method 参数赋值给对象的 cluster_method 属性
        self.eps = eps  # 将传入的 eps 参数赋值给对象的 eps 属性
        self.xi = xi  # 将传入的 xi 参数赋值给对象的 xi 属性
        self.predecessor_correction = predecessor_correction  # 将传入的 predecessor_correction 参数赋值给对象的 predecessor_correction 属性
        self.memory = memory  # 将传入的 memory 参数赋值给对象的 memory 属性
        self.n_jobs = n_jobs  # 将传入的 n_jobs 参数赋值给对象的 n_jobs 属性
    
    @_fit_context(
        # Optics.metric is not validated yet
        prefer_skip_nested_validation=False
    )
# 确保给定的 size 不超过 n_samples，否则引发 ValueError 异常
def _validate_size(size, n_samples, param_name):
    if size > n_samples:
        raise ValueError(
            "%s must be no greater than the number of samples (%d). Got %d"
            % (param_name, n_samples, size)
        )


# OPTICS 辅助函数
def _compute_core_distances_(X, neighbors, min_samples, working_memory):
    """计算每个样本的第 k 个最近邻的距离。

    相当于 neighbors.kneighbors(X, self.min_samples)[0][:, -1]
    但更加内存有效。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        数据集。
    neighbors : NearestNeighbors 实例
        已拟合的最近邻估计器。
    min_samples : int
        用于判断核心点的邻域中最少样本数。
    working_memory : int, 默认为 None
        用于临时距离矩阵块的最大内存。
        当为 None（默认）时，使用 ``sklearn.get_config()['working_memory']`` 的值。

    Returns
    -------
    core_distances : ndarray of shape (n_samples,)
        每个样本成为核心点的距离。
        永远不会成为核心点的点具有无穷远的距离。
    """
    n_samples = X.shape[0]
    core_distances = np.empty(n_samples)
    core_distances.fill(np.nan)

    # 计算每个片段的行数
    chunk_n_rows = get_chunk_n_rows(
        row_bytes=16 * min_samples, max_n_rows=n_samples, working_memory=working_memory
    )
    # 生成分段
    slices = gen_batches(n_samples, chunk_n_rows)
    for sl in slices:
        # 计算每个分段中样本的第 k 个最近邻的距离
        core_distances[sl] = neighbors.kneighbors(X[sl], min_samples)[0][:, -1]
    return core_distances


@validate_params(
    {
        "X": [np.ndarray, "sparse matrix"],
        "min_samples": [
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="both"),
        ],
        "max_eps": [Interval(Real, 0, None, closed="both")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric_params": [dict, None],
        "algorithm": [StrOptions({"auto", "brute", "ball_tree", "kd_tree"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=False,  # metric is not validated yet
)
def compute_optics_graph(
    X, *, min_samples, max_eps, metric, p, metric_params, algorithm, leaf_size, n_jobs
):
    """计算 OPTICS 可达图。

    更多信息请参阅 :ref:`User Guide <optics>`.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples) if metric='precomputed'
        特征数组，或者如果 metric='precomputed' 则是样本之间的距离数组。

    min_samples : int > 1 or float between 0 and 1
        被认为是核心点邻域中的样本数，可以是绝对数目或样本数的比例（至少为 2）。
    max_eps : float, default=np.inf
        # 定义最大邻域距离，用于确定样本点之间的邻近关系。默认值为 np.inf 表示在所有尺度上识别聚类；减小 max_eps 可以减少运行时间。

    metric : str or callable, default='minkowski'
        # 用于距离计算的度量标准。可以使用 scikit-learn 或 scipy.spatial.distance 中的任何度量标准。

        # 如果 metric 是一个可调用函数，则它将被用于计算每对实例（行）之间的距离，并记录结果值。该函数应接受两个数组作为输入，并返回一个表示它们之间距离的值。这对于 Scipy 的度量标准有效，但比直接传递度量标准名称效率低。如果 metric 是 "precomputed"，则假定 X 是一个距离矩阵，并且必须是方阵。

        # metric 的有效值包括：

        # - 从 scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

        # - 从 scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

        # 参见 scipy.spatial.distance 文档以获取有关这些度量标准的详细信息。

        # .. note::
        #    `'kulsinski'` 自 SciPy 1.9 起已被弃用，并将在 SciPy 1.11 中移除。

    p : float, default=2
        # Minkowski 距离度量的参数，用于 :class:`~sklearn.metrics.pairwise_distances`。当 p = 1 时，相当于使用曼哈顿距离（l1），当 p = 2 时，相当于使用欧氏距离（l2）。对于任意的 p 值，使用 Minkowski 距离（l_p）。

    metric_params : dict, default=None
        # 度量函数的额外关键字参数。

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        # 用于计算最近邻居的算法选择：

        # - 'ball_tree' 将使用 :class:`~sklearn.neighbors.BallTree`。
        # - 'kd_tree' 将使用 :class:`~sklearn.neighbors.KDTree`。
        # - 'brute' 将使用蛮力搜索。
        # - 'auto' 将根据 `fit` 方法传入的值尝试决定最合适的算法（默认）。

        # 注意：对稀疏输入进行拟合将覆盖此参数设置，使用蛮力搜索。

    leaf_size : int, default=30
        # 传递给 :class:`~sklearn.neighbors.BallTree` 或 :class:`~sklearn.neighbors.KDTree` 的叶子大小。这会影响构建和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质。
    # 获取样本数量
    n_samples = X.shape[0]
    # 验证最小样本数的有效性
    _validate_size(min_samples, n_samples, "min_samples")
    # 如果最小样本数小于等于1，则设定为大于等于2，并且确保它不超过样本数
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))

    # 初始化 reachability_ 数组，所有值为无穷大
    reachability_ = np.empty(n_samples)
    reachability_.fill(np.inf)
    # 初始化 predecessor_ 数组，所有值为 -1
    predecessor_ = np.empty(n_samples, dtype=int)
    predecessor_.fill(-1)

    # 创建一个最近邻对象
    nbrs = NearestNeighbors(
        n_neighbors=min_samples,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        metric_params=metric_params,
        p=p,
        n_jobs=n_jobs,
    )

    # 将数据 X 拟合到最近邻对象上
    nbrs.fit(X)
    
    # 在这里，我们首先对每个点进行 k 近邻查询，这与原始的 OPTICS 算法有所不同，
    # 后者仅使用了 epsilon 范围查询。
    # TODO: 如何处理工作内存（working_memory）？
    # 计算核心距离，基于输入数据 X、邻居信息 nbrs、最小样本数 min_samples，不使用工作内存
    core_distances_ = _compute_core_distances_(
        X=X, neighbors=nbrs, min_samples=min_samples, working_memory=None
    )
    
    # 将核心距离中大于 max_eps 的值设为无穷大
    core_distances_[core_distances_ > max_eps] = np.inf
    
    # 对核心距离进行四舍五入，小数位数由核心距离数据类型的精度决定
    np.around(
        core_distances_,
        decimals=np.finfo(core_distances_.dtype).precision,
        out=core_distances_,
    )

    # OPTICS 的主要循环。不支持并行化。写入 'ordering_' 列表的顺序很重要！
    # 注意，这个实现在理论上是 O(n^2)，但据说具有非常低的常数因子。
    processed = np.zeros(X.shape[0], dtype=bool)  # 初始化处理标记数组
    ordering = np.zeros(X.shape[0], dtype=int)   # 初始化顺序数组

    # 对于每个数据点，按照最小可达距离进行排序
    for ordering_idx in range(X.shape[0]):
        # 从未处理的点中选择下一个点，依据其最小可达距离
        index = np.where(processed == 0)[0]
        point = index[np.argmin(reachability_[index])]

        # 标记该点为已处理，并记录其顺序
        processed[point] = True
        ordering[ordering_idx] = point

        # 如果该点的核心距离不是无穷大，则更新其可达距离
        if core_distances_[point] != np.inf:
            _set_reach_dist(
                core_distances_=core_distances_,
                reachability_=reachability_,
                predecessor_=predecessor_,
                point_index=point,
                processed=processed,
                X=X,
                nbrs=nbrs,
                metric=metric,
                metric_params=metric_params,
                p=p,
                max_eps=max_eps,
            )

    # 如果所有数据点的可达距离都是无穷大，发出警告
    if np.all(np.isinf(reachability_)):
        warnings.warn(
            (
                "All reachability values are inf. Set a larger"
                " max_eps or all data will be considered outliers."
            ),
            UserWarning,
        )
    
    # 返回排序后的顺序列表、核心距离、可达距离和前驱节点列表
    return ordering, core_distances_, reachability_, predecessor_
defpython
def _set_reach_dist(
    core_distances_,
    reachability_,
    predecessor_,
    point_index,
    processed,
    X,
    nbrs,
    metric,
    metric_params,
    p,
    max_eps,
):
    # Extract the current point as a single-row array from X
    P = X[point_index : point_index + 1]

    # Assume that radius_neighbors is faster without distances
    # and we don't need all distances, nevertheless, this means
    # we may be doing some work twice.
    # Query the neighborhood within max_eps radius around point P
    indices = nbrs.radius_neighbors(P, radius=max_eps, return_distance=False)[0]

    # Getting indices of neighbors that have not been processed
    # Compress indices to get unprocessed neighbors
    unproc = np.compress(~np.take(processed, indices), indices)

    # If all neighbors of the current point are processed, return
    if not unproc.size:
        return

    # Only compute distances to unprocessed neighbors:
    if metric == "precomputed":
        # If the metric is precomputed, directly use distances from X
        dists = X[[point_index], unproc]
        if isinstance(dists, np.matrix):
            dists = np.asarray(dists)
        dists = dists.ravel()
    else:
        # Prepare parameters for pairwise distance calculation
        _params = dict() if metric_params is None else metric_params.copy()
        if metric == "minkowski" and "p" not in _params:
            # Ensure 'p' parameter is set if using Minkowski metric
            _params["p"] = p
        # Calculate distances using specified metric and parameters
        dists = pairwise_distances(P, X[unproc], metric, n_jobs=None, **_params).ravel()

    # Compute revised reachability distances (rdists)
    rdists = np.maximum(dists, core_distances_[point_index])
    # Round distances to the precision of the data type
    np.around(rdists, decimals=np.finfo(rdists.dtype).precision, out=rdists)
    # Identify where distances are improved (smaller) than current reachability
    improved = np.where(rdists < np.take(reachability_, unproc))
    # Update reachability distances and predecessors for improved points
    reachability_[unproc[improved]] = rdists[improved]
    predecessor_[unproc[improved]] = point_index



@validate_params(
    {
        "reachability": [np.ndarray],
        "core_distances": [np.ndarray],
        "ordering": [np.ndarray],
        "eps": [Interval(Real, 0, None, closed="both")],
    },
    prefer_skip_nested_validation=True,
)
def cluster_optics_dbscan(*, reachability, core_distances, ordering, eps):
    """Perform DBSCAN extraction for an arbitrary epsilon.

    Extracting the clusters runs in linear time. Note that this results in
    ``labels_`` which are close to a :class:`~sklearn.cluster.DBSCAN` with
    similar settings and ``eps``, only if ``eps`` is close to ``max_eps``.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (``reachability_``).

    core_distances : ndarray of shape (n_samples,)
        Distances at which points become core (``core_distances_``).

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (``ordering_``).

    eps : float
        DBSCAN ``eps`` parameter. Must be set to < ``max_eps``. Results
        will be close to DBSCAN algorithm if ``eps`` and ``max_eps`` are close
        to one another.

    Returns
    -------
    labels_ : array of shape (n_samples,)
        The estimated labels.

    Examples
    --------
    >>> import numpy as np


注释：
    # 计算输入数据集 X 的 OPTICS 图，并返回相关结果
    >>> from sklearn.cluster import cluster_optics_dbscan, compute_optics_graph
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    
    # 调用 compute_optics_graph 函数计算 OPTICS 图的相关参数
    >>> ordering, core_distances, reachability, predecessor = compute_optics_graph(
    ...     X,
    ...     min_samples=2,
    ...     max_eps=np.inf,
    ...     metric="minkowski",
    ...     p=2,
    ...     metric_params=None,
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     n_jobs=None,
    ... )
    
    # 设定 DBSCAN 算法中的 epsilon 参数
    >>> eps = 4.5
    
    # 调用 cluster_optics_dbscan 函数执行基于 OPTICS 图的 DBSCAN 聚类
    >>> labels = cluster_optics_dbscan(
    ...     reachability=reachability,
    ...     core_distances=core_distances,
    ...     ordering=ordering,
    ...     eps=eps,
    ... )
    
    # 输出聚类结果标签
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    
    # 根据 OPTICS 图计算得到聚类标签的函数定义
    def optics_clustering_labels(core_distances, reachability, ordering, eps):
        # 获取样本数
        n_samples = len(core_distances)
        # 初始化标签数组
        labels = np.zeros(n_samples, dtype=int)
        
        # 标记远离核心区域的样本
        far_reach = reachability > eps
        # 标记接近核心区域的样本
        near_core = core_distances <= eps
        
        # 根据 OPTICS 图的排序顺序设置聚类标签
        labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
        # 设置不属于任何聚类的样本标签为 -1
        labels[far_reach & ~near_core] = -1
        
        # 返回计算得到的聚类标签数组
        return labels
# 使用装饰器 validate_params 对 cluster_optics_xi 函数的参数进行验证
@validate_params(
    {
        "reachability": [np.ndarray],  # reachability 参数类型为 ndarray
        "predecessor": [np.ndarray],  # predecessor 参数类型为 ndarray
        "ordering": [np.ndarray],  # ordering 参数类型为 ndarray
        "min_samples": [  # min_samples 参数类型为整数或浮点数区间
            Interval(Integral, 2, None, closed="left"),  # 整数范围为 [2, 正无穷)，左闭右开
            Interval(RealNotInt, 0, 1, closed="both"),  # 浮点数范围为 [0, 1]，闭区间
        ],
        "min_cluster_size": [  # min_cluster_size 参数类型为整数或浮点数区间或 None
            Interval(Integral, 2, None, closed="left"),  # 整数范围为 [2, 正无穷)，左闭右开
            Interval(RealNotInt, 0, 1, closed="both"),  # 浮点数范围为 [0, 1]，闭区间
            None,  # 可以为 None
        ],
        "xi": [Interval(Real, 0, 1, closed="both")],  # xi 参数类型为 [0, 1] 闭区间的浮点数
        "predecessor_correction": ["boolean"],  # predecessor_correction 参数类型为布尔值
    },
    prefer_skip_nested_validation=True,  # 设置跳过嵌套验证
)
def cluster_optics_xi(
    *,
    reachability,  # reachability 参数
    predecessor,  # predecessor 参数
    ordering,  # ordering 参数
    min_samples,  # min_samples 参数
    min_cluster_size=None,  # min_cluster_size 参数，默认为 None
    xi=0.05,  # xi 参数，默认值为 0.05
    predecessor_correction=True,  # predecessor_correction 参数，默认值为 True
):
    """Automatically extract clusters according to the Xi-steep method.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (`reachability_`).

    predecessor : ndarray of shape (n_samples,)
        Predecessors calculated by OPTICS.

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (`ordering_`).

    min_samples : int > 1 or float between 0 and 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.
        Expressed as an absolute number or a fraction of the number of samples
        (rounded to be at least 2).

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    predecessor_correction : bool, default=True
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The labels assigned to samples. Points which are not included
        in any cluster are labeled as -1.

    clusters : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to ``(end,
        -start)`` (ascending) so that larger clusters encompassing smaller
        clusters come after such nested smaller clusters. Since ``labels`` does
        not reflect the hierarchy, usually ``len(clusters) >
        np.unique(labels)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import cluster_optics_xi, compute_optics_graph
    # 计算样本的数量
    n_samples = len(reachability)
    
    # 校验最小样本数是否合法，并根据需要调整为有效值
    _validate_size(min_samples, n_samples, "min_samples")
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))
    
    # 如果未指定最小聚类大小，则设为最小样本数
    if min_cluster_size is None:
        min_cluster_size = min_samples
    
    # 校验最小聚类大小是否合法，并根据需要调整为有效值
    _validate_size(min_cluster_size, n_samples, "min_cluster_size")
    if min_cluster_size <= 1:
        min_cluster_size = max(2, int(min_cluster_size * n_samples))
    
    # 使用 xi 进行聚类计算，返回聚类结果
    clusters = _xi_cluster(
        reachability[ordering],
        predecessor[ordering],
        ordering,
        xi,
        min_samples,
        min_cluster_size,
        predecessor_correction,
    )
    
    # 从聚类结果中提取标签
    labels = _extract_xi_labels(ordering, clusters)
    
    # 返回标签和聚类结果
    return labels, clusters
def _extend_region(steep_point, xward_point, start, min_samples):
    """Extend the area until it's maximal.

    It's the same function for both upward and downward regions, depending on
    the given input parameters. Assuming:

        - steep_{upward/downward}: bool array indicating whether a point is a
          steep {upward/downward};
        - upward/downward: bool array indicating whether a point is
          upward/downward;

    To extend an upward region, `steep_point=steep_upward` and
    `xward_point=downward` are expected, and to extend a downward region,
    `steep_point=steep_downward` and `xward_point=upward`.

    Parameters
    ----------
    steep_point : ndarray of shape (n_samples,), dtype=bool
        True if the point is steep downward (upward).

    xward_point : ndarray of shape (n_samples,), dtype=bool
        True if the point is an upward (respectively downward) point.

    start : int
        The start of the xward region.

    min_samples : int
        The same as the min_samples given to OPTICS. Up and down steep
        regions can't have more than `min_samples` consecutive non-steep
        points.

    Returns
    -------
    index : int
        The current index iterating over all the samples, i.e. where we are up
        to in our search.

    end : int
        The end of the region, which can be behind the index. The region
        includes the `end` index.
    """
    n_samples = len(steep_point)  # 获取 steep_point 数组的长度
    non_xward_points = 0  # 记录非 xward 点的数量
    index = start  # 从指定的 start 索引开始
    end = start  # 初始时将 end 设置为 start
    # 找到最大的区域
    while index < n_samples:
        if steep_point[index]:  # 如果当前点是 steep point
            non_xward_points = 0  # 重置非 xward 点的计数
            end = index  # 更新 end 到当前索引
        elif not xward_point[index]:  # 如果当前点不是 xward point
            non_xward_points += 1  # 增加非 xward 点的计数
            # 区域不应包含超过 min_samples 个连续的非 steep 的 xward 点
            if non_xward_points > min_samples:
                break  # 超过则跳出循环
        else:
            return end  # 如果是 xward point，则返回当前的 end 索引
        index += 1  # 移动到下一个点
    return end  # 返回找到的最大区域的 end 索引


def _update_filter_sdas(sdas, mib, xi_complement, reachability_plot):
    """Update steep down areas (SDAs) using the new maximum in between (mib)
    value, and the given complement of xi, i.e. ``1 - xi``.
    """
    if np.isinf(mib):  # 如果 mib 是无穷大，则返回空列表
        return []
    # 更新 steep down areas (SDAs)，使用新的 mib 值和 xi 的补集
    res = [
        sda for sda in sdas if mib <= reachability_plot[sda["start"]] * xi_complement
    ]
    for sda in res:
        sda["mib"] = max(sda["mib"], mib)  # 更新每个 SDA 的 mib 值
    return res  # 返回更新后的 steep down areas (SDAs)


def _correct_predecessor(reachability_plot, predecessor_plot, ordering, s, e):
    """Correct for predecessors.

    Applies Algorithm 2 of [1]_.

    Input parameters are ordered by the computer OPTICS ordering.

    .. [1] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.
    """
    # 当起点 s 小于终点 e 时，执行循环
    while s < e:
        # 如果起点 s 对应的可达性值大于终点 e 对应的可达性值
        if reachability_plot[s] > reachability_plot[e]:
            # 返回起点 s 和终点 e
            return s, e
        # 取出终点 e 的前驱节点 p_e
        p_e = predecessor_plot[e]
        # 遍历从起点 s 到终点 e 之间的节点索引 i
        for i in range(s, e):
            # 如果终点 e 的前驱节点 p_e 等于排序列表中索引为 i 的节点
            if p_e == ordering[i]:
                # 返回起点 s 和终点 e
                return s, e
        # 缩小终点 e 的范围，继续循环
        e -= 1
    
    # 当循环结束时，如果没有找到满足条件的 s 和 e，返回空值
    return None, None
def _xi_cluster(
    reachability_plot,
    predecessor_plot,
    ordering,
    xi,
    min_samples,
    min_cluster_size,
    predecessor_correction,
):
    """Automatically extract clusters according to the Xi-steep method.

    This is rouphly an implementation of Figure 19 of the OPTICS paper.

    Parameters
    ----------
    reachability_plot : array-like of shape (n_samples,)
        The reachability plot, i.e. reachability ordered according to
        the calculated ordering, all computed by OPTICS.

    predecessor_plot : array-like of shape (n_samples,)
        Predecessors ordered according to the calculated ordering.

    xi : float, between 0 and 1
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    min_samples : int > 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.

    min_cluster_size : int > 1
        Minimum number of samples in an OPTICS cluster.

    predecessor_correction : bool
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    clusters : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of [start, end] in each row, with all
        indices inclusive. The clusters are ordered in a way that larger
        clusters encompassing smaller clusters come after those smaller
        clusters.
    """

    # Our implementation adds an inf to the end of reachability plot
    # this helps to find potential clusters at the end of the
    # reachability plot even if there's no upward region at the end of it.
    reachability_plot = np.hstack((reachability_plot, np.inf))

    xi_complement = 1 - xi
    sdas = []  # steep down areas, introduced in section 4.3.2 of the paper
    clusters = []
    index = 0
    mib = 0.0  # maximum in between, section 4.3.2

    # Our implementation corrects a mistake in the original
    # paper, i.e., in Definition 9 steep downward point,
    # r(p) * (1 - x1) <= r(p + 1) should be
    # r(p) * (1 - x1) >= r(p + 1)
    with np.errstate(invalid="ignore"):
        ratio = reachability_plot[:-1] / reachability_plot[1:]
        steep_upward = ratio <= xi_complement
        steep_downward = ratio >= 1 / xi_complement
        downward = ratio > 1
        upward = ratio < 1

    # the following loop is almost exactly as Figure 19 of the paper.
    # it jumps over the areas which are not either steep down or up areas
    return np.array(clusters)
    # 创建一个与 `ordering` 长度相同的全为 -1 的整数数组，用于存储聚类标签
    labels = np.full(len(ordering), -1, dtype=int)
    
    # 初始化标签为 0，用于给每个聚类分配一个唯一标签
    label = 0
    
    # 遍历每个聚类区间 (start, end)，通过 `_xi_cluster` 函数返回的列表
    for c in clusters:
        # 如果当前聚类区间内的标签都是 -1（即未被分配过标签），则为该区间内所有点分配同一个标签
        if not np.any(labels[c[0] : (c[1] + 1)] != -1):
            labels[c[0] : (c[1] + 1)] = label
            # 分配完标签后，更新标签值，以便下一个聚类使用不同的标签
            label += 1
    
    # 根据 `ordering` 数组重新排列 `labels` 数组，以确保与原始输入顺序对应的标签分配
    labels[ordering] = labels.copy()
    
    # 返回标签数组作为函数的结果
    return labels
```