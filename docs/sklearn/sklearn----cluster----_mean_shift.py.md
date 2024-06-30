# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_mean_shift.py`

```
"""Mean shift clustering algorithm.

Mean shift clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These
candidates are then filtered in a post-processing stage to eliminate
near-duplicates to form the final set of centroids.

Seeding is performed using a binning technique for scalability.
"""

# Authors: Conrad Lee <conradlee@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Martino Sorbaro <martino.sorbaro@ed.ac.uk>

# 引入警告模块，用于显示警告信息
import warnings
# 引入默认字典，用于聚类过程中的临时数据存储
from collections import defaultdict
# 引入整数和实数类型的支持
from numbers import Integral, Real

# 引入NumPy库，并使用别名np
import numpy as np

# 从内部配置中引入上下文管理器
from .._config import config_context
# 引入基本估计器、聚类混合类、拟合上下文管理器
from ..base import BaseEstimator, ClusterMixin, _fit_context
# 从度量模块中引入用于计算点之间距离的函数
from ..metrics.pairwise import pairwise_distances_argmin
# 引入最近邻模块，用于近邻搜索
from ..neighbors import NearestNeighbors
# 引入数组检查、随机状态检查、批处理生成等实用工具
from ..utils import check_array, check_random_state, gen_batches
# 引入参数验证模块，用于验证函数参数的合法性
from ..utils._param_validation import Interval, validate_params
# 引入并行计算模块，支持并行计算任务的执行
from ..utils.parallel import Parallel, delayed
# 引入检查是否拟合的函数，用于验证估计器是否已经适合数据
from ..utils.validation import check_is_fitted

# 使用参数验证装饰器，验证函数的输入参数合法性
@validate_params(
    {
        "X": ["array-like"],
        "quantile": [Interval(Real, 0, 1, closed="both")],
        "n_samples": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=True,
)
# 估算用于均值漂移算法的带宽
def estimate_bandwidth(X, *, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
    """Estimate the bandwidth to use with the mean-shift algorithm.

    This function takes time at least quadratic in `n_samples`. For large
    datasets, it is wise to subsample by setting `n_samples`. Alternatively,
    the parameter `bandwidth` can be set to a small value without estimating
    it.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input points.

    quantile : float, default=0.3
        Should be between [0, 1]
        0.5 means that the median of all pairwise distances is used.

    n_samples : int, default=None
        The number of samples to use. If not given, all samples are used.

    random_state : int, RandomState instance, default=None
        The generator used to randomly select the samples from input points
        for bandwidth estimation. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    bandwidth : float
        The bandwidth parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import estimate_bandwidth
    """
    X = check_array(X)
    # 调用函数 check_array 对输入的 X 进行验证和转换，确保其是一个合法的 NumPy 数组

    random_state = check_random_state(random_state)
    # 调用函数 check_random_state 对 random_state 进行验证和转换，确保其是一个随机状态对象

    if n_samples is not None:
        # 如果指定了 n_samples，则从 X 的索引中随机选择 n_samples 个样本
        idx = random_state.permutation(X.shape[0])[:n_samples]
        X = X[idx]
    
    # 根据 quantile 计算近邻的数量
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:
        # 如果计算出的近邻数量小于 1，则将其设置为 1，因为 NearestNeighbors 不能使用 n_neighbors = 0
        n_neighbors = 1
    
    # 创建 NearestNeighbors 对象，设置近邻的数量和并行工作的进程数
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
    nbrs.fit(X)

    # 初始化带宽为 0.0
    bandwidth = 0.0
    
    # 使用 gen_batches 生成器对 X 进行分批处理，每批大小为 500
    for batch in gen_batches(len(X), 500):
        # 计算每个批次中样本点到其最近邻居的距离，并返回距离矩阵 d
        d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
        # 将每个批次中最大距离的和累加到带宽 bandwidth 中
        bandwidth += np.max(d, axis=1).sum()

    # 计算平均带宽并返回
    return bandwidth / X.shape[0]
# separate function for each seed's iterative loop
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    # 对每个种子点，通过迭代循环找到梯度最大值，直到收敛或达到最大迭代次数
    bandwidth = nbrs.get_params()["radius"]
    stop_thresh = 1e-3 * bandwidth  # 当均值收敛时的停止阈值
    completed_iterations = 0
    while True:
        # 找出带宽内的点的均值
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break  # 根据种子策略，这种情况可能会发生
        my_old_mean = my_mean  # 保存旧均值
        my_mean = np.mean(points_within, axis=0)
        # 如果收敛或达到最大迭代次数，则添加该聚类
        if (
            np.linalg.norm(my_mean - my_old_mean) <= stop_thresh
            or completed_iterations == max_iter
        ):
            break
        completed_iterations += 1
    return tuple(my_mean), len(points_within), completed_iterations


@validate_params(
    {"X": ["array-like"]},
    prefer_skip_nested_validation=False,
)
def mean_shift(
    X,
    *,
    bandwidth=None,
    seeds=None,
    bin_seeding=False,
    min_bin_freq=1,
    cluster_all=True,
    max_iter=300,
    n_jobs=None,
):
    """Perform mean shift clustering of data using a flat kernel.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Input data.

    bandwidth : float, default=None
        Kernel bandwidth. If not None, must be in the range [0, +inf).

        If None, the bandwidth is determined using a heuristic based on
        the median of all pairwise distances. This will take quadratic time in
        the number of samples. The sklearn.cluster.estimate_bandwidth function
        can be used to do this more efficiently.

    seeds : array-like of shape (n_seeds, n_features) or None
        Point used as initial kernel locations. If None and bin_seeding=False,
        each data point is used as a seed. If None and bin_seeding=True,
        see bin_seeding.

    bin_seeding : bool, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        Ignored if seeds argument is not None.

    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : bool, default=True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.
    """
    # max_iter参数表示每个种子点在聚类操作未收敛前的最大迭代次数。
    max_iter : int, default=300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

    # n_jobs参数指定用于计算的作业数量。以下任务可以从并行化中受益：
    # - 用于带宽估计和标签分配的最近邻搜索。详见NearestNeighbors类的文档字符串。
    # - 所有种子点的爬山优化。
    # 
    # 详情请参阅术语表中的“n_jobs”。
    # 
    # “None”表示除非处于joblib.parallel_backend上下文中，否则为1。
    # “-1”表示使用所有处理器。详情请参阅术语表中的“n_jobs”。
    # 
    # .. versionadded:: 0.17
    #    使用*n_jobs*进行并行执行。
    n_jobs : int, default=None
        The number of jobs to use for the computation. The following tasks benefit
        from the parallelization:

        - The search of nearest neighbors for bandwidth estimation and label
          assignments. See the details in the docstring of the
          ``NearestNeighbors`` class.
        - Hill-climbing optimization for all seeds.

        See :term:`Glossary <n_jobs>` for more details.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.17
           Parallel Execution using *n_jobs*.

    # 返回值
    # cluster_centers是形状为(n_clusters, n_features)的聚类中心的坐标。
    # labels是形状为(n_samples,)的每个样本点的聚类标签。
    Returns
    -------

    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    # 注意事项
    # ---------
    # 参见示例 :ref:`examples/cluster/plot_mean_shift.py
    # <sphx_glr_auto_examples_cluster_plot_mean_shift.py>` 查看详细示例。
    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_mean_shift.py
    <sphx_glr_auto_examples_cluster_plot_mean_shift.py>`.

    # 示例
    # --------
    # >>> import numpy as np
    # >>> from sklearn.cluster import mean_shift
    # >>> X = np.array([[1, 1], [2, 1], [1, 0],
    # ...               [4, 7], [3, 5], [3, 6]])
    # >>> cluster_centers, labels = mean_shift(X, bandwidth=2)
    # >>> cluster_centers
    # array([[3.33..., 6.     ],
    #        [1.33..., 0.66...]])
    # >>> labels
    # array([1, 1, 1, 0, 0, 0])
    """
    # 使用MeanShift对象进行聚类，基于给定的参数进行初始化和拟合
    model = MeanShift(
        bandwidth=bandwidth,
        seeds=seeds,
        min_bin_freq=min_bin_freq,
        bin_seeding=bin_seeding,
        cluster_all=cluster_all,
        n_jobs=n_jobs,
        max_iter=max_iter,
    ).fit(X)
    # 返回聚类后的聚类中心和样本点的标签
    return model.cluster_centers_, model.labels_
def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """Find seeds for mean_shift.

    Finds seeds by first binning data onto a grid whose lines are
    spaced bin_size apart, and then choosing those bins with at least
    min_bin_freq points.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Input points, the same points that will be used in mean_shift.

    bin_size : float
        Controls the coarseness of the binning. Smaller values lead
        to more seeding (which is computationally more expensive). If you're
        not sure how to set this, set it to the value of the bandwidth used
        in clustering.mean_shift.

    min_bin_freq : int, default=1
        Only bins with at least min_bin_freq will be selected as seeds.
        Raising this value decreases the number of seeds found, which
        makes mean_shift computationally cheaper.

    Returns
    -------
    bin_seeds : array-like of shape (n_samples, n_features)
        Points used as initial kernel positions in clustering.mean_shift.
    """
    # 如果 bin_size 为零，则直接返回原始数据 X
    if bin_size == 0:
        return X

    # 创建一个 defaultdict 用于计算每个 bin 的大小
    bin_sizes = defaultdict(int)
    
    # 遍历输入数据 X 中的每一个点
    for point in X:
        # 将点按照 bin_size 舍入到最接近的整数倍，以便进行分组
        binned_point = np.round(point / bin_size)
        # 统计每个分组中的点的数量
        bin_sizes[tuple(binned_point)] += 1

    # 从 bin_sizes 中选择那些点数大于等于 min_bin_freq 的分组作为种子点
    bin_seeds = np.array(
        [point for point, freq in bin_sizes.items() if freq >= min_bin_freq],
        dtype=np.float32,
    )
    
    # 如果选出的种子点数和原始数据 X 的点数相同，发出警告并返回原始数据 X
    if len(bin_seeds) == len(X):
        warnings.warn(
            "Binning data failed with provided bin_size=%f, using data points as seeds."
            % bin_size
        )
        return X
    
    # 将种子点坐标乘以 bin_size，以得到最终的种子点位置
    bin_seeds = bin_seeds * bin_size
    return bin_seeds
    bin_seeding : bool, default=False
        如果为 True，则初始核位置不是所有点的位置，而是点的离散化版本的位置，
        其中点被放置在一个网格上，其粗细程度对应于带宽。设置此选项为 True 将加速
        算法，因为初始化的种子数量将减少。
        默认值为 False。
        如果 seeds 参数不为 None，则忽略此选项。

    min_bin_freq : int, default=1
        为了加速算法，仅接受至少有 min_bin_freq 个点的那些箱子作为种子。

    cluster_all : bool, default=True
        如果为 True，则所有点都被聚类，即使那些不在任何核心内的孤立点也是如此。
        孤立点将被分配给最近的核心。如果为 False，则孤立点被赋予聚类标签 -1。

    n_jobs : int, default=None
        计算中使用的作业数量。以下任务受并行化的好处：

        - 用于带宽估计和标签分配的最近邻搜索。详细信息请参见“NearestNeighbors”类的文档字符串。
        - 所有种子的爬山优化。

        更多详细信息请参见术语表中的“n_jobs”。

        “None”表示除非在“joblib.parallel_backend”上下文中，否则为 1。
        “-1”表示使用所有处理器。更多详细信息请参见术语表中的“n_jobs”。

    max_iter : int, default=300
        每个种子点在聚类操作在其收敛之前的最大迭代次数。

        .. versionadded:: 0.22

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        聚类中心的坐标。

    labels_ : ndarray of shape (n_samples,)
        每个点的标签。

    n_iter_ : int
        在每个种子上执行的最大迭代次数。

        .. versionadded:: 0.22

    n_features_in_ : int
        在“fit”期间观察到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在“fit”期间观察到的特征的名称。仅当 `X` 的特征名全部为字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : K-Means 聚类算法。

    Notes
    -----

    Scalability:

    因为此实现使用平坦核和 Ball Tree 来查找每个核的成员，所以在较低维度中，复杂度将趋向于 O(T*n*log(n))，
    其中 n 为样本数，T 为点的数量。在更高维度中，复杂度将趋向于 O(T*n^2)。

    通过使用更少的种子来提高可扩展性，例如在 get_bin_seeds 函数中使用更高的 min_bin_freq 值。

    请注意，estimate_bandwidth 函数比
    """
        mean shift algorithm and will be the bottleneck if it is used.
    
        References
        ----------
    
        Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
        feature space analysis". IEEE Transactions on Pattern Analysis and
        Machine Intelligence. 2002. pp. 603-619.
    
        Examples
        --------
        >>> from sklearn.cluster import MeanShift
        >>> import numpy as np
        >>> X = np.array([[1, 1], [2, 1], [1, 0],
        ...               [4, 7], [3, 5], [3, 6]])
        >>> clustering = MeanShift(bandwidth=2).fit(X)
        >>> clustering.labels_
        array([1, 1, 1, 0, 0, 0])
        >>> clustering.predict([[0, 0], [5, 5]])
        array([1, 0])
        >>> clustering
        MeanShift(bandwidth=2)
        """
    
        # _parameter_constraints 定义了 MeanShift 类的参数约束
        _parameter_constraints: dict = {
            "bandwidth": [Interval(Real, 0, None, closed="neither"), None],
            "seeds": ["array-like", None],
            "bin_seeding": ["boolean"],
            "min_bin_freq": [Interval(Integral, 1, None, closed="left")],
            "cluster_all": ["boolean"],
            "n_jobs": [Integral, None],
            "max_iter": [Interval(Integral, 0, None, closed="left")],
        }
    
        # MeanShift 类的构造函数，初始化对象时设置参数
        def __init__(
            self,
            *,
            bandwidth=None,
            seeds=None,
            bin_seeding=False,
            min_bin_freq=1,
            cluster_all=True,
            n_jobs=None,
            max_iter=300,
        ):
            self.bandwidth = bandwidth  # 设置 bandwidth 参数
            self.seeds = seeds  # 设置 seeds 参数
            self.bin_seeding = bin_seeding  # 设置 bin_seeding 参数
            self.cluster_all = cluster_all  # 设置 cluster_all 参数
            self.min_bin_freq = min_bin_freq  # 设置 min_bin_freq 参数
            self.n_jobs = n_jobs  # 设置 n_jobs 参数
            self.max_iter = max_iter  # 设置 max_iter 参数
    
        # predict 方法用于预测输入样本所属的最近簇
        @_fit_context(prefer_skip_nested_validation=True)
        def predict(self, X):
            """Predict the closest cluster each sample in X belongs to.
    
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                New data to predict.
    
            Returns
            -------
            labels : ndarray of shape (n_samples,)
                Index of the cluster each sample belongs to.
            """
            check_is_fitted(self)  # 检查模型是否已拟合
            X = self._validate_data(X, reset=False)  # 验证输入数据
            with config_context(assume_finite=True):
                return pairwise_distances_argmin(X, self.cluster_centers_)
```