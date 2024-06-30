# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_kde.py`

```
# 为核密度估计实现创建一个基类可能的TODO，用于测试目的实现一个蛮力版本的TODO
# 作者信息
# 导入必要的库和模块
import itertools
# 导入用于判断数据类型的模块
from numbers import Integral, Real

# 导入科学计算相关的库和模块
import numpy as np
# 导入用于伽马函数的积分计算的函数
from scipy.special import gammainc

# 导入基本估计器的基类
from ..base import BaseEstimator, _fit_context
# 导入邻居模块中有效的距离度量
from ..neighbors._base import VALID_METRICS
# 导入用于检查随机状态的工具函数
from ..utils import check_random_state
# 导入用于参数验证的工具函数和类
from ..utils._param_validation import Interval, StrOptions
# 导入扩展数学计算相关的工具函数
from ..utils.extmath import row_norms
# 导入验证函数，用于检查样本权重和是否已拟合
from ..utils.validation import _check_sample_weight, check_is_fitted
# 导入BallTree算法类
from ._ball_tree import BallTree
# 导入KDTree算法类
from ._kd_tree import KDTree

# 有效的核函数选项列表
VALID_KERNELS = [
    "gaussian",
    "tophat",
    "epanechnikov",
    "exponential",
    "linear",
    "cosine",
]

# 树类型字典，映射树类型字符串到对应的树算法类
TREE_DICT = {"ball_tree": BallTree, "kd_tree": KDTree}
    # 表示在 `fit` 过程中观察到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 用于快速广义 N 点问题的树算法的实例，类型为 `BinaryTree`
    tree_ : ``BinaryTree`` instance
        The tree algorithm for fast generalized N-point problems.

    # 在 `fit` 过程中观察到的特征的名称数组，形状为 (`n_features_in_`,)
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    # 带宽的值，可以是直接给定的参数值，也可以是使用 'scott' 或 'silverman' 方法估计的值
    bandwidth_ : float
        Value of the bandwidth, given directly by the bandwidth parameter or
        estimated using the 'scott' or 'silverman' method.

        .. versionadded:: 1.0

    # 相关链接
    See Also
    --------
    sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
        problems.
    sklearn.neighbors.BallTree : Ball tree for fast generalized N-point
        problems.

    # 示例
    Examples
    --------
    Compute a gaussian kernel density estimate with a fixed bandwidth.

    >>> from sklearn.neighbors import KernelDensity
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X = rng.random_sample((100, 3))
    >>> kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    >>> log_density = kde.score_samples(X[:3])
    >>> log_density
    array([-1.52955942, -1.51462041, -1.60244657])
    """

    # 参数约束字典，用于描述 `_parameter_constraints` 属性
    _parameter_constraints: dict = {
        "bandwidth": [
            Interval(Real, 0, None, closed="neither"),  # 带宽的数值约束条件
            StrOptions({"scott", "silverman"}),  # 带宽的字符串选项
        ],
        "algorithm": [StrOptions(set(TREE_DICT.keys()) | {"auto"})],  # 算法选择的字符串选项
        "kernel": [StrOptions(set(VALID_KERNELS))],  # 核函数的字符串选项
        "metric": [
            StrOptions(
                set(itertools.chain(*[VALID_METRICS[alg] for alg in TREE_DICT.keys()]))
            )  # 度量方法的字符串选项
        ],
        "atol": [Interval(Real, 0, None, closed="left")],  # 绝对容差的数值约束条件
        "rtol": [Interval(Real, 0, None, closed="left")],  # 相对容差的数值约束条件
        "breadth_first": ["boolean"],  # 广度优先搜索的布尔值
        "leaf_size": [Interval(Integral, 1, None, closed="left")],  # 叶子大小的整数约束条件
        "metric_params": [None, dict],  # 度量参数，可以是 `None` 或字典类型
    }

    # 初始化方法，设置参数
    def __init__(
        self,
        *,
        bandwidth=1.0,  # 带宽，默认值为 1.0
        algorithm="auto",  # 算法，默认自动选择
        kernel="gaussian",  # 核函数，默认为高斯核
        metric="euclidean",  # 度量方法，默认为欧氏距离
        atol=0,  # 绝对容差，默认为 0
        rtol=0,  # 相对容差，默认为 0
        breadth_first=True,  # 广度优先搜索，默认为 True
        leaf_size=40,  # 叶子大小，默认为 40
        metric_params=None,  # 度量参数，默认为 None
    ):
        self.algorithm = algorithm
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params
    def _choose_algorithm(self, algorithm, metric):
        # 根据给定的算法字符串和度量字符串选择最优的算法来计算结果。
        if algorithm == "auto":
            # 如果度量在 KD 树的有效度量列表中，则使用 KD 树
            if metric in KDTree.valid_metrics:
                return "kd_tree"
            # 如果度量在 Ball 树的有效度量列表中，则使用 Ball 树
            elif metric in BallTree.valid_metrics:
                return "ball_tree"
        else:  # 如果算法是 kd_tree 或 ball_tree
            # 检查给定算法对应的有效度量列表中是否包含指定的度量，如果不包含则抛出 ValueError 异常
            if metric not in TREE_DICT[algorithm].valid_metrics:
                raise ValueError(
                    "invalid metric for {0}: '{1}'".format(TREE_DICT[algorithm], metric)
                )
            return algorithm

    @_fit_context(
        # KernelDensity.metric 尚未经过验证
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, sample_weight=None):
        """在数据上拟合核密度估计模型。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            n_features 维数据点的列表。每一行对应一个单独的数据点。

        y : None
            忽略。此参数仅用于与 :class:`~sklearn.pipeline.Pipeline` 兼容。

        sample_weight : array-like of shape (n_samples,), default=None
            附加到数据 X 的样本权重列表。

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            返回实例本身。
        """
        # 根据设定选择最优算法
        algorithm = self._choose_algorithm(self.algorithm, self.metric)

        # 如果带宽是字符串类型
        if isinstance(self.bandwidth, str):
            # 如果带宽是 "scott"
            if self.bandwidth == "scott":
                self.bandwidth_ = X.shape[0] ** (-1 / (X.shape[1] + 4))
            # 如果带宽是 "silverman"
            elif self.bandwidth == "silverman":
                self.bandwidth_ = (X.shape[0] * (X.shape[1] + 2) / 4) ** (
                    -1 / (X.shape[1] + 4)
                )
        else:
            self.bandwidth_ = self.bandwidth

        # 验证数据 X，确保其为 C 阶连续的 np.float64 类型
        X = self._validate_data(X, order="C", dtype=np.float64)

        # 如果有样本权重，则验证样本权重
        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=np.float64, only_non_negative=True
            )

        kwargs = self.metric_params
        if kwargs is None:
            kwargs = {}

        # 使用选定的算法和参数初始化树结构
        self.tree_ = TREE_DICT[algorithm](
            X,
            metric=self.metric,
            leaf_size=self.leaf_size,
            sample_weight=sample_weight,
            **kwargs,
        )
        return self
    def score_samples(self, X):
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        
        # 确保输入数据X符合要求，并转换为float64类型的数组
        X = self._validate_data(X, order="C", dtype=np.float64, reset=False)
        
        # 计算样本点的数量N
        if self.tree_.sample_weight is None:
            N = self.tree_.data.shape[0]
        else:
            N = self.tree_.sum_weight
        
        # 根据样本数量N调整公差atol_N
        atol_N = self.atol * N
        
        # 使用树结构计算核密度估计的对数概率密度
        log_density = self.tree_.kernel_density(
            X,
            h=self.bandwidth_,
            kernel=self.kernel,
            atol=atol_N,
            rtol=self.rtol,
            breadth_first=self.breadth_first,
            return_log=True,
        )
        
        # 对数概率密度减去样本数量N的对数
        log_density -= np.log(N)
        
        # 返回计算得到的对数概率密度
        return log_density

    def score(self, X, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        # 调用score_samples方法计算X数据集的对数概率密度的总和
        return np.sum(self.score_samples(X))
    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            List of samples.
        """
        # 确保模型已拟合
        check_is_fitted(self)
        
        # TODO: implement sampling for other valid kernel shapes
        # 如果核函数不是高斯核或者tophat核，抛出未实现错误
        if self.kernel not in ["gaussian", "tophat"]:
            raise NotImplementedError()

        # 将数据转换为NumPy数组
        data = np.asarray(self.tree_.data)

        # 确定随机数生成器
        rng = check_random_state(random_state)
        
        # 生成均匀分布的随机数
        u = rng.uniform(0, 1, size=n_samples)
        
        # 根据样本权重进行采样
        if self.tree_.sample_weight is None:
            i = (u * data.shape[0]).astype(np.int64)
        else:
            cumsum_weight = np.cumsum(np.asarray(self.tree_.sample_weight))
            sum_weight = cumsum_weight[-1]
            i = np.searchsorted(cumsum_weight, u * sum_weight)
        
        # 如果核函数是高斯核
        if self.kernel == "gaussian":
            # 生成服从正态分布的样本
            return np.atleast_2d(rng.normal(data[i], self.bandwidth_))

        # 如果核函数是tophat核
        elif self.kernel == "tophat":
            # 从d维正态分布中抽取点
            dim = data.shape[1]
            X = rng.normal(size=(n_samples, dim))
            s_sq = row_norms(X, squared=True)
            
            # 使用不完全Gamma函数将它们映射到均匀分布的tophat分布
            correction = (
                gammainc(0.5 * dim, 0.5 * s_sq) ** (1.0 / dim)
                * self.bandwidth_
                / np.sqrt(s_sq)
            )
            return data[i] + X * correction[:, np.newaxis]

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "sample_weight must have positive values"
                ),
            }
        }
```