# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_factor_analysis.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入数学模块中的对数和平方根函数
from math import log, sqrt
# 导入整数和实数类型验证模块
from numbers import Integral, Real

# 导入科学计算库中的numpy，并使用别名np
import numpy as np
# 导入线性代数模块中的函数
from scipy import linalg

# 导入基础估计器、特征名称前缀混合类和转换器混合类
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
# 导入迭代收敛警告异常类
from ..exceptions import ConvergenceWarning
# 导入工具函数，用于验证随机状态
from ..utils import check_random_state
# 导入参数验证模块中的区间和字符串选项验证
from ..utils._param_validation import Interval, StrOptions
# 导入扩展数学工具中的快速对数行列式计算、随机化奇异值分解和平方范数计算函数
from ..utils.extmath import fast_logdet, randomized_svd, squared_norm
# 导入验证模块中的检查拟合函数
from ..utils.validation import check_is_fitted

# 定义因子分析类，继承于类名前缀特征输出混合类、转换器混合类和基础估计器
class FactorAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """因子分析 (FA)。

    一个简单的线性生成模型，具有高斯潜变量。

    假设观测数据由低维潜在因子的线性变换和加性高斯噪声引起。
    在不失一般性的情况下，因子服从均值为零、单位协方差的高斯分布。噪声也是均值为零，
    并具有任意对角协方差矩阵。

    如果我们进一步限制模型，假设高斯噪声甚至是各向同性的（所有对角项相同），
    我们将得到:class:`PCA`。

    因子分析使用基于SVD的方法对所谓的“加载”矩阵进行最大似然估计，
    即潜在变量到观察变量的转换。

    详细信息请参阅 :ref:`用户指南 <FA>`。

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int, default=None
        潜在空间的维度，即在``transform``后得到的``X``的组件数量。
        如果为None，则n_components设置为特征的数量。

    tol : float, default=1e-2
        对数似然增加的停止容限。

    copy : bool, default=True
        是否复制X的副本。如果为``False``，则在拟合期间覆盖输入X。

    max_iter : int, default=1000
        最大迭代次数。

    noise_variance_init : array-like of shape (n_features,), default=None
        每个特征的噪声方差的初始猜测。
        如果为None，则默认为np.ones(n_features)。
    
    """
    # svd_method参数: {'lapack', 'randomized'}，默认为'randomized'
    # 使用哪种奇异值分解方法。如果选择'lapack'，则使用scipy.linalg中的标准SVD；如果选择'randomized'，则使用快速的randomized_svd函数。
    # 默认为'randomized'。对于大多数应用场景，'randomized'通常足够精确且提供显著的速度优势。
    # 可以通过设置较高的iterated_power值来提高精度。如果这仍然不够，为了最大精度，应选择'lapack'。

    iterated_power : int, default=3
        # iterated_power参数：int类型，默认为3
        # 功率方法的迭代次数。仅在svd_method参数为'randomized'时使用。

    rotation : {'varimax', 'quartimax'}, default=None
        # rotation参数：{'varimax', 'quartimax'}或None，默认为None
        # 如果不为None，则应用指定的旋转方法。目前实现了varimax和quartimax方法。

        # .. versionadded:: 0.24
        # 版本新增功能（自版本0.24起）

    random_state : int or RandomState instance, default=0
        # random_state参数：int或RandomState实例，默认为0
        # 仅在svd_method参数为'randomized'时使用。传递一个整数以确保多次函数调用产生可重现的结果。
        # 参见术语表中的"随机状态（random_state）"。

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        # components_属性：形状为(n_components, n_features)的ndarray
        # 具有最大方差的主成分。

    loglike_ : list of shape (n_iterations,)
        # loglike_属性：形状为(n_iterations,)的列表
        # 每次迭代的对数似然。

    noise_variance_ : ndarray of shape (n_features,)
        # noise_variance_属性：形状为(n_features,)的ndarray
        # 每个特征的估计噪声方差。

    n_iter_ : int
        # n_iter_属性：int类型
        # 运行的迭代次数。

    mean_ : ndarray of shape (n_features,)
        # mean_属性：形状为(n_features,)的ndarray
        # 从训练集中估计的每个特征的经验均值。

    n_features_in_ : int
        # n_features_in_属性：int类型
        # 在拟合过程中看到的特征数。

        # .. versionadded:: 0.24
        # 版本新增功能（自版本0.24起）

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # feature_names_in_属性：形状为(`n_features_in_`,)的ndarray
        # 在拟合过程中看到的特征名称。仅当X具有全部为字符串的特征名称时定义。

        # .. versionadded:: 1.0
        # 版本新增功能（自版本1.0起）

    See Also
    --------
    PCA: Principal component analysis也是一种潜在线性变量模型，但假设每个特征的噪声方差相等。
        这个额外的假设使得概率PCA更快，因为它可以以闭合形式计算。
    FastICA: 独立分量分析，一种具有非高斯潜在变量的潜在变量模型。

    References
    ----------
    - David Barber，《Bayesian Reasoning and Machine Learning》，算法21.1。
    - Christopher M. Bishop，《Pattern Recognition and Machine Learning》，第12.2.4节。

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import FactorAnalysis
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = FactorAnalysis(n_components=7, random_state=0)
    >>> X_transformed = transformer.fit_transform(X)
    # 对输入数据 X 进行拟合和转换，得到转换后的数据 X_transformed
    
    >>> X_transformed.shape
    # 打印转换后的数据 X_transformed 的形状
    (1797, 7)
    """
    
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 0, None, closed="left"), None],
        # 参数约束字典，指定了各个参数的类型和取值范围
        "tol": [Interval(Real, 0.0, None, closed="left")],
        # 参数 tol 的取值范围为大于等于 0.0 的实数
        "copy": ["boolean"],
        # 参数 copy 应为布尔值
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        # 参数 max_iter 的取值范围为大于等于 1 的整数
        "noise_variance_init": ["array-like", None],
        # 参数 noise_variance_init 可以是类数组对象或者为 None
        "svd_method": [StrOptions({"randomized", "lapack"})],
        # 参数 svd_method 应为 {"randomized", "lapack"} 中的一个字符串
        "iterated_power": [Interval(Integral, 0, None, closed="left")],
        # 参数 iterated_power 的取值范围为大于等于 0 的整数
        "rotation": [StrOptions({"varimax", "quartimax"}), None],
        # 参数 rotation 应为 {"varimax", "quartimax"} 中的一个字符串或者为 None
        "random_state": ["random_state"],
        # 参数 random_state 应为随机状态对象
    }
    
    def __init__(
        self,
        n_components=None,
        *,
        tol=1e-2,
        copy=True,
        max_iter=1000,
        noise_variance_init=None,
        svd_method="randomized",
        iterated_power=3,
        rotation=None,
        random_state=0,
    ):
        # 构造函数，初始化 FactorAnalysis 模型的各个参数
        self.n_components = n_components
        self.copy = copy
        self.tol = tol
        self.max_iter = max_iter
        self.svd_method = svd_method
        self.noise_variance_init = noise_variance_init
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.rotation = rotation
    
    @_fit_context(prefer_skip_nested_validation=True)
    def transform(self, X):
        """Apply dimensionality reduction to X using the model.
    
        Compute the expected mean of the latent variables.
        See Barber, 21.2.33 (or Bishop, 12.66).
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
    
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            The latent variables of X.
        """
        check_is_fitted(self)
    
        X = self._validate_data(X, reset=False)
        Ih = np.eye(len(self.components_))
    
        X_transformed = X - self.mean_
    
        Wpsi = self.components_ / self.noise_variance_
        cov_z = linalg.inv(Ih + np.dot(Wpsi, self.components_.T))
        tmp = np.dot(X_transformed, Wpsi.T)
        X_transformed = np.dot(tmp, cov_z)
    
        return X_transformed
    
    def get_covariance(self):
        """Compute data covariance with the FactorAnalysis model.
    
        ``cov = components_.T * components_ + diag(noise_variance)``
    
        Returns
        -------
        cov : ndarray of shape (n_features, n_features)
            Estimated covariance of data.
        """
        check_is_fitted(self)
    
        cov = np.dot(self.components_.T, self.components_)
        cov.flat[:: len(cov) + 1] += self.noise_variance_  # modify diag inplace
        return cov
    def get_precision(self):
        """Compute data precision matrix with the FactorAnalysis model.

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            Estimated precision of data.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 获取数据特征数
        n_features = self.components_.shape[1]

        # 处理特殊情况
        if self.n_components == 0:
            # 如果主成分数为0，返回噪声方差的逆矩阵的对角线
            return np.diag(1.0 / self.noise_variance_)
        if self.n_components == n_features:
            # 如果主成分数等于特征数，返回协方差矩阵的逆矩阵
            return linalg.inv(self.get_covariance())

        # 使用矩阵求逆引理计算数据精度
        components_ = self.components_
        precision = np.dot(components_ / self.noise_variance_, components_.T)
        precision.flat[:: len(precision) + 1] += 1.0
        precision = np.dot(components_.T, np.dot(linalg.inv(precision), components_))
        precision /= self.noise_variance_[:, np.newaxis]
        precision /= -self.noise_variance_[np.newaxis, :]
        precision.flat[:: len(precision) + 1] += 1.0 / self.noise_variance_
        return precision

    def score_samples(self, X):
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data.

        Returns
        -------
        ll : ndarray of shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 验证数据
        X = self._validate_data(X, reset=False)
        # 对数据进行中心化处理
        Xr = X - self.mean_
        # 获取数据精度矩阵
        precision = self.get_precision()
        # 获取特征数
        n_features = X.shape[1]
        # 计算每个样本的对数似然
        log_like = -0.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        log_like -= 0.5 * (n_features * log(2.0 * np.pi) - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """Compute the average log-likelihood of the samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data.

        y : Ignored
            Ignored parameter.

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model.
        """
        # 返回样本的平均对数似然
        return np.mean(self.score_samples(X))

    def _rotate(self, components, n_components=None, tol=1e-6):
        "Rotate the factor analysis solution."
        # 注意tol参数未被公开
        return _ortho_rotation(components.T, method=self.rotation, tol=tol)[
            : self.n_components
        ]

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # 返回转换后的输出特征数
        return self.components_.shape[0]
```python`
# 使用正交旋转方法对给定的成分矩阵进行旋转，返回旋转后的成分矩阵。
def _ortho_rotation(components, method="varimax", tol=1e-6, max_iter=100):
    """Return rotated components."""
    # 获取成分矩阵的行数和列数
    nrow, ncol = components.shape
    # 初始化旋转矩阵为单位矩阵
    rotation_matrix = np.eye(ncol)
    # 初始化方差
    var = 0

    # 迭代最大次数
    for _ in range(max_iter):
        # 计算旋转后的成分矩阵
        comp_rot = np.dot(components, rotation_matrix)
        # 根据旋转方法选择临时变量
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        # 进行奇异值分解
        u, s, v = np.linalg.svd(np.dot(components.T, comp_rot**3 - tmp))
        # 更新旋转矩阵
        rotation_matrix = np.dot(u, v)
        # 计算新的方差
        var_new = np.sum(s)
        # 检查是否满足收敛条件
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    # 返回旋转后的成分矩阵的转置
    return np.dot(components, rotation_matrix).T
```