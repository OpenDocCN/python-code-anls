# `D:\src\scipysrc\scikit-learn\sklearn\random_projection.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 引入警告模块，用于处理警告信息
import warnings
# 引入抽象基类模块及其方法
from abc import ABCMeta, abstractmethod
# 引入整数和实数类型
from numbers import Integral, Real

# 引入科学计算库NumPy
import numpy as np
# 引入稀疏矩阵处理模块
import scipy.sparse as sp
# 引入线性代数模块
from scipy import linalg

# 引入自定义的基础模块和类
from .base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
# 引入数据维度异常警告类
from .exceptions import DataDimensionalityWarning
# 引入工具函数检查随机状态
from .utils import check_random_state
# 引入参数验证模块中的一些验证方法
from .utils._param_validation import Interval, StrOptions, validate_params
# 引入数学运算扩展工具中的稀疏点积方法
from .utils.extmath import safe_sparse_dot
# 引入随机采样工具中的无重复采样方法
from .utils.random import sample_without_replacement
# 引入数据验证工具中的数据数组检查方法
from .utils.validation import check_array, check_is_fitted

# 定义导出的模块列表
__all__ = [
    "SparseRandomProjection",
    "GaussianRandomProjection",
    "johnson_lindenstrauss_min_dim",
]

# 参数验证装饰器函数
@validate_params(
    {
        "n_samples": ["array-like", Interval(Real, 1, None, closed="left")],
        "eps": ["array-like", Interval(Real, 0, 1, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
# 计算Johnson-Lindenstrauss定理中的最小维度
def johnson_lindenstrauss_min_dim(n_samples, *, eps=0.1):
    """Find a 'safe' number of components to randomly project to.

    The distortion introduced by a random projection `p` only changes the
    distance between two points by a factor (1 +- eps) in a euclidean space
    with good probability. The projection `p` is an eps-embedding as defined
    by:

      (1 - eps) ||u - v||^2 < ||p(u) - p(v)||^2 < (1 + eps) ||u - v||^2

    Where u and v are any rows taken from a dataset of shape (n_samples,
    n_features), eps is in ]0, 1[ and p is a projection by a random Gaussian
    N(0, 1) matrix of shape (n_components, n_features) (or a sparse
    Achlioptas matrix).

    The minimum number of components to guarantee the eps-embedding is
    given by:

      n_components >= 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3)
    """
    """
    Note that the number of dimensions is independent of the original
    number of features but instead depends on the size of the dataset:
    the larger the dataset, the higher is the minimal dimensionality of
    an eps-embedding.

    Read more in the :ref:`User Guide <johnson_lindenstrauss>`.

    Parameters
    ----------
    n_samples : int or array-like of int
        Number of samples that should be an integer greater than 0. If an array
        is given, it will compute a safe number of components array-wise.

    eps : float or array-like of shape (n_components,), dtype=float, \
            default=0.1
        Maximum distortion rate in the range (0, 1) as defined by the
        Johnson-Lindenstrauss lemma. If an array is given, it will compute a
        safe number of components array-wise.

    Returns
    -------
    n_components : int or ndarray of int
        The minimal number of components to guarantee with good probability
        an eps-embedding with n_samples.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

    .. [2] `Sanjoy Dasgupta and Anupam Gupta, 1999,
           "An elementary proof of the Johnson-Lindenstrauss Lemma."
           <https://citeseerx.ist.psu.edu/doc_view/pid/95cd464d27c25c9c8690b378b894d337cdf021f9>`_

    Examples
    --------
    >>> from sklearn.random_projection import johnson_lindenstrauss_min_dim
    >>> johnson_lindenstrauss_min_dim(1e6, eps=0.5)
    663

    >>> johnson_lindenstrauss_min_dim(1e6, eps=[0.5, 0.1, 0.01])
    array([    663,   11841, 1112658])

    >>> johnson_lindenstrauss_min_dim([1e4, 1e5, 1e6], eps=0.1)
    array([ 7894,  9868, 11841])
    """

    # 将 eps 和 n_samples 转换为 NumPy 数组，以便统一处理
    eps = np.asarray(eps)
    n_samples = np.asarray(n_samples)

    # 检查 eps 是否在合理范围内 (0, 1)
    if np.any(eps <= 0.0) or np.any(eps >= 1):
        raise ValueError("The JL bound is defined for eps in ]0, 1[, got %r" % eps)

    # 检查 n_samples 是否大于 0
    if np.any(n_samples <= 0):
        raise ValueError(
            "The JL bound is defined for n_samples greater than zero, got %r"
            % n_samples
        )

    # 计算 Johnson-Lindenstrauss lemma 中的分母部分
    denominator = (eps**2 / 2) - (eps**3 / 3)
    
    # 返回最小的维度数量，以保证在给定的 n_samples 下具有很高的概率实现 eps-embedding
    return (4 * np.log(n_samples) / denominator).astype(np.int64)
# 检查稀疏度参数是否在有效范围内，如果是 "auto"，则根据特征数自动设置密度
def _check_density(density, n_features):
    """Factorize density check according to Li et al."""
    if density == "auto":
        # 根据特征数设置自动密度
        density = 1 / np.sqrt(n_features)

    elif density <= 0 or density > 1:
        # 如果密度不在有效范围内，抛出值错误异常
        raise ValueError("Expected density in range ]0, 1], got: %r" % density)
    return density


# 检查输入尺寸参数是否合法，确保随机矩阵生成时参数正确
def _check_input_size(n_components, n_features):
    """Factorize argument checking for random matrix generation."""
    if n_components <= 0:
        # 如果主成分数小于等于0，抛出值错误异常
        raise ValueError(
            "n_components must be strictly positive, got %d" % n_components
        )
    if n_features <= 0:
        # 如果特征数小于等于0，抛出值错误异常
        raise ValueError("n_features must be strictly positive, got %d" % n_features)


# 生成一个稠密的高斯随机矩阵
def _gaussian_random_matrix(n_components, n_features, random_state=None):
    """Generate a dense Gaussian random matrix.

    The components of the random matrix are drawn from

        N(0, 1.0 / n_components).

    Read more in the :ref:`User Guide <gaussian_random_matrix>`.

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    components : ndarray of shape (n_components, n_features)
        The generated Gaussian random matrix.

    See Also
    --------
    GaussianRandomProjection
    """
    _check_input_size(n_components, n_features)
    # 检查随机数生成器状态，如果为None则创建一个新的
    rng = check_random_state(random_state)
    # 生成高斯随机矩阵，均值为0，标准差为1 / sqrt(n_components)
    components = rng.normal(
        loc=0.0, scale=1.0 / np.sqrt(n_components), size=(n_components, n_features)
    )
    return components


# 生成一个稀疏的随机矩阵，用于随机投影
def _sparse_random_matrix(n_components, n_features, density="auto", random_state=None):
    """Generalized Achlioptas random sparse matrix for random projection.

    Setting density to 1 / 3 will yield the original matrix by Dimitris
    Achlioptas while setting a lower value will yield the generalization
    by Ping Li et al.

    If we note :math:`s = 1 / density`, the components of the random matrix are
    drawn from:

      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Read more in the :ref:`User Guide <sparse_random_matrix>`.

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    density : float or str, default="auto",
        The density of non-zero components in the random matrix. Use "auto"
        to automatically set the density to 1 / sqrt(n_features).

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    components : ndarray of shape (n_components, n_features)
        The generated sparse random matrix.

    See Also
    --------
    SparseRandomProjection
    """
    # 检查输入的组件数量和特征数量是否合法
    _check_input_size(n_components, n_features)
    # 检查并设置随机投影矩阵的密度，确保在有效范围内
    density = _check_density(density, n_features)
    # 确定随机数生成器，用于在拟合时生成矩阵
    rng = check_random_state(random_state)

    # 如果密度为1，则完全稠密情况下跳过索引生成
    if density == 1:
        # 生成随机正负1值的矩阵，乘以1/sqrt(n_components)进行缩放
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return 1 / np.sqrt(n_components) * components

    else:
        # 生成非零元素的位置索引
        indices = []
        offset = 0
        indptr = [offset]
        for _ in range(n_components):
            # 对于第i行，找到非零元素的索引
            n_nonzero_i = rng.binomial(n_features, density)
            indices_i = sample_without_replacement(
                n_features, n_nonzero_i, random_state=rng
            )
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # 在非零元素中，正负号的概率各为50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # 通过连接行来构建CSR结构
        components = sp.csr_matrix(
            (data, indices, indptr), shape=(n_components, n_features)
        )

        # 返回经过缩放的稀疏矩阵
        return np.sqrt(1 / density) / np.sqrt(n_components) * components
class BaseRandomProjection(
    TransformerMixin, BaseEstimator, ClassNamePrefixFeaturesOutMixin, metaclass=ABCMeta
):
    """Base class for random projections.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    # 参数约束字典，指定各参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),  # n_components参数为正整数或"auto"
            StrOptions({"auto"}),  # n_components参数可以为"auto"
        ],
        "eps": [Interval(Real, 0, None, closed="neither")],  # eps参数为大于0的实数
        "compute_inverse_components": ["boolean"],  # compute_inverse_components参数为布尔值
        "random_state": ["random_state"],  # random_state参数为随机状态对象
    }

    @abstractmethod
    def __init__(
        self,
        n_components="auto",
        *,
        eps=0.1,
        compute_inverse_components=False,
        random_state=None,
    ):
        """Initialize the base random projection.

        Parameters
        ----------
        n_components : int or str, default='auto'
            Dimensionality of the target projection space, or 'auto' for automatic determination.

        eps : float, default=0.1
            Parameter controlling the quality of the projection.

        compute_inverse_components : bool, default=False
            Whether to compute the pseudo-inverse of the components.

        random_state : {None, int, array_like}, default=None
            Random number generator seed control.

        """
        # 设置各参数值
        self.n_components = n_components
        self.eps = eps
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state

    @abstractmethod
    def _make_random_matrix(self, n_components, n_features):
        """Generate the random projection matrix.

        Parameters
        ----------
        n_components : int
            Dimensionality of the target projection space.

        n_features : int
            Dimensionality of the original source space.

        Returns
        -------
        components : {ndarray, sparse matrix} of shape (n_components, n_features)
            The generated random matrix. Sparse matrix will be of CSR format.

        """
        # 抽象方法，生成随机投影矩阵的接口

    def _compute_inverse_components(self):
        """Compute the pseudo-inverse of the (densified) components."""
        # 计算(components的)伪逆
        components = self.components_
        if sp.issparse(components):  # 如果components是稀疏矩阵
            components = components.toarray()  # 转换为稠密矩阵
        return linalg.pinv(components, check_finite=False)  # 计算伪逆矩阵，允许非有限检查

    @_fit_context(prefer_skip_nested_validation=True)
    # 使用_fit_context装饰器，跳过嵌套验证首选项的上下文管理
    # 定义一个方法，用于拟合模型，生成稀疏随机投影矩阵。
    def fit(self, X, y=None):
        """Generate a sparse random projection matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training set: only the shape is used to find optimal random
            matrix dimensions based on the theory referenced in the
            afore mentioned papers.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            BaseRandomProjection class instance.
        """
        # 验证数据 X 的合法性，接受稀疏矩阵格式 "csr" 或 "csc"，数据类型可以是 np.float64 或 np.float32
        X = self._validate_data(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        # 获取样本数和特征数
        n_samples, n_features = X.shape

        # 如果 n_components 设为 "auto"，根据理论计算最小的随机投影维度
        if self.n_components == "auto":
            self.n_components_ = johnson_lindenstrauss_min_dim(
                n_samples=n_samples, eps=self.eps
            )

            # 如果计算出的 n_components_ 小于等于 0，则引发值错误
            if self.n_components_ <= 0:
                raise ValueError(
                    "eps=%f and n_samples=%d lead to a target dimension of "
                    "%d which is invalid" % (self.eps, n_samples, self.n_components_)
                )

            # 如果计算出的 n_components_ 大于特征数，则引发值错误
            elif self.n_components_ > n_features:
                raise ValueError(
                    "eps=%f and n_samples=%d lead to a target dimension of "
                    "%d which is larger than the original space with "
                    "n_features=%d"
                    % (self.eps, n_samples, self.n_components_, n_features)
                )
        else:
            # 如果指定了具体的 n_components，检查是否大于特征数，发出警告
            if self.n_components > n_features:
                warnings.warn(
                    "The number of components is higher than the number of"
                    " features: n_features < n_components (%s < %s)."
                    "The dimensionality of the problem will not be reduced."
                    % (n_features, self.n_components),
                    DataDimensionalityWarning,
                )

            # 将 n_components_ 设置为指定的 n_components 值
            self.n_components_ = self.n_components

        # 生成一个投影矩阵，大小为 [n_components, n_features]
        self.components_ = self._make_random_matrix(
            self.n_components_, n_features
        ).astype(X.dtype, copy=False)

        # 如果需要计算逆投影矩阵，则计算之
        if self.compute_inverse_components:
            self.inverse_components_ = self._compute_inverse_components()

        # 由于某些方法需要使用 self._n_features_out 获取输出特征数
        self._n_features_out = self.n_components

        # 返回自身对象
        return self
    def inverse_transform(self, X):
        """
        将数据投影回原始空间。

        返回一个数组 X_original，其转换后为 X。注意，即使 X 是稀疏的，X_original 是密集的：这可能会使用大量内存。

        如果 `compute_inverse_components` 为 False，在每次调用 `inverse_transform` 时将计算组件的逆，这可能代价高昂。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_components)
            要进行反向转换的数据。

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            重构的数据。
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 检查并转换输入数据 X，确保类型和稀疏性满足要求
        X = check_array(X, dtype=[np.float64, np.float32], accept_sparse=("csr", "csc"))

        # 如果设置为计算逆组件，则返回 X 乘以逆组件的转置
        if self.compute_inverse_components:
            return X @ self.inverse_components_.T

        # 否则，计算逆组件并返回 X 乘以逆组件的转置
        inverse_components = self._compute_inverse_components()
        return X @ inverse_components.T

    def _more_tags(self):
        """
        返回一个字典，指定了更多的标签。

        Returns
        -------
        dict
            包含更多标签信息的字典。
        """
        return {
            "preserves_dtype": [np.float64, np.float32],
        }
class GaussianRandomProjection(BaseRandomProjection):
    """Reduce dimensionality through Gaussian random projection.

    The components of the random matrix are drawn from N(0, 1 / n_components).

    Read more in the :ref:`User Guide <gaussian_random_matrix>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    eps : float, default=0.1
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when `n_components` is set to
        'auto'. The value should be strictly positive.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    compute_inverse_components : bool, default=False
        Learn the inverse transform by computing the pseudo-inverse of the
        components during fit. Note that computing the pseudo-inverse does not
        scale well to large matrices.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the
        projection matrix at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    n_components_ : int
        Concrete number of components computed when n_components="auto".

    components_ : ndarray of shape (n_components, n_features)
        Random matrix used for the projection.

    inverse_components_ : ndarray of shape (n_features, n_components)
        Pseudo-inverse of the components, only computed if
        `compute_inverse_components` is True.

        .. versionadded:: 1.1

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SparseRandomProjection : Reduce dimensionality through sparse
        random projection.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.random_projection import GaussianRandomProjection
    >>> rng = np.random.RandomState(42)
    >>> X = rng.rand(25, 3000)
    >>> transformer = GaussianRandomProjection(random_state=rng)

    """

    def __init__(self, n_components='auto', eps=0.1, compute_inverse_components=False, random_state=None):
        # 调用基类的初始化方法
        super().__init__()
        # 设置目标投影空间的维度
        self.n_components = n_components
        # 控制嵌入质量的参数，当 n_components 设置为 'auto' 时根据 Johnson-Lindenstrauss lemma 自动调整
        self.eps = eps
        # 是否计算反向变换的伪逆矩阵
        self.compute_inverse_components = compute_inverse_components
        # 控制随机数生成器的种子，以确保在多次调用中生成的投影矩阵是可复现的
        self.random_state = random_state

    def fit(self, X, y=None):
        # 在这里实现具体的拟合逻辑，用于计算随机投影矩阵
        pass

    def transform(self, X):
        # 使用已经学习到的随机投影矩阵来对数据 X 进行变换
        pass

    def fit_transform(self, X, y=None):
        # 结合拟合和变换的方法，通常会同时进行学习和转换
        pass
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (25, 2759)
    """
    # 对数据集 X 进行拟合和转换，生成新的转换后的数据集 X_new
    X_new = transformer.fit_transform(X)
    # 打印转换后的数据集 X_new 的形状
    >>> X_new.shape
    (25, 2759)
    """

    def __init__(
        self,
        n_components="auto",
        *,
        eps=0.1,
        compute_inverse_components=False,
        random_state=None,
    ):
        # 调用父类 RandomizedPCA 的初始化方法，并传入参数
        super().__init__(
            n_components=n_components,
            eps=eps,
            compute_inverse_components=compute_inverse_components,
            random_state=random_state,
        )

    def _make_random_matrix(self, n_components, n_features):
        """Generate the random projection matrix.

        Parameters
        ----------
        n_components : int,
            Dimensionality of the target projection space.

        n_features : int,
            Dimensionality of the original source space.

        Returns
        -------
        components : ndarray of shape (n_components, n_features)
            The generated random matrix.
        """
        # 检查随机状态并生成符合高斯分布的随机投影矩阵
        random_state = check_random_state(self.random_state)
        return _gaussian_random_matrix(
            n_components, n_features, random_state=random_state
        )

    def transform(self, X):
        """Project the data by using matrix product with the random matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected array.
        """
        # 确保模型已拟合
        check_is_fitted(self)
        # 验证输入数据 X 的有效性，并接受稀疏矩阵格式，不重置数据类型
        X = self._validate_data(
            X, accept_sparse=["csr", "csc"], reset=False, dtype=[np.float64, np.float32]
        )

        # 使用随机投影矩阵将数据 X 进行投影到较小维度的空间中
        return X @ self.components_.T
class SparseRandomProjection(BaseRandomProjection):
    """Reduce dimensionality through sparse random projection.

    Sparse random matrix is an alternative to dense random
    projection matrix that guarantees similar embedding quality while being
    much more memory efficient and allowing faster computation of the
    projected data.

    If we note `s = 1 / density` the components of the random matrix are
    drawn from:

      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Read more in the :ref:`User Guide <sparse_random_matrix>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    density : float or 'auto', default='auto'
        Ratio in the range (0, 1] of non-zero component in the random
        projection matrix.

        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).

        Use density = 1 / 3.0 if you want to reproduce the results from
        Achlioptas, 2001.

    eps : float, default=0.1
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'. This value should be strictly positive.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    dense_output : bool, default=False
        If True, ensure that the output of the random projection is a
        dense numpy array even if the input and random projection matrix
        are both sparse. In practice, if the number of components is
        small the number of zero components in the projected data will
        be very small and it will be more CPU and memory efficient to
        use a dense representation.

        If False, the projected data uses a sparse representation if
        the input is sparse.
    """
    compute_inverse_components : bool, default=False
        是否计算反向转换的组件。在拟合过程中通过计算组件的伪逆来学习反向转换。需要注意的是，即使训练数据是稀疏的，伪逆始终是一个密集的数组。这意味着可能需要分批次对少量样本调用 `inverse_transform`，以避免在主机上耗尽可用内存。此外，计算伪逆不适用于大矩阵。

    random_state : int, RandomState instance or None, default=None
        控制用于在拟合时生成投影矩阵的伪随机数生成器。为了在多次函数调用中产生可重复的输出，请传递一个整数。
        参见 :term:`Glossary <random_state>`。

    Attributes
    ----------
    n_components_ : int
        当 n_components="auto" 时计算的具体组件数量。

    components_ : sparse matrix of shape (n_components, n_features)
        用于投影的随机矩阵。稀疏矩阵将采用CSR格式。

    inverse_components_ : ndarray of shape (n_features, n_components)
        组件的伪逆，仅在 `compute_inverse_components` 为 True 时计算。

        .. versionadded:: 1.1

    density_ : float in range 0.0 - 1.0
        从 density = "auto" 时计算的具体密度。

    n_features_in_ : int
        在 :term:`fit` 过程中看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 :term:`fit` 过程中看到的特征名称。仅当 `X` 的特征名称全是字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    GaussianRandomProjection : 通过高斯随机投影减少维度。

    References
    ----------

    .. [1] Ping Li, T. Hastie and K. W. Church, 2006,
           "Very Sparse Random Projections".
           https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf

    .. [2] D. Achlioptas, 2001, "Database-friendly random projections",
           https://cgi.di.uoa.gr/~optas/papers/jl.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.random_projection import SparseRandomProjection
    >>> rng = np.random.RandomState(42)
    >>> X = rng.rand(25, 3000)
    >>> transformer = SparseRandomProjection(random_state=rng)
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (25, 2759)
    >>> # very few components are non-zero
    >>> np.mean(transformer.components_ != 0)
    0.0182...
    # 初始化方法，用于初始化随机投影器的参数
    def __init__(
        self,
        n_components="auto",  # 目标投影空间的维度，可以是"auto"或具体的整数值
        *,
        density="auto",  # 稀疏随机矩阵的密度，可以是"auto"或具体的浮点数
        eps=0.1,  # 随机投影矩阵的误差
        dense_output=False,  # 是否输出稠密的投影结果
        compute_inverse_components=False,  # 是否计算逆向投影分量
        random_state=None,  # 随机数生成器的种子值
    ):
        # 调用父类的初始化方法，设置通用的投影器参数
        super().__init__(
            n_components=n_components,
            eps=eps,
            compute_inverse_components=compute_inverse_components,
            random_state=random_state,
        )

        self.dense_output = dense_output  # 记录是否输出稠密的投影结果
        self.density = density  # 记录稀疏随机矩阵的密度

    # 生成随机投影矩阵的方法
    def _make_random_matrix(self, n_components, n_features):
        """Generate the random projection matrix

        Parameters
        ----------
        n_components : int
            Dimensionality of the target projection space.

        n_features : int
            Dimensionality of the original source space.

        Returns
        -------
        components : sparse matrix of shape (n_components, n_features)
            The generated random matrix in CSR format.

        """
        random_state = check_random_state(self.random_state)
        self.density_ = _check_density(self.density, n_features)  # 检查并设置稀疏矩阵的密度
        return _sparse_random_matrix(
            n_components, n_features, density=self.density_, random_state=random_state
        )

    # 数据投影方法，使用随机矩阵进行矩阵乘法投影
    def transform(self, X):
        """Project the data by using matrix product with the random matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Projected array. It is a sparse matrix only when the input is sparse and
            `dense_output = False`.
        """
        check_is_fitted(self)  # 检查投影器是否已经适配了数据
        X = self._validate_data(
            X, accept_sparse=["csr", "csc"], reset=False, dtype=[np.float64, np.float32]
        )  # 验证输入数据的格式和类型

        return safe_sparse_dot(X, self.components_.T, dense_output=self.dense_output)
```