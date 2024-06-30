# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_truncated_svd.py`

```
"""Truncated SVD for sparse matrices, aka latent semantic analysis (LSA)."""

# Author: Lars Buitinck
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Michael Becker <mike@beckerfuffle.com>
# License: 3-clause BSD.

# 导入必要的库
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# 导入基类和辅助函数
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..utils import check_array, check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import randomized_svd, safe_sparse_dot, svd_flip
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import check_is_fitted

__all__ = ["TruncatedSVD"]


class TruncatedSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Dimensionality reduction using truncated SVD (aka LSA).

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition. This means it can work with sparse matrices
    efficiently.

    In particular, truncated SVD works on term count/tf-idf matrices as
    returned by the vectorizers in :mod:`sklearn.feature_extraction.text`. In
    that context, it is known as latent semantic analysis (LSA).

    This estimator supports two algorithms: a fast randomized SVD solver, and
    a "naive" algorithm that uses ARPACK as an eigensolver on `X * X.T` or
    `X.T * X`, whichever is more efficient.

    Read more in the :ref:`User Guide <LSA>`.

    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data.
        If algorithm='arpack', must be strictly less than the number of features.
        If algorithm='randomized', must be less than or equal to the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    algorithm : {'arpack', 'randomized'}, default='randomized'
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    n_iter : int, default=5
        Number of iterations for randomized SVD solver. Not used by ARPACK. The
        default is larger than the default in
        :func:`~sklearn.utils.extmath.randomized_svd` to handle sparse
        matrices that may have large slowly decaying spectrum.

    n_oversamples : int, default=10
        Number of oversamples for randomized SVD solver. Not used by ARPACK.
        See :func:`~sklearn.utils.extmath.randomized_svd` for a complete
        description.

        .. versionadded:: 1.1
"""
    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Power iteration normalizer for randomized SVD solver.
        Not used by ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd`
        for more details.

        .. versionadded:: 1.1

    random_state : int, RandomState instance or None, default=None
        Used during randomized svd. Pass an int for reproducible results across
        multiple function calls.
        See :term:`Glossary <random_state>`.

        随机种子或随机状态对象，用于控制随机化SVD过程中的随机性，传递一个整数以确保多次函数调用时结果可重现。

    tol : float, default=0.0
        Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
        SVD solver.

        ARPACK的容差值。0表示机器精度。在随机化SVD求解器中被忽略。

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The right singular vectors of the input data.

        输入数据的右奇异向量。

    explained_variance_ : ndarray of shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.

        通过投影到每个成分后转换的训练样本的方差。

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

        每个选定成分解释的方差百分比。

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

        对应于每个选定成分的奇异值。奇异值等于低维空间中``n_components``变量的二范数。

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        训练期间观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        训练期间观察到的特征名称。仅在`X`具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    FactorAnalysis : A simple linear generative model with
        Gaussian latent variables.
    IncrementalPCA : Incremental principal components analysis.
    KernelPCA : Kernel Principal component analysis.
    NMF : Non-Negative Matrix Factorization.
    PCA : Principal component analysis.

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.

    SVD存在一种称为“符号不定性”的问题，这意味着``components_``的符号和transform的输出取决于算法和随机状态。为了解决这个问题，应该对数据的实例进行一次拟合，然后保留该实例以执行变换。

    References
    ----------
    :arxiv:`Halko, et al. (2009). "Finding structure with randomness:
    Stochastic algorithms for constructing approximate matrix decompositions"
    <0909.4061>`

    Examples
    --------
    >>> from sklearn.decomposition import TruncatedSVD
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_dense = np.random.rand(100, 100)
    >>> X_dense[:, 2 * np.arange(50)] = 0
    >>> X = csr_matrix(X_dense)
    >>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> svd.fit(X)
    TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> print(svd.explained_variance_ratio_)
    [0.0157... 0.0512... 0.0499... 0.0479... 0.0453...]
    >>> print(svd.explained_variance_ratio_.sum())
    0.2102...
    >>> print(svd.singular_values_)
    [35.2410...  4.5981...   4.5420...  4.4486...  4.3288...]

    """
    这段代码片段是关于TruncatedSVD（截断奇异值分解）类的演示，用于降维和特征提取。

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "algorithm": [StrOptions({"arpack", "randomized"})],
        "n_iter": [Interval(Integral, 0, None, closed="left")],
        "n_oversamples": [Interval(Integral, 1, None, closed="left")],
        "power_iteration_normalizer": [StrOptions({"auto", "OR", "LU", "none"})],
        "random_state": ["random_state"],
        "tol": [Interval(Real, 0, None, closed="left")],
    }
    这是一个参数约束字典，定义了TruncatedSVD类的各个参数的取值约束。

    def __init__(
        self,
        n_components=2,
        *,
        algorithm="randomized",
        n_iter=5,
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
        tol=0.0,
    ):
        这是TruncatedSVD类的构造函数，用于初始化对象的各个参数。

    def fit(self, X, y=None):
        """Fit model on training data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        self.fit_transform(X)
        这是TruncatedSVD类的fit方法，用于在训练数据X上拟合模型。

    @_fit_context(prefer_skip_nested_validation=True)
    这是一个装饰器，可能用于内部逻辑控制或验证。
    def fit_transform(self, X, y=None):
        """Fit model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        # Validate input data X, ensuring it's a sparse matrix if specified
        X = self._validate_data(X, accept_sparse=["csr", "csc"], ensure_min_features=2)
        
        # Set up random state for reproducibility
        random_state = check_random_state(self.random_state)

        # Perform SVD based on chosen algorithm
        if self.algorithm == "arpack":
            # Initialize starting vector v0 for ARPACK
            v0 = _init_arpack_v0(min(X.shape), random_state)
            # Perform SVD using ARPACK algorithm
            U, Sigma, VT = svds(X, k=self.n_components, tol=self.tol, v0=v0)
            # Reverse the order of singular values Sigma to match conventions
            Sigma = Sigma[::-1]
            # Ensure consistency with PCA by flipping signs of components
            U, VT = svd_flip(U[:, ::-1], VT[::-1], u_based_decision=False)

        elif self.algorithm == "randomized":
            # Check if n_components is valid given the number of features
            if self.n_components > X.shape[1]:
                raise ValueError(
                    f"n_components({self.n_components}) must be <="
                    f" n_features({X.shape[1]})."
                )
            # Perform randomized SVD
            U, Sigma, VT = randomized_svd(
                X,
                self.n_components,
                n_iter=self.n_iter,
                n_oversamples=self.n_oversamples,
                power_iteration_normalizer=self.power_iteration_normalizer,
                random_state=random_state,
                flip_sign=False,
            )
            # Ensure consistency with PCA by flipping signs of components
            U, VT = svd_flip(U, VT, u_based_decision=False)

        # Store the principal components in self.components_
        self.components_ = VT

        # Calculate transformed data based on SVD approximation
        if self.algorithm == "randomized" or (
            self.algorithm == "arpack" and self.tol > 0
        ):
            # Project X onto the components to obtain transformed X
            X_transformed = safe_sparse_dot(X, self.components_.T)
        else:
            # Directly compute transformed X based on U and Sigma
            X_transformed = U * Sigma

        # Calculate explained variance and explained variance ratio
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        if sp.issparse(X):
            # For sparse matrix X, calculate total variance differently
            _, full_var = mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            # For dense matrix X, sum up variances across features
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = Sigma  # Store the singular values.

        # Return the transformed data X_transformed
        return X_transformed
    # 执行对输入数据 X 的降维操作
    def transform(self, X):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        # 检查模型是否已拟合（适用于 sklearn 的检查方法）
        check_is_fitted(self)
        # 验证输入数据 X 的格式，接受稀疏矩阵格式 "csr" 或 "csc"，不重置数据
        X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)
        # 使用安全的稀疏矩阵乘法计算降维后的数据
        return safe_sparse_dot(X, self.components_.T)

    # 将降维后的数据 X 还原到原始空间
    def inverse_transform(self, X):
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        # 检查并确保输入数据 X 符合数组格式
        X = check_array(X)
        # 使用矩阵乘法将 X 转换回原始空间的数据
        return np.dot(X, self.components_)

    # 返回额外的标签信息，指定输出数据类型为浮点数
    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}

    # 返回转换后输出的特征数量
    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components_.shape[0]
```