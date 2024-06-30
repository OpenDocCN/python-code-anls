# `D:\src\scipysrc\scikit-learn\sklearn\kernel_approximation.py`

```
"""Approximate kernel feature maps based on Fourier transforms and count sketches."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings                                    # 导入警告模块
from numbers import Integral, Real                  # 导入整数和实数类

import numpy as np                                 # 导入NumPy库
import scipy.sparse as sp                           # 导入SciPy稀疏矩阵模块
from scipy.linalg import svd                        # 导入SciPy线性代数模块的奇异值分解函数

try:
    from scipy.fft import fft, ifft                # 尝试导入SciPy的快速傅里叶变换函数
except ImportError:                                # 如果导入错误（一般是因为SciPy版本低于1.4）
    from scipy.fftpack import fft, ifft            # 则导入旧版本的傅里叶变换函数

from .base import (                                # 从当前包中导入基础模块
    BaseEstimator,                                 # 基础评估器类
    ClassNamePrefixFeaturesOutMixin,               # 类名前缀特征输出混合类
    TransformerMixin,                              # 转换器混合类
    _fit_context,                                  # 拟合上下文函数
)
from .metrics.pairwise import (                    # 从当前包中导入度量模块的成对函数
    KERNEL_PARAMS,                                 # 核参数
    PAIRWISE_KERNEL_FUNCTIONS,                     # 成对核函数
    pairwise_kernels                               # 成对核函数计算
)
from .utils import check_random_state              # 从当前包中导入随机状态检查函数
from .utils._param_validation import (              # 从当前包中导入参数验证模块
    Interval,                                      # 区间类
    StrOptions                                     # 字符串选项类
)
from .utils.extmath import safe_sparse_dot         # 从当前包中导入稀疏矩阵安全点积函数
from .utils.validation import (                    # 从当前包中导入验证模块的验证函数
    _check_feature_names_in,                       # 检查特征名称在内的函数
    check_is_fitted,                               # 检查是否拟合函数
    check_non_negative                             # 检查非负函数
)


class PolynomialCountSketch(
    ClassNamePrefixFeaturesOutMixin,                # 类名前缀特征输出混合类
    TransformerMixin,                              # 转换器混合类
    BaseEstimator                                 # 基础评估器类
):
    """Polynomial kernel approximation via Tensor Sketch.

    Implements Tensor Sketch, which approximates the feature map
    of the polynomial kernel::

        K(X, Y) = (gamma * <X, Y> + coef0)^degree

    by efficiently computing a Count Sketch of the outer product of a
    vector with itself using Fast Fourier Transforms (FFT). Read more in the
    :ref:`User Guide <polynomial_kernel_approx>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    gamma : float, default=1.0
        Parameter of the polynomial kernel whose feature map
        will be approximated.

    degree : int, default=2
        Degree of the polynomial kernel whose feature map
        will be approximated.

    coef0 : int, default=0
        Constant term of the polynomial kernel whose feature map
        will be approximated.

    n_components : int, default=100
        Dimensionality of the output feature space. Usually, `n_components`
        should be greater than the number of features in input samples in
        order to achieve good performance. The optimal score / run time
        balance is typically achieved around `n_components` = 10 * `n_features`,
        but this depends on the specific dataset being used.

    random_state : int, RandomState instance, default=None
        Determines random number generation for indexHash and bitHash
        initialization. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

    Attributes
    ----------
    indexHash_ : ndarray of shape (degree, n_features), dtype=int64
        Array of indexes in range [0, n_components) used to represent
        the 2-wise independent hash functions for Count Sketch computation.

    bitHash_ : ndarray of shape (degree, n_features), dtype=float32
        Array with random entries in {+1, -1}, used to represent
        the 2-wise independent hash functions for Count Sketch computation.
"""
    # Number of features seen during fitting process
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # Names of features seen during fitting process
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # Related classes and functions
    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    Nystroem : Approximate a kernel map using a subset of the training data.
    RBFSampler : Approximate a RBF kernel feature map using random Fourier
        features.
    SkewedChi2Sampler : Approximate feature map for "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    # Example usage of the class PolynomialCountSketch
    Examples
    --------
    >>> from sklearn.kernel_approximation import PolynomialCountSketch
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> ps = PolynomialCountSketch(degree=3, random_state=1)
    >>> X_features = ps.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=10, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=10)
    >>> clf.score(X_features, y)
    1.0

    # Link to a detailed usage example
    For a more detailed example of usage, see
    :ref:`sphx_glr_auto_examples_kernel_approximation_plot_scalable_poly_kernels.py`
    """

    # Constraints on parameters used by the class
    _parameter_constraints: dict = {
        "gamma": [Interval(Real, 0, None, closed="left")],
        "degree": [Interval(Integral, 1, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    # Constructor for initializing the class
    def __init__(
        self, *, gamma=1.0, degree=2, coef0=0, n_components=100, random_state=None
    ):
        self.gamma = gamma  # Assigning gamma parameter
        self.degree = degree  # Assigning degree parameter
        self.coef0 = coef0  # Assigning coef0 parameter
        self.n_components = n_components  # Assigning n_components parameter
        self.random_state = random_state  # Assigning random_state parameter

    # Decorator for fitting context with nested validation skip preference
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the model with X.

        Initializes the internal variables. The method needs no information
        about the distribution of data, so we only care about n_features in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate the input data X, accepting sparse matrices in CSC format
        X = self._validate_data(X, accept_sparse="csc")
        
        # Initialize the random state for consistent random number generation
        random_state = check_random_state(self.random_state)

        # Determine the number of features in the input data
        n_features = X.shape[1]
        
        # Adjust the number of features if coef0 is not zero
        if self.coef0 != 0:
            n_features += 1

        # Generate index hash for hashing features
        self.indexHash_ = random_state.randint(
            0, high=self.n_components, size=(self.degree, n_features)
        )

        # Generate bit hash for hashing features
        self.bitHash_ = random_state.choice(a=[-1, 1], size=(self.degree, n_features))
        
        # Set the number of output features to be equal to n_components
        self._n_features_out = self.n_components
        
        # Return the instance of the object itself
        return self
    def transform(self, X):
        """生成X的特征映射近似。

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            新数据，其中 `n_samples` 是样本数量，`n_features` 是特征数量。

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            返回实例本身。
        """

        # 检查模型是否已拟合
        check_is_fitted(self)
        # 验证并处理输入数据X，接受稀疏矩阵格式"csc"，不重置数据
        X = self._validate_data(X, accept_sparse="csc", reset=False)

        # 对输入数据应用 gamma 的平方根乘积
        X_gamma = np.sqrt(self.gamma) * X

        # 如果X_gamma是稀疏矩阵且 coef0 不等于 0
        if sp.issparse(X_gamma) and self.coef0 != 0:
            # 将一个由 coef0 根号乘以1数组与X_gamma水平堆叠，格式为"csc"
            X_gamma = sp.hstack(
                [X_gamma, np.sqrt(self.coef0) * np.ones((X_gamma.shape[0], 1))],
                format="csc",
            )

        # 如果X_gamma不是稀疏矩阵且 coef0 不等于 0
        elif not sp.issparse(X_gamma) and self.coef0 != 0:
            # 将一个由 coef0 根号乘以1数组与X_gamma水平堆叠
            X_gamma = np.hstack(
                [X_gamma, np.sqrt(self.coef0) * np.ones((X_gamma.shape[0], 1))]
            )

        # 如果X_gamma的特征数量不等于模型存储的 indexHash_ 的特征数量，抛出异常
        if X_gamma.shape[1] != self.indexHash_.shape[1]:
            raise ValueError(
                "测试样本的特征数与训练样本不匹配。"
            )

        # 初始化一个三维数组用于存储计数草图
        count_sketches = np.zeros((X_gamma.shape[0], self.degree, self.n_components))

        # 如果X_gamma是稀疏矩阵
        if sp.issparse(X_gamma):
            # 遍历X_gamma的每一列特征
            for j in range(X_gamma.shape[1]):
                # 遍历草图的度数
                for d in range(self.degree):
                    # 获取索引哈希值和位哈希值
                    iHashIndex = self.indexHash_[d, j]
                    iHashBit = self.bitHash_[d, j]
                    # 将计数草图的对应位置增加计算结果
                    count_sketches[:, d, iHashIndex] += (
                        (iHashBit * X_gamma[:, [j]]).toarray().ravel()
                    )

        # 如果X_gamma不是稀疏矩阵
        else:
            # 遍历X_gamma的每一列特征
            for j in range(X_gamma.shape[1]):
                # 遍历草图的度数
                for d in range(self.degree):
                    # 获取索引哈希值和位哈希值
                    iHashIndex = self.indexHash_[d, j]
                    iHashBit = self.bitHash_[d, j]
                    # 将计数草图的对应位置增加计算结果
                    count_sketches[:, d, iHashIndex] += iHashBit * X_gamma[:, j]

        # 对每个草图使用FFT计算乘积，生成计数草图的FFT
        count_sketches_fft = fft(count_sketches, axis=2, overwrite_x=True)
        # 对计数草图的FFT进行所有草图的乘积
        count_sketches_fft_prod = np.prod(count_sketches_fft, axis=1)
        # 执行逆FFT，生成实数的数据草图
        data_sketch = np.real(ifft(count_sketches_fft_prod, overwrite_x=True))

        # 返回数据草图
        return data_sketch
class RBFSampler(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Approximate a RBF kernel feature map using random Fourier features.

    It implements a variant of Random Kitchen Sinks.[1]

    Read more in the :ref:`User Guide <rbf_kernel_approx>`.

    Parameters
    ----------
    gamma : 'scale' or float, default=1.0
        Parameter of RBF kernel: exp(-gamma * x^2).
        If ``gamma='scale'`` is passed then it uses
        1 / (n_features * X.var()) as value of gamma.

        .. versionadded:: 1.2
           The option `"scale"` was added in 1.2.

    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    random_offset_ : ndarray of shape (n_components,), dtype={np.float64, np.float32}
        Random offset used to compute the projection in the `n_components`
        dimensions of the feature space.

    random_weights_ : ndarray of shape (n_features, n_components),\
        dtype={np.float64, np.float32}
        Random projection directions drawn from the Fourier transform
        of the RBF kernel.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    Nystroem : Approximate a kernel map using a subset of the training data.
    PolynomialCountSketch : Polynomial kernel approximation via Tensor Sketch.
    SkewedChi2Sampler : Approximate feature map for
        "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.

    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)

    Examples
    --------
    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSampler(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5, tol=1e-3)

    """

    def __init__(self, gamma=1.0, n_components=100, random_state=None):
        """Initialize the RBFSampler.

        Parameters
        ----------
        gamma : 'scale' or float, default=1.0
            Parameter of RBF kernel: exp(-gamma * x^2).

        n_components : int, default=100
            Number of Monte Carlo samples per original feature.

        random_state : int, RandomState instance or None, default=None
            Pseudo-random number generator for reproducible output.

        """
        # 调用父类的构造函数，设置初始参数
        super(RBFSampler, self).__init__()

        # 设置 gamma 参数
        self.gamma = gamma

        # 设置 n_components 参数
        self.n_components = n_components

        # 设置 random_state 参数
        self.random_state = random_state

        # 初始化用于计算投影的随机偏移
        self.random_offset_ = None

        # 初始化用于计算投影的随机权重
        self.random_weights_ = None

        # 初始化输入特征数
        self.n_features_in_ = None

        # 初始化特征名称数组
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """Fit the RBFSampler to the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : None
            Ignored. This parameter exists only for compatibility with
            Scikit-learn pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 获取输入特征数目
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else len(X[0])

        # 为输入特征设置名称数组
        if isinstance(X, np.ndarray) and X.dtype.names:
            self.feature_names_in_ = np.array(X.dtype.names)
        elif hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)

        # 随机生成投影的偏移量
        rng = check_random_state(self.random_state)
        self.random_offset_ = rng.uniform(size=self.n_components) * 2 * np.pi

        # 随机生成投影的权重矩阵
        self.random_weights_ = rng.normal(
            size=(self.n_features_in_, self.n_components)
        ) / np.sqrt(self.n_components)

        return self

    def transform(self, X):
        """Apply the RBFSampler to the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # 计算变换后的特征
        X_transformed = np.sqrt(2.0 / self.n_components) * np.cos(
            np.dot(X, self.random_weights_) + self.random_offset_
        )

        return X_transformed
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=5)
    >>> clf.score(X_features, y)
    1.0
    """
    # 调用分类器对象的 fit 方法，用训练数据 X_features 和目标标签 y 进行模型训练
    # 返回一个已经训练好的 SGDClassifier 对象，最大迭代次数为 5

    _parameter_constraints: dict = {
        "gamma": [
            StrOptions({"scale"}),  # gamma 参数的约束条件，可以是字符串 "scale"
            Interval(Real, 0.0, None, closed="left"),  # gamma 参数的约束条件，是大于等于 0 的实数
        ],
        "n_components": [Interval(Integral, 1, None, closed="left")],  # n_components 参数的约束条件，是大于等于 1 的整数
        "random_state": ["random_state"],  # random_state 参数的约束条件，必须是字符串 "random_state"
    }

    def __init__(self, *, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma  # 初始化对象的 gamma 属性
        self.n_components = n_components  # 初始化对象的 n_components 属性
        self.random_state = random_state  # 初始化对象的 random_state 属性

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X.

        Samples random projection according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, accept_sparse="csr")  # 使用 _validate_data 方法验证输入数据 X，接受稀疏矩阵 csr 格式
        random_state = check_random_state(self.random_state)  # 检查并返回一个随机状态对象
        n_features = X.shape[1]  # 获取输入数据 X 的特征数
        sparse = sp.issparse(X)  # 检查输入数据 X 是否为稀疏矩阵
        if self.gamma == "scale":
            # 如果 gamma 参数为 "scale"，根据输入数据 X 的稀疏性计算方差
            X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
            self._gamma = 1.0 / (n_features * X_var) if X_var != 0 else 1.0
        else:
            self._gamma = self.gamma  # 否则直接使用给定的 gamma 参数值
        # 生成随机权重矩阵，用于随机投影
        self.random_weights_ = (2.0 * self._gamma) ** 0.5 * random_state.normal(
            size=(n_features, self.n_components)
        )

        # 生成随机偏移量数组
        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)

        if X.dtype == np.float32:
            # 如果输入数据 X 的数据类型是 np.float32，将随机权重和偏移量的数据类型设置为相同类型
            self.random_weights_ = self.random_weights_.astype(X.dtype, copy=False)
            self.random_offset_ = self.random_offset_.astype(X.dtype, copy=False)

        self._n_features_out = self.n_components  # 设置输出特征数为 n_components
        return self  # 返回对象本身，表示训练完成
    # 对象方法：对输入数据集 X 应用近似特征映射。

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        新数据集，其中 `n_samples` 是样本数，`n_features` 是特征数。

    Returns
    -------
    X_new : array-like, shape (n_samples, n_components)
        返回转换后的数据集。

    """
    # 检查是否已拟合，确保模型已经训练完成
    check_is_fitted(self)

    # 验证数据 X，并将其转换为稀疏矩阵格式（压缩稀疏行格式）保留现场数据
    X = self._validate_data(X, accept_sparse="csr", reset=False)

    # 使用随机权重向量对数据 X 进行投影
    projection = safe_sparse_dot(X, self.random_weights_)

    # 添加随机偏移量到投影结果中
    projection += self.random_offset_

    # 对投影结果应用余弦函数映射
    np.cos(projection, projection)

    # 将投影结果缩放以进行特征映射的归一化
    projection *= (2.0 / self.n_components) ** 0.5

    # 返回映射后的数据集
    return projection

# 对象方法：返回有关模型更多标签信息，表明该方法保留指定数据类型的精度
def _more_tags(self):
    return {"preserves_dtype": [np.float64, np.float32]}
# 定义一个名为 SkewedChi2Sampler 的类，它继承自 ClassNamePrefixFeaturesOutMixin、TransformerMixin 和 BaseEstimator
class SkewedChi2Sampler(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Approximate feature map for "skewed chi-squared" kernel.
    
    "偏斜卡方"核的近似特征映射类。

    Read more in the :ref:`User Guide <skewed_chi_kernel_approx>`.
    
    在用户指南中详细了解：ref:`User Guide <skewed_chi_kernel_approx>`。

    Parameters
    ----------
    skewedness : float, default=1.0
        "skewedness" parameter of the kernel. Needs to be cross-validated.
        
        核的"偏斜度"参数。需要交叉验证。

    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
        
        每个原始特征的蒙特卡洛样本数。等于计算特征空间的维度。

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
        
        伪随机数生成器，用于控制在拟合训练数据时生成随机权重和随机偏移量。
        传递一个整数以便在多次函数调用中获得可重复的输出。
        参见：term:`Glossary <random_state>`。

    Attributes
    ----------
    random_weights_ : ndarray of shape (n_features, n_components)
        Weight array, sampled from a secant hyperbolic distribution, which will
        be used to linearly transform the log of the data.
        
        权重数组，从切线双曲线分布中采样，用于线性转换数据的对数。

    random_offset_ : ndarray of shape (n_features, n_components)
        Bias term, which will be added to the data. It is uniformly distributed
        between 0 and 2*pi.
        
        偏置项，将添加到数据中。均匀分布在0到2*pi之间。

    n_features_in_ : int
        Number of features seen during :term:`fit`.
        
        在`fit`期间看到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        
        在`fit`期间看到的特征名称。仅在`X`具有全部是字符串的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    Nystroem : Approximate a kernel map using a subset of the training data.
    RBFSampler : Approximate a RBF kernel feature map using random Fourier
        features.
    SkewedChi2Sampler : Approximate feature map for "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
    
    参见：
    - AdditiveChi2Sampler：加性卡方核的近似特征映射。
    - Nystroem：使用训练数据子集近似核映射。
    - RBFSampler：使用随机傅里叶特征近似RBF核特征映射。
    - SkewedChi2Sampler：偏斜卡方核的近似特征映射。
    - sklearn.metrics.pairwise.chi2_kernel：精确的卡方核。
    - sklearn.metrics.pairwise.kernel_metrics：内置核列表。

    References
    ----------
    See "Random Fourier Approximations for Skewed Multiplicative Histogram
    Kernels" by Fuxin Li, Catalin Ionescu and Cristian Sminchisescu.
    
    参考文献：
    参见 Fuxin Li、Catalin Ionescu 和 Cristian Sminchisescu 的《用于偏斜乘法直方图的随机傅里叶逼近》。

    Examples
    --------
    >>> from sklearn.kernel_approximation import SkewedChi2Sampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> chi2_feature = SkewedChi2Sampler(skewedness=.01,
    ...                                  n_components=10,
    ...                                  random_state=0)
    >>> X_features = chi2_feature.fit_transform(X, y)
    >>> clf = SGDClassifier(max_iter=10, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=10)
    >>> clf.score(X_features, y)
    1.0
    """
    # 定义参数约束字典，包括偏斜度、组件数和随机状态
    _parameter_constraints: dict = {
        "skewedness": [Interval(Real, None, None, closed="neither")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    # 初始化函数，设置默认参数偏斜度、组件数和随机状态
    def __init__(self, *, skewedness=1.0, n_components=100, random_state=None):
        self.skewedness = skewedness  # 设置偏斜度
        self.n_components = n_components  # 设置组件数
        self.random_state = random_state  # 设置随机状态

    # 使用装饰器 `_fit_context` 包装的拟合函数，支持跳过嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X.

        Samples random projection according to n_features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 验证输入数据 X
        X = self._validate_data(X)
        # 检查随机状态，确保其类型正确
        random_state = check_random_state(self.random_state)
        # 获取输入数据的特征数
        n_features = X.shape[1]
        # 从随机状态生成均匀分布的随机数矩阵
        uniform = random_state.uniform(size=(n_features, self.n_components))
        # 使用 sech 的反函数进行变换
        self.random_weights_ = 1.0 / np.pi * np.log(np.tan(np.pi / 2.0 * uniform))
        # 生成随机偏移量
        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)

        # 如果输入数据 X 的数据类型为 np.float32，则将拟合属性的数据类型设置为相同以确保 `transform` 输出一致
        if X.dtype == np.float32:
            self.random_weights_ = self.random_weights_.astype(X.dtype, copy=False)
            self.random_offset_ = self.random_offset_.astype(X.dtype, copy=False)

        # 设置输出特征数为组件数
        self._n_features_out = self.n_components
        return self  # 返回实例本身

    # 应用近似特征映射到输入数据 X 的转换函数
    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features. All values of X must be
            strictly greater than "-skewedness".

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        # 检查是否已拟合
        check_is_fitted(self)
        # 验证输入数据 X，并进行数据复制和类型重置
        X = self._validate_data(
            X, copy=True, dtype=[np.float64, np.float32], reset=False
        )
        # 如果 X 中有小于 -self.skewedness 的值，抛出异常
        if (X <= -self.skewedness).any():
            raise ValueError("X may not contain entries smaller than -skewedness.")

        # 将 X 的每个元素加上偏斜度
        X += self.skewedness
        # 对 X 中的每个元素取自然对数
        np.log(X, X)
        # 使用稀疏矩阵点积计算投影
        projection = safe_sparse_dot(X, self.random_weights_)
        # 加上随机偏移量
        projection += self.random_offset_
        # 对投影进行余弦变换
        np.cos(projection, projection)
        # 乘以常数以调整幅度
        projection *= np.sqrt(2.0) / np.sqrt(self.n_components)
        return projection  # 返回投影结果
    # 定义一个方法 `_more_tags(self)`
    def _more_tags(self):
        # 返回一个字典，该字典包含键 `preserves_dtype`，其值是包含 np.float64 和 np.float32 的列表
        return {"preserves_dtype": [np.float64, np.float32]}
class AdditiveChi2Sampler(TransformerMixin, BaseEstimator):
    """Approximate feature map for additive chi2 kernel.

    Uses sampling the fourier transform of the kernel characteristic
    at regular intervals.

    Since the kernel that is to be approximated is additive, the components of
    the input vectors can be treated separately.  Each entry in the original
    space is transformed into 2*sample_steps-1 features, where sample_steps is
    a parameter of the method. Typical values of sample_steps include 1, 2 and
    3.

    Optimal choices for the sampling interval for certain data ranges can be
    computed (see the reference). The default values should be reasonable.

    Read more in the :ref:`User Guide <additive_chi_kernel_approx>`.

    Parameters
    ----------
    sample_steps : int, default=2
        Gives the number of (complex) sampling points.

    sample_interval : float, default=None
        Sampling interval. Must be specified when sample_steps not in {1,2,3}.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SkewedChi2Sampler : A Fourier-approximation to a non-additive variant of
        the chi squared kernel.

    sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.

    sklearn.metrics.pairwise.additive_chi2_kernel : The exact additive chi
        squared kernel.

    Notes
    -----
    This estimator approximates a slightly different version of the additive
    chi squared kernel then ``metric.additive_chi2`` computes.

    This estimator is stateless and does not need to be fitted. However, we
    recommend to call :meth:`fit_transform` instead of :meth:`transform`, as
    parameter validation is only performed in :meth:`fit`.

    References
    ----------
    See `"Efficient additive kernels via explicit feature maps"
    <http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf>`_
    A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence,
    2011

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.kernel_approximation import AdditiveChi2Sampler
    >>> X, y = load_digits(return_X_y=True)
    >>> chi2sampler = AdditiveChi2Sampler(sample_steps=2)
    >>> X_transformed = chi2sampler.fit_transform(X, y)
    >>> clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)
    >>> clf.fit(X_transformed, y)
    SGDClassifier(max_iter=5, random_state=0)
    >>> clf.score(X_transformed, y)
    0.9499...
    """


注释：

# AdditiveChi2Sampler 类定义，用于近似加法 chi2 核的特征映射
# 继承自 TransformerMixin 和 BaseEstimator
class AdditiveChi2Sampler(TransformerMixin, BaseEstimator):
    """Approximate feature map for additive chi2 kernel.

    Uses sampling the fourier transform of the kernel characteristic
    at regular intervals.

    Since the kernel that is to be approximated is additive, the components of
    the input vectors can be treated separately.  Each entry in the original
    space is transformed into 2*sample_steps-1 features, where sample_steps is
    a parameter of the method. Typical values of sample_steps include 1, 2 and
    3.

    Optimal choices for the sampling interval for certain data ranges can be
    computed (see the reference). The default values should be reasonable.

    Read more in the :ref:`User Guide <additive_chi_kernel_approx>`.

    Parameters
    ----------
    sample_steps : int, default=2
        Gives the number of (complex) sampling points.

    sample_interval : float, default=None
        Sampling interval. Must be specified when sample_steps not in {1,2,3}.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SkewedChi2Sampler : A Fourier-approximation to a non-additive variant of
        the chi squared kernel.

    sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.

    sklearn.metrics.pairwise.additive_chi2_kernel : The exact additive chi
        squared kernel.

    Notes
    -----
    This estimator approximates a slightly different version of the additive
    chi squared kernel then ``metric.additive_chi2`` computes.

    This estimator is stateless and does not need to be fitted. However, we
    recommend to call :meth:`fit_transform` instead of :meth:`transform`, as
    parameter validation is only performed in :meth:`fit`.

    References
    ----------
    See `"Efficient additive kernels via explicit feature maps"
    <http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf>`_
    A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence,
    2011

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.kernel_approximation import AdditiveChi2Sampler
    >>> X, y = load_digits(return_X_y=True)
    >>> chi2sampler = AdditiveChi2Sampler(sample_steps=2)
    >>> X_transformed = chi2sampler.fit_transform(X, y)
    >>> clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)
    >>> clf.fit(X_transformed, y)
    SGDClassifier(max_iter=5, random_state=0)
    >>> clf.score(X_transformed, y)
    0.9499...
    """
    # 定义参数约束字典，限定了两个参数的取值范围和类型
    _parameter_constraints: dict = {
        "sample_steps": [Interval(Integral, 1, None, closed="left")],
        "sample_interval": [Interval(Real, 0, None, closed="left"), None],
    }

    # 初始化方法，设置对象的样本步数和样本间隔参数
    def __init__(self, *, sample_steps=2, sample_interval=None):
        self.sample_steps = sample_steps  # 设置样本步数
        self.sample_interval = sample_interval  # 设置样本间隔

    # 使用装饰器 `_fit_context` 标记的 `fit` 方法，用于拟合模型
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """只验证估计器的参数。

        此方法允许：(i) 验证估计器的参数，和
                   (ii) 与 scikit-learn 转换器 API 保持一致。

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            训练数据，其中 `n_samples` 是样本数量，`n_features` 是特征数量。

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            目标值（用于无监督转换时为 None）。

        Returns
        -------
        self : object
            返回转换器对象。
        """
        # 验证输入数据 X，并将其稀疏表示转换为 CSR 格式
        X = self._validate_data(X, accept_sparse="csr")
        # 检查 X 是否非负，用于检验过程中的辅助信息
        check_non_negative(X, "X in AdditiveChi2Sampler.fit")

        # 如果 `sample_interval` 为 None，且 `sample_steps` 不在 (1, 2, 3) 中，
        # 则抛出值错误异常
        if self.sample_interval is None and self.sample_steps not in (1, 2, 3):
            raise ValueError(
                "If sample_steps is not in [1, 2, 3],"
                " you need to provide sample_interval"
            )

        # 返回对象自身
        return self
    def transform(self, X):
        """Apply approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, \
               shape = (n_samples, n_features * (2*sample_steps - 1))
            Whether the return value is an array or sparse matrix depends on
            the type of the input X.
        """
        # Validate and prepare the input data X
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        # Check if X contains non-negative values
        check_non_negative(X, "X in AdditiveChi2Sampler.transform")
        # Determine if X is sparse
        sparse = sp.issparse(X)

        if self.sample_interval is None:
            # Set sample_interval based on the number of sample_steps
            # Reference: "Efficient additive kernels via explicit feature maps"
            # A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence, 2011
            if self.sample_steps == 1:
                sample_interval = 0.8
            elif self.sample_steps == 2:
                sample_interval = 0.5
            elif self.sample_steps == 3:
                sample_interval = 0.4
            else:
                # Raise an error if sample_steps is outside [1, 2, 3]
                raise ValueError(
                    "If sample_steps is not in [1, 2, 3],"
                    " you need to provide sample_interval"
                )
        else:
            # Use the provided sample_interval if specified
            sample_interval = self.sample_interval

        # Define the transformation function based on the sparsity of X
        transf = self._transform_sparse if sparse else self._transform_dense
        # Apply the transformation to X and return the result
        return transf(X, self.sample_steps, sample_interval)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Only used to validate feature names with the names seen in :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # Ensure that the model has been fitted
        check_is_fitted(self, "n_features_in_")
        # Validate and generate output feature names based on input_features
        input_features = _check_feature_names_in(
            self, input_features, generate_names=True
        )
        # Determine the name of the estimator class in lowercase
        est_name = self.__class__.__name__.lower()

        # Generate names for the zeroth component transformation
        names_list = [f"{est_name}_{name}_sqrt" for name in input_features]

        # Generate names for cosine and sine transformations for each sample step
        for j in range(1, self.sample_steps):
            cos_names = [f"{est_name}_{name}_cos{j}" for name in input_features]
            sin_names = [f"{est_name}_{name}_sin{j}" for name in input_features]
            names_list.extend(cos_names + sin_names)

        # Return the list of generated feature names as a numpy array
        return np.asarray(names_list, dtype=object)
    # 定义一个静态方法，用于将稠密矩阵 X 转换为多个步骤的变换结果
    def _transform_dense(X, sample_steps, sample_interval):
        # 找出 X 中非零元素的位置
        non_zero = X != 0.0
        # 提取出非零元素的值
        X_nz = X[non_zero]

        # 创建一个与 X 同样大小的全零数组 X_step
        X_step = np.zeros_like(X)
        # 对非零元素位置上的 X_step 进行赋值，计算采样间隔的平方根乘以非零元素值的结果
        X_step[non_zero] = np.sqrt(X_nz * sample_interval)

        # 初始化一个列表 X_new，将第一个变换结果 X_step 加入其中
        X_new = [X_step]

        # 计算非零元素值的对数乘以采样间隔的结果
        log_step_nz = sample_interval * np.log(X_nz)
        # 计算两倍非零元素值乘以采样间隔的结果
        step_nz = 2 * X_nz * sample_interval

        # 循环进行多步变换，从 j = 1 开始到 sample_steps 结束
        for j in range(1, sample_steps):
            # 计算变换系数 factor_nz
            factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * sample_interval))

            # 重新创建一个与 X 同样大小的全零数组 X_step
            X_step = np.zeros_like(X)
            # 对非零元素位置上的 X_step 进行赋值，计算变换公式的余弦部分
            X_step[non_zero] = factor_nz * np.cos(j * log_step_nz)
            # 将变换后的 X_step 加入 X_new 列表
            X_new.append(X_step)

            # 重新创建一个与 X 同样大小的全零数组 X_step
            X_step = np.zeros_like(X)
            # 对非零元素位置上的 X_step 进行赋值，计算变换公式的正弦部分
            X_step[non_zero] = factor_nz * np.sin(j * log_step_nz)
            # 将变换后的 X_step 加入 X_new 列表
            X_new.append(X_step)

        # 将 X_new 列表中所有变换结果沿水平方向堆叠起来，形成最终的变换后的稠密矩阵
        return np.hstack(X_new)

    # 定义一个静态方法，用于将稀疏矩阵 X 转换为多个步骤的变换结果
    @staticmethod
    def _transform_sparse(X, sample_steps, sample_interval):
        # 复制稀疏矩阵 X 的 indices 和 indptr 属性
        indices = X.indices.copy()
        indptr = X.indptr.copy()

        # 计算每个非零元素值乘以采样间隔的平方根
        data_step = np.sqrt(X.data * sample_interval)
        # 创建一个新的稀疏矩阵 X_step，用计算出的 data_step、indices 和 indptr 构建
        X_step = sp.csr_matrix(
            (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
        )
        # 初始化一个列表 X_new，将第一个变换结果 X_step 加入其中
        X_new = [X_step]

        # 计算每个非零元素值乘以采样间隔的对数
        log_step_nz = sample_interval * np.log(X.data)
        # 计算两倍非零元素值乘以采样间隔的结果
        step_nz = 2 * X.data * sample_interval

        # 循环进行多步变换，从 j = 1 开始到 sample_steps 结束
        for j in range(1, sample_steps):
            # 计算变换系数 factor_nz
            factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * sample_interval))

            # 计算余弦变换后的 data_step
            data_step = factor_nz * np.cos(j * log_step_nz)
            # 创建一个新的稀疏矩阵 X_step，用计算出的 data_step、indices 和 indptr 构建
            X_step = sp.csr_matrix(
                (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
            )
            # 将变换后的 X_step 加入 X_new 列表
            X_new.append(X_step)

            # 计算正弦变换后的 data_step
            data_step = factor_nz * np.sin(j * log_step_nz)
            # 创建一个新的稀疏矩阵 X_step，用计算出的 data_step、indices 和 indptr 构建
            X_step = sp.csr_matrix(
                (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
            )
            # 将变换后的 X_step 加入 X_new 列表
            X_new.append(X_step)

        # 将 X_new 列表中所有变换结果沿水平方向堆叠起来，形成最终的变换后的稀疏矩阵
        return sp.hstack(X_new)

    # 定义一个方法，返回一个字典，标记该方法的特性
    def _more_tags(self):
        return {"stateless": True, "requires_positive_X": True}
class Nystroem(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Approximate a kernel map using a subset of the training data.

    Constructs an approximate feature map for an arbitrary kernel
    using a subset of the data as basis.

    Read more in the :ref:`User Guide <nystroem_kernel_approx>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as `kernel_params`, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    coef0 : float, default=None
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    degree : float, default=None
        Degree of the polynomial kernel. Ignored by other kernels.

    kernel_params : dict, default=None
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    n_components : int, default=100
        Number of features to construct.
        How many data points will be used to construct the mapping.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the uniform sampling without
        replacement of `n_components` of the training data to construct the
        basis kernel.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking
        down the kernel matrix into `n_jobs` even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.24

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Subset of training points used to construct the feature map.

    component_indices_ : ndarray of shape (n_components)
        Indices of ``components_`` in the training set.

    normalization_ : ndarray of shape (n_components, n_components)
        Normalization matrix needed for embedding.
        Square root of the kernel matrix on ``components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    """

    def __init__(self, kernel='rbf', gamma=None, coef0=None, degree=None,
                 kernel_params=None, n_components=100, random_state=None, n_jobs=None):
        """Initialize Nystroem approximation with specified parameters.

        Parameters
        ----------
        kernel : str or callable, default='rbf'
            Specifies the kernel function to approximate.

        gamma : float, default=None
            Gamma parameter for certain kernels like RBF and sigmoid.

        coef0 : float, default=None
            Coefficient for polynomial and sigmoid kernels.

        degree : float, default=None
            Degree of the polynomial kernel.

        kernel_params : dict, default=None
            Additional parameters for the kernel function.

        n_components : int, default=100
            Number of components to construct the approximation.

        random_state : int, RandomState instance or None, default=None
            Random number generator for sampling training data.

        n_jobs : int, default=None
            Number of parallel jobs to use for computation.
        """
        # Call superclass constructors with mixin classes
        super().__init__()
        # Initialize kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        # Number of components for approximation
        self.n_components = n_components
        # Random state for reproducibility
        self.random_state = random_state
        # Number of jobs for parallel computation
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the Nystroem approximation model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : None
            Ignored variable.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate and set the number of training samples and features
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        # Initialize random state for sampling
        rng = check_random_state(self.random_state)
        # Sample indices for components
        n_samples = X.shape[0]
        if self.n_components > n_samples:
            raise ValueError("Number of components must be less than or equal "
                             "to number of samples")
        # Randomly select component indices
        permutation = rng.permutation(n_samples)
        indices = permutation[:self.n_components]
        self.component_indices_ = indices
        # Select components from data
        self.components_ = X[indices]
        # Compute kernel matrix on components
        K = pairwise_kernels(self.components_, metric=self.kernel,
                             filter_params=True, **self.kernel_params)
        # Compute normalization matrix
        self.normalization_ = np.linalg.cholesky(K + 1e-12 * np.eye(self.n_components))
        return self
    # _parameter_constraints 字典，定义了每个参数的约束条件
    _parameter_constraints: dict = {
        "kernel": [
            StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS.keys()) | {"precomputed"}),
            callable,
        ],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "coef0": [Interval(Real, None, None, closed="neither"), None],
        "degree": [Interval(Real, 1, None, closed="left"), None],
        "kernel_params": [dict, None],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "n_jobs": [Integral, None],
    }

    # 初始化方法，设置各个参数的初始值
    def __init__(
        self,
        kernel="rbf",  # 默认核函数为 RBF 核
        *,
        gamma=None,  # gamma 参数，默认为 None
        coef0=None,  # coef0 参数，默认为 None
        degree=None,  # degree 参数，默认为 None
        kernel_params=None,  # kernel_params 参数，默认为 None
        n_components=100,  # n_components 参数，默认为 100
        random_state=None,  # random_state 参数，默认为 None
        n_jobs=None,  # n_jobs 参数，默认为 None
    ):
        self.kernel = kernel  # 设置对象的 kernel 属性
        self.gamma = gamma  # 设置对象的 gamma 属性
        self.coef0 = coef0  # 设置对象的 coef0 属性
        self.degree = degree  # 设置对象的 degree 属性
        self.kernel_params = kernel_params  # 设置对象的 kernel_params 属性
        self.n_components = n_components  # 设置对象的 n_components 属性
        self.random_state = random_state  # 设置对象的 random_state 属性
        self.n_jobs = n_jobs  # 设置对象的 n_jobs 属性

    # 使用装饰器 _fit_context 进行修饰，设定 prefer_skip_nested_validation 参数为 True
    @_fit_context(prefer_skip_nested_validation=True)
    # 将估计器适应于数据。

    # 从训练点的子集中进行采样，计算这些点之间的核，计算归一化矩阵。

    # 参数
    # ------
    # X : 类数组，形状为 (n_samples, n_features)
    #     训练数据，其中 `n_samples` 是样本数量，`n_features` 是特征数量。
    # y : 类数组，形状为 (n_samples,) 或 (n_samples, n_outputs)，默认为 None
    #     目标值（无监督转换时为 None）。

    # 返回
    # -------
    # self : 对象
    #     返回实例本身。
    def fit(self, X, y=None):
        # 验证数据 X，并接受稀疏矩阵格式
        X = self._validate_data(X, accept_sparse="csr")
        # 检查随机状态
        rnd = check_random_state(self.random_state)
        # 获取样本数量
        n_samples = X.shape[0]

        # 获取基础向量
        if self.n_components > n_samples:
            # 如果指定的主成分数大于样本数，警告并设置为样本数
            n_components = n_samples
            warnings.warn(
                "n_components > n_samples. This is not possible.\n"
                "n_components was set to n_samples, which results"
                " in inefficient evaluation of the full kernel."
            )
        else:
            # 否则设置为指定的主成分数
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        # 对样本进行随机排列
        inds = rnd.permutation(n_samples)
        # 选择基础向量的索引
        basis_inds = inds[:n_components]
        # 获取基础向量
        basis = X[basis_inds]

        # 计算基础向量之间的核矩阵
        basis_kernel = pairwise_kernels(
            basis,
            metric=self.kernel,
            filter_params=True,
            n_jobs=self.n_jobs,
            **self._get_kernel_params(),
        )

        # 对基础向量的核矩阵进行奇异值分解（SVD）
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        # 计算归一化矩阵
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        # 存储基础向量
        self.components_ = basis
        # 存储基础向量的索引
        self.component_indices_ = basis_inds
        # 记录输出特征的数量
        self._n_features_out = n_components
        # 返回实例本身
        return self

    # 将特征映射应用于 X。

    # 使用某些训练点与 X 之间的核来计算近似特征映射。

    # 参数
    # ------
    # X : 类数组，形状为 (n_samples, n_features)
    #     要转换的数据。

    # 返回
    # -------
    # X_transformed : 形状为 (n_samples, n_components) 的 ndarray
    #     转换后的数据。
    def transform(self, X):
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 验证数据 X，并接受稀疏矩阵格式，但不重置
        X = self._validate_data(X, accept_sparse="csr", reset=False)

        # 获取核函数的参数
        kernel_params = self._get_kernel_params()
        # 计算 X 和存储的基础向量之间的核矩阵
        embedded = pairwise_kernels(
            X,
            self.components_,
            metric=self.kernel,
            filter_params=True,
            n_jobs=self.n_jobs,
            **kernel_params,
        )
        # 返回嵌入数据与归一化矩阵转置的乘积
        return np.dot(embedded, self.normalization_.T)
    # 获取核函数参数的方法
    def _get_kernel_params(self):
        # 将类属性中的核函数参数赋值给局部变量 params
        params = self.kernel_params
        # 如果核函数参数为 None，则将其设为一个空字典
        if params is None:
            params = {}
        
        # 如果核函数不是可调用对象且不是"precomputed"字符串
        if not callable(self.kernel) and self.kernel != "precomputed":
            # 遍历与当前核函数相关的参数列表
            for param in KERNEL_PARAMS[self.kernel]:
                # 如果当前实例对象中对应的属性不为 None，则将其添加到 params 字典中
                if getattr(self, param) is not None:
                    params[param] = getattr(self, param)
        else:
            # 如果核函数是可调用对象或者是"precomputed"字符串，则不应传递 gamma、coef0 或 degree 参数
            if (
                self.gamma is not None
                or self.coef0 is not None
                or self.degree is not None
            ):
                # 抛出 ValueError 异常，提示不要传递 gamma、coef0 或 degree 参数给 Nystroem
                raise ValueError(
                    "Don't pass gamma, coef0 or degree to "
                    "Nystroem if using a callable "
                    "or precomputed kernel"
                )

        # 返回处理后的参数字典
        return params

    # 返回额外的标签信息
    def _more_tags(self):
        return {
            # 标记一些测试用例预期会失败的情况
            "_xfail_checks": {
                "check_transformer_preserve_dtypes": (
                    "dtypes are preserved but not at a close enough precision"
                )
            },
            # 标记该对象保留的数据类型
            "preserves_dtype": [np.float64, np.float32],
        }
```