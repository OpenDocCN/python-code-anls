# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_pca.py`

```
"""Principal Component Analysis."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from math import log, sqrt                              # 导入数学函数log和sqrt
from numbers import Integral, Real                       # 导入Integral和Real类

import numpy as np                                       # 导入NumPy库
from scipy import linalg                                 # 导入SciPy linalg模块
from scipy.sparse import issparse                        # 导入SciPy issparse函数
from scipy.sparse.linalg import svds                     # 导入SciPy svds函数
from scipy.special import gammaln                        # 导入SciPy gammaln函数

from ..base import _fit_context                          # 导入内部模块_fit_context
from ..utils import check_random_state                   # 导入内部工具函数check_random_state
from ..utils._arpack import _init_arpack_v0              # 导入内部Arpack初始化函数_init_arpack_v0
from ..utils._array_api import _convert_to_numpy, get_namespace  # 导入内部Array API转换函数_convert_to_numpy和命名空间获取函数get_namespace
from ..utils._param_validation import Interval, RealNotInt, StrOptions  # 导入内部参数验证函数Interval, RealNotInt, StrOptions
from ..utils.extmath import fast_logdet, randomized_svd, stable_cumsum, svd_flip  # 导入内部数学扩展函数fast_logdet, randomized_svd, stable_cumsum, svd_flip
from ..utils.sparsefuncs import _implicit_column_offset, mean_variance_axis  # 导入内部稀疏函数_implicit_column_offset, mean_variance_axis
from ..utils.validation import check_is_fitted           # 导入内部验证函数check_is_fitted
from ._base import _BasePCA                              # 导入PCA基类_BasePCA


def _assess_dimension(spectrum, rank, n_samples):
    """Compute the log-likelihood of a rank ``rank`` dataset.

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``. This implements the method of
    T. P. Minka.

    Parameters
    ----------
    spectrum : ndarray of shape (n_features,)
        Data spectrum.
    rank : int
        Tested rank value. It should be strictly lower than n_features,
        otherwise the method isn't specified (division by zero in equation
        (31) from the paper).
    n_samples : int
        Number of samples.

    Returns
    -------
    ll : float
        The log-likelihood.

    References
    ----------
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
    <https://proceedings.neurips.cc/paper/2000/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf>`_
    """
    xp, _ = get_namespace(spectrum)                       # 获取数组API和命名空间

    n_features = spectrum.shape[0]                       # 获取频谱数组的维度数
    if not 1 <= rank < n_features:
        raise ValueError("the tested rank should be in [1, n_features - 1]")

    eps = 1e-15                                          # 定义小量eps

    if spectrum[rank - 1] < eps:
        # 当测试的秩与一个小的特征值相关联时，计算对数似然没有意义：它将非常小，并且不会是最大值。
        # 此外，它可能会导致计算pa时出现数值问题，特别是在log((spectrum[i] - spectrum[j])中，因为这将取小数的对数。
        return -xp.inf                                   # 返回负无穷大

    pu = -rank * log(2.0)                                # 初始化pu
    for i in range(1, rank + 1):
        pu += (
            gammaln((n_features - i + 1) / 2.0)           # 计算对数Gamma函数
            - log(xp.pi) * (n_features - i + 1) / 2.0    # 计算对数π
        )

    pl = xp.sum(xp.log(spectrum[:rank]))                 # 计算前rank个特征值的对数和
    pl = -pl * n_samples / 2.0                           # 根据样本数调整

    v = max(eps, xp.sum(spectrum[rank:]) / (n_features - rank))  # 计算后面特征值的平均值
    pv = -log(v) * n_samples * (n_features - rank) / 2.0   # 计算pv

    m = n_features * rank - rank * (rank + 1.0) / 2.0     # 计算m
    pp = log(2.0 * xp.pi) * (m + rank) / 2.0              # 计算pp

    pa = 0.0                                              # 初始化pa
    spectrum_ = xp.asarray(spectrum, copy=True)           # 复制特征值数组
    spectrum_[rank:n_features] = v                         # 更新数组的部分特征值为v
    # 遍历排名范围内的每个索引 i
    for i in range(rank):
        # 对于每个 i，遍历比其索引大的所有索引 j
        for j in range(i + 1, spectrum.shape[0]):
            # 计算对数值，包括差异和逆频率的加权
            pa += log(
                (spectrum[i] - spectrum[j]) * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])
            ) + log(n_samples)
    
    # 计算最终的对数似然值 ll，包括前景、背景和惩罚项，以及正则化项
    ll = pu + pl + pv + pp - pa / 2.0 - rank * log(n_samples) / 2.0
    
    # 返回计算得到的对数似然值
    return ll
def _infer_dimension(spectrum, n_samples):
    """Infers the dimension of a dataset with a given spectrum.

    The returned value will be in [1, n_features - 1].
    """
    xp, _ = get_namespace(spectrum)  # 获取 spectrum 的命名空间

    ll = xp.empty_like(spectrum)  # 创建一个与 spectrum 相同大小的空数组 ll
    ll[0] = -xp.inf  # 将 ll 的第一个元素设为负无穷，避免返回 n_components = 0
    for rank in range(1, spectrum.shape[0]):  # 循环计算不同的 rank
        ll[rank] = _assess_dimension(spectrum, rank, n_samples)  # 计算给定 rank 下的维度估计值
    return xp.argmax(ll)  # 返回 ll 中最大值的索引，即估计的最佳维度


class PCA(_BasePCA):
    """Principal component analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    It uses the LAPACK implementation of the full SVD or a randomized truncated
    SVD by the method of Halko et al. 2009, depending on the shape of the input
    data and the number of components to extract.

    With sparse inputs, the ARPACK implementation of the truncated SVD can be
    used (i.e. through :func:`scipy.sparse.linalg.svds`). Alternatively, one
    may consider :class:`TruncatedSVD` where the data are not centered.

    Notice that this class only supports sparse inputs for some solvers such as
    "arpack" and "covariance_eigh". See :class:`TruncatedSVD` for an
    alternative with sparse data.

    For a usage example, see
    :ref:`sphx_glr_auto_examples_decomposition_plot_pca_iris.py`

    Read more in the :ref:`User Guide <PCA>`.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.
    """
    whiten : bool, default=False
        当为 True 时（默认为 False），`components_` 向量将乘以 n_samples 的平方根，
        然后除以奇异值，以确保输出为无相关且单位分量方差。

        白化将从转换后的信号中移除一些信息（组件的相对方差比例），但有时可以通过使其数据
        符合某些内置假设来提高下游估计器的预测精度。

    svd_solver : {'auto', 'full', 'covariance_eigh', 'arpack', 'randomized'},\
            default='auto'
        "auto" :
            根据 `X.shape` 和 `n_components` 的默认 'auto' 策略选择求解器：
            如果输入数据的特征少于 1000 且样本数超过特征数的10倍，则使用 "covariance_eigh" 求解器。
            否则，如果输入数据大于 500x500 并且要提取的组件数量小于数据最小维度的80%，则选择更高效的
            "randomized" 方法。否则，计算精确的 "full" SVD 并选择是否截断。
        "full" :
            使用标准 LAPACK 求解器通过 `scipy.linalg.svd` 运行精确的完全 SVD，并通过后处理选择组件。
        "covariance_eigh" :
            预先计算协方差矩阵（在中心化数据上），在协方差矩阵上运行经典特征值分解，通常使用 LAPACK，
            并通过后处理选择组件。对于 n_samples >> n_features 和小 n_features，此求解器非常高效。
            然而，对于大 n_features，它不可行（需要大量内存来实现协方差矩阵）。还要注意，与 "full" 求解器
            相比，此求解器实际上会增加条件数，因此数值稳定性较差（例如，在具有大量奇异值范围的输入数据上）。
        "arpack" :
            通过 `scipy.sparse.linalg.svds` 调用 ARPACK 求解器运行截断的 SVD，要求严格
            `0 < n_components < min(X.shape)`
        "randomized" :
            通过 Halko 等人的方法运行随机化 SVD。

        .. versionadded:: 0.18.0

        .. versionchanged:: 1.5
            添加了 'covariance_eigh' 求解器。

    tol : float, default=0.0
        当 svd_solver == 'arpack' 时计算奇异值的容差。
        必须在 [0.0, 无穷) 范围内。

        .. versionadded:: 0.18.0

    iterated_power : int or 'auto', default='auto'
        由 svd_solver == 'randomized' 计算的幂法迭代次数。
        必须在 [0, 无穷) 范围内。

        .. versionadded:: 0.18.0
    n_oversamples : int, default=10
        这个参数仅在 `svd_solver="randomized"` 时有效。
        它表示额外随机向量的数量，用于采样 `X` 的范围，以确保适当的条件性。
        更多细节请参见 :func:`~sklearn.utils.extmath.randomized_svd`。

        .. versionadded:: 1.1

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        用于随机化 SVD 求解器的幂迭代正则化器。
        ARPACK 求解器不使用此参数。更多细节请参见 :func:`~sklearn.utils.extmath.randomized_svd`。

        .. versionadded:: 1.1

    random_state : int, RandomState instance or None, default=None
        当使用 'arpack' 或 'randomized' 求解器时使用此参数。
        传递一个整数以在多次函数调用间获得可重现的结果。
        详见 :term:`Glossary <random_state>`。

        .. versionadded:: 0.18.0

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        特征空间中的主轴，表示数据中方差的最大方向。
        相当于中心化输入数据的右奇异向量，与其特征向量平行。
        组件按照 `explained_variance_` 的降序排列。

    explained_variance_ : ndarray of shape (n_components,)
        每个选定组件解释的方差量。
        方差估计使用 `n_samples - 1` 自由度。

        等于 `X` 协方差矩阵的 `n_components` 最大特征值。

        .. versionadded:: 0.18

    explained_variance_ratio_ : ndarray of shape (n_components,)
        每个选定组件解释的方差百分比。

        如果未设置 `n_components`，则存储所有组件，并且比率的总和等于 1.0。

    singular_values_ : ndarray of shape (n_components,)
        每个选定组件对应的奇异值。
        奇异值等于低维空间中 `n_components` 变量的二范数。

        .. versionadded:: 0.19

    mean_ : ndarray of shape (n_features,)
        从训练集估计的每个特征的经验均值。

        等于 `X.mean(axis=0)`。

    n_components_ : int
        估计的组件数量。当 `n_components` 设置为 'mle' 或介于 0 和 1 之间的数字（当 `svd_solver == 'full'` 时）时，
        此数字是从输入数据估算而来。否则，它等于参数 `n_components`，或者是 `n_features` 和 `n_samples` 的较小值，
        如果 `n_components` 为 None。

    n_samples_ : int
        训练数据中的样本数。
    # 估计的噪声协方差，根据 Tipping 和 Bishop 1999 年提出的概率PCA模型计算得出
    # 参见 "Pattern Recognition and Machine Learning" by C. Bishop, 12.2.1 p. 574 或者
    # http://www.miketipping.com/papers/met-mppca.pdf
    # 用于计算估计的数据协方差并对样本进行评分
    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        compute the estimated data covariance and score samples.

    # 在拟合过程中看到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在拟合过程中看到的特征的名称数组
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 相关链接和引用
    See Also
    --------
    KernelPCA : Kernel Principal Component Analysis.
    SparsePCA : Sparse Principal Component Analysis.
    TruncatedSVD : Dimensionality reduction using truncated SVD.
    IncrementalPCA : Incremental Principal Component Analysis.

    # 关于参数 n_components == 'mle' 的使用方法
    References
    ----------
    For n_components == 'mle', this class uses the method from:
    `Minka, T. P.. "Automatic choice of dimensionality for PCA".
    In NIPS, pp. 598-604 <https://tminka.github.io/papers/pca/minka-pca.pdf>`_

    # 实现了概率PCA模型，参考文献为 Tipping 和 Bishop 1999 年的文章
    Implements the probabilistic PCA model from:
    `Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal
    component analysis". Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 61(3), 611-622.
    <http://www.miketipping.com/papers/met-mppca.pdf>`_
    via the score and score_samples methods.

    # 当 svd_solver == 'arpack' 时，参考 scipy.sparse.linalg.svds
    For svd_solver == 'arpack', refer to `scipy.sparse.linalg.svds`.

    # 当 svd_solver == 'randomized' 时，参考 Halko et al. (2011) 和 Martinsson et al. (2011) 的文献
    For svd_solver == 'randomized', see:
    :doi:`Halko, N., Martinsson, P. G., and Tropp, J. A. (2011).
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions".
    SIAM review, 53(2), 217-288.
    <10.1137/090771806>`
    and also
    :doi:`Martinsson, P. G., Rokhlin, V., and Tygert, M. (2011).
    "A randomized algorithm for the decomposition of matrices".
    Applied and Computational Harmonic Analysis, 30(1), 47-68.
    <10.1016/j.acha.2010.02.003>`

    # 示例代码
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(n_components=2)
    >>> print(pca.explained_variance_ratio_)
    [0.9924... 0.0075...]
    >>> print(pca.singular_values_)
    [6.30061... 0.54980...]

    >>> pca = PCA(n_components=2, svd_solver='full')
    >>> pca.fit(X)
    PCA(n_components=2, svd_solver='full')
    >>> print(pca.explained_variance_ratio_)
    [0.9924... 0.00755...]
    >>> print(pca.singular_values_)
    [6.30061... 0.54980...]

    >>> pca = PCA(n_components=1, svd_solver='arpack')
    >>> pca.fit(X)
    # 创建一个主成分分析（PCA）对象，设置主成分数量为1，使用ARPACK方法求解奇异值分解
    PCA(n_components=1, svd_solver='arpack')
    # 打印出PCA模型中解释方差的比例
    >>> print(pca.explained_variance_ratio_)
    [0.99244...]
    # 打印出PCA模型中奇异值
    >>> print(pca.singular_values_)
    [6.30061...]
    """

    # 定义参数约束字典，描述了主成分分析（PCA）类的各个参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_components": [
            # 主成分数量的约束条件，必须是大于等于0的整数
            Interval(Integral, 0, None, closed="left"),
            # 主成分数量的约束条件，必须是介于0到1之间的实数但不能包括0和1
            Interval(RealNotInt, 0, 1, closed="neither"),
            # 主成分数量的约束条件，特定取值为字符串"mle"
            StrOptions({"mle"}),
            # 主成分数量的约束条件，无特定要求，即可以为任意值
            None,
        ],
        "copy": ["boolean"],  # 是否复制数据的约束条件，必须为布尔值
        "whiten": ["boolean"],  # 是否进行白化处理的约束条件，必须为布尔值
        "svd_solver": [
            # 奇异值分解求解器的约束条件，可选取值为{"auto", "full", "covariance_eigh", "arpack", "randomized"}中的一个
            StrOptions({"auto", "full", "covariance_eigh", "arpack", "randomized"})
        ],
        "tol": [
            # 数值容差的约束条件，必须为大于等于0的实数
            Interval(Real, 0, None, closed="left")
        ],
        "iterated_power": [
            # 迭代幂次的约束条件，特定取值为字符串"auto"，或者必须是大于等于0的整数
            StrOptions({"auto"}),
            Interval(Integral, 0, None, closed="left"),
        ],
        "n_oversamples": [
            # 过采样数量的约束条件，必须是大于等于1的整数
            Interval(Integral, 1, None, closed="left")
        ],
        "power_iteration_normalizer": [
            # 幂迭代正则化方法的约束条件，可选取值为{"auto", "QR", "LU", "none"}中的一个
            StrOptions({"auto", "QR", "LU", "none"})
        ],
        "random_state": ["random_state"],  # 随机数生成器的约束条件，必须是随机数生成器对象
    }

    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        # 初始化主成分分析（PCA）对象的各个参数
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 调用内部的_fit方法进行模型的拟合
        self._fit(X)
        # 返回拟合后的实例本身
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.

        Notes
        -----
        This method returns a Fortran-ordered array. To convert it to a
        C-ordered array, use 'np.ascontiguousarray'.
        """
        # Perform the fit operation, obtaining necessary components
        U, S, _, X, x_is_centered, xp = self._fit(X)
        
        # Check if U is not None (i.e., U is computed)
        if U is not None:
            # Truncate U to retain only the first `n_components_` components
            U = U[:, : self.n_components_]

            # Scale U based on the type of transformation (whitening or not)
            if self.whiten:
                # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
                U *= sqrt(X.shape[0] - 1)
            else:
                # X_new = X * V = U * S * Vt * V = U * S
                U *= S[: self.n_components_]

            # Return the transformed matrix U
            return U
        else:  # solver="covariance_eigh" does not compute U at fit time.
            # Perform transformation without fitting
            return self._transform(X, xp, x_is_centered=x_is_centered)
    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""
        xp, is_array_api_compliant = get_namespace(X)

        # 如果输入是稀疏矩阵并且 svd_solver 不是支持的选项之一，则抛出错误
        if issparse(X) and self.svd_solver not in ["auto", "arpack", "covariance_eigh"]:
            raise TypeError(
                'PCA only support sparse inputs with the "arpack" and'
                f' "covariance_eigh" solvers, while "{self.svd_solver}" was passed. See'
                " TruncatedSVD for a possible alternative."
            )
        # 如果 svd_solver 是 'arpack' 并且输入数组符合数组 API，则抛出值错误
        if self.svd_solver == "arpack" and is_array_api_compliant:
            raise ValueError(
                "PCA with svd_solver='arpack' is not supported for Array API inputs."
            )

        # 对数据进行验证，不强制复制，因为支持稀疏输入数据和 'covariance_eigh' 求解器的任何求解器
        # 都被设计为避免对输入数据进行任何原地修改，与其他求解器相反。
        # 复制将在需要时（求解器协商完成后）进行。
        X = self._validate_data(
            X,
            dtype=[xp.float64, xp.float32],
            force_writeable=True,
            accept_sparse=("csr", "csc"),
            ensure_2d=True,
            copy=False,
        )
        self._fit_svd_solver = self.svd_solver

        # 如果 _fit_svd_solver 是 'auto' 并且输入是稀疏矩阵，则设置为 'arpack'
        if self._fit_svd_solver == "auto" and issparse(X):
            self._fit_svd_solver = "arpack"

        # 如果未指定主成分数量，则根据求解器确定默认值
        if self.n_components is None:
            if self._fit_svd_solver != "arpack":
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        # 如果 _fit_svd_solver 是 'auto'，根据问题的大小和 n_components 的值选择最合适的求解器
        if self._fit_svd_solver == "auto":
            # 高而瘦的问题最适合通过预计算协方差矩阵来处理
            if X.shape[1] <= 1000 and X.shape[0] >= 10 * X.shape[1]:
                self._fit_svd_solver = "covariance_eigh"
            # 小问题或 n_components == 'mle'，直接调用全 PCA
            elif max(X.shape) <= 500 or n_components == "mle":
                self._fit_svd_solver = "full"
            # 对于 n_components 在 (0, 1) 之间的情况
            elif 1 <= n_components < 0.8 * min(X.shape):
                self._fit_svd_solver = "randomized"
            else:
                self._fit_svd_solver = "full"

        # 根据选择的求解器调用不同的拟合方法，完整或截断的 SVD
        if self._fit_svd_solver in ("full", "covariance_eigh"):
            return self._fit_full(X, n_components, xp, is_array_api_compliant)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            return self._fit_truncated(X, n_components, xp)
    def score_samples(self, X):
        """
        Return the log-likelihood of each sample.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        Returns
        -------
        ll : ndarray of shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 获取数据类型的命名空间
        xp, _ = get_namespace(X)
        # 验证数据，并确保数据类型正确
        X = self._validate_data(X, dtype=[xp.float64, xp.float32], reset=False)
        # 计算数据与模型均值的差异
        Xr = X - self.mean_
        # 获取精度矩阵
        precision = self.get_precision()
        # 计算每个样本的对数似然
        log_like = -0.5 * xp.sum(Xr * (Xr @ precision), axis=1)
        # 加上常数项以计算对数似然
        log_like -= 0.5 * (X.shape[1] * log(2.0 * np.pi) - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """
        Return the average log-likelihood of all samples.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        y : Ignored
            Ignored.

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model.
        """
        # 获取数据类型的命名空间
        xp, _ = get_namespace(X)
        # 计算并返回平均对数似然
        return float(xp.mean(self.score_samples(X)))

    def _more_tags(self):
        """
        Provide additional tags for the estimator.

        Returns
        -------
        dict
            A dictionary of additional tags describing the estimator.
        """
        return {"preserves_dtype": [np.float64, np.float32], "array_api_support": True}
```