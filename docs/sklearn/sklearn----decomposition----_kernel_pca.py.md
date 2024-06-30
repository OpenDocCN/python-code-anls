# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_kernel_pca.py`

```
"""Kernel Principal Components Analysis."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从 numbers 模块导入 Integral（整数类型）、Real（实数类型）
from numbers import Integral, Real

# 导入必要的库
import numpy as np
from scipy import linalg
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

# 从 scikit-learn 中导入基础类和异常处理
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..exceptions import NotFittedError
# 从 scikit-learn 中导入度量工具和预处理工具
from ..metrics.pairwise import pairwise_kernels
from ..preprocessing import KernelCenterer
# 从 scikit-learn 中导入工具函数和验证函数
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import _randomized_eigsh, svd_flip
from ..utils.validation import (
    _check_psd_eigenvalues,
    check_is_fitted,
)

# 定义 KernelPCA 类，继承自多个基础类
class KernelPCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Kernel Principal component analysis (KPCA).

    Non-linear dimensionality reduction through the use of kernels [1]_, see also
    :ref:`metrics`.

    It uses the :func:`scipy.linalg.eigh` LAPACK implementation of the full SVD
    or the :func:`scipy.sparse.linalg.eigsh` ARPACK implementation of the
    truncated SVD, depending on the shape of the input data and the number of
    components to extract. It can also use a randomized truncated SVD by the
    method proposed in [3]_, see `eigen_solver`.

    For a usage example and comparison between
    Principal Components Analysis (PCA) and its kernelized version (KPCA), see
    :ref:`sphx_glr_auto_examples_decomposition_plot_kernel_pca.py`.

    For a usage example in denoising images using KPCA, see
    :ref:`sphx_glr_auto_examples_applications_plot_digits_denoising.py`.

    Read more in the :ref:`User Guide <kernel_PCA>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of components. If None, all non-zero components are kept.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'} \
            or callable, default='linear'
        Kernel used for PCA.

    gamma : float, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels. If ``gamma`` is ``None``, then it is set to ``1/n_features``.

    degree : float, default=3
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict, default=None
        Parameters (keyword arguments) and
        values for kernel passed as callable object.
        Ignored by other kernels.

    alpha : float, default=1.0
        Hyperparameter of the ridge regression that learns the
        inverse transform (when fit_inverse_transform=True).

    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels
        (i.e. learn to find the pre-image of a point). This method is based
        on [2]_.
    """
    # 类的初始化方法，定义 KPCA 的参数和属性
    def __init__(
        self,
        n_components=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 KPCA 的各个参数
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform

    # 省略了其余的类方法和实现细节
    eigen_solver : {'auto', 'dense', 'arpack', 'randomized'}, \
            default='auto'
        # 选择要使用的特征值求解器。如果 `n_components` 远小于训练样本数量，
        # 随机化方法（或在较小程度上 arpack）可能比稠密特征值求解器更有效。
        # 根据 Halko 等人的方法执行随机化奇异值分解 [3]_。
        
        auto :
            # 根据默认策略基于 n_samples（训练样本数）和 `n_components` 选择求解器：
            # 如果要提取的组件数量小于 10（严格）且样本数量大于 200（严格），
            # 则启用 'arpack' 方法。否则，计算精确的完整特征值分解，
            # 并选择后续进行截断（'dense' 方法）。
        dense :
            # 调用 `scipy.linalg.eigh` 使用标准 LAPACK 求解器运行精确的完整特征值分解，
            # 并通过后处理选择组件。
        arpack :
            # 调用 `scipy.sparse.linalg.eigsh` 运行截断到 n_components 的 SVD，
            # 使用 ARPACK 求解器。要求严格 0 < n_components < n_samples。
        randomized :
            # 使用 Halko 等人的方法 [3]_ 运行随机化奇异值分解。当前实现根据它们的模块选择特征值；
            # 因此，如果核不是半正定的，则使用此方法可能会导致意外结果。另见 [4]_。
        
        .. versionchanged:: 1.0
           #'randomized'` 被添加。

    tol : float, default=0
        # arpack 的收敛容限。如果为 0，则 arpack 将选择最佳值。

    max_iter : int, default=None
        # arpack 的最大迭代次数。如果为 None，则 arpack 将选择最佳值。

    iterated_power : int >= 0, or 'auto', default='auto'
        # 由 svd_solver == 'randomized' 计算的幂法迭代次数。
        # 当 'auto' 时，如果 `n_components < 0.1 * min(X.shape)`，则设置为 7；
        # 其他情况设置为 4。
        
        .. versionadded:: 1.0

    remove_zero_eig : bool, default=False
        # 如果为 True，则移除所有具有零特征值的组件，因此输出中的组件数量可能小于 n_components
        # （有时甚至为零，由于数值不稳定性）。当 n_components 为 None 时，此参数被忽略，
        # 并且所有具有零特征值的组件将被移除。

    random_state : int, RandomState instance or None, default=None
        # 当 ``eigen_solver`` == 'arpack' 或 'randomized' 时使用。传递一个整数以在多次函数调用间获得可重现的结果。
        # 参见 :term:`Glossary <random_state>`。

        .. versionadded:: 0.18
    copy_X : bool, default=True
        # 如果为True，将输入X复制并存储在模型的X_fit_属性中。
        # 如果不再对X进行更改，则设置`copy_X=False`可以通过存储引用来节省内存。
        .. versionadded:: 0.18

    n_jobs : int, default=None
        # 要运行的并行作业数。
        # ``None`` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则默认为1。
        # ``-1`` 表示使用所有处理器。有关详细信息，请参见 :term:`Glossary <n_jobs>`。
        .. versionadded:: 0.18

    Attributes
    ----------
    eigenvalues_ : ndarray of shape (n_components,)
        # 中心化内核矩阵的特征值，按降序排列。
        # 如果未设置 `n_components` 和 `remove_zero_eig`，则存储所有值。

    eigenvectors_ : ndarray of shape (n_samples, n_components)
        # 中心化内核矩阵的特征向量。
        # 如果未设置 `n_components` 和 `remove_zero_eig`，则存储所有组件。

    dual_coef_ : ndarray of shape (n_samples, n_features)
        # 逆变换矩阵。仅在 `fit_inverse_transform=True` 时可用。

    X_transformed_fit_ : ndarray of shape (n_samples, n_components)
        # 拟合数据在核主成分上的投影。
        # 仅在 `fit_inverse_transform=True` 时可用。

    X_fit_ : ndarray of shape (n_samples, n_features)
        # 用于拟合模型的数据。如果 `copy_X=False`，则 `X_fit_` 是一个引用。
        # 此属性用于调用 transform。

    n_features_in_ : int
        # :term:`fit` 过程中观察到的特征数。
        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # :term:`fit` 过程中观察到的特征名称。仅当 `X` 的特征名称全为字符串时定义。
        .. versionadded:: 1.0

    gamma_ : float
        # rbf、poly 和 sigmoid 核的核系数。当显式提供 `gamma` 时，与 `gamma` 相同。
        # 当 `gamma` 为 `None` 时，这是核系数的实际值。
        .. versionadded:: 1.3

    See Also
    --------
    FastICA : 独立成分分析的快速算法。
    IncrementalPCA : 增量主成分分析。
    NMF : 非负矩阵分解。
    PCA : 主成分分析。
    SparsePCA : 稀疏主成分分析。
    TruncatedSVD : 使用截断SVD进行降维。

    References
    ----------
    .. [1] `Schölkopf, Bernhard, Alexander Smola, and Klaus-Robert Müller.
       "Kernel principal component analysis."
       International conference on artificial neural networks.
       Springer, Berlin, Heidelberg, 1997.
       <https://people.eecs.berkeley.edu/~wainwrig/stat241b/scholkopf_kernel.pdf>`_
    # 定义参数约束字典，用于描述 `KernelPCA` 类的参数约束条件
    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),  # n_components 必须是整数且大于等于1
            None,  # n_components 可以为 None
        ],
        "kernel": [
            StrOptions({"linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"}),  # kernel 可以是预定义的字符串选项集合
            callable,  # kernel 可以是可调用对象
        ],
        "gamma": [
            Interval(Real, 0, None, closed="left"),  # gamma 必须是大于0的实数
            None,  # gamma 可以为 None
        ],
        "degree": [Interval(Real, 0, None, closed="left")],  # degree 必须是大于0的实数
        "coef0": [Interval(Real, None, None, closed="neither")],  # coef0 可以是任意实数，但不包括无穷
        "kernel_params": [dict, None],  # kernel_params 可以是字典或者 None
        "alpha": [Interval(Real, 0, None, closed="left")],  # alpha 必须是大于0的实数
        "fit_inverse_transform": ["boolean"],  # fit_inverse_transform 必须是布尔值
        "eigen_solver": [StrOptions({"auto", "dense", "arpack", "randomized"})],  # eigen_solver 必须是预定义的字符串选项集合
        "tol": [Interval(Real, 0, None, closed="left")],  # tol 必须是大于0的实数
        "max_iter": [
            Interval(Integral, 1, None, closed="left"),  # max_iter 必须是整数且大于等于1
            None,  # max_iter 可以为 None
        ],
        "iterated_power": [
            Interval(Integral, 0, None, closed="left"),  # iterated_power 必须是整数且大于等于0
            StrOptions({"auto"}),  # iterated_power 可以是字符串 "auto"
        ],
        "remove_zero_eig": ["boolean"],  # remove_zero_eig 必须是布尔值
        "random_state": ["random_state"],  # random_state 必须是随机状态对象
        "copy_X": ["boolean"],  # copy_X 必须是布尔值
        "n_jobs": [None, Integral],  # n_jobs 可以为 None 或者整数
    }
    
    def __init__(
        self,
        n_components=None,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        random_state=None,
        copy_X=True,
        n_jobs=None,
    ):
        self.n_components = n_components  # 设置模型的主成分数量
        self.kernel = kernel  # 设置核函数类型
        self.kernel_params = kernel_params  # 设置核函数的参数
        self.gamma = gamma  # 设置核函数的 gamma 参数
        self.degree = degree  # 设置核函数的多项式度数
        self.coef0 = coef0  # 设置核函数的常数项
        self.alpha = alpha  # 设置正则化参数
        self.fit_inverse_transform = fit_inverse_transform  # 是否进行逆变换的标志
        self.eigen_solver = eigen_solver  # 设置特征值求解器类型
        self.tol = tol  # 设置迭代收敛的容忍度
        self.max_iter = max_iter  # 设置最大迭代次数
        self.iterated_power = iterated_power  # 设置幂方法的迭代次数
        self.remove_zero_eig = remove_zero_eig  # 是否移除特征值为零的特征向量的标志
        self.random_state = random_state  # 设置随机数生成器的种子
        self.n_jobs = n_jobs  # 并行计算时使用的 CPU 核数
        self.copy_X = copy_X  # 是否复制输入数据的标志

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}  # 如果核函数是可调用的，则使用指定的参数
        else:
            params = {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}  # 否则使用预定义的参数
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params  # 计算指定核函数下的核矩阵
        )

    def _fit_inverse_transform(self, X_transformed, X):
        if hasattr(X, "tocsr"):
            raise NotImplementedError(
                "Inverse transform not implemented for sparse matrices!"  # 当输入是稀疏矩阵时抛出未实现的异常
            )

        n_samples = X_transformed.shape[0]
        K = self._get_kernel(X_transformed)
        K.flat[:: n_samples + 1] += self.alpha  # 在核矩阵的对角线上增加正则化参数
        self.dual_coef_ = linalg.solve(K, X, assume_a="pos", overwrite_a=True)  # 求解线性方程组来估计逆变换的系数
        self.X_transformed_fit_ = X_transformed  # 保存变换后的数据用于逆变换

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.fit_inverse_transform and self.kernel == "precomputed":
            raise ValueError("Cannot fit_inverse_transform with a precomputed kernel.")  # 如果设置了逆变换并且核函数为预计算，则抛出异常

        X = self._validate_data(X, accept_sparse="csr", copy=self.copy_X)  # 验证输入数据格式，并进行必要的复制

        self.gamma_ = 1 / X.shape[1] if self.gamma is None else self.gamma  # 计算默认 gamma 值
        self._centerer = KernelCenterer().set_output(transform="default")  # 设置核中心化对象及其输出类型
        K = self._get_kernel(X)  # 计算输入数据的核矩阵

        # 当核函数为预计算时，K 与 X 相等，可以安全地原地操作 K
        self._fit_transform_in_place(K)

        if self.fit_inverse_transform:
            # 不需要使用核函数来转换 X，使用简化的表达式
            X_transformed = self.eigenvectors_ * np.sqrt(self.eigenvalues_)

            self._fit_inverse_transform(X_transformed, X)  # 执行逆变换的拟合

        self.X_fit_ = X  # 保存拟合的输入数据
        return self  # 返回模型实例本身
    def fit_transform(self, X, y=None, **params):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        **params : kwargs
            Parameters (keyword arguments) and values passed to
            the fit_transform instance.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Returns the instance itself.
        """
        # 调用fit方法对模型进行训练
        self.fit(X, **params)

        # 根据特征值和特征向量计算变换后的数据
        X_transformed = self.eigenvectors_ * np.sqrt(self.eigenvalues_)

        # 如果设置了逆变换的训练，则进行逆变换的计算
        if self.fit_inverse_transform:
            self._fit_inverse_transform(X_transformed, X)

        # 返回变换后的数据
        return X_transformed

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Returns the instance itself.
        """
        # 检查模型是否已经进行过拟合
        check_is_fitted(self)
        
        # 验证输入数据X，并接受稀疏矩阵格式的数据，不重置数据
        X = self._validate_data(X, accept_sparse="csr", reset=False)

        # 计算中心化的Gram矩阵，用于输入数据X和训练数据X_fit_之间的核变换
        K = self._centerer.transform(self._get_kernel(X, self.X_fit_))

        # 缩放特征向量（正确地处理零空间以进行点积计算）
        non_zeros = np.flatnonzero(self.eigenvalues_)
        scaled_alphas = np.zeros_like(self.eigenvectors_)
        scaled_alphas[:, non_zeros] = self.eigenvectors_[:, non_zeros] / np.sqrt(
            self.eigenvalues_[non_zeros]
        )

        # 用K和缩放后的特征向量进行点积投影
        return np.dot(K, scaled_alphas)
    def inverse_transform(self, X):
        """
        Transform X back to original space.

        ``inverse_transform`` approximates the inverse transformation using
        a learned pre-image. The pre-image is learned by kernel ridge
        regression of the original data on their low-dimensional representation
        vectors.

        .. note:
            :meth:`~sklearn.decomposition.fit` internally uses a centered
            kernel. As the centered kernel no longer contains the information
            of the mean of kernel features, such information is not taken into
            account in reconstruction.

        .. note::
            When users want to compute inverse transformation for 'linear'
            kernel, it is recommended that they use
            :class:`~sklearn.decomposition.PCA` instead. Unlike
            :class:`~sklearn.decomposition.PCA`,
            :class:`~sklearn.decomposition.KernelPCA`'s ``inverse_transform``
            does not reconstruct the mean of data when 'linear' kernel is used
            due to the use of centered kernel.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_components)
            Training vector, where `n_samples` is the number of samples
            and `n_components` is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Returns the instance itself.

        References
        ----------
        `Bakır, Gökhan H., Jason Weston, and Bernhard Schölkopf.
        "Learning to find pre-images."
        Advances in neural information processing systems 16 (2004): 449-456.
        <https://papers.nips.cc/paper/2003/file/ac1ad983e08ad3304a97e147f522747e-Paper.pdf>`_
        """
        # 如果未设置 fit_inverse_transform 参数为 True，则抛出 NotFittedError 异常
        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        # 使用 _get_kernel 方法计算输入数据 X 和拟合时转换后的数据之间的核矩阵 K
        K = self._get_kernel(X, self.X_transformed_fit_)
        
        # 返回核矩阵 K 与 dual_coef_ 的乘积，以完成逆变换过程
        return np.dot(K, self.dual_coef_)

    def _more_tags(self):
        """
        Return additional tags for the estimator.

        This method provides additional information about the estimator's behavior
        such as whether it preserves dtype and whether it operates pairwise.

        Returns
        -------
        dict
            A dictionary containing additional tags.
        """
        # 返回一个字典，包含有关估计器行为的额外标签信息
        return {
            "preserves_dtype": [np.float64, np.float32],
            "pairwise": self.kernel == "precomputed",
        }

    @property
    def _n_features_out(self):
        """
        Number of transformed output features.

        Returns
        -------
        int
            Number of features in the transformed output.
        """
        # 返回特征转换后的输出特征数量，由 eigenvalues_ 的形状决定
        return self.eigenvalues_.shape[0]
```