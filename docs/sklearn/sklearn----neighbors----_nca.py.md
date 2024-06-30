# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_nca.py`

```
# 导入必要的模块和库
import sys  # 导入系统模块
import time  # 导入时间模块
from numbers import Integral, Real  # 导入数字相关模块
from warnings import warn  # 导入警告模块

import numpy as np  # 导入NumPy库
from scipy.optimize import minimize  # 导入SciPy库中的minimize函数

# 导入scikit-learn中的各种模块和类
from ..base import (
    BaseEstimator,  # 导入基础估计器类
    ClassNamePrefixFeaturesOutMixin,  # 导入类名前缀特征输出混合类
    TransformerMixin,  # 导入变换器混合类
    _fit_context,  # 导入内部_fit上下文
)
from ..decomposition import PCA  # 导入PCA分解类
from ..exceptions import ConvergenceWarning  # 导入收敛警告异常
from ..metrics import pairwise_distances  # 导入成对距离度量函数
from ..preprocessing import LabelEncoder  # 导入标签编码器类
from ..utils._param_validation import Interval, StrOptions  # 导入参数验证相关工具
from ..utils.extmath import softmax  # 导入softmax函数
from ..utils.multiclass import check_classification_targets  # 导入检查分类目标函数
from ..utils.random import check_random_state  # 导入随机状态检查函数
from ..utils.validation import check_array, check_is_fitted  # 导入数组验证和拟合验证函数


class NeighborhoodComponentsAnalysis(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """邻域成分分析 (Neighborhood Components Analysis, NCA)。

    邻域成分分析是一种用于度量学习的机器学习算法。它以监督方式学习线性变换，
    以改善转换空间中随机最近邻规则的分类准确性。

    更多信息请参考 :ref:`用户指南 <nca>`。

    Parameters
    ----------
    n_components : int, default=None
        期望的投影空间的维数。
        如果为None，则将设置为`n_features`。
    # 初始化线性变换的方式。可以是以下几种选项之一：
    # `'auto'`, `'pca'`, `'lda'`, `'identity'`, `'random'`，或者是一个形状为 `(n_features_a, n_features_b)` 的 numpy 数组。
    init : {'auto', 'pca', 'lda', 'identity', 'random'} or ndarray of shape \
            (n_features_a, n_features_b), default='auto'
        Initialization of the linear transformation. Possible options are
        `'auto'`, `'pca'`, `'lda'`, `'identity'`, `'random'`, and a numpy
        array of shape `(n_features_a, n_features_b)`.

        - `'auto'`
            Depending on `n_components`, the most reasonable initialization
            will be chosen. If `n_components <= n_classes` we use `'lda'`, as
            it uses labels information. If not, but
            `n_components < min(n_features, n_samples)`, we use `'pca'`, as
            it projects data in meaningful directions (those of higher
            variance). Otherwise, we just use `'identity'`.

        - `'pca'`
            `n_components` principal components of the inputs passed
            to :meth:`fit` will be used to initialize the transformation.
            (See :class:`~sklearn.decomposition.PCA`)

        - `'lda'`
            `min(n_components, n_classes)` most discriminative
            components of the inputs passed to :meth:`fit` will be used to
            initialize the transformation. (If `n_components > n_classes`,
            the rest of the components will be zero.) (See
            :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)

        - `'identity'`
            If `n_components` is strictly smaller than the
            dimensionality of the inputs passed to :meth:`fit`, the identity
            matrix will be truncated to the first `n_components` rows.

        - `'random'`
            The initial transformation will be a random array of shape
            `(n_components, n_features)`. Each value is sampled from the
            standard normal distribution.

        - numpy array
            `n_features_b` must match the dimensionality of the inputs passed
            to :meth:`fit` and n_features_a must be less than or equal to that.
            If `n_components` is not `None`, `n_features_a` must match it.

    # 若为 True，并且之前已调用过 :meth:`fit`，则会使用前一次调用 :meth:`fit` 的解作为初始线性变换（`n_components` 和 `init` 将被忽略）。
    warm_start : bool, default=False
        If `True` and :meth:`fit` has been called before, the solution of the
        previous call to :meth:`fit` is used as the initial linear
        transformation (`n_components` and `init` will be ignored).

    # 优化过程中的最大迭代次数。
    max_iter : int, default=50
        Maximum number of iterations in the optimization.

    # 优化过程的收敛容忍度。
    tol : float, default=1e-5
        Convergence tolerance for the optimization.

    # 回调函数，如果不为 `None`，则会在优化器的每次迭代后调用，传入当前解（扁平化的变换矩阵）和迭代次数作为参数。
    callback : callable, default=None
        If not `None`, this function is called after every iteration of the
        optimizer, taking as arguments the current solution (flattened
        transformation matrix) and the number of iterations. This might be
        useful in case one wants to examine or store the transformation
        found after each iteration.
    verbose : int, default=0
        如果为0，则不打印任何进度消息。
        如果为1，则将进度消息打印到标准输出。
        如果大于1，则将打印进度消息，并且 :func:`scipy.optimize.minimize` 的 `disp` 参数将设置为 `verbose - 2`。

    random_state : int or numpy.RandomState, default=None
        伪随机数生成器对象或其种子（如果是 int）。如果 `init='random'`，则使用 `random_state` 初始化随机变换。
        如果 `init='pca'`，则在初始化变换时将 `random_state` 传递给 PCA。为了获得可重复的结果，请传递一个 int。
        参见 :term:`Glossary <random_state>`。

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        在拟合过程中学习到的线性变换。

    n_features_in_ : int
        在 :term:`fit` 过程中看到的特征数。

        .. versionadded:: 0.24

    n_iter_ : int
        优化器执行的迭代次数。

    random_state_ : numpy.RandomState
        初始化过程中使用的伪随机数生成器对象。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 :term:`fit` 过程中看到的特征名称。仅当 `X` 中的特征名称全为字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis : 线性判别分析。
    sklearn.decomposition.PCA : 主成分分析（PCA）。

    References
    ----------
    .. [1] J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov.
           "Neighbourhood Components Analysis". Advances in Neural Information
           Processing Systems. 17, 513-520, 2005.
           http://www.cs.nyu.edu/~roweis/papers/ncanips.pdf

    .. [2] Wikipedia entry on Neighborhood Components Analysis
           https://en.wikipedia.org/wiki/Neighbourhood_components_analysis

    Examples
    --------
    >>> from sklearn.neighbors import NeighborhoodComponentsAnalysis
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ... stratify=y, test_size=0.7, random_state=42)
    >>> nca = NeighborhoodComponentsAnalysis(random_state=42)
    >>> nca.fit(X_train, y_train)
    NeighborhoodComponentsAnalysis(...)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> knn.fit(X_train, y_train)
    KNeighborsClassifier(...)
    >>> print(knn.score(X_test, y_test))
    0.933333...
    >>> knn.fit(nca.transform(X_train), y_train)
    KNeighborsClassifier(...)
    >>> print(knn.score(nca.transform(X_test), y_test))
    0.961904...
    # 参数约束字典，定义了模型参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),  # n_components必须为整数且大于等于1
            None,  # n_components可以为None
        ],
        "init": [
            StrOptions({"auto", "pca", "lda", "identity", "random"}),  # init必须是预定义的字符串集合中的一员
            np.ndarray,  # init可以是NumPy数组
        ],
        "warm_start": ["boolean"],  # warm_start必须是布尔类型
        "max_iter": [Interval(Integral, 1, None, closed="left")],  # max_iter必须为大于等于1的整数
        "tol": [Interval(Real, 0, None, closed="left")],  # tol必须为大于等于0的实数
        "callback": [callable, None],  # callback可以是可调用对象或者None
        "verbose": ["verbose"],  # verbose必须是预定义的字符串"verbose"
        "random_state": ["random_state"],  # random_state必须是预定义的字符串"random_state"
    }

    # 模型初始化函数，设置模型的各个参数
    def __init__(
        self,
        n_components=None,
        *,
        init="auto",
        warm_start=False,
        max_iter=50,
        tol=1e-5,
        callback=None,
        verbose=0,
        random_state=None,
    ):
        self.n_components = n_components  # 设置模型参数n_components
        self.init = init  # 设置模型参数init
        self.warm_start = warm_start  # 设置模型参数warm_start
        self.max_iter = max_iter  # 设置模型参数max_iter
        self.tol = tol  # 设置模型参数tol
        self.callback = callback  # 设置模型参数callback
        self.verbose = verbose  # 设置模型参数verbose
        self.random_state = random_state  # 设置模型参数random_state

    # 装饰器函数，用于包装transform方法，在_fit_context中进行预处理
    @_fit_context(prefer_skip_nested_validation=True)
    def transform(self, X):
        """Apply the learned transformation to the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data samples.

        Returns
        -------
        X_embedded: ndarray of shape (n_samples, n_components)
            The data samples transformed.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called before.
        """

        check_is_fitted(self)  # 检查模型是否已经拟合过，若未拟合则抛出NotFittedError异常
        X = self._validate_data(X, reset=False)  # 验证输入数据X的格式并复制数据，不重置数据

        return np.dot(X, self.components_.T)  # 返回数据X与模型的主成分(components_)的转置的乘积
    def _initialize(self, X, y, init):
        """Initialize the transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练样本数据集，形状为 (样本数, 特征数)。

        y : array-like of shape (n_samples,)
            训练标签数据，形状为 (样本数,)。

        init : str or ndarray of shape (n_features_a, n_features_b)
            线性转换的初始化参数，可以是字符串或者二维数组，形状为 (特征数a, 特征数b)。

        Returns
        -------
        transformation : ndarray of shape (n_components, n_features)
            初始化后的线性转换矩阵，形状为 (转换组件数, 特征数)。

        """

        transformation = init
        if self.warm_start and hasattr(self, "components_"):
            transformation = self.components_
        elif isinstance(init, np.ndarray):
            pass
        else:
            n_samples, n_features = X.shape
            n_components = self.n_components or n_features
            if init == "auto":
                n_classes = len(np.unique(y))
                if n_components <= min(n_features, n_classes - 1):
                    init = "lda"
                elif n_components < min(n_features, n_samples):
                    init = "pca"
                else:
                    init = "identity"
            if init == "identity":
                # 初始化为单位矩阵
                transformation = np.eye(n_components, X.shape[1])
            elif init == "random":
                # 随机初始化
                transformation = self.random_state_.standard_normal(
                    size=(n_components, X.shape[1])
                )
            elif init in {"pca", "lda"}:
                init_time = time.time()
                if init == "pca":
                    # 使用主成分分析 (PCA) 初始化
                    pca = PCA(
                        n_components=n_components, random_state=self.random_state_
                    )
                    if self.verbose:
                        print("Finding principal components... ", end="")
                        sys.stdout.flush()
                    pca.fit(X)
                    transformation = pca.components_
                elif init == "lda":
                    from ..discriminant_analysis import LinearDiscriminantAnalysis

                    # 使用线性判别分析 (LDA) 初始化
                    lda = LinearDiscriminantAnalysis(n_components=n_components)
                    if self.verbose:
                        print("Finding most discriminative components... ", end="")
                        sys.stdout.flush()
                    lda.fit(X, y)
                    transformation = lda.scalings_.T[:n_components]
                if self.verbose:
                    print("done in {:5.2f}s".format(time.time() - init_time))
        return transformation
    def _callback(self, transformation):
        """
        Called after each iteration of the optimizer.

        Parameters
        ----------
        transformation : ndarray of shape (n_components * n_features,)
            The solution computed by the optimizer in this iteration.
        """
        # 检查是否定义了回调函数，如果有则调用回调函数，并传入当前的变换结果和迭代次数
        if self.callback is not None:
            self.callback(transformation, self.n_iter_)

        # 增加迭代次数计数器
        self.n_iter_ += 1
    def _loss_grad_lbfgs(self, transformation, X, same_class_mask, sign=1.0):
        """Compute the loss and the loss gradient w.r.t. `transformation`.

        Parameters
        ----------
        transformation : ndarray of shape (n_components * n_features,)
            The raveled linear transformation on which to compute loss and
            evaluate gradient.

        X : ndarray of shape (n_samples, n_features)
            The training samples.

        same_class_mask : ndarray of shape (n_samples, n_samples)
            A mask where `mask[i, j] == 1` if `X[i]` and `X[j]` belong
            to the same class, and `0` otherwise.

        Returns
        -------
        loss : float
            The loss computed for the given transformation.

        gradient : ndarray of shape (n_components * n_features,)
            The new (flattened) gradient of the loss.
        """

        if self.n_iter_ == 0:  # 如果迭代次数为0
            self.n_iter_ += 1  # 增加迭代次数计数
            if self.verbose:  # 如果设置了详细输出
                header_fields = ["Iteration", "Objective Value", "Time(s)"]
                header_fmt = "{:>10} {:>20} {:>10}"
                header = header_fmt.format(*header_fields)  # 格式化表头
                cls_name = self.__class__.__name__
                print("[{}]".format(cls_name))  # 打印类名
                print(
                    "[{}] {}\n[{}] {}".format(
                        cls_name, header, cls_name, "-" * len(header)
                    )
                )  # 打印带格式的表头

        t_funcall = time.time()  # 记录函数调用时间

        transformation = transformation.reshape(-1, X.shape[1])  # 调整变换矩阵的形状
        X_embedded = np.dot(X, transformation.T)  # 计算嵌入空间中的转换后样本点坐标 (n_samples, n_components)

        # 计算 softmax 距离
        p_ij = pairwise_distances(X_embedded, squared=True)  # 计算样本点之间的平方欧氏距离
        np.fill_diagonal(p_ij, np.inf)  # 将对角线上的距离设置为无穷大
        p_ij = softmax(-p_ij)  # 对距离进行 softmax 变换，得到相似度 (n_samples, n_samples)

        # 计算损失
        masked_p_ij = p_ij * same_class_mask  # 根据类别掩码计算掩码后的相似度
        p = np.sum(masked_p_ij, axis=1, keepdims=True)  # 沿行求和，得到每个样本点的相似度之和 (n_samples, 1)
        loss = np.sum(p)  # 总损失为所有相似度之和

        # 计算损失相对于 `transform` 的梯度
        weighted_p_ij = masked_p_ij - p_ij * p  # 计算加权后的相似度差
        weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T  # 对称化加权后的相似度差
        np.fill_diagonal(weighted_p_ij_sym, -weighted_p_ij.sum(axis=0))  # 对角线上的元素设为负数
        gradient = 2 * X_embedded.T.dot(weighted_p_ij_sym).dot(X)  # 计算损失的梯度 (flattened)

        # 梯度的时间复杂度: O(n_components x n_samples x (n_samples + n_features))

        if self.verbose:  # 如果设置了详细输出
            t_funcall = time.time() - t_funcall  # 计算函数调用时间
            values_fmt = "[{}] {:>10} {:>20.6e} {:>10.2f}"
            print(
                values_fmt.format(
                    self.__class__.__name__, self.n_iter_, loss, t_funcall
                )
            )  # 输出详细信息
            sys.stdout.flush()  # 刷新标准输出

        return sign * loss, sign * gradient.ravel()  # 返回损失及其梯度的压平形式

    def _more_tags(self):
        """Return additional tags."""
        return {"requires_y": True}  # 返回一个包含额外标签的字典

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components_.shape[0]  # 返回转换后输出特征的数量
```