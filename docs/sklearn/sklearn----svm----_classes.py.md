# `D:\src\scipysrc\scikit-learn\sklearn\svm\_classes.py`

```
# 从 numbers 模块中导入 Integral（整数）和 Real（实数）类型
from numbers import Integral, Real

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 从父级目录中导入以下模块：
# BaseEstimator：基础估计器
# OutlierMixin：异常值混合器
# RegressorMixin：回归器混合器
# _fit_context：拟合上下文
from ..base import BaseEstimator, OutlierMixin, RegressorMixin, _fit_context

# 从 linear_model 模块中导入以下模块：
# LinearClassifierMixin：线性分类器混合器
# LinearModel：线性模型
# SparseCoefMixin：稀疏系数混合器
from ..linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin

# 从 utils._param_validation 模块中导入以下类：
# Interval：区间验证器
# StrOptions：字符串选项验证器
from ..utils._param_validation import Interval, StrOptions

# 从 utils.multiclass 模块中导入 check_classification_targets 函数
from ..utils.multiclass import check_classification_targets

# 从 utils.validation 模块中导入 _num_samples 函数
from ..utils.validation import _num_samples

# 从当前目录下的 _base 模块中导入以下类和函数：
# BaseLibSVM：基础 LibSVM 类
# BaseSVC：基础支持向量分类器类
# _fit_liblinear：拟合 Liblinear 函数
# _get_liblinear_solver_type：获取 Liblinear 求解器类型函数
from ._base import BaseLibSVM, BaseSVC, _fit_liblinear, _get_liblinear_solver_type


def _validate_dual_parameter(dual, loss, penalty, multi_class, X):
    """Helper function to assign the value of dual parameter."""
    # 如果 dual 参数为 "auto"
    if dual == "auto":
        # 如果样本数小于特征数
        if X.shape[0] < X.shape[1]:
            try:
                # 尝试获取 Liblinear 求解器类型，multi_class、penalty、loss 和 True 作为参数
                _get_liblinear_solver_type(multi_class, penalty, loss, True)
                # 返回 True
                return True
            # 捕获 ValueError 异常，表示该组合不支持 dual
            except ValueError:
                # 返回 False
                return False
        else:
            try:
                # 尝试获取 Liblinear 求解器类型，multi_class、penalty、loss 和 False 作为参数
                _get_liblinear_solver_type(multi_class, penalty, loss, False)
                # 返回 False
                return False
            # 捕获 ValueError 异常，表示该组合不支持 primal
            except ValueError:
                # 返回 True
                return True
    else:
        # 如果 dual 参数不为 "auto"，直接返回 dual 参数的值
        return dual


class LinearSVC(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    """Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    The main differences between :class:`~sklearn.svm.LinearSVC` and
    :class:`~sklearn.svm.SVC` lie in the loss function used by default, and in
    the handling of intercept regularization between those two implementations.

    This class supports both dense and sparse input and the multiclass support
    is handled according to a one-vs-the-rest scheme.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    penalty : {'l1', 'l2'}, default='l2'
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to ``coef_``
        vectors that are sparse.

    loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
        Specifies the loss function. 'hinge' is the standard SVM loss
        (used e.g. by the SVC class) while 'squared_hinge' is the
        square of the hinge loss. The combination of ``penalty='l1'``
        and ``loss='hinge'`` is not supported.
    """
    # 双重优化问题求解方式选择，默认为"auto"
    dual : "auto" or bool, default="auto"
        选择解决双重优化问题的算法，可以是"auto"或布尔值
        当样本数大于特征数时，推荐设置 dual=False。
        `dual="auto"` 将根据 `n_samples`、`n_features`、`loss`、`multi_class` 和 `penalty` 的值自动选择参数值。
        如果 `n_samples` < `n_features` 并且优化器支持选择的 `loss`、`multi_class` 和 `penalty`，则 dual 将被设置为 True；
        否则将设置为 False。

        .. versionchanged:: 1.3
           版本 1.3 中添加了 `"auto"` 选项，并将在版本 1.5 中成为默认选项。

    # 停止标准的容忍度
    tol : float, default=1e-4
        停止标准的容忍度阈值。

    # 正则化参数 C
    C : float, default=1.0
        正则化参数，正则化强度与 C 的倒数成正比关系。必须严格为正数。
        为了直观地理解正则化参数 C 的缩放效果，请参见
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`。

    # 多类别分类策略选择
    multi_class : {'ovr', 'crammer_singer'}, default='ovr'
        如果 `y` 包含多于两个类别，确定多类别策略。
        ``"ovr"`` 训练 n_classes 个一对多分类器；
        ``"crammer_singer"`` 优化所有类别的联合目标。
        尽管 `crammer_singer` 在理论上是一致的，但在实践中很少使用，因为它很少提高准确性，并且计算成本更高。
        如果选择了 ``"crammer_singer"``，则会忽略选项 loss、penalty 和 dual。

    # 是否拟合截距
    fit_intercept : bool, default=True
        是否拟合截距。如果设置为 True，则特征向量会扩展以包括截距项 `[x_1, ..., x_n, 1]`，其中 1 对应截距。
        如果设置为 False，则计算中不会使用截距（即预期数据已经居中）。
    intercept_scaling : float, default=1.0
        # 当 fit_intercept=True 时，将一个常量值等于 intercept_scaling 添加到实例向量 x 的末尾，
        # 使其变为 ``[x_1, ..., x_n, intercept_scaling]``。这相当于添加了一个“合成”特征。
        # 截距变为 intercept_scaling * synthetic feature weight。注意，liblinear 内部对截距进行惩罚，
        # 将其视为特征向量中的任何其他项。为了减少正则化对截距的影响，可以将 intercept_scaling 参数设置为大于 1 的值；
        # intercept_scaling 越高，正则化对其的影响越小。因此，权重变为 `[w_x_1, ..., w_x_n, w_intercept*intercept_scaling]`，
        # 其中 `w_x_1, ..., w_x_n` 表示特征权重，而截距权重则按 `intercept_scaling` 进行缩放。
        # 这种缩放使截距项与其他特征的正则化行为不同。

    class_weight : dict or 'balanced', default=None
        # 对于 SVC，将类 i 的参数 C 设置为 ``class_weight[i]*C``。
        # 如果未给出，则假定所有类的权重都为一。
        # "balanced" 模式根据 y 的值自动调整权重，与输入数据中类频率成反比，计算方式为 ``n_samples / (n_classes * np.bincount(y))``。

    verbose : int, default=0
        # 启用详细输出。注意，此设置利用 liblinear 中的每个进程运行时设置，
        # 如果启用，在多线程环境中可能无法正常工作。

    random_state : int, RandomState instance or None, default=None
        # 控制为双重坐标下降(shuffling)数据而进行的伪随机数生成。
        # 当 `dual=True` 时，如果 `dual=False`，则 :class:`LinearSVC` 的底层实现不是随机的，
        # `random_state` 对结果没有影响。为了跨多个函数调用产生可重复的输出，请传递一个整数。
        # 参见 :term:`Glossary <random_state>`。

    max_iter : int, default=1000
        # 要运行的最大迭代次数。

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 \
            else (n_classes, n_features)
        # 分配给特征的权重（原始问题中的系数）。

        # `coef_` 是从 `raw_coef_` 派生的只读属性，遵循 liblinear 的内部内存布局。

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        # 决策函数中的常数。

    classes_ : ndarray of shape (n_classes,)
        # 唯一的类标签。

    n_features_in_ : int
        # 在拟合期间看到的特征数。

        # .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中出现的特征名称列表，仅当 `X` 的特征名称全部为字符串时定义。

        .. versionadded:: 1.0
        # 版本 1.0 中添加的特性

    n_iter_ : int
        # 所有类别中运行的最大迭代次数。

    See Also
    --------
    SVC : 使用 libsvm 实现的支持向量机分类器：
        核可以是非线性的，但其 SMO 算法不适合大样本数，而 LinearSVC 适用。
        
        此外，SVC 使用一对一方案实现多类别模式，而 LinearSVC 使用一对其余方案。
        可以通过使用 :class:`~sklearn.multiclass.OneVsRestClassifier` 封装器实现 SVC 的一对其余方案。
        
        最后，如果输入是 C 连续的，SVC 可以在不进行内存复制的情况下适应密集数据。但是，稀疏数据仍然会导致内存复制。

    sklearn.linear_model.SGDClassifier : SGDClassifier 可以通过调整惩罚和损失参数优化与 LinearSVC 相同的成本函数。
        此外，它需要更少的内存，允许增量（在线）学习，并实现各种损失函数和正则化机制。

    Notes
    -----
    底层 C 实现在拟合模型时使用随机数生成器选择特征。因此，对于相同的输入数据，结果可能略有不同。
    如果发生这种情况，请尝试使用较小的 ``tol`` 参数。

    底层实现 liblinear 对数据使用稀疏内部表示，这将导致内存复制。

    在某些情况下，预测输出可能与独立 liblinear 的输出不匹配。请参阅文档中的 :ref:`liblinear_differences` 了解详细信息。

    References
    ----------
    `LIBLINEAR: A Library for Large Linear Classification
    <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__
        # 大规模线性分类的库 LIBLINEAR 的参考链接

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=0)
    >>> clf = make_pipeline(StandardScaler(),
    ...                     LinearSVC(random_state=0, tol=1e-5))
    >>> clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('linearsvc', LinearSVC(random_state=0, tol=1e-05))])

    >>> print(clf.named_steps['linearsvc'].coef_)
    [[0.141...   0.526... 0.679... 0.493...]]
        # 打印线性 SVM 模型的系数

    >>> print(clf.named_steps['linearsvc'].intercept_)
    [0.1693...]
        # 打印线性 SVM 模型的截距

    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
        # 对新样本进行预测，输出预测的类别
    # 参数约束字典，定义了每个参数的类型和可选取值范围
    _parameter_constraints: dict = {
        "penalty": [StrOptions({"l1", "l2"})],  # 惩罚项类型，可选取值为 {"l1", "l2"}
        "loss": [StrOptions({"hinge", "squared_hinge"})],  # 损失函数类型，可选取值为 {"hinge", "squared_hinge"}
        "dual": ["boolean", StrOptions({"auto"})],  # 是否使用对偶形式，可选为布尔值或者字符串 "auto"
        "tol": [Interval(Real, 0.0, None, closed="neither")],  # 收敛容限值，实数类型，大于0
        "C": [Interval(Real, 0.0, None, closed="neither")],  # 正则化参数，实数类型，大于0
        "multi_class": [StrOptions({"ovr", "crammer_singer"})],  # 多分类策略，可选为 {"ovr", "crammer_singer"}
        "fit_intercept": ["boolean"],  # 是否计算截距，布尔值
        "intercept_scaling": [Interval(Real, 0, None, closed="neither")],  # 截距缩放因子，实数类型，大于0
        "class_weight": [None, dict, StrOptions({"balanced"})],  # 类别权重，可以为 None、字典或字符串 "balanced"
        "verbose": ["verbose"],  # 冗余模式，字符串 "verbose"
        "random_state": ["random_state"],  # 随机数种子，字符串 "random_state"
        "max_iter": [Interval(Integral, 0, None, closed="left")],  # 最大迭代次数，整数类型，不小于0
    }

    # 初始化方法，设置分类器的参数
    def __init__(
        self,
        penalty="l2",  # 惩罚项类型，默认为 "l2"
        loss="squared_hinge",  # 损失函数类型，默认为 "squared_hinge"
        *,
        dual="auto",  # 对偶形式，默认为 "auto"
        tol=1e-4,  # 收敛容限值，默认为 0.0001
        C=1.0,  # 正则化参数，默认为 1.0
        multi_class="ovr",  # 多分类策略，默认为 "ovr"
        fit_intercept=True,  # 是否计算截距，默认为 True
        intercept_scaling=1,  # 截距缩放因子，默认为 1
        class_weight=None,  # 类别权重，默认为 None
        verbose=0,  # 冗余模式，默认为 0
        random_state=None,  # 随机数种子，默认为 None
        max_iter=1000,  # 最大迭代次数，默认为 1000
    ):
        self.dual = dual  # 设置对偶形式参数
        self.tol = tol  # 设置收敛容限值参数
        self.C = C  # 设置正则化参数
        self.multi_class = multi_class  # 设置多分类策略参数
        self.fit_intercept = fit_intercept  # 设置是否计算截距参数
        self.intercept_scaling = intercept_scaling  # 设置截距缩放因子参数
        self.class_weight = class_weight  # 设置类别权重参数
        self.verbose = verbose  # 设置冗余模式参数
        self.random_state = random_state  # 设置随机数种子参数
        self.max_iter = max_iter  # 设置最大迭代次数参数
        self.penalty = penalty  # 设置惩罚项类型参数
        self.loss = loss  # 设置损失函数类型参数

    # 装饰器，用于拟合过程的上下文处理，设置为优先跳过嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

            .. versionadded:: 0.18

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        # 验证并转换输入数据 X 和 y，确保其符合预期格式和类型
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",  # 接受稀疏矩阵格式的输入数据
            dtype=np.float64,     # 数据类型设定为浮点数
            order="C",             # 输入数据的存储顺序为 C 风格（行优先）
            accept_large_sparse=False,  # 不接受大规模稀疏矩阵
        )
        # 检查目标向量 y 是否符合分类任务的要求
        check_classification_targets(y)
        # 确定类别标签并保存到 self.classes_
        self.classes_ = np.unique(y)

        # 验证并确定 dual 参数是否有效
        _dual = _validate_dual_parameter(
            self.dual, self.loss, self.penalty, self.multi_class, X
        )

        # 调用底层函数 _fit_liblinear 进行模型拟合，获取模型参数和迭代次数
        self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
            X,
            y,
            self.C,
            self.fit_intercept,
            self.intercept_scaling,
            self.class_weight,
            self.penalty,
            _dual,
            self.verbose,
            self.max_iter,
            self.tol,
            self.random_state,
            self.multi_class,
            self.loss,
            sample_weight=sample_weight,
        )
        # 对于旧版本的兼容性，如果是 LogisticRegression，则设置结构化的 n_iter_ 属性
        # LinearSVC/R 只报告最大迭代次数的值
        self.n_iter_ = n_iter_.max().item()

        # 如果 multi_class 是 "crammer_singer" 并且类别数量为 2，则调整 coef_ 和 intercept_
        if self.multi_class == "crammer_singer" and len(self.classes_) == 2:
            self.coef_ = (self.coef_[1] - self.coef_[0]).reshape(1, -1)
            if self.fit_intercept:
                intercept = self.intercept_[1] - self.intercept_[0]
                self.intercept_ = np.array([intercept])

        # 返回模型实例本身
        return self

    def _more_tags(self):
        # 返回额外的标签信息，用于测试或文档生成等用途
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
class LinearSVR(RegressorMixin, LinearModel):
    """Linear Support Vector Regression.

    Similar to SVR with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    The main differences between :class:`~sklearn.svm.LinearSVR` and
    :class:`~sklearn.svm.SVR` lie in the loss function used by default, and in
    the handling of intercept regularization between those two implementations.

    This class supports both dense and sparse input.

    Read more in the :ref:`User Guide <svm_regression>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    epsilon : float, default=0.0
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set ``epsilon=0``.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    loss : {'epsilon_insensitive', 'squared_epsilon_insensitive'}, \
            default='epsilon_insensitive'
        Specifies the loss function. The epsilon-insensitive loss
        (standard SVR) is the L1 loss, while the squared epsilon-insensitive
        loss ('squared_epsilon_insensitive') is the L2 loss.

    fit_intercept : bool, default=True
        Whether or not to fit an intercept. If set to True, the feature vector
        is extended to include an intercept term: `[x_1, ..., x_n, 1]`, where
        1 corresponds to the intercept. If set to False, no intercept will be
        used in calculations (i.e. data is expected to be already centered).

    intercept_scaling : float, default=1.0
        When `fit_intercept` is True, the instance vector x becomes `[x_1, ...,
        x_n, intercept_scaling]`, i.e. a "synthetic" feature with a constant
        value equal to `intercept_scaling` is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight.
        Note that liblinear internally penalizes the intercept, treating it
        like any other term in the feature vector. To reduce the impact of the
        regularization on the intercept, the `intercept_scaling` parameter can
        be set to a value greater than 1; the higher the value of
        `intercept_scaling`, the lower the impact of regularization on it.
        Then, the weights become `[w_x_1, ..., w_x_n,
        w_intercept*intercept_scaling]`, where `w_x_1, ..., w_x_n` represent
        the feature weights and the intercept weight is scaled by
        `intercept_scaling`. This scaling allows the intercept term to have a
        different regularization behavior compared to the other features.
    """
    dual : "auto" or bool, default="auto"
        # 参数 `dual` 控制选择解决对偶问题还是原始问题的算法。
        # 当 `n_samples > n_features` 时，推荐设置 `dual=False`。
        # 当 `dual="auto"` 时，根据 `n_samples`、`n_features` 和 `loss` 的值自动选择参数值。
        # 如果 `n_samples` < `n_features` 并且优化器支持选择的 `loss`，则设置 `dual=True`，否则设置为 `False`.

        .. versionchanged:: 1.3
           # 在版本 1.3 中添加了 `"auto"` 选项，并且将在版本 1.5 中成为默认选项。

    verbose : int, default=0
        # 启用详细输出。注意，此设置利用 liblinear 中的每个进程运行时设置，
        # 如果启用，可能在多线程环境下工作不正常。

    random_state : int, RandomState instance or None, default=None
        # 控制伪随机数生成，用于对数据进行洗牌。
        # 传递一个整数可以实现在多次函数调用中获得可重现的输出。
        # 参见 :term:`Glossary <random_state>`。

    max_iter : int, default=1000
        # 要运行的最大迭代次数。

    Attributes
    ----------
    coef_ : ndarray of shape (n_features) if n_classes == 2 \
            else (n_classes, n_features)
        # 分配给特征的权重（原始问题中的系数）。

        `coef_` 是从 `raw_coef_` 派生的只读属性，遵循 liblinear 的内部内存布局。

    intercept_ : ndarray of shape (1) if n_classes == 2 else (n_classes)
        # 决策函数中的常数项。

    n_features_in_ : int
        # 在 `fit` 过程中观察到的特征数量。

        .. versionadded:: 0.24
           # 版本 0.24 中新增。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中观察到的特征名称。仅当 `X` 具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0
           # 版本 1.0 中新增。

    n_iter_ : int
        # 所有类别中运行的最大迭代次数。

    See Also
    --------
    LinearSVC : 使用与本类相同库的支持向量机分类器的实现（liblinear）。

    SVR : 使用 libsvm 实现的支持向量机回归：
        核函数可以是非线性的，但其 SMO 算法不像 :class:`~sklearn.svm.LinearSVR` 那样适用于大量样本。

    sklearn.linear_model.SGDRegressor : SGDRegressor 可以通过调整惩罚和损失参数来优化与 LinearSVR 相同的成本函数。
        此外，它需要更少的内存，允许增量（在线）学习，并实现各种损失函数和正则化方案。

    Examples
    --------
    >>> from sklearn.svm import LinearSVR
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_regression
    # 使用 make_regression 函数生成具有指定特征数量和随机状态的回归数据集 X, y
    >>> X, y = make_regression(n_features=4, random_state=0)
    # 创建一个管道，包含数据标准化（StandardScaler）和线性支持向量回归器（LinearSVR）
    >>> regr = make_pipeline(StandardScaler(),
    ...                      LinearSVR(random_state=0, tol=1e-5))
    # 使用管道对象 regr 对数据集 X, y 进行拟合
    >>> regr.fit(X, y)
    # 打印管道中 LinearSVR 步骤的系数
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('linearsvr', LinearSVR(random_state=0, tol=1e-05))])

    # 打印线性支持向量回归器（LinearSVR）的系数
    >>> print(regr.named_steps['linearsvr'].coef_)
    [18.582... 27.023... 44.357... 64.522...]
    # 打印线性支持向量回归器（LinearSVR）的截距
    >>> print(regr.named_steps['linearsvr'].intercept_)
    [-4...]
    # 对输入 [[0, 0, 0, 0]] 进行预测并打印预测结果
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-2.384...]

    """
    
    # 定义参数约束字典 _parameter_constraints
    _parameter_constraints: dict = {
        "epsilon": [Real],  # epsilon 参数为实数
        "tol": [Interval(Real, 0.0, None, closed="neither")],  # tol 参数为大于0的实数
        "C": [Interval(Real, 0.0, None, closed="neither")],  # C 参数为大于0的实数
        "loss": [StrOptions({"epsilon_insensitive", "squared_epsilon_insensitive"})],  # loss 参数为指定的字符串选项集合
        "fit_intercept": ["boolean"],  # fit_intercept 参数为布尔类型
        "intercept_scaling": [Interval(Real, 0, None, closed="neither")],  # intercept_scaling 参数为大于0的实数
        "dual": ["boolean", StrOptions({"auto"})],  # dual 参数为布尔类型或者字符串选项集合中的值
        "verbose": ["verbose"],  # verbose 参数为 verbose 类型
        "random_state": ["random_state"],  # random_state 参数为 random_state 类型
        "max_iter": [Interval(Integral, 0, None, closed="left")],  # max_iter 参数为大于等于0的整数
    }

    # 定义初始化方法，初始化各个参数
    def __init__(
        self,
        *,
        epsilon=0.0,
        tol=1e-4,
        C=1.0,
        loss="epsilon_insensitive",
        fit_intercept=True,
        intercept_scaling=1.0,
        dual="auto",
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        # 初始化对象的参数
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.dual = dual
        self.loss = loss

    # 应用装饰器 _fit_context，设置 prefer_skip_nested_validation 为 True
    @_fit_context(prefer_skip_nested_validation=True)
    # 将输入数据 X 和标签 y 进行验证，确保格式正确
    X, y = self._validate_data(
        X,
        y,
        accept_sparse="csr",  # 接受稀疏矩阵的格式
        dtype=np.float64,  # 数据类型为 64 位浮点数
        order="C",  # 数据存储顺序为 C 风格（行优先）
        accept_large_sparse=False,  # 不接受大型稀疏矩阵
    )
    penalty = "l2"  # 使用 l2 惩罚项，因为 SVR 只接受 l2 惩罚

    # 验证双重参数 _dual，用于支持向量机的二元分类问题
    _dual = _validate_dual_parameter(self.dual, self.loss, penalty, "ovr", X)

    # 使用 _fit_liblinear 函数拟合线性模型，返回模型系数、截距和迭代次数
    self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
        X,
        y,
        self.C,
        self.fit_intercept,
        self.intercept_scaling,
        None,
        penalty,
        _dual,
        self.verbose,
        self.max_iter,
        self.tol,
        self.random_state,
        loss=self.loss,
        epsilon=self.epsilon,
        sample_weight=sample_weight,
    )
    self.coef_ = self.coef_.ravel()  # 将系数展平为一维数组

    # 向后兼容性处理：_fit_liblinear 函数同时被 LinearSVC/R 和 LogisticRegression 使用，
    # LogisticRegression 设置结构化的 `n_iter_` 属性，提供 OvR 拟合的信息，
    # 而 LinearSVC/R 只报告最大值。
    self.n_iter_ = n_iter_.max().item()  # 设置模型的迭代次数属性为最大值

    return self  # 返回模型实例本身
# SVC 类，继承自 BaseSVC，用于 C 支持向量分类。

"""C-Support Vector Classification.

The implementation is based on libsvm. The fit time scales at least
quadratically with the number of samples and may be impractical
beyond tens of thousands of samples. For large datasets
consider using :class:`~sklearn.svm.LinearSVC` or
:class:`~sklearn.linear_model.SGDClassifier` instead, possibly after a
:class:`~sklearn.kernel_approximation.Nystroem` transformer or
other :ref:`kernel_approximation`.
"""

# 多类别支持采用一对一方案。
"""The multiclass support is handled according to a one-vs-one scheme."""

# 关于提供的核函数的数学公式和 gamma、coef0、degree 之间的关系，请参阅 narratvie 文档中的相应部分。
"""For details on the precise mathematical formulation of the provided
kernel functions and how `gamma`, `coef0` and `degree` affect each
other, see the corresponding section in the narrative documentation:
:ref:`svm_kernels`.
"""

# 若要了解如何调整 SVC 的超参数，请参阅以下示例：
# :ref:`sphx_glr_auto_examples_model_selection_plot_nested_cross_validation_iris.py`
"""To learn how to tune SVC's hyperparameters, see the following example:
:ref:`sphx_glr_auto_examples_model_selection_plot_nested_cross_validation_iris.py`
"""

# 在 :ref:`User Guide <svm_classification>` 中详细阅读更多。
"""Read more in the :ref:`User Guide <svm_classification>`.
"""

# 参数说明开始

# 正则化参数 C，默认为 1.0，正则化强度与 C 值成反比。惩罚是平方 l2 惩罚。
"""Parameters
----------
C : float, default=1.0
    Regularization parameter. The strength of the regularization is
    inversely proportional to C. Must be strictly positive. The penalty
    is a squared l2 penalty. For an intuitive visualization of the effects
    of scaling the regularization parameter C, see
    :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.
"""

# 核函数类型，可以是 {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} 或可调用函数，默认为 'rbf'。
"""kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
    default='rbf'
    Specifies the kernel type to be used in the algorithm. If
    none is given, 'rbf' will be used. If a callable is given it is used to
    pre-compute the kernel matrix from data matrices; that matrix should be
    an array of shape ``(n_samples, n_samples)``. For an intuitive
    visualization of different kernel types see
    :ref:`sphx_glr_auto_examples_svm_plot_svm_kernels.py`.
"""

# 多项式核函数的次数，仅对 'poly' 核函数有效，默认为 3。
"""degree : int, default=3
    Degree of the polynomial kernel function ('poly').
    Must be non-negative. Ignored by all other kernels.
"""

# 'rbf'、'poly' 和 'sigmoid' 核函数的核系数。
# - 如果 ``gamma='scale'``（默认），则使用 1 / (n_features * X.var()) 作为 gamma 值；
# - 如果 'auto'，则使用 1 / n_features；
# - 如果是 float 值，必须为非负数。
"""gamma : {'scale', 'auto'} or float, default='scale'
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features
    - if float, must be non-negative.
"""

# 核函数中的独立项，仅对 'poly' 和 'sigmoid' 核函数有效，默认为 0.0。
"""coef0 : float, default=0.0
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.
"""

# 是否使用收缩启发式。
"""shrinking : bool, default=True
    Whether to use the shrinking heuristic.
    See the :ref:`User Guide <shrinking_svm>`.
"""
    # probability : bool, default=False
    # 是否启用概率估计。必须在调用 `fit` 方法之前启用，因为它在内部使用5折交叉验证，可能导致方法速度变慢，
    # 而且 `predict_proba` 可能与 `predict` 不一致。详细信息请参阅用户指南中的 :ref:`scores_probabilities`。

    # tol : float, default=1e-3
    # 停止标准的容差值。

    # cache_size : float, default=200
    # 指定核缓存的大小（单位为MB）。

    # class_weight : dict or 'balanced', default=None
    # 为SVC设置类别权重参数C。如果未给定，则假定所有类别权重均为1。
    # "balanced" 模式根据输入数据中y的类别频率自动调整权重，与类频率成反比例，即 ``n_samples / (n_classes * np.bincount(y))``。

    # verbose : bool, default=False
    # 启用详细输出。请注意，此设置利用libsvm中的每个进程运行时设置，如果启用，可能在多线程环境中工作不正常。

    # max_iter : int, default=-1
    # 解算器中迭代次数的硬限制，或者设置为-1表示没有限制。

    # decision_function_shape : {'ovo', 'ovr'}, default='ovr'
    # 返回一个与所有其他分类器相同的一对多('ovr')决策函数形状 (n_samples, n_classes)，或者返回libsvm原始的一对一('ovo')决策函数形状 (n_samples, n_classes * (n_classes - 1) / 2)。
    # 然而，请注意，内部始终使用一对一('ovo')作为多类策略来训练模型；ovr矩阵仅从ovo矩阵构建而来。
    # 对于二元分类，此参数被忽略。

    # break_ties : bool, default=False
    # 如果为True，则 `decision_function_shape='ovr'`，并且类别数 > 2，则 :term:`predict` 将根据 :term:`decision_function` 的置信值来打破并列；否则将返回并列类别中的第一个类别。
    # 请注意，与简单预测相比，打破并列需要相对较高的计算成本。

    # random_state : int, RandomState instance or None, default=None
    # 控制用于概率估计数据洗牌的伪随机数生成。当 `probability` 为False时被忽略。
    # 传递一个int以在多次函数调用间获得可重复的输出。
    # 参见 :term:`Glossary <random_state>`。

    # Attributes
    # ----------
    # 每个类别的权重数组，形状为 (n_classes,)
    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C for each class.
        Computed based on the ``class_weight`` parameter.

    # 类别标签数组，形状为 (n_classes,)
    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    # 系数数组，形状为 (n_classes * (n_classes - 1) / 2, n_features)
    # 仅在线性核函数情况下可用，表示特征的权重（原始问题中的系数）。
    # `coef_` 是一个只读属性，由 `dual_coef_` 和 `support_vectors_` 推导而来。
    coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

    # 支持向量的对偶系数数组，形状为 (n_classes - 1, n_SV)
    # 在决策函数中，支持向量的对偶系数乘以它们的目标值。
    # 对于多类分类，对所有的 1-vs-1 分类器都有系数。
    # 在多类情况下，系数的布局比较复杂，请参考用户指南的多类分类部分获取详细信息。
    dual_coef_ : ndarray of shape (n_classes - 1, n_SV)
        Dual coefficients of the support vector in the decision
        function (see :ref:`sgd_mathematical_formulation`), multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the :ref:`multi-class section of the User Guide
        <svm_multi_class>` for details.

    # 拟合状态，整数值
    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    # 决策函数中的常数数组，形状为 (n_classes * (n_classes - 1) / 2,)
    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.

    # 在拟合期间观察到的特征数目
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在拟合期间观察到的特征名称数组
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 用于拟合模型的优化过程运行的迭代次数数组
    n_iter_ : ndarray of shape (n_classes * (n_classes - 1) // 2,)
        Number of iterations run by the optimization routine to fit the model.
        The shape of this attribute depends on the number of models optimized
        which in turn depends on the number of classes.

        .. versionadded:: 1.1

    # 支持向量的索引数组，形状为 (n_SV)
    support_ : ndarray of shape (n_SV)
        Indices of support vectors.

    # 支持向量的数组，形状为 (n_SV, n_features)
    # 如果使用预先计算的核，则为空数组。
    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors. An empty array if kernel is precomputed.

    # 每个类别的支持向量数目数组，形状为 (n_classes,)
    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.

    # Platt 校准的参数数组，形状为 (n_classes * (n_classes - 1) / 2)
    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
        If `probability=True`, it corresponds to the parameters learned in
        Platt scaling to produce probability estimates from decision values.
        If `probability=False`, it's an empty array. Platt scaling uses the
        logistic function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
        more information on the multiclass case and training procedure see
        section 8 of [1]_.

    # 训练向量 `X` 的维度数组
    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    See Also
    --------
    SVR : Support Vector Machine for Regression implemented using libsvm.

    LinearSVC : Scalable Linear Support Vector Machine for classification
        implemented using liblinear. Check the See Also section of
        LinearSVC for more comparison element.

    References
    ----------
    .. [1] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

    .. [2] `Platt, John (1999). "Probabilistic Outputs for Support Vector
        Machines and Comparisons to Regularized Likelihood Methods"
        <https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm import SVC
    >>> clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    >>> clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svc', SVC(gamma='auto'))])

    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    _impl = "c_svc"

    # 初始化函数，设置SVC的参数
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        # 调用父类的初始化函数，设置SVC的参数
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=0.0,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    # 返回更多的标签信息
    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
# 定义 Nu-Support Vector Classification 类，继承自 BaseSVC
class NuSVC(BaseSVC):
    """Nu-Support Vector Classification.

    Similar to SVC but uses a parameter to control the number of support
    vectors.

    The implementation is based on libsvm.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    nu : float, default=0.5
        An upper bound on the fraction of margin errors (see :ref:`User Guide
        <nu_svc>`) and a lower bound of the fraction of support vectors.
        Should be in the interval (0, 1].

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
        Specifies the kernel type to be used in the algorithm.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to precompute the kernel matrix. For an intuitive
        visualization of different kernel types see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_kernels.py`.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    class_weight : {dict, 'balanced'}, default=None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The "balanced" mode uses the values of y to automatically
        adjust weights inversely proportional to class frequencies as
        ``n_samples / (n_classes * np.bincount(y))``.

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    """
    # 决策函数的形状，可选值为 {'ovo', 'ovr'}，默认为 'ovr'
    # 对于二分类，该参数被忽略
    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
        ('ovo') is always used as multi-class strategy. The parameter is
        ignored for binary classification.

        # 0.19 版本更改：默认 decision_function_shape 现在为 'ovr'
        .. versionchanged:: 0.19
            decision_function_shape is 'ovr' by default.

        # 0.17 版本新增：推荐使用 decision_function_shape='ovr'
        .. versionadded:: 0.17
           *decision_function_shape='ovr'* is recommended.

        # 0.17 版本更改：弃用 decision_function_shape='ovo' 和 None
        .. versionchanged:: 0.17
           Deprecated *decision_function_shape='ovo' and None*.

    # break_ties : bool, default=False
    # 如果为真，且 decision_function_shape='ovr'，并且类别数大于 2，
    # predict 将根据 decision_function 的置信度值来解决并列的情况；
    # 否则将返回并列类别中的第一个类别。请注意，解决并列类别会相对于简单的预测而言
    # 需要较高的计算成本。
    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.

        # 0.22 版本新增
        .. versionadded:: 0.22

    # random_state : int, RandomState instance or None, default=None
    # 控制伪随机数生成，用于对数据进行概率估计时的数据洗牌。当 `probability` 为 False 时被忽略。
    # 设置一个整数可以保证多次函数调用时产生可重现的输出。
    # 参见 :term:`Glossary <random_state>`。
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    # Attributes
    # ----------

    # class_weight_ : ndarray of shape (n_classes,)
    # 每个类别参数 C 的乘数，根据 `class_weight` 参数计算得出。
    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C of each class.
        Computed based on the ``class_weight`` parameter.

    # classes_ : ndarray of shape (n_classes,)
    # 唯一的类别标签。
    classes_ : ndarray of shape (n_classes,)
        The unique classes labels.

    # coef_ : ndarray of shape (n_classes * (n_classes -1) / 2, n_features)
    # 特征权重（原始问题中的系数）。仅在线性核函数情况下可用。
    # `coef_` 是只读属性，从 `dual_coef_` 和 `support_vectors_` 派生而来。
    coef_ : ndarray of shape (n_classes * (n_classes -1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    # dual_coef_ : ndarray of shape (n_classes - 1, n_SV)
    # 决策函数中支持向量的对偶系数，乘以它们的目标值。
    # 对于多类分类，所有 1-vs-1 分类器的系数。
    # 在多类情况下，系数的布局略显复杂。详细信息请参见用户指南的多类分类部分。
    dual_coef_ : ndarray of shape (n_classes - 1, n_SV)
        Dual coefficients of the support vector in the decision
        function (see :ref:`sgd_mathematical_formulation`), multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the :ref:`multi-class section of the User Guide
        <svm_multi_class>` for details.

    # fit_status_ : int
    # 0 表示正确拟合，1 表示算法未收敛。
    fit_status_ : int
        0 if correctly fitted, 1 if the algorithm did not converge.

    # intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
    # 决策函数中的常数。
    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.

    # n_features_in_ : int
    # 在拟合过程中看到的特征数量。
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        # 0.24 版本新增
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 记录在训练过程中观察到的特征名称数组，仅当输入数据 `X` 的特征名全为字符串时定义。

        .. versionadded:: 1.0
        # 版本新增说明：从版本 1.0 开始添加此特性。

    n_iter_ : ndarray of shape (n_classes * (n_classes - 1) // 2,)
        # 优化过程中用于拟合模型的迭代次数数组。该属性的形状取决于优化的模型数量，
        # 这又取决于类的数量。

        .. versionadded:: 1.1
        # 版本新增说明：从版本 1.1 开始添加此特性。

    support_ : ndarray of shape (n_SV,)
        # 支持向量的索引数组。

    support_vectors_ : ndarray of shape (n_SV, n_features)
        # 支持向量的数据数组。

    n_support_ : ndarray of shape (n_classes,), dtype=int32
        # 每个类别的支持向量数量数组。

    fit_status_ : int
        # 模型拟合状态：0 表示正确拟合，1 表示算法未收敛。

    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        # 如果 `probability=True`，则这是从决策值到概率估计的 Platt 缩放参数数组。
        # 如果 `probability=False`，则为空数组。Platt 缩放使用 logistic 函数
        # ``1 / (1 + exp(decision_value * probA_ + probB_))``
        # 其中 ``probA_`` 和 ``probB_`` 是从数据集中学习到的参数 [2]_。
        # 关于多类情况和训练过程的更多信息，请参见 [1]_ 第 8 节。

    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        # 同 `probA_`，用于 Platt 缩放的第二个参数数组描述。

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        # 训练向量 `X` 的数组维度。

    See Also
    --------
    SVC : 使用 libsvm 进行分类的支持向量机。
    
    LinearSVC : 使用 liblinear 进行分类的线性可扩展支持向量机。

    References
    ----------
    .. [1] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_
        # 参考文献 [1]: 支持向量机的库 LIBSVM。

    .. [2] `Platt, John (1999). "Probabilistic Outputs for Support Vector
        Machines and Comparisons to Regularized Likelihood Methods"
        <https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393>`_
        # 参考文献 [2]: Platt 的文章，支持向量机的概率输出及其与正则化似然方法的比较。

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import NuSVC
    >>> clf = make_pipeline(StandardScaler(), NuSVC())
    >>> clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()), ('nusvc', NuSVC())])
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    _impl = "nu_svc"
        # 实现类型指示为 "nu_svc"。

    _parameter_constraints: dict = {
        **BaseSVC._parameter_constraints,
        "nu": [Interval(Real, 0.0, 1.0, closed="right")],
    }
    # 参数约束字典，继承自 BaseSVC 的参数约束，并指定了 "nu" 参数为实数区间 [0.0, 1.0]，右闭。

    _parameter_constraints.pop("C")
    # 移除参数约束字典中的 "C" 参数。
    # 定义 SVM 类的初始化方法，接收多个命名参数
    def __init__(
        self,
        *,
        nu=0.5,  # nu 参数，默认为 0.5
        kernel="rbf",  # 核函数类型，默认为 rbf
        degree=3,  # 多项式核函数的阶数，默认为 3
        gamma="scale",  # gamma 参数，默认为 "scale"
        coef0=0.0,  # 核函数中的常数项，默认为 0.0
        shrinking=True,  # 是否使用收缩启发式算法，默认为 True
        probability=False,  # 是否启用概率估计，默认为 False
        tol=1e-3,  # 允许的误差容限，默认为 1e-3
        cache_size=200,  # 内存缓存大小，默认为 200 MB
        class_weight=None,  # 类别权重，默认为 None
        verbose=False,  # 是否输出详细日志，默认为 False
        max_iter=-1,  # 最大迭代次数，默认为 -1（无限制）
        decision_function_shape="ovr",  # 决策函数形状，默认为 "ovr"
        break_ties=False,  # 在决策函数输出中是否决胜负，默认为 False
        random_state=None,  # 随机种子，默认为 None
    ):
        # 调用父类的初始化方法，传递部分参数以设置 SVM 模型
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=0.0,  # 将 C 参数设为 0.0，表示在后续设置中需指定
            nu=nu,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    # 定义一个私有方法 _more_tags，返回一个包含额外标签的字典
    def _more_tags(self):
        return {
            "_xfail_checks": {
                # 指明在进行 check_methods_subset_invariance 检查时 decision_function 方法失败的情况
                "check_methods_subset_invariance": (
                    "fails for the decision_function method"
                ),
                # 指明在进行 check_class_weight_classifiers 检查时 class_weight 参数被忽略的情况
                "check_class_weight_classifiers": "class_weight is ignored.",
                # 指明在进行 check_sample_weights_invariance 检查时零权重样本与移除样本不等效的情况
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
                # 指明在进行 check_classifiers_one_label_sample_weights 检查时指定的 nu 参数对拟合不可行的情况
                "check_classifiers_one_label_sample_weights": (
                    "specified nu is infeasible for the fit."
                ),
            }
        }
class SVR(RegressorMixin, BaseLibSVM):
    """Epsilon-Support Vector Regression.

    The free parameters in the model are C and epsilon.

    The implementation is based on libsvm. The fit time complexity
    is more than quadratic with the number of samples which makes it hard
    to scale to datasets with more than a couple of 10000 samples. For large
    datasets consider using :class:`~sklearn.svm.LinearSVR` or
    :class:`~sklearn.linear_model.SGDRegressor` instead, possibly after a
    :class:`~sklearn.kernel_approximation.Nystroem` transformer or
    other :ref:`kernel_approximation`.

    Read more in the :ref:`User Guide <svm_regression>`.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        The penalty is a squared l2. For an intuitive visualization of the
        effects of scaling the regularization parameter C, see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.

    epsilon : float, default=0.1
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value. Must be non-negative.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    ----------
    coef_ : ndarray of shape (1, n_features)
        特征的权重（原始问题中的系数），仅在线性核函数情况下可用。

        `coef_` 是只读属性，由 `dual_coef_` 和 `support_vectors_` 派生而来。

    dual_coef_ : ndarray of shape (1, n_SV)
        决策函数中支持向量的系数。

    fit_status_ : int
        模型拟合状态。0 表示正确拟合，1 表示有问题（会触发警告）。

    intercept_ : ndarray of shape (1,)
        决策函数中的常数项。

    n_features_in_ : int
        在拟合过程中观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在拟合过程中观察到的特征名称。仅当 `X` 中的特征名称全为字符串时定义。

        .. versionadded:: 1.0

    n_iter_ : int
        优化程序运行以拟合模型所需的迭代次数。

        .. versionadded:: 1.1

    n_support_ : ndarray of shape (1,), dtype=int32
        支持向量的数量。

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        训练向量 `X` 的数组维度。

    support_ : ndarray of shape (n_SV,)
        支持向量的索引。

    support_vectors_ : ndarray of shape (n_SV, n_features)
        支持向量。

    See Also
    --------
    NuSVR : 使用 libsvm 实现的支持向量机回归，使用参数控制支持向量的数量。

    LinearSVR : 使用 liblinear 实现的可扩展线性支持向量机回归。

    References
    ----------
    .. [1] `LIBSVM: 支持向量机的库 <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

    .. [2] `Platt, John (1999). "Probabilistic Outputs for Support Vector
        Machines and Comparisons to Regularized Likelihood Methods"
        <https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393>`_

    Examples
    --------
    >>> from sklearn.svm import SVR
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    >>> regr.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svr', SVR(epsilon=0.2))])
    """

    _impl = "epsilon_svr"

    _parameter_constraints: dict = {**BaseLibSVM._parameter_constraints}
    for unused_param in ["class_weight", "nu", "probability", "random_state"]:
        _parameter_constraints.pop(unused_param)
    # 初始化方法，用于设置 SVM 模型的参数
    def __init__(
        self,
        *,
        kernel="rbf",  # 设置核函数，默认为径向基函数
        degree=3,  # 多项式核函数的阶数，默认为3
        gamma="scale",  # 核函数的系数，默认为'scale'
        coef0=0.0,  # 核函数中的独立项，默认为0.0
        tol=1e-3,  # 求解器容忍的误差值，默认为1e-3
        C=1.0,  # 错误项的惩罚参数，默认为1.0
        epsilon=0.1,  # ε-SVR 中的 ε，默认为0.1
        shrinking=True,  # 是否使用启发式方法进行收缩，默认为True
        cache_size=200,  # 内核缓存的大小（MB），默认为200MB
        verbose=False,  # 是否启用详细输出，默认为False
        max_iter=-1,  # 求解器内迭代的最大次数，默认为-1表示无限制
    ):
        # 调用父类的初始化方法，设置 SVM 模型的参数
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=0.0,
            epsilon=epsilon,
            verbose=verbose,
            shrinking=shrinking,
            probability=False,
            cache_size=cache_size,
            class_weight=None,
            max_iter=max_iter,
            random_state=None,
        )
    
    # 返回一个字典，用于指定更多的标签信息
    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
class NuSVR(RegressorMixin, BaseLibSVM):
    """Nu Support Vector Regression.

    Similar to NuSVC, for regression, uses a parameter nu to control
    the number of support vectors. However, unlike NuSVC, where nu
    replaces C, here nu replaces the parameter epsilon of epsilon-SVR.

    The implementation is based on libsvm.

    Read more in the :ref:`User Guide <svm_regression>`.

    Parameters
    ----------
    nu : float, default=0.5
        An upper bound on the fraction of training errors and a lower bound of
        the fraction of support vectors. Should be in the interval (0, 1].  By
        default 0.5 will be taken.

    C : float, default=1.0
        Penalty parameter C of the error term. For an intuitive visualization
        of the effects of scaling the regularization parameter C, see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (1, n_SV)
        Coefficients of the support vector in the decision function.
    """
    # 拟合状态码，若正确拟合则为 0，否则为 1（会引发警告）
    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    # 决策函数中的常数项
    intercept_ : ndarray of shape (1,)
        Constants in decision function.

    # 拟合过程中看到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 拟合过程中看到的特征名称，仅在 `X` 全部为字符串特征名称时定义
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 优化过程中运行的迭代次数
    n_iter_ : int
        Number of iterations run by the optimization routine to fit the model.

        .. versionadded:: 1.1

    # 支持向量的数量
    n_support_ : ndarray of shape (1,), dtype=int32
        Number of support vectors.

    # 训练向量 `X` 的数组维度
    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    # 支持向量的索引
    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    # 支持向量
    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.

    # 参见
    # ----
    # NuSVC : 使用 libsvm 实现的分类支持向量机，带有控制支持向量数量的参数。
    #
    # SVR : 使用 libsvm 实现的回归的 epsilon 支持向量机。
    #
    # 引用
    # ----
    # .. [1] `LIBSVM: A Library for Support Vector Machines
    #        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_
    #
    # .. [2] `Platt, John (1999). "Probabilistic Outputs for Support Vector
    #        Machines and Comparisons to Regularized Likelihood Methods"
    #        <https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393>`_
    #
    # 示例
    # ----
    # >>> from sklearn.svm import NuSVR
    # >>> from sklearn.pipeline import make_pipeline
    # >>> from sklearn.preprocessing import StandardScaler
    # >>> import numpy as np
    # >>> n_samples, n_features = 10, 5
    # >>> np.random.seed(0)
    # >>> y = np.random.randn(n_samples)
    # >>> X = np.random.randn(n_samples, n_features)
    # >>> regr = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1))
    # >>> regr.fit(X, y)
    # Pipeline(steps=[('standardscaler', StandardScaler()),
    #                 ('nusvr', NuSVR(nu=0.1))])

    _impl = "nu_svr"

    # 参数约束字典，继承自 BaseLibSVM 的参数约束，移除了 "class_weight", "epsilon", "probability", "random_state" 参数
    _parameter_constraints: dict = {**BaseLibSVM._parameter_constraints}
    for unused_param in ["class_weight", "epsilon", "probability", "random_state"]:
        _parameter_constraints.pop(unused_param)

    # 初始化函数，设置 NuSVR 的各种参数
    def __init__(
        self,
        *,
        nu=0.5,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        tol=1e-3,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    ):
        # 调用父类初始化方法，设置 SVM 模型的参数
        super().__init__(
            kernel=kernel,               # 核函数类型
            degree=degree,               # 多项式核函数的阶数
            gamma=gamma,                 # 核函数的系数
            coef0=coef0,                 # 核函数中的独立项
            tol=tol,                     # 容忍度
            C=C,                         # 惩罚参数
            nu=nu,                       # 对于 NU-SVM 的参数
            epsilon=0.0,                 # SVR 模型中的 Epsilon
            shrinking=shrinking,         # 是否使用启发式收缩法
            probability=False,           # 是否启用概率估计
            cache_size=cache_size,       # 核矩阵缓存大小
            class_weight=None,           # 类别权重
            verbose=verbose,             # 控制详细程度
            max_iter=max_iter,           # 迭代的最大次数
            random_state=None,           # 随机数种子
        )

    def _more_tags(self):
        # 返回额外的标签信息，用于测试目的
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
# 定义一个类 OneClassSVM，它继承自 OutlierMixin 和 BaseLibSVM
class OneClassSVM(OutlierMixin, BaseLibSVM):
    """Unsupervised Outlier Detection.

    Estimate the support of a high-dimensional distribution.

    The implementation is based on libsvm.

    Read more in the :ref:`User Guide <outlier_detection>`.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.
    
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    nu : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (1, n_SV)
        Coefficients of the support vectors in the decision function.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (1,)
        Constant in the decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    """
    # 模型内部实现类型标识为 "one_class"
    _impl = "one_class"

    # 参数约束字典，继承自基类 BaseLibSVM 的参数约束，并移除了不适用于 OneClassSVM 的参数
    _parameter_constraints: dict = {**BaseLibSVM._parameter_constraints}
    for unused_param in ["C", "class_weight", "epsilon", "probability", "random_state"]:
        _parameter_constraints.pop(unused_param)

    # 初始化方法，用于设置 OneClassSVM 的各项参数
    def __init__(
        self,
        *,
        kernel="rbf",            # 核函数，默认为 RBF 核
        degree=3,                # 多项式核的阶数，默认为 3
        gamma="scale",           # 核系数，默认为 "scale"
        coef0=0.0,               # 核函数的常数项，默认为 0.0
        tol=1e-3,                # 迭代优化的容忍度，默认为 1e-3
        nu=0.5,                  # 拟合模型时的参数，默认为 0.5
        shrinking=True,          # 是否使用启发式收缩，默认为 True
        cache_size=200,          # 内核缓存大小，默认为 200MB
        verbose=False,           # 是否输出详细日志，默认为 False
        max_iter=-1,             # 最大迭代次数，默认为无限制
    ):
        # 调用父类的初始化方法，设置 OneClassSVM 的各个参数
        super().__init__(
            kernel,
            degree,
            gamma,
            coef0,
            tol,
            0.0,                   # OneClassSVM 不使用 C 参数，因此设为 0.0
            nu,
            0.0,                   # OneClassSVM 不使用 intercept_ 参数，因此设为 0.0
            shrinking,
            False,                 # OneClassSVM 不使用 probability 参数，因此设为 False
            cache_size,
            None,                  # OneClassSVM 不使用 class_weight 参数，因此设为 None
            verbose,
            max_iter,
            random_state=None,     # OneClassSVM 不使用 random_state 参数，因此设为 None
        )
    # 定义一个函数，用于在样本集 X 上检测软边界。

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        样本集，其中 `n_samples` 是样本数，`n_features` 是特征数。

    y : 被忽略
        不被使用，出于 API 的一致性而存在。

    sample_weight : array-like of shape (n_samples,), default=None
        每个样本的权重。重新缩放每个样本的 C 值。更高的权重
        会迫使分类器更加关注这些点。

    Returns
    -------
    self : object
        已拟合的估计器。

    Notes
    -----
    如果 X 不是 C 顺序连续的数组，则会进行复制。
    """
    super().fit(X, np.ones(_num_samples(X)), sample_weight=sample_weight)
    # 设置偏移量为负的截距
    self.offset_ = -self._intercept_
    return self

    # 返回拟合好的估计器。



    # 返回分离超平面的符号距离。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        数据矩阵。

    Returns
    -------
    dec : ndarray of shape (n_samples,)
        返回样本的决策函数。
    """
    dec = self._decision_function(X).ravel()
    return dec

    # 返回样本的决策函数。



    # 返回样本的原始评分函数。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        数据矩阵。

    Returns
    -------
    score_samples : ndarray of shape (n_samples,)
        返回样本的（未偏移）评分函数。
    """
    return self.decision_function(X) + self.offset_

    # 返回样本的评分函数，加上偏移量。



    # 在 X 中执行分类。

    For a one-class model, +1 or -1 is returned.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples_test, n_samples_train)
        对于 kernel="precomputed"，X 的预期形状是
        (n_samples_test, n_samples_train)。

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        X 中样本的类标签。
    """
    y = super().predict(X)
    return np.asarray(y, dtype=np.intp)

    # 返回 X 中样本的类标签。



    # 返回更多的标签信息。

    Returns
    -------
    dict
        包含额外标签信息的字典。
    """
    return {
        "_xfail_checks": {
            "check_sample_weights_invariance": (
                "zero sample_weight is not equivalent to removing samples"
            ),
        }
    }

    # 返回一个字典，其中包含有关标签的更多信息。
```