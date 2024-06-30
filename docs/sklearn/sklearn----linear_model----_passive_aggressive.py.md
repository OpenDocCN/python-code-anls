# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_passive_aggressive.py`

```
# 从 numbers 模块导入 Real 类型
from numbers import Real

# 导入 _fit_context 模块
from ..base import _fit_context

# 从 utils._param_validation 模块导入 Interval 和 StrOptions 类
from ..utils._param_validation import Interval, StrOptions

# 从 _stochastic_gradient 模块导入 DEFAULT_EPSILON, BaseSGDClassifier, BaseSGDRegressor 类
from ._stochastic_gradient import DEFAULT_EPSILON, BaseSGDClassifier, BaseSGDRegressor

# PassiveAggressiveClassifier 类，继承自 BaseSGDClassifier
class PassiveAggressiveClassifier(BaseSGDClassifier):
    """Passive Aggressive Classifier.

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------
    C : float, default=1.0
        Maximum step size (regularization). Defaults to 1.0.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`~sklearn.linear_model.PassiveAggressiveClassifier.partial_fit` method.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a stratified fraction of training data as validation and terminate
        training when validation score is not improving by at least `tol` for
        `n_iter_no_change` consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    loss : str, default="hinge"
        The loss function to be used:
        hinge: equivalent to PA-I in the reference paper.
        squared_hinge: equivalent to PA-II in the reference paper.

    n_jobs : int or None, default=None
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Used to shuffle the training data, when ``shuffle`` is set to
        ``True``. Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.
    """
    warm_start : bool, default=False
        当设置为 True 时，使用上一次调用 fit 的解作为初始化，否则会清除上一次的解。
        参见：术语表中的“热启动”。

    class_weight : dict, {class_label: weight} or "balanced" or None, \
            default=None
        fit 参数 class_weight 的预设值。

        类别权重。如果未给出，则假定所有类别权重为一。

        "balanced" 模式根据 y 的值自动调整权重，与输入数据中类别频率成反比，
        计算方式为 ``n_samples / (n_classes * np.bincount(y))``。

        .. versionadded:: 0.17
           参数 *class_weight* 自动加权样本。

    average : bool or int, default=False
        当设置为 True 时，计算平均的 SGD 权重并存储在 ``coef_`` 属性中。如果设置为大于1的整数，
        则在看到足够数量的样本后（即 average=10），开始计算平均值。

        .. versionadded:: 0.19
           参数 *average* 用于在 SGD 中使用权重平均。

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 else \
            (n_classes, n_features)
        分配给特征的权重。

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        决策函数中的常数项。

    n_features_in_ : int
        在拟合期间看到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在拟合期间看到的特征名称。仅在 `X` 具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    n_iter_ : int
        达到停止准则的实际迭代次数。对于多类拟合，它是每个二元拟合中的最大值。

    classes_ : ndarray of shape (n_classes,)
        唯一的类别标签。

    t_ : int
        训练过程中执行的权重更新次数。与 ``(n_iter_ * n_samples + 1)`` 相同。

    See Also
    --------
    SGDClassifier : 增量训练的逻辑回归。
    Perceptron : 线性感知器分类器。

    References
    ----------
    在线被动-侵略性算法
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)

    Examples
    --------
    >>> from sklearn.linear_model import PassiveAggressiveClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=0)
    >>> clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,
    ... tol=1e-3)
    # 创建一个PassiveAggressiveClassifier分类器对象，设置最大迭代次数为1000，随机状态为0，收敛阈值为1e-3

    >>> clf.fit(X, y)
    # 使用分类器对象clf对输入的特征矩阵X和目标向量y进行拟合训练

    PassiveAggressiveClassifier(random_state=0)
    # 返回一个已配置好参数的PassiveAggressiveClassifier分类器对象

    >>> print(clf.coef_)
    [[0.26642044 0.45070924 0.67251877 0.64185414]]
    # 打印已训练分类器clf的权重系数数组

    >>> print(clf.intercept_)
    [1.84127814]
    # 打印已训练分类器clf的截距项

    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    # 使用已训练分类器clf预测输入特征向量[[0, 0, 0, 0]]的类别，并打印预测结果

    """

    _parameter_constraints: dict = {
        **BaseSGDClassifier._parameter_constraints,
        "loss": [StrOptions({"hinge", "squared_hinge"})],
        "C": [Interval(Real, 0, None, closed="right")],
    }
    # 定义了一个字典_parameter_constraints，继承了BaseSGDClassifier的参数约束，包括损失函数loss和正则化参数C的取值范围约束

    def __init__(
        self,
        *,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="hinge",
        n_jobs=None,
        random_state=None,
        warm_start=False,
        class_weight=None,
        average=False,
    ):
        # 初始化方法，设置分类器的各种参数
        super().__init__(
            penalty=None,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            eta0=1.0,
            warm_start=warm_start,
            class_weight=class_weight,
            average=average,
            n_jobs=n_jobs,
        )

        self.C = C  # 设置正则化参数C
        self.loss = loss  # 设置损失函数类型

    @_fit_context(prefer_skip_nested_validation=True)
    # 应用装饰器_fit_context，用于设置拟合上下文环境，优先跳过嵌套验证
    # 使用 Passive Aggressive 算法拟合线性模型。

    # 模型参数：
    # X: 形状为 (n_samples, n_features) 的训练数据子集，可以是数组或稀疏矩阵。
    # y: 形状为 (n_samples,) 的目标值子集。
    # classes: 形状为 (n_classes,) 的数组，跨所有 partial_fit 调用的类。
    #         可以通过 `np.unique(y_all)` 获得，其中 y_all 是整个数据集的目标向量。
    #         在第一次调用 partial_fit 时需要此参数，后续调用可以省略。
    #         注意 y 不需要包含 `classes` 中的所有标签。

    def partial_fit(self, X, y, classes=None):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Subset of the training data.

        y : array-like of shape (n_samples,)
            Subset of the target values.

        classes : ndarray of shape (n_classes,)
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # 如果对象没有 `classes_` 属性，则进行更多的参数验证
        if not hasattr(self, "classes_"):
            self._more_validate_params(for_partial_fit=True)

            # 如果 class_weight 为 "balanced"，则抛出 ValueError
            if self.class_weight == "balanced":
                raise ValueError(
                    "class_weight 'balanced' is not supported for "
                    "partial_fit. For 'balanced' weights, use "
                    "`sklearn.utils.compute_class_weight` with "
                    "`class_weight='balanced'`. In place of y you "
                    "can use a large enough subset of the full "
                    "training set target to properly estimate the "
                    "class frequency distributions. Pass the "
                    "resulting weights as the class_weight "
                    "parameter."
                )

        # 根据损失函数类型选择学习率类型
        lr = "pa1" if self.loss == "hinge" else "pa2"
        
        # 调用内部方法 _partial_fit 进行部分拟合
        return self._partial_fit(
            X,
            y,
            alpha=1.0,
            C=self.C,
            loss="hinge",
            learning_rate=lr,
            max_iter=1,
            classes=classes,
            sample_weight=None,
            coef_init=None,
            intercept_init=None,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    # 使用 Passive Aggressive 算法拟合线性模型。

    # 调用内部方法，进一步验证参数设置的有效性。
    self._more_validate_params()

    # 根据损失函数类型确定学习率类型，如果损失函数是"hinge"，则使用"pa1"，否则使用"pa2"。
    lr = "pa1" if self.loss == "hinge" else "pa2"

    # 调用内部方法进行模型拟合，传入参数包括输入数据 X 和目标值 y，以及其他初始化参数如 alpha, C, loss, learning_rate, coef_init 和 intercept_init。
    return self._fit(
        X,
        y,
        alpha=1.0,
        C=self.C,
        loss="hinge",
        learning_rate=lr,
        coef_init=coef_init,
        intercept_init=intercept_init,
    )
class PassiveAggressiveRegressor(BaseSGDRegressor):
    """Passive Aggressive Regressor.

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------

    C : float, default=1.0
        Maximum step size (regularization). Defaults to 1.0.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`~sklearn.linear_model.PassiveAggressiveRegressor.partial_fit` method.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    loss : str, default="epsilon_insensitive"
        The loss function to be used:
        epsilon_insensitive: equivalent to PA-I in the reference paper.
        squared_epsilon_insensitive: equivalent to PA-II in the reference
        paper.

    epsilon : float, default=0.1
        If the difference between the current prediction and the correct label
        is below this threshold, the model is not updated.

    random_state : int, RandomState instance, default=None
        Used to shuffle the training data, when ``shuffle`` is set to
        ``True``. Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.
    _parameter_constraints: dict = {
        **BaseSGDRegressor._parameter_constraints,
        "loss": [StrOptions({"epsilon_insensitive", "squared_epsilon_insensitive"})],
        "C": [Interval(Real, 0, None, closed="right")],
        "epsilon": [Interval(Real, 0, None, closed="left")],
    }



    # 定义参数约束字典，继承自 BaseSGDRegressor 的参数约束
    _parameter_constraints: dict = {
        **BaseSGDRegressor._parameter_constraints,
        # 损失函数可选项，包括 "epsilon_insensitive" 和 "squared_epsilon_insensitive"
        "loss": [StrOptions({"epsilon_insensitive", "squared_epsilon_insensitive"})],
        # 正则化参数 C 的约束为大于等于0的实数
        "C": [Interval(Real, 0, None, closed="right")],
        # 不敏感度参数 epsilon 的约束为大于0的实数
        "epsilon": [Interval(Real, 0, None, closed="left")],
    }


这段代码定义了一个参数约束字典 `_parameter_constraints`，用于控制 `PassiveAggressiveRegressor` 类中的参数的合法取值范围和选项。
    def __init__(
        self,
        *,
        C=1.0,  # 初始化函数，设置Passive Aggressive算法的参数，C为惩罚参数，默认为1.0
        fit_intercept=True,  # 是否拟合截距，默认为True
        max_iter=1000,  # 最大迭代次数，默认为1000
        tol=1e-3,  # 收敛判据，默认为0.001
        early_stopping=False,  # 是否启用早停，默认为False
        validation_fraction=0.1,  # 用于早停的验证集比例，默认为0.1
        n_iter_no_change=5,  # 连续多少次迭代效果未改善时停止迭代，默认为5
        shuffle=True,  # 每次迭代是否打乱数据，默认为True
        verbose=0,  # 控制详细程度的参数，默认为0（不输出信息）
        loss="epsilon_insensitive",  # 损失函数类型，默认为epsilon不敏感损失
        epsilon=DEFAULT_EPSILON,  # epsilon不敏感损失的参数，默认值由DEFAULT_EPSILON给定
        random_state=None,  # 随机数种子，用于可重复性，默认为None
        warm_start=False,  # 是否热启动，即接着上次训练结果继续训练，默认为False
        average=False,  # 是否计算平均模型，默认为False
    ):
        super().__init__(
            penalty=None,  # 不使用正则化惩罚项
            l1_ratio=0,  # L1正则化系数，不适用于Passive Aggressive算法，设为0
            epsilon=epsilon,  # epsilon不敏感损失的参数
            eta0=1.0,  # 初始学习率，设为1.0
            fit_intercept=fit_intercept,  # 是否拟合截距
            max_iter=max_iter,  # 最大迭代次数
            tol=tol,  # 收敛判据
            early_stopping=early_stopping,  # 是否启用早停
            validation_fraction=validation_fraction,  # 早停所用的验证集比例
            n_iter_no_change=n_iter_no_change,  # 连续多少次迭代效果未改善时停止迭代
            shuffle=shuffle,  # 每次迭代是否打乱数据
            verbose=verbose,  # 控制详细程度的参数
            random_state=random_state,  # 随机数种子
            warm_start=warm_start,  # 是否热启动
            average=average,  # 是否计算平均模型
        )
        self.C = C  # 设置类成员变量C，即惩罚参数
        self.loss = loss  # 设置类成员变量loss，即损失函数类型

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Subset of training data.

        y : numpy array of shape [n_samples]
            Subset of target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not hasattr(self, "coef_"):
            self._more_validate_params(for_partial_fit=True)

        lr = "pa1" if self.loss == "epsilon_insensitive" else "pa2"  # 根据损失函数类型选择学习率类型
        return self._partial_fit(
            X,
            y,
            alpha=1.0,  # 学习率，固定为1.0
            C=self.C,  # 惩罚参数
            loss="epsilon_insensitive",  # 损失函数类型，固定为epsilon不敏感损失
            learning_rate=lr,  # 学习率类型
            max_iter=1,  # 每次迭代的最大次数，这里是1
            sample_weight=None,
            coef_init=None,
            intercept_init=None,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, coef_init=None, intercept_init=None):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : numpy array of shape [n_samples]
            Target values.

        coef_init : array, shape = [n_features]
            The initial coefficients to warm-start the optimization.

        intercept_init : array, shape = [1]
            The initial intercept to warm-start the optimization.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 调用内部方法进行参数验证和设置
        self._more_validate_params()

        # 根据损失函数类型选择合适的学习率策略
        lr = "pa1" if self.loss == "epsilon_insensitive" else "pa2"
        
        # 调用内部方法进行模型拟合
        return self._fit(
            X,
            y,
            alpha=1.0,
            C=self.C,
            loss="epsilon_insensitive",
            learning_rate=lr,
            coef_init=coef_init,
            intercept_init=intercept_init,
        )
```