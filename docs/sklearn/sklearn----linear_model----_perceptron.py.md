# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_perceptron.py`

```
# 从 numbers 模块中导入 Real 类型
from numbers import Real

# 从 _param_validation.py 文件中导入 Interval 和 StrOptions 类
from ..utils._param_validation import Interval, StrOptions
# 从 _stochastic_gradient.py 文件中导入 BaseSGDClassifier 类
from ._stochastic_gradient import BaseSGDClassifier

# 定义 Perceptron 类，继承自 BaseSGDClassifier 类
class Perceptron(BaseSGDClassifier):
    """Linear perceptron classifier.

    The implementation is a wrapper around :class:`~sklearn.linear_model.SGDClassifier`
    by fixing the `loss` and `learning_rate` parameters as::

        SGDClassifier(loss="perceptron", learning_rate="constant")

    Other available parameters are described below and are forwarded to
    :class:`~sklearn.linear_model.SGDClassifier`.

    Read more in the :ref:`User Guide <perceptron>`.

    Parameters
    ----------

    penalty : {'l2','l1','elasticnet'}, default=None
        The penalty (aka regularization term) to be used.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term if regularization is
        used.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with `0 <= l1_ratio <= 1`.
        `l1_ratio=0` corresponds to L2 penalty, `l1_ratio=1` to L1.
        Only used if `penalty='elasticnet'`.

        .. versionadded:: 0.24

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    eta0 : float, default=1
        Constant by which the updates are multiplied.

    n_jobs : int, default=None
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, default=0
        Used to shuffle the training data, when ``shuffle`` is set to
        ``True``. Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.
    """
    # Perceptron 类的构造函数，继承自 BaseSGDClassifier
    def __init__(
        self,
        penalty=None,
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        eta0=1.0,
        n_jobs=None,
        random_state=0,
    ):
        # 调用父类 BaseSGDClassifier 的构造函数，传递相应参数
        super().__init__(
            loss="perceptron",  # 设置 SGDClassifier 的 loss 参数为 'perceptron'
            learning_rate="constant",  # 设置 SGDClassifier 的 learning_rate 参数为 'constant'
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            eta0=eta0,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        # Perceptron 类的初始化函数没有额外的代码
    # 是否使用早停策略来在验证分数不再提升时终止训练。
    # 如果设置为 True，则会自动将一部分分层训练数据作为验证集，
    # 并在连续 `n_iter_no_change` 轮中验证分数未至少 `tol` 提升时终止训练。
    early_stopping : bool, default=False

        # 用作早停验证集的训练数据的比例。
        # 必须在 0 到 1 之间。
        # 仅在 early_stopping 设置为 True 时使用。
        validation_fraction : float, default=0.1

        # 在早停之前等待的连续没有改进的迭代次数。
        n_iter_no_change : int, default=5

    # 类别权重的预设值。可以是字典 {类别标签: 权重} 或 "balanced"。
    # "balanced" 模式根据输入数据中类别频率的倒数自动调整权重。
    class_weight : dict, {class_label: weight} or "balanced", default=None

    # 是否启用热启动。如果设置为 True，则重用前一次拟合的解作为初始化。
    # 否则，只是擦除前一个解。
    warm_start : bool, default=False

    # 唯一类别标签的 ndarray。
    classes_ : ndarray of shape (n_classes,)

    # 特征权重数组。如果 n_classes == 2，则形状为 (1, n_features)；否则为 (n_classes, n_features)。
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 else (n_classes, n_features)

    # 决策函数中的常数。如果 n_classes == 2，则形状为 (1,)；否则为 (n_classes,)。
    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)

    # 在拟合期间看到的特征数量。
    n_features_in_ : int

        # 在拟合期间看到的特征名称。仅当 `X` 具有全部为字符串的特征名称时定义。
        feature_names_in_ : ndarray of shape (`n_features_in_`,)

    # 达到停止条件时的实际迭代次数。对于多类别拟合，是每个二元拟合的最大值。
    n_iter_ : int

    # 训练期间执行的权重更新次数。
    t_ : int

    # 相关联的 sklearn.linear_model.SGDClassifier：使用 SGD 训练的线性分类器（如 SVM、逻辑回归等）。
    See Also
    --------
    sklearn.linear_model.SGDClassifier : Linear classifiers
        (SVM, logistic regression, etc.) with SGD training.

    # "Perceptron" 是一个分类算法，与 "SGDClassifier" 共享相同的实现。
    # 实际上，"Perceptron()" 等效于 `SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)`。

    Notes
    -----
    # "Perceptron" 是一种分类算法，与 "SGDClassifier" 共享相同的底层实现。
    References
    ----------
    """
    https://en.wikipedia.org/wiki/Perceptron and references therein.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import Perceptron
    >>> X, y = load_digits(return_X_y=True)
    >>> clf = Perceptron(tol=1e-3, random_state=0)
    >>> clf.fit(X, y)
    Perceptron()
    >>> clf.score(X, y)
    0.939...
    """

    # 定义一个字典，包含参数的约束条件，继承自BaseSGDClassifier的参数约束，并去除'loss'和'average'键
    _parameter_constraints: dict = {**BaseSGDClassifier._parameter_constraints}
    _parameter_constraints.pop("loss")
    _parameter_constraints.pop("average")
    # 更新字典，添加新的参数约束条件
    _parameter_constraints.update(
        {
            "penalty": [StrOptions({"l2", "l1", "elasticnet"}), None],  # 罚项的取值范围约束
            "alpha": [Interval(Real, 0, None, closed="left")],  # alpha参数的取值范围约束
            "l1_ratio": [Interval(Real, 0, 1, closed="both")],  # l1_ratio参数的取值范围约束
            "eta0": [Interval(Real, 0, None, closed="left")],  # eta0参数的取值范围约束
        }
    )

    # 初始化方法，设置Perceptron分类器的各种参数
    def __init__(
        self,
        *,
        penalty=None,  # 惩罚项，默认为None
        alpha=0.0001,  # 正则化项的系数，默认为0.0001
        l1_ratio=0.15,  # l1正则化的比例，默认为0.15
        fit_intercept=True,  # 是否计算截距，默认为True
        max_iter=1000,  # 最大迭代次数，默认为1000
        tol=1e-3,  # 迭代收敛的阈值，默认为1e-3
        shuffle=True,  # 是否在每次迭代时打乱数据，默认为True
        verbose=0,  # 控制详细程度的标志，默认为0，即不输出额外信息
        eta0=1.0,  # 学习率，默认为1.0
        n_jobs=None,  # 并行工作的数量，默认为None，即不使用并行
        random_state=0,  # 随机数种子，默认为0
        early_stopping=False,  # 是否启用早停，默认为False
        validation_fraction=0.1,  # 用作早停的验证集比例，默认为0.1
        n_iter_no_change=5,  # 连续多少次迭代结果不变时停止训练，默认为5
        class_weight=None,  # 类别权重，默认为None，即所有类别权重相同
        warm_start=False,  # 是否重用上一次调用的解，默认为False
    ):
        # 调用父类BaseSGDClassifier的初始化方法，设置损失函数为'perceptron'，并传入其他参数
        super().__init__(
            loss="perceptron",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            learning_rate="constant",
            eta0=eta0,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            power_t=0.5,
            warm_start=warm_start,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
```