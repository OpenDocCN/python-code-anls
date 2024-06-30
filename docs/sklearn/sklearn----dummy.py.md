# `D:\src\scipysrc\scikit-learn\sklearn\dummy.py`

```
"""Dummy estimators that implement simple rules of thumb."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings
# 导入数值相关的类
from numbers import Integral, Real

# 导入科学计算相关库
import numpy as np
import scipy.sparse as sp

# 从当前包中导入以下模块和类
from .base import (
    BaseEstimator,               # 导入基本估算器类
    ClassifierMixin,             # 导入分类器混合类
    MultiOutputMixin,            # 导入多输出混合类
    RegressorMixin,              # 导入回归器混合类
    _fit_context,                # 导入私有函数_fit_context
)
# 从当前包中导入检查随机状态的函数
from .utils import check_random_state
# 从当前包中导入参数验证相关模块
from .utils._param_validation import Interval, StrOptions
# 从当前包中导入类分布函数
from .utils.multiclass import class_distribution
# 从当前包中导入稀疏矩阵随机选择函数
from .utils.random import _random_choice_csc
# 从当前包中导入加权百分位数计算函数
from .utils.stats import _weighted_percentile
# 从当前包中导入数据验证相关函数
from .utils.validation import (
    _check_sample_weight,        # 导入私有函数_check_sample_weight
    _num_samples,                # 导入私有函数_num_samples
    check_array,                 # 导入检查数组函数
    check_consistent_length,     # 导入检查一致长度函数
    check_is_fitted,             # 导入检查是否拟合函数
)


class DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):
    """DummyClassifier makes predictions that ignore the input features.

    This classifier serves as a simple baseline to compare against other more
    complex classifiers.

    The specific behavior of the baseline is selected with the `strategy`
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`. The predictions, however,
    typically depend on values observed in the `y` parameter passed to `fit`.

    Note that the "stratified" and "uniform" strategies lead to
    non-deterministic predictions that can be rendered deterministic by setting
    the `random_state` parameter if needed. The other strategies are naturally
    deterministic and, once fit, always return the same constant prediction
    for any value of `X`.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    .. versionadded:: 0.13

    Parameters
    ----------

"""
    strategy : {"most_frequent", "prior", "stratified", "uniform", \
            "constant"}, default="prior"
        # 预测策略，用于生成预测结果的方法。
        # 可选值包括："most_frequent", "prior", "stratified", "uniform", "constant"
        # 默认为 "prior"。

        Strategy to use to generate predictions.

        * "most_frequent": the `predict` method always returns the most
          frequent class label in the observed `y` argument passed to `fit`.
          The `predict_proba` method returns the matching one-hot encoded
          vector.
          # 使用出现最频繁的类标签作为预测结果。`predict_proba` 方法返回匹配的独热编码向量。

        * "prior": the `predict` method always returns the most frequent
          class label in the observed `y` argument passed to `fit` (like
          "most_frequent"). ``predict_proba`` always returns the empirical
          class distribution of `y` also known as the empirical class prior
          distribution.
          # 使用训练集 `y` 中出现最频繁的类标签作为预测结果。`predict_proba` 方法返回 `y` 的经验类分布。

        * "stratified": the `predict_proba` method randomly samples one-hot
          vectors from a multinomial distribution parametrized by the empirical
          class prior probabilities.
          The `predict` method returns the class label which got probability
          one in the one-hot vector of `predict_proba`.
          Each sampled row of both methods is therefore independent and
          identically distributed.
          # `predict_proba` 方法从由经验类先验概率参数化的多项分布中随机抽样独热向量。
          # `predict` 方法返回在 `predict_proba` 的独热向量中概率为1的类标签。
          # 因此，每个方法抽样的行是独立且同分布的。

        * "uniform": generates predictions uniformly at random from the list
          of unique classes observed in `y`, i.e. each class has equal
          probability.
          # 均匀地随机生成预测结果，从 `y` 中观察到的唯一类列表中，每个类具有相等的概率。

        * "constant": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class.
          # 始终预测由用户提供的常数标签。对于评估非多数类的度量非常有用。

          .. versionchanged:: 0.24
             The default value of `strategy` has changed to "prior" in version
             0.24.

    random_state : int, RandomState instance or None, default=None
        # 控制生成预测时的随机性，当 `strategy='stratified'` 或 `strategy='uniform'` 时使用。
        # 可传入一个整数以便于在多次函数调用中获得可重复的输出。
        # 参见 :term:`Glossary <random_state>`。

        Controls the randomness to generate the predictions when
        ``strategy='stratified'`` or ``strategy='uniform'``.

    constant : int or str or array-like of shape (n_outputs,), default=None
        # "constant" 策略下的显式常数预测值。此参数仅对 "constant" 策略有效。

        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of such arrays
        # 在 `y` 中观察到的唯一类标签。对于多输出分类问题，此属性是一个数组列表，因为每个输出有一个独立的可能类集。

        Unique class labels observed in `y`. For multi-output classification
        problems, this attribute is a list of arrays as each output has an
        independent set of possible classes.

    n_classes_ : int or list of int
        # 每个输出的标签数量。

        Number of label for each output.

    class_prior_ : ndarray of shape (n_classes,) or list of such arrays
        # 在 `y` 中观察到的每个类别的频率。对于多输出分类问题，这是每个输出独立计算的。

        Frequency of each class observed in `y`. For multioutput classification
        problems, this is computed independently for each output.

    n_features_in_ : int
        # 在 `fit` 过程中看到的特征数量。

        Number of features seen during :term:`fit`.
    # `feature_names_in_`：形状为 (`n_features_in_`,) 的 ndarray
    # 在 `fit` 过程中出现的特征名称。仅当 `X` 的特征名称都是字符串时才定义。

    # `n_outputs_`：整数类型
    # 输出的数量。

    # `sparse_output_`：布尔类型
    # 如果预测返回的数组以稀疏的 CSC 格式呈现，则为 True。
    # 如果输入的 `y` 是稀疏格式，这个值会自动设为 True。

    # See Also
    # --------
    # DummyRegressor : 使用简单规则进行预测的回归器。

    # Examples
    # --------
    # >>> import numpy as np
    # >>> from sklearn.dummy import DummyClassifier
    # >>> X = np.array([-1, 1, 1, 1])
    # >>> y = np.array([0, 1, 1, 1])
    # >>> dummy_clf = DummyClassifier(strategy="most_frequent")
    # >>> dummy_clf.fit(X, y)
    # DummyClassifier(strategy='most_frequent')
    # >>> dummy_clf.predict(X)
    # array([1, 1, 1, 1])
    # >>> dummy_clf.score(X, y)
    # 0.75

    # `_parameter_constraints`：字典类型
    # 包含参数约束信息的字典，指定了每个参数的可接受取值范围或类型。

    # `__init__` 方法
    # 构造函数，初始化 DummyClassifier 的实例。
    def __init__(self, *, strategy="prior", random_state=None, constant=None):
        # 设置策略
        self.strategy = strategy
        # 设置随机状态
        self.random_state = random_state
        # 设置常数值
        self.constant = constant

    # `_fit_context` 装饰器
    # 在进行拟合操作时的上下文装饰器，支持跳过嵌套验证。
    @_fit_context(prefer_skip_nested_validation=True)
    def predict(self, X):
        """对测试向量 X 进行分类。

        参数
        ----------
        X : 形状为 (n_samples, n_features) 的数组
            测试数据。

        返回
        -------
        y : 形状为 (n_samples,) 或 (n_samples, n_outputs) 的数组
            X 的预测目标值。
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # numpy 的 random_state 期望 Python int 而不是 long 作为 size 参数在 Windows 下
        n_samples = _num_samples(X)
        rs = check_random_state(self.random_state)

        n_classes_ = self.n_classes_
        classes_ = self.classes_
        class_prior_ = self.class_prior_
        constant = self.constant
        if self.n_outputs_ == 1:
            # 即使 self.n_outputs_ == 1 也要获取相同类型
            n_classes_ = [n_classes_]
            classes_ = [classes_]
            class_prior_ = [class_prior_]
            constant = [constant]
        # 仅计算概率一次
        if self._strategy == "stratified":
            proba = self.predict_proba(X)
            if self.n_outputs_ == 1:
                proba = [proba]

        if self.sparse_output_:
            class_prob = None
            if self._strategy in ("most_frequent", "prior"):
                classes_ = [np.array([cp.argmax()]) for cp in class_prior_]

            elif self._strategy == "stratified":
                class_prob = class_prior_

            elif self._strategy == "uniform":
                raise ValueError(
                    "不支持使用 uniform 策略进行稀疏目标预测"
                )

            elif self._strategy == "constant":
                classes_ = [np.array([c]) for c in constant]

            y = _random_choice_csc(n_samples, classes_, class_prob, self.random_state)
        else:
            if self._strategy in ("most_frequent", "prior"):
                y = np.tile(
                    [
                        classes_[k][class_prior_[k].argmax()]
                        for k in range(self.n_outputs_)
                    ],
                    [n_samples, 1],
                )

            elif self._strategy == "stratified":
                y = np.vstack(
                    [
                        classes_[k][proba[k].argmax(axis=1)]
                        for k in range(self.n_outputs_)
                    ]
                ).T

            elif self._strategy == "uniform":
                ret = [
                    classes_[k][rs.randint(n_classes_[k], size=n_samples)]
                    for k in range(self.n_outputs_)
                ]
                y = np.vstack(ret).T

            elif self._strategy == "constant":
                y = np.tile(self.constant, (n_samples, 1))

            if self.n_outputs_ == 1:
                y = np.ravel(y)

        return y
    # 返回测试向量 X 的概率估计值。

    # 检查模型是否已拟合
    check_is_fitted(self)

    # 获取样本数目
    n_samples = _num_samples(X)

    # 根据 random_state 初始化随机状态对象
    rs = check_random_state(self.random_state)

    # 获取类别数目、类别列表、类先验概率和常数值
    n_classes_ = self.n_classes_
    classes_ = self.classes_
    class_prior_ = self.class_prior_
    constant = self.constant

    # 处理单输出情况下的变量类型一致性
    if self.n_outputs_ == 1:
        n_classes_ = [n_classes_]
        classes_ = [classes_]
        class_prior_ = [class_prior_]
        constant = [constant]

    # 初始化概率结果列表 P
    P = []

    # 对每个输出维度进行概率预测
    for k in range(self.n_outputs_):
        if self._strategy == "most_frequent":
            # 使用类先验概率中最大值的索引作为预测值
            ind = class_prior_[k].argmax()
            out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
            out[:, ind] = 1.0
        elif self._strategy == "prior":
            # 将每个样本的概率置为类先验概率
            out = np.ones((n_samples, 1)) * class_prior_[k]
        elif self._strategy == "stratified":
            # 使用多项分布生成样本的概率
            out = rs.multinomial(1, class_prior_[k], size=n_samples)
            out = out.astype(np.float64)
        elif self._strategy == "uniform":
            # 将每个样本的概率均匀分布
            out = np.ones((n_samples, n_classes_[k]), dtype=np.float64)
            out /= n_classes_[k]
        elif self._strategy == "constant":
            # 使用常数值作为预测类别的索引
            ind = np.where(classes_[k] == constant[k])
            out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
            out[:, ind] = 1.0

        # 将当前维度的预测结果添加到概率列表 P 中
        P.append(out)

    # 如果只有一个输出维度，则返回单个数组而不是列表
    if self.n_outputs_ == 1:
        P = P[0]

    # 返回最终的概率预测结果 P
    return P


    # 返回测试向量 X 的对数概率估计值。

    # 使用 predict_proba 方法获取概率估计值
    proba = self.predict_proba(X)

    # 如果只有一个输出维度，则返回对数概率的单个数组
    if self.n_outputs_ == 1:
        return np.log(proba)
    else:
        # 对多个输出维度的每个数组计算对数概率，并返回结果列表
        return [np.log(p) for p in proba]
    # 返回一个字典，包含额外的标签信息，指示此分类器的性能特征
    def _more_tags(self):
        return {
            "poor_score": True,  # 指示此分类器在某些评估指标上可能表现较差
            "no_validation": True,  # 指示此分类器不需要进行额外的验证步骤
            "_xfail_checks": {  # 一个字典，包含某些检查点失败的相关信息
                "check_methods_subset_invariance": "fails for the predict method",  # 指出某个检查点在预测方法中失败
                "check_methods_sample_order_invariance": "fails for the predict method",  # 指出某个检查点在预测方法中失败
            },
        }

    # 计算给定测试数据和标签的平均准确率
    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : None or array-like of shape (n_samples, n_features)
            Test samples. Passing None as test samples gives the same result
            as passing real test samples, since DummyClassifier
            operates independently of the sampled observations.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))  # 如果测试数据为None，则创建一个零数组作为测试数据
        return super().score(X, y, sample_weight)  # 调用父类的score方法计算预测准确率
# 定义一个名为 DummyRegressor 的类，它继承自 MultiOutputMixin、RegressorMixin 和 BaseEstimator
class DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Regressor that makes predictions using simple rules.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.

    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : ndarray of shape (1, n_outputs)
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X` has
        feature names that are all strings.

    n_outputs_ : int
        Number of outputs.

    See Also
    --------
    DummyClassifier: Classifier that makes predictions using simple rules.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyRegressor
    >>> X = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([2.0, 3.0, 5.0, 10.0])
    >>> dummy_regr = DummyRegressor(strategy="mean")
    >>> dummy_regr.fit(X, y)
    DummyRegressor()
    >>> dummy_regr.predict(X)
    array([5., 5., 5., 5.])
    >>> dummy_regr.score(X, y)
    0.0
    """

    # 定义参数约束字典，指定了各个参数的可选值范围或类型
    _parameter_constraints: dict = {
        "strategy": [StrOptions({"mean", "median", "quantile", "constant"})],
        "quantile": [Interval(Real, 0.0, 1.0, closed="both"), None],
        "constant": [
            Interval(Real, None, None, closed="neither"),
            "array-like",
            None,
        ],
    }

    # 初始化方法，接收 strategy、constant 和 quantile 三个参数
    def __init__(self, *, strategy="mean", constant=None, quantile=None):
        # 将传入的参数值分别赋给对象的属性
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    # 使用装饰器 @_fit_context(prefer_skip_nested_validation=True)，但未提供完整代码
    def fit(self, X, y, sample_weight=None):
        """Fit the random regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate the input data X
        self._validate_data(X, cast_to_ndarray=False)

        # Check and reshape the target array y if necessary
        y = check_array(y, ensure_2d=False, input_name="y")
        if len(y) == 0:
            raise ValueError("y must not be empty.")

        # Ensure y is reshaped to (n_samples, 1) if it's 1-dimensional
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]

        # Check consistency of lengths between X, y, and sample_weight
        check_consistent_length(X, y, sample_weight)

        # Check and adjust sample weights if provided
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Fit the estimator based on the chosen strategy
        if self.strategy == "mean":
            # Compute the weighted mean of y
            self.constant_ = np.average(y, axis=0, weights=sample_weight)

        elif self.strategy == "median":
            # Compute the median of y, considering sample weights if provided
            if sample_weight is None:
                self.constant_ = np.median(y, axis=0)
            else:
                self.constant_ = [
                    _weighted_percentile(y[:, k], sample_weight, percentile=50.0)
                    for k in range(self.n_outputs_)
                ]

        elif self.strategy == "quantile":
            # Ensure a valid quantile value is provided for the quantile strategy
            if self.quantile is None:
                raise ValueError(
                    "When using `strategy='quantile', you have to specify the desired "
                    "quantile in the range [0, 1]."
                )
            percentile = self.quantile * 100.0
            # Compute the weighted quantile of y based on the specified quantile
            if sample_weight is None:
                self.constant_ = np.percentile(y, axis=0, q=percentile)
            else:
                self.constant_ = [
                    _weighted_percentile(y[:, k], sample_weight, percentile=percentile)
                    for k in range(self.n_outputs_)
                ]

        elif self.strategy == "constant":
            # Ensure a constant value is provided for the constant strategy
            if self.constant is None:
                raise TypeError(
                    "Constant target value has to be specified "
                    "when the constant strategy is used."
                )
            # Validate and reshape the constant value if necessary
            self.constant_ = check_array(
                self.constant,
                accept_sparse=["csr", "csc", "coo"],
                ensure_2d=False,
                ensure_min_samples=0,
            )
            # Ensure the shape of constant_ matches the expected shape based on y
            if self.n_outputs_ != 1 and self.constant_.shape[0] != y.shape[1]:
                raise ValueError(
                    "Constant target value should have shape (%d, 1)." % y.shape[1]
                )

        # Reshape constant_ to ensure it's in the format (1, n_outputs)
        self.constant_ = np.reshape(self.constant_, (1, -1))
        # Return the fitted estimator object
        return self
    def predict(self, X, return_std=False):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.
            All zeros in this case.

            .. versionadded:: 0.20

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.

        y_std : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Standard deviation of predictive distribution of query points.
        """
        # 确保模型已拟合
        check_is_fitted(self)
        # 获取样本数
        n_samples = _num_samples(X)

        # 初始化预测值 y，使用常数 self.constant_ 填充
        y = np.full(
            (n_samples, self.n_outputs_),
            self.constant_,
            dtype=np.array(self.constant_).dtype,
        )
        # 初始化预测标准差 y_std，全部设为零
        y_std = np.zeros((n_samples, self.n_outputs_))

        # 如果只有一个输出，将 y 和 y_std 压平为一维数组
        if self.n_outputs_ == 1:
            y = np.ravel(y)
            y_std = np.ravel(y_std)

        # 根据 return_std 参数决定返回 y 还是 (y, y_std)
        return (y, y_std) if return_std else y

    def _more_tags(self):
        """Return additional tags for the estimator."""
        return {"poor_score": True, "no_validation": True}

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as `(1 - u/v)`, where `u` is the
        residual sum of squares `((y_true - y_pred) ** 2).sum()` and `v` is the
        total sum of squares `((y_true - y_true.mean()) ** 2).sum()`. The best
        possible score is 1.0 and it can be negative (because the model can be
        arbitrarily worse). A constant model that always predicts the expected
        value of y, disregarding the input features, would get a R^2 score of
        0.0.

        Parameters
        ----------
        X : None or array-like of shape (n_samples, n_features)
            Test samples. Passing None as test samples gives the same result
            as passing real test samples, since `DummyRegressor`
            operates independently of the sampled observations.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of `self.predict(X)` w.r.t. y.
        """
        # 如果 X 为 None，则将其替换为零矩阵，形状为 (len(y), 1)
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        # 调用父类的 score 方法计算 R^2 分数
        return super().score(X, y, sample_weight)
```