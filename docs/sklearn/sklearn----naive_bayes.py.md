# `D:\src\scipysrc\scikit-learn\sklearn\naive_bayes.py`

```
"""
Naive Bayes algorithms.

These are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import warnings  # 导入警告模块
from abc import ABCMeta, abstractmethod  # 导入抽象基类元类和抽象方法装饰器
from numbers import Integral, Real  # 导入整数和实数类型的支持

import numpy as np  # 导入NumPy库
from scipy.special import logsumexp  # 导入logsumexp函数

from .base import BaseEstimator, ClassifierMixin, _fit_context  # 从当前包中导入基础估计器类、分类器混合类和_fit_context
from .preprocessing import LabelBinarizer, binarize, label_binarize  # 从当前包中导入标签二值化类和相关函数
from .utils._param_validation import Interval  # 从当前包中导入参数验证模块中的Interval类
from .utils.extmath import safe_sparse_dot  # 从当前包中导入扩展数学计算模块中的safe_sparse_dot函数
from .utils.multiclass import _check_partial_fit_first_call  # 从当前包中导入多类别分类中的_partial_fit_first_call函数
from .utils.validation import _check_sample_weight, check_is_fitted, check_non_negative  # 从当前包中导入验证模块中的相关函数

__all__ = [
    "BernoulliNB",
    "GaussianNB",
    "MultinomialNB",
    "ComplementNB",
    "CategoricalNB",
]

class _BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X

        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape (n_samples, n_classes).

        Public methods predict, predict_proba, predict_log_proba, and
        predict_joint_log_proba pass the input through _check_X before handing it
        over to _joint_log_likelihood. The term "joint log likelihood" is used
        interchangeably with "joint log probability".
        """

    @abstractmethod
    def _check_X(self, X):
        """To be overridden in subclasses with the actual checks.

        Only used in predict* methods.
        """

    def predict_joint_log_proba(self, X):
        """Return joint log probability estimates for the test vector X.

        For each row x of X and class y, the joint log probability is given by
        ``log P(x, y) = log P(y) + log P(x|y),``
        where ``log P(y)`` is the class prior probability and ``log P(x|y)`` is
        the class-conditional probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Returns the joint log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)  # 检查估计器是否已经拟合
        X = self._check_X(X)  # 对输入数据进行检查
        return self._joint_log_likelihood(X)  # 返回联合对数概率估计
    def predict(self, X):
        """
        对测试向量 X 进行分类预测。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入样本。

        Returns
        -------
        C : ndarray of shape (n_samples,)
            预测的目标值数组，对应于 X 中每个样本的预测结果。
        """
        # 检查模型是否已经拟合（训练）
        check_is_fitted(self)
        # 校验输入的 X，并可能对其进行必要的预处理
        X = self._check_X(X)
        # 计算联合对数似然
        jll = self._joint_log_likelihood(X)
        # 返回每个样本预测的类别，这里选择联合对数似然最大的类别
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """
        返回测试向量 X 的对数概率估计。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入样本。

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            返回模型中每个类别的样本的对数概率估计。列按照在属性 :term:`classes_` 中出现的顺序排序。
        """
        # 检查模型是否已经拟合（训练）
        check_is_fitted(self)
        # 校验输入的 X，并可能对其进行必要的预处理
        X = self._check_X(X)
        # 计算联合对数似然
        jll = self._joint_log_likelihood(X)
        # 对联合对数似然进行归一化处理，得到对数概率
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        返回测试向量 X 的概率估计。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入样本。

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            返回模型中每个类别的样本的概率估计。列按照在属性 :term:`classes_` 中出现的顺序排序。
        """
        # 返回对数概率的指数化，得到概率估计
        return np.exp(self.predict_log_proba(X))
class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB).

    Can perform online updates to model parameters via :meth:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    Read more in the :ref:`User Guide <gaussian_naive_bayes>`.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

        .. versionadded:: 0.20

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        number of training samples observed in each class.

    class_prior_ : ndarray of shape (n_classes,)
        probability of each class.

    classes_ : ndarray of shape (n_classes,)
        class labels known to the classifier.

    epsilon_ : float
        absolute additive value to variances.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    var_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.

        .. versionadded:: 1.0

    theta_ : ndarray of shape (n_classes, n_features)
        mean of each feature per class.

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : Complement Naive Bayes classifier.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]
    """

    _parameter_constraints: dict = {
        "priors": ["array-like", None],
        "var_smoothing": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, *, priors=None, var_smoothing=1e-9):
        # 初始化 GaussianNB 分类器
        self.priors = priors  # 设置先验概率 priors
        self.var_smoothing = var_smoothing  # 设置平滑因子 var_smoothing

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit Gaussian Naive Bayes according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17
               Gaussian Naive Bayes supports fitting with *sample_weight*.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 调用 _validate_data 方法验证并处理目标值 y
        y = self._validate_data(y=y)
        # 调用 _partial_fit 方法执行部分拟合，传入参数 X, y, 类别列表，是否重新拟合以及样本权重
        return self._partial_fit(
            X, y, np.unique(y), _refit=True, sample_weight=sample_weight
        )

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        # 调用 _validate_data 方法验证数据 X，但不重置状态
        return self._validate_data(X, reset=False)
    # 定义一个函数，用于在线更新高斯分布的均值和方差
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        """Compute online update of Gaussian mean and variance.

        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in X is treated as independent
        -- you get variance, not covariance).

        Can take scalar mean and variance, or vector mean and variance to
        simultaneously update a number of independent Gaussians.

        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample
            weights were given, this should contain the sum of sample
            weights represented in old mean and variance.

        mu : array-like of shape (number of Gaussians,)
            Means for Gaussians in original set.

        var : array-like of shape (number of Gaussians,)
            Variances for Gaussians in original set.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        total_mu : array-like of shape (number of Gaussians,)
            Updated mean for each Gaussian over the combined set.

        total_var : array-like of shape (number of Gaussians,)
            Updated variance for each Gaussian over the combined set.
        """
        # 如果新数据集为空，则直接返回原始均值和方差
        if X.shape[0] == 0:
            return mu, var

        # 计算（可能带权重的）新数据点的均值和方差
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            if np.isclose(n_new, 0.0):
                return mu, var
            # 加权平均值
            new_mu = np.average(X, axis=0, weights=sample_weight)
            # 加权方差
            new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
        else:
            n_new = X.shape[0]
            # 计算未加权情况下的方差和均值
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        # 如果过去没有样本，则直接返回新计算的均值和方差
        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # 组合旧数据和新数据的均值，考虑观测次数（可能带权重）
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # 组合旧数据和新数据的方差，考虑观测次数（可能带权重）
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        total_var = total_ssd / n_total

        return total_mu, total_var

    @_fit_context(prefer_skip_nested_validation=True)
    # 对象的增量拟合方法，用于处理数据集的批量样本
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 调用内部方法 _partial_fit，设置 _refit 参数为 False
        return self._partial_fit(
            X, y, classes, _refit=False, sample_weight=sample_weight
        )

    # 计算给定数据集 X 的联合对数似然
    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        # 对每个类别计算联合对数似然
        for i in range(np.size(self.classes_)):
            # 计算当前类别的先验概率的自然对数
            jointi = np.log(self.class_prior_[i])
            # 计算当前类别的每个特征的方差的自然对数
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
            # 计算当前类别的每个特征的负半高斯距离平方和
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.var_[i, :]), 1)
            # 将当前类别的联合对数似然添加到结果列表中
            joint_log_likelihood.append(jointi + n_ij)

        # 转换为 NumPy 数组并返回
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood
# 定义一个抽象基类，用于处理离散/分类数据的朴素贝叶斯算法

# 定义类的参数约束，描述了各参数的类型和可能取值范围
class _BaseDiscreteNB(_BaseNB):
    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "fit_prior": ["boolean"],
        "class_prior": ["array-like", None],
        "force_alpha": ["boolean"],
    }

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, force_alpha=True):
        # 初始化朴素贝叶斯模型的参数
        self.alpha = alpha  # 平滑参数 alpha
        self.fit_prior = fit_prior  # 是否拟合先验概率
        self.class_prior = class_prior  # 类别的先验概率
        self.force_alpha = force_alpha  # 是否强制使用 alpha

    @abstractmethod
    def _count(self, X, Y):
        """更新用于计算概率的计数。

        这些计数是从数据中提取的充分统计量。因此，每次“fit”或“partial_fit”
        更新模型时，都会调用此方法。必须在这里更新 `class_count_` 和 `feature_count_`
        以及任何特定于模型的计数。

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            输入样本。
        Y : ndarray of shape (n_samples, n_classes)
            二值化的类标签。
        """

    @abstractmethod
    def _update_feature_log_prob(self, alpha):
        """基于计数更新特征的对数概率。

        每次“fit”或“partial_fit”更新模型时，都会调用此方法。

        Parameters
        ----------
        alpha : float
            平滑参数。参见 :meth:`_check_alpha`。
        """

    def _check_X(self, X):
        """验证 X，仅在预测方法中使用。"""
        return self._validate_data(X, accept_sparse="csr", reset=False)

    def _check_X_y(self, X, y, reset=True):
        """验证拟合方法中的 X 和 y。"""
        return self._validate_data(X, y, accept_sparse="csr", reset=reset)
    # 更新类别的对数先验概率。

    # 获取当前类别的数量
    n_classes = len(self.classes_)

    # 如果给定了 class_prior 参数，则使用给定的先验概率来更新类别的对数先验概率
    if class_prior is not None:
        # 检查给定的先验概率长度是否与类别数量相匹配
        if len(class_prior) != n_classes:
            raise ValueError("Number of priors must match number of classes.")
        # 计算给定先验概率的对数值并存储在 class_log_prior_ 属性中
        self.class_log_prior_ = np.log(class_prior)
    # 如果未指定 class_prior，并且 fit_prior=True
    elif self.fit_prior:
        # 忽略警告，当某类别的样本数量为 0 时
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # 计算每个类别样本数量的对数值
            log_class_count = np.log(self.class_count_)

        # 计算经验先验概率，考虑样本权重的影响，并存储在 class_log_prior_ 属性中
        self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
    # 如果未指定 class_prior，并且 fit_prior=False
    else:
        # 设置默认的对数先验概率为每个类别均匀分布的情况
        self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    # 检查并返回有效的 alpha 值
    def _check_alpha(self):
        # 将 alpha 转换为 numpy 数组（如果不是实数），或者保持 alpha 不变（如果是实数）
        alpha = np.asarray(self.alpha) if not isinstance(self.alpha, Real) else self.alpha

        # 获取 alpha 的最小值
        alpha_min = np.min(alpha)

        # 如果 alpha 是 numpy 数组
        if isinstance(alpha, np.ndarray):
            # 检查 alpha 的长度是否与特征数相匹配
            if not alpha.shape[0] == self.n_features_in_:
                raise ValueError(
                    "When alpha is an array, it should contains `n_features`. "
                    f"Got {alpha.shape[0]} elements instead of {self.n_features_in_}."
                )
            # 检查所有 alpha 是否为正数
            if alpha_min < 0:
                raise ValueError("All values in alpha must be greater than 0.")

        # 设定 alpha 的下限值
        alpha_lower_bound = 1e-10

        # 如果 alpha 的最小值小于下限值，并且未设置 force_alpha=True
        if alpha_min < alpha_lower_bound and not self.force_alpha:
            # 发出警告并设置 alpha 的最小值为 alpha_lower_bound
            warnings.warn(
                "alpha too small will result in numeric errors, setting alpha ="
                f" {alpha_lower_bound:.1e}. Use `force_alpha=True` to keep alpha"
                " unchanged."
            )
            # 返回 alpha 的各元素与 alpha_lower_bound 中较大的值
            return np.maximum(alpha, alpha_lower_bound)

        # 返回 alpha
        return alpha

    # 用于设置 fit_context 装饰器，标识在拟合过程中是否跳过嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
    @_fit_context(prefer_skip_nested_validation=True)
    # 初始化朴素贝叶斯分类器，根据输入的训练数据 X 和目标值 y 进行拟合
    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 检查并返回处理后的输入数据 X 和目标值 y
        X, y = self._check_X_y(X, y)
        # 获取特征数量
        _, n_features = X.shape

        # 初始化标签二值化器
        labelbin = LabelBinarizer()
        # 对目标值 y 进行二值化转换
        Y = labelbin.fit_transform(y)
        # 记录类别信息到分类器实例中
        self.classes_ = labelbin.classes_

        # 处理二分类问题中 Y 的形状，确保 Y 是一个二维数组
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                # 如果只有两类且 Y 是一列，将 Y 转换为两列的形式
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # 特殊情况：只有一类
                # 如果只有一类，则 Y 全为 1
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() 返回的数组类型为 np.int64，
        # 为了支持样本权重的一致性，将其转换为 np.float64 类型；
        # 这样就不需要将 X 强制转换为浮点数类型了
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            # 检查和处理样本权重，保证其形状与 X 一致
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            # 将样本权重应用到 Y 上
            Y *= sample_weight.T

        # 获取类先验概率
        class_prior = self.class_prior

        # 初始化类计数器和特征计数器
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        # 对数据进行计数操作，以便更新类别对数先验和特征对数概率
        self._count(X, Y)
        # 检查并获取平滑参数 alpha
        alpha = self._check_alpha()
        # 更新特征对数概率
        self._update_feature_log_prob(alpha)
        # 更新类别对数先验
        self._update_class_log_prior(class_prior=class_prior)
        # 返回分类器实例本身
        return self

    # 初始化类计数器和特征计数器
    def _init_counters(self, n_classes, n_features):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    # 提供额外的标签，用于分类器的更多标记信息
    def _more_tags(self):
        return {"poor_score": True}
# 定义 MultinomialNB 类，继承自 _BaseDiscreteNB 类
class MultinomialNB(_BaseDiscreteNB):
    """
    多项式朴素贝叶斯分类器。

    多项式朴素贝叶斯分类器适用于具有离散特征的分类（例如文本分类中的单词计数）。多项式分布通常要求特征计数为整数。但在实践中，如 tf-idf 等分数计数也可以工作。

    详细信息请参阅 :ref:`用户指南 <multinomial_naive_bayes>`。

    Parameters
    ----------
    alpha : float 或形状为 (n_features,) 的数组，默认为 1.0
        添加的（拉普拉斯/利德斯通）平滑参数
        （设置 alpha=0 并且 force_alpha=True，以取消平滑）。

    force_alpha : bool，默认为 True
        如果为 False 并且 alpha 小于 1e-10，则将 alpha 设置为 1e-10。如果为 True，则保持 alpha 不变。如果 alpha 接近于 0，可能会导致数值错误。

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           `force_alpha` 的默认值更改为 `True`。

    fit_prior : bool，默认为 True
        是否学习类的先验概率。
        如果为 false，将使用均匀先验。

    class_prior : 形状为 (n_classes,) 的数组，默认为 None
        类的先验概率。如果指定，则不根据数据调整先验概率。

    Attributes
    ----------
    class_count_ : 形状为 (n_classes,) 的 ndarray
        在拟合过程中遇到的每个类别的样本数量。当提供样本权重时，此值加权。

    class_log_prior_ : 形状为 (n_classes,) 的 ndarray
        每个类别的平滑经验对数概率。

    classes_ : 形状为 (n_classes,) 的 ndarray
        分类器已知的类标签

    feature_count_ : 形状为 (n_classes, n_features) 的 ndarray
        在拟合过程中遇到的每个（类别，特征）的样本数量。当提供样本权重时，此值加权。

    feature_log_prob_ : 形状为 (n_classes, n_features) 的 ndarray
        给定类别时特征的经验对数概率，即 ``P(x_i|y)``。

    n_features_in_ : int
        在 :term:`fit` 过程中看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : 形状为 (`n_features_in_`,) 的 ndarray
        在 :term:`fit` 过程中看到的特征名称。仅在 `X` 具有全部字符串特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    BernoulliNB : 多变量伯努利模型的朴素贝叶斯分类器。
    CategoricalNB : 用于分类特征的朴素贝叶斯分类器。
    ComplementNB : 补充朴素贝叶斯分类器。
    GaussianNB : 高斯朴素贝叶斯。

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
    """
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)  # 创建一个指定种子的随机数生成器对象
    >>> X = rng.randint(5, size=(6, 100))  # 生成一个6行100列的随机整数矩阵
    >>> y = np.array([1, 2, 3, 4, 5, 6])  # 创建一个包含类标签的NumPy数组
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()  # 创建MultinomialNB分类器的实例
    >>> clf.fit(X, y)  # 使用训练数据X和类标签y拟合分类器
    MultinomialNB()
    >>> print(clf.predict(X[2:3]))  # 打印预测结果，预测输入X的第3行
    [3]
    """

    def __init__(
        self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None
    ):
        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            force_alpha=force_alpha,
        )
        # 调用父类的构造函数，设置alpha值、是否强制使用alpha、是否拟合先验概率、类先验概率

    def _more_tags(self):
        return {"requires_positive_X": True}
        # 返回一个字典，指示分类器要求输入数据X必须为正值

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        check_non_negative(X, "MultinomialNB (input X)")  # 检查输入矩阵X中是否存在负值
        self.feature_count_ += safe_sparse_dot(Y.T, X)  # 计算特征计数并平滑处理
        self.class_count_ += Y.sum(axis=0)  # 计算类别计数

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha  # 平滑处理特征计数
        smoothed_cc = smoothed_fc.sum(axis=1)  # 计算平滑后的类别计数

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )
        # 更新特征的对数概率估计值

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_
        # 计算样本X的后验对数概率
class ComplementNB(_BaseDiscreteNB):
    """The Complement Naive Bayes classifier described in Rennie et al. (2003).

    The Complement Naive Bayes classifier was designed to correct the "severe
    assumptions" made by the standard Multinomial Naive Bayes classifier. It is
    particularly suited for imbalanced data sets.

    Read more in the :ref:`User Guide <complement_naive_bayes>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.

    fit_prior : bool, default=True
        Only used in edge case with a single class in the training set.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. Not used.

    norm : bool, default=False
        Whether or not a second normalization of the weights is performed. The
        default behavior mirrors the implementations found in Mahout and Weka,
        which do not follow the full algorithm described in Table 9 of the
        paper.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class. Only used in edge
        case with a single class in the training set.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_all_ : ndarray of shape (n_features,)
        Number of samples encountered for each feature during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature) during fitting.
        This value is weighted by the sample weight when provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical weights for class complements.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    """

    # 定义 ComplementNB 类，继承自 _BaseDiscreteNB 基类
    def __init__(self, alpha=1.0, force_alpha=True, fit_prior=True,
                 class_prior=None, norm=False):
        # 调用父类 _BaseDiscreteNB 的初始化方法
        super().__init__(alpha=alpha, fit_prior=fit_prior,
                         class_prior=class_prior)

        # 设置是否强制使用 alpha 的标志
        self.force_alpha = force_alpha

        # 设置是否进行第二次权重规范化的标志
        self.norm = norm

    def _check_alpha(self, alpha):
        # 如果 force_alpha 为 False 且 alpha 小于 1e-10，则将 alpha 设置为 1e-10
        if not self.force_alpha and alpha < 1e-10:
            alpha = 1e-10
        return alpha

    def _update_class_log_prior(self, class_prior=None):
        # 如果 fit_prior 为 False 或者 class_prior 为 None，则不进行操作
        if not self.fit_prior or class_prior is None:
            return

        # 更新类的对数先验概率
        self.class_log_prior_ = np.log(class_prior + self.alpha)

    def _update_feature_log_prob(self, smoothed_fc, smoothed_cc, class_prior=None):
        # 更新特征的对数概率
        self.feature_log_prob_ = (np.log(smoothed_cc) -
                                  np.log(smoothed_fc.reshape(-1, 1) +
                                         self.n_features_in_ * self.alpha))

    def fit(self, X, y, sample_weight=None):
        # 使用 ComplementNB 模型拟合数据集 X 和标签 y
        self._check_params(X, y)
        _, n_features = X.shape

        # 初始化类别数量、特征计数和所有特征数量
        self.class_count_ = np.zeros(self.classes_.shape[0], dtype=np.float64)
        self.feature_count_ = np.zeros((self.classes_.shape[0], n_features), dtype=np.float64)
        self.feature_all_ = np.zeros(n_features, dtype=np.float64)

        # 更新类别数量、特征计数和所有特征数量
        self._count(X, y, sample_weight)

        # 计算类别的对数先验概率
        self._update_class_log_prior(class_prior)

        # 计算特征的对数概率
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = self.feature_all_ + self.alpha * self.n_features_in_

        # 更新特征的对数概率
        self._update_feature_log_prob(smoothed_fc, smoothed_cc, class_prior)

        # 返回 ComplementNB 对象本身
        return self
    """
    CategoricalNB : Naive Bayes classifier for categorical features.
    GaussianNB : Gaussian Naive Bayes.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    References
    ----------
    Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
    Tackling the poor assumptions of naive bayes text classifiers. In ICML
    (Vol. 3, pp. 616-623).
    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import ComplementNB
    >>> clf = ComplementNB()
    >>> clf.fit(X, y)
    ComplementNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    # 定义参数约束，继承自_BaseDiscreteNB的参数约束，并添加"norm"参数的约束
    _parameter_constraints: dict = {
        **_BaseDiscreteNB._parameter_constraints,
        "norm": ["boolean"],
    }

    def __init__(
        self,
        *,
        alpha=1.0,
        force_alpha=True,
        fit_prior=True,
        class_prior=None,
        norm=False,
    ):
        # 调用父类_BaseDiscreteNB的初始化方法，设置类的参数
        super().__init__(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
        )
        # 设置类的额外参数norm
        self.norm = norm

    def _more_tags(self):
        # 返回更多标签，指示需要正值的X
        return {"requires_positive_X": True}

    def _count(self, X, Y):
        """Count feature occurrences."""
        # 检查输入X是否非负
        check_non_negative(X, "ComplementNB (input X)")
        # 更新特征计数和类别计数
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and compute the weights."""
        # 计算平滑后的特征对数概率
        comp_count = self.feature_all_ + alpha - self.feature_count_
        logged = np.log(comp_count / comp_count.sum(axis=1, keepdims=True))
        # _BaseNB.predict使用argmax，但ComplementNB使用argmin
        if self.norm:
            summed = logged.sum(axis=1, keepdims=True)
            feature_log_prob = logged / summed
        else:
            feature_log_prob = -logged
        self.feature_log_prob_ = feature_log_prob

    def _joint_log_likelihood(self, X):
        """Calculate the class scores for the samples in X."""
        # 计算X中样本的类别得分
        jll = safe_sparse_dot(X, self.feature_log_prob_.T)
        if len(self.classes_) == 1:
            jll += self.class_log_prior_
        return jll
# 定义 BernoulliNB 类，继承自 _BaseDiscreteNB 类
class BernoulliNB(_BaseDiscreteNB):
    
    """Naive Bayes classifier for multivariate Bernoulli models.
    
    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.
    
    Read more in the :ref:`User Guide <bernoulli_naive_bayes>`.
    
    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).
        
    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.
        
        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.
    
    binarize : float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.
    
    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    
    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.
    
    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.
    
    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class (smoothed).
    
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier
    
    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.
    
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        
        .. versionadded:: 1.0
    
    See Also
    --------
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : The Complement Naive Bayes classifier
        described in Rennie et al. (2003).
    GaussianNB : Gaussian Naive Bayes (GaussianNB).
    MultinomialNB : Naive Bayes classifier for multinomial models.
    
    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    """

    # 初始化函数，设置 BernoulliNB 的参数
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True,
                 class_prior=None, force_alpha=True):
        # 调用父类的初始化函数
        super().__init__(alpha=alpha, binarize=binarize,
                         fit_prior=fit_prior, class_prior=class_prior)
        # 设置 force_alpha 参数
        self.force_alpha = force_alpha
    
    # 拟合（训练）模型的方法
    def fit(self, X, y, sample_weight=None):
        # 调用父类的拟合方法进行模型训练
        super().fit(X, y, sample_weight=sample_weight)
        # 返回对象自身，用于方法链式调用
        return self

    # 预测新数据的方法
    def predict(self, X):
        # 调用父类的预测方法进行预测
        return super().predict(X)

    # 计算对数概率的方法
    def predict_log_proba(self, X):
        # 调用父类的对数概率预测方法
        return super().predict_log_proba(X)

    # 计算概率的方法
    def predict_proba(self, X):
        # 调用父类的概率预测方法
        return super().predict_proba(X)
    _parameter_constraints: dict = {
        **_BaseDiscreteNB._parameter_constraints,
        "binarize": [None, Interval(Real, 0, None, closed="left")],
    }


# 定义参数约束字典，包括继承自_BaseDiscreteNB的约束条件和额外的"binarize"参数约束
_parameter_constraints: dict = {
    **_BaseDiscreteNB._parameter_constraints,  # 继承基类_BaseDiscreteNB的参数约束
    "binarize": [None, Interval(Real, 0, None, closed="left")],  # 添加对"binarize"参数的约束
}



    def __init__(
        self,
        *,
        alpha=1.0,
        force_alpha=True,
        binarize=0.0,
        fit_prior=True,
        class_prior=None,
    ):


# 初始化函数，设定朴素贝叶斯伯努利模型的参数
def __init__(
    self,
    *,
    alpha=1.0,          # 平滑参数，默认为1.0
    force_alpha=True,   # 是否强制使用alpha进行平滑，默认为True
    binarize=0.0,       # 二值化阈值，默认为0.0
    fit_prior=True,     # 是否学习类先验概率，默认为True
    class_prior=None,   # 各类别的先验概率，默认为None
):



        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            force_alpha=force_alpha,
        )
        self.binarize = binarize


# 调用父类的初始化方法，并设置本类特有的binarize参数
super().__init__(
    alpha=alpha,            # 设置平滑参数alpha
    fit_prior=fit_prior,    # 设置是否学习类先验概率
    class_prior=class_prior,# 设置类别的先验概率
    force_alpha=force_alpha,# 设置是否强制使用alpha进行平滑
)
self.binarize = binarize   # 设置本类特有的二值化阈值参数



    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        X = super()._check_X(X)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return X


# 验证输入数据X的有效性，仅在预测方法中使用
def _check_X(self, X):
    """Validate X, used only in predict* methods."""
    X = super()._check_X(X)                     # 调用父类方法验证X的有效性
    if self.binarize is not None:
        X = binarize(X, threshold=self.binarize) # 如果设置了binarize参数，则对X进行二值化处理
    return X



    def _check_X_y(self, X, y, reset=True):
        X, y = super()._check_X_y(X, y, reset=reset)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return X, y


# 验证输入数据X和y的有效性，并返回验证后的数据
def _check_X_y(self, X, y, reset=True):
    X, y = super()._check_X_y(X, y, reset=reset)   # 调用父类方法验证X和y的有效性
    if self.binarize is not None:
        X = binarize(X, threshold=self.binarize)  # 如果设置了binarize参数，则对X进行二值化处理
    return X, y



    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)


# 计算并平滑特征的出现次数
def _count(self, X, Y):
    """Count and smooth feature occurrences."""
    self.feature_count_ += safe_sparse_dot(Y.T, X)   # 计算特征出现次数并累加到feature_count_
    self.class_count_ += Y.sum(axis=0)              # 计算类别出现次数并累加到class_count_



    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )


# 对原始计数应用平滑，并重新计算对数概率
def _update_feature_log_prob(self, alpha):
    """Apply smoothing to raw counts and recompute log probabilities"""
    smoothed_fc = self.feature_count_ + alpha         # 对特征计数应用平滑
    smoothed_cc = self.class_count_ + alpha * 2        # 对类别计数应用平滑

    self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
        smoothed_cc.reshape(-1, 1)
    )  # 计算特征的对数概率
    # 计算样本 X 的后验对数概率
    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        # 获取特征对数概率矩阵的列数，即特征的数量
        n_features = self.feature_log_prob_.shape[1]
        # 获取输入样本 X 的特征数量
        n_features_X = X.shape[1]

        # 如果输入样本 X 的特征数量与模型的特征数量不匹配，则抛出 ValueError 异常
        if n_features_X != n_features:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (n_features, n_features_X)
            )

        # 计算负概率对数，即 log(1 - exp(feature_log_prob))
        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        
        # 计算 jll，即后验对数概率，使用稀疏矩阵乘法计算 X · (feature_log_prob - neg_prob).T
        jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        
        # 加上类别的先验对数概率以及负概率对数的和
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        # 返回计算得到的后验对数概率
        return jll
class CategoricalNB(_BaseDiscreteNB):
    """Naive Bayes classifier for categorical features.

    The categorical Naive Bayes classifier is suitable for classification with
    discrete features that are categorically distributed. The categories of
    each feature are drawn from a categorical distribution.

    Read more in the :ref:`User Guide <categorical_naive_bayes>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    min_categories : int or array-like of shape (n_features,), default=None
        Minimum number of categories per feature.

        - integer: Sets the minimum number of categories per feature to
          `n_categories` for each feature.
        - array-like: shape (n_features,) where `n_categories[i]` holds the
          minimum number of categories for the ith column of the input.
        - None (default): Determines the number of categories automatically
          from the training data.

        .. versionadded:: 0.24

    Attributes
    ----------
    category_count_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the number of samples
        encountered for each class and category of the specific feature.

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_log_prob_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the empirical log probability
        of categories given the respective feature and class, ``P(x_i|y)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    """
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_categories_ : ndarray of shape (n_features,), dtype=np.int64
        Number of categories for each feature. This value is
        inferred from the data or set by the minimum number of categories.

        .. versionadded:: 0.24

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    ComplementNB : Complement Naive Bayes classifier.
    GaussianNB : Gaussian Naive Bayes.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import CategoricalNB
    >>> clf = CategoricalNB()
    >>> clf.fit(X, y)
    CategoricalNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """
    # 定义参数约束字典，继承自_BaseDiscreteNB的参数约束，同时增加了两个新的约束条件
    _parameter_constraints: dict = {
        **_BaseDiscreteNB._parameter_constraints,
        "min_categories": [
            None,  # 可以为None
            "array-like",  # 必须是类数组结构
            Interval(Integral, 1, None, closed="left"),  # 必须是整数且大于等于1
        ],
        "alpha": [Interval(Real, 0, None, closed="left")],  # alpha值必须是实数且大于等于0
    }

    def __init__(
        self,
        *,
        alpha=1.0,
        force_alpha=True,
        fit_prior=True,
        class_prior=None,
        min_categories=None,
    ):
        # 调用父类的初始化方法，设置对象的基本属性
        super().__init__(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
        )
        self.min_categories = min_categories  # 设置对象的min_categories属性

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().fit(X, y, sample_weight=sample_weight)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 调用父类的partial_fit方法来实现增量学习
        return super().partial_fit(X, y, classes, sample_weight=sample_weight)

    def _more_tags(self):
        """Returns additional tags for the estimator."""
        return {"requires_positive_X": True}

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        # 验证输入数据X的合法性，确保数据类型为整数，不接受稀疏矩阵，且所有元素非无穷大或NaN
        X = self._validate_data(
            X, dtype="int", accept_sparse=False, force_all_finite=True, reset=False
        )
        # 检查输入数据X是否为非负数
        check_non_negative(X, "CategoricalNB (input X)")
        return X

    def _check_X_y(self, X, y, reset=True):
        """Validate X and y, used in fit methods."""
        # 验证输入数据X和标签y的合法性，确保数据类型为整数，且所有元素非无穷大或NaN
        X, y = self._validate_data(
            X, y, dtype="int", accept_sparse=False, force_all_finite=True, reset=reset
        )
        # 检查输入数据X是否为非负数
        check_non_negative(X, "CategoricalNB (input X)")
        return X, y

    def _init_counters(self, n_classes, n_features):
        """Initialize counters for class and category counts."""
        # 初始化类别和特征计数器，用于存储类别和特征的统计信息
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.category_count_ = [np.zeros((n_classes, 0)) for _ in range(n_features)]
    def _validate_n_categories(X, min_categories):
        # 计算每个特征列的最大值，并加1得到特征列的类别数
        n_categories_X = X.max(axis=0) + 1
        min_categories_ = np.array(min_categories)
        if min_categories is not None:
            # 检查 min_categories 是否为整数类型
            if not np.issubdtype(min_categories_.dtype, np.signedinteger):
                raise ValueError(
                    "'min_categories' should have integral type. Got "
                    f"{min_categories_.dtype} instead."
                )
            # 将 X 的类别数与 min_categories 进行比较，取较大值
            n_categories_ = np.maximum(n_categories_X, min_categories_, dtype=np.int64)
            # 检查 n_categories_ 的形状是否与 n_categories_X 相同
            if n_categories_.shape != n_categories_X.shape:
                raise ValueError(
                    f"'min_categories' should have shape ({X.shape[1]},"
                    ") when an array-like is provided. Got"
                    f" {min_categories_.shape} instead."
                )
            return n_categories_
        else:
            return n_categories_X

    def _count(self, X, Y):
        def _update_cat_count_dims(cat_count, highest_feature):
            # 计算需要增加的列数，以便为新的类别添加一列全为零的数组
            diff = highest_feature + 1 - cat_count.shape[1]
            if diff > 0:
                return np.pad(cat_count, [(0, 0), (0, diff)], "constant")
            return cat_count

        def _update_cat_count(X_feature, Y, cat_count, n_classes):
            # 更新类别计数
            for j in range(n_classes):
                mask = Y[:, j].astype(bool)
                if Y.dtype.type == np.int64:
                    weights = None
                else:
                    weights = Y[mask, j]
                counts = np.bincount(X_feature[mask], weights=weights)
                indices = np.nonzero(counts)[0]
                cat_count[j, indices] += counts[indices]

        # 更新每个类别的总数
        self.class_count_ += Y.sum(axis=0)
        # 验证特征列的类别数，并返回验证后的结果
        self.n_categories_ = self._validate_n_categories(X, self.min_categories)
        # 遍历特征列
        for i in range(self.n_features_in_):
            X_feature = X[:, i]
            # 更新特征列的类别计数维度
            self.category_count_[i] = _update_cat_count_dims(
                self.category_count_[i], self.n_categories_[i] - 1
            )
            # 更新特征列的类别计数
            _update_cat_count(
                X_feature, Y, self.category_count_[i], self.class_count_.shape[0]
            )

    def _update_feature_log_prob(self, alpha):
        feature_log_prob = []
        # 遍历每个特征列
        for i in range(self.n_features_in_):
            # 平滑后的类别计数
            smoothed_cat_count = self.category_count_[i] + alpha
            smoothed_class_count = smoothed_cat_count.sum(axis=1)
            # 计算特征的对数概率
            feature_log_prob.append(
                np.log(smoothed_cat_count) - np.log(smoothed_class_count.reshape(-1, 1))
            )
        # 更新特征的对数概率
        self.feature_log_prob_ = feature_log_prob
    # 计算联合对数似然值的方法，用于评估样本在各类别下的似然度
    def _joint_log_likelihood(self, X):
        # 检查输入特征的数量是否符合预期，并且不重置状态
        self._check_n_features(X, reset=False)
        
        # 初始化一个全零矩阵，用于存储每个样本在每个类别下的对数似然值
        jll = np.zeros((X.shape[0], self.class_count_.shape[0]))
        
        # 遍历每个特征维度
        for i in range(self.n_features_in_):
            # 获取当前特征维度上的所有样本的索引值
            indices = X[:, i]
            # 将当前特征维度上每个样本的对数概率加总到 jll 矩阵中
            jll += self.feature_log_prob_[i][:, indices].T
        
        # 将类先验的对数概率加到 jll 矩阵上，得到每个样本在每个类别下的总对数似然值
        total_ll = jll + self.class_log_prior_
        
        # 返回每个样本在每个类别下的总对数似然值
        return total_ll
```