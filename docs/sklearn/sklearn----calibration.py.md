# `D:\src\scipysrc\scikit-learn\sklearn\calibration.py`

```
"""Methods for calibrating predicted probabilities."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 引入警告模块，用于处理警告信息
import warnings
# 从inspect模块中导入signature函数，用于获取函数的签名信息
from inspect import signature
# 导入log函数，用于计算自然对数
from math import log
# 导入Integral和Real类，用于检查数值类型
from numbers import Integral, Real

# 导入numpy库，并使用别名np
import numpy as np
# 导入scipy.optimize模块中的minimize函数，用于最小化优化
from scipy.optimize import minimize
# 导入scipy.special模块中的expit函数，用于逻辑函数计算
from scipy.special import expit

# 导入sklearn.utils中的Bunch类，用于创建包含任意属性的对象
from sklearn.utils import Bunch

# 从本地模块中导入HalfBinomialLoss类
from ._loss import HalfBinomialLoss
# 从本地模块中导入多个类和函数，包括基础估计器、分类器、元估计器、回归器、_fit_context函数和clone函数
from .base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    _fit_context,
    clone,
)
# 从本地模块中导入IsotonicRegression类
from .isotonic import IsotonicRegression
# 从本地模块中导入model_selection模块中的check_cv函数和cross_val_predict函数
from .model_selection import check_cv, cross_val_predict
# 从本地模块中导入preprocessing模块中的LabelEncoder类和label_binarize函数
from .preprocessing import LabelEncoder, label_binarize
# 从本地模块中导入svm模块中的LinearSVC类
from .svm import LinearSVC
# 从本地模块中导入utils模块中的多个函数，包括_safe_indexing函数、column_or_1d函数和indexable函数
from .utils import (
    _safe_indexing,
    column_or_1d,
    indexable,
)
# 从本地模块中导入utils._param_validation模块中的多个类，包括HasMethods类、Interval类、StrOptions类和validate_params函数
from .utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    validate_params,
)
# 从本地模块中导入utils._plotting模块中的_BinaryClassifierCurveDisplayMixin类
from .utils._plotting import _BinaryClassifierCurveDisplayMixin
# 从本地模块中导入utils._response模块中的多个函数，包括_get_response_values函数和_process_predict_proba函数
from .utils._response import _get_response_values, _process_predict_proba
# 从本地模块中导入utils.metadata_routing模块中的MetadataRouter类、MethodMapping类、_routing_enabled函数和process_routing函数
from .utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    process_routing,
)
# 从本地模块中导入utils.multiclass模块中的check_classification_targets函数
from .utils.multiclass import check_classification_targets
# 从本地模块中导入utils.parallel模块中的Parallel类和delayed函数
from .utils.parallel import Parallel, delayed
# 从本地模块中导入utils.validation模块中的多个函数，包括_check_method_params函数、_check_pos_label_consistency函数和_check_sample_weight函数
from .utils.validation import (
    _check_method_params,
    _check_pos_label_consistency,
    _check_response_method,
    _check_sample_weight,
    _num_samples,
    check_consistent_length,
    check_is_fitted,
)

# 定义一个名为CalibratedClassifierCV的类，继承自ClassifierMixin、MetaEstimatorMixin和BaseEstimator
class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Probability calibration with isotonic regression or logistic regression.

    This class uses cross-validation to both estimate the parameters of a
    classifier and subsequently calibrate a classifier. With default
    `ensemble=True`, for each cv split it
    fits a copy of the base estimator to the training subset, and calibrates it
    using the testing subset. For prediction, predicted probabilities are
    averaged across these individual calibrated classifiers. When
    `ensemble=False`, cross-validation is used to obtain unbiased predictions,
    via :func:`~sklearn.model_selection.cross_val_predict`, which are then
    used for calibration. For prediction, the base estimator, trained using all
    the data, is used. This is the prediction method implemented when
    `probabilities=True` for :class:`~sklearn.svm.SVC` and :class:`~sklearn.svm.NuSVC`
    estimators (see :ref:`User Guide <scores_probabilities>` for details).

    Already fitted classifiers can be calibrated via the parameter
    `cv="prefit"`. In this case, no cross-validation is used and all provided
    data is used for calibration. The user has to take care manually that data
    for model fitting and calibration are disjoint.

    The calibration is based on the :term:`decision_function` method of the
    `estimator` if it exists, else on :term:`predict_proba`.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    # estimator：分类器实例，默认为 None
    # 需要校准其输出以提供更准确的 `predict_proba` 结果。默认分类器为 :class:`~sklearn.svm.LinearSVC`。
    
    # method：{'sigmoid', 'isotonic'}，默认为 'sigmoid'
    # 校准的方法。可以选择 'sigmoid'，对应于 Platt 方法（即逻辑回归模型），或者 'isotonic'，这是一种非参数化方法。
    # 不建议在校准样本过少（<<1000）的情况下使用 isotonic 校准，因为它容易过拟合。
    
    # cv：int、交叉验证生成器、可迭代对象或者 "prefit"，默认为 None
    # 确定交叉验证的划分策略。
    # 可能的输入包括：
    # - None，使用默认的 5 折交叉验证
    # - 整数，指定折数
    # - :term:`CV splitter`
    # - 一个可迭代对象，生成 (train, test) 索引数组的划分
    
    # 对于整数/None 输入，如果 `y` 是二元或多类别，使用 :class:`~sklearn.model_selection.StratifiedKFold`。
    # 如果 `y` 既不是二元也不是多类别，则使用 :class:`~sklearn.model_selection.KFold`。
    
    # 详细的交叉验证策略请参考 :ref:`User Guide <cross_validation>`。
    
    # 如果传入 "prefit"，假定 `estimator` 已经被拟合过，所有数据将用于校准。
    
    # .. versionchanged:: 0.22
    #    当 None 时，`cv` 默认值从 3 折改为 5 折。
    
    # n_jobs：int，默认为 None
    # 并行运行的作业数。
    # `None` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则为 1。
    # `-1` 表示使用所有处理器。
    
    # 在交叉验证的迭代中，基础估计器的克隆是并行拟合的。因此，仅当 `cv != "prefit"` 时才会并行处理。
    
    # 更多细节请参阅 :term:`Glossary <n_jobs>`。
    
    # .. versionadded:: 0.24
    ensemble : bool, default=True
        # 确定在`cv`不是'prefit'时如何拟合校准器。
        # 如果`cv='prefit'`，则忽略。

        # 如果为`True`，则使用训练数据拟合`estimator`，并使用测试数据校准，
        # 对每个`cv`折叠进行操作。最终的估计器是`n_cv`个拟合的分类器和校准器对的集合，
        # 其中`n_cv`是交叉验证折叠的数量。输出是所有对的平均预测概率。

        # 如果为`False`，则使用`cv`计算无偏预测，通过
        # :func:`~sklearn.model_selection.cross_val_predict`，然后用于校准。
        # 在预测时，使用的分类器是在所有数据上训练的`estimator`。
        # 注意，当`probabilities=True`参数在`sklearn.svm`估计器中也是内部实现的。

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        # 类别标签。

    n_features_in_ : int
        # 在`fit`期间看到的特征数量。仅在底层估计器在拟合时暴露此属性时才定义。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在`fit`期间看到的特征名称。仅在底层估计器在拟合时暴露此属性时才定义。

        .. versionadded:: 1.0

    calibrated_classifiers_ : list (len() equal to cv or 1 if `cv="prefit"` \
            or `ensemble=False`)
        # 分类器和校准器对的列表。

        - 当`cv="prefit"`时，拟合的`estimator`和拟合的校准器。
        - 当`cv`不是"prefit"且`ensemble=True`时，`n_cv`个拟合的`estimator`和校准器对。`n_cv`是交叉验证折叠的数量。
        - 当`cv`不是"prefit"且`ensemble=False`时，拟合在所有数据上的`estimator`和拟合的校准器。

        .. versionchanged:: 0.24
            当`ensemble=False`时，单个校准分类器情况。

    See Also
    --------
    calibration_curve : 计算校准曲线的真实和预测概率。

    References
    ----------
    .. [1] 从决策树和朴素贝叶斯分类器中获得校准的概率估计，B. Zadrozny & C. Elkan, ICML 2001

    .. [2] 将分类器分数转换为准确的多类概率估计，B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] 支持向量机的概率输出及与正则化似然方法的比较，J. Platt, (1999)

    .. [4] 使用监督学习预测良好概率，A. Niculescu-Mizil & R. Caruana, ICML 2005

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.naive_bayes import GaussianNB
    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),  # estimator需要具有fit和predict_proba方法
            HasMethods(["fit", "decision_function"]),  # 或者具有fit和decision_function方法
            None,  # 或者为None
        ],
        "method": [StrOptions({"isotonic", "sigmoid"})],  # method必须是{"isotonic", "sigmoid"}中的一个字符串
        "cv": ["cv_object", StrOptions({"prefit"})],  # cv可以是一个cv_object对象或者字符串"prefit"
        "n_jobs": [Integral, None],  # n_jobs可以是整数或者None
        "ensemble": ["boolean"],  # ensemble必须是布尔值
    }

    def __init__(
        self,
        estimator=None,
        *,
        method="sigmoid",
        cv=None,
        n_jobs=None,
        ensemble=True,
    ):
        self.estimator = estimator  # 初始化self.estimator为传入的estimator参数
        self.method = method  # 初始化self.method为传入的method参数，默认为"sigmoid"
        self.cv = cv  # 初始化self.cv为传入的cv参数
        self.n_jobs = n_jobs  # 初始化self.n_jobs为传入的n_jobs参数
        self.ensemble = ensemble  # 初始化self.ensemble为传入的ensemble参数

    def _get_estimator(self):
        """Resolve which estimator to return (default is LinearSVC)"""
        if self.estimator is None:
            # 如果没有指定estimator，则默认返回LinearSVC分类器
            estimator = LinearSVC(random_state=0)
            if _routing_enabled():
                estimator.set_fit_request(sample_weight=True)  # 如果_routing_enabled()为True，则设置样本权重
        else:
            estimator = self.estimator  # 否则使用传入的estimator

        return estimator

    @_fit_context(
        # CalibratedClassifierCV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def predict_proba(self, X):
        """Calibrated probabilities of classification.

        This function returns calibrated probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict_proba`.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            The predicted probas.
        """
        # 确保模型已经被拟合
        check_is_fitted(self)
        # 初始化一个全零数组，用于累加各个校准分类器的预测概率
        mean_proba = np.zeros((_num_samples(X), len(self.classes_)))
        # 遍历所有的校准分类器
        for calibrated_classifier in self.calibrated_classifiers_:
            # 获取当前分类器对给定样本 X 的预测概率
            proba = calibrated_classifier.predict_proba(X)
            # 累加各个分类器的预测概率
            mean_proba += proba

        # 对累加后的概率除以分类器的数量，计算平均值
        mean_proba /= len(self.calibrated_classifiers_)

        # 返回平均概率预测值
        return mean_proba

    def predict(self, X):
        """Predict the target of new samples.

        The predicted class is the class that has the highest probability,
        and can thus be different from the prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict`.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            The predicted class.
        """
        # 确保模型已经被拟合
        check_is_fitted(self)
        # 返回具有最高概率的类别作为预测结果
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，用于包含元数据的路由信息
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # 将当前对象添加到路由请求中
            .add_self_request(self)
            # 添加估计器对象及其方法映射
            .add(
                estimator=self._get_estimator(),
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
            # 添加拆分器及其方法映射
            .add(
                splitter=self.cv,
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
        )
        # 返回包含路由信息的 MetadataRouter 对象
        return router

    def _more_tags(self):
        # 返回额外的标签信息，此处包含待修复的测试检查项
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "Due to the cross-validation and sample ordering, removing a sample"
                    " is not strictly equal to putting is weight to zero. Specific unit"
                    " tests are added for CalibratedClassifierCV specifically."
                ),
            }
        }
def _fit_classifier_calibrator_pair(
    estimator,
    X,
    y,
    train,
    test,
    method,
    classes,
    sample_weight=None,
    fit_params=None,
):
    """Fit a classifier/calibration pair on a given train/test split.

    Fit the classifier on the train set, compute its predictions on the test
    set and use the predictions as input to fit the calibrator along with the
    test labels.

    Parameters
    ----------
    estimator : estimator instance
        Cloned base estimator.

    X : array-like, shape (n_samples, n_features)
        Sample data.

    y : array-like, shape (n_samples,)
        Targets.

    train : ndarray, shape (n_train_indices,)
        Indices of the training subset.

    test : ndarray, shape (n_test_indices,)
        Indices of the testing subset.

    method : {'sigmoid', 'isotonic'}
        Method to use for calibration.

    classes : ndarray, shape (n_classes,)
        The target classes.

    sample_weight : array-like, default=None
        Sample weights for `X`.

    fit_params : dict, default=None
        Parameters to pass to the `fit` method of the underlying
        classifier.

    Returns
    -------
    calibrated_classifier : _CalibratedClassifier instance
        Instance of a calibrated classifier.
    """
    fit_params_train = _check_method_params(X, params=fit_params, indices=train)
    # Select training subset of X and y
    X_train, y_train = _safe_indexing(X, train), _safe_indexing(y, train)
    # Select testing subset of X and y
    X_test, y_test = _safe_indexing(X, test), _safe_indexing(y, test)

    # Fit the estimator on the training data
    estimator.fit(X_train, y_train, **fit_params_train)

    # Obtain predictions from the estimator on the test data
    predictions, _ = _get_response_values(
        estimator,
        X_test,
        response_method=["decision_function", "predict_proba"],
    )
    if predictions.ndim == 1:
        # Reshape binary output from `(n_samples,)` to `(n_samples, 1)`
        predictions = predictions.reshape(-1, 1)

    # Prepare sample weights for the test data
    sw_test = None if sample_weight is None else _safe_indexing(sample_weight, test)
    # Fit the calibrator using the predictions and test labels
    calibrated_classifier = _fit_calibrator(
        estimator, predictions, y_test, classes, method, sample_weight=sw_test
    )
    # Return the calibrated classifier
    return calibrated_classifier


def _fit_calibrator(clf, predictions, y, classes, method, sample_weight=None):
    """Fit calibrator(s) and return a `_CalibratedClassifier`
    instance.

    `n_classes` (i.e. `len(clf.classes_)`) calibrators are fitted.
    However, if `n_classes` equals 2, one calibrator is fitted.

    Parameters
    ----------
    clf : estimator instance
        Fitted classifier.

    predictions : array-like, shape (n_samples, n_classes) or (n_samples, 1) \
                    when binary.
        Raw predictions returned by the un-calibrated base classifier.

    y : array-like, shape (n_samples,)
        The targets.

    classes : ndarray, shape (n_classes,)
        All the prediction classes.

    method : {'sigmoid', 'isotonic'}
        The method to use for calibration.

    sample_weight : ndarray, shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    calibrated_classifier : _CalibratedClassifier instance
        Instance of a calibrated classifier.
    """
    # Logic for fitting calibrators based on number of classes
    # (not fully provided in the snippet)
    # 创建一个名为 pipeline 的 _CalibratedClassifier 实例
    pipeline : _CalibratedClassifier instance
    """
    # 使用 label_binarize 函数将 y 转换为二进制形式的 Y
    Y = label_binarize(y, classes=classes)
    # 创建一个 LabelEncoder 实例并使用 classes 进行 fit 操作
    label_encoder = LabelEncoder().fit(classes)
    # 根据 clf 对象的 classes_ 属性转换为正类索引的数组
    pos_class_indices = label_encoder.transform(clf.classes_)
    # 初始化一个空列表用于存储校准器
    calibrators = []
    # 遍历正类索引数组和预测矩阵的每一列
    for class_idx, this_pred in zip(pos_class_indices, predictions.T):
        # 根据 method 的值选择不同的校准器
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
        else:  # "sigmoid"
            calibrator = _SigmoidCalibration()
        # 使用当前预测值、对应的二进制标签 Y[:, class_idx] 和样本权重进行校准器训练
        calibrator.fit(this_pred, Y[:, class_idx], sample_weight)
        # 将训练好的校准器加入到 calibrators 列表中
        calibrators.append(calibrator)

    # 创建一个 _CalibratedClassifier 实例作为 pipeline，包括原始分类器 clf、校准器列表、method 和 classes
    pipeline = _CalibratedClassifier(clf, calibrators, method=method, classes=classes)
    # 返回构建好的 pipeline 对象
    return pipeline
# 定义一个名为 _CalibratedClassifier 的类，用于管理已经拟合的分类器及其拟合的校准器链式管道。

class _CalibratedClassifier:
    """Pipeline-like chaining a fitted classifier and its fitted calibrators.

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier.

    calibrators : list of fitted estimator instances
        List of fitted calibrators (either 'IsotonicRegression' or
        '_SigmoidCalibration'). The number of calibrators equals the number of
        classes. However, if there are 2 classes, the list contains only one
        fitted calibrator.

    classes : array-like of shape (n_classes,)
        All the prediction classes.

    method : {'sigmoid', 'isotonic'}, default='sigmoid'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach based on isotonic regression.
    """

    def __init__(self, estimator, calibrators, *, classes, method="sigmoid"):
        # 初始化方法，设置对象的属性
        self.estimator = estimator
        self.calibrators = calibrators
        self.classes = classes
        self.method = method
    def predict_proba(self, X):
        """Calculate calibrated probabilities.

        Calculates classification calibrated probabilities
        for each class, in a one-vs-all manner, for `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The sample data.

        Returns
        -------
        proba : array, shape (n_samples, n_classes)
            The predicted probabilities. Can be exact zeros.
        """
        # 获取预测结果和额外信息
        predictions, _ = _get_response_values(
            self.estimator,
            X,
            response_method=["decision_function", "predict_proba"],
        )
        if predictions.ndim == 1:
            # 将二元输出从 `(n_samples,)` 重塑为 `(n_samples, 1)`
            predictions = predictions.reshape(-1, 1)

        # 获取类别数量
        n_classes = len(self.classes)

        # 创建并适配标签编码器
        label_encoder = LabelEncoder().fit(self.classes)
        pos_class_indices = label_encoder.transform(self.estimator.classes_)

        # 初始化概率数组
        proba = np.zeros((_num_samples(X), n_classes))
        for class_idx, this_pred, calibrator in zip(
            pos_class_indices, predictions.T, self.calibrators
        ):
            if n_classes == 2:
                # 当为二元分类时，`predictions` 只包含对 clf.classes_[1] 的预测，
                # 但 `pos_class_indices` 对应的是 0
                class_idx += 1
            # 使用校准器预测每个类别的概率
            proba[:, class_idx] = calibrator.predict(this_pred)

        # 归一化概率
        if n_classes == 2:
            proba[:, 0] = 1.0 - proba[:, 1]
        else:
            denominator = np.sum(proba, axis=1)[:, np.newaxis]
            # 处理极端情况，如果对于某个样本，每个类别的校准器返回的概率均为零，
            # 则使用均匀分布作为备选方案
            uniform_proba = np.full_like(proba, 1 / n_classes)
            proba = np.divide(
                proba, denominator, out=uniform_proba, where=denominator != 0
            )

        # 处理预测概率微小超过 1.0 的情况
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba
# The max_abs_prediction_threshold was approximated using
# logit(np.finfo(np.float64).eps) which is about -36
def _sigmoid_calibration(
    predictions, y, sample_weight=None, max_abs_prediction_threshold=30
):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    predictions : ndarray of shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray of shape (n_samples,)
        The targets.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    a : float
        The slope.

    b : float
        The intercept.

    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    # Ensure predictions and y are converted to 1-dimensional arrays
    predictions = column_or_1d(predictions)
    y = column_or_1d(y)

    F = predictions  # F follows Platt's notations

    scale_constant = 1.0
    max_prediction = np.max(np.abs(F))

    # If the predictions have large values we scale them in order to bring
    # them within a suitable range. This has no effect on the final
    # (prediction) result because linear models like Logisitic Regression
    # without a penalty are invariant to multiplying the features by a
    # constant.
    if max_prediction >= max_abs_prediction_threshold:
        scale_constant = max_prediction
        # We rescale the features in a copy: inplace rescaling could confuse
        # the caller and make the code harder to reason about.
        F = F / scale_constant

    # Bayesian priors (see Platt end of section 2.2):
    # It corresponds to the number of samples, taking into account the
    # `sample_weight`.
    mask_negative_samples = y <= 0
    if sample_weight is not None:
        prior0 = (sample_weight[mask_negative_samples]).sum()
        prior1 = (sample_weight[~mask_negative_samples]).sum()
    else:
        prior0 = float(np.sum(mask_negative_samples))
        prior1 = y.shape[0] - prior0
    # T is a vector used for calibration based on the class labels
    T = np.zeros_like(y, dtype=predictions.dtype)
    T[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
    T[y <= 0] = 1.0 / (prior0 + 2.0)

    # Initialize a half binomial loss function for use in calibration
    bin_loss = HalfBinomialLoss()
    # 定义函数 `loss_grad`，计算损失和梯度
    def loss_grad(AB):
        # 使用公式进行原始预测计算，确保数据类型与预测数据相同
        raw_prediction = -(AB[0] * F + AB[1]).astype(dtype=predictions.dtype)
        # 调用二元损失函数的损失和梯度计算
        l, g = bin_loss.loss_gradient(
            y_true=T,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )
        # 计算总损失
        loss = l.sum()
        # 根据 SciPy 的版本，对梯度进行类型转换以支持不同的版本需求
        # 在 SciPy >= 1.11.2，LBFGS 实现会自动转换为 float64
        # 这里手动转换为 float64 是为了支持 SciPy < 1.11.2
        grad = np.asarray([-g @ F, -g.sum()], dtype=np.float64)
        return loss, grad

    # 初始化参数 AB0，其中第一个元素为 0.0，第二个元素为对数变换的先验比率
    AB0 = np.array([0.0, log((prior0 + 1.0) / (prior1 + 1.0))])

    # 使用 minimize 函数进行优化，调用 loss_grad 函数计算损失和梯度
    opt_result = minimize(
        loss_grad,
        AB0,
        method="L-BFGS-B",
        jac=True,
        options={
            "gtol": 1e-6,
            "ftol": 64 * np.finfo(float).eps,
        },
    )
    # 获取优化后的参数 AB_
    AB_ = opt_result.x

    # 将调整后的乘法参数重新转换回原始输入特征的比例尺
    # 偏移参数不需要重新缩放，因为我们没有对输出变量进行缩放
    return AB_[0] / scale_constant, AB_[1]
class _SigmoidCalibration(RegressorMixin, BaseEstimator):
    """Sigmoid regression model.

    Attributes
    ----------
    a_ : float
        The slope.

    b_ : float
        The intercept.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # Ensure X and y are 1-dimensional
        X = column_or_1d(X)
        y = column_or_1d(y)
        # Ensure X and y are indexable
        X, y = indexable(X, y)

        # Perform sigmoid calibration using helper function
        self.a_, self.b_ = _sigmoid_calibration(X, y, sample_weight)
        return self

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : ndarray of shape (n_samples,)
            The predicted data.
        """
        # Ensure T is 1-dimensional
        T = column_or_1d(T)
        # Compute sigmoid prediction
        return expit(-(self.a_ * T + self.b_))


@validate_params(
    {
        "y_true": ["array-like"],
        "y_prob": ["array-like"],
        "pos_label": [Real, str, "boolean", None],
        "n_bins": [Interval(Integral, 1, None, closed="left")],
        "strategy": [StrOptions({"uniform", "quantile"})],
    },
    prefer_skip_nested_validation=True,
)
def calibration_curve(
    y_true,
    y_prob,
    *,
    pos_label=None,
    n_bins=5,
    strategy="uniform",
):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    pos_label : int, float, bool or str, default=None
        The label of the positive class.

        .. versionadded:: 1.1

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    tuple
        Tuple of arrays (true_probabilities, predicted_probabilities).
    """
    # prob_true : ndarray of shape (n_bins,) or smaller
    # 每个分箱中正类样本的比例，形状为 (n_bins,) 或更小

    # prob_pred : ndarray of shape (n_bins,) or smaller
    # 每个分箱中的平均预测概率，形状为 (n_bins,) 或更小

    # References
    # ----------
    # Alexandru Niculescu-Mizil 和 Rich Caruana (2005) 在第22届国际机器学习会议(ICML)上的论文
    # 《Predicting Good Probabilities With Supervised Learning》。
    # 见第4节《Qualitative Analysis of Predictions》。

    # Examples
    # --------
    # >>> import numpy as np
    # >>> from sklearn.calibration import calibration_curve
    # >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    # >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    # >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    # >>> prob_true
    # array([0. , 0.5, 1. ])
    # >>> prob_pred
    # array([0.2  , 0.525, 0.85 ])

    y_true = column_or_1d(y_true)
    # 确保 y_true 是一维数组或列向量

    y_prob = column_or_1d(y_prob)
    # 确保 y_prob 是一维数组或列向量

    check_consistent_length(y_true, y_prob)
    # 检查 y_true 和 y_prob 的长度是否一致

    pos_label = _check_pos_label_consistency(pos_label, y_true)
    # 检查并确保正类标签的一致性

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")
    # 如果 y_prob 中有超出 [0, 1] 范围的值，则引发 ValueError

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    # 如果标签数大于2，则引发 ValueError，只支持二分类

    y_true = y_true == pos_label
    # 将 y_true 转换为布尔数组，表示是否为正类

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )
    # 根据策略确定分箱的边界

    binids = np.searchsorted(bins[1:-1], y_prob)
    # 根据分箱边界将 y_prob 分箱

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    # 计算每个分箱中预测概率的加权和

    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    # 计算每个分箱中真实正例的加权和

    bin_total = np.bincount(binids, minlength=len(bins))
    # 计算每个分箱中样本总数

    nonzero = bin_total != 0
    # 找出非零分箱

    prob_true = bin_true[nonzero] / bin_total[nonzero]
    # 计算每个非零分箱中的正类样本比例

    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    # 计算每个非零分箱中的平均预测概率

    return prob_true, prob_pred
# 定义一个 CalibrationDisplay 类，用于显示校准曲线（也称可靠性图表）的可视化效果
class CalibrationDisplay(_BinaryClassifierCurveDisplayMixin):
    """Calibration curve (also known as reliability diagram) visualization.

    It is recommended to use
    :func:`~sklearn.calibration.CalibrationDisplay.from_estimator` or
    :func:`~sklearn.calibration.CalibrationDisplay.from_predictions`
    to create a `CalibrationDisplay`. All parameters are stored as attributes.

    Read more about calibration in the :ref:`User Guide <calibration>` and
    more about the scikit-learn visualization API in :ref:`visualizations`.

    .. versionadded:: 1.0

    Parameters
    ----------
    prob_true : ndarray of shape (n_bins,)
        The proportion of samples whose class is the positive class (fraction
        of positives), in each bin.

    prob_pred : ndarray of shape (n_bins,)
        The mean predicted probability in each bin.

    y_prob : ndarray of shape (n_samples,)
        Probability estimates for the positive class, for each sample.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The positive class when computing the calibration curve.
        By default, `pos_label` is set to `estimators.classes_[1]` when using
        `from_estimator` and set to 1 when using `from_predictions`.

        .. versionadded:: 1.1

    Attributes
    ----------
    line_ : matplotlib Artist
        Calibration curve.

    ax_ : matplotlib Axes
        Axes with calibration curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    calibration_curve : Compute true and predicted probabilities for a
        calibration curve.
    CalibrationDisplay.from_predictions : Plot calibration curve using true
        and predicted labels.
    CalibrationDisplay.from_estimator : Plot calibration curve using an
        estimator and data.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.calibration import calibration_curve, CalibrationDisplay
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> clf = LogisticRegression(random_state=0)
    >>> clf.fit(X_train, y_train)
    LogisticRegression(random_state=0)
    >>> y_prob = clf.predict_proba(X_test)[:, 1]
    >>> prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    >>> disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
    >>> disp.plot()
    <...>
    """

    # 初始化方法，接受校准曲线绘制所需的各种参数并存储为对象属性
    def __init__(
        self, prob_true, prob_pred, y_prob, *, estimator_name=None, pos_label=None
    ):
        self.prob_true = prob_true  # 正样本比例数组，每个箱子内正样本占比
        self.prob_pred = prob_pred  # 每个箱子内的平均预测概率
        self.y_prob = y_prob  # 每个样本的正样本概率估计
        self.estimator_name = estimator_name  # 模型名称，如果为None，则不显示模型名称
        self.pos_label = pos_label  # 计算校准曲线时的正类标签，默认为 None
    def plot(self, *, ax=None, name=None, ref_line=True, **kwargs):
        """Plot visualization.

        Extra keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name for labeling curve. If `None`, use `estimator_name` if
            not `None`, otherwise no labeling is shown.

        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.

        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`
            Object that stores computed values.
        """
        # Validate and set plot parameters like axes and name
        self.ax_, self.figure_, name = self._validate_plot_params(ax=ax, name=name)

        # Create a string indicating the positive class label, if specified
        info_pos_label = (
            f"(Positive class: {self.pos_label})" if self.pos_label is not None else ""
        )

        # Define default line properties for the plot
        line_kwargs = {"marker": "s", "linestyle": "-"}
        if name is not None:
            line_kwargs["label"] = name
        line_kwargs.update(**kwargs)

        # Add a reference line for perfect calibration if requested and not already plotted
        ref_line_label = "Perfectly calibrated"
        existing_ref_line = ref_line_label in self.ax_.get_legend_handles_labels()[1]
        if ref_line and not existing_ref_line:
            self.ax_.plot([0, 1], [0, 1], "k:", label=ref_line_label)

        # Plot the main line based on predicted and true probabilities
        self.line_ = self.ax_.plot(self.prob_pred, self.prob_true, **line_kwargs)[0]

        # Always show the legend, at least for the reference line
        self.ax_.legend(loc="lower right")

        # Set labels for x and y axes based on the plot's context
        xlabel = f"Mean predicted probability {info_pos_label}"
        ylabel = f"Fraction of positives {info_pos_label}"
        self.ax_.set(xlabel=xlabel, ylabel=ylabel)

        # Return the modified instance of CalibrationDisplay for further operations
        return self



    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        n_bins=5,
        strategy="uniform",
        pos_label=None,
        name=None,
        ref_line=True,
        ax=None,
        **kwargs,
    ):
        """
        Construct a CalibrationDisplay object from an estimator's predictions.

        Parameters
        ----------
        cls : class
            Class of the object being constructed.
        
        estimator : object
            Estimator object implementing `predict_proba` method.
        
        X : array-like of shape (n_samples, n_features)
            Input data used to generate predictions.
        
        y : array-like of shape (n_samples,)
            True labels or target values.
        
        n_bins : int, default=5
            Number of bins to discretize the predicted probabilities.
        
        strategy : {'uniform', 'quantile'}, default='uniform'
            Strategy used to discretize the predicted probabilities.
        
        pos_label : int or str, default=None
            Positive class label. If None, the estimator's default will be used.
        
        name : str, default=None
            Name for labeling curve in the plot. If None, no label will be shown.
        
        ref_line : bool, default=True
            If True, a reference line for perfect calibration will be plotted.
        
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If None, a new figure and axes will be created.
        
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`
            Object that stores computed values.
        """



    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_prob,
        *,
        n_bins=5,
        strategy="uniform",
        pos_label=None,
        name=None,
        ref_line=True,
        ax=None,
        **kwargs,
    ):
        """
        Construct a CalibrationDisplay object from predictions.

        Parameters
        ----------
        cls : class
            Class of the object being constructed.
        
        y_true : array-like of shape (n_samples,)
            True labels or target values.
        
        y_prob : array-like of shape (n_samples,)
            Predicted probabilities or confidence scores.
        
        n_bins : int, default=5
            Number of bins to discretize the predicted probabilities.
        
        strategy : {'uniform', 'quantile'}, default='uniform'
            Strategy used to discretize the predicted probabilities.
        
        pos_label : int or str, default=None
            Positive class label. If None, the default from the predictions will be used.
        
        name : str, default=None
            Name for labeling curve in the plot. If None, no label will be shown.
        
        ref_line : bool, default=True
            If True, a reference line for perfect calibration will be plotted.
        
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If None, a new figure and axes will be created.
        
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`
            Object that stores computed values.
        """
```