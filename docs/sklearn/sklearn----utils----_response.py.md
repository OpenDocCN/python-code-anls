# `D:\src\scipysrc\scikit-learn\sklearn\utils\_response.py`

```
# 导入必要的库
import numpy as np

# 导入基础函数和验证模块
from ..base import is_classifier
from .multiclass import type_of_target
from .validation import _check_response_method, check_is_fitted

# 处理当响应方法为 `predict_proba` 时获取响应值的函数
def _process_predict_proba(*, y_pred, target_type, classes, pos_label):
    """Get the response values when the response method is `predict_proba`.

    This function process the `y_pred` array in the binary and multi-label cases.
    In the binary case, it selects the column corresponding to the positive
    class. In the multi-label case, it stacks the predictions if they are not
    in the "compressed" format `(n_samples, n_outputs)`.

    Parameters
    ----------
    y_pred : ndarray
        Output of `estimator.predict_proba`. The shape depends on the target type:

        - for binary classification, it is a 2d array of shape `(n_samples, 2)`;
        - for multiclass classification, it is a 2d array of shape
          `(n_samples, n_classes)`;
        - for multilabel classification, it is either a list of 2d arrays of shape
          `(n_samples, 2)` (e.g. `RandomForestClassifier` or `KNeighborsClassifier`) or
          an array of shape `(n_samples, n_outputs)` (e.g. `MLPClassifier` or
          `RidgeClassifier`).

    target_type : {"binary", "multiclass", "multilabel-indicator"}
        Type of the target.

    classes : ndarray of shape (n_classes,) or list of such arrays
        Class labels as reported by `estimator.classes_`.

    pos_label : int, float, bool or str
        Only used with binary and multiclass targets.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,), (n_samples, n_classes) or \
            (n_samples, n_output)
        Compressed predictions format as requested by the metrics.
    """
    # 检查二分类情况下 `y_pred` 的形状，确保有两列
    if target_type == "binary" and y_pred.shape[1] < 2:
        # 如果是单类分类器，则抛出值错误
        raise ValueError(
            f"Got predict_proba of shape {y_pred.shape}, but need "
            "classifier with two classes."
        )

    # 处理二分类情况，选择对应正类的列
    if target_type == "binary":
        col_idx = np.flatnonzero(classes == pos_label)[0]
        return y_pred[:, col_idx]
    # 处理多标签指示的情况，使用压缩格式 `(n_samples, n_output)`
    elif target_type == "multilabel-indicator":
        # 如果 `y_pred` 是列表，即 `(n_samples, 2)` 的数组列表
        if isinstance(y_pred, list):
            return np.vstack([p[:, -1] for p in y_pred]).T
        else:
            # 如果 `y_pred` 是形状为 `(n_samples, n_outputs)` 的数组
            return y_pred

    # 返回 `y_pred`
    return y_pred
    # 在二分类情况下，如果正类标签不是 `classes[1]`，则反转得分的符号。
    # 在多标签分类情况下，如果预测结果不是以 "(n_samples, n_outputs)" 格式压缩，则堆叠预测结果。
    
    def _invert_score_if_necessary(y_pred, target_type, classes, pos_label):
        """
        In the binary case, it inverts the sign of the score if the positive label
        is not `classes[1]`. In the multi-label case, it stacks the predictions if
        they are not in the "compressed" format `(n_samples, n_outputs)`.
    
        Parameters
        ----------
        y_pred : ndarray
            Output of `estimator.predict_proba`. The shape depends on the target type:
    
            - for binary classification, it is a 1d array of shape `(n_samples,)` where the
              sign is assuming that `classes[1]` is the positive class;
            - for multiclass classification, it is a 2d array of shape
              `(n_samples, n_classes)`;
            - for multilabel classification, it is a 2d array of shape `(n_samples,
              n_outputs)`.
    
        target_type : {"binary", "multiclass", "multilabel-indicator"}
            Type of the target.
    
        classes : ndarray of shape (n_classes,) or list of such arrays
            Class labels as reported by `estimator.classes_`.
    
        pos_label : int, float, bool or str
            Only used with binary and multiclass targets.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples,), (n_samples, n_classes) or \
                (n_samples, n_output)
            Compressed predictions format as requested by the metrics.
        """
        if target_type == "binary" and pos_label == classes[0]:
            # 如果目标类型是二分类且正类标签不是 `classes[1]`，则反转预测结果的符号
            return -1 * y_pred
        # 否则直接返回原始预测结果
        return y_pred
# 定义函数 _get_response_values，用于计算分类器、异常检测器或回归器的响应值
def _get_response_values(
    estimator,
    X,
    response_method,
    pos_label=None,
    return_response_method_used=False,
):
    """Compute the response values of a classifier, an outlier detector, or a regressor.

    The response values are predictions such that it follows the following shape:

    - for binary classification, it is a 1d array of shape `(n_samples,)`;
    - for multiclass classification, it is a 2d array of shape `(n_samples, n_classes)`;
    - for multilabel classification, it is a 2d array of shape `(n_samples, n_outputs)`;
    - for outlier detection, it is a 1d array of shape `(n_samples,)`;
    - for regression, it is a 1d array of shape `(n_samples,)`.

    If `estimator` is a binary classifier, also return the label for the
    effective positive class.

    This utility is used primarily in the displays and the scikit-learn scorers.

    .. versionadded:: 1.3

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier, outlier detector, or regressor or a
        fitted :class:`~sklearn.pipeline.Pipeline` in which the last estimator is a
        classifier, an outlier detector, or a regressor.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    response_method : {"predict_proba", "predict_log_proba", "decision_function", \
            "predict"} or list of such str
        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`predict_log_proba`,
        :term:`decision_function` or :term:`predict`). Possible choices are:

        - if `str`, it corresponds to the name to the method to return;
        - if a list of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class when computing
        the metrics. If `None` and target is 'binary', `estimators.classes_[1]` is
        considered as the positive class.

    return_response_method_used : bool, default=False
        Whether to return the response method used to compute the response
        values.

        .. versionadded:: 1.4

    Returns
    -------
    y_pred : ndarray of shape (n_samples,), (n_samples, n_classes) or \
            (n_samples, n_outputs)
        Target scores calculated from the provided `response_method`
        and `pos_label`.

    pos_label : int, float, bool, str or None
        The class considered as the positive class when computing
        the metrics. Returns `None` if `estimator` is a regressor or an outlier
        detector.

    response_method_used : str
        The response method used to compute the response values. Only returned
        if `return_response_method_used` is `True`.

        .. versionadded:: 1.4

    Raises
    ------
    """
    # 根据 estimator 的类型选择相应的方法
    if isinstance(estimator, ClassifierMixin):
        if response_method == "predict_proba":
            y_pred = estimator.predict_proba(X)
        elif response_method == "predict_log_proba":
            y_pred = estimator.predict_log_proba(X)
        elif response_method == "decision_function":
            y_pred = estimator.decision_function(X)
        else:
            y_pred = estimator.predict(X)
            if pos_label is None:
                pos_label = estimator.classes_[1]
    elif isinstance(estimator, OutlierMixin):
        y_pred = estimator.predict(X)
    else:  # Regressor case
        y_pred = estimator.predict(X)
        pos_label = None

    # 返回相应的值，根据参数决定是否返回使用的响应方法
    if return_response_method_used:
        return y_pred, pos_label, response_method
    else:
        return y_pred, pos_label
    ValueError
        If `pos_label` is not a valid label.
        If the shape of `y_pred` is not consistent for binary classifier.
        If the response method can be applied to a classifier only and
        `estimator` is a regressor.
    """
    from sklearn.base import is_classifier, is_outlier_detector  # noqa

    # 检查`estimator`是否为分类器
    if is_classifier(estimator):
        # 获取预测方法
        prediction_method = _check_response_method(estimator, response_method)
        # 获取分类器的类别
        classes = estimator.classes_
        # 确定目标类型
        target_type = type_of_target(classes)

        # 如果目标类型是二元或多类别
        if target_type in ("binary", "multiclass"):
            # 如果`pos_label`不为None且不在类别列表中，则抛出值错误异常
            if pos_label is not None and pos_label not in classes.tolist():
                raise ValueError(
                    f"pos_label={pos_label} is not a valid label: It should be "
                    f"one of {classes}"
                )
            # 如果`pos_label`为None且目标类型为二元分类，则设置`pos_label`为最后一个类别
            elif pos_label is None and target_type == "binary":
                pos_label = classes[-1]

        # 对输入数据`X`进行预测
        y_pred = prediction_method(X)

        # 如果预测方法是`predict_proba`或`predict_log_proba`
        if prediction_method.__name__ in ("predict_proba", "predict_log_proba"):
            # 对预测结果进行处理
            y_pred = _process_predict_proba(
                y_pred=y_pred,
                target_type=target_type,
                classes=classes,
                pos_label=pos_label,
            )
        # 如果预测方法是`decision_function`
        elif prediction_method.__name__ == "decision_function":
            # 对决策函数的预测结果进行处理
            y_pred = _process_decision_function(
                y_pred=y_pred,
                target_type=target_type,
                classes=classes,
                pos_label=pos_label,
            )
    # 如果`estimator`是异常检测器
    elif is_outlier_detector(estimator):
        # 获取预测方法
        prediction_method = _check_response_method(estimator, response_method)
        # 对输入数据`X`进行预测，并将`pos_label`设置为None
        y_pred, pos_label = prediction_method(X), None
    else:  # 如果`estimator`是回归器
        # 如果`response_method`不是`predict`，则抛出值错误异常
        if response_method != "predict":
            raise ValueError(
                f"{estimator.__class__.__name__} should either be a classifier to be "
                f"used with response_method={response_method} or the response_method "
                "should be 'predict'. Got a regressor with response_method="
                f"{response_method} instead."
            )
        # 获取预测方法为`predict`
        prediction_method = estimator.predict
        # 对输入数据`X`进行预测，并将`pos_label`设置为None
        y_pred, pos_label = prediction_method(X), None

    # 如果需要返回使用的响应方法信息，则返回预测结果、`pos_label`及预测方法的名称
    if return_response_method_used:
        return y_pred, pos_label, prediction_method.__name__
    # 否则，只返回预测结果和`pos_label`
    return y_pred, pos_label
def _get_response_values_binary(
    estimator, X, response_method, pos_label=None, return_response_method_used=False
):
    """Compute the response values of a binary classifier.

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a binary classifier.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    response_method : {'auto', 'predict_proba', 'decision_function'}
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. If set to 'auto',
        :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class when computing
        the metrics. By default, `estimators.classes_[1]` is
        considered as the positive class.

    return_response_method_used : bool, default=False
        Whether to return the response method used to compute the response
        values.

        .. versionadded:: 1.5

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Target scores calculated from the provided response_method
        and pos_label.

    pos_label : int, float, bool or str
        The class considered as the positive class when computing
        the metrics.

    response_method_used : str
        The response method used to compute the response values. Only returned
        if `return_response_method_used` is `True`.

        .. versionadded:: 1.5
    """
    # 错误信息，用于非二元分类器时引发错误
    classification_error = "Expected 'estimator' to be a binary classifier."

    # 检查模型是否已经拟合
    check_is_fitted(estimator)
    # 如果不是分类器，抛出 ValueError 异常
    if not is_classifier(estimator):
        raise ValueError(
            classification_error + f" Got {estimator.__class__.__name__} instead."
        )
    # 如果分类器类别数不等于 2，抛出 ValueError 异常
    elif len(estimator.classes_) != 2:
        raise ValueError(
            classification_error + f" Got {len(estimator.classes_)} classes instead."
        )

    # 如果 response_method 设置为 'auto'，则尝试使用 'predict_proba' 或 'decision_function'
    if response_method == "auto":
        response_method = ["predict_proba", "decision_function"]

    # 调用 _get_response_values 函数获取响应值
    return _get_response_values(
        estimator,
        X,
        response_method,
        pos_label=pos_label,
        return_response_method_used=return_response_method_used,
    )
```