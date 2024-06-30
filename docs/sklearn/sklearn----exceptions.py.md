# `D:\src\scipysrc\scikit-learn\sklearn\exceptions.py`

```
# 定义一个列表，包含了所有会公开的自定义警告和错误类的名称
__all__ = [
    "NotFittedError",
    "ConvergenceWarning",
    "DataConversionWarning",
    "DataDimensionalityWarning",
    "EfficiencyWarning",
    "FitFailedWarning",
    "SkipTestWarning",
    "UndefinedMetricWarning",
    "PositiveSpectrumWarning",
    "UnsetMetadataPassedError",
]

# 定义一个自定义异常类 UnsetMetadataPassedError，用于在传递了未被明确请求的元数据时抛出异常
class UnsetMetadataPassedError(ValueError):
    """Exception class to raise if a metadata is passed which is not explicitly \
        requested (metadata=True) or not requested (metadata=False).

    .. versionadded:: 1.3

    Parameters
    ----------
    message : str
        The message

    unrequested_params : dict
        A dictionary of parameters and their values which are provided but not
        requested.

    routed_params : dict
        A dictionary of routed parameters.
    """
    
    # 初始化方法，接受一个 message 参数作为异常消息，还有两个额外的参数 unrequested_params 和 routed_params
    def __init__(self, *, message, unrequested_params, routed_params):
        # 调用父类的初始化方法，将 message 传递给父类
        super().__init__(message)
        # 将 unrequested_params 赋值给对象的属性 unrequested_params
        self.unrequested_params = unrequested_params
        # 将 routed_params 赋值给对象的属性 routed_params
        self.routed_params = routed_params


# 定义一个异常类 NotFittedError，用于在模型未拟合时使用模型进行操作时抛出异常
class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This LinearSVC instance is not fitted yet. Call 'fit' with
    appropriate arguments before using this estimator."...)

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


# 定义一个警告类 ConvergenceWarning，用于捕获收敛问题的警告信息
class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


# 定义一个警告类 DataConversionWarning，用于通知代码中发生的隐式数据转换
class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.

    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.

    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


# 定义一个警告类 DataDimensionalityWarning，用于通知数据维度可能存在问题的警告
class DataDimensionalityWarning(UserWarning):
    """Custom warning to notify potential issues with data dimensionality.

    For example, in random projection, this warning is raised when the
    number of components, which quantifies the dimensionality of the target
    projection space, is higher than the number of features, which quantifies
    the dimensionality of the original source space, to imply that the
    dimensionality of the problem will not be reduced.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
# 定义一个自定义警告类，用于通知用户计算效率可能不佳的警告信息
class EfficiencyWarning(UserWarning):
    """Warning used to notify the user of inefficient computation.

    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.

    .. versionadded:: 0.18
    """


# 定义一个自定义警告类，用于在拟合估算器时出现错误时通知用户
class FitFailedWarning(RuntimeWarning):
    """Warning class used if there is an error while fitting the estimator.

    This Warning is used in meta estimators GridSearchCV and RandomizedSearchCV
    and the cross-validation helper function cross_val_score to warn when there
    is an error while fitting the estimator.

    .. versionchanged:: 0.18
       Moved from sklearn.cross_validation.
    """


# 定义一个自定义警告类，用于通知用户跳过了某些测试
class SkipTestWarning(UserWarning):
    """Warning class used to notify the user of a test that was skipped.

    For example, one of the estimator checks requires a pandas import.
    If the pandas package cannot be imported, the test will be skipped rather
    than register as a failure.
    """


# 定义一个自定义警告类，用于在指标无效时通知用户
class UndefinedMetricWarning(UserWarning):
    """Warning used when the metric is invalid

    .. versionchanged:: 0.18
       Moved from sklearn.base.
    """


# 定义一个自定义警告类，用于在正定半定矩阵的特征值存在问题时通知用户
class PositiveSpectrumWarning(UserWarning):
    """Warning raised when the eigenvalues of a PSD matrix have issues

    This warning is typically raised by ``_check_psd_eigenvalues`` when the
    eigenvalues of a positive semidefinite (PSD) matrix such as a gram matrix
    (kernel) present significant negative eigenvalues, or bad conditioning i.e.
    very small non-zero eigenvalues compared to the largest eigenvalue.

    .. versionadded:: 0.22
    """


# 定义一个自定义警告类，用于在反序列化估算器时版本不一致时通知用户
class InconsistentVersionWarning(UserWarning):
    """Warning raised when an estimator is unpickled with a inconsistent version.

    Parameters
    ----------
    estimator_name : str
        Estimator name.

    current_sklearn_version : str
        Current scikit-learn version.

    original_sklearn_version : str
        Original scikit-learn version.
    """

    def __init__(
        self, *, estimator_name, current_sklearn_version, original_sklearn_version
    ):
        self.estimator_name = estimator_name
        self.current_sklearn_version = current_sklearn_version
        self.original_sklearn_version = original_sklearn_version

    def __str__(self):
        return (
            f"Trying to unpickle estimator {self.estimator_name} from version"
            f" {self.original_sklearn_version} when "
            f"using version {self.current_sklearn_version}. This might lead to breaking"
            " code or "
            "invalid results. Use at your own risk. "
            "For more info please refer to:\n"
            "https://scikit-learn.org/stable/model_persistence.html"
            "#security-maintainability-limitations"
        )
```