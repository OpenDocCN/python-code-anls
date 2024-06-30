# `D:\src\scipysrc\scikit-learn\sklearn\utils\_mocking.py`

```
import numpy as np  # 导入NumPy库，用于科学计算

from ..base import BaseEstimator, ClassifierMixin  # 导入基础估计器和分类器混合类
from ..utils._metadata_requests import RequestMethod  # 导入请求方法
from .metaestimators import available_if  # 导入元估计器的条件可用性
from .validation import (  # 导入验证模块的多个函数
    _check_sample_weight,
    _num_samples,
    check_array,
    check_is_fitted,
    check_random_state,
)

class ArraySlicingWrapper:
    """
    Parameters
    ----------
    array
    """

    def __init__(self, array):
        self.array = array  # 初始化数组属性

    def __getitem__(self, aslice):
        return MockDataFrame(self.array[aslice])  # 返回模拟数据框，基于数组切片


class MockDataFrame:
    """
    Parameters
    ----------
    array
    """

    # have shape and length but don't support indexing.

    def __init__(self, array):
        self.array = array  # 初始化数组属性
        self.values = array  # 设置值属性为数组
        self.shape = array.shape  # 设置形状属性为数组的形状
        self.ndim = array.ndim  # 设置维度属性为数组的维度
        # ugly hack to make iloc work.
        self.iloc = ArraySlicingWrapper(array)  # 使用数组切片包装器使iloc工作

    def __len__(self):
        return len(self.array)  # 返回数组的长度

    def __array__(self, dtype=None):
        # Pandas data frames also are array-like: we want to make sure that
        # input validation in cross-validation does not try to call that
        # method.
        return self.array  # 返回数组，确保在交叉验证中不调用此方法

    def __eq__(self, other):
        return MockDataFrame(self.array == other.array)  # 比较两个模拟数据框是否相等

    def __ne__(self, other):
        return not self == other  # 比较两个模拟数据框是否不相等

    def take(self, indices, axis=0):
        return MockDataFrame(self.array.take(indices, axis=axis))  # 返回根据指定轴和索引获取的数据框


class CheckingClassifier(ClassifierMixin, BaseEstimator):
    """Dummy classifier to test pipelining and meta-estimators.

    Checks some property of `X` and `y`in fit / predict.
    This allows testing whether pipelines / cross-validation or metaestimators
    changed the input.

    Can also be used to check if `fit_params` are passed correctly, and
    to force a certain score to be returned.

    Parameters
    ----------
    check_y, check_X : callable, default=None
        The callable used to validate `X` and `y`. These callable should return
        a bool where `False` will trigger an `AssertionError`. If `None`, the
        data is not validated. Default is `None`.

    check_y_params, check_X_params : dict, default=None
        The optional parameters to pass to `check_X` and `check_y`. If `None`,
        then no parameters are passed in.

    methods_to_check : "all" or list of str, default="all"
        The methods in which the checks should be applied. By default,
        all checks will be done on all methods (`fit`, `predict`,
        `predict_proba`, `decision_function` and `score`).

    foo_param : int, default=0
        A `foo` param. When `foo > 1`, the output of :meth:`score` will be 1
        otherwise it is 0.

    expected_sample_weight : bool, default=False
        Whether to check if a valid `sample_weight` was passed to `fit`.

    expected_fit_params : list of str, default=None
        A list of the expected parameters given when calling `fit`.

    Attributes
    ----------
    
    classes_ : int
        The number of classes seen during the fitting of the classifier.

    n_features_in_ : int
        The number of features seen during the fitting of the classifier.

    Examples
    --------
    >>> from sklearn.utils._mocking import CheckingClassifier

    This helper allows assertions regarding `X` or `y` specificities. Here,
    `check_X` or `check_y` is expected to return a boolean.

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = CheckingClassifier(check_X=lambda x: x.shape == (150, 4))
    >>> clf.fit(X, y)
    CheckingClassifier(...)

    We can also provide a check that might raise an error. In this case,
    `check_X` is expected to return `X` and `check_y` to return `y`.

    >>> from sklearn.utils import check_array
    >>> clf = CheckingClassifier(check_X=check_array)
    >>> clf.fit(X, y)
    CheckingClassifier(...)
    """

    def __init__(
        self,
        *,
        check_y=None,
        check_y_params=None,
        check_X=None,
        check_X_params=None,
        methods_to_check="all",
        foo_param=0,
        expected_sample_weight=None,
        expected_fit_params=None,
        random_state=None,
    ):
        # Initialize the CheckingClassifier instance with specified parameters
        self.check_y = check_y
        self.check_y_params = check_y_params
        self.check_X = check_X
        self.check_X_params = check_X_params
        self.methods_to_check = methods_to_check
        self.foo_param = foo_param
        self.expected_sample_weight = expected_sample_weight
        self.expected_fit_params = expected_fit_params
        self.random_state = random_state

    def _check_X_y(self, X, y=None, should_be_fitted=True):
        """Validate X and y and perform additional checks.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The dataset.
            `X` is checked only if `check_X` is not `None` (default is None).
        y : array-like of shape (n_samples), default=None
            The corresponding targets, default is `None`.
            `y` is checked only if `check_y` is not `None` (default is None).
        should_be_fitted : bool, default=True
            Whether or not the classifier should already be fitted.
            Default is True.

        Returns
        -------
        X, y
            Validated and possibly transformed X and y.
        """
        if should_be_fitted:
            check_is_fitted(self)  # Check if the classifier is fitted

        # Check X if a validation function is provided
        if self.check_X is not None:
            params = {} if self.check_X_params is None else self.check_X_params
            checked_X = self.check_X(X, **params)
            if isinstance(checked_X, (bool, np.bool_)):
                assert checked_X  # Ensure the result is True
            else:
                X = checked_X  # Use the checked result of X

        # Check y if provided and a validation function is provided
        if y is not None and self.check_y is not None:
            params = {} if self.check_y_params is None else self.check_y_params
            checked_y = self.check_y(y, **params)
            if isinstance(checked_y, (bool, np.bool_)):
                assert checked_y  # Ensure the result is True
            else:
                y = checked_y  # Use the checked result of y

        return X, y  # Return validated X and y
    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_outputs) or (n_samples,), \
                default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Returns
        -------
        self
        """
        # 断言训练集 X 和标签 y 的样本数相等
        assert _num_samples(X) == _num_samples(y)

        # 如果方法检查方式为 "all" 或包含 "fit"，则进行输入数据验证
        if self.methods_to_check == "all" or "fit" in self.methods_to_check:
            # 调用 _check_X_y 方法验证输入数据 X 和标签 y
            X, y = self._check_X_y(X, y, should_be_fitted=False)

        # 设置类属性 n_features_in_ 为 X 的特征数
        self.n_features_in_ = np.shape(X)[1]

        # 计算并设置类属性 classes_ 为标签 y 的唯一值
        self.classes_ = np.unique(check_array(y, ensure_2d=False, allow_nd=True))

        # 如果期望的拟合参数不为空，则验证是否提供了所有期望的参数
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(fit_params)
            if missing:
                raise AssertionError(
                    f"Expected fit parameter(s) {list(missing)} not seen."
                )
            # 检查每个拟合参数的样本数是否与 X 的样本数相等
            for key, value in fit_params.items():
                if _num_samples(value) != _num_samples(X):
                    raise AssertionError(
                        f"Fit parameter {key} has length {_num_samples(value)}"
                        f"; expected {_num_samples(X)}."
                    )

        # 如果期望的样本权重不为空，则确保传入了样本权重 sample_weight
        if self.expected_sample_weight:
            if sample_weight is None:
                raise AssertionError("Expected sample_weight to be passed")
            # 检查样本权重的合法性
            _check_sample_weight(sample_weight, X)

        # 返回对象本身
        return self

    def predict(self, X):
        """Predict the first class seen in `classes_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds : ndarray of shape (n_samples,)
            Predictions of the first class seens in `classes_`.
        """
        # 如果方法检查方式为 "all" 或包含 "predict"，则验证输入数据 X
        if self.methods_to_check == "all" or "predict" in self.methods_to_check:
            # 调用 _check_X_y 方法验证输入数据 X
            X, y = self._check_X_y(X)

        # 生成一个随机数发生器对象 rng
        rng = check_random_state(self.random_state)

        # 返回从 classes_ 中随机选择的预测结果
        return rng.choice(self.classes_, size=_num_samples(X))
    def predict_proba(self, X):
        """Predict probabilities for each class.

        Here, the dummy classifier will provide a probability of 1 for the
        first class of `classes_` and 0 otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The probabilities for each sample and class.
        """
        # 检查是否需要检查所有方法或仅检查预测概率方法
        if self.methods_to_check == "all" or "predict_proba" in self.methods_to_check:
            # 调用内部方法检查输入数据是否符合预期
            X, y = self._check_X_y(X)
        # 使用随机状态生成器创建随机数据
        rng = check_random_state(self.random_state)
        # 生成服从标准正态分布的随机数，形状为 (样本数, 类别数)
        proba = rng.randn(_num_samples(X), len(self.classes_))
        # 取绝对值，并按行对每个样本的数据归一化
        proba = np.abs(proba, out=proba)
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
        return proba

    def decision_function(self, X):
        """Confidence score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        decision : ndarray of shape (n_samples,) if n_classes == 2\
                else (n_samples, n_classes)
            Confidence score.
        """
        # 检查是否需要检查所有方法或仅检查决策函数方法
        if (
            self.methods_to_check == "all"
            or "decision_function" in self.methods_to_check
        ):
            # 调用内部方法检查输入数据是否符合预期
            X, y = self._check_X_y(X)
        # 使用随机状态生成器创建不同形状的随机数据，取决于类别数目
        rng = check_random_state(self.random_state)
        if len(self.classes_) == 2:
            # 对于二元分类器，置信度分数与 classes_[1] 相关，因此应为空。
            return rng.randn(_num_samples(X))
        else:
            return rng.randn(_num_samples(X), len(self.classes_))

    def score(self, X=None, Y=None):
        """Fake score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Y : array-like of shape (n_samples, n_output) or (n_samples,)
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
            Either 0 or 1 depending of `foo_param` (i.e. `foo_param > 1 =>
            score=1` otherwise `score=0`).
        """
        # 检查是否需要检查所有方法或仅检查得分方法
        if self.methods_to_check == "all" or "score" in self.methods_to_check:
            # 调用内部方法检查输入数据是否符合预期
            self._check_X_y(X, Y)
        # 根据 foo_param 的值返回得分，如果大于 1 则得分为 1.0，否则为 0.0
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    def _more_tags(self):
        """Additional tags for this estimator."""
        # 返回额外的标签，用于跳过测试和 X 类型为 1 维标签
        return {"_skip_test": True, "X_types": ["1dlabel"]}
# 禁用 CheckingClassifier 的键验证，因为我们希望能够使用任意的 fit_params 调用 fit 并记录它们。
# 如果不进行这个更改，由于这些任意参数不被期望，我们将会收到错误。
CheckingClassifier.set_fit_request = RequestMethod(  # type: ignore
    name="fit", keys=[], validate_keys=False
)

# 定义一个包装器类 NoSampleWeightWrapper，用于封装的估计器不会暴露 sample_weight 参数。
class NoSampleWeightWrapper(BaseEstimator):
    """Wrap estimator which will not expose `sample_weight`.

    Parameters
    ----------
    est : estimator, default=None
        要封装的估计器。
    """

    def __init__(self, est=None):
        self.est = est

    def fit(self, X, y):
        return self.est.fit(X, y)

    def predict(self, X):
        return self.est.predict(X)

    def predict_proba(self, X):
        return self.est.predict_proba(X)

    def _more_tags(self):
        return {"_skip_test": True}


# 定义一个装饰器函数 _check_response，用于检查是否可以调用特定的响应方法。
def _check_response(method):
    def check(self):
        return self.response_methods is not None and method in self.response_methods

    return check


# 定义一个名为 _MockEstimatorOnOffPrediction 的估计器类，可以控制预测方法的开启和关闭。
class _MockEstimatorOnOffPrediction(BaseEstimator):
    """Estimator for which we can turn on/off the prediction methods.

    Parameters
    ----------
    response_methods: list of \
            {"predict", "predict_proba", "decision_function"}, default=None
        包含估计器实现的响应方法的列表。当响应在列表中时，调用时将返回响应方法的名称。
        否则，将引发 `AttributeError`。这允许像常规估计器一样使用 `getattr`。
        默认情况下，不模拟任何响应方法。
    """

    def __init__(self, response_methods=None):
        self.response_methods = response_methods

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    @available_if(_check_response("predict"))
    def predict(self, X):
        return "predict"

    @available_if(_check_response("predict_proba"))
    def predict_proba(self, X):
        return "predict_proba"

    @available_if(_check_response("decision_function"))
    def decision_function(self, X):
        return "decision_function"
```