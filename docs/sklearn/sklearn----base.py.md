# `D:\src\scipysrc\scikit-learn\sklearn\base.py`

```
`
"""Base classes for all estimators and various utility functions."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库和模块
import copy                      # 导入深拷贝函数
import functools                 # 导入函数工具模块
import inspect                   # 导入检查模块
import platform                  # 导入平台信息模块
import re                        # 导入正则表达式模块
import warnings                  # 导入警告处理模块
from collections import defaultdict  # 导入默认字典模块

import numpy as np              # 导入NumPy库

from . import __version__       # 导入当前模块的版本信息
from ._config import config_context, get_config  # 导入配置上下文和获取配置函数
from .exceptions import InconsistentVersionWarning  # 导入版本不一致警告异常
from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr  # 导入HTML文档链接混合类和估计器HTML表示
from .utils._metadata_requests import _MetadataRequester, _routing_enabled  # 导入元数据请求器和路由使能状态
from .utils._param_validation import validate_parameter_constraints  # 导入参数验证函数
from .utils._set_output import _SetOutputMixin  # 导入设置输出混合类
from .utils._tags import (  # 导入默认标签
    _DEFAULT_TAGS,
)
from .utils.fixes import _IS_32BIT  # 导入32位修复工具
from .utils.validation import (  # 导入验证工具
    _check_feature_names_in,
    _check_y,
    _generate_get_feature_names_out,
    _get_feature_names,
    _is_fitted,
    _num_features,
    check_array,
    check_is_fitted,
    check_X_y,
)


def clone(estimator, *, safe=True):
    """Construct a new unfitted estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It returns a new estimator
    with the same parameters that has not been fitted on any data.

    .. versionchanged:: 1.3
        Delegates to `estimator.__sklearn_clone__` if the method exists.

    Parameters
    ----------
    estimator : {list, tuple, set} of estimator instance or a single \
            estimator instance
        The estimator or group of estimators to be cloned.
    safe : bool, default=True
        If safe is False, clone will fall back to a deep copy on objects
        that are not estimators. Ignored if `estimator.__sklearn_clone__`
        exists.

    Returns
    -------
    estimator : object
        The deep copy of the input, an estimator if input is an estimator.

    Notes
    -----
    If the estimator's `random_state` parameter is an integer (or if the
    estimator doesn't have a `random_state` parameter), an *exact clone* is
    returned: the clone and the original estimator will give the exact same
    results. Otherwise, *statistical clone* is returned: the clone might
    return different results from the original estimator. More details can be
    found in :ref:`randomness`.

    Examples
    --------
    >>> from sklearn.base import clone
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = [[-1, 0], [0, 1], [0, -1], [1, 0]]
    >>> y = [0, 0, 1, 1]
    >>> classifier = LogisticRegression().fit(X, y)
    >>> cloned_classifier = clone(classifier)
    >>> hasattr(classifier, "classes_")
    True
    >>> hasattr(cloned_classifier, "classes_")
    False
    >>> classifier is cloned_classifier
    False
    """
    # 如果估计器有__sklearn_clone__方法并且不是一个类，则调用该方法
    if hasattr(estimator, "__sklearn_clone__") and not inspect.isclass(estimator):
        return estimator.__sklearn_clone__()
    # 否则调用参数化克隆函数
    return _clone_parametrized(estimator, safe=safe)
# 定义了一个函数 `_clone_parametrized`，用于克隆给定的 estimator 对象。
# 这是默认的克隆实现，详细信息请参考 `sklearn.base.clone` 函数的文档。

def _clone_parametrized(estimator, *, safe=True):
    """Default implementation of clone. See :func:`sklearn.base.clone` for details."""
    
    # 获取 estimator 对象的类型
    estimator_type = type(estimator)
    
    # 如果 estimator 是一个字典，则对其每个键值对进行递归克隆
    if estimator_type is dict:
        return {k: clone(v, safe=safe) for k, v in estimator.items()}
    
    # 如果 estimator 是列表、元组、集合或不具有 get_params 方法的实例化对象，则递归克隆其中的每个元素
    elif estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    
    # 如果 estimator 不具有 get_params 方法或是一个类对象，则根据安全标志进行深度复制或引发错误
    elif not hasattr(estimator, "get_params") or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            if isinstance(estimator, type):
                raise TypeError(
                    "Cannot clone object. "
                    + "You should provide an instance of "
                    + "scikit-learn estimator instead of a class."
                )
            else:
                raise TypeError(
                    "Cannot clone object '%s' (type %s): "
                    "it does not seem to be a scikit-learn "
                    "estimator as it does not implement a "
                    "'get_params' method." % (repr(estimator), type(estimator))
                )
    
    # 获取 estimator 的类对象
    klass = estimator.__class__
    
    # 获取 estimator 的参数，不进行深度复制
    new_object_params = estimator.get_params(deep=False)
    
    # 对新对象的参数进行深度复制
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    
    # 使用新对象的参数创建一个新的对象
    new_object = klass(**new_object_params)
    
    # 尝试复制 estimator 的 `_metadata_request` 属性
    try:
        new_object._metadata_request = copy.deepcopy(estimator._metadata_request)
    except AttributeError:
        pass
    
    # 获取新对象的参数，不进行深度复制
    params_set = new_object.get_params(deep=False)
    
    # 对新旧参数进行快速检查，确保它们相同
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError(
                "Cannot clone object %s, as the constructor "
                "either does not set or modifies parameter %s" % (estimator, name)
            )
    
    # 如果 estimator 具有 `_sklearn_output_config` 属性，则复制到新对象中
    if hasattr(estimator, "_sklearn_output_config"):
        new_object._sklearn_output_config = copy.deepcopy(
            estimator._sklearn_output_config
        )
    
    # 返回克隆后的新对象
    return new_object
    @classmethod
    def _get_param_names(cls):
        """获取估算器的参数名称"""
        # 获取构造函数或原始的构造函数（如果有过时包装）
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # 没有显式的构造函数可以内省
            return []

        # 内省构造函数参数，以查找要表示的模型参数
        init_signature = inspect.signature(init)
        # 考虑构造函数参数，不包括 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn 估算器应始终在其 __init__ 的签名中指定参数（不接受可变参数）。"
                    " %s 的构造函数 %s 不遵循此约定。" % (cls, init_signature)
                )
        # 提取和排序参数名称，不包括 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        获取此估算器的参数。

        Parameters
        ----------
        deep : bool, 默认=True
            如果为 True，将返回此估算器及其作为估算器的子对象的参数。

        Returns
        -------
        params : dict
            参数名称映射到它们的值。
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out
    # 定义一个方法，用于设置估算器的参数
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        # 如果没有传入参数，直接返回当前对象，进行简单的优化以提高速度（检查操作较慢）
        if not params:
            return self
        # 获取当前估算器的有效参数
        valid_params = self.get_params(deep=True)

        # 使用 defaultdict 创建一个嵌套参数字典，按前缀分组
        nested_params = defaultdict(dict)  # grouped by prefix
        # 遍历传入的参数字典
        for key, value in params.items():
            # 使用分隔符 '__' 对参数键进行分割，以支持对嵌套对象各组件的更新
            key, delim, sub_key = key.partition("__")
            # 如果键不在有效参数中，抛出 ValueError 异常
            if key not in valid_params:
                # 获取本地有效参数列表
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            # 如果存在分隔符，则将参数添加到嵌套参数字典中对应的子键下
            if delim:
                nested_params[key][sub_key] = value
            else:
                # 否则直接设置当前对象的属性值为传入值，并更新有效参数列表
                setattr(self, key, value)
                valid_params[key] = value

        # 遍历嵌套参数字典，递归设置每个嵌套组件的参数
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        # 返回设置后的估算器实例
        return self

    # 定义一个方法，用于克隆带参数的对象
    def __sklearn_clone__(self):
        # 调用 _clone_parametrized 函数来执行实际的克隆操作
        return _clone_parametrized(self)
    # 返回对象的字符串表示形式，限制最大非空字符数为 N_CHAR_MAX
    def __repr__(self, N_CHAR_MAX=700):
        # N_CHAR_MAX 是最大非空字符数的近似值，作为可选参数传递以便于测试

        # 导入 _EstimatorPrettyPrinter 类来美化打印输出
        from .utils._pprint import _EstimatorPrettyPrinter

        # 定义在序列中要显示的最大元素数量
        N_MAX_ELEMENTS_TO_SHOW = 30

        # 使用省略号来显示具有大量元素的序列
        pp = _EstimatorPrettyPrinter(
            compact=True,
            indent=1,
            indent_at_name=True,
            n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW,
        )

        # 生成对象的漂亮打印形式的字符串表示
        repr_ = pp.pformat(self)

        # 当非空字符数超过 N_CHAR_MAX 时使用强制省略号
        n_nonblank = len("".join(repr_.split()))
        if n_nonblank > N_CHAR_MAX:
            lim = N_CHAR_MAX // 2  # 分别保留两端的大约字符数
            regex = r"^(\s*\S){%d}" % lim
            # 正则表达式 '^(\s*\S){%d}' % n
            # 从字符串开头匹配到第 n 个非空字符：
            # - ^ 匹配字符串开头
            # - (pattern){n} 匹配模式的 n 个重复
            # - \s*\S 匹配零个或多个空格后的非空字符
            left_lim = re.match(regex, repr_).end()
            right_lim = re.match(regex, repr_[::-1]).end()

            if "\n" in repr_[left_lim:-right_lim]:
                # 左右两侧不在同一行时，为避免奇怪的切割，例如：
                # categoric...ore',
                # 我们需要适当的换行字符开始右侧，以便正确显示为：
                # categoric...
                # handle_unknown='ignore',
                # 因此我们添加 [^\n]*\n，匹配直到下一个 \n
                regex += r"[^\n]*\n"
                right_lim = re.match(regex, repr_[::-1]).end()

            ellipsis = "..."
            if left_lim + len(ellipsis) < len(repr_) - right_lim:
                # 仅在省略号可以使 repr 更短时添加省略号
                repr_ = repr_[:left_lim] + "..." + repr_[-right_lim:]

        # 返回处理后的字符串表示形式
        return repr_

    # 返回对象的状态字典
    def __getstate__(self):
        if getattr(self, "__slots__", None):
            # 如果对象使用了 __slots__，则抛出类型错误
            raise TypeError(
                "You cannot use `__slots__` in objects inheriting from "
                "`sklearn.base.BaseEstimator`."
            )

        try:
            # 尝试调用父类的 __getstate__ 方法获取状态字典
            state = super().__getstate__()
            if state is None:
                # 对于 Python 3.11+，空实例（没有 `__slots__` 和 `__dict__`）会返回状态等于 `None`
                state = self.__dict__.copy()
        except AttributeError:
            # Python < 3.11 的处理方式
            state = self.__dict__.copy()

        # 如果对象属于 sklearn 模块，将 sklearn 版本信息添加到状态字典中返回
        if type(self).__module__.startswith("sklearn."):
            return dict(state.items(), _sklearn_version=__version__)
        else:
            # 否则，直接返回状态字典
            return state
    # 如果对象所属的模块是以 "sklearn." 开头的，执行以下操作
    def __setstate__(self, state):
        # 弹出对象状态中的 "_sklearn_version" 键值，默认为 "pre-0.18"
        pickle_version = state.pop("_sklearn_version", "pre-0.18")
        # 如果 "_sklearn_version" 不等于当前的 sklearn 版本
        if pickle_version != __version__:
            # 发出警告，指示不一致的 sklearn 版本
            warnings.warn(
                InconsistentVersionWarning(
                    estimator_name=self.__class__.__name__,
                    current_sklearn_version=__version__,
                    original_sklearn_version=pickle_version,
                ),
            )
        
        # 尝试调用父类的 __setstate__ 方法
        try:
            super().__setstate__(state)
        # 处理 AttributeError 异常
        except AttributeError:
            # 更新对象的 __dict__ 属性，将状态中的内容添加到对象中
            self.__dict__.update(state)

    # 返回默认的标签 _DEFAULT_TAGS
    def _more_tags(self):
        return _DEFAULT_TAGS

    # 获取对象的标签
    def _get_tags(self):
        # 初始化一个空字典用于收集标签
        collected_tags = {}
        # 反向遍历对象类的方法解析顺序（Method Resolution Order, MRO）
        for base_class in reversed(inspect.getmro(self.__class__)):
            # 如果基类具有 _more_tags 方法
            if hasattr(base_class, "_more_tags"):
                # 调用基类的 _more_tags 方法，并将返回的标签更新到 collected_tags 中
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        # 返回收集到的所有标签
        return collected_tags
    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        try:
            # 获取输入数据 X 的特征数
            n_features = _num_features(X)
        except TypeError as e:
            if not reset and hasattr(self, "n_features_in_"):
                # 如果不是重置模式且已经存在 n_features_in_ 属性，则验证特征数是否匹配
                raise ValueError(
                    "X does not contain any features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features"
                ) from e
            # 如果特征数未定义且 reset=True，则跳过此检查
            return

        if reset:
            # 如果是重置模式，则将 n_features_in_ 属性设置为输入数据的特征数
            self.n_features_in_ = n_features
            return

        if not hasattr(self, "n_features_in_"):
            # 如果 n_features_in_ 属性不存在，则跳过检查
            return

        if n_features != self.n_features_in_:
            # 如果输入数据的特征数与 n_features_in_ 属性不匹配，则引发异常
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )

    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        cast_to_ndarray=True,
        **check_params,
    ):
        """Validate the input data and its parameters."""
        # 省略_validate_data方法的具体实现，其余参数在该方法中进行处理

    def _validate_params(self):
        """Validate types and values of constructor parameters

        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        # 调用validate_parameter_constraints函数，验证构造函数参数的类型和值
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )

    @property
    def _repr_html_(self):
        """
        HTML representation of estimator.

        This is redundant with the logic of `_repr_mimebundle_`. The latter
        should be favored in the long term, `_repr_html_` is only
        implemented for consumers who do not interpret `_repr_mimebundle_`.
        """
        # 检查配置中的"display"选项是否设置为"diagram"
        if get_config()["display"] != "diagram":
            # 如果"display"配置选项不是"diagram"，则抛出属性错误异常
            raise AttributeError(
                "_repr_html_ is only defined when the "
                "'display' configuration option is set to "
                "'diagram'"
            )
        # 返回内部 HTML 表示
        return self._repr_html_inner

    def _repr_html_inner(self):
        """
        This function is returned by the @property `_repr_html_` to make
        `hasattr(estimator, "_repr_html_") return `True` or `False` depending
        on `get_config()["display"]`.
        """
        # 返回由`estimator_html_repr`函数生成的 HTML 表示
        return estimator_html_repr(self)

    def _repr_mimebundle_(self, **kwargs):
        """
        Mime bundle used by jupyter kernels to display estimator
        """
        # 初始化输出为包含"text/plain"键的字典，值为对象的字符串表示
        output = {"text/plain": repr(self)}
        # 检查配置中的"display"选项是否设置为"diagram"
        if get_config()["display"] == "diagram":
            # 如果"display"配置选项是"diagram"，则将"text/html"键添加到输出字典中，
            # 值为对象的 HTML 表示，由`estimator_html_repr`函数生成
            output["text/html"] = estimator_html_repr(self)
        # 返回完整的输出字典
        return output
class ClassifierMixin:
    """Mixin class for all classifiers in scikit-learn.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"classifier"`;
    - `score` method that default to :func:`~sklearn.metrics.accuracy_score`.
    - enforce that `fit` requires `y` to be passed through the `requires_y` tag.

    Read more in the :ref:`User Guide <rolling_your_own_estimator>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, ClassifierMixin
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(ClassifierMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=1)
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([1, 1, 1])
    >>> estimator.score(X, y)
    0.66...
    """

    _estimator_type = "classifier"  # 设定分类器的类型标识为 "classifier"

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        from .metrics import accuracy_score  # 导入 accuracy_score 函数

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)  # 返回预测准确率

    def _more_tags(self):
        return {"requires_y": True}  # 返回一个字典，指示 `fit` 方法需要传递 `y` 参数


class RegressorMixin:
    """Mixin class for all regression estimators in scikit-learn.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"regressor"`;
    - `score` method that default to :func:`~sklearn.metrics.r2_score`.
    - enforce that `fit` requires `y` to be passed through the `requires_y` tag.

    Read more in the :ref:`User Guide <rolling_your_own_estimator>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, RegressorMixin
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(RegressorMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...
    """
    # 定义一个类变量 `_estimator_type`，表明这个类是一个回归器
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` w.r.t. `y`.

        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """

        # 导入 r2_score 函数
        from .metrics import r2_score

        # 使用当前模型预测 X 的结果
        y_pred = self.predict(X)
        # 返回预测结果与真实值 y 之间的 R^2 分数
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        # 返回一个字典，指示这个估算器需要真实值 y
        return {"requires_y": True}
class ClusterMixin:
    """Mixin class for all cluster estimators in scikit-learn.

    - `_estimator_type` class attribute defaulting to `"clusterer"`;
    - `fit_predict` method returning the cluster labels associated to each sample.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, ClusterMixin
    >>> class MyClusterer(ClusterMixin, BaseEstimator):
    ...     def fit(self, X, y=None):
    ...         self.labels_ = np.ones(shape=(len(X),), dtype=np.int64)
    ...         return self
    >>> X = [[1, 2], [2, 3], [3, 4]]
    >>> MyClusterer().fit_predict(X)
    array([1, 1, 1])
    """

    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None, **kwargs):
        """
        Perform clustering on `X` and returns cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by convention.

        **kwargs : dict
            Arguments to be passed to ``fit``.

            .. versionadded:: 1.4

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=np.int64
            Cluster labels.
        """
        # 非优化的默认实现；针对给定的聚类算法可能有更好的方法时，应该进行重写
        self.fit(X, **kwargs)  # 调用 fit 方法进行聚类
        return self.labels_  # 返回聚类结果的标签数组

    def _more_tags(self):
        return {"preserves_dtype": []}


class BiclusterMixin:
    """Mixin class for all bicluster estimators in scikit-learn.

    This mixin defines the following functionality:

    - `biclusters_` property that returns the row and column indicators;
    - `get_indices` method that returns the row and column indices of a bicluster;
    - `get_shape` method that returns the shape of a bicluster;
    - `get_submatrix` method that returns the submatrix corresponding to a bicluster.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, BiclusterMixin
    >>> class DummyBiClustering(BiclusterMixin, BaseEstimator):
    ...     def fit(self, X, y=None):
    ...         self.rows_ = np.ones(shape=(1, X.shape[0]), dtype=bool)
    ...         self.columns_ = np.ones(shape=(1, X.shape[1]), dtype=bool)
    ...         return self
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> bicluster = DummyBiClustering().fit(X)
    >>> hasattr(bicluster, "biclusters_")
    True
    >>> bicluster.get_indices(0)
    (array([0, 1, 2, 3, 4, 5]), array([0, 1]))
    """

    @property
    def biclusters_(self):
        """Convenient way to get row and column indicators together.

        Returns the ``rows_`` and ``columns_`` members.
        """
        return self.rows_, self.columns_
    def get_indices(self, i):
        """获取第 i 个双聚类的行和列索引。

        只有在存在 ``rows_`` 和 ``columns_`` 属性时有效。

        Parameters
        ----------
        i : int
            聚类的索引。

        Returns
        -------
        row_ind : ndarray, dtype=np.intp
            属于双聚类的数据集中行的索引。
        col_ind : ndarray, dtype=np.intp
            属于双聚类的数据集中列的索引。
        """
        rows = self.rows_[i]
        columns = self.columns_[i]
        return np.nonzero(rows)[0], np.nonzero(columns)[0]

    def get_shape(self, i):
        """获取第 i 个双聚类的形状信息。

        Parameters
        ----------
        i : int
            聚类的索引。

        Returns
        -------
        n_rows : int
            双聚类中的行数。
        n_cols : int
            双聚类中的列数。
        """
        indices = self.get_indices(i)
        return tuple(len(i) for i in indices)

    def get_submatrix(self, i, data):
        """返回与第 i 个双聚类对应的子矩阵。

        Parameters
        ----------
        i : int
            聚类的索引。
        data : array-like of shape (n_samples, n_features)
            数据集。

        Returns
        -------
        submatrix : ndarray of shape (n_rows, n_cols)
            与第 i 个双聚类对应的子矩阵。

        Notes
        -----
        可以处理稀疏矩阵。只有在存在 ``rows_`` 和 ``columns_`` 属性时有效。
        """
        from .utils.validation import check_array

        data = check_array(data, accept_sparse="csr")
        row_ind, col_ind = self.get_indices(i)
        return data[row_ind[:, np.newaxis], col_ind]
# 定义一个 TransformerMixin 类，它是 scikit-learn 中所有变换器的混合类

class TransformerMixin(_SetOutputMixin):
    """Mixin class for all transformers in scikit-learn.
    
    This mixin defines the following functionality:
    
    - a `fit_transform` method that delegates to `fit` and `transform`;
    - a `set_output` method to output `X` as a specific container type.
    
    If :term:`get_feature_names_out` is defined, then :class:`BaseEstimator` will
    automatically wrap `transform` and `fit_transform` to follow the `set_output`
    API. See the :ref:`developer_api_set_output` for details.
    
    :class:`OneToOneFeatureMixin` and
    :class:`ClassNamePrefixFeaturesOutMixin` are helpful mixins for
    defining :term:`get_feature_names_out`.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, TransformerMixin
    >>> class MyTransformer(TransformerMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         return self
    ...     def transform(self, X):
    ...         return np.full(shape=len(X), fill_value=self.param)
    >>> transformer = MyTransformer()
    >>> X = [[1, 2], [2, 3], [3, 4]]
    >>> transformer.fit_transform(X)
    array([1, 1, 1])
    """
    # 对数据进行拟合和转换的方法，继承自TransformerMixin类，用于机器学习中的转换器
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # 默认非优化的实现方式；当给定聚类算法时，可以覆盖此方法以提供更好的实现
        #
        # 这里不进行参数路由，因为消费者不进行路由。但是，由于`transform`方法有可能
        # 也会消费元数据，我们检查是否存在这种情况，并提出警告，告知用户应该实现自定义的
        # `fit_transform` 方法以将元数据转发到`transform`方法。
        #
        # 为此，我们计算路由并检查如果我们要路由它们到`transform`方法，是否会有任何内容被路由。
        if _routing_enabled():
            transform_params = self.get_metadata_routing().consumes(
                method="transform", params=fit_params.keys()
            )
            if transform_params:
                warnings.warn(
                    (
                        f"This object ({self.__class__.__name__}) has a `transform`"
                        " method which consumes metadata, but `fit_transform` does not"
                        " forward metadata to `transform`. Please implement a custom"
                        " `fit_transform` method to forward metadata to `transform` as"
                        " well. Alternatively, you can explicitly do"
                        " `set_transform_request`and set all values to `False` to"
                        " disable metadata routed to `transform`, if that's an option."
                    ),
                    UserWarning,
                )

        if y is None:
            # 如果y为None，则表明是单参数的拟合方法（无监督转换）
            return self.fit(X, **fit_params).transform(X)
        else:
            # 如果y不为None，则表明是双参数的拟合方法（有监督转换）
            return self.fit(X, y, **fit_params).transform(X)
class OneToOneFeatureMixin:
    """Provides `get_feature_names_out` for simple transformers.

    This mixin assumes there's a 1-to-1 correspondence between input features
    and output features, such as :class:`~sklearn.preprocessing.StandardScaler`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import OneToOneFeatureMixin
    >>> class MyEstimator(OneToOneFeatureMixin):
    ...     def fit(self, X, y=None):
    ...         self.n_features_in_ = X.shape[1]
    ...         return self
    >>> X = np.array([[1, 2], [3, 4]])
    >>> MyEstimator().fit(X).get_feature_names_out()
    array(['x0', 'x1'], dtype=object)
    """

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Same as input features.
        """
        # 检查是否已经拟合过，确保 n_features_in_ 属性存在
        check_is_fitted(self, "n_features_in_")
        # 调用内部函数 _check_feature_names_in 处理输入特征，返回输出特征名
        return _check_feature_names_in(self, input_features)


class ClassNamePrefixFeaturesOutMixin:
    """Mixin class for transformers that generate their own names by prefixing.

    This mixin is useful when the transformer needs to generate its own feature
    names out, such as :class:`~sklearn.decomposition.PCA`. For example, if
    :class:`~sklearn.decomposition.PCA` outputs 3 features, then the generated feature
    names out are: `["pca0", "pca1", "pca2"]`.

    This mixin assumes that a `_n_features_out` attribute is defined when the
    transformer is fitted. `_n_features_out` is the number of output features
    that the transformer will return in `transform` of `fit_transform`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import ClassNamePrefixFeaturesOutMixin
    >>> class MyEstimator(ClassNamePrefixFeaturesOutMixin):
    ...     def fit(self, X, y=None):
    ...         self._n_features_out = X.shape[1]
    ...         return self
    >>> X = np.array([[1, 2], [3, 4]])
    >>> MyEstimator().fit(X).get_feature_names_out()
    array(['myestimator0', 'myestimator1'], dtype=object)
    """
    # 获取输出转换后的特征名称列表

    """Get output feature names for transformation.
    
    The feature names out will prefixed by the lowercased class name. For
    example, if the transformer outputs 3 features, then the feature names
    out are: `["class_name0", "class_name1", "class_name2"]`.
    """
    
    # 检查模型是否已经拟合，即是否存在 _n_features_out 属性
    check_is_fitted(self, "_n_features_out")
    
    # 调用辅助函数 _generate_get_feature_names_out 生成输出特征名称列表
    return _generate_get_feature_names_out(
        self, self._n_features_out, input_features=input_features
    )
# 密度估计器的 Mixin 类，用于所有 scikit-learn 中的密度估计器。

# 定义了以下功能：
# - `_estimator_type` 类属性，默认为 `"DensityEstimator"`
# - `score` 方法，默认不执行任何操作。

class DensityMixin:
    _estimator_type = "DensityEstimator"

    def score(self, X, y=None):
        """返回模型在数据 `X` 上的得分。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            测试样本。

        y : 忽略
            不使用，仅为 API 一致性而存在。

        Returns
        -------
        score : float
        """
        pass


# 异常检测估计器的 Mixin 类，用于所有 scikit-learn 中的异常检测估计器。

# 定义了以下功能：
# - `_estimator_type` 类属性，默认为 `"outlier_detector"`
# - `fit_predict` 方法，默认为 `fit` 和 `predict`。

class OutlierMixin:
    _estimator_type = "outlier_detector"

    def fit_predict(self, X, y=None):
        """用于拟合数据并预测异常值。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入样本。

        y : 忽略
            不使用，仅为 API 一致性而存在。

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            每个样本的异常值预测结果。
        """
        pass
    def fit_predict(self, X, y=None, **kwargs):
        """Perform fit on X and returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        **kwargs : dict
            Arguments to be passed to ``fit``.

            .. versionadded:: 1.4

        Returns
        -------
        y : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        
        # 检查是否启用了路由功能
        if _routing_enabled():
            # 获取元数据路由信息，检查是否有参数会被路由到 `predict` 方法
            transform_params = self.get_metadata_routing().consumes(
                method="predict", params=kwargs.keys()
            )
            # 如果有参数会被路由到 `predict` 方法，则发出警告提示用户实现自定义的 `fit_predict` 方法
            if transform_params:
                warnings.warn(
                    (
                        f"This object ({self.__class__.__name__}) has a `predict` "
                        "method which consumes metadata, but `fit_predict` does not "
                        "forward metadata to `predict`. Please implement a custom "
                        "`fit_predict` method to forward metadata to `predict` as well."
                        "Alternatively, you can explicitly do `set_predict_request`"
                        "and set all values to `False` to disable metadata routed to "
                        "`predict`, if that's an option."
                    ),
                    UserWarning,
                )

        # 对于类似 LocalOutlierFactor 的转导异常检测器进行重写
        # 调用 self.fit(X, **kwargs) 进行模型拟合，然后使用拟合后的模型进行预测，并返回结果
        return self.fit(X, **kwargs).predict(X)
class MetaEstimatorMixin:
    """Mixin class for all meta estimators in scikit-learn.

    This mixin defines the following functionality:

    - define `_required_parameters` that specify the mandatory `estimator` parameter.
    """

    _required_parameters = ["estimator"]


class MultiOutputMixin:
    """Mixin to mark estimators that support multioutput."""

    def _more_tags(self):
        """Returns additional tags for the estimator, indicating multioutput support."""
        return {"multioutput": True}


class _UnstableArchMixin:
    """Mark estimators that are non-determinstic on 32bit or PowerPC"""

    def _more_tags(self):
        """Returns additional tags for the estimator, indicating non-determinism based on architecture."""
        return {
            "non_deterministic": _IS_32BIT
            or platform.machine().startswith(("ppc", "powerpc"))
        }


def is_classifier(estimator):
    """Return True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"


def is_regressor(estimator):
    """Return True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "regressor"


def is_clusterer(estimator):
    """Return True if the given estimator is (probably) a clusterer.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a clusterer and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "clusterer"
    """
    .. versionadded:: 1.6
    标记：在版本1.6中添加

    Parameters
    ----------
    estimator : object
        Estimator object to test.
    参数：estimator：对象
        要测试的评估器对象。

    Returns
    -------
    out : bool
        True if estimator is a clusterer and False otherwise.
    返回：
        如果评估器是一个聚类器则返回True，否则返回False。

    Examples
    --------
    >>> from sklearn.base import is_clusterer
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.svm import SVC, SVR
    >>> classifier = SVC()
    >>> regressor = SVR()
    >>> kmeans = KMeans()
    >>> is_clusterer(classifier)
    False
    >>> is_clusterer(regressor)
    False
    >>> is_clusterer(kmeans)
    True
    """
    return getattr(estimator, "_estimator_type", None) == "clusterer"
    # 检查评估器的属性"_estimator_type"是否为"clusterer"，并返回比较结果
# 检测给定的估算器是否是异常检测器
def is_outlier_detector(estimator):
    """Return True if the given estimator is (probably) an outlier detector.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an outlier detector and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "outlier_detector"


# 创建一个装饰器，用于在上下文管理器中运行估算器的fit方法
def _fit_context(*, prefer_skip_nested_validation):
    """Decorator to run the fit methods of estimators within context managers.

    Parameters
    ----------
    prefer_skip_nested_validation : bool
        If True, the validation of parameters of inner estimators or functions
        called during fit will be skipped.

        This is useful to avoid validating many times the parameters passed by the
        user from the public facing API. It's also useful to avoid validating
        parameters that we pass internally to inner functions that are guaranteed to
        be valid by the test suite.

        It should be set to True for most estimators, except for those that receive
        non-validated objects as parameters, such as meta-estimators that are given
        estimator objects.

    Returns
    -------
    decorated_fit : method
        The decorated fit method.
    """

    def decorator(fit_method):
        @functools.wraps(fit_method)
        def wrapper(estimator, *args, **kwargs):
            global_skip_validation = get_config()["skip_parameter_validation"]

            # 不希望对每次调用partial_fit重新验证
            partial_fit_and_fitted = (
                fit_method.__name__ == "partial_fit" and _is_fitted(estimator)
            )

            # 如果全局设置不跳过参数验证，并且不是partial_fit已经拟合过的情况下
            if not global_skip_validation and not partial_fit_and_fitted:
                estimator._validate_params()

            # 在配置上下文中运行fit方法
            with config_context(
                skip_parameter_validation=(
                    prefer_skip_nested_validation or global_skip_validation
                )
            ):
                return fit_method(estimator, *args, **kwargs)

        return wrapper

    return decorator
```