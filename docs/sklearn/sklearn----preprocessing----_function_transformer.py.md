# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\_function_transformer.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库

from ..base import BaseEstimator, TransformerMixin, _fit_context  # 导入自定义模块
from ..utils._param_validation import StrOptions  # 导入参数验证工具
from ..utils._set_output import (  # 导入设置输出工具中的特定函数
    _get_adapter_from_container,
    _get_output_config,
)
from ..utils.metaestimators import available_if  # 导入元估计器中的特定函数
from ..utils.validation import (  # 导入验证工具中的特定函数
    _allclose_dense_sparse,
    _check_feature_names_in,
    _get_feature_names,
    _is_pandas_df,
    _is_polars_df,
    check_array,
)


def _identity(X):
    """The identity function."""
    return X  # 返回输入的参数本身


class FunctionTransformer(TransformerMixin, BaseEstimator):
    """Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <function_transformer>`.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.

    inverse_func : callable, default=None
        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If inverse_func is None, then inverse_func
        will be the identity function.

    validate : bool, default=False
        Indicate that the input X array should be checked before calling
        ``func``. The possibilities are:

        - If False, there is no input validation.
        - If True, then X will be converted to a 2-dimensional NumPy array or
          sparse matrix. If the conversion is not possible an exception is
          raised.

        .. versionchanged:: 0.22
           The default of ``validate`` changed from True to False.

    accept_sparse : bool, default=False
        Indicate that func accepts a sparse matrix as input. If validate is
        False, this has no effect. Otherwise, if accept_sparse is false,
        sparse matrix inputs will cause an exception to be raised.

    check_inverse : bool, default=True
       Whether to check that or ``func`` followed by ``inverse_func`` leads to
       the original inputs. It can be used for a sanity check, raising a
       warning when the condition is not fulfilled.

       .. versionadded:: 0.20
    """
    pass  # FunctionTransformer 类的定义，暂时没有实现额外的功能
    """
    feature_names_out : callable, 'one-to-one' or None, default=None
        确定将由 `get_feature_names_out` 方法返回的特征名列表。如果是 'one-to-one'，则输出的特征名与输入的特征名相同。
        如果是一个可调用对象，则必须接受两个位置参数：此 `FunctionTransformer` (`self`) 和一个类似数组的输入特征名 (`input_features`)。
        它必须返回一个类似数组的输出特征名。只有在 `feature_names_out` 不为 None 时，才定义 `get_feature_names_out` 方法。
    
        查看 ``get_feature_names_out`` 获取更多详情。
    
        .. versionadded:: 1.1
    
    kw_args : dict, default=None
        传递给 func 的额外关键字参数的字典。
    
        .. versionadded:: 0.18
    
    inv_kw_args : dict, default=None
        传递给 inverse_func 的额外关键字参数的字典。
    
        .. versionadded:: 0.18
    
    Attributes
    ----------
    n_features_in_ : int
        在 :term:`fit` 过程中观察到的特征数量。
    
        .. versionadded:: 0.24
    
    feature_names_in_ : 形状为 (`n_features_in_`,) 的 ndarray
        在 :term:`fit` 过程中观察到的特征名称。仅当 `X` 的特征名称都是字符串时才定义。
    
        .. versionadded:: 1.0
    
    See Also
    --------
    MaxAbsScaler : 将每个特征按其最大绝对值进行缩放。
    StandardScaler : 通过去除均值并缩放到单位方差来标准化特征。
    LabelBinarizer : 以一对所有方式对标签进行二进制化。
    MultiLabelBinarizer : 在可迭代的可迭代对象和多标签格式之间进行转换。
    
    Notes
    -----
    如果 `func` 返回的输出具有 `columns` 属性，则强制该列与 `get_feature_names_out` 的输出保持一致。
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> transformer = FunctionTransformer(np.log1p)
    >>> X = np.array([[0, 1], [2, 3]])
    >>> transformer.transform(X)
    array([[0.       , 0.6931...],
           [1.0986..., 1.3862...]])
    """
    
    _parameter_constraints: dict = {
        "func": [callable, None],
        "inverse_func": [callable, None],
        "validate": ["boolean"],
        "accept_sparse": ["boolean"],
        "check_inverse": ["boolean"],
        "feature_names_out": [callable, StrOptions({"one-to-one"}), None],
        "kw_args": [dict, None],
        "inv_kw_args": [dict, None],
    }
    
    def __init__(
        self,
        func=None,
        inverse_func=None,
        *,
        validate=False,
        accept_sparse=False,
        check_inverse=True,
        feature_names_out=None,
        kw_args=None,
        inv_kw_args=None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args


        # 初始化函数变量和参数
        self.func = func
        # 初始化反函数变量和参数
        self.inverse_func = inverse_func
        # 是否进行数据验证
        self.validate = validate
        # 是否接受稀疏矩阵作为输入
        self.accept_sparse = accept_sparse
        # 是否检查逆转换函数
        self.check_inverse = check_inverse
        # 输出的特征名称
        self.feature_names_out = feature_names_out
        # 传递给函数的关键字参数
        self.kw_args = kw_args
        # 传递给反函数的关键字参数
        self.inv_kw_args = inv_kw_args


    def _check_input(self, X, *, reset):
        if self.validate:
            return self._validate_data(X, accept_sparse=self.accept_sparse, reset=reset)
        elif reset:
            # 设置特征名称和特征数量，即使 validate=False 也要运行此代码以存储属性但不验证它们，因为 validate=False
            self._check_n_features(X, reset=reset)
            self._check_feature_names(X, reset=reset)
        return X


    def _check_inverse_transform(self, X):
        """Check that func and inverse_func are the inverse."""
        # 选择要检查的索引范围，确保是 X.shape[0] 的百分之一
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        # 对 X 的部分数据进行逆转换
        X_round_trip = self.inverse_transform(self.transform(X[idx_selected]))

        if hasattr(X, "dtype"):
            dtypes = [X.dtype]
        elif hasattr(X, "dtypes"):
            # 数据框可能有多个数据类型
            dtypes = X.dtypes

        # 检查 X 中的所有元素是否都是数值类型
        if not all(np.issubdtype(d, np.number) for d in dtypes):
            raise ValueError(
                "'check_inverse' is only supported when all the elements in `X` is"
                " numerical."
            )

        # 如果逆转换的结果与原始数据不接近，发出警告
        if not _allclose_dense_sparse(X[idx_selected], X_round_trip):
            warnings.warn(
                (
                    "The provided functions are not strictly"
                    " inverse of each other. If you are sure you"
                    " want to proceed regardless, set"
                    " 'check_inverse=False'."
                ),
                UserWarning,
            )


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit transformer by checking X.

        If ``validate`` is ``True``, ``X`` will be checked.

        Parameters
        ----------
        X : {array-like, sparse-matrix} of shape (n_samples, n_features) \
                if `validate=True` else any object that `func` can handle
            Input array.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FunctionTransformer class instance.
        """
        # 检查输入数据 X，并进行必要的重置操作
        X = self._check_input(X, reset=True)
        # 如果需要检查逆转换，并且 func 和 inverse_func 都已定义，则执行逆转换的检查
        if self.check_inverse and not (self.func is None or self.inverse_func is None):
            self._check_inverse_transform(X)
        return self
    def inverse_transform(self, X):
        """Transform X using the inverse function.

        Parameters
        ----------
        X : {array-like, sparse-matrix} of shape (n_samples, n_features) \
                if `validate=True` else any object that `inverse_func` can handle
            Input array.

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        # 如果需要验证，则调用check_array函数验证输入的X
        if self.validate:
            X = check_array(X, accept_sparse=self.accept_sparse)
        # 调用_transform方法，使用inverse_func进行逆变换
        return self._transform(X, func=self.inverse_func, kw_args=self.inv_kw_args)

    @available_if(lambda self: self.feature_names_out is not None)
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        This method is only defined if `feature_names_out` is not None.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names.

            - If `input_features` is None, then `feature_names_in_` is
              used as the input feature names. If `feature_names_in_` is not
              defined, then names are generated:
              `[x0, x1, ..., x(n_features_in_ - 1)]`.
            - If `input_features` is array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.

            - If `feature_names_out` is 'one-to-one', the input feature names
              are returned (see `input_features` above). This requires
              `feature_names_in_` and/or `n_features_in_` to be defined, which
              is done automatically if `validate=True`. Alternatively, you can
              set them in `func`.
            - If `feature_names_out` is a callable, then it is called with two
              arguments, `self` and `input_features`, and its return value is
              returned by this method.
        """
        # 如果self有'n_features_in_'属性或者input_features不为空，则调用_check_feature_names_in函数验证input_features
        if hasattr(self, "n_features_in_") or input_features is not None:
            input_features = _check_feature_names_in(self, input_features)
        # 根据self.feature_names_out的不同取值进行不同的处理
        if self.feature_names_out == "one-to-one":
            names_out = input_features
        elif callable(self.feature_names_out):
            # 如果feature_names_out是callable，则调用它处理self和input_features，得到结果
            names_out = self.feature_names_out(self, input_features)
        else:
            # 如果feature_names_out既不是'one-to-one'也不是callable，则抛出异常
            raise ValueError(
                f"feature_names_out={self.feature_names_out!r} is invalid. "
                'It must either be "one-to-one" or a callable with two '
                "arguments: the function transformer and an array-like of "
                "input feature names. The callable must return an array-like "
                "of output feature names."
            )
        # 将处理后的结果转换为ndarray并返回
        return np.asarray(names_out, dtype=object)
    def _transform(self, X, func=None, kw_args=None):
        # 如果 func 参数为 None，则默认使用 _identity 函数
        if func is None:
            func = _identity

        # 调用指定的 func 函数对输入 X 进行转换，并传入可选的关键字参数
        return func(X, **(kw_args if kw_args else {}))

    def __sklearn_is_fitted__(self):
        """Return True since FunctionTransfomer is stateless."""
        # 返回 True，因为 FunctionTransfomer 是无状态的
        return True

    def _more_tags(self):
        # 返回一个字典，包含两个标签信息：
        # - "no_validation": 根据 self.validate 的值判断是否进行验证
        # - "stateless": 始终为 True，表明该函数无状态
        return {"no_validation": not self.validate, "stateless": True}

    def set_output(self, *, transform=None):
        """Set output container.

        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        for an example on how to use the API.

        Parameters
        ----------
        transform : {"default", "pandas", "polars"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: DataFrame output
            - `"polars"`: Polars output
            - `None`: Transform configuration is unchanged

            .. versionadded:: 1.4
                `"polars"` option was added.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        # 如果对象没有 "_sklearn_output_config" 属性，则创建一个空字典
        if not hasattr(self, "_sklearn_output_config"):
            self._sklearn_output_config = {}

        # 将 transform 参数设置到 "_sklearn_output_config" 字典中
        self._sklearn_output_config["transform"] = transform
        # 返回当前对象的实例，用于链式调用
        return self
```