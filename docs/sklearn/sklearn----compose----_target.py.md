# `D:\src\scipysrc\scikit-learn\sklearn\compose\_target.py`

```
# 导入警告模块
import warnings

# 导入 NumPy 库并使用别名 np
import numpy as np

# 从 scikit-learn 中导入基类和混合类，以及相关函数和异常
from ..base import BaseEstimator, RegressorMixin, _fit_context, clone
from ..exceptions import NotFittedError

# 从线性模型中导入线性回归模型
from ..linear_model import LinearRegression

# 从预处理模块中导入函数转换器
from ..preprocessing import FunctionTransformer

# 从 scikit-learn 工具模块中导入 Bunch 类、安全索引检查函数和数组检查函数
from ..utils import Bunch, _safe_indexing, check_array

# 从元数据请求模块中导入元数据路由器、方法映射、路由是否启用和处理路由函数
from ..utils._metadata_requests import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    process_routing,
)

# 从参数验证模块中导入具有方法的验证类
from ..utils._param_validation import HasMethods

# 从标签模块中导入安全标签函数
from ..utils._tags import _safe_tags

# 从元数据路由模块中导入不支持路由混合类
from ..utils.metadata_routing import (
    _RoutingNotSupportedMixin,
)

# 从验证模块中导入检查是否拟合的函数
from ..utils.validation import check_is_fitted

# 定义该模块导出的公共接口
__all__ = ["TransformedTargetRegressor"]


# 定义转换目标回归器类，继承自不支持路由混合类、回归混合类和基类估计器
class TransformedTargetRegressor(
    _RoutingNotSupportedMixin, RegressorMixin, BaseEstimator
):
    """Meta-estimator to regress on a transformed target.

    Useful for applying a non-linear transformation to the target `y` in
    regression problems. This transformation can be given as a Transformer
    such as the :class:`~sklearn.preprocessing.QuantileTransformer` or as a
    function and its inverse such as `np.log` and `np.exp`.

    The computation during :meth:`fit` is::

        regressor.fit(X, func(y))

    or::

        regressor.fit(X, transformer.transform(y))

    The computation during :meth:`predict` is::

        inverse_func(regressor.predict(X))

    or::

        transformer.inverse_transform(regressor.predict(X))

    Read more in the :ref:`User Guide <transformed_target_regressor>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    regressor : object, default=None
        Regressor object such as derived from
        :class:`~sklearn.base.RegressorMixin`. This regressor will
        automatically be cloned each time prior to fitting. If `regressor is
        None`, :class:`~sklearn.linear_model.LinearRegression` is created and used.

    transformer : object, default=None
        Estimator object such as derived from
        :class:`~sklearn.base.TransformerMixin`. Cannot be set at the same time
        as `func` and `inverse_func`. If `transformer is None` as well as
        `func` and `inverse_func`, the transformer will be an identity
        transformer. Note that the transformer will be cloned during fitting.
        Also, the transformer is restricting `y` to be a numpy array.

    func : function, default=None
        Function to apply to `y` before passing to :meth:`fit`. Cannot be set
        at the same time as `transformer`. If `func is None`, the function used will be
        the identity function. If `func` is set, `inverse_func` also needs to be
        provided. The function needs to return a 2-dimensional array.

    """
    _parameter_constraints: dict = {
        "regressor": [HasMethods(["fit", "predict"]), None],
        # 定义参数约束字典，包括对应参数的要求和默认值
        "transformer": [HasMethods("transform"), None],
        # 定义 transformer 参数的要求为具有 transform 方法，无默认值
        "func": [callable, None],
        # 定义 func 参数的要求为可调用对象，无默认值
        "inverse_func": [callable, None],
        # 定义 inverse_func 参数的要求为可调用对象，无默认值
        "check_inverse": ["boolean"],
        # 定义 check_inverse 参数的要求为布尔值类型
    }

    def __init__(
        self,
        regressor=None,
        *,
        transformer=None,
        func=None,
        inverse_func=None,
        check_inverse=True,
    ):
        # 初始化函数，接受多个可选参数，并将其赋值给实例变量
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
    def _fit_transformer(self, y):
        """Check transformer and fit transformer.

        Create the default transformer, fit it and make additional inverse
        check on a subset (optional).

        """
        # 检查是否同时设置了'transformer'和'func'/'inverse_func'，如果是则抛出异常
        if self.transformer is not None and (
            self.func is not None or self.inverse_func is not None
        ):
            raise ValueError(
                "'transformer' and functions 'func'/'inverse_func' cannot both be set."
            )
        elif self.transformer is not None:
            # 如果只设置了'transformer'，则克隆该transformer对象
            self.transformer_ = clone(self.transformer)
        else:
            # 如果没有设置'transformer'，则检查是否设置了'func'和'inverse_func'其中之一
            if (self.func is not None and self.inverse_func is None) or (
                self.func is None and self.inverse_func is not None
            ):
                lacking_param, existing_param = (
                    ("func", "inverse_func")
                    if self.func is None
                    else ("inverse_func", "func")
                )
                # 如果设置了一个函数但没有设置另一个，则抛出异常
                raise ValueError(
                    f"When '{existing_param}' is provided, '{lacking_param}' must also"
                    f" be provided. If {lacking_param} is supposed to be the default,"
                    " you need to explicitly pass it the identity function."
                )
            # 根据提供的'func'和'inverse_func'创建FunctionTransformer对象
            self.transformer_ = FunctionTransformer(
                func=self.func,
                inverse_func=self.inverse_func,
                validate=True,
                check_inverse=self.check_inverse,
            )
        
        # XXX: sample_weight is not currently passed to the
        # transformer. However, if transformer starts using sample_weight, the
        # code should be modified accordingly. At the time to consider the
        # sample_prop feature, it is also a good use case to be considered.
        # 对数据'y'进行拟合操作
        self.transformer_.fit(y)
        
        # 如果设置了检查逆变换的标志，则进行逆变换的验证
        if self.check_inverse:
            # 选择数据'y'的子集进行验证逆变换
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = _safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)
            # 检查逆变换是否严格逆，如果不是则发出警告
            if not np.allclose(y_sel, self.transformer_.inverse_transform(y_sel_t)):
                warnings.warn(
                    (
                        "The provided functions or transformer are"
                        " not strictly inverse of each other. If"
                        " you are sure you want to proceed regardless"
                        ", set 'check_inverse=False'"
                    ),
                    UserWarning,
                )

    @_fit_context(
        # TransformedTargetRegressor.regressor/transformer are not validated yet.
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            - If `enable_metadata_routing=False` (default):

                Parameters directly passed to the `fit` method of the
                underlying regressor.

            - If `enable_metadata_routing=True`:

                Parameters safely routed to the `fit` method of the
                underlying regressor.

                .. versionchanged:: 1.6
                    See :ref:`Metadata Routing User Guide <metadata_routing>` for
                    more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        if y is None:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )

        # Validate and prepare the target variable y
        y = check_array(
            y,
            input_name="y",
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype="numeric",
            allow_nd=True,
        )

        # Store the number of dimensions of the target variable y for prediction
        self._training_dim = y.ndim

        # Adjust y to be 2D if it's 1D, as transformers expect 2D input
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y

        # Fit the transformer with the adjusted target y
        self._fit_transformer(y_2d)

        # Transform y using the fitted transformer and ensure it's 1D if necessary
        y_trans = self.transformer_.transform(y_2d)

        # Handle edge case where FunctionTransformer can return a 1D array even with validate=True
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        # Obtain the regressor instance to use for fitting
        self.regressor_ = self._get_regressor(get_clone=True)

        # Process fit parameters based on metadata routing status
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch(regressor=Bunch(fit=fit_params))

        # Fit the regressor using the provided X and transformed y
        self.regressor_.fit(X, y_trans, **routed_params.regressor.fit)

        # Store feature names if available in the regressor
        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_

        # Return the fitted estimator object
        return self
    def predict(self, X, **predict_params):
        """
        Predict using the base regressor, applying inverse.

        The regressor is used to predict and the `inverse_func` or
        `inverse_transform` is applied before returning the prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        **predict_params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters directly passed to the `predict` method of the
                underlying regressor.

            - If `enable_metadata_routing=True`:

                Parameters safely routed to the `predict` method of the
                underlying regressor.

                .. versionchanged:: 1.6
                    See :ref:`Metadata Routing User Guide <metadata_routing>`
                    for more details.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Ensure the estimator is fitted
        check_is_fitted(self)
        
        # Check if routing is enabled and process predict parameters accordingly
        if _routing_enabled():
            routed_params = process_routing(self, "predict", **predict_params)
        else:
            routed_params = Bunch(regressor=Bunch(predict=predict_params))

        # Predict using the regressor, applying routed parameters
        pred = self.regressor_.predict(X, **routed_params.regressor.predict)
        
        # Apply inverse transformation to predicted values
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        
        # Adjust predicted values if necessary
        if (
            self._training_dim == 1
            and pred_trans.ndim == 2
            and pred_trans.shape[1] == 1
        ):
            pred_trans = pred_trans.squeeze(axis=1)

        return pred_trans

    def _more_tags(self):
        """
        Provides additional tags for the estimator.

        Returns
        -------
        dict
            Additional tags including 'poor_score' and 'multioutput'.
        """
        # Retrieve the underlying regressor
        regressor = self._get_regressor()

        # Return additional tags
        return {
            "poor_score": True,
            "multioutput": _safe_tags(regressor, key="multioutput"),
        }

    @property
    def n_features_in_(self):
        """
        Number of features seen during fit.

        Raises
        ------
        AttributeError
            If the estimator is not fitted.

        Returns
        -------
        int
            Number of features.
        """
        # Check if the estimator is fitted; raise AttributeError if not
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        # Return the number of features from the regressor
        return self.regressor_.n_features_in_
    # 获取该对象的元数据路由信息
    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.6

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，使用当前类名作为所有者
        router = MetadataRouter(owner=self.__class__.__name__).add(
            # 设置回归器的元数据路由，包括 fit 和 predict 方法的映射
            regressor=self._get_regressor(),
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict"),
        )
        # 返回创建的 MetadataRouter 对象
        return router

    # 获取回归器对象，可选择是否返回克隆对象
    def _get_regressor(self, get_clone=False):
        # 如果回归器对象为空，返回一个新的线性回归器对象
        if self.regressor is None:
            return LinearRegression()

        # 如果需要返回克隆对象，则返回回归器对象的克隆
        return clone(self.regressor) if get_clone else self.regressor
```