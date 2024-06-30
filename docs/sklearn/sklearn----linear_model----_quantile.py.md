# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_quantile.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# 引入警告模块，用于处理警告信息
import warnings
# 引入 Real 类型，用于检查参数是否为实数
from numbers import Real

# 引入 numpy 库并使用 np 别名
import numpy as np
# 引入 scipy 库中的 sparse 模块
from scipy import sparse
# 从 scipy.optimize 模块中引入 linprog 函数
from scipy.optimize import linprog

# 从当前包中的 base 模块引入 BaseEstimator、RegressorMixin 和 _fit_context
from ..base import BaseEstimator, RegressorMixin, _fit_context
# 从 exceptions 模块中引入 ConvergenceWarning
from ..exceptions import ConvergenceWarning
# 从 utils 模块中引入 _safe_indexing 函数
from ..utils import _safe_indexing
# 从 utils._param_validation 模块引入 Interval 和 StrOptions 类
from ..utils._param_validation import Interval, StrOptions
# 从 utils.fixes 模块中引入 parse_version 和 sp_version 函数
from ..utils.fixes import parse_version, sp_version
# 从 utils.validation 模块中引入 _check_sample_weight 函数
from ..utils.validation import _check_sample_weight
# 从当前包中的 _base 模块引入 LinearModel 类
from ._base import LinearModel

# 定义 QuantileRegressor 类，继承自 LinearModel、RegressorMixin 和 BaseEstimator
class QuantileRegressor(LinearModel, RegressorMixin, BaseEstimator):
    """Linear regression model that predicts conditional quantiles.

    The linear :class:`QuantileRegressor` optimizes the pinball loss for a
    desired `quantile` and is robust to outliers.

    This model uses an L1 regularization like
    :class:`~sklearn.linear_model.Lasso`.

    Read more in the :ref:`User Guide <quantile_regression>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    quantile : float, default=0.5
        The quantile that the model tries to predict. It must be strictly
        between 0 and 1. If 0.5 (default), the model predicts the 50%
        quantile, i.e. the median.

    alpha : float, default=1.0
        Regularization constant that multiplies the L1 penalty term.

    fit_intercept : bool, default=True
        Whether or not to fit the intercept.

    solver : {'highs-ds', 'highs-ipm', 'highs', 'interior-point', \
            'revised simplex'}, default='highs'
        Method used by :func:`scipy.optimize.linprog` to solve the linear
        programming formulation.

        From `scipy>=1.6.0`, it is recommended to use the highs methods because
        they are the fastest ones. Solvers "highs-ds", "highs-ipm" and "highs"
        support sparse input data and, in fact, always convert to sparse csc.

        From `scipy>=1.11.0`, "interior-point" is not available anymore.

        .. versionchanged:: 1.4
           The default of `solver` changed to `"highs"` in version 1.4.

    solver_options : dict, default=None
        Additional parameters passed to :func:`scipy.optimize.linprog` as
        options. If `None` and if `solver='interior-point'`, then
        `{"lstsq": True}` is passed to :func:`scipy.optimize.linprog` for the
        sake of stability.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the features.

    intercept_ : float
        The intercept of the model, aka bias term.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The actual number of iterations performed by the solver.

    See Also
    --------
    ```
    # 参数约束字典，定义了每个参数的类型和取值范围
    _parameter_constraints: dict = {
        "quantile": [Interval(Real, 0, 1, closed="neither")],  # quantile 参数为实数，取值范围在 (0, 1) 开区间
        "alpha": [Interval(Real, 0, None, closed="left")],  # alpha 参数为非负实数
        "fit_intercept": ["boolean"],  # fit_intercept 参数为布尔类型
        "solver": [  # solver 参数为字符串，限定于指定的几个选项
            StrOptions(
                {
                    "highs-ds",
                    "highs-ipm",
                    "highs",
                    "interior-point",
                    "revised simplex",
                }
            ),
        ],
        "solver_options": [dict, None],  # solver_options 参数可以为字典类型或者 None
    }
    
    def __init__(
        self,
        *,
        quantile=0.5,
        alpha=1.0,
        fit_intercept=True,
        solver="highs",
        solver_options=None,
    ):
        # 初始化函数，设置模型的参数
        self.quantile = quantile  # 设置 quantile 参数
        self.alpha = alpha  # 设置 alpha 参数
        self.fit_intercept = fit_intercept  # 设置 fit_intercept 参数
        self.solver = solver  # 设置 solver 参数
        self.solver_options = solver_options  # 设置 solver_options 参数
    
    @_fit_context(prefer_skip_nested_validation=True)
```