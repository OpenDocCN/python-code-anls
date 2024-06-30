# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\_polynomial.py`

```
"""
This file contains preprocessing tools based on polynomials.
"""

import collections  # 导入 collections 模块，用于特定数据结构和操作
from itertools import chain, combinations  # 导入 itertools 模块的 chain 和 combinations 函数
from itertools import combinations_with_replacement as combinations_w_r  # 导入 itertools 模块的 combinations_with_replacement 函数，并重命名为 combinations_w_r
from numbers import Integral  # 导入 numbers 模块的 Integral 类型

import numpy as np  # 导入 NumPy 库，并重命名为 np
from scipy import sparse  # 导入 SciPy 库的 sparse 模块
from scipy.interpolate import BSpline  # 导入 SciPy 库的 BSpline 插值函数
from scipy.special import comb  # 导入 SciPy 库的 comb 函数

from ..base import BaseEstimator, TransformerMixin, _fit_context  # 从相对路径的 ..base 模块中导入 BaseEstimator、TransformerMixin 和 _fit_context
from ..utils import check_array  # 从相对路径的 ..utils 模块中导入 check_array 函数
from ..utils._param_validation import Interval, StrOptions  # 从相对路径的 ..utils._param_validation 模块中导入 Interval 和 StrOptions 类
from ..utils.fixes import parse_version, sp_version  # 从相对路径的 ..utils.fixes 模块中导入 parse_version 和 sp_version 函数
from ..utils.stats import _weighted_percentile  # 从相对路径的 ..utils.stats 模块中导入 _weighted_percentile 函数
from ..utils.validation import (  # 从相对路径的 ..utils.validation 模块中导入以下函数和变量
    FLOAT_DTYPES,
    _check_feature_names_in,
    _check_sample_weight,
    check_is_fitted,
)
from ._csr_polynomial_expansion import (  # 从相对路径的 ._csr_polynomial_expansion 模块中导入以下函数
    _calc_expanded_nnz,
    _calc_total_nnz,
    _csr_polynomial_expansion,
)

__all__ = [  # 将以下类和函数添加到模块的公共接口
    "PolynomialFeatures",
    "SplineTransformer",
]


def _create_expansion(X, interaction_only, deg, n_features, cumulative_size=0):
    """Helper function for creating and appending sparse expansion matrices"""

    total_nnz = _calc_total_nnz(X.indptr, interaction_only, deg)  # 计算总的非零元素数目
    expanded_col = _calc_expanded_nnz(n_features, interaction_only, deg)  # 计算扩展后的列数

    if expanded_col == 0:  # 如果扩展后的列数为0，返回空值
        return None
    
    # 下面的代码段检查每个块在扩展时是否需要64位整数索引。优先保留int32索引，因为当前SciPy的CSR构造在可能时会进行降级转换，因此避免不必要的类型转换。如果需要，拼接过程中的数据类型仍可能更改。
    max_indices = expanded_col - 1
    max_indptr = total_nnz
    max_int32 = np.iinfo(np.int32).max
    needs_int64 = max(max_indices, max_indptr) > max_int32
    index_dtype = np.int64 if needs_int64 else np.int32

    # cumulative_size 表示累积的扩展列数，用于检查异常情况
    cumulative_size += expanded_col
    if (
        sp_version < parse_version("1.8.0")
        and cumulative_size - 1 > max_int32
        and not needs_int64
    ):
        raise ValueError(
            "In scipy versions `<1.8.0`, the function `scipy.sparse.hstack`"
            " sometimes produces negative columns when the output shape contains"
            " `n_cols` too large to be represented by a 32bit signed"
            " integer. To avoid this error, either use a version"
            " of scipy `>=1.8.0` or alter the `PolynomialFeatures`"
            " transformer to produce fewer than 2^31 output features."
        )

    # expanded_data 和 expanded_indices 是扩展过程中的结果，由 _csr_polynomial_expansion 函数修改
    expanded_data = np.empty(shape=total_nnz, dtype=X.data.dtype)
    expanded_indices = np.empty(shape=total_nnz, dtype=index_dtype)
    # 创建一个空的 NumPy 数组，用于存储扩展后的 CSR 格式稀疏矩阵的指针数组
    expanded_indptr = np.empty(shape=X.indptr.shape[0], dtype=index_dtype)
    # 调用函数 `_csr_polynomial_expansion` 对 CSR 格式稀疏矩阵 X 进行多项式扩展
    # 扩展使用的数据包括 X 的数据部分、列索引部分、行指针部分，还有扩展后的数据、索引和指针数组
    _csr_polynomial_expansion(
        X.data,                 # 原始稀疏矩阵 X 的数据部分
        X.indices,              # 原始稀疏矩阵 X 的列索引部分
        X.indptr,               # 原始稀疏矩阵 X 的行指针部分
        X.shape[1],             # 原始稀疏矩阵 X 的列数
        expanded_data,          # 扩展后的稀疏矩阵的数据部分
        expanded_indices,       # 扩展后的稀疏矩阵的列索引部分
        expanded_indptr,        # 扩展后的稀疏矩阵的行指针部分
        interaction_only,       # 是否仅考虑交互项的布尔值
        deg,                    # 多项式扩展的度数
    )
    # 返回一个新的 CSR 格式稀疏矩阵，使用扩展后的数据、索引和指针数组
    return sparse.csr_matrix(
        (expanded_data, expanded_indices, expanded_indptr),
        shape=(X.indptr.shape[0] - 1, expanded_col),   # 设置稀疏矩阵的形状，行数为 X 的行数减一，列数为扩展后的列数
        dtype=X.dtype,          # 设置稀疏矩阵的数据类型与原始稀疏矩阵 X 相同
    )
class PolynomialFeatures(TransformerMixin, BaseEstimator):
    """Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Read more in the :ref:`User Guide <polynomial_features>`.

    Parameters
    ----------
    degree : int or tuple (min_degree, max_degree), default=2
        If a single int is given, it specifies the maximal degree of the
        polynomial features. If a tuple `(min_degree, max_degree)` is passed,
        then `min_degree` is the minimum and `max_degree` is the maximum
        polynomial degree of the generated features. Note that `min_degree=0`
        and `min_degree=1` are equivalent as outputting the degree zero term is
        determined by `include_bias`.

    interaction_only : bool, default=False
        If `True`, only interaction features are produced: features that are
        products of at most `degree` *distinct* input features, i.e. terms with
        power of 2 or higher of the same input feature are excluded:

            - included: `x[0]`, `x[1]`, `x[0] * x[1]`, etc.
            - excluded: `x[0] ** 2`, `x[0] ** 2 * x[1]`, etc.

    include_bias : bool, default=True
        If `True` (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    order : {'C', 'F'}, default='C'
        Order of output array in the dense case. `'F'` order is faster to
        compute, but may slow down subsequent estimators.

        .. versionadded:: 0.21

    Attributes
    ----------
    powers_ : ndarray of shape (`n_output_features_`, `n_features_in_`)
        `powers_[i, j]` is the exponent of the jth input in the ith output.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_output_features_ : int
        The total number of polynomial output features. The number of output
        features is computed by iterating over all suitably sized combinations
        of input features.

    See Also
    --------
    SplineTransformer : Transformer that generates univariate B-spline bases
        for features.

    Notes
    -----
    Be aware that the number of features in the output array scales
    polynomially in the number of features of the input array, and
    exponentially in the degree. High degrees can cause overfitting.

    See :ref:`examples/linear_model/plot_polynomial_interpolation.py`
    """
    # Initialize the PolynomialFeatures transformer with specified parameters
    def __init__(self, degree=2, interaction_only=False, include_bias=True, order='C'):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order

    # Fit the PolynomialFeatures transformer to the input data X
    def fit(self, X, y=None):
        # Validate input X and store its shape
        X = self._validate_data(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        return self

    # Transform the input data X into polynomial features
    def transform(self, X):
        # Validate input X and store its shape
        X = self._validate_data(X, accept_sparse=True)

        # Initialize an empty list to store output features
        combinations = []

        # Iterate over all possible combinations of input features up to the specified degree
        for degree in range(1, self.degree + 1):
            comb = combinations_with_replacement(range(self.n_features_in_), degree)
            combinations.extend(comb)

        # Generate polynomial features based on the computed combinations
        X_poly = np.empty((X.shape[0], len(combinations)), dtype=X.dtype, order=self.order)
        for i, comb in enumerate(combinations):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)

        # Store the total number of output features
        self.n_output_features_ = X_poly.shape[1]

        return X_poly

    def get_feature_names_out(self, input_features=None):
        # Return feature names based on the input features
        input_features = np.array(input_features)
        powers = self.powers_.astype(int)
        feature_names_out = []

        for row in powers:
            feature_names = []
            for power, input_name in zip(row, input_features):
                if power == 0:
                    continue
                elif power == 1:
                    feature_names.append(input_name)
                else:
                    feature_names.append(f"{input_name}^{power}")
            feature_names_out.append(" * ".join(feature_names))

        return feature_names_out
    _parameter_constraints: dict = {
        # 参数约束字典，定义了PolynomialFeatures类的各个参数的类型和约束条件
        "degree": [Interval(Integral, 0, None, closed="left"), "array-like"],
        "interaction_only": ["boolean"],
        "include_bias": ["boolean"],
        "order": [StrOptions({"C", "F"})],
    }

    def __init__(
        self, degree=2, *, interaction_only=False, include_bias=True, order="C"
    ):
        # PolynomialFeatures类的初始化方法，设置类的各个属性
        self.degree = degree  # 多项式的次数
        self.interaction_only = interaction_only  # 是否只生成交互项
        self.include_bias = include_bias  # 是否包含截距项
        self.order = order  # 数组的存储顺序

    @staticmethod
    def _combinations(
        n_features, min_degree, max_degree, interaction_only, include_bias
    ):
        # 静态方法，用于生成多项式特征组合
        comb = combinations if interaction_only else combinations_w_r
        start = max(1, min_degree)
        # 使用生成器表达式生成特征组合的迭代器
        iter = chain.from_iterable(
            comb(range(n_features), i) for i in range(start, max_degree + 1)
        )
        if include_bias:
            iter = chain(comb(range(n_features), 0), iter)
        return iter

    @staticmethod
    def _num_combinations(
        n_features, min_degree, max_degree, interaction_only, include_bias
    ):
        """Calculate number of terms in polynomial expansion

        This should be equivalent to counting the number of terms returned by
        _combinations(...) but much faster.
        """
        # 静态方法，计算多项式展开中的项数

        if interaction_only:
            # 如果只有交互项，计算特征组合的数量
            combinations = sum(
                [
                    comb(n_features, i, exact=True)
                    for i in range(max(1, min_degree), min(max_degree, n_features) + 1)
                ]
            )
        else:
            # 否则，计算所有可能的组合数，并减去不满足最小度数要求的组合数
            combinations = comb(n_features + max_degree, max_degree, exact=True) - 1
            if min_degree > 0:
                d = min_degree - 1
                combinations -= comb(n_features + d, d, exact=True) - 1

        if include_bias:
            combinations += 1

        return combinations

    @property


这些注释详细解释了每行代码的作用，包括变量定义、方法功能以及静态方法的具体计算过程和用途。
    # 定义一个方法 `powers_`，用于生成输入的每个特征的指数。
    def powers_(self):
        """Exponent for each of the inputs in the output."""
        # 检查对象是否已经拟合
        check_is_fitted(self)

        # 生成输入特征的组合，根据指定的最小和最大度数、是否仅交互、是否包含偏置等参数
        combinations = self._combinations(
            n_features=self.n_features_in_,
            min_degree=self._min_degree,
            max_degree=self._max_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        
        # 返回一个数组，其中每行表示对应组合的特征的指数
        return np.vstack(
            [np.bincount(c, minlength=self.n_features_in_) for c in combinations]
        )

    # 获取转换后的输出特征名称
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features is None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # 获取特征的指数数组
        powers = self.powers_
        
        # 检查输入的特征名称
        input_features = _check_feature_names_in(self, input_features)
        
        # 初始化特征名称列表
        feature_names = []
        
        # 遍历每行的指数数组
        for row in powers:
            # 找到非零元素的索引
            inds = np.where(row)[0]
            
            # 如果有非零元素
            if len(inds):
                # 构建特征名称，格式为 "特征名^指数"，如果指数为1则省略指数部分
                name = " ".join(
                    (
                        "%s^%d" % (input_features[ind], exp)
                        if exp != 1
                        else input_features[ind]
                    )
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                # 如果没有非零元素，则该特征为常数1
                name = "1"
            
            # 将生成的特征名称添加到列表中
            feature_names.append(name)
        
        # 返回转换后的特征名称数组
        return np.asarray(feature_names, dtype=object)

    # 应用拟合上下文装饰器，跳过嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
class SplineTransformer(TransformerMixin, BaseEstimator):
    """Generate univariate B-spline bases for features.

    Generate a new feature matrix consisting of
    `n_splines=n_knots + degree - 1` (`n_knots - 1` for
    `extrapolation="periodic"`) spline basis functions
    (B-splines) of polynomial order=`degree` for each feature.

    In order to learn more about the SplineTransformer class go to:
    :ref:`sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`

    Read more in the :ref:`User Guide <spline_transformer>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    n_knots : int, default=5
        Number of knots of the splines if `knots` equals one of
        {'uniform', 'quantile'}. Must be larger or equal 2. Ignored if `knots`
        is array-like.

    degree : int, default=3
        The polynomial degree of the spline basis. Must be a non-negative
        integer.

    knots : {'uniform', 'quantile'} or array-like of shape \
        (n_knots, n_features), default='uniform'
        Set knot positions such that first knot <= features <= last knot.

        - If 'uniform', `n_knots` number of knots are distributed uniformly
          from min to max values of the features.
        - If 'quantile', they are distributed uniformly along the quantiles of
          the features.
        - If an array-like is given, it directly specifies the sorted knot
          positions including the boundary knots. Note that, internally,
          `degree` number of knots are added before the first knot, the same
          after the last knot.

    extrapolation : {'error', 'constant', 'linear', 'continue', 'periodic'}, \
        default='constant'
        If 'error', values outside the min and max values of the training
        features raises a `ValueError`. If 'constant', the value of the
        splines at minimum and maximum value of the features is used as
        constant extrapolation. If 'linear', a linear extrapolation is used.
        If 'continue', the splines are extrapolated as is, i.e. option
        `extrapolate=True` in :class:`scipy.interpolate.BSpline`. If
        'periodic', periodic splines with a periodicity equal to the distance
        between the first and last knot are used. Periodic splines enforce
        equal function values and derivatives at the first and last knot.
        For example, this makes it possible to avoid introducing an arbitrary
        jump between Dec 31st and Jan 1st in spline features derived from a
        naturally periodic "day-of-year" input feature. In this case it is
        recommended to manually set the knot values to control the period.
    """
    # _parameter_constraints 是一个字典，用于定义 SplineTransformer 的参数约束条件

    _parameter_constraints: dict = {
        # "n_knots" 的约束条件，必须为整数且大于等于 2
        "n_knots": [Interval(Integral, 2, None, closed="left")],
        
        # "degree" 的约束条件，必须为整数且大于等于 0
        "degree": [Interval(Integral, 0, None, closed="left")],
        
        # "knots" 的约束条件，可以是 "uniform" 或 "quantile" 中的一种，并且是一个类数组对象
        "knots": [StrOptions({"uniform", "quantile"}), "array-like"],
        
        # "extrapolation" 的约束条件，可以是 {"error", "constant", "linear", "continue", "periodic"} 中的一种
        "extrapolation": [
            StrOptions({"error", "constant", "linear", "continue", "periodic"})
        ],
        
        # "include_bias" 的约束条件，必须是布尔值
        "include_bias": ["boolean"],
        
        # "order" 的约束条件，可以是 {"C", "F"} 中的一种
        "order": [StrOptions({"C", "F"})],
        
        # "sparse_output" 的约束条件，必须是布尔值
        "sparse_output": ["boolean"],
    }
    def __init__(
        self,
        n_knots=5,
        degree=3,
        *,
        knots="uniform",
        extrapolation="constant",
        include_bias=True,
        order="C",
        sparse_output=False,
    ):
        # 初始化 B-spline 拟合器的参数
        self.n_knots = n_knots  # 设置 B-spline 拟合中的节点数目
        self.degree = degree  # 设置 B-spline 拟合的多项式阶数
        self.knots = knots  # 设置节点位置的类型，默认为均匀分布
        self.extrapolation = extrapolation  # 设置外推方式，默认为常数外推
        self.include_bias = include_bias  # 设置是否包含偏置项，默认为True
        self.order = order  # 设置输出的顺序，默认为C（C风格的行优先顺序）
        self.sparse_output = sparse_output  # 设置是否使用稀疏矩阵作为输出，默认为False

    @staticmethod
    def _get_base_knot_positions(X, n_knots=10, knots="uniform", sample_weight=None):
        """Calculate base knot positions.

        Base knots such that first knot <= feature <= last knot. For the
        B-spline construction with scipy.interpolate.BSpline, 2*degree knots
        beyond the base interval are added.

        Returns
        -------
        knots : ndarray of shape (n_knots, n_features), dtype=np.float64
            Knot positions (points) of base interval.
        """
        if knots == "quantile":
            # 如果节点类型为分位数类型，计算基础节点位置
            percentiles = 100 * np.linspace(
                start=0, stop=1, num=n_knots, dtype=np.float64
            )

            if sample_weight is None:
                knots = np.percentile(X, percentiles, axis=0)
            else:
                knots = np.array(
                    [
                        _weighted_percentile(X, sample_weight, percentile)
                        for percentile in percentiles
                    ]
                )

        else:
            # 如果节点类型为均匀分布，默认处理
            # 注意：这里的变量 `knots` 已经经过验证，因此 `else` 分支是安全的。
            # 忽略权重为零的观测。
            mask = slice(None, None, 1) if sample_weight is None else sample_weight > 0
            x_min = np.amin(X[mask], axis=0)
            x_max = np.amax(X[mask], axis=0)

            knots = np.linspace(
                start=x_min,
                stop=x_max,
                num=n_knots,
                endpoint=True,
                dtype=np.float64,
            )

        return knots
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
            Transformed feature names.
        """
        # Ensure the model is fitted and determine the number of splines
        check_is_fitted(self, "n_features_in_")
        n_splines = self.bsplines_[0].c.shape[1]

        # Validate input feature names against `feature_names_in_`
        input_features = _check_feature_names_in(self, input_features)
        
        # Initialize an empty list to store feature names
        feature_names = []
        
        # Iterate over input features to generate transformed feature names
        for i in range(self.n_features_in_):
            # Iterate over splines and bias terms to create feature names
            for j in range(n_splines - 1 + self.include_bias):
                feature_names.append(f"{input_features[i]}_sp_{j}")
        
        # Convert the list of feature names to a numpy array
        return np.asarray(feature_names, dtype=object)

    @_fit_context(prefer_skip_nested_validation=True)
    def _more_tags(self):
        """Provide additional tags for the estimator."""
        # Return a dictionary with extra tags for the estimator
        return {
            "_xfail_checks": {
                "check_estimators_pickle": (
                    "Current Scipy implementation of _bsplines does not"
                    "support const memory views."
                ),
            }
        }
```