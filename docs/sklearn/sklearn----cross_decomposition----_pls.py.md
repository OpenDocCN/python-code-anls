# `D:\src\scipysrc\scikit-learn\sklearn\cross_decomposition\_pls.py`

```
"""
The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 引入警告模块，用于处理警告信息
import warnings
# 引入抽象基类相关的元类和抽象方法装饰器
from abc import ABCMeta, abstractmethod
# 引入整数和实数相关的类型
from numbers import Integral, Real

# 引入数值计算和科学计算相关的库
import numpy as np
from scipy.linalg import svd

# 引入scikit-learn基础模块中的相关类和方法
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
)
# 引入scikit-learn中的收敛警告异常
from ..exceptions import ConvergenceWarning
# 引入scikit-learn中的数组验证和长度一致性检查工具
from ..utils import check_array, check_consistent_length
# 引入参数验证相关的工具模块
from ..utils._param_validation import Interval, StrOptions
# 引入数值计算中的矩阵分解和翻转操作
from ..utils.extmath import svd_flip
# 引入scipy版本解析和版本检查工具
from ..utils.fixes import parse_version, sp_version
# 引入scikit-learn中的模型验证方法
from ..utils.validation import FLOAT_DTYPES, check_is_fitted

__all__ = ["PLSCanonical", "PLSRegression", "PLSSVD"]

# 根据SciPy版本选择合适的伪逆计算方法
if sp_version >= parse_version("1.7"):
    # 从SciPy 1.7开始，pinv2已被弃用，推荐使用pinv计算伪逆
    from scipy.linalg import pinv as pinv2
else:
    # 对于较早版本的SciPy，仍然使用pinv2来计算伪逆
    from scipy.linalg import pinv2


def _pinv2_old(a):
    # 使用旧版本的SciPy pinv2方法，详见：
    # https://github.com/scipy/scipy/pull/10067
    # 在SciPy >= 1.3的版本中，不能设置`cond`或`rcond`以保持与旧版本相同的行为，
    # 因为条件判断依赖于svd的输出。
    
    # 对输入矩阵进行奇异值分解
    u, s, vh = svd(a, full_matrices=False, check_finite=False)

    # 选择合适的类型以确定条件阈值
    t = u.dtype.char.lower()
    factor = {"f": 1e3, "d": 1e6}
    cond = np.max(s) * factor[t] * np.finfo(t).eps
    # 计算奇异值大于阈值的个数，确定有效秩
    rank = np.sum(s > cond)

    # 截取有效部分并计算伪逆
    u = u[:, :rank]
    u /= s[:rank]
    return np.transpose(np.conjugate(np.dot(u, vh[:rank])))


def _get_first_singular_vectors_power_method(
    X, Y, mode="A", max_iter=500, tol=1e-06, norm_y_weights=False
):
    """Return the first left and right singular vectors of X'Y.

    Provides an alternative to the svd(X'Y) and uses the power method instead.
    With norm_y_weights to True and in mode A, this corresponds to the
    algorithm section 11.3 of the Wegelin's review, except this starts at the
    "update saliences" part.
    """

    # 确定机器精度
    eps = np.finfo(X.dtype).eps
    try:
        # 选择第一个非常数的y分量作为y的得分向量
        y_score = next(col for col in Y.T if np.any(np.abs(col) > eps))
    except StopIteration as e:
        # 如果y的残差为常数，则引发异常
        raise StopIteration("y residual is constant") from e

    # 初始化X权重的旧值，用于第一次收敛检查
    x_weights_old = 100

    if mode == "B":
        # 预先计算伪逆矩阵X_pinv和Y_pinv
        # 基本上：X_pinv = (X.T X)^-1 X.T
        # 这需要反转一个(n_features, n_features)的矩阵。
        # 正如Wegelin的评论中详细描述的那样，如果n_features > n_samples或n_targets > n_samples，则CCA（即模式B）将不稳定。
        X_pinv, Y_pinv = _pinv2_old(X), _pinv2_old(Y)
    # 对于给定的最大迭代次数循环
    for i in range(max_iter):
        # 如果模式为 "B"，使用 X 的伪逆和 y_score 的点积作为 x_weights
        if mode == "B":
            x_weights = np.dot(X_pinv, y_score)
        else:
            # 否则，使用 X 的转置与 y_score 的点积除以 y_score 的点积作为 x_weights
            x_weights = np.dot(X.T, y_score) / np.dot(y_score, y_score)

        # 对 x_weights 进行归一化处理，并加上一个小的 eps 避免除零错误
        x_weights /= np.sqrt(np.dot(x_weights, x_weights)) + eps
        # 计算 x_score，即 X 与 x_weights 的点积
        x_score = np.dot(X, x_weights)

        # 如果模式为 "B"，使用 Y 的伪逆和 x_score 的点积作为 y_weights
        if mode == "B":
            y_weights = np.dot(Y_pinv, x_score)
        else:
            # 否则，使用 Y 的转置与 x_score 的点积除以 x_score 的点积作为 y_weights
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)

        # 如果需要对 y_weights 进行归一化处理，则除以其模的平方加上 eps
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights, y_weights)) + eps

        # 计算新的 y_score，即 Y 与 y_weights 的点积除以 y_weights 的模的平方加上 eps
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights, y_weights) + eps)

        # 计算当前 x_weights 与上一次迭代的 x_weights 的差值
        x_weights_diff = x_weights - x_weights_old
        # 如果差值的平方和小于给定的容差 tol 或者 Y 的列数为 1，则终止迭代
        if np.dot(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        # 更新上一次的 x_weights
        x_weights_old = x_weights

    # 记录迭代次数（从零开始计数）
    n_iter = i + 1
    # 如果达到了最大迭代次数，则发出警告
    if n_iter == max_iter:
        warnings.warn("Maximum number of iterations reached", ConvergenceWarning)

    # 返回计算得到的 x_weights、y_weights 和迭代次数 n_iter
    return x_weights, y_weights, n_iter
# 返回矩阵 X'Y 的第一个左奇异向量和右奇异向量
def _get_first_singular_vectors_svd(X, Y):
    """Return the first left and right singular vectors of X'Y.

    Here the whole SVD is computed.
    """
    # 计算矩阵 C = X^T * Y
    C = np.dot(X.T, Y)
    # 对矩阵 C 进行奇异值分解 (SVD)，仅保留必要的部分
    U, _, Vt = svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :]


# 居中和缩放矩阵 X 和 Y，如果 scale 参数为 True 的话
# 返回居中后的 X 和 Y，以及它们的均值和标准差
def _center_scale_xy(X, Y, scale=True):
    """Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # 计算 X 和 Y 的均值
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # 如果需要缩放
    if scale:
        # 计算 X 和 Y 的标准差，确保不为零
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        # 如果不需要缩放，则标准差设为 1
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


# 在 1 维数组上执行 svd_flip 的功能，且是原地操作
def _svd_flip_1d(u, v):
    """Same as svd_flip but works on 1d arrays, and is inplace"""
    # svd_flip 强制我们将数组转换为 2 维数组，并返回 2 维数组，这里我们不需要那样的功能
    # 找到绝对值最大的索引
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign


# TODO(1.7): Remove
# 当 Y 是可选参数时，发出警告并返回 Y
def _deprecate_Y_when_optional(y, Y):
    if Y is not None:
        warnings.warn(
            "`Y` is deprecated in 1.5 and will be removed in 1.7. Use `y` instead.",
            FutureWarning,
        )
        if y is not None:
            raise ValueError(
                "Cannot use both `y` and `Y`. Use only `y` as `Y` is deprecated."
            )
        return Y
    return y


# TODO(1.7): Remove
# 当 Y 是必需参数时，返回 Y 或引发 ValueError
def _deprecate_Y_when_required(y, Y):
    if y is None and Y is None:
        raise ValueError("y is required.")
    return _deprecate_Y_when_optional(y, Y)


# 实现偏最小二乘（Partial Least Squares, PLS）算法的基础类
class _PLS(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
    BaseEstimator,
    metaclass=ABCMeta,
):
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm.

    Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
    with emphasis on the two-block case
    https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf
    """

    # 参数的约束条件字典，指定每个参数的类型和取值范围
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": ["boolean"],
        "deflation_mode": [StrOptions({"regression", "canonical"})],
        "mode": [StrOptions({"A", "B"})],
        "algorithm": [StrOptions({"svd", "nipals"})],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "copy": ["boolean"],
    }

    @abstractmethod
    def __init__(
        self,
        n_components=2,
        *,
        scale=True,
        deflation_mode="regression",
        mode="A",
        algorithm="nipals",
        max_iter=500,
        tol=1e-06,
        copy=True,
    ):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy



# 初始化函数，设置主成分分析器的各项参数
def __init__(
        self, n_components=None, deflation_mode="regression", mode="a", scale=True,
        algorithm="svd", max_iter=None, tol=0.0, copy=True
    ):
        # 设置主成分数量
        self.n_components = n_components
        # 设置deflation模式
        self.deflation_mode = deflation_mode
        # 设置模式
        self.mode = mode
        # 是否进行标准化
        self.scale = scale
        # 使用的算法
        self.algorithm = algorithm
        # 最大迭代次数
        self.max_iter = max_iter
        # 容差
        self.tol = tol
        # 是否复制数据
        self.copy = copy



    @_fit_context(prefer_skip_nested_validation=True)
    def transform(self, X, y=None, Y=None, copy=True):
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

        Y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

            .. deprecated:: 1.5
               `Y` is deprecated in 1.5 and will be removed in 1.7. Use `y` instead.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        x_scores, y_scores : array-like or tuple of array-like
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        # 处理过时的参数 `Y`，使用 `y` 替代
        y = _deprecate_Y_when_optional(y, Y)

        # 检查模型是否已拟合
        check_is_fitted(self)
        # 验证输入数据，并进行数据类型验证和复制
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # 数据标准化
        X -= self._x_mean
        X /= self._x_std
        # 应用旋转变换
        x_scores = np.dot(X, self.x_rotations_)
        # 如果有目标向量 `y`，也对其进行相同的处理
        if y is not None:
            # 验证和处理目标向量 `y`
            y = check_array(
                y, input_name="y", ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES
            )
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y -= self._y_mean
            y /= self._y_std
            # 应用旋转变换到目标向量
            y_scores = np.dot(y, self.y_rotations_)
            return x_scores, y_scores

        # 返回主成分分析后的结果
        return x_scores



        # transform 方法结束



        # 类定义结束
    # 将数据反向转换回原始空间。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_components)
        新数据，其中 `n_samples` 是样本数，`n_components` 是 PLS 组件的数量。

    y : array-like of shape (n_samples,) or (n_samples, n_components), optional
        新的目标数据，其中 `n_samples` 是样本数，`n_components` 是 PLS 组件的数量。

    Y : array-like of shape (n_samples, n_components), optional
        新的目标数据，其中 `n_samples` 是样本数，`n_components` 是 PLS 组件的数量。
        
        .. deprecated:: 1.5
           在 1.5 版本中弃用 `Y`，将在 1.7 版本中移除。请使用 `y` 替代。

    Returns
    -------
    X_reconstructed : ndarray of shape (n_samples, n_features)
        返回重建的 `X` 数据。

    y_reconstructed : ndarray of shape (n_samples, n_targets), optional
        返回重建的 `y` 数据。仅在给定 `y` 的情况下返回。

    Notes
    -----
    如果 `n_components=n_features`，则此转换只有在这种情况下才是精确的。
    """
    # 在需要时处理 `Y` 的过时警告
    y = _deprecate_Y_when_optional(y, Y)

    # 检查模型是否已拟合
    check_is_fitted(self)

    # 检查并转换输入数据 `X` 为浮点类型数组
    X = check_array(X, input_name="X", dtype=FLOAT_DTYPES)

    # 从 PLS 空间转换回原始空间
    X_reconstructed = np.matmul(X, self.x_loadings_.T)

    # 反标准化 `X_reconstructed`
    X_reconstructed *= self._x_std
    X_reconstructed += self._x_mean

    if y is not None:
        # 如果存在 `y`，检查并转换为浮点类型数组
        y = check_array(y, input_name="y", dtype=FLOAT_DTYPES)

        # 从 PLS 空间转换回原始空间
        y_reconstructed = np.matmul(y, self.y_loadings_.T)

        # 反标准化 `y_reconstructed`
        y_reconstructed *= self._y_std
        y_reconstructed += self._y_mean

        return X_reconstructed, y_reconstructed

    # 如果没有给定 `y`，则只返回 `X_reconstructed`
    return X_reconstructed


# 预测给定样本的目标值
def predict(self, X, copy=True):
    """Predict targets of given samples.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        样本数据。

    copy : bool, default=True
        是否复制 `X` 和 `Y`，或者进行原地归一化。

    Returns
    -------
    y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
        返回预测的值。

    Notes
    -----
    此调用需要估计一个形状为 `(n_features, n_targets)` 的系数矩阵，这在高维空间中可能会有问题。
    """
    # 检查模型是否已拟合
    check_is_fitted(self)

    # 验证并转换输入数据 `X`，可以选择复制数据，确保数据类型为浮点数
    X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)

    # 只对 `X` 中心化，不缩放，因为系数已经被缩放过
    X -= self._x_mean

    # 预测 `Ypred`，使用系数和截距
    Ypred = X @ self.coef_.T + self.intercept_

    return Ypred.ravel() if self._predict_1d else Ypred
    # 学习并在训练数据上应用维度减少操作
    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : ndarray of shape (n_samples, n_components)
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        # 调用 fit 方法学习数据的特征，并应用转换
        return self.fit(X, y).transform(X, y)

    # 返回一个包含额外标签的字典
    def _more_tags(self):
        return {"poor_score": True, "requires_y": False}
# 定义 PLS 回归类，继承自 _PLS 类
class PLSRegression(_PLS):
    """PLS regression.

    PLSRegression is also known as PLS2 or PLS1, depending on the number of
    targets.

    For a comparison between other cross decomposition algorithms, see
    :ref:`sphx_glr_auto_examples_cross_decomposition_plot_compare_cross_decomposition.py`.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, n_features]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in :term:`fit` before applying centering,
        and potentially scaling. If `False`, these operations will be done
        inplace, modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_target, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    """
    # 类初始化方法，定义 PLS 回归的各个参数和属性
    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True):
        # 调用父类的初始化方法
        super().__init__()
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    # 略去其它方法和实现细节
    # 定义了一个名为 PLSCanonical 的类，该类用于做偏最小二乘变换和回归。

    # 从 sklearn.cross_decomposition 模块导入 PLSRegression 类
    from sklearn.cross_decomposition import PLSRegression
    # 定义示例数据集 X 和对应的目标值 y
    X = [[0., 0., 1.], [1., 0., 0.], [2., 2., 2.], [2., 5., 4.]]
    y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    # 创建一个 PLSRegression 对象 pls2，设置主成分数量为 2
    pls2 = PLSRegression(n_components=2)
    # 对数据 X 和目标值 y 进行拟合
    pls2.fit(X, y)
    # 打印拟合后的对象 pls2
    PLSRegression()
    # 使用拟合后的模型进行预测
    Y_pred = pls2.predict(X)

    # 用于比较 PLS 回归和 sklearn.decomposition.PCA 之间的差异，详见指定的文档链接。
    """

    # 从 _PLS 类的 _parameter_constraints 字典复制内容到 _parameter_constraints 字典中
    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    # 移除特定参数 ("deflation_mode", "mode", "algorithm")，这些参数不在当前类中使用
    for param in ("deflation_mode", "mode", "algorithm"):
        _parameter_constraints.pop(param)

    # 此实现与 R 语言中的 3 个 PLS 包提供的结果一致：
    #     - "mixOmics" 中的 pls(X, Y, mode = "regression")
    #     - "plspm " 中的 plsreg2(X, Y)
    #     - "pls" 中的 oscorespls.fit(X, Y)

    # 初始化方法，设置初始参数
    def __init__(
        self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True
    ):
        # 调用父类的初始化方法，设置参数
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="regression",
            mode="A",
            algorithm="nipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )

    # 拟合模型到数据的方法
    def fit(self, X, y=None, Y=None):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

            .. deprecated:: 1.5
               `Y` is deprecated in 1.5 and will be removed in 1.7. Use `y` instead.

        Returns
        -------
        self : object
            Fitted model.
        """
        # 根据需要，调整参数 y 和 Y 的使用
        y = _deprecate_Y_when_required(y, Y)

        # 调用父类的拟合方法，传入数据 X 和目标值 y
        super().fit(X, y)
        # 暴露出拟合后的属性 `x_scores_` 和 `y_scores_`
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        # 返回拟合后的对象本身
        return self
class PLSCanonical(_PLS):
    """Partial Least Squares transformer and regressor.

    For a comparison between other cross decomposition algorithms, see
    :ref:`sphx_glr_auto_examples_cross_decomposition_plot_compare_cross_decomposition.py`.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    algorithm : {'nipals', 'svd'}, default='nipals'
        The algorithm used to estimate the first singular vectors of the
        cross-covariance matrix. 'nipals' uses the power method while 'svd'
        will compute the whole SVD.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component. Empty if `algorithm='svd'`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0
    """
    # 以下是一个多行注释，描述了该类的相关信息，包括相关的函数和示例用法
    """
    See Also
    --------
    CCA : Canonical Correlation Analysis.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSCanonical
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> plsca = PLSCanonical(n_components=2)
    >>> plsca.fit(X, y)
    PLSCanonical()
    >>> X_c, y_c = plsca.transform(X, y)
    """

    # 创建一个参数约束字典，从_PLS._parameter_constraints继承而来，并排除了"deflation_mode"和"mode"两个参数
    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    for param in ("deflation_mode", "mode"):
        _parameter_constraints.pop(param)

    # 这个实现提供了与R语言中"plspm"包中的plsca(X, Y)函数相同的结果。
    # 结果等同于或共线于"mixOmics"包中的``pls(..., mode = "canonical")``函数。
    # 区别在于mixOmics的实现没有完全实现Wold算法，因为它没有将y_weights归一化为1。

    # 初始化方法，设定了一些参数，调用了父类的初始化方法
    def __init__(
        self,
        n_components=2,
        *,
        scale=True,
        algorithm="nipals",
        max_iter=500,
        tol=1e-06,
        copy=True,
    ):
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="canonical",
            mode="A",
            algorithm=algorithm,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )
# CCA 类定义，继承自 _PLS 类
class CCA(_PLS):
    """Canonical Correlation Analysis, also known as "Mode B" PLS.

    For a comparison between other cross decomposition algorithms, see
    :ref:`sphx_glr_auto_examples_cross_decomposition_plot_compare_cross_decomposition.py`.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    max_iter : int, default=500
        The maximum number of iterations of the power method.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from sklearn.cross_decomposition import CCA
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    # 创建一个名为 cca 的 CCA 对象，设置主成分数量为 1
    >>> cca = CCA(n_components=1)
    # 使用给定的 X 和 y 数据拟合 CCA 模型
    >>> cca.fit(X, y)
    # 输出已创建的 CCA 对象的信息
    CCA(n_components=1)
    # 使用已训练的 CCA 模型对 X 和 y 进行变换，得到变换后的 X_c 和 Y_c
    >>> X_c, Y_c = cca.transform(X, y)
    """

    # 创建一个新的字典 _parameter_constraints，并将 _PLS._parameter_constraints 的内容复制进去
    _parameter_constraints: dict = {**_PLS._parameter_constraints}
    # 针对一些特定参数，在 _parameter_constraints 中删除对应的条目
    for param in ("deflation_mode", "mode", "algorithm"):
        _parameter_constraints.pop(param)

    # 初始化函数，设置对象的各种参数
    def __init__(
        self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True
    ):
        # 调用父类的初始化方法，设置对象的主成分数量、是否进行缩放等参数
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="canonical",
            mode="B",
            algorithm="nipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )
class PLSSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Partial Least Square SVD.

    This transformer simply performs a SVD on the cross-covariance matrix
    `X'Y`. It is able to project both the training data `X` and the targets
    `Y`. The training data `X` is projected on the left singular vectors, while
    the targets are projected on the right singular vectors.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If `False`, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    y_weights_ : ndarray of (n_targets, n_components)
        The right singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    CCA : Canonical Correlation Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_decomposition import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...               [1., 0., 0.],
    ...               [2., 2., 2.],
    ...               [2., 5., 4.]])
    >>> y = np.array([[0.1, -0.2],
    ...               [0.9, 1.1],
    ...               [6.2, 5.9],
    ...               [11.9, 12.3]])
    >>> pls = PLSSVD(n_components=2).fit(X, y)
    >>> X_c, y_c = pls.transform(X, y)
    >>> X_c.shape, y_c.shape
    ((4, 2), (4, 2))
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": ["boolean"],
        "copy": ["boolean"],
    }

    def __init__(self, n_components=2, *, scale=True, copy=True):
        # 初始化方法，设置PLSSVD对象的初始属性
        self.n_components = n_components  # 设置要保留的成分数量
        self.scale = scale  # 设置是否对X和Y进行缩放
        self.copy = copy  # 设置是否在拟合之前复制X和Y

    @_fit_context(prefer_skip_nested_validation=True)
    # 使用装饰器定义_fit_context方法，这可能是一个装饰了的内部方法或上下文管理器
    # 将模型拟合到数据上。

    # 根据需要处理过时参数 `Y`，确保兼容性
    y = _deprecate_Y_when_required(y, Y)

    # 检查输入数据 `X` 和目标值 `y` 的长度是否一致
    check_consistent_length(X, y)

    # 验证数据 `X` 的格式，确保其为浮点数类型，并进行必要的拷贝和写入权限设置
    X = self._validate_data(
        X,
        dtype=np.float64,
        force_writeable=True,
        copy=self.copy,
        ensure_min_samples=2,
    )

    # 检查目标值 `y` 的格式，确保其为浮点数类型，并进行必要的拷贝和写入权限设置
    y = check_array(
        y,
        input_name="y",
        dtype=np.float64,
        force_writeable=True,
        copy=self.copy,
        ensure_2d=False,
    )

    # 如果目标值 `y` 的维度为1，则将其转换为二维数组形式
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # 计算交叉协方差矩阵的奇异值分解（SVD），矩阵为 X.T.dot(y)
    # 该矩阵的秩最多为 min(n_samples, n_features, n_targets)，因此 n_components 不能超过这个值
    n_components = self.n_components
    rank_upper_bound = min(X.shape[0], X.shape[1], y.shape[1])
    if n_components > rank_upper_bound:
        raise ValueError(
            f"`n_components` 的上限为 {rank_upper_bound}。"
            f"当前值为 {n_components}。请减少 `n_components`。"
        )

    # 将数据 `X` 和目标值 `y` 居中并进行标准化处理
    X, y, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
        X, y, self.scale
    )

    # 计算交叉协方差矩阵 C = X.T.dot(y) 的 SVD 分解
    C = np.dot(X.T, y)
    U, s, Vt = svd(C, full_matrices=False)

    # 选择前 n_components 个主成分
    U = U[:, :n_components]
    Vt = Vt[:n_components]

    # 翻转奇异向量，以确保符号一致性
    U, Vt = svd_flip(U, Vt)

    # 计算 V 矩阵
    V = Vt.T

    # 设置模型的属性值，表示计算结果
    self.x_weights_ = U
    self.y_weights_ = V
    self._n_features_out = self.x_weights_.shape[1]

    # 返回已拟合的估算器对象本身
    return self
    def transform(self, X, y=None, Y=None):
        """
        Apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to be transformed.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

            .. deprecated:: 1.5
               `Y` is deprecated in 1.5 and will be removed in 1.7. Use `y` instead.

        Returns
        -------
        x_scores : array-like or tuple of array-like
            The transformed data `X_transformed` if `Y is not None`,
            `(X_transformed, Y_transformed)` otherwise.
        """
        # Handle deprecation of `Y` by calling `_deprecate_Y_when_optional`
        y = _deprecate_Y_when_optional(y, Y)
        
        # Ensure the estimator is fitted
        check_is_fitted(self)
        
        # Validate input data `X` and ensure it is of type np.float64
        X = self._validate_data(X, dtype=np.float64, reset=False)
        
        # Standardize `X` using precomputed mean (`_x_mean`) and standard deviation (`_x_std`)
        Xr = (X - self._x_mean) / self._x_std
        
        # Compute scores by projecting standardized `X` onto principal components (`x_weights_`)
        x_scores = np.dot(Xr, self.x_weights_)
        
        # If `y` is provided, transform `y` in a similar manner
        if y is not None:
            # Validate and standardize `y`, reshaping if necessary
            y = check_array(y, input_name="y", ensure_2d=False, dtype=np.float64)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            yr = (y - self._y_mean) / self._y_std
            
            # Compute scores for `y` using principal components (`y_weights_`)
            y_scores = np.dot(yr, self.y_weights_)
            
            # Return both `x_scores` and `y_scores`
            return x_scores, y_scores
        
        # Return only `x_scores` if `y` is not provided
        return x_scores

    def fit_transform(self, X, y=None):
        """Learn and apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        out : array-like or tuple of array-like
            The transformed data `X_transformed` if `Y is not None`,
            `(X_transformed, Y_transformed)` otherwise.
        """
        # Fit the model on `X` and then transform `X`
        return self.fit(X, y).transform(X, y)
```