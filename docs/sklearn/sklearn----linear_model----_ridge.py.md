# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_ridge.py`

```
"""
Ridge regression
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import numbers  # 导入用于数值判断的模块
import warnings  # 导入警告处理模块
from abc import ABCMeta, abstractmethod  # 导入抽象基类和抽象方法装饰器
from functools import partial  # 导入函数部分应用模块
from numbers import Integral, Real  # 导入整数和实数类型

import numpy as np  # 导入数值计算模块numpy
from scipy import linalg, optimize, sparse  # 导入线性代数、优化和稀疏矩阵模块
from scipy.sparse import linalg as sp_linalg  # 导入稀疏矩阵的线性代数运算模块

from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier  # 导入基础模块中的混合类、回归器混合类、上下文适配器和分类器判断函数
from ..exceptions import ConvergenceWarning  # 导入收敛警告异常类
from ..metrics import check_scoring, get_scorer_names  # 导入评分检查函数和获取评分器名称函数
from ..model_selection import GridSearchCV  # 导入网格搜索交叉验证类
from ..preprocessing import LabelBinarizer  # 导入标签二值化处理器
from ..utils import (  # 导入实用工具函数
    Bunch,
    check_array,
    check_consistent_length,
    check_scalar,
    column_or_1d,
    compute_sample_weight,
    deprecated,
)
from ..utils._array_api import (  # 导入数组API函数
    _is_numpy_namespace,
    _ravel,
    device,
    get_namespace,
    get_namespace_and_device,
)
from ..utils._param_validation import (  # 导入参数验证函数
    Hidden,
    Interval,
    StrOptions,
    validate_params,
)
from ..utils.extmath import row_norms, safe_sparse_dot  # 导入扩展数学函数中的行范数和稀疏矩阵点积函数
from ..utils.fixes import _sparse_linalg_cg  # 导入修复功能中的稀疏矩阵共轭梯度函数
from ..utils.metadata_routing import (  # 导入元数据路由函数
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from ..utils.sparsefuncs import mean_variance_axis  # 导入稀疏函数中的轴均值方差计算函数
from ..utils.validation import _check_sample_weight, check_is_fitted  # 导入验证函数中的样本权重检查和拟合检查函数
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data  # 导入基础模块中的线性分类器混合类、线性模型、数据预处理和数据重新缩放函数
from ._sag import sag_solver  # 导入SAG求解器


def _get_rescaled_operator(X, X_offset, sample_weight_sqrt):
    """Create LinearOperator for matrix products with implicit centering.

    Matrix product `LinearOperator @ coef` returns `(X - X_offset) @ coef`.
    """
    # 定义创建带有隐式中心化矩阵乘积的线性操作器函数

    def matvec(b):
        return X.dot(b) - sample_weight_sqrt * b.dot(X_offset)
        # 矩阵向量乘积函数，返回 (X - X_offset) @ coef

    def rmatvec(b):
        return X.T.dot(b) - X_offset * b.dot(sample_weight_sqrt)
        # 反向矩阵向量乘积函数，返回 (X - X_offset)^T @ coef

    X1 = sparse.linalg.LinearOperator(shape=X.shape, matvec=matvec, rmatvec=rmatvec)
    return X1
    # 返回创建的线性操作器对象


def _solve_sparse_cg(
    X,
    y,
    alpha,
    max_iter=None,
    tol=1e-4,
    verbose=0,
    X_offset=None,
    X_scale=None,
    sample_weight_sqrt=None,
):
    if sample_weight_sqrt is None:
        sample_weight_sqrt = np.ones(X.shape[0], dtype=X.dtype)
        # 如果未提供样本权重的平方根，则默认为每个样本的权重为1

    n_samples, n_features = X.shape  # 获取样本数和特征数

    if X_offset is None or X_scale is None:
        X1 = sp_linalg.aslinearoperator(X)
        # 如果未提供X的偏移量或缩放比例，则使用稀疏线性运算转换X为线性操作器
    else:
        X_offset_scale = X_offset / X_scale
        X1 = _get_rescaled_operator(X, X_offset_scale, sample_weight_sqrt)
        # 否则，根据提供的偏移量和缩放比例创建带有样本权重平方根的重新缩放后的线性操作器

    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)
    # 创建存储系数的数组，形状为 (输出数, 特征数)

    if n_features > n_samples:

        def create_mv(curr_alpha):
            def _mv(x):
                return X1.matvec(X1.rmatvec(x)) + curr_alpha * x
                # 定义矩阵向量乘积函数，返回 (X - X_offset) @ coef + curr_alpha * x

            return _mv

    else:

        def create_mv(curr_alpha):
            def _mv(x):
                return X1.rmatvec(X1.matvec(x)) + curr_alpha * x
                # 定义反向矩阵向量乘积函数，返回 (X - X_offset)^T @ coef + curr_alpha * x

            return _mv
    # 对 y 的每一列进行迭代
    for i in range(y.shape[1]):
        # 获取 y 的第 i 列数据
        y_column = y[:, i]

        # 根据 alpha[i] 创建移动向量函数 mv
        mv = create_mv(alpha[i])

        # 判断特征数是否大于样本数
        if n_features > n_samples:
            # 使用 Kernel Ridge 回归
            # 构建线性操作对象 C，用于求解 w = X.T * inv(X X^t + alpha*Id) y
            C = sp_linalg.LinearOperator(
                (n_samples, n_samples), matvec=mv, dtype=X.dtype
            )
            # 使用稀疏共轭梯度法求解系数 coef，并记录迭代信息 info
            coef, info = _sparse_linalg_cg(C, y_column, rtol=tol)
            # 计算并存储系数 coefs[i]
            coefs[i] = X1.rmatvec(coef)
        else:
            # 使用线性 Ridge 回归
            # 将 y_column 转换为 X1 的相应形式
            y_column = X1.rmatvec(y_column)
            # 构建线性操作对象 C，用于求解 w = inv(X^t X + alpha*Id) * X.T y
            C = sp_linalg.LinearOperator(
                (n_features, n_features), matvec=mv, dtype=X.dtype
            )
            # 使用稀疏共轭梯度法求解系数 coefs[i]，并记录迭代信息 info
            coefs[i], info = _sparse_linalg_cg(C, y_column, maxiter=max_iter, rtol=tol)

        # 如果 info 小于 0，则抛出值错误异常
        if info < 0:
            raise ValueError("Failed with error code %d" % info)

        # 如果 max_iter 为 None，且 info 大于 0，并且 verbose 为真，则发出警告
        if max_iter is None and info > 0 and verbose:
            warnings.warn(
                "sparse_cg did not converge after %d iterations." % info,
                ConvergenceWarning,
            )

    # 返回计算得到的系数数组 coefs
    return coefs
# 使用 LSQR 方法解决 Ridge 回归问题
def _solve_lsqr(
    X,
    y,
    *,
    alpha,
    fit_intercept=True,
    max_iter=None,
    tol=1e-4,
    X_offset=None,
    X_scale=None,
    sample_weight_sqrt=None,
):
    """Solve Ridge regression via LSQR.

    We expect that y is always mean centered.
    If X is dense, we expect it to be mean centered such that we can solve
        ||y - Xw||_2^2 + alpha * ||w||_2^2

    If X is sparse, we expect X_offset to be given such that we can solve
        ||y - (X - X_offset)w||_2^2 + alpha * ||w||_2^2

    With sample weights S=diag(sample_weight), this becomes
        ||sqrt(S) (y - (X - X_offset) w)||_2^2 + alpha * ||w||_2^2
    and we expect y and X to already be rescaled, i.e. sqrt(S) @ y, sqrt(S) @ X. In
    this case, X_offset is the sample_weight weighted mean of X before scaling by
    sqrt(S). The objective then reads
       ||y - (X - sqrt(S) X_offset) w)||_2^2 + alpha * ||w||_2^2
    """
    # 如果未提供样本权重的平方根，则默认为全为1的数组
    if sample_weight_sqrt is None:
        sample_weight_sqrt = np.ones(X.shape[0], dtype=X.dtype)

    # 如果 X 是稀疏矩阵且需要拟合截距，则计算 X 的偏移和缩放比例
    if sparse.issparse(X) and fit_intercept:
        X_offset_scale = X_offset / X_scale
        # 使用 _get_rescaled_operator 函数获取重新缩放后的操作符 X1
        X1 = _get_rescaled_operator(X, X_offset_scale, sample_weight_sqrt)
    else:
        # 否则，不需要修改 X1
        X1 = X

    # 获取样本数和特征数
    n_samples, n_features = X.shape
    # 初始化系数矩阵和迭代次数数组
    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)
    n_iter = np.empty(y.shape[1], dtype=np.int32)

    # 根据 lsqr 方法的文档，计算 sqrt(alpha)
    sqrt_alpha = np.sqrt(alpha)

    # 对每一个目标变量进行求解
    for i in range(y.shape[1]):
        # 获取当前目标变量的 y 列向量
        y_column = y[:, i]
        # 使用 sp_linalg.lsqr 求解线性方程组
        info = sp_linalg.lsqr(
            X1, y_column, damp=sqrt_alpha[i], atol=tol, btol=tol, iter_lim=max_iter
        )
        # 将求解得到的系数存入 coefs 数组
        coefs[i] = info[0]
        # 存储迭代次数
        n_iter[i] = info[2]

    # 返回系数矩阵和迭代次数数组
    return coefs, n_iter


# 使用 Cholesky 分解求解 Ridge 回归问题
def _solve_cholesky(X, y, alpha):
    # w = inv(X^t X + alpha*Id) * X.T y
    n_features = X.shape[1]
    n_targets = y.shape[1]

    # 计算 A = X^t X 和 Xy = X^t y
    A = safe_sparse_dot(X.T, X, dense_output=True)
    Xy = safe_sparse_dot(X.T, y, dense_output=True)

    # 检查 alpha 是否全部相同
    one_alpha = np.array_equal(alpha, len(alpha) * [alpha[0]])

    # 如果 alpha 全部相同，则简化 A 的更新过程
    if one_alpha:
        A.flat[:: n_features + 1] += alpha[0]
        # 使用 Cholesky 分解求解线性方程组
        return linalg.solve(A, Xy, assume_a="pos", overwrite_a=True).T
    else:
        # 否则，对每个目标变量进行求解
        coefs = np.empty([n_targets, n_features], dtype=X.dtype)
        for coef, target, current_alpha in zip(coefs, Xy.T, alpha):
            A.flat[:: n_features + 1] += current_alpha
            # 使用 Cholesky 分解求解线性方程组
            coef[:] = linalg.solve(A, target, assume_a="pos", overwrite_a=False).ravel()
            A.flat[:: n_features + 1] -= current_alpha
        return coefs


# 使用 Cholesky 分解求解核化的 Ridge 回归问题
def _solve_cholesky_kernel(K, y, alpha, sample_weight=None, copy=False):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    n_targets = y.shape[1]

    # 如果需要复制输入矩阵 K，则进行复制操作
    if copy:
        K = K.copy()

    # 将 alpha 转换为至少包含一个元素的数组
    alpha = np.atleast_1d(alpha)
    # 检查 alpha 是否全部相同
    one_alpha = (alpha == alpha[0]).all()
    # 检查是否提供了样本权重
    has_sw = isinstance(sample_weight, np.ndarray) or sample_weight not in [1.0, None]
    if has_sw:
        # 如果有样本权重（sample_weight）
        # 需要直接支持样本权重，因为 K 可能是预先计算的核矩阵。
        sw = np.sqrt(np.atleast_1d(sample_weight))
        # 对标签 y 加权处理
        y = y * sw[:, np.newaxis]
        # 对核矩阵 K 进行加权处理
        K *= np.outer(sw, sw)

    if one_alpha:
        # 如果只有一个惩罚项，可以同时解决多目标问题。
        # 在对角线上加入 alpha[0]，以处理惩罚项
        K.flat[:: n_samples + 1] += alpha[0]

        try:
            # 注意：我们必须使用 overwrite_a=False，以便在抛出 LinAlgError 的情况下
            #       可以使用下面的备用解决方案。
            dual_coef = linalg.solve(K, y, assume_a="pos", overwrite_a=False)
        except np.linalg.LinAlgError:
            # 如果解决双重问题时出现奇异矩阵，会发出警告，并使用最小二乘解决方案。
            warnings.warn(
                "Singular matrix in solving dual problem. Using "
                "least-squares solution instead."
            )
            dual_coef = linalg.lstsq(K, y)[0]

        # 恢复 K 到原始状态，因为计算和存储 K 是昂贵的。
        K.flat[:: n_samples + 1] -= alpha[0]

        if has_sw:
            # 如果有样本权重，需要将 dual_coef 乘以权重。
            dual_coef *= sw[:, np.newaxis]

        # 返回解决的 dual_coef
        return dual_coef
    else:
        # 每个目标有一个惩罚项。需要分别解决每个目标。
        # 初始化保存结果的数组 dual_coefs
        dual_coefs = np.empty([n_targets, n_samples], K.dtype)

        # 对每个目标进行循环处理
        for dual_coef, target, current_alpha in zip(dual_coefs, y.T, alpha):
            # 在对角线上加入当前惩罚项 current_alpha
            K.flat[:: n_samples + 1] += current_alpha

            # 解决当前目标的问题，得到对应的 dual_coef
            dual_coef[:] = linalg.solve(
                K, target, assume_a="pos", overwrite_a=False
            ).ravel()

            # 恢复 K 到原始状态，以备下一个目标的处理
            K.flat[:: n_samples + 1] -= current_alpha

        if has_sw:
            # 如果有样本权重，需要将 dual_coefs 每行乘以权重。
            dual_coefs *= sw[np.newaxis, :]

        # 返回解决的 dual_coefs，转置以匹配预期输出形状
        return dual_coefs.T
def _solve_svd(X, y, alpha, xp=None):
    xp, _ = get_namespace(X, xp=xp)  # 获取并设置计算环境，例如使用CPU或GPU
    U, s, Vt = xp.linalg.svd(X, full_matrices=False)  # 对输入矩阵进行奇异值分解
    idx = s > 1e-15  # 筛选非零奇异值的索引，用于后续计算
    s_nnz = s[idx][:, None]  # 选择非零奇异值，并将其转换为列向量
    UTy = U.T @ y  # 计算U的转置与y的乘积
    d = xp.zeros((s.shape[0], alpha.shape[0]), dtype=X.dtype, device=device(X))  # 初始化权重矩阵d
    d[idx] = s_nnz / (s_nnz**2 + alpha)  # 计算权重矩阵的非零部分
    d_UT_y = d * UTy  # 计算权重矩阵与U的转置乘以y的乘积
    return (Vt.T @ d_UT_y).T  # 返回正则化后的系数


def _solve_lbfgs(
    X,
    y,
    alpha,
    positive=True,
    max_iter=None,
    tol=1e-4,
    X_offset=None,
    X_scale=None,
    sample_weight_sqrt=None,
):
    """Solve ridge regression with LBFGS.

    The main purpose is fitting with forcing coefficients to be positive.
    For unconstrained ridge regression, there are faster dedicated solver methods.
    Note that with positive bounds on the coefficients, LBFGS seems faster
    than scipy.optimize.lsq_linear.
    """
    n_samples, n_features = X.shape  # 获取样本数和特征数

    options = {}
    if max_iter is not None:
        options["maxiter"] = max_iter  # 设置最大迭代次数的选项
    config = {
        "method": "L-BFGS-B",
        "tol": tol,  # 设置容差
        "jac": True,
        "options": options,
    }
    if positive:
        config["bounds"] = [(0, np.inf)] * n_features  # 如果要求系数为正，则设置边界条件

    if X_offset is not None and X_scale is not None:
        X_offset_scale = X_offset / X_scale  # 计算偏移量与缩放比例
    else:
        X_offset_scale = None

    if sample_weight_sqrt is None:
        sample_weight_sqrt = np.ones(X.shape[0], dtype=X.dtype)  # 初始化样本权重平方根为1

    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)  # 初始化系数矩阵

    for i in range(y.shape[1]):
        x0 = np.zeros((n_features,))  # 初始化优化函数的起始点
        y_column = y[:, i]  # 选择当前列的目标值

        def func(w):
            residual = X.dot(w) - y_column  # 计算残差
            if X_offset_scale is not None:
                residual -= sample_weight_sqrt * w.dot(X_offset_scale)  # 考虑偏移量与缩放比例对残差的影响
            f = 0.5 * residual.dot(residual) + 0.5 * alpha[i] * w.dot(w)  # 定义优化目标函数
            grad = X.T @ residual + alpha[i] * w  # 计算梯度
            if X_offset_scale is not None:
                grad -= X_offset_scale * residual.dot(sample_weight_sqrt)  # 考虑偏移量与缩放比例对梯度的影响

            return f, grad

        result = optimize.minimize(func, x0, **config)  # 调用优化器求解
        if not result["success"]:
            warnings.warn(
                (
                    "The lbfgs solver did not converge. Try increasing max_iter "
                    f"or tol. Currently: max_iter={max_iter} and tol={tol}"
                ),
                ConvergenceWarning,
            )
        coefs[i] = result["x"]  # 将优化得到的系数存入结果矩阵

    return coefs  # 返回最终的系数矩阵


def _get_valid_accept_sparse(is_X_sparse, solver):
    if is_X_sparse and solver in ["auto", "sag", "saga"]:
        return "csr"  # 如果输入稀疏矩阵且使用特定求解器，则返回"csr"
    else:
        return ["csr", "csc", "coo"]  # 否则返回所有支持的稀疏矩阵格式
    {
        # 参数字典，包含模型训练所需的各种参数及其类型或取值范围描述
        "X": ["array-like", "sparse matrix", sp_linalg.LinearOperator],
        # 输入特征数据，可以是数组形式或稀疏矩阵，也可以是线性操作符
        "y": ["array-like"],
        # 目标变量数据，一般为数组形式
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        # 正则化参数，可以是实数且大于等于0，或者是数组形式
        "sample_weight": [
            Interval(Real, None, None, closed="neither"),
            "array-like",
            None,
        ],
        # 样本权重，可以是实数区间，也可以是数组形式，或者为空
        "solver": [
            StrOptions(
                {"auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"}
            )
        ],
        # 求解方法，从多个可选字符串中选择
        "max_iter": [Interval(Integral, 0, None, closed="left"), None],
        # 最大迭代次数，可以是大于等于0的整数，或者为空
        "tol": [Interval(Real, 0, None, closed="left")],
        # 迭代停止的容差阈值，为正实数
        "verbose": ["verbose"],
        # 是否显示详细信息的布尔值
        "positive": ["boolean"],
        # 是否强制系数为正的布尔值
        "random_state": ["random_state"],
        # 随机数生成器的种子值
        "return_n_iter": ["boolean"],
        # 是否返回迭代次数的布尔值
        "return_intercept": ["boolean"],
        # 是否返回截距的布尔值
        "check_input": ["boolean"],
    },
    # 是否优先跳过嵌套验证的布尔值，默认为True
    prefer_skip_nested_validation=True,
# 定义岭回归函数，用于求解岭回归方程
def ridge_regression(
    X,
    y,
    alpha,
    *,
    sample_weight=None,
    solver="auto",
    max_iter=None,
    tol=1e-4,
    verbose=0,
    positive=False,
    random_state=None,
    return_n_iter=False,
    return_intercept=False,
    check_input=True,
):
    """通过正规方程法求解岭回归方程。

    详细内容请参阅 :ref:`用户指南 <ridge_regression>`。

    Parameters
    ----------
    X : {array-like, sparse matrix, LinearOperator} of shape \
        (n_samples, n_features)
        训练数据。

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        目标值。

    alpha : float or array-like of shape (n_targets,)
        控制正则化强度的常数，乘以L2项。`alpha`必须是非负浮点数，即在 `[0, inf)` 范围内。

        当 `alpha = 0` 时，目标等同于普通最小二乘法，由 :class:`LinearRegression` 对象解决。
        出于数值原因，不建议在 `Ridge` 对象中使用 `alpha = 0`。相反，应该使用 :class:`LinearRegression` 对象。

        如果传递了一个数组，则惩罚项被假定为特定于目标。因此它们在数量上必须对应。

    sample_weight : float or array-like of shape (n_samples,), default=None
        每个样本的个体权重。如果给定一个浮点数，每个样本将具有相同的权重。如果 sample_weight 不为 None，并且
        solver='auto'，则求解器将设置为 'cholesky'。

        .. versionadded:: 0.17
    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', \
            'sag', 'saga', 'lbfgs'}, default='auto'
        # 求解器，在计算过程中使用的方法：

        - 'auto' 根据数据类型自动选择求解器。

        - 'svd' 使用矩阵 X 的奇异值分解来计算岭回归系数。这是最稳定的求解器，特别是对于奇异矩阵比 'cholesky' 更稳定，但速度较慢。

        - 'cholesky' 使用标准的 scipy.linalg.solve 函数通过 X.T * X 的 Cholesky 分解获得封闭形式的解。

        - 'sparse_cg' 使用在 scipy.sparse.linalg.cg 中找到的共轭梯度法求解器。作为迭代算法，这个求解器比 'cholesky' 更适合处理大规模数据（可以设置 `tol` 和 `max_iter`）。

        - 'lsqr' 使用专用的正则化最小二乘例程 scipy.sparse.linalg.lsqr。它是最快的，并使用迭代过程。

        - 'sag' 使用随机平均梯度下降法，'saga' 使用其改进且无偏差的版本 SAGA。这两种方法也使用迭代过程，在 n_samples 和 n_features 都很大时通常比其他求解器更快。请注意，'sag' 和 'saga' 的快速收敛仅在特征大致相同尺度时才保证。您可以使用 sklearn.preprocessing 中的缩放器预处理数据。

        - 'lbfgs' 使用 L-BFGS-B 算法，在 `scipy.optimize.minimize` 中实现。仅当 `positive` 为 True 时才能使用。

        除 'svd' 外的所有求解器都支持稠密和稀疏数据。然而，只有 'lsqr'、'sag'、'sparse_cg' 和 'lbfgs' 在 `fit_intercept` 为 True 时支持稀疏输入。

        .. versionadded:: 0.17
           添加了随机平均梯度下降法求解器。
        .. versionadded:: 0.19
           添加了 SAGA 求解器。

    max_iter : int, default=None
        # 共轭梯度法求解器的最大迭代次数。对于 'sparse_cg' 和 'lsqr' 求解器，默认值由 scipy.sparse.linalg 决定。对于 'sag' 和 'saga' 求解器，默认值为 1000。对于 'lbfgs' 求解器，默认值为 15000。

    tol : float, default=1e-4
        # 解的精度。注意，对于 'svd' 和 'cholesky' 求解器，`tol` 不起作用。

        .. versionchanged:: 1.2
           为了与其他线性模型一致性，默认值从 1e-3 更改为 1e-4。

    verbose : int, default=0
        # 冗余级别。设置 verbose > 0 将根据所使用的求解器显示额外信息。

    positive : bool, default=False
        # 当设置为 ``True`` 时，强制系数为正数。仅支持 'lbfgs' 求解器在这种情况下使用。
    random_state : int, RandomState instance, default=None
        # 控制随机数生成器的种子或状态，用于在solver为'sag'或'saga'时对数据进行洗牌。
        # 参见术语表中的“随机状态”以获取详细信息。
        用于'sag'或'saga'求解器以对数据进行洗牌的随机种子或实例。

    return_n_iter : bool, default=False
        # 如果为True，则方法还返回`n_iter`，即求解器执行的实际迭代次数。
        # 自版本0.17起添加。
        如果为True，则方法还返回`n_iter`，即求解器执行的实际迭代次数。

    return_intercept : bool, default=False
        # 如果为True且X为稀疏矩阵，则方法还返回截距，并且求解器会自动更改为'sag'。
        # 这仅是用于在稀疏数据中拟合截距的临时修复。对于稠密数据，请在回归之前使用sklearn.linear_model._preprocess_data。
        # 自版本0.17起添加。
        如果为True且X为稀疏矩阵，则方法还返回截距，并且求解器会自动更改为'sag'。

    check_input : bool, default=True
        # 如果为False，则不会检查输入数组X和y。
        # 自版本0.21起添加。
        如果为False，则不会检查输入数组X和y。

    Returns
    -------
    coef : ndarray of shape (n_features,) or (n_targets, n_features)
        # 权重向量或矩阵。

    n_iter : int, optional
        # 求解器执行的实际迭代次数。
        # 仅在`return_n_iter`为True时返回。

    intercept : float or ndarray of shape (n_targets,)
        # 模型的截距。
        # 仅在`return_intercept`为True且X为scipy稀疏数组时返回。

    Notes
    -----
    This function won't compute the intercept.
    # 此函数不会计算截距。

    Regularization improves the conditioning of the problem and
    reduces the variance of the estimates. Larger values specify stronger
    regularization. Alpha corresponds to ``1 / (2C)`` in other linear
    models such as :class:`~sklearn.linear_model.LogisticRegression` or
    :class:`~sklearn.svm.LinearSVC`. If an array is passed, penalties are
    assumed to be specific to the targets. Hence they must correspond in
    number.
    # 正则化改善了问题的条件性，并减少了估计值的方差。较大的值指定了更强的正则化。
    # Alpha在其他线性模型中对应于``1 / (2C)``，例如:class:`~sklearn.linear_model.LogisticRegression`
    # 或:class:`~sklearn.svm.LinearSVC`。如果传递了数组，则惩罚被认为是特定于目标的。因此，它们必须数量上对应。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import ridge_regression
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 4)
    >>> y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.standard_normal(100)
    >>> coef, intercept = ridge_regression(X, y, alpha=1.0, return_intercept=True)
    >>> list(coef)
    [1.9..., -1.0..., -0.0..., -0.0...]
    >>> intercept
    -0.0...
    """
    return _ridge_regression(
        X,
        y,
        alpha,
        sample_weight=sample_weight,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        positive=positive,
        random_state=random_state,
        return_n_iter=return_n_iter,
        return_intercept=return_intercept,
        X_scale=None,
        X_offset=None,
        check_input=check_input,
    )
# 定义岭回归函数，用于拟合线性回归模型并处理正则化
def _ridge_regression(
    X,
    y,
    alpha,
    sample_weight=None,
    solver="auto",
    max_iter=None,
    tol=1e-4,
    verbose=0,
    positive=False,
    random_state=None,
    return_n_iter=False,
    return_intercept=False,
    return_solver=False,
    X_scale=None,
    X_offset=None,
    check_input=True,
    fit_intercept=False,
):
    # 获取命名空间和设备信息，用于数组计算和设备选择
    xp, is_array_api_compliant, device_ = get_namespace_and_device(
        X, y, sample_weight, X_scale, X_offset
    )
    # 检查是否为 NumPy 命名空间
    is_numpy_namespace = _is_numpy_namespace(xp)
    # 检查输入 X 是否为稀疏矩阵
    X_is_sparse = sparse.issparse(X)

    # 检查是否有样本权重
    has_sw = sample_weight is not None

    # 解析并选择适当的求解器
    solver = resolve_solver(solver, positive, return_intercept, X_is_sparse, xp)

    # 如果使用 NumPy 命名空间且 X 不是稀疏矩阵，则转换 X 为 NumPy 数组
    if is_numpy_namespace and not X_is_sparse:
        X = np.asarray(X)

    # 如果不是 NumPy 命名空间且求解器不是 'svd'，则引发错误
    if not is_numpy_namespace and solver != "svd":
        raise ValueError(
            f"Array API dispatch to namespace {xp.__name__} only supports "
            f"solver 'svd'. Got '{solver}'."
        )

    # 如果 positive=True 且求解器不是 'lbfgs'，则引发错误
    if positive and solver != "lbfgs":
        raise ValueError(
            "When positive=True, only 'lbfgs' solver can be used. "
            f"Please change solver {solver} to 'lbfgs' "
            "or set positive=False."
        )

    # 如果求解器是 'lbfgs' 且 positive=False，则引发错误
    if solver == "lbfgs" and not positive:
        raise ValueError(
            "'lbfgs' solver can be used only when positive=True. "
            "Please use another solver."
        )

    # 如果返回截距参数且求解器不是 'sag'，则引发错误
    if return_intercept and solver != "sag":
        raise ValueError(
            "In Ridge, only 'sag' solver can directly fit the "
            "intercept. Please change solver to 'sag' or set "
            "return_intercept=False."
        )

    # 如果需要检查输入，则验证输入数据的类型和稀疏性
    if check_input:
        # 确定数据类型
        _dtype = [xp.float64, xp.float32]
        # 获取接受稀疏输入的配置
        _accept_sparse = _get_valid_accept_sparse(X_is_sparse, solver)
        # 检查并转换输入 X，确保其类型和稀疏性符合要求
        X = check_array(X, accept_sparse=_accept_sparse, dtype=_dtype, order="C")
        # 检查并转换目标 y，确保其类型和维度符合要求
        y = check_array(y, dtype=X.dtype, ensure_2d=False, order=None)
    # 检查输入 X 和 y 的长度是否一致
    check_consistent_length(X, y)

    # 获取样本数和特征数
    n_samples, n_features = X.shape

    # 检查目标 y 的维度是否正确
    if y.ndim > 2:
        raise ValueError("Target y has the wrong shape %s" % str(y.shape))

    # 如果目标 y 是一维数组，则将其重塑为二维数组
    ravel = False
    if y.ndim == 1:
        y = xp.reshape(y, (-1, 1))
        ravel = True

    # 获取目标 y 的样本数和目标数
    n_samples_, n_targets = y.shape

    # 检查 X 和 y 的样本数是否一致
    if n_samples != n_samples_:
        raise ValueError(
            "Number of samples in X and y does not correspond: %d != %d"
            % (n_samples, n_samples_)
        )

    # 如果有样本权重，则验证并调整样本权重的格式
    if has_sw:
        # 检查和调整样本权重的格式，以匹配输入 X 的数据类型
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # 对于除 'sag', 'saga' 之外的其他求解器，通过简单缩放实现样本权重
        if solver not in ["sag", "saga"]:
            X, y, sample_weight_sqrt = _rescale_data(X, y, sample_weight)

    # 一些调用该方法的函数可能会将 alpha 作为单个元素数组传递，这已经在其他地方验证过
    # 检查 alpha 参数是否不为 None，并且不是由 xp.asarray([0.0]) 创建的对象类型
    if alpha is not None and not isinstance(alpha, type(xp.asarray([0.0]))):
        # 将 alpha 参数转换为实数类型，并检查其值是否在指定范围内
        alpha = check_scalar(
            alpha,
            "alpha",
            target_type=numbers.Real,
            min_val=0.0,
            include_boundaries="left",
        )

    # 将 alpha 参数展平，并确保其长度与目标数相符
    alpha = _ravel(xp.asarray(alpha, device=device_, dtype=X.dtype), xp=xp)
    if alpha.shape[0] not in [1, n_targets]:
        # 如果 alpha 的长度与目标数不符合预期，则抛出 ValueError 异常
        raise ValueError(
            "Number of targets and number of penalties do not correspond: %d != %d"
            % (alpha.shape[0], n_targets)
        )

    # 如果 alpha 的长度为 1，但目标数大于 1，则将 alpha 扩展为长度为 n_targets 的数组
    if alpha.shape[0] == 1 and n_targets > 1:
        alpha = xp.full(
            shape=(n_targets,), fill_value=alpha[0], dtype=alpha.dtype, device=device_
        )

    # 初始化迭代次数为 None
    n_iter = None
    # 根据选择的求解器进行不同的求解过程
    if solver == "sparse_cg":
        # 使用稀疏共轭梯度法求解系数
        coef = _solve_sparse_cg(
            X,
            y,
            alpha,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            X_offset=X_offset,
            X_scale=X_scale,
            sample_weight_sqrt=sample_weight_sqrt if has_sw else None,
        )

    elif solver == "lsqr":
        # 使用 LSQR 方法求解系数，并返回迭代次数
        coef, n_iter = _solve_lsqr(
            X,
            y,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            X_offset=X_offset,
            X_scale=X_scale,
            sample_weight_sqrt=sample_weight_sqrt if has_sw else None,
        )

    elif solver == "cholesky":
        # 如果特征数大于样本数，则计算特征矩阵的内积
        if n_features > n_samples:
            K = safe_sparse_dot(X, X.T, dense_output=True)
            try:
                # 使用 Cholesky 分解方法求解系数
                dual_coef = _solve_cholesky_kernel(K, y, alpha)

                # 计算最终系数
                coef = safe_sparse_dot(X.T, dual_coef, dense_output=True).T
            except linalg.LinAlgError:
                # 如果矩阵奇异，使用 SVD 求解器
                solver = "svd"
        else:
            try:
                # 使用 Cholesky 分解方法求解系数
                coef = _solve_cholesky(X, y, alpha)
            except linalg.LinAlgError:
                # 如果矩阵奇异，使用 SVD 求解器
                solver = "svd"
    elif solver in ["sag", "saga"]:
        # 计算所有目标的最大平方和
        max_squared_sum = row_norms(X, squared=True).max()

        # 初始化 coef 数组，存储模型系数
        coef = np.empty((y.shape[1], n_features), dtype=X.dtype)
        # 初始化 n_iter 数组，存储迭代次数
        n_iter = np.empty(y.shape[1], dtype=np.int32)
        # 初始化 intercept 数组，存储截距
        intercept = np.zeros((y.shape[1],), dtype=X.dtype)
        
        # 遍历 alpha 和 y 的转置，其中 alpha 是正则化系数，target 是目标变量
        for i, (alpha_i, target) in enumerate(zip(alpha, y.T)):
            # 设置初始值字典 init，用于 sag_solver 函数
            init = {
                "coef": np.zeros((n_features + int(return_intercept), 1), dtype=X.dtype)
            }
            # 调用 sag_solver 求解线性模型
            coef_, n_iter_, _ = sag_solver(
                X,
                target.ravel(),
                sample_weight,
                "squared",
                alpha_i,
                0,
                max_iter,
                tol,
                verbose,
                random_state,
                False,
                max_squared_sum,
                init,
                is_saga=solver == "saga",
            )
            # 如果需要截距，更新 coef 和 intercept
            if return_intercept:
                coef[i] = coef_[:-1]
                intercept[i] = coef_[-1]
            else:
                coef[i] = coef_
            # 更新迭代次数数组
            n_iter[i] = n_iter_

        # 如果 intercept 只有一个元素，将其转为标量
        if intercept.shape[0] == 1:
            intercept = intercept[0]

    elif solver == "lbfgs":
        # 调用 _solve_lbfgs 函数求解线性模型
        coef = _solve_lbfgs(
            X,
            y,
            alpha,
            positive=positive,
            tol=tol,
            max_iter=max_iter,
            X_offset=X_offset,
            X_scale=X_scale,
            sample_weight_sqrt=sample_weight_sqrt if has_sw else None,
        )

    if solver == "svd":
        # 如果使用 SVD 求解，检查输入是否为稀疏矩阵
        if X_is_sparse:
            raise TypeError("SVD solver does not support sparse inputs currently")
        # 调用 _solve_svd 函数求解线性模型
        coef = _solve_svd(X, y, alpha, xp)

    if ravel:
        # 如果需要展平 coef
        coef = _ravel(coef)

    # 将 coef 转换为 xp 数组类型
    coef = xp.asarray(coef)

    # 根据返回参数组织返回结果
    if return_n_iter and return_intercept:
        res = coef, n_iter, intercept
    elif return_intercept:
        res = coef, intercept
    elif return_n_iter:
        res = coef, n_iter
    else:
        res = coef

    # 如果需要返回 solver 类型，则将其添加到返回结果中
    return (*res, solver) if return_solver else res
# 根据指定参数解析并返回合适的求解器名称
def resolve_solver(solver, positive, return_intercept, is_sparse, xp):
    # 如果 solver 参数不是 "auto"，直接返回该参数
    if solver != "auto":
        return solver

    # 检查是否处于 NumPy 命名空间
    is_numpy_namespace = _is_numpy_namespace(xp)

    # 为 NumPy 求解器解析自动求解器
    auto_solver_np = resolve_solver_for_numpy(positive, return_intercept, is_sparse)
    # 如果在 NumPy 命名空间中，返回 NumPy 求解器
    if is_numpy_namespace:
        return auto_solver_np

    # 如果 positive 参数为 True，抛出值错误
    if positive:
        raise ValueError(
            "The solvers that support positive fitting do not support "
            f"Array API dispatch to namespace {xp.__name__}. Please "
            "either disable Array API dispatch, or use a numpy-like "
            "namespace, or set `positive=False`."
        )

    # 当前情况下，Array API dispatch 仅支持 "svd" 求解器
    # 设置求解器为 "svd"
    solver = "svd"
    # 如果选择的求解器不是 NumPy 求解器，发出警告
    if solver != auto_solver_np:
        warnings.warn(
            f"Using Array API dispatch to namespace {xp.__name__} with "
            f"`solver='auto'` will result in using the solver '{solver}'. "
            "The results may differ from those when using a Numpy array, "
            f"because in that case the preferred solver would be {auto_solver_np}. "
            f"Set `solver='{solver}'` to suppress this warning."
        )

    # 返回最终确定的求解器
    return solver


# 根据 NumPy 参数解析返回适合的求解器名称
def resolve_solver_for_numpy(positive, return_intercept, is_sparse):
    # 如果 positive 参数为 True，返回 "lbfgs" 求解器
    if positive:
        return "lbfgs"

    # 如果 return_intercept 参数为 True，返回 "sag" 求解器
    if return_intercept:
        # sag 支持直接拟合截距
        return "sag"

    # 如果不是稀疏矩阵，返回 "cholesky" 求解器
    if not is_sparse:
        return "cholesky"

    # 对于稀疏矩阵，返回 "sparse_cg" 求解器
    return "sparse_cg"


# _BaseRidge 类的定义，包含参数约束和抽象方法
class _BaseRidge(LinearModel, metaclass=ABCMeta):
    # 参数约束字典定义
    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left"), np.ndarray],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left")],
        "solver": [
            StrOptions(
                {"auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"}
            )
        ],
        "positive": ["boolean"],
        "random_state": ["random_state"],
    }

    # 抽象方法 __init__ 定义
    @abstractmethod
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        # 初始化各参数
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state


# Ridge 类的定义，继承自 MultiOutputMixin, RegressorMixin, _BaseRidge
class Ridge(MultiOutputMixin, RegressorMixin, _BaseRidge):
    """Linear least squares with l2 regularization.

    Minimizes the objective function::

    ||y - Xw||^2_2 + alpha * ||w||^2_2

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    """
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape (n_samples, n_targets)).

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alpha : {float, ndarray of shape (n_targets,)}, default=1.0
        Constant that multiplies the L2 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

        When `alpha = 0`, the objective is equivalent to ordinary least
        squares, solved by the :class:`LinearRegression` object. For numerical
        reasons, using `alpha = 0` with the `Ridge` object is not advised.
        Instead, you should use the :class:`LinearRegression` object.

        If an array is passed, penalties are assumed to be specific to the
        targets. Hence they must correspond in number.


    fit_intercept : bool, default=True
        Whether to fit the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. ``X`` and ``y`` are expected to be centered).


    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.


    max_iter : int, default=None
        Maximum number of iterations for conjugate gradient solver.
        For 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
        For 'lbfgs' solver, the default value is 15000.


    tol : float, default=1e-4
        The precision of the solution (`coef_`) is determined by `tol` which
        specifies a different convergence criterion for each solver:

        - 'svd': `tol` has no impact.

        - 'cholesky': `tol` has no impact.

        - 'sparse_cg': norm of residuals smaller than `tol`.

        - 'lsqr': `tol` is set as atol and btol of scipy.sparse.linalg.lsqr,
          which control the norm of the residual vector in terms of the norms of
          matrix and coefficients.

        - 'sag' and 'saga': relative change of coef smaller than `tol`.

        - 'lbfgs': maximum of the absolute (projected) gradient=max|residuals|
          smaller than `tol`.

        .. versionchanged:: 1.2
           Default value changed from 1e-3 to 1e-4 for consistency with other linear
           models.
    # solver 参数用于指定岭回归中所使用的求解器，可以选择不同的算法来计算岭回归系数

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', \
            'sag', 'saga', 'lbfgs'}, default='auto'
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. It is the most stable solver, in particular more stable
          for singular matrices than 'cholesky' at the cost of being slower.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution.

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.

        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its improved, unbiased version named SAGA. Both methods also use an
          iterative procedure, and are often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.

        - 'lbfgs' uses L-BFGS-B algorithm implemented in
          `scipy.optimize.minimize`. It can be used only when `positive`
          is True.

        All solvers except 'svd' support both dense and sparse data. However, only
        'lsqr', 'sag', 'sparse_cg', and 'lbfgs' support sparse input when
        `fit_intercept` is True.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    # positive 参数用于强制岭回归系数为正数，只有在 solver 为 'lbfgs' 时支持此选项

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.
        Only 'lbfgs' solver is supported in this case.

    # random_state 参数用于在 solver 为 'sag' 或 'saga' 时用于数据的随机化处理

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag' or 'saga' to shuffle the data.
        See :term:`Glossary <random_state>` for details.

        .. versionadded:: 0.17
           `random_state` to support Stochastic Average Gradient.

    # coef_ 属性返回岭回归的权重向量或矩阵

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    # intercept_ 属性返回岭回归的截距，如果 fit_intercept 为 False，则设置为 0.0

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    # n_iter_ 属性返回每个目标的实际迭代次数，仅适用于 sag 和 lsqr 求解器，其他求解器返回 None

    n_iter_ : None or ndarray of shape (n_targets,)
        Actual number of iterations for each target. Available only for
        sag and lsqr solvers. Other solvers will return None.

        .. versionadded:: 0.17
    # 特征数目，记录在拟合过程中观察到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 特征名称列表，记录在拟合过程中观察到的特征名称数组
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 使用的求解器，记录在拟合时使用的求解器
    solver_ : str
        The solver that was used at fit time by the computational
        routines.

        .. versionadded:: 1.5

    # 相关内容
    See Also
    --------
    RidgeClassifier : Ridge classifier.
    RidgeCV : Ridge regression with built-in cross validation.
    :class:`~sklearn.kernel_ridge.KernelRidge` : Kernel ridge regression
        combines ridge regression with the kernel trick.

    # 注释
    Notes
    -----
    Regularization improves the conditioning of the problem and
    reduces the variance of the estimates. Larger values specify stronger
    regularization. Alpha corresponds to ``1 / (2C)`` in other linear
    models such as :class:`~sklearn.linear_model.LogisticRegression` or
    :class:`~sklearn.svm.LinearSVC`.

    # 示例
    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> clf = Ridge(alpha=1.0)
    >>> clf.fit(X, y)
    Ridge()

    # 初始化方法，设定各种参数以及调用父类初始化方法
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state,
        )

    # 装饰器，用于拟合过程的上下文管理
    @_fit_context(prefer_skip_nested_validation=True)
    # 定义一个方法用于拟合 Ridge 回归模型
    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            训练数据，可以是 ndarray 或稀疏矩阵，形状为 (样本数, 特征数)。

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            目标数值，形状为 (样本数,) 或 (样本数, 目标数)。

        sample_weight : float or ndarray of shape (n_samples,), default=None
            每个样本的个体权重。如果是 float，则每个样本具有相同的权重。

        Returns
        -------
        self : object
            拟合好的估计器对象。
        """
        # 确定是否接受稀疏矩阵作为输入，根据求解器类型获取有效的接受稀疏矩阵参数
        _accept_sparse = _get_valid_accept_sparse(sparse.issparse(X), self.solver)
        # 获取命名空间和对应的执行器
        xp, _ = get_namespace(X, y, sample_weight)
        # 验证并转换输入数据 X 和 y
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=_accept_sparse,
            dtype=[xp.float64, xp.float32],
            force_writeable=True,
            multi_output=True,
            y_numeric=True,
        )
        # 调用父类的 fit 方法进行模型拟合，传入样本权重参数
        return super().fit(X, y, sample_weight=sample_weight)

    # 返回一个包含额外标签的字典，表明数组 API 的支持情况
    def _more_tags(self):
        return {"array_api_support": True}
class _RidgeClassifierMixin(LinearClassifierMixin):
    def _prepare_data(self, X, y, sample_weight, solver):
        """Validate `X` and `y` and binarize `y`.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        solver : str
            The solver used in `Ridge` to know which sparse format to support.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Validated training data.

        y : ndarray of shape (n_samples,)
            Validated target values.

        sample_weight : ndarray of shape (n_samples,)
            Validated sample weights.

        Y : ndarray of shape (n_samples, n_classes)
            The binarized version of `y`.
        """
        # Determine if `X` is sparse and validate accordingly
        accept_sparse = _get_valid_accept_sparse(sparse.issparse(X), solver)
        # Validate `X` and `y` against specified criteria
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=accept_sparse,
            multi_output=True,
            y_numeric=False,
            force_writeable=True,
        )

        # Initialize a LabelBinarizer instance to binarize `y`
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        
        # Ensure `y` is shaped as a 1d array or column vector
        if not self._label_binarizer.y_type_.startswith("multilabel"):
            y = column_or_1d(y, warn=True)

        # Validate and adjust sample weights if provided
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        
        # Apply class weights to sample weights if specified
        if self.class_weight:
            sample_weight = sample_weight * compute_sample_weight(self.class_weight, y)
        
        # Return validated data and labels
        return X, y, sample_weight, Y

    def predict(self, X):
        """Predict class labels for samples in `X`.

        Parameters
        ----------
        X : {array-like, spare matrix} of shape (n_samples, n_features)
            The data matrix for which we want to predict the targets.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Vector or matrix containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`. In
            a multilabel problem, it returns a matrix of shape
            `(n_samples, n_outputs)`.
        """
        # Ensure the model has been fitted and `_label_binarizer` is available
        check_is_fitted(self, attributes=["_label_binarizer"])
        
        # Predictions for multilabel problems are derived from decision function
        if self._label_binarizer.y_type_.startswith("multilabel"):
            # Apply thresholding for binary predictions
            scores = 2 * (self.decision_function(X) > 0) - 1
            return self._label_binarizer.inverse_transform(scores)
        
        # For other problems, delegate prediction to superclass
        return super().predict(X)

    @property
    # 返回类的标签
    def classes_(self):
        """Classes labels."""
        return self._label_binarizer.classes_

    # 返回一个包含额外标签信息的字典，表明这是一个多标签分类问题
    def _more_tags(self):
        return {"multilabel": True}
class RidgeClassifier(_RidgeClassifierMixin, _BaseRidge):
    """Classifier using Ridge regression.

    This classifier first converts the target values into ``{-1, 1}`` and
    then treats the problem as a regression task (multi-output regression in
    the multiclass case).

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``1 / (2C)`` in other linear models such as
        :class:`~sklearn.linear_model.LogisticRegression` or
        :class:`~sklearn.svm.LinearSVC`.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations (e.g. data is expected to be
        already centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    max_iter : int, default=None
        Maximum number of iterations for conjugate gradient solver.
        The default value is determined by scipy.sparse.linalg.

    tol : float, default=1e-4
        The precision of the solution (`coef_`) is determined by `tol` which
        specifies a different convergence criterion for each solver:

        - 'svd': `tol` has no impact.

        - 'cholesky': `tol` has no impact.

        - 'sparse_cg': norm of residuals smaller than `tol`.

        - 'lsqr': `tol` is set as atol and btol of scipy.sparse.linalg.lsqr,
          which control the norm of the residual vector in terms of the norms of
          matrix and coefficients.

        - 'sag' and 'saga': relative change of coef smaller than `tol`.

        - 'lbfgs': maximum of the absolute (projected) gradient=max|residuals|
          smaller than `tol`.

        .. versionchanged:: 1.2
           Default value changed from 1e-3 to 1e-4 for consistency with other linear
           models.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
    """

    # 初始化函数，继承自 _RidgeClassifierMixin 和 _BaseRidge
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True,
                 max_iter=None, tol=1e-4, class_weight=None):
        # 调用父类 _BaseRidge 的初始化函数
        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         normalize=False, copy_X=copy_X,
                         max_iter=max_iter, tol=tol, solver='auto',
                         random_state=None)

        # 将 class_weight 参数设置为传入的值，或使用 'balanced' 模式
        self.class_weight = class_weight
    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', \
            'sag', 'saga', 'lbfgs'}, default='auto'
        # 使用的求解器，在计算过程中使用的求解方法：

        - 'auto' 根据数据类型自动选择求解器。

        - 'svd' 使用 X 的奇异值分解来计算岭回归系数。这是最稳定的求解器，尤其在处理奇异矩阵时比 'cholesky' 更稳定，但速度较慢。

        - 'cholesky' 使用标准的 scipy.linalg.solve 函数来获得闭式解。

        - 'sparse_cg' 使用 scipy.sparse.linalg.cg 中的共轭梯度求解器。作为迭代算法，对于大规模数据比 'cholesky' 更合适（可以设置 `tol` 和 `max_iter`）。

        - 'lsqr' 使用专用的正则化最小二乘例程 scipy.sparse.linalg.lsqr。它是最快的求解器，使用迭代过程。

        - 'sag' 使用随机平均梯度下降法，'saga' 使用其无偏且更灵活的版本 SAGA。这两种方法都使用迭代过程，在样本数和特征数较大时通常比其他求解器更快。注意，'sag' 和 'saga' 的快速收敛仅在特征大致相同的情况下才能保证。您可以使用 sklearn.preprocessing 中的缩放器预处理数据。

          .. versionadded:: 0.17
             随机平均梯度下降法求解器。
          .. versionadded:: 0.19
             SAGA 求解器。

        - 'lbfgs' 使用 `scipy.optimize.minimize` 中实现的 L-BFGS-B 算法。仅当 `positive` 为 True 时可用。

    positive : bool, default=False
        # 当设置为 ``True`` 时，强制系数为正数。在这种情况下仅支持 'lbfgs' 求解器。

    random_state : int, RandomState instance, default=None
        # 当 ``solver`` == 'sag' 或 'saga' 时用于对数据进行随机排列。详见 :term:`术语表 <random_state>`。

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        # 决策函数中特征的系数。

        当问题为二元问题时，``coef_`` 的形状为 (1, n_features)。

    intercept_ : float or ndarray of shape (n_targets,)
        # 决策函数中的独立项。如果 ``fit_intercept = False``，则设置为 0.0。

    n_iter_ : None or ndarray of shape (n_targets,)
        # 每个目标的实际迭代次数。仅对 sag 和 lsqr 求解器可用。其他求解器将返回 None。

    classes_ : ndarray of shape (n_classes,)
        # 类别标签。

    n_features_in_ : int
        # 在 :term:`拟合` 过程中看到的特征数。

        .. versionadded:: 0.24
    # feature_names_in_ 是一个形状为 (`n_features_in_`,) 的 ndarray
    # 表示在 `fit` 过程中观察到的特征名称，仅当 `X` 中的特征名都是字符串时才定义。

    # solver_ 是一个字符串
    # 表示在拟合时由计算程序使用的求解器。

    # 查看
    # --------
    # Ridge : 岭回归。
    # RidgeClassifierCV : 带有内置交叉验证的岭分类器。

    # 注意
    # -----
    # 对于多类分类问题，采用一对多方法训练 n_class 个分类器。
    # 具体来说，通过利用 Ridge 中的多变量响应支持来实现。

    # 示例
    # --------
    # >>> from sklearn.datasets import load_breast_cancer
    # >>> from sklearn.linear_model import RidgeClassifier
    # >>> X, y = load_breast_cancer(return_X_y=True)
    # >>> clf = RidgeClassifier().fit(X, y)
    # >>> clf.score(X, y)
    # 0.9595...

    _parameter_constraints: dict = {
        **_BaseRidge._parameter_constraints,
        # 继承基类 _BaseRidge 的参数约束
        "class_weight": [dict, StrOptions({"balanced"}), None],
    }

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        class_weight=None,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        # 调用父类的构造方法初始化参数
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state,
        )
        # 设置 class_weight 属性
        self.class_weight = class_weight

    @_fit_context(prefer_skip_nested_validation=True)
    # 使用装饰器 `_fit_context` 包装 fit 方法
    def fit(self, X, y, sample_weight=None):
        """Fit Ridge classifier model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

            .. versionadded:: 0.17
               *sample_weight* support to RidgeClassifier.

        Returns
        -------
        self : object
            Instance of the estimator.
        """
        # 准备数据，根据当前求解器进行处理
        X, y, sample_weight, Y = self._prepare_data(X, y, sample_weight, self.solver)

        # 调用父类的 fit 方法进行模型拟合
        super().fit(X, Y, sample_weight=sample_weight)
        return self
# 检查广义交叉验证模式的有效性，返回有效的模式名称
def _check_gcv_mode(X, gcv_mode):
    if gcv_mode in ["eigen", "svd"]:
        return gcv_mode
    # 如果 X 的行数大于列数，使用 X^T.X 的分解，否则使用 X.X^T 的分解
    if X.shape[0] > X.shape[1]:
        return "svd"
    return "eigen"


def _find_smallest_angle(query, vectors):
    """Find the column of vectors that is most aligned with the query.

    Both query and the columns of vectors must have their l2 norm equal to 1.

    Parameters
    ----------
    query : ndarray of shape (n_samples,)
        Normalized query vector.

    vectors : ndarray of shape (n_samples, n_features)
        Vectors to which we compare query, as columns. Must be normalized.
    """
    # 计算查询向量与向量集合中每列的绝对余弦值
    abs_cosine = np.abs(query.dot(vectors))
    # 返回具有最大绝对余弦值的列的索引
    index = np.argmax(abs_cosine)
    return index


class _X_CenterStackOp(sparse.linalg.LinearOperator):
    """Behaves as centered and scaled X with an added intercept column.

    This operator behaves as
    np.hstack([X - sqrt_sw[:, None] * X_mean, sqrt_sw[:, None]])
    """

    def __init__(self, X, X_mean, sqrt_sw):
        # 初始化方法，设置操作的属性
        n_samples, n_features = X.shape
        super().__init__(X.dtype, (n_samples, n_features + 1))
        self.X = X
        self.X_mean = X_mean
        self.sqrt_sw = sqrt_sw

    def _matvec(self, v):
        # 向量矩阵乘法，返回处理后的向量
        v = v.ravel()
        return (
            safe_sparse_dot(self.X, v[:-1], dense_output=True)
            - self.sqrt_sw * self.X_mean.dot(v[:-1])
            + v[-1] * self.sqrt_sw
        )

    def _matmat(self, v):
        # 矩阵乘法，返回处理后的矩阵
        return (
            safe_sparse_dot(self.X, v[:-1], dense_output=True)
            - self.sqrt_sw[:, None] * self.X_mean.dot(v[:-1])
            + v[-1] * self.sqrt_sw[:, None]
        )

    def _transpose(self):
        # 返回该线性操作的转置
        return _XT_CenterStackOp(self.X, self.X_mean, self.sqrt_sw)


class _XT_CenterStackOp(sparse.linalg.LinearOperator):
    """Behaves as transposed centered and scaled X with an intercept column.

    This operator behaves as
    np.hstack([X - sqrt_sw[:, None] * X_mean, sqrt_sw[:, None]]).T
    """

    def __init__(self, X, X_mean, sqrt_sw):
        # 初始化方法，设置操作的属性
        n_samples, n_features = X.shape
        super().__init__(X.dtype, (n_features + 1, n_samples))
        self.X = X
        self.X_mean = X_mean
        self.sqrt_sw = sqrt_sw

    def _matvec(self, v):
        # 向量矩阵乘法，返回处理后的向量
        v = v.ravel()
        n_features = self.shape[0]
        res = np.empty(n_features, dtype=self.X.dtype)
        res[:-1] = safe_sparse_dot(self.X.T, v, dense_output=True) - (
            self.X_mean * self.sqrt_sw.dot(v)
        )
        res[-1] = np.dot(v, self.sqrt_sw)
        return res

    def _matmat(self, v):
        # 矩阵乘法，返回处理后的矩阵
        n_features = self.shape[0]
        res = np.empty((n_features, v.shape[1]), dtype=self.X.dtype)
        res[:-1] = safe_sparse_dot(self.X.T, v, dense_output=True) - self.X_mean[
            :, None
        ] * self.sqrt_sw.dot(v)
        res[-1] = np.dot(self.sqrt_sw, v)
        return res


class _IdentityRegressor:
    """Fake regressor which will directly output the prediction."""

    # 定义一个虚拟的回归器类，其决策函数直接返回预测值
    def decision_function(self, y_predict):
        # 直接返回输入的预测值作为决策函数的结果
        return y_predict

    # 定义一个虚拟的回归器类，其预测函数直接返回预测值
    def predict(self, y_predict):
        # 直接返回输入的预测值作为预测函数的结果
        return y_predict
class _IdentityClassifier(LinearClassifierMixin):
    """Fake classifier which will directly output the prediction.

    We inherit from LinearClassifierMixin to get the proper shape for the
    output `y`.
    """

    def __init__(self, classes):
        self.classes_ = classes  # 初始化分类器的类别属性

    def decision_function(self, y_predict):
        return y_predict  # 直接返回预测结果


class _RidgeGCV(LinearModel):
    """Ridge regression with built-in Leave-one-out Cross-Validation.

    This class is not intended to be used directly. Use RidgeCV instead.

    Notes
    -----

    We want to solve (K + alpha*Id)c = y,
    where K = X X^T is the kernel matrix.

    Let G = (K + alpha*Id).

    Dual solution: c = G^-1y
    Primal solution: w = X^T c

    Compute eigendecomposition K = Q V Q^T.
    Then G^-1 = Q (V + alpha*Id)^-1 Q^T,
    where (V + alpha*Id) is diagonal.
    It is thus inexpensive to inverse for many alphas.

    Let loov be the vector of prediction values for each example
    when the model was fitted with all examples but this example.

    loov = (KG^-1Y - diag(KG^-1)Y) / diag(I-KG^-1)

    Let looe be the vector of prediction errors for each example
    when the model was fitted with all examples but this example.

    looe = y - loov = c / diag(G^-1)

    The best score (negative mean squared error or user-provided scoring) is
    stored in the `best_score_` attribute, and the selected hyperparameter in
    `alpha_`.

    References
    ----------
    http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf
    https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf
    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        copy_X=True,
        gcv_mode=None,
        store_cv_results=False,
        is_clf=False,
        alpha_per_target=False,
    ):
        self.alphas = alphas  # 设置 Ridge 模型的超参数 alpha 列表
        self.fit_intercept = fit_intercept  # 是否拟合截距项
        self.scoring = scoring  # 评分函数，用于模型评估
        self.copy_X = copy_X  # 是否复制输入数据 X
        self.gcv_mode = gcv_mode  # 交叉验证模式
        self.store_cv_results = store_cv_results  # 是否存储交叉验证结果
        self.is_clf = is_clf  # 是否为分类器
        self.alpha_per_target = alpha_per_target  # 是否为每个目标设置单独的 alpha

    @staticmethod
    def _decomp_diag(v_prime, Q):
        # 计算矩阵的对角线：dot(Q, dot(diag(v_prime), Q^T))
        return (v_prime * Q**2).sum(axis=-1)

    @staticmethod
    def _diag_dot(D, B):
        # 计算 dot(diag(D), B)
        if len(B.shape) > 1:
            # 处理 B 是多维的情况
            D = D[(slice(None),) + (np.newaxis,) * (len(B.shape) - 1)]
        return D * B
    def _compute_gram(self, X, sqrt_sw):
        """Computes the Gram matrix XX^T with possible centering.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The preprocessed design matrix.

        sqrt_sw : ndarray of shape (n_samples,)
            square roots of sample weights

        Returns
        -------
        gram : ndarray of shape (n_samples, n_samples)
            The Gram matrix.
        X_mean : ndarray of shape (n_features,)
            The weighted mean of ``X`` for each feature.

        Notes
        -----
        When X is dense the centering has been done in preprocessing
        so the mean is 0 and we just compute XX^T.

        When X is sparse it has not been centered in preprocessing, but it has
        been scaled by sqrt(sample weights).

        When self.fit_intercept is False no centering is done.

        The centered X is never actually computed because centering would break
        the sparsity of X.
        """
        # Determine if centering is necessary based on fit_intercept and sparsity of X
        center = self.fit_intercept and sparse.issparse(X)
        
        if not center:
            # If no centering is required:
            # X_mean is a zero array of shape (n_features,)
            X_mean = np.zeros(X.shape[1], dtype=X.dtype)
            # Return the Gram matrix XX^T and X_mean
            return safe_sparse_dot(X, X.T, dense_output=True), X_mean
        
        # If X is sparse and centering is required:
        n_samples = X.shape[0]
        # Create a sparse diagonal matrix of sample weights
        sample_weight_matrix = sparse.dia_matrix(
            (sqrt_sw, 0), shape=(n_samples, n_samples)
        )
        # Weight X by sample_weight_matrix
        X_weighted = sample_weight_matrix.dot(X)
        # Compute the mean of X_weighted along axis 0
        X_mean, _ = mean_variance_axis(X_weighted, axis=0)
        # Scale X_mean by a factor related to sample weights
        X_mean *= n_samples / sqrt_sw.dot(sqrt_sw)
        
        # Compute X_mX and X_mX_m terms for Gram matrix adjustment
        X_mX = sqrt_sw[:, None] * safe_sparse_dot(X_mean, X.T, dense_output=True)
        X_mX_m = np.outer(sqrt_sw, sqrt_sw) * np.dot(X_mean, X_mean)
        
        # Return the adjusted Gram matrix and X_mean
        return (
            safe_sparse_dot(X, X.T, dense_output=True) + X_mX_m - X_mX - X_mX.T,
            X_mean,
        )
    def _compute_covariance(self, X, sqrt_sw):
        """Computes covariance matrix X^TX with possible centering.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            The preprocessed design matrix.

        sqrt_sw : ndarray of shape (n_samples,)
            square roots of sample weights

        Returns
        -------
        covariance : ndarray of shape (n_features, n_features)
            The covariance matrix.
        X_mean : ndarray of shape (n_feature,)
            The weighted mean of ``X`` for each feature.

        Notes
        -----
        Since X is sparse it has not been centered in preprocessing, but it has
        been scaled by sqrt(sample weights).

        When self.fit_intercept is False no centering is done.

        The centered X is never actually computed because centering would break
        the sparsity of X.
        """
        if not self.fit_intercept:
            # 如果不拟合截距，说明在预处理中已经进行了中心化
            # 或者不需要拟合截距。
            X_mean = np.zeros(X.shape[1], dtype=X.dtype)
            # 直接返回 X^T * X 的乘积作为协方差矩阵，以及特征的加权平均值
            return safe_sparse_dot(X.T, X, dense_output=True), X_mean
        # 这个函数只用于稀疏矩阵 X
        n_samples = X.shape[0]
        # 创建一个对角稀疏矩阵，用于加权 X
        sample_weight_matrix = sparse.dia_matrix(
            (sqrt_sw, 0), shape=(n_samples, n_samples)
        )
        X_weighted = sample_weight_matrix.dot(X)
        # 计算加权后的 X 的平均值和方差
        X_mean, _ = mean_variance_axis(X_weighted, axis=0)
        # 根据样本权重调整平均值
        X_mean = X_mean * n_samples / sqrt_sw.dot(sqrt_sw)
        weight_sum = sqrt_sw.dot(sqrt_sw)
        # 返回 X^T * X 的乘积，减去加权平均值的外积，以及特征的加权平均值
        return (
            safe_sparse_dot(X.T, X, dense_output=True)
            - weight_sum * np.outer(X_mean, X_mean),
            X_mean,
        )
    def _sparse_multidot_diag(self, X, A, X_mean, sqrt_sw):
        """Compute the diagonal of (X - X_mean).dot(A).dot((X - X_mean).T)
        without explicitly centering X nor computing X.dot(A)
        when X is sparse.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            输入稀疏矩阵 X，形状为 (n_samples, n_features)

        A : ndarray of shape (n_features, n_features)
            矩阵 A，形状为 (n_features, n_features)

        X_mean : ndarray of shape (n_features,)
            特征均值 X_mean，形状为 (n_features,)

        sqrt_sw : ndarray of shape (n_features,)
            样本权重的平方根 sqrt_sw，形状为 (n_features,)

        Returns
        -------
        diag : np.ndarray, shape (n_samples,)
            计算得到的对角线数组 diag，形状为 (n_samples,)
        """
        intercept_col = scale = sqrt_sw
        batch_size = X.shape[1]  # 获取批处理大小，等于 X 的列数
        diag = np.empty(X.shape[0], dtype=X.dtype)  # 创建空的对角线数组 diag，长度为 X 的行数
        for start in range(0, X.shape[0], batch_size):  # 遍历 X 的行数，步长为 batch_size
            batch = slice(start, min(X.shape[0], start + batch_size), 1)  # 切片，取出当前批次的行索引
            X_batch = np.empty(
                (X[batch].shape[0], X.shape[1] + self.fit_intercept), dtype=X.dtype
            )  # 创建空的批处理数组 X_batch，形状为当前批次行数乘以（X 列数加上拦截项）
            if self.fit_intercept:  # 如果启用拦截项
                X_batch[:, :-1] = X[batch].toarray() - X_mean * scale[batch][:, None]  # 对 X_batch 赋值，去除最后一列的拦截项
                X_batch[:, -1] = intercept_col[batch]  # 设置 X_batch 最后一列为拦截项列
            else:  # 如果未启用拦截项
                X_batch = X[batch].toarray()  # 直接将 X[batch] 转换为稠密数组赋给 X_batch
            diag[batch] = (X_batch.dot(A) * X_batch).sum(axis=1)  # 计算对角线的值，将结果存入 diag 的对应批次中
        return diag  # 返回计算得到的对角线数组 diag

    def _eigen_decompose_gram(self, X, y, sqrt_sw):
        """Eigendecomposition of X.X^T, used when n_samples <= n_features."""
        # if X is dense it has already been centered in preprocessing
        K, X_mean = self._compute_gram(X, sqrt_sw)  # 计算 Gram 矩阵 K 和特征均值 X_mean
        if self.fit_intercept:
            # to emulate centering X with sample weights,
            # ie removing the weighted average, we add a column
            # containing the square roots of the sample weights.
            # by centering, it is orthogonal to the other columns
            K += np.outer(sqrt_sw, sqrt_sw)  # 如果启用拦截项，调整 Gram 矩阵 K，使其包含样本权重的平方根
        eigvals, Q = linalg.eigh(K)  # 对 Gram 矩阵 K 进行特征值分解，得到特征值 eigvals 和特征向量 Q
        QT_y = np.dot(Q.T, y)  # 计算 Q 的转置与向量 y 的乘积
        return X_mean, eigvals, Q, QT_y  # 返回特征均值 X_mean, 特征值 eigvals, 特征向量 Q, Q 的转置与向量 y 的乘积 QT_y
    def _solve_eigen_gram(self, alpha, y, sqrt_sw, X_mean, eigvals, Q, QT_y):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X.X^T (n_samples <= n_features).
        """
        # 计算权重向量 w，用于对 G 的对角线元素进行求逆
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            # 如果需要拟合截距，取消截距维度上的正则化
            # 样本权重的平方根是 XX^T 的特征向量，对应于截距；
            # 我们在这个维度上取消正则化。对应的特征值是样本权重之和。
            normalized_sw = sqrt_sw / np.linalg.norm(sqrt_sw)
            intercept_dim = _find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0  # 取消截距的正则化

        # 计算最终的系数向量 c
        c = np.dot(Q, self._diag_dot(w, QT_y))
        # 计算 G^-1 的对角线部分
        G_inverse_diag = self._decomp_diag(w, Q)
        # 处理 y 是二维数组的情况
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, np.newaxis]
        return G_inverse_diag, c

    def _eigen_decompose_covariance(self, X, y, sqrt_sw):
        """Eigendecomposition of X^T.X, used when n_samples > n_features
        and X is sparse.
        """
        # 获取样本数量和特征数量
        n_samples, n_features = X.shape
        # 初始化协方差矩阵
        cov = np.empty((n_features + 1, n_features + 1), dtype=X.dtype)
        # 计算样本协方差矩阵和特征均值
        cov[:-1, :-1], X_mean = self._compute_covariance(X, sqrt_sw)
        if not self.fit_intercept:
            # 如果不需要拟合截距，去除最后一行和最后一列
            cov = cov[:-1, :-1]
        else:
            # 如果需要拟合截距，添加一列和一行作为样本权重的平方根
            cov[-1] = 0
            cov[:, -1] = 0
            cov[-1, -1] = sqrt_sw.dot(sqrt_sw)  # 最后一个元素为样本权重的平方和
        # 计算 X^T.X 的零空间维度
        nullspace_dim = max(0, n_features - n_samples)
        # 对协方差矩阵进行特征值分解
        eigvals, V = linalg.eigh(cov)
        # 去除 X^T.X 的零空间中的特征值和特征向量
        eigvals = eigvals[nullspace_dim:]
        V = V[:, nullspace_dim:]
        return X_mean, eigvals, V, X

    def _solve_eigen_covariance_no_intercept(
        self, alpha, y, sqrt_sw, X_mean, eigvals, V, X
    ):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X^T.X
        (n_samples > n_features and X is sparse), and not fitting an intercept.
        """
        # 计算权重向量 w，用于对 G 的对角线元素进行求逆
        w = 1 / (eigvals + alpha)
        # 计算矩阵 A，A = V * diag(w) * V^T
        A = (V * w).dot(V.T)
        # 计算 AXy = A * X^T * y
        AXy = A.dot(safe_sparse_dot(X.T, y, dense_output=True))
        # 计算预测值 y_hat = X * AXy
        y_hat = safe_sparse_dot(X, AXy, dense_output=True)
        # 计算 hat_diag，表示 G^-1 的对角线部分
        hat_diag = self._sparse_multidot_diag(X, A, X_mean, sqrt_sw)
        # 处理 y 是二维数组的情况
        if len(y.shape) != 1:
            hat_diag = hat_diag[:, np.newaxis]
        # 返回 G^-1 的对角线部分和系数向量
        return (1 - hat_diag) / alpha, (y - y_hat) / alpha
    def _solve_eigen_covariance_intercept(
        self, alpha, y, sqrt_sw, X_mean, eigvals, V, X
    ):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X^T.X
        (n_samples > n_features and X is sparse),
        and we are fitting an intercept.
        """
        # the vector [0, 0, ..., 0, 1]
        # is the eigenvector of X^TX which
        # corresponds to the intercept; we cancel the regularization on
        # this dimension. the corresponding eigenvalue is
        # sum(sample_weight), e.g. n when uniform sample weights.
        
        # 创建一个长度为 V.shape[0] 的零向量，最后一个元素设为 1
        intercept_sv = np.zeros(V.shape[0])
        intercept_sv[-1] = 1
        
        # 找到与拦截向量 intercept_sv 最小角度对应的维度
        intercept_dim = _find_smallest_angle(intercept_sv, V)
        
        # 计算权重向量 w，其中拦截维度的权重是 1 / eigvals[intercept_dim]，其他维度的权重是 1 / (eigvals + alpha)
        w = 1 / (eigvals + alpha)
        w[intercept_dim] = 1 / eigvals[intercept_dim]
        
        # 构建对角矩阵 A = V * diag(w) * V^T
        A = (V * w).dot(V.T)
        
        # 将样本权重的平方根添加为 X 的新列
        X_op = _X_CenterStackOp(X, X_mean, sqrt_sw)
        
        # 计算 AXy = A * X_op.T * y
        AXy = A.dot(X_op.T.dot(y))
        
        # 计算预测值 y_hat = X_op * AXy
        y_hat = X_op.dot(AXy)
        
        # 计算 hat_diag = X * A * X_mean * sqrt_sw 的对角线
        hat_diag = self._sparse_multidot_diag(X, A, X_mean, sqrt_sw)
        
        # 返回结果 (1 - hat_diag) / alpha, (y - y_hat) / alpha
        if len(y.shape) != 1:
            # 处理 y 是二维的情况
            hat_diag = hat_diag[:, np.newaxis]
        return (1 - hat_diag) / alpha, (y - y_hat) / alpha

    def _solve_eigen_covariance(self, alpha, y, sqrt_sw, X_mean, eigvals, V, X):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X^T.X
        (n_samples > n_features and X is sparse).
        """
        # 如果拟合过程中包括截距项，调用 _solve_eigen_covariance_intercept 方法
        if self.fit_intercept:
            return self._solve_eigen_covariance_intercept(
                alpha, y, sqrt_sw, X_mean, eigvals, V, X
            )
        # 否则调用 _solve_eigen_covariance_no_intercept 方法
        return self._solve_eigen_covariance_no_intercept(
            alpha, y, sqrt_sw, X_mean, eigvals, V, X
        )

    def _svd_decompose_design_matrix(self, X, y, sqrt_sw):
        # X 已经中心化
        X_mean = np.zeros(X.shape[1], dtype=X.dtype)
        
        # 如果拟合过程中包括截距项
        if self.fit_intercept:
            # 模拟 fit_intercept=True 的情况，添加一个包含样本权重平方根的列
            intercept_column = sqrt_sw[:, None]
            X = np.hstack((X, intercept_column))
        
        # 对 X 进行奇异值分解，得到 U, singvals_sq 和 _
        U, singvals, _ = linalg.svd(X, full_matrices=0)
        
        # 计算 singvals 的平方
        singvals_sq = singvals**2
        
        # 计算 UT_y = U^T * y
        UT_y = np.dot(U.T, y)
        
        # 返回结果 X_mean, singvals_sq, U, UT_y
        return X_mean, singvals_sq, U, UT_y
    def _solve_svd_design_matrix(self, alpha, y, sqrt_sw, X_mean, singvals_sq, U, UT_y):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have an SVD decomposition of X
        (n_samples > n_features and X is dense).
        """
        # 计算双重系数和G^-1的对角线

        # 计算权重向量w
        w = ((singvals_sq + alpha) ** -1) - (alpha**-1)

        if self.fit_intercept:
            # 检测拦截列
            normalized_sw = sqrt_sw / np.linalg.norm(sqrt_sw)
            intercept_dim = _find_smallest_angle(normalized_sw, U)
            # 取消拦截的正则化
            w[intercept_dim] = -(alpha**-1)

        # 计算系数向量c
        c = np.dot(U, self._diag_dot(w, UT_y)) + (alpha**-1) * y

        # 计算G^-1的对角线
        G_inverse_diag = self._decomp_diag(w, U) + (alpha**-1)

        if len(y.shape) != 1:
            # 处理y为二维数组的情况
            G_inverse_diag = G_inverse_diag[:, np.newaxis]

        return G_inverse_diag, c

    def _get_scorer(self):
        """返回评分器对象，用于评分"""
        return check_scoring(self, scoring=self.scoring, allow_none=True)

    def _score_without_scorer(self, squared_errors):
        """当评分器为None时，使用平方误差进行评分"""
        if self.alpha_per_target:
            _score = -squared_errors.mean(axis=0)
        else:
            _score = -squared_errors.mean()

        return _score

    def _score(self, *, predictions, y, n_y, scorer, score_params):
        """使用指定的评分器对预测值和真实值进行评分"""
        if self.is_clf:
            # 创建一个身份分类器对象
            identity_estimator = _IdentityClassifier(classes=np.arange(n_y))
            _score = scorer(
                identity_estimator,
                predictions,
                y.argmax(axis=1),
                **score_params,
            )
        else:
            # 创建一个身份回归器对象
            identity_estimator = _IdentityRegressor()
            if self.alpha_per_target:
                # 对每个目标值分别进行评分
                _score = np.array(
                    [
                        scorer(
                            identity_estimator,
                            predictions[:, j],
                            y[:, j],
                            **score_params,
                        )
                        for j in range(n_y)
                    ]
                )
            else:
                # 对展平后的预测值和真实值进行评分
                _score = scorer(
                    identity_estimator,
                    predictions.ravel(),
                    y.ravel(),
                    **score_params,
                )

        return _score
# 定义一个名为 _BaseRidgeCV 的类，它继承自 LinearModel 类
class _BaseRidgeCV(LinearModel):
    # 定义一个类属性 _parameter_constraints，它是一个字典，描述了参数的约束条件
    _parameter_constraints: dict = {
        "alphas": ["array-like", Interval(Real, 0, None, closed="neither")],
        "fit_intercept": ["boolean"],
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],
        "cv": ["cv_object"],
        "gcv_mode": [StrOptions({"auto", "svd", "eigen"}), None],
        "store_cv_results": ["boolean", Hidden(None)],
        "alpha_per_target": ["boolean"],
        "store_cv_values": ["boolean", Hidden(StrOptions({"deprecated"}))],
    }

    # 初始化方法，设置类的各种属性
    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        cv=None,
        gcv_mode=None,
        store_cv_results=None,
        alpha_per_target=False,
        store_cv_values="deprecated",
    ):
        self.alphas = alphas  # 设置 alphas 属性
        self.fit_intercept = fit_intercept  # 设置 fit_intercept 属性
        self.scoring = scoring  # 设置 scoring 属性
        self.cv = cv  # 设置 cv 属性
        self.gcv_mode = gcv_mode  # 设置 gcv_mode 属性
        self.store_cv_results = store_cv_results  # 设置 store_cv_results 属性
        self.alpha_per_target = alpha_per_target  # 设置 alpha_per_target 属性
        self.store_cv_values = store_cv_values  # 设置 store_cv_values 属性

    # 获取对象的元数据路由信息
    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，并配置其路由信息
        router = (
            MetadataRouter(owner=self.__class__.__name__)  # 设置路由器的所有者信息
            .add_self_request(self)  # 添加对自身请求的路由
            .add(
                scorer=self._get_scorer(),  # 添加评分器的路由
                method_mapping=MethodMapping().add(callee="score", caller="fit"),  # 配置方法映射
            )
        )
        return router  # 返回配置好的 MetadataRouter 对象

    # 获取评分器的方法
    def _get_scorer(self):
        return check_scoring(self, scoring=self.scoring, allow_none=True)

    # 属性访问器方法，用于获取 cv_values_ 属性的值，标记为即将移除的属性
    # TODO(1.7): Remove
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `cv_values_` is deprecated in version 1.5 and will be removed "
        "in 1.7. Use `cv_results_` instead."
    )
    @property
    def cv_values_(self):
        return self.cv_results_


# RidgeCV 类，继承自 MultiOutputMixin 和 RegressorMixin，以及 _BaseRidgeCV 类
class RidgeCV(MultiOutputMixin, RegressorMixin, _BaseRidgeCV):
    """Ridge regression with built-in cross-validation.

    See glossary entry for :term:`cross-validation estimator`.

    By default, it performs efficient Leave-One-Out Cross-Validation.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alphas : array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        # 用于尝试的 alpha 值数组。
        # 正则化强度，必须是正浮点数。正则化可以改善问题的条件，并减少估计的方差。较大的值表示更强的正则化。
        # 在其他线性模型中，例如 LogisticRegression 或 LinearSVC，alpha 对应于 ``1 / (2C)``
        # 如果使用 Leave-One-Out 交叉验证，alphas 必须严格为正数。

    fit_intercept : bool, default=True
        # 是否计算模型的截距。如果设置为 False，则计算中不使用截距（即预期数据已居中）。

    scoring : str, callable, default=None
        # 评分方法，可以是字符串（参见 scoring_parameter）或者具有 ``scorer(estimator, X, y)`` 签名的可调用对象/函数。
        # 如果为 None，在 cv 是 'auto' 或 None 时（即使用 Leave-One-Out 交叉验证时），使用负均方误差；否则使用 R2 分数。

    cv : int, cross-validation generator or an iterable, default=None
        # 确定交叉验证的拆分策略。
        # cv 的可能输入包括：
        # - None，使用高效的 Leave-One-Out 交叉验证
        # - 整数，指定折数。
        # - CV splitter，
        # - 一个可迭代对象，产生（训练集，测试集）索引数组。

        # 对于整数/None 输入，如果 y 是二元或多类别的，则使用 StratifiedKFold；否则使用 KFold。

        # 请参阅用户指南了解可以在此处使用的各种交叉验证策略。

    gcv_mode : {'auto', 'svd', 'eigen'}, default='auto'
        # 指示在执行 Leave-One-Out 交叉验证时使用的策略。
        # 可选项包括：
        # 'auto'：如果 n_samples > n_features，则使用 'svd'，否则使用 'eigen'
        # 'svd'：当 X 是密集时，强制使用 X 的奇异值分解；当 X 是稀疏时，使用 X^T.X 的特征值分解。
        # 'eigen'：强制通过 X.X^T 的特征分解进行计算。

        # 'auto' 模式是默认值，旨在根据训练数据的形状选择更便宜的选项。

    store_cv_results : bool, default=False
        # 标志，指示是否应将每个 alpha 对应的交叉验证值存储在 ``cv_values_`` 属性中。
        # 此标志仅与 ``cv=None`` 兼容（即使用 Leave-One-Out 交叉验证）。

        # .. versionchanged:: 1.5
        #    参数名称从 `store_cv_values` 更改为 `store_cv_results`。
    alpha_per_target : bool, default=False
        # 是否针对每个目标单独优化 alpha 值（从 alphas 参数列表中选取）。对于多输出设置（多个预测目标），设置为 True 后，
        # 在拟合后，alpha_ 属性将包含每个目标的一个值。设置为 False 时，所有目标共享一个 alpha 值。

        .. versionadded:: 0.24
            # 添加于版本 0.24

    store_cv_values : bool
        # 指示是否将每个 alpha 对应的交叉验证值存储在 cv_values_ 属性中的标志（参见下文）。此标志仅与 ``cv=None`` 兼容
        # （即使用留一交叉验证）。

        .. deprecated:: 1.5
            # 在 1.5 版本中弃用了 `store_cv_values`，建议使用 `store_cv_results`，并将在 1.7 版本中移除。

    Attributes
    ----------
    cv_results_ : ndarray of shape (n_samples, n_alphas) or \
            shape (n_samples, n_targets, n_alphas), optional
        # 每个 alpha 的交叉验证值（仅当 ``store_cv_results=True`` 且 ``cv=None`` 时可用）。调用 ``fit()`` 后，
        # 此属性将包含均方误差（如果 `scoring` 为 None）；否则，将包含标准化的每个点的预测值。

        .. versionchanged:: 1.5
            # `cv_values_` 更名为 `cv_results_`。

    coef_ : ndarray of shape (n_features) or (n_targets, n_features)
        # 权重向量。

    intercept_ : float or ndarray of shape (n_targets,)
        # 决策函数中的独立项。如果 ``fit_intercept = False``，则设置为 0.0。

    alpha_ : float or ndarray of shape (n_targets,)
        # 估计的正则化参数，或者如果 ``alpha_per_target=True``，则为每个目标的估计正则化参数。

    best_score_ : float or ndarray of shape (n_targets,)
        # 使用最佳 alpha 值的基础估计器的得分，或者如果 ``alpha_per_target=True``，则为每个目标的得分。

        .. versionadded:: 0.23
            # 添加于版本 0.23

    n_features_in_ : int
        # 在拟合期间观察到的特征数。

        .. versionadded:: 0.24
            # 添加于版本 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `X` 具有所有字符串特征名称时，在拟合期间观察到的特征名称。

        .. versionadded:: 1.0
            # 添加于版本 1.0

    See Also
    --------
    Ridge : 岭回归。
    RidgeClassifier : 基于岭回归的分类器，针对 {-1, 1} 标签。
    RidgeClassifierCV : 带内置交叉验证的岭回归分类器。

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import RidgeCV
    >>> X, y = load_diabetes(return_X_y=True)
    >>> clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
    >>> clf.score(X, y)
    0.5166...
    ```

    @_fit_context(prefer_skip_nested_validation=True)
        # 使用 `_fit_context` 装饰器，设置 `prefer_skip_nested_validation=True`。
    def fit(self, X, y, sample_weight=None, **params):
        """
        Fit Ridge regression model with cv.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data. If using GCV, will be cast to float64
            if necessary.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        **params : dict, default=None
            Parameters to be passed to the underlying scorer.

            .. versionadded:: 1.5
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        When sample_weight is provided, the selected hyperparameter may depend
        on whether we use leave-one-out cross-validation (cv=None or cv='auto')
        or another form of cross-validation, because only leave-one-out
        cross-validation takes the sample weights into account when computing
        the validation score.
        """
        # 调用父类的fit方法，传入训练数据X，目标值y，以及可能的样本权重sample_weight和额外的参数params
        super().fit(X, y, sample_weight=sample_weight, **params)
        # 返回已经拟合好的估计器对象本身
        return self
class RidgeClassifierCV(_RidgeClassifierMixin, _BaseRidgeCV):
    """Ridge classifier with built-in cross-validation.

    See glossary entry for :term:`cross-validation estimator`.

    By default, it performs Leave-One-Out Cross-Validation. Currently,
    only the n_features > n_samples case is handled efficiently.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alphas : array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        Array of alpha values to try.
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``1 / (2C)`` in other linear models such as
        :class:`~sklearn.linear_model.LogisticRegression` or
        :class:`~sklearn.svm.LinearSVC`.
        If using Leave-One-Out cross-validation, alphas must be strictly positive.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    scoring : str, callable, default=None
        A string (see :ref:`scoring_parameter`) or a scorer callable object /
        function with signature ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    store_cv_results : bool, default=False
        Flag indicating if the cross-validation results corresponding to
        each alpha should be stored in the ``cv_results_`` attribute (see
        below). This flag is only compatible with ``cv=None`` (i.e. using
        Leave-One-Out Cross-Validation).

        .. versionchanged:: 1.5
            Parameter name changed from `store_cv_values` to `store_cv_results`.
    """
    # 定义一个布尔值，指示是否应将每个 alpha 对应的交叉验证值存储在 `cv_values_` 属性中
    # 仅当 `cv=None`（即使用留一交叉验证）时，此标志才有效。
    store_cv_values : bool
        Flag indicating if the cross-validation values corresponding to
        each alpha should be stored in the ``cv_values_`` attribute (see
        below). This flag is only compatible with ``cv=None`` (i.e. using
        Leave-One-Out Cross-Validation).

        # 在版本 1.5 中已弃用 `store_cv_values`，推荐使用 `store_cv_results`，并将在版本 1.7 中移除。
        .. deprecated:: 1.5
            `store_cv_values` is deprecated in version 1.5 in favor of
            `store_cv_results` and will be removed in version 1.7.

    Attributes
    ----------
    # 如果 `store_cv_results=True` 并且 `cv=None`，则为每个 alpha 的交叉验证结果
    cv_results_ : ndarray of shape (n_samples, n_targets, n_alphas), optional
        Cross-validation results for each alpha (only if ``store_cv_results=True`` and
        ``cv=None``). After ``fit()`` has been called, this attribute will
        contain the mean squared errors if `scoring is None` otherwise it
        will contain standardized per point prediction values.

        # 在版本 1.5 中，`cv_values_` 更名为 `cv_results_`。
        .. versionchanged:: 1.5
            `cv_values_` changed to `cv_results_`.

    # 决策函数中特征的系数数组
    coef_ : ndarray of shape (1, n_features) or (n_targets, n_features)
        Coefficient of the features in the decision function.

        # 在二分类问题中，`coef_` 的形状为 (1, n_features)。

    # 决策函数中的独立项
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    # 估计的正则化参数
    alpha_ : float
        Estimated regularization parameter.

    # 最佳 alpha 对应的基础评估器得分
    best_score_ : float
        Score of base estimator with best alpha.

        # 在版本 0.23 中添加
        .. versionadded:: 0.23

    # 类别标签数组
    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    # 在拟合过程中看到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        # 在版本 0.24 中添加
        .. versionadded:: 0.24

    # 在拟合过程中看到的特征名称数组
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        # 在版本 1.0 中添加
        .. versionadded:: 1.0

    See Also
    --------
    Ridge : Ridge regression.
    RidgeClassifier : Ridge classifier.
    RidgeCV : Ridge regression with built-in cross validation.

    Notes
    -----
    # 对于多类分类，采用一对多方法训练 n_class 分类器
    # 具体来说，利用 Ridge 中的多变量响应支持来实现。
    For multi-class classification, n_class classifiers are trained in
    a one-versus-all approach. Concretely, this is implemented by taking
    advantage of the multi-variate response support in Ridge.

    Examples
    --------
    # 示例：从 sklearn.datasets 导入乳腺癌数据集，使用 RidgeClassifierCV 进行分类器拟合和评分
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import RidgeClassifierCV
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
    >>> clf.score(X, y)
    0.9630...

    """

    # 参数约束字典，继承自 _BaseRidgeCV._parameter_constraints
    _parameter_constraints: dict = {
        **_BaseRidgeCV._parameter_constraints,
        "class_weight": [dict, StrOptions({"balanced"}), None],
    }

    # 移除不再支持的参数约束
    for param in ("gcv_mode", "alpha_per_target"):
        _parameter_constraints.pop(param)
    # 初始化方法，用于设置 RidgeClassifierCV 的参数和属性
    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),  # 默认的 alpha 参数值，用于正则化
        *,
        fit_intercept=True,  # 是否拟合截距，默认为 True
        scoring=None,  # 评分标准，默认为 None
        cv=None,  # 交叉验证的折数，默认为 None
        class_weight=None,  # 类别权重，默认为 None
        store_cv_results=None,  # 是否存储交叉验证结果，默认为 None
        store_cv_values="deprecated",  # 是否存储交叉验证的值，默认为 "deprecated"
    ):
        # 调用父类 RidgeClassifierCV 的初始化方法，传入参数
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring,
            cv=cv,
            store_cv_results=store_cv_results,
            store_cv_values=store_cv_values,
        )
        # 设置当前类的 class_weight 属性
        self.class_weight = class_weight

    # 使用装饰器 _fit_context 包装的方法，用于拟合 Ridge 分类器与交叉验证
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, **params):
        """Fit Ridge classifier with cv.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features. When using GCV,
            will be cast to float64 if necessary.

        y : ndarray of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        **params : dict, default=None
            Parameters to be passed to the underlying scorer.

            .. versionadded:: 1.5
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 准备数据，包括类型转换和处理稀疏矩阵，使用 "eigen" solver 强制接受所有稀疏格式
        X, y, sample_weight, Y = self._prepare_data(X, y, sample_weight, solver="eigen")

        # 如果 cv 为 None，表示使用 gcv 模式，并且使用二值化后的 Y
        # 因为在 _RidgeGCV 估计器中，y 不会被二值化
        # 如果 cv 不为 None，则使用包含多个 RidgeClassifier 估计器的 GridSearchCV，其中 y 将被二值化
        # 因此我们传递 y 而不是二值化后的 Y
        target = Y if self.cv is None else y
        
        # 调用父类的 fit 方法来拟合模型
        super().fit(X, target, sample_weight=sample_weight, **params)
        
        # 返回拟合后的自身对象
        return self

    # 返回额外的标签信息，指定多标签为 True，并且设置特定的失败检查
    def _more_tags(self):
        return {
            "multilabel": True,
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            },
        }
```