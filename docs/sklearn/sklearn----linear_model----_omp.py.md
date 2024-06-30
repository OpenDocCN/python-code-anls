# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_omp.py`

```
"""Orthogonal matching pursuit algorithms"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings  # 导入警告模块，用于处理警告信息
from math import sqrt  # 导入平方根函数
from numbers import Integral, Real  # 导入整数和实数类型的判断函数

import numpy as np  # 导入NumPy库
from scipy import linalg  # 导入SciPy线性代数模块
from scipy.linalg.lapack import get_lapack_funcs  # 导入获取LAPACK函数接口的方法

from ..base import MultiOutputMixin, RegressorMixin, _fit_context  # 导入基类和Mixin类
from ..model_selection import check_cv  # 导入交叉验证函数
from ..utils import Bunch, as_float_array, check_array  # 导入工具函数和数据结构
from ..utils._param_validation import Interval, StrOptions, validate_params  # 导入参数验证相关函数
from ..utils.metadata_routing import (  # 导入元数据路由相关模块和函数
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from ..utils.parallel import Parallel, delayed  # 导入并行处理相关函数
from ._base import LinearModel, _pre_fit  # 导入线性模型基类和预训练方法

premature = (
    "Orthogonal matching pursuit ended prematurely due to linear"
    " dependence in the dictionary. The requested precision might"
    " not have been met."
)  # 提前终止的错误消息

def _cholesky_omp(X, y, n_nonzero_coefs, tol=None, copy_X=True, return_path=False):
    """Orthogonal Matching Pursuit step using the Cholesky decomposition.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input dictionary. Columns are assumed to have unit norm.

    y : ndarray of shape (n_samples,)
        Input targets.

    n_nonzero_coefs : int
        Targeted number of non-zero elements.

    tol : float, default=None
        Targeted squared error, if not None overrides n_nonzero_coefs.

    copy_X : bool, default=True
        Whether the design matrix X must be copied by the algorithm. A false
        value is only helpful if X is already Fortran-ordered, otherwise a
        copy is made anyway.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    Returns
    -------
    gamma : ndarray of shape (n_nonzero_coefs,)
        Non-zero elements of the solution.

    idx : ndarray of shape (n_nonzero_coefs,)
        Indices of the positions of the elements in gamma within the solution
        vector.

    coef : ndarray of shape (n_features, n_nonzero_coefs)
        The first k values of column k correspond to the coefficient value
        for the active features at that step. The lower left triangle contains
        garbage. Only returned if ``return_path=True``.

    n_active : int
        Number of active features at convergence.
    """
    if copy_X:
        X = X.copy("F")  # 如果需要复制X，则按Fortran顺序复制
    else:  # 即使允许覆盖，如果顺序不好，仍然会复制
        X = np.asfortranarray(X)

    min_float = np.finfo(X.dtype).eps  # 获取X的数据类型的最小浮点数值
    nrm2, swap = linalg.get_blas_funcs(("nrm2", "swap"), (X,))  # 获取BLAS库中的nrm2和swap函数
    (potrs,) = get_lapack_funcs(("potrs",), (X,))  # 获取LAPACK库中的potrs函数

    alpha = np.dot(X.T, y)  # 计算X的转置与y的点积，得到alpha向量
    residual = y  # 初始化残差为y向量
    gamma = np.empty(0)  # 初始化gamma为空数组
    n_active = 0  # 初始化活跃特征数为0
    indices = np.arange(X.shape[1])  # 初始化索引数组，用于记录交换位置

    max_features = X.shape[1] if tol is not None else n_nonzero_coefs  # 如果tol不为None，则使用X的特征数，否则使用n_nonzero_coefs
    # 创建一个空的 max_features x max_features 的数组 L，其数据类型与输入数据 X 的数据类型相同
    L = np.empty((max_features, max_features), dtype=X.dtype)

    # 如果需要返回路径信息，则创建一个与 L 大小相同的空数组 coefs
    if return_path:
        coefs = np.empty_like(L)

    # 进入主循环，直到满足终止条件才退出
    while True:
        # 选择当前残差与 X 的转置矩阵的内积绝对值最大的索引 lam
        lam = np.argmax(np.abs(np.dot(X.T, residual)))

        # 如果选择的索引 lam 小于当前活跃变量数目 n_active，或者 alpha[lam] 的平方小于 min_float，则发出运行时警告并退出循环
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # 原子已经被选择或内积过小
            warnings.warn(premature, RuntimeWarning, stacklevel=2)
            break

        # 如果当前活跃变量数目大于 0，则更新 X'X 的 Cholesky 分解
        if n_active > 0:
            # 更新 L 的部分行列，用于 Cholesky 分解
            L[n_active, :n_active] = np.dot(X[:, :n_active].T, X[:, lam])
            # 解三角系统 L[:n_active, :n_active] * L[n_active, :n_active] = 0
            linalg.solve_triangular(
                L[:n_active, :n_active],
                L[n_active, :n_active],
                trans=0,
                lower=1,
                overwrite_b=True,
                check_finite=False,
            )
            # 计算 L[n_active, n_active] 的值
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = linalg.norm(X[:, lam]) ** 2 - v
            # 如果 Lkk 小于等于 min_float，则表示选择的原子是相关的，发出运行时警告并退出循环
            if Lkk <= min_float:
                warnings.warn(premature, RuntimeWarning, stacklevel=2)
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            # 如果当前活跃变量数目为 0，则直接将 L[0, 0] 设置为 X[:, lam] 的二范数
            L[0, 0] = linalg.norm(X[:, lam])

        # 交换 X.T[n_active] 和 X.T[lam]
        X.T[n_active], X.T[lam] = swap(X.T[n_active], X.T[lam])
        # 交换 alpha[n_active] 和 alpha[lam]
        alpha[n_active], alpha[lam] = alpha[lam], alpha[n_active]
        # 交换 indices[n_active] 和 indices[lam]
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        # 增加当前活跃变量数目
        n_active += 1

        # 解 LL'x = X'y 作为两个三角系统的组合解
        gamma, _ = potrs(
            L[:n_active, :n_active], alpha[:n_active], lower=True, overwrite_b=False
        )

        # 如果需要返回路径信息，则将当前 gamma 存储在 coefs 中
        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        # 更新残差为 y - X[:, :n_active] * gamma
        residual = y - np.dot(X[:, :n_active], gamma)
        # 如果指定了 tol 并且残差的二范数平方小于等于 tol，则退出循环
        if tol is not None and nrm2(residual) ** 2 <= tol:
            break
        # 如果当前活跃变量数目达到 max_features，则退出循环
        elif n_active == max_features:
            break

    # 如果需要返回路径信息，则返回 gamma、indices[:n_active]、coefs[:, :n_active] 和当前活跃变量数目 n_active
    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        # 否则只返回 gamma、indices[:n_active] 和当前活跃变量数目 n_active
        return gamma, indices[:n_active], n_active
    # 如果需要复制 Gram 矩阵，使用 Fortran 顺序创建副本
    Gram = Gram.copy("F") if copy_Gram else np.asfortranarray(Gram)

    # 如果需要复制 Xy 向量或者 Xy 向量不可写，则创建 Xy 的副本
    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()

    # 获取 Gram 数据类型的机器精度信息，用于设置最小浮点数值
    min_float = np.finfo(Gram.dtype).eps

    # 获取 BLAS 和 LAPACK 函数
    nrm2, swap = linalg.get_blas_funcs(("nrm2", "swap"), (Gram,))
    (potrs,) = get_lapack_funcs(("potrs",), (Gram,))

    # 初始化索引数组，用于记录交换操作
    indices = np.arange(len(Gram))
    
    # 初始化 alpha 为 Xy
    alpha = Xy
    
    # 初始化当前容差为 tol_0
    tol_curr = tol_0
    
    # 初始化 delta 为 0
    delta = 0
    
    # 初始化 gamma 为空数组
    gamma = np.empty(0)
    
    # 初始化活跃特征数为 0
    n_active = 0
    
    # 如果 tol 不为 None，则最大特征数为 Gram 矩阵的长度，否则为 n_nonzero_coefs
    max_features = len(Gram) if tol is not None else n_nonzero_coefs
    
    # 初始化 L 矩阵，用于存储计算过程中的系数路径，数据类型与 Gram 矩阵相同
    L = np.empty((max_features, max_features), dtype=Gram.dtype)
    
    # 初始化 L 的左上角元素为 1.0
    L[0, 0] = 1.0
    
    # 如果需要返回路径信息，则初始化 coefs 矩阵，形状与 L 相同
    if return_path:
        coefs = np.empty_like(L)
    while True:
        # 选择具有最大绝对值的 alpha 元素的索引
        lam = np.argmax(np.abs(alpha))
        
        # 如果选择的索引 lam 小于当前活跃集的大小 n_active
        # 或者 alpha[lam] 的平方小于最小浮点数 min_float，则结束循环
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # 发出运行时警告，表示选择了相同的原子两次，或者内积太小
            warnings.warn(premature, RuntimeWarning, stacklevel=3)
            break
        
        # 如果当前活跃集的大小大于 0
        if n_active > 0:
            # 将 Gram 矩阵的部分复制到 L 矩阵中，并解三角系统
            L[n_active, :n_active] = Gram[lam, :n_active]
            linalg.solve_triangular(
                L[:n_active, :n_active],
                L[n_active, :n_active],
                trans=0,
                lower=1,
                overwrite_b=True,
                check_finite=False,
            )
            # 计算 L[n_active, :n_active] 的二范数的平方
            v = nrm2(L[n_active, :n_active]) ** 2
            # 计算 Lkk，表示 Gram[lam, lam] 减去 v
            Lkk = Gram[lam, lam] - v
            
            # 如果 Lkk 小于等于最小浮点数 min_float，则结束循环
            if Lkk <= min_float:
                warnings.warn(premature, RuntimeWarning, stacklevel=3)
                break
            
            # 将 sqrt(Lkk) 赋值给 L 矩阵的对角线元素 L[n_active, n_active]
            L[n_active, n_active] = sqrt(Lkk)
        else:
            # 当前活跃集大小为 0 时，将 sqrt(Gram[lam, lam]) 赋值给 L[0, 0]
            L[0, 0] = sqrt(Gram[lam, lam])
        
        # 交换 Gram 矩阵中的两行和两列的元素
        Gram[n_active], Gram[lam] = swap(Gram[n_active], Gram[lam])
        Gram.T[n_active], Gram.T[lam] = swap(Gram.T[n_active], Gram.T[lam])
        
        # 交换 indices 数组中的两个索引位置的元素
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        
        # 交换 Xy 数组中的两个元素
        Xy[n_active], Xy[lam] = Xy[lam], Xy[n_active]
        
        # 增加当前活跃集的大小 n_active
        n_active += 1
        
        # 解 LL'x = X'y 作为两个三角系统的组合
        gamma, _ = potrs(
            L[:n_active, :n_active], Xy[:n_active], lower=True, overwrite_b=False
        )
        
        # 如果需要返回路径，则将当前的 gamma 存入 coefs 数组
        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        
        # 计算 beta，表示 Gram 矩阵与 gamma 的乘积
        beta = np.dot(Gram[:, :n_active], gamma)
        
        # 更新 alpha 数组
        alpha = Xy - beta
        
        # 如果指定了 tol（容差）
        if tol is not None:
            tol_curr += delta
            delta = np.inner(gamma, beta[:n_active])
            tol_curr -= delta
            
            # 如果当前容差的绝对值小于等于指定的 tol，则结束循环
            if abs(tol_curr) <= tol:
                break
        
        # 如果当前活跃集的大小达到了最大特征数 max_features，则结束循环
        elif n_active == max_features:
            break
    
    # 如果需要返回路径，则返回 gamma、indices、coefs 和当前活跃集的大小 n_active
    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        # 否则，返回 gamma、indices 和当前活跃集的大小 n_active
        return gamma, indices[:n_active], n_active
# 使用装饰器 @validate_params 进行参数验证，确保函数输入的参数满足特定类型和条件
@validate_params(
    {
        "X": ["array-like"],  # 参数 X 应为类数组
        "y": [np.ndarray],    # 参数 y 应为 numpy 数组
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],  # 参数 n_nonzero_coefs 应为整数，且大于等于1
        "tol": [Interval(Real, 0, None, closed="left"), None],  # 参数 tol 应为非负实数
        "precompute": ["boolean", StrOptions({"auto"})],  # 参数 precompute 可以为布尔值或字符串 "auto"
        "copy_X": ["boolean"],  # 参数 copy_X 应为布尔值
        "return_path": ["boolean"],  # 参数 return_path 应为布尔值
        "return_n_iter": ["boolean"],  # 参数 return_n_iter 应为布尔值
    },
    prefer_skip_nested_validation=True,  # 设定优先跳过嵌套验证
)
# 定义函数 orthogonal_mp，实现正交匹配追踪算法（OMP）
def orthogonal_mp(
    X,
    y,
    *,
    n_nonzero_coefs=None,  # 非零系数的数量，默认为 None
    tol=None,  # 允许的残差的最大平方范数，默认为 None
    precompute=False,  # 是否预计算，默认为 False
    copy_X=True,  # 是否复制输入数据 X，默认为 True
    return_path=False,  # 是否返回非零系数的完整路径，默认为 False
    return_n_iter=False,  # 是否返回迭代次数，默认为 False
):
    r"""Orthogonal Matching Pursuit (OMP).

    解决 n_targets 个正交匹配追踪问题。
    每个问题的形式为：

    当以非零系数的数量 `n_nonzero_coefs` 作为参数时：
    argmin ||y - X\gamma||^2，满足 ||\gamma||_0 <= n_{nonzero coefs}

    当以误差 `tol` 作为参数时：
    argmin ||\gamma||_0，满足 ||y - X\gamma||^2 <= tol

    详细信息请参阅 :ref:`User Guide <omp>`。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        输入数据。假设列已归一化。

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        输入的目标值。

    n_nonzero_coefs : int, default=None
        解决方案中所需的非零条目数。如果为 None（默认），则该值设为 n_features 的 10%。

    tol : float, default=None
        残差的最大平方范数。如果不为 None，则覆盖 n_nonzero_coefs。

    precompute : 'auto' 或 bool, default=False
        是否执行预计算。当 n_targets 或 n_samples 很大时，可以提高性能。

    copy_X : bool, default=True
        算法是否必须复制设计矩阵 X。如果 X 已经按 Fortran 顺序排列，则 false 值很有用，否则仍会进行复制。

    return_path : bool, default=False
        是否返回沿前向路径的每个非零系数的值。对交叉验证很有用。

    return_n_iter : bool, default=False
        是否返回迭代次数。

    Returns
    -------
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        OMP 解的系数。如果 `return_path=True`，则包含整个系数路径。在这种情况下，其形状为
        (n_features, n_features) 或 (n_features, n_targets, n_features)，并且沿最后轴迭代会生成
        按照活动特征增加的系数。

    n_iters : array-like or int
        每个目标上的活动特征数量。仅在 `return_n_iter=True` 时返回。

    See Also
    --------
    OrthogonalMatchingPursuit : 正交匹配追踪模型。
    orthogonal_mp_gram : 使用 Gram 矩阵和乘积 X.T * y 解决 OMP 问题。
    # 检查并确保 X 是按列主序排列的数组，如果不是则进行复制
    X = check_array(X, order="F", copy=copy_X)
    copy_X = False  # 标记是否复制输入数组 X，默认为 False

    # 如果 y 的维度为 1，则将其重塑为列向量
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y = check_array(y)  # 确保 y 是一个数组

    # 如果 y 的列数大于 1，则需要复制输入数组 X
    if y.shape[1] > 1:  # 后续的目标将受到影响
        copy_X = True

    # 如果未指定非零系数的个数且未指定容差
    if n_nonzero_coefs is None and tol is None:
        # 默认情况下，非零系数个数为特征数的 10%，但至少为 1 个
        n_nonzero_coefs = max(int(0.1 * X.shape[1]), 1)

    # 如果未指定容差且非零系数个数大于特征数，则引发 ValueError
    if tol is None and n_nonzero_coefs > X.shape[1]:
        raise ValueError(
            "The number of atoms cannot be more than the number of features"
        )

    # 如果 precompute 设置为 "auto"，则根据 X 的形状决定是否预先计算 Gram 矩阵
    if precompute == "auto":
        precompute = X.shape[0] > X.shape[1]

    # 如果需要预先计算 Gram 矩阵
    if precompute:
        # 计算 Gram 矩阵 G 和 X.T @ y
        G = np.dot(X.T, X)
        G = np.asfortranarray(G)  # 将 G 转换为列主序存储
        Xy = np.dot(X.T, y)

        # 如果指定了容差 tol，则计算 y 的平方范数
        if tol is not None:
            norms_squared = np.sum((y**2), axis=0)
        else:
            norms_squared = None

        # 调用 orthogonal_mp_gram 函数并返回结果
        return orthogonal_mp_gram(
            G,
            Xy,
            n_nonzero_coefs=n_nonzero_coefs,
            tol=tol,
            norms_squared=norms_squared,
            copy_Gram=copy_X,
            copy_Xy=False,
            return_path=return_path,
        )

    # 如果需要返回路径信息
    if return_path:
        # 初始化系数矩阵 coef，形状为 (X.shape[1], y.shape[1], X.shape[1])
        coef = np.zeros((X.shape[1], y.shape[1], X.shape[1]))
    else:
        # 初始化系数矩阵 coef，形状为 (X.shape[1], y.shape[1])
        coef = np.zeros((X.shape[1], y.shape[1]))

    # 存储每个目标的迭代次数
    n_iters = []

    # 遍历每个目标
    for k in range(y.shape[1]):
        # 调用 _cholesky_omp 函数进行正交匹配追踪
        out = _cholesky_omp(
            X, y[:, k], n_nonzero_coefs, tol, copy_X=copy_X, return_path=return_path
        )

        # 如果需要返回路径信息
        if return_path:
            _, idx, coefs, n_iter = out
            coef = coef[:, :, : len(idx)]
            # 将系数数组 coefs 复制到 coef 中
            for n_active, x in enumerate(coefs.T):
                coef[idx[: n_active + 1], k, n_active] = x[: n_active + 1]
        else:
            # 否则，直接将稀疏解 x 和对应的索引 idx 复制到 coef 中
            x, idx, n_iter = out
            coef[idx, k] = x

        # 记录当前目标的迭代次数
        n_iters.append(n_iter)

    # 如果 y 的列数为 1，则将 n_iters 重置为单个值
    if y.shape[1] == 1:
        n_iters = n_iters[0]

    # 如果需要返回迭代次数，将系数矩阵 coef 和迭代次数列表 n_iters 返回
    if return_n_iter:
        return np.squeeze(coef), n_iters
    # 如果条件不满足（即 coef 不是一个数组或者数组的维度大于1），返回 coef 的压缩版本
    else:
        return np.squeeze(coef)
# 使用装饰器 @validate_params 进行参数验证，确保函数输入参数的类型和取值符合要求
@validate_params(
    {
        "Gram": ["array-like"],
        "Xy": ["array-like"],
        "n_nonzero_coefs": [Interval(Integral, 0, None, closed="neither"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "norms_squared": ["array-like", None],
        "copy_Gram": ["boolean"],
        "copy_Xy": ["boolean"],
        "return_path": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
# 定义函数 orthogonal_mp_gram，实现 Gram Orthogonal Matching Pursuit (OMP) 算法
def orthogonal_mp_gram(
    Gram,
    Xy,
    *,
    n_nonzero_coefs=None,  # 非零系数的期望数量，默认为 None
    tol=None,              # 残差的最大平方范数，默认为 None
    norms_squared=None,    # y 的每行的平方 L2 范数的数组，默认为 None
    copy_Gram=True,        # 是否复制 Gram 矩阵，默认为 True
    copy_Xy=True,          # 是否复制协方差向量 Xy，默认为 True
    return_path=False,     # 是否返回沿着前向路径的每个非零系数的值，默认为 False
    return_n_iter=False,   # 是否返回迭代次数，默认为 False
):
    """Gram Orthogonal Matching Pursuit (OMP).

    Solves n_targets Orthogonal Matching Pursuit problems using only
    the Gram matrix X.T * X and the product X.T * y.

    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    Gram : array-like of shape (n_features, n_features)
        Gram matrix of the input data: `X.T * X`.

    Xy : array-like of shape (n_features,) or (n_features, n_targets)
        Input targets multiplied by `X`: `X.T * y`.

    n_nonzero_coefs : int, default=None
        Desired number of non-zero entries in the solution. If `None` (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum squared norm of the residual. If not `None`,
        overrides `n_nonzero_coefs`.

    norms_squared : array-like of shape (n_targets,), default=None
        Squared L2 norms of the lines of `y`. Required if `tol` is not None.

    copy_Gram : bool, default=True
        Whether the gram matrix must be copied by the algorithm. A `False`
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, default=True
        Whether the covariance vector `Xy` must be copied by the algorithm.
        If `False`, it may be overwritten.

    return_path : bool, default=False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        Coefficients of the OMP solution. If `return_path=True`, this contains
        the whole coefficient path. In this case its shape is
        `(n_features, n_features)` or `(n_features, n_targets, n_features)` and
        iterating over the last axis yields coefficients in increasing order
        of active features.

    n_iters : list or int
        Number of active features across every target. Returned only if
        `return_n_iter` is set to True.

    See Also
    --------
    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model (OMP).
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    """
    # 根据给定的 Gram 矩阵和 Xy 数据进行 Gram 矩阵正交匹配追踪算法
    def orthogonal_mp_gram(Gram, Xy, n_nonzero_coefs=None, tol=None, norms_squared=None, copy_Xy=False, return_path=False, copy_Gram=False):
        # 确保 Gram 是 Fortran order，同时进行必要的复制
        Gram = check_array(Gram, order="F", copy=copy_Gram)
        # 将 Xy 转换为 ndarray 格式
        Xy = np.asarray(Xy)
        
        # 如果 Xy 的维度大于 1 且列数大于 1，则进行特定处理以确保一致性
        if Xy.ndim > 1 and Xy.shape[1] > 1:
            copy_Gram = True  # 或后续的目标将受到影响
        
        # 如果 Xy 的维度为 1，则将其转换为二维数组
        if Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]
            # 如果给定了 tol 参数，则需要计算 norms_squared
            if tol is not None:
                norms_squared = [norms_squared]
        
        # 如果需要复制 Xy 或者 Xy 不可写，则进行一次性复制操作
        if copy_Xy or not Xy.flags.writeable:
            Xy = Xy.copy()
    
        # 如果未指定非零系数的数量且未指定 tol 参数，则设定默认非零系数数量为 Gram 矩阵长度的 10%
        if n_nonzero_coefs is None and tol is None:
            n_nonzero_coefs = int(0.1 * len(Gram))
        
        # 如果给定了 tol 参数但未给出 norms_squared，则抛出 ValueError
        if tol is not None and norms_squared is None:
            raise ValueError(
                "Gram OMP 需要预先计算的 norms 以评估误差平方和."
            )
        
        # 如果 tol 参数小于 0，则抛出 ValueError
        if tol is not None and tol < 0:
            raise ValueError("Epsilon 不能为负数")
        
        # 如果未指定 tol 参数且非零系数数量小于等于 0，则抛出 ValueError
        if tol is None and n_nonzero_coefs <= 0:
            raise ValueError("原子的数量必须为正数")
        
        # 如果未指定 tol 参数且非零系数数量大于 Gram 矩阵长度，则抛出 ValueError
        if tol is None and n_nonzero_coefs > len(Gram):
            raise ValueError(
                "原子的数量不能超过特征的数量"
            )
    
        # 如果需要返回路径信息，则初始化一个全零数组，形状根据参数决定
        if return_path:
            coef = np.zeros((len(Gram), Xy.shape[1], len(Gram)), dtype=Gram.dtype)
        else:
            coef = np.zeros((len(Gram), Xy.shape[1]), dtype=Gram.dtype)
    
        # 初始化迭代次数的列表
        n_iters = []
    # 对于每列特征向量执行 Gram 矩阵的正交匹配追踪算法（OMP）
    for k in range(Xy.shape[1]):
        # 调用 _gram_omp 函数，执行正交匹配追踪算法
        out = _gram_omp(
            Gram,  # Gram 矩阵
            Xy[:, k],  # 第 k 列特征向量
            n_nonzero_coefs,  # 非零系数的数量
            norms_squared[k] if tol is not None else None,  # 若 tol 不为 None，则使用 norms_squared[k]
            tol,  # 容差阈值
            copy_Gram=copy_Gram,  # 是否复制 Gram 矩阵
            copy_Xy=False,  # 不复制 Xy
            return_path=return_path,  # 是否返回路径
        )
        # 如果设置了 return_path，则解压 out 的返回值
        if return_path:
            _, idx, coefs, n_iter = out
            # 将 coef 裁剪至 idx 列的大小
            coef = coef[:, :, : len(idx)]
            # 遍历系数 coefs 的转置，并记录每个活跃迭代的结果
            for n_active, x in enumerate(coefs.T):
                coef[idx[: n_active + 1], k, n_active] = x[: n_active + 1]
        else:
            # 否则，解压 out 的返回值
            x, idx, n_iter = out
            # 将系数 x 放入 coef 的对应位置
            coef[idx, k] = x
        # 记录迭代次数
        n_iters.append(n_iter)

    # 如果 Xy 的列数为 1，只保留 n_iters 的第一个元素
    if Xy.shape[1] == 1:
        n_iters = n_iters[0]

    # 如果需要返回迭代次数，则返回压缩后的 coef 和 n_iters
    if return_n_iter:
        return np.squeeze(coef), n_iters
    else:
        # 否则，仅返回压缩后的 coef
        return np.squeeze(coef)
# 定义 Orthogonal Matching Pursuit (OMP) 模型，继承自 MultiOutputMixin、RegressorMixin 和 LinearModel
class OrthogonalMatchingPursuit(MultiOutputMixin, RegressorMixin, LinearModel):
    """Orthogonal Matching Pursuit model (OMP).
    
    Read more in the :ref:`User Guide <omp>`.

    Parameters
    ----------
    n_nonzero_coefs : int, default=None
        Desired number of non-zero entries in the solution. Ignored if `tol` is set.
        When `None` and `tol` is also `None`, this value is either set to 10% of
        `n_features` or 1, whichever is greater.

    tol : float, default=None
        Maximum squared norm of the residual. If not None, overrides n_nonzero_coefs.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : 'auto' or bool, default='auto'
        Whether to use a precomputed Gram and Xy matrix to speed up
        calculations. Improves performance when :term:`n_targets` or
        :term:`n_samples` is very large. Note that if you already have such
        matrices, you can pass them directly to the fit method.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formula).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : int or array-like
        Number of active features across every target.

    n_nonzero_coefs_ : int or None
        The number of non-zero coefficients in the solution or `None` when `tol` is
        set. If `n_nonzero_coefs` is None and `tol` is None this value is either set
        to 10% of `n_features` or 1, whichever is greater.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    orthogonal_mp_gram :  Solves n_targets Orthogonal Matching Pursuit
        problems using only the Gram matrix X.T * X and the product X.T * y.
    lars_path : Compute Least Angle Regression or Lasso path using LARS algorithm.
    Lars : Least Angle Regression model a.k.a. LAR.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    sklearn.decomposition.sparse_encode : Generic sparse coding.
        Each column of the result is the solution to a Lasso problem.
    OrthogonalMatchingPursuitCV : Cross-validated
        Orthogonal Matching Pursuit model (OMP).

    Notes
    -----
    Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
    Matching pursuits with time-frequency dictionaries, IEEE Transactions on
    Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
    """
    """
    (https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf)
    
    This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
    M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
    Matching Pursuit Technical Report - CS Technion, April 2008.
    https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
    
    Examples
    --------
    >>> from sklearn.linear_model import OrthogonalMatchingPursuit
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> reg = OrthogonalMatchingPursuit().fit(X, y)
    >>> reg.score(X, y)
    0.9991...
    >>> reg.predict(X[:1,])
    array([-78.3854...])
    """
    
    # 定义参数约束字典，描述了类的参数限制和类型
    _parameter_constraints: dict = {
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "fit_intercept": ["boolean"],
        "precompute": [StrOptions({"auto"}), "boolean"],
    }
    
    def __init__(
        self,
        *,
        n_nonzero_coefs=None,
        tol=None,
        fit_intercept=True,
        precompute="auto",
    ):
        # 初始化类实例时设置参数
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.precompute = precompute
    
    @_fit_context(prefer_skip_nested_validation=True)
    # 使用输入的训练数据 X 和目标数据 y 进行模型拟合
    def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # 调用 _validate_data 方法验证输入数据的格式，并确保类型匹配
        X, y = self._validate_data(X, y, multi_output=True, y_numeric=True)
        # 获取特征的数量
        n_features = X.shape[1]

        # 预处理数据，包括中心化、标准化等，并返回处理后的数据及相关信息
        X, y, X_offset, y_offset, X_scale, Gram, Xy = _pre_fit(
            X, y, None, self.precompute, self.fit_intercept, copy=True
        )

        # 如果 y 的维度为 1，将其转换为列向量
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # 根据参数设置 n_nonzero_coefs_ 的值，用于控制稀疏性
        if self.n_nonzero_coefs is None and self.tol is None:
            # 默认情况下，n_nonzero_coefs 的值为特征数量的 10%，但至少为 1
            self.n_nonzero_coefs_ = max(int(0.1 * n_features), 1)
        elif self.tol is not None:
            self.n_nonzero_coefs_ = None
        else:
            self.n_nonzero_coefs_ = self.n_nonzero_coefs

        # 根据 Gram 矩阵的情况，调用相应的正交匹配追踪方法进行拟合
        if Gram is False:
            # 使用正交匹配追踪算法拟合模型，返回系数和迭代次数
            coef_, self.n_iter_ = orthogonal_mp(
                X,
                y,
                n_nonzero_coefs=self.n_nonzero_coefs_,
                tol=self.tol,
                precompute=False,
                copy_X=True,
                return_n_iter=True,
            )
        else:
            # 使用基于 Gram 矩阵的正交匹配追踪算法拟合模型，返回系数和迭代次数
            norms_sq = np.sum(y**2, axis=0) if self.tol is not None else None

            coef_, self.n_iter_ = orthogonal_mp_gram(
                Gram,
                Xy=Xy,
                n_nonzero_coefs=self.n_nonzero_coefs_,
                tol=self.tol,
                norms_squared=norms_sq,
                copy_Gram=True,
                copy_Xy=True,
                return_n_iter=True,
            )
        
        # 将系数转置后保存到模型属性中
        self.coef_ = coef_.T
        # 设置模型的截距属性
        self._set_intercept(X_offset, y_offset, X_scale)
        # 返回拟合后的模型实例
        return self
# 定义一个函数 `_omp_path_residues`，用于计算完整 LARS 路径上的左出数据的残差。

def _omp_path_residues(
    X_train,
    y_train,
    X_test,
    y_test,
    copy=True,
    fit_intercept=True,
    max_iter=100,
):
    """Compute the residues on left-out data for a full LARS path.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        The data to fit the LARS on.

    y_train : ndarray of shape (n_samples)
        The target variable to fit LARS on.

    X_test : ndarray of shape (n_samples, n_features)
        The data to compute the residues on.

    y_test : ndarray of shape (n_samples)
        The target variable to compute the residues on.

    copy : bool, default=True
        Whether X_train, X_test, y_train and y_test should be copied.  If
        False, they may be overwritten.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    max_iter : int, default=100
        Maximum numbers of iterations to perform, therefore maximum features
        to include. 100 by default.

    Returns
    -------
    residues : ndarray of shape (n_samples, max_features)
        Residues of the prediction on the test data.
    """

    # 如果 copy 参数为 True，则复制输入的数据，以保护原始数据
    if copy:
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_test = X_test.copy()
        y_test = y_test.copy()

    # 如果 fit_intercept 参数为 True，则进行数据中心化处理
    if fit_intercept:
        # 计算 X_train 的均值并进行中心化
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        # 计算 X_test 的均值并进行中心化
        X_test -= X_mean
        # 计算 y_train 的均值并转换为浮点数组后进行中心化
        y_mean = y_train.mean(axis=0)
        y_train = as_float_array(y_train, copy=False)
        y_train -= y_mean
        # 计算 y_test 的均值并转换为浮点数组后进行中心化
        y_test = as_float_array(y_test, copy=False)
        y_test -= y_mean

    # 调用 orthogonal_mp 函数进行正交匹配追踪
    coefs = orthogonal_mp(
        X_train,
        y_train,
        n_nonzero_coefs=max_iter,
        tol=None,
        precompute=False,
        copy_X=False,
        return_path=True,
    )
    
    # 如果 coefs 的维度为 1，则添加一个维度使其成为列向量
    if coefs.ndim == 1:
        coefs = coefs[:, np.newaxis]

    # 返回预测结果的残差，使用 coefs 的转置与 X_test 的转置相乘再减去 y_test
    return np.dot(coefs.T, X_test.T) - y_test
    cv : int, cross-validation generator or iterable, default=None
        # 定义变量 cv，用于指定交叉验证的策略，可以是整数、交叉验证生成器或可迭代对象，默认为 None。

        Determines the cross-validation splitting strategy.
        # 确定交叉验证的分割策略。

        Possible inputs for cv are:
        # cv 参数可以接受以下输入：

        - None, to use the default 5-fold cross-validation,
        # None 表示使用默认的 5 折交叉验证。

        - integer, to specify the number of folds.
        # 整数表示指定的折数。

        - :term:`CV splitter`,
        # CV splitter 表示交叉验证分离器。

        - An iterable yielding (train, test) splits as arrays of indices.
        # 返回一个可迭代对象，产生 (训练集, 测试集) 分割的索引数组。

        For integer/None inputs, :class:`~sklearn.model_selection.KFold` is used.
        # 对于整数或 None 输入，使用 :class:`~sklearn.model_selection.KFold` 进行交叉验证。

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        # 参考用户指南了解可用的交叉验证策略。

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
            # 如果 cv 参数为 None，默认值从 3 折更改为 5 折。

    n_jobs : int, default=None
        # 定义变量 n_jobs，用于指定在交叉验证过程中使用的 CPU 核数，默认为 None。

        Number of CPUs to use during the cross validation.
        # 在交叉验证期间要使用的 CPU 核数。

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        # ``None`` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则默认为 1。

        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        # ``-1`` 表示使用所有处理器。详细信息请参阅 :term:`Glossary <n_jobs>`。

    verbose : bool or int, default=False
        # 定义变量 verbose，用于设置详细程度的参数，默认为 False。

        Sets the verbosity amount.
        # 设置详细程度的参数。

    Attributes
    ----------
    intercept_ : float or ndarray of shape (n_targets,)
        # 定义属性 intercept_，表示决策函数中的独立项，可以是浮点数或形状为 (n_targets,) 的数组。

        Independent term in decision function.
        # 决策函数中的独立项。

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        # 定义属性 coef_，表示参数向量，在问题表述中为 w。

        Parameter vector (w in the problem formulation).
        # 参数向量（问题表述中的 w）。

    n_nonzero_coefs_ : int
        # 定义属性 n_nonzero_coefs_，表示在交叉验证的所有折叠中得到最佳均方误差的非零系数的数量估计。

        Estimated number of non-zero coefficients giving the best mean squared
        error over the cross-validation folds.
        # 在交叉验证的所有折叠中得到最佳均方误差的非零系数的数量估计。

    n_iter_ : int or array-like
        # 定义属性 n_iter_，表示使用通过所有折叠交叉验证得到的最佳超参数重新拟合模型时，每个目标的活动特征数。

        Number of active features across every target for the model refit with
        the best hyperparameters got by cross-validating across all folds.
        # 使用通过所有折叠交叉验证得到的最佳超参数重新拟合模型时，每个目标的活动特征数。

    n_features_in_ : int
        # 定义属性 n_features_in_，表示在拟合过程中看到的特征数。

        Number of features seen during :term:`fit`.
        # 在拟合过程中看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 定义属性 feature_names_in_，表示在拟合过程中看到的特征名称数组。

        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        # 在拟合过程中看到的特征名称。仅当 `X` 的特征名称都是字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    orthogonal_mp : Solves n_targets Orthogonal Matching Pursuit problems.
    orthogonal_mp_gram : Solves n_targets Orthogonal Matching Pursuit
        problems using only the Gram matrix X.T * X and the product X.T * y.
    lars_path : Compute Least Angle Regression or Lasso path using LARS algorithm.
    Lars : Least Angle Regression model a.k.a. LAR.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model (OMP).
    LarsCV : Cross-validated Least Angle Regression model.
    LassoLarsCV : Cross-validated Lasso model fit with Least Angle Regression.
    sklearn.decomposition.sparse_encode : Generic sparse coding.
        Each column of the result is the solution to a Lasso problem.
        # 参考其他与本模型相关的类和方法。

    Notes
    -----
    In `fit`, once the optimal number of non-zero coefficients is found through
    cross-validation, the model is fit again using the entire training set.
    # 在 `fit` 方法中，一旦通过交叉验证找到了最佳的非零系数数量，模型会再次使用整个训练集进行拟合。

    Examples
    --------
    _parameter_constraints: dict = {
        # 参数约束字典，指定了每个参数的类型约束
        "copy": ["boolean"],  # 'copy' 参数类型应为布尔值
        "fit_intercept": ["boolean"],  # 'fit_intercept' 参数类型应为布尔值
        "max_iter": [Interval(Integral, 0, None, closed="left"), None],
        # 'max_iter' 参数类型应为整数或None，在区间 [0, ∞) 内
        "cv": ["cv_object"],  # 'cv' 参数类型应为交叉验证对象
        "n_jobs": [Integral, None],  # 'n_jobs' 参数类型应为整数或None
        "verbose": ["verbose"],  # 'verbose' 参数类型应为详细输出控制
    }

    def __init__(
        self,
        *,
        copy=True,  # 默认值为 True
        fit_intercept=True,  # 默认值为 True
        max_iter=None,  # 默认值为 None
        cv=None,  # 默认值为 None
        n_jobs=None,  # 默认值为 None
        verbose=False,  # 默认值为 False
    ):
        # 初始化函数，设定参数值
        self.copy = copy  # 设置实例变量 'copy'
        self.fit_intercept = fit_intercept  # 设置实例变量 'fit_intercept'
        self.max_iter = max_iter  # 设置实例变量 'max_iter'
        self.cv = cv  # 设置实例变量 'cv'
        self.n_jobs = n_jobs  # 设置实例变量 'n_jobs'
        self.verbose = verbose  # 设置实例变量 'verbose'

    @_fit_context(prefer_skip_nested_validation=True)
    # 应用 _fit_context 装饰器，指定 prefer_skip_nested_validation 参数为 True
    def fit(self, X, y, **fit_params):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        **fit_params : dict
            Parameters to pass to the underlying splitter.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # 检查参数的有效性并抛出异常
        _raise_for_params(fit_params, self, "fit")

        # 验证数据的有效性，确保数据可用于拟合，转换成浮点类型数组
        X, y = self._validate_data(X, y, y_numeric=True, ensure_min_features=2)
        X = as_float_array(X, copy=False, force_all_finite=False)
        
        # 检查并获取交叉验证策略
        cv = check_cv(self.cv, classifier=False)
        
        # 根据元数据路由的开启情况，处理拟合过程中的参数
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            # 当元数据路由不可关闭时，创建空的参数集合，用于兼容性处理
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            routed_params.splitter = Bunch(split={})
        
        # 计算最大迭代次数，根据输入数据维度动态调整
        max_iter = (
            min(max(int(0.1 * X.shape[1]), 5), X.shape[1])
            if not self.max_iter
            else self.max_iter
        )
        
        # 并行执行交叉验证拟合路径计算
        cv_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_omp_path_residues)(
                X[train],
                y[train],
                X[test],
                y[test],
                self.copy,
                self.fit_intercept,
                max_iter,
            )
            for train, test in cv.split(X, **routed_params.splitter.split)
        )

        # 获取最小早停步数，以及每折的均方误差
        min_early_stop = min(fold.shape[0] for fold in cv_paths)
        mse_folds = np.array(
            [(fold[:min_early_stop] ** 2).mean(axis=1) for fold in cv_paths]
        )
        
        # 获取最佳非零系数个数
        best_n_nonzero_coefs = np.argmin(mse_folds.mean(axis=0)) + 1
        self.n_nonzero_coefs_ = best_n_nonzero_coefs
        
        # 使用正交匹配追踪算法拟合模型
        omp = OrthogonalMatchingPursuit(
            n_nonzero_coefs=best_n_nonzero_coefs,
            fit_intercept=self.fit_intercept,
        ).fit(X, y)

        # 设置模型系数、截距和迭代次数
        self.coef_ = omp.coef_
        self.intercept_ = omp.intercept_
        self.n_iter_ = omp.n_iter_
        
        # 返回当前对象实例
        return self
    def get_metadata_routing(self):
        """
        Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """

        # 创建一个 MetadataRouter 对象，指定所有者为当前对象的类名
        router = MetadataRouter(owner=self.__class__.__name__).add(
            splitter=self.cv,  # 设置路由的分隔器为当前对象的交叉验证器
            method_mapping=MethodMapping().add(caller="fit", callee="split"),  # 添加方法映射，将 fit 方法映射到 split 方法
        )
        # 返回包含路由信息的 MetadataRouter 对象
        return router
```