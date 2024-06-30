# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_nmf.py`

```
# 非负矩阵分解的实现。

# 作者：scikit-learn 开发者
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
import itertools  # 提供迭代工具的函数
import time  # 时间相关的功能
import warnings  # 控制警告消息的显示
from abc import ABC  # Python 抽象基类的支持
from math import sqrt  # 提供数学函数 sqrt()
from numbers import Integral, Real  # 数字类型的抽象基类

import numpy as np  # 数组和矩阵运算
import scipy.sparse as sp  # 稀疏矩阵和相关算法
from scipy import linalg  # 线性代数函数库

# 导入 scikit-learn 内部模块和函数
from .._config import config_context  # 导入配置上下文
from ..base import (  # 导入基础估算器、特征输出前缀、转换器混合类、_fit 上下文
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..exceptions import ConvergenceWarning  # 导入收敛警告
from ..utils import (  # 导入数组验证、随机状态验证、批次生成器、元数据路由等工具函数
    check_array,
    check_random_state,
    gen_batches,
    metadata_routing,
)
from ..utils._param_validation import (  # 导入隐藏参数、区间验证、字符串选项验证、参数验证
    Hidden,
    Interval,
    StrOptions,
    validate_params,
)
from ..utils.deprecation import _deprecate_Xt_in_inverse_transform  # 导入反向转换警告
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm  # 导入随机 SVD、稀疏点积、平方范数函数
from ..utils.validation import (  # 导入已安装验证、非负验证
    check_is_fitted,
    check_non_negative,
)
from ._cdnmf_fast import _update_cdnmf_fast  # 导入更新 CDNMF 快速函数

EPSILON = np.finfo(np.float32).eps  # 机器 epsilon，用于浮点数比较


def norm(x):
    """基于点积的欧几里得范数实现。

    参考：http://fa.bianp.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        要计算范数的向量。
    """
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    """计算 np.dot(X, Y.T) 的迹。

    Parameters
    ----------
    X : array-like
        第一个矩阵。
    Y : array-like
        第二个矩阵。
    """
    return np.dot(X.ravel(), Y.ravel())


def _check_init(A, shape, whom):
    """验证初始化数组 A 是否符合形状要求。

    Parameters
    ----------
    A : array-like
        要验证的数组。
    shape : tuple
        期望的数组形状。
    whom : str
        调用此验证的函数名称。
    """
    A = check_array(A)  # 使用 scikit-learn 的 check_array 验证数组 A
    if shape[0] != "auto" and A.shape[0] != shape[0]:
        raise ValueError(
            f"Array with wrong first dimension passed to {whom}. Expected {shape[0]}, "
            f"but got {A.shape[0]}."
        )
    if shape[1] != "auto" and A.shape[1] != shape[1]:
        raise ValueError(
            f"Array with wrong second dimension passed to {whom}. Expected {shape[1]}, "
            f"but got {A.shape[1]}."
        )
    check_non_negative(A, whom)  # 检查数组 A 是否为非负数
    if np.max(A) == 0:
        raise ValueError(f"Array passed to {whom} is full of zeros.")


def _beta_divergence(X, W, H, beta, square_root=False):
    """计算 X 和 dot(W, H) 的 beta 散度。

    Parameters
    ----------
    X : float or array-like of shape (n_samples, n_features)
        第一个矩阵或向量。

    W : float or array-like of shape (n_samples, n_components)
        第二个矩阵或向量。

    H : float or array-like of shape (n_components, n_features)
        第三个矩阵。

    beta : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}
        beta 散度的参数。
        如果 beta == 2，这是 Frobenius 范数的一半的平方。
        如果 beta == 1，这是广义 Kullback-Leibler 散度。
        如果 beta == 0，这是 Itakura-Saito 散度。
        否则，这是一般的 beta 散度。

    square_root : bool, default=False
        如果为 True，返回 np.sqrt(2 * res)。
        对于 beta == 2，对应于 Frobenius 范数。

    Returns
    -------
    """
    -------
        res : float
            Beta divergence of X and np.dot(X, H).
    """
    # 将 beta 转换为浮点数
    beta = _beta_loss_to_float(beta)

    # 方法可以接受标量输入，将 X 至少转换为二维数组
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    # 将 W 和 H 至少转换为二维数组
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius 范数
    if beta == 2:
        # 如果 X 是稀疏矩阵，避免创建稠密的 np.dot(W, H)
        if sp.issparse(X):
            # 计算 X 的数据部分的内积
            norm_X = np.dot(X.data, X.data)
            # 计算 np.linalg.multi_dot([W.T, W, H]) 与 H 的迹的内积
            norm_WH = trace_dot(np.linalg.multi_dot([W.T, W, H]), H)
            # 计算 (X @ H.T) 与 W 的迹的内积
            cross_prod = trace_dot((X @ H.T), W)
            # 计算 Beta divergence
            res = (norm_X + norm_WH - 2.0 * cross_prod) / 2.0
        else:
            # 计算 Beta divergence
            res = squared_norm(X - np.dot(W, H)) / 2.0

        # 如果需要平方根，则返回 Beta divergence 的平方根
        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    # 如果 X 是稀疏矩阵，仅计算 X 非零部分的 np.dot(W, H)
    if sp.issparse(X):
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # 不影响零点：这里 0 ** (-1) = 0 而不是无穷大
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # 用于避免除以零
    WH_data[WH_data < EPSILON] = EPSILON

    # 广义 Kullback-Leibler divergence
    if beta == 1:
        # 快速且内存高效地计算 np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # 仅在 X 非零部分计算 np.sum(X * log(X / WH))
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # 加上 np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.prod(X.shape) - np.sum(np.log(div))

    # beta-divergence，beta 不在 (0, 1, 2) 范围内
    else:
        if sp.issparse(X):
            # 慢速循环，但内存高效地计算 np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)
        else:
            sum_WH_beta = np.sum(WH**beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data**beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    # 如果需要平方根，则取正数避免由于舍入误差而产生负数
    if square_root:
        res = max(res, 0)
        return np.sqrt(2 * res)
    else:
        return res
def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    # Check if X is sparse
    if sp.issparse(X):
        # Get indices of non-zero elements in X
        ii, jj = X.nonzero()
        # Number of non-zero values
        n_vals = ii.shape[0]
        # Initialize array for dot products
        dot_vals = np.empty(n_vals)
        # Number of components in W
        n_components = W.shape[1]

        # Determine batch size for computation efficiency
        batch_size = max(n_components, n_vals // n_components)
        # Compute dot products in batches
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            # Perform element-wise multiplication and sum along rows
            dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(axis=1)

        # Construct sparse COO matrix from computed dot products
        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()  # Convert COO matrix to CSR format for efficiency
    else:
        # If X is dense, compute the full dot product
        return np.dot(W, H)


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float."""
    # Mapping of string beta_loss to numerical value
    beta_loss_map = {"frobenius": 2, "kullback-leibler": 1, "itakura-saito": 0}
    # Convert if beta_loss is a string
    if isinstance(beta_loss, str):
        beta_loss = beta_loss_map[beta_loss]
    return beta_loss


def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------

    """
    """
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """

    # 检查输入矩阵 X 是否非负，用于 NMF 初始化
    check_non_negative(X, "NMF initialization")
    
    # 获取矩阵 X 的样本数和特征数
    n_samples, n_features = X.shape

    # 如果初始化参数 init 存在且不为 "random"，并且 n_components 大于最小的样本数和特征数
    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        # 抛出值错误，要求 init 只能在 n_components <= min(n_samples, n_features) 时使用
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    # 如果初始化参数 init 为空
    if init is None:
        # 如果 n_components 小于等于最小的样本数和特征数，则使用 "nndsvda" 初始化
        if n_components <= min(n_samples, n_features):
            init = "nndsvda"
        else:
            # 否则使用随机初始化
            init = "random"

    # 随机初始化
    if init == "random":
        # 计算平均值作为初始化的标准差
        avg = np.sqrt(X.mean() / n_components)
        # 检查随机状态并创建随机数生成器
        rng = check_random_state(random_state)
        # 使用标准正态分布生成 W 和 H
        H = avg * rng.standard_normal(size=(n_components, n_features)).astype(
            X.dtype, copy=False
        )
        W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(
            X.dtype, copy=False
        )
        # 将 W 和 H 中小于 eps 的元素设为 0
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # 使用 NNDSVD 初始化
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # 第一个奇异三元组是非负的，可以直接用于初始化
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    # 对于剩余的每个成分进行处理
    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # 提取列向量的正负部分
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # 计算它们的范数
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # 选择更新
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    # 将小于 eps 的元素设为 0
    W[W < eps] = 0
    H[H < eps] = 0

    # 根据不同的初始化方法进行后处理
    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        # 将 W 和 H 中值为 0 的元素设为均值 avg
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        # 使用均值和随机数生成器填充 W 和 H 中值为 0 的元素
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        # 抛出值错误，表示初始化参数不合法
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H
    """Helper function for _fit_coordinate_descent.

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...).

    """
    # 获取矩阵 Ht 的列数，即成分的数量
    n_components = Ht.shape[1]

    # 计算矩阵 Ht 的转置与自身的乘积
    HHt = np.dot(Ht.T, Ht)
    
    # 计算矩阵 X 与矩阵 Ht 的乘积，并确保结果是稀疏矩阵乘积
    XHt = safe_sparse_dot(X, Ht)

    # 如果有 L2 正则化参数，则将其加到 HHt 的对角线上
    if l2_reg != 0.0:
        # 仅在对角线上加入 l2_reg
        HHt.flat[:: n_components + 1] += l2_reg
        
    # 如果有 L1 正则化参数，则从 XHt 的每个元素中减去 l1_reg
    if l1_reg != 0.0:
        XHt -= l1_reg

    # 如果需要打乱顺序，则根据 random_state 打乱排列
    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    
    # 下面的操作似乎在 64 位 Windows 上的 Python 3.5 中是必需的。
    # 将排列转换为 np.intp 类型的数组
    permutation = np.asarray(permutation, dtype=np.intp)
    
    # 调用 _update_cdnmf_fast 函数进行快速更新，并返回结果
    return _update_cdnmf_fast(W, HHt, XHt, permutation)
    # W: 形状为 (n_samples, n_components) 的 ndarray，非负最小二乘问题的解决方案。
    # H: 形状为 (n_components, n_features) 的 ndarray，非负最小二乘问题的解决方案。
    # n_iter: 整数，算法执行的迭代次数。

    # 引用文献
    # ----------
    # .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
    #    factorizations" <10.1587/transfun.E92.A.708>`
    #    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
    #    of electronics, communications and computer sciences 92.3: 708-721, 2009.
    """
    # 确保 Ht 和 W 在内存中都是按照 C 顺序存储
    Ht = check_array(H.T, order="C")
    # 检查并接受稀疏矩阵 X 的输入
    X = check_array(X, accept_sparse="csr")

    # 检查随机数生成器的状态
    rng = check_random_state(random_state)

    # 迭代更新过程
    for n_iter in range(1, max_iter + 1):
        violation = 0.0

        # 更新 W
        violation += _update_coordinate_descent(
            X, W, Ht, l1_reg_W, l2_reg_W, shuffle, rng
        )
        # 更新 H
        if update_H:
            violation += _update_coordinate_descent(
                X.T, Ht, W, l1_reg_H, l2_reg_H, shuffle, rng
            )

        # 第一次迭代时记录违反度初始化值
        if n_iter == 1:
            violation_init = violation

        # 如果初始违反度为零，直接跳出循环
        if violation_init == 0:
            break

        # 如果设置了详细输出，打印当前违反度与初始违反度的比值
        if verbose:
            print("violation:", violation / violation_init)

        # 如果当前违反度相对于初始违反度的比值小于等于容忍度 tol，则认为收敛
        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    # 返回 W, Ht 的转置以及迭代次数 n_iter
    return W, Ht.T, n_iter
def _multiplicative_update_w(
    X,
    W,
    H,
    beta_loss,
    l1_reg_W,
    l2_reg_W,
    gamma,
    H_sum=None,
    HHt=None,
    XHt=None,
    update_H=True,
):
    """Update W in Multiplicative Update NMF."""
    # 根据 Multiplicative Update NMF 更新 W
    if beta_loss == 2:
        # 如果 beta_loss 为 2
        # 计算分子
        if XHt is None:
            XHt = safe_sparse_dot(X, H.T)
        if update_H:
            # 避免复制 XHt，因为将重新计算（update_H=True）
            numerator = XHt
        else:
            # 保留 XHt，因为不会重新计算（update_H=False）
            numerator = XHt.copy()

        # 计算分母
        if HHt is None:
            HHt = np.dot(H, H.T)
        denominator = np.dot(W, HHt)

    else:
        # 如果 beta_loss 不为 2
        # 计算分子
        # 如果 X 是稀疏的，在 X 非零的地方计算 WH
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # 在分母中使用的复制
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH < EPSILON] = EPSILON

        # 避免对零取负幂
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # 加快计算速度
            # 参考 /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # 逐元素乘法
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # 逐元素乘法
            WH_safe_X_data *= X_data

        # 这里分子 = 点乘（X * （点乘（W，H）**（beta_loss - 2）），H.T）
        numerator = safe_sparse_dot(WH_safe_X, H.T)

        # 计算分母
        if beta_loss == 1:
            if H_sum is None:
                H_sum = np.sum(H, axis=1)  # shape(n_components, )
            denominator = H_sum[np.newaxis, :]

        else:
            # 计算 WHHt = 点乘（点乘（W，H）**（beta_loss - 1），H.T）
            if sp.issparse(X):
                # 内存高效计算
                # （逐行计算，避免稠密矩阵 WH）
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H)
                    if beta_loss - 1 < 0:
                        WHi[WHi < EPSILON] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(WHi, H.T)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(WH, H.T)
            denominator = WHHt

    # 添加 L1 和 L2 正则化
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    # 将分母中为零的元素替换为一个非零小值 EPSILON，避免除零错误
    denominator[denominator == 0] = EPSILON

    # 对分子进行除法操作，得到更新的 delta_W
    numerator /= denominator
    delta_W = numerator

    # 如果 gamma 不等于 1，则对 delta_W 进行幂运算
    # gamma 的取值范围为 (0, 1]，表示进行小于 1 的幂运算
    if gamma != 1:
        delta_W **= gamma

    # 将 W 更新为 W * delta_W，完成权重更新
    W *= delta_W

    # 返回更新后的 W、H_sum、HHt 和 XHt
    return W, H_sum, HHt, XHt
# 在 Multiplicative Update NMF 中更新矩阵 H
def _multiplicative_update_h(
    X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma, A=None, B=None, rho=None
):
    """update H in Multiplicative Update NMF."""
    if beta_loss == 2:
        # 计算分子
        numerator = safe_sparse_dot(W.T, X)
        # 计算分母
        denominator = np.linalg.multi_dot([W.T, W, H])

    else:
        # Numerator
        # 计算 WH_safe_X = dot(W, H) * X，用于处理稀疏矩阵
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # 在 Denominator 中使用的拷贝
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH < EPSILON] = EPSILON

        # 避免除以零
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON

        if beta_loss == 1:
            # 计算 element-wise 的除法 X / WH_safe_X
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # 优化计算时间，参考 numpy 问题报告
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise 的乘法
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise 的乘法
            WH_safe_X_data *= X_data

        # 计算分子，即 dot(W.T, (dot(W, H) ** (beta_loss - 2)) * X)
        numerator = safe_sparse_dot(W.T, WH_safe_X)

        # Denominator
        if beta_loss == 1:
            # 计算 W 按列求和，形状为 (n_components, )
            W_sum = np.sum(W, axis=0)
            W_sum[W_sum == 0] = 1.0
            denominator = W_sum[:, np.newaxis]

        else:
            # 当 beta_loss 不在 (1, 2) 范围内时
            if sp.issparse(X):
                # 内存高效的计算方式，逐列计算，避免生成稠密矩阵 WH
                WtWH = np.empty(H.shape)
                for i in range(X.shape[1]):
                    WHi = np.dot(W, H[:, i])
                    if beta_loss - 1 < 0:
                        WHi[WHi < EPSILON] = EPSILON
                    WHi **= beta_loss - 1
                    WtWH[:, i] = np.dot(W.T, WHi)
            else:
                WH **= beta_loss - 1
                WtWH = np.dot(W.T, WH)
            denominator = WtWH

    # 添加 L1 和 L2 正则化
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = EPSILON

    if A is not None and B is not None:
        # 在线 NMF 的更新
        if gamma != 1:
            H **= 1 / gamma
        numerator *= H
        A *= rho
        B *= rho
        A += numerator
        B += denominator
        H = A / B

        if gamma != 1:
            H **= gamma
    # 如果条件不成立，计算 delta_H 为分子的值
    else:
        delta_H = numerator
        # 计算 delta_H 除以分母的结果
        delta_H /= denominator
        # 如果 gamma 不等于 1，对 delta_H 进行 gamma 次幂运算
        if gamma != 1:
            delta_H **= gamma
        # 将 H 乘以 delta_H 的值
        H *= delta_H

    # 返回计算结果 H
    return H
    start_time = time.time()
    # 记录算法开始执行的时间戳

    beta_loss = _beta_loss_to_float(beta_loss)
    # 将 beta_loss 转换为浮点数，以便后续计算使用

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    # 根据 beta_loss 计算 gamma，用于最大化-最小化算法（MM 算法）[Fevotte 2011]
    if beta_loss < 1:
        gamma = 1.0 / (2.0 - beta_loss)
    elif beta_loss > 2:
        gamma = 1.0 / (beta_loss - 1.0)
    else:
        gamma = 1.0

    # used for the convergence criterion
    # 用于收敛准则的参数设定
    # 计算初始时刻的误差
    error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
    # 保存初始误差值，用于后续比较收敛情况
    previous_error = error_at_init

    # 初始化变量，用于存储中间结果
    H_sum, HHt, XHt = None, None, None
    # 迭代更新参数 W 和 H
    for n_iter in range(1, max_iter + 1):
        # 更新 W
        # 如果不更新 H，则重复使用已保存的 H_sum、HHt 和 XHt
        W, H_sum, HHt, XHt = _multiplicative_update_w(
            X,
            W,
            H,
            beta_loss=beta_loss,
            l1_reg_W=l1_reg_W,
            l2_reg_W=l2_reg_W,
            gamma=gamma,
            H_sum=H_sum,
            HHt=HHt,
            XHt=XHt,
            update_H=update_H,
        )

        # 当 beta_loss < 1 时，为保证稳定性，将小于机器精度的 W 设置为 0
        if beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.0

        # 更新 H（仅在 fit 或 fit_transform 时更新）
        if update_H:
            H = _multiplicative_update_h(
                X,
                W,
                H,
                beta_loss=beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=gamma,
            )

            # 由于 H 发生了变化，需要重新计算 H_sum、HHt 和 XHt
            H_sum, HHt, XHt = None, None, None

            # 当 beta_loss <= 1 时，为保证稳定性，将小于机器精度的 H 设置为 0
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.0

        # 每 10 次迭代测试收敛标准
        if tol > 0 and n_iter % 10 == 0:
            # 计算当前误差
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            # 若 verbose 为真，则打印当前迭代信息
            if verbose:
                iter_time = time.time()
                print(
                    "Epoch %02d reached after %.3f seconds, error: %f"
                    % (n_iter, iter_time - start_time, error)
                )

            # 判断是否满足收敛条件
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # 若 verbose 为真且不在收敛测试打印的情况下，打印最终迭代信息
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print(
            "Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time)
        )

    # 返回计算得到的 W、H 和迭代次数
    return W, H, n_iter
# 使用装饰器定义函数，并验证参数类型和允许的取值范围
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数 X 应为数组或稀疏矩阵
        "W": ["array-like", None],  # 参数 W 应为数组或 None
        "H": ["array-like", None],  # 参数 H 应为数组或 None
        "update_H": ["boolean"],  # 参数 update_H 应为布尔值
    },
    prefer_skip_nested_validation=False,  # 不偏向跳过嵌套验证
)
def non_negative_factorization(
    X,  # 输入矩阵 X，形状为 (n_samples, n_features)
    W=None,  # 因子矩阵 W，默认为 None
    H=None,  # 因子矩阵 H，默认为 None
    n_components="warn",  # 组件数量，默认为 "warn"
    *,  # 以下参数为关键字参数
    init=None,  # 初始化方法，默认为 None
    update_H=True,  # 是否更新 H，默认为 True
    solver="cd",  # 求解方法，默认为 "cd"
    beta_loss="frobenius",  # 损失函数类型，默认为 "frobenius"
    tol=1e-4,  # 迭代收敛容忍度，默认为 1e-4
    max_iter=200,  # 最大迭代次数，默认为 200
    alpha_W=0.0,  # W 的正则化参数，默认为 0.0
    alpha_H="same",  # H 的正则化参数，默认为 "same"
    l1_ratio=0.0,  # L1 正则化比例，默认为 0.0
    random_state=None,  # 随机数种子，默认为 None
    verbose=0,  # 控制详细程度，默认为 0
    shuffle=False,  # 是否打乱数据，默认为 False
):
    """Compute Non-negative Matrix Factorization (NMF).

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is:

        .. math::

            L(W, H) &= 0.5 * ||X - WH||_{loss}^2

            &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

            &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

            &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

            &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The regularization terms are scaled by `n_features` for `W` and by `n_samples` for
    `H` to keep their impact balanced with respect to one another and to the data fit
    term as independent as possible of the size `n_samples` of the training set.

    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.

    Note that the transformed data is named W and the components matrix is named H. In
    the NMF literature, the naming convention is usually the opposite since the data
    matrix X is transposed.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components), default=None
        If `init='custom'`, it is used as initial guess for the solution.
        If `update_H=False`, it is initialised as an array of zeros, unless
        `solver='mu'`, then it is filled with values calculated by
        `np.sqrt(X.mean() / self._n_components)`.
        If `None`, uses the initialisation method specified in `init`.
    H : array-like of shape (n_components, n_features), default=None
        # 如果 `init='custom'`，将此参数作为解的初始猜测
        # 如果 `update_H=False`，此参数作为常量，仅解决 W 的值
        # 如果为 `None`，则使用 `init` 参数指定的初始化方法

    n_components : int or {'auto'} or None, default=None
        # 成分的数量，如果未设置 `n_components`，则保留所有特征
        # 如果 `n_components='auto'`，则从 `W` 或 `H` 的形状自动推断出成分数量
        # .. versionchanged:: 1.4
        #    添加了 `'auto'` 的值

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        # 初始化过程使用的方法

        # 有效选项:
        # - None: 如果 `n_components < n_features`，则为 'nndsvda'，否则为 'random'
        # - 'random': 非负随机矩阵，按比例缩放为 `sqrt(X.mean() / n_components)`
        # - 'nndsvd': 非负双奇异值分解（NNDSVD）初始化（适合稀疏性）
        # - 'nndsvda': NNDSVD，使用 X 的平均值填充零元素（不希望稀疏时更好）
        # - 'nndsvdar': NNDSVD，使用小随机值填充零元素（通常比 NNDSVDa 更快，但不太准确）
        # - 'custom': 如果 `update_H=True`，则使用自定义矩阵 W 和 H，两者都必须提供；
        #            如果 `update_H=False`，则仅使用自定义矩阵 H

        # .. versionchanged:: 0.23
        #    在 0.23 版本中，默认值从 'random' 改为 None

        # .. versionchanged:: 1.1
        #    当 `init=None` 且 `n_components` 小于 `n_samples` 和 `n_features` 时，默认为 `nndsvda`

    update_H : bool, default=True
        # 设置为 True，将从初始猜测中估计出 W 和 H
        # 设置为 False，仅估计 W

    solver : {'cd', 'mu'}, default='cd'
        # 使用的数值求解器：

        # - 'cd' 是坐标下降求解器，使用快速分层交替最小二乘（Fast HALS）
        # - 'mu' 是乘法更新求解器

        # .. versionadded:: 0.17
        #    坐标下降求解器添加于版本 0.17

        # .. versionadded:: 0.19
        #    乘法更新求解器添加于版本 0.19

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        # 要最小化的 beta 散度，衡量 X 和 dot product WH 之间的距离
        # 注意，与 'frobenius'（或 2）和 'kullback-leibler'（或 1）不同的值会导致显著较慢的拟合速度
        # 注意，对于 beta_loss <= 0（或 'itakura-saito'），输入矩阵 X 不能包含零
        # 仅在 'mu' 求解器中使用

        # .. versionadded:: 0.19

    tol : float, default=1e-4
        # 停止条件的容差
    # 最大迭代次数，默认为200
    max_iter : int, default=200
        Maximum number of iterations before timing out.

    # W 的正则化常数，默认为0.0，设置为0以取消对 W 的正则化
    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

        .. versionadded:: 1.0

    # H 的正则化常数，默认为 "same"，如果为 "same"，则与 alpha_W 相同
    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

        .. versionadded:: 1.0

    # 正则化混合参数，0 <= l1_ratio <= 1
    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    # 随机数种子或随机状态实例，用于初始化和求解过程的随机性
    random_state : int, RandomState instance or None, default=None
        Used for NMF initialization (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    # 冗余度级别，控制输出详细程度
    verbose : int, default=0
        The verbosity level.

    # 在 CD solver 中是否对坐标顺序进行随机化
    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

    # 返回结果：
    # W：形状为 (n_samples, n_components) 的 ndarray
    # 非负最小二乘问题的解
    #
    # H：形状为 (n_components, n_features) 的 ndarray
    # 非负最小二乘问题的解
    #
    # n_iter：实际迭代次数
    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        Actual number of iterations.

    # 参考文献：
    #
    # [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
    #    factorizations" <10.1587/transfun.E92.A.708>`
    #    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
    #    of electronics, communications and computer sciences 92.3: 708-721, 2009.
    #
    # [2] :doi:`"Algorithms for nonnegative matrix factorization with the
    #    beta-divergence" <10.1162/NECO_a_00168>`
    #    Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    # 示例：
    #
    # >>> import numpy as np
    # >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    # >>> from sklearn.decomposition import non_negative_factorization
    # >>> W, H, n_iter = non_negative_factorization(
    # ...     X, n_components=2, init='random', random_state=0)
    est = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        beta_loss=beta_loss,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        verbose=verbose,
        shuffle=shuffle,
    )
    # 验证 NMF 模型参数的有效性
    est._validate_params()

    # 检查输入数组 X，接受稀疏格式 "csr" 或 "csc"，数据类型为 np.float64 或 np.float32
    X = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])
    # 使用配置上下文，设置 assume_finite 为 True，确保在处理数据时假设数据为有限的
    with config_context(assume_finite=True):
        # 调用 est 对象的 _fit_transform 方法，进行模型拟合和转换操作
        # X 是输入的数据，W 和 H 是传入的参数，update_H 是控制是否更新 H 的布尔值
        # 返回 W（拟合后的 W 矩阵）、H（拟合后的 H 矩阵）、n_iter（迭代次数）
        W, H, n_iter = est._fit_transform(X, W=W, H=H, update_H=update_H)
    
    # 返回拟合后的 W 矩阵、H 矩阵以及迭代次数 n_iter
    return W, H, n_iter
# 类 _BaseNMF 继承了 ClassNamePrefixFeaturesOutMixin、TransformerMixin、BaseEstimator 和 ABC
class _BaseNMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, ABC):
    """Base class for NMF and MiniBatchNMF."""

    # 防止在 inverse_transform 方法中针对非标准的 Xt 参数生成 set_split_inverse_transform 方法
    # TODO(1.7): 在 v1.7 版本中删除 Xt 参数时移除此处
    __metadata_request__inverse_transform = {"Xt": metadata_routing.UNUSED}

    # 参数约束字典，定义了每个参数的类型和可选值范围
    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
            StrOptions({"auto"}),
            Hidden(StrOptions({"warn"})),
        ],
        "init": [
            StrOptions({"random", "nndsvd", "nndsvda", "nndsvdar", "custom"}),
            None,
        ],
        "beta_loss": [
            StrOptions({"frobenius", "kullback-leibler", "itakura-saito"}),
            Real,
        ],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "alpha_W": [Interval(Real, 0, None, closed="left")],
        "alpha_H": [Interval(Real, 0, None, closed="left"), StrOptions({"same"})],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "verbose": ["verbose"],
    }

    # 初始化方法，设置各个参数的初始值
    def __init__(
        self,
        n_components="warn",
        *,
        init=None,
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
    ):
        self.n_components = n_components
        self.init = init
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose

    # 检查参数方法，根据给定的数据 X 进行参数验证和设置
    def _check_params(self, X):
        # 复制 n_components 参数，若其为 "warn" 则发出警告并设为 None（保留旧的默认值）
        self._n_components = self.n_components
        if self.n_components == "warn":
            warnings.warn(
                (
                    "The default value of `n_components` will change from `None` to"
                    " `'auto'` in 1.6. Set the value of `n_components` to `None`"
                    " explicitly to suppress the warning."
                ),
                FutureWarning,
            )
            self._n_components = None  # 保留旧的默认值
        if self._n_components is None:
            self._n_components = X.shape[1]

        # 将 beta_loss 参数转换为浮点数
        self._beta_loss = _beta_loss_to_float(self.beta_loss)
    def _check_w_h(self, X, W, H, update_H):
        """Check W and H, or initialize them."""
        # 获取样本数和特征数
        n_samples, n_features = X.shape

        if self.init == "custom" and update_H:
            # 如果初始化方式为自定义且需要更新H
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, self._n_components), "NMF (input W)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            # 检查H和W的数据类型是否与X相同
            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError(
                    "H and W should have the same dtype as X. Got "
                    "H.dtype = {} and W.dtype = {}.".format(H.dtype, W.dtype)
                )

        elif not update_H:
            # 如果不需要更新H
            if W is not None:
                # 发出警告，因为当update_H=False时，提供的初始W未被使用
                warnings.warn(
                    "When update_H=False, the provided initial W is not used.",
                    RuntimeWarning,
                )

            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            # 检查H的数据类型是否与X相同
            if H.dtype != X.dtype:
                raise TypeError(
                    "H should have the same dtype as X. Got H.dtype = {}.".format(
                        H.dtype
                    )
                )

            # 如果solver为'mu'，则不应该用零来初始化W
            if self.solver == "mu":
                avg = np.sqrt(X.mean() / self._n_components)
                W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
            else:
                W = np.zeros((n_samples, self._n_components), dtype=X.dtype)

        else:
            # 如果init不为'custom'
            if W is not None or H is not None:
                # 发出警告，因为当init!='custom'时，提供的W或H将被忽略。设置init='custom'以使用它们作为初始化
                warnings.warn(
                    (
                        "When init!='custom', provided W or H are ignored. Set "
                        " init='custom' to use them as initialization."
                    ),
                    RuntimeWarning,
                )

            if self._n_components == "auto":
                self._n_components = X.shape[1]

            # 初始化W和H
            W, H = _initialize_nmf(
                X, self._n_components, init=self.init, random_state=self.random_state
            )

        return W, H

    def _compute_regularization(self, X):
        """Compute scaled regularization terms."""
        # 获取样本数和特征数
        n_samples, n_features = X.shape
        alpha_W = self.alpha_W
        alpha_H = self.alpha_W if self.alpha_H == "same" else self.alpha_H

        # 计算正则化项
        l1_reg_W = n_features * alpha_W * self.l1_ratio
        l1_reg_H = n_samples * alpha_H * self.l1_ratio
        l2_reg_W = n_features * alpha_W * (1.0 - self.l1_ratio)
        l2_reg_H = n_samples * alpha_H * (1.0 - self.l1_ratio)

        return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H
    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        **params : kwargs
            Parameters (keyword arguments) and values passed to
            the fit_transform instance.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 参数验证在 fit_transform 方法中完成

        # 调用 fit_transform 方法来训练模型
        self.fit_transform(X, **params)
        return self

    def inverse_transform(self, X=None, *, Xt=None):
        """Transform data back to its original space.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.

        Xt : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.

            .. deprecated:: 1.5
                `Xt` was deprecated in 1.5 and will be removed in 1.7. Use `X` instead.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Returns a data matrix of the original shape.
        """

        # 将传入的参数 X 和 Xt 统一处理，Xt 已被弃用，将在未来版本中移除
        X = _deprecate_Xt_in_inverse_transform(X, Xt)

        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 返回反向转换后的数据，使用模型的 components_ 属性
        return X @ self.components_

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # 返回转换后输出特征的数量，即模型的 components_ 的行数
        return self.components_.shape[0]

    def _more_tags(self):
        # 返回更多的标签信息，表明模型对输入数据要求为正数，同时保留数据类型为 np.float64 或 np.float32
        return {
            "requires_positive_X": True,
            "preserves_dtype": [np.float64, np.float32],
        }
# 定义 NMF 类，继承自 _BaseNMF 类，实现非负矩阵分解（NMF）算法
class NMF(_BaseNMF):
    """Non-Negative Matrix Factorization (NMF).

    Find two non-negative matrices, i.e. matrices with all non-negative elements, (W, H)
    whose product approximates the non-negative matrix X. This factorization can be used
    for example for dimensionality reduction, source separation or topic extraction.

    The objective function is:

        .. math::

            L(W, H) &= 0.5 * ||X - WH||_{loss}^2

            &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

            &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

            &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

            &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The regularization terms are scaled by `n_features` for `W` and by `n_samples` for
    `H` to keep their impact balanced with respect to one another and to the data fit
    term as independent as possible of the size `n_samples` of the training set.

    The objective function is minimized with an alternating minimization of W
    and H.

    Note that the transformed data is named W and the components matrix is named H. In
    the NMF literature, the naming convention is usually the opposite since the data
    matrix X is transposed.

    Read more in the :ref:`User Guide <NMF>`.

    Parameters
    ----------
    n_components : int or {'auto'} or None, default=None
        Number of components, if n_components is not set all features
        are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from W or H shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.
    # 初始化方法，用于指定NMF（非负矩阵分解）过程的初始化方式。
    # 可选项包括：'None'、'random'、'nndsvd'、'nndsvda'、'nndsvdar'、'custom'。
    # 
    # - 'None': 如果 n_components <= min(n_samples, n_features)，则使用'nndsvda'，否则随机初始化。
    # 
    # - 'random': 非负随机矩阵，缩放因子为 sqrt(X.mean() / n_components)。
    # 
    # - 'nndsvd': 非负双重奇异值分解（NNDSVD）初始化（适用于稀疏性较高的情况）。
    # 
    # - 'nndsvda': 带有用X的平均值填充零值的NNDSVD（在不希望稀疏性的情况下更好）。
    # 
    # - 'nndsvdar': 带有小随机值填充零值的NNDSVD（通常比NNDSVDA更快，但不太精确）。
    # 
    # - 'custom': 使用自定义矩阵W和H（必须同时提供）。
    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
    
    # 数值求解器选择，用于指定NMF问题的求解方法。
    # 可选项包括：
    # 
    # - 'cd': 坐标下降（Coordinate Descent）求解器。
    # 
    # - 'mu': 乘法更新（Multiplicative Update）求解器。
    solver : {'cd', 'mu'}, default='cd'
    
    # 要最小化的Beta散度，衡量X和WH的点积之间的距离。
    # 注意，与'frobenius'（或2）和'kullback-leibler'（或1）不同的值会导致拟合速度显著减慢。
    # 当beta_loss <= 0（或'itakura-saito'）时，输入矩阵X不能包含零。
    # 仅在'solver'为'mu'时使用。
    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
    
    # 停止条件的容差值。
    tol : float, default=1e-4
    
    # 最大迭代次数，在超时之前。
    max_iter : int, default=200
    
    # 随机种子，用于初始化（当“init”为'nndsvdar'或'random'时）和在坐标下降中使用。
    # 传递整数以实现在多次函数调用中可复现的结果。
    random_state : int, RandomState instance or None, default=None
    
    # 常数，用于调整W的正则化项。设置为零（默认）表示不对W进行正则化。
    alpha_W : float, default=0.0
    
    # 常数，用于调整H的正则化项。设置为零表示不对H进行正则化。
    # 如果设置为"same"（默认），则与alpha_W取相同的值。
    alpha_H : float or "same", default="same"
    l1_ratio : float, default=0.0
        正则化混合参数，取值范围为 0 <= l1_ratio <= 1。
        当 l1_ratio = 0 时，使用逐元素 L2 正则化（即 Frobenius 范数）。
        当 l1_ratio = 1 时，使用逐元素 L1 正则化。
        当 0 < l1_ratio < 1 时，使用 L1 和 L2 的组合正则化。

        .. versionadded:: 0.17
           在坐标下降求解器中使用的正则化参数 *l1_ratio*。

    verbose : int, default=0
        是否详细输出信息。

    shuffle : bool, default=False
        若为 True，则在坐标下降求解器中随机排列坐标的顺序。

        .. versionadded:: 0.17
           坐标下降求解器中使用的 *shuffle* 参数。

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        因子化矩阵，有时称为“字典”。

    n_components_ : int
        组件的数量。如果给定了 `n_components` 参数，则与之相同。否则，与特征的数量相同。

    reconstruction_err_ : float
        训练数据 ``X`` 与拟合模型重建数据 ``WH`` 之间的 Frobenius 范数或 beta-divergence。

    n_iter_ : int
        实际迭代次数。

    n_features_in_ : int
        在拟合过程中看到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在拟合过程中看到的特征名称。仅当 `X` 的特征名称都是字符串时定义。

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : 寻找稀疏编码数据的字典。
    MiniBatchSparsePCA : 小批量稀疏主成分分析。
    PCA : 主成分分析。
    SparseCoder : 从固定的、预计算的字典中找到数据的稀疏表示。
    SparsePCA : 稀疏主成分分析。
    TruncatedSVD : 使用截断奇异值分解进行降维。

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    # 定义参数约束字典，继承自基类 _BaseNMF 的参数约束，并添加额外约束
    _parameter_constraints: dict = {
        **_BaseNMF._parameter_constraints,
        "solver": [StrOptions({"mu", "cd"})],  # solver 参数可选取值为 "mu" 或 "cd"
        "shuffle": ["boolean"],  # shuffle 参数必须是布尔类型
    }

    # 初始化函数，设置 NMF 模型的各项参数
    def __init__(
        self,
        n_components="warn",
        *,
        init=None,
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    ):
        # 调用父类 _BaseNMF 的初始化方法，设置公共参数
        super().__init__(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
        )

        # 设置本类特有的参数 solver 和 shuffle
        self.solver = solver  # 设置求解器类型
        self.shuffle = shuffle  # 设置是否打乱数据顺序

    # 参数检查函数，验证参数是否符合要求
    def _check_params(self, X):
        # 调用父类方法检查参数
        super()._check_params(X)

        # 检查 solver 参数是否支持当前的 beta_loss 参数设置
        if self.solver != "mu" and self.beta_loss not in (2, "frobenius"):
            # 如果 solver 不是 "mu"，并且 beta_loss 不是 2 或 "frobenius"，则抛出数值错误
            raise ValueError(
                f"Invalid beta_loss parameter: solver {self.solver!r} does not handle "
                f"beta_loss = {self.beta_loss!r}"
            )

        # 如果 solver 是 "mu" 并且 init 是 "nndsvd"，发出警告
        if self.solver == "mu" and self.init == "nndsvd":
            warnings.warn(
                (
                    "The multiplicative update ('mu') solver cannot update "
                    "zeros present in the initialization, and so leads to "
                    "poorer results when used jointly with init='nndsvd'. "
                    "You may try init='nndsvda' or init='nndsvdar' instead."
                ),
                UserWarning,
            )

        return self

    # 使用装饰器 @_fit_context 装饰的方法，用于模型拟合上下文
    @_fit_context(prefer_skip_nested_validation=True)
    # 学习一个 NMF 模型以适应数据 X，并返回转换后的数据
    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # 校验并接受稀疏矩阵输入，将数据 X 转换为适合处理的格式（浮点数类型）
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        # 在假定数据为有限的上下文中执行以下代码块
        with config_context(assume_finite=True):
            # 调用内部方法 _fit_transform 进行模型拟合和转换
            W, H, n_iter = self._fit_transform(X, W=W, H=H)

        # 使用 beta 散度计算重构误差，用于评估模型拟合的质量
        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True
        )

        # 记录模型的成分数、成分矩阵 H 和迭代次数
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter

        # 返回转换后的数据 W
        return W

    # 根据已拟合的 NMF 模型，对数据 X 进行转换
    def transform(self, X):
        """Transform the data X according to the fitted NMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # 检查模型是否已拟合，若未拟合则引发异常
        check_is_fitted(self)
        # 校验并接受稀疏矩阵输入，将数据 X 转换为适合处理的格式（浮点数类型），但不重置数据
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
        )

        # 在假定数据为有限的上下文中执行以下代码块
        with config_context(assume_finite=True):
            # 使用当前模型的成分矩阵 components_，对数据 X 进行转换
            W, *_ = self._fit_transform(X, H=self.components_, update_H=False)

        # 返回转换后的数据 W
        return W
class MiniBatchNMF(_BaseNMF):
    """Mini-Batch Non-Negative Matrix Factorization (NMF).

    .. versionadded:: 1.1

    Find two non-negative matrices, i.e. matrices with all non-negative elements,
    (`W`, `H`) whose product approximates the non-negative matrix `X`. This
    factorization can be used for example for dimensionality reduction, source
    separation or topic extraction.

    The objective function is:

        .. math::

            L(W, H) &= 0.5 * ||X - WH||_{loss}^2

            &+ alpha_W * l1_ratio * n_features * ||vec(W)||_1

            &+ alpha_H * l1_ratio * n_samples * ||vec(H)||_1

            &+ 0.5 * alpha_W * (1 - l1_ratio) * n_features * ||W||_{Fro}^2

            &+ 0.5 * alpha_H * (1 - l1_ratio) * n_samples * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The objective function is minimized with an alternating minimization of `W`
    and `H`.

    Note that the transformed data is named `W` and the components matrix is
    named `H`. In the NMF literature, the naming convention is usually the opposite
    since the data matrix `X` is transposed.

    Read more in the :ref:`User Guide <MiniBatchNMF>`.

    Parameters
    ----------
    n_components : int or {'auto'} or None, default=None
        Number of components, if `n_components` is not set all features
        are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from W or H shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - `None`: 'nndsvda' if `n_components <= min(n_samples, n_features)`,
          otherwise random.

        - `'random'`: non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness).

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired).

        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired).

        - `'custom'`: Use custom matrices `W` and `H` which must both be provided.

    batch_size : int, default=1024
        Number of samples in each mini-batch. Large batch sizes
        give better long-term convergence at the cost of a slower start.
    """
    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        # Beta divergence to be minimized, measuring the distance between `X`
        # and the dot product `WH`. Different choices affect convergence speed.
        # Note: Use 'frobenius' for Euclidean distance and 'kullback-leibler'
        # for divergence based on information theory.

    tol : float, default=1e-4
        # Tolerance for early stopping based on changes in `H`. Set to 0.0 to disable
        # early stopping based on `H` updates.

    max_no_improvement : int, default=10
        # Maximum number of consecutive mini-batches without improvement in cost
        # to trigger early stopping. Set to None to disable cost-based convergence.

    max_iter : int, default=200
        # Maximum number of iterations over the dataset before stopping.

    alpha_W : float, default=0.0
        # Regularization strength multiplier for matrix `W`. Set to 0.0 for no
        # regularization on `W`.

    alpha_H : float or "same", default="same"
        # Regularization strength multiplier for matrix `H`. Set to 0.0 for no
        # regularization on `H`. If set to "same", it mirrors `alpha_W`.

    l1_ratio : float, default=0.0
        # Mixing parameter for L1 (Lasso) and L2 (Ridge) penalties in regularization.
        # 0.0 corresponds to pure L2 penalty (Frobenius norm), 1.0 corresponds to
        # pure L1 penalty, and values in between represent a mix of both.

    forget_factor : float, default=0.7
        # Factor determining how past information is weighted. Use 1.0 for finite datasets
        # and values less than 1.0 for online learning, where recent batches have more weight.

    fresh_restarts : bool, default=False
        # Whether to restart the optimization of `W` completely at each step. This can lead
        # to a better solution at the cost of increased computation.

    fresh_restarts_max_iter : int, default=30
        # Maximum number of iterations for solving `W` when `fresh_restarts` is enabled.
        # Early stopping based on small `W` changes controlled by `tol` may occur.

    transform_max_iter : int, default=None
        # Maximum number of iterations for solving `W` during transformation. Defaults
        # to `max_iter` if not specified.
    # random_state : int, RandomState instance or None, default=None
    # Used for initializing random number generator for reproducible results
    # during initialization (when `init` == 'nndsvdar' or 'random') and in Coordinate Descent.

    # verbose : bool, default=False
    # Whether to print progress messages to stdout.

    Attributes
    ----------
    # components_ : ndarray of shape (n_components, n_features)
    # Factorization matrix, also known as 'dictionary'.

    # n_components_ : int
    # The number of components. It equals the `n_components` parameter if specified,
    # otherwise, it matches the number of features.

    # reconstruction_err_ : float
    # Frobenius norm or beta-divergence between the training data `X`
    # and the reconstructed data `WH` from the fitted model.

    # n_iter_ : int
    # Actual number of iterations run over the entire dataset.

    # n_steps_ : int
    # Number of mini-batches processed.

    # n_features_in_ : int
    # Number of features seen during the fit.

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    # Names of features seen during the fit. Defined only if `X` has feature names
    # that are all strings.

    See Also
    --------
    # NMF : Non-negative matrix factorization.
    # MiniBatchDictionaryLearning : Finds a dictionary that can best be used to represent
    # data using a sparse code.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    .. [3] :doi:`"Online algorithms for nonnegative matrix factorization with the
       Itakura-Saito divergence" <10.1109/ASPAA.2011.6082314>`
       Lefevre, A., Bach, F., Fevotte, C. (2011). WASPA.

    Examples
    --------
    # >>> import numpy as np
    # >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    # >>> from sklearn.decomposition import MiniBatchNMF
    # >>> model = MiniBatchNMF(n_components=2, init='random', random_state=0)
    # >>> W = model.fit_transform(X)
    # >>> H = model.components_
    # 定义参数约束字典，继承自_BaseNMF类的参数约束
    _parameter_constraints: dict = {
        **_BaseNMF._parameter_constraints,  # 合并继承的参数约束
        "max_no_improvement": [Interval(Integral, 1, None, closed="left"), None],  # 设置max_no_improvement参数的约束条件
        "batch_size": [Interval(Integral, 1, None, closed="left")],  # 设置batch_size参数的约束条件
        "forget_factor": [Interval(Real, 0, 1, closed="both")],  # 设置forget_factor参数的约束条件
        "fresh_restarts": ["boolean"],  # fresh_restarts参数为布尔类型
        "fresh_restarts_max_iter": [Interval(Integral, 1, None, closed="left")],  # 设置fresh_restarts_max_iter参数的约束条件
        "transform_max_iter": [Interval(Integral, 1, None, closed="left"), None],  # 设置transform_max_iter参数的约束条件
    }

    # 初始化方法，设置各种参数
    def __init__(
        self,
        n_components="warn",
        *,
        init=None,
        batch_size=1024,
        beta_loss="frobenius",
        tol=1e-4,
        max_no_improvement=10,
        max_iter=200,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        forget_factor=0.7,
        fresh_restarts=False,
        fresh_restarts_max_iter=30,
        transform_max_iter=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            n_components=n_components,  # 初始化父类_BaseNMF的参数
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
        )

        # 将传入的参数赋值给对象的对应属性
        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter  # 可选参数，用于变换的最大迭代次数

    # 检查参数的有效性方法
    def _check_params(self, X):
        super()._check_params(X)  # 调用父类的参数检查方法

        # 计算实际的批处理大小，取传入的batch_size和数据集X的行数的较小值
        self._batch_size = min(self.batch_size, X.shape[0])

        # 计算忘记因子，根据传入的forget_factor和批处理大小计算
        self._rho = self.forget_factor ** (self._batch_size / X.shape[0])

        # 根据beta_loss的值计算gamma，用于特定算法（MM算法）
        if self._beta_loss < 1:
            self._gamma = 1.0 / (2.0 - self._beta_loss)
        elif self._beta_loss > 2:
            self._gamma = 1.0 / (self._beta_loss - 1.0)
        else:
            self._gamma = 1.0

        # 设置变换的最大迭代次数，如果transform_max_iter为None，则使用max_iter作为默认值
        self._transform_max_iter = (
            self.max_iter
            if self.transform_max_iter is None
            else self.transform_max_iter
        )

        return self  # 返回对象本身
    def _solve_W(self, X, H, max_iter):
        """Minimize the objective function w.r.t W.

        Update W with H being fixed, until convergence. This is the heart
        of `transform` but it's also used during `fit` when doing fresh restarts.
        """
        avg = np.sqrt(X.mean() / self._n_components)
        # 初始化 W 矩阵，填充平均值作为初始值
        W = np.full((X.shape[0], self._n_components), avg, dtype=X.dtype)
        # 创建 W 的备份
        W_buffer = W.copy()

        # 获取缩放后的正则化项，针对每个小批量进行计算以考虑小批量的不同大小
        l1_reg_W, _, l2_reg_W, _ = self._compute_regularization(X)

        # 迭代更新 W
        for _ in range(max_iter):
            # 使用乘法更新法更新 W
            W, *_ = _multiplicative_update_w(
                X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma
            )

            # 计算当前 W 与上一次迭代 W 的差异
            W_diff = linalg.norm(W - W_buffer) / linalg.norm(W)
            # 如果差异小于设定的容忍度 tol，则提前结束迭代
            if self.tol > 0 and W_diff <= self.tol:
                break

            # 更新 W_buffer 为当前 W 的值
            W_buffer[:] = W

        return W

    def _minibatch_step(self, X, W, H, update_H):
        """Perform the update of W and H for one minibatch."""
        batch_size = X.shape[0]

        # 获取缩放后的正则化项，针对每个小批量进行计算以考虑小批量的不同大小
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)

        # 更新 W
        if self.fresh_restarts or W is None:
            # 如果是新的开始或者 W 为空，则调用 _solve_W 函数重新计算 W
            W = self._solve_W(X, H, self.fresh_restarts_max_iter)
        else:
            # 否则使用乘法更新法更新 W
            W, *_ = _multiplicative_update_w(
                X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma
            )

        # 当 beta_loss 小于 1 时，为了稳定性进行必要的调整
        if self._beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.0

        # 计算当前批量的成本
        batch_cost = (
            _beta_divergence(X, W, H, self._beta_loss)
            + l1_reg_W * W.sum()
            + l1_reg_H * H.sum()
            + l2_reg_W * (W**2).sum()
            + l2_reg_H * (H**2).sum()
        ) / batch_size

        # 更新 H（仅在 fit 或 fit_transform 时）
        if update_H:
            # 使用乘法更新法更新 H
            H[:] = _multiplicative_update_h(
                X,
                W,
                H,
                beta_loss=self._beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=self._gamma,
                A=self._components_numerator,
                B=self._components_denominator,
                rho=self._rho,
            )

            # 当 beta_loss 小于等于 1 时，为了稳定性进行必要的调整
            if self._beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.0

        return batch_cost

    def _minibatch_convergence(
        self, X, batch_cost, H, H_buffer, n_samples, step, n_steps
    ):
        # This function is not fully provided in the snippet; thus, it won't be commented.
    ):
        """
        Helper function to encapsulate the early stopping logic.
        This function checks for convergence criteria during training iterations.

        Parameters:
        - X: Input data matrix.
        - H: Current value of the parameter matrix H.
        - H_buffer: Buffer storing the previous value of H for comparison.
        - n_steps: Total number of training steps.
        - batch_cost: Cost associated with the current minibatch.
        - n_samples: Number of samples in the dataset.
        """
        batch_size = X.shape[0]

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because H is not updated yet.
        if step == 1:
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}")
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )

        # Early stopping based on change of H
        H_diff = linalg.norm(H - H_buffer) / linalg.norm(H)
        if self.tol > 0 and H_diff <= self.tol:
            if self.verbose:
                print(f"Converged (small H change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True

        return False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored
            Not used, present here for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialization method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialization method specified in `init`.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # Validate the input data X, ensuring it's compatible with expected formats
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        # Fit the NMF model to the input data X and return the decomposed matrices W and H
        with config_context(assume_finite=True):
            W, H, n_iter, n_steps = self._fit_transform(X, W=W, H=H)

        # Calculate and store the reconstruction error based on the decomposed matrices
        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True
        )

        # Store the number of components, the components matrix, and iteration information
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        self.n_steps_ = n_steps

        # Return the transformed data matrix W
        return W

    def transform(self, X):
        """Transform the data X according to the fitted MiniBatchNMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # Ensure that the model has been fitted by checking if necessary attributes exist
        check_is_fitted(self)

        # Validate the input data X, ensuring it's compatible with expected formats
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
        )

        # Solve for the transformed data matrix W using the fitted components matrix
        W = self._solve_W(X, self.components_, self._transform_max_iter)

        # Return the transformed data matrix W
        return W

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, W=None, H=None):
        """
        Update the model using the data in `X` as a mini-batch.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once (see :ref:`scaling_strategies`).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored
            Not used, present here for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            Only used for the first call to `partial_fit`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            Only used for the first call to `partial_fit`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check if the model has already been fitted and initialized
        has_components = hasattr(self, "components_")

        # Validate input data `X`, ensuring it meets requirements
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=not has_components,
        )

        if not has_components:
            # Initialize model parameters if not already fitted
            self._check_params(X)
            _, H = self._check_w_h(X, W=W, H=H, update_H=True)

            # Initialize numerator and denominator components
            self._components_numerator = H.copy()
            self._components_denominator = np.ones(H.shape, dtype=H.dtype)
            self.n_steps_ = 0
        else:
            # Retrieve current components if already fitted
            H = self.components_

        # Perform a step of the mini-batch update
        self._minibatch_step(X, None, H, update_H=True)

        # Update attributes related to the current state
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_steps_ += 1

        return self
```