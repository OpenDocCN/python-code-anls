# `D:\src\scipysrc\scikit-learn\sklearn\utils\sparsefuncs_fast.pyx`

```
"""Utilities to work with sparse matrices and arrays written in Cython."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.math cimport fabs, sqrt, isnan   # 导入数学函数fabs, sqrt, isnan
from libc.stdint cimport intptr_t   # 导入整型类型intptr_t

import numpy as np   # 导入NumPy库
from cython cimport floating   # 导入Cython的浮点类型
from ..utils._typedefs cimport float64_t, int32_t, int64_t, intp_t, uint64_t   # 从相关位置导入自定义类型

ctypedef fused integral:   # 定义融合类型integral，包含int32_t和int64_t

    int32_t
    int64_t


def csr_row_norms(X):
    """Squared L2 norm of each row in CSR matrix X."""
    if X.dtype not in [np.float32, np.float64]:   # 检查X的数据类型是否为浮点数，如果不是则转换为np.float64
        X = X.astype(np.float64)
    return _sqeuclidean_row_norms_sparse(X.data, X.indptr)


def _sqeuclidean_row_norms_sparse(
    const floating[::1] X_data,   # X_data为浮点数数组
    const integral[::1] X_indptr,   # X_indptr为整型数组
):
    cdef:
        integral n_samples = X_indptr.shape[0] - 1   # 计算样本数，即行数
        integral i, j   # 声明整型变量i和j

    dtype = np.float32 if floating is float else np.float64   # 根据浮点类型确定dtype

    cdef floating[::1] squared_row_norms = np.zeros(n_samples, dtype=dtype)   # 初始化存储行平方L2范数的数组

    with nogil:   # 使用nogil上下文以释放全局解释器锁
        for i in range(n_samples):   # 遍历每一行
            for j in range(X_indptr[i], X_indptr[i + 1]):   # 遍历当前行中的每个元素
                squared_row_norms[i] += X_data[j] * X_data[j]   # 计算当前行的平方L2范数

    return np.asarray(squared_row_norms)   # 将结果转换为NumPy数组并返回


def csr_mean_variance_axis0(X, weights=None, return_sum_weights=False):
    """Compute mean and variance along axis 0 on a CSR matrix

    Uses a np.float64 accumulator.

    Parameters
    ----------
    X : CSR sparse matrix, shape (n_samples, n_features)
        Input data.

    weights : ndarray of shape (n_samples,), dtype=floating, default=None
        If it is set to None samples will be equally weighted.

        .. versionadded:: 0.24

    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature.

        .. versionadded:: 0.24

    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means

    variances : float array with shape (n_features,)
        Feature-wise variances

    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if return_sum_weights is True.
    """
    if X.dtype not in [np.float32, np.float64]:   # 检查X的数据类型是否为浮点数，如果不是则转换为np.float64
        X = X.astype(np.float64)

    if weights is None:   # 如果weights未指定，则设置为均匀权重
        weights = np.ones(X.shape[0], dtype=X.dtype)

    means, variances, sum_weights = _csr_mean_variance_axis0(
        X.data, X.shape[0], X.shape[1], X.indices, X.indptr, weights)   # 调用底层函数计算均值、方差和权重和

    if return_sum_weights:   # 如果return_sum_weights为True，则返回均值、方差和权重和
        return means, variances, sum_weights
    return means, variances   # 否则仅返回均值和方差


def _csr_mean_variance_axis0(
    const floating[::1] X_data,   # X_data为浮点数数组
    uint64_t n_samples,   # 样本数
    uint64_t n_features,   # 特征数
    const integral[:] X_indices,   # X的列索引数组
    const integral[:] X_indptr,   # X的行指针数组
    const floating[:] weights,   # 权重数组
):
    # Implement the function here since variables using fused types
    # cannot be declared directly and can only be passed as function arguments
    cdef:
        intp_t row_ind                       # 声明行索引变量
        uint64_t feature_idx                 # 声明特征索引变量
        integral i, col_ind                  # 声明整数变量 i 和列索引变量 col_ind
        float64_t diff                       # 声明浮点数变量 diff，用于存储差值

        # means[j] contains the mean of feature j
        float64_t[::1] means = np.zeros(n_features)  # 初始化数组 means，存储每个特征 j 的均值

        # variances[j] contains the variance of feature j
        float64_t[::1] variances = np.zeros(n_features)  # 初始化数组 variances，存储每个特征 j 的方差

        float64_t[::1] sum_weights = np.full(
            fill_value=np.sum(weights, dtype=np.float64), shape=n_features
        )                                    # 初始化数组 sum_weights，存储每个特征的总权重和

        float64_t[::1] sum_weights_nz = np.zeros(shape=n_features)  # 初始化数组 sum_weights_nz，存储非零元素的总权重和

        float64_t[::1] correction = np.zeros(shape=n_features)      # 初始化数组 correction，存储修正值

        uint64_t[::1] counts = np.full(
            fill_value=weights.shape[0], shape=n_features, dtype=np.uint64
        )                                    # 初始化数组 counts，存储每个特征的计数

        uint64_t[::1] counts_nz = np.zeros(shape=n_features, dtype=np.uint64)  # 初始化数组 counts_nz，存储非零元素的计数

    for row_ind in range(len(X_indptr) - 1):  # 遍历行索引范围
        for i in range(X_indptr[row_ind], X_indptr[row_ind + 1]):  # 遍历当前行对应的列索引范围
            col_ind = X_indices[i]           # 获取当前列索引
            if not isnan(X_data[i]):         # 如果当前数据不是 NaN
                means[col_ind] += <float64_t>(X_data[i]) * weights[row_ind]  # 更新特征 col_ind 的均值
                # sum of weights where X[:, col_ind] is non-zero
                sum_weights_nz[col_ind] += weights[row_ind]  # 更新非零元素总权重和
                # number of non-zero elements of X[:, col_ind]
                counts_nz[col_ind] += 1       # 更新非零元素计数
            else:
                # sum of weights where X[:, col_ind] is not nan
                sum_weights[col_ind] -= weights[row_ind]  # 更新非 NaN 元素总权重和
                # number of non nan elements of X[:, col_ind]
                counts[col_ind] -= 1           # 更新非 NaN 元素计数

    for feature_idx in range(n_features):    # 遍历所有特征
        means[feature_idx] /= sum_weights[feature_idx]  # 计算每个特征的均值

    for row_ind in range(len(X_indptr) - 1):  # 再次遍历行索引范围
        for i in range(X_indptr[row_ind], X_indptr[row_ind + 1]):  # 遍历当前行对应的列索引范围
            col_ind = X_indices[i]           # 获取当前列索引
            if not isnan(X_data[i]):         # 如果当前数据不是 NaN
                diff = X_data[i] - means[col_ind]  # 计算当前数据与均值的差值
                # correction term of the corrected 2 pass algorithm.
                # See "Algorithms for computing the sample variance: analysis
                # and recommendations", by Chan, Golub, and LeVeque.
                correction[col_ind] += diff * weights[row_ind]  # 更新修正值
                variances[col_ind] += diff * diff * weights[row_ind]  # 更新方差
    # 遍历特征索引范围内的每一个特征
    for feature_idx in range(n_features):
        # 如果某特征的计数与非零计数不相等
        if counts[feature_idx] != counts_nz[feature_idx]:
            # 对该特征的校正值进行修正，减去权重总和的差乘以该特征的均值
            correction[feature_idx] -= (
                sum_weights[feature_idx] - sum_weights_nz[feature_idx]
            ) * means[feature_idx]
        
        # 对校正值进行平方，并除以该特征的权重总和
        correction[feature_idx] = correction[feature_idx]**2 / sum_weights[feature_idx]
        
        # 再次检查特征的计数与非零计数是否不相等
        if counts[feature_idx] != counts_nz[feature_idx]:
            # 仅在确保非零时计算以避免灾难性的数值抵消
            # 更新该特征的方差，加上权重总和的差乘以该特征均值的平方
            variances[feature_idx] += (
                sum_weights[feature_idx] - sum_weights_nz[feature_idx]
            ) * means[feature_idx]**2
        
        # 最终计算该特征的方差修正值，减去之前计算的校正值，并除以该特征的权重总和
        variances[feature_idx] = (
            (variances[feature_idx] - correction[feature_idx]) /
            sum_weights[feature_idx]
        )

    # 如果指定了浮点数类型为 float
    if floating is float:
        # 返回转换为 np.float32 类型的均值、方差和权重总和的数组
        return (
            np.array(means, dtype=np.float32),
            np.array(variances, dtype=np.float32),
            np.array(sum_weights, dtype=np.float32),
        )
    else:
        # 否则返回转换为 np.ndarray 类型的均值、方差和权重总和的数组
        return (
            np.asarray(means), np.asarray(variances), np.asarray(sum_weights)
        )
# 定义函数 csc_mean_variance_axis0，计算 CSC 稀疏矩阵沿轴 0 的均值和方差
def csc_mean_variance_axis0(X, weights=None, return_sum_weights=False):
    """Compute mean and variance along axis 0 on a CSC matrix

    Uses a np.float64 accumulator.

    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    weights : ndarray of shape (n_samples,), dtype=floating, default=None
        If it is set to None samples will be equally weighted.

        .. versionadded:: 0.24

    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature.

        .. versionadded:: 0.24

    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means

    variances : float array with shape (n_features,)
        Feature-wise variances

    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if return_sum_weights is True.
    """
    # 如果 X 的数据类型不是 np.float32 或 np.float64，则将其转换为 np.float64
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)

    # 如果 weights 为 None，则将其设为长度为 n_samples 的 np.float64 数组，所有权重值均为 1
    if weights is None:
        weights = np.ones(X.shape[0], dtype=X.dtype)

    # 调用 _csc_mean_variance_axis0 函数计算均值、方差和加权和
    means, variances, sum_weights = _csc_mean_variance_axis0(
        X.data, X.shape[0], X.shape[1], X.indices, X.indptr, weights)

    # 如果 return_sum_weights 为 True，则返回 means、variances 和 sum_weights
    if return_sum_weights:
        return means, variances, sum_weights
    # 否则只返回 means 和 variances
    return means, variances


def _csc_mean_variance_axis0(
    const floating[::1] X_data,
    uint64_t n_samples,
    uint64_t n_features,
    const integral[:] X_indices,
    const integral[:] X_indptr,
    const floating[:] weights,
):
    # Implement the function here since variables using fused types
    # cannot be declared directly and can only be passed as function arguments
    # 使用 fused types 的变量不能直接声明，只能作为函数参数传递

    # cdef: 用于声明 Cython 变量
    cdef:
        integral i, row_ind
        uint64_t feature_idx, col_ind
        float64_t diff
        # means[j] contains the mean of feature j
        float64_t[::1] means = np.zeros(n_features)
        # variances[j] contains the variance of feature j
        float64_t[::1] variances = np.zeros(n_features)

        # sum_weights[j] 包含特征 j 的加权和
        float64_t[::1] sum_weights = np.full(
            fill_value=np.sum(weights, dtype=np.float64), shape=n_features
        )
        # sum_weights_nz[j] 用于存储特征 j 的非零权重和
        float64_t[::1] sum_weights_nz = np.zeros(shape=n_features)
        # correction[j] 用于修正特征 j 的统计值
        float64_t[::1] correction = np.zeros(shape=n_features)

        # counts[j] 包含特征 j 的样本计数
        uint64_t[::1] counts = np.full(
            fill_value=weights.shape[0], shape=n_features, dtype=np.uint64
        )
        # counts_nz[j] 用于存储特征 j 的非零样本计数
        uint64_t[::1] counts_nz = np.zeros(shape=n_features, dtype=np.uint64)
    # 遍历特征列的索引范围
    for col_ind in range(n_features):
        # 遍历当前特征列中的非零元素
        for i in range(X_indptr[col_ind], X_indptr[col_ind + 1]):
            # 获取当前元素所在行的索引
            row_ind = X_indices[i]
            # 如果当前元素不是 NaN，则执行以下操作
            if not isnan(X_data[i]):
                # 计算加权均值，乘以对应行的权重
                means[col_ind] += <float64_t>(X_data[i]) * weights[row_ind]
                # 统计 X[:, col_ind] 中非零元素的权重总和
                sum_weights_nz[col_ind] += weights[row_ind]
                # 统计 X[:, col_ind] 中非零元素的个数
                counts_nz[col_ind] += 1
            else:
                # 统计 X[:, col_ind] 中 NaN 元素的权重总和
                sum_weights[col_ind] -= weights[row_ind]
                # 统计 X[:, col_ind] 中 NaN 元素的个数
                counts[col_ind] -= 1

    # 计算每个特征列的加权平均值
    for feature_idx in range(n_features):
        means[feature_idx] /= sum_weights[feature_idx]

    # 计算方差和修正项，应用于“修正的二遍算法”中
    for col_ind in range(n_features):
        for i in range(X_indptr[col_ind], X_indptr[col_ind + 1]):
            row_ind = X_indices[i]
            if not isnan(X_data[i]):
                # 计算当前元素与均值的差值
                diff = X_data[i] - means[col_ind]
                # 计算修正项，乘以对应行的权重
                correction[col_ind] += diff * weights[row_ind]
                # 计算方差，乘以差值的平方和权重
                variances[col_ind] += diff * diff * weights[row_ind]

    # 根据计数检查是否需要进行额外的修正
    for feature_idx in range(n_features):
        if counts[feature_idx] != counts_nz[feature_idx]:
            # 调整修正项，以解决计数不一致的情况
            correction[feature_idx] -= (
                sum_weights[feature_idx] - sum_weights_nz[feature_idx]
            ) * means[feature_idx]
        # 计算修正后的平方项，以及对方差的影响
        correction[feature_idx] = correction[feature_idx]**2 / sum_weights[feature_idx]
        if counts[feature_idx] != counts_nz[feature_idx]:
            # 仅在保证非零时计算，避免严重的抵消效应
            variances[feature_idx] += (
                sum_weights[feature_idx] - sum_weights_nz[feature_idx]
            ) * means[feature_idx]**2
        # 计算最终的方差值
        variances[feature_idx] = (
            (variances[feature_idx] - correction[feature_idx])
        ) / sum_weights[feature_idx]

    # 根据浮点类型返回相应的数组
    if floating is float:
        return (np.array(means, dtype=np.float32),
                np.array(variances, dtype=np.float32),
                np.array(sum_weights, dtype=np.float32))
    else:
        return (
            np.asarray(means), np.asarray(variances), np.asarray(sum_weights)
        )
# 定义函数 `incr_mean_variance_axis0`，用于在 CSR 或 CSC 稀疏矩阵的轴 0 上计算均值和方差。

# 如果输入矩阵 X 的数据类型不是 np.float32 或 np.float64，则将其转换为 np.float64 类型
if X.dtype not in [np.float32, np.float64]:
    X = X.astype(np.float64)

# 保存输入矩阵 X 的数据类型
X_dtype = X.dtype

# 如果 weights 参数为 None，则将 weights 初始化为长度为 X.shape[0] 的全 1 数组，并指定数据类型为 X_dtype
if weights is None:
    weights = np.ones(X.shape[0], dtype=X_dtype)
# 如果 weights 参数的数据类型不是 np.float32 或 np.float64，则将其转换为 np.float64 类型
elif weights.dtype not in [np.float32, np.float64]:
    weights = weights.astype(np.float64, copy=False)

# 如果 last_n 的数据类型不是 np.float32 或 np.float64，则将其转换为 np.float64 类型
if last_n.dtype not in [np.float32, np.float64]:
    last_n = last_n.astype(np.float64, copy=False)

# 调用内部函数 `_incr_mean_variance_axis0` 进行实际的均值和方差计算
return _incr_mean_variance_axis0(X.data,            # 矩阵 X 的数据部分
                                 np.sum(weights),  # weights 数组的总和
                                 X.shape[1],       # 矩阵 X 的列数，即特征数
                                 X.indices,        # CSR 或 CSC 矩阵的索引数组
                                 X.indptr,         # CSR 或 CSC 矩阵的指针数组
                                 X.format,         # 矩阵 X 的格式
                                 last_mean.astype(X_dtype, copy=False),  # 上一步的均值数组
                                 last_var.astype(X_dtype, copy=False),   # 上一步的方差数组
                                 last_n.astype(X_dtype, copy=False),     # 上一步的样本数数组
                                 weights.astype(X_dtype, copy=False))    # 权重数组，与 X 的数据类型相同
    # 定义一个常量，表示之前权重的总和，数据类型为浮点数数组
    const floating[:] weights,
):
    # 实现函数在此，因为使用了融合类型的变量，不能直接声明，只能作为函数参数传递
    cdef:
        uint64_t i

        # last = 目前的统计数据
        # new = 当前增量
        # updated = 聚合后的统计数据
        # 对于数组，它们以每个特征的索引 i 进行索引
        floating[::1] new_mean
        floating[::1] new_var
        floating[::1] updated_mean
        floating[::1] updated_var

    # 根据 floating 的类型确定 dtype
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # 初始化各个统计数据数组
    new_mean = np.zeros(n_features, dtype=dtype)
    new_var = np.zeros_like(new_mean, dtype=dtype)
    updated_mean = np.zeros_like(new_mean, dtype=dtype)
    updated_var = np.zeros_like(new_mean, dtype=dtype)

    cdef:
        floating[::1] new_n
        floating[::1] updated_n
        floating[::1] last_over_new_n

    # 首先获取新的统计数据
    updated_n = np.zeros(shape=n_features, dtype=dtype)
    last_over_new_n = np.zeros_like(updated_n, dtype=dtype)

    # X 可能是 CSR 或 CSC 矩阵
    if X_format == 'csr':
        # 调用 CSR 矩阵的均值和方差计算函数
        new_mean, new_var, new_n = _csr_mean_variance_axis0(
            X_data, n_samples, n_features, X_indices, X_indptr, weights)
    else:  # X_format == 'csc'
        # 调用 CSC 矩阵的均值和方差计算函数
        new_mean, new_var, new_n = _csc_mean_variance_axis0(
            X_data, n_samples, n_features, X_indices, X_indptr, weights)

    # 第一遍处理
    cdef bint is_first_pass = True
    for i in range(n_features):
        # 如果 last_n 中有正数，则不是第一遍处理
        if last_n[i] > 0:
            is_first_pass = False
            break

    if is_first_pass:
        return np.asarray(new_mean), np.asarray(new_var), np.asarray(new_n)

    # 更新累计样本数
    for i in range(n_features):
        updated_n[i] = last_n[i] + new_n[i]

    # 后续处理
    for i in range(n_features):
        if new_n[i] > 0:
            # 计算 last_n[i] 与 new_n[i] 的比例
            last_over_new_n[i] = dtype(last_n[i]) / dtype(new_n[i])
            # 非归一化的统计数据
            last_mean[i] *= last_n[i]
            last_var[i] *= last_n[i]
            new_mean[i] *= new_n[i]
            new_var[i] *= new_n[i]
            # 更新统计数据
            updated_var[i] = (
                last_var[i] + new_var[i] +
                last_over_new_n[i] / updated_n[i] *
                (last_mean[i] / last_over_new_n[i] - new_mean[i])**2
            )
            updated_mean[i] = (last_mean[i] + new_mean[i]) / updated_n[i]
            updated_var[i] /= updated_n[i]
        else:
            # 若 new_n[i] <= 0，则保持不变
            updated_var[i] = last_var[i]
            updated_mean[i] = last_mean[i]
            updated_n[i] = last_n[i]

    # 返回更新后的统计数据
    return (
        np.asarray(updated_mean),
        np.asarray(updated_var),
        np.asarray(updated_n),
    )


def inplace_csr_row_normalize_l1(X):
    """原地归一化 CSR 矩阵或数组的行，按其 L1 范数。

    Parameters
    ----------
    X : scipy.sparse.csr_matrix 和 scipy.sparse.csr_array，\
            形状=(n_samples, n_features)
        要就地修改的输入矩阵或数组。

    Examples
    ----------
    # 导入必要的库和模块
    >>> from scipy.sparse import csr_matrix
    >>> from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
    # 创建一个稀疏矩阵 X，数据为 [1.0, 2.0, 3.0]，行索引为 [0, 2, 3]，列指针为 [0, 3, 4]，形状为 (3, 4)
    >>> X = csr_matrix(([1.0, 2.0, 3.0], [0, 2, 3], [0, 3, 4]), shape=(3, 4))
    # 将稀疏矩阵 X 转换为稠密数组并打印出来
    >>> X.toarray()
    array([[1., 2., 0., 0.],
           [0., 0., 3., 0.],
           [0., 0., 0., 4.]])
    # 对稀疏矩阵 X 进行原地归一化操作（每行归一化为 L1 范数）
    >>> inplace_csr_row_normalize_l1(X)
    # 再次将归一化后的稀疏矩阵 X 转换为稠密数组并打印出来
    >>> X.toarray()
    array([[0.33..., 0.66..., 0.        , 0.        ],
           [0.        , 0.        , 1.        , 0.        ],
           [0.        , 0.        , 0.        , 1.        ]])
    """
    # 调用底层的 C 函数 _inplace_csr_row_normalize_l1 来执行原地 CSR 行归一化操作
    _inplace_csr_row_normalize_l1(X.data, X.shape, X.indices, X.indptr)
# 导入所需的库
from libc.stdint cimport uint64_t  # 导入 uint64_t 类型
from libc.stdlib cimport intptr_t  # 导入 intptr_t 类型
from libc.math cimport fabs, sqrt  # 导入 fabs 和 sqrt 函数

# 定义一个函数，用于对 CSR 稀疏矩阵的行进行 L1 范数归一化（原地操作）
def _inplace_csr_row_normalize_l1(
    floating[:] X_data,            # CSR 矩阵数据的数组
    shape,                         # 矩阵形状元组
    const integral[:] X_indices,   # CSR 矩阵列索引的数组
    const integral[:] X_indptr,    # CSR 矩阵行指针的数组
):
    cdef:
        uint64_t n_samples = shape[0]  # 矩阵的行数

        # 对于行 i，列索引存储在：
        #    indices[indptr[i]:indices[i+1]]
        # 对应的数值存储在：
        #    data[indptr[i]:indptr[i+1]]
        uint64_t i  # 行索引
        integral j  # 列索引
        double sum_  # 用于存储行的 L1 范数

    # 遍历矩阵的每一行
    for i in range(n_samples):
        sum_ = 0.0  # 初始化行的 L1 范数和

        # 计算当前行的 L1 范数
        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += fabs(X_data[j])

        # 如果当前行的 L1 范数为 0，跳过归一化（可能是由于 CSR 矩阵未正确修剪）
        if sum_ == 0.0:
            continue

        # 对当前行进行 L1 范数归一化
        for j in range(X_indptr[i], X_indptr[i + 1]):
            X_data[j] /= sum_


# 定义一个函数，用于对 CSR 稀疏矩阵的行进行 L2 范数归一化（原地操作）
def _inplace_csr_row_normalize_l2(
    floating[:] X_data,            # CSR 矩阵数据的数组
    shape,                         # 矩阵形状元组
    const integral[:] X_indices,   # CSR 矩阵列索引的数组
    const integral[:] X_indptr,    # CSR 矩阵行指针的数组
):
    cdef:
        uint64_t n_samples = shape[0]  # 矩阵的行数
        uint64_t i  # 行索引
        integral j  # 列索引
        double sum_  # 用于存储行的 L2 范数

    # 遍历矩阵的每一行
    for i in range(n_samples):
        sum_ = 0.0  # 初始化行的 L2 范数和

        # 计算当前行的平方和
        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += (X_data[j] * X_data[j])

        # 如果当前行的 L2 范数为 0，跳过归一化（可能是由于 CSR 矩阵未正确修剪）
        if sum_ == 0.0:
            continue

        # 计算当前行的 L2 范数
        sum_ = sqrt(sum_)

        # 对当前行进行 L2 范数归一化
        for j in range(X_indptr[i], X_indptr[i + 1]):
            X_data[j] /= sum_


# 定义一个函数，用于将 CSR 稀疏矩阵的选定行密集化到预分配的数组中
def assign_rows_csr(
    X,                              # 输入的 CSR 矩阵
    const intptr_t[:] X_rows,       # 输入矩阵中要复制的行索引数组
    const intptr_t[:] out_rows,     # 目标数组中的行索引数组
    floating[:, ::1] out,           # 预分配的目标数组
):
    """Densify selected rows of a CSR matrix into a preallocated array.

    Like out[out_rows] = X[X_rows].toarray() but without copying.
    No-copy supported for both dtype=np.float32 and dtype=np.float64.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix, shape=(n_samples, n_features)
        输入的 CSR 矩阵
    X_rows : array, dtype=np.intp, shape=n_rows
        要复制的输入矩阵中的行索引数组
    out_rows : array, dtype=np.intp, shape=n_rows
        目标数组中的行索引数组
    out : array, shape=(arbitrary, n_features)
        预分配的目标数组
    """
    cdef:
        # 定义多个变量，用于迭代和索引数组
        # intptr_t (npy_intp, np.intp in Python) 是 np.where 返回的类型，
        # 但是 scipy.sparse 使用的是 int 类型。
        intp_t i, ind, j, k
        intptr_t rX
        const floating[:] data = X.data
        const int32_t[:] indices = X.indices
        const int32_t[:] indptr = X.indptr

    # 检查输入的行数是否匹配输出的行数，如果不匹配则抛出 ValueError 异常
    if X_rows.shape[0] != out_rows.shape[0]:
        raise ValueError("cannot assign %d rows to %d"
                         % (X_rows.shape[0], out_rows.shape[0]))

    # 使用 nogil 上下文，表示以下代码块是无 GIL（全局解释器锁）的
    with nogil:
        # 对输出数组进行初始化，将指定行的元素置为 0.0
        for k in range(out_rows.shape[0]):
            out[out_rows[k]] = 0.0

        # 遍历输入行数组
        for i in range(X_rows.shape[0]):
            rX = X_rows[i]  # 获取当前输入行的索引
            # 遍历当前行对应的列的索引范围
            for ind in range(indptr[rX], indptr[rX + 1]):
                j = indices[ind]  # 获取列索引
                out[out_rows[i], j] = data[ind]  # 将数据数组中的值赋给输出数组的对应位置
```