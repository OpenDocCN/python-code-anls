# `D:\src\scipysrc\scikit-learn\sklearn\metrics\pairwise.py`

```
    # 导入必要的库和模块
    """
    Metrics for pairwise distances and affinity of sets of samples.
    """

    # 作者信息和许可证
    """
    Authors: The scikit-learn developers
    SPDX-License-Identifier: BSD-3-Clause
    """

    # 导入所需的库和模块
    import itertools  # 提供高效的迭代工具
    import warnings  # 提供警告管理功能
    from functools import partial  # 函数工具，用于创建偏函数
    from numbers import Integral, Real  # 数字类型的基类

    import numpy as np  # 数组处理库
    from joblib import effective_n_jobs  # 用于计算有效的并行工作数量
    from scipy.sparse import csr_matrix, issparse  # 稀疏矩阵和相关函数
    from scipy.spatial import distance  # 空间距离计算函数

    from .. import config_context  # 导入配置上下文
    from ..exceptions import DataConversionWarning  # 导入数据转换警告
    from ..preprocessing import normalize  # 数据预处理模块中的归一化函数
    from ..utils import (  # 导入常用的工具函数
        check_array,  # 数组检查函数
        gen_batches,  # 生成数据批次的函数
        gen_even_slices,  # 生成均匀切片的函数
    )
    from ..utils._array_api import (  # 导入数组API函数
        _find_matching_floating_dtype,  # 查找匹配的浮点数类型函数
        _is_numpy_namespace,  # 判断是否为NumPy命名空间函数
        get_namespace,  # 获取命名空间函数
    )
    from ..utils._chunking import get_chunk_n_rows  # 获取行数的分块函数
    from ..utils._mask import _get_mask  # 获取掩码的函数
    from ..utils._missing import is_scalar_nan  # 判断是否为标量NaN函数
    from ..utils._param_validation import (  # 参数验证模块
        Hidden,  # 隐藏参数
        Interval,  # 区间参数
        MissingValues,  # 缺失值处理
        Options,  # 选项参数
        StrOptions,  # 字符串选项参数
        validate_params,  # 验证参数函数
    )
    from ..utils.extmath import row_norms, safe_sparse_dot  # 数学扩展函数
    from ..utils.fixes import parse_version, sp_base_version  # 修复和版本解析函数
    from ..utils.parallel import Parallel, delayed  # 并行处理函数和延迟执行函数
    from ..utils.validation import _num_samples, check_non_negative  # 验证函数和非负数检查函数
    from ._pairwise_distances_reduction import ArgKmin  # 距离降维相关函数
    from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan  # 快速距离计算函数


# Utility Functions
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    # 如果 X 不是稀疏矩阵或者不是 NumPy 数组，则转换为 NumPy 数组
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    # 如果 Y 为 None，则 Y_dtype 为 X 的数据类型
    if Y is None:
        Y_dtype = X.dtype
    # 否则，如果 Y 不是稀疏矩阵或者不是 NumPy 数组，则转换为 NumPy 数组并获取数据类型
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    # 否则，获取 Y 的数据类型
    else:
        Y_dtype = Y.dtype

    # 如果 X 和 Y 的数据类型都是 np.float32，则返回 np.float32，否则返回 float
    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = float

    return X, Y, dtype


def check_pairwise_arrays(
    X,
    Y,
    *,
    precomputed=False,
    dtype="infer_float",
    accept_sparse="csr",
    force_all_finite=True,
    ensure_2d=True,
    copy=False,
):
    """
    Set X and Y appropriately and checks inputs.

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.
    """
    # 如果 X 不是稀疏矩阵或者不是 NumPy 数组，则转换为 NumPy 数组
    X = check_array(
        X,
        accept_sparse=accept_sparse,
        dtype=dtype,
        copy=copy,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
    )

    # 如果 Y 为 None，则将 Y 设置为 X 的引用而不是复制
    if Y is None:
        Y = X
    else:
        # 否则，如果 Y 不是稀疏矩阵或者不是 NumPy 数组，则转换为 NumPy 数组
        Y = check_array(
            Y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=ensure_2d,
        )

    return X, Y
    # precomputed : bool, default=False
    #     是否将X视为与Y中样本之间的预计算距离。

    # dtype : str, type, list of type or None default="infer_float"
    #     X和Y所需的数据类型。如果为"infer_float"，则dtype将由_return_float_dtype选择适当的浮点类型。
    #     如果为None，则保留输入的数据类型。
    #
    #     .. versionadded:: 0.18
    #        添加版本说明0.18

    # accept_sparse : str, bool or list/tuple of str, default='csr'
    #     表示允许的稀疏矩阵格式的字符串，如'csc'，'csr'等。如果输入是稀疏的但不在允许的格式中，
    #     将转换为第一个列出的格式。True允许输入是任何格式。False意味着稀疏矩阵输入会引发错误。

    # force_all_finite : bool or 'allow-nan', default=True
    #     是否在数组中出现np.inf，np.nan，pd.NA时引发错误。可能的取值有：
    #     - True: 强制数组中所有值都是有限的。
    #     - False: 接受数组中的np.inf，np.nan，pd.NA。
    #     - 'allow-nan': 只接受数组中的np.nan和pd.NA值。不能是无限值。
    #
    #     .. versionadded:: 0.22
    #        添加版本说明0.22
    #        force_all_finite接受字符串'allow-nan'。

    # ensure_2d : bool, default=True
    #     当输入数组不是2维时是否引发错误。在使用某些非数值输入（例如字符串列表）与自定义度量标准时，
    #     将此设置为`False`是必要的。
    #
    #     .. versionadded:: 1.5
    #        添加版本说明1.5

    # copy : bool, default=False
    #     是否会触发强制复制。如果copy=False，则可能会由转换触发复制。
    #
    #     .. versionadded:: 0.22
    #        添加版本说明0.22

    # Returns
    # -------
    # safe_X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
    #     与X相等的数组，确保是一个numpy数组。
    #
    # safe_Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
    #     如果Y不是None，则与Y相等的数组，确保是一个numpy数组。
    #     如果Y为None，则safe_Y将指向X。
    ```
    else:
        # 如果不是预先计算的情况，需要对输入的特征矩阵 X 和目标矩阵 Y 进行检查和转换
        X = check_array(
            X,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
            ensure_2d=ensure_2d,
        )
        Y = check_array(
            Y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
            ensure_2d=ensure_2d,
        )

    if precomputed:
        # 如果是预先计算的情况，确保 X 的列数与 Y 的行数匹配
        if X.shape[1] != Y.shape[0]:
            raise ValueError(
                "Precomputed metric requires shape "
                "(n_queries, n_indexed). Got (%d, %d) "
                "for %d indexed." % (X.shape[0], X.shape[1], Y.shape[0])
            )
    elif ensure_2d and X.shape[1] != Y.shape[1]:
        # 如果 enforce 2d 数组并且 X 和 Y 的列数不匹配，则抛出错误
        # 否则，对于自定义的度量标准，验证留给用户处理
        raise ValueError(
            "Incompatible dimension for X and Y matrices: "
            "X.shape[1] == %d while Y.shape[1] == %d" % (X.shape[1], Y.shape[1])
        )

    return X, Y
# 定义函数：检查并设置X和Y，确保输入参数适用于配对距离计算
def check_paired_arrays(X, Y):
    """Set X and Y appropriately and checks inputs for paired distances.

    All paired distance metrics should use this function first to assert that
    the given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the dimensions of the two arrays are equal.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        The first input array or sparse matrix. Each row represents a sample,
        and each column represents a feature.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
        The second input array or sparse matrix, or None if pairwise distances
        are computed within X. Each row represents a sample, and each column
        represents a feature.

    Returns
    -------
    safe_X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    # 使用函数 check_pairwise_arrays 检查并设置 X 和 Y
    X, Y = check_pairwise_arrays(X, Y)
    # 如果 X 和 Y 的形状不相同，则引发 ValueError 异常
    if X.shape != Y.shape:
        raise ValueError(
            "X and Y should be of same shape. They were respectively %r and %r long."
            % (X.shape, Y.shape)
        )
    # 返回经过检查后的 X 和 Y
    return X, Y


# Pairwise distances
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", "sparse matrix", None],
        "Y_norm_squared": ["array-like", None],
        "squared": ["boolean"],
        "X_norm_squared": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def euclidean_distances(
    X, Y=None, *, Y_norm_squared=None, squared=False, X_norm_squared=None
):
    """
    Compute the distance matrix between each pair from a vector array X and Y.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.

    However, this is not the most precise way of doing this computation,
    because this equation potentially suffers from "catastrophic cancellation".
    Also, the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), optional
        An array where each row is a sample and each column is a feature.
        Default is None, which means that the distance matrix is computed
        within X.

    Y_norm_squared : array-like of shape (n_samples_Y,), optional
        Pre-computed dot products of the rows in Y; see the parameter
        squared.

    squared : boolean, optional
        If True, return squared Euclidean distances. If False, return
        standard Euclidean distances.

    X_norm_squared : array-like of shape (n_samples_X,), optional
        Pre-computed dot products of the rows in X; see the parameter
        squared.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        The computed distances between each pair of samples. If Y is None,
        the distances are computed within X.
    """
    # 实现计算向量数组 X 和 Y 之间的距离矩阵的功能
    pass
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        # Y是一个数组或稀疏矩阵，形状为(n_samples_Y, n_features)，默认为None
        # 每一行是一个样本，每一列是一个特征。

    Y_norm_squared : array-like of shape (n_samples_Y,) or (n_samples_Y, 1) \
            or (1, n_samples_Y), default=None
        # Y的向量之间的预计算点积（例如，`(Y**2).sum(axis=1)`）
        # 在某些情况下可能会被忽略，请参阅下面的说明。

    squared : bool, default=False
        # 是否返回平方欧氏距离，默认为False。

    X_norm_squared : array-like of shape (n_samples_X,) or (n_samples_X, 1) \
            or (1, n_samples_X), default=None
        # X的向量之间的预计算点积（例如，`(X**2).sum(axis=1)`）
        # 在某些情况下可能会被忽略，请参阅下面的说明。

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        # 返回`X`的行向量与`Y`的行向量之间的距离数组。

    See Also
    --------
    paired_distances : X和Y元素对之间的距离。

    Notes
    -----
    # 为了获得更好的精度，如果`X_norm_squared`和`Y_norm_squared`被传递为`np.float32`，
    # 可能会被忽略。

    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # 计算X的行之间的距离
    >>> euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # 计算到原点的距离
    >>> euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    """
    # 检查并调整X和Y，确保它们是成对的数组
    X, Y = check_pairwise_arrays(X, Y)

    if X_norm_squared is not None:
        # 检查并调整X_norm_squared，确保其维度符合预期
        X_norm_squared = check_array(X_norm_squared, ensure_2d=False)
        original_shape = X_norm_squared.shape
        if X_norm_squared.shape == (X.shape[0],):
            X_norm_squared = X_norm_squared.reshape(-1, 1)
        if X_norm_squared.shape == (1, X.shape[0]):
            X_norm_squared = X_norm_squared.T
        if X_norm_squared.shape != (X.shape[0], 1):
            raise ValueError(
                f"Incompatible dimensions for X of shape {X.shape} and "
                f"X_norm_squared of shape {original_shape}."
            )

    if Y_norm_squared is not None:
        # 检查并调整Y_norm_squared，确保其维度符合预期
        Y_norm_squared = check_array(Y_norm_squared, ensure_2d=False)
        original_shape = Y_norm_squared.shape
        if Y_norm_squared.shape == (Y.shape[0],):
            Y_norm_squared = Y_norm_squared.reshape(1, -1)
        if Y_norm_squared.shape == (Y.shape[0], 1):
            Y_norm_squared = Y_norm_squared.T
        if Y_norm_squared.shape != (1, Y.shape[0]):
            raise ValueError(
                f"Incompatible dimensions for Y of shape {Y.shape} and "
                f"Y_norm_squared of shape {original_shape}."
            )

    # 调用内部函数计算欧氏距离并返回结果
    return _euclidean_distances(X, Y, X_norm_squared, Y_norm_squared, squared)
# 计算欧氏距离的计算部分，用于支持 euclidean_distances 函数
def _euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False):
    """Computational part of euclidean_distances

    Assumes inputs are already checked.

    If norms are passed as float32, they are unused. If arrays are passed as
    float32, norms needs to be recomputed on upcast chunks.
    TODO: use a float64 accumulator in row_norms to avoid the latter.
    """
    # 如果 X_norm_squared 已经给定且其数据类型不是 np.float32
    if X_norm_squared is not None and X_norm_squared.dtype != np.float32:
        # 将 X_norm_squared 重新形状为 (-1, 1)，作为 XX
        XX = X_norm_squared.reshape(-1, 1)
    # 或者如果 X 的数据类型不是 np.float32
    elif X.dtype != np.float32:
        # 计算 X 的行范数，返回结果作为 XX
        XX = row_norms(X, squared=True)[:, np.newaxis]
    else:
        # 否则将 XX 设为 None
        XX = None

    # 如果 Y 和 X 是相同的数组对象
    if Y is X:
        # 则 YY 设为 None，或者如果 XX 不为 None，则将其转置后作为 YY
        YY = None if XX is None else XX.T
    else:
        # 如果 Y_norm_squared 已经给定且其数据类型不是 np.float32
        if Y_norm_squared is not None and Y_norm_squared.dtype != np.float32:
            # 将 Y_norm_squared 重新形状为 (1, -1)，作为 YY
            YY = Y_norm_squared.reshape(1, -1)
        # 或者如果 Y 的数据类型不是 np.float32
        elif Y.dtype != np.float32:
            # 计算 Y 的行范数，返回结果作为 YY
            YY = row_norms(Y, squared=True)[np.newaxis, :]
        else:
            # 否则将 YY 设为 None
            YY = None

    # 如果 X 或 Y 的数据类型是 np.float32
    if X.dtype == np.float32 or Y.dtype == np.float32:
        # 为了最小化 float32 的精度问题，我们在将 X 和 Y 升级为 float64 后计算距离矩阵
        distances = _euclidean_distances_upcast(X, XX, Y, YY)
    else:
        # 如果数据类型已经是 float64，则无需分块计算和升级
        distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
    # 将 distances 中所有元素与 0 进行比较，取较大值
    np.maximum(distances, 0, out=distances)

    # 确保向量与自身之间的距离设置为 0.0
    if X is Y:
        np.fill_diagonal(distances, 0)

    # 如果 squared 参数为 False，则返回 distances 的平方根，否则返回原始 distances
    return distances if squared else np.sqrt(distances, out=distances)
    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        表示样本的数组，每行为一个样本，每列为一个特征。

    Y : array-like of shape (n_samples_Y, n_features), default=None
        表示样本的数组，每行为一个样本，每列为一个特征。
        如果为 `None`，则使用 `Y=X`。

    squared : bool, default=False
        是否返回平方欧氏距离。

    missing_values : np.nan, float or int, default=np.nan
        表示缺失值的标识。

    copy : bool, default=True
        是否创建并使用 X 和 Y 的深拷贝（如果 Y 存在）。

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        返回 `X` 的行向量与 `Y` 的行向量之间的距离。

    See Also
    --------
    paired_distances : 计算 X 和 Y 中元素对之间的距离。

    References
    ----------
    * John K. Dixon, "Pattern Recognition with Partly Missing Data",
      IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
      10, pp. 617 - 621, Oct. 1979.
      http://ieeexplore.ieee.org/abstract/document/4310090/

    Examples
    --------
    >>> from sklearn.metrics.pairwise import nan_euclidean_distances
    >>> nan = float("NaN")
    >>> X = [[0, 1], [1, nan]]
    >>> nan_euclidean_distances(X, X) # 计算 X 的行之间的距离
    array([[0.        , 1.41421356],
           [1.41421356, 0.        ]])

    >>> # 计算到原点的距离
    >>> nan_euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    """

    # 根据缺失值类型确定是否允许 NaN，用于强制确定所有元素为有限值
    force_all_finite = "allow-nan" if is_scalar_nan(missing_values) else True
    X, Y = check_pairwise_arrays(
        X, Y, accept_sparse=False, force_all_finite=force_all_finite, copy=copy
    )

    # 获取 X 的缺失值掩码
    missing_X = _get_mask(X, missing_values)

    # 获取 Y 的缺失值掩码
    missing_Y = missing_X if Y is X else _get_mask(Y, missing_values)

    # 将缺失值置为零
    X[missing_X] = 0
    Y[missing_Y] = 0

    # 计算欧氏距离的平方
    distances = euclidean_distances(X, Y, squared=True)

    # 调整带有缺失值的距离
    XX = X * X
    YY = Y * Y
    distances -= np.dot(XX, missing_Y.T)
    distances -= np.dot(missing_X, YY.T)

    # 将距离限制在非负范围内
    np.clip(distances, 0, None, out=distances)

    # 如果 X 和 Y 是同一个对象，确保向量与自身的距离为 0.0
    if X is Y:
        np.fill_diagonal(distances, 0.0)

    # 计算非缺失值的数量
    present_X = 1 - missing_X
    present_Y = present_X if Y is X else ~missing_Y
    present_count = np.dot(present_X, present_Y.T)

    # 避免除以零
    np.maximum(1, present_count, out=present_count)
    distances /= present_count
    distances *= X.shape[1]

    # 如果不是 squared 格式，取欧氏距离的平方根
    if not squared:
        np.sqrt(distances, out=distances)

    return distances
# 计算两个数据集 X 和 Y 之间的欧氏距离。

# 假设 X 和 Y 的数据类型为 float32。
# 假设 XX 和 YY 的数据类型为 float64 或者为 None。

def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None, batch_size=None):
    """Euclidean distances between X and Y.

    Assumes X and Y have float32 dtype.
    Assumes XX and YY have float64 dtype or are None.

    X and Y are upcast to float64 by chunks, which size is chosen to limit
    memory increase by approximately 10% (at least 10MiB).
    """
    # 获取数据集 X 和 Y 的样本数量及特征数量
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    n_features = X.shape[1]

    # 初始化一个空的距离矩阵，用于存储计算出的欧氏距离
    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)

    # 如果 batch_size 未指定，则根据稀疏性计算适当的批次大小
    if batch_size is None:
        # 计算 X 和 Y 的稀疏度
        x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
        y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1

        # 允许的内存增加量为 X、Y 和距离矩阵占用内存的10%（至少10MiB）
        maxmem = max(
            (
                (x_density * n_samples_X + y_density * n_samples_Y) * n_features
                + (x_density * n_samples_X * y_density * n_samples_Y)
            )
            / 10,
            10 * 2**17,
        )

        # 计算批次大小，使得内存增加量控制在允许范围内
        tmp = (x_density + y_density) * n_features
        batch_size = (-tmp + np.sqrt(tmp**2 + 4 * maxmem)) / 2
        batch_size = max(int(batch_size), 1)

    # 生成 X 的批次
    x_batches = gen_batches(n_samples_X, batch_size)

    # 遍历 X 的每个批次
    for i, x_slice in enumerate(x_batches):
        # 获取 X 的当前批次数据，并转换为 float64 类型
        X_chunk = X[x_slice].astype(np.float64)

        # 根据是否提供了 XX 数据，计算 XX 的部分
        if XX is None:
            XX_chunk = row_norms(X_chunk, squared=True)[:, np.newaxis]
        else:
            XX_chunk = XX[x_slice]

        # 生成 Y 的批次
        y_batches = gen_batches(n_samples_Y, batch_size)

        # 遍历 Y 的每个批次
        for j, y_slice in enumerate(y_batches):
            # 当 X 等于 Y 且 j < i 时，距离矩阵是对称的，只需计算一半
            if X is Y and j < i:
                d = distances[y_slice, x_slice].T
            else:
                # 获取 Y 的当前批次数据，并转换为 float64 类型
                Y_chunk = Y[y_slice].astype(np.float64)

                # 根据是否提供了 YY 数据，计算 YY 的部分
                if YY is None:
                    YY_chunk = row_norms(Y_chunk, squared=True)[np.newaxis, :]
                else:
                    YY_chunk = YY[:, y_slice]

                # 计算两个批次数据之间的欧氏距离
                d = -2 * safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)
                d += XX_chunk
                d += YY_chunk

            # 将计算得到的距离存储在距离矩阵中，以 float32 类型存储
            distances[x_slice, y_slice] = d.astype(np.float32, copy=False)

    # 返回计算得到的距离矩阵
    return distances


def _argmin_min_reduce(dist, start):
    # `start` 在函数签名中被指定但未被使用。这是因为高阶的 `pairwise_distances_chunked` 函数需要将作为参数传递的缩减函数具有两个参数的签名。
    pass
    # 使用numpy的argmin函数计算每行中最小值的索引，axis=1表示沿着列的方向计算
    indices = dist.argmin(axis=1)
    # 使用numpy的fancy indexing，根据indices获取每行最小值的实际数值
    values = dist[np.arange(dist.shape[0]), indices]
    # 返回计算得到的索引数组和对应的数值数组
    return indices, values
# 定义一个函数 `_argmin_reduce`，用于在距离数组 `dist` 上执行 argmin 操作，沿着轴 `axis=1` 进行计算。
# `start` 参数在函数签名中指定但未使用。这是因为高阶函数 `pairwise_distances_chunked` 需要将作为参数传递的减少函数具有两个参数的签名。
def _argmin_reduce(dist, start):
    return dist.argmin(axis=1)


# 定义一个常量 `_VALID_METRICS`，包含了一组有效的距离度量标准，用于距离计算的选择。
_VALID_METRICS = [
    "euclidean",
    "l2",
    "l1",
    "manhattan",
    "cityblock",
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "cosine",
    "dice",
    "hamming",
    "jaccard",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
    "wminkowski",
    "nan_euclidean",
    "haversine",
]

# 如果 SciPy 的版本低于 1.11，则向 `_VALID_METRICS` 中添加一个被废弃的距离度量标准 "kulsinski"。
if sp_base_version < parse_version("1.11"):  # pragma: no cover
    _VALID_METRICS += ["kulsinski"]

# 如果 SciPy 的版本低于 1.9，则向 `_VALID_METRICS` 中添加一个被废弃的距离度量标准 "matching"。
if sp_base_version < parse_version("1.9"):
    _VALID_METRICS += ["matching"]

# 定义一个列表 `_NAN_METRICS`，包含了仅用于处理 NaN 值的距离度量标准 "nan_euclidean"。
_NAN_METRICS = ["nan_euclidean"]


# 使用装饰器 `@validate_params` 对函数 `pairwise_distances_argmin_min` 进行参数验证和修饰。
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数 X 可以是数组或稀疏矩阵
        "Y": ["array-like", "sparse matrix"],  # 参数 Y 可以是数组或稀疏矩阵
        "axis": [Options(Integral, {0, 1})],  # 参数 axis 必须是整数 0 或 1
        "metric": [
            StrOptions(set(_VALID_METRICS).union(ArgKmin.valid_metrics())),  # 参数 metric 可以是预定义的有效距离度量标准或可调用对象
            callable,  # 参数 metric 也可以是可调用对象
        ],
        "metric_kwargs": [dict, None],  # 参数 metric_kwargs 可以是字典类型或者 None
    },
    prefer_skip_nested_validation=False,  # 指定 metric 参数不进行嵌套验证
)
# 定义函数 `pairwise_distances_argmin_min`，计算一个点与一组点之间的最小距离及其索引。
def pairwise_distances_argmin_min(
    X, Y, *, axis=1, metric="euclidean", metric_kwargs=None
):
    """Compute minimum distances between one point and a set of points.

    This function computes for each row in X, the index of the row of Y which
    is closest (according to the specified distance). The minimal distances are
    also returned.

    This is mostly equivalent to calling:

        (pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis),
         pairwise_distances(X, Y=Y, metric=metric).min(axis=axis))

    but uses much less memory, and is faster for large arrays.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        Array containing points.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
        Array containing points.

    axis : int, default=1
        Axis along which the argmin and distances are to be computed.

    metric : str or callable, default="euclidean"
        The metric to use when calculating distance between instances in a
        feature array. If a string, it must be one of the pre-defined
        metrics from `_VALID_METRICS` or a valid callable.

    metric_kwargs : dict or None, default=None
        Additional keyword arguments for the metric function.

    """
    # 检查并确保输入的 X 和 Y 都是成对的数组，如果需要转置，则进行转置操作
    X, Y = check_pairwise_arrays(X, Y)

    # 如果 axis 参数为 0，则交换 X 和 Y，以便于后续计算
    if axis == 0:
        X, Y = Y, X

    # 如果 metric_kwargs 为 None，则设为一个空字典
    if metric_kwargs is None:
        metric_kwargs = {}

    # 如果可以使用 ArgKmin 来计算给定的 metric，则执行以下操作
    if ArgKmin.is_usable_for(X, Y, metric):
        # 如果 metric_kwargs 中指定了 "squared" 且 metric 是 "euclidean"，
        # 则将 metric 设置为 "sqeuclidean"，并清空 metric_kwargs
        if metric_kwargs.get("squared", False) and metric == "euclidean":
            metric = "sqeuclidean"
            metric_kwargs = {}

        # 调用 ArgKmin.compute 方法计算最近邻的值和索引
        values, indices = ArgKmin.compute(
            X=X,
            Y=Y,
            k=1,
            metric=metric,
            metric_kwargs=metric_kwargs,
            strategy="auto",
            return_distance=True,
        )
        # 将计算结果展平，以方便后续处理
        values = values.flatten()
        indices = indices.flatten()
    else:
        # Joblib-based backend, which is used when user-defined callable
        # are passed for metric.
        # 当度量指标使用用户定义的可调用函数时，使用基于Joblib的后端。

        # This won't be used in the future once PairwiseDistancesReductions support:
        #   - DistanceMetrics which work on supposedly binary data
        #   - CSR-dense and dense-CSR case if 'euclidean' in metric.
        # 在未来，一旦PairwiseDistancesReductions支持以下内容，这将不再使用：
        #   - 应该作用于二进制数据的距离度量
        #   - 如果metric中包含'euclidean'，则支持CSR稠密和稠密-CSR情况。

        # Turn off check for finiteness because this is costly and because arrays
        # have already been validated.
        # 关闭有限性检查，因为这是昂贵的，并且数组已经过验证。
        with config_context(assume_finite=True):
            indices, values = zip(
                *pairwise_distances_chunked(
                    X, Y, reduce_func=_argmin_min_reduce, metric=metric, **metric_kwargs
                )
            )
        # Concatenate indices and values into single arrays
        # 将indices和values连接成单个数组
        indices = np.concatenate(indices)
        values = np.concatenate(values)

    # Return the concatenated indices and values arrays
    # 返回连接后的indices和values数组
    return indices, values
# 使用装饰器 @validate_params 对函数进行参数验证
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数 X 应为数组或稀疏矩阵
        "Y": ["array-like", "sparse matrix"],  # 参数 Y 应为数组或稀疏矩阵
        "axis": [Options(Integral, {0, 1})],  # 参数 axis 应为整数，可选值为 0 或 1
        "metric": [  # 参数 metric 应为字符串或可调用对象
            StrOptions(set(_VALID_METRICS).union(ArgKmin.valid_metrics())),  # 可选字符串为预定义的一组距离度量
            callable,  # 或者可调用对象，用于自定义度量
        ],
        "metric_kwargs": [dict, None],  # 参数 metric_kwargs 应为字典或者 None
    },
    prefer_skip_nested_validation=False,  # 关闭嵌套验证，即不验证 metric 参数
)
def pairwise_distances_argmin(X, Y, *, axis=1, metric="euclidean", metric_kwargs=None):
    """计算两组点之间的最小距离。

    对于 X 中的每一行，计算到 Y 中最近点的索引（根据指定的距离度量）。

    这个函数基本等价于调用：

        pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis)

    但使用的内存更少，并且对于大型数组更快。

    此函数仅适用于密集的二维数组。

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        包含点的数组。

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
        包含点的数组。

    axis : int, default=1
        要计算 argmin 和距离的轴。

    metric : str or callable, default="euclidean"
        用于距离计算的度量。可以使用 scikit-learn 或 scipy.spatial.distance 中的任何度量。

        如果 metric 是可调用函数，则会对每一对实例（行）调用它，并记录结果值。该可调用函数应接受两个数组作为输入，并返回一个值表示它们之间的距离。对于 Scipy 的度量，这种方式有效，但比传递字符串名字的方式效率低。

        不支持距离矩阵。

        metric 的有效值包括：

        - 来自 scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

        - 来自 scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

        详细信息请参阅 scipy.spatial.distance 的文档。

        .. note::
           `'kulsinski'` 自 SciPy 1.9 起已弃用，并将在 SciPy 1.11 中移除。

        .. note::
           SciPy 1.9 中删除了 `'matching'`（使用 `'hamming'` 替代）。

    metric_kwargs : dict, default=None
        传递给指定度量函数的关键字参数。

    Returns
    -------
    argmin : numpy.ndarray
        Y[argmin[i], :] 是与 X[i, :] 最接近的 Y 中的行。

    See Also
    --------
    pairwise_distances : 计算 X 和 Y 的每对样本之间的距离。

    """
    # 返回与给定度量下 X 和 Y 的最小距离的索引数组
    pairwise_distances_argmin_min : Same as `pairwise_distances_argmin` but also
        returns the distances.
    
    Examples
    --------
    >>> from sklearn.metrics.pairwise import pairwise_distances_argmin
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> pairwise_distances_argmin(X, Y)
    array([0, 1])
    """
    # 检查并调整 X 和 Y 为适当的成对数组形式
    X, Y = check_pairwise_arrays(X, Y)
    
    # 如果指定轴为0，交换 X 和 Y
    if axis == 0:
        X, Y = Y, X
    
    # 如果 metric_kwargs 为 None，设为空字典
    if metric_kwargs is None:
        metric_kwargs = {}
    
    # 如果可以使用 ArgKmin 处理 X 和 Y 对于给定的度量
    if ArgKmin.is_usable_for(X, Y, metric):
        # 如果 metric_kwargs 包含 "squared" 为 True 并且 metric 为 "euclidean"
        # 则将 metric 设置为 "sqeuclidean" 并清空 metric_kwargs
        if metric_kwargs.get("squared", False) and metric == "euclidean":
            metric = "sqeuclidean"
            metric_kwargs = {}
    
        # 计算 X 和 Y 之间的最小距离索引，k=1 表示找到最近的一个点
        indices = ArgKmin.compute(
            X=X,
            Y=Y,
            k=1,
            metric=metric,
            metric_kwargs=metric_kwargs,
            strategy="auto",
            return_distance=False,
        )
        indices = indices.flatten()
    else:
        # 使用基于 Joblib 的后端，当度量标准是用户自定义的可调用函数时使用
    
        # 这将不再使用一旦 PairwiseDistancesReductions 支持：
        #   - 在二进制数据上工作的 DistanceMetrics
        #   - 'euclidean' 在度量中处理 CSR-密集 和 稀疏-CSR 情况
    
        # 关闭有限性检查，因为这很耗费资源，而且数组已经过验证
        with config_context(assume_finite=True):
            # 将生成的 np.ndarray 生成器中的数组连接成一个
            indices = np.concatenate(
                list(
                    # 这里返回一个 np.ndarray 生成器，需要将其展平成一个数组
                    pairwise_distances_chunked(
                        X, Y, reduce_func=_argmin_reduce, metric=metric, **metric_kwargs
                    )
                )
            )
    
    return indices
# 使用装饰器 validate_params 对函数进行参数验证，确保参数 X 是 array-like 或 sparse matrix 类型，Y 是 array-like 或 sparse matrix 类型或者 None
@validate_params(
    {"X": ["array-like", "sparse matrix"], "Y": ["array-like", "sparse matrix", None]},
    prefer_skip_nested_validation=True,
)
# 计算 X 和 Y 中样本之间的 Haversine 距离
def haversine_distances(X, Y=None):
    """Compute the Haversine distance between samples in X and Y.

    The Haversine (or great circle) distance is the angular distance between
    two points on the surface of a sphere. The first coordinate of each point
    is assumed to be the latitude, the second is the longitude, given
    in radians. The dimension of the data must be 2.

    .. math::
       D(x, y) = 2\\arcsin[\\sqrt{\\sin^2((x_{lat} - y_{lat}) / 2)
                                + \\cos(x_{lat})\\cos(y_{lat})\\
                                sin^2((x_{lon} - y_{lon}) / 2)}]

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, 2)
        A feature array.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, 2), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        The distance matrix.

    Notes
    -----
    As the Earth is nearly spherical, the haversine formula provides a good
    approximation of the distance between two points of the Earth surface, with
    a less than 1% error on average.

    Examples
    --------
    We want to calculate the distance between the Ezeiza Airport
    (Buenos Aires, Argentina) and the Charles de Gaulle Airport (Paris,
    France).

    >>> from sklearn.metrics.pairwise import haversine_distances
    >>> from math import radians
    >>> bsas = [-34.83333, -58.5166646]
    >>> paris = [49.0083899664, 2.53844117956]
    >>> bsas_in_radians = [radians(_) for _ in bsas]
    >>> paris_in_radians = [radians(_) for _ in paris]
    >>> result = haversine_distances([bsas_in_radians, paris_in_radians])
    >>> result * 6371000/1000  # multiply by Earth radius to get kilometers
    array([[    0.        , 11099.54035582],
           [11099.54035582,     0.        ]])
    """
    # 导入 DistanceMetric 类
    from ..metrics import DistanceMetric

    # 使用 Haversine 距离测量方式计算 X 和 Y 之间的距离矩阵
    return DistanceMetric.get_metric("haversine").pairwise(X, Y)


# 使用装饰器 validate_params 对函数进行参数验证，确保参数 X 是 array-like 或 sparse matrix 类型，Y 是 array-like 或 sparse matrix 类型或者 None
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", "sparse matrix", None],
    },
    prefer_skip_nested_validation=True,
)
# 计算 X 和 Y 中向量之间的曼哈顿距离
def manhattan_distances(X, Y=None):
    """Compute the L1 distances between the vectors in X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An array where each row is a sample and each column is a feature.
        If `None`, method uses `Y=X`.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Pairwise L1 distances.

    Notes
    -----
    When X and/or Y are CSR sparse matrices and they are not already
    in canonical format, this function modifies them in-place to
    make them canonical.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import manhattan_distances
    >>> manhattan_distances([[3]], [[3]])
    array([[0.]])
    >>> manhattan_distances([[3]], [[2]])
    array([[1.]])
    >>> manhattan_distances([[2]], [[3]])
    array([[1.]])
    >>> manhattan_distances([[1, 2], [3, 4]],\
         [[1, 2], [0, 3]])
    array([[0., 2.],
           [4., 4.]])
    """
    # 检查并调整输入的X和Y，使其成为标准的成对数组
    X, Y = check_pairwise_arrays(X, Y)

    # 如果X或Y是稀疏矩阵，则将它们转换为CSR格式，并在不复制的情况下修改它们
    if issparse(X) or issparse(Y):
        X = csr_matrix(X, copy=False)  # 转换X为CSR格式
        Y = csr_matrix(Y, copy=False)  # 转换Y为CSR格式
        X.sum_duplicates()  # 这一步同时会原地排序索引
        Y.sum_duplicates()  # 这一步同时会原地排序索引
        D = np.zeros((X.shape[0], Y.shape[0]))  # 创建一个全零矩阵D，用于存储距离
        _sparse_manhattan(X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, D)  # 使用稀疏数据计算曼哈顿距离
        return D  # 返回计算得到的距离矩阵D

    # 如果X和Y都不是稀疏矩阵，则使用cityblock距离计算它们之间的距离
    return distance.cdist(X, Y, "cityblock")
# 使用装饰器 validate_params 对 cosine_distances 函数进行参数验证
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数 X 应为数组或稀疏矩阵
        "Y": ["array-like", "sparse matrix", None],  # 参数 Y 应为数组、稀疏矩阵或者为 None
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
# 计算 X 和 Y 之间的余弦距离
def cosine_distances(X, Y=None):
    """Compute cosine distance between samples in X and Y.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        Matrix `X`.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        Matrix `Y`.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the cosine distance between samples in X and Y.

    See Also
    --------
    cosine_similarity : Compute cosine similarity between samples in X and Y.
    scipy.spatial.distance.cosine : Dense matrices only.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import cosine_distances
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> cosine_distances(X, Y)
    array([[1.     , 1.     ],
           [0.42..., 0.18...]])
    """
    # 计算余弦相似度，并转换为距离
    S = cosine_similarity(X, Y)
    S *= -1  # 将相似度转换为距离
    S += 1  # 1 减去相似度得到距离
    np.clip(S, 0, 2, out=S)  # 限制距离在 [0, 2] 范围内
    if X is Y or Y is None:
        # 确保向量与自身的距离为 0.0
        # 这可能由于浮点舍入误差而不是这种情况
        np.fill_diagonal(S, 0.0)
    return S


# 使用 validate_params 装饰器验证参数
@validate_params(
    {"X": ["array-like", "sparse matrix"], "Y": ["array-like", "sparse matrix"]},
    prefer_skip_nested_validation=True,
)
# 计算 X 和 Y 之间的配对欧氏距离
def paired_euclidean_distances(X, Y):
    """Compute the paired euclidean distances between X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input array/matrix X.

    Y : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input array/matrix Y.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Output array/matrix containing the calculated paired euclidean
        distances.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_euclidean_distances
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> paired_euclidean_distances(X, Y)
    array([1., 1.])
    """
    # 检查并确保 X 和 Y 是成对的数组或稀疏矩阵
    X, Y = check_paired_arrays(X, Y)
    return row_norms(X - Y)


# 使用 validate_params 装饰器验证参数
@validate_params(
    {"X": ["array-like", "sparse matrix"], "Y": ["array-like", "sparse matrix"]},
    prefer_skip_nested_validation=True,
)
# 计算 X 和 Y 之间的配对曼哈顿距离
def paired_manhattan_distances(X, Y):
    """Compute the paired L1 distances between X and Y.

    Distances are calculated between (X[0], Y[0]), (X[1], Y[1]), ...,
    (X[n_samples], Y[n_samples]).

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        An array-like where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples, n_features)
        An array-like where each row is a sample and each column is a feature.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        L1 paired distances between the row vectors of `X`
        and the row vectors of `Y`.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_manhattan_distances
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    >>> Y = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> paired_manhattan_distances(X, Y)
    array([1., 2., 1.])
    """
    # 检查并确保输入的X和Y是配对的数组，并返回处理后的数组
    X, Y = check_paired_arrays(X, Y)
    # 计算X和Y的元素差异
    diff = X - Y
    # 如果差异是稀疏矩阵，则将其元素取绝对值后计算每行的和，并压缩成一维数组返回
    if issparse(diff):
        diff.data = np.abs(diff.data)
        return np.squeeze(np.array(diff.sum(axis=1)))
    else:
        # 如果差异是密集矩阵，则将其元素取绝对值后沿最后一个轴（即每行）求和并返回
        return np.abs(diff).sum(axis=-1)
# 使用装饰器验证参数类型和格式是否正确，可以接受数组或稀疏矩阵作为参数
@validate_params(
    {"X": ["array-like", "sparse matrix"], "Y": ["array-like", "sparse matrix"]},
    prefer_skip_nested_validation=True,
)
# 定义函数计算X和Y之间的成对余弦距离
def paired_cosine_distances(X, Y):
    """
    Compute the paired cosine distances between X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples, n_features)
        An array where each row is a sample and each column is a feature.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`, where `distances[i]` is the
        distance between `X[i]` and `Y[i]`.

    Notes
    -----
    The cosine distance is equivalent to the half the squared
    euclidean distance if each sample is normalized to unit norm.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_cosine_distances
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> paired_cosine_distances(X, Y)
    array([0.5       , 0.18...])
    """
    # 调用函数检查并准备成对数组X和Y
    X, Y = check_paired_arrays(X, Y)
    # 返回成对向量的余弦距离的一半，使用归一化后的向量计算
    return 0.5 * row_norms(normalize(X) - normalize(Y), squared=True)


# 定义一个字典，包含了不同距离度量方式的函数引用
PAIRED_DISTANCES = {
    "cosine": paired_cosine_distances,
    "euclidean": paired_euclidean_distances,
    "l2": paired_euclidean_distances,
    "l1": paired_manhattan_distances,
    "manhattan": paired_manhattan_distances,
    "cityblock": paired_manhattan_distances,
}


# 使用装饰器验证参数类型和格式是否正确，接受X和Y作为数组参数，metric参数可以是预定义字符串或可调用函数
@validate_params(
    {
        "X": ["array-like"],
        "Y": ["array-like"],
        "metric": [StrOptions(set(PAIRED_DISTANCES)), callable],
    },
    prefer_skip_nested_validation=True,
)
# 定义函数计算X和Y之间的成对距离
def paired_distances(X, Y, *, metric="euclidean", **kwds):
    """
    Compute the paired distances between X and Y.

    Compute the distances between (X[0], Y[0]), (X[1], Y[1]), etc...

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Array 1 for distance computation.

    Y : ndarray of shape (n_samples, n_features)
        Array 2 for distance computation.

    metric : str or callable, default="euclidean"
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        specified in PAIRED_DISTANCES, including "euclidean",
        "manhattan", or "cosine".
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from `X` as input and return a value indicating
        the distance between them.

    **kwds : dict
        Unused parameters.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        返回 `X` 的行向量和 `Y` 的行向量之间的距离数组。

    See Also
    --------
    sklearn.metrics.pairwise_distances : 计算每对样本之间的距离。

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_distances
    >>> X = [[0, 1], [1, 1]]
    >>> Y = [[0, 1], [2, 1]]
    >>> paired_distances(X, Y)
    array([0., 1.])
    """

    # 如果 metric 在 PAIRED_DISTANCES 中已定义
    if metric in PAIRED_DISTANCES:
        # 获取相应的距离计算函数
        func = PAIRED_DISTANCES[metric]
        # 调用并返回距离函数计算结果
        return func(X, Y)
    # 如果 metric 是可调用的函数
    elif callable(metric):
        # 先检查数组 X, Y 的结构
        X, Y = check_paired_arrays(X, Y)
        # 初始化距离数组
        distances = np.zeros(len(X))
        # 逐个计算每对向量的距离
        for i in range(len(X)):
            distances[i] = metric(X[i], Y[i])
        # 返回计算得到的距离数组
        return distances
# Kernels
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 参数可以是数组或稀疏矩阵
        "Y": ["array-like", "sparse matrix", None],  # Y 参数可以是数组、稀疏矩阵或空值
        "dense_output": ["boolean"],  # dense_output 参数必须是布尔值
    },
    prefer_skip_nested_validation=True,
)
def linear_kernel(X, Y=None, dense_output=True):
    """
    Compute the linear kernel between X and Y.

    Read more in the :ref:`User Guide <linear_kernel>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        A feature array.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    dense_output : bool, default=True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.

        .. versionadded:: 0.20

    Returns
    -------
    kernel : ndarray of shape (n_samples_X, n_samples_Y)
        The Gram matrix of the linear kernel, i.e. `X @ Y.T`.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import linear_kernel
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> linear_kernel(X, Y)
    array([[0., 0.],
           [1., 2.]])
    """
    X, Y = check_pairwise_arrays(X, Y)  # 调用函数确保 X 和 Y 是配对的数组或稀疏矩阵
    return safe_sparse_dot(X, Y.T, dense_output=dense_output)


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 参数可以是数组或稀疏矩阵
        "Y": ["array-like", "sparse matrix", None],  # Y 参数可以是数组、稀疏矩阵或空值
        "degree": [Interval(Real, 1, None, closed="left")],  # degree 参数是大于等于1的实数
        "gamma": [
            Interval(Real, 0, None, closed="left"),  # gamma 参数是大于等于0的实数
            None,
            Hidden(np.ndarray),  # 隐藏的 numpy 数组类型参数
        ],
        "coef0": [Interval(Real, None, None, closed="neither")],  # coef0 参数是任意的实数
    },
    prefer_skip_nested_validation=True,
)
def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """
    Compute the polynomial kernel between X and Y.

        K(X, Y) = (gamma <X, Y> + coef0) ^ degree

    Read more in the :ref:`User Guide <polynomial_kernel>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        A feature array.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    degree : float, default=3
        Kernel degree.

    gamma : float, default=None
        Coefficient of the vector inner product. If None, defaults to 1.0 / n_features.

    coef0 : float, default=1
        Constant offset added to scaled inner product.

    Returns
    -------
    kernel : ndarray of shape (n_samples_X, n_samples_Y)
        The polynomial kernel.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import polynomial_kernel
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> polynomial_kernel(X, Y, degree=2)
    array([[1.     , 1.     ],
           [1.77..., 2.77...]])
    """
    X, Y = check_pairwise_arrays(X, Y)  # 调用函数确保 X 和 Y 是配对的数组或稀疏矩阵
    # 如果 gamma 为 None，则将其设为 1.0 除以 X 的列数
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    # 计算 X 和 Y 的转置的点积，并确保结果为密集矩阵
    K = safe_sparse_dot(X, Y.T, dense_output=True)
    
    # 将点积结果乘以 gamma
    K *= gamma
    
    # 将 coef0 加到 K 上
    K += coef0
    
    # 将 K 的每个元素都乘以 degree 次方
    K **= degree
    
    # 返回最终的核矩阵 K
    return K
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 参数应为 array-like 或 sparse matrix 类型
        "Y": ["array-like", "sparse matrix", None],  # Y 参数应为 array-like、sparse matrix 类型或 None
        "gamma": [
            Interval(Real, 0, None, closed="left"),  # gamma 参数应为大于等于 0 的实数，左闭区间
            None,  # gamma 参数可为 None
            Hidden(np.ndarray),  # gamma 参数类型应为 np.ndarray，但隐藏
        ],
        "coef0": [Interval(Real, None, None, closed="neither")],  # coef0 参数应为实数，但不限制范围闭合
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    """Compute the sigmoid kernel between X and Y.

        K(X, Y) = tanh(gamma <X, Y> + coef0)

    Read more in the :ref:`User Guide <sigmoid_kernel>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        A feature array.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    gamma : float, default=None
        Coefficient of the vector inner product. If None, defaults to 1.0 / n_features.

    coef0 : float, default=1
        Constant offset added to scaled inner product.

    Returns
    -------
    kernel : ndarray of shape (n_samples_X, n_samples_Y)
        Sigmoid kernel between two arrays.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import sigmoid_kernel
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> sigmoid_kernel(X, Y)
    array([[0.76..., 0.76...],
           [0.87..., 0.93...]])
    """
    X, Y = check_pairwise_arrays(X, Y)  # 检查并确保 X 和 Y 是成对的数组

    if gamma is None:
        gamma = 1.0 / X.shape[1]  # 如果 gamma 为 None，则设置为 1.0 / 特征数

    K = safe_sparse_dot(X, Y.T, dense_output=True)  # 计算稀疏矩阵乘积并输出稠密结果
    K *= gamma  # 乘以 gamma
    K += coef0  # 加上 coef0
    np.tanh(K, K)  # 原地计算 tanh
    return K


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 参数应为 array-like 或 sparse matrix 类型
        "Y": ["array-like", "sparse matrix", None],  # Y 参数应为 array-like、sparse matrix 类型或 None
        "gamma": [
            Interval(Real, 0, None, closed="left"),  # gamma 参数应为大于等于 0 的实数，左闭区间
            None,  # gamma 参数可为 None
            Hidden(np.ndarray),  # gamma 参数类型应为 np.ndarray，但隐藏
        ],
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def rbf_kernel(X, Y=None, gamma=None):
    """Compute the rbf (gaussian) kernel between X and Y.

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.

    Read more in the :ref:`User Guide <rbf_kernel>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        A feature array.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    gamma : float, default=None
        If None, defaults to 1.0 / n_features.

    Returns
    -------
    kernel : ndarray of shape (n_samples_X, n_samples_Y)
        The RBF kernel.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import rbf_kernel
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> rbf_kernel(X, Y)
    array([[0.71..., 0.51...],
           [0.51..., 0.71...]])
    """
    X, Y = check_pairwise_arrays(X, Y)  # 检查并确保 X 和 Y 是成对的数组
    # 如果 gamma 参数为 None，则设定默认值为 1.0 / X.shape[1]
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    # 计算输入数据集 X 和 Y 之间的欧氏距离的平方，并将结果存储在 K 中
    K = euclidean_distances(X, Y, squared=True)
    
    # 将 K 中的每个元素乘以 -gamma
    K *= -gamma
    
    # 对 K 中的每个元素进行指数运算，即计算 exp(K)，并将结果存储在 K 中（原地操作）
    np.exp(K, K)
    
    # 返回处理后的 K，它现在包含了经过指数运算的距离度量
    return K
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数X可以是数组或稀疏矩阵
        "Y": ["array-like", "sparse matrix", None],  # 参数Y可以是数组、稀疏矩阵或None
        "gamma": [
            Interval(Real, 0, None, closed="neither"),  # gamma必须是大于0的实数，不包括0
            Hidden(np.ndarray),  # gamma可以是numpy的ndarray类型，但此处隐藏了
            None,  # gamma也可以是None
        ],
    },
    prefer_skip_nested_validation=True,  # 偏好跳过嵌套验证
)
def laplacian_kernel(X, Y=None, gamma=None):
    """Compute the laplacian kernel between X and Y.

    The laplacian kernel is defined as::

        K(x, y) = exp(-gamma ||x-y||_1)

    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <laplacian_kernel>`.

    .. versionadded:: 0.17  # 添加于版本0.17

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        A feature array.  # 特征数组X，形状为(n_samples_X, n_features)

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.  # 可选的第二个特征数组Y，如果为None，则使用Y=X

    gamma : float, default=None
        If None, defaults to 1.0 / n_features. Otherwise it should be strictly positive.  # 如果为None，则默认为1.0 / n_features，否则应为严格正数

    Returns
    -------
    kernel : ndarray of shape (n_samples_X, n_samples_Y)
        The kernel matrix.  # 核矩阵，形状为(n_samples_X, n_samples_Y)

    Examples
    --------
    >>> from sklearn.metrics.pairwise import laplacian_kernel
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> laplacian_kernel(X, Y)
    array([[0.71..., 0.51...],
           [0.51..., 0.71...]])
    """
    X, Y = check_pairwise_arrays(X, Y)  # 检查并确保X和Y是配对的数组或稀疏矩阵

    if gamma is None:
        gamma = 1.0 / X.shape[1]  # 如果gamma为None，则设置为1.0 / X的特征数

    K = -gamma * manhattan_distances(X, Y)  # 计算曼哈顿距离，并乘以-gamma
    np.exp(K, K)  # 在原地对K进行指数运算
    return K


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数X可以是数组或稀疏矩阵
        "Y": ["array-like", "sparse matrix", None],  # 参数Y可以是数组、稀疏矩阵或None
        "dense_output": ["boolean"],  # dense_output参数必须是布尔类型
    },
    prefer_skip_nested_validation=True,  # 偏好跳过嵌套验证
)
def cosine_similarity(X, Y=None, dense_output=True):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Read more in the :ref:`User Guide <cosine_similarity>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        Input data.  # 输入数据X，形状为(n_samples_X, n_features)

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.  # 输入数据Y，如果为None，则输出将是X中所有样本的成对相似度

    dense_output : bool, default=True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.

        .. versionadded:: 0.17
           parameter ``dense_output`` for dense output.  # 用于密集输出的参数dense_output，添加于版本0.17

    Returns
    -------
    similarities : ndarray or sparse matrix of shape (n_samples_X, n_samples_Y)
        Returns the cosine similarity between samples in X and Y.  # 返回X和Y样本之间的余弦相似度

    Examples
    --------
    """
    # 为了避免递归导入问题，在导入模块时加上注释说明
    """
    # 检查并确保输入的X和Y是成对的数组
    X, Y = check_pairwise_arrays(X, Y)
    
    # 对X进行归一化处理，并复制结果
    X_normalized = normalize(X, copy=True)
    
    # 如果X与Y相同，则直接使用X_normalized作为Y_normalized
    if X is Y:
        Y_normalized = X_normalized
    # 否则对Y进行归一化处理，并复制结果
    else:
        Y_normalized = normalize(Y, copy=True)
    
    # 计算X_normalized和Y_normalized的稀疏点积，得到核矩阵K
    K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)
    
    # 返回计算得到的核矩阵K
    return K
# 使用装饰器进行参数验证，指定X和Y为数组类型，gamma为实数区间(0, ∞)，并且在验证时偏好跳过嵌套验证
@validate_params(
    {
        "X": ["array-like"],
        "Y": ["array-like", None],
        "gamma": [Interval(Real, 0, None, closed="neither"), Hidden(np.ndarray)],
    },
    prefer_skip_nested_validation=True,
)
# 计算指数卡方核函数（exponential chi-squared kernel）在X和Y之间的值
def chi2_kernel(X, Y=None, gamma=1.0):
    """Compute the exponential chi-squared kernel between X and Y.
    # 计算基于卡方核的核矩阵，用于衡量样本集合之间的相似性
    # X : 形状为 (n_samples_X, n_features) 的特征数组
    # Y : 形状为 (n_samples_Y, n_features) 的可选第二个特征数组，默认为 None 时使用 Y=X
    # gamma : float，卡方核的缩放参数，默认为 1
    def chi2_kernel(X, Y=None, gamma=1):
        # 获得特征数组的命名空间
        xp, _ = get_namespace(X, Y)
        # 计算加性卡方核矩阵
        K = additive_chi2_kernel(X, Y)
        # 根据参数 gamma 缩放卡方核矩阵
        K *= gamma
        # 如果命名空间是 numpy，则使用 numpy 的指数函数计算核矩阵
        if _is_numpy_namespace(xp):
            return np.exp(K, out=K)
        # 否则使用 xp 的指数函数计算核矩阵
        return xp.exp(K)
# Helper functions - distance
# 定义了一组用于计算成对距离的函数和它们的映射关系
PAIRWISE_DISTANCE_FUNCTIONS = {
    # 如果更新了这个字典，请同时更新 distance_metrics() 和 pairwise_distances() 中的文档！
    "cityblock": manhattan_distances,  # 使用曼哈顿距离计算成对距离
    "cosine": cosine_distances,  # 使用余弦距离计算成对距离
    "euclidean": euclidean_distances,  # 使用欧氏距离计算成对距离
    "haversine": haversine_distances,  # 使用球面距离（haversine）计算成对距离
    "l2": euclidean_distances,  # 使用欧氏距离计算成对距离（同 'euclidean'）
    "l1": manhattan_distances,  # 使用曼哈顿距离计算成对距离（同 'cityblock'）
    "manhattan": manhattan_distances,  # 使用曼哈顿距离计算成对距离（同 'cityblock'）
    "precomputed": None,  # HACK: 预计算距离矩阵，总是被允许但从不调用
    "nan_euclidean": nan_euclidean_distances,  # 使用欧氏距离计算成对距离，处理 NaN 值
}


def distance_metrics():
    """Valid metrics for pairwise_distances.

    This function simply returns the valid pairwise distance metrics.
    It exists to allow for a description of the mapping for
    each of the valid strings.

    The valid distance metrics, and the function they map to, are:

    =============== ========================================
    metric          Function
    =============== ========================================
    'cityblock'     metrics.pairwise.manhattan_distances
    'cosine'        metrics.pairwise.cosine_distances
    'euclidean'     metrics.pairwise.euclidean_distances
    'haversine'     metrics.pairwise.haversine_distances
    'l1'            metrics.pairwise.manhattan_distances
    'l2'            metrics.pairwise.euclidean_distances
    'manhattan'     metrics.pairwise.manhattan_distances
    'nan_euclidean' metrics.pairwise.nan_euclidean_distances
    =============== ========================================

    Read more in the :ref:`User Guide <metrics>`.

    Returns
    -------
    distance_metrics : dict
        Returns valid metrics for pairwise_distances.
    """
    return PAIRWISE_DISTANCE_FUNCTIONS


def _dist_wrapper(dist_func, dist_matrix, slice_, *args, **kwargs):
    """Write in-place to a slice of a distance matrix."""
    # 将距离函数的计算结果直接写入距离矩阵的一个切片中
    dist_matrix[:, slice_] = dist_func(*args, **kwargs)


def _parallel_pairwise(X, Y, func, n_jobs, **kwds):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel."""

    if Y is None:
        Y = X
    X, Y, dtype = _return_float_dtype(X, Y)

    if effective_n_jobs(n_jobs) == 1:
        return func(X, Y, **kwds)

    # enforce a threading backend to prevent data communication overhead
    fd = delayed(_dist_wrapper)
    # 创建一个空的距离矩阵，用于存储计算结果
    ret = np.empty((X.shape[0], Y.shape[0]), dtype=dtype, order="F")
    # 使用线程化的方式并行计算距离矩阵的各个切片
    Parallel(backend="threading", n_jobs=n_jobs)(
        fd(func, ret, s, X, Y[s], **kwds)
        for s in gen_even_slices(_num_samples(Y), effective_n_jobs(n_jobs))
    )

    if (X is Y or Y is None) and func is euclidean_distances:
        # 对于欧氏距离，将对角线上的元素设为零
        # TODO: 也考虑其他范数的情况
        np.fill_diagonal(ret, 0)

    return ret


def _pairwise_callable(X, Y, metric, force_all_finite=True, **kwds):
    """Handle the callable case for pairwise_{distances,kernels}."""
    # 调用 check_pairwise_arrays 函数，确保 X 和 Y 是可用于比较的数组，并根据需要进行类型转换和填充
    X, Y = check_pairwise_arrays(
        X,
        Y,
        dtype=None,
        force_all_finite=force_all_finite,
        ensure_2d=False,
    )

    # 如果 X 和 Y 是同一个对象
    if X is Y:
        # 只计算上三角部分的度量值
        out = np.zeros((X.shape[0], Y.shape[0]), dtype="float")
        # 创建一个迭代器，用于生成 X.shape[0] 中两两组合的索引
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            # 如果 X 是稀疏矩阵，使用 X[i,:] 进行切片；否则直接使用 X[i]
            x = X[[i], :] if issparse(X) else X[i]
            # 如果 Y 是稀疏矩阵，使用 Y[j,:] 进行切片；否则直接使用 Y[j]
            y = Y[[j], :] if issparse(Y) else Y[j]
            # 计算度量函数 metric 的值并存储在 out[i, j] 中
            out[i, j] = metric(x, y, **kwds)

        # 使 out 变为对称矩阵
        # 注意：out += out.T 会导致不正确的结果
        out = out + out.T

        # 计算对角线上的值
        # 注意：度量和核函数允许对角线上有非零值
        for i in range(X.shape[0]):
            # 如果 X 是稀疏矩阵，使用 X[i,:] 进行切片；否则直接使用 X[i]
            x = X[[i], :] if issparse(X) else X[i]
            # 计算度量函数 metric 的值并存储在 out[i, i] 中
            out[i, i] = metric(x, x, **kwds)

    else:
        # 计算所有单元格的度量值
        out = np.empty((X.shape[0], Y.shape[0]), dtype="float")
        # 创建一个迭代器，用于生成 X.shape[0] 和 Y.shape[0] 的所有组合的索引
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            # 如果 X 是稀疏矩阵，使用 X[i,:] 进行切片；否则直接使用 X[i]
            x = X[[i], :] if issparse(X) else X[i]
            # 如果 Y 是稀疏矩阵，使用 Y[j,:] 进行切片；否则直接使用 Y[j]
            y = Y[[j], :] if issparse(Y) else Y[j]
            # 计算度量函数 metric 的值并存储在 out[i, j] 中
            out[i, j] = metric(x, y, **kwds)

    # 返回计算出的度量值矩阵
    return out
# 检查减少后的数据是否符合预期大小的序列或相同大小的元组
def _check_chunk_size(reduced, chunk_size):
    """Checks chunk is a sequence of expected size or a tuple of same."""
    # 如果减少后的数据为 None，则直接返回，不进行检查
    if reduced is None:
        return
    # 检查减少后的数据是否为元组
    is_tuple = isinstance(reduced, tuple)
    # 如果不是元组，则将其转换为单元素元组
    if not is_tuple:
        reduced = (reduced,)
    # 检查每个减少后的数据是否为元组或者是否没有 "__iter__" 属性
    # 如果是，则抛出类型错误异常
    if any(isinstance(r, tuple) or not hasattr(r, "__iter__") for r in reduced):
        raise TypeError(
            "reduce_func returned %r. Expected sequence(s) of length %d."
            % (reduced if is_tuple else reduced[0], chunk_size)
        )
    # 检查每个减少后的数据是否与期望的 chunk_size 大小相等
    # 如果不相等，则抛出数值错误异常
    if any(_num_samples(r) != chunk_size for r in reduced):
        actual_size = tuple(_num_samples(r) for r in reduced)
        raise ValueError(
            "reduce_func returned object of length %s. "
            "Expected same length as input: %d."
            % (actual_size if is_tuple else actual_size[0], chunk_size)
        )


# 预先计算基于数据的度量参数（metric），如果未提供的话
def _precompute_metric_params(X, Y, metric=None, **kwds):
    """Precompute data-derived metric parameters if not provided."""
    # 如果度量（metric）是 "seuclidean" 并且 kwds 中没有 "V" 参数
    if metric == "seuclidean" and "V" not in kwds:
        # 如果 X 与 Y 相同，则计算 X 的方差
        if X is Y:
            V = np.var(X, axis=0, ddof=1)
        else:
            # 否则抛出值错误异常，要求在传递 Y 参数时提供 "V" 参数
            raise ValueError(
                "The 'V' parameter is required for the seuclidean metric "
                "when Y is passed."
            )
        # 返回一个包含 "V" 参数的字典
        return {"V": V}
    # 如果度量（metric）是 "mahalanobis" 并且 kwds 中没有 "VI" 参数
    if metric == "mahalanobis" and "VI" not in kwds:
        # 如果 X 与 Y 相同，则计算 X 转置后的协方差矩阵的逆转置
        if X is Y:
            VI = np.linalg.inv(np.cov(X.T)).T
        else:
            # 否则抛出值错误异常，要求在传递 Y 参数时提供 "VI" 参数
            raise ValueError(
                "The 'VI' parameter is required for the mahalanobis metric "
                "when Y is passed."
            )
        # 返回一个包含 "VI" 参数的字典
        return {"VI": VI}
    # 如果以上条件都不满足，则返回一个空字典
    return {}


# 对参数进行验证，用于 pairwise_distances_chunked 函数的装饰器
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", "sparse matrix", None],
        "reduce_func": [callable, None],
        "metric": [StrOptions({"precomputed"}.union(_VALID_METRICS)), callable],
        "n_jobs": [Integral, None],
        "working_memory": [Interval(Real, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=False,  # metric is not validated yet
)
# 分块生成距离矩阵，可选地进行减少操作
def pairwise_distances_chunked(
    X,
    Y=None,
    *,
    reduce_func=None,
    metric="euclidean",
    n_jobs=None,
    working_memory=None,
    **kwds,
):
    """Generate a distance matrix chunk by chunk with optional reduction.

    In cases where not all of a pairwise distance matrix needs to be
    stored at once, this is used to calculate pairwise distances in
    ``working_memory``-sized chunks.  If ``reduce_func`` is given, it is
    run on each chunk and its return values are concatenated into lists,
    arrays or sparse matrices.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise distances between samples, or a feature array.
        The shape the array should be (n_samples_X, n_samples_X) if
        metric='precomputed' and (n_samples_X, n_features) otherwise.

    """
    # Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
    #     An optional second feature array. Only allowed if
    #     metric != "precomputed".
    
    # reduce_func : callable, default=None
    #     The function which is applied on each chunk of the distance matrix,
    #     reducing it to needed values.  ``reduce_func(D_chunk, start)``
    #     is called repeatedly, where ``D_chunk`` is a contiguous vertical
    #     slice of the pairwise distance matrix, starting at row ``start``.
    #     It should return one of: None; an array, a list, or a sparse matrix
    #     of length ``D_chunk.shape[0]``; or a tuple of such objects.
    #     Returning None is useful for in-place operations, rather than
    #     reductions.
    
    #     If None, pairwise_distances_chunked returns a generator of vertical
    #     chunks of the distance matrix.
    
    # metric : str or callable, default='euclidean'
    #     The metric to use when calculating distance between instances in a
    #     feature array. If metric is a string, it must be one of the options
    #     allowed by scipy.spatial.distance.pdist for its metric parameter,
    #     or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    #     If metric is "precomputed", X is assumed to be a distance matrix.
    #     Alternatively, if metric is a callable function, it is called on
    #     each pair of instances (rows) and the resulting value recorded.
    #     The callable should take two arrays from X as input and return a
    #     value indicating the distance between them.
    
    # n_jobs : int, default=None
    #     The number of jobs to use for the computation. This works by
    #     breaking down the pairwise matrix into n_jobs even slices and
    #     computing them in parallel.
    #
    #     ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    #     ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    #     for more details.
    
    # working_memory : float, default=None
    #     The sought maximum memory for temporary distance matrix chunks.
    #     When None (default), the value of
    #     ``sklearn.get_config()['working_memory']`` is used.
    
    # **kwds : optional keyword parameters
    #     Any further parameters are passed directly to the distance function.
    #     If using a scipy.spatial.distance metric, the parameters are still
    #     metric dependent. See the scipy docs for usage examples.
    
    # Yields
    # ------
    # D_chunk : {ndarray, sparse matrix}
    #     A contiguous slice of distance matrix, optionally processed by
    #     ``reduce_func``.
    
    # Examples
    # --------
    # Without reduce_func:
    
    # >>> import numpy as np
    # >>> from sklearn.metrics import pairwise_distances_chunked
    # >>> X = np.random.RandomState(0).rand(5, 3)
    # >>> D_chunk = next(pairwise_distances_chunked(X))
    # >>> D_chunk
    array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
           [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
           [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
           [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
           [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])

    Retrieve all neighbors and average distance within radius r:

    >>> r = .2
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
    ...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
    ...     return neigh, avg_dist
    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
    >>> neigh, avg_dist = next(gen)
    >>> neigh
    [array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
    >>> avg_dist
    array([0.039..., 0.        , 0.        , 0.039..., 0.        ])

    Where r is defined per sample, we need to make use of ``start``:

    >>> r = [.2, .4, .4, .3, .1]
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r[i])
    ...              for i, d in enumerate(D_chunk, start)]
    ...     return neigh
    >>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
    >>> neigh
    [array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]

    Force row-by-row generation by reducing ``working_memory``:

    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
    ...                                  working_memory=0)
    >>> next(gen)
    [array([0, 3])]
    >>> next(gen)
    [array([0, 1])]

    """
    n_samples_X = _num_samples(X)
    # 如果距离度量为预先计算的情况下，仅需计算上三角矩阵
    if metric == "precomputed":
        slices = (slice(0, n_samples_X),)
    else:
        # 如果 Y 未定义，则将其设为 X
        if Y is None:
            Y = X
        # 计算每个输出行所需的最大行数，以 bytes 表示，并根据工作内存限制确定
        #
        # 注意：
        #  - 即使 1 行距离超出工作内存，也会得到至少 1 行距离。
        #  - 这不考虑计算距离时的任何临时内存使用（例如曼哈顿距离中向量的差异）。
        chunk_n_rows = get_chunk_n_rows(
            row_bytes=8 * _num_samples(Y),
            max_n_rows=n_samples_X,
            working_memory=working_memory,
        )
        # 生成批次切片以分批处理样本
        slices = gen_batches(n_samples_X, chunk_n_rows)

    # 预先计算基于数据的度量参数
    params = _precompute_metric_params(X, Y, metric=metric, **kwds)
    # 更新关键字参数
    kwds.update(**params)
    # 对于每个切片对象 sl 在 slices 中进行迭代
    for sl in slices:
        # 如果切片 sl 的起始位置为0且结束位置为 n_samples_X，表示整体处理 X
        if sl.start == 0 and sl.stop == n_samples_X:
            # 当 X 与 Y 相同或者 Y 为 None 时，启用针对 X 优化的路径
            X_chunk = X  # enable optimised paths for X is Y
        else:
            # 否则，根据当前的切片 sl 获取 X 的一个子集
            X_chunk = X[sl]
        
        # 计算 X_chunk 和 Y 之间的成对距离，使用指定的度量(metric)，并行计算(n_jobs)，传入额外的参数(**kwds)
        D_chunk = pairwise_distances(X_chunk, Y, metric=metric, n_jobs=n_jobs, **kwds)
        
        # 如果 X 和 Y 相同，或者 Y 为 None，并且度量(metric)为欧氏距离(euclidean_distances)
        if (X is Y or Y is None) and PAIRWISE_DISTANCE_FUNCTIONS.get(
            metric, None
        ) is euclidean_distances:
            # 对角线置零，处理 "euclidean" 的别名，如 "l2"
            D_chunk.flat[sl.start :: _num_samples(X) + 1] = 0
        
        # 如果存在 reduce_func 函数
        if reduce_func is not None:
            # 获取当前 D_chunk 的行数作为 chunk_size
            chunk_size = D_chunk.shape[0]
            # 使用 reduce_func 函数对 D_chunk 进行处理，传入起始位置 sl.start
            D_chunk = reduce_func(D_chunk, sl.start)
            # 检查处理后的 D_chunk 是否与原始 chunk_size 相符
            _check_chunk_size(D_chunk, chunk_size)
        
        # 生成器函数，产生当前处理的 D_chunk
        yield D_chunk
# 使用装饰器 validate_params 进行参数验证，确保函数参数的类型和取值符合指定的规范
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数 X 应为数组或稀疏矩阵
        "Y": ["array-like", "sparse matrix", None],  # 参数 Y 可以为数组、稀疏矩阵或 None
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],  # 参数 metric 应为指定的字符串选项或可调用对象
        "n_jobs": [Integral, None],  # 参数 n_jobs 应为整数或 None
        "force_all_finite": ["boolean", StrOptions({"allow-nan"})],  # 参数 force_all_finite 应为布尔值或指定字符串选项
    },
    prefer_skip_nested_validation=True,  # 设置 prefer_skip_nested_validation 为 True，优先跳过嵌套验证
)
def pairwise_distances(
    X,
    Y=None,
    metric="euclidean",
    *,
    n_jobs=None,
    force_all_finite=True,
    **kwds,
):
    """Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns
    a distance matrix.
    If the input is a vector array, the distances are computed.
    If the input is a distances matrix, it is returned instead.
    If the input is a collection of non-numeric data (e.g. a list of strings or a
    boolean array), a custom metric must be passed.

    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix
      inputs.
      ['nan_euclidean'] but it does not yet support sparse matrices.

    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    .. note::
        `'kulsinski'` is deprecated from SciPy 1.9 and will be removed in SciPy 1.11.

    .. note::
        `'matching'` has been removed in SciPy 1.9 (use `'hamming'` instead).

    Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
    valid scipy.spatial.distance metrics), the scikit-learn implementation
    will be used, which is faster and has support for sparse matrices (except
    for 'cityblock'). For a verbose description of the metrics from
    scikit-learn, see :func:`sklearn.metrics.pairwise.distance_metrics`
    function.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise distances between samples, or a feature array.
        The shape of the array should be (n_samples_X, n_samples_X) if
        metric == "precomputed" and (n_samples_X, n_features) otherwise.
    # Y : 形状为 (n_samples_Y, n_features) 的数组或稀疏矩阵，默认为 None
    #     第二个可选的特征数组。仅当 metric 不为 "precomputed" 时允许使用。

    # metric : str 或可调用对象，默认为 'euclidean'
    #     计算特征数组中实例之间距离时要使用的度量标准。如果 metric 是一个字符串，
    #     它必须是 scipy.spatial.distance.pdist 函数的 metric 参数允许的选项之一，
    #     或者是在 "pairwise.PAIRWISE_DISTANCE_FUNCTIONS" 列表中列出的度量标准。
    #     如果 metric 是 "precomputed"，则假定 X 是一个距离矩阵。
    #     如果 metric 是一个可调用函数，则将其应用于每一对实例（行），并记录结果值。
    #     可调用函数应该接受 X 的两个数组作为输入，并返回一个指示它们之间距离的值。

    # n_jobs : int，默认为 None
    #     用于计算的作业数。通过将成对矩阵分解为 n_jobs 个切片并并行计算它们来实现。
    #     
    #     ``None`` 表示使用 1 个作业，除非在 :obj:`joblib.parallel_backend` 上下文中。
    #     ``-1`` 表示使用所有处理器。有关详细信息，请参见 :term:`术语表 <n_jobs>`。
    #     
    #     "euclidean" 和 "cosine" 度量严重依赖于已经多线程化的 BLAS。因此，增加 `n_jobs` 
    #     可能会导致过度订阅，并迅速降低性能。

    # force_all_finite : bool 或 'allow-nan'，默认为 True
    #     是否在数组中出现 np.inf、np.nan、pd.NA 时引发错误。在 "pairwise.PAIRWISE_DISTANCE_FUNCTIONS" 
    #     列表中列出的度量标准中被忽略。可能的选项包括：
    #     
    #     - True: 强制数组的所有值都是有限的。
    #     - False: 允许数组中存在 np.inf、np.nan、pd.NA。
    #     - 'allow-nan': 只允许数组中存在 np.nan 和 pd.NA 值。值不能是无限的。
    #     
    #     .. versionadded:: 0.22
    #        ``force_all_finite`` 支持字符串 ``'allow-nan'``。
    #     
    #     .. versionchanged:: 0.23
    #        接受 `pd.NA` 并将其转换为 `np.nan`。

    # **kwds : 可选的关键字参数
    #     所有其他参数将直接传递给距离函数。如果使用 scipy.spatial.distance 度量标准，
    #     则参数仍然依赖于具体的度量标准。请参阅 scipy 文档获取使用示例。

    # Returns
    # -------
    # D : 形状为 (n_samples_X, n_samples_X) 或 (n_samples_X, n_samples_Y) 的 ndarray
    #     距离矩阵 D，其中 D_{i, j} 是给定矩阵 X 的第 i 和第 j 个向量之间的距离，
    #     如果 Y 为 None。如果 Y 不为 None，则 D_{i, j} 是矩阵 X 的第 i 个数组与 Y 的
    #     第 j 个数组之间的距离。

    # See Also
    # --------
    # pairwise_distances_chunked : 执行与此函数相同的计算，但返回距离矩阵的分块生成器，
    #     以限制内存使用。
    sklearn.metrics.pairwise.paired_distances : Computes the distances between
        corresponding elements of two arrays.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import pairwise_distances
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> pairwise_distances(X, Y, metric='sqeuclidean')
    array([[1., 2.],
           [2., 1.]])
    """
    # 如果距离度量(metric)为"precomputed"，则需要检查并返回预先计算的距离矩阵 X
    if metric == "precomputed":
        X, _ = check_pairwise_arrays(
            X, Y, precomputed=True, force_all_finite=force_all_finite
        )

        whom = (
            "`pairwise_distances`. Precomputed distance "
            " need to have non-negative values."
        )
        # 检查预先计算的距离矩阵 X 中的值是否非负
        check_non_negative(X, whom=whom)
        return X
    # 如果 metric 在 PAIRWISE_DISTANCE_FUNCTIONS 中，选择对应的距离函数
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    # 如果 metric 是可调用对象，则使用偏函数 _pairwise_callable 进行处理
    elif callable(metric):
        func = partial(
            _pairwise_callable,
            metric=metric,
            force_all_finite=force_all_finite,
            **kwds,
        )
    else:
        # 如果 X 或 Y 是稀疏矩阵，则抛出类型错误
        if issparse(X) or issparse(Y):
            raise TypeError("scipy distance metrics do not support sparse matrices.")

        # 确定 dtype 类型，根据 metric 是否在 PAIRWISE_BOOLEAN_FUNCTIONS 中进行判断
        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else "infer_float"

        # 如果 dtype 为 bool，但 X 或 Y 的 dtype 不为 bool，则发出警告
        if dtype == bool and (X.dtype != bool or (Y is not None and Y.dtype != bool)):
            msg = "Data was converted to boolean for metric %s" % metric
            warnings.warn(msg, DataConversionWarning)

        # 对 X 和 Y 进行检查，确保它们可以用于距离计算
        X, Y = check_pairwise_arrays(
            X, Y, dtype=dtype, force_all_finite=force_all_finite
        )

        # 预先计算从数据导出的度量参数
        params = _precompute_metric_params(X, Y, metric=metric, **kwds)
        kwds.update(**params)

        # 如果只有一个工作线程或 X 等于 Y，则使用 pdist 函数计算距离并返回方阵形式的距离矩阵
        if effective_n_jobs(n_jobs) == 1 and X is Y:
            return distance.squareform(distance.pdist(X, metric=metric, **kwds))
        
        # 否则，使用 cdist 函数计算距离
        func = partial(distance.cdist, metric=metric, **kwds)

    # 使用并行计算处理 X 和 Y 的距离计算，并返回结果
    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
# 需要布尔数组的距离度量函数列表，用于 scipy.spatial.distance
PAIRWISE_BOOLEAN_FUNCTIONS = [
    "dice",
    "jaccard",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
]

# 如果 SciPy 版本低于 1.11，则添加已废弃的 kulsinski 函数
if sp_base_version < parse_version("1.11"):
    PAIRWISE_BOOLEAN_FUNCTIONS += ["kulsinski"]

# 如果 SciPy 版本低于 1.9，则添加已废弃的 matching 函数
if sp_base_version < parse_version("1.9"):
    PAIRWISE_BOOLEAN_FUNCTIONS += ["matching"]

# 距离度量的辅助函数字典
PAIRWISE_KERNEL_FUNCTIONS = {
    "additive_chi2": additive_chi2_kernel,
    "chi2": chi2_kernel,
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel,
    "laplacian": laplacian_kernel,
    "sigmoid": sigmoid_kernel,
    "cosine": cosine_similarity,
}

def kernel_metrics():
    """Valid metrics for pairwise_kernels.

    This function simply returns the valid pairwise distance metrics.
    It exists, however, to allow for a verbose description of the mapping for
    each of the valid strings.

    The valid distance metrics, and the function they map to, are:
      ===============   ========================================
      metric            Function
      ===============   ========================================
      'additive_chi2'   sklearn.pairwise.additive_chi2_kernel
      'chi2'            sklearn.pairwise.chi2_kernel
      'linear'          sklearn.pairwise.linear_kernel
      'poly'            sklearn.pairwise.polynomial_kernel
      'polynomial'      sklearn.pairwise.polynomial_kernel
      'rbf'             sklearn.pairwise.rbf_kernel
      'laplacian'       sklearn.pairwise.laplacian_kernel
      'sigmoid'         sklearn.pairwise.sigmoid_kernel
      'cosine'          sklearn.pairwise.cosine_similarity
      ===============   ========================================

    Read more in the :ref:`User Guide <metrics>`.

    Returns
    -------
    kernel_metrics : dict
        Returns valid metrics for pairwise_kernels.
    """
    return PAIRWISE_KERNEL_FUNCTIONS

# 不同核函数的参数字典
KERNEL_PARAMS = {
    "additive_chi2": (),
    "chi2": frozenset(["gamma"]),
    "cosine": (),
    "linear": (),
    "poly": frozenset(["gamma", "degree", "coef0"]),
    "polynomial": frozenset(["gamma", "degree", "coef0"]),
    "rbf": frozenset(["gamma"]),
    "laplacian": frozenset(["gamma"]),
    "sigmoid": frozenset(["gamma", "coef0"]),
}

@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", "sparse matrix", None],
        "metric": [
            StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS) | {"precomputed"}),
            callable,
        ],
        "filter_params": ["boolean"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def pairwise_kernels(
    X, Y=None, metric="linear", *, filter_params=False, n_jobs=None, **kwds


    # 定义函数的参数列表，包括以下参数：
    # - X: 默认为None，通常用于输入特征数据
    # - Y: 默认为None，通常用于目标数据
    # - metric: 默认为"linear"，指定距离度量方法的参数
    # - *: 表示后续参数为命名关键字参数，即必须以关键字形式传入
    # - filter_params: 默认为False，控制是否过滤参数的标志
    # - n_jobs: 默认为None，控制并行运行的作业数
    # - **kwds: 接受额外的关键字参数并将它们保存在字典kwds中
# 计算数组 X 和可选数组 Y 之间的核函数值的矩阵。

"""Compute the kernel between arrays X and optional array Y.

This method takes either a vector array or a kernel matrix, and returns
a kernel matrix. If the input is a vector array, the kernels are
computed. If the input is a kernel matrix, it is returned instead.

This method provides a safe way to take a kernel matrix as input, while
preserving compatibility with many other algorithms that take a vector
array.

If Y is given (default is None), then the returned matrix is the pairwise
kernel between the arrays from both X and Y.

Valid values for metric are:
    ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
    'laplacian', 'sigmoid', 'cosine']

Read more in the :ref:`User Guide <metrics>`.

Parameters
----------
X : {array-like, sparse matrix}  of shape (n_samples_X, n_samples_X) or \
        (n_samples_X, n_features)
    Array of pairwise kernels between samples, or a feature array.
    The shape of the array should be (n_samples_X, n_samples_X) if
    metric == "precomputed" and (n_samples_X, n_features) otherwise.

Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
    A second feature array only if X has shape (n_samples_X, n_features).

metric : str or callable, default="linear"
    The metric to use when calculating kernel between instances in a
    feature array. If metric is a string, it must be one of the metrics
    in ``pairwise.PAIRWISE_KERNEL_FUNCTIONS``.
    If metric is "precomputed", X is assumed to be a kernel matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two rows from X as input and return the corresponding
    kernel value as a single number. This means that callables from
    :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
    matrices, not single samples. Use the string identifying the kernel
    instead.

filter_params : bool, default=False
    Whether to filter invalid parameters or not.

n_jobs : int, default=None
    The number of jobs to use for the computation. This works by breaking
    down the pairwise matrix into n_jobs even slices and computing them in
    parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

**kwds : optional keyword parameters
    Any further parameters are passed directly to the kernel function.

Returns
-------
    # K 是一个 ndarray，其形状可以是 (n_samples_X, n_samples_X) 或者 (n_samples_X, n_samples_Y)
    # 它表示的是一个核矩阵，其中 K_{i, j} 是给定矩阵 X 中第 i 个和第 j 个向量之间的核函数值。
    # 如果 Y 为 None，则计算 X 内部向量之间的核函数；如果 Y 不为 None，则计算 X 和 Y 之间的核函数。

    Notes
    -----
    如果 metric 是 'precomputed'，则忽略 Y 并返回 X。

    Examples
    --------
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> pairwise_kernels(X, Y, metric='linear')
    array([[0., 0.],
           [1., 2.]])
    """
    # 本地导入 GPKernel，以防止循环导入问题
    from ..gaussian_process.kernels import Kernel as GPKernel

    if metric == "precomputed":
        # 如果 metric 是 'precomputed'，则忽略 Y，直接返回 X
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif isinstance(metric, GPKernel):
        # 如果 metric 是 GPKernel 类型，则使用其 __call__ 方法
        func = metric.__call__
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            # 如果需要过滤参数，只保留在 KERNEL_PARAMS[metric] 中的参数
            kwds = {k: kwds[k] for k in kwds if k in KERNEL_PARAMS[metric]}
        # 获取指定 metric 对应的核函数
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        # 如果 metric 是可调用对象，则使用 _pairwise_callable 函数进行计算
        func = partial(_pairwise_callable, metric=metric, **kwds)

    # 使用并行计算计算 X 和 Y 之间的核函数
    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
```