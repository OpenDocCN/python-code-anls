# `D:\src\scipysrc\scikit-learn\sklearn\utils\fixes.py`

```
"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fix is no longer needed.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入 platform 和 struct 模块
import platform
import struct

# 导入 numpy 和 scipy 库
import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats

# 从内部模块导入版本解析函数
from ..externals._packaging.version import parse as parse_version
# 从本地模块中导入线程池控制器函数
from .parallel import _get_threadpool_controller

# 检查系统是否为 32 位
_IS_32BIT = 8 * struct.calcsize("P") == 32
# 检查平台是否为 WASM（WebAssembly）
_IS_WASM = platform.machine() in ["wasm32", "wasm64"]

# 解析并获取 numpy 和 scipy 的版本信息
np_version = parse_version(np.__version__)
np_base_version = parse_version(np_version.base_version)
sp_version = parse_version(scipy.__version__)
sp_base_version = parse_version(sp_version.base_version)

# TODO: 可以考虑删除 containers，并直接从 SciPy 中导入，当稀疏矩阵被弃用时
# 存储 SciPy 稀疏矩阵的容器列表
CSR_CONTAINERS = [scipy.sparse.csr_matrix]
CSC_CONTAINERS = [scipy.sparse.csc_matrix]
COO_CONTAINERS = [scipy.sparse.coo_matrix]
LIL_CONTAINERS = [scipy.sparse.lil_matrix]
DOK_CONTAINERS = [scipy.sparse.dok_matrix]
BSR_CONTAINERS = [scipy.sparse.bsr_matrix]
DIA_CONTAINERS = [scipy.sparse.dia_matrix]

# 如果当前 SciPy 版本大于等于 1.8，添加稀疏数组到容器列表中
if parse_version(scipy.__version__) >= parse_version("1.8"):
    CSR_CONTAINERS.append(scipy.sparse.csr_array)
    CSC_CONTAINERS.append(scipy.sparse.csc_array)
    COO_CONTAINERS.append(scipy.sparse.coo_array)
    LIL_CONTAINERS.append(scipy.sparse.lil_array)
    DOK_CONTAINERS.append(scipy.sparse.dok_array)
    BSR_CONTAINERS.append(scipy.sparse.bsr_array)
    DIA_CONTAINERS.append(scipy.sparse.dia_array)


# 当最小的 SciPy 版本是 1.11.0 时移除此段代码
try:
    from scipy.sparse import sparray  # noqa

    SPARRAY_PRESENT = True
except ImportError:
    SPARRAY_PRESENT = False


# 当最小的 SciPy 版本是 1.8 时移除此段代码
try:
    from scipy.sparse import csr_array  # noqa

    SPARSE_ARRAY_PRESENT = True
except ImportError:
    SPARSE_ARRAY_PRESENT = False


# 尝试导入 scipy.optimize._linesearch 中的函数，如果版本低于 1.8 则导入 scipy.optimize.linesearch 中的函数
try:
    from scipy.optimize._linesearch import line_search_wolfe1, line_search_wolfe2
except ImportError:  # SciPy < 1.8
    from scipy.optimize.linesearch import line_search_wolfe2, line_search_wolfe1  # type: ignore  # noqa


# 检查对象的数据类型是否为 NaN
def _object_dtype_isnan(X):
    return X != X


# 对于 NumPy < 1.22，将 `method` 参数重命名为 `interpolation`，因为在 NumPy >= 1.22 中 `interpolation` 已弃用
def _percentile(a, q, *, method="linear", **kwargs):
    return np.percentile(a, q, interpolation=method, **kwargs)


# 如果 NumPy 版本小于 1.22，则使用自定义的 `_percentile` 函数
if np_version < parse_version("1.22"):
    percentile = _percentile
else:  # >= 1.22
    from numpy import percentile  # type: ignore  # noqa


# TODO: 当 SciPy 1.11 是最小支持版本时移除此段代码
# 如果 SciPy 版本大于等于 "1.9.0"，使用 scipy.stats.mode 函数计算数组 a 的统计模式
# axis 参数指定计算模式的轴，默认为第0轴，keepdims=True 保持维度
if sp_version >= parse_version("1.9.0"):
    mode = scipy.stats.mode(a, axis=axis, keepdims=True)
    # 如果 SciPy 版本大于等于 "1.10.999"，修正模式返回的数组形状
    # 当 axis=None 且 keepdims=True 时，需要将模式展平为一维数组
    if sp_version >= parse_version("1.10.999"):
        if axis is None:
            mode = np.ravel(mode)
    return mode
# 如果 SciPy 版本不满足要求，则使用旧版 scipy.stats.mode 函数计算模式
return scipy.stats.mode(a, axis=axis)


# 如果 SciPy 的基础版本大于等于 "1.12.0"，使用 scipy.sparse.linalg.cg 作为 _sparse_linalg_cg 函数
if sp_base_version >= parse_version("1.12.0"):
    _sparse_linalg_cg = scipy.sparse.linalg.cg
else:
    # 否则定义 _sparse_linalg_cg 函数，处理稀疏矩阵的共轭梯度法求解
    # 如果 kwargs 中包含 "rtol"，则将其改为 "tol"
    # 如果 kwargs 中不包含 "atol"，则设为 "legacy"
    def _sparse_linalg_cg(A, b, **kwargs):
        if "rtol" in kwargs:
            kwargs["tol"] = kwargs.pop("rtol")
        if "atol" not in kwargs:
            kwargs["atol"] = "legacy"
        return scipy.sparse.linalg.cg(A, b, **kwargs)


# 如果 SciPy 的基础版本大于等于 "1.11.0"，定义 _sparse_min_max 和 _sparse_nan_min_max 函数
if sp_base_version >= parse_version("1.11.0"):

    # 计算稀疏矩阵 X 沿指定轴的最小值和最大值，并处理返回结果
    def _sparse_min_max(X, axis):
        the_min = X.min(axis=axis)
        the_max = X.max(axis=axis)

        # 如果 axis 不为 None，则将稀疏矩阵转为密集数组，并展平为一维数组
        if axis is not None:
            the_min = the_min.toarray().ravel()
            the_max = the_max.toarray().ravel()

        return the_min, the_max

    # 计算稀疏矩阵 X 沿指定轴的非 NaN 最小值和最大值，并处理返回结果
    def _sparse_nan_min_max(X, axis):
        the_min = X.nanmin(axis=axis)
        the_max = X.nanmax(axis=axis)

        # 如果 axis 不为 None，则将稀疏矩阵转为密集数组，并展平为一维数组
        if axis is not None:
            the_min = the_min.toarray().ravel()
            the_max = the_max.toarray().ravel()

        return the_min, the_max

else:
    # 如果 SciPy 的基础版本不满足要求，则定义旧版本的 _minor_reduce 函数
    # 该函数用于处理稀疏矩阵 X 并应用给定的 ufunc 函数进行降维操作
    def _minor_reduce(X, ufunc):
        major_index = np.flatnonzero(np.diff(X.indptr))

        # 由于 reduceat 函数在 32 位系统上可能会出错，重新初始化 X 防止这种情况发生
        X = type(X)((X.data, X.indices, X.indptr), shape=X.shape)
        value = ufunc.reduceat(X.data, X.indptr[major_index])
        return major_index, value
    # 定义一个函数用于在稀疏矩阵中沿指定轴计算最小值或最大值
    def _min_or_max_axis(X, axis, min_or_max):
        # 获取矩阵 X 在指定轴上的大小
        N = X.shape[axis]
        # 如果大小为 0，则抛出数值错误异常
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        # 获取矩阵 X 在另一轴上的大小
        M = X.shape[1 - axis]
        # 根据指定轴将 X 转换为压缩稀疏列矩阵 (CSC) 或压缩稀疏行矩阵 (CSR)
        mat = X.tocsc() if axis == 0 else X.tocsr()
        # 去除重复的元素
        mat.sum_duplicates()
        # 调用 _minor_reduce 函数，计算主要索引和值
        major_index, value = _minor_reduce(mat, min_or_max)
        # 找出那些行或列不是完全填满的情况，并将其对应的值与零进行比较，小于则替换为最小值或最大值
        not_full = np.diff(mat.indptr)[major_index] < N
        value[not_full] = min_or_max(value[not_full], 0)
        # 创建一个布尔掩码，标记值不为零的位置
        mask = value != 0
        # 压缩数组，仅保留掩码为真的元素
        major_index = np.compress(mask, major_index)
        value = np.compress(mask, value)

        # 根据轴的不同，创建不同形状的稀疏矩阵结果
        if axis == 0:
            res = scipy.sparse.coo_matrix(
                (value, (np.zeros(len(value)), major_index)),
                dtype=X.dtype,
                shape=(1, M),
            )
        else:
            res = scipy.sparse.coo_matrix(
                (value, (major_index, np.zeros(len(value)))),
                dtype=X.dtype,
                shape=(M, 1),
            )
        # 将稀疏矩阵转换为密集矩阵，并且将其扁平化为一维数组返回
        return res.A.ravel()

    # 定义一个函数用于在稀疏矩阵中计算最小值或最大值
    def _sparse_min_or_max(X, axis, min_or_max):
        # 如果轴为空，则检查数组是否为零大小
        if axis is None:
            if 0 in X.shape:
                raise ValueError("zero-size array to reduction operation")
            # 获取零值，用于比较
            zero = X.dtype.type(0)
            # 如果稀疏矩阵没有非零元素，则直接返回零
            if X.nnz == 0:
                return zero
            # 将数据展平并通过 min_or_max 函数计算其最小值或最大值
            m = min_or_max.reduce(X.data.ravel())
            # 如果稀疏矩阵不是全填充的，则再次与零值进行比较
            if X.nnz != np.prod(X.shape):
                m = min_or_max(zero, m)
            return m
        # 如果轴为负数，则转换为有效的轴索引
        if axis < 0:
            axis += 2
        # 如果轴为 0 或 1，则调用 _min_or_max_axis 函数进行处理
        if (axis == 0) or (axis == 1):
            return _min_or_max_axis(X, axis, min_or_max)
        else:
            # 若轴不合法，则抛出值错误异常
            raise ValueError("invalid axis, use 0 for rows, or 1 for columns")

    # 定义一个函数用于计算稀疏矩阵在指定轴上的最小值和最大值
    def _sparse_min_max(X, axis):
        # 分别调用 _sparse_min_or_max 函数计算最小值和最大值
        return (
            _sparse_min_or_max(X, axis, np.minimum),
            _sparse_min_or_max(X, axis, np.maximum),
        )

    # 定义一个函数用于计算稀疏矩阵在指定轴上的非 NaN 最小值和最大值
    def _sparse_nan_min_max(X, axis):
        # 分别调用 _sparse_min_or_max 函数计算非 NaN 最小值和最大值
        return (
            _sparse_min_or_max(X, axis, np.fmin),
            _sparse_min_or_max(X, axis, np.fmax),
        )
# 如果 NumPy 版本大于或等于 1.25.0，则导入 ComplexWarning 和 VisibleDeprecationWarning
# 否则，从 NumPy 直接导入 ComplexWarning 和 VisibleDeprecationWarning（类型：忽略类型，不要报警）
if np_version >= parse_version("1.25.0"):
    from numpy.exceptions import ComplexWarning, VisibleDeprecationWarning
else:
    from numpy import ComplexWarning, VisibleDeprecationWarning  # type: ignore  # noqa


# TODO: 当 Scipy 版本升级到 1.6 时移除此段代码
try:
    # 尝试导入 trapezoid 函数（类型：忽略类型，不要报警）
    from scipy.integrate import trapezoid  # type: ignore  # noqa
except ImportError:
    # 导入 trapz 函数作为 trapezoid（类型：忽略类型，不要报警）
    from scipy.integrate import trapz as trapezoid  # type: ignore  # noqa


# TODO: 当 Pandas 版本升级到 2.2 时适配此函数
def pd_fillna(pd, frame):
    # 解析 Pandas 版本号的基础版本
    pd_version = parse_version(pd.__version__).base_version
    if parse_version(pd_version) < parse_version("2.2"):
        # 如果 Pandas 版本小于 2.2，则使用 np.nan 填充缺失值
        frame = frame.fillna(value=np.nan)
    else:
        # 否则，根据 Pandas 版本调整 infer_objects 的参数
        infer_objects_kwargs = (
            {} if parse_version(pd_version) >= parse_version("3") else {"copy": False}
        )
        with pd.option_context("future.no_silent_downcasting", True):
            # 使用 np.nan 填充缺失值，并根据版本调整数据类型
            frame = frame.fillna(value=np.nan).infer_objects(**infer_objects_kwargs)
    return frame


# TODO: 当 SciPy 版本升级到 1.12 时移除此函数
def _preserve_dia_indices_dtype(
    sparse_container, original_container_format, requested_sparse_format
):
    """保留 SciPy < 1.12 版本在从 DIA 格式转换为 CSR/CSC 时的索引数据类型。

    对于 SciPy < 1.12，DIA 格式的索引被升级为 `np.int64`，这与 DIA 矩阵不一致。
    我们将索引数据类型降级为 `np.int32`，以保持一致性。

    转换后的索引数组会直接影响到稀疏容器本身。

    Parameters
    ----------
    sparse_container : 稀疏容器
        需要检查的稀疏容器。
    requested_sparse_format : str or bool
        `sparse_container` 的期望稀疏格式类型。

    Notes
    -----
    更多详情请参见 https://github.com/scipy/scipy/issues/19245
    """
    if original_container_format == "dia_array" and requested_sparse_format in (
        "csr",
        "coo",
    ):
        if requested_sparse_format == "csr":
            # 确定最小可接受的索引数据类型
            index_dtype = _smallest_admissible_index_dtype(
                arrays=(sparse_container.indptr, sparse_container.indices),
                maxval=max(sparse_container.nnz, sparse_container.shape[1]),
                check_contents=True,
            )
            # 将稀疏容器的 indices 数组数据类型转换为 index_dtype（在原地修改，不复制）
            sparse_container.indices = sparse_container.indices.astype(
                index_dtype, copy=False
            )
            # 将稀疏容器的 indptr 数组数据类型转换为 index_dtype（在原地修改，不复制）
            sparse_container.indptr = sparse_container.indptr.astype(
                index_dtype, copy=False
            )
        else:  # requested_sparse_format == "coo"
            # 确定最小可接受的索引数据类型
            index_dtype = _smallest_admissible_index_dtype(
                maxval=max(sparse_container.shape)
            )
            # 将稀疏容器的 row 和 col 数组数据类型转换为 index_dtype（在原地修改，不复制）
            sparse_container.row = sparse_container.row.astype(index_dtype, copy=False)
            sparse_container.col = sparse_container.col.astype(index_dtype, copy=False)
# TODO: 当 SciPy 版本升级到 1.12 以上时移除这段代码
def _smallest_admissible_index_dtype(arrays=(), maxval=None, check_contents=False):
    """根据输入的整数数组 `a`，确定一个适当的索引数据类型，能够容纳这些数组中的数据。

    如果需要根据 `maxval` 或者输入数组的最大精度来确定，这个函数会返回 `np.int64`；或者根据它们的内容（当 `check_contents` 为 True 时）来确定。如果以上条件都不需要 `np.int64`，则函数返回 `np.int32`。

    Parameters
    ----------
    arrays : ndarray 或者 ndarray 的元组，默认=()
        要检查类型/内容的输入数组。

    maxval : float，默认=None
        所需的最大值。

    check_contents : bool，默认=False
        是否检查数组中的值而不只是它们的类型。
        默认只检查类型。

    Returns
    -------
    dtype : {np.int32, np.int64}
        适合的索引数据类型（int32 或 int64）。
    """

    # 定义 int32 的最小值和最大值
    int32min = np.int32(np.iinfo(np.int32).min)
    int32max = np.int32(np.iinfo(np.int32).max)

    # 如果存在 maxval，根据其值来确定返回的数据类型
    if maxval is not None:
        if maxval > np.iinfo(np.int64).max:
            raise ValueError(
                f"maxval={maxval} 超出了 np.int64 能表示的范围."
            )
        if maxval > int32max:
            return np.int64

    # 如果 arrays 是单个 ndarray，转换为元组
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    # 遍历所有输入的数组
    for arr in arrays:
        # 检查数组类型是否为 np.ndarray
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"数组应为 np.ndarray 类型，而不是 {type(arr)}."
            )
        # 检查数组元素是否为整数类型
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"数组的 dtype {arr.dtype} 不支持作为索引数据类型。我们期望整数值."
            )
        # 检查是否能够将数组的 dtype 转换为 int32
        if not np.can_cast(arr.dtype, np.int32):
            if not check_contents:
                # 当 check_contents 为 False 时，保守起见返回 np.int64
                return np.int64
            if arr.size == 0:
                # 数组为空，暂时不需要更大的类型，继续下一个数组的检查
                continue
            else:
                maxval = arr.max()
                minval = arr.min()
                if minval < int32min or maxval > int32max:
                    # 实际上需要更大的索引类型
                    return np.int64

    # 如果以上条件都不满足，默认返回 int32
    return np.int32


# TODO: 当 Scipy 版本小于 1.12 时移除这段代码
if sp_version < parse_version("1.12"):
    from ..externals._scipy.sparse.csgraph import laplacian  # type: ignore  # noqa
else:
    from scipy.sparse.csgraph import laplacian  # type: ignore  # noqa  # pragma: no cover


# TODO: 当我们停止支持 Python 3.9 时移除这段代码。注意，filter 参数在 3.9.17 中已经回溯，但是我们不能假设微版本的情况，参见
# https://docs.python.org/3.9/library/tarfile.html#tarfile.TarFile.extractall
# for more details
# 定义一个函数用于解压 tar 文件到指定路径
def tarfile_extractall(tarfile, path):
    try:
        # 尝试使用指定的过滤器（filter）来解压 tar 文件到指定路径
        tarfile.extractall(path, filter="data")
    except TypeError:
        # 如果指定的过滤器参数不支持，则以默认方式解压 tar 文件到指定路径
        tarfile.extractall(path)


# 定义一个函数，用于检测是否处于不稳定的 OpenBLAS 配置中
def _in_unstable_openblas_configuration():
    """Return True if in an unstable configuration for OpenBLAS"""

    # 导入可能会加载 OpenBLAS 的库
    import numpy  # noqa
    import scipy  # noqa

    # 获取线程池控制器的信息
    modules_info = _get_threadpool_controller().info()

    # 检查是否使用了 OpenBLAS
    open_blas_used = any(info["internal_api"] == "openblas" for info in modules_info)
    if not open_blas_used:
        return False

    # OpenBLAS 0.3.16 修复了 arm64 的不稳定性问题，参见：
    # https://github.com/xianyi/OpenBLAS/blob/1b6db3dbba672b4f8af935bd43a1ff6cff4d20b7/Changelog.txt#L56-L58 # noqa
    openblas_arm64_stable_version = parse_version("0.3.16")
    for info in modules_info:
        if info["internal_api"] != "openblas":
            continue
        openblas_version = info.get("version")
        openblas_architecture = info.get("architecture")
        if openblas_version is None or openblas_architecture is None:
            # 无法确定 OpenBLAS 是否足够稳定，假定为不稳定状态：
            return True  # pragma: no cover
        if (
            openblas_architecture == "neoversen1"
            and parse_version(openblas_version) < openblas_arm64_stable_version
        ):
            # 参见 https://github.com/numpy/numpy/issues/19411 中的讨论
            return True  # pragma: no cover
    return False
```