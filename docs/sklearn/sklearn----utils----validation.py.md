# `D:\src\scipysrc\scikit-learn\sklearn\utils\validation.py`

```
# 导入所需的库和模块
import numbers  # 导入 numbers 模块，用于数字类型的判断
import operator  # 导入 operator 模块，用于操作符函数的支持
import sys  # 导入 sys 模块，提供对 Python 解释器的访问
import warnings  # 导入 warnings 模块，用于警告处理
from contextlib import suppress  # 导入 suppress 函数，用于上下文管理
from functools import reduce, wraps  # 导入 reduce 和 wraps 函数，用于函数式编程支持
from inspect import Parameter, isclass, signature  # 导入 Parameter、isclass 和 signature 函数，用于参数和签名处理

import joblib  # 导入 joblib 库，用于并行任务处理
import numpy as np  # 导入 NumPy 库，并重命名为 np，用于数值计算
import scipy.sparse as sp  # 导入 SciPy 库中的 sparse 模块，用于稀疏矩阵支持

from .. import get_config as _get_config  # 导入相对路径下的 get_config 函数
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning  # 导入自定义异常类
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace  # 导入数组操作相关函数
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype  # 导入修复函数和警告类
from ._isfinite import FiniteStatus, cy_isfinite  # 导入有限性检测函数
from .fixes import _object_dtype_isnan  # 导入特定修复函数

FLOAT_DTYPES = (np.float64, np.float32, np.float16)  # 定义浮点数类型的元组

# 这个函数在当前代码库中不再使用，但保留以防意外合并不带关键字参数的新公共函数，
# 这可能需要一个弃用周期来修复。
def _deprecate_positional_args(func=None, *, version="1.3"):
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    version : callable, default="1.3"
        The version when positional arguments will result in error.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []

        # 遍历函数签名中的参数，区分位置或关键字参数和仅关键字参数
        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            # 计算传递的额外位置参数个数
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0，生成警告信息
            args_msg = [
                "{}={}".format(name, arg)
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg = ", ".join(args_msg)
            warnings.warn(
                (
                    f"Pass {args_msg} as keyword args. From version "
                    f"{version} passing these as positional arguments "
                    "will result in an error"
                ),
                FutureWarning,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args


def _assert_all_finite(
    X, allow_nan=False, msg_dtype=None, estimator_name=None, input_name=""
):
    """Like assert_all_finite, but only for ndarray."""

    # 获取适当的命名空间和数组 API 类型（例如 numpy 或 cupy）
    xp, is_array_api = get_namespace(X)

    # 如果配置中假设输入都是有限的，则直接返回，无需检查
    if _get_config()["assume_finite"]:
        return

    # 将输入 X 转换为数组，以便后续处理
    X = xp.asarray(X)

    # 对于对象类型的数据，只检查是否包含 NaN（GH-13254）
    if not is_array_api and X.dtype == np.dtype("object") and not allow_nan:
        # 如果输入包含 NaN，则引发 ValueError 异常
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")

    # 只需考虑浮点数组，因此对于其他类型的数组，可以提前返回
    if not xp.isdtype(X.dtype, ("real floating", "complex floating")):
        return

    # 首先尝试对常见情况使用 O(n) 时间复杂度、O(1) 空间复杂度的解决方案，
    # 即所有元素都是有限的情况；如果不是，则采用 O(n) 空间复杂度的 `np.isinf/isnan` 或自定义 Cython 实现，
    # 以避免误报并提供详细的错误消息。
    with np.errstate(over="ignore"):
        first_pass_isfinite = xp.isfinite(xp.sum(X))
    
    # 如果第一次检查所有元素都是有限的，则直接返回
    if first_pass_isfinite:
        return

    # 否则，针对每个元素逐个检查是否有限，以提供详细的错误消息和检查
    _assert_all_finite_element_wise(
        X,
        xp=xp,
        allow_nan=allow_nan,
        msg_dtype=msg_dtype,
        estimator_name=estimator_name,
        input_name=input_name,
    )
# 如果使用 Cython 实现，则不支持 FP16 或复数类型数据
use_cython = (
    xp is np and X.data.contiguous and X.dtype.type in {np.float32, np.float64}
)

# 如果使用 Cython 实现，则调用 cy_isfinite 函数检查是否有 NaN 或无限大值
if use_cython:
    out = cy_isfinite(X.reshape(-1), allow_nan=allow_nan)
    # 如果不允许 NaN，则检查是否有 NaN，否则错误标志为 False
    has_nan_error = False if allow_nan else out == FiniteStatus.has_nan
    # 检查是否有无限大值
    has_inf = out == FiniteStatus.has_infinite
else:
    # 使用 NumPy 或其他库的方法检查是否有无限大值
    has_inf = xp.any(xp.isinf(X))
    # 如果不允许 NaN，则检查是否有 NaN，否则错误标志为 False
    has_nan_error = False if allow_nan else xp.any(xp.isnan(X))

# 如果有无限大值或者有 NaN 错误
if has_inf or has_nan_error:
    # 如果错误为 NaN
    if has_nan_error:
        type_err = "NaN"
    else:
        # 否则，根据数据类型构造错误消息
        msg_dtype = msg_dtype if msg_dtype is not None else X.dtype
        type_err = f"infinity or a value too large for {msg_dtype!r}"
    
    # 根据输入名称构造带有错误类型的错误消息
    padded_input_name = input_name + " " if input_name else ""
    msg_err = f"Input {padded_input_name}contains {type_err}."

    # 如果有估算器名称，并且输入名称为 "X"，并且有 NaN 错误
    if estimator_name and input_name == "X" and has_nan_error:
        # 改进错误消息，指导如何处理缺失值在 scikit-learn 中
        msg_err += (
            f"\n{estimator_name} does not accept missing values"
            " encoded as NaN natively. For supervised learning, you might want"
            " to consider sklearn.ensemble.HistGradientBoostingClassifier and"
            " Regressor which accept missing values encoded as NaNs natively."
            " Alternatively, it is possible to preprocess the data, for"
            " instance by using an imputer transformer in a pipeline or drop"
            " samples with missing values. See"
            " https://scikit-learn.org/stable/modules/impute.html"
            " You can find a list of all estimators that handle NaN values"
            " at the following page:"
            " https://scikit-learn.org/stable/modules/impute.html"
            "#estimators-that-handle-nan-values"
        )
    
    # 抛出值错误，并包含详细的错误消息
    raise ValueError(msg_err)
    # 使用断言检查数组是否包含有限值，如果出现非有限值会引发 ValueError 异常
    try:
        # 调用 _assert_all_finite 函数，检查数组 X 中的数据是否包含有限值
        _assert_all_finite(
            X.data if sp.issparse(X) else X,  # 如果 X 是稀疏矩阵，则使用 X.data，否则使用 X
            allow_nan=allow_nan,  # 是否允许 NaN 值
            estimator_name=estimator_name,  # 估计器名称，用于错误消息
            input_name=input_name,  # 输入数据的名称，用于错误消息
        )
        # 如果断言通过，则打印测试通过的消息
        print("Test passed: Array contains only finite values.")
    except ValueError:
        # 如果出现 ValueError 异常，则打印测试失败的消息，表示数组包含非有限值
        print("Test failed: Array contains non-finite values.")
# 将输入数组或稀疏矩阵转换为浮点数数组。

def as_float_array(X, *, copy=True, force_all_finite=True):
    """Convert an array-like to an array of floats.

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        The input data.

    copy : bool, default=True
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in X. The
        possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan and pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    Returns
    -------
    XT : {ndarray, sparse matrix}
        An array of type float.

    Examples
    --------
    >>> from sklearn.utils import as_float_array
    >>> import numpy as np
    >>> array = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    >>> as_float_array(array)
    array([0., 0., 1., 2., 2.])
    """

    # 如果 X 是 np.matrix 或者既不是 np.ndarray 也不是稀疏矩阵
    if isinstance(X, np.matrix) or (
        not isinstance(X, np.ndarray) and not sp.issparse(X)
    ):
        # 调用 check_array 函数，将 X 转换为指定类型的数组
        return check_array(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype=np.float64,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=False,
        )
    # 如果 X 是稀疏矩阵且数据类型是 np.float32 或 np.float64
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        # 如果 copy=True，返回 X 的深拷贝，否则返回 X 本身
        return X.copy() if copy else X
    # 如果 X 是 numpy 数组且数据类型是 np.float32 或 np.float64
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        # 如果 copy=True，按 F 或 C 格式拷贝 X，否则返回 X 本身
        return X.copy("F" if X.flags["F_CONTIGUOUS"] else "C") if copy else X
    else:
        # 如果 X 的数据类型属于 'uib' 类型且数据大小不超过 4 字节，则转换为 np.float32
        # 否则转换为 np.float64
        if X.dtype.kind in "uib" and X.dtype.itemsize <= 4:
            return_dtype = np.float32
        else:
            return_dtype = np.float64
        # 将 X 转换为指定类型的数组
        return X.astype(return_dtype)


# 返回输入是否为类数组对象的布尔值。

def _is_arraylike(x):
    """Returns whether the input is array-like."""
    # 如果 x 是稀疏矩阵，则返回 False
    if sp.issparse(x):
        return False

    # 否则，如果 x 具有 __len__ 属性、shape 属性或 __array__ 方法，则返回 True
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


# 返回数组是否是类数组对象且不是标量的布尔值。

def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    # 如果 array 是类数组对象且不是标量，则返回 True
    return _is_arraylike(array) and not np.isscalar(array)


# 对于不是 Pandas 数据框架且实现了数据框架协议的非 Pandas 数据框架，使用数据框架协议。

def _use_interchange_protocol(X):
    """Use interchange protocol for non-pandas dataframes that follow the protocol.

    Note: at this point we chose not to use the interchange API on pandas dataframe
    to ensure strict behavioral backward compatibility with older versions of
    scikit-learn.
    """
    # 如果 X 不是 Pandas 数据框架且具有 __dataframe__ 属性，则使用数据框架协议
    return not _is_pandas_df(X) and hasattr(X, "__dataframe__")


# 返回输入是否具有多个特征的布尔值。

def _num_features(X):
    """Return the number of features in an array-like X.

    This helper function tries hard to avoid to materialize an array version
    of X unless necessary. For instance, if X is a list of lists,
    this function will return the length of the first element, assuming
    that subsequent elements are all lists of the same length without
    checking.
    Parameters
    ----------
    X : array-like
        array-like to get the number of features.

    Returns
    -------
    features : int
        Number of features
    """
    # 获取参数 X 的类型
    type_ = type(X)
    # 检查类型是否为内置类型
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    # 构造错误消息模板
    message = f"Unable to find the number of features from X of type {type_name}"
    
    # 如果 X 没有 __len__ 方法也没有 shape 属性
    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        # 如果 X 没有 __array__ 方法，则抛出类型错误异常
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        # 只有在没有更便宜的启发式选项时才将 X 转换为 numpy 数组
        X = np.asarray(X)

    # 如果 X 有 shape 属性
    if hasattr(X, "shape"):
        # 如果 X.shape 没有 __len__ 方法或者长度小于等于 1，则抛出类型错误异常
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        # 返回 X 的第二维度长度，即特征数量
        return X.shape[1]

    # 获取 X 的第一个样本
    first_sample = X[0]

    # 如果第一个样本是字符串、字节或字典类型，则抛出类型错误异常
    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type {type(first_sample).__qualname__}"
        raise TypeError(message)

    try:
        # 如果 X 是列表的列表，则假设所有嵌套列表的长度相同，返回第一个样本的长度作为特征数量
        return len(first_sample)
    except Exception as err:
        # 捕获并重新抛出异常，附加错误消息
        raise TypeError(message) from err
# 返回数组样本数目的函数
def _num_samples(x):
    # 函数中用到的消息字符串，指明预期的输入类型
    message = "Expected sequence or array-like, got %s" % type(x)
    
    # 如果输入对象有 `fit` 方法且可调用，则不应从模型集合中获取样本数
    if hasattr(x, "fit") and callable(x.fit):
        raise TypeError(message)  # 抛出类型错误异常
    
    # 如果使用交换协议处理对象，则返回其数据框的行数
    if _use_interchange_protocol(x):
        return x.__dataframe__().num_rows()
    
    # 如果对象既没有 `__len__` 方法也没有 `shape` 属性
    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        # 如果对象有 `__array__` 方法，则转换为 NumPy 数组
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)  # 抛出类型错误异常
    
    # 如果对象有 `shape` 属性且其 `shape` 不为 `None`
    if hasattr(x, "shape") and x.shape is not None:
        # 如果对象是单元素数组，则抛出类型错误异常
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # 检查 `shape[0]` 是否返回整数值，否则默认使用 `len` 函数获取长度
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]
    
    try:
        return len(x)  # 返回对象的长度
    except TypeError as type_error:
        raise TypeError(message) from type_error  # 抛出类型错误异常


# 检查 `memory` 参数是否类似于 joblib.Memory 的函数
def check_memory(memory):
    """Check that ``memory`` is joblib.Memory-like.

    joblib.Memory-like means that ``memory`` can be converted into a
    joblib.Memory instance (typically a str denoting the ``location``)
    or has the same interface (has a ``cache`` method).

    Parameters
    ----------
    memory : None, str or object with the joblib.Memory interface
        - If string, the location where to create the `joblib.Memory` interface.
        - If None, no caching is done and the Memory object is completely transparent.

    Returns
    -------
    memory : object with the joblib.Memory interface
        A correct joblib.Memory object.

    Raises
    ------
    ValueError
        If ``memory`` is not joblib.Memory-like.

    Examples
    --------
    >>> from sklearn.utils.validation import check_memory
    >>> check_memory("caching_dir")
    Memory(location=caching_dir/joblib)
    """
    # 如果 `memory` 是 `None` 或者是字符串，则创建 `joblib.Memory` 对象
    if memory is None or isinstance(memory, str):
        memory = joblib.Memory(location=memory, verbose=0)
    # 否则，如果 `memory` 对象没有 `cache` 方法，则抛出值错误异常
    elif not hasattr(memory, "cache"):
        raise ValueError(
            "'memory' should be None, a string or have the same"
            " interface as joblib.Memory."
            " Got memory='{}' instead.".format(memory)
        )
    return memory  # 返回正确的 `joblib.Memory` 对象


# 检查所有数组是否具有一致的第一个维度长度的函数
def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.

    Examples
    --------
    >>> from sklearn.utils.validation import check_consistent_length
    >>> a = [1, 2, 3]
    >>> b = [2, 3, 4]
    >>> check_consistent_length(a, b)
    """
    # 使用 `_num_samples` 函数获取所有数组的长度
    lengths = [_num_samples(X) for X in arrays if X is not None]
    # 获取所有不同长度值
    uniques = np.unique(lengths)
    # 如果发现唯一值列表中的元素数量大于1，则抛出值错误异常
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )
def _make_indexable(iterable):
    """Ensure iterable supports indexing or convert to an indexable variant.

    Convert sparse matrices to csr and other non-indexable iterable to arrays.
    Let `None` and indexable objects (e.g. pandas dataframes) pass unchanged.

    Parameters
    ----------
    iterable : {list, dataframe, ndarray, sparse matrix} or None
        Object to be converted to an indexable iterable.
    """
    # 如果输入是稀疏矩阵，则转换为 csr 格式
    if sp.issparse(iterable):
        return iterable.tocsr()
    # 如果输入具有 __getitem__ 或 iloc 属性（例如 pandas 数据框），则直接返回
    elif hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    # 如果输入为 None，则直接返回
    elif iterable is None:
        return iterable
    # 否则将输入转换为 NumPy 数组并返回
    return np.array(iterable)


def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-iterable objects to arrays.

    Parameters
    ----------
    *iterables : {lists, dataframes, ndarrays, sparse matrices}
        List of objects to ensure sliceability.

    Returns
    -------
    result : list of {ndarray, sparse matrix, dataframe} or None
        Returns a list containing indexable arrays (i.e. NumPy array,
        sparse matrix, or dataframe) or `None`.

    Examples
    --------
    >>> from sklearn.utils import indexable
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> iterables = [
    ...     [1, 2, 3], np.array([2, 3, 4]), None, csr_matrix([[5], [6], [7]])
    ... ]
    >>> indexable(*iterables)
    [[1, 2, 3], array([2, 3, 4]), None, <3x1 sparse matrix ...>]
    """

    # 对每个输入的对象应用 _make_indexable 函数，确保它们是可索引的
    result = [_make_indexable(X) for X in iterables]
    # 检查所有结果的长度是否一致
    check_consistent_length(*result)
    # 返回处理后的结果列表
    return result


def _ensure_sparse_format(
    sparse_container,
    accept_sparse,
    dtype,
    copy,
    force_all_finite,
    accept_large_sparse,
    estimator_name=None,
    input_name="",
):
    """Convert a sparse container to a given format.

    Checks the sparse format of `sparse_container` and converts if necessary.

    Parameters
    ----------
    sparse_container : sparse matrix or array
        Input to validate and convert.

    accept_sparse : str, bool or list/tuple of str
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : str, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : bool
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : bool
        Whether to raise an error on np.inf, np.nan, pd.NA or pd.nan.

    accept_large_sparse : bool
        Whether to accept large sparse inputs, such as those with a large
        number of non-zero elements.

    estimator_name : str or None
        Name of the estimator or function calling this function.

    input_name : str
        Name of the input.

    """
    # 稀疏容器的格式检查和必要的转换
    # 根据 accept_sparse 参数确定允许的稀疏矩阵格式，进行必要的转换
    pass  # 省略函数体的详细解释，因为后续代码可能会继续添加更多功能
    # 如果未指定 dtype，则使用 sparse_container 的 dtype
    if dtype is None:
        dtype = sparse_container.dtype

    # 初始化变量，用于记录是否更改了 sparse_container 的格式
    changed_format = False

    # 获取 sparse_container 的类型名称
    sparse_container_type_name = type(sparse_container).__name__

    # 如果 accept_sparse 是字符串，则转换为列表
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # 检查 sparse_container 是否是大型稀疏矩阵，进行相关验证
    _check_large_sparse(sparse_container, accept_large_sparse)

    # 如果不接受稀疏数据，抛出类型错误
    if accept_sparse is False:
        padded_input = " for " + input_name if input_name else ""
        raise TypeError(
            f"Sparse data was passed{padded_input}, but dense data is required. "
            "Use '.toarray()' to convert to a dense numpy array."
        )
    # 如果 accept_sparse 是列表或元组
    elif isinstance(accept_sparse, (list, tuple)):
        # 如果列表或元组为空，则引发值错误
        if len(accept_sparse) == 0:
            raise ValueError(
                "When providing 'accept_sparse' as a tuple or list, it must contain at "
                "least one string value."
            )
        # 确保 sparse_container 使用正确的稀疏格式
        if sparse_container.format not in accept_sparse:
            # 将 sparse_container 转换为指定格式
            sparse_container = sparse_container.asformat(accept_sparse[0])
            changed_format = True
    # 如果 accept_sparse 不是 True，也不是列表或元组，则引发值错误
    elif accept_sparse is not True:
        raise ValueError(
            "Parameter 'accept_sparse' should be a string, boolean or list of strings."
            f" You provided 'accept_sparse={accept_sparse}'."
        )

    # 如果指定了 dtype 并且与 sparse_container 的 dtype 不同，则进行类型转换
    if dtype != sparse_container.dtype:
        sparse_container = sparse_container.astype(dtype)
    # 如果需要复制 sparse_container 并且未更改格式，则执行复制操作
    elif copy and not changed_format:
        sparse_container = sparse_container.copy()
    # 如果 force_all_finite 为真，则进行下列操作
    if force_all_finite:
        # 如果 sparse_container 没有 "data" 属性
        if not hasattr(sparse_container, "data"):
            # 发出警告，指出无法检查格式为 sparse_container.format 的稀疏矩阵中的 NaN 或 Inf
            warnings.warn(
                f"Can't check {sparse_container.format} sparse matrix for nan or inf.",
                stacklevel=2,
            )
        else:
            # 否则，调用 _assert_all_finite 函数，检查 sparse_container.data 中的数据
            _assert_all_finite(
                sparse_container.data,
                allow_nan=force_all_finite == "allow-nan",
                estimator_name=estimator_name,
                input_name=input_name,
            )

    # TODO: Remove when the minimum version of SciPy supported is 1.12
    # 当支持的 SciPy 最小版本为 1.12 时，移除以下内容
    # 对于 SciPy 稀疏数组，从 DIA 格式转换为 COO、CSR 或 BSR 格式时，
    # 即使数据适合使用 `np.int32` 索引，也会触发使用 `np.int64` 索引。
    # https://github.com/scipy/scipy/issues/19245 由于并非所有 scikit-learn
    # 算法都支持大索引，以下代码在安全的情况下将索引降级为 `np.int32`。
    if changed_format:
        # 如果发生了格式转换，则按指定格式接受稀疏格式
        requested_sparse_format = accept_sparse[0]
        # 调用 _preserve_dia_indices_dtype 函数，保留 DIA 格式的索引数据类型
        _preserve_dia_indices_dtype(
            sparse_container, sparse_container_type_name, requested_sparse_format
        )

    # 返回稀疏矩阵或容器
    return sparse_container
# 确保数组中不包含复杂数据，如果包含则抛出 ValueError 异常
def _ensure_no_complex_data(array):
    if (
        hasattr(array, "dtype")                       # 检查数组是否有 dtype 属性
        and array.dtype is not None                   # 确保 dtype 不为 None
        and hasattr(array.dtype, "kind")              # 检查 dtype 是否有 kind 属性
        and array.dtype.kind == "c"                   # 判断 dtype 的 kind 是否为 'c'（复数）
    ):
        raise ValueError("Complex data not supported\n{}\n".format(array))


# 检查估计器的名称，如果是字符串直接返回，否则返回其类名，若为 None 则返回 None
def _check_estimator_name(estimator):
    if estimator is not None:                         # 如果估计器不为 None
        if isinstance(estimator, str):                # 如果估计器是字符串类型
            return estimator                          # 直接返回估计器本身
        else:
            return estimator.__class__.__name__        # 返回估计器的类名
    return None                                       # 若估计器为 None，则返回 None


# 检查 pandas 的数据类型是否需要早期转换
def _pandas_dtype_needs_early_conversion(pd_dtype):
    """Return True if pandas extension pd_dtype need to be converted early."""
    # 提前检查 pandas 扩展类型 pd_dtype 是否需要早期转换

    # 导入必要的 pandas 类型和函数
    from pandas import SparseDtype
    from pandas.api.types import (
        is_bool_dtype,
        is_float_dtype,
        is_integer_dtype,
    )

    if is_bool_dtype(pd_dtype):
        # 布尔类型和扩展布尔类型需要早期转换，因为 __array__ 方法将混合类型的 DataFrame 转换为对象类型
        return True

    if isinstance(pd_dtype, SparseDtype):
        # 稀疏数组将在 `check_array` 中后续转换
        return False

    try:
        from pandas.api.types import is_extension_array_dtype
    except ImportError:
        return False

    if isinstance(pd_dtype, SparseDtype) or not is_extension_array_dtype(pd_dtype):
        # 稀疏数组将在 `check_array` 中后续转换
        # 仅处理整数和浮点数的扩展数组
        return False
    elif is_float_dtype(pd_dtype):
        # 浮点数 ndarray 通常支持 NaN。需要先转换以将 pd.NA 映射到 np.nan
        return True
    elif is_integer_dtype(pd_dtype):
        # XXX: 转换高整数到浮点数时发出警告
        return True

    return False


# 判断数组是否为 pandas 扩展数组类型
def _is_extension_array_dtype(array):
    # Pandas 扩展数组具有带有 na_value 的 dtype
    return hasattr(array, "dtype") and hasattr(array.dtype, "na_value")


# 对数组、列表、稀疏矩阵或类似对象进行输入验证
def check_array(
    array,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_writeable=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
    input_name="",
):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.
    """
    # 接受稀疏矩阵的设置，可以是字符串、布尔值或字符串列表/元组，默认为False
    accept_sparse : str, bool or list/tuple of str, default=False
        # 允许的稀疏矩阵格式，如 'csc'、'csr' 等。如果输入是稀疏的但不在允许的格式列表中，
        # 将被转换为列表中的第一个格式。True 表示接受任何格式。False 表示如果输入是稀疏矩阵，
        # 但不在允许的格式列表中，则会引发错误。

    # 是否接受大型稀疏矩阵，默认为True
    accept_large_sparse : bool, default=True
        # 如果输入是 CSR、CSC、COO 或 BSR 格式的稀疏矩阵，并且被 accept_sparse 接受，
        # 当 accept_large_sparse=False 时，只有当其索引以32位 dtype 存储时才会被接受。

        .. versionadded:: 0.20

    # 结果的数据类型设置。可以是 'numeric'、类型、类型列表或None，默认为 'numeric'
    dtype : 'numeric', type, list of type or None, default='numeric'
        # 结果的数据类型。如果为 None，则保留输入的 dtype。如果为 "numeric"，则除非数组的 dtype 是 object，
        # 否则保留输入的 dtype。如果 dtype 是类型列表，则仅在输入的 dtype 不在列表中时，才对第一个类型进行转换。

    # 数组的内存布局设置。可以是 {'F', 'C'} 或 None，默认为 None
    order : {'F', 'C'} or None, default=None
        # 是否强制数组的内存布局为 Fortran 风格或 C 风格。
        # 当 order 为 None（默认）时，如果 copy=False，则不确保返回的数组的内存布局；
        # 否则（copy=True），尽量保持返回的数组的内存布局与原始数组尽可能接近。

    # 是否触发强制复制。默认为 False
    copy : bool, default=False
        # 是否会触发强制复制。如果 copy=False，则可能会由转换触发复制。

    # 是否强制返回的数组可写。默认为 False
    force_writeable : bool, default=False
        # 是否强制返回的数组可写。如果为 True，则保证返回的数组可写，这可能需要进行复制。
        # 否则，保持输入数组的可写性。

        .. versionadded:: 1.6

    # 是否强制数组中所有元素为有限值。默认为 True
    force_all_finite : bool or 'allow-nan', default=True
        # 是否在数组中出现 np.inf、np.nan、pd.NA 时引发错误。选项包括：

        - True: 强制数组的所有值为有限值。
        - False: 接受数组中的 np.inf、np.nan、pd.NA。
        - 'allow-nan': 仅接受数组中的 np.nan 和 pd.NA 值，不允许无穷大值。

        .. versionadded:: 0.20
           ``force_all_finite`` 支持字符串 ``'allow-nan'``。

        .. versionchanged:: 0.23
           接受 `pd.NA` 并将其转换为 `np.nan`

    # 是否要求数组是二维的。默认为 True
    ensure_2d : bool, default=True
        # 是否在数组不是二维时引发值错误。

    # 是否允许数组的维数大于2。默认为 False
    allow_nd : bool, default=False
        # 是否允许数组的维数大于2。

    # 确保数组在第一轴（2D 数组的行）上具有最小数量的样本。默认为 1
    ensure_min_samples : int, default=1
        # 确保数组在其第一轴（2D 数组的行）上至少有指定数量的样本。
        # 将其设置为 0 可禁用此检查。
    # 检查是否为 np.matrix 类型，如果是则抛出类型错误，建议使用 np.asarray 转换成 numpy 数组
    if isinstance(array, np.matrix):
        raise TypeError(
            "np.matrix is not supported. Please convert to a numpy array with "
            "np.asarray. For more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"
        )

    # 获取数据数组的命名空间和其是否符合数组 API 的标准
    xp, is_array_api_compliant = get_namespace(array)

    # 存储原始数组的引用，以便在函数返回时检查是否需要复制
    array_orig = array

    # 存储原始的 dtype 是否为 numeric 类型
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    # 获取数组的原始 dtype
    dtype_orig = getattr(array, "dtype", None)
    if not is_array_api_compliant and not hasattr(dtype_orig, "kind"):
        # 如果不符合数组 API 标准且 dtype_orig 没有 kind 属性（例如在 pandas DataFrame 中的 dtype 列）
        dtype_orig = None

    # 检查对象是否包含多种 dtype（通常是 pandas DataFrame），并存储它们；如果不是，存储为 None
    dtypes_orig = None
    pandas_requires_conversion = False
    # 跟踪是否有类似 Series 的对象，以提供更好的错误信息
    type_if_series = None
    # 检查是否具有 'dtypes' 属性和 '__array__' 方法
    if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
        # 如果列是稀疏的，抛出警告。如果所有列都是稀疏的，那么array.sparse 存在且后续会保留稀疏性。
        with suppress(ImportError):
            from pandas import SparseDtype

            # 检查数据类型是否为稀疏类型
            def is_sparse(dtype):
                return isinstance(dtype, SparseDtype)

            # 如果 array 没有 'sparse' 属性且存在任何稀疏数据类型的列，发出警告
            if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
                warnings.warn(
                    "pandas.DataFrame with sparse columns found."
                    "It will be converted to a dense numpy array."
                )

        # 保存原始的数据类型列表
        dtypes_orig = list(array.dtypes)
        # 检查是否有列需要早期转换为 numpy 类型
        pandas_requires_conversion = any(
            _pandas_dtype_needs_early_conversion(i) for i in dtypes_orig
        )
        # 如果所有的 dtype 都是 np.dtype 类型
        if all(isinstance(dtype_iter, np.dtype) for dtype_iter in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)
        # 如果需要早期转换并且存在 dtype 为 object 类型的列，强制为 object 类型
        elif pandas_requires_conversion and any(d == object for d in dtypes_orig):
            dtype_orig = object

    # 如果 array 是扩展数组类型或具有 'iloc' 属性且具有 'dtype' 属性
    elif (_is_extension_array_dtype(array) or hasattr(array, "iloc")) and hasattr(
        array, "dtype"
    ):
        # array 是 pandas Series
        type_if_series = type(array)
        # 检查是否需要早期转换
        pandas_requires_conversion = _pandas_dtype_needs_early_conversion(array.dtype)
        # 如果 array.dtype 是 np.dtype 类型
        if isinstance(array.dtype, np.dtype):
            dtype_orig = array.dtype
        else:
            # 设置为 None，以便让 array.astype 找到最佳的 dtype
            dtype_orig = None

    # 如果 dtype_numeric 为 True
    if dtype_numeric:
        # 如果 dtype_orig 不为 None 并且具有 'kind' 属性且其 kind 为 'O'
        if (
            dtype_orig is not None
            and hasattr(dtype_orig, "kind")
            and dtype_orig.kind == "O"
        ):
            # 如果输入类型为 object，转换为 float
            dtype = xp.float64
        else:
            dtype = None

    # 如果 dtype 是 list 或 tuple 类型
    if isinstance(dtype, (list, tuple)):
        # 如果 dtype_orig 不为 None 并且 dtype 中包含 dtype_orig
        if dtype_orig is not None and dtype_orig in dtype:
            # 不需要进行 dtype 转换
            dtype = None
        else:
            # 需要进行 dtype 转换。选择接受类型列表的第一个元素
            dtype = dtype[0]

    # 如果 pandas_requires_conversion 为 True
    if pandas_requires_conversion:
        # pandas DataFrame 需要早期转换以处理扩展数据类型和 NaN
        # 如果 dtype 为 None，则使用原始 dtype 进行转换
        new_dtype = dtype_orig if dtype is None else dtype
        array = array.astype(new_dtype)
        # 由于我们已经在这里进行了转换，因此后续不需要再次转换

    # 如果 force_all_finite 不是 True、False 或 "allow-nan" 中的任何一个，抛出 ValueError
    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError(
            'force_all_finite should be a bool or "allow-nan". Got {!r} instead'.format(
                force_all_finite
            )
        )
    if dtype is not None and _is_numpy_namespace(xp):
        # 如果指定了dtype并且xp是numpy命名空间，则将dtype转换为dtype对象，以符合Array API的要求，稍后将使用`xp.isdtype`

    estimator_name = _check_estimator_name(estimator)
    # 检查评估器的名称，返回规范化的名称字符串
    context = " by %s" % estimator_name if estimator is not None else ""
    # 如果评估器不是None，则设置上下文字符串为" by {estimator_name}"，否则为空字符串

    # 当所有数据框列都是稀疏时，转换为稀疏数组
    if hasattr(array, "sparse") and array.ndim > 1:
        # 检查数组是否具有稀疏属性并且维度大于1
        with suppress(ImportError):
            from pandas import SparseDtype  # noqa: F811
            # 导入稀疏数据类型，忽略ImportError异常

            def is_sparse(dtype):
                return isinstance(dtype, SparseDtype)
                # 判断dtype是否为SparseDtype类型的实例的辅助函数

            if array.dtypes.apply(is_sparse).all():
                # 如果所有列的数据类型都是稀疏类型
                # DataFrame.sparse仅支持`to_coo`
                array = array.sparse.to_coo()
                # 转换DataFrame为稀疏COO格式
                if array.dtype == np.dtype("object"):
                    unique_dtypes = set([dt.subtype.name for dt in array_orig.dtypes])
                    # 获取原始数据框中所有列的数据类型的子类型名称集合
                    if len(unique_dtypes) > 1:
                        raise ValueError(
                            "Pandas DataFrame with mixed sparse extension arrays "
                            "generated a sparse matrix with object dtype which "
                            "can not be converted to a scipy sparse matrix."
                            "Sparse extension arrays should all have the same "
                            "numeric type."
                        )
                        # 如果子类型名称集合长度大于1，则抛出数值错误异常

    if sp.issparse(array):
        # 如果数组是稀疏数组
        _ensure_no_complex_data(array)
        # 确保数组中没有复杂数据
        array = _ensure_sparse_format(
            array,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            accept_large_sparse=accept_large_sparse,
            estimator_name=estimator_name,
            input_name=input_name,
        )
        # 确保数组符合稀疏格式要求，传递给_ensure_sparse_format函数的参数包括接受稀疏数据、数据类型、是否复制等
        if ensure_2d and array.ndim < 2:
            raise ValueError(
                f"Expected 2D input, got input with shape {array.shape}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample."
            )
            # 如果确保是2维数组但实际维度小于2，则引发数值错误异常

    if ensure_min_samples > 0:
        # 如果确保最小样本数大于0
        n_samples = _num_samples(array)
        # 获取数组的样本数
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required%s."
                % (n_samples, array.shape, ensure_min_samples, context)
            )
            # 如果样本数小于所需最小样本数，则引发数值错误异常

    if ensure_min_features > 0 and array.ndim == 2:
        # 如果确保最小特征数大于0且数组是二维的
        n_features = array.shape[1]
        # 获取数组的特征数
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required%s."
                % (n_features, array.shape, ensure_min_features, context)
            )
            # 如果特征数小于所需最小特征数，则引发数值错误异常
    # 如果 force_writeable 参数为真
    if force_writeable:
        # 默认情况下，array.copy() 创建一个 C 顺序的副本。我们设置 order='K' 来保持数组的顺序。
        copy_params = {"order": "K"} if not sp.issparse(array) else {}

        # 如果 array 是稀疏矩阵，array_data 就是 array 的数据部分；否则 array_data 就是 array 本身
        array_data = array.data if sp.issparse(array) else array
        # 获取 array_data 的 flags 属性，如果没有则为 None
        flags = getattr(array_data, "flags", None)
        # 如果 flags 存在且 writeable 属性为 False
        if not getattr(flags, "writeable", True):
            # 这种情况只会发生在 copy=False 且 array 是只读的情况下，同时请求可写的输出。
            # 这是一个模棱两可的设置，因此我们选择总是（除非特定设置，见下文）做一个副本，
            # 以确保输出是可写的，即使可以避免，也不会意外覆盖用户的数据。

            # 如果 array_orig 是 pandas 的 DataFrame 或 Series
            if _is_pandas_df_or_series(array_orig):
                try:
                    # 在 pandas >= 3 中，np.asarray(df) 在 check_array 中调用时，
                    # 返回一个只读的中间数组。它可以安全地设置为可写，而不需要复制，
                    # 因为如果原始 DataFrame 是由只读数组支持的，尝试更改标志将会引发错误，
                    # 在这种情况下，我们会进行复制。
                    array_data.flags.writeable = True
                except ValueError:
                    # 如果无法设置为可写，则进行复制操作
                    array = array.copy(**copy_params)
            else:
                # 对于其他类型的 array，进行复制操作
                array = array.copy(**copy_params)

    # 返回处理后的 array
    return array
# 如果 accept_large_sparse 参数为 False 并且 X 的索引是 64 位整数，则引发 ValueError 异常
def _check_large_sparse(X, accept_large_sparse=False):
    if not accept_large_sparse:
        # 支持的索引数据类型列表
        supported_indices = ["int32"]
        # 根据 X 的格式确定索引的键值
        if X.format == "coo":
            index_keys = ["col", "row"]  # COO 格式的列和行索引
        elif X.format in ["csr", "csc", "bsr"]:
            index_keys = ["indices", "indptr"]  # CSR, CSC, BSR 格式的索引和指针
        else:
            return  # 如果格式不在支持列表中，则直接返回
        # 遍历索引键列表
        for key in index_keys:
            # 获取当前索引的数据类型
            indices_datatype = getattr(X, key).dtype
            # 如果数据类型不在支持的列表中，则引发 ValueError 异常
            if indices_datatype not in supported_indices:
                raise ValueError(
                    "Only sparse matrices with 32-bit integer indices are accepted."
                    f" Got {indices_datatype} indices. Please do report a minimal"
                    " reproducer on scikit-learn issue tracker so that support for"
                    " your use-case can be studied by maintainers. See:"
                    " https://scikit-learn.org/dev/developers/minimal_reproducer.html"
                )


# 输入验证函数，用于标准估算器
def check_X_y(
    X,
    y,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_writeable=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    multi_output=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    y_numeric=False,
    estimator=None,
):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : {ndarray, list, sparse matrix}
        Input data.

    y : {ndarray, list, sparse matrix}
        Labels.

    accept_sparse : str, bool or list of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20
    """
    # 数据类型，可以是'numeric'、type、type列表或None，默认为'numeric'
    dtype : 'numeric', type, list of type or None, default='numeric'
        # 结果的数据类型。如果为None，则保留输入的dtype。
        # 如果为"numeric"，除非array.dtype为object，否则保留dtype。
        # 如果dtype是类型列表，则仅在输入的dtype不在列表中时才执行第一个类型的转换。

    # 排序方式，可以是{'F', 'C'}，默认为None
    order : {'F', 'C'}, default=None
        # 是否强制数组为Fortran或C风格。如果为`None`，则在可能的情况下保留输入数据的顺序。

    # 是否触发强制复制的标志，布尔值，默认为False
    copy : bool, default=False
        # 是否会触发强制复制。如果copy=False，则可能通过转换触发复制。

    # 是否强制输出数组可写的标志，布尔值，默认为False
    force_writeable : bool, default=False
        # 是否强制输出数组可写。如果为True，则返回的数组保证可写，这可能需要进行复制。
        # 否则保持输入数组的可写性。

        # .. versionadded:: 1.6

    # 是否强制所有元素为有限数的标志，布尔值或'allow-nan'，默认为True
    force_all_finite : bool or 'allow-nan', default=True
        # 是否在X中出现np.inf、np.nan、pd.NA时引发错误。该参数不影响y是否可以具有np.inf、np.nan、pd.NA值。
        # 可选值包括：

        # - True: 强制X的所有值为有限数。
        # - False: 接受X中的np.inf、np.nan、pd.NA。
        # - 'allow-nan': 仅接受X中的np.nan或pd.NA值。值不能是无穷大。

        # .. versionadded:: 0.20
        #    ``force_all_finite`` 支持字符串 ``'allow-nan'``.

        # .. versionchanged:: 0.23
        #    接受 `pd.NA` 并将其转换为 `np.nan`

    # 是否确保X是二维数组的标志，布尔值，默认为True
    ensure_2d : bool, default=True
        # 是否在X不是2D时引发值错误。

    # 是否允许X.ndim > 2的标志，布尔值，默认为False
    allow_nd : bool, default=False
        # 是否允许X.ndim > 2。

    # 是否允许y是多输出的标志，布尔值，默认为False
    multi_output : bool, default=False
        # 是否允许y是2D数组或稀疏矩阵。如果为False，则验证y为向量。如果multi_output=True，则y不能有np.nan或np.inf值。

    # 确保X的最小样本数，整数，默认为1
    ensure_min_samples : int, default=1
        # 确保X在其第一轴（2D数组的行）中具有最小样本数。

    # 确保2D数组具有最小特征数，整数，默认为1
    ensure_min_features : int, default=1
        # 确保2D数组具有最小特征数（列）。默认值为1拒绝空数据集。
        # 当X有效为2维或原始为1维且 ``ensure_2d`` 为True 时，才执行此检查。设置为0将禁用此检查。

    # 是否确保y具有数值类型的标志，布尔值，默认为False
    y_numeric : bool, default=False
        # 是否确保y具有数值类型。如果y的dtype为object，则将其转换为float64。仅应用于回归算法使用。

    # 评估器的名称或评估器实例，字符串或评估器实例，默认为None
    estimator : str or estimator instance, default=None
        # 如果提供，包含评估器名称在警告消息中。

    # 返回
    # -------
    X_converted : object
        # 转换和验证后的X。

    y_converted : object
        # 转换和验证后的y。

    # 示例
    """
    检查目标变量 y 是否为 None，如果是则抛出 ValueError 异常
    """
    if y is None:
        # 如果未提供 estimator，则将 estimator_name 设为 "estimator"，否则获取 estimator 的名称
        if estimator is None:
            estimator_name = "estimator"
        else:
            estimator_name = _check_estimator_name(estimator)
        
        # 抛出值错误异常，指明缺少目标变量 y
        raise ValueError(
            f"{estimator_name} requires y to be passed, but the target y is None"
        )

    # 检查并转换输入特征矩阵 X，确保其符合预期的格式和属性
    X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_writeable=force_writeable,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
        input_name="X",
    )

    # 检查并转换目标变量 y，确保其符合预期的格式和属性
    y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)

    # 检查特征矩阵 X 和目标变量 y 的长度是否一致
    check_consistent_length(X, y)

    # 返回经过检查和转换后的特征矩阵 X 和目标变量 y
    return X, y
# 将参数 `y` 转换为合适的数组格式，以便后续处理
def _check_y(y, multi_output=False, y_numeric=False, estimator=None):
    """Isolated part of check_X_y dedicated to y validation"""
    # 如果 `multi_output` 为 True，对 `y` 进行验证和转换
    if multi_output:
        # 使用 `check_array` 函数检查 `y`，确保其接受稀疏矩阵 `csr` 格式，
        # 并强制要求所有元素有限，不指定数据类型，指定输入名称为 "y"，并传递估计器
        y = check_array(
            y,
            accept_sparse="csr",
            force_all_finite=True,
            ensure_2d=False,
            dtype=None,
            input_name="y",
            estimator=estimator,
        )
    else:
        # 获取估计器的名称
        estimator_name = _check_estimator_name(estimator)
        # 将 `y` 转换为一维数组或列向量，如果存在警告则发出警告
        y = column_or_1d(y, warn=True)
        # 断言 `y` 中所有元素都是有限的，指定输入名称为 "y"，并传递估计器的名称
        _assert_all_finite(y, input_name="y", estimator_name=estimator_name)
        # 确保 `y` 中没有复数数据
        _ensure_no_complex_data(y)
    # 如果要求 `y` 是数值类型，并且 `y` 的数据类型为对象类型，则将其转换为 `np.float64`
    if y_numeric and hasattr(y.dtype, "kind") and y.dtype.kind == "O":
        y = y.astype(np.float64)

    return y


def column_or_1d(y, *, dtype=None, warn=False):
    """Ravel column or 1d numpy array, else raises an error.

    Parameters
    ----------
    y : array-like
       Input data.

    dtype : data-type, default=None
        Data type for `y`.

        .. versionadded:: 1.2

    warn : bool, default=False
       To control display of warnings.

    Returns
    -------
    y : ndarray
       Output data.

    Raises
    ------
    ValueError
        If `y` is not a 1D array or a 2D array with a single row or column.

    Examples
    --------
    >>> from sklearn.utils.validation import column_or_1d
    >>> column_or_1d([1, 1])
    array([1, 1])
    """
    # 获取 `y` 的命名空间及其估计器
    xp, _ = get_namespace(y)
    # 检查 `y`，确保其不是二维数组，允许使用指定的数据类型，指定输入名称为 "y"，不强制所有样本有限
    # 并确保至少有 0 个样本
    y = check_array(
        y,
        ensure_2d=False,
        dtype=dtype,
        input_name="y",
        force_all_finite=False,
        ensure_min_samples=0,
    )

    shape = y.shape
    # 如果 `y` 是一维数组，则将其展平并按照 C 风格存储顺序返回
    if len(shape) == 1:
        return _asarray_with_order(xp.reshape(y, (-1,)), order="C", xp=xp)
    # 如果 `y` 是二维数组且第二维度长度为 1，则按照 C 风格存储顺序返回，并可能发出警告
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples, ), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )
        return _asarray_with_order(xp.reshape(y, (-1,)), order="C", xp=xp)

    # 如果 `y` 不符合上述条件，则引发 ValueError 异常，说明 `y` 应为一维数组
    raise ValueError(
        "y should be a 1d array, got an array of shape {} instead.".format(shape)
    )


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)

    """
    # 将 `seed` 转换为一个 `np.random.RandomState` 实例
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("Invalid seed: must be None, int, or np.random.RandomState instance")
    # 如果种子（seed）为 None 或者是 np.random 本身，则返回全局的随机数生成器
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    
    # 如果种子是整数类型，则使用该种子创建一个新的随机数生成器对象并返回
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    
    # 如果种子已经是 np.random.RandomState 类型的对象，则直接返回该对象
    if isinstance(seed, np.random.RandomState):
        return seed
    
    # 如果种子不符合以上任何条件，则抛出 ValueError 异常，说明种子无法用于初始化一个 numpy.random.RandomState 实例
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
# 检查估算器的fit方法是否支持给定的参数。

def has_fit_parameter(estimator, parameter):
    """Check whether the estimator's fit method supports the given parameter.

    Parameters
    ----------
    estimator : object
        An estimator to inspect.

    parameter : str
        The searched parameter.

    Returns
    -------
    is_parameter : bool
        Whether the parameter was found to be a named parameter of the
        estimator's fit method.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.utils.validation import has_fit_parameter
    >>> has_fit_parameter(SVC(), "sample_weight")
    True
    """
    # 获取估算器的fit方法的参数签名
    return parameter in signature(estimator.fit).parameters


# 确保数组是二维、方阵且对称的。

def check_symmetric(array, *, tol=1e-10, raise_warning=True, raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : {ndarray, sparse matrix}
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.

    tol : float, default=1e-10
        Absolute tolerance for equivalence of arrays. Default = 1E-10.

    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.

    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : {ndarray, sparse matrix}
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.validation import check_symmetric
    >>> symmetric_array = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    >>> check_symmetric(symmetric_array)
    array([[0, 1, 2],
           [1, 0, 1],
           [2, 1, 0]])
    >>> from scipy.sparse import csr_matrix
    >>> sparse_symmetric_array = csr_matrix(symmetric_array)
    >>> check_symmetric(sparse_symmetric_array)
    <3x3 sparse matrix of type '<class 'numpy.int64'>'
        with 6 stored elements in Compressed Sparse Row format>
    """
    # 检查数组维度和形状，确保是二维方阵
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError(
            "array must be 2-dimensional and square. shape = {0}".format(array.shape)
        )

    # 如果数组是稀疏矩阵
    if sp.issparse(array):
        # 计算数组与其转置的差异
        diff = array - array.T
        # 只有csr、csc和coo格式的稀疏矩阵有`data`属性
        if diff.format not in ["csr", "csc", "coo"]:
            diff = diff.tocsr()
        # 检查差异数据的绝对值是否都小于给定的容差tol
        symmetric = np.all(abs(diff.data) < tol)
    else:
        # 检查数组是否在给定容差范围内与其转置相等
        symmetric = np.allclose(array, array.T, atol=tol)

    # 返回对称化后的数组或原始数组，具体取决于是否对称
    return symmetric if not raise_warning else array if symmetric else (
        warnings.warn("Array is not symmetric", stacklevel=2) if raise_warning else array + array.T - np.diag(array.diagonal())
        NotImplementedError
    # 如果数组不是对称的情况下执行以下操作
    if not symmetric:
        # 如果设置了 raise_exception 参数为 True，则抛出值错误异常
        if raise_exception:
            raise ValueError("Array must be symmetric")
        
        # 如果设置了 raise_warning 参数为 True，则发出警告
        if raise_warning:
            warnings.warn(
                (
                    "Array is not symmetric, and will be converted "
                    "to symmetric by average with its transpose."
                ),
                stacklevel=2,
            )
        
        # 如果数组是稀疏矩阵（使用 scipy 库检查），进行转换操作
        if sp.issparse(array):
            # 根据数组的格式，设置转换名称
            conversion = "to" + array.format
            # 计算数组与其转置矩阵的平均值，并转换成指定格式的稀疏矩阵
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            # 如果数组是普通数组，计算其与转置矩阵的平均值
            array = 0.5 * (array + array.T)

    # 返回处理后的数组
    return array
def _is_fitted(estimator, attributes=None, all_or_any=all):
    """Determine if an estimator is fitted

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    fitted : bool
        Whether the estimator is fitted.
    """
    # 如果给定了attributes参数
    if attributes is not None:
        # 如果attributes不是列表或元组，转换为列表
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        # 检查estimator是否具有所有指定的属性
        return all_or_any([hasattr(estimator, attr) for attr in attributes])

    # 如果estimator具有特定的内部方法__sklearn_is_fitted__
    if hasattr(estimator, "__sklearn_is_fitted__"):
        # 调用__sklearn_is_fitted__方法检查estimator是否已经拟合
        return estimator.__sklearn_is_fitted__()

    # 获取所有以下划线结尾且不以双下划线开头的属性
    fitted_attrs = [
        v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
    ]
    # 返回是否找到任何符合条件的属性
    return len(fitted_attrs) > 0


def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to
    specify if the estimator is fitted or not. See
    :ref:`sphx_glr_auto_examples_developing_estimators_sklearn_is_fitted.py`
    for an example on how to use the API.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance
    """
    # 通过调用_is_fitted函数来检查estimator是否已经拟合
    if not _is_fitted(estimator, attributes, all_or_any=all):
        # 如果未拟合，生成适当的错误消息
        if msg is None:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
        # 替换错误消息中的%(name)s为estimator的名称
        raise TypeError(msg % {"name": type(estimator).__name__})
    """
    检查估计器是否已经拟合，如果没有拟合则引发异常。

    Parameters
    ----------
    estimator : object
        要检查的估计器对象。
    attributes : str or iterable of str
        需要检查的属性或属性列表。
    all_or_any : {'all', 'any'}, default='all'
        指定所有属性是否都存在 ('all') 或者任何一个属性是否存在 ('any')。

    Raises
    ------
    TypeError
        如果估计器是一个类而不是实例。
    NotFittedError
        如果未找到属性。

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.utils.validation import check_is_fitted
    >>> from sklearn.exceptions import NotFittedError
    >>> lr = LogisticRegression()
    >>> try:
    ...     check_is_fitted(lr)
    ... except NotFittedError as exc:
    ...     print(f"Model is not fitted yet.")
    Model is not fitted yet.
    >>> lr.fit([[1, 2], [1, 3]], [1, 0])
    LogisticRegression()
    >>> check_is_fitted(lr)
    """
    检查是否是一个类而不是实例
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    如果消息是空的
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )
    如果没有安装属性
    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    如果没有安装 (estimator, attributes, all_or_any):
    estimator, raise
def check_non_negative(X, whom):
    """
    Check if there is any negative value in an array.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Input data.

    whom : str
        Who passed X to this function.
    """
    # 获取 X 的命名空间和类型
    xp, _ = get_namespace(X)
    
    # 针对稀疏矩阵，避免在其上使用 X.min()，因为它会对索引进行排序
    if sp.issparse(X):
        # 对于 lil 或 dok 格式的稀疏矩阵，转换为 csr 格式
        if X.format in ["lil", "dok"]:
            X = X.tocsr()
        
        # 如果稀疏矩阵没有数据，则将最小值设为 0
        if X.data.size == 0:
            X_min = 0
        else:
            X_min = X.data.min()
    else:
        # 对于普通数组，获取最小值
        X_min = xp.min(X)

    # 如果最小值小于 0，则抛出异常
    if X_min < 0:
        raise ValueError("Negative values in data passed to %s" % whom)


def check_scalar(
    x,
    name,
    target_type,
    *,
    min_val=None,
    max_val=None,
    include_boundaries="both",
):
    """Validate scalar parameters type and value.

    Parameters
    ----------
    x : object
        The scalar parameter to validate.

    name : str
        The name of the parameter to be printed in error messages.

    target_type : type or tuple
        Acceptable data types for the parameter.

    min_val : float or int, default=None
        The minimum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have a lower bound.

    max_val : float or int, default=None
        The maximum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have an upper bound.

    include_boundaries : {"left", "right", "both", "neither"}, default="both"
        Whether the interval defined by `min_val` and `max_val` should include
        the boundaries. Possible choices are:

        - `"left"`: only `min_val` is included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val )`.
        - `"right"`: only `max_val` is included in the valid interval.
          It is equivalent to the interval `( min_val, max_val ]`.
        - `"both"`: `min_val` and `max_val` are included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val ]`.
        - `"neither"`: neither `min_val` nor `max_val` are included in the
          valid interval. It is equivalent to the interval `( min_val, max_val )`.

    Returns
    -------
    x : numbers.Number
        The validated number.

    Raises
    ------
    TypeError
        If the parameter's type does not match the desired type.

    ValueError
        If the parameter's value violates the given bounds.
        If `min_val`, `max_val` and `include_boundaries` are inconsistent.

    Examples
    --------
    >>> from sklearn.utils.validation import check_scalar
    >>> check_scalar(10, "x", int, min_val=1, max_val=20)
    10
    """
    def type_name(t):
        """Convert type into human readable string."""
        # 获取类型对象的模块和限定名称
        module = t.__module__
        qualname = t.__qualname__
        # 如果类型对象属于内置类型，直接返回限定名称
        if module == "builtins":
            return qualname
        # 对于特定的数值类型，返回相应的字符串表示
        elif t == numbers.Real:
            return "float"
        elif t == numbers.Integral:
            return "int"
        # 否则，返回完整的模块限定名称
        return f"{module}.{qualname}"

    # 检查 x 是否不是目标类型 target_type 的实例
    if not isinstance(x, target_type):
        # 如果 target_type 是一个元组，生成包含多种类型的字符串表示
        if isinstance(target_type, tuple):
            types_str = ", ".join(type_name(t) for t in target_type)
            target_type_str = f"{{{types_str}}}"
        else:
            # 否则，直接生成目标类型的字符串表示
            target_type_str = type_name(target_type)

        # 抛出类型错误，指明 x 的类型不符合预期的 target_type
        raise TypeError(
            f"{name} must be an instance of {target_type_str}, not"
            f" {type(x).__qualname__}."
        )

    # 预期的 include_boundaries 的有效取值
    expected_include_boundaries = ("left", "right", "both", "neither")
    # 如果 include_boundaries 不在预期的取值范围内，抛出数值错误
    if include_boundaries not in expected_include_boundaries:
        raise ValueError(
            f"Unknown value for `include_boundaries`: {repr(include_boundaries)}. "
            f"Possible values are: {expected_include_boundaries}."
        )

    # 如果 max_val 未指定且 include_boundaries 为 'right'，抛出数值错误
    if max_val is None and include_boundaries == "right":
        raise ValueError(
            "`include_boundaries`='right' without specifying explicitly `max_val` "
            "is inconsistent."
        )

    # 如果 min_val 未指定且 include_boundaries 为 'left'，抛出数值错误
    if min_val is None and include_boundaries == "left":
        raise ValueError(
            "`include_boundaries`='left' without specifying explicitly `min_val` "
            "is inconsistent."
        )

    # 根据 include_boundaries 确定比较运算符
    comparison_operator = (
        operator.lt if include_boundaries in ("left", "both") else operator.le
    )
    # 如果 min_val 已指定且 x 小于（或小于等于）min_val，抛出数值错误
    if min_val is not None and comparison_operator(x, min_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'>=' if include_boundaries in ('left', 'both') else '>'} {min_val}."
        )

    # 根据 include_boundaries 确定比较运算符
    comparison_operator = (
        operator.gt if include_boundaries in ("right", "both") else operator.ge
    )
    # 如果 max_val 已指定且 x 大于（或大于等于）max_val，抛出数值错误
    if max_val is not None and comparison_operator(x, max_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'<=' if include_boundaries in ('right', 'both') else '<'} {max_val}."
        )

    # 如果所有验证通过，返回 x
    return x
# 定义函数 _check_psd_eigenvalues，用于检查正半定（PSD）矩阵的特征值
def _check_psd_eigenvalues(lambdas, enable_warnings=False):
    """Check the eigenvalues of a positive semidefinite (PSD) matrix.

    Checks the provided array of PSD matrix eigenvalues for numerical or
    conditioning issues and returns a fixed validated version. This method
    should typically be used if the PSD matrix is user-provided (e.g. a
    Gram matrix) or computed using a user-provided dissimilarity metric
    (e.g. kernel function), or if the decomposition process uses approximation
    methods (randomized SVD, etc.).

    It checks for three things:

    - that there are no significant imaginary parts in eigenvalues (more than
      1e-5 times the maximum real part). If this check fails, it raises a
      ``ValueError``. Otherwise all non-significant imaginary parts that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    - that eigenvalues are not all negative. If this check fails, it raises a
      ``ValueError``

    - that there are no significant negative eigenvalues with absolute value
      more than 1e-10 (1e-6) and more than 1e-5 (5e-3) times the largest
      positive eigenvalue in double (simple) precision. If this check fails,
      it raises a ``ValueError``. Otherwise all negative eigenvalues that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    Finally, all the positive eigenvalues that are too small (with a value
    smaller than the maximum eigenvalue multiplied by 1e-12 (2e-7)) are set to
    zero. This operation is traced with a ``PositiveSpectrumWarning`` when
    ``enable_warnings=True``.

    Parameters
    ----------
    lambdas : array-like of shape (n_eigenvalues,)
        Array of eigenvalues to check / fix.

    enable_warnings : bool, default=False
        When this is set to ``True``, a ``PositiveSpectrumWarning`` will be
        raised when there are imaginary parts, negative eigenvalues, or
        extremely small non-zero eigenvalues. Otherwise no warning will be
        raised. In both cases, imaginary parts, negative eigenvalues, and
        extremely small non-zero eigenvalues will be set to zero.

    Returns
    -------
    lambdas_fixed : ndarray of shape (n_eigenvalues,)
        A fixed validated copy of the array of eigenvalues.

    Examples
    --------
    >>> from sklearn.utils.validation import _check_psd_eigenvalues
    >>> _check_psd_eigenvalues([1, 2])      # nominal case
    array([1, 2])
    >>> _check_psd_eigenvalues([5, 5j])     # significant imag part
    Traceback (most recent call last):
        ...
    ValueError: There are significant imaginary parts in eigenvalues (1
        of the maximum real part). Either the matrix is not PSD, or there was
        an issue while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, 5e-5j])  # insignificant imag part
    array([5., 0.])
    """
    # 检查是否启用警告，用于记录数值或条件问题
    if enable_warnings:
        # 计算最大实部的1e-5倍作为阈值，检查特征值中的显著虚部
        imaginary_threshold = 1e-5 * np.max(np.real(lambdas))
        # 检查特征值中是否存在显著虚部，若存在则抛出 ValueError
        if np.any(np.abs(np.imag(lambdas)) > imaginary_threshold):
            raise ValueError(f"There are significant imaginary parts in eigenvalues "
                             f"({np.sum(np.abs(np.imag(lambdas)) > imaginary_threshold)}"
                             f" of the maximum real part). Either the matrix is not PSD, "
                             f"or there was an issue while computing the eigendecomposition "
                             f"of the matrix.")
    
    # 将所有非显著虚部设为零
    lambdas = np.real(lambdas)
    
    # 检查特征值是否全为负数，若是则抛出 ValueError
    if np.all(lambdas < 0):
        raise ValueError("All eigenvalues are negative.")
    
    # 计算最大正特征值和设定阈值，检查负特征值的显著性
    pos_max = np.max(lambdas[lambdas > 0])
    neg_threshold = 1e-10 * pos_max if np.finfo(lambdas.dtype).precision == 53 else 1e-6 * pos_max
    
    # 检查特征值中是否存在显著负数，若存在则抛出 ValueError
    if np.any(lambdas < -neg_threshold):
        raise ValueError(f"There are significant negative eigenvalues with absolute value "
                         f"more than {neg_threshold}.")
    
    # 将所有非显著负数设为零
    lambdas[lambdas < 0] = 0.0
    
    # 计算极小正特征值的阈值，将其设为零
    tiny_threshold = 1e-12 * pos_max if np.finfo(lambdas.dtype).precision == 53 else 2e-7 * pos_max
    lambdas[lambdas < tiny_threshold] = 0.0
    
    # 返回修正后的特征值数组
    return lambdas
    # 转换为 NumPy 数组，确保处理的是特征值列表
    lambdas = np.array(lambdas)
    # 检查特征值是否是双精度浮点数
    is_double_precision = lambdas.dtype == np.float64
    
    # 注意：可用的最小值为
    # - 单精度：np.finfo('float32').eps = 1.2e-07
    # - 双精度：np.finfo('float64').eps = 2.2e-16
    
    # 不同阈值用于验证
    # 可能根据精度级别进行值的更改。
    significant_imag_ratio = 1e-5
    significant_neg_ratio = 1e-5 if is_double_precision else 5e-3
    significant_neg_value = 1e-10 if is_double_precision else 1e-6
    small_pos_ratio = 1e-12 if is_double_precision else 2e-7
    
    # 检查是否存在显著的虚部特征值
    if not np.isreal(lambdas).all():
        # 计算最大的虚部和实部绝对值
        max_imag_abs = np.abs(np.imag(lambdas)).max()
        max_real_abs = np.abs(np.real(lambdas)).max()
        if max_imag_abs > significant_imag_ratio * max_real_abs:
            # 如果存在显著的虚部，则引发 ValueError
            raise ValueError(
                "There are significant imaginary parts in eigenvalues (%g "
                "of the maximum real part). Either the matrix is not PSD, or "
                "there was an issue while computing the eigendecomposition "
                "of the matrix." % (max_imag_abs / max_real_abs)
            )
    
            # 如果启用警告，警告有虚部被移除
            if enable_warnings:
                warnings.warn(
                    "There are imaginary parts in eigenvalues (%g "
                    "of the maximum real part). Either the matrix is not"
                    " PSD, or there was an issue while computing the "
                    "eigendecomposition of the matrix. Only the real "
                    "parts will be kept." % (max_imag_abs / max_real_abs),
                    PositiveSpectrumWarning,
                )
    
    # 移除所有的虚部（即使为零）
    lambdas = np.real(lambdas)
    
    # 检查是否存在显著的负特征值
    max_eig = lambdas.max()
    if max_eig < 0:
        # 如果存在负特征值，引发 ValueError
        raise ValueError(
            "All eigenvalues are negative (maximum is %g). "
            "Either the matrix is not PSD, or there was an "
            "issue while computing the eigendecomposition of "
            "the matrix." % max_eig
        )
    else:
        # 找到最小特征值
        min_eig = lambdas.min()
        # 如果最小特征值小于一定的负数比例乘以最大特征值，并且小于一个显著的负值
        if (
            min_eig < -significant_neg_ratio * max_eig
            and min_eig < -significant_neg_value
        ):
            # 抛出值错误，说明存在显著的负特征值，可能是矩阵不是半正定的，或者在计算矩阵的特征分解时出现问题
            raise ValueError(
                "There are significant negative eigenvalues (%g"
                " of the maximum positive). Either the matrix is "
                "not PSD, or there was an issue while computing "
                "the eigendecomposition of the matrix." % (-min_eig / max_eig)
            )
        elif min_eig < 0:
            # 移除所有负值，并发出警告
            if enable_warnings:
                warnings.warn(
                    "There are negative eigenvalues (%g of the "
                    "maximum positive). Either the matrix is not "
                    "PSD, or there was an issue while computing the"
                    " eigendecomposition of the matrix. Negative "
                    "eigenvalues will be replaced with 0." % (-min_eig / max_eig),
                    PositiveSpectrumWarning,
                )
            # 将所有负特征值替换为0
            lambdas[lambdas < 0] = 0

    # 检查条件数（小的正非零值）
    too_small_lambdas = (0 < lambdas) & (lambdas < small_pos_ratio * max_eig)
    # 如果存在太小的特征值
    if too_small_lambdas.any():
        # 发出警告，说明半正定矩阵的谱很差，最大特征值超过最小特征值的多倍，小的特征值将被替换为0
        if enable_warnings:
            warnings.warn(
                "Badly conditioned PSD matrix spectrum: the largest "
                "eigenvalue is more than %g times the smallest. "
                "Small eigenvalues will be replaced with 0."
                "" % (1 / small_pos_ratio),
                PositiveSpectrumWarning,
            )
        # 将小的特征值替换为0
        lambdas[too_small_lambdas] = 0

    # 返回处理后的特征值数组
    return lambdas
# 校验样本权重的有效性及一致性
def _check_sample_weight(
    sample_weight, X, dtype=None, copy=False, only_non_negative=False
):
    """Validate sample weights.

    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)

    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
        Input sample weights.

    X : {ndarray, list, sparse matrix}
        Input data.

    only_non_negative : bool, default=False,
        Whether or not the weights are expected to be non-negative.

        .. versionadded:: 1.0

    dtype : dtype, default=None
        dtype of the validated `sample_weight`.
        If None, and the input `sample_weight` is an array, the dtype of the
        input is preserved; otherwise an array with the default numpy dtype
        is be allocated.  If `dtype` is not one of `float32`, `float64`,
        `None`, the output will be of dtype `float64`.

    copy : bool, default=False
        If True, a copy of sample_weight will be created.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,)
        Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    # 获取样本数量
    n_samples = _num_samples(X)

    # 如果指定了 dtype 但不是 float32 或 float64，则使用 float64
    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    # 如果 sample_weight 为 None，则创建一个全为 1 的数组作为样本权重
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=dtype)
    # 如果 sample_weight 是数值类型，则创建一个指定数值的数组作为样本权重
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
    else:
        # 如果未指定 dtype，则默认为 float64 或 float32
        if dtype is None:
            dtype = [np.float64, np.float32]
        # 校验并转换 sample_weight 为 numpy 数组
        sample_weight = check_array(
            sample_weight,
            accept_sparse=False,
            ensure_2d=False,
            dtype=dtype,
            order="C",
            copy=copy,
            input_name="sample_weight",
        )
        # 确保 sample_weight 是 1 维数组或标量
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # 确保 sample_weight 的形状与样本数相匹配
        if sample_weight.shape != (n_samples,):
            raise ValueError(
                "sample_weight.shape == {}, expected {}!".format(
                    sample_weight.shape, (n_samples,)
                )
            )

    # 如果要求只有非负权重，则检查 sample_weight 是否非负
    if only_non_negative:
        check_non_negative(sample_weight, "`sample_weight`")

    # 返回经验证后的样本权重数组
    return sample_weight


def _allclose_dense_sparse(x, y, rtol=1e-7, atol=1e-9):
    """Check allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : {array-like, sparse matrix}
        First array to compare.

    y : {array-like, sparse matrix}
        Second array to compare.

    rtol : float, default=1e-7
        Relative tolerance; see numpy.allclose.
    atol : float, default=1e-9
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.
    """
    # 如果 x 和 y 都是稀疏矩阵
    if sp.issparse(x) and sp.issparse(y):
        # 将稀疏矩阵 x 和 y 转换为 CSR 格式
        x = x.tocsr()
        y = y.tocsr()
        # 确保 CSR 格式的稀疏矩阵没有重复条目
        x.sum_duplicates()
        y.sum_duplicates()
        # 比较稀疏矩阵的 indices 和 indptr 数组是否相等，以及数据元素是否在指定误差范围内相等
        return (
            np.array_equal(x.indices, y.indices)
            and np.array_equal(x.indptr, y.indptr)
            and np.allclose(x.data, y.data, rtol=rtol, atol=atol)
        )
    # 如果 x 和 y 都不是稀疏矩阵
    elif not sp.issparse(x) and not sp.issparse(y):
        # 直接比较两个数组或矩阵是否在指定误差范围内相等
        return np.allclose(x, y, rtol=rtol, atol=atol)
    # 如果一个是稀疏矩阵，一个是数组，抛出异常
    raise ValueError(
        "Can only compare two sparse matrices, not a sparse matrix and an array"
    )
# 检查 `response_method` 方法是否在估计器中可用，并返回该方法
# 这个函数从版本 1.3 开始被添加

def _check_response_method(estimator, response_method):
    """Check if `response_method` is available in estimator and return it.

    .. versionadded:: 1.3

    Parameters
    ----------
    estimator : estimator instance
        Classifier or regressor to check.

    response_method : {"predict_proba", "predict_log_proba", "decision_function",
            "predict"} or list of such str
        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`predict_log_proba`,
        :term:`decision_function` or :term:`predict`). Possible choices are:
        - if `str`, it corresponds to the name to the method to return;
        - if a list of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.

    Returns
    -------
    prediction_method : callable
        Prediction method of estimator.

    Raises
    ------
    AttributeError
        If `response_method` is not available in `estimator`.
    """
    # 如果 `response_method` 是字符串，则将其放入列表中
    if isinstance(response_method, str):
        list_methods = [response_method]
    else:
        list_methods = response_method

    # 获取估计器中可用的方法，按列表中的顺序查找第一个实现的方法
    prediction_method = [getattr(estimator, method, None) for method in list_methods]
    # 使用 `reduce` 函数，返回第一个非空的方法
    prediction_method = reduce(lambda x, y: x or y, prediction_method)
    # 如果没有找到有效的方法，抛出 AttributeError 异常
    if prediction_method is None:
        raise AttributeError(
            f"{estimator.__class__.__name__} has none of the following attributes: "
            f"{', '.join(list_methods)}."
        )

    return prediction_method


def _check_method_params(X, params, indices=None):
    """Check and validate the parameters passed to a specific
    method like `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    params : dict
        Dictionary containing the parameters passed to the method.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    Returns
    -------
    method_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """
    # 导入 `_safe_indexing` 函数，确保参数支持索引操作
    from . import _safe_indexing

    # 初始化一个空的字典，用于存储验证后的方法参数
    method_params_validated = {}
    # 遍历参数字典中的键和值
    for param_key, param_value in params.items():
        # 检查参数值是否不是类数组并且不是稀疏矩阵，或者样本数量与输入数据 X 的样本数量不相等
        if (
            not _is_arraylike(param_value)
            and not sp.issparse(param_value)
            or _num_samples(param_value) != _num_samples(X)
        ):
            # 对于不可索引的情况，暂时直接传递参数值（为了向后兼容）。
            # 参考：https://github.com/scikit-learn/scikit-learn/issues/15805
            method_params_validated[param_key] = param_value
        else:
            # 对于支持索引的其他方法参数
            # 将参数值转换为支持索引操作的形式（例如用于交叉验证）
            method_params_validated[param_key] = _make_indexable(param_value)
            # 安全索引操作，确保在给定索引下安全地访问数据
            method_params_validated[param_key] = _safe_indexing(
                method_params_validated[param_key], indices
            )

    # 返回经验证的方法参数字典
    return method_params_validated
def _get_feature_names(X):
    """Get feature names from X.

    Support for other array containers should place its implementation here.

    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        Array container to extract feature names.

        - pandas dataframe : The columns will be considered to be feature
          names. If the dataframe contains non-string feature names, `None` is
          returned.
        - All other array containers will return `None`.

    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    feature_names = None  # 初始化特征名称为None

    # 从不同的数组容器中提取特征名称
    if _is_pandas_df(X):
        # 确保即使是过旧版本的 pandas，我们也能够检查列名，避免引入额外的复制
        # TODO: 一旦 pandas 的最低支持版本有了工作正常的 __dataframe__.column_names() 实现，
        #       可以移除这个 pandas 特定的分支。
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        # 获取数据帧协议并将列名转换为 ndarray
        df_protocol = X.__dataframe__()
        feature_names = np.asarray(list(df_protocol.column_names()), dtype=object)

    if feature_names is None or len(feature_names) == 0:
        return

    # 获取特征名称的类型集合并排序
    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))

    # 混合字符串和非字符串类型的情况不受支持
    # 如果输入的特征名类型超过一个，并且包含字符串类型，抛出类型错误异常
    if len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names are only supported if all input features have string names, "
            f"but your input has {types} as feature name / column name types. "
            "If you want feature names to be stored and validated, you must convert "
            "them all to strings, by using X.columns = X.columns.astype(str) for "
            "example. Otherwise you can remove feature / column names from your input "
            "data, or convert them all to a non-string data type."
        )

    # 如果输入的特征名类型只有一个，并且为字符串类型，则返回特征名
    if len(types) == 1 and types[0] == "str":
        return feature_names
# 检查 `input_features` 并在需要时生成名称。

# 常用于 :term:`get_feature_names_out`。

# Parameters 参数
# ----------
# input_features : array-like of str or None, default=None
#     输入特征。

#     - 如果 `input_features` 是 `None`，那么将使用 `feature_names_in_` 作为输入特征名。如果 `feature_names_in_` 未定义，
#       则生成以下输入特征名：`["x0", "x1", ..., "x(n_features_in_ - 1)"]`。
#     - 如果 `input_features` 是 array-like，则必须与 `feature_names_in_` 匹配（如果 `feature_names_in_` 已定义）。

# generate_names : bool, default=True
#     当 `input_features` 为 `None` 且 `estimator.feature_names_in_` 未定义时，是否生成名称。对于验证 `input_features` 但不在
#     :term:`get_feature_names_out` 中需要它们的转换器（如 `PCA`），这很有用。

# Returns 返回
# -------
# feature_names_in : ndarray of str or `None`
#     输入特征名。
def _check_feature_names_in(estimator, input_features=None, *, generate_names=True):
    # 获取 estimator 对象的 `feature_names_in_` 属性，如果不存在则为 `None`
    feature_names_in_ = getattr(estimator, "feature_names_in_", None)
    # 获取 estimator 对象的 `n_features_in_` 属性，如果不存在则为 `None`
    n_features_in_ = getattr(estimator, "n_features_in_", None)

    # 如果 `input_features` 不为 `None`
    if input_features is not None:
        # 将 `input_features` 转换为 ndarray 类型，dtype 设置为 object
        input_features = np.asarray(input_features, dtype=object)
        # 如果 `feature_names_in_` 不为 `None` 并且 `feature_names_in_` 与 `input_features` 不相等
        if feature_names_in_ is not None and not np.array_equal(
            feature_names_in_, input_features
        ):
            raise ValueError("input_features is not equal to feature_names_in_")

        # 如果 `n_features_in_` 不为 `None` 并且 `input_features` 的长度不等于 `n_features_in_`
        if n_features_in_ is not None and len(input_features) != n_features_in_:
            raise ValueError(
                "input_features should have length equal to number of "
                f"features ({n_features_in_}), got {len(input_features)}"
            )
        # 返回 `input_features`
        return input_features

    # 如果 `feature_names_in_` 不为 `None`，返回 `feature_names_in_`
    if feature_names_in_ is not None:
        return feature_names_in_

    # 如果不生成名称，返回 `None`
    if not generate_names:
        return

    # 如果无法生成特征名称（`n_features_in_` 未定义），抛出 ValueError 异常
    if n_features_in_ is None:
        raise ValueError("Unable to generate feature names without n_features_in_")

    # 生成特征名称，如果 `n_features_in_` 已定义
    return np.asarray([f"x{i}" for i in range(n_features_in_)], dtype=object)


# 生成 `estimator` 的输出特征名，使用 `estimator` 名称作为前缀。
# `input_features` 的名称进行验证但不使用。此函数对于基于 `n_features_out` 生成自己名称的估计器（如 `PCA`）非常有用。
#
# Parameters 参数
# ----------
# estimator : estimator instance
#     生成输出特征名的估计器实例。
#
# n_feature_out : int
#     输出特征名的数量。
#
# input_features : array-like of str or None, default=None
#     仅用于与 `estimator.feature_names_in_` 验证特征名。
#
# Returns 返回
# -------
    # 输入参数 feature_names_in 是一个 ndarray 数组，包含字符串或为 None
    feature_names_in : ndarray of str or `None`
        Feature names in.
    """
    # 调用 _check_feature_names_in 函数，验证输入的特征名
    _check_feature_names_in(estimator, input_features, generate_names=False)
    # 获取 estimator 对象的类名并转换为小写字符串
    estimator_name = estimator.__class__.__name__.lower()
    # 创建一个包含 n_features_out 个元素的 ndarray，每个元素格式为 "{estimator_name}{i}"
    # 其中 i 从 0 到 n_features_out-1
    return np.asarray(
        [f"{estimator_name}{i}" for i in range(n_features_out)], dtype=object
    )
def _check_monotonic_cst(estimator, monotonic_cst=None):
    """Check the monotonic constraints and return the corresponding array.

    This helper function should be used in the `fit` method of an estimator
    that supports monotonic constraints and called after the estimator has
    introspected input data to set the `n_features_in_` and optionally the
    `feature_names_in_` attributes.

    .. versionadded:: 1.2

    Parameters
    ----------
    estimator : estimator instance
        模型估计器实例，用于拟合数据。

    monotonic_cst : array-like of int, dict of str or None, default=None
        Monotonic constraints for the features.
        特征的单调性约束。

        - If array-like, then it should contain only -1, 0 or 1. Each value
          will be checked to be in [-1, 0, 1]. If a value is -1, then the
          corresponding feature is required to be monotonically decreasing.
          如果是数组，则只能包含 -1、0 或 1。每个值将被检查是否在 [-1, 0, 1] 范围内。如果是 -1，则对应的特征必须是单调递减的。
        
        - If dict, then it the keys should be the feature names occurring in
          `estimator.feature_names_in_` and the values should be -1, 0 or 1.
          如果是字典，则键应该是出现在 `estimator.feature_names_in_` 中的特征名，值应该是 -1、0 或 1。
        
        - If None, then an array of 0s will be allocated.
          如果为 None，则会分配一个全为 0 的数组。

    Returns
    -------
    monotonic_cst : ndarray of int
        Monotonic constraints for each feature.
        每个特征的单调性约束数组。
    """
    original_monotonic_cst = monotonic_cst
    
    # 如果 monotonic_cst 是 None 或者是字典类型
    if monotonic_cst is None or isinstance(monotonic_cst, dict):
        # 初始化一个形状为 estimator.n_features_in_，填充值为 0，数据类型为 np.int8 的数组
        monotonic_cst = np.full(
            shape=estimator.n_features_in_,
            fill_value=0,
            dtype=np.int8,
        )
        
        # 如果 original_monotonic_cst 是字典类型
        if isinstance(original_monotonic_cst, dict):
            # 如果 estimator 没有 feature_names_in_ 属性，则抛出 ValueError
            if not hasattr(estimator, "feature_names_in_"):
                raise ValueError(
                    f"{estimator.__class__.__name__} was not fitted on data "
                    "with feature names. Pass monotonic_cst as an integer "
                    "array instead."
                )
            
            # 计算不在 estimator.feature_names_in_ 中的意外特征名
            unexpected_feature_names = list(
                set(original_monotonic_cst) - set(estimator.feature_names_in_)
            )
            unexpected_feature_names.sort()  # 确定性错误消息
            n_unexpeced = len(unexpected_feature_names)
            
            # 如果存在意外特征名
            if unexpected_feature_names:
                if len(unexpected_feature_names) > 5:
                    unexpected_feature_names = unexpected_feature_names[:5]
                    unexpected_feature_names.append("...")
                raise ValueError(
                    f"monotonic_cst contains {n_unexpeced} unexpected feature "
                    f"names: {unexpected_feature_names}."
                )
            
            # 遍历 estimator.feature_names_in_，更新 monotonic_cst 数组
            for feature_idx, feature_name in enumerate(estimator.feature_names_in_):
                if feature_name in original_monotonic_cst:
                    cst = original_monotonic_cst[feature_name]
                    if cst not in [-1, 0, 1]:
                        raise ValueError(
                            f"monotonic_cst['{feature_name}'] must be either "
                            f"-1, 0 or 1. Got {cst!r}."
                        )
                    monotonic_cst[feature_idx] = cst
    else:
        # 找出不符合预期的单调约束值，即不在 [-1, 0, 1] 范围内的值
        unexpected_cst = np.setdiff1d(monotonic_cst, [-1, 0, 1])
        # 如果存在不符合预期的值，则抛出 ValueError 异常
        if unexpected_cst.shape[0]:
            raise ValueError(
                "monotonic_cst must be an array-like of -1, 0 or 1. Observed "
                f"values: {unexpected_cst.tolist()}."
            )

        # 将 monotonic_cst 转换为 int8 类型的 NumPy 数组
        monotonic_cst = np.asarray(monotonic_cst, dtype=np.int8)
        # 检查转换后的 monotonic_cst 的长度是否与输入数据 X 的特征数相符合
        if monotonic_cst.shape[0] != estimator.n_features_in_:
            raise ValueError(
                f"monotonic_cst has shape {monotonic_cst.shape} but the input data "
                f"X has {estimator.n_features_in_} features."
            )
    # 返回处理后的 monotonic_cst 数组
    return monotonic_cst
# 检查正类标签的一致性，确定是否需要指定 `pos_label` 参数
def _check_pos_label_consistency(pos_label, y_true):
    """Check if `pos_label` need to be specified or not.

    In binary classification, we fix `pos_label=1` if the labels are in the set
    {-1, 1} or {0, 1}. Otherwise, we raise an error asking to specify the
    `pos_label` parameters.

    Parameters
    ----------
    pos_label : int, float, bool, str or None
        The positive label.
    y_true : ndarray of shape (n_samples,)
        The target vector.

    Returns
    -------
    pos_label : int, float, bool or str
        If `pos_label` can be inferred, it will be returned.

    Raises
    ------
    ValueError
        In the case that `y_true` does not have label in {-1, 1} or {0, 1},
        it will raise a `ValueError`.
    """
    # 如果未指定 pos_label，则进行二元分类的检查
    if pos_label is None:
        # 计算目标向量 y_true 的唯一值
        classes = np.unique(y_true)
        # 如果 classes 的数据类型种类为 'O', 'U', 'S'，或者它们不满足以下条件之一：
        # [0, 1], [-1, 1], [0], [-1], [1] 中的任何一个，就会触发 ValueError 异常
        if classes.dtype.kind in "OUS" or not (
            np.array_equal(classes, [0, 1])
            or np.array_equal(classes, [-1, 1])
            or np.array_equal(classes, [0])
            or np.array_equal(classes, [-1])
            or np.array_equal(classes, [1])
        ):
            # 构建 classes_repr，其中包含 classes 的每个元素的表示形式
            classes_repr = ", ".join([repr(c) for c in classes.tolist()])
            # 抛出异常，说明 y_true 的取值范围不符合预期，要求显式传入 pos_label
            raise ValueError(
                f"y_true takes value in {{{classes_repr}}} and pos_label is not "
                "specified: either make y_true take value in {0, 1} or "
                "{-1, 1} or pass pos_label explicitly."
            )
        # 设置 pos_label 为默认值 1
        pos_label = 1

    return pos_label


def _to_object_array(sequence):
    """Convert sequence to a 1-D NumPy array of object dtype.

    numpy.array constructor has a similar use but it's output
    is ambiguous. It can be 1-D NumPy array of object dtype if
    the input is a ragged array, but if the input is a list of
    equal length arrays, then the output is a 2D numpy.array.
    _to_object_array solves this ambiguity by guarantying that
    the output is a 1-D NumPy array of objects for any input.

    Parameters
    ----------
    sequence : array-like of shape (n_elements,)
        The sequence to be converted.

    Returns
    -------
    out : ndarray of shape (n_elements,), dtype=object
        The converted sequence into a 1-D NumPy array of object dtype.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.validation import _to_object_array
    >>> _to_object_array([np.array([0]), np.array([1])])
    array([array([0]), array([1])], dtype=object)
    >>> _to_object_array([np.array([0]), np.array([1, 2])])
    array([array([0]), array([1, 2])], dtype=object)
    >>> _to_object_array([np.array([0]), np.array([1, 2])])
    array([array([0]), array([1, 2])], dtype=object)
    """
    # 创建一个空的 NumPy 数组，长度与给定的 sequence 相同，数据类型为 object
    out = np.empty(len(sequence), dtype=object)
    # 将 sequence 中的所有元素复制到新创建的 NumPy 数组 out 中
    out[:] = sequence
    # 返回填充了 sequence 元素的 NumPy 数组 out
    return out
```