# `D:\src\scipysrc\scikit-learn\sklearn\utils\_indexing.py`

```
# 导入必要的模块和函数
import numbers  # 导入 numbers 模块，用于数字相关的类型判断
import sys  # 导入 sys 模块，提供与 Python 解释器相关的功能
import warnings  # 导入 warnings 模块，用于警告控制
from collections import UserList  # 导入 UserList 类，用于自定义列表类
from itertools import compress, islice  # 导入 compress 和 islice 函数，用于迭代和过滤迭代器

import numpy as np  # 导入 NumPy 库，并用 np 作为别名
from scipy.sparse import issparse  # 从 SciPy 的 sparse 模块中导入 issparse 函数

from ._array_api import _is_numpy_namespace, get_namespace  # 从私有模块中导入部分函数和变量
from ._param_validation import Interval, validate_params  # 导入参数验证相关函数
from .extmath import _approximate_mode  # 从 extmath 模块导入 _approximate_mode 函数
from .validation import (  # 从 validation 模块导入以下函数
    _is_arraylike_not_scalar,
    _is_pandas_df,
    _is_polars_df_or_series,
    _use_interchange_protocol,
    check_array,
    check_consistent_length,
    check_random_state,
)


def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    xp, is_array_api = get_namespace(array)
    # 如果是使用数组 API，使用 xp.take() 方法进行索引
    if is_array_api:
        return xp.take(array, key, axis=axis)
    # 如果是稀疏矩阵并且索引类型为布尔型，将 key 转换为 NumPy 数组
    if issparse(array) and key_dtype == "bool":
        key = np.asarray(key)
    # 如果 key 是元组类型，则转换为列表
    if isinstance(key, tuple):
        key = list(key)
    # 根据轴向进行索引操作，返回对应切片或索引的数组部分
    return array[key, ...] if axis == 0 else array[:, key]


def _pandas_indexing(X, key, key_dtype, axis):
    """Index a pandas dataframe or a series."""
    # 如果 key 是类似数组的非标量对象，转换为 NumPy 数组
    if _is_arraylike_not_scalar(key):
        key = np.asarray(key)

    if key_dtype == "int" and not (isinstance(key, slice) or np.isscalar(key)):
        # 使用 take() 替代 iloc[] 确保返回的是一个适当的副本，避免触发 SettingWithCopyWarning
        return X.take(key, axis=axis)
    else:
        # 检查是否应该使用 loc 或 iloc 进行索引
        indexer = X.iloc if key_dtype == "int" else X.loc
        return indexer[:, key] if axis else indexer[key]


def _list_indexing(X, key, key_dtype):
    """Index a Python list."""
    # 如果 key 是标量或者切片，直接进行索引操作
    if np.isscalar(key) or isinstance(key, slice):
        return X[key]
    # 如果 key 是布尔类型的数组，使用 compress 函数进行筛选
    if key_dtype == "bool":
        return list(compress(X, key))
    # 如果 key 是整数类型的数组，按照索引逐个取出列表中的元素
    return [X[idx] for idx in key]


def _polars_indexing(X, key, key_dtype, axis):
    """Indexing X with polars interchange protocol."""
    # Polars 的行为更接近于列表
    if isinstance(key, np.ndarray):
        # 将数组的每个元素转换为 Python 标量
        key = key.tolist()
    elif not (np.isscalar(key) or isinstance(key, slice)):
        # 如果 key 不是标量或者切片，则转换为列表
        key = list(key)

    if axis == 1:
        # 在这里我们可以确定 X 是 polars DataFrame；可以使用整数和字符串标量，以及整数、字符串和布尔型列表进行索引
        return X[:, key]

    if key_dtype == "bool":
        # 布尔掩码可以在 Series 和 DataFrame（axis=0）中以相同的方式进行索引
        return X.filter(key)

    # 整数标量和整数列表在 Series 和 DataFrame（axis=0）中可以以相同的方式进行索引
    X_indexed = X[key]
    # 检查 `key` 是否是标量（scalar），并且 `X` 的形状是否是二维
    if np.isscalar(key) and len(X.shape) == 2:
        # 如果条件满足，说明 `X_indexed` 是一个 DataFrame，且只有一行；
        # 我们返回一个 Series，以保持与 pandas 的一致性
        pl = sys.modules["polars"]
        # 使用 Polars 模块的 Series 函数将 `X_indexed` 的第一行转换为 Series 并返回
        return pl.Series(X_indexed.row(0))
    # 如果不满足上述条件，则直接返回 `X_indexed`
    return X_indexed
def _determine_key_type(key, accept_slice=True):
    """Determine the data type of key.

    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.

    accept_slice : bool, default=True
        Whether or not to raise an error if the key is a slice.

    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    # 错误信息，用于在发生错误时提供详细的说明
    err_msg = (
        "No valid specification of the columns. Only a scalar, list or "
        "slice of all integers or all strings, or boolean mask is "
        "allowed"
    )

    # 映射标量类型到对应的字符串表示
    dtype_to_str = {int: "int", str: "str", bool: "bool", np.bool_: "bool"}
    # 映射数组类型的数据类型代码到对应的字符串表示
    array_dtype_to_str = {
        "i": "int",
        "u": "int",
        "b": "bool",
        "O": "str",
        "U": "str",
        "S": "str",
    }

    # 如果 key 是 None，返回 None
    if key is None:
        return None
    # 如果 key 是 tuple 中指定的数据类型之一，返回对应的字符串表示
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    # 如果 key 是 slice 类型
    if isinstance(key, slice):
        # 如果不接受 slice 类型，并且 key 是 slice，抛出 TypeError
        if not accept_slice:
            raise TypeError(
                "Only array-like or scalar are supported. A Python slice was given."
            )
        # 如果 start 和 stop 都是 None，返回 None
        if key.start is None and key.stop is None:
            return None
        # 递归调用 _determine_key_type 确定 start 和 stop 的数据类型
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        # 如果 start 和 stop 都不是 None，且它们的数据类型不同，抛出 ValueError
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        # 如果 start 的数据类型不是 None，返回 start 的数据类型
        if key_start_type is not None:
            return key_start_type
        # 否则返回 stop 的数据类型
        return key_stop_type
    # 如果 key 是 list、tuple 或 UserList 中的一种
    if isinstance(key, (list, tuple, UserList)):
        # 将 key 转换成集合去除重复元素
        unique_key = set(key)
        # 递归调用 _determine_key_type 确定每个元素的数据类型
        key_type = {_determine_key_type(elt) for elt in unique_key}
        # 如果 key_type 为空，返回 None
        if not key_type:
            return None
        # 如果 key_type 中不止一种数据类型，抛出 ValueError
        if len(key_type) != 1:
            raise ValueError(err_msg)
        # 返回集合中唯一的数据类型
        return key_type.pop()
    # 如果 key 有 dtype 属性，即 key 是数组类型
    if hasattr(key, "dtype"):
        # 获取 key 的命名空间和数组 API 的状态
        xp, is_array_api = get_namespace(key)
        # 如果是数组 API，并且不是 NumPy 命名空间，根据 dtype 判断数据类型
        if is_array_api and not _is_numpy_namespace(xp):
            if xp.isdtype(key.dtype, "bool"):
                return "bool"
            elif xp.isdtype(key.dtype, "integral"):
                return "int"
            else:
                raise ValueError(err_msg)
        else:
            # 根据 key 的 dtype.kind 使用 array_dtype_to_str 映射返回数据类型字符串表示
            try:
                return array_dtype_to_str[key.dtype.kind]
            except KeyError:
                raise ValueError(err_msg)
    # 如果 key 类型不在上述情况中，则抛出 ValueError
    raise ValueError(err_msg)


def _safe_indexing(X, indices, *, axis=0):
    """Return rows, items or columns of X using indices.
    .. warning::

        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset
        Subset of X on axis 0 or 1.

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils import _safe_indexing
    >>> data = np.array([[1, 2], [3, 4], [5, 6]])
    >>> _safe_indexing(data, 0, axis=0)  # select the first row
    array([1, 2])
    >>> _safe_indexing(data, 0, axis=1)  # select the first column
    array([1, 3, 5])
    """

    # 如果 indices 为 None，则直接返回 X，无需进行索引操作
    if indices is None:
        return X

    # 如果 axis 不是 0 或 1，则抛出 ValueError 异常
    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    # 确定 indices 的类型
    indices_dtype = _determine_key_type(indices)

    # 如果 axis=0 且 indices 类型为字符串，则抛出 ValueError 异常
    if axis == 0 and indices_dtype == "str":
        raise ValueError("String indexing is not supported with 'axis=0'")

    # 如果 axis=1 且 X 是列表，则抛出 ValueError 异常
    if axis == 1 and isinstance(X, list):
        raise ValueError("axis=1 is not supported for lists")

    # 如果 axis=1 且 X 具有 shape 属性，但不是 2D 结构，则抛出 ValueError 异常
    if axis == 1 and hasattr(X, "shape") and len(X.shape) != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), len(X.shape))
        )

    # 如果 axis=1、indices 类型为字符串，并且 X 不是 pandas DataFrame 或不支持交换协议，则抛出 ValueError 异常
    if (
        axis == 1
        and indices_dtype == "str"
        and not (_is_pandas_df(X) or _use_interchange_protocol(X))
    ):
        raise ValueError(
            "Specifying the columns using strings is only supported for dataframes."
        )
    # 检查对象 X 是否具有属性 "iloc"，即判断是否为 Pandas 的 DataFrame 或 Series
    if hasattr(X, "iloc"):
        # TODO: 可能应该使用 _is_pandas_df_or_series(X) 函数来替代，但这会涉及更新一些测试如 test_train_test_split_mock_pandas。
        # 如果是 Pandas DataFrame 或 Series，则调用 _pandas_indexing 函数进行索引操作
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    # 检查对象 X 是否为 Polars 的 DataFrame 或 Series
    elif _is_polars_df_or_series(X):
        # 如果是 Polars DataFrame 或 Series，则调用 _polars_indexing 函数进行索引操作
        return _polars_indexing(X, indices, indices_dtype, axis=axis)
    # 如果对象 X 具有 "shape" 属性，则假定为普通数组或类数组对象
    elif hasattr(X, "shape"):
        # 调用 _array_indexing 函数进行索引操作
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    # 如果以上条件均不满足，则假定 X 是一个普通列表或类列表对象
    else:
        # 调用 _list_indexing 函数进行索引操作
        return _list_indexing(X, indices, indices_dtype)
# 对 numpy 数组、稀疏矩阵或 pandas dataframe 进行安全赋值的函数

def _safe_assign(X, values, *, row_indexer=None, column_indexer=None):
    """Safe assignment to a numpy array, sparse matrix, or pandas dataframe.

    Parameters
    ----------
    X : {ndarray, sparse-matrix, dataframe}
        Array to be modified. It is expected to be 2-dimensional.

    values : ndarray
        The values to be assigned to `X`.

    row_indexer : array-like, dtype={int, bool}, default=None
        A 1-dimensional array to select the rows of interest. If `None`, all
        rows are selected.

    column_indexer : array-like, dtype={int, bool}, default=None
        A 1-dimensional array to select the columns of interest. If `None`, all
        columns are selected.
    """
    # 如果 `row_indexer` 为 `None`，选择所有行
    row_indexer = slice(None, None, None) if row_indexer is None else row_indexer
    # 如果 `column_indexer` 为 `None`，选择所有列
    column_indexer = (
        slice(None, None, None) if column_indexer is None else column_indexer
    )

    # 如果 X 有 iloc 属性，即为 pandas dataframe
    if hasattr(X, "iloc"):
        with warnings.catch_warnings():
            # pandas >= 1.5 在使用 iloc 设置列时，如果列的类型与被设置的类型不同会产生警告
            # 例如，将字符串赋值给分类列
            # 未来该行为不会改变，警告应该消失
            warnings.simplefilter("ignore", FutureWarning)
            # 使用 iloc 方法安全地将 values 赋值给指定的行和列
            X.iloc[row_indexer, column_indexer] = values
    else:
        # 对于 numpy 数组或稀疏矩阵，直接使用索引赋值
        X[row_indexer, column_indexer] = values


def _get_column_indices_for_bool_or_int(key, n_columns):
    # 将 key 转换为正整数索引列表
    try:
        # 使用 _safe_indexing 函数获取索引
        idx = _safe_indexing(np.arange(n_columns), key)
    except IndexError as e:
        # 如果索引超出范围，抛出 ValueError 异常
        raise ValueError(
            f"all features must be in [0, {n_columns - 1}] or [-{n_columns}, 0]"
        ) from e
    # 将索引转换为至少为 1 维的列表
    return np.atleast_1d(idx).tolist()


def _get_column_indices(X, key):
    """Get feature column indices for input data X and key.

    For accepted values of `key`, see the docstring of
    :func:`_safe_indexing`.
    """
    # 确定 key 的类型
    key_dtype = _determine_key_type(key)
    # 如果支持数据交换协议，使用交换协议获取列索引
    if _use_interchange_protocol(X):
        return _get_column_indices_interchange(X.__dataframe__(), key, key_dtype)

    # 获取 X 的列数
    n_columns = X.shape[1]
    # 如果 key 是空列表，则返回空列表
    if isinstance(key, (list, tuple)) and not key:
        return []
    # 如果 key 的类型为布尔型或整型，则调用 _get_column_indices_for_bool_or_int 函数获取索引
    elif key_dtype in ("bool", "int"):
        return _get_column_indices_for_bool_or_int(key, n_columns)
    else:
        # 如果不是字符串或者切片，抛出错误，只支持字符串方式指定列名
        try:
            # 获取数据集的所有列名
            all_columns = X.columns
        except AttributeError:
            # 如果不是数据框，抛出错误
            raise ValueError(
                "Specifying the columns using strings is only supported for dataframes."
            )
        # 如果 key 是字符串，转换为单元素列表
        if isinstance(key, str):
            columns = [key]
        # 如果 key 是切片对象
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            # 获取起始位置的索引，如果有的话
            if start is not None:
                start = all_columns.get_loc(start)
            # 获取结束位置的索引，如果有的话
            if stop is not None:
                # pandas 使用字符串索引是包含终点的
                stop = all_columns.get_loc(stop) + 1
            else:
                # 默认取到最后一列的下一个位置
                stop = n_columns + 1
            # 返回索引范围对应的列名列表
            return list(islice(range(n_columns), start, stop))
        else:
            # key 是列表，直接使用
            columns = list(key)

        try:
            # 存储列索引的列表
            column_indices = []
            # 遍历每个列名
            for col in columns:
                # 获取列名对应的索引
                col_idx = all_columns.get_loc(col)
                # 如果索引不是整数，抛出错误，说明列名不唯一
                if not isinstance(col_idx, numbers.Integral):
                    raise ValueError(
                        f"Selected columns, {columns}, are not unique in dataframe"
                    )
                # 将索引加入列表
                column_indices.append(col_idx)

        except KeyError as e:
            # 如果出现键错误，说明指定的列名不在数据框中，抛出错误
            raise ValueError("A given column is not a column of the dataframe") from e

        # 返回列索引列表
        return column_indices
# 从 X_interchange 中获取列索引，用于 __dataframe__ 协议的 X
def _get_column_indices_interchange(X_interchange, key, key_dtype):
    """Same as _get_column_indices but for X with __dataframe__ protocol."""

    # 获取 X_interchange 的列数
    n_columns = X_interchange.num_columns()

    # 如果 key 是列表或元组且为空，则返回空列表
    if isinstance(key, (list, tuple)) and not key:
        return []
    # 如果 key_dtype 是 "bool" 或 "int"，则调用 _get_column_indices_for_bool_or_int 函数获取列索引
    elif key_dtype in ("bool", "int"):
        return _get_column_indices_for_bool_or_int(key, n_columns)
    else:
        # 获取 X_interchange 的列名列表
        column_names = list(X_interchange.column_names())

        # 如果 key 是切片对象
        if isinstance(key, slice):
            # 如果 key.step 不是 1 或 None，则抛出 NotImplementedError
            if key.step not in [1, None]:
                raise NotImplementedError("key.step must be 1 or None")
            start, stop = key.start, key.stop
            if start is not None:
                start = column_names.index(start)

            if stop is not None:
                stop = column_names.index(stop) + 1
            else:
                stop = n_columns + 1
            return list(islice(range(n_columns), start, stop))

        # 如果 key 是标量或数组
        selected_columns = [key] if np.isscalar(key) else key

        try:
            # 返回选定列的索引列表
            return [column_names.index(col) for col in selected_columns]
        except ValueError as e:
            raise ValueError("A given column is not a column of the dataframe") from e


# 对数组或稀疏矩阵进行重采样
@validate_params(
    {
        "replace": ["boolean"],
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "stratify": ["array-like", "sparse matrix", None],
    },
    prefer_skip_nested_validation=True,
)
def resample(*arrays, replace=True, n_samples=None, random_state=None, stratify=None):
    """Resample arrays or sparse matrices in a consistent way.

    The default strategy implements one step of the bootstrapping
    procedure.

    Parameters
    ----------
    *arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Indexable data-structures can be arrays, lists, dataframes or scipy
        sparse matrices with consistent first dimension.

    replace : bool, default=True
        Implements resampling with replacement. If False, this will implement
        (sliced) random permutations.

    n_samples : int, default=None
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.
        If replace is False it should not be larger than the length of
        arrays.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling
        the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    stratify : {array-like, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_outputs), default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    # 将 n_samples 赋值给 max_n_samples，表示要采样的最大样本数
    max_n_samples = n_samples
    # 检查并设置随机数生成器，确保 random_state 是一个有效的随机状态
    random_state = check_random_state(random_state)

    # 如果 arrays 列表为空，则返回 None
    if len(arrays) == 0:
        return None

    # 获取 arrays 列表中的第一个元素
    first = arrays[0]
    # 计算第一个元素的样本数量，如果具有 shape 属性，则取 shape[0]，否则取 len(first)
    n_samples = first.shape[0] if hasattr(first, "shape") else len(first)

    # 如果 max_n_samples 为 None，则设为 n_samples；如果 max_n_samples 大于 n_samples 且不允许替换，则报错
    if max_n_samples is None:
        max_n_samples = n_samples
    elif (max_n_samples > n_samples) and (not replace):
        raise ValueError(
            "Cannot sample %d out of arrays with dim %d when replace is False"
            % (max_n_samples, n_samples)
        )

    # 检查 arrays 中所有数组的长度是否一致，否则会引发异常
    check_consistent_length(*arrays)

    # 如果未指定分层抽样 (stratify=None)
    if stratify is None:
        # 如果允许替换 (replace=True)，则生成 max_n_samples 个随机索引
        if replace:
            indices = random_state.randint(0, n_samples, size=(max_n_samples,))
        # 如果不允许替换 (replace=False)，则生成 n_samples 个连续索引并进行洗牌，再取前 max_n_samples 个
        else:
            indices = np.arange(n_samples)
            random_state.shuffle(indices)
            indices = indices[:max_n_samples]
    else:
        # 从 StratifiedShuffleSplit() 函数改编而来的代码

        # 检查并转换成数组格式，确保是二维数组，数据类型不指定
        y = check_array(stratify, ensure_2d=False, dtype=None)

        # 如果 y 是二维数组，则将每行转换成字符串表示，以空格分隔各元素
        if y.ndim == 2:
            y = np.array([" ".join(row.astype("str")) for row in y])

        # 获取 y 中的唯一类别及其对应的索引
        classes, y_indices = np.unique(y, return_inverse=True)

        # 类别的数量
        n_classes = classes.shape[0]

        # 统计每个类别出现的次数
        class_counts = np.bincount(y_indices)

        # 找到每个类别对应的实例的排序列表：
        # （np.unique 已经执行了排序，所以代码已经是 O(n logn) 的时间复杂度）
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        # 根据 _approximate_mode 函数估算每个类别需要采样的实例数
        n_i = _approximate_mode(class_counts, max_n_samples, random_state)

        # 初始化索引列表
        indices = []

        # 对每个类别进行采样
        for i in range(n_classes):
            # 从每个类别的索引中随机选择 n_i[i] 个样本，可以选择是否替换
            indices_i = random_state.choice(class_indices[i], n_i[i], replace=replace)
            indices.extend(indices_i)

        # 将索引列表随机打乱
        indices = random_state.permutation(indices)

    # 将稀疏矩阵转换为 CSR 格式，以便行索引
    arrays = [a.tocsr() if issparse(a) else a for a in arrays]

    # 对采样后的数组进行索引操作
    resampled_arrays = [_safe_indexing(a, indices) for a in arrays]

    # 如果 resampled_arrays 只有一个元素，则返回该元素
    if len(resampled_arrays) == 1:
        return resampled_arrays[0]
    else:
        return resampled_arrays
# 对传入的数组或稀疏矩阵进行一致的随机重排列，返回重排列后的副本集合。
def shuffle(*arrays, random_state=None, n_samples=None):
    """Shuffle arrays or sparse matrices in a consistent way.

    This is a convenience alias to ``resample(*arrays, replace=False)`` to do
    random permutations of the collections.

    Parameters
    ----------
    *arrays : sequence of indexable data-structures
        Indexable data-structures can be arrays, lists, dataframes or scipy
        sparse matrices with consistent first dimension.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling
        the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_samples : int, default=None
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.  It should
        not be larger than the length of arrays.

    Returns
    -------
    shuffled_arrays : sequence of indexable data-structures
        Sequence of shuffled copies of the collections. The original arrays
        are not impacted.

    See Also
    --------
    resample : Resample arrays or sparse matrices in a consistent way.

    Examples
    --------
    It is possible to mix sparse and dense arrays in the same run::

      >>> import numpy as np
      >>> X = np.array([[1., 0.], [2., 1.], [0., 0.]])
      >>> y = np.array([0, 1, 2])

      >>> from scipy.sparse import coo_matrix
      >>> X_sparse = coo_matrix(X)

      >>> from sklearn.utils import shuffle
      >>> X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)
      >>> X
      array([[0., 0.],
             [2., 1.],
             [1., 0.]])

      >>> X_sparse
      <3x2 sparse matrix of type '<... 'numpy.float64'>'
          with 3 stored elements in Compressed Sparse Row format>

      >>> X_sparse.toarray()
      array([[0., 0.],
             [2., 1.],
             [1., 0.]])

      >>> y
      array([2, 1, 0])

      >>> shuffle(y, n_samples=2, random_state=0)
      array([0, 1])
    """
    # 调用 resample 函数来进行非替换式重抽样，实现集合的随机排列
    return resample(
        *arrays, replace=False, n_samples=n_samples, random_state=random_state
    )
```