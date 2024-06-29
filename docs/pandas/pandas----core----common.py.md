# `D:\src\scipysrc\pandas\pandas\core\common.py`

```
"""
Misc tools for implementing data structures

Note: pandas.core.common is *not* part of the public API.
"""

# 导入未来版本支持的注解功能
from __future__ import annotations

# 导入内建模块
import builtins
# 导入 collections 模块中的 abc 和 defaultdict 类
from collections import (
    abc,
    defaultdict,
)
# 从 collections.abc 中导入 Callable、Collection、Generator、Hashable、Iterable、Sequence 类
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Sequence,
)
# 导入 contextlib 模块
import contextlib
# 导入 functools 模块中的 partial 函数
from functools import partial
# 导入 inspect 模块
import inspect
# 导入 typing 模块中的各种类型和装饰器
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)
# 导入 warnings 模块
import warnings

# 导入 numpy 库，并简写为 np
import numpy as np

# 从 pandas._libs 中导入 lib 模块
from pandas._libs import lib
# 从 pandas.compat.numpy 中导入 np_version_gte1p24 函数
from pandas.compat.numpy import np_version_gte1p24

# 从 pandas.core.dtypes.cast 中导入 construct_1d_object_array_from_listlike 函数
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
# 从 pandas.core.dtypes.common 中导入 is_bool_dtype 和 is_integer 函数
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
)
# 从 pandas.core.dtypes.generic 中导入 ABCExtensionArray、ABCIndex、ABCMultiIndex、ABCSeries 类
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
# 从 pandas.core.dtypes.inference 中导入 iterable_not_string 函数
from pandas.core.dtypes.inference import iterable_not_string

# 如果是类型检查模式，从 pandas._typing 中导入特定类型
if TYPE_CHECKING:
    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        Concatenate,
        NpDtype,
        P,
        RandomState,
        T,
    )

    from pandas import Index


def flatten(line):
    """
    Flatten an arbitrarily nested sequence.

    Parameters
    ----------
    line : sequence
        The non string sequence to flatten

    Notes
    -----
    This doesn't consider strings sequences.

    Returns
    -------
    flattened : generator
    """
    for element in line:
        if iterable_not_string(element):
            yield from flatten(element)
        else:
            yield element


def consensus_name_attr(objs):
    name = objs[0].name
    for obj in objs[1:]:
        try:
            if obj.name != name:
                name = None
        except ValueError:
            name = None
    return name


def is_bool_indexer(key: Any) -> bool:
    """
    Check whether `key` is a valid boolean indexer.

    Parameters
    ----------
    key : Any
        Only list-likes may be considered boolean indexers.
        All other types are not considered a boolean indexer.
        For array-like input, boolean ndarrays or ExtensionArrays
        with ``_is_boolean`` set are considered boolean indexers.

    Returns
    -------
    bool
        Whether `key` is a valid boolean indexer.

    Raises
    ------
    ValueError
        When the array is an object-dtype ndarray or ExtensionArray
        and contains missing values.

    See Also
    --------
    check_array_indexer : Check that `key` is a valid array to index,
        and convert to an ndarray.
    """
    # 检查 key 是否是有效的布尔索引器
    if isinstance(
        key, (ABCSeries, np.ndarray, ABCIndex, ABCExtensionArray)
        # 如果 key 是 ABCSeries、np.ndarray、ABCIndex 或 ABCExtensionArray 类型中的一种
    ) and not isinstance(key, ABCMultiIndex):
        # 检查 key 不是 slice 对象，并且不是 pandas 的 MultiIndex 类型的实例
        if key.dtype == np.object_:
            # 如果 key 的数据类型是 np.object_
            key_array = np.asarray(key)

            if not lib.is_bool_array(key_array):
                # 如果 key_array 不是布尔类型的数组
                na_msg = "Cannot mask with non-boolean array containing NA / NaN values"
                if lib.is_bool_array(key_array, skipna=True):
                    # 如果 key_array 是布尔类型的数组，但是含有 NA 或 NaN 值，则抛出 ValueError
                    # 不会在例如 ["A", "B", np.nan] 这种情况下抛出异常，详见 test_loc_getitem_list_of_labels_categoricalindex_with_na
                    raise ValueError(na_msg)
                return False
            return True
        elif is_bool_dtype(key.dtype):
            # 如果 key 的数据类型是布尔类型
            return True
    elif isinstance(key, list):
        # 如果 key 是列表类型
        # 检查 np.array(key).dtype 是否为布尔类型
        if len(key) > 0:
            if type(key) is not list:  # noqa: E721
                # GH#42461 Cython 会在传递子类时引发 TypeError
                key = list(key)
            return lib.is_bool_list(key)

    # 默认情况下返回 False
    return False
# 防止使用浮点数作为索引键，即使该浮点数是一个整数。
def cast_scalar_indexer(val):
    # 假设 val 是标量
    if lib.is_float(val) and val.is_integer():
        raise IndexError(
            # 提示用户不再支持使用浮点数作为索引。建议手动转换为整数键。
            "Indexing with a float is no longer supported. Manually convert "
            "to an integer key instead."
        )
    return val


# 返回一个生成器，其中包含非 None 的参数。
def not_none(*args):
    return (arg for arg in args if arg is not None)


# 返回一个布尔值，指示是否有任何一个参数为 None。
def any_none(*args) -> bool:
    return any(arg is None for arg in args)


# 返回一个布尔值，指示是否所有参数都为 None。
def all_none(*args) -> bool:
    return all(arg is None for arg in args)


# 返回一个布尔值，指示是否有任何一个参数不为 None。
def any_not_none(*args) -> bool:
    return any(arg is not None for arg in args)


# 返回一个布尔值，指示是否所有参数都不为 None。
def all_not_none(*args) -> bool:
    return all(arg is not None for arg in args)


# 返回不为 None 的参数的计数。
def count_not_none(*args) -> int:
    return sum(x is not None for x in args)


# 根据值的类型安全地创建一个数组。处理不同类型的输入，返回对应的数组。
@overload
def asarray_tuplesafe(
    values: ArrayLike | list | tuple | zip, dtype: NpDtype | None = ...
) -> np.ndarray:
    # 当 values 是 Index 时，可以返回 ExtensionArray；其他可迭代对象则返回 np.ndarray。
    ...


@overload
def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = ...) -> ArrayLike: ...


# 实现函数的主体部分，根据输入值的类型安全地创建数组。
def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = None) -> ArrayLike:
    # 如果 values 不是列表、元组或具有 "__array__" 属性的对象，则转换为列表。
    if not (isinstance(values, (list, tuple)) or hasattr(values, "__array__")):
        values = list(values)
    # 如果 values 是 Index 类型，则直接返回其 _values 属性。
    elif isinstance(values, ABCIndex):
        return values._values
    # 如果 values 是 Series 类型，则直接返回其 _values 属性。
    elif isinstance(values, ABCSeries):
        return values._values

    # 如果 values 是列表且 dtype 是 np.object_ 或 object 类型，则调用特定函数处理。
    if isinstance(values, list) and dtype in [np.object_, object]:
        return construct_1d_object_array_from_listlike(values)

    # 尝试使用 NumPy 的 asarray 函数创建数组，支持指定 dtype。
    try:
        with warnings.catch_warnings():
            # 可以移除警告过滤器，一旦 NumPy 版本 >= 1.24
            if not np_version_gte1p24:
                warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
            result = np.asarray(values, dtype=dtype)
    # 处理 ValueError 异常，这里使用 try/except 更有效率，避免每个元素都检查 is_list_like
    # error: Argument 1 to "construct_1d_object_array_from_listlike"
    # has incompatible type "Iterable[Any]"; expected "Sized"
    # 调用 construct_1d_object_array_from_listlike 函数，忽略类型检查
    return construct_1d_object_array_from_listlike(values)  # type: ignore[arg-type]

    # 如果结果的数据类型是字符串的子类
    if issubclass(result.dtype.type, str):
        # 将 values 转换为对象类型的 NumPy 数组
        result = np.asarray(values, dtype=object)

    # 如果结果的维度是 2
    if result.ndim == 2:
        # 避免构建数组的数组：
        # 将 values 中的每个元素转换为元组，并重新调用 construct_1d_object_array_from_listlike 函数
        values = [tuple(x) for x in values]
        result = construct_1d_object_array_from_listlike(values)

    # 返回处理后的 result 结果
    return result
def index_labels_to_array(
    labels: np.ndarray | Iterable, dtype: NpDtype | None = None
) -> np.ndarray:
    """
    Transform label or iterable of labels to array, for use in Index.

    Parameters
    ----------
    labels : np.ndarray or Iterable
        Input labels to be transformed into an array.
    dtype : dtype or None, optional
        If specified, the dtype of the resulting array, otherwise inferred.

    Returns
    -------
    np.ndarray
        Array representation of the input labels.
    """
    if isinstance(labels, (str, tuple)):
        # Convert single string or tuple to list
        labels = [labels]

    if not isinstance(labels, (list, np.ndarray)):
        try:
            # Try converting labels to list (handles non-iterable case)
            labels = list(labels)
        except TypeError:  # non-iterable
            # Fallback to list with a single element
            labels = [labels]

    # Ensure labels are converted to a numpy array
    labels = asarray_tuplesafe(labels, dtype=dtype)

    return labels


def maybe_make_list(obj):
    """
    Convert obj to a list if it is not already a tuple or list.

    Parameters
    ----------
    obj : any
        Object to potentially convert to a list.

    Returns
    -------
    list or obj
        If obj is not None and not already a tuple or list, return [obj], otherwise return obj.
    """
    if obj is not None and not isinstance(obj, (tuple, list)):
        return [obj]
    return obj


def maybe_iterable_to_list(obj: Iterable[T] | T) -> Collection[T] | T:
    """
    Convert obj to a list if it is iterable but not already list-like.

    Parameters
    ----------
    obj : Iterable[T] or T
        Object to potentially convert to a list.

    Returns
    -------
    Collection[T] or T
        If obj is iterable but not list-like, return as list, otherwise return obj.
    """
    if isinstance(obj, abc.Iterable) and not isinstance(obj, abc.Sized):
        return list(obj)
    obj = cast(Collection, obj)
    return obj


def is_null_slice(obj) -> bool:
    """
    Check if obj is a null slice (slice with start, stop, and step all None).

    Parameters
    ----------
    obj : any
        Object to check for being a null slice.

    Returns
    -------
    bool
        True if obj is a null slice, False otherwise.
    """
    return (
        isinstance(obj, slice)
        and obj.start is None
        and obj.stop is None
        and obj.step is None
    )


def is_empty_slice(obj) -> bool:
    """
    Check if obj is an empty slice (slice with start and stop defined and equal).

    Parameters
    ----------
    obj : any
        Object to check for being an empty slice.

    Returns
    -------
    bool
        True if obj is an empty slice, False otherwise.
    """
    return (
        isinstance(obj, slice)
        and obj.start is not None
        and obj.stop is not None
        and obj.start == obj.stop
    )


def is_true_slices(line: abc.Iterable) -> abc.Generator[bool, None, None]:
    """
    Iterate over line to find non-null non-empty slices.

    Parameters
    ----------
    line : abc.Iterable
        Iterable to iterate over and check for non-null non-empty slices.

    Yields
    ------
    bool
        True for each non-null non-empty slice found in line.
    """
    for k in line:
        yield isinstance(k, slice) and not is_null_slice(k)


# TODO: used only once in indexing; belongs elsewhere?
def is_full_slice(obj, line: int) -> bool:
    """
    Check if obj is a full-length slice (slice from 0 to line with step=None).

    Parameters
    ----------
    obj : any
        Object to check for being a full-length slice.
    line : int
        Length of the line to compare with.

    Returns
    -------
    bool
        True if obj is a full-length slice, False otherwise.
    """
    return (
        isinstance(obj, slice)
        and obj.start == 0
        and obj.stop == line
        and obj.step is None
    )


def get_callable_name(obj):
    """
    Get the name of a callable object.

    Parameters
    ----------
    obj : callable
        Callable object to retrieve the name from.

    Returns
    -------
    str or None
        Name of the callable object or None if no name is found.
    """
    # typical case has name
    if hasattr(obj, "__name__"):
        return getattr(obj, "__name__")
    # some objects don't; could recurse
    if isinstance(obj, partial):
        return get_callable_name(obj.func)
    # fall back to class name
    if callable(obj):
        return type(obj).__name__
    # everything failed (probably because the argument
    # wasn't actually callable); we return None
    # instead of the empty string in this case to allow
    # distinguishing between no name and a name of ''
    return None


def apply_if_callable(maybe_callable, obj, **kwargs):
    """
    Apply obj and kwargs to maybe_callable if it is callable, otherwise return as it is.

    Parameters
    ----------
    maybe_callable : callable
        Possibly callable object to apply obj and kwargs to.
    obj : NDFrame
        Object to apply to maybe_callable if it is callable.
    **kwargs
        Additional keyword arguments to pass to maybe_callable if it is callable.

    Returns
    -------
    any
        Result of maybe_callable(obj, **kwargs) if maybe_callable is callable, otherwise maybe_callable.
    """
    if callable(maybe_callable):
        return maybe_callable(obj, **kwargs)
    # 返回变量 maybe_callable 的值作为函数的返回结果
    return maybe_callable
# 辅助函数，用于标准化提供的映射对象。

from typing import TypeVar, Callable, Any, Type, Union, overload
from collections import defaultdict, abc
import inspect
from functools import partial
import numpy as np

_T = TypeVar("_T")  # Secondary TypeVar for use in pipe's type hints


@overload
def pipe(
    obj: _T,
    func: Callable[Concatenate[_T, P], T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T: ...


@overload
def pipe(
    obj: Any,
    func: tuple[Callable[..., T], str],
    *args: Any,
    **kwargs: Any,
) -> T: ...


def pipe(
    obj: _T,
    func: Callable[Concatenate[_T, P], T] | tuple[Callable[..., T], str],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Apply a function ``func`` to object ``obj`` either by passing obj as the
    first argument to func or by calling obj's method named by func[1].

    Parameters
    ----------
    obj : _T
        The object to which func will be applied.
    func : Callable[Concatenate[_T, P], T] | tuple[Callable[..., T], str]
        A function or a tuple where the first element is a function and the
        second element is a string representing the method name to call on obj.
    args : Any
        Additional positional arguments to pass to func.
    kwargs : Any
        Additional keyword arguments to pass to func.

    Returns
    -------
    T
        The result of applying func to obj.

    Raises
    ------
    TypeError
        If func is not callable or if obj does not have a method named func[1].
    """
    pass
    if isinstance(func, tuple):
        # 如果 func 是一个元组，则解包元组，func_ 是元组的第一个元素（可调用对象），target 是元组的第二个元素（关键字字符串）
        func_, target = func
        # 检查是否在 kwargs 中已经存在了与 target 同名的关键字参数，如果存在则抛出 ValueError 异常
        if target in kwargs:
            msg = f"{target} is both the pipe target and a keyword argument"
            raise ValueError(msg)
        # 将 obj 赋值给 kwargs 中以 target 为键的位置，以便传递给 func_
        kwargs[target] = obj
        # 调用 func_ 函数，并传入 *args 和 **kwargs，返回结果
        return func_(*args, **kwargs)
    else:
        # 如果 func 不是元组，则直接调用 func，并传入 obj、*args 和 **kwargs，返回结果
        return func(obj, *args, **kwargs)
def get_rename_function(mapper):
    """
    Returns a function that will map names/labels, dependent if mapper
    is a dict, Series or just a function.
    """

    def f(x):
        # 如果 x 在 mapper 中，则返回 mapper[x]，否则返回 x 自身
        if x in mapper:
            return mapper[x]
        else:
            return x

    # 如果 mapper 是 dict 或 Series，则返回函数 f；否则直接返回 mapper 本身
    return f if isinstance(mapper, (abc.Mapping, ABCSeries)) else mapper


def convert_to_list_like(
    values: Hashable | Iterable | AnyArrayLike,
) -> list | AnyArrayLike:
    """
    Convert list-like or scalar input to list-like. List, numpy and pandas array-like
    inputs are returned unmodified whereas others are converted to list.
    """
    # 如果 values 是 list, np.ndarray, pandas 中的数组类型（Index 或 Series），则直接返回 values
    if isinstance(values, (list, np.ndarray, ABCIndex, ABCSeries, ABCExtensionArray)):
        return values
    # 如果 values 是可迭代对象但不是字符串，则转换为列表并返回
    elif isinstance(values, abc.Iterable) and not isinstance(values, str):
        return list(values)

    # 对于其他情况，将 values 包装成列表并返回
    return [values]


@contextlib.contextmanager
def temp_setattr(
    obj, attr: str, value, condition: bool = True
) -> Generator[None, None, None]:
    """
    Temporarily set attribute on an object.

    Parameters
    ----------
    obj : object
        Object whose attribute will be modified.
    attr : str
        Attribute to modify.
    value : Any
        Value to temporarily set attribute to.
    condition : bool, default True
        Whether to set the attribute. Provided in order to not have to
        conditionally use this context manager.

    Yields
    ------
    object : obj with modified attribute.
    """
    # 如果 condition 为 True，则获取原始属性值并设置新值
    if condition:
        old_value = getattr(obj, attr)
        setattr(obj, attr, value)
    try:
        # 使用 yield 将修改后的对象暴露给调用者
        yield obj
    finally:
        # 最终将属性恢复为原始值（如果 condition 为 True）
        if condition:
            setattr(obj, attr, old_value)


def require_length_match(data, index: Index) -> None:
    """
    Check the length of data matches the length of the index.
    """
    # 检查 data 和 index 的长度是否相等，如果不相等则抛出 ValueError 异常
    if len(data) != len(index):
        raise ValueError(
            "Length of values "
            f"({len(data)}) "
            "does not match length of index "
            f"({len(index)})"
        )


_cython_table = {
    builtins.sum: "sum",
    builtins.max: "max",
    builtins.min: "min",
    np.all: "all",
    np.any: "any",
    np.sum: "sum",
    np.nansum: "sum",
    np.mean: "mean",
    np.nanmean: "mean",
    np.prod: "prod",
    np.nanprod: "prod",
    np.std: "std",
    np.nanstd: "std",
    np.var: "var",
    np.nanvar: "var",
    np.median: "median",
    np.nanmedian: "median",
    np.max: "max",
    np.nanmax: "max",
    np.min: "min",
    np.nanmin: "min",
    np.cumprod: "cumprod",
    np.nancumprod: "cumprod",
    np.cumsum: "cumsum",
    np.nancumsum: "cumsum",
}


def get_cython_func(arg: Callable) -> str | None:
    """
    if we define an internal function for this argument, return it
    """
    # 根据传入的函数参数 arg，在 _cython_table 中查找并返回对应的字符串函数名
    return _cython_table.get(arg)


def fill_missing_names(names: Sequence[Hashable | None]) -> list[Hashable]:
    """
    If a name is missing then replace it by level_n, where n is the count

    .. versionadded:: 1.4.0

    Parameters
    ----------
    names : Sequence[Hashable | None]
        List of names which may contain None values.

    Returns
    -------
    list[Hashable]
        List of names with None values replaced by level_n.

    """
    # 接受一个类似列表的参数 names，表示列名列表或者包含 None 值的列表
    def replace_none_names(names):
        # 使用列表推导式遍历 names 列表，对每个元素进行处理
        # 如果元素为 None，则用形如 "level_i" 的字符串替代，其中 i 是元素的索引值
        # 如果元素不为 None，则保留原来的值
        return [f"level_{i}" if name is None else name for i, name in enumerate(names)]
```