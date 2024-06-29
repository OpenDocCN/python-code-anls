# `D:\src\scipysrc\pandas\pandas\util\_validators.py`

```
# 导入必要的模块和类
from __future__ import annotations

from collections.abc import (
    Iterable,  # 导入 Iterable 类，用于检查对象是否可迭代
    Sequence,  # 导入 Sequence 类，用于检查对象是否序列化
)
from typing import (
    TypeVar,   # 导入 TypeVar，用于定义泛型变量
    overload,  # 导入 overload 装饰器，用于定义函数重载
)

import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas._libs import lib  # 导入 Pandas 内部库

from pandas.core.dtypes.common import (
    is_bool,      # 导入 Pandas 中的 is_bool 函数，用于检查是否为布尔值
    is_integer,   # 导入 Pandas 中的 is_integer 函数，用于检查是否为整数
)

BoolishT = TypeVar("BoolishT", bool, int)  # 定义一个泛型类型 BoolishT，可以是 bool 或 int
BoolishNoneT = TypeVar("BoolishNoneT", bool, int, None)  # 定义一个泛型类型 BoolishNoneT，可以是 bool、int 或 None


def _check_arg_length(fname, args, max_fname_arg_count, compat_args) -> None:
    """
    Checks whether 'args' has length of at most 'compat_args'. Raises
    a TypeError if that is not the case, similar to in Python when a
    function is called with too many arguments.
    """
    # 检查 max_fname_arg_count 是否为负数，如果是则抛出 ValueError 异常
    if max_fname_arg_count < 0:
        raise ValueError("'max_fname_arg_count' must be non-negative")

    # 检查 args 的长度是否超过 compat_args 的长度，如果是则抛出 TypeError 异常
    if len(args) > len(compat_args):
        max_arg_count = len(compat_args) + max_fname_arg_count
        actual_arg_count = len(args) + max_fname_arg_count
        argument = "argument" if max_arg_count == 1 else "arguments"

        raise TypeError(
            f"{fname}() takes at most {max_arg_count} {argument} "
            f"({actual_arg_count} given)"
        )


def _check_for_default_values(fname, arg_val_dict, compat_args) -> None:
    """
    Check that the keys in `arg_val_dict` are mapped to their
    default values as specified in `compat_args`.

    Note that this function is to be called only when it has been
    checked that arg_val_dict.keys() is a subset of compat_args
    """
    # 遍历 arg_val_dict 中的键，检查其对应的值是否与 compat_args 中指定的默认值匹配
    for key in arg_val_dict:
        try:
            v1 = arg_val_dict[key]
            v2 = compat_args[key]

            # 检查 v1 和 v2 是否为相同类型的值
            if (v1 is not None and v2 is None) or (v1 is None and v2 is not None):
                match = False
            else:
                match = v1 == v2

            # 检查 match 是否为布尔类型，如果不是则抛出 ValueError 异常
            if not is_bool(match):
                raise ValueError("'match' is not a boolean")

        # 如果直接比较抛出异常，则尝试使用 'is' 操作符进行比较
        except ValueError:
            match = arg_val_dict[key] is compat_args[key]

        # 如果 match 不为 True，则抛出 ValueError 异常，指出参数未在默认值中
        if not match:
            raise ValueError(
                f"the '{key}' parameter is not supported in "
                f"the pandas implementation of {fname}()"
            )


def validate_args(fname, args, max_fname_arg_count, compat_args) -> None:
    """
    Checks whether the length of the `*args` argument passed into a function
    has at most `len(compat_args)` arguments and whether or not all of these
    elements in `args` are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `*args` parameter
    args : tuple
        The arguments passed into the function
    max_fname_arg_count : int
        Maximum number of additional arguments allowed
    compat_args : dict
        Dictionary mapping argument names to their default values
    """
    # 检查参数长度是否符合预期
    _check_arg_length(fname, args, max_fname_arg_count, compat_args)

    # 创建一个关键字参数字典，以便在 pandas 的 'fname' 实现中提供更详细的错误信息
    kwargs = dict(zip(compat_args, args))

    # 检查关键字参数是否使用了默认值，以便在 pandas 的 'fname' 实现中进行处理
    _check_for_default_values(fname, kwargs, compat_args)
# 检查 'kwargs' 中是否包含任何不在 'compat_args' 中的键，并在有的情况下引发 TypeError 异常。
def _check_for_invalid_keys(fname, kwargs, compat_args) -> None:
    # 使用集合操作求出 'kwargs' 中存在但 'compat_args' 中不存在的键的集合
    diff = set(kwargs) - set(compat_args)

    # 如果 diff 集合不为空，则取第一个非兼容键，并通过 TypeError 抛出异常
    if diff:
        bad_arg = next(iter(diff))
        raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")


# 检查参数传递给函数 `fname` 的 `**kwargs` 是否是有效的参数，如 `*compat_args` 中指定的，以及它们是否设置为其默认值。
def validate_kwargs(fname, kwargs, compat_args) -> None:
    """
    检查传递给函数 `fname` 的 `**kwargs` 参数是否是有效的参数，
    如 `*compat_args` 中指定的，并检查它们是否设置为其默认值。

    Parameters
    ----------
    fname : str
        被传递 `**kwargs` 参数的函数的名称
    kwargs : dict
        传递给 `fname` 的 `**kwargs` 参数
    compat_args: dict
        `kwargs` 允许具有的键及其关联的默认值的字典

    Raises
    ------
    TypeError 如果 `kwargs` 包含不在 `compat_args` 中的键
    ValueError 如果 `kwargs` 包含在 `compat_args` 中，但未映射到 `compat_args` 中指定的默认值

    """
    # 复制 kwargs，以便在检查默认值时不更改原始字典
    kwds = kwargs.copy()
    # 检查 kwargs 中是否有无效的键
    _check_for_invalid_keys(fname, kwargs, compat_args)
    # 检查 kwargs 中的键是否具有默认值
    _check_for_default_values(fname, kwds, compat_args)


# 检查传递给函数 `fname` 的 `*args` 和 `**kwargs` 参数是否是有效的参数，
# 如 `*compat_args` 中指定的，并检查它们是否设置为其默认值。
def validate_args_and_kwargs(
    fname, args, kwargs, max_fname_arg_count, compat_args
) -> None:
    """
    检查传递给函数 `fname` 的 `*args` 和 `**kwargs` 参数是否是有效的参数，
    如 `*compat_args` 中指定的，并检查它们是否设置为其默认值。

    Parameters
    ----------
    fname: str
        被传递 `**kwargs` 参数的函数的名称
    args: tuple
        传递给函数的 `*args` 参数
    kwargs: dict
        传递给 `fname` 的 `**kwargs` 参数
    max_fname_arg_count: int
        函数 `fname` 需要的最小参数数量，不包括 `args`。用于显示适当的错误消息。必须为非负数。
    compat_args: dict
        `kwargs` 允许具有的键及其关联的默认值的字典。

    Raises
    ------
    TypeError 如果 `args` 包含比 `compat_args` 中更多的值
             或者 `kwargs` 包含不在 `compat_args` 中的键
    ValueError 如果 `args` 包含未设置为默认值 (`None`) 的值
             或者 `kwargs` 包含在 `compat_args` 中，但未映射到 `compat_args` 中指定的默认值

    See Also
    --------
    validate_args : 仅限 args 验证。
    validate_kwargs : 仅限 kwargs 验证。

    """
    # 检查传递的参数总数（即 args 和 kwargs）是否超过 compat_args 的长度
    _check_arg_length(
        fname, args + tuple(kwargs.values()), max_fname_arg_count, compat_args
    )
    # 将位置参数和关键字参数合并成字典，确保没有重叠
    args_dict = dict(zip(compat_args, args))
    
    # 检查合并后的位置参数是否与关键字参数中的任何键重复
    for key in args_dict:
        if key in kwargs:
            # 如果有重复的键，抛出类型错误异常
            raise TypeError(
                f"{fname}() got multiple values for keyword argument '{key}'"
            )

    # 更新关键字参数，将合并的位置参数加入其中
    kwargs.update(args_dict)
    
    # 调用函数验证关键字参数的有效性
    validate_kwargs(fname, kwargs, compat_args)
# 确保参数能够被解释为布尔值的有效性
def validate_bool_kwarg(
    value: BoolishNoneT,
    arg_name: str,
    none_allowed: bool = True,
    int_allowed: bool = False,
) -> BoolishNoneT:
    """
    Ensure that argument passed in arg_name can be interpreted as boolean.

    Parameters
    ----------
    value : bool
        Value to be validated.
    arg_name : str
        Name of the argument. To be reflected in the error message.
    none_allowed : bool, default True
        Whether to consider None to be a valid boolean.
    int_allowed : bool, default False
        Whether to consider integer value to be a valid boolean.

    Returns
    -------
    value
        The same value as input.

    Raises
    ------
    ValueError
        If the value is not a valid boolean.
    """
    good_value = is_bool(value)  # 检查是否为布尔值
    if none_allowed:
        good_value = good_value or (value is None)  # 如果允许 None，则将 None 视为有效布尔值

    if int_allowed:
        good_value = good_value or isinstance(value, int)  # 如果允许整数，则将整数视为有效布尔值

    if not good_value:
        raise ValueError(
            f'For argument "{arg_name}" expected type bool, received '
            f"type {type(value).__name__}."
        )  # 如果值不是有效的布尔值，则引发 ValueError
    return value


# 验证 'fillna' 的关键字参数的有效性
def validate_fillna_kwargs(value, method, validate_scalar_dict_value: bool = True):
    """
    Validate the keyword arguments to 'fillna'.

    This checks that exactly one of 'value' and 'method' is specified.
    If 'method' is specified, this validates that it's a valid method.

    Parameters
    ----------
    value, method : object
        The 'value' and 'method' keyword arguments for 'fillna'.
    validate_scalar_dict_value : bool, default True
        Whether to validate that 'value' is a scalar or dict. Specifically,
        validate that it is not a list or tuple.

    Returns
    -------
    value, method : object
    """
    from pandas.core.missing import clean_fill_method

    if value is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")  # 必须指定 'value' 或 'method'

    if value is None and method is not None:
        method = clean_fill_method(method)  # 清理并验证填充方法

    elif value is not None and method is None:
        if validate_scalar_dict_value and isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                f'you passed a "{type(value).__name__}"'
            )  # 如果 'value' 是列表或元组，则引发类型错误

    elif value is not None and method is not None:
        raise ValueError("Cannot specify both 'value' and 'method'.")  # 不能同时指定 'value' 和 'method'

    return value, method


# 验证百分位数（用于 describe 和 quantile）的有效性
def validate_percentile(q: float | Iterable[float]) -> np.ndarray:
    """
    Validate percentiles (used by describe and quantile).

    This function checks if the given float or iterable of floats is a valid percentile
    otherwise raises a ValueError.

    Parameters
    ----------
    q: float or iterable of floats
        A single percentile or an iterable of percentiles.

    Returns
    -------
    ndarray
        An ndarray of the percentiles if valid.

    Raises
    ------
    """
    ValueError if percentiles are not in given interval([0, 1]).
    """
    将输入的 percentiles 转换为 NumPy 数组
    q_arr = np.asarray(q)
    # 不要将此处的字符串格式化改为 f-string，因为在不需要格式化的情况下，字符串格式化开销太大。
    定义错误消息字符串，用于指示 percentiles 应该在区间 [0, 1] 内
    msg = "percentiles should all be in the interval [0, 1]"
    检查如果 percentiles 是标量（0维数组）的情况
    if q_arr.ndim == 0:
        检查单个 percentiles 是否在 [0, 1] 区间内，否则抛出 ValueError 异常
        if not 0 <= q_arr <= 1:
            raise ValueError(msg)
    如果 percentiles 是数组，则检查所有 percentiles 是否在 [0, 1] 区间内，否则抛出 ValueError 异常
    elif not all(0 <= qs <= 1 for qs in q_arr):
        raise ValueError(msg)
    返回处理后的 percentiles 数组
    return q_arr
@overload
def validate_ascending(ascending: BoolishT) -> BoolishT:
    """Function signature overload for a single BoolishT argument."""
    ...


@overload
def validate_ascending(ascending: Sequence[BoolishT]) -> list[BoolishT]:
    """Function signature overload for a sequence of BoolishT arguments."""
    ...


def validate_ascending(
    ascending: bool | int | Sequence[BoolishT],
) -> bool | int | list[BoolishT]:
    """
    Validate ``ascending`` kwargs for ``sort_index`` method.

    Parameters
    ----------
    ascending : bool | int | Sequence[BoolishT]
        The argument to validate. It can be a single bool or int, or a sequence of BoolishT.
    
    Returns
    -------
    bool | int | list[BoolishT]
        Validated result based on the type of `ascending`.

    Raises
    ------
    ValueError
        If `ascending` doesn't match expected types or values.
    """
    kwargs = {"none_allowed": False, "int_allowed": True}
    if not isinstance(ascending, Sequence):
        return validate_bool_kwarg(ascending, "ascending", **kwargs)

    return [validate_bool_kwarg(item, "ascending", **kwargs) for item in ascending]


def validate_endpoints(closed: str | None) -> tuple[bool, bool]:
    """
    Check that the `closed` argument is among [None, "left", "right"]

    Parameters
    ----------
    closed : {None, "left", "right"}
        The argument to validate for endpoint closure.

    Returns
    -------
    left_closed : bool
        True if left endpoint is closed, False otherwise.
    right_closed : bool
        True if right endpoint is closed, False otherwise.

    Raises
    ------
    ValueError
        If `closed` is not among valid values.
    """
    left_closed = False
    right_closed = False

    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == "left":
        left_closed = True
    elif closed == "right":
        right_closed = True
    else:
        raise ValueError("Closed has to be either 'left', 'right' or None")

    return left_closed, right_closed


def validate_inclusive(inclusive: str | None) -> tuple[bool, bool]:
    """
    Check that the `inclusive` argument is among {"both", "neither", "left", "right"}.

    Parameters
    ----------
    inclusive : {"both", "neither", "left", "right"}
        The argument to validate for inclusivity.

    Returns
    -------
    left_right_inclusive : tuple[bool, bool]
        Tuple indicating if left and right are inclusive (True) or exclusive (False).

    Raises
    ------
    ValueError
        If `inclusive` is not among valid values.
    """
    left_right_inclusive: tuple[bool, bool] | None = None

    if isinstance(inclusive, str):
        left_right_inclusive = {
            "both": (True, True),
            "left": (True, False),
            "right": (False, True),
            "neither": (False, False),
        }.get(inclusive)

    if left_right_inclusive is None:
        raise ValueError(
            "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
        )

    return left_right_inclusive


def validate_insert_loc(loc: int, length: int) -> int:
    """
    Check that we have an integer between -length and length, inclusive.

    Standardize negative loc to within [0, length].

    Parameters
    ----------
    loc : int
        The location to validate.
    length : int
        The maximum length for the location.

    Returns
    -------
    int
        Validated location within the specified range.

    Raises
    ------
    TypeError
        If `loc` is not an integer.
    IndexError
        If `loc` is out of the valid range [-length, length].
    """
    if not is_integer(loc):
        raise TypeError(f"loc must be an integer between -{length} and {length}")

    if loc < 0:
        loc += length
    if not 0 <= loc <= length:
        raise IndexError(f"loc must be an integer between -{length} and {length}")
    return loc  # pyright: ignore[reportReturnType]


def check_dtype_backend(dtype_backend) -> None:
    """Placeholder function with no implementation details."""
    pass
    # 检查 dtype_backend 是否不等于 lib.no_default
    if dtype_backend is not lib.no_default:
        # 如果 dtype_backend 不在有效的类型列表 ["numpy_nullable", "pyarrow"] 中，抛出值错误异常
        if dtype_backend not in ["numpy_nullable", "pyarrow"]:
            raise ValueError(
                f"dtype_backend {dtype_backend} is invalid, only 'numpy_nullable' and "
                f"'pyarrow' are allowed.",
            )
```