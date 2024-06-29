# `D:\src\scipysrc\pandas\pandas\compat\numpy\function.py`

```
"""
For compatibility with numpy libraries, pandas functions or methods have to
accept '*args' and '**kwargs' parameters to accommodate numpy arguments that
are not actually used or respected in the pandas implementation.

To ensure that users do not abuse these parameters, validation is performed in
'validators.py' to make sure that any extra parameters passed correspond ONLY
to those in the numpy signature. Part of that validation includes whether or
not the user attempted to pass in non-default values for these extraneous
parameters. As we want to discourage users from relying on these parameters
when calling the pandas implementation, we want them only to pass in the
default values for these parameters.

This module provides a set of commonly used default arguments for functions and
methods that are spread throughout the codebase. This module will make it
easier to adjust to future upstream changes in the analogous numpy signatures.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from numpy import ndarray

from pandas._libs.lib import (
    is_bool,
    is_integer,
)
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import (
    validate_args,
    validate_args_and_kwargs,
    validate_kwargs,
)

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        AxisInt,
    )

    AxisNoneT = TypeVar("AxisNoneT", Axis, None)


class CompatValidator:
    """
    A validator class for checking and validating arguments and keyword arguments
    passed to pandas functions, ensuring compatibility with numpy signatures.

    Attributes:
        defaults: Default arguments dictionary.
        fname: Function name associated with the validation.
        method: Validation method ('args', 'kwargs', or 'both').
        max_fname_arg_count: Maximum number of arguments expected for the function.
    """

    def __init__(
        self,
        defaults,
        fname=None,
        method: str | None = None,
        max_fname_arg_count=None,
    ) -> None:
        """
        Initialize the validator with default values and function details.

        Args:
            defaults: Default arguments dictionary.
            fname: Function name associated with the validation.
            method: Validation method ('args', 'kwargs', or 'both').
            max_fname_arg_count: Maximum number of arguments expected for the function.
        """
        self.fname = fname
        self.method = method
        self.defaults = defaults
        self.max_fname_arg_count = max_fname_arg_count

    def __call__(
        self,
        args,
        kwargs,
        fname=None,
        max_fname_arg_count=None,
        method: str | None = None,
    ) -> None:
        """
        Validate arguments and keyword arguments passed to the function.

        Args:
            args: Positional arguments passed to the function.
            kwargs: Keyword arguments passed to the function.
            fname: Function name associated with the validation.
            max_fname_arg_count: Maximum number of arguments expected for the function.
            method: Validation method ('args', 'kwargs', or 'both').
        """
        if not args and not kwargs:
            return None

        # Set defaults if not provided
        fname = self.fname if fname is None else fname
        max_fname_arg_count = (
            self.max_fname_arg_count
            if max_fname_arg_count is None
            else max_fname_arg_count
        )
        method = self.method if method is None else method

        # Validate based on the chosen method
        if method == "args":
            validate_args(fname, args, max_fname_arg_count, self.defaults)
        elif method == "kwargs":
            validate_kwargs(fname, kwargs, self.defaults)
        elif method == "both":
            validate_args_and_kwargs(
                fname, args, kwargs, max_fname_arg_count, self.defaults
            )
        else:
            raise ValueError(f"invalid validation method '{method}'")


ARGMINMAX_DEFAULTS = {"out": None}
validate_argmin = CompatValidator(
    ARGMINMAX_DEFAULTS, fname="argmin", method="both", max_fname_arg_count=1
)
validate_argmax = CompatValidator(
    ARGMINMAX_DEFAULTS, fname="argmax", method="both", max_fname_arg_count=1
)
    # 定义函数的参数和默认值
    ARGMINMAX_DEFAULTS, fname="argmax", method="both", max_fname_arg_count=1
# 定义一个函数，根据给定的 skipna 参数和其他参数来处理并返回一个元组
def process_skipna(skipna: bool | ndarray | None, args) -> tuple[bool, Any]:
    # 如果 skipna 是 ndarray 类型或者 None，则将其加入参数列表，并将 skipna 设为 True
    if isinstance(skipna, ndarray) or skipna is None:
        args = (skipna,) + args
        skipna = True

    return skipna, args


# 验证参数中的 skipna 是否为 ndarray 或 None，如果是，则调用 process_skipna 处理后返回 True，否则返回 False
def validate_argmin_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    """
    If 'Series.argmin' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """
    skipna, args = process_skipna(skipna, args)
    validate_argmin(args, kwargs)
    return skipna


# 验证参数中的 skipna 是否为 ndarray 或 None，如果是，则调用 process_skipna 处理后返回 True，否则返回 False
def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    """
    If 'Series.argmax' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """
    skipna, args = process_skipna(skipna, args)
    validate_argmax(args, kwargs)
    return skipna


# 默认的参数设置字典，用于 argsort 函数，初始化几个参数
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None
ARGSORT_DEFAULTS["stable"] = None


# 创建一个用于验证 argsort 函数参数的对象，使用 CompatValidator 进行验证
validate_argsort = CompatValidator(
    ARGSORT_DEFAULTS, fname="argsort", max_fname_arg_count=0, method="both"
)

# 另一种 argsort 函数的参数设置，这里验证带有 kind 参数的情况
ARGSORT_DEFAULTS_KIND: dict[str, int | None] = {}
ARGSORT_DEFAULTS_KIND["axis"] = -1
ARGSORT_DEFAULTS_KIND["order"] = None
ARGSORT_DEFAULTS_KIND["stable"] = None

# 创建一个用于验证带有 kind 参数的 argsort 函数参数的对象，使用 CompatValidator 进行验证
validate_argsort_kind = CompatValidator(
    ARGSORT_DEFAULTS_KIND, fname="argsort", max_fname_arg_count=0, method="both"
)


# 验证参数中的 ascending 是否为整数或 None，如果是，则将其加入参数列表，并将 ascending 设为 True
def validate_argsort_with_ascending(ascending: bool | int | None, args, kwargs) -> bool:
    """
    If 'Categorical.argsort' is called via the 'numpy' library, the first
    parameter in its signature is 'axis', which takes either an integer or
    'None', so check if the 'ascending' parameter has either integer type or is
    None, since 'ascending' itself should be a boolean
    """
    if is_integer(ascending) or ascending is None:
        args = (ascending,) + args
        ascending = True

    validate_argsort_kind(args, kwargs, max_fname_arg_count=3)
    ascending = cast(bool, ascending)
    return ascending


# 默认的参数设置字典，用于 clip 函数，初始化 out 参数
CLIP_DEFAULTS: dict[str, Any] = {"out": None}

# 创建一个用于验证 clip 函数参数的对象，使用 CompatValidator 进行验证
validate_clip = CompatValidator(
    CLIP_DEFAULTS, fname="clip", method="both", max_fname_arg_count=3
)


# 重载函数定义，用于处理 clip 函数的 axis 参数，支持两种不同的参数签名
@overload
def validate_clip_with_axis(axis: ndarray, args, kwargs) -> None: ...


@overload
def validate_clip_with_axis(axis: AxisNoneT, args, kwargs) -> AxisNoneT: ...


# 验证 clip 函数的 axis 参数是否为 ndarray 或 AxisNoneT 类型，使用不同的参数签名进行验证
def validate_clip_with_axis(
    axis: ndarray | AxisNoneT, args, kwargs
) -> AxisNoneT | None:
    """
    Validate the 'axis' parameter for the 'clip' function in numpy library.
    It can be either an ndarray or of type 'AxisNoneT'.
    """
    """
    如果通过 numpy 库调用 'NDFrame.clip'，其签名中的第三个参数是 'out'，可以接受一个 ndarray。
    因此，检查 'axis' 参数是否是 ndarray 的实例，因为 'axis' 本身应该是整数或 None。
    """
    if isinstance(axis, ndarray):
        # 如果 'axis' 是 ndarray 的实例，则将其作为参数和原有的 args 组合
        args = (axis,) + args
        # 错误：在赋值时类型不兼容（表达式的类型为 "None"，变量的类型为 "Union[ndarray[Any, Any], str, int]"）
        axis = None  # type: ignore[assignment]

    # 验证 clip 方法的参数有效性
    validate_clip(args, kwargs)
    # 错误：返回值类型不兼容（得到 "Union[ndarray[Any, Any], str, int]"，期望 "Union[str, int, None]"）
    return axis  # type: ignore[return-value]
# 定义默认的累积函数参数字典，初始为空字典
CUM_FUNC_DEFAULTS: dict[str, Any] = {}

# 设置累积函数参数字典的 "dtype" 键，默认为 None
CUM_FUNC_DEFAULTS["dtype"] = None

# 设置累积函数参数字典的 "out" 键，默认为 None
CUM_FUNC_DEFAULTS["out"] = None

# 创建一个用于验证累积函数的 CompatValidator 实例，使用默认参数字典，并指定验证方法和最大参数计数
validate_cum_func = CompatValidator(
    CUM_FUNC_DEFAULTS, method="both", max_fname_arg_count=1
)

# 创建一个用于验证累积和函数的 CompatValidator 实例，使用默认参数字典，并指定函数名、验证方法和最大参数计数
validate_cumsum = CompatValidator(
    CUM_FUNC_DEFAULTS, fname="cumsum", method="both", max_fname_arg_count=1
)


def validate_cum_func_with_skipna(skipna: bool, args, kwargs, name) -> bool:
    """
    If this function is called via the 'numpy' library, the third parameter in
    its signature is 'dtype', which takes either a 'numpy' dtype or 'None', so
    check if the 'skipna' parameter is a boolean or not
    """
    # 如果 skipna 参数不是布尔型，则将其添加到参数元组中并设置 skipna 为 True
    if not is_bool(skipna):
        args = (skipna,) + args
        skipna = True
    # 如果 skipna 参数是 numpy 中的布尔类型 np.bool_，则转换为 Python 布尔型
    elif isinstance(skipna, np.bool_):
        skipna = bool(skipna)

    # 使用 validate_cum_func 进行参数验证，确保参数符合预期
    validate_cum_func(args, kwargs, fname=name)
    # 返回 skipna 参数，经过可能的转换或更新
    return skipna


# 定义默认的 all 和 any 函数参数字典
ALLANY_DEFAULTS: dict[str, bool | None] = {}

# 设置 all 和 any 函数参数字典的 "dtype" 键，默认为 None
ALLANY_DEFAULTS["dtype"] = None

# 设置 all 和 any 函数参数字典的 "out" 键，默认为 None
ALLANY_DEFAULTS["out"] = None

# 设置 all 和 any 函数参数字典的 "keepdims" 键，默认为 False
ALLANY_DEFAULTS["keepdims"] = False

# 设置 all 和 any 函数参数字典的 "axis" 键，默认为 None
ALLANY_DEFAULTS["axis"] = None

# 创建一个用于验证 all 函数的 CompatValidator 实例，使用默认参数字典，并指定函数名、验证方法和最大参数计数
validate_all = CompatValidator(
    ALLANY_DEFAULTS, fname="all", method="both", max_fname_arg_count=1
)

# 创建一个用于验证 any 函数的 CompatValidator 实例，使用默认参数字典，并指定函数名、验证方法和最大参数计数
validate_any = CompatValidator(
    ALLANY_DEFAULTS, fname="any", method="both", max_fname_arg_count=1
)

# 定义默认的逻辑函数参数字典，包含 "out" 和 "keepdims" 键
LOGICAL_FUNC_DEFAULTS = {"out": None, "keepdims": False}

# 创建一个用于验证逻辑函数的 CompatValidator 实例，使用默认参数字典，并指定验证方法为关键字参数验证
validate_logical_func = CompatValidator(LOGICAL_FUNC_DEFAULTS, method="kwargs")

# 定义默认的最大最小函数参数字典，包含 "axis"、"dtype"、"out" 和 "keepdims" 键
MINMAX_DEFAULTS = {"axis": None, "dtype": None, "out": None, "keepdims": False}

# 创建一个用于验证最小值函数的 CompatValidator 实例，使用默认参数字典，并指定函数名、验证方法和最大参数计数
validate_min = CompatValidator(
    MINMAX_DEFAULTS, fname="min", method="both", max_fname_arg_count=1
)

# 创建一个用于验证最大值函数的 CompatValidator 实例，使用默认参数字典，并指定函数名、验证方法和最大参数计数
validate_max = CompatValidator(
    MINMAX_DEFAULTS, fname="max", method="both", max_fname_arg_count=1
)


# 定义默认的 repeat 函数参数字典，包含 "axis" 键，默认为 None
REPEAT_DEFAULTS: dict[str, Any] = {"axis": None}

# 创建一个用于验证 repeat 函数的 CompatValidator 实例，使用默认参数字典，并指定函数名、验证方法和最大参数计数
validate_repeat = CompatValidator(
    REPEAT_DEFAULTS, fname="repeat", method="both", max_fname_arg_count=1
)

# 定义默认的 round 函数参数字典，包含 "out" 键，默认为 None
ROUND_DEFAULTS: dict[str, Any] = {"out": None}

# 创建一个用于验证 round 函数的 CompatValidator 实例，使用默认参数字典，并指定函数名、验证方法和最大参数计数
validate_round = CompatValidator(
    ROUND_DEFAULTS, fname="round", method="both", max_fname_arg_count=1
)

# 定义默认的统计函数参数字典，初始为空字典
STAT_FUNC_DEFAULTS: dict[str, Any | None] = {}

# 设置统计函数参数字典的 "dtype" 键，默认为 None
STAT_FUNC_DEFAULTS["dtype"] = None

# 设置统计函数参数字典的 "out" 键，默认为 None
STAT_FUNC_DEFAULTS["out"] = None

# 创建一个用于验证统计函数的 CompatValidator 实例，使用默认参数字典，并指定验证方法为关键字参数验证
validate_stat_func = CompatValidator(STAT_FUNC_DEFAULTS, method="kwargs")

# 定义默认的求和函数参数字典，复制统计函数参数字典并添加 "axis"、"keepdims" 和 "initial" 键
SUM_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
SUM_DEFAULTS["axis"] = None
SUM_DEFAULTS["keepdims"] = False
SUM_DEFAULTS["initial"] = None

# 创建一个用于验证求和函数的 CompatValidator 实例，使用默认参数字典，并指定函数名、验证方法和最大参数计数
validate_sum = CompatValidator(
    SUM_DEFAULTS, fname="sum", method="both", max_fname_arg_count=1
)

# 创建一个用于验证乘积函数的 CompatValidator 实例，使用求和函数参数字典，并指定函数名、验证方法和最大参数计数
validate_prod = CompatValidator(
    PROD_DEFAULTS, fname="prod", method="both", max_fname_arg_count=1
)

# 创建一个用于验证平均数函数的 CompatValidator 实例，使用求和函数参数字典，并指定函数名、验证方法和最大参数计数
validate_mean = CompatValidator(
    MEAN_DEFAULTS, fname="mean", method="both", max_fname_arg_count=1
)

# 创建一个用于验证中位数函数的 CompatValidator 实例，使用统计函数参数字典，并指定函数名、验证方法和最大参数计数
validate_median = CompatValidator(
    MEDIAN_DEFAULTS, fname="median", method="both", max_fname_arg_count=1
)
# 默认的统计函数参数字典，用于存储统计函数的默认参数设置
STAT_DDOF_FUNC_DEFAULTS: dict[str, bool | None] = {}

# 设置默认的 dtype 参数为 None
STAT_DDOF_FUNC_DEFAULTS["dtype"] = None

# 设置默认的 out 参数为 None
STAT_DDOF_FUNC_DEFAULTS["out"] = None

# 设置默认的 keepdims 参数为 False
STAT_DDOF_FUNC_DEFAULTS["keepdims"] = False

# 创建一个验证器对象，用于验证统计函数的参数是否兼容
validate_stat_ddof_func = CompatValidator(STAT_DDOF_FUNC_DEFAULTS, method="kwargs")

# 默认的 take 函数参数字典，用于存储 take 函数的默认参数设置
TAKE_DEFAULTS: dict[str, str | None] = {}

# 设置默认的 out 参数为 None
TAKE_DEFAULTS["out"] = None

# 设置默认的 mode 参数为 "raise"
TAKE_DEFAULTS["mode"] = "raise"

# 创建一个验证器对象，用于验证 take 函数的参数是否兼容
validate_take = CompatValidator(TAKE_DEFAULTS, fname="take", method="kwargs")

# 默认的 transpose 函数参数字典，用于存储 transpose 函数的默认参数设置
TRANSPOSE_DEFAULTS = {"axes": None}

# 创建一个验证器对象，用于验证 transpose 函数的参数是否兼容
validate_transpose = CompatValidator(
    TRANSPOSE_DEFAULTS, fname="transpose", method="both", max_fname_arg_count=0
)

# 验证 groupby 函数参数是否有效的函数
def validate_groupby_func(name: str, args, kwargs, allowed=None) -> None:
    """
    'args' and 'kwargs' should be empty, except for allowed kwargs because all
    of their necessary parameters are explicitly listed in the function
    signature
    """
    if allowed is None:
        allowed = []

    # 从 kwargs 中移除 allowed 中指定的参数
    kwargs = set(kwargs) - set(allowed)

    # 如果 args 或者 kwargs 中有任何参数存在，则抛出异常
    if len(args) + len(kwargs) > 0:
        raise UnsupportedFunctionCall(
            "numpy operations are not valid with groupby. "
            f"Use .groupby(...).{name}() instead"
        )

# 验证 min, max, argmin, argmax 函数中 axis 参数的有效性的函数
def validate_minmax_axis(axis: AxisInt | None, ndim: int = 1) -> None:
    """
    Ensure that the axis argument passed to min, max, argmin, or argmax is zero
    or None, as otherwise it will be incorrectly ignored.

    Parameters
    ----------
    axis : int or None
    ndim : int, default 1

    Raises
    ------
    ValueError
    """
    # 如果 axis 参数为 None，则直接返回
    if axis is None:
        return

    # 如果 axis 超出了有效范围，则抛出 ValueError 异常
    if axis >= ndim or (axis < 0 and ndim + axis < 0):
        raise ValueError(f"`axis` must be fewer than the number of dimensions ({ndim})")

# 支持的统计函数及其对应的验证函数的映射关系字典
_validation_funcs = {
    "median": validate_median,
    "mean": validate_mean,
    "min": validate_min,
    "max": validate_max,
    "sum": validate_sum,
    "prod": validate_prod,
}

# 根据给定的函数名验证函数的参数是否有效的函数
def validate_func(fname, args, kwargs) -> None:
    # 如果函数名不在 _validation_funcs 字典中，则调用 validate_stat_func 进行验证
    if fname not in _validation_funcs:
        return validate_stat_func(args, kwargs, fname=fname)

    # 根据函数名从 _validation_funcs 字典中获取对应的验证函数
    validation_func = _validation_funcs[fname]

    # 调用获取到的验证函数进行参数验证
    return validation_func(args, kwargs)
```