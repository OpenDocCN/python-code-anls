# `D:\src\scipysrc\pandas\pandas\core\missing.py`

```
"""
Routines for filling missing data.
"""

from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    NaT,
    algos,
    lib,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    F,
    ReindexMethod,
    npt,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_numeric_dtype,
    is_numeric_v_string_like,
    is_object_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

if TYPE_CHECKING:
    from pandas import Index


def check_value_size(value, mask: npt.NDArray[np.bool_], length: int):
    """
    Validate the size of the values passed to ExtensionArray.fillna.
    """
    if is_array_like(value):
        if len(value) != length:
            raise ValueError(
                f"Length of 'value' does not match. Got ({len(value)}) "
                f" expected {length}"
            )
        value = value[mask]

    return value


def mask_missing(arr: ArrayLike, values_to_mask) -> npt.NDArray[np.bool_]:
    """
    Return a masking array of same size/shape as arr
    with entries equaling any member of values_to_mask set to True

    Parameters
    ----------
    arr : ArrayLike
    values_to_mask: list, tuple, or scalar

    Returns
    -------
    np.ndarray[bool]
    """
    # When called from Block.replace/replace_list, values_to_mask is a scalar
    # known to be holdable by arr.
    # When called from Series._single_replace, values_to_mask is tuple or list

    # Infer the dtype from values_to_mask to ensure compatibility with arr
    dtype, values_to_mask = infer_dtype_from(values_to_mask)

    # Convert values_to_mask to a numpy array of dtype if it's not already
    if isinstance(dtype, np.dtype):
        values_to_mask = np.array(values_to_mask, dtype=dtype)
    else:
        cls = dtype.construct_array_type()
        if not lib.is_list_like(values_to_mask):
            values_to_mask = [values_to_mask]
        values_to_mask = cls._from_sequence(values_to_mask, dtype=dtype, copy=False)

    potential_na = False
    if is_object_dtype(arr.dtype):
        # Create a mask for arr to avoid comparisons to NA values
        potential_na = True
        arr_mask = ~isna(arr)

    # Create a mask for NA values in values_to_mask
    na_mask = isna(values_to_mask)
    nonna = values_to_mask[~na_mask]

    # Initialize a boolean mask array with zeros of shape arr
    mask = np.zeros(arr.shape, dtype=bool)

    # Special case handling based on dtype comparisons
    if (
        is_numeric_dtype(arr.dtype)
        and not is_bool_dtype(arr.dtype)
        and is_bool_dtype(nonna.dtype)
    ):
        pass
    elif (
        is_bool_dtype(arr.dtype)
        and is_numeric_dtype(nonna.dtype)
        and not is_bool_dtype(nonna.dtype)
    ):
        pass
    else:
        # 遍历 nonna 中的每个元素
        for x in nonna:
            # 检查 x 是否为数值或类似字符串的类型
            if is_numeric_v_string_like(arr, x):
                # 如果是，执行以下操作以避免 numpy 的弃用警告
                # GH#29553 prevent numpy deprecation warnings
                pass
            else:
                # 如果不是数值或类似字符串的类型
                if potential_na:
                    # 创建一个新的布尔掩码，与 arr 形状相同
                    new_mask = np.zeros(arr.shape, dtype=np.bool_)
                    # 将新的布尔掩码设置为 arr 中等于 x 的位置
                    new_mask[arr_mask] = arr[arr_mask] == x
                else:
                    # 创建一个新的掩码，指示 arr 中等于 x 的位置
                    new_mask = arr == x

                    # 如果 new_mask 不是 numpy 数组
                    if not isinstance(new_mask, np.ndarray):
                        # 通常为 BooleanArray 类型，转换为 numpy 数组
                        new_mask = new_mask.to_numpy(dtype=bool, na_value=False)
                
                # 将当前 new_mask 合并到总的 mask 中
                mask |= new_mask

    # 检查是否存在 NA 值的掩码，并将其合并到总的 mask 中
    if na_mask.any():
        mask |= isna(arr)

    # 返回最终的 mask
    return mask
@overload
# 定义一个重载函数 clean_fill_method，接受参数 method 为 ["ffill", "pad", "bfill", "backfill"] 中的一个字符串
# 参数 allow_nearest 为字面值类型 False
def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill"],
    *,
    allow_nearest: Literal[False] = ...,
) -> Literal["pad", "backfill"]: ...


@overload
# 定义一个重载函数 clean_fill_method，接受参数 method 为 ["ffill", "pad", "bfill", "backfill", "nearest"] 中的一个字符串
# 参数 allow_nearest 为字面值类型 True
def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill", "nearest"],
    *,
    allow_nearest: Literal[True],
) -> Literal["pad", "backfill", "nearest"]: ...


def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill", "nearest"],
    *,
    allow_nearest: bool = False,
) -> Literal["pad", "backfill", "nearest"]:
    # 如果 method 是字符串类型
    if isinstance(method, str):
        # 将 method 转换为小写，忽略类型检查
        method = method.lower()  # type: ignore[assignment]
        # 根据不同的 method 字符串进行转换
        if method == "ffill":
            method = "pad"
        elif method == "bfill":
            method = "backfill"

    # 初始有效填充方法为 ["pad", "backfill"]
    valid_methods = ["pad", "backfill"]
    # 初始期望的填充方法描述
    expecting = "pad (ffill) or backfill (bfill)"
    # 如果允许最近邻方法
    if allow_nearest:
        # 添加最近邻方法到有效方法列表
        valid_methods.append("nearest")
        # 更新期望的填充方法描述
        expecting = "pad (ffill), backfill (bfill) or nearest"
    # 如果 method 不在有效方法列表中，抛出值错误异常
    if method not in valid_methods:
        raise ValueError(f"Invalid fill method. Expecting {expecting}. Got {method}")
    # 返回经过处理的填充方法
    return method


# 插值方法列表，用于调用 np.interp
NP_METHODS = ["linear", "time", "index", "values"]

# 插值方法列表，用于调用 _interpolate_scipy_wrapper
SP_METHODS = [
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "krogh",
    "spline",
    "polynomial",
    "from_derivatives",
    "piecewise_polynomial",
    "pchip",
    "akima",
    "cubicspline",
]


def clean_interp_method(method: str, index: Index, **kwargs) -> str:
    # 获取参数 kwargs 中的 order 值
    order = kwargs.get("order")

    # 对于 "spline", "polynomial" 方法，需要指定 order
    if method in ("spline", "polynomial") and order is None:
        raise ValueError("You must specify the order of the spline or polynomial.")

    # 所有有效的插值方法
    valid = NP_METHODS + SP_METHODS
    # 如果 method 不在有效方法列表中，抛出值错误异常
    if method not in valid:
        raise ValueError(f"method must be one of {valid}. Got '{method}' instead.")

    # 对于 "krogh", "piecewise_polynomial", "pchip" 方法，需要确保索引是单调递增的
    if method in ("krogh", "piecewise_polynomial", "pchip"):
        if not index.is_monotonic_increasing:
            raise ValueError(
                f"{method} interpolation requires that the index be monotonic."
            )

    # 返回经过验证的插值方法
    return method


def find_valid_index(how: str, is_valid: npt.NDArray[np.bool_]) -> int | None:
    """
    获取第一个有效值的位置索引。

    Parameters
    ----------
    how : {'first', 'last'}
        用于指定获取第一个或最后一个有效索引。
    is_valid: np.ndarray
        用于查找 na_values 的掩码。

    Returns
    -------
    int or None
    """
    # 确保 how 参数为 "first" 或 "last"
    assert how in ["first", "last"]

    # 如果 is_valid 的长度为 0，立即返回 None
    if len(is_valid) == 0:  # early stop
        return None

    # 如果 is_valid 是二维的，对 axis=1 进行任意值聚合
    if is_valid.ndim == 2:
        is_valid = is_valid.any(axis=1)  # reduce axis 1

    # 如果 how 是 "first"，获取第一个 True 的索引位置
    if how == "first":
        idxpos = is_valid[::].argmax()
    elif how == "last":
        # 如果 how 参数为 "last"，则计算最后一个有效元素的索引位置
        idxpos = len(is_valid) - 1 - is_valid[::-1].argmax()

    chk_notna = is_valid[idxpos]

    if not chk_notna:
        # 如果最后一个有效元素不存在，则返回 None
        return None
    # 以下注释是类型提示，指出返回值类型与预期不符合
    return idxpos  # type: ignore[return-value]
# 验证限制方向是否有效，并返回小写的限制方向字符串
def validate_limit_direction(
    limit_direction: str,
) -> Literal["forward", "backward", "both"]:
    # 定义有效的限制方向列表
    valid_limit_directions = ["forward", "backward", "both"]
    # 将输入的限制方向转换为小写
    limit_direction = limit_direction.lower()
    # 如果输入的限制方向不在有效的限制方向列表中，则引发值错误异常
    if limit_direction not in valid_limit_directions:
        raise ValueError(
            "Invalid limit_direction: expecting one of "
            f"{valid_limit_directions}, got '{limit_direction}'."
        )
    # 返回验证后的限制方向字符串
    return limit_direction  # type: ignore[return-value]


# 验证限制区域是否有效，并返回小写的限制区域字符串或者None
def validate_limit_area(limit_area: str | None) -> Literal["inside", "outside"] | None:
    # 如果限制区域不为None，则进行验证
    if limit_area is not None:
        # 定义有效的限制区域列表
        valid_limit_areas = ["inside", "outside"]
        # 将输入的限制区域转换为小写
        limit_area = limit_area.lower()
        # 如果输入的限制区域不在有效的限制区域列表中，则引发值错误异常
        if limit_area not in valid_limit_areas:
            raise ValueError(
                f"Invalid limit_area: expecting one of {valid_limit_areas}, got "
                f"{limit_area}."
            )
    # 返回验证后的限制区域字符串或者None
    return limit_area  # type: ignore[return-value]


# 推断并返回限制方向的有效值，根据给定的方法进行判断
def infer_limit_direction(
    limit_direction: Literal["backward", "forward", "both"] | None, method: str
) -> Literal["backward", "forward", "both"]:
    # 如果限制方向为None，则根据方法进行设置
    if limit_direction is None:
        if method in ("backfill", "bfill"):
            limit_direction = "backward"
        else:
            limit_direction = "forward"
    else:
        # 如果限制方向不为None，则根据方法进一步验证
        if method in ("pad", "ffill") and limit_direction != "forward":
            raise ValueError(
                f"`limit_direction` must be 'forward' for method `{method}`"
            )
        if method in ("backfill", "bfill") and limit_direction != "backward":
            raise ValueError(
                f"`limit_direction` must be 'backward' for method `{method}`"
            )
    # 返回推断后的限制方向字符串
    return limit_direction


# 根据指定的方法和索引类型，返回处理后的索引对象
def get_interp_index(method, index: Index) -> Index:
    # 如果方法是"linear"，则根据索引的类型进行相应的处理
    if method == "linear":
        # 导入必要的模块和类
        from pandas import Index
        # 如果索引的类型是DatetimeTZDtype或者是'mM'类型的numpy数据类型，则转换为int64
        if isinstance(index.dtype, DatetimeTZDtype) or lib.is_np_dtype(
            index.dtype, "mM"
        ):
            index = Index(index.view("i8"))  # 将索引转换为int64类型

        elif not is_numeric_dtype(index.dtype):
            # 对于非数值型且非日期时间类型的索引，保持与早期版本pandas的行为一致
            index = Index(range(len(index)))  # 使用索引长度创建一个新的索引范围
    # 如果 method 不在 NP_METHODS 和 SP_METHODS 的有效方法列表中，则引发值错误异常
    else:
        # 可接受的插值方法集合
        methods = {"index", "values", "nearest", "time"}
        # 判断索引是否为数值型或日期时间型
        is_numeric_or_datetime = (
            is_numeric_dtype(index.dtype)
            or isinstance(index.dtype, DatetimeTZDtype)
            or lib.is_np_dtype(index.dtype, "mM")
        )
        # 合并所有有效的插值方法
        valid = NP_METHODS + SP_METHODS
        # 如果指定的方法不在有效方法集合中，且不在可接受的方法集合中，并且索引列不是数值型或日期时间型，则引发值错误异常
        if method in valid:
            if method not in methods and not is_numeric_or_datetime:
                raise ValueError(
                    "Index column must be numeric or datetime type when "
                    f"using {method} method other than linear. "
                    "Try setting a numeric or datetime index column before "
                    "interpolating."
                )
        else:
            # 如果指定的方法不在有效方法集合中，则引发值错误异常
            raise ValueError(f"Can not interpolate with method={method}.")

    # 如果索引中存在缺失值，则引发未实现错误异常
    if isna(index).any():
        raise NotImplementedError(
            "Interpolation with NaNs in the index "
            "has not been implemented. Try filling "
            "those NaNs before interpolating."
        )
    # 返回索引
    return index
# 在二维数组 `data` 上进行就地插值操作，即在指定轴上应用插值方法
# `data`：浮点数类型的 NumPy 数组
# `index`：索引对象，用于指定插值的索引
# `axis`：整数，指定在哪个轴上进行插值操作
# `method`：字符串，默认为 "linear"，指定插值方法
# `limit`：整数或 None，指定限制的最大数量
# `limit_direction`：字符串，默认为 "forward"，指定限制的方向
# `limit_area`：字符串或 None，指定限制的区域
# `fill_value`：任意类型或 None，指定用于填充缺失值的值
# `mask`：可选，掩码数组，用于指定哪些值参与插值
# `**kwargs`：额外的关键字参数，传递给 _interpolate_1d 函数
def interpolate_2d_inplace(
    data: np.ndarray,  # floating dtype
    index: Index,
    axis: AxisInt,
    method: str = "linear",
    limit: int | None = None,
    limit_direction: str = "forward",
    limit_area: str | None = None,
    fill_value: Any | None = None,
    mask=None,
    **kwargs,
) -> None:
    """
    Column-wise application of _interpolate_1d.

    Notes
    -----
    Alters 'data' in-place.

    The signature does differ from _interpolate_1d because it only
    includes what is needed for Block.interpolate.
    """

    # 验证插值方法的有效性
    clean_interp_method(method, index, **kwargs)

    # 如果 fill_value 是有效的 NaN 值类型，则根据数据类型设置 fill_value
    if is_valid_na_for_dtype(fill_value, data.dtype):
        fill_value = na_value_for_dtype(data.dtype, compat=False)

    # 如果 method 是 "time"，则需要检查 index 的数据类型是否需要转换
    if method == "time":
        if not needs_i8_conversion(index.dtype):
            raise ValueError(
                "time-weighted interpolation only works "
                "on Series or DataFrames with a "
                "DatetimeIndex"
            )
        method = "values"

    # 验证 limit_direction 的有效性
    limit_direction = validate_limit_direction(limit_direction)
    # 验证 limit_area 的有效性
    limit_area_validated = validate_limit_area(limit_area)

    # 确定默认的限制数量，如果未指定则设为无限
    limit = algos.validate_limit(nobs=None, limit=limit)

    # 将 index 转换为适合传递给 _interpolate_1d 的索引数组
    indices = _index_to_interp_indices(index, method)

    def func(yvalues: np.ndarray) -> None:
        # 在指定的轴方向上处理一维切片数据
        _interpolate_1d(
            indices=indices,
            yvalues=yvalues,
            method=method,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area_validated,
            fill_value=fill_value,
            bounds_error=False,
            mask=mask,
            **kwargs,
        )

    # 错误：参数 1 传递给 "apply_along_axis" 的类型与预期不兼容
    # "Callable[[ndarray[Any, Any]], None]"; 预期为
    # "Callable[..., Union[_SupportsArray[dtype[<nothing>]], Sequence[_SupportsArray
    # [dtype[<nothing>]]], Sequence[Sequence[_SupportsArray[dtype[<nothing>]]]],
    # Sequence[Sequence[Sequence[_SupportsArray[dtype[<nothing>]]]]],
    # Sequence[Sequence[Sequence[Sequence[_SupportsArray[dtype[<nothing>]]]]]]]]"
    # 在指定的轴上应用 func 函数，将 func 应用于 data 的每个元素
    np.apply_along_axis(func, axis, data)  # type: ignore[arg-type]


# 将 Index 对象转换为用于传递给 NumPy/SciPy 的索引数组
def _index_to_interp_indices(index: Index, method: str) -> np.ndarray:
    """
    Convert Index to ndarray of indices to pass to NumPy/SciPy.
    """
    xarr = index._values
    # 如果需要将 dtype 转换为 'i8'，则进行转换，适用于 dt64tz 类型
    if needs_i8_conversion(xarr.dtype):
        # GH#1646 for dt64tz
        xarr = xarr.view("i8")

    # 如果 method 是 "linear"，直接使用 xarr 作为索引数组
    if method == "linear":
        inds = xarr
        inds = cast(np.ndarray, inds)
    else:
        # 否则将 xarr 转换为 NumPy 数组作为索引数组
        inds = np.asarray(xarr)

        # 如果 method 是 "values" 或 "index"，且索引数组的 dtype 是对象类型，则尝试转换为其他类型
        if method in ("values", "index"):
            if inds.dtype == np.object_:
                inds = lib.maybe_convert_objects(inds)

    return inds


# 在一维数据上应用插值方法
def _interpolate_1d(
    indices: np.ndarray,
    yvalues: np.ndarray,
    method: str = "linear",
    limit: int | None = None,
    limit_direction: str = "forward",
    limit_area: Literal["inside", "outside"] | None = None,
    fill_value: Any | None = None,
    bounds_error: bool = False,
    order: int | None = None,
    mask=None,
    **kwargs,
# 定义一个函数，用于进行一维插值操作
def _interpolate_1d(indices: np.ndarray, yvalues: np.ndarray, mask=None, limit=None,
                    limit_direction="both", limit_area=None, method="linear") -> None:
    """
    Logic for the 1-d interpolation.  The input
    indices and yvalues will each be 1-d arrays of the same length.

    Bounds_error is currently hardcoded to False since non-scipy ones don't
    take it as an argument.

    Notes
    -----
    Fills 'yvalues' in-place.
    """
    # 如果给定了掩码，则使用掩码标记无效值
    if mask is not None:
        invalid = mask
    else:
        # 否则，找出 yvalues 中的无效值
        invalid = isna(yvalues)
    valid = ~invalid

    # 如果没有有效值，直接返回
    if not valid.any():
        return

    # 如果所有值都有效，也直接返回
    if valid.all():
        return

    # 找出所有的 NaN 值的索引
    all_nans = np.flatnonzero(invalid)

    # 找出第一个有效值的索引
    first_valid_index = find_valid_index(how="first", is_valid=valid)
    if first_valid_index is None:  # 如果开头没有 NaN，则从头开始
        first_valid_index = 0
    start_nans = np.arange(first_valid_index)

    # 找出最后一个有效值的索引
    last_valid_index = find_valid_index(how="last", is_valid=valid)
    if last_valid_index is None:  # 如果结尾没有 NaN，则到结尾为止
        last_valid_index = len(yvalues)
    end_nans = np.arange(1 + last_valid_index, len(valid))

    # preserve_nans 包含需要保留为 NaN 的索引集合，根据插值限制和方向设置
    if limit_direction == "forward":
        preserve_nans = np.union1d(start_nans, _interp_limit(invalid, limit, 0))
    elif limit_direction == "backward":
        preserve_nans = np.union1d(end_nans, _interp_limit(invalid, 0, limit))
    else:
        preserve_nans = np.unique(_interp_limit(invalid, limit, limit))

    # 如果设置了 limit_area，根据设置添加内部或外部的索引到 preserve_nans
    if limit_area == "inside":
        preserve_nans = np.union1d(preserve_nans, start_nans)
        preserve_nans = np.union1d(preserve_nans, end_nans)
    elif limit_area == "outside":
        mid_nans = np.setdiff1d(all_nans, start_nans, assume_unique=True)
        mid_nans = np.setdiff1d(mid_nans, end_nans, assume_unique=True)
        preserve_nans = np.union1d(preserve_nans, mid_nans)

    # 判断 yvalues 是否为日期时间类型
    is_datetimelike = yvalues.dtype.kind in "mM"

    # 如果是日期时间类型，将 yvalues 视图转换为整数型
    if is_datetimelike:
        yvalues = yvalues.view("i8")

    # 如果插值方法为 numpy 中定义的方法，则使用 np.interp 进行插值操作
    if method in NP_METHODS:
        # np.interp 要求 X 值（这里是 indices）必须是排序的
        indexer = np.argsort(indices[valid])
        yvalues[invalid] = np.interp(
            indices[invalid], indices[valid][indexer], yvalues[valid][indexer]
        )
    # 如果存在无效数据索引，则使用插值方法填充无效位置的数值
    else:
        # 使用 SciPy 包装器对有效和无效索引处的数值进行插值
        yvalues[invalid] = _interpolate_scipy_wrapper(
            indices[valid],      # 有效数据索引
            yvalues[valid],      # 对应有效数据的数值
            indices[invalid],    # 无效数据索引
            method=method,       # 插值方法
            fill_value=fill_value,   # 填充值
            bounds_error=bounds_error,   # 越界错误处理
            order=order,         # 插值顺序
            **kwargs,            # 其他可选参数
        )

    # 如果存在遮罩数组，则进行下列操作
    if mask is not None:
        mask[:] = False           # 将遮罩数组全部置为 False
        mask[preserve_nans] = True   # 保留 NaN 值的位置设为 True
    # 如果是日期时间类型数据，则将保留 NaN 值的位置填充为 NaT
    elif is_datetimelike:
        yvalues[preserve_nans] = NaT.value
    # 否则，将保留 NaN 值的位置填充为 NaN
    else:
        yvalues[preserve_nans] = np.nan
    return
def _interpolate_scipy_wrapper(
    x: np.ndarray,
    y: np.ndarray,
    new_x: np.ndarray,
    method: str,
    fill_value=None,
    bounds_error: bool = False,
    order=None,
    **kwargs,
):
    """
    Passed off to scipy.interpolate.interp1d. method is scipy's kind.
    Returns an array interpolated at new_x.  Add any new methods to
    the list in _clean_interp_method.
    """
    # 导入必要的依赖 scipy
    extra = f"{method} interpolation requires SciPy."
    import_optional_dependency("scipy", extra=extra)
    from scipy import interpolate

    # 将 new_x 转换为 ndarray
    new_x = np.asarray(new_x)

    # 定义备选的插值方法字典
    alt_methods = {
        "barycentric": interpolate.barycentric_interpolate,
        "krogh": interpolate.krogh_interpolate,
        "from_derivatives": _from_derivatives,
        "piecewise_polynomial": _from_derivatives,
        "cubicspline": _cubicspline_interpolate,
        "akima": _akima_interpolate,
        "pchip": interpolate.pchip_interpolate,
    }

    # 插值方法的列表
    interp1d_methods = [
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "polynomial",
    ]
    
    # 根据指定的插值方法进行处理
    if method in interp1d_methods:
        # 对于多项式插值，需要使用指定的阶数
        if method == "polynomial":
            kind = order
        else:
            kind = method
        
        # 使用 scipy 的 interp1d 函数进行插值
        terp = interpolate.interp1d(
            x, y, kind=kind, fill_value=fill_value, bounds_error=bounds_error
        )
        # 对新的 x 进行插值得到新的 y 值
        new_y = terp(new_x)
    elif method == "spline":
        # 处理样条插值的情况
        if isna(order) or (order <= 0):
            raise ValueError(
                f"order needs to be specified and greater than 0; got order: {order}"
            )
        # 使用 UnivariateSpline 进行插值
        terp = interpolate.UnivariateSpline(x, y, k=order, **kwargs)
        new_y = terp(new_x)
    else:
        # 处理备选插值方法的情况
        # 如果数组不可写，则复制数组
        if not x.flags.writeable:
            x = x.copy()
        if not y.flags.writeable:
            y = y.copy()
        if not new_x.flags.writeable:
            new_x = new_x.copy()
        
        # 获取指定方法对应的插值函数
        terp = alt_methods.get(method, None)
        if terp is None:
            raise ValueError(f"Can not interpolate with method={method}.")
        
        # 确保 kwargs 中没有 downcast 参数
        kwargs.pop("downcast", None)
        
        # 使用备选方法进行插值
        new_y = terp(x, y, new_x, **kwargs)

    return new_y
    # 导入 scipy 中的 interpolate 模块
    from scipy import interpolate
    
    # 为了与不同版本的 scipy 兼容及向后兼容性，返回的方法是 interpolate.BPoly.from_derivatives
    method = interpolate.BPoly.from_derivatives
    
    # 使用 interpolate.BPoly.from_derivatives 方法创建一个 BPoly 对象 m
    # 该方法接受 xi 和 yi 作为输入，其中 yi 被重新形状为列向量 (-1, 1)
    # orders 参数指定了局部多项式的阶数，如果为 None，则一些导数被忽略
    # extrapolate 参数指定了是否对超出边界的点进行外推，True 表示进行外推
    m = method(xi, yi.reshape(-1, 1), orders=order, extrapolate=extrapolate)
    
    # 返回 BPoly 对象 m 在给定 x 值处的求值结果
    return m(x)
def _akima_interpolate(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    der: int | list[int] | None = 0,
    axis: AxisInt = 0,
):
    """
    Convenience function for akima interpolation.
    xi and yi are arrays of values used to approximate some function f,
    with ``yi = f(xi)``.

    See `Akima1DInterpolator` for details.

    Parameters
    ----------
    xi : np.ndarray
        A sorted list of x-coordinates, of length N.
    yi : np.ndarray
        A 1-D array of real values.  `yi`'s length along the interpolation
        axis must be equal to the length of `xi`. If N-D array, use axis
        parameter to select correct axis.
    x : np.ndarray
        Of length M.
    der : int, optional
        How many derivatives to extract; None for all potentially
        nonzero derivatives (that is a number equal to the number
        of points), or a list of derivatives to extract. This number
        includes the function value as 0th derivative.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    See Also
    --------
    scipy.interpolate.Akima1DInterpolator

    Returns
    -------
    y : scalar or array-like
        The result, of length R or length M or M by R,

    """
    from scipy import interpolate  # 导入 scipy.interpolate 模块中的 interpolate 函数

    # 创建 Akima1DInterpolator 对象 P，使用 xi 和 yi 进行初始化
    P = interpolate.Akima1DInterpolator(xi, yi, axis=axis)

    # 使用 P 对象进行插值计算，并返回结果
    return P(x, nu=der)


def _cubicspline_interpolate(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    axis: AxisInt = 0,
    bc_type: str | tuple[Any, Any] = "not-a-knot",
    extrapolate=None,
):
    """
    Convenience function for cubic spline data interpolator.

    See `scipy.interpolate.CubicSpline` for details.

    Parameters
    ----------
    xi : np.ndarray, shape (n,)
        1-d array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    yi : np.ndarray
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    x : np.ndarray, shape (m,)
        Array containing values where to interpolate.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : str or tuple of ({'not-a-knot', 'natural', 'clamped', 'periodic'}, optional)
        Type of boundary condition. Default is 'not-a-knot'.
    extrapolate : bool or 'periodic', optional
        Whether to extrapolate beyond the provided endpoints. If
        extrapolate is 'periodic', periodic extrapolation is used.

    Returns
    -------
    y : ndarray
        The interpolated values, shape determined by replacing
        the interpolation axis in `x` with the shape of `xi`.

    See Also
    --------
    scipy.interpolate.CubicSpline

    """
    bc_type : string or 2-tuple, optional
        Boundary condition type for the cubic spline interpolation.
        Two additional equations, defined by the boundary conditions, are
        necessary to determine coefficients of polynomials on each segment [2]_.
        
        If `bc_type` is a string, it applies the same condition at both ends
        of the spline. Available options include:
        * 'not-a-knot' (default): Ensures continuity up to the second derivative
          at each internal knot.
        * 'periodic': Assumes the interpolated function is periodic with a period
          determined by the range of `x`. Requires `y[0] == y[-1]`.
        * 'clamped': Sets the first derivative at curve ends to zero. 
          Equivalent to `bc_type=((1, 0.0), (1, 0.0))` for 1D `y`.
        * 'natural': Sets the second derivative at curve ends to zero.
          Equivalent to `bc_type=((2, 0.0), (2, 0.0))` for 1D `y`.
        
        If `bc_type` is a 2-tuple, the first and second values apply at the
        start and end of the curve respectively. These values can be strings
        (except 'periodic') or tuples `(order, deriv_values)` specifying custom
        derivatives at curve ends.

    extrapolate : {bool, 'periodic', None}, optional
        Determines extrapolation behavior for out-of-bounds points.
        - If bool: Extrapolates based on the first and last intervals if True,
          returns NaNs if False.
        - If 'periodic': Uses periodic extrapolation.
        - If None (default): Sets to 'periodic' if `bc_type='periodic'`, otherwise True.

    See Also
    --------
    scipy.interpolate.CubicHermiteSpline

    Returns
    -------
    y : scalar or array-like
        The result of cubic spline interpolation evaluated at `x`.

    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """
    from scipy import interpolate

    # 创建 CubicSpline 对象 P，用给定的 xi, yi 数据进行初始化
    P = interpolate.CubicSpline(
        xi, yi, axis=axis, bc_type=bc_type, extrapolate=extrapolate
    )

    # 返回在给定点 x 处的插值结果
    return P(x)
# 定义一个函数，用于在指定轴向上就地填充或后向填充数据
def pad_or_backfill_inplace(
    values: np.ndarray,
    method: Literal["pad", "backfill"] = "pad",  # 指定填充方法，默认为"pad"
    axis: AxisInt = 0,  # 指定填充的轴向，默认为0
    limit: int | None = None,  # 插值的最大数量限制
    limit_area: Literal["inside", "outside"] | None = None,  # 插值限制的区域，内部或外部
) -> None:
    """
    Perform an actual interpolation of values, values will be make 2-d if
    needed fills inplace, returns the result.

    Parameters
    ----------
    values: np.ndarray
        Input array.
    method: str, default "pad"
        Interpolation method. Could be "bfill" or "pad"
    axis: 0 or 1
        Interpolation axis
    limit: int, optional
        Index limit on interpolation.
    limit_area: str, optional
        Limit area for interpolation. Can be "inside" or "outside"

    Notes
    -----
    Modifies values in-place.
    """
    # 根据填充轴选择相应的转换函数
    transf = (lambda x: x) if axis == 0 else (lambda x: x.T)

    # 如果输入数组是一维的，根据轴向调整形状为二维
    if values.ndim == 1:
        if axis != 0:  # 如果轴不是0，则抛出错误
            raise AssertionError("cannot interpolate on a ndim == 1 with axis != 0")
        values = values.reshape(tuple((1,) + values.shape))

    # 清理填充方法名称，确保与内部标准一致
    method = clean_fill_method(method)
    tvalues = transf(values)

    # 根据填充方法获取相应的填充函数，这些函数会就地修改 tvalues
    func = get_fill_func(method, ndim=2)
    func(tvalues, limit=limit, limit_area=limit_area)


def _fillna_prep(
    values, mask: npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.bool_]:
    # 为 _pad_1d、_backfill_1d、_pad_2d、_backfill_2d 提供的通用预处理模板

    # 如果没有提供掩码，则根据数据是否缺失生成掩码
    if mask is None:
        mask = isna(values)

    return mask


def _datetimelike_compat(func: F) -> F:
    """
    Wrapper to handle datetime64 and timedelta64 dtypes.
    """
    
    @wraps(func)
    def new_func(
        values,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        mask=None,
    ):
        # 如果值的数据类型需要转换为 int64 型，则先生成对应的缺失值掩码
        if needs_i8_conversion(values.dtype):
            if mask is None:
                # 在转换为 int64 型之前，需要先生成掩码
                mask = isna(values)

            # 调用原始函数并返回转换后的结果及掩码
            result, mask = func(
                values.view("i8"), limit=limit, limit_area=limit_area, mask=mask
            )
            return result.view(values.dtype), mask

        # 直接调用原始函数处理值，如果不需要转换，则直接返回结果
        return func(values, limit=limit, limit_area=limit_area, mask=mask)

    return cast(F, new_func)


@_datetimelike_compat
def _pad_1d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # 对一维数组进行填充操作的函数，就地修改 values 和 mask
    mask = _fillna_prep(values, mask)
    
    # 如果指定了限制区域且掩码不全为 True，则在掩码中填充限制区域
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    
    # 调用算法模块中的 pad_inplace 函数就地填充 values，并返回填充后的 values 和 mask
    algos.pad_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _backfill_1d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # 对一维数组进行后向填充操作的函数，就地修改 values 和 mask
    mask = _fillna_prep(values, mask)
    
    # 如果指定了限制区域且掩码不全为 True，则在掩码中填充限制区域
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    
    # 调用算法模块中的 backfill_inplace 函数就地后向填充 values，并返回填充后的 values 和 mask
    algos.backfill_inplace(values, mask, limit=limit)
    return values, mask
    # 使用 _fillna_prep 函数对 values 和 mask 进行预处理，返回处理后的 mask
    mask = _fillna_prep(values, mask)
    # 如果指定了 limit_area 并且 mask 中存在 False 值（即未填充部分），则调用 _fill_limit_area_1d 函数
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    # 调用 algos 模块中的 backfill_inplace 函数，对 values 和 mask 进行就地向后填充操作，可选地限制填充数量为 limit
    algos.backfill_inplace(values, mask, limit=limit)
    # 返回填充后的 values 和 mask
    return values, mask
# 对函数进行日期时间兼容性修饰符的装饰
@_datetimelike_compat
# 用于在二维数据中进行填充操作，返回填充后的数据和掩码
def _pad_2d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # 准备填充前的掩码数据
    mask = _fillna_prep(values, mask)
    
    # 如果指定了填充限制区域，则调用 _fill_limit_area_2d 函数
    if limit_area is not None:
        _fill_limit_area_2d(mask, limit_area)

    # 如果数据数组不为空
    if values.size:
        # 使用 in-place 方法进行二维填充操作
        algos.pad_2d_inplace(values, mask, limit=limit)
    
    # 返回填充后的数据数组和掩码数组
    return values, mask


# 对函数进行日期时间兼容性修饰符的装饰
@_datetimelike_compat
# 用于在二维数据中进行反向填充操作，返回填充后的数据和掩码
def _backfill_2d(
    values,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
):
    # 准备填充前的掩码数据
    mask = _fillna_prep(values, mask)
    
    # 如果指定了填充限制区域，则调用 _fill_limit_area_2d 函数
    if limit_area is not None:
        _fill_limit_area_2d(mask, limit_area)

    # 如果数据数组不为空
    if values.size:
        # 使用 in-place 方法进行二维反向填充操作
        algos.backfill_2d_inplace(values, mask, limit=limit)
    else:
        # 为了测试覆盖率而添加的占位符分支
        pass
    
    # 返回填充后的数据数组和掩码数组
    return values, mask


# 准备一维掩码数据以进行前向/后向填充，并根据限制区域准备掩码
def _fill_limit_area_1d(
    mask: npt.NDArray[np.bool_], limit_area: Literal["outside", "inside"]
) -> None:
    """Prepare 1d mask for ffill/bfill with limit_area.

    Caller is responsible for checking at least one value of mask is False.
    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    """
    # 反转掩码并找到第一个和最后一个 False 值的索引
    neg_mask = ~mask
    first = neg_mask.argmax()
    last = len(neg_mask) - neg_mask[::-1].argmax() - 1
    
    # 根据限制区域设置掩码值
    if limit_area == "inside":
        mask[:first] = False
        mask[last + 1 :] = False
    elif limit_area == "outside":
        mask[first + 1 : last] = False


# 准备二维掩码数据以进行前向/后向填充，并根据限制区域准备掩码
def _fill_limit_area_2d(
    mask: npt.NDArray[np.bool_], limit_area: Literal["outside", "inside"]
) -> None:
    """Prepare 2d mask for ffill/bfill with limit_area.

    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    """
    # 反转掩码并根据限制区域设置掩码值
    neg_mask = ~mask.T
    if limit_area == "outside":
        # 确定外部填充区域
        la_mask = (
            np.maximum.accumulate(neg_mask, axis=0)
            & np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
        )
    else:
        # 确定内部填充区域
        la_mask = (
            ~np.maximum.accumulate(neg_mask, axis=0)
            | ~np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
        )
    
    # 根据计算得到的 la_mask 设置二维掩码
    mask[la_mask.T] = False


# 填充方法的字典，包含一维填充方法的名称和对应的函数
_fill_methods = {"pad": _pad_1d, "backfill": _backfill_1d}


# 根据填充方法名称获取相应的填充函数
def get_fill_func(method, ndim: int = 1):
    # 清理填充方法名称
    method = clean_fill_method(method)
    
    # 如果数据维度为一维，则返回对应方法名称的填充函数
    if ndim == 1:
        return _fill_methods[method]
    # 返回一个字典，根据给定的方法名选择性地返回对应的函数
    return {"pad": _pad_2d, "backfill": _backfill_2d}[method]
def clean_reindex_fill_method(method) -> ReindexMethod | None:
    # 如果方法为 None，则直接返回 None
    if method is None:
        return None
    # 调用 clean_fill_method 函数，允许使用最近的填充方法进行清理
    return clean_fill_method(method, allow_nearest=True)


def _interp_limit(
    invalid: npt.NDArray[np.bool_], fw_limit: int | None, bw_limit: int | None
) -> np.ndarray:
    """
    获取不会填充的值的索引器，因为它们超出了限制。

    Parameters
    ----------
    invalid : np.ndarray[bool]
        表示哪些值需要填充的布尔数组
    fw_limit : int or None
        正向索引的限制
    bw_limit : int or None
        反向索引的限制

    Returns
    -------
    np.ndarray
        索引器的集合

    Notes
    -----
    这等同于更易读但速度较慢的实现

    .. code-block:: python

        def _interp_limit(invalid, fw_limit, bw_limit):
            for x in np.where(invalid)[0]:
                if invalid[max(0, x - fw_limit) : x + bw_limit + 1].all():
                    yield x
    """
    # 处理正向索引；反向索引类似，但是
    # 1. 操作于反转的数组
    # 2. 从 N - 1 减去返回的索引
    N = len(invalid)
    # 初始化空数组，用于存储索引
    f_idx = np.array([], dtype=np.int64)
    b_idx = np.array([], dtype=np.int64)
    assume_unique = True

    # 内部函数，计算索引
    def inner(invalid, limit: int):
        # 将 limit 限制在 N 内
        limit = min(limit, N)
        # 使用滑动窗口查看技巧，检查是否全为 True
        windowed = np.lib.stride_tricks.sliding_window_view(invalid, limit + 1).all(1)
        # 计算并返回索引
        idx = np.union1d(
            np.where(windowed)[0] + limit,
            np.where((~invalid[: limit + 1]).cumsum() == 0)[0],
        )
        return idx

    # 如果存在正向限制
    if fw_limit is not None:
        # 如果正向限制为 0，则直接返回所有无效值的索引
        if fw_limit == 0:
            f_idx = np.where(invalid)[0]
            assume_unique = False
        else:
            # 否则，调用 inner 函数计算正向索引
            f_idx = inner(invalid, fw_limit)

    # 如果存在反向限制
    if bw_limit is not None:
        # 如果反向限制为 0，则直接返回正向索引结果
        if bw_limit == 0:
            return f_idx
        else:
            # 否则，计算反向索引
            b_idx = N - 1 - inner(invalid[::-1], bw_limit)
            # 如果正向限制为 0，则直接返回反向索引结果
            if fw_limit == 0:
                return b_idx

    # 返回正向索引和反向索引的交集
    return np.intersect1d(f_idx, b_idx, assume_unique=assume_unique)
```