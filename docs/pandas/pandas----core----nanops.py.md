# `D:\src\scipysrc\pandas\pandas\core\nanops.py`

```
# 引入了从未来模块中的类型注解，使得代码可以向后兼容 Python 3.9 之前的版本
from __future__ import annotations

# 引入了 functools 和 itertools 等工具模块，用于高阶函数和迭代器操作
import functools
import itertools

# 从 typing 模块中导入了类型相关的声明，用于类型提示
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

# 引入了警告模块，用于处理警告信息
import warnings

# 导入了 numpy 库，并使用 np 别名表示
import numpy as np

# 从 pandas._config 模块中导入了 get_option 函数
from pandas._config import get_option

# 从 pandas._libs 模块中导入了一些特定的符号
from pandas._libs import (
    NaT,
    NaTType,
    iNaT,
    lib,
)

# 从 pandas._typing 模块中导入了各种类型声明
from pandas._typing import (
    ArrayLike,
    AxisInt,
    CorrelationMethod,
    Dtype,
    DtypeObj,
    F,
    Scalar,
    Shape,
    npt,
)

# 从 pandas.compat._optional 模块中导入了 import_optional_dependency 函数
from pandas.compat._optional import import_optional_dependency

# 从 pandas.core.dtypes.common 模块中导入了一些用于数据类型检查的函数
from pandas.core.dtypes.common import (
    is_complex,
    is_float,
    is_float_dtype,
    is_integer,
    is_numeric_dtype,
    is_object_dtype,
    needs_i8_conversion,
    pandas_dtype,
)

# 从 pandas.core.dtypes.missing 模块中导入了一些缺失值处理相关的函数
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

# 如果是类型检查的模式，导入 collections.abc 模块中的 Callable 类型
if TYPE_CHECKING:
    from collections.abc import Callable

# 尝试导入 bottleneck 库，如果失败则给出警告
bn = import_optional_dependency("bottleneck", errors="warn")
# 检查是否成功导入 bottleneck 库
_BOTTLENECK_INSTALLED = bn is not None
# 默认不使用 bottleneck
_USE_BOTTLENECK = False


def set_use_bottleneck(v: bool = True) -> None:
    # 设置是否使用 bottleneck 的全局变量
    global _USE_BOTTLENECK
    # 如果 bottleneck 库可用，则根据参数设置使用或不使用
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v


# 根据配置文件中的设置来决定是否使用 bottleneck
set_use_bottleneck(get_option("compute.use_bottleneck"))


class disallow:
    def __init__(self, *dtypes: Dtype) -> None:
        # 初始化不允许的数据类型列表
        super().__init__()
        self.dtypes = tuple(pandas_dtype(dtype).type for dtype in dtypes)

    def check(self, obj) -> bool:
        # 检查对象是否属于不允许的数据类型之一
        return hasattr(obj, "dtype") and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:
        # 定义装饰器函数，用于检查参数是否包含不允许的数据类型
        @functools.wraps(f)
        def _f(*args, **kwargs):
            # 合并参数和关键字参数中的所有对象成为迭代器
            obj_iter = itertools.chain(args, kwargs.values())
            # 如果任何一个对象的数据类型属于不允许的类型，则抛出错误
            if any(self.check(obj) for obj in obj_iter):
                # 获取函数名，并移除其中的 'nan' 字符串
                f_name = f.__name__.replace("nan", "")
                # 抛出类型错误，说明该数据类型不支持对应的操作
                raise TypeError(
                    f"reduction operation '{f_name}' not allowed for this dtype"
                )
            try:
                # 尝试调用原始函数
                return f(*args, **kwargs)
            except ValueError as e:
                # 如果出现值错误，则根据情况转换为类型错误并抛出
                # 例如，通常这是在包含字符串的对象数组上不允许的函数操作
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise

        return cast(F, _f)


class bottleneck_switch:
    def __init__(self, name=None, **kwargs) -> None:
        # 初始化 bottleneck_switch 类的实例，保存名称和参数
        self.name = name
        self.kwargs = kwargs
    # 定义一个可调用对象，接受一个类型为 F 的参数并返回类型也为 F 的结果
    def __call__(self, alt: F) -> F:
        # 使用 self.name 或 alt.__name__ 作为备用函数名
        bn_name = self.name or alt.__name__

        try:
            # 尝试从 bn 对象中获取名为 bn_name 的属性或函数
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):  # 如果属性或函数不存在，捕获异常
            bn_func = None

        # 使用 functools.wraps 装饰器来保留原始函数 alt 的元数据
        @functools.wraps(alt)
        # 定义一个内部函数 f，接受一些参数，并返回计算结果
        def f(
            values: np.ndarray,
            *,
            axis: AxisInt | None = None,
            skipna: bool = True,
            **kwds,
        ):
            # 如果 self.kwargs 不为空，将其中的键值对添加到 kwds 中
            if len(self.kwargs) > 0:
                for k, v in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v

            # 如果 values 为空且 min_count 未指定，返回特定类型的 NA 值
            if values.size == 0 and kwds.get("min_count") is None:
                # 我们为空，根据类型返回 NA
                # 只适用于默认的 `min_count` 为 None 的情况
                # 因为这影响了如何处理空数组。
                # TODO(GH-18976) 更新所有的 nanops 方法以正确处理空输入并移除此检查。
                # 可能只是 `var`
                return _na_for_min_count(values, axis)

            # 如果启用了 Bottleneck 并且 skipna 为 True 且 bn_func 可以处理 values 的类型
            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get("mask", None) is None:
                    # Bottleneck 不识别 `mask` 参数，会导致 TypeError
                    # 如果存在，从 kwds 中移除 `mask` 参数
                    kwds.pop("mask", None)
                    # 使用 bn_func 计算结果
                    result = bn_func(values, axis=axis, **kwds)

                    # 更倾向于将 inf/-inf 视为 NA，但必须计算函数两次 :(
                    if _has_infs(result):
                        # 如果结果包含 inf/-inf，则用 alt 函数再次计算结果
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    # 如果存在 `mask` 参数，使用 alt 函数计算结果
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                # 如果未启用 Bottleneck 或者条件不满足，使用 alt 函数计算结果
                result = alt(values, axis=axis, skipna=skipna, **kwds)

            # 返回最终的计算结果
            return result

        # 返回内部函数 f 的类型转换后的结果
        return cast(F, f)
def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool:
    # Bottleneck chokes on datetime64, PeriodDtype (or and EA)
    # Bottleneck库对于datetime64和PeriodDtype类型（或and EA）存在问题
    if dtype != object and not needs_i8_conversion(dtype):
        # GH 42878
        # Bottleneck uses naive summation leading to O(n) loss of precision
        # unlike numpy which implements pairwise summation, which has O(log(n)) loss
        # crossref: https://github.com/pydata/bottleneck/issues/379
        # Bottleneck使用简单的求和方法导致O(n)的精度损失，
        # 而numpy实现了成对求和，精度损失为O(log(n))
        
        # GH 15507
        # bottleneck does not properly upcast during the sum
        # so can overflow
        # Bottleneck在求和过程中没有正确地进行上溯转换，可能会导致溢出

        # GH 9422
        # further we also want to preserve NaN when all elements
        # are NaN, unlike bottleneck/numpy which consider this
        # to be 0
        # 当所有元素都是NaN时，我们希望保留NaN，而不像bottleneck/numpy将其视为0
        return name not in ["nansum", "nanprod", "nanmean"]
    return False


def _has_infs(result) -> bool:
    if isinstance(result, np.ndarray):
        if result.dtype in ("f8", "f4"):
            # Note: outside of an nanops-specific test, we always have
            #  result.ndim == 1, so there is no risk of this ravel making a copy.
            # 注意：在nanops特定测试之外，我们始终有result.ndim == 1，因此这里的ravel不会复制数组。
            return lib.has_infs(result.ravel("K"))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        # if it doesn't support infs, then it can't have infs
        # 如果不支持无穷大，则返回False
        return False


def _get_fill_value(
    dtype: DtypeObj, fill_value: Scalar | None = None, fill_value_typ=None
):
    """return the correct fill value for the dtype of the values"""
    # 根据值的数据类型返回正确的填充值
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        else:
            if fill_value_typ == "+inf":
                return np.inf
            else:
                return -np.inf
    else:
        if fill_value_typ == "+inf":
            # need the max int here
            # 在这里需要最大整数
            return lib.i8max
        else:
            return iNaT


def _maybe_get_mask(
    values: np.ndarray, skipna: bool, mask: npt.NDArray[np.bool_] | None
) -> npt.NDArray[np.bool_] | None:
    """
    Compute a mask if and only if necessary.

    This function will compute a mask iff it is necessary. Otherwise,
    return the provided mask (potentially None) when a mask does not need to be
    computed.

    A mask is never necessary if the values array is of boolean or integer
    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable
    dtype that is interpretable as either boolean or integer data (eg,
    timedelta64), a mask must be provided.

    If the skipna parameter is False, a new mask will not be computed.

    The mask is computed using isna() by default. Setting invert=True selects
    notna() as the masking function.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    mask : Optional[ndarray]
        nan-mask if known

    Returns
    -------
    Optional[np.ndarray[bool]]
    """
    # 计算需要时的掩码（mask）


    Compute a mask if and only if necessary.

    This function will compute a mask iff it is necessary. Otherwise,
    return the provided mask (potentially None) when a mask does not need to be
    computed.

    A mask is never necessary if the values array is of boolean or integer
    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable
    dtype that is interpretable as either boolean or integer data (eg,
    timedelta64), a mask must be provided.

    If the skipna parameter is False, a new mask will not be computed.

    The mask is computed using isna() by default. Setting invert=True selects
    notna() as the masking function.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    mask : Optional[ndarray]
        nan-mask if known

    Returns
    -------
    Optional[np.ndarray[bool]]
    # 如果未提供掩码（mask），则根据数据类型判断是否为布尔型、整型或无符号整型
    if mask is None:
        # 如果数据类型是布尔型、整型或无符号整型，则无法包含空值，因此返回空的掩码（mask为None）
        if values.dtype.kind in "biu":
            # Boolean data cannot contain nulls, so signal via mask being None
            return None
        
        # 如果需要跳过空值或者数据类型是日期时间类型，则生成掩码
        if skipna or values.dtype.kind in "mM":
            # 调用isna函数生成掩码，标记数据中的空值
            mask = isna(values)

    # 返回生成的掩码（可能为None或者经过isna生成的布尔型数组）
    return mask
# 返回输入值数组的视图、可能的掩码、数据类型、数据类型的最大值和填充值的实用程序函数

def _get_values(
    values: np.ndarray,
    skipna: bool,
    fill_value: Any = None,
    fill_value_typ: str | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_] | None]:
    """
    Utility to get the values view, mask, dtype, dtype_max, and fill_value.

    If both mask and fill_value/fill_value_typ are not None and skipna is True,
    the values array will be copied.

    For input arrays of boolean or integer dtypes, copies will only occur if a
    precomputed mask, a fill_value/fill_value_typ, and skipna=True are
    provided.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    fill_value : Any
        value to fill NaNs with
    fill_value_typ : str
        Set to '+inf' or '-inf' to handle dtype-specific infinities
    mask : Optional[np.ndarray[bool]]
        nan-mask if known

    Returns
    -------
    values : ndarray
        Potential copy of input value array
    mask : Optional[ndarray[bool]]
        Mask for values, if deemed necessary to compute
    """

    # 从输入数组中获取可能的掩码
    mask = _maybe_get_mask(values, skipna, mask)

    # 获取输入数组的数据类型
    dtype = values.dtype

    # 检查数据类型是否为日期时间类型
    datetimelike = False
    if values.dtype.kind in "mM":
        # 如果是时间类型，则转换为int64类型以便后续处理
        values = np.asarray(values.view("i8"))
        datetimelike = True

    # 如果需要跳过NaN值，并且存在掩码
    if skipna and (mask is not None):
        # 获取填充值，用于替换NaN
        fill_value = _get_fill_value(
            dtype, fill_value=fill_value, fill_value_typ=fill_value_typ
        )

        # 如果填充值不为空
        if fill_value is not None:
            # 如果掩码中有任何True值
            if mask.any():
                # 如果是日期时间类型或者是可以接受NaN的数据类型
                if datetimelike or _na_ok_dtype(dtype):
                    # 复制数值数组，并用填充值替换掩码对应位置的值
                    values = values.copy()
                    np.putmask(values, mask, fill_value)
                else:
                    # 否则使用np.where根据掩码和填充值进行值替换
                    values = np.where(~mask, values, fill_value)

    # 返回处理后的数值数组和掩码
    return values, mask


def _get_dtype_max(dtype: np.dtype) -> np.dtype:
    # 返回与平台无关的最大精度数据类型
    dtype_max = dtype
    if dtype.kind in "bi":
        dtype_max = np.dtype(np.int64)
    elif dtype.kind == "u":
        dtype_max = np.dtype(np.uint64)
    elif dtype.kind == "f":
        dtype_max = np.dtype(np.float64)
    return dtype_max


def _na_ok_dtype(dtype: DtypeObj) -> bool:
    # 检查数据类型是否可以接受NaN值
    if needs_i8_conversion(dtype):
        return False
    return not issubclass(dtype.type, np.integer)


def _wrap_results(result, dtype: np.dtype, fill_value=None):
    """wrap our results if needed"""
    if result is NaT:
        pass
    # 如果数据类型是日期时间类型
    elif dtype.kind == "M":
        # 如果填充值为空
        if fill_value is None:
            # 设置填充值为 iNaT
            fill_value = iNaT
        # 如果结果不是 NumPy 数组
        if not isinstance(result, np.ndarray):
            # 断言填充值不为空，如果为空则抛出异常
            assert not isna(fill_value), "Expected non-null fill_value"
            # 如果结果等于填充值
            if result == fill_value:
                # 将结果设置为 NaN
                result = np.nan

            # 如果结果是 NaN
            if isna(result):
                # 将结果设置为 NaT 的日期时间类型
                result = np.datetime64("NaT", "ns").astype(dtype)
            else:
                # 将结果转换为 int64 类型，并按照指定的数据类型进行查看
                result = np.int64(result).view(dtype)
            # 保留原始单位
            result = result.astype(dtype, copy=False)
        else:
            # 如果数据类型是浮点型，直接转换为指定的数据类型
            result = result.astype(dtype)
    # 如果数据类型是时间间隔类型
    elif dtype.kind == "m":
        # 如果结果不是 NumPy 数组
        if not isinstance(result, np.ndarray):
            # 如果结果等于填充值或者是 NaN
            if result == fill_value or np.isnan(result):
                # 将结果设置为 NaT 的时间间隔类型
                result = np.timedelta64("NaT").astype(dtype)

            # 如果结果的绝对值大于 lib.i8max
            elif np.fabs(result) > lib.i8max:
                # 如果时间间隔过大，抛出异常
                raise ValueError("overflow in timedelta operation")
            else:
                # 返回一个带有原始单位的时间间隔类型
                result = np.int64(result).astype(dtype, copy=False)

        else:
            # 将结果转换为 m8[ns] 类型，并按照指定的数据类型进行查看
            result = result.astype("m8[ns]").view(dtype)

    # 返回处理后的结果
    return result
def _datetimelike_compat(func: F) -> F:
    """
    如果我们的值是 datetime64 或 timedelta64 类型，确保在调用包装函数之前具有正确的掩码，然后在之后进行类型转换。
    """

    @functools.wraps(func)
    def new_func(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: npt.NDArray[np.bool_] | None = None,
        **kwargs,
    ):
        orig_values = values  # 保存原始的输入值

        datetimelike = values.dtype.kind in "mM"  # 检查是否为 datetime64 或 timedelta64 类型
        if datetimelike and mask is None:
            mask = isna(values)  # 如果是时间类型且掩码为空，则使用 isna 函数生成掩码

        result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)  # 调用传入的函数处理数据

        if datetimelike:
            result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)  # 包装结果以恢复原始数据类型
            if not skipna:
                assert mask is not None  # 确保掩码已经生成（前面检查过）
                result = _mask_datetimelike_result(result, axis, mask, orig_values)  # 根据掩码处理结果数据

        return result  # 返回处理后的结果数据

    return cast(F, new_func)


def _na_for_min_count(values: np.ndarray, axis: AxisInt | None) -> Scalar | np.ndarray:
    """
    返回给定 `values` 的缺失值。

    Parameters
    ----------
    values : ndarray
    axis : int or None
        如果 values.ndim > 1，则需要进行缩减的轴。

    Returns
    -------
    result : scalar or ndarray
        对于 1-D 数组，返回正确缺失类型的标量。
        对于 2-D 数组，返回每个元素都是缺失值的 1-D 数组。
    """
    # 选择返回 np.nan 或 pd.NaT
    if values.dtype.kind in "iufcb":
        values = values.astype("float64")  # 如果是整数或布尔型，转换为 float64 类型
    fill_value = na_value_for_dtype(values.dtype)  # 根据数据类型确定缺失值

    if values.ndim == 1:
        return fill_value  # 对于一维数组，返回填充值
    elif axis is None:
        return fill_value  # 如果未指定轴，则返回填充值
    else:
        result_shape = values.shape[:axis] + values.shape[axis + 1 :]  # 计算结果数组的形状

        return np.full(result_shape, fill_value, dtype=values.dtype)  # 返回填充的多维数组


def maybe_operate_rowwise(func: F) -> F:
    """
    对于 C 连续的二维 ndarray，如果 axis=1 的长度远大于 axis=0，NumPy 的操作可能会非常慢。
    按行操作并拼接结果。
    """

    @functools.wraps(func)
    def wrapper(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        **kwargs,
    ):
        # 实现行级操作的具体逻辑在 func 函数中完成
        pass  # 占位符，实际操作在 func 函数中完成

    return cast(F, wrapper)
    # 定义一个新的函数，接受一个 numpy 数组作为输入，还可以接受指定的轴参数和任意额外的关键字参数
    def newfunc(values: np.ndarray, *, axis: AxisInt | None = None, **kwargs):
        # 如果满足以下条件，则执行特定的处理逻辑：
        # - 轴参数为 1
        # - 数组维度为 2
        # - 数组以 C 风格存储
        # - 第二维度的长度超过第一维度的长度乘以 1000
        # - 数组的数据类型既不是对象也不是布尔型
        if (
            axis == 1
            and values.ndim == 2
            and values.flags["C_CONTIGUOUS"]
            # 仅当数组宽度较大时采取此路径（例如长数据框），阈值详见 https://github.com/pandas-dev/pandas/pull/43311#issuecomment-974891737
            and (values.shape[1] / 1000) > values.shape[0]
            and values.dtype != object
            and values.dtype != bool
        ):
            # 将数组转换为列表形式
            arrs = list(values)
            # 如果传入的关键字参数中存在 "mask"，则将其从 kwargs 中取出
            if kwargs.get("mask") is not None:
                mask = kwargs.pop("mask")
                # 对每个数组执行函数 func，并传入对应的 mask 参数和其他关键字参数，将结果收集到 results 中
                results = [
                    func(arrs[i], mask=mask[i], **kwargs) for i in range(len(arrs))
                ]
            else:
                # 对每个数组执行函数 func，并传入其他关键字参数，将结果收集到 results 中
                results = [func(x, **kwargs) for x in arrs]
            # 将 results 转换为 numpy 数组并返回
            return np.array(results)

        # 如果不满足特定条件，则直接调用原始的 func 函数，传入相同的参数并返回结果
        return func(values, axis=axis, **kwargs)

    # 返回经类型转换后的新函数 newfunc
    return cast(F, newfunc)
# 检查是否存在沿指定轴的任意元素为 True
def nanany(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
        数组数据
    axis : int, optional
        沿着该轴进行操作，可选参数
    skipna : bool, default True
        是否跳过 NaN 值，默认为 True
    mask : ndarray[bool], optional
        NaN 掩码，如果已知的话

    Returns
    -------
    result : bool
        结果为 True 或 False

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s.values)
    False
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.any(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        raise TypeError("datetime64 type does not support operation 'any'")

    values, _ = _get_values(values, skipna, fill_value=False, mask=mask)

    # 对于对象类型，any 函数不一定返回布尔值 (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.any(axis)  # type: ignore[return-value]


# 检查是否沿指定轴的所有元素都为 True
def nanall(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    """
    Check if all elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
        数组数据
    axis : int, optional
        沿着该轴进行操作，可选参数
    skipna : bool, default True
        是否跳过 NaN 值，默认为 True
    mask : ndarray[bool], optional
        NaN 掩码，如果已知的话

    Returns
    -------
    result : bool
        结果为 True 或 False

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanall(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 0])
    >>> nanops.nanall(s.values)
    False
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.all(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        raise TypeError("datetime64 type does not support operation 'all'")

    values, _ = _get_values(values, skipna, fill_value=True, mask=mask)

    # 对于对象类型，all 函数不一定返回布尔值 (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.all(axis)  # type: ignore[return-value]


@disallow("M8")
@_datetimelike_compat
@maybe_operate_rowwise
def nansum(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Sum the elements along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray[dtype]
        Array containing the elements to sum.
    axis : int, optional
        The axis along which to sum. If None, sum all elements.
    skipna : bool, default True
        Whether to exclude NaN values when summing.
    min_count: int, default 0
        Minimum number of valid values required to compute the result; if fewer
        than min_count non-NaN values are present, the result will be NaN.
    mask : ndarray[bool], optional
        Optional boolean mask indicating NaN positions in `values`.

    Returns
    -------
    result : dtype
        Sum of the elements along the specified axis.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nansum(s.values)
    3.0
    """
    # Determine the dtype of the input values
    dtype = values.dtype
    # Obtain the valid values and corresponding mask after handling NaNs
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    # Determine the dtype to be used for summing
    dtype_sum = _get_dtype_max(dtype)
    # Adjust dtype_sum based on the kind of dtype
    if dtype.kind == "f":
        dtype_sum = dtype
    elif dtype.kind == "m":
        dtype_sum = np.dtype(np.float64)

    # Compute the sum of values along the specified axis
    the_sum = values.sum(axis, dtype=dtype_sum)
    # Adjust the_sum considering the mask and minimum count requirement
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)

    return the_sum


def _mask_datetimelike_result(
    result: np.ndarray | np.datetime64 | np.timedelta64,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_],
    orig_values: np.ndarray,
) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
    """
    Mask the result array or datetime/timedelta result based on the provided mask.

    Parameters
    ----------
    result : ndarray or datetime64 or timedelta64
        Result array or datetime/timedelta result to be masked.
    axis : int, optional
        The axis along which to apply the mask.
    mask : ndarray[bool]
        Boolean mask indicating positions to be masked.
    orig_values : ndarray
        Original array values corresponding to the result.

    Returns
    -------
    masked_result : ndarray or datetime64 or timedelta64 or NaTType
        Masked result based on the provided mask.

    Notes
    -----
    This function modifies 'result' in place using the provided 'mask'.

    Examples
    --------
    >>> result = np.array([1, 2, 3])
    >>> mask = np.array([False, True, False])
    >>> orig_values = np.array([1, 2, 3])
    >>> _mask_datetimelike_result(result, None, mask, orig_values)
    array([1, NaT, 3], dtype='timedelta64[ns]')
    """
    if isinstance(result, np.ndarray):
        # Convert result to int64 and view as original dtype, then apply mask
        result = result.astype("i8").view(orig_values.dtype)
        # Determine which positions in result to mask out based on axis and mask
        axis_mask = mask.any(axis=axis)
        # Replace masked positions with iNaT (Not-a-Time)
        result[axis_mask] = iNaT  # type: ignore[index]
    else:
        # If result is not an ndarray, return NaT if any mask positions are True
        if mask.any():
            return np.int64(iNaT).view(orig_values.dtype)
    return result


@bottleneck_switch()
@_datetimelike_compat
def nanmean(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the mean of the element along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray
        Array containing the elements to compute the mean.
    axis : int, optional
        The axis along which to compute the mean. If None, compute over all elements.
    skipna : bool, default True
        Whether to exclude NaN values when computing the mean.
    mask : ndarray[bool], optional
        Optional boolean mask indicating NaN positions in `values`.

    Returns
    -------
    float
        Mean of the elements along the specified axis.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanmean(s.values)
    1.5
    """
    # Determine the dtype of the input values
    dtype = values.dtype
    # Obtain the valid values and corresponding mask after handling NaNs
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    # Determine the dtype to be used for summing (float64)
    dtype_sum = _get_dtype_max(dtype)
    # Always use float64 for counting non-NaN values
    dtype_count = np.dtype(np.float64)

    # Adjust dtype_sum and dtype_count based on the kind of dtype
    if dtype.kind in "mM":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind in "iu":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind == "f":
        dtype_sum = dtype
        dtype_count = dtype
    # 调用函数 _get_counts 获取符合条件的值的数量
    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    # 求取沿指定轴的总和
    the_sum = values.sum(axis, dtype=dtype_sum)
    # 确保 the_sum 是数值型数据
    the_sum = _ensure_numeric(the_sum)

    # 如果指定了轴并且 the_sum 是多维数组
    if axis is not None and getattr(the_sum, "ndim", False):
        # 将 count 强制转换为 NumPy 数组类型
        count = cast(np.ndarray, count)
        # 忽略所有错误状态（例如除以零的警告）
        with np.errstate(all="ignore"):
            # 计算均值，可能会有除以零的情况，但不会引发警告
            the_mean = the_sum / count
        # 找到所有 count 为零的位置
        ct_mask = count == 0
        # 如果有 count 为零的情况，将对应的均值设为 NaN
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        # 如果 count 大于零，则计算均值；否则设为 NaN
        the_mean = the_sum / count if count > 0 else np.nan

    # 返回计算得到的均值
    return the_mean
# 装饰器，用于标记函数作为性能瓶颈的候选
@bottleneck_switch()
# 计算给定数组的中位数，支持沿指定轴进行计算，可以跳过 NaN 值
def nanmedian(values, *, axis: AxisInt | None = None, skipna: bool = True, mask=None):
    """
    Parameters
    ----------
    values : ndarray
        输入的数组
    axis : int, optional
        沿此轴计算中位数，默认为 None
    skipna : bool, default True
        是否跳过 NaN 值，默认为 True
    mask : ndarray[bool], optional
        NaN 值的遮罩数组，如果已知的话

    Returns
    -------
    result : float
        返回中位数，除非输入数组为浮点数组，在这种情况下，返回与输入数组相同的精度。

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 2])
    >>> nanops.nanmedian(s.values)
    2.0

    >>> s = pd.Series([np.nan, np.nan, np.nan])
    >>> nanops.nanmedian(s.values)
    nan
    """

    # 如果数据类型为浮点且没有提供 mask，使用 NaN 作为缺失值标识，下面会从数据中计算出 mask
    using_nan_sentinel = values.dtype.kind == "f" and mask is None

    # 内部函数，用于计算数组的中位数，可以根据给定的 mask 进行过滤
    def get_median(x, _mask=None):
        if _mask is None:
            _mask = notna(x)  # 使用 notna 函数创建 mask
        else:
            _mask = ~_mask  # 反转 mask
        if not skipna and not _mask.all():  # 如果不跳过 NaN 值且 mask 不全为 True，则返回 NaN
            return np.nan
        with warnings.catch_warnings():
            # 忽略关于全 NaN 切片的 RuntimeWarning
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            # 忽略关于空切片均值的 RuntimeWarning
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            res = np.nanmedian(x[_mask])  # 计算经过 mask 过滤后的中位数
        return res

    dtype = values.dtype
    # 获取实际的数值数组和对应的 mask
    values, mask = _get_values(values, skipna, mask=mask, fill_value=None)

    # 如果数值数组的类型不是浮点型
    if values.dtype.kind != "f":
        if values.dtype == object:
            # 避免将字符串转换为数值类型，如果是字符串则抛出错误
            inferred = lib.infer_dtype(values)
            if inferred in ["string", "mixed"]:
                raise TypeError(f"Cannot convert {values} to numeric")
        try:
            values = values.astype("f8")  # 尝试将数组转换为浮点类型
        except ValueError as err:
            # 捕获可能的值错误，例如无法将字符串转换为浮点数
            raise TypeError(str(err)) from err

    # 如果不是使用 NaN 作为缺失值标识，并且提供了 mask，则将 mask 应用到数值数组上
    if not using_nan_sentinel and mask is not None:
        if not values.flags.writeable:
            values = values.copy()
        values[mask] = np.nan  # 将 mask 中对应位置的数值设为 NaN

    notempty = values.size  # 获取数组的元素数量

    # 从数据框中获取的数组
    # 检查数组的维度是否大于1并且 axis 参数不为 None
    if values.ndim > 1 and axis is not None:
        # 如果数组非空，则进行操作；否则 numpy 抛出异常
        if notempty:
            # 如果不跳过 NaN 值
            if not skipna:
                # 沿指定轴应用 get_median 函数
                res = np.apply_along_axis(get_median, axis, values)
            else:
                # 对于跳过 NaN 值的情况，采用快速路径
                with warnings.catch_warnings():
                    # 忽略关于全 NaN 切片遇到的 RuntimeWarning
                    warnings.filterwarnings(
                        "ignore", "All-NaN slice encountered", RuntimeWarning
                    )
                    # 如果数组是可压缩的，并且在轴为0或1的情况下，采用快速路径
                    if (values.shape[1] == 1 and axis == 0) or (
                        values.shape[0] == 1 and axis == 1
                    ):
                        # GH52788: 对于可压缩的情况，对2D数组进行 nanmedian 操作速度较慢
                        res = np.nanmedian(np.squeeze(values), keepdims=True)
                    else:
                        # 否则对数组进行 nanmedian 操作
                        res = np.nanmedian(values, axis=axis)
        else:
            # 数组为空集时，返回对应形状的 NaN 值，因为对空集定义中位数是未定义的
            res = _get_empty_reduction_result(values.shape, axis)
    else:
        # 如果数组的维度为1或者 axis 参数为 None，则返回标量值
        res = get_median(values, mask) if notempty else np.nan
    # 将结果进行包装并返回
    return _wrap_results(res, dtype)
# 定义一个函数用于处理空的 ndarray 在进行减少操作时的结果
def _get_empty_reduction_result(
    shape: Shape,
    axis: AxisInt,
) -> np.ndarray:
    """
    The result from a reduction on an empty ndarray.

    Parameters
    ----------
    shape : Tuple[int, ...]
        ndarray 的形状
    axis : int
        指定的轴

    Returns
    -------
    np.ndarray
        返回一个空的 ndarray，填充为 NaN
    """
    shp = np.array(shape)  # 将形状转换为 NumPy 数组
    dims = np.arange(len(shape))  # 创建一个包含数组维度的索引数组
    ret = np.empty(shp[dims != axis], dtype=np.float64)  # 创建一个指定轴以外的形状的空 ndarray，数据类型为 float64
    ret.fill(np.nan)  # 用 NaN 填充创建的 ndarray
    return ret  # 返回填充了 NaN 的 ndarray


# 定义一个函数用于获取 nanvar 的计数
def _get_counts_nanvar(
    values_shape: Shape,
    mask: npt.NDArray[np.bool_] | None,
    axis: AxisInt | None,
    ddof: int,
    dtype: np.dtype = np.dtype(np.float64),
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Get the count of non-null values along an axis, accounting
    for degrees of freedom.

    Parameters
    ----------
    values_shape : Tuple[int, ...]
        values ndarray 的形状，如果 mask 为 None 时使用
    mask : Optional[ndarray[bool]]
        values 中被视为缺失的位置
    axis : Optional[int]
        要计数的轴
    ddof : int
        自由度
    dtype : type, optional
        计数使用的数据类型

    Returns
    -------
    count : int, np.nan or np.ndarray
        非空值的计数，可能是整数、NaN 或者 ndarray
    d : int, np.nan or np.ndarray
        自由度，可能是整数、NaN 或者 ndarray
    """
    count = _get_counts(values_shape, mask, axis, dtype=dtype)  # 获取非空值的计数
    d = count - dtype.type(ddof)  # 计算自由度

    # 始终返回 NaN，不返回无穷大
    if is_float(count):  # 检查计数是否为浮点数
        if count <= ddof:
            count = np.nan  # 如果计数小于等于自由度，则设为 NaN
            d = np.nan
    else:
        count = cast(np.ndarray, count)  # 将 count 强制转换为 ndarray
        mask = count <= ddof
        if mask.any():
            np.putmask(d, mask, np.nan)  # 使用 mask 将 d 中符合条件的位置设为 NaN
            np.putmask(count, mask, np.nan)  # 使用 mask 将 count 中符合条件的位置设为 NaN
    return count, d  # 返回计数和自由度


# 使用 bottleneck_switch 装饰器定义 nanstd 函数
@bottleneck_switch(ddof=1)
def nanstd(
    values,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask=None,
):
    """
    Compute the standard deviation along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
        输入数组
    axis : int, optional
        要沿其计算标准差的轴
    skipna : bool, default True
        是否跳过 NaN 值
    ddof : int, default 1
        自由度。计算中使用的除数是 N - ddof，其中 N 表示元素个数。
    mask : ndarray[bool], optional
        已知的 NaN 掩码

    Returns
    -------
    result : float
        除非输入是浮点数组，否则使用与输入数组相同的精度。

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanstd(s.values)
    1.0
    """
    if values.dtype == "M8[ns]":
        values = values.view("m8[ns]")  # 将日期时间数组视图转换为日期时间

    orig_dtype = values.dtype  # 保存原始的数据类型
    values, mask = _get_values(values, skipna, mask=mask)  # 获取处理后的数组和掩码
    # 计算给定数据中非NaN值的方差，并返回其算术平方根
    result = np.sqrt(nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask))
    # 将计算结果进行包装并返回
    return _wrap_results(result, orig_dtype)
# 应用装饰器 @disallow("M8", "m8")，阻止 "M8" 和 "m8" 作为参数传递给函数
# 应用装饰器 @bottleneck_switch(ddof=1)，启用性能优化，指定 ddof=1
def nanvar(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,  # 定义参数 axis，可以是整数或 None
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    ddof: int = 1,  # Delta 自由度，默认为 1
    mask=None,  # 可选参数，NaN 掩码数组
):
    """
    Compute the variance along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray  # 输入数组
    axis : int, optional  # 计算方差的轴，可选参数
    skipna : bool, default True  # 是否跳过 NaN 值，默认为 True
    ddof : int, default 1  # Delta 自由度，N - ddof 是计算中的除数
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional  # NaN 掩码数组，如果已知

    Returns
    -------
    result : float  # 返回计算得到的方差
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanvar(s.values)
    1.0
    """
    dtype = values.dtype  # 获取输入数组的数据类型
    mask = _maybe_get_mask(values, skipna, mask)  # 获取 NaN 掩码

    # 如果数据类型是整数或无符号整数，转换为双精度浮点数类型
    if dtype.kind in "iu":
        values = values.astype("f8")
        # 如果有掩码，则将掩码位置设置为 NaN
        if mask is not None:
            values[mask] = np.nan

    # 获取计数和除数
    if values.dtype.kind == "f":
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof)

    # 如果需要跳过 NaN 值并且存在掩码，则将掩码位置设置为 0
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)

    # 计算平均值 avg
    avg = _ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count
    if axis is not None:
        avg = np.expand_dims(avg, axis)

    # 计算平方差 sqr
    sqr = _ensure_numeric((avg - values) ** 2)
    # 如果存在掩码，则将掩码位置的 sqr 设置为 0
    if mask is not None:
        np.putmask(sqr, mask, 0)

    # 计算最终的方差结果
    result = sqr.sum(axis=axis, dtype=np.float64) / d

    # 返回方差结果，数据类型为 np.float64（累加器中使用的数据类型）
    # 如果输入数组是浮点数组，则保持与原始数组相同的精度
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    return result


@disallow("M8", "m8")  # 阻止 "M8" 和 "m8" 作为参数传递给函数
def nansem(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,  # 定义参数 axis，可以是整数或 None
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    ddof: int = 1,  # Delta 自由度，默认为 1
    mask: npt.NDArray[np.bool_] | None = None,  # NaN 掩码数组，可选参数
) -> float:
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray  # 输入数组
    axis : int, optional  # 计算标准误差的轴，可选参数
    skipna : bool, default True  # 是否跳过 NaN 值，默认为 True
    ddof : int, default 1  # Delta 自由度，N - ddof 是计算中的除数
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional  # NaN 掩码数组，如果已知

    Returns
    -------
    float  # 返回计算得到的标准误差

    """
    # 初始化变量 result 作为 float64 类型的浮点数
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s.values)
     0.5773502691896258
    """
    # 检查如果 values 不是 float 数组且 numeric_only=False，会引发 TypeError
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    # 获取可能的掩码 mask
    mask = _maybe_get_mask(values, skipna, mask)
    
    # 如果 values 的数据类型不是 "f"（float），则将其转换为 "f8"（float64）
    if values.dtype.kind != "f":
        values = values.astype("f8")

    # 如果不跳过 NaN 值且存在 mask，则返回 NaN
    if not skipna and mask is not None and mask.any():
        return np.nan

    # 获取计数和其他相关值以计算 nanvar
    count, _ = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    # 返回计算后的标准误差值
    return np.sqrt(var) / np.sqrt(count)
# 定义一个函数 _nanminmax，用于生成处理 NaN 值的最小值和最大值函数
def _nanminmax(meth, fill_value_typ):
    # 使用装饰器定义一个内部函数 reduction，用于执行具体的最小值或最大值计算
    @bottleneck_switch(name=f"nan{meth}")
    @_datetimelike_compat
    def reduction(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: npt.NDArray[np.bool_] | None = None,
    ):
        # 如果 values 的大小为 0，则调用 _na_for_min_count 函数处理返回结果
        if values.size == 0:
            return _na_for_min_count(values, axis)

        # 获取处理后的 values 和 mask，其中跳过 NaN 值，并使用给定的 fill_value_typ 填充 NaN
        values, mask = _get_values(
            values, skipna, fill_value_typ=fill_value_typ, mask=mask
        )
        # 调用 values 的特定方法（meth），计算最小值或最大值
        result = getattr(values, meth)(axis)
        # 对计算结果进行可能的空值处理，根据 axis、mask 和 values 的形状来处理
        result = _maybe_null_out(result, axis, mask, values.shape)
        return result

    return reduction

# 使用 _nanminmax 函数创建 nanmin 函数，用于计算数组的最小值（跳过 NaN 值）
nanmin = _nanminmax("min", fill_value_typ="+inf")
# 使用 _nanminmax 函数创建 nanmax 函数，用于计算数组的最大值（跳过 NaN 值）
nanmax = _nanminmax("max", fill_value_typ="-inf")


# 定义 nanargmax 函数，用于返回数组中最大值的索引
def nanargmax(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices  of max value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmax(arr)
    4

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 2] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7., nan],
           [ 9., 10., nan]])
    >>> nanops.nanargmax(arr, axis=1)
    array([2, 2, 1, 1])
    """
    # 获取处理后的 values 和 mask，跳过 NaN 值，并使用 "-inf" 填充 NaN
    values, mask = _get_values(values, True, fill_value_typ="-inf", mask=mask)
    # 计算在指定 axis 上的最大值的索引
    result = values.argmax(axis)
    # 对计算结果进行可能的空值处理，根据 axis、mask 和 skipna 来处理
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    return result


# 定义 nanargmin 函数，用于返回数组中最小值的索引
def nanargmin(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices of min value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmin(arr)
    0

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 0] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [nan,  7.,  8.],
           [nan, 10., 11.]])
    >>> nanops.nanargmin(arr, axis=1)
    array([0, 0, 1, 1])
    """
    # 获取处理后的 values 和 mask，跳过 NaN 值，并使用 "-inf" 填充 NaN
    values, mask = _get_values(values, True, fill_value_typ="-inf", mask=mask)
    # 计算在指定 axis 上的最小值的索引
    result = values.argmin(axis)
    # 对计算结果进行可能的空值处理，根据 axis、mask 和 skipna 来处理
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    return result
    # 调用 _get_values 函数，获取处理后的值和掩码
    values, mask = _get_values(values, True, fill_value_typ="+inf", mask=mask)
    # 在指定轴上找到 values 数组中的最小值的索引
    result = values.argmin(axis)
    # 调用 _maybe_arg_null_out 函数处理结果，处理类型为忽略类型检查
    # 在类型检查中，指出参数 1 的类型与预期不兼容，应为 "ndarray[Any, Any]"
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    # 返回处理后的结果
    return result
# 禁止操作指定类型的数据列（"M8", "m8"）
# 可能对每行数据执行操作的装饰器
@disallow("M8", "m8")
@maybe_operate_rowwise
def nanskew(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,  # 指定操作的轴向，默认为 None
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    mask: npt.NDArray[np.bool_] | None = None,  # NaN 值的掩码数组，可选

) -> float:
    """
    Compute the sample skewness.

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G1. The algorithm computes this coefficient directly
    from the second and third central moment.

    Parameters
    ----------
    values : ndarray  # 输入的数组数据
    axis : int, optional  # 操作的轴向，可选参数
    skipna : bool, default True  # 是否跳过 NaN 值，默认为 True
    mask : ndarray[bool], optional  # 如果已知 NaN 的掩码数组，可选参数

    Returns
    -------
    result : float64  # 返回结果的数据类型，除非输入为浮点数组，否则与输入数组保持相同的精度

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 2])
    >>> nanops.nanskew(s.values)
    1.7320508075688787
    """
    mask = _maybe_get_mask(values, skipna, mask)  # 根据 skipna 和 mask 获取 NaN 掩码

    if values.dtype.kind != "f":
        values = values.astype("f8")  # 如果数据类型不是浮点数，转换为 float64 类型
        count = _get_counts(values.shape, mask, axis)  # 获取有效数据的计数
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)  # 否则，根据指定数据类型获取有效数据的计数

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)  # 如果需要跳过 NaN 值并且存在掩码，则将 NaN 值替换为 0
    elif not skipna and mask is not None and mask.any():
        return np.nan  # 如果不跳过 NaN 值且存在 NaN 值，则返回 NaN

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count  # 计算均值
    if axis is not None:
        mean = np.expand_dims(mean, axis)  # 在指定轴向扩展均值的维度

    adjusted = values - mean  # 数据减去均值得到调整后的数据
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)  # 如果需要跳过 NaN 值并且存在掩码，则将调整后的数据中的 NaN 值替换为 0
    adjusted2 = adjusted**2  # 调整后的数据的平方
    adjusted3 = adjusted2 * adjusted  # 调整后的数据的平方乘以调整后的数据
    m2 = adjusted2.sum(axis, dtype=np.float64)  # 计算调整后数据的平方的和
    m3 = adjusted3.sum(axis, dtype=np.float64)  # 计算调整后数据的平方乘以调整后的数据的和

    # 处理浮点数误差
    #
    # 在 _libs/windows.pyx 中的 #18044 中 calc_skew 遵循此行为
    # 修复浮点数误差，将 m2 < 1e-14 视为零
    m2 = _zero_out_fperr(m2)
    m3 = _zero_out_fperr(m3)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2**1.5)  # 计算样本偏度

    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)  # 如果输入是浮点数数组，将结果转换为相同的数据类型

    if isinstance(result, np.ndarray):
        result = np.where(m2 == 0, 0, result)  # 如果 m2 等于 0，则将结果设置为 0
        result[count < 3] = np.nan  # 如果有效数据计数小于 3，则结果设置为 NaN
    else:
        result = dtype.type(0) if m2 == 0 else result  # 如果 m2 等于 0，则结果设置为 0，否则保持结果不变
        if count < 3:
            return np.nan  # 如果有效数据计数小于 3，则返回 NaN

    return result  # 返回计算得到的样本偏度的结果


# 禁止操作指定类型的数据列（"M8", "m8"）
# 可能对每行数据执行操作的装饰器
@disallow("M8", "m8")
@maybe_operate_rowwise
def nankurt(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,  # 指定操作的轴向，默认为 None
    skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    mask: npt.NDArray[np.bool_] | None = None,  # NaN 值的掩码数组，可选
) -> float:
    """
    Compute the sample excess kurtosis

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G2, computed directly from the second and fourth
    central moment.

    Parameters
    ----------
    values : ndarray  # 输入的数组数据
    axis : int, optional  # 操作的轴向，可选参数
    skipna : bool, default True  # 是否跳过 NaN 值，默认为 True
    mask : ndarray[bool], optional  # 如果已知 NaN 的掩码数组，可选参数
    """
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 3, 2])
    >>> nanops.nankurt(s.values)
    -1.2892561983471076
    """
    # 获取可能存在的数据掩码
    mask = _maybe_get_mask(values, skipna, mask)
    
    # 如果输入数组的数据类型不是浮点数，则转换为浮点数类型
    if values.dtype.kind != "f":
        values = values.astype("f8")
    
    # 获取计数值
    count = _get_counts(values.shape, mask, axis)
    
    # 如果数据类型是浮点数，则使用相同的数据类型获取计数值
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)
    
    # 如果需要跳过 NaN 值并且存在数据掩码，则将相应位置的值替换为 0
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    
    # 如果不跳过 NaN 值并且存在数据掩码并且有任何 NaN 值存在，则返回 NaN
    elif not skipna and mask is not None and mask.any():
        return np.nan
    
    # 计算均值，忽略无效值和除法错误
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    
    # 如果指定了轴，则扩展均值的维度
    if axis is not None:
        mean = np.expand_dims(mean, axis)
    
    # 计算调整后的值（减去均值），并根据需要处理 NaN 值
    adjusted = values - mean
    
    # 如果需要跳过 NaN 值并且存在数据掩码，则将相应位置的调整后的值替换为 0
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    
    # 计算调整后的值的平方、四次方，并求和，忽略无效值和除法错误
    adjusted2 = adjusted**2
    adjusted4 = adjusted2**2
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)
    
    # 计算峰度公式中的调整值、分子和分母
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2**2
    
    # 处理浮点数误差
    #
    # #18044 in _libs/windows.pyx calc_kurt follow this behavior
    # to fix the fperr to treat denom <1e-14 as zero
    numerator = _zero_out_fperr(numerator)
    denominator = _zero_out_fperr(denominator)
    
    # 如果分母是标量且满足特定条件，则返回 NaN 或特定数据类型的零值
    if not isinstance(denominator, np.ndarray):
        if count < 4:
            return np.nan
        if denominator == 0:
            return values.dtype.type(0)
    
    # 计算最终的峰度值，忽略无效值和除法错误
    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator - adj
    
    # 根据输入数组的数据类型，将结果转换为相应的数据类型
    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    
    # 如果结果是数组，则在特定条件下将结果中的某些元素设置为零或 NaN
    if isinstance(result, np.ndarray):
        result = np.where(denominator == 0, 0, result)
        result[count < 4] = np.nan
    
    # 返回计算得到的峰度值
    return result
# 应用装饰器，防止参数中包含 "M8" 或 "m8"
@disallow("M8", "m8")
# 可能对每行进行操作的装饰器
@maybe_operate_rowwise
def nanprod(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,  # 定义可选的轴参数
    skipna: bool = True,  # 跳过 NaN 值的标志，默认为 True
    min_count: int = 0,  # 最小计数，默认为 0
    mask: npt.NDArray[np.bool_] | None = None,  # 可选的 NaN 掩码

) -> float:
    """
    Parameters
    ----------
    values : ndarray[dtype]
        输入的 ndarray 数据
    axis : int, optional
        指定的轴
    skipna : bool, default True
        是否跳过 NaN 值
    min_count: int, default 0
        最小计数
    mask : ndarray[bool], optional
        如果已知的话，NaN 掩码

    Returns
    -------
    Dtype
        给定轴上所有元素的乘积（NaN 被视为 1）

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, 3, np.nan])
    >>> nanops.nanprod(s.values)
    6.0
    """
    # 获取可能的掩码
    mask = _maybe_get_mask(values, skipna, mask)

    # 如果跳过 NaN 并且存在掩码，则将对应位置的值设为 1
    if skipna and mask is not None:
        values = values.copy()
        values[mask] = 1

    # 计算给定轴上的乘积
    result = values.prod(axis)

    # 返回可能的空值
    return _maybe_null_out(  # type: ignore[return-value]
        result, axis, mask, values.shape, min_count=min_count
    )


def _maybe_arg_null_out(
    result: np.ndarray,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_] | None,
    skipna: bool,
) -> np.ndarray | int:
    # nanargmin/nanargmax 的辅助函数

    # 如果没有掩码，则直接返回结果
    if mask is None:
        return result

    # 如果轴为 None 或者结果没有维度，则检查是否所有值为 NA
    if axis is None or not getattr(result, "ndim", False):
        if skipna and mask.all():
            raise ValueError("Encountered all NA values")
        elif not skipna and mask.any():
            raise ValueError("Encountered an NA value with skipna=False")
    else:
        # 否则，检查指定轴上是否有所有 NA 值
        if skipna and mask.all(axis).any():
            raise ValueError("Encountered all NA values")
        elif not skipna and mask.any(axis).any():
            raise ValueError("Encountered an NA value with skipna=False")
    return result


def _get_counts(
    values_shape: Shape,
    mask: npt.NDArray[np.bool_] | None,
    axis: AxisInt | None,
    dtype: np.dtype[np.floating] = np.dtype(np.float64),
) -> np.floating | npt.NDArray[np.floating]:
    """
    Get the count of non-null values along an axis

    Parameters
    ----------
    values_shape : tuple of int
        values ndarray 的形状元组，如果没有 mask 的话使用
    mask : Optional[ndarray[bool]]
        要考虑为缺失的 values 中的位置
    axis : Optional[int]
        要进行计数的轴
    dtype : type, optional
        用于计数的类型

    Returns
    -------
    count : scalar or array
    """
    # 如果轴为 None
    if axis is None:
        # 如果存在掩码，则计算非空值的数量
        if mask is not None:
            n = mask.size - mask.sum()
        else:
            # 否则直接计算 values 形状的乘积
            n = np.prod(values_shape)
        return dtype.type(n)

    # 如果存在掩码，则计算指定轴上的非空值数量
    if mask is not None:
        count = mask.shape[axis] - mask.sum(axis)
    else:
        count = values_shape[axis]

    # 如果计数是整数，则返回指定类型的计数
    if is_integer(count):
        return dtype.type(count)
    return count.astype(dtype, copy=False)


def _maybe_null_out(
    result: np.ndarray | float | NaTType,
    axis: AxisInt | None,  # 声明一个变量 axis，可以是 AxisInt 类型或者 None
    mask: npt.NDArray[np.bool_] | None,  # 声明一个变量 mask，可以是 numpy 中的布尔类型数组或者 None
    shape: tuple[int, ...],  # 声明一个变量 shape，是一个由整数组成的元组，元组中的元素个数不定
    min_count: int = 1,  # 声明一个变量 min_count，默认值为 1，是一个整数类型
# 返回类型可以是 np.ndarray、float 或 NaTType 类型
def ) -> np.ndarray | float | NaTType:
    """
    返回
    -------
    Dtype
        给定轴上所有元素的乘积。（将 NaN 视为 1）
    """
    if mask is None and min_count == 0:
        # 如果 mask 为 None 且 min_count 为 0，则直接返回结果
        return result

    if axis is not None and isinstance(result, np.ndarray):
        if mask is not None:
            # 计算 null_mask，用于标记缺失值
            null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        else:
            # 如果没有空值，_maybe_get_mask 中保持 mask=None
            below_count = shape[axis] - min_count < 0
            new_shape = shape[:axis] + shape[axis + 1 :]
            null_mask = np.broadcast_to(below_count, new_shape)

        if np.any(null_mask):
            if is_numeric_dtype(result):
                if np.iscomplexobj(result):
                    # 如果结果是复数类型，则转换为复数类型
                    result = result.astype("c16")
                elif not is_float_dtype(result):
                    # 如果结果不是浮点数类型，则转换为浮点数类型
                    result = result.astype("f8", copy=False)
                # 将 null_mask 对应位置设置为 NaN
                result[null_mask] = np.nan
            else:
                # GH12941，使用 None 自动转换为 null
                result[null_mask] = None
    elif result is not NaT:
        if check_below_min_count(shape, mask, min_count):
            result_dtype = getattr(result, "dtype", None)
            if is_float_dtype(result_dtype):
                # 错误：Item "None" of "Optional[Any]" has no attribute "type"
                # 将结果设置为 NaN
                result = result_dtype.type("nan")  # type: ignore[union-attr]
            else:
                # 将结果设置为 NaN
                result = np.nan

    return result


def check_below_min_count(
    shape: tuple[int, ...], mask: npt.NDArray[np.bool_] | None, min_count: int
) -> bool:
    """
    检查 `min_count` 关键字。如果低于 `min_count`，则返回 True（当从缩减中应返回缺失值时）。

    Parameters
    ----------
    shape : tuple
        值的形状 (`values.shape`)。
    mask : ndarray[bool] or None
        布尔类型的 numpy 数组（通常与 `shape` 具有相同的形状）或 None。
    min_count : int
        从 sum/prod 调用传递的关键字。

    Returns
    -------
    bool
    """
    if min_count > 0:
        if mask is None:
            # 没有缺失值，仅检查大小
            non_nulls = np.prod(shape)
        else:
            non_nulls = mask.size - mask.sum()
        if non_nulls < min_count:
            return True
    return False


def _zero_out_fperr(arg):
    # #18044 引用此行为以修复滚动偏度/峰度问题
    if isinstance(arg, np.ndarray):
        # 如果参数是 np.ndarray 类型，则根据条件将小于 1e-14 的值设为 0
        return np.where(np.abs(arg) < 1e-14, 0, arg)
    else:
        # 如果参数不是 np.ndarray 类型，则根据条件将小于 1e-14 的值设为其类型的 0
        return arg.dtype.type(0) if np.abs(arg) < 1e-14 else arg


@disallow("M8", "m8")
def nancorr(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: int | None = None,
) -> float:
    """
    计算两个数组 a 和 b 之间的相关性。

    Parameters
    ----------
    a, b : ndarray
        输入数组。

    method : CorrelationMethod, optional
        相关性计算方法，默认为 "pearson"。

    min_periods : int or None, optional
        最小周期数，默认为 None。

    Returns
    -------
    float
        相关系数。
    """
    if len(a) != len(b):
        # 如果数组 a 和 b 的长度不相等，则抛出 AssertionError
        raise AssertionError("Operands to nancorr must have same size")
    # 如果未指定最小期数，则默认为1
    if min_periods is None:
        min_periods = 1

    # 检查数组 a 和 b 中的有效数值，即非 NaN 值
    valid = notna(a) & notna(b)
    
    # 如果存在 NaN 值，则从数组 a 和 b 中仅选择有效数值部分
    if not valid.all():
        a = a[valid]
        b = b[valid]

    # 如果数组 a 的长度小于最小期数，则返回 NaN
    if len(a) < min_periods:
        return np.nan

    # 确保数组 a 和 b 的元素都是数值类型
    a = _ensure_numeric(a)
    b = _ensure_numeric(b)

    # 根据指定的方法获取相关系数计算函数
    f = get_corr_func(method)
    
    # 返回数组 a 和 b 的相关系数计算结果
    return f(a, b)
# 定义一个函数，根据给定的相关性计算方法返回对应的计算函数
def get_corr_func(
    method: CorrelationMethod,
) -> Callable[[np.ndarray, np.ndarray], float]:
    if method == "kendall":
        # 导入 scipy 库中的 kendalltau 函数，定义一个计算 Kendall Tau 相关系数的函数
        from scipy.stats import kendalltau

        def func(a, b):
            return kendalltau(a, b)[0]

        return func
    elif method == "spearman":
        # 导入 scipy 库中的 spearmanr 函数，定义一个计算 Spearman 相关系数的函数
        from scipy.stats import spearmanr

        def func(a, b):
            return spearmanr(a, b)[0]

        return func
    elif method == "pearson":
        # 定义一个计算 Pearson 相关系数的函数
        def func(a, b):
            return np.corrcoef(a, b)[0, 1]

        return func
    elif callable(method):
        # 如果传入的方法是一个可调用的函数，则直接返回
        return method

    # 如果传入的方法不是预期的任何一种，抛出 ValueError 异常
    raise ValueError(
        f"Unknown method '{method}', expected one of "
        "'kendall', 'spearman', 'pearson', or callable"
    )


# 使用装饰器定义一个函数，用于计算两个数组的协方差，忽略 NaN 值
@disallow("M8", "m8")
def nancov(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_periods: int | None = None,
    ddof: int | None = 1,
) -> float:
    # 检查输入数组的长度是否相同，如果不同则抛出 AssertionError
    if len(a) != len(b):
        raise AssertionError("Operands to nancov must have same size")

    # 如果未指定最小周期，则默认为1
    if min_periods is None:
        min_periods = 1

    # 检查两个数组中的非 NaN 值，如果有 NaN 则将其过滤掉
    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    # 如果有效数据长度小于最小周期数，则返回 NaN
    if len(a) < min_periods:
        return np.nan

    # 确保数组元素为数值类型
    a = _ensure_numeric(a)
    b = _ensure_numeric(b)

    # 计算两个数组的协方差，返回结果
    return np.cov(a, b, ddof=ddof)[0, 1]


# 内部函数，用于确保输入是数值类型
def _ensure_numeric(x):
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "biu":
            x = x.astype(np.float64)
        elif x.dtype == object:
            inferred = lib.infer_dtype(x)
            if inferred in ["string", "mixed"]:
                # 避免将字符串等类型转换为数值类型，抛出 TypeError
                raise TypeError(f"Could not convert {x} to numeric")
            try:
                x = x.astype(np.complex128)
            except (TypeError, ValueError):
                try:
                    x = x.astype(np.float64)
                except ValueError as err:
                    # 当数组包含字符串时，可能出现此错误
                    raise TypeError(f"Could not convert {x} to numeric") from err
            else:
                if not np.any(np.imag(x)):
                    x = x.real
    elif not (is_float(x) or is_integer(x) or is_complex(x)):
        if isinstance(x, str):
            # 避免将字符串转换为数值类型，抛出 TypeError
            raise TypeError(f"Could not convert string '{x}' to numeric")
        try:
            x = float(x)
        except (TypeError, ValueError):
            # 尝试转换为复数类型或者抛出 TypeError
            try:
                x = complex(x)
            except ValueError as err:
                raise TypeError(f"Could not convert {x} to numeric") from err
    return x


# 定义一个函数，使用指定的累积函数对数组进行累积计算，支持跳过 NaN 值
def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    """
    Cumulative function with skipna support.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        待累积计算的数组
    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}
        指定的累积函数
    skipna : bool
    是否跳过 NaN 值的标志，类型为布尔值

    Returns
    -------
    np.ndarray or ExtensionArray
    返回一个 NumPy 数组或扩展数组

    """
    mask_a, mask_b = {
        np.cumprod: (1.0, np.nan),                # 累积乘积函数的初始掩码值
        np.maximum.accumulate: (-np.inf, np.nan), # 最大累积函数的初始掩码值
        np.cumsum: (0.0, np.nan),                 # 累积和函数的初始掩码值
        np.minimum.accumulate: (np.inf, np.nan),  # 最小累积函数的初始掩码值
    }[accum_func]

    # This should go through ea interface
    # 这应该通过 ea 接口进行处理

    assert values.dtype.kind not in "mM"
    # 确保值的数据类型不是日期时间类型或时间增量类型

    # We will be applying this function to block values
    # 我们将把这个函数应用到块值上

    if skipna and not issubclass(values.dtype.type, (np.integer, np.bool_)):
        # 如果 skipna 为 True，并且值的数据类型不是整数或布尔类型的子类
        vals = values.copy()
        mask = isna(vals)
        # 创建一个 NaN 掩码
        vals[mask] = mask_a
        # 应用累积函数到处理后的值
        result = accum_func(vals, axis=0)
        # 将处理后的结果中的 NaN 值替换为初始掩码值
        result[mask] = mask_b
    else:
        # 否则，直接应用累积函数到原始值
        result = accum_func(values, axis=0)

    return result
```