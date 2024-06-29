# `D:\src\scipysrc\pandas\pandas\core\array_algos\quantile.py`

```
# 导入未来版本的类型注解支持
from __future__ import annotations

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入 NumPy 库并使用别名 np
import numpy as np

# 导入 Pandas 库中处理缺失值相关的函数
from pandas.core.dtypes.missing import (
    isna,                  # 导入 isna 函数，用于检测缺失值
    na_value_for_dtype,    # 导入 na_value_for_dtype 函数，用于确定指定数据类型的缺失值
)

# 如果类型检查开启
if TYPE_CHECKING:
    # 导入额外的类型声明
    from pandas._typing import (
        ArrayLike,    # 导入 ArrayLike 类型别名，用于表示类数组结构
        Scalar,       # 导入 Scalar 类型别名，表示标量数据类型
        npt,          # 导入 npt 模块，用于 NumPy 类型的声明
    )


def quantile_compat(
    values: ArrayLike, qs: npt.NDArray[np.float64], interpolation: str
) -> ArrayLike:
    """
    计算给定值在每个分位数 `qs` 上的分位数。

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        待计算的值数组或扩展数组
    qs : np.ndarray[float64]
        分位数的数组
    interpolation : str
        插值方法

    Returns
    -------
    np.ndarray or ExtensionArray
        分位数数组
    """
    if isinstance(values, np.ndarray):
        # 获取与值数据类型相关的非缺失值
        fill_value = na_value_for_dtype(values.dtype, compat=False)
        # 生成值数组的缺失值掩码
        mask = isna(values)
        # 调用 quantile_with_mask 函数计算带有掩码的分位数
        return quantile_with_mask(values, mask, fill_value, qs, interpolation)
    else:
        # 对扩展数组调用内部的 `_quantile` 方法
        return values._quantile(qs, interpolation)


def quantile_with_mask(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    fill_value,
    qs: npt.NDArray[np.float64],
    interpolation: str,
) -> np.ndarray:
    """
    计算给定值在每个分位数 `qs` 上的分位数。

    Parameters
    ----------
    values : np.ndarray
        对于扩展数组，这是 _values_for_factorize()[0]
    mask : np.ndarray[bool]
        mask = isna(values)
        对于扩展数组，在调用 _value_for_factorize 前计算的掩码
    fill_value : Scalar
        用于填充 NA 条目的值
        对于扩展数组，这是 _values_for_factorize()[1]
    qs : np.ndarray[float64]
        分位数数组
    interpolation : str
        插值类型

    Returns
    -------
    np.ndarray
        分位数数组

    Notes
    -----
    假设 values 已经是二维的。对于扩展数组，这意味着已经在 _values_for_factorize()[0] 上调用了 np.atleast_2d。
    
    在 axis=1 上计算分位数。
    """
    assert values.shape == mask.shape
    if values.ndim == 1:
        # 将一维数组扩展为二维，进行操作后再压缩回一维
        values = np.atleast_2d(values)
        mask = np.atleast_2d(mask)
        res_values = quantile_with_mask(values, mask, fill_value, qs, interpolation)
        return res_values[0]

    assert values.ndim == 2

    is_empty = values.shape[1] == 0

    if is_empty:
        # 创建 NA 值的数组
        flat = np.array([fill_value] * len(qs))
        result = np.repeat(flat, len(values)).reshape(len(values), len(qs))
    else:
        # 调用 _nanpercentile 函数计算百分位数
        result = _nanpercentile(
            values,
            qs * 100.0,
            na_value=fill_value,
            mask=mask,
            interpolation=interpolation,
        )

        result = np.asarray(result)
        result = result.T

    return result


def _nanpercentile_1d(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    qs: npt.NDArray[np.float64],
    na_value: Scalar,
    interpolation: str,
) -> Scalar | np.ndarray:
    """
    计算给定值在每个分位数 `qs` 上的分位数。

    Parameters
    ----------
    values : np.ndarray
        对于扩展数组，这是 _values_for_factorize()[0]
    mask : np.ndarray[bool]
        mask = isna(values)
        对于扩展数组，在调用 _value_for_factorize 前计算的掩码
    qs : np.ndarray[float64]
        分位数数组
    na_value : Scalar
        用于填充 NA 条目的值
    interpolation : str
        插值类型

    Returns
    -------
    Scalar | np.ndarray
        分位数数组
    """
    """
    Wrapper for np.percentile that skips missing values, specialized to
    1-dimensional case.

    Parameters
    ----------
    values : array over which to find quantiles
        要计算分位数的数组
    mask : ndarray[bool]
        locations in values that should be considered missing
        指示哪些位置的值应该视为缺失的布尔型数组
    qs : np.ndarray[float64] of quantile indices to find
        要计算的分位数的索引值数组
    na_value : scalar
        value to return for empty or all-null values
        空值或全部为null时返回的值
    interpolation : str
        插值方法

    Returns
    -------
    quantiles : scalar or array
        分位数值，可以是标量或数组
    """
    # mask is Union[ExtensionArray, ndarray]
    # 从 values 数组中剔除掉 mask 中为 True 的位置对应的值
    values = values[~mask]

    if len(values) == 0:
        # 如果剔除掉 mask 中所有为 True 的位置后 values 为空
        # 由于可能 na_value=np.nan 且 values.dtype=int64，这里不能使用 dtype=values.dtype
        # 参见 test_quantile_empty，相当于使用 'np.array([na_value] * len(qs))' 但速度更快
        return np.full(len(qs), na_value)

    return np.percentile(
        values,
        qs,
        method=interpolation,  # 指定使用的插值方法
        # error: No overload variant of "percentile" matches argument
        # types "ndarray[Any, Any]", "ndarray[Any, dtype[floating[_64Bit]]]"
        # , "Dict[str, str]"  [call-overload]
        # 错误：没有与参数类型 "ndarray[Any, Any]", "ndarray[Any, dtype[floating[_64Bit]]]" 匹配的 "percentile" 的重载变体
        # 忽略类型检查以解决调用问题
        # type: ignore[call-overload]
    )
def _nanpercentile(
    values: np.ndarray,
    qs: npt.NDArray[np.float64],
    *,
    na_value,  # 标量，用于空值或全部空值情况下的返回值
    mask: npt.NDArray[np.bool_],  # 表示哪些位置的值被视为缺失的布尔数组
    interpolation: str,  # 插值方法，用于计算百分位数时的插值方式
):
    """
    Wrapper for np.percentile that skips missing values.

    Parameters
    ----------
    values : np.ndarray[ndim=2]  over which to find quantiles  # 用于计算百分位数的二维数组
    qs : np.ndarray[float64] of quantile indices to find  # 要计算的百分位数索引数组
    na_value : scalar
        value to return for empty or all-null values  # 空值或全部空值情况下的返回值
    mask : np.ndarray[bool]
        locations in values that should be considered missing  # 被视为缺失的位置的布尔数组
    interpolation : str  # 插值方法，用于计算百分位数时的插值方式

    Returns
    -------
    quantiles : scalar or array  # 返回计算得到的百分位数值或数组
    """

    if values.dtype.kind in "mM":
        # need to cast to integer to avoid rounding errors in numpy
        result = _nanpercentile(
            values.view("i8"),
            qs=qs,
            na_value=na_value.view("i8"),
            mask=mask,
            interpolation=interpolation,
        )

        # Note: we have to do `astype` and not view because in general
        #  we have float result at this point, not i8
        return result.astype(values.dtype)  # 将结果转换回原始数据类型

    if mask.any():
        # Caller is responsible for ensuring mask shape match
        assert mask.shape == values.shape  # 断言确保 mask 的形状与 values 相匹配
        result = [
            _nanpercentile_1d(val, m, qs, na_value, interpolation=interpolation)
            for (val, m) in zip(list(values), list(mask))
        ]
        if values.dtype.kind == "f":
            # preserve itemsize
            result = np.asarray(result, dtype=values.dtype).T  # 保留原始数据类型的项目大小
        else:
            result = np.asarray(result).T
            if (
                result.dtype != values.dtype
                and not mask.all()
                and (result == result.astype(values.dtype, copy=False)).all()
            ):
                # mask.all() will never get cast back to int
                # e.g. values id integer dtype and result is floating dtype,
                #  only cast back to integer dtype if result values are all-integer.
                result = result.astype(values.dtype, copy=False)  # 如果结果值全部为整数，则将其转换回整数数据类型
        return result
    else:
        return np.percentile(
            values,
            qs,
            axis=1,
            method=interpolation,  # 使用指定的插值方法计算百分位数
        )
```