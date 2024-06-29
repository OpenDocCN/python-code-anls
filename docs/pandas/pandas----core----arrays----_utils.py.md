# `D:\src\scipysrc\pandas\pandas\core\arrays\_utils.py`

```
# 导入必要的模块和类
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np  # 导入NumPy库，命名为np

from pandas._libs import lib  # 导入Pandas内部库
from pandas.errors import LossySetitemError  # 导入异常类

from pandas.core.dtypes.cast import np_can_hold_element  # 导入类型转换函数
from pandas.core.dtypes.common import is_numeric_dtype  # 导入判断数据类型函数

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,  # 导入数组样式类型
        npt,  # 导入NumPy类型
    )


def to_numpy_dtype_inference(
    arr: ArrayLike, dtype: npt.DTypeLike | None, na_value, hasna: bool
) -> tuple[npt.DTypeLike, Any]:
    # 检查是否需要推断dtype，并且输入数据arr的dtype是数值类型
    if dtype is None and is_numeric_dtype(arr.dtype):
        dtype_given = False
        # 如果存在缺失值
        if hasna:
            # 如果数据类型是布尔型
            if arr.dtype.kind == "b":
                dtype = np.dtype(np.object_)  # 设置dtype为NumPy对象类型
            else:
                # 如果数据类型是整型或无符号整型
                if arr.dtype.kind in "iu":
                    dtype = np.dtype(np.float64)  # 设置dtype为NumPy浮点数类型
                else:
                    dtype = arr.dtype.numpy_dtype  # type: ignore[union-attr]  # 获取arr的NumPy数据类型
                if na_value is lib.no_default:
                    na_value = np.nan  # 设置缺失值为NaN
        else:
            dtype = arr.dtype.numpy_dtype  # type: ignore[union-attr]  # 获取arr的NumPy数据类型
    elif dtype is not None:
        dtype = np.dtype(dtype)  # 如果指定了dtype，则转换为NumPy数据类型
        dtype_given = True
    else:
        dtype_given = True

    # 如果未指定缺失值
    if na_value is lib.no_default:
        if dtype is None or not hasna:
            na_value = arr.dtype.na_value  # 获取arr的缺失值
        elif dtype.kind == "f":  # type: ignore[union-attr]
            na_value = np.nan  # 如果dtype是浮点型，设置缺失值为NaN
        elif dtype.kind == "M":  # type: ignore[union-attr]
            na_value = np.datetime64("nat")  # 如果dtype是日期时间类型，设置缺失值为NaT
        elif dtype.kind == "m":  # type: ignore[union-attr]
            na_value = np.timedelta64("nat")  # 如果dtype是时间间隔类型，设置缺失值为NaT
        else:
            na_value = arr.dtype.na_value  # 获取arr的缺失值

    # 如果未指定dtype且存在缺失值
    if not dtype_given and hasna:
        try:
            np_can_hold_element(dtype, na_value)  # type: ignore[arg-type]  # 检查dtype是否可以容纳缺失值
        except LossySetitemError:
            dtype = np.dtype(np.object_)  # 如果无法容纳，则设置dtype为NumPy对象类型
    return dtype, na_value  # 返回推断的dtype和缺失值
```