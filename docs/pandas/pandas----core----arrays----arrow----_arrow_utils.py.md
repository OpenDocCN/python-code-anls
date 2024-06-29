# `D:\src\scipysrc\pandas\pandas\core\arrays\arrow\_arrow_utils.py`

```
# 从未来导入注解支持，使得函数签名中的类型提示能够正确工作
from __future__ import annotations

# 导入警告模块，用于在特定情况下发出警告
import warnings

# 导入 NumPy 库，用于处理数据数组
import numpy as np

# 导入 pyarrow 库，用于与 Arrow 格式数据交互
import pyarrow

# 从 pandas 库中导入获取配置选项的函数
from pandas._config.config import get_option

# 从 pandas 库中导入性能警告类
from pandas.errors import PerformanceWarning

# 从 pandas 库中导入查找调用栈级别的函数
from pandas.util._exceptions import find_stack_level


def fallback_performancewarning(version: str | None = None) -> None:
    """
    发出性能警告，表示将回退到 ExtensionArray 的非 pyarrow 方法。
    """
    # 如果开启了性能警告选项
    if get_option("performance_warnings"):
        # 构造警告消息
        msg = "Falling back on a non-pyarrow code path which may decrease performance."
        # 如果提供了版本号，添加升级到指定版本的提示信息
        if version is not None:
            msg += f" Upgrade to pyarrow >={version} to possibly suppress this warning."
        # 发出警告，使用 PerformanceWarning 类，指定警告的堆栈级别
        warnings.warn(msg, PerformanceWarning, stacklevel=find_stack_level())


def pyarrow_array_to_numpy_and_mask(
    arr, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """
    将基本的 pyarrow.Array 转换为 NumPy 数组，并根据 Array 的缓冲区创建布尔掩码。

    目前不支持 pyarrow.BooleanArray。

    Parameters
    ----------
    arr : pyarrow.Array
        要转换的 pyarrow 数组
    dtype : numpy.dtype
        所期望的 NumPy 数组的数据类型

    Returns
    -------
    (data, mask)
        由两个 NumPy 数组组成的元组，第一个是原始数据数组（使用指定的 dtype），
        第二个是布尔掩码数组（有效性掩码，False 表示缺失）
    """
    dtype = np.dtype(dtype)

    # 如果数组类型是空的（null）
    if pyarrow.types.is_null(arr.type):
        # 由于一切都是空的，不需要初始化数据
        data = np.empty(len(arr), dtype=dtype)
        mask = np.zeros(len(arr), dtype=bool)
        return data, mask
    
    # 获取数组的缓冲区列表
    buflist = arr.buffers()
    
    # 计算偏移量和长度，以从缓冲区中提取数据
    offset = arr.offset * dtype.itemsize
    length = len(arr) * dtype.itemsize
    
    # 从数据缓冲区中创建 NumPy 数组
    data_buf = buflist[1][offset : offset + length]
    data = np.frombuffer(data_buf, dtype=dtype)
    
    # 处理位掩码（bitmask）
    bitmask = buflist[0]
    if bitmask is not None:
        # 如果存在位掩码，则从缓冲区创建布尔数组
        mask = pyarrow.BooleanArray.from_buffers(
            pyarrow.bool_(), len(arr), [None, bitmask], offset=arr.offset
        )
        mask = np.asarray(mask)
    else:
        # 否则，创建全为 True 的掩码数组
        mask = np.ones(len(arr), dtype=bool)
    
    return data, mask
```