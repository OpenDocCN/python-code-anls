# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\strptime.pyi`

```
# 导入 NumPy 库，简称为 np
import numpy as np

# 导入 pandas 库中的 _typing 模块中的 npt 类型
from pandas._typing import npt

# 定义 array_strptime 函数，接受以下参数和返回类型
def array_strptime(
    values: npt.NDArray[np.object_],  # 接受一个 NumPy 对象数组，其中元素为 np.object_ 类型
    fmt: str | None,                  # 接受一个字符串或者 None 类型作为日期时间格式
    exact: bool = ...,                # 是否进行精确匹配的布尔类型参数，未指定具体值
    errors: str = ...,                # 处理错误的字符串类型参数，未指定具体值
    utc: bool = ...,                  # 是否使用 UTC 时间的布尔类型参数，未指定具体值
    creso: int = ...,                 # 代表日期时间单位的整数参数，默认为 NPY_DATETIMEUNIT
) -> tuple[np.ndarray, np.ndarray]:  # 返回一个包含两个 NumPy 数组的元组，第一个是 M8[ns] 类型，第二个是 object 数组，其中元素为 tzinfo | None 类型

# first ndarray is M8[ns], second is object ndarray of tzinfo | None
# 函数说明注释，指出函数返回的两个 NumPy 数组分别表示 M8[ns] 类型和 object 数组，元素可以是 tzinfo 或 None 类型
```