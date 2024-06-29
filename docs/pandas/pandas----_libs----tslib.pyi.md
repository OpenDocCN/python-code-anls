# `D:\src\scipysrc\pandas\pandas\_libs\tslib.pyi`

```
# 从 datetime 模块中导入 tzinfo 类型，用于处理时区信息
from datetime import tzinfo

# 导入 numpy 库，并使用 np 作为别名
import numpy as np

# 导入 pandas 库中的类型定义 npt
from pandas._typing import npt

# 定义一个函数 format_array_from_datetime，接受以下参数：
# - values: 一个 numpy 的 NDArray，其元素类型为 np.int64，用于存储日期时间值
# - tz: 可选参数，表示时区信息，类型为 tzinfo 或 None
# - format: 可选参数，表示日期时间格式的字符串，类型为 str 或 None
# - na_rep: 可选参数，用于替换缺失值的字符串或浮点数，类型为 str 或 float
# - reso: 必选参数，默认值为 NPY_DATETIMEUNIT，表示日期时间的精度，类型为 int
# 返回一个 numpy 的 NDArray，其元素类型为 np.object_
def format_array_from_datetime(
    values: npt.NDArray[np.int64],
    tz: tzinfo | None = ...,
    format: str | None = ...,
    na_rep: str | float = ...,
    reso: int = ...,  # NPY_DATETIMEUNIT
) -> npt.NDArray[np.object_]: ...

# 定义一个函数 first_non_null，接受一个参数：
# - values: 一个 numpy 的 NDArray，用于存储数据
# 返回一个 int 值，表示第一个非空值的索引位置
def first_non_null(values: np.ndarray) -> int: ...

# 定义一个函数 array_to_datetime，接受以下参数：
# - values: 一个 numpy 的 NDArray，其元素类型为 np.object_，存储日期时间值
# - errors: 可选参数，指定错误处理方式的字符串，类型为 str
# - dayfirst: 可选参数，布尔值，表示日期是否在月份之前
# - yearfirst: 可选参数，布尔值，表示年份是否在月份之前
# - utc: 可选参数，布尔值，表示是否使用 UTC 时间
# - creso: 可选参数，默认值为 NPY_DATETIMEUNIT，表示日期时间的精度，类型为 int
# - unit_for_numerics: 可选参数，指定数值类型的单位，类型为 str 或 None
# 返回一个元组，包含两个元素：
# - 第一个元素是一个 numpy 的 NDArray，存储转换后的日期时间值
# - 第二个元素是 tzinfo 类型或 None，表示时区信息
def array_to_datetime(
    values: npt.NDArray[np.object_],
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    creso: int = ...,
    unit_for_numerics: str | None = ...,
) -> tuple[np.ndarray, tzinfo | None]: ...

# 定义一个函数 array_to_datetime_with_tz，接受以下参数：
# - values: 一个 numpy 的 NDArray，其元素类型为 np.object_，存储日期时间值
# - tz: 必选参数，时区信息，类型为 tzinfo
# - dayfirst: 布尔值，表示日期是否在月份之前
# - yearfirst: 布尔值，表示年份是否在月份之前
# - creso: 必选参数，默认值为 NPY_DATETIMEUNIT，表示日期时间的精度，类型为 int
# 返回一个 numpy 的 NDArray，其元素类型为 np.int64，存储带有时区信息的日期时间值
def array_to_datetime_with_tz(
    values: npt.NDArray[np.object_],
    tz: tzinfo,
    dayfirst: bool,
    yearfirst: bool,
    creso: int,
) -> npt.NDArray[np.int64]: ...
```