# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\timezones.pyi`

```
# 从 datetime 模块中导入 datetime 和 tzinfo 类
# 从 typing 模块中导入 Callable 类型
from datetime import (
    datetime,
    tzinfo,
)

# 从 numpy 模块中导入 np 别名
import numpy as np

# 从 dateutil.tz 模块导入 dateutil_gettz 函数，它是一个接受字符串参数并返回 tzinfo 对象的可调用函数
# 类型提示：Callable[[str], tzinfo]
dateutil_gettz: Callable[[str], tzinfo]

# tz_standardize 函数定义，接受一个 tzinfo 参数并返回 tzinfo 对象
def tz_standardize(tz: tzinfo) -> tzinfo: ...

# tz_compare 函数定义，接受两个 tzinfo 或 None 参数，返回布尔值
def tz_compare(start: tzinfo | None, end: tzinfo | None) -> bool: ...

# infer_tzinfo 函数定义，接受两个 datetime 或 None 参数，返回 tzinfo 或 None
def infer_tzinfo(
    start: datetime | None,
    end: datetime | None,
) -> tzinfo | None: ...

# maybe_get_tz 函数定义，接受字符串、整数、np.int64 或 tzinfo 或 None 参数，返回 tzinfo 或 None
def maybe_get_tz(tz: str | int | np.int64 | tzinfo | None) -> tzinfo | None: ...

# get_timezone 函数定义，接受 tzinfo 参数，返回 tzinfo 或字符串
def get_timezone(tz: tzinfo) -> tzinfo | str: ...

# is_utc 函数定义，接受 tzinfo 或 None 参数，返回布尔值
def is_utc(tz: tzinfo | None) -> bool: ...

# is_fixed_offset 函数定义，接受 tzinfo 参数，返回布尔值
def is_fixed_offset(tz: tzinfo) -> bool: ...
```