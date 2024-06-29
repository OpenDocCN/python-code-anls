# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\tzconversion.pyi`

```
# 导入 datetime 模块中的 timedelta 和 tzinfo 类
from datetime import (
    timedelta,
    tzinfo,
)
# 导入 typing 模块中的 Iterable 类型
from typing import Iterable

# 导入 numpy 库，并将其命名为 np
import numpy as np

# 从 pandas._typing 模块中导入 npt 类型
from pandas._typing import npt

# tz_convert_from_utc_single 用于测试目的而暴露出来的函数
def tz_convert_from_utc_single(
    utc_val: np.int64, tz: tzinfo, creso: int = ...
) -> np.int64: ...
    # 这里应该有函数的具体实现，未提供

# 将时区本地化为 UTC 时间的函数
def tz_localize_to_utc(
    vals: npt.NDArray[np.int64],
    tz: tzinfo | None,
    ambiguous: str | bool | Iterable[bool] | None = ...,
    nonexistent: str | timedelta | np.timedelta64 | None = ...,
    creso: int = ...,  # NPY_DATETIMEUNIT
) -> npt.NDArray[np.int64]: ...
    # 这里应该有函数的具体实现，未提供
```