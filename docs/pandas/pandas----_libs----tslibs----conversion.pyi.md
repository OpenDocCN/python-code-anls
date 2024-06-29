# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\conversion.pyi`

```
# 导入 datetime 模块中的 datetime 和 tzinfo 类
from datetime import (
    datetime,
    tzinfo,
)

# 导入 numpy 库，并将其命名为 np
import numpy as np

# 定义全局变量 DT64NS_DTYPE，表示 numpy 的 datetime64 类型
DT64NS_DTYPE: np.dtype

# 定义全局变量 TD64NS_DTYPE，表示 numpy 的 timedelta64 类型
TD64NS_DTYPE: np.dtype

# 定义函数 localize_pydatetime，用于将普通 datetime 对象本地化为特定时区的 datetime 对象
def localize_pydatetime(dt: datetime, tz: tzinfo | None) -> datetime: ...

# 定义函数 cast_from_unit_vectorized，用于将输入数组中的值从一个单位转换为另一个单位
# 参数 values: 输入的 numpy 数组
# 参数 unit: 输入的单位字符串
# 参数 out_unit: 可选的输出单位字符串，默认为 ...
# 返回值: 转换后的 numpy 数组
def cast_from_unit_vectorized(
    values: np.ndarray, unit: str, out_unit: str = ...
) -> np.ndarray: ...
```