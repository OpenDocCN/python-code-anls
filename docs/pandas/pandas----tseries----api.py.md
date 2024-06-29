# `D:\src\scipysrc\pandas\pandas\tseries\api.py`

```
"""
Timeseries API
"""

# 导入 pandas 库中的日期时间格式猜测函数
from pandas._libs.tslibs.parsing import guess_datetime_format

# 导入 pandas 库中的时间序列偏移量相关模块
from pandas.tseries import offsets

# 导入 pandas 库中推断频率的函数
from pandas.tseries.frequencies import infer_freq

# 模块级别的公开接口列表，包括了可以直接访问的函数和类名
__all__ = ["infer_freq", "offsets", "guess_datetime_format"]
```