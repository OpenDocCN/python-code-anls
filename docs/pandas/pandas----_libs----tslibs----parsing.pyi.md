# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\parsing.pyi`

```
# 导入 datetime 模块中的 datetime 类，用于处理日期和时间
from datetime import datetime

# 导入 numpy 库，并使用 np 别名引用
import numpy as np

# 导入 pandas 库中的 npt 类型定义，用于类型提示
from pandas._typing import npt

# 定义一个自定义异常类 DateParseError，继承自 ValueError
class DateParseError(ValueError): ...

# 定义一个函数 py_parse_datetime_string，用于解析日期时间字符串并返回 datetime 对象
def py_parse_datetime_string(
    date_string: str,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
) -> datetime: ...

# 定义一个函数 parse_datetime_string_with_reso，解析带有频率信息的日期时间字符串并返回元组
def parse_datetime_string_with_reso(
    date_string: str,
    freq: str | None = ...,
    dayfirst: bool | None = ...,
    yearfirst: bool | None = ...,
) -> tuple[datetime, str]: ...

# 定义一个函数 _does_string_look_like_datetime，判断给定字符串是否像是日期时间格式
def _does_string_look_like_datetime(py_string: str) -> bool: ...

# 定义一个函数 quarter_to_myear，将季度表示转换为年和月份的元组
def quarter_to_myear(year: int, quarter: int, freq: str) -> tuple[int, int]: ...

# 定义一个函数 try_parse_dates，尝试解析日期时间数组并返回相同长度的对象数组
def try_parse_dates(
    values: npt.NDArray[np.object_],  # object[:]
    parser,
) -> npt.NDArray[np.object_]: ...

# 定义一个函数 guess_datetime_format，猜测日期时间字符串的格式并返回推测的格式字符串或 None
def guess_datetime_format(
    dt_str: str,
    dayfirst: bool | None = ...,
) -> str | None: ...

# 定义一个函数 get_rule_month，根据输入的源字符串获取相关的月份规则信息并返回字符串
def get_rule_month(source: str) -> str: ...
```