# `arknights-mower\arknights_mower\utils\datetime.py`

```
# 导入 datetime 模块中的 datetime 类
# 导入 pytz 模块
from datetime import datetime
import pytz

# 检查两个日期是否是同一天
def the_same_day(a: datetime = None, b: datetime = None) -> bool:
    # 如果 a 或 b 为 None，则返回 False
    if a is None or b is None:
        return False
    # 返回两个日期的年、月、日是否相同的布尔值
    return a.year == b.year and a.month == b.month and a.day == b.day

# 检查两个日期时间是否是同一时刻
def the_same_time(a: datetime = None, b: datetime = None) -> bool:
    # 如果 a 或 b 为 None，则返回 False
    if a is None or b is None:
        return False
    # 返回两个日期时间的时间差是否小于 1.5 秒的布尔值
    return abs(a - b).total_seconds() < 1.5

# 获取服务器当前的星期几
def get_server_weekday():
    # 返回当前时区为 'Asia/Dubai' 的日期时间对象的星期几
    return datetime.now(pytz.timezone('Asia/Dubai')).weekday()
```