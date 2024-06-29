# `.\numpy\numpy\_pyinstaller\__init__.py`

```
# 导入所需的模块：datetime 和 timedelta
from datetime import datetime, timedelta

# 定义一个函数，计算指定日期加上指定天数后的日期，并返回结果
def add_days(date_str, days):
    # 将日期字符串转换为 datetime 对象
    date = datetime.strptime(date_str, '%Y-%m-%d')
    # 创建一个 timedelta 对象，表示要增加的天数
    delta = timedelta(days=days)
    # 使用 timedelta 对象增加日期
    new_date = date + delta
    # 将计算后的日期对象格式化为字符串，返回结果
    return new_date.strftime('%Y-%m-%d')
```