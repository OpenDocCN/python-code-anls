# `.\DB-GPT-src\dbgpt\util\date_utils.py`

```py
# 导入 datetime 模块，用于处理日期和时间相关的操作
import datetime


# 检查给定值是否为 datetime.datetime 对象
def is_datetime(value):
    return isinstance(value, datetime.datetime)


# 将行中的每个元素转换为指定格式的字符串表示，如果是 datetime 对象则进行格式化
def convert_datetime_in_row(row):
    return [
        value.strftime("%Y-%m-%d %H:%M:%S") if is_datetime(value) else value
        for value in row
    ]
```