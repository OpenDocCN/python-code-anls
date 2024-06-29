# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimelike_\__init__.py`

```
# 导入模块 datetime
import datetime

# 定义函数 calculate_age，接受参数 birthdate
def calculate_age(birthdate):
    # 获取当前日期
    today = datetime.date.today()
    # 计算年龄，当前日期减去出生日期的年份差
    age = today.year - birthdate.year
    # 如果出生日期今年尚未过生日，年龄需减一
    if today < datetime.date(today.year, birthdate.month, birthdate.day):
        age -= 1
    # 返回计算出的年龄
    return age
```