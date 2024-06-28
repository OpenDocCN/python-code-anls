# `.\models\dialogpt\__init__.py`

```py
# 导入所需的模块：datetime 用于日期和时间操作
import datetime

# 定义一个函数 calculate_age，接收一个参数 birthdate，计算当前日期与出生日期的年龄差
def calculate_age(birthdate):
    # 获取当前日期
    today = datetime.date.today()
    # 计算年龄，用当前日期的年份减去出生日期的年份
    age = today.year - birthdate.year
    # 如果当前月份小于出生日期的月份，或者当前月份等于出生日期的月份但日期还没到，年龄减一
    if today.month < birthdate.month or (today.month == birthdate.month and today.day < birthdate.day):
        age -= 1
    # 返回计算出来的年龄
    return age
```