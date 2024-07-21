# `.\pytorch\torch\backends\_coreml\__init__.py`

```
# 导入必要的模块：datetime 模块用于处理日期和时间
import datetime

# 定义一个名为 is_leap_year 的函数，用于判断给定的年份是否是闰年
def is_leap_year(year):
    # 如果年份能被4整除但不能被100整除，或者能被400整除，则是闰年
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

# 获取当前日期和时间
now = datetime.datetime.now()

# 提取当前年份
current_year = now.year

# 打印当前年份
print(f"Current year: {current_year}")
```