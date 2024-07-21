# `.\pytorch\test\delete.py`

```py
# 导入Python内置的datetime模块，用于处理日期和时间
import datetime

# 定义一个名为format_date的函数，接收一个日期参数d
def format_date(d):
    # 返回将日期格式化为'YYYY-MM-DD'形式的字符串
    return d.strftime('%Y-%m-%d')

# 创建一个名为today的变量，存储当前日期
today = datetime.date.today()

# 调用format_date函数，将当前日期格式化并打印输出
print(format_date(today))
```